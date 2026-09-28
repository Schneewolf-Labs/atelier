import gc
import logging
import os

import numpy as np
import torch
from PIL import Image

from ..models import qwen_image_2_1 as qi21
from .base import strip_peft_prefix
from .flow import FlowMatchAdapter

logger = logging.getLogger(__name__)


def pil_to_rgba_tensor(image, height=None, width=None):
    """PIL / path / array → ``[4, H, W]`` in [-1, 1]; opaque images get alpha = +1."""
    if not isinstance(image, Image.Image):
        image = Image.open(image) if isinstance(image, str) else Image.fromarray(np.uint8(image))
    image = image.convert("RGBA")
    if height and width:
        image = image.resize((width, height), Image.LANCZOS)
    tensor = torch.from_numpy(np.array(image).astype(np.float32) / 127.5 - 1.0)
    return tensor.permute(2, 0, 1)


class QwenImage21Adapter(FlowMatchAdapter):
    """Adapter for Qwen-Image 2.1 (text-to-image and reference editing).

    diffusers has no Qwen-Image 2.1 support, so the DiT is Atelier's own port
    (:mod:`atelier.models.qwen_image_2_1`), checked to match
    stable-diffusion.cpp's implementation. Components load from the released
    single files:

    - ``transformer_path`` — the bf16 diffusion model (e.g.
      ``Comfy-Org/Qwen-Image-2.1/diffusion_models/*_bf16.safetensors``). The
      int8-convrot / fp8 releases are inference-only and are rejected.
    - ``vae_path`` — ``qwen_image_2.1_vae_bf16.safetensors`` (original or
      diffusers key names). The Qwen-Image / Wan VAEs are *not* compatible.
    - ``text_encoder_path`` — stock ``Qwen/Qwen3-VL-8B-Instruct`` (needs
      ``torchvision`` for its image processor when editing).

    Training conventions (all from sd.cpp):

    - 64-channel latents at 16x, normalized ``(z - mean) / std`` with the
      VAE's per-channel stats, one DiT token per latent pixel.
    - The VAE takes **RGBA**. Opaque images get alpha = 1; RGBA training
      images keep their alpha, so transparent-output LoRAs work.
    - Timestep fed as ``sigma * 1000``; target is the velocity ``noise - x0``.
    - Sampling uses FLUX-style resolution-dependent shift; ``shift=None``
      picks the equivalent static shift for ``resolution`` (≈3.16 at 1024²).
    - Editing: the reference image (``control_latents`` / the dataset's
      ``rejected`` column) is shown to Qwen3-VL at the *same* resolution as
      its VAE latent, so the prompt's vision slots line up 1:4 with latent
      tokens. Image sides must be multiples of 32.

    The DiT processes one sample at a time (each has its own sequence
    layout); ``forward`` loops over the batch.
    """

    def __init__(self, transformer_path, vae_path=None, text_encoder_path="Qwen/Qwen3-VL-8B-Instruct",
                 device="cuda", dtype=None, load_encoders=True, load_transformer=True,
                 defer_transformer=True, shift=None, resolution=1024, timestep_sampling="logit_normal"):
        from diffusers import FlowMatchEulerDiscreteScheduler

        self._dtype = dtype or torch.bfloat16
        self._device = device
        self._conditioner = None
        self._vae = None
        self._model = None
        self._transformer_on_device = False

        if load_encoders:
            if vae_path is None:
                raise ValueError("vae_path is required when load_encoders=True")
            self._vae = qi21.load_vae(vae_path, dtype=torch.float32, device=device).eval().requires_grad_(False)
            self._conditioner = qi21.Qwen3VLConditioner.from_pretrained(
                text_encoder_path, device=device, dtype=self._dtype,
            )

        if load_transformer:
            on_cpu = defer_transformer and load_encoders
            self._model = qi21.load_transformer(transformer_path, dtype=self._dtype, device="cpu" if on_cpu else device)
            self._transformer_on_device = not on_cpu
            if on_cpu:
                logger.info("Transformer loaded to CPU; call move_transformer_to_device() after free_encoders()")

        self._latents_mean = torch.tensor(qi21.VAE_LATENTS_MEAN).view(1, -1, 1, 1)
        self._latents_std = torch.tensor(qi21.VAE_LATENTS_STD).view(1, -1, 1, 1)

        # sd.cpp samples 2.1 with the FLUX scheduler (mu linear in token count);
        # the scheduler here only carries that config into default_shift().
        scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
            base_image_seq_len=256, max_image_seq_len=4096,
        )
        self._init_flow(scheduler, shift=shift, resolution=resolution, timestep_sampling=timestep_sampling)
        logger.info("QwenImage21Adapter: shift=%.3f, timestep_sampling=%s", self.shift, self.timestep_sampling)

    @property
    def model(self):
        return self._model

    def move_transformer_to_device(self, device=None):
        device = device or self._device
        self._model.to(device)
        self._device = device
        self._transformer_on_device = True
        logger.info("Transformer moved to %s", device)

    # -- encoding -----------------------------------------------------------
    def normalize(self, z):
        mean = self._latents_mean.to(z.device, z.dtype)
        std = self._latents_std.to(z.device, z.dtype)
        return (z - mean) / std

    def encode_image_tensor(self, image_tensor, device=None):
        """``[B, 3 or 4, H, W]`` in [-1, 1] → normalized ``[B, 64, H/16, W/16]``."""
        device = device or self._device
        x = image_tensor.to(device=device, dtype=torch.float32)
        if x.shape[1] == 3:
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        with torch.no_grad():
            z = self._vae.encode(x.unsqueeze(2)).latent_dist.mode().squeeze(2)
        return self.normalize(z)

    def encode_images(self, images, height=None, width=None, device=None, **kwargs):
        tensors = torch.stack([pil_to_rgba_tensor(im, height, width) for im in images])
        return self.encode_image_tensor(tensors, device=device)

    def encode_text(self, prompts, device=None, images=None, height=None, width=None, **kwargs):
        """Encode prompts; ``images`` (one list entry per reference) enables editing.

        Returns padded ``prompt_embeds`` [B, L, D], ``prompt_embeds_mask`` and
        ``prompt_token_types`` [B, L] (0 text, i + 1 vision slot of reference i).
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        refs = list(images or [])
        if height and width:
            refs = [im.resize((width, height), Image.LANCZOS) if isinstance(im, Image.Image) else im for im in refs]
        encoded = [self._conditioner.encode(p, refs) for p in prompts]
        L = max(h.shape[0] for h, _ in encoded)
        dim = encoded[0][0].shape[-1]
        embeds = encoded[0][0].new_zeros(len(encoded), L, dim)
        mask = torch.zeros(len(encoded), L, dtype=torch.long)
        types = torch.zeros(len(encoded), L, dtype=torch.long)
        for i, (h, t) in enumerate(encoded):
            embeds[i, : h.shape[0]] = h
            mask[i, : h.shape[0]] = 1
            types[i, : h.shape[0]] = t
        device = device or self._device
        return {"prompt_embeds": embeds, "prompt_embeds_mask": mask.to(device), "prompt_token_types": types.to(device)}

    # -- forward ------------------------------------------------------------
    def forward(self, model, noisy_latents, timesteps, batch):
        device, dtype = noisy_latents.device, noisy_latents.dtype
        embeds = batch["prompt_embeds"].to(device, dtype)
        bsz, L = embeds.shape[:2]
        mask = batch.get("prompt_embeds_mask")
        types = batch.get("prompt_token_types")
        control = batch.get("control_latents")
        timesteps = timesteps.to(device=device, dtype=torch.float32).reshape(-1).expand(bsz)

        outputs = []
        for i in range(bsz):
            n = int(mask[i].sum()) if isinstance(mask, torch.Tensor) else L
            token_types = types[i, :n].cpu() if isinstance(types, torch.Tensor) else None
            refs = [control[i:i + 1].to(device, dtype)] if isinstance(control, torch.Tensor) else []
            outputs.append(model(noisy_latents[i:i + 1], timesteps[i:i + 1], embeds[i:i + 1, :n], token_types, refs))
        return torch.cat(outputs, dim=0)

    # -- saving -------------------------------------------------------------
    def save_lora(self, model, path):
        """Write ``lora.safetensors`` in the key layout sd.cpp and ComfyUI load.

        Keys are ``diffusion_model.<module>.lora_A.weight`` / ``lora_B.weight``
        plus ``.alpha`` (= PEFT scaling x rank, so rsLoRA / alpha patterns
        survive the ``alpha / rank`` convention those loaders use).
        """
        from peft.tuners.lora import LoraLayer
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        os.makedirs(path, exist_ok=True)
        state = strip_peft_prefix(get_peft_model_state_dict(model))
        if any("lora_magnitude_vector" in k for k in state):
            logger.warning("DoRA magnitude vectors are saved but sd.cpp / ComfyUI may ignore them")
        out = {f"diffusion_model.{k}": v.detach().contiguous() for k, v in state.items()}
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) and "default" in module.r:
                key = strip_peft_prefix({name: None}).popitem()[0]
                alpha = module.scaling["default"] * module.r["default"]
                out[f"diffusion_model.{key}.alpha"] = torch.tensor(float(alpha))
        save_file(out, os.path.join(path, "lora.safetensors"))
        logger.info("LoRA weights saved to %s", path)

    def save_model(self, model, path):
        from safetensors.torch import save_file

        os.makedirs(path, exist_ok=True)
        state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
        save_file(state, os.path.join(path, "qwen_image_2.1.safetensors"))
        logger.info("Transformer saved to %s", path)

    def free_encoders(self):
        for component in (self._conditioner, self._vae):
            if component is not None:
                try:
                    component.to("cpu")
                except Exception as e:
                    logger.warning("%s.to('cpu') failed: %s", type(component).__name__, e)
        self._conditioner = None
        self._vae = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.info("Freed VAE and Qwen3-VL from memory")
