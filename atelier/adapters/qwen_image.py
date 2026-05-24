import copy
import gc
import logging

import numpy as np
import torch
from PIL import Image

from .base import ModelAdapter

logger = logging.getLogger(__name__)


class QwenImageAdapter(ModelAdapter):
    """Adapter for Qwen-Image (DiT + video VAE + flow matching), text-to-image.

    Sibling to QwenEditAdapter, with two differences:

    - Loads QwenImagePipeline (no reference-image conditioning on the
      text encoder) instead of QwenImageEditPipeline.
    - forward() takes only the target latents — no control image, no
      concat, no second img_shape.

    Everything else (video VAE, latent normalization, flow matching
    schedule, packing/unpacking, LoRA saving) is shared with the
    Edit variant because Qwen-Image and Qwen-Image-Edit share text
    encoder, VAE, and transformer architecture — only the conditioning
    path differs.
    """

    def __init__(self, pretrained_path, device="cuda", dtype=None,
                 defer_transformer=True, load_encoders=True, load_transformer=True):
        """Load the Qwen-Image components.

        Args:
            pretrained_path: HF repo id or local path.
            device: Target device for the encoders + (eventually) the transformer.
            dtype: Weight dtype; defaults to bfloat16.
            defer_transformer: If True (default) and ``load_transformer`` is
                True, the trainable transformer is loaded to CPU at init and
                only moved to ``device`` when
                :meth:`move_transformer_to_device` is called — typically after
                :meth:`free_encoders` has reclaimed the text encoder + VAE.
                The peak VRAM during init is then max(encoders, transformer)
                instead of their sum.
            load_encoders: If False, skip loading the text-encoder pipeline
                and VAE entirely. Use this in a training process when
                embeddings have been pre-computed in a SEPARATE process and
                cached to disk — the OS reclaims encoder VRAM cleanly when
                the encoding process exits, which is more reliable than
                trying to ``del`` references in a single process.
            load_transformer: If False, skip loading the transformer entirely.
                Use this in a dedicated encoding process to avoid paying the
                disk-read cost for ~38 GiB of weights that won't be used.
        """
        from diffusers import (
            AutoencoderKLQwenImage,
            FlowMatchEulerDiscreteScheduler,
            QwenImagePipeline,
            QwenImageTransformer2DModel,
        )

        self._dtype = dtype or torch.bfloat16
        self._device = device
        self._pipeline = None
        self._vae = None
        self._model = None
        self._transformer_on_device = False

        # ── Encoders (pipeline + VAE) ─────────────────────────────
        if load_encoders:
            self._pipeline = QwenImagePipeline.from_pretrained(
                pretrained_path, transformer=None, vae=None, torch_dtype=self._dtype,
            )
            self._pipeline.to(device)

            self._vae = AutoencoderKLQwenImage.from_pretrained(pretrained_path, subfolder="vae")
            self._vae.to(device, dtype=self._dtype)
            self._vae.eval()
            self._vae.requires_grad_(False)

        # VAE config is metadata only — needed for latent normalization
        # in the loss function even when load_encoders=False.
        self._vae_config = AutoencoderKLQwenImage.load_config(pretrained_path, subfolder="vae")
        self._init_vae_normalization()

        if "temporal_downsample" in self._vae_config:
            self._vae_scale_factor = 2 ** len(self._vae_config["temporal_downsample"])
        elif "temperal_downsample" in self._vae_config:
            self._vae_scale_factor = 2 ** len(self._vae_config["temperal_downsample"])
        else:
            logger.warning("Could not find temporal_downsample in VAE config, using default scale factor of 8")
            self._vae_scale_factor = 8

        # ── Transformer ───────────────────────────────────────────
        if load_transformer:
            self._model = QwenImageTransformer2DModel.from_pretrained(pretrained_path, subfolder="transformer")
            if defer_transformer and load_encoders:
                # Only worth deferring when encoders share the GPU.
                self._model.to("cpu", dtype=self._dtype)
                logger.info("Transformer loaded to CPU; call move_transformer_to_device() after free_encoders()")
            else:
                self._model.to(device, dtype=self._dtype)
                self._transformer_on_device = True

        # Scheduler (always cheap, CPU-only)
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
        self._scheduler_copy = copy.deepcopy(self._scheduler)

        # Warm up VAE if loaded
        if self._vae is not None:
            dummy = torch.zeros(1, 3, 1, 64, 64).to(device=device, dtype=self._dtype)
            self._vae.encode(dummy)

    def move_transformer_to_device(self, device=None):
        """Move the trainable transformer onto the GPU.

        Call this after :meth:`free_encoders` when ``defer_transformer=True``
        was used at init, so the freed text-encoder VRAM is available.
        """
        device = device or self._device
        self._model.to(device, dtype=self._dtype)
        self._device = device
        self._transformer_on_device = True
        logger.info("Transformer moved to %s", device)

    def _init_vae_normalization(self):
        cfg = self._vae_config
        if "latents_mean" in cfg and "latents_std" in cfg and "z_dim" in cfg:
            self._latents_mean = torch.tensor(cfg["latents_mean"]).view(1, 1, cfg["z_dim"], 1, 1)
            self._latents_std = 1.0 / torch.tensor(cfg["latents_std"]).view(1, 1, cfg["z_dim"], 1, 1)
            self._has_normalization = True
        else:
            self._has_normalization = False
            logger.warning("VAE config missing normalization parameters, skipping latent normalization")

    @property
    def model(self):
        return self._model

    @property
    def noise_scheduler(self):
        return self._scheduler

    def encode_image_tensor(self, image_tensor, device=None):
        device = device or self._device
        pixel_values = image_tensor.unsqueeze(2).to(device=device, dtype=self._dtype)
        with torch.no_grad():
            latents = self._vae.encode(pixel_values).latent_dist.sample()
        return latents

    def encode_images(self, images, height=None, width=None, device=None):
        device = device or self._device
        latents_list = []

        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.open(image) if isinstance(image, str) else Image.fromarray(np.uint8(image))
            image = image.convert("RGB")

            if height and width:
                image = self._pipeline.image_processor.resize(image, height, width)

            img_np = np.array(image).astype(np.float32)
            img_tensor = torch.from_numpy(img_np / 127.5 - 1.0).permute(2, 0, 1)

            pixel_values = img_tensor.unsqueeze(0).unsqueeze(2).to(device=device, dtype=self._dtype)
            latents = self._vae.encode(pixel_values).latent_dist.sample()
            latents_list.append(latents[0])

        return torch.stack(latents_list)

    def encode_text(self, prompts, device=None, max_sequence_length=1024, **kwargs):
        """Encode text prompts via QwenImagePipeline.encode_prompt.

        Unlike the Edit variant, no reference image is involved — the
        text encoder runs on prompt tokens alone. ``images`` is accepted
        and ignored so callers (e.g. cache_embeddings) can pass the same
        kwargs they pass to the Edit adapter.
        """
        device = device or self._device

        prompt_embeds, prompt_embeds_mask = self._pipeline.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        result = {"prompt_embeds": prompt_embeds}
        if prompt_embeds_mask is not None:
            result["prompt_embeds_mask"] = prompt_embeds_mask
        return result

    def normalize_latents(self, latents):
        if not self._has_normalization:
            return latents
        mean = self._latents_mean.to(latents.device, latents.dtype)
        std = self._latents_std.to(latents.device, latents.dtype)
        return (latents - mean) * std

    def sample_timesteps(self, batch_size, device):
        from diffusers.training_utils import compute_density_for_timestep_sampling

        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=batch_size,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self._scheduler_copy.config.num_train_timesteps).long()
        timesteps = self._scheduler_copy.timesteps[indices].to(device=device)
        sigmas = self._get_sigmas(timesteps, device=device, dtype=torch.float32)
        return timesteps, sigmas

    def _get_sigmas(self, timesteps, device, dtype=torch.float32, n_dim=5):
        sigmas = self._scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self._scheduler_copy.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def add_noise(self, latents, noise, timesteps, sigmas):
        sigmas = sigmas.to(latents.device, latents.dtype)
        return (1.0 - sigmas) * latents + sigmas * noise

    def compute_target(self, noise, latents, sigmas):
        return noise - latents

    def forward(self, model, noisy_latents, timesteps, batch):
        """Forward pass for Qwen-Image T2I.

        Unlike the Edit variant, no control image is concatenated;
        the transformer sees only the noised target latents plus the
        text conditioning.
        """
        from diffusers import QwenImagePipeline

        prompt_embeds = batch["prompt_embeds"].to(noisy_latents.device, noisy_latents.dtype)

        if "prompt_embeds_mask" in batch and isinstance(batch["prompt_embeds_mask"], torch.Tensor):
            prompt_mask = batch["prompt_embeds_mask"].to(dtype=torch.int32, device=noisy_latents.device)
        else:
            prompt_mask = torch.ones(
                prompt_embeds.shape[:2], dtype=torch.int32, device=noisy_latents.device,
            )

        bsz = noisy_latents.shape[0]

        packed_noisy = QwenImagePipeline._pack_latents(
            noisy_latents, bsz,
            noisy_latents.shape[2], noisy_latents.shape[3], noisy_latents.shape[4],
        )

        # Single image in the packed sequence — no control.
        img_shapes = [
            [(1, noisy_latents.shape[3] // 2, noisy_latents.shape[4] // 2)]
        ] * bsz

        txt_seq_lens = prompt_mask.sum(dim=1).tolist()

        output = model(
            hidden_states=packed_noisy,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        output = QwenImagePipeline._unpack_latents(
            output,
            height=noisy_latents.shape[3] * self._vae_scale_factor,
            width=noisy_latents.shape[4] * self._vae_scale_factor,
            vae_scale_factor=self._vae_scale_factor,
        )

        return output

    def save_lora(self, model, path):
        import os

        from diffusers import QwenImagePipeline
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft.utils import get_peft_model_state_dict

        os.makedirs(path, exist_ok=True)
        state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
        QwenImagePipeline.save_lora_weights(path, state_dict, safe_serialization=True)
        logger.info("LoRA weights saved to %s", path)

    def save_model(self, model, path):
        import os

        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path, safe_serialization=True)
        logger.info("Model saved to %s", path)

    def free_encoders(self):
        """Drop VAE + text encoder from GPU and reclaim VRAM.

        ``del self._pipeline`` alone doesn't release the underlying GPU
        tensors — Qwen-VL has multiple sub-modules (language tower,
        vision tower) and lingering refs through tokenizer / processor
        wiring keep tensors alive. The reliable path is to explicitly
        ``.to("cpu")`` the components first so the parameter buffers
        physically migrate off-device, then drop refs.
        """
        if self._pipeline is not None:
            try:
                self._pipeline.to("cpu")
            except Exception as e:
                logger.warning("pipeline.to('cpu') failed: %s", e)
        if self._vae is not None:
            try:
                self._vae.to("cpu")
            except Exception as e:
                logger.warning("vae.to('cpu') failed: %s", e)
        self._pipeline = None
        self._vae = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Freed VAE and text encoder from memory")
