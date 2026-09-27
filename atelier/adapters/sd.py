import gc
import logging

import torch

from .base import is_single_file, pil_to_tensor, save_peft_lora
from .ddpm import DDPMAdapter

logger = logging.getLogger(__name__)


class StableDiffusionAdapter(DDPMAdapter):
    """Adapter for Stable Diffusion 1.x / 2.x (UNet + single CLIP + DDPM).

    - SD 1.x: CLIP ViT-L/14, epsilon prediction, 512px
    - SD 2.x: OpenCLIP ViT-H/14, epsilon (``-base``) or v-prediction (768-v)

    The prediction type comes from the scheduler config; override with
    ``prediction_type=`` for single-file checkpoints whose embedded config
    guesses wrong (SD 2.x 768-v is the usual offender).

    ``clip_skip`` follows the diffusers / A1111 convention: ``None`` or
    ``1`` uses the final CLIP layer, ``2`` the penultimate (still run
    through the final layer norm), and so on.
    """

    def __init__(self, base_model, device="cuda", dtype=None, prediction_type=None, clip_skip=None):
        from diffusers import DDPMScheduler, StableDiffusionPipeline

        self._dtype = dtype or torch.float16
        self._device = device
        self._clip_skip = clip_skip

        load_kwargs = {"torch_dtype": self._dtype, "safety_checker": None, "requires_safety_checker": False}
        if is_single_file(base_model):
            pipe = StableDiffusionPipeline.from_single_file(base_model, **load_kwargs)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(base_model, **load_kwargs)

        self._pipe = pipe
        self._model = pipe.unet
        self._model.to(device)
        self._tokenizer = pipe.tokenizer
        self._text_encoder = pipe.text_encoder
        self._text_encoder.to(device)
        self._text_encoder.requires_grad_(False)

        # VAE in float32 — the SD 1.x/2.x VAE overflows in fp16 at high res.
        self._vae = pipe.vae
        self._vae.to(device, dtype=torch.float32)
        self._vae.eval()
        self._vae.requires_grad_(False)

        self._init_ddpm(DDPMScheduler.from_config(pipe.scheduler.config), prediction_type)
        logger.info("Stable Diffusion adapter: prediction_type=%s", self.prediction_type)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def encode_image_tensor(self, image_tensor, device=None):
        device = device or self._device
        image_tensor = image_tensor.to(dtype=torch.float32, device=device)
        with torch.no_grad():
            latents = self._vae.encode(image_tensor).latent_dist.sample()
        return latents * self._vae.config.scaling_factor

    def encode_images(self, images, height=None, width=None, device=None, **kwargs):
        device = device or self._device
        latents_list = []
        for image in images:
            tensor = pil_to_tensor(image, height, width).unsqueeze(0)
            latents_list.append(self.encode_image_tensor(tensor, device=device)[0])
        return torch.stack(latents_list)

    def encode_text(self, prompts=None, device=None, batch=None, **kwargs):
        """Encode prompts (or a pre-tokenized ``batch`` with ``input_ids``)."""
        device = device or self._device
        if batch is not None:
            input_ids = batch["input_ids"].to(device)
        else:
            input_ids = self._tokenizer(
                prompts, padding="max_length", max_length=self._tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)

        if not self._clip_skip or self._clip_skip <= 1:
            prompt_embeds = self._text_encoder(input_ids)[0]
        else:
            out = self._text_encoder(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-self._clip_skip]
            prompt_embeds = self._text_encoder.text_model.final_layer_norm(hidden)
        return {"prompt_embeds": prompt_embeds}

    def forward(self, model, noisy_latents, timesteps, batch):
        prompt_embeds = batch["prompt_embeds"].to(noisy_latents.device, noisy_latents.dtype)
        return model(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

    def save_lora(self, model, path):
        from diffusers import StableDiffusionPipeline

        save_peft_lora(StableDiffusionPipeline, model, path, layers_kwarg="unet_lora_layers")
        logger.info("LoRA weights saved to %s", path)

    def save_model(self, model, path):
        import os

        os.makedirs(path, exist_ok=True)
        self._pipe.unet = model
        self._pipe.save_pretrained(path, safe_serialization=True)
        logger.info("Full pipeline saved to %s", path)

    def free_encoders(self):
        """Free the text encoder. The VAE stays for on-the-fly image encoding."""
        if self._text_encoder is not None:
            self._text_encoder.to("cpu")
        self._text_encoder = None
        self._pipe.text_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Freed text encoder from memory")

