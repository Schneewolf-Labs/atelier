import gc
import logging

import numpy as np
import torch
from PIL import Image

from .base import ModelAdapter

logger = logging.getLogger(__name__)


class SDXLAdapter(ModelAdapter):
    """Adapter for Stable Diffusion XL (UNet + dual CLIP + DDPM).

    Handles:
    - Loading SDXL pipeline, optional safetensors weight loading
    - Dual text encoder encoding (CLIP-L + CLIP-G)
    - VAE encoding with float32 stability
    - UNet forward pass with added_cond_kwargs
    - DDPM noise scheduling (epsilon prediction)
    - UNet layer freezing strategies
    - Full pipeline or UNet-only saving
    """

    def __init__(self, base_model, weights=None, use_base_vae=False, device="cuda", dtype=None):
        from diffusers import DDPMScheduler, StableDiffusionXLPipeline

        self._dtype = dtype or torch.float16
        self._device = device

        # Load base pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model, torch_dtype=self._dtype, use_safetensors=True,
            variant="fp16" if self._dtype == torch.float16 else None,
        )

        # Load custom weights if provided
        if weights is not None:
            self._load_weights(pipe, weights, use_base_vae)

        self._model = pipe.unet
        self._text_encoder = pipe.text_encoder
        self._text_encoder_2 = pipe.text_encoder_2
        self._tokenizer = pipe.tokenizer
        self._tokenizer_2 = pipe.tokenizer_2
        self._pipe = pipe

        # VAE in float32 to prevent NaN
        self._vae = pipe.vae
        self._vae.to(dtype=torch.float32)
        self._vae.eval()
        self._vae.requires_grad_(False)
        self._vae.enable_slicing()
        self._vae.enable_tiling()

        # Freeze text encoders
        self._text_encoder.requires_grad_(False)
        self._text_encoder_2.requires_grad_(False)

        # Noise scheduler
        self._scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

        # Test VAE
        self._test_vae(base_model)

    def _load_weights(self, pipe, weights_path, use_base_vae):
        """Load UNet (and optionally VAE) weights from a safetensors file."""
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
        has_vae = any(k.startswith("vae.") for k in state_dict)

        # Extract UNet weights (either prefixed or unprefixed)
        unet_sd = {
            k: v for k, v in state_dict.items()
            if k.startswith("unet.") or not any(
                k.startswith(p) for p in ["text_encoder.", "text_encoder_2.", "vae."]
            )
        }
        unet_sd = {k.replace("unet.", ""): v for k, v in unet_sd.items()}
        pipe.unet.load_state_dict(unet_sd, strict=False)
        logger.info("Loaded %d UNet parameters from %s", len(unet_sd), weights_path)

        if has_vae and not use_base_vae:
            vae_sd = {k.replace("vae.", ""): v for k, v in state_dict.items() if k.startswith("vae.")}
            pipe.vae.load_state_dict(vae_sd, strict=False)
            logger.info("Loaded %d VAE parameters from %s", len(vae_sd), weights_path)

    def _test_vae(self, base_model):
        """Verify VAE produces valid outputs; reload from base if needed."""
        with torch.no_grad():
            test = torch.randn(1, 3, 256, 256, device=self._device, dtype=self._vae.dtype)
            latent = self._vae.encode(test).latent_dist.sample()

            if torch.isnan(latent).any():
                logger.warning("VAE produced NaN, reloading from base model")
                from diffusers import AutoencoderKL

                self._vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float32)
                self._vae.eval()
                self._vae.requires_grad_(False)
                self._vae.to(self._device)
                self._pipe.vae = self._vae

    @property
    def model(self):
        return self._model

    @property
    def noise_scheduler(self):
        return self._scheduler

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def tokenizer_2(self):
        return self._tokenizer_2

    def encode_image_tensor(self, image_tensor, device=None):
        """Encode image tensors [B, C, H, W] in [-1, 1] to latents."""
        device = device or self._device
        image_tensor = image_tensor.to(dtype=torch.float32, device=device)
        with torch.no_grad():
            latents = self._vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self._vae.config.scaling_factor
        return latents

    def encode_images(self, images, device=None, **kwargs):
        """Encode PIL images to latents via VAE (float32 for stability)."""
        device = device or self._device
        latents_list = []

        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.open(image) if isinstance(image, str) else Image.fromarray(np.uint8(image))
            image = image.convert("RGB")

            img_np = np.array(image).astype(np.float32)
            img_tensor = torch.from_numpy(img_np / 127.5 - 1.0).permute(2, 0, 1)
            img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=self._vae.dtype)

            latents = self._vae.encode(img_tensor).latent_dist.sample()
            latents = latents * self._vae.config.scaling_factor
            latents_list.append(latents[0])

        return torch.stack(latents_list)

    def encode_text(self, prompts=None, device=None, batch=None, **kwargs):
        """Encode text using both SDXL text encoders.

        Can work from raw prompts or from a pre-tokenized batch.
        Returns dict with prompt_embeds, pooled_prompt_embeds, time_ids.
        """
        device = device or self._device

        if batch is not None:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            input_ids_2 = batch["input_ids_2"].to(device)
            attn_2 = batch["attention_mask_2"].to(device)
        else:
            tokens = self._tokenizer(
                prompts, padding="max_length",
                max_length=self._tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            tokens_2 = self._tokenizer_2(
                prompts, padding="max_length",
                max_length=self._tokenizer_2.model_max_length,
                truncation=True, return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device)
            attn = tokens.attention_mask.to(device)
            input_ids_2 = tokens_2.input_ids.to(device)
            attn_2 = tokens_2.attention_mask.to(device)

        out1 = self._text_encoder(input_ids, attention_mask=attn, output_hidden_states=True)
        prompt_embeds_1 = out1.hidden_states[-2]

        out2 = self._text_encoder_2(input_ids_2, attention_mask=attn_2, output_hidden_states=True)
        pooled_prompt_embeds = out2.text_embeds
        prompt_embeds_2 = out2.hidden_states[-2]

        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        bsz = input_ids.shape[0]
        time_ids = torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]] * bsz,
            dtype=torch.float32, device=device,
        )

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "time_ids": time_ids,
        }

    def sample_timesteps(self, batch_size, device):
        """Sample DDPM timesteps. Returns (timesteps, None) — no sigmas for DDPM."""
        T = self._scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)
        return timesteps, None

    def add_noise(self, latents, noise, timesteps, sigmas=None):
        """DDPM noise addition via scheduler."""
        return self._scheduler.add_noise(latents, noise, timesteps)

    def compute_target(self, noise, latents, sigmas=None):
        """Epsilon prediction target: just the noise."""
        return noise

    def forward(self, model, noisy_latents, timesteps, batch):
        """Run UNet forward pass with SDXL conditioning."""
        prompt_embeds = batch["prompt_embeds"]
        added_cond_kwargs = {
            "text_embeds": batch["pooled_prompt_embeds"],
            "time_ids": batch["time_ids"],
        }

        return model(
            noisy_latents, timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

    def freeze_layers(self, strategy="none", layers="0,1"):
        """Freeze selected UNet layers to reduce trainable parameters.

        Strategies:
            none: Don't freeze anything
            input_blocks / early_blocks: Freeze specified down_block indices
            color_blocks: Freeze specified down_blocks + conv_in
        """
        if strategy == "none":
            return

        freeze_indices = [int(x.strip()) for x in layers.split(",")]

        for idx, block in enumerate(self._model.down_blocks):
            if idx in freeze_indices:
                for param in block.parameters():
                    param.requires_grad = False
                logger.info("Frozen down_block %d", idx)

        if strategy == "color_blocks":
            self._model.conv_in.requires_grad_(False)
            logger.info("Frozen conv_in")

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info("UNet: %s trainable / %s total (%.1f%%)", f"{trainable:,}", f"{total:,}", trainable / total * 100)

    def save_lora(self, model, path):
        """Save LoRA weights."""
        import os

        from peft.utils import get_peft_model_state_dict

        os.makedirs(path, exist_ok=True)
        state_dict = get_peft_model_state_dict(model)
        torch.save(state_dict, os.path.join(path, "lora_weights.pt"))
        logger.info("LoRA weights saved to %s", path)

    def save_model(self, model, path):
        """Save full SDXL pipeline."""
        import os

        os.makedirs(path, exist_ok=True)
        model.save_pretrained(os.path.join(path, "unet"), safe_serialization=True)
        self._pipe.unet = model
        self._pipe.save_pretrained(path, safe_serialization=True)
        logger.info("Full pipeline saved to %s", path)

    def free_encoders(self):
        """Free text encoders from memory (VAE kept for on-the-fly encoding)."""
        del self._text_encoder
        del self._text_encoder_2
        self._text_encoder = None
        self._text_encoder_2 = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Freed text encoders from memory")
