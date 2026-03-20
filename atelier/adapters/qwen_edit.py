import copy
import gc
import logging

import numpy as np
import torch
from PIL import Image

from .base import ModelAdapter

logger = logging.getLogger(__name__)


class QwenEditAdapter(ModelAdapter):
    """Adapter for Qwen-Image-Edit (DiT + video VAE + flow matching).

    Handles:
    - Loading QwenImageTransformer2DModel, AutoencoderKLQwenImage, QwenImageEditPipeline
    - Text encoding via pipeline.encode_prompt (with control image conditioning)
    - Image encoding via video VAE ([B, C, 1, H, W] shape)
    - Latent normalization, packing/unpacking
    - Flow matching timestep sampling and noise schedule
    - LoRA saving via QwenImagePipeline.save_lora_weights
    """

    def __init__(self, pretrained_path, device="cuda", dtype=None):
        from diffusers import (
            AutoencoderKLQwenImage,
            FlowMatchEulerDiscreteScheduler,
            QwenImageEditPipeline,
            QwenImageTransformer2DModel,
        )

        self._dtype = dtype or torch.bfloat16

        # Load text encoding pipeline (no transformer/VAE — just text encoder)
        self._pipeline = QwenImageEditPipeline.from_pretrained(
            pretrained_path, transformer=None, vae=None, torch_dtype=self._dtype,
        )
        self._pipeline.to(device)

        # Load VAE
        self._vae = AutoencoderKLQwenImage.from_pretrained(pretrained_path, subfolder="vae")
        self._vae.to(device, dtype=self._dtype)
        self._vae.eval()
        self._vae.requires_grad_(False)

        # Load VAE config for latent normalization
        self._vae_config = AutoencoderKLQwenImage.load_config(pretrained_path, subfolder="vae")
        self._init_vae_normalization()

        # Compute VAE scale factor
        if "temporal_downsample" in self._vae_config:
            self._vae_scale_factor = 2 ** len(self._vae_config["temporal_downsample"])
        elif "temperal_downsample" in self._vae_config:
            self._vae_scale_factor = 2 ** len(self._vae_config["temperal_downsample"])
        else:
            logger.warning("Could not find temporal_downsample in VAE config, using default scale factor of 8")
            self._vae_scale_factor = 8

        # Load transformer (the trainable model)
        self._model = QwenImageTransformer2DModel.from_pretrained(pretrained_path, subfolder="transformer")
        self._model.to(device, dtype=self._dtype)

        # Load noise scheduler
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
        self._scheduler_copy = copy.deepcopy(self._scheduler)

        self._device = device

        # Warm up VAE
        dummy = torch.zeros(1, 3, 1, 64, 64).to(device=device, dtype=self._dtype)
        self._vae.encode(dummy)

    def _init_vae_normalization(self):
        """Pre-compute latent normalization tensors from VAE config."""
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

    def encode_images(self, images, height=None, width=None, device=None):
        """Encode PIL images to latents via video VAE.

        Returns tensor of shape [B, C, 1, H', W'] (video VAE format).
        """
        device = device or self._device
        latents_list = []

        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.open(image) if isinstance(image, str) else Image.fromarray(np.uint8(image))
            image = image.convert("RGB")

            if height and width:
                image = self._pipeline.image_processor.resize(image, height, width)

            # Convert to tensor [C, H, W] in [-1, 1]
            img_np = np.array(image).astype(np.float32)
            img_tensor = torch.from_numpy(img_np / 127.5 - 1.0).permute(2, 0, 1)

            # Video VAE expects [B, C, 1, H, W]
            pixel_values = img_tensor.unsqueeze(0).unsqueeze(2).to(device=device, dtype=self._dtype)
            latents = self._vae.encode(pixel_values).latent_dist.sample()
            latents_list.append(latents[0])

        return torch.stack(latents_list)

    def encode_text(self, prompts, images=None, device=None, max_sequence_length=1024, **kwargs):
        """Encode text prompts via QwenImageEditPipeline.encode_prompt.

        Args:
            prompts: List of prompt strings.
            images: Optional list of control images for conditioning.
            device: Target device.
            max_sequence_length: Max token length.

        Returns dict with 'prompt_embeds' and 'prompt_embeds_mask'.
        """
        device = device or self._device
        image = images[0] if images else None

        prompt_embeds, prompt_embeds_mask = self._pipeline.encode_prompt(
            image=image,
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
        """Apply latent normalization from VAE config."""
        if not self._has_normalization:
            return latents
        mean = self._latents_mean.to(latents.device, latents.dtype)
        std = self._latents_std.to(latents.device, latents.dtype)
        return (latents - mean) * std

    def sample_timesteps(self, batch_size, device):
        """Sample timesteps using density-based sampling for flow matching."""
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
        """Look up sigmas for given timesteps."""
        sigmas = self._scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self._scheduler_copy.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def add_noise(self, latents, noise, timesteps, sigmas):
        """Flow matching noise addition: (1 - sigma) * latents + sigma * noise."""
        sigmas = sigmas.to(latents.device, latents.dtype)
        return (1.0 - sigmas) * latents + sigmas * noise

    def compute_target(self, noise, latents, sigmas):
        """Flow matching target: noise - latents."""
        return noise - latents

    def forward(self, model, noisy_latents, timesteps, batch):
        """Run QwenImageTransformer2DModel with latent packing and control conditioning."""
        from diffusers import QwenImageEditPipeline

        control_latents = batch["control_latents"].to(noisy_latents.device, noisy_latents.dtype)
        prompt_embeds = batch["prompt_embeds"].to(noisy_latents.device, noisy_latents.dtype)

        if "prompt_embeds_mask" in batch and isinstance(batch["prompt_embeds_mask"], torch.Tensor):
            prompt_mask = batch["prompt_embeds_mask"].to(dtype=torch.int32, device=noisy_latents.device)
        else:
            prompt_mask = torch.ones(
                prompt_embeds.shape[:2], dtype=torch.int32, device=noisy_latents.device,
            )

        bsz = noisy_latents.shape[0]

        # Pack latents
        packed_noisy = QwenImageEditPipeline._pack_latents(
            noisy_latents, bsz,
            noisy_latents.shape[2], noisy_latents.shape[3], noisy_latents.shape[4],
        )
        packed_control = QwenImageEditPipeline._pack_latents(
            control_latents, bsz,
            control_latents.shape[2], control_latents.shape[3], control_latents.shape[4],
        )

        # Concatenate target + control
        packed_input = torch.cat([packed_noisy, packed_control], dim=1)

        # Image shapes for RoPE
        img_shapes = [
            [
                (1, noisy_latents.shape[3] // 2, noisy_latents.shape[4] // 2),
                (1, control_latents.shape[3] // 2, control_latents.shape[4] // 2),
            ]
        ] * bsz

        txt_seq_lens = prompt_mask.sum(dim=1).tolist()

        # Forward pass
        output = model(
            hidden_states=packed_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        # Extract prediction for target (first half)
        output = output[:, :packed_noisy.size(1)]

        # Unpack
        output = QwenImageEditPipeline._unpack_latents(
            output,
            height=noisy_latents.shape[3] * self._vae_scale_factor,
            width=noisy_latents.shape[4] * self._vae_scale_factor,
            vae_scale_factor=self._vae_scale_factor,
        )

        return output

    def save_lora(self, model, path):
        """Save LoRA weights in diffusers format."""
        import os

        from diffusers import QwenImagePipeline
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft.utils import get_peft_model_state_dict

        os.makedirs(path, exist_ok=True)
        state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
        QwenImagePipeline.save_lora_weights(path, state_dict, safe_serialization=True)
        logger.info("LoRA weights saved to %s", path)

    def save_model(self, model, path):
        """Save full model weights."""
        import os

        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path, safe_serialization=True)
        logger.info("Model saved to %s", path)

    def free_encoders(self):
        """Free VAE and text encoding pipeline from memory."""
        del self._vae
        del self._pipeline
        self._vae = None
        self._pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Freed VAE and text encoder from memory")
