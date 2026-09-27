"""Shared DDPM plumbing for UNet-era models (SD 1.x / 2.x / SDXL).

Covers the three prediction types those checkpoints ship with:

- ``epsilon``       — SD 1.x, SD 2.x-base, SDXL
- ``v_prediction``  — SD 2.x 768-v, and the v-pred SDXL finetunes
- ``sample``        — rare, x0-prediction

The prediction type is read from the scheduler config and can be
overridden, since single-file / community checkpoints often ship a
scheduler config that doesn't match how the UNet was actually trained.
"""

import torch

from .base import ModelAdapter

PREDICTION_TYPES = ("epsilon", "v_prediction", "sample")


class DDPMAdapter(ModelAdapter):
    """Base for DDPM-scheduled adapters. Subclasses call :meth:`_init_ddpm`."""

    def _init_ddpm(self, scheduler, prediction_type=None):
        prediction_type = prediction_type or scheduler.config.prediction_type
        if prediction_type not in PREDICTION_TYPES:
            raise ValueError(f"prediction_type must be one of {PREDICTION_TYPES}, got '{prediction_type}'")
        if scheduler.config.prediction_type != prediction_type:
            scheduler.register_to_config(prediction_type=prediction_type)
        self._scheduler = scheduler
        self.prediction_type = prediction_type

    @property
    def noise_scheduler(self):
        return self._scheduler

    def sample_timesteps(self, batch_size, device):
        """Uniform DDPM timesteps. Returns (timesteps, None) — DDPM has no sigmas."""
        T = self._scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)
        return timesteps, None

    def add_noise(self, latents, noise, timesteps, sigmas=None):
        return self._scheduler.add_noise(latents, noise, timesteps)

    def compute_target(self, noise, latents, sigmas=None, timesteps=None):
        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "sample":
            return latents
        if timesteps is None:
            raise ValueError("v_prediction target needs timesteps; pass timesteps= to compute_target()")
        return self._scheduler.get_velocity(latents, noise, timesteps)
