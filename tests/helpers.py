"""Shared test fixtures — mock adapter and helpers."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from atelier.adapters.base import ModelAdapter


class DummyModel(nn.Module):
    """A tiny model that returns noise-shaped output for testing."""

    def __init__(self, channels=4, spatial=8):
        super().__init__()
        # Use a scalar parameter so it works with any input shape
        self.scale = nn.Parameter(torch.ones(1))
        self.channels = channels
        self.spatial = spatial

    @property
    def device(self):
        return self.scale.device

    @property
    def dtype(self):
        return self.scale.dtype

    def forward(self, x, *args, **kwargs):
        # Multiply by learnable scale to create gradient graph
        if x is not None:
            return x * self.scale
        return x


class MockAdapter(ModelAdapter):
    """Minimal adapter for testing loss functions and trainer."""

    def __init__(self, channels=4, spatial=8, num_train_timesteps=1000):
        self._model = DummyModel(channels, spatial)
        self._num_train_timesteps = num_train_timesteps
        self._noise_scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=num_train_timesteps),
        )

    @property
    def model(self):
        return self._model

    @property
    def noise_scheduler(self):
        return self._noise_scheduler

    def encode_image_tensor(self, image_tensor, device=None):
        b = image_tensor.shape[0]
        return torch.randn(b, self._model.channels, self._model.spatial, self._model.spatial)

    def encode_text(self, prompts=None, device=None, **kwargs):
        return {"prompt_embeds": torch.randn(1, 16, 64)}

    def sample_timesteps(self, batch_size, device):
        timesteps = torch.randint(0, self._num_train_timesteps, (batch_size,), device=device)
        sigmas = timesteps.float() / self._num_train_timesteps
        return timesteps, sigmas

    def add_noise(self, latents, noise, timesteps, sigmas):
        if sigmas is not None:
            s = sigmas.view(-1, *([1] * (latents.ndim - 1)))
            return (1 - s) * latents + s * noise
        return latents + noise

    def compute_target(self, noise, latents, sigmas):
        return noise - latents

    def forward(self, model, noisy_latents, timesteps, batch):
        out = model(noisy_latents)
        # If 5D (flow matching), permute [B, 1, C, H, W] -> [B, C, 1, H, W] like real models
        if out.ndim == 5 and out.shape[1] == 1:
            out = out.permute(0, 2, 1, 3, 4)
        return out

    def save_lora(self, model, path):
        pass

    def save_model(self, model, path):
        pass


class FlowMatchingMockAdapter(MockAdapter):
    """MockAdapter that returns 5D sigmas for flow matching (like QwenEditAdapter)."""

    def sample_timesteps(self, batch_size, device):
        timesteps, sigmas = super().sample_timesteps(batch_size, device)
        # Expand to 5D [B, 1, 1, 1, 1] like real flow matching adapters
        while sigmas.ndim < 5:
            sigmas = sigmas.unsqueeze(-1)
        return timesteps, sigmas
