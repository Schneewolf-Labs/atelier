import os

import numpy as np
import torch
from PIL import Image


def strip_peft_prefix(state_dict: dict) -> dict:
    """Strip ``base_model.model.`` (PEFT wrapper prefix) from state-dict keys.

    ``get_peft_model_state_dict`` retains the wrapper prefix in some PEFT
    versions; ``pipe.load_lora_weights`` then can't match the keys against
    the unwrapped model's module tree. Stripping here is the
    minimal-change fix that works across PEFT versions.

    Module-level so callers can use it without subclassing an adapter,
    and so it's trivially unit-testable without any ML stack imports.
    """
    return {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}


def save_peft_lora(pipeline_cls, model, path, layers_kwarg="transformer_lora_layers"):
    """Save a PEFT-wrapped model's LoRA via ``pipeline_cls.save_lora_weights``.

    Keys stay in PEFT format (``.lora_A`` / ``.lora_B``) — modern
    ``pipe.load_lora_weights`` expects that, not the legacy
    ``convert_state_dict_to_diffusers`` layout. ``layers_kwarg`` names the
    component: ``transformer_lora_layers`` for DiTs, ``unet_lora_layers``
    for UNets.
    """
    from peft.utils import get_peft_model_state_dict

    os.makedirs(path, exist_ok=True)
    state_dict = strip_peft_prefix(get_peft_model_state_dict(model))
    pipeline_cls.save_lora_weights(path, **{layers_kwarg: state_dict}, safe_serialization=True)


def is_single_file(path) -> bool:
    """True for a single checkpoint file (.safetensors / .ckpt / .pt / .bin).

    Community checkpoints — and everything stable-diffusion.cpp loads —
    usually ship as one file rather than a diffusers directory; those go
    through ``Pipeline.from_single_file`` instead of ``from_pretrained``.
    """
    return os.path.isfile(str(path)) and str(path).endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin"))


def pil_to_tensor(image, height=None, width=None):
    """PIL / path / array → ``[C, H, W]`` float tensor in [-1, 1], optionally resized."""
    if not isinstance(image, Image.Image):
        image = Image.open(image) if isinstance(image, str) else Image.fromarray(np.uint8(image))
    image = image.convert("RGB")
    if height and width:
        image = image.resize((width, height), Image.LANCZOS)
    tensor = torch.from_numpy(np.array(image).astype(np.float32) / 127.5 - 1.0)
    return tensor.permute(2, 0, 1)


class ModelAdapter:
    """Base class for model adapters.

    Adapters encapsulate everything that varies per model architecture:
    loading, encoding, forward pass, noise scheduling, and saving.
    """

    @property
    def model(self):
        """The trainable model (transformer or UNet)."""
        raise NotImplementedError

    @property
    def noise_scheduler(self):
        """The noise scheduler for this model."""
        raise NotImplementedError

    @property
    def device(self):
        """The active device for encoders + model.

        Override if the adapter doesn't store ``self._device``. Used by
        cache_embeddings + other utilities that need a device to run on
        but can't ask ``adapter.model.device`` (the model may not be
        loaded yet when only encoders are present).
        """
        return getattr(self, "_device", "cpu")

    def encode_images(self, images, device=None):
        """Encode PIL images to latent space via VAE.

        Returns a tensor of latents.
        """
        raise NotImplementedError

    def encode_image_tensor(self, image_tensor, device=None):
        """Encode a batch of image tensors [B, C, H, W] in [-1, 1] to latents.

        Used by loss functions for on-the-fly encoding from pre-processed tensors.
        """
        raise NotImplementedError

    def encode_text(self, prompts, device=None, **kwargs):
        """Encode text prompts to embeddings.

        Returns a dict of tensors (prompt_embeds, masks, etc.).
        """
        raise NotImplementedError

    def sample_timesteps(self, batch_size, device):
        """Sample timesteps and compute sigmas.

        Returns (timesteps, sigmas) tensors.
        """
        raise NotImplementedError

    def sigmas_for_timesteps(self, timesteps, device):
        """Return sigmas for explicitly-chosen timesteps, or None.

        Preference losses (DPO/IPO/...) can bias the timestep *range* instead
        of drawing from :meth:`sample_timesteps`. DDPM-style adapters don't use
        sigmas — their ``add_noise`` / ``compute_target`` ignore the argument —
        so the default returns None. Flow-matching adapters override this to
        look up the matching sigmas, so those losses stay correct on flow
        models instead of crashing on ``(1 - None)``.
        """
        return None

    def add_noise(self, latents, noise, timesteps, sigmas):
        """Create noisy input from clean latents.

        Flow matching: (1 - sigma) * latents + sigma * noise
        DDPM: scheduler.add_noise(latents, noise, timesteps)
        """
        raise NotImplementedError

    def compute_target(self, noise, latents, sigmas, timesteps=None):
        """Compute what the model should predict.

        Flow matching: noise - latents
        Epsilon: noise
        V-prediction: alpha_t * noise - sigma_t * latents (needs ``timesteps``)

        Losses always pass ``timesteps`` as a keyword; adapters whose target
        depends only on sigmas can ignore it.
        """
        raise NotImplementedError

    def forward(self, model, noisy_latents, timesteps, batch):
        """Run the model forward pass.

        Handles architecture-specific kwargs (packing, conditioning, etc.).
        Returns the model prediction tensor.
        """
        raise NotImplementedError

    def save_lora(self, model, path):
        """Save LoRA weights in architecture-specific format."""
        raise NotImplementedError

    def save_model(self, model, path):
        """Save full model weights."""
        raise NotImplementedError

    @torch.no_grad()
    def free_encoders(self):
        """Free VAE and text encoder(s) from memory.

        Call after pre-computing embeddings to reclaim VRAM before training.
        """
        pass
