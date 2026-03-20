import torch


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

    def add_noise(self, latents, noise, timesteps, sigmas):
        """Create noisy input from clean latents.

        Flow matching: (1 - sigma) * latents + sigma * noise
        DDPM: scheduler.add_noise(latents, noise, timesteps)
        """
        raise NotImplementedError

    def compute_target(self, noise, latents, sigmas):
        """Compute what the model should predict.

        Flow matching: noise - latents
        Epsilon: noise
        V-prediction: sigma * noise - (1 - sigma) * latents (approx)
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
