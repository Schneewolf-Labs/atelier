# Writing a Custom Adapter

Adapters encapsulate everything that varies per model architecture. To add support for a new diffusion model, write an adapter class that inherits from `ModelAdapter`.

## The protocol

```python
from atelier.adapters.base import ModelAdapter

class MyAdapter(ModelAdapter):

    def __init__(self, pretrained_path, device="cuda", dtype=None):
        # Load your model components here:
        # - The trainable denoising model (transformer or UNet)
        # - The VAE
        # - Text encoder(s)
        # - Noise scheduler
        pass

    @property
    def model(self):
        """Return the trainable model (transformer or UNet)."""
        return self._model

    @property
    def noise_scheduler(self):
        """Return the noise scheduler."""
        return self._scheduler

    def encode_images(self, images, device=None, **kwargs):
        """Encode PIL images to latent space via VAE.

        Returns a tensor of latents, shape depends on your VAE:
        - Standard VAE: [B, C, H, W]
        - Video VAE (Qwen): [B, C, 1, H, W]
        """
        ...

    def encode_text(self, prompts=None, device=None, **kwargs):
        """Encode text prompts to embeddings.

        Returns a dict of tensors. The keys are architecture-specific
        and will be passed through to forward() via the batch.
        """
        ...

    def sample_timesteps(self, batch_size, device):
        """Sample training timesteps.

        Returns (timesteps, sigmas) where sigmas may be None for DDPM models.
        """
        ...

    def add_noise(self, latents, noise, timesteps, sigmas):
        """Create noisy input from clean latents.

        Flow matching: (1 - sigma) * latents + sigma * noise
        DDPM: scheduler.add_noise(latents, noise, timesteps)
        """
        ...

    def compute_target(self, noise, latents, sigmas):
        """What the model should predict.

        Flow matching: noise - latents (velocity)
        Epsilon: noise
        V-prediction: depends on scheduler
        """
        ...

    def forward(self, model, noisy_latents, timesteps, batch):
        """Run the model forward pass.

        This is where architecture-specific logic lives:
        latent packing, conditioning, special kwargs, etc.

        Args:
            model: The trainable model (may be wrapped by accelerate/PEFT)
            noisy_latents: Noised latent tensors
            timesteps: Sampled timesteps
            batch: Full batch dict (contains text embeddings, control images, etc.)

        Returns the model prediction tensor.
        """
        ...

    def save_lora(self, model, path):
        """Save LoRA weights in the format expected by this architecture."""
        ...

    def save_model(self, model, path):
        """Save full model weights."""
        ...
```

## Example: adding FLUX support

```python
class FluxAdapter(ModelAdapter):

    def __init__(self, pretrained_path, device="cuda", dtype=None):
        from diffusers import FluxPipeline, FluxTransformer2DModel

        self._dtype = dtype or torch.bfloat16
        pipe = FluxPipeline.from_pretrained(pretrained_path, torch_dtype=self._dtype)

        self._model = pipe.transformer
        self._vae = pipe.vae
        self._text_encoder = pipe.text_encoder
        self._text_encoder_2 = pipe.text_encoder_2
        self._scheduler = pipe.scheduler
        self._pipe = pipe
        # ... freeze encoders, etc.

    @property
    def model(self):
        return self._model

    def forward(self, model, noisy_latents, timesteps, batch):
        # FLUX-specific: pack latents, pass text/pooled embeds, guidance
        return model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=batch["prompt_embeds"],
            pooled_projections=batch["pooled_prompt_embeds"],
            return_dict=False,
        )[0]

    # ... implement remaining methods
```

## Key decisions

**What goes in the adapter vs the loss function?**

- **Adapter**: Model loading, encoding, the forward pass signature, saving. Anything that changes when you switch model architecture but not training objective.
- **Loss function**: The training objective. Noise sampling, target computation, loss calculation. Anything that changes when you switch from SFT-style to DPO but not model architecture.

**Why not just one big class?**

Because adapters and losses compose independently. `FlowMatchingLoss` works with `QwenEditAdapter`, `FluxAdapter`, or any future flow matching model. `DiffusionDPOLoss` works with `SDXLAdapter` or any epsilon-prediction model. You get M x N combinations from M adapters and N losses.

## Tips

- Keep the VAE in float32 if you see NaN latents (especially SDXL).
- Free text encoders after pre-computing embeddings via `adapter.free_encoders()`.
- The `forward()` method receives the full batch dict, so you can access any data the collator puts there.
- Test your adapter with a single training step before running a full job.
