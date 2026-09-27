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

    def compute_target(self, noise, latents, sigmas, timesteps=None):
        """What the model should predict.

        Flow matching: noise - latents (velocity)
        Epsilon: noise
        V-prediction: alpha_t * noise - sigma_t * latents (needs timesteps)

        Losses always pass ``timesteps=`` as a keyword.
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

## Start from a shared base

Most new models don't need the full protocol written from scratch:

| Base | Gives you | Used by |
|---|---|---|
| `DDPMAdapter` (`adapters/ddpm.py`) | DDPM timesteps, `add_noise`, epsilon / v-prediction / sample targets | `StableDiffusionAdapter`, `SDXLAdapter` |
| `FlowMatchAdapter` (`adapters/flow.py`) | Rectified-flow noising + velocity target, shifted logit-normal / uniform / mode timestep sampling | everything below |
| `DiffusersFlowAdapter` (`adapters/flow.py`) | + loading (encoders / VAE / deferred transformer / single-file transformer), VAE-normalized `encode_images`, `free_encoders`, PEFT-format `save_lora` | `SD3Adapter`, `FluxAdapter`, `ChromaAdapter`, `ZImageAdapter` |

A new diffusers-backed flow DiT is then two class attributes and two methods.
The FLUX adapter is the worked example:

```python
class FluxAdapter(DiffusersFlowAdapter):
    pipeline_class = "FluxPipeline"              # diffusers attribute names
    transformer_class = "FluxTransformer2DModel"

    def encode_text(self, prompts, device=None, **kwargs):
        prompt_embeds, pooled, _ = self._pipeline.encode_prompt(prompt=prompts, prompt_2=None, device=device)
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled}

    def forward(self, model, noisy_latents, timesteps, batch):
        # pack 2x2 patches, build RoPE ids, feed sigma (= t / 1000) + guidance, unpack
        ...
```

Whatever `encode_text` returns is cached by `cache_embeddings`, carried by
`EditingDataset`, and stacked by `EditingCollator` (`prompt_embeds` /
`prompt_embeds_mask` are padded to the longest sequence; other tensors are
stacked as-is), so extra conditioning like `pooled_prompt_embeds` reaches
`forward` without touching the data pipeline.

**Get the conventions from a reference implementation, not from vibes.**
Things like "Z-Image is fed `1 - sigma` and its output is negated" or "FLUX
text RoPE ids are all zero" are invisible in a loss curve until samples come
out wrong. [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
is a compact, single-codebase reference for per-model latent scale/shift,
timestep conventions, flow shift defaults, and text-encoder layer choices;
the diffusers pipeline's `__call__` is the other. Pin each one down with a
test against a tiny randomly-initialized transformer (see
`tests/test_model_adapters.py`).

## Key decisions

**What goes in the adapter vs the loss function?**

- **Adapter**: Model loading, encoding, the forward pass signature, saving. Anything that changes when you switch model architecture but not training objective.
- **Loss function**: The training objective. Noise sampling, target computation, loss calculation. Anything that changes when you switch from SFT-style to DPO but not model architecture.

**Why not just one big class?**

Because adapters and losses compose independently. `FlowMatchingLoss` works with `QwenEditAdapter`, `FluxAdapter`, `SD3Adapter`, `ZImageAdapter`, or any future flow matching model. `DiffusionDPOLoss` works with any adapter — DDPM or flow — because it only talks to the adapter protocol. You get M x N combinations from M adapters and N losses.

## Tips

- Keep the VAE in float32 if you see NaN latents (especially SDXL).
- Free text encoders after pre-computing embeddings via `adapter.free_encoders()`.
- The `forward()` method receives the full batch dict, so you can access any data the collator puts there.
- Test your adapter with a single training step before running a full job.
