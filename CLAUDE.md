# Atelier

Simple, multi-GPU diffusion model fine-tuning library. Sister project to Grimoire. Training engine for Merlina.

## Philosophy

One training loop, pluggable adapters and loss functions. Grimoire handles text (transformers), Atelier handles images (diffusers). Same principles: no CLI, no plugins, no unnecessary abstractions.

Adding a new model means writing an adapter. Adding a new training objective means writing a loss function. The trainer never changes.

## Stack

- `accelerate` for multi-GPU / DeepSpeed / FSDP (NOT diffusers trainers or transformers.Trainer)
- `diffusers` for diffusion models, VAEs, schedulers
- `peft` for LoRA
- `torch` for everything else

## Structure

```
atelier/
├── __init__.py          # Public API
├── config.py            # TrainingConfig dataclass
├── trainer.py           # AtelierTrainer — the training loop
├── callbacks.py         # TrainerCallback base class (same interface as Grimoire)
├── adapters/
│   ├── base.py          # ModelAdapter protocol
│   ├── qwen_edit.py     # Qwen-Image-Edit (DiT + video VAE + flow matching)
│   ├── sdxl.py          # SDXL (UNet + dual CLIP + DDPM)
│   └── flux.py          # FLUX.1-dev (DiT + T5/CLIP)
├── losses/
│   ├── flow_matching.py # Flow matching MSE (Qwen, SD3, FLUX)
│   ├── diffusion_dpo.py # DPO on noise prediction + SFT regularization
│   ├── epsilon.py       # Epsilon prediction (DDPM)
│   └── v_prediction.py  # V-prediction
└── data/
    ├── editing.py       # Paired image editing dataset + collator
    ├── generation.py    # Text-to-image dataset + collator
    └── cache.py         # Embedding pre-computation + disk caching
```

## Key Design Decisions

- Uses `accelerate.Accelerator` directly for full control over the training loop
- **Adapters** encapsulate model-specific behavior (loading, forward pass, latent packing, saving)
  - In Grimoire, every model has the same forward signature (`model(input_ids)` → logits)
  - In diffusion, forward passes vary wildly per architecture (latent packing, conditioning, kwargs)
  - The adapter is the main pluggable unit — it owns the model-specific forward pass
- **Loss functions** are callables: `loss, metrics = loss_fn(adapter, model, batch)`
  - Loss functions orchestrate: sample noise → add noise → forward → compute objective
  - Loss functions own their data collators via `create_collator()`
- Multi-GPU, DeepSpeed, FSDP work out of the box via `accelerate config`
- LoRA via PEFT is optional — supports both LoRA and full model training
- Embedding pre-computation supported via `data/cache.py` for memory-constrained training
- VAE and text encoders are always frozen — only the denoising model (transformer/UNet) trains

## Adapter Protocol

Adapters handle everything that varies per model architecture:

```python
class ModelAdapter:
    def load_components(self, path, device, dtype)  # Load model + VAE + text encoder + scheduler
    def model -> nn.Module                           # The trainable model
    def encode_images(self, images) -> Tensor        # VAE encode
    def encode_text(self, prompts, **kw) -> dict     # Text encode
    def sample_timesteps(self, bsz) -> (t, sigmas)  # Timestep sampling
    def add_noise(self, latents, noise, t, sigmas)   # Create noisy input
    def compute_target(self, noise, latents, sigmas)  # What model should predict
    def forward(self, noisy, timesteps, batch)        # Architecture-specific forward
    def save_lora(self, model, path)                  # LoRA weight saving
    def save_model(self, model, path)                 # Full model saving
```

## Prior Art

Atelier consolidates and generalizes two existing trainers in this project:

- **Qwen-Image-Edit-LoRA-Trainer** — Flow matching LoRA training for Qwen-Image-Edit DiT
  - Becomes: `adapters/qwen_edit.py` + `losses/flow_matching.py` + `data/editing.py`
- **diffusion-dpo-trainer** — DPO training for SDXL UNet
  - Becomes: `adapters/sdxl.py` + `losses/diffusion_dpo.py` + `data/generation.py`

## Usage

```python
from atelier import AtelierTrainer, TrainingConfig
from atelier.adapters import QwenEditAdapter, SDXLAdapter
from atelier.losses import FlowMatchingLoss, DiffusionDPOLoss
from peft import LoraConfig

# Qwen Image Edit LoRA training
adapter = QwenEditAdapter("Qwen/Qwen-Image-Edit")
trainer = AtelierTrainer(
    adapter=adapter,
    config=TrainingConfig(output_dir="./output", num_epochs=50, batch_size=1),
    loss_fn=FlowMatchingLoss(),
    train_dataset=dataset,
    peft_config=LoraConfig(r=64, lora_alpha=128, target_modules=["to_k", "to_q", "to_v", "to_out.0"]),
)
trainer.train()
trainer.save_model("./my-lora")

# SDXL DPO training
adapter = SDXLAdapter("stabilityai/stable-diffusion-xl-base-1.0", weights="model.safetensors")
trainer = AtelierTrainer(
    adapter=adapter,
    config=TrainingConfig(output_dir="./output", num_epochs=10, batch_size=1),
    loss_fn=DiffusionDPOLoss(beta=0.4, sft_weight=0.3),
    train_dataset=dpo_dataset,
)
trainer.train()
trainer.save_model("./my-sdxl")
```

## Commands

```bash
pip install -e .                    # Install in dev mode
pip install -e ".[quantization]"    # With bitsandbytes
pip install -e ".[logging]"         # With wandb
accelerate config                   # Configure multi-GPU / DeepSpeed
accelerate launch script.py         # Run distributed training
pytest                              # Run tests
```

## Relationship to Grimoire and Merlina

Atelier is a standalone library, sister to Grimoire. Merlina dispatches to either:
- **Grimoire** for text model training (causal LMs via transformers)
- **Atelier** for image model training (diffusion models via diffusers)

Both share the same callback interface, config patterns, and accelerate-based training loop design.

## CI Requirements

Before considering any work done, you MUST ensure:
1. `ruff check .` passes with no errors
2. `pytest` passes with no failures

## Testing

```bash
pytest                              # All tests
pytest tests/test_losses.py         # Loss computation tests
pytest tests/test_trainer.py        # Trainer tests
```
