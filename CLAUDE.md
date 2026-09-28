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
├── registry.py          # String → adapter/loss class resolution (for YAML/CLI)
├── train.py             # YAML config + CLI entry point (python -m atelier.train)
├── adapters/
│   ├── base.py          # ModelAdapter protocol + shared helpers (PEFT LoRA save, single-file detection)
│   ├── ddpm.py          # DDPMAdapter base — epsilon / v-prediction / sample targets
│   ├── flow.py          # FlowMatchAdapter (shift + timestep sampling) + DiffusersFlowAdapter (loading/VAE/save)
│   ├── sd.py            # SD 1.x / 2.x (UNet + CLIP + DDPM)
│   ├── sdxl.py          # SDXL (UNet + dual CLIP + DDPM)
│   ├── sd3.py           # SD 3 / 3.5 (MMDiT + CLIP-L/G + T5 + flow matching)
│   ├── flux.py          # FLUX.1 dev/schnell, Kontext (reference tokens), Chroma (T5-only, masked)
│   ├── z_image.py       # Z-Image (single-stream DiT + Qwen3; reversed time, negated output)
│   ├── qwen_edit.py     # Qwen-Image-Edit (DiT + video VAE + flow matching, image-conditioned text encoder)
│   ├── qwen_image.py    # Qwen-Image (DiT + video VAE + flow matching, text-to-image)
│   └── qwen_image_2_1.py # Qwen-Image 2.1 (native DiT port, Qwen3-VL, RGBA 64-ch VAE; per-sample layouts)
├── models/
│   └── qwen_image_2_1.py  # Native PyTorch port of the Qwen-Image 2.1 DiT (+ VAE config/key converter,
│                          #   Qwen3-VL conditioning) — diffusers has no support; parity-checked vs sd.cpp
├── losses/
│   ├── flow_matching.py   # Flow matching MSE (5-D Qwen video-VAE and 4-D SD3/FLUX/Z-Image latents)
│   ├── epsilon.py         # Epsilon prediction (DDPM)
│   ├── diffusion_dpo.py   # DPO on denoising MSE + SFT regularization
│   ├── diffusion_cpo.py   # CPO (reference-free contrastive preference)
│   ├── diffusion_ipo.py   # IPO (squared-loss preference; needs reference)
│   ├── diffusion_kto.py   # KTO (unpaired binary good/bad feedback)
│   ├── diffusion_simpo.py # SimPO (reference-free, length-normalized margin)
│   ├── diffusion_orpo.py  # ORPO (odds-ratio preference + SFT)
│   └── utils.py           # Shared paired/single denoising-loss helpers
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
    def compute_target(self, noise, latents, sigmas, timesteps=None)  # What model should predict
    def forward(self, noisy, timesteps, batch)        # Architecture-specific forward
    def save_lora(self, model, path)                  # LoRA weight saving
    def save_model(self, model, path)                 # Full model saving
```

## Adding a Model

- DDPM UNet → subclass `DDPMAdapter`; flow DiT on diffusers → subclass `DiffusersFlowAdapter`
  (set `pipeline_class` / `transformer_class`, implement `encode_text` + `forward`).
- Take per-model conventions (VAE scale/shift, timestep scaling/direction, flow shift, text-encoder
  layer, RoPE ids) from the diffusers pipeline `__call__` and cross-check against
  stable-diffusion.cpp (`src/model/`, `src/runtime/denoiser.hpp`, `src/model/vae/auto_encoder_kl.hpp`).
- Pin them with a test against a tiny random transformer in `tests/test_model_adapters.py`
  (no checkpoint downloads in tests).
- No diffusers support? Port the model into `atelier/models/` with checkpoint-compatible parameter
  names, and check it against sd.cpp's runner with `scripts/sdcpp_parity/` (C++ harness + golden
  outputs committed under `tests/data/` so CI re-checks parity without building sd.cpp).

## Prior Art

Atelier consolidates and generalizes two existing trainers in this project:

- **Qwen-Image-Edit-LoRA-Trainer** — Flow matching LoRA training for Qwen-Image-Edit DiT
  - Becomes: `adapters/qwen_edit.py` + `losses/flow_matching.py` + `data/editing.py`
- **diffusion-dpo-trainer** — DPO training for SDXL UNet
  - Becomes: `adapters/sdxl.py` + `losses/diffusion_dpo.py` + `data/generation.py`
  - Generalized into a family of preference losses sharing `losses/utils.py`:
    DPO, CPO, IPO, KTO, SimPO, ORPO

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
pip install -e ".[yaml]"            # With PyYAML (for the config CLI)
accelerate config                   # Configure multi-GPU / DeepSpeed
accelerate launch script.py         # Run distributed training
python -m atelier.train config.yaml # Train from a YAML config (adapter/loss by name)
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
