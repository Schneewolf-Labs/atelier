# 🎨 Atelier 🔨

A simple, multi-GPU diffusion model fine-tuning library. One training loop, pluggable adapters and loss functions.

Sister project to [Grimoire](https://github.com/Schneewolf-Labs/Grimoire) (LLM fine-tuning). Both serve as training engines for [Merlina](https://github.com/Schneewolf-Labs/Merlina).

## Why

Diffusion model training scripts tend to be monolithic — model loading, data processing, the training loop, and architecture-specific forward passes all tangled together. Switching from SDXL to Qwen-Image-Edit means rewriting the whole script.

Atelier separates what varies (model architecture, training objective) from what doesn't (the training loop, multi-GPU, checkpointing, logging). Adding a new model means writing an adapter. Adding a new training objective means writing a loss function. The trainer never changes.

## Install

```bash
pip install -e .

# With optional dependencies
pip install -e ".[quantization]"   # bitsandbytes for 8-bit optimizers
pip install -e ".[logging]"        # wandb
pip install -e ".[all]"            # everything
```

## Supported models

| Adapter | Registry name | Models | Objective |
|---|---|---|---|
| `StableDiffusionAdapter` | `sd` | SD 1.x, SD 2.x (base + 768-v) | DDPM, epsilon / v-pred |
| `SDXLAdapter` | `sdxl` | SDXL + finetunes (incl. v-pred) | DDPM, epsilon / v-pred |
| `SD3Adapter` | `sd3` | SD 3 / 3.5 (Medium, Large) | flow, shift 3.0 |
| `FluxAdapter` | `flux` | FLUX.1 dev / schnell | flow, dynamic shift |
| `FluxKontextAdapter` | `flux_kontext` | FLUX.1 Kontext (editing) | flow + reference tokens |
| `ChromaAdapter` | `chroma` | Chroma | flow, T5-only, masked |
| `ZImageAdapter` | `z_image` | Z-Image, Z-Image-Turbo | flow, shift 3.0 |
| `QwenImageAdapter` | `qwen_image` | Qwen-Image | flow |
| `QwenEditAdapter` | `qwen_edit` | Qwen-Image-Edit | flow + control image |
| `QwenImage21Adapter` | `qwen_image_2_1` | Qwen-Image 2.1 (T2I + editing, RGBA) | flow, native port |

Every DiT above takes LoRA on `["to_q", "to_k", "to_v", "to_out.0"]` (the
UNets too). `FlowMatchingLoss` is for the flow models; `EpsilonLoss` (SFT) and
the preference family (DPO / CPO / IPO / KTO / SimPO / ORPO) work with any
adapter, DDPM or flow.

Per-model conventions (VAE scale/shift, how the timestep is fed, flow shift,
text-encoder layers) were cross-checked against
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and the
diffusers pipelines, and pinned by tests against tiny random transformers —
so a LoRA trained here samples correctly in both. Worth knowing:

- **FLUX.1-dev** is trained at `guidance=1.0` by default (kohya / ai-toolkit
  convention); pass `guidance=3.5` for diffusers' reference behavior. The time
  shift defaults to the resolution-dependent `exp(mu)` the sampler uses at
  `resolution=1024` (≈3.16); override with `shift=`.
- **FLUX single-file weights** (e.g. the `flux1-dev.safetensors` sd.cpp loads)
  work via `transformer_path=`; `pretrained_path` still supplies configs, VAE
  and encoders. SD / SDXL accept a single `.safetensors` checkpoint directly.
- **Z-Image** runs reversed time (`1 - sigma`) with a negated output; the
  adapter handles both so the shared velocity target holds.
- **v-prediction** SD 2.x / SDXL finetunes often ship an epsilon scheduler
  config — pass `prediction_type="v_prediction"`.

## Quick start

### Qwen-Image-Edit LoRA (flow matching)

```python
from peft import LoraConfig
from atelier import AtelierTrainer, TrainingConfig
from atelier.adapters import QwenEditAdapter
from atelier.losses import FlowMatchingLoss
from atelier.data import EditingDataset, cache_embeddings

# Load adapter (handles model, VAE, text encoder, scheduler)
adapter = QwenEditAdapter("Qwen/Qwen-Image-Edit")

# Pre-compute embeddings to save VRAM during training
text_emb, target_emb, control_emb = cache_embeddings(
    raw_dataset, adapter, cache_dir="./output/cache",
)
adapter.free_encoders()  # reclaim VRAM

dataset = EditingDataset(
    raw_dataset,
    cached_text_embeddings=text_emb,
    cached_target_embeddings=target_emb,
    cached_control_embeddings=control_emb,
)

trainer = AtelierTrainer(
    adapter=adapter,
    config=TrainingConfig(
        output_dir="./output",
        num_epochs=50,
        batch_size=1,
        learning_rate=1e-4,
        gradient_accumulation_steps=2,
    ),
    loss_fn=FlowMatchingLoss(),
    train_dataset=dataset,
    peft_config=LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ),
)

trainer.train()
trainer.save_model("./my-lora")
```

### Qwen-Image LoRA (text-to-image, flow matching)

Same loss as Qwen-Image-Edit; the adapter differs because the text encoder
is not vision-conditioned and the transformer sees only the noised target
(no control image concat).

```python
from peft import LoraConfig
from atelier import AtelierTrainer, TrainingConfig
from atelier.adapters import QwenImageAdapter
from atelier.losses import FlowMatchingLoss
from atelier.data import EditingDataset, cache_embeddings

adapter = QwenImageAdapter("Qwen/Qwen-Image")

# Dataset only needs (prompt, chosen) — no "rejected" column.
text_emb, target_emb, _ = cache_embeddings(
    raw_dataset, adapter, cache_dir="./output/cache",
)
adapter.free_encoders()

dataset = EditingDataset(
    raw_dataset,
    cached_text_embeddings=text_emb,
    cached_target_embeddings=target_emb,
)

trainer = AtelierTrainer(
    adapter=adapter,
    config=TrainingConfig(output_dir="./output", num_epochs=8, batch_size=1),
    loss_fn=FlowMatchingLoss(),
    train_dataset=dataset,
    peft_config=LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        # DoRA — magnitude + direction decomposition; meaningfully better
        # quality on small aesthetic datasets vs vanilla LoRA, ~5-10%
        # extra step time. Requires peft >= 0.10.
        use_dora=True,
    ),
)
trainer.train()
trainer.save_model("./my-qwen-image-lora")
```

### Qwen-Image 2.1 LoRA (native port — no diffusers support)

diffusers doesn't ship Qwen-Image 2.1, so Atelier carries its own PyTorch port of
the DiT (`atelier/models/qwen_image_2_1.py`), checked against
stable-diffusion.cpp's implementation (`scripts/sdcpp_parity/`; a golden
test keeps CI honest). It loads the released single files directly:

```python
from atelier.adapters import QwenImage21Adapter

adapter = QwenImage21Adapter(
    "Qwen-Image-2.1/diffusion_models/<bf16 file>.safetensors",  # bf16 — not the int8-convrot / fp8 files
    vae_path="Qwen-Image-2.1/vae/qwen_image_2.1_vae_bf16.safetensors",
    text_encoder_path="Qwen/Qwen3-VL-8B-Instruct",                       # stock HF release
)
text_emb, target_emb, control_emb = cache_embeddings(raw_dataset, adapter, cache_dir="./output/cache")
adapter.free_encoders()
adapter.move_transformer_to_device()
# ... EditingDataset + FlowMatchingLoss + LoraConfig(target_modules=["to_q", "to_k", "to_v", "to_out.0"])
```

- **Editing** works like Qwen-Image-Edit: give the dataset a `rejected` (source)
  column. The source is shown to Qwen3-VL and VAE-encoded at the same
  resolution, so image sides must be multiples of 32 (`cache_embeddings`
  rounds to 32 already).
- **Transparency**: the VAE is RGBA. Opaque images train as alpha = 1; RGBA
  images keep their alpha. Prompt with the official
  "This is an RGBA image with transparency. … The image has alpha channel and
  the background is transparent." template for transparent LoRAs.
- **LoRA output** is `lora.safetensors` with `diffusion_model.*` keys +
  `alpha`, which stable-diffusion.cpp and ComfyUI load directly.
- Needs `transformers >= 4.57` (Qwen3-VL) and `torchvision` (its image processor).
- One sample per DiT call (each has its own text/image layout); `batch_size > 1`
  works but loops.

### SDXL DPO (preference optimization)

Same trainer, different adapter and loss function.

```python
from atelier.adapters import SDXLAdapter
from atelier.losses import DiffusionDPOLoss
from atelier.data import GenerationDataset

adapter = SDXLAdapter(
    "stabilityai/stable-diffusion-xl-base-1.0",
    weights="/path/to/model.safetensors",
)
adapter.freeze_layers(strategy="color_blocks", layers="0,1")

dataset = GenerationDataset(
    raw_dataset,
    tokenizer=adapter.tokenizer,
    tokenizer_2=adapter.tokenizer_2,
)

trainer = AtelierTrainer(
    adapter=adapter,
    config=TrainingConfig(
        output_dir="./output",
        num_epochs=10,
        batch_size=1,
        learning_rate=2e-6,
        optimizer="adamw_8bit",
        mixed_precision="fp16",
    ),
    loss_fn=DiffusionDPOLoss(beta=0.4, sft_weight=0.3),
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./my-sdxl")
```

### FLUX.1 / SD3 / Z-Image LoRA (flow matching)

Same shape as Qwen-Image: cache embeddings, free the encoders, train. Swap the
adapter class and the rest stays put.

```python
from peft import LoraConfig
from atelier import AtelierTrainer, TrainingConfig
from atelier.adapters import FluxAdapter  # or SD3Adapter, ZImageAdapter, ChromaAdapter
from atelier.data import EditingDataset, cache_embeddings
from atelier.losses import FlowMatchingLoss

adapter = FluxAdapter("black-forest-labs/FLUX.1-dev")

text_emb, target_emb, _ = cache_embeddings(raw_dataset, adapter, cache_dir="./output/cache")
adapter.free_encoders()
adapter.move_transformer_to_device()

trainer = AtelierTrainer(
    adapter=adapter,
    config=TrainingConfig(output_dir="./output", num_epochs=10, batch_size=1,
                          learning_rate=1e-4, gradient_checkpointing=True),
    loss_fn=FlowMatchingLoss(),
    train_dataset=EditingDataset(raw_dataset, cached_text_embeddings=text_emb,
                                 cached_target_embeddings=target_emb),
    peft_config=LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
                           target_modules=["to_k", "to_q", "to_v", "to_out.0"]),
)
trainer.train()
trainer.save_model("./my-flux-lora")   # loads with FluxPipeline.load_lora_weights
```

For **Kontext** editing, use `FluxKontextAdapter("black-forest-labs/FLUX.1-Kontext-dev")`
with a dataset that has a `rejected` (source image) column — exactly like the
Qwen-Image-Edit flow. The source latents become reference tokens.

### With LoRA

Pass a `peft_config` and Atelier handles the rest.

```python
from peft import LoraConfig

trainer = AtelierTrainer(
    adapter=adapter,
    config=TrainingConfig(...),
    loss_fn=FlowMatchingLoss(),
    train_dataset=dataset,
    peft_config=LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ),
)
```

## Guides

- **[Loss Formulas](docs/loss-formulas.md)** — Math for flow matching and diffusion DPO
- **[Adapters](docs/adapters.md)** — Writing a custom adapter for a new model architecture
- **[Callbacks](docs/callbacks.md)** — Hooking into the training loop
- **[Multi-GPU and DeepSpeed](docs/deepspeed.md)** — Distributed training setup

## YAML config + CLI

For orchestrators (e.g. [Merlina](https://github.com/Schneewolf-Labs/Merlina)) or
when you just don't want to write a Python wrapper per run, train from a YAML config:

```bash
pip install -e ".[yaml]"
python -m atelier.train --config configs/qwen_image_lora_example.yaml

# Override anything on the CLI (JSON-decoded values):
python -m atelier.train --config configs/my.yaml \
    --set training.num_epochs=2 \
    --set training.output_dir=./output-quick \
    --set 'peft.target_modules=["to_q","to_v"]'
```

The YAML schema mirrors the Python API one-for-one — `model.adapter` picks an
adapter from `atelier.registry.ADAPTERS`, `loss.type` picks from `LOSSES`,
`peft` becomes a `LoraConfig`, `training` becomes a `TrainingConfig`, and
`dataset` accepts an HF hub name, a local JSONL, or a `load_from_disk` path.
See `atelier/train.py` for the full schema and `configs/qwen_image_lora_example.yaml`
for a worked example.

## Multi-GPU

No code changes. Configure with `accelerate` and launch:

```bash
accelerate config
accelerate launch --multi_gpu --num_processes 4 train.py
accelerate launch --use_deepspeed --deepspeed_config ds_config.json train.py
```

## Callbacks

Subclass `TrainerCallback` and override the hooks you need:

```python
from atelier import TrainerCallback

class MyCallback(TrainerCallback):
    def on_step_end(self, trainer, step, loss, metrics):
        if should_stop():
            trainer.request_stop()

    def on_log(self, trainer, metrics):
        print(f"Step {trainer.global_step}: {metrics}")

trainer = AtelierTrainer(..., callbacks=[MyCallback()])
```

Available hooks: `on_train_begin`, `on_train_end`, `on_epoch_begin`, `on_epoch_end`, `on_step_end`, `on_log`, `on_evaluate`, `on_save`.

## Configuration

`TrainingConfig` fields with defaults:

| Field | Default | Description |
|---|---|---|
| `output_dir` | `"./output"` | Checkpoints and saved models |
| `num_epochs` | `3` | Number of training epochs |
| `batch_size` | `1` | Per-device batch size |
| `gradient_accumulation_steps` | `1` | Steps before optimizer update |
| `learning_rate` | `1e-4` | Peak learning rate |
| `weight_decay` | `0.01` | L2 regularization |
| `warmup_ratio` | `0.1` | Fraction of steps for LR warmup |
| `warmup_steps` | `0` | Overrides `warmup_ratio` if > 0 |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `mixed_precision` | `"bf16"` | `"no"`, `"fp16"`, or `"bf16"` |
| `gradient_checkpointing` | `True` | Trade compute for memory |
| `optimizer` | `"adamw"` | See supported optimizers below |
| `lr_scheduler` | `"cosine"` | `"linear"`, `"cosine"`, `"constant"`, `"constant_with_warmup"` |
| `logging_steps` | `10` | Log metrics every N steps |
| `eval_steps` | `None` | Evaluate every N steps |
| `save_steps` | `None` | Checkpoint every N steps |
| `save_total_limit` | `2` | Max checkpoints to keep |
| `save_on_epoch_end` | `True` | Checkpoint after each epoch |
| `resume_from_checkpoint` | `None` | Path to resume from |
| `seed` | `42` | Random seed |
| `log_with` | `None` | `"wandb"` for W&B tracking |

**Supported optimizers:** `adamw`, `adamw_8bit`, `paged_adamw_8bit`, `adafactor`, `sgd`

## Architecture

```
atelier/
├── trainer.py           # AtelierTrainer — the training loop
├── config.py            # TrainingConfig dataclass
├── callbacks.py         # TrainerCallback base class
├── adapters/
│   ├── base.py          # ModelAdapter protocol + shared helpers
│   ├── ddpm.py          # DDPMAdapter base (epsilon / v-pred)
│   ├── flow.py          # FlowMatchAdapter + DiffusersFlowAdapter bases (shift, sampling, loading)
│   ├── sd.py            # SD 1.x / 2.x (UNet + CLIP + DDPM)
│   ├── sdxl.py          # SDXL (UNet + dual CLIP + DDPM)
│   ├── sd3.py           # SD 3 / 3.5 (MMDiT + CLIP-L/G + T5 + flow)
│   ├── flux.py          # FLUX.1 dev / schnell / Kontext, Chroma
│   ├── z_image.py       # Z-Image (single-stream DiT + Qwen3 + flow)
│   ├── qwen_edit.py     # Qwen-Image-Edit (DiT + flow matching, image-conditioned)
│   ├── qwen_image.py    # Qwen-Image (DiT + flow matching, text-to-image)
│   └── qwen_image_2_1.py # Qwen-Image 2.1 (native DiT port + Qwen3-VL + RGBA VAE)
├── models/
│   └── qwen_image_2_1.py # PyTorch ports of architectures diffusers doesn't ship
├── losses/
│   ├── flow_matching.py # Flow matching MSE
│   └── diffusion_dpo.py # DPO + SFT regularization
└── data/
    ├── editing.py       # Paired image editing dataset
    ├── generation.py    # Text-to-image dataset
    └── cache.py         # Embedding pre-computation
```

### How it fits together

The **adapter** encapsulates everything that varies per model architecture — loading, encoding, the forward pass, and saving. In Grimoire (LLM training), every model has the same forward signature (`model(input_ids)` → logits). In diffusion training, forward passes vary wildly: Qwen-Image-Edit needs latent packing, control image concatenation, and RoPE shapes; SDXL needs dual CLIP conditioning and time embeddings. The adapter hides this.

The **loss function** orchestrates the training objective — sampling noise and timesteps, calling the adapter's forward pass, and computing the loss. Flow matching predicts the velocity field; DPO compares noise predictions for chosen vs rejected images.

The **trainer** owns the loop — optimizer, gradient accumulation, checkpointing, logging. It calls `loss_fn(adapter, model, batch)` and never needs to know what model architecture or training objective is being used.

### Loss function interface

```python
class MyLoss:
    def __call__(self, adapter, model, batch, training=True):
        # Use adapter for noise sampling, forward pass, target computation
        return loss, metrics_dict

    def create_collator(self):
        return MyCollator()
```

### Adapter interface

```python
class MyAdapter(ModelAdapter):
    def model(self):            ...  # The trainable model
    def encode_images(self):    ...  # VAE encode
    def encode_text(self):      ...  # Text encode
    def sample_timesteps(self): ...  # Timestep sampling
    def add_noise(self):        ...  # Create noisy input
    def compute_target(self):   ...  # What model should predict (noise, latents, sigmas, timesteps=)
    def forward(self):          ...  # Architecture-specific forward
    def save_lora(self):        ...  # Save LoRA weights
    def save_model(self):       ...  # Save full model
```

## License

MIT
