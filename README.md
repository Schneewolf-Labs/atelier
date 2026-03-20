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
│   ├── base.py          # ModelAdapter protocol
│   ├── qwen_edit.py     # Qwen-Image-Edit (DiT + flow matching)
│   └── sdxl.py          # SDXL (UNet + DDPM)
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
    def compute_target(self):   ...  # What model should predict
    def forward(self):          ...  # Architecture-specific forward
    def save_lora(self):        ...  # Save LoRA weights
    def save_model(self):       ...  # Save full model
```

## License

MIT
