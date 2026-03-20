# Multi-GPU, DeepSpeed, and FSDP

Atelier uses `accelerate` for distributed training. No code changes needed — configure once, launch with the same script.

## Quick start

```bash
# Interactive setup (pick DDP, DeepSpeed, or FSDP)
accelerate config

# Launch training
accelerate launch --multi_gpu --num_processes 4 train.py

# Or with DeepSpeed
accelerate launch --use_deepspeed --deepspeed_config ds_config.json train.py
```

## DDP (Distributed Data Parallel)

The simplest option. Replicates the model on each GPU.

```bash
accelerate launch --multi_gpu --num_processes 4 train.py
```

Best when: The full model + optimizer fits in one GPU's memory.

## DeepSpeed ZeRO

Shards optimizer state (Stage 1), gradients (Stage 2), or parameters (Stage 3) across GPUs.

### ZeRO Stage 2 (recommended for most diffusion training)

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### ZeRO Stage 3 (for very large models)

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

Launch:

```bash
accelerate launch --use_deepspeed --deepspeed_config ds_z2.json train.py
```

## FSDP (Fully Sharded Data Parallel)

PyTorch-native alternative to DeepSpeed. Configure via `accelerate config`.

```bash
accelerate config
# Select FSDP, choose sharding strategy, wrapping policy
```

## Memory tips

- **Gradient checkpointing** (`config.gradient_checkpointing = True`): Trades ~30% compute for ~50% memory savings. Enabled by default.
- **8-bit optimizers** (`config.optimizer = "adamw_8bit"`): Halves optimizer memory with minimal quality impact.
- **Pre-compute embeddings**: Use `cache_embeddings()` to encode all images/text upfront, then free the VAE and text encoder before training. This can save 10+ GB of VRAM.
- **Gradient accumulation**: Use `config.gradient_accumulation_steps` to simulate larger batches without the memory cost.
- **Mixed precision**: `bf16` is recommended for most hardware (Ampere+). Use `fp16` for older GPUs.

## Example: 4-GPU training with DeepSpeed

```python
# train.py — no multi-GPU code needed
from atelier import AtelierTrainer, TrainingConfig

trainer = AtelierTrainer(
    adapter=adapter,
    config=TrainingConfig(
        output_dir="./output",
        num_epochs=50,
        batch_size=1,
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
    ),
    loss_fn=loss_fn,
    train_dataset=dataset,
)
trainer.train()
```

```bash
# Launch on 4 GPUs with ZeRO-2
accelerate launch \
    --use_deepspeed \
    --deepspeed_config ds_z2.json \
    --num_processes 4 \
    train.py
```
