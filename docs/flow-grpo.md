# Flow-GRPO (online RL)

Flow-GRPO / DanceGRPO turns Atelier from an offline regression trainer into an
**online reinforcement-learning** trainer. Instead of fitting cached latents to fixed
targets, every step **samples** images with the current policy, **scores** them with a
reward function, and does **policy gradient** over the denoising trajectory.

This inverts the SFT path:

| | SFT / DPO (AtelierTrainer) | Flow-GRPO (FlowGRPOTrainer) |
|---|---|---|
| Targets | Cached target latents | None — images are sampled online |
| VAE decoder | Freed after caching | **Stays resident** (renders images for the reward) |
| Text encoder | Freed after caching | Can still be freed (prompts cached up front) |
| Sampler | n/a | SDE sampler returning per-step logprobs |
| Signal | MSE to target | Reward → group-normalized advantage |

## The four pieces

```python
from atelier import FlowGRPOTrainer, FlowGRPOConfig
from atelier.losses import FlowGRPOLoss
from atelier.data import PromptDataset
from atelier.rewards import ImageRewardFn  # Protocol only

adapter = QwenImageAdapter("Qwen/Qwen-Image")          # must implement SupportsSDESampling
dataset = PromptDataset(raw_dataset, adapter, cache_dir="./cache", cache_embeddings=True)

def reward_fn(prompts, images, **columns):             # your ImageRewardFn
    return [score(p, img) for p, img in zip(prompts, images)]

trainer = FlowGRPOTrainer(
    adapter=adapter,
    config=FlowGRPOConfig(output_dir="./out", num_generations=8, num_inference_steps=16),
    loss_fn=FlowGRPOLoss(beta=0.0, epsilon=0.2, loss_type="grpo", scale_rewards=True),
    reward_fn=reward_fn,
    train_dataset=dataset,
    peft_config=LoraConfig(r=32, target_modules=["to_q", "to_k", "to_v", "to_out.0"]),
)
trainer.train()
trainer.save_model("./my-grpo-lora")
```

### 1. The reward — the membrane (`ImageRewardFn`)

Atelier defines the **Protocol only**. A reward is one synchronous callable that maps
decoded images → finite floats. Aesthetic predictors, CLIP/SigLIP alignment, HPSv2 /
ImageReward / PickScore preference models, VLM-as-judge, sandboxed user code, and any
weighted composition of them all live **in Merlina**, hidden behind this one callable.
The trainer blocks on it, sanitizes the output (NaN/inf → 0.0), and never learns its
internals.

Images are passed in **group-major** order: `[p0g0, p0g1, …, p0g{G-1}, p1g0, …]`.

### 2. The loss — pure tensors (`FlowGRPOLoss`)

A pure `logprobs → scalar` function (no adapter, no I/O), sitting beside
`FlowMatchingLoss`. See [loss formulas](loss-formulas.md#flow-grpo-flowgrpoloss) for the
math. It also owns advantage normalization via `compute_advantages(rewards, G)`.

### 3. The sampler — the new adapter capability (`SupportsSDESampling`)

Flow matching is a deterministic ODE; to get per-step logprobs the trajectory is sampled
as an **SDE** (`sde_noise_level = eta > 0`) so each step is a Gaussian transition with a
tractable logprob. Each GRPO-trainable adapter implements:

- `sample_with_logprobs(...) -> SDERollout` — roll out the chain, return the trajectory,
  per-step logprobs, a step mask, and decoded PIL images for the reward.
- `recompute_logprobs(rollout, with_grad=...)` — re-score a stored trajectory under the
  current (grad-enabled) and reference (LoRA-disabled) policies for the ratio + KL.

### 4. The trainer — the online loop (`FlowGRPOTrainer`)

Per optimization step:

1. `sample_with_logprobs` → SDE rollout (trajectory + logprobs + images).
2. `reward_fn(prompts, images, **columns)` → floats (one blocking sync call).
3. group-normalize rewards → advantages.
4. `recompute_logprobs(with_grad=True)` (+ reference via disabled adapter if `beta>0`);
   `timestep_fraction` backprops only a random subset of steps to fit memory.
5. `loss_fn(...)` → backward → step.
6. emit `reward/mean`, `reward/std`, `kl`, `clip_frac`, `ratio_mean`, plus a few
   `(prompt, image, reward)` samples (`trainer.last_samples`) for the wandb table /
   websocket preview.

The lifecycle, callbacks, and checkpointing match `AtelierTrainer`, so
`MidTrainingSampleCallback` / `WebSocketCallback` keep working.

## Cost & knobs

Rollout cost is brutal — every step samples `batch_size * G` full trajectories. Keep
`num_inference_steps` low (≈16), `image_resolution` small (≈512), and use
`timestep_fraction < 1.0` to bound backward-pass memory. `num_iterations > 1` enables PPO
inner epochs (reuse a rollout for several updates).
