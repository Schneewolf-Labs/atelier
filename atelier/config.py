from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training configuration for AtelierTrainer."""

    # Output
    output_dir: str = "./output"

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # overrides warmup_ratio if > 0
    max_grad_norm: float = 1.0

    # Precision and optimization
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = True
    optimizer: str = "adamw"  # "adamw", "adamw_8bit", "paged_adamw_8bit", "adafactor", "sgd"
    lr_scheduler: str = "cosine"  # "linear", "cosine", "constant", "constant_with_warmup"

    # Data loading
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # Logging
    logging_steps: int = 10
    log_with: Optional[str] = None  # "wandb" or None
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None

    # Evaluation
    eval_steps: Optional[int] = None

    # Checkpointing
    save_steps: Optional[int] = None
    save_total_limit: int = 2
    save_on_epoch_end: bool = True
    resume_from_checkpoint: Optional[str] = None

    # Reproducibility
    seed: int = 42


@dataclass
class FlowGRPOConfig(TrainingConfig):
    """Configuration for FlowGRPOTrainer (Flow-GRPO / DanceGRPO online RL).

    Additive over TrainingConfig — Merlina forwards the shared fields and sets the
    rollout/optimization knobs below. The loss-side knobs (beta, epsilon, loss_type,
    scale_rewards) live on FlowGRPOLoss, not here.
    """

    # Rollout
    num_generations: int = 8          # G — images sampled per prompt
    num_inference_steps: int = 16     # denoising steps per sample (kept low — cost is brutal)
    guidance_scale: float = 4.5
    sde_noise_level: float = 0.7      # eta for SDE-ification; 0 == deterministic (no logprobs)
    image_resolution: int = 512       # smaller than SFT default — rollout cost scales hard

    # Optimization
    num_iterations: int = 1           # PPO inner epochs (mu)
    timestep_fraction: float = 1.0    # DDPO trick: backprop a random fraction of steps to fit memory
