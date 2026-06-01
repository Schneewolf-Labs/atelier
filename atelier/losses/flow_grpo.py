"""Flow-GRPO / DanceGRPO policy-gradient loss.

Pure tensor->scalar loss. No adapter, no sampling, no I/O — it operates on
per-denoising-step logprobs and per-image advantages produced upstream by
FlowGRPOTrainer. This is the diffusion analog of Grimoire's GRPOLoss: a
denoising STEP is the analog of a TOKEN, and a fully sampled image is the
analog of a completion.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class FlowGRPOLossOutput:
    """Output of FlowGRPOLoss.__call__ — a differentiable scalar + detached metrics."""

    loss: torch.Tensor
    kl: float
    clip_frac: float
    ratio_mean: float


class FlowGRPOLoss:
    """Policy-gradient loss over an SDE denoising trajectory (Flow-GRPO / DanceGRPO).

    Advantage is per-image (group-normalized) and broadcast across that image's
    trajectory steps. The per-step PPO ratio is clipped exactly like token-level
    GRPO; an optional reference-KL penalty (k3 estimator) pulls the policy back
    toward the reference (LoRA-disabled) model.

    Args:
        beta:          KL coefficient toward the reference policy. 0.0 disables KL
                       (the trainer then skips reference logprobs entirely).
        epsilon:       PPO clip range for the per-step importance ratio.
        loss_type:     "grpo"    -> normalize each trajectory by its length.
                       "dr_grpo" -> divide by a constant (N * S); no length/std norm.
        scale_rewards: True (GRPO) divide group-centered rewards by the group std;
                       False (Dr.GRPO) mean-center only. Consumed by
                       ``compute_advantages`` so the grouping policy lives with the loss.
    """

    def __init__(
        self,
        beta: float = 0.0,
        epsilon: float = 0.2,
        loss_type: Literal["grpo", "dr_grpo"] = "grpo",
        scale_rewards: bool = True,
    ) -> None:
        if loss_type not in ("grpo", "dr_grpo"):
            raise ValueError(f"Unknown loss_type: {loss_type!r} (expected 'grpo' or 'dr_grpo')")
        self.beta = beta
        self.epsilon = epsilon
        self.loss_type = loss_type
        self.scale_rewards = scale_rewards

    def compute_advantages(self, rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
        """Group-normalize raw per-image rewards into advantages.

        Args:
            rewards: (N,) finite rewards in group-major order
                     [p0g0, p0g1, ..., p0g{G-1}, p1g0, ...].
            num_generations: G — images sampled per prompt. N must be divisible by G.

        Returns:
            (N,) advantages. Mean-centered within each prompt group; divided by the
            group std when ``scale_rewards`` is True.
        """
        rewards = rewards.float()
        if rewards.numel() % num_generations != 0:
            raise ValueError(
                f"rewards length {rewards.numel()} not divisible by num_generations {num_generations}"
            )
        grouped = rewards.view(-1, num_generations)
        advantages = grouped - grouped.mean(dim=1, keepdim=True)
        if self.scale_rewards:
            advantages = advantages / (grouped.std(dim=1, keepdim=True) + 1e-8)
        return advantages.reshape(-1)

    def __call__(
        self,
        *,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor],
        advantages: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> FlowGRPOLossOutput:
        mask = step_mask.to(logprobs.dtype)
        adv = advantages.to(logprobs.dtype).unsqueeze(1)  # (N, 1) -> broadcast over steps

        # Per-step importance ratio under the sampling policy (PPO surrogate).
        log_ratio = logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
        per_step_loss = -torch.min(unclipped, clipped)  # (N, S)

        # Optional reference KL (k3 estimator: non-negative, low-variance).
        if ref_logprobs is not None and self.beta != 0.0:
            diff = ref_logprobs - logprobs
            per_step_kl = torch.exp(diff) - diff - 1.0
            per_step_loss = per_step_loss + self.beta * per_step_kl
        else:
            per_step_kl = torch.zeros_like(per_step_loss)

        if self.loss_type == "grpo":
            seq_len = mask.sum(dim=1).clamp(min=1.0)
            loss = ((per_step_loss * mask).sum(dim=1) / seq_len).mean()
        else:  # dr_grpo — constant normalizer, bias-corrected
            denom = mask.shape[0] * mask.shape[1]
            loss = (per_step_loss * mask).sum() / denom

        # Detached diagnostics.
        with torch.no_grad():
            denom = mask.sum().clamp(min=1.0)
            kl = (per_step_kl * mask).sum() / denom
            is_clipped = (torch.abs(ratio - 1.0) > self.epsilon).to(mask.dtype)
            clip_frac = (is_clipped * mask).sum() / denom
            ratio_mean = (ratio * mask).sum() / denom

        return FlowGRPOLossOutput(
            loss=loss,
            kl=kl.item(),
            clip_frac=clip_frac.item(),
            ratio_mean=ratio_mean.item(),
        )
