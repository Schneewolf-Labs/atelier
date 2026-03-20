"""Diffusion DPO — Direct Preference Optimization for diffusion models.

Translates Grimoire's DPOLoss to the diffusion setting:
- LLM: log_prob(chosen) - log_prob(rejected) → preference signal
- Diffusion: MSE(rejected) - MSE(chosen) → preference signal
  (lower MSE = better denoising = model "prefers" this image)
"""

import numpy as np
import torch
import torch.nn.functional as F

from ..data.generation import GenerationCollator
from .utils import get_paired_denoising_losses


class DiffusionDPOLoss:
    """DPO (Direct Preference Optimization) loss for diffusion models.

    Given (prompt, chosen_image, rejected_image) triples:
    1. Add shared noise at a shared timestep to both latents
    2. Predict target with the model for both
    3. Apply DPO: loss = -log_sigmoid(beta * (loss_rejected - loss_chosen))
    4. Add SFT regularization on chosen examples

    Total loss = dpo_loss + sft_weight * sft_loss

    Works with any adapter — uses adapter protocol for noise addition,
    forward pass, and target computation.
    """

    def __init__(
        self,
        beta=0.1,
        sft_weight=0.1,
        logit_clamp=5.0,
        beta_schedule="constant",
        beta_warmup_steps=100,
        timestep_bias_range=(0.3, 0.8),
    ):
        self.beta = beta
        self.sft_weight = sft_weight
        self.logit_clamp = logit_clamp
        self.beta_schedule = beta_schedule
        self.beta_warmup_steps = beta_warmup_steps
        self.timestep_bias_range = timestep_bias_range

        # Set by trainer
        self.global_step = 0
        self.total_steps = 1

    def __call__(self, adapter, model, batch, training=True):
        chosen_per, rejected_per, sft_loss, _ = get_paired_denoising_losses(
            adapter, model, batch, timestep_bias=self.timestep_bias_range,
        )

        # DPO log-ratio with clamping
        pi_logratios = rejected_per - chosen_per
        pi_logratios = torch.clamp(pi_logratios, min=-self.logit_clamp, max=self.logit_clamp)

        current_beta = self._get_beta()
        dpo_loss = -F.logsigmoid(current_beta * pi_logratios).mean()

        total_loss = dpo_loss + self.sft_weight * sft_loss

        metrics = {
            "dpo_loss": dpo_loss.item() if not torch.isnan(dpo_loss) else 0.0,
            "sft_loss": sft_loss.item() if not torch.isnan(sft_loss) else 0.0,
            "total_loss": total_loss.item() if not torch.isnan(total_loss) else 0.0,
            "reward_accuracy": (rejected_per > chosen_per).float().mean().item(),
        }

        return total_loss, metrics

    def _get_beta(self):
        """Compute beta with optional scheduling."""
        step = self.global_step

        if self.beta_schedule == "linear":
            if step < self.beta_warmup_steps:
                return self.beta * (step / self.beta_warmup_steps)
            tail_start = int(0.7 * self.total_steps)
            if step >= tail_start:
                frac = (step - tail_start) / max(1, self.total_steps - tail_start)
                return self.beta + (0.3 - self.beta) * frac
            return self.beta

        elif self.beta_schedule == "cosine":
            progress = step / self.total_steps
            return self.beta * 0.5 * (1 + np.cos(np.pi * (progress - 1)))

        return self.beta

    def create_collator(self):
        return GenerationCollator()
