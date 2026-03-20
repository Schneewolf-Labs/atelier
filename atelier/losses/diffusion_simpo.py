"""Diffusion SimPO — Simple Preference Optimization for diffusion models.

Translates Grimoire's SimPOLoss: reference-free with a reward margin gamma.
- LLM: -log_sigmoid(beta * (avg_logp_chosen - avg_logp_rejected - gamma))
- Diffusion: -log_sigmoid(beta * (loss_rejected - loss_chosen - gamma))
"""

import torch.nn.functional as F

from ..data.generation import GenerationCollator
from .utils import get_paired_denoising_losses


class DiffusionSimPOLoss:
    """SimPO (Simple Preference Optimization) loss for diffusion models.

    Reference-free preference optimization with a target reward margin.
    No reference model needed — simpler than DPO.

    Loss = -mean(log(sigmoid(beta * (loss_rejected - loss_chosen - gamma))))

    gamma enforces a minimum gap between chosen and rejected denoising quality.
    """

    def __init__(self, beta=2.0, gamma=0.5, timestep_bias_range=None):
        self.beta = beta
        self.gamma = gamma
        self.timestep_bias_range = timestep_bias_range

    def __call__(self, adapter, model, batch, training=True):
        chosen_per, rejected_per, _, _ = get_paired_denoising_losses(
            adapter, model, batch, timestep_bias=self.timestep_bias_range,
        )

        # SimPO: preference margin with gamma
        logits_diff = rejected_per - chosen_per - self.gamma
        loss = -F.logsigmoid(self.beta * logits_diff).mean()

        # Implicit rewards (no reference model)
        chosen_rewards = (-self.beta * chosen_per).detach()
        rejected_rewards = (-self.beta * rejected_per).detach()

        metrics = {
            "loss": loss.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "reward_accuracy": (rejected_per > chosen_per).float().mean().item(),
        }

        return loss, metrics

    def create_collator(self):
        return GenerationCollator()
