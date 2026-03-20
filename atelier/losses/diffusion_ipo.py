"""Diffusion IPO — Identity Preference Optimization for diffusion models.

Translates Grimoire's IPOLoss: squared-loss variant of DPO.
Prevents overfitting on noisy preference data.

Key mapping:
- LLM: ((log(pi/ref)(chosen) - log(pi/ref)(rejected) - 1/(2*beta))^2)
- Diffusion: ((policy_margin - ref_margin - 1/(2*beta))^2)
  where margin = MSE(rejected) - MSE(chosen) for each model
"""

import torch

from ..data.generation import GenerationCollator
from .utils import get_paired_denoising_losses


class DiffusionIPOLoss:
    """IPO (Identity Preference Optimization) loss for diffusion models.

    Like DPO but replaces log-sigmoid with squared loss for robustness
    to noisy preference labels. Requires a reference model.

    Loss = mean((policy_margin - ref_margin - 1/(2*beta))^2)

    The 1/(2*beta) term acts as a target margin that the policy should
    exceed over the reference.
    """

    def __init__(self, beta=0.1, ref_model=None, timestep_bias_range=None):
        self.beta = beta
        self.ref_model = ref_model
        self.timestep_bias_range = timestep_bias_range

    def __call__(self, adapter, model, batch, training=True):
        # Policy denoising losses
        chosen_per, rejected_per, _, _ = get_paired_denoising_losses(
            adapter, model, batch, timestep_bias=self.timestep_bias_range,
        )

        # Reference denoising losses
        with torch.no_grad():
            if self.ref_model is not None:
                ref_chosen, ref_rejected, _, _ = get_paired_denoising_losses(
                    adapter, self.ref_model, batch, timestep_bias=self.timestep_bias_range,
                )
            elif hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    ref_chosen, ref_rejected, _, _ = get_paired_denoising_losses(
                        adapter, model, batch, timestep_bias=self.timestep_bias_range,
                    )
            else:
                raise ValueError(
                    "DiffusionIPOLoss requires either a ref_model or a PEFT model with disable_adapter()"
                )

        # Policy and reference preference margins
        # (positive = model is better at denoising chosen than rejected)
        pi_margin = rejected_per - chosen_per
        ref_margin = ref_rejected - ref_chosen

        # IPO: squared loss with target margin
        logits_diff = pi_margin - ref_margin
        loss = ((logits_diff - 1.0 / (2.0 * self.beta)) ** 2).mean()

        # Implicit rewards
        chosen_rewards = self.beta * (ref_chosen - chosen_per).detach()
        rejected_rewards = self.beta * (ref_rejected - rejected_per).detach()

        metrics = {
            "loss": loss.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        }

        return loss, metrics

    def create_collator(self):
        return GenerationCollator()
