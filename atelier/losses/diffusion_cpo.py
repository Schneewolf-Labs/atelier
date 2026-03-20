"""Diffusion CPO — Contrastive Preference Optimization for diffusion models.

Translates Grimoire's CPOLoss: SFT on chosen + contrastive preference term.
Reference-free, like ORPO but with a simpler contrastive term.

Key mapping:
- LLM: NLL(chosen) + beta * -log(sigmoid(beta * (logp_chosen - logp_rejected)))
- Diffusion: MSE(chosen) + beta * -log(sigmoid(beta * (loss_rejected - loss_chosen)))
"""

import torch.nn.functional as F

from ..data.generation import GenerationCollator
from .utils import get_paired_denoising_losses


class DiffusionCPOLoss:
    """CPO (Contrastive Preference Optimization) loss for diffusion models.

    Combines SFT on chosen with a contrastive preference loss.
    Reference-free. Theoretically cleaner than ORPO's odds ratio.

    Loss = MSE(chosen) + beta * L_preference
    L_preference = -mean((1-eps)*log(sigmoid(x)) + eps*log(sigmoid(-x)))
    x = beta * (loss_rejected - loss_chosen)
    """

    def __init__(self, beta=0.1, label_smoothing=0.0, timestep_bias_range=None):
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.timestep_bias_range = timestep_bias_range

    def __call__(self, adapter, model, batch, training=True):
        chosen_per, rejected_per, sft_loss, _ = get_paired_denoising_losses(
            adapter, model, batch, timestep_bias=self.timestep_bias_range,
        )

        # Contrastive preference: model should denoise chosen better
        logits_diff = rejected_per - chosen_per
        scaled_diff = self.beta * logits_diff
        eps = self.label_smoothing
        preference_loss = -(
            (1 - eps) * F.logsigmoid(scaled_diff)
            + eps * F.logsigmoid(-scaled_diff)
        ).mean()

        total_loss = sft_loss + self.beta * preference_loss

        metrics = {
            "sft_loss": sft_loss.item(),
            "preference_loss": preference_loss.item(),
            "reward_margin": logits_diff.detach().mean().item(),
            "reward_accuracy": (rejected_per > chosen_per).float().mean().item(),
        }

        return total_loss, metrics

    def create_collator(self):
        return GenerationCollator()
