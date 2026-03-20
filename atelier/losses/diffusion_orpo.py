"""Diffusion ORPO — Odds Ratio Preference Optimization for diffusion models.

Translates Grimoire's ORPOLoss: SFT on chosen + odds ratio preference term.
No reference model needed.

Key mapping:
- LLM: NLL(chosen) + beta * -log(sigmoid(log_odds_ratio))
- Diffusion: MSE(chosen) + beta * -log(sigmoid(log_odds_ratio))
  where odds are based on denoising quality (exp(-MSE) as proxy for "probability")
"""

import torch
import torch.nn.functional as F

from ..data.generation import GenerationCollator
from .utils import get_paired_denoising_losses


def _log1mexp(x):
    """Numerically stable log(1 - exp(x)) for x <= 0."""
    return torch.where(
        x > -0.6931,  # ln(2)
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


class DiffusionORPOLoss:
    """ORPO (Odds Ratio Preference Optimization) loss for diffusion models.

    Combines SFT loss on chosen with an odds ratio preference term.
    No reference model needed.

    Loss = MSE(chosen) + beta * -mean(log(sigmoid(log_odds_ratio)))

    The odds ratio compares denoising quality: we use -MSE as a proxy
    for log-probability, so lower MSE = higher implicit "probability".
    """

    def __init__(self, beta=0.1, timestep_bias_range=None):
        self.beta = beta
        self.timestep_bias_range = timestep_bias_range

    def __call__(self, adapter, model, batch, training=True):
        chosen_per, rejected_per, sft_loss, _ = get_paired_denoising_losses(
            adapter, model, batch, timestep_bias=self.timestep_bias_range,
        )

        # Use -MSE as implicit log-probability (lower MSE = higher "prob")
        chosen_logps = -chosen_per
        rejected_logps = -rejected_per

        # Odds ratio: log(odds_chosen / odds_rejected)
        # log_odds = (logp_c - logp_r) - (log(1-exp(logp_c)) - log(1-exp(logp_r)))
        log_odds = (chosen_logps - rejected_logps) - (
            _log1mexp(chosen_logps) - _log1mexp(rejected_logps)
        )
        or_loss = -self.beta * F.logsigmoid(log_odds).mean()

        total_loss = sft_loss + or_loss

        metrics = {
            "sft_loss": sft_loss.item(),
            "or_loss": or_loss.item(),
            "log_odds_ratio": log_odds.detach().mean().item(),
            "reward_accuracy": (chosen_per < rejected_per).float().mean().item(),
        }

        return total_loss, metrics

    def create_collator(self):
        return GenerationCollator()
