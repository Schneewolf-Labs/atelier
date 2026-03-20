"""Diffusion KTO — Kahneman-Tversky Optimization for diffusion models.

Translates Grimoire's KTOLoss: works with unpaired binary feedback.
Each image is labeled good (desirable) or bad (undesirable) independently.

Key mapping:
- LLM: log_ratio = avg_logp_policy - avg_logp_ref
- Diffusion: log_ratio = ref_mse - policy_mse (positive = policy is better)
"""

import torch
import torch.nn.functional as F

from ..data.generation import GenerationCollator
from .utils import get_single_denoising_loss


class DiffusionKTOLoss:
    """KTO (Kahneman-Tversky Optimization) loss for diffusion models.

    Works with unpaired binary feedback: each sample is an image with
    a boolean quality label (good/bad). No chosen/rejected pairs needed.

    Requires a reference model (frozen copy or base model via disable_adapter).

    Desirable loss:   lambda_d * (1 - sigmoid(beta * (log_ratio - KL_ref)))
    Undesirable loss: lambda_u * (1 - sigmoid(beta * (KL_ref - log_ratio)))

    where log_ratio = ref_mse - policy_mse (positive = policy improved).
    """

    def __init__(self, beta=0.1, lambda_d=1.0, lambda_u=1.0, ref_model=None):
        self.beta = beta
        self.lambda_d = lambda_d
        self.lambda_u = lambda_u
        self.ref_model = ref_model

    def __call__(self, adapter, model, batch, training=True):
        device = next(model.parameters()).device
        kto_label = batch["kto_label"].to(device)  # bool: True=desirable

        # Policy denoising loss (per sample)
        policy_per, _, forward_batch = get_single_denoising_loss(adapter, model, batch)

        # Reference denoising loss (per sample)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_per, _, _ = get_single_denoising_loss(adapter, self.ref_model, batch)
            elif hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    ref_per, _, _ = get_single_denoising_loss(adapter, model, batch)
            else:
                raise ValueError(
                    "DiffusionKTOLoss requires either a ref_model or a PEFT model with disable_adapter()"
                )

        # Log ratio: positive means policy is better than reference
        # (ref has higher MSE = worse denoising, policy has lower MSE = better)
        log_ratio = (ref_per - policy_per).detach()
        log_ratio.requires_grad_(False)

        # Recompute policy_per with gradients for the loss
        # (the one from get_single_denoising_loss already has gradients)

        # KL estimate from batch
        kl_ref = log_ratio.mean().clamp(min=0)

        # Compute per-sample implicit log ratios with gradient
        policy_ratio = ref_per.detach() - policy_per

        # Split by label
        desirable = kto_label
        undesirable = ~kto_label

        loss = torch.tensor(0.0, device=device)
        n_terms = 0

        if desirable.any():
            d_loss = self.lambda_d * (1 - F.sigmoid(self.beta * (policy_ratio[desirable] - kl_ref)))
            loss = loss + d_loss.mean()
            n_terms += 1

        if undesirable.any():
            u_loss = self.lambda_u * (1 - F.sigmoid(self.beta * (kl_ref - policy_ratio[undesirable])))
            loss = loss + u_loss.mean()
            n_terms += 1

        if n_terms > 1:
            loss = loss / n_terms

        # Metrics
        rewards = self.beta * log_ratio
        d_rewards = rewards[desirable] if desirable.any() else torch.zeros(1, device=device)
        u_rewards = rewards[undesirable] if undesirable.any() else torch.zeros(1, device=device)

        metrics = {
            "desirable_rewards": d_rewards.mean().item(),
            "undesirable_rewards": u_rewards.mean().item(),
            "reward_margin": (d_rewards.mean() - u_rewards.mean()).item(),
            "reward_accuracy": (
                (d_rewards > 0).float().sum() + (u_rewards < 0).float().sum()
            ).item() / max(len(rewards), 1),
            "kl_ref": kl_ref.item(),
        }

        return loss, metrics

    def create_collator(self):
        return GenerationCollator()
