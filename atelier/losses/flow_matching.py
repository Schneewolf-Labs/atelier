import torch

from ..data.editing import EditingCollator


class FlowMatchingLoss:
    """Flow matching MSE loss for diffusion training.

    Predicts the velocity field ``target = noise - latents`` at a sampled
    timestep and minimizes the (optionally weighted) MSE against the model's
    prediction.

    Built for the Qwen-Image / Qwen-Image-Edit DiTs, whose video VAE emits
    5-D latents ``[B, C, 1, H, W]``; those get permuted to the
    ``[B, 1, C, H, W]`` layout the transformer expects. 4-D latents (the
    SD3 / FLUX layout) are passed through unchanged, so the velocity-field
    objective transfers to those models once an adapter + collator provide
    the matching batch shape — no fork of the loss required.

    The collator is pluggable: it defaults to :class:`EditingCollator` (which
    also handles the no-control text-to-image case) but can be overridden for
    datasets with a different batch shape.
    """

    def __init__(self, weighting_scheme="none", collator=None):
        self.weighting_scheme = weighting_scheme
        self._collator = collator

    def __call__(self, adapter, model, batch, training=True):
        target_latents = batch["target_latents"].to(model.device, model.dtype)

        # Video-VAE latents arrive as [B, C, 1, H, W]; the model wants
        # [B, 1, C, H, W]. 4-D latents (SD3/FLUX layout) pass through as-is.
        is_video = target_latents.ndim == 5
        if is_video:
            target_latents = target_latents.permute(0, 2, 1, 3, 4)

        # Normalize latents if adapter supports it
        if hasattr(adapter, "normalize_latents"):
            target_latents = adapter.normalize_latents(target_latents)
            if "control_latents" in batch:
                batch = dict(batch)  # shallow copy to avoid mutating original
                control = batch["control_latents"].to(model.device, model.dtype)
                if is_video:
                    control = control.permute(0, 2, 1, 3, 4)
                batch["control_latents"] = adapter.normalize_latents(control)

        bsz = target_latents.shape[0]
        noise = torch.randn_like(target_latents)

        # Sample timesteps and sigmas
        timesteps, sigmas = adapter.sample_timesteps(bsz, target_latents.device)

        # Create noisy input
        noisy_latents = adapter.add_noise(target_latents, noise, timesteps, sigmas)

        # Forward pass through model
        model_pred = adapter.forward(model, noisy_latents, timesteps, batch)

        # Compute target and loss
        target = adapter.compute_target(noise, target_latents, sigmas)
        if is_video:
            target = target.permute(0, 2, 1, 3, 4)  # back to model_pred layout

        weighting = self._compute_weighting(sigmas)
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(bsz, -1),
            dim=1,
        )
        loss = loss.mean()

        metrics = {"mse": loss.item()}
        return loss, metrics

    def _compute_weighting(self, sigmas):
        """Compute loss weighting based on scheme."""
        if self.weighting_scheme == "none":
            return torch.ones_like(sigmas)
        elif self.weighting_scheme == "sigma_sqrt":
            return (sigmas ** -2.0).clamp(max=10.0)
        return torch.ones_like(sigmas)

    def create_collator(self):
        return self._collator if self._collator is not None else EditingCollator()
