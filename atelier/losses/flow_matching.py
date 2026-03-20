import torch

from ..data.editing import EditingCollator


class FlowMatchingLoss:
    """Flow matching MSE loss for diffusion training.

    Used with flow matching models (Qwen-Image-Edit, SD3, FLUX).
    Predicts the velocity field (noise - latents) at a given timestep.

    Loss = weighted_mean(MSE(model_pred, target))

    where target = noise - latents and weighting is determined by the
    weighting scheme.
    """

    def __init__(self, weighting_scheme="none"):
        self.weighting_scheme = weighting_scheme

    def __call__(self, adapter, model, batch, training=True):
        target_latents = batch["target_latents"].to(model.device, model.dtype)

        # Permute from [B, C, 1, H, W] to [B, 1, C, H, W] for model
        target_latents = target_latents.permute(0, 2, 1, 3, 4)

        # Normalize latents if adapter supports it
        if hasattr(adapter, "normalize_latents"):
            target_latents = adapter.normalize_latents(target_latents)
            if "control_latents" in batch:
                batch = dict(batch)  # shallow copy to avoid mutating original
                control = batch["control_latents"].to(model.device, model.dtype).permute(0, 2, 1, 3, 4)
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
        target = target.permute(0, 2, 1, 3, 4)  # back to [B, C, 1, H, W] to match model_pred

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
        return EditingCollator()
