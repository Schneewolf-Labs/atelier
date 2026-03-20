import numpy as np
import torch
import torch.nn.functional as F

from ..data.generation import GenerationCollator


class DiffusionDPOLoss:
    """DPO (Direct Preference Optimization) loss for diffusion models.

    Given (prompt, chosen_image, rejected_image) triples:
    1. Encode both images to latent space
    2. Add shared noise at a shared timestep
    3. Predict noise with the model for both
    4. Apply DPO objective: loss = -log_sigmoid(beta * (loss_rejected - loss_chosen))
    5. Add SFT regularization on chosen examples

    Total loss = dpo_loss + sft_weight * sft_loss

    The model learns to predict noise more accurately for chosen images,
    implicitly learning that chosen is a better match for the prompt.
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
        # Encode text and build batch for adapter.forward()
        text_data = adapter.encode_text(batch=batch, device=model.device)
        forward_batch = {**text_data}

        # Encode images to latents
        chosen_image = batch["chosen_image"].to(dtype=torch.float32, device=adapter._vae.device)
        rejected_image = batch["rejected_image"].to(dtype=torch.float32, device=adapter._vae.device)

        with torch.no_grad():
            chosen_latents = adapter._vae.encode(chosen_image).latent_dist.sample()
            rejected_latents = adapter._vae.encode(rejected_image).latent_dist.sample()

            if torch.isnan(chosen_latents).any() or torch.isnan(rejected_latents).any():
                return torch.tensor(0.0, device=model.device, requires_grad=True), {
                    "dpo_loss": 0.0, "sft_loss": 0.0, "total_loss": 0.0,
                }

            chosen_latents = chosen_latents * adapter._vae.config.scaling_factor
            rejected_latents = rejected_latents * adapter._vae.config.scaling_factor

        # Sample shared noise and timesteps (biased to mid-range)
        noise = torch.randn_like(chosen_latents)
        bsz = chosen_latents.shape[0]
        T = adapter.noise_scheduler.config.num_train_timesteps
        lo, hi = self.timestep_bias_range
        u = torch.rand(bsz, device=chosen_latents.device)
        u = lo + (hi - lo) * u
        timesteps = (u * T).long().clamp_(0, T - 1)

        # Add same noise to both
        noisy_chosen = adapter.noise_scheduler.add_noise(chosen_latents, noise, timesteps)
        noisy_rejected = adapter.noise_scheduler.add_noise(rejected_latents, noise, timesteps)

        # Predict noise for both
        pred_chosen = adapter.forward(model, noisy_chosen, timesteps, forward_batch)
        pred_rejected = adapter.forward(model, noisy_rejected, timesteps, forward_batch)

        # Compute losses in float32
        pred_chosen = pred_chosen.float()
        pred_rejected = pred_rejected.float()
        noise = noise.float()

        # SFT regularization: mean MSE on chosen
        sft_loss = F.mse_loss(pred_chosen, noise, reduction="mean")

        # Per-sample losses for DPO
        loss_chosen_per = F.mse_loss(pred_chosen, noise, reduction="none").mean(dim=[1, 2, 3])
        loss_rejected_per = F.mse_loss(pred_rejected, noise, reduction="none").mean(dim=[1, 2, 3])

        # DPO log-ratio with clamping
        pi_logratios = loss_rejected_per - loss_chosen_per
        pi_logratios = torch.clamp(pi_logratios, min=-self.logit_clamp, max=self.logit_clamp)

        current_beta = self._get_beta()
        dpo_loss = -F.logsigmoid(current_beta * pi_logratios).mean()

        total_loss = dpo_loss + self.sft_weight * sft_loss

        metrics = {
            "dpo_loss": dpo_loss.item() if not torch.isnan(dpo_loss) else 0.0,
            "sft_loss": sft_loss.item() if not torch.isnan(sft_loss) else 0.0,
            "total_loss": total_loss.item() if not torch.isnan(total_loss) else 0.0,
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
