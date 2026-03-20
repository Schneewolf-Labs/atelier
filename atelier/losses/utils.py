"""Shared utilities for diffusion loss functions.

The core mapping from LLM training to diffusion training:
- LLM: avg_log_prob(sequence) — higher = model prefers this
- Diffusion: -MSE(pred_noise, target) — lower MSE = model prefers this

So in preference losses:
- LLM: chosen_logps - rejected_logps (positive = prefers chosen)
- Diffusion: loss_rejected - loss_chosen (positive = prefers chosen)
"""

import torch
import torch.nn.functional as F


def get_paired_denoising_losses(adapter, model, batch, timestep_bias=None):
    """Core computation shared by all diffusion preference losses.

    Handles encoding, noise sampling, forward passes, and MSE computation
    for both chosen and rejected images using the adapter protocol.

    Args:
        adapter: ModelAdapter instance.
        model: The trainable model (may be wrapped by accelerate/PEFT).
        batch: Batch dict with either:
            - chosen_latents/rejected_latents (pre-computed), or
            - chosen_image/rejected_image (raw tensors for on-the-fly encoding)
        timestep_bias: Optional (lo, hi) tuple to bias timestep sampling range.

    Returns:
        chosen_per: [B] per-sample MSE for chosen images
        rejected_per: [B] per-sample MSE for rejected images
        sft_loss: scalar mean MSE on chosen (for SFT regularization)
        forward_batch: the batch dict used for forward passes (for metrics)
    """
    device = next(model.parameters()).device

    # Get text conditioning
    forward_batch = _get_text_conditioning(adapter, batch, device)

    # Get latents (pre-computed or encode on-the-fly)
    chosen_latents = _get_latents(adapter, batch, "chosen_latents", "chosen_image", device)
    rejected_latents = _get_latents(adapter, batch, "rejected_latents", "rejected_image", device)

    if chosen_latents is None or rejected_latents is None:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, forward_batch

    # Sample shared noise and timesteps
    noise = torch.randn_like(chosen_latents)
    bsz = chosen_latents.shape[0]

    if timestep_bias is not None:
        lo, hi = timestep_bias
        T = adapter.noise_scheduler.config.num_train_timesteps
        u = torch.rand(bsz, device=device)
        u = lo + (hi - lo) * u
        timesteps = (u * T).long().clamp_(0, T - 1)
        sigmas = None
    else:
        timesteps, sigmas = adapter.sample_timesteps(bsz, device)

    # Add same noise to both
    noisy_chosen = adapter.add_noise(chosen_latents, noise, timesteps, sigmas)
    noisy_rejected = adapter.add_noise(rejected_latents, noise, timesteps, sigmas)

    # Forward passes
    pred_chosen = adapter.forward(model, noisy_chosen, timesteps, forward_batch)
    pred_rejected = adapter.forward(model, noisy_rejected, timesteps, forward_batch)

    # Compute targets via adapter
    target_chosen = adapter.compute_target(noise, chosen_latents, sigmas)
    target_rejected = adapter.compute_target(noise, rejected_latents, sigmas)

    # Per-sample MSE in float32
    pred_chosen = pred_chosen.float()
    pred_rejected = pred_rejected.float()
    target_chosen = target_chosen.float()
    target_rejected = target_rejected.float()

    spatial_dims = list(range(1, pred_chosen.ndim))
    chosen_per = F.mse_loss(pred_chosen, target_chosen, reduction="none").mean(dim=spatial_dims)
    rejected_per = F.mse_loss(pred_rejected, target_rejected, reduction="none").mean(dim=spatial_dims)
    sft_loss = F.mse_loss(pred_chosen, target_chosen, reduction="mean")

    return chosen_per, rejected_per, sft_loss, forward_batch


def get_single_denoising_loss(adapter, model, batch, timestep_bias=None):
    """Core computation for single-image losses (SFT, KTO).

    Args:
        adapter: ModelAdapter instance.
        model: The trainable model.
        batch: Batch dict with either image_latents or image tensors.
        timestep_bias: Optional (lo, hi) tuple.

    Returns:
        per_sample: [B] per-sample MSE
        mean_loss: scalar mean MSE
        forward_batch: the batch dict used for forward passes
    """
    device = next(model.parameters()).device

    forward_batch = _get_text_conditioning(adapter, batch, device)

    latents = _get_latents(adapter, batch, "image_latents", "image", device)
    if latents is None:
        latents = _get_latents(adapter, batch, "target_latents", "chosen_image", device)

    if latents is None:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, forward_batch

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    if timestep_bias is not None:
        lo, hi = timestep_bias
        T = adapter.noise_scheduler.config.num_train_timesteps
        u = torch.rand(bsz, device=device)
        u = lo + (hi - lo) * u
        timesteps = (u * T).long().clamp_(0, T - 1)
        sigmas = None
    else:
        timesteps, sigmas = adapter.sample_timesteps(bsz, device)

    noisy_latents = adapter.add_noise(latents, noise, timesteps, sigmas)
    prediction = adapter.forward(model, noisy_latents, timesteps, forward_batch)
    target = adapter.compute_target(noise, latents, sigmas)

    prediction = prediction.float()
    target = target.float()

    spatial_dims = list(range(1, prediction.ndim))
    per_sample = F.mse_loss(prediction, target, reduction="none").mean(dim=spatial_dims)
    mean_loss = F.mse_loss(prediction, target, reduction="mean")

    return per_sample, mean_loss, forward_batch


def _get_latents(adapter, batch, latents_key, image_key, device):
    """Get latents from batch: use pre-computed if available, else encode on-the-fly."""
    if latents_key in batch:
        latents = batch[latents_key]
        if isinstance(latents, torch.Tensor):
            return latents.to(device)
        return None

    if image_key in batch:
        images = batch[image_key]
        if isinstance(images, torch.Tensor):
            with torch.no_grad():
                return adapter.encode_image_tensor(images, device=device)
        return None

    return None


def _get_text_conditioning(adapter, batch, device):
    """Get text conditioning: use pre-computed embeddings or encode on-the-fly."""
    # Pre-computed embeddings
    if "prompt_embeds" in batch:
        result = {}
        for key in ("prompt_embeds", "prompt_embeds_mask", "pooled_prompt_embeds", "time_ids"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                result[key] = batch[key].to(device)
        return result

    # On-the-fly encoding via adapter (SDXL tokenized batch)
    if "input_ids" in batch:
        return adapter.encode_text(batch=batch, device=device)

    return {}
