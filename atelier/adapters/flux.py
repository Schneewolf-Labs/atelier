import logging

import torch

from .flow import DiffusersFlowAdapter

logger = logging.getLogger(__name__)


def pack_latents(latents):
    """``[B, C, H, W]`` → ``[B, (H/2)*(W/2), C*4]`` 2x2-patch tokens (FLUX layout)."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(tokens, height, width):
    """Inverse of :func:`pack_latents` for latent-space ``height`` × ``width``."""
    b, _, c4 = tokens.shape
    tokens = tokens.view(b, height // 2, width // 2, c4 // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return tokens.reshape(b, c4 // 4, height, width)


def latent_position_ids(height, width, index=0, device=None, dtype=None):
    """3-axis RoPE ids ``[index, row, col]`` for a packed ``height/2 × width/2`` grid.

    Target tokens use index 0; FLUX Kontext marks reference-image tokens
    with index 1 on the same row/col grid.
    """
    h, w = height // 2, width // 2
    ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    ids[..., 0] = index
    ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
    return ids.reshape(h * w, 3)


class FluxAdapter(DiffusersFlowAdapter):
    """Adapter for FLUX.1 dev / schnell (+ Kontext editing) — MMDiT + CLIP-L + T5 + flow matching.

    - 16-channel VAE, latents ``(z - 0.1159) * 0.3611``, packed into 2x2
      patches; text RoPE ids are all zero, image ids ``[0, row, col]``
    - The transformer takes ``sigma`` (it multiplies by 1000 internally)
    - dev is guidance-distilled: a guidance embedding is fed on every step.
      Training on real images uses ``guidance=1.0`` by default (the kohya /
      ai-toolkit convention); diffusers' reference scripts use 3.5.
    - Shift defaults to the dynamic-shift ``exp(mu)`` at ``resolution``
      (≈3.16 at 1024²) for dev, 1.0 for schnell.

    **Editing (Kontext).** If the batch carries ``control_latents`` (the
    ``rejected`` column in :class:`~atelier.data.EditingDataset`), they're
    packed and appended after the target tokens with RoPE index 1 —
    exactly what ``FluxKontextPipeline`` does at inference — and the
    prediction is cropped back to the target tokens. Use
    :class:`FluxKontextAdapter` to load a Kontext checkpoint.
    """

    pipeline_class = "FluxPipeline"
    transformer_class = "FluxTransformer2DModel"

    def __init__(self, pretrained_path, max_sequence_length=512, guidance=1.0, **kwargs):
        self.max_sequence_length = max_sequence_length
        self.guidance = guidance
        super().__init__(pretrained_path, **kwargs)
        self._guidance_embeds = self._read_guidance_embeds(pretrained_path)

    def _read_guidance_embeds(self, pretrained_path):
        # Read once here: inside forward the model may be wrapped by
        # PEFT / DDP, which don't all forward ``.config``.
        if self._model is not None:
            return bool(getattr(self._model.config, "guidance_embeds", False))
        import diffusers

        cfg = getattr(diffusers, self.transformer_class).load_config(pretrained_path, subfolder="transformer")
        return bool(cfg.get("guidance_embeds", False))

    def encode_text(self, prompts, device=None, **kwargs):
        device = device or self._device
        prompt_embeds, pooled_prompt_embeds, _ = self._pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=self.max_sequence_length,
        )
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

    def _prepare_image_tokens(self, noisy_latents, batch):
        """Pack the noisy target (+ optional Kontext reference) into one token sequence."""
        device, dtype = noisy_latents.device, noisy_latents.dtype
        _, _, h, w = noisy_latents.shape
        hidden = pack_latents(noisy_latents)
        img_ids = latent_position_ids(h, w, device=device, dtype=dtype)

        control = batch.get("control_latents")
        if isinstance(control, torch.Tensor):
            control = control.to(device, dtype)
            ch, cw = control.shape[-2:]
            hidden = torch.cat([hidden, pack_latents(control)], dim=1)
            img_ids = torch.cat([img_ids, latent_position_ids(ch, cw, index=1, device=device, dtype=dtype)], dim=0)
        return hidden, img_ids

    def forward(self, model, noisy_latents, timesteps, batch):
        device, dtype = noisy_latents.device, noisy_latents.dtype
        bsz, _, h, w = noisy_latents.shape
        num_target_tokens = (h // 2) * (w // 2)

        hidden, img_ids = self._prepare_image_tokens(noisy_latents, batch)
        prompt_embeds = batch["prompt_embeds"].to(device, dtype)
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
        guidance = (
            torch.full((bsz,), self.guidance, device=device, dtype=torch.float32)
            if self._guidance_embeds else None
        )

        out = model(
            hidden_states=hidden,
            timestep=timesteps.to(device=device, dtype=torch.float32) / self.num_train_timesteps,
            guidance=guidance,
            pooled_projections=batch["pooled_prompt_embeds"].to(device, dtype),
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        return unpack_latents(out[:, :num_target_tokens], h, w)


class FluxKontextAdapter(FluxAdapter):
    """FLUX.1 Kontext (instruction editing). Same forward as :class:`FluxAdapter`;
    loads via ``FluxKontextPipeline`` and expects ``control_latents`` in the batch."""

    pipeline_class = "FluxKontextPipeline"


class ChromaAdapter(FluxAdapter):
    """Adapter for Chroma (FLUX-derived, T5-only, no guidance embedding).

    Differences from FLUX that matter for training:

    - No CLIP / pooled vector; conditioning is T5 only (512 tokens).
    - The T5 padding mask is kept and extended over the image tokens as the
      DiT's joint attention mask, with one pad token left unmasked — the
      same mask ``ChromaPipeline`` builds, so train and sample agree.
    - No guidance embedding; modulation comes from Chroma's approximator.
    """

    pipeline_class = "ChromaPipeline"
    transformer_class = "ChromaTransformer2DModel"

    def __init__(self, pretrained_path, max_sequence_length=512, **kwargs):
        kwargs.pop("guidance", None)
        super().__init__(pretrained_path, max_sequence_length=max_sequence_length, guidance=None, **kwargs)

    def _read_guidance_embeds(self, pretrained_path):
        return False

    def encode_text(self, prompts, device=None, **kwargs):
        device = device or self._device
        prompt_embeds, _, prompt_mask, *_ = self._pipeline.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=self.max_sequence_length,
        )
        return {"prompt_embeds": prompt_embeds, "prompt_embeds_mask": prompt_mask}

    def forward(self, model, noisy_latents, timesteps, batch):
        device, dtype = noisy_latents.device, noisy_latents.dtype
        bsz, _, h, w = noisy_latents.shape
        num_target_tokens = (h // 2) * (w // 2)

        hidden, img_ids = self._prepare_image_tokens(noisy_latents, batch)
        prompt_embeds = batch["prompt_embeds"].to(device, dtype)
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

        attention_mask = None
        mask = batch.get("prompt_embeds_mask")
        if isinstance(mask, torch.Tensor):
            image_mask = torch.ones(bsz, hidden.shape[1], device=device, dtype=dtype)
            attention_mask = torch.cat([mask.to(device, dtype), image_mask], dim=1)

        out = model(
            hidden_states=hidden,
            timestep=timesteps.to(device=device, dtype=torch.float32) / self.num_train_timesteps,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]
        return unpack_latents(out[:, :num_target_tokens], h, w)
