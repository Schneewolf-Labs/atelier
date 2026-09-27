import logging

import torch

from .flow import DiffusersFlowAdapter

logger = logging.getLogger(__name__)


class ZImageAdapter(DiffusersFlowAdapter):
    """Adapter for Z-Image / Z-Image-Turbo (single-stream DiT + Qwen3 + flow matching).

    Two conventions a naive flow trainer gets wrong (both confirmed against
    ``ZImagePipeline`` and stable-diffusion.cpp's ``z_image.hpp``):

    - **Reversed time.** The transformer is fed ``t = 1 - sigma`` (1 = clean),
      not sigma.
    - **Negated output.** The network natively predicts ``x0 - noise``; the
      pipeline flips the sign. :meth:`forward` does the same, so the
      shared ``target = noise - x0`` holds and every Atelier loss works
      unchanged.

    Other details: FLUX VAE (16 ch, ``(z - 0.1159) * 0.3611``), Qwen3
    penultimate hidden state as caption features, chat-templated prompt,
    padding tokens dropped per sample (the transformer pads with its own
    learned ``cap_pad_token``). The transformer takes *lists* of
    per-sample tensors, so text embeds are carried as a padded
    ``[B, L, D]`` + ``prompt_embeds_mask`` and unpadded in :meth:`forward`.
    """

    pipeline_class = "ZImagePipeline"
    transformer_class = "ZImageTransformer2DModel"

    def __init__(self, pretrained_path, max_sequence_length=512, **kwargs):
        self.max_sequence_length = max_sequence_length
        super().__init__(pretrained_path, **kwargs)

    def encode_text(self, prompts, device=None, **kwargs):
        device = device or self._device
        if isinstance(prompts, str):
            prompts = [prompts]
        embeds, _ = self._pipeline.encode_prompt(
            prompt=list(prompts),
            device=device,
            do_classifier_free_guidance=False,
            max_sequence_length=self.max_sequence_length,
        )
        return pad_caption_features(embeds)

    def forward(self, model, noisy_latents, timesteps, batch):
        device, dtype = noisy_latents.device, noisy_latents.dtype
        prompt_embeds = batch["prompt_embeds"].to(device, dtype)
        mask = batch.get("prompt_embeds_mask")
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device).bool()
            cap_feats = [emb[m] for emb, m in zip(prompt_embeds, mask)]
        else:
            cap_feats = list(prompt_embeds.unbind(0))

        # [B, C, H, W] → list of [C, 1, H, W] (single-frame video layout)
        x = list(noisy_latents.unsqueeze(2).unbind(0))
        t = 1.0 - timesteps.to(device=device, dtype=torch.float32) / self.num_train_timesteps

        out = model(x, t, cap_feats, return_dict=False)[0]
        return -torch.stack(out, dim=0).squeeze(2)


def pad_caption_features(embeds):
    """List of ``[L_i, D]`` → ``{"prompt_embeds": [B, L, D], "prompt_embeds_mask": [B, L]}``."""
    max_len = max(e.shape[0] for e in embeds)
    dim = embeds[0].shape[-1]
    out = embeds[0].new_zeros(len(embeds), max_len, dim)
    mask = torch.zeros(len(embeds), max_len, dtype=torch.long, device=embeds[0].device)
    for i, e in enumerate(embeds):
        out[i, : e.shape[0]] = e
        mask[i, : e.shape[0]] = 1
    return {"prompt_embeds": out, "prompt_embeds_mask": mask}
