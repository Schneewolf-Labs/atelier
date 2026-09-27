import logging

import torch

from .flow import DiffusersFlowAdapter

logger = logging.getLogger(__name__)


class SD3Adapter(DiffusersFlowAdapter):
    """Adapter for Stable Diffusion 3 / 3.5 (MMDiT + CLIP-L + CLIP-G + T5 + flow matching).

    - 16-channel VAE, latents ``(z - 0.0609) * 1.5305``
    - The MMDiT takes ``timestep = sigma * 1000`` unscaled
    - Default shift 3.0 (from the scheduler config), logit-normal sampling
      — the SD3 paper's recipe

    ``load_t5=False`` drops the 4.7B T5-XXL encoder; diffusers then feeds
    zeros for the T5 half of the context, which SD3 was trained to tolerate.
    Only do that if you'll also sample without T5.
    """

    pipeline_class = "StableDiffusion3Pipeline"
    transformer_class = "SD3Transformer2DModel"

    def __init__(self, pretrained_path, max_sequence_length=256, load_t5=True, **kwargs):
        self.max_sequence_length = max_sequence_length
        if not load_t5:
            kwargs.setdefault("pipeline_kwargs", {}).update(text_encoder_3=None, tokenizer_3=None)
        super().__init__(pretrained_path, **kwargs)

    def encode_text(self, prompts, device=None, **kwargs):
        device = device or self._device
        prompt_embeds, _, pooled_prompt_embeds, _ = self._pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=self.max_sequence_length,
        )
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

    def forward(self, model, noisy_latents, timesteps, batch):
        dtype = noisy_latents.dtype
        device = noisy_latents.device
        return model(
            hidden_states=noisy_latents,
            timestep=timesteps.to(device=device, dtype=torch.float32),
            encoder_hidden_states=batch["prompt_embeds"].to(device, dtype),
            pooled_projections=batch["pooled_prompt_embeds"].to(device, dtype),
            return_dict=False,
        )[0]
