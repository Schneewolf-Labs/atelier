"""SDE-sampling contract — the new capability adapters must add to be GRPO-trainable.

Flow matching is a deterministic ODE; to get per-step logprobs you must sample with
the SDE ("SDE-ification" a la Flow-GRPO) so each denoising step is a Gaussian
transition with a tractable logprob. This is the diffusion analog of an LLM
``generate()`` that also returns per-token logprobs, and it is the single biggest
new piece each adapter (QwenImageAdapter / QwenEditAdapter / SDXLAdapter) must
implement to participate in Flow-GRPO training.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import torch


@dataclass
class SDERollout:
    """A sampled denoising trajectory plus everything needed to score it.

    Attributes:
        latents_trajectory: (N, S+1, ...) the sampled chain. Retained so the trainer
            can recompute per-step logprobs under the current (grad-enabled) and
            reference (adapter-disabled) policies for the ratio + KL terms.
        logprobs:  (N, S) per-step logprob under the policy that produced the rollout.
        step_mask: (N, S) 1 for valid denoising steps (handles ragged/subsampled steps).
        images:    list[PIL.Image] when sampled with ``decode=True`` (for the reward).
        timesteps: (S,) the realized schedule, for logging / reproducibility.
        prompt_embeddings: the cached text conditioning used for this rollout, so the
            adapter can re-run the transformer during ``recompute_logprobs``.
    """

    latents_trajectory: torch.Tensor
    logprobs: torch.Tensor
    step_mask: torch.Tensor
    images: Optional[list] = None
    timesteps: Optional[torch.Tensor] = None
    prompt_embeddings: Optional[dict] = field(default=None)

    @property
    def num_samples(self) -> int:
        return self.logprobs.shape[0]

    @property
    def num_steps(self) -> int:
        return self.logprobs.shape[1]


@runtime_checkable
class SupportsSDESampling(Protocol):
    """Adapters must implement this to be Flow-GRPO trainable.

    The trainer calls ``sample_with_logprobs`` once per step for the rollout, then
    ``recompute_logprobs`` to score that trajectory under the current (grad-enabled)
    and reference (adapter-disabled) policies.
    """

    def sample_with_logprobs(
        self,
        *,
        prompt_embeddings,
        num_inference_steps: int,
        guidance_scale: float,
        sde_noise_level: float,
        generator: torch.Generator,
        decode: bool = True,
    ) -> SDERollout:
        """Sample images by running the SDE denoising trajectory under the current policy.

        Args:
            prompt_embeddings: cached text embeddings for the batch (group-expanded).
            num_inference_steps: denoising steps per sample (S).
            guidance_scale: classifier-free guidance scale.
            sde_noise_level: eta>0 converts the flow ODE to an SDE so steps have logprobs;
                0 is deterministic (no logprobs) and is invalid for GRPO.
            generator: seeded for reproducibility.
            decode: when True, decode final latents -> PIL images via the (resident) VAE
                so the reward function can score them.
        """
        ...

    def recompute_logprobs(self, rollout: SDERollout, *, with_grad: bool) -> torch.Tensor:
        """Recompute (N, S) per-step logprobs for a stored rollout.

        Called with ``with_grad=True`` for the current policy (backprop target) and
        ``with_grad=False`` for old/reference policies. The reference policy is the
        same model with its LoRA adapter disabled — no second model copy.
        """
        ...
