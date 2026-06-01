"""Reward contract for Flow-GRPO — the membrane between Atelier and Merlina.

Atelier defines the Protocol ONLY. The concrete rewards (aesthetic / CLIP /
preference models, VLM-as-judge, sandboxed user image->float) and any weighted
composition of them live in Merlina, hidden behind a single synchronous callable.
Atelier's trainer only ever sees one ImageRewardFn.
"""

import logging
import math
from typing import Sequence, runtime_checkable

try:  # Protocol is stdlib on 3.8+, but keep the import defensive.
    from typing import Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol

logger = logging.getLogger(__name__)


@runtime_checkable
class ImageRewardFn(Protocol):
    """Synchronous callable scoring generated images.

    Called once per rollout group-batch by FlowGRPOTrainer. MUST be synchronous
    and return one finite float per image (same length/order as ``images``). Any
    GPU reward model / network VLM-judge / sandboxed user code is the caller's
    problem and must be hidden behind this sync interface before it reaches the
    trainer.

    Args:
        prompts: text prompts (the prompts in this step).
        images:  decoded RGB images, length ``len(prompts) * num_generations`` in
                 group-major order [p0g0, p0g1, ..., p0g{G-1}, p1g0, ...]. Images
                 are ``PIL.Image.Image`` (chosen for reward-model interop).
        **columns: extra dataset columns aligned/broadcast to ``images``.

    Returns:
        list[float] of length ``len(images)``. NaN/inf are coerced to 0.0 with a
        logged warning by the trainer (see ``sanitize_rewards``).
    """

    def __call__(
        self,
        prompts: Sequence[str],
        images: Sequence,
        **columns: Sequence,
    ) -> "list[float]": ...


def sanitize_rewards(rewards: Sequence[float], expected: int) -> "list[float]":
    """Coerce a reward callable's output into a clean list of finite floats.

    Non-finite values (NaN/inf) become 0.0 with a logged warning. Raises if the
    length does not match the number of images that were scored — a length
    mismatch is a contract violation, not something to paper over.
    """
    values = list(rewards)
    if len(values) != expected:
        raise ValueError(
            f"reward_fn returned {len(values)} scores but {expected} images were scored"
        )
    clean = []
    for i, value in enumerate(values):
        f = float(value)
        if not math.isfinite(f):
            logger.warning("reward_fn returned non-finite score %r at index %d -> 0.0", value, i)
            f = 0.0
        clean.append(f)
    return clean
