"""Epsilon prediction loss — standard SFT for DDPM/any diffusion model.

The diffusion equivalent of Grimoire's SFTLoss: just train the model
to predict the target (noise for epsilon, velocity for flow matching)
without any preference signal.
"""

from ..data.generation import GenerationCollator
from .utils import get_single_denoising_loss


class EpsilonLoss:
    """Standard denoising loss for any diffusion model.

    Works with any adapter — uses adapter.compute_target() to determine
    what the model should predict (noise for DDPM, velocity for flow matching).

    Loss = mean(MSE(model_pred, target))

    Despite the name, this works for epsilon, v-prediction, and flow matching
    by delegating target computation to the adapter.
    """

    def __call__(self, adapter, model, batch, training=True):
        per_sample, mean_loss, _ = get_single_denoising_loss(adapter, model, batch)

        metrics = {"mse": mean_loss.item()}
        return mean_loss, metrics

    def create_collator(self):
        return GenerationCollator()
