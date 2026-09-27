"""Shared rectified-flow plumbing for 4-D latent DiTs (SD3, FLUX, Chroma, Z-Image).

All of these train on the same objective::

    x_t    = (1 - sigma) * x_0 + sigma * noise
    target = noise - x_0                    (velocity)

and differ only in how sigma is sampled (shift, logit-normal vs uniform)
and in how sigma is fed to the network, which each adapter's ``forward``
handles. Latents are ``[B, C, H, W]`` with the VAE scale/shift already
applied at encode time, so :class:`~atelier.losses.FlowMatchingLoss`
passes them straight through.

(Qwen-Image predates this base and keeps its own table-driven sampler over
the video-VAE ``[B, C, 1, H, W]`` layout.)
"""

import gc
import logging
import math
import os

import torch

from .base import ModelAdapter, is_single_file, pil_to_tensor, save_peft_lora

logger = logging.getLogger(__name__)

TIMESTEP_SAMPLING = ("uniform", "logit_normal", "mode")


def shift_sigmas(sigmas, shift):
    """SD3-style static time shift: ``shift * s / (1 + (shift - 1) * s)``.

    ``shift > 1`` pushes training toward high noise, which is what high-res
    flow models need (SD3 uses 3.0, Z-Image 3.0; FLUX's resolution-dependent
    ``mu`` is equivalent to ``shift = exp(mu)`` — ~3.16 at 1024²).
    """
    if shift == 1.0:
        return sigmas
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def flux_mu(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    """FLUX's resolution-dependent shift exponent (linear in packed token count)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * m + (base_shift - m * base_seq_len)


def default_shift(scheduler_config, resolution=1024, patch_pixels=16):
    """Static training shift implied by a scheduler config.

    Schedulers with ``use_dynamic_shifting`` (FLUX.1-dev, Chroma) pick
    ``mu`` from the image token count at inference time; training at a
    fixed ``resolution`` maps that to the equivalent static shift
    ``exp(mu)`` so the train-time noise distribution matches what the
    sampler will ask of the model. Otherwise the config's ``shift``
    (SD3: 3.0, Z-Image: 3.0, schnell: 1.0) is used as-is.
    """
    cfg = scheduler_config
    if cfg.get("use_dynamic_shifting", False):
        seq_len = (resolution // patch_pixels) ** 2
        mu = flux_mu(
            seq_len,
            base_seq_len=cfg.get("base_image_seq_len", 256),
            max_seq_len=cfg.get("max_image_seq_len", 4096),
            base_shift=cfg.get("base_shift", 0.5),
            max_shift=cfg.get("max_shift", 1.15),
        )
        return math.exp(mu)
    return float(cfg.get("shift", 1.0))


class FlowMatchAdapter(ModelAdapter):
    """Base for rectified-flow adapters. Subclasses call :meth:`_init_flow`.

    Timestep sampling (``timestep_sampling``):

    - ``"logit_normal"`` — SD3 paper default, ``u = sigmoid(N(mean, std))``
    - ``"uniform"``      — ``u ~ U(0, 1)``
    - ``"mode"``         — SD3 paper's mode sampling with ``mode_scale``

    ``u`` is then time-shifted by ``shift`` to get sigma. ``timesteps`` are
    ``sigma * num_train_timesteps`` as floats — each adapter's ``forward``
    maps them to whatever its transformer expects.
    """

    num_train_timesteps = 1000

    def _init_flow(self, scheduler, shift=None, resolution=1024, timestep_sampling="logit_normal",
                   logit_mean=0.0, logit_std=1.0, mode_scale=1.29):
        if timestep_sampling not in TIMESTEP_SAMPLING:
            raise ValueError(f"timestep_sampling must be one of {TIMESTEP_SAMPLING}, got '{timestep_sampling}'")
        self._scheduler = scheduler
        if scheduler is not None:
            self.num_train_timesteps = scheduler.config.num_train_timesteps
        if shift is None:
            shift = default_shift(scheduler.config, resolution) if scheduler is not None else 1.0
        self.shift = float(shift)
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale

    @property
    def noise_scheduler(self):
        return self._scheduler

    def _sample_u(self, batch_size, device):
        if self.timestep_sampling == "logit_normal":
            u = torch.normal(self.logit_mean, self.logit_std, size=(batch_size,), device=device)
            return torch.sigmoid(u)
        u = torch.rand(batch_size, device=device)
        if self.timestep_sampling == "mode":
            u = 1 - u - self.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        return u

    def sample_timesteps(self, batch_size, device):
        sigmas = shift_sigmas(self._sample_u(batch_size, device).float(), self.shift)
        timesteps = sigmas * self.num_train_timesteps
        return timesteps, sigmas.view(-1, 1, 1, 1)

    def sigmas_for_timesteps(self, timesteps, device):
        """Explicit (e.g. range-biased) timesteps map linearly to sigma = t / T."""
        sigmas = timesteps.to(device=device, dtype=torch.float32) / self.num_train_timesteps
        return sigmas.clamp(0.0, 1.0).view(-1, 1, 1, 1)

    def add_noise(self, latents, noise, timesteps, sigmas):
        sigmas = sigmas.to(latents.device, latents.dtype).view(-1, *([1] * (latents.ndim - 1)))
        return (1.0 - sigmas) * latents + sigmas * noise

    def compute_target(self, noise, latents, sigmas=None, timesteps=None):
        return noise - latents

    def _sigmas_from_timesteps(self, timesteps, dtype):
        """Recover sigma in [0, 1] from whatever ``sample_timesteps`` returned."""
        return (timesteps.float() / self.num_train_timesteps).to(dtype)


class DiffusersFlowAdapter(FlowMatchAdapter):
    """Loading / encoding / saving shared by the diffusers-backed flow DiTs.

    Subclasses set ``pipeline_class`` + ``transformer_class`` (diffusers
    attribute names) and implement ``encode_text`` and ``forward``.

    Components load like :class:`QwenImageAdapter`: the text-encoder
    pipeline and VAE go to ``device``; the transformer is parked on CPU
    until :meth:`move_transformer_to_device` when ``defer_transformer`` is
    set, so peak VRAM during embedding caching is max(encoders,
    transformer), not the sum.

    ``transformer_path`` may point at a single-file transformer checkpoint
    (e.g. the ``flux1-dev.safetensors`` stable-diffusion.cpp loads) while
    ``pretrained_path`` supplies the diffusers configs, VAE and encoders.

    Latents are stored VAE-normalized — ``(z - shift_factor) * scaling_factor``
    — at encode time, so cached latents are ready for the transformer.
    """

    pipeline_class = None
    transformer_class = None

    def __init__(self, pretrained_path, device="cuda", dtype=None, transformer_path=None,
                 load_encoders=True, load_transformer=True, defer_transformer=True,
                 shift=None, resolution=1024, timestep_sampling="logit_normal",
                 logit_mean=0.0, logit_std=1.0, pipeline_kwargs=None):
        import diffusers

        self._dtype = dtype or torch.bfloat16
        self._device = device
        self._pipeline = None
        self._vae = None
        self._model = None

        if load_encoders:
            self._pipeline = getattr(diffusers, self.pipeline_class).from_pretrained(
                pretrained_path, transformer=None, vae=None, torch_dtype=self._dtype, **(pipeline_kwargs or {}),
            )
            self._pipeline.to(device)
            # fp32 VAE: it only runs while caching, and the 16-channel VAEs
            # are cheap enough that stability wins over speed.
            self._vae = diffusers.AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae")
            self._vae.to(device, dtype=torch.float32)
            self._vae.eval()
            self._vae.requires_grad_(False)

        # VAE config is metadata only — needed even when load_encoders=False.
        vae_config = diffusers.AutoencoderKL.load_config(pretrained_path, subfolder="vae")
        self.scaling_factor = vae_config["scaling_factor"]
        self.shift_factor = vae_config.get("shift_factor") or 0.0
        self.vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1)

        if load_transformer:
            transformer_cls = getattr(diffusers, self.transformer_class)
            if transformer_path is not None and is_single_file(transformer_path):
                self._model = transformer_cls.from_single_file(
                    transformer_path, config=pretrained_path, subfolder="transformer", torch_dtype=self._dtype,
                )
            else:
                self._model = transformer_cls.from_pretrained(
                    transformer_path or pretrained_path,
                    subfolder=None if transformer_path else "transformer",
                    torch_dtype=self._dtype,
                )
            if defer_transformer and load_encoders:
                self._model.to("cpu")
                self._transformer_on_device = False
                logger.info("Transformer loaded to CPU; call move_transformer_to_device() after free_encoders()")
            else:
                self._model.to(device)
                self._transformer_on_device = True

        scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
        self._init_flow(scheduler, shift=shift, resolution=resolution, timestep_sampling=timestep_sampling,
                        logit_mean=logit_mean, logit_std=logit_std)
        logger.info("%s: shift=%.3f, timestep_sampling=%s", type(self).__name__, self.shift, self.timestep_sampling)

    @property
    def model(self):
        return self._model

    @property
    def pipeline(self):
        """The text-encoder pipeline (no transformer / VAE), or None once freed."""
        return self._pipeline

    def move_transformer_to_device(self, device=None):
        """Move the trainable transformer onto the GPU (after :meth:`free_encoders`)."""
        device = device or self._device
        self._model.to(device)
        self._device = device
        self._transformer_on_device = True
        logger.info("Transformer moved to %s", device)

    def encode_image_tensor(self, image_tensor, device=None):
        device = device or self._device
        pixels = image_tensor.to(device=device, dtype=self._vae.dtype)
        with torch.no_grad():
            latents = self._vae.encode(pixels).latent_dist.sample()
        return (latents - self.shift_factor) * self.scaling_factor

    def encode_images(self, images, height=None, width=None, device=None, **kwargs):
        device = device or self._device
        latents = [
            self.encode_image_tensor(pil_to_tensor(image, height, width).unsqueeze(0), device=device)[0]
            for image in images
        ]
        return torch.stack(latents)

    def save_lora(self, model, path):
        import diffusers

        save_peft_lora(getattr(diffusers, self.pipeline_class), model, path)
        logger.info("LoRA weights saved to %s", path)

    def save_model(self, model, path):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path, safe_serialization=True)
        logger.info("Model saved to %s", path)

    def free_encoders(self):
        """Drop the text encoder(s) + VAE and reclaim VRAM.

        Components are moved to CPU before dropping refs — ``del`` alone
        leaves tensors alive through tokenizer / processor wiring.
        """
        for component in (self._pipeline, self._vae):
            if component is not None:
                try:
                    component.to("cpu")
                except Exception as e:
                    logger.warning("%s.to('cpu') failed: %s", type(component).__name__, e)
        self._pipeline = None
        self._vae = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.info("Freed VAE and text encoder(s) from memory")
