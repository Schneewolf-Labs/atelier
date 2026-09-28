# Changelog

All notable changes to Atelier will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Qwen-Image 2.1** (`QwenImage21Adapter`, `qwen_image_2_1`) via a native PyTorch
  port — diffusers has no support. `atelier.models.qwen_image_2_1` ports sd.cpp's
  single-stream DiT (block-causal text/reference/target sequence, t / t=0 dual
  modulation, centred 3-axis RoPE) with checkpoint-compatible names; the RGBA,
  64-channel VAE reuses diffusers' `AutoencoderKLWan` with an original→diffusers
  key converter; Qwen3-VL conditioning follows sd.cpp's template, vision slots
  and pre-norm hidden state. Text-to-image and reference editing, RGBA training
  images, and LoRA export in sd.cpp / ComfyUI key layout.
- `scripts/sdcpp_parity/`: C++ harness running sd.cpp's own Qwen-Image 2.1 runners
  plus DiT / VAE comparison scripts; DiT golden outputs in `tests/data/` are checked
  in CI.
- `cache_embeddings` keeps alpha for RGBA sources and passes the resize target to
  `encode_text` (vision-conditioned encoders see references at VAE resolution);
  `EditingCollator` pads `prompt_token_types` like the mask.
- **New adapters**, cross-checked against stable-diffusion.cpp and the diffusers
  pipelines:
  - `StableDiffusionAdapter` (`sd`) — SD 1.x / 2.x, epsilon or v-prediction, `clip_skip`.
  - `SD3Adapter` (`sd3`) — SD 3 / 3.5, optional T5 drop (`load_t5=False`).
  - `FluxAdapter` (`flux`) — FLUX.1 dev / schnell; `guidance` (default 1.0),
    resolution-matched dynamic shift, single-file transformer via `transformer_path=`.
  - `FluxKontextAdapter` (`flux_kontext`) — FLUX.1 Kontext editing; `control_latents`
    become index-1 reference tokens, prediction cropped to the target.
  - `ChromaAdapter` (`chroma`) — T5-only, padding mask carried into the DiT joint attention.
  - `ZImageAdapter` (`z_image`) — Z-Image / Turbo; reversed time and negated output handled.
- Shared bases: `DDPMAdapter` (epsilon / v-pred / sample targets) and
  `FlowMatchAdapter` / `DiffusersFlowAdapter` (shifted logit-normal / uniform / mode
  timestep sampling, deferred-transformer loading, VAE-normalized latents,
  PEFT-format LoRA saving).
- `SDXLAdapter`: `prediction_type=` (v-pred finetunes) and single-file checkpoints;
  `encode_images` now honors `height` / `width` from `cache_embeddings`.
- `EditingDataset` / `EditingCollator` carry any extra cached text tensors
  (e.g. `pooled_prompt_embeds`); `GenerationDataset` always emits `prompt`, and
  losses encode it on the fly for adapters that own their tokenizers.

### Changed
- Adapter protocol: `compute_target(noise, latents, sigmas, timesteps=None)`. Losses
  pass `timesteps=` (needed for v-prediction). Custom adapters should accept the
  keyword.
- `strip_peft_prefix` moved to `atelier.adapters.base` (still importable from
  `atelier.adapters.qwen_image`).

## [0.1.1] - 2026-05-30

### Added
- **DoRA support documented + wired through the YAML config path**
  (`peft.use_dora: true`). DoRA (Weight-Decomposed Low-Rank Adaptation,
  Liu et al. 2024) decomposes the LoRA update into a magnitude vector
  + a directional matrix, giving meaningfully better quality on small /
  medium aesthetic datasets where vanilla LoRA underfits — at ~5-10%
  extra step time. Requires `peft >= 0.10`.
- DoRA was already supported as a `LoraConfig` kwarg (Atelier's YAML
  builder passes through any PEFT-config field), but it wasn't
  discoverable. This release just makes it a first-class option in
  the example config + README quick-start.

## [0.1.0] - 2026-05-29

### Added
- Initial PyPI release as `atelier-diffusion` (the bare `atelier` name was taken;
  the import name stays `atelier`, mirroring the `grimoire-rl` → `import grimoire`
  pattern used by Atelier's sister project).
- **Adapters**: `QwenImageAdapter` (Qwen-Image text-to-image, DiT + flow matching),
  `QwenEditAdapter` (Qwen-Image-Edit, image-to-image, DiT + flow matching with a
  vision-conditioned text encoder), `SDXLAdapter` (SDXL, UNet + dual CLIP + DDPM).
- **Losses**: `FlowMatchingLoss`, `EpsilonLoss`, plus six preference-optimization
  variants — `DiffusionDPOLoss`, `DiffusionCPOLoss`, `DiffusionIPOLoss`,
  `DiffusionKTOLoss`, `DiffusionORPOLoss`, `DiffusionSimPOLoss`.
- **Data**: `EditingDataset` + `EditingCollator` (paired image editing; also
  serves the no-control T2I case), `GenerationDataset` + `GenerationCollator`
  (SDXL-style), `cache_embeddings` (pre-computes text + image embeddings to
  disk so the encoder + transformer don't have to coexist in VRAM during
  training).
- **CLI**: `python -m atelier.train --config foo.yaml` for orchestrators
  (e.g. [Merlina](https://github.com/Schneewolf-Labs/Merlina)). YAML schema
  mirrors the Python API; `--set key.sub=value` JSON-aware overrides.
- **Registry**: `ADAPTERS` + `LOSSES` short-name resolution; full
  `pkg.mod:ClassName` specs also accepted.
- **Subprocess-isolated cache stage** + `load_encoders` / `load_transformer`
  flags on `QwenImageAdapter` for the Qwen-Image VRAM dance (38 GiB transformer
  + 14 GiB text encoder don't fit on 48 GB simultaneously).

### Fixed
- **LoRA save format**: `save_lora` was writing the legacy diffusers layout
  (`base_model.model.…lora.down/up.weight`) that modern
  `pipe.load_lora_weights` rejects. Now strips the PEFT wrapper prefix and
  writes the PEFT-format keys directly.
