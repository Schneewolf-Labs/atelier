# Changelog

All notable changes to Atelier will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
