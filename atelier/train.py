"""
CLI entry point for Atelier training.

Usage:
    python -m atelier.train --config configs/my_run.yaml

The YAML config drives everything: which adapter, which loss, dataset
source, peft setup, and TrainingConfig fields. Built so Merlina (or any
other orchestrator) can spawn training runs without writing a Python
wrapper per job.

Schema (all sections optional unless noted):

    model:
      pretrained: "Qwen/Qwen-Image"           # REQUIRED
      adapter: "qwen_image"                   # REQUIRED — see registry.ADAPTERS
      adapter_args:                           # passed verbatim to adapter ctor
        defer_transformer: true

    dataset:                                  # REQUIRED
      # one of:
      name: "user/dataset"                    # HF hub
      split: "train"
      # OR
      jsonl: "./train.jsonl"                  # local file (HF datasets json loader)
      # OR
      path: "./dataset_dir"                   # load_from_disk
      max_samples: null

    loss:
      type: "flow_matching"                   # see registry.LOSSES
      args:                                   # passed to loss ctor
        weighting_scheme: "none"

    peft:                                     # optional — omit for full-model
      type: "lora"
      r: 32
      lora_alpha: 64
      target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
      lora_dropout: 0.05
      init_lora_weights: "gaussian"

    cache:                                    # optional — embedding pre-compute
      enable: true
      dir: "${training.output_dir}/cache"
      target_area: 1048576                    # 1024×1024

    training:                                 # all TrainingConfig fields supported
      output_dir: "./output"
      num_epochs: 8
      batch_size: 1
      learning_rate: 1.0e-4
      optimizer: "adafactor"
      mixed_precision: "bf16"
      gradient_checkpointing: true
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_yaml(path: str) -> dict:
    """Load a YAML config. Tries PyYAML then OmegaConf — neither is a hard dep."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except ImportError as e:
            raise ImportError(
                "YAML config loading needs `pyyaml` or `omegaconf` installed."
            ) from e


def load_dataset_from_spec(spec: dict):
    """Load a HF dataset from the YAML 'dataset' section.

    Accepts: {name, split}, {jsonl}, or {path}. Always returns a HF Dataset.
    """
    from datasets import Dataset, load_dataset, load_from_disk
    from datasets import Image as DSImage

    max_samples = spec.get("max_samples")
    if "jsonl" in spec:
        # JSONL with at least 'prompt' and either 'image' (path) or 'chosen' (path/image).
        rows = []
        with open(os.path.expanduser(spec["jsonl"])) as f:
            for line in f:
                rows.append(json.loads(line))
        # Normalize: 'image' → 'chosen' if 'chosen' missing
        for r in rows:
            if "chosen" not in r and "image" in r:
                r["chosen"] = r["image"]
        ds = Dataset.from_list(rows)
        if "chosen" in ds.column_names:
            ds = ds.cast_column("chosen", DSImage())
        if "rejected" in ds.column_names:
            ds = ds.cast_column("rejected", DSImage())
    elif "path" in spec:
        ds = load_from_disk(os.path.expanduser(spec["path"]))
    elif "name" in spec:
        ds = load_dataset(spec["name"], split=spec.get("split", "train"))
    else:
        raise ValueError("dataset section must include one of: jsonl, path, name")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def build_peft_config(spec: dict | None):
    """Build a PEFT config from the YAML 'peft' section. None → no PEFT."""
    if not spec:
        return None
    kwargs = dict(spec)
    peft_type = kwargs.pop("type", "lora").lower()
    if peft_type == "lora":
        from peft import LoraConfig
        # init_lora_weights="gaussian" is the diffusion-LoRA default that most
        # public training recipes use — make it the implicit default here too.
        kwargs.setdefault("init_lora_weights", "gaussian")
        return LoraConfig(**kwargs)
    raise ValueError(f"unsupported peft type: {peft_type}")


def build_training_config(spec: dict | None):
    """Map the YAML 'training' section onto TrainingConfig."""
    from .config import TrainingConfig
    return TrainingConfig(**(spec or {}))


def run_from_config(cfg: dict) -> None:
    """Drive a full training run from a parsed YAML config dict."""
    from .data.cache import cache_embeddings
    from .data.editing import EditingDataset
    from .registry import get_adapter_class, get_loss_class
    from .trainer import AtelierTrainer

    model_cfg = cfg.get("model") or {}
    if "pretrained" not in model_cfg or "adapter" not in model_cfg:
        raise ValueError("config.model requires 'pretrained' and 'adapter'")
    adapter_cls = get_adapter_class(model_cfg["adapter"])
    adapter_args = model_cfg.get("adapter_args") or {}

    loss_cfg = cfg.get("loss") or {"type": "flow_matching"}
    loss_cls = get_loss_class(loss_cfg["type"])
    loss_fn = loss_cls(**(loss_cfg.get("args") or {}))

    dataset_spec = cfg.get("dataset")
    if not dataset_spec:
        raise ValueError("config requires a 'dataset' section")
    raw_dataset = load_dataset_from_spec(dataset_spec)
    logger.info("dataset: %d rows", len(raw_dataset))

    training = build_training_config(cfg.get("training"))
    Path(training.output_dir).mkdir(parents=True, exist_ok=True)

    cache_cfg = cfg.get("cache") or {}
    cache_dir = None
    if cache_cfg.get("enable", True):
        cache_dir = cache_cfg.get("dir") or os.path.join(training.output_dir, "cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    logger.info("loading adapter %s", model_cfg["adapter"])
    adapter = adapter_cls(model_cfg["pretrained"], **adapter_args)

    text_emb, target_emb, control_emb = ({}, {}, {})
    if cache_dir:
        logger.info("pre-computing embeddings → %s", cache_dir)
        text_emb, target_emb, control_emb = cache_embeddings(
            raw_dataset, adapter, cache_dir=cache_dir,
            target_area=cache_cfg.get("target_area", 1024 * 1024),
        )
        if hasattr(adapter, "free_encoders"):
            adapter.free_encoders()
        if hasattr(adapter, "move_transformer_to_device"):
            adapter.move_transformer_to_device()

    train_dataset = EditingDataset(
        raw_dataset,
        cached_text_embeddings=text_emb or None,
        cached_target_embeddings=target_emb or None,
        cached_control_embeddings=control_emb or None,
    )

    trainer = AtelierTrainer(
        adapter=adapter,
        config=training,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        peft_config=build_peft_config(cfg.get("peft")),
    )
    trainer.train()

    final_dir = cfg.get("save_to") or os.path.join(training.output_dir, "final_lora")
    trainer.save_model(final_dir)
    logger.info("saved to %s", final_dir)


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse `--set key.sub=value` overrides into a nested dict.

    Values are JSON-decoded when possible so `--set training.num_epochs=4`
    yields an int and `--set peft.target_modules='[\"to_q\",\"to_v\"]'` yields
    a list.
    """
    out: dict[str, Any] = {}
    for item in items:
        key, _, value = item.partition("=")
        if not key or not value:
            raise SystemExit(f"--set expects key=value, got: {item}")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        # Walk into nested dicts
        parts = key.split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = parsed
    return out


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge over into base (returns a new dict)."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atelier.train",
                                     description="Run an Atelier training job from a YAML config.")
    parser.add_argument("--config", required=True, help="path to YAML config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override a config key (repeatable). Values JSON-decoded.")
    parser.add_argument("--log-level", default="INFO",
                        help="logging level (default INFO)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_yaml(args.config)
    if args.overrides:
        cfg = _deep_merge(cfg, _parse_overrides(args.overrides))

    run_from_config(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
