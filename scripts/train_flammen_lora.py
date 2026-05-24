"""
Train the flammen.ai aesthetic LoRA on Qwen-Image.

Two-stage VRAM-isolated pipeline:

  1. Shell out to scripts/encode_flammen.py (separate Python process):
     loads text encoder + VAE, runs cache_embeddings, writes cache to
     disk, EXITS. OS reclaims encoder VRAM cleanly.
  2. This process then loads only the transformer (no encoders), pulls
     the cache from disk, and trains.

Why two processes: Qwen-VL's multi-tower text encoder doesn't fully
release on .to('cpu') + del + empty_cache in a single process — leaves
~12 GiB stranded that prevents the 38 GiB transformer from fitting on
a 48 GiB A6000. Process isolation is the bulletproof fix.

Dataset: ~/flammen-lora-dataset/train.jsonl — one row per (image,
caption) pair; 634 unique images × 2 captions = 1268 rows. Captions
are VLM-grounded; 4 quarantine cases already dropped.

Usage:
    cd ~/Projects/atelier && source .venv/bin/activate
    python scripts/train_flammen_lora.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import datasets
from peft import LoraConfig

from atelier import AtelierTrainer, TrainingConfig
from atelier.adapters import QwenImageAdapter
from atelier.data import EditingDataset, cache_embeddings
from atelier.losses import FlowMatchingLoss

DATA_DIR    = Path(os.path.expanduser("~/flammen-lora-dataset"))
TRAIN_JSONL = DATA_DIR / "train.jsonl"
OUTPUT_DIR  = Path(os.path.expanduser("~/flammen-lora-output"))
CACHE_DIR   = OUTPUT_DIR / "cache"
QWEN_PATH   = os.environ.get("QWEN_IMAGE_PATH", "Qwen/Qwen-Image")
ENCODE_SCRIPT = Path(__file__).parent / "encode_flammen.py"


def load_flammen_dataset():
    rows = []
    with open(TRAIN_JSONL) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "prompt":       r["prompt"],
                "chosen":       str(DATA_DIR / r["image"]),
                "flame_id":     r["flame_id"],
                "caption_type": r["caption_type"],
            })
    ds = datasets.Dataset.from_list(rows)
    ds = ds.cast_column("chosen", datasets.Image())
    return ds


def ensure_cache_exists():
    """Shell out to encode_flammen.py if the cache isn't already on disk."""
    text_p   = CACHE_DIR / "text_embeddings.pt"
    target_p = CACHE_DIR / "target_embeddings.pt"
    if text_p.exists() and target_p.exists():
        print(f"[main] cache already at {CACHE_DIR} — skipping encode stage")
        return
    print(f"[main] no cache at {CACHE_DIR} — launching encode subprocess …")
    cmd = [sys.executable, str(ENCODE_SCRIPT)]
    print(f"[main] $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, env=os.environ).returncode
    if rc != 0:
        raise RuntimeError(f"encode subprocess failed with exit {rc}")
    print("[main] encode subprocess complete; VRAM released.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: cache embeddings in subprocess ───────────────────
    ensure_cache_exists()

    # ── Stage 2: train (transformer only, no encoders) ────────────
    print("\n[main] loading dataset …")
    raw_dataset = load_flammen_dataset()
    print(f"[main]   {len(raw_dataset)} rows ({len(set(raw_dataset['flame_id']))} unique flames)")

    print("\n[main] loading QwenImageAdapter (load_encoders=False) …")
    adapter = QwenImageAdapter(QWEN_PATH, load_encoders=False)

    print(f"[main] loading cached embeddings from {CACHE_DIR} …")
    text_emb, target_emb, _ = cache_embeddings(
        raw_dataset, adapter, cache_dir=str(CACHE_DIR),
    )
    print(f"[main]   loaded text={len(text_emb)} targets={len(target_emb)}")

    train_dataset = EditingDataset(
        raw_dataset,
        cached_text_embeddings=text_emb,
        cached_target_embeddings=target_emb,
    )

    config = TrainingConfig(
        output_dir=str(OUTPUT_DIR),
        num_epochs=8,
        batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        mixed_precision="bf16",
        gradient_checkpointing=True,
        optimizer="adafactor",
        lr_scheduler="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        save_on_epoch_end=True,
        seed=42,
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )

    print(f"\n[main] building trainer (output_dir={OUTPUT_DIR})")
    trainer = AtelierTrainer(
        adapter=adapter,
        config=config,
        loss_fn=FlowMatchingLoss(),
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    print("\n[main] *** starting training ***")
    trainer.train()

    final_lora = OUTPUT_DIR / "flammen-aesthetic-v1"
    print(f"\n[main] saving LoRA → {final_lora}")
    trainer.save_model(str(final_lora))
    print("[main] done.")


if __name__ == "__main__":
    main()
