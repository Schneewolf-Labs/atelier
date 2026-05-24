"""
Encode-only subprocess for the flammen LoRA training pipeline.

Loads ONLY the text encoder + VAE (no transformer), runs cache_embeddings
over the flammen dataset, writes the cache to disk, and exits. Process
exit releases all encoder VRAM cleanly to the OS — which is more
reliable than ``del`` + ``empty_cache`` for multi-tower vision-language
text encoders that retain refs in tokenizer/processor wiring.

The companion train_flammen_lora.py script invokes this via subprocess
before loading the transformer, then loads the cache from disk in a
fresh process where no encoder ever existed.
"""
import json
import os
from pathlib import Path

import datasets

from atelier.adapters import QwenImageAdapter
from atelier.data import cache_embeddings

DATA_DIR    = Path(os.path.expanduser("~/flammen-lora-dataset"))
TRAIN_JSONL = DATA_DIR / "train.jsonl"
OUTPUT_DIR  = Path(os.path.expanduser("~/flammen-lora-output"))
CACHE_DIR   = OUTPUT_DIR / "cache"
QWEN_PATH   = os.environ.get("QWEN_IMAGE_PATH", "Qwen/Qwen-Image")


def load_flammen_dataset():
    rows = []
    with open(TRAIN_JSONL) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "prompt": r["prompt"],
                "chosen": str(DATA_DIR / r["image"]),
            })
    ds = datasets.Dataset.from_list(rows)
    ds = ds.cast_column("chosen", datasets.Image())
    return ds


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[encode] loading dataset from {TRAIN_JSONL} …")
    ds = load_flammen_dataset()
    print(f"[encode]   {len(ds)} rows")

    print("[encode] loading QwenImageAdapter — encoders only "
          "(load_transformer=False) …")
    adapter = QwenImageAdapter(QWEN_PATH, load_transformer=False)

    print(f"[encode] caching embeddings → {CACHE_DIR}")
    text_emb, target_emb, _ = cache_embeddings(
        ds, adapter, cache_dir=str(CACHE_DIR), target_area=1024 * 1024,
    )
    print(f"[encode] cached: text={len(text_emb)} targets={len(target_emb)}")
    print("[encode] done. exiting so OS releases encoder VRAM.")


if __name__ == "__main__":
    main()
