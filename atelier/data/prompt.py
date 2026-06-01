"""Prompt-only dataset + collator for Flow-GRPO rollout.

The mirror of the text-side ``tokenize_prompt``: there are NO target images in
GRPO. Each item is a prompt (plus any pass-through columns the reward needs).
Text embeddings CAN still be cached up front — prompts are reused across the
G-group and across epochs — but unlike the SFT path the VAE *decoder* must stay
resident through training, so caching here only touches the text encoder.
"""

import logging
import os

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PromptDataset(Dataset):
    """Prompt-only dataset for GRPO rollout.

    Args:
        raw_dataset: a HuggingFace ``datasets.Dataset`` (or any column-style
            dataset exposing ``column_names`` + integer indexing).
        adapter: a ModelAdapter — used to cache text embeddings up front when
            ``cache_embeddings`` is True.
        prompt_column: column holding the text prompt.
        cache_dir: optional directory to persist/restore cached text embeddings.
        cache_embeddings: when True, pre-encode every prompt with ``adapter.encode_text``
            so the text encoder can be freed before rollout begins.
        max_samples: optional cap on dataset size.
    """

    def __init__(
        self,
        raw_dataset,
        adapter,
        *,
        prompt_column: str = "prompt",
        cache_dir: "str | None" = None,
        cache_embeddings: bool = False,
        max_samples: "int | None" = None,
    ) -> None:
        if max_samples and hasattr(raw_dataset, "select"):
            raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))

        self.raw_dataset = raw_dataset
        self.adapter = adapter
        self.prompt_column = prompt_column
        self.cache_dir = cache_dir

        column_names = list(getattr(raw_dataset, "column_names", []))
        if prompt_column not in column_names and len(raw_dataset) and prompt_column not in raw_dataset[0]:
            raise ValueError(f"prompt_column {prompt_column!r} not found in dataset columns {column_names}")
        # Extra columns are passed through to the reward function untouched.
        self.extra_columns = [c for c in column_names if c != prompt_column]

        self.text_embeddings = None
        if cache_embeddings:
            self.text_embeddings = self._build_text_cache()

    def _build_text_cache(self):
        if self.cache_dir:
            path = os.path.join(self.cache_dir, "prompt_text_embeddings.pt")
            if os.path.exists(path):
                logger.info("Loading cached prompt embeddings from %s", path)
                return torch.load(path, weights_only=False)

        embeddings = []
        with torch.no_grad():
            for idx in range(len(self.raw_dataset)):
                prompt = self._prompt_at(idx)
                encoded = self.adapter.encode_text([prompt], device=self.adapter.device)
                embeddings.append(
                    {k: v[0].cpu() if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}
                )

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(embeddings, os.path.join(self.cache_dir, "prompt_text_embeddings.pt"))
            logger.info("Saved %d prompt embeddings to %s", len(embeddings), self.cache_dir)
        return embeddings

    def _prompt_at(self, idx):
        return self.raw_dataset[idx][self.prompt_column]

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        row = self.raw_dataset[idx]
        item = {"prompt": row[self.prompt_column]}
        for col in self.extra_columns:
            item[col] = row[col]
        if self.text_embeddings is not None:
            item["text_embeddings"] = self.text_embeddings[idx]
        return item


class PromptCollator:
    """Collate prompt items into a batch dict for FlowGRPOTrainer.

    Produces:
        prompts:          list[str], length B
        columns:          dict[str, list], extra pass-through columns aligned to prompts
        text_embeddings:  dict[str, Tensor] of stacked cached embeddings, or None
    """

    def __call__(self, examples):
        prompts = [ex["prompt"] for ex in examples]

        reserved = {"prompt", "text_embeddings"}
        columns = {}
        for key in examples[0]:
            if key in reserved:
                continue
            columns[key] = [ex[key] for ex in examples]

        text_embeddings = None
        if "text_embeddings" in examples[0]:
            keys = examples[0]["text_embeddings"].keys()
            text_embeddings = {}
            for k in keys:
                values = [ex["text_embeddings"][k] for ex in examples]
                if isinstance(values[0], torch.Tensor):
                    text_embeddings[k] = torch.stack(values)
                else:
                    text_embeddings[k] = values

        return {"prompts": prompts, "columns": columns, "text_embeddings": text_embeddings}
