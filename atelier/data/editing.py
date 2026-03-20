import math

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class EditingDataset(Dataset):
    """Dataset for image editing training with paired before/after images.

    Expects a HuggingFace dataset with columns:
        - prompt: str (edit instruction)
        - chosen: PIL.Image (target / improved image)
        - rejected: PIL.Image (source / control image)

    Supports pre-computed embeddings via cached_* dicts (keyed by "sample_{idx}").
    When embeddings are provided, raw images/prompts are not returned.
    """

    def __init__(
        self,
        dataset,
        cached_text_embeddings=None,
        cached_target_embeddings=None,
        cached_control_embeddings=None,
        max_samples=None,
    ):
        self.dataset = dataset
        self.cached_text = cached_text_embeddings or {}
        self.cached_targets = cached_target_embeddings or {}
        self.cached_controls = cached_control_embeddings or {}

        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        # Filter to samples with cached embeddings if caching was used
        if self.cached_text:
            valid = [i for i in range(len(self.dataset)) if f"sample_{i}" in self.cached_text]
            self.dataset = self.dataset.select(valid)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        key = f"sample_{idx}"

        result = {}

        # Text embeddings
        if key in self.cached_text:
            text_data = self.cached_text[key]
            result["prompt_embeds"] = text_data["prompt_embeds"]
            result["prompt_embeds_mask"] = text_data.get("prompt_embeds_mask")
        else:
            result["prompt"] = item["prompt"]

        # Target image latents (chosen)
        if key in self.cached_targets:
            result["target_latents"] = self.cached_targets[key]
        else:
            result["target_image"] = item["chosen"]

        # Control image latents (rejected / source)
        if key in self.cached_controls:
            result["control_latents"] = self.cached_controls[key]
        else:
            result["control_image"] = item["rejected"]

        return result


class EditingCollator:
    """Collates editing samples into batches with padding for variable-length embeddings."""

    def __call__(self, examples):
        batch = {}

        # Stack latent tensors
        for key in ("target_latents", "control_latents"):
            tensors = [ex[key] for ex in examples if key in ex]
            if tensors:
                batch[key] = torch.stack(tensors)

        # Pad and stack prompt embeddings (variable sequence length)
        embeds = [ex["prompt_embeds"] for ex in examples if "prompt_embeds" in ex]
        if embeds:
            max_seq_len = max(e.shape[0] for e in embeds)
            padded_embeds = []
            padded_masks = []

            for i, emb in enumerate(embeds):
                seq_len = emb.shape[0]
                if seq_len < max_seq_len:
                    padding = torch.zeros(max_seq_len - seq_len, emb.shape[1], dtype=emb.dtype)
                    padded_embeds.append(torch.cat([emb, padding], dim=0))
                else:
                    padded_embeds.append(emb)

                mask = examples[i].get("prompt_embeds_mask")
                if mask is not None:
                    if mask.shape[0] < max_seq_len:
                        mask_pad = torch.zeros(max_seq_len - mask.shape[0], dtype=mask.dtype)
                        padded_masks.append(torch.cat([mask, mask_pad], dim=0))
                    else:
                        padded_masks.append(mask)

            batch["prompt_embeds"] = torch.stack(padded_embeds)
            if padded_masks:
                batch["prompt_embeds_mask"] = torch.stack(padded_masks)

        return batch


def calculate_dimensions(target_area, ratio):
    """Calculate dimensions fitting target area while maintaining aspect ratio, rounded to 32."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


def prepare_image(image, height, width):
    """Convert a PIL image to a normalized tensor [C, H, W] in [-1, 1]."""
    if not isinstance(image, Image.Image):
        if isinstance(image, str):
            image = Image.open(image)
        else:
            image = Image.fromarray(np.uint8(image))
    image = image.convert("RGB").resize((width, height), Image.LANCZOS)

    tensor = torch.from_numpy(np.array(image).astype(np.float32))
    tensor = tensor / 127.5 - 1.0
    tensor = tensor.permute(2, 0, 1)  # [C, H, W]
    return tensor
