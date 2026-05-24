import logging
import os

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .editing import calculate_dimensions

logger = logging.getLogger(__name__)


def cache_embeddings(dataset, adapter, cache_dir=None, target_area=1024 * 1024, max_samples=None):
    """Pre-compute text and image embeddings for a dataset.

    Uses the adapter's encode_text() and encode_images() methods.
    Results are saved to disk if cache_dir is provided, and loaded
    from cache on subsequent calls.

    Args:
        dataset: HuggingFace dataset. Required columns: prompt, chosen.
            If a ``rejected`` column is present, control embeddings are
            also computed (editing / DPO mode). If absent, the dataset
            is treated as text-to-image (target only).
        adapter: A ModelAdapter with encode_text() and encode_images().
        cache_dir: Directory to save/load cached embeddings.
        target_area: Target pixel area for image resizing.
        max_samples: Limit number of samples to process.

    Returns:
        (text_embeddings, target_embeddings, control_embeddings) dicts
        keyed by "sample_{idx}". ``control_embeddings`` is an empty dict
        when the dataset has no ``rejected`` column.
    """
    has_control = "rejected" in getattr(dataset, "column_names", [])

    # Try loading from cache
    if cache_dir:
        cached = _load_cache(cache_dir, has_control=has_control)
        if cached is not None:
            return cached

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    device = adapter.model.device if hasattr(adapter.model, "device") else "cpu"

    text_embeddings = {}
    target_embeddings = {}
    control_embeddings = {}

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Pre-computing embeddings"):
            item = dataset[idx]
            key = f"sample_{idx}"

            # Prepare images
            target_image = _to_pil(item["chosen"])
            control_image = _to_pil(item["rejected"]) if has_control else None
            ref_image = control_image or target_image
            width, height = calculate_dimensions(target_area, ref_image.width / ref_image.height)

            # Encode text. Pass control_image as the reference image
            # for adapters whose text encoder is vision-conditioned
            # (e.g. Qwen-Image-Edit); pure-T2I adapters ignore it.
            prompt = item.get("prompt", "")
            encode_kwargs = {"device": device}
            if has_control:
                encode_kwargs["images"] = [control_image]
            text_data = adapter.encode_text([prompt], **encode_kwargs)
            text_embeddings[key] = {k: v[0].cpu() if isinstance(v, torch.Tensor) else v for k, v in text_data.items()}

            # Encode images
            target_latents = adapter.encode_images([target_image], height=height, width=width, device=device)
            target_embeddings[key] = target_latents[0].cpu()
            if has_control:
                control_latents = adapter.encode_images([control_image], height=height, width=width, device=device)
                control_embeddings[key] = control_latents[0].cpu()

            if idx % 100 == 0:
                torch.cuda.empty_cache()

    # Save to disk
    if cache_dir:
        _save_cache(cache_dir, text_embeddings, target_embeddings, control_embeddings)

    return text_embeddings, target_embeddings, control_embeddings


def _load_cache(cache_dir, has_control=True):
    """Load cached embeddings from disk if all required files exist."""
    text_path = os.path.join(cache_dir, "text_embeddings.pt")
    target_path = os.path.join(cache_dir, "target_embeddings.pt")
    control_path = os.path.join(cache_dir, "control_embeddings.pt")

    required = [text_path, target_path] + ([control_path] if has_control else [])
    if all(os.path.exists(p) for p in required):
        logger.info("Loading cached embeddings from %s", cache_dir)
        text = torch.load(text_path, weights_only=False)
        targets = torch.load(target_path, weights_only=False)
        controls = (
            torch.load(control_path, weights_only=False)
            if has_control and os.path.exists(control_path)
            else {}
        )
        logger.info("Loaded %d cached embeddings", len(text))
        return text, targets, controls

    return None


def _save_cache(cache_dir, text_embeddings, target_embeddings, control_embeddings):
    """Save embeddings to disk. Control file is only written when non-empty."""
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(text_embeddings, os.path.join(cache_dir, "text_embeddings.pt"))
    torch.save(target_embeddings, os.path.join(cache_dir, "target_embeddings.pt"))
    if control_embeddings:
        torch.save(control_embeddings, os.path.join(cache_dir, "control_embeddings.pt"))
    logger.info("Saved embeddings to %s", cache_dir)


def _to_pil(image):
    """Convert various image types to RGB PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return Image.fromarray(np.uint8(image)).convert("RGB")
