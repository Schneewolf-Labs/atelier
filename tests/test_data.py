"""Tests for data modules: datasets, collators, and utilities."""

import os
import tempfile

import numpy as np
import torch
from PIL import Image

from atelier.data.cache import _load_cache, _save_cache, _to_pil
from atelier.data.editing import (
    EditingCollator,
    EditingDataset,
    calculate_dimensions,
    prepare_image,
)
from atelier.data.generation import GenerationCollator, _image_to_tensor

# ---- calculate_dimensions ----

class TestCalculateDimensions:
    def test_square_aspect_ratio(self):
        w, h = calculate_dimensions(1024 * 1024, 1.0)
        assert w == h
        assert w % 32 == 0

    def test_landscape_ratio(self):
        w, h = calculate_dimensions(1024 * 1024, 2.0)
        assert w > h
        assert w % 32 == 0
        assert h % 32 == 0

    def test_portrait_ratio(self):
        w, h = calculate_dimensions(1024 * 1024, 0.5)
        assert h > w
        assert w % 32 == 0
        assert h % 32 == 0

    def test_result_near_target_area(self):
        target = 1024 * 1024
        w, h = calculate_dimensions(target, 1.5)
        actual_area = w * h
        # Allow 10% tolerance due to rounding to 32
        assert abs(actual_area - target) / target < 0.1


# ---- prepare_image ----

class TestPrepareImage:
    def test_pil_image(self):
        img = Image.new("RGB", (64, 48))
        tensor = prepare_image(img, 32, 32)
        assert tensor.shape == (3, 32, 32)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_numpy_array(self):
        arr = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        tensor = prepare_image(arr, 16, 16)
        assert tensor.shape == (3, 16, 16)

    def test_string_path(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (64, 48), color="red")
            img.save(f.name)
            tensor = prepare_image(f.name, 32, 32)
            assert tensor.shape == (3, 32, 32)
            os.unlink(f.name)

    def test_rgba_converted(self):
        img = Image.new("RGBA", (64, 48))
        tensor = prepare_image(img, 32, 32)
        assert tensor.shape == (3, 32, 32)  # Should be RGB, not RGBA


# ---- _image_to_tensor ----

class TestImageToTensor:
    def test_pil_image(self):
        img = Image.new("RGB", (64, 48))
        tensor = _image_to_tensor(img, 32)
        assert tensor.shape == (3, 32, 32)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_numpy_array(self):
        arr = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        tensor = _image_to_tensor(arr, 16)
        assert tensor.shape == (3, 16, 16)

    def test_string_path(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (64, 48), color="blue")
            img.save(f.name)
            tensor = _image_to_tensor(f.name, 32)
            assert tensor.shape == (3, 32, 32)
            os.unlink(f.name)


# ---- GenerationCollator ----

class TestGenerationCollator:
    def test_stacks_tensors(self):
        collator = GenerationCollator()
        examples = [
            {"image": torch.randn(3, 32, 32), "label": 0},
            {"image": torch.randn(3, 32, 32), "label": 1},
        ]
        batch = collator(examples)
        assert batch["image"].shape == (2, 3, 32, 32)
        assert batch["label"] == [0, 1]

    def test_preserves_non_tensor(self):
        collator = GenerationCollator()
        examples = [
            {"text": "hello", "val": torch.tensor(1.0)},
            {"text": "world", "val": torch.tensor(2.0)},
        ]
        batch = collator(examples)
        assert batch["text"] == ["hello", "world"]
        assert batch["val"].shape == (2,)


# ---- EditingCollator ----

class TestEditingCollator:
    def test_stacks_latent_tensors(self):
        collator = EditingCollator()
        examples = [
            {"target_latents": torch.randn(4, 8, 8), "prompt_embeds": torch.randn(10, 64)},
            {"target_latents": torch.randn(4, 8, 8), "prompt_embeds": torch.randn(10, 64)},
        ]
        batch = collator(examples)
        assert batch["target_latents"].shape == (2, 4, 8, 8)

    def test_pads_variable_length_embeds(self):
        collator = EditingCollator()
        examples = [
            {
                "prompt_embeds": torch.randn(8, 64),
                "prompt_embeds_mask": torch.ones(8),
            },
            {
                "prompt_embeds": torch.randn(12, 64),
                "prompt_embeds_mask": torch.ones(12),
            },
        ]
        batch = collator(examples)
        # Should be padded to max length (12)
        assert batch["prompt_embeds"].shape == (2, 12, 64)
        assert batch["prompt_embeds_mask"].shape == (2, 12)

    def test_no_latents(self):
        collator = EditingCollator()
        examples = [
            {"prompt_embeds": torch.randn(5, 64)},
            {"prompt_embeds": torch.randn(5, 64)},
        ]
        batch = collator(examples)
        assert "target_latents" not in batch
        assert batch["prompt_embeds"].shape == (2, 5, 64)

    def test_equal_length_no_padding_needed(self):
        collator = EditingCollator()
        examples = [
            {"prompt_embeds": torch.randn(10, 64)},
            {"prompt_embeds": torch.randn(10, 64)},
        ]
        batch = collator(examples)
        assert batch["prompt_embeds"].shape == (2, 10, 64)


# ---- EditingDataset ----

class TestEditingDataset:
    def _make_hf_dataset(self, n=5):
        """Create a mock HF dataset-like object."""

        class FakeDataset:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

            def select(self, indices):
                return FakeDataset([self._data[i] for i in indices])

        data = []
        for i in range(n):
            data.append({
                "prompt": f"edit instruction {i}",
                "chosen": Image.new("RGB", (64, 64), color="green"),
                "rejected": Image.new("RGB", (64, 64), color="red"),
            })
        return FakeDataset(data)

    def test_length(self):
        ds = self._make_hf_dataset(5)
        edit_ds = EditingDataset(ds)
        assert len(edit_ds) == 5

    def test_max_samples(self):
        ds = self._make_hf_dataset(10)
        edit_ds = EditingDataset(ds, max_samples=3)
        assert len(edit_ds) == 3

    def test_getitem_no_cache(self):
        ds = self._make_hf_dataset(3)
        edit_ds = EditingDataset(ds)
        item = edit_ds[0]
        assert "prompt" in item
        assert "target_image" in item
        assert "control_image" in item

    def test_getitem_with_cache(self):
        ds = self._make_hf_dataset(3)
        text_cache = {
            f"sample_{i}": {
                "prompt_embeds": torch.randn(10, 64),
                "prompt_embeds_mask": torch.ones(10),
            }
            for i in range(3)
        }
        target_cache = {f"sample_{i}": torch.randn(4, 8, 8) for i in range(3)}
        control_cache = {f"sample_{i}": torch.randn(4, 8, 8) for i in range(3)}

        edit_ds = EditingDataset(
            ds,
            cached_text_embeddings=text_cache,
            cached_target_embeddings=target_cache,
            cached_control_embeddings=control_cache,
        )
        item = edit_ds[0]
        assert "prompt_embeds" in item
        assert "target_latents" in item
        assert "control_latents" in item
        assert "prompt" not in item


# ---- Cache utilities ----

class TestCacheUtils:
    def test_to_pil_from_pil(self):
        img = Image.new("RGBA", (32, 32))
        result = _to_pil(img)
        assert result.mode == "RGB"

    def test_to_pil_from_numpy(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = _to_pil(arr)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_to_pil_from_string(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (32, 32))
            img.save(f.name)
            result = _to_pil(f.name)
            assert isinstance(result, Image.Image)
            assert result.mode == "RGB"
            os.unlink(f.name)

    def test_save_and_load_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            text = {"sample_0": {"embeds": torch.randn(10, 64)}}
            targets = {"sample_0": torch.randn(4, 8, 8)}
            controls = {"sample_0": torch.randn(4, 8, 8)}

            _save_cache(tmpdir, text, targets, controls)

            # Files should exist
            assert os.path.exists(os.path.join(tmpdir, "text_embeddings.pt"))
            assert os.path.exists(os.path.join(tmpdir, "target_embeddings.pt"))
            assert os.path.exists(os.path.join(tmpdir, "control_embeddings.pt"))

            # Load and verify
            loaded = _load_cache(tmpdir)
            assert loaded is not None
            loaded_text, loaded_targets, loaded_controls = loaded
            assert "sample_0" in loaded_text
            assert torch.equal(loaded_targets["sample_0"], targets["sample_0"])

    def test_load_cache_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _load_cache(tmpdir)
            assert result is None
