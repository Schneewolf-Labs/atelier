"""Tests for the ModelAdapter base class."""

import torch

from atelier.adapters.base import ModelAdapter


class TestModelAdapterBase:
    def test_model_not_implemented(self):
        adapter = ModelAdapter()
        try:
            _ = adapter.model
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_noise_scheduler_not_implemented(self):
        adapter = ModelAdapter()
        try:
            _ = adapter.noise_scheduler
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_encode_images_not_implemented(self):
        adapter = ModelAdapter()
        try:
            adapter.encode_images([])
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_encode_image_tensor_not_implemented(self):
        adapter = ModelAdapter()
        try:
            adapter.encode_image_tensor(torch.randn(1, 3, 32, 32))
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_encode_text_not_implemented(self):
        adapter = ModelAdapter()
        try:
            adapter.encode_text(["test"])
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_sample_timesteps_not_implemented(self):
        adapter = ModelAdapter()
        try:
            adapter.sample_timesteps(1, "cpu")
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_add_noise_not_implemented(self):
        adapter = ModelAdapter()
        try:
            adapter.add_noise(None, None, None, None)
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_compute_target_not_implemented(self):
        adapter = ModelAdapter()
        try:
            adapter.compute_target(None, None, None)
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_forward_not_implemented(self):
        adapter = ModelAdapter()
        try:
            adapter.forward(None, None, None, None)
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_save_lora_not_implemented(self):
        adapter = ModelAdapter()
        try:
            adapter.save_lora(None, None)
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_save_model_not_implemented(self):
        adapter = ModelAdapter()
        try:
            adapter.save_model(None, None)
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass

    def test_free_encoders_default(self):
        adapter = ModelAdapter()
        # Should not raise - default implementation is a no-op
        adapter.free_encoders()


class TestQwenImageAdapterImport:
    """QwenImageAdapter cannot be instantiated without the Qwen-Image weights
    (~20 GB download), so we limit testing to public-API import shape +
    inheritance. Behavioral tests run through smoke-test scripts."""

    def test_importable_via_package(self):
        from atelier.adapters import QwenImageAdapter as Adapter

        assert Adapter.__name__ == "QwenImageAdapter"

    def test_inherits_model_adapter(self):
        from atelier.adapters import ModelAdapter, QwenImageAdapter

        assert issubclass(QwenImageAdapter, ModelAdapter)

    def test_defines_required_overrides(self):
        from atelier.adapters import QwenImageAdapter

        for method in ("encode_images", "encode_text", "sample_timesteps",
                       "add_noise", "compute_target", "forward",
                       "save_lora", "save_model"):
            assert method in QwenImageAdapter.__dict__, (
                f"QwenImageAdapter must override {method}()"
            )
