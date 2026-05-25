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


class TestStripPeftPrefix:
    """Regression tests for the LoRA-save key fix.

    Before this fix, ``QwenImageAdapter.save_lora`` wrote keys like
    ``transformer.base_model.model.transformer_blocks.0.attn.to_k.lora.down.weight``
    that ``pipe.load_lora_weights`` couldn't match against the
    transformer's actual module tree. The strip_peft_prefix helper +
    dropping convert_state_dict_to_diffusers from the save path produces
    the format diffusers + PEFT actually load.
    """

    def test_strip_simple(self):
        from atelier.adapters.qwen_image import strip_peft_prefix
        out = strip_peft_prefix({
            "base_model.model.foo.lora_A.weight": 1,
            "base_model.model.bar.lora_B.weight": 2,
        })
        assert out == {"foo.lora_A.weight": 1, "bar.lora_B.weight": 2}

    def test_strip_no_prefix_passthrough(self):
        from atelier.adapters.qwen_image import strip_peft_prefix
        sd = {"foo.weight": 1, "bar.weight": 2}
        assert strip_peft_prefix(sd) == sd

    def test_strip_preserves_real_module_paths(self):
        """Ensure the prefix only strips the wrapper, not legitimate
        module path components that contain similar substrings."""
        from atelier.adapters.qwen_image import strip_peft_prefix
        out = strip_peft_prefix({
            "base_model.model.transformer_blocks.0.attn.to_k.lora_A.weight": 1,
            "base_model.model.transformer_blocks.41.attn.to_out.0.lora_B.weight": 2,
        })
        assert "transformer_blocks.0.attn.to_k.lora_A.weight" in out
        assert "transformer_blocks.41.attn.to_out.0.lora_B.weight" in out
        assert not any("base_model" in k for k in out)

    def test_strip_does_not_touch_lora_A_B_suffixes(self):
        """The fix is about the PREFIX. The lora_A / lora_B suffix must
        survive untouched — that's the PEFT format diffusers wants."""
        from atelier.adapters.qwen_image import strip_peft_prefix
        out = strip_peft_prefix({"base_model.model.x.lora_A.weight": 1})
        key = next(iter(out))
        assert key.endswith(".lora_A.weight"), \
            "lora_A suffix must remain — diffusers expects PEFT format, not legacy .lora.down.weight"


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
