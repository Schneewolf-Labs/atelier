"""Tests for atelier.train CLI module + atelier.registry."""

import json
import os
import tempfile

import pytest

from atelier import registry
from atelier.train import (
    _deep_merge,
    _parse_overrides,
    build_peft_config,
    build_training_config,
    load_dataset_from_spec,
    load_yaml,
)

# ---- registry ----

class TestRegistry:
    def test_get_adapter_class_known(self):
        cls = registry.get_adapter_class("qwen_image")
        assert cls.__name__ == "QwenImageAdapter"

    def test_get_adapter_class_full_spec(self):
        cls = registry.get_adapter_class("atelier.adapters.base:ModelAdapter")
        assert cls.__name__ == "ModelAdapter"

    def test_get_adapter_class_unknown(self):
        with pytest.raises(KeyError, match="unknown adapter"):
            registry.get_adapter_class("frobnicate_v9")

    def test_get_loss_class_known(self):
        cls = registry.get_loss_class("flow_matching")
        assert cls.__name__ == "FlowMatchingLoss"

    def test_get_loss_class_unknown(self):
        with pytest.raises(KeyError, match="unknown loss"):
            registry.get_loss_class("not_a_loss")


# ---- CLI helpers ----

class TestParseOverrides:
    def test_simple_string(self):
        assert _parse_overrides(["foo=bar"]) == {"foo": "bar"}

    def test_json_int(self):
        assert _parse_overrides(["training.num_epochs=4"]) == {"training": {"num_epochs": 4}}

    def test_json_float(self):
        assert _parse_overrides(["training.lr=1.5e-4"]) == {"training": {"lr": 0.00015}}

    def test_json_list(self):
        result = _parse_overrides(['peft.target=["a","b"]'])
        assert result == {"peft": {"target": ["a", "b"]}}

    def test_json_bool(self):
        assert _parse_overrides(["training.bf16=true"]) == {"training": {"bf16": True}}

    def test_nested_three_deep(self):
        assert _parse_overrides(["a.b.c=1"]) == {"a": {"b": {"c": 1}}}

    def test_multiple(self):
        assert _parse_overrides(["a=1", "b=2"]) == {"a": 1, "b": 2}

    def test_missing_value(self):
        with pytest.raises(SystemExit):
            _parse_overrides(["nokey"])


class TestDeepMerge:
    def test_simple(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested(self):
        result = _deep_merge(
            {"training": {"lr": 1e-4, "epochs": 8}},
            {"training": {"epochs": 4}},
        )
        assert result == {"training": {"lr": 1e-4, "epochs": 4}}

    def test_overriding_value_with_dict(self):
        assert _deep_merge({"a": 1}, {"a": {"x": 2}}) == {"a": {"x": 2}}

    def test_base_unchanged(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestLoadYaml:
    def test_round_trip(self):
        pytest.importorskip("yaml")
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  adapter: qwen_image\n  pretrained: foo\n")
            path = f.name
        try:
            cfg = load_yaml(path)
            assert cfg["model"]["adapter"] == "qwen_image"
        finally:
            os.unlink(path)


# ---- build_* helpers ----

class TestBuildPeftConfig:
    def test_none_returns_none(self):
        assert build_peft_config(None) is None
        assert build_peft_config({}) is None

    def test_lora(self):
        cfg = build_peft_config({
            "type": "lora", "r": 16, "lora_alpha": 32,
            "target_modules": ["to_q", "to_v"],
        })
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        # init_lora_weights default should be gaussian per atelier convention
        assert cfg.init_lora_weights == "gaussian"

    def test_lora_explicit_init(self):
        cfg = build_peft_config({
            "type": "lora", "r": 8, "lora_alpha": 16,
            "target_modules": ["to_q"], "init_lora_weights": True,
        })
        assert cfg.init_lora_weights is True

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="unsupported peft type"):
            build_peft_config({"type": "ia3"})


class TestBuildTrainingConfig:
    def test_defaults(self):
        cfg = build_training_config(None)
        assert cfg.num_epochs == 3  # TrainingConfig default

    def test_override(self):
        cfg = build_training_config({"num_epochs": 12, "batch_size": 4})
        assert cfg.num_epochs == 12
        assert cfg.batch_size == 4


class TestLoadDatasetFromSpec:
    def test_jsonl(self):
        # Write a tiny JSONL with a non-image 'image' value — we just want to
        # confirm the loader builds a Dataset; image-casting opens the file
        # lazily so a non-existent path won't error here.
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "x", "image": "/tmp/nope.png"}) + "\n")
            f.write(json.dumps({"prompt": "y", "image": "/tmp/nope2.png"}) + "\n")
            path = f.name
        try:
            ds = load_dataset_from_spec({"jsonl": path})
            assert len(ds) == 2
            assert "chosen" in ds.column_names  # 'image' was normalized to 'chosen'
            assert "prompt" in ds.column_names
        finally:
            os.unlink(path)

    def test_jsonl_max_samples(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                f.write(json.dumps({"prompt": f"x{i}", "image": "/tmp/x.png"}) + "\n")
            path = f.name
        try:
            ds = load_dataset_from_spec({"jsonl": path, "max_samples": 2})
            assert len(ds) == 2
        finally:
            os.unlink(path)

    def test_no_source(self):
        with pytest.raises(ValueError, match="must include one of"):
            load_dataset_from_spec({})
