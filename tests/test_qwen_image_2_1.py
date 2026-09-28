"""Tests for the native Qwen-Image 2.1 port (atelier.models.qwen_image_2_1) and its adapter.

The DiT is pinned to stable-diffusion.cpp by a golden file (see
scripts/sdcpp_parity/); the rest checks the pieces around it — sequence layout,
RoPE, VAE key conversion, Qwen3-VL conditioning mechanics, and the adapter's
training / LoRA-export path — without any checkpoint downloads.
"""

import json
import math
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from atelier.models.qwen_image_2_1 import (
    TEXT,
    QwenImage21Config,
    QwenImage21Transformer,
    apply_rope,
    attention_mask,
    build_layout,
    build_prompt,
    convert_vae_key,
    load_transformer,
    rope_frequencies,
    token_types_from_ids,
)

from .qwen_image_2_1_cases import CASES, build_case

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "qwen_image_2_1_sdcpp.json")


# ---------------------------------------------------------------------------
# Parity with stable-diffusion.cpp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_dit_matches_sdcpp_golden(name):
    """sd.cpp's QwenImage21Runner output for identical weights / inputs (fp16 GELU table → ~3e-4)."""
    with open(GOLDEN) as f:
        golden = json.load(f)
    model, x, t, context, token_types, refs = build_case(name)
    with torch.no_grad():
        ours = model(x, t, context, token_types, refs)
    ref = torch.tensor(golden[name]).reshape(ours.shape)
    rel = ((ours - ref).abs().max() / ref.abs().max()).item()
    assert rel < 1e-3, f"{name}: rel diff {rel:.2e} vs sd.cpp"


# ---------------------------------------------------------------------------
# Layout, mask, RoPE
# ---------------------------------------------------------------------------


class TestLayout:
    def test_text_to_image(self):
        layout = build_layout([0, 0, 0], [(2, 4)])
        assert [(s.start, s.end, s.image_index) for s in layout.segments] == [(0, 3, TEXT), (3, 11, 0)]
        assert layout.prefix_length == 3
        assert layout.positions[:3] == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        # target: axis0 = next text position, centred grid (h - ceil(H/2), w - ceil(W/2))
        assert layout.positions[3] == (3, -1, -2)
        assert layout.positions[-1] == (3, 0, 1)

    def test_reference_replaces_its_slots(self):
        # 2 slot tokens ↔ a 2x4 latent (one VLM token per 2x2 latent pixels)
        layout = build_layout([0, 1, 1, 0], [(2, 4), (4, 4)])
        segs = [(s.start, s.end, s.context_start, s.image_index) for s in layout.segments]
        assert segs == [(0, 1, 0, TEXT), (1, 9, 1, 0), (9, 10, 3, TEXT), (10, 26, 4, 1)]
        # the reference advances the position counter by max(H, W)
        assert layout.positions[9] == (1 + 4, 1 + 4, 1 + 4)
        assert layout.prefix_length == 10

    def test_slot_count_must_match_latent(self):
        with pytest.raises(ValueError, match="matching sizes"):
            build_layout([0, 1, 1, 1, 0], [(2, 4), (4, 4)])

    def test_missing_reference_slots(self):
        with pytest.raises(ValueError, match="missing reference"):
            build_layout([0, 0], [(2, 2), (4, 4)])

    def test_mask_is_block_causal(self):
        layout = build_layout([0, 1, 1, 0], [(2, 4), (2, 2)])
        m = attention_mask(layout)
        text0, ref, text1, target = 0, slice(1, 9), 9, slice(10, 14)
        assert m[text0, text0] and not m[text0, 1], "text is causal"
        assert m[ref][:, ref].all(), "a reference attends bidirectionally within itself"
        assert m[ref][:, text0].all() and not m[ref][:, text1].any(), "…and to earlier, not later, segments"
        assert m[target].all(), "the target sees the whole sequence"
        assert not m[: layout.prefix_length, target].any(), "the prefix never sees the target"


def test_rope_rotates_adjacent_pairs():
    freqs = rope_frequencies([(3, -1, 2)], (4, 2, 2), 10000.0)
    x = torch.randn(1, 1, 1, 8)
    out = apply_rope(x, freqs)
    angles = [3 * 1.0, 3 * 10000 ** -0.5, -1 * 1.0, 2 * 1.0]  # axis0: 2 pairs, axis1/2: 1 pair each
    for i, a in enumerate(angles):
        x0, x1 = x[..., 2 * i], x[..., 2 * i + 1]
        assert torch.allclose(out[..., 2 * i], x0 * math.cos(a) - x1 * math.sin(a), atol=1e-6)
        assert torch.allclose(out[..., 2 * i + 1], x0 * math.sin(a) + x1 * math.cos(a), atol=1e-6)


# ---------------------------------------------------------------------------
# Model + checkpoint I/O
# ---------------------------------------------------------------------------


def _tiny_model(fused=False):
    torch.manual_seed(0)
    cfg = QwenImage21Config(in_channels=8, out_channels=8, hidden_size=256, context_dim=32, head_dim=128,
                            intermediate_size=64, num_layers=2, fused_mlp=fused)
    model = QwenImage21Transformer(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.08)
    return model


class TestCheckpoint:
    @pytest.mark.parametrize("fused", [False, True])
    def test_config_inferred_from_state_dict(self, fused):
        model = _tiny_model(fused)
        cfg = QwenImage21Config.from_state_dict(model.state_dict())
        assert (cfg.hidden_size, cfg.in_channels, cfg.out_channels, cfg.context_dim, cfg.head_dim,
                cfg.intermediate_size, cfg.num_layers, cfg.fused_mlp) == (256, 8, 8, 32, 128, 64, 2, fused)

    @pytest.mark.parametrize("prefix", ["", "model.diffusion_model."])
    def test_load_transformer_round_trip(self, tmp_path, prefix):
        model = _tiny_model().eval()
        save_file({prefix + k: v.contiguous() for k, v in model.state_dict().items()}, tmp_path / "dit.safetensors")
        loaded = load_transformer(tmp_path / "dit.safetensors", dtype=torch.float32).eval()
        _, x, t, ctx, types, refs = build_case("t2i")
        loaded_sd = loaded.state_dict()
        assert loaded_sd.keys() == model.state_dict().keys()
        assert all(torch.equal(v, loaded_sd[k]) for k, v in model.state_dict().items())
        # Outputs: numerically equal (not bitwise — BLAS may take different paths per buffer layout)
        with torch.no_grad():
            assert torch.allclose(model(x, t, ctx, types, refs), loaded(x, t, ctx, types, refs), atol=1e-6)

    def test_quantized_checkpoint_rejected(self, tmp_path):
        sd = {k: v.contiguous() for k, v in _tiny_model().state_dict().items()}
        sd["img_in.weight"] = sd["img_in.weight"].to(torch.int8)
        save_file(sd, tmp_path / "q.safetensors")
        with pytest.raises(ValueError, match="quantized"):
            load_transformer(tmp_path / "q.safetensors")


def test_gradient_checkpointing_is_transparent():
    model = _tiny_model()
    _, x, t, ctx, types, refs = build_case("edit_ref_first")
    out_plain = model(x, t, ctx, types, refs)
    out_plain.square().mean().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    model.enable_gradient_checkpointing()
    out_ckpt = model(x, t, ctx, types, refs)
    out_ckpt.square().mean().backward()
    assert torch.allclose(out_plain, out_ckpt)
    assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(grads, model.parameters()))


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


class TestVAE:
    """The key mapping is checked against a port of sd.cpp's diffusers→original converter."""

    @staticmethod
    def to_original(name):
        for i in range(5):
            for side in ("encoder", "decoder"):
                enc = side == "encoder"
                old = f"{side}.{'down_blocks' if enc else 'up_blocks'}.{i}."
                new = f"{side}.{'downsamples' if enc else 'upsamples'}.{i}."
                if not name.startswith(old):
                    continue
                name = new + name[len(old):]
                layers = "downsamples." if enc else "upsamples."
                for j in range(2 if enc else 3):
                    orn, nrn = f"{new}resnets.{j}.", f"{new}{layers}{j}."
                    if name.startswith(orn + "conv_shortcut."):
                        name = nrn + "shortcut." + name[len(orn + "conv_shortcut."):]
                    elif name.startswith(orn):
                        name = nrn + "residual." + name[len(orn):]
                sampler = new + ("downsampler." if enc else "upsampler.")
                if name.startswith(sampler):
                    name = new + layers + ("2." if enc else "3.") + name[len(sampler):]
        for a, b in [(".conv_in.", ".conv1."), (".norm_out.", ".head.0."), (".conv_out.", ".head.2."),
                     (".mid_block.attentions.0.", ".middle.1."), (".mid_block.resnets.0.", ".middle.0.residual."),
                     (".mid_block.resnets.1.", ".middle.2.residual.")]:
            name = name.replace(a, b)
        for a, b in [("quant_conv.", "conv1."), ("post_quant_conv.", "conv2.")]:
            if name.startswith(a):
                name = b + name[len(a):]
        if ".residual." in name:
            for a, b in [(".norm1.", ".0."), (".conv1.", ".2."), (".norm2.", ".3."), (".conv2.", ".6.")]:
                name = name.replace(a, b)
        return name

    @pytest.fixture
    def tiny_vae(self):
        pytest.importorskip("diffusers")
        from atelier.models.qwen_image_2_1 import build_vae

        return build_vae(base_dim=8, decoder_base_dim=12, z_dim=4, latents_mean=[0.0] * 4, latents_std=[1.0] * 4)

    def test_key_mapping_inverts_sdcpp(self, tiny_vae):
        for key in tiny_vae.state_dict():
            assert convert_vae_key(self.to_original(key)) == key

    def test_loads_image_export_2d_kernels(self, tiny_vae):
        """Causal 3D convs exported as 2D kernels land in the last temporal tap."""
        from atelier.models.qwen_image_2_1 import build_vae

        original = {}
        for k, v in tiny_vae.state_dict().items():
            v = torch.randn_like(v)
            original["first_stage_model." + self.to_original(k)] = v[:, :, -1] if v.ndim == 5 else v
        vae = build_vae(original, base_dim=8, decoder_base_dim=12, z_dim=4,
                        latents_mean=[0.0] * 4, latents_std=[1.0] * 4)
        w = vae.state_dict()["encoder.conv_in.weight"]
        assert torch.equal(w[:, :, -1], original["first_stage_model.encoder.conv1.weight"])
        assert w[:, :, :-1].abs().sum() == 0

    def test_rejects_unknown_keys(self, tiny_vae):
        from atelier.models.qwen_image_2_1 import convert_vae_state_dict

        with pytest.raises(KeyError):
            convert_vae_state_dict({"encoder.bogus.weight": torch.zeros(1)}, tiny_vae.state_dict())


# ---------------------------------------------------------------------------
# Qwen3-VL conditioning
# ---------------------------------------------------------------------------


def test_prompt_template_matches_sdcpp():
    assert build_prompt("a cat") == (
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
    )
    two = build_prompt("", 2)
    assert "<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|>" in two
    assert two.endswith("<|vision_end|> <|im_end|>\n<|im_start|>assistant\n"), "empty prompt → ' '"


def test_token_types_number_image_runs():
    assert token_types_from_ids([5, 9, 9, 5, 9, 9, 9, 5], 9).tolist() == [0, 1, 1, 0, 2, 2, 2, 0]


@pytest.fixture(scope="module")
def tiny_conditioner():
    pytest.importorskip("torchvision")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    from atelier.models.qwen_image_2_1 import Qwen3VLConditioner

    specials = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
                "<|video_pad|>", "<|endoftext|>"]
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(
        ["system user assistant Comprehend and analyze the provided prompt. a cat <image1>"] * 4,
        trainers.BpeTrainer(vocab_size=300, special_tokens=specials,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2,
                                              image_mean=[0.5] * 3, image_std=[0.5] * 3),
        tokenizer=tok, video_processor=Qwen3VLVideoProcessor(),
    )
    ids = {t: tok.convert_tokens_to_ids(t) for t in specials}
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
                           deepstack_visual_indexes=[0], num_position_embeddings=64),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
    )
    torch.manual_seed(0)
    return Qwen3VLConditioner(processor, Qwen3VLForConditionalGeneration(cfg), device="cpu")


class TestConditioner:
    def test_system_prefix_stripped(self, tiny_conditioner):
        c = tiny_conditioner
        hidden, types = c.encode("a cat")
        full = c.processor(text=[build_prompt("a cat")], return_tensors="pt").input_ids
        assert hidden.shape[0] == full.shape[1] - c.prefix_length
        assert (types == 0).all()

    def test_hidden_state_is_pre_final_norm(self, tiny_conditioner):
        """sd.cpp takes the last decoder layer's output *before* the final RMSNorm."""
        c = tiny_conditioner
        hidden, _ = c.encode("a cat")
        inputs = c.processor(text=[build_prompt("a cat")], return_tensors="pt")
        with torch.no_grad():
            normed = c.model.model(**inputs, use_cache=False).last_hidden_state[0, c.prefix_length:]
        assert torch.allclose(c.model.model.language_model.norm(hidden), normed, atol=1e-5)
        assert not torch.allclose(hidden, normed, atol=1e-3)

    def test_reference_slots_are_one_per_32px(self, tiny_conditioner):
        from PIL import Image

        _, types = tiny_conditioner.encode("a cat", [Image.new("RGBA", (64, 96), (255, 0, 0, 128))])
        assert int((types == 1).sum()) == (64 // 32) * (96 // 32)

    def test_reference_must_be_32px_aligned(self, tiny_conditioner):
        from PIL import Image

        with pytest.raises(ValueError, match="multiples of 32"):
            tiny_conditioner.encode("x", [Image.new("RGB", (48, 64))])


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter(tmp_path):
    pytest.importorskip("diffusers")
    from atelier.adapters import QwenImage21Adapter

    model = _tiny_model()
    save_file({f"model.diffusion_model.{k}": v.contiguous() for k, v in model.state_dict().items()},
              tmp_path / "dit.safetensors")
    return QwenImage21Adapter(tmp_path / "dit.safetensors", device="cpu", dtype=torch.float32, load_encoders=False)


class TestAdapter:
    def test_shift_matches_sdcpp_flux_schedule(self, adapter):
        assert adapter.shift == pytest.approx(math.exp(1.15))  # 1024² → 4096 tokens → mu = max_shift

    def test_forward_matches_per_sample_model_calls(self, adapter):
        """Padded batch + per-sample layouts == calling the DiT on each sample directly."""
        from atelier.data import EditingCollator

        samples = [
            {"target_latents": torch.randn(8, 4, 4), "control_latents": torch.randn(8, 2, 4),
             "prompt_embeds": torch.randn(5, 32), "prompt_embeds_mask": torch.ones(5, dtype=torch.long),
             "prompt_token_types": torch.tensor([0, 1, 1, 0, 0])},
            {"target_latents": torch.randn(8, 4, 4), "control_latents": torch.randn(8, 2, 4),
             "prompt_embeds": torch.randn(3, 32), "prompt_embeds_mask": torch.ones(3, dtype=torch.long),
             "prompt_token_types": torch.tensor([1, 1, 0])},
        ]
        batch = EditingCollator()(samples)
        assert batch["prompt_token_types"].tolist() == [[0, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
        t = torch.tensor([300.0, 800.0])
        model = adapter.model.eval()
        with torch.no_grad():
            out = adapter.forward(model, batch["target_latents"], t, batch)
            for i, s in enumerate(samples):
                ref = model(s["target_latents"][None], t[i:i + 1], s["prompt_embeds"][None],
                            s["prompt_token_types"], [s["control_latents"][None]])
                assert torch.allclose(out[i:i + 1], ref, atol=1e-5)

    def test_flow_matching_lora_training_and_export(self, adapter, tmp_path):
        from peft import LoraConfig, get_peft_model

        from atelier.data import EditingCollator
        from atelier.losses import FlowMatchingLoss

        model = get_peft_model(adapter.model, LoraConfig(r=4, lora_alpha=8, init_lora_weights="gaussian",
                                                         target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        batch = EditingCollator()([
            {"target_latents": torch.randn(8, 4, 4), "prompt_embeds": torch.randn(6, 32),
             "prompt_embeds_mask": torch.ones(6, dtype=torch.long),
             "prompt_token_types": torch.zeros(6, dtype=torch.long)},
        ])
        loss, _ = FlowMatchingLoss()(adapter, model, batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if "lora_" in n)

        adapter.save_lora(model, tmp_path / "lora")
        sd = load_file(tmp_path / "lora" / "lora.safetensors")
        key = "diffusion_model.transformer_blocks.0.attn.to_q"
        assert sd[f"{key}.lora_A.weight"].shape == (4, 256)
        assert sd[f"{key}.lora_B.weight"].shape == (256, 4)
        assert sd[f"{key}.alpha"].item() == pytest.approx(8.0), "sd.cpp applies alpha / rank"
        assert all(k.startswith("diffusion_model.transformer_blocks.") for k in sd)

    def test_preference_loss_with_timestep_bias(self, adapter):
        from atelier.losses import DiffusionDPOLoss

        batch = {
            "prompt_embeds": torch.randn(2, 5, 32),
            "prompt_embeds_mask": torch.ones(2, 5, dtype=torch.long),
            "prompt_token_types": torch.zeros(2, 5, dtype=torch.long),
            "chosen_latents": torch.randn(2, 8, 4, 4),
            "rejected_latents": torch.randn(2, 8, 4, 4),
        }
        loss, _ = DiffusionDPOLoss()(adapter, adapter.model, batch)
        assert torch.isfinite(loss)

    def test_rgba_pixels(self):
        from PIL import Image

        from atelier.adapters.qwen_image_2_1 import pil_to_rgba_tensor

        opaque = pil_to_rgba_tensor(Image.new("RGB", (4, 4), (255, 0, 0)))
        assert opaque.shape == (4, 4, 4) and torch.allclose(opaque[3], torch.ones(4, 4))
        clear = pil_to_rgba_tensor(Image.new("RGBA", (4, 4), (0, 0, 0, 0)))
        assert torch.allclose(clear[3], -torch.ones(4, 4))

    def test_registered(self):
        from atelier import registry
        from atelier.adapters import QwenImage21Adapter

        assert registry.get_adapter_class("qwen_image_2_1") is QwenImage21Adapter


def test_cache_embeddings_editing_slots_match_reference_latent(tiny_conditioner):
    """VLM slots and the reference's VAE latent come out 1:4, so the DiT layout builds."""
    pytest.importorskip("diffusers")
    from datasets import Dataset
    from PIL import Image

    from atelier.adapters import QwenImage21Adapter
    from atelier.data import cache_embeddings
    from atelier.models.qwen_image_2_1 import build_vae

    adapter = object.__new__(QwenImage21Adapter)
    adapter._device = "cpu"
    adapter._conditioner = tiny_conditioner
    adapter._vae = build_vae(base_dim=8, decoder_base_dim=12, z_dim=4, latents_mean=[0.0] * 4, latents_std=[1.0] * 4)
    adapter._latents_mean = torch.zeros(1, 4, 1, 1)
    adapter._latents_std = torch.ones(1, 4, 1, 1)

    raw = Dataset.from_dict({
        "prompt": ["make it blue"],
        "chosen": [Image.new("RGB", (100, 60), (0, 0, 255))],
        "rejected": [Image.new("RGBA", (100, 60), (255, 0, 0, 90))],
    })
    text, target, control = cache_embeddings(raw, adapter, target_area=96 * 64)
    types, ref = text["sample_0"]["prompt_token_types"], control["sample_0"]
    assert ref.shape[0] == 4 and target["sample_0"].shape == ref.shape
    assert int((types == 1).sum()) * 4 == ref.shape[1] * ref.shape[2]
    build_layout(types.tolist(), [tuple(ref.shape[1:]), tuple(target["sample_0"].shape[1:])])
