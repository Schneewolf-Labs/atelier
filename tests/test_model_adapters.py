"""Behavioral tests for the SD / SD3 / FLUX / Chroma / Z-Image adapters.

Real checkpoints are tens of GB, so these build *tiny* randomly-initialized
diffusers transformers and a bare adapter around them (``__init__`` — the
part that downloads weights — is skipped). That's enough to pin down the
parts that are easy to get subtly wrong: latent packing + RoPE ids against
diffusers' own pipeline helpers, Kontext reference-token handling, Z-Image's
reversed time / negated output / caption unpadding, and v-prediction
targets.

Pure-math tests (shift, sampling, pack/unpack) don't need diffusers.
"""

import math

import pytest
import torch

from atelier.adapters.flow import FlowMatchAdapter, default_shift, flux_mu, shift_sigmas
from atelier.adapters.flux import latent_position_ids, pack_latents, unpack_latents
from atelier.adapters.z_image import pad_caption_features

# ---------------------------------------------------------------------------
# Flow-matching math (no diffusers needed)
# ---------------------------------------------------------------------------


class _Flow(FlowMatchAdapter):
    def __init__(self, **kwargs):
        self._init_flow(None, **kwargs)


class TestShift:
    def test_identity_at_one(self):
        s = torch.linspace(0, 1, 11)
        assert torch.equal(shift_sigmas(s, 1.0), s)

    def test_endpoints_fixed_and_monotone(self):
        s = torch.linspace(0, 1, 101)
        out = shift_sigmas(s, 3.0)
        assert out[0] == 0 and torch.isclose(out[-1], torch.tensor(1.0))
        assert (out[1:] >= out[:-1]).all()
        assert (out[1:-1] > s[1:-1]).all(), "shift > 1 must push toward high noise"

    def test_static_shift_equals_flux_exponential_shift(self):
        """exp(mu) / (exp(mu) + 1/t - 1) is the same curve as static shift = exp(mu)."""
        mu = 1.15
        t = torch.linspace(0.01, 0.99, 50)
        flux_form = math.exp(mu) / (math.exp(mu) + (1 / t - 1))
        assert torch.allclose(shift_sigmas(t, math.exp(mu)), flux_form, atol=1e-6)

    def test_flux_mu_endpoints(self):
        assert flux_mu(256) == pytest.approx(0.5)
        assert flux_mu(4096) == pytest.approx(1.15)

    def test_default_shift_dynamic_matches_flux_1024(self):
        cfg = {"use_dynamic_shifting": True, "shift": 3.0}
        # 1024px → 64x64 packed tokens = 4096 → mu = max_shift
        assert default_shift(cfg, resolution=1024) == pytest.approx(math.exp(1.15))
        assert default_shift(cfg, resolution=512) == pytest.approx(math.exp(flux_mu(1024)))

    def test_default_shift_static(self):
        assert default_shift({"shift": 3.0}) == 3.0
        assert default_shift({}) == 1.0


class TestFlowSampling:
    @pytest.mark.parametrize("scheme", ["uniform", "logit_normal", "mode"])
    def test_sigmas_in_range_and_shape(self, scheme):
        a = _Flow(shift=3.0, timestep_sampling=scheme)
        t, s = a.sample_timesteps(64, "cpu")
        assert t.shape == (64,) and s.shape == (64, 1, 1, 1)
        assert (s >= 0).all() and (s <= 1).all()
        assert torch.allclose(t, s.flatten() * 1000)

    def test_bad_scheme(self):
        with pytest.raises(ValueError):
            _Flow(timestep_sampling="nope")

    def test_sigmas_for_explicit_timesteps(self):
        a = _Flow()
        s = a.sigmas_for_timesteps(torch.tensor([0, 500, 999]), "cpu")
        assert s.shape == (3, 1, 1, 1)
        assert torch.allclose(s.flatten(), torch.tensor([0.0, 0.5, 0.999]))

    def test_add_noise_and_target(self):
        a = _Flow()
        x0, noise = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
        s = torch.tensor([0.0, 1.0])
        xt = a.add_noise(x0, noise, None, s)
        assert torch.allclose(xt[0], x0[0]) and torch.allclose(xt[1], noise[1])
        assert torch.equal(a.compute_target(noise, x0, s), noise - x0)


class TestPacking:
    def test_roundtrip(self):
        x = torch.randn(2, 16, 12, 8)
        tokens = pack_latents(x)
        assert tokens.shape == (2, 6 * 4, 64)
        assert torch.equal(unpack_latents(tokens, 12, 8), x)

    def test_position_ids(self):
        ids = latent_position_ids(4, 6, index=1)
        assert ids.shape == (2 * 3, 3)
        assert (ids[:, 0] == 1).all()
        assert ids[:, 1].tolist() == [0, 0, 0, 1, 1, 1]
        assert ids[:, 2].tolist() == [0, 1, 2, 0, 1, 2]


class TestPadCaptionFeatures:
    def test_pads_and_masks(self):
        out = pad_caption_features([torch.ones(3, 4), torch.ones(5, 4)])
        assert out["prompt_embeds"].shape == (2, 5, 4)
        assert out["prompt_embeds_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        assert out["prompt_embeds"][0, 3:].abs().sum() == 0


# ---------------------------------------------------------------------------
# Tiny real transformers (needs diffusers)
# ---------------------------------------------------------------------------


def _bare(cls, model, **attrs):
    """Adapter instance around ``model`` without running the weight-loading __init__."""
    adapter = object.__new__(cls)
    adapter._model = model
    adapter._device = "cpu"
    adapter._dtype = torch.float32
    adapter._pipeline = None
    adapter._vae = None
    if hasattr(adapter, "_init_flow"):
        from diffusers import FlowMatchEulerDiscreteScheduler

        adapter._init_flow(FlowMatchEulerDiscreteScheduler(), shift=1.0, timestep_sampling="uniform")
    for k, v in attrs.items():
        setattr(adapter, k, v)
    return adapter


@pytest.fixture(scope="module")
def diffusers():
    return pytest.importorskip("diffusers")


def _tiny_flux(diffusers, guidance_embeds=True):
    torch.manual_seed(0)
    return diffusers.FluxTransformer2DModel(
        patch_size=1, in_channels=16, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
        pooled_projection_dim=24, guidance_embeds=guidance_embeds, axes_dims_rope=(4, 6, 6),
    ).eval()


class TestFluxAdapter:
    def _adapter(self, diffusers, **kw):
        from atelier.adapters import FluxAdapter

        return _bare(FluxAdapter, _tiny_flux(diffusers), _guidance_embeds=True, guidance=1.0, **kw)

    def _batch(self, bsz=2):
        return {"prompt_embeds": torch.randn(bsz, 7, 32), "pooled_prompt_embeds": torch.randn(bsz, 24)}

    def test_matches_diffusers_pipeline_convention(self, diffusers):
        """Adapter forward == FluxPipeline's own pack / ids / unpack around the transformer."""
        adapter = self._adapter(diffusers)
        model = adapter.model
        batch = self._batch()
        x = torch.randn(2, 4, 8, 12)
        t = torch.tensor([250.0, 750.0])

        with torch.no_grad():
            ours = adapter.forward(model, x, t, batch)

            P = diffusers.FluxPipeline
            packed = P._pack_latents(x, 2, 4, 8, 12)
            img_ids = P._prepare_latent_image_ids(2, 4, 6, "cpu", torch.float32)
            ref = model(
                hidden_states=packed, timestep=t / 1000, guidance=torch.full((2,), 1.0),
                pooled_projections=batch["pooled_prompt_embeds"], encoder_hidden_states=batch["prompt_embeds"],
                txt_ids=torch.zeros(7, 3), img_ids=img_ids, return_dict=False,
            )[0]
            # vae_scale_factor=8 → pixel dims are latent dims * 8
            ref = P._unpack_latents(ref, 8 * 8, 12 * 8, 8)

        assert ours.shape == x.shape
        assert torch.allclose(ours, ref, atol=1e-5)

    def test_kontext_reference_tokens(self, diffusers):
        adapter = self._adapter(diffusers)
        batch = self._batch()
        x = torch.randn(2, 4, 8, 8)
        t = torch.tensor([500.0, 500.0])
        with torch.no_grad():
            plain = adapter.forward(adapter.model, x, t, batch)
            edited = adapter.forward(adapter.model, x, t, {**batch, "control_latents": torch.randn(2, 4, 8, 8)})
        assert edited.shape == x.shape, "prediction must be cropped back to the target tokens"
        assert not torch.allclose(plain, edited), "reference tokens must be attended to"

    def test_flow_matching_loss_trains_lora(self, diffusers):
        from peft import LoraConfig, get_peft_model

        from atelier.data import EditingCollator
        from atelier.losses import FlowMatchingLoss

        adapter = self._adapter(diffusers)
        model = get_peft_model(adapter.model, LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_v"]))
        # cache_embeddings-shaped samples, incl. the pooled extra
        samples = [
            {"target_latents": torch.randn(4, 8, 8), "prompt_embeds": torch.randn(7, 32),
             "pooled_prompt_embeds": torch.randn(24)}
            for _ in range(2)
        ]
        batch = EditingCollator()(samples)
        assert batch["pooled_prompt_embeds"].shape == (2, 24)

        loss, metrics = FlowMatchingLoss()(adapter, model, batch)
        loss.backward()
        assert torch.isfinite(loss)
        grads = [p.grad for n, p in model.named_parameters() if "lora_" in n]
        assert grads and any(g is not None and g.abs().sum() > 0 for g in grads)

    def test_preference_loss_with_timestep_bias(self, diffusers):
        """DPO's biased integer timesteps must map to sigmas on flow adapters."""
        from atelier.losses import DiffusionDPOLoss

        adapter = self._adapter(diffusers)
        batch = {
            **self._batch(),
            "chosen_latents": torch.randn(2, 4, 8, 8),
            "rejected_latents": torch.randn(2, 4, 8, 8),
        }
        loss, metrics = DiffusionDPOLoss()(adapter, adapter.model, batch)
        assert torch.isfinite(loss)


class TestChromaAdapter:
    def test_forward_with_mask(self, diffusers):
        from atelier.adapters import ChromaAdapter

        torch.manual_seed(0)
        model = diffusers.ChromaTransformer2DModel(
            patch_size=1, in_channels=16, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
            axes_dims_rope=(4, 6, 6), approximator_num_channels=8, approximator_hidden_dim=32,
            approximator_layers=1,
        ).eval()
        adapter = _bare(ChromaAdapter, model, _guidance_embeds=False, guidance=None)
        x = torch.randn(2, 4, 8, 8)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        batch = {"prompt_embeds": torch.randn(2, 6, 32), "prompt_embeds_mask": mask}
        with torch.no_grad():
            out = adapter.forward(model, x, torch.tensor([100.0, 900.0]), batch)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestSD3Adapter:
    def test_forward_shape_and_timestep_scale(self, diffusers):
        from atelier.adapters import SD3Adapter

        torch.manual_seed(0)
        model = diffusers.SD3Transformer2DModel(
            sample_size=8, patch_size=2, in_channels=4, num_layers=1, attention_head_dim=8,
            num_attention_heads=2, joint_attention_dim=32, caption_projection_dim=16,
            pooled_projection_dim=16, out_channels=4, pos_embed_max_size=16,
        ).eval()
        adapter = _bare(SD3Adapter, model)
        x = torch.randn(2, 4, 8, 8)
        batch = {"prompt_embeds": torch.randn(2, 5, 32), "pooled_prompt_embeds": torch.randn(2, 16)}
        t = torch.tensor([300.0, 600.0])
        with torch.no_grad():
            ours = adapter.forward(model, x, t, batch)
            ref = model(hidden_states=x, timestep=t, encoder_hidden_states=batch["prompt_embeds"],
                        pooled_projections=batch["pooled_prompt_embeds"], return_dict=False)[0]
        assert torch.allclose(ours, ref, atol=1e-6), "SD3 MMDiT takes sigma * 1000 unscaled"


class TestZImageAdapter:
    def _model(self, diffusers):
        torch.manual_seed(0)
        return diffusers.ZImageTransformer2DModel(
            all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=32, n_layers=1,
            n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=16,
            axes_dims=[4, 6, 6], axes_lens=[64, 32, 32],
        ).eval()

    def test_reversed_time_and_negated_output(self, diffusers):
        from atelier.adapters import ZImageAdapter

        model = self._model(diffusers)
        adapter = _bare(ZImageAdapter, model)
        x = torch.randn(1, 4, 8, 8)
        feats = torch.randn(5, 16)
        sigma = 0.25
        with torch.no_grad():
            ours = adapter.forward(model, x, torch.tensor([sigma * 1000]), {"prompt_embeds": feats[None]})
            native = model([x[0].unsqueeze(1)], torch.tensor([1 - sigma]), [feats], return_dict=False)[0][0]
        assert torch.allclose(ours[0], -native.squeeze(1), atol=1e-5)

    def test_padded_batch_matches_per_sample(self, diffusers):
        """Padding + mask must reproduce unbatched per-caption results."""
        from atelier.adapters import ZImageAdapter

        model = self._model(diffusers)
        adapter = _bare(ZImageAdapter, model)
        x = torch.randn(2, 4, 8, 8)
        t = torch.tensor([400.0, 700.0])
        caps = [torch.randn(3, 16), torch.randn(6, 16)]
        with torch.no_grad():
            batched = adapter.forward(model, x, t, pad_caption_features(caps))
            singles = [
                adapter.forward(model, x[i:i + 1], t[i:i + 1], {"prompt_embeds": caps[i][None]})
                for i in range(2)
            ]
        assert torch.allclose(batched, torch.cat(singles), atol=1e-5)


class TestDDPMAdapters:
    def _unet(self, diffusers):
        torch.manual_seed(0)
        return diffusers.UNet2DConditionModel(
            block_out_channels=(32, 64), layers_per_block=1, sample_size=8, in_channels=4, out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=32, attention_head_dim=8, norm_num_groups=32,
        ).eval()

    def _sd(self, diffusers, prediction_type):
        from atelier.adapters import StableDiffusionAdapter

        adapter = _bare(StableDiffusionAdapter, self._unet(diffusers))
        adapter._init_ddpm(diffusers.DDPMScheduler(num_train_timesteps=1000), prediction_type)
        return adapter

    def test_epsilon_target(self, diffusers):
        adapter = self._sd(diffusers, None)
        assert adapter.prediction_type == "epsilon"
        noise, x0 = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
        assert torch.equal(adapter.compute_target(noise, x0, None, timesteps=torch.tensor([1, 2])), noise)

    def test_v_prediction_target(self, diffusers):
        adapter = self._sd(diffusers, "v_prediction")
        assert adapter.noise_scheduler.config.prediction_type == "v_prediction"
        noise, x0 = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
        t = torch.tensor([10, 900])
        target = adapter.compute_target(noise, x0, None, timesteps=t)
        a = adapter.noise_scheduler.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        s = (1 - adapter.noise_scheduler.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        assert torch.allclose(target, a * noise - s * x0, atol=1e-6)
        with pytest.raises(ValueError):
            adapter.compute_target(noise, x0, None)

    def test_bad_prediction_type(self, diffusers):
        with pytest.raises(ValueError):
            self._sd(diffusers, "x0")

    def test_single_denoising_loss_v_pred(self, diffusers):
        from atelier.losses import EpsilonLoss

        adapter = self._sd(diffusers, "v_prediction")
        batch = {"image_latents": torch.randn(2, 4, 8, 8), "prompt_embeds": torch.randn(2, 77, 32)}
        loss, _ = EpsilonLoss()(adapter, adapter.model, batch)
        assert torch.isfinite(loss)

    def test_on_the_fly_prompt_encoding(self, diffusers):
        """GenerationDataset without a tokenizer ships raw prompts; the loss encodes them."""
        from atelier.losses import EpsilonLoss

        adapter = self._sd(diffusers, None)
        seen = {}

        def encode_text(prompts, device=None, **kw):
            seen["prompts"] = prompts
            return {"prompt_embeds": torch.randn(len(prompts), 77, 32)}

        adapter.encode_text = encode_text
        batch = {"image_latents": torch.randn(2, 4, 8, 8), "prompt": ["a cat", "a dog"]}
        loss, _ = EpsilonLoss()(adapter, adapter.model, batch)
        assert seen["prompts"] == ["a cat", "a dog"]
        assert torch.isfinite(loss)


class TestRegistryNewAdapters:
    @pytest.mark.parametrize("name,cls_name", [
        ("sd", "StableDiffusionAdapter"),
        ("sd3", "SD3Adapter"),
        ("flux", "FluxAdapter"),
        ("flux_kontext", "FluxKontextAdapter"),
        ("chroma", "ChromaAdapter"),
        ("z_image", "ZImageAdapter"),
    ])
    def test_resolves(self, name, cls_name):
        from atelier import registry
        from atelier.adapters import ModelAdapter

        cls = registry.get_adapter_class(name)
        assert cls.__name__ == cls_name
        assert issubclass(cls, ModelAdapter)



class TestDiffusersFlowAdapterLoading:
    """Real ``__init__`` against a tiny on-disk checkpoint (encoders skipped —
    the embedding cache is computed in a separate process in that mode)."""

    def _checkpoint(self, diffusers, tmp_path, dynamic):
        _tiny_flux(diffusers).save_pretrained(tmp_path / "transformer")
        diffusers.AutoencoderKL(
            in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(32, 32, 32, 32),
            down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
            norm_num_groups=32, scaling_factor=0.3611, shift_factor=0.1159,
        ).save_config(tmp_path / "vae")
        diffusers.FlowMatchEulerDiscreteScheduler(
            shift=3.0, use_dynamic_shifting=dynamic,
        ).save_pretrained(tmp_path / "scheduler")
        return str(tmp_path)

    def test_flux_loads_without_encoders(self, diffusers, tmp_path):
        from atelier.adapters import FluxAdapter

        path = self._checkpoint(diffusers, tmp_path, dynamic=True)
        adapter = FluxAdapter(path, device="cpu", dtype=torch.float32, load_encoders=False)

        assert adapter.pipeline is None
        assert adapter.scaling_factor == pytest.approx(0.3611)
        assert adapter.shift_factor == pytest.approx(0.1159)
        assert adapter.vae_scale_factor == 8
        assert adapter.shift == pytest.approx(math.exp(1.15)), "dev-style dynamic shift at 1024px"
        assert adapter._guidance_embeds is True
        assert next(adapter.model.parameters()).device.type == "cpu"

        t, s = adapter.sample_timesteps(4, "cpu")
        out = adapter.forward(
            adapter.model, torch.randn(1, 4, 8, 8), t[:1],
            {"prompt_embeds": torch.randn(1, 3, 32), "pooled_prompt_embeds": torch.randn(1, 24)},
        )
        assert out.shape == (1, 4, 8, 8)

    def test_static_shift_and_override(self, diffusers, tmp_path):
        from atelier.adapters import FluxAdapter

        path = self._checkpoint(diffusers, tmp_path, dynamic=False)
        assert FluxAdapter(path, device="cpu", load_encoders=False).shift == 3.0
        assert FluxAdapter(path, device="cpu", load_encoders=False, shift=1.5).shift == 1.5

    def test_transformer_config_read_when_not_loaded(self, diffusers, tmp_path):
        from atelier.adapters import FluxAdapter

        path = self._checkpoint(diffusers, tmp_path, dynamic=True)
        adapter = FluxAdapter(path, device="cpu", load_encoders=False, load_transformer=False)
        assert adapter.model is None
        assert adapter._guidance_embeds is True
