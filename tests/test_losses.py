"""Tests for all loss functions."""

import torch
from helpers import FlowMatchingMockAdapter, MockAdapter

from atelier.data.editing import EditingCollator
from atelier.data.generation import GenerationCollator
from atelier.losses import (
    DiffusionCPOLoss,
    DiffusionDPOLoss,
    DiffusionIPOLoss,
    DiffusionKTOLoss,
    DiffusionORPOLoss,
    DiffusionSimPOLoss,
    EpsilonLoss,
    FlowMatchingLoss,
)
from atelier.losses.diffusion_orpo import _log1mexp


def _make_paired_batch(bsz=2, channels=4, spatial=8):
    """Create a batch with chosen/rejected latents and prompt embeddings."""
    return {
        "chosen_latents": torch.randn(bsz, channels, spatial, spatial),
        "rejected_latents": torch.randn(bsz, channels, spatial, spatial),
        "prompt_embeds": torch.randn(bsz, 16, 64),
    }


def _make_single_batch(bsz=2, channels=4, spatial=8):
    """Create a batch with image latents and prompt embeddings."""
    return {
        "image_latents": torch.randn(bsz, channels, spatial, spatial),
        "prompt_embeds": torch.randn(bsz, 16, 64),
    }


def _make_flow_matching_batch(bsz=2, channels=4, spatial=8):
    """Create a batch for flow matching loss with 5D latents."""
    return {
        "target_latents": torch.randn(bsz, channels, 1, spatial, spatial),
        "control_latents": torch.randn(bsz, channels, 1, spatial, spatial),
        "prompt_embeds": torch.randn(bsz, 16, 64),
    }


# ---- FlowMatchingLoss ----

class TestFlowMatchingLoss:
    def test_basic_forward(self):
        adapter = FlowMatchingMockAdapter()
        loss_fn = FlowMatchingLoss()
        batch = _make_flow_matching_batch()
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert loss.shape == ()
        assert loss.item() > 0
        assert "mse" in metrics

    def test_weighting_none(self):
        loss_fn = FlowMatchingLoss(weighting_scheme="none")
        sigmas = torch.tensor([0.2, 0.5, 0.8])
        w = loss_fn._compute_weighting(sigmas)
        assert torch.allclose(w, torch.ones_like(sigmas))

    def test_weighting_sigma_sqrt(self):
        loss_fn = FlowMatchingLoss(weighting_scheme="sigma_sqrt")
        sigmas = torch.tensor([0.5, 1.0])
        w = loss_fn._compute_weighting(sigmas)
        expected = (sigmas ** -2.0).clamp(max=10.0)
        assert torch.allclose(w, expected)

    def test_weighting_unknown_falls_back(self):
        loss_fn = FlowMatchingLoss(weighting_scheme="unknown")
        sigmas = torch.tensor([0.3])
        w = loss_fn._compute_weighting(sigmas)
        assert torch.allclose(w, torch.ones_like(sigmas))

    def test_create_collator(self):
        loss_fn = FlowMatchingLoss()
        collator = loss_fn.create_collator()
        assert isinstance(collator, EditingCollator)

    def test_normalize_latents_called(self):
        adapter = FlowMatchingMockAdapter()
        called = []

        def mock_normalize(latents):
            called.append(True)
            return latents * 0.5

        adapter.normalize_latents = mock_normalize
        loss_fn = FlowMatchingLoss()
        batch = _make_flow_matching_batch()
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert len(called) >= 1  # normalize called for target_latents at minimum

    def test_control_latents_normalized(self):
        adapter = FlowMatchingMockAdapter()
        normalize_calls = []

        def mock_normalize(latents):
            normalize_calls.append(latents.shape)
            return latents

        adapter.normalize_latents = mock_normalize
        loss_fn = FlowMatchingLoss()
        batch = _make_flow_matching_batch()
        loss, _ = loss_fn(adapter, adapter.model, batch)
        # Called for target_latents and control_latents
        assert len(normalize_calls) == 2


# ---- EpsilonLoss ----

class TestEpsilonLoss:
    def test_basic_forward(self):
        adapter = MockAdapter()
        loss_fn = EpsilonLoss()
        batch = _make_single_batch()
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert loss.shape == ()
        assert loss.item() >= 0
        assert "mse" in metrics

    def test_create_collator(self):
        loss_fn = EpsilonLoss()
        assert isinstance(loss_fn.create_collator(), GenerationCollator)


# ---- DiffusionDPOLoss ----

class TestDiffusionDPOLoss:
    def test_basic_forward(self):
        adapter = MockAdapter()
        loss_fn = DiffusionDPOLoss(beta=0.1, sft_weight=0.1)
        batch = _make_paired_batch()
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert loss.shape == ()
        assert "dpo_loss" in metrics
        assert "sft_loss" in metrics
        assert "total_loss" in metrics
        assert "reward_accuracy" in metrics

    def test_beta_constant(self):
        loss_fn = DiffusionDPOLoss(beta=0.5, beta_schedule="constant")
        loss_fn.global_step = 50
        loss_fn.total_steps = 100
        assert loss_fn._get_beta() == 0.5

    def test_beta_linear_warmup(self):
        loss_fn = DiffusionDPOLoss(beta=1.0, beta_schedule="linear", beta_warmup_steps=100)
        loss_fn.total_steps = 1000
        loss_fn.global_step = 50
        assert loss_fn._get_beta() == 0.5  # 50/100 * 1.0

    def test_beta_linear_after_warmup(self):
        loss_fn = DiffusionDPOLoss(beta=1.0, beta_schedule="linear", beta_warmup_steps=10)
        loss_fn.total_steps = 100
        loss_fn.global_step = 50  # past warmup, before tail
        assert loss_fn._get_beta() == 1.0

    def test_beta_linear_tail(self):
        loss_fn = DiffusionDPOLoss(beta=0.1, beta_schedule="linear", beta_warmup_steps=10)
        loss_fn.total_steps = 100
        loss_fn.global_step = 100  # at end
        beta = loss_fn._get_beta()
        assert beta > 0.1  # should be approaching 0.3

    def test_beta_cosine(self):
        loss_fn = DiffusionDPOLoss(beta=1.0, beta_schedule="cosine")
        loss_fn.total_steps = 100
        loss_fn.global_step = 0
        beta_start = loss_fn._get_beta()
        loss_fn.global_step = 50
        beta_mid = loss_fn._get_beta()
        loss_fn.global_step = 100
        beta_end = loss_fn._get_beta()
        # Cosine: starts near 0, peaks at middle, back to 0
        assert beta_mid > beta_start or beta_mid > beta_end

    def test_logit_clamping(self):
        adapter = MockAdapter()
        loss_fn = DiffusionDPOLoss(beta=100.0, logit_clamp=1.0)
        batch = _make_paired_batch()
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        # Should not be NaN/Inf due to clamping
        assert torch.isfinite(loss)

    def test_create_collator(self):
        assert isinstance(DiffusionDPOLoss().create_collator(), GenerationCollator)


# ---- DiffusionIPOLoss ----

class TestDiffusionIPOLoss:
    def test_with_ref_model(self):
        adapter = MockAdapter()
        ref_model = MockAdapter().model
        loss_fn = DiffusionIPOLoss(beta=0.1, ref_model=ref_model)
        batch = _make_paired_batch()
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert loss.shape == ()
        assert "loss" in metrics
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "reward_margin" in metrics
        assert "reward_accuracy" in metrics

    def test_no_ref_model_no_peft_raises(self):
        adapter = MockAdapter()
        loss_fn = DiffusionIPOLoss(beta=0.1)
        batch = _make_paired_batch()
        try:
            loss_fn(adapter, adapter.model, batch)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "ref_model" in str(e)

    def test_create_collator(self):
        assert isinstance(DiffusionIPOLoss().create_collator(), GenerationCollator)


# ---- DiffusionCPOLoss ----

class TestDiffusionCPOLoss:
    def test_basic_forward(self):
        adapter = MockAdapter()
        loss_fn = DiffusionCPOLoss(beta=0.1)
        batch = _make_paired_batch()
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert loss.shape == ()
        assert "sft_loss" in metrics
        assert "preference_loss" in metrics
        assert "reward_margin" in metrics
        assert "reward_accuracy" in metrics

    def test_label_smoothing(self):
        adapter = MockAdapter()
        torch.manual_seed(42)
        loss_fn_no_smooth = DiffusionCPOLoss(beta=0.1, label_smoothing=0.0)
        torch.manual_seed(42)
        loss_fn_smooth = DiffusionCPOLoss(beta=0.1, label_smoothing=0.1)
        batch = _make_paired_batch()
        # Both should produce valid losses
        torch.manual_seed(0)
        l1, _ = loss_fn_no_smooth(adapter, adapter.model, batch)
        torch.manual_seed(0)
        l2, _ = loss_fn_smooth(adapter, adapter.model, batch)
        assert torch.isfinite(l1) and torch.isfinite(l2)

    def test_create_collator(self):
        assert isinstance(DiffusionCPOLoss().create_collator(), GenerationCollator)


# ---- DiffusionKTOLoss ----

class TestDiffusionKTOLoss:
    def _make_kto_batch(self, bsz=4):
        batch = _make_single_batch(bsz=bsz)
        # Half desirable, half undesirable
        batch["kto_label"] = torch.tensor([True, True, False, False])
        return batch

    def test_with_ref_model(self):
        adapter = MockAdapter()
        ref_model = MockAdapter().model
        loss_fn = DiffusionKTOLoss(beta=0.1, ref_model=ref_model)
        batch = self._make_kto_batch()
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert loss.shape == ()
        assert "desirable_rewards" in metrics
        assert "undesirable_rewards" in metrics
        assert "reward_accuracy" in metrics
        assert "kl_ref" in metrics

    def test_all_desirable(self):
        adapter = MockAdapter()
        ref_model = MockAdapter().model
        loss_fn = DiffusionKTOLoss(beta=0.1, ref_model=ref_model)
        batch = _make_single_batch(bsz=2)
        batch["kto_label"] = torch.tensor([True, True])
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert torch.isfinite(loss)

    def test_all_undesirable(self):
        adapter = MockAdapter()
        ref_model = MockAdapter().model
        loss_fn = DiffusionKTOLoss(beta=0.1, ref_model=ref_model)
        batch = _make_single_batch(bsz=2)
        batch["kto_label"] = torch.tensor([False, False])
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert torch.isfinite(loss)

    def test_no_ref_model_no_peft_raises(self):
        adapter = MockAdapter()
        loss_fn = DiffusionKTOLoss(beta=0.1)
        batch = self._make_kto_batch()
        try:
            loss_fn(adapter, adapter.model, batch)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "ref_model" in str(e)

    def test_create_collator(self):
        assert isinstance(DiffusionKTOLoss().create_collator(), GenerationCollator)


# ---- DiffusionORPOLoss ----

class TestDiffusionORPOLoss:
    def test_basic_forward(self):
        adapter = MockAdapter()
        loss_fn = DiffusionORPOLoss(beta=0.1)
        batch = _make_paired_batch()
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert loss.shape == ()
        assert "sft_loss" in metrics
        assert "or_loss" in metrics
        assert "log_odds_ratio" in metrics
        assert "reward_accuracy" in metrics

    def test_create_collator(self):
        assert isinstance(DiffusionORPOLoss().create_collator(), GenerationCollator)


class TestLog1mexp:
    def test_small_values(self):
        x = torch.tensor([-2.0, -3.0, -5.0])
        result = _log1mexp(x)
        expected = torch.log(1 - torch.exp(x))
        assert torch.allclose(result, expected, atol=1e-5)

    def test_values_near_zero(self):
        x = torch.tensor([-0.1, -0.01])
        result = _log1mexp(x)
        # Should not be NaN
        assert torch.all(torch.isfinite(result))

    def test_boundary(self):
        x = torch.tensor([-0.6931])  # ln(2)
        result = _log1mexp(x)
        assert torch.isfinite(result)


# ---- DiffusionSimPOLoss ----

class TestDiffusionSimPOLoss:
    def test_basic_forward(self):
        adapter = MockAdapter()
        loss_fn = DiffusionSimPOLoss(beta=2.0, gamma=0.5)
        batch = _make_paired_batch()
        loss, metrics = loss_fn(adapter, adapter.model, batch)
        assert loss.shape == ()
        assert "loss" in metrics
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "reward_margin" in metrics
        assert "reward_accuracy" in metrics

    def test_gamma_zero(self):
        adapter = MockAdapter()
        loss_fn = DiffusionSimPOLoss(beta=1.0, gamma=0.0)
        batch = _make_paired_batch()
        loss, _ = loss_fn(adapter, adapter.model, batch)
        assert torch.isfinite(loss)

    def test_create_collator(self):
        assert isinstance(DiffusionSimPOLoss().create_collator(), GenerationCollator)


# ---- Loss utils ----

class TestLossUtils:
    def test_get_paired_with_precomputed_latents(self):
        from atelier.losses.utils import get_paired_denoising_losses
        adapter = MockAdapter()
        batch = _make_paired_batch()
        chosen_per, rejected_per, sft_loss, fwd_batch = get_paired_denoising_losses(
            adapter, adapter.model, batch,
        )
        assert chosen_per.shape == (2,)
        assert rejected_per.shape == (2,)
        assert sft_loss.shape == ()

    def test_get_paired_with_timestep_bias(self):
        from atelier.losses.utils import get_paired_denoising_losses
        adapter = MockAdapter()
        batch = _make_paired_batch()
        chosen_per, rejected_per, sft_loss, _ = get_paired_denoising_losses(
            adapter, adapter.model, batch, timestep_bias=(0.3, 0.8),
        )
        assert chosen_per.shape == (2,)

    def test_get_paired_missing_latents(self):
        from atelier.losses.utils import get_paired_denoising_losses
        adapter = MockAdapter()
        batch = {"prompt_embeds": torch.randn(2, 16, 64)}
        chosen_per, rejected_per, sft_loss, _ = get_paired_denoising_losses(
            adapter, adapter.model, batch,
        )
        # Should return zeros when no latents
        assert chosen_per.item() == 0.0

    def test_get_single_with_precomputed_latents(self):
        from atelier.losses.utils import get_single_denoising_loss
        adapter = MockAdapter()
        batch = _make_single_batch()
        per_sample, mean_loss, fwd_batch = get_single_denoising_loss(
            adapter, adapter.model, batch,
        )
        assert per_sample.shape == (2,)
        assert mean_loss.shape == ()

    def test_get_single_with_timestep_bias(self):
        from atelier.losses.utils import get_single_denoising_loss
        adapter = MockAdapter()
        batch = _make_single_batch()
        per_sample, mean_loss, _ = get_single_denoising_loss(
            adapter, adapter.model, batch, timestep_bias=(0.2, 0.9),
        )
        assert per_sample.shape == (2,)

    def test_get_single_fallback_to_target_latents(self):
        from atelier.losses.utils import get_single_denoising_loss
        adapter = MockAdapter()
        batch = {
            "target_latents": torch.randn(2, 4, 8, 8),
            "prompt_embeds": torch.randn(2, 16, 64),
        }
        per_sample, mean_loss, _ = get_single_denoising_loss(
            adapter, adapter.model, batch,
        )
        assert per_sample.shape == (2,)

    def test_get_single_missing_latents(self):
        from atelier.losses.utils import get_single_denoising_loss
        adapter = MockAdapter()
        batch = {"prompt_embeds": torch.randn(2, 16, 64)}
        per_sample, mean_loss, _ = get_single_denoising_loss(
            adapter, adapter.model, batch,
        )
        assert per_sample.item() == 0.0
        assert mean_loss.item() == 0.0

    def test_get_latents_precomputed(self):
        from atelier.losses.utils import _get_latents
        adapter = MockAdapter()
        latents = torch.randn(2, 4, 8, 8)
        batch = {"my_latents": latents}
        result = _get_latents(adapter, batch, "my_latents", "my_image", "cpu")
        assert torch.equal(result, latents)

    def test_get_latents_none_when_missing(self):
        from atelier.losses.utils import _get_latents
        adapter = MockAdapter()
        result = _get_latents(adapter, {}, "latents", "image", "cpu")
        assert result is None

    def test_get_latents_none_when_not_tensor(self):
        from atelier.losses.utils import _get_latents
        adapter = MockAdapter()
        batch = {"my_latents": "not_a_tensor"}
        result = _get_latents(adapter, batch, "my_latents", "my_image", "cpu")
        assert result is None

    def test_get_text_conditioning_precomputed(self):
        from atelier.losses.utils import _get_text_conditioning
        adapter = MockAdapter()
        batch = {
            "prompt_embeds": torch.randn(2, 16, 64),
            "pooled_prompt_embeds": torch.randn(2, 64),
        }
        result = _get_text_conditioning(adapter, batch, "cpu")
        assert "prompt_embeds" in result
        assert "pooled_prompt_embeds" in result

    def test_get_text_conditioning_empty(self):
        from atelier.losses.utils import _get_text_conditioning
        adapter = MockAdapter()
        result = _get_text_conditioning(adapter, {}, "cpu")
        assert result == {}

    def test_get_paired_with_image_tensors(self):
        from atelier.losses.utils import get_paired_denoising_losses
        adapter = MockAdapter()
        batch = {
            "chosen_image": torch.randn(2, 3, 32, 32),
            "rejected_image": torch.randn(2, 3, 32, 32),
            "prompt_embeds": torch.randn(2, 16, 64),
        }
        chosen_per, rejected_per, sft_loss, _ = get_paired_denoising_losses(
            adapter, adapter.model, batch,
        )
        assert chosen_per.shape == (2,)
