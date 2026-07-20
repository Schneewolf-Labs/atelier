"""Tests for Flow-GRPO support: loss, rewards, dataset, config, and trainer."""

import shutil
import tempfile

import pytest
import torch
from helpers import DummyModel
from PIL import Image

from atelier import FlowGRPOConfig, FlowGRPOTrainer, ImageRewardFn, TrainingConfig
from atelier.adapters import SDERollout, SupportsSDESampling
from atelier.adapters.base import ModelAdapter
from atelier.data import PromptCollator, PromptDataset
from atelier.losses import FlowGRPOLoss
from atelier.rewards import sanitize_rewards

# ---- Test doubles ----

class FakeDataset:
    """Minimal HF-dataset-like object."""

    def __init__(self, data, column_names):
        self._data = data
        self.column_names = column_names

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def select(self, indices):
        return FakeDataset([self._data[i] for i in indices], self.column_names)


def _make_prompt_dataset(n=4, with_extra=False):
    cols = ["prompt"] + (["style"] if with_extra else [])
    data = []
    for i in range(n):
        row = {"prompt": f"a photo of thing {i}"}
        if with_extra:
            row["style"] = f"style_{i}"
        data.append(row)
    return FakeDataset(data, cols)


class MockSDEAdapter(ModelAdapter):
    """Adapter satisfying SupportsSDESampling, backed by a tiny differentiable model."""

    def __init__(self, latent_dim=8):
        self._model = DummyModel()
        self.latent_dim = latent_dim

    @property
    def model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    def encode_text(self, prompts, device=None, **kwargs):
        return {"prompt_embeds": torch.randn(len(prompts), 4, 8)}

    def sample_with_logprobs(
        self, *, prompt_embeddings, num_inference_steps, guidance_scale,
        sde_noise_level, generator, decode=True,
    ):
        n = prompt_embeddings["prompt_embeds"].shape[0]
        s = num_inference_steps
        traj = torch.randn(n, s + 1, self.latent_dim, generator=generator)
        logprobs = torch.randn(n, s, generator=generator)
        step_mask = torch.ones(n, s)
        images = [Image.new("RGB", (8, 8)) for _ in range(n)] if decode else None
        return SDERollout(
            latents_trajectory=traj,
            logprobs=logprobs,
            step_mask=step_mask,
            images=images,
            prompt_embeddings=prompt_embeddings,
        )

    def recompute_logprobs(self, rollout, *, with_grad):
        s = rollout.num_steps
        base = rollout.latents_trajectory[:, :s, 0]  # (N, S), detached random
        lp = base * self._model.scale  # depends on the trainable param
        return lp if with_grad else lp.detach()

    def save_lora(self, model, path):
        pass

    def save_model(self, model, path):
        pass


def _constant_reward(prompts, images, **columns):
    # Distinct per image so advantages are non-trivial.
    return [float(i % 3) for i in range(len(images))]


# ---- FlowGRPOLoss ----

class TestFlowGRPOLoss:
    def _inputs(self, n=4, s=5):
        logprobs = torch.randn(n, s, requires_grad=True)
        old = logprobs.detach().clone()
        advantages = torch.randn(n)
        mask = torch.ones(n, s)
        return logprobs, old, advantages, mask

    def test_basic_grpo(self):
        logprobs, old, adv, mask = self._inputs()
        loss_fn = FlowGRPOLoss(loss_type="grpo")
        out = loss_fn(
            logprobs=logprobs, old_logprobs=old, ref_logprobs=None,
            advantages=adv, step_mask=mask,
        )
        assert out.loss.shape == ()
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert logprobs.grad is not None

    def test_dr_grpo_normalizer(self):
        logprobs, old, adv, mask = self._inputs(n=4, s=5)
        loss_fn = FlowGRPOLoss(loss_type="dr_grpo")
        out = loss_fn(
            logprobs=logprobs, old_logprobs=old, ref_logprobs=None,
            advantages=adv, step_mask=mask,
        )
        assert torch.isfinite(out.loss)

    def test_unknown_loss_type_raises(self):
        with pytest.raises(ValueError):
            FlowGRPOLoss(loss_type="ppo")

    def test_ratio_one_when_logprobs_match(self):
        logprobs, old, adv, mask = self._inputs()
        loss_fn = FlowGRPOLoss()
        out = loss_fn(
            logprobs=logprobs, old_logprobs=old, ref_logprobs=None,
            advantages=adv, step_mask=mask,
        )
        # logprobs == old => ratio == 1 everywhere.
        assert abs(out.ratio_mean - 1.0) < 1e-5
        assert out.clip_frac == 0.0

    def test_kl_zero_when_ref_equals_policy(self):
        logprobs, old, adv, mask = self._inputs()
        loss_fn = FlowGRPOLoss(beta=0.5)
        out = loss_fn(
            logprobs=logprobs, old_logprobs=old, ref_logprobs=logprobs.detach().clone(),
            advantages=adv, step_mask=mask,
        )
        assert abs(out.kl) < 1e-5

    def test_kl_positive_when_ref_differs(self):
        logprobs, old, adv, mask = self._inputs()
        ref = logprobs.detach() + 0.5
        loss_fn = FlowGRPOLoss(beta=1.0)
        out = loss_fn(
            logprobs=logprobs, old_logprobs=old, ref_logprobs=ref,
            advantages=adv, step_mask=mask,
        )
        assert out.kl > 0.0

    def test_step_mask_excludes_steps(self):
        torch.manual_seed(0)
        logprobs = torch.randn(2, 4)
        old = logprobs - 0.3  # nonzero ratio everywhere
        adv = torch.tensor([1.0, -1.0])
        full_mask = torch.ones(2, 4)
        partial_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        # dr_grpo uses a constant normalizer, so masking steps changes the loss.
        # (grpo's per-sequence length norm would cancel a constant per-step loss.)
        loss_fn = FlowGRPOLoss(loss_type="dr_grpo")
        out_full = loss_fn(
            logprobs=logprobs, old_logprobs=old, ref_logprobs=None,
            advantages=adv, step_mask=full_mask,
        )
        out_partial = loss_fn(
            logprobs=logprobs, old_logprobs=old, ref_logprobs=None,
            advantages=adv, step_mask=partial_mask,
        )
        assert not torch.allclose(out_full.loss, out_partial.loss)


class TestComputeAdvantages:
    def test_mean_centered_without_scaling(self):
        loss_fn = FlowGRPOLoss(scale_rewards=False)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])  # two groups of 2
        adv = loss_fn.compute_advantages(rewards, num_generations=2)
        # group [1,2] -> [-0.5, 0.5]; group [3,4] -> [-0.5, 0.5]
        assert torch.allclose(adv, torch.tensor([-0.5, 0.5, -0.5, 0.5]))

    def test_scaling_divides_by_std(self):
        loss_fn = FlowGRPOLoss(scale_rewards=True)
        rewards = torch.tensor([0.0, 2.0, 10.0, 14.0])
        adv = loss_fn.compute_advantages(rewards, num_generations=2)
        # Each group normalized to unit-ish scale; magnitudes match across groups.
        assert torch.allclose(adv[:2], adv[2:], atol=1e-5)

    def test_not_divisible_raises(self):
        loss_fn = FlowGRPOLoss()
        with pytest.raises(ValueError):
            loss_fn.compute_advantages(torch.tensor([1.0, 2.0, 3.0]), num_generations=2)


# ---- rewards ----

class TestSanitizeRewards:
    def test_passthrough(self):
        assert sanitize_rewards([1.0, 2.5, 3.0], expected=3) == [1.0, 2.5, 3.0]

    def test_non_finite_coerced(self):
        out = sanitize_rewards([float("nan"), float("inf"), 1.0], expected=3)
        assert out == [0.0, 0.0, 1.0]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            sanitize_rewards([1.0, 2.0], expected=3)

    def test_protocol_runtime_checkable(self):
        assert isinstance(_constant_reward, ImageRewardFn)

        class NotAReward:
            pass

        assert not isinstance(NotAReward(), ImageRewardFn)


# ---- SDERollout ----

class TestSDERollout:
    def test_shape_properties(self):
        rollout = SDERollout(
            latents_trajectory=torch.randn(3, 6, 8),
            logprobs=torch.randn(3, 5),
            step_mask=torch.ones(3, 5),
        )
        assert rollout.num_samples == 3
        assert rollout.num_steps == 5

    def test_adapter_satisfies_protocol(self):
        assert isinstance(MockSDEAdapter(), SupportsSDESampling)


# ---- PromptDataset / PromptCollator ----

class TestPromptDataset:
    def test_basic(self):
        ds = PromptDataset(_make_prompt_dataset(4), MockSDEAdapter())
        assert len(ds) == 4
        item = ds[0]
        assert item["prompt"] == "a photo of thing 0"
        assert "text_embeddings" not in item

    def test_extra_columns_passthrough(self):
        ds = PromptDataset(_make_prompt_dataset(3, with_extra=True), MockSDEAdapter())
        item = ds[1]
        assert item["style"] == "style_1"

    def test_max_samples(self):
        ds = PromptDataset(_make_prompt_dataset(10), MockSDEAdapter(), max_samples=3)
        assert len(ds) == 3

    def test_missing_prompt_column_raises(self):
        with pytest.raises(ValueError):
            PromptDataset(_make_prompt_dataset(2), MockSDEAdapter(), prompt_column="nope")

    def test_cached_embeddings(self):
        ds = PromptDataset(_make_prompt_dataset(3), MockSDEAdapter(), cache_embeddings=True)
        item = ds[0]
        assert "text_embeddings" in item
        assert "prompt_embeds" in item["text_embeddings"]

    def test_collator_batches_prompts_and_columns(self):
        ds = PromptDataset(_make_prompt_dataset(4, with_extra=True), MockSDEAdapter())
        collator = PromptCollator()
        batch = collator([ds[0], ds[1]])
        assert batch["prompts"] == ["a photo of thing 0", "a photo of thing 1"]
        assert batch["columns"]["style"] == ["style_0", "style_1"]
        assert batch["text_embeddings"] is None

    def test_collator_stacks_cached_embeddings(self):
        ds = PromptDataset(_make_prompt_dataset(4), MockSDEAdapter(), cache_embeddings=True)
        collator = PromptCollator()
        batch = collator([ds[0], ds[1]])
        assert batch["text_embeddings"]["prompt_embeds"].shape[0] == 2


# ---- FlowGRPOConfig ----

class TestFlowGRPOConfig:
    def test_is_training_config(self):
        config = FlowGRPOConfig()
        assert isinstance(config, TrainingConfig)

    def test_defaults(self):
        config = FlowGRPOConfig()
        assert config.num_generations == 8
        assert config.sde_noise_level == 0.7
        assert config.num_iterations == 1

    def test_inherits_shared_fields(self):
        config = FlowGRPOConfig(output_dir="/tmp/x", learning_rate=5e-5)
        assert config.output_dir == "/tmp/x"
        assert config.learning_rate == 5e-5


# ---- FlowGRPOTrainer ----

class TestFlowGRPOTrainer:
    def _config(self, **overrides):
        base = dict(
            output_dir=tempfile.mkdtemp(),
            num_epochs=1,
            batch_size=2,
            num_generations=2,
            num_inference_steps=4,
            mixed_precision="no",
            gradient_checkpointing=False,
            logging_steps=1,
        )
        base.update(overrides)
        return FlowGRPOConfig(**base)

    def test_train_advances_steps(self):
        config = self._config()
        trainer = FlowGRPOTrainer(
            adapter=MockSDEAdapter(),
            config=config,
            loss_fn=FlowGRPOLoss(beta=0.0),
            reward_fn=_constant_reward,
            train_dataset=PromptDataset(_make_prompt_dataset(4), MockSDEAdapter()),
        )
        trainer.train()
        assert trainer.global_step == 2  # 4 prompts / batch_size 2
        assert len(trainer.last_samples) > 0
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_train_with_kl(self):
        config = self._config()
        trainer = FlowGRPOTrainer(
            adapter=MockSDEAdapter(),
            config=config,
            loss_fn=FlowGRPOLoss(beta=0.2),
            reward_fn=_constant_reward,
            train_dataset=PromptDataset(_make_prompt_dataset(4), MockSDEAdapter()),
        )
        trainer.train()
        assert trainer.global_step == 2
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_multiple_inner_iterations(self):
        config = self._config(num_iterations=3)
        trainer = FlowGRPOTrainer(
            adapter=MockSDEAdapter(),
            config=config,
            loss_fn=FlowGRPOLoss(),
            reward_fn=_constant_reward,
            train_dataset=PromptDataset(_make_prompt_dataset(4), MockSDEAdapter()),
        )
        trainer.train()
        assert trainer.global_step == 2
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_timestep_fraction_subsample(self):
        config = self._config(timestep_fraction=0.5)
        trainer = FlowGRPOTrainer(
            adapter=MockSDEAdapter(),
            config=config,
            loss_fn=FlowGRPOLoss(),
            reward_fn=_constant_reward,
            train_dataset=PromptDataset(_make_prompt_dataset(4), MockSDEAdapter()),
        )
        trainer.train()
        assert trainer.global_step == 2
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_callbacks_and_stop(self):
        from atelier import TrainerCallback

        class StopCallback(TrainerCallback):
            def __init__(self):
                self.steps = 0

            def on_step_end(self, trainer, step, loss, metrics):
                self.steps += 1
                trainer.request_stop()

        config = self._config(num_epochs=5)
        cb = StopCallback()
        trainer = FlowGRPOTrainer(
            adapter=MockSDEAdapter(),
            config=config,
            loss_fn=FlowGRPOLoss(),
            reward_fn=_constant_reward,
            train_dataset=PromptDataset(_make_prompt_dataset(4), MockSDEAdapter()),
            callbacks=[cb],
        )
        trainer.train()
        assert trainer.stopped_early
        assert cb.steps == 1
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_save_model(self):
        import os

        config = self._config()
        trainer = FlowGRPOTrainer(
            adapter=MockSDEAdapter(),
            config=config,
            loss_fn=FlowGRPOLoss(),
            reward_fn=_constant_reward,
            train_dataset=PromptDataset(_make_prompt_dataset(4), MockSDEAdapter()),
        )
        save_dir = os.path.join(config.output_dir, "saved")
        trainer.save_model(save_dir)
        assert os.path.isdir(save_dir)
        shutil.rmtree(config.output_dir, ignore_errors=True)
