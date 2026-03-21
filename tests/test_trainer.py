"""Tests for AtelierTrainer and related utilities."""

import glob
import os
import shutil
import tempfile
from unittest.mock import MagicMock

import torch
from helpers import MockAdapter
from torch.utils.data import TensorDataset

from atelier import AtelierTrainer, TrainerCallback, TrainingConfig
from atelier.trainer import _config_to_dict, _fmt, _is_cuda_error


def _make_dataset(n=8, channels=4, spatial=8):
    """Create a simple tensor dataset with image latents and prompt embeds."""
    return TensorDataset(
        torch.randn(n, channels, spatial, spatial),  # image_latents
        torch.randn(n, 16, 64),  # prompt_embeds
    )


class SimpleLoss:
    """Minimal loss function for testing trainer."""

    def __call__(self, adapter, model, batch, training=True):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch.get("image_latents", list(batch.values())[0])
        else:
            x = batch
        # Pass through model to create gradient graph
        x = x.to(next(model.parameters()).device)
        pred = adapter.forward(model, x, None, batch)
        loss = torch.nn.functional.mse_loss(pred, torch.zeros_like(pred))
        return loss, {"mse": loss.item()}

    def create_collator(self):
        return self._collator

    def _collator(self, examples):
        # TensorDataset returns tuples
        batch = {}
        latents = torch.stack([ex[0] for ex in examples])
        embeds = torch.stack([ex[1] for ex in examples])
        batch["image_latents"] = latents
        batch["prompt_embeds"] = embeds
        return batch


# ---- Helper function tests ----

class TestFmt:
    def test_normal_float(self):
        assert _fmt(0.1234) == "0.1234"

    def test_small_float(self):
        result = _fmt(0.0001)
        assert "e" in result

    def test_zero(self):
        assert _fmt(0.0) == "0.0000"


class TestIsCudaError:
    def test_cuda_error(self):
        assert _is_cuda_error(RuntimeError("CUDA error: device-side assert"))

    def test_cuda_illegal(self):
        assert _is_cuda_error(RuntimeError("cuda illegal memory access"))

    def test_not_cuda(self):
        assert not _is_cuda_error(RuntimeError("some other error"))


class TestConfigToDict:
    def test_basic(self):
        config = TrainingConfig(output_dir="/tmp/test", num_epochs=5)
        d = _config_to_dict(config)
        assert d["output_dir"] == "/tmp/test"
        assert d["num_epochs"] == 5
        assert "seed" in d


# ---- Trainer tests ----

class TestTrainerInit:
    def test_basic_init(self):
        adapter = MockAdapter()
        config = TrainingConfig(
            output_dir=tempfile.mkdtemp(),
            num_epochs=1,
            batch_size=2,
            mixed_precision="no",
            gradient_checkpointing=False,
        )
        dataset = _make_dataset(4)
        loss_fn = SimpleLoss()

        trainer = AtelierTrainer(
            adapter=adapter,
            config=config,
            loss_fn=loss_fn,
            train_dataset=dataset,
        )
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert not trainer.stopped_early
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_with_eval_dataset(self):
        adapter = MockAdapter()
        config = TrainingConfig(
            output_dir=tempfile.mkdtemp(),
            num_epochs=1,
            batch_size=2,
            mixed_precision="no",
            gradient_checkpointing=False,
        )
        train_ds = _make_dataset(4)
        eval_ds = _make_dataset(4)
        loss_fn = SimpleLoss()

        trainer = AtelierTrainer(
            adapter=adapter,
            config=config,
            loss_fn=loss_fn,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )
        assert trainer.eval_dataloader is not None
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_with_callbacks(self):
        adapter = MockAdapter()
        config = TrainingConfig(
            output_dir=tempfile.mkdtemp(),
            num_epochs=1,
            batch_size=2,
            mixed_precision="no",
            gradient_checkpointing=False,
        )
        dataset = _make_dataset(4)
        loss_fn = SimpleLoss()
        cb = TrainerCallback()

        trainer = AtelierTrainer(
            adapter=adapter,
            config=config,
            loss_fn=loss_fn,
            train_dataset=dataset,
            callbacks=[cb],
        )
        assert len(trainer.callbacks) == 1
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_total_steps_propagated_to_loss(self):
        adapter = MockAdapter()
        config = TrainingConfig(
            output_dir=tempfile.mkdtemp(),
            num_epochs=2,
            batch_size=2,
            mixed_precision="no",
            gradient_checkpointing=False,
        )
        dataset = _make_dataset(8)
        loss_fn = SimpleLoss()
        loss_fn.total_steps = 0

        trainer = AtelierTrainer(
            adapter=adapter,
            config=config,
            loss_fn=loss_fn,
            train_dataset=dataset,
        )
        assert loss_fn.total_steps == trainer.max_steps
        shutil.rmtree(config.output_dir, ignore_errors=True)


class TestTrainerTrain:
    def test_basic_train_loop(self):
        adapter = MockAdapter()
        output_dir = tempfile.mkdtemp()
        config = TrainingConfig(
            output_dir=output_dir,
            num_epochs=1,
            batch_size=2,
            mixed_precision="no",
            gradient_checkpointing=False,
            logging_steps=1,
        )
        dataset = _make_dataset(4)
        loss_fn = SimpleLoss()

        trainer = AtelierTrainer(
            adapter=adapter,
            config=config,
            loss_fn=loss_fn,
            train_dataset=dataset,
        )
        trainer.train()
        assert trainer.global_step > 0
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_callbacks_called(self):
        adapter = MockAdapter()
        output_dir = tempfile.mkdtemp()
        config = TrainingConfig(
            output_dir=output_dir,
            num_epochs=1,
            batch_size=4,
            mixed_precision="no",
            gradient_checkpointing=False,
            logging_steps=1,
        )
        dataset = _make_dataset(4)
        loss_fn = SimpleLoss()

        cb = MagicMock(spec=TrainerCallback)
        trainer = AtelierTrainer(
            adapter=adapter,
            config=config,
            loss_fn=loss_fn,
            train_dataset=dataset,
            callbacks=[cb],
        )
        trainer.train()

        cb.on_train_begin.assert_called()
        cb.on_train_end.assert_called()
        cb.on_epoch_begin.assert_called()
        cb.on_epoch_end.assert_called()
        cb.on_step_end.assert_called()
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_request_stop(self):
        adapter = MockAdapter()
        output_dir = tempfile.mkdtemp()
        config = TrainingConfig(
            output_dir=output_dir,
            num_epochs=100,
            batch_size=2,
            mixed_precision="no",
            gradient_checkpointing=False,
        )
        dataset = _make_dataset(8)
        loss_fn = SimpleLoss()

        class StopCallback(TrainerCallback):
            def on_step_end(self, trainer, step, loss, metrics):
                if step >= 2:
                    trainer.request_stop()

        trainer = AtelierTrainer(
            adapter=adapter,
            config=config,
            loss_fn=loss_fn,
            train_dataset=dataset,
            callbacks=[StopCallback()],
        )
        trainer.train()
        assert trainer.stopped_early
        assert trainer.global_step <= 3  # stopped shortly after step 2
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_save_model(self):
        adapter = MockAdapter()
        output_dir = tempfile.mkdtemp()
        config = TrainingConfig(
            output_dir=output_dir,
            num_epochs=1,
            batch_size=4,
            mixed_precision="no",
            gradient_checkpointing=False,
        )
        dataset = _make_dataset(4)
        loss_fn = SimpleLoss()

        trainer = AtelierTrainer(
            adapter=adapter,
            config=config,
            loss_fn=loss_fn,
            train_dataset=dataset,
        )
        trainer.train()
        save_dir = os.path.join(output_dir, "saved")
        trainer.save_model(save_dir)
        assert os.path.isdir(save_dir)
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_checkpointing(self):
        adapter = MockAdapter()
        output_dir = tempfile.mkdtemp()
        config = TrainingConfig(
            output_dir=output_dir,
            num_epochs=1,
            batch_size=2,
            mixed_precision="no",
            gradient_checkpointing=False,
            save_steps=1,
            save_total_limit=2,
        )
        dataset = _make_dataset(8)
        loss_fn = SimpleLoss()

        trainer = AtelierTrainer(
            adapter=adapter,
            config=config,
            loss_fn=loss_fn,
            train_dataset=dataset,
        )
        trainer.train()
        # Check that checkpoints were created and rotated
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        assert len(checkpoints) <= config.save_total_limit
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_eval_during_training(self):
        adapter = MockAdapter()
        output_dir = tempfile.mkdtemp()
        config = TrainingConfig(
            output_dir=output_dir,
            num_epochs=1,
            batch_size=2,
            mixed_precision="no",
            gradient_checkpointing=False,
            eval_steps=1,
        )
        train_ds = _make_dataset(4)
        eval_ds = _make_dataset(4)
        loss_fn = SimpleLoss()

        cb = MagicMock(spec=TrainerCallback)
        trainer = AtelierTrainer(
            adapter=adapter,
            config=config,
            loss_fn=loss_fn,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            callbacks=[cb],
        )
        trainer.train()
        cb.on_evaluate.assert_called()
        shutil.rmtree(output_dir, ignore_errors=True)


class TestTrainerOptimizer:
    def test_adamw(self):
        adapter = MockAdapter()
        config = TrainingConfig(
            output_dir=tempfile.mkdtemp(),
            optimizer="adamw",
            mixed_precision="no",
            gradient_checkpointing=False,
        )
        dataset = _make_dataset(4)
        trainer = AtelierTrainer(
            adapter=adapter, config=config, loss_fn=SimpleLoss(), train_dataset=dataset,
        )
        assert trainer.optimizer is not None
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_sgd(self):
        adapter = MockAdapter()
        config = TrainingConfig(
            output_dir=tempfile.mkdtemp(),
            optimizer="sgd",
            mixed_precision="no",
            gradient_checkpointing=False,
        )
        dataset = _make_dataset(4)
        trainer = AtelierTrainer(
            adapter=adapter, config=config, loss_fn=SimpleLoss(), train_dataset=dataset,
        )
        assert trainer.optimizer is not None
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_unknown_optimizer_raises(self):
        adapter = MockAdapter()
        config = TrainingConfig(
            output_dir=tempfile.mkdtemp(),
            optimizer="nonexistent",
            mixed_precision="no",
            gradient_checkpointing=False,
        )
        dataset = _make_dataset(4)
        try:
            AtelierTrainer(
                adapter=adapter, config=config, loss_fn=SimpleLoss(), train_dataset=dataset,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)
        finally:
            shutil.rmtree(config.output_dir, ignore_errors=True)
