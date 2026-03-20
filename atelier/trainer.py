import glob
import logging
import math
import os
import shutil

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from .config import TrainingConfig

logger = logging.getLogger(__name__)


class AtelierTrainer:
    """Multi-GPU diffusion training loop powered by accelerate.

    Works with any adapter + loss function combination:
        adapter: ModelAdapter — handles architecture-specific loading, forward pass, saving
        loss_fn: callable — loss, metrics = loss_fn(adapter, model, batch, training=True)
        loss_fn.create_collator() — returns a data collator for the dataloader
    """

    def __init__(
        self,
        adapter,
        config: TrainingConfig,
        loss_fn,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        peft_config=None,
        callbacks=None,
    ):
        self.adapter = adapter
        self.config = config
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []
        self.global_step = 0
        self.current_epoch = 0
        self._stop_requested = False

        model = adapter.model

        # Apply PEFT / LoRA
        if peft_config is not None:
            from peft import get_peft_model

            model = get_peft_model(model, peft_config)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()

        # Gradient checkpointing
        if config.gradient_checkpointing:
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
            elif hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

        # Initialize accelerator
        tracker_kwargs = {}
        if config.log_with == "wandb":
            wandb_kwargs = {}
            if config.run_name:
                wandb_kwargs["name"] = config.run_name
            if config.wandb_tags:
                wandb_kwargs["tags"] = config.wandb_tags
            if config.wandb_notes:
                wandb_kwargs["notes"] = config.wandb_notes
            tracker_kwargs["wandb"] = wandb_kwargs

        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with=config.log_with,
            project_dir=config.output_dir,
        )

        set_seed(config.seed)

        # Data collator
        if data_collator is None:
            data_collator = loss_fn.create_collator()

        # Dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            drop_last=True,
        )

        self.eval_dataloader = None
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=data_collator,
                num_workers=config.dataloader_num_workers,
                pin_memory=config.dataloader_pin_memory,
                drop_last=True,
            )

        # Optimizer
        optimizer = self._create_optimizer(model)

        # LR scheduler
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / config.gradient_accumulation_steps
        )
        self.max_steps = num_update_steps_per_epoch * config.num_epochs

        warmup_steps = config.warmup_steps if config.warmup_steps > 0 else int(self.max_steps * config.warmup_ratio)
        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.max_steps,
        )

        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(model, optimizer, self.train_dataloader, lr_scheduler)
        )

        if self.eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        # Propagate total steps to loss function (for beta scheduling etc.)
        if hasattr(loss_fn, "total_steps"):
            loss_fn.total_steps = self.max_steps

        # Init experiment tracking
        if config.log_with:
            self.accelerator.init_trackers(
                config.project_name or "atelier",
                config=_config_to_dict(config),
                init_kwargs=tracker_kwargs,
            )

        # Resume from checkpoint
        if config.resume_from_checkpoint:
            self.accelerator.load_state(config.resume_from_checkpoint)
            try:
                self.global_step = int(config.resume_from_checkpoint.rstrip("/").split("-")[-1])
            except ValueError:
                logger.warning("Could not parse step from checkpoint path, starting from step 0")

    def train(self):
        config = self.config

        self._log_info("***** Starting training *****")
        self._log_info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        self._log_info(f"  Num epochs = {config.num_epochs}")
        self._log_info(f"  Batch size per device = {config.batch_size}")
        self._log_info(
            f"  Total batch size = "
            f"{config.batch_size * self.accelerator.num_processes * config.gradient_accumulation_steps}"
        )
        self._log_info(f"  Gradient accumulation steps = {config.gradient_accumulation_steps}")
        self._log_info(f"  Total optimization steps = {self.max_steps}")
        self._log_info(f"  Number of processes = {self.accelerator.num_processes}")

        self._fire("on_train_begin")

        # Handle resuming mid-epoch
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / config.gradient_accumulation_steps
        )
        starting_epoch = self.global_step // num_update_steps_per_epoch if self.global_step > 0 else 0
        resume_step_in_epoch = self.global_step - (starting_epoch * num_update_steps_per_epoch)

        progress_bar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            desc="Training",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
        )

        for epoch in range(starting_epoch, config.num_epochs):
            self.current_epoch = epoch
            self._fire("on_epoch_begin", epoch=epoch)
            self.model.train()

            # Skip already-completed batches when resuming
            active_dataloader = self.train_dataloader
            if epoch == starting_epoch and resume_step_in_epoch > 0:
                active_dataloader = self.accelerator.skip_first_batches(
                    self.train_dataloader,
                    resume_step_in_epoch * config.gradient_accumulation_steps,
                )

            running_loss = 0.0
            steps_in_epoch = 0

            for step, batch in enumerate(active_dataloader):
                with self.accelerator.accumulate(self.model):
                    # Propagate step to loss function
                    if hasattr(self.loss_fn, "global_step"):
                        self.loss_fn.global_step = self.global_step

                    loss, metrics = self.loss_fn(self.adapter, self.model, batch, training=True)
                    del batch
                    self.accelerator.backward(loss)

                    if config.max_grad_norm and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)

                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    steps_in_epoch += 1
                    running_loss += loss.detach().item()

                    avg_loss = running_loss / steps_in_epoch
                    lr = self.lr_scheduler.get_last_lr()[0]
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                    self._fire("on_step_end", step=self.global_step, loss=loss.item(), metrics=metrics)

                    # Logging
                    if self.global_step % config.logging_steps == 0:
                        progress = self.global_step / self.max_steps
                        log_metrics = {
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": epoch + (step + 1) / len(self.train_dataloader),
                            "train/global_step": self.global_step,
                            "train/progress": progress,
                            **{f"train/{k}": v for k, v in metrics.items()},
                        }
                        self._log_metrics(log_metrics)
                        self._fire("on_log", metrics=log_metrics)

                    # Evaluation
                    if config.eval_steps and self.eval_dataloader and self.global_step % config.eval_steps == 0:
                        try:
                            self.evaluate()
                        except RuntimeError as e:
                            self._log_info(f"Eval failed at step {self.global_step}: {e}")
                            if _is_cuda_error(e):
                                self._log_info("CUDA context corrupted — stopping training")
                                self._stop_requested = True
                        self.model.train()

                    # Checkpointing
                    if config.save_steps and self.global_step % config.save_steps == 0:
                        try:
                            self._save_checkpoint()
                        except RuntimeError as e:
                            self._log_info(f"Checkpoint failed at step {self.global_step}: {e}")

                    if self._stop_requested:
                        self._log_info(f"Stopping early at step {self.global_step}")
                        break

            self._fire("on_epoch_end", epoch=epoch)

            if self._stop_requested:
                break

            if config.save_on_epoch_end:
                try:
                    self._save_checkpoint()
                except RuntimeError as e:
                    self._log_info(f"End-of-epoch checkpoint failed: {e}")

            if self.eval_dataloader:
                try:
                    self.evaluate()
                except RuntimeError as e:
                    self._log_info(f"End-of-epoch eval failed: {e}")
                    if _is_cuda_error(e):
                        self._log_info("CUDA context corrupted — stopping training")
                        self._stop_requested = True

        progress_bar.close()
        self._fire("on_train_end")

        if config.log_with:
            self.accelerator.end_training()

        self._log_info("***** Training complete *****")

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation loop and return metrics."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0

        for batch in self.eval_dataloader:
            loss, metrics = self.loss_fn(self.adapter, self.model, batch, training=False)

            loss = self.accelerator.reduce(loss, reduction="mean")
            total_loss += loss.item()

            if metrics:
                keys = sorted(metrics.keys())
                vals = torch.tensor([metrics[k] for k in keys], device=self.accelerator.device)
                vals = self.accelerator.reduce(vals, reduction="mean")
                for k, v in zip(keys, vals):
                    total_metrics[k] = total_metrics.get(k, 0.0) + v.item()
            num_batches += 1

            del batch, loss, metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}

        eval_results = {"eval/loss": avg_loss, **{f"eval/{k}": v for k, v in avg_metrics.items()}}

        self._log_info(
            f"  Eval — loss: {avg_loss:.4f}" + "".join(f" | {k}: {v:.4f}" for k, v in avg_metrics.items())
        )
        self._log_metrics(eval_results)
        self._fire("on_evaluate", metrics=eval_results)

        self.model.train()
        return eval_results

    def request_stop(self):
        """Request graceful stop at the end of the current step."""
        self._stop_requested = True
        self._log_info("Stop requested — will stop after current step")

    @property
    def stopped_early(self):
        """Whether training was stopped early via request_stop()."""
        return self._stop_requested

    def save_model(self, output_dir=None):
        """Save model via the adapter (handles LoRA vs full model)."""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)

        if self.accelerator.is_main_process:
            # Check if model has LoRA adapters
            if hasattr(unwrapped, "peft_config"):
                self.adapter.save_lora(unwrapped, output_dir)
            else:
                self.adapter.save_model(unwrapped, output_dir)
            self._log_info(f"Model saved to {output_dir}")

    # ---- Internal helpers ----

    def _create_optimizer(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        lr = self.config.learning_rate

        opt = self.config.optimizer

        if opt in ("adamw", "adamw_torch"):
            kwargs = {}
            if torch.cuda.is_available():
                kwargs["fused"] = True
            return torch.optim.AdamW(params, lr=lr, weight_decay=self.config.weight_decay, **kwargs)
        elif opt in ("adamw_8bit", "adamw_bnb_8bit"):
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=self.config.weight_decay)
        elif opt == "paged_adamw_8bit":
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=self.config.weight_decay)
        elif opt == "adafactor":
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=lr, relative_step=False, scale_parameter=False)
        elif opt == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

    def _save_checkpoint(self):
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(checkpoint_dir)
        self._fire("on_save", path=checkpoint_dir)
        self._log_info(f"Checkpoint saved to {checkpoint_dir}")

        if self.accelerator.is_main_process and self.config.save_total_limit:
            self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        checkpoints = sorted(
            glob.glob(os.path.join(self.config.output_dir, "checkpoint-*")),
            key=lambda x: int(x.rsplit("-", 1)[-1]),
        )
        if len(checkpoints) > self.config.save_total_limit:
            for old in checkpoints[: len(checkpoints) - self.config.save_total_limit]:
                shutil.rmtree(old)
                logger.debug("Deleted old checkpoint: %s", old)

    def _log_metrics(self, metrics):
        if self.config.log_with:
            self.accelerator.log(metrics, step=self.global_step)
        if self.accelerator.is_main_process:
            parts = [f"{k}: {_fmt(v)}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]
            logger.info("[step %d] %s", self.global_step, " | ".join(parts))

    def _log_info(self, msg):
        if self.accelerator.is_main_process:
            logger.info(msg)

    def _fire(self, event, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn:
                fn(self, **kwargs)


def _fmt(v):
    """Format a float for logging."""
    if abs(v) < 1e-3 and v != 0.0:
        return f"{v:.2e}"
    return f"{v:.4f}"


def _is_cuda_error(exc):
    """Check if a RuntimeError is a fatal CUDA error."""
    msg = str(exc).lower()
    return "cuda error" in msg or ("cuda" in msg and "illegal" in msg)


def _config_to_dict(config):
    """Convert a dataclass config to a dict for experiment tracking."""
    return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
