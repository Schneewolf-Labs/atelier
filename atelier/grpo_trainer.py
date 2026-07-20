"""FlowGRPOTrainer — the online Flow-GRPO / DanceGRPO loop.

Inverts the offline SFT path. There are NO target images: every step SAMPLES images
by running the SDE denoising trajectory with the *current* policy (via the adapter's
``sample_with_logprobs``), decodes them to pixels, scores them through a single
synchronous ``ImageRewardFn``, and does policy gradient over the trajectory.

Consequences honored here:
  * The VAE decoder must stay resident — we never call ``free_encoders`` in GRPO.
  * Text embeddings can still be cached (PromptDataset handles that).
  * The reference policy is the same model with its LoRA adapter DISABLED — no second
    model copy.

Same lifecycle/callbacks/checkpointing surface as AtelierTrainer, so the existing
callbacks (MidTrainingSampleCallback, WebSocketCallback) keep working.
"""

import contextlib
import logging
import math
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from .config import FlowGRPOConfig
from .data.prompt import PromptCollator
from .losses.flow_grpo import FlowGRPOLoss
from .rewards import sanitize_rewards
from .trainer import _config_to_dict, _fmt

logger = logging.getLogger(__name__)


class FlowGRPOTrainer:
    """On-policy Flow-GRPO trainer.

    Args:
        adapter:      a ModelAdapter that also satisfies SupportsSDESampling.
        config:       FlowGRPOConfig.
        loss_fn:      FlowGRPOLoss (owns advantage normalization + the PPO/KL math).
        reward_fn:    the single ImageRewardFn injection point.
        train_dataset: a PromptDataset.
        peft_config:  optional LoRA config (=> reference-via-disabled-adapter).
        callbacks:    optional list of TrainerCallback.
    """

    def __init__(
        self,
        *,
        adapter,
        config: FlowGRPOConfig,
        loss_fn: FlowGRPOLoss,
        reward_fn,
        train_dataset,
        peft_config=None,
        callbacks=None,
    ):
        self.adapter = adapter
        self.config = config
        self.loss_fn = loss_fn
        self.reward_fn = reward_fn
        self.callbacks = callbacks or []
        self.global_step = 0
        self.current_epoch = 0
        self._stop_requested = False
        self.last_samples = []  # [(prompt, PIL.Image, reward)] from the most recent step

        model = adapter.model

        if peft_config is not None:
            from peft import get_peft_model

            model = get_peft_model(model, peft_config)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
        self._has_peft = peft_config is not None

        if config.gradient_checkpointing:
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
            elif hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

        # One *prompt* per dataloader item; G generations are expanded per step.
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=PromptCollator(),
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            drop_last=True,
        )

        optimizer = self._create_optimizer(model)

        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / config.gradient_accumulation_steps
        )
        self.max_steps = num_update_steps_per_epoch * config.num_epochs

        warmup_steps = (
            config.warmup_steps if config.warmup_steps > 0 else int(self.max_steps * config.warmup_ratio)
        )
        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.max_steps,
        )

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, optimizer, self.train_dataloader, lr_scheduler
        )

        # The adapter samples + recomputes logprobs against the trainable policy. Hand
        # it the prepared model so rollouts and gradients share one set of parameters.
        self._bind_adapter_model(self.model)

        if config.log_with:
            self.accelerator.init_trackers(
                config.project_name or "atelier",
                config=_config_to_dict(config),
                init_kwargs=tracker_kwargs,
            )

        if config.resume_from_checkpoint:
            self.accelerator.load_state(config.resume_from_checkpoint)
            try:
                self.global_step = int(config.resume_from_checkpoint.rstrip("/").split("-")[-1])
            except ValueError:
                logger.warning("Could not parse step from checkpoint path, starting from step 0")

    # ---- Training loop ----

    def train(self):
        config = self.config

        self._log_info("***** Starting Flow-GRPO training *****")
        self._log_info(f"  Num prompts = {len(self.train_dataloader.dataset)}")
        self._log_info(f"  Generations per prompt (G) = {config.num_generations}")
        self._log_info(f"  Inference steps per sample = {config.num_inference_steps}")
        self._log_info(f"  Total optimization steps = {self.max_steps}")

        self._fire("on_train_begin")

        progress_bar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            desc="GRPO",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
        )

        for epoch in range(config.num_epochs):
            self.current_epoch = epoch
            self._fire("on_epoch_begin", epoch=epoch)
            self.model.train()

            for batch in self.train_dataloader:
                metrics = self._rollout_and_update(batch)

                self.global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    reward=f"{metrics['reward/mean']:.3f}", loss=f"{metrics['loss']:.4f}"
                )
                self._fire("on_step_end", step=self.global_step, loss=metrics["loss"], metrics=metrics)

                if self.global_step % config.logging_steps == 0:
                    log_metrics = {f"train/{k}": v for k, v in metrics.items()}
                    log_metrics["train/global_step"] = self.global_step
                    self._log_metrics(log_metrics)
                    self._fire("on_log", metrics=log_metrics)

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

        progress_bar.close()
        self._fire("on_train_end")
        if config.log_with:
            self.accelerator.end_training()
        self._log_info("***** Training complete *****")

    def _rollout_and_update(self, batch):
        config = self.config
        prompts = batch["prompts"]
        generator = torch.Generator(device=self._generator_device()).manual_seed(
            config.seed + self.global_step
        )

        # 1) Group-expand the conditioning by G and sample a trajectory per image.
        prompt_embeddings = self._group_expand(self._prompt_embeddings(batch), config.num_generations)
        rollout = self.adapter.sample_with_logprobs(
            prompt_embeddings=prompt_embeddings,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            sde_noise_level=config.sde_noise_level,
            generator=generator,
            decode=True,
        )

        # 2) Score the decoded images. One synchronous reward call; we block here.
        columns = {k: _broadcast(v, config.num_generations) for k, v in batch["columns"].items()}
        raw_rewards = self.reward_fn(prompts, rollout.images, **columns)
        rewards = torch.tensor(
            sanitize_rewards(raw_rewards, expected=rollout.num_samples), dtype=torch.float32
        )
        self.last_samples = self._collect_samples(prompts, rollout.images, rewards)

        # 3) Group-normalize -> advantages.
        advantages = self.loss_fn.compute_advantages(rewards, config.num_generations)
        advantages = advantages.to(self.accelerator.device)

        # 4 + 5) PPO inner epochs: recompute logprobs (current / old / ref), step.
        old_logprobs = rollout.logprobs.detach().to(self.accelerator.device)
        ref_logprobs = None
        if self.loss_fn.beta != 0.0:
            ref_logprobs = self._reference_logprobs(rollout).to(self.accelerator.device)

        last_out = None
        for _ in range(config.num_iterations):
            with self.accelerator.accumulate(self.model):
                logprobs = self.adapter.recompute_logprobs(rollout, with_grad=True).to(
                    self.accelerator.device
                )
                step_mask = self._timestep_subsample(rollout.step_mask).to(self.accelerator.device)

                out = self.loss_fn(
                    logprobs=logprobs,
                    old_logprobs=old_logprobs,
                    ref_logprobs=ref_logprobs,
                    advantages=advantages,
                    step_mask=step_mask,
                )
                self.accelerator.backward(out.loss)
                if config.max_grad_norm and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()
                if not self.accelerator.optimizer_step_was_skipped:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                last_out = out

        return {
            "loss": last_out.loss.detach().item(),
            "reward/mean": rewards.mean().item(),
            "reward/std": rewards.std(unbiased=False).item(),
            "reward/max": rewards.max().item(),
            "kl": last_out.kl,
            "clip_frac": last_out.clip_frac,
            "ratio_mean": last_out.ratio_mean,
            "learning_rate": self.lr_scheduler.get_last_lr()[0],
        }

    # ---- Rollout helpers ----

    def _prompt_embeddings(self, batch):
        """Cached text embeddings if PromptDataset cached them, else encode on the fly."""
        text_embeddings = batch.get("text_embeddings")
        if text_embeddings is not None:
            return {
                k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v)
                for k, v in text_embeddings.items()
            }
        return self.adapter.encode_text(batch["prompts"], device=self.accelerator.device)

    @staticmethod
    def _group_expand(embeddings, num_generations):
        """Repeat each prompt's conditioning G times along the batch dim (group-major)."""
        expanded = {}
        for key, value in embeddings.items():
            if isinstance(value, torch.Tensor):
                expanded[key] = value.repeat_interleave(num_generations, dim=0)
            else:
                expanded[key] = _broadcast(value, num_generations)
        return expanded

    def _reference_logprobs(self, rollout):
        """Logprobs under the reference policy = the model with LoRA disabled."""
        with torch.no_grad(), self._reference_context():
            return self.adapter.recompute_logprobs(rollout, with_grad=False)

    def _reference_context(self):
        unwrapped = self.accelerator.unwrap_model(self.model)
        if self._has_peft and hasattr(unwrapped, "disable_adapter"):
            return unwrapped.disable_adapter()
        return contextlib.nullcontext()

    def _timestep_subsample(self, step_mask):
        """DDPO trick — keep a random fraction of valid steps for the backward pass."""
        if self.config.timestep_fraction >= 1.0:
            return step_mask
        keep = torch.bernoulli(torch.full(step_mask.shape, self.config.timestep_fraction))
        return step_mask * keep.to(step_mask.dtype)

    def _collect_samples(self, prompts, images, rewards, limit=4):
        """A few (prompt, image, reward) tuples for the wandb table / websocket preview."""
        if not images:
            return []
        samples = []
        g = self.config.num_generations
        for i in range(min(limit, len(images))):
            samples.append((prompts[i // g], images[i], float(rewards[i].item())))
        return samples

    def _generator_device(self):
        device = self.accelerator.device
        # CUDA generators must be created on-device; CPU otherwise.
        return device if device.type == "cuda" else "cpu"

    # ---- Lifecycle ----

    def request_stop(self):
        self._stop_requested = True
        self._log_info("Stop requested — will stop after current step")

    @property
    def stopped_early(self):
        return self._stop_requested

    def save_model(self, output_dir=None):
        """Save the LoRA / full-model artifact via the adapter (same shape as AtelierTrainer)."""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            if hasattr(unwrapped, "peft_config"):
                self.adapter.save_lora(unwrapped, output_dir)
            else:
                self.adapter.save_model(unwrapped, output_dir)
            self._log_info(f"Model saved to {output_dir}")

    # ---- Internal helpers ----

    def _bind_adapter_model(self, model):
        """Point the adapter at the prepared policy model so sampling uses it."""
        if hasattr(self.adapter, "set_model"):
            self.adapter.set_model(model)
        elif hasattr(self.adapter, "_model"):
            self.adapter._model = model

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


def _broadcast(values, num_generations):
    """Expand a per-prompt list to per-image (group-major) ordering."""
    return [v for v in values for _ in range(num_generations)]
