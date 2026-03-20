# Callbacks

Callbacks let you hook into the training loop without modifying the trainer. Subclass `TrainerCallback` and override the methods you need.

## Available hooks

| Hook | Called when | Arguments |
|---|---|---|
| `on_train_begin` | Training starts | `trainer` |
| `on_train_end` | Training ends | `trainer` |
| `on_epoch_begin` | Epoch starts | `trainer`, `epoch` |
| `on_epoch_end` | Epoch ends | `trainer`, `epoch` |
| `on_step_end` | After each optimization step | `trainer`, `step`, `loss`, `metrics` |
| `on_log` | After logging metrics | `trainer`, `metrics` |
| `on_evaluate` | After evaluation | `trainer`, `metrics` |
| `on_save` | After saving a checkpoint | `trainer`, `path` |

## Examples

### Early stopping

```python
from atelier import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0

    def on_evaluate(self, trainer, metrics):
        loss = metrics.get("eval/loss", float("inf"))
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping at step {trainer.global_step}")
                trainer.request_stop()
```

### Custom logging

```python
class PrintCallback(TrainerCallback):
    def on_step_end(self, trainer, step, loss, metrics):
        if step % 50 == 0:
            print(f"Step {step}/{trainer.max_steps} | loss={loss:.4f} | {metrics}")

    def on_epoch_end(self, trainer, epoch):
        print(f"--- Epoch {epoch + 1} complete ---")
```

### Save best model

```python
class SaveBestCallback(TrainerCallback):
    def __init__(self, output_dir="./best_model"):
        self.output_dir = output_dir
        self.best_loss = float("inf")

    def on_evaluate(self, trainer, metrics):
        loss = metrics.get("eval/loss", float("inf"))
        if loss < self.best_loss:
            self.best_loss = loss
            trainer.save_model(self.output_dir)
            print(f"New best model saved (loss={loss:.4f})")
```

## Usage

Pass callbacks as a list to the trainer:

```python
trainer = AtelierTrainer(
    adapter=adapter,
    config=config,
    loss_fn=loss_fn,
    train_dataset=dataset,
    callbacks=[
        EarlyStoppingCallback(patience=10),
        PrintCallback(),
    ],
)
```
