from atelier import TrainerCallback, TrainingConfig


def test_config_defaults():
    config = TrainingConfig()
    assert config.output_dir == "./output"
    assert config.num_epochs == 3
    assert config.batch_size == 1
    assert config.learning_rate == 1e-4
    assert config.mixed_precision == "bf16"
    assert config.gradient_checkpointing is True
    assert config.optimizer == "adamw"
    assert config.lr_scheduler == "cosine"
    assert config.seed == 42


def test_config_override():
    config = TrainingConfig(
        output_dir="/tmp/test",
        num_epochs=50,
        batch_size=2,
        learning_rate=5e-5,
    )
    assert config.output_dir == "/tmp/test"
    assert config.num_epochs == 50
    assert config.batch_size == 2
    assert config.learning_rate == 5e-5


def test_callback_interface():
    cb = TrainerCallback()
    # All methods should be callable and return None
    assert cb.on_train_begin(None) is None
    assert cb.on_train_end(None) is None
    assert cb.on_epoch_begin(None, 0) is None
    assert cb.on_epoch_end(None, 0) is None
    assert cb.on_step_end(None, 0, 0.0, {}) is None
    assert cb.on_log(None, {}) is None
    assert cb.on_evaluate(None, {}) is None
    assert cb.on_save(None, "/tmp") is None
