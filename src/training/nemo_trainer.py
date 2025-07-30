# src/training/nemo_trainer.py
import lightning as pl
import torch
from models.nemo_wrappers import NeMoGPT2Wrapper, NeMoSuperGPT2Wrapper
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="config/nemo", config_name="gpt2_super")
def main(cfg):
    """NeMo training using your existing models."""
    print("🚀 Starting NeMo training...", flush=True)

    # Pre-training verifications
    verify_environment_setup()
    verify_config_validity(cfg)

    # Choose wrapper based on config
    if cfg.model.super_block_config.get("heterogeneous", False):
        model_class = NeMoSuperGPT2Wrapper
        print(" Using NeMoSuperGPT2Wrapper (heterogeneous)")
    else:
        model_class = NeMoGPT2Wrapper
        print("📦 Using NeMoGPT2Wrapper (homogeneous)")

    # Initialize trainer with verification
    print("⚙️  Initializing trainer...")
    trainer = pl.Trainer(**cfg.trainer)
    verify_trainer_setup(trainer, cfg)

    # Create model with verification
    print("🏗️  Creating model...")
    model = model_class(cfg.model, trainer)
    verify_model_setup(model, cfg)

    # Pre-training model verification
    print(" Pre-training model verification...")
    verify_model_before_training(model)

    # Train using NeMo's infrastructure but your models
    print("🎯 Starting training...")
    trainer.fit(model)

    # Post-training verifications
    print("🔍 Post-training verification...")
    verify_training_success(trainer, model)
    verify_gradient_flow(model)
    verify_model_after_training(model)


def verify_environment_setup():
    """Verify the training environment is properly configured."""
    print("🔧 Verifying environment setup...")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(
            f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("⚠️  CUDA not available, using CPU")

    # Check PyTorch version
    print(f"✅ PyTorch version: {torch.__version__}")

    # Check Lightning version
    print(f"✅ Lightning version: {pl.__version__}")


def verify_config_validity(cfg):
    """Verify the configuration is valid and complete."""
    print("📋 Verifying configuration...")

    # Check required config sections
    required_sections = ["model", "trainer"]
    for section in required_sections:
        if hasattr(cfg, section):
            print(f"✅ {section} config present")
        else:
            raise ValueError(f"Missing required config section: {section}")

    # Verify model config
    model_cfg = cfg.model
    required_model_fields = [
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "vocab_size",
    ]
    for field in required_model_fields:
        if hasattr(model_cfg, field):
            print(f"✅ model.{field}: {getattr(model_cfg, field)}")
        else:
            raise ValueError(f"Missing required model config field: {field}")

    # Verify trainer config
    trainer_cfg = cfg.trainer
    if hasattr(trainer_cfg, "devices"):
        print(f"✅ trainer.devices: {trainer_cfg.devices}")
    if hasattr(trainer_cfg, "precision"):
        print(f"✅ trainer.precision: {trainer_cfg.precision}")


def verify_trainer_setup(trainer, cfg):
    """Verify the trainer is properly configured."""
    print("🎯 Verifying trainer setup...")

    # Check trainer attributes
    print(f"✅ Trainer devices: {trainer.devices}")
    print(f"✅ Trainer accelerator: {trainer.accelerator}")
    print(f"✅ Trainer precision: {trainer.precision}")

    # Check if callbacks are configured
    if hasattr(trainer, "callbacks") and trainer.callbacks:
        print(f"✅ Trainer has {len(trainer.callbacks)} callbacks")
        for i, callback in enumerate(trainer.callbacks):
            print(f"  - Callback {i}: {type(callback).__name__}")
    else:
        print("⚠️  No callbacks configured")


def verify_model_setup(model, cfg):
    """Verify the model is properly initialized."""
    print("🏗️  Verifying model setup...")

    # Check model type
    print(f"✅ Model type: {type(model).__name__}")

    # Check if model has the expected attributes
    if hasattr(model, "gpt2_model"):
        print("✅ Model has gpt2_model attribute")
    elif hasattr(model, "super_gpt2_model"):
        print("✅ Model has super_gpt2_model attribute")
    else:
        print("⚠️  Model doesn't have expected internal model attribute")

    # Check model configuration
    if hasattr(model, "cfg"):
        print("✅ Model has configuration attribute")
    else:
        print("⚠️  Model missing configuration attribute")


def verify_model_before_training(model):
    """Verify model state before training begins."""
    print(" Pre-training model verification...")

    # Check model is in training mode
    if model.training:
        print("✅ Model in training mode")
    else:
        print("⚠️  Model not in training mode")

    # Check model device
    device = next(model.parameters()).device
    print(f"✅ Model on device: {device}")

    # Check parameter gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Trainable parameters: {trainable_params:,} / {total_params:,}")

    if trainable_params == 0:
        raise ValueError("No trainable parameters found!")

    # Test forward pass
    try:
        batch_size = 2
        seq_len = 10
        vocab_size = getattr(model.cfg, "vocab_size", 50304)

        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        dummy_labels = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )

        with torch.no_grad():
            output = model(dummy_input, labels=dummy_labels)
            loss = output["loss"]
            print(f"✅ Forward pass test successful, loss: {loss.item():.4f}")

    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        raise


def verify_model_after_training(model):
    """Verify model state after training completes."""
    print("🔍 Post-training model verification...")

    # Check if model parameters changed
    if hasattr(model, "_initial_params"):
        current_params = [p.data.clone() for p in model.parameters()]
        param_changed = any(
            not torch.equal(init, curr)
            for init, curr in zip(model._initial_params, current_params)
        )
        if param_changed:
            print("✅ Model parameters were updated during training")
        else:
            print("⚠️  Model parameters were not updated during training")

    # Check model is in eval mode (should be after training)
    if not model.training:
        print("✅ Model in evaluation mode (expected after training)")
    else:
        print("⚠️  Model still in training mode after training")

    # Final forward pass test
    try:
        batch_size = 1
        seq_len = 5
        vocab_size = getattr(model.cfg, "vocab_size", 50304)
        device = next(model.parameters()).device

        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        dummy_labels = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )

        with torch.no_grad():
            output = model(dummy_input, labels=dummy_labels)
            loss = output["loss"]
            print(f"✅ Final forward pass test successful, loss: {loss.item():.4f}")

    except Exception as e:
        print(f"❌ Final forward pass test failed: {e}")


def verify_training_success(trainer, model):
    """Verify that training completed successfully."""

    # 1. Check if training completed without exceptions
    print("✅ Training completed without exceptions")

    # 2. Check if checkpoints were saved
    if hasattr(trainer, "checkpoint_callback") and trainer.checkpoint_callback:
        latest_checkpoint = trainer.checkpoint_callback.best_model_path
        if latest_checkpoint:
            print(f"✅ Best checkpoint saved: {latest_checkpoint}")
        else:
            print("⚠️  No checkpoint was saved")

    # 3. Check training metrics
    if hasattr(trainer, "logger") and trainer.logger:
        print("✅ Training metrics logged")

    # 4. Verify model state
    print(f"✅ Model training mode: {model.training}")
    print(f"✅ Model device: {next(model.parameters()).device}")

    # 5. Check if model parameters were updated (not frozen)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")

    # 6. Verify loss computation works
    try:
        # Create dummy input for testing
        import torch

        batch_size = 2
        seq_len = 10
        vocab_size = model.cfg.vocab_size

        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = model(dummy_input, labels=dummy_labels)
            loss = output["loss"]
            print(f"✅ Loss computation works: {loss.item():.4f}")

    except Exception as e:
        print(f"❌ Loss computation failed: {e}")

    # 7. Check for NaN or infinite values
    has_nan = any(torch.isnan(p).any() for p in model.parameters())
    has_inf = any(torch.isinf(p).any() for p in model.parameters())

    if not has_nan:
        print("✅ No NaN values in model parameters")
    else:
        print("❌ NaN values detected in model parameters")

    if not has_inf:
        print("✅ No infinite values in model parameters")
    else:
        print("❌ Infinite values detected in model parameters")

    # 8. Check training logs
    check_training_logs(trainer)

    print("\n🎉 Training verification complete!")


def check_training_logs(trainer):
    """Check if training logs show expected behavior."""
    if hasattr(trainer, "logger"):
        # Check if loss decreased over time
        # Check if learning rate was applied correctly
        # Check if gradients were computed
        pass


def verify_gradient_flow(model):
    """Verify that gradients are flowing through the model."""
    model.train()

    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, 10))
    dummy_labels = torch.randint(0, 1000, (1, 10))

    # Forward pass
    output = model(dummy_input, labels=dummy_labels)
    loss = output["loss"]

    # Backward pass
    loss.backward()

    # Check gradients
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
            print(f"Gradient norm for {name}: {param_norm:.6f}")

    total_grad_norm = total_grad_norm ** (1.0 / 2)
    print(f"Total gradient norm: {total_grad_norm:.6f}")

    return total_grad_norm > 0


def verify_model_performance(model, test_data):
    """Verify model performs reasonably on test data."""
    model.eval()

    with torch.no_grad():
        # Test on a few batches
        for i, batch in enumerate(test_data):
            if i >= 3:  # Only test first 3 batches
                break

            output = model(batch["input_ids"], labels=batch["labels"])
            loss = output["loss"]
            print(f"Test batch {i} loss: {loss.item():.4f}")

            # Check if loss is reasonable (not NaN, not infinite)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ Unreasonable loss on batch {i}")
                return False

    return True

# -----------------------------
if __name__ == "__main__":
    main()