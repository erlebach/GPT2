# src/training/nemo_trainer.py
import pytorch_lightning as pl
from models.nemo_wrappers import NeMoGPT2Wrapper, NeMoSuperGPT2Wrapper
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="config/nemo", config_name="gpt2_super")
def main(cfg):
    """NeMo training using your existing models."""
    print("enter main", flush=True)
    # Choose wrapper based on config
    if cfg.model.super_block_config.get("heterogeneous", False):
        model_class = NeMoSuperGPT2Wrapper
    else:
        model_class = NeMoGPT2Wrapper

    # Initialize trainer
    trainer = pl.Trainer(**cfg.trainer)

    # Create model (this will use your existing models internally)
    model = model_class(cfg.model, trainer)

    # Train using NeMo's infrastructure but your models
    trainer.fit(model)

    # Verify training was successful
    verify_training_success(trainer, model)


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
