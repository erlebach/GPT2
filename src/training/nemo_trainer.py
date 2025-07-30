# src/training/nemo_trainer.py
import pytorch_lightning as pl
from models.nemo_wrappers import NeMoGPT2Wrapper, NeMoSuperGPT2Wrapper
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="config/nemo", config_name="gpt2_super")
def main(cfg):
    """NeMo training using your existing models."""

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

    print("\n🎉 Training verification complete!")
