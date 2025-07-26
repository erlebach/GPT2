"""Lightning module for GPT-2 training with SuperBlocks."""

import math
import time
from typing import Any, Dict, Optional

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from data_utils import TextDataModule
from jaxtyping import Float, Integer
from model import GPT, GPTConfig  # Updated import
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


@beartype
class GPTLightningModule(pl.LightningModule):
    """Lightning module for training GPT-2 with SuperBlocks.

    This module handles training and validation steps, optimizer configuration,
    and learning rate scheduling.

    Args:
        config: GPT configuration parameters.
        weight_decay: Weight decay for optimizer.
        learning_rate: Initial learning rate.
        warmup_steps: Number of warmup steps for learning rate.
        max_steps: Maximum number of training steps.
        min_lr_ratio: Ratio of minimum learning rate to maximum learning rate.
    """

    def __init__(
        self,
        config: GPTConfig,
        weight_decay: float = 0.1,
        learning_rate: float = 6e-2,
        warmup_steps: int = 10,
        max_steps: int = 500,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

        # Initialize the GPT model
        self.model = GPT(config)

        # Track metrics
        self.train_losses = []
        self.val_losses = []

    def forward(
        self,
        idx: Integer[Tensor, "b seq"],
        targets: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Forward pass through the GPT model.

        Args:
            idx: Input token indices of shape (batch_size, sequence_length).
            targets: Target token indices for loss calculation.

        Returns:
            Tuple of (logits, loss) where loss is None if targets is None.
        """
        return self.model(idx, targets)

    def training_step(
        self,
        batch: tuple[Integer[Tensor, "b seq"], Integer[Tensor, "b seq"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        """Training step for a single batch.

        Args:
            batch: Tuple of (input_tokens, target_tokens).
            batch_idx: Index of the current batch.

        Returns:
            Training loss for the batch.
        """
        x, y = batch
        logits, loss = self(x, y)

        # Log training loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Store loss for potential custom logging
        self.train_losses.append(loss.detach().cpu().item())

        return loss

    def validation_step(
        self,
        batch: tuple[Integer[Tensor, "b seq"], Integer[Tensor, "b seq"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        """Validation step for a single batch.

        Args:
            batch: Tuple of (input_tokens, target_tokens).
            batch_idx: Index of the current batch.

        Returns:
            Validation loss for the batch.
        """
        x, y = batch
        logits, loss = self(x, y)

        # Log validation loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Store loss for potential custom logging
        self.val_losses.append(loss.detach().cpu().item())

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary containing optimizer and scheduler configuration.
        """
        # Get parameters and separate by dimension for weight decay
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Log parameter counts
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        self.log("num_decay_params", num_decay_params)
        self.log("num_nodecay_params", num_nodecay_params)

        # Create optimizer
        optimizer = AdamW(
            optim_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Create learning rate scheduler
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=self._get_lr_lambda(),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _get_lr_lambda(self):
        """Create learning rate lambda function for scheduler.

        Returns:
            Lambda function that computes learning rate based on current step.
        """
        max_lr = self.learning_rate
        min_lr = max_lr * self.min_lr_ratio

        def lr_lambda(step: int) -> float:
            # Linear warmup for warmup_steps
            if step < self.warmup_steps:
                return max_lr * (step + 1) / self.warmup_steps

            # Return min lr if step > max_steps
            if step > self.max_steps:
                return min_lr

            # Use cosine decay if in between
            decay_ratio = (step - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)

        return lr_lambda

    def on_train_start(self) -> None:
        """Called when training starts."""
        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.log("total_params", total_params)
        self.log("trainable_params", trainable_params)

        print(f"Starting training with {total_params:,} total parameters")
        print(f"({trainable_params:,} trainable)")

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Log epoch-level metrics
        if self.train_losses:
            avg_train_loss = sum(self.train_losses) / len(self.train_losses)
            self.log("epoch_train_loss", avg_train_loss)
            self.train_losses.clear()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        # Log epoch-level metrics
        if self.val_losses:
            avg_val_loss = sum(self.val_losses) / len(self.val_losses)
            self.log("epoch_val_loss", avg_val_loss)
            self.val_losses.clear()


def train_with_lightning(
    data_path: str = "input.txt",
    block_size: int = 64,
    batch_size: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 128,
    vocab_size: int = 50304,
    learning_rate: float = 6e-2,
    weight_decay: float = 0.1,
    warmup_steps: int = 10,
    max_steps: int = 500,
    val_split: float = 0.1,
    num_workers: int = 0,
    accelerator: str = "auto",
    devices: int = 1,
    precision: str = "32-true",
) -> None:
    """Train GPT-2 model using Lightning.

    Args:
        data_path: Path to the input text file.
        block_size: Length of each sequence (context window).
        batch_size: Batch size for training.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        vocab_size: Size of the vocabulary.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps for learning rate.
        max_steps: Maximum number of training steps.
        val_split: Fraction of data to use for validation.
        num_workers: Number of DataLoader workers.
        accelerator: Lightning accelerator type.
        devices: Number of devices to use.
        precision: Training precision.
    """
    # Set random seed for reproducibility
    pl.seed_everything(1337)

    # Create configuration
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )

    # Create data module
    data_module = TextDataModule(
        data_path=data_path,
        block_size=block_size,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
    )

    # Create model
    model = GPTLightningModule(
        config=config,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=1,
        val_check_interval=0.25,  # Validate every 25% of training
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        logger=True,
    )

    # Train the model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    # Test the Lightning module
    print("Testing GPTLightningModule...")

    # Create a small test configuration
    test_config = GPTConfig(
        block_size=32,
        vocab_size=1000,
        n_layer=1,
        n_head=2,
        n_embd=64,
    )

    # Create model
    model = GPTLightningModule(
        config=test_config,
        learning_rate=1e-3,
        warmup_steps=5,
        max_steps=10,
    )

    # Test forward pass
    batch_size = 4
    seq_len = 32
    x = torch.randint(0, 1000, (batch_size, seq_len))
    y = torch.randint(0, 1000, (batch_size, seq_len))

    logits, loss = model(x, y)
    assert logits.shape == (batch_size, seq_len, 1000)
    assert loss is not None and loss.item() > 0
    print("Test 1 passed: Forward pass works correctly.")

    # Test optimizer configuration
    optimizer_config = model.configure_optimizers()
    assert "optimizer" in optimizer_config
    assert "lr_scheduler" in optimizer_config
    print("Test 2 passed: Optimizer configuration works correctly.")

    # Test training step
    loss = model.training_step((x, y), batch_idx=0)
    assert loss.shape == ()  # Scalar tensor
    print("Test 3 passed: Training step works correctly.")

    # Test validation step
    loss = model.validation_step((x, y), batch_idx=0)
    assert loss.shape == ()  # Scalar tensor
    print("Test 4 passed: Validation step works correctly.")

    print("All tests passed! GPTLightningModule is working correctly.")
