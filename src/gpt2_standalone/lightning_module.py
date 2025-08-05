"""Lightning module for GPT-2 training with SuperBlocks."""

import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
import torch
from beartype import beartype

# Clean absolute imports
# Assumes that src/ is in sys.path, else need from src.models ...
# from models.gpt2.model import GPT, GPTConfig
from gpt2_standalone.model import GPT, GPTConfig
from jaxtyping import Float, Integer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from utils.data_utils import TextDataModule, get_project_root
from utils.metrics_extensions import (
    MetricsModelCheckpoint,
    get_metrics_collector,
    measure_performance,
)


# Might move to utils
def check_gpu_allocation():
    """Check GPU memory allocation across all available devices.

    Returns:
        list[str]: List of allocation status strings for each GPU.

    """
    results = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i)
        results.append(f"GPU {i}: {allocated/1e6:.2f} MB allocated")
    return results


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

    def print_gpu_allocation(self):
        rank_info = f"[Rank {self.global_rank}/{self.trainer.world_size}]"
        device_info = f"Device: {self.device}"
        print(f"{rank_info} {device_info}", flush=True)
        allocs = check_gpu_allocation()
        print(f"[Rank {self.global_rank}] GPU Allocation:")
        for alloc in allocs:
            print(f"[Rank {self.global_rank}] {alloc}", flush=True)

    @measure_performance(memory_enabled=True, timing_enabled=True)
    def training_step(
        self,
        batch: tuple[Integer[Tensor, "b seq"], Integer[Tensor, "b seq"]]
        | list[Integer[Tensor, "b seq"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        """Training step for a single batch.

        Args:
            batch: Tuple or list of (input_tokens, target_tokens).
            batch_idx: Index of the current batch.

        Returns:
            Training loss for the batch.
        """
        # print(f"Training step: Batch {batch_idx}, {batch[0].shape=}", flush=True)
        # if batch_idx <= 2:
        #     print(
        #         f"[Rank {self.global_rank}] Batch shape: {batch[0].shape}"
        #     )  # Add this back!
        #     self.print_gpu_allocation()  # Memory BEFORE forward/backward

        # Handle both tuple and list batch formats
        if isinstance(batch, tuple | list) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError(
                f"Expected batch to be tuple or list of length 2, got {type(batch)} with length {len(batch) if hasattr(batch, '__len__') else 'unknown'}"
            )

        logits, loss = self(x, y)  # Forward pass

        # # Debug loss and gradients
        # if batch_idx == 0:
        #     print(f"[Rank {self.global_rank}] Loss: {loss.item()}")
        #     print(f"[Rank {self.global_rank}] Loss requires grad: {loss.requires_grad}")
        #     print(f"[Rank {self.global_rank}] Loss grad_fn: {loss.grad_fn}")

        #     # Check if model parameters require grad
        #     for name, param in self.named_parameters():
        #         print(
        #             f"[Rank {self.global_rank}] {name} requires_grad: {param.requires_grad}"
        #         )

        # Log training loss
        # self.log(
        #     "train_loss",
        #     loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

        # Store loss for potential custom logging
        self.train_losses.append(loss.detach().cpu().item())

        # ADD THIS: Print memory AFTER forward/backward
        # if batch_idx <= 2:
        #     # print(f"After forward/backward - Batch {batch_idx}")
        #     self.print_gpu_allocation()

        return loss

    # @measure_performance(memory_enabled=True, timing_enabled=True)
    def validation_step(
        self,
        batch: tuple[Integer[Tensor, "b seq"], Integer[Tensor, "b seq"]]
        | list[Integer[Tensor, "b seq"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        """Validate step for a single batch.

        Args:
            batch: Tuple or list of (input_tokens, target_tokens).
            batch_idx: Index of the current batch.

        Returns:
            Validation loss for the batch.
        """
        # if batch_idx >= 0:
        #     print(f"Validation step: Batch {batch_idx}")
        #     self.print_gpu_allocation()

        # Handle both tuple and list batch formats
        if isinstance(batch, tuple | list) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError(
                f"Expected batch to be tuple or list of length 2, got {type(batch)} with length {len(batch) if hasattr(batch, '__len__') else 'unknown'}"
            )

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
        # Add DDP debugging
        print(f"🔍 DDP DEBUG:")
        print(f"   Trainer strategy: {self.trainer.strategy}")
        print(f"   Trainer world_size: {self.trainer.world_size}")
        print(f"   Trainer global_rank: {self.trainer.global_rank}")
        print(f"   Trainer local_rank: {self.trainer.local_rank}")
        print(f"   Trainer is_global_zero: {self.trainer.is_global_zero}")

        # Handle Fabric vs Trainer differences
        try:
            num_devices = self.trainer.num_devices
            print(f"   Trainer num_devices: {num_devices}")
        except AttributeError:
            # When using Fabric, num_devices might not be available
            # Use world_size as a fallback
            num_devices = getattr(self.trainer, "world_size", "unknown")
            print(f"   Trainer num_devices: {num_devices} (from world_size)")

        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.log("total_params", total_params)
        self.log("trainable_params", trainable_params)

        # Log optimizer parameter counts
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        self.log("num_decay_params", num_decay_params)
        self.log("num_nodecay_params", num_nodecay_params)

        print(f"Starting training with {total_params:,} total parameters")
        print(f"({trainable_params:,} trainable)")
        print(
            f"({num_decay_params:,} decay params, {num_nodecay_params:,} no-decay params)"
        )

        # Check NCCL communication
        if torch.distributed.is_initialized():
            print(
                f"[Rank {self.global_rank}] NCCL backend: {torch.distributed.get_backend()}"
            )
            print(
                f"[Rank {self.global_rank}] Process group: {torch.distributed.get_world_size()}"
            )

            # Test communication
            tensor = torch.tensor([self.global_rank], device=self.device)
            torch.distributed.all_reduce(tensor)
            print(f"[Rank {self.global_rank}] All-reduce test result: {tensor.item()}")

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Log epoch-level metrics
        if self.train_losses:
            avg_train_loss = sum(self.train_losses) / len(self.train_losses)
            self.log("epoch_train_loss", avg_train_loss, sync_dist=True)
            self.train_losses.clear()

        # # Check if models are synchronized
        # if torch.distributed.is_initialized():
        #     for name, param in self.named_parameters():
        #         torch.distributed.broadcast(param.data, src=0)
        #         if self.global_rank == 0:
        #             print(f"Parameter {name} synchronized across GPUs")

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        # Log epoch-level metrics
        if self.val_losses:
            avg_val_loss = sum(self.val_losses) / len(self.val_losses)
            self.log("epoch_val_loss", avg_val_loss, sync_dist=True)
            self.val_losses.clear()


# @measure_performance(memory_enabled=True, timing_enabled=True)
def train_with_lightning(
    data_path: Path | None = None,
    block_size: int = 64,
    batch_size: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 128,
    n_blocks_per_super: int = 2,  # NEW: Number of blocks per SuperBlock
    vocab_size: int = 50304,
    learning_rate: float = 6e-2,
    weight_decay: float = 0.1,
    warmup_steps: int = 10,
    max_steps: int | None = None,
    max_epochs: int | None = None,
    val_split: float = 0.1,
    num_workers: int = 0,
    accelerator: str = "auto",
    devices: str | int = "auto",
    strategy: str | Any = "auto",
    precision: str = "32-true",
    checkpoint_path: Path = Path("src/gpt2_standalone/checkpoints/"),
    checkpoint: str | None = None,
    callbacks: list | None = None,
) -> None:
    """Train GPT-2 model using Lightning.

    Args:
        data_path: Path to the input text file.
        block_size: Length of each sequence (context window).
        batch_size: Batch size for training.
        n_layer: Number of super-layers (SuperBlocks).
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        n_blocks_per_super: Number of blocks within each SuperBlock.
        vocab_size: Size of the vocabulary.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps for learning rate.
        max_steps: Maximum number of training steps (optional).
        max_epochs: Maximum number of training epochs (optional).
        val_split: Fraction of data to use for validation.
        num_workers: Number of DataLoader workers.
        accelerator: Lightning accelerator type ('auto', 'cpu', 'gpu').
        devices: Number of devices to use ('auto' or integer).
        precision: Training precision.
    """
    # Set random seed for reproducibility
    pl.seed_everything(1337)

    if data_path is None:
        data_path = get_project_root() / "data" / "input.txt"

    # Create configuration
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_blocks_per_super=n_blocks_per_super,  # NEW: Pass the new parameter
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
        max_steps=max_steps or 1000,
    )
    checkpoint_callback = MetricsModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=2,
        mode="min",
        monitor="val_loss",
        filename="model-epoch{epoch:02d}-val_loss{val_loss:.2f}",
    )

    ckpt_path = None if checkpoint is None else checkpoint_path / checkpoint

    # Prepare callbacks list
    default_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2000,
            mode="min",
        ),
        checkpoint_callback,
    ]

    # Add custom callbacks if provided
    if callbacks:
        default_callbacks.extend(callbacks)

    # Debug trainer configuration
    print(f"\n🔧 TRAINER CONFIGURATION:")
    print(f"   Accelerator: {accelerator}")
    print(f"   Devices: {devices}")
    print(f"   Strategy: {strategy}")
    print(f"   Strategy Type: {type(strategy).__name__}")
    print(f"   Precision: {precision}")
    print(f"   Callbacks: {len(default_callbacks)} callbacks")

    # Create trainer with automatic device detection
    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": devices,
        "strategy": strategy,
        "precision": precision,
        "log_every_n_steps": 50,
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "enable_checkpointing": True,
        "barebones": False,  # Default False
        "logger": True,
        "default_root_dir": Path.cwd(),
        "callbacks": default_callbacks,
        "max_epochs": 10,
        "max_steps": max_steps or 100,
    }

    # Add max_steps and max_epochs if specified
    if max_steps is not None:
        trainer_kwargs["max_steps"] = max_steps
    if max_epochs is not None:
        trainer_kwargs["max_epochs"] = max_epochs

    print(f"   Final trainer kwargs: {trainer_kwargs}")
    trainer = pl.Trainer(**trainer_kwargs)

    # Debug trainer after creation - FIXED ATTRIBUTES
    print(f"\n🔧 TRAINER CREATED:")
    print(f"   Trainer strategy: {trainer.strategy}")
    print(f"   Trainer accelerator: {trainer.accelerator}")
    print(f"   Trainer device_ids: {getattr(trainer, 'device_ids', 'Not available')}")
    print(f"   Trainer num_devices: {getattr(trainer, 'num_devices', 'Not available')}")

    # Train the model
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


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

    # Test training step with tuple
    loss = model.training_step((x, y), batch_idx=0)
    assert loss.shape == ()  # Scalar tensor
    print("Test 3 passed: Training step with tuple works correctly.")

    # Test training step with list
    loss = model.training_step([x, y], batch_idx=0)
    assert loss.shape == ()  # Scalar tensor
    print("Test 4 passed: Training step with list works correctly.")

    # Test validation step with tuple
    loss = model.validation_step((x, y), batch_idx=0)
    assert loss.shape == ()  # Scalar tensor
    print("Test 5 passed: Validation step with tuple works correctly.")

    # Test validation step with list
    loss = model.validation_step([x, y], batch_idx=0)
    assert loss.shape == ()  # Scalar tensor
    print("Test 6 passed: Validation step with list works correctly.")

    print("All tests passed! GPTLightningModule is working correctly.")
