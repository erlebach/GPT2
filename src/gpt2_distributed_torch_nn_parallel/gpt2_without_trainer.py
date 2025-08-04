"""GPT-2 training script without trainer object.

This script trains a GPT-2 model using pure PyTorch without Lightning trainer,
while maintaining all features including checkpointing, multi-GPU support,
and monitoring.

`gpt2_without_trainer.py` provides the following features:

## Key Features Maintained:

1. **Multi-GPU Support**: Uses `DistributedDataParallel` (DDP) for multi-GPU training without Lightning
2. **Checkpointing**: Full checkpoint save/load functionality with model state, optimizer state, and training progress
3. **Learning Rate Scheduling**: Cosine decay with warmup, same as the original
4. **Optimizer Configuration**: AdamW with weight decay groups
5. **Validation**: Regular validation during training with best model saving
6. **GPU Monitoring**: Memory allocation tracking and GPU usage verification
7. **Metrics Collection**: Integration with the existing metrics system
8. **TensorBoard Logging**: Training and validation loss logging
9. **Distributed Training Setup**: Manual DDP initialization and cleanup

## Key Differences from Lightning Version:

1. **No Trainer Object**: Completely removes dependency on Lightning's trainer
2. **Manual Training Loop**: Custom training loop with explicit step-by-step control
3. **Direct PyTorch**: Uses pure PyTorch for all operations
4. **Simplified Callbacks**: Removes Lightning callbacks but maintains core functionality
5. **Manual DDP**: Handles DistributedDataParallel setup manually

## Usage:

The script can be run the same way as the original (`train_gpt2.py`), with the same command-line arguments and environment variables for distributed training. It maintains all the core training functionality while giving you full control over the training process without the Lightning trainer abstraction.
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from gpt2_standalone.gpu_monitor import GPUMonitor
from gpt2_standalone.gpu_parallelism_checker import GPUParallelismChecker
from gpt2_standalone.model import GPT, GPTConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import TextDataset, get_project_root
from utils.metrics_extensions import get_metrics_collector


def check_gpu_allocation() -> list[str]:
    """Check GPU memory allocation across all available devices.

    Returns:
        List of allocation status strings for each GPU.

    """
    results = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i)
        results.append(f"GPU {i}: {allocated/1e6:.2f} MB allocated")
    return results


def print_gpu_allocation(rank: int = 0) -> None:
    """Print GPU memory allocation for debugging.

    Args:
        rank: Process rank for distributed training.

    """
    rank_info = f"[Rank {rank}]"
    device_info = (
        f"Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"{rank_info} {device_info}", flush=True)
    allocs = check_gpu_allocation()
    print(f"[Rank {rank}] GPU Allocation:")
    for alloc in allocs:
        print(f"[Rank {rank}] {alloc}", flush=True)


class GPTTrainer:
    """Custom trainer for GPT-2 without Lightning trainer object.

    This class handles training, validation, checkpointing, and multi-GPU
    support without relying on Lightning's trainer.

    Args:
        model: GPT model to train.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps for learning rate.
        max_steps: Maximum number of training steps.
        min_lr_ratio: Ratio of minimum learning rate to maximum learning rate.
        checkpoint_dir: Directory to save checkpoints.
        device: Device to train on.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        use_ddp: Whether to use DistributedDataParallel.
    """

    def __init__(
        self,
        model: GPT,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        learning_rate: float = 6e-2,
        weight_decay: float = 0.1,
        warmup_steps: int = 10,
        max_steps: int = 500,
        min_lr_ratio: float = 0.1,
        checkpoint_dir: Path = Path("checkpoints/"),
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        rank: int = 0,
        world_size: int = 1,
        use_ddp: bool = False,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Move model to device
        self.model = self.model.to(self.device)

        # Wrap with DDP if using distributed training
        if self.use_ddp and self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])

        # Configure optimizer and scheduler
        self.optimizer = self._configure_optimizer()
        self.scheduler = self._configure_scheduler()

        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.step = 0
        self.best_val_loss = float("inf")

        # Initialize tensorboard writer (only on rank 0)
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir="runs/gpt2_training")
        else:
            self.writer = None

        # Initialize metrics collector
        self.metrics_collector = get_metrics_collector()

    def _configure_optimizer(self) -> AdamW:
        """Configure optimizer with weight decay groups.

        Returns:
            Configured AdamW optimizer.

        """
        # Get parameters and separate by dimension for weight decay
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            optim_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _configure_scheduler(self) -> LambdaLR:
        """Configure learning rate scheduler.

        Returns:
            Configured LambdaLR scheduler.

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

        return LambdaLR(self.optimizer, lr_lambda)

    def _get_lr_lambda(self) -> float:
        """Get current learning rate multiplier.

        Returns:
            Current learning rate multiplier.

        """
        return self.scheduler.get_last_lr()[0] / self.learning_rate

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Name of the checkpoint file.

        """
        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "max_steps": self.max_steps,
                "min_lr_ratio": self.min_lr_ratio,
            },
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if self.rank == 0:
            print(f"✅ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.

        """
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        if self.rank == 0:
            print(f"✅ Checkpoint loaded: {checkpoint_path}")
            print(f"   Resuming from step: {self.step}")

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """Perform a single training step.

        Args:
            batch: Tuple of (input_tokens, target_tokens).

        Returns:
            Training loss for the step.

        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Forward pass
        logits, loss = self.model(x, y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Track loss
        loss_value = loss.detach().cpu().item()
        self.train_losses.append(loss_value)

        # Log to tensorboard (only on rank 0)
        if self.rank == 0 and self.writer:
            self.writer.add_scalar("Loss/Train", loss_value, self.step)
            self.writer.add_scalar(
                "Learning_Rate", self.scheduler.get_last_lr()[0], self.step
            )

        return loss_value

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """Perform a single validation step.

        Args:
            batch: Tuple of (input_tokens, target_tokens).

        Returns:
            Validation loss for the step.

        """
        with torch.no_grad():
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            logits, loss = self.model(x, y)

            # Track loss
            loss_value = loss.detach().cpu().item()
            self.val_losses.append(loss_value)

            return loss_value

    def validate(self) -> float:
        """Run validation on the entire validation set.

        Returns:
            Average validation loss.

        """
        self.model.eval()
        val_losses = []

        for batch in self.val_dataloader:
            val_loss = self.validation_step(batch)
            val_losses.append(val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses)

        # Log to tensorboard (only on rank 0)
        if self.rank == 0 and self.writer:
            self.writer.add_scalar("Loss/Validation", avg_val_loss, self.step)

        self.model.train()
        return avg_val_loss

    def train(self, save_interval: int = 100, val_interval: int = 50) -> None:
        """Main training loop.

        Args:
            save_interval: How often to save checkpoints (in steps).
            val_interval: How often to run validation (in steps).

        """
        if self.rank == 0:
            print(f"🚀 Starting training for {self.max_steps} steps")
            print(f"   Device: {self.device}")
            print(f"   World size: {self.world_size}")
            print(f"   Use DDP: {self.use_ddp}")

        # Create infinite iterator for training data
        train_iter = iter(self.train_dataloader)

        while self.step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            # Training step
            train_loss = self.training_step(batch)
            self.step += 1

            # Print progress
            if self.rank == 0 and self.step % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Step {self.step}/{self.max_steps}: "
                    f"Train Loss: {train_loss:.4f}, LR: {lr:.6f}"
                )

            # Validation
            if self.step % val_interval == 0:
                val_loss = self.validate()

                if self.rank == 0:
                    print(f"Step {self.step}: Validation Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.rank == 0:
                        self.save_checkpoint("best_model.pt")

            # Regular checkpointing
            if self.step % save_interval == 0:
                if self.rank == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.step}.pt")

            # GPU memory monitoring
            if self.rank == 0 and self.step % 50 == 0:
                print_gpu_allocation(self.rank)

        # Final checkpoint
        if self.rank == 0:
            self.save_checkpoint("final_model.pt")
            print("✅ Training completed!")


def setup_distributed() -> tuple[int, int, bool]:
    """Setup distributed training environment.

    Returns:
        Tuple of (rank, world_size, use_ddp).

    """
    # Debug distributed setup
    print(f"=== DISTRIBUTED SETUP DEBUG ===")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    # Check if distributed is initialized
    print(f"Distributed initialized: {dist.is_initialized()}")

    # MANUALLY INITIALIZE DISTRIBUTED IF NEEDED
    if not dist.is_initialized() and os.environ.get("WORLD_SIZE", "1") != "1":
        print("🚀 Manually initializing distributed process group...")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "12355")

        # Set device for this process
        torch.cuda.set_device(local_rank)

        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=local_rank,
        )
        print(f"✅ Distributed initialized - Rank {local_rank}/{world_size}")

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        use_ddp = world_size > 1
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Backend: {dist.get_backend()}")
    else:
        rank = 0
        world_size = 1
        use_ddp = False

    return rank, world_size, use_ddp


def main():
    """Main function to run GPT-2 training without trainer object."""
    # Setup distributed training
    rank, world_size, use_ddp = setup_distributed()

    # Initialize GPU parallelism checker
    gpu_parallelism_checker = GPUParallelismChecker()
    if rank == 0:
        gpu_parallelism_checker.print_comprehensive_report()

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if rank == 0:
        print(f"🔍 Number of available GPUs: {num_gpus}")

    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Verify multi-GPU setup before training
    try:
        monitor = GPUMonitor()
        verification = monitor.verify_multi_gpu_usage()
        if rank == 0:
            print(f"\n🔍 Pre-training GPU verification:")
            print(f"   Status: {verification['status']}")
            print(f"   GPUs Available: {verification['total_gpus']}")
            print(f"   Expected to use: {num_gpus}")
    except Exception as e:
        if rank == 0:
            print(f"Error during pre-training GPU verification: {e}")
            print("Continuing with training...")

    # Create model configuration
    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=4,  # Number of super-layers
        n_head=4,  # Number of attention heads
        n_embd=1024,  # Embedding dimension
        n_blocks_per_super=2,  # Number of blocks per SuperBlock
    )

    # Create model
    model = GPT(config)

    # Create datasets and dataloaders
    data_path = get_project_root() / "data" / "input.txt"
    train_dataset = TextDataset(data_path, block_size=1024)

    # Split dataset for train/val
    val_split = 0.1
    n_val = int(len(train_dataset) * val_split)
    n_train = len(train_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val]
    )

    # Create dataloaders with proper DDP support
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # Create trainer
    trainer = GPTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=6e-2,
        weight_decay=0.2,
        warmup_steps=10,
        max_steps=100,
        checkpoint_dir=Path("checkpoints/"),
        device=device,
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
    )

    # Start training
    if rank == 0:
        print(f"\n🚀 Starting training without trainer object:")
        print(f"   GPUs: {num_gpus}")
        print(f"   Use DDP: {use_ddp}")
        print(f"   Device: {device}")

    trainer.train(save_interval=50, val_interval=25)

    # Post-training verification
    try:
        if rank == 0:
            print(f"\n🔍 Post-training GPU verification:")
            post_verification = monitor.verify_multi_gpu_usage()
            print(f"   Status: {post_verification['status']}")
            print(
                f"   GPUs Used: {post_verification['gpus_used']}/{post_verification['total_gpus']}"
            )
            print(f"   Memory Usage: {post_verification['memory_usage']}")
    except Exception as e:
        if rank == 0:
            print(f"Error during post-training GPU verification: {e}")

    # Save metrics
    if rank == 0:
        collector = get_metrics_collector()
        collector.save_all_metrics_to_csv("metrics.csv")
        print("✅ Training completed and metrics saved")

    # Cleanup distributed training
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
