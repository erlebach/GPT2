"""GPT-2 training script using LightningModule without Trainer.

This script trains a GPT-2 model using LightningModule for clean organization
but without the Lightning Trainer object, giving full control over the training loop.

   collector = get_metrics_collector()

## Key Advantages of Using LightningModule Without Trainer:

1. **Clean Organization**: The LightningModule provides a well-structured way to organize training logic, optimizer configuration, and model setup.

2. **Automatic Device Management**: LightningModule handles device placement automatically.

3. **Built-in Logging**: The `self.log()` method provides consistent logging that can be easily extended.

4. **Hook System**: We can still use Lightning's lifecycle hooks like `on_train_start()`, `on_train_epoch_end()`, etc.

5. **Hyperparameter Management**: LightningModule's `save_hyperparameters()` and `hparams` provide clean hyperparameter tracking.

6. **Consistent Interface**: The training and validation step methods provide a consistent interface that's easy to understand and modify.

7. **Performance Monitoring**: The `@measure_performance` decorator and other Lightning utilities are still available.

8. **Checkpoint Compatibility**: Checkpoints can still be saved and loaded in a format compatible with Lightning.

## Key Features Maintained:

- **Multi-GPU Support**: DDP integration
- **Checkpointing**: Full save/load functionality
- **Learning Rate Scheduling**: Cosine decay with warmup
- **Optimizer Configuration**: AdamW with weight decay groups
- **Validation**: Regular validation with best model saving
- **GPU Monitoring**: Memory tracking and usage verification
- **Metrics Collection**: Integration with existing metrics system
- **TensorBoard Logging**: Training and validation loss logging
- **Distributed Training**: Manual DDP setup and cleanup

This approach gives you the best of both worlds: the organizational benefits and conveniences of LightningModule without the abstraction layer of the Trainer object.
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from gpt2_standalone.gpu_monitor import GPUMonitor
from gpt2_standalone.gpu_parallelism_checker import GPUParallelismChecker
from gpt2_standalone.lightning_module import GPTLightningModule
from torch.nn.parallel import DistributedDataParallel as DDP
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


class LightningModuleTrainer:
    """Custom trainer that uses LightningModule without Lightning Trainer.

    This class provides a training loop that leverages the benefits of LightningModule
    (clean organization, automatic device management, built-in logging) while giving
    full control over the training process.

    Args:
        lightning_module: LightningModule instance to train.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        checkpoint_dir: Directory to save checkpoints.
        device: Device to train on.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        use_ddp: Whether to use DistributedDataParallel.
        max_steps: Maximum number of training steps.
    """

    def __init__(
        self,
        lightning_module: GPTLightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        checkpoint_dir: Path = Path("checkpoints/"),
        device: torch.device | None = None,
        rank: int = 0,
        world_size: int = 1,
        use_ddp: bool = False,
        max_steps: int = 500,
    ):
        if device is None:
            device = torch.device(
                f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
            )

        self.lightning_module = lightning_module
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.max_steps = max_steps

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Move module to device
        self.lightning_module = self.lightning_module.to(self.device)

        # Wrap with DDP if using distributed training
        if self.use_ddp and self.world_size > 1:
            self.lightning_module = DDP(self.lightning_module, device_ids=[self.rank])

        # Configure optimizer and scheduler
        self.optimizer_config = self.lightning_module.configure_optimizers()
        self.optimizer = self.optimizer_config["optimizer"]
        self.scheduler = self.optimizer_config["lr_scheduler"]["scheduler"]

        # Initialize training state
        self.step = 0
        self.best_val_loss = float("inf")

        # Initialize tensorboard writer (only on rank 0)
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir="runs/gpt2_training")
        else:
            self.writer = None

        # Initialize metrics collector
        self.metrics_collector = get_metrics_collector()

        # Call on_train_start hook
        self.lightning_module.on_train_start()

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Name of the checkpoint file.

        """
        # Get the actual model (unwrap DDP if needed)
        model_to_save = (
            self.lightning_module.module
            if isinstance(self.lightning_module, DDP)
            else self.lightning_module
        )

        checkpoint = {
            "step": self.step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": model_to_save.train_losses,
            "val_losses": model_to_save.val_losses,
            "hyperparameters": model_to_save.hparams,
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

        # Get the actual model (unwrap DDP if needed)
        model_to_load = (
            self.lightning_module.module
            if isinstance(self.lightning_module, DDP)
            else self.lightning_module
        )

        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        model_to_load.train_losses = checkpoint.get("train_losses", [])
        model_to_load.val_losses = checkpoint.get("val_losses", [])

        if self.rank == 0:
            print(f"✅ Checkpoint loaded: {checkpoint_path}")
            print(f"   Resuming from step: {self.step}")

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """Perform a single training step using LightningModule.

        Args:
            batch: Tuple of (input_tokens, target_tokens).

        Returns:
            Training loss for the step.

        """
        # Use LightningModule's training_step
        loss = self.lightning_module.training_step(batch, batch_idx=self.step)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.detach().cpu().item()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """Perform a single validation step using LightningModule.

        Args:
            batch: Tuple of (input_tokens, target_tokens).

        Returns:
            Validation loss for the step.

        """
        # Use LightningModule's validation_step
        loss = self.lightning_module.validation_step(batch, batch_idx=0)
        return loss.detach().cpu().item()

    def validate(self) -> float:
        """Run validation on the entire validation set.

        Returns:
            Average validation loss.

        """
        self.lightning_module.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                val_loss = self.validation_step(batch)
                val_losses.append(val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses)

        # Log to tensorboard (only on rank 0)
        if self.rank == 0 and self.writer:
            self.writer.add_scalar("Loss/Validation", avg_val_loss, self.step)

        self.lightning_module.train()
        return avg_val_loss

    def train(self, save_interval: int = 100, val_interval: int = 50) -> None:
        """Run training loop using LightningModule.

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

            # Training step using LightningModule
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
            if self.step % save_interval == 0 and self.rank == 0:
                self.save_checkpoint(f"checkpoint_step_{self.step}.pt")

            # GPU memory monitoring
            if self.rank == 0 and self.step % 50 == 0:
                print_gpu_allocation(self.rank)

        # Call on_train_epoch_end hook
        self.lightning_module.on_train_epoch_end()

        # Final checkpoint
        if self.rank == 0:
            self.save_checkpoint("final_model.pt")
            print("✅ Training completed!")


def setup_distributed() -> tuple[int, int, bool]:
    """Set up distributed training environment.

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
    """Run GPT-2 training using LightningModule without Trainer."""
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

    # Create LightningModule with configuration
    from gpt2_standalone.model import GPTConfig

    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=4,  # Number of super-layers
        n_head=4,  # Number of attention heads
        n_embd=1024,  # Embedding dimension
        n_blocks_per_super=2,  # Number of blocks per SuperBlock
    )

    # Create LightningModule
    lightning_module = GPTLightningModule(
        config=config,
        weight_decay=0.2,
        learning_rate=6e-2,
        warmup_steps=10,
        max_steps=100,
    )

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

    # Create custom trainer using LightningModule
    trainer = LightningModuleTrainer(
        lightning_module=lightning_module,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        checkpoint_dir=Path("checkpoints/"),
        device=device,
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
        max_steps=100,
    )

    # Start training
    if rank == 0:
        print(f"\n🚀 Starting training with LightningModule (no Trainer):")
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
