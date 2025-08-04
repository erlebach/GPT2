"""GPT-2 training script using Lightning Fabric.

This script trains a GPT-2 model using Lightning Fabric, which provides
the organizational benefits of Lightning without the Trainer abstraction.
Fabric handles device management, DDP setup, and other boilerplate automatically.
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
import torch
import torch.distributed as dist
from gpt2_standalone.gpu_monitor import GPUMonitor
from gpt2_standalone.gpu_parallelism_checker import GPUParallelismChecker
from gpt2_standalone.lightning_module import GPTLightningModule
from lightning.fabric import Fabric

# from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils.data_utils import TextDataset, get_project_root
from utils.metrics_extensions import get_metrics_collector, measure_performance


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
    # Get the actual current device
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"

    rank_info = f"[Rank {rank}]"
    device_info = f"Device: {current_device}"
    print(f"{rank_info} {device_info}", flush=True)
    allocs = check_gpu_allocation()
    print(f"[Rank {rank}] GPU Allocation:")
    for alloc in allocs:
        print(f"[Rank {rank}] {alloc}", flush=True)


class FabricTrainer:
    """Custom trainer using Lightning Fabric for simplified training.

    This class leverages Lightning Fabric to handle device management,
    DDP setup, and other boilerplate while maintaining full control
    over the training loop.

    Args:
        lightning_module: LightningModule instance to train.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        fabric: Lightning Fabric instance for device management.
        checkpoint_dir: Directory to save checkpoints.
        max_steps: Maximum number of training steps.
    """

    def __init__(
        self,
        lightning_module: GPTLightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        fabric: Fabric,
        checkpoint_dir: Path = Path("checkpoints/"),
        max_steps: int = 500,
    ):
        self.lightning_module = lightning_module
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.fabric = fabric
        self.checkpoint_dir = checkpoint_dir
        self.max_steps = max_steps

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Setup model, optimizer, and dataloaders with Fabric
        self.lightning_module, self.optimizer = self.fabric.setup(
            self.lightning_module,
            self.lightning_module.configure_optimizers()["optimizer"],
        )
        self.train_dataloader, self.val_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader, self.val_dataloader
        )

        # Get scheduler from LightningModule
        scheduler_config = self.lightning_module.configure_optimizers()["lr_scheduler"]
        self.scheduler = scheduler_config["scheduler"]

        # Initialize training state
        self.step = 0
        self.best_val_loss = float("inf")

        # Initialize metrics collector
        self.metrics_collector = get_metrics_collector()

        # Call on_train_start hook
        self.lightning_module.on_train_start()

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint using Fabric.

        Args:
            filename: Name of the checkpoint file.

        """
        checkpoint = {
            "step": self.step,
            "model": self.lightning_module,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.lightning_module.train_losses,
            "val_losses": self.lightning_module.val_losses,
            "hyperparameters": self.lightning_module.hparams,
        }

        checkpoint_path = self.checkpoint_dir / filename
        self.fabric.save(checkpoint_path, checkpoint)

        if self.fabric.is_global_zero:
            print(f"✅ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint using Fabric.

        Args:
            checkpoint_path: Path to the checkpoint file.

        """
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = self.fabric.load(checkpoint_path)

        self.lightning_module = checkpoint["model"]
        self.optimizer = checkpoint["optimizer"]
        self.scheduler = checkpoint["scheduler"]

        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.lightning_module.train_losses = checkpoint.get("train_losses", [])
        self.lightning_module.val_losses = checkpoint.get("val_losses", [])

        if self.fabric.is_global_zero:
            print(f"✅ Checkpoint loaded: {checkpoint_path}")
            print(f"   Resuming from step: {self.step}")

    @measure_performance(memory_enabled=True, timing_enabled=True)
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """Perform a single training step using LightningModule.

        Args:
            batch: Tuple of (input_tokens, target_tokens).

        Returns:
            Training loss for the step.

        """
        # Use LightningModule's training_step
        loss = self.lightning_module.training_step(batch, batch_idx=self.step)

        # Backward pass with Fabric
        self.fabric.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.scheduler.step()  # Disabled scheduler

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
        self.lightning_module.train()
        return avg_val_loss

    def train(self, save_interval: int = 100, val_interval: int = 50) -> None:
        """Main training loop using Fabric.

        Args:
            save_interval: How often to save checkpoints (in steps).
            val_interval: How often to run validation (in steps).

        """
        if self.fabric.global_rank == 0:  # Changed from self.fabric.is_global_zero
            print(f"🚀 Starting training for {self.max_steps} steps")
            print(f"   Device: {self.fabric.device}")
            print(f"   World size: {self.fabric.world_size}")
            print(f"   Precision: {self.fabric._precision}")

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
            if self.fabric.global_rank == 0 and self.step % 10 == 0:  # Changed from self.fabric.is_global_zero
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Step {self.step}/{self.max_steps}: "
                    f"Train Loss: {train_loss:.4f}, LR: {lr:.6f}"
                )

            # Validation
            if self.step % val_interval == 0:
                val_loss = self.validate()

                if self.fabric.global_rank == 0:  # Changed from self.fabric.is_global_zero
                    print(f"Step {self.step}: Validation Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.fabric.global_rank == 0:  # Changed from self.fabric.is_global_zero
                        self.save_checkpoint("best_model.pt")

            # Regular checkpointing
            if self.step % save_interval == 0 and self.fabric.global_rank == 0:  # Changed from self.fabric.is_global_zero
                self.save_checkpoint(f"checkpoint_step_{self.step}.pt")

            # GPU memory monitoring (only for first 2 steps)
            if self.fabric.global_rank == 0 and self.step < 2:  # Changed from self.fabric.is_global_zero
                print_gpu_allocation(self.fabric.global_rank)

        # Call on_train_epoch_end hook
        self.lightning_module.on_train_epoch_end()

        # Final checkpoint
        if self.fabric.global_rank == 0:  # Changed from self.fabric.is_global_zero
            self.save_checkpoint("final_model.pt")
            print("✅ Training completed!")


def setup_fabric(
    accelerator: str = "auto",
    devices: str | int = "auto",
    precision: str = "32-true",
    strategy: str = "auto",
) -> Fabric:
    """Setup Lightning Fabric for training.

    Args:
        accelerator: Type of accelerator ('auto', 'cpu', 'gpu').
        devices: Number of devices to use.
        precision: Training precision.
        strategy: Distributed strategy.

    Returns:
        Configured Fabric instance.

    """
    # Create logger
    # logger = TensorBoardLogger("runs", name="gpt2_fabric_training")

    # Create Fabric
    fabric = Fabric(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        strategy=strategy,
        # loggers=logger,
    )

    # Setup Fabric
    fabric.launch()

    return fabric


def main():
    """Main function to run GPT-2 training using Lightning Fabric."""
    # Initialize GPU parallelism checker
    gpu_parallelism_checker = GPUParallelismChecker()
    gpu_parallelism_checker.print_comprehensive_report()

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Number of available GPUs: {num_gpus}")

    # Setup Fabric
    fabric = setup_fabric(
        accelerator="gpu",
        devices=num_gpus,
        precision="32",
        strategy="ddp" if num_gpus > 1 else "auto",
    )

    # Verify multi-GPU setup before training
    try:
        monitor = GPUMonitor()
        verification = monitor.verify_multi_gpu_usage()
        if fabric.is_global_zero:
            print(f"\n🔍 Pre-training GPU verification:")
            print(f"   Status: {verification['status']}")
            print(f"   GPUs Available: {verification['total_gpus']}")
            print(f"   Expected to use: {num_gpus}")
    except Exception as e:
        if fabric.is_global_zero:
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

    max_steps = 10

    # Create LightningModule
    lightning_module = GPTLightningModule(
        config=config,
        weight_decay=0.2,
        learning_rate=6e-2,
        warmup_steps=10,
        max_steps=max_steps,  # Match the FabricTrainer's max_steps
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

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # Create trainer using Fabric
    trainer = FabricTrainer(
        lightning_module=lightning_module,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        fabric=fabric,
        checkpoint_dir=Path("checkpoints/"),
        max_steps=max_steps,
    )

    print(f"--------------------------------")
    print(f"🔍 Fabric device: {fabric.is_global_zero}")
    print(f"🔍 Fabric global_rank: {fabric.global_rank}")
    print(f"🔍 Fabric world_size: {fabric.world_size}")

    # Start training
    if fabric.global_rank == 0:  # Changed from fabric.is_global_zero
        print(f"\n🚀 Starting training with Lightning Fabric:")
        print(f"   GPUs: {num_gpus}")
        print(f"   Device: {fabric.device}")
        precision = getattr(
            fabric,device "_precision", getattr(fabric, "precision", "unknown")
        )
        print(f"   Precision: {precision}")

    trainer.train(save_interval=50, val_interval=25)
    print(f"exit trainer.train, {fabric.global_rank=}")

    # Add debug print to see if we reach here
    if fabric.global_rank == 0:  # Changed from fabric.is_global_zero
        print("🔍 Reached post-training section")

    # Post-training verification
    try:
        if fabric.global_rank == 0:  # Changed from fabric.is_global_zero
            print(f"\n🔍 Post-training GPU verification:")
            post_verification = monitor.verify_multi_gpu_usage()
            print(f"   Status: {post_verification['status']}")
            print(
                f"   GPUs Used: {post_verification['gpus_used']}/{post_verification['total_gpus']}"
            )
            print(f"   Memory Usage: {post_verification['memory_usage']}")
    except Exception as e:
        if fabric.global_rank == 0:  # Changed from fabric.is_global_zero
            print(f"Error during post-training GPU verification: {e}")

    # Save metrics
    if fabric.global_rank == 0:  # Changed from fabric.is_global_zero
        collector = get_metrics_collector()
        collector.save_all_metrics_to_csv("metrics.csv")
        print("✅ Training completed and metrics saved")


if __name__ == "__main__":
    main()
    print("✅ Training completed; last statement.")
