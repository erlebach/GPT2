# lightning_mwe_2gpu_slurm.py
"""
Minimum Working Example: PyTorch Lightning with Two GPUs
- Demonstrates the simplest parallelism (DDP)
- Includes checks for GPU allocation
- Intended for SLURM
"""

import os

import lightning.pytorch as pl
import torch
from lightning import LightningModule, Trainer

# Use DDPStrategy object instead of string
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def print_environment_info():
    """Print relevant environment and device information for debugging."""
    print(f"[Environment Info]")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(
        f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'None'}"
    )
    print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID', 'Not set')}")
    print(f"SLURM_LOCALID: {os.environ.get('SLURM_LOCALID', 'Not set')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")


class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
        self.current_batch_shape = None
        self.current_first_element = None
        self.current_target_element = None

    def forward(self, x):
        return self.layer(x)

    def print_gpu_allocation(self):
        rank_info = f"[Rank {self.global_rank}/{self.trainer.world_size}]"
        device_info = f"Device: {self.device}"
        print(f"{rank_info} {device_info}", flush=True)

        # Print strategy info
        print(f"[Rank {self.global_rank}] Strategy: {self.trainer.strategy}")
        print(f"[Rank {self.global_rank}] Strategy type: {type(self.trainer.strategy)}")

        # Print GPU allocation (original functionality)
        allocs = check_gpu_allocation()
        print(f"[Rank {self.global_rank}] GPU Allocation:")
        for alloc in allocs:
            print(f"[Rank {self.global_rank}] {alloc}", flush=True)

        # Print data info (new functionality)
        print(f"[Rank {self.global_rank}] Batch shape: {self.current_batch_shape}")
        print(
            f"[Rank {self.global_rank}] First element: {self.current_first_element:.6f}"
        )
        print(
            f"[Rank {self.global_rank}] Target first element: {self.current_target_element:.6f}"
        )

    def check_weight_synchronization(self):
        """Check if weight matrices are synchronized across GPUs."""
        print(f"[Rank {self.global_rank}] Weight Synchronization Check:")
        for name, param in self.named_parameters():
            if "weight" in name:  # Check weight matrices
                weight_norm = param.data.norm().item()
                print(f"[Rank {self.global_rank}] {name} norm: {weight_norm:.6f}")
                break  # Just check the first weight matrix

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)

        # Store data for printing (first step only)
        if batch_idx == 0:
            self.current_batch_shape = x.shape
            self.current_first_element = x[0, 0].item()
            self.current_target_element = y[0, 0].item()

            # Check weight synchronization
            self.check_weight_synchronization()

            # Print from both ranks to compare
            self.print_gpu_allocation()

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


if __name__ == "__main__":
    # Print environment info for debugging
    if os.environ.get("SLURM_PROCID", "0") == "0":
        print_environment_info()
        print()

    # Simulate small random dataset
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    ds = TensorDataset(X, y)

    # Remove shuffle and reduce num_workers for faster execution
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = SimpleModel()

    strategy = DDPStrategy(
        find_unused_parameters=False,
        static_graph=True,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        strategy=strategy,  # Use strategy object instead of "ddp"
        max_epochs=1,
        logger=False,  # suppress logging
        enable_checkpointing=False,
        # Add these for faster execution
        sync_batchnorm=False,
        precision="32",  # Use 32-bit precision for speed
    )

    trainer.fit(model, dl)

    # After training, check GPU allocation again
    if torch.cuda.is_available() and (os.environ.get("SLURM_PROCID", "0") == "0"):
        print("[Post-training GPU Allocation]")
        for alloc in check_gpu_allocation():
            print(alloc)

    # To run with SLURM, request 2 GPUs per node. Example SLURM sbatch:
    # #SBATCH --nodes=1
    # #SBATCH --ntasks=1
    # #SBATCH --gres=gpu:2
    # #SBATCH --cpus-per-task=4
    # module load cuda/12.x
    # pip install torch pytorch-lightning
    # srun python lightning_mwe_2gpu_slurm.py
