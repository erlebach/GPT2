# lightning_mwe_2gpu_slurm.py
"""
Minimum Working Example: PyTorch Lightning with Two GPUs
- Demonstrates the simplest parallelism (DDP)
- Includes checks for GPU allocation
- Intended for SLURM
"""

import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def check_gpu_allocation():
    """Check GPU memory allocation across all available devices.

    Returns:
        list[str]: List of allocation status strings for each GPU.

    """
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_id)
        return f"GPU {device_id}: {allocated/1e6:.2f} MB allocated"


return "No CUDA available"


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

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)

        # Check and print GPU allocation (first step only)
        if batch_idx == 0:
            rank_info = f"[Rank {self.global_rank}/{self.trainer.world_size}]"
            device_info = f"Device: {self.device}"
            print(f"{rank_info} {device_info}", flush=True)

            # Only print allocation from rank 0 to avoid blocking
            if self.global_rank == 0:
                allocs = check_gpu_allocation()
                print("[GPU Allocation]", flush=True)
                for alloc in allocs:
                    print(alloc)

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

    # Lightning will detect SLURM environment variables
    # To force 2 GPUs, set devices=2, accelerator="gpu", strategy="ddp"
    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",  # DDP is simplest, standard parallelism
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
