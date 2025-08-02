# lightning_mwe_2gpu_slurm.py
"""
Minimum Working Example: PyTorch Lightning with Two GPUs
- Demonstrates the simplest parallelism (DDP)
- Includes checks for GPU allocation
- Intended for SLURM
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import os

def check_gpu_allocation():
    results = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i)
        results.append(f"GPU {i}: {allocated/1e6:.2f} MB allocated")
    return results

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
        if batch_idx == 0 and self.global_rank == 0:
            allocs = check_gpu_allocation()
            print("[GPU Allocation]", *allocs, sep='\n')
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

if __name__ == "__main__":
    # Simulate small random dataset
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = SimpleModel()

    # Lightning will detect SLURM environment variables
    # To force 2 GPUs, set devices=2, accelerator="gpu", strategy="ddp"
    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",  # DDP is simplest, standard parallelism
        max_epochs=1,
        logger=False,  # suppress logging
        enable_checkpointing=False
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

