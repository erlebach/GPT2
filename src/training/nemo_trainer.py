# src/training/nemo_trainer.py
import lightning as pl
import torch
from models.minimal_nemo_wrapper import MinimalNeMoWrapper
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="config/nemo", config_name="gpt2_super")
def main(cfg):
    """NeMo training using your existing Lightning module."""
    print("🚀 Starting NeMo training...", flush=True)
    
    # Create trainer
    trainer = pl.Trainer(**cfg.trainer)
    
    # Create model using your working Lightning module
    model = MinimalNeMoWrapper(cfg)
    
    # Train using your reliable infrastructure
    trainer.fit(model)

if __name__ == "__main__":
    main()
