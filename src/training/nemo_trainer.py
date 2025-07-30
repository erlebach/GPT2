# src/training/nemo_trainer.py
from nemo.core.config import hydra_runner
from nemo.utils import logging

from models.nemo_wrappers import NeMoGPT2Wrapper, NeMoSuperGPT2Wrapper

@hydra_runner(config_path="config/nemo", config_name="gpt2_super")
def main(cfg):
    """NeMo training using your existing models."""
    
    # Choose wrapper based on config
    if cfg.model.super_block_config.get('heterogeneous', False):
        model_class = NeMoSuperGPT2Wrapper
    else:
        model_class = NeMoGPT2Wrapper
    
    # Initialize trainer
    trainer = pl.Trainer(**cfg.trainer)
    
    # Create model (this will use your existing models internally)
    model = model_class(cfg.model, trainer)
    
    # Train using NeMo's infrastructure but your models
    trainer.fit(model)
