# training/super_gpt2/__init__.py
"""
Training implementation for SuperBlock GPT-2.

This module contains Lightning modules and training scripts for SuperBlock GPT-2.
"""

from .lightning_module import GPTLightningModule, train_with_lightning

__all__ = ["GPTLightningModule", "train_with_lightning"]
