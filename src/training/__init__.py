# training/__init__.py
"""
Training modules for GPT-2 variants.

This package contains training implementations for different models:
- gpt2: Training for standard GPT-2
- super_gpt2: Training for SuperBlock GPT-2
- common: Shared training utilities
"""

from . import gpt2
from . import super_gpt2
from . import common

__all__ = ["gpt2", "super_gpt2", "common"]
