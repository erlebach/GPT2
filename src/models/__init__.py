# models/__init__.py
"""
Model implementations for GPT-2 variants.

This package contains different model architectures:
- gpt2: Standard GPT-2 implementation
- super_gpt2: GPT-2 with SuperBlock architecture
- experimental: Experimental model variants
"""

from . import gpt2
from . import super_gpt2
from . import experimental

__all__ = ["gpt2", "super_gpt2", "experimental"]
