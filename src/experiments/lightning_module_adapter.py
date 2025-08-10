"""Adapter that lets us pass config fields directly via kwargs."""

from __future__ import annotations

from typing import Any

import gpt2_standalone.lightning_module as lm
from gpt2_standalone.model import GPTConfig


class GPTLMAdapter(lm.GPTLightningModule):
    """Adapter that lets us pass config fields directly via kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        block_size = kwargs.pop("block_size")
        vocab_size = kwargs.pop("vocab_size")
        n_layer = kwargs.pop("n_layer", 2)
        n_head = kwargs.pop("n_head", 4)
        n_embd = kwargs.pop("n_embd", 128)
        n_blocks_per_super = kwargs.pop("n_blocks_per_super", 2)

        cfg = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_blocks_per_super=n_blocks_per_super,
        )
        super().__init__(cfg, **kwargs)
