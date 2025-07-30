# src/models/minimal_nemo_wrapper.py
from training.super_gpt2.lightning_module import GPTLightningModule

from models.gpt2.model import GPTConfig
from models.super_gpt2.model import GPTConfig as SuperGPTConfig


class MinimalNeMoWrapper(GPTLightningModule):
    """Minimal wrapper that uses NeMo config but inherits from your working GPTLightningModule."""

    def __init__(self, cfg, trainer=None):
        # Convert NeMo config to your config format
        if cfg.model.super_block_config.get("heterogeneous", False):
            config = self._convert_to_super_gpt2_config(cfg)
        else:
            config = self._convert_to_gpt2_config(cfg)

        # Initialize the parent class with your config
        super().__init__(
            config=config,
            weight_decay=cfg.model.optim.weight_decay
            if hasattr(cfg.model, "optim")
            else 0.1,
            learning_rate=cfg.model.optim.lr if hasattr(cfg.model, "optim") else 6e-2,
            warmup_steps=cfg.model.scheduler.warmup_steps
            if hasattr(cfg.model, "scheduler")
            else 10,
            max_steps=cfg.trainer.max_steps
            if hasattr(cfg.trainer, "max_steps")
            else 500,
        )

    def _convert_to_gpt2_config(self, cfg):
        """Convert NeMo config to GPT2Config."""
        from models.gpt2.model import GPTConfig

        return GPTConfig(
            block_size=cfg.model.max_position_embeddings,
            vocab_size=cfg.model.vocab_size,
            n_layer=cfg.model.num_layers,
            n_head=cfg.model.num_attention_heads,
            n_embd=cfg.model.hidden_size,
            n_blocks_per_super=cfg.model.super_block_config.n_blocks_per_super,
            dropout=cfg.model.hidden_dropout,
        )

    def _convert_to_super_gpt2_config(self, cfg):
        """Convert NeMo config to SuperGPT2Config."""
        from models.super_gpt2.model import BlockConfig
        from models.super_gpt2.model import GPTConfig as SuperGPTConfig

        # Extract heterogeneous block configs if they exist
        block_configs = []
        if hasattr(cfg.model.super_block_config, "layer_configs"):
            for layer_config in cfg.model.super_block_config.layer_configs:
                layer_blocks = []
                for block_config in layer_config.blocks:
                    layer_blocks.append(
                        BlockConfig(
                            n_embd=block_config.hidden_size,
                            n_head=block_config.num_attention_heads,
                            dropout=block_config.dropout,
                            use_swiglu=block_config.use_swiglu,
                        )
                    )
                block_configs.append(layer_blocks)

        return SuperGPTConfig(
            block_size=cfg.model.max_position_embeddings,
            vocab_size=cfg.model.vocab_size,
            n_layer=cfg.model.num_layers,
            base_embd=cfg.model.hidden_size,
            block_configs=block_configs,
            use_swiglu=cfg.model.super_block_config.use_swiglu,
        )
