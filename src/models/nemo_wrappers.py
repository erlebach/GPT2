from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
)
from nemo.core.classes import ExportableArtifactMixin
from nemo.core.classes.common import typecheck
from nemo.utils import logging

from models.gpt2.model import GPT as GPT2Model
from models.gpt2.model import GPTConfig as GPT2Config
from models.super_gpt2.model import GPT as SuperGPT2Model
from models.super_gpt2.model import GPTConfig as SuperGPT2Config


class NeMoGPT2Wrapper(MegatronGPTModel):
    """NeMo wrapper for your existing GPT-2 with SuperBlocks."""

    def __init__(self, cfg, trainer=None):
        # Initialize NeMo base class
        super().__init__(cfg, trainer)

        # Create your existing model
        config = self._convert_nemo_config_to_gpt2_config(cfg)
        self.gpt2_model = GPT2Model(config)

        # Apply timing decorators in-place (no reassignment)
        self._apply_timing_decorators()

    def _convert_nemo_config_to_gpt2_config(self, nemo_cfg) -> GPT2Config:
        """Convert NeMo config to your GPT2Config."""
        return GPT2Config(
            block_size=nemo_cfg.max_position_embeddings,
            vocab_size=nemo_cfg.vocab_size,
            n_layer=nemo_cfg.num_layers,
            n_head=nemo_cfg.num_attention_heads,
            n_embd=nemo_cfg.hidden_size,
            n_blocks_per_super=nemo_cfg.super_block_config.n_blocks_per_super,
            dropout=nemo_cfg.hidden_dropout,
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """NeMo-compatible forward pass that uses your model."""
        # Convert NeMo format to your format
        logits, loss = self.gpt2_model(input_ids, labels)

        # Return in NeMo format
        return {
            "logits": logits,
            "loss": loss,
        }

    def _apply_timing_decorators(self):
        """Safely apply timing decorators, avoiding conflicts."""
        from utils.metrics_extensions import measure_performance

        # Check if already decorated
        if self._is_already_decorated(self.gpt2_model.forward):
            print(
                "ℹ️  Forward method already decorated - preserving existing decorators"
            )
            return

        # Apply decorators safely
        self._decorate_forward_safely()

    def _is_already_decorated(self, method):
        """Check if a method already has our timing decorators."""
        return (
            hasattr(
                method, "last_metrics"
            )  # ← Added by our @measure_performance decorator
            or hasattr(
                method, "__wrapped__"
            )  # ← Added by any decorator (including ours)
            or method.__name__ != "forward"  # ← Method name changed by decorator
        )

    def _decorate_forward_safely(self):
        """Apply decorators only if not already decorated."""
        from utils.metrics_extensions import measure_performance

        # Get the original method (strip any existing decorators)
        original_method = self._strip_decorators(self.gpt2_model.forward)

        @measure_performance(memory_enabled=True, timing_enabled=True)
        def timed_forward(*args, **kwargs):
            return original_method(*args, **kwargs)

        # Replace the method
        self.gpt2_model.forward = timed_forward

    def _strip_decorators(self, method):
        """Strip decorators to get the original method."""
        # Follow the decorator chain to get the original
        while hasattr(method, "__wrapped__"):
            method = method.__wrapped__

        return method


class NeMoSuperGPT2Wrapper(MegatronGPTModel):
    """NeMo wrapper for your heterogeneous SuperBlock GPT-2."""

    def __init__(self, cfg, trainer=None):
        super().__init__(cfg, trainer)

        # Create your existing heterogeneous model
        config = self._convert_nemo_config_to_super_gpt2_config(cfg)
        self.super_gpt2_model = SuperGPT2Model(config)

        # Preserve your timing decorators
        self.super_gpt2_model = self._add_timing_decorators(self.super_gpt2_model)

    def _convert_nemo_config_to_super_gpt2_config(self, nemo_cfg) -> SuperGPT2Config:
        """Convert NeMo config to your SuperGPT2Config."""
        # Extract heterogeneous block configs
        block_configs = []
        for layer_config in nemo_cfg.super_block_config.layer_configs:
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

        return SuperGPT2Config(
            block_size=nemo_cfg.max_position_embeddings,
            vocab_size=nemo_cfg.vocab_size,
            n_layer=nemo_cfg.num_layers,
            base_embd=nemo_cfg.hidden_size,
            block_configs=block_configs,
            use_swiglu=nemo_cfg.super_block_config.use_swiglu,
        )
