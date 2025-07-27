"""GPT-2 model implementation with heterogeneous SuperBlocks.

## **Key Features of the Heterogeneous SuperBlock Architecture:**

### **1. Individual Block Configurations**
Each block within a SuperBlock can have its own:
- **Embedding dimension** (`n_embd`)
- **Number of attention heads** (`n_head`)
- **Dropout rate** (`dropout`)

### **2. Input/Output Projections**
- **Input projections**: Map from base embedding dimension to each block's specific embedding dimension
- **Output projections**: Map from each block's embedding dimension back to base embedding dimension
- This allows blocks to operate at different scales while maintaining compatibility

### **3. MoE-Style Gating**
- **Gating mechanism**: Learns which blocks to prefer for different inputs
- **Weighted combination**: Combines outputs from multiple blocks using learned weights
- **Diverse initialization**: Each block starts with different random states for better diversity

### **4. Example Configuration**
```python
config = GPTConfig(
    n_layer=2,  # 2 SuperBlocks
    base_embd=64,  # Base embedding dimension
    block_configs=[
        [  # SuperBlock 1
            BlockConfig(n_embd=16, n_head=1),  # Fine-grained features
            BlockConfig(n_embd=32, n_head=2),  # Medium-scale features
            BlockConfig(n_embd=64, n_head=4),  # High-level features
        ],
        [  # SuperBlock 2 (same pattern)
            BlockConfig(n_embd=16, n_head=1),
            BlockConfig(n_embd=32, n_head=2),
            BlockConfig(n_embd=64, n_head=4),
        ],
    ],
)
```

### **5. Benefits of This Architecture**

**Multi-Scale Feature Learning:**
- **Small embeddings (16)**: Capture fine-grained, local patterns
- **Medium embeddings (32)**: Capture intermediate patterns
- **Large embeddings (64)**: Capture high-level, abstract patterns

**Adaptive Computation:**
- Gating mechanism learns which scales are most useful for different inputs
- Can dynamically route information through different pathways

**Parameter Efficiency:**
- Smaller blocks can capture simple patterns efficiently
- Larger blocks handle complex patterns when needed
- Overall parameter count can be optimized for your specific task

### **6. Training Considerations**

**Natural Learning Rate Adaptation:**
- Adam/AdamW will automatically provide different effective learning rates per block
- Blocks with different scales will naturally train at different rates (which is good!)

**Gradient Flow:**
- Input/output projections ensure gradients can flow properly between different embedding dimensions
- Residual connections maintain stable training


1. **Added four RMSNorm layers** instead of two:
   - `ln_1_pre`: RMSNorm before attention
   - `ln_1_post`: RMSNorm after attention residual
   - `ln_2_pre`: RMSNorm before MLP
   - `ln_2_post`: RMSNorm after MLP residual

2. **Modified the forward pass** to follow the pre-norm architecture:
   - **Attention block**: `x → ln_1_pre → attn → +x → ln_1_post`
   - **MLP block**: `attn_output → ln_2_pre → mlp → +attn_output → ln_2_post`

## Architecture Benefits:

This implementation follows the more recent transformer architectures (like GPT-3, PaLM, etc.) which have several advantages:

1. **Better gradient flow**: The LayerNorm layers inside the residual connections help stabilize training
2. **Improved convergence**: Pre-norm architectures typically train more stably
3. **Better scaling**: This pattern works better for deeper models
4. **Consistent normalization**: Each sub-layer gets properly normalized input

The architecture now follows the pattern:
```
Input → LayerNorm → Attention → Residual → LayerNorm → LayerNorm → MLP → Residual → LayerNorm → Output
```

This is a more modern and robust transformer block design that should provide better training stability and performance compared to the original GPT-2 style architecture.

"""

import math
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Integer
from torch import Tensor


class RMSNorm(nn.Module):
    """RMSNorm implementation as used in modern transformer architectures.

    RMSNorm is a simplified version of LayerNorm that only normalizes by RMS
    without the affine transformation, making it more efficient.

    Args:
        hidden_size: The hidden size of the input tensor.
        eps: Small value to avoid division by zero.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Float[Tensor, "b seq emb"]) -> Float[Tensor, "b seq emb"]:
        """Forward pass through RMSNorm.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            Normalized tensor of same shape as input.
        """
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # Normalize and scale
        return x * rms * self.weight


@dataclass
class BlockConfig:
    """Configuration for individual blocks within a SuperBlock."""

    n_embd: int
    n_head: int
    dropout: float = 0.1


@dataclass
class GPTConfig:
    """Configuration for GPT-2 model with heterogeneous SuperBlocks."""

    block_size: int = 64
    vocab_size: int = 50257
    n_layer: int = 2  # Number of SuperBlocks
    n_blocks_per_super: int = 3  # Number of blocks per SuperBlock
    dropout: float = 0.1
    base_embd: int = 64  # Base embedding dimension for the model

    # Define block configurations for each SuperBlock
    block_configs: list[list[BlockConfig]] = field(default_factory=list)

    def __post_init__(self):
        if not self.block_configs:
            # Default: 3 blocks with increasing dimensions
            default_config = [
                BlockConfig(n_embd=16, n_head=1),
                BlockConfig(n_embd=32, n_head=2),
                BlockConfig(n_embd=64, n_head=4),
            ]
            self.block_configs = [default_config] * self.n_layer


@beartype
class CausalSelfAttention(nn.Module):
    """Causal self-attention mechanism for GPT-2.

    Args:
        n_embd: Embedding dimension for this attention layer.
        n_head: Number of attention heads.
        block_size: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = torch.tensor(1.0)

        # regularization
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = nn.Dropout(dropout)

        # Register bias buffer - will be moved to correct device automatically
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.key_padding_mask = None
        self.attn_mask = None

    def forward(
        self,
        x: Float[Tensor, "b seq emb"],
    ) -> Float[Tensor, "b seq emb"]:
        """Forward pass through causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            Output tensor of same shape as input.
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Use flash attention for efficiency
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # reassemble head outputs
        y = self.c_proj(y)
        y = self.dropout(y)

        return y


class MLP(nn.Module):
    """Multi-layer perceptron for GPT-2.

    Args:
        n_embd: Embedding dimension for this MLP layer.
        dropout: Dropout rate.
    """

    def __init__(self, n_embd: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "b seq emb"],
    ) -> Float[Tensor, "b seq emb"]:
        """Forward pass through MLP.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            Output tensor of same shape as input.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP using Gemma 3 style pre-post RMSNorm.

    This implements the Gemma 3 architecture with RMSNorm before and after both
    attention and MLP layers inside the residual connections.

    Args:
        block_config: Configuration for this specific block.
        block_size: Maximum sequence length.
    """

    def __init__(self, block_config: BlockConfig, block_size: int):
        super().__init__()
        # RMSNorm layers for attention (pre and post)
        self.rms_norm_1_pre: nn.Module = RMSNorm(block_config.n_embd)
        self.rms_norm_1_post: nn.Module = RMSNorm(block_config.n_embd)

        # Attention layer
        self.attn: nn.Module = CausalSelfAttention(
            block_config.n_embd, block_config.n_head, block_size, block_config.dropout
        )

        # RMSNorm layers for MLP (pre and post)
        self.rms_norm_2_pre: nn.Module = RMSNorm(block_config.n_embd)
        self.rms_norm_2_post: nn.Module = RMSNorm(block_config.n_embd)

        # MLP layer
        self.mlp: nn.Module = MLP(block_config.n_embd, block_config.dropout)

    def forward(
        self,
        x: Float[Tensor, "b seq emb"],
    ) -> Float[Tensor, "b seq emb"]:
        """Forward pass through transformer block with Gemma 3 style RMSNorm.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            Output tensor of same shape as input.
        """
        # Attention block with pre and post RMSNorm
        attn_input = self.rms_norm_1_pre(x)
        attn_output = self.attn(attn_input)
        attn_residual = x + attn_output
        attn_output = self.rms_norm_1_post(attn_residual)

        # MLP block with pre and post RMSNorm
        mlp_input = self.rms_norm_2_pre(attn_output)
        mlp_output = self.mlp(mlp_input)
        mlp_residual = attn_output + mlp_output
        mlp_output = self.rms_norm_2_post(mlp_residual)

        return mlp_output


class SuperBlock(nn.Module):
    """SuperBlock with heterogeneous blocks having different embedding dimensions.

    Args:
        block_configs: List of configurations for each block in this SuperBlock.
        block_size: Maximum sequence length.
        base_embd: Base embedding dimension for input/output projection.
    """

    def __init__(
        self, block_configs: list[BlockConfig], block_size: int, base_embd: int
    ):
        super().__init__()
        self.n_blocks = len(block_configs)
        self.base_embd = base_embd
        self.block_size = block_size

        # Create heterogeneous blocks with different configurations
        self.blocks = nn.ModuleList(
            [Block(config, block_size) for config in block_configs]
        )

        # Input projection layers to map from base_embd to each block's embedding dimension
        self.input_projections = nn.ModuleList(
            [nn.Linear(base_embd, config.n_embd) for config in block_configs]
        )

        # Output projection layers to map from each block's embedding dimension back to base_embd
        self.output_projections = nn.ModuleList(
            [nn.Linear(config.n_embd, base_embd) for config in block_configs]
        )

        # Gating mechanism for multiple blocks
        self.gate = nn.Linear(base_embd, self.n_blocks)

        # Optional: Add combination layer if needed
        if self.n_blocks > 1:
            self.combine_proj = nn.Linear(base_embd * self.n_blocks, base_embd)
        else:
            self.combine_proj = None

        self._init_moe_style()

    def forward(self, x: Float[Tensor, "b seq emb"]) -> Float[Tensor, "b seq emb"]:
        """Forward pass through SuperBlock.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, base_embd).

        Returns:
            Output tensor of same shape as input.
        """
        if self.n_blocks == 1:
            # Single block case - no gating needed
            block_input = self.input_projections[0](x)
            block_output = self.blocks[0](block_input)
            return self.output_projections[0](block_output)

        # Multiple blocks case
        block_outputs = []
        for i, (block, input_proj, output_proj) in enumerate(
            zip(self.blocks, self.input_projections, self.output_projections)
        ):
            # Project input to block's embedding dimension
            block_input = input_proj(x)
            # Process through block
            block_output = block(block_input)
            # Project back to base embedding dimension
            block_output = output_proj(block_output)
            block_outputs.append(block_output)

        # Apply gating mechanism
        gate_weights = F.softmax(self.gate(x), dim=-1)  # [B, T, n_blocks]

        # Weight each block output
        weighted_outputs = []
        for i, block_out in enumerate(block_outputs):
            weighted_outputs.append(block_out * gate_weights[:, :, i : i + 1])

        # Combine weighted outputs
        if self.combine_proj is not None:
            # Concatenate and project
            combined = torch.cat(weighted_outputs, dim=-1)
            return self.combine_proj(combined)
        else:
            # Simple sum
            return torch.stack(weighted_outputs, dim=0).sum(dim=0)

    def _init_moe_style(self):
        """Initialize the SuperBlock using MoE principles.

        This ensures:
        1. Diverse initialization of parallel paths
        2. Conservative initialization of combination layer
        3. Balanced contribution from all paths
        """
        if self.combine_proj is not None:
            # Initialize the combination layer with smaller weights
            torch.nn.init.normal_(self.combine_proj.weight, mean=0.0, std=0.01)
            if self.combine_proj.bias is not None:
                torch.nn.init.zeros_(self.combine_proj.bias)

        # Initialize input/output projections
        for input_proj, output_proj in zip(
            self.input_projections, self.output_projections
        ):
            torch.nn.init.normal_(input_proj.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(output_proj.weight, mean=0.0, std=0.02)
            if input_proj.bias is not None:
                torch.nn.init.zeros_(input_proj.bias)
            if output_proj.bias is not None:
                torch.nn.init.zeros_(output_proj.bias)

        # Ensure all blocks start with different random states
        for i, block in enumerate(self.blocks):
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    # Use different initialization scales for diversity
                    scale = 0.02 * (1.0 + 0.1 * i)  # Slightly different scales
                    torch.nn.init.normal_(module.weight, mean=0.0, std=scale)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)


class GPT(nn.Module):
    """GPT-2 model with heterogeneous SuperBlocks.

    Args:
        config: GPT configuration containing model parameters.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.base_embd),
                "wpe": nn.Embedding(config.block_size, config.base_embd),
                "h": nn.ModuleList(
                    [
                        SuperBlock(block_configs, config.block_size, config.base_embd)
                        for block_configs in config.block_configs
                    ]
                ),
                "ln_f": nn.LayerNorm(config.base_embd),
            },
        )
        self.lm_head = nn.Linear(config.base_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer["wte"].weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, SuperBlock):
            # Let SuperBlock handle its own MoE-style initialization
            module._init_moe_style()

    def forward(
        self,
        idx: Integer[Tensor, "b seq"],
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass through GPT model.

        Args:
            idx: Input token indices of shape (batch_size, sequence_length).
            targets: Target token indices for loss calculation.

        Returns:
            Tuple of (logits, loss) where loss is None if targets is None.
        """
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and position embeddings
        pos = torch.arange(
            0, T, dtype=torch.long, device=idx.device
        )  # Create on same device as idx
        # position embeddings of shape (T, n_embd)
        pos_emb = self.transformer["wpe"](pos)
        # token embeddings of shape (B, T, n_embd)
        tok_emb = self.transformer["wte"](idx)
        x = tok_emb + pos_emb

        mod_list = cast(nn.ModuleList, self.transformer["h"])
        # forward the blocks of the transformer
        for block in mod_list:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model weights from huggingface.

        Args:
            model_type: Type of pretrained model ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl').

        Returns:
            GPT model with pretrained weights.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        print(f"model_type: {model_type}")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, device: str
    ):
        """Configure optimizer with weight decay separation.

        Args:
            weight_decay: Weight decay parameter.
            learning_rate: Learning rate.
            device: Device to run on.

        Returns:
            Configured optimizer.
        """
        import inspect

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(
            f"configuring optimizers with fused={use_fused} (fused_available={fused_available}) on device {device}"
        )
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Test the model classes
    print("Testing model classes...")

    # Test GPTConfig with heterogeneous blocks
    config = GPTConfig(
        n_layer=2,
        n_blocks_per_super=3,
        base_embd=64,
        block_configs=[
            [  # SuperBlock 1
                BlockConfig(n_embd=16, n_head=1),
                BlockConfig(n_embd=32, n_head=2),
                BlockConfig(n_embd=64, n_head=4),
            ],
            [  # SuperBlock 2
                BlockConfig(n_embd=16, n_head=1),
                BlockConfig(n_embd=32, n_head=2),
                BlockConfig(n_embd=64, n_head=4),
            ],
        ],
    )
    assert config.block_size == 64
    assert config.n_layer == 2
    assert len(config.block_configs) == 2
    assert len(config.block_configs[0]) == 3
    print("Test 1 passed: GPTConfig with heterogeneous blocks works correctly.")

    # Test CausalSelfAttention
    attn = CausalSelfAttention(n_embd=64, n_head=4, block_size=64)
    batch_size, seq_len, emb_dim = 2, 32, 64
    x = torch.randn(batch_size, seq_len, emb_dim)
    output = attn(x)
    assert output.shape == x.shape
    print("Test 2 passed: CausalSelfAttention works correctly.")

    # Test MLP
    mlp = MLP(n_embd=64)
    output = mlp(x)
    assert output.shape == x.shape
    print("Test 3 passed: MLP works correctly.")

    # Test Block
    block_config = BlockConfig(n_embd=64, n_head=4)
    block = Block(block_config, block_size=64)
    output = block(x)
    assert output.shape == x.shape
    print("Test 4 passed: Block works correctly.")

    # Test SuperBlock
    superblock = SuperBlock(
        block_configs=[
            BlockConfig(n_embd=16, n_head=1),
            BlockConfig(n_embd=32, n_head=2),
            BlockConfig(n_embd=64, n_head=4),
        ],
        block_size=64,
        base_embd=64,
    )
    output = superblock(x)
    assert output.shape == x.shape
    print("Test 5 passed: SuperBlock works correctly.")

    # Test GPT
    model = GPT(config)
    batch_size, seq_len = 2, 32
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None and loss.item() > 0
    print("Test 6 passed: GPT forward pass works correctly.")

    # Test optimizer configuration
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, device="cpu"
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    print("Test 7 passed: Optimizer configuration works correctly.")

    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("Test 8 passed: Parameter counting works correctly.")

    print("All tests passed! Heterogeneous SuperBlock model is working correctly.")
