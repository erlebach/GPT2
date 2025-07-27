import torch
from torch import Tensor
from torch import nn
from jaxtyping import Float
import torch.nn.functional as F

def swiglu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """SwiGLU activation function: Swish(x) * Gate(x).

    SwiGLU (Swish-Gated Linear Unit) is an activation function that combines
    the benefits of Swish activation with gating mechanisms. It typically
    provides better performance than GELU but requires more parameters.

    The function expects input to be split into two equal halves along the last dimension:
    - First half: goes through Swish activation
    - Second half: used as a gate (no activation)

    Args:
        x: Input tensor that should be split into two equal halves along the last dimension.

    Returns:
        SwiGLU activated tensor of same shape as input.

    Example:
        >>> x = torch.randn(2, 3, 8)  # Last dim must be even
        >>> x1, x2 = x.chunk(2, dim=-1)  # Split into two halves
        >>> result = swiglu(x)  # Equivalent to F.silu(x1) * x2
    """
    # Split input into two equal halves along the last dimension
    x1, x2 = x.chunk(2, dim=-1)
    # Apply Swish activation to x1 and use x2 as gate
    return F.silu(x1) * x2


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
