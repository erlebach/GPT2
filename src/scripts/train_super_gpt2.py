from training.super_gpt2 import train_with_lightning
from utils.data_utils import get_project_root

# Train with SwiGLU activation (enabled by default)
train_with_lightning(
    data_path=get_project_root() / "data" / "input.txt",
    block_size=64,
    batch_size=64,
    max_steps=5000,
    n_layer=2,  # Number of super-layers
    n_blocks_per_super=3,  # Number of blocks per SuperBlock
    base_embd=64,  # Base embedding dimension
    weight_decay=0.1,
    accelerator="auto",
    devices="auto",
    use_swiglu=True,  # Explicitly enable SwiGLU
)

# # Alternative: Train with GELU activation (for comparison)
# train_with_lightning(
#     data_path=get_project_root() / "data" / "input.txt",
#     block_size=64,
#     batch_size=64,
#     max_steps=5000,
#     n_layer=2,
#     n_blocks_per_super=3,
#     base_embd=64,
#     weight_decay=0.1,
#     accelerator="auto",
#     devices="auto",
#     use_swiglu=False,  # Use GELU instead
# )

# # Force CPU training with SwiGLU
# train_with_lightning(
#     data_path=get_project_root() / "data" / "input.txt",
#     block_size=64,
#     batch_size=64,
#     max_epochs=10,
#     accelerator="cpu",
#     devices=1,
#     use_swiglu=True,
# )

# # Force GPU training with SwiGLU
# train_with_lightning(
#     data_path=get_project_root() / "data" / "input.txt",
#     block_size=64,
#     batch_size=64,
#     max_epochs=10,
#     accelerator="gpu",
#     devices=1,
#     use_swiglu=True,
# )
