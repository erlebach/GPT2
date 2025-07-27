from training.super_gpt2 import train_with_lightning
from utils.data_utils import get_project_root

# Automatic device detection (recommended)
train_with_lightning(
    data_path=get_project_root() / "data" / "input.txt",
    block_size=64,
    batch_size=64,
    max_steps=500,
    n_layer=2,  # Number of super-layers
    n_blocks_per_super=3,  # Number of blocks per SuperBlock
    base_embd=64,  # Base embedding dimension
    weight_decay=0.2,
    accelerator="auto",
    devices="auto",
)

# # Force CPU
# train_with_lightning(
#     data_path="input.txt",
#     block_size=64,
#     batch_size=64,
#     max_epochs=10,
#     accelerator="cpu",
#     devices=1,
# )

# # Force GPU
# train_with_lightning(
#     data_path="input.txt",
#     block_size=64,
#     batch_size=64,
#     max_epochs=10,
#     accelerator="gpu",
#     devices=1,
# )
