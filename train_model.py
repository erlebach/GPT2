from training import train_with_lightning

# Automatic device detection (recommended)
train_with_lightning(
    data_path="input.txt",
    block_size=64,
    batch_size=64,
    max_steps=20,
    n_layer=1,  # Number of super-layers
    n_head=2,  # Number of attention heads
    n_embd=64,  # Embedding dimension
    n_blocks_per_super=2,  # NEW: Number of blocks per SuperBlock
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
