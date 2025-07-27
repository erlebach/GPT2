from training import train_with_lightning

# Automatic device detection (recommended)
train_with_lightning(
    data_path="input.txt",
    block_size=64,
    batch_size=64,
    # max_epochs=10,
    max_steps=5000,
    accelerator="auto",  # Will use GPU if available, CPU otherwise
    devices="auto",  # Will use all available devices
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
