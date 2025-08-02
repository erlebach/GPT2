# from training.gpt2 import train_with_lightning
from gpt2_standalone.lightning_module import train_with_lightning
from utils.data_utils import get_project_root


def main():
    """Main function to run GPT-2 training."""
    train_with_lightning(
        data_path=get_project_root() / "data" / "input.txt",
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


if __name__ == "__main__":
    main()
