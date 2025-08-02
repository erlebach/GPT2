# from training.gpt2 import train_with_lightning
from pathlib import Path

from gpt2_standalone.lightning_module import train_with_lightning
from utils.data_utils import get_project_root
from utils.metrics_extensions import get_metrics_collector, save_metrics_to_csv


def main():
    """Main function to run GPT-2 training."""
    train_with_lightning(
        data_path=get_project_root() / "data" / "input.txt",
        block_size=1024,
        batch_size=64,
        max_steps=10000,
        n_layer=4,  # Number of super-layers
        n_head=8,  # Number of attention heads
        n_embd=512,  # Embedding dimension
        n_blocks_per_super=2,  # NEW: Number of blocks per SuperBlock
        weight_decay=0.2,
        accelerator="auto",
        devices="auto",
        # resume=True,  # True: resume from checkpoint
        checkpoint_path=Path("checkpoints/"),
        # checkpoint="model-epochepoch=01-val_lossval_loss=6.54.ckpt",
        checkpoint=None  # no restart
    )

    collector = get_metrics_collector()
    collector.save_all_metrics_to_csv("metrics.csv")


if __name__ == "__main__":
    main()
