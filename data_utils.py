import os

import lightning as pl
import torch
from beartype import beartype
from jaxtyping import Int
from torch.utils.data import DataLoader, Dataset


@beartype
class TextDataset(Dataset):
    """A dataset for loading and tokenizing text for GPT-2 training.

    Args:
        file_path: Path to the input text file.
        block_size: Length of each sequence (context window).
        vocab: Optional mapping from character to integer token.
        stride: Step size for moving the window (default: block_size).

    Attributes:
        data: List of tokenized integers.
        block_size: Length of each sequence.
        stride: Step size for moving the window.
        vocab: Mapping from character to integer token.
        inv_vocab: Mapping from integer token to character.

    """

    def __init__(
        self,
        file_path: str,
        block_size: int,
        vocab: dict[str, int] = None,
        stride: int = None,
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        if vocab is None:
            chars = sorted(list(set(text)))
            self.vocab = {ch: i for i, ch in enumerate(chars)}
        else:
            self.vocab = vocab
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        self.data = [self.vocab[ch] for ch in text if ch in self.vocab]
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size

    def __len__(self) -> int:
        return (len(self.data) - self.block_size) // self.stride

    def __getitem__(
        self, idx: int
    ) -> tuple[Int[torch.Tensor, "block_size"], Int[torch.Tensor, "block_size"]]:
        start = idx * self.stride
        end = start + self.block_size
        x = torch.tensor(self.data[start:end], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : end + 1], dtype=torch.long)
        return x, y


class TextDataModule(pl.LightningDataModule):
    """LightningDataModule for text data.

    Args:
        data_path: Path to the input text file.
        block_size: Length of each sequence.
        batch_size: Batch size for DataLoader.
        num_workers: Number of DataLoader workers.
        val_split: Fraction of data to use for validation.

    """

    def __init__(
        self,
        data_path: str,
        block_size: int,
        batch_size: int,
        num_workers: int = 0,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage: str = None) -> None:
        full_dataset = TextDataset(self.data_path, self.block_size)
        n_val = int(len(full_dataset) * self.val_split)
        n_train = len(full_dataset) - n_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    # Test the dataset and datamodule
    # Create a small test file
    test_file = "input.txt"

    # Test Dataset
    dataset = TextDataset(test_file, block_size=8)
    x, y = dataset[0]
    assert len(x) == 8 and len(y) == 8
    print(f"Test 1 passed: Dataset returns correct shapes.")

    # Test DataModule
    batch_size = 12
    print("batch_size", batch_size)
    dm = TextDataModule(test_file, block_size=8, batch_size=batch_size)
    dm.setup()
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert batch[0].shape[1] == 8
    print(f"Test 2 passed: DataModule returns correct batch shapes.")

    # Test 3: Check batch sizes, including last batch
    all_batches = list(train_loader)
    print(f"total batches: {len(all_batches)}")
    for i, batch in enumerate(all_batches):
        print("batch", i)
        current_batch_size = batch[0].shape[0]
        print(f"{batch[0].shape=}, {batch[1].shape=}")
        print(f"{batch[0]=}, {batch[1]=}")
        if i < len(all_batches) - 1:
            assert (
                current_batch_size == batch_size
            ), f"Batch {i} size {current_batch_size} != {batch_size}"
        else:
            # Last batch can be <= batch_size
            print("last batch size", current_batch_size)
            assert (
                current_batch_size <= batch_size
            ), f"Last batch size {current_batch_size} > {batch_size}"
    print(f"Test 3 passed: All batch sizes are correct (last batch may be smaller).")

    os.remove(test_file)
