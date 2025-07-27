"""Example script showing how to use evaluation metrics with GPT-2 model."""

import tiktoken
import torch
from data_utils import TextDataModule
from evaluation_metrics import (
    evaluate_model_on_dataset,
    generate_text_for_evaluation,
    print_evaluation_summary,
)
from train_gpt2 import GPT, GPTConfig


def main():
    """Main function demonstrating evaluation metrics usage."""

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create a small model for testing
    config = GPTConfig(
        block_size=32,
        vocab_size=50257,
        n_layer=4,
        n_head=4,
        n_embd=64,
    )

    model = GPT(config)

    # Create data module
    data_module = TextDataModule(
        data_path="input.txt",
        block_size=32,
        batch_size=4,
        val_split=0.2,
    )

    # Setup data
    data_module.setup()

    # Evaluate on validation set
    print("Evaluating model on validation dataset...")
    val_metrics = evaluate_model_on_dataset(
        model=model,
        dataloader=data_module.val_dataloader(),
        vocab_size=config.vocab_size,
    )

    # Print results
    print_evaluation_summary(val_metrics)

    # Generate some text for evaluation
    print("\nGenerating sample text...")
    prompt = "The quick brown fox"
    generated_text = generate_text_for_evaluation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=50,
        temperature=0.8,
    )

    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
