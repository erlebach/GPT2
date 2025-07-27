"""Evaluation metrics for language models, including perplexity and other quality measures.

## Key Points About Perplexity and Evaluation Metrics:

### **Perplexity**
- **Definition**: `perplexity = exp(cross_entropy_loss)`
- **Interpretation**: Lower is better. It measures how "surprised" the model is by the next token
- **Example**: A perplexity of 2.0 means the model is as uncertain as if it had to choose between 2 equally likely options

### **Other Important Metrics**:
1. **Loss**: Direct cross-entropy loss (lower is better)
2. **Accuracy**: Token-level prediction accuracy
3. **Top-k Accuracy**: Whether the correct token appears in top-k predictions
4. **Entropy**: Measures prediction uncertainty
5. **Repetition Penalty**: Detects repetitive text generation
6. **Diversity Score**: Measures vocabulary variety

### **Usage**:
- Use `evaluate_model_on_batch()` for single batch evaluation
- Use `evaluate_model_on_dataset()` for full dataset evaluation
- Use `generate_text_for_evaluation()` to generate text for qualitative assessment

"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Integer


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics results.

    Args:
        perplexity: Perplexity score (lower is better).
        loss: Cross-entropy loss.
        accuracy: Token-level accuracy.
        top_k_accuracy: Top-k accuracy for k=1,3,5.
        entropy: Average entropy of predictions.
        repetition_penalty: Penalty for repetitive text.
        diversity_score: Measure of vocabulary diversity.
    """

    perplexity: float
    loss: float
    accuracy: float
    top_k_accuracy: dict[int, float]
    entropy: float
    repetition_penalty: float
    diversity_score: float


@beartype
def calculate_perplexity(
    logits: Float[torch.Tensor, "batch seq vocab"],
    targets: Integer[torch.Tensor, "batch seq"],
    ignore_index: int = -100,
) -> float:
    """Calculate perplexity from model logits and targets.

    Perplexity is defined as exp(cross_entropy_loss). It measures how well
    the model predicts the next token. Lower perplexity indicates better
    performance.

    Args:
        logits: Model output logits of shape (batch, seq, vocab_size).
        targets: Target token indices of shape (batch, seq).
        ignore_index: Index to ignore in loss calculation (e.g., padding).

    Returns:
        Perplexity score (float).

    """
    # Reshape logits and targets for cross entropy
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    # Calculate cross entropy loss
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)

    # Perplexity is exp(loss)
    perplexity = torch.exp(loss).item()

    return perplexity


@beartype
def calculate_loss(
    logits: Float[torch.Tensor, "batch seq vocab"],
    targets: Integer[torch.Tensor, "batch seq"],
    ignore_index: int = -100,
) -> float:
    """Calculate cross-entropy loss from model logits and targets.

    Args:
        logits: Model output logits of shape (batch, seq, vocab_size).
        targets: Target token indices of shape (batch, seq).
        ignore_index: Index to ignore in loss calculation.

    Returns:
        Cross-entropy loss (float).

    """
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)
    return loss.item()


@beartype
def calculate_accuracy(
    logits: Float[torch.Tensor, "batch seq vocab"],
    targets: Integer[torch.Tensor, "batch seq"],
    ignore_index: int = -100,
) -> float:
    """Calculate token-level accuracy.

    Args:
        logits: Model output logits of shape (batch, seq, vocab_size).
        targets: Target token indices of shape (batch, seq).
        ignore_index: Index to ignore in accuracy calculation.

    Returns:
        Accuracy score between 0 and 1.

    """
    # Get predicted tokens
    predictions = torch.argmax(logits, dim=-1)

    # Create mask for non-ignored tokens
    mask = targets != ignore_index

    # Calculate accuracy only on non-ignored tokens
    correct = (predictions == targets) & mask
    total = mask.sum()

    if total == 0:
        return 0.0

    accuracy = correct.sum().float() / total
    return accuracy.item()


@beartype
def calculate_top_k_accuracy(
    logits: Float[torch.Tensor, "batch seq vocab"],
    targets: Integer[torch.Tensor, "batch seq"],
    k_values: list[int] = [1, 3, 5],
    ignore_index: int = -100,
) -> Dict[int, float]:
    """Calculate top-k accuracy for multiple k values.

    Args:
        logits: Model output logits of shape (batch, seq, vocab_size).
        targets: Target token indices of shape (batch, seq).
        k_values: List of k values to calculate accuracy for.
        ignore_index: Index to ignore in accuracy calculation.

    Returns:
        Dictionary mapping k values to accuracy scores.

    """
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    # Create mask for non-ignored tokens
    mask = targets_flat != ignore_index

    results = {}

    for k in k_values:
        # Get top-k predictions
        top_k_preds = torch.topk(logits_flat, k, dim=-1)[1]

        # Check if target is in top-k predictions
        target_expanded = targets_flat.unsqueeze(-1).expand_as(top_k_preds)
        correct = (top_k_preds == target_expanded).any(dim=-1)

        # Calculate accuracy only on non-ignored tokens
        correct_masked = correct & mask
        total = mask.sum()

        if total == 0:
            accuracy = 0.0
        else:
            accuracy = correct_masked.sum().float() / total

        results[k] = accuracy.item()

    return results


@beartype
def calculate_entropy(
    logits: Float[torch.Tensor, "batch seq vocab"],
    targets: Integer[torch.Tensor, "batch seq"],
    ignore_index: int = -100,
) -> float:
    """Calculate average entropy of model predictions.

    Higher entropy indicates more uncertainty in predictions.

    Args:
        logits: Model output logits of shape (batch, seq, vocab_size).
        targets: Target token indices of shape (batch, seq).
        ignore_index: Index to ignore in calculation.

    Returns:
        Average entropy score.

    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

    # Create mask for non-ignored tokens
    mask = targets != ignore_index

    # Calculate average entropy only on non-ignored tokens
    if mask.sum() == 0:
        return 0.0

    avg_entropy = (entropy * mask).sum() / mask.sum()
    return avg_entropy.item()


@beartype
def calculate_repetition_penalty(
    generated_tokens: Integer[torch.Tensor, "seq"],
    window_size: int = 10,
) -> float:
    """Calculate repetition penalty for generated text.

    This measures how repetitive the generated text is by counting
    repeated n-grams within a sliding window.

    Args:
        generated_tokens: Sequence of generated token indices.
        window_size: Size of sliding window for n-gram detection.

    Returns:
        Repetition penalty score (higher = more repetitive).

    """
    if len(generated_tokens) < window_size:
        return 0.0

    tokens = generated_tokens.tolist()
    repeated_ngrams = 0
    total_ngrams = 0

    for i in range(len(tokens) - window_size + 1):
        ngram = tuple(tokens[i : i + window_size])
        total_ngrams += 1

        # Check if this n-gram appears later in the sequence
        for j in range(i + 1, len(tokens) - window_size + 1):
            if tuple(tokens[j : j + window_size]) == ngram:
                repeated_ngrams += 1
                break

    if total_ngrams == 0:
        return 0.0

    repetition_ratio = repeated_ngrams / total_ngrams
    return repetition_ratio


@beartype
def calculate_diversity_score(
    generated_tokens: Integer[torch.Tensor, "seq"],
    vocab_size: int,
) -> float:
    """Calculate vocabulary diversity score.

    This measures how diverse the vocabulary usage is in generated text.
    Higher score indicates more diverse vocabulary usage.

    Args:
        generated_tokens: Sequence of generated token indices.
        vocab_size: Size of the vocabulary.

    Returns:
        Diversity score between 0 and 1 (higher = more diverse).

    """
    if len(generated_tokens) == 0:
        return 0.0

    # Count unique tokens
    unique_tokens = torch.unique(generated_tokens)
    num_unique = len(unique_tokens)

    # Calculate diversity as ratio of unique tokens to total tokens
    diversity = num_unique / len(generated_tokens)

    return diversity


@beartype
def evaluate_model_on_batch(
    model: torch.nn.Module,
    batch: tuple[
        Integer[torch.Tensor, "batch seq"], Integer[torch.Tensor, "batch seq"]
    ],
    vocab_size: int,
    ignore_index: int = -100,
) -> EvaluationMetrics:
    """Evaluate model on a single batch and return comprehensive metrics.

    Args:
        model: The language model to evaluate.
        batch: Tuple of (input_ids, target_ids).
        vocab_size: Size of the vocabulary.
        ignore_index: Index to ignore in calculations.

    Returns:
        EvaluationMetrics object containing all calculated metrics.

    """
    input_ids, target_ids = batch

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Forward pass
        logits, _ = model(input_ids, target_ids)

        # Calculate basic metrics
        loss = calculate_loss(logits, target_ids, ignore_index)
        perplexity = calculate_perplexity(logits, target_ids, ignore_index)
        accuracy = calculate_accuracy(logits, target_ids, ignore_index)
        top_k_acc = calculate_top_k_accuracy(
            logits, target_ids, ignore_index=ignore_index
        )
        entropy = calculate_entropy(logits, target_ids, ignore_index)

        # For repetition and diversity, we need generated text
        # For now, we'll use the input sequence as a proxy
        repetition_penalty = calculate_repetition_penalty(input_ids[0])
        diversity_score = calculate_diversity_score(input_ids[0], vocab_size)

    return EvaluationMetrics(
        perplexity=perplexity,
        loss=loss,
        accuracy=accuracy,
        top_k_accuracy=top_k_acc,
        entropy=entropy,
        repetition_penalty=repetition_penalty,
        diversity_score=diversity_score,
    )


@beartype
def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    vocab_size: int,
    ignore_index: int = -100,
) -> EvaluationMetrics:
    """Evaluate model on entire dataset and return averaged metrics.

    Args:
        model: The language model to evaluate.
        dataloader: DataLoader containing evaluation data.
        vocab_size: Size of the vocabulary.
        ignore_index: Index to ignore in calculations.

    Returns:
        EvaluationMetrics object with averaged metrics across all batches.

    """
    model.eval()

    # Initialize accumulators
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    total_entropy = 0.0
    total_repetition_penalty = 0.0
    total_diversity_score = 0.0
    total_top_k_correct = defaultdict(int)
    total_top_k_total = defaultdict(int)
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, target_ids = batch

            # Forward pass
            logits, _ = model(input_ids, target_ids)

            # Calculate metrics for this batch
            batch_loss = calculate_loss(logits, target_ids, ignore_index)
            batch_perplexity = calculate_perplexity(logits, target_ids, ignore_index)
            batch_accuracy = calculate_accuracy(logits, target_ids, ignore_index)
            batch_entropy = calculate_entropy(logits, target_ids, ignore_index)
            batch_top_k_acc = calculate_top_k_accuracy(
                logits, target_ids, ignore_index=ignore_index
            )

            # Accumulate metrics
            total_loss += batch_loss
            total_perplexity += batch_perplexity
            total_accuracy += batch_accuracy
            total_entropy += batch_entropy

            # Handle top-k accuracy accumulation
            for k, acc in batch_top_k_acc.items():
                # We need to calculate the actual counts for proper averaging
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = target_ids.view(-1)
                mask = targets_flat != ignore_index
                total_tokens = mask.sum().item()

                top_k_preds = torch.topk(logits_flat, k, dim=-1)[1]
                target_expanded = targets_flat.unsqueeze(-1).expand_as(top_k_preds)
                correct = (top_k_preds == target_expanded).any(dim=-1)
                correct_masked = correct & mask

                total_top_k_correct[k] += correct_masked.sum().item()
                total_top_k_total[k] += total_tokens

            # For repetition and diversity, use first sequence in batch
            batch_repetition = calculate_repetition_penalty(input_ids[0])
            batch_diversity = calculate_diversity_score(input_ids[0], vocab_size)
            total_repetition_penalty += batch_repetition
            total_diversity_score += batch_diversity

            num_batches += 1

    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_entropy = total_entropy / num_batches
    avg_repetition_penalty = total_repetition_penalty / num_batches
    avg_diversity_score = total_diversity_score / num_batches

    # Calculate final top-k accuracies
    final_top_k_acc = {}
    for k in total_top_k_correct.keys():
        if total_top_k_total[k] > 0:
            final_top_k_acc[k] = total_top_k_correct[k] / total_top_k_total[k]
        else:
            final_top_k_acc[k] = 0.0

    return EvaluationMetrics(
        perplexity=avg_perplexity,
        loss=avg_loss,
        accuracy=avg_accuracy,
        top_k_accuracy=final_top_k_acc,
        entropy=avg_entropy,
        repetition_penalty=avg_repetition_penalty,
        diversity_score=avg_diversity_score,
    )


@beartype
def generate_text_for_evaluation(
    model: torch.nn.Module,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """Generate text for evaluation purposes.

    Args:
        model: The language model to use for generation.
        tokenizer: Tiktoken tokenizer for encoding/decoding.
        prompt: Input prompt text.
        max_length: Maximum length of generated text.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.

    Returns:
        Generated text string.

    """
    model.eval()

    # Encode the prompt
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits, _ = model(input_ids, None)

            # Get logits for the last token
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits[top_k_indices] = top_k_logits

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated tokens
            generated_tokens.append(next_token.item())

            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Decode the generated text
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


@beartype
def print_evaluation_summary(metrics: EvaluationMetrics) -> None:
    """Print a formatted summary of evaluation metrics.

    Args:
        metrics: EvaluationMetrics object containing the results.

    """
    print("=" * 60)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(f"Perplexity: {metrics.perplexity:.4f}")
    print(f"Loss: {metrics.loss:.4f}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Entropy: {metrics.entropy:.4f}")
    print(f"Repetition Penalty: {metrics.repetition_penalty:.4f}")
    print(f"Diversity Score: {metrics.diversity_score:.4f}")
    print("\nTop-k Accuracy:")
    for k, acc in metrics.top_k_accuracy.items():
        print(f"  Top-{k}: {acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    # Test the evaluation functions with a simple example
    print("Testing evaluation metrics functions...")

    # Create dummy data
    batch_size, seq_len, vocab_size = 2, 10, 1000
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test individual metric functions
    print(f"Perplexity: {calculate_perplexity(logits, targets):.4f}")
    print(f"Loss: {calculate_loss(logits, targets):.4f}")
    print(f"Accuracy: {calculate_accuracy(logits, targets):.4f}")
    print(f"Top-k Accuracy: {calculate_top_k_accuracy(logits, targets)}")
    print(f"Entropy: {calculate_entropy(logits, targets):.4f}")

    # Test repetition penalty and diversity
    tokens = torch.randint(0, vocab_size, (seq_len,))
    print(f"Repetition Penalty: {calculate_repetition_penalty(tokens):.4f}")
    print(f"Diversity Score: {calculate_diversity_score(tokens, vocab_size):.4f}")

    print("All tests passed!")
