"""Memory measurement experiments with triplet-based structure."""

import torch
from gpt2_standalone.lightning_module import GPTLightningModule
from gpt2_standalone.model import GPTConfig
from lightning import Fabric

from experiments.clean_palate import deep_gpu_reset, reset_model_state


def run_single_experiment(
    fabric: Fabric,
    model_name: str,
    batch_size: int,
    sequence_length: int,
    model_config: dict,
    mode: str = "training",
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run a single memory experiment for a specific (model, batch_size, sequence_length) triplet.

    Args:
        fabric: Lightning Fabric instance.
        model_name: Name of the model (e.g., 'tiny', 'small').
        batch_size: Batch size to test.
        sequence_length: Sequence length to test.
        model_config: Model configuration dictionary.
        mode: Either 'training' or 'evaluation'.
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing memory experiment results for this specific triplet.
    """
    import gc
    import statistics
    import time

    start_time = time.time()
    print(
        f"   Testing: {model_name}, batch_size={batch_size}, seq_len={sequence_length}, mode={mode}"
    )

    try:
        # Clean palate before starting
        print(f"     Setting up model...")
        deep_gpu_reset()

        # Create fresh model
        model_config_obj = GPTConfig(
            block_size=sequence_length,
            vocab_size=50304,
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            n_embd=model_config["n_embd"],
            n_blocks_per_super=2,
        )

        model = GPTLightningModule(model_config_obj)
        model, optimizer = fabric.setup(
            model, model.configure_optimizers()["optimizer"]
        )

        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Create test data
        print(f"     Creating test data...")
        x = torch.randint(0, 50304, (batch_size, sequence_length), device=fabric.device)
        y = torch.randint(0, 50304, (batch_size, sequence_length), device=fabric.device)
        test_batch = (x, y)

        # Set model mode
        if mode == "evaluation":
            model.eval()
        else:
            model.train()

        # Warmup with clean palate between iterations
        print(f"     Warming up ({warmup_iterations} iterations)...")
        for i in range(warmup_iterations):
            print(f"       Warmup {i+1}/{warmup_iterations}")
            deep_gpu_reset()
            if mode == "evaluation":
                with torch.no_grad():
                    _ = model(x)
            else:
                _ = model.training_step(test_batch, batch_idx=0)

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

        # Measure memory over multiple iterations
        print(f"     Measuring memory ({num_iterations} iterations)...")
        memory_readings = []
        forward_memory_readings = []
        backward_memory_readings = []
        peak_forward_readings = []
        peak_backward_readings = []

        for i in range(num_iterations):
            print(f"       Iteration {i+1}/{num_iterations}")

            # Clean palate before each measurement
            deep_gpu_reset()
            reset_model_state(model, optimizer)

            # Clear gradients
            optimizer.zero_grad()

            # Measure forward pass memory
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()

            with torch.no_grad():
                _ = model(x)

            forward_mem = torch.cuda.memory_allocated()
            peak_forward_mem = torch.cuda.max_memory_allocated()
            forward_memory_readings.append(forward_mem / 1e9)
            peak_forward_readings.append(peak_forward_mem / 1e9)

            # Clean up forward pass
            del _
            torch.cuda.empty_cache()

            if mode == "training":
                # Measure backward pass only (with fresh forward pass)
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated()

                # Do forward pass again (needed for backward)
                with torch.no_grad():
                    _ = model(x)

                # Now do backward pass
                loss = model.training_step(test_batch, batch_idx=i)

                total_mem = torch.cuda.memory_allocated()
                peak_total_mem = torch.cuda.max_memory_allocated()

                # Backward memory is the additional memory used beyond the forward pass
                backward_mem = total_mem - start_mem
                peak_backward_mem = peak_total_mem

                memory_readings.append(total_mem / 1e9)
                backward_memory_readings.append(backward_mem / 1e9)
                peak_backward_readings.append(peak_backward_mem / 1e9)
            else:
                # In evaluation mode, total memory is just forward memory
                total_mem = forward_mem
                memory_readings.append(total_mem / 1e9)
                backward_memory_readings.append(0.0)
                peak_backward_readings.append(0.0)

        # Calculate statistics
        avg_memory = statistics.mean(memory_readings)
        std_memory = (
            statistics.stdev(memory_readings) if len(memory_readings) > 1 else 0.0
        )
        avg_forward_memory = statistics.mean(forward_memory_readings)
        avg_backward_memory = statistics.mean(backward_memory_readings)
        avg_peak_forward_memory = statistics.mean(peak_forward_readings)
        avg_peak_backward_memory = statistics.mean(peak_backward_readings)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        memory_per_sample = avg_memory / batch_size
        memory_per_token = avg_memory / (batch_size * sequence_length)

        result = {
            "model_name": model_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "config": model_config,
            "total_params": total_params,
            "avg_memory_gb": avg_memory,
            "std_memory_gb": std_memory,
            "avg_forward_memory_gb": avg_forward_memory,
            "avg_backward_memory_gb": avg_backward_memory,
            "avg_peak_forward_memory_gb": avg_peak_forward_memory,
            "avg_peak_backward_memory_gb": avg_peak_backward_memory,
            "peak_memory_gb": peak_memory,
            "memory_per_sample_gb": memory_per_sample,
            "memory_per_token_gb": memory_per_token,
            "memory_readings": memory_readings,
            "forward_memory_readings": forward_memory_readings,
            "backward_memory_readings": backward_memory_readings,
            "peak_forward_readings": peak_forward_readings,
            "peak_backward_readings": peak_backward_readings,
            "status": "success",
        }

        elapsed_time = time.time() - start_time
        print(
            f"       ✅ Completed in {elapsed_time:.1f}s - "
            f"Total: {avg_memory:.2f}GB ± {std_memory:.2f}GB, "
            f"Forward: {avg_forward_memory:.2f}GB, Backward: {avg_backward_memory:.2f}GB, "
            f"Peak F: {avg_peak_forward_memory:.2f}GB, Peak B: {avg_peak_backward_memory:.2f}GB"
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            elapsed_time = time.time() - start_time
            print(
                f"       ❌ Out of memory for {model_name}, batch_size={batch_size}, seq_len={sequence_length} (after {elapsed_time:.1f}s)"
            )
            result = {
                "model_name": model_name,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "mode": mode,
                "config": model_config,
                "status": "out_of_memory",
                "error": str(e),
            }
        else:
            elapsed_time = time.time() - start_time
            print(
                f"       ❌ Runtime error for {model_name}, batch_size={batch_size}, seq_len={sequence_length} (after {elapsed_time:.1f}s): {e}"
            )
            result = {
                "model_name": model_name,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "mode": mode,
                "config": model_config,
                "status": "runtime_error",
                "error": str(e),
            }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(
            f"       ❌ Unexpected error for {model_name}, batch_size={batch_size}, seq_len={sequence_length} (after {elapsed_time:.1f}s): {e}"
        )
        result = {
            "model_name": model_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "config": model_config,
            "status": "error",
            "error": str(e),
        }
    finally:
        # Clean up
        try:
            del model, optimizer, x, y, test_batch
        except NameError:
            pass  # Variables might not exist if error occurred early
        deep_gpu_reset()

    return result


def run_experiment_grid(
    fabric: Fabric,
    model_configs: list,
    batch_sizes: list,
    sequence_lengths: list,
    modes: list = ["training", "evaluation"],
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run experiments for all combinations of (model, batch_size, sequence_length) triplets.

    Args:
        fabric: Lightning Fabric instance.
        model_configs: List of model configuration dictionaries.
        batch_sizes: List of batch sizes to test (ordered from smallest to largest).
        sequence_lengths: List of sequence lengths to test (ordered from shortest to longest).
        modes: List of modes to test ('training' and/or 'evaluation').
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary with triplet keys and experiment results as values.
    """
    from datetime import datetime

    if fabric.global_rank != 0:
        return {}

    print(f"\n🧪 Starting Memory Scaling Experiments")
    print(f"   Models: {[config['name'] for config in model_configs]}")
    print(f"   Batch sizes: {batch_sizes}")
    print(f"   Sequence lengths: {sequence_lengths}")
    print(f"   Modes: {modes}")
    print(f"   Warmup iterations: {warmup_iterations}")
    print(f"   Measurement iterations: {num_iterations}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(fabric.device),
        "experiments": {},
    }

    total_experiments = (
        len(model_configs) * len(batch_sizes) * len(sequence_lengths) * len(modes)
    )
    experiment_count = 0

    for config in model_configs:
        model_name = config["name"]
        print(f"\n==> Testing model: {model_name}")

        for mode in modes:
            print(f"   Testing mode: {mode}")

            for sequence_length in sequence_lengths:
                print(f"     Testing sequence length: {sequence_length}")

                # Track OOM status for this model/mode/sequence_length combination
                oom_detected = False

                for batch_size in batch_sizes:
                    # Skip if OOM was detected for a smaller batch size
                    if oom_detected:
                        print(
                            f"       ⚠️  Skipping batch_size={batch_size} (OOM detected for smaller batch)"
                        )
                        continue

                    experiment_count += 1
                    print(f"\n       [{experiment_count}/{total_experiments}] ", end="")

                    # Create triplet key
                    triplet = (model_name, batch_size, sequence_length)

                    # Run experiment
                    result = run_single_experiment(
                        fabric,
                        model_name,
                        batch_size,
                        sequence_length,
                        config,
                        mode,
                        num_iterations,
                        warmup_iterations,
                    )

                    # Store result with mode as part of the key
                    mode_key = (*triplet, mode)
                    results["experiments"][mode_key] = result

                    # Check for OOM for this batch_size
                    if result.get("status") == "out_of_memory":
                        print(
                            f"       ⚠️  OOM detected for batch_size={batch_size}, stopping larger batch sizes"
                        )
                        oom_detected = True
                        break

                # Early stopping for OOM at sequence length level
                # If all batch sizes failed for this sequence length, skip larger sequence lengths
                successful_batches = [
                    batch_size
                    for batch_size in batch_sizes
                    if results["experiments"]
                    .get((model_name, batch_size, sequence_length, mode), {})
                    .get("status")
                    == "success"
                ]

                if not successful_batches:
                    print(
                        f"       ⚠️  All batch sizes failed for seq_len={sequence_length}, stopping larger sequence lengths"
                    )
                    break

    return results


def save_results_csv(results: dict, timestamp: str = None) -> None:
    """Save experiment results in CSV format with detailed memory metrics.

    Args:
        results: Experiment results dictionary.
        timestamp: Optional timestamp string for filenames.
    """
    import csv
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_filename = f"memory_results_{timestamp}.csv"

    # Define CSV headers with mean and std dev for each metric
    headers = [
        "model_name",
        "batch_size",
        "sequence_length",
        "mode",
        "total_params_millions",
        "total_mean_gb",
        "total_std_gb",
        "forward_mean_gb",
        "forward_std_gb",
        "backward_mean_gb",
        "backward_std_gb",
        "peak_forward_mean_gb",
        "peak_forward_std_gb",
        "peak_backward_mean_gb",
        "peak_backward_std_gb",
        "memory_per_sample_gb",
        "memory_per_token_gb",
        "status",
    ]

    csv_rows = []

    for key, experiment in results["experiments"].items():
        if experiment.get("status") == "success":
            # Calculate standard deviations from the raw readings
            import statistics

            total_std = (
                statistics.stdev(experiment["memory_readings"])
                if len(experiment["memory_readings"]) > 1
                else 0.0
            )
            forward_std = (
                statistics.stdev(experiment["forward_memory_readings"])
                if len(experiment["forward_memory_readings"]) > 1
                else 0.0
            )
            backward_std = (
                statistics.stdev(experiment["backward_memory_readings"])
                if len(experiment["backward_memory_readings"]) > 1
                else 0.0
            )
            peak_forward_std = (
                statistics.stdev(experiment["peak_forward_readings"])
                if len(experiment["peak_forward_readings"]) > 1
                else 0.0
            )
            peak_backward_std = (
                statistics.stdev(experiment["peak_backward_readings"])
                if len(experiment["peak_backward_readings"]) > 1
                else 0.0
            )

            row = [
                experiment["model_name"],
                experiment["batch_size"],
                experiment["sequence_length"],
                experiment["mode"],
                f"{experiment['total_params'] / 1e6:.1f}",
                f"{experiment['avg_memory_gb']:.3f}",
                f"{total_std:.3f}",
                f"{experiment['avg_forward_memory_gb']:.3f}",
                f"{forward_std:.3f}",
                f"{experiment['avg_backward_memory_gb']:.3f}",
                f"{backward_std:.3f}",
                f"{experiment['avg_peak_forward_memory_gb']:.3f}",
                f"{peak_forward_std:.3f}",
                f"{experiment['avg_peak_backward_memory_gb']:.3f}",
                f"{peak_backward_std:.3f}",
                f"{experiment['memory_per_sample_gb']:.3f}",
                f"{experiment['memory_per_token_gb']:.3f}",
                experiment["status"],
            ]
        else:
            # For failed experiments, fill with empty values
            row = [
                experiment["model_name"],
                experiment["batch_size"],
                experiment["sequence_length"],
                experiment["mode"],
                "",  # total_params_millions
                "",  # total_mean_gb
                "",  # total_std_gb
                "",  # forward_mean_gb
                "",  # forward_std_gb
                "",  # backward_mean_gb
                "",  # backward_std_gb
                "",  # peak_forward_mean_gb
                "",  # peak_forward_std_gb
                "",  # peak_backward_mean_gb
                "",  # peak_backward_std_gb
                "",  # memory_per_sample_gb
                "",  # memory_per_token_gb
                experiment["status"],
            ]

        csv_rows.append(row)

    # Write CSV file
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    print(f"✅ CSV results saved to: {csv_filename}")


def save_results(results: dict, timestamp: str = None) -> None:
    """Save experiment results in JSON and CSV formats.

    Args:
        results: Experiment results dictionary.
        timestamp: Optional timestamp string for filenames.
    """
    import json
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert tuple keys to strings for JSON serialization
    json_safe_results = {
        "timestamp": results["timestamp"],
        "device": results["device"],
        "experiments": {},
    }

    for key, experiment in results["experiments"].items():
        # Convert tuple key to string key
        if isinstance(key, tuple):
            key_str = "_".join(str(k) for k in key)
        else:
            key_str = str(key)
        json_safe_results["experiments"][key_str] = experiment

    # Save full results
    full_filename = f"memory_results_full_{timestamp}.json"
    with open(full_filename, "w") as f:
        json.dump(json_safe_results, f, indent=2)

    # Create simplified results
    simplified_data = {
        "timestamp": results["timestamp"],
        "device": results["device"],
        "experiments": {},
    }

    for key, experiment in results["experiments"].items():
        # Convert tuple key to string key
        if isinstance(key, tuple):
            key_str = "_".join(str(k) for k in key)
        else:
            key_str = str(key)

        if experiment.get("status") == "success":
            simplified_data["experiments"][key_str] = {
                "model_name": experiment["model_name"],
                "batch_size": experiment["batch_size"],
                "sequence_length": experiment["sequence_length"],
                "mode": experiment["mode"],
                "total_params_millions": experiment["total_params"] / 1e6,
                "avg_memory_gb": experiment["avg_memory_gb"],
                "peak_memory_gb": experiment["peak_memory_gb"],
                "avg_forward_memory_gb": experiment["avg_forward_memory_gb"],
                "peak_forward_memory_gb": experiment["avg_peak_forward_memory_gb"],
                "avg_backward_memory_gb": experiment["avg_backward_memory_gb"],
                "peak_backward_memory_gb": experiment["avg_peak_backward_memory_gb"],
                "memory_per_sample_gb": experiment["memory_per_sample_gb"],
                "memory_per_token_gb": experiment["memory_per_token_gb"],
                "status": experiment["status"],
            }
        else:
            simplified_data["experiments"][key_str] = {
                "model_name": experiment["model_name"],
                "batch_size": experiment["batch_size"],
                "sequence_length": experiment["sequence_length"],
                "mode": experiment["mode"],
                "status": experiment["status"],
                "error": experiment.get("error", ""),
            }

    # Save simplified results
    simplified_filename = f"memory_results_simplified_{timestamp}.json"
    with open(simplified_filename, "w") as f:
        json.dump(simplified_data, f, indent=2)

    # Save CSV results
    save_results_csv(results, timestamp)

    print(f"✅ Full results saved to: {full_filename}")
    print(f"✅ Simplified results saved to: {simplified_filename}")


def tuple_to_key(tuple_key: tuple) -> str:
    """Convert a tuple key to a string key for JSON serialization.

    Args:
        tuple_key: Tuple key like ('tiny', 32, 128, 'training').

    Returns:
        String key like 'tiny_32_128_training'.
    """
    return "_".join(str(k) for k in tuple_key)


def key_to_tuple(key_str: str) -> tuple:
    """Convert a string key back to a tuple key.

    Args:
        key_str: String key like 'tiny_32_128_training'.

    Returns:
        Tuple key like ('tiny', 32, 128, 'training').
    """
    parts = key_str.split("_")
    # Convert numeric parts back to integers
    result = []
    for part in parts:
        try:
            result.append(int(part))
        except ValueError:
            result.append(part)
    return tuple(result)


def get_experiments_by_model(results: dict, model_name: str) -> dict:
    """Get all experiments for a specific model.

    Args:
        results: Experiment results dictionary.
        model_name: Name of the model to filter by.

    Returns:
        Dictionary containing only experiments for the specified model.
    """
    filtered = {}
    for key, experiment in results["experiments"].items():
        if experiment["model_name"] == model_name:
            filtered[key] = experiment
    return filtered


def get_experiments_by_batch_size(results: dict, batch_size: int) -> dict:
    """Get all experiments for a specific batch size.

    Args:
        results: Experiment results dictionary.
        batch_size: Batch size to filter by.

    Returns:
        Dictionary containing only experiments for the specified batch size.
    """
    filtered = {}
    for key, experiment in results["experiments"].items():
        if experiment["batch_size"] == batch_size:
            filtered[key] = experiment
    return filtered


def get_experiments_by_sequence_length(results: dict, sequence_length: int) -> dict:
    """Get all experiments for a specific sequence length.

    Args:
        results: Experiment results dictionary.
        sequence_length: Sequence length to filter by.

    Returns:
        Dictionary containing only experiments for the specified sequence length.
    """
    filtered = {}
    for key, experiment in results["experiments"].items():
        if experiment["sequence_length"] == sequence_length:
            filtered[key] = experiment
    return filtered


def get_experiments_by_mode(results: dict, mode: str) -> dict:
    """Get all experiments for a specific mode.

    Args:
        results: Experiment results dictionary.
        mode: Mode to filter by ('training' or 'evaluation').

    Returns:
        Dictionary containing only experiments for the specified mode.
    """
    filtered = {}
    for key, experiment in results["experiments"].items():
        if experiment["mode"] == mode:
            filtered[key] = experiment
    return filtered


def measure_memory_scaling_experiments(
    fabric: Fabric,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Comprehensive memory scaling experiments using triplet-based structure.

    Args:
        fabric: Lightning Fabric instance.
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing all experiment results with triplet keys.
    """
    from datetime import datetime

    if fabric.global_rank != 0:
        return {}

    # Define test configurations
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]  # Ordered from smallest to largest
    sequence_lengths = [128, 256, 512, 1024]  # Ordered from shortest to longest
    model_configs = [
        {"n_layer": 1, "n_head": 2, "n_embd": 256, "name": "tiny256"},
        {"n_layer": 1, "n_head": 2, "n_embd": 512, "name": "tiny512"},
        {"n_layer": 2, "n_head": 4, "n_embd": 512, "name": "small512"},
        {"n_layer": 2, "n_head": 4, "n_embd": 1024, "name": "small1024"},
        {"n_layer": 4, "n_head": 8, "n_embd": 1024, "name": "medium1024"},
        {"n_layer": 4, "n_head": 8, "n_embd": 2048, "name": "medium2048"},
    ]
    modes = ["training", "evaluation"]

    # Run all experiments
    results = run_experiment_grid(
        fabric,
        model_configs,
        batch_sizes,
        sequence_lengths,
        modes,
        num_iterations,
        warmup_iterations,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, timestamp)

    return results


if __name__ == "__main__":
    fabric = Fabric(accelerator="cuda", devices=1)
    measure_memory_scaling_experiments(fabric)
