"""Memory measurement experiments with triplet-based structure."""

import torch
from gpt2_standalone.lightning_module import GPTLightningModule
from gpt2_standalone.model import GPTConfig
from lightning import Fabric

from experiments.clean_palate import deep_gpu_reset, reset_model_state


def memory_measurement(func):
    """Measure GPU memory usage for any function via decorator.

    This decorator handles the common pattern of:
    1. Reset memory stats
    2. Measure start memory
    3. Call the function
    4. Measure end memory
    5. Calculate net allocation
    6. Clean up

    Args:
        func: Function to measure memory for.

    Returns:
        Wrapped function that returns memory measurements.
    """

    def wrapper(*args, **kwargs):
        """Reset memory stats before function call."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

        # Call the original function
        result = func(*args, **kwargs)

        # Measure memory after function call
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        # Calculate net memory allocation
        net_mem = end_mem - start_mem

        # Clean up
        torch.cuda.empty_cache()

        # Return standardized memory measurements
        memory_measurements = {
            "memory_gb": end_mem / 1e9,
            "net_memory_gb": net_mem / 1e9,
            "peak_memory_gb": peak_mem / 1e9,
        }

        # Combine original result with memory measurements
        if isinstance(result, dict):
            result.update(memory_measurements)
        else:
            result = {"function_result": result, **memory_measurements}

        return result

    return wrapper


@memory_measurement
def run_inference(model, x):
    """Run inference (forward pass without gradients) and measure GPU memory usage.

    Args:
        model: The model to run inference on.
        x: Input tensor.

    Returns:
        dict: Memory measurements for inference.
    """
    # Run forward pass without gradients
    with torch.no_grad():
        output = model(x)

    # Clean up
    del output

    return {"operation": "inference"}


@memory_measurement
def run_forward_with_gradients(model, x, y):
    """Run forward pass with gradients enabled and measure GPU memory usage.

    This function runs the forward pass with gradients enabled but does NOT
    compute loss or run backward pass. It's used to measure the memory
    required for the forward pass when preparing for backpropagation.

    Args:
        model: The model to run forward pass on.
        x: Input tensor.
        y: Target tensor.

    Returns:
        dict: Memory measurements for forward pass with gradients.
    """
    # Run forward pass with gradients enabled (no loss computation)
    logits, _ = model(
        x, y
    )  # This creates the computation graph but doesn't compute loss

    # Clean up
    del logits

    return {"operation": "forward_with_gradients"}


@memory_measurement
def run_training_step(model, x, y):
    """Run complete training step and measure GPU memory usage.

    This function runs the full training step: forward pass, loss computation,
    backward pass, and optimizer step.

    Args:
        model: The model to run training step on.
        x: Input tensor.
        y: Target tensor.

    Returns:
        dict: Memory measurements for the complete training step.
    """
    # Run complete training step (forward + loss + backward)
    loss = model.training_step((x, y), batch_idx=0)

    # Clean up
    del loss

    return {"operation": "training_step"}


def run_single_experiment(
    fabric: Fabric,
    model_name: str,
    batch_size: int,
    sequence_length: int,
    model_config: dict,
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
        f"   Testing: {model_name}, batch_size={batch_size}, seq_len={sequence_length}"
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

        # Set model mode
        model.train()

        # Warmup
        print(f"     Warming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            deep_gpu_reset()
            run_forward_with_gradients(model, x, y)

        # Measure memory over multiple iterations
        print(f"     Measuring memory ({num_iterations} iterations)...", flush=True)
        memory_readings = []
        inf_memory_readings = []
        fwd_memory_readings = []
        ts_memory_readings = []
        peak_inf_readings = []
        peak_fwd_readings = []
        peak_ts_readings = []

        # Just use one list of complete experiment results:
        experiment_results = []

        for _ in range(num_iterations):
            # Clean palate before each measurement
            deep_gpu_reset()
            reset_model_state(model, optimizer)
            optimizer.zero_grad()

            # Run all three measurements for every experiment
            inf_result = run_inference(model, x)
            fwd_result = run_forward_with_gradients(model, x, y)
            ts_result = run_training_step(model, x, y)

            # Create unified structure
            experiment_result = {
                "inf": inf_result,
                "fwd": fwd_result,
                "ts": ts_result,
            }
            experiment_results.append(experiment_result)

            # Store memory readings
            inf_memory_readings.append(inf_result["memory_gb"])
            fwd_memory_readings.append(fwd_result["memory_gb"])
            ts_memory_readings.append(ts_result["memory_gb"])
            peak_inf_readings.append(inf_result["peak_memory_gb"])
            peak_fwd_readings.append(fwd_result["peak_memory_gb"])
            peak_ts_readings.append(ts_result["peak_memory_gb"])

        # Calculate statistics
        avg_inf_memory = statistics.mean(inf_memory_readings)
        avg_inf_net_memory = statistics.mean(
            [r["inf"]["net_memory_gb"] for r in experiment_results]
        )
        avg_inf_peak_memory = statistics.mean(peak_inf_readings)

        avg_fwd_memory = statistics.mean(fwd_memory_readings)
        avg_fwd_net_memory = statistics.mean(
            [r["fwd"]["net_memory_gb"] for r in experiment_results]
        )
        avg_fwd_peak_memory = statistics.mean(peak_fwd_readings)

        avg_ts_memory = statistics.mean(ts_memory_readings)
        avg_ts_net_memory = statistics.mean(
            [r["ts"]["net_memory_gb"] for r in experiment_results]
        )
        avg_ts_peak_memory = statistics.mean(peak_ts_readings)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        result = {
            "model_name": model_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "config": model_config,
            "total_params": total_params,
            "avg_inf_memory_gb": avg_inf_memory,
            "avg_inf_net_memory_gb": avg_inf_net_memory,
            "avg_inf_peak_memory_gb": avg_inf_peak_memory,
            "avg_fwd_memory_gb": avg_fwd_memory,
            "avg_fwd_net_memory_gb": avg_fwd_net_memory,
            "avg_fwd_peak_memory_gb": avg_fwd_peak_memory,
            "avg_ts_memory_gb": avg_ts_memory,
            "avg_ts_net_memory_gb": avg_ts_net_memory,
            "avg_ts_peak_memory_gb": avg_ts_peak_memory,
            "peak_memory_gb": peak_memory,
            "inf_memory_readings": inf_memory_readings,
            "fwd_memory_readings": fwd_memory_readings,
            "ts_memory_readings": ts_memory_readings,
            "peak_inf_readings": peak_inf_readings,
            "peak_fwd_readings": peak_fwd_readings,
            "peak_ts_readings": peak_ts_readings,
            "status": "success",
        }

        elapsed_time = time.time() - start_time
        print(
            f"       ✅ Completed in {elapsed_time:.1f}s - "
            f"    (mem, net mem, peak mem) - "
            f"INF: {avg_inf_memory:.2f}GB, {avg_inf_net_memory:.2f}GB, {avg_inf_peak_memory:.2f}GB, "
            f"FWD: {avg_fwd_memory:.2f}GB, {avg_fwd_net_memory:.2f}GB, {avg_fwd_peak_memory:.2f}GB, "
            f"TS: {avg_ts_memory:.2f}GB, {avg_ts_net_memory:.2f}GB, {avg_ts_peak_memory:.2f}GB",
            flush=True,
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
                "config": model_config,
                "status": "out_of_memory",
                "error": str(e),
            }
        else:
            elapsed_time = time.time() - start_time
            print(
                f"       ❌ Runtime error for {model_name}, batch_size={batch_size}, seq_len={sequence_length} (after {elapsed_time:.1f}s): {e}",
                flush=True,
            )
            result = {
                "model_name": model_name,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
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
            "config": model_config,
            "status": "error",
            "error": str(e),
        }
    finally:
        # Clean up
        try:
            del model, optimizer, x, y
        except NameError:
            pass  # Variables might not exist if error occurred early
        deep_gpu_reset()

    return result


def run_experiment_grid(
    fabric: Fabric,
    model_configs: list,
    batch_sizes: list,
    sequence_lengths: list,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
    max_experiments: int | None = None,
) -> dict:
    """Run experiments for all combinations of (model, batch_size, sequence_length) triplets.

    Args:
        fabric: Lightning Fabric instance.
        model_configs: List of model configuration dictionaries.
        batch_sizes: List of batch sizes to test (ordered from smallest to largest).
        sequence_lengths: List of sequence lengths to test (ordered from shortest to longest).
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.
        max_experiments: Maximum number of experiments to run. If None, all combinations are run.

    Returns:
        Dictionary with triplet keys and experiment results as values.
    """
    from datetime import datetime

    if max_experiments is None:
        max_experiments = len(model_configs) * len(batch_sizes) * len(sequence_lengths)

    if fabric.global_rank != 0:
        return {}

    print(f"\n🧪 Starting Memory Scaling Experiments")
    print(f"   Models: {[config['name'] for config in model_configs]}")
    print(f"   Batch sizes: {batch_sizes}")
    print(f"   Sequence lengths: {sequence_lengths}")
    print(f"   Warmup iterations: {warmup_iterations}")
    print(f"   Measurement iterations: {num_iterations}", flush=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(fabric.device),
        "experiments": {},
    }

    total_experiments = len(model_configs) * len(batch_sizes) * len(sequence_lengths)
    experiment_count = 0

    for model_config in model_configs:
        model_name = model_config["name"]
        print(f"\n🔬 Testing model: {model_name}")

        for sequence_length in sequence_lengths:
            print(f"   Testing sequence length: {sequence_length}")

            for batch_size in batch_sizes:
                experiment_count += 1
                if max_experiments and experiment_count > max_experiments:
                    print(f"   ⏹️  Reached max experiments limit ({max_experiments})")
                    return results

                print(
                    f"     [{experiment_count}/{total_experiments}] Testing: {model_name}, batch_size={batch_size}, seq_len={sequence_length}"
                )

                # Run experiment (no mode parameter needed)
                experiment_result = run_single_experiment(
                    fabric=fabric,
                    model_name=model_name,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    model_config=model_config,
                    num_iterations=num_iterations,
                    warmup_iterations=warmup_iterations,
                )

                # Store result with tuple key
                key = (model_name, batch_size, sequence_length)
                results["experiments"][key] = experiment_result

                # Check if we should stop for this sequence length
                if experiment_result.get("status") != "success":
                    print(
                        f"       ⚠️  All batch sizes failed for seq_len={sequence_length}, stopping larger sequence lengths"
                    )
                    break

    return results


def save_results_csv(results: dict, timestamp: str | None = None) -> None:
    """Save experiment results in CSV format with detailed memory metrics.

    Args:
        results: Experiment results dictionary.
        timestamp: Optional timestamp string for filenames.
    """
    import csv
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV headers - update to use new naming scheme
    headers = [
        "model_name",
        "batch_size",
        "sequence_length",
        "total_params_millions",
        "INF_mem",
        "INF_net_mem",
        "INF_peak_mem",
        "FWD_mem",
        "FWD_net_mem",
        "FWD_peak_mem",
        "TS_mem",
        "TS_net_mem",
        "TS_peak_mem",
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
                f"{experiment['total_params'] / 1e6:.1f}",
                f"{experiment['avg_inf_memory_gb']:.3f}",
                f"{experiment['avg_inf_net_memory_gb']:.3f}",
                f"{experiment['avg_inf_peak_memory_gb']:.3f}",
                f"{experiment['avg_fwd_memory_gb']:.3f}",
                f"{experiment['avg_fwd_net_memory_gb']:.3f}",
                f"{experiment['avg_fwd_peak_memory_gb']:.3f}",
                f"{experiment['avg_ts_memory_gb']:.3f}",
                f"{experiment['avg_ts_net_memory_gb']:.3f}",
                f"{experiment['avg_ts_peak_memory_gb']:.3f}",
                experiment["status"],
            ]
        else:
            # For failed experiments, fill with empty values
            row = [
                experiment["model_name"],
                experiment["batch_size"],
                experiment["sequence_length"],
                f"{experiment['total_params'] / 1e6:.1f}",
                "",  # INF_mem
                "",  # INF_net_mem
                "",  # INF_peak_mem
                "",  # FWD_mem
                "",  # FWD_net_mem
                "",  # FWD_peak_mem
                "",  # TS_mem
                "",  # TS_net_mem
                "",  # TS_peak_mem
                experiment["status"],
            ]

        csv_rows.append(row)

    # Write CSV file
    csv_filename = f"memory_results_{timestamp}.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    print(f"✅ CSV results saved to: {csv_filename}")


def save_results(results: dict, timestamp: str | None = None) -> None:
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
                "total_params_millions": experiment["total_params"] / 1e6,
                "INF_mem": experiment["avg_inf_memory_gb"],
                "INF_net_mem": experiment["avg_inf_net_memory_gb"],
                "INF_peak_mem": experiment["avg_inf_peak_memory_gb"],
                "FWD_mem": experiment["avg_fwd_memory_gb"],
                "FWD_net_mem": experiment["avg_fwd_net_memory_gb"],
                "FWD_peak_mem": experiment["avg_fwd_peak_memory_gb"],
                "TS_mem": experiment["avg_ts_memory_gb"],
                "TS_net_mem": experiment["avg_ts_net_memory_gb"],
                "TS_peak_mem": experiment["avg_ts_peak_memory_gb"],
                "status": experiment["status"],
            }
        else:
            simplified_data["experiments"][key_str] = {
                "model_name": experiment["model_name"],
                "batch_size": experiment["batch_size"],
                "sequence_length": experiment["sequence_length"],
                "config": experiment["config"],
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
    """Run memory scaling experiments for different model configurations.

    Args:
        fabric: Lightning Fabric instance.
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing all experiment results.
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

    # batch_sizes = [1, 8, 32]  # Ordered from smallest to largest
    batch_sizes = [1, 32]  # Ordered from smallest to largest
    # sequence_lengths = [128, 256, 512, 1024]  # Ordered from shortest to longest
    sequence_lengths = [256, 1024]  # Ordered from shortest to longest
    model_configs = [
        {"n_layer": 1, "n_head": 2, "n_embd": 256, "name": "tiny256"},
        # {"n_layer": 2, "n_head": 4, "n_embd": 512, "name": "small512"},
        {"n_layer": 2, "n_head": 4, "n_embd": 1024, "name": "small1024"},
        # {"n_layer": 4, "n_head": 8, "n_embd": 2048, "name": "medium2048"},
    ]

    modes = ["training", "evaluation"]

    # Run experiments (no modes parameter)
    results = run_experiment_grid(
        fabric=fabric,
        model_configs=model_configs,
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, timestamp)

    return results


if __name__ == "__main__":
    fabric = Fabric(accelerator="cuda", devices=1)
    measure_memory_scaling_experiments(fabric)
