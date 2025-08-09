"""Timing measurement experiments with triplet-based structure."""

import csv
import time
from collections.abc import Callable
from contextlib import suppress

import torch
from gpt2_standalone.lightning_module import GPTLightningModule
from gpt2_standalone.model import GPTConfig
from lightning import Fabric

from experiments.clean_palate import deep_gpu_reset, reset_model_state


def timing_measurement(func):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            stream = torch.cuda.current_stream()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream.synchronize()  # clear prior work on this stream
            start.record(stream)
            result = func(*args, **kwargs)
            end.record(stream)
            end.synchronize()  # wait only for this stream
            ms = start.elapsed_time(end)  # milliseconds
            timing = {"execution_time_ms": ms, "execution_time_s": ms / 1000.0}
        else:
            import time

            t0 = time.time()
            result = func(*args, **kwargs)
            timing = {
                "execution_time_ms": (time.time() - t0) * 1000,
                "execution_time_s": time.time() - t0,
            }
        if isinstance(result, dict):
            result.update(timing)
        else:
            result = {"function_result": result, **timing}
        return result

    return wrapper


@timing_measurement
def run_inference(model: GPTLightningModule, x: torch.Tensor) -> dict:
    """Run inference (forward pass without gradients) and measure execution time."""
    # Run forward pass without gradients
    with torch.no_grad():
        output = model(x)

    # Don't delete output - let the decorator handle cleanup
    return {"operation": "inference"}


@timing_measurement
def run_forward_with_gradients(
    model: GPTLightningModule, x: torch.Tensor, y: torch.Tensor
) -> dict:
    """Run forward pass with gradients enabled and measure execution time."""
    # Run forward pass with gradients enabled (no loss computation)
    logits, _ = model(
        x, y
    )  # This creates the computation graph but doesn't compute loss

    # Don't delete logits - let the decorator handle cleanup
    return {"operation": "forward_with_gradients"}


@timing_measurement
def run_training_step(
    model: GPTLightningModule, x: torch.Tensor, y: torch.Tensor
) -> dict:
    """Run complete training step and measure execution time."""
    # Run complete training step (forward + loss + backward)
    loss = model.training_step((x, y), batch_idx=0)

    # Don't delete loss - let the decorator handle cleanup
    return {"operation": "training_step"}


def calculate_params_from_config(model_config: dict, sequence_length: int) -> int:
    """Calculate total parameters from model configuration.

    Args:
        model_config: Model configuration dictionary.
        sequence_length: Sequence length.

    Returns:
        Total number of parameters.
    """
    # Create config object to calculate parameters
    config_obj = GPTConfig(
        block_size=sequence_length,
        vocab_size=50304,
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
        n_embd=model_config["n_embd"],
        n_blocks_per_super=2,
    )

    # Create temporary model to count parameters
    temp_model = GPTLightningModule(config_obj)
    total_params = sum(p.numel() for p in temp_model.parameters())
    del temp_model  # Clean up

    return total_params


def run_single_experiment(
    fabric: Fabric,
    model_name: str,
    batch_size: int,
    sequence_length: int,
    model_config: dict,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run a single timing experiment for a specific (model, batch_size, sequence_length) triplet.

    Args:
        fabric: Lightning Fabric instance.
        model_name: Name of the model (e.g., 'tiny', 'small').
        batch_size: Batch size to test.
        sequence_length: Sequence length to test.
        model_config: Model configuration dictionary.
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing timing experiment results for this specific triplet.
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

        # Measure timing over multiple iterations
        # inf:inference, fwd:forward pass, ts:training step
        print(f"     Measuring timing ({num_iterations} iterations)...", flush=True)
        inf_times_ms = []
        inf_times_s = []
        fwd_times_ms = []
        fwd_times_s = []
        ts_times_ms = []
        ts_times_s = []

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

            # Store timing readings
            inf_times_ms.append(inf_result["execution_time_ms"])
            inf_times_s.append(inf_result["execution_time_s"])
            fwd_times_ms.append(fwd_result["execution_time_ms"])
            fwd_times_s.append(fwd_result["execution_time_s"])
            ts_times_ms.append(ts_result["execution_time_ms"])
            ts_times_s.append(ts_result["execution_time_s"])

        # Calculate statistics using the correct lists populated above
        import statistics

        avg_inf_time_ms = statistics.mean(inf_times_ms)
        avg_inf_time_s = statistics.mean(inf_times_s)
        avg_fwd_time_ms = statistics.mean(fwd_times_ms)
        avg_fwd_time_s = statistics.mean(fwd_times_s)
        avg_ts_time_ms = statistics.mean(ts_times_ms)
        avg_ts_time_s = statistics.mean(ts_times_s)

        # replace the result dict assembly to add raw samples
        result = {
            "model_name": model_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "config": model_config,
            "total_params": total_params,
            "total_params_millions": total_params / 1e6,
            "avg_inf_time_ms": avg_inf_time_ms,
            "avg_inf_time_s": avg_inf_time_s,
            "avg_fwd_time_ms": avg_fwd_time_ms,
            "avg_fwd_time_s": avg_fwd_time_s,
            "avg_ts_time_ms": avg_ts_time_ms,
            "avg_ts_time_s": avg_ts_time_s,
            "samples": {
                "INF_time_ms": inf_times_ms,
                "FWD_time_ms": fwd_times_ms,
                "TS_time_ms": ts_times_ms,
            },
            "status": "success",
        }

        elapsed_time = time.time() - start_time
        print(
            f"       ✅ Completed in {elapsed_time:.1f}s - "
            f"    (time ms, time s) - "
            f"INF: {avg_inf_time_ms:.2f}ms, {avg_inf_time_s:.4f}s, "
            f"FWD: {avg_fwd_time_ms:.2f}ms, {avg_fwd_time_s:.4f}s, "
            f"TS: {avg_ts_time_ms:.2f}ms, {avg_ts_time_s:.4f}s",
            flush=True,
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            elapsed_time = time.time() - start_time
            total_params = calculate_params_from_config(model_config, sequence_length)
            result = {
                "model_name": model_name,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "config": model_config,
                "total_params": total_params,
                "total_params_millions": total_params / 1e6,
                "status": "out_of_memory",
                "error": str(e),
            }
        else:
            elapsed_time = time.time() - start_time
            print(
                f"       ❌ Runtime error for {model_name}, batch_size={batch_size}, seq_len={sequence_length} (after {elapsed_time:.1f}s): {e}",
                flush=True,
            )
            total_params = calculate_params_from_config(model_config, sequence_length)
            result = {
                "model_name": model_name,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "config": model_config,
                "total_params": total_params,
                "total_params_millions": total_params / 1e6,
                "status": "runtime_error",
                "error": str(e),
            }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(
            f"       ❌ Unexpected error for {model_name}, batch_size={batch_size}, seq_len={sequence_length} (after {elapsed_time:.1f}s): {e}"
        )
        total_params = calculate_params_from_config(model_config, sequence_length)
        result = {
            "model_name": model_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "config": model_config,
            "total_params": total_params,
            "total_params_millions": total_params / 1e6,
            "status": "error",
            "error": str(e),
        }
    finally:
        # Clean up
        with suppress(NameError):
            del model
        with suppress(NameError):
            del optimizer
        with suppress(NameError):
            del x
        with suppress(NameError):
            del y
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

    print(f"\n🧪 Starting Timing Scaling Experiments")
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
    """Save experiment results to CSV file.

    Args:
        results: Experiment results dictionary.
        timestamp: Timestamp string for filename. If None, current timestamp is used.
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV headers
    headers = [
        "model_name",
        "batch_size",
        "sequence_length",
        "total_params_millions",
        "INF_time_ms",
        "INF_time_s",
        "FWD_time_ms",
        "FWD_time_s",
        "TS_time_ms",
        "TS_time_s",
        "status",
    ]

    rows = [headers]

    for experiment in results["experiments"].values():
        if experiment.get("status") == "success":
            row = [
                experiment["model_name"],
                experiment["batch_size"],
                experiment["sequence_length"],
                f"{experiment['total_params_millions']:.1f}",
                f"{experiment['avg_inf_time_ms']:.3f}",
                f"{experiment['avg_inf_time_s']:.6f}",
                f"{experiment['avg_fwd_time_ms']:.3f}",
                f"{experiment['avg_fwd_time_s']:.6f}",
                f"{experiment['avg_ts_time_ms']:.3f}",
                f"{experiment['avg_ts_time_s']:.6f}",
                experiment["status"],
            ]
        else:
            total_m = experiment.get(
                "total_params_millions",
                experiment.get("total_params", 0) / 1e6,
            )
            row = [
                experiment["model_name"],
                experiment["batch_size"],
                experiment["sequence_length"],
                f"{total_m:.1f}",
                "",  # INF_time_ms
                "",  # INF_time_s
                "",  # FWD_time_ms
                "",  # FWD_time_s
                "",  # TS_time_ms
                "",  # TS_time_s
                experiment["status"],
            ]
        rows.append(row)

    # Write CSV file
    filename = f"timing_results_{timestamp}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"✅ CSV results saved to: {filename}")


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
        key_str = "_".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        json_safe_results["experiments"][key_str] = experiment

    # Save full results
    full_filename = f"timing_results_full_{timestamp}.json"
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
        key_str = "_".join(str(k) for k in key) if isinstance(key, tuple) else str(key)

        if experiment.get("status") == "success":
            simplified_data["experiments"][key_str] = {
                "model_name": experiment["model_name"],
                "batch_size": experiment["batch_size"],
                "sequence_length": experiment["sequence_length"],
                "total_params_millions": experiment["total_params_millions"],
                "INF_time_ms": experiment["avg_inf_time_ms"],
                "INF_time_s": experiment["avg_inf_time_s"],
                "FWD_time_ms": experiment["avg_fwd_time_ms"],
                "FWD_time_s": experiment["avg_fwd_time_s"],
                "TS_time_ms": experiment["avg_ts_time_ms"],
                "TS_time_s": experiment["avg_ts_time_s"],
                "status": experiment["status"],
            }
        else:
            simplified_data["experiments"][key_str] = {
                "model_name": experiment["model_name"],
                "batch_size": experiment["batch_size"],
                "sequence_length": experiment["sequence_length"],
                "total_params_millions": experiment.get(
                    "total_params_millions",
                    experiment.get("total_params", 0) / 1e6,
                ),
                "config": experiment["config"],
                "status": experiment["status"],
                "error": experiment.get("error", ""),
            }

    # Save simplified results
    simplified_filename = f"timing_results_simplified_{timestamp}.json"
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
    return tuple(int(part) if part.isdigit() else part for part in parts)


def get_experiments_by_model(results: dict, model_name: str) -> dict:
    """Get all experiments for a specific model.

    Args:
        results: Experiment results dictionary.
        model_name: Name of the model to filter by.

    Returns:
        Dictionary containing only experiments for the specified model.
    """
    return {
        key: experiment
        for key, experiment in results["experiments"].items()
        if experiment["model_name"] == model_name
    }


def get_experiments_by_batch_size(results: dict, batch_size: int) -> dict:
    """Get all experiments for a specific batch size.

    Args:
        results: Experiment results dictionary.
        batch_size: Batch size to filter by.

    Returns:
        Dictionary containing only experiments for the specified batch size.
    """
    return {
        key: experiment
        for key, experiment in results["experiments"].items()
        if experiment["batch_size"] == batch_size
    }


def get_experiments_by_sequence_length(results: dict, sequence_length: int) -> dict:
    """Get all experiments for a specific sequence length.

    Args:
        results: Experiment results dictionary.
        sequence_length: Sequence length to filter by.

    Returns:
        Dictionary containing only experiments for the specified sequence length.
    """
    return {
        key: experiment
        for key, experiment in results["experiments"].items()
        if experiment["sequence_length"] == sequence_length
    }


def get_experiments_by_mode(results: dict, mode: str) -> dict:
    """Get all experiments for a specific mode.

    Args:
        results: Experiment results dictionary.
        mode: Mode to filter by ('training' or 'evaluation').

    Returns:
        Dictionary containing only experiments for the specified mode.
    """
    return {
        key: experiment
        for key, experiment in results["experiments"].items()
        if experiment["mode"] == mode
    }


def measure_timing_scaling_experiments(
    fabric: Fabric,
    num_iterations: int = 5,
    warmup_iterations: int = 2,
) -> dict:
    """Run timing scaling experiments for different model configurations.

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
    batch_sizes = [1, 16, 32, 64, 128]  # Ordered from smallest to largest
    # batch_sizes = [1, 128]  # Ordered from smallest to largest
    # sequence_lengths = [128, 256, 512, 1024]  # Ordered from shortest to longest
    sequence_lengths = [256, 512, 1024, 2048]  # Ordered from shortest to longest
    # sequence_lengths = [256, 2048]  # Ordered from shortest to longest
    model_configs = [
        {"n_layer": 1, "n_head": 2, "n_embd": 256, "name": "tiny256"},
        {"n_layer": 2, "n_head": 4, "n_embd": 512, "name": "small512"},
        {"n_layer": 4, "n_head": 8, "n_embd": 1024, "name": "medium1024"},
        {"n_layer": 8, "n_head": 16, "n_embd": 2048, "name": "large2048"},
    ]

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
    measure_timing_scaling_experiments(fabric)
#
