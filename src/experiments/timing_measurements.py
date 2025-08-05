"""Timing scaling experiments."""

import torch
from gpt2_standalone.lightning_module import GPTLightningModule
from gpt2_standalone.model import GPTConfig
from lightning import Fabric

from experiments.clean_palate import deep_gpu_reset, reset_model_state


def run_batch_size_experiment(
    fabric: Fabric,
    batch_size: int,
    model_configs: list,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run batch size experiment for a specific batch size across all models.

    Args:
        fabric: Lightning Fabric instance.
        batch_size: Batch size to test.
        model_configs: List of model configurations to test.
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing experiment results for this batch size across all models.
    """
    import gc
    import statistics
    import time

    print(f"   Testing batch size: {batch_size}")

    batch_results = {
        "batch_size": batch_size,
        "models": [],
        "status": "success",
    }

    for config in model_configs:
        print(f"     Testing model: {config['name']}")

        try:
            # Clean palate before starting
            deep_gpu_reset()

            # Create fresh model
            model_config = GPTConfig(
                block_size=1024,
                vocab_size=50304,
                n_layer=config["n_layer"],
                n_head=config["n_head"],
                n_embd=config["n_embd"],
                n_blocks_per_super=2,
            )

            model = GPTLightningModule(model_config)
            model, optimizer = fabric.setup(
                model, model.configure_optimizers()["optimizer"]
            )

            # Calculate model parameters
            total_params = sum(p.numel() for p in model.parameters())

            # Create test data
            x = torch.randint(0, 50304, (batch_size, 1024), device=fabric.device)
            y = torch.randint(0, 50304, (batch_size, 1024), device=fabric.device)
            test_batch = (x, y)

            # Warmup with clean palate between iterations
            model.train()
            for _ in range(warmup_iterations):
                deep_gpu_reset()
                _ = model.training_step(test_batch, batch_idx=0)

            # Synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Measure timing over multiple iterations
            timings = []
            for i in range(num_iterations):
                # Clean palate before each measurement
                deep_gpu_reset()
                reset_model_state(model, optimizer)

                # Synchronize before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.time()
                loss = model.training_step(test_batch, batch_idx=i)

                # Synchronize after timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.time()
                timings.append(end_time - start_time)

            # Calculate statistics
            avg_time = statistics.mean(timings)
            std_time = statistics.stdev(timings) if len(timings) > 1 else 0.0
            min_time = min(timings)
            max_time = max(timings)
            time_per_sample = avg_time / batch_size

            model_result = {
                "model_name": config["name"],
                "config": config,
                "total_params": total_params,
                "avg_time_sec": avg_time,
                "std_time_sec": std_time,
                "min_time_sec": min_time,
                "max_time_sec": max_time,
                "time_per_sample_sec": time_per_sample,
                "timings": timings,
                "status": "success",
            }

            print(
                f"       Params: {total_params/1e6:.1f}M, Time: {avg_time:.4f}s ± {std_time:.4f}s"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"       ❌ Out of memory for model {config['name']}")
                model_result = {
                    "model_name": config["name"],
                    "config": config,
                    "status": "out_of_memory",
                    "error": str(e),
                }
            else:
                print(f"       ❌ Runtime error for model {config['name']}: {e}")
                model_result = {
                    "model_name": config["name"],
                    "config": config,
                    "status": "runtime_error",
                    "error": str(e),
                }
        except Exception as e:
            print(f"       ❌ Unexpected error for model {config['name']}: {e}")
            model_result = {
                "model_name": config["name"],
                "config": config,
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

        batch_results["models"].append(model_result)

    return batch_results


def run_model_size_experiment(
    fabric: Fabric,
    config: dict,
    batch_sizes: list,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run model size experiment for a specific model across all batch sizes.

    Args:
        fabric: Lightning Fabric instance.
        config: Model configuration dictionary.
        batch_sizes: List of batch sizes to test (assumed to be ordered from smallest to largest).
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing experiment results for this model across all batch sizes.
    """
    import gc
    import statistics
    import time

    print(f"   Testing model: {config['name']}")

    model_results = {
        "model_name": config["name"],
        "config": config,
        "batch_sizes": [],
        "status": "success",
    }

    for batch_size in batch_sizes:
        print(f"     Testing batch size: {batch_size}")

        try:
            # Clean palate before starting
            deep_gpu_reset()

            # Create fresh model
            model_config = GPTConfig(
                block_size=1024,
                vocab_size=50304,
                n_layer=config["n_layer"],
                n_head=config["n_head"],
                n_embd=config["n_embd"],
                n_blocks_per_super=2,
            )

            model = GPTLightningModule(model_config)
            model, optimizer = fabric.setup(
                model, model.configure_optimizers()["optimizer"]
            )

            # Calculate model parameters
            total_params = sum(p.numel() for p in model.parameters())

            # Create test data
            x = torch.randint(0, 50304, (batch_size, 1024), device=fabric.device)
            y = torch.randint(0, 50304, (batch_size, 1024), device=fabric.device)
            test_batch = (x, y)

            # Warmup with clean palate between iterations
            model.train()
            for _ in range(warmup_iterations):
                deep_gpu_reset()
                _ = model.training_step(test_batch, batch_idx=0)

            # Synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Measure timing over multiple iterations
            timings = []
            forward_timings = []
            backward_timings = []

            for i in range(num_iterations):
                # Clean palate before each measurement
                deep_gpu_reset()
                reset_model_state(model, optimizer)

                # Synchronize before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Measure forward pass separately
                start_time = time.time()
                with torch.no_grad():
                    _ = model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_time = time.time() - start_time

                # Measure full training step
                start_time = time.time()
                loss = model.training_step(test_batch, batch_idx=i)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()

                total_time = end_time - start_time
                backward_time = total_time - forward_time

                timings.append(total_time)
                forward_timings.append(forward_time)
                backward_timings.append(backward_time)

            # Calculate statistics
            avg_time = statistics.mean(timings)
            std_time = statistics.stdev(timings) if len(timings) > 1 else 0.0
            avg_forward_time = statistics.mean(forward_timings)
            avg_backward_time = statistics.mean(backward_timings)
            time_per_sample = avg_time / batch_size

            batch_result = {
                "batch_size": batch_size,
                "total_params": total_params,
                "avg_time_sec": avg_time,
                "std_time_sec": std_time,
                "avg_forward_time_sec": avg_forward_time,
                "avg_backward_time_sec": avg_backward_time,
                "time_per_sample_sec": time_per_sample,
                "timings": timings,
                "forward_timings": forward_timings,
                "backward_timings": backward_timings,
                "status": "success",
            }

            print(
                f"       Time: {avg_time:.4f}s ± {std_time:.4f}s, Forward: {avg_forward_time:.4f}s, Backward: {avg_backward_time:.4f}s"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"       ❌ Out of memory for batch size {batch_size}")
                batch_result = {
                    "batch_size": batch_size,
                    "status": "out_of_memory",
                    "error": str(e),
                }
                # Stop testing larger batch sizes since they will also fail
                print(
                    f"       ⚠️  Stopping batch size tests for model {config['name']} (will fail for larger batches)"
                )
                model_results["batch_sizes"].append(batch_result)
                break
            else:
                print(f"       ❌ Runtime error for batch size {batch_size}: {e}")
                batch_result = {
                    "batch_size": batch_size,
                    "status": "runtime_error",
                    "error": str(e),
                }
        except Exception as e:
            print(f"       ❌ Unexpected error for batch size {batch_size}: {e}")
            batch_result = {
                "batch_size": batch_size,
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

        model_results["batch_sizes"].append(batch_result)

    return model_results


def measure_timing_scaling_experiments(
    fabric: Fabric,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Comprehensive timing scaling experiments.

    Args:
        fabric: Lightning Fabric instance.
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing all experiment results.
    """
    import json
    from datetime import datetime

    if fabric.global_rank != 0:
        return {}

    print(f"\n⏱️  Starting Timing Scaling Experiments")
    print(f"   Warmup iterations: {warmup_iterations}")
    print(f"   Measurement iterations: {num_iterations}")

    # Define test configurations
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]  # Ordered from smallest to largest
    model_configs = [
        {"n_layer": 1, "n_head": 2, "n_embd": 256, "name": "tiny"},
        {"n_layer": 2, "n_head": 4, "n_embd": 512, "name": "small"},
        {"n_layer": 4, "n_head": 4, "n_embd": 1024, "name": "medium"},
        {"n_layer": 6, "n_head": 6, "n_embd": 1024, "name": "large"},
        {"n_layer": 8, "n_head": 8, "n_embd": 1024, "name": "xlarge"},
    ]

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(fabric.device),
        "batch_size_experiment": [],
        "model_size_experiment": [],
    }

    # ----------------------------------------------------------------------
    # Experiment 1: Batch Size vs Timing (for each batch size, test all models)
    print(f"\n==> 📊 Experiment 1: Batch Size vs Timing")
    print(f"   Testing each batch size across all models")

    for batch_size in batch_sizes:
        batch_result = run_batch_size_experiment(
            fabric, batch_size, model_configs, num_iterations, warmup_iterations
        )
        results["batch_size_experiment"].append(batch_result)

        # Check if all models failed for this batch size
        successful_models = [
            m for m in batch_result["models"] if m.get("status") == "success"
        ]
        if not successful_models:
            print(
                f"     ⚠️  All models failed for batch size {batch_size}, stopping batch size experiments"
            )
            break

    # ----------------------------------------------------------------------
    # Experiment 2: Model Size vs Timing (for each model, test all batch sizes)
    print(f"\n==> 📊 Experiment 2: Model Size vs Timing")
    print(
        f"   Testing each model across all batch sizes (ordered from smallest to largest)"
    )

    for config in model_configs:
        model_result = run_model_size_experiment(
            fabric, config, batch_sizes, num_iterations, warmup_iterations
        )
        results["model_size_experiment"].append(model_result)

        # No need to check if all batch sizes failed - the function will stop early
        # when it hits the first memory limit

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"timing_scaling_experiment_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {filename}")

    return results


if __name__ == "__main__":
    fabric = Fabric(accelerator="cuda", devices=1)
    measure_timing_scaling_experiments(fabric)
