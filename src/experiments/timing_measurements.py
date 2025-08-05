"""Timing scaling experiments."""

import torch
from gpt2_standalone.lightning_module import GPTLightningModule
from gpt2_standalone.model import GPTConfig
from lightning import Fabric


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
    import gc
    import json
    import statistics
    import time
    from datetime import datetime

    if fabric.global_rank != 0:
        return {}

    print(f"\n⏱️  Starting Timing Scaling Experiments")
    print(f"   Warmup iterations: {warmup_iterations}")
    print(f"   Measurement iterations: {num_iterations}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(fabric.device),
        "batch_size_experiment": [],
        "model_size_experiment": [],
        "precision_experiment": [],
    }

    # ----------------------------------------------------------------------
    # Experiment 1: Batch Size vs Timing
    print(f"\n==> 📊 Experiment 1: Batch Size vs Timing")
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    for batch_size in batch_sizes:
        print(f"   Testing batch size: {batch_size}")

        # Create fresh model
        config = GPTConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=4,
            n_head=4,
            n_embd=1024,
            n_blocks_per_super=2,
        )

        model = GPTLightningModule(config)
        model, optimizer = fabric.setup(
            model, model.configure_optimizers()["optimizer"]
        )

        # Create test data
        x = torch.randint(0, 50304, (batch_size, 1024), device=fabric.device)
        y = torch.randint(0, 50304, (batch_size, 1024), device=fabric.device)
        test_batch = (x, y)

        # Warmup
        model.train()
        for _ in range(warmup_iterations):
            _ = model.training_step(test_batch, batch_idx=0)

        # Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure timing over multiple iterations
        timings = []
        for i in range(num_iterations):
            optimizer.zero_grad()

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

        results["batch_size_experiment"].append(
            {
                "batch_size": batch_size,
                "avg_time_sec": avg_time,
                "std_time_sec": std_time,
                "min_time_sec": min_time,
                "max_time_sec": max_time,
                "time_per_sample_sec": time_per_sample,
                "timings": timings,
            }
        )

        print(f"     Avg Time: {avg_time:.4f}s ± {std_time:.4f}s")

        # Clean up
        del model, optimizer, x, y, test_batch
        torch.cuda.empty_cache()
        gc.collect()

    # ----------------------------------------------------------------------
    # Experiment 2: Model Size vs Timing
    print(f"\n==> 📊 Experiment 2: Model Size vs Timing")
    model_configs = [
        {"n_layer": 1, "n_head": 2, "n_embd": 256, "name": "tiny"},
        {"n_layer": 2, "n_head": 4, "n_embd": 512, "name": "small"},
        {"n_layer": 4, "n_head": 4, "n_embd": 1024, "name": "medium"},
        {"n_layer": 6, "n_head": 6, "n_embd": 1024, "name": "large"},
        {"n_layer": 8, "n_head": 8, "n_embd": 1024, "name": "xlarge"},
    ]

    batch_size = 32  # Fixed batch size

    for config in model_configs:
        print(f"   Testing model: {config['name']}")

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

        # Warmup
        model.train()
        for _ in range(warmup_iterations):
            _ = model.training_step(test_batch, batch_idx=0)

        # Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure timing over multiple iterations
        timings = []
        for i in range(num_iterations):
            optimizer.zero_grad()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.time()
            loss = model.training_step(test_batch, batch_idx=i)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            timings.append(end_time - start_time)

        # Calculate statistics
        avg_time = statistics.mean(timings)
        std_time = statistics.stdev(timings) if len(timings) > 1 else 0.0
        time_per_param = avg_time / (total_params / 1e6)  # seconds per million params

        results["model_size_experiment"].append(
            {
                "model_name": config["name"],
                "config": config,
                "total_params": total_params,
                "avg_time_sec": avg_time,
                "std_time_sec": std_time,
                "time_per_param_sec": time_per_param,
                "timings": timings,
            }
        )

        print(
            f"     Params: {total_params/1e6:.1f}M, Time: {avg_time:.4f}s ± {std_time:.4f}s"
        )

        # Clean up
        del model, optimizer, x, y, test_batch
        torch.cuda.empty_cache()
        gc.collect()

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
