"""Memory measurement experiments."""

import torch
from gpt2_standalone.lightning_module import GPTLightningModule
from gpt2_standalone.model import GPTConfig
from lightning import Fabric


def measure_memory_scaling_experiments(
    fabric: Fabric,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Comprehensive memory scaling experiments.

    Args:
        fabric: Lightning Fabric instance.
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing all experiment results.
    """
    import gc
    import json
    from datetime import datetime

    if fabric.global_rank != 0:
        return {}

    print(f"\n🧪 Starting Memory Scaling Experiments")
    print(f"   Warmup iterations: {warmup_iterations}")
    print(f"   Measurement iterations: {num_iterations}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(fabric.device),
        "batch_size_experiment": [],
        "model_size_experiment": [],
        "precision_experiment": [],
    }

    # Experiment 1: Batch Size vs Memory
    print(f"\n📊 Experiment 1: Batch Size vs Memory")
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    for batch_size in batch_sizes:
        print(f"   Testing batch size: {batch_size}")

        # Clear memory completely
        torch.cuda.empty_cache()
        gc.collect()

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

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

        # Measure memory over multiple iterations
        memory_readings = []
        for i in range(num_iterations):
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            loss = model.training_step(test_batch, batch_idx=i)

            # Measure memory
            current_mem = torch.cuda.memory_allocated()
            memory_readings.append(current_mem / 1e9)  # Convert to GB

        # Calculate statistics
        avg_memory = sum(memory_readings) / len(memory_readings)
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        memory_per_sample = avg_memory / batch_size

        results["batch_size_experiment"].append(
            {
                "batch_size": batch_size,
                "avg_memory_gb": avg_memory,
                "peak_memory_gb": peak_memory,
                "memory_per_sample_gb": memory_per_sample,
                "memory_readings": memory_readings,
            }
        )

        print(f"     Avg Memory: {avg_memory:.2f} GB, Peak: {peak_memory:.2f} GB")

        # Clean up
        del model, optimizer, x, y, test_batch
        torch.cuda.empty_cache()
        gc.collect()

    # Experiment 2: Model Size vs Memory
    print(f"\n📊 Experiment 2: Model Size vs Memory")
    model_configs = [
        {"n_layer": 1, "n_head": 2, "n_embd": 256, "name": "tiny"},
        {"n_layer": 2, "n_head": 4, "n_embd": 512, "name": "small"},
        {"n_layer": 4, "n_head": 4, "n_embd": 1024, "name": "medium"},
        {"n_layer": 6, "n_head": 6, "n_embd": 1024, "name": "large"},
        {"n_layer": 8, "n_head": 8, "n_embd": 1024, "name": "xlarge"},
        {"n_layer": 12, "n_head": 12, "n_embd": 768, "name": "gpt2-small"},
        {"n_layer": 12, "n_head": 12, "n_embd": 1024, "name": "gpt2-medium"},
    ]

    batch_size = 32  # Fixed batch size

    for config in model_configs:
        print(f"   Testing model: {config['name']}")

        # Clear memory completely
        torch.cuda.empty_cache()
        gc.collect()

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
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Create test data
        x = torch.randint(0, 50304, (batch_size, 1024), device=fabric.device)
        y = torch.randint(0, 50304, (batch_size, 1024), device=fabric.device)
        test_batch = (x, y)

        # Warmup
        model.train()
        for _ in range(warmup_iterations):
            _ = model.training_step(test_batch, batch_idx=0)

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

        # Measure memory over multiple iterations
        memory_readings = []
        for i in range(num_iterations):
            optimizer.zero_grad()
            loss = model.training_step(test_batch, batch_idx=i)
            current_mem = torch.cuda.memory_allocated()
            memory_readings.append(current_mem / 1e9)

        # Calculate statistics
        avg_memory = sum(memory_readings) / len(memory_readings)
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        memory_per_param = avg_memory / (total_params / 1e6)  # GB per million params

        results["model_size_experiment"].append(
            {
                "model_name": config["name"],
                "config": config,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "avg_memory_gb": avg_memory,
                "peak_memory_gb": peak_memory,
                "memory_per_param_gb": memory_per_param,
                "memory_readings": memory_readings,
            }
        )

        print(f"     Params: {total_params/1e6:.1f}M, Memory: {avg_memory:.2f} GB")

        # Clean up
        del model, optimizer, x, y, test_batch
        torch.cuda.empty_cache()
        gc.collect()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"memory_scaling_experiment_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {filename}")

    return results


if __name__ == "__main__":
    fabric = Fabric(accelerator="cuda", devices=1)
    measure_memory_scaling_experiments(fabric)
