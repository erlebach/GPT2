import torch


def reset_gpu_palate():
    """Reset GPU state between measurements."""
    if torch.cuda.is_available():
        # Clear all cached memory
        torch.cuda.empty_cache()

        # Reset peak memory statistics
        torch.cuda.reset_peak_memory_stats()

        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()


def deep_gpu_reset():
    """More thorough GPU reset."""
    if torch.cuda.is_available():
        # Clear memory pools
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Force garbage collection
        import gc

        gc.collect()

        # Synchronize
        torch.cuda.synchronize()

        # Optional: Reset CUDA streams
        torch.cuda.current_stream().synchronize()


def reset_model_state(model, optimizer):
    """Reset model and optimizer state."""
    # Zero gradients
    optimizer.zero_grad()

    # Reset model to eval mode and back to train
    model.eval()
    model.train()

    # Clear any cached computations
    if hasattr(model, "clear_cache"):
        model.clear_cache()


def measure_with_clean_palate(func, *args, **kwargs):
    """Measure function performance with clean GPU state."""

    # Step 1: Deep GPU reset
    deep_gpu_reset()

    # Step 2: Warmup (optional)
    # Run function once to establish CUDA kernels
    _ = func(*args, **kwargs)

    # Step 3: Reset again
    deep_gpu_reset()

    # Step 4: Actual measurement
    start_time = time.time()
    start_mem = torch.cuda.memory_allocated()

    result = func(*args, **kwargs)

    end_time = time.time()
    end_mem = torch.cuda.memory_allocated()

    return {
        "time": end_time - start_time,
        "memory": end_mem - start_mem,
        "result": result,
    }


def measure_batch_size_with_clean_palate(fabric, model, batch_size, num_iterations=10):
    """Measure batch size scaling with clean palate between tests."""

    results = []

    for i in range(num_iterations):
        # Clean palate between each measurement
        deep_gpu_reset()

        # Create fresh batch
        x = torch.randint(0, 50304, (batch_size, 1024), device=fabric.device)
        y = torch.randint(0, 50304, (batch_size, 1024), device=fabric.device)
        batch = (x, y)

        # Measure
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated()

        loss = model.training_step(batch, batch_idx=i)

        end_time = time.time()
        end_mem = torch.cuda.memory_allocated()

        results.append({"time": end_time - start_time, "memory": end_mem - start_mem})

        # Clean up batch
        del x, y, batch, loss

    return results
