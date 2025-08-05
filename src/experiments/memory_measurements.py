"""Memory measurement experiments."""

import torch
from gpt2_standalone.lightning_module import GPTLightningModule
from gpt2_standalone.model import GPTConfig
from lightning import Fabric

from experiments.clean_palate import deep_gpu_reset, reset_model_state


def run_batch_size_memory_experiment(
    fabric: Fabric,
    batch_size: int,
    model_configs: list,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run batch size memory experiment for a specific batch size across all models.

    Args:
        fabric: Lightning Fabric instance.
        batch_size: Batch size to test.
        model_configs: List of model configurations to test.
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing memory experiment results for this batch size across all models.
    """
    import gc
    import statistics

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

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()

            # Measure memory over multiple iterations
            memory_readings = []
            forward_memory_readings = []
            backward_memory_readings = []
            peak_forward_readings = []
            peak_backward_readings = []

            for i in range(num_iterations):
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

            model_result = {
                "model_name": config["name"],
                "config": config,
                "total_params": total_params,
                "avg_memory_gb": avg_memory,
                "std_memory_gb": std_memory,
                "avg_forward_memory_gb": avg_forward_memory,
                "avg_backward_memory_gb": avg_backward_memory,
                "avg_peak_forward_memory_gb": avg_peak_forward_memory,
                "avg_peak_backward_memory_gb": avg_peak_backward_memory,
                "peak_memory_gb": peak_memory,
                "memory_per_sample_gb": memory_per_sample,
                "memory_readings": memory_readings,
                "forward_memory_readings": forward_memory_readings,
                "backward_memory_readings": backward_memory_readings,
                "peak_forward_readings": peak_forward_readings,
                "peak_backward_readings": peak_backward_readings,
                "status": "success",
            }

            print(
                f"       Total: {avg_memory:.2f}GB ± {std_memory:.2f}GB, "
                f"Forward: {avg_forward_memory:.2f}GB, Backward: {avg_backward_memory:.2f}GB, "
                f"Peak F: {avg_peak_forward_memory:.2f}GB, Peak B: {avg_peak_backward_memory:.2f}GB"
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


def run_batch_size_memory_experiment_eval(
    fabric: Fabric,
    batch_size: int,
    model_configs: list,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run batch size memory experiment in evaluation mode for a specific batch size across all models.

    Args:
        fabric: Lightning Fabric instance.
        batch_size: Batch size to test.
        model_configs: List of model configurations to test.
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing evaluation memory experiment results for this batch size across all models.
    """
    import gc
    import statistics

    print(f"   Testing batch size (eval): {batch_size}")

    batch_results = {
        "batch_size": batch_size,
        "models": [],
        "status": "success",
    }

    for config in model_configs:
        print(f"     Testing model (eval): {config['name']}")

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
            model.eval()
            for _ in range(warmup_iterations):
                deep_gpu_reset()
                with torch.no_grad():
                    _ = model(x)

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()

            # Measure memory over multiple iterations
            memory_readings = []
            forward_memory_readings = []
            peak_forward_readings = []

            for i in range(num_iterations):
                # Clean palate before each measurement
                deep_gpu_reset()
                reset_model_state(model, optimizer)

                # Set to evaluation mode
                model.eval()

                # Measure forward pass memory only (no backward in eval mode)
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated()

                with torch.no_grad():
                    _ = model(x)

                forward_mem = torch.cuda.memory_allocated()
                peak_forward_mem = torch.cuda.max_memory_allocated()
                forward_memory_readings.append(forward_mem / 1e9)
                peak_forward_readings.append(peak_forward_mem / 1e9)

                # In evaluation mode, total memory is just forward memory
                total_mem = forward_mem
                memory_readings.append(total_mem / 1e9)

            # Calculate statistics
            avg_memory = statistics.mean(memory_readings)
            std_memory = (
                statistics.stdev(memory_readings) if len(memory_readings) > 1 else 0.0
            )
            avg_forward_memory = statistics.mean(forward_memory_readings)
            avg_peak_forward_memory = statistics.mean(peak_forward_readings)

            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            memory_per_sample = avg_memory / batch_size

            model_result = {
                "model_name": config["name"],
                "config": config,
                "total_params": total_params,
                "avg_memory_gb": avg_memory,
                "std_memory_gb": std_memory,
                "avg_forward_memory_gb": avg_forward_memory,
                "avg_backward_memory_gb": 0.0,  # No backward pass in eval mode
                "avg_peak_forward_memory_gb": avg_peak_forward_memory,
                "avg_peak_backward_memory_gb": 0.0,  # No backward pass in eval mode
                "peak_memory_gb": peak_memory,
                "memory_per_sample_gb": memory_per_sample,
                "memory_readings": memory_readings,
                "forward_memory_readings": forward_memory_readings,
                "backward_memory_readings": [0.0] * num_iterations,  # No backward pass
                "peak_forward_readings": peak_forward_readings,
                "peak_backward_readings": [0.0] * num_iterations,  # No backward pass
                "status": "success",
            }

            print(
                f"       Total (eval): {avg_memory:.2f}GB ± {std_memory:.2f}GB, "
                f"Forward: {avg_forward_memory:.2f}GB, Peak F: {avg_peak_forward_memory:.2f}GB"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"       ❌ Out of memory for model {config['name']} (eval)")
                model_result = {
                    "model_name": config["name"],
                    "config": config,
                    "status": "out_of_memory",
                    "error": str(e),
                }
            else:
                print(f"       ❌ Runtime error for model {config['name']} (eval): {e}")
                model_result = {
                    "model_name": config["name"],
                    "config": config,
                    "status": "runtime_error",
                    "error": str(e),
                }
        except Exception as e:
            print(f"       ❌ Unexpected error for model {config['name']} (eval): {e}")
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


def run_model_size_memory_experiment(
    fabric: Fabric,
    config: dict,
    batch_sizes: list,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run model size memory experiment for a specific model across all batch sizes.

    Args:
        fabric: Lightning Fabric instance.
        config: Model configuration dictionary.
        batch_sizes: List of batch sizes to test (ordered from smallest to largest).
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing memory experiment results for this model across all batch sizes.
    """
    import gc
    import statistics

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

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()

            # Measure memory over multiple iterations
            memory_readings = []
            forward_memory_readings = []
            backward_memory_readings = []
            peak_forward_readings = []
            peak_backward_readings = []

            for i in range(num_iterations):
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
            memory_per_param = avg_memory / (
                total_params / 1e6
            )  # GB per million params

            batch_result = {
                "batch_size": batch_size,
                "total_params": total_params,
                "avg_memory_gb": avg_memory,
                "std_memory_gb": std_memory,
                "avg_forward_memory_gb": avg_forward_memory,
                "avg_backward_memory_gb": avg_backward_memory,
                "avg_peak_forward_memory_gb": avg_peak_forward_memory,
                "avg_peak_backward_memory_gb": avg_peak_backward_memory,
                "peak_memory_gb": peak_memory,
                "memory_per_sample_gb": memory_per_sample,
                "memory_per_param_gb": memory_per_param,
                "memory_readings": memory_readings,
                "forward_memory_readings": forward_memory_readings,
                "backward_memory_readings": backward_memory_readings,
                "peak_forward_readings": peak_forward_readings,
                "peak_backward_readings": peak_backward_readings,
                "status": "success",
            }

            print(
                f"       Total: {avg_memory:.2f}GB ± {std_memory:.2f}GB, "
                f"Forward: {avg_forward_memory:.2f}GB, Backward: {avg_backward_memory:.2f}GB, "
                f"Peak F: {avg_peak_forward_memory:.2f}GB, Peak B: {avg_peak_backward_memory:.2f}GB"
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


def run_model_size_memory_experiment_eval(
    fabric: Fabric,
    config: dict,
    batch_sizes: list,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run model size memory experiment in evaluation mode."""

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
            model.eval()
            for _ in range(warmup_iterations):
                deep_gpu_reset()
                _ = model.training_step(test_batch, batch_idx=0)

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()

            # Measure memory over multiple iterations
            memory_readings = []
            forward_memory_readings = []
            backward_memory_readings = []
            peak_forward_readings = []
            peak_backward_readings = []

            for i in range(num_iterations):
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
            memory_per_param = avg_memory / (
                total_params / 1e6
            )  # GB per million params

            batch_result = {
                "batch_size": batch_size,
                "total_params": total_params,
                "avg_memory_gb": avg_memory,
                "std_memory_gb": std_memory,
                "avg_forward_memory_gb": avg_forward_memory,
                "avg_backward_memory_gb": avg_backward_memory,
                "avg_peak_forward_memory_gb": avg_peak_forward_memory,
                "avg_peak_backward_memory_gb": avg_peak_backward_memory,
                "peak_memory_gb": peak_memory,
                "memory_per_sample_gb": memory_per_sample,
                "memory_per_param_gb": memory_per_param,
                "memory_readings": memory_readings,
                "forward_memory_readings": forward_memory_readings,
                "backward_memory_readings": backward_memory_readings,
                "peak_forward_readings": peak_forward_readings,
                "peak_backward_readings": peak_backward_readings,
                "status": "success",
            }

            print(
                f"       Total: {avg_memory:.2f}GB ± {std_memory:.2f}GB, "
                f"Forward: {avg_forward_memory:.2f}GB, Backward: {avg_backward_memory:.2f}GB, "
                f"Peak F: {avg_peak_forward_memory:.2f}GB, Peak B: {avg_peak_backward_memory:.2f}GB"
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


def run_sequence_length_memory_experiment(
    fabric: Fabric,
    config: dict,
    batch_size: int,
    sequence_lengths: list,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run sequence length memory experiment for a specific model and batch size.

    Args:
        fabric: Lightning Fabric instance.
        config: Model configuration dictionary.
        batch_size: Batch size to use for testing.
        sequence_lengths: List of sequence lengths to test (ordered from shortest to longest).
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dictionary containing memory experiment results for this model across all sequence lengths.
    """
    import gc
    import statistics

    print(f"   Testing model: {config['name']}, batch_size: {batch_size}")

    seq_results = {
        "model_name": config["name"],
        "config": config,
        "batch_size": batch_size,
        "sequence_lengths": [],
        "status": "success",
    }

    for seq_len in sequence_lengths:
        print(f"     Testing sequence length: {seq_len}")

        try:
            # Clean palate before starting
            deep_gpu_reset()

            # Create fresh model with current sequence length
            model_config = GPTConfig(
                block_size=seq_len,
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

            # Create test data with current sequence length
            x = torch.randint(0, 50304, (batch_size, seq_len), device=fabric.device)
            y = torch.randint(0, 50304, (batch_size, seq_len), device=fabric.device)
            test_batch = (x, y)

            # Warmup with clean palate between iterations
            model.train()
            for _ in range(warmup_iterations):
                deep_gpu_reset()
                _ = model.training_step(test_batch, batch_idx=0)

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()

            # Measure memory over multiple iterations
            memory_readings = []
            forward_memory_readings = []
            backward_memory_readings = []
            peak_forward_readings = []
            peak_backward_readings = []

            for i in range(num_iterations):
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
            memory_per_token = avg_memory / (batch_size * seq_len)

            seq_result = {
                "sequence_length": seq_len,
                "total_params": total_params,
                "avg_memory_gb": avg_memory,
                "std_memory_gb": std_memory,
                "avg_forward_memory_gb": avg_forward_memory,
                "avg_backward_memory_gb": avg_backward_memory,
                "avg_peak_forward_memory_gb": avg_peak_forward_memory,
                "avg_peak_backward_memory_gb": avg_peak_backward_memory,
                "peak_memory_gb": peak_memory,
                "memory_per_token_gb": memory_per_token,
                "memory_readings": memory_readings,
                "forward_memory_readings": forward_memory_readings,
                "backward_memory_readings": backward_memory_readings,
                "peak_forward_readings": peak_forward_readings,
                "peak_backward_readings": peak_backward_readings,
                "status": "success",
            }

            print(
                f"       Total: {avg_memory:.2f}GB ± {std_memory:.2f}GB, "
                f"Forward: {avg_forward_memory:.2f}GB, Backward: {avg_backward_memory:.2f}GB, "
                f"Peak F: {avg_peak_forward_memory:.2f}GB, Peak B: {avg_peak_backward_memory:.2f}GB"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"       ❌ Out of memory for sequence length {seq_len}")
                seq_result = {
                    "sequence_length": seq_len,
                    "status": "out_of_memory",
                    "error": str(e),
                }
                # Stop testing longer sequences since they will also fail
                print(
                    f"       ⚠️  Stopping sequence length tests (will fail for longer sequences)"
                )
                seq_results["sequence_lengths"].append(seq_result)
                break
            else:
                print(f"       ❌ Runtime error for sequence length {seq_len}: {e}")
                seq_result = {
                    "sequence_length": seq_len,
                    "status": "runtime_error",
                    "error": str(e),
                }
        except Exception as e:
            print(f"       ❌ Unexpected error for sequence length {seq_len}: {e}")
            seq_result = {
                "sequence_length": seq_len,
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

        seq_results["sequence_lengths"].append(seq_result)

    return seq_results


def run_sequence_length_memory_experiment_eval(
    fabric: Fabric,
    config: dict,
    batch_size: int,
    sequence_lengths: list,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Run sequence length memory experiment in evaluation mode."""

    print(f"   Testing model: {config['name']}, batch_size: {batch_size}")

    seq_results = {
        "model_name": config["name"],
        "config": config,
        "batch_size": batch_size,
        "sequence_lengths": [],
        "status": "success",
    }

    for seq_len in sequence_lengths:
        print(f"     Testing sequence length: {seq_len}")

        try:
            # Clean palate before starting
            deep_gpu_reset()

            # Create fresh model with current sequence length
            model_config = GPTConfig(
                block_size=seq_len,
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

            # Create test data with current sequence length
            x = torch.randint(0, 50304, (batch_size, seq_len), device=fabric.device)
            y = torch.randint(0, 50304, (batch_size, seq_len), device=fabric.device)
            test_batch = (x, y)

            # Warmup with clean palate between iterations
            model.eval()
            for _ in range(warmup_iterations):
                deep_gpu_reset()
                _ = model.training_step(test_batch, batch_idx=0)

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()

            # Measure memory over multiple iterations
            memory_readings = []
            forward_memory_readings = []
            backward_memory_readings = []
            peak_forward_readings = []
            peak_backward_readings = []

            for i in range(num_iterations):
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
            memory_per_token = avg_memory / (batch_size * seq_len)

            seq_result = {
                "sequence_length": seq_len,
                "total_params": total_params,
                "avg_memory_gb": avg_memory,
                "std_memory_gb": std_memory,
                "avg_forward_memory_gb": avg_forward_memory,
                "avg_backward_memory_gb": avg_backward_memory,
                "avg_peak_forward_memory_gb": avg_peak_forward_memory,
                "avg_peak_backward_memory_gb": avg_peak_backward_memory,
                "peak_memory_gb": peak_memory,
                "memory_per_token_gb": memory_per_token,
                "memory_readings": memory_readings,
                "forward_memory_readings": forward_memory_readings,
                "backward_memory_readings": backward_memory_readings,
                "peak_forward_readings": peak_forward_readings,
                "peak_backward_readings": peak_backward_readings,
                "status": "success",
            }

            print(
                f"       Total: {avg_memory:.2f}GB ± {std_memory:.2f}GB, "
                f"Forward: {avg_forward_memory:.2f}GB, Backward: {avg_backward_memory:.2f}GB, "
                f"Peak F: {avg_peak_forward_memory:.2f}GB, Peak B: {avg_peak_backward_memory:.2f}GB"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"       ❌ Out of memory for sequence length {seq_len}")
                seq_result = {
                    "sequence_length": seq_len,
                    "status": "out_of_memory",
                    "error": str(e),
                }
                # Stop testing longer sequences since they will also fail
                print(
                    f"       ⚠️  Stopping sequence length tests (will fail for longer sequences)"
                )
                seq_results["sequence_lengths"].append(seq_result)
                break
            else:
                print(f"       ❌ Runtime error for sequence length {seq_len}: {e}")
                seq_result = {
                    "sequence_length": seq_len,
                    "status": "runtime_error",
                    "error": str(e),
                }
        except Exception as e:
            print(f"       ❌ Unexpected error for sequence length {seq_len}: {e}")
            seq_result = {
                "sequence_length": seq_len,
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

        seq_results["sequence_lengths"].append(seq_result)

    return seq_results


def measure_memory_scaling_experiments(
    fabric: Fabric,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict:
    """Comprehensive memory scaling experiments (training + evaluation).

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

    print(f"\n🧪 Starting Memory Scaling Experiments")
    print(f"   Warmup iterations: {warmup_iterations}")
    print(f"   Measurement iterations: {num_iterations}")

    # Define test configurations
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]  # Ordered from smallest to largest
    sequence_lengths = [
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ]  # Ordered from shortest to longest
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
        "sequence_length_experiment": [],
        "batch_size_experiment_eval": [],
        "model_size_experiment_eval": [],
        "sequence_length_experiment_eval": [],
    }

    # ----------------------------------------------------------------------
    # Experiment 1: Batch Size vs Memory (Training Mode)
    print(f"\n==> 📊 Experiment 1: Batch Size vs Memory (Training)")
    print(f"   Testing each batch size across all models")

    for batch_size in batch_sizes:
        batch_result = run_batch_size_memory_experiment(
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
    # Experiment 2: Model Size vs Memory (Training Mode)
    print(f"\n==> 📊 Experiment 2: Model Size vs Memory (Training)")
    print(
        f"   Testing each model across all batch sizes (ordered from smallest to largest)"
    )

    for config in model_configs:
        model_result = run_model_size_memory_experiment(
            fabric, config, batch_sizes, num_iterations, warmup_iterations
        )
        results["model_size_experiment"].append(model_result)

    # ----------------------------------------------------------------------
    # Experiment 3: Sequence Length vs Memory (Training Mode)
    print(f"\n==> 📊 Experiment 3: Sequence Length vs Memory (Training)")
    print(
        f"   Testing each model across all sequence lengths (ordered from shortest to longest)"
    )

    # Use a moderate batch size for sequence length experiments
    moderate_batch_size = 16

    for config in model_configs:
        seq_result = run_sequence_length_memory_experiment(
            fabric,
            config,
            moderate_batch_size,
            sequence_lengths,
            num_iterations,
            warmup_iterations,
        )
        results["sequence_length_experiment"].append(seq_result)

    # ----------------------------------------------------------------------
    # Experiment 1 (Eval): Batch Size vs Memory (Evaluation Mode)
    print(f"\n==> 📊 Experiment 1 (Eval): Batch Size vs Memory (Evaluation)")
    print(f"   Testing each batch size across all models")

    for batch_size in batch_sizes:
        batch_result = run_batch_size_memory_experiment_eval(
            fabric, batch_size, model_configs, num_iterations, warmup_iterations
        )
        results["batch_size_experiment_eval"].append(batch_result)

        # Check if all models failed for this batch size
        successful_models = [
            m for m in batch_result["models"] if m.get("status") == "success"
        ]
        if not successful_models:
            print(
                f"     ⚠️  All models failed for batch size {batch_size} (eval), stopping batch size experiments"
            )
            break

    # ----------------------------------------------------------------------
    # Experiment 2 (Eval): Model Size vs Memory (Evaluation Mode)
    print(f"\n==> 📊 Experiment 2 (Eval): Model Size vs Memory (Evaluation)")
    print(
        f"   Testing each model across all batch sizes (ordered from smallest to largest)"
    )

    for config in model_configs:
        model_result = run_model_size_memory_experiment_eval(
            fabric, config, batch_sizes, num_iterations, warmup_iterations
        )
        results["model_size_experiment_eval"].append(model_result)

    # ----------------------------------------------------------------------
    # Experiment 3 (Eval): Sequence Length vs Memory (Evaluation Mode)
    print(f"\n==> 📊 Experiment 3 (Eval): Sequence Length vs Memory (Evaluation)")
    print(
        f"   Testing each model across all sequence lengths (ordered from shortest to longest)"
    )

    # Use a moderate batch size for sequence length experiments
    moderate_batch_size = 16

    for config in model_configs:
        seq_result = run_sequence_length_memory_experiment_eval(
            fabric,
            config,
            moderate_batch_size,
            sequence_lengths,
            num_iterations,
            warmup_iterations,
        )
        results["sequence_length_experiment_eval"].append(seq_result)

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
