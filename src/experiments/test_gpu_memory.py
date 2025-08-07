"""Test GPU memory measurement behavior using the same routines as memory_measurements.py."""

import gc
import statistics

import torch

from experiments.clean_palate import deep_gpu_reset


def print_memory_status(label: str):
    """Print current memory status using same functions as memory_measurements.py."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    max_reserved = torch.cuda.max_memory_reserved() / 1e9

    # Calculate cached memory (reserved - allocated)
    cached = reserved - allocated

    print(
        f"{label:20s} | Allocated: {allocated:8.3f}GB | Reserved: {reserved:8.3f}GB | Cached: {cached:8.3f}GB | Max Alloc: {max_allocated:8.3f}GB | Max Reserved: {max_reserved:8.3f}GB"
    )


def calculate_tensor_memory(size: int) -> float:
    """Calculate expected memory usage for a tensor of given size."""
    # Float32 tensor: 4 bytes per element
    elements = size**3
    memory_bytes = elements * 4
    memory_gb = memory_bytes / 1e9
    return memory_gb


def memory_measurement(func):
    """Copy of the exact decorator from memory_measurements.py."""

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
def create_small_tensor():
    """Create a small tensor."""
    size = 100
    expected_memory = calculate_tensor_memory(size)
    print(
        f"Creating tensor of size {size}x{size}x{size} (expected: {expected_memory:.3f}GB)"
    )

    tensor = torch.randn(size, size, size, device="cuda")
    del tensor
    return {
        "operation": "small_tensor",
        "size": size,
        "expected_memory_gb": expected_memory,
    }


@memory_measurement
def create_medium_tensor():
    """Create a medium tensor."""
    size = 500
    expected_memory = calculate_tensor_memory(size)
    print(
        f"Creating tensor of size {size}x{size}x{size} (expected: {expected_memory:.3f}GB)"
    )

    tensor = torch.randn(size, size, size, device="cuda")
    del tensor
    return {
        "operation": "medium_tensor",
        "size": size,
        "expected_memory_gb": expected_memory,
    }


@memory_measurement
def create_large_tensor():
    """Create a large tensor."""
    size = 1000
    expected_memory = calculate_tensor_memory(size)
    print(
        f"Creating tensor of size {size}x{size}x{size} (expected: {expected_memory:.3f}GB)"
    )

    tensor = torch.randn(size, size, size, device="cuda")
    del tensor
    return {
        "operation": "large_tensor",
        "size": size,
        "expected_memory_gb": expected_memory,
    }


@memory_measurement
def create_small_tensor_no_delete():
    """Create a small tensor and DON'T delete it."""
    size = 100
    expected_memory = calculate_tensor_memory(size)
    print(
        f"Creating tensor of size {size}x{size}x{size} (expected: {expected_memory:.3f}GB) - NOT DELETING"
    )

    tensor = torch.randn(size, size, size, device="cuda")
    # NO del tensor here!
    return {
        "operation": "small_tensor_no_delete",
        "size": size,
        "expected_memory_gb": expected_memory,
    }


@memory_measurement
def create_medium_tensor_no_delete():
    """Create a medium tensor and DON'T delete it."""
    size = 500
    expected_memory = calculate_tensor_memory(size)
    print(
        f"Creating tensor of size {size}x{size}x{size} (expected: {expected_memory:.3f}GB) - NOT DELETING"
    )

    tensor = torch.randn(size, size, size, device="cuda")
    # NO del tensor here!
    return {
        "operation": "medium_tensor_no_delete",
        "size": size,
        "expected_memory_gb": expected_memory,
    }


@memory_measurement
def create_tensor_with_reference():
    """Create a tensor and keep a reference to it."""
    size = 500
    expected_memory = calculate_tensor_memory(size)
    print(
        f"Creating tensor of size {size}x{size}x{size} (expected: {expected_memory:.3f}GB) - KEEPING REFERENCE"
    )

    tensor = torch.randn(size, size, size, device="cuda")

    # Return the tensor itself to keep a reference
    return {
        "operation": "tensor_with_reference",
        "size": size,
        "expected_memory_gb": expected_memory,
        "tensor": tensor,  # Keep reference to prevent GC
    }


def test_single_measurement():
    """Test a single memory measurement."""
    print("=== Single Memory Measurement Test ===")
    print("Device:", torch.cuda.get_device_name(0))
    print()

    # Initial state
    print_memory_status("Initial")

    # Test small tensor
    print("\n--- Testing small tensor ---")
    result = create_small_tensor()
    print(f"Result: {result}")
    print(
        f"Expected: {result['expected_memory_gb']:.3f}GB, Measured: {result['memory_gb']:.3f}GB, Net: {result['net_memory_gb']:.3f}GB"
    )
    print_memory_status("After small tensor")

    # Test medium tensor
    print("\n--- Testing medium tensor ---")
    result = create_medium_tensor()
    print(f"Result: {result}")
    print(
        f"Expected: {result['expected_memory_gb']:.3f}GB, Measured: {result['memory_gb']:.3f}GB, Net: {result['net_memory_gb']:.3f}GB",
        flush=True,
    )
    print_memory_status("After medium tensor")

    # Test large tensor
    print("\n--- Testing large tensor ---")
    result = create_large_tensor()
    print(f"Result: {result}")
    print(
        f"Expected: {result['expected_memory_gb']:.3f}GB, Measured: {result['memory_gb']:.3f}GB, Net: {result['net_memory_gb']:.3f}GB"
    )
    print_memory_status("After large tensor", flush=True)


def test_multiple_iterations():
    """Test multiple iterations like in memory_measurements.py."""
    print("\n\n=== Multiple Iterations Test ===")

    num_iterations = 5
    memory_readings = []
    net_memory_readings = []
    peak_memory_readings = []

    print(f"Running {num_iterations} iterations...")

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1} ---")

        # Clean palate before each measurement (like in memory_measurements.py)
        deep_gpu_reset()

        # Run measurement
        result = create_medium_tensor()

        # Store readings
        memory_readings.append(result["memory_gb"])
        net_memory_readings.append(result["net_memory_gb"])
        peak_memory_readings.append(result["peak_memory_gb"])

        print(f"Expected: {result['expected_memory_gb']:.3f}GB")
        print(f"Memory: {result['memory_gb']:.3f}GB")
        print(f"Net memory: {result['net_memory_gb']:.3f}GB")
        print(f"Peak memory: {result['peak_memory_gb']:.3f}GB")

    # Calculate statistics
    avg_memory = statistics.mean(memory_readings)
    avg_net_memory = statistics.mean(net_memory_readings)
    avg_peak_memory = statistics.mean(peak_memory_readings)

    print(f"\n--- Statistics ---")
    print(f"Average memory: {avg_memory:.3f}GB")
    print(f"Average net memory: {avg_net_memory:.3f}GB")
    print(f"Average peak memory: {avg_peak_memory:.3f}GB")
    print(f"All net memory values: {[f'{x:.3f}' for x in net_memory_readings]}")


def test_without_deep_gpu_reset():
    """Test without deep_gpu_reset to see if that's the issue."""
    print("\n\n=== Test Without Deep GPU Reset ===")

    num_iterations = 3
    memory_readings = []
    net_memory_readings = []

    print(f"Running {num_iterations} iterations WITHOUT deep_gpu_reset...")

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1} ---")

        # Only do basic cleanup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Run measurement
        result = create_medium_tensor()

        # Store readings
        memory_readings.append(result["memory_gb"])
        net_memory_readings.append(result["net_memory_gb"])

        print(f"Expected: {result['expected_memory_gb']:.3f}GB")
        print(f"Memory: {result['memory_gb']:.3f}GB")
        print(f"Net memory: {result['net_memory_gb']:.3f}GB")

    print(f"\n--- Statistics ---")
    print(f"Average memory: {statistics.mean(memory_readings):.3f}GB")
    print(f"Average net memory: {statistics.mean(net_memory_readings):.3f}GB")
    print(f"All net memory values: {[f'{x:.3f}' for x in net_memory_readings]}")


def test_without_deletion():
    """Test what happens when we DON'T delete tensors."""
    print("\n\n=== Test Without Deleting Tensors ===")

    # Test small tensor without deletion
    print("\n--- Testing small tensor (no delete) ---")
    result = create_small_tensor_no_delete()
    print(f"Result: {result}")
    print(
        f"Expected: {result['expected_memory_gb']:.3f}GB, Measured: {result['memory_gb']:.3f}GB, Net: {result['net_memory_gb']:.3f}GB"
    )
    print_memory_status("After small tensor (no delete)")

    # Test medium tensor without deletion
    print("\n--- Testing medium tensor (no delete) ---")
    result = create_medium_tensor_no_delete()
    print(f"Result: {result}")
    print(
        f"Expected: {result['expected_memory_gb']:.3f}GB, Measured: {result['memory_gb']:.3f}GB, Net: {result['net_memory_gb']:.3f}GB"
    )
    print_memory_status("After medium tensor (no delete)")


def test_gpu_functions():
    """Test if GPU memory functions are working correctly."""
    print("\n\n=== Testing GPU Memory Functions ===")

    print("Testing basic GPU memory functions:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

    # Test memory functions directly
    print("\n--- Direct Memory Function Tests ---")

    # Clear everything first
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("After clearing:")
    print_memory_status("Cleared")

    # Create a tensor and measure immediately
    print("\nCreating tensor...")
    tensor = torch.randn(100, 100, 100, device="cuda")

    print("Immediately after creation:")
    print_memory_status("After creation")

    # Test individual functions
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()

    print(f"\nIndividual function results:")
    print(f"memory_allocated(): {allocated} bytes ({allocated/1e9:.6f} GB)")
    print(f"memory_reserved(): {reserved} bytes ({reserved/1e9:.6f} GB)")
    print(f"max_memory_allocated(): {max_allocated} bytes ({max_allocated/1e9:.6f} GB)")
    print(f"max_memory_reserved(): {max_reserved} bytes ({max_reserved/1e9:.6f} GB)")

    # Calculate expected memory
    expected_bytes = 100 * 100 * 100 * 4  # float32
    print(f"Expected memory: {expected_bytes} bytes ({expected_bytes/1e9:.6f} GB)")

    # Test if functions are returning reasonable values
    print(f"\nFunction validation:")
    print(f"allocated > 0: {allocated > 0}")
    print(f"reserved >= allocated: {reserved >= allocated}")
    print(f"max_allocated >= allocated: {max_allocated >= allocated}")
    print(f"max_reserved >= reserved: {max_reserved >= reserved}")
    print(f"allocated close to expected: {abs(allocated - expected_bytes) < 1000}")

    # Clean up
    del tensor
    torch.cuda.empty_cache()

    print("\nAfter cleanup:")
    print_memory_status("After cleanup")


def test_memory_measurement_with_cached():
    """Test memory measurement including cached memory."""
    print("\n\n=== Memory Measurement with Cached Memory ===")

    def memory_measurement_with_cache(func):
        """Enhanced memory measurement that tracks cached memory."""

        def wrapper(*args, **kwargs):
            # Reset stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            start_allocated = torch.cuda.memory_allocated()
            start_reserved = torch.cuda.memory_reserved()
            print_memory_status("Start")

            # Call function
            result = func(*args, **kwargs)

            # Measure after function
            end_allocated = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            print_memory_status("End")

            # Calculate differences
            net_allocated = end_allocated - start_allocated
            net_reserved = end_reserved - start_reserved

            print(f"Net allocated: {net_allocated/1e9:.6f}GB")
            print(f"Net reserved: {net_reserved/1e9:.6f}GB")
            print(f"Peak allocated: {peak_allocated/1e9:.6f}GB")
            print(f"Peak reserved: {peak_reserved/1e9:.6f}GB")

            return {
                "memory_gb": end_allocated / 1e9,
                "reserved_gb": end_reserved / 1e9,
                "net_memory_gb": net_allocated / 1e9,
                "net_reserved_gb": net_reserved / 1e9,
                "peak_memory_gb": peak_allocated / 1e9,
                "peak_reserved_gb": peak_reserved / 1e9,
            }

        return wrapper

    @memory_measurement_with_cache
    def create_tensor_test():
        """Create a tensor for testing."""
        size = 500
        expected_memory = calculate_tensor_memory(size)
        print(
            f"Creating tensor of size {size}x{size}x{size} (expected: {expected_memory:.3f}GB)"
        )

        tensor = torch.randn(size, size, size, device="cuda")
        return {"tensor": tensor, "expected": expected_memory}

    # Run the test
    result = create_tensor_test()
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    test_gpu_functions()  # Test if GPU functions work
    test_memory_measurement_with_cached()  # Test with cached memory
    test_single_measurement()
    test_multiple_iterations()
    test_without_deep_gpu_reset()
    test_without_deletion()
