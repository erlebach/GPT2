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
    print(
        f"{label:20s} | Allocated: {allocated:8.3f}GB | Reserved: {reserved:8.3f}GB | Max: {max_allocated:8.3f}GB"
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
        f"Expected: {result['expected_memory_gb']:.3f}GB, Measured: {result['memory_gb']:.3f}GB, Net: {result['net_memory_gb']:.3f}GB"
    )
    print_memory_status("After medium tensor")

    # Test large tensor
    print("\n--- Testing large tensor ---")
    result = create_large_tensor()
    print(f"Result: {result}")
    print(
        f"Expected: {result['expected_memory_gb']:.3f}GB, Measured: {result['memory_gb']:.3f}GB, Net: {result['net_memory_gb']:.3f}GB"
    )
    print_memory_status("After large tensor")


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


if __name__ == "__main__":
    test_single_measurement()
    test_multiple_iterations()
    test_without_deep_gpu_reset()
    test_without_deletion()  # Add this new test
