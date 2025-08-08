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


# ============================================================================
# TEST 1: Basic GPU Memory Function Validation
# ============================================================================


def test_gpu_functions():
    """TEST 1: Test if GPU memory functions are working correctly."""
    print("=== TEST 1: GPU Memory Function Validation ===")
    print("Device:", torch.cuda.get_device_name(0))
    print()

    # Test memory functions directly
    print("--- Direct Memory Function Tests ---")

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


# ============================================================================
# TEST 2: Memory Measurement with Cached Memory
# ============================================================================


def test_memory_measurement_with_cached():
    """TEST 2: Test memory measurement including cached memory."""
    print("\n\n=== TEST 2: Memory Measurement with Cached Memory ===")

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


# ============================================================================
# TEST 3: Test Functions Without Deleting Tensors
# ============================================================================


@memory_measurement
def create_tensor_no_delete():
    """Create a tensor and DON'T delete it."""
    size = 500
    expected_memory = calculate_tensor_memory(size)
    print(
        f"Creating tensor of size {size}x{size}x{size} (expected: {expected_memory:.3f}GB) - NOT DELETING"
    )

    tensor = torch.randn(size, size, size, device="cuda")
    # NO del tensor here!
    return {
        "operation": "tensor_no_delete",
        "size": size,
        "expected_memory_gb": expected_memory,
    }


def to_gb(mem):
    return round(mem / (1024.0 * 1024.0 * 1024.0), 2)


def print_gpu_info(msg: str = ""):
    print(f"{msg}")
    mem_allocated_bytes = torch.cuda.memory_allocated()
    mem_reserved_bytes = torch.cuda.memory_reserved()
    mem_cached_bytes = mem_reserved_bytes - mem_allocated_bytes
    # torch.cuda.memory_stats()
    # memory_summary = torch.cuda.memory_summary()
    # print(f"""
    # =======================================================================================
    # memory summary: {memory_summary}
    # """)
    print(f"""
    ---------------------------------------------------------------------------------------
    memory_allocated: {to_gb(mem_allocated_bytes)} Gb
    max_memory_allocated: {to_gb(torch.cuda.max_memory_allocated())} Gb
    max_memory_reserved: {to_gb(mem_reserved_bytes)} Gb
    mem cached: {to_gb(mem_cached_bytes)} Gb
    =======================================================================================
    """)


def test_detailed_gpu_mem_ops() -> None:
    """Test detailed GPU memory operations."""
    print("=== TEST 4: GPU Memory Function Validation ===")
    print("Device:", torch.cuda.get_device_name(0))
    print_gpu_info("\nBefore allocation")
    a = torch.randn(1000, 1000, 1000, device="cuda")
    print_gpu_info("\nAfter allocation of a = 10^9 floats")
    a1 = torch.randn(1000, 1000, 1000, device="cuda")
    print_gpu_info("\nAfter allocation of a1 = 10^9 floats")
    b = torch.randn(10, 10, 10, device="cuda")
    print_gpu_info("\nAfter allocation of b = 10^3 floats")
    c = torch.randn(1000, 1000, 1000, device="cuda")
    print_gpu_info("\nAfter allocation of c = 10^9 floats")
    del b
    print_gpu_info("\nAfter del b of 10^3 floats")
    del a1
    print_gpu_info("\nAfter del b of 10^3 floats")


def test_detailed_gpu_mem_ops_empty_cache() -> None:
    """Test detailed GPU memory operations."""
    print("=== TEST 4: GPU Memory Function Validation ===")
    print("Device:", torch.cuda.get_device_name(0))
    print_gpu_info("\nBefore allocation")
    torch.cuda.empty_cache()
    print_gpu_info("\nAfter empty cache")
    a = torch.randn(1000, 1000, 1000, device="cuda")
    print_gpu_info("\nAfter allocation of a = 10^9 floats")
    a1 = torch.randn(1000, 1000, 1000, device="cuda")
    print_gpu_info("\nAfter allocation of a1 = 10^9 floats")
    b = torch.randn(10, 10, 10, device="cuda")
    print_gpu_info("\nAfter allocation of b = 10^3 floats")
    c = torch.randn(1000, 1000, 1000, device="cuda")
    print_gpu_info("\nAfter allocation of c = 10^9 floats")
    del b
    print_gpu_info("\nAfter del b of 10^3 floats")
    del a1
    print_gpu_info("\nAfter del b of 10^3 floats")


def test_without_deletion():
    """TEST 3: Test what happens when we DON'T delete tensors."""
    print("\n\n=== TEST 3: Test Without Deleting Tensors ===")

    # Test tensor without deletion
    print("\n--- Testing tensor (no delete) ---")
    result = create_tensor_no_delete()
    print(f"Result: {result}")
    print(
        f"Expected: {result['expected_memory_gb']:.3f}GB, Measured: {result['memory_gb']:.3f}GB, Net: {result['net_memory_gb']:.3f}GB"
    )
    print_memory_status("After tensor (no delete)")


# ============================================================================
# COMMENTED OUT OLD TESTS - NO LONGER NEEDED
# ============================================================================

# def test_single_measurement():
#     """OLD TEST: Test a single memory measurement."""
#     # ... commented out ...

# def test_multiple_iterations():
#     """OLD TEST: Test multiple iterations like in memory_measurements.py."""
#     # ... commented out ...

# def test_without_deep_gpu_reset():
#     """OLD TEST: Test without deep_gpu_reset to see if that's the issue."""
#     # ... commented out ...


if __name__ == "__main__":
    print(" Running GPU Memory Tests")
    print("=" * 50)

    # Run the three main tests
    # test_gpu_functions()  # Test 1: Validate GPU functions work
    # test_memory_measurement_with_cached()  # Test 2: Test with cached memory
    # test_without_deletion()  # Test 3: Test without deleting tensors
    print("\n\ntest_detailed_gpu_mem_ops()")
    test_detailed_gpu_mem_ops()
    print("\n\ntest_detailed_gpu_mem_ops_empty_cache()")
    test_detailed_gpu_mem_ops_empty_cache()
    print("\n" + "=" * 50)
    print("✅ All tests completed", flush=True)
