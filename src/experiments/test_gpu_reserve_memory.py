"""Test GPU memory measurement behavior using the same routines as memory_measurements.py."""

import torch


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
    mem_allocated: {to_gb(mem_allocated_bytes)} Gb
    mem_reserved: {to_gb(mem_reserved_bytes)} Gb
    max_memory_allocated: {to_gb(torch.cuda.max_memory_allocated())} Gb
    max_memory_reserved: {to_gb(torch.cuda.max_memory_reserved())} Gb
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
    print_gpu_info("\nAfter del a1 of 10^9 floats")


def test_detailed_gpu_mem_ops_empty_cache() -> None:
    """Test detailed GPU memory operations."""
    print("=== TEST 5: GPU Memory Function Validation ===")
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
    print_gpu_info("\nAfter del a1 of 10^9 floats")


if __name__ == "__main__":
    print(" Running GPU Memory Tests")
    print("=" * 50)

    print("\n\ntest_detailed_gpu_mem_ops()")
    test_detailed_gpu_mem_ops()
    print("\n\ntest_detailed_gpu_mem_ops_empty_cache()")
    test_detailed_gpu_mem_ops_empty_cache()
    print("\n" + "=" * 50)
    print("✅ All tests completed", flush=True)
