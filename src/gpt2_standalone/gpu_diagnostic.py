#!/usr/bin/env python3
"""GPU Diagnostic Script

This script performs comprehensive GPU diagnostics to identify hardware issues,
accessibility problems, and configuration issues that might prevent multi-GPU training.
"""

import os
import subprocess
import sys
from typing import Any, Dict, List

import torch
import torch.distributed as dist


def run_system_command(cmd: str) -> str:
    """Run a system command and return the output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"Command timed out: {cmd}"
    except Exception as e:
        return f"Error running command: {e}"


def check_nvidia_smi() -> Dict[str, Any]:
    """Check NVIDIA-SMI output for GPU status."""
    print("🔍 Checking NVIDIA-SMI...")
    nvidia_smi = run_system_command("nvidia-smi")

    # Check if nvidia-smi is available
    if "NVIDIA-SMI" not in nvidia_smi:
        return {"available": False, "error": "nvidia-smi not found or not working"}

    # Parse GPU information
    lines = nvidia_smi.split("\n")
    gpu_info = []

    for line in lines:
        if "|" in line and "GPU" in line and "Memory" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                gpu_id = parts[0].strip()
                memory_info = parts[3].strip()
                gpu_info.append({"gpu_id": gpu_id, "memory_info": memory_info})

    return {"available": True, "output": nvidia_smi, "gpu_info": gpu_info}


def check_cuda_devices() -> Dict[str, Any]:
    """Check CUDA device accessibility through PyTorch."""
    print("🔍 Checking CUDA devices through PyTorch...")

    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device()
        if torch.cuda.is_available()
        else None,
        "devices": [],
    }

    if not info["cuda_available"]:
        return info

    for i in range(info["device_count"]):
        try:
            device_info = {
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
                "accessible": True,
            }

            # Test if we can actually use this device
            try:
                with torch.cuda.device(i):
                    test_tensor = torch.randn(100, 100, device=f"cuda:{i}")
                    del test_tensor
                    device_info["functional"] = True
            except Exception as e:
                device_info["functional"] = False
                device_info["error"] = str(e)

        except Exception as e:
            device_info = {"device_id": i, "accessible": False, "error": str(e)}

        info["devices"].append(device_info)

    return info


def check_environment_variables() -> Dict[str, Any]:
    """Check environment variables that affect GPU usage."""
    print("🔍 Checking environment variables...")

    relevant_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUDA_LAUNCH_BLOCKING",
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]

    env_info = {}
    for var in relevant_vars:
        value = os.environ.get(var)
        env_info[var] = value if value else "Not set"

    return env_info


def test_gpu_memory_allocation() -> Dict[str, Any]:
    """Test GPU memory allocation on each device."""
    print("🔍 Testing GPU memory allocation...")

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    results = {}
    device_count = torch.cuda.device_count()

    for i in range(device_count):
        try:
            with torch.cuda.device(i):
                # Get initial memory state
                initial_allocated = torch.cuda.memory_allocated()
                initial_reserved = torch.cuda.memory_reserved()

                # Try to allocate some memory
                test_tensor = torch.randn(1000, 1000, device=f"cuda:{i}")

                # Get memory state after allocation
                final_allocated = torch.cuda.memory_allocated()
                final_reserved = torch.cuda.memory_reserved()

                # Clean up
                del test_tensor
                torch.cuda.empty_cache()

                results[f"gpu_{i}"] = {
                    "initial_allocated_mb": initial_allocated / (1024**2),
                    "initial_reserved_mb": initial_reserved / (1024**2),
                    "final_allocated_mb": final_allocated / (1024**2),
                    "final_reserved_mb": final_reserved / (1024**2),
                    "allocation_successful": True,
                }

        except Exception as e:
            results[f"gpu_{i}"] = {"allocation_successful": False, "error": str(e)}

    return results


def test_multi_gpu_communication() -> Dict[str, Any]:
    """Test communication between GPUs."""
    print("🔍 Testing multi-GPU communication...")

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return {"error": "Need at least 2 GPUs for communication test"}

    try:
        # Test basic tensor operations across devices
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")

        # Create tensors on different devices
        tensor0 = torch.randn(100, 100, device=device0)
        tensor1 = torch.randn(100, 100, device=device1)

        # Test copying between devices
        tensor0_to_1 = tensor0.to(device1)
        tensor1_to_0 = tensor1.to(device0)

        # Test operations across devices
        result = tensor0 + tensor1_to_0

        return {
            "communication_successful": True,
            "cross_device_copy": True,
            "cross_device_operations": True,
        }

    except Exception as e:
        return {"communication_successful": False, "error": str(e)}


def main():
    """Run comprehensive GPU diagnostics."""
    print("🚀 Starting comprehensive GPU diagnostics...")
    print("=" * 60)

    # 1. Check NVIDIA-SMI
    print("\n1. NVIDIA-SMI Status:")
    nvidia_info = check_nvidia_smi()
    if nvidia_info["available"]:
        print("✅ NVIDIA-SMI is working")
        for gpu in nvidia_info.get("gpu_info", []):
            print(f"   {gpu['gpu_id']}: {gpu['memory_info']}")
    else:
        print(f"❌ NVIDIA-SMI issue: {nvidia_info.get('error', 'Unknown error')}")

    # 2. Check CUDA devices
    print("\n2. CUDA Device Status:")
    cuda_info = check_cuda_devices()
    print(f"   CUDA Available: {cuda_info['cuda_available']}")
    print(f"   Device Count: {cuda_info['device_count']}")
    print(f"   Current Device: {cuda_info['current_device']}")

    for device in cuda_info.get("devices", []):
        if device.get("accessible"):
            print(f"   GPU {device['device_id']}: {device['name']}")
            print(f"     Capability: {device['capability']}")
            print(f"     Memory: {device['total_memory'] / (1024**3):.1f}GB")
            print(f"     Functional: {device.get('functional', 'Unknown')}")
        else:
            print(
                f"   GPU {device['device_id']}: ❌ Not accessible - {device.get('error', 'Unknown error')}"
            )

    # 3. Check environment variables
    print("\n3. Environment Variables:")
    env_info = check_environment_variables()
    for var, value in env_info.items():
        print(f"   {var}: {value}")

    # 4. Test memory allocation
    print("\n4. Memory Allocation Test:")
    memory_info = test_gpu_memory_allocation()
    for gpu, info in memory_info.items():
        if info.get("allocation_successful"):
            print(f"   {gpu}: ✅ Allocation successful")
            print(
                f"     Initial: {info['initial_allocated_mb']:.1f}MB allocated, {info['initial_reserved_mb']:.1f}MB reserved"
            )
            print(
                f"     After allocation: {info['final_allocated_mb']:.1f}MB allocated, {info['final_reserved_mb']:.1f}MB reserved"
            )
        else:
            print(
                f"   {gpu}: ❌ Allocation failed - {info.get('error', 'Unknown error')}"
            )

    # 5. Test multi-GPU communication
    print("\n5. Multi-GPU Communication Test:")
    comm_info = test_multi_gpu_communication()
    if "error" in comm_info:
        print(f"   {comm_info['error']}")
    else:
        print(
            f"   Cross-device communication: {'✅' if comm_info['communication_successful'] else '❌'}"
        )
        print(
            f"   Cross-device copy: {'✅' if comm_info['cross_device_copy'] else '❌'}"
        )
        print(
            f"   Cross-device operations: {'✅' if comm_info['cross_device_operations'] else '❌'}"
        )

    # 6. Summary and recommendations
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY:")

    issues_found = []

    if not cuda_info["cuda_available"]:
        issues_found.append("CUDA not available")

    if cuda_info["device_count"] < 2:
        issues_found.append("Less than 2 GPUs detected")

    for device in cuda_info.get("devices", []):
        if not device.get("accessible"):
            issues_found.append(f"GPU {device['device_id']} not accessible")
        elif not device.get("functional"):
            issues_found.append(f"GPU {device['device_id']} not functional")

    for gpu, info in memory_info.items():
        if not info.get("allocation_successful"):
            issues_found.append(f"{gpu} memory allocation failed")

    if "error" in comm_info:
        issues_found.append("Multi-GPU communication failed")

    if issues_found:
        print("❌ Issues found:")
        for issue in issues_found:
            print(f"   - {issue}")

        print("\n🔧 Recommendations:")
        if "CUDA not available" in issues_found:
            print("   - Check CUDA installation and drivers")
        if "Less than 2 GPUs detected" in issues_found:
            print("   - Verify GPU hardware is properly connected")
        if any("not accessible" in issue for issue in issues_found):
            print("   - Check GPU permissions and driver compatibility")
        if any("memory allocation failed" in issue for issue in issues_found):
            print("   - Check GPU memory availability and other processes")
        if "Multi-GPU communication failed" in issues_found:
            print("   - Check PCIe connectivity and GPU topology")
    else:
        print("✅ No issues detected - GPUs should work for multi-GPU training")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
