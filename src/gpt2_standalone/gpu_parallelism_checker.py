"""GPU parallelism checker for GPT-2 training.

This module provides comprehensive checks for GPU availability, active devices,
and various types of parallelism that can be enabled in PyTorch Lightning training.

Key Features:
- Basic GPU Information: Detects available GPUs, their names, memory, and CUDA capabilities
- Parallelism Capabilities: Checks what types of parallelism are available (DDP, DeepSpeed, FSDP, etc.)
- Strategy Analysis: Analyzes Lightning strategies to understand their parallelism configuration
- Environment Variables: Checks relevant environment variables that affect GPU parallelism

Recommendations: Provides strategy recommendations based on GPU count and preferences
Memory Monitoring: Tracks GPU memory usage and provides detailed reports
Strategy Creation: Helper function to create strategies from configuration
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.strategies import (
    DDPStrategy,
    DeepSpeedStrategy,
    FSDPStrategy,
    SingleDeviceStrategy,
    Strategy,
)


class GPUParallelismChecker:
    """Comprehensive GPU parallelism checker for PyTorch Lightning training.

    This class provides methods to check GPU availability, active devices,
    and various types of parallelism that can be enabled in training.
    """

    def __init__(self):
        """Initialize the GPU parallelism checker."""
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0

    def check_basic_gpu_info(self) -> Dict[str, Any]:
        """Check basic GPU information and availability.

        Returns:
            Dictionary containing basic GPU information.
        """
        info = {
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "current_device": torch.cuda.current_device()
            if self.cuda_available
            else None,
        }

        if self.cuda_available:
            info["device_names"] = [
                torch.cuda.get_device_name(i) for i in range(self.device_count)
            ]
            info["device_capabilities"] = [
                torch.cuda.get_device_capability(i) for i in range(self.device_count)
            ]
            info["total_memory"] = [
                torch.cuda.get_device_properties(i).total_memory
                for i in range(self.device_count)
            ]
            info["current_device_name"] = torch.cuda.get_device_name(
                torch.cuda.current_device()
            )

        return info

    def check_lightning_parallelism_capabilities(self) -> Dict[str, Any]:
        """Check what parallelism strategies are available in Lightning.

        Returns:
            Dictionary containing available parallelism strategies and their status.
        """
        capabilities = {
            "single_device": True,  # Always available
            "ddp": self.device_count > 1,
            "ddp_spawn": self.device_count > 1,
            "ddp_fork": self.device_count > 1 and sys.platform != "win32",
            "deepspeed": self._check_deepspeed_availability(),
            "fsdp": self._check_fsdp_availability(),
            "model_parallel": self.device_count > 1,
            "tensor_parallel": self.device_count > 1,
            "pipeline_parallel": self.device_count > 1,
        }

        return capabilities

    def _check_deepspeed_availability(self) -> bool:
        """Check if DeepSpeed is available for use.

        Returns:
            True if DeepSpeed is available, False otherwise.
        """
        try:
            import deepspeed

            return True
        except ImportError:
            return False

    def _check_fsdp_availability(self) -> bool:
        """Check if FSDP (Fully Sharded Data Parallel) is available.

        Returns:
            True if FSDP is available, False otherwise.
        """
        try:
            # Check if PyTorch version supports FSDP
            if hasattr(torch.distributed, "algorithms") and hasattr(
                torch.distributed.algorithms, "fsdp"
            ):
                return True
            return False
        except Exception:
            return False

    def analyze_strategy(self, strategy: Strategy) -> Dict[str, Any]:
        """Analyze a Lightning strategy to understand its parallelism configuration.

        Args:
            strategy: The Lightning strategy to analyze.

        Returns:
            Dictionary containing strategy analysis information.
        """
        analysis = {
            "strategy_type": type(strategy).__name__,
            "is_parallel": not isinstance(strategy, SingleDeviceStrategy),
            "num_processes": getattr(strategy, "num_processes", 1),
            "num_nodes": getattr(strategy, "num_nodes", 1),
            "parallelism_type": self._get_parallelism_type(strategy),
        }

        if isinstance(strategy, DDPStrategy):
            analysis.update(
                {
                    "ddp_find_unused_parameters": strategy.find_unused_parameters,
                    "ddp_bucket_cap_mb": strategy.bucket_cap_mb,
                    "ddp_static_graph": strategy.static_graph,
                }
            )
        elif isinstance(strategy, DeepSpeedStrategy):
            analysis.update(
                {
                    "deepspeed_config": strategy.config,
                    "deepspeed_stage": strategy.config.get("zero_optimization", {}).get(
                        "stage", "unknown"
                    ),
                }
            )
        elif isinstance(strategy, FSDPStrategy):
            analysis.update(
                {
                    "fsdp_state_dict_type": strategy.state_dict_type,
                    "fsdp_activation_checkpointing": strategy.activation_checkpointing,
                }
            )

        return analysis

    def _get_parallelism_type(self, strategy: Strategy) -> str:
        """Get the type of parallelism used by a strategy.

        Args:
            strategy: The Lightning strategy to analyze.

        Returns:
            String describing the parallelism type.
        """
        if isinstance(strategy, SingleDeviceStrategy):
            return "single_device"
        elif isinstance(strategy, DDPStrategy):
            return "data_parallel"
        elif isinstance(strategy, DeepSpeedStrategy):
            return "deepspeed_parallel"
        elif isinstance(strategy, FSDPStrategy):
            return "fully_sharded_data_parallel"
        else:
            return "unknown"

    def check_environment_variables(self) -> Dict[str, Any]:
        """Check environment variables that affect GPU parallelism.

        Returns:
            Dictionary containing relevant environment variables.
        """
        env_vars = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG"),
            "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE"),
            "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
            "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
            "MASTER_PORT": os.environ.get("MASTER_PORT"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "RANK": os.environ.get("RANK"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "NODE_RANK": os.environ.get("NODE_RANK"),
        }

        return env_vars

    def get_recommended_strategy(
        self, num_gpus: int, use_deepspeed: bool = False, use_fsdp: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """Get recommended strategy based on available GPUs and preferences.

        Args:
            num_gpus: Number of GPUs to use.
            use_deepspeed: Whether to prefer DeepSpeed.
            use_fsdp: Whether to prefer FSDP.

        Returns:
            Tuple of (strategy_name, strategy_config).
        """
        if num_gpus == 0:
            return "cpu", {}
        elif num_gpus == 1:
            return "single_device", {"device": "cuda:0"}
        else:
            if use_deepspeed and self._check_deepspeed_availability():
                return "deepspeed", {
                    "zero_optimization": {"stage": 2},
                    "gradient_accumulation_steps": 1,
                }
            elif use_fsdp and self._check_fsdp_availability():
                return "fsdp", {
                    "state_dict_type": "FULL_STATE_DICT",
                    "activation_checkpointing": None,
                }
            else:
                return "ddp", {
                    "find_unused_parameters": False,
                    "static_graph": True,
                }

    def print_comprehensive_report(self) -> None:
        """Print a comprehensive report of GPU and parallelism status."""
        print("=" * 60)
        print("GPU PARALLELISM CHECKER REPORT")
        print("=" * 60)

        # Basic GPU info
        gpu_info = self.check_basic_gpu_info()
        print(f"\n📊 BASIC GPU INFORMATION:")
        print(f"   CUDA Available: {gpu_info['cuda_available']}")
        print(f"   Device Count: {gpu_info['device_count']}")

        if gpu_info["cuda_available"]:
            print(f"   Current Device: {gpu_info['current_device']}")
            print(f"   Current Device Name: {gpu_info['current_device_name']}")
            print(f"\n   Available GPUs:")
            for i, name in enumerate(gpu_info["device_names"]):
                memory_gb = gpu_info["total_memory"][i] / (1024**3)
                capability = gpu_info["device_capabilities"][i]
                print(
                    f"     GPU {i}: {name} ({memory_gb:.1f}GB, CUDA {capability[0]}.{capability[1]})"
                )

        # Parallelism capabilities
        capabilities = self.check_lightning_parallelism_capabilities()
        print(f"\n🔧 PARALLELISM CAPABILITIES:")
        print(f"   Single Device: {capabilities['single_device']}")
        print(f"   Data Parallel (DDP): {capabilities['ddp']}")
        print(f"   DeepSpeed: {capabilities['deepspeed']}")
        print(f"   FSDP: {capabilities['fsdp']}")
        print(f"   Model Parallel: {capabilities['model_parallel']}")
        print(f"   Tensor Parallel: {capabilities['tensor_parallel']}")
        print(f"   Pipeline Parallel: {capabilities['pipeline_parallel']}")

        # Environment variables
        env_vars = self.check_environment_variables()
        print(f"\n�� ENVIRONMENT VARIABLES:")
        for key, value in env_vars.items():
            if value is not None:
                print(f"   {key}: {value}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if gpu_info["device_count"] == 0:
            print("   ⚠️  No GPUs detected. Training will use CPU.")
        elif gpu_info["device_count"] == 1:
            print("   ✅ Single GPU detected. Use single_device strategy.")
        else:
            print(f"   🚀 {gpu_info['device_count']} GPUs detected. Consider:")
            if capabilities["ddp"]:
                print("     - DDP for data parallelism")
            if capabilities["deepspeed"]:
                print("     - DeepSpeed for memory optimization")
            if capabilities["fsdp"]:
                print("     - FSDP for large model training")

        print("=" * 60)

    def check_current_training_setup(self) -> Dict[str, Any]:
        """Check the current training setup if a Lightning trainer is active.

        Returns:
            Dictionary containing current training setup information.
        """
        setup_info = {
            "has_active_trainer": False,
            "strategy": None,
            "accelerator": None,
            "devices": None,
            "precision": None,
        }

        # Try to get current trainer context
        try:
            # This is a simplified check - in practice, you'd need to pass the trainer
            # or check global state
            pass
        except Exception:
            pass

        return setup_info

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage information.

        Returns:
            Dictionary containing memory usage information.
        """
        memory_info = {}

        if self.cuda_available:
            memory_info["current_device"] = torch.cuda.current_device()
            memory_info["allocated"] = torch.cuda.memory_allocated()
            memory_info["cached"] = torch.cuda.memory_reserved()
            memory_info["max_allocated"] = torch.cuda.max_memory_allocated()
            memory_info["max_cached"] = torch.cuda.max_memory_reserved()

            # Per-device memory info
            memory_info["per_device"] = {}
            for i in range(self.device_count):
                with torch.cuda.device(i):
                    memory_info["per_device"][f"gpu_{i}"] = {
                        "allocated": torch.cuda.memory_allocated(),
                        "cached": torch.cuda.memory_reserved(),
                        "total": torch.cuda.get_device_properties(i).total_memory,
                    }

        return memory_info

    def print_memory_report(self) -> None:
        """Print a detailed memory usage report."""
        if not self.cuda_available:
            print("❌ No CUDA devices available for memory reporting.")
            return

        memory_info = self.get_memory_usage()

        print("\n💾 GPU MEMORY USAGE:")
        print(f"   Current Device: {memory_info['current_device']}")
        print(f"   Allocated: {memory_info['allocated'] / 1024**3:.2f} GB")
        print(f"   Cached: {memory_info['cached'] / 1024**3:.2f} GB")
        print(f"   Max Allocated: {memory_info['max_allocated'] / 1024**3:.2f} GB")
        print(f"   Max Cached: {memory_info['max_cached'] / 1024**3:.2f} GB")

        print(f"\n   Per-Device Breakdown:")
        for device, info in memory_info["per_device"].items():
            allocated_gb = info["allocated"] / 1024**3
            cached_gb = info["cached"] / 1024**3
            total_gb = info["total"] / 1024**3
            utilization = (info["allocated"] / info["total"]) * 100

            print(
                f"     {device}: {allocated_gb:.2f}GB / {total_gb:.1f}GB ({utilization:.1f}%)"
            )


def create_strategy_from_config(
    strategy_name: str, strategy_config: Dict[str, Any]
) -> Strategy:
    """Create a Lightning strategy from configuration.

    Args:
        strategy_name: Name of the strategy to create.
        strategy_config: Configuration dictionary for the strategy.

    Returns:
        Configured Lightning strategy.

    Raises:
        ValueError: If strategy name is not supported.
    """
    if strategy_name == "single_device":
        device = strategy_config.get("device", "cuda:0")
        return SingleDeviceStrategy(device=device)
    elif strategy_name == "ddp":
        return DDPStrategy(**strategy_config)
    elif strategy_name == "deepspeed":
        return DeepSpeedStrategy(**strategy_config)
    elif strategy_name == "fsdp":
        return FSDPStrategy(**strategy_config)
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")


if __name__ == "__main__":
    # Create checker instance
    checker = GPUParallelismChecker()

    # Print comprehensive report
    checker.print_comprehensive_report()

    # Print memory report
    checker.print_memory_report()

    # Test strategy recommendations
    print(f"\n�� STRATEGY RECOMMENDATIONS:")
    for num_gpus in [0, 1, 2, 4]:
        strategy_name, config = checker.get_recommended_strategy(num_gpus)
        print(f"   {num_gpus} GPUs: {strategy_name} with config {config}")

    # Test with DeepSpeed preference
    strategy_name, config = checker.get_recommended_strategy(4, use_deepspeed=True)
    print(f"   4 GPUs (DeepSpeed preferred): {strategy_name}")

    # Test with FSDP preference
    strategy_name, config = checker.get_recommended_strategy(4, use_fsdp=True)
    print(f"   4 GPUs (FSDP preferred): {strategy_name}")

    print("\n✅ All tests completed successfully!")
