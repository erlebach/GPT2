"""GPU monitoring utilities for tracking GPU usage during training.

This module provides real-time monitoring of GPU utilization, memory usage,
and verification that multiple GPUs are being used in distributed training.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch


@dataclass
class GPUStats:
    """Container for GPU statistics."""

    device_id: int
    name: str
    memory_allocated: int
    memory_reserved: int
    memory_total: int
    utilization: Optional[float] = None
    temperature: Optional[float] = None
    power_usage: Optional[float] = None


class GPUMonitor:
    """Real-time GPU monitoring for training verification."""

    def __init__(self, log_interval: float = 5.0, log_file: Optional[Path] = None):
        """Initialize GPU monitor.

        Args:
            log_interval: Interval between log entries in seconds.
            log_file: Optional file to save monitoring logs.
        """
        self.log_interval = log_interval
        self.log_file = log_file
        self.monitoring = False
        self.monitor_thread = None
        self.stats_history: List[Dict[str, Any]] = []

        # Check if nvidia-ml-py is available for detailed monitoring
        self.has_nvml = self._check_nvml_availability()

    def _check_nvml_availability(self) -> bool:
        """Check if nvidia-ml-py is available for detailed monitoring.

        Returns:
            True if nvidia-ml-py is available, False otherwise.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            return True
        except ImportError:
            return False

    def get_gpu_stats(self) -> List[GPUStats]:
        """Get current GPU statistics for all devices.

        Returns:
            List of GPU statistics for each device.
        """
        if not torch.cuda.is_available():
            return []

        stats = []
        device_count = torch.cuda.device_count()

        for device_id in range(device_count):
            with torch.cuda.device(device_id):
                # Basic PyTorch stats
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                memory_total = torch.cuda.get_device_properties(device_id).total_memory
                name = torch.cuda.get_device_name(device_id)

                gpu_stat = GPUStats(
                    device_id=device_id,
                    name=name,
                    memory_allocated=memory_allocated,
                    memory_reserved=memory_reserved,
                    memory_total=memory_total,
                )

                # Add detailed stats if nvidia-ml-py is available
                if self.has_nvml:
                    try:
                        import pynvml

                        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        temp = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        power = (
                            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        )  # Convert to Watts

                        gpu_stat.utilization = util.gpu
                        gpu_stat.temperature = temp
                        gpu_stat.power_usage = power
                    except Exception:
                        pass

                stats.append(gpu_stat)

        return stats

    def print_gpu_status(self, stats: Optional[List[GPUStats]] = None) -> None:
        """Print current GPU status.

        Args:
            stats: GPU statistics to print. If None, will get current stats.
        """
        if stats is None:
            stats = self.get_gpu_stats()

        if not stats:
            print("❌ No GPU statistics available.")
            return

        print(f"\n🖥️  GPU STATUS ({len(stats)} devices):")
        print("-" * 80)

        for stat in stats:
            memory_allocated_gb = stat.memory_allocated / (1024**3)
            memory_reserved_gb = stat.memory_reserved / (1024**3)
            memory_total_gb = stat.memory_total / (1024**3)
            memory_utilization = (stat.memory_allocated / stat.memory_total) * 100

            print(f"GPU {stat.device_id}: {stat.name}")
            print(
                f"  Memory: {memory_allocated_gb:.2f}GB / {memory_total_gb:.1f}GB ({memory_utilization:.1f}%)"
            )
            print(f"  Reserved: {memory_reserved_gb:.2f}GB")

            if stat.utilization is not None:
                print(f"  Utilization: {stat.utilization}%")
            if stat.temperature is not None:
                print(f"  Temperature: {stat.temperature}°C")
            if stat.power_usage is not None:
                print(f"  Power: {stat.power_usage:.1f}W")
            print()

    def verify_multi_gpu_usage(
        self, min_memory_threshold_mb: int = 100
    ) -> Dict[str, Any]:
        """Verify that multiple GPUs are being used.

        Args:
            min_memory_threshold_mb: Minimum memory usage to consider GPU as "used".

        Returns:
            Dictionary with verification results.
        """
        stats = self.get_gpu_stats()

        if len(stats) < 2:
            return {
                "multi_gpu_available": False,
                "reason": f"Only {len(stats)} GPU(s) available",
                "gpus_used": len(stats),
                "total_gpus": len(stats),
            }

        # Check which GPUs have significant memory usage
        used_gpus = []
        for stat in stats:
            memory_mb = stat.memory_allocated / (1024**2)
            if memory_mb > min_memory_threshold_mb:
                used_gpus.append(stat.device_id)

        result = {
            "multi_gpu_available": len(stats) >= 2,
            "gpus_used": len(used_gpus),
            "total_gpus": len(stats),
            "used_gpu_ids": used_gpus,
            "all_gpu_ids": [stat.device_id for stat in stats],
            "memory_usage": {
                stat.device_id: stat.memory_allocated / (1024**3) for stat in stats
            },
        }

        if len(used_gpus) >= 2:
            result["status"] = "✅ Multiple GPUs are being used"
        elif len(used_gpus) == 1:
            result["status"] = "⚠️  Only one GPU is being used"
        else:
            result["status"] = "❌ No GPUs are being used"

        return result

    def start_monitoring(self) -> None:
        """Start continuous GPU monitoring in a background thread."""
        if self.monitoring:
            print("⚠️  Monitoring is already running.")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"🚀 Started GPU monitoring (interval: {self.log_interval}s)")

    def stop_monitoring(self) -> None:
        """Stop continuous GPU monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("🛑 Stopped GPU monitoring")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                stats = self.get_gpu_stats()
                timestamp = time.time()

                # Create log entry
                log_entry = {
                    "timestamp": timestamp,
                    "gpu_stats": [
                        {
                            "device_id": stat.device_id,
                            "memory_allocated_gb": stat.memory_allocated / (1024**3),
                            "memory_reserved_gb": stat.memory_reserved / (1024**3),
                            "memory_total_gb": stat.memory_total / (1024**3),
                            "utilization": stat.utilization,
                            "temperature": stat.temperature,
                            "power_usage": stat.power_usage,
                        }
                        for stat in stats
                    ],
                }

                self.stats_history.append(log_entry)

                # Save to file if specified
                if self.log_file:
                    self._save_log_entry(log_entry)

                time.sleep(self.log_interval)

            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(self.log_interval)

    def _save_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """Save a log entry to file.

        Args:
            log_entry: Log entry to save.
        """
        try:
            import json

            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"⚠️  Failed to save log entry: {e}")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of monitoring data.

        Returns:
            Dictionary with monitoring summary.
        """
        if not self.stats_history:
            return {"error": "No monitoring data available"}

        # Calculate averages
        total_entries = len(self.stats_history)
        avg_memory_usage = {}
        max_memory_usage = {}

        for entry in self.stats_history:
            for gpu_stat in entry["gpu_stats"]:
                device_id = gpu_stat["device_id"]
                memory_gb = gpu_stat["memory_allocated_gb"]

                if device_id not in avg_memory_usage:
                    avg_memory_usage[device_id] = []
                    max_memory_usage[device_id] = 0

                avg_memory_usage[device_id].append(memory_gb)
                max_memory_usage[device_id] = max(
                    max_memory_usage[device_id], memory_gb
                )

        # Calculate averages
        for device_id in avg_memory_usage:
            avg_memory_usage[device_id] = sum(avg_memory_usage[device_id]) / len(
                avg_memory_usage[device_id]
            )

        return {
            "total_entries": total_entries,
            "monitoring_duration": self.stats_history[-1]["timestamp"]
            - self.stats_history[0]["timestamp"],
            "avg_memory_usage": avg_memory_usage,
            "max_memory_usage": max_memory_usage,
            "gpus_monitored": list(avg_memory_usage.keys()),
        }

    def print_monitoring_summary(self) -> None:
        """Print a summary of monitoring data."""
        summary = self.get_monitoring_summary()

        if "error" in summary:
            print(f"❌ {summary['error']}")
            return

        print(f"\n📊 MONITORING SUMMARY:")
        print(f"   Duration: {summary['monitoring_duration']:.1f} seconds")
        print(f"   Entries: {summary['total_entries']}")
        print(f"   GPUs Monitored: {summary['gpus_monitored']}")

        print(f"\n   Memory Usage Summary:")
        for device_id in summary["gpus_monitored"]:
            avg_gb = summary["avg_memory_usage"][device_id]
            max_gb = summary["max_memory_usage"][device_id]
            print(f"     GPU {device_id}: Avg {avg_gb:.2f}GB, Max {max_gb:.2f}GB")


try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    Callback = None


class GPUMonitorCallback(Callback):
    """Lightning callback for GPU monitoring."""

    def __init__(self, log_interval: float = 5.0):
        if Callback is None:
            raise ImportError("Lightning not available, cannot create callback")
        super().__init__()
        self.monitor = GPUMonitor(log_interval=log_interval)
        self.last_log_time = 0

    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        print("🚀 Training started - GPU monitoring active")
        self.monitor.start_monitoring()

    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        self.monitor.stop_monitoring()
        self.monitor.print_monitoring_summary()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Called at the start of each training batch."""
        current_time = time.time()
        if current_time - self.last_log_time >= self.monitor.log_interval:
            self.monitor.print_gpu_status()
            self.last_log_time = current_time


def create_gpu_monitor_callback(
    log_interval: float = 5.0,
) -> Optional[GPUMonitorCallback]:
    """Create a Lightning callback for GPU monitoring.

    Args:
        log_interval: Interval between GPU status logs.

    Returns:
        GPU monitor callback for Lightning.
    """
    try:
        return GPUMonitorCallback(log_interval)
    except ImportError:
        print("⚠️  Lightning not available, cannot create callback")
        return None


if __name__ == "__main__":
    # Test the GPU monitor
    monitor = GPUMonitor(log_interval=2.0)

    print("🔍 Testing GPU Monitor...")

    # Get current stats
    stats = monitor.get_gpu_stats()
    monitor.print_gpu_status(stats)

    # Verify multi-GPU usage
    verification = monitor.verify_multi_gpu_usage()
    print(f"\n🔍 Multi-GPU Verification:")
    print(f"   Status: {verification['status']}")
    print(f"   GPUs Used: {verification['gpus_used']}/{verification['total_gpus']}")
    print(f"   Used GPU IDs: {verification['used_gpu_ids']}")

    # Test monitoring for a short period
    print(f"\n📊 Starting short monitoring test...")
    monitor.start_monitoring()
    time.sleep(10)  # Monitor for 10 seconds
    monitor.stop_monitoring()

    print("✅ GPU monitor test completed!")
