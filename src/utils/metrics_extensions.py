# bridge/utils/metric_extensions.py

import csv
import functools
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

try:
    import psutil
except ImportError:
    psutil = None

from utils.memory_utils import (
    check_cuda_memory,
    track_differential_memory,
    track_gpu_memory,
)


class MetricsCollector:
    """Global metrics collector for all decorated functions.

    This class provides a centralized way to collect and manage metrics
    from all functions decorated with @measure_performance.
    """

    def __init__(self):
        """Initialize the metrics collector."""
        self._all_metrics: List[Dict[str, Any]] = []
        self._function_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._call_count = 0

    def add_metrics(self, func_name: str, metrics: Dict[str, Any]) -> None:
        """Add metrics for a function call.

        Args:
            func_name: Name of the function that generated the metrics.
            metrics: Dictionary containing the metrics data.
        """
        # Add function name and timestamp to metrics
        enhanced_metrics = metrics.copy()
        enhanced_metrics["function_name"] = func_name
        enhanced_metrics["timestamp"] = datetime.now().isoformat()
        enhanced_metrics["call_id"] = self._call_count

        # Store in global list
        self._all_metrics.append(enhanced_metrics)

        # Store in function-specific list
        if func_name not in self._function_metrics:
            self._function_metrics[func_name] = []
        self._function_metrics[func_name].append(enhanced_metrics)

        self._call_count += 1

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get all collected metrics.

        Returns:
            List of all metrics dictionaries.
        """
        return self._all_metrics.copy()

    def get_function_metrics(self, func_name: str) -> List[Dict[str, Any]]:
        """Get metrics for a specific function.

        Args:
            func_name: Name of the function.

        Returns:
            List of metrics dictionaries for the specified function.
        """
        return self._function_metrics.get(func_name, []).copy()

    def get_function_names(self) -> List[str]:
        """Get list of all function names that have metrics.

        Returns:
            List of function names.
        """
        return list(self._function_metrics.keys())

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self._all_metrics.clear()
        self._function_metrics.clear()
        self._call_count = 0

    def save_all_metrics_to_csv(self, filename: str) -> None:
        """Save all collected metrics to a CSV file.

        Args:
            filename: Path to the CSV file to save metrics.
        """
        if not self._all_metrics:
            print("No metrics to save.")
            return

        # Get all unique field names
        fieldnames = set()
        for metrics in self._all_metrics:
            fieldnames.update(metrics.keys())
        fieldnames = sorted(list(fieldnames))

        # Write to CSV
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in self._all_metrics:
                # Fill missing fields with None
                row = {field: metrics.get(field, None) for field in fieldnames}
                writer.writerow(row)

        print(f"Saved {len(self._all_metrics)} metrics to {filename}")

    def save_function_metrics_to_csv(self, func_name: str, filename: str) -> None:
        """Save metrics for a specific function to a CSV file.

        Args:
            func_name: Name of the function.
            filename: Path to the CSV file to save metrics.
        """
        metrics = self.get_function_metrics(func_name)
        if not metrics:
            print(f"No metrics found for function '{func_name}'.")
            return

        # Get all unique field names
        fieldnames = set()
        for metric in metrics:
            fieldnames.update(metric.keys())
        fieldnames = sorted(list(fieldnames))

        # Write to CSV
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metric in metrics:
                # Fill missing fields with None
                row = {field: metric.get(field, None) for field in fieldnames}
                writer.writerow(row)

        print(f"Saved {len(metrics)} metrics for '{func_name}' to {filename}")


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        The global MetricsCollector instance.
    """
    return _metrics_collector


def get_gpu_memory_metrics() -> dict:
    """Get current and peak GPU memory usage in MB for current device only.

    Returns:
        A dictionary with current and peak GPU memory usage in MB.
    """
    import torch

    if not torch.cuda.is_available():
        return {}

    # Get current device to measure only this GPU's memory
    current_device = torch.cuda.current_device()
    current = torch.cuda.memory_allocated(current_device) / 1024**2
    peak = torch.cuda.max_memory_allocated(current_device) / 1024**2

    return {
        "gpu_memory_current_mb": current,
        "gpu_memory_peak_mb": peak,
    }


def measure_performance(memory_enabled: bool = True, timing_enabled: bool = False):
    """Measure time and (optionally) memory usage of a function.

    Args:
        timing_enabled: Whether to enable timing measurements.
        memory_enabled: Whether to enable CPU and GPU memory
            measurements (requires psutil for the CPU).

    Returns:
        Decorated function with timing and memory metrics attached as an attribute.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics = {}
            gpu_available = torch.cuda.is_available()
            if memory_enabled or timing_enabled:
                # CPU memory
                if memory_enabled and psutil is not None:
                    process = psutil.Process(os.getpid())
                    start_cpu_mem = process.memory_info().rss / 1024**2

                # GPU memory - FIXED: Only measure current device
                if memory_enabled and gpu_available:
                    current_device = torch.cuda.current_device()
                    torch.cuda.empty_cache()
                    # Only reset peak stats for current device
                    torch.cuda.reset_peak_memory_stats(current_device)
                    start_gpu_mem = (
                        torch.cuda.memory_allocated(current_device) / 1024**2
                    )

                start_time = time.time()

                # Call the function to monitor
                result = func(*args, **kwargs)

                end_time = time.time()

                # CPU memory
                if memory_enabled and psutil is not None:
                    end_cpu_mem = process.memory_info().rss / 1024**2
                    metrics["cpu_memory_usage_mb"] = end_cpu_mem - start_cpu_mem
                    metrics["cpu_memory_rss_mb"] = end_cpu_mem

                # GPU memory (use utility)
                if memory_enabled and gpu_available:
                    gpu_metrics = get_gpu_memory_metrics()
                    metrics.update(gpu_metrics)
                    metrics["gpu_memory_usage_mb"] = (
                        gpu_metrics["gpu_memory_current_mb"] - start_gpu_mem
                    )

                if timing_enabled:
                    metrics["step_time_sec"] = end_time - start_time
            else:
                result = func(*args, **kwargs)

            # Store metrics locally on the wrapper
            wrapper.last_metrics = metrics
            wrapper.metrics.append(metrics.copy())

            # Add to global collector
            func_name = f"{func.__module__}.{func.__qualname__}"
            _metrics_collector.add_metrics(func_name, metrics)

            return result

        wrapper.last_metrics = {}
        wrapper.metrics = []
        return wrapper

    return decorator


def measure_timing():
    """Measure only timing of a function.

    Returns:
        Decorated function with timing metrics attached.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics = {}

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            metrics["step_time_sec"] = end_time - start_time

            # Store metrics locally on the wrapper
            wrapper.last_metrics = metrics
            wrapper.metrics.append(metrics.copy())

            # Add to global collector
            func_name = f"{func.__module__}.{func.__qualname__}"
            _metrics_collector.add_metrics(func_name, metrics)

            return result

        wrapper.last_metrics = {}
        wrapper.metrics = []
        return wrapper

    return decorator


def measure_memory():
    """Measure only memory usage of a function.

    Returns:
        Decorated function with memory metrics attached.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics = {}
            gpu_available = torch.cuda.is_available()

            # CPU memory
            if psutil is not None:
                process = psutil.Process(os.getpid())
                start_cpu_mem = process.memory_info().rss / 1024**2

            # GPU memory
            if gpu_available:
                current_device = torch.cuda.current_device()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(current_device)
                start_gpu_mem = torch.cuda.memory_allocated(current_device) / 1024**2

            # Call the function
            result = func(*args, **kwargs)

            # CPU memory
            if psutil is not None:
                end_cpu_mem = process.memory_info().rss / 1024**2
                metrics["cpu_memory_usage_mb"] = end_cpu_mem - start_cpu_mem
                metrics["cpu_memory_rss_mb"] = end_cpu_mem

            # GPU memory
            if gpu_available:
                gpu_metrics = get_gpu_memory_metrics()
                metrics.update(gpu_metrics)
                metrics["gpu_memory_usage_mb"] = (
                    gpu_metrics["gpu_memory_current_mb"] - start_gpu_mem
                )

            # Store metrics locally on the wrapper
            wrapper.last_metrics = metrics
            wrapper.metrics.append(metrics.copy())

            # Add to global collector
            func_name = f"{func.__module__}.{func.__qualname__}"
            _metrics_collector.add_metrics(func_name, metrics)

            return result

        wrapper.last_metrics = {}
        wrapper.metrics = []
        return wrapper

    return decorator


class MetricsModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint with performance measurement capabilities.

    This callback extends the standard ModelCheckpoint to include
    performance metrics for checkpoint operations.
    """


def __init__(self, *args, **kwargs):
    """Initialize the metrics checkpoint callback.

    Args:
        *args: Arguments to pass to ModelCheckpoint.
        **kwargs: Keyword arguments to pass to ModelCheckpoint.
    """
    super().__init__(*args, **kwargs)


@measure_performance(memory_enabled=True, timing_enabled=True)
def _save_checkpoint(self, trainer, filepath: str) -> None:
    """Save checkpoint with performance measurement.

    Args:
        trainer: The trainer instance.
        filepath: Path where to save the checkpoint.
    """
    return super()._save_checkpoint(trainer, filepath)


@measure_performance(memory_enabled=True, timing_enabled=True)
def on_save_checkpoint(
    self, trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
) -> Dict[str, Any]:
    """Save checkpoint callback state with performance measurement.

    Args:
        trainer: The trainer instance.
        pl_module: The lightning module.
        checkpoint: The checkpoint dictionary.

    Returns:
        Dictionary containing the checkpoint callback state.
    """
    return super().on_save_checkpoint(trainer, pl_module, checkpoint)


@measure_performance(memory_enabled=True, timing_enabled=True)
def on_load_checkpoint(
    self, trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
) -> None:
    """Load checkpoint callback state with performance measurement.

    Args:
        trainer: The trainer instance.
        pl_module: The lightning module.
        checkpoint: The checkpoint dictionary.
    """
    return super().on_load_checkpoint(trainer, pl_module, checkpoint)


def save_metrics_to_csv(filename: str, metrics: dict):
    """Append memory and timing metrics to a CSV file, rounding floats to 5 significant digits.

    Only keys containing 'memory' or 'time' are saved.
    """


import csv
import os


def round_floats(d):
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}


# Only keep keys related to memory or timing
filtered_metrics = {k: v for k, v in metrics.items() if "memory" in k or "time" in k}
filtered_metrics = round_floats(filtered_metrics)

if not filtered_metrics:
    print("No memory or timing metrics to save.")

file_exists = os.path.isfile(filename)
with open(filename, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=filtered_metrics.keys())
    if not file_exists or os.stat(filename).st_size == 0:
        writer.writeheader()
    writer.writerow(filtered_metrics)


# Add test cases for both regular functions and class methods
@measure_performance(memory_enabled=True, timing_enabled=True)
def regular_function(n: int) -> float:
    """A regular function to test the decorator."""
    x = [i**2 for i in range(n)]
    return sum(x) / n


class MyClass:
    @measure_performance(memory_enabled=True, timing_enabled=True)
    def compute(self, n: int) -> float:
        """A class method to test the decorator."""
        x = [i**2 for i in range(n)]
        return sum(x) / n


if __name__ == "__main__":
    # Test with regular function
    result1 = regular_function(1000000)
    print("Regular function result:", result1)
    print("Regular function metrics:", regular_function.last_metrics)

    # Test with class method
    obj = MyClass()
    result2 = obj.compute(1000000)
    print("Class method result:", result2)
    print("Class method metrics:", obj.compute.last_metrics)

    # Test with another call to show accumulation
    result3 = regular_function(500000)
    print("Second call result:", result3)

    # Get global metrics collector
    collector = get_metrics_collector()
    print("All function names:", collector.get_function_names())
    print("Total metrics collected:", len(collector.get_all_metrics()))

    # Save all metrics to CSV
    collector.save_all_metrics_to_csv("all_metrics.csv")

    # Save metrics for specific function
    collector.save_function_metrics_to_csv(
        "__main__.regular_function", "regular_function_metrics.csv"
    )
