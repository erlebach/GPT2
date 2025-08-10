# src/experiments/memory_measurements_generic.py
from __future__ import annotations

import csv
import importlib
import inspect
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Optional

import torch
from jaxtyping import Int
from lightning import Fabric
from torch import Tensor, nn

from experiments.clean_palate import deep_gpu_reset, reset_model_state


class LineNo:
    def __str__(self):
        return str(inspect.currentframe().f_back.f_lineno)


# __line__ = LineNo()


class ExceptionLine:
    def __init__(self, exc):
        tb = exc.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        self.lineno = tb.tb_lineno

    def __str__(self):
        return str(self.lineno)


def _cuda_on() -> bool:
    """Check if CUDA is available.

    Returns:
        True if CUDA is available; False otherwise.

    """
    return torch.cuda.is_available()


def _cuda_reset_peak() -> None:
    """Reset CUDA peak memory stats if available."""
    if _cuda_on():
        torch.cuda.reset_peak_memory_stats()


def _cuda_mem_allocated() -> int:
    """Return current allocated CUDA memory in bytes (0 on CPU)."""
    return int(torch.cuda.memory_allocated()) if _cuda_on() else 0


def _cuda_mem_reserved() -> int:
    """Return current reserved CUDA memory in bytes (0 on CPU)."""
    return int(torch.cuda.memory_reserved()) if _cuda_on() else 0


def _cuda_mem_max_allocated() -> int:
    """Return peak allocated CUDA memory in bytes (0 on CPU)."""
    return int(torch.cuda.max_memory_allocated()) if _cuda_on() else 0


def memory_measurement(func: Callable[..., Any]) -> Callable[..., dict]:
    """Run decorator to measure GPU memory usage around a function call.

    Args:
        func: Function to wrap. Must return any value.

    Returns:
        A function that returns a dict including memory measurements merged
        with the original function result (if it was a dict), or a dict with
        the function result under 'function_result' otherwise.

    """

    def wrapper(*args, **kwargs) -> dict:
        """Execute wrapped function with standardized memory measurements."""
        _cuda_reset_peak()
        start_alloc = _cuda_mem_allocated()

        result = func(*args, **kwargs)

        mem_allocated_bytes = _cuda_mem_allocated()
        peak_mem = _cuda_mem_max_allocated()
        mem_reserved_bytes = _cuda_mem_reserved()
        mem_cached_bytes = max(mem_reserved_bytes - mem_allocated_bytes, 0)

        memory_measurements = {
            "memory_gb": mem_allocated_bytes / 1e9,
            "mem_alloc_gb": mem_allocated_bytes / 1e9,
            "mem_peak_gb": peak_mem / 1e9,
            "mem_cached_gb": mem_cached_bytes / 1e9,
            "mem_delta_gb": (mem_allocated_bytes - start_alloc) / 1e9,
        }

        if isinstance(result, dict):
            result.update(memory_measurements)
        else:
            result = {"function_result": result, **memory_measurements}

        return result

    return wrapper


def _call_model(model: nn.Module, x: Tensor, y: Optional[Tensor] = None) -> Any:
    """Call model(x) or model(x, y) depending on signature.

    Args:
        model: The model.
        x: Input tensor.
        y: Optional target tensor.

    Returns:
        Model output.

    """
    try:
        return model(x, y)
    except TypeError:
        return model(x)


@memory_measurement
def run_inference_generic(model: nn.Module, x: Int[Tensor, "b t"]) -> dict:
    """Run inference without gradients and measure memory.

    Args:
        model: PyTorch model.
        x: Input batch tokens.

    Returns:
        Dictionary with operation label and memory stats.

    """
    model.eval()
    with torch.no_grad():
        _ = _call_model(model, x)
    return {"operation": "inference"}


@memory_measurement
def run_forward_with_gradients_generic(
    model: nn.Module, x: Int[Tensor, "b t"], y: Optional[Int[Tensor, "b t"]] = None
) -> dict:
    """Run forward with gradients enabled (no loss/backward) and measure memory.

    Args:
        model: PyTorch model.
        x: Input batch tokens.
        y: Optional target tokens if model forward supports it.

    Returns:
        Dictionary with operation label and memory stats.

    """
    model.train()
    _ = _call_model(model, x, y)
    return {"operation": "forward_with_gradients"}


@memory_measurement
def run_training_step_generic(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Int[Tensor, "b t"],
    y: Int[Tensor, "b t"],
    vocab_size: int,
    loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> dict:
    """Run a full training step (forward + loss + backward) and measure memory.

    If the model has a `training_step((x, y), batch_idx)` method, it will be
    used; otherwise, a standard CE loss on logits is computed.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        x: Input batch tokens.
        y: Target batch tokens.
        vocab_size: Vocabulary size for reshaping logits.
        loss_fn: Optional loss function. Defaults to CrossEntropyLoss.

    Returns:
        Dictionary with operation label and memory stats.

    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    if hasattr(model, "training_step"):
        loss = model.training_step((x, y), batch_idx=0)
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
    else:
        logits = _call_model(model, x, y)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        logits_flat = logits.view(-1, vocab_size)
        y_flat = y.view(-1)
        loss = loss_fn(logits_flat, y_flat)

    assert isinstance(loss, Tensor)
    loss.backward()
    return {"operation": "training_step"}


@dataclass
class ModelBuildSpec:
    """Specification used by a model factory.

    Attributes:
        name: Human-friendly model name for reporting.
        vocab_size: Vocabulary size for token generation and loss.
        sequence_length: Sequence length to test.
        config: Arbitrary model configuration dictionary.

    """

    name: str
    vocab_size: int
    sequence_length: int
    config: dict


def load_from_yaml_minimal(yaml_path: str, section: str) -> dict:
    """Load a minimal Hydra-like section containing a `_target_`.

    This does not require Hydra; it just parses YAML and returns a dict.

    Args:
        yaml_path: Path to YAML file.
        section: Top-level key to load (e.g., 'model' or 'optimizer').

    Returns:
        Dictionary for that section.

    """
    import yaml

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    if section not in cfg:
        raise KeyError(f"Section '{section}' not found in {yaml_path}")
    return cfg[section]


def instantiate_from_target(target: str, kwargs: dict) -> Any:
    """Instantiate a class given a dotted path string.

    Args:
        target: Dotted path 'module.submodule.ClassName'.
        kwargs: Keyword arguments for the constructor.

    Returns:
        Instantiated object.

    """
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    # start DEBUG
    print(f"Module path: {module_path}")
    print(f"module: {module}")
    print(f"class_name: {class_name}")
    print(f"Instantiating {cls=} with {kwargs=}")
    # end DEBUG
    return cls(**kwargs)


def model_factory_from_yaml(
    yaml_path: str,
    model_key: str = "model",
    optimizer_key: str = "optimizer",
    param_key: str = "params",
) -> Callable[[ModelBuildSpec], tuple[nn.Module, torch.optim.Optimizer]]:
    """Create a factory that builds model/optimizer from a YAML `_target_`.

    The YAML is expected to contain:
      model:
        _target_: path.to.ModelClass
        arg1: ...
      optimizer:
        _target_: torch.optim.AdamW
        lr: 3e-4
        [any other optimizer args]

    Args:
        yaml_path: YAML file path.
        model_key: Key for model section.
        optimizer_key: Key for optimizer section.
        param_key: Name of the optimizer's param arg (usually 'params').

    Returns:
        A callable that builds (model, optimizer) from a ModelBuildSpec.

    """

    def factory(spec: ModelBuildSpec) -> tuple[nn.Module, torch.optim.Optimizer]:
        model_cfg = load_from_yaml_minimal(yaml_path, model_key)
        opt_cfg = load_from_yaml_minimal(yaml_path, optimizer_key)

        model_target = model_cfg.pop("_target_")
        # Allow spec.config to override/extend YAML
        model_kwargs = {**model_cfg, **spec.config}
        # Ensure sequence length and vocab can be passed if the model takes them
        model_kwargs.setdefault("block_size", spec.sequence_length)
        model_kwargs.setdefault("vocab_size", spec.vocab_size)

        model = instantiate_from_target(model_target, model_kwargs)

        opt_target = opt_cfg.pop("_target_")
        # Optimizers usually take params= first positional, but we pass as kw.
        opt_kwargs = {**opt_cfg, "params": model.parameters()}
        # Instantiate the optimizer directly with all required arguments
        optimizer = instantiate_from_target(opt_target, opt_kwargs)
        return model, optimizer

    return factory


def model_factory_from_callable(
    builder: Callable[[ModelBuildSpec], tuple[nn.Module, torch.optim.Optimizer]],
) -> Callable[[ModelBuildSpec], tuple[nn.Module, torch.optim.Optimizer]]:
    """Wrap a user-provided callable as a model factory.

    Args:
        builder: Callable that builds (model, optimizer) from a spec.

    Returns:
        The same callable, for symmetry with other factories.

    """
    return builder


def get_experiments_by_model_generic(results: dict, model_name: str) -> dict:
    """Get all experiments for a specific model.

    Args:
        results: Experiment results dictionary.
        model_name: Name of the model to filter by.

    Returns:
        Dictionary containing only experiments for the specified model.
    """
    return {
        key: experiment
        for key, experiment in results["experiments"].items()
        if experiment["model_name"] == model_name
    }


def get_experiments_by_batch_size_generic(results: dict, batch_size: int) -> dict:
    """Get all experiments for a specific batch size.

    Args:
        results: Experiment results dictionary.
        batch_size: Batch size to filter by.

    Returns:
        Dictionary containing only experiments for the specified batch size.
    """
    return {
        key: experiment
        for key, experiment in results["experiments"].items()
        if experiment["batch_size"] == batch_size
    }


def get_experiments_by_sequence_length_generic(
    results: dict, sequence_length: int
) -> dict:
    """Get all experiments for a specific sequence length.

    Args:
        results: Experiment results dictionary.
        sequence_length: Sequence length to filter by.

    Returns:
        Dictionary containing only experiments for the specified sequence length.
    """
    return {
        key: experiment
        for key, experiment in results["experiments"].items()
        if experiment["sequence_length"] == sequence_length
    }


def print_experiment_summary_generic(results: dict) -> None:
    """Print a summary of all experiments.

    Args:
        results: Experiment results dictionary.
    """
    print(f"\n📊 Memory Experiment Summary")
    print(f"   Device: {results['device']}")
    print(f"   Timestamp: {results['timestamp']}")
    print(f"   Total experiments: {len(results['experiments'])}")

    successful = sum(
        1 for exp in results["experiments"].values() if exp.get("status") == "success"
    )
    failed = len(results["experiments"]) - successful

    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")

    if successful > 0:
        print(f"\n   Memory Usage Summary (successful experiments):")
        inf_memories = [
            exp["avg_inf_memory_gb"]
            for exp in results["experiments"].values()
            if exp.get("status") == "success"
        ]
        fwd_memories = [
            exp["avg_fwd_memory_gb"]
            for exp in results["experiments"].values()
            if exp.get("status") == "success"
        ]
        ts_memories = [
            exp["avg_ts_memory_gb"]
            for exp in results["experiments"].values()
            if exp.get("status") == "success"
        ]

        print(
            f"     INF: min={min(inf_memories):.2f}GB, max={max(inf_memories):.2f}GB, avg={sum(inf_memories)/len(inf_memories):.2f}GB"
        )
        print(
            f"     FWD: min={min(fwd_memories):.2f}GB, max={max(fwd_memories):.2f}GB, avg={sum(fwd_memories)/len(fwd_memories):.2f}GB"
        )
        print(
            f"     TS:  min={min(ts_memories):.2f}GB, max={max(ts_memories):.2f}GB, avg={sum(ts_memories)/len(ts_memories):.2f}GB"
        )


def run_single_experiment_generic(
    fabric: Fabric,
    spec: ModelBuildSpec,
    batch_size: int,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
    factory: Optional[
        Callable[[ModelBuildSpec], tuple[nn.Module, torch.optim.Optimizer]]
    ] = None,
) -> dict:
    """Run a memory experiment for an arbitrary model via a factory.

    Args:
        fabric: Lightning Fabric instance.
        spec: ModelBuildSpec describing name, vocab, seq len, config.
        batch_size: Batch size to test.
        num_iterations: Number of measured iterations.
        warmup_iterations: Number of warmup iterations.
        factory: Model factory producing (model, optimizer).

    Returns:
        Dictionary with averaged memory stats and metadata.

    """
    import time

    if factory is None:
        raise ValueError("A model factory must be provided.")

    start_time = time.time()
    print(
        f"   Testing: {spec.name}, batch_size={batch_size}, seq_len={spec.sequence_length}"
    )

    try:
        print(f"     Setting up model...")
        deep_gpu_reset()

        model, optimizer = factory(spec)
        model, optimizer = fabric.setup(model, optimizer)

        total_params = sum(p.numel() for p in model.parameters())

        print(f"     Creating test data...")
        device = fabric.device
        x = torch.randint(
            0, spec.vocab_size, (batch_size, spec.sequence_length), device=device
        )
        y = torch.randint(
            0, spec.vocab_size, (batch_size, spec.sequence_length), device=device
        )

        model.train()

        print(f"     Warming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            deep_gpu_reset()
            run_forward_with_gradients_generic(model, x, y)

        print(f"     Measuring memory ({num_iterations} iterations)...", flush=True)
        inf_alloc, inf_peak, inf_cached = [], [], []
        fwd_alloc, fwd_peak, fwd_cached = [], [], []
        ts_alloc, ts_peak, ts_cached = [], [], []

        for _ in range(num_iterations):
            deep_gpu_reset()
            reset_model_state(model, optimizer)
            optimizer.zero_grad(set_to_none=True)

            inf = run_inference_generic(model, x)
            fwd = run_forward_with_gradients_generic(model, x, y)
            ts = run_training_step_generic(model, optimizer, x, y, spec.vocab_size)

            inf_alloc.append(inf["mem_alloc_gb"])
            inf_peak.append(inf["mem_peak_gb"])
            inf_cached.append(inf["mem_cached_gb"])

            fwd_alloc.append(fwd["mem_alloc_gb"])
            fwd_peak.append(fwd["mem_peak_gb"])
            fwd_cached.append(fwd["mem_cached_gb"])

            ts_alloc.append(ts["mem_alloc_gb"])
            ts_peak.append(ts["mem_peak_gb"])
            ts_cached.append(ts["mem_cached_gb"])

        import statistics as stats

        result = {
            "model_name": spec.name,
            "batch_size": batch_size,
            "sequence_length": spec.sequence_length,
            "config": spec.config,
            "total_params": total_params,
            "total_params_millions": total_params / 1e6,
            "avg_inf_memory_gb": stats.mean(inf_alloc),
            "avg_inf_cached_memory_gb": stats.mean(inf_cached),
            "avg_inf_peak_memory_gb": stats.mean(inf_peak),
            "avg_fwd_memory_gb": stats.mean(fwd_alloc),
            "avg_fwd_cached_memory_gb": stats.mean(fwd_cached),
            "avg_fwd_peak_memory_gb": stats.mean(fwd_peak),
            "avg_ts_memory_gb": stats.mean(ts_alloc),
            "avg_ts_cached_memory_gb": stats.mean(ts_cached),
            "avg_ts_peak_memory_gb": stats.mean(ts_peak),
            "peak_memory_gb": _cuda_mem_max_allocated() / 1e9,
            "status": "success",
        }

        elapsed = time.time() - start_time
        print(
            (
                f"       ✅ Completed in {elapsed:.1f}s - "
                f"INF: {result['avg_inf_memory_gb']:.2f}/{result['avg_inf_cached_memory_gb']:.2f}/"
                f"{result['avg_inf_peak_memory_gb']:.2f} GB, "
                f"FWD: {result['avg_fwd_memory_gb']:.2f}/{result['avg_fwd_cached_memory_gb']:.2f}/"
                f"{result['avg_fwd_peak_memory_gb']:.2f} GB, "
                f"TS: {result['avg_ts_memory_gb']:.2f}/{result['avg_ts_cached_memory_gb']:.2f}/"
                f"{result['avg_ts_peak_memory_gb']:.2f} GB"
            ),
            flush=True,
        )
    except RuntimeError as e:
        elapsed = time.time() - start_time
        __line__ = ExceptionLine(e)
        print(
            f"       ❌ Runtime error for {spec.name}, batch={batch_size}, "
            f"seq={spec.sequence_length} (after {elapsed:.1f}s): {e} in file {__file__}, line {__line__}",
            flush=True,
        )
        result = {
            "model_name": spec.name,
            "batch_size": batch_size,
            "sequence_length": spec.sequence_length,
            "config": spec.config,
            "total_params": 0,
            "total_params_millions": 0.0,
            "status": "runtime_error",
            "error": str(e),
        }
    except Exception as e:
        __line__ = ExceptionLine(e)
        elapsed = time.time() - start_time
        print(
            f"       ❌ Unexpected error for {spec.name}, batch={batch_size}, "
            f"seq={spec.sequence_length} (after {elapsed:.1f}s): {e} in file {__file__}, line {__line__}",
            flush=True,
        )
        result = {
            "model_name": spec.name,
            "batch_size": batch_size,
            "sequence_length": spec.sequence_length,
            "config": spec.config,
            "total_params": 0,
            "total_params_millions": 0.0,
            "status": "error",
            "error": str(e),
        }
    finally:
        with suppress(NameError):
            del model
        with suppress(NameError):
            del optimizer
        with suppress(NameError):
            del x
        with suppress(NameError):
            del y
        deep_gpu_reset()

    return result


# ----------------------------- Examples --------------------------------- #


def run_experiment_grid_generic(
    fabric: Fabric,
    model_specs: list[ModelBuildSpec],
    batch_sizes: list[int],
    num_iterations: int = 10,
    warmup_iterations: int = 5,
    max_experiments: int | None = None,
    factory: Optional[
        Callable[[ModelBuildSpec], tuple[nn.Module, torch.optim.Optimizer]]
    ] = None,
) -> dict:
    """Run experiments for all combinations of (model, batch_size) pairs.

    Args:
        fabric: Lightning Fabric instance.
        model_specs: List of ModelBuildSpec objects.
        batch_sizes: List of batch sizes to test (ordered from smallest to largest).
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.
        max_experiments: Maximum number of experiments to run. If None, all combinations are run.
        factory: Model factory producing (model, optimizer).

    Returns:
        Dictionary with tuple keys and experiment results as values.
    """
    from datetime import datetime

    if factory is None:
        raise ValueError("A model factory must be provided.")

    if max_experiments is None:
        max_experiments = len(model_specs) * len(batch_sizes)

    if fabric.global_rank != 0:
        return {}

    print(f"\n🧪 Starting Memory Scaling Experiments (Generic)")
    print(f"   Models: {[spec.name for spec in model_specs]}")
    print(f"   Batch sizes: {batch_sizes}")
    print(f"   Warmup iterations: {warmup_iterations}")
    print(f"   Measurement iterations: {num_iterations}", flush=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(fabric.device),
        "experiments": {},
    }

    total_experiments = len(model_specs) * len(batch_sizes)
    experiment_count = 0

    for model_spec in model_specs:
        print(f"\n🔬 Testing model: {model_spec.name}")

        for batch_size in batch_sizes:
            experiment_count += 1
            if max_experiments and experiment_count > max_experiments:
                print(f"   ⏹️  Reached max experiments limit ({max_experiments})")
                return results

            print(
                f"     [{experiment_count}/{total_experiments}] Testing: {model_spec.name}, batch_size={batch_size}, seq_len={model_spec.sequence_length}"
            )

            # Run experiment
            experiment_result = run_single_experiment_generic(
                fabric=fabric,
                spec=model_spec,
                batch_size=batch_size,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
                factory=factory,
            )

            # Store result with tuple key
            key = (model_spec.name, batch_size, model_spec.sequence_length)
            results["experiments"][key] = experiment_result

            # Check if we should stop for this model
            if experiment_result.get("status") != "success":
                print(
                    f"       ⚠️  All batch sizes failed for {model_spec.name}, stopping larger batch sizes"
                )
                break

    return results


def save_results_csv_generic(results: dict, timestamp: str | None = None) -> None:
    """Save experiment results to CSV file.

    Args:
        results: Experiment results dictionary.
        timestamp: Timestamp string for filename. If None, current timestamp is used.
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV headers
    headers = [
        "model_name",
        "batch_size",
        "sequence_length",
        "total_params_millions",
        "INF_mem",
        "INF_cached_mem",
        "INF_peak_mem",
        "FWD_mem",
        "FWD_cached_mem",
        "FWD_peak_mem",
        "TS_mem",
        "TS_cached_mem",
        "TS_peak_mem",
        "status",
    ]

    rows = [headers]

    for experiment in results["experiments"].values():
        if experiment.get("status") == "success":
            row = [
                experiment["model_name"],
                experiment["batch_size"],
                experiment["sequence_length"],
                f"{experiment['total_params_millions']:.1f}",
                f"{experiment['avg_inf_memory_gb']:.3f} Gb",
                f"{experiment['avg_inf_cached_memory_gb']:.3f}",
                f"{experiment['avg_inf_peak_memory_gb']:.3f}",
                f"{experiment['avg_fwd_memory_gb']:.3f}",
                f"{experiment['avg_fwd_cached_memory_gb']:.3f}",
                f"{experiment['avg_fwd_peak_memory_gb']:.3f}",
                f"{experiment['avg_ts_memory_gb']:.3f}",
                f"{experiment['avg_ts_cached_memory_gb']:.3f}",
                f"{experiment['avg_ts_peak_memory_gb']:.3f}",
                experiment["status"],
            ]
        else:
            total_m = experiment.get(
                "total_params_millions",
                experiment.get("total_params", 0) / 1e6,
            )
            row = [
                experiment["model_name"],
                experiment["batch_size"],
                experiment["sequence_length"],
                f"{total_m:.1f}",
                "",  # INF_mem
                "",  # INF_cached_mem
                "",  # INF_peak_mem
                "",  # FWD_mem
                "",  # FWD_cached_mem
                "",  # FWD_peak_mem
                "",  # TS_mem
                "",  # TS_cached_mem
                "",  # TS_peak_mem
                experiment["status"],
            ]
        rows.append(row)

    # Write CSV file
    filename = f"memory_results_generic_{timestamp}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"✅ CSV results saved to: {filename}")


def save_results_generic(results: dict, timestamp: str | None = None) -> None:
    """Save experiment results in JSON and CSV formats.

    Args:
        results: Experiment results dictionary.
        timestamp: Optional timestamp string for filenames.
    """
    import json
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert tuple keys to strings for JSON serialization
    json_safe_results = {
        "timestamp": results["timestamp"],
        "device": results["device"],
        "experiments": {},
    }

    for key, experiment in results["experiments"].items():
        # Convert tuple key to string key
        key_str = "_".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        json_safe_results["experiments"][key_str] = experiment

    # Save full results
    full_filename = f"memory_results_generic_full_{timestamp}.json"
    with open(full_filename, "w") as f:
        json.dump(json_safe_results, f, indent=2)

    # Create simplified results
    simplified_data = {
        "timestamp": results["timestamp"],
        "device": results["device"],
        "experiments": {},
    }

    for key, experiment in results["experiments"].items():
        # Convert tuple key to string key
        key_str = "_".join(str(k) for k in key) if isinstance(key, tuple) else str(key)

        if experiment.get("status") == "success":
            simplified_data["experiments"][key_str] = {
                "model_name": experiment["model_name"],
                "batch_size": experiment["batch_size"],
                "sequence_length": experiment["sequence_length"],
                "total_params_millions": experiment["total_params_millions"],
                "INF_mem": experiment["avg_inf_memory_gb"],
                "INF_cached_mem": experiment["avg_inf_cached_memory_gb"],
                "INF_peak_mem": experiment["avg_inf_peak_memory_gb"],
                "FWD_mem": experiment["avg_fwd_memory_gb"],
                "FWD_cached_mem": experiment["avg_fwd_cached_memory_gb"],
                "FWD_peak_mem": experiment["avg_fwd_peak_memory_gb"],
                "TS_mem": experiment["avg_ts_memory_gb"],
                "TS_cached_mem": experiment["avg_ts_cached_memory_gb"],
                "TS_peak_mem": experiment["avg_ts_peak_memory_gb"],
                "status": experiment["status"],
            }
        else:
            simplified_data["experiments"][key_str] = {
                "model_name": experiment["model_name"],
                "batch_size": experiment["batch_size"],
                "sequence_length": experiment["sequence_length"],
                "total_params_millions": experiment.get(
                    "total_params_millions",
                    experiment.get("total_params", 0) / 1e6,
                ),
                "config": experiment["config"],
                "status": experiment["status"],
                "error": experiment.get("error", ""),
            }

    # Save simplified results
    simplified_filename = f"memory_results_generic_simplified_{timestamp}.json"
    with open(simplified_filename, "w") as f:
        json.dump(simplified_data, f, indent=2)

    # Save CSV results
    save_results_csv_generic(results, timestamp)

    print(f"✅ Full results saved to: {full_filename}")
    print(f"✅ Simplified results saved to: {simplified_filename}")


def measure_memory_scaling_experiments_generic(
    fabric: Fabric,
    num_iterations: int = 5,
    warmup_iterations: int = 2,
    factory: Optional[
        Callable[[ModelBuildSpec], tuple[nn.Module, torch.optim.Optimizer]]
    ] = None,
) -> dict:
    """Run memory scaling experiments for different model configurations.

    Args:
        fabric: Lightning Fabric instance.
        num_iterations: Number of iterations to measure after warmup.
        warmup_iterations: Number of warmup iterations.
        factory: Model factory producing (model, optimizer).

    Returns:
        Dictionary containing all experiment results.
    """
    from datetime import datetime

    if fabric.global_rank != 0:
        return {}

    if factory is None:
        # Use the example factory if none provided
        factory = example_callable_factory()

    # Define test configurations
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]  # Ordered from smallest to largest
    model_specs = [
        ModelBuildSpec(
            name="tiny256",
            vocab_size=50304,
            sequence_length=256,
            config={"n_layer": 1, "n_head": 2, "n_embd": 256},
        ),
        ModelBuildSpec(
            name="small512",
            vocab_size=50304,
            sequence_length=512,
            config={"n_layer": 2, "n_head": 4, "n_embd": 512},
        ),
        ModelBuildSpec(
            name="medium1024",
            vocab_size=50304,
            sequence_length=1024,
            config={"n_layer": 4, "n_head": 8, "n_embd": 1024},
        ),
        ModelBuildSpec(
            name="large2048",
            vocab_size=50304,
            sequence_length=2048,
            config={"n_layer": 8, "n_head": 16, "n_embd": 2048},
        ),
    ]

    # Run experiments
    results = run_experiment_grid_generic(
        fabric=fabric,
        model_specs=model_specs,
        batch_sizes=batch_sizes,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
        factory=factory,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results_generic(results, timestamp)

    return results


def example_callable_factory() -> (
    Callable[[ModelBuildSpec], tuple[nn.Module, torch.optim.Optimizer]]
):
    """Example Python-only factory for a simple language model.

    Returns:
        A callable factory that builds a tiny model and AdamW optimizer.

    """

    class TinyLM(nn.Module):
        def __init__(self, vocab_size: int, n_embd: int, block_size: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, n_embd)
            self.fc = nn.Linear(n_embd, vocab_size)
            self.block_size = block_size

        def forward(self, x: Int[Tensor, "b t"]) -> Tensor:
            emb = self.embed(x)  # [B, T, C]
            logits = self.fc(emb)  # [B, T, V]
            return logits

    def factory(spec: ModelBuildSpec) -> tuple[nn.Module, torch.optim.Optimizer]:
        model = TinyLM(
            vocab_size=spec.vocab_size,
            n_embd=128,
            block_size=spec.sequence_length,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        return model, optimizer

    return factory


if __name__ == "__main__":
    # Self-contained tests (CPU-safe)
    fab = Fabric(accelerator="cpu", devices=1)

    spec = ModelBuildSpec(
        name="tinylm",
        vocab_size=257,
        sequence_length=16,
        config={"n_embd": 128},
    )

    fac = example_callable_factory()
    res = run_single_experiment_generic(
        fabric=fab,
        spec=spec,
        batch_size=2,
        num_iterations=2,
        warmup_iterations=1,
        factory=fac,
    )

    assert isinstance(res, dict)
    assert res["model_name"] == "tinylm"
    assert res["status"] in {"success", "runtime_error", "error"}
    print(f"Test 1 passed: generic runner basic result OK")

    # Test the measurement decorator shape on CPU
    model, opt = fac(spec)
    model = model.to(fab.device)
    x = torch.randint(0, spec.vocab_size, (2, spec.sequence_length), device=fab.device)
    y = torch.randint(0, spec.vocab_size, (2, spec.sequence_length), device=fab.device)
    out = run_training_step_generic(model, opt, x, y, spec.vocab_size)
    for k in ["mem_alloc_gb", "mem_peak_gb", "mem_cached_gb", "mem_delta_gb"]:
        assert k in out
    print(f"Test 2 passed: decorator memory keys present on CPU")

    # Test the grid experiment runner
    print(f"\nTesting grid experiment runner...")
    grid_results = run_experiment_grid_generic(
        fabric=fab,
        model_specs=[spec],
        batch_sizes=[1, 2],
        num_iterations=1,
        warmup_iterations=1,
        factory=fac,
    )

    assert "experiments" in grid_results
    assert len(grid_results["experiments"]) > 0
    print(f"Test 3 passed: grid experiment runner OK")

    # Test data saving functions
    print(f"\nTesting data saving functions...")
    save_results_generic(grid_results, "test")
    print(f"Test 4 passed: data saving functions OK")

    # Test utility functions
    print(f"\nTesting utility functions...")
    model_exps = get_experiments_by_model_generic(grid_results, "tinylm")
    assert len(model_exps) > 0
    print(f"Test 5 passed: utility functions OK")

    print_experiment_summary_generic(grid_results)
    print(f"Test 6 passed: summary printing OK")

    print(f"\nAll tests passed.")

    # Example of running full experiments (commented out for safety)
    # print(f"\nTo run full experiments on GPU, uncomment the following lines:")
    # print(f"# fabric = Fabric(accelerator='cuda', devices=1)")
    # print(f"# results = measure_memory_scaling_experiments_generic(fabric)")
    # print(f"# print_experiment_summary_generic(results)")
