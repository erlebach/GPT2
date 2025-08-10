# src/experiments/memory_measurements_generic.py
from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Optional

import torch
from jaxtyping import Int
from lightning import Fabric
from torch import Tensor, nn

from experiments.clean_palate import deep_gpu_reset, reset_model_state


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
    """Decorator to measure GPU memory usage around a function call.

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
        opt_kwargs = {**opt_cfg}
        optimizer_cls = instantiate_from_target(opt_target, {})
        optimizer = optimizer_cls(params=model.parameters(), **opt_kwargs)
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
        print(
            f"       ❌ Runtime error for {spec.name}, batch={batch_size}, "
            f"seq={spec.sequence_length} (after {elapsed:.1f}s): {e} in file {__file__}",
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
        elapsed = time.time() - start_time
        print(
            f"       ❌ Unexpected error for {spec.name}, batch={batch_size}, "
            f"seq={spec.sequence_length} (after {elapsed:.1f}s): {e} in file {__file__}",
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

    print(f"All tests passed.")
