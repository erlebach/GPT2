"""
Run it:
```
python / Users / erlebach / src / 2025 / GPT2 / src / experiments / run_mem_from_yaml.py
```

What this applies to:
- The YAML route is for “another model” you want to measure by just
  pointing at a _target_. The small adapter above makes your existing
  LightningModule accept flat kwargs from YAML. If the “another model”
  is a plain nn.Module that already takes kwargs directly, you don’t need
  an adapter—point _target_ to that class, and it will work with
  model_factory_from_yaml as-is.
"""

# src/experiments/run_mem_from_yaml.py
from __future__ import annotations

import sys
import types
from typing import Any

import torch
from lightning import Fabric

# Ensure `src/` is on path when running as a script
if str(__file__).endswith("run_mem_from_yaml.py"):
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))

# Because our YAML refers to the LightningModule, but that module's constructor
# expects a GPTConfig, we provide a thin wrapper that adapts kwargs.
# If your `_target_` is a plain nn.Module that accepts kwargs directly,
# you can skip this adapter and use the YAML factory as-is.
import gpt2_standalone.lightning_module as lm  # noqa: E402
from gpt2_standalone.model import GPTConfig  # noqa: E402

from experiments.memory_measurements_generic import (  # noqa: E402
    ModelBuildSpec,
    model_factory_from_yaml,
    run_single_experiment_generic,
)


class GPTLMAdapter(lm.GPTLightningModule):
    """Adapter that lets us pass config fields directly via kwargs.

    This makes YAML simpler: we build a GPTConfig inside the adapter.

    """

    def __init__(self, **kwargs: Any):
        block_size = kwargs.pop("block_size")
        vocab_size = kwargs.pop("vocab_size")
        n_layer = kwargs.pop("n_layer", 2)
        n_head = kwargs.pop("n_head", 4)
        n_embd = kwargs.pop("n_embd", 128)
        n_blocks_per_super = kwargs.pop("n_blocks_per_super", 2)

        cfg = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_blocks_per_super=n_blocks_per_super,
        )
        super().__init__(cfg, **kwargs)


# Expose the adapter via a stable path so YAML can reference it
# Path to reference in YAML: gpt2_standalone.lightning_module_adapter.GPTLMAdapter
sys.modules["gpt2_standalone.lightning_module_adapter"] = types.ModuleType(
    "gpt2_standalone.lightning_module_adapter"
)
setattr(
    sys.modules["gpt2_standalone.lightning_module_adapter"],
    "GPTLMAdapter",
    GPTLMAdapter,
)

if __name__ == "__main__":
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    fab = Fabric(accelerator=accelerator, devices=1)

    # Use Hydra to resolve the path to the YAML file relative to the
    # current file

    # Get the script's directory to resolve relative paths correctly
    script_dir = pathlib.Path(__file__).resolve().parent

    # Use the adapter in YAML:
    # model:
    #   _target_: gpt2_standalone.lightning_module_adapter.GPTLMAdapter
    #   n_layer: 4
    #   n_head: 8
    #   n_embd: 1024
    factory = model_factory_from_yaml(
        yaml_path=str(script_dir / "config/memory/my_model.yaml"),
        model_key="model",
        optimizer_key="optimizer",
    )

    spec = ModelBuildSpec(
        name="yaml_medium1024",
        vocab_size=50304,
        sequence_length=1024,
        config={"n_layer": 4, "n_head": 8, "n_embd": 1024},
    )

    result = run_single_experiment_generic(
        fabric=fab,
        spec=spec,
        batch_size=32,
        num_iterations=3,
        warmup_iterations=2,
        factory=factory,
    )

    print(
        f"{result['model_name']} (B={result['batch_size']}, "
        f"T={result['sequence_length']}) → TS peak "
        f"{result.get('avg_ts_peak_memory_gb', 0.0):.3f} GB"
    )
