"""Run from a YAML file.

```
python / Users / erlebach / src / 2025 / GPT2 / src / experiments / run_mem_from_yaml.py
```

What this applies to:

- The YAML route is for "another model" you want to measure by just
  pointing at a _target_. The small adapter above makes your existing
  LightningModule accept flat kwargs from YAML. If the "another model"
  is a plain nn.Module that already takes kwargs directly, you don't need
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

# Import the real adapter from the dedicated file
import experiments.lightning_module_adapter  # noqa: E402
from experiments.memory_measurements_generic import (  # noqa: E402
    ModelBuildSpec,
    model_factory_from_yaml,
    run_single_experiment_generic,
)

# Expose the adapter via a stable path so YAML can reference it
# Path to reference in YAML: gpt2_standalone.lightning_module_adapter.GPTLMAdapter
sys.modules["gpt2_standalone.lightning_module_adapter"] = (
    experiments.lightning_module_adapter
)

if __name__ == "__main__":
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    fab = Fabric(accelerator=accelerator, devices=1)

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
        config={},
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
