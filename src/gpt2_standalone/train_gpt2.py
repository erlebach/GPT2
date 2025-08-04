"""GPT-2 training script with GPU parallelism and monitoring.

This script trains a GPT-2 model using PyTorch Lightning with proper
GPU parallelism (DDP) and real-time GPU monitoring.
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from gpt2_standalone.gpu_monitor import GPUMonitor, create_gpu_monitor_callback
from gpt2_standalone.gpu_parallelism_checker import (
    GPUParallelismChecker,
    create_strategy_from_config,
)
from gpt2_standalone.lightning_module import train_with_lightning
from lightning.pytorch.strategies import DDPStrategy
from utils.data_utils import get_project_root
from utils.metrics_extensions import get_metrics_collector, save_metrics_to_csv


def main():
    """Main function to run GPT-2 training with GPU monitoring."""
    # Debug distributed setup
    print(f"=== DISTRIBUTED SETUP DEBUG ===")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    # Check if distributed is initialized
    print(f"Distributed initialized: {dist.is_initialized()}")

    # MANUALLY INITIALIZE DISTRIBUTED IF NEEDED
    if not dist.is_initialized() and os.environ.get("WORLD_SIZE", "1") != "1":
        print("🚀 Manually initializing distributed process group...")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "12355")

        # Set device for this process
        torch.cuda.set_device(local_rank)

        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=local_rank,
        )
        print(f"✅ Distributed initialized - Rank {local_rank}/{world_size}")

    if dist.is_initialized():
        print(f"World size: {dist.get_world_size()}")
        print(f"Rank: {dist.get_rank()}")
        print(f"Backend: {dist.get_backend()}")

    # Initialize GPU parallelism checker
    gpu_parallelism_checker = GPUParallelismChecker()
    gpu_parallelism_checker.print_comprehensive_report()

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Number of available GPUs: {num_gpus}")
    print(
        f"🔍 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}"
    )
    print(f"🔍 WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"🔍 RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"🔍 LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")

    # Force DDP strategy for multi-GPU training
    if num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
        )
        print(f"🚀 Using DDP strategy for {num_gpus} GPUs")
        print(f"   Strategy type: {type(strategy)}")
        print(f"   Strategy config: {strategy.__dict__}")
    else:
        strategy_name, strategy_config = (
            gpu_parallelism_checker.get_recommended_strategy(num_gpus=num_gpus)
        )
        strategy = create_strategy_from_config(strategy_name, strategy_config)
        print(f"Stragegy for {num_gpus} GPUs.")

    print(f"✅ Strategy: {strategy}")

    # Create GPU monitor callback
    gpu_monitor_callback = create_gpu_monitor_callback(log_interval=10.0)
    if gpu_monitor_callback:
        print("✅ GPU monitor callback created")
    else:
        print("⚠️  Could not create GPU monitor callback")

    # Verify multi-GPU setup before training
    try:
        monitor = GPUMonitor()
        verification = monitor.verify_multi_gpu_usage()
        print(f"\n🔍 Pre-training GPU verification:")
        print(f"   Status: {verification['status']}")
        print(f"   GPUs Available: {verification['total_gpus']}")
        print(f"   Expected to use: {num_gpus}")
    except Exception as e:
        print(f"Error during pre-training GPU verification: {e}")
        print("Continuing with training...")

    # Force multi-GPU setup
    print(f"\n🚀 Starting training with explicit multi-GPU configuration:")
    print(f"   GPUs: {num_gpus}")
    print(f"   Strategy: {strategy}")
    print(f"   Devices: {num_gpus}")
    print(f"   Accelerator: gpu")

    # Start training with proper strategy and monitoring
    train_with_lightning(
        data_path=get_project_root() / "data" / "input.txt",
        block_size=1024,
        batch_size=32,
        max_steps=100,
        n_layer=4,  # Number of super-layers
        n_head=4,  # Number of attention heads
        n_embd=1024,  # Embedding dimension
        n_blocks_per_super=2,  # NEW: Number of blocks per SuperBlock
        weight_decay=0.2,
        accelerator="gpu",  # Explicitly set to GPU
        devices=num_gpus,  # Explicitly set number of devices
        strategy=strategy,  # Use the created strategy
        precision="32-true",
        checkpoint_path=Path("checkpoints/"),
        checkpoint=None,  # no restart
        callbacks=[gpu_monitor_callback] if gpu_monitor_callback else None,
    )

    # Post-training verification
    try:
        print(f"\n🔍 Post-training GPU verification:")
        post_verification = monitor.verify_multi_gpu_usage()
        print(f"   Status: {post_verification['status']}")
        print(
            f"   GPUs Used: {post_verification['gpus_used']}/{post_verification['total_gpus']}"
        )
        print(f"   Memory Usage: {post_verification['memory_usage']}")
    except Exception as e:
        print(f"Error during post-training GPU verification: {e}")
        print("Continuing with training...")

    # Save metrics
    collector = get_metrics_collector()
    collector.save_all_metrics_to_csv("metrics.csv")
    print("✅ Training completed and metrics saved")


if __name__ == "__main__":
    main()
