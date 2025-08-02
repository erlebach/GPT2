"""GPT-2 training script with GPU parallelism and monitoring.

This script trains a GPT-2 model using PyTorch Lightning with proper
GPU parallelism (DDP) and real-time GPU monitoring.
"""

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from gpt2_standalone.gpu_monitor import GPUMonitor, create_gpu_monitor_callback
from gpt2_standalone.gpu_parallelism_checker import (
    GPUParallelismChecker,
    create_strategy_from_config,
)
from gpt2_standalone.lightning_module import train_with_lightning
from utils.data_utils import get_project_root
from utils.metrics_extensions import get_metrics_collector, save_metrics_to_csv


def find_free_port():
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def main():
    """Main function to run GPT-2 training with GPU monitoring."""
    # Set up distributed training environment variables
    if os.environ.get("MASTER_PORT") is None:
        free_port = find_free_port()
        os.environ["MASTER_PORT"] = str(free_port)
        print(f"🔧 Set MASTER_PORT to {free_port}")

    if os.environ.get("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"] = "localhost"
        print(f"🔧 Set MASTER_ADDR to localhost")

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

    # NOTE: Let Lightning handle distributed initialization automatically
    # Do NOT manually initialize distributed when using Lightning's ddp_spawn strategy
    # Lightning will handle all the distributed setup internally

    # Initialize GPU parallelism checker
    gpu_parallelism_checker = GPUParallelismChecker()
    gpu_parallelism_checker.print_comprehensive_report()

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Number of available GPUs: {num_gpus}")

    # Force DDP strategy for multi-GPU training
    if num_gpus > 1:
        # Use Lightning's DDP spawn strategy (more reliable)
        strategy = "ddp_spawn"  # Use DDP spawn instead of DDP
        print(f"🚀 Using Lightning's DDP spawn strategy for {num_gpus} GPUs")
        print(f"   This will automatically handle distributed setup")
    else:
        strategy_name, strategy_config = (
            gpu_parallelism_checker.get_recommended_strategy(num_gpus=num_gpus)
        )
        strategy = create_strategy_from_config(strategy_name, strategy_config)

    print(f"✅ Strategy: {strategy}")

    # Create GPU monitor callback
    gpu_monitor_callback = create_gpu_monitor_callback(log_interval=10.0)
    if gpu_monitor_callback:
        print("✅ GPU monitor callback created")
    else:
        print("⚠️  Could not create GPU monitor callback")

    # Verify multi-GPU setup before training
    monitor = GPUMonitor()
    verification = monitor.verify_multi_gpu_usage()
    print(f"\n🔍 Pre-training GPU verification:")
    print(f"   Status: {verification['status']}")
    print(f"   GPUs Available: {verification['total_gpus']}")
    print(f"   Expected to use: {num_gpus}")

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
        batch_size=64,
        max_steps=10000,
        n_layer=4,  # Number of super-layers
        n_head=8,  # Number of attention heads
        n_embd=512,  # Embedding dimension
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
    print(f"\n🔍 Post-training GPU verification:")
    post_verification = monitor.verify_multi_gpu_usage()
    print(f"   Status: {post_verification['status']}")
    print(
        f"   GPUs Used: {post_verification['gpus_used']}/{post_verification['total_gpus']}"
    )
    print(f"   Memory Usage: {post_verification['memory_usage']}")

    # Save metrics
    collector = get_metrics_collector()
    collector.save_all_metrics_to_csv("metrics.csv")
    print("✅ Training completed and metrics saved")


if __name__ == "__main__":
    main()
