#!/bin/bash
#SBATCH --job-name=mwe_2gpu
#SBATCH --output=mwe-%x-%j.out
#SBATCH --error=mwe-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=60GB
#SBATCH --time=01:00:00
#SBATCH -A pilotgpu


# Ensure proper SLURM environment variables for PyTorch Lightning
export SLURM_NTASKS_PER_NODE=2

# Print cluster info for debugging
echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs requested: $SLURM_GPUS_PER_NODE"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "======================="

# Load modules and setup environment
module load cuda/12.1 || echo "Warning: Could not load CUDA module"
module load webproxy
pip install torch pytorch-lightning

# nproc_per_node == number of GPUs (or else won't work)
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    mwe.py

