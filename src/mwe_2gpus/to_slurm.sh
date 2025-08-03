#!/bin/bash
#SBATCH --job-name=mwe_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --output=mwe_%j.out
#SBATCH --error=mwe_%j.err

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
pip install torch pytorch-lightning

# Run with srun to properly launch distributed processes
srun python mwe.py
