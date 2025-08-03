
#!/bin/bash
# To run with SLURM, request 2 GPUs per node. Example SLURM sbatch:
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=2
module load cuda/12.1
pip install torch pytorch-lightning
srun python mwe.py
