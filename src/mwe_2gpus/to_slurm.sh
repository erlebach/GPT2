
# To run with SLURM, request 2 GPUs per node. Example SLURM sbatch:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
module load cuda/12.1
pip install torch pytorch-lightning
srun python lightning_mwe_2gpu_slurm.py
