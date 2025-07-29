#!/bin/bash -x
#SBATCH --job-name=nemo_python_job
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH -A pilotgpu

# --- 1. Get Python script from command-line argument ---
if [ -z "$1" ]; then
    echo "Error: No Python script provided."
    echo "Usage: sbatch $0 your_script.py"
    exit 1
fi
PYTHON_SCRIPT=$1

# --- 2. Set up cache directories in scratch space ---
SCRATCH_CACHE="/gpfs/research/scratch/gerlebacher/cache_${SLURM_JOB_ID}"
export HF_HOME="$SCRATCH_CACHE/huggingface"
export XDG_CACHE_HOME="$SCRATCH_CACHE"

# --- 3. Set up paths using SLURM_SUBMIT_DIR ---
PROJECT_ROOT="$(dirname "$SLURM_SUBMIT_DIR")"  # Parent of submission directory
CONTAINER_IMAGE="/gpfs/research/scratch/gerlebacher/containers/nemo_25.07.00.sif"

echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Container image: $CONTAINER_IMAGE"
echo "Using cache directory: $SCRATCH_CACHE"

module load webproxy

# --- 4. Run the container and execute Python directly ---
apptainer exec \
    --nv \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "/gpfs/research/scratch/gerlebacher:/gpfs/research/scratch/gerlebacher" \
    "$CONTAINER_IMAGE" \
    bash -c '
        set -x
        # Set environment variables to use scratch space
        export TMPDIR="/gpfs/research/scratch/gerlebacher/tmp_${SLURM_JOB_ID}"
        export HF_HOME="'"$SCRATCH_CACHE"'/huggingface"
        export XDG_CACHE_HOME="'"$SCRATCH_CACHE"'"
        mkdir -p "$TMPDIR" "$HF_HOME"
        
        # Change to the submit directory and run the Python script
        cd "'"$SLURM_SUBMIT_DIR"'"
        python "'"$PYTHON_SCRIPT"'" "${SCRIPT_ARGS[@]}"
    '
