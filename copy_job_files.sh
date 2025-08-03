#!/bin/bash

# Script to copy job-related files to a version-specific folder
# Usage: ./copy_job_files.sh <JOBID>

# Check if JOBID is provided
if [ $# -eq 0 ]; then
    echo "Error: JOBID is required"
    echo "Usage: $0 <JOBID>"
    exit 1
fi

JOBID=$1

# Define destination folder
DEST_FOLDER="lightning_logs/version_${JOBID}"

# Create destination folder if it doesn't exist
echo "Creating destination folder: ${DEST_FOLDER}"
mkdir -p "${DEST_FOLDER}"

# Copy Python files from gpt2_standalone directory
echo "Copying Python files from src/gpt2_standalone/..."
cp src/gpt2_standalone/*.py "${DEST_FOLDER}/"

# Copy SLURM job files
echo "Copying SLURM job files..."
cp run_srun_job.slurm "${DEST_FOLDER}/"
cp run_torchrun_job.slurm "${DEST_FOLDER}/"
cp run_python_job.slurm "${DEST_FOLDER}/"

if [ -f "metrics.csv" ]; then
        mv metrics.csv "${DEST_FOLDER}/"
else
    echo "Warning: metrics.csv not found"
fi

# Copy submit script
echo "Copying submit script..."
cp submit_script.sh "${DEST_FOLDER}/"

# Copy SLURM output files (if they exist)
echo "Copying SLURM output files..."
if [ -f "slurm-nemo_python_job-${JOBID}.err" ]; then
    cp "slurm-nemo_python_job-${JOBID}.err" "${DEST_FOLDER}/"
else
    echo "Warning: slurm-nemo_python_job-${JOBID}.err not found"
fi

if [ -f "slurm-nemo_python_job-${JOBID}.out" ]; then
    cp "slurm-nemo_python_job-${JOBID}.out" "${DEST_FOLDER}/"
else
    echo "Warning: slurm-nemo_python_job-${JOBID}.out not found"
fi

if [ -f "README.md" ]; then
    cp README.md "${DEST_FOLDER}/"
else
    echo "Warning: README.md not found"
fi

echo "Files copied successfully to ${DEST_FOLDER}"
echo "Contents of ${DEST_FOLDER}:"
ls -la "${DEST_FOLDER}/"
