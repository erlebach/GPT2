#!/bin/bash
# Alternative SLURM configurations for different cluster types

echo "=== SLURM Configuration Alternatives ==="
echo "Choose the configuration that matches your cluster setup:"
echo ""

echo "--- Option 1: Standard Multi-GPU Setup ---"
cat << 'EOF1'
#!/bin/bash
#SBATCH --job-name=mwe_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --output=mwe_%j.out
#SBATCH --error=mwe_%j.err
EOF1

echo ""
echo "--- Option 2: Specific GPU Type (if cluster has multiple GPU types) ---"
cat << 'EOF2'
#!/bin/bash
#SBATCH --job-name=mwe_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h100:2
#SBATCH --time=00:10:00
#SBATCH --output=mwe_%j.out
#SBATCH --error=mwe_%j.err
EOF2

echo ""
echo "--- Option 3: Conservative Resource Request ---"
cat << 'EOF3'
#!/bin/bash
#SBATCH --job-name=mwe_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:05:00
#SBATCH --output=mwe_%j.out
#SBATCH --error=mwe_%j.err
EOF3

echo ""
echo "--- Option 4: Single GPU for Testing ---"
cat << 'EOF4'
#!/bin/bash
#SBATCH --job-name=mwe_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=mwe_%j.out
#SBATCH --error=mwe_%j.err
EOF4

echo ""
echo "=== Debugging Commands ==="
echo "Run these commands to check your cluster configuration:"
echo ""
echo "1. Check available partitions:"
echo "   sinfo"
echo ""
echo "2. Check GPU availability:"
echo "   sinfo -o '%N %G' | grep gpu"
echo ""
echo "3. Check node features:"
echo "   scontrol show nodes | grep -E '(NodeName|Gres|CPUTot)'"
echo ""
echo "4. Test resource availability:"
echo "   salloc --nodes=1 --gres=gpu:1 --time=00:01:00 echo 'Resource test successful'"
echo ""