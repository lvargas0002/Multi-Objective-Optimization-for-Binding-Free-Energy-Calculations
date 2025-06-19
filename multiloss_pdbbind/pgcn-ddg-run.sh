#!/bin/bash
#SBATCH --job-name=NMODE              # Job name
#SBATCH --time=72:00:00                    # Time limit (HH:MM:SS)
#SBATCH --partition=cpu                     # Partition name (adjust as necessary)
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=8
#SBATCH --exclusive                 # Ensure exclusive node access


# Navigate to your project directory
cd /home/lthoma21/BFE-Loss-Function || exit 1

# Print environment information for debugging
echo "Starting job at $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "GPU information: $(nvidia-smi || echo 'nvidia-smi not available')"

# Run your Python script
python /home/lthoma21/BFE-Loss-Function/FINAL-PDBBIND-FILES/BFE_with_loss_function.py

echo "Job completed at $(date)"