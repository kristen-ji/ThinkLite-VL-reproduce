#!/bin/bash
#SBATCH -p accelerated              # Partition/queue name
#SBATCH -N 2                     # Number of nodes
#SBATCH --mem=100G                # Memory per node
#SBATCH --gres=gpu:4             # GPUs per node
#SBATCH --time=08:00:00          # Max run time
#SBATCH --job-name=verl_train    # Job name
#SBATCH -o verl_train_%j.out     # Standard output log (%j will be replaced by job ID)
#SBATCH -e verl_train_%j.err     # Standard error log

# Load any necessary modules (if required by your cluster)
# module load python/3.8

# Activate your environment if needed
# source ~/myenv/bin/activate

# Run your command
python3 -m verl.trainer.main config=examples/config.yaml