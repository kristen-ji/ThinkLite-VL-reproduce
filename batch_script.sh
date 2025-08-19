#!/bin/bash
#
# ===== HEADER SECTION =====
#SBATCH --job-name=ThinkLite_reproduce_job         # Job name
#SBATCH --output=log_%j.out                 # Stdout (%j = job ID)
#SBATCH --error=log_%j.err                  # Stderr
#SBATCH --nodes=2                            # Two nodes
#SBATCH --ntasks=1                           # One task
#SBATCH --mem=256000mb                       # Memory (RAM) per node (increased)
#SBATCH --time=01:30:00                      # Time limit (hh:mm:ss)
#SBATCH --partition=accelerated              # GPU partition (check with sinfo)
#SBATCH --gres=gpu:4                         # Request 4 GPU (increased)
#SBATCH --account=hk-project-pai00012              # Project account ID


# ===== SHELL SCRIPT SECTION =====

echo "Starting GPU job on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Load modules for GPU and Python
module load devel/cuda/12.4

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# Debug: Check if nvidia-smi is available
echo "Checking for nvidia-smi..."
which nvidia-smi || echo "nvidia-smi not found in PATH"
/usr/bin/nvidia-smi || echo "nvidia-smi not found in /usr/bin/"

# Check CUDA environment
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PATH: $PATH"

# (Optional) Activate your virtual environment
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Install required packages for monitoring
pip install psutil

#Create and enter working directory on node-local storage
WORKDIR=$TMPDIR/gpujob_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

# Stage in: copy input data or model from the submit directory
cp "$SLURM_SUBMIT_DIR/mcts.py" .
#cp -r $HOME/.vscode-server/ThinkLite-VL/data .

# Ensure results directory exists in the submit directory
mkdir -p "$SLURM_SUBMIT_DIR/results"

# Run your GPU compute job (example: PyTorch training)
# python mcts.py --epochs 10 --batch-size 32 > training_log.txt
python mcts.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --eval_model_name Qwen/Qwen2.5-3B-Instruct --output_file results_chunk_100.parquet --max-samples 100

# Stage out: copy results back to the submit directory
cp results_chunk_100.parquet "$SLURM_SUBMIT_DIR/results/job_${SLURM_JOB_ID}.parquet"
#cp -r checkpoints $HOME/.vscode-server/ThinkLite-VL/results/checkpoints_${SLURM_JOB_ID}/
# Only copy checkpoints if they exist
if [ -d checkpoints ]; then
    cp -r checkpoints "$SLURM_SUBMIT_DIR/results/checkpoints_${SLURM_JOB_ID}/"
fi

# Cleanup (TMPDIR is auto-cleaned but good practice)
cd ~
echo "Job completed."
