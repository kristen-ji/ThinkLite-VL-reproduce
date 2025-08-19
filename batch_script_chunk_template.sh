#!/bin/bash
#
# ===== HEADER SECTION =====
#SBATCH --job-name=ThinkLite_chunk_${CHUNK_ID}         # Job name
#SBATCH --output=log_chunk_${CHUNK_ID}_%j.out          # Stdout (%j = job ID)
#SBATCH --error=log_chunk_${CHUNK_ID}_%j.err           # Stderr
#SBATCH --nodes=1                                      # One node
#SBATCH --ntasks=1                                     # One task
#SBATCH --mem=256000mb                                 # Memory (RAM) per node
#SBATCH --time=08:00:00                                # Time limit (hh:mm:ss) - increased for 50 samples (7.7 hours)
#SBATCH --partition=accelerated                        # GPU partition
#SBATCH --gres=gpu:1                                   # Request 1 GPU
#SBATCH --account=hk-project-pai00012                  # Project account ID 

# ===== SHELL SCRIPT SECTION =====

# Set chunk parameters (modifiable via environment variables)
CHUNK_ID=${CHUNK_ID:-0}             # Which chunk index (0-based)
MAX_SAMPLES=${MAX_SAMPLES:-50}      # Max samples to actually process within this chunk (acts as an upper bound)
CHUNK_SIZE=${CHUNK_SIZE:-$MAX_SAMPLES}  # Logical chunk size (defaults to MAX_SAMPLES for simplicity)
DATASET_SIZE=${DATASET_SIZE:-69996} # Total dataset size (update if dataset changes)
if [ -z "${NUM_CHUNKS}" ]; then
    # Auto-compute number of chunks if not provided
    NUM_CHUNKS=$(( (DATASET_SIZE + CHUNK_SIZE - 1) / CHUNK_SIZE ))
fi

echo "[CHUNK CONFIG] CHUNK_ID=${CHUNK_ID} CHUNK_SIZE=${CHUNK_SIZE} MAX_SAMPLES=${MAX_SAMPLES} NUM_CHUNKS=${NUM_CHUNKS} DATASET_SIZE=${DATASET_SIZE}"
if [ ${CHUNK_ID} -ge ${NUM_CHUNKS} ]; then
    echo "[CHUNK CONFIG][ERROR] CHUNK_ID ${CHUNK_ID} >= NUM_CHUNKS ${NUM_CHUNKS}. Exiting."
    exit 1
fi

echo "Starting GPU job for chunk ${CHUNK_ID} on node: $(hostname)"
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

# Activate your virtual environment
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Install required packages for monitoring
pip install psutil

# Create and enter working directory on node-local storage
WORKDIR=$TMPDIR/gpujob_chunk_${CHUNK_ID}_$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

# Stage in: copy input data or model from the submit directory
cp "$SLURM_SUBMIT_DIR/mcts.py" .

# Ensure results directory exists in the submit directory
mkdir -p "$SLURM_SUBMIT_DIR/results"

# Run your GPU compute job for this chunk
python mcts.py \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --eval_model_name Qwen/Qwen2.5-3B-Instruct \
    --output_file results_chunk_${CHUNK_ID}.parquet \
    --max-samples ${MAX_SAMPLES} \
    --num-chunks ${NUM_CHUNKS} \
    --chunk-idx ${CHUNK_ID} \
    --max_num_iterations 100

# Stage out: copy results back to the submit directory
cp results_chunk_${CHUNK_ID}.parquet "$SLURM_SUBMIT_DIR/results/chunk_${CHUNK_ID}_job_${SLURM_JOB_ID}.parquet"

# Only copy checkpoints if they exist
if [ -d checkpoints ]; then
    cp -r checkpoints "$SLURM_SUBMIT_DIR/results/checkpoints_chunk_${CHUNK_ID}_${SLURM_JOB_ID}/"
fi

# Cleanup
cd ~
echo "Job completed for chunk ${CHUNK_ID}. To submit next chunk:"
echo "  sbatch --export=ALL,CHUNK_ID=$((CHUNK_ID+1)),MAX_SAMPLES=${MAX_SAMPLES},CHUNK_SIZE=${CHUNK_SIZE},DATASET_SIZE=${DATASET_SIZE},NUM_CHUNKS=${NUM_CHUNKS} $0"
