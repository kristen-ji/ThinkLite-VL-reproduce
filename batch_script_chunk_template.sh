#!/bin/bash
#
# ===== CHUNK PROCESSING SCRIPT =====
# This script processes chunks of data using MCTS
# Run with: bash batch_script_chunk_template.sh

# Set chunk parameters (modifiable via environment variables)
NUM_CHUNKS=${NUM_CHUNKS:-8}          # Total number of chunks (default 8)
DATASET_SIZE=${DATASET_SIZE:-69996}  # Total dataset size
START_SAMPLE=${START_SAMPLE:-1834}   # Start from sample 1834

echo "[PARALLEL CHUNK CONFIG] NUM_CHUNKS=${NUM_CHUNKS} DATASET_SIZE=${DATASET_SIZE} START_SAMPLE=${START_SAMPLE}"
echo "Starting parallel chunk processing on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of tasks: $SLURM_NTASKS"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Load modules for GPU and Python
module load devel/cuda/12.4

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# Activate your virtual environment
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Install required packages for monitoring
pip install psutil

# Ensure results directory exists in the submit directory
mkdir -p "$SLURM_SUBMIT_DIR/results"

# Get the local task ID (0-7 for 8 tasks)
LOCAL_TASK_ID=$SLURM_PROCID
CHUNK_ID=$LOCAL_TASK_ID

echo "Task $LOCAL_TASK_ID processing chunk $CHUNK_ID on GPU $LOCAL_TASK_ID"
echo "[PARALLEL] Starting chunk $CHUNK_ID at $(date) on node $(hostname)"

# Set GPU device for this task
export CUDA_VISIBLE_DEVICES=$LOCAL_TASK_ID
echo "[GPU] CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES for chunk $CHUNK_ID"

# Create and enter working directory on node-local storage
WORKDIR=$TMPDIR/gpujob_chunk_${CHUNK_ID}_${SLURM_JOB_ID}_${LOCAL_TASK_ID}
mkdir -p $WORKDIR
cd $WORKDIR

# Stage in: copy input data or model from the submit directory
cp "$SLURM_SUBMIT_DIR/mcts.py" .

# Run your GPU compute job for this chunk
python mcts.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --eval_model_name Qwen/Qwen2.5-7B-Instruct \
    --output_file results_chunk_${CHUNK_ID}.parquet \
    --num-chunks ${NUM_CHUNKS} \
    --chunk-idx ${CHUNK_ID} \
    --gpu-id ${LOCAL_TASK_ID} \
    --max_num_iterations 5 \
    --skip-samples ${START_SAMPLE}

# Stage out: copy results back to the submit directory
cp results_chunk_${CHUNK_ID}.parquet "$SLURM_SUBMIT_DIR/results/chunk_${CHUNK_ID}_job_${SLURM_JOB_ID}_task_${LOCAL_TASK_ID}.parquet"

# Only copy checkpoints if they exist
if [ -d checkpoints ]; then
    cp -r checkpoints "$SLURM_SUBMIT_DIR/results/checkpoints_chunk_${CHUNK_ID}_${SLURM_JOB_ID}_task_${LOCAL_TASK_ID}/"
fi

# Cleanup
cd ~
echo "Task $LOCAL_TASK_ID completed for chunk ${CHUNK_ID} on GPU $LOCAL_TASK_ID"

# Wait for all tasks to complete
srun --wait
echo "All chunks completed successfully! Processing started from sample ${START_SAMPLE}"
