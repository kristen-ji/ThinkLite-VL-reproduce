#!/bin/bash
#
# ===== SLURM HEADER SECTION =====
#SBATCH --job-name=ThinkLite_chunk_parallel    # Job name
#SBATCH --output=log_chunk_%j.out              # Stdout (%j = job ID)
#SBATCH --error=log_chunk_%j.err               # Stderr
#SBATCH --nodes=1                              # Two nodes
#SBATCH --ntasks=1                             # Total 8 tasks (instead of per-node)
#SBATCH --array=0-7                            # Array of tasks to run
#SBATCH --cpus-per-task=38                      # Reduced CPUs per task
#SBATCH --mem=100000mb                          # Reduced memory per task
#SBATCH --time=24:00:00                        # Increased time limit to 24 hours
#SBATCH --partition=accelerated                # GPU partition
#SBATCH --gres=gpu:1                           # Request 4 GPUs per node
#SBATCH --account=hk-project-pai00012          # Project account ID
#SBATCH --distribution=block                   # Ensure tasks are distributed across nodes

# ===== MULTI-NODE PARALLEL CHUNK PROCESSING WITH MONITORING =====
# This script processes 8 chunks of data using MCTS in parallel across 2 nodes
# Run with: sbatch batch_script_chunk_template.sh

echo "=== MULTI-NODE PARALLEL CHUNK PROCESSING WITH MONITORING ==="
echo "Starting multi-node parallel chunk processing"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Total tasks: $SLURM_NTASKS"
echo "Current node: $SLURM_NODEID"
echo "Local task ID: $SLURM_PROCID (0-7 total)"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo ""

# ===== COMPREHENSIVE DIAGNOSTICS =====
echo "=== SLURM ENVIRONMENT DIAGNOSTICS ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo ""

# Check cluster resources
echo "=== CLUSTER RESOURCE CHECK ==="
echo "Current node: $(hostname)"
echo "Node IP: $(hostname -I | awk '{print $1}')"
echo "Available GPUs on this node:"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Check if this is the first task (task 0) for coordination
IS_COORDINATOR=false
if [ "$SLURM_PROCID" -eq 0 ]; then
    IS_COORDINATOR=true
    echo "=== COORDINATOR TASK DETECTED ==="
    echo "This task will coordinate monitoring and final summary"
    echo ""
fi

# Load modules for GPU and Python
module load devel/cuda/12.4

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# Activate virtual environment
if [ -f "$SLURM_SUBMIT_DIR/.venv/bin/activate" ]; then
    source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
    echo "[ENV] Activated virtual environment"
else
    echo "[ENV] Using system Python"
fi

# Ensure results directory exists
mkdir -p "$SLURM_SUBMIT_DIR/results"

# Create monitoring directory for this job
MONITOR_DIR="$SLURM_SUBMIT_DIR/monitor_job_${SLURM_JOB_ID}"
mkdir -p "$MONITOR_DIR"

# Create task status file for this task
TASK_STATUS_FILE="$MONITOR_DIR/task_${SLURM_PROCID}.status"
echo "STARTED: $(date)" > "$TASK_STATUS_FILE"
echo "NODE: $SLURM_NODEID" >> "$TASK_STATUS_FILE"
echo "TASK: $SLURM_PROCID" >> "$TASK_STATUS_FILE"
echo "HOSTNAME: $(hostname)" >> "$TASK_STATUS_FILE"

# Task assignment - ch task gets its own chunk ID (0-7)
CHUNK_ID=$SLURM_ARRAY_TASK_ID
echo "[DEBUG] SLURM_PROCID: $SLURM_PROCID"
echo "[DEBUG] SLURM_NTASKS: $SLURM_NTASKS"
echo "[DEBUG] SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "[DEBUG] SLURM_NNODES: $SLURM_NNODES"
echo "[DEBUG] SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

echo "Task $SLURM_PROCID processing chunk $CHUNK_ID on node $(hostname)"
echo "[MULTI-NODE] Starting chunk $CHUNK_ID at $(date) on node $(hostname)"

# Set GPU device for this task (local GPU within the node)
# Calculate local GPU ID: task_id % 4 (since 4 GPUs per node)
LOCAL_GPU_ID=$((SLURM_PROCID % 4))
export CUDA_VISIBLE_DEVICES=$LOCAL_GPU_ID
echo "[GPU] CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES for chunk $CHUNK_ID"

# Verify GPU is available
if ! nvidia-smi -i $LOCAL_GPU_ID > /dev/null 2>&1; then
    echo "[ERROR] GPU $LOCAL_GPU_ID not available on node $(hostname)"
    echo "[ERROR] Available GPUs:"
    nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"
    exit 1
fi
echo "[GPU] GPU $LOCAL_GPU_ID verified as available"

# Create and enter working directory
WORKDIR=$TMPDIR/gpujob_chunk_${CHUNK_ID}_${SLURM_JOB_ID}_${SLURM_PROCID}
mkdir -p $WORKDIR
cd $WORKDIR

# Copy mcts.py to working directory
cp "$SLURM_SUBMIT_DIR/mcts.py" .
echo "[CHUNK $CHUNK_ID] Copied mcts.py to working directory"

# Check if data file exists (optional - will use Hugging Face if not found)
if [ -f "$SLURM_SUBMIT_DIR/ThinkLite-VL-70k.parquet" ]; then
    echo "[INFO] Found local parquet file, will use it"
    DATA_SOURCE="$SLURM_SUBMIT_DIR/ThinkLite-VL-70k.parquet"
else
    echo "[INFO] No local parquet file found, will use Hugging Face dataset"
    echo "[INFO] Dataset: https://huggingface.co/datasets/russwang/ThinkLite-VL-70k"
    DATA_SOURCE="huggingface"
fi

# Run MCTS for this chunk
echo "[CHUNK $CHUNK_ID] Starting MCTS processing..."
if [ "$DATA_SOURCE" = "huggingface" ]; then
    echo "[CHUNK $CHUNK_ID] Using Hugging Face dataset"
    python mcts.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --eval_model_name Qwen/Qwen2.5-7B-Instruct \
        --output_file results_chunk_${CHUNK_ID}.parquet \
        --num-chunks 8 \
        --chunk-idx ${CHUNK_ID} \
        --gpu-id ${LOCAL_GPU_ID} \
        --max_num_iterations 5 \
        --skip-samples 0
else
    echo "[CHUNK $CHUNK_ID] Using local parquet file: $DATA_SOURCE"
    python mcts.py \
        --data_pths "$DATA_SOURCE" \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --eval_model_name Qwen/Qwen2.5-7B-Instruct \
        --output_file results_chunk_${CHUNK_ID}.parquet \
        --num-chunks 8 \
        --chunk-idx ${CHUNK_ID} \
        --gpu-id ${LOCAL_GPU_ID} \
        --max_num_iterations 5 \
        --skip-samples 0
fi

# Check if MCTS completed successfully
if [ $? -ne 0 ]; then
    echo "[ERROR] MCTS processing failed for chunk $CHUNK_ID"
    echo "FAILED: $(date)" >> "$TASK_STATUS_FILE"
    echo "ERROR: MCTS processing failed" >> "$TASK_STATUS_FILE"
    exit 1
fi

# Copy results back
cp results_chunk_${CHUNK_ID}.parquet "$SLURM_SUBMIT_DIR/results/chunk_${CHUNK_ID}_job_${SLURM_JOB_ID}_task_${SLURM_PROCID}.parquet"

# Update task status
echo "COMPLETED: $(date)" >> "$TASK_STATUS_FILE"
echo "CHUNK_ID: $CHUNK_ID" >> "$TASK_STATUS_FILE"
echo "OUTPUT_FILE: chunk_${CHUNK_ID}_job_${SLURM_JOB_ID}_task_${SLURM_PROCID}.parquet" >> "$TASK_STATUS_FILE"

# Cleanup
cd ~
echo "[CHUNK $CHUNK_ID] Task $SLURM_PROCID completed for chunk ${CHUNK_ID} on GPU $LOCAL_GPU_ID"

# Only task 0 prints summary and monitoring
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo ""
    echo "=== MULTI-NODE PARALLEL EXECUTION COMPLETED ==="
    echo "All 8 chunks processed successfully across 2 nodes!"
    echo "Node 0: Chunks 0-3 (Tasks 0-3)"
    echo "Node 1: Chunks 4-7 (Tasks 4-7)"
    echo "Results saved in: $SLURM_SUBMIT_DIR/results/"
    
    # Generate comprehensive monitoring report
    echo ""
    echo "=== COMPREHENSIVE MONITORING REPORT ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Completion time: $(date)"
    echo ""
    
    # Check all task status files
    echo "=== TASK STATUS SUMMARY ==="
    if [ -d "$MONITOR_DIR" ]; then
        for status_file in "$MONITOR_DIR"/task_*.status; do
            if [ -f "$status_file" ]; then
                echo "--- $(basename "$status_file") ---"
                cat "$status_file"
                echo ""
            fi
        done
    fi
    
    # Count completed chunks
    echo "=== COMPLETED CHUNKS SUMMARY ==="
    COMPLETED_CHUNKS=$(find "$SLURM_SUBMIT_DIR/results" -name "chunk_*_job_${SLURM_JOB_ID}_*.parquet" -type f 2>/dev/null | wc -l)
    echo "Total completed chunks: $COMPLETED_CHUNKS out of 8"
    
    if [ $COMPLETED_CHUNKS -eq 8 ]; then
        echo "✅ SUCCESS: All 8 chunks completed successfully!"
    else
        echo "⚠️  WARNING: Only $COMPLETED_CHUNKS chunks completed out of 8"
        echo "Missing chunks:"
        for i in {0..7}; do
            if [ ! -f "$SLURM_SUBMIT_DIR/results/chunk_${i}_job_${SLURM_JOB_ID}_task_*.parquet" ]; then
                echo "  - Chunk $i"
            fi
        done
    fi
    
    # Show file sizes
    echo ""
    echo "=== OUTPUT FILE SIZES ==="
    find "$SLURM_SUBMIT_DIR/results" -name "chunk_*_job_${SLURM_JOB_ID}_*.parquet" -type f -exec ls -lh {} \;
    
    echo ""
    echo "================================================"
    echo "Monitoring data saved in: $MONITOR_DIR"
    echo "Results saved in: $SLURM_SUBMIT_DIR/results/"
    echo "================================================"
fi
