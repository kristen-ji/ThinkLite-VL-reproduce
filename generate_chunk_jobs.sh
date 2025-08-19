#!/bin/bash

# Script to generate multiple SLURM batch jobs for dataset chunks
# Usage: ./generate_chunk_jobs.sh <total_chunks> <samples_per_chunk>

TOTAL_CHUNKS=${1:-10}  # Default to 10 chunks
SAMPLES_PER_CHUNK=${2:-1000}  # Default to 1000 samples per chunk

echo "Generating $TOTAL_CHUNKS chunk jobs with $SAMPLES_PER_CHUNK samples each"

# Create jobs directory
mkdir -p chunk_jobs

for chunk_idx in $(seq 0 $((TOTAL_CHUNKS-1))); do
    echo "Creating job for chunk $chunk_idx"
    
    # Create chunk-specific batch script
    cat > "chunk_jobs/chunk_${chunk_idx}.sh" << EOF
#!/bin/bash
#
# ===== HEADER SECTION =====
#SBATCH --job-name=ThinkLite_chunk_${chunk_idx}         # Job name
#SBATCH --output=log_chunk_${chunk_idx}_%j.out          # Stdout (%j = job ID)
#SBATCH --error=log_chunk_${chunk_idx}_%j.err           # Stderr
#SBATCH --nodes=1                                       # One node
#SBATCH --ntasks=1                                      # One task
#SBATCH --mem=256000mb                                  # Memory (RAM) per node
#SBATCH --time=02:00:00                                 # Time limit (hh:mm:ss)
#SBATCH --partition=accelerated                         # GPU partition
#SBATCH --gres=gpu:1                                    # Request 1 GPU
#SBATCH --account=hk-project-pai00012                   # Project account ID

# ===== SHELL SCRIPT SECTION =====

echo "Starting chunk ${chunk_idx} job on node: \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Chunk: ${chunk_idx}/${TOTAL_CHUNKS}"

# Load modules for GPU and Python
module load devel/cuda/12.4

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# Activate virtual environment
source "\$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Install required packages for monitoring
pip install psutil

# Create and enter working directory on node-local storage
WORKDIR=\$TMPDIR/gpujob_chunk_${chunk_idx}_\$SLURM_JOB_ID
mkdir -p \$WORKDIR
cd \$WORKDIR

# Stage in: copy input data or model from the submit directory
cp "\$SLURM_SUBMIT_DIR/mcts.py" .

# Ensure results directory exists in the submit directory
mkdir -p "\$SLURM_SUBMIT_DIR/results"

# Run your GPU compute job for this chunk
python mcts.py \\
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \\
    --eval_model_name Qwen/Qwen2.5-3B-Instruct \\
    --output_file results.parquet \\
    --max-samples ${SAMPLES_PER_CHUNK} \\
    --num-chunks ${TOTAL_CHUNKS} \\
    --chunk-idx ${chunk_idx} \\
    --max_num_iterations 10

# Stage out: copy results back to the submit directory
cp results.parquet "\$SLURM_SUBMIT_DIR/results/chunk_${chunk_idx}_job_\${SLURM_JOB_ID}.parquet"

# Cleanup
cd ~
echo "Chunk ${chunk_idx} job completed."
EOF

    # Make the script executable
    chmod +x "chunk_jobs/chunk_${chunk_idx}.sh"
done

echo "Generated $TOTAL_CHUNKS chunk jobs in chunk_jobs/ directory"
echo ""
echo "To submit all jobs:"
echo "for i in {0..$((TOTAL_CHUNKS-1))}; do sbatch chunk_jobs/chunk_\$i.sh; done"
echo ""
echo "To submit a specific chunk (e.g., chunk 0):"
echo "sbatch chunk_jobs/chunk_0.sh"
