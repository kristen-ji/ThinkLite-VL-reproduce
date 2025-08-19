#!/bin/bash

# Script to submit multiple chunk jobs
# Usage: ./submit_chunks.sh [start_chunk] [end_chunk]

START_CHUNK=${1:-0}
END_CHUNK=${2:-9}  # Default: submit chunks 0-9

echo "Submitting chunks from ${START_CHUNK} to ${END_CHUNK}"

for chunk_id in $(seq $START_CHUNK $END_CHUNK); do
    echo "Submitting chunk ${chunk_id}..."
    
    # Submit the job with the chunk ID as an environment variable
    sbatch --export=CHUNK_ID=$chunk_id batch_script_chunk_template.sh
    
    # Wait a bit between submissions to avoid overwhelming the scheduler
    sleep 2
done

echo "Submitted chunks ${START_CHUNK} to ${END_CHUNK}"
echo "Check job status with: squeue -u $USER"
