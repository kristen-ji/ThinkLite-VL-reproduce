#!/bin/bash

# Enhanced script to submit multiple chunk jobs in parallel
# Usage: ./submit_chunks_parallel.sh [start_chunk] [end_chunk] [max_concurrent_jobs]

START_CHUNK=${1:-0}
END_CHUNK=${2:-9}
MAX_CONCURRENT=${3:-5}  # Default: max 5 concurrent jobs

echo "Submitting chunks from ${START_CHUNK} to ${END_CHUNK}"
echo "Maximum concurrent jobs: ${MAX_CONCURRENT}"

# Function to count running jobs
count_running_jobs() {
    squeue -u $USER -h | wc -l
}

# Function to wait for job slots to become available
wait_for_slot() {
    while [ $(count_running_jobs) -ge $MAX_CONCURRENT ]; do
        echo "Waiting for job slots... (currently running: $(count_running_jobs))"
        sleep 30
    done
}

# Submit jobs with controlled concurrency
for chunk_id in $(seq $START_CHUNK $END_CHUNK); do
    # Wait if we've reached the maximum concurrent jobs
    wait_for_slot
    
    echo "Submitting chunk ${chunk_id}..."
    
    # Submit the job with the chunk ID as an environment variable
    job_id=$(sbatch --export=CHUNK_ID=$chunk_id batch_script_chunk_template.sh | awk '{print $4}')
    
    echo "Submitted chunk ${chunk_id} with job ID: ${job_id}"
    
    # Small delay between submissions
    sleep 5
done

echo "All chunks submitted!"
echo "Check job status with: squeue -u $USER"
echo "Monitor progress with: tail -f log_chunk_*_*.out"
