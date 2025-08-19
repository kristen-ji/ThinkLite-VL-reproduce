#!/bin/bash

# Script to monitor parallel chunk jobs
# Usage: ./monitor_chunks.sh

echo "=== PARALLEL CHUNK MONITORING ==="
echo "Time: $(date)"
echo ""

# Show running jobs
echo "=== RUNNING JOBS ==="
squeue -u $USER --format="%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R" | head -1
squeue -u $USER --format="%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R" | grep "ThinkLite_chunk"

echo ""
echo "=== JOB COUNTS ==="
TOTAL_JOBS=$(squeue -u $USER -h | wc -l)
RUNNING_JOBS=$(squeue -u $USER -h | grep "R" | wc -l)
PENDING_JOBS=$(squeue -u $USER -h | grep "PD" | wc -l)
COMPLETED_JOBS=$(squeue -u $USER -h | grep "CD" | wc -l)

echo "Total jobs: $TOTAL_JOBS"
echo "Running: $RUNNING_JOBS"
echo "Pending: $PENDING_JOBS"
echo "Completed: $COMPLETED_JOBS"

echo ""
echo "=== RECENT LOG FILES ==="
# Show recent log files
find . -name "log_chunk_*_*.out" -type f -exec ls -lt {} + | head -5

echo ""
echo "=== LATEST PROGRESS ==="
# Show latest progress from log files
for log_file in $(find . -name "log_chunk_*_*.out" -type f | head -3); do
    echo "--- $log_file ---"
    tail -3 "$log_file" 2>/dev/null || echo "No recent output"
    echo ""
done

echo "=== COMPLETED CHUNK FILES ==="
# Show completed chunk files
find ./results -name "chunk_*_*.parquet" -type f 2>/dev/null | wc -l | xargs echo "Completed chunk files:"
