#!/bin/bash

# ===== Multi-Node GPU Mapping Wrapper =====
# This script ensures proper GPU mapping before training

echo "=== Multi-Node GPU Mapping Setup ==="

# Get node information
HOSTNAME=$(hostname)
NODE_ID=$(echo $HOSTNAME | sed 's/hkn//' | sed 's/\.localdomain//')

echo "Hostname: $HOSTNAME"
echo "Node ID: $NODE_ID"

# Set node-specific environment variables
if [ "$NODE_ID" = "602" ]; then
    export NODE_RANK=0
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "Node 0 (hkn0602) - GPUs: 0,1,2,3"
elif [ "$NODE_ID" = "611" ]; then
    export NODE_RANK=1
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    echo "Node 1 (hkn0611) - GPUs: 4,5,6,7"
else
    echo "Unknown node ID: $NODE_ID"
    echo "Defaulting to Node 0 with GPUs 0,1,2,3"
    export NODE_RANK=0
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# Set distributed training environment variables
export WORLD_SIZE=8
export MASTER_ADDR=10.0.1.106
export MASTER_PORT=29500

echo "NODE_RANK: $NODE_RANK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Run GPU mapping verification
echo "=== Running GPU Mapping Verification ==="
python3 gpu_mapping.py

if [ $? -eq 0 ]; then
    echo "GPU mapping verification successful"
else
    echo "GPU mapping verification failed"
    exit 1
fi

# Launch the actual training script
echo "=== Launching Training ==="
bash examples/qwen2_5_vl_32b_geo3k_grpo.sh
