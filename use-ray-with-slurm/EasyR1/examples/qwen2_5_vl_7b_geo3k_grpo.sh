#!/bin/bash

# ===== Qwen2.5-VL-7B GRPO Training Script =====
# This script configures and runs GRPO training for the 7B model
# 
# Optional: Set HF_TOKEN environment variable if accessing gated models:
# export HF_TOKEN="your_huggingface_token_here"

# Set model path
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# Set your Hugging Face token here (for gated models)
export HF_TOKEN="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"

echo "=== Qwen2.5-VL-7B GRPO Training Setup ==="
echo "Model: $MODEL_PATH"
echo "Training Type: GRPO (Group Relative Policy Optimization)"
echo "Dataset: Geometry3K"

# Environment variables specific to 7B model - let Ray handle GPU mapping
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:24,garbage_collection_threshold:0.2,roundup_power2_divisions:16
export OMP_NUM_THREADS=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export CUDA_GRAPH_CAPTURE_DISABLE=1
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export CUDA_MEMORY_FRACTION=0.7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:24,garbage_collection_threshold:0.2,roundup_power2_divisions:16
export CUDA_LAUNCH_BLOCKING=1

export TORCH_CUDNN_V8_API_DISABLED=1
export TORCH_CUDNN_V8_API_ENABLED=0

# Hugging Face model loading environment variables
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
# Add Hugging Face token if available (for gated models)
if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=RING
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_OVERLOAD_THRESHOLD=8192
export NCCL_CROSS_NIC=0
export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=8
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG_SUBSYS=ALL

# Fix GPU device mapping for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_RANK=0
export NCCL_WORLD_SIZE=8
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=RING
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# Ray setup for 7B model
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_TASK_RETRY_DELAY_MS=2000
export RAY_OBJECT_STORE_MEMORY=2000000000  # 2GB for 7B model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:24,garbage_collection_threshold:0.2,roundup_power2_divisions:16
export ALLOW_GPU_UNDERPROVISION=1  # Allow GPU underprovisioning for memory efficiency

# HuggingFace optimizations for large model
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Add these environment variables to reduce concurrent downloads
export HF_HUB_DOWNLOAD_TIMEOUT=1800  # 30 minutes
export HF_HUB_RETRY_DELAY=60  # 1 minute between retries
export HF_HUB_MAX_RETRIES=30  # More retries
export HF_HUB_CONCURRENT_DOWNLOADS=1  # Limit concurrent downloads

echo "Environment configured for 7B model training"
echo "NODE_RANK: ${NODE_RANK:-0}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "MASTER_ADDR: ${MASTER_ADDR:-10.0.1.106}"
echo "MASTER_PORT: ${MASTER_PORT:-29500}"

# Check if Ray cluster is available
if [ -n "$RAY_ADDRESS" ]; then
    echo "Ray cluster detected at: $RAY_ADDRESS"
    echo "Multi-node training will be used"
else
    echo "No Ray cluster detected"
    echo "Ray will be initialized by the training script itself"
fi
# Launch training with all parameters for 7B model
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.video_key=videos \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.fsdp.enable_cpu_offload=true \
    worker.actor.fsdp.enable_full_shard=true \
    worker.actor.fsdp.enable_rank0_init=true \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.enforce_eager=true \
    worker.rollout.max_num_batched_tokens=448 \
    worker.ref.fsdp.enable_cpu_offload=false \
    worker.ref.offload.offload_params=false \
    data.rollout_batch_size=2 \
    data.mini_rollout_batch_size=2 \
    data.val_batch_size=2 \
    worker.actor.global_batch_size=2 \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo_phase3 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=2
