#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
# Set memory optimization environment variables
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:16
# export CUDA_LAUNCH_BLOCKING=1
# export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1
# export TORCH_USE_CUDA_DSA=1
# export CUDA_GRAPH_CAPTURE_DISABLE=1
# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
# export RAY_DISABLE_IMPORT_WARNING=1
# export HF_HUB_ENABLE_HF_TRANSFER=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # Using VL model with memory-optimized config

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.fsdp.enable_cpu_offload=false \
    worker.rollout.enforce_eager=true \
    worker.actor.fsdp.torch_dtype=fp16 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    data.rollout_batch_size=16 \
    worker.actor.global_batch_size=16 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=4
