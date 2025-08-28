#!/bin/bash

# ===== SLURM Configuration =====
#SBATCH --job-name=ThinkLite_training
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated-h200 #need to change when using different partition
#SBATCH --time=08:00:00
#SBATCH --account=hk-project-pai00012

# ===== Environment Setup =====
echo "Starting ThinkLite training job on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Load any required modules
module load devel/cuda/12.4

# Activate your virtual environment (if needed)
# source /path/to/your/venv/bin/activate

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# ===== Ray Cluster Setup =====
echo "Setting up Ray cluster for 2 nodes..."

# Get node information from SLURM
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# Get the head node (first node)
node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address)

# Handle IPv6 addresses
if [[ $ip == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$ip"
  if [[ ${#ADDR[0]} > 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "We detect space in ip! You are using IPV6 address. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

# Generate Redis password
redis_password=$(uuidgen)
export redis_password

# Start Ray head on the first node
echo "Starting Ray head on $node_1"
head_num_gpus=$(srun --nodes=1 --ntasks=1 -w $node_1 bash -c "nvidia-smi -L | wc -l")
echo "Head node $node_1 reports $head_num_gpus GPUs"

srun --nodes=1 --ntasks=1 -w $node_1 \
  ray start --head --node-ip-address=$ip --port=6379 --redis-password=$redis_password --num-gpus=${head_num_gpus} --block &
sleep 30

# Start Ray workers on the second node
if [ ${#nodes_array[@]} -gt 1 ]; then
  node_2=${nodes_array[1]}
  echo "Starting Ray worker on $node_2"
  worker_num_gpus=$(srun --nodes=1 --ntasks=1 -w $node_2 bash -c "nvidia-smi -L | wc -l")
  echo "Worker node $node_2 reports $worker_num_gpus GPUs"

  srun --nodes=1 --ntasks=1 -w $node_2 ray start --address $ip_head --redis-password=$redis_password --num-gpus=${worker_num_gpus} --block &
  sleep 10
fi

# Wait for Ray cluster to be ready and verify connection
echo "Waiting for Ray cluster to be ready..."
sleep 15

# Set Ray address to connect to the cluster we just created
export RAY_ADDRESS=auto

# Verify Ray cluster is working
echo "Verifying Ray cluster connection..."
ray status --address=auto || {
  echo "ERROR: Ray cluster is not accessible. Trying alternative connection..."
  export RAY_ADDRESS=$ip_head
  ray status --address=$ip_head || {
    echo "ERROR: Ray cluster setup failed!"
    exit 1
  }
}

#Ray status
ray status

# Debug: Print SLURM environment variables
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

# Calculate total GPUs across all nodes
total_gpus=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
echo "Total GPUs allocated: $total_gpus"

# ===== Run Training =====
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/EasyR1"

echo "Current directory: $(pwd)"
echo "Executing training script..."
bash examples/qwen2_5_vl_32b_geo3k_grpo.sh # need to change when we use different model

# ===== Cleanup =====
echo "Cleaning up Ray cluster..."
ray stop

echo "Training completed on $(date)"
