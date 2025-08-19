#!/bin/bash


# === Static SLURM directives (templated placeholders replaced) ===
# Adjust as needed before submitting.
#SBATCH --partition=accelerated
#SBATCH --job-name=ray_multi
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00   # Format: days-hours:minutes:seconds OR hours:minutes:seconds
## (Optional) constrain to specific nodes: uncomment next line and edit
# #SBATCH --nodelist=hkn0511,hkn0512

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=2
#SBATCH --exclusive

### Give all resources to a single Ray task, ray can manage the resources internally
# One Ray head/worker process per node that sees ALL GPUs on that node.
# IMPORTANT: Remove conflicting per-task GPU settings so Ray can detect all GPUs.
# Each node has 4 physical GPUs => 2 nodes * 4 = 8 total GPUs.
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
# (Optional) set CPUs per task based on your node CPU count
#SBATCH --cpus-per-task=16

# Load modules or your own conda environment here
# module load pytorch
# source /path/to/venv/bin/activate

# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address

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

echo "STARTING HEAD at $node_1"
# Launch the Ray head. Because this job step has all 4 GPUs on the head node (ntasks-per-node=1), Ray will register 4 GPUs.
HEAD_GPU_COUNT_CMD="nvidia-smi -L | wc -l"
head_num_gpus=$(srun --nodes=1 --ntasks=1 -w $node_1 bash -c "$HEAD_GPU_COUNT_CMD")
echo "Head node $node_1 reports $head_num_gpus GPUs"
srun --nodes=1 --ntasks=1 -w $node_1 \
  ray start --head --node-ip-address=$ip --port=6379 --redis-password=$redis_password --num-gpus=${head_num_gpus} --block &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= $worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  # Each worker node Ray process will also see all GPUs on that node.
  worker_num_gpus=$(srun --nodes=1 --ntasks=1 -w $node_i bash -c "$HEAD_GPU_COUNT_CMD")
  echo "Worker node $node_i reports $worker_num_gpus GPUs"
  srun --nodes=1 --ntasks=1 -w $node_i ray start --address $ip_head --redis-password=$redis_password --num-gpus=${worker_num_gpus} --block &
  sleep 5
done

##############################################################################################

#### call your code below
# Ensure the training driver connects to the existing Ray cluster instead of starting a local one.
export RAY_ADDRESS=auto
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "RAY cluster head: $ip_head"
echo "Expecting $SLURM_JOB_NUM_NODES nodes"

# (Optional) NCCL envs for multi-node stability; uncomment / adjust as needed
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_ASYNC_ERROR_HANDLING=1

# Override trainer nnodes & gpus-per-node if your config file differs (OmegaConf CLI style):
#   --trainer.nnodes=$SLURM_JOB_NUM_NODES --trainer.n_gpus_per_node=4

# === Launch your training driver (edit below) ===
# Enter package root so relative imports (..single_controller) work
cd EasyR1
echo "Executing: python -m verl.trainer.main trainer.nnodes=$SLURM_JOB_NUM_NODES trainer.n_gpus_per_node=${head_num_gpus}"
python -u -m verl.trainer.main trainer.nnodes=$SLURM_JOB_NUM_NODES trainer.n_gpus_per_node=${head_num_gpus}
