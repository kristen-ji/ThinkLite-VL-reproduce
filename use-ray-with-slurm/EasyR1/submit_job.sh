#!/bin/bash
# ===== SLURM Configuration =====
#SBATCH --job-name=GRPO_training_7B_fixed
#SBATCH --output=training_7b_fixed_%j.out
#SBATCH --error=training_7b_fixed_%j.err
#SBATCH --nodes=2  # Two-node setup for 7B model
#SBATCH --ntasks=2  # Exactly 2 tasks: 1 per node
#SBATCH --ntasks-per-node=1  # 1 task per node for proper coordination
#SBATCH --distribution=block  # Force tasks to be distributed across nodes
#SBATCH --cpus-per-task=16                # CPUs for 7B model
#SBATCH --exclusive  # Request exclusive access to nodes
#SBATCH --gres=gpu:8  # 8 GPUs total across 2 nodes (4 per node)
#SBATCH --mem=128000mb                    # 128GB RAM per node for 7B model
#SBATCH --mem-per-gpu=32000mb             # 32GB per GPU
#SBATCH --partition=accelerated
#SBATCH --time=12:00:00                   # Extended time for 2-node 7B training
#SBATCH --account=hk-project-pai00012

# ===== Environment Setup =====
echo "=== Environment Setup ==="

# Set environment variables for Ray
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DISABLE_LOGGING=1
export RAY_DEDUP_LOGS=1
export RAY_LOG_TO_STDERR=0
export RAY_TMPDIR=/tmp/ray_cluster
export RAY_GCS_RPC_TIMEOUT_MS=120000
export RAY_GCS_SERVER_RPC_TIMEOUT_MS=120000
export RAY_RAYLET_STARTUP_TIMEOUT_SECONDS=120
export RAY_WORKER_STARTUP_TIMEOUT_SECONDS=120
export RAY_AUTOSCALER_UPDATE_INTERVAL_S=60
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_enable_windows_or_osx_cluster=1
export RAY_USE_MULTI_PROCESSES_ON_LOCAL_HOST=1
export RAY_DISABLE_RAY_CLIENT=1
export RAY_IGNORE_UNHANDLED_ERRORS=1

# Set environment variables for PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export TORCH_USE_CUDA_DSA=1

# Set environment variables for CUDA graphs
export CUDA_GRAPH_CAPTURE_DISABLE=1

# Set environment variables for GPU detection
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SLURM_GPUS_ON_NODE=4
export SLURM_GPUS_PER_NODE=4
export ALLOW_GPU_UNDERPROVISION=1

# Set environment variables for HuggingFace cache - use remote workspace for better performance
export HF_HOME=/hkfs/work/workspace/scratch/st_st190232-myspace/hf_cache_7b
export TRANSFORMERS_CACHE=/hkfs/work/workspace/scratch/st_st190232-myspace/hf_cache_7b
export TMPDIR=/hkfs/work/workspace/scratch/st_st190232-myspace/tmp_7b

# Use short path for Ray temp to avoid socket filename length limit
export RAY_TMPDIR=/tmp/ray_7b_short

# Create necessary directories in remote workspace
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/hf_cache_7b
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/tmp_7b
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/ray_cluster_7b
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/datasets_cache_7b
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/models_cache_7b
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/torch_cache_7b
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/xdg_cache_7b
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/checkpoints_7b
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/temp_7b
mkdir -p /hkfs/work/workspace/scratch/st_st190232-myspace/cache_7b

# Create short Ray temp directory to avoid socket path length issues
mkdir -p "$RAY_TMPDIR"

mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$TMPDIR"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00012/st_st190232/ThinkLite-VL"

# ===== Ray Cluster Setup =====
echo "Setting up Ray cluster..."

# Get head node IP
head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
head_node_ip=$(scontrol show node $head_node | grep NodeAddr | awk '{print $2}')
if [ -z "$head_node_ip" ]; then
    head_node_ip=$(getent hosts $head_node | awk '{print $1}' | head -n1)
fi
if [ -z "$head_node_ip" ]; then
    if [ "$(hostname)" = "$head_node" ]; then
        head_node_ip=$(hostname -i | awk '{print $1}')
    else
        echo "ERROR: Cannot determine head node IP"
        exit 1
    fi
fi
port=6379

# Check if ports are available
if netstat -tuln 2>/dev/null | grep -q ":$port "; then
    echo "WARNING: Port $port is already in use, trying next available port..."
    port=6380
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        port=6381
    fi
    echo "Using port: $port"
fi

# Check dashboard port availability
dashboard_port=8265
if netstat -tuln 2>/dev/null | grep -q ":$dashboard_port "; then
    echo "WARNING: Dashboard port $dashboard_port is already in use, trying next available port..."
    dashboard_port=8266
    if netstat -tuln 2>/dev/null | grep -q ":$dashboard_port "; then
        dashboard_port=8267
    fi
    echo "Using dashboard port: $dashboard_port"
fi

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "Ray port: $port"
echo "Current hostname: $(hostname)"

# Debug SLURM node information
echo "=== SLURM Node Debug ==="
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Current hostname: $(hostname)"
echo "Total tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"

# Parse node list to get actual node names
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node_name=${nodes[0]}
worker_node_name=${nodes[1]}

echo "Parsed nodes: head=$head_node_name, worker=$worker_node_name"
echo "Current hostname: $(hostname)"

# Determine if this is the head node or worker node based on SLURM_PROCID
echo "SLURM_PROCID: $SLURM_PROCID"

# Set Ray configuration for SLURM
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_SCHEDULER_EVENTS=0
export RAY_DISABLE_USAGE_STATS=1
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_USE_MULTI_PROCESSES_ON_LOCAL_HOST=1

# Start Ray cluster using Ray's built-in multi-node startup
echo "=== STARTING RAY CLUSTER ==="

# Get node information from SLURM environment variables
head_node_name=$(hostname)
# Parse SLURM_JOB_NODELIST to get worker node - handle SLURM node list format
if [[ "$SLURM_JOB_NODELIST" == *"["* ]]; then
    # Handle SLURM node list format like "hkn[0602,0611]"
    base_name=$(echo $SLURM_JOB_NODELIST | sed 's/\[.*//')
    nodes=$(echo $SLURM_JOB_NODELIST | sed 's/.*\[\([^]]*\)\].*/\1/' | tr ',' '\n')
    all_nodes=()
    for node_num in $nodes; do
        all_nodes+=("${base_name}${node_num}")
    done
    head_node_name=${all_nodes[0]}
    worker_node_name=${all_nodes[1]}
else
    # Handle simple comma-separated list
    nodes=($(echo $SLURM_JOB_NODELIST | tr ',' '\n'))
    head_node_name=${nodes[0]}
    worker_node_name=${nodes[1]}
fi

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Head node: $head_node_name"
echo "Worker node: $worker_node_name"

# Validate worker node name
if [ -z "$worker_node_name" ] || [ "$worker_node_name" = "$head_node_name" ]; then
    echo "WARNING: No valid worker node found, using single-node setup"
    worker_node_name=""
fi

# Start head node
echo "Starting Ray head node on $head_node_name..."

# Debug GPU environment before starting Ray
echo "=== Pre-Ray GPU Environment ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-<not set>}"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-<not set>}"
echo "SLURM_NODEID: ${SLURM_NODEID:-<not set>}"
echo "SLURM_NNODES: ${SLURM_NNODES:-<not set>}"
echo "Current node: $(hostname)"

# Stop any existing Ray processes first
ray stop --force 2>/dev/null || true
sleep 5

# Start Ray head node with SLURM integration
echo "Starting Ray head node with reduced memory settings..."
head_output=$(ray start --head --dashboard-host=0.0.0.0 --port=$port --num-gpus=4 --disable-usage-stats --include-dashboard=true --object-store-memory=1000000000 --temp-dir="$RAY_TMPDIR" 2>&1)
head_exit_code=$?

if [ $head_exit_code -eq 0 ]; then
    echo "✅ Head node started successfully"
    echo "Head output: $head_output"
    
    # Extract GCS address
    gcs_address=$(echo "$head_output" | grep "GCS address:" | sed 's/.*GCS address: //')
    if [ -z "$gcs_address" ]; then
        gcs_address="$head_node_ip:$port"
        echo "Using fallback GCS address: $gcs_address"
    else
        echo "GCS address: $gcs_address"
    fi
    
    # Wait for Ray to be ready
    echo "Waiting for Ray head node to be ready..."
    sleep 30
    
    # Export Ray address for worker nodes
    export RAY_ADDRESS="$gcs_address"
    echo "RAY_ADDRESS set to: $RAY_ADDRESS"
    
    # Check initial Ray status
    echo "Checking initial Ray status..."
    ray status
    
    # Check Ray cluster resources
    echo "=== Ray Cluster Resources ==="
    if command -v ray >/dev/null 2>&1; then
        echo "Ray cluster resources:"
        ray cluster-resources 2>/dev/null || echo "Failed to get cluster resources"
        echo "Ray available resources:"
        ray available-resources 2>/dev/null || echo "Failed to get available resources"
    else
        echo "Ray command not found in PATH"
    fi
    
    echo "✅ Ray head node ready at $gcs_address"
    
    # Parse SLURM node list to get actual node names
    echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
    head_node_name=${nodes[0]}
    worker_node_name=${nodes[1]}
    
    echo "Parsed nodes: head=$head_node_name, worker=$worker_node_name"
    
    # Start worker node to join the cluster (only if we have a valid worker node)
    if [ -n "$worker_node_name" ] && [ "$worker_node_name" != "$head_node_name" ]; then
        echo "Starting worker node on $worker_node_name..."
        echo "Command: ray start --address=$gcs_address --num-gpus=4 --disable-usage-stats"
        
        # Start worker node
        worker_output=$(ray start --address=$gcs_address --num-gpus=4 --disable-usage-stats 2>&1)
        worker_exit_code=$?
        
        if [ $worker_exit_code -eq 0 ]; then
            echo "✅ Worker node started successfully"
            echo "Worker output: $worker_output"
        else
            echo "❌ Worker node failed to start (exit code: $worker_exit_code)"
            echo "Worker error output: $worker_output"
            echo "⚠️  Proceeding with single-node training"
        fi
        
        # Wait for worker to join cluster
        echo "Waiting for worker node to join cluster..."
        sleep 30
        
        # Check Ray status after worker joins
        echo "Checking Ray status after worker joins..."
        ray status
        
        # Verify worker node resources are visible
        echo "Verifying worker node resources..."
        if command -v ray >/dev/null 2>&1; then
            echo "Ray cluster resources after worker joins:"
            ray cluster-resources 2>/dev/null || echo "Failed to get cluster resources"
            echo "Ray available resources after worker joins:"
            ray available-resources 2>/dev/null || echo "Failed to get available resources"
        fi
    else
        echo "No valid worker node found, using single-node setup"
    fi
    
    # Check final Ray status
    echo "Checking final Ray status..."
    ray status
    
    # Debug GPU detection
    echo "=== GPU Detection Debug ==="
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
    echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-<not set>}"
    echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-<not set>}"
    echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "SLURM_NODEID: ${SLURM_NODEID:-<not set>}"
    echo "SLURM_NNODES: ${SLURM_NNODES:-<not set>}"
    
    # Verify cluster has both nodes
    echo "Verifying cluster has both nodes..."
    ray_status_output=$(ray status 2>/dev/null)
    echo "Ray status output:"
    echo "$ray_status_output"
    
    # Check for 2 nodes and 8 total GPUs (4 per node)
    echo "Expected: 2 nodes with 4 GPUs each = 8 total GPUs"
    if echo "$ray_status_output" | grep -q "2 node"; then
        echo "✅ SUCCESS: 2 nodes detected in cluster"
        
        # Check total GPU count across all nodes
        total_gpus=$(echo "$ray_status_output" | grep -o '[0-9]\+\.0 GPU' | head -1 | grep -o '[0-9]\+')
        if [ -n "$total_gpus" ] && [ "$total_gpus" -ge 8 ]; then
            echo "✅ SUCCESS: $total_gpus GPUs detected across all nodes (expected 8+)"
        else
            echo "⚠️  WARNING: Only $total_gpus GPUs detected (expected 8+)"
            echo "This may cause GPU underprovisioning issues"
        fi
    elif echo "$ray_status_output" | grep -q "8\.0 GPU"; then
        echo "✅ SUCCESS: 8 GPUs detected in cluster"
    else
        echo "⚠️  WARNING: Only 1 node detected or insufficient GPUs"
        echo "This may cause training to fail if 8 GPUs are required"
        echo "Proceeding anyway..."
    fi
    
    # Additional debugging for SLURM integration
    echo "=== SLURM INTEGRATION DEBUG ==="
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_NODEID: $SLURM_NODEID"
    echo "SLURM_PROCID: $SLURM_PROCID"
    echo "SLURM_NNODES: $SLURM_NNODES"
    echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "Current hostname: $(hostname)"
    echo "RAY_ADDRESS: $RAY_ADDRESS"
    
    # Start training script - let Ray's training framework handle multi-node discovery
    echo "=== STARTING TRAINING ==="
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    
    TRAINING_SCRIPT="examples/qwen2_5_vl_7b_geo3k_grpo.sh"
    export RAY_ADDRESS="$gcs_address"
    
    echo "Executing training script: $TRAINING_SCRIPT"
    echo "RAY_ADDRESS set to: $RAY_ADDRESS"
    echo "Training framework will automatically discover SLURM nodes and GPUs"
    
    bash "$TRAINING_SCRIPT"
    
else
    echo "ERROR: Ray head node failed to start"
    echo "Attempting to start Ray with minimal configuration..."
    
    # Try with minimal Ray configuration
    ray stop --force 2>/dev/null || true
    sleep 5
    
    minimal_head_output=$(ray start --head --port=$port --dashboard-port=$dashboard_port --num-gpus=4 --disable-usage-stats 2>&1)
    minimal_head_exit_code=$?
    
    if [ $minimal_head_exit_code -eq 0 ]; then
        echo "✅ Ray started with minimal configuration"
        echo "Minimal head output: $minimal_head_output"
        
        # Extract GCS address for minimal setup
        gcs_address=$(echo "$minimal_head_output" | grep "GCS address:" | sed 's/.*GCS address: //')
        if [ -z "$gcs_address" ]; then
            gcs_address="$head_node_ip:$port"
        fi
        export RAY_ADDRESS="$gcs_address"
        
        echo "Proceeding with minimal Ray setup..."
        
        # Start training script with minimal setup
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        cd "$SCRIPT_DIR"
        
        TRAINING_SCRIPT="examples/qwen2_5_vl_7b_geo3k_grpo.sh"
        
        echo "Executing training script: $TRAINING_SCRIPT"
        echo "RAY_ADDRESS set to: $RAY_ADDRESS"
        echo "Training framework will use minimal Ray setup"
        
        bash "$TRAINING_SCRIPT"
    else
        echo "❌ Ray failed to start even with minimal configuration"
        echo "Minimal head error: $minimal_head_output"
        echo "Exiting due to Ray startup failure"
        exit 1
    fi
fi
