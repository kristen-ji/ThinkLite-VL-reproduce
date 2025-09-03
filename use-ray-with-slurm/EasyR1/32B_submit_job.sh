#!/bin/bash
# ===== SLURM Configuration =====
#SBATCH --job-name=GRPO_training_32B_fixed
#SBATCH --output=training_32b_fixed_%j.out
#SBATCH --error=training_32b_fixed_%j.err
#SBATCH --nodes=2  # Two-node setup for 32B model
#SBATCH --ntasks=2  # Exactly 2 tasks: 1 per node
#SBATCH --ntasks-per-node=1  # 1 task per node for proper coordination
#SBATCH --distribution=block  # Force tasks to be distributed across nodes
#SBATCH --exclusive  # Request exclusive access to nodes
#SBATCH --gres=gpu:8  # 8 GPUs total across 2 nodes (4 per node)
#SBATCH --mem=256000mb                    # 256GB RAM per node for 32B model
#SBATCH --mem-per-gpu=80000mb             # 80GB per GPU for 32B model
#SBATCH --cpus-per-task=32                # More CPUs for 32B model
#SBATCH --partition=accelerated
#SBATCH --time=12:00:00                   # Extended time for 2-node 32B training
#SBATCH --account=hk-project-pai00012

# ===== Environment Setup =====
echo "=== Environment Setup ==="

# Use workspace-backed directories for caches and temporary files
WORKSPACE_DIR="/hkfs/work/workspace/scratch/st_st190232-myspace"
MODEL_CACHE_DIR="$WORKSPACE_DIR/model_cache"
HF_CACHE_DIR="$WORKSPACE_DIR/hf_cache"
TRANSFORMERS_CACHE_DIR="$WORKSPACE_DIR/transformers_cache"
TORCH_CACHE_DIR="$WORKSPACE_DIR/torch_cache"
TEMP_DIR="$WORKSPACE_DIR/temp"
RAY_DIR="$WORKSPACE_DIR/ray_cluster"
CHECKPOINT_DIR="$WORKSPACE_DIR/checkpoints"
# Use shorter path for Ray temp to avoid socket filename length limit
RAY_TEMP_DIR="/tmp/ray_32b_short"

mkdir -p "$MODEL_CACHE_DIR" "$HF_CACHE_DIR" "$TRANSFORMERS_CACHE_DIR" "$TORCH_CACHE_DIR" "$TEMP_DIR" "$RAY_DIR" "$CHECKPOINT_DIR" "$RAY_TEMP_DIR"

# Set environment variables for Ray
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DISABLE_LOGGING=1
export RAY_DEDUP_LOGS=1
export RAY_LOG_TO_STDERR=0
export RAY_TMPDIR="$RAY_TEMP_DIR"
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
export SLURM_GPUS_ON_NODE=8
export SLURM_GPUS_PER_NODE=4
export ALLOW_GPU_UNDERPROVISION=1

# Set environment variables for HuggingFace/Transformers/Torch caches and tmp
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HF_MODELS_CACHE="$HF_CACHE_DIR/models"
export TORCH_HOME="$TORCH_CACHE_DIR"
export TMPDIR="$TEMP_DIR"
export TEMP="$TEMP_DIR"
export TMP="$TEMP_DIR"

# Create necessary directories (workspace-backed)
mkdir -p "$RAY_DIR"
mkdir -p "$HF_CACHE_DIR"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/home/hk-project-pai00012/st_st190232/ThinkLite-VL"

# ===== Ray Cluster Setup =====
echo "Setting up Ray cluster..."

# Debug SLURM environment
echo "=== SLURM ENVIRONMENT DEBUG ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "Current hostname: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Current user: $(whoami)"
echo "SLURM job distribution:"
squeue -j $SLURM_JOB_ID -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "Failed to get SLURM job status"
echo "=================================="

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

# Validate node names
if [ -z "$head_node_name" ] || [ -z "$worker_node_name" ]; then
    echo "ERROR: Failed to parse node names from SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "Falling back to single-node setup"
    head_node_name=$(hostname)
    worker_node_name=""
fi

# Set Ray configuration for SLURM
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_SCHEDULER_EVENTS=0
export RAY_DISABLE_USAGE_STATS=1
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_USE_MULTI_PROCESSES_ON_LOCAL_HOST=1
export RAY_DEDUP_LOGS=1
export RAY_LOG_TO_STDERR=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_TASK_RETRY_DELAY_MS=2000

# Start Ray cluster using Ray's built-in multi-node startup
echo "=== STARTING RAY CLUSTER ==="

# Get node information from SLURM environment variables
head_node_name=$(hostname)
# Parse SLURM_JOB_NODELIST to get worker node - handle SLURM node list format
echo "Raw SLURM_JOB_NODELIST: '$SLURM_JOB_NODELIST'"

if [[ "$SLURM_JOB_NODELIST" == *"["* ]]; then
    # Handle SLURM node list format like "hkn[0602,0611]" or "hkn[0701-0702]"
    echo "Detected bracket format, parsing..."
    
    # Use a completely different approach - extract base name and content separately
    base_name=$(echo "$SLURM_JOB_NODELIST" | grep -o '^[^[]*')
    bracket_content=$(echo "$SLURM_JOB_NODELIST" | grep -o '\[[^]]*\]' | tr -d '[]')
    
    echo "Base name: '$base_name'"
    echo "Bracket content: '$bracket_content'"
    echo "Full SLURM_JOB_NODELIST: '$SLURM_JOB_NODELIST'"
    
    # Check if bracket content contains a range (e.g., 0701-0702)
    if [[ "$bracket_content" == *"-"* ]]; then
        echo "Detected range within brackets, parsing range..."
        
        # Use a simple approach - split on dash
        IFS='-' read -ra range_parts <<< "$bracket_content"
        start_num="${range_parts[0]}"
        end_num="${range_parts[1]}"
        
        echo "Raw bracket content: '$bracket_content'"
        echo "Start number: '$start_num'"
        echo "End number: '$end_num'"
        
        # Validate the numbers are numeric
        if [[ "$start_num" =~ ^[0-9]+$ ]] && [[ "$end_num" =~ ^[0-9]+$ ]]; then
            echo "Valid numeric range detected"
            
            # Generate node names for the range
            all_nodes=()
            for ((i=start_num; i<=end_num; i++)); do
                node_name="${base_name}${i}"
                echo "Generated node name: $node_name"
                all_nodes+=("$node_name")
            done
        else
            echo "❌ ERROR: Invalid numeric range: start='$start_num', end='$end_num'"
            echo "Falling back to comma-separated parsing..."
            IFS=',' read -ra nodes <<< "$bracket_content"
            all_nodes=()
            for node_num in "${nodes[@]}"; do
                all_nodes+=("${base_name}${node_num}")
            done
        fi
    else
        echo "Detected comma-separated list within brackets, parsing..."
        IFS=',' read -ra nodes <<< "$bracket_content"
        all_nodes=()
        for node_num in "${nodes[@]}"; do
            all_nodes+=("${base_name}${node_num}")
        done
    fi
    
    head_node_name=${all_nodes[0]}
    worker_node_name=${all_nodes[1]}
    echo "Parsed bracket format: head=$head_node_name, worker=$worker_node_name"
elif [[ "$SLURM_JOB_NODELIST" == *"-"* ]]; then
    # Handle SLURM node list format like "hkn0701-0702" (range)
    echo "Detected range format, parsing..."
    base_name=$(echo $SLURM_JOB_NODELIST | sed 's/[0-9].*//')
    range=$(echo $SLURM_JOB_NODELIST | sed 's/.*[^0-9]\([0-9]*-[0-9]*\).*/\1/')
    start_num=$(echo $range | cut -d'-' -f1)
    end_num=$(echo $range | cut -d'-' -f2)
    echo "Range: $start_num to $end_num"
    
    # Generate node names for the range
    all_nodes=()
    for ((i=start_num; i<=end_num; i++)); do
        # Pad with leading zeros if needed (e.g., 0701, 0702)
        padded_num=$(printf "%04d" $i)
        all_nodes+=("${base_name}${padded_num}")
    done
    head_node_name=${all_nodes[0]}
    worker_node_name=${all_nodes[1]}
    echo "Parsed range format: head=$head_node_name, worker=$worker_node_name"
else
    # Handle simple comma-separated list
    echo "Detected comma format, parsing..."
    nodes=($(echo $SLURM_JOB_NODELIST | tr ',' '\n'))
    head_node_name=${nodes[0]}
    worker_node_name=${nodes[1]}
    echo "Parsed comma format: head=$head_node_name, worker=$worker_node_name"
fi

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "Current hostname: $(hostname)"
echo "Head node: $head_node_name"
echo "Worker node: $worker_node_name"
echo "Current node is: $(hostname)"
echo "Expected to be on: $head_node_name"

# Verify we're on the head node
if [ "$(hostname)" != "$head_node_name" ]; then
    echo "⚠️  WARNING: Current node $(hostname) is not the expected head node $head_node_name"
    echo "This may cause issues with Ray cluster setup"
fi

# Validate worker node name
if [ -z "$worker_node_name" ] || [ "$worker_node_name" = "$head_node_name" ]; then
    echo "WARNING: No valid worker node found, using single-node setup"
    worker_node_name=""
fi

# Verify we're on one of the expected nodes
current_hostname=$(hostname)
current_hostname_short=$(echo $current_hostname | sed 's/\.localdomain//')
echo "Current hostname: $current_hostname"
echo "Current hostname (short): $current_hostname_short"
echo "Expected head node: $head_node_name"
echo "Expected worker node: $worker_node_name"

# Check if current node matches either head or worker
if [ "$current_hostname_short" = "$head_node_name" ]; then
    echo "✅ SUCCESS: Current node matches expected head node"
    is_head_node=true
elif [ "$current_hostname_short" = "$worker_node_name" ]; then
    echo "✅ SUCCESS: Current node matches expected worker node"
    is_head_node=false
    # If we're on the worker node, we need to wait for the head node to start
    echo "⚠️  WARNING: Running on worker node, waiting for head node to start Ray cluster..."
else
    echo "❌ ERROR: Current node $current_hostname_short doesn't match expected nodes"
    echo "Expected: $head_node_name or $worker_node_name"
    echo "This may cause Ray cluster setup issues"
    is_head_node=true  # Assume we're on head node as fallback
fi

# Start head node (only if we're on the head node)
if [ "$is_head_node" = true ]; then
    echo "Starting Ray head node on $head_node_name..."
    
    # Stop any existing Ray processes first
    ray stop --force 2>/dev/null || true
    sleep 5
else
    echo "Not on head node, waiting for head node to start Ray cluster..."
    echo "Current node: $current_hostname_short"
    echo "Expected head node: $head_node_name"
    echo "Waiting for Ray cluster to be available..."
    
    # Wait for head node to start Ray cluster
    while ! ray status >/dev/null 2>&1; do
        echo "Waiting for Ray cluster to be available..."
        sleep 10
    done
    
    echo "✅ Ray cluster is now available!"
    exit 0  # Worker node should exit after joining
fi

# Start Ray head node with SLURM integration
echo "Starting Ray head node with reduced memory settings..."
head_output=$(ray start --head --dashboard-host=0.0.0.0 --port=6379 --num-gpus=4 --disable-usage-stats --include-dashboard=true --object-store-memory=1000000000 --temp-dir="$RAY_TEMP_DIR" 2>&1)
head_exit_code=$?

if [ $head_exit_code -eq 0 ]; then
    echo "✅ Head node started successfully"
    echo "Head output: $head_output"
    
    # Extract GCS address
    gcs_address=$(echo "$head_output" | grep "GCS address:" | sed 's/.*GCS address: //')
    echo "GCS address: $gcs_address"
    
    # Verify GCS address is not empty
    if [ -z "$gcs_address" ]; then
        echo "ERROR: GCS address is empty. Using fallback address..."
        gcs_address="$head_node_ip:6379"
        echo "Using fallback GCS address: $gcs_address"
    fi
    
    # Wait for Ray to be ready
    echo "Waiting for Ray head node to be ready..."
    sleep 30
    
    # Start worker node to join the cluster using SLURM srun
    if [ -n "$worker_node_name" ]; then
        echo "=== WORKER NODE STARTUP DEBUG ==="
        echo "Starting worker node on $worker_node_name using SLURM srun..."
        echo "GCS address: $gcs_address"
        echo "Head node IP: $head_node_ip"
        echo "Current hostname: $(hostname)"
        echo "Worker node name: $worker_node_name"
        
        # Test network connectivity to worker node
        echo "Testing network connectivity to worker node..."
        if ping -c 1 "$worker_node_name" >/dev/null 2>&1; then
            echo "✅ Network connectivity to $worker_node_name: SUCCESS"
        else
            echo "❌ Network connectivity to $worker_node_name: FAILED"
        fi
        
        # Test if we can reach the worker node via SLURM
        echo "Testing SLURM access to worker node..."
        test_slurm=$(srun --nodes=1 --ntasks=1 --nodelist=$worker_node_name hostname 2>&1)
        if [ $? -eq 0 ]; then
            echo "✅ SLURM access to $worker_node_name: SUCCESS (hostname: $test_slurm)"
        else
            echo "❌ SLURM access to $worker_node_name: FAILED"
            echo "SLURM test output: $test_slurm"
        fi
        
        # Check if Ray is already running on worker node
        echo "Checking if Ray is already running on worker node..."
        ray_check=$(srun --nodes=1 --ntasks=1 --nodelist=$worker_node_name ray status 2>&1)
        echo "Ray status on worker node: $ray_check"
        
        # Stop any existing Ray processes on worker node
        echo "Stopping any existing Ray processes on worker node..."
        srun --nodes=1 --ntasks=1 --nodelist=$worker_node_name ray stop --force 2>/dev/null || true
        sleep 5
        
        echo "Command: srun --nodes=1 --ntasks=1 --nodelist=$worker_node_name ray start --address=$gcs_address --num-gpus=4 --disable-usage-stats --object-store-memory=1000000000 --temp-dir=$RAY_TEMP_DIR"
        
        # Start worker node on the actual worker node using srun with verbose output
        echo "Starting worker node with verbose SLURM output..."
        worker_output=$(srun --verbose --nodes=1 --ntasks=1 --nodelist=$worker_node_name ray start --address=$gcs_address --num-gpus=4 --disable-usage-stats --object-store-memory=1000000000 --temp-dir=$RAY_TEMP_DIR 2>&1)
        worker_exit_code=$?
        
        echo "Worker startup exit code: $worker_exit_code"
        echo "Worker startup output: $worker_output"
        
        if [ $worker_exit_code -eq 0 ]; then
            echo "✅ Worker node started successfully via SLURM"
        else
            echo "❌ Worker node failed to start via SLURM (exit code: $worker_exit_code)"
            echo "⚠️  Proceeding with single-node training"
            
            # Additional debugging for worker node failure
            echo "=== ADDITIONAL WORKER DEBUG ==="
            echo "Checking SLURM job status..."
            squeue_output=$(squeue -j $SLURM_JOB_ID -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" 2>&1)
            echo "SLURM job status: $squeue_output"
            
            echo "Checking SLURM node list..."
            scontrol_output=$(scontrol show hostnames $SLURM_JOB_NODELIST 2>&1)
            echo "SLURM node list: $scontrol_output"
            
            echo "Checking if worker node is accessible via SSH..."
            ssh_test=$(srun --nodes=1 --ntasks=1 --nodelist=$worker_node_name whoami 2>&1)
            echo "SSH test to worker node: $ssh_test"
        fi
        
        # Wait for worker to join cluster
        echo "Waiting for worker node to join cluster..."
        sleep 30
        
        # Additional verification - check if worker node is actually visible
        echo "Verifying worker node visibility in cluster..."
        ray_status_after_worker=$(ray status 2>/dev/null)
        echo "Ray status after worker startup: $ray_status_after_worker"
        
        if echo "$ray_status_after_worker" | grep -q "2 node"; then
            echo "✅ SUCCESS: 2 nodes now visible in Ray cluster"
        else
            echo "⚠️  WARNING: Still only 1 node visible after worker startup"
            echo "Worker node may not have joined properly"
            
            # Try to manually check worker node status
            echo "Attempting to check worker node status directly..."
            if [ -n "$worker_node_name" ]; then
                worker_status=$(srun --nodes=1 --ntasks=1 --nodelist=$worker_node_name ray status 2>/dev/null)
                echo "Worker node Ray status: $worker_status"
                
                # Check if Ray process is running on worker node
                echo "Checking Ray processes on worker node..."
                ray_processes=$(srun --nodes=1 --ntasks=1 --nodelist=$worker_node_name ps aux | grep ray 2>/dev/null)
                echo "Ray processes on worker node: $ray_processes"
                
                # Check Ray logs on worker node
                echo "Checking Ray logs on worker node..."
                ray_logs=$(srun --nodes=1 --ntasks=1 --nodelist=$worker_node_name ls -la /tmp/ray_* 2>/dev/null)
                echo "Ray logs on worker node: $ray_logs"
            fi
        fi
    else
        echo "No worker node configured, using single-node setup"
    fi
    
    # Check Ray status
    echo "Checking Ray status..."
    ray status
    
    # Verify cluster has both nodes before starting training
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
            echo "✅ SUCCESS: $total_gpus GPUs detected across all nodes - expected 8+"
        else
            echo "⚠️  WARNING: Only $total_gpus GPUs detected - expected 8+"
            echo "This may cause GPU underprovisioning issues"
        fi
    elif echo "$ray_status_output" | grep -q "8\.0 GPU"; then
        echo "✅ SUCCESS: 8 GPUs detected in cluster"
    elif echo "$ray_status_output" | grep -q "4\.0 GPU"; then
        echo "⚠️  WARNING: Only 4 GPUs detected - single node"
        echo "This may cause training to fail if 8 GPUs are required"
        echo "Proceeding anyway..."
    else
        echo "⚠️  WARNING: Only 1 node detected or insufficient GPUs"
        echo "This may cause training to fail if 8 GPUs are required"
        echo "Proceeding anyway..."
    fi
    
    # Start training script - let Ray's training framework handle multi-node discovery
    echo "=== STARTING TRAINING ==="
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    
    TRAINING_SCRIPT="examples/qwen2_5_vl_32b_geo3k_grpo.sh"
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
    
    minimal_head_output=$(ray start --head --port=6379 --num-gpus=4 --disable-usage-stats --object-store-memory=500000000 --temp-dir="$RAY_TEMP_DIR" 2>&1)
    minimal_head_exit_code=$?
    
    if [ $minimal_head_exit_code -eq 0 ]; then
        echo "✅ Ray started with minimal configuration"
        echo "Minimal head output: $minimal_head_output"
        
        # Extract GCS address for minimal setup
        gcs_address=$(echo "$minimal_head_output" | grep "GCS address:" | sed 's/.*GCS address: //')
        if [ -z "$gcs_address" ]; then
            gcs_address="$head_node_ip:6379"
        fi
        export RAY_ADDRESS="$gcs_address"
        
        echo "Proceeding with minimal Ray setup..."
        
        # Start training script with minimal setup
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        cd "$SCRIPT_DIR"
        
        TRAINING_SCRIPT="examples/qwen2_5_vl_32b_geo3k_grpo.sh"
        
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

# If we reach here, the script has completed successfully
echo "=== SCRIPT COMPLETED ==="
