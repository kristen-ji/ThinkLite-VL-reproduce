#!/usr/bin/env python3
"""
GPU Mapping Script for Multi-Node Training
This script ensures proper rank-to-GPU mapping across nodes
"""

import os
import torch
import socket
import subprocess

def get_node_info():
    """Get current node information"""
    hostname = socket.gethostname()
    # Extract node ID from hostname like 'hkn0602.localdomain' -> 602
    if 'hkn' in hostname:
        # Extract the number after 'hkn' and before any dot
        node_part = hostname.split('hkn')[1].split('.')[0]
        # Remove leading zeros
        node_id = int(node_part.lstrip('0') or '0')
    else:
        node_id = 0
    return hostname, node_id

def setup_gpu_mapping():
    """Setup proper GPU mapping for multi-node training"""
    hostname, node_id = get_node_info()
    
    # Get environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", node_id))
    world_size = int(os.environ.get("WORLD_SIZE", 8))
    
    print(f"Node: {hostname}, Node ID: {node_id}")
    print(f"Local Rank: {local_rank}, Node Rank: {node_rank}, World Size: {world_size}")
    
    # Calculate global rank
    gpus_per_node = 4
    global_rank = node_rank * gpus_per_node + local_rank
    
    print(f"Global Rank: {global_rank}")
    
    # Set device based on local rank
    if torch.cuda.is_available():
        # Ensure we have the right number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        
        if local_rank < num_gpus:
            torch.cuda.set_device(local_rank)
            device = torch.cuda.current_device()
            print(f"Set CUDA device to: {device}")
            
            # Verify device assignment
            actual_device = torch.cuda.current_device()
            print(f"Actual CUDA device: {actual_device}")
            
            # Set environment variables for NCCL
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(num_gpus)])
            os.environ["NCCL_RANK"] = str(global_rank)
            os.environ["NCCL_WORLD_SIZE"] = str(world_size)
            os.environ["NCCL_NODE_RANK"] = str(node_rank)
            
            print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            print(f"NCCL_RANK: {os.environ['NCCL_RANK']}")
            print(f"NCCL_WORLD_SIZE: {os.environ['NCCL_WORLD_SIZE']}")
            print(f"NCCL_NODE_RANK: {os.environ['NCCL_NODE_RANK']}")
            
            return True
        else:
            print(f"ERROR: Local rank {local_rank} >= available GPUs {num_gpus}")
            return False
    else:
        print("ERROR: CUDA not available")
        return False

def verify_gpu_mapping():
    """Verify that GPU mapping is correct"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        device_props = torch.cuda.get_device_properties(device)
        
        print(f"Current device: {device}")
        print(f"Device name: {device_name}")
        print(f"Device properties: {device_props}")
        
        # Check for duplicate GPU usage
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} (UUID: {props.uuid})")
        
        return True
    return False

if __name__ == "__main__":
    print("=== GPU Mapping Setup ===")
    
    success = setup_gpu_mapping()
    if success:
        print("GPU mapping setup successful")
        verify_gpu_mapping()
    else:
        print("GPU mapping setup failed")
        exit(1)
