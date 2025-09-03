import os
# Set PyTorch memory allocation configuration BEFORE any other imports
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import wandb

import ray
from omegaconf import OmegaConf

# Add this line after imports
wandb.login(key="7c8d1d26c178471c3ec13086170a3a93fc62f199")


from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
import json
import wandb  # Add this import




@ray.remote
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        
        # Check if we're in a multi-node Ray cluster
        ray_nodes = ray.nodes()
        print(f"Ray cluster nodes: {[node['NodeManagerAddress'] for node in ray_nodes]}")
        
        # If we're in a multi-node cluster, only use GPUs from current node
        if len(ray_nodes) > 1:
            print(f"Multi-node Ray cluster detected ({len(ray_nodes)} nodes). Using only current node GPUs.")
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node],  # Only current node
            }
        else:
            print(f"Single-node Ray cluster. Using all configured GPUs.")
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        # Use Ray remote reward managers for multi-node setup
        print("DEBUG: Creating Ray remote reward managers for multi-node setup")
        RemoteRewardManager = ray.remote(RewardManager)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        # Add delay to prevent simultaneous Hugging Face downloads
        import time
        print("DEBUG: Adding delay to prevent simultaneous HF downloads...")
        time.sleep(5)
        
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    # Auto-adjust trainer.nnodes / n_gpus_per_node if running under SLURM and user didn't override via CLI
    try:
        slurm_nnodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "0"))
        if slurm_nnodes > 0 and ppo_config.trainer.nnodes != slurm_nnodes:
            print(f"[AutoConfig] Overriding trainer.nnodes {ppo_config.trainer.nnodes} -> {slurm_nnodes} from SLURM_JOB_NUM_NODES")
            ppo_config.trainer.nnodes = slurm_nnodes
        # Only adjust n_gpus_per_node if user left default (8) but actual visible GPUs differ OR value is 0/None
        visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_cuda:
            per_node_visible = len(visible_cuda.split(','))
        else:
            import torch
            per_node_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if per_node_visible > 0 and ppo_config.trainer.n_gpus_per_node != per_node_visible:
            print(f"[AutoConfig] Overriding trainer.n_gpus_per_node {ppo_config.trainer.n_gpus_per_node} -> {per_node_visible} (visible GPUs on this node)")
            ppo_config.trainer.n_gpus_per_node = per_node_visible
    except Exception as e:
        print(f"[AutoConfig] Skipped auto-detect due to error: {e}")

    # Initialize Ray with minimal configuration to avoid core worker crashes
    print("DEBUG: Initializing Ray with minimal configuration...")
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                "HF_HUB_DOWNLOAD_TIMEOUT": "300",
                "HF_HUB_RETRY_DELAY": "10",
                "HF_HUB_MAX_RETRIES": "10",
            }
        }
        # Use Ray cluster setup for multi-node
        print("DEBUG: Connecting to Ray cluster...")
        import os
        
        # Set environment variable for Ray temp directory - use short path to avoid socket filename length limit
        short_temp_dir = "/tmp/ray_32b_short"
        os.makedirs(short_temp_dir, exist_ok=True)
        os.environ["RAY_TMPDIR"] = short_temp_dir
        print(f"DEBUG: Set RAY_TMPDIR to: {short_temp_dir}")
        
        if "RAY_ADDRESS" in os.environ and os.environ["RAY_ADDRESS"]:
            print(f"DEBUG: Connecting to Ray cluster at: {os.environ['RAY_ADDRESS']}")
            
            # Add timeout and retry logic for Ray init
            MAX_RAY_INIT_RETRIES = 3
            ray_init_retry = 0
            
            while ray_init_retry < MAX_RAY_INIT_RETRIES:
                try:
                    print(f"DEBUG: Ray init attempt {ray_init_retry + 1}/{MAX_RAY_INIT_RETRIES}")
                    
                    # Use signal timeout wrapper for Ray init
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Ray init timed out")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(120)  # 2 minute timeout
                    
                    try:
                        ray.init(
                            address=os.environ["RAY_ADDRESS"],
                            runtime_env=runtime_env,
                            ignore_reinit_error=True,
                            _node_ip_address=None,  # Let Ray auto-detect
                            log_to_driver=True
                        )
                        signal.alarm(0)  # Cancel timeout
                        print("DEBUG: Ray cluster connection successful!")
                        break
                        
                    except TimeoutError:
                        signal.alarm(0)
                        print(f"DEBUG: Ray init timed out on attempt {ray_init_retry + 1}")
                        if ray.is_initialized():
                            ray.shutdown()
                        
                except Exception as e:
                    signal.alarm(0)
                    print(f"DEBUG: Ray init failed on attempt {ray_init_retry + 1}: {e}")
                    if ray.is_initialized():
                        ray.shutdown()
                
                ray_init_retry += 1
                if ray_init_retry < MAX_RAY_INIT_RETRIES:
                    print("DEBUG: Waiting 30 seconds before retry...")
                    import time
                    time.sleep(30)
            
            if ray_init_retry == MAX_RAY_INIT_RETRIES:
                print("ERROR: Failed to connect to Ray cluster after all retries")
                print("DEBUG: Falling back to local Ray...")
                ray.init(
                    runtime_env=runtime_env, 
                    local_mode=False, 
                    ignore_reinit_error=True
                )
        else:
            print("DEBUG: No RAY_ADDRESS found, using local Ray...")
            ray.init(
                runtime_env=runtime_env, 
                local_mode=False, 
                ignore_reinit_error=True
            )
        print("DEBUG: Ray initialized successfully")

    # Simple diagnostics to help track GPU resource discovery
    try:
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        print(f"[Ray Diagnostics] cluster_resources={cluster_resources}")
        print(f"[Ray Diagnostics] available_resources={available_resources}")
    except Exception as e:
        print(f"Failed to fetch Ray resource diagnostics: {e}")

    # Run training with Ray remote runner
    print("DEBUG: Running training with Ray remote runner...")
    try:
        runner = Runner.remote()
        ray.get(runner.run.remote(ppo_config))
    except Exception as e:
        print(f"ERROR: Ray training failed: {e}")
        print("Attempting to run training without Ray...")
        # Fallback to direct training without Ray
        print("Starting direct training without Ray framework...")
        # Import and run the training directly
        import sys
        sys.path.append("/home/hk-project-pai00012/st_st190232/ThinkLite-VL")
        from verl.trainer.ray_trainer import RayPPOTrainer
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import FSDPWorker
        from verl.workers.reward import BatchFunctionRewardManager
        from verl.trainer.ray_trainer import ResourcePoolManager, Role
        
        # Initialize Ray locally if not already initialized
        if not ray.is_initialized():
            ray.init(local_mode=True)
        
        # Create dataloaders for fallback training
        from .data_loader import create_dataloader
        from ..utils.tokenizer import get_processor, get_tokenizer
        
        print("Creating dataloaders for fallback training...")
        tokenizer = get_tokenizer(
            ppo_config.worker.actor.model.model_path,
            override_chat_template=ppo_config.data.override_chat_template,
            trust_remote_code=ppo_config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            ppo_config.worker.actor.model.model_path,
            override_chat_template=ppo_config.data.override_chat_template,
            trust_remote_code=ppo_config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        train_dataloader, val_dataloader = create_dataloader(ppo_config.data, tokenizer, processor)
        
        # Create resource pool manager for fallback
        from verl.trainer.ray_trainer import ResourcePoolManager
        global_pool_id = "global"
        resource_pool_spec = {
            global_pool_id: [ppo_config.trainer.n_gpus_per_node] * ppo_config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, 
            mapping=mapping
        )
        
        # Create trainer directly
        trainer = RayPPOTrainer(
            config=ppo_config,
            tokenizer=ppo_config.worker.actor.model.model_path,
            processor=ppo_config.worker.actor.model.model_path,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping={Role.ActorRolloutRef: ray.remote(FSDPWorker)},
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=None,  # Will be created by trainer
            val_reward_fn=None,  # Will be created by trainer
        )
        
        # Initialize the trainer properly before calling fit
        print("Initializing trainer for fallback mode...")
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
