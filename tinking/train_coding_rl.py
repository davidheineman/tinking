"""
Simple RL training script for coding tasks.

Usage:
    python train_coding_rl.py --model-name "meta-llama/Llama-3.2-1B" --log-path ./logs
"""

import asyncio
import logging
import os
from typing import Sequence

import chz
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData

# Import from tinker-cookbook
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@chz.chz
class Config:
    """Configuration for RL training on coding tasks."""
    
    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    lora_rank: int = 32
    
    # Training hyperparameters
    groups_per_batch: int = 4
    group_size: int = 4  # number of samples per problem
    learning_rate: float = 1e-4
    max_tokens: int = 512
    num_batches: int = 100
    
    # Logging and checkpointing
    log_path: str = "/tmp/tinker-coding-rl"
    save_every: int = 10
    base_url: str | None = None
    
    # Dataset (simple examples - customize as needed)
    prompts: list[str] = chz.field(default_factory=lambda: [
        "Write a Python function that returns the sum of two numbers.",
        "Write a Python function that checks if a number is even.",
        "Write a Python function that returns the factorial of n.",
    ])


class SimpleCodingDataset(RLDataset):
    """Simple dataset that cycles through coding prompts."""
    
    def __init__(self, builders: list[EnvGroupBuilder], groups_per_batch: int):
        self.builders = builders
        self.groups_per_batch = groups_per_batch
    
    def __len__(self) -> int:
        return (len(self.builders) + self.groups_per_batch - 1) // self.groups_per_batch
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.groups_per_batch
        end = min(start + self.groups_per_batch, len(self.builders))
        return self.builders[start:end]


async def main(config: Config):
    """Main training loop."""
    # Setup logging
    os.makedirs(config.log_path, exist_ok=True)
    
    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")
    
    # Create dataset
    from coding_env import CodingGroupBuilder
    builders = [
        CodingGroupBuilder(
            renderer=renderer,
            prompts=[config.prompts[i % len(config.prompts)]],
        )
        for i in range(config.num_batches * config.groups_per_batch)
    ]
    dataset = SimpleCodingDataset(builders, config.groups_per_batch)
    
    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)
    
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = await service_client.create_training_client_from_state_async(
            resume_info["state_path"]
        )
        start_batch = resume_info.get("batch", 0)
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0
    
    # Create sampling client
    sampling_path = (
        await training_client.save_weights_for_sampler_async(name="initial")
    ).path
    sampling_client = service_client.create_sampling_client(model_path=sampling_path)
    
    # Create token completer (policy)
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
    )
    
    # Training loop
    num_batches = len(dataset)
    logger.info(f"Training for {num_batches} batches")
    
    sampling_params = tinker.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    
    for batch_idx in range(start_batch, num_batches):
        logger.info(f"Batch {batch_idx}/{num_batches}")
        
        # Get batch of environment builders
        env_group_builders = dataset.get_batch(batch_idx)
        
        # Run rollouts
        trajectory_groups: list[TrajectoryGroup] = []
        for builder in env_group_builders:
            traj_group = await do_group_rollout(builder, policy)
            trajectory_groups.append(traj_group)
        
        # Filter out groups with zero variance in rewards
        trajectory_groups_filtered = []
        for group in trajectory_groups:
            rewards = group.get_total_rewards()
            if len(set(rewards)) > 1:  # Has variance
                trajectory_groups_filtered.append(group)
        
        if not trajectory_groups_filtered:
            logger.warning("All rewards are uniform. Skipping batch.")
            continue
        
        # Compute advantages
        advantages_P = compute_advantages(trajectory_groups_filtered)
        
        # Assemble training data
        data_D, metadata_D = assemble_training_data(
            trajectory_groups_filtered, advantages_P
        )
        
        # Training step
        fwd_bwd_future = await training_client.forward_backward_async(
            data_D, loss_fn="importance_sampling"
        )
        await fwd_bwd_future.result_async()
        
        # Optimizer step
        adam_params = types.AdamParams(
            learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
        )
        await training_client.optim_step_async(adam_params)
        
        # Log metrics
        mean_reward = sum(
            sum(g.get_total_rewards()) / len(g.get_total_rewards())
            for g in trajectory_groups_filtered
        ) / len(trajectory_groups_filtered)
        logger.info(f"Batch {batch_idx}: Mean reward = {mean_reward:.3f}")
        
        # Save checkpoint
        if (batch_idx + 1) % config.save_every == 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=config.log_path,
                loop_state={"batch": batch_idx},
                kind="both",
            )
            # Update sampling client
            sampling_path = (
                await training_client.save_weights_for_sampler_async(name=f"{batch_idx:06d}")
            ).path
            sampling_client = service_client.create_sampling_client(model_path=sampling_path)
            policy.sampling_client = sampling_client
    
    # Save final checkpoint
    num_batches = len(dataset)
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        loop_state={"batch": num_batches},
        kind="both",
    )
    logger.info("Training completed")


if __name__ == "__main__":
    import sys
    config = chz.parse(Config, sys.argv[1:])
    asyncio.run(main(config))

