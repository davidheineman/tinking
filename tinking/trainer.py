import asyncio
import logging
import os

import chz
import tinker
from tinker import types

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages

from tb_rollouts import MinitbConfig, do_terminalbench_rollouts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@chz.chz
class Config:    
    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    lora_rank: int = 32
    
    # Training hyperparameters
    group_size: int = 4  # number of samples per problem
    learning_rate: float = 1e-4
    num_batches: int = 100
    
    # TerminalBench / minitb configuration
    dataset_path: str = ""  # Path to dataset for minitb (required)
    agent: str = "terminus-tinker"  # Agent to use with minitb
    n_concurrent: int = 4  # Number of concurrent rollouts for minitb
    rollout_output_dir: str = "/tmp/tinking/rollouts"  # Base directory for rollout outputs
    grader_model: str | None = None  # Optional grader model (not implemented yet)
    
    # Logging and checkpointing
    log_path: str = "/tmp/tinker-coding-rl"
    save_every: int = 10
    base_url: str | None = None




async def main(config: Config):
    """Main training loop using TerminalBench."""
    # Validate configuration
    if not config.dataset_path:
        raise ValueError("dataset_path is required for TerminalBench rollouts")
    
    # Setup logging
    os.makedirs(config.log_path, exist_ok=True)
    os.makedirs(config.rollout_output_dir, exist_ok=True)
    
    logger.info(f"Using TerminalBench with dataset: {config.dataset_path}")
    
    # Create minitb configuration
    minitb_config = MinitbConfig(
        agent=config.agent,
        model_name=config.model_name,
        dataset_path=config.dataset_path,
        n_concurrent=config.n_concurrent,
        output_dir=config.rollout_output_dir,
    )
    
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
    
    # Save initial weights for sampling
    sampling_result = await training_client.save_weights_for_sampler_async(name="initial")
    sampling_path = (await sampling_result.result_async()).path
    
    logger.info(f"Initial sampling weights at: {sampling_path}")
    
    # Training loop
    logger.info(f"Training for {config.num_batches} batches")
    
    for batch_idx in range(start_batch, config.num_batches):
        logger.info(f"Batch {batch_idx}/{config.num_batches}")
        
        # Run TerminalBench rollouts
        trajectory_groups = do_terminalbench_rollouts(
            config=minitb_config,
            sampling_client_path=sampling_path,
            batch_idx=batch_idx,
            group_size=config.group_size,
            grader_model=config.grader_model,
        )
        
        # Check if we got any trajectories
        if not trajectory_groups:
            logger.warning("No trajectory groups returned. Skipping batch.")
            continue
        
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
            # Update sampling weights for next rollouts
            sampling_result = await training_client.save_weights_for_sampler_async(name=f"{batch_idx:06d}")
            sampling_path = (await sampling_result.result_async()).path
            logger.info(f"Updated sampling weights to: {sampling_path}")
    
    # Save final checkpoint
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        loop_state={"batch": config.num_batches},
        kind="both",
    )
    logger.info("Training completed")


def main_sync(config: Config):
    """Synchronous wrapper for async main function."""
    asyncio.run(main(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main_sync, allow_hyphens=True)

