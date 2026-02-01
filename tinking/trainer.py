import asyncio
import logging
import os
import random
import string
import time
from datetime import datetime
from typing import Optional

import chz
import wandb
import tinker
from pprint import pformat
from rich.console import Console
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer

from tinking.envs.base import Environment, EnvironmentConfig
from tinking.envs.terminal import TerminalBenchEnvironment
from tinking.envs.minerva import MinervaEnvironment
from tinking.logs import log_batch_info

console = Console()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@chz.chz
class WandbConfig:
    enabled: bool = True
    project: str = "tinker"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: list[str] = None


@chz.chz
class OptimConfig:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@chz.chz
class Config:    
    # model config
    model_name: str = "meta-llama/Llama-3.2-1B"
    lora_rank: int = 32
    
    # hparams
    group_size: int = 4
    num_batches: int = 100
    
    # optimizer config
    optim: OptimConfig = chz.field(default_factory=OptimConfig)
    
    # environment config
    env: EnvironmentConfig = chz.field(default_factory=EnvironmentConfig)
    
    # wandb config
    wandb: WandbConfig = chz.field(default_factory=WandbConfig)
    
    # administravista
    log_path: str = "logs" # "~/.cache/tinking/logs"
    save_every: int = 10
    base_url: str | None = None




async def setup_config(config: Config):
    # Override model_name in env config to match training model
    config.env.model_name = config.model_name
    config.env.n_concurrent = config.group_size
    
    # Get logging path
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(config.log_path, f"run-{run_timestamp}")
    rollouts_dir = os.path.join(config.log_path, f"run-{run_timestamp}", "rollouts")

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(rollouts_dir, exist_ok=True)

    console.print(pformat(config, indent=2, width=100))

    # Initialize wandb if enabled
    if config.wandb.enabled:
        tags = config.wandb.tags or []
        tags.extend([
            f"model:{config.model_name}",
            f"lora_rank:{config.lora_rank}",
        ])

        model_name = config.model_name.split("/")[-1]

        # add random 4 char string suffix
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        run_name = f"{config.wandb.run_name or 'run'}-{model_name}-{random_suffix}"
        
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=run_name,
            tags=tags,
            config=config
        )

    return config, log_path, rollouts_dir


async def main(config: Config):
    """main trainer loop"""

    config, log_path, rollouts_dir = await setup_config(config)
    
    # Create tokenizer and renderer for rollout conversion
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")
    
    # Init environment
    match config.env.name:
        case "minerva":
            env: Environment = MinervaEnvironment(
                config=config.env,
                group_size=config.group_size,
                renderer=renderer,
                output_dir=rollouts_dir,
            )
        case "terminal":
            env: Environment = TerminalBenchEnvironment(
                config=config.env,
                group_size=config.group_size,
                renderer=renderer,
                output_dir=rollouts_dir,
            )
        case _:
            raise ValueError(f"Unknown environment name: {config.env.name!r}")
    
    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)
    
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
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
        step_start_time = time.time()
        
        # Run environment rollouts
        trajectory_groups = await env.do_rollouts(
            sampling_client_path=sampling_path,
            batch_idx=batch_idx,
        )
        
        # Check if we got any trajectories
        if not trajectory_groups:
            logger.warning("No trajectory groups returned. Skipping batch.")
            continue
        
        # Log outputs
        log_batch_info(trajectory_groups, tokenizer, num_examples=1)
        
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
        
        # Remove mask from loss_fn_inputs (importance_sampling doesn't use it)
        def remove_mask(datum: tinker.Datum) -> tinker.Datum:
            return tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
            )
        
        data_D_no_mask = list(map(remove_mask, data_D))
        
        # Training step
        fwd_bwd_future = await training_client.forward_backward_async(
            data_D_no_mask, loss_fn="importance_sampling"
        )
        await fwd_bwd_future.result_async()
        
        # Optimizer step
        adam_params = types.AdamParams(
            learning_rate=config.optim.lr,
            beta1=config.optim.beta1,
            beta2=config.optim.beta2,
            eps=config.optim.eps,
        )
        await training_client.optim_step_async(adam_params)
        
        # Update sampling weights for next rollouts (do this after every step)
        sampling_result = await training_client.save_weights_for_sampler_async(name=f"batch{batch_idx:06d}")
        sampling_path = (await sampling_result.result_async()).path
        logger.info(f"Updated sampling weights to: {sampling_path}")
        
        # Calculate metrics
        step_time = time.time() - step_start_time
        
        # Calculate reward statistics
        all_rewards = []
        for group in trajectory_groups_filtered:
            all_rewards.extend(group.get_total_rewards())
        
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        reward_variance = (
            sum((r - mean_reward) ** 2 for r in all_rewards) / len(all_rewards)
            if len(all_rewards) > 0 else 0.0
        )
        
        # Calculate token statistics
        total_tokens = sum(
            datum.model_input.length
            for datum in data_D
        )
        num_rollouts = sum(len(group.trajectories_G) for group in trajectory_groups_filtered)
        avg_tokens_per_rollout = total_tokens / num_rollouts if num_rollouts > 0 else 0.0
        
        # Log to console
        logger.info(
            f"Batch {batch_idx}: reward_mean={mean_reward:.3f}, "
            f"reward_var={reward_variance:.3f}, "
            f"total_tokens={total_tokens}, "
            f"avg_tokens={avg_tokens_per_rollout:.1f}, "
            f"step_time={step_time:.1f}s"
        )
        
        # Log to wandb
        if config.wandb.enabled:
            wandb.log({
                "batch": batch_idx,
                "reward/mean": mean_reward,
                "reward/variance": reward_variance,
                "tokens/total": total_tokens,
                "tokens/avg_per_rollout": avg_tokens_per_rollout,
                "train/lr": config.optim.lr,
                "time/step_seconds": step_time,
            }, step=batch_idx)
        
        # Save checkpoint
        if (batch_idx + 1) % config.save_every == 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=log_path,
                loop_state={"batch": batch_idx},
                kind="both",
            )
    
    # Save final checkpoint
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=log_path,
        loop_state={"batch": config.num_batches},
        kind="both",
    )
    
    if config.wandb.enabled:
        wandb.finish()
    
    logger.info("Training completed")


def main_sync(config: Config):
    """ a sync wrapper for our async runner """
    asyncio.run(main(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main_sync, allow_hyphens=True)
