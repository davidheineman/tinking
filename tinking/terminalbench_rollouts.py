import asyncio
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tinker_cookbook.rl.types import TrajectoryGroup

from constants import CURRENT_TASKS

logger = logging.getLogger(__name__)


@dataclass
class MinitbConfig:
    """Configuration for TerminalBench rollouts."""
    
    agent: str = "terminus-tinker"  # Agent to use
    model_name: str = ""  # Model name (e.g., Qwen/Qwen3-8B)
    dataset_path: str = ""  # Path to dataset directory
    n_concurrent: int = 1  # Number of concurrent rollouts
    output_dir: str = "/tmp/tinking/rollouts"  # Base output directory
    

async def run_minitb_rollouts(
    config: MinitbConfig,
    sampling_client_path: str,
    batch_idx: int,
) -> Path:
    """ launch tbench """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = Path(config.output_dir) / f"rollouts-{timestamp}-batch{batch_idx:06d}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample a random task-id from constants
    task_id = random.choice(CURRENT_TASKS)
    
    # Build the tb command
    cmd = [
        "tb",
        "run",
        "--agent", config.agent,
        "--agent-kwarg", f"checkpoint_path={sampling_client_path}",
        "--agent-kwarg", f"model_name={config.model_name}",
        "--dataset-path", config.dataset_path,
        "--task-id", task_id,
        "--n-concurrent", str(config.n_concurrent),
        "--output-path", str(output_path),
    ]
    
    logger.info(f"Running tb: {' '.join(cmd)}")
    
    # Run tb as subprocess
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"tb failed with return code {process.returncode}")
            logger.error(f"stderr: {stderr.decode()}")
            raise RuntimeError(f"tb run failed: {stderr.decode()}")
        
        logger.info(f"tb completed successfully. Output at: {output_path}")
        logger.debug(f"stdout: {stdout.decode()}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error running tb: {e}")
        raise


def parse_rollout_file(rollout_path: Path) -> dict[str, Any]:
    """ Parse a rollout file """
    try:
        with open(rollout_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error parsing rollout file {rollout_path}: {e}")
        raise


def extract_rollouts(output_dir: Path) -> list[dict[str, Any]]:
    """ Parse tbench output """
    rollouts = []
    
    # Look for JSON files in the output directory
    # Adjust this pattern based on actual tb output structure
    rollout_files = list(output_dir.glob("*.json")) + list(output_dir.glob("**/*.json"))
    
    if not rollout_files:
        logger.warning(f"No rollout files found in {output_dir}")
        return rollouts
    
    logger.info(f"Found {len(rollout_files)} rollout files")
    
    for rollout_file in rollout_files:
        try:
            rollout = parse_rollout_file(rollout_file)
            rollouts.append(rollout)
        except Exception as e:
            logger.warning(f"Skipping invalid rollout file {rollout_file}: {e}")
            continue
    
    return rollouts


async def grade_rollouts(rollouts: list[dict[str, Any]], grader_model: str | None = None) -> list[float]:
    """ Grade rollout """
    # TODO: Implement actual grading logic
    # For now, return placeholder rewards
    logger.warning("Grader model not implemented yet. Returning placeholder rewards.")
    
    rewards = []
    for i, rollout in enumerate(rollouts):
        # Placeholder: extract some metric or use a dummy reward
        # In the real implementation, this would:
        # 1. Send the rollout to a grader model
        # 2. Get a reward score back
        # 3. Return the reward
        
        # For now, return a random-ish reward based on rollout length or other heuristic
        placeholder_reward = 0.5  # Neutral reward
        rewards.append(placeholder_reward)
    
    logger.info(f"Generated {len(rewards)} placeholder rewards")
    return rewards


def rollouts_to_trajectory_groups(
    rollouts: list[dict[str, Any]], 
    rewards: list[float],
    group_size: int,
) -> list[TrajectoryGroup]:
    """ tbench rollout -> tinker rollout """
    # TODO: Implement conversion from tb rollouts to tinker TrajectoryGroup format
    # This will depend on the structure of tb's output
    
    logger.warning("rollouts_to_trajectory_groups not fully implemented yet")
    
    # Placeholder implementation
    trajectory_groups = []
    
    # Group rollouts
    for i in range(0, len(rollouts), group_size):
        group_rollouts = rollouts[i:i+group_size]
        group_rewards = rewards[i:i+group_size]
        
        # TODO: Create actual TrajectoryGroup from rollouts
        # For now, just log that we would create a group
        logger.info(f"Would create TrajectoryGroup with {len(group_rollouts)} trajectories")
    
    return trajectory_groups


async def do_terminalbench_rollouts(
    config: MinitbConfig,
    sampling_client_path: str,
    batch_idx: int,
    group_size: int,
    grader_model: str | None = None,
) -> list[TrajectoryGroup]:
    """ run tb -> extract rollouts -> grade -> convert to TrajectoryGroups """
    # Step 1: Run tb to generate rollouts
    output_dir = await run_minitb_rollouts(config, sampling_client_path, batch_idx)
    
    # Step 2: Extract rollouts from output directory
    rollouts = extract_rollouts(output_dir)
    
    if not rollouts:
        logger.warning("No rollouts extracted. Returning empty list.")
        return []
    
    # Step 3: Grade rollouts to get rewards
    rewards = await grade_rollouts(rollouts, grader_model)
    
    # Step 4: Convert to TrajectoryGroups
    trajectory_groups = rollouts_to_trajectory_groups(rollouts, rewards, group_size)
    
    return trajectory_groups

