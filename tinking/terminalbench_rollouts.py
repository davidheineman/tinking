"""
TerminalBench rollout system for RL training.

This module handles:
1. Running minitb to generate rollouts
2. Parsing the rollout outputs
3. Sending to grader for reward calculation
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import tinker
from tinker_cookbook.rl.types import TrajectoryGroup

logger = logging.getLogger(__name__)


@dataclass
class MinitbConfig:
    """Configuration for minitb rollouts."""
    
    agent: str = "terminus"  # Agent to use (e.g., terminus)
    model_path: str = ""  # Tinker model path (e.g., tinker://...)
    dataset_path: str = ""  # Path to dataset file
    n_concurrent: int = 4  # Number of concurrent rollouts
    output_dir: str = "/tmp/tinking/rollouts"  # Base output directory
    

async def run_minitb_rollouts(
    config: MinitbConfig,
    sampling_client_path: str,
    batch_idx: int,
) -> Path:
    """
    Run minitb to generate rollouts.
    
    Args:
        config: MinitbConfig with rollout parameters
        sampling_client_path: Path to the current model weights for sampling
        batch_idx: Current batch index for naming
        
    Returns:
        Path to the output directory containing rollouts
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = Path(config.output_dir) / f"rollouts-{timestamp}-batch{batch_idx:06d}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build the minitb command
    cmd = [
        "minitb",
        "run",
        "-a", config.agent,
        "-m", sampling_client_path,
        "--dataset-path", config.dataset_path,
        "--n-concurrent", str(config.n_concurrent),
        "--output-path", str(output_path),
    ]
    
    logger.info(f"Running minitb: {' '.join(cmd)}")
    
    # Run minitb as subprocess
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"minitb failed with return code {process.returncode}")
            logger.error(f"stderr: {stderr.decode()}")
            raise RuntimeError(f"minitb run failed: {stderr.decode()}")
        
        logger.info(f"minitb completed successfully. Output at: {output_path}")
        logger.debug(f"stdout: {stdout.decode()}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error running minitb: {e}")
        raise


def parse_rollout_file(rollout_path: Path) -> dict[str, Any]:
    """
    Parse a single rollout file from minitb output.
    
    Args:
        rollout_path: Path to a rollout JSON file
        
    Returns:
        Parsed rollout data as a dictionary
    """
    try:
        with open(rollout_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error parsing rollout file {rollout_path}: {e}")
        raise


def extract_rollouts(output_dir: Path) -> list[dict[str, Any]]:
    """
    Extract all rollouts from a minitb output directory.
    
    Args:
        output_dir: Path to the minitb output directory
        
    Returns:
        List of parsed rollout dictionaries
    """
    rollouts = []
    
    # Look for JSON files in the output directory
    # Adjust this pattern based on actual minitb output structure
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
    """
    Grade rollouts using a grader model to calculate rewards.
    
    Args:
        rollouts: List of rollout dictionaries
        grader_model: Optional path/name of grader model
        
    Returns:
        List of reward scores, one per rollout
    """
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
    """
    Convert minitb rollouts and rewards into TrajectoryGroup objects for RL training.
    
    Args:
        rollouts: List of parsed rollout dictionaries
        rewards: Corresponding reward scores
        group_size: Number of rollouts per group (for advantage computation)
        
    Returns:
        List of TrajectoryGroup objects
    """
    # TODO: Implement conversion from minitb rollouts to tinker TrajectoryGroup format
    # This will depend on the structure of minitb's output
    
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
    """
    Complete pipeline: run minitb, extract rollouts, grade, and convert to TrajectoryGroups.
    
    Args:
        config: MinitbConfig with rollout parameters
        sampling_client_path: Path to the current model weights
        batch_idx: Current batch index
        group_size: Number of rollouts per trajectory group
        grader_model: Optional grader model for reward calculation
        
    Returns:
        List of TrajectoryGroup objects ready for RL training
    """
    # Step 1: Run minitb to generate rollouts
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

