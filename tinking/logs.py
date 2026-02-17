import logging
import numpy as np
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer
from rich.console import Console

console = Console()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def decode_truncated(tokens, tokenizer, num_tokens):
    if len(tokens) > num_tokens * 2:
        first_part = tokenizer.decode(tokens[:num_tokens])
        last_part = tokenizer.decode(tokens[-num_tokens:])
        return f"{first_part}\n\n... [{len(tokens)} tokens total] ...\n\n{last_part}"
    else:
        return tokenizer.decode(tokens)


def log_batch_info(
    trajectory_groups: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    num_examples: int = 1,
    num_tokens: int = 50,
) -> None:
    """Log debug info about trajectory groups."""
    if not trajectory_groups:
        return

    # Print (a subset of) rewards as [num_problems x group_size]
    rewards_2d = np.array([group.get_total_rewards() for group in trajectory_groups])
    n_rows = min(5, rewards_2d.shape[0])
    n_cols = min(5, rewards_2d.shape[1])
    rewards_subset = rewards_2d[:n_rows, :n_cols]
    logger.info(
        f"Rewards ({rewards_2d.shape[0]} problems × {rewards_2d.shape[1]} rollouts), "
        f"showing [{n_rows} rows x {n_cols} cols]:\n{rewards_subset}"
    )

    # Log example trajectories
    for i, group in enumerate(trajectory_groups[:num_examples]):
        if not group.trajectories_G:
            continue

        traj = group.trajectories_G[0]
        reward = group.get_total_rewards()[0]

        if not traj.transitions:
            continue

        trans = traj.transitions[0]

        # Decode prompt and response
        prompt_tokens = trans.ob.to_ints()
        response_tokens = trans.ac.tokens

        prompt_text = decode_truncated(prompt_tokens, tokenizer, num_tokens)
        response_text = decode_truncated(response_tokens, tokenizer, num_tokens)

        console.print(f"\n[bold]Example {i+1}:[/bold]")
        console.print(f"[green]{prompt_text}[/green]", end="")
        console.print(f"[blue]{response_text}[/blue]")
        console.print(
            f"[yellow]Reward:[/yellow] {reward:.3f}, [yellow]Metrics:[/yellow] {trans.metrics}"
        )


def format_rollout_for_wandb(
    trajectory_groups: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    group_idx: int = 0,
    rollout_idx: int = 0,
    num_tokens: int = 200,
) -> str | None:
    """Format a single rollout as a string for wandb logging. Returns None if no valid rollout."""
    if not trajectory_groups or group_idx >= len(trajectory_groups):
        return None
    group = trajectory_groups[group_idx]
    if not group.trajectories_G or rollout_idx >= len(group.trajectories_G):
        return None
    traj = group.trajectories_G[rollout_idx]
    rewards = group.get_total_rewards()
    reward = rewards[rollout_idx] if rollout_idx < len(rewards) else 0.0
    if not traj.transitions:
        return None
    trans = traj.transitions[0]
    prompt_tokens = trans.ob.to_ints()
    response_tokens = trans.ac.tokens
    prompt_text = decode_truncated(prompt_tokens, tokenizer, num_tokens)
    response_text = decode_truncated(response_tokens, tokenizer, num_tokens)
    lines = [
        "--- Prompt ---",
        prompt_text,
        "--- Response ---",
        response_text,
        f"--- Reward: {reward:.4f} | Metrics: {trans.metrics} ---",
    ]
    return "\n".join(lines)


def compute_bigram_diversity(
    trajectory_groups: list[TrajectoryGroup],
) -> dict[str, int | float]:
    """
    Compute n-gram diversity (unique bigrams) across all response tokens in the batch.
    Returns dict with unique_bigrams, total_bigrams, and diversity_ratio (unique/total).
    """
    all_bigrams: set[tuple[int, int]] = set()
    total_bigrams = 0
    for group in trajectory_groups:
        for traj in group.trajectories_G:
            for trans in traj.transitions:
                tokens = trans.ac.tokens
                for i in range(len(tokens) - 1):
                    bigram = (int(tokens[i]), int(tokens[i + 1]))
                    all_bigrams.add(bigram)
                    total_bigrams += 1
    ratio = len(all_bigrams) / total_bigrams if total_bigrams else 0.0
    return {
        "unique_bigrams": len(all_bigrams),
        "total_bigrams": total_bigrams,
        "diversity_ratio": ratio,
    }
