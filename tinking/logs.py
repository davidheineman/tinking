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

    # Print rewards as [num_problems x group_size]
    rewards_2d = np.array([group.get_total_rewards() for group in trajectory_groups])
    logger.info(f"Rewards ({rewards_2d.shape[0]} problems × {rewards_2d.shape[1]} rollouts):\n{rewards_2d}")

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
