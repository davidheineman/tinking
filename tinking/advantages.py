"""
TTT training utilities: entropic advantages + ε-greedy reuse sampler.

Provides:
- Entropic (leave-one-out softmax) advantages as an alternative to the
  default advantage estimator from tinker_cookbook.
- An ε-greedy state buffer that maintains top-k solutions and samples
  parents for the next batch with exploration/exploitation balance.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tinker_cookbook.rl.types import TrajectoryGroup


# =============================================================================
# Solution State + ε-Greedy Reuse Sampler
# =============================================================================

@dataclass
class SolutionState:
    """A solution state stored in the ε-greedy buffer."""
    id: str
    value: float  # higher is better (for Erdős: 1/(ε + C₅))
    data: dict[str, Any] = field(default_factory=dict)


class EpsilonGreedySampler:
    """
    Maintains a buffer of top-k states, sorted by value (descending).

    Sampling: with probability ε pick a random state from the buffer
    (exploration), otherwise pick the best (exploitation).
    """

    def __init__(self, buffer_size: int = 16, epsilon: float = 0.125):
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self._states: list[SolutionState] = []

    def sample(self) -> SolutionState:
        """Sample a state using ε-greedy strategy."""
        if not self._states:
            raise ValueError("Buffer empty!")
        if np.random.random() < self.epsilon and len(self._states) > 1:
            return np.random.choice(self._states)
        return self._states[0]

    def update(self, new_states: list[SolutionState]):
        """Add new states and keep top-k by value."""
        self._states.extend(new_states)
        self._states.sort(key=lambda s: s.value, reverse=True)
        self._states = self._states[:self.buffer_size]

    def get_best(self) -> SolutionState:
        """Return the highest-valued state in the buffer."""
        if not self._states:
            raise ValueError("Buffer empty!")
        return self._states[0]

    def __len__(self) -> int:
        return len(self._states)

    def __repr__(self) -> str:
        if not self._states:
            return "EpsilonGreedySampler(empty)"
        best = self._states[0].value
        worst = self._states[-1].value
        return (
            f"EpsilonGreedySampler(size={len(self._states)}/{self.buffer_size}, "
            f"best={best:.6f}, worst={worst:.6f})"
        )


# =============================================================================
# Training History
# =============================================================================

@dataclass
class TrainingHistory:
    """Tracks training metrics across steps for TTT runs."""
    raw_scores_best: list[float] = field(default_factory=list)
    raw_scores_mean: list[float] = field(default_factory=list)
    advantages: list[np.ndarray] = field(default_factory=list)

    def record(self, best: float, mean: float, adv: np.ndarray):
        """Record one step's metrics."""
        self.raw_scores_best.append(best)
        self.raw_scores_mean.append(mean)
        self.advantages.append(adv)

    def improvement_summary(self) -> str:
        """Return a string summarising initial → final improvement."""
        if len(self.raw_scores_best) < 2:
            return "Not enough data for summary"
        initial = self.raw_scores_best[0]
        final = self.raw_scores_best[-1]
        pct = 100 * (initial - final) / (abs(initial) + 1e-12)
        return f"{initial:.6f} → {final:.6f} ({pct:.2f}% reduction over {len(self.raw_scores_best)} steps)"


# =============================================================================
# Entropic Advantages
# =============================================================================

def compute_entropic_advantages(rewards: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute advantages using entropic (leave-one-out softmax) weighting.

    w_i = exp(β * r_i) / Z_i  where Z_i = (1/(k-1)) * Σ_{j≠i} exp(β * r_j)
    advantage_i = w_i - 1

    Args:
        rewards: Shape (k,) rewards for k rollouts in a group.
        beta: Temperature parameter (higher = more peaked toward best).

    Returns:
        advantages: Shape (k,) advantages, sum approximately 0.
    """
    r_safe = rewards - rewards.max()
    e = torch.exp(beta * r_safe)
    k = len(rewards)

    if k == 1:
        return torch.zeros(1)

    # Leave-one-out normalization: Z_i excludes sample i
    Z = (e.sum() - e) / (k - 1)
    w = e / (Z + 1e-12)
    return w - 1.0


def compute_entropic_group_advantages(
    trajectory_groups: list[TrajectoryGroup],
    beta: float = 2.0,
) -> list[np.ndarray]:
    """
    Compute entropic advantages for each trajectory group.

    Drop-in replacement for tinker_cookbook's compute_advantages() when using
    the entropic advantage estimator.

    Args:
        trajectory_groups: List of trajectory groups with rewards.
        beta: Temperature parameter for entropic weighting.

    Returns:
        List of advantage arrays, one per group.
    """
    all_advantages = []
    for group in trajectory_groups:
        rewards = torch.tensor(group.get_total_rewards(), dtype=torch.float32)
        advantages = compute_entropic_advantages(rewards, beta=beta)
        all_advantages.append(advantages.numpy())
    return all_advantages
