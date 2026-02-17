"""
Erdős Minimum Overlap Problem environment for TTT training.

Find h: [0,2] → [0,1] minimizing C₅ = max_k ∫ h(x)(1-h(x+k)) dx
Subject to: ∫ h(x) dx = 1, h(x) ∈ [0,1]

The LLM generates Python code defining a run() function that returns
(h_values, c5_bound, n_points). Code is executed in a subprocess and
verified. The reward is 1/(ε + C₅) so that higher reward corresponds
to a better (lower) C₅.

Matches the discover repo's tasks/erdos_min_overlap task.
"""
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tinker
from tinker import types as tinker_types
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.types import TrajectoryGroup, Trajectory, Transition

from tinking.advantages import EpsilonGreedySampler, SolutionState, TrainingHistory
from tinking.envs.base import Environment, EnvironmentConfig

console = Console()
logger = logging.getLogger(__name__)


ERDOS_PROMPT = """\
You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

C₅ = max_k ∫ h(x)(1 - h(x+k)) dx

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as {n_points} samples over [0, 2].
With dx = 2.0 / {n_points}:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == {n_points} / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: {eval_timeout}s for your code to run

## Rules
- Define `run()` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- Your function must complete within the time budget and return the best solution found

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.
{parent_info}"""

# When teacher forcing is on and the model hits the thinking cap, we append this
# and continue generating until the context window is full.
TEACHER_FORCING_PHRASE = (
    " ... okay, I am out of thinking tokens. I need to send my final message now."
)

# Verifier source code (matching discover/tasks/erdos_min_overlap/verifier.py),
# injected into subprocess scripts for code execution.
VERIFIER_SOURCE = '''
def verify_c5_solution(h_values, c5_achieved, n_points):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")
    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")
    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")
    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")
    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")
    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)
    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError("After normalization, h(x) is not in [0, 1].")
    dx = 2.0 / n_points
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)
    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")
    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    return computed_c5

def evaluate_erdos_solution(h_values, c5_bound, n_points):
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)
'''


def extract_code(response: str) -> str | None:
    """Extract Python code from ```python ... ``` blocks in LLM output.

    Matches discover's BaseRewardTask._extract_code.
    """
    m = re.search(r"```python\s+([\s\S]*?)\s*```", response)
    return m.group(1).strip() if m is not None else None


async def execute_erdos_code(
    code: str,
    n_points: int,
    initial_h: np.ndarray | None,
    eval_timeout: int,
) -> dict | None:
    """Execute LLM-generated code in a subprocess with timeout.

    Matches discover's code execution pipeline: injects verifier code and
    initial_h_values, then calls the run() function defined in the LLM's code.

    Returns dict with {"h_values": list, "c5": float, "n_points": int}
    or None on failure.
    """
    initial_h_code = ""
    if initial_h is not None:
        initial_h_code = f"initial_h_values = np.array({initial_h.tolist()!r})"

    script = (
        "import sys, json, traceback\n"
        "import numpy as np\n"
        "try:\n"
        "    import scipy\n"
        "except ImportError:\n"
        "    pass\n\n"
        + VERIFIER_SOURCE + "\n"
        + initial_h_code + "\n\n"
        + code + "\n\n"
        "try:\n"
        "    result = run()\n"
        "    h_values, c5_bound, _n_points = result\n"
        "    h_values = np.array(h_values, dtype=np.float64)\n"
        "    c5_verified = evaluate_erdos_solution(h_values, c5_bound, _n_points)\n"
        '    print(json.dumps({"h_values": h_values.tolist(), '
        '"c5": float(c5_verified), "n_points": int(_n_points)}))\n'
        "except Exception as e:\n"
        '    print(json.dumps({"error": str(e)}))\n'
    )

    fd, script_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)

        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=eval_timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except (asyncio.CancelledError, ProcessLookupError, OSError):
                pass
            return None

        if proc.returncode != 0:
            return None

        output = stdout.decode().strip()
        if not output:
            return None

        result = json.loads(output)
        if "error" in result:
            logger.debug(f"Code execution error: {result['error']}")
            return None

        return result
    except (Exception, asyncio.CancelledError) as e:
        logger.debug(f"Code execution failed: {e}")
        return None
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ErdosConfig(EnvironmentConfig):
    """Config for the Erdős Minimum Overlap Problem environment."""
    name: Literal["erdos"] = "erdos"
    n_points: int = 50
    mutation_strength: float = 0.1
    model_name: str = ""
    n_concurrent: int = 1
    max_tokens: int = 8192
    # ε-greedy reuse sampler
    buffer_size: int = 16
    epsilon: float = 0.125
    # Teacher forcing: when the LM hits max_tokens without finishing, append
    # TEACHER_FORCING_PHRASE and continue generating until context_window_tokens.
    teacher_forcing: bool = False
    context_window_tokens: int = 32_768
    # Code execution timeout (seconds) for LLM-generated run() functions
    eval_timeout: int = 60


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ErdosEnvironment(Environment):
    """
    TTT environment for the Erdős Minimum Overlap Problem.

    Each rollout batch:
    1. ε-greedy samples a parent solution from the state buffer
    2. Prompts the LLM to improve on that parent
    3. Evaluates candidate h-functions by computing C₅
    4. Adds improved children back to the buffer

    Rewards are ``1/(ε + C₅)`` (higher is better since we minimize C₅).
    """

    def __init__(
        self,
        config: ErdosConfig,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        output_dir: str,
    ):
        self.config = config
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.output_dir = output_dir
        self.service_client = tinker.ServiceClient()

        # ε-greedy reuse sampler: maintains top-k solutions across batches
        self.sampler = EpsilonGreedySampler(
            buffer_size=config.buffer_size,
            epsilon=config.epsilon,
        )

        # Training history for tracking progress
        self.history = TrainingHistory()

        # Seed the buffer with the constant h=0.5 baseline
        h_init = np.ones(config.n_points) * 0.5
        c5_init = compute_c5(h_init)
        self.sampler.update([SolutionState(
            id="init",
            value=1.0 / (1e-8 + c5_init),
            data={"h_values": h_init, "c5": c5_init},
        )])
        self._initial_c5 = c5_init
        logger.info(f"Seeded buffer with h=0.5 baseline (C₅={c5_init:.6f})")

    async def _continue_with_teacher_forcing(
        self,
        prompt,  # ModelInput (from renderer.build_generation_prompt)
        seq,  # sequence with .tokens, .logprobs
        tokenizer,
        sampling_client,
    ) -> tuple[list, list]:
        """Append TEACHER_FORCING_PHRASE and continue generating to context window. Returns (tokens, logprobs)."""
        prompt_tokens = list(prompt.to_ints())
        forcing_enc = tokenizer.encode(TEACHER_FORCING_PHRASE, add_special_tokens=False)
        forcing_tokens = list(forcing_enc) if hasattr(forcing_enc, "__len__") else list(forcing_enc)
        continuation_tokens = prompt_tokens + list(seq.tokens) + forcing_tokens
        cont_prompt_len = len(continuation_tokens)
        remaining = self.config.context_window_tokens - cont_prompt_len
        if remaining <= 0:
            return list(seq.tokens), list(seq.logprobs) if seq.logprobs else [0.0] * len(seq.tokens)

        continuation_model_input = tinker_types.ModelInput(
            chunks=[tinker_types.EncodedTextChunk(tokens=continuation_tokens, type="encoded_text")]
        )
        cont_result = await sampling_client.sample_async(
            prompt=continuation_model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=1.0,
                max_tokens=remaining,
                stop=self.renderer.get_stop_sequences(),
            ),
        )
        cont_seq = cont_result.sequences[0]
        full_tokens = list(seq.tokens) + forcing_tokens + list(cont_seq.tokens)
        orig_lp = list(seq.logprobs) if seq.logprobs else [0.0] * len(seq.tokens)
        cont_lp = list(cont_seq.logprobs) if cont_seq.logprobs else [0.0] * len(cont_seq.tokens)
        full_logprobs = orig_lp + [0.0] * len(forcing_tokens) + cont_lp
        return full_tokens, full_logprobs

    # -- public API ----------------------------------------------------------

    async def do_rollouts(
        self,
        sampling_client_path: str,
        batch_idx: int,
    ) -> list[TrajectoryGroup]:
        """Generate candidate h-functions, evaluate C₅, return trajectory groups.

        Produces num_groups groups of group_size rollouts each (same context per
        batch), with num_groups = batch_size // group_size. Paper setup: 512
        rollouts = 8 groups × 64 with same context from reuse buffer.
        """
        sampling_client = self.service_client.create_sampling_client(
            model_path=sampling_client_path
        )
        num_groups = self.batch_size // self.group_size
        if num_groups < 1:
            num_groups = 1

        # 1. ε-greedy sample a parent from the buffer (same context for all groups)
        parent = self.sampler.sample()
        parent_c5 = parent.data["c5"]
        parent_h = parent.data["h_values"]
        logger.info(
            f"Batch {batch_idx}: sampled parent '{parent.id}' "
            f"(C₅={parent_c5:.6f}, ε-greedy from {len(self.sampler)} states), "
            f"{num_groups} groups × {self.group_size} rollouts"
        )

        # 2. Build prompt conditioned on the sampled parent
        messages = self._build_messages(parent)
        prompt = self.renderer.build_generation_prompt(messages)
        tokenizer = self.renderer.tokenizer

        all_groups: list[TrajectoryGroup] = []
        all_improved_states: list[SolutionState] = []
        all_c5_values: list[float] = []
        total_valid = 0
        total_failed = 0
        total_no_code = 0

        # 3. Sample num_groups × group_size rollouts in parallel (each group same context)
        async def sample_one_group(g: int):
            sample_result = await sampling_client.sample_async(
                prompt=prompt,
                num_samples=self.group_size,
                sampling_params=tinker.SamplingParams(
                    temperature=1.0,
                    max_tokens=self.config.max_tokens,
                    stop=self.renderer.get_stop_sequences(),
                ),
            )
            # 3b. Teacher forcing: if a sequence hit max_tokens without finishing,
            # append forcing phrase and continue to context_window_tokens.
            merged: dict[int, tuple[list, list]] = {}
            if self.config.teacher_forcing and self.config.context_window_tokens > 0:
                truncated = [
                    (i, seq)
                    for i, seq in enumerate(sample_result.sequences)
                    if len(seq.tokens) >= self.config.max_tokens
                ]
                if truncated:

                    async def continue_one(idx: int, s) -> tuple[int, list, list]:
                        t, lp = await self._continue_with_teacher_forcing(
                            prompt, s, tokenizer, sampling_client
                        )
                        return (idx, t, lp)

                    continuations = await asyncio.gather(
                        *[continue_one(idx, s) for idx, s in truncated]
                    )
                    for idx, t, lp in continuations:
                        merged[idx] = (t, lp)

            # 4. Evaluate candidates: extract code, execute in subprocesses, verify
            code_items: list[tuple[list, list, str | None]] = []
            for i, seq in enumerate(sample_result.sequences):
                if i in merged:
                    tokens, logprobs = merged[i]
                else:
                    tokens = seq.tokens
                    logprobs = seq.logprobs
                text = tokenizer.decode(tokens)
                code = extract_code(text)
                code_items.append((tokens, logprobs, code))

            # Execute all extracted code in parallel subprocesses
            async def _null():
                return None

            exec_results = await asyncio.gather(*[
                execute_erdos_code(
                    code=code,
                    n_points=self.config.n_points,
                    initial_h=parent_h,
                    eval_timeout=self.config.eval_timeout,
                ) if code is not None else _null()
                for _, _, code in code_items
            ])

            trajectories = []
            rewards = []
            group_metrics = []
            group_improved: list[SolutionState] = []
            group_c5: list[float] = []
            g_valid = 0
            g_failed = 0
            g_no_code = 0

            for i, (tokens, logprobs, code) in enumerate(code_items):
                result = exec_results[i]
                if result is not None:
                    c5 = result["c5"]
                    h = np.array(result["h_values"])
                    reward = 1.0 / (1e-8 + c5)
                    is_valid = True
                else:
                    c5 = None
                    h = None
                    reward = 0.0
                    is_valid = False
                    if code is None:
                        g_no_code += 1

                metrics = {"valid": is_valid}
                if c5 is not None:
                    metrics["c5"] = c5

                if is_valid and reward > parent.value:
                    group_improved.append(SolutionState(
                        id=f"b{batch_idx}_g{g}_c{i}",
                        value=reward,
                        data={"h_values": h.copy(), "c5": c5},
                    ))

                ac = TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs)
                transition = Transition(
                    ob=prompt,
                    ac=ac,
                    reward=reward,
                    episode_done=True,
                    metrics=metrics,
                )
                trajectory = Trajectory(
                    transitions=[transition],
                    final_ob=prompt,
                )
                trajectories.append(trajectory)
                rewards.append(reward)
                group_metrics.append(metrics)
                if c5 is not None:
                    group_c5.append(c5)
                if is_valid:
                    g_valid += 1
                else:
                    g_failed += 1

            return (
                TrajectoryGroup(
                    trajectories_G=trajectories,
                    final_rewards_G=rewards,
                    metrics_G=group_metrics,
                ),
                group_improved,
                group_c5,
                g_valid,
                g_failed,
                g_no_code,
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Sampling Erdős candidates (batch {batch_idx})...",
                total=num_groups * self.group_size,
            )
            results = await asyncio.gather(*[sample_one_group(g) for g in range(num_groups)])
            progress.advance(task, advance=num_groups * self.group_size)

        for traj_grp, improved, c5_vals, g_valid_n, g_failed_n, g_no_code_n in results:
            all_groups.append(traj_grp)
            all_improved_states.extend(improved)
            all_c5_values.extend(c5_vals)
            total_valid += g_valid_n
            total_failed += g_failed_n
            total_no_code += g_no_code_n

        # 5. Update the buffer with all improved children from this batch
        if all_improved_states:
            self.sampler.update(all_improved_states)

        # Record history (one entry per batch, using all rewards for adv shape)
        best = self.sampler.get_best()
        if all_c5_values:
            all_rewards = []
            for grp in all_groups:
                all_rewards.extend(grp.get_total_rewards())
            self.history.record(
                best=best.data["c5"],
                mean=float(np.mean(all_c5_values)),
                adv=np.array(all_rewards),
            )

        total_rollouts = num_groups * self.group_size
        summary = (
            f"Batch {batch_idx}: {total_valid}/{total_rollouts} valid"
        )
        if total_failed:
            exec_fail = total_failed - total_no_code
            parts = []
            if total_no_code:
                parts.append(f"{total_no_code} no code")
            if exec_fail:
                parts.append(f"{exec_fail} exec failures")
            summary += f", {total_failed} failed ({', '.join(parts)})"
        summary += f", {len(all_improved_states)} improved"
        if all_c5_values:
            summary += f", C₅ range [{min(all_c5_values):.6f}, {max(all_c5_values):.6f}]"
        summary += f", buffer best C₅={best.data['c5']:.6f} ({self.sampler})"
        logger.info(summary)

        return all_groups

    @property
    def best_c5(self) -> float | None:
        """Best C₅ found so far across all batches."""
        if len(self.sampler) == 0:
            return None
        return self.sampler.get_best().data["c5"]

    @property
    def best_h(self) -> np.ndarray | None:
        """h-function corresponding to best C₅."""
        if len(self.sampler) == 0:
            return None
        return self.sampler.get_best().data["h_values"]

    def print_summary(self):
        """Print final results summary (matches discover trainer's _print_summary)."""
        best = self.sampler.get_best()
        best_c5 = best.data["c5"]

        console.print("\n" + "=" * 60)
        console.print("[bold]FINAL RESULTS[/bold]")
        console.print("=" * 60)
        console.print(f"Best C₅={best_c5:.6f}")
        console.print(f"Best state: {best.id}")
        console.print(f"Buffer: {self.sampler}")

        pct = 100 * (self._initial_c5 - best_c5) / (abs(self._initial_c5) + 1e-12)
        console.print(
            f"Improvement: {self._initial_c5:.6f} → {best_c5:.6f} ({pct:.2f}% reduction)"
        )

        if self.history.raw_scores_best:
            console.print(f"History: {self.history.improvement_summary()}")

    # -- internals -----------------------------------------------------------

    def _build_messages(self, parent: SolutionState) -> list[dict]:
        parent_c5 = parent.data["c5"]

        parent_info = (
            f"\nCurrent solution achieves C₅={parent_c5:.6f}.\n"
            f"`initial_h_values` is pre-imported as a numpy array with "
            f"{self.config.n_points} values representing this solution.\n"
            f"Try to find a function with lower C₅.\n"
        )

        prompt_text = ERDOS_PROMPT.format(
            n_points=self.config.n_points,
            eval_timeout=self.config.eval_timeout,
            parent_info=parent_info,
        )
        return [{"role": "user", "content": prompt_text}]


# ---------------------------------------------------------------------------
# Erdős problem utilities (standalone, usable outside the environment)
# ---------------------------------------------------------------------------

def parse_h_values(text: str, n_points: int) -> np.ndarray | None:
    """Extract h values (Python list of floats) from LLM output."""
    try:
        match = re.search(r'\[[\d\s.,\-e+]+\]', text, re.DOTALL)
        if not match:
            return None
        values = json.loads(match.group())
        if len(values) != n_points:
            return None
        h = np.array(values, dtype=np.float64)
        if not np.all(np.isfinite(h)):
            return None
        return np.clip(h, 0.0, 1.0)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def compute_c5(h_values: np.ndarray) -> float:
    """Compute C₅ bound for a discretized h function."""
    n = len(h_values)
    dx = 2.0 / n
    h = h_values.copy()

    # Normalize to satisfy integral constraint: sum(h) * dx = 1  =>  sum(h) = n/2
    target_sum = n / 2.0
    current_sum = np.sum(h)
    if current_sum > 0 and current_sum != target_sum:
        h = h * (target_sum / current_sum)

    j = 1.0 - h
    correlation = np.correlate(h, j, mode="full") * dx
    return float(np.max(correlation))


def compute_c5_batch(H: np.ndarray) -> np.ndarray:
    """Compute C₅ for many h functions at once. H shape (N, n_points). Returns (N,) C₅ values."""
    N, n = H.shape
    dx = 2.0 / n
    target_sum = n / 2.0
    h = H.astype(np.float64, copy=True)
    current_sums = h.sum(axis=1, keepdims=True)
    scale = np.where(
        (current_sums > 0) & (current_sums != target_sum),
        target_sum / current_sums,
        1.0,
    )
    h = h * scale
    j = 1.0 - h
    n_pad = 2 * n - 1
    h_pad = np.pad(h, ((0, 0), (0, n_pad - n)), mode="constant", constant_values=0)
    j_pad = np.pad(j, ((0, 0), (0, n_pad - n)), mode="constant", constant_values=0)
    h_fft = np.fft.rfft(h_pad, n=n_pad, axis=1)
    j_fft = np.fft.rfft(j_pad, n=n_pad, axis=1)
    corr = np.fft.irfft(h_fft * np.conj(j_fft), n=n_pad, axis=1).real
    return (corr.max(axis=1) * dx).astype(np.float64)


def verify_erdos_solution(h_values: np.ndarray, c5: float) -> bool:
    """Verify that the claimed C₅ matches re-computation."""
    recomputed = compute_c5(h_values)
    return abs(recomputed - c5) < 1e-6


# ---------------------------------------------------------------------------
# Mutation strategies (from discover/problems/erdos.py)
# ---------------------------------------------------------------------------

def normalize_h(h: np.ndarray) -> np.ndarray:
    """Clip h to [0, 1]."""
    h = np.clip(h, 0.0, 1.0)
    h = np.minimum(h, 1.0)
    h = np.maximum(h, 0.0)
    return h


def create_random_h(n_points: int) -> np.ndarray:
    """Create a random h function satisfying constraints."""
    h = 0.2 + 0.6 * np.random.rand(n_points)
    target_sum = n_points / 2.0
    h = h * (target_sum / h.sum())
    return np.clip(h, 0.0, 1.0)


def mutate_h(h_values: np.ndarray, mutation_strength: float = 0.1) -> np.ndarray:
    """Apply random mutations, ensuring valid result after normalization."""
    n = len(h_values)
    h = h_values + np.random.randn(n) * mutation_strength
    h = normalize_h(h)

    target_sum = n / 2.0
    current_sum = h.sum()
    if current_sum > 0:
        scale = target_sum / current_sum
        if scale > 1.0 and h.max() * scale > 1.0:
            # Would exceed 1.0 after scaling, blend toward center
            h = 0.9 * h + 0.1 * 0.5
            current_sum = h.sum()
            scale = target_sum / current_sum
        h = h * scale

    return np.clip(h, 0.0, 1.0)


def local_optimize_h(
    h_values: np.ndarray,
    steps: int = 50,
    mutation_strength: float = 0.1,
) -> np.ndarray:
    """Gradient-free local optimization via repeated mutation."""
    h_best = h_values.copy()
    c5_best = compute_c5(h_best)

    for _ in range(steps):
        h_candidate = mutate_h(h_best, mutation_strength)
        c5_candidate = compute_c5(h_candidate)
        if c5_candidate < c5_best:
            h_best = h_candidate
            c5_best = c5_candidate

    return h_best


def generate_children_via_mutation(
    parent_h: np.ndarray,
    num_children: int,
    mutation_strength: float = 0.1,
) -> list[tuple[np.ndarray, float]]:
    """
    Generate child h-functions via mutations (no LLM).

    Uses a mix of strategies: 60% mutate, 30% local_opt, 10% reinit.
    Returns list of (h_values, c5) tuples.
    """
    children = []
    for _ in range(num_children):
        strategy = np.random.choice(
            ["mutate", "local_opt", "reinit"], p=[0.6, 0.3, 0.1]
        )
        if strategy == "mutate":
            h_new = mutate_h(parent_h, mutation_strength)
        elif strategy == "local_opt":
            h_new = local_optimize_h(parent_h, steps=20, mutation_strength=mutation_strength)
        else:
            h_new = create_random_h(len(parent_h))

        c5 = compute_c5(h_new)
        children.append((h_new, c5))

    return children
