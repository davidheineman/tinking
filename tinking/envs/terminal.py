import asyncio
import logging
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from tinker import SamplingClient, ServiceClient
import tinker
from tinker_cookbook.rl.types import TrajectoryGroup, Trajectory, Transition
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.renderers import Renderer

from tinking.constants import CURRENT_TASKS, BUILT_TBENCH_TASKS
from tinking.envs.base import Environment, EnvironmentConfig

console = Console()

logger = logging.getLogger(__name__)


@dataclass
class TerminalBenchConfig(EnvironmentConfig):
    name: Literal["terminal"] = "terminal"
    agent: str = "terminus-tinker"
    model_name: str = ""
    dataset_path: str = ""
    dataset: str = ""
    task_id: list[str] = None
    n_concurrent: int = 1
    grader_model: str = "random"


class TerminalBenchEnvironment(Environment):
    def __init__(
        self,
        config: TerminalBenchConfig,
        group_size: int,
        renderer: Renderer,
        output_dir: str,
    ):
        self.config = config
        self.group_size = group_size
        self.renderer = renderer
        self.output_dir = output_dir
    
    async def do_rollouts(
        self,
        sampling_client_path: str,
        batch_idx: int,
    ) -> list[TrajectoryGroup]:
        trajectory_groups = self._do_terminalbench_rollouts(
            config=self.config,
            sampling_client_path=sampling_client_path,
            batch_idx=batch_idx,
            group_size=self.group_size,
            renderer=self.renderer,
            grader_model=self.config.grader_model,
            output_dir=self.output_dir,
        )

        # Fill in logprobs using the sampling client
        trajectory_groups = await self._recompute_logprobs(
            trajectory_groups, sampling_client_path, self.renderer
        )

        return trajectory_groups

    
    def _do_terminalbench_rollouts(
        self,
        config: TerminalBenchConfig,
        sampling_client_path: str,
        batch_idx: int,
        group_size: int,
        renderer: Renderer,
        grader_model: str,
        output_dir: str,
    ) -> list[TrajectoryGroup]:
        """ run tb -> extract rollouts -> grade -> convert to TrajectoryGroups """
        # Step 1: Run tb to generate rollouts (group_size runs in parallel)
        def run_single_rollout(i):
            return self._run_minitb_rollouts(config, sampling_client_path, f"{batch_idx:06d}_{i}", output_dir)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running rollouts...", total=group_size)
            
            with ThreadPoolExecutor(max_workers=group_size) as executor:
                futures = [executor.submit(run_single_rollout, i) for i in range(group_size)]
                output_dirs = []
                
                for future in futures:
                    output_dirs.append(future.result())
                    progress.advance(task)
        
        # Step 2: Extract rollouts as List[Message]
        rollouts = []
        for output_dir in output_dirs:
            rollouts.extend(self._extract_rollouts(output_dir))
        
        if not rollouts:
            logger.warning("No rollouts extracted! Returning empty list.")
            return []
        
        # Step 3: Grade rollouts
        rewards = self._grade_rollouts(rollouts, grader_model)
        
        # Step 4: List[Message] -> TrajectoryGroups
        trajectory_groups = self._rollouts_to_trajectory_groups(rollouts, rewards, group_size, renderer)
        
        return trajectory_groups


    def _preflight_checks(self, config: TerminalBenchConfig):
        # Validate that exactly one of dataset_path or dataset is set
        if not config.dataset_path and not config.dataset:
            raise ValueError("Either dataset_path or dataset must be set in TerminalBenchConfig")
        elif config.dataset_path and config.dataset:
            raise ValueError("Only one of dataset_path or dataset should be set, not both")


    def _run_minitb_rollouts(
        self,
        config: TerminalBenchConfig,
        sampling_client_path: str,
        batch_idx: int,
        output_dir: str,
    ) -> Path:
        """ launch tbench """
        self._preflight_checks(config)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(output_dir) / f"{timestamp}-batch{batch_idx}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        if "papergym" in config.dataset_path:
            # manual override for built images in beaker
            task_id = random.choice(CURRENT_TASKS)
        elif "terminal-bench" in config.dataset:
            task_id = random.choice(BUILT_TBENCH_TASKS)
        else:
            task_id = random.choice(config.task_id)
        
        cmd = [
            "minitb",
            "run",
            "--agent", config.agent,
            "--agent-kwarg", f"checkpoint_path={sampling_client_path}",
            "--agent-kwarg", f"model_name={config.model_name}",
            "--task-id", task_id,
            "--n-concurrent", str(config.n_concurrent),
            "--output-path", str(output_path),
        ]

        if config.dataset:
            cmd += [
                "--dataset", config.dataset,
            ]
        
        if config.dataset_path:
            dataset_path_expanded = str(Path(config.dataset_path).expanduser())
            cmd += [
                "--dataset-path", dataset_path_expanded,
            ]
        
        logger.info(f"Running tb: {' '.join(cmd)}")
        
        # Run tb as subprocess with streaming logs
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            logger.info(f"minitb: {line.rstrip()}")
        
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"minitb run failed with return code {return_code}")
        
        return output_path
        


    def _extract_rollouts(self, output_dir: Path) -> list[dict[str, Any]]:
        """ agent-logs/ -> openai compatible messages """
        rollouts = []

        print(f'Seeing output_dir={output_dir}')
        
        # Get episodes: output_dir/datetime/task/instance/agent-logs/episode-N
        for instance_dir in output_dir.glob("*/*/*/"):
            agent_logs = instance_dir / "agent-logs"
            if not agent_logs.exists():
                continue
            
            # Build messages from episodes
            messages = []
            for episode_dir in sorted(agent_logs.glob("episode-*")):
                prompt_file = episode_dir / "prompt.txt"
                response_file = episode_dir / "response.txt"
                
                if prompt_file.exists() and response_file.exists():
                    messages.append({"role": "user", "content": prompt_file.read_text()})
                    messages.append({"role": "assistant", "content": response_file.read_text()})
            
            if messages:
                rollouts.append({"messages": messages})
        
        return rollouts

    def _grade_rollouts(self, rollouts: list[dict[str, Any]], grader_model: str | None = None) -> list[float]:
        """ Grade rollout """
        logger.info(f"Using grader: {grader_model}.")
        
        if grader_model == "random":
            # Channeling my inner demons https://arxiv.org/abs/2506.10947
            rewards = []
            for i, rollout in enumerate(rollouts):
                placeholder_reward = random.random()
                rewards.append(placeholder_reward)
        else:
            raise ValueError(f"grader_model={grader_model}")

        return rewards

    def _rollouts_to_trajectory_groups(
        self,
        rollouts: list[dict[str, Any]], 
        rewards: list[float],
        group_size: int,
        renderer: Renderer,
    ) -> list[TrajectoryGroup]:
        """ rollouts -> TrajectoryGroup """
        if len(rollouts) != len(rewards):
            raise ValueError(f"Length mismatch: {len(rollouts)} rollouts but {len(rewards)} rewards")
        
        # Convert each rollout to a Trajectory
        trajectories = []
        for rollout, final_reward in zip(rollouts, rewards):
            messages = rollout.get("messages", [])
            trajectory = self._messages_to_trajectory(messages, renderer)
            trajectories.append(trajectory)
        
        # Group trajectories into TrajectoryGroups
        trajectory_groups = []
        for i in range(0, len(trajectories), group_size):
            group_trajectories = trajectories[i:i+group_size]
            group_rewards = rewards[i:i+group_size]
            
            # Create metrics dict for each trajectory (empty for now)
            group_metrics = [{} for _ in group_trajectories]
            
            trajectory_group = TrajectoryGroup(
                trajectories_G=group_trajectories,
                final_rewards_G=group_rewards,
                metrics_G=group_metrics,
            )
            trajectory_groups.append(trajectory_group)
        
        return trajectory_groups

    def _messages_to_trajectory(messages: list[dict[str, Any]], renderer: Renderer) -> Trajectory:
        """ rollout -> Trajectory """
        # Find user-assistant pairs
        transitions = []
        for i in range(0, len(messages) - 1, 2):
            if messages[i]["role"] != "user" or messages[i+1]["role"] != "assistant":
                continue
            
            # Observation: conversation up to and including user message
            ob = renderer.build_generation_prompt(messages[:i+1])
            
            # Action: tokenized assistant response
            ac = TokensWithLogprobs(
                tokens=renderer.tokenizer.encode(messages[i+1]["content"], add_special_tokens=False),
                maybe_logprobs=None  # Will be filled in later
            )
            
            transitions.append(Transition(
                ob=ob,
                ac=ac,
                reward=0.0,
                episode_done=(i + 2 >= len(messages)),  # Last transition is done
                metrics={},
            ))
        
        return Trajectory(
            transitions=transitions,
            final_ob=renderer.build_generation_prompt(messages)
        )

    async def _recompute_logprobs(
        self,
        trajectory_groups: list[TrajectoryGroup],
        sampling_client_path: str,
        renderer: Renderer,
    ) -> list[TrajectoryGroup]:
        """ fill in logprobs for trajectories  """
        # Create sampling client
        service_client: ServiceClient = ServiceClient()
        sampling_client: SamplingClient = service_client.create_sampling_client(
            model_path=sampling_client_path
        )
        
        logger.info("Re-computing logprobs for trajectories...")
        
        # Collect all transitions that need logprob computation
        transitions_to_process = []
        for traj_group in trajectory_groups:
            for trajectory in traj_group.trajectories_G:
                for transition in trajectory.transitions:
                    transitions_to_process.append(transition)
        
        async def process_transition(transition: Transition):
            """ Compute logprobs on a single transititon """
            sample_result = await sampling_client.sample_async(
                prompt=transition.ob,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=1.0,
                    max_tokens=len(transition.ac.tokens) + 100,
                    stop=renderer.get_stop_sequences(),
                ),
            )
            
            # Get logprobs from sampled sequence
            sampled_logprobs = sample_result.sequences[0].logprobs
            
            # Match logprobs to the actual tokens taken
            if sampled_logprobs and len(sampled_logprobs) >= len(transition.ac.tokens):
                transition.ac.maybe_logprobs = sampled_logprobs[:len(transition.ac.tokens)]
            else:
                logger.warning(
                    f"Not enough logprobs: got {len(sampled_logprobs) if sampled_logprobs else 0}, "
                    f"need {len(transition.ac.tokens)}. Using zeros."
                )
                transition.ac.maybe_logprobs = [0.0] * len(transition.ac.tokens)
        
        # Process all transitions concurrently
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Computing logprobs...", total=len(transitions_to_process))
            
            # Process transitions and update progress
            async def process_with_progress(transition):
                result = await process_transition(transition)
                progress.advance(task)
                return result
            
            await asyncio.gather(*[process_with_progress(transition) for transition in transitions_to_process])
        
        logger.info("Finished re-computing logprobs")
        return trajectory_groups