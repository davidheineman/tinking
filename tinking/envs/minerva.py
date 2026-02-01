import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Literal

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import tinker
from tinker import SamplingClient, ServiceClient
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.rl.types import TrajectoryGroup, Trajectory, Transition

from minieval.tasks.minerva import MinervaMath, Math500
from minieval.datatypes import TaskConfig as MiniEvalTaskConfig, Instance, LMOutput
from minieval.score.core import ExactMatchFlex
from minieval.extract.math_latex import MathExtractor
from minieval.formatters import CoT

from tinking.envs.base import Environment, EnvironmentConfig

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class MinervaConfig(EnvironmentConfig):
    name: Literal["minerva"] = "minerva"
    model_name: str = ""
    dataset: Literal["minerva", "math500"] = "minerva"
    subset: str | None = None  # For minerva dataset, e.g., "algebra"
    max_tokens: int = 2048
    temperature: float = 0.7
    n_concurrent: int = 1
    num_questions_per_batch: int = 4  # How many questions to sample per batch


class MinervaEnvironment(Environment):
    def __init__(
        self,
        config: MinervaConfig,
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
        rollouts, rewards = await self._run_minerva_rollouts(
            config=self.config,
            sampling_client_path=sampling_client_path,
            group_size=self.group_size,
            renderer=self.renderer,
        )
        
        if not rollouts:
            logger.warning("No rollouts generated!")
            return []
        
        # Convert to trajectory groups
        trajectory_groups = self._rollouts_to_trajectory_groups(
            rollouts, rewards, self.group_size, self.renderer
        )
        
        return trajectory_groups


    def _create_minieval_task(self, config: MinervaConfig) -> MinervaMath | Math500:
        """Create a minieval task for loading the dataset."""
        # Create a minimal TaskConfig for minieval
        task_config = MiniEvalTaskConfig(
            alias="minerva_rl",
            formatter=CoT(instruction="Present the answer in LaTex format: \\boxed{Your answer}"),
            scorer=[ExactMatchFlex()],
            metric=[],
            subset=config.subset,
        )
        
        match config.dataset:
            case "math500":
                return Math500(task_config)
            case "minerva":
                return MinervaMath(task_config)
            case other:
                raise ValueError(f"Unknown dataset: {other!r}")


    def _grade_response(
        self, 
        instance: Instance, 
        response_text: str
    ) -> float:
        lm_output = LMOutput(text=response_text)
        
        extracted_answers = MathExtractor.extract_answer(response_text)
        lm_output.extracted_answer = extracted_answers

        scorer = ExactMatchFlex()
        score = scorer._score_response_single(instance, lm_output)
        
        return score


    async def _run_minerva_rollouts(
        self,
        config: MinervaConfig,
        sampling_client_path: str,
        group_size: int,
        renderer: Renderer,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Run rollouts on Minerva math problems."""
        
        # Create minieval task to load dataset
        task = self._create_minieval_task(config)
        
        # Get all instances (questions) from the task
        all_instances: list[Instance] = task.requests
        
        # Sample questions for this batch
        sampled_instances = random.sample(
            all_instances, 
            min(config.num_questions_per_batch, len(all_instances))
        )
        
        # Create sampling client
        service_client: ServiceClient = ServiceClient()
        sampling_client: SamplingClient = service_client.create_sampling_client(
            model_path=sampling_client_path
        )
        
        rollouts = []
        rewards = []
        
        async def process_instance(instance: Instance, num_samples: int):
            """Generate samples for a single question."""
            # Use the task's formatter to build the message
            formatter = task.config.formatter
            lm_request = formatter._build_message(instance)
            messages = lm_request.messages
            
            prompt = renderer.build_generation_prompt(messages)
            
            # Generate multiple samples for this question
            sample_result = await sampling_client.sample_async(
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=tinker.SamplingParams(
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    stop=renderer.get_stop_sequences(),
                ),
            )
            
            instance_rollouts = []
            instance_rewards = []
            
            for seq in sample_result.sequences:
                response_text = renderer.tokenizer.decode(seq.tokens)
                
                # Build the full conversation
                full_messages = messages + [{"role": "assistant", "content": response_text}]
                
                # Grade the response using MathExtractor + ExactMatchFlex
                reward = self._grade_response(instance, response_text)
                
                instance_rollouts.append({
                    "messages": full_messages,
                    "instance": instance,
                    "response": response_text,
                    "tokens": seq.tokens,
                    "logprobs": seq.logprobs,
                })
                instance_rewards.append(reward)
            
            return instance_rollouts, instance_rewards
        
        # Determine samples per question to reach group_size total
        samples_per_question = max(1, group_size // len(sampled_instances))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            prog_task = progress.add_task("Generating math solutions...", total=len(sampled_instances))
            
            # Process questions concurrently
            async def process_with_progress(instance: Instance):
                result = await process_instance(instance, samples_per_question)
                progress.advance(prog_task)
                return result
            
            results = await asyncio.gather(*[
                process_with_progress(inst) for inst in sampled_instances
            ])
            
            for inst_rollouts, inst_rewards in results:
                rollouts.extend(inst_rollouts)
                rewards.extend(inst_rewards)
        
        logger.info(f"Generated {len(rollouts)} rollouts, mean reward: {sum(rewards)/len(rewards):.3f}")
        
        return rollouts, rewards


    def _rollouts_to_trajectory_groups(
        self,
        rollouts: list[dict[str, Any]], 
        rewards: list[float],
        group_size: int,
        renderer: Renderer,
    ) -> list[TrajectoryGroup]:
        """Convert rollouts to TrajectoryGroups."""
        if len(rollouts) != len(rewards):
            raise ValueError(f"Length mismatch: {len(rollouts)} rollouts but {len(rewards)} rewards")
        
        # Convert each rollout to a Trajectory
        trajectories = []
        for rollout in rollouts:
            messages = rollout.get("messages", [])
            tokens = rollout.get("tokens", [])
            logprobs = rollout.get("logprobs", [])
            
            trajectory = self._messages_to_trajectory(messages, tokens, logprobs, renderer)
            trajectories.append(trajectory)
        
        # Group trajectories into TrajectoryGroups
        trajectory_groups = []
        for i in range(0, len(trajectories), group_size):
            group_trajectories = trajectories[i:i+group_size]
            group_rewards = rewards[i:i+group_size]
            
            # Create metrics dict for each trajectory
            group_metrics = [{} for _ in group_trajectories]
            
            trajectory_group = TrajectoryGroup(
                trajectories_G=group_trajectories,
                final_rewards_G=group_rewards,
                metrics_G=group_metrics,
            )
            trajectory_groups.append(trajectory_group)
        
        return trajectory_groups


    def _messages_to_trajectory(
        self,
        messages: list[dict[str, Any]], 
        tokens: list[int],
        logprobs: list[float],
        renderer: Renderer,
    ) -> Trajectory:
        """Convert messages to a Trajectory with logprobs."""
        transitions = []
        
        # Find the user message index (skip system message)
        user_idx = None
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                user_idx = i
                break
        
        if user_idx is not None and user_idx + 1 < len(messages):
            # Observation: conversation up to and including user message
            ob = renderer.build_generation_prompt(messages[:user_idx + 1])
            
            # Action: tokenized assistant response
            ac = TokensWithLogprobs(
                tokens=tokens,
                maybe_logprobs=logprobs if logprobs else None
            )
            
            transitions.append(Transition(
                ob=ob,
                ac=ac,
                reward=0.0,  # Final reward is in TrajectoryGroup
                episode_done=True,  # Single turn for math problems
                metrics={},
            ))
        
        return Trajectory(
            transitions=transitions,
            final_ob=renderer.build_generation_prompt(messages)
        )
