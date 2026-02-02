import asyncio
import math
from dataclasses import dataclass
from functools import partial
import random
from typing import Literal, Sequence

import tinker
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, TrajectoryGroup

from datasets import load_dataset
from minieval.tasks.minerva import Instance, MinervaMath, Math500
from minieval.datatypes import TaskConfig as MiniEvalTaskConfig, LMOutput
from minieval.score.core import ExactMatchFlex
from minieval.extract.math_latex import MathExtractor
from minieval.formatters import CoT

from tinking.envs.base import Environment, EnvironmentConfig

console = Console()

@dataclass
class HFInstance:
    question: str
    solution: str
    
    @property
    def answer(self) -> list[str]:
        return [self.solution]


@dataclass
class MathConfig(EnvironmentConfig):
    """Config for Minerva math environment."""
    name: Literal["minerva"] = "minerva"
    dataset: Literal["minerva", "math500", "dolci", "nemotron", "dolci_think"] = "minerva"
    subset: str | None = None
    seed: int = 0
    max_tokens: int = 2048


class MathEnv(ProblemEnv):
    def __init__(
        self,
        instance: Instance,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        super().__init__(renderer, convo_prefix)
        self.instance = instance
    
    @classmethod
    def question_suffix(cls) -> str:
        return " Present the answer in LaTeX format: \\boxed{Your answer}"
    
    def get_question(self) -> str:
        return self.instance.question + self.question_suffix()
    
    def check_format(self, sample_str: str) -> bool:
        answers = MathExtractor.extract_answer(sample_str)
        return len(answers) > 0
    
    def check_answer(self, sample_str: str) -> bool:
        lm_output = LMOutput(text=sample_str)
        lm_output.extracted_answer = MathExtractor.extract_answer(sample_str)
        
        scorer = ExactMatchFlex()
        score = scorer._score_response_single(self.instance, lm_output)
        return bool(score > 0)
    
    def get_reference_answer(self) -> str:
        return str(self.instance.solution)


class MathDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        dataset: Literal["minerva", "math500"] = "minerva",
        subset: str | None = None,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        
        # Load data
        self.instances = self._load_instances(dataset, subset)
        
        # Shuffle
        rng = random.Random(seed)
        rng.shuffle(self.instances)
    
    def _load_instances(self, dataset: str, subset: str | None) -> list:
        match dataset:
            case "olmo3_rlzero_7b":
                ds = load_dataset("allenai/Dolci-RL-Zero-Math-7B", split="train")
                return [
                    HFInstance(question=row["prompt"], solution=row["ground_truth"])
                    for row in ds
                ]
            case "nemotron":
                nemotron_subset = subset or "high_part02"
                ds = load_dataset("nvidia/Nemotron-Math-v2", split=f"{nemotron_subset}")
                return [
                    HFInstance(question=row["problem"], solution=row["expected_answer"])
                    for row in ds
                ]
            case "olmo3_rl_32b":
                ds = load_dataset("allenai/Dolci-Think-RL-32B", split="train")
                return [
                    HFInstance(question=row["prompt"], solution=row["ground_truth"][0])
                    for row in ds
                    if "math" in row["dataset"]
                ]
            case "math500" | "minerva":
                task_config = MiniEvalTaskConfig(
                    alias="minerva_rl",
                    formatter=CoT(instruction=""),
                    scorer=[ExactMatchFlex()],
                    metric=[],
                    subset=subset,
                )
                if dataset == "math500":
                    task = Math500(task_config)
                else:
                    task = MinervaMath(task_config)
                return list(task.requests)
            case _:
                raise ValueError(f"Unknown dataset: {dataset}")
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.instances))
        
        if batch_start >= len(self.instances):
            # Wrap around for continuous training
            batch_start = batch_start % len(self.instances)
            batch_end = min(batch_start + self.batch_size, len(self.instances))
        
        builders = []
        for instance in self.instances[batch_start:batch_end]:
            builder = self._make_env_group_builder(instance)
            if builder is not None:
                builders.append(builder)
        
        return builders
    
    def _make_env_group_builder(self, instance: Instance) -> ProblemGroupBuilder | None:
        if not instance.question or not instance.solution:
            logger.warning("Skipping instance with missing question/solution")
            return None
        
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                instance,
                self.renderer,
                convo_prefix=self.convo_prefix,
            ),
            num_envs=self.group_size,
            dataset_name="minerva",
        )
    
    def __len__(self) -> int:
        return math.ceil(len(self.instances) / self.batch_size)


class MathEnvironment(Environment):
    def __init__(
        self,
        config: MathConfig,
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
        
        # Create service client once for reuse across batches
        self.service_client = tinker.ServiceClient()
        
        self.dataset = MathDataset(
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            dataset=config.dataset,
            subset=config.subset,
            seed=config.seed,
        )
    
    async def do_rollouts(
        self,
        sampling_client_path: str,
        batch_idx: int,
    ) -> list[TrajectoryGroup]:
        # Get env group builders for this batch
        env_group_builders = self.dataset.get_batch(batch_idx)
        
        if not env_group_builders:
            logger.warning("No env group builders for this batch")
            return []
        
        # Create sampling client and policy
        sampling_client = self.service_client.create_sampling_client(model_path=sampling_client_path)
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.config.max_tokens)
        
        # Run rollouts with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running rollouts...", total=len(env_group_builders))
            
            async def rollout_with_progress(builder):
                result = await do_group_rollout(builder, policy)
                progress.advance(task)
                return result
            
            trajectory_groups: list[TrajectoryGroup] = await asyncio.gather(*[
                rollout_with_progress(builder) for builder in env_group_builders
            ])
        
        return trajectory_groups
