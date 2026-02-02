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

from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from minieval.tasks.minerva import Instance, MinervaMath, Math500
from minieval.datatypes import TaskConfig as MiniEvalTaskConfig, LMOutput
from minieval.score.core import ExactMatchFlex
from minieval.extract.math_latex import MathExtractor
from minieval.formatters import CoT

from tinking.envs.arithmetic import ArithmeticEnv, ArithmeticInstance
from tinking.envs.base import Environment, EnvironmentConfig

console = Console()


@dataclass
class MathInstance:
    question: str
    solution: str
    
    @property
    def answer(self) -> list[str]:
        return [self.solution]


@dataclass
class MathConfig(EnvironmentConfig):
    """Config for Minerva math environment."""
    name: Literal["minerva"] = "minerva"
    dataset: Literal["arithmetic", "minerva", "math500", "hendrycks", "olmo3_rlzero_7b", "olmo3_rl_32b", "nemotron", "deepmath", "polaris"] = "minerva"
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
        dataset: str = "minerva",
        subset: str | None = None,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.dataset = dataset
        self.seed = seed
        
        # Load data
        self.instances = self._load_instances(dataset, subset)
        
        # Shuffle
        rng = random.Random(seed)
        rng.shuffle(self.instances)
    
    def _load_instances(self, dataset: str, subset: str | None) -> list:
        match dataset:
            case "arithmetic":
                # Generate 10k arithmetic problems
                rng = random.Random(self.seed)
                return [
                    ArithmeticInstance(x=rng.randint(0, 100), y=rng.randint(0, 100))
                    for _ in range(10000)
                ]
            case "olmo3_rlzero_7b":
                ds = load_dataset("allenai/Dolci-RL-Zero-Math-7B", split="train")
                return [
                    MathInstance(question=row["prompt"], solution=row["ground_truth"])
                    for row in ds
                ]
            case "nemotron":
                nemotron_subset = subset or "high_part02"
                ds = load_dataset("nvidia/Nemotron-Math-v2", split=f"{nemotron_subset}")
                return [
                    MathInstance(question=row["problem"], solution=row["expected_answer"])
                    for row in ds
                ]
            case "olmo3_rl_32b":
                ds = load_dataset("allenai/Dolci-Think-RL-32B", split="train")
                return [
                    MathInstance(question=row["prompt"], solution=row["ground_truth"][0])
                    for row in ds
                    if "math" in row["dataset"]
                ]
            case "deepmath":
                ds = load_dataset("zwhe99/DeepMath-103K", split="train")
                # Sort by difficulty descending and take top 1K
                ds = ds.sort("difficulty", reverse=True).select(range(1000))
                return [
                    MathInstance(question=row["question"], solution=row["final_answer"])
                    for row in ds
                ]
            case "polaris":
                ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
                # Filter to hardest problems (0/8 or 1/8 pass rate)
                return [
                    MathInstance(question=row["problem"], solution=row["answer"])
                    for row in ds
                    if row["difficulty"] in ("0/8", "1/8")
                ]
            case "hendrycks":
                # For Hendrycks MATH, the standard is to use both the "train" and "test" splits for
                # training. The "test" split here is NOT the same as the MATH-500 test split,
                # which is a commonly-held-out subset of 500 of the below 12.5k problems. To construct
                # a clean training set, we filter out problems that exist in the MATH-500 test set,
                # resulting in ~12000 train and 500 test problems.
                
                # Get MATH-500 test problems to exclude
                math500_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
                test_problems: set[str] = {row["problem"] for row in math500_ds}
                
                # Load all Hendrycks MATH configs and splits, excluding MATH-500 test problems
                dataset_name = "EleutherAI/hendrycks_math"
                configs = get_dataset_config_names(dataset_name)
                pieces = []
                for cfg in configs:
                    for split in ("train", "test"):
                        ds = load_dataset(dataset_name, name=cfg, split=split)
                        ds = ds.filter(lambda example: example["problem"] not in test_problems)
                        pieces.append(ds)
                full_dataset = concatenate_datasets(pieces)
                
                return [
                    MathInstance(question=row["problem"], solution=row["solution"])
                    for row in full_dataset
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
    
    def _make_env_group_builder(self, instance) -> ProblemGroupBuilder | None:
        if not instance.question or not instance.solution:
            logger.warning("Skipping instance with missing question/solution")
            return None
        
        if self.dataset == "arithmetic":
            environ = ArithmeticEnv
            convo_prefix = ArithmeticEnv.standard_fewshot_prefix()
        else:
            environ = MathEnv
            convo_prefix = self.convo_prefix
        
        return ProblemGroupBuilder(
            env_thunk=partial(
                environ, instance, 
                renderer=self.renderer,
                convo_prefix=convo_prefix,
            ),
            num_envs=self.group_size,
            dataset_name=self.dataset,
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
