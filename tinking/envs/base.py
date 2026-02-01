from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from tinker_cookbook.rl.types import TrajectoryGroup

EnvironmentName = Literal["minerva", "terminal"]


@dataclass
class EnvironmentConfig(ABC):
    name: EnvironmentName


class Environment(ABC):
    @abstractmethod
    async def do_rollouts(
        self,
        sampling_client_path: str,
        batch_idx: int,
    ) -> list[TrajectoryGroup]:
        """Run rollouts and return trajectory groups."""
        pass