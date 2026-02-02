from dataclasses import dataclass
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv


class ArithmeticEnv(ProblemEnv):
    """ toy test env """

    def __init__(
        self,
        instance,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        super().__init__(renderer, convo_prefix)
        self.x = instance.x
        self.y = instance.y

    def get_question(self) -> str:
        return f"What is {self.x} + {self.y}?"

    def check_answer(self, sample_str: str) -> bool:
        chunks = sample_str.split()
        try:
            answer = int(chunks[0])
        except (ValueError, IndexError):
            return False
        return answer == self.x + self.y

    def check_format(self, sample_str: str) -> bool:
        return True

    def get_reference_answer(self) -> str:
        return str(self.x + self.y)

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {"role": "user", "content": "What is 4 + 5?"},
            {"role": "assistant", "content": "9"},
        ]

@dataclass
class ArithmeticInstance:
    """Instance for arithmetic problems."""
    x: int
    y: int
    
    @property
    def question(self) -> str:
        return f"What is {self.x} + {self.y}?"
    
    @property
    def solution(self) -> str:
        return str(self.x + self.y)


