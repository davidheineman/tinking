"""
Simple coding environment for RL training.

This environment can be customized for different coding tasks.
"""

import subprocess
from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence

import tinker
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, StepResult


class CodingEnv(Env):
    """
    A simple coding environment that rewards correct code execution.
    
    Subclass this to implement specific coding tasks.
    """
    
    def __init__(self, renderer, prompt: str, expected_output: str | None = None):
        self.renderer = renderer
        self.prompt = prompt
        self.expected_output = expected_output
        self.stop_condition = self.renderer.get_stop_sequences()
        self.response_text = ""
        
    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
        """Create the initial prompt for the coding task."""
        convo = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(convo)
        return model_input, self.stop_condition
    
    @abstractmethod
    async def evaluate_code(self, code: str) -> tuple[bool, float, dict]:
        """
        Evaluate the generated code.
        
        Returns:
            (is_correct, reward, metrics)
        """
        pass
    
    async def step(self, action: list[int]) -> StepResult:
        """Execute one step - parse the response and evaluate it."""
        message, parse_success = self.renderer.parse_response(action)
        self.response_text = message.get("content", "")
        
        # Extract code from response (simple heuristic - can be customized)
        code = self._extract_code(self.response_text)
        
        # Evaluate the code
        is_correct, reward, metrics = await self.evaluate_code(code)
        
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={"correct": 1.0 if is_correct else 0.0, **metrics},
        )
    
    def _extract_code(self, text: str) -> str:
        """Extract code from response text. Simple implementation - can be improved."""
        # Look for code blocks
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                # Extract code between first set of triple backticks
                code = parts[1]
                # Remove language identifier if present
                if "\n" in code:
                    lines = code.split("\n")
                    if lines[0].strip() in ["python", "py", ""]:
                        code = "\n".join(lines[1:])
                    else:
                        code = "\n".join(lines)
                return code.strip()
        return text.strip()


class PythonCodingEnv(CodingEnv):
    """
    A Python coding environment that executes Python code and checks output.
    """
    
    async def evaluate_code(self, code: str) -> tuple[bool, float, dict]:
        """Execute Python code and check if it produces expected output."""
        try:
            # Run the code in a subprocess with timeout
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            output = result.stdout.strip()
            has_error = result.returncode != 0
            
            # If we have expected output, check against it
            if self.expected_output is not None:
                is_correct = output == self.expected_output and not has_error
            else:
                # Reward for no errors
                is_correct = not has_error
            
            reward = 1.0 if is_correct else 0.0
            metrics = {
                "has_error": 1.0 if has_error else 0.0,
                "output": output[:100],  # Truncate for logging
            }
            
            return is_correct, reward, metrics
            
        except subprocess.TimeoutExpired:
            return False, 0.0, {"timeout": 1.0}
        except Exception as e:
            return False, 0.0, {"exception": str(e)[:100]}


@dataclass
class CodingGroupBuilder(EnvGroupBuilder):
    """Builder for groups of coding environments."""
    
    renderer: any
    prompts: list[str]
    expected_outputs: list[str] | None = None
    env_class: type = PythonCodingEnv
    
    async def make_envs(self) -> Sequence[Env]:
        """Create environments for each prompt."""
        envs = []
        for i, prompt in enumerate(self.prompts):
            expected = None
            if self.expected_outputs and i < len(self.expected_outputs):
                expected = self.expected_outputs[i]
            envs.append(self.env_class(self.renderer, prompt, expected))
        return envs
    
    def logging_tags(self) -> list[str]:
        return ["coding"]

