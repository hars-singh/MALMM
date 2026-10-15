"""Original planner, coder, and task prompts."""

from .coder import CODER_PROMPT
from .planner import PLANNER_PROMPT
from .task import TASK_PROMPT

__all__ = ["CODER_PROMPT", "PLANNER_PROMPT", "TASK_PROMPT", "format_task_prompt"]


def format_task_prompt(instruction: str, position, orientation, scene: str) -> str:
    values = {
        "[INSERT TASK]": instruction,
        "[INSERT EE POSITION]": str(position),
        "[INSERT EE ORIENTATION]": str(orientation),
        "[STATE]": scene,
    }
    prompt = TASK_PROMPT
    for token, value in values.items():
        prompt = prompt.replace(token, value)
    return prompt
