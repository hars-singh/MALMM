"""Original planner, coder, and task prompts."""

from .engineer_system_message_m import ENGINEER_SYSTEM_MESSAGE
from .planner_system_message_m import PLANNER_SYSTEM_MESSAGE
from .task_prompt_m import TASK_PROMPT


def task_prompt(command, position, orientation, state):
    values = {
        "[INSERT TASK]": command,
        "[INSERT EE POSITION]": str(position),
        "[INSERT EE ORIENTATION]": str(orientation),
        "[STATE]": state,
    }
    prompt = TASK_PROMPT
    for token, value in values.items():
        prompt = prompt.replace(token, value)
    return prompt
