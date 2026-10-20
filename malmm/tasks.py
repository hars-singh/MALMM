"""Nine evaluation tasks and their scene-object configuration."""

import json
from importlib.resources import files

_DATA = json.loads(files("malmm").joinpath("tasks.json").read_text())
TASK_OBJECTS = _DATA["tasks"]
TASK_INSTRUCTIONS = _DATA.get("instructions", {})
TASK_LABELS = {
    task: {name: label for label, name in objects.items()} for task, objects in TASK_OBJECTS.items()
}
COLORS = {tuple(item["rgb"]): item["name"] for item in _DATA["colors"]}
