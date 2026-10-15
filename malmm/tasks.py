"""Object names and labels used to describe RLBench scenes to the agents."""

import json
from importlib.resources import files

_DATA = json.loads(files("malmm").joinpath("task_objects.json").read_text())
TASK_OBJECTS = _DATA["tasks"]
TASK_LABELS = _DATA["labels"]
COLORS = {tuple(item["rgb"]): item["name"] for item in _DATA["colors"]}


def task_names():
    return sorted(TASK_OBJECTS)
