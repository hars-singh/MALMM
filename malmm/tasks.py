"""Object names and labels used to describe RLBench scenes to the agents."""

import json
from importlib.resources import files

_DATA = json.loads(files("malmm").joinpath("task_objects.json").read_text())
TASK_OBJECTS = _DATA["tasks"]
TASK_LABELS = _DATA["labels"]
COLORS = {tuple(item["rgb"]): item["name"] for item in _DATA["colors"]}
TASK_NAMES = tuple(sorted(TASK_OBJECTS))

EVALUATION_TASKS = (
    "basketball_in_hoop",
    "close_jar",
    "empty_container",
    "insert_onto_square_peg",
    "meat_off_grill",
    "open_wine_bottle",
    "slide_block_to_target",
    "put_rubbish_in_bin",
    "stack_blocks",
)
