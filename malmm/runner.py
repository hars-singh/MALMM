"""Run independent manipulation episodes and save their conversations and outcomes."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .agents import create_agent_team
from .prompts import format_task_prompt
from .robot import RLBenchRobot


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_episodes(
    task_name: str,
    model_config: dict,
    *,
    episodes: int = 25,
    seed: int = 0,
    max_rounds: int = 100,
    headless: bool = False,
    output_dir: Path = Path("outputs"),
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_dir / task_name / timestamp
    run_dir.mkdir(parents=True)
    settings = {
        "task": task_name,
        "model": model_config["config_list"][0]["model"],
        "episodes_requested": episodes,
        "seed": seed,
        "max_rounds": max_rounds,
        "headless": headless,
    }
    results = {**settings, "episodes": []}
    # Only explicit run settings enter metadata, never model credentials.
    _write_json(run_dir / "config.json", settings)
    _set_seed(seed)
    robot = RLBenchRobot(task_name, headless=headless)
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            _set_seed(episode_seed)
            instruction = robot.reset(episode_seed)
            position, orientation = robot.get_end_effector_pose()
            prompt = format_task_prompt(instruction, position, orientation, robot.describe_scene())
            team = create_agent_team(robot, model_config, max_rounds)
            outcome = {"episode": episode, "seed": episode_seed, "instruction": instruction}
            try:
                team.run(prompt)
                outcome.update(
                    success=robot.is_successful(),
                    stopped_by_agent=robot.completion_requested,
                )
            except (Exception, KeyboardInterrupt) as error:
                outcome.update(success=False, error_type=type(error).__name__)
                raise
            finally:
                outcome["token_usage"] = team.token_usage()
                results["episodes"].append(outcome)
                _write_json(
                    run_dir / f"episode_{episode:03d}.json",
                    {
                        "result": outcome,
                        "messages": team.conversation.messages,
                    },
                )
                _write_json(run_dir / "results.json", results)
            print(f"Episode {episode + 1}/{episodes}: success={outcome['success']}")
    finally:
        robot.close()
    return run_dir
