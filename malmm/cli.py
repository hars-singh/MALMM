"""Command-line entry point for the MALMM method."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import random

from .prompts import task_prompt
from .tasks import task_names


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser():
    result = argparse.ArgumentParser(description="MALMM zero-shot robotic manipulation")
    result.add_argument("--list-tasks", action="store_true", help="List tasks with saved object mappings")
    result.add_argument("--task", choices=task_names(), default="put_rubbish_in_bin")
    result.add_argument("--episodes", type=positive_int, default=25)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-rounds", type=positive_int, default=100)
    result.add_argument("--headless", action="store_true")
    result.add_argument("--env-file", type=Path, default=Path(".env"))
    result.add_argument("--model", help="Overrides MALMM_MODEL")
    result.add_argument("--base-url", help="Overrides MALMM_BASE_URL; omit for the provider's default endpoint")
    result.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return result


def llm_config(args):
    from dotenv import load_dotenv

    load_dotenv(args.env_file, override=False)
    api_key = os.getenv("MALMM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set MALMM_API_KEY in .env (use 'local' for an unauthenticated local server).")
    model = args.model or os.getenv("MALMM_MODEL", "gpt-4-turbo")
    base_url = args.base_url or os.getenv("MALMM_BASE_URL")
    config = {"model": model, "api_key": api_key}
    if base_url:
        config["base_url"] = base_url
    return {"config_list": [config], "cache_seed": None}


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.list_tasks:
        print("\n".join(task_names()))
        return 0
    try:
        config = llm_config(args)
    except ValueError as exc:
        arguments.error(str(exc))

    import numpy as np
    from .agents import build_team
    from .environment import RLBenchRobot

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = args.output_dir / args.task / stamp
    directory.mkdir(parents=True)
    public_config = config["config_list"][0]
    results = {
        "task": args.task, "model": public_config["model"],
        "episodes_requested": args.episodes, "seed": args.seed, "episodes": [],
    }
    # Credentials are never serialized in run metadata or conversation histories.
    (directory / "config.json").write_text(json.dumps({
        **{key: value for key, value in results.items() if key != "episodes"},
        "max_rounds": args.max_rounds, "headless": args.headless,
    }, indent=2) + "\n")
    robot = None
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        robot = RLBenchRobot(args.task, headless=args.headless)
        for episode in range(args.episodes):
            seed = args.seed + episode
            random.seed(seed)
            np.random.seed(seed)
            command = robot.reset(seed)
            state = StringIO()
            with redirect_stdout(state):
                robot.detect_object()
            prompt = task_prompt(command, robot.arm.get_tip().get_position(),
                                 robot.arm.get_tip().get_orientation(), state.getvalue())
            # Build a new team so no conversation or executor state leaks between episodes.
            team = build_team(robot, config, args.max_rounds)
            record = {"episode": episode, "seed": seed, "instruction": command}
            try:
                team.run(prompt)
                record.update(success=robot.success(), stopped_by_agent=robot.finished)
            except Exception as exc:
                record.update(success=False, error_type=type(exc).__name__)
                raise
            finally:
                results["episodes"].append(record)
                (directory / f"episode_{episode:03d}.json").write_text(
                    json.dumps({"result": record, "messages": team.groupchat.messages}, indent=2) + "\n"
                )
                (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"Episode {episode + 1}/{args.episodes}: success={record['success']}")
    finally:
        if robot is not None:
            robot.close()
    print(f"Results: {directory}")
    return 0
