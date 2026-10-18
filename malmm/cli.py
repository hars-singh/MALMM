"""Command-line entry point for the MALMM method."""

import argparse
import os
from pathlib import Path

from .tasks import TASK_NAMES


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def create_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="MALMM zero-shot robotic manipulation")
    result.add_argument(
        "--list-tasks", action="store_true", help="List tasks with saved object mappings"
    )
    result.add_argument("--task", choices=TASK_NAMES, default="put_rubbish_in_bin")
    result.add_argument("--episodes", type=positive_int, default=25)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-rounds", type=positive_int, default=300)
    result.add_argument("--max-actions", type=positive_int, default=60)
    result.add_argument("--headless", action="store_true")
    result.add_argument("--env-file", type=Path, default=Path(".env"))
    result.add_argument("--model", help="Overrides MALMM_MODEL")
    result.add_argument(
        "--base-url", help="Overrides MALMM_BASE_URL; omit for the provider's default endpoint"
    )
    result.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return result


def load_model_config(args: argparse.Namespace) -> dict:
    from dotenv import dotenv_values

    settings = {**dotenv_values(args.env_file), **os.environ}
    api_key = settings.get("MALMM_API_KEY") or settings.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set MALMM_API_KEY in .env (use 'local' for an unauthenticated local server)."
        )
    model = args.model or settings.get("MALMM_MODEL", "gpt-4-turbo")
    base_url = args.base_url or settings.get("MALMM_BASE_URL")
    config = {"model": model, "api_key": api_key}
    if base_url:
        config["base_url"] = base_url
    return {"config_list": [config], "cache_seed": None}


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.list_tasks:
        print("\n".join(TASK_NAMES))
        return 0
    try:
        model_config = load_model_config(args)
    except ValueError as error:
        parser.error(str(error))

    from .runner import run_episodes

    run_dir = run_episodes(
        args.task,
        model_config,
        episodes=args.episodes,
        seed=args.seed,
        max_rounds=args.max_rounds,
        max_actions=args.max_actions,
        headless=args.headless,
        output_dir=args.output_dir,
    )
    print(f"Results: {run_dir}")
    return 0
