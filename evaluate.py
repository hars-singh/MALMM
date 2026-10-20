"""Evaluate MALMM on RLBench tasks: python evaluate.py --help."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from malmm.tasks import TASK_OBJECTS

ROOT = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def settings(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate MALMM with independent simulator processes for each task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tasks", nargs="+", metavar="TASK", choices=["all", *TASK_OBJECTS], default=["all"],
                        help="Tasks to evaluate; use all or one or more task names.")
    parser.add_argument("--episodes", dest="episodes_per_task", metavar="N", type=positive_int, default=4,
                        help="Number of episodes per task.")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed; episodes use consecutive seeds.")
    parser.add_argument("--workers", type=positive_int, default=4, help="Maximum parallel task processes.")
    parser.add_argument("--model", default="openai/gpt-6-luna", help="Model ID accepted by your API provider.")
    parser.add_argument("--base-url", help="API endpoint; otherwise use MALMM_BASE_URL from the environment.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env", help="Provider environment file.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluations",
                        help="Directory for timestamped evaluation runs.")
    parser.add_argument("--max-actions", type=positive_int, default=60, help="Robot action limit per episode.")
    parser.add_argument("--max-messages", type=positive_int, default=300, help="Group-chat message limit per episode.")
    parser.add_argument("--task-timeout", type=positive_int, default=3600,
                        help="Timeout in seconds for all episodes of one task.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True,
                        help="Hide the simulator window; --no-headless enables it.")
    parser.add_argument("--list-tasks", action="store_true", help="List task names and exit without starting evaluation.")
    args = parser.parse_args(argv)
    if args.list_tasks:
        print("\n".join(TASK_OBJECTS))
        parser.exit()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if "all" in args.tasks and args.tasks != ["all"]:
        parser.error("use --tasks all by itself, or specify individual task names")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicate names")
    config = vars(args)
    config.pop("list_tasks")
    config["tasks"] = list(TASK_OBJECTS) if args.tasks == ["all"] else args.tasks
    config["env_file"] = str(args.env_file.resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    return config


def model_config(config):
    from dotenv import dotenv_values

    environment = {**dotenv_values(config["env_file"]), **os.environ}
    key = environment.get("MALMM_API_KEY") or environment.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set MALMM_API_KEY in .env before launching evaluation.")
    model = {"model": config["model"], "api_key": key}
    endpoint = config["base_url"] or environment.get("MALMM_BASE_URL")
    if endpoint:
        model["base_url"] = endpoint
    return {"config_list": [model], "cache_seed": None}


def run_worker(task, directory):
    from malmm.runner import run_episodes

    config = json.loads((directory / "config.json").read_text())
    run_episodes(
        task,
        model_config(config),
        episodes=config["episodes_per_task"],
        seed=config["seed"],
        max_rounds=config["max_messages"],
        max_actions=config["max_actions"],
        headless=config["headless"],
        output_dir=directory,
    )


def summarize_task(directory):
    results = [
        json.loads(path.read_text())["result"]
        for path in sorted(directory.glob("*/episode_*.json"))
    ]
    completed = [result for result in results if "error_type" not in result]
    successes = sum(bool(result.get("success")) for result in completed)
    return {
        "episodes_recorded": len(results),
        "episodes_completed": len(completed),
        "successes": successes,
        "episode_errors": [result for result in results if "error_type" in result],
        "success_rate": successes / len(completed) if completed else None,
    }


def save_summary(directory, report):
    temporary = directory / "summary.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(directory / "summary.json")
    with (directory / "summary.csv").open("w", newline="") as output:
        fields = [
            "task",
            "requested",
            "completed",
            "successes",
            "success_rate",
            "status",
            "seconds",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for task, result in report["tasks"].items():
            writer.writerow(
                {
                    "task": task,
                    "requested": report["episodes_per_task"],
                    "completed": result["episodes_completed"],
                    "successes": result["successes"],
                    "success_rate": result["success_rate"],
                    "status": result["status"],
                    "seconds": result["seconds"],
                }
            )


def main(argv=None):
    config = settings(argv)
    model_config(config)  # Validate credentials locally before starting any simulators.
    directory = Path(config["output_dir"]) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    started = time.monotonic()
    report = {**config, "tasks": {}}

    def run_task(task):
        task_dir = directory / task
        task_dir.mkdir()
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            task,
            str(directory),
        ]
        before = time.monotonic()
        with (task_dir / "worker.log").open("w") as log:
            try:
                process = subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=config["task_timeout"],
                )
                status = "completed" if process.returncode == 0 else "error"
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                status, return_code = "timeout", None
        result = summarize_task(task_dir)
        if status == "completed" and result["episodes_completed"] != config["episodes_per_task"]:
            status = "incomplete"
        return task, {
            **result,
            "status": status,
            "return_code": return_code,
            "seconds": round(time.monotonic() - before, 2),
        }

    print(f"Output: {directory}", flush=True)
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        jobs = [pool.submit(run_task, task) for task in config["tasks"]]
        for job in as_completed(jobs):
            task, result = job.result()
            report["tasks"][task] = result
            report["elapsed_seconds"] = round(time.monotonic() - started, 2)
            save_summary(directory, report)
            print(
                f"{task}: {result['successes']}/{result['episodes_completed']} "
                f"({result['status']})",
                flush=True,
            )
    successes = sum(result["successes"] for result in report["tasks"].values())
    requested = len(config["tasks"]) * config["episodes_per_task"]
    print(f"Total: {successes}/{requested} requested episodes succeeded.")
    return int(any(result["status"] != "completed" for result in report["tasks"].values()))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2], Path(sys.argv[3]))
    else:
        try:
            raise SystemExit(main())
        except ValueError as error:
            raise SystemExit(str(error)) from None
