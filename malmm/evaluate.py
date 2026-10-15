"""Evaluate independent RLBench tasks in separate simulator processes."""

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

from .cli import positive_int
from .tasks import EVALUATION_TASKS, TASK_NAMES, TASK_OBJECTS


def check_task(task_name: str, episodes: int, destination: str) -> None:
    """Check scene loading, seeded resets, feedback, and shutdown without API calls."""
    import random

    import numpy as np

    from .robot import RLBenchRobot

    random.seed(0)
    np.random.seed(0)
    robot = RLBenchRobot(task_name, headless=True)
    samples = []
    try:
        for seed in range(episodes):
            random.seed(seed)
            np.random.seed(seed)
            instruction = robot.reset(seed)
            objects = robot._scene_objects()
            expected = set(TASK_OBJECTS[task_name].values())
            missing = sorted(expected - objects.keys())
            if expected and len(missing) == len(expected):
                raise RuntimeError("All mapped objects are missing from the task scene.")
            samples.append(
                {
                    "seed": seed,
                    "instruction": instruction,
                    "missing_objects": missing,
                    "scene": robot.describe_scene(),
                    "initial_success": robot.is_successful(),
                }
            )
    finally:
        robot.close()
    Path(destination).write_text(
        json.dumps({"task": task_name, "samples": samples}, indent=2) + "\n"
    )


def summarize_task(task_dir: Path) -> dict:
    """Keep infrastructure errors separate from completed unsuccessful episodes."""
    results = []
    for path in sorted(task_dir.glob("*/episode_*.json")):
        results.append(json.loads(path.read_text())["result"])
    completed = [result for result in results if "error_type" not in result]
    return {
        "episodes_recorded": len(results),
        "episodes_completed": len(completed),
        "successes": sum(result.get("success", False) for result in completed),
        "episode_errors": [result for result in results if "error_type" in result],
        "success_rate": (
            sum(result.get("success", False) for result in completed) / len(completed)
            if completed
            else None
        ),
    }


def save_summary(directory: Path, report: dict) -> None:
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
                    "completed": result.get("episodes_completed"),
                    "successes": result.get("successes"),
                    "success_rate": result.get("success_rate"),
                    "status": result["status"],
                    "seconds": result["seconds"],
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(EVALUATION_TASKS))
    parser.add_argument("--episodes", type=positive_int, default=2)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--model", default="openai/gpt-6-luna")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-rounds", type=positive_int, default=100)
    parser.add_argument(
        "--timeout", type=positive_int, default=1800, help="Seconds per task process"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluations"))
    parser.add_argument(
        "--check-only", action="store_true", help="Check simulators without model calls"
    )
    args = parser.parse_args(argv)
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Each task may appear only once.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = (args.output_dir / timestamp).resolve()
    directory.mkdir(parents=True)
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    started = time.monotonic()
    report = {
        "model": None if args.check_only else args.model,
        "check_only": args.check_only,
        "workers": args.workers,
        "episodes_per_task": args.episodes,
        "tasks_requested": args.tasks,
        "tasks": {},
    }

    def run_task(task_name):
        task_dir = directory / task_name
        task_dir.mkdir()
        if args.check_only:
            command = [
                sys.executable,
                "-c",
                (
                    "from malmm.evaluate import check_task; import sys; "
                    "check_task(sys.argv[1], int(sys.argv[2]), sys.argv[3])"
                ),
                task_name,
                str(args.episodes),
                str(task_dir / "preflight.json"),
            ]
        else:
            command = [
                sys.executable,
                "-u",
                "-m",
                "malmm",
                "--task",
                task_name,
                "--episodes",
                str(args.episodes),
                "--model",
                args.model,
                "--env-file",
                str(args.env_file.resolve()),
                "--max-rounds",
                str(args.max_rounds),
                "--headless",
                "--output-dir",
                str(directory),
            ]
        task_started = time.monotonic()
        with (task_dir / "worker.log").open("w") as log:
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                return_code = process.wait(timeout=args.timeout)
                status = (
                    ("passed" if args.check_only else "completed") if return_code == 0 else "error"
                )
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
                status = "timeout"
        result = {
            "status": status,
            "return_code": return_code,
            "seconds": round(time.monotonic() - task_started, 2),
            **({} if args.check_only else summarize_task(task_dir)),
        }
        if (
            not args.check_only
            and status == "completed"
            and result["episodes_completed"] != args.episodes
        ):
            result["status"] = "incomplete"
        return task_name, result

    print(f"Output: {directory}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_task, task) for task in args.tasks]
        for job in as_completed(jobs):
            task, result = job.result()
            report["tasks"][task] = result
            report["elapsed_seconds"] = round(time.monotonic() - started, 2)
            save_summary(directory, report)
            print(f"{task}: {result['status']} ({result['seconds']}s)", flush=True)
    return int(
        any(result["status"] not in {"passed", "completed"} for result in report["tasks"].values())
    )


if __name__ == "__main__":
    raise SystemExit(main())
