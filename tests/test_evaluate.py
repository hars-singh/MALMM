import csv
import json

from malmm.evaluate import save_summary, summarize_task


def test_summary_distinguishes_task_failures_from_api_errors(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    episodes = [
        {"success": True},
        {"success": False},
        {"success": False, "error_type": "RateLimitError"},
    ]
    for number, result in enumerate(episodes):
        (run_dir / f"episode_{number:03d}.json").write_text(json.dumps({"result": result}))
    summary = summarize_task(tmp_path)
    assert summary["episodes_recorded"] == 3
    assert summary["episodes_completed"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["episode_errors"] == [episodes[-1]]


def test_partial_evaluation_exposes_missing_episodes_in_csv(tmp_path):
    result = {
        "episodes_completed": 1,
        "successes": 1,
        "success_rate": 1.0,
        "status": "incomplete",
        "seconds": 10,
    }
    report = {"episodes_per_task": 2, "tasks": {"close_jar": result}}
    save_summary(tmp_path, report)
    assert json.loads((tmp_path / "summary.json").read_text()) == report
    with (tmp_path / "summary.csv").open() as source:
        row = next(csv.DictReader(source))
    assert row["requested"] == "2" and row["completed"] == "1"
    assert row["status"] == "incomplete"
