import json

import pytest

from malmm.cli import create_parser, load_model_config, main
from malmm.executor import execute_robot_code
from malmm.prompts import format_task_prompt
from malmm.tasks import TASK_NAMES, TASK_OBJECTS


class FakeRobot:
    def __init__(self):
        self.actions = []
        self.position = [0, 0, 0]
        self.completion_requested = False
        self.abort_reason = None
        self.action_count = 0

    def is_successful(self):
        return self.completion_requested

    def execute_trajectory(self, position, orientation):
        self.position = position
        self.actions.append((position, orientation))

    def describe_scene(self):
        return f"position={self.position}"

    def detect_object(self):
        print(self.describe_scene())

    def get_end_effector_pose(self):
        return self.position, [0, 0, 0]

    def task_completed(self):
        self.completion_requested = True
        print("Taskexec")

    def action_functions(self):
        return {
            name: getattr(self, name)
            for name in ("execute_trajectory", "detect_object", "task_completed")
        }


def test_multiline_code_keeps_one_namespace_and_reports_updated_state():
    robot = FakeRobot()
    code = (
        "position = [0.1, 0.2, 0.8]\n\ndef move():\n    execute_trajectory(position, 0.5)\n\nmove()"
    )
    status, output, _ = execute_robot_code(code, robot)
    assert status == 0
    assert robot.actions == [([0.1, 0.2, 0.8], 0.5)]
    assert "position=[0.1, 0.2, 0.8]" in output


def test_partial_action_failure_returns_error_and_current_state():
    robot = FakeRobot()
    status, output, _ = execute_robot_code(
        "execute_trajectory([1, 2, 3], 0)\nraise ValueError('retry the next step')", robot
    )
    assert status == 1
    assert "ValueError: retry the next step" in output
    assert "position=[1, 2, 3]" in output


def test_supervisor_routing_and_autogen_execution_path():
    from malmm.agents import create_agent_team

    robot = FakeRobot()
    team = create_agent_team(robot, llm_config=False)
    assert team.token_usage() == {}
    transitions = team.conversation.allowed_speaker_transitions_dict
    assert transitions[team.coordinator] == [team.planner]
    assert transitions[team.planner] == [team.coder]
    assert transitions[team.coder] == [team.executor]
    assert transitions[team.executor] == [team.coder, team.planner]
    handled, response = team.executor.generate_code_execution_reply(
        messages=[{"role": "user", "content": "```python\nexecute_trajectory([1, 2, 3], 0)\n```"}]
    )
    assert handled
    assert "exitcode: 0" in response
    assert "position=[1, 2, 3]" in response
    assert robot.actions == [([1, 2, 3], 0)]


def test_completion_stops_remaining_blocks():
    from malmm.agents import ExecutorAgent

    robot = FakeRobot()
    status, output = ExecutorAgent(robot).execute_code_blocks(
        [
            ("python", "task_completed()"),
            ("python", "execute_trajectory([1, 2, 3], 0)"),
        ]
    )
    assert status == 0 and "Taskexec" in output
    assert robot.completion_requested and not robot.actions


def test_executor_detects_success_without_model_completion_request():
    robot = FakeRobot()
    robot.is_successful = lambda: True
    status, output, _ = execute_robot_code("execute_trajectory([1, 2, 3], 0)", robot)
    assert status == 0 and "Taskexec" in output
    assert robot.completion_requested


def test_prompt_contains_task_initial_pose_and_scene():
    prompt = format_task_prompt("close the jar", [1, 2, 3], [0, 0, 0], "jar lid at [2, 3, 4]")
    assert "close the jar" in prompt and "jar lid at [2, 3, 4]" in prompt
    assert "[INSERT" not in prompt and "[STATE]" not in prompt


def test_task_listing_needs_no_simulator_or_credentials(capsys):
    assert main(["--list-tasks"]) == 0
    assert capsys.readouterr().out.splitlines() == list(TASK_NAMES)
    assert len(TASK_OBJECTS) == 40


def test_config_cli_overrides_env_file(tmp_path, monkeypatch):
    for name in ["MALMM_API_KEY", "OPENAI_API_KEY", "MALMM_MODEL", "MALMM_BASE_URL"]:
        monkeypatch.delenv(name, raising=False)
    config_file = tmp_path / ".env"
    config_file.write_text("MALMM_API_KEY=example-test-key\nMALMM_MODEL=file-model\n")
    args = create_parser().parse_args(
        [
            "--env-file",
            str(config_file),
            "--model",
            "cli-model",
            "--base-url",
            "http://localhost:8000/v1",
        ]
    )
    config = load_model_config(args)["config_list"][0]
    assert config["model"] == "cli-model"
    assert config["base_url"] == "http://localhost:8000/v1"
    assert config["api_key"] == "example-test-key"


def test_episode_count_must_be_positive():
    with pytest.raises(SystemExit):
        create_parser().parse_args(["--episodes", "0"])


@pytest.mark.parametrize("fail_second_episode", [False, True])
def test_episode_runner_saves_results_and_closes_simulator(
    tmp_path, monkeypatch, fail_second_episode
):
    from types import SimpleNamespace

    import malmm.runner

    robots = []
    teams = []

    class FakeEnvironment(FakeRobot):
        def __init__(self, task_name, headless):
            super().__init__()
            self.closed = False
            self.seeds = []
            robots.append(self)

        def reset(self, seed):
            self.completion_requested = False
            self.seeds.append(seed)
            return "put the rubbish in the bin"

        def is_successful(self):
            return self.completion_requested

        def close(self):
            self.closed = True

    def fake_team(robot, config, rounds):
        messages = []

        def run(prompt):
            assert not robot.completion_requested
            messages.append({"role": "user", "content": prompt})
            if fail_second_episode and len(teams) == 2:
                raise RuntimeError("model unavailable")
            robot.task_completed()

        team = SimpleNamespace(
            conversation=SimpleNamespace(messages=messages), run=run, token_usage=lambda: {}
        )
        teams.append(team)
        return team

    monkeypatch.setattr(malmm.runner, "RLBenchRobot", FakeEnvironment)
    monkeypatch.setattr(malmm.runner, "create_agent_team", fake_team)
    monkeypatch.setenv("MALMM_API_KEY", "example-private-test-key")
    arguments = [
        "--episodes",
        "2",
        "--seed",
        "8",
        "--headless",
        "--env-file",
        str(tmp_path / "missing.env"),
        "--output-dir",
        str(tmp_path / "results"),
    ]
    if fail_second_episode:
        with pytest.raises(RuntimeError, match="model unavailable"):
            main(arguments)
    else:
        assert main(arguments) == 0
    assert robots[0].closed and robots[0].seeds == [8, 9]
    assert len(teams) == 2 and teams[0] is not teams[1]
    result_file = next(tmp_path.rglob("results.json"))
    results = json.loads(result_file.read_text())
    assert [r["success"] for r in results["episodes"]] == [True, not fail_second_episode]
    assert len(list(tmp_path.rglob("episode_*.json"))) == 2
    for path in tmp_path.rglob("*.json"):
        assert "example-private-test-key" not in path.read_text()
