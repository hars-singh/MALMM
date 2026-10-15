import json

import pytest

from malmm.cli import llm_config, main, parser
from malmm.execution import RobotCodeExecutor
from malmm.prompts import task_prompt
from malmm.tasks import TASK_OBJECTS, task_names


class FakeRobot:
    def __init__(self):
        self.actions = []
        self.position = [0, 0, 0]
        self.finished = False

    def execute_trajectory(self, position, orientation):
        self.position = position
        self.actions.append((position, orientation))

    def detect_object(self):
        print(f"position={self.position}")

    def task_completed(self):
        self.finished = True
        print("Taskexec")

    def functions(self):
        return {name: getattr(self, name) for name in (
            "execute_trajectory", "detect_object", "task_completed"
        )}


def test_multiline_code_keeps_one_namespace_and_reports_updated_state():
    robot = FakeRobot()
    code = "position = [0.1, 0.2, 0.8]\n\ndef move():\n    execute_trajectory(position, 0.5)\n\nmove()"
    status, output, _ = RobotCodeExecutor(robot).execute(code)
    assert status == 0
    assert robot.actions == [([0.1, 0.2, 0.8], 0.5)]
    assert "position=[0.1, 0.2, 0.8]" in output


def test_partial_action_failure_returns_error_and_current_state():
    robot = FakeRobot()
    status, output, _ = RobotCodeExecutor(robot).execute(
        "execute_trajectory([1, 2, 3], 0)\nraise ValueError('retry the next step')"
    )
    assert status == 1
    assert "ValueError: retry the next step" in output
    assert "position=[1, 2, 3]" in output


def test_supervisor_routing_and_autogen_execution_path():
    from malmm.agents import build_team

    robot = FakeRobot()
    team = build_team(robot, llm_config=False)
    transitions = team.groupchat.allowed_speaker_transitions_dict
    assert transitions[team.admin] == [team.planner]
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
    status, output = ExecutorAgent(robot).execute_code_blocks([
        ("python", "task_completed()"),
        ("python", "execute_trajectory([1, 2, 3], 0)"),
    ])
    assert status == 0 and "Taskexec" in output
    assert robot.finished and not robot.actions


def test_prompt_contains_task_initial_pose_and_scene():
    prompt = task_prompt("close the jar", [1, 2, 3], [0, 0, 0], "jar lid at [2, 3, 4]")
    assert "close the jar" in prompt and "jar lid at [2, 3, 4]" in prompt
    assert "[INSERT" not in prompt and "[STATE]" not in prompt


def test_task_listing_needs_no_simulator_or_credentials(capsys):
    assert main(["--list-tasks"]) == 0
    assert capsys.readouterr().out.splitlines() == task_names()
    assert len(TASK_OBJECTS) == 40


def test_config_cli_overrides_env_file(tmp_path, monkeypatch):
    for name in ["MALMM_API_KEY", "OPENAI_API_KEY", "MALMM_MODEL", "MALMM_BASE_URL"]:
        monkeypatch.delenv(name, raising=False)
    # monkeypatch restores values loaded by dotenv when this test exits.
    for name in ["MALMM_API_KEY", "MALMM_MODEL", "MALMM_BASE_URL"]:
        monkeypatch.setenv(name, "")
    config_file = tmp_path / ".env"
    config_file.write_text("MALMM_API_KEY=example-test-key\nMALMM_MODEL=file-model\n")
    # Empty shell values deliberately take precedence, so remove them for this load.
    monkeypatch.delenv("MALMM_API_KEY")
    monkeypatch.delenv("MALMM_MODEL")
    args = parser().parse_args(["--env-file", str(config_file), "--model", "cli-model",
                                "--base-url", "http://localhost:8000/v1"])
    config = llm_config(args)["config_list"][0]
    assert config["model"] == "cli-model"
    assert config["base_url"] == "http://localhost:8000/v1"
    assert config["api_key"] == "example-test-key"


def test_episode_count_must_be_positive():
    with pytest.raises(SystemExit):
        parser().parse_args(["--episodes", "0"])


def test_episode_runner_resets_saves_results_and_closes_simulator(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import malmm.agents
    import malmm.environment

    robots = []
    teams = []

    class FakeEnvironment(FakeRobot):
        def __init__(self, task_name, headless):
            super().__init__()
            self.closed = False
            self.seeds = []
            self.arm = SimpleNamespace(get_tip=lambda: SimpleNamespace(
                get_position=lambda: self.position, get_orientation=lambda: [0, 0, 0]
            ))
            robots.append(self)

        def reset(self, seed):
            self.finished = False
            self.seeds.append(seed)
            return "put the rubbish in the bin"

        def success(self):
            return self.finished

        def close(self):
            self.closed = True

    def fake_team(robot, config, rounds):
        messages = []

        def run(prompt):
            assert not robot.finished
            messages.append({"role": "user", "content": prompt})
            robot.task_completed()

        team = SimpleNamespace(groupchat=SimpleNamespace(messages=messages), run=run)
        teams.append(team)
        return team

    monkeypatch.setattr(malmm.environment, "RLBenchRobot", FakeEnvironment)
    monkeypatch.setattr(malmm.agents, "build_team", fake_team)
    monkeypatch.setenv("MALMM_API_KEY", "example-private-test-key")
    assert main(["--episodes", "2", "--seed", "8", "--headless",
                 "--env-file", str(tmp_path / "missing.env"),
                 "--output-dir", str(tmp_path / "results")]) == 0
    assert robots[0].closed and robots[0].seeds == [8, 9]
    assert len(teams) == 2 and teams[0] is not teams[1]
    result_file = next(tmp_path.rglob("results.json"))
    results = json.loads(result_file.read_text())
    assert [r["success"] for r in results["episodes"]] == [True, True]
    assert len(list(tmp_path.rglob("episode_*.json"))) == 2
    for path in tmp_path.rglob("*.json"):
        assert "example-private-test-key" not in path.read_text()
