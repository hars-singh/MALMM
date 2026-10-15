# MALMM

**Multi-Agent Large Language Models for Zero-Shot Robotic Manipulation**

[Paper](https://arxiv.org/abs/2411.17636) · [Project](https://malmm1.github.io/)

MALMM uses a planner to break a manipulation instruction into steps, a coder to generate Python robot actions, and a supervisor to coordinate the agents. The executor runs each action in RLBench and returns object positions, orientations, dimensions, and gripper state for feedback and replanning.

## Setup

Use Python 3.10 or 3.11. Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

In this environment, install CoppeliaSim, PyRep, and RLBench following the [RLBench installation instructions](https://github.com/stepjam/RLBench#install). Then install MALMM:

```bash
python -m pip install -e .
cp .env.example .env
```

Install this package in the environment where PyRep and RLBench are available. The agent implementation uses the [AutoGen 0.2 API](https://microsoft.github.io/autogen/0.2/docs/reference/agentchat/conversable_agent/), pinned in `pyproject.toml`.

Set the model name, endpoint, and API key in `.env`. The example connects to a local OpenAI-compatible server on port 8000. For a hosted provider, supply its model and key; omit `MALMM_BASE_URL` to use the client's default endpoint. Existing shell environment variables take precedence over `.env`.

## Run

```bash
python -m malmm --list-tasks
python -m malmm --task put_rubbish_in_bin --episodes 1
python -m malmm --task close_jar --episodes 25 --seed 0 --headless
```

The `malmm` command is equivalent to `python -m malmm`. Use `--model`, `--base-url`, `--max-rounds`, and `--output-dir` to override run settings. Headless simulation still requires a working graphics/display setup; see the RLBench instructions.

The task list contains 40 saved scene-object mappings. Tasks and scene names must match the installed RLBench version; these mappings are not a claim that every task has been validated. Runs save conversation histories and episode outcomes under `outputs/<task>/<run>/`. Success comes from RLBench's task success condition, independently of the agent's completion message.

Generated Python runs in the local process with the robot API, as in the original method. It is not sandboxed.

## Code

| Path | Purpose |
| --- | --- |
| `malmm/agents.py` | Planner, coder, executor agent, and supervisor routing |
| `malmm/execution.py` | Python execution and scene feedback |
| `malmm/environment.py` | RLBench lifecycle and manipulation primitives |
| `malmm/prompts/` | Original planner, coder, and task prompts |
| `malmm/task_objects.json` | Original object mappings, labels, and colors |
| `malmm/cli.py` | Model configuration, episodes, and results |

The method is extracted from the saved `Desktop/main_code` project, using its March 5, 2025 multi-agent entry point. `SOURCE.json` records source file hashes. The original prompts and task mappings are preserved. The refactor adds explicit simulator startup/shutdown, configurable tasks and endpoints, independent episode histories, and reliable execution-state feedback. Simulation, models, datasets, experiment outputs, baselines, and training code are kept outside this repository.

## Checks

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

The checks exercise agent routing, robot-code execution, error feedback, prompt formatting, and configuration without a model server or simulator. Full RLBench runs require the external simulator setup.
