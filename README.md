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
python -m pip install -e '.[simulation]'
cp .env.example .env
```

The agent implementation uses the [AutoGen 0.2 API](https://microsoft.github.io/autogen/0.2/docs/reference/agentchat/conversable_agent/), pinned in `pyproject.toml`.

Set the model name, endpoint, and API key in `.env`. The example connects to a local OpenAI-compatible server on port 8000. For a hosted provider, supply its model and key; omit `MALMM_BASE_URL` to use the client's default endpoint. Existing shell environment variables take precedence over `.env`.

For OpenRouter, set `MALMM_BASE_URL=https://openrouter.ai/api/v1` and use its full model ID, such as `openai/gpt-4-turbo`. Set `MALMM_API_KEY` to an OpenRouter key. A valid key can still require credits or be subject to provider rate limits.

Before launching the simulator, configure its location:

```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
```

## Run

```bash
python -m malmm --list-tasks
python -m malmm --task put_rubbish_in_bin --episodes 1
python -m malmm --task close_jar --episodes 25 --seed 0 --headless
```

The `malmm` command is equivalent to `python -m malmm`. Use `--model`, `--base-url`, `--max-rounds`, `--max-actions`, and `--output-dir` to override run settings. Each episode permits 60 attempted robot actions and has a separate 300-message conversation cap by default. Headless simulation still requires a working graphics/display setup; see the RLBench instructions.

The task list contains 40 saved scene-object mappings. Tasks and scene names must match the installed RLBench version; these mappings are not a claim that every task has been validated. Runs save conversation histories and episode outcomes under `outputs/<task>/<run>/`. Success comes from RLBench's task success condition, independently of the agent's completion message.

Generated Python runs in the local process with the robot API, as in the original method. It is not sandboxed.

### Parallel evaluation

Check all nine evaluation task scenes and two seeded resets per task with four simulator processes:

```bash
python -m malmm.evaluate --check-only --workers 4 --episodes 2
```

Run the same task set with a model:

```bash
python -m malmm.evaluate --workers 4 --episodes 2 --model openai/gpt-6-luna --env-file .env
```

Each task runs in a fresh process with its own output directory and a default 30-minute timeout. JSON and CSV summaries distinguish completed task failures from API errors, timeouts, and missing episodes. Per-episode results include token usage across the agents. The `slide_block_to_target` task is the saved code's mapping for the paper's “Put Block” column. Two trials per task form a pilot evaluation, not the paper's 25-trial protocol.

## Code

| Path | Purpose |
| --- | --- |
| `malmm/agents.py` | Planner, coder, executor agent, and supervisor routing |
| `malmm/executor.py` | Python execution and scene feedback |
| `malmm/robot.py` | RLBench lifecycle, scene descriptions, and manipulation primitives |
| `malmm/runner.py` | Independent episodes, conversation histories, and outcomes |
| `malmm/prompts/{planner,coder,task}.py` | Role prompts and initial task formatting |
| `malmm/task_objects.json` | Original object mappings, labels, and colors |
| `malmm/cli.py` | Command-line options and model configuration |

The method is extracted from the saved `Desktop/main_code` project, using its March 5, 2025 multi-agent entry point. `SOURCE.json` records the original source file hashes. Scene-object mappings are preserved, with the tray-label spelling corrected; planner and coder prompts have been updated to describe the corrected robot interface. The refactor adds explicit simulator startup/shutdown, configurable tasks and endpoints, independent episode histories, and execution-state feedback. These controller and prompt changes mean new evaluations are not an exact reproduction of the original implementation. Simulation, models, datasets, experiment outputs, baselines, and training code are kept outside this repository.

Positions, yaw, and bounding-box limits use world coordinates. The motion API passes yaw directly to the simulator. `execute_trajectory(position, yaw, interaction=True)` permits contact during grasp descent and initial lift, matching RLBench reference interaction waypoints. Transit and placement retain collision checking. Existing arm contacts and Empty Container's nonphysical sampling helper are temporarily excluded during planning; their collision flags are restored before execution, and contact physics remains enabled.

Gripping follows the reference controller's full-close command and reports an unsuccessful sensor grasp as an error. A motion that stops more than 1.5 cm short reports an error, except for a contact approach with a graspable object already detected by the gripper sensor. That case reports the shortfall and detected objects; closing still verifies the requested target. Feedback includes the held-object bottom offset. Destination selection and placement assessment use the instruction and scene geometry; policy observations do not include internal target identities, per-item success detectors, or a task-success Boolean. Release poses account for that offset; physics settles for ten steps after opening. All motion steps run RLBench scene/task updates. Completion requires simulator success; Empty Container also requires a released gripper. Unsuccessful completion requests continue the episode. `abort_task(reason)` records an unsuccessful stop separately, and action-budget exhaustion also aborts. The executor automatically requests completion after a successful action when the task condition is satisfied. Episode records include action counts, abort reasons, and stopping reasons.

## Checks

```bash
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
ruff format --check .
```

The checks exercise agent routing, robot-code execution, error feedback, prompt formatting, and configuration without a model server or simulator. Full RLBench runs require the external simulator setup.
