# MALMM

**Multi-Agent Large Language Models for Zero-Shot Robotics Manipulation**

[Paper](https://arxiv.org/abs/2411.17636) · [Project page](https://malmm1.github.io/)

MALMM coordinates a Planner, Coder, and Supervisor to solve manipulation tasks
in RLBench. The Planner reasons about the scene, the Coder generates robot actions,
and the Executor returns updated simulator observations for the next step.

This repository includes the prompts, simulator interface, nine task models, and
one command-line evaluation script. No model training is required.

## Installation

### 1. Create a Python environment

From the repository directory, using Python 3.10 or 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Install the simulator

Install **CoppeliaSim 4.1.0**, [PyRep](https://github.com/stepjam/PyRep), and
[RLBench](https://github.com/stepjam/RLBench#install) following their installation
instructions. Install the Python packages in the same environment created above.

Set the simulator paths before running evaluation:

```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
```

The `.ttm` task models are included in `malmm/assets/`. RLBench and PyRep are still
required to load them and run the robot. Headless execution also needs a working
graphics/display setup; see [RLBench's headless guide](https://github.com/stepjam/RLBench#running-headless).

### 3. Configure the model API

```bash
cp .env.example .env
```

Edit `.env` with your API key and an OpenAI-compatible endpoint. For example:

```dotenv
MALMM_API_KEY=your-key-here
MALMM_BASE_URL=https://openrouter.ai/api/v1
```

Select a model supported by that endpoint with `--model`. Shell environment
variables take precedence over `.env`; `--base-url` overrides `MALMM_BASE_URL`.
`OPENAI_API_KEY` is also accepted when `MALMM_API_KEY` is not set.

## Evaluation


**Evaluate one task:**

```bash
python evaluate.py \
  --tasks meat_off_grill \
  --episodes 5 \
  --seed 0 \
  --workers 1 \
  --model openai/gpt-6-luna
```

**Evaluate all nine tasks:**

```bash
python evaluate.py --tasks all --episodes 4 --workers 4 --model openai/gpt-6-luna
```

**Evaluate a subset:**

```bash
python evaluate.py --tasks close_jar stack_blocks --episodes 3 --seed 10 --workers 2
```

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--tasks NAME [NAME ...]` | `all` | One or more task names, or `all`. |
| `--episodes N` | `4` | Episodes per task. |
| `--seed N` | `0` | First episode seed; must be nonnegative. |
| `--workers N` | `4` | Maximum number of parallel task processes. |
| `--model ID` | `openai/gpt-6-luna` | Model identifier accepted by the provider. |
| `--base-url URL` | Environment / provider default | Override the model API endpoint. |
| `--env-file PATH` | Repository `.env` | Read provider settings from this file. |
| `--output-dir PATH` | Repository `outputs/evaluations` | Parent directory for timestamped runs. |
| `--max-actions N` | `60` | Robot actions allowed per episode. |
| `--max-messages N` | `300` | Group-chat messages allowed per episode. |
| `--task-timeout N` | `3600` | Seconds allowed for all episodes of one task. |
| `--headless` / `--no-headless` | Headless | Hide or display the simulator window. |

## Repository layout

```text
evaluate.py           Command-line evaluation and parallel task workers
requirements.txt      Python dependencies
.env.example          Example provider configuration
outputs/              Evaluation output directory; empty in the source release
malmm/
  agents.py           Agent definitions and Supervisor routing
  executor.py         Generated-code execution and feedback
  robot.py            Simulator observations and robot actions
  bounding_box.py     World bounding boxes
  runner.py           Episode execution and logging
  tasks.py            Configuration loading
  tasks.json          Task objects, color names, and instruction configuration
  prompts/            Planner, Coder, and initial Supervisor prompt
  assets/             Nine task models
```

## Citation

If you use MALMM in your research, please cite:

```bibtex
@misc{singh2024malmm,
  title={MALMM: Multi-Agent Large Language Models for Zero-Shot Robotics Manipulation},
  author={Harsh Singh and Rocktim Jyoti Das and Mingfei Han and Preslav Nakov and Ivan Laptev},
  year={2024},
  eprint={2411.17636},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2411.17636}
}
```
