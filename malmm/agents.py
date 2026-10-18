"""Planner, coder, executor, and group-chat supervisor."""

from dataclasses import dataclass

import autogen
from autogen.agentchat.utils import gather_usage_summary

from .executor import execute_robot_code
from .prompts import CODER_PROMPT, PLANNER_PROMPT


def episode_finished(robot) -> bool:
    return robot.completion_requested or robot.abort_reason is not None


class ExecutorAgent(autogen.ConversableAgent):
    def __init__(self, robot):
        super().__init__(
            name="Executor",
            system_message="Execute the coder's Python and report the current scene.",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config={"last_n_messages": 1, "use_docker": False},
            default_auto_reply="Please continue. If everything is done, execute task_completed().",
            is_termination_msg=lambda _: episode_finished(robot),
        )
        self.robot = robot

    def run_code(self, code, **kwargs):
        return execute_robot_code(code, self.robot)

    def execute_code_blocks(self, code_blocks):
        logs = []
        for language, code in code_blocks:
            if language.lower() not in {"python", "python3", "py", ""}:
                return 1, "Use a Python code block for robot actions."
            exit_code, output, _ = self.run_code(code)
            logs.append(output)
            if exit_code or episode_finished(self.robot):
                return exit_code, "\n".join(logs)
        return 0, "\n".join(logs)


@dataclass
class AgentTeam:
    coordinator: autogen.ConversableAgent
    planner: autogen.ConversableAgent
    coder: autogen.AssistantAgent
    executor: ExecutorAgent
    supervisor: autogen.GroupChatManager
    conversation: autogen.GroupChat

    def run(self, prompt: str):
        return self.coordinator.initiate_chat(self.supervisor, message=prompt)

    def token_usage(self) -> dict:
        usage = gather_usage_summary(
            [self.coordinator, self.planner, self.coder, self.executor, self.supervisor]
        )["usage_excluding_cached_inference"]
        return {
            model: {
                name: values.get(name, 0)
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            for model, values in usage.items()
            if model != "total_cost"
        }


def create_agent_team(robot, llm_config: dict | bool, max_rounds: int = 300) -> AgentTeam:
    common = dict(
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda _: episode_finished(robot),
    )
    coordinator = autogen.ConversableAgent(
        name="Coordinator",
        system_message="Give the task and instruct the planner to execute it using the robot end effector.",
        code_execution_config=False,
        **common,
    )
    planner = autogen.ConversableAgent(
        name="Planner",
        system_message=PLANNER_PROMPT,
        description="Plan the task, check progress after execution, and replan remaining steps.",
        code_execution_config=False,
        **common,
    )
    coder = autogen.AssistantAgent(
        name="Coder",
        system_message=CODER_PROMPT,
        description="Generate Python robot actions one step at a time from the planner's instructions.",
        **common,
    )
    executor = ExecutorAgent(robot)
    conversation = autogen.GroupChat(
        agents=[coordinator, coder, executor, planner],
        messages=[],
        max_round=max_rounds,
        allowed_or_disallowed_speaker_transitions={
            coordinator: [planner],
            planner: [coder],
            coder: [executor],
            executor: [coder, planner],
        },
        speaker_transitions_type="allowed",
    )
    supervisor = autogen.GroupChatManager(
        name="Supervisor",
        groupchat=conversation,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda _: episode_finished(robot),
    )
    return AgentTeam(coordinator, planner, coder, executor, supervisor, conversation)
