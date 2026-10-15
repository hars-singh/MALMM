"""Planner, coder, executor, and group-chat supervisor."""

from dataclasses import dataclass

import autogen

from .execution import RobotCodeExecutor
from .prompts import ENGINEER_SYSTEM_MESSAGE, PLANNER_SYSTEM_MESSAGE


def terminated(message):
    return "taskexec" in (message.get("content") or "").lower()


class ExecutorAgent(autogen.ConversableAgent):
    def __init__(self, robot):
        super().__init__(
            name="TestAgent",
            system_message="Execute the coder's Python and report the current scene.",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config={"last_n_messages": 1, "use_docker": False},
            default_auto_reply="Please continue. If everything is done, execute task_completed().",
            is_termination_msg=terminated,
        )
        self.robot_executor = RobotCodeExecutor(robot)

    def run_code(self, code, **kwargs):
        return self.robot_executor.execute(code)

    def execute_code_blocks(self, code_blocks):
        logs = []
        for language, code in code_blocks:
            if language.lower() not in {"python", "python3", "py", ""}:
                return 1, "Use a Python code block for robot actions."
            exit_code, output, _ = self.run_code(code)
            logs.append(output)
            if exit_code or "taskexec" in output.lower():
                return exit_code, "\n".join(logs)
        return 0, "\n".join(logs)


@dataclass
class AgentTeam:
    admin: object
    planner: object
    coder: object
    executor: object
    supervisor: object
    groupchat: object

    def run(self, prompt):
        return self.admin.initiate_chat(self.supervisor, message=prompt)


def build_team(robot, llm_config, max_rounds=100):
    common = dict(llm_config=llm_config, human_input_mode="NEVER", is_termination_msg=terminated)
    admin = autogen.ConversableAgent(
        name="Admin",
        system_message="Give the task and instruct the planner to execute it using the robot end effector.",
        code_execution_config=False,
        **common,
    )
    planner = autogen.ConversableAgent(
        name="Planner",
        system_message=PLANNER_SYSTEM_MESSAGE,
        description="Plan the task, check progress after execution, and replan remaining steps.",
        code_execution_config=False,
        **common,
    )
    coder = autogen.AssistantAgent(
        name="Engineer",
        system_message=ENGINEER_SYSTEM_MESSAGE,
        description="Generate Python robot actions one step at a time from the planner's instructions.",
        **common,
    )
    executor = ExecutorAgent(robot)
    groupchat = autogen.GroupChat(
        agents=[admin, coder, executor, planner],
        messages=[],
        max_round=max_rounds,
        allowed_or_disallowed_speaker_transitions={
            admin: [planner],
            planner: [coder],
            coder: [executor],
            executor: [coder, planner],
        },
        speaker_transitions_type="allowed",
    )
    supervisor = autogen.GroupChatManager(
        name="Supervisor", groupchat=groupchat, llm_config=llm_config,
        human_input_mode="NEVER", is_termination_msg=terminated,
    )
    return AgentTeam(admin, planner, coder, executor, supervisor, groupchat)
