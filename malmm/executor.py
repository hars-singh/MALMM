"""Execute robot actions and collect feedback for replanning."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


def execute_robot_code(code: str, robot) -> tuple[int, str, None]:
    output = StringIO()
    exit_code = 0
    with redirect_stdout(output), redirect_stderr(output):
        try:
            exec(compile(code, "<malmm-coder>", "exec"), robot.action_functions())
        except Exception as error:
            exit_code = 1
            print(f"{type(error).__name__}: {error}")
        # Failed actions still need fresh scene feedback for the next plan.
        try:
            robot.detect_object()
        except Exception as error:
            exit_code = 1
            print(f"State feedback error: {type(error).__name__}: {error}")
    return exit_code, output.getvalue(), None
