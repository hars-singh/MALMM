"""Execute coder output with the live robot API and return state feedback."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


class RobotCodeExecutor:
    def __init__(self, robot):
        self.robot = robot

    def execute(self, code):
        namespace = dict(self.robot.functions())
        output = StringIO()
        errors = StringIO()
        exit_code = 0
        with redirect_stdout(output), redirect_stderr(errors):
            try:
                exec(compile(code, "<malmm-coder>", "exec"), namespace)
            except Exception as exc:
                exit_code = 1
                print(f"{type(exc).__name__}: {exc}", file=errors)
            # Return current state even after a failed action, for replanning.
            try:
                self.robot.detect_object()
            except Exception as exc:
                exit_code = 1
                print(f"State feedback error: {type(exc).__name__}: {exc}", file=errors)
        return exit_code, errors.getvalue() + output.getvalue(), None
