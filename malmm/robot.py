"""RLBench scene feedback and the manipulation primitives exposed to the coder."""

import math
import random

from .tasks import COLORS, TASK_LABELS, TASK_OBJECTS


class RLBenchRobot:
    def __init__(self, task_name: str, headless: bool = False):
        # Lazy imports keep task listing and prompt inspection independent of the simulator.
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.backend.utils import task_file_to_task_class
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig

        if task_name not in TASK_OBJECTS:
            raise ValueError(f"No object mapping for task {task_name!r}.")
        task_class = task_file_to_task_class(task_name)
        observation = ObservationConfig()
        observation.set_all(False)
        self.environment = Environment(
            MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
            obs_config=observation,
            headless=headless,
        )
        try:
            self.environment.launch()
            self.task = self.environment.get_task(task_class)
            # RLBench owns the simulator and robot handles throughout the run.
            self.simulator = self.environment._pyrep
            self.arm = self.environment._robot.arm
            self.gripper = self.environment._robot.gripper
        except Exception:
            self.environment.shutdown()
            raise
        self.task_name = task_name
        self.completion_requested = False
        self._random = random.Random()

    def reset(self, seed: int) -> str:
        self.completion_requested = False
        self._random.seed(seed)
        self.task.set_variation(self._random.randrange(self.task.variation_count()))
        descriptions, _ = self.task.reset()
        return descriptions[0]

    def _scene_objects(self):
        return {obj.get_name(): obj for obj in self.simulator.get_objects_in_tree()}

    def describe_scene(self) -> str:
        lines = []
        scene = self._scene_objects()
        labels = TASK_LABELS[self.task_name]
        for name in TASK_OBJECTS[self.task_name].values():
            lines.append(f"**{labels.get(name, name)}**:")
            target = scene.get(name)
            if target is None:
                lines.append("No position or orientation information available.")
                continue
            lines.extend(
                [
                    f"Center Position: {target.get_position()}",
                    f"Orientation: {target.get_orientation()}",
                ]
            )
            try:
                box = target.get_bounding_box()
                # Preserve the scene description convention in the method source.
                lines.extend(
                    [
                        f"Height: {2 * box[5]}",
                        f"Width: {2 * box[3]}",
                        f"Length: {2 * box[1]}",
                    ]
                )
            except (AttributeError, IndexError):
                lines.append("Bounding box information not available.")
            try:
                color = COLORS.get(tuple(target.get_color()), "unknown")
                lines.append(f"Color: {color}")
            except AttributeError:
                lines.append("Color information not available.")
        position, orientation = self.get_end_effector_pose()
        lines.extend(
            [
                "**End-effector**:",
                f"robot arm end-effector Position: {position}",
                f"robot arm end-effector Orientation: {orientation}",
            ]
        )
        grasped = self.gripper.get_grasped_objects()
        if grasped:
            names = [labels.get(obj.get_name(), obj.get_name()) for obj in grasped]
            lines.append(f"Grasped Object: {names}")
        else:
            lines.append("No object has been grasped")
        return "\n".join(lines) + "\n"

    def detect_object(self) -> None:
        """Keep the scene-feedback function available to generated robot code."""
        print(self.describe_scene(), end="")

    def get_end_effector_pose(self):
        tip = self.arm.get_tip()
        return tip.get_position(), tip.get_orientation()

    def execute_trajectory(self, position: list[float], orientation: float) -> None:
        # Original top-down controller rotates only around the z axis.
        path = self.arm.get_path(position=position, euler=[3.14, 0, -orientation])
        done = False
        while not done:
            done = path.step()
            self.simulator.step()

    def _actuate_gripper(self, amount: float) -> None:
        for _ in range(2000):
            if self.gripper.actuate(amount, velocity=0.05):
                return
            self.simulator.step()
        raise RuntimeError("Gripper did not reach its target position.")

    def close_gripper(self, object_name: str) -> None:
        mapping = TASK_OBJECTS[self.task_name]
        if object_name not in mapping:
            raise ValueError(f"Unknown object {object_name!r}; choose from {list(mapping)}")
        target = self._scene_objects().get(mapping[object_name])
        if target is None:
            raise ValueError(f"Object {object_name!r} is missing from this RLBench scene.")
        try:
            parts = self.gripper.get_objects_in_tree()
            left, right = parts[8].get_position(), parts[9].get_position()
            distance = sum(abs(float(a) - float(b)) for a, b in zip(left, right))
            box = target.get_bounding_box()
            amount = 2 * min(box[3], box[1]) / distance
            if not math.isfinite(amount):
                raise ValueError("Non-finite gripper opening")
            amount = min(1.0, max(0.0, amount))
        except (AttributeError, IndexError, ZeroDivisionError, ValueError):
            amount = 0.5
        self._actuate_gripper(amount)
        self.gripper.grasp(target)

    def open_gripper(self) -> None:
        self.gripper.release()
        self._actuate_gripper(1.0)

    def is_successful(self) -> bool:
        success, _ = self.task._task.success()
        return bool(success)

    def task_completed(self) -> None:
        self.completion_requested = True
        print("Task is done successfully." if self.is_successful() else "Task is not completed.")
        print("Taskexec")

    def action_functions(self):
        return {
            name: getattr(self, name)
            for name in (
                "execute_trajectory",
                "open_gripper",
                "close_gripper",
                "task_completed",
                "detect_object",
            )
        }

    def close(self) -> None:
        self.environment.shutdown()
