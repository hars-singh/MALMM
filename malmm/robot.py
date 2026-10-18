"""RLBench scene feedback and the manipulation primitives exposed to the coder."""

import math
import random
from functools import wraps
from itertools import product

import numpy as np

from .tasks import COLORS, TASK_LABELS, TASK_OBJECTS


def world_bounds(box, matrix):
    """Transform all local bounding-box corners into world coordinates."""
    box, matrix = np.asarray(box), np.asarray(matrix)
    corners = np.array(list(product(box[:2], box[2:4], box[4:6])))
    corners = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return corners.min(axis=0), corners.max(axis=0)


class ActionBudgetExceeded(RuntimeError):
    pass


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
        self.abort_reason = None
        self.action_count = 0
        self.max_actions = 60
        self._random = random.Random()

    def reset(self, seed: int) -> str:
        self.completion_requested = False
        self.abort_reason = None
        self.action_count = 0
        self._random.seed(seed)
        np.random.seed(seed)
        self.task.set_variation(self._random.randrange(self.task.variation_count()))
        descriptions, _ = self.task.reset()
        return descriptions[0]

    def _scene_objects(self):
        return {obj.get_name(): obj for obj in self.simulator.get_objects_in_tree()}

    def _grasp_candidates(self):
        return [
            obj
            for obj in self.task._task.get_graspable_objects()
            if self.gripper._proximity_sensor.is_detected(obj)
        ]

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
                lower, upper = world_bounds(box, target.get_matrix())
                lines.extend(
                    [
                        f"Height (world z): {upper[2] - lower[2]}",
                        f"Width (world y): {upper[1] - lower[1]}",
                        f"Length (world x): {upper[0] - lower[0]}",
                        f"World bounding-box minimum: {lower}",
                        f"World bounding-box maximum: {upper}",
                        f"Top surface z: {upper[2]}",
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
            bottom = min(
                world_bounds(obj.get_bounding_box(), obj.get_matrix())[0][2] for obj in grasped
            )
            lines.append(f"Held-object bottom offset below end effector: {position[2] - bottom}")
        else:
            lines.append("No object has been grasped")
        lines.append(
            "Objects in grasp sensor: "
            f"{[labels.get(obj.get_name(), obj.get_name()) for obj in self._grasp_candidates()]}"
        )
        lines.append(f"Robot actions used: {self.action_count}/{self.max_actions}")
        return "\n".join(lines) + "\n"

    def detect_object(self) -> None:
        """Keep the scene-feedback function available to generated robot code."""
        print(self.describe_scene(), end="")

    def get_end_effector_pose(self):
        tip = self.arm.get_tip()
        return tip.get_position(), tip.get_orientation()

    def execute_trajectory(
        self, position: list[float], orientation: float, *, interaction: bool = False
    ) -> None:
        """Move with world yaw; allow contact for grasp descent and initial lift."""
        from pyrep.const import ObjectType

        shapes = self.simulator.get_objects_in_tree(object_type=ObjectType.SHAPE)
        # EmptyContainer's sampling volume is not a physical obstacle.
        contacts = [
            shape
            for shape in shapes
            if self.task_name == "empty_container"
            and shape.get_name() == "spawn_boundary"
            and not shape.is_respondable()
            and shape.is_collidable()
        ]
        # Match RLBench's handling of contacts at the starting pose.
        if self.arm.check_arm_collision():
            robot_shapes = self.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)
            held_objects = self.gripper.get_grasped_objects()
            contacts.extend(
                shape
                for shape in shapes
                if shape not in robot_shapes
                and shape not in held_objects
                and shape not in contacts
                and shape.is_collidable()
                and self.arm.check_arm_collision(shape)
            )
        # RLBench reference waypoints permit contact during grasp and lift phases.
        try:
            for shape in contacts:
                shape.set_collidable(False)
            path = self.arm.get_path(
                position=position,
                euler=[math.pi, 0, orientation],
                ignore_collisions=interaction,
            )
        finally:
            for shape in contacts:
                shape.set_collidable(True)
        done = False
        while not done:
            done = path.step()
            self.task._scene.step()
        actual_position, _ = self.get_end_effector_pose()
        error = float(np.linalg.norm(np.asarray(actual_position) - np.asarray(position)))
        if error > 0.015:
            if interaction and not self.gripper.get_grasped_objects():
                candidates = self._grasp_candidates()
                if candidates:
                    print(
                        f"Contact stopped the approach {error:.3f} m short; "
                        f"grasp sensor detects {[obj.get_name() for obj in candidates]}. "
                        "Verify the requested object by closing the gripper."
                    )
                    return
            raise RuntimeError(
                f"Motion stopped {error:.3f} m from the requested position; "
                "check the measured pose and change the approach or grasp yaw."
            )

    def _actuate_gripper(self, amount: float) -> None:
        for _ in range(2000):
            if self.gripper.actuate(amount, velocity=0.05):
                return
            self.task._scene.step()
        raise RuntimeError("Gripper did not reach its target position.")

    def close_gripper(self, object_name: str) -> None:
        mapping = TASK_OBJECTS[self.task_name]
        if object_name not in mapping:
            raise ValueError(f"Unknown object {object_name!r}; choose from {list(mapping)}")
        target = self._scene_objects().get(mapping[object_name])
        if target is None:
            raise ValueError(f"Object {object_name!r} is missing from this RLBench scene.")
        self._actuate_gripper(0.0)
        if not self.gripper.grasp(target):
            raise RuntimeError(f"Grasp failed: {object_name} is outside the gripper sensor.")

    def open_gripper(self) -> None:
        self.gripper.release()
        self._actuate_gripper(1.0)
        # Allow dropped objects to settle before checking tray detections or success.
        for _ in range(10):
            self.task._scene.step()

    def is_successful(self) -> bool:
        success, _ = self.task._task.success()
        return bool(success) and (
            self.task_name != "empty_container" or not self.gripper.get_grasped_objects()
        )

    def task_completed(self) -> None:
        if not self.is_successful():
            print("Task is not completed. Reassess the scene and continue the remaining actions.")
            return
        self.completion_requested = True
        print("Task is done successfully.")
        print("Taskexec")

    def abort_task(self, reason: str) -> None:
        self.abort_reason = reason
        print(f"Task aborted: {reason}")
        print("TaskAborted")

    def _counted_action(self, function):
        @wraps(function)
        def run(*args, **kwargs):
            if self.action_count >= self.max_actions:
                self.abort_task(f"Robot action limit reached ({self.max_actions}).")
                raise ActionBudgetExceeded(self.abort_reason)
            self.action_count += 1
            return function(*args, **kwargs)

        return run

    def action_functions(self):
        return {
            name: self._counted_action(getattr(self, name))
            if name in {"execute_trajectory", "open_gripper", "close_gripper"}
            else getattr(self, name)
            for name in (
                "execute_trajectory",
                "open_gripper",
                "close_gripper",
                "task_completed",
                "abort_task",
                "detect_object",
            )
        }

    def close(self) -> None:
        self.environment.shutdown()
