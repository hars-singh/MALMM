"""RLBench scene feedback and the manipulation primitives exposed to the coder."""

import math
import random

from .tasks import COLORS, TASK_LABELS, TASK_OBJECTS


class RLBenchRobot:
    def __init__(self, task_name, headless=False):
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
        self.env = Environment(
            MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
            obs_config=observation,
            headless=headless,
        )
        try:
            self.env.launch()
            self.task_env = self.env.get_task(task_class)
            # RLBench owns the simulator and robot handles throughout the run.
            self.pr = self.env._pyrep
            self.arm = self.env._robot.arm
            self.gripper = self.env._robot.gripper
        except Exception:
            self.env.shutdown()
            raise
        self.task_name = task_name
        self.finished = False
        self.rng = random.Random()

    def reset(self, seed):
        self.finished = False
        self.rng.seed(seed)
        self.task_env.set_variation(self.rng.randrange(self.task_env.variation_count()))
        descriptions, _ = self.task_env.reset()
        return descriptions[0]

    def objects(self):
        return {obj.get_name(): obj for obj in self.pr.get_objects_in_tree()}

    def detect_object(self):
        scene = self.objects()
        labels = TASK_LABELS[self.task_name]
        for name in TASK_OBJECTS[self.task_name].values():
            print(f"**{labels.get(name, name)}**:")
            target = scene.get(name)
            if target is None:
                print("No position or orientation information available.")
                continue
            print("Center Position:", target.get_position())
            print("Orientation:", target.get_orientation())
            try:
                box = target.get_bounding_box()
                # Preserve the scene description convention in the method source.
                print("Height:", 2 * box[5])
                print("Width:", 2 * box[3])
                print("Length:", 2 * box[1])
            except (AttributeError, IndexError):
                print("Bounding box information not available.")
            try:
                print("Color:", COLORS.get(tuple(target.get_color()), "unknown"))
            except AttributeError:
                print("Color information not available.")
        print("**End-effector**:")
        print("robot arm end-effector Position:", self.arm.get_tip().get_position())
        print("robot arm end-effector Orientation:", self.arm.get_tip().get_orientation())
        grasped = self.gripper.get_grasped_objects()
        if grasped:
            print("Grasped Object:", [labels.get(obj.get_name(), obj.get_name()) for obj in grasped])
        else:
            print("No object has been grasped")

    def execute_trajectory(self, position, orientation):
        # Original top-down controller rotates only around the z axis.
        path = self.arm.get_path(position=position, euler=[3.14, 0, -orientation])
        done = False
        while not done:
            done = path.step()
            self.pr.step()

    def _actuate(self, amount):
        for _ in range(2000):
            if self.gripper.actuate(amount, velocity=0.05):
                return
            self.pr.step()
        raise RuntimeError("Gripper did not reach its target position.")

    def close_gripper(self, object_name):
        mapping = TASK_OBJECTS[self.task_name]
        if object_name not in mapping:
            raise ValueError(f"Unknown object {object_name!r}; choose from {list(mapping)}")
        target = self.objects().get(mapping[object_name])
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
        self._actuate(amount)
        self.gripper.grasp(target)

    def open_gripper(self):
        self.gripper.release()
        self._actuate(1.0)

    def success(self):
        success, _ = self.task_env._task.success()
        return bool(success)

    def task_completed(self):
        self.finished = True
        print("Task is done successfully." if self.success() else "Task is not completed.")
        print("Taskexec")

    def functions(self):
        return {name: getattr(self, name) for name in (
            "execute_trajectory", "open_gripper", "close_gripper", "task_completed", "detect_object"
        )}

    def close(self):
        self.env.shutdown()
