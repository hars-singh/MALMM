import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from malmm.robot import RLBenchRobot, world_bounds
from malmm.tasks import TASK_OBJECTS


@pytest.mark.parametrize("task_name", ["empty_container", "stack_blocks"])
def test_scene_feedback_does_not_access_task_target_or_success_metadata(task_name):
    class PrivateTask:
        def __getattr__(self, name):
            raise AssertionError(f"Scene feedback accessed private task metadata: {name}")

    shape = SimpleNamespace(
        get_position=lambda: [0.2, 0.1, 0.8],
        get_orientation=lambda: [0, 0, 0],
        get_bounding_box=lambda: [-0.02, 0.02, -0.02, 0.02, -0.02, 0.02],
        get_matrix=lambda: [[1, 0, 0, 0.2], [0, 1, 0, 0.1], [0, 0, 1, 0.8]],
        get_color=lambda: [1, 0, 0],
    )
    robot = RLBenchRobot.__new__(RLBenchRobot)
    robot.task_name = task_name
    robot.task = SimpleNamespace(_task=PrivateTask())
    robot._scene_objects = lambda: dict.fromkeys(TASK_OBJECTS[task_name].values(), shape)
    robot.get_end_effector_pose = lambda: ([0.2, 0.1, 1.0], [math.pi, 0, 0])
    robot.gripper = SimpleNamespace(get_grasped_objects=lambda: [])
    robot._grasp_candidates = lambda: []
    robot.action_count, robot.max_actions = 0, 60

    feedback = robot.describe_scene()
    assert "Color: red" in feedback
    assert "World bounding-box minimum:" in feedback
    assert "No object has been grasped" in feedback
    assert "Required target tray:" not in feedback
    assert "Items detected in target tray:" not in feedback
    assert "Task success:" not in feedback


@pytest.mark.parametrize("planning_fails", [False, True])
@pytest.mark.parametrize("interaction", [False, True])
@pytest.mark.parametrize("tracking_error", [0.0, 0.04])
def test_initial_contacts_are_restored_before_motion_or_after_planning_failure(
    monkeypatch, planning_fails, interaction, tracking_error
):
    constants = ModuleType("pyrep.const")
    constants.ObjectType = SimpleNamespace(SHAPE=0)
    monkeypatch.setitem(sys.modules, "pyrep", ModuleType("pyrep"))
    monkeypatch.setitem(sys.modules, "pyrep.const", constants)

    class Shape:
        collidable = True

        def is_collidable(self):
            return self.collidable

        def set_collidable(self, value):
            self.collidable = value

    contact, obstacle, held, arm_shape = [Shape() for _ in range(4)]
    shapes = [contact, obstacle, held, arm_shape]
    steps = []

    def step():
        assert all(shape.collidable for shape in shapes)
        steps.append("motion")
        return True

    def get_path(**kwargs):
        assert not contact.collidable
        assert all(shape.collidable for shape in [obstacle, held, arm_shape])
        assert kwargs == {
            "position": [0.2, 0.1, 1.0],
            "euler": [math.pi, 0, 0.5],
            "ignore_collisions": interaction,
        }
        if planning_fails:
            raise RuntimeError("no path")
        return SimpleNamespace(step=step)

    robot = RLBenchRobot.__new__(RLBenchRobot)
    robot.task_name = "put_rubbish_in_bin"
    robot.get_end_effector_pose = lambda: ([0.2 + tracking_error, 0.1, 1.0], [math.pi, 0, 0.5])
    robot.arm = SimpleNamespace(
        check_arm_collision=lambda shape=None: shape is None or shape in [contact, held, arm_shape],
        get_objects_in_tree=lambda **kwargs: [arm_shape],
        get_path=get_path,
    )
    robot.gripper = SimpleNamespace(get_grasped_objects=lambda: [held])
    robot.simulator = SimpleNamespace(
        get_objects_in_tree=lambda **kwargs: shapes,
        step=lambda: steps.append("simulation"),
    )
    robot.task = SimpleNamespace(_scene=SimpleNamespace(step=robot.simulator.step))
    if planning_fails:
        with pytest.raises(RuntimeError, match="no path"):
            robot.execute_trajectory([0.2, 0.1, 1.0], 0.5, interaction=interaction)
    elif tracking_error:
        with pytest.raises(RuntimeError, match="Motion stopped"):
            robot.execute_trajectory([0.2, 0.1, 1.0], 0.5, interaction=interaction)
    else:
        robot.execute_trajectory([0.2, 0.1, 1.0], 0.5, interaction=interaction)
        assert steps == ["motion", "simulation"]
    assert all(shape.collidable for shape in shapes)


@pytest.mark.parametrize("planning_fails", [False, True])
@pytest.mark.parametrize("respondable", [False, True])
def test_only_nonphysical_sampling_helper_is_excluded(monkeypatch, planning_fails, respondable):
    constants = ModuleType("pyrep.const")
    constants.ObjectType = SimpleNamespace(SHAPE=0)
    monkeypatch.setitem(sys.modules, "pyrep", ModuleType("pyrep"))
    monkeypatch.setitem(sys.modules, "pyrep.const", constants)

    class Shape:
        collidable = True

        def __init__(self, name):
            self.name = name

        def get_name(self):
            return self.name

        def is_collidable(self):
            return self.collidable

        def is_respondable(self):
            return respondable

        def set_collidable(self, value):
            self.collidable = value

    helper, obstacle = Shape("spawn_boundary"), Shape("large_container")

    def step():
        assert helper.collidable and obstacle.collidable
        return True

    def get_path(**kwargs):
        assert helper.collidable == respondable
        assert obstacle.collidable
        assert kwargs["ignore_collisions"] is False
        if planning_fails:
            raise RuntimeError("no path")
        return SimpleNamespace(step=step)

    robot = RLBenchRobot.__new__(RLBenchRobot)
    robot.task_name = "empty_container"
    robot.get_end_effector_pose = lambda: ([0.2, 0.1, 1.0], [math.pi, 0, 0.5])
    robot.arm = SimpleNamespace(check_arm_collision=lambda: False, get_path=get_path)
    robot.simulator = SimpleNamespace(
        get_objects_in_tree=lambda **kwargs: [helper, obstacle], step=lambda: None
    )
    robot.task = SimpleNamespace(_scene=SimpleNamespace(step=robot.simulator.step))
    if planning_fails:
        with pytest.raises(RuntimeError, match="no path"):
            robot.execute_trajectory([0.2, 0.1, 1.0], 0.5)
    else:
        robot.execute_trajectory([0.2, 0.1, 1.0], 0.5)
    assert helper.collidable and obstacle.collidable


def test_rotated_tray_dimensions_are_world_aligned():
    lower, upper = world_bounds(
        [-0.025, 0.025, -0.06, 0.06, -0.1, 0.1],
        [[0, 0, 1, 0.2], [0, 1, 0, 0.3], [-1, 0, 0, 0.776], [0, 0, 0, 1]],
    )
    assert upper - lower == pytest.approx([0.2, 0.12, 0.05])
    assert upper[2] == pytest.approx(0.801)


@pytest.mark.parametrize("success,held", [(False, []), (True, ["item"]), (True, [])])
@pytest.mark.parametrize("task_name", ["empty_container", "pick_and_lift"])
def test_completion_respects_task_gripper_requirement(capsys, success, held, task_name):
    robot = RLBenchRobot.__new__(RLBenchRobot)
    robot.completion_requested = False
    robot.task_name = task_name
    robot.task = SimpleNamespace(_task=SimpleNamespace(success=lambda: (success, False)))
    robot.gripper = SimpleNamespace(get_grasped_objects=lambda: held)
    robot.task_completed()
    expected = success and (task_name != "empty_container" or not held)
    assert robot.completion_requested == expected
    assert ("Taskexec" in capsys.readouterr().out) == expected


def test_action_budget_aborts_without_executing_extra_motion():
    from malmm.agents import episode_finished
    from malmm.executor import execute_robot_code

    robot = RLBenchRobot.__new__(RLBenchRobot)
    robot.completion_requested = False
    robot.abort_reason = None
    robot.action_count = 0
    robot.max_actions = 1
    actions = []
    robot.open_gripper = lambda: actions.append("open")
    robot.detect_object = lambda: None
    robot.is_successful = lambda: False
    status, output, _ = execute_robot_code("open_gripper()\nopen_gripper()", robot)
    assert status == 1 and "ActionBudgetExceeded" in output
    assert actions == ["open"] and robot.action_count == 1
    assert episode_finished(robot) and not robot.completion_requested


@pytest.mark.parametrize("detected", [False, True])
def test_contact_shortfall_requires_positive_grasp_sensor(monkeypatch, capsys, detected):
    constants = ModuleType("pyrep.const")
    constants.ObjectType = SimpleNamespace(SHAPE=0)
    monkeypatch.setitem(sys.modules, "pyrep", ModuleType("pyrep"))
    monkeypatch.setitem(sys.modules, "pyrep.const", constants)
    target = SimpleNamespace(get_name=lambda: "Shape")
    robot = RLBenchRobot.__new__(RLBenchRobot)
    robot.task_name = "empty_container"
    robot.task = SimpleNamespace(_task=SimpleNamespace(get_graspable_objects=lambda: [target]))
    robot.gripper = SimpleNamespace(
        get_grasped_objects=lambda: [],
        _proximity_sensor=SimpleNamespace(is_detected=lambda obj: detected),
    )
    robot.arm = SimpleNamespace(
        check_arm_collision=lambda: False,
        get_path=lambda **kwargs: SimpleNamespace(step=lambda: True),
    )
    robot.simulator = SimpleNamespace(get_objects_in_tree=lambda **kwargs: [], step=lambda: None)
    robot.task._scene = SimpleNamespace(step=robot.simulator.step)
    robot.get_end_effector_pose = lambda: ([0.2, 0.1, 0.82], [math.pi, 0, 0])
    if detected:
        robot.execute_trajectory([0.2, 0.1, 0.8], 0, interaction=True)
        assert "grasp sensor detects ['Shape']" in capsys.readouterr().out
    else:
        with pytest.raises(RuntimeError, match="Motion stopped"):
            robot.execute_trajectory([0.2, 0.1, 0.8], 0, interaction=True)
