import sys
from types import ModuleType, SimpleNamespace

import pytest

from malmm.robot import RLBenchRobot


@pytest.mark.parametrize("planning_fails", [False, True])
def test_initial_contacts_are_restored_before_motion_or_after_planning_failure(
    monkeypatch, planning_fails
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
        assert kwargs == {"position": [0.2, 0.1, 1.0], "euler": [3.14, 0, -0.5]}
        if planning_fails:
            raise RuntimeError("no path")
        return SimpleNamespace(step=step)

    robot = RLBenchRobot.__new__(RLBenchRobot)
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
    if planning_fails:
        with pytest.raises(RuntimeError, match="no path"):
            robot.execute_trajectory([0.2, 0.1, 1.0], 0.5)
    else:
        robot.execute_trajectory([0.2, 0.1, 1.0], 0.5)
        assert steps == ["motion", "simulation"]
    assert all(shape.collidable for shape in shapes)
