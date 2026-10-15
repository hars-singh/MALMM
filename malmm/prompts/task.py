TASK_PROMPT = """
Use the robot arm end effector to "[INSERT TASK]" in the environment.
The robot arm end-effector is currently positioned at [INSERT EE POSITION], with the orientation [INSERT EE ORIENTATION], and the gripper is open.
**Remember you can only grasp the objects from its *CENTER*. Not from any other position. So to grasp the object, end effector has to be lowered down to center of object**
The positions and orientations  of all objects in the environment as follows:
<CURRENT ENVIRONMENT STATE>:
"[STATE]"
"""
