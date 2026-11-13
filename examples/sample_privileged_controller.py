"""Example privileged Python controller exercising world, kinematics, end_effector, and sensors."""

import math
from trajectory_forge.sdk import Pose3D


def solve(ctx):
    # Demonstrate reading either scenario objects or falling back cleanly in smoke mode
    start_pose = Pose3D(xyz=(0.0, -0.15, 0.95), wxyz=(1.0, 0.0, 0.0, 0.0))

    ctx.end_effector.set_posture("right_gripper", "open")
    ctx.kinematics.solve_ik("right_arm", start_pose, check_collision=True)
    ctx.step(5)

    # Guarded contact close loop (reserves clean upgrade path to tactile grippers)
    ctx.end_effector.set_posture("right_gripper", "closed")
    for _ in range(3):
        tactile = ctx.sensors.tactile("right_gripper")
        if tactile.slip_detected:
            break
        ctx.step(1)

    # Constrained revolute arc motion (no manual quaternion math in agent code)
    arc = ctx.kinematics.plan_revolute_arc(
        chain="right_arm",
        start_pose=start_pose,
        hinge_pivot_world=(0.0, 0.0, 0.95),
        hinge_axis_world=(1.0, 0.0, 0.0),
        delta_angle_rad=math.pi / 4.0,
        num_waypoints=5,
    )
    for wp in arc:
        ctx.kinematics.solve_ik("right_arm", wp, check_collision=True)
        ctx.step(2)
