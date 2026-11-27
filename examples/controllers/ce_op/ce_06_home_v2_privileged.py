"""Open the stand and bring both connector faces against their moving guides."""

import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    Vec3,
    compose_pose,
    invert_pose,
    quat_from_axis_angle_wxyz,
    rotate_vec_by_quat_wxyz,
    sample_revolute_arc_poses,
)

_X = (1.0, 0.0, 0.0)
_Y = (0.0, 1.0, 0.0)
_Z = (0.0, 0.0, 1.0)


def _tip_to_tcp(xyz: Vec3, approach: Vec3, closing: Vec3) -> Pose3D:
    tip = Pose3D.from_approach(xyz=xyz, approach_dir=approach, closing_dir=closing)
    return _along(tip, -0.0226)


def _along(pose: Pose3D, distance: float) -> Pose3D:
    return compose_pose(pose, Pose3D((0.0, 0.0, distance)))


def _approach(ctx, arm: str, contact: Pose3D, distance: float = 0.080) -> None:
    ctx.kinematics.move_to_pose(arm, _along(contact, -distance), steps=48)
    ctx.step(6)
    ctx.kinematics.move_to_pose(arm, contact, steps=26)
    ctx.step(6)


def _riser(ctx, stand: Pose3D) -> Pose3D:
    tilt = ctx.world.joint_value("laptop_stand", "riser_tilt_lock")
    return compose_pose(stand, Pose3D((0.0, 0.025, 0.094), quat_from_axis_angle_wxyz(_X, tilt)))


def _lid(ctx, riser: Pose3D) -> Pose3D:
    angle = ctx.world.joint_value("laptop_stand", "laptop_lid")
    return compose_pose(riser, Pose3D((0.0, 0.079, 0.038), quat_from_axis_angle_wxyz(_X, angle)))


def _swing(ctx, pivot: Pose3D, contact: Pose3D, joint: str, target: float, count: int) -> Pose3D:
    delta = target - ctx.world.joint_value("laptop_stand", joint)
    arc = sample_revolute_arc_poses(
        start_pose=contact,
        hinge_pivot_world=pivot.xyz,
        hinge_axis_world=rotate_vec_by_quat_wxyz(pivot.wxyz, _X),
        delta_angle_rad=delta,
        num_waypoints=count + 1,
    )
    for pose in arc[1:]:
        ctx.kinematics.move_to_pose("left_arm", pose, steps=8)
    return arc[-1]


def _open_lid(ctx, stand: Pose3D) -> None:
    home_r = tuple(ctx.sensors.proprio()[8:14])
    riser = _riser(ctx, stand)
    lid = _lid(ctx, riser)
    # Bimanual hold-and-push: left_arm hooks under LidLift, swings to -0.55 rad and holds
    # while right_arm braces the exposed screen underside and pushes past -90 deg to -1.96 rad.
    ctx.end_effector.set_posture("left_gripper", "open", steps=20)
    ctx.end_effector.command_targets("left_gripper", (0.011, 0.011), steps=22)
    lid_app = (0.0, 0.94, -0.34)
    lid_cls = (0.0, 0.34, 0.94)
    hinge = compose_pose(riser, Pose3D((0.0, 0.079, 0.038)))
    peel_local = (-0.020, -0.209, 0.0085)
    grip_local = (-0.020, -0.200, 0.0029)

    def _lid_arc(arm: str, pt_local, app_dir, cls_dir, goal_ang: float, count: int) -> Pose3D:
        start_a = ctx.world.joint_value("laptop_stand", "laptop_lid")
        app_w = rotate_vec_by_quat_wxyz(riser.wxyz, app_dir)
        cls_w = rotate_vec_by_quat_wxyz(riser.wxyz, cls_dir)
        end_p = _tip_to_tcp(compose_pose(hinge, Pose3D(pt_local)).xyz, app_w, cls_w)
        for idx in range(1, count + 1):
            ang_i = start_a + (goal_ang - start_a) * idx / count
            lid_i = compose_pose(hinge, Pose3D((0.0, 0.0, 0.0), quat_from_axis_angle_wxyz(_X, ang_i)))
            end_p = _tip_to_tcp(compose_pose(lid_i, Pose3D(pt_local)).xyz, app_w, cls_w)
            ctx.kinematics.move_to_pose(arm, end_p, steps=8)
        return end_p

    peel = _tip_to_tcp(compose_pose(lid, Pose3D(peel_local)).xyz, lid_app, lid_cls)
    _approach(ctx, "left_arm", peel, 0.042)
    _lid_arc("left_arm", peel_local, lid_app, lid_cls, -0.22, 7)
    lid = _lid(ctx, _riser(ctx, stand))
    bite_local = grip_local if ctx.world.joint_value("laptop_stand", "laptop_lid") <= -0.07 else peel_local
    grip = _tip_to_tcp(compose_pose(lid, Pose3D(bite_local)).xyz, lid_app, lid_cls)
    ctx.kinematics.move_to_pose("left_arm", grip, steps=26)
    ctx.step(6)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=32)
    left_hold = _lid_arc("left_arm", bite_local, lid_app, lid_cls, -0.55, 8)

    # Right arm braces the exposed screen underside at the current lid angle before left_arm releases.
    ctx.end_effector.set_posture("right_gripper", "closed", steps=20)
    ang_cur = ctx.world.joint_value("laptop_stand", "laptop_lid")
    scr_cur = compose_pose(hinge, Pose3D((0.0, 0.0, 0.0), quat_from_axis_angle_wxyz(_X, ang_cur)))
    push_pt = (0.045, -0.155, 0.0005)
    push_app = (0.0, 0.95, -0.312)
    brace = _tip_to_tcp(
        compose_pose(scr_cur, Pose3D(push_pt)).xyz,
        rotate_vec_by_quat_wxyz(riser.wxyz, push_app),
        rotate_vec_by_quat_wxyz(riser.wxyz, _X),
    )
    _approach(ctx, "right_arm", brace, 0.055)

    ctx.end_effector.command_targets("left_gripper", (0.012, 0.012), steps=20)
    ctx.kinematics.move_to_pose("left_arm", _along(left_hold, -0.060), steps=28)
    end = _lid_arc("right_arm", push_pt, push_app, _X, -1.96, 16)
    ctx.step(6)
    ctx.kinematics.move_to_pose("right_arm", _along(end, -0.065), steps=28)
    ctx.kinematics.move_joints("right_arm", home_r, steps=36)


def _tilt_riser(ctx, stand: Pose3D) -> None:
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    riser = _riser(ctx, stand)
    # Pinch the 10 mm thickness at the free end of the paddle. Lateral entry
    # leaves the wrist outside the low front edge as the stand tilts forward.
    ctx.end_effector.command_targets("left_gripper", (0.0065, 0.0065), steps=22)
    direction = (0.86, 0.28, -0.426)
    closing = (0.426, 0.0, 0.905)
    grip = compose_pose(riser, _tip_to_tcp((-0.105, -0.024, -0.001), direction, closing))
    high = compose_pose(stand, _tip_to_tcp((-0.185, -0.095, 0.165), direction, closing))
    ctx.kinematics.move_to_pose("left_arm", high, steps=48)
    ctx.step(6)
    _approach(ctx, "left_arm", grip, 0.050)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=32)
    ctx.step(10)
    end = _swing(ctx, riser, grip, "riser_tilt_lock", math.radians(24.0), 12)
    ctx.end_effector.command_targets("left_gripper", (0.018, 0.018), steps=22)
    ctx.step(8)
    ctx.kinematics.move_to_pose("left_arm", _along(end, -0.060), steps=36)


def _dock(ctx, stand: Pose3D, prop_id: str, side: str, inward: int, half_length: float) -> None:
    arm, gripper = side + "_arm", side + "_gripper"
    prop = ctx.world.object_pose(prop_id)
    # Close across the body's local Y. Hold the tail outside the cantilever fences.
    hand = _tip_to_tcp((-0.012, 0.0, 0.006), (0.15, 0.0, -0.9887), _Y)
    grip = compose_pose(prop, hand)
    ctx.end_effector.set_posture(gripper, "open", steps=22)
    _approach(ctx, arm, grip, 0.110)
    ctx.end_effector.set_posture(gripper, "closed", steps=32)
    ctx.step(12)
    held = compose_pose(invert_pose(ctx.world.object_pose(prop_id)), ctx.kinematics.ee_pose(arm))
    raised = Pose3D((prop.xyz[0], prop.xyz[1], stand.xyz[2] + 0.215), prop.wxyz)
    ctx.kinematics.move_to_pose(arm, compose_pose(raised, held), steps=44)

    # Both authored noses point along prop +X. Turn at clearance height so
    # the right-hand adapter faces -X and both bodies roll with the shelf.
    riser = _riser(ctx, stand)
    facing = quat_from_axis_angle_wxyz(_Z, 0.0 if inward == 1 else math.pi)
    # The package uses full-width box colliders. Their front faces meet the
    # guide cheeks 2.8 mm outside the wall. The sprung guides have 2.4 mm travel.
    x = -inward * (0.162 + 0.0028 + half_length - 0.0015)
    above_seat = compose_pose(riser, Pose3D((x, 0.018, 0.0250), facing))
    outside_above = compose_pose(above_seat, Pose3D((-0.065, 0.0, 0.0)))
    seated = compose_pose(riser, Pose3D((x, 0.018, 0.0224), facing))
    # Align yaw in the clear. A half turn needs more than 94 ticks at 2 rad/s.
    level = Pose3D(raised.xyz, seated.wxyz)
    ctx.kinematics.move_to_pose(arm, compose_pose(level, held), steps=112)
    ctx.step(6)
    ctx.kinematics.move_to_pose(arm, compose_pose(outside_above, held), steps=48)
    ctx.step(6)
    ctx.kinematics.move_to_pose(arm, compose_pose(above_seat, held), steps=36)
    ctx.step(6)
    ctx.kinematics.move_to_pose(arm, compose_pose(seated, held), steps=18)
    ctx.step(16)
    if inward == -1:
        ctx.end_effector.set_posture(gripper, "open", steps=22)
        ctx.step(8)
        ctx.kinematics.move_to_pose(arm, compose_pose(outside_above, held), steps=32)
    # The PD body's centre remains outside its 26 mm shelf. The left
    # hand supports it while the right hand seats HDMI and presses Confirm.


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    stand = ctx.world.object_pose("laptop_stand")
    _open_lid(ctx, stand)
    _tilt_riser(ctx, stand)
    ctx.kinematics.set_torso_lift(0.46, steps=32)
    _dock(ctx, stand, "pd_plug", "left", 1, 0.03425)
    _dock(ctx, stand, "hdmi_dongle", "right", -1, 0.02605)

    # Confirm is a 23 x 13 mm cap in a 25 x 15 mm well. Track the deck
    # normal through the 2.8 mm spring stroke and hold the terminal contact.
    ctx.end_effector.set_posture("right_gripper", "closed", steps=32)
    riser = _riser(ctx, stand)
    hand = _tip_to_tcp((0.111, -0.097, 0.0357), (0.0, 0.15, -0.9887), _Y)
    surface = compose_pose(riser, hand)
    _approach(ctx, "right_arm", surface, 0.072)
    for depth in (0.0007, 0.0014, 0.0020):
        # Read the tilt again under load before each normal increment.
        riser = _riser(ctx, stand)
        depressed = compose_pose(riser, compose_pose(Pose3D((0.0, 0.0, -depth)), hand))
        ctx.kinematics.move_to_pose("right_arm", depressed, steps=8)
        if ctx.world.joint_value("laptop_stand", "confirm_key") <= -0.0022:
            break
    hold = compose_pose(_riser(ctx, stand), compose_pose(Pose3D((0.0, 0.0, -0.0027)), hand))
    ctx.kinematics.move_to_pose("right_arm", hold, steps=12)
    ctx.step(64)
