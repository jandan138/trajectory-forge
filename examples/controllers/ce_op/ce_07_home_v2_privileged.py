"""Open the drawer, stow the pen, dock the mouse and set the riser controls."""

import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    Vec3,
    compose_pose,
    invert_pose,
)

_X = (1.0, 0.0, 0.0)
_Y = (0.0, 1.0, 0.0)
_Z = (0.0, 0.0, 1.0)
_DOWN = (0.0, 0.15, -0.9887)


def _tip_to_tcp(xyz: Vec3, approach: Vec3, closing: Vec3) -> Pose3D:
    tip = Pose3D.from_approach(xyz=xyz, approach_dir=approach, closing_dir=closing)
    return _along(tip, -0.0226)


def _along(pose: Pose3D, distance: float) -> Pose3D:
    return compose_pose(pose, Pose3D((0.0, 0.0, distance)))


def _approach(ctx, arm: str, contact: Pose3D, distance: float = 0.080) -> None:
    ctx.kinematics.move_to_pose(arm, _along(contact, -distance), steps=24)
    ctx.step(4)
    ctx.kinematics.move_to_pose(arm, contact, steps=16)
    ctx.step(4)


def _drawer(ctx, base: Pose3D, opening: bool) -> None:
    arm = "left_arm"
    ctx.end_effector.set_posture("left_gripper", "closed", steps=20)
    q = ctx.world.joint_value("monitor_riser", "drawer_slide")
    if opening:
        # The gap from the back of the pull to the fascia is 22.5 mm.
        # Drop closed tips 9 mm below its top, between the two mounting feet.
        xyz, approach, closing = (-0.050, -0.1415, 0.043), _DOWN, (0.0, -1.0, 0.0)
        goal = -0.142
    else:
        # Push the exterior of the 9 mm deep bar at its centre height.
        xyz, approach, closing = (-0.050, -0.1575, 0.047), (0.0, 0.78, -0.6258), _X
        goal = 0.0
    contact = _tip_to_tcp(xyz, approach, closing)
    start = compose_pose(base, compose_pose(Pose3D((0.0, q, 0.0)), contact))
    _approach(ctx, arm, start)
    end = start
    for i in range(1, 17):
        # Take up the hook clearance before the pull. The push stops at flush.
        end_q = goal - 0.006 if opening else goal + 0.0005
        y = q + (end_q - q) * i / 16
        end = compose_pose(base, compose_pose(Pose3D((0.0, y, 0.0)), contact))
        ctx.kinematics.move_to_pose(arm, end, steps=8)
        actual = ctx.world.joint_value("monitor_riser", "drawer_slide")
        if (opening and actual <= goal) or (not opening and actual >= -0.002):
            break
    ctx.step(8)
    ctx.kinematics.move_to_pose(arm, _along(end, -0.080), steps=20)


def _place_prop(ctx, side: str, prop_id: str, hand: Pose3D, base: Pose3D) -> None:
    arm, gripper = side + "_arm", side + "_gripper"
    prop = ctx.world.object_pose(prop_id)
    if prop_id == "presenter_pen":
        ctx.end_effector.command_targets(gripper, (0.026, 0.026), steps=20)
        ctx.kinematics.move_to_pose(arm, _along(compose_pose(prop, hand), -0.110), steps=28)
        ctx.step(6)
        ctx.kinematics.move_to_pose(arm, compose_pose(prop, hand), steps=20)
        ctx.step(12)
        ctx.end_effector.set_posture(gripper, "closed", steps=40)
        ctx.step(12)
    else:
        ctx.end_effector.set_posture(gripper, "open", steps=16)
        _approach(ctx, arm, compose_pose(prop, hand), 0.110)
        ctx.end_effector.set_posture(gripper, "closed", steps=28)
        ctx.step(8)
    held = compose_pose(invert_pose(ctx.world.object_pose(prop_id)), ctx.kinematics.ee_pose(arm))
    if prop_id == "presenter_pen":
        q = ctx.world.joint_value("monitor_riser", "drawer_slide")
        # Centre inside the open presenter-pen felt bay ahead of PenCradleBack (-0.0535 + q).
        seat = compose_pose(base, Pose3D((-0.169, -0.084 + q, 0.036)))
        clearance = 0.130
    else:
        # The floating traction skin sits 2 mm above the charging-pad pivot.
        q = ctx.world.joint_value("monitor_riser", "charging_pad")
        seat = compose_pose(base, Pose3D((0.227, 0.025, 0.109 + q)))
        clearance = 0.095
    transit_z = seat.xyz[2] + clearance
    raised = Pose3D((prop.xyz[0], prop.xyz[1], transit_z), prop.wxyz)
    ctx.kinematics.move_to_pose(arm, compose_pose(raised, held), steps=28 if prop_id == "presenter_pen" else 24)
    above = compose_pose(seat, Pose3D((0.0, 0.0, clearance)))
    ctx.kinematics.move_to_pose(arm, compose_pose(above, held), steps=32 if prop_id == "presenter_pen" else 28)
    ctx.step(6)
    release = compose_pose(compose_pose(seat, Pose3D((0.0, 0.0, 0.001))), held)
    ctx.kinematics.move_to_pose(arm, release, steps=24 if prop_id == "presenter_pen" else 20)
    ctx.step(12 if prop_id == "presenter_pen" else 8)
    if prop_id == "presenter_pen":
        # Clear the 14.8 mm barrel cleanly inside the centered pocket.
        ctx.end_effector.command_targets(gripper, (0.022, 0.022), steps=20)
        ctx.step(8)
    else:
        ctx.end_effector.set_posture(gripper, "open", steps=18)
        ctx.step(8)
    ctx.kinematics.move_to_pose(
        arm,
        Pose3D((release.xyz[0], release.xyz[1], release.xyz[2] + 0.085), release.wxyz),
        steps=22,
    )


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=12)
    home = ctx.sensors.proprio()
    base = ctx.world.object_pose("monitor_riser")
    _drawer(ctx, base, True)
    pen_hand = _tip_to_tcp((0.0, 0.0, 0.0085), _DOWN, _Y)
    _place_prop(ctx, "left", "presenter_pen", pen_hand, base)
    _drawer(ctx, base, False)

    # The mouse spans 94 mm along Y and 60 mm along X.
    mouse_hand = _tip_to_tcp((0.0, -0.020, 0.006), (0.0, 0.29, -0.9570), _X)
    _place_prop(ctx, "right", "desk_mouse", mouse_hand, base)

    # Push on the left end of the 31 mm thumb shoe, forward of the camera bridge.
    ctx.end_effector.set_posture("left_gripper", "closed", steps=20)
    q = ctx.world.joint_value("monitor_riser", "cam_privacy_slider")
    contact = _tip_to_tcp((-0.223 - 0.018, -0.120, 0.1465), _DOWN, _X)
    start_cam = compose_pose(base, compose_pose(Pose3D((q, 0.0, 0.0)), contact))
    _approach(ctx, "left_arm", start_cam)
    end = start_cam
    for i in range(1, 11):
        dx = q + (0.044 - q) * i / 10
        end = compose_pose(base, compose_pose(Pose3D((dx, 0.0, 0.0)), contact))
        ctx.kinematics.move_to_pose("left_arm", end, steps=8)
        if ctx.world.joint_value("monitor_riser", "cam_privacy_slider") >= 0.040:
            break
    ctx.step(8)
    ctx.kinematics.move_to_pose("left_arm", _along(end, -0.075), steps=20)
    ctx.kinematics.move_joints("left_arm", tuple(home[:6]), steps=28)

    # Closed-tip tangential flute ratchet around BrightnessDial spindle (base + (0.227, -0.087, 0.119)).
    dial_piv = compose_pose(base, Pose3D((0.227, -0.087, 0.119))).xyz
    cx, cy, cz = dial_piv[0], dial_piv[1], dial_piv[2] - 0.002
    rim_dir = (0.0, 0.12, -0.9928)
    r_c, r_out = 0.0144, 0.027
    ctx.end_effector.set_posture("right_gripper", "closed", steps=14)
    for _ in range(12):
        if ctx.world.joint_value("monitor_riser", "brightness_dial") >= 0.94:
            break
        th0 = -1.25
        cls0 = (-math.sin(th0), math.cos(th0), 0.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_out * math.cos(th0), cy + r_out * math.sin(th0), cz + 0.045), rim_dir, cls0), steps=10)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_out * math.cos(th0), cy + r_out * math.sin(th0), cz), rim_dir, cls0), steps=8)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_c * math.cos(th0), cy + r_c * math.sin(th0), cz), rim_dir, cls0), steps=12)
        ctx.step(6)
        th_end = th0
        for k in range(1, 13):
            th = -1.25 + 1.65 * (k / 12.0)
            th_end = th
            t_cls = (-math.sin(th), math.cos(th), 0.0)
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_c * math.cos(th), cy + r_c * math.sin(th), cz), rim_dir, t_cls), steps=8)
            if ctx.world.joint_value("monitor_riser", "brightness_dial") >= 0.94:
                break
        ctx.step(6)
        cls_end = (-math.sin(th_end), math.cos(th_end), 0.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_out * math.cos(th_end), cy + r_out * math.sin(th_end), cz), rim_dir, cls_end), steps=10)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_out * math.cos(th_end), cy + r_out * math.sin(th_end), cz + 0.045), rim_dir, cls_end), steps=8)

    ctx.end_effector.set_posture("right_gripper", "closed", steps=18)
    cap = _tip_to_tcp((0.099, -0.088, 0.116), _DOWN, _X)
    start_sleep = compose_pose(base, cap)
    _approach(ctx, "right_arm", start_sleep)
    end = start_sleep
    for depth in (0.0010, 0.0020, 0.0032, 0.0044, 0.0056, 0.0068):
        end = compose_pose(base, compose_pose(Pose3D((0.0, 0.0, -depth)), cap))
        ctx.kinematics.move_to_pose("right_arm", end, steps=8)
        if ctx.world.joint_value("monitor_riser", "sleep_button") <= -0.0022:
            break
    ctx.step(20)
    ctx.kinematics.move_to_pose("right_arm", _along(end, -0.080), steps=20)
    ctx.kinematics.move_joints("right_arm", tuple(home[8:14]), steps=28)
    ctx.step(64)
