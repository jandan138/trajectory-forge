"""Set the projector's shutter, two gimbal axes and focus in rubric order."""

import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    Vec3,
    compose_pose,
    quat_from_axis_angle_wxyz,
    rotate_vec_by_quat_wxyz,
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
    ctx.kinematics.move_to_pose(arm, _along(contact, -distance), steps=24)
    ctx.step(3)
    ctx.kinematics.move_to_pose(arm, contact, steps=14)
    ctx.step(3)


def _cradle(base: Pose3D, yaw: float) -> Pose3D:
    return compose_pose(base, Pose3D((0.0, 0.0, 0.029), quat_from_axis_angle_wxyz(_Z, yaw)))


def _head(cradle: Pose3D, pitch: float) -> Pose3D:
    # GimbalPitchJoint localRot0=(-X) means positive pitch raises the front (-Y) lens.
    return compose_pose(cradle, Pose3D((0.0, 0.0, 0.124), quat_from_axis_angle_wxyz((-1.0, 0.0, 0.0), pitch)))


def _live_head(ctx, base: Pose3D) -> Pose3D:
    yaw = ctx.world.joint_value("projector", "gimbal_yaw")
    pitch = ctx.world.joint_value("projector", "gimbal_pitch")
    return _head(_cradle(base, yaw), pitch)


def _press(ctx, base: Pose3D, side: str, joint: str, x: float) -> None:
    arm = side + "_arm"
    direction = 1.0 if side == "right" else -1.0
    yaw = ctx.world.joint_value("projector", "gimbal_yaw")
    exposed = -direction * yaw
    contact_y = 0.102
    if exposed >= 0.28:
        app_local = (-direction * 0.65, 0.0, -0.760)
    elif exposed >= 0.17:
        app_local = (-direction * 0.85, 0.08, -0.521)
        contact_y = 0.103
    else:
        app_local = (-direction * 0.75, 0.24, -0.617)
    closing_local = _Y if side == "right" else (0.0, -1.0, 0.0)
    ctx.end_effector.set_posture(side + "_gripper", "closed", steps=16)
    # The outboard edge leaves clearance for the knuckles behind the head shell.
    def contact(z: float) -> Pose3D:
        head = _live_head(ctx, base)
        point = compose_pose(head, Pose3D((x, contact_y, z))).xyz
        app = rotate_vec_by_quat_wxyz(head.wxyz, app_local)
        closing = rotate_vec_by_quat_wxyz(head.wxyz, closing_local)
        return _tip_to_tcp(point, app, closing)

    ready = contact(0.017)
    _approach(ctx, arm, ready, 0.044)
    end = ready
    for i in range(1, 15):
        end = contact(0.017 - 0.0012 * i)
        ctx.kinematics.move_to_pose(arm, end, steps=8)
        if ctx.world.joint_value("projector", joint) <= -0.0020:
            break
    ctx.step(24)
    ctx.kinematics.move_to_pose(arm, _along(end, -0.060), steps=24)


def _open_shutter(ctx, base: Pose3D) -> None:
    # Start 8.5 mm left of ThumbGrip [-0.0655..-0.0185, -0.1228..-0.1158] using -Y closing.
    ctx.end_effector.set_posture("left_gripper", "closed", steps=18)
    for _ in range(2):
        q = ctx.world.joint_value("projector", "lens_shutter")
        if q >= 0.066:
            break
        head = _live_head(ctx, base)
        hand = _tip_to_tcp((-0.074, -0.1195, 0.014), (0.0, 0.50, -0.8660), (0.0, -1.0, 0.0))
        _approach(ctx, "left_arm", compose_pose(head, compose_pose(Pose3D((q, 0.0, 0.0)), hand)), 0.050)
        end = compose_pose(head, compose_pose(Pose3D((q, 0.0, 0.0)), hand))
        for i in range(1, 15):
            dx = q + (0.086 - q) * i / 14
            end = compose_pose(head, compose_pose(Pose3D((dx, 0.0, 0.0)), hand))
            ctx.kinematics.move_to_pose("left_arm", end, steps=6)
            if ctx.world.joint_value("projector", "lens_shutter") >= 0.066:
                break
        ctx.kinematics.move_to_pose("left_arm", _along(end, -0.060), steps=18)


def _lower_head_for_power(ctx, base: Pose3D) -> None:
    arm = "right_arm"
    app = (0.0, 0.45, -0.893)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=16)
    for attempt in range(2):
        yaw = ctx.world.joint_value("projector", "gimbal_yaw")
        pitch = ctx.world.joint_value("projector", "gimbal_pitch")
        if attempt and pitch <= -0.065:
            break
        cradle = _cradle(base, yaw)
        point = compose_pose(_head(cradle, pitch), Pose3D((0.030, -0.060, 0.058))).xyz
        end = _tip_to_tcp(point, app, _X)
        _approach(ctx, arm, end, 0.045)
        for i in range(1, 13):
            t = i / 12
            head = _head(cradle, pitch + (-0.102 - pitch) * t)
            point = compose_pose(head, Pose3D((0.030, -0.060, 0.058 - 0.012 * t))).xyz
            end = _tip_to_tcp(point, app, _X)
            ctx.kinematics.move_to_pose(arm, end, steps=8)
            if ctx.world.joint_value("projector", "gimbal_pitch") <= -0.073 and i >= 6:
                break
        ctx.step(24)
        ctx.kinematics.move_to_pose(arm, _along(end, -0.055), steps=24)


def _set_gimbal(ctx, base: Pose3D, home_right: tuple[float, ...]) -> None:
    # Grip the left fork low on the yoke to steady yaw while adjusting pitch.
    fork_app = (0.92, 0.08, -0.3837)
    fork_closing = (0.0, -1.0, 0.0)
    fork_hand = _tip_to_tcp((-0.127, 0.0, 0.036), fork_app, fork_closing)
    side = compose_pose(base, _tip_to_tcp((-0.180, -0.050, 0.120), fork_app, fork_closing))
    yaw = ctx.world.joint_value("projector", "gimbal_yaw")
    ctx.end_effector.command_targets("left_gripper", (0.030, 0.030), steps=24)
    ctx.kinematics.move_to_pose("left_arm", side, steps=28)
    _approach(ctx, "left_arm", compose_pose(_cradle(base, yaw), fork_hand), 0.049)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=40)
    ctx.step(16)

    # Pinch the pitch handle across its narrow dimension and follow its hinge arc.
    pitch = ctx.world.joint_value("projector", "gimbal_pitch")
    cradle = _cradle(base, ctx.world.joint_value("projector", "gimbal_yaw"))
    pitch_app = rotate_vec_by_quat_wxyz(cradle.wxyz, (-0.35, 0.42, -0.837))
    pitch_closing = rotate_vec_by_quat_wxyz(cradle.wxyz, _Y)

    def pitch_pose(angle: float) -> Pose3D:
        point = compose_pose(_head(cradle, angle), Pose3D((0.029, 0.016, 0.062))).xyz
        return _tip_to_tcp(point, pitch_app, pitch_closing)

    ctx.end_effector.command_targets("right_gripper", (0.015, 0.015), steps=24)
    end = pitch_pose(pitch)
    _approach(ctx, "right_arm", end, 0.058)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=36)
    ctx.step(16)
    for i in range(1, 14):
        end = pitch_pose(pitch + (0.361 - pitch) * i / 13)
        ctx.kinematics.move_to_pose("right_arm", end, steps=8)
        if ctx.world.joint_value("projector", "gimbal_pitch") >= 0.33:
            break
    ctx.step(24)
    ctx.end_effector.command_targets("right_gripper", (0.018, 0.018), steps=24)
    ctx.kinematics.move_to_pose("right_arm", _along(end, -0.064), steps=24)
    ctx.kinematics.move_joints("right_arm", home_right, steps=24)

    # Turn the cradle with the same fork grasp and release after it settles.
    yaw = ctx.world.joint_value("projector", "gimbal_yaw")
    for i in range(1, 17):
        angle = yaw + (0.60 - yaw) * i / 16
        end = compose_pose(_cradle(base, angle), fork_hand)
        ctx.kinematics.move_to_pose("left_arm", end, steps=8)
        if ctx.world.joint_value("projector", "gimbal_yaw") >= 0.56:
            break
    ctx.step(24)
    ctx.end_effector.command_targets("left_gripper", (0.030, 0.030), steps=24)
    ctx.kinematics.move_to_pose("left_arm", _along(end, -0.058), steps=24)


def _pan(ctx, base: Pose3D, target: float) -> None:
    # Push +Y on the front of either fork to turn the cradle in the desired direction.
    yaw = ctx.world.joint_value("projector", "gimbal_yaw")
    side = "right" if target > yaw else "left"
    arm = side + "_arm"
    direction = 1 if side == "right" else -1
    if abs(target - yaw) <= 0.035:
        return
    pan_app = (0.0, 0.92, -0.3919)
    ctx.end_effector.set_posture(side + "_gripper", "closed", steps=16)
    cradle = _cradle(base, yaw)
    outside = compose_pose(cradle, _tip_to_tcp((direction * 0.147, -0.130, 0.134), (0.0, 0.80, -0.60), _X))
    ctx.kinematics.move_to_pose(arm, outside, steps=26)
    end = compose_pose(cradle, _tip_to_tcp((direction * 0.127, -0.024, 0.040), pan_app, _X))
    _approach(ctx, arm, end, 0.055)
    n = min(18, 7 + int(abs(target - yaw) / 0.075))
    for i in range(1, n + 1):
        t = i / n
        angle = yaw + (target + direction * 0.04 - yaw) * t
        cradle = _cradle(base, angle)
        end = compose_pose(cradle, _tip_to_tcp((direction * 0.127, -0.024 + 0.010 * t, 0.040), pan_app, _X))
        ctx.kinematics.move_to_pose(arm, end, steps=8)
        if direction * (target - ctx.world.joint_value("projector", "gimbal_yaw")) <= 0.0:
            break
    ctx.step(24)
    ctx.kinematics.move_to_pose(arm, _along(end, -0.075), steps=24)


def _focus(ctx, base: Pose3D) -> None:
    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    # Sweep UPWARD (+Z) across unobstructed 9 o'clock (-X) FocusRing flank into [0.5105, 0.7854].
    f_app = (0.58, 0.55, -0.601)
    r_c, r_out = 0.0338, 0.068
    for stroke in range(6):
        # Several full sweeps give the resampled actions time to seat against the knurl.
        if stroke >= 4 and ctx.world.joint_value("projector", "focus_ring") >= 0.66:
            break
        head = _live_head(ctx, base)
        th0 = -0.45

        def ring_pose(radius: float, angle: float) -> Pose3D:
            local = (-0.042 - radius * math.cos(angle), -0.100, 0.007 + radius * math.sin(angle))
            return _tip_to_tcp(compose_pose(head, Pose3D(local)).xyz, f_app, _Z)

        start = ring_pose(r_out, th0)
        if stroke == 0:
            ctx.kinematics.move_to_pose("left_arm", _along(start, -0.035), steps=24)
        ctx.kinematics.move_to_pose("left_arm", start, steps=12)
        ctx.kinematics.move_to_pose("left_arm", ring_pose(r_c, th0), steps=12)
        th_end = th0
        for i in range(1, 13):
            th = -0.45 + 0.90 * i / 12
            th_end = th
            ctx.kinematics.move_to_pose("left_arm", ring_pose(r_c, th), steps=8)
            if stroke >= 4 and ctx.world.joint_value("projector", "focus_ring") >= 0.70:
                break
        ctx.step(12)
        ctx.kinematics.move_to_pose("left_arm", ring_pose(r_out, th_end), steps=12)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=16)
    home = ctx.sensors.proprio()
    base = ctx.world.object_pose("projector")
    _pan(ctx, base, -0.63)
    ctx.kinematics.move_joints("left_arm", tuple(home[:6]), steps=24)
    _lower_head_for_power(ctx, base)
    ctx.kinematics.move_joints("right_arm", tuple(home[8:14]), steps=24)
    _press(ctx, base, "right", "power_button", 0.056)
    ctx.kinematics.move_joints("right_arm", tuple(home[8:14]), steps=24)
    _pan(ctx, base, 0.0)
    ctx.kinematics.move_joints("right_arm", tuple(home[8:14]), steps=24)
    _open_shutter(ctx, base)
    ctx.kinematics.move_joints("left_arm", tuple(home[:6]), steps=24)
    _set_gimbal(ctx, base, tuple(home[8:14]))
    ctx.step(16)
    _focus(ctx, base)
    ctx.kinematics.move_joints("left_arm", tuple(home[:6]), steps=24)
    _press(ctx, base, "left", "keystone_lock", -0.051)
    ctx.kinematics.move_joints("left_arm", tuple(home[:6]), steps=24)
    ctx.step(40)
