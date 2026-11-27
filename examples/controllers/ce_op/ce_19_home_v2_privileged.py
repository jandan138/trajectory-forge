"""Operate the projector and speaker in the Cinema Cascade task."""

import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    Vec3,
    compose_pose,
    normalize_vec3,
    quat_from_axis_angle_wxyz,
    rotate_vec_by_quat_wxyz,
)

_X = (1.0, 0.0, 0.0)
_Y = (0.0, 1.0, 0.0)
_Z = (0.0, 0.0, 1.0)
_NEG_X = (-1.0, 0.0, 0.0)
_PROJECTOR = "projector"
_SPEAKER = "smart_speaker"


def _offset(point: Vec3, direction: Vec3, distance: float) -> Vec3:
    return (
        point[0] + distance * direction[0],
        point[1] + distance * direction[1],
        point[2] + distance * direction[2],
    )


def _tip_to_tcp(tip: Vec3, approach: Vec3, closing: Vec3) -> Pose3D:
    approach = normalize_vec3(approach)
    return Pose3D.from_approach(
        xyz=_offset(tip, approach, -0.0226), approach_dir=approach, closing_dir=closing
    )


def _along(pose: Pose3D, distance: float) -> Pose3D:
    return compose_pose(pose, Pose3D((0.0, 0.0, distance)))


def _fingertip_tcp(tip: Vec3, approach: Vec3, closing: Vec3, finger_offset: float) -> Pose3D:
    pose = _tip_to_tcp(tip, approach, closing)
    jaw_axis = rotate_vec_by_quat_wxyz(pose.wxyz, _Y)
    return Pose3D(_offset(pose.xyz, jaw_axis, -finger_offset), pose.wxyz)


def _cradle(base: Pose3D, yaw: float) -> Pose3D:
    return compose_pose(base, Pose3D((0.0, 0.0, 0.029), quat_from_axis_angle_wxyz(_Z, yaw)))


def _head(base: Pose3D, yaw: float, pitch: float) -> Pose3D:
    return compose_pose(
        _cradle(base, yaw), Pose3D((0.0, 0.0, 0.124), quat_from_axis_angle_wxyz(_NEG_X, pitch))
    )


def _live_head(ctx, base: Pose3D) -> Pose3D:
    yaw = ctx.world.joint_value(_PROJECTOR, "gimbal_yaw")
    pitch = ctx.world.joint_value(_PROJECTOR, "gimbal_pitch")
    return _head(base, yaw, pitch)


def _press_projector(ctx, base: Pose3D, joint: str, x: float):
    # Work at the back edge of the cap so the fingertips clear the head shell.
    outboard = x < 0.0
    app = (0.65, 0.20, -0.733) if outboard else (0.0, 0.04, -0.9992)
    yaw = ctx.world.joint_value(_PROJECTOR, "gimbal_yaw")
    closing = rotate_vec_by_quat_wxyz(
        _cradle(base, yaw).wxyz, (0.0, -1.0, 0.0) if outboard else _Y
    )
    y = 0.098 if outboard else 0.1005
    raised = outboard and ctx.world.joint_value(_PROJECTOR, "gimbal_pitch") > 0.18
    if raised:
        app = (0.20, -0.10, -0.9747)
        closing = (-0.4724, 0.8617, -0.1853)
        y = 0.100
    top = 0.016 if outboard else 0.012

    def press_pose(z):
        tip = compose_pose(_live_head(ctx, base), Pose3D((x, y, z))).xyz
        return _tip_to_tcp(tip, app, closing)

    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    ready = press_pose(top)
    high = _along(ready, -0.055)
    if outboard:
        if raised:
            high = Pose3D(_offset(ready.xyz, _Z, 0.021), ready.wxyz)
            over = Pose3D(_offset(ready.xyz, (0.0, -0.4, 1.0), 0.070), ready.wxyz)
        else:
            over = Pose3D(_offset(high.xyz, _Z, 0.070), high.wxyz)
        ctx.kinematics.move_to_pose("left_arm", over, steps=36)
    ctx.kinematics.move_to_pose("left_arm", high, steps=20)
    ctx.step(4)
    ctx.kinematics.move_to_pose("left_arm", ready, steps=16)
    ctx.step(4)
    for i in range(1, 17 if outboard else 23):
        depth = (0.0012 if outboard else 0.001) * i
        ctx.kinematics.move_to_pose("left_arm", press_pose(top - depth), steps=8 if outboard else 4)
        ctx.step(4)
        if ctx.world.joint_value(_PROJECTOR, joint) <= -0.0022:
            ctx.step(8)
            break
    ctx.step(16)
    ctx.kinematics.move_to_pose("left_arm", high, steps=16)


def _align_pitch(ctx, base: Pose3D, target: float, low: float, high: float):
    pitch = ctx.world.joint_value(_PROJECTOR, "gimbal_pitch")
    if low <= pitch <= high:
        return
    yaw = ctx.world.joint_value(_PROJECTOR, "gimbal_yaw")
    cradle = _cradle(base, yaw)
    app = rotate_vec_by_quat_wxyz(cradle.wxyz, (0.0, 0.63, -0.7766))
    closing = rotate_vec_by_quat_wxyz(cradle.wxyz, _X)
    front = 0.004 if pitch < target else 0.027
    push = 0.006 if pitch < target else -0.006

    def handle(angle, travel):
        local = (0.0, front + travel * push, 0.061)
        tip = compose_pose(_head(base, yaw, angle), Pose3D(local)).xyz
        return _tip_to_tcp(tip, app, closing)

    start = handle(pitch, 0.0)
    ctx.kinematics.move_to_pose("left_arm", _along(start, -0.048), steps=28)
    ctx.kinematics.move_to_pose("left_arm", start, steps=20)
    end = start
    for i in range(1, 14):
        u = i / 13
        end = handle(pitch + (target - pitch) * u, u)
        ctx.kinematics.move_to_pose("left_arm", end, steps=8)
    ctx.step(12)
    ctx.kinematics.move_to_pose("left_arm", _along(end, -0.042), steps=24)


def _open_shutter(ctx, base: Pose3D):
    app = (0.0, 0.45, -0.893)
    closing = (0.0, -1.0, 0.0)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    for stroke in range(2):
        q = ctx.world.joint_value(_PROJECTOR, "lens_shutter")
        # The free yoke turns with the first push. Approach the handle again
        # from its left in the new head frame to finish the remaining travel.
        start_q = q if stroke == 0 else min(q, 0.030)
        head = _live_head(ctx, base)
        edge = compose_pose(head, Pose3D((-0.074 + start_q, -0.1195, 0.014))).xyz
        slide = rotate_vec_by_quat_wxyz(head.wxyz, _X)
        start = _tip_to_tcp(_offset(edge, app, 0.004), app, closing)
        ctx.kinematics.move_to_pose("left_arm", _along(start, -0.050), steps=20)
        ctx.step(4)
        ctx.kinematics.move_to_pose("left_arm", start, steps=16)
        end = start
        for i in range(1, 17):
            end = Pose3D(_offset(start.xyz, slide, (0.098 - start_q) * i / 16), start.wxyz)
            ctx.kinematics.move_to_pose("left_arm", end, steps=4)
            if stroke == 0 and i >= 14 and ctx.world.joint_value(_PROJECTOR, "lens_shutter") >= 0.066:
                break
        ctx.kinematics.move_to_pose("left_arm", _along(end, -0.055), steps=16)


def _set_focus(ctx, base: Pose3D):
    # Sweep upward on the left flank of the ring, clear of the shutter handle.
    app = (0.58, 0.55, -0.601)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    for _ in range(7):
        if ctx.world.joint_value(_PROJECTOR, "focus_ring") >= 0.63:
            break
        head = _live_head(ctx, base)

        def ring_pose(radius, angle):
            tip = (
                -0.042 - radius * math.cos(angle), -0.100, 0.007 + radius * math.sin(angle)
            )
            return _tip_to_tcp(compose_pose(head, Pose3D(tip)).xyz, app, _Z)

        angle = -0.45
        outside = ring_pose(0.068, angle)
        ctx.kinematics.move_to_pose("left_arm", _along(outside, -0.040), steps=16)
        ctx.kinematics.move_to_pose("left_arm", outside, steps=8)
        ctx.kinematics.move_to_pose("left_arm", ring_pose(0.0335, angle), steps=8)
        for i in range(1, 13):
            angle = -0.45 + 0.90 * i / 12
            ctx.kinematics.move_to_pose("left_arm", ring_pose(0.0335, angle), steps=4)
            if ctx.world.joint_value(_PROJECTOR, "focus_ring") >= 0.63:
                break
        outside = ring_pose(0.068, angle)
        ctx.kinematics.move_to_pose("left_arm", outside, steps=8)
        ctx.kinematics.move_to_pose("left_arm", _along(outside, -0.040), steps=16)


def _press_speaker(ctx, base: Pose3D, joint: str, x: float):
    # Tilt toward the outside of the deck to lower one open fingertip into the key well.
    sign = math.copysign(1.0, x)
    app = (sign * 0.45, 0.10, -0.8874)

    def hand(z):
        return compose_pose(base, _fingertip_tcp((x, 0.0, z), app, _X, -sign * 0.044))

    high = hand(0.249)
    ctx.kinematics.move_to_pose("right_arm", high, steps=24)
    ctx.end_effector.set_posture("right_gripper", "open", steps=24)
    ctx.kinematics.move_to_pose("right_arm", hand(0.1985), steps=20)
    ctx.step(8)
    for i in range(1, 12):
        ctx.kinematics.move_to_pose("right_arm", hand(0.1985 - 0.0008 * i), steps=4)
        ctx.step(8)
        if -0.0028 <= ctx.world.joint_value(_SPEAKER, joint) <= -0.0017:
            ctx.step(16)
            break
    ctx.kinematics.move_to_pose("right_arm", high, steps=20)


def _turn_crown(ctx, base: Pose3D):
    # Pinch a chord outside the key deck and rotate about the crown bearing.
    pivot = compose_pose(base, Pose3D((0.0, 0.0, 0.187)))
    sector = Pose3D((0.0, 0.0, 0.0), quat_from_axis_angle_wxyz(_Z, -1.60))
    hand = compose_pose(
        sector, _tip_to_tcp((0.065, 0.0, 0.0055), (-0.796, 0.0, -0.605), _Y)
    )
    start = compose_pose(pivot, hand)
    for _ in range(2):
        if ctx.world.joint_value(_SPEAKER, "volume_crown") >= 0.88:
            break
        ctx.end_effector.set_posture("right_gripper", "open", steps=20)
        ctx.kinematics.move_to_pose("right_arm", _along(start, -0.050), steps=24)
        ctx.step(4)
        ctx.kinematics.move_to_pose("right_arm", start, steps=16)
        ctx.step(8)
        ctx.end_effector.set_posture("right_gripper", "closed", steps=36)
        ctx.step(8)
        q = ctx.world.joint_value(_SPEAKER, "volume_crown")
        for i in range(1, 15):
            rotation = quat_from_axis_angle_wxyz(_Z, (1.00 - q) * i / 14)
            end = compose_pose(pivot, compose_pose(Pose3D((0.0, 0.0, 0.0), rotation), hand))
            ctx.kinematics.move_to_pose("right_arm", end, steps=8)
            if ctx.world.joint_value(_SPEAKER, "volume_crown") >= 0.88:
                break
        ctx.step(8)
        ctx.end_effector.set_posture("right_gripper", "open", steps=20)
        ctx.step(12)
        ctx.kinematics.move_to_pose("right_arm", _along(end, -0.050), steps=24)


def _open_port(ctx, base: Pose3D) -> None:
    # The upper inside corner of the lower open jaw fits beneath the lip.
    corner = (-0.0305, 0.05245, -0.05343)

    def move_corner(point, heading, steps):
        angle = math.radians(heading)
        app = (0.92718 * math.cos(angle), 0.92718 * math.sin(angle), -0.37461)
        closing = rotate_vec_by_quat_wxyz(
            quat_from_axis_angle_wxyz(app, math.radians(3.0)), (0.0, 0.0, -1.0)
        )
        palm = Pose3D.from_approach(xyz=point, approach_dir=app, closing_dir=closing)
        offset = rotate_vec_by_quat_wxyz(palm.wxyz, corner)
        hand = Pose3D(_offset(point, offset, -1.0), palm.wxyz)
        ctx.kinematics.move_to_pose("right_arm", compose_pose(base, hand), steps=steps)

    ctx.end_effector.set_posture("right_gripper", "open", steps=20)
    x = 0.016
    move_corner((0.056, 0.072, 0.084), 153.0, 32)
    # A light first pull lifts the lower edge off the sill. Re-engaging from
    # below lets the jaw carry a partially opened flap through the full arc.
    strokes = (
        (0.0758, 0.0481, 19),
        (0.0768, 0.0490, 19),
        (0.0758, 0.0481, 12),
        (0.0768, 0.0492, 19),
    )
    for stroke, (y0, height, count) in enumerate(strokes):
        move_corner((0.044, y0, 0.043), 157.0, 20)
        move_corner((x, y0, 0.043), 165.0, 20)
        for k in range(1, 9):
            move_corner((x, y0, 0.043 + (height - 0.043) * k / 8), 165.0, 8)
        for k in range(1, count + 1):
            angle = 1.12 * k / 19
            y = y0 + 0.027 * math.sin(angle)
            z = height + 0.027 * (1.0 - math.cos(angle))
            move_corner((x, y, z), 165.0, 8)
            if stroke < 2:
                opening = ctx.world.joint_value(_SPEAKER, "acoustic_port")
                if opening > 1.02 or (k >= 12 and angle - opening > 0.42):
                    break
        ctx.step(8)
        move_corner((x + 0.026, y + 0.003, z - 0.006), 157.0, 16)


def _set_privacy(ctx, base: Pose3D):
    # Push the slider with the lower open fingertip, keeping the upper jaw above the crown.
    app = (0.8518, 0.04, -0.5223)
    closing = (0.4697, -0.4996, 0.7279)

    def hand(tip):
        return compose_pose(base, _fingertip_tcp(tip, app, closing, -0.044))

    for tip in ((-0.105, -0.035, 0.224), (-0.104, 0.055, 0.203)):
        clear = compose_pose(base, _tip_to_tcp(tip, (0.0, 0.45, -0.893), _X))
        ctx.kinematics.move_to_pose("right_arm", clear, steps=24)
    ctx.kinematics.move_to_pose("right_arm", hand((-0.039, 0.085, 0.148)), steps=28)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "open", steps=24)
    ctx.kinematics.move_to_pose("right_arm", hand((-0.039, 0.086, 0.138)), steps=20)
    ctx.step(8)
    ctx.kinematics.move_to_pose("right_arm", hand((-0.039, 0.079, 0.138)), steps=16)
    ctx.step(8)
    x = -0.039
    for i in range(1, 16):
        x = -0.039 + 0.058 * i / 15
        ctx.kinematics.move_to_pose("right_arm", hand((x, 0.079, 0.138)), steps=8)
        ctx.step(4)
        if ctx.world.joint_value(_SPEAKER, "mic_privacy") >= 0.019:
            break
    ctx.kinematics.move_to_pose("right_arm", hand((x, 0.085, 0.146)), steps=20)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=20)
    home = ctx.sensors.proprio()
    home_l, home_r = tuple(home[:6]), tuple(home[8:14])

    projector = ctx.world.object_pose(_PROJECTOR)
    _press_projector(ctx, projector, "power_button", 0.047)
    ctx.kinematics.move_joints("left_arm", home_l, steps=24)
    _open_shutter(ctx, projector)
    ctx.kinematics.move_joints("left_arm", home_l, steps=24)
    _align_pitch(ctx, projector, 0.32, 0.27, 0.43)
    ctx.kinematics.move_joints("left_arm", home_l, steps=24)
    _set_focus(ctx, projector)
    ctx.kinematics.move_joints("left_arm", home_l, steps=24)
    _press_projector(ctx, projector, "keystone_lock", -0.052)
    ctx.kinematics.move_joints("left_arm", home_l, steps=24)

    speaker = ctx.world.object_pose(_SPEAKER)
    _press_speaker(ctx, speaker, "wake_touch", -0.022)
    _press_speaker(ctx, speaker, "source_select", 0.022)
    _turn_crown(ctx, speaker)
    ctx.kinematics.move_joints("right_arm", home_r, steps=16)
    _open_port(ctx, speaker)
    ctx.kinematics.move_joints("right_arm", home_r, steps=16)
    _set_privacy(ctx, speaker)
    ctx.kinematics.move_joints("right_arm", home_r, steps=20)
    ctx.step(16)
