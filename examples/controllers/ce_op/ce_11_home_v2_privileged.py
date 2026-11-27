"""Set the speaker keys, crown, rear acoustic flap and privacy switch."""

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
_DOWN = (0.0, 0.15, -0.9887)
_SPEAKER = "smart_speaker"


def _offset(point: Vec3, direction: Vec3, distance: float) -> Vec3:
    return (
        point[0] + distance * direction[0],
        point[1] + distance * direction[1],
        point[2] + distance * direction[2],
    )


def _tip_to_tcp(xyz: Vec3, approach: Vec3, closing: Vec3) -> Pose3D:
    app = normalize_vec3(approach)
    return Pose3D.from_approach(
        xyz=_offset(xyz, app, -0.0226), approach_dir=app, closing_dir=closing
    )


def _along(pose: Pose3D, distance: float) -> Pose3D:
    return compose_pose(pose, Pose3D((0.0, 0.0, distance)))


def _fingertip_tcp(tip: Vec3, approach: Vec3, closing: Vec3, finger_offset: float) -> Pose3D:
    pose = _tip_to_tcp(tip, approach, closing)
    jaw_axis = rotate_vec_by_quat_wxyz(pose.wxyz, _Y)
    return Pose3D(_offset(pose.xyz, jaw_axis, -finger_offset), pose.wxyz)


def _approach(ctx, arm: str, contact: Pose3D, distance: float = 0.080) -> None:
    ctx.kinematics.move_to_pose(arm, _along(contact, -distance), steps=28)
    ctx.step(4)
    ctx.kinematics.move_to_pose(arm, contact, steps=16)
    ctx.step(4)


def _press(ctx, base: Pose3D, side: str, joint: str, x: float) -> None:
    arm = side + "_arm"
    # Descend past compliance into -0.0022 latch window without clipping the 21.6 mm well rim.
    cap = _tip_to_tcp((x, 0.0, 0.1945), _DOWN, _X)
    ctx.end_effector.set_posture(side + "_gripper", "closed", steps=24)
    _approach(ctx, arm, compose_pose(base, cap))
    end = compose_pose(base, cap)
    for k in range(1, 13):
        end = compose_pose(base, compose_pose(Pose3D((0.0, 0.0, -0.0110 * k / 12)), cap))
        ctx.kinematics.move_to_pose(arm, end, steps=4)
        ctx.step(4)
        if ctx.world.joint_value("smart_speaker", joint) <= -0.0022:
            ctx.step(8)
            break
    ctx.step(20)
    ctx.kinematics.move_to_pose(arm, _along(end, -0.080), steps=20)


def _turn_crown(ctx, base: Pose3D) -> None:
    # Take a chord from the outside of the 146.9 mm crown; stay well inside [0.7069, 1.0472].
    pivot = compose_pose(base, Pose3D((0.0, 0.0, 0.187)))
    sector = Pose3D((0.0, 0.0, 0.0), quat_from_axis_angle_wxyz(_Z, -1.45))
    hand = compose_pose(
        sector, _tip_to_tcp((0.065, 0.0, 0.0055), (-0.94, 0.0, -0.342), (0.0, 1.0, 0.0))
    )
    ctx.end_effector.set_posture("right_gripper", "open", steps=20)
    start = compose_pose(pivot, hand)
    _approach(ctx, "right_arm", start)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=36)
    ctx.step(8)
    q = ctx.world.joint_value(_SPEAKER, "volume_crown")
    end = start
    for i in range(1, 15):
        angle = (1.00 - q) * i / 14
        turn = Pose3D((0.0, 0.0, 0.0), quat_from_axis_angle_wxyz(_Z, angle))
        end = compose_pose(pivot, compose_pose(turn, hand))
        ctx.kinematics.move_to_pose("right_arm", end, steps=8)
        if ctx.world.joint_value(_SPEAKER, "volume_crown") >= 0.88:
            break
    ctx.step(4)
    ctx.end_effector.set_posture("right_gripper", "open", steps=20)
    ctx.step(12)
    ctx.kinematics.move_to_pose("right_arm", _along(end, -0.050), steps=24)


def _open_port(ctx, base: Pose3D) -> None:
    arm = "right_arm"
    gripper = "right_gripper"
    # TCP relative to the upper inside corner of the fully open lower finger.
    corner_to_tcp = _fingertip_tcp(
        (-0.04379, 0.06905, -0.00984),
        (-0.8000, 0.5650, -0.2019),
        (0.2400, -0.2800, -0.9295),
        0.044,
    )

    def corner_pose(x: float, y: float, z: float) -> Pose3D:
        contact = compose_pose(base, Pose3D((x, y, z)))
        return compose_pose(contact, corner_to_tcp)

    def move_corner(x: float, y: float, z: float, ticks: int) -> None:
        ctx.kinematics.follow_cartesian_path(
            arm, [corner_pose(x, y, z)], steps_per_waypoint=ticks
        )

    def set_gap(start: float, stop: float) -> None:
        for tick in range(1, 5):
            gap = start + (stop - start) * tick / 4
            ctx.end_effector.command_targets(gripper, (gap, gap), steps=1)
        ctx.step(4)

    # Start the ramps one physics tick after a recorded target, so their
    # endpoints fall on the 15 Hz samples that replay interpolates between.
    ctx.step(1)
    ctx.end_effector.set_posture(gripper, "open", steps=20)
    ctx.kinematics.move_to_pose(arm, corner_pose(0.051, 0.094, 0.082), steps=32)
    move_corner(0.039, 0.082, 0.0435, 24)
    # Each pull starts underneath the preceding contact. The jaw supplies
    # upward pressure while the wrist draws it along the 29 mm hinge radius.
    x = 0.0135
    for start in (0.0, 0.025, 0.085, 0.15, 0.22, 0.30, 0.39, 0.49, 0.60, 0.68):
        radial_y = -0.0013 - 0.004 * min(start, 0.4)

        def arc(angle: float) -> tuple[float, float]:
            sn, cs = math.sin(angle), math.cos(angle)
            # At larger angles, lift from the inner face with the wrist in reach.
            y = min(0.092, 0.076 + radial_y * cs + 0.029 * sn)
            z = 0.045 + radial_y * sn + 0.029 * (1.0 - cs)
            return y, z

        y, z = arc(start)
        move_corner(x, y, z - 0.0035, 16)
        move_corner(x, y, z, 12)
        set_gap(0.044, 0.014)
        for k in range(13):
            y, z = arc(start + k * 0.0105)
            move_corner(x, y, z, 8)
        set_gap(0.014, 0.044)
        move_corner(x + 0.001, y, z - 0.0035, 8)

    move_corner(0.047, 0.096, 0.052, 24)
    move_corner(0.055, 0.094, 0.084, 20)
    ctx.step(3)


def _mute(ctx, base: Pose3D) -> None:
    arm = "left_arm"
    # A shallow sweep engages the thumb grip from the side while clearing the crown.
    app = (0.97, 0.08, -0.2296)
    y, z = 0.0822, 0.138
    ctx.end_effector.set_posture("left_gripper", "closed", steps=20)
    end = compose_pose(base, _tip_to_tcp((-0.034, y, z), app, _Z))
    _approach(ctx, arm, end, 0.044)
    for i in range(1, 17):
        end = compose_pose(base, _tip_to_tcp((-0.034 + 0.050 * i / 16, y, z), app, _Z))
        ctx.kinematics.move_to_pose(arm, end, steps=8)
        if ctx.world.joint_value(_SPEAKER, "mic_privacy") >= 0.0185:
            break
    ctx.step(16)
    ctx.kinematics.move_to_pose(arm, _along(end, -0.055), steps=24)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=20)
    home = ctx.sensors.proprio()
    base = ctx.world.object_pose(_SPEAKER)
    _press(ctx, base, "left", "wake_touch", -0.022)
    ctx.kinematics.move_joints("left_arm", tuple(home[:6]), steps=28)
    _press(ctx, base, "right", "source_select", 0.022)
    _turn_crown(ctx, base)
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    ctx.kinematics.move_joints("right_arm", tuple(home[8:14]), steps=16)
    _open_port(ctx, base)
    ctx.kinematics.move_joints("right_arm", tuple(home[8:14]), steps=28)
    _mute(ctx, base)
    ctx.kinematics.move_joints("left_arm", tuple(home[:6]), steps=28)
    ctx.step(40)
