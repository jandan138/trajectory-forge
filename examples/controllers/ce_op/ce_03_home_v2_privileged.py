import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    compose_pose,
    invert_pose,
    quat_from_axis_angle_wxyz,
    rotate_vec_by_quat_wxyz,
)

_FINGER_PAD_M = 0.0226


def _tip_to_tcp(
    tip_xyz: tuple[float, float, float],
    approach_dir: tuple[float, float, float],
    closing_dir: tuple[float, float, float],
) -> Pose3D:
    ax, ay, az = approach_dir
    norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    ux, uy, uz = ax / norm, ay / norm, az / norm
    pad_ext = _FINGER_PAD_M
    return Pose3D.from_approach(
        xyz=(tip_xyz[0] - pad_ext * ux, tip_xyz[1] - pad_ext * uy, tip_xyz[2] - pad_ext * uz),
        approach_dir=(ux, uy, uz),
        closing_dir=closing_dir,
    )


def _yaw_pose(pose: Pose3D) -> Pose3D:
    x_axis = rotate_vec_by_quat_wxyz(pose.wxyz, (1.0, 0.0, 0.0))
    yaw = math.atan2(x_axis[1], x_axis[0])
    return Pose3D(pose.xyz, quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), yaw))


def _lid_tcp_for_seat(ctx, kettle: Pose3D, height: float, nominal_tcp: Pose3D) -> Pose3D:
    lid = ctx.world.object_pose("infuser_lid")
    tcp = ctx.kinematics.ee_pose("right_arm")
    if math.dist(lid.xyz, tcp.xyz) > 0.110:
        return nominal_tcp
    lid_in_kettle = compose_pose(invert_pose(kettle), lid)
    x_axis = rotate_vec_by_quat_wxyz(lid_in_kettle.wxyz, (1.0, 0.0, 0.0))
    # Center the square basket corners between adjacent staves.
    spacing = 2.0 * math.pi / 32
    yaw = (round(math.atan2(x_axis[1], x_axis[0]) / spacing - 0.5) + 0.5) * spacing
    lid_seat = compose_pose(
        kettle, Pose3D((0.0, 0.0, height), quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), yaw))
    )
    held_lid = compose_pose(invert_pose(lid), tcp)
    return compose_pose(lid_seat, held_lid)


def _move_lid_to(ctx, kettle: Pose3D, height: float, nominal_tcp: Pose3D) -> None:
    tcp = _lid_tcp_for_seat(ctx, kettle, height, nominal_tcp)
    try:
        ctx.kinematics.move_to_pose("right_arm", tcp, steps=48)
    except RuntimeError:
        ctx.kinematics.move_to_pose("right_arm", nominal_tcp, steps=48)


def _ramp_jaws(ctx, group: str, start: float, end: float, ticks: int = 32) -> None:
    for i in range(1, ticks + 1):
        width = start + (end - start) * (i / ticks)
        ctx.end_effector.command_targets(group, (width, width), steps=1)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=24)
    # Action index zero is sampled. End each later ramp at an index divisible by eight.
    ctx.step(17)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    x_cls = (1.0, 0.0, 0.0)
    btn_dir = (0.0, 0.22, -0.9755)

    socket = ctx.world.named_frame("kettle_dock", "dock_socket")
    dial_f = ctx.world.named_frame("kettle_dock", "temp_select_dial").xyz
    warm_f = ctx.world.named_frame("kettle_dock", "keep_warm_slide").xyz
    boil_f = ctx.world.named_frame("kettle_dock", "boil_button").xyz

    # Straddle the handle stem below the top bar. Keep the inner finger in the handle.
    kettle = ctx.world.object_pose("kettle_vessel")
    upright = _yaw_pose(kettle)
    approach = rotate_vec_by_quat_wxyz(upright.wxyz, (-0.18, 0.869, -0.459))
    closing = rotate_vec_by_quat_wxyz(upright.wxyz, x_cls)
    center = compose_pose(upright, Pose3D((0.108, 0.022, 0.089))).xyz
    tip = _tip_to_tcp(center, approach, closing)
    above = compose_pose(tip, Pose3D((0.0, 0.0, -0.075)))
    ctx.end_effector.command_targets("left_gripper", (0.03, 0.03), steps=24)
    ctx.kinematics.move_to_pose("left_arm", Pose3D((above.xyz[0], above.xyz[1], above.xyz[2] + 0.050), tip.wxyz), steps=56)
    ctx.kinematics.move_to_pose("left_arm", above, steps=40)
    ctx.kinematics.move_to_pose("left_arm", tip, steps=64)
    ctx.step(16)
    _ramp_jaws(ctx, "left_gripper", 0.03, 0.0, ticks=48)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=32)
    # Ease the loaded handle into and out of the lift at the replay sample points.
    lift_poses = []
    for frame in range(1, 26):
        t = frame / 25.0
        height = 0.075 * t * t * (3.0 - 2.0 * t)
        lift_poses.append(Pose3D((tip.xyz[0], tip.xyz[1], tip.xyz[2] + height), tip.wxyz))
    ctx.kinematics.follow_cartesian_path("left_arm", lift_poses, steps_per_waypoint=8)
    ctx.step(32)
    dock_pose = ctx.world.object_pose("kettle_dock")
    sock_target = socket.xyz
    # Recompute the grasp transform under load before each level descent.
    for height, min_steps in ((0.034, 240), (0.010, 64), (0.001, 72)):
        kettle = ctx.world.object_pose("kettle_vessel")
        tcp = ctx.kinematics.ee_pose("left_arm")
        held = compose_pose(invert_pose(kettle), tcp)
        destination = Pose3D((sock_target[0], sock_target[1], sock_target[2] + height), _yaw_pose(kettle).wxyz)
        dest_tcp = compose_pose(destination, held)
        # Budget the loaded translation at 0.08 m/s, rounded up to a full camera period.
        periods = math.ceil(math.dist(tcp.xyz, dest_tcp.xyz) * 15.0 / 0.08)
        ctx.kinematics.move_to_pose("left_arm", dest_tcp, steps=max(min_steps, 8 * periods))
        ctx.step(24)
    seat_tcp = ctx.kinematics.ee_pose("left_arm")
    _ramp_jaws(ctx, "left_gripper", 0.0, 0.03, ticks=64)
    ctx.step(32)
    retreat = compose_pose(seat_tcp, Pose3D((0.0, 0.0, -0.100)))
    ctx.kinematics.move_to_pose("left_arm", retreat, steps=72)
    ctx.kinematics.move_joints("left_arm", home_l, steps=40)
    ctx.step(8)
    lid_dir = (-0.14, 0.25, -0.958)
    ctx.end_effector.set_posture("right_gripper", "open", steps=24)
    lid = ctx.world.object_pose("infuser_lid")
    knob_xyz = compose_pose(lid, Pose3D((0.0, 0.0, 0.0415))).xyz
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((knob_xyz[0], knob_xyz[1], knob_xyz[2] + 0.120), lid_dir, x_cls), steps=48)
    ctx.step(8)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(knob_xyz, lid_dir, x_cls), steps=48)
    ctx.step(16)
    _ramp_jaws(ctx, "right_gripper", 0.044, 0.0)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=32)
    ctx.step(16)

    kv_now = ctx.world.object_pose("kettle_vessel")
    if math.dist(kv_now.xyz, sock_target) > 0.045:
        kv_now = Pose3D(sock_target, _yaw_pose(dock_pose).wxyz)
    lid_target = compose_pose(kv_now, Pose3D((0.0, 0.0, 0.122 + 0.0415))).xyz
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((knob_xyz[0], knob_xyz[1], lid_target[2] + 0.065), lid_dir, x_cls), steps=56)
    ctx.step(16)
    above_rim = _tip_to_tcp((lid_target[0], lid_target[1], lid_target[2] + 0.055), lid_dir, x_cls)
    above_rim_joints = ctx.kinematics.move_to_pose("right_arm", above_rim, steps=96)
    ctx.step(24)
    nominal = _tip_to_tcp((lid_target[0], lid_target[1], lid_target[2] + 0.028), lid_dir, x_cls)
    _move_lid_to(ctx, kv_now, 0.155, nominal)
    ctx.step(16)
    # Let the centered basket settle under its own weight at the 143 mm rim.
    rim = _tip_to_tcp((lid_target[0], lid_target[1], lid_target[2] + 0.024), lid_dir, x_cls)
    _move_lid_to(ctx, kv_now, 0.146, rim)
    ctx.step(16)
    _ramp_jaws(ctx, "right_gripper", 0.0, 0.044)
    ctx.end_effector.set_posture("right_gripper", "open", steps=32)
    # Retrace to the reachable approach configuration after clearing the knob.
    ctx.kinematics.move_joints("right_arm", above_rim_joints, steps=56)
    ax, ay, az = above_rim.xyz
    ctx.kinematics.move_to_pose("right_arm", Pose3D((ax + 0.036, ay - 0.045, az), above_rim.wxyz), steps=40)

    cx, cy, cz = dial_f[0], dial_f[1] + 0.017, dial_f[2] - 0.001
    rim_dir = (0.0, 0.12, -0.9928)
    r_c, r_out = 0.0218, 0.032
    ctx.end_effector.set_posture("right_gripper", "closed", steps=24)
    for _stroke in range(3):
        if ctx.world.joint_value("kettle_dock", "temp_select_dial") >= 0.84:
            break
        th0 = -1.15
        cls0 = (-math.sin(th0), math.cos(th0), 0.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_out * math.cos(th0), cy + r_out * math.sin(th0), cz + 0.045), rim_dir, cls0), steps=24)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_out * math.cos(th0), cy + r_out * math.sin(th0), cz), rim_dir, cls0), steps=24)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_c * math.cos(th0), cy + r_c * math.sin(th0), cz), rim_dir, cls0), steps=16)
        th_end = th0
        for k in range(1, 13):
            th = -1.15 + 1.45 * (k / 12.0)
            th_end = th
            t_cls = (-math.sin(th), math.cos(th), 0.0)
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_c * math.cos(th), cy + r_c * math.sin(th), cz), rim_dir, t_cls), steps=16)
            if ctx.world.joint_value("kettle_dock", "temp_select_dial") >= 0.84:
                break
        cls_end = (-math.sin(th_end), math.cos(th_end), 0.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_out * math.cos(th_end), cy + r_out * math.sin(th_end), cz), rim_dir, cls_end), steps=16)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx + r_out * math.cos(th_end), cy + r_out * math.sin(th_end), cz + 0.045), rim_dir, cls_end), steps=24)

    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    wx0, wy, wz = warm_f[0] - 0.014, warm_f[1], warm_f[2]
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((wx0, wy - 0.065, wz + 0.105), btn_dir, x_cls), steps=40)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((wx0, wy - 0.025, wz + 0.025), btn_dir, x_cls), steps=24)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((wx0, wy, wz), btn_dir, x_cls), steps=24)
    wx = wx0
    for k in range(1, 10):
        wx = wx0 + 0.020 * (k / 9.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((wx, wy, wz), btn_dir, x_cls), steps=16)
        if ctx.world.joint_value("kettle_dock", "keep_warm_slider") >= 0.0112:
            break
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((wx, wy - 0.025, wz + 0.025), btn_dir, x_cls), steps=24)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((wx, wy - 0.065, wz + 0.105), btn_dir, x_cls), steps=32)
    ctx.kinematics.move_joints("left_arm", home_l, steps=32)

    ctx.end_effector.set_posture("right_gripper", "closed", steps=16)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((boil_f[0], boil_f[1], boil_f[2] + 0.075), btn_dir, x_cls), steps=32)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((boil_f[0], boil_f[1], boil_f[2] + 0.014), btn_dir, x_cls), steps=32)
    ctx.step(8)
    for k in range(1, 18):
        bz = boil_f[2] + 0.006 - 0.0009 * k
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((boil_f[0], boil_f[1], bz), btn_dir, x_cls), steps=16)
        ctx.step(8)
        if ctx.world.joint_value("kettle_dock", "boil_button") <= -0.0022:
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((boil_f[0], boil_f[1], bz - 0.0008), btn_dir, x_cls), steps=16)
            break
    ctx.step(64)
