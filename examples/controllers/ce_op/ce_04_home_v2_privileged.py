import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    compose_pose,
    invert_pose,
    quat_from_axis_angle_wxyz,
    rotate_vec_by_quat_wxyz,
)


def _tip_to_tcp(
    tip_xyz: tuple[float, float, float],
    approach_dir: tuple[float, float, float],
    closing_dir: tuple[float, float, float],
) -> Pose3D:
    ax, ay, az = approach_dir
    norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    ux, uy, uz = ax / norm, ay / norm, az / norm
    pad_ext = 0.0226
    return Pose3D.from_approach(
        xyz=(tip_xyz[0] - pad_ext * ux, tip_xyz[1] - pad_ext * uy, tip_xyz[2] - pad_ext * uz),
        approach_dir=(ux, uy, uz),
        closing_dir=closing_dir,
    )


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    x_cls = (1.0, 0.0, 0.0)
    down_dir = (0.0, 0.15, -0.9887)

    ws_pose = ctx.world.object_pose("workstation")
    _, wy, wz = ws_pose.xyz
    bay_f = ctx.world.named_frame("workstation", "backup_bay_slot").xyz
    mouse_f = ctx.world.named_frame("workstation", "mouse_dock").xyz
    shutter_f = ctx.world.named_frame("workstation", "io_shutter_handle").xyz
    enter_f = ctx.world.named_frame("workstation", "keyboard_enter").xyz
    tower_f = ctx.world.named_frame("workstation", "tower_power").xyz
    mon_f = ctx.world.named_frame("workstation", "monitor_power").xyz

    # The trailing-end grasp tilts the wrist away from the tower after the drive turns.
    drive_dir = (-0.22, 0.115, -0.9687)
    for grip_x, grip_height in ((0.030, 0.008), (0.025, 0.0065)):
        ctx.end_effector.set_posture("left_gripper", "open", steps=16)
        drive = ctx.world.object_pose("backup_drive")
        drive_x = rotate_vec_by_quat_wxyz(drive.wxyz, (1.0, 0.0, 0.0))
        yaw = math.atan2(drive_x[1], drive_x[0])
        level_drive = Pose3D(drive.xyz, quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), yaw))
        closing = rotate_vec_by_quat_wxyz(level_drive.wxyz, (0.0, 1.0, 0.0))
        approach = rotate_vec_by_quat_wxyz(level_drive.wxyz, drive_dir)
        tip = compose_pose(level_drive, Pose3D((grip_x, 0.0, grip_height))).xyz
        grip = _tip_to_tcp(tip, approach, closing)
        ctx.kinematics.move_to_pose(
            "left_arm", compose_pose(grip, Pose3D((0.0, 0.0, -0.105))), steps=24
        )
        ctx.kinematics.move_to_pose("left_arm", grip, steps=20)
        ctx.step(8)
        ctx.end_effector.set_posture("left_gripper", "closed", steps=36)
        ctx.step(8)
        grip = ctx.kinematics.ee_pose("left_arm")
        ctx.kinematics.move_to_pose(
            "left_arm", Pose3D((grip.xyz[0], grip.xyz[1], grip.xyz[2] + 0.085), grip.wxyz), steps=24
        )
        if ctx.world.object_pose("backup_drive").xyz[2] >= wz + 0.045:
            break
    else:
        return

    drive = ctx.world.object_pose("backup_drive")
    held = compose_pose(invert_pose(drive), ctx.kinematics.ee_pose("left_arm"))
    carry_z = wz + 0.100
    approach_y = wy - 0.170
    front_y = wy - 0.120
    crosswise = quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), -math.pi / 2)
    ctx.kinematics.move_to_pose(
        "left_arm", compose_pose(Pose3D((drive.xyz[0], approach_y, carry_z), drive.wxyz), held), steps=24
    )
    ctx.kinematics.move_to_pose(
        "left_arm", compose_pose(Pose3D((bay_f[0], approach_y, carry_z), crosswise), held), steps=32
    )
    ctx.step(8)

    # Set the drive on the table to square it before taking the insertion grasp.
    staging_y = wy - 0.154
    ctx.kinematics.move_to_pose(
        "left_arm",
        compose_pose(Pose3D((bay_f[0], staging_y, wz + 0.0025), crosswise), held),
        steps=34,
    )
    ctx.step(16)
    ctx.end_effector.set_posture("left_gripper", "open", steps=24)
    ctx.step(16)
    released = ctx.kinematics.ee_pose("left_arm")
    ctx.kinematics.move_to_pose(
        "left_arm",
        Pose3D((released.xyz[0], released.xyz[1] - 0.008, released.xyz[2] + 0.095), released.wxyz),
        steps=24,
    )

    drive = ctx.world.object_pose("backup_drive")
    drive_x = rotate_vec_by_quat_wxyz(drive.wxyz, (1.0, 0.0, 0.0))
    yaw = math.atan2(drive_x[1], drive_x[0])
    level_drive = Pose3D(drive.xyz, quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), yaw))
    closing = rotate_vec_by_quat_wxyz(level_drive.wxyz, (0.0, 1.0, 0.0))
    approach = rotate_vec_by_quat_wxyz(level_drive.wxyz, (-0.40, 0.0, -0.9165))
    tip = compose_pose(level_drive, Pose3D((0.026, 0.0, 0.0075))).xyz
    grip = _tip_to_tcp(tip, approach, closing)
    ctx.kinematics.move_to_pose(
        "left_arm", compose_pose(grip, Pose3D((0.0, 0.0, -0.063))), steps=24
    )
    ctx.kinematics.move_to_pose("left_arm", grip, steps=28)
    ctx.step(8)
    for opening in (0.035, 0.032, 0.029, 0.024):
        ctx.end_effector.command_targets("left_gripper", (opening, opening), steps=8)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=24)
    ctx.step(16)
    grip = ctx.kinematics.ee_pose("left_arm")
    ctx.kinematics.move_to_pose(
        "left_arm", Pose3D((grip.xyz[0], grip.xyz[1], grip.xyz[2] + 0.048), grip.wxyz), steps=32
    )
    ctx.step(16)

    drive = ctx.world.object_pose("backup_drive")
    if drive.xyz[2] < wz + 0.023:
        return
    held = compose_pose(invert_pose(drive), ctx.kinematics.ee_pose("left_arm"))
    insert_z = bay_f[2] + 0.0020
    ctx.kinematics.move_to_pose(
        "left_arm", compose_pose(Pose3D((bay_f[0], front_y, insert_z), crosswise), held), steps=32
    )
    ctx.step(12)
    for depth in (-0.088, -0.065, -0.049, -0.040):
        ctx.kinematics.move_to_pose(
            "left_arm", compose_pose(Pose3D((bay_f[0], wy + depth, insert_z), crosswise), held), steps=20
        )
        ctx.step(6)

    ctx.end_effector.command_targets("left_gripper", (0.036, 0.036), steps=18)
    ctx.step(12)
    ctx.kinematics.move_to_pose(
        "left_arm", compose_pose(Pose3D((bay_f[0], front_y, insert_z + 0.045), crosswise), held), steps=22
    )
    ctx.kinematics.move_joints("left_arm", home_l, steps=18)

    ctx.end_effector.set_posture("right_gripper", "open", steps=10)
    ms = ctx.world.object_pose("desk_mouse")
    mx_v = rotate_vec_by_quat_wxyz(ms.wxyz, (1.0, 0.0, 0.0))
    if mx_v[0] < 0.0:
        mx_v = (-mx_v[0], -mx_v[1], -mx_v[2])
    ms_cls = (float(mx_v[0]), float(mx_v[1]), 0.0)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ms.xyz[0], ms.xyz[1], ms.xyz[2] + 0.120), down_dir, ms_cls), steps=20)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ms.xyz[0], ms.xyz[1], ms.xyz[2] + 0.014), down_dir, ms_cls), steps=18)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=36)
    ctx.step(8)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ms.xyz[0], ms.xyz[1], wz + 0.165), down_dir, ms_cls), steps=20)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mouse_f[0], mouse_f[1], mouse_f[2] + 0.130), down_dir, x_cls), steps=22)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mouse_f[0], mouse_f[1], mouse_f[2] + 0.018), down_dir, x_cls), steps=18)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "open", steps=20)
    ctx.step(12)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mouse_f[0], mouse_f[1], mouse_f[2] + 0.135), down_dir, x_cls), steps=16)

    enter_dir = (-0.12, 0.04, -0.992)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((enter_f[0], enter_f[1], enter_f[2] + 0.080), enter_dir, x_cls), steps=18)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((enter_f[0], enter_f[1], enter_f[2] + 0.012), enter_dir, x_cls), steps=16)
    ctx.step(4)
    for k in range(1, 18):
        pz = enter_f[2] + 0.006 - 0.014 * (k / 17.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((enter_f[0], enter_f[1], pz), enter_dir, x_cls), steps=5)
        ctx.step(3)
        if ctx.world.joint_value("workstation", "keyboard_enter") <= -0.0022:
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((enter_f[0], enter_f[1], pz - 0.0008), enter_dir, x_cls), steps=6)
            break
    ctx.step(24)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((enter_f[0], enter_f[1], enter_f[2] + 0.095), enter_dir, x_cls), steps=16)

    ctx.end_effector.set_posture("left_gripper", "closed", steps=14)
    sh_dir = (0.0, 0.7880, -0.6157)
    sx0, sy0, sz0 = shutter_f[0] - 0.010, shutter_f[1] + 0.003, shutter_f[2]
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((sx0 - 0.012, sy0 - 0.055, sz0 + 0.030), sh_dir, x_cls), steps=18)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((sx0, sy0, sz0), sh_dir, x_cls), steps=16)
    for k in range(1, 14):
        sx = sx0 + 0.046 * (k / 13.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((sx, sy0, sz0), sh_dir, x_cls), steps=5)
        if ctx.world.joint_value("workstation", "io_shutter") >= -0.006:
            break
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((sx0 + 0.040, sy0 - 0.065, sz0 + 0.065), sh_dir, x_cls), steps=16)

    tower_dir = (0.0, 0.12, -0.9928)
    ty_t = tower_f[1] - 0.0055
    ctx.end_effector.set_posture("left_gripper", "closed", steps=10)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((tower_f[0], ty_t - 0.025, tower_f[2] + 0.080), tower_dir, x_cls), steps=18)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((tower_f[0], ty_t, tower_f[2] + 0.012), tower_dir, x_cls), steps=16)
    ctx.step(4)
    for k in range(1, 18):
        pz = tower_f[2] + 0.008 - 0.015 * (k / 17.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((tower_f[0], ty_t, pz), tower_dir, x_cls), steps=5)
        ctx.step(3)
        if ctx.world.joint_value("workstation", "tower_power") <= -0.0022:
            ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((tower_f[0], ty_t, pz - 0.0008), tower_dir, x_cls), steps=6)
            break
    ctx.step(24)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((tower_f[0], ty_t - 0.040, tower_f[2] + 0.095), tower_dir, x_cls), steps=16)
    ctx.kinematics.move_joints("left_arm", home_l, steps=16)

    mon_dir = (0.0, 0.65, -0.7600)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mon_f[0], mon_f[1] - 0.055, mon_f[2] + 0.065), mon_dir, x_cls), steps=18)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mon_f[0], mon_f[1] - 0.015, mon_f[2] + 0.025), mon_dir, x_cls), steps=14)
    for k in range(1, 18):
        pz = mon_f[2] + 0.008 - 0.015 * (k / 17.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mon_f[0], mon_f[1], pz), mon_dir, x_cls), steps=4)
        ctx.step(3)
        if ctx.world.joint_value("workstation", "monitor_power") <= -0.0022:
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mon_f[0], mon_f[1], pz - 0.0008), mon_dir, x_cls), steps=6)
            break
    ctx.step(64)
