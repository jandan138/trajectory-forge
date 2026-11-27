import math

from trajectory_forge.sdk.math3d import Pose3D, rotate_vec_by_quat_wxyz


def _tip_to_tcp(
    tip_xyz: tuple[float, float, float],
    approach_dir: tuple[float, float, float],
    closing_dir: tuple[float, float, float],
) -> Pose3D:
    ax, ay, az = approach_dir
    norm = math.sqrt(ax * ax + ay * ay + az * az)
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
    home_r = tuple(ctx.sensors.proprio()[8:14])
    x_cls = (1.0, 0.0, 0.0)

    # Stage 0: Left arm depresses keyboard Enter key (`enter_key >= 0.0028 m`)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=12)
    ep = ctx.world.named_frame("conference_pc_rig", "enter_press")
    enter_dir = (0.0, 0.2588190, -0.9659258)
    e_pre = _tip_to_tcp((ep.xyz[0], ep.xyz[1] - 0.020, ep.xyz[2] + 0.070), enter_dir, x_cls)
    ctx.kinematics.move_to_pose("left_arm", e_pre, steps=26)
    for i in range(1, 16):
        dz = 0.006 - 0.017 * (i / 15.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ep.xyz[0], ep.xyz[1] + 0.002, ep.xyz[2] + dz), enter_dir, x_cls), steps=5)
        if ctx.world.joint_value("conference_pc_rig", "enter_key") >= 0.0042:
            break
    ctx.step(6)
    ctx.kinematics.move_to_pose("left_arm", e_pre, steps=18)
    ctx.kinematics.move_joints("left_arm", home_l, steps=18)

    # Stage 1: Right arm presses Sangfor front-right circular silver shutdown button
    ctx.end_effector.set_posture("right_gripper", "closed", steps=12)
    sp = ctx.world.named_frame("conference_pc_rig", "sangfor_power_press")
    angled_dir = (0.0, 0.9396926, -0.3420201)
    sf_pre = _tip_to_tcp((sp.xyz[0], sp.xyz[1] - 0.070, sp.xyz[2] + 0.026), angled_dir, x_cls)
    sf_touch = _tip_to_tcp((sp.xyz[0], sp.xyz[1] - 0.008, sp.xyz[2] + 0.003), angled_dir, x_cls)
    ctx.kinematics.move_to_pose("right_arm", sf_pre, steps=24)
    ctx.kinematics.move_to_pose("right_arm", sf_touch, steps=18)
    for i in range(1, 16):
        dy = -0.008 + 0.020 * (i / 15.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((sp.xyz[0], sp.xyz[1] + dy, sp.xyz[2] + 0.002), angled_dir, x_cls), steps=5)
        if ctx.world.joint_value("conference_pc_rig", "sangfor_power_btn") >= 0.0042:
            break
    ctx.step(6)
    ctx.kinematics.move_to_pose("right_arm", sf_pre, steps=18)

    # Stage 2: Open right gripper + +36mm X offset so L7 presses MonitorPowerButton cleanly
    ctx.end_effector.set_posture("right_gripper", "open", steps=12)
    mp_btn = ctx.world.named_frame("conference_pc_rig", "monitor_power_press")
    mon_x = mp_btn.xyz[0] + 0.036
    mon_pre = _tip_to_tcp((mon_x, mp_btn.xyz[1] - 0.070, mp_btn.xyz[2] + 0.024), angled_dir, x_cls)
    mon_touch = _tip_to_tcp((mon_x, mp_btn.xyz[1] - 0.010, mp_btn.xyz[2] + 0.002), angled_dir, x_cls)
    ctx.kinematics.move_to_pose("right_arm", mon_pre, steps=24)
    ctx.kinematics.move_to_pose("right_arm", mon_touch, steps=18)
    for i in range(1, 28):
        dy = -0.010 + 0.021 * (i / 27.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mon_x, mp_btn.xyz[1] + dy, mp_btn.xyz[2] + 0.002), angled_dir, x_cls), steps=5)
        if 0.0035 <= ctx.world.joint_value("conference_pc_rig", "monitor_power_btn") <= 0.0055:
            break
    ctx.step(6)
    ctx.kinematics.move_to_pose("right_arm", mon_pre, steps=18)

    # Stage 3: Grasp dynamic HP wired mouse & align upright in mouse_parking_zone
    mouse_pose = ctx.world.object_pose("hp_wired_mouse")
    rx, ry, _ = rotate_vec_by_quat_wxyz(mouse_pose.wxyz, (1.0, 0.0, 0.0))
    mouse_local_x = (float(rx), float(ry), 0.0)
    top_dir = (0.0, 0.1391731, -0.9902681)

    m_pre = _tip_to_tcp((mouse_pose.xyz[0], mouse_pose.xyz[1], mouse_pose.xyz[2] + 0.125), top_dir, mouse_local_x)
    m_grasp = _tip_to_tcp((mouse_pose.xyz[0], mouse_pose.xyz[1], mouse_pose.xyz[2] + 0.016), top_dir, mouse_local_x)
    m_lift = _tip_to_tcp((mouse_pose.xyz[0], mouse_pose.xyz[1], mouse_pose.xyz[2] + 0.150), top_dir, mouse_local_x)

    park = ctx.world.named_frame("conference_pc_rig", "mouse_parking_zone")
    p_above = _tip_to_tcp((park.xyz[0], park.xyz[1], park.xyz[2] + 0.135), top_dir, x_cls)
    p_drop = _tip_to_tcp((park.xyz[0], park.xyz[1], 0.768), top_dir, x_cls)
    p_ret = _tip_to_tcp((park.xyz[0], park.xyz[1], park.xyz[2] + 0.145), top_dir, x_cls)

    ctx.kinematics.move_to_pose("right_arm", m_pre, steps=26)
    ctx.kinematics.move_to_pose("right_arm", m_grasp, steps=20)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose("right_arm", m_lift, steps=24)
    ctx.kinematics.move_to_pose("right_arm", p_above, steps=28)
    ctx.kinematics.move_to_pose("right_arm", p_drop, steps=22)
    ctx.end_effector.set_posture("right_gripper", "open", steps=14)
    ctx.step(8)
    ctx.kinematics.move_to_pose("right_arm", p_ret, steps=18)
    ctx.kinematics.move_joints("right_arm", home_r, steps=20)
    ctx.step(10)
