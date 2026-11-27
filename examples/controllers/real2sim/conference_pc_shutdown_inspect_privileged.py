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
    tcp_xyz = (
        tip_xyz[0] - pad_ext * ux,
        tip_xyz[1] - pad_ext * uy,
        tip_xyz[2] - pad_ext * uz,
    )
    return Pose3D.from_approach(
        xyz=tcp_xyz,
        approach_dir=(ux, uy, uz),
        closing_dir=closing_dir,
    )


def _move_base_to(ctx, target_x: float, target_y: float, steps: int = 35) -> None:
    bx, by, _ = ctx.kinematics.base_pose_world().xyz
    ctx.kinematics.move_base_world(target_x - bx, target_y - by, 0.0, steps=steps)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.44, steps=12)
    home_r = tuple(ctx.sensors.proprio()[8:14])

    # ------------------------------------------------------------------
    # Stage 0: Depress HP Keyboard Enter key (`enter_key >= 0.0030 m`)
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.30, -1.00, steps=35)
    ctx.kinematics.set_torso_lift(0.42, steps=10)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=18)

    ep = ctx.world.named_frame("conference_pc_rig", "enter_press")
    enter_dir = (0.0, 0.2588190, -0.9659258)
    x_cls = (1.0, 0.0, 0.0)

    e_pre = _tip_to_tcp((ep.xyz[0], ep.xyz[1] - 0.020, ep.xyz[2] + 0.070), enter_dir, x_cls)
    ctx.kinematics.move_to_pose("right_arm", e_pre, steps=30)
    for i in range(1, 15):
        dz = 0.006 - 0.016 * (i / 14.0)
        wp = _tip_to_tcp((ep.xyz[0], ep.xyz[1] + 0.002, ep.xyz[2] + dz), enter_dir, x_cls)
        ctx.kinematics.move_to_pose("right_arm", wp, steps=6)
        if ctx.world.joint_value("conference_pc_rig", "enter_key") >= 0.0042:
            break
    ctx.step(6)
    ctx.kinematics.move_to_pose("right_arm", e_pre, steps=20)
    print(
        "AFTER_S0",
        round(ctx.world.joint_value("conference_pc_rig", "enter_key"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage0_enter_confirmed")

    # ------------------------------------------------------------------
    # Stage 1: Press Sangfor aDesk front-right circular silver shutdown button
    # Horizontal +Y approach with slight downward pitch (12 deg) well below HKC bezel
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.31, -0.96, steps=24)
    ctx.kinematics.set_torso_lift(0.36, steps=12)

    sp = ctx.world.named_frame("conference_pc_rig", "sangfor_power_press")
    horiz_dir = (0.0, 0.9781476, -0.2079117)
    sf_pre = _tip_to_tcp((sp.xyz[0], sp.xyz[1] - 0.075, sp.xyz[2] + 0.018), horiz_dir, x_cls)
    sf_touch = _tip_to_tcp((sp.xyz[0], sp.xyz[1] - 0.008, sp.xyz[2] + 0.002), horiz_dir, x_cls)
    ctx.kinematics.move_to_pose("right_arm", sf_pre, steps=28)
    ctx.kinematics.move_to_pose("right_arm", sf_touch, steps=20)
    for i in range(1, 15):
        dy = -0.008 + 0.020 * (i / 14.0)
        wp = _tip_to_tcp((sp.xyz[0], sp.xyz[1] + dy, sp.xyz[2] + 0.001), horiz_dir, x_cls)
        ctx.kinematics.move_to_pose("right_arm", wp, steps=6)
        if ctx.world.joint_value("conference_pc_rig", "sangfor_power_btn") >= 0.0042:
            break
    ctx.step(6)
    ctx.kinematics.move_to_pose("right_arm", sf_pre, steps=22)
    print(
        "AFTER_S1",
        round(ctx.world.joint_value("conference_pc_rig", "sangfor_power_btn"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage1_sangfor_shutdown")

    # ------------------------------------------------------------------
    # Stage 2: Press HKC Monitor bottom-right bezel power button
    # Open right_gripper + +36mm X offset so slim left finger (L7) presses button cleanly
    # while right finger (L8) clears the right monitor rim in free air.
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.50, -0.90, steps=26)
    ctx.kinematics.set_torso_lift(0.43, steps=12)
    ctx.end_effector.set_posture("right_gripper", "open", steps=14)

    mp_btn = ctx.world.named_frame("conference_pc_rig", "monitor_power_press")
    mon_dir = (0.0, 0.9781476, -0.2079117)
    mon_x = mp_btn.xyz[0] + 0.036
    mon_pre = _tip_to_tcp((mon_x, mp_btn.xyz[1] - 0.075, mp_btn.xyz[2] + 0.015), mon_dir, x_cls)
    mon_touch = _tip_to_tcp((mon_x, mp_btn.xyz[1] - 0.010, mp_btn.xyz[2]), mon_dir, x_cls)
    ctx.kinematics.move_to_pose("right_arm", mon_pre, steps=28)
    ctx.kinematics.move_to_pose("right_arm", mon_touch, steps=20)
    for i in range(1, 31):
        dy = -0.010 + 0.021 * (i / 30.0)
        wp = _tip_to_tcp((mon_x, mp_btn.xyz[1] + dy, mp_btn.xyz[2]), mon_dir, x_cls)
        ctx.kinematics.move_to_pose("right_arm", wp, steps=5)
        if 0.0035 <= ctx.world.joint_value("conference_pc_rig", "monitor_power_btn") <= 0.0055:
            break
    ctx.step(6)
    ctx.kinematics.move_to_pose("right_arm", mon_pre, steps=20)
    ctx.kinematics.move_joints("right_arm", home_r, steps=20)
    print(
        "AFTER_S2",
        round(ctx.world.joint_value("conference_pc_rig", "monitor_power_btn"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage2_monitor_power_off")

    # ------------------------------------------------------------------
    # Stage 3: Grasp angled HP wired mouse & align upright in standard inspection zone
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.48, -0.94, steps=26)
    ctx.kinematics.set_torso_lift(0.46, steps=12)
    ctx.end_effector.set_posture("right_gripper", "open", steps=10)

    mouse_pose = ctx.world.object_pose("hp_wired_mouse")
    rx, ry, _ = rotate_vec_by_quat_wxyz(mouse_pose.wxyz, (1.0, 0.0, 0.0))
    mouse_local_x = (float(rx), float(ry), 0.0)
    top_dir = (0.0, 0.1391731, -0.9902681)

    m_pre = _tip_to_tcp((mouse_pose.xyz[0], mouse_pose.xyz[1], 0.885), top_dir, mouse_local_x)
    m_grasp = _tip_to_tcp((mouse_pose.xyz[0], mouse_pose.xyz[1], 0.768), top_dir, mouse_local_x)
    m_lift = _tip_to_tcp((mouse_pose.xyz[0], mouse_pose.xyz[1], 0.905), top_dir, mouse_local_x)

    park = ctx.world.named_frame("conference_pc_rig", "mouse_parking_zone")
    p_above = _tip_to_tcp((park.xyz[0], park.xyz[1], 0.905), top_dir, x_cls)
    p_drop = _tip_to_tcp((park.xyz[0], park.xyz[1], 0.768), top_dir, x_cls)
    p_ret = _tip_to_tcp((park.xyz[0], park.xyz[1], 0.910), top_dir, x_cls)

    ctx.kinematics.move_to_pose("right_arm", m_pre, steps=28)
    ctx.kinematics.move_to_pose("right_arm", m_grasp, steps=22)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=22)
    ctx.kinematics.move_to_pose("right_arm", m_lift, steps=28)
    ctx.kinematics.move_to_pose("right_arm", p_above, steps=34)
    ctx.kinematics.move_to_pose("right_arm", p_drop, steps=26)
    ctx.end_effector.set_posture("right_gripper", "open", steps=16)
    ctx.step(10)
    ctx.kinematics.move_to_pose("right_arm", p_ret, steps=22)
    ctx.kinematics.move_joints("right_arm", home_r, steps=22)
    _move_base_to(ctx, 0.30, -1.18, steps=28)
    ctx.step(12)
    print(
        "AFTER_S3",
        [round(v, 4) for v in ctx.world.object_pose("hp_wired_mouse").xyz],
        "score=",
        ctx.world.current_score(),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage3_mouse_inspected")
