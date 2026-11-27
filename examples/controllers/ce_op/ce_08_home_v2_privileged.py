"""Load the optical target, close the lid and start a one-copy scan."""

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
_LID_PIVOT = (0.0, 0.153, 0.252)
_PANEL_PIVOT = (0.0, -0.118, 0.232)
_STOPPER_PIVOT = (0.0, -0.128, 0.168)
_MOUNT = math.radians(52.0)


def _along(pose: Pose3D, distance: float) -> Pose3D:
    return compose_pose(pose, Pose3D((0.0, 0.0, distance)))


def _tip_to_tcp(xyz: Vec3, approach: Vec3, closing: Vec3) -> Pose3D:
    tip = Pose3D.from_approach(xyz=xyz, approach_dir=approach, closing_dir=closing)
    return _along(tip, -0.0226)


def _approach(ctx, arm: str, contact: Pose3D, distance: float = 0.080) -> None:
    ctx.kinematics.move_to_pose(arm, _along(contact, -distance), steps=24)
    ctx.step(4)
    ctx.kinematics.move_to_pose(arm, contact, steps=16)
    ctx.step(4)


def _hinge(base: Pose3D, pivot: Vec3, angle: float) -> Pose3D:
    return compose_pose(base, Pose3D(pivot, quat_from_axis_angle_wxyz(_X, angle)))


def _swing(
    ctx, arm: str, base: Pose3D, pivot: Vec3, mount: float, joint: str, tool: Pose3D, goal: float, count: int
) -> Pose3D:
    start = ctx.world.joint_value("laser_mfp", joint)
    pose = compose_pose(_hinge(base, pivot, mount + start), tool)
    for i in range(1, count + 1):
        angle = mount + start + (goal - start) * i / count
        pose = compose_pose(_hinge(base, pivot, angle), tool)
        ctx.kinematics.move_to_pose(arm, pose, steps=8)
    return pose


def _push_lid(
    ctx, base: Pose3D, point_local: Vec3, direction: Vec3, closing: Vec3, target: float, count: int
) -> Pose3D:
    start = ctx.world.joint_value("laser_mfp", "scanner_lid")
    pose = base
    for i in range(count + 1):
        q = start + (target - start) * i / count
        lid = _hinge(base, _LID_PIVOT, q)
        point = compose_pose(lid, Pose3D(point_local))
        pose = _tip_to_tcp(
            point.xyz,
            rotate_vec_by_quat_wxyz(base.wxyz, direction),
            rotate_vec_by_quat_wxyz(base.wxyz, closing),
        )
        if i == 0:
            _approach(ctx, "left_arm", pose, 0.045)
        else:
            ctx.kinematics.move_to_pose("left_arm", pose, steps=8)
    return pose


def _open_lid(ctx, base: Pose3D) -> None:
    arm = "left_arm"
    q = ctx.world.joint_value("laser_mfp", "scanner_lid")
    ctx.end_effector.set_posture("left_gripper", "open", steps=16)
    ctx.end_effector.command_targets("left_gripper", (0.016, 0.016), steps=16)
    lid_app = (0.0, 0.961, -0.276)
    lid_cls = (0.0, 0.276, 0.961)
    peel = _tip_to_tcp((-0.023, -0.265, 0.019), lid_app, lid_cls)
    _approach(ctx, arm, compose_pose(_hinge(base, _LID_PIVOT, q), peel), 0.042)
    _swing(ctx, arm, base, _LID_PIVOT, 0.0, "scanner_lid", peel, -0.14, 5)
    q = ctx.world.joint_value("laser_mfp", "scanner_lid")
    grip = _tip_to_tcp((-0.023, -0.258, 0.0115), lid_app, lid_cls) if q <= -0.07 else peel
    ctx.kinematics.move_to_pose(arm, compose_pose(_hinge(base, _LID_PIVOT, q), grip), steps=16)
    ctx.step(4)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=22)
    end = _swing(ctx, arm, base, _LID_PIVOT, 0.0, "scanner_lid", grip, -0.43, 7)
    ctx.end_effector.command_targets("left_gripper", (0.016, 0.016), steps=14)
    ctx.kinematics.move_to_pose(arm, _along(end, -0.045), steps=16)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=20)
    start = ctx.world.joint_value("laser_mfp", "scanner_lid")
    contact = (-0.096, -0.225, 0.0005) if start <= -0.25 else (-0.096, -0.266, 0.003)
    hand = _push_lid(ctx, base, contact, (0.0, 0.97, -0.24), _X, -0.96, 10)
    ctx.kinematics.move_to_pose(arm, _along(hand, -0.055), steps=16)


def _close_lid(ctx, base: Pose3D) -> None:
    ctx.end_effector.set_posture("left_gripper", "closed", steps=14)
    raw_q0 = ctx.world.joint_value("laser_mfp", "scanner_lid")
    q0 = -0.95 if raw_q0 > -0.05 else max(min(raw_q0, -0.82), -1.02)
    close_dir1 = (0.45, 0.72, -0.528)
    point_local1 = (-0.145, -0.135, 0.015)
    w_dir1 = rotate_vec_by_quat_wxyz(base.wxyz, close_dir1)
    w_cls = rotate_vec_by_quat_wxyz(base.wxyz, _X)
    detour_front = compose_pose(base, _tip_to_tcp((-0.235, -0.140, 0.355), close_dir1, _X))
    detour_back = compose_pose(base, _tip_to_tcp((-0.235, 0.115, 0.395), close_dir1, _X))
    ctx.kinematics.move_to_pose("left_arm", detour_front, steps=18)
    ctx.kinematics.move_to_pose("left_arm", detour_back, steps=18)
    pre_lid1 = _tip_to_tcp(
        compose_pose(_hinge(base, _LID_PIVOT, q0), Pose3D((-0.145, -0.135, 0.048))).xyz,
        w_dir1,
        w_cls,
    )
    ctx.kinematics.move_to_pose("left_arm", pre_lid1, steps=16)
    pose = pre_lid1
    for i in range(9):
        ang = q0 + (-0.35 - q0) * i / 8
        lid = _hinge(base, _LID_PIVOT, ang)
        pt = compose_pose(lid, Pose3D(point_local1)).xyz
        pose = _tip_to_tcp(pt, w_dir1, w_cls)
        ctx.kinematics.move_to_pose("left_arm", pose, steps=8)
    ctx.kinematics.move_to_pose(
        "left_arm", Pose3D((pose.xyz[0], pose.xyz[1], pose.xyz[2] + 0.065), pose.wxyz), steps=12
    )
    q_start = -0.46
    close_dir2 = (0.20, 0.45, -0.8703)
    w_dir2 = rotate_vec_by_quat_wxyz(base.wxyz, close_dir2)
    point_local2 = (-0.100, -0.210, 0.020)
    pt0 = compose_pose(_hinge(base, _LID_PIVOT, q_start), Pose3D(point_local2)).xyz
    pre_lid2 = _tip_to_tcp((pt0[0], pt0[1], pt0[2] + 0.080), w_dir2, w_cls)
    if True:
        ctx.kinematics.move_to_pose("left_arm", pre_lid2, steps=14)
        pose = pre_lid2
        for i in range(9):
            ang = q_start + (0.018 - q_start) * i / 8
            lid = _hinge(base, _LID_PIVOT, ang)
            pt = compose_pose(lid, Pose3D(point_local2)).xyz
            pose = _tip_to_tcp(pt, w_dir2, w_cls)
            ctx.kinematics.move_to_pose("left_arm", pose, steps=8)
    ctx.step(12)
    ctx.kinematics.move_to_pose(
        "left_arm", Pose3D((pose.xyz[0], pose.xyz[1], pose.xyz[2] + 0.085), pose.wxyz), steps=16
    )


def _close_cassette(ctx, base: Pose3D) -> None:
    q = ctx.world.joint_value("laser_mfp", "paper_cassette")
    face = _tip_to_tcp((-0.106, -0.1685, 0.056), (0.0, 0.78, -0.6258), _X)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=20)
    contact = compose_pose(base, compose_pose(Pose3D((0.0, q, 0.0)), face))
    _approach(ctx, "left_arm", contact)
    end = contact
    for i in range(1, 13):
        y = q + (0.0005 - q) * i / 12
        end = compose_pose(base, compose_pose(Pose3D((0.0, y, 0.0)), face))
        ctx.kinematics.move_to_pose("left_arm", end, steps=6)
        if ctx.world.joint_value("laser_mfp", "paper_cassette") >= -0.002:
            break
    ctx.kinematics.move_to_pose("left_arm", _along(end, -0.080), steps=18)


def _load_platen_left(ctx, base: Pose3D, home_l: tuple[float, ...]) -> None:
    # Single stride-4-aligned quasi-static drag to y_com ~ -0.366 (zero conditional 2nd-drag branch),
    # high-front detour before opening jaws, neck-cleat lock behind front_grip_tab, flat lay on GlassPlaten.
    ctx.kinematics.move_joints("left_arm", home_l, steps=16)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    z_pin = base.xyz[2] + 0.0046
    app_l = (0.0, 0.25, -0.9682)
    sh_cur = ctx.world.object_pose("calib_sheet")
    if sh_cur.xyz[1] > -0.362:
        p0 = compose_pose(sh_cur, Pose3D((0.0, 0.055, 0.0))).xyz
        dy_drag = min(0.0, -0.371 - sh_cur.xyz[1])
        y_goal = p0[1] + dy_drag
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((p0[0], p0[1], z_pin + 0.075), app_l, _X), steps=16)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((p0[0], p0[1], z_pin), app_l, _X), steps=12)
        ctx.step(8)
        for k in range(1, 7):
            yk = p0[1] + (y_goal - p0[1]) * k / 6
            ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((p0[0], yk, z_pin), app_l, _X), steps=12)
        ctx.step(8)
        cur_e = ctx.kinematics.ee_pose("left_arm")
        ctx.kinematics.move_to_pose("left_arm", Pose3D((cur_e.xyz[0], cur_e.xyz[1], z_pin + 0.120), cur_e.wxyz), steps=12)

    sh_o = ctx.world.object_pose("calib_sheet")
    pitch_g = 0.46
    app_g = (0.0, math.cos(pitch_g), -math.sin(pitch_g))
    cls_g = (0.0, math.sin(pitch_g), math.cos(pitch_g))
    p_g = compose_pose(sh_o, Pose3D((0.0, -0.096, 0.0))).xyz
    gx = min(max(p_g[0], -0.425), -0.250)
    gy = min(max(p_g[1], -0.510), -0.452)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((gx, gy - 0.065, base.xyz[2] + 0.110), app_g, cls_g), steps=16)
    ctx.end_effector.set_posture("left_gripper", "open", steps=16)
    hand_l = _tip_to_tcp((gx, gy, base.xyz[2] + 0.0035), app_g, cls_g)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((gx, gy - 0.065, base.xyz[2] + 0.012), app_g, cls_g), steps=16)
    ctx.kinematics.move_to_pose("left_arm", hand_l, steps=16)
    ctx.step(8)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=36)
    ctx.step(12)

    dx_com = min(max(sh_o.xyz[0] - gx, -0.030), 0.030)
    tx_pl = base.xyz[0] - dx_com
    for wx, wy, wz in (
        (gx, gy + 0.015, 0.900),
        (gx, -0.420, 1.040),
        (gx, -0.330, 1.155),
        (gx, -0.265, 1.235),
        (0.5 * (gx + tx_pl), -0.260, 1.235),
        (tx_pl, -0.255, 1.232),
        (tx_pl, -0.230, 1.125),
    ):
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((wx, wy, wz), app_g, cls_g), steps=16)
    ctx.step(8)
    drop_pl = _tip_to_tcp((tx_pl, base.xyz[1] - 0.102, base.xyz[2] + 0.262), app_g, cls_g)
    ctx.kinematics.move_to_pose("left_arm", drop_pl, steps=16)
    ctx.step(8)
    ctx.end_effector.set_posture("left_gripper", "open", steps=16)
    ctx.step(8)
    ctx.kinematics.move_to_pose(
        "left_arm",
        Pose3D((drop_pl.xyz[0], drop_pl.xyz[1] - 0.052, drop_pl.xyz[2] + 0.003), drop_pl.wxyz),
        steps=16,
    )
    ctx.kinematics.move_to_pose(
        "left_arm",
        Pose3D((drop_pl.xyz[0] - 0.110, drop_pl.xyz[1] - 0.045, drop_pl.xyz[2] + 0.105), drop_pl.wxyz),
        steps=16,
    )
    ctx.step(8)


def _set_panel(ctx, base: Pose3D) -> None:
    arm = "right_arm"
    q = ctx.world.joint_value("laser_mfp", "control_panel")
    push = _tip_to_tcp((0.120, -0.050, 0.0045), (0.0, 0.0, -1.0), _X)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=18)
    _approach(ctx, arm, compose_pose(_hinge(base, _PANEL_PIVOT, _MOUNT + q), push))
    end = _swing(ctx, arm, base, _PANEL_PIVOT, _MOUNT, "control_panel", push, 0.004, 9)
    ctx.kinematics.move_to_pose(arm, _along(end, -0.070), steps=18)


def _press(ctx, base: Pose3D, joint: str, cap_local: Vec3) -> bool:
    q = ctx.world.joint_value("laser_mfp", "control_panel")
    panel = _hinge(base, _PANEL_PIVOT, _MOUNT + q)
    press = compose_pose(panel, _tip_to_tcp(cap_local, (0.0, 0.0, -1.0), _X))
    ctx.end_effector.set_posture("right_gripper", "closed", steps=16)
    _approach(ctx, "right_arm", press)
    end = press
    pressed = False
    for depth in (0.0020, 0.0045, 0.0070, 0.0095):
        end = _along(press, depth)
        ctx.kinematics.move_to_pose("right_arm", end, steps=8)
        if ctx.world.joint_value("laser_mfp", joint) <= -0.0016:
            pressed = True
    ctx.step(20)
    if ctx.world.joint_value("laser_mfp", joint) <= -0.0016:
        pressed = True
    ctx.kinematics.move_to_pose("right_arm", _along(end, -0.080), steps=16)
    return pressed


def _set_dial_zero(ctx, base: Pose3D) -> None:
    arm = "right_arm"
    pitch = ctx.world.joint_value("laser_mfp", "control_panel")
    panel = _hinge(base, _PANEL_PIVOT, _MOUNT + pitch)
    spindle = compose_pose(panel, Pose3D((0.087, -0.029, 0.007)))
    hand = _tip_to_tcp((0.0, 0.0, 0.0045), (0.0, 0.0, -1.0), _X)
    for _ in range(5):
        q = ctx.world.joint_value("laser_mfp", "copy_count_dial")
        if q <= 0.035:
            break
        delta_d = min(q + 0.10, 0.90)
        ctx.end_effector.set_posture("right_gripper", "open", steps=8)
        tw0 = Pose3D((0.0, 0.0, 0.0), quat_from_axis_angle_wxyz(_Z, 0.5 * delta_d))
        p_d = compose_pose(spindle, compose_pose(tw0, hand))
        ctx.kinematics.move_to_pose(arm, _along(p_d, -0.050), steps=12)
        ctx.kinematics.move_to_pose(arm, p_d, steps=10)
        ctx.end_effector.set_posture("right_gripper", "closed", steps=26)
        ctx.step(4)
        end = p_d
        for i in range(1, 9):
            ang = 0.5 * delta_d - delta_d * i / 8
            twist = Pose3D((0.0, 0.0, 0.0), quat_from_axis_angle_wxyz(_Z, ang))
            end = compose_pose(spindle, compose_pose(twist, hand))
            ctx.kinematics.move_to_pose(arm, end, steps=7)
            if ctx.world.joint_value("laser_mfp", "copy_count_dial") <= 0.035:
                break
        ctx.end_effector.set_posture("right_gripper", "open", steps=8)
        ctx.kinematics.move_to_pose(arm, _along(end, -0.055), steps=12)
    if ctx.world.joint_value("laser_mfp", "copy_count_dial") > 0.055:
        pitch = ctx.world.joint_value("laser_mfp", "control_panel")
        panel = _hinge(base, _PANEL_PIVOT, _MOUNT + pitch)
        spindle = compose_pose(panel, Pose3D((0.087, -0.029, 0.007)))
        rim_dir = rotate_vec_by_quat_wxyz(panel.wxyz, (0.0, 0.0, -1.0))
        r_c, r_out = 0.0125, 0.026
        ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
        for _ in range(3):
            if ctx.world.joint_value("laser_mfp", "copy_count_dial") <= 0.045:
                break
            th0 = -math.pi / 2 + 0.85
            cls0 = rotate_vec_by_quat_wxyz(panel.wxyz, (-math.sin(th0), math.cos(th0), 0.0))
            p_pre = compose_pose(spindle, Pose3D((r_out * math.cos(th0), r_out * math.sin(th0), 0.045))).xyz
            p_in = compose_pose(spindle, Pose3D((r_out * math.cos(th0), r_out * math.sin(th0), 0.0035))).xyz
            p_eng = compose_pose(spindle, Pose3D((r_c * math.cos(th0), r_c * math.sin(th0), 0.0035))).xyz
            ctx.kinematics.move_to_pose(arm, _tip_to_tcp(p_pre, rim_dir, cls0), steps=10)
            ctx.kinematics.move_to_pose(arm, _tip_to_tcp(p_in, rim_dir, cls0), steps=8)
            ctx.kinematics.move_to_pose(arm, _tip_to_tcp(p_eng, rim_dir, cls0), steps=6)
            th_end = th0
            for ii in range(1, 9):
                th = th0 - 1.50 * ii / 8
                th_end = th
                t_cls = rotate_vec_by_quat_wxyz(panel.wxyz, (-math.sin(th), math.cos(th), 0.0))
                p_sw = compose_pose(spindle, Pose3D((r_c * math.cos(th), r_c * math.sin(th), 0.0035))).xyz
                ctx.kinematics.move_to_pose(arm, _tip_to_tcp(p_sw, rim_dir, t_cls), steps=6)
                if ctx.world.joint_value("laser_mfp", "copy_count_dial") <= 0.045:
                    break
            cls_e = rotate_vec_by_quat_wxyz(panel.wxyz, (-math.sin(th_end), math.cos(th_end), 0.0))
            p_out = compose_pose(spindle, Pose3D((r_out * math.cos(th_end), r_out * math.sin(th_end), 0.0035))).xyz
            p_up = compose_pose(spindle, Pose3D((r_out * math.cos(th_end), r_out * math.sin(th_end), 0.045))).xyz
            ctx.kinematics.move_to_pose(arm, _tip_to_tcp(p_out, rim_dir, cls_e), steps=6)
            ctx.kinematics.move_to_pose(arm, _tip_to_tcp(p_up, rim_dir, cls_e), steps=8)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=10)
    home = ctx.sensors.proprio()
    home_l = tuple(home[:6])
    home_r = tuple(home[8:14])
    _close_cassette(ctx, ctx.world.object_pose("laser_mfp"))
    _open_lid(ctx, ctx.world.object_pose("laser_mfp"))
    _load_platen_left(ctx, ctx.world.object_pose("laser_mfp"), home_l)
    _close_lid(ctx, ctx.world.object_pose("laser_mfp"))
    ctx.kinematics.move_joints("left_arm", home_l, steps=20)
    _set_panel(ctx, ctx.world.object_pose("laser_mfp"))
    _set_dial_zero(ctx, ctx.world.object_pose("laser_mfp"))
    _press(ctx, ctx.world.object_pose("laser_mfp"), "start_button", (0.148, -0.030, 0.0167))
    ctx.kinematics.move_joints("right_arm", home_r, steps=20)
    ctx.step(36)
