"""File the calibration sheet in the walnut tray and set the printer to standby."""

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

_PANEL_Q_SEQ = (
    ((0.54208, 1.45773, 0.75676, -0.41053, 0.26091, 0.48076), 16),
    ((0.46926, 1.76853, 0.74203, -0.06878, 0.22812, 0.41381), 12),
    ((0.48056, 1.76961, 0.69183, 0.02948, 0.25450, 0.41212), 8),
    ((0.49159, 1.77696, 0.64377, 0.13209, 0.28114, 0.40874), 8),
    ((0.50236, 1.79160, 0.59850, 0.23957, 0.30786, 0.40348), 8),
    ((0.51277, 1.81426, 0.55638, 0.35241, 0.33448, 0.39623), 8),
    ((0.52273, 1.84563, 0.51784, 0.47103, 0.36081, 0.38685), 8),
    ((0.53212, 1.88631, 0.48340, 0.59565, 0.38662, 0.37525), 8),
    ((0.54086, 1.93668, 0.45370, 0.72615, 0.41167, 0.36130), 8),
    ((0.54886, 1.99681, 0.42960, 0.86192, 0.43566, 0.34492), 8),
    ((0.71018, 1.57918, 0.27055, 0.54475, 0.55554, 0.46778), 12),
)


def _close_cassette_firm(ctx, base: Pose3D) -> None:
    q = ctx.world.joint_value("laser_mfp", "paper_cassette")
    face = _tip_to_tcp((-0.106, -0.1685, 0.056), (0.0, 0.78, -0.6258), _X)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=20)
    contact = compose_pose(base, compose_pose(Pose3D((0.0, q, 0.0)), face))
    ctx.kinematics.move_joints("left_arm", (-1.05000, 1.35000, 0.62000, 0.05000, -0.74000, -0.83000), steps=12)
    ctx.kinematics.move_to_pose("left_arm", _along(contact, -0.080), steps=12)
    ctx.step(4)
    ctx.kinematics.move_to_pose("left_arm", contact, steps=16)
    ctx.step(4)
    end = contact
    for i in range(1, 13):
        y = q + (0.004 - q) * i / 12
        end = compose_pose(base, compose_pose(Pose3D((0.0, y, 0.0)), face))
        ctx.kinematics.move_to_pose("left_arm", end, steps=8)
        if ctx.world.joint_value("laser_mfp", "paper_cassette") >= -0.0015:
            break
    ctx.kinematics.move_to_pose("left_arm", _along(end, -0.080), steps=16)


def _set_panel_det(ctx, home_r: tuple[float, ...]) -> None:
    ctx.kinematics.move_joints("right_arm", home_r, steps=8)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=12)
    for idx_p, (q_wp, stp) in enumerate(_PANEL_Q_SEQ):
        ctx.kinematics.move_joints("right_arm", q_wp, steps=stp)
        if idx_p in (0, 1):
            ctx.step(4)
    ctx.kinematics.move_joints("right_arm", home_r, steps=8)


def _relay_roof_overhang_left(ctx, base: Pose3D, home_l: tuple[float, ...]) -> None:
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
    tx_pl = min(base.xyz[0] - dx_com, 0.0)
    for wx, wy, wz in (
        (gx, gy + 0.015, 0.900),
        (gx, -0.420, 1.050),
        (gx, -0.330, 1.175),
        (gx, -0.265, 1.255),
        (0.5 * (gx + tx_pl), -0.260, 1.255),
        (tx_pl, -0.250, 1.250),
        (tx_pl, -0.228, 1.150),
    ):
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((wx, wy, wz), app_g, cls_g), steps=16)
    ctx.step(8)
    drop_pl = _tip_to_tcp((tx_pl, base.xyz[1] - 0.130, base.xyz[2] + 0.289), app_g, cls_g)
    ctx.kinematics.move_to_pose("left_arm", drop_pl, steps=16)
    ctx.step(8)
    ctx.end_effector.set_posture("left_gripper", "open", steps=16)
    ctx.step(8)
    ctx.kinematics.move_to_pose(
        "left_arm",
        Pose3D((drop_pl.xyz[0], drop_pl.xyz[1] - 0.038, drop_pl.xyz[2] + 0.015), drop_pl.wxyz),
        steps=16,
    )
    ctx.kinematics.move_to_pose(
        "left_arm",
        Pose3D((drop_pl.xyz[0] - 0.120, drop_pl.xyz[1] - 0.035, drop_pl.xyz[2] + 0.095), drop_pl.wxyz),
        steps=16,
    )
    ctx.step(8)


def _forklift_roof_to_tray_right(ctx, base: Pose3D, home_l: tuple[float, ...], home_r: tuple[float, ...]) -> None:
    ctx.kinematics.move_joints("left_arm", home_l, steps=12)
    pitch_r = 0.46
    app_r = (0.0, math.cos(pitch_r), -math.sin(pitch_r))
    cls_r = (0.0, math.sin(pitch_r), math.cos(pitch_r))

    sh_pl = ctx.world.object_pose("calib_sheet")
    if sh_pl.xyz[2] < base.xyz[2] + 0.180:
        sh_pl = compose_pose(base, Pose3D((0.0, -0.024, 0.2765)))
    p_gr = compose_pose(sh_pl, Pose3D((0.0, -0.100, 0.0))).xyz
    gx = min(max(p_gr[0], -0.035), 0.045)
    gy = min(max(p_gr[1], -0.242), -0.210)
    gz = min(max(sh_pl.xyz[2] + 0.0015, base.xyz[2] + 0.275), base.xyz[2] + 0.285)
    app_gr = rotate_vec_by_quat_wxyz(sh_pl.wxyz, app_r)
    cls_gr = rotate_vec_by_quat_wxyz(sh_pl.wxyz, cls_r)

    ctx.end_effector.set_posture("right_gripper", "open", steps=16)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((gx, gy - 0.065, gz + 0.035), app_gr, cls_gr), steps=20)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((gx, gy - 0.045, gz + 0.003), app_gr, cls_gr), steps=16)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((gx, gy, gz), app_gr, cls_gr), steps=16)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=36)
    ctx.step(12)

    tray = ctx.world.object_pose("doc_tray")
    pitch_d = 0.48
    app_d = (0.0, math.cos(pitch_d), -math.sin(pitch_d))
    cls_d = (0.0, math.sin(pitch_d), math.cos(pitch_d))
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((gx + 0.020, -0.285, 1.165), app_r, cls_r), steps=16)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((0.245, -0.295, 1.165), app_d, cls_d), steps=20)

    hov_pose = compose_pose(tray, _tip_to_tcp((0.0, -0.105, 0.205), app_d, cls_d))
    ctx.kinematics.move_to_pose("right_arm", hov_pose, steps=24)
    ctx.step(12)

    sh_h = ctx.world.object_pose("calib_sheet")
    ee_h = ctx.kinematics.ee_pose("right_arm")
    goal_sh = compose_pose(tray, Pose3D((0.0, -0.010, 0.0))).xyz
    if abs(sh_h.xyz[0] - tray.xyz[0]) < 0.24 and sh_h.xyz[2] > tray.xyz[2] + 0.050:
        dx_w = min(max(goal_sh[0] - sh_h.xyz[0], -0.055), 0.055)
        dy_w = min(max(goal_sh[1] - sh_h.xyz[1], -0.055), 0.055)
        drop_x = ee_h.xyz[0] + dx_w
        drop_y = ee_h.xyz[1] + dy_w
    else:
        drop_x = hov_pose.xyz[0]
        drop_y = hov_pose.xyz[1]

    mid_pose = Pose3D((drop_x, drop_y - 0.010, hov_pose.xyz[2] - 0.055), hov_pose.wxyz)
    drop_pose = Pose3D((drop_x, drop_y, hov_pose.xyz[2] - 0.097), hov_pose.wxyz)
    ctx.kinematics.move_to_pose("right_arm", mid_pose, steps=16)
    ctx.kinematics.move_to_pose("right_arm", drop_pose, steps=16)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "open", steps=16)
    ctx.step(12)
    ctx.kinematics.move_to_pose("right_arm", Pose3D((drop_x, drop_y - 0.025, hov_pose.xyz[2] - 0.020), hov_pose.wxyz), steps=16)
    ctx.kinematics.move_joints("right_arm", home_r, steps=12)


_STOPPER_Q4 = (
    (-0.35, 0.45004, 0.2066, 0.01667, -0.24667, -0.27667),
    (-0.7, 0.90002, 0.4133, 0.03333, -0.49333, -0.55333),
    (-1.05, 1.35, 0.62, 0.05, -0.74, -0.83),
    (-1.05233, 1.5252, 0.61276, 0.16348, -0.74184, -0.83248),
    (-1.05465, 1.70041, 0.60552, 0.27695, -0.74368, -0.83495),
    (-1.05698, 1.87561, 0.59827, 0.39043, -0.74551, -0.83743),
    (-1.0593, 2.05082, 0.59103, 0.5039, -0.74735, -0.8399),
    (-1.06163, 2.22602, 0.58379, 0.61738, -0.74919, -0.84238),
    (-1.06163, 2.22602, 0.58379, 0.61738, -0.74919, -0.84238),
    (-1.06163, 2.22602, 0.58379, 0.61738, -0.74919, -0.84238),
    (-1.0273, 2.29661, 0.61282, 0.68514, -0.73079, -0.8033),
    (-0.99401, 2.3654, 0.64551, 0.74573, -0.71244, -0.76627),
    (-0.96176, 2.43241, 0.68186, 0.79916, -0.69416, -0.73129),
    (-0.96176, 2.43241, 0.68186, 0.79916, -0.69416, -0.73129),
    (-0.91622, 2.4414, 0.7167, 0.80244, -0.66691, -0.68459),
    (-0.87292, 2.45085, 0.75422, 0.80069, -0.64036, -0.64144),
    (-0.83189, 2.46078, 0.79441, 0.79391, -0.61453, -0.60185),
    (-0.83189, 2.46078, 0.79441, 0.79391, -0.61453, -0.60185),
    (-0.83189, 2.46078, 0.79441, 0.79391, -0.61453, -0.60185),
    (-0.84394, 2.43007, 0.77497, 0.77329, -0.61437, -0.60163),
    (-0.83185, 2.39963, 0.77578, 0.75142, -0.61446, -0.60177),
    (-0.83381, 2.39007, 0.75522, 0.7327, -0.61565, -0.60291),
    (-0.83417, 2.34615, 0.75948, 0.71295, -0.616, -0.60402),
    (-0.83749, 2.34859, 0.74818, 0.69425, -0.61807, -0.60707),
    (-0.84054, 2.291, 0.74166, 0.67217, -0.62005, -0.60998),
    (-0.84599, 2.30081, 0.72896, 0.65256, -0.62331, -0.61523),
    (-0.85134, 2.23359, 0.72175, 0.62867, -0.62694, -0.62025),
    (-0.85927, 2.24738, 0.70746, 0.60734, -0.63164, -0.62802),
    (-0.86717, 2.17346, 0.70004, 0.58127, -0.63693, -0.63546),
    (-0.86717, 2.17346, 0.70004, 0.58127, -0.63693, -0.63546),
    (-0.94224, 2.2016, 0.63586, 0.57751, -0.67975, -0.71331),
    (-1.02308, 2.13856, 0.58971, 0.55379, -0.72867, -0.79818),
    (-1.02308, 2.13856, 0.58971, 0.55379, -0.72867, -0.79818),
    (-1.00223, 2.23834, 0.61473, 0.64376, -0.71712, -0.7751),
    (-0.98179, 2.3363, 0.64545, 0.72556, -0.70562, -0.7528),
    (-0.96176, 2.43246, 0.68189, 0.79919, -0.69416, -0.73128),
    (-0.96176, 2.43246, 0.68189, 0.79919, -0.69416, -0.73128),
    (-0.91622, 2.44143, 0.71672, 0.80249, -0.66692, -0.68457),
    (-0.87293, 2.45087, 0.75423, 0.80073, -0.64037, -0.64142),
    (-0.83188, 2.46077, 0.79441, 0.7939, -0.61452, -0.60185),
    (-0.83188, 2.46077, 0.79441, 0.7939, -0.61452, -0.60185),
    (-0.83188, 2.46077, 0.79441, 0.7939, -0.61452, -0.60185),
    (-0.84387, 2.43006, 0.7749, 0.77333, -0.61437, -0.60163),
    (-0.83185, 2.39963, 0.77578, 0.75142, -0.61446, -0.60177),
    (-0.84274, 2.39353, 0.76607, 0.73381, -0.61491, -0.603),
    (-0.8342, 2.34614, 0.75946, 0.71295, -0.61602, -0.60405),
    (-0.84066, 2.35135, 0.74717, 0.69432, -0.61857, -0.60713),
    (-0.8405, 2.29101, 0.74167, 0.67219, -0.62003, -0.60995),
    (-0.84635, 2.29997, 0.72692, 0.65245, -0.6233, -0.61522),
    (-0.85132, 2.23361, 0.72179, 0.62866, -0.62692, -0.62022),
    (-0.85948, 2.24159, 0.70693, 0.60685, -0.63229, -0.62841),
    (-0.86724, 2.17351, 0.70016, 0.5812, -0.63693, -0.63546),
    (-0.86724, 2.17351, 0.70016, 0.5812, -0.63693, -0.63546),
    (-0.95451, 2.18504, 0.63221, 0.57619, -0.68034, -0.71296),
    (-1.02305, 2.13857, 0.58966, 0.55386, -0.72868, -0.79816),
    (-1.02305, 2.13857, 0.58966, 0.55386, -0.72868, -0.79816),
    (-1.03203, 1.87572, 0.59978, 0.38591, -0.73245, -0.80877),
    (-1.04102, 1.61286, 0.60989, 0.21795, -0.73623, -0.81939),
    (-1.05, 1.35, 0.62, 0.05, -0.74, -0.83),
    (-1.07809, 1.68441, 0.57292, 0.38773, -0.81844, -0.79235),
    (-1.10617, 2.01882, 0.52585, 0.72546, -0.89688, -0.75471),
    (-1.13426, 2.35323, 0.47877, 1.06319, -0.97532, -0.71706),
    (-1.13426, 2.35323, 0.47877, 1.06319, -0.97532, -0.71706),
    (-1.13428, 2.54104, 0.53988, 1.18988, -0.97532, -0.71707),
    (-1.13431, 2.7125, 0.62056, 1.28066, -0.97533, -0.71707),
    (-1.13431, 2.7125, 0.62056, 1.28066, -0.97533, -0.71707),
    (-1.06527, 2.72267, 0.64722, 1.33201, -0.92612, -0.63374),
    (-1.00115, 2.72801, 0.67946, 1.35845, -0.87774, -0.5658),
    (-0.96682, 2.72933, 0.70065, 1.36321, -0.85084, -0.53323),
    (-0.93258, 2.73027, 0.7237, 1.36426, -0.82366, -0.50207),
    (-0.89843, 2.73081, 0.74861, 1.36159, -0.79622, -0.47234),
    (-0.89843, 2.73081, 0.74861, 1.36159, -0.79622, -0.47234),
    (-0.89843, 2.73081, 0.74861, 1.36159, -0.79622, -0.47234),
    (-0.90133, 2.69488, 0.72563, 1.34571, -0.79615, -0.4723),
    (-0.89836, 2.65815, 0.70936, 1.32824, -0.79611, -0.47225),
    (-0.89917, 2.63369, 0.6965, 1.31611, -0.79672, -0.47287),
    (-0.89982, 2.60675, 0.68316, 1.30215, -0.79733, -0.4735),
    (-0.90304, 2.5836, 0.67018, 1.29004, -0.79995, -0.47618),
    (-0.90631, 2.55852, 0.65683, 1.27634, -0.8026, -0.47894),
    (-0.91255, 2.53714, 0.64321, 1.26422, -0.80762, -0.48417),
    (-0.91891, 2.51318, 0.62924, 1.25084, -0.81279, -0.48967),
    (-0.9246, 2.49668, 0.61875, 1.24093, -0.81731, -0.49451),
    (-0.93056, 2.47785, 0.60784, 1.22954, -0.82215, -0.49977),
    (-0.93309, 2.46377, 0.60114, 1.22029, -0.82415, -0.50199),
    (-0.93591, 2.44812, 0.59405, 1.21012, -0.82644, -0.5045),
    (-0.93591, 2.44812, 0.59405, 1.21012, -0.82644, -0.5045),
    (-0.93591, 2.44812, 0.59405, 1.21012, -0.82644, -0.5045),
    (-0.97596, 2.49539, 0.5852, 1.23791, -0.85759, -0.541),
    (-1.01753, 2.54136, 0.58107, 1.25737, -0.89029, -0.58238),
    (-1.07276, 2.53111, 0.55385, 1.22691, -0.93156, -0.64227),
    (-1.1343, 2.51977, 0.53163, 1.17686, -0.97534, -0.71706),
    (-1.13248, 2.61904, 0.57395, 1.23559, -0.97404, -0.71484),
    (-1.13429, 2.71249, 0.62056, 1.28066, -0.97532, -0.71706),
    (-1.13429, 2.71249, 0.62056, 1.28066, -0.97532, -0.71706),
    (-1.06528, 2.72266, 0.64721, 1.332, -0.92612, -0.63376),
    (-1.00115, 2.72801, 0.67946, 1.35843, -0.87773, -0.5658),
    (-0.96682, 2.72933, 0.70065, 1.36319, -0.85083, -0.53323),
    (-0.93258, 2.73027, 0.7237, 1.36424, -0.82366, -0.50208),
    (-0.89843, 2.73081, 0.74861, 1.36157, -0.79621, -0.47234),
    (-0.89843, 2.73081, 0.74861, 1.36157, -0.79621, -0.47234),
    (-0.89843, 2.73081, 0.74861, 1.36157, -0.79621, -0.47234),
    (-0.90135, 2.69491, 0.72566, 1.34566, -0.79625, -0.47233),
    (-0.89836, 2.65815, 0.70937, 1.32824, -0.79611, -0.47224),
    (-0.89917, 2.63369, 0.6965, 1.31611, -0.79673, -0.47287),
    (-0.89982, 2.60675, 0.68316, 1.30215, -0.79733, -0.4735),
    (-0.90304, 2.5836, 0.67018, 1.29004, -0.79995, -0.47618),
    (-0.90631, 2.55852, 0.65683, 1.27634, -0.8026, -0.47894),
    (-0.91248, 2.5368, 0.64324, 1.26444, -0.8076, -0.48417),
    (-0.91892, 2.51323, 0.62925, 1.25087, -0.8128, -0.48967),
    (-0.92449, 2.49644, 0.61879, 1.24105, -0.8173, -0.49444),
    (-0.93058, 2.47786, 0.60781, 1.22954, -0.82218, -0.49978),
    (-0.93301, 2.46364, 0.60111, 1.22048, -0.82413, -0.50193),
    (-0.9359, 2.44816, 0.59403, 1.21021, -0.82645, -0.50448),
    (-0.9359, 2.44816, 0.59403, 1.21021, -0.82645, -0.50448),
    (-0.9359, 2.44816, 0.59403, 1.21021, -0.82645, -0.50448),
    (-0.97583, 2.49497, 0.58515, 1.23811, -0.85786, -0.54096),
    (-1.01754, 2.54143, 0.5811, 1.2574, -0.89029, -0.58239),
    (-1.07276, 2.53112, 0.55388, 1.22692, -0.93157, -0.64227),
    (-1.1343, 2.51978, 0.5316, 1.17693, -0.97534, -0.71708),
    (-1.13248, 2.61904, 0.57394, 1.23561, -0.97404, -0.71486),
    (-1.13429, 2.7125, 0.62056, 1.28068, -0.97532, -0.71705),
    (-1.13429, 2.7125, 0.62056, 1.28068, -0.97532, -0.71705),
    (-1.06527, 2.72267, 0.64721, 1.33197, -0.92611, -0.63375),
    (-1.00114, 2.72801, 0.67946, 1.35846, -0.87773, -0.56582),
    (-0.96682, 2.72933, 0.70065, 1.36321, -0.85084, -0.53324),
    (-0.93258, 2.73026, 0.7237, 1.36425, -0.82366, -0.50208),
    (-0.89842, 2.73082, 0.74861, 1.3616, -0.79621, -0.47234),
    (-0.89842, 2.73082, 0.74861, 1.3616, -0.79621, -0.47234),
    (-0.89842, 2.73082, 0.74861, 1.3616, -0.79621, -0.47234),
    (-0.90131, 2.6949, 0.7256, 1.34571, -0.79615, -0.4723),
    (-0.89836, 2.65815, 0.70935, 1.32824, -0.79611, -0.47225),
    (-0.89917, 2.63369, 0.69649, 1.31611, -0.79672, -0.47287),
    (-0.89982, 2.60675, 0.68316, 1.30215, -0.79733, -0.4735),
    (-0.90304, 2.5836, 0.67018, 1.29004, -0.79995, -0.47618),
    (-0.90631, 2.55852, 0.65683, 1.27634, -0.8026, -0.47894),
    (-0.91248, 2.5368, 0.64324, 1.26444, -0.8076, -0.48417),
    (-0.91892, 2.51323, 0.62925, 1.25087, -0.8128, -0.48967),
    (-0.92452, 2.4964, 0.6188, 1.24101, -0.81731, -0.49449),
    (-0.93059, 2.47776, 0.60791, 1.22935, -0.82218, -0.49982),
    (-0.93305, 2.46362, 0.60115, 1.22036, -0.82415, -0.50196),
    (-0.93589, 2.44809, 0.594, 1.21011, -0.82643, -0.5045),
    (-0.93589, 2.44809, 0.594, 1.21011, -0.82643, -0.5045),
    (-0.93589, 2.44809, 0.594, 1.21011, -0.82643, -0.5045),
    (-0.9767, 2.4956, 0.58437, 1.23762, -0.8578, -0.54108),
    (-1.01744, 2.54132, 0.58112, 1.25736, -0.8902, -0.58226),
    (-1.07274, 2.53109, 0.55386, 1.22687, -0.93153, -0.64222),
    (-1.1343, 2.51978, 0.5316, 1.1769, -0.97534, -0.71706),
    (-1.13248, 2.35374, 0.47942, 1.06482, -0.97404, -0.71484),
    (-1.13429, 2.17703, 0.44425, 0.92153, -0.97532, -0.71705),
    (-1.03418, 2.21528, 0.48311, 1.01575, -0.90289, -0.59981),
    (-0.94491, 2.23803, 0.53547, 1.0527, -0.83367, -0.51251),
    (-0.94665, 2.27697, 0.54209, 1.08373, -0.83501, -0.51422),
    (-0.94664, 2.3161, 0.55055, 1.1144, -0.83499, -0.51423),
    (-0.94489, 2.35544, 0.56084, 1.14473, -0.83361, -0.51253),
    (-0.94489, 2.35544, 0.56084, 1.14473, -0.83361, -0.51253),
    (-0.94489, 2.35544, 0.56084, 1.14473, -0.83361, -0.51253),
    (-1.01262, 2.27896, 0.5045, 1.07541, -0.88655, -0.57734),
    (-1.08653, 2.19331, 0.45947, 0.97082, -0.94162, -0.65817),
    (-1.06827, 1.77165, 0.53973, 0.51041, -0.84081, -0.74408),
    (-1.05, 1.35, 0.62, 0.05, -0.74, -0.83),
    (-0.99543, 1.79538, 0.31088, 0.67176, -0.70742, -0.77266),
    (-0.91971, 2.66764, 0.35918, 1.38462, -0.66925, -0.68759),
    (-0.84503, 2.68969, 0.42062, 1.3943, -0.62244, -0.61479),
    (-0.77366, 2.75164, 0.49139, 1.41764, -0.57643, -0.54866),
    (-0.73931, 2.73768, 0.51338, 1.39787, -0.55332, -0.51862),
    (-0.70373, 2.72762, 0.54112, 1.37562, -0.52903, -0.48823),
    (-0.67906, 2.71866, 0.56223, 1.35575, -0.51194, -0.46773),
    (-0.65396, 2.70976, 0.58596, 1.33299, -0.49438, -0.44724),
    (-0.63689, 2.70377, 0.60365, 1.31573, -0.48234, -0.43356),
    (-0.61924, 2.69778, 0.62332, 1.29649, -0.4698, -0.41956),
    (-0.69657, 2.61806, 0.49067, 1.31145, -0.52053, -0.47903),
    (-0.77594, 2.52484, 0.38635, 1.29478, -0.57796, -0.55063),
    (-0.91297, 1.93742, 0.50318, 0.67239, -0.65898, -0.69032),
    (-1.05, 1.35, 0.62, 0.05, -0.74, -0.83),
)


def _deploy_stopper_det(ctx, home_l: tuple[float, ...]) -> None:
    ctx.kinematics.move_joints("left_arm", home_l, steps=12)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=8)
    for q_wp in _STOPPER_Q4:
        ctx.kinematics.move_joints("left_arm", q_wp, steps=4)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    home = ctx.sensors.proprio()
    home_l = tuple(home[:6])
    home_r = tuple(home[8:14])
    _close_cassette_firm(ctx, ctx.world.object_pose("laser_mfp"))
    _open_lid(ctx, ctx.world.object_pose("laser_mfp"))
    _close_lid(ctx, ctx.world.object_pose("laser_mfp"))
    ctx.kinematics.move_joints("left_arm", home_l, steps=12)
    _set_panel_det(ctx, home_r)
    _relay_roof_overhang_left(ctx, ctx.world.object_pose("laser_mfp"), home_l)
    _forklift_roof_to_tray_right(ctx, ctx.world.object_pose("laser_mfp"), home_l, home_r)
    _deploy_stopper_det(ctx, home_l)
    ctx.kinematics.move_joints("left_arm", home_l, steps=8)
    ctx.kinematics.move_joints("right_arm", _PANEL_Q_SEQ[0][0], steps=8)
    _press(ctx, ctx.world.object_pose("laser_mfp"), "mode_button", (0.001, -0.031, 0.0152))
    ctx.kinematics.move_joints("right_arm", home_r, steps=8)
    ctx.step(24)
