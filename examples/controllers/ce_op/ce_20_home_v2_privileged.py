import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    Vec3,
    compose_pose,
    invert_pose,
    normalize_vec3,
    rotate_vec_by_quat_wxyz,
    sample_revolute_arc_poses,
)


def _offset(point: Vec3, direction: Vec3, distance: float) -> Vec3:
    return (
        point[0] + distance * direction[0],
        point[1] + distance * direction[1],
        point[2] + distance * direction[2],
    )


def _tip_to_tcp(tip_xyz: Vec3, approach_dir: Vec3, closing_dir: Vec3) -> Pose3D:
    approach = normalize_vec3(approach_dir)
    return Pose3D.from_approach(
        xyz=_offset(tip_xyz, approach, -0.0226),
        approach_dir=approach,
        closing_dir=closing_dir,
    )


def _grip(ctx, side, pre_tip, tip, approach, closing, *, close_steps=36):
    arm = f"{side}_arm"
    gripper = f"{side}_gripper"
    ctx.end_effector.set_posture(gripper, "open", steps=16)
    ctx.kinematics.move_to_pose(arm, _tip_to_tcp(pre_tip, approach, closing), steps=22)
    ctx.step(4)
    pose = _tip_to_tcp(tip, approach, closing)
    ctx.kinematics.move_to_pose(arm, pose, steps=16)
    ctx.step(8)
    ctx.end_effector.set_posture(gripper, "closed", steps=close_steps)
    ctx.step(8)
    return pose


def _back_off(ctx, side, pose, distance):
    approach = rotate_vec_by_quat_wxyz(pose.wxyz, (0.0, 0.0, 1.0))
    clear = Pose3D(_offset(pose.xyz, approach, -distance), pose.wxyz)
    ctx.kinematics.move_to_pose(f"{side}_arm", clear, steps=12)


def _turn(ctx, side, device, joint, pose, pivot, axis, goal, reached, segments):
    start = ctx.world.joint_value(device, joint)
    sign = 1 if goal >= start else -1
    arc = sample_revolute_arc_poses(
        start_pose=pose, hinge_pivot_world=pivot, hinge_axis_world=axis,
        delta_angle_rad=goal - start, num_waypoints=segments + 1,
    )
    for pose in arc[1:]:
        ctx.kinematics.move_to_pose(f"{side}_arm", pose, steps=4)
        if sign * (ctx.world.joint_value(device, joint) - reached) >= 0:
            break
    return pose


def _press(ctx, side, device, joint, face, normal, approach, closing):
    arm = f"{side}_arm"
    ctx.end_effector.set_posture(f"{side}_gripper", "closed", steps=10)
    above = _tip_to_tcp(_offset(face, normal, 0.070), approach, closing)
    ctx.kinematics.move_to_pose(arm, above, steps=14)
    ctx.step(2)
    ready = _tip_to_tcp(_offset(face, normal, 0.004), approach, closing)
    ctx.kinematics.move_to_pose(arm, ready, steps=10)
    for k in range(1, 13):
        # Travel past fingertip compliance until joint hits latched -0.0020.
        tip = _offset(face, normal, 0.004 - 0.0110 * k / 12)
        ctx.kinematics.move_to_pose(arm, _tip_to_tcp(tip, approach, closing), steps=4)
        ctx.step(4)
        if ctx.world.joint_value(device, joint) <= -0.0022:
            break
    ctx.step(20)
    ctx.kinematics.move_to_pose(arm, above, steps=10)


def _plug_and_switch(ctx, home_l):
    device = "power_station"
    station = ctx.world.object_pose(device)
    px, py, pz = station.xyz
    up = (0.0, 0.0, 1.0)
    x_axis = (1.0, 0.0, 0.0)
    down = (0.0, 0.15, -0.988686)
    # Closed-tip single-arc upward push under SafetyCover LiftLip around (-0.072, +0.033, +0.080).
    cov_piv = compose_pose(station, Pose3D((-0.072, 0.033, 0.080)))
    cov_app = rotate_vec_by_quat_wxyz(station.wxyz, (0.0, 0.78, -0.6258))
    cov_cls = rotate_vec_by_quat_wxyz(station.wxyz, x_axis)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    for _cov_try in range(2):
        q_cov = ctx.world.joint_value(device, "safety_cover")
        if q_cov <= -1.52:
            break
        pre_cov = compose_pose(cov_piv, Pose3D((0.0, -0.068, 0.004)))
        hook_cov = compose_pose(cov_piv, Pose3D((0.0, -0.050, 0.005)))
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp(pre_cov.xyz, cov_app, cov_cls), steps=20)
        ctx.step(4)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp(hook_cov.xyz, cov_app, cov_cls), steps=16)
        ctx.step(4)
        pose_cov = _tip_to_tcp(hook_cov.xyz, cov_app, cov_cls)
        for k in range(1, 17):
            ang_k = q_cov + (-1.74 - q_cov) * (k / 16.0)
            p_k = compose_pose(
                compose_pose(cov_piv, Pose3D((0.0, 0.0, 0.0), (math.cos(ang_k / 2), math.sin(ang_k / 2), 0.0, 0.0))),
                Pose3D((0.0, -0.050, 0.005)),
            )
            pose_cov = _tip_to_tcp(p_k.xyz, cov_app, cov_cls)
            ctx.kinematics.move_to_pose("left_arm", pose_cov, steps=6)
            if ctx.world.joint_value(device, "safety_cover") <= -1.52:
                break
        ctx.step(6)
        _back_off(ctx, "left", pose_cov, 0.065)

    ctx.kinematics.move_joints("left_arm", home_l, steps=14)
    plug = ctx.world.object_pose("ac_plug")
    local_tip = (0.0, -0.004, 0.029)
    tip = compose_pose(plug, Pose3D(local_tip)).xyz
    plug_app = rotate_vec_by_quat_wxyz(plug.wxyz, down)
    plug_cls = rotate_vec_by_quat_wxyz(plug.wxyz, x_axis)
    grip = _grip(ctx, "left", _offset(tip, up, 0.115), tip, plug_app, plug_cls, close_steps=26)
    ctx.step(4)
    held_plug = compose_pose(invert_pose(ctx.world.object_pose("ac_plug")), ctx.kinematics.ee_pose("left_arm"))
    ctx.kinematics.move_to_pose("left_arm", Pose3D(_offset(grip.xyz, up, 0.125), grip.wxyz), steps=20)
    at_socket = compose_pose(station, Pose3D((-0.076, -0.083, 0.0185)))
    ctx.kinematics.move_to_pose("left_arm", compose_pose(compose_pose(at_socket, Pose3D((0.0, -0.055, 0.080))), held_plug), steps=22)
    ctx.kinematics.move_to_pose("left_arm", compose_pose(compose_pose(at_socket, Pose3D((0.0, -0.055, 0.0020))), held_plug), steps=16)
    ctx.step(4)
    target = compose_pose(at_socket, held_plug)
    for travel, height, steps in (
        (-0.028, 0.0010, 14),
        (-0.014, 0.0005, 10),
        (-0.005, 0.0, 6),
        (0.0, 0.0, 6),
        (0.0010, 0.0, 5),
        (0.0018, 0.0, 5),
    ):
        target = compose_pose(compose_pose(at_socket, Pose3D((0.0, travel, height))), held_plug)
        ctx.kinematics.move_to_pose("left_arm", target, steps=steps)
        ctx.step(2)
        if travel >= -0.005 and ctx.world.joint_value(device, "ac_socket") >= 0.0012:
            break
    ctx.step(8)
    ctx.end_effector.set_posture("left_gripper", "open", steps=18)
    ctx.step(10)
    _back_off(ctx, "left", target, 0.075)

    switch = ctx.world.named_frame(device, "master_switch_on").xyz
    # The named point is already on the forward end of the 22 mm rocker paddle.
    ctx.end_effector.set_posture("left_gripper", "closed", steps=12)
    ready = _tip_to_tcp(_offset(switch, up, 0.065), down, x_axis)
    ctx.kinematics.move_to_pose("left_arm", ready, steps=14)
    switch_tip = _offset(switch, up, -0.0006)
    contact = _tip_to_tcp(switch_tip, down, x_axis)
    ctx.kinematics.move_to_pose("left_arm", contact, steps=12)
    rocker_pivot = (px - 0.072, py + 0.009, pz + 0.0813)
    pressed = _turn(ctx, "left", device, "master_switch", contact, rocker_pivot, x_axis, 0.32, 0.25, 10)
    ctx.step(6)
    _back_off(ctx, "left", pressed, 0.075)


def _set_outputs(ctx, side):
    device = "power_station"
    up = (0.0, 0.0, 1.0)
    x_axis = (1.0, 0.0, 0.0)
    down = (0.0, 0.15, -0.988686)
    for joint, frame in (("ac_zone_button", "ac_zone_press"), ("usb_pd_button", "usb_pd_press")):
        key = ctx.world.named_frame(device, frame).xyz
        _press(ctx, side, device, joint, _offset(key, up, -0.0001), up, down, x_axis)

    dial = ctx.world.named_frame(device, "mode_selector_grip").xyz
    _ = dial
    ps_pose = ctx.world.object_pose(device)
    shaft = compose_pose(ps_pose, Pose3D((0.080, 0.008, 0.075))).xyz
    arm_d = f"{side}_arm"
    grp_d = f"{side}_gripper"
    rim_dir = (0.0, 0.0, -1.0)
    cx_d, cy_d, cz_d = shaft[0], shaft[1], shaft[2] + 0.0085
    r_c, r_out = 0.0118, 0.026
    ctx.end_effector.set_posture(grp_d, "closed", steps=16)
    q_vert_seed = (-0.1794, 2.4791, 1.6762, -0.7679, 0.0, -0.0002)
    ctx.kinematics.move_joints(arm_d, q_vert_seed, steps=12)
    for dial_try in range(13):
        if dial_try >= 11 and ctx.world.joint_value(device, "mode_selector") >= 1.55:
            break
        th0 = -1.75
        cls0 = (-math.sin(th0), math.cos(th0), 0.0)
        ctx.kinematics.move_to_pose(arm_d, _tip_to_tcp((cx_d + r_out * math.cos(th0), cy_d + r_out * math.sin(th0), cz_d + 0.040), rim_dir, cls0), steps=8)
        ctx.kinematics.move_to_pose(arm_d, _tip_to_tcp((cx_d + r_out * math.cos(th0), cy_d + r_out * math.sin(th0), cz_d), rim_dir, cls0), steps=8)
        ctx.step(4)
        ctx.kinematics.move_to_pose(arm_d, _tip_to_tcp((cx_d + r_c * math.cos(th0), cy_d + r_c * math.sin(th0), cz_d), rim_dir, cls0), steps=8)
        th_end = th0
        for k in range(1, 9):
            th = -1.75 + 1.87 * (k / 8.0)
            th_end = th
            t_cls = (-math.sin(th), math.cos(th), 0.0)
            ctx.kinematics.move_to_pose(arm_d, _tip_to_tcp((cx_d + r_c * math.cos(th), cy_d + r_c * math.sin(th), cz_d), rim_dir, t_cls), steps=8)
        ctx.step(4)
        cls_end = (-math.sin(th_end), math.cos(th_end), 0.0)
        ctx.kinematics.move_to_pose(arm_d, _tip_to_tcp((cx_d + r_out * math.cos(th_end), cy_d + r_out * math.sin(th_end), cz_d), rim_dir, cls_end), steps=8)
        ctx.kinematics.move_to_pose(arm_d, _tip_to_tcp((cx_d + r_out * math.cos(th_end), cy_d + r_out * math.sin(th_end), cz_d + 0.040), rim_dir, cls_end), steps=8)
    ctx.end_effector.set_posture(grp_d, "closed", steps=8)


def _dock_key(ctx, joint, frame):
    face = ctx.world.named_frame("wireless_uv_dock", frame).xyz
    up = (0.0, 0.0, 1.0)
    down = (0.0, 0.15, -0.988686)
    closing = (1.0, 0.0, 0.0)
    # Both closed tips fit the 16 x 14 mm cap and its 22 x 20 mm deck opening.
    hover = _tip_to_tcp(_offset(face, up, 0.075), down, closing)
    ctx.kinematics.move_to_pose("right_arm", hover, steps=14)
    ctx.step(2)
    ready = _tip_to_tcp(_offset(face, up, -0.002), down, closing)
    ctx.kinematics.move_to_pose("right_arm", ready, steps=10)
    ctx.step(2)
    height = -0.0035
    for _ in range(8):
        tip = _offset(face, up, height)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(tip, down, closing), steps=4)
        q = ctx.world.joint_value("wireless_uv_dock", joint)
        if -0.0052 <= q <= -0.0028:
            break
        correction = max(-0.0005, min(0.0005, -0.004 - q))
        height = max(-0.0050, min(0.0, height + correction))
    return hover


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=20)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    _plug_and_switch(ctx, home_l)
    ctx.kinematics.move_joints("left_arm", home_l, steps=14)
    _set_outputs(ctx, "left")
    ctx.kinematics.move_joints("left_arm", home_l, steps=16)

    device = "wireless_uv_dock"
    dock = ctx.world.object_pose(device)
    dx, dy, dz = dock.xyz
    x_axis = (1.0, 0.0, 0.0)
    up = (0.0, 0.0, 1.0)
    down = (0.0, 0.15, -0.988686)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=12)
    for joint, frame in (("power_btn", "power_btn_press"), ("pod_eject_btn", "pod_eject_press")):
        hover = _dock_key(ctx, joint, frame)
        ctx.step(12)
        ctx.kinematics.move_to_pose("right_arm", hover, steps=10)
    for _ in range(4):
        if ctx.world.joint_value(device, "uv_pod_lid") <= -1.17:
            break
        ctx.step(8)

    for obj, frame, grip_z in (
        ("smartphone", "left_phone_coil", 0.0055),
        ("tws_earbuds_case", "right_earbuds_bay", 0.015),
    ):
        prop = ctx.world.object_pose(obj)
        approach = rotate_vec_by_quat_wxyz(prop.wxyz, down)
        closing = rotate_vec_by_quat_wxyz(prop.wxyz, x_axis)
        tip = compose_pose(prop, Pose3D((0.0, 0.0, grip_z))).xyz
        grip = _grip(ctx, "right", _offset(tip, up, 0.120), tip, approach, closing, close_steps=36)
        ctx.kinematics.move_to_pose("right_arm", Pose3D(_offset(grip.xyz, up, 0.140), grip.wxyz), steps=14)
        seat = ctx.world.named_frame(device, frame).xyz
        is_phone = obj == "smartphone"
        drop_z = seat[2] + grip_z + 0.001 if is_phone else dz + 0.052
        drop = (seat[0], seat[1], drop_z)
        entry = (drop[0], drop[1] - 0.032 if is_phone else drop[1], drop_z)
        hover = _tip_to_tcp(_offset(entry, up, 0.125), down, x_axis)
        ctx.kinematics.move_to_pose("right_arm", hover, steps=14)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(entry, down, x_axis), steps=12)
        if is_phone:
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(drop, down, x_axis), steps=12)
        ctx.step(4)
        ctx.end_effector.set_posture("right_gripper", "open", steps=14)
        ctx.step(6)
        ctx.kinematics.move_to_pose("right_arm", hover, steps=12)

    # Push UvPodFrontLeaf -Y past -0.68 rad latch with IK-clean top-down wrist, then retract upward (matching ce_12).
    ctx.end_effector.set_posture("right_gripper", "closed", steps=12)
    lid_x = dx + 0.068
    lid_app = (0.0, 0.22, -0.9755)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((lid_x, dy + 0.088, dz + 0.155), lid_app, x_axis), steps=12)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((lid_x, dy + 0.088, dz + 0.100), lid_app, x_axis), steps=10)
    for k in range(1, 11):
        frac = k / 10.0
        wy = dy + 0.088 + (-0.004 - 0.088) * frac
        wz = dz + 0.100 + (0.072 - 0.100) * frac
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((lid_x, wy, wz), lid_app, x_axis), steps=8)
        if ctx.world.joint_value(device, "uv_pod_lid") >= -0.68:
            break
    ctx.step(4)
    cur_y = float(ctx.kinematics.ee_pose("right_arm").xyz[1])
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((lid_x, cur_y, dz + 0.150), lid_app, x_axis), steps=10)
    for _ in range(14):
        if (abs(ctx.world.joint_value(device, "uv_pod_lid")) <= 0.007
                and abs(ctx.world.joint_value(device, "uv_pod_leaf_knuckle")) <= 0.007):
            break
        ctx.step(8)

    # BOOST is terminal: open 10mm -X jaw offset matching ce_12 so button stays in [-0.0052, -0.0028] for 64 ticks.
    ctx.end_effector.set_posture("right_gripper", "open", steps=14)
    bbtn = ctx.world.named_frame(device, "boost_btn_press").xyz
    bx_tcp = bbtn[0] + 0.0439
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx_tcp, bbtn[1], bbtn[2] + 0.085), down, x_axis), steps=14)
    for i in range(1, 18):
        off_z = 0.008 - 0.017 * (i / 17.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx_tcp, bbtn[1], bbtn[2] + off_z), down, x_axis), steps=4)
        if -0.0052 <= ctx.world.joint_value(device, "boost_btn") <= -0.0028:
            break
    ctx.step(48)
