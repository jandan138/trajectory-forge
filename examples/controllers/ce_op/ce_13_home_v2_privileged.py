import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    Vec3,
    compose_pose,
    invert_pose,
    normalize_vec3,
    quat_from_axis_angle_wxyz,
    rotate_vec_by_quat_wxyz,
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


def _grip(ctx, side, pre_tip, tip, approach, closing, *, close_steps=72):
    arm = f"{side}_arm"
    gripper = f"{side}_gripper"
    ctx.end_effector.set_posture(gripper, "open", steps=32)
    ctx.kinematics.move_to_pose(arm, _tip_to_tcp(pre_tip, approach, closing), steps=40)
    ctx.step(8)
    pose = _tip_to_tcp(tip, approach, closing)
    ctx.kinematics.move_to_pose(arm, pose, steps=32)
    ctx.step(16)
    _ramp_jaws(ctx, gripper, 0.044, 0.0, ticks=close_steps)
    return pose


def _back_off(ctx, side, pose, distance):
    approach = rotate_vec_by_quat_wxyz(pose.wxyz, (0.0, 0.0, 1.0))
    clear = Pose3D(_offset(pose.xyz, approach, -distance), pose.wxyz)
    ctx.kinematics.move_to_pose(f"{side}_arm", clear, steps=24)


def _press(ctx, side, face, normal, approach, closing):
    arm = f"{side}_arm"
    ctx.end_effector.set_posture(f"{side}_gripper", "closed", steps=16)
    above = _tip_to_tcp(_offset(face, normal, 0.070), approach, closing)
    ctx.kinematics.move_to_pose(arm, above, steps=56)
    ctx.step(8)
    ready = _tip_to_tcp(_offset(face, normal, 0.004), approach, closing)
    ctx.kinematics.move_to_pose(arm, ready, steps=40)
    pressed = _tip_to_tcp(_offset(face, normal, -0.006), approach, closing)
    ctx.kinematics.move_to_pose(arm, pressed, steps=48)
    ctx.step(40)
    ctx.kinematics.move_to_pose(arm, above, steps=32)


def _ramp_jaws(ctx, group, start, end, ticks=32):
    for tick in range(1, ticks + 1):
        target = start + (end - start) * tick / ticks
        ctx.end_effector.command_targets(group, (target, target), steps=1)
    ctx.step(16)


def _pour_water(ctx, hx, hy, hz):
    arm = "right_arm"
    group = "right_gripper"
    carafe = ctx.world.object_pose("pour_carafe")
    approach = rotate_vec_by_quat_wxyz(carafe.wxyz, (0.0, 0.8660254, -0.5))
    closing = rotate_vec_by_quat_wxyz(carafe.wxyz, (0.0, 0.0, 1.0))
    # The lower finger supports the bottom bar while the upper finger closes on the top bar.
    handle = compose_pose(carafe, Pose3D((-0.063, 0.031, 0.0601))).xyz
    grip = _tip_to_tcp(handle, approach, closing)
    pre = compose_pose(grip, Pose3D((0.0, 0.0, -0.055)))
    above = Pose3D(_offset(pre.xyz, (0.0, 0.0, 1.0), 0.100), grip.wxyz)
    ctx.end_effector.set_posture(group, "open", steps=32)
    ctx.kinematics.move_to_pose(arm, above, steps=80)
    ctx.kinematics.move_to_pose(arm, pre, steps=80)
    ctx.kinematics.move_to_pose(arm, grip, steps=72)
    ctx.step(16)
    _ramp_jaws(ctx, group, 0.044, 0.0)

    up = (0.0, 0.0, 1.0)
    lift = Pose3D(_offset(grip.xyz, up, 0.060), grip.wxyz)
    ctx.kinematics.move_to_pose(arm, lift, steps=72)
    ctx.step(16)
    loaded = ctx.world.object_pose("pour_carafe")
    if loaded.xyz[2] < carafe.xyz[2] + 0.045:
        raise RuntimeError("Carafe slipped during lift")
    high = Pose3D(_offset(grip.xyz, up, 0.298), grip.wxyz)
    ctx.kinematics.move_to_pose(arm, high, steps=264)
    ctx.step(16)
    loaded = ctx.world.object_pose("pour_carafe")
    if loaded.xyz[2] < hz + 0.255:
        raise RuntimeError("Carafe slipped during transport")
    held_tcp = compose_pose(invert_pose(loaded), ctx.kinematics.ee_pose(arm))

    # Sweep the notched lip over the cavity. Lower it as the floor clears the tank rim.
    spout = Pose3D((0.042, 0.0, 0.149))
    heading = quat_from_axis_angle_wxyz(up, math.radians(10.0))

    def pour_pose(angle: float) -> Pose3D:
        tilt = Pose3D((0.0, 0.0, 0.0), quat_from_axis_angle_wxyz((0.0, 1.0, 0.0), angle))
        height = hz + 0.285 + 0.160 * max(0.0, math.cos(angle))
        outside = max(0.0, 1.0 - angle / math.radians(70.0))
        lip_xyz = (hx + 0.140 * outside, hy + 0.005 - 0.040 * outside, height)
        lip = compose_pose(Pose3D(lip_xyz, heading), tilt)
        return compose_pose(compose_pose(lip, invert_pose(spout)), held_tcp)

    ctx.kinematics.move_to_pose(arm, pour_pose(0.0), steps=248)
    ctx.step(16)
    for i in range(1, 23):
        ctx.kinematics.move_to_pose(arm, pour_pose(math.radians(5.0 * i)), steps=16)
    ctx.step(96)
    for i in range(21, -1, -1):
        ctx.kinematics.move_to_pose(arm, pour_pose(math.radians(5.0 * i)), steps=16)

    ctx.kinematics.move_to_pose(arm, high, steps=248)
    # Set the floor down before releasing the carafe.
    ctx.kinematics.move_to_pose(arm, grip, steps=312)
    ctx.step(32)
    _ramp_jaws(ctx, group, 0.0, 0.044)
    ctx.kinematics.move_to_pose(arm, pre, steps=48)
    ctx.kinematics.move_to_pose(arm, above, steps=40)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=40)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])
    x_axis = (1.0, 0.0, 0.0)
    y_axis = (0.0, 1.0, 0.0)
    up = (0.0, 0.0, 1.0)
    down = (0.0, 0.15, -0.988686)
    hx, hy, hz = ctx.world.object_pose("humidifier").xyz
    lid_f = ctx.world.named_frame("humidifier", "lid_handle").xyz
    ar_f = ctx.world.named_frame("humidifier", "aroma_drawer_handle").xyz
    lid_pivot = (hx, hy + 0.081, hz + 0.275)

    # Keep the fingers in the hinge plane as the lid handle swings toward the rear.
    lid_app = (0.0, 0.75, -0.661)
    lid_tip = _offset(lid_f, lid_app, 0.009)
    dy_l = lid_tip[1] - lid_pivot[1]
    dz_l = lid_tip[2] - lid_pivot[2]
    _grip(ctx, "left", _offset(lid_tip, lid_app, -0.075), lid_tip, lid_app, x_axis)
    q_lid = ctx.world.joint_value("humidifier", "top_lid")
    lid_path = [_tip_to_tcp(lid_tip, lid_app, x_axis)]
    for k in range(1, 15):
        ang = (-1.63 - q_lid) * (k / 14.0)
        ca, sa = math.cos(ang), math.sin(ang)
        tip = (lid_f[0], lid_pivot[1] + dy_l * ca - dz_l * sa, lid_pivot[2] + dy_l * sa + dz_l * ca)
        pose = _tip_to_tcp(tip, lid_app, x_axis)
        ctx.kinematics.move_to_pose("left_arm", pose, steps=16)
        lid_path.append(pose)
        if ctx.world.joint_value("humidifier", "top_lid") <= -1.53:
            break
    ctx.step(24)

    _pour_water(ctx, hx, hy, hz)

    # Keep the left fingers on the handle through the pour and retrace the hinge arc.
    for pose in lid_path[-2::-1]:
        ctx.kinematics.move_to_pose("left_arm", pose, steps=16)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp(_offset(lid_tip, up, -0.006), lid_app, x_axis), steps=16)
    ctx.step(24)
    _ramp_jaws(ctx, "left_gripper", 0.0, 0.044)
    _back_off(ctx, "left", lid_path[0], 0.070)

    # Pinch the ends of the 42 mm drawer pull. Its front edge is 10 mm ahead of the face.
    pull_dir = (0.0, 0.35, -0.93675)
    q_drawer = ctx.world.joint_value("humidifier", "aroma_drawer")
    pull_tip = _offset(_offset(ar_f, y_axis, q_drawer), pull_dir, 0.009)
    grip = _grip(ctx, "left", _offset(pull_tip, pull_dir, -0.065), pull_tip, pull_dir, x_axis)
    pulled = grip
    for k in range(1, 13):
        pulled = Pose3D(_offset(grip.xyz, y_axis, (-0.094 - q_drawer) * k / 12), grip.wxyz)
        ctx.kinematics.move_to_pose("left_arm", pulled, steps=16)
        if ctx.world.joint_value("humidifier", "aroma_drawer") <= -0.089:
            break
    ctx.step(24)
    ctx.end_effector.set_posture("left_gripper", "open", steps=32)
    ctx.step(16)
    _back_off(ctx, "left", pulled, 0.070)

    tablet = ctx.world.object_pose("aroma_tablet")
    tab_cls = rotate_vec_by_quat_wxyz(tablet.wxyz, x_axis)
    tablet_tip = _offset(tablet.xyz, up, 0.0055)
    grip = _grip(ctx, "left", _offset(tablet_tip, up, 0.105), tablet_tip, down, tab_cls, close_steps=72)
    ctx.step(16)
    ctx.kinematics.move_to_pose("left_arm", Pose3D(_offset(grip.xyz, up, 0.115), grip.wxyz), steps=32)
    drawer_y = hy - 0.072 + ctx.world.joint_value("humidifier", "aroma_drawer")
    # The bed is at 19.5 mm. Release above its side walls so the opening jaws clear them.
    seat = (hx, drawer_y, hz + 0.036)
    clear = _tip_to_tcp(_offset(seat, up, 0.085), down, x_axis)
    ctx.kinematics.move_to_pose("left_arm", clear, steps=40)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp(seat, down, x_axis), steps=32)
    ctx.step(16)
    ctx.end_effector.set_posture("left_gripper", "open", steps=40)
    ctx.step(24)
    ctx.kinematics.move_to_pose("left_arm", clear, steps=32)

    ctx.end_effector.set_posture("left_gripper", "closed", steps=32)
    push_dr = (0.0, 0.82, -0.572)
    for _ in range(2):
        q_drawer = ctx.world.joint_value("humidifier", "aroma_drawer")
        if q_drawer >= -0.004:
            break
        start_d = (hx, ar_f[1] + q_drawer - 0.004, hz + 0.029)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((start_d[0], start_d[1] - 0.055, start_d[2] + 0.025), push_dr, x_axis), steps=32)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp(start_d, push_dr, x_axis), steps=24)
        for k in range(1, 9):
            y_push = start_d[1] + (ar_f[1] + 0.005 - start_d[1]) * (k / 8)
            ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((hx, y_push, start_d[2]), push_dr, x_axis), steps=16)
            if ctx.world.joint_value("humidifier", "aroma_drawer") >= -0.001:
                break
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((hx, ar_f[1] - 0.065, start_d[2] + 0.040), push_dr, x_axis), steps=32)

    # Tangential closed-tip flute ratchet across the 12 o'clock crest into [-1.0472, -0.6109].
    hum_pose = ctx.world.object_pose("humidifier")
    f_app = rotate_vec_by_quat_wxyz(hum_pose.wxyz, (0.0, 0.65, -0.7599))
    f_cls = rotate_vec_by_quat_wxyz(hum_pose.wxyz, x_axis)
    r_c, r_out = 0.0178, 0.028
    ctx.end_effector.set_posture("right_gripper", "closed", steps=32)

    def _dial_pt(radius: float, ang_rad: float) -> Vec3:
        return compose_pose(
            hum_pose,
            Pose3D((0.020 + radius * math.sin(ang_rad), -0.0785, 0.119 + radius * math.cos(ang_rad))),
        ).xyz

    for _ in range(8):
        if ctx.world.joint_value("humidifier", "hygrostat_dial") <= -0.85:
            break
        a0 = 0.50
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(_dial_pt(r_out, a0), f_app, f_cls), steps=8)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(_dial_pt(r_c, a0), f_app, f_cls), steps=8)
        a_end = a0
        for k in range(1, 10):
            a = 0.50 - 1.00 * k / 9
            a_end = a
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(_dial_pt(r_c, a), f_app, f_cls), steps=8)
            if ctx.world.joint_value("humidifier", "hygrostat_dial") <= -0.85:
                break
        ctx.step(8)
        out_pt = _dial_pt(r_out, a_end)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(out_pt, f_app, f_cls), steps=8)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(_offset(out_pt, up, 0.025), f_app, f_cls), steps=8)

    btn_app = (0.0, 0.68, -0.733)
    for side, frame, home in (
        ("left", "night_mist_press", home_l),
        ("right", "uv_sterilize_press", home_r),
    ):
        key = ctx.world.named_frame("humidifier", frame).xyz
        # Front-angled press on outer keycap half (y=hy-0.098) clears FrontWall (y=-0.064).
        _press(ctx, side, (key[0], hy - 0.098, hz + 0.0645), up, btn_app, x_axis)
        ctx.kinematics.move_joints(f"{side}_arm", home, steps=24)
    ctx.step(56)
