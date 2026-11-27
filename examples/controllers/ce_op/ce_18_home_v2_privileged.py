import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    Vec3,
    compose_pose,
    invert_pose,
    normalize_vec3,
    quat_from_axis_angle_wxyz,
    quat_multiply_wxyz,
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


def _grip(ctx, side, pre_tip, tip, approach, closing, *, close_steps=40):
    arm = f"{side}_arm"
    gripper = f"{side}_gripper"
    ctx.end_effector.set_posture(gripper, "open", steps=32)
    ctx.kinematics.move_to_pose(arm, _tip_to_tcp(pre_tip, approach, closing), steps=40)
    ctx.step(8)
    pose = _tip_to_tcp(tip, approach, closing)
    ctx.kinematics.move_to_pose(arm, pose, steps=32)
    ctx.step(16)
    ctx.end_effector.set_posture(gripper, "half_open", steps=16)
    ctx.end_effector.set_posture(gripper, "closed", steps=close_steps)
    ctx.step(16)
    return pose


def _back_off(ctx, side, pose, distance):
    approach = rotate_vec_by_quat_wxyz(pose.wxyz, (0.0, 0.0, 1.0))
    clear = Pose3D(_offset(pose.xyz, approach, -distance), pose.wxyz)
    ctx.kinematics.move_to_pose(f"{side}_arm", clear, steps=24)


def _press(ctx, side, device, joint, face, normal, approach, closing):
    arm = f"{side}_arm"
    ctx.end_effector.set_posture(f"{side}_gripper", "closed", steps=24)
    above = _tip_to_tcp(_offset(face, normal, 0.070), approach, closing)
    ctx.kinematics.move_to_pose(arm, above, steps=40)
    ctx.step(8)
    ready = _tip_to_tcp(_offset(face, normal, 0.004), approach, closing)
    ctx.kinematics.move_to_pose(arm, ready, steps=24)
    for k in range(1, 13):
        tip = _offset(face, normal, 0.004 - 0.010 * k / 12)
        ctx.kinematics.move_to_pose(arm, _tip_to_tcp(tip, approach, closing), steps=8)
        ctx.step(8)
        if ctx.world.joint_value(device, joint) <= -0.0022:
            break
    ctx.step(24)
    ctx.kinematics.move_to_pose(arm, above, steps=24)


def _set_hygrostat(ctx) -> None:
    center = compose_pose(ctx.world.object_pose("humidifier"), Pose3D((0.020, -0.077, 0.119)))
    hand = _tip_to_tcp((0.0, 0.002, 0.0), (0.0, 0.819, -0.574), (1.0, 0.0, 0.0))

    def knob_pose(degrees: float) -> Pose3D:
        turn = quat_from_axis_angle_wxyz((0.0, 1.0, 0.0), math.radians(degrees))
        return compose_pose(center, compose_pose(Pose3D((0.0, 0.0, 0.0), turn), hand))

    start = knob_pose(-34.0)
    ctx.end_effector.set_posture("right_gripper", "open", steps=24)
    direction = rotate_vec_by_quat_wxyz(start.wxyz, (0.0, 0.0, 1.0))
    ctx.kinematics.move_to_pose("right_arm", Pose3D(_offset(start.xyz, direction, -0.072), start.wxyz), steps=48)
    ctx.kinematics.move_to_pose("right_arm", start, steps=56)
    ctx.step(16)
    ctx.end_effector.set_posture("right_gripper", "half_open", steps=16)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=40)
    # Turn the barrel with both pads engaged. Slow rotation keeps the knob tracking the wrist.
    for k in range(1, 19):
        pose = knob_pose(-34.0 + 5.0 * k)
        ctx.kinematics.move_to_pose("right_arm", pose, steps=32)
        if ctx.world.joint_value("humidifier", "hygrostat_dial") >= 0.90:
            break
    ctx.step(24)
    ctx.end_effector.set_posture("right_gripper", "open", steps=32)
    _back_off(ctx, "right", pose, 0.065)


def _carry(ctx, pose: Pose3D, min_ticks: int = 16) -> None:
    current = ctx.kinematics.ee_pose("right_arm")
    # At 120 Hz, 0.8 mm per tick leaves margin below the 0.12 m/s lift limit.
    ticks = max(min_ticks, 8 * math.ceil(math.dist(current.xyz, pose.xyz) / 0.0064))
    ctx.kinematics.move_to_pose("right_arm", pose, steps=ticks)


def _pour(ctx, tank_lip: Vec3) -> None:
    carafe = ctx.world.object_pose("pour_carafe")
    azimuth = math.radians(36.0)
    sa, ca = math.sin(azimuth), math.cos(azimuth)
    approach = rotate_vec_by_quat_wxyz(carafe.wxyz, (0.90 * sa, 0.90 * ca, -0.436))
    closing = rotate_vec_by_quat_wxyz(carafe.wxyz, (ca, -sa, 0.0))
    # Oppose the two circular sides above the handle, with the pads past the midline.
    tip = compose_pose(carafe, Pose3D((0.010 * sa, 0.010 * ca, 0.134))).xyz
    arm = "right_arm"
    gripper = "right_gripper"
    ctx.end_effector.set_posture(gripper, "open", steps=32)
    pregrasp = _tip_to_tcp(_offset(tip, approach, -0.095), approach, closing)
    ctx.kinematics.move_to_pose(arm, pregrasp, steps=80)
    grip = _tip_to_tcp(tip, approach, closing)
    ctx.kinematics.move_to_pose(arm, grip, steps=104)
    ctx.step(16)
    ctx.end_effector.set_posture(gripper, "half_open", steps=24)
    ctx.end_effector.set_posture(gripper, "closed", steps=32)
    _carry(ctx, Pose3D(_offset(grip.xyz, (0.0, 0.0, 1.0), 0.064), grip.wxyz))
    ctx.step(24)

    # Measure the seated wrist offset after the load transfers to the gripper.
    held = ctx.world.object_pose("pour_carafe")
    if held.xyz[2] < carafe.xyz[2] + 0.026:
        raise RuntimeError("Carafe did not rise with the right gripper")
    carafe_from_tcp = compose_pose(invert_pose(held), ctx.kinematics.ee_pose("right_arm"))
    spout = Pose3D((0.040, 0.0, 0.149))
    lip_from_tcp = compose_pose(invert_pose(spout), carafe_from_tcp)
    yaw = quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), math.radians(32.0))
    tx, ty, tz = tank_lip

    def pour_pose(angle: float) -> Pose3D:
        pitch = quat_from_axis_angle_wxyz((0.0, 1.0, 0.0), angle)
        # Clear the tank wall with the floor while lowering the notch to the opening.
        approach_frac = max(0.0, 1.0 - angle / math.radians(70.0))
        clearance = max(0.0, 0.152 * math.cos(angle) - 0.012)
        lip = Pose3D((tx + 0.14 * approach_frac, ty - 0.050 * approach_frac,
                      tz + clearance), quat_multiply_wxyz(yaw, pitch))
        return compose_pose(lip, lip_from_tcp)

    high = Pose3D((carafe.xyz[0], carafe.xyz[1], tz - 0.009), carafe.wxyz)
    high_tcp = compose_pose(high, carafe_from_tcp)
    _carry(ctx, high_tcp)
    _carry(ctx, pour_pose(0.0), min_ticks=176)
    angles = [math.radians(113.0 * k / 20) for k in range(21)]
    for angle in angles[1:]:
        ticks = 32 if angle >= math.radians(85.0) else 24
        ctx.kinematics.move_to_pose(arm, pour_pose(angle), steps=ticks)
    ctx.step(80)
    for _ in range(30):
        if ctx.world.fluid_containment_ratio("pour_carafe") <= 0.04:
            break
        ctx.step(8)
    for angle in angles[-2::-1]:
        ctx.kinematics.move_to_pose(arm, pour_pose(angle), steps=16)
    _carry(ctx, high_tcp)
    rest = compose_pose(carafe, carafe_from_tcp)
    _carry(ctx, rest)
    ctx.step(24)
    ctx.end_effector.set_posture(gripper, "half_open", steps=16)
    ctx.end_effector.set_posture(gripper, "open", steps=32)
    ctx.kinematics.move_to_pose(arm, pregrasp, steps=80)


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

    # Keep the lid clear of the pouring path with the left hand.
    lid_app = (0.0, 0.75, -0.661)
    lid_tip = _offset(lid_f, lid_app, 0.009)
    dy_l = lid_tip[1] - lid_pivot[1]
    dz_l = lid_tip[2] - lid_pivot[2]
    _grip(ctx, "left", _offset(lid_tip, lid_app, -0.075), lid_tip, lid_app, x_axis, close_steps=72)
    q_lid = ctx.world.joint_value("humidifier", "top_lid")
    lid_path = [_tip_to_tcp(lid_tip, lid_app, x_axis)]
    for k in range(1, 17):
        ang = (-1.62 - q_lid) * (k / 16.0)
        ca, sa = math.cos(ang), math.sin(ang)
        tip_k = (lid_f[0], lid_pivot[1] + dy_l * ca - dz_l * sa, lid_pivot[2] + dy_l * sa + dz_l * ca)
        lid_path.append(_tip_to_tcp(tip_k, lid_app, x_axis))
        ctx.kinematics.move_to_pose("left_arm", lid_path[-1], steps=16)
        if ctx.world.joint_value("humidifier", "top_lid") <= -1.49:
            break
    ctx.step(24)
    _pour(ctx, (hx, hy + 0.005, hz + 0.285))
    # Follow the captured handle back to its seat before releasing the lid.
    lid_pose = lid_path[-1]
    for lid_pose in lid_path[-2::-1]:
        ctx.kinematics.move_to_pose("left_arm", lid_pose, steps=16)
        if ctx.world.joint_value("humidifier", "top_lid") >= -0.022:
            break
    ctx.step(24)
    ctx.end_effector.set_posture("left_gripper", "open", steps=24)
    _back_off(ctx, "left", lid_pose, 0.065)

    # Pinch the ends of the 42 mm drawer pull. Its front edge is 10 mm ahead of the face.
    pull_dir = (0.0, 0.35, -0.93675)
    q_drawer = ctx.world.joint_value("humidifier", "aroma_drawer")
    pull_tip = _offset(_offset(ar_f, y_axis, q_drawer), pull_dir, 0.009)
    grip = _grip(ctx, "left", _offset(pull_tip, pull_dir, -0.065), pull_tip, pull_dir, x_axis)
    pulled = grip
    for k in range(1, 13):
        pulled = Pose3D(_offset(grip.xyz, y_axis, (-0.094 - q_drawer) * k / 12), grip.wxyz)
        ctx.kinematics.move_to_pose("left_arm", pulled, steps=8)
        if ctx.world.joint_value("humidifier", "aroma_drawer") <= -0.089:
            break
    ctx.step(16)
    ctx.end_effector.set_posture("left_gripper", "open", steps=24)
    ctx.step(8)
    _back_off(ctx, "left", pulled, 0.070)

    tablet = ctx.world.object_pose("aroma_tablet")
    tab_cls = rotate_vec_by_quat_wxyz(tablet.wxyz, x_axis)
    tablet_tip = _offset(tablet.xyz, up, 0.0055)
    grip = _grip(ctx, "left", _offset(tablet_tip, up, 0.105), tablet_tip, down, tab_cls, close_steps=40)
    ctx.step(8)
    ctx.kinematics.move_to_pose("left_arm", Pose3D(_offset(grip.xyz, up, 0.115), grip.wxyz), steps=24)
    drawer_y = hy - 0.072 + ctx.world.joint_value("humidifier", "aroma_drawer")
    # The bed is at 19.5 mm. Release above its side walls so the opening jaws clear them.
    seat = (hx, drawer_y, hz + 0.036)
    clear = _tip_to_tcp(_offset(seat, up, 0.085), down, x_axis)
    ctx.kinematics.move_to_pose("left_arm", clear, steps=24)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp(seat, down, x_axis), steps=24)
    ctx.step(8)
    ctx.end_effector.set_posture("left_gripper", "open", steps=32)
    ctx.step(16)
    ctx.kinematics.move_to_pose("left_arm", clear, steps=24)

    ctx.end_effector.set_posture("left_gripper", "closed", steps=24)
    push_dr = (0.0, 0.82, -0.572)
    for _ in range(2):
        q_drawer = ctx.world.joint_value("humidifier", "aroma_drawer")
        if q_drawer >= -0.004:
            break
        start_d = (hx, ar_f[1] + q_drawer - 0.004, hz + 0.029)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((start_d[0], start_d[1] - 0.055, start_d[2] + 0.025), push_dr, x_axis), steps=24)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp(start_d, push_dr, x_axis), steps=16)
        for k in range(1, 9):
            y_push = start_d[1] + (ar_f[1] + 0.005 - start_d[1]) * (k / 8)
            ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((hx, y_push, start_d[2]), push_dr, x_axis), steps=8)
            if ctx.world.joint_value("humidifier", "aroma_drawer") >= -0.001:
                break
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((hx, ar_f[1] - 0.065, start_d[2] + 0.040), push_dr, x_axis), steps=24)

    ctx.kinematics.move_joints("left_arm", home_l, steps=32)
    _set_hygrostat(ctx)

    btn_app = (0.0, 0.68, -0.733)
    for side, joint, frame, home in (
        ("left", "night_mist_button", "night_mist_press", home_l),
        ("right", "uv_sterilize_button", "uv_sterilize_press", home_r),
    ):
        key = ctx.world.named_frame("humidifier", frame).xyz
        # Front-angled press on outer keycap half (y=hy-0.098) clears FrontWall (y=-0.064).
        _press(ctx, side, "humidifier", joint, (key[0], hy - 0.098, hz + 0.0645), up, btn_app, x_axis)
        ctx.kinematics.move_joints(f"{side}_arm", home, steps=32)
    ctx.step(48)
