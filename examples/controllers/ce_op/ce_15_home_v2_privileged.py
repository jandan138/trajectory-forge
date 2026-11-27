import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    compose_pose,
    invert_pose,
    quat_from_axis_angle_wxyz,
    rotate_vec_by_quat_wxyz,
    slerp_wxyz,
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


def _revolute_push_waypoints(
    start_tip_xyz: tuple[float, float, float],
    pivot_world: tuple[float, float, float],
    delta_angle_x_rad: float,
    approach_dir: tuple[float, float, float],
    closing_dir: tuple[float, float, float],
    num_waypoints: int,
) -> list[Pose3D]:
    dy0 = start_tip_xyz[1] - pivot_world[1]
    dz0 = start_tip_xyz[2] - pivot_world[2]
    wps: list[Pose3D] = []
    for idx in range(num_waypoints):
        ang = delta_angle_x_rad * (idx / max(1, num_waypoints - 1))
        c, s = math.cos(ang), math.sin(ang)
        tip = (
            start_tip_xyz[0],
            pivot_world[1] + dy0 * c - dz0 * s,
            pivot_world[2] + dy0 * s + dz0 * c,
        )
        wps.append(_tip_to_tcp(tip, approach_dir, closing_dir))
    return wps


def _move_sampled(ctx, arm, goal, *, steps):
    # The action replay interpolates between the targets at eight-tick
    # intervals. Put each change of direction on one of those samples.
    start = ctx.kinematics.ee_pose(arm)
    frames = steps // 8
    path = []
    for frame in range(1, frames + 1):
        u = frame / frames
        xyz = tuple(a + u * (b - a) for a, b in zip(start.xyz, goal.xyz))
        path.append(Pose3D(xyz, slerp_wxyz(start.wxyz, goal.wxyz, u)))
    try:
        ctx.kinematics.follow_cartesian_path(arm, path, steps_per_waypoint=8)
    except RuntimeError:
        ctx.kinematics.follow_cartesian_path(arm, [goal], steps_per_waypoint=steps)


def _turn_and_brew(ctx, left_home, right_home):
    center = ctx.world.joint_info("espresso_center", "select_dial").pivot_world
    # The first path wipes the face of the knob. The other two paths push
    # different lobes of the cross, always turning about the positive shaft axis.
    face = ((-0.004, 0.005), (-0.006, 0.009), (0.008, 0.010),
            (0.010, 0.000), (0.009, -0.014))
    outside = ((-0.022, 0.016), (-0.005, 0.014), (0.013, 0.012),
               (0.013, -0.002), (0.011, -0.018))
    across_top = tuple((-z, x) for x, z in outside)
    under_left = tuple((-x, -z) for x, z in outside)
    upright = (0.0, 0.5724, 0.82)
    left_approach = (0.50, 0.73, -0.466)
    face_stroke = ("left_arm", "left_gripper", left_home, left_approach, upright, face)
    cross_grip = ("left_arm", "left_gripper", left_home, left_approach, (1.0, 0.0, 0.0), face)
    lower_stroke = ("left_arm", "left_gripper", left_home, left_approach, upright, under_left)
    top_stroke = ("right_arm", "right_gripper", right_home, (0.0, 0.82, -0.5724), upright, across_top)

    # Keep both approaches in the recorded trajectory. The amount of rotation
    # from any one sweep varies with contact, including on a 15 Hz replay.
    strokes = (face_stroke, cross_grip, face_stroke, top_stroke, face_stroke,
               lower_stroke, top_stroke, face_stroke, face_stroke)
    press_pose = None
    best_angle = -1.0
    brew_held = False
    progress = []
    closed = set()
    for number, (arm, group, home, approach, closing, edge) in enumerate(strokes):
        def tip(x, y, z):
            return _tip_to_tcp((center[0] + x, center[1] + y, center[2] + z), approach, closing)

        if group in closed:
            ctx.step(8)
        else:
            ctx.end_effector.set_posture(group, "closed", steps=16)
            closed.add(group)
        _move_sampled(ctx, arm, tip(edge[0][0], -0.015, edge[0][1]), steps=32)
        for x, z in edge:
            target = tip(x, 0.003, z)
            ctx.kinematics.follow_cartesian_path(arm, [target], steps_per_waypoint=16)
            angle = ctx.world.joint_value("espresso_center", "select_dial")
            pressure = ctx.world.joint_value("espresso_center", "dial_press")
            if pressure >= 0.0018 and angle >= best_angle:
                best_angle = angle
                press_pose = (arm, home, target)
                if not brew_held and angle >= 1.20:
                    ctx.step(24)
                    brew_held = ctx.world.joint_value("espresso_center", "dial_press") >= 0.0015
        _move_sampled(ctx, arm, tip(x, -0.015, z), steps=16)
        ctx.step(16)
        if number == len(strokes) - 1 and ctx.world.joint_value("espresso_center", "select_dial") < 1.1345:
            # Re-enter below the left lobe when the upper face stops turning
            # the selector. The arm is already clear of the shaft here.
            _move_sampled(ctx, arm, tip(under_left[0][0], -0.015, under_left[0][1]), steps=16)
            for x, z in under_left:
                target = tip(x, 0.003, z)
                ctx.kinematics.follow_cartesian_path(arm, [target], steps_per_waypoint=16)
                angle = ctx.world.joint_value("espresso_center", "select_dial")
                pressure = ctx.world.joint_value("espresso_center", "dial_press")
                if angle >= 1.18 and pressure >= 0.0018:
                    press_pose = (arm, home, target)
                    ctx.step(24)
                    brew_held = ctx.world.joint_value("espresso_center", "dial_press") >= 0.0015
                    if brew_held:
                        break
            _move_sampled(ctx, arm, tip(x, -0.015, z), steps=16)
        ctx.kinematics.move_joints(arm, home, steps=24)
        progress.append(round(ctx.world.joint_value("espresso_center", "select_dial"), 3))

    # Press axially at a pad position that actually compressed the shaft.
    angle = ctx.world.joint_value("espresso_center", "select_dial")
    if angle < 1.1345 or brew_held:
        print("Selector angles (rad)", progress, "brew hold", brew_held)
        return
    if press_pose is None:
        target = _tip_to_tcp((center[0], center[1] + 0.005, center[2] - 0.002),
                             left_approach, upright)
        press_pose = ("left_arm", left_home, target)
    arm, home, target = press_pose
    clear = Pose3D((target.xyz[0], target.xyz[1] - 0.016, target.xyz[2]), target.wxyz)
    _move_sampled(ctx, arm, clear, steps=24)
    _move_sampled(ctx, arm, target, steps=16)
    ctx.step(32)
    press = ctx.world.joint_value("espresso_center", "dial_press")
    if press < 0.0018:
        deeper = Pose3D((target.xyz[0], target.xyz[1] + 0.003, target.xyz[2]), target.wxyz)
        _move_sampled(ctx, arm, deeper, steps=16)
        ctx.step(24)
        press = ctx.world.joint_value("espresso_center", "dial_press")
    _move_sampled(ctx, arm, clear, steps=16)
    ctx.kinematics.move_joints(arm, home, steps=24)
    print("Selector angles (rad)", progress, "brew travel (m)", press)


def _spread(ctx, group, start, end, ticks):
    # Move the opposed pads gradually through first contact.
    for tick in range(1, ticks + 1):
        width = start + (end - start) * tick / ticks
        ctx.end_effector.command_targets(group, (width, width), steps=1)


def _dock_vessel(ctx, dock):
    socket = compose_pose(dock, Pose3D((-0.077, 0.012, 0.033)))
    sx, sy, sz = socket.xyz
    body = ctx.world.object_pose("kettle_vessel")
    axis = rotate_vec_by_quat_wxyz(body.wxyz, (1.0, 0.0, 0.0))
    yaw = quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), math.atan2(axis[1], axis[0]))
    approach = rotate_vec_by_quat_wxyz(yaw, (0.0, 0.916515, -0.4))
    closing = rotate_vec_by_quat_wxyz(yaw, (1.0, 0.0, 0.0))

    # Enter the handle loop from the front and pinch the stem radially.
    grasp = compose_pose(Pose3D(body.xyz, yaw), Pose3D((0.108, 0.005, 0.099))).xyz
    pre = tuple(v - 0.095 * a for v, a in zip(grasp, approach))

    def tip(xyz):
        return _tip_to_tcp(xyz, approach, closing)

    ctx.end_effector.command_targets("left_gripper", (0.024, 0.024), steps=24)
    ctx.kinematics.move_to_pose("left_arm", tip((pre[0], pre[1], pre[2] + 0.065)), steps=40)
    ctx.kinematics.move_to_pose("left_arm", tip(pre), steps=40)
    ctx.kinematics.move_to_pose("left_arm", tip(grasp), steps=40)
    ctx.step(16)
    _spread(ctx, "left_gripper", 0.024, 0.0, 40)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=32)
    seated = ctx.world.object_pose("kettle_vessel")
    for k in range(1, 9):
        u = k / 8.0
        height = 0.074 * u * u * (3.0 - 2.0 * u)
        body = ctx.world.object_pose("kettle_vessel")
        tcp = ctx.kinematics.ee_pose("left_arm")
        held = compose_pose(invert_pose(body), tcp)
        goal = Pose3D((seated.xyz[0], seated.xyz[1], seated.xyz[2] + height), yaw)
        target = compose_pose(goal, held)
        travel = math.dist(tcp.xyz, target.xyz)
        ticks = 8 * max(4, math.ceil(120 * travel / (0.075 * 8)))
        ctx.kinematics.follow_cartesian_path("left_arm", [target], steps_per_waypoint=ticks)
    ctx.step(16)

    body = ctx.world.object_pose("kettle_vessel")
    if body.xyz[2] < seated.xyz[2] + 0.035:
        return None
    start = ctx.kinematics.ee_pose("left_arm")
    # Park the handle toward the rear of the dock, clear of the wrist when
    # it turns the temperature dial in front of the kettle.
    dock_heading = math.radians(55.0)
    dock_yaw = quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), dock_heading)
    held = compose_pose(invert_pose(body), start)
    distance = math.hypot(sx - body.xyz[0], sy - body.xyz[1])
    # Accelerate over the first sixth of the path and decelerate over the
    # last sixth. The constant-speed middle stays below 0.10 m/s.
    ramp = 1.0 / 6.0
    segment_ticks = 8 * max(1, math.ceil(distance * 120 / ((1.0 - ramp) * 0.095 * 16 * 8)))
    route = []
    for k in range(1, 17):
        u = k / 16.0
        if u < ramp:
            f = u * u / (2.0 * ramp * (1.0 - ramp))
        elif u <= 1.0 - ramp:
            f = (u - 0.5 * ramp) / (1.0 - ramp)
        else:
            f = 1.0 - (1.0 - u) ** 2 / (2.0 * ramp * (1.0 - ramp))
        turn_pose = Pose3D((body.xyz[0] + f * (sx - body.xyz[0]),
                            body.xyz[1] + f * (sy - body.xyz[1]), body.xyz[2]),
                           slerp_wxyz(body.wxyz, dock_yaw, f))
        route.append(compose_pose(turn_pose, held))
    ctx.kinematics.follow_cartesian_path("left_arm", route, steps_per_waypoint=segment_ticks)
    ctx.step(16)

    # Align the heel and connector before transferring the weight to the dock.
    # Seat the heel against the connector on the withdrawal side. Its 44 mm
    # opening has lateral play around the 30 mm pin.
    for gap, ticks, lateral in ((0.027, 56, 0.0), (0.007, 40, -0.0035), (-0.001, 32, -0.009)):
        body = ctx.world.object_pose("kettle_vessel")
        tcp = ctx.kinematics.ee_pose("left_arm")
        axis = rotate_vec_by_quat_wxyz(body.wxyz, (1.0, 0.0, 0.0))
        flat = quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), math.atan2(axis[1], axis[0]))
        if math.hypot(sx - body.xyz[0], sy - body.xyz[1]) >= 0.070:
            return None
        held = compose_pose(invert_pose(body), tcp)
        heel = compose_pose(Pose3D((sx, sy, sz + gap), flat), Pose3D((0.0, lateral, 0.0)))
        goal = compose_pose(heel, held)
        _move_sampled(ctx, "left_arm", goal, steps=ticks)
        ctx.step(16 if gap < 0.0 else 8)

    # Open just past the two gripping surfaces and withdraw along the entry path.
    _spread(ctx, "left_gripper", 0.0, 0.024, 32)
    ctx.step(8)
    tcp = ctx.kinematics.ee_pose("left_arm")
    retreat = compose_pose(tcp, Pose3D((0.0, 0.0, -0.076)))
    ctx.kinematics.move_to_pose("left_arm", retreat, steps=40)
    return socket.xyz


def _turn_temperature(ctx, frame):
    # Press both pads against the top face on opposite sides of the shaft.
    # Turning about the shaft keeps the normal load on both pads.
    center = (frame[0], frame[1] + 0.017, frame[2] - 0.002)

    def pose(angle, height):
        rot = quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), angle)
        return _tip_to_tcp(
            (center[0], center[1], center[2] + height),
            rotate_vec_by_quat_wxyz(rot, (0.0, 0.12, -0.99277)),
            rotate_vec_by_quat_wxyz(rot, (1.0, 0.0, 0.0)),
        )

    start = -0.90
    end = start + 2.05
    arc = [pose(start + (end - start) * k / 16, 0.0) for k in range(1, 17)]
    ctx.end_effector.command_targets("left_gripper", (0.012, 0.012), steps=16)
    for _ in range(2):
        ctx.kinematics.move_to_pose("left_arm", pose(start, 0.072), steps=32)
        ctx.kinematics.move_to_pose("left_arm", pose(start, 0.008), steps=24)
        ctx.kinematics.move_to_pose("left_arm", pose(start, 0.0), steps=24)
        ctx.step(16)
        for index in range(0, len(arc), 2):
            ctx.kinematics.follow_cartesian_path("left_arm", arc[index:index + 2], steps_per_waypoint=16)
            if ctx.world.joint_value("kettle_dock", "temp_select_dial") >= 0.82:
                break
        finish = start + (end - start) * (index + 2) / len(arc)
        ctx.kinematics.move_to_pose("left_arm", pose(finish, 0.045), steps=16)
        if ctx.world.joint_value("kettle_dock", "temp_select_dial") >= 0.76:
            break


def _settle_lid(ctx, vessel):
    # Straddle the knob. A seated flange carries the load while a tilted
    # flange receives pressure on its high edge.
    x, y, z = vessel.xyz
    approach = (0.0, 0.20, -0.979796)
    closing = (1.0, 0.0, 0.0)
    _spread(ctx, "left_gripper", 0.024, 0.033, 8)
    for dx, dy, height, tilt in (
        (0.0, 0.0, 0.165, 0.0), (-0.002, 0.001, 0.150, 0.045),
        (0.002, -0.001, 0.147, -0.045), (0.0, 0.0, 0.148, 0.0),
    ):
        rotation = quat_from_axis_angle_wxyz((0.0, 1.0, 0.0), tilt)
        pose = _tip_to_tcp(
            (x + dx, y + dy, z + height),
            rotate_vec_by_quat_wxyz(rotation, approach), rotate_vec_by_quat_wxyz(rotation, closing),
        )
        _move_sampled(ctx, "left_arm", pose, steps=16)
    ctx.step(8)
    clear = Pose3D((pose.xyz[0], pose.xyz[1], pose.xyz[2] + 0.066), pose.wxyz)
    _move_sampled(ctx, "left_arm", clear, steps=24)


def _press_boil(ctx, frame, left_home, right_home):
    x, y, z = frame
    # First press across the key with the closed pads. An offset pad can
    # approach from the espresso side if the first grip slides past the key.
    for arm, group, home, offset, width, approach, closing in (
        ("left_arm", "left_gripper", left_home, -0.003, 0.0, (0.32, 0.08, -0.944), (0.0, 1.0, 0.0)),
        ("right_arm", "right_gripper", right_home, 0.025, 0.021, (0.0, 0.22, -0.9755), (1.0, 0.0, 0.0)),
    ):
        def key(height):
            return _tip_to_tcp((x + offset, y, z + height), approach, closing)

        ctx.end_effector.command_targets(group, (width, width), steps=16)
        ctx.kinematics.move_to_pose(arm, key(0.065), steps=40)
        ctx.kinematics.move_to_pose(arm, key(0.006), steps=32)
        for height in (0.002, 0.0, -0.002, -0.004, -0.007):
            ctx.kinematics.move_to_pose(arm, key(height), steps=24)
            ctx.step(8)
            travel = ctx.world.joint_value("kettle_dock", "boil_button")
            if travel <= -0.0019:
                ctx.step(24)
                travel = ctx.world.joint_value("kettle_dock", "boil_button")
                if travel <= -0.0015:
                    break
        print("Boil button travel (m)", arm, travel)
        ctx.kinematics.move_to_pose(arm, key(0.065), steps=32)
        ctx.kinematics.move_joints(arm, home, steps=32)
        if travel <= -0.0015:
            break


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=16)
    # The first motion sample follows a held target at a camera tick.
    ctx.step(17)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])
    x_cls = (1.0, 0.0, 0.0)
    top_dir = (0.0, 0.25, -0.9682458)

    dial_f = ctx.world.named_frame("kettle_dock", "temp_select_dial").xyz
    boil_f = ctx.world.named_frame("kettle_dock", "boil_button").xyz

    mx, my, mz = ctx.world.object_pose("espresso_center").xyz
    spout = ctx.world.named_frame("espresso_center", "cup_spout_target").xyz
    chute = ctx.world.named_frame("espresso_center", "capsule_chute").xyz

    sock_target = _dock_vessel(ctx, ctx.world.object_pose("kettle_dock"))

    if sock_target is None:
        return

    # Grip the lid knob and check that the basket clears the counter before carrying it.
    for knob_height in (0.0335, 0.0380):
        lid = ctx.world.object_pose("infuser_lid")
        knob_xyz = compose_pose(lid, Pose3D((0.0, 0.0, knob_height))).xyz
        kx, ky, kz = knob_xyz
        ctx.end_effector.command_targets("left_gripper", (0.023, 0.023), steps=16)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((kx, ky, kz + 0.115), top_dir, x_cls), steps=32)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((kx, ky, kz + 0.018), top_dir, x_cls), steps=32)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp(knob_xyz, top_dir, x_cls), steps=16)
        ctx.step(8)
        _spread(ctx, "left_gripper", 0.023, 0.0, 32)
        ctx.end_effector.set_posture("left_gripper", "closed", steps=24)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((kx, ky, kz + 0.060), top_dir, x_cls), steps=32)
        if ctx.world.object_pose("infuser_lid").xyz[2] >= lid.xyz[2] + 0.028:
            break

    vessel = ctx.world.object_pose("kettle_vessel")
    tx, ty, tz = (
        vessel.xyz
        if math.hypot(vessel.xyz[0] - sock_target[0], vessel.xyz[1] - sock_target[1]) < 0.045
        else sock_target
    )
    seat_tip_z = tz + 0.145 + knob_height
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((kx, ky, seat_tip_z + 0.095), top_dir, x_cls), steps=32)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((tx, ty, seat_tip_z + 0.075), top_dir, x_cls), steps=32)
    ctx.step(8)
    lid_air = ctx.world.object_pose("infuser_lid")
    lid_release = _tip_to_tcp((tx, ty, seat_tip_z), top_dir, x_cls)
    if lid_air.xyz[2] >= lid.xyz[2] + 0.028:
        lid_up = rotate_vec_by_quat_wxyz(vessel.wxyz, (0.0, 0.0, 1.0))
        # Center the square basket with its corners between adjacent staves.
        relative_lid = compose_pose(invert_pose(vessel), lid_air)
        axis = rotate_vec_by_quat_wxyz(relative_lid.wxyz, (1.0, 0.0, 0.0))
        pitch = 2.0 * math.pi / 32
        yaw = (math.floor(math.atan2(axis[1], axis[0]) / pitch) + 0.5) * pitch
        lid_y = rotate_vec_by_quat_wxyz(vessel.wxyz, (-math.sin(yaw), math.cos(yaw), 0.0))
        tcp = ctx.kinematics.ee_pose("left_arm")
        held_lid = compose_pose(invert_pose(lid_air), tcp)
        seat = Pose3D.from_approach(
            xyz=tuple(center + 0.145 * up for center, up in zip((tx, ty, tz), lid_up)),
            approach_dir=lid_up,
            closing_dir=lid_y,
        )
        guide = Pose3D(tuple(p + 0.030 * up for p, up in zip(seat.xyz, lid_up)), seat.wxyz)
        lid_release = compose_pose(seat, held_lid)
        try:
            ctx.kinematics.move_to_pose("left_arm", compose_pose(guide, held_lid), steps=24)
            ctx.kinematics.move_to_pose("left_arm", lid_release, steps=32)
            # Guide the basket below the rim before opening the jaws. Small
            # lateral motions accommodate a vessel that shifted on release.
            insertion = []
            for dx, dy, dz in (
                (-0.003, 0.003, -0.012), (0.003, 0.003, -0.016),
                (0.003, -0.003, -0.018), (-0.003, -0.003, -0.019),
                (0.0, 0.0, -0.020),
            ):
                lid_goal = compose_pose(seat, Pose3D((dx, dy, dz)))
                insertion.append(compose_pose(lid_goal, held_lid))
            ctx.kinematics.follow_cartesian_path("left_arm", insertion, steps_per_waypoint=16)
            lid_release = insertion[-1]
        except RuntimeError:
            lid_release = _tip_to_tcp((tx, ty, seat_tip_z), top_dir, x_cls)
            ctx.kinematics.move_to_pose("left_arm", lid_release, steps=32)
    else:
        ctx.kinematics.move_to_pose("left_arm", lid_release, steps=32)
    ctx.step(8)
    _spread(ctx, "left_gripper", 0.0, 0.024, 32)
    ctx.step(8)
    _settle_lid(ctx, ctx.world.object_pose("kettle_vessel"))
    ctx.end_effector.set_posture("left_gripper", "open", steps=16)

    ctx.kinematics.move_joints("left_arm", home_l, steps=24)
    _turn_temperature(ctx, dial_f)
    ctx.kinematics.move_joints("left_arm", home_l, steps=24)
    _press_boil(ctx, boil_f, home_l, home_r)

    # Stage 4/5. Pull the tray from the center and lift the cup clear of the steam wand.
    ctx.end_effector.set_posture("right_gripper", "closed", steps=16)
    tray_dir = (0.0, 0.7071068, -0.7071068)
    hook_xyz = (mx - 0.025, my - 0.174, mz + 0.038)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mx - 0.025, my - 0.265, mz + 0.155), tray_dir, x_cls), steps=32)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((hook_xyz[0], hook_xyz[1] - 0.055, hook_xyz[2] + 0.060), tray_dir, x_cls), steps=24)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((hook_xyz[0], hook_xyz[1], hook_xyz[2] + 0.022), tray_dir, x_cls), steps=16)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(hook_xyz, tray_dir, x_cls), steps=16)
    for k in range(1, 15):
        py = hook_xyz[1] - 0.064 * (k / 14.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((hook_xyz[0], py, hook_xyz[2]), tray_dir, x_cls), steps=8)
        if ctx.world.joint_value("espresso_center", "drip_tray") <= -0.041:
            break
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((hook_xyz[0], hook_xyz[1] - 0.085, mz + 0.165), tray_dir, x_cls), steps=24)

    ctx.end_effector.set_posture("right_gripper", "open", steps=24)
    cup_pose = ctx.world.object_pose("espresso_cup")
    cx, cy, cz = cup_pose.xyz
    ux = rotate_vec_by_quat_wxyz(cup_pose.wxyz, (1.0, 0.0, 0.0))
    uy = rotate_vec_by_quat_wxyz(cup_pose.wxyz, (0.0, 1.0, 0.0))
    cup_cls = max(
        [(ux[0], ux[1], 0.0), (-ux[0], -ux[1], 0.0), (uy[0], uy[1], 0.0), (-uy[0], -uy[1], 0.0)],
        key=lambda v: v[0],
    )
    cup_dir = (0.0, 0.55, -0.835)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx, cy, cz + 0.135), cup_dir, cup_cls), steps=32)
    ctx.step(8)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((cx, cy, cz + 0.028), cup_dir, cup_cls), steps=24)
    ctx.step(8)
    _spread(ctx, "right_gripper", 0.044, 0.0, 24)
    ctx.step(16)
    _move_sampled(ctx, "right_arm", _tip_to_tcp((cx, cy, cz + 0.090), cup_dir, cup_cls), steps=40)
    tray_val = ctx.world.joint_value("espresso_center", "drip_tray")
    target_cup_y = my - 0.146 + tray_val

    # Clear the wand on the outside before moving across the front of the tray.
    # Keep the cup in its pickup orientation through the turn and release.
    front_y = target_cup_y - 0.075
    for xyz, ticks in (
        ((cx, front_y, spout[2] + 0.078), 48),
        ((spout[0], front_y, spout[2] + 0.078), 56),
        ((spout[0], target_cup_y - 0.055, spout[2] + 0.068), 16),
        ((spout[0], target_cup_y, spout[2] + 0.058), 24),
        ((spout[0], target_cup_y, spout[2] + 0.025), 24),
    ):
        _move_sampled(ctx, "right_arm", _tip_to_tcp(xyz, cup_dir, cup_cls), steps=ticks)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "open", steps=24)
    ctx.step(8)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((spout[0], target_cup_y - 0.095, spout[2] + 0.062), cup_dir, x_cls), steps=24)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((spout[0] + 0.08, target_cup_y - 0.110, mz + 0.250), cup_dir, x_cls), steps=24)

    # Clear the machine side before approaching the capsule with the wrist leaning outward.
    ctx.kinematics.move_to_pose(
        "right_arm",
        _tip_to_tcp((mx + 0.270, target_cup_y - 0.110, mz + 0.260), cup_dir, x_cls),
        steps=32,
    )
    cap_x, cap_y, _ = ctx.world.object_pose("coffee_capsule").xyz
    cap_dir = (-0.56, 0.18, -0.809)
    if cap_x >= mx + 0.172 and cap_y >= my - 0.030:
        cap_dir = (-0.50, 0.28, -0.8195)
    cap_cls = (0.0, 1.0, 0.0)
    for pick_z in (0.016, 0.021):
        ctx.end_effector.command_targets("right_gripper", (0.024, 0.024), steps=24)
        kx, ky, kz = ctx.world.object_pose("coffee_capsule").xyz
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((kx, ky, kz + 0.145), cap_dir, cap_cls), steps=32
        )
        kx, ky, kz = ctx.world.object_pose("coffee_capsule").xyz
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((kx, ky, kz + pick_z), cap_dir, cap_cls), steps=32
        )
        ctx.step(8)
        _spread(ctx, "right_gripper", 0.024, 0.0, 24)
        ctx.step(8)
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((kx, ky, kz + 0.075), cap_dir, cap_cls), steps=32
        )
        if ctx.world.object_pose("coffee_capsule").xyz[2] >= kz + 0.028:
            break

    # Carry forward beside the machine before crossing in front of its lever.
    front_tip = (chute[0], chute[1] - 0.130, chute[2] + 0.110)
    side = (kx, my - 0.084, chute[2] + 0.059)
    corner = (mx + 0.120, chute[1] - 0.095, chute[2] + 0.070)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(side, cap_dir, cap_cls), steps=96)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(corner, cap_dir, cap_cls), steps=48)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(front_tip, cap_dir, cap_cls), steps=64)
    drop_dir = (cap_dir[1], -cap_dir[0], cap_dir[2])
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(front_tip, drop_dir, x_cls), steps=32)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((chute[0], chute[1] - 0.055, chute[2] + 0.075), drop_dir, x_cls), steps=32
    )
    ctx.step(8)
    cap_xyz = ctx.world.object_pose("coffee_capsule").xyz
    release_origin = (chute[0], chute[1], chute[2] + 0.046)
    release_tcp = _tip_to_tcp((chute[0], chute[1], chute[2] + 0.055), drop_dir, x_cls)
    if cap_xyz[2] > chute[2]:
        tcp = ctx.kinematics.ee_pose("right_arm")
        release_tcp = Pose3D(
            tuple(t + g - a for t, g, a in zip(tcp.xyz, release_origin, cap_xyz)),
            tcp.wxyz,
        )
    ctx.kinematics.move_to_pose("right_arm", release_tcp, steps=32)
    ctx.step(8)
    cap_xyz = ctx.world.object_pose("coffee_capsule").xyz
    dx, dy = release_origin[0] - cap_xyz[0], release_origin[1] - cap_xyz[1]
    if cap_xyz[2] > chute[2] + 0.032 and 0.003 < math.hypot(dx, dy) < 0.035:
        tcp = ctx.kinematics.ee_pose("right_arm")
        centered = Pose3D((tcp.xyz[0] + dx, tcp.xyz[1] + dy, tcp.xyz[2]), tcp.wxyz)
        ctx.kinematics.move_to_pose("right_arm", centered, steps=24)
        ctx.step(8)
    # Lower the capsule into the opening, then lift the narrow jaws clear of the chute.
    cap_xyz = ctx.world.object_pose("coffee_capsule").xyz
    tcp = ctx.kinematics.ee_pose("right_arm")
    seated_tcp = _tip_to_tcp((chute[0], chute[1], chute[2] + 0.039), drop_dir, x_cls)
    if cap_xyz[2] > chute[2] + 0.032:
        seated_tcp = Pose3D(
            (tcp.xyz[0], tcp.xyz[1], tcp.xyz[2] + chute[2] + 0.023 - cap_xyz[2]),
            tcp.wxyz,
        )
    ctx.kinematics.move_to_pose("right_arm", seated_tcp, steps=32)
    ctx.step(8)
    opening = min(0.023, max(ctx.sensors.proprio()[14:16]) + 0.0025)
    opening = max(0.020, opening)
    ctx.end_effector.command_targets("right_gripper", (opening, opening), steps=16)
    ctx.step(8)
    tcp = ctx.kinematics.ee_pose("right_arm")
    ctx.kinematics.move_to_pose(
        "right_arm", Pose3D((tcp.xyz[0], tcp.xyz[1], tcp.xyz[2] + 0.054), tcp.wxyz), steps=32
    )
    ctx.end_effector.set_posture("right_gripper", "open", steps=16)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((chute[0] + 0.06, chute[1] - 0.105, chute[2] + 0.105), drop_dir, x_cls), steps=32
    )

    ctx.end_effector.set_posture("right_gripper", "closed", steps=16)
    lever_dir = (0.0, 0.5735764, -0.8191520)
    lever_start = (mx, my - 0.068, mz + 0.399)
    lever_piv = (mx, my - 0.055, mz + 0.296)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mx, my - 0.050, mz + 0.435), lever_dir, x_cls), steps=24)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(lever_start, lever_dir, x_cls), steps=24)
    arc = _revolute_push_waypoints(lever_start, lever_piv, 1.28, lever_dir, x_cls, 12)
    ctx.kinematics.follow_cartesian_path("right_arm", arc, steps_per_waypoint=8)
    end_tcp = arc[-1]
    for k in range(1, 5):
        if k > 1 and ctx.world.joint_value("espresso_center", "brew_lever") >= -0.03:
            break
        ctx.kinematics.move_to_pose(
            "right_arm",
            Pose3D(xyz=(end_tcp.xyz[0], end_tcp.xyz[1] - 0.008 * k, end_tcp.xyz[2] - 0.010 * k), wxyz=end_tcp.wxyz),
            steps=16,
        )
    ctx.step(16)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mx, my - 0.24, mz + 0.315), lever_dir, x_cls), steps=16)

    ctx.kinematics.move_joints("right_arm", home_r, steps=24)

    _turn_and_brew(ctx, home_l, home_r)
    # Release the spring and park with both hands unloaded.
    if math.dist(ctx.sensors.proprio()[8:14], home_r) > 0.04:
        ctx.end_effector.command_targets("right_gripper", (0.026, 0.026), steps=16)
        tcp = ctx.kinematics.ee_pose("right_arm")
        if tcp.xyz[1] > my - 0.255:
            clear = Pose3D((tcp.xyz[0], tcp.xyz[1] - 0.020, tcp.xyz[2] + 0.008), tcp.wxyz)
            ctx.kinematics.move_to_pose("right_arm", clear, steps=24)
        ctx.kinematics.move_joints("right_arm", home_r, steps=32)
    ctx.end_effector.set_posture("left_gripper", "open", steps=16)
    ctx.end_effector.set_posture("right_gripper", "open", steps=16)
