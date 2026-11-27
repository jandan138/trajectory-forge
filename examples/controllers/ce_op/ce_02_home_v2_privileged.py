import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    compose_pose,
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


def _oven_door_handle_pose(
    ox: float, oy: float, oz: float, angle_rad: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    rot_x = -float(angle_rad)
    c, s = math.cos(rot_x), math.sin(rot_x)
    piv_y, piv_z = oy - 0.177, oz + 0.052
    dy0, dz0 = -0.042, 0.174
    hy = piv_y + dy0 * c - dz0 * s
    hz = piv_z + dy0 * s + dz0 * c
    app = (0.0, 0.55, -0.8351647)
    c_dir = (0.0, -1.0, 0.0)
    return (ox - 0.095, hy, hz), app, c_dir


def _ramp_both_jaws(ctx, start: float, end: float, frames: int) -> None:
    # Two alternating hand commands take two physics ticks. Four pairs fill a replay frame.
    pairs = 4 * frames
    for pair in range(1, pairs + 1):
        t = pair / pairs
        width = start + (end - start) * t * t * (3.0 - 2.0 * t)
        for gripper in ("left_gripper", "right_gripper"):
            ctx.end_effector.command_targets(gripper, (width, width), steps=1)


def _move_on_camera_frames(ctx, chain: str, target: Pose3D, *, steps: int) -> None:
    start = ctx.kinematics.ee_pose(chain)
    frames = steps // 8
    poses = []
    for frame in range(1, frames + 1):
        fraction = frame / frames
        xyz = tuple(p + fraction * (q - p) for p, q in zip(start.xyz, target.xyz))
        poses.append(Pose3D(xyz, slerp_wxyz(start.wxyz, target.wxyz, fraction)))
    try:
        ctx.kinematics.follow_cartesian_path(chain, poses, steps_per_waypoint=8)
    except RuntimeError:
        goal = ctx.kinematics.solve_ik(chain, target)
        ctx.kinematics.move_joints(chain, goal, steps=steps)


def _move_both_arms_one_frame(ctx, chains, starts, ends) -> None:
    for pair in range(4):
        for side, chain in enumerate(chains):
            fraction = (2 * pair + side + 1) / 8
            command = tuple(q + fraction * (end - q) for q, end in zip(starts[side], ends[side]))
            ctx.kinematics.move_joints(chain, command, steps=1)


def _load_cocotte(ctx, oven_xyz: tuple[float, float, float]) -> None:
    ox, oy, oz = oven_xyz
    dish = ctx.world.object_pose("baking_cocotte")
    # Keep the open right jaw and palm outside the left edge of the drop-down door.
    staging_x = ox - 0.385
    staging_y = oy - 0.312
    closing_x = (1.0, 0.0, 0.0)

    # Push the lower back wall along the worktop so the far side of the base stays supported.
    dx, dy = staging_x - dish.xyz[0], staging_y - dish.xyz[1]
    distance = math.hypot(dx, dy)
    if distance > 0.005:
        ux, uy = dx / distance, dy / distance
        axis_x = rotate_vec_by_quat_wxyz(dish.wxyz, (1.0, 0.0, 0.0))
        axis_y = rotate_vec_by_quat_wxyz(dish.wxyz, (0.0, 1.0, 0.0))
        radius = math.hypot(
            0.075 * (ux * axis_x[0] + uy * axis_x[1]),
            0.060 * (ux * axis_y[0] + uy * axis_y[1]),
        )
        x0 = dish.xyz[0] - (radius + 0.016) * ux
        y0 = dish.xyz[1] - (radius + 0.016) * uy
        height = dish.xyz[2] + 0.024
        down = (0.0, 0.20, -0.9798)
        def back_wall(offset: float, z: float) -> Pose3D:
            return _tip_to_tcp((x0 + ux * offset, y0 + uy * offset, z), down, closing_x)
        ctx.end_effector.set_posture("left_gripper", "closed", steps=8)
        ctx.kinematics.move_to_pose("left_arm", back_wall(-0.024, height + 0.090), steps=16)
        ctx.kinematics.move_to_pose("left_arm", back_wall(-0.024, height), steps=32)
        ctx.kinematics.move_to_pose("left_arm", back_wall(0.0, height), steps=40)
        stroke = 0.0
        increment = 0.0
        max_frames = math.ceil(15 * (distance + 0.045) / 0.093) + 12
        for _ in range(max_frames):
            center = ctx.world.object_pose("baking_cocotte").xyz
            remaining = (staging_x - center[0]) * ux + (staging_y - center[1]) * uy
            if remaining <= 0.0015:
                break
            increment = min(0.093 / 15, increment + 0.0008, math.sqrt(0.24 * remaining) / 15)
            stroke += increment
            ctx.kinematics.follow_cartesian_path(
                "left_arm", [back_wall(stroke, height)], steps_per_waypoint=8
            )
        ctx.step(16)
        ctx.kinematics.move_to_pose("left_arm", back_wall(stroke - 0.020, height), steps=24)
        ctx.kinematics.move_to_pose("left_arm", back_wall(stroke - 0.025, height + 0.080), steps=24)

    dish = ctx.world.object_pose("baking_cocotte")
    angle = math.radians(18.0)
    wall_axis = (math.cos(angle), math.sin(angle), 0.0)
    tangent = (-wall_axis[1], wall_axis[0], 0.0)
    approach = (0.60 * tangent[0], 0.60 * tangent[1], -0.80)
    # The normal of an ellipse locates two opposed staves even after a yaw change.
    local_x = rotate_vec_by_quat_wxyz(dish.wxyz, (1.0, 0.0, 0.0))
    local_y = rotate_vec_by_quat_wxyz(dish.wxyz, (0.0, 1.0, 0.0))
    nx = sum(v * a for v, a in zip(wall_axis, local_x))
    ny = sum(v * a for v, a in zip(wall_axis, local_y))
    rx, ry = 0.068, 0.053
    denominator = math.hypot(rx * nx, ry * ny)
    rim_x = rx * rx * nx / denominator
    rim_y = ry * ry * ny / denominator
    grasp_specs = (
        ("left_arm", "left_gripper", -1.0),
        ("right_arm", "right_gripper", 1.0),
    )
    # Descend vertically astride each stave. Keep both jaws open until both wrists are seated.
    pads = []
    for chain, gripper, side in grasp_specs:
        wall = compose_pose(dish, Pose3D((side * rim_x, side * rim_y, 0.050))).xyz
        tip = tuple(v - 0.006 * t for v, t in zip(wall, tangent))
        grasp = _tip_to_tcp(tip, approach, wall_axis)
        x, y, z = grasp.xyz
        ctx.end_effector.command_targets(gripper, (0.040, 0.040), steps=8)
        if chain == "right_arm":
            # Clear the open door on the way across, with the wrist still close to the shoulders.
            high = Pose3D((x + 0.055, y - 0.055, oz + 0.255), grasp.wxyz)
            ctx.kinematics.move_to_pose(chain, high, steps=24)
        ctx.kinematics.move_to_pose(chain, Pose3D((x, y, z + 0.065), grasp.wxyz), steps=16)
        pads.append((chain, grasp))

    chains = tuple(chain for chain, _ in pads)
    joints = []
    for chain, grasp in pads:
        x, y, z = grasp.xyz
        joints.append(ctx.kinematics.solve_ik(chain, Pose3D((x, y, z + 0.065), grasp.wxyz)))
    for upper, lower, frames in ((0.065, 0.020, 2), (0.020, 0.0, 5)):
        for k in range(1, frames + 1):
            offset = upper + (lower - upper) * k / frames
            ends = []
            for chain, grasp in pads:
                x, y, z = grasp.xyz
                ends.append(ctx.kinematics.solve_ik(chain, Pose3D((x, y, z + offset), grasp.wxyz)))
            _move_both_arms_one_frame(ctx, chains, joints, ends)
            joints = ends

    ctx.step(8)
    _ramp_both_jaws(ctx, 0.040, 0.0, frames=8)
    ctx.step(16)
    previous = ctx.sensors.proprio()
    for _ in range(3):
        ctx.step(8)
        current = ctx.sensors.proprio()
        if max(abs(current[j] - previous[j]) for j in (6, 7, 14, 15)) < 0.0004:
            break
        previous = current
    grips = []
    for chain, _, side in grasp_specs:
        tcp = ctx.kinematics.ee_pose(chain)
        inward = -0.002 * side
        x, y, z = tcp.xyz
        brace = Pose3D((x + inward * wall_axis[0], y + inward * wall_axis[1], z), tcp.wxyz)
        ctx.kinematics.move_to_pose(chain, brace, steps=8)
        grips.append((chain, brace))
    ctx.step(8)
    # Lift, cross in front of the oven, advance over the wires, then lower onto them.
    for i, speed in ((-1, 0.028), (0, 0.078), (1, 0.076), (2, 0.065), (3, 0.032)):
        prior = ctx.world.object_pose("baking_cocotte").xyz
        front_rail_y = oy - 0.1425 + ctx.world.joint_value("steam_oven", "rack_slide")
        if i < 0:
            # Unload the base vertically before translating the suspended vessel.
            target = (prior[0], prior[1], prior[2] + 0.020)
        elif i == 0:
            # Rise along the clear left side. Reach rack height before entering its footprint.
            target = (min(ox - 0.278, prior[0] + 0.105), prior[1], oz + 0.153)
        elif i == 1:
            target = (ox, min(oy - 0.255, prior[1] + 0.055), prior[2])
        else:
            seat_y = min(front_rail_y + 0.045, oy - 0.198)
            target = (ox, seat_y, oz + (0.149 if i == 2 else 0.1348))
        displacement = tuple(b - a for a, b in zip(prior, target))
        # Level the suspended vessel about its measured center while translating both wrists.
        up = rotate_vec_by_quat_wxyz(ctx.world.object_pose("baking_cocotte").wxyz, (0.0, 0.0, 1.0))
        horizontal = math.hypot(up[0], up[1])
        correction = min(0.20, math.atan2(horizontal, up[2])) if i > 0 else 0.0
        if horizontal > 0.0001:
            tilt_axis = (up[1] / horizontal, -up[0] / horizontal)
        else:
            tilt_axis = (0.0, 0.0)
        offsets = [
            (chain, Pose3D(tuple(x - c for x, c in zip(tcp.xyz, prior)), tcp.wxyz))
            for chain, tcp in grips
        ]
        # Interpolate both arms every pair of ticks and solve their endpoints once per replay frame.
        frames = max(6, math.ceil(15 * math.dist(prior, target) / speed) + 3)
        joints = [ctx.kinematics.solve_ik(chain, tcp) for chain, tcp in grips]
        for k in range(1, frames + 1):
            if k <= 3:
                part = k * k / (6 * (frames - 3))
            elif k > frames - 3:
                part = 1.0 - (frames - k) ** 2 / (6 * (frames - 3))
            else:
                part = (k - 1.5) / (frames - 3)
            half = 0.5 * correction * part
            sine = math.sin(half)
            rotation = (math.cos(half), tilt_axis[0] * sine, tilt_axis[1] * sine, 0.0)
            center = Pose3D(tuple(c + d * part for c, d in zip(prior, displacement)), rotation)
            targets = [(chain, compose_pose(center, offset)) for chain, offset in offsets]
            ends = [ctx.kinematics.solve_ik(chain, pose) for chain, pose in targets]
            _move_both_arms_one_frame(ctx, chains, joints, ends)
            joints = ends
        grips = targets
        ctx.step(8)
        dish_at_waypoint = ctx.world.object_pose("baking_cocotte").xyz
        dish_x, dish_y, dish_z = dish_at_waypoint
        if (
            i >= 2
            and abs(dish_x - ox) <= 0.050
            and dish_y >= front_rail_y + 0.025
            and oz + 0.130 <= dish_z <= oz + 0.1355
        ):
            # The wires are supporting the dish. Release it before making any further corrections.
            break
        min_height = target[2] - 0.035
        if dish_at_waypoint[2] < min_height:
            raise RuntimeError(
                f"Cocotte slipped during carry leg {i + 1}. "
                f"Expected height {target[2]:.3f} m and measured {dish_at_waypoint[2]:.3f} m. "
                f"Jaw positions: {ctx.sensors.proprio()[6:8]} and {ctx.sensors.proprio()[14:16]}."
            )

    # Brace the handle while the second hand leaves the rim. The free rack can otherwise
    # roll inward while a pad is still inside the vessel.
    hook_dir = (0.0, 0.16, -0.9871)
    # The narrow center handle sits below and ahead of the full-width front rail.
    # Brace the rail outside the dish footprint.
    rail_y = oy - 0.1425
    hook_x, hook_z = ox - 0.1425, oz + 0.126

    def release_wall(chain: str, gripper: str, first: int) -> None:
        jaw = ctx.sensors.proprio()[first:first + 2]
        clear = min(0.044, max(jaw) + 0.013)
        frames = 6 if chain == "right_arm" else 2
        for k in range(1, frames + 1):
            fraction = k / frames
            targets = tuple(q + fraction * (clear - q) for q in jaw)
            ctx.end_effector.command_targets(gripper, targets, steps=8)
        tcp = ctx.kinematics.ee_pose(chain)
        x, y, z = tcp.xyz
        ctx.kinematics.move_to_pose(chain, Pose3D((x, y, z + 0.050), tcp.wxyz), steps=32)

    release_wall("left_arm", "left_gripper", 6)
    ctx.end_effector.command_targets("left_gripper", (0.018, 0.018), steps=8)

    def hook(y: float, z: float) -> Pose3D:
        return _tip_to_tcp((hook_x, y, z), hook_dir, (0.0, 1.0, 0.0))

    for attempt in range(3):
        if attempt:
            ctx.end_effector.command_targets("left_gripper", (0.025, 0.025), steps=16)
            held_tcp = ctx.kinematics.ee_pose("left_arm")
            hx, hy, hz = held_tcp.xyz
            ctx.kinematics.move_to_pose("left_arm", Pose3D((hx, hy, hz + 0.045), held_tcp.wxyz), steps=24)
            ctx.end_effector.command_targets("left_gripper", (0.018, 0.018), steps=8)
        extension = ctx.world.joint_value("steam_oven", "rack_slide")
        hook_y = rail_y + extension
        ctx.kinematics.move_to_pose("left_arm", hook(hook_y, hook_z + 0.055), steps=24)
        hook_y = rail_y + ctx.world.joint_value("steam_oven", "rack_slide")
        ctx.kinematics.move_to_pose("left_arm", hook(hook_y, hook_z + 0.018), steps=16)
        ctx.kinematics.move_to_pose("left_arm", hook(hook_y, hook_z), steps=24)
        for width in (0.011, 0.005, 0.0):
            ctx.end_effector.command_targets("left_gripper", (width, width), steps=8)
        # Pinch the rail between two wires so the shelf cannot run out from under the dish.
        hold_y = min(hook_y, rail_y - 0.112)
        extend_frames = max(1, math.ceil(15 * (hook_y - hold_y) / 0.085))
        ctx.kinematics.move_to_pose("left_arm", hook(hold_y, hook_z), steps=8 * extend_frames)
        ctx.step(8)
        front = rail_y + ctx.world.joint_value("steam_oven", "rack_slide")
        center = ctx.world.object_pose("baking_cocotte").xyz
        if center[1] >= front + 0.030 and center[2] >= oz + 0.127:
            break
    else:
        raise RuntimeError("Rack front did not stay under the cocotte before release.")

    release_wall("right_arm", "right_gripper", 14)
    ctx.step(16)
    dish = ctx.world.object_pose("baking_cocotte")
    up = rotate_vec_by_quat_wxyz(dish.wxyz, (0.0, 0.0, 1.0))
    if dish.xyz[2] < oz + 0.120 or up[2] < math.cos(math.radians(15)):
        raise RuntimeError(f"Cocotte is not level on the supported shelf at {dish.xyz}.")

    # With the shelf held, push low on the front wall until the whole base is on the wires.
    front_edge = rail_y + ctx.world.joint_value("steam_oven", "rack_slide")
    seat_y = max(front_edge + 0.076, oy - 0.185)
    if dish.xyz[1] < seat_y:
        x, y, z = dish.xyz
        front = y - 0.077
        height = z + 0.025
        nudge_dir = (0.0, 0.75, -0.6614)
        def nudge(y_pos: float, z_pos: float) -> Pose3D:
            return _tip_to_tcp((x, y_pos, z_pos), nudge_dir, closing_x)
        ctx.end_effector.set_posture("right_gripper", "closed", steps=8)
        ctx.kinematics.move_to_pose("right_arm", nudge(front - 0.025, height + 0.050), steps=24)
        ctx.kinematics.move_to_pose("right_arm", nudge(front, height), steps=24)
        for k in range(1, 30):
            ctx.kinematics.move_to_pose("right_arm", nudge(front + 0.003 * k, height), steps=8)
            if ctx.world.object_pose("baking_cocotte").xyz[1] >= seat_y:
                break
        stop_y = front + 0.003 * k
        ctx.step(8)
        ctx.kinematics.move_to_pose("right_arm", nudge(stop_y - 0.018, height), steps=32)
        ctx.kinematics.move_to_pose("right_arm", nudge(stop_y - 0.035, height + 0.050), steps=24)
    # Keep the rail stationary while withdrawing this hand, then return the rack in one stroke.
    ctx.step(1)
    # Release the stationary rail vertically before pushing on the front of the pull rod.
    jaw = ctx.sensors.proprio()[6:8]
    for part in (0.25, 0.5, 0.75, 1.0):
        targets = tuple(q + part * (0.025 - q) for q in jaw)
        ctx.end_effector.command_targets("left_gripper", targets, steps=8)
    tcp = ctx.kinematics.ee_pose("left_arm")
    x, y, z = tcp.xyz
    ctx.kinematics.move_to_pose("left_arm", Pose3D((x, y, z + 0.024), tcp.wxyz), steps=32)
    ctx.kinematics.move_to_pose("left_arm", Pose3D((x, y, z + 0.060), tcp.wxyz), steps=16)



def _turn_dial(ctx, center: tuple[float, float, float]) -> None:
    x, y, z = center
    approach = (0.0, 0.12, -0.9928)
    contact_z = z - 0.008
    start = -1.20

    def rim_pose(angle: float, radius: float, height: float) -> Pose3D:
        dx = max(-0.030, min(0.030, radius * math.cos(angle)))
        tip = (x + dx, y + radius * math.sin(angle), height)
        tangent = (-math.sin(angle), math.cos(angle), 0.0)
        return _tip_to_tcp(tip, approach, tangent)

    ctx.end_effector.set_posture("right_gripper", "closed", steps=8)
    ctx.kinematics.move_to_pose("right_arm", rim_pose(start, 0.035, z + 0.034), steps=16)
    # Align the ends of the selector waypoints with the pre-step camera samples.
    ctx.step(1)
    for stroke in range(18):
        ctx.kinematics.move_to_pose("right_arm", rim_pose(start, 0.035, contact_z), steps=8)
        ctx.kinematics.move_to_pose("right_arm", rim_pose(start, 0.0180, contact_z), steps=8)
        ctx.step(8)
        previous = ctx.world.joint_value("steam_oven", "temp_mode_dial")
        for k in range(1, 12):
            angle = start + 1.25 * k / 8
            ticks = 16 if stroke < 3 and k <= 4 else 8
            ctx.kinematics.move_to_pose("right_arm", rim_pose(angle, 0.0180, contact_z), steps=ticks)
            position = ctx.world.joint_value("steam_oven", "temp_mode_dial")
            advance = position - previous
            if position >= 1.67 or (k >= 8 and advance < 0.007):
                break
            previous = position
        # Lift above the rim before crossing back. Keep the robot clear of both keys.
        tcp = ctx.kinematics.ee_pose("right_arm")
        px, py, pz = tcp.xyz
        ctx.kinematics.move_to_pose(
            "right_arm", Pose3D((px, py, max(pz + 0.032, z + 0.040)), tcp.wxyz), steps=8
        )
        if ctx.world.joint_value("steam_oven", "temp_mode_dial") >= 1.64:
            break
        ctx.kinematics.move_to_pose("right_arm", rim_pose(start, 0.035, z + 0.026), steps=8)
    # Resume the jaw-command cadence after the pads are clear.
    ctx.step(7)
    if ctx.world.joint_value("steam_oven", "temp_mode_dial") < 1.58:
        raise RuntimeError("Selector did not hold the Air Fry setting after the pads cleared.")


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])
    x_cls = (1.0, 0.0, 0.0)
    btn_dir = (0.0, 0.25, -0.9682458)

    ox, oy, oz = ctx.world.object_pose("steam_oven").xyz
    wake_f = ctx.world.named_frame("steam_oven", "wake_press").xyz
    start_f = ctx.world.named_frame("steam_oven", "start_press").xyz
    dial_f = ctx.world.named_frame("steam_oven", "temp_mode_dial").xyz
    vent_f = ctx.world.named_frame("steam_oven", "steam_vent_latch").xyz
    rack_f = ctx.world.named_frame("steam_oven", "rack_pull").xyz

    ctx.end_effector.set_posture("right_gripper", "closed", steps=8)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((wake_f[0], wake_f[1] - 0.03, wake_f[2] + 0.090), btn_dir, x_cls), steps=16)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((wake_f[0], wake_f[1], wake_f[2] + 0.025), btn_dir, x_cls), steps=16)
    for k in range(1, 13):
        pz = wake_f[2] + 0.003 - 0.0014 * k
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((wake_f[0], wake_f[1], pz), btn_dir, x_cls), steps=8)
        if ctx.world.joint_value("steam_oven", "wake_button") <= -0.0022:
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((wake_f[0], wake_f[1], pz - 0.0008), btn_dir, x_cls), steps=8)
            break
    ctx.step(16)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((wake_f[0], wake_f[1], wake_f[2] + 0.095), btn_dir, x_cls), steps=8)
    ctx.kinematics.move_joints("right_arm", home_r, steps=8)

    ctx.end_effector.set_posture("left_gripper", "half_open", steps=8)
    h0, a0, c0 = _oven_door_handle_pose(ox, oy, oz, 0.0)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((h0[0], h0[1] - 0.035, h0[2] + 0.065), a0, c0), steps=8)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((h0[0], h0[1] + 0.005, h0[2] - 0.010), a0, c0), steps=16)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    for k in range(1, 16):
        ang = -1.46 * (k / 15.0)
        dh, ap, cp = _oven_door_handle_pose(ox, oy, oz, ang)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((dh[0], dh[1] + 0.005, dh[2] - 0.010), ap, cp), steps=8)
        if ctx.world.joint_value("steam_oven", "door_hinge") <= -1.38:
            break
    ctx.step(16)
    ctx.end_effector.set_posture("left_gripper", "open", steps=16)
    dh_now, ap_now, cp_now = _oven_door_handle_pose(ox, oy, oz, ctx.world.joint_value("steam_oven", "door_hinge"))
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((dh_now[0], dh_now[1] - 0.055, dh_now[2] + 0.085), ap_now, cp_now), steps=16)

    hook_dir = (0.0, 0.60, -0.8000)
    ctx.kinematics.move_joints("left_arm", home_l, steps=16)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    rx0, ry0, rz0 = rack_f[0], rack_f[1] + 0.0085, rack_f[2] - 0.006
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((rx0, ry0 - 0.060, rack_f[2] + 0.090), hook_dir, x_cls), steps=16)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((rx0, ry0, rack_f[2] + 0.042), hook_dir, x_cls), steps=16)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((rx0, ry0, rz0), hook_dir, x_cls), steps=16)
    ctx.step(8)
    py = ry0
    for k in range(1, 15):
        py = ry0 - 0.118 * (k / 14.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((rx0, py, rz0), hook_dir, x_cls), steps=8)
        if ctx.world.joint_value("steam_oven", "rack_slide") <= -0.108:
            break
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((rx0, py, rack_f[2] + 0.055), hook_dir, x_cls), steps=16)

    # Seat the door flat before the wrist descends. The inner glass face is accessible
    # ahead of the extended rack throughout the last part of the opening arc.
    pivot_z = oz + 0.052
    ahead = 0.148
    door_y = oy - 0.177 - ahead
    def door_top(angle: float) -> float:
        angle = max(0.75, min(1.4835, -angle))
        return pivot_z + (0.008 + ahead * math.cos(angle)) / math.sin(angle)
    door_z = door_top(ctx.world.joint_value("steam_oven", "door_hinge"))
    flat_z = door_top(-1.4835) - 0.002
    down = (0.0, 0.16, -0.9871)
    def door_pad(z: float) -> Pose3D:
        return _tip_to_tcp((ox - 0.105, door_y, z), down, x_cls)
    ctx.kinematics.move_to_pose("left_arm", door_pad(door_z + 0.030), steps=16)
    open_frames = max(2, math.ceil(15 * abs(door_z + 0.030 - flat_z) / 0.165))
    ctx.kinematics.move_to_pose("left_arm", door_pad(flat_z), steps=8 * open_frames)
    ctx.step(16)
    ctx.kinematics.move_to_pose("left_arm", door_pad(flat_z + 0.065), steps=16)

    ctx.kinematics.move_joints("left_arm", home_l, steps=16)

    _load_cocotte(ctx, (ox, oy, oz))
    ctx.kinematics.move_joints("right_arm", home_r, steps=16)

    # Hold the loaded rack at the rear stop until the door starts to close.
    ctx.end_effector.set_posture("left_gripper", "closed", steps=8)
    rq_push = ctx.world.joint_value("steam_oven", "rack_slide")
    rack_dir = (0.0, 0.7071, -0.7071)
    push_y = rack_f[1] + rq_push - 0.026
    seated_y = rack_f[1] + 0.003
    push_z = rack_f[2] - 0.006
    def pusher(y: float, z: float, dx: float = 0.0) -> Pose3D:
        return _tip_to_tcp((ox + dx, y, z), rack_dir, x_cls)
    _move_on_camera_frames(ctx, "left_arm", pusher(push_y - 0.025, push_z + 0.050), steps=24)
    _move_on_camera_frames(ctx, "left_arm", pusher(push_y, push_z), steps=16)
    travel = seated_y - push_y
    frames = max(4, math.ceil(15 * travel / 0.082) + 2)
    for k in range(1, frames + 1):
        if k < 2:
            part = k * k / (4 * (frames - 2))
        elif k > frames - 2:
            part = 1 - (frames - k) ** 2 / (4 * (frames - 2))
        else:
            part = (k - 1) / (frames - 2)
        ctx.kinematics.follow_cartesian_path(
            "left_arm", [pusher(push_y + part * travel, push_z)], steps_per_waypoint=8
        )
    for _ in range(5):
        if ctx.world.joint_value("steam_oven", "rack_slide") >= -0.003:
            break
        seated_y += 0.001
        _move_on_camera_frames(ctx, "left_arm", pusher(seated_y, push_z), steps=8)
    ctx.step(16)

    pivot_y, pivot_z = oy - 0.177, oz + 0.052
    def handle(angle: float, backoff: float, pressure: float = 0.0) -> Pose3D:
        cosine, sine = math.cos(-angle), math.sin(-angle)
        approach = (0.0, sine, -cosine)
        closing = (0.0, cosine, sine)
        y = pivot_y + (-0.042 + pressure) * cosine - 0.174 * sine
        z = pivot_z + (-0.042 + pressure) * sine + 0.174 * cosine
        tip = (ox + 0.075, y + (0.0115 - backoff) * sine, z - (0.0115 - backoff) * cosine)
        return _tip_to_tcp(tip, approach, closing)

    def glass(angle: float, pressure: float) -> Pose3D:
        cosine, sine = math.cos(-angle), math.sin(-angle)
        y = pivot_y + (-0.008 + pressure) * cosine - 0.122 * sine
        z = pivot_z + (-0.008 + pressure) * sine + 0.122 * cosine
        return _tip_to_tcp((ox - 0.178, y, z), (0.0, 0.50, -0.8660254), (0.0, 1.0, 0.0))

    # The open handle overhangs the worktop. Approach its rod from the front.
    angle = ctx.world.joint_value("steam_oven", "door_hinge")
    ctx.end_effector.command_targets("right_gripper", (0.0155, 0.0155), steps=8)
    outside = handle(angle, 0.055)
    x, y, z = outside.xyz
    high = Pose3D((x, y - 0.020, z + 0.075), outside.wxyz)
    try:
        high_joints = ctx.kinematics.solve_ik("right_arm", high)
    except RuntimeError:
        ctx.kinematics.follow_cartesian_path("right_arm", [outside], steps_per_waypoint=40)
    else:
        ctx.kinematics.move_joints("right_arm", high_joints, steps=16)
        _move_on_camera_frames(ctx, "right_arm", outside, steps=16)
    ctx.kinematics.follow_cartesian_path("right_arm", [handle(angle, 0.0)], steps_per_waypoint=32)
    for width in (0.013, 0.009, 0.004, 0.0):
        ctx.end_effector.command_targets("right_gripper", (width, width), steps=8)
    ctx.step(8)

    # Rotate the wrist with the handle, keeping the rod between the pads all the way to the stop.
    open_angle = ctx.world.joint_value("steam_oven", "door_hinge")
    radius = math.hypot(0.042, 0.174)
    frames = max(6, math.ceil(15 * radius * abs(open_angle) / 0.105) + 3)
    rack_held = True
    for k in range(1, frames + 1):
        if k <= 3:
            part = k * k / (6 * (frames - 3))
        elif k > frames - 3:
            part = 1.0 - (frames - k) ** 2 / (6 * (frames - 3))
        else:
            part = (k - 1.5) / (frames - 3)
        angle = open_angle * (1.0 - part)
        _move_on_camera_frames(ctx, "right_arm", handle(angle, 0.0, 0.002 * part), steps=8)
        if rack_held and (angle >= -0.91 or ctx.world.joint_value("steam_oven", "door_hinge") >= -0.92):
            # Release along the rod and clear the entire door swing before it resumes.
            _move_on_camera_frames(ctx, "left_arm", pusher(seated_y, push_z, -0.087), steps=32)
            _move_on_camera_frames(ctx, "left_arm", pusher(seated_y, push_z + 0.077, -0.095), steps=16)
            _move_on_camera_frames(ctx, "left_arm", _tip_to_tcp((ox - 0.095, oy - 0.310, oz + 0.310), btn_dir, x_cls), steps=24)
            rack_held = False
    ctx.step(16)
    ctx.step(7)

    vent_dir = (0.0, 0.75, -0.6614)
    ctx.end_effector.command_targets("left_gripper", (0.016, 0.016), steps=16)
    vx0, vy0, vz0 = vent_f[0], vent_f[1], vent_f[2] - 0.005
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((vx0, vy0 - 0.050 * vent_dir[1], vz0 - 0.050 * vent_dir[2]), vent_dir, x_cls), steps=16)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((vx0, vy0, vz0), vent_dir, x_cls), steps=16)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    vx = vx0
    for k in range(1, 9):
        vx = vx0 + 0.030 * (k / 8.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((vx, vy0, vz0), vent_dir, x_cls), steps=8)
        if ctx.world.joint_value("steam_oven", "steam_vent") >= 0.022:
            break
    ctx.end_effector.command_targets("left_gripper", (0.018, 0.018), steps=8)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((vx, vy0 - 0.060 * vent_dir[1], vz0 - 0.060 * vent_dir[2]), vent_dir, x_cls), steps=16)
    # Transfer the closed door to the left hand before turning the selector.
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ox - 0.095, oy - 0.310, oz + 0.310), btn_dir, x_cls), steps=16)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=8)
    ctx.kinematics.move_to_pose("left_arm", glass(-0.25, -0.055), steps=8)
    door_joints = ctx.kinematics.move_to_pose("left_arm", glass(0.0, 0.003), steps=16)
    ctx.step(8)
    jaw = ctx.sensors.proprio()[14:16]
    for fraction in (0.5, 1.0):
        targets = tuple(q + fraction * (0.027 - q) for q in jaw)
        ctx.end_effector.command_targets("right_gripper", targets, steps=8)
    angle = ctx.world.joint_value("steam_oven", "door_hinge")
    ctx.kinematics.move_to_pose("right_arm", handle(angle, 0.040), steps=16)
    ctx.kinematics.move_joints("right_arm", home_r, steps=8)
    _turn_dial(ctx, dial_f)
    if ctx.world.joint_value("steam_oven", "door_hinge") < -0.045:
        door_joints = ctx.kinematics.move_to_pose("left_arm", glass(0.0, 0.007), steps=24)
        ctx.step(8)

    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((start_f[0], start_f[1], start_f[2] + 0.022), btn_dir, x_cls), steps=16
    )
    for k in range(4):
        pz = start_f[2] - 0.0030 - 0.0014 * k
        button_joints = ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((start_f[0], start_f[1], pz), btn_dir, x_cls), steps=16)
        ctx.step(8)
        if ctx.world.joint_value("steam_oven", "start_bake") <= -0.0022:
            break
    # Keep START depressed for two camera samples, then let the switch spring back.
    ctx.step(8)
    chains = ("left_arm", "right_arm")
    cleared = (
        ctx.kinematics.solve_ik("left_arm", glass(0.0, -0.027)),
        ctx.kinematics.solve_ik("right_arm", _tip_to_tcp((start_f[0], start_f[1], start_f[2] + 0.025), btn_dir, x_cls)),
    )
    _move_both_arms_one_frame(ctx, chains, (door_joints, button_joints), cleared)
    parked_joints = (home_l, home_r)
    middle = [tuple((a + b) * 0.5 for a, b in zip(q, h)) for q, h in zip(cleared, parked_joints)]
    _move_both_arms_one_frame(ctx, chains, cleared, middle)
    _move_both_arms_one_frame(ctx, chains, middle, parked_joints)
    _move_both_arms_one_frame(ctx, chains, parked_joints, parked_joints)
    # Both wrists are clear while replay checks the final guarded props.
    ctx.step(16)
