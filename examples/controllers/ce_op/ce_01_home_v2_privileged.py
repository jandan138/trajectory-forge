import math

from trajectory_forge.sdk.math3d import Pose3D, rotate_vec_by_quat_wxyz


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


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])
    x_cls = (1.0, 0.0, 0.0)
    top_dir = (0.0, 0.25, -0.9682458)

    mx, my, mz = ctx.world.object_pose("espresso_center").xyz
    spout = ctx.world.named_frame("espresso_center", "cup_spout_target").xyz
    chute = ctx.world.named_frame("espresso_center", "capsule_chute").xyz

    # Stage 0a: Press top-right ButtonDrink (power/wake) and hold >= 18 steps past debounce
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    bx, by, bz = mx + 0.0625, my - 0.091, mz + 0.3055
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx, by - 0.04, bz + 0.090), top_dir, x_cls), steps=36)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx, by, bz + 0.020), top_dir, x_cls), steps=20)
    ctx.step(6)
    pz = bz + 0.004
    for k in range(1, 20):
        pz = bz + 0.004 - 0.0007 * k
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx, by, pz), top_dir, x_cls), steps=5)
        ctx.step(3)
        if k >= 16 and ctx.world.joint_value("espresso_center", "button_drink") <= -0.0024:
            ctx.kinematics.move_to_pose(
                "right_arm",
                _tip_to_tcp((bx, by, min(pz - 0.0012, bz - 0.0086)), top_dir, x_cls),
                steps=6,
            )
            break
    ctx.step(28)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx, by, bz + 0.095), top_dir, x_cls), steps=16)
    ctx.kinematics.move_joints("right_arm", home_r, steps=18)

    # Stage 0b: 45-deg forward basin hook behind DripTray rim & pull -Y cleanly
    ctx.end_effector.set_posture("left_gripper", "closed", steps=10)
    tray_dir = (0.0, 0.7071068, -0.7071068)
    hook_xyz = (mx - 0.030, my - 0.174, mz + 0.038)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((hook_xyz[0], hook_xyz[1] - 0.055, hook_xyz[2] + 0.060), tray_dir, x_cls), steps=20
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((hook_xyz[0], hook_xyz[1], hook_xyz[2] + 0.022), tray_dir, x_cls), steps=14
    )
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp(hook_xyz, tray_dir, x_cls), steps=12)
    for k in range(1, 15):
        py = hook_xyz[1] - 0.058 * (k / 14.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((hook_xyz[0], py, hook_xyz[2]), tray_dir, x_cls), steps=5)
        if ctx.world.joint_value("espresso_center", "drip_tray") <= -0.039:
            break
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((hook_xyz[0], hook_xyz[1] - 0.085, hook_xyz[2] + 0.085), tray_dir, x_cls), steps=16
    )

    # Stage 1: Pick espresso_cup dynamically over near clutter & place onto extended drip_tray spout target
    ctx.end_effector.set_posture("left_gripper", "open", steps=10)
    cup_pose = ctx.world.object_pose("espresso_cup")
    cx, cy, cz = cup_pose.xyz
    ux = rotate_vec_by_quat_wxyz(cup_pose.wxyz, (1.0, 0.0, 0.0))
    uy = rotate_vec_by_quat_wxyz(cup_pose.wxyz, (0.0, 1.0, 0.0))
    cup_cls = max(
        [(ux[0], ux[1], 0.0), (-ux[0], -ux[1], 0.0), (uy[0], uy[1], 0.0), (-uy[0], -uy[1], 0.0)],
        key=lambda v: v[0],
    )
    cup_dir = (0.0, 0.55, -0.835)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((cx, cy, cz + 0.135), cup_dir, cup_cls), steps=22)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((cx, cy, cz + 0.028), cup_dir, cup_cls), steps=18)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((cx, cy, cz + 0.145), cup_dir, cup_cls), steps=22)
    tray_val = ctx.world.joint_value("espresso_center", "drip_tray")
    target_cup_y = my - 0.150 + tray_val
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((spout[0] - 0.06, target_cup_y - 0.075, spout[2] + 0.078), cup_dir, x_cls), steps=20
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((spout[0], target_cup_y - 0.055, spout[2] + 0.068), cup_dir, x_cls), steps=18
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((spout[0], target_cup_y, spout[2] + 0.058), cup_dir, x_cls), steps=18
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((spout[0], target_cup_y, spout[2] + 0.025), cup_dir, x_cls), steps=18
    )
    ctx.end_effector.set_posture("left_gripper", "open", steps=16)
    ctx.step(6)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((spout[0], target_cup_y - 0.095, spout[2] + 0.062), cup_dir, x_cls), steps=16
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((spout[0] - 0.08, target_cup_y - 0.110, spout[2] + 0.115), cup_dir, x_cls), steps=16
    )
    ctx.kinematics.move_joints("left_arm", home_l, steps=20)

    # Stage 2: Pick coffee_capsule dynamically with right_arm high above clutter & drop into top chute
    ctx.end_effector.set_posture("right_gripper", "open", steps=10)
    kx, ky, kz = ctx.world.object_pose("coffee_capsule").xyz
    cap_dir = (0.0, 0.45, -0.893)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((kx, ky, kz + 0.145), cap_dir, x_cls), steps=22)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((kx, ky, kz + 0.013), cap_dir, x_cls), steps=18)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((kx, ky, chute[2] + 0.130), cap_dir, x_cls), steps=22)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((chute[0], chute[1] - 0.11, chute[2] + 0.105), cap_dir, x_cls), steps=20
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((chute[0], chute[1], chute[2] + 0.078), cap_dir, x_cls), steps=18
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((chute[0], chute[1], chute[2] + 0.044), cap_dir, x_cls), steps=16
    )
    ctx.end_effector.set_posture("right_gripper", "open", steps=16)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((chute[0] + 0.06, chute[1] - 0.14, chute[2] + 0.105), cap_dir, x_cls), steps=16
    )

    # Stage 3: Close brew_lever with circular arc around pivot
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    lever_dir = (0.0, 0.5735764, -0.8191520)
    lever_start = (mx, my - 0.068, mz + 0.399)
    lever_piv = (mx, my - 0.055, mz + 0.296)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mx, my - 0.050, mz + 0.440), lever_dir, x_cls), steps=20)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(lever_start, lever_dir, x_cls), steps=16)
    arc = _revolute_push_waypoints(lever_start, lever_piv, 1.28, lever_dir, x_cls, 12)
    ctx.kinematics.follow_cartesian_path("right_arm", arc, steps_per_waypoint=4)
    end_tcp = arc[-1]
    for k in range(1, 5):
        if ctx.world.joint_value("espresso_center", "brew_lever") >= -0.03:
            break
        ctx.kinematics.move_to_pose(
            "right_arm",
            Pose3D(xyz=(end_tcp.xyz[0], end_tcp.xyz[1] - 0.008 * k, end_tcp.xyz[2] - 0.010 * k), wxyz=end_tcp.wxyz),
            steps=8,
        )
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mx, my - 0.24, mz + 0.315), lever_dir, x_cls), steps=16)

    # Stage 4: Handshake tooth-ratchet sweep on SelectDialRotor to Espresso (>= 1.25 rad)
    dz = mz + 0.228
    d_dir = (0.0, 0.9396926, -0.3420201)
    b_cls = (0.0, 0.3420201, 0.9396926)
    y_in = my - 0.1885
    y_out = my - 0.2155
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    start_p = _tip_to_tcp((mx - 0.027, y_out, dz + 0.0150), d_dir, b_cls)
    ctx.kinematics.move_to_pose("right_arm", start_p, steps=20)
    ctx.step(4)
    for st in range(14):
        if ctx.world.joint_value("espresso_center", "select_dial") >= 1.32:
            break
        x_down = 0.0165 if (st % 2 == 0) else 0.0145
        wps = [
            _tip_to_tcp((mx - 0.027, y_in, dz + 0.0150), d_dir, b_cls),
            _tip_to_tcp((mx - 0.004, y_in, dz + 0.0150), d_dir, b_cls),
            _tip_to_tcp((mx + x_down, y_in, dz + 0.0140), d_dir, b_cls),
            _tip_to_tcp((mx + x_down, y_in, dz - 0.0020), d_dir, b_cls),
            _tip_to_tcp((mx + x_down - 0.002, y_in, dz - 0.0180), d_dir, b_cls),
        ]
        ctx.kinematics.follow_cartesian_path("right_arm", wps, steps_per_waypoint=8)
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((mx + x_down - 0.002, y_out, dz - 0.0180), d_dir, b_cls), steps=12
        )
        ctx.step(4)
        ctx.kinematics.move_to_pose("right_arm", start_p, steps=12)

    # Stage 5: Center press & HOLD dial_press >= 64 steps through end so dual espresso jets fill cup + crema to 100%
    dy = my - 0.1915
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mx, dy - 0.030, dz - 0.002), d_dir, b_cls), steps=14)
    for i in range(1, 32):
        ty = dy - 0.012 + 0.0008 * i
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mx, ty, dz - 0.002), d_dir, b_cls), steps=5)
        ctx.step(3)
        pv = ctx.world.joint_value("espresso_center", "dial_press")
        if pv >= 0.0022:
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mx, ty + 0.0008, dz - 0.002), d_dir, b_cls), steps=6)
            break
    ctx.step(64)
