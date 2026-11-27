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


def _lid_arc_waypoints(
    x_world: float,
    pivot_world: tuple[float, float, float],
    y_local: float,
    z_local: float,
    start_deg: float,
    end_deg: float,
    approach_dir: tuple[float, float, float],
    closing_dir: tuple[float, float, float],
    n: int,
) -> list[Pose3D]:
    _, py, pz = pivot_world
    wps: list[Pose3D] = []
    for i in range(n):
        th = math.radians(start_deg + (end_deg - start_deg) * (i / max(1, n - 1)))
        c, s = math.cos(th), math.sin(th)
        wy = py + y_local * c - z_local * s
        wz = pz + y_local * s + z_local * c
        wps.append(_tip_to_tcp((x_world, wy, wz), approach_dir, closing_dir))
    return wps


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])
    x_cls = (1.0, 0.0, 0.0)

    bx, by, bz = ctx.world.object_pose("brother_printer").xyz
    platen = ctx.world.named_frame("brother_printer", "platen_center").xyz

    # Stage 0: Push paper_cassette +Y completely closed
    ctx.end_effector.set_posture("left_gripper", "closed", steps=10)
    cas_dir = (0.0, 0.5000000, -0.8660254)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((bx, by - 0.420, bz + 0.075), cas_dir, x_cls), steps=20)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((bx, by - 0.375, bz + 0.035), cas_dir, x_cls), steps=16)
    for i in range(1, 15):
        py = by - 0.375 + 0.185 * (i / 14.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((bx, py, bz + 0.035), cas_dir, x_cls), steps=6)
        if ctx.world.joint_value("brother_printer", "paper_cassette") >= -0.004:
            break
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((bx, by - 0.420, bz + 0.075), cas_dir, x_cls), steps=16)

    # Stage 1: Open left_gripper & pinch 44mm-wide HandleCol along local +-X, place flat on platen_center
    ctx.end_effector.set_posture("left_gripper", "open", steps=12)
    sh_pose = ctx.world.object_pose("calibration_sheet")
    off_bar = rotate_vec_by_quat_wxyz(sh_pose.wxyz, (0.0, -0.054, 0.011))
    bar_w = (sh_pose.xyz[0] + off_bar[0], sh_pose.xyz[1] + off_bar[1], sh_pose.xyz[2] + off_bar[2])
    sx = rotate_vec_by_quat_wxyz(sh_pose.wxyz, (1.0, 0.0, 0.0))
    bar_x_cls = (float(sx[0]), float(sx[1]), 0.0)
    pick_dir = (0.0, 0.22, -0.9755)
    place_dir = (0.0, 0.45, -0.8930)

    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((bar_w[0], bar_w[1], bar_w[2] + 0.110), pick_dir, bar_x_cls), steps=24
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((bar_w[0], bar_w[1], bar_w[2] + 0.001), pick_dir, bar_x_cls), steps=22
    )
    ctx.end_effector.set_posture("left_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((bar_w[0], bar_w[1], platen[2] + 0.095), pick_dir, bar_x_cls), steps=26
    )

    place_bar = (platen[0], platen[1] - 0.054, platen[2] + 0.014)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((place_bar[0] - 0.10, place_bar[1] - 0.06, place_bar[2] + 0.085), place_dir, x_cls), steps=22
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((place_bar[0], place_bar[1], place_bar[2] + 0.075), place_dir, x_cls), steps=20
    )
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp(place_bar, place_dir, x_cls), steps=20)
    ctx.end_effector.set_posture("left_gripper", "open", steps=14)
    ctx.step(8)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((place_bar[0] - 0.08, place_bar[1] - 0.08, place_bar[2] + 0.080), place_dir, x_cls), steps=18
    )
    ctx.kinematics.move_joints("left_arm", home_l, steps=20)

    # Stage 2: Pass cleanly over ColLidShell top edge with -25 deg +Y pitch, hook behind & sweep -Y/-Z closed
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    lid_piv = (bx, by + 0.184, bz + 0.282)
    lip_dir = (0.0, 0.9063078, -0.4226183)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx + 0.150, by - 0.020, bz + 0.572), lip_dir, x_cls), steps=24)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx + 0.150, by + 0.088, bz + 0.566), lip_dir, x_cls), steps=20)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx + 0.150, by + 0.088, bz + 0.534), lip_dir, x_cls), steps=14)
    arc_wps = _lid_arc_waypoints(bx + 0.150, lid_piv, -0.275, 0.020, -58.0, 2.0, lip_dir, x_cls, 14)
    ctx.kinematics.follow_cartesian_path("right_arm", arc_wps, steps_per_waypoint=5)
    down_dir = (0.0, 0.5000000, -0.8660254)
    for k in range(1, 6):
        if ctx.world.joint_value("brother_printer", "scanner_lid") >= -0.04:
            break
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((bx + 0.150, by - 0.080, bz + 0.312 - 0.006 * k), down_dir, x_cls), steps=7
        )
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx + 0.175, by - 0.155, bz + 0.355), down_dir, x_cls), steps=16)

    # Stage 3: Adaptive press & HOLD button_start inside [-0.0025, -0.0011]
    btn_dir = (0.0, 0.3420201, -0.9396926)
    sx_b, sy_b, sz_b = bx + 0.176, by - 0.174, bz + 0.2722
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((sx_b, sy_b, sz_b + 0.060), btn_dir, x_cls), steps=18)
    pz = sz_b
    for _ in range(22):
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((sx_b, sy_b, pz), btn_dir, x_cls), steps=6)
        bval = ctx.world.joint_value("brother_printer", "button_start")
        if -0.0023 <= bval <= -0.0014:
            break
        pz += -0.00025 if bval > -0.0014 else 0.00020
    ctx.step(20)
    for _ in range(8):
        bval = ctx.world.joint_value("brother_printer", "button_start")
        if -0.0024 <= bval <= -0.0012:
            break
        pz += 0.00020 if bval < -0.0024 else -0.00020
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((sx_b, sy_b, pz), btn_dir, x_cls), steps=6)
    _ = home_r
