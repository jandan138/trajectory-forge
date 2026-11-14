import math

from trajectory_forge.sdk.math3d import Pose3D


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


def _revolute_push_waypoints(
    start_tip_xyz: tuple[float, float, float],
    pivot_world: tuple[float, float, float],
    delta_angle_x_rad: float,
    approach_dir: tuple[float, float, float],
    closing_dir: tuple[float, float, float],
    num_waypoints: int,
) -> list[Pose3D]:
    px, py, pz = pivot_world
    ry = start_tip_xyz[1] - py
    rz = start_tip_xyz[2] - pz
    wps: list[Pose3D] = []
    for i in range(1, num_waypoints + 1):
        ang = delta_angle_x_rad * (i / float(num_waypoints))
        c, s = math.cos(ang), math.sin(ang)
        tip = (start_tip_xyz[0], py + ry * c - rz * s, pz + ry * s + rz * c)
        wps.append(_tip_to_tcp(tip, approach_dir, closing_dir))
    return wps


def _move_base_to(ctx, target_x: float, target_y: float, steps: int = 32) -> None:
    bx, by, _ = ctx.kinematics.base_pose_world().xyz
    ctx.kinematics.move_base_world(target_x - bx, target_y - by, 0.0, steps=steps)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=10)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])

    # Stage 0: Pinch cantilevered a5_media_pack, slide off horizontally, center over y=-0.175,
    # and drop above ColWallFront.
    _move_base_to(ctx, -0.11, -1.12, steps=34)
    ctx.kinematics.set_torso_lift(0.22, steps=16)
    ctx.end_effector.command_targets("left_gripper", (0.028, 0.028), steps=14)
    px0 = ctx.world.object_pose("a5_media_pack").xyz[0]
    pk_dir = (0.0, 0.9848078, -0.1736482)
    pk_cls = (0.0, 0.1736482, 0.9848078)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((px0, -0.490, 0.775), pk_dir, pk_cls), steps=26
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((px0, -0.405, 0.765), pk_dir, pk_cls), steps=22
    )
    ctx.end_effector.set_posture("left_gripper", "closed", steps=22)
    for k in range(1, 6):
        ctx.kinematics.move_to_pose(
            "left_arm",
            _tip_to_tcp((px0, -0.405 - 0.014 * k, 0.766 + 0.002 * k), pk_dir, pk_cls),
            steps=6,
        )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((px0, -0.475, 0.910), pk_dir, pk_cls), steps=24
    )

    ctx.kinematics.set_torso_lift(0.26, steps=14)
    _move_base_to(ctx, 0.265, -0.92, steps=32)
    tx_cmd, ty_cmd = 0.0, -0.242
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((tx_cmd, ty_cmd, 0.875), pk_dir, pk_cls), steps=26
    )
    for _ in range(2):
        p_xyz = ctx.world.object_pose("a5_media_pack").xyz
        tx_cmd = max(-0.05, min(0.05, tx_cmd + (0.0 - p_xyz[0])))
        ty_cmd = max(-0.258, min(-0.205, ty_cmd + (-0.175 - p_xyz[1])))
        ctx.kinematics.move_to_pose(
            "left_arm", _tip_to_tcp((tx_cmd, ty_cmd, 0.858), pk_dir, pk_cls), steps=14
        )
    print(
        "S0_pre_drop",
        [round(v, 4) for v in ctx.world.object_pose("a5_media_pack").xyz],
        (round(tx_cmd, 4), round(ty_cmd, 4)),
    )
    ctx.end_effector.command_targets("left_gripper", (0.028, 0.028), steps=14)
    ctx.step(16)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((tx_cmd, ty_cmd - 0.080, 0.895), pk_dir, pk_cls), steps=18
    )
    ctx.kinematics.move_joints("left_arm", home_l, steps=20)
    print(
        "AFTER_S0",
        [round(v, 4) for v in ctx.world.object_pose("a5_media_pack").xyz],
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage0_media")

    # Stage 1: Push paper_cassette +Y from -0.165 into [-0.012, 0.0]
    _move_base_to(ctx, 0.25, -0.76, steps=28)
    ctx.kinematics.set_torso_lift(0.36, steps=16)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=10)
    cas_dir = (0.0, 0.7071068, -0.7071068)
    cas_cls = (1.0, 0.0, 0.0)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((0.0, -0.325, 0.830), cas_dir, cas_cls), steps=22
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((0.0, -0.285, 0.790), cas_dir, cas_cls), steps=16
    )
    for i in range(1, 15):
        py = -0.285 + 0.185 * (i / 14.0)
        ctx.kinematics.move_to_pose(
            "left_arm", _tip_to_tcp((0.0, py, 0.790), cas_dir, cas_cls), steps=7
        )
        if ctx.world.joint_value("brother_printer", "paper_cassette") >= -0.004:
            break
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((0.0, -0.325, 0.830), cas_dir, cas_cls), steps=18
    )
    ctx.kinematics.set_torso_lift(0.46, steps=14)
    ctx.kinematics.move_joints("left_arm", home_l, steps=20)
    print(
        "AFTER_S1",
        round(ctx.world.joint_value("brother_printer", "paper_cassette"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage1_cassette")

    # Stage 2: Pinch cantilevered calibration_sheet & place on platen
    _move_base_to(ctx, 0.13, -1.12, steps=34)
    ctx.kinematics.set_torso_lift(0.22, steps=16)
    ctx.end_effector.command_targets("right_gripper", (0.016, 0.016), steps=12)
    sh_pose = ctx.world.object_pose("calibration_sheet")
    sh_dir = (0.0, 0.9848078, -0.1736482)
    sh_cls = (0.0, 0.1736482, 0.9848078)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((sh_pose.xyz[0], -0.490, 0.775), sh_dir, sh_cls), steps=24
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((sh_pose.xyz[0], -0.415, 0.758), sh_dir, sh_cls), steps=20
    )
    ctx.end_effector.set_posture("right_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((sh_pose.xyz[0], -0.465, 0.850), sh_dir, sh_cls), steps=20
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((sh_pose.xyz[0], -0.465, 0.940), sh_dir, sh_cls), steps=20
    )
    ctx.kinematics.set_torso_lift(0.46, steps=18)
    _move_base_to(ctx, -0.25, -0.68, steps=30)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.0, 0.032, 1.120), sh_dir, sh_cls), steps=26
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.0, 0.032, 1.062), sh_dir, sh_cls), steps=20
    )
    ctx.end_effector.command_targets("right_gripper", (0.016, 0.016), steps=12)
    ctx.step(10)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.0, 0.032, 1.135), sh_dir, sh_cls), steps=16
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.120, -0.090, 1.160), sh_dir, sh_cls), steps=18
    )
    print(
        "AFTER_S2",
        [round(v, 4) for v in ctx.world.object_pose("calibration_sheet").xyz],
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage2_sheet")

    # Stage 3: Two-step scanner_lid closure (right-side hook + top-shell press to 0.0 rad)
    _move_base_to(ctx, 0.05, -0.55, steps=22)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    piv = (0.0, 0.274, 1.037)
    lid_dir = (-0.50, 0.80, -0.3316625)
    lid_cls = (0.0, 0.3834825, 0.9235481)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.250, -0.020, 1.185), lid_dir, lid_cls), steps=20
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.250, 0.235, 1.185), lid_dir, lid_cls), steps=22
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.172, 0.235, 1.185), lid_dir, lid_cls), steps=16
    )
    arc_wps = _revolute_push_waypoints((0.172, 0.235, 1.185), piv, 1.10, lid_dir, lid_cls, 14)
    ctx.kinematics.follow_cartesian_path("right_arm", arc_wps, steps_per_waypoint=5)
    end_y = piv[1] + (0.235 - piv[1]) * math.cos(1.10) - (1.185 - piv[2]) * math.sin(1.10)
    end_z = piv[2] + (0.235 - piv[1]) * math.sin(1.10) + (1.185 - piv[2]) * math.cos(1.10)
    last_y, last_z = end_y, end_z
    for k in range(1, 7):
        if ctx.world.joint_value("brother_printer", "scanner_lid") >= -0.48:
            break
        last_y = end_y - 0.006 * k
        last_z = end_z - 0.008 * k
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((0.172, last_y, last_z), lid_dir, lid_cls), steps=8
        )
    print("S3_after_hook", round(ctx.world.joint_value("brother_printer", "scanner_lid"), 5))
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.270, last_y, last_z), lid_dir, lid_cls), steps=14
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.270, -0.010, 1.210), lid_dir, lid_cls), steps=16
    )
    _move_base_to(ctx, -0.04, -0.52, steps=18)
    lip_dir = (0.0, 0.50, -0.8660254)
    btn_dir = (0.0, 0.3420201, -0.9396926)
    btn_cls = (1.0, 0.0, 0.0)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.245, 0.024, 1.220), lip_dir, btn_cls), steps=18
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.150, 0.024, 1.220), lip_dir, btn_cls), steps=14
    )
    ty_lid, tz_lid = 0.024, 1.212
    for k in range(19):
        if ctx.world.joint_value("brother_printer", "scanner_lid") >= -0.035:
            break
        ty_lid = 0.024 - 0.046 * (k / 18.0)
        tz_lid = 1.212 - 0.198 * (k / 18.0)
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((0.150, ty_lid, tz_lid), lip_dir, btn_cls), steps=7
        )
    print("S3_after_press", round(ctx.world.joint_value("brother_printer", "scanner_lid"), 5))
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.150, -0.065, tz_lid + 0.020), lip_dir, btn_cls), steps=10
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.150, -0.065, 1.120), lip_dir, btn_cls), steps=12
    )
    _move_base_to(ctx, -0.07, -0.58, steps=14)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.176, -0.084, 1.085), btn_dir, btn_cls), steps=16
    )
    print(
        "AFTER_S3",
        round(ctx.world.joint_value("brother_printer", "scanner_lid"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage3_lid")

    # Stage 4: Adaptive press & HOLD button_start inside [-0.0025, -0.0011]
    pz = 1.0272
    for _ in range(22):
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((0.176, -0.084, pz), btn_dir, btn_cls), steps=6
        )
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
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((0.176, -0.084, pz), btn_dir, btn_cls), steps=6
        )
    print(
        "AFTER_S4",
        round(ctx.world.joint_value("brother_printer", "button_start"), 5),
        ctx.world.stage_scores(),
        ctx.world.current_score(),
    )
    _ = home_r
