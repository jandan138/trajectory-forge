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
    dy0 = start_tip_xyz[1] - pivot_world[1]
    dz0 = start_tip_xyz[2] - pivot_world[2]
    waypoints: list[Pose3D] = []
    for idx in range(num_waypoints):
        ang = delta_angle_x_rad * (idx / max(1, num_waypoints - 1))
        c = math.cos(ang)
        s = math.sin(ang)
        tip_xyz = (
            start_tip_xyz[0],
            pivot_world[1] + dy0 * c - dz0 * s,
            pivot_world[2] + dy0 * s + dz0 * c,
        )
        waypoints.append(_tip_to_tcp(tip_xyz, approach_dir, closing_dir))
    return waypoints


def _move_base_to(ctx, target_x: float, target_y: float, steps: int = 35) -> None:
    bx, by, _ = ctx.kinematics.base_pose_world().xyz
    ctx.kinematics.move_base_world(target_x - bx, target_y - by, 0.0, steps=steps)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=10)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])

    # ------------------------------------------------------------------
    # Stage 0: a5_media_pack grasp attempt (keeps pack on table for clean AABB diagnostic)
    # ------------------------------------------------------------------
    _move_base_to(ctx, -0.04, -0.58, steps=35)
    ctx.end_effector.set_posture("left_gripper", "open", steps=12)
    pack_pose = ctx.world.object_pose("a5_media_pack")
    board_dir = (0.0, 0.0871557, -0.9961947)
    place_cls = (0.0, -0.9961947, -0.0871557)
    pack_pre = _tip_to_tcp((pack_pose.xyz[0], pack_pose.xyz[1], 0.880), board_dir, (1.0, 0.0, 0.0))
    pack_grasp = _tip_to_tcp((pack_pose.xyz[0], pack_pose.xyz[1], 0.772), board_dir, (1.0, 0.0, 0.0))
    pack_lift = _tip_to_tcp((pack_pose.xyz[0], pack_pose.xyz[1], 0.940), board_dir, (1.0, 0.0, 0.0))
    cass_place = _tip_to_tcp((-0.020, -0.175, 0.880), board_dir, place_cls)
    pack_retract = _tip_to_tcp((-0.180, -0.220, 0.960), board_dir, (1.0, 0.0, 0.0))
    ctx.kinematics.move_to_pose("left_arm", pack_pre, steps=28)
    ctx.kinematics.move_to_pose("left_arm", pack_grasp, steps=20)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose("left_arm", pack_lift, steps=28)
    ctx.kinematics.move_to_pose("left_arm", cass_place, steps=36)
    ctx.end_effector.set_posture("left_gripper", "open", steps=14)
    ctx.kinematics.move_to_pose("left_arm", pack_retract, steps=22)
    ctx.kinematics.move_joints("left_arm", home_l, steps=24)
    print(
        "AFTER_S0",
        [round(v, 4) for v in ctx.world.object_pose("a5_media_pack").xyz],
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage0_media")

    # ------------------------------------------------------------------
    # Stage 1: Push paper_cassette +Y from -0.165 into [-0.012, 0.0]
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.25, -0.76, steps=30)
    ctx.kinematics.set_torso_lift(0.36, steps=16)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=10)
    cas_dir = (0.0, 0.7071068, -0.7071068)
    cas_cls = (1.0, 0.0, 0.0)
    s1_pre = _tip_to_tcp((0.0, -0.325, 0.830), cas_dir, cas_cls)
    s1_start = _tip_to_tcp((0.0, -0.285, 0.790), cas_dir, cas_cls)
    ctx.kinematics.move_to_pose("left_arm", s1_pre, steps=24)
    ctx.kinematics.move_to_pose("left_arm", s1_start, steps=18)
    for i in range(1, 15):
        py = -0.285 + 0.185 * (i / 14.0)
        ctx.kinematics.move_to_pose(
            "left_arm",
            _tip_to_tcp((0.0, py, 0.790), cas_dir, cas_cls),
            steps=8,
        )
        if ctx.world.joint_value("brother_printer", "paper_cassette") >= -0.004:
            break
    ctx.kinematics.move_to_pose("left_arm", s1_pre, steps=20)
    ctx.kinematics.set_torso_lift(0.46, steps=16)
    ctx.kinematics.move_joints("left_arm", home_l, steps=24)
    print(
        "AFTER_S1",
        round(ctx.world.joint_value("brother_printer", "paper_cassette"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage1_cassette")

    # ------------------------------------------------------------------
    # Stage 2: calibration_sheet grasp attempt (keeps sheet on table)
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.04, -0.58, steps=30)
    ctx.end_effector.set_posture("right_gripper", "open", steps=12)
    sh_pose = ctx.world.object_pose("calibration_sheet")
    sh_pre = _tip_to_tcp((sh_pose.xyz[0], sh_pose.xyz[1], 0.880), board_dir, (1.0, 0.0, 0.0))
    sh_grasp = _tip_to_tcp((sh_pose.xyz[0], sh_pose.xyz[1], 0.768), board_dir, (1.0, 0.0, 0.0))
    sh_lift = _tip_to_tcp((sh_pose.xyz[0], sh_pose.xyz[1], 1.090), board_dir, (1.0, 0.0, 0.0))
    ctx.kinematics.move_to_pose("right_arm", sh_pre, steps=28)
    ctx.kinematics.move_to_pose("right_arm", sh_grasp, steps=20)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose("right_arm", sh_lift, steps=28)
    _move_base_to(ctx, -0.22, -0.58, steps=26)
    ctx.kinematics.move_to_pose(
        "right_arm",
        _tip_to_tcp((0.020, 0.030, 1.090), board_dir, (1.0, 0.0, 0.0)),
        steps=28,
    )
    ctx.end_effector.set_posture("right_gripper", "open", steps=14)
    ctx.kinematics.move_to_pose(
        "right_arm",
        _tip_to_tcp((0.020, -0.200, 1.090), board_dir, (1.0, 0.0, 0.0)),
        steps=22,
    )
    print(
        "AFTER_S2",
        [round(v, 4) for v in ctx.world.object_pose("calibration_sheet").xyz],
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage2_sheet")

    # ------------------------------------------------------------------
    # Stage 3: Close scanner_lid (pivot [0, 0.274, 1.037], axis +X, -1.0821 -> 0.0 rad)
    # ------------------------------------------------------------------
    _move_base_to(ctx, -0.24, -0.58, steps=16)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    lid_dir = (0.0, 0.7071068, -0.7071068)
    lid_close = (1.0, 0.0, 0.0)
    lid_pre = _tip_to_tcp((0.0, -0.050, 1.250), lid_dir, lid_close)
    lid_start_tip = (0.0, 0.135, 1.240)
    lid_start = _tip_to_tcp(lid_start_tip, lid_dir, lid_close)
    ctx.kinematics.move_to_pose("right_arm", lid_pre, steps=24)
    ctx.kinematics.move_to_pose("right_arm", lid_start, steps=20)
    lid_arc = _revolute_push_waypoints(
        start_tip_xyz=lid_start_tip,
        pivot_world=(0.0, 0.274, 1.037),
        delta_angle_x_rad=0.92,
        approach_dir=lid_dir,
        closing_dir=lid_close,
        num_waypoints=12,
    )
    ctx.kinematics.follow_cartesian_path("right_arm", lid_arc, steps_per_waypoint=4)
    lid_retract = _tip_to_tcp((0.0, -0.180, 1.150), lid_dir, lid_close)
    ctx.kinematics.move_to_pose("right_arm", lid_retract, steps=22)
    print(
        "AFTER_S3",
        round(ctx.world.joint_value("brother_printer", "scanner_lid"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage3_lid")

    # ------------------------------------------------------------------
    # Stage 4: Adaptive press & HOLD button_start inside [-0.0025, -0.0011]
    # ------------------------------------------------------------------
    _move_base_to(ctx, -0.07, -0.58, steps=22)
    btn_dir = (0.0, 0.3420201, -0.9396926)
    btn_cls = (1.0, 0.0, 0.0)
    ctx.kinematics.move_to_pose(
        "right_arm",
        _tip_to_tcp((0.176, -0.084, 1.080), btn_dir, btn_cls),
        steps=22,
    )
    pz = 1.0272
    for _ in range(22):
        ctx.kinematics.move_to_pose(
            "right_arm",
            _tip_to_tcp((0.176, -0.084, pz), btn_dir, btn_cls),
            steps=6,
        )
        bval = ctx.world.joint_value("brother_printer", "button_start")
        if -0.0023 <= bval <= -0.0014:
            break
        if bval > -0.0014:
            pz -= 0.00025
        elif bval < -0.0023:
            pz += 0.00020
    ctx.step(20)
    for _ in range(8):
        bval = ctx.world.joint_value("brother_printer", "button_start")
        if -0.0024 <= bval <= -0.0012:
            break
        pz += 0.00020 if bval < -0.0024 else -0.00020
        ctx.kinematics.move_to_pose(
            "right_arm",
            _tip_to_tcp((0.176, -0.084, pz), btn_dir, btn_cls),
            steps=6,
        )
    print(
        "AFTER_S4",
        round(ctx.world.joint_value("brother_printer", "button_start"), 5),
        ctx.world.stage_scores(),
    )
    _ = home_r

