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
    # R5a link7/link8 pinch pads extend 0.0226 m beyond the 0.135 m TCP frame along approach_dir.
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
    # Stage 0: Pull drip_tray by pinching 10mm RimF at x=-0.045 (clear of dial)
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.14, -0.70, steps=35)
    ctx.end_effector.set_posture("left_gripper", "open", steps=12)
    tray_dir = (0.0, 0.50, -0.8660254)
    tray_close = (0.0, 0.8660254, 0.50)
    hook_pre = _tip_to_tcp((-0.045, -0.235, 0.865), tray_dir, tray_close)
    hook_engage = _tip_to_tcp((-0.045, -0.211, 0.793), tray_dir, tray_close)
    ctx.kinematics.move_to_pose("left_arm", hook_pre, steps=32)
    ctx.kinematics.move_to_pose("left_arm", hook_engage, steps=22)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=18)
    pull_poses = ctx.kinematics.plan_linear_sweep(
        chain="left_arm",
        start_pose=hook_engage,
        direction_world=(0.0, -1.0, 0.0),
        distance_m=0.040,
        num_waypoints=8,
    )
    ctx.kinematics.follow_cartesian_path("left_arm", pull_poses, steps_per_waypoint=4)
    ctx.end_effector.set_posture("left_gripper", "open", steps=14)
    hook_retract = _tip_to_tcp((-0.045, -0.275, 0.900), tray_dir, tray_close)
    ctx.kinematics.move_to_pose("left_arm", hook_retract, steps=22)
    print(
        "AFTER_S0",
        round(ctx.world.joint_value("creatista_machine", "drip_tray"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage0_drip_tray")

    # ------------------------------------------------------------------
    # Stage 1: Pick espresso_cup and place onto extended drip_tray (no base move while holding)
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.06, -0.70, steps=25)
    cup_pose = ctx.world.object_pose("espresso_cup")
    cup_dir = (0.0, 0.38, -0.925)
    cup_close = (1.0, 0.0, 0.0)
    cup_pre = _tip_to_tcp((cup_pose.xyz[0], cup_pose.xyz[1], 0.880), cup_dir, cup_close)
    cup_grasp = _tip_to_tcp((cup_pose.xyz[0], cup_pose.xyz[1], 0.785), cup_dir, cup_close)
    ctx.kinematics.move_to_pose("left_arm", cup_pre, steps=32)
    ctx.kinematics.move_to_pose("left_arm", cup_grasp, steps=22)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=20)

    cup_lift = _tip_to_tcp((cup_pose.xyz[0], cup_pose.xyz[1], 0.940), cup_dir, cup_close)
    tray_approach = _tip_to_tcp((-0.080, -0.260, 0.940), cup_dir, cup_close)
    tray_above = _tip_to_tcp((0.0, -0.210, 0.940), cup_dir, cup_close)
    tray_seat = _tip_to_tcp((0.0, -0.210, 0.822), cup_dir, cup_close)
    tray_retract = _tip_to_tcp((0.0, -0.300, 0.920), cup_dir, cup_close)
    ctx.kinematics.move_to_pose("left_arm", cup_lift, steps=40)
    ctx.kinematics.move_to_pose("left_arm", tray_approach, steps=35)
    ctx.kinematics.move_to_pose("left_arm", tray_above, steps=30)
    ctx.kinematics.move_to_pose("left_arm", tray_seat, steps=40)
    ctx.end_effector.set_posture("left_gripper", "open", steps=16)
    ctx.kinematics.move_to_pose("left_arm", tray_retract, steps=25)
    ctx.kinematics.move_joints("left_arm", home_l, steps=25)
    print(
        "AFTER_S1",
        [round(v, 4) for v in ctx.world.object_pose("espresso_cup").xyz],
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage1_cup")

    # ------------------------------------------------------------------
    # Stage 2: Pick coffee_capsule with right_arm (tilted approach) -> drop into chute
    # ------------------------------------------------------------------
    _move_base_to(ctx, -0.06, -0.68, steps=25)
    ctx.end_effector.set_posture("right_gripper", "open", steps=12)
    cap_pose = ctx.world.object_pose("coffee_capsule")
    cap_dir = (0.0, 0.45, -0.893)
    cap_close = (1.0, 0.0, 0.0)
    cap_pre = _tip_to_tcp((cap_pose.xyz[0], cap_pose.xyz[1], 0.870), cap_dir, cap_close)
    cap_grasp = _tip_to_tcp((cap_pose.xyz[0], cap_pose.xyz[1], 0.770), cap_dir, cap_close)
    ctx.kinematics.move_to_pose("right_arm", cap_pre, steps=35)
    ctx.kinematics.move_to_pose("right_arm", cap_grasp, steps=22)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=20)

    cap_lift = _tip_to_tcp((cap_pose.xyz[0], cap_pose.xyz[1], 1.100), cap_dir, cap_close)
    chute_approach = _tip_to_tcp((0.0, -0.220, 1.090), cap_dir, cap_close)
    chute_above = _tip_to_tcp((0.0, -0.130, 1.085), cap_dir, cap_close)
    chute_drop = _tip_to_tcp((0.0, -0.130, 1.050), cap_dir, cap_close)
    chute_retract = _tip_to_tcp((0.06, -0.260, 1.090), cap_dir, cap_close)
    ctx.kinematics.move_to_pose("right_arm", cap_lift, steps=40)
    ctx.kinematics.move_to_pose("right_arm", chute_approach, steps=35)
    ctx.kinematics.move_to_pose("right_arm", chute_above, steps=25)
    ctx.kinematics.move_to_pose("right_arm", chute_drop, steps=22)
    ctx.end_effector.set_posture("right_gripper", "open", steps=18)
    ctx.kinematics.move_to_pose("right_arm", chute_retract, steps=25)
    print(
        "AFTER_S2",
        [round(v, 4) for v in ctx.world.object_pose("coffee_capsule").xyz],
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage2_capsule")

    # ------------------------------------------------------------------
    # Stage 3: Close brew_lever with constant-orientation circular arc
    # ------------------------------------------------------------------
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    lever_dir = (0.0, 0.5735764, -0.8191520)
    lever_close = (1.0, 0.0, 0.0)
    lever_pre = _tip_to_tcp((0.0, -0.070, 1.195), lever_dir, lever_close)
    lever_start_tip = (0.0, -0.088, 1.154)
    lever_start = _tip_to_tcp(lever_start_tip, lever_dir, lever_close)
    ctx.kinematics.move_to_pose("right_arm", lever_pre, steps=25)
    ctx.kinematics.move_to_pose("right_arm", lever_start, steps=20)
    lever_arc = _revolute_push_waypoints(
        start_tip_xyz=lever_start_tip,
        pivot_world=(0.0, -0.075, 1.051),
        delta_angle_x_rad=1.24,
        approach_dir=lever_dir,
        closing_dir=lever_close,
        num_waypoints=12,
    )
    ctx.kinematics.follow_cartesian_path("right_arm", lever_arc, steps_per_waypoint=4)
    lever_retract = _tip_to_tcp((0.0, -0.260, 1.070), lever_dir, lever_close)
    ctx.kinematics.move_to_pose("right_arm", lever_retract, steps=22)
    print(
        "AFTER_S3",
        round(ctx.world.joint_value("creatista_machine", "brew_lever"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage3_lever")

    # ------------------------------------------------------------------
    # Stage 4: Turn select_dial (+Y axis, pivot [0.0, -0.214, 0.983], target [1.1345, 1.5708])
    # ------------------------------------------------------------------
    _move_base_to(ctx, -0.10, -0.74, steps=30)
    ctx.kinematics.set_torso_lift(0.28, steps=15)
    ctx.step(5)
    ctx.end_effector.set_posture("right_gripper", "open", steps=12)
    dial_dir = (0.0, 0.9396926, -0.3420201)
    dial_close = (1.0, 0.0, 0.0)
    dial_pre = _tip_to_tcp((0.0, -0.275, 1.005), dial_dir, dial_close)
    dial_grasp = _tip_to_tcp((0.0, -0.210, 0.983), dial_dir, dial_close)
    ctx.kinematics.move_to_pose("right_arm", dial_pre, steps=28)
    ctx.kinematics.move_to_pose("right_arm", dial_grasp, steps=20)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=18)
    dial_arc = ctx.kinematics.plan_revolute_arc(
        chain="right_arm",
        start_pose=dial_grasp,
        hinge_pivot_world=(0.0, -0.214, 0.983),
        hinge_axis_world=(0.0, 1.0, 0.0),
        delta_angle_rad=1.38,
        num_waypoints=10,
    )
    ctx.kinematics.follow_cartesian_path("right_arm", dial_arc, steps_per_waypoint=4)
    print(
        "AFTER_S4",
        round(ctx.world.joint_value("creatista_machine", "select_dial"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage4_dial")

    # ------------------------------------------------------------------
    # Stage 5: Push & HOLD spring-loaded dial_press along +Y ([0.0014, 0.0031])
    # ------------------------------------------------------------------
    dial_end = dial_arc[-1]
    press_hold = Pose3D(
        xyz=(dial_end.xyz[0], dial_end.xyz[1] + 0.0022, dial_end.xyz[2]),
        wxyz=dial_end.wxyz,
    )
    ctx.kinematics.move_to_pose("right_arm", press_hold, steps=20)
    ctx.step(20)
    print(
        "AFTER_S5",
        round(ctx.world.joint_value("creatista_machine", "dial_press"), 5),
        ctx.world.stage_scores(),
    )
    _ = home_r
