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


def _move_base_to(ctx, target_x: float, target_y: float, steps: int = 35) -> None:
    bx, by, _ = ctx.kinematics.base_pose_world().xyz
    ctx.kinematics.move_base_world(target_x - bx, target_y - by, 0.0, steps=steps)


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=10)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])

    # ------------------------------------------------------------------
    # Stage 0: Pick pcba_carrier_board (no base move while holding), rotate -90 deg Z, drop into nest
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.04, -0.58, steps=35)
    ctx.end_effector.set_posture("left_gripper", "open", steps=12)
    board_pose = ctx.world.object_pose("pcba_carrier_board")
    board_dir = (0.0, 0.0871557, -0.9961947)
    pcba_pre = _tip_to_tcp((board_pose.xyz[0], board_pose.xyz[1], 0.880), board_dir, (1.0, 0.0, 0.0))
    pcba_grasp = _tip_to_tcp((board_pose.xyz[0], board_pose.xyz[1], 0.768), board_dir, (1.0, 0.0, 0.0))
    ctx.kinematics.move_to_pose("left_arm", pcba_pre, steps=32)
    ctx.kinematics.move_to_pose("left_arm", pcba_grasp, steps=22)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=22)

    place_close = (0.0, -0.9961947, -0.0871557)
    pcba_lift = _tip_to_tcp((board_pose.xyz[0], board_pose.xyz[1], 0.965), board_dir, (1.0, 0.0, 0.0))
    nest_pre = _tip_to_tcp((0.0, -0.015, 0.960), board_dir, place_close)
    nest_drop = _tip_to_tcp((0.0, -0.015, 0.903), board_dir, place_close)
    nest_retract = _tip_to_tcp((0.0, -0.015, 1.010), board_dir, place_close)
    ctx.kinematics.move_to_pose("left_arm", pcba_lift, steps=40)
    ctx.kinematics.move_to_pose("left_arm", nest_pre, steps=50)
    ctx.kinematics.move_to_pose("left_arm", nest_drop, steps=35)
    ctx.end_effector.set_posture("left_gripper", "open", steps=20)
    ctx.step(10)
    ctx.kinematics.move_to_pose("left_arm", nest_retract, steps=25)
    ctx.kinematics.move_joints("left_arm", home_l, steps=25)
    print(
        "AFTER_S0",
        [round(v, 4) for v in ctx.world.object_pose("pcba_carrier_board").xyz],
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage0_board")

    # ------------------------------------------------------------------
    # Stage 3: Flip mode_toggle FIRST while ClampLever & Hood are up out of the way
    # ------------------------------------------------------------------
    _move_base_to(ctx, -0.326, -0.84, steps=35)
    ctx.end_effector.set_posture("right_gripper", "open", steps=12)
    tog_app = (0.0, 0.9848078, -0.1736482)
    tog_cls = (0.0, 0.1736482, 0.9848078)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((-0.044, -0.275, 1.063), tog_app, tog_cls), steps=22
    )
    ctx.kinematics.set_torso_lift(0.28, steps=18)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((-0.044, -0.265, 0.883), tog_app, tog_cls), steps=18
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((-0.044, -0.212, 0.883), tog_app, tog_cls), steps=24
    )
    for i in range(1, 13):
        frac = i / 12.0
        wp = (-0.044, -0.212 + 0.009 * frac, 0.883 + 0.025 * frac)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(wp, tog_app, tog_cls), steps=40)
        mval = ctx.world.joint_value("ingun_fixture", "mode_toggle")
        print("  TOG_SCOOP", i, [round(v, 4) for v in wp], "val=", round(mval, 5))
        if mval <= -0.32:
            break
    ctx.step(8)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((-0.044, -0.265, 0.910), tog_app, tog_cls), steps=18
    )
    ctx.kinematics.set_torso_lift(0.46, steps=18)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    ctx.kinematics.move_joints("right_arm", home_r, steps=22)
    print(
        "AFTER_S3",
        round(ctx.world.joint_value("ingun_fixture", "mode_toggle"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage3_toggle")

    # ------------------------------------------------------------------
    # Pre-clear ClampLever & Close pressure_hood (Stage 1)
    # ------------------------------------------------------------------
    _move_base_to(ctx, -0.18, -0.62, steps=30)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    clamp_close = (1.0, 0.0, 0.0)
    clamp_dir0 = (0.0, 0.7071068, -0.7071068)
    clamp_air = _tip_to_tcp((0.0, -0.100, 1.275), clamp_dir0, clamp_close)
    clamp_pre = _tip_to_tcp((0.0, 0.068, 1.250), clamp_dir0, clamp_close)
    clamp_start_tip = (0.0, 0.068, 1.200)
    ctx.kinematics.move_to_pose("right_arm", clamp_air, steps=22)
    ctx.kinematics.move_to_pose("right_arm", clamp_pre, steps=20)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp(clamp_start_tip, clamp_dir0, clamp_close), steps=16
    )
    pivot_c = (0.0, 0.085, 0.897)
    cdy0 = clamp_start_tip[1] - pivot_c[1]
    cdz0 = clamp_start_tip[2] - pivot_c[2]
    for i in range(1, 13):
        ang = 1.05 * (i / 12.0)
        c, s = math.cos(ang), math.sin(ang)
        tip = (0.0, pivot_c[1] + cdy0 * c - cdz0 * s, pivot_c[2] + cdy0 * s + cdz0 * c)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(tip, clamp_dir0, clamp_close), steps=10)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.0, -0.260, 1.060), clamp_dir0, clamp_close), steps=18
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.175, -0.200, 1.180), clamp_dir0, clamp_close), steps=18
    )
    print("AFTER_PRE_CLAMP", round(ctx.world.joint_value("ingun_fixture", "clamp_lever"), 5))

    _move_base_to(ctx, -0.08, -0.58, steps=24)
    hood_dir = (0.0, 0.7071068, -0.7071068)
    hood_close = (1.0, 0.0, 0.0)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.170, -0.050, 1.230), hood_dir, hood_close), steps=20
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.170, 0.175, 1.225), hood_dir, hood_close), steps=20
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.04, 0.175, 1.225), hood_dir, hood_close), steps=20
    )
    hood_start_tip = (0.04, 0.175, 1.162)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp(hood_start_tip, hood_dir, hood_close), steps=16
    )
    pivot_h = (0.0, 0.190, 0.929)
    hdy0 = hood_start_tip[1] - pivot_h[1]
    hdz0 = hood_start_tip[2] - pivot_h[2]
    tip = hood_start_tip
    for i in range(1, 18):
        ang = 1.40 * (i / 17.0)
        c, s = math.cos(ang), math.sin(ang)
        tip = (0.04, pivot_h[1] + hdy0 * c - hdz0 * s, pivot_h[2] + hdy0 * s + hdz0 * c)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp(tip, hood_dir, hood_close), steps=10)
        hval = ctx.world.joint_value("ingun_fixture", "pressure_hood")
        if hval >= -0.004:
            print("  HOOD_DONE", i, round(hval, 5), [round(x, 3) for x in tip])
            break
    ctx.step(6)
    ctx.end_effector.set_posture("right_gripper", "open", steps=10)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.04, tip[1], 1.150), hood_dir, hood_close), steps=16
    )
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.04, -0.175, 1.150), hood_dir, hood_close), steps=16
    )
    print(
        "AFTER_HOOD",
        round(ctx.world.joint_value("ingun_fixture", "pressure_hood"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage1_hood")

    # ------------------------------------------------------------------
    # Stage 2: Lock clamp_lever inside [-0.10472, 0.0]
    # ------------------------------------------------------------------
    _move_base_to(ctx, -0.18, -0.62, steps=20)
    down_dir = (0.0, 0.1736, -0.9848)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.0, -0.175, 1.130), down_dir, clamp_close), steps=20
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.0, -0.195, 0.990), down_dir, clamp_close), steps=16
    )
    for k in range(1, 10):
        frac = k / 9.0
        cy = -0.195 - 0.014 * frac
        cz = 0.990 - 0.075 * frac
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((0.0, cy, cz), down_dir, clamp_close), steps=8
        )
        cval = ctx.world.joint_value("ingun_fixture", "clamp_lever")
        if cval >= -0.060:
            print("  CLAMP_DONE", k, round(cval, 5), (round(cy, 3), round(cz, 3)))
            break
    ctx.step(8)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.0, -0.255, 0.945), down_dir, clamp_close), steps=14
    )
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.0, -0.275, 1.130), down_dir, clamp_close), steps=18
    )
    print(
        "AFTER_CLAMP_LOCK",
        round(ctx.world.joint_value("ingun_fixture", "clamp_lever"), 5),
        round(ctx.world.joint_value("ingun_fixture", "pressure_hood"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage2_clamp")

    # ------------------------------------------------------------------
    # Stage 4: Approach nearly horizontally UNDER FrontCrossHandle to press & hold TestStartButton
    # ------------------------------------------------------------------
    ctx.kinematics.set_torso_lift(0.28, steps=18)
    _move_base_to(ctx, -0.24, -0.84, steps=24)
    btn_dir = (0.0, 0.9396926, -0.3420201)
    btn_close = (1.0, 0.0, 0.0)
    slide_dir = (0.0, 0.6964754, -0.7175807)
    ctx.kinematics.move_to_pose(
        "right_arm", _tip_to_tcp((0.042, -0.255, 0.855), btn_dir, btn_close), steps=20
    )
    depth = -0.0010
    for k in range(22):
        py = -0.2080 + depth * slide_dir[1]
        pz = 0.8405 + depth * slide_dir[2]
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((0.042, py, pz), btn_dir, btn_close), steps=6
        )
        bval = ctx.world.joint_value("ingun_fixture", "test_start_button")
        if -0.0030 <= bval <= -0.0018:
            print("  BTN_HIT", k, round(depth, 5), round(bval, 5))
            break
        if bval > -0.0018:
            depth += 0.0002
        else:
            depth -= 0.00015
    ctx.step(20)
    for k in range(10):
        bval = ctx.world.joint_value("ingun_fixture", "test_start_button")
        if -0.0031 <= bval <= -0.0017:
            print("  BTN_TRIM_OK", k, round(depth, 5), round(bval, 5))
            break
        depth += 0.00015 if bval > -0.0017 else -0.00015
        py = -0.2080 + depth * slide_dir[1]
        pz = 0.8405 + depth * slide_dir[2]
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((0.042, py, pz), btn_dir, btn_close), steps=6
        )
    ctx.step(15)
    print(
        "AFTER_S4",
        round(ctx.world.joint_value("ingun_fixture", "test_start_button"), 5),
        ctx.world.stage_scores(),
    )
