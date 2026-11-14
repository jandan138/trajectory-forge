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
    ctx.kinematics.set_torso_lift(0.46, steps=12)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])
    _ = home_l

    riser_pose = ctx.world.object_pose("desktop_riser_rig")
    rx, ry, rz = riser_pose.xyz

    # ------------------------------------------------------------------
    # Stage 0: Pull open desktop_riser_rig.drawer_right into [0.120, 0.165] m
    # Approach at 38 deg pitch so wrist stays clear of ColMidShelf (z=0.825) & keyboard (z=0.790).
    # ------------------------------------------------------------------
    _move_base_to(ctx, -0.12, -0.78, steps=35)
    ctx.kinematics.set_torso_lift(0.38, steps=14)
    ctx.end_effector.set_posture("right_gripper", "open", steps=12)

    handle_world = ctx.world.named_frame("desktop_riser_rig", "handle_right")
    hx, hy, hz = handle_world.xyz
    pull_dir = (0.0, 0.7880108, -0.6156615)
    pull_cls = (1.0, 0.0, 0.0)

    pull_pre = _tip_to_tcp((hx, hy - 0.090, hz + 0.070), pull_dir, pull_cls)
    pull_grasp = _tip_to_tcp((hx, hy - 0.001, hz - 0.001), pull_dir, pull_cls)
    ctx.kinematics.move_to_pose("right_arm", pull_pre, steps=28)
    ctx.kinematics.move_to_pose("right_arm", pull_grasp, steps=22)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=20)

    for i in range(1, 17):
        dy = -0.160 * (i / 16.0)
        wp = _tip_to_tcp((hx, hy - 0.001 + dy, hz - 0.001), pull_dir, pull_cls)
        ctx.kinematics.move_to_pose("right_arm", wp, steps=8)
        dval = ctx.world.joint_value("desktop_riser_rig", "drawer_right")
        if dval >= 0.152:
            break

    ctx.end_effector.set_posture("right_gripper", "open", steps=14)
    cur_d = ctx.world.joint_value("desktop_riser_rig", "drawer_right")
    pull_ret = _tip_to_tcp((hx, hy - cur_d - 0.075, hz + 0.090), pull_dir, pull_cls)
    ctx.kinematics.move_to_pose("right_arm", pull_ret, steps=20)
    ctx.kinematics.set_torso_lift(0.46, steps=14)
    ctx.kinematics.move_joints("right_arm", home_r, steps=22)
    print(
        "AFTER_S0",
        round(ctx.world.joint_value("desktop_riser_rig", "drawer_right"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage0_drawer_open")

    # ------------------------------------------------------------------
    # Stage 1: Pick chocopie_pack (70mm Y-width) & place in left half of open right drawer
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.05, -0.66, steps=30)
    ctx.end_effector.set_posture("right_gripper", "open", steps=12)

    cp_pose = ctx.world.object_pose("chocopie_pack")
    top_dir = (0.0, 0.1391731, -0.9902681)
    pick_y_cls = (0.0, -0.9902681, -0.1391731)
    place_x_cls = (1.0, 0.0, 0.0)

    cp_pre = _tip_to_tcp((cp_pose.xyz[0], cp_pose.xyz[1], 0.890), top_dir, pick_y_cls)
    cp_grasp = _tip_to_tcp((cp_pose.xyz[0], cp_pose.xyz[1], 0.768), top_dir, pick_y_cls)
    cp_lift = _tip_to_tcp((cp_pose.xyz[0], cp_pose.xyz[1], 0.945), top_dir, pick_y_cls)

    cur_d = max(0.140, ctx.world.joint_value("desktop_riser_rig", "drawer_right"))
    cp_target_xy = (rx + 0.080, ry - cur_d - 0.042)
    cp_above = _tip_to_tcp((cp_target_xy[0], cp_target_xy[1], 0.935), top_dir, place_x_cls)
    cp_drop = _tip_to_tcp((cp_target_xy[0], cp_target_xy[1], rz + 0.044), top_dir, place_x_cls)
    cp_ret = _tip_to_tcp((cp_target_xy[0], cp_target_xy[1], 0.945), top_dir, place_x_cls)

    ctx.kinematics.move_to_pose("right_arm", cp_pre, steps=28)
    ctx.kinematics.move_to_pose("right_arm", cp_grasp, steps=22)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=20)
    ctx.kinematics.move_to_pose("right_arm", cp_lift, steps=30)
    ctx.kinematics.move_to_pose("right_arm", cp_above, steps=38)
    ctx.kinematics.move_to_pose("right_arm", cp_drop, steps=28)
    ctx.end_effector.set_posture("right_gripper", "open", steps=16)
    ctx.step(8)
    ctx.kinematics.move_to_pose("right_arm", cp_ret, steps=22)
    print(
        "AFTER_S1",
        [round(v, 4) for v in ctx.world.object_pose("chocopie_pack").xyz],
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage1_chocopie_stored")

    # ------------------------------------------------------------------
    # Stage 2: Pick blue_ballpoint_pen (12mm Y-width) & place in right half of open right drawer
    # ------------------------------------------------------------------
    ctx.end_effector.set_posture("right_gripper", "open", steps=10)
    pen_pose = ctx.world.object_pose("blue_ballpoint_pen")
    pen_place_cls = (0.8660254, -0.4951340, -0.0695866)

    pen_pre = _tip_to_tcp((pen_pose.xyz[0], pen_pose.xyz[1], 0.880), top_dir, pick_y_cls)
    pen_grasp = _tip_to_tcp((pen_pose.xyz[0], pen_pose.xyz[1], 0.758), top_dir, pick_y_cls)
    pen_lift = _tip_to_tcp((pen_pose.xyz[0], pen_pose.xyz[1], 0.935), top_dir, pick_y_cls)

    cur_d = max(0.140, ctx.world.joint_value("desktop_riser_rig", "drawer_right"))
    pen_target_xy = (rx + 0.165, ry - cur_d - 0.038)
    pen_above = _tip_to_tcp((pen_target_xy[0], pen_target_xy[1], 0.930), top_dir, pen_place_cls)
    pen_drop = _tip_to_tcp((pen_target_xy[0], pen_target_xy[1], rz + 0.034), top_dir, pen_place_cls)
    pen_ret = _tip_to_tcp((pen_target_xy[0], pen_target_xy[1], 0.945), top_dir, pen_place_cls)

    ctx.kinematics.move_to_pose("right_arm", pen_pre, steps=28)
    ctx.kinematics.move_to_pose("right_arm", pen_grasp, steps=22)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=20)
    ctx.kinematics.move_to_pose("right_arm", pen_lift, steps=28)
    ctx.kinematics.move_to_pose("right_arm", pen_above, steps=36)
    ctx.kinematics.move_to_pose("right_arm", pen_drop, steps=26)
    ctx.end_effector.set_posture("right_gripper", "open", steps=16)
    ctx.step(8)
    ctx.kinematics.move_to_pose("right_arm", pen_ret, steps=22)
    ctx.kinematics.move_joints("right_arm", home_r, steps=22)
    print(
        "AFTER_S2",
        [round(v, 4) for v in ctx.world.object_pose("blue_ballpoint_pen").xyz],
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage2_pen_stored")

    # ------------------------------------------------------------------
    # Stage 3: Push desktop_riser_rig.drawer_right closed into [0.0, 0.025] m
    # ------------------------------------------------------------------
    _move_base_to(ctx, -0.12, -0.78, steps=28)
    ctx.kinematics.set_torso_lift(0.38, steps=14)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=12)

    cur_d = ctx.world.joint_value("desktop_riser_rig", "drawer_right")
    push_start_y = hy - cur_d - 0.030
    push_pre = _tip_to_tcp((hx, push_start_y - 0.030, hz + 0.060), pull_dir, pull_cls)
    push_start = _tip_to_tcp((hx, push_start_y, hz + 0.005), pull_dir, pull_cls)
    ctx.kinematics.move_to_pose("right_arm", push_pre, steps=24)
    ctx.kinematics.move_to_pose("right_arm", push_start, steps=18)

    total_push = (hy + 0.004) - push_start_y
    for i in range(1, 17):
        py = push_start_y + total_push * (i / 16.0)
        ctx.kinematics.move_to_pose(
            "right_arm",
            _tip_to_tcp((hx, py, hz + 0.005), pull_dir, pull_cls),
            steps=8,
        )
        if ctx.world.joint_value("desktop_riser_rig", "drawer_right") <= 0.010:
            break

    ctx.kinematics.move_to_pose(
        "right_arm",
        _tip_to_tcp((hx, hy - 0.085, hz + 0.085), pull_dir, pull_cls),
        steps=20,
    )
    ctx.kinematics.set_torso_lift(0.46, steps=14)
    ctx.kinematics.move_joints("right_arm", home_r, steps=22)
    print(
        "AFTER_S3",
        round(ctx.world.joint_value("desktop_riser_rig", "drawer_right"), 5),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage3_drawer_closed")

    # ------------------------------------------------------------------
    # Stage 4: Pick glass_water_mug (82mm Y-diameter) & place upright left of sprite_bottle_500ml
    # ------------------------------------------------------------------
    _move_base_to(ctx, 0.28, -0.68, steps=32)
    ctx.end_effector.set_posture("right_gripper", "open", steps=14)

    mug_pose = ctx.world.object_pose("glass_water_mug")
    sprite_pose = ctx.world.object_pose("sprite_bottle_500ml")
    mug_dir = (0.0, 0.1218693, -0.9925462)
    mug_cls = (0.0, -0.9925462, -0.1218693)

    mug_gx = mug_pose.xyz[0] + 0.006
    mug_gy = mug_pose.xyz[1]
    mug_pre = _tip_to_tcp((mug_gx, mug_gy, 0.915), mug_dir, mug_cls)
    mug_grasp = _tip_to_tcp((mug_gx, mug_gy, 0.806), mug_dir, mug_cls)
    mug_lift = _tip_to_tcp((mug_gx, mug_gy, 0.925), mug_dir, mug_cls)

    mug_tx = sprite_pose.xyz[0] - 0.115 + 0.006
    mug_ty = sprite_pose.xyz[1]
    mug_above = _tip_to_tcp((mug_tx, mug_ty, 0.925), mug_dir, mug_cls)
    mug_place = _tip_to_tcp((mug_tx, mug_ty, 0.808), mug_dir, mug_cls)
    mug_ret = _tip_to_tcp((mug_tx, mug_ty, 0.935), mug_dir, mug_cls)

    ctx.kinematics.move_to_pose("right_arm", mug_pre, steps=28)
    ctx.kinematics.move_to_pose("right_arm", mug_grasp, steps=24)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=24)
    ctx.kinematics.move_to_pose("right_arm", mug_lift, steps=32)
    ctx.kinematics.move_to_pose("right_arm", mug_above, steps=34)
    ctx.kinematics.move_to_pose("right_arm", mug_place, steps=30)
    ctx.end_effector.set_posture("right_gripper", "open", steps=18)
    ctx.step(12)
    ctx.kinematics.move_to_pose("right_arm", mug_ret, steps=24)
    ctx.kinematics.move_joints("right_arm", home_r, steps=22)
    _move_base_to(ctx, 0.0, -0.96, steps=28)
    ctx.step(12)
    print(
        "AFTER_S4",
        [round(v, 4) for v in ctx.world.object_pose("glass_water_mug").xyz],
        "score=",
        ctx.world.current_score(),
        ctx.world.stage_scores(),
    )
    ctx.checkpoint.save_if_search("stage4_mug_placed")
