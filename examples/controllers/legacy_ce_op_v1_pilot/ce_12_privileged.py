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


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])
    x_cls = (1.0, 0.0, 0.0)
    top_dir = (0.0, 0.0500000, -0.9987492)
    btn_dir = (0.0, 0.2588190, -0.9659258)

    pbtn = ctx.world.named_frame("qi_charger_rig", "power_btn_press").xyz
    bbtn = ctx.world.named_frame("qi_charger_rig", "qi_boost_press").xyz
    lcoil = ctx.world.named_frame("qi_charger_rig", "left_phone_coil").xyz
    rcoil = ctx.world.named_frame("qi_charger_rig", "right_earbuds_coil").xyz

    # Stage 0: Open left_gripper so single 10mm +X jaw (tcp_x + 0.0439) drops cleanly into 20mm power_btn trench
    ctx.end_effector.set_posture("left_gripper", "open", steps=10)
    px_tcp = pbtn[0] - 0.0439
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((px_tcp, pbtn[1], pbtn[2] + 0.065), btn_dir, x_cls), steps=22)
    for i in range(1, 18):
        dz = 0.008 - 0.018 * (i / 17.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((px_tcp, pbtn[1], pbtn[2] + dz), btn_dir, x_cls), steps=5)
        if ctx.world.joint_value("qi_charger_rig", "power_btn") >= 0.0038:
            break
    ctx.step(5)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((px_tcp, pbtn[1], pbtn[2] + 0.075), btn_dir, x_cls), steps=16)

    # Stage 1: Left arm picks 62mm CompactQiSmartphone dynamically & aligns onto left_phone_coil
    ph = ctx.world.object_pose("smartphone")
    px, py, _ = rotate_vec_by_quat_wxyz(ph.wxyz, (1.0, 0.0, 0.0))
    ph_cls = (float(px), float(py), 0.0)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ph.xyz[0], ph.xyz[1], ph.xyz[2] + 0.095), top_dir, ph_cls), steps=24)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ph.xyz[0], ph.xyz[1], ph.xyz[2] + 0.0040), top_dir, ph_cls), steps=20)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ph.xyz[0], ph.xyz[1], lcoil[2] + 0.115), top_dir, ph_cls), steps=22)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((lcoil[0], lcoil[1], lcoil[2] + 0.110), top_dir, x_cls), steps=24)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((lcoil[0], lcoil[1], lcoil[2] + 0.006), top_dir, x_cls), steps=22)
    ctx.end_effector.set_posture("left_gripper", "open", steps=14)
    ctx.step(6)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((lcoil[0], lcoil[1], lcoil[2] + 0.110), top_dir, x_cls), steps=18)
    ctx.kinematics.move_joints("left_arm", home_l, steps=18)

    # Stage 2: Right arm picks 50mm TwsQiEarbudsCase dynamically & places on right_earbuds_coil
    ctx.end_effector.set_posture("right_gripper", "open", steps=10)
    bd = ctx.world.object_pose("tws_earbuds_case")
    bx, by, _ = rotate_vec_by_quat_wxyz(bd.wxyz, (1.0, 0.0, 0.0))
    bd_cls = (float(bx), float(by), 0.0)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bd.xyz[0], bd.xyz[1], bd.xyz[2] + 0.095), top_dir, bd_cls), steps=24)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bd.xyz[0], bd.xyz[1], bd.xyz[2] + 0.009), top_dir, bd_cls), steps=20)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bd.xyz[0], bd.xyz[1], rcoil[2] + 0.115), top_dir, bd_cls), steps=22)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((rcoil[0], rcoil[1], rcoil[2] + 0.110), top_dir, x_cls), steps=24)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((rcoil[0], rcoil[1], rcoil[2] + 0.012), top_dir, x_cls), steps=22)
    ctx.end_effector.set_posture("right_gripper", "open", steps=14)
    ctx.step(6)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((rcoil[0], rcoil[1], rcoil[2] + 0.095), top_dir, x_cls), steps=18)

    # Stage 3: Open right_gripper so single 10mm -X jaw (tcp_x - 0.0439) drops cleanly into 20mm qi_boost_btn trench
    bx_tcp = bbtn[0] + 0.0439
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx_tcp, bbtn[1], bbtn[2] + 0.065), btn_dir, x_cls), steps=20)
    for i in range(1, 18):
        dz = 0.008 - 0.018 * (i / 17.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx_tcp, bbtn[1], bbtn[2] + dz), btn_dir, x_cls), steps=5)
        if ctx.world.joint_value("qi_charger_rig", "qi_boost_btn") >= 0.0038:
            break
    ctx.step(12)
    _ = home_r
