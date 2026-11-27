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


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])
    x_cls = (1.0, 0.0, 0.0)
    top_dir = (0.0, 0.0500000, -0.9987492)
    btn_dir = (0.0, 0.2588190, -0.9659258)

    dx, dy, dz = ctx.world.object_pose("wireless_uv_dock").xyz
    pbtn = ctx.world.named_frame("wireless_uv_dock", "power_btn_press").xyz
    ebtn = ctx.world.named_frame("wireless_uv_dock", "pod_eject_press").xyz
    bbtn = ctx.world.named_frame("wireless_uv_dock", "boost_btn_press").xyz
    lcoil = ctx.world.named_frame("wireless_uv_dock", "left_phone_coil").xyz
    rcoil = ctx.world.named_frame("wireless_uv_dock", "right_earbuds_bay").xyz

    # Stage 0: Open left_gripper so single 10mm +X jaw presses PWR into [-0.0052, -0.0024] & holds >= 20 steps
    ctx.end_effector.set_posture("left_gripper", "open", steps=10)
    px_tcp = pbtn[0] - 0.0439
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((px_tcp, pbtn[1], pbtn[2] + 0.075), btn_dir, x_cls), steps=20)
    for i in range(1, 18):
        off_z = 0.008 - 0.017 * (i / 17.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((px_tcp, pbtn[1], pbtn[2] + off_z), btn_dir, x_cls), steps=4)
        if ctx.world.joint_value("wireless_uv_dock", "power_btn") <= -0.0030:
            break
    ctx.step(22)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((px_tcp, pbtn[1], pbtn[2] + 0.090), btn_dir, x_cls), steps=16)

    # Stage 1: Right arm presses EJECT with single 10mm -X jaw (<= -0.0030 m for >= 20 steps) -> wait motorized lid open
    ctx.end_effector.set_posture("right_gripper", "open", steps=10)
    ex_tcp = ebtn[0] + 0.0439
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ex_tcp, ebtn[1], ebtn[2] + 0.085), btn_dir, x_cls), steps=20)
    for i in range(1, 18):
        off_z = 0.008 - 0.017 * (i / 17.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ex_tcp, ebtn[1], ebtn[2] + off_z), btn_dir, x_cls), steps=4)
        if ctx.world.joint_value("wireless_uv_dock", "pod_eject_btn") <= -0.0030:
            break
    ctx.step(22)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ex_tcp, ebtn[1], ebtn[2] + 0.115), btn_dir, x_cls), steps=16)
    ctx.kinematics.move_joints("right_arm", home_r, steps=18)
    for _ in range(4):
        if ctx.world.joint_value("wireless_uv_dock", "uv_pod_lid") <= -1.05:
            break
        ctx.step(12)

    # Stage 2: Left arm picks smartphone dynamically high above eyeglasses/candle & aligns onto left_phone_coil
    ph = ctx.world.object_pose("smartphone")
    px, py, _ = rotate_vec_by_quat_wxyz(ph.wxyz, (1.0, 0.0, 0.0))
    if px < 0.0:
        px, py = -px, -py
    ph_cls = (float(px), float(py), 0.0)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ph.xyz[0], ph.xyz[1], ph.xyz[2] + 0.125), top_dir, ph_cls), steps=22)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ph.xyz[0], ph.xyz[1], ph.xyz[2] + 0.0040), top_dir, ph_cls), steps=18)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ph.xyz[0], ph.xyz[1], lcoil[2] + 0.145), top_dir, ph_cls), steps=22)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((lcoil[0], lcoil[1] - 0.032, lcoil[2] + 0.130), top_dir, x_cls), steps=22)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((lcoil[0], lcoil[1] - 0.032, lcoil[2] + 0.0065), top_dir, x_cls), steps=20)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((lcoil[0], lcoil[1] - 0.002, lcoil[2] + 0.0065), top_dir, x_cls), steps=18)
    ctx.end_effector.set_posture("left_gripper", "open", steps=14)
    ctx.step(6)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((lcoil[0], lcoil[1] - 0.032, lcoil[2] + 0.014), top_dir, x_cls), steps=14)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((lcoil[0], lcoil[1] - 0.032, lcoil[2] + 0.135), top_dir, x_cls), steps=18)
    ctx.kinematics.move_joints("left_arm", home_l, steps=18)

    # Stage 3: Right arm picks tws_earbuds_case dynamically high above hand cream/tray & drops inside open UV-C bay
    ctx.end_effector.set_posture("right_gripper", "open", steps=10)
    bd = ctx.world.object_pose("tws_earbuds_case")
    bx, by, _ = rotate_vec_by_quat_wxyz(bd.wxyz, (1.0, 0.0, 0.0))
    if bx < 0.0:
        bx, by = -bx, -by
    bd_cls = (float(bx), float(by), 0.0)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bd.xyz[0], bd.xyz[1], bd.xyz[2] + 0.125), top_dir, bd_cls), steps=22)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bd.xyz[0], bd.xyz[1], bd.xyz[2] + 0.009), top_dir, bd_cls), steps=18)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=18)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bd.xyz[0], bd.xyz[1], rcoil[2] + 0.165), top_dir, bd_cls), steps=22)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((rcoil[0], rcoil[1], rcoil[2] + 0.155), top_dir, x_cls), steps=22)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((rcoil[0], rcoil[1], rcoil[2] + 0.010), top_dir, x_cls), steps=20)
    ctx.end_effector.set_posture("right_gripper", "open", steps=14)
    ctx.step(6)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((rcoil[0], rcoil[1], rcoil[2] + 0.145), top_dir, x_cls), steps=18)

    # Stage 4: Canonicalize right arm, push UvPodFrontLeaf -Y past -0.68 rad latch, retract upward & wait for motor seal
    ctx.kinematics.move_joints("right_arm", home_r, steps=16)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    lid_x = dx + 0.068
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((lid_x, dy + 0.088, dz + 0.145), top_dir, x_cls), steps=20)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((lid_x, dy + 0.088, dz + 0.100), top_dir, x_cls), steps=16)
    for k in range(1, 15):
        frac = k / 14.0
        wy = dy + 0.088 + (-0.004 - 0.088) * frac
        wz = dz + 0.100 + (0.072 - 0.100) * frac
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((lid_x, wy, wz), top_dir, x_cls), steps=6)
        if ctx.world.joint_value("wireless_uv_dock", "uv_pod_lid") >= -0.68:
            break
    cur_y = float(ctx.kinematics.ee_pose("right_arm").xyz[1])
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((lid_x, cur_y, dz + 0.145), top_dir, x_cls), steps=16)
    for _ in range(14):
        lid_val = abs(ctx.world.joint_value("wireless_uv_dock", "uv_pod_lid"))
        leaf_val = abs(ctx.world.joint_value("wireless_uv_dock", "uv_pod_leaf_knuckle"))
        if lid_val <= 0.007 and leaf_val <= 0.007:
            break
        ctx.step(6)
    ctx.step(8)

    # Stage 5: Right arm opens & presses BOOST with single 10mm -X jaw and HOLDS >= 60 steps through end
    ctx.end_effector.set_posture("right_gripper", "open", steps=10)
    bx_tcp = bbtn[0] + 0.0439
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx_tcp, bbtn[1], bbtn[2] + 0.085), btn_dir, x_cls), steps=20)
    for i in range(1, 18):
        off_z = 0.008 - 0.017 * (i / 17.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((bx_tcp, bbtn[1], bbtn[2] + off_z), btn_dir, x_cls), steps=4)
        if -0.0052 <= ctx.world.joint_value("wireless_uv_dock", "boost_btn") <= -0.0028:
            break
    ctx.step(60)
