import math

from trajectory_forge.sdk.math3d import (
    Pose3D,
    compose_pose,
    invert_pose,
    quat_from_axis_angle_wxyz,
    rotate_vec_by_quat_wxyz,
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


def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=8)
    home_l = tuple(ctx.sensors.proprio()[0:6])
    home_r = tuple(ctx.sensors.proprio()[8:14])
    x_cls = (1.0, 0.0, 0.0)
    btn_dir = (0.0, 0.22, -0.9755)

    dx, dy, dz = ctx.world.object_pose("tb_dock").xyz
    pwr_f = ctx.world.named_frame("tb_dock", "host_power_press").xyz
    nvme_slot = ctx.world.named_frame("tb_dock", "drive_cartridge_seat").xyz
    sd_slot = ctx.world.named_frame("tb_dock", "sd_card_seat").xyz
    dial_f = ctx.world.named_frame("tb_dock", "mode_dial_grip").xyz
    ing_f = ctx.world.named_frame("tb_dock", "ingest_press").xyz

    ctx.end_effector.set_posture("left_gripper", "closed", steps=10)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((pwr_f[0], pwr_f[1], pwr_f[2] + 0.075), btn_dir, x_cls), steps=18)
    for k in range(1, 18):
        pz = pwr_f[2] + 0.008 - 0.015 * (k / 17.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((pwr_f[0], pwr_f[1], pz), btn_dir, x_cls), steps=4)
        ctx.step(3)
        if ctx.world.joint_value("tb_dock", "host_power_button") <= -0.0022:
            ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((pwr_f[0], pwr_f[1], pz - 0.0008), btn_dir, x_cls), steps=6)
            break
    ctx.step(22)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((pwr_f[0], pwr_f[1], pwr_f[2] + 0.085), btn_dir, x_cls), steps=14)

    pivot = (dx - 0.098, dy - 0.0565, dz + 0.0255)
    radius = 0.053
    front_dir = (0.0, 0.7880, -0.6157)

    # Drag the top edge of the door around the hinge with the closed fingertips.
    press_dir = (0.0, 0.02, -0.9998)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=12)
    for offset, height in ((0.0075, 0.0095), (0.0064, 0.0080)):
        start = ctx.world.joint_value("tb_dock", "drive_bay_door")
        if start < -1.30:
            break

        def top_pose(angle, rise=0.0):
            c, s = math.cos(angle), math.sin(angle)
            tip = (
                pivot[0] + 0.050 * c + offset * s,
                pivot[1] + 0.050 * s - offset * c,
                pivot[2] + height + rise,
            )
            return _tip_to_tcp(tip, press_dir, (c, s, 0.0))

        ctx.kinematics.move_to_pose("left_arm", top_pose(start, rise=0.044), steps=20)
        ctx.kinematics.move_to_pose("left_arm", top_pose(start), steps=20)
        ctx.step(8)
        for k in range(1, 17):
            angle = start + (-1.52 - start) * k / 16
            ctx.kinematics.move_to_pose("left_arm", top_pose(angle), steps=9)
            ctx.step(3)
            if ctx.world.joint_value("tb_dock", "drive_bay_door") < -1.46:
                break
        ctx.kinematics.move_to_pose("left_arm", top_pose(angle, rise=0.044), steps=20)
        ctx.step(8)

    # Sweep the inside of the open door to its stop before bringing over the cartridge.
    sweep_y = pivot[1] - 0.050
    sweep_z = pivot[2] + 0.003
    sweep_dir = (0.0, 0.18, -0.9837)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((pivot[0] + 0.032, sweep_y, sweep_z + 0.050), sweep_dir, x_cls), steps=18
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((pivot[0] + 0.032, sweep_y, sweep_z), sweep_dir, x_cls), steps=18
    )
    for offset in (0.020, 0.008, -0.004, -0.011):
        ctx.kinematics.move_to_pose(
            "left_arm", _tip_to_tcp((pivot[0] + offset, sweep_y, sweep_z), sweep_dir, x_cls), steps=14
        )
        ctx.step(4)
    ctx.step(8)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((pivot[0] - 0.011, sweep_y, sweep_z + 0.046), sweep_dir, x_cls), steps=20
    )

    # Grasp the trailing lip and square the cartridge before lowering in front of the lintel.
    cartridge_dir = (0.0, 0.18, -0.9837)
    for pick_y in (-0.030, -0.026):
        cartridge = ctx.world.object_pose("nvme_cartridge")
        cartridge_x = rotate_vec_by_quat_wxyz(cartridge.wxyz, x_cls)
        yaw = math.atan2(cartridge_x[1], cartridge_x[0])
        level = Pose3D(cartridge.xyz, quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), yaw))
        closing = rotate_vec_by_quat_wxyz(level.wxyz, x_cls)
        if closing[0] < 0.0:
            closing = (-closing[0], -closing[1], 0.0)
        ctx.end_effector.command_targets("left_gripper", (0.026, 0.026), steps=16)
        tip = compose_pose(level, Pose3D((0.0, pick_y, 0.0048))).xyz
        approach = rotate_vec_by_quat_wxyz(level.wxyz, cartridge_dir)
        grip = _tip_to_tcp(tip, approach, closing)
        ctx.kinematics.move_to_pose(
            "left_arm", compose_pose(grip, Pose3D((0.0, 0.0, -0.120))), steps=24
        )
        ctx.kinematics.move_to_pose("left_arm", grip, steps=20)
        ctx.step(8)
        ctx.end_effector.set_posture("left_gripper", "closed", steps=36)
        ctx.step(12)
        grip = ctx.kinematics.ee_pose("left_arm")
        ctx.kinematics.move_to_pose(
            "left_arm", Pose3D((grip.xyz[0], grip.xyz[1], grip.xyz[2] + 0.080), grip.wxyz), steps=24
        )
        if ctx.world.object_pose("nvme_cartridge").xyz[2] >= dz + 0.033:
            break

    cartridge = ctx.world.object_pose("nvme_cartridge")
    held = compose_pose(invert_pose(cartridge), ctx.kinematics.ee_pose("left_arm"))
    slot_x = nvme_slot[0]
    approach_y = dy - 0.115

    def cartridge_pose(y, z, orientation=(1.0, 0.0, 0.0, 0.0)):
        return compose_pose(Pose3D((slot_x, y, dz + z), orientation), held)

    ctx.kinematics.move_to_pose("left_arm", cartridge_pose(approach_y, 0.090), steps=28)
    ctx.step(8)
    cartridge = ctx.world.object_pose("nvme_cartridge")
    held = compose_pose(invert_pose(cartridge), ctx.kinematics.ee_pose("left_arm"))
    # Let the cartridge settle flat before taking the insertion grasp.
    staging_y = dy - 0.170
    ctx.kinematics.move_to_pose("left_arm", cartridge_pose(staging_y, 0.0025), steps=28)
    ctx.step(12)
    ctx.end_effector.command_targets("left_gripper", (0.028, 0.028), steps=20)
    ctx.step(16)
    cartridge = ctx.world.object_pose("nvme_cartridge")
    cartridge_x = rotate_vec_by_quat_wxyz(cartridge.wxyz, x_cls)
    yaw = math.atan2(cartridge_x[1], cartridge_x[0])
    level = Pose3D(cartridge.xyz, quat_from_axis_angle_wxyz((0.0, 0.0, 1.0), yaw))
    closing = rotate_vec_by_quat_wxyz(level.wxyz, x_cls)
    approach = rotate_vec_by_quat_wxyz(level.wxyz, cartridge_dir)
    tip = compose_pose(level, Pose3D((0.0, -0.026, 0.0048))).xyz
    grip = _tip_to_tcp(tip, approach, closing)
    ctx.kinematics.move_to_pose(
        "left_arm", compose_pose(grip, Pose3D((0.0, 0.0, -0.075))), steps=22
    )
    ctx.kinematics.move_to_pose("left_arm", grip, steps=24)
    ctx.step(8)
    for opening in (0.024, 0.020, 0.014):
        ctx.end_effector.command_targets("left_gripper", (opening, opening), steps=8)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=20)
    ctx.step(12)
    grip = ctx.kinematics.ee_pose("left_arm")
    ctx.kinematics.move_to_pose(
        "left_arm", Pose3D((grip.xyz[0], grip.xyz[1], grip.xyz[2] + 0.044), grip.wxyz), steps=28
    )
    ctx.step(12)
    cartridge = ctx.world.object_pose("nvme_cartridge")
    if cartridge.xyz[2] < dz + 0.022:
        return
    # Keep the wrist orientation from the resting grasp through insertion.
    wrist = ctx.kinematics.ee_pose("left_arm")
    wrist_offset = tuple(wrist.xyz[i] - cartridge.xyz[i] for i in range(3))

    def insertion_pose(y, z):
        x_off, y_off, z_off = wrist_offset
        return Pose3D((slot_x + x_off, y + y_off, dz + z + z_off), wrist.wxyz)

    ctx.kinematics.move_to_pose("left_arm", insertion_pose(approach_y, 0.044), steps=32)
    ctx.step(8)
    ctx.kinematics.move_to_pose("left_arm", insertion_pose(approach_y, 0.0225), steps=28)
    ctx.step(8)
    for depth in (-0.090, -0.069, -0.046):
        ctx.kinematics.move_to_pose("left_arm", insertion_pose(dy + depth, 0.0225), steps=18)
        ctx.step(6)

    # The floor supports the cartridge once its center crosses the mouth.
    ctx.kinematics.move_to_pose("left_arm", insertion_pose(dy - 0.046, 0.0180), steps=10)
    ctx.step(6)
    ctx.end_effector.command_targets("left_gripper", (0.026, 0.026), steps=18)
    ctx.step(12)
    ctx.kinematics.move_to_pose("left_arm", insertion_pose(dy - 0.055, 0.067), steps=20)

    push_dir = (0.0, 0.65, -0.7599)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=14)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((slot_x, dy - 0.114, dz + 0.062), push_dir, x_cls), steps=18
    )
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((slot_x, dy - 0.110, dz + 0.0215), push_dir, x_cls), steps=16
    )
    for k in range(1, 13):
        y = dy - 0.110 + 0.064 * k / 12
        ctx.kinematics.move_to_pose(
            "left_arm", _tip_to_tcp((slot_x, y, dz + 0.0215), push_dir, x_cls), steps=8
        )
        ctx.step(6)
        if ctx.world.object_pose("nvme_cartridge").xyz[1] >= dy - 0.0145:
            break
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((slot_x, dy - 0.117, dz + 0.075), push_dir, x_cls), steps=22
    )

    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((pivot[0] - 0.028, pivot[1] - 0.035, pivot[2] + 0.045), front_dir, x_cls), steps=16)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((pivot[0] - 0.028, pivot[1] - 0.035, pivot[2]), front_dir, x_cls), steps=14)
    ang_now = min(-0.20, ctx.world.joint_value("tb_dock", "drive_bay_door"))
    for k in range(1, 16):
        ang = ang_now * (1.0 - k / 15.0)
        hx = pivot[0] + radius * math.cos(ang) - 0.010 * (1.0 - k / 15.0)
        hy = pivot[1] + radius * math.sin(ang) + 0.004 * (k / 15.0) - 0.008 * (1.0 - k / 15.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((hx, hy, pivot[2]), front_dir, x_cls), steps=5)
        if ctx.world.joint_value("tb_dock", "drive_bay_door") >= -0.045:
            break
    # Finish the swing by pressing the free end of the face against the closed stop.
    ctx.kinematics.move_to_pose(
        "left_arm",
        _tip_to_tcp((pivot[0] + 0.048, pivot[1] + 0.004, pivot[2] + 0.002), front_dir, x_cls),
        steps=16,
    )
    ctx.step(8)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((dx - 0.06, dy - 0.120, dz + 0.080), front_dir, x_cls), steps=16)
    ctx.kinematics.move_joints("left_arm", home_l, steps=16)

    # Grasp the sides of the flap from above and lift it out of the slot.
    px, py, pz = dx + 0.0165, dy + 0.0265, dz + 0.0478
    vertical = (0.0, 0.0, -1.0)
    start = ctx.world.joint_value("tb_dock", "s_d_slot_gate")

    def gate_pose(angle, lift=0.0):
        tip = (
            px,
            py - 0.0112 * math.cos(angle),
            pz - 0.0112 * math.sin(angle) - 0.0003 + lift,
        )
        return _tip_to_tcp(tip, vertical, x_cls)

    ctx.end_effector.command_targets("right_gripper", (0.025, 0.025), steps=16)
    ctx.kinematics.move_to_pose("right_arm", gate_pose(start, lift=0.052), steps=22)
    ctx.kinematics.move_to_pose("right_arm", gate_pose(start), steps=20)
    ctx.step(10)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=32)
    ctx.step(8)
    for k in range(1, 15):
        angle = start + (-1.59 - start) * k / 14
        ctx.kinematics.move_to_pose("right_arm", gate_pose(angle), steps=8)
        ctx.step(3)
    ctx.step(8)
    ctx.end_effector.command_targets("right_gripper", (0.026, 0.026), steps=18)
    ctx.kinematics.move_to_pose("right_arm", gate_pose(angle, lift=0.055), steps=18)

    if ctx.world.joint_value("tb_dock", "s_d_slot_gate") > -1.40:
        press_dir = (0.0, 0.55, -0.8352)
        ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((px, py - 0.019, pz + 0.0095), press_dir, x_cls), steps=20
        )
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((px, py + 0.006, pz + 0.0095), press_dir, x_cls), steps=22
        )
        ctx.step(6)
        ctx.kinematics.move_to_pose(
            "right_arm", _tip_to_tcp((px, py + 0.006, pz + 0.065), press_dir, x_cls), steps=16
        )

    ctx.end_effector.set_posture("right_gripper", "open", steps=16)
    sd = ctx.world.object_pose("sd_card_holder")
    sx_v = rotate_vec_by_quat_wxyz(sd.wxyz, (1.0, 0.0, 0.0))
    if sx_v[0] < 0.0:
        sx_v = (-sx_v[0], -sx_v[1], -sx_v[2])
    sxn = math.hypot(sx_v[0], sx_v[1]) or 1.0
    sd_cls = (float(sx_v[0] / sxn), float(sx_v[1] / sxn), 0.0)
    sd_app = (-0.04 * sd_cls[1], 0.04 * sd_cls[0], -0.9992)
    sd_vdir = (0.0, 0.02, -0.9998)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((sd.xyz[0], sd.xyz[1], sd.xyz[2] + 0.110), sd_app, sd_cls), steps=20)
    ctx.step(4)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((sd.xyz[0], sd.xyz[1], sd.xyz[2] + 0.0025), sd_app, sd_cls), steps=16)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=36)
    ctx.step(8)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((sd.xyz[0], sd.xyz[1], dz + 0.145), sd_app, sd_cls), steps=18)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((sd_slot[0], dy + 0.0155, dz + 0.125), sd_vdir, x_cls), steps=18)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((sd_slot[0], dy + 0.0155, dz + 0.0485), sd_vdir, x_cls), steps=16)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "open", steps=20)
    ctx.step(12)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((sd_slot[0], dy + 0.0155, dz + 0.115), sd_vdir, x_cls), steps=16)

    # Turn the dial to its stop with two sweeps, then return it to the ingest band.
    cx, cy = dial_f[:2]
    dial_dir = (0.0, 0.06, -0.9982)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=12)
    for stroke, (dial_height, count, stride) in enumerate(((0.0550, 13, 8), (0.0532, 12, 7))):
        start = -1.10
        z = dz + dial_height

        def dial_pose(theta, rise=0.0):
            c, s = math.cos(theta), math.sin(theta)
            return _tip_to_tcp((cx + 0.010 * c, cy + 0.010 * s, z + rise), dial_dir, (-s, c, 0.0))

        ctx.kinematics.move_to_pose("right_arm", dial_pose(start, rise=0.045), steps=14)
        ctx.kinematics.move_to_pose("right_arm", dial_pose(start), steps=16)
        ctx.step(4)
        for k in range(1, count + 1):
            angle = start + 1.42 * k / count
            ctx.kinematics.move_to_pose("right_arm", dial_pose(angle), steps=stride)
        ctx.step(6)
        if stroke == 1:
            for k in range(1, 5):
                angle = 0.32 - 0.30 * k / 4
                ctx.kinematics.move_to_pose("right_arm", dial_pose(angle), steps=8)
            ctx.step(10)
        ctx.kinematics.move_to_pose("right_arm", dial_pose(angle, rise=0.045), steps=12)

    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ing_f[0], ing_f[1], ing_f[2] + 0.075), btn_dir, x_cls), steps=18)
    for k in range(1, 18):
        pz = ing_f[2] + 0.008 - 0.015 * (k / 17.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ing_f[0], ing_f[1], pz), btn_dir, x_cls), steps=4)
        ctx.step(3)
        if ctx.world.joint_value("tb_dock", "ingest_button") <= -0.0022:
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ing_f[0], ing_f[1], pz - 0.0008), btn_dir, x_cls), steps=6)
            break
    ctx.step(24)
    _ = home_r
