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
    pick_dir = (0.0, 0.15, -0.9887)
    btn_dir = (0.0, 0.24, -0.9708)

    dx, dy, dz = ctx.world.object_pose("tb_dock").xyz
    pwr_f = ctx.world.named_frame("tb_dock", "host_power_press").xyz
    nvme_slot = ctx.world.named_frame("tb_dock", "drive_cartridge_seat").xyz
    dial_f = ctx.world.named_frame("tb_dock", "mode_dial_grip").xyz
    ing_f = ctx.world.named_frame("tb_dock", "ingest_press").xyz

    mouse_f = ctx.world.named_frame("workstation", "mouse_dock").xyz
    shutter_f = ctx.world.named_frame("workstation", "io_shutter_handle").xyz
    enter_f = ctx.world.named_frame("workstation", "keyboard_enter").xyz
    tower_f = ctx.world.named_frame("workstation", "tower_power").xyz
    mon_f = ctx.world.named_frame("workstation", "monitor_power").xyz

    # Left arm operates left tb_dock: host_power -> drive_bay_door open -> insert nvme -> close bay -> mode_dial -> ingest
    ctx.end_effector.set_posture("left_gripper", "closed", steps=10)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((pwr_f[0], pwr_f[1], pwr_f[2] + 0.075), btn_dir, x_cls), steps=18)
    for k in range(1, 20):
        pz = pwr_f[2] + 0.008 - 0.0009 * k
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((pwr_f[0], pwr_f[1], pz), btn_dir, x_cls), steps=4)
        ctx.step(3)
        if ctx.world.joint_value("tb_dock", "host_power_button") <= -0.0022:
            ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((pwr_f[0], pwr_f[1], pz - 0.0008), btn_dir, x_cls), steps=6)
            break
    ctx.step(22)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((max(pwr_f[0], -0.50), pwr_f[1], max(pwr_f[2] + 0.085, dz + 0.145)), btn_dir, x_cls), steps=14)

    pivot = (dx - 0.098, dy - 0.0565, dz + 0.0255)
    radius = 0.053
    front_dir = (0.0, 0.7880, -0.6157)

    # Clamp the outer edge of the door above and below, then follow its hinge arc.
    for inset, height in ((0.003, -0.001), (0.0055, 0.0)):
        start = ctx.world.joint_value("tb_dock", "drive_bay_door")
        if start <= -1.56:
            break

        def door_pose(angle, retreat=0.0, rise=0.0, radial=0.0):
            c, s = math.cos(angle), math.sin(angle)
            tip = (
                pivot[0] + (radius + radial) * c - (inset - retreat) * s,
                pivot[1] + (radius + radial) * s + (inset - retreat) * c,
                pivot[2] + height + rise,
            )
            # Steepen the wrist near full opening to stay within reach at the left dock.
            opening = max(0.0, min(1.0, (-angle - 1.12) / 0.525))
            horizontal = 0.802 - 0.110 * opening
            down = -math.sqrt(1.0 - horizontal * horizontal)
            return _tip_to_tcp(tip, (-horizontal * s, horizontal * c, down), (0.0, 0.0, 1.0))

        ctx.end_effector.command_targets("left_gripper", (0.017, 0.017), steps=16)
        ctx.kinematics.move_to_pose("left_arm", door_pose(start, retreat=0.040, rise=0.018), steps=24)
        ctx.kinematics.move_to_pose("left_arm", door_pose(start), steps=26)
        ctx.step(8)
        ctx.end_effector.set_posture("left_gripper", "closed", steps=32)
        ctx.step(8)
        for k in range(1, 20):
            angle = start + (-1.645 - start) * k / 19
            ctx.kinematics.move_to_pose("left_arm", door_pose(angle), steps=8)
            ctx.step(4)
            if ctx.world.joint_value("tb_dock", "drive_bay_door") <= -1.60:
                break
        ctx.end_effector.command_targets("left_gripper", (0.022, 0.022), steps=16)
        ctx.kinematics.move_to_pose("left_arm", door_pose(angle, rise=0.012, radial=0.026), steps=18)
        ctx.kinematics.move_to_pose(
            "left_arm",
            _tip_to_tcp((pivot[0], pivot[1] - 0.083, dz + 0.145), front_dir, x_cls),
            steps=22,
        )
        ctx.step(12)

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
    for offset in (0.020, 0.008, -0.004, -0.009):
        ctx.kinematics.move_to_pose(
            "left_arm", _tip_to_tcp((pivot[0] + offset, sweep_y, sweep_z), sweep_dir, x_cls), steps=14
        )
        ctx.step(4)
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((pivot[0] + 0.032, sweep_y, sweep_z + 0.046), sweep_dir, x_cls), steps=20
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

    def cartridge_pose(y, z):
        return compose_pose(Pose3D((slot_x, y, dz + z)), held)

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
        if ctx.world.object_pose("nvme_cartridge").xyz[1] >= dy - 0.013:
            break
    ctx.kinematics.move_to_pose(
        "left_arm", _tip_to_tcp((slot_x, dy - 0.117, dz + 0.075), push_dir, x_cls), steps=22
    )

    ctx.end_effector.set_posture("left_gripper", "closed", steps=10)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((max(pivot[0] - 0.028, -0.50), pivot[1] - 0.035, dz + 0.145), front_dir, x_cls), steps=16)
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
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((max(dx - 0.06, -0.50), dy - 0.120, dz + 0.155), front_dir, x_cls), steps=14)

    # Drive the dial to its upper stop, then bring it back into the ingest band.
    cx, cy = dial_f[:2]
    dial_dir = (0.0, 0.06, -0.9982)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=16)
    for stroke, dial_height in enumerate((0.0550, 0.0532)):
        start = -1.10
        z = dz + dial_height

        def dial_pose(theta, rise=0.0):
            c, s = math.cos(theta), math.sin(theta)
            return _tip_to_tcp((cx + 0.010 * c, cy + 0.010 * s, z + rise), dial_dir, (-s, c, 0.0))

        ctx.kinematics.move_to_pose("left_arm", dial_pose(start, rise=0.045), steps=18)
        ctx.kinematics.move_to_pose("left_arm", dial_pose(start), steps=22)
        ctx.step(6)
        for k in range(1, 15):
            angle = start + 1.42 * k / 14
            ctx.kinematics.move_to_pose("left_arm", dial_pose(angle), steps=8)
        ctx.step(12)
        if stroke == 1:
            for k in range(1, 5):
                angle = 0.32 - 0.30 * k / 4
                ctx.kinematics.move_to_pose("left_arm", dial_pose(angle), steps=8)
            ctx.step(12)
        ctx.kinematics.move_to_pose("left_arm", dial_pose(angle, rise=0.045), steps=18)

    ctx.end_effector.set_posture("left_gripper", "closed", steps=10)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ing_f[0], ing_f[1], ing_f[2] + 0.070), btn_dir, x_cls), steps=16)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ing_f[0], ing_f[1], ing_f[2] + 0.014), btn_dir, x_cls), steps=14)
    ctx.step(4)
    for k in range(1, 20):
        pz = ing_f[2] + 0.008 - 0.0009 * k
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ing_f[0], ing_f[1], pz), btn_dir, x_cls), steps=4)
        ctx.step(3)
        if ctx.world.joint_value("tb_dock", "ingest_button") <= -0.0022:
            ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((ing_f[0], ing_f[1], pz - 0.0008), btn_dir, x_cls), steps=6)
            break
    ctx.step(24)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((max(ing_f[0], -0.50), ing_f[1], dz + 0.160), btn_dir, x_cls), steps=14)

    # Right arm docks mouse on workstation + confirms KeyboardEnter OSD dialog
    ctx.end_effector.set_posture("right_gripper", "open", steps=16)
    ms = ctx.world.object_pose("desk_mouse")
    mx_v = rotate_vec_by_quat_wxyz(ms.wxyz, (1.0, 0.0, 0.0))
    if mx_v[0] < 0.0:
        mx_v = (-mx_v[0], -mx_v[1], -mx_v[2])
    ms_cls = (float(mx_v[0]), float(mx_v[1]), 0.0)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ms.xyz[0], ms.xyz[1], ms.xyz[2] + 0.115), pick_dir, ms_cls), steps=28)
    ctx.step(4)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ms.xyz[0], ms.xyz[1], ms.xyz[2] + 0.014), pick_dir, ms_cls), steps=20)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=36)
    ctx.step(8)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((ms.xyz[0], ms.xyz[1], ms.xyz[2] + 0.155), pick_dir, ms_cls), steps=20)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mouse_f[0], mouse_f[1], mouse_f[2] + 0.125), pick_dir, x_cls), steps=22)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mouse_f[0], mouse_f[1], mouse_f[2] + 0.018), pick_dir, x_cls), steps=18)
    ctx.step(8)
    ctx.end_effector.set_posture("right_gripper", "open", steps=20)
    ctx.step(12)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mouse_f[0], mouse_f[1], mouse_f[2] + 0.125), pick_dir, x_cls), steps=16)

    enter_dir = (-0.12, 0.04, -0.992)
    ctx.end_effector.set_posture("right_gripper", "closed", steps=10)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((enter_f[0], enter_f[1], enter_f[2] + 0.075), enter_dir, x_cls), steps=18)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((enter_f[0], enter_f[1], enter_f[2] + 0.012), enter_dir, x_cls), steps=16)
    ctx.step(4)
    for k in range(1, 20):
        pz = enter_f[2] + 0.006 - 0.0009 * k
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((enter_f[0], enter_f[1], pz), enter_dir, x_cls), steps=5)
        ctx.step(3)
        if ctx.world.joint_value("workstation", "keyboard_enter") <= -0.0022:
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((enter_f[0], enter_f[1], pz - 0.0008), enter_dir, x_cls), steps=6)
            break
    ctx.step(24)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((enter_f[0], enter_f[1], enter_f[2] + 0.090), enter_dir, x_cls), steps=14)

    # Left arm closes workstation IO shutter & presses front half of tower_power (clearing TopCover)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=14)
    sh_dir = (0.0, 0.7880, -0.6157)
    sx0, sy0, sz0 = shutter_f[0] - 0.010, shutter_f[1] + 0.003, shutter_f[2]
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((sx0 - 0.012, sy0 - 0.055, sz0 + 0.030), sh_dir, x_cls), steps=18)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((sx0, sy0, sz0), sh_dir, x_cls), steps=16)
    for k in range(1, 14):
        sx = sx0 + 0.046 * (k / 13.0)
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((sx, sy0, sz0), sh_dir, x_cls), steps=5)
        if ctx.world.joint_value("workstation", "io_shutter") >= -0.006:
            break
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((sx0 + 0.040, sy0 - 0.065, dz + 0.160), sh_dir, x_cls), steps=14)

    tower_dir = (0.0, 0.12, -0.9928)
    ty_t = tower_f[1] - 0.0055
    ctx.end_effector.set_posture("left_gripper", "closed", steps=10)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((tower_f[0], ty_t - 0.025, tower_f[2] + 0.075), tower_dir, x_cls), steps=18)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((tower_f[0], ty_t, tower_f[2] + 0.012), tower_dir, x_cls), steps=16)
    ctx.step(4)
    for k in range(1, 20):
        pz = tower_f[2] + 0.008 - 0.0009 * k
        ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((tower_f[0], ty_t, pz), tower_dir, x_cls), steps=5)
        ctx.step(3)
        if ctx.world.joint_value("workstation", "tower_power") <= -0.0022:
            ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((tower_f[0], ty_t, pz - 0.0008), tower_dir, x_cls), steps=6)
            break
    ctx.step(24)
    ctx.kinematics.move_to_pose("left_arm", _tip_to_tcp((max(tower_f[0], -0.50), ty_t - 0.040, dz + 0.160), tower_dir, x_cls), steps=14)
    ctx.kinematics.move_joints("left_arm", home_l, steps=16)

    # Right arm presses monitor_power and holds terminal contact
    mon_dir = (0.0, 0.65, -0.7600)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mon_f[0], mon_f[1] - 0.055, mon_f[2] + 0.065), mon_dir, x_cls), steps=18)
    ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mon_f[0], mon_f[1] - 0.015, mon_f[2] + 0.025), mon_dir, x_cls), steps=14)
    for k in range(1, 18):
        pz = mon_f[2] + 0.008 - 0.015 * (k / 17.0)
        ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mon_f[0], mon_f[1], pz), mon_dir, x_cls), steps=4)
        ctx.step(3)
        if ctx.world.joint_value("workstation", "monitor_power") <= -0.0022:
            ctx.kinematics.move_to_pose("right_arm", _tip_to_tcp((mon_f[0], mon_f[1], pz - 0.0008), mon_dir, x_cls), steps=6)
            break
    ctx.step(64)
    _ = home_r
