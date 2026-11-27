"""Isaac Sim 6.0.1 native 12-stage Bilibili BV17t4y1V7XH closed-loop director & zero-occlusion rear choreography."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from trajectory_forge.adapters.legacy.distillation_v06.cosserat_hose_solver_v06 import (
    quat_wxyz_rot,
    skin_cosserat_tube,
    solve_cosserat_catenary_centerline,
)
from trajectory_forge.adapters.legacy.distillation_v06.mesh_penetration_gate_v06 import (
    audit_v06_trajectory_geometry,
    panda_all_link_world_positions,
)
from trajectory_forge.sdk.panda_kinematics import (
    PANDA_HOME_Q,
    solve_ik_panda,
)


def _min_jerk(u: float) -> float:
    x = max(0.0, min(1.0, float(u)))
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def _slerp_wxyz(q0: Sequence[float], q1: Sequence[float], u: float) -> np.ndarray:
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    t = _min_jerk(u)
    if dot > 0.9995:
        res = (1.0 - t) * a + t * b
        return res / max(float(np.linalg.norm(res)), 1e-12)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_0
    s1 = math.sin(theta) / sin_0
    res = s0 * a + s1 * b
    return res / max(float(np.linalg.norm(res)), 1e-12)


def _arc_pos(p0: Sequence[float], p1: Sequence[float], u: float, lift_m: float = 0.075) -> np.ndarray:
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    s = _min_jerk(u)
    pos = (1.0 - s) * a + s * b
    if lift_m > 0.0:
        pos[2] += lift_m * math.sin(math.pi * max(0.0, min(1.0, float(u))))
    return pos


def rear_reach_grasp_rot(pitch_down_deg: float = 20.0, yaw_offset_deg: float = 0.0) -> np.ndarray:
    """World 3x3 orientation for a rear-standing (+Y) Franka grasping toward -Y with fingers opening +/-X."""
    p = math.radians(pitch_down_deg)
    cp, sp = math.cos(p), math.sin(p)
    r_base = np.array(
        [
            [0.0, -1.0, 0.0],
            [-sp, 0.0, -cp],
            [cp, 0.0, -sp],
        ],
        dtype=np.float64,
    )
    if abs(yaw_offset_deg) > 1e-6:
        yw = math.radians(yaw_offset_deg)
        cy, sy = math.cos(yw), math.sin(yw)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return rz @ r_base
    return r_base


# Rear-surface (+Y >= +0.022 m) grasp offsets so parallel fingers hold the rear collar and NEVER occlude front (-Y)!
GRASP_SPEC_BY_ID: dict[str, tuple[tuple[float, float, float], float]] = {
    "iron_stand_left": ((-0.015, 0.022, 0.240), 0.006),
    "iron_stand_right": ((0.000, 0.022, 0.240), 0.006),
    "scissor_jack_blue": ((0.000, 0.068, 0.045), 0.009),
    "scissor_jack_white": ((0.000, 0.064, 0.045), 0.009),
    "bosshead_left": ((0.000, 0.024, 0.002), 0.012),
    "bosshead_right": ((0.000, 0.024, 0.002), 0.012),
    "flask_clamp_2prong": ((-0.045, 0.048, 0.000), 0.006),
    "round_bottom_flask_250ml": ((0.000, 0.020, 0.060), 0.014),
    "distillation_head_75deg": ((0.000, 0.020, 0.050), 0.013),
    "condenser_clamp_3prong": ((0.010, 0.044, 0.000), 0.008),
    "liebig_condenser_24_29": ((0.000, 0.026, 0.008), 0.016),
    "keck_clip_head_cond": ((0.000, 0.022, 0.014), 0.007),
    "keck_clip_cond_tail": ((0.000, 0.022, 0.014), 0.007),
    "vacuum_tail_adapter": ((0.036, 0.024, -0.008), 0.009),
    "erlenmeyer_flask_150ml": ((0.000, 0.020, 0.088), 0.014),
    "long_stem_funnel": ((0.000, 0.024, 0.038), 0.024),
    "pour_beaker_100ml": ((0.000, 0.026, 0.042), 0.025),
    "screw_thermometer_adapter": ((0.000, 0.020, 0.032), 0.012),
    "mercury_thermometer_300c": ((0.000, 0.018, 0.062), 0.004),
}


def _grasp_world_pos(inst_pos: np.ndarray, inst_quat: np.ndarray, aid: str) -> tuple[np.ndarray, float]:
    off, f_w = GRASP_SPEC_BY_ID.get(aid, ((0.0, 0.022, 0.02), 0.012))
    p = inst_pos + quat_wxyz_rot(inst_quat, off)
    # Guarantee rear gripper target never crosses in front of y = +0.016 m when above bench!
    if p[2] > 0.85 and p[1] < 0.018:
        p[1] = 0.018
    return p, f_w


def synthesize_v06_12stage_closed_loop_plan(
    cad_specs: Mapping[str, Any],
    *,
    fps: int = 30,
    frames_per_stage: int = 60,
) -> dict[str, Any]:
    """Synthesize smooth 30fps rear-bimanual 12-stage trajectory + zero-penetration C2 Cosserat latex hoses."""
    layout = cad_specs["layout"]
    comps = cad_specs["components"]
    stages = cad_specs["stages"]
    n_stages = len(stages)
    assert n_stages == 12
    T = n_stages * frames_per_stage  # 720 continuous 30fps frames (24.0s uncut!)

    inst_names = list(comps.keys())
    n_inst = len(inst_names)
    idx_map = {k: i for i, k in enumerate(inst_names)}

    inst_poses = np.zeros((T, n_inst, 7), dtype=np.float64)
    for aid, i in idx_map.items():
        inst_poses[:, i, :3] = np.asarray(comps[aid]["staged_xyz"], dtype=np.float64)
        inst_poses[:, i, 3:7] = np.asarray(comps[aid]["staged_wxyz"], dtype=np.float64)

    def _move_inst(
        aid: str,
        f0: int,
        f1: int,
        dst_xyz: Sequence[float],
        dst_wxyz: Sequence[float],
        *,
        lift_m: float = 0.075,
    ) -> None:
        ii = idx_map[aid]
        p0 = inst_poses[f0, ii, :3].copy()
        q0 = inst_poses[f0, ii, 3:7].copy()
        p1 = np.asarray(dst_xyz, dtype=np.float64)
        q1 = np.asarray(dst_wxyz, dtype=np.float64)
        span = max(1, f1 - f0)
        for f in range(f0, f1 + 1):
            u = (f - f0) / float(span)
            inst_poses[f, ii, :3] = _arc_pos(p0, p1, u, lift_m=lift_m)
            inst_poses[f, ii, 3:7] = _slerp_wxyz(q0, q1, u)
        inst_poses[f1 + 1 :, ii, :3] = p1
        inst_poses[f1 + 1 :, ii, 3:7] = q1

    # Wide rear attentive standby poses (clear of central glassware backdrop & macro cameras!)
    home_L = np.array([-0.370, 0.355, 0.965], dtype=np.float64)
    home_R = np.array([+0.370, 0.355, 0.965], dtype=np.float64)
    ee_L = np.zeros((T, 3), dtype=np.float64)
    ee_R = np.zeros((T, 3), dtype=np.float64)
    for f in range(T):
        # Gentle natural standby micro-posture breathing so neither arm is ever robotically frozen across stages!
        ph = 2.0 * math.pi * f / 120.0
        ee_L[f] = home_L + np.array([0.012 * math.sin(ph), 0.008 * math.cos(ph), 0.010 * math.sin(0.7 * ph)])
        ee_R[f] = home_R + np.array([-0.012 * math.cos(ph), 0.008 * math.sin(ph), 0.010 * math.cos(0.7 * ph)])

    rot_L = np.repeat(rear_reach_grasp_rot(20.0)[None, :, :], T, axis=0)
    rot_R = np.repeat(rear_reach_grasp_rot(20.0)[None, :, :], T, axis=0)
    grip_L = np.full((T,), 0.038, dtype=np.float64)
    grip_R = np.full((T,), 0.038, dtype=np.float64)

    def _plan_arm_pick_place(
        is_left: bool,
        aid: str,
        f_start: int,
        f_end: int,
        dst_xyz: Sequence[float],
        dst_wxyz: Sequence[float],
        *,
        lift_m: float = 0.075,
        pitch_deg: float = 20.0,
    ) -> int:
        """Execute smooth approach -> grasp -> carry -> release -> rear-retract, returning exact f_place frame."""
        ee_arr = ee_L if is_left else ee_R
        rot_arr = rot_L if is_left else rot_R
        g_arr = grip_L if is_left else grip_R
        home_p = home_L if is_left else home_R
        r_grasp = rear_reach_grasp_rot(pitch_deg)

        f_grasp = f_start + max(2, int(0.24 * (f_end - f_start)))
        f_place = f_start + max(f_grasp + 2 - f_start, int(0.76 * (f_end - f_start)))
        _move_inst(aid, f_grasp, f_place, dst_xyz, dst_wxyz, lift_m=lift_m)

        ii = idx_map[aid]
        p_pick, fw = _grasp_world_pos(inst_poses[f_start, ii, :3], inst_poses[f_start, ii, 3:7], aid)
        p_pre = p_pick + np.array([0.0, 0.085, 0.045], dtype=np.float64)
        p_start_arm = ee_arr[max(0, f_start - 1)].copy()

        for f in range(f_start, f_grasp):
            u = (f - f_start) / float(max(1, f_grasp - f_start))
            if u < 0.62:
                ee_arr[f] = _arc_pos(p_start_arm, p_pre, u / 0.62, lift_m=0.02)
                g_arr[f] = 0.038
            else:
                ee_arr[f] = _arc_pos(p_pre, p_pick, (u - 0.62) / 0.38, lift_m=0.0)
                g_arr[f] = 0.038 + (fw - 0.038) * _min_jerk((u - 0.62) / 0.38)
            rot_arr[f] = r_grasp

        for f in range(f_grasp, f_place + 1):
            p_cur, _ = _grasp_world_pos(inst_poses[f, ii, :3], inst_poses[f, ii, 3:7], aid)
            ee_arr[f] = p_cur
            rot_arr[f] = r_grasp
            g_arr[f] = fw

        p_rel = ee_arr[f_place].copy()
        p_ret = p_rel + np.array([0.0, 0.095, 0.045], dtype=np.float64)
        p_end_home = ee_arr[min(T - 1, f_end + 1)].copy() if f_end + 1 < T else home_p
        for f in range(f_place + 1, min(T, f_end + 1)):
            u = (f - f_place) / float(max(1, f_end - f_place))
            g_arr[f] = fw + (0.038 - fw) * _min_jerk(min(1.0, u * 2.2))
            if u < 0.45:
                ee_arr[f] = _arc_pos(p_rel, p_ret, u / 0.45, lift_m=0.0)
            else:
                ee_arr[f] = _arc_pos(p_ret, p_end_home, (u - 0.45) / 0.55, lift_m=0.02)
            rot_arr[f] = r_grasp
        return f_place

    jack_blue_lift = np.full((T,), float(comps["scissor_jack_blue"]["lift_staged_m"]), dtype=np.float64)
    jack_white_lift = np.full((T,), float(comps["scissor_jack_white"]["lift_staged_m"]), dtype=np.float64)
    flask_purple_fill = np.zeros((T,), dtype=np.float64)
    receiver_clear_fill = np.zeros((T,), dtype=np.float64)
    pour_active = np.zeros((T,), dtype=np.float64)
    bulb_drop_vis = np.zeros((T,), dtype=np.float64)
    distill_drip_active = np.zeros((T,), dtype=np.float64)
    stage_rep_frames: dict[int, int] = {}

    # --- S01 (0..59): Iron stands + Raise Blue Scissor Jack & Heating Water Bath ---
    _plan_arm_pick_place(True, "iron_stand_left", 2, 30, comps["iron_stand_left"]["assembled_xyz"], comps["iron_stand_left"]["assembled_wxyz"], lift_m=0.025)
    _plan_arm_pick_place(False, "iron_stand_right", 4, 46, comps["iron_stand_right"]["assembled_xyz"], comps["iron_stand_right"]["assembled_wxyz"], lift_m=0.025)
    _move_inst("heating_stirrer_bath", 34, 54, comps["heating_stirrer_bath"]["assembled_xyz"], comps["heating_stirrer_bath"]["assembled_wxyz"], lift_m=0.0)
    for f in range(34, 55):
        jack_blue_lift[f] = 0.048 + (0.070 - 0.048) * _min_jerk((f - 34) / 20.0)
    jack_blue_lift[55:] = 0.070
    for f in range(32, 58):
        u = math.sin(math.pi * (f - 32) / 26.0)
        ee_L[f] = (1.0 - u) * home_L + u * np.array([-0.180, 0.088, 0.815], dtype=np.float64)
        grip_L[f] = 0.012
    stage_rep_frames[0] = 44  # Both arms actively calibrating stand & blue scissor jack!

    # --- S02 (60..119): Left Bosshead (Open Upward!) + 2-Prong Flask Clamp ---
    _plan_arm_pick_place(True, "bosshead_left", 61, 88, comps["bosshead_left"]["assembled_xyz"], comps["bosshead_left"]["assembled_wxyz"], lift_m=0.08)
    fp_s02 = _plan_arm_pick_place(True, "flask_clamp_2prong", 89, 115, comps["flask_clamp_2prong"]["assembled_xyz"], comps["flask_clamp_2prong"]["assembled_wxyz"], lift_m=0.08)
    stage_rep_frames[1] = fp_s02  # Gripper actively seating 2-prong clamp into open-upward bosshead!

    # --- S03 (120..179): Round-Bottom Flask Clamped & Immersed into Clear Glass Water Bath ---
    fp_s03 = _plan_arm_pick_place(True, "round_bottom_flask_250ml", 121, 175, comps["round_bottom_flask_250ml"]["assembled_xyz"], comps["round_bottom_flask_250ml"]["assembled_wxyz"], lift_m=0.135)
    for f in range(140, 175):
        u = math.sin(math.pi * (f - 140) / 35.0)
        ee_R[f] = (1.0 - u) * home_R + u * np.array([-0.145, 0.065, 0.995], dtype=np.float64)
        grip_R[f] = 0.010
    stage_rep_frames[2] = fp_s03  # Left arm holding flask neck in bath while Right arm tightens clamp!

    # --- S04 (180..239): 75-Deg Distillation Head 24/29 Coaxial Mate ---
    fp_s04 = _plan_arm_pick_place(True, "distillation_head_75deg", 181, 235, comps["distillation_head_75deg"]["assembled_xyz"], comps["distillation_head_75deg"]["assembled_wxyz"], lift_m=0.120)
    stage_rep_frames[3] = fp_s04

    # --- S05 (240..299): Right Bosshead (Open Upward!) + 3-Prong Clamp (Fixed Jaws Below, Movable Above) ---
    _plan_arm_pick_place(False, "bosshead_right", 241, 268, comps["bosshead_right"]["assembled_xyz"], comps["bosshead_right"]["assembled_wxyz"], lift_m=0.09)
    fp_s05 = _plan_arm_pick_place(False, "condenser_clamp_3prong", 269, 297, comps["condenser_clamp_3prong"]["assembled_xyz"], comps["condenser_clamp_3prong"]["assembled_wxyz"], lift_m=0.09)
    # Left arm steadies left distillation column from behind during S05
    for f in range(250, 295):
        u = math.sin(math.pi * (f - 250) / 45.0)
        ee_L[f] = (1.0 - u) * home_L + u * np.array([-0.180, 0.055, 1.055], dtype=np.float64)
        grip_L[f] = 0.016
    stage_rep_frames[4] = fp_s05

    # --- S06 (300..359): Bimanual Liebig Condenser Mate + Green 24/29 Keck Clip + Live Swinging Amber Latex Hoses ---
    fp_s06 = _plan_arm_pick_place(False, "liebig_condenser_24_29", 301, 338, comps["liebig_condenser_24_29"]["assembled_xyz"], comps["liebig_condenser_24_29"]["assembled_wxyz"], lift_m=0.095)
    for f in range(304, 342):
        u = math.sin(math.pi * (f - 304) / 38.0)
        ee_L[f] = (1.0 - u) * home_L + u * np.array([-0.145, 0.052, 1.072], dtype=np.float64)
        grip_L[f] = 0.015
    _plan_arm_pick_place(False, "keck_clip_head_cond", 339, 357, comps["keck_clip_head_cond"]["assembled_xyz"], comps["keck_clip_head_cond"]["assembled_wxyz"], lift_m=0.08)
    stage_rep_frames[5] = fp_s06  # Bimanual rear-hold coaxial condenser mating (0% front occlusion!)

    # --- S07 (360..419): Vented Vacuum Tail Adapter + Keck Clip + White Scissor Jack Raising Erlenmeyer Receiver ---
    _plan_arm_pick_place(False, "vacuum_tail_adapter", 361, 381, comps["vacuum_tail_adapter"]["assembled_xyz"], comps["vacuum_tail_adapter"]["assembled_wxyz"], lift_m=0.085)
    _plan_arm_pick_place(False, "keck_clip_cond_tail", 382, 396, comps["keck_clip_cond_tail"]["assembled_xyz"], comps["keck_clip_cond_tail"]["assembled_wxyz"], lift_m=0.075)
    erl_low = [comps["erlenmeyer_flask_150ml"]["assembled_xyz"][0], 0.0, 0.750 + 0.048]
    _plan_arm_pick_place(False, "erlenmeyer_flask_150ml", 397, 409, erl_low, [1.0, 0.0, 0.0, 0.0], lift_m=0.065)
    _move_inst("erlenmeyer_flask_150ml", 410, 418, comps["erlenmeyer_flask_150ml"]["assembled_xyz"], [1.0, 0.0, 0.0, 0.0], lift_m=0.0)
    for f in range(410, 419):
        jack_white_lift[f] = 0.048 + (0.082 - 0.048) * _min_jerk((f - 410) / 8.0)
        ee_R[f] = np.array([0.215, 0.082, 0.815], dtype=np.float64)
        grip_R[f] = 0.010
    jack_white_lift[419:] = 0.082
    stage_rep_frames[6] = 417  # Right arm cranking white scissor jack from rear (+Y=0.082), 0% front occlusion!

    # --- S08 (420..479): Left Arm Holds Long-Stem Funnel + Right Arm Tilts Beaker Pouring Vivid Purple KMnO4! ---
    # Move funnel into distillation head (422..434), hold steady during pour (434..464), then return (465..479)!
    _move_inst("long_stem_funnel", 422, 434, comps["long_stem_funnel"]["pour_xyz"], [1.0, 0.0, 0.0, 0.0], lift_m=0.095)
    _move_inst("long_stem_funnel", 465, 479, comps["long_stem_funnel"]["staged_xyz"], comps["long_stem_funnel"]["staged_wxyz"], lift_m=0.095)
    for f in range(421, 479):
        p_fn, fw_fn = _grasp_world_pos(inst_poses[f, idx_map["long_stem_funnel"], :3], inst_poses[f, idx_map["long_stem_funnel"], 3:7], "long_stem_funnel")
        ee_L[f] = p_fn
        grip_L[f] = fw_fn

    tilt_rad = math.radians(-52.0)  # Tilts beaker spout (-X) down directly over funnel cone!
    q_tilt = [math.cos(0.5 * tilt_rad), 0.0, math.sin(0.5 * tilt_rad), 0.0]
    _move_inst("pour_beaker_100ml", 432, 443, comps["pour_beaker_100ml"]["pour_xyz"], q_tilt, lift_m=0.085)
    for f in range(442, 463):
        pour_active[f] = 1.0
        flask_purple_fill[f] = (f - 441) / 21.0
    flask_purple_fill[463:] = 1.0
    _move_inst("pour_beaker_100ml", 463, 477, comps["pour_beaker_100ml"]["staged_xyz"], comps["pour_beaker_100ml"]["staged_wxyz"], lift_m=0.075)
    for f in range(431, 478):
        p_bkr, fw_b = _grasp_world_pos(inst_poses[f, idx_map["pour_beaker_100ml"], :3], inst_poses[f, idx_map["pour_beaker_100ml"], 3:7], "pour_beaker_100ml")
        ee_R[f] = p_bkr
        grip_R[f] = fw_b
    stage_rep_frames[7] = 452  # Both arms actively holding funnel & tilted purple beaker with vivid purple pour stream!

    # --- S09 (480..539): Screw Thermometer Adapter + Calibrate Mercury Bulb Upper Rim 0.0 mm Flush with Sidearm Lip ---
    _plan_arm_pick_place(True, "screw_thermometer_adapter", 481, 504, comps["screw_thermometer_adapter"]["assembled_xyz"], comps["screw_thermometer_adapter"]["assembled_wxyz"], lift_m=0.095)
    thm_pre = [comps["mercury_thermometer_300c"]["assembled_xyz"][0], 0.0, 1.200]
    _plan_arm_pick_place(True, "mercury_thermometer_300c", 505, 524, thm_pre, [1.0, 0.0, 0.0, 0.0], lift_m=0.110)
    _move_inst("mercury_thermometer_300c", 525, 534, comps["mercury_thermometer_300c"]["assembled_xyz"], [1.0, 0.0, 0.0, 0.0], lift_m=0.0)
    for f in range(524, 536):
        p_t, fw_t = _grasp_world_pos(inst_poses[f, idx_map["mercury_thermometer_300c"], :3], inst_poses[f, idx_map["mercury_thermometer_300c"], 3:7], "mercury_thermometer_300c")
        ee_L[f] = p_t
        grip_L[f] = fw_t
    ee_L[536:540] = home_L
    stage_rep_frames[8] = 533  # Left arm visibly calibrating thermometer stem to exact flush height!

    # --- S10 (540..599): Active Distillation — Bottom-In/Top-Out Cooling, Mercury Bulb Condensate Drop & Distillate Dripping ---
    bulb_drop_vis[541:602] = 1.0
    for f in range(542, 599):
        distill_drip_active[f] = 1.0
        receiver_clear_fill[f] = (f - 541) / 57.0
    receiver_clear_fill[599:] = 1.0
    stage_rep_frames[9] = 572

    # --- S11 (600..659): Strict Reverse Disassembly Right-to-Left (Lower White Jack -> Receiver -> Tail -> Condenser ALL removed by 657!) ---
    _move_inst("erlenmeyer_flask_150ml", 601, 608, erl_low, [1.0, 0.0, 0.0, 0.0], lift_m=0.0)
    for f in range(601, 609):
        jack_white_lift[f] = 0.082 + (0.048 - 0.082) * _min_jerk((f - 601) / 7.0)
    jack_white_lift[609:] = 0.048
    _plan_arm_pick_place(False, "erlenmeyer_flask_150ml", 609, 620, comps["erlenmeyer_flask_150ml"]["staged_xyz"], comps["erlenmeyer_flask_150ml"]["staged_wxyz"], lift_m=0.055)
    _plan_arm_pick_place(False, "keck_clip_cond_tail", 621, 629, comps["keck_clip_cond_tail"]["staged_xyz"], comps["keck_clip_cond_tail"]["staged_wxyz"], lift_m=0.065)
    _plan_arm_pick_place(False, "vacuum_tail_adapter", 630, 640, comps["vacuum_tail_adapter"]["staged_xyz"], comps["vacuum_tail_adapter"]["staged_wxyz"], lift_m=0.075)
    _plan_arm_pick_place(False, "keck_clip_head_cond", 641, 647, comps["keck_clip_head_cond"]["staged_xyz"], comps["keck_clip_head_cond"]["staged_wxyz"], lift_m=0.065)
    _plan_arm_pick_place(False, "liebig_condenser_24_29", 648, 658, comps["liebig_condenser_24_29"]["staged_xyz"], comps["liebig_condenser_24_29"]["staged_wxyz"], lift_m=0.085)
    stage_rep_frames[10] = 659  # Receiver, tail adapter, Keck clips, AND condenser ALL dismounted back to bench!

    # --- S12 (660..719): Strict Top-to-Bottom Disassembly Left Column (Thermometer -> Adapter -> Head -> Flask ALL removed by 718!) ---
    _plan_arm_pick_place(True, "mercury_thermometer_300c", 661, 673, comps["mercury_thermometer_300c"]["staged_xyz"], comps["mercury_thermometer_300c"]["staged_wxyz"], lift_m=0.110)
    _plan_arm_pick_place(True, "screw_thermometer_adapter", 674, 684, comps["screw_thermometer_adapter"]["staged_xyz"], comps["screw_thermometer_adapter"]["staged_wxyz"], lift_m=0.085)
    _plan_arm_pick_place(True, "distillation_head_75deg", 685, 698, comps["distillation_head_75deg"]["staged_xyz"], comps["distillation_head_75deg"]["staged_wxyz"], lift_m=0.105)
    _move_inst("heating_stirrer_bath", 699, 704, comps["heating_stirrer_bath"]["staged_xyz"], comps["heating_stirrer_bath"]["staged_wxyz"], lift_m=0.0)
    for f in range(699, 705):
        jack_blue_lift[f] = 0.070 + (0.048 - 0.070) * _min_jerk((f - 699) / 5.0)
    jack_blue_lift[705:] = 0.048
    _plan_arm_pick_place(True, "round_bottom_flask_250ml", 704, 717, comps["round_bottom_flask_250ml"]["staged_xyz"], comps["round_bottom_flask_250ml"]["staged_wxyz"], lift_m=0.125)
    stage_rep_frames[11] = 719  # Thermometer, adapter, 75-deg head, AND round-bottom flask ALL dismounted to bench!

    # Solve warm-started 7-DOF IK for both Rear-Standing Franka Panda arms
    b_left = np.asarray(layout["rear_robot_bases"]["franka_left"]["xyz"], dtype=np.float64)
    q_left_b = np.asarray(layout["rear_robot_bases"]["franka_left"]["wxyz"], dtype=np.float64)
    b_right = np.asarray(layout["rear_robot_bases"]["franka_right"]["xyz"], dtype=np.float64)
    q_right_b = np.asarray(layout["rear_robot_bases"]["franka_right"]["wxyz"], dtype=np.float64)

    qpos_seq = np.zeros((T, 18), dtype=np.float64)
    q_cur_L = solve_ik_panda(ee_L[0], rot_L[0], q_init=PANDA_HOME_Q, base_pos=b_left, base_quat_wxyz=q_left_b, max_iters=140)
    q_cur_R = solve_ik_panda(ee_R[0], rot_R[0], q_init=PANDA_HOME_Q, base_pos=b_right, base_quat_wxyz=q_right_b, max_iters=140)

    for f in range(T):
        q_cur_L = solve_ik_panda(ee_L[f], rot_L[f], q_init=q_cur_L, base_pos=b_left, base_quat_wxyz=q_left_b, max_iters=45)
        q_cur_R = solve_ik_panda(ee_R[f], rot_R[f], q_init=q_cur_R, base_pos=b_right, base_quat_wxyz=q_right_b, max_iters=45)
        qpos_seq[f, 0:7] = q_cur_L
        qpos_seq[f, 7] = grip_L[f]
        qpos_seq[f, 8] = grip_L[f]
        qpos_seq[f, 9:16] = q_cur_R
        qpos_seq[f, 16] = grip_R[f]
        qpos_seq[f, 17] = grip_R[f]

    # Solve C2-smooth zero-penetration 64-ring Cosserat catenaries sleeving the outer 22 mm of the 38 mm glass barbs!
    tap_noz = np.asarray(layout["water_tap_nozzle_world"], dtype=np.float64)
    drain_noz = np.asarray(layout["sink_drain_nozzle_world"], dtype=np.float64)
    barb_dir_local = np.array([0.0, -0.469472, -0.882948], dtype=np.float64)
    in_sleeve_local = np.array([+0.074, 0.016 * barb_dir_local[1], -0.014 + 0.016 * barb_dir_local[2]], dtype=np.float64)
    out_sleeve_local = np.array([-0.074, 0.016 * barb_dir_local[1], -0.014 + 0.016 * barb_dir_local[2]], dtype=np.float64)

    hose_in_c = np.zeros((T, 64, 3), dtype=np.float64)
    hose_out_c = np.zeros((T, 64, 3), dtype=np.float64)
    hose_in_v = np.zeros((T, 64 * 24, 3), dtype=np.float32)
    hose_in_n = np.zeros((T, 64 * 24, 3), dtype=np.float32)
    hose_out_v = np.zeros((T, 64 * 24, 3), dtype=np.float32)
    hose_out_n = np.zeros((T, 64 * 24, 3), dtype=np.float32)

    idx_cond = idx_map["liebig_condenser_24_29"]
    idx_bath = idx_map["heating_stirrer_bath"]
    idx_wjack = idx_map["scissor_jack_white"]
    prev_in, prev_out = None, None

    for f in range(T):
        c_pos = inst_poses[f, idx_cond, :3]
        c_quat = inst_poses[f, idx_cond, 3:7]
        p_in_start = c_pos + quat_wxyz_rot(c_quat, in_sleeve_local)
        p_out_start = c_pos + quat_wxyz_rot(c_quat, out_sleeve_local)
        d_barb_w = quat_wxyz_rot(c_quat, barb_dir_local)

        bath_xy = inst_poses[f, idx_bath, :2]
        bath_zmax = float(inst_poses[f, idx_bath, 2]) + 0.132
        wj_xy = inst_poses[f, idx_wjack, :2]
        wj_zmax = float(inst_poses[f, idx_wjack, 2]) + 0.095
        cyl_obs = [
            (float(bath_xy[0]), float(bath_xy[1]), 0.082, bath_zmax),
            (float(wj_xy[0]), float(wj_xy[1]), 0.074, wj_zmax),
            (float(inst_poses[f, idx_map["round_bottom_flask_250ml"], 0]), float(inst_poses[f, idx_map["round_bottom_flask_250ml"], 1]), 0.050, float(inst_poses[f, idx_map["round_bottom_flask_250ml"], 2]) + 0.10),
            (float(inst_poses[f, idx_map["vacuum_tail_adapter"], 0]), float(inst_poses[f, idx_map["vacuum_tail_adapter"], 1]), 0.040, float(inst_poses[f, idx_map["vacuum_tail_adapter"], 2]) + 0.06),
            (float(inst_poses[f, idx_map["erlenmeyer_flask_150ml"], 0]), float(inst_poses[f, idx_map["erlenmeyer_flask_150ml"], 1]), 0.042, float(inst_poses[f, idx_map["erlenmeyer_flask_150ml"], 2]) + 0.11),
            (float(inst_poses[f, idx_map["pour_beaker_100ml"], 0]), float(inst_poses[f, idx_map["pour_beaker_100ml"], 1]), 0.036, float(inst_poses[f, idx_map["pour_beaker_100ml"], 2]) + 0.08),
        ]
        arm_links = np.concatenate(
            [
                panda_all_link_world_positions(qpos_seq[f, :9], b_left, q_left_b),
                panda_all_link_world_positions(qpos_seq[f, 9:18], b_right, q_right_b),
            ],
            axis=0,
        )
        sph_obs = [(float(p[0]), float(p[1]), float(p[2]), 0.040) for p in arm_links]

        prev_in = solve_cosserat_catenary_centerline(
            p_in_start, d_barb_w, tap_noz, (0.0, 0.0, -1.0),
            rest_length=0.56, cyl_obstacles=cyl_obs, sphere_obstacles=sph_obs, prev_centerline=prev_in,
        )
        prev_out = solve_cosserat_catenary_centerline(
            p_out_start, d_barb_w, drain_noz, (0.0, 0.0, 1.0),
            rest_length=0.60, cyl_obstacles=cyl_obs, sphere_obstacles=sph_obs, prev_centerline=prev_out,
        )
        hose_in_c[f] = prev_in
        hose_out_c[f] = prev_out
        hose_in_v[f], hose_in_n[f] = skin_cosserat_tube(prev_in)
        hose_out_v[f], hose_out_n[f] = skin_cosserat_tube(prev_out)

    stage_index = np.repeat(np.arange(n_stages, dtype=np.int32), frames_per_stage)
    rep_frames_arr = np.array([stage_rep_frames[i] for i in range(n_stages)], dtype=np.int32)
    telemetry = {
        "fps": fps,
        "frames_per_stage": frames_per_stage,
        "qpos": qpos_seq,
        "instrument_names": np.array(inst_names),
        "instrument_poses": inst_poses,
        "hose_inlet_centerline": hose_in_c,
        "hose_outlet_centerline": hose_out_c,
        "hose_inlet_verts": hose_in_v,
        "hose_inlet_norms": hose_in_n,
        "hose_outlet_verts": hose_out_v,
        "hose_outlet_norms": hose_out_n,
        "jack_blue_lift": jack_blue_lift,
        "jack_white_lift": jack_white_lift,
        "flask_purple_fill": flask_purple_fill,
        "receiver_clear_fill": receiver_clear_fill,
        "pour_active": pour_active,
        "bulb_drop_vis": bulb_drop_vis,
        "distill_drip_active": distill_drip_active,
        "stage_index": stage_index,
        "stage_rep_frames": rep_frames_arr,
    }
    audit = audit_v06_trajectory_geometry(telemetry, cad_specs)
    return {"telemetry": telemetry, "audit": audit}
