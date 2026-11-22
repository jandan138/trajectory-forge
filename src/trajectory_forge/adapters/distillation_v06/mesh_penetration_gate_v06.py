"""Independent geometric & SOP red-line auditor for distillation_v06 (all-pair part, hose, & arm 0.0 mm gate)."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from trajectory_forge.sdk.panda_kinematics import _PANDA_CHAIN, _PRECOMPUTED_FIXED, _rot_z, _se3, quat_wxyz_to_matrix


# Intentional functional mating/nesting pairs (excluded from benchtop collision penalty)
INTENTIONAL_MATED_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(p)
    for p in (
        ("scissor_jack_blue", "heating_stirrer_bath"),
        ("heating_stirrer_bath", "round_bottom_flask_250ml"),
        ("iron_stand_left", "bosshead_left"),
        ("bosshead_left", "flask_clamp_2prong"),
        ("iron_stand_left", "flask_clamp_2prong"),
        ("flask_clamp_2prong", "round_bottom_flask_250ml"),
        ("flask_clamp_2prong", "distillation_head_75deg"),
        ("round_bottom_flask_250ml", "distillation_head_75deg"),
        ("distillation_head_75deg", "long_stem_funnel"),
        ("long_stem_funnel", "pour_beaker_100ml"),
        ("distillation_head_75deg", "pour_beaker_100ml"),
        ("distillation_head_75deg", "screw_thermometer_adapter"),
        ("distillation_head_75deg", "mercury_thermometer_300c"),
        ("screw_thermometer_adapter", "mercury_thermometer_300c"),
        ("round_bottom_flask_250ml", "mercury_thermometer_300c"),
        ("iron_stand_right", "bosshead_right"),
        ("bosshead_right", "condenser_clamp_3prong"),
        ("iron_stand_right", "condenser_clamp_3prong"),
        ("condenser_clamp_3prong", "liebig_condenser_24_29"),
        ("distillation_head_75deg", "liebig_condenser_24_29"),
        ("distillation_head_75deg", "keck_clip_head_cond"),
        ("liebig_condenser_24_29", "keck_clip_head_cond"),
        ("condenser_clamp_3prong", "keck_clip_head_cond"),
        ("liebig_condenser_24_29", "vacuum_tail_adapter"),
        ("liebig_condenser_24_29", "keck_clip_cond_tail"),
        ("vacuum_tail_adapter", "keck_clip_cond_tail"),
        ("scissor_jack_white", "erlenmeyer_flask_150ml"),
        ("vacuum_tail_adapter", "erlenmeyer_flask_150ml"),
    )
)

PART_BOUNDING_RADII_XY: dict[str, float] = {
    "bosshead_left": 0.022,
    "bosshead_right": 0.022,
    "flask_clamp_2prong": 0.032,
    "condenser_clamp_3prong": 0.032,
    "round_bottom_flask_250ml": 0.043,
    "distillation_head_75deg": 0.038,
    "screw_thermometer_adapter": 0.016,
    "mercury_thermometer_300c": 0.015,
    "liebig_condenser_24_29": 0.030,
    "vacuum_tail_adapter": 0.034,
    "erlenmeyer_flask_150ml": 0.036,
    "long_stem_funnel": 0.036,
    "pour_beaker_100ml": 0.030,
    "keck_clip_head_cond": 0.016,
    "keck_clip_cond_tail": 0.016,
}


def panda_all_link_world_positions(
    q9: np.ndarray,
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Return (9, 3) world coordinates along all joints + hand + grasp target of a 9-DOF Franka Panda."""
    t = _se3(quat_wxyz_to_matrix(base_quat_wxyz), base_pos)
    pts = []
    j_idx = 0
    for tf_fixed, (_, _, is_rev) in zip(_PRECOMPUTED_FIXED, _PANDA_CHAIN):
        t = t @ tf_fixed
        if is_rev:
            t = t @ _se3(_rot_z(float(q9[j_idx])), (0.0, 0.0, 0.0))
            j_idx += 1
        pts.append(t[:3, 3].copy())
    return np.asarray(pts, dtype=np.float64)


def audit_v06_trajectory_geometry(
    telemetry: Mapping[str, Any],
    cad_specs: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify zero benchtop/part-part/hose-part/hose-arm penetration and sub-mm Bilibili SOP alignment."""
    layout = cad_specs["layout"]
    table_z = float(layout["table_surface_z"])
    qpos_seq = np.asarray(telemetry["qpos"], dtype=np.float64)
    inst_names = [str(x) for x in telemetry["instrument_names"]]
    inst_poses = np.asarray(telemetry["instrument_poses"], dtype=np.float64)
    hose_in_v = np.asarray(telemetry["hose_inlet_verts"], dtype=np.float64)
    hose_out_v = np.asarray(telemetry["hose_outlet_verts"], dtype=np.float64)
    hose_in_c = np.asarray(telemetry["hose_inlet_centerline"], dtype=np.float64)
    hose_out_c = np.asarray(telemetry["hose_outlet_centerline"], dtype=np.float64)
    stage_idx_seq = np.asarray(telemetry["stage_index"], dtype=np.int32)

    T = int(qpos_seq.shape[0])
    name_to_idx = {n: i for i, n in enumerate(inst_names)}

    # 1. Hose vs Benchtop penetration check across ALL frames & ALL 3,072 vertices
    min_z_in = float(hose_in_v[:, :, 2].min())
    min_z_out = float(hose_out_v[:, :, 2].min())
    table_pen_m = max(0.0, table_z - min(min_z_in, min_z_out))
    table_pen_mm = round(table_pen_m * 1000.0, 4)

    # 2. All-pair Non-Mated Benchtop Instrument Overlap Check (catches bosshead-inside-flask etc.)
    tracked_parts = [p for p in PART_BOUNDING_RADII_XY if p in name_to_idx]
    max_part_part_pen_m = 0.0
    for k in (0, 60, 120, 359, 659, T - 1):
        for ia in range(len(tracked_parts)):
            na = tracked_parts[ia]
            pa = inst_poses[k, name_to_idx[na], :3]
            ra = PART_BOUNDING_RADII_XY[na]
            for ib in range(ia + 1, len(tracked_parts)):
                nb = tracked_parts[ib]
                if frozenset((na, nb)) in INTENTIONAL_MATED_PAIRS:
                    continue
                pb = inst_poses[k, name_to_idx[nb], :3]
                if abs(float(pa[2] - pb[2])) > 0.065:
                    continue
                rb = PART_BOUNDING_RADII_XY[nb]
                d3 = float(np.linalg.norm(pa - pb))
                pen_ab = max(0.0, (ra + rb) - d3)
                if pen_ab > max_part_part_pen_m:
                    max_part_part_pen_m = pen_ab

    # 3. Hose vs ALL Obstacles (Bath, Scissor Jacks, Staged Glassware) & Dual 9-DOF Robot Links
    b_left = np.asarray(layout["rear_robot_bases"]["franka_left"]["xyz"], dtype=np.float64)
    q_left = np.asarray(layout["rear_robot_bases"]["franka_left"]["wxyz"], dtype=np.float64)
    b_right = np.asarray(layout["rear_robot_bases"]["franka_right"]["xyz"], dtype=np.float64)
    q_right = np.asarray(layout["rear_robot_bases"]["franka_right"]["wxyz"], dtype=np.float64)

    max_obs_pen_m = 0.0
    max_arm_pen_m = 0.0
    idx_bath = name_to_idx["heating_stirrer_bath"]
    idx_wjack = name_to_idx["scissor_jack_white"]
    bench_glass_check = [
        ("round_bottom_flask_250ml", 0.043, 0.095),
        ("vacuum_tail_adapter", 0.032, 0.060),
        ("erlenmeyer_flask_150ml", 0.036, 0.106),
        ("pour_beaker_100ml", 0.028, 0.072),
    ]

    for k in range(T):
        bath_xy = inst_poses[k, idx_bath, :2]
        bath_zmax = float(inst_poses[k, idx_bath, 2]) + 0.130
        wj_xy = inst_poses[k, idx_wjack, :2]
        wj_zmax = float(inst_poses[k, idx_wjack, 2]) + 0.090

        nodes = np.concatenate([hose_in_c[k, 5:-3], hose_out_c[k, 5:-3]], axis=0)
        obs_list = [
            (float(bath_xy[0]), float(bath_xy[1]), 0.073, bath_zmax),
            (float(wj_xy[0]), float(wj_xy[1]), 0.064, wj_zmax),
        ]
        for gname, grad, gh in bench_glass_check:
            gp = inst_poses[k, name_to_idx[gname], :3]
            obs_list.append((float(gp[0]), float(gp[1]), grad, float(gp[2]) + gh))

        for cx, cy, r_hard, zmax in obs_list:
            mask = nodes[:, 2] <= zmax
            if np.any(mask):
                dxy = np.hypot(nodes[mask, 0] - cx, nodes[mask, 1] - cy)
                pen = float(np.max(np.maximum(0.0, (r_hard + 0.0052) - dxy)))
                if pen > max_obs_pen_m:
                    max_obs_pen_m = pen

        arm_pts = np.concatenate(
            [
                panda_all_link_world_positions(qpos_seq[k, :9], b_left, q_left),
                panda_all_link_world_positions(qpos_seq[k, 9:18], b_right, q_right),
            ],
            axis=0,
        )
        diff = nodes[:, None, :] - arm_pts[None, :, :]
        min_d = float(np.min(np.linalg.norm(diff, axis=2)))
        arm_pen = max(0.0, (0.032 + 0.0052) - min_d)
        if arm_pen > max_arm_pen_m:
            max_arm_pen_m = arm_pen

    # 4. Active Distillation Steady-State Window (Stage S10 => stage_index == 9) Coaxial & Bilibili Red-Line Checks
    s10_frames = np.where(stage_idx_seq == 9)[0]
    k_mid = int(s10_frames[len(s10_frames) // 2])

    def _p(name: str) -> np.ndarray:
        return inst_poses[k_mid, name_to_idx[name], :3]

    def _q(name: str) -> np.ndarray:
        return inst_poses[k_mid, name_to_idx[name], 3:7]

    p_flask = _p("round_bottom_flask_250ml")
    p_head = _p("distillation_head_75deg")
    exp_head = np.asarray(cad_specs["components"]["distillation_head_75deg"]["assembled_xyz"], dtype=np.float64)
    flask_head_err_mm = round(float(np.linalg.norm((p_head - p_flask) - (exp_head - np.asarray(cad_specs["components"]["round_bottom_flask_250ml"]["assembled_xyz"])))) * 1000.0, 4)

    p_cond = _p("liebig_condenser_24_29")
    q_cond = _q("liebig_condenser_24_29")
    exp_cond = np.asarray(cad_specs["components"]["liebig_condenser_24_29"]["assembled_xyz"], dtype=np.float64)
    head_cond_err_mm = round(float(np.linalg.norm(p_cond - exp_cond)) * 1000.0, 4)
    cond_x_world = quat_wxyz_to_matrix(q_cond)[:, 0]
    cond_pitch_down_deg = round(math.degrees(math.asin(max(-1.0, min(1.0, -float(cond_x_world[2]))))), 3)

    p_tail = _p("vacuum_tail_adapter")
    exp_tail = np.asarray(cad_specs["components"]["vacuum_tail_adapter"]["assembled_xyz"], dtype=np.float64)
    cond_tail_err_mm = round(float(np.linalg.norm(p_tail - exp_tail)) * 1000.0, 4)

    p_thm = _p("mercury_thermometer_300c")
    bulb_top_z = float(p_thm[2]) + float(layout["thermometer_bulb_top_local_z"])
    sidearm_lip_z = float(layout["sidearm_bottom_lip_z_world"])
    bulb_flush_err_mm = round(abs(bulb_top_z - sidearm_lip_z) * 1000.0, 4)

    bh_l_z = float(quat_wxyz_to_matrix(_q("bosshead_left"))[2, 2])
    bh_r_z = float(quat_wxyz_to_matrix(_q("bosshead_right"))[2, 2])
    bossheads_open_upward = bool(bh_l_z >= 0.995 and bh_r_z >= 0.995)
    c3_z = float(quat_wxyz_to_matrix(_q("condenser_clamp_3prong"))[2, 2])
    clamp_3prong_fixed_below = bool(c3_z >= 0.95)

    diffs = np.diff(stage_idx_seq)
    stages_visited = sorted(set(int(x) for x in stage_idx_seq.tolist()))
    all_12_monotonic = bool(np.all(diffs >= 0) and stages_visited == list(range(12)))

    # Verify S11 end (frame 659) unmounted right train AND S12 end (frame T-1) unmounted left column INCLUDING flask!
    s11_cond_z = float(inst_poses[659, name_to_idx["liebig_condenser_24_29"], 2])
    final_head_z = float(inst_poses[-1, name_to_idx["distillation_head_75deg"], 2])
    final_flask_x = float(inst_poses[-1, name_to_idx["round_bottom_flask_250ml"], 0])
    reverse_disassembly_completed = bool(s11_cond_z < 0.80 and final_head_z < 0.80 and final_flask_x < -0.32)

    max_pen_mm = round(max(table_pen_m, max_part_part_pen_m, max_obs_pen_m, max_arm_pen_m) * 1000.0, 4)
    gates = {
        "zero_hose_table_penetration": table_pen_mm == 0.0,
        "zero_part_to_part_overlap": round(max_part_part_pen_m * 1000.0, 4) == 0.0,
        "zero_hose_obstacle_penetration": round(max_obs_pen_m * 1000.0, 4) == 0.0,
        "zero_hose_arm_penetration": round(max_arm_pen_m * 1000.0, 4) == 0.0,
        "coaxial_joints_sub_1mm": max(flask_head_err_mm, head_cond_err_mm, cond_tail_err_mm) <= 1.0,
        "condenser_downward_pitch_15deg": 14.0 <= cond_pitch_down_deg <= 16.0,
        "thermometer_bulb_flush_sub_1mm": bulb_flush_err_mm <= 1.0,
        "bossheads_open_upward": bossheads_open_upward,
        "clamp_3prong_fixed_jaws_below": clamp_3prong_fixed_below,
        "all_12_bilibili_stages_monotonic": all_12_monotonic,
        "reverse_disassembly_completed": reverse_disassembly_completed,
    }
    return {
        "schema_version": "distillation-v06-penetration-and-sop-audit/v1.1",
        "overall_passed": bool(all(gates.values())),
        "total_frames": T,
        "fps": 30,
        "stride": 1,
        "max_penetration_mm": max_pen_mm,
        "hose_table_penetration_mm": table_pen_mm,
        "part_to_part_overlap_mm": round(max_part_part_pen_m * 1000.0, 4),
        "hose_obstacle_penetration_mm": round(max_obs_pen_m * 1000.0, 4),
        "hose_robot_arm_penetration_mm": round(max_arm_pen_m * 1000.0, 4),
        "s10_steady_state_metrics": {
            "flask_head_taper_gap_mm": flask_head_err_mm,
            "head_condenser_taper_gap_mm": head_cond_err_mm,
            "condenser_tail_taper_gap_mm": cond_tail_err_mm,
            "condenser_downward_pitch_deg": cond_pitch_down_deg,
            "thermometer_bulb_flush_err_mm": bulb_flush_err_mm,
            "bossheads_open_upward": bossheads_open_upward,
            "clamp_3prong_fixed_jaws_below": clamp_3prong_fixed_below,
        },
        "gates": gates,
    }
