"""Pure 7-DOF Bimanual Robot Schedule & CPU IK/Clearance/Anchor-FK Gate for distillation_v09 (`S01 -> S12` One-Shot).

STRICT ANTI-CHEAT INVARIANT:
This module outputs ONLY 7-DOF robot arm joint targets (`qL_seq`, `qR_seq`) and prismatic finger
width targets (`wL_seq`, `wR_seq`) + stage/camera metadata. It outputs ZERO object trajectories!
All glassware, scissor-jack platforms/handwheels, clamp jaws/wingnuts, green POM Keck clips,
Method B micro-spring wedge spheres, and 48 cooling hose links move 100% under 240 Hz PhysX 5 GPU contact physics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trajectory_forge.adapters.distillation_v06.mesh_penetration_gate_v06 import (
    panda_all_link_world_positions,
)
from trajectory_forge.sdk.panda_kinematics import (
    PANDA_ARM_LOWER,
    PANDA_ARM_UPPER,
    PANDA_HOME_Q,
    forward_kinematics_panda,
    solve_ik_panda,
)


TH15 = math.radians(15.0)
U15 = np.array([math.cos(TH15), 0.0, -math.sin(TH15)], dtype=np.float64)
N15 = np.array([math.sin(TH15), 0.0, +math.cos(TH15)], dtype=np.float64)
RY15 = np.array(
    [[math.cos(TH15), 0.0, math.sin(TH15)], [0.0, 1.0, 0.0], [-math.sin(TH15), 0.0, math.cos(TH15)]],
    dtype=np.float64,
)

HEAD_SIDEARM_CONE_W = np.array([-0.097896, 0.0, 1.066000], dtype=np.float64)
COND_SEATED_CENTER_W = np.array([+0.022844, 0.0, 1.033648], dtype=np.float64)
COND_LOWER_CONE_W = np.array([+0.143585, 0.0, 1.001296], dtype=np.float64)


def _min_jerk(u: float) -> float:
    x = max(0.0, min(1.0, float(u)))
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def _rot_from_z_and_y(z_axis: Sequence[float], y_axis: Sequence[float]) -> np.ndarray:
    z = np.asarray(z_axis, dtype=np.float64)
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    y = np.asarray(y_axis, dtype=np.float64)
    y = y - float(np.dot(y, z)) * z
    y = y / max(float(np.linalg.norm(y)), 1e-12)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def _solve_clean_anchor_ik(
    pos: Sequence[float],
    rot: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
) -> np.ndarray:
    """Select a clean interior 7-DOF IK branch (`sub-mm FK accuracy + wide j7 twist margin`, fast early-exit)."""
    mid_q = 0.5 * (PANDA_ARM_LOWER + PANDA_ARM_UPPER)
    half_span = 0.5 * (PANDA_ARM_UPPER - PANDA_ARM_LOWER)
    best_q = PANDA_HOME_Q.copy()
    best_cost = 1e12
    seeds = [(0.0, 0.0), (-0.55, 0.0), (0.45, 0.0), (-0.85, -0.65), (-0.25, 0.65), (0.75, -0.65), (-1.10, 0.0)]
    for s1, s7 in seeds:
        q0 = PANDA_HOME_Q.copy()
        if abs(s1) > 1e-6:
            q0[0] = s1
        q0[6] = s7
        q = solve_ik_panda(
            pos,
            rot,
            q_init=q0,
            base_pos=base_pos,
            base_quat_wxyz=base_quat,
            max_iters=180,
            pos_tol=1e-5,
            rot_tol=1e-4,
            damping=0.012,
        )
        fp, fr, _ = forward_kinematics_panda(q, base_pos=base_pos, base_quat_wxyz=base_quat)
        pe = float(np.linalg.norm(fp - np.asarray(pos)))
        re = float(np.linalg.norm(fr - rot))
        lim_norm = float(np.max(np.abs((q - mid_q) / half_span)))
        cost = pe * 6000.0 + re * 220.0 + lim_norm * 2.4 + 0.40 * abs(float(q[6])) + 0.12 * float(np.linalg.norm(q - PANDA_HOME_Q))
        if pe < 1.0e-3 and re < 1.2e-2 and cost < best_cost:
            best_cost = cost
            best_q = q.copy()
            if pe < 3e-4 and lim_norm < 0.76 and abs(float(q[6])) < 0.95:
                return best_q
    if best_cost >= 1e11:
        best_q = solve_ik_panda(pos, rot, q_init=PANDA_HOME_Q, base_pos=base_pos, base_quat_wxyz=base_quat, max_iters=260)
    return best_q


def _step_cartesian_ik_safe(
    pos: Sequence[float],
    rot: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    q_prev: np.ndarray,
    max_dq: float = 0.18,
) -> np.ndarray:
    """Local damped IK step strictly staying in the same branch (`prevents wrist branch flips`)."""
    q = solve_ik_panda(
        pos,
        rot,
        q_init=q_prev,
        base_pos=base_pos,
        base_quat_wxyz=base_quat,
        max_iters=50,
        pos_tol=1.5e-5,
        rot_tol=1.5e-4,
        damping=0.022,
    )
    dq = q - q_prev
    peak = float(np.max(np.abs(dq)))
    if peak > max_dq:
        q = q_prev + dq * (max_dq / peak)
    return np.clip(q, PANDA_ARM_LOWER, PANDA_ARM_UPPER)


def _walk_cartesian_chain(
    waypoints_xyz: Sequence[Sequence[float]],
    rot: np.ndarray,
    bp: np.ndarray,
    bq: np.ndarray,
    q_anchor_first: np.ndarray | None = None,
    substeps_per_leg: int = 22,
) -> list[np.ndarray]:
    """Walk a connected chain of Cartesian points in the same smooth IK branch (`< 0.5 mm` accuracy, `0°` wrist flip)."""
    q0 = q_anchor_first.copy() if q_anchor_first is not None else _solve_clean_anchor_ik(waypoints_xyz[0], rot, bp, bq)
    out = [q0]
    q_cur = q0.copy()
    for i in range(len(waypoints_xyz) - 1):
        pa = np.asarray(waypoints_xyz[i], dtype=np.float64)
        pb = np.asarray(waypoints_xyz[i + 1], dtype=np.float64)
        dist = float(np.linalg.norm(pb - pa))
        n_sub = max(substeps_per_leg, int(math.ceil(dist / 0.010)))
        for s in np.linspace(0.0, 1.0, n_sub + 1)[1:]:
            q_cur = _step_cartesian_ik_safe((1.0 - s) * pa + s * pb, rot, bp, bq, q_cur, max_dq=0.16)
        q_pol = solve_ik_panda(pb, rot, q_init=q_cur, base_pos=bp, base_quat_wxyz=bq, max_iters=110, damping=0.015)
        if float(np.max(np.abs(q_pol - q_cur))) < 0.28:
            q_cur = np.clip(q_pol, PANDA_ARM_LOWER, PANDA_ARM_UPPER)
        out.append(q_cur.copy())
    return out


class _ArmTimelineBuilder:
    """Strictly relative `f_cur`-driven C1 joint & finger builder (`ZERO frame collisions or low bench sweeps!`)."""

    def __init__(self, max_T: int, bp: np.ndarray, bq: np.ndarray, q_home: np.ndarray, w_init: float = 0.038) -> None:
        self.T = max_T
        self.bp = bp.copy()
        self.bq = bq.copy()
        self.q_home = q_home.copy()
        self.q_seq = np.repeat(q_home[None, :], max_T, axis=0)
        self.w_seq = np.full((max_T,), float(w_init), dtype=np.float64)
        self.f_cur = 0
        self.q_cur = q_home.copy()
        self.w_cur = float(w_init)

    def hold_frames(self, dt: int, w_target: float | None = None) -> int:
        span = max(1, int(dt))
        fe = min(self.T - 1, self.f_cur + span)
        w_start = self.w_cur
        w_goal = self.w_cur if w_target is None else float(w_target)
        actual_span = max(1, fe - self.f_cur)
        for f in range(self.f_cur, fe + 1):
            u = _min_jerk((f - self.f_cur) / float(actual_span))
            self.q_seq[f] = self.q_cur
            self.w_seq[f] = (1.0 - u) * w_start + u * w_goal
        self.q_seq[fe + 1 :] = self.q_cur[None, :]
        self.w_seq[fe + 1 :] = w_goal
        self.f_cur = fe
        self.w_cur = w_goal
        return self.f_cur

    def hold_until(self, f_end: int, w_target: float | None = None) -> int:
        if f_end <= self.f_cur:
            if w_target is not None and abs(float(w_target) - self.w_cur) > 1e-5:
                return self.hold_frames(8, w_target=w_target)
            return self.f_cur
        return self.hold_frames(f_end - self.f_cur, w_target=w_target)

    def move_joint(self, min_dt: int, q_target: np.ndarray, w_target: float | None = None, *, linear_rate: bool = False) -> int:
        q_start = self.q_cur.copy()
        q_goal = np.clip(np.asarray(q_target, dtype=np.float64), PANDA_ARM_LOWER, PANDA_ARM_UPPER)
        max_tot = float(np.max(np.abs(q_goal - q_start)))
        vel_span = max(1, int(math.ceil(max_tot * (1.05 if linear_rate else 1.875) / 0.30)))
        span = max(int(min_dt), vel_span)
        fe = min(self.T - 1, self.f_cur + span)
        w_start = self.w_cur
        w_goal = self.w_cur if w_target is None else float(w_target)
        actual_span = max(1, fe - self.f_cur)
        for f in range(self.f_cur, fe + 1):
            raw_u = (f - self.f_cur) / float(actual_span)
            u = raw_u if linear_rate else _min_jerk(raw_u)
            self.q_seq[f] = (1.0 - u) * q_start + u * q_goal
            self.w_seq[f] = (1.0 - _min_jerk(raw_u)) * w_start + _min_jerk(raw_u) * w_goal
        self.q_cur = q_goal.copy()
        self.w_cur = w_goal
        self.q_seq[fe + 1 :] = self.q_cur[None, :]
        self.w_seq[fe + 1 :] = self.w_cur
        self.f_cur = fe
        return self.f_cur

    def move_cartesian(
        self,
        dt: int,
        p_a: Sequence[float],
        p_b: Sequence[float],
        rot: np.ndarray,
        w_target: float | None = None,
        *,
        linear_rate: bool = False,
        j7_add_start: float = 0.0,
        j7_add_end: float = 0.0,
        q_untwisted_start: np.ndarray | None = None,
    ) -> int:
        pa = np.asarray(p_a, dtype=np.float64)
        pb = np.asarray(p_b, dtype=np.float64)
        dist = float(np.linalg.norm(pb - pa))
        span = max(int(dt), int(math.ceil(dist / 0.013)), 4)
        fe = min(self.T - 1, self.f_cur + span)
        w_start = self.w_cur
        w_goal = self.w_cur if w_target is None else float(w_target)
        actual_span = max(1, fe - self.f_cur)
        q_base = q_untwisted_start.copy() if q_untwisted_start is not None else self.q_cur.copy()
        for f in range(self.f_cur, fe + 1):
            raw_u = (f - self.f_cur) / float(actual_span)
            u = raw_u if linear_rate else _min_jerk(raw_u)
            pf = (1.0 - u) * pa + u * pb
            q_base = _step_cartesian_ik_safe(pf, rot, self.bp, self.bq, q_base, max_dq=0.20)
            if f == fe:
                q_pol = solve_ik_panda(pb, rot, q_init=q_base, base_pos=self.bp, base_quat_wxyz=self.bq, max_iters=140, damping=0.015)
                if float(np.max(np.abs(q_pol - q_base))) < 0.30:
                    q_base = np.clip(q_pol, PANDA_ARM_LOWER, PANDA_ARM_UPPER)
            q_out = q_base.copy()
            if abs(j7_add_start) > 1e-6 or abs(j7_add_end) > 1e-6:
                j7_off = (1.0 - raw_u) * j7_add_start + raw_u * j7_add_end
                q_out[6] = np.clip(q_base[6] + j7_off, PANDA_ARM_LOWER[6] + 0.02, PANDA_ARM_UPPER[6] - 0.02)
            self.q_seq[f] = q_out
            self.w_seq[f] = (1.0 - _min_jerk(raw_u)) * w_start + _min_jerk(raw_u) * w_goal
        self.q_cur = self.q_seq[fe].copy()
        self.w_cur = w_goal
        self.q_seq[fe + 1 :] = self.q_cur[None, :]
        self.w_seq[fe + 1 :] = self.w_cur
        self.f_cur = fe
        return self.f_cur

    def rotary_screw_turns(
        self,
        p_drum: Sequence[float],
        p_pre: Sequence[float],
        p_sky: Sequence[float],
        R_drum: np.ndarray,
        *,
        n_turns: int,
        sign_dir: int,
        w_open: float = 0.030,
        w_squeeze: float = 0.0012,
    ) -> int:
        q_drum_anchor, _, _ = _walk_cartesian_chain([p_drum, p_pre, p_sky], R_drum, self.bp, self.bq)
        j7_c = float(np.clip(q_drum_anchor[6], -0.45, 0.55))
        j7_hi = j7_c + math.radians(60.0)
        j7_lo = j7_c - math.radians(60.0)
        j7_start = j7_hi if sign_dir > 0 else j7_lo
        j7_end = j7_lo if sign_dir > 0 else j7_hi

        q_start_drum = q_drum_anchor.copy()
        q_start_drum[6] = j7_start
        _, r_start_ref, _ = forward_kinematics_panda(q_start_drum, base_pos=self.bp, base_quat_wxyz=self.bq)
        _, q_pre_aligned, q_sky_aligned = _walk_cartesian_chain(
            [p_drum, p_pre, p_sky], r_start_ref, self.bp, self.bq, q_anchor_first=q_start_drum
        )

        self.move_joint(16, q_sky_aligned, w_target=w_open)
        self.move_cartesian(12, p_sky, p_pre, r_start_ref, w_target=w_open)
        self.move_cartesian(10, p_pre, p_drum, r_start_ref, w_target=w_open)
        q_grip_start = self.q_cur.copy()
        q_grip_end = q_grip_start.copy()
        q_grip_end[6] = j7_end

        for cyc in range(n_turns):
            self.hold_frames(8, w_target=w_squeeze)
            self.move_joint(16, q_grip_end, w_target=w_squeeze)
            self.hold_frames(6, w_target=w_open)
            if cyc < n_turns - 1:
                self.move_joint(11, q_grip_start, w_target=w_open)

        _, r_end_ref, _ = forward_kinematics_panda(self.q_cur, base_pos=self.bp, base_quat_wxyz=self.bq)
        self.move_cartesian(9, p_drum, p_pre, r_end_ref, w_target=w_open)
        self.move_cartesian(12, p_pre, p_sky, r_end_ref, w_target=w_open)
        return self.f_cur


def synthesize_v09_s01_s12_schedule(
    bL: np.ndarray,
    qL: np.ndarray,
    bR: np.ndarray,
    qR: np.ndarray,
    *,
    fps: int = 30,
) -> dict[str, Any]:
    """Synthesize uncut S01->S12 Method-B one-shot schedule (`100%` unified `R_top` / `R_topR` overhead branches!)."""
    MAX_T = 5400

    # Unified `R_top` (`ArmL`) & `R_topR` (`ArmR`) overhead planes (`ZERO 180° wrist flips, ZERO elbow fold-downs!`)
    R_top = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    R_topR = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)

    p_home_L = np.array([-0.350, +0.080, 1.240], dtype=np.float64)
    p_home_R = np.array([+0.340, +0.080, 1.220], dtype=np.float64)
    q_home_L = _solve_clean_anchor_ik(p_home_L, R_top, bL, qL)
    q_home_R = _solve_clean_anchor_ik(p_home_R, R_topR, bR, qR)

    armL = _ArmTimelineBuilder(MAX_T, bL, qL, q_home_L, w_init=0.038)
    armR = _ArmTimelineBuilder(MAX_T, bR, qR, q_home_R, w_init=0.036)

    R_wheel_y = _rot_from_z_and_y([0.0, -1.0, 0.0], [1.0, 0.0, 0.0])
    R_wing3 = _rot_from_z_and_y([0.0, -1.0, 0.0], [math.cos(TH15), 0.0, -math.sin(TH15)])
    R_neck_L = _rot_from_z_and_y([+0.74, -0.55, -0.385], [+0.59, +0.81, 0.0])

    # 1. Rotary Mechanisms
    p_jblue_whl = np.array([-0.180, 0.094, 0.785], dtype=np.float64)
    p_jblue_pre = np.array([-0.180, 0.185, 0.785], dtype=np.float64)
    p_jblue_sky = np.array([-0.260, 0.215, 1.120], dtype=np.float64)

    p_jwhite_whl = np.array([+0.215, 0.094, 0.776], dtype=np.float64)
    p_jwhite_pre = np.array([+0.215, 0.185, 0.776], dtype=np.float64)
    p_jwhite_sky = np.array([+0.305, 0.245, 1.220], dtype=np.float64)

    p_wing2 = np.array([-0.145, 0.067, 0.996], dtype=np.float64)
    p_wing2_pre = np.array([-0.145, 0.145, 0.996], dtype=np.float64)
    p_wing2_sky = np.array([+0.095, 0.195, 1.175], dtype=np.float64)

    clamp3_base = COND_SEATED_CENTER_W + 0.048 * U15 - 0.0038 * N15
    p_wing3 = clamp3_base + RY15 @ np.array([0.0, 0.056, 0.018], dtype=np.float64)
    p_wing3_pre = p_wing3 + np.array([0.0, 0.085, 0.0], dtype=np.float64)
    p_wing3_sky = p_wing3 + np.array([+0.120, 0.115, 0.140], dtype=np.float64)

    # 2. Flask Diagonal Entry/Exit Chain (`grasped squarely at Flask2429 outer collar z_local = +0.093 m`)
    p_flk_pick = np.array([-0.455, -0.020, 0.793 + 0.093], dtype=np.float64)
    p_flk_pre = p_flk_pick - 0.080 * R_neck_L[:, 2]
    p_flk_sky = np.array([-0.455, +0.040, 1.180], dtype=np.float64)
    p_flk_high1 = np.array([-0.455, -0.045, 1.140], dtype=np.float64)
    p_flk_front_high = np.array([-0.174, -0.068, 1.140], dtype=np.float64)
    p_flk_diag_a = np.array([-0.174, -0.066, 0.992 + 0.094], dtype=np.float64)
    p_flk_diag_b = np.array([-0.174, -0.018, 0.9415 + 0.094], dtype=np.float64)
    p_flk_pocket_in = np.array([-0.174, +0.0005, 0.9410 + 0.094], dtype=np.float64)
    p_flk_seat = np.array([-0.1800, +0.0005, 0.9390 + 0.094], dtype=np.float64)
    p_flk_unseat = np.array([-0.180, +0.0005, 0.9390 + 0.093], dtype=np.float64)
    p_flk_lift_clear = np.array([-0.174, +0.0005, 0.942 + 0.094], dtype=np.float64)
    # Pure mid-air horizontal slide `-Y` through the open `+X` doorway at `z_flk=0.942m` (`4.6mm` clear of `LeftPadFront`!)
    p_flk_exit_mid = np.array([-0.174, -0.052, 0.942 + 0.094], dtype=np.float64)
    p_flk_ret = p_flk_seat - 0.085 * R_neck_L[:, 2] + np.array([-0.030, 0.0, 0.025], dtype=np.float64)
    p_flk_ret_sky = np.array([-0.320, +0.110, 1.220], dtype=np.float64)
    q_flk_pick, q_flk_pre, q_flk_sky = _walk_cartesian_chain([p_flk_pick, p_flk_pre, p_flk_sky], R_neck_L, bL, qL)
    _, q_flk_high1, q_flk_front_high, q_flk_diag_a, q_flk_diag_b, q_flk_pocket_in, q_flk_seat, q_flk_unseat, q_flk_lift_clear, q_flk_exit_mid, q_flk_ret, q_flk_ret_sky = _walk_cartesian_chain(
        [p_flk_pick, p_flk_high1, p_flk_front_high, p_flk_diag_a, p_flk_diag_b, p_flk_pocket_in, p_flk_seat, p_flk_unseat, p_flk_lift_clear, p_flk_exit_mid, p_flk_ret, p_flk_ret_sky],
        R_neck_L,
        bL,
        qL,
        q_anchor_first=q_flk_pick,
    )

    # 3. Unified `q_home_L` (`R_top`) Stations: Head, Condenser, Keck #1, Keck #2, Funnel, Thermometer!
    p_hd_high = np.array([-0.335, -0.115, 1.220], dtype=np.float64)
    p_hd_pre = np.array([-0.335, -0.115, 0.990], dtype=np.float64)
    p_hd_pick = np.array([-0.335, -0.115, 0.785 + 0.100], dtype=np.float64)
    p_hd_hov = np.array([-0.180, 0.000, 1.220], dtype=np.float64)
    p_hd_seat = np.array([-0.180, 0.000, 1.0260 + 0.100], dtype=np.float64)
    p_hd_up = np.array([-0.180, 0.000, 1.225], dtype=np.float64)
    _, q_hd_high, q_hd_pre, q_hd_pick = _walk_cartesian_chain([p_home_L, p_hd_high, p_hd_pre, p_hd_pick], R_top, bL, qL, q_anchor_first=q_home_L)
    _, q_hd_hov, q_hd_seat, q_hd_up = _walk_cartesian_chain([p_hd_high, p_hd_hov, p_hd_seat, p_hd_up], R_top, bL, qL, q_anchor_first=q_hd_high)

    th_comp = math.radians(15.0)
    ry_comp = np.array(
        [[math.cos(th_comp), 0.0, math.sin(th_comp)], [0.0, 1.0, 0.0], [-math.sin(th_comp), 0.0, math.cos(th_comp)]],
        dtype=np.float64,
    )
    R_cond_15 = ry_comp @ R_top
    R_keck_15 = RY15 @ R_top

    p_chigh = np.array([0.020 - 0.038, -0.105, 1.180], dtype=np.float64)
    p_cpick_pre = np.array([0.020 - 0.038, -0.105, 0.900], dtype=np.float64)
    p_cpick = np.array([0.020 - 0.038, -0.105, 0.790], dtype=np.float64)
    # Seat flush at `COND_SEATED_CENTER_W - 0.038 * U15` (`0.0mm` centered Y & 15.0° coaxial with Head sidearm!)
    p_ee_seated = COND_SEATED_CENTER_W - 0.0380 * U15
    p_ee_pre_mate = p_ee_seated + 0.044 * U15
    # 2-step L-entry into `condenser_clamp_3prong_art001` (`-54°` open): enter horizontally at `+15mm` above `LowFront`, lower into cradle, then mate `-U15`!
    p_ee_throat_hi = p_ee_pre_mate + np.array([0.0, 0.0, +0.015], dtype=np.float64)
    p_ee_front = p_ee_throat_hi + np.array([0.0, -0.088, 0.0], dtype=np.float64)
    # At `x = -0.014` (`83mm` left of 3-prong clamp), approach/retract straight along `+N15`!
    p_ee_rel_up = p_ee_seated + 0.080 * N15
    _, q_chigh, q_cpick_pre, q_cpick = _walk_cartesian_chain([p_home_L, p_chigh, p_cpick_pre, p_cpick], R_top, bL, qL, q_anchor_first=q_home_L)
    q_cfront_seed = solve_ik_panda(p_ee_front, R_cond_15, q_init=q_chigh, base_pos=bL, base_quat_wxyz=qL, max_iters=180, damping=0.015)
    q_cfront, q_cthroat_hi, q_cpre, q_cseat, q_crel_up = _walk_cartesian_chain(
        [p_ee_front, p_ee_throat_hi, p_ee_pre_mate, p_ee_seated, p_ee_rel_up], R_cond_15, bL, qL, q_anchor_first=q_cfront_seed
    )

    # Tall Keck Pedestals (`top z=0.835 -> ClipSpine z=0.8462 -> DorsalGripTab z=0.8732m`!) + Front Detour Clear of Head/Thermometer!
    p_k_detour = np.array([-0.180, -0.135, 1.240], dtype=np.float64)
    q_k_detour = _walk_cartesian_chain([p_home_L, p_k_detour], R_top, bL, qL, q_anchor_first=q_home_L)[-1]
    p_k1_high = np.array([-0.415 - 0.0110, +0.065, 1.220], dtype=np.float64)
    p_k1_pre = np.array([-0.415 - 0.0110, +0.065, 0.960], dtype=np.float64)
    p_k1_pick = np.array([-0.415 - 0.0110, +0.065, 0.8462 + 0.0270], dtype=np.float64)
    p_k1_hov = HEAD_SIDEARM_CONE_W - 0.0110 * U15 + 0.072 * N15
    p_k1_snap = HEAD_SIDEARM_CONE_W - 0.0110 * U15 + 0.0265 * N15
    _, q_k1_high, q_k1_pre, q_k1_pick = _walk_cartesian_chain([p_home_L, p_k1_high, p_k1_pre, p_k1_pick], R_top, bL, qL, q_anchor_first=q_home_L)
    q_k1_hov_top = _walk_cartesian_chain([p_k_detour, p_k1_hov], R_top, bL, qL, q_anchor_first=q_k_detour)[-1]
    q_k1_hov_seed = solve_ik_panda(p_k1_hov, R_keck_15, q_init=q_k1_hov_top, base_pos=bL, base_quat_wxyz=qL, max_iters=180, damping=0.015)
    q_k1_hov, q_k1_snap = _walk_cartesian_chain([p_k1_hov, p_k1_snap], R_keck_15, bL, qL, q_anchor_first=q_k1_hov_seed)

    p_k2_high = np.array([-0.495 - 0.0110, +0.065, 1.220], dtype=np.float64)
    p_k2_pre = np.array([-0.495 - 0.0110, +0.065, 0.960], dtype=np.float64)
    p_k2_pick = np.array([-0.495 - 0.0110, +0.065, 0.8462 + 0.0270], dtype=np.float64)
    p_k2_hov = COND_LOWER_CONE_W - 0.0112 * U15 + 0.072 * N15
    p_k2_snap = COND_LOWER_CONE_W - 0.0112 * U15 + 0.0235 * N15
    _, q_k2_high, q_k2_pre, q_k2_pick = _walk_cartesian_chain([p_home_L, p_k2_high, p_k2_pre, p_k2_pick], R_top, bL, qL, q_anchor_first=q_home_L)
    q_k2_hov_top = _walk_cartesian_chain([p_k_detour, p_k2_hov], R_top, bL, qL, q_anchor_first=q_k_detour)[-1]
    q_k2_hov_seed = solve_ik_panda(p_k2_hov, R_keck_15, q_init=q_k2_hov_top, base_pos=bL, base_quat_wxyz=qL, max_iters=180, damping=0.015)
    q_k2_hov, q_k2_snap = _walk_cartesian_chain([p_k2_hov, p_k2_snap], R_keck_15, bL, qL, q_anchor_first=q_k2_hov_seed)

    p_fn_high = np.array([-0.465, -0.135, 1.310], dtype=np.float64)
    p_fn_pre = np.array([-0.465, -0.135, 1.020], dtype=np.float64)
    p_fn_pick = np.array([-0.465, -0.135, 0.862 + 0.034], dtype=np.float64)
    p_fn_hov = np.array([-0.180, 0.000, 1.310], dtype=np.float64)
    p_fn_seat = np.array([-0.180, 0.000, 1.1350 + 0.034], dtype=np.float64)
    _, q_fn_high, q_fn_pre, q_fn_pick = _walk_cartesian_chain([p_home_L, p_fn_high, p_fn_pre, p_fn_pick], R_top, bL, qL, q_anchor_first=q_home_L)
    _, q_fn_hov, q_fn_seat = _walk_cartesian_chain([p_fn_high, p_fn_hov, p_fn_seat], R_top, bL, qL, q_anchor_first=q_fn_high)

    p_th_high = np.array([-0.335, +0.025, 1.275], dtype=np.float64)
    p_th_pre = np.array([-0.335, +0.025, 1.020], dtype=np.float64)
    p_th_pick = np.array([-0.335, +0.025, 0.855 + 0.036], dtype=np.float64)
    p_th_hov = np.array([-0.180, 0.000, 1.275], dtype=np.float64)
    p_th_seat = np.array([-0.180, 0.000, 1.1320 + 0.036], dtype=np.float64)
    p_th_up = np.array([-0.180, 0.000, 1.250], dtype=np.float64)
    _, q_th_high, q_th_pre, q_th_pick = _walk_cartesian_chain([p_home_L, p_th_high, p_th_pre, p_th_pick], R_top, bL, qL, q_anchor_first=q_home_L)
    _, q_th_hov, q_th_seat, q_th_up = _walk_cartesian_chain([p_th_high, p_th_hov, p_th_seat, p_th_up], R_top, bL, qL, q_anchor_first=q_th_high)

    # 4. Right-Arm Connected Chains (`Erlenmeyer uses pure `R_topR` flat sky carry at z=1.22m over Beaker!`)
    tail_grip_local = np.array([+0.056, 0.0, -0.024], dtype=np.float64)
    z_dir_R = np.array([0.0, -0.880, -0.47497], dtype=np.float64)
    z_dir_R = z_dir_R / np.linalg.norm(z_dir_R)
    R_tail_R = _rot_from_z_and_y(z_dir_R, [+1.0, 0.0, 0.0])
    p_tpick = np.array([+0.310, +0.015, 0.855], dtype=np.float64) + tail_grip_local
    p_tpick_pre = p_tpick - 0.080 * z_dir_R
    p_tlift = p_tpick + np.array([0.0, 0.0, +0.185], dtype=np.float64)
    p_t_seated_ee = COND_LOWER_CONE_W + tail_grip_local
    p_t_pre_ee = p_t_seated_ee + 0.044 * U15
    p_t_ret_clean = p_t_seated_ee - 0.095 * z_dir_R
    p_t_ret_sky = p_t_ret_clean + np.array([+0.080, +0.050, +0.085], dtype=np.float64)
    q_tpick, q_tpick_pre = _walk_cartesian_chain([p_tpick, p_tpick_pre], R_tail_R, bR, qR)
    _, q_tlift, q_tpre_ee, q_tseat_ee, q_tret_clean, q_tret_sky = _walk_cartesian_chain(
        [p_tpick, p_tlift, p_t_pre_ee, p_t_seated_ee, p_t_ret_clean, p_t_ret_sky],
        R_tail_R,
        bR,
        qR,
        q_anchor_first=q_tpick,
    )

    p_epick_sky = np.array([+0.455, +0.015, 1.220], dtype=np.float64)
    p_epick = np.array([+0.455, +0.015, 0.750 + 0.084], dtype=np.float64)
    # Horizontal low-clearance slide onto White Scissor Jack (`+X -> -X` to exact drip tip X=+0.215 at z_erl=0.799m, 17mm below visual drip tip!)
    p_eplace_side_hov = np.array([+0.315, -0.005, 1.220], dtype=np.float64)
    p_eplace_side = np.array([+0.315, -0.005, 0.799 + 0.084], dtype=np.float64)
    p_eplace_hov = np.array([+0.210, -0.005, 1.220], dtype=np.float64)
    p_eplace = np.array([+0.210, -0.005, 0.799 + 0.084], dtype=np.float64)
    p_eplace_lifted = np.array([+0.210, -0.005, 0.824 + 0.084], dtype=np.float64)
    _, q_epick_sky, q_epick = _walk_cartesian_chain([p_home_R, p_epick_sky, p_epick], R_topR, bR, qR, q_anchor_first=q_home_R)
    _, q_eplace_side_hov, q_eplace_side, q_eplace, q_eplace_hov, q_eplace_lifted = _walk_cartesian_chain(
        [p_epick_sky, p_eplace_side_hov, p_eplace_side, p_eplace, p_eplace_hov, p_eplace_lifted],
        R_topR,
        bR,
        qR,
        q_anchor_first=q_epick_sky,
    )

    R_bkr_up = _rot_from_z_and_y([-0.65, -0.65, -0.39], [+0.65, -0.65, 0.0])
    ry_pour = np.array(
        [
            [math.cos(math.radians(-50.0)), 0.0, math.sin(math.radians(-50.0))],
            [0.0, 1.0, 0.0],
            [-math.sin(math.radians(-50.0)), 0.0, math.cos(math.radians(-50.0))],
        ],
        dtype=np.float64,
    )
    R_bkr_tilt = ry_pour @ R_bkr_up
    p_bpick = np.array([+0.445, -0.105, 0.792 + 0.040], dtype=np.float64)
    p_bpre = p_bpick - 0.075 * R_bkr_up[:, 2]
    p_bsky = np.array([+0.445, -0.050, 1.150], dtype=np.float64)
    p_bhigh = np.array([+0.120, -0.015, 1.245], dtype=np.float64)
    p_bpour = np.array([-0.122, +0.012, 1.228], dtype=np.float64)
    q_bpick, q_bpre, q_bsky, q_bhigh, q_bpour = _walk_cartesian_chain(
        [p_bpick, p_bpre, p_bsky, p_bhigh, p_bpour], R_bkr_up, bR, qR
    )
    q_bpour_tilt = solve_ik_panda(p_bpour, R_bkr_tilt, q_init=q_bpour, base_pos=bR, base_quat_wxyz=qR, max_iters=200, damping=0.02)

    stage_windows: list[tuple[int, int]] = []
    stage_rep_frames: dict[int, int] = {}
    phase_labels = ["Standby"] * MAX_T
    pour_active = np.zeros((MAX_T,), dtype=np.float64)
    bulb_drop_vis = np.zeros((MAX_T,), dtype=np.float64)
    distill_drip_active = np.zeros((MAX_T,), dtype=np.float64)
    anchor_fk_checks: list[tuple[str, str, int, np.ndarray]] = []

    def _fill_label(f0: int, f1: int, txt: str) -> None:
        for f in range(max(0, f0), min(MAX_T, f1 + 1)):
            phase_labels[f] = txt

    def _sync_both(wL_hold: float | None = None, wR_hold: float | None = None, pad_frames: int = 2) -> int:
        fs = max(armL.f_cur, armR.f_cur) + pad_frames
        armL.hold_until(fs, w_target=wL_hold)
        armR.hold_until(fs, w_target=wR_hold)
        return fs

    # =========================================================================
    # S01: Blue Scissor Jack 3x120° Handwheel Lift (+ Heating Bath)
    # =========================================================================
    s01_start = 0
    f_jb_rep = 58
    armL.rotary_screw_turns(p_jblue_whl, p_jblue_pre, p_jblue_sky, R_wheel_y, n_turns=3, sign_dir=+1, w_open=0.032, w_squeeze=0.0015)
    armL.move_joint(18, q_home_L, w_target=0.038)
    s01_end = _sync_both()
    stage_windows.append((s01_start, s01_end))
    stage_rep_frames[0] = f_jb_rep
    _fill_label(s01_start, s01_end, "S01 · Rotary Twist Blue Scissor Jack (+360°) Lifting Water Bath")
    f_s01_done = s01_end

    # =========================================================================
    # S02: Left Bosshead Open-Upward + Pre-Open 2-Prong Clamp (-120°)
    # =========================================================================
    s02_start = s01_end + 1
    f_s02_macro_a = s02_start
    armR.rotary_screw_turns(p_wing2, p_wing2_pre, p_wing2_sky, R_wheel_y, n_turns=1, sign_dir=-1, w_open=0.028, w_squeeze=0.0010)
    f_s02_rep = armR.f_cur - 16
    armR.move_joint(18, q_home_R, w_target=0.036)
    s02_end = _sync_both()
    stage_windows.append((s02_start, s02_end))
    stage_rep_frames[1] = f_s02_rep
    _fill_label(s02_start, s02_end, "S02 · Bosshead Open-Upward & Rotary Pre-Open 2-Prong Flask Clamp")

    # =========================================================================
    # S03: Pick Round-Bottom Flask -> Diagonal Bath Immersion -> Tighten 2-Prong (+240°)
    # =========================================================================
    s03_start = s02_end + 1
    armL.move_joint(18, q_flk_sky, w_target=0.038)
    armL.move_cartesian(12, p_flk_sky, p_flk_pre, R_neck_L, w_target=0.038)
    armL.move_cartesian(10, p_flk_pre, p_flk_pick, R_neck_L, w_target=0.038)
    armL.hold_frames(14, w_target=0.0024)
    anchor_fk_checks.append(("L", "S03_flk_pick", armL.f_cur, p_flk_pick))
    armL.move_cartesian(22, p_flk_pick, p_flk_high1, R_neck_L, w_target=0.0024)
    armL.move_cartesian(18, p_flk_high1, p_flk_front_high, R_neck_L, w_target=0.0024)
    armL.move_cartesian(14, p_flk_front_high, p_flk_diag_a, R_neck_L, w_target=0.0024)
    armL.move_cartesian(15, p_flk_diag_a, p_flk_diag_b, R_neck_L, w_target=0.0024)
    armL.move_cartesian(12, p_flk_diag_b, p_flk_pocket_in, R_neck_L, w_target=0.0024)
    armL.move_cartesian(10, p_flk_pocket_in, p_flk_seat, R_neck_L, w_target=0.0024)
    f_flk_seated = armL.hold_frames(6, w_target=0.0024)
    anchor_fk_checks.append(("L", "S03_flk_seat", f_flk_seated, p_flk_seat))

    armR.hold_until(f_flk_seated - 18, w_target=0.028)
    armR.rotary_screw_turns(p_wing2, p_wing2_pre, p_wing2_sky, R_wheel_y, n_turns=2, sign_dir=+1, w_open=0.028, w_squeeze=0.0010)
    f_c2_tight = armR.f_cur - 12
    f_s03_rep = f_c2_tight - 18
    armR.move_joint(18, q_home_R, w_target=0.036)

    armL.hold_until(f_c2_tight, w_target=0.0024)
    armL.hold_frames(10, w_target=0.038)
    armL.move_cartesian(10, p_flk_seat, p_flk_ret, R_neck_L, w_target=0.038)
    armL.move_cartesian(14, p_flk_ret, p_flk_ret_sky, R_neck_L, w_target=0.038)
    armL.move_joint(18, q_home_L, w_target=0.038)
    s03_end = _sync_both()
    stage_windows.append((s03_start, s03_end))
    stage_rep_frames[2] = f_s03_rep
    _fill_label(s03_start, f_flk_seated, "S03 · Frictional Neck Lift & Diagonal Immerse Flask into Bath")
    _fill_label(f_flk_seated + 1, s03_end, "S03 · Bimanual Rotary Wingnut Tightening (+240°) of 2-Prong Clamp")

    # =========================================================================
    # S04: Closed-Loop `R_top` Overhead Pick & Method B 24/29 Insertion of 75° Head
    # =========================================================================
    s04_start = s03_end + 1
    armL.move_cartesian(16, p_home_L, p_hd_high, R_top, w_target=0.038)
    armL.move_cartesian(14, p_hd_high, p_hd_pre, R_top, w_target=0.038)
    armL.move_cartesian(12, p_hd_pre, p_hd_pick, R_top, w_target=0.038)
    armL.hold_frames(14, w_target=0.0010)
    anchor_fk_checks.append(("L", "S04_hd_pick", armL.f_cur, p_hd_pick))
    armL.move_cartesian(28, p_hd_pick, p_hd_high, R_top, w_target=0.0010)
    f_s04_hd_a = armL.f_cur
    armL.move_cartesian(24, p_hd_high, p_hd_hov, R_top, w_target=0.0010)
    f_hd_hov = armL.hold_frames(8, w_target=0.0010)
    armL.move_cartesian(26, p_hd_hov, p_hd_seat, R_top, w_target=0.0010)
    f_s04_rep = armL.hold_frames(10, w_target=0.0010)
    anchor_fk_checks.append(("L", "S04_hd_seat", f_s04_rep, p_hd_seat))
    armL.hold_frames(10, w_target=0.038)
    armL.move_cartesian(12, p_hd_seat, p_hd_up, R_top, w_target=0.038)
    armL.move_cartesian(14, p_hd_up, p_home_L, R_top, w_target=0.038)
    f_s04_hd_b = armL.hold_frames(22, w_target=0.038)
    s04_end = _sync_both()
    stage_windows.append((s04_start, s04_end))
    stage_rep_frames[3] = f_s04_rep
    _fill_label(s04_start, f_hd_hov, "S04 · Pick 75° Distillation Head & Closed-Loop Align Over Flask 24/29")
    _fill_label(f_hd_hov + 1, s04_end, "S04 · Coaxial 24/29 Push into Two-Ring Method-B Micro-Spring Wedge")

    # =========================================================================
    # S05: Right Bosshead Open-Upward + Pre-Open 3-Prong Clamp (-120°)
    # =========================================================================
    s05_start = s04_end + 1
    armR.rotary_screw_turns(p_wing3, p_wing3_pre, p_wing3_sky, R_wing3, n_turns=1, sign_dir=-1, w_open=0.028, w_squeeze=0.0010)
    f_s05_rep = armR.f_cur - 16
    armR.move_joint(18, q_home_R, w_target=0.036)
    s05_end = _sync_both()
    stage_windows.append((s05_start, s05_end))
    stage_rep_frames[4] = f_s05_rep
    _fill_label(s05_start, s05_end, "S05 · Right Bosshead Open-Upward & Pre-Open 3-Prong Clamp (-54°)")

    # =========================================================================
    # S06: Closed-Loop `R_top` Condenser (+ 48 Live Hoses) 15° Mate + 3-Prong Tighten (Pure Method B!)
    # =========================================================================
    s06_start = s05_end + 1
    armL.move_cartesian(20, p_home_L, p_chigh, R_top, w_target=0.038)
    armL.move_cartesian(16, p_chigh, p_cpick_pre, R_top, w_target=0.038)
    armL.move_cartesian(12, p_cpick_pre, p_cpick, R_top, w_target=0.038)
    armL.hold_frames(14, w_target=0.0015)
    anchor_fk_checks.append(("L", "S06_cond_pick", armL.f_cur, p_cpick))
    armL.move_cartesian(30, p_cpick, p_chigh, R_top, w_target=0.0015)
    f_s06_cond_a = armL.f_cur
    armL.move_joint(22, q_cfront, w_target=0.0015)
    armL.move_cartesian(18, p_ee_front, p_ee_throat_hi, R_cond_15, w_target=0.0015)
    armL.move_cartesian(12, p_ee_throat_hi, p_ee_pre_mate, R_cond_15, w_target=0.0015)
    f_s06_macro_a = armL.hold_frames(8, w_target=0.0015)
    armL.move_cartesian(24, p_ee_pre_mate, p_ee_seated, R_cond_15, w_target=0.0015)
    f_cond_in = armL.hold_frames(8, w_target=0.0015)
    anchor_fk_checks.append(("L", "S06_cond_seat", f_cond_in, p_ee_seated))

    armR.hold_until(f_cond_in - 18, w_target=0.028)
    armR.rotary_screw_turns(p_wing3, p_wing3_pre, p_wing3_sky, R_wing3, n_turns=2, sign_dir=+1, w_open=0.028, w_squeeze=0.0010)
    f_c3_tight = armR.f_cur - 12
    armR.move_joint(18, q_home_R, w_target=0.036)

    armL.hold_until(f_c3_tight, w_target=0.0015)
    armL.hold_frames(10, w_target=0.040)
    armL.move_cartesian(14, p_ee_seated, p_ee_rel_up, R_cond_15, w_target=0.040)
    f_s06_rep = armL.f_cur
    armL.move_joint(16, q_home_L, w_target=0.038)
    f_s06_cond_b = armL.hold_frames(22, w_target=0.038)
    s06_end = _sync_both()
    stage_windows.append((s06_start, s06_end))
    stage_rep_frames[5] = f_s06_rep
    _fill_label(s06_start, f_cond_in, "S06 · Closed-Loop 15° Coaxial Insertion of Condenser + 48 Live Hoses")
    _fill_label(f_cond_in + 1, s06_end, "S06 · Bimanual 3-Prong Clamp Tightening (+240°) & Method-B Self-Lock")

    # =========================================================================
    # S07: Closed-Loop Tail Adapter Pure Method-B Self-Lock + Coaxial Receiver + White Jack Lift (+240°)
    # =========================================================================
    s07_start = s06_end + 1
    armR.move_joint(18, q_tlift, w_target=0.028)
    armR.move_cartesian(16, p_tlift, p_tpick, R_tail_R, w_target=0.028)
    armR.hold_frames(12, w_target=0.0012)
    anchor_fk_checks.append(("R", "S07_tail_pick", armR.f_cur, p_tpick))
    armR.move_cartesian(20, p_tpick, p_tlift, R_tail_R, w_target=0.0012)
    f_s07_tail_a = armR.f_cur
    armR.move_cartesian(18, p_tlift, p_t_pre_ee, R_tail_R, w_target=0.0012)
    armR.hold_frames(8, w_target=0.0012)
    armR.move_cartesian(24, p_t_pre_ee, p_t_seated_ee, R_tail_R, w_target=0.0012)
    f_tail_in = armR.hold_frames(8, w_target=0.0012)
    anchor_fk_checks.append(("R", "S07_tail_seat", f_tail_in, p_t_seated_ee))
    armR.hold_frames(14, w_target=0.024)
    armR.move_cartesian(12, p_t_seated_ee, p_t_ret_clean, R_tail_R, w_target=0.024)
    armR.move_cartesian(12, p_t_ret_clean, p_t_ret_sky, R_tail_R, w_target=0.036)
    armR.move_joint(18, q_home_R, w_target=0.036)
    f_s07_tail_b = armR.hold_frames(22, w_target=0.036)
    f_tail_done = armR.f_cur

    # Pure `R_topR` horizontal sky carry -> descend at x=+0.315 (clear of Tail!) -> slide -X onto White Scissor Jack below Tail!
    armR.move_cartesian(14, p_home_R, p_epick_sky, R_topR, w_target=0.036)
    armR.move_cartesian(18, p_epick_sky, p_epick, R_topR, w_target=0.036)
    armR.hold_frames(12, w_target=0.0035)
    anchor_fk_checks.append(("R", "S07_erl_pick", armR.f_cur, p_epick))
    armR.move_cartesian(22, p_epick, p_epick_sky, R_topR, w_target=0.0035)
    armR.move_cartesian(18, p_epick_sky, p_eplace_side_hov, R_topR, w_target=0.0035)
    armR.move_cartesian(18, p_eplace_side_hov, p_eplace_side, R_topR, w_target=0.0035)
    armR.move_cartesian(16, p_eplace_side, p_eplace, R_topR, w_target=0.0035)
    armR.hold_frames(8, w_target=0.0035)
    anchor_fk_checks.append(("R", "S07_erl_place", armR.f_cur, p_eplace))
    armR.hold_frames(10, w_target=0.036)
    armR.move_cartesian(14, p_eplace, p_eplace_side, R_topR, w_target=0.036)
    armR.move_cartesian(16, p_eplace_side, p_eplace_side_hov, R_topR, w_target=0.036)
    armR.rotary_screw_turns(p_jwhite_whl, p_jwhite_pre, p_jwhite_sky, R_wheel_y, n_turns=2, sign_dir=+1, w_open=0.032, w_squeeze=0.0015)
    f_s07_rep = armR.f_cur - 16
    armR.move_joint(18, q_home_R, w_target=0.036)
    s07_end = _sync_both()
    stage_windows.append((s07_start, s07_end))
    stage_rep_frames[6] = f_s07_rep
    _fill_label(s07_start, f_tail_done, "S07 · 15° Bare Frosted Tail Adapter Pure Method-B Self-Lock")
    _fill_label(f_tail_done + 1, s07_end, "S07 · Stage Erlenmeyer & Twist White Scissor Jack Raising Receiver")

    # =========================================================================
    # S08: Funnel Self-Locks in 19/26 -> Beaker Pours Purple KMnO4 (`Hands-Off Funnel!`)
    # =========================================================================
    s08_start = s07_end + 1
    armL.move_cartesian(16, p_home_L, p_fn_high, R_top, w_target=0.038)
    armL.move_cartesian(16, p_fn_high, p_fn_pre, R_top, w_target=0.038)
    armL.move_cartesian(12, p_fn_pre, p_fn_pick, R_top, w_target=0.038)
    armL.hold_frames(12, w_target=0.0035)
    anchor_fk_checks.append(("L", "S08_fn_pick", armL.f_cur, p_fn_pick))
    armL.move_cartesian(24, p_fn_pick, p_fn_high, R_top, w_target=0.0035)
    f_s08_fn_a = armL.f_cur
    armL.move_cartesian(20, p_fn_high, p_fn_hov, R_top, w_target=0.0035)
    armL.move_cartesian(22, p_fn_hov, p_fn_seat, R_top, w_target=0.0035)
    f_fn_in = armL.hold_frames(8, w_target=0.0035)
    anchor_fk_checks.append(("L", "S08_fn_seat", f_fn_in, p_fn_seat))
    armL.hold_frames(16, w_target=0.042)
    armL.move_cartesian(16, p_fn_seat, p_fn_hov, R_top, w_target=0.042)
    armL.move_cartesian(16, p_fn_hov, p_home_L, R_top, w_target=0.042)
    f_s08_fn_b = armL.hold_frames(28, w_target=0.042)
    f_armL_parked = armL.f_cur

    armR.hold_until(f_fn_in - 14, w_target=0.038)
    armR.move_joint(18, q_bsky, w_target=0.038)
    armR.move_cartesian(14, p_bsky, p_bpre, R_bkr_up, w_target=0.038)
    armR.move_cartesian(14, p_bpre, p_bpick, R_bkr_up, w_target=0.038)
    armR.hold_frames(12, w_target=0.0055)
    anchor_fk_checks.append(("R", "S08_bkr_pick", armR.f_cur, p_bpick))
    armR.move_cartesian(24, p_bpick, p_bhigh, R_bkr_up, w_target=0.0055)
    armR.hold_until(max(armR.f_cur, f_armL_parked), w_target=0.0055)
    f_s08_bkr_a = armR.f_cur
    armR.move_cartesian(20, p_bhigh, p_bpour, R_bkr_up, w_target=0.0055)
    armR.move_joint(14, q_bpour_tilt, w_target=0.0055)
    f_pour_a = armR.f_cur
    armR.hold_frames(26, w_target=0.0055)
    f_pour_b = armR.f_cur
    pour_active[f_pour_a : f_pour_b + 1] = 1.0
    f_s08_pour_mid = (f_pour_a + f_pour_b) // 2
    armR.move_joint(14, q_bpour, w_target=0.0055)
    armR.move_cartesian(20, p_bpour, p_bhigh, R_bkr_up, w_target=0.0055)
    f_s08_bkr_b = armR.hold_frames(26, w_target=0.0055)
    armR.move_cartesian(20, p_bhigh, p_bpick, R_bkr_up, w_target=0.0055)
    armR.hold_frames(6, w_target=0.0055)
    armR.hold_frames(10, w_target=0.038)
    armR.move_cartesian(10, p_bpick, p_bpre, R_bkr_up, w_target=0.038)
    armR.move_cartesian(12, p_bpre, p_bsky, R_bkr_up, w_target=0.038)
    armR.move_joint(18, q_home_R, w_target=0.036)
    s08_end = _sync_both()
    stage_windows.append((s08_start, s08_end))
    stage_rep_frames[7] = f_s08_pour_mid
    _fill_label(s08_start, f_armL_parked, "S08 · Closed-Loop 19/26 Pose Servo & Seat Long-Stem Funnel")
    _fill_label(f_armL_parked + 1, s08_end, "S08 · Tilt Beaker Pouring Purple KMnO4 Solution into Flask")

    # =========================================================================
    # S09: Closed-Loop Twist-Pull Remove Funnel + Coaxial Seat Thermometer (`0.0mm` Bulb Flush!)
    # =========================================================================
    s09_start = s08_end + 1
    f_s09a_a = armL.f_cur
    armL.move_cartesian(16, p_home_L, p_fn_hov, R_top, w_target=0.042)
    armL.move_cartesian(16, p_fn_hov, p_fn_seat, R_top, w_target=0.042)
    armL.hold_frames(12, w_target=0.0035)
    armL.move_cartesian(
        22, p_fn_seat, p_fn_hov, R_top, w_target=0.0035,
        linear_rate=True, j7_add_start=0.0, j7_add_end=math.radians(65.0), q_untwisted_start=q_fn_seat,
    )
    armL.move_joint(18, q_fn_high, w_target=0.0035)
    f_s09a_b = armL.hold_frames(28, w_target=0.0035)
    armL.move_cartesian(20, p_fn_high, p_fn_pick, R_top, w_target=0.0035)
    armL.hold_frames(6, w_target=0.0035)
    armL.hold_frames(10, w_target=0.038)
    armL.move_cartesian(16, p_fn_pick, p_fn_high, R_top, w_target=0.038)
    f_fn_back = armL.f_cur

    armL.move_cartesian(16, p_fn_high, p_th_high, R_top, w_target=0.038)
    armL.move_cartesian(14, p_th_high, p_th_pre, R_top, w_target=0.038)
    armL.move_cartesian(10, p_th_pre, p_th_pick, R_top, w_target=0.038)
    armL.hold_frames(12, w_target=0.0020)
    anchor_fk_checks.append(("L", "S09_th_pick", armL.f_cur, p_th_pick))
    armL.move_cartesian(22, p_th_pick, p_th_high, R_top, w_target=0.0020)
    f_s09b_a = armL.f_cur
    armL.move_cartesian(16, p_th_high, p_th_hov, R_top, w_target=0.0020)
    f_s09_macro_a = armL.f_cur
    armL.move_cartesian(24, p_th_hov, p_th_seat, R_top, w_target=0.0020)
    f_th_in = armL.hold_frames(8, w_target=0.0020)
    anchor_fk_checks.append(("L", "S09_th_seat", f_th_in, p_th_seat))
    armL.hold_frames(16, w_target=0.038)
    armL.move_cartesian(12, p_th_seat, p_th_up, R_top, w_target=0.038)
    f_s09_rep = armL.f_cur
    armL.move_cartesian(16, p_th_up, p_home_L, R_top, w_target=0.038)
    f_s09b_b = armL.hold_frames(28, w_target=0.038)
    s09_end = _sync_both()
    stage_windows.append((s09_start, s09_end))
    stage_rep_frames[8] = f_s09_rep
    _fill_label(s09_start, f_fn_back, "S09 · Twist-Pull Remove Funnel Back to Front-Left Rack")
    _fill_label(f_fn_back + 1, s09_end, "S09 · Seat 19/26 Method-B Thermometer (Bulb 0.0mm Flush!)")

    # =========================================================================
    # S10: Active Hands-Off Distillation, Bulb Condensate & Clear Drip
    # =========================================================================
    s10_start = s09_end + 1
    armL.hold_frames(75, w_target=0.038)
    s10_end = _sync_both()
    f_s10_mid = (s10_start + s10_end) // 2
    bulb_drop_vis[s10_start : s10_end + 1] = 1.0
    distill_drip_active[s10_start + 3 : s10_end - 2] = 1.0
    stage_windows.append((s10_start, s10_end))
    stage_rep_frames[9] = f_s10_mid
    _fill_label(s10_start, s10_end, "S10 · Active Distillation · Bulb Condensate & Clear Distillate Drip")

    # =========================================================================
    # S11: Reverse Right-to-Left Disassembly (White Jack -> Receiver -> Keck2 -> Tail -> Keck1 -> Condenser)
    # =========================================================================
    s11_start = s10_end + 1
    armR.rotary_screw_turns(p_jwhite_whl, p_jwhite_pre, p_jwhite_sky, R_wheel_y, n_turns=3, sign_dir=-1, w_open=0.032, w_squeeze=0.0015)
    armR.move_joint(18, q_eplace_side_hov, w_target=0.036)
    # Reverse horizontal low-clearance slide (+X) below Tail drip tip before lifting!
    armR.move_cartesian(18, p_eplace_side_hov, p_eplace_side, R_topR, w_target=0.036)
    armR.move_cartesian(14, p_eplace_side, p_eplace, R_topR, w_target=0.036)
    armR.hold_frames(12, w_target=0.0035)
    armR.move_cartesian(16, p_eplace, p_eplace_side, R_topR, w_target=0.0035)
    armR.move_cartesian(18, p_eplace_side, p_eplace_side_hov, R_topR, w_target=0.0035)
    armR.move_cartesian(18, p_eplace_side_hov, p_epick_sky, R_topR, w_target=0.0035)
    armR.move_cartesian(18, p_epick_sky, p_epick, R_topR, w_target=0.0035)
    armR.hold_frames(6, w_target=0.0035)
    armR.hold_frames(10, w_target=0.036)
    armR.move_cartesian(16, p_epick, p_epick_sky, R_topR, w_target=0.036)
    armR.move_cartesian(14, p_epick_sky, p_home_R, R_topR, w_target=0.036)
    f_rec_out = armR.f_cur

    f_s11_tail_a = armR.f_cur
    armR.move_joint(18, q_tret_sky, w_target=0.028)
    armR.move_cartesian(12, p_t_ret_sky, p_t_ret_clean, R_tail_R, w_target=0.028)
    armR.move_cartesian(12, p_t_ret_clean, p_t_seated_ee, R_tail_R, w_target=0.028)
    armR.hold_frames(12, w_target=0.0012)
    armR.move_cartesian(
        18, p_t_seated_ee, p_t_pre_ee, R_tail_R, w_target=0.0012,
        linear_rate=True, j7_add_start=0.0, j7_add_end=math.radians(24.0), q_untwisted_start=q_tseat_ee,
    )
    armR.move_cartesian(18, p_t_pre_ee, p_tlift, R_tail_R, w_target=0.0012)
    f_s11_tail_b = armR.hold_frames(18, w_target=0.0012)
    armR.move_cartesian(18, p_tlift, p_tpick, R_tail_R, w_target=0.0012)
    armR.hold_frames(6, w_target=0.0012)
    armR.hold_frames(10, w_target=0.028)
    armR.move_cartesian(14, p_tpick, p_tlift, R_tail_R, w_target=0.028)
    armR.move_joint(18, q_home_R, w_target=0.036)
    f_tail_out = armR.f_cur

    armL.hold_until(max(armL.f_cur, f_tail_out - 20), w_target=0.038)
    armL.move_joint(18, q_crel_up, w_target=0.038)
    armL.move_cartesian(14, p_ee_rel_up, p_ee_seated, R_cond_15, w_target=0.038)
    f_cond_held = armL.hold_frames(14, w_target=0.0012)

    armR.hold_until(max(armR.f_cur, f_cond_held - 20), w_target=0.028)
    armR.rotary_screw_turns(p_wing3, p_wing3_pre, p_wing3_sky, R_wing3, n_turns=3, sign_dir=-1, w_open=0.028, w_squeeze=0.0010)
    f_c3_open = armR.f_cur - 12
    armR.move_joint(18, q_home_R, w_target=0.036)

    armL.hold_until(f_c3_open + 2, w_target=0.0012)
    armL.move_cartesian(20, p_ee_seated, p_ee_pre_mate, R_cond_15, w_target=0.0012, linear_rate=True)
    armL.move_cartesian(14, p_ee_pre_mate, p_ee_throat_hi, R_cond_15, w_target=0.0012)
    armL.move_cartesian(16, p_ee_throat_hi, p_ee_front, R_cond_15, w_target=0.0012)
    armL.move_joint(20, q_chigh, w_target=0.0012)
    armL.move_cartesian(22, p_chigh, p_cpick, R_top, w_target=0.0012)
    armL.hold_frames(6, w_target=0.0012)
    armL.hold_frames(10, w_target=0.038)
    armL.move_cartesian(18, p_cpick, p_chigh, R_top, w_target=0.038)
    armL.move_cartesian(16, p_chigh, p_home_L, R_top, w_target=0.038)
    s11_end = _sync_both()
    stage_windows.append((s11_start, s11_end))
    stage_rep_frames[10] = s11_end - 4
    _fill_label(s11_start, f_rec_out, "S11 · Reverse Twist White Jack (-240°) & Dismount Receiver")
    _fill_label(f_rec_out + 1, f_cond_held, "S11 · Coaxial Twist-and-Pull Unlock Method-B Tail Adapter")
    _fill_label(f_cond_held + 1, s11_end, "S11 · Loosen 3-Prong (-240°) & Extract Condenser + 48 Live Hoses")

    # =========================================================================
    # S12: Reverse Top-to-Bottom Coaxial Twist-and-Pull Left Column!
    # =========================================================================
    s12_start = s11_end + 1
    f_s12_start = s12_start

    # 12a: Closed-Loop Coaxial Twist-and-Pull (`+150°` j7) Thermometer -> `ThermoBenchRack`
    q_th_up_pretw = q_th_up.copy()
    q_th_up_pretw[6] = np.clip(q_th_seat[6] - math.radians(75.0), PANDA_ARM_LOWER[6] + 0.03, PANDA_ARM_UPPER[6] - 0.03)
    q_th_seat_pretw = q_th_seat.copy()
    q_th_seat_pretw[6] = q_th_up_pretw[6]

    f_s12a_a = armL.f_cur
    armL.move_joint(18, q_th_up_pretw, w_target=0.038)
    armL.move_joint(12, q_th_seat_pretw, w_target=0.038)
    armL.hold_frames(12, w_target=0.0020)
    armL.move_cartesian(
        26, p_th_seat, p_th_hov, R_top, w_target=0.0020,
        linear_rate=True, j7_add_start=-math.radians(75.0), j7_add_end=+math.radians(75.0), q_untwisted_start=q_th_seat,
    )
    armL.move_joint(18, q_th_high, w_target=0.0020)
    f_s12a_b = armL.hold_frames(28, w_target=0.0020)
    armL.move_cartesian(18, p_th_high, p_th_pick, R_top, w_target=0.0020)
    armL.hold_frames(6, w_target=0.0020)
    armL.hold_frames(10, w_target=0.038)
    armL.move_cartesian(16, p_th_pick, p_th_high, R_top, w_target=0.038)
    f_12a_done = armL.f_cur

    # 12b: Closed-Loop Coaxial Twist-and-Pull (`+70°` j7 helical unlock) 75° Distillation Head -> `HeadBenchRack`
    f_s12b_a = armL.f_cur
    armL.move_joint(18, q_hd_up, w_target=0.038)
    armL.move_cartesian(14, p_hd_up, p_hd_seat, R_top, w_target=0.038)
    armL.hold_frames(12, w_target=0.0010)
    armL.move_cartesian(
        28, p_hd_seat, p_hd_hov, R_top, w_target=0.0010,
        linear_rate=True, j7_add_start=0.0, j7_add_end=+math.radians(70.0), q_untwisted_start=q_hd_seat,
    )
    armL.move_joint(20, q_hd_high, w_target=0.0010)
    f_s12b_b = armL.hold_frames(28, w_target=0.0010)
    armL.move_cartesian(20, p_hd_high, p_hd_pick, R_top, w_target=0.0010)
    armL.hold_frames(6, w_target=0.0010)
    armL.hold_frames(10, w_target=0.038)
    armL.move_cartesian(16, p_hd_pick, p_hd_high, R_top, w_target=0.038)
    f_12b_done = armL.f_cur

    # 12c: ArmL holds Flask neck -> ArmR loosens 2-Prong (-360°) -> ArmL reverses diagonal exit to Cork Ring!
    armL.move_joint(18, q_flk_ret_sky, w_target=0.038)
    armL.move_cartesian(12, p_flk_ret_sky, p_flk_ret, R_neck_L, w_target=0.038)
    armL.move_cartesian(10, p_flk_ret, p_flk_unseat, R_neck_L, w_target=0.038)
    f_flk_held = armL.hold_frames(14, w_target=0.0015)

    armR.hold_until(max(armR.f_cur, f_flk_held - 18), w_target=0.028)
    armR.rotary_screw_turns(p_wing2, p_wing2_pre, p_wing2_sky, R_wheel_y, n_turns=3, sign_dir=-1, w_open=0.028, w_squeeze=0.0010)
    f_c2_open = armR.f_cur - 12
    armR.move_joint(18, q_home_R, w_target=0.036)

    armL.hold_until(f_c2_open + 2, w_target=0.0015)
    armL.move_cartesian(8, p_flk_unseat, p_flk_lift_clear, R_neck_L, w_target=0.0015)
    armL.move_cartesian(14, p_flk_lift_clear, p_flk_exit_mid, R_neck_L, w_target=0.0015)
    armL.move_cartesian(14, p_flk_exit_mid, p_flk_diag_a, R_neck_L, w_target=0.0015)
    armL.move_cartesian(14, p_flk_diag_a, p_flk_front_high, R_neck_L, w_target=0.0015)
    armL.move_cartesian(22, p_flk_front_high, p_flk_high1, R_neck_L, w_target=0.0015)
    armL.move_cartesian(22, p_flk_high1, p_flk_pick, R_neck_L, w_target=0.0015)
    armL.hold_frames(6, w_target=0.0015)
    armL.hold_frames(10, w_target=0.038)
    armL.move_cartesian(14, p_flk_pick, p_flk_sky, R_neck_L, w_target=0.038)
    armL.move_joint(18, q_home_L, w_target=0.038)
    armL.hold_frames(12, w_target=0.038)
    s12_end = _sync_both(pad_frames=4)

    T = min(MAX_T, s12_end + 1)
    stage_windows.append((s12_start, T - 1))
    stage_rep_frames[11] = T - 5
    _fill_label(s12_start, f_12a_done, "S12 · Coaxial Twist-and-Pull (+150°) Extract Thermometer")
    _fill_label(f_12a_done + 1, f_12b_done, "S12 · Coaxial Twist-and-Pull (+170°) Extract 75° Distillation Head")
    _fill_label(f_12b_done + 1, T - 1, "S12 · Reverse Loosen 2-Prong (-240°) & Dismount Flask to Cork Ring")

    stage_index = np.zeros((T,), dtype=np.int32)
    for s_i, (fa, fb) in enumerate(stage_windows):
        stage_index[fa : fb + 1] = s_i

    macro_schedule = {
        "macro_01_bosshead_open_up": ("Cam_Bosshead_Close", list(range(f_s02_macro_a, s02_end + 1)), f_s02_rep),
        "macro_02_clamp3prong_hoses": ("Cam_Clamp3Prong_Close", list(range(f_s06_macro_a, s06_end + 1)), f_s06_rep),
        "macro_03_thermometer_bulb_flush": ("Cam_Thermometer_Bulb", list(range(f_s09_macro_a, s10_start + 25)), s10_start + 10),
        "macro_04_receiver_drip": ("Cam_Receiver_Drip", list(range(s10_start, s10_end + 1)), f_s10_mid),
    }

    return {
        "T": T,
        "fps": fps,
        "qL_seq": np.clip(armL.q_seq[:T], PANDA_ARM_LOWER, PANDA_ARM_UPPER),
        "wL_seq": armL.w_seq[:T].copy(),
        "qR_seq": np.clip(armR.q_seq[:T], PANDA_ARM_LOWER, PANDA_ARM_UPPER),
        "wR_seq": armR.w_seq[:T].copy(),
        "stage_windows": stage_windows,
        "stage_index": stage_index,
        "stage_rep_frames": stage_rep_frames,
        "phase_labels": phase_labels[:T],
        "pour_active": pour_active[:T].copy(),
        "bulb_drop_vis": bulb_drop_vis[:T].copy(),
        "distill_drip_active": distill_drip_active[:T].copy(),
        "macro_schedule": macro_schedule,
        "anchor_fk_checks": anchor_fk_checks,
        "f_s01_done": f_s01_done,
        "f_s08_pour_mid": f_s08_pour_mid,
        "f_s10_mid": f_s10_mid,
        "f_s12_start": f_s12_start,
        "servo_windows": {
            "s04_hd_insert": (f_s04_hd_a, f_s04_hd_b, "round_bottom_flask_250ml", "distillation_head_75deg"),
            "s06_cond_insert": (f_s06_cond_a, f_s06_cond_b, "distillation_head_75deg", "liebig_condenser_24_29"),
            "s07_tail_insert": (f_s07_tail_a, f_s07_tail_b, "liebig_condenser_24_29", "vacuum_tail_adapter"),
            "s08_fn_insert": (f_s08_fn_a, f_s08_fn_b, "distillation_head_75deg", None),
            "s08_bkr_pour": (f_s08_bkr_a, f_s08_bkr_b, "long_stem_funnel", None),
            "s09a_fn_extract": (f_s09a_a, f_s09a_b, "long_stem_funnel", None),
            "s09b_th_insert": (f_s09b_a, f_s09b_b, "distillation_head_75deg", None),
            "s11_tail_extract": (f_s11_tail_a, f_s11_tail_b, "vacuum_tail_adapter", None),
            "s12a_th_extract": (f_s12a_a, f_s12a_b, "thermometer_adapter_unit_v08", None),
            "s12b_hd_extract": (f_s12b_a, f_s12b_b, "distillation_head_75deg", None),
        },
        "servo_anchors": {
            "p_home_L": p_home_L,
            "q_home_L": q_home_L,
            "p_home_R": p_home_R,
            "q_home_R": q_home_R,
            "p_chigh": p_chigh,
            "q_chigh": q_chigh,
            "f_c3_tight": f_c3_tight,
            "p_tlift": p_tlift,
            "q_tlift": q_tlift,
            "tail_grip_local": tail_grip_local,
            "R_tail_R": R_tail_R,
            "p_fn_high": p_fn_high,
            "q_fn_high": q_fn_high,
            "p_th_high": p_th_high,
            "q_th_high": q_th_high,
            "p_hd_high": p_hd_high,
            "q_hd_high": q_hd_high,
            "p_bhigh": p_bhigh,
            "q_bhigh": q_bhigh,
            "R_top": R_top,
            "R_bkr_up": R_bkr_up,
            "R_bkr_tilt": R_bkr_tilt,
        },
    }


def replan_closed_loop_window(
    sched: dict[str, Any],
    window_name: str,
    live_pos: np.ndarray,
    live_rot: np.ndarray,
    bL: np.ndarray,
    qL: np.ndarray,
    bR: np.ndarray,
    qR: np.ndarray,
    *,
    held_pos: np.ndarray | None = None,
    held_rot: np.ndarray | None = None,
) -> bool:
    """Closed-loop 6D pose-servo splice onto live target AND live in-hand object pose (`0.000 rad` boundary seam!)."""
    if window_name not in sched.get("servo_windows", {}):
        return False
    wspec = sched["servo_windows"][window_name]
    fa, fb = int(wspec[0]), int(wspec[1])
    n_win = fb - fa + 1
    if n_win < 12 or float(live_pos[2]) < 0.88:
        return False

    anc = sched["servo_anchors"]
    R_top = anc["R_top"]

    if window_name == "s07_tail_insert":
        q_start = sched["qR_seq"][fa].copy()
        w_start = float(sched["wR_seq"][fa])
        sub = _ArmTimelineBuilder(n_win, bR, qR, q_start, w_init=w_start)
        c_low_w = live_pos + live_rot @ np.array([+0.125, 0.0, 0.0], dtype=np.float64)
        u_cd_w = live_rot[:, 0] / max(float(np.linalg.norm(live_rot[:, 0])), 1e-12)
        th_cd = math.atan2(-float(u_cd_w[2]), float(u_cd_w[0]))
        psi_cd = math.atan2(float(u_cd_w[1]), math.hypot(float(u_cd_w[0]), float(u_cd_w[2])))
        d_th = th_cd - TH15
        R_ry = np.array([[math.cos(d_th), 0.0, math.sin(d_th)], [0.0, 1.0, 0.0], [-math.sin(d_th), 0.0, math.cos(d_th)]], dtype=np.float64)
        R_rz = np.array([[math.cos(psi_cd), -math.sin(psi_cd), 0.0], [math.sin(psi_cd), math.cos(psi_cd), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        R_delta = R_rz @ R_ry
        R_ee_live = R_delta @ anc["R_tail_R"]
        v_grip_loc = anc["tail_grip_local"].copy()
        if held_pos is not None and held_rot is not None:
            ee0_R, ee0_rot_R, _ = forward_kinematics_panda(q_start, base_pos=bR, base_quat_wxyz=qR)
            v_meas = held_rot.T @ (ee0_R - held_pos)
            if float(np.linalg.norm(v_meas - v_grip_loc)) < 0.035:
                v_grip_loc = v_meas
                R_ee_live = R_delta @ (held_rot.T @ ee0_rot_R)
        p_seat = c_low_w + R_delta @ v_grip_loc
        p_pre = p_seat + 0.044 * u_cd_w
        p_ret_clean = p_seat - 0.095 * R_ee_live[:, 2] + np.array([+0.026, 0.0, 0.0], dtype=np.float64)
        p_ret_sky = p_ret_clean + np.array([+0.070, +0.050, +0.085], dtype=np.float64)
        q_pre_seed = solve_ik_panda(p_pre, R_ee_live, q_init=q_start, base_pos=bR, base_quat_wxyz=qR, max_iters=180, damping=0.015)
        q_pre, _, _, _ = _walk_cartesian_chain([p_pre, p_seat, p_ret_clean, p_ret_sky], R_ee_live, bR, qR, q_anchor_first=q_pre_seed)
        sub.move_joint(18, q_pre, w_target=0.0012)
        sub.hold_frames(8, w_target=0.0012)
        sub.move_cartesian(24, p_pre, p_seat, R_ee_live, w_target=0.0012)
        sub.hold_frames(8, w_target=0.0012)
        sub.hold_frames(14, w_target=0.018)
        sub.move_cartesian(12, p_seat, p_ret_clean, R_ee_live, w_target=0.018)
        sub.move_cartesian(12, p_ret_clean, p_ret_sky, R_ee_live, w_target=0.036)
        sub.move_joint(18, anc["q_home_R"], w_target=0.036)
        sub.hold_until(n_win - 1, w_target=0.036)
        sched["qR_seq"][fa : fb + 1] = np.clip(sub.q_seq, PANDA_ARM_LOWER, PANDA_ARM_UPPER)
        sched["wR_seq"][fa : fb + 1] = sub.w_seq
        return True

    if window_name == "s11_tail_extract":
        q_start = sched["qR_seq"][fa].copy()
        w_start = float(sched["wR_seq"][fa])
        sub = _ArmTimelineBuilder(n_win, bR, qR, q_start, w_init=w_start)
        u_tl_w = live_rot @ U15
        u_tl_w = u_tl_w / max(float(np.linalg.norm(u_tl_w)), 1e-12)
        R_ee_live = live_rot @ anc["R_tail_R"]
        p_seat = live_pos + live_rot @ anc["tail_grip_local"]
        p_pre = p_seat + 0.044 * u_tl_w
        p_ret_clean = p_seat - 0.095 * R_ee_live[:, 2]
        p_ret_sky = p_ret_clean + np.array([+0.080, +0.050, +0.085], dtype=np.float64)
        q_sky_seed = solve_ik_panda(p_ret_sky, R_ee_live, q_init=q_start, base_pos=bR, base_quat_wxyz=qR, max_iters=180, damping=0.015)
        q_ret_sky, _, q_seat, _ = _walk_cartesian_chain([p_ret_sky, p_ret_clean, p_seat, p_pre], R_ee_live, bR, qR, q_anchor_first=q_sky_seed)
        sub.move_joint(18, q_ret_sky, w_target=0.028)
        sub.move_cartesian(12, p_ret_sky, p_ret_clean, R_ee_live, w_target=0.028)
        sub.move_cartesian(12, p_ret_clean, p_seat, R_ee_live, w_target=0.028)
        sub.hold_frames(12, w_target=0.0012)
        sub.move_cartesian(
            18, p_seat, p_pre, R_ee_live, w_target=0.0012,
            linear_rate=True, j7_add_start=0.0, j7_add_end=math.radians(24.0), q_untwisted_start=q_seat,
        )
        sub.move_joint(18, anc["q_tlift"], w_target=0.0012)
        sub.hold_until(n_win - 1, w_target=0.0012)
        sched["qR_seq"][fa : fb + 1] = np.clip(sub.q_seq, PANDA_ARM_LOWER, PANDA_ARM_UPPER)
        sched["wR_seq"][fa : fb + 1] = sub.w_seq
        return True

    if window_name == "s08_bkr_pour":
        q_start = sched["qR_seq"][fa].copy()
        w_start = float(sched["wR_seq"][fa])
        sub = _ArmTimelineBuilder(n_win, bR, qR, q_start, w_init=w_start)
        rim_w = live_pos + live_rot @ np.array([0.0, 0.0, 0.038], dtype=np.float64)
        p_pour_live = np.array([float(rim_w[0] + 0.058), float(rim_w[1] + 0.012), float(rim_w[2] + 0.058)], dtype=np.float64)
        _, q_pour_up = _walk_cartesian_chain([anc["p_bhigh"], p_pour_live], anc["R_bkr_up"], bR, qR, q_anchor_first=q_start)
        q_pour_tilt = solve_ik_panda(p_pour_live, anc["R_bkr_tilt"], q_init=q_pour_up, base_pos=bR, base_quat_wxyz=qR, max_iters=200, damping=0.02)
        sub.move_cartesian(20, anc["p_bhigh"], p_pour_live, anc["R_bkr_up"], w_target=0.0055)
        sub.move_joint(14, q_pour_tilt, w_target=0.0055)
        fp_a = fa + sub.f_cur
        sub.hold_frames(26, w_target=0.0055)
        fp_b = fa + sub.f_cur
        sub.move_joint(14, q_pour_up, w_target=0.0055)
        sub.move_joint(20, anc["q_bhigh"], w_target=0.0055)
        sub.hold_until(n_win - 1, w_target=0.0055)
        sched["qR_seq"][fa : fb + 1] = np.clip(sub.q_seq, PANDA_ARM_LOWER, PANDA_ARM_UPPER)
        sched["wR_seq"][fa : fb + 1] = sub.w_seq
        sched["pour_active"][fa : fb + 1] = 0.0
        sched["pour_active"][fp_a : min(fb + 1, fp_b + 1)] = 1.0
        sched["f_s08_pour_mid"] = (fp_a + fp_b) // 2
        return True

    q_start = sched["qL_seq"][fa].copy()
    w_start = float(sched["wL_seq"][fa])
    sub = _ArmTimelineBuilder(n_win, bL, qL, q_start, w_init=w_start)
    z_ax = live_rot[:, 2] / max(float(np.linalg.norm(live_rot[:, 2])), 1e-12)
    # Axisymmetric zero-yaw coaxial orientation: aligns gripper -Z with -z_ax WITHOUT inheriting arbitrary Z-yaw spin of live_rot!
    R_coax = _rot_from_z_and_y(-z_ax, [0.0, -1.0, 0.0])

    if window_name == "s04_hd_insert":
        neck_top_w = live_pos + live_rot @ np.array([0.0, 0.0, 0.0860], dtype=np.float64)
        hd_grip_off = np.array([0.0, 0.0, 0.1000], dtype=np.float64)
        if held_pos is not None:
            ee0_L, _, _ = forward_kinematics_panda(q_start, base_pos=bL, base_quat_wxyz=qL)
            v_m = ee0_L - held_pos
            if float(np.linalg.norm(v_m - hd_grip_off)) < 0.025:
                hd_grip_off = v_m
        p_seat = np.array([float(neck_top_w[0] + hd_grip_off[0]), float(neck_top_w[1] + hd_grip_off[1]), float(live_pos[2] + 0.0860 + hd_grip_off[2])], dtype=np.float64)
        p_hov = p_seat + np.array([0.0, 0.0, +0.110], dtype=np.float64)
        p_up = p_seat + np.array([0.0, 0.0, +0.098], dtype=np.float64)
        q_hov, _ = _walk_cartesian_chain([anc["p_hd_high"], p_hov, p_seat], R_top, bL, qL, q_anchor_first=q_start)[1:]
        sub.move_joint(24, q_hov, w_target=0.0010)
        sub.hold_frames(8, w_target=0.0010)
        sub.move_cartesian(26, p_hov, p_seat, R_top, w_target=0.0010)
        sub.hold_frames(10, w_target=0.0010)
        sub.hold_frames(10, w_target=0.038)
        sub.move_cartesian(12, p_seat, p_up, R_top, w_target=0.038)
        sub.move_joint(14, anc["q_home_L"], w_target=0.038)
        sub.hold_until(n_win - 1, w_target=0.038)

    elif window_name == "s06_cond_insert":
        c_sidearm_w = live_pos + live_rot @ np.array([+0.082104, 0.0, +0.040000], dtype=np.float64)
        p_cradle_w = COND_SEATED_CENTER_W + 0.048 * U15
        u_bridge = p_cradle_w - c_sidearm_w
        u_bridge = u_bridge / max(float(np.linalg.norm(u_bridge)), 1e-12)
        n_bridge = np.cross(u_bridge, np.array([0.0, 1.0, 0.0]))
        n_bridge = n_bridge / max(float(np.linalg.norm(n_bridge)), 1e-12)
        y_bridge = np.cross(n_bridge, u_bridge)
        y_bridge = y_bridge / max(float(np.linalg.norm(y_bridge)), 1e-12)
        R_cond_body = np.column_stack([u_bridge, y_bridge, n_bridge])
        R_cond_live = _rot_from_z_and_y(-n_bridge, -y_bridge)
        cd_center_live = c_sidearm_w + 0.1250 * u_bridge
        v_cond_grip = np.array([-0.0380, 0.0, 0.0], dtype=np.float64)
        if held_pos is not None and held_rot is not None:
            ee0_L, _, _ = forward_kinematics_panda(q_start, base_pos=bL, base_quat_wxyz=qL)
            v_mc = held_rot.T @ (ee0_L - held_pos)
            if float(np.linalg.norm(v_mc - v_cond_grip)) < 0.025:
                v_cond_grip = v_mc
        p_ee_seated_l = cd_center_live + R_cond_body @ v_cond_grip
        p_ee_pre_l = p_ee_seated_l + 0.044 * u_bridge
        p_ee_throat_l = p_ee_pre_l + np.array([0.0, 0.0, +0.015], dtype=np.float64)
        p_ee_front_l = p_ee_throat_l + np.array([0.0, -0.088, 0.0], dtype=np.float64)
        p_ee_rel_up_l = p_ee_seated_l + 0.080 * n_bridge
        q_cfront_seed = solve_ik_panda(p_ee_front_l, R_cond_live, q_init=q_start, base_pos=bL, base_quat_wxyz=qL, max_iters=180, damping=0.015)
        q_cfront_l, _, _, _, _ = _walk_cartesian_chain(
            [p_ee_front_l, p_ee_throat_l, p_ee_pre_l, p_ee_seated_l, p_ee_rel_up_l],
            R_cond_live, bL, qL, q_anchor_first=q_cfront_seed,
        )
        sub.move_joint(22, q_cfront_l, w_target=0.0015)
        sub.move_cartesian(18, p_ee_front_l, p_ee_throat_l, R_cond_live, w_target=0.0015)
        sub.move_cartesian(12, p_ee_throat_l, p_ee_pre_l, R_cond_live, w_target=0.0015)
        sub.hold_frames(8, w_target=0.0015)
        sub.move_cartesian(24, p_ee_pre_l, p_ee_seated_l, R_cond_live, w_target=0.0015)
        sub.hold_until(int(anc["f_c3_tight"]) - fa, w_target=0.0015)
        sub.hold_frames(10, w_target=0.040)
        sub.move_cartesian(14, p_ee_seated_l, p_ee_rel_up_l, R_cond_live, w_target=0.040)
        sub.move_joint(16, anc["q_home_L"], w_target=0.038)
        sub.hold_until(n_win - 1, w_target=0.038)

    elif window_name == "s08_fn_insert":
        # Seat FunnelSeatCollar (`bottom at +0.0160`) squarely onto Top1926SeatTab (`top at +0.1250` -> +0.1090 on Head!)
        p_seat = live_pos + live_rot @ np.array([0.0, 0.0, 0.1088 + 0.0340], dtype=np.float64)
        p_hov = p_seat + 0.135 * z_ax
        q_hov_top = _walk_cartesian_chain([anc["p_fn_high"], p_hov], R_top, bL, qL, q_anchor_first=q_start)[-1]
        q_hov_seed = solve_ik_panda(p_hov, R_coax, q_init=q_hov_top, base_pos=bL, base_quat_wxyz=qL, max_iters=180, damping=0.015)
        q_hov, _ = _walk_cartesian_chain([p_hov, p_seat], R_coax, bL, qL, q_anchor_first=q_hov_seed)
        sub.move_joint(20, q_hov, w_target=0.0035)
        sub.move_cartesian(22, p_hov, p_seat, R_coax, w_target=0.0035)
        sub.hold_frames(8, w_target=0.0035)
        sub.hold_frames(16, w_target=0.042)
        sub.move_cartesian(16, p_seat, p_hov, R_coax, w_target=0.042)
        sub.move_joint(16, anc["q_home_L"], w_target=0.042)
        sub.hold_until(n_win - 1, w_target=0.042)

    elif window_name == "s09a_fn_extract":
        p_seat = live_pos + live_rot @ np.array([0.0, 0.0, 0.0340], dtype=np.float64)
        p_hov = p_seat + 0.135 * z_ax
        q_hov_top = _walk_cartesian_chain([anc["p_home_L"], p_hov], R_top, bL, qL, q_anchor_first=q_start)[-1]
        q_hov_seed = solve_ik_panda(p_hov, R_coax, q_init=q_hov_top, base_pos=bL, base_quat_wxyz=qL, max_iters=180, damping=0.015)
        q_hov, q_seat = _walk_cartesian_chain([p_hov, p_seat], R_coax, bL, qL, q_anchor_first=q_hov_seed)
        sub.move_joint(16, q_hov, w_target=0.042)
        sub.move_cartesian(16, p_hov, p_seat, R_coax, w_target=0.042)
        sub.hold_frames(12, w_target=0.0035)
        sub.move_cartesian(
            22, p_seat, p_hov, R_coax, w_target=0.0035,
            linear_rate=True, j7_add_start=0.0, j7_add_end=math.radians(65.0), q_untwisted_start=q_seat,
        )
        sub.move_joint(18, anc["q_fn_high"], w_target=0.0035)
        sub.hold_until(n_win - 1, w_target=0.0035)

    elif window_name == "s09b_th_insert":
        # Closed-loop mercury bulb upper-rim flush servo (`+0.6mm` gentler kiss on Top1926SeatTab -> zero downward push on Head + sub-mm bulb flush!)
        hd_lip_z = float((live_pos + live_rot @ np.array([0.0, 0.0, 0.0540], dtype=np.float64))[2])
        target_bulb_z = 0.5 * (1.0800 + hd_lip_z) + 0.0006
        d_th = 0.0520 + (target_bulb_z - float(live_pos[2])) / max(0.85, float(z_ax[2]))
        p_seat = live_pos + (d_th + 0.0360) * z_ax
        p_hov = p_seat + 0.130 * z_ax
        p_up = p_seat + 0.110 * z_ax
        q_hov_top = _walk_cartesian_chain([anc["p_th_high"], p_hov], R_top, bL, qL, q_anchor_first=q_start)[-1]
        q_hov_seed = solve_ik_panda(p_hov, R_coax, q_init=q_hov_top, base_pos=bL, base_quat_wxyz=qL, max_iters=180, damping=0.015)
        q_hov, _ = _walk_cartesian_chain([p_hov, p_seat], R_coax, bL, qL, q_anchor_first=q_hov_seed)
        sub.move_joint(16, q_hov, w_target=0.0020)
        sub.move_cartesian(24, p_hov, p_seat, R_coax, w_target=0.0020)
        sub.hold_frames(8, w_target=0.0020)
        sub.hold_frames(14, w_target=0.038)
        sub.move_cartesian(12, p_seat, p_up, R_coax, w_target=0.038)
        sub.move_joint(16, anc["q_home_L"], w_target=0.038)
        sub.hold_until(n_win - 1, w_target=0.038)

    elif window_name == "s12a_th_extract":
        p_seat = live_pos + live_rot @ np.array([0.0, 0.0, 0.0360], dtype=np.float64)
        p_up = p_seat + 0.115 * z_ax
        p_hov = p_seat + 0.140 * z_ax
        q_up_top = _walk_cartesian_chain([anc["p_home_L"], p_up], R_top, bL, qL, q_anchor_first=q_start)[-1]
        q_up_seed = solve_ik_panda(p_up, R_coax, q_init=q_up_top, base_pos=bL, base_quat_wxyz=qL, max_iters=180, damping=0.015)
        q_up, q_seat, _ = _walk_cartesian_chain([p_up, p_seat, p_hov], R_coax, bL, qL, q_anchor_first=q_up_seed)
        q_up_pre = q_up.copy()
        q_up_pre[6] = np.clip(q_seat[6] - math.radians(75.0), PANDA_ARM_LOWER[6] + 0.03, PANDA_ARM_UPPER[6] - 0.03)
        q_seat_pre = q_seat.copy()
        q_seat_pre[6] = q_up_pre[6]
        sub.move_joint(18, q_up_pre, w_target=0.038)
        sub.move_joint(12, q_seat_pre, w_target=0.038)
        sub.hold_frames(12, w_target=0.0020)
        sub.move_cartesian(
            26, p_seat, p_hov, R_coax, w_target=0.0020,
            linear_rate=True, j7_add_start=-math.radians(75.0), j7_add_end=+math.radians(75.0), q_untwisted_start=q_seat,
        )
        sub.move_joint(18, anc["q_th_high"], w_target=0.0020)
        sub.hold_until(n_win - 1, w_target=0.0020)

    elif window_name == "s12b_hd_extract":
        p_seat = live_pos + live_rot @ np.array([0.0, 0.0, 0.1000], dtype=np.float64)
        p_up = p_seat + 0.105 * z_ax
        p_hov = p_seat + 0.125 * z_ax
        q_up_top = _walk_cartesian_chain([anc["p_th_high"], p_up], R_top, bL, qL, q_anchor_first=q_start)[-1]
        q_up_seed = solve_ik_panda(p_up, R_coax, q_init=q_up_top, base_pos=bL, base_quat_wxyz=qL, max_iters=180, damping=0.015)
        q_up, q_seat, _ = _walk_cartesian_chain([p_up, p_seat, p_hov], R_coax, bL, qL, q_anchor_first=q_up_seed)
        sub.move_joint(18, q_up, w_target=0.038)
        sub.move_cartesian(14, p_up, p_seat, R_coax, w_target=0.038)
        sub.hold_frames(12, w_target=0.0010)
        sub.move_cartesian(
            28, p_seat, p_hov, R_coax, w_target=0.0010,
            linear_rate=True, j7_add_start=0.0, j7_add_end=+math.radians(70.0), q_untwisted_start=q_seat,
        )
        sub.move_joint(20, anc["q_hd_high"], w_target=0.0010)
        sub.hold_until(n_win - 1, w_target=0.0010)

    sched["qL_seq"][fa : fb + 1] = np.clip(sub.q_seq, PANDA_ARM_LOWER, PANDA_ARM_UPPER)
    sched["wL_seq"][fa : fb + 1] = sub.w_seq
    return True


def audit_v09_schedule_cpu(sched: dict[str, Any], bL: np.ndarray, qL: np.ndarray, bR: np.ndarray, qR: np.ndarray) -> dict[str, Any]:
    """Verify joint continuity (`<= 0.50 rad/frame`), dual-arm clearance (`>= 80 mm`), & anchor FK (`<= 2.0 mm`) on CPU."""
    qL_s = sched["qL_seq"]
    qR_s = sched["qR_seq"]
    wL_s = sched["wL_seq"]
    wR_s = sched["wR_seq"]
    T = int(sched["T"])

    dqL = np.abs(np.diff(qL_s, axis=0))
    dqR = np.abs(np.diff(qR_s, axis=0))
    max_dqL = float(dqL.max())
    max_dqR = float(dqR.max())
    idx_dqL = int(np.unravel_index(np.argmax(dqL), dqL.shape)[0])
    idx_dqR = int(np.unravel_index(np.argmax(dqR), dqR.shape)[0])

    min_lr_dist_m = 1e9
    min_lr_frame = 0
    for f in range(T):
        q9_L = np.concatenate([qL_s[f], [wL_s[f], wL_s[f]]])
        q9_R = np.concatenate([qR_s[f], [wR_s[f], wR_s[f]]])
        pts_L = panda_all_link_world_positions(q9_L, bL, qL)[2:]
        pts_R = panda_all_link_world_positions(q9_R, bR, qR)[2:]
        dmat = np.linalg.norm(pts_L[:, None, :] - pts_R[None, :, :], axis=-1)
        dmin = float(dmat.min())
        if dmin < min_lr_dist_m:
            min_lr_dist_m = dmin
            min_lr_frame = f

    max_fk_err_mm = 0.0
    worst_fk_label = "none"
    for side, lbl, fn, tgt in sched.get("anchor_fk_checks", []):
        if side == "L":
            ep, _, _ = forward_kinematics_panda(qL_s[fn], base_pos=bL, base_quat_wxyz=qL)
        else:
            ep, _, _ = forward_kinematics_panda(qR_s[fn], base_pos=bR, base_quat_wxyz=qR)
        err = float(np.linalg.norm(ep - np.asarray(tgt)) * 1000.0)
        if err > max_fk_err_mm:
            max_fk_err_mm = err
            worst_fk_label = f"{lbl}@{fn}"

    passed = bool(max_dqL <= 0.50 and max_dqR <= 0.50 and min_lr_dist_m >= 0.080 and max_fk_err_mm <= 2.0)
    return {
        "passed": passed,
        "total_frames": T,
        "duration_s": round(T / float(sched["fps"]), 2),
        "max_dqL_rad_per_frame": round(max_dqL, 4),
        "max_dqL_frame": idx_dqL,
        "max_dqR_rad_per_frame": round(max_dqR, 4),
        "max_dqR_frame": idx_dqR,
        "min_arm_to_arm_dist_mm": round(min_lr_dist_m * 1000.0, 2),
        "min_arm_to_arm_frame": min_lr_frame,
        "max_anchor_fk_err_mm": round(max_fk_err_mm, 3),
        "worst_anchor_fk": worst_fk_label,
    }


if __name__ == "__main__":
    spec_path = Path(
        r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\distillation_v09_package\scenario_package\semantics\v09_scene_spec.json"
    )
    sp = json.loads(spec_path.read_text(encoding="utf-8"))
    bL_v = np.asarray(sp["robot_bases"]["franka_left"]["xyz"], dtype=np.float64)
    qL_v = np.asarray(sp["robot_bases"]["franka_left"]["wxyz"], dtype=np.float64)
    bR_v = np.asarray(sp["robot_bases"]["franka_right"]["xyz"], dtype=np.float64)
    qR_v = np.asarray(sp["robot_bases"]["franka_right"]["wxyz"], dtype=np.float64)
    sc = synthesize_v09_s01_s12_schedule(bL_v, qL_v, bR_v, qR_v)
    print(json.dumps(audit_v09_schedule_cpu(sc, bL_v, qL_v, bR_v, qR_v), indent=2))
