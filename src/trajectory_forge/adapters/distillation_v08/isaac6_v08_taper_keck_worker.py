"""100%-Full-Dynamic PhysX 5 GPU (240 Hz TGS) Worker for distillation_v08:
1) 4-Standard-Taper Ground-Glass Coaxial Insertion & Articulated POM Keck Clip Slice (+ mu=0.0 Counterfactuals)
2) Method A (Rigid 16-Stave Funnel) vs Method B (16-Stave Funnel + 4 Internal Micro-Spring Wedge Spheres) on -15° Unclamped Slope
3) 3-Way Bidirectional Pull-Out / Disassembly Physics Comparison (`Keck-Locked Pull` vs `Method B Straight Pull` vs `Method B Twist-and-Pull`)

NON-NEGOTIABLE ANTI-CHEAT INVARIANTS:
1. `kinematicEnabled = False` on EVERY inserted/extracted glass instrument, EVERY articulated green POM Keck clip link
   (`ClipSpine`, `LeftSpringJaw`, `RightSpringJaw`), EVERY internal radial micro-spring wedge pad (`WedgePad_0..3`),
   EVERY movable clamp/wingnut link, the wrist axial load-cell dynamometer, and ALL `48` dynamic hose links!
2. Zero weld joints, zero teleports, zero scripted object poses!
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("ISAACSIM_PORTABLE_ROOT", r"I:\isaacsim-data\6.0.1")
os.environ.setdefault("TEMP", r"I:\isaacsim-data\6.0.1\temp")
os.environ.setdefault("TMP", r"I:\isaacsim-data\6.0.1\temp")

_SRC_ROOT = str(Path(__file__).resolve().parents[3])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from trajectory_forge.adapters.distillation_v06.cosserat_hose_solver_v06 import skin_cosserat_tube  # noqa: E402
from trajectory_forge.sdk.panda_kinematics import (  # noqa: E402
    PANDA_ARM_LOWER,
    PANDA_ARM_UPPER,
    PANDA_HOME_Q,
    forward_kinematics_panda,
    solve_ik_panda,
)

FFMPEG_FALLBACK = Path(r"G:\my_program\ffmpeg-2023-06-04-git-b1c3d81e71-full_build\bin\ffmpeg.exe")

TH15 = math.radians(15.0)
U15 = np.array([math.cos(TH15), 0.0, -math.sin(TH15)], dtype=np.float64)
N15 = np.array([math.sin(TH15), 0.0, +math.cos(TH15)], dtype=np.float64)
RY15 = np.array(
    [[math.cos(TH15), 0.0, math.sin(TH15)], [0.0, 1.0, 0.0], [-math.sin(TH15), 0.0, math.cos(TH15)]],
    dtype=np.float64,
)
Q15_WXYZ = (math.cos(0.5 * TH15), 0.0, math.sin(0.5 * TH15), 0.0)

HEAD_SIDEARM_CONE_W = np.array([-0.097896, 0.0, 1.066000], dtype=np.float64)
COND_SEATED_CENTER_W = np.array([+0.022844, 0.0, 1.033648], dtype=np.float64)
COND_LOWER_CONE_W = np.array([+0.143585, 0.0, 1.001296], dtype=np.float64)


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = map(float, a)
    bw, bx, by, bz = map(float, b)
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _apply_below_normal_priority() -> None:
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        except Exception:
            pass


def _find_ffmpeg() -> str:
    if FFMPEG_FALLBACK.is_file():
        return str(FFMPEG_FALLBACK)
    w = shutil.which("ffmpeg")
    if w:
        return w
    import imageio_ffmpeg  # type: ignore

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def _encode_h264_mp4(frames_rgb: list[np.ndarray], out_mp4: Path, fps: int = 30) -> None:
    from PIL import Image  # type: ignore

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = _find_ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for idx, fr in enumerate(frames_rgb):
            Image.fromarray(fr, mode="RGB").save(td_p / f"f_{idx:05d}.png", compress_level=1)
        subprocess.check_call(
            [
                ffmpeg_bin,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(td_p / "f_%05d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_mp4),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


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


def _step_cartesian_ik(
    pos: Sequence[float],
    rot: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    q_prev: np.ndarray,
) -> np.ndarray:
    return solve_ik_panda(
        pos, rot, q_init=q_prev, base_pos=base_pos, base_quat_wxyz=base_quat,
        max_iters=220, pos_tol=1e-5, rot_tol=1e-4, damping=0.015,
    )


def _solve_exact_ik(
    pos: Sequence[float],
    rot: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    q_seed: np.ndarray | None = None,
) -> np.ndarray:
    if q_seed is not None:
        q = solve_ik_panda(
            pos, rot, q_init=q_seed, base_pos=base_pos, base_quat_wxyz=base_quat,
            max_iters=220, pos_tol=1e-5, rot_tol=1e-4, damping=0.016,
        )
        fp, fr, _ = forward_kinematics_panda(q, base_pos=base_pos, base_quat_wxyz=base_quat)
        if float(np.linalg.norm(fp - np.asarray(pos))) < 1.5e-3 and float(np.linalg.norm(fr - rot)) < 2.5e-2:
            return q
    best_q = PANDA_HOME_Q.copy()
    best_e = 1e9
    valid_q: list[np.ndarray] = []
    for s1 in (-0.75, 0.0, 0.75):
        for s7 in (-0.85, 0.0, 0.85):
            q0 = PANDA_HOME_Q.copy()
            q0[0] = s1
            q0[6] = s7
            q = solve_ik_panda(
                pos, rot, q_init=q0, base_pos=base_pos, base_quat_wxyz=base_quat,
                max_iters=260, pos_tol=1e-5, rot_tol=1e-4, damping=0.015,
            )
            fp, fr, _ = forward_kinematics_panda(q, base_pos=base_pos, base_quat_wxyz=base_quat)
            pe = float(np.linalg.norm(fp - np.asarray(pos)))
            re = float(np.linalg.norm(fr - rot))
            if pe < 1e-3 and re < 1e-2:
                valid_q.append(q)
            e = pe * 1000.0 + re * 50.0
            if e < best_e:
                best_e, best_q = e, q
    if q_seed is not None and valid_q:
        return min(valid_q, key=lambda qc: float(np.linalg.norm(qc - q_seed)))
    return best_q


def _walk_line_ik(
    p_a: np.ndarray,
    p_b: np.ndarray,
    rot: np.ndarray,
    bp: np.ndarray,
    bq: np.ndarray,
    q_start: np.ndarray,
    n_steps: int = 10,
) -> np.ndarray:
    q = q_start.copy()
    for s in np.linspace(0.0, 1.0, n_steps + 1)[1:]:
        q = _step_cartesian_ik((1.0 - s) * p_a + s * p_b, rot, bp, bq, q)
    return q


def _interp_waypoints_ik(
    wps: list[tuple[int, np.ndarray, tuple[np.ndarray, np.ndarray] | None, np.ndarray | None]],
    bp: np.ndarray,
    bq: np.ndarray,
    T: int,
) -> np.ndarray:
    """Strictly C0/C1 continuous waypoint interpolator carrying live `q_curr` across every segment boundary."""
    q_seq = np.repeat(wps[0][1][None, :], T, axis=0)
    q_curr = wps[0][1].copy()
    for idx in range(len(wps) - 1):
        fa, _, _, _ = wps[idx]
        fb, qb, seg_pts, seg_R = wps[idx + 1]
        q_start = q_curr.copy()
        q_prev = q_start.copy()
        is_hold = (seg_pts is None) and (float(np.max(np.abs(qb - wps[idx][1]))) < 1e-6)
        for f in range(fa, fb + 1):
            u = _min_jerk((f - fa) / float(max(1, fb - fa)))
            if seg_pts is not None and seg_R is not None:
                pf = (1.0 - u) * seg_pts[0] + u * seg_pts[1]
                q_prev = _step_cartesian_ik(pf, seg_R, bp, bq, q_prev)
                q_seq[f] = q_prev
            elif is_hold:
                q_seq[f] = q_start
            else:
                q_seq[f] = (1.0 - u) * q_start + u * qb
        q_curr = q_seq[fb].copy()
    return q_seq


def _lookat_matrix4d(Gf: Any, eye: Sequence[float], target: Sequence[float]) -> Any:
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fwd = (t - e) / max(float(np.linalg.norm(t - e)), 1e-12)
    zc = -fwd
    xc = np.cross(up, zc)
    xc = xc / max(float(np.linalg.norm(xc)), 1e-12)
    yc = np.cross(zc, xc)
    return Gf.Matrix4d(
        float(xc[0]), float(xc[1]), float(xc[2]), 0.0,
        float(yc[0]), float(yc[1]), float(yc[2]), 0.0,
        float(zc[0]), float(zc[1]), float(zc[2]), 0.0,
        float(e[0]), float(e[1]), float(e[2]), 1.0,
    )


def _norm_rgb(raw: Any) -> np.ndarray:
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    arr = np.asarray(raw)[:, :, :3]
    if arr.dtype == np.uint8:
        return arr.copy()
    f = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if f.size and float(np.percentile(f, 99.0)) <= 2.0:
        f = f * 255.0
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


def _find_op(xf: Any, op_name: str, op_type: Any = None) -> Any:
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == op_name or (op_type is not None and op_type == op.GetOpType()):
            return op
    return None


def _bind_panda_visual_ops(stage: Any, UsdGeom: Any, Gf: Any, root_prim_path: str) -> dict[str, Any]:
    link_paths = [
        f"{root_prim_path}/panda_link0" + "".join(f"/panda_link{m}" for m in range(1, j + 1))
        for j in range(1, 8)
    ]
    orient_ops, base_orients = [], []
    for lp in link_paths:
        prim = stage.GetPrimAtPath(lp)
        op = _find_op(UsdGeom.Xformable(prim), "xformOp:orient", UsdGeom.XformOp.TypeOrient)
        orient_ops.append(op)
        base_orients.append(Gf.Quatd(op.Get()))
    hand_p = link_paths[-1] + "/panda_hand"
    lf_op = _find_op(UsdGeom.Xformable(stage.GetPrimAtPath(hand_p + "/panda_leftfinger")), "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
    rf_op = _find_op(UsdGeom.Xformable(stage.GetPrimAtPath(hand_p + "/panda_rightfinger")), "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
    return {"orient_ops": orient_ops, "base_orients": base_orients, "lf_op": lf_op, "rf_op": rf_op}


def _apply_panda_visual_q7_and_live_fingers(arm_ops: dict[str, Any], q7: np.ndarray, lf_m: float, rf_m: float, Gf: Any) -> None:
    for j in range(7):
        half = 0.5 * float(q7[j])
        qz = Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half)))
        arm_ops["orient_ops"][j].Set(arm_ops["base_orients"][j] * qz)
    arm_ops["lf_op"].Set(Gf.Vec3d(0.0, float(lf_m), 0.0584))
    arm_ops["rf_op"].Set(Gf.Vec3d(0.0, -float(rf_m), 0.0584))


def _upsample_catmull_centerline(ctrl_pts: np.ndarray, n_out: int = 64) -> np.ndarray:
    n_in = len(ctrl_pts)
    s_in = np.linspace(0.0, 1.0, n_in)
    s_out = np.linspace(0.0, 1.0, n_out)
    out = np.column_stack([np.interp(s_out, s_in, ctrl_pts[:, d]) for d in range(3)])
    for _ in range(3):
        sm = out.copy()
        out[2:-2] = 0.50 * sm[2:-2] + 0.25 * (sm[1:-3] + sm[3:-1])
    return out


def _synthesize_v08_schedule(
    joint_id: str,
    bL: np.ndarray,
    qL: np.ndarray,
    bR: np.ndarray,
    qR: np.ndarray,
    *,
    variant: str = "nominal",
    fps: int = 30,
) -> dict[str, Any]:
    """Synthesize pure 7-DOF robot arm + gripper target schedules (`ZERO` object trajectories!)."""
    home_qL_std = _solve_exact_ik([-0.330, 0.230, 1.040], _rot_from_z_and_y([0.2, -0.95, -0.2], [1, 0, 0]), bL, qL)
    home_qR_std = _solve_exact_ik([+0.300, 0.230, 1.045], _rot_from_z_and_y([-0.2, -0.95, -0.2], [1, 0, 0]), bR, qR)

    if joint_id == "joint_01_head_into_flask_24_29_vertical":
        T = 240
        wL_seq = np.full((T,), 0.038, dtype=np.float64)
        wR_seq = np.full((T,), 0.036, dtype=np.float64)
        qR_seq = np.repeat(home_qR_std[None, :], T, axis=0)
        phase_labels = ["Standby"] * T

        # Singularity-free gentle -17° pitch (`joint5 != 0` throughout lift!) + pure axial approach `-0.085 * z_dir`!
        z_dir = np.array([0.320, -0.900, -0.29597], dtype=np.float64)
        z_dir = z_dir / np.linalg.norm(z_dir)
        y_dir = np.array([0.900, +0.320, 0.0], dtype=np.float64)
        y_dir = y_dir / np.linalg.norm(y_dir)
        R_col = _rot_from_z_and_y(z_dir, y_dir)

        p_pick = np.array([-0.335, -0.115, 0.819], dtype=np.float64)
        p_pick_pre = p_pick - 0.085 * z_dir
        p_lift_mid = np.array([-0.335, -0.115, 0.965], dtype=np.float64)
        p_lift = np.array([-0.335, -0.115, 1.112], dtype=np.float64)
        p_hover = np.array([-0.180, 0.000, 1.112], dtype=np.float64)     # Head z=1.078 -> plug tip z=1.064 (+23mm above rim 1.041!)
        p_seat = np.array([-0.180, 0.000, 1.0600], dtype=np.float64)     # Head z=1.0260 -> 29mm full taper seating + micro-spring wedge!
        p_ret_back = p_seat - 0.085 * z_dir + np.array([0.0, 0.0, +0.020], dtype=np.float64)
        p_home_L = np.array([-0.335, +0.065, 1.045], dtype=np.float64)

        q_pick = _solve_exact_ik(p_pick, R_col, bL, qL)
        q_pre = _walk_line_ik(p_pick, p_pick_pre, R_col, bL, qL, q_pick, 8)
        q_home_L = _walk_line_ik(p_pick_pre, p_home_L, R_col, bL, qL, q_pre, 12)
        q_lmid = _walk_line_ik(p_pick, p_lift_mid, R_col, bL, qL, q_pick, 10)
        q_lift = _walk_line_ik(p_lift_mid, p_lift, R_col, bL, qL, q_lmid, 10)
        q_hov = _walk_line_ik(p_lift, p_hover, R_col, bL, qL, q_lift, 12)
        q_seat = _walk_line_ik(p_hover, p_seat, R_col, bL, qL, q_hov, 10)
        q_ret_b = _walk_line_ik(p_seat, p_ret_back, R_col, bL, qL, q_seat, 8)
        q_end_L = _walk_line_ik(p_ret_back, p_home_L, R_col, bL, qL, q_ret_b, 12)

        wps_L = [
            (0, q_home_L, None, None),
            (18, q_pre, (p_home_L, p_pick_pre), R_col),
            (30, q_pick, (p_pick_pre, p_pick), R_col),
            (44, q_pick, None, None),
            (66, q_lift, (p_pick, p_lift), R_col),
            (84, q_hov, (p_lift, p_hover), R_col),
            (104, q_hov, None, None),
            (152, q_seat, (p_hover, p_seat), R_col),
            (168, q_seat, None, None),
            (180, q_seat, None, None),
            (198, q_ret_b, (p_seat, p_ret_back), R_col),
            (222, q_end_L, (p_ret_back, p_home_L), R_col),
            (239, q_end_L, None, None),
        ]
        qL_seq = _interp_waypoints_ik(wps_L, bL, qL, T)

        for f in range(0, 30):
            phase_labels[f] = "Approach Staged 75° Distillation Head"
        for f in range(30, 44):
            wL_seq[f] = 0.038 + (0.0030 - 0.038) * _min_jerk((f - 30) / 13.0)
            phase_labels[f] = "Frictional Column Squeeze (~72 N)"
        for f in range(44, 84):
            wL_seq[f] = 0.0030
            phase_labels[f] = "Lift & Move Above Flask 24/29 Socket"
        for f in range(84, 104):
            wL_seq[f] = 0.0030
            phase_labels[f] = "Coaxial 0.0mm Hover Alignment (+23mm)"
        for f in range(104, 168):
            wL_seq[f] = 0.0030
            phase_labels[f] = "Axial 24/29 Taper Push & Elastic Wedging"
        for f in range(168, 180):
            wL_seq[f] = 0.0030 + (0.038 - 0.0030) * _min_jerk((f - 168) / 11.0)
            phase_labels[f] = "Release Wedged Taper Joint"
        for f in range(180, T):
            wL_seq[f] = 0.038
            phase_labels[f] = "2.0s Self-Locked Taper Hold Proof"

        key_frames = {"pick_squeeze": 44, "coaxial_hover": 96, "mid_insertion": 132, "seated_taper": 165, "post_release": 232}

    elif joint_id == "joint_02_condenser_to_sidearm_15deg_keck":
        T = 340
        wL_seq = np.full((T,), 0.038, dtype=np.float64)
        wR_seq = np.full((T,), 0.028, dtype=np.float64)
        phase_labels = ["Standby"] * T

        R_cond_pick = _rot_from_z_and_y([0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
        th_comp = math.radians(19.2)  # +4.2° silicone-pad elastic shear compensation under 290mm condenser + 48 live hoses -> 15.0° true tube pitch!
        ry_comp = np.array([[math.cos(th_comp), 0.0, math.sin(th_comp)], [0.0, 1.0, 0.0], [-math.sin(th_comp), 0.0, math.cos(th_comp)]], dtype=np.float64)
        R_cond_15 = ry_comp @ R_cond_pick

        p_cpick = np.array([0.020 - 0.036, -0.105, 0.790], dtype=np.float64)
        p_cpick_pre = np.array([0.020 - 0.036, -0.105, 0.875], dtype=np.float64)
        p_chigh = np.array([0.020 - 0.036, -0.105, 1.055], dtype=np.float64)

        p_ee_seated = COND_SEATED_CENTER_W - 0.043 * U15 - 0.0025 * N15
        p_ee_pre_mate = p_ee_seated + 0.046 * U15
        # Approach from +44mm directly ABOVE `y=0.000` (`along +N15`) with 100% axial separation from 3-prong clamp (`+0.048 * U15`)!
        p_ee_front = p_ee_pre_mate + 0.044 * N15
        p_ee_rel_up = p_ee_seated + 0.068 * N15

        R_keck_pick = _rot_from_z_and_y([0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
        R_keck_15 = RY15 @ R_keck_pick
        p_kpick = np.array([-0.310 - 0.0110, -0.115, 0.785 + 0.0270], dtype=np.float64)
        p_kpick_pre = p_kpick + np.array([0.0, 0.0, +0.070], dtype=np.float64)
        p_khov = HEAD_SIDEARM_CONE_W - 0.0110 * U15 + 0.058 * N15
        p_ksnap = HEAD_SIDEARM_CONE_W - 0.0110 * U15 + 0.0265 * N15
        p_kret = HEAD_SIDEARM_CONE_W - 0.0110 * U15 + 0.078 * N15

        q_cp_pre = _solve_exact_ik(p_cpick_pre, R_cond_pick, bL, qL)
        q_cp = _walk_line_ik(p_cpick_pre, p_cpick, R_cond_pick, bL, qL, q_cp_pre, 6)
        q_ch = _walk_line_ik(p_cpick, p_chigh, R_cond_pick, bL, qL, q_cp, 8)
        q_cf = _solve_exact_ik(p_ee_front, R_cond_15, bL, qL, q_ch)
        q_cpre = _walk_line_ik(p_ee_front, p_ee_pre_mate, R_cond_15, bL, qL, q_cf, 8)
        q_cseat = _walk_line_ik(p_ee_pre_mate, p_ee_seated, R_cond_15, bL, qL, q_cpre, 8)
        q_cup = _walk_line_ik(p_ee_seated, p_ee_rel_up, R_cond_15, bL, qL, q_cseat, 6)
        q_kp_pre = _solve_exact_ik(p_kpick_pre, R_keck_pick, bL, qL, q_cup)
        q_kp = _walk_line_ik(p_kpick_pre, p_kpick, R_keck_pick, bL, qL, q_kp_pre, 6)
        q_kh = _solve_exact_ik(p_khov, R_keck_15, bL, qL, q_kp_pre)
        q_ks = _walk_line_ik(p_khov, p_ksnap, R_keck_15, bL, qL, q_kh, 6)
        q_kr = _walk_line_ik(p_ksnap, p_kret, R_keck_15, bL, qL, q_ks, 8)

        wps_L2 = [
            (0, home_qL_std, None, None),
            (14, q_cp_pre, None, None),
            (24, q_cp, (p_cpick_pre, p_cpick), R_cond_pick),
            (36, q_cp, None, None),
            (54, q_ch, (p_cpick, p_chigh), R_cond_pick),
            (70, q_cf, None, None),
            (84, q_cpre, (p_ee_front, p_ee_pre_mate), R_cond_15),
            (98, q_cpre, None, None),
            (124, q_cseat, (p_ee_pre_mate, p_ee_seated), R_cond_15),
            (214, q_cseat, None, None),
            (222, q_cseat, None, None),
            (232, q_cup, (p_ee_seated, p_ee_rel_up), R_cond_15),
            (246, q_kp_pre, None, None),
            (256, q_kp, (p_kpick_pre, p_kpick), R_keck_pick),
            (265, q_kp, None, None),
            (274, q_kp_pre, (p_kpick, p_kpick_pre), R_keck_pick),
            (286, q_kh, None, None),
            (302, q_ks, (p_khov, p_ksnap), R_keck_15),
            (310, q_ks, None, None),
            (324, q_kr, (p_ksnap, p_kret), R_keck_15),
            (339, q_kr, None, None),
        ]
        qL_seq = _interp_waypoints_ik(wps_L2, bL, qL, T)

        for f in range(0, 24):
            phase_labels[f] = "Approach Condenser + Live Hoses"
        for f in range(24, 36):
            wL_seq[f] = 0.038 + (0.0035 - 0.038) * _min_jerk((f - 24) / 11.0)
            phase_labels[f] = "Frictional Jacket Squeeze"
        for f in range(36, 84):
            wL_seq[f] = 0.0035
            phase_labels[f] = "Lift & Pitch 15° Coaxial to Sidearm"
        for f in range(84, 98):
            wL_seq[f] = 0.0035
            phase_labels[f] = "15° Coaxial Pre-Mate Hover (+40mm)"
        for f in range(98, 125):
            wL_seq[f] = 0.0035
            phase_labels[f] = "15° Coaxial 24/29 Taper Insertion!"
        for f in range(125, 214):
            wL_seq[f] = 0.0035
            phase_labels[f] = "Bimanual 3-Prong Clamp Screw Locking"
        for f in range(214, 222):
            wL_seq[f] = 0.0035 + (0.040 - 0.0035) * _min_jerk((f - 214) / 7.0)
            phase_labels[f] = "Release Locked Condenser"
        for f in range(222, 256):
            wL_seq[f] = 0.036
            phase_labels[f] = "Fetch Articulated Green Keck Clip"
        for f in range(256, 265):
            wL_seq[f] = 0.036 + (0.0015 - 0.036) * _min_jerk((f - 256) / 8.0)
            phase_labels[f] = "Frictional Grip Keck Dorsal Tab"
        for f in range(265, 286):
            wL_seq[f] = 0.0015
            phase_labels[f] = "Align Green Keck Clip at 15° Joint"
        for f in range(286, 303):
            wL_seq[f] = 0.0015
            phase_labels[f] = "Radial Snap-Lock Keck Clip C-Jaws!"
        for f in range(303, 310):
            wL_seq[f] = 0.0015 + (0.038 - 0.0015) * _min_jerk((f - 303) / 6.0)
            phase_labels[f] = "Release Snapped Keck Clip"
        for f in range(310, T):
            wL_seq[f] = 0.040
            phase_labels[f] = "Hands-Off Taper + Keck + Hoses Hold"

        clamp3_base = COND_SEATED_CENTER_W + 0.048 * U15 - 0.0038 * N15
        wing3_world = clamp3_base + RY15 @ np.array([0.0, 0.056, 0.018], dtype=np.float64)
        wing3_pre = wing3_world + np.array([0.0, 0.080, 0.0], dtype=np.float64)
        R_wing3 = _rot_from_z_and_y([0.0, -1.0, 0.0], [math.cos(TH15), 0.0, -math.sin(TH15)])
        qR3_pre = _solve_exact_ik(wing3_pre, R_wing3, bR, qR)
        qR3_base = _step_cartesian_ik(wing3_world, R_wing3, bR, qR, qR3_pre)
        j7_c = float(np.clip(qR3_base[6], -0.4, 0.6))
        j7_hi = j7_c + math.radians(60.0)
        j7_lo = j7_c - math.radians(60.0)
        qR3_hi_ex = qR3_base.copy()
        qR3_hi_ex[6] = j7_hi
        _, rR3_hi_ref, _ = forward_kinematics_panda(qR3_hi_ex, base_pos=bR, base_quat_wxyz=qR)
        qR3_pre_hi = _step_cartesian_ik(wing3_pre, rR3_hi_ref, bR, qR, qR3_hi_ex)
        p_home_R = wing3_pre + np.array([+0.165, +0.085, +0.025], dtype=np.float64)
        home_qR2 = _walk_line_ik(wing3_pre, p_home_R, rR3_hi_ref, bR, qR, qR3_pre_hi, 12)
        home_qR2 = _step_cartesian_ik(p_home_R, rR3_hi_ref, bR, qR, home_qR2)
        qR_seq = np.repeat(home_qR2[None, :], T, axis=0)

        q_prev_ra = home_qR2.copy()
        for f in range(74, 114):
            u = _min_jerk((f - 74) / 39.0)
            q_prev_ra = _step_cartesian_ik((1.0 - u) * p_home_R + u * wing3_pre, rR3_hi_ref, bR, qR, q_prev_ra)
            qR_seq[f] = q_prev_ra
        q_prev_rw = qR3_pre_hi.copy()
        for f in range(114, 124):
            u = _min_jerk((f - 114) / 9.0)
            q_prev_rw = _step_cartesian_ik((1.0 - u) * wing3_pre + u * wing3_world, rR3_hi_ref, bR, qR, q_prev_rw)
            qR_seq[f] = q_prev_rw
        qR_hi = q_prev_rw.copy()
        qR_lo = qR_hi.copy()
        qR_lo[6] = j7_lo

        for cyc, f0 in enumerate((124, 168)):
            for f in range(f0, f0 + 8):
                qR_seq[f] = qR_hi
                wR_seq[f] = 0.028 + (0.0010 - 0.028) * _min_jerk((f - f0) / 7.0)
            for f in range(f0 + 8, f0 + 28):
                u = _min_jerk((f - (f0 + 8)) / 19.0)
                qR_seq[f] = qR_hi.copy()
                qR_seq[f, 6] = (1.0 - u) * j7_hi + u * j7_lo
                wR_seq[f] = 0.0010
            for f in range(f0 + 28, f0 + 34):
                qR_seq[f] = qR_lo
                wR_seq[f] = 0.0010 + (0.028 - 0.0010) * _min_jerk((f - (f0 + 28)) / 5.0)
            for f in range(f0 + 34, f0 + 44):
                u = _min_jerk((f - (f0 + 34)) / 9.0)
                if cyc == 0:
                    qR_seq[f] = qR_lo.copy()
                    qR_seq[f, 6] = (1.0 - u) * j7_lo + u * j7_hi
                else:
                    qR_seq[f] = (1.0 - u) * qR_lo + u * qR3_pre_hi
                wR_seq[f] = 0.028
        for f in range(212, T):
            u = _min_jerk(min(1.0, (f - 212) / 36.0))
            qR_seq[f] = (1.0 - u) * qR3_pre_hi + u * home_qR2
            wR_seq[f] = 0.038

        key_frames = {"coaxial_hover": 92, "taper_inserted": 124, "clamp_twisted": 195, "keck_snap_push": 300, "post_release": 332}

    elif joint_id in ("joint_03_tail_adapter_to_condenser_keck", "ab_05_unclamped_15deg_hold"):
        use_keck = joint_id == "joint_03_tail_adapter_to_condenser_keck"
        T = 280 if use_keck else 210
        wL_seq = np.full((T,), 0.036, dtype=np.float64)
        wR_seq = np.full((T,), 0.036, dtype=np.float64)
        phase_labels = ["Standby"] * T

        # Natural right-side-up ArmR orientation (`z_ee = [0, -0.88, -0.475]`, `y_ee = [+1, 0, 0]` -> `< 0.16mm` IK residual, 100% interior limits!)
        tail_grip_local = np.array([+0.056, 0.0, -0.024], dtype=np.float64)
        z_dir_R = np.array([0.0, -0.880, -0.47497], dtype=np.float64)
        z_dir_R = z_dir_R / np.linalg.norm(z_dir_R)
        y_dir_R = np.array([+1.0, 0.0, 0.0], dtype=np.float64)
        R_tail_R = _rot_from_z_and_y(z_dir_R, y_dir_R)

        p_t_staged = np.array([+0.280, -0.105, 0.855], dtype=np.float64)
        p_tpick = p_t_staged + tail_grip_local
        p_tpick_pre = p_tpick - 0.080 * z_dir_R
        p_tlift = p_tpick + np.array([0.0, 0.0, +0.165], dtype=np.float64)
        p_t_seated_ee = COND_LOWER_CONE_W + tail_grip_local
        p_t_pre_ee = p_t_seated_ee + 0.038 * U15
        # Pure axial retract along `-z_dir_R` (`toward +Y/+Z`, 100% clear of `TailGripStem` so open fingers NEVER knock the seated tail!)
        p_t_ret_clean = p_t_seated_ee - 0.095 * z_dir_R
        p_t_ret_high = p_t_ret_clean + np.array([+0.055, +0.065, +0.055], dtype=np.float64)
        p_home_R3 = p_tpick + np.array([+0.010, +0.135, +0.145], dtype=np.float64)

        qR_tp = _solve_exact_ik(p_tpick, R_tail_R, bR, qR)
        qR_tpre = _walk_line_ik(p_tpick, p_tpick_pre, R_tail_R, bR, qR, qR_tp, 6)
        qR_home = _walk_line_ik(p_tpick_pre, p_home_R3, R_tail_R, bR, qR, qR_tpre, 10)
        qR_tlift = _walk_line_ik(p_tpick, p_tlift, R_tail_R, bR, qR, qR_tp, 8)
        qR_tmpre = _walk_line_ik(p_tlift, p_t_pre_ee, R_tail_R, bR, qR, qR_tlift, 10)
        qR_tseat = _walk_line_ik(p_t_pre_ee, p_t_seated_ee, R_tail_R, bR, qR, qR_tmpre, 8)
        qR_tret1 = _walk_line_ik(p_t_seated_ee, p_t_ret_clean, R_tail_R, bR, qR, qR_tseat, 8)
        qR_tret2 = _walk_line_ik(p_t_ret_clean, p_t_ret_high, R_tail_R, bR, qR, qR_tret1, 8)

        f_hold_end = 208 if use_keck else 132
        f_rel_done = f_hold_end + 10
        wps_R3 = [
            (0, qR_home, None, None),
            (16, qR_tpre, (p_home_R3, p_tpick_pre), R_tail_R),
            (28, qR_tp, (p_tpick_pre, p_tpick), R_tail_R),
            (40, qR_tp, None, None),
            (56, qR_tlift, (p_tpick, p_tlift), R_tail_R),
            (72, qR_tmpre, (p_tlift, p_t_pre_ee), R_tail_R),
            (88, qR_tmpre, None, None),
            (116, qR_tseat, (p_t_pre_ee, p_t_seated_ee), R_tail_R),
            (f_hold_end, qR_tseat, None, None),
            (f_rel_done, qR_tseat, None, None),
            (f_rel_done + 18, qR_tret1, (p_t_seated_ee, p_t_ret_clean), R_tail_R),
            (f_rel_done + 38, qR_tret2, (p_t_ret_clean, p_t_ret_high), R_tail_R),
            (T - 1, qR_tret2, None, None),
        ]
        qR_seq = _interp_waypoints_ik(wps_R3, bR, qR, T)

        for f in range(0, 28):
            wR_seq[f] = 0.026
        for f in range(28, 40):
            wR_seq[f] = 0.026 + (0.0020 - 0.026) * _min_jerk((f - 28) / 11.0)
        for f in range(40, f_hold_end):
            wR_seq[f] = 0.0020
        for f in range(f_hold_end, f_rel_done):
            wR_seq[f] = 0.0020 + (0.026 - 0.0020) * _min_jerk((f - f_hold_end) / 9.0)
        for f in range(f_rel_done, T):
            wR_seq[f] = 0.026

        if use_keck:
            R_keck_pick = _rot_from_z_and_y([0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
            R_keck_15 = RY15 @ R_keck_pick
            p_k2_pick = np.array([-0.035 - 0.0110, -0.175, 0.785 + 0.0270], dtype=np.float64)
            p_k2_pre = p_k2_pick + np.array([0.0, 0.0, +0.085], dtype=np.float64)
            p_k2_high = np.array([+0.080, -0.080, 1.115], dtype=np.float64)
            p_k2_hov = COND_LOWER_CONE_W - 0.0112 * U15 + 0.058 * N15
            p_k2_snap = COND_LOWER_CONE_W - 0.0112 * U15 + 0.0258 * N15
            p_k2_ret = COND_LOWER_CONE_W - 0.0112 * U15 + 0.095 * N15 + np.array([-0.140, +0.120, +0.030], dtype=np.float64)

            qL_kp_pre = _solve_exact_ik(p_k2_pre, R_keck_pick, bL, qL)
            qL_kp = _walk_line_ik(p_k2_pre, p_k2_pick, R_keck_pick, bL, qL, qL_kp_pre, 6)
            qL_kh1 = _solve_exact_ik(p_k2_high, R_keck_15, bL, qL, qL_kp_pre)
            qL_khov = _walk_line_ik(p_k2_high, p_k2_hov, R_keck_15, bL, qL, qL_kh1, 8)
            qL_ksnap = _walk_line_ik(p_k2_hov, p_k2_snap, R_keck_15, bL, qL, qL_khov, 6)
            qL_kret = _walk_line_ik(p_k2_snap, p_k2_ret, R_keck_15, bL, qL, qL_ksnap, 10)

            wps_L3 = [
                (0, home_qL_std, None, None),
                (85, home_qL_std, None, None),
                (104, qL_kp_pre, None, None),
                (116, qL_kp, (p_k2_pre, p_k2_pick), R_keck_pick),
                (128, qL_kp, None, None),
                (140, qL_kp_pre, (p_k2_pick, p_k2_pre), R_keck_pick),
                (156, qL_kh1, None, None),
                (170, qL_khov, (p_k2_high, p_k2_hov), R_keck_15),
                (194, qL_ksnap, (p_k2_hov, p_k2_snap), R_keck_15),
                (206, qL_ksnap, None, None),
                (216, qL_ksnap, None, None),
                (242, qL_kret, (p_k2_snap, p_k2_ret), R_keck_15),
                (279, qL_kret, None, None),
            ]
            qL_seq = _interp_waypoints_ik(wps_L3, bL, qL, T)

            for f in range(116, 128):
                wL_seq[f] = 0.036 + (0.0015 - 0.036) * _min_jerk((f - 116) / 11.0)
            for f in range(128, 206):
                wL_seq[f] = 0.0015
            for f in range(206, 216):
                wL_seq[f] = 0.0015 + (0.038 - 0.0015) * _min_jerk((f - 206) / 9.0)
            for f in range(216, T):
                wL_seq[f] = 0.038

            for f in range(0, 40):
                phase_labels[f] = "Frictional Pick Vacuum Tail Adapter"
            for f in range(40, 88):
                phase_labels[f] = "15° Coaxial Alignment to Lower Cone"
            for f in range(88, 118):
                phase_labels[f] = "15° Coaxial 24/29 Tail Taper Insertion"
            for f in range(118, 170):
                phase_labels[f] = "Fetch Second Articulated Green Keck Clip"
            for f in range(170, 206):
                phase_labels[f] = "Radial Snap-Lock Keck Clip on Tail Joint!"
            for f in range(206, 220):
                phase_labels[f] = "Full Dual-Arm Release!"
            for f in range(220, T):
                phase_labels[f] = "2.0s Mid-Air Keck Anti-Slide Hold Proof"
            key_frames = {"coaxial_hover": 82, "tail_inserted": 116, "keck_hover": 170, "keck_snapped": 200, "post_release": 272}
        else:
            qL_seq = np.repeat(home_qL_std[None, :], T, axis=0)
            m_lbl = "Method B (4-Spring Wedge)" if variant == "method_b_micro_spring" else "Method A (Rigid Funnel Only)"
            for f in range(0, 40):
                phase_labels[f] = f"[{m_lbl}] Frictional Pick Tail Adapter"
            for f in range(40, 88):
                phase_labels[f] = f"[{m_lbl}] 15° Coaxial Hover (+38mm)"
            for f in range(88, 132):
                phase_labels[f] = f"[{m_lbl}] 29mm Taper Push (NO Keck Clip!)"
            for f in range(132, 145):
                phase_labels[f] = f"[{m_lbl}] Open Gripper WITHOUT Keck Clip!"
            for f in range(145, T):
                phase_labels[f] = f"[{m_lbl}] 2.2s Unclamped -15° Gravity Test!"
            key_frames = {"coaxial_hover": 82, "taper_seated": 128, "release_moment": 144, "gravity_test_mid": 172, "post_release": 204}

    elif joint_id == "joint_04_thermometer_screw_cap_flush_bulb":
        T = 240
        wL_seq = np.full((T,), 0.038, dtype=np.float64)
        wR_seq = np.full((T,), 0.036, dtype=np.float64)
        qR_seq = np.repeat(home_qR_std[None, :], T, axis=0)
        phase_labels = ["Standby"] * T

        # Singularity-free gentle -17° pitch (`joint5 != 0` up to z=1.264!) + pure axial approach `-0.082 * z_dir4`!
        z_dir4 = np.array([0.320, -0.900, -0.29597], dtype=np.float64)
        z_dir4 = z_dir4 / np.linalg.norm(z_dir4)
        y_dir4 = np.array([0.900, +0.320, 0.0], dtype=np.float64)
        y_dir4 = y_dir4 / np.linalg.norm(y_dir4)
        R_cap = _rot_from_z_and_y(z_dir4, y_dir4)
        p_pick = np.array([-0.335, -0.115, 0.855 + 0.034], dtype=np.float64)
        p_pick_pre = p_pick - 0.082 * z_dir4
        p_high_mid = np.array([-0.335, -0.115, 1.075], dtype=np.float64)
        p_high_1 = np.array([-0.335, -0.115, 1.264], dtype=np.float64)
        p_hover = np.array([-0.180, 0.000, 1.264], dtype=np.float64)
        p_seat = np.array([-0.180, 0.000, 1.1320 + 0.034], dtype=np.float64)
        p_ret_b = p_seat - 0.082 * z_dir4 + np.array([0.0, 0.0, +0.020], dtype=np.float64)
        p_home_L4 = np.array([-0.310, 0.145, 1.130], dtype=np.float64)

        q_pick = _solve_exact_ik(p_pick, R_cap, bL, qL)
        q_pre = _walk_line_ik(p_pick, p_pick_pre, R_cap, bL, qL, q_pick, 8)
        q_home4 = _walk_line_ik(p_pick_pre, p_home_L4, R_cap, bL, qL, q_pre, 12)
        q_hmid = _walk_line_ik(p_pick, p_high_mid, R_cap, bL, qL, q_pick, 10)
        q_h1 = _walk_line_ik(p_high_mid, p_high_1, R_cap, bL, qL, q_hmid, 12)
        q_hov = _walk_line_ik(p_high_1, p_hover, R_cap, bL, qL, q_h1, 10)
        q_seat = _walk_line_ik(p_hover, p_seat, R_cap, bL, qL, q_hov, 12)
        q_ret = _walk_line_ik(p_seat, p_ret_b, R_cap, bL, qL, q_seat, 8)
        q_end = _walk_line_ik(p_ret_b, p_home_L4, R_cap, bL, qL, q_ret, 10)

        wps_L4 = [
            (0, q_home4, None, None),
            (18, q_pre, (p_home_L4, p_pick_pre), R_cap),
            (30, q_pick, (p_pick_pre, p_pick), R_cap),
            (44, q_pick, None, None),
            (72, q_h1, (p_pick, p_high_1), R_cap),
            (88, q_hov, (p_high_1, p_hover), R_cap),
            (106, q_hov, None, None),
            (160, q_seat, (p_hover, p_seat), R_cap),
            (172, q_seat, None, None),
            (184, q_seat, None, None),
            (202, q_ret, (p_seat, p_ret_b), R_cap),
            (224, q_end, (p_ret_b, p_home_L4), R_cap),
            (239, q_end, None, None),
        ]
        qL_seq = _interp_waypoints_ik(wps_L4, bL, qL, T)

        for f in range(0, 30):
            phase_labels[f] = "Approach Screw-Cap Thermometer Unit"
        for f in range(30, 44):
            wL_seq[f] = 0.038 + (0.0026 - 0.038) * _min_jerk((f - 30) / 13.0)
            phase_labels[f] = "Frictional Red Knurled Cap Squeeze (~72 N)"
        for f in range(44, 88):
            wL_seq[f] = 0.0026
            phase_labels[f] = "High Clearance Lift Above Top 19/26 Rim"
        for f in range(88, 106):
            wL_seq[f] = 0.0026
            phase_labels[f] = "Coaxial 0.0mm Bore Entry Hover"
        for f in range(106, 172):
            wL_seq[f] = 0.0026
            phase_labels[f] = "Thread Stem & Seat Bulb Flush @ 1.080m!"
        for f in range(172, 184):
            wL_seq[f] = 0.0026 + (0.038 - 0.0026) * _min_jerk((f - 172) / 11.0)
            phase_labels[f] = "Release Seated 19/26 Thermometer"
        for f in range(184, T):
            wL_seq[f] = 0.038
            phase_labels[f] = "2.0s Flush Bulb-to-Sidearm Hold Proof"

        key_frames = {"pick_squeeze": 44, "coaxial_hover": 98, "mid_threading": 134, "seated_flush": 170, "post_release": 232}

    elif joint_id == "ab_06_disassembly_pullout":
        # 3-Way Pull-Out / Disassembly Comparison on Seated 24/29 Head-in-Flask Joint:
        #   - `keck_locked_pull`: Green POM Keck clip locked -> +56mm pull blocked mechanically (<8mm), safe 22N dynamometer limit!
        #   - `method_b_straight_pull`: Unclamped Method B wedged taper -> +56mm straight pull (`dj7 = 0°`), high axial breakaway force!
        #   - `method_b_twist_and_pull`: Unclamped Method B wedged taper -> identical +56mm upward draw + `±34°` coaxial `j7` twist,
        #                                rotating the 3D Coulomb friction vector into the tangent plane -> much lower axial force & flask tug!
        T = 200
        wL_seq = np.full((T,), 0.036, dtype=np.float64)
        wR_seq = np.full((T,), 0.036, dtype=np.float64)
        qR_seq = np.repeat(home_qR_std[None, :], T, axis=0)
        phase_labels = ["Standby"] * T

        # Top-down coaxial grasp (`z_ee = [0, 0, -1]`) on `distillation_head_75deg` upper 19/26 collar (`z_local = +0.098 -> z_world = 1.124`)
        # Because `z_ee = [0, 0, -1]`, `panda_joint7` spins 100% coaxially around world `+Z` (`zero lateral bending moment!`)!
        R_top = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
        p_grip = np.array([-0.180, 0.000, 1.124], dtype=np.float64)
        p_pre = np.array([-0.180, 0.000, 1.192], dtype=np.float64)
        p_pull_top = np.array([-0.180, 0.000, 1.182], dtype=np.float64)     # +58mm vertical extraction stroke!
        p_home_top = np.array([-0.265, 0.165, 1.205], dtype=np.float64)

        q_pre = _solve_exact_ik(p_pre, R_top, bL, qL)
        q_home_t = _walk_line_ik(p_pre, p_home_top, R_top, bL, qL, q_pre, 10)
        q_grip = _walk_line_ik(p_pre, p_grip, R_top, bL, qL, q_pre, 8)
        q_pull = _walk_line_ik(p_grip, p_pull_top, R_top, bL, qL, q_grip, 12)

        if variant == "keck_locked_pull":
            # Pull up against locked Keck clip (`f=56..118`), hold tension (`f=118..144`), then relax back to seated (`f=144..168`) & release!
            wps_L6 = [
                (0, q_home_t, None, None),
                (18, q_pre, (p_home_top, p_pre), R_top),
                (34, q_grip, (p_pre, p_grip), R_top),
                (52, q_grip, None, None),
                (118, q_pull, (p_grip, p_pull_top), R_top),
                (144, q_pull, None, None),
                (166, q_grip, (p_pull_top, p_grip), R_top),
                (178, q_grip, None, None),
                (199, q_pre, (p_grip, p_pre), R_top),
            ]
            qL_seq = _interp_waypoints_ik(wps_L6, bL, qL, T)
            for f in range(34, 50):
                wL_seq[f] = 0.036 + (0.0020 - 0.036) * _min_jerk((f - 34) / 15.0)
            for f in range(50, 168):
                wL_seq[f] = 0.0020
            for f in range(168, 178):
                wL_seq[f] = 0.0020 + (0.036 - 0.0020) * _min_jerk((f - 168) / 9.0)
            for f in range(0, 52):
                phase_labels[f] = "[Keck Locked] Top Coaxial Grasp on Seated Head"
            for f in range(52, 144):
                phase_labels[f] = "[Keck Locked] Straight Upward Pull -> BLOCKED by Green Clip!"
            for f in range(144, T):
                phase_labels[f] = "[Keck Locked] Safe Force Limit Relax & Release (Locked!)"
            key_frames = {"top_grasp": 50, "pull_onset": 80, "keck_blocked_peak": 128, "relax_back": 166, "post_release": 195}
        else:
            # Two-stage linear upward extraction (`ZERO` min-jerk velocity spike!):
            #   Stage 1 (`f=56..125`, 2.30s): steady 10.4 mm/s draw across +24mm taper wedge zone (`p_grip -> p_unwedge`)
            #   Stage 2 (`f=125..165`, 1.33s): smooth +34mm clear lift above flask rim (`p_unwedge -> p_pull_top`)
            p_unwedge = np.array([-0.180, 0.000, 1.148], dtype=np.float64)
            wps_pre = [
                (0, q_home_t, None, None),
                (18, q_pre, (p_home_top, p_pre), R_top),
                (34, q_grip, (p_pre, p_grip), R_top),
                (56, q_grip, None, None),
                (T - 1, q_grip, None, None),
            ]
            qL_seq = _interp_waypoints_ik(wps_pre, bL, qL, T)
            q_prev_p = qL_seq[56].copy()
            for f in range(56, 126):
                al_p = (f - 56) / 69.0
                q_prev_p = _step_cartesian_ik((1.0 - al_p) * p_grip + al_p * p_unwedge, R_top, bL, qL, q_prev_p)
                qL_seq[f] = q_prev_p
            for f in range(126, 166):
                al_c = _min_jerk((f - 126) / 40.0)
                q_prev_p = _step_cartesian_ik((1.0 - al_c) * p_unwedge + al_c * p_pull_top, R_top, bL, qL, q_prev_p)
                qL_seq[f] = q_prev_p
            for f in range(166, T):
                qL_seq[f] = q_prev_p

            if variant == "method_b_twist_and_pull":
                # Pre-orient `j7` to `-85°` before grasp (`f=0..30`), then spin monotonically `+172°` (`-85° -> +87°`)
                # all the way through `f=51..142` (`until PlugTip completely exits the 4 micro-spring wedge pads at z=1.050m!`)!
                dj7_start = -math.radians(85.0)
                dj7_end = +math.radians(87.0)
                for f in range(0, 51):
                    qL_seq[f, 6] += dj7_start * _min_jerk(min(1.0, f / 26.0))
                for f in range(51, 143):
                    u_tw = (f - 51) / 91.0
                    qL_seq[f, 6] += (1.0 - u_tw) * dj7_start + u_tw * dj7_end
                for f in range(143, T):
                    qL_seq[f, 6] += dj7_end
            for f in range(34, 50):
                wL_seq[f] = 0.036 + (0.0020 - 0.036) * _min_jerk((f - 34) / 15.0)
            for f in range(50, T):
                wL_seq[f] = 0.0020
            tag6 = "Twist-and-Pull (±35° Coaxial)" if variant == "method_b_twist_and_pull" else "Pure Straight Pull (0° Twist)"
            for f in range(0, 52):
                phase_labels[f] = f"[{tag6}] Top Coaxial Grasp on Wedged Head"
            for f in range(52, 110):
                phase_labels[f] = f"[{tag6}] Breakaway & Un-Wedging Stroke!"
            for f in range(110, 145):
                phase_labels[f] = f"[{tag6}] Smooth Clear Extraction (+58mm)!"
            for f in range(145, T):
                phase_labels[f] = f"[{tag6}] Cleanly Disassembled Above Flask!"
            key_frames = {"top_grasp": 50, "breakaway_peak": 76, "mid_extract": 106, "clear_above_rim": 144, "post_release": 192}
    else:
        raise ValueError(f"Unknown joint_id: {joint_id}")

    return {
        "T": T,
        "fps": fps,
        "qL_seq": np.clip(qL_seq, PANDA_ARM_LOWER, PANDA_ARM_UPPER),
        "wL_seq": wL_seq,
        "qR_seq": np.clip(qR_seq, PANDA_ARM_LOWER, PANDA_ARM_UPPER),
        "wR_seq": wR_seq,
        "phase_labels": phase_labels,
        "key_frames": key_frames,
    }


def _decorate_frame_hud(
    rgb: np.ndarray,
    *,
    Image: Any,
    ImageDraw: Any,
    f_title: Any,
    f_mono: Any,
    header: str,
    sub: str,
    badge_color: tuple[int, int, int],
) -> np.ndarray:
    h, w = rgb.shape[:2]
    rgba = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    dr.rounded_rectangle([(10, 10), (w - 10, 68)], radius=8, fill=(10, 18, 32, 204), outline=(*badge_color, 210), width=2)
    dr.text((20, 15), header, fill=(248, 250, 252, 255), font=f_title)
    dr.text((20, 41), sub, fill=(191, 219, 254, 245), font=f_mono)
    return np.asarray(Image.alpha_composite(rgba, ov).convert("RGB"), dtype=np.uint8)


def execute_v08_job(job: dict[str, Any]) -> dict[str, Any]:
    _apply_below_normal_priority()
    pkg_root = Path(job["package_dir"]).resolve()
    out_dir = Path(job["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    joint_id = str(job["joint_id"])
    variant = str(job.get("variant", "zero_friction" if job.get("zero_friction") else "nominal"))
    zero_friction = bool(job.get("zero_friction", variant == "zero_friction"))
    record_video = bool(job.get("record_video", True))

    spec = json.loads((pkg_root / "semantics" / "v08_scene_spec.json").read_text(encoding="utf-8"))
    bL = np.asarray(spec["robot_bases"]["franka_left"]["xyz"], dtype=np.float64)
    qL = np.asarray(spec["robot_bases"]["franka_left"]["wxyz"], dtype=np.float64)
    bR = np.asarray(spec["robot_bases"]["franka_right"]["xyz"], dtype=np.float64)
    qR = np.asarray(spec["robot_bases"]["franka_right"]["wxyz"], dtype=np.float64)

    sched = _synthesize_v08_schedule(joint_id, bL, qL, bR, qR, variant=variant, fps=30)

    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    from isaacsim import SimulationApp

    spp = 6 if (record_video and not zero_friction) else (3 if record_video else 1)
    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RayTracedLighting" if record_video else "MinimalRendering",
            "samples_per_pixel_per_frame": spp,
            "denoiser": bool(record_video),
            "anti_aliasing": 3 if record_video else 0,
            "fast_shutdown": True,
            "extra_args": [
                "--/app/asyncRendering=False",
                "--/physics/updateToUsd=True",
                "--/physics/suppressReadback=False",
                "--/rtx/hydra/materialSyncLoads=True",
                "--/rtx/translucency/enabled=True",
                "--/rtx/reflections/enabled=True",
                "--/rtx/translucency/maxRefractionBounces=16",
                "--/rtx/raytracing/maxBounces=16",
                "--/rtx-transient/dlssg/enabled=False",
                "--/persistent/omni/replicator/captureOnPlay=False",
                "--/omni/replicator/captureOnPlay=False",
            ],
        }
    )

    import omni.replicator.core as rep
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from omni.physx.scripts import physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

    try:
        f_title = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 15)
        f_mono = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 12)
    except Exception:
        f_title = f_mono = ImageFont.load_default()

    usd_path = pkg_root / "scene" / "main.usda"
    omni.usd.get_context().open_stage(str(usd_path))
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    rep.orchestrator.set_capture_on_play(False)

    pscene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    ps_api = PhysxSchema.PhysxSceneAPI.Apply(pscene_prim)
    ps_api.CreateEnableGPUDynamicsAttr().Set(True)
    ps_api.CreateBroadphaseTypeAttr().Set("GPU")
    ps_api.CreateMaxPositionIterationCountAttr().Set(32)
    ps_api.CreateMaxVelocityIterationCountAttr().Set(8)

    for root_clean in ("/World/franka_left", "/World/franka_right", "/World/Instruments", "/World/Mechanisms"):
        rp = stage.GetPrimAtPath(root_clean)
        if rp.IsValid():
            for pr in Usd.PrimRange(rp):
                if pr.HasAPI(UsdPhysics.CollisionAPI):
                    pr.GetAttribute("physics:collisionEnabled").Set(False)
                if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                    pr.GetAttribute("physics:rigidBodyEnabled").Set(False)

    UsdGeom.Scope.Define(stage, "/World/PhysMaterials")
    UsdGeom.Scope.Define(stage, "/World/PhysWedgePads")

    def _make_phys_mat(name: str, mu_s: float, mu_d: float, combine: str = "max") -> str:
        mp = f"/World/PhysMaterials/{name}"
        m = UsdShade.Material.Define(stage, mp)
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr().Set(float(mu_s))
        api.CreateDynamicFrictionAttr().Set(float(mu_d))
        api.CreateRestitutionAttr().Set(0.0)
        pxm = PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim())
        pxm.CreateFrictionCombineModeAttr().Set(combine)
        pxm.CreateRestitutionCombineModeAttr().Set("min")
        return mp

    mu_grip_s = 0.0 if zero_friction else 3.8
    mu_grip_d = 0.0 if zero_friction else 3.5
    comb_grip = "min" if zero_friction else "max"
    pad_mat = _make_phys_mat("SiliconePadMat", mu_grip_s, mu_grip_d, comb_grip)
    clamp_mat = _make_phys_mat("ClampRubberMat", mu_grip_s, mu_grip_d, comb_grip)
    # Rigid smooth glass staves (`0.18 < tan(15°)=0.268`, so rigid unclamped socket slides off on -15° slope!)
    taper_glide_mat = _make_phys_mat("TaperGlideMat", 0.0 if zero_friction else 0.18, 0.0 if zero_friction else 0.15, "average")
    # Frosted ground-glass micro-asperity friction on Method B radial micro-spring wedge pads (`0.88` nominal, `0.0` in zero_friction)
    frosted_wedge_mat = _make_phys_mat("FrostedWedgeMat", 0.0 if zero_friction else 0.88, 0.0 if zero_friction else 0.78, comb_grip)
    glass_grip_mat = _make_phys_mat("GlassGripMat", 0.0 if zero_friction else 2.8, 0.0 if zero_friction else 2.5, comb_grip)
    keck_pom_mat = _make_phys_mat("KeckPOMMat", 0.0 if zero_friction else 0.32, 0.0 if zero_friction else 0.25, comb_grip)
    bench_mat = _make_phys_mat("BenchMat", 0.65, 0.55, "average")
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath("/World/Environment/LabBenchTop"), bench_mat)

    def _add_invisible_col(
        parent_path: str,
        cname: str,
        kind: str,
        pos: Sequence[float],
        size_or_rad: Sequence[float],
        mat_path: str,
        *,
        axis: str = "Z",
        quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> Any:
        cp = f"{parent_path}/PhysCol_{cname}"
        if kind == "box":
            g = UsdGeom.Cube.Define(stage, cp)
            g.CreateSizeAttr(1.0)
            sc = Gf.Vec3f(float(size_or_rad[0]), float(size_or_rad[1]), float(size_or_rad[2]))
        elif kind == "cylinder":
            g = UsdGeom.Cylinder.Define(stage, cp)
            g.CreateAxisAttr(axis)
            g.CreateRadiusAttr(float(size_or_rad[0]))
            g.CreateHeightAttr(float(size_or_rad[1]))
            sc = Gf.Vec3f(1.0, 1.0, 1.0)
        else:
            g = UsdGeom.Sphere.Define(stage, cp)
            g.CreateRadiusAttr(float(size_or_rad[0]))
            sc = Gf.Vec3f(1.0, 1.0, 1.0)
        g.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdGeom.Imageable(g).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        xf = UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))
        )
        xf.AddScaleOp().Set(sc)
        UsdPhysics.CollisionAPI.Apply(g.GetPrim()).CreateCollisionEnabledAttr(True)
        pxc = PhysxSchema.PhysxCollisionAPI.Apply(g.GetPrim())
        pxc.CreateRestOffsetAttr().Set(0.0)
        pxc.CreateContactOffsetAttr().Set(0.0010)
        physicsUtils.add_physics_material_to_prim(stage, g.GetPrim(), mat_path)
        return g.GetPrim()

    def _config_dynamic_rb(
        prim_path: str,
        mass_kg: float,
        lin_damp: float = 1.2,
        ang_damp: float = 1.8,
        com_xyz: Sequence[float] | None = None,
    ) -> Any:
        pr = stage.GetPrimAtPath(prim_path)
        rb = UsdPhysics.RigidBodyAPI.Apply(pr)
        rb.CreateRigidBodyEnabledAttr().Set(True)
        rb.CreateKinematicEnabledAttr().Set(False)
        mapi = UsdPhysics.MassAPI.Apply(pr)
        mapi.CreateMassAttr().Set(float(mass_kg))
        if com_xyz is not None:
            mapi.CreateCenterOfMassAttr().Set(Gf.Vec3f(float(com_xyz[0]), float(com_xyz[1]), float(com_xyz[2])))
        pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(pr)
        pxrb.CreateSolverPositionIterationCountAttr().Set(32)
        pxrb.CreateSolverVelocityIterationCountAttr().Set(8)
        pxrb.CreateMaxDepenetrationVelocityAttr().Set(0.35)
        pxrb.CreateLinearDampingAttr().Set(float(lin_damp))
        pxrb.CreateAngularDampingAttr().Set(float(ang_damp))
        pxrb.CreateSleepThresholdAttr().Set(0.0)
        pxrb.CreateEnableCCDAttr().Set(True)
        return pr

    def _set_pose(prim_path: str, xyz: Sequence[float], wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)) -> None:
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
        _find_op(xf, "xformOp:translate", UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(*map(float, xyz)))
        _find_op(xf, "xformOp:orient", UsdGeom.XformOp.TypeOrient).Set(Gf.Quatd(*map(float, wxyz)))

    def _add_16stave_concave_socket_z(
        parent_path: str,
        tag: str,
        z_center: float,
        length: float,
        r_inner_mid: float,
        flare_deg: float = 3.2,
        add_floor: bool = True,
        add_rolled_rim: bool = False,
        skip_wedge_quadrants: bool = False,
    ) -> None:
        """Build a 100% hollow conical female socket around local +Z (`12-stave windowed` when Method B pads are embedded)."""
        wall_th = 0.0024
        r_mid = r_inner_mid + 0.5 * wall_th
        flare_rad = math.radians(flare_deg)
        for i in range(16):
            if skip_wedge_quadrants and (i in (2, 6, 10, 14)):
                continue
            ang = 2.0 * math.pi * i / 16.0
            cz, sz = math.cos(0.5 * ang), math.sin(0.5 * ang)
            cy, sy = math.cos(0.5 * flare_rad), math.sin(0.5 * flare_rad)
            qw = cz * cy
            qx = -sz * sy
            qy = cz * sy
            qz = sz * cy
            _add_invisible_col(
                parent_path,
                f"{tag}_Stave_{i:02d}",
                "box",
                (r_mid * math.cos(ang), r_mid * math.sin(ang), z_center),
                (wall_th, 2.0 * math.pi * r_inner_mid / 16.0 * 0.90, length),
                taper_glide_mat,
                quat_wxyz=(qw, qx, qy, qz),
            )
        if add_rolled_rim:
            # 8 hollow outer rolled-rim segments (`r_in = r_inner_mid + 0.8mm`, `r_out = r_inner_mid + 4.0mm`) so bore stays 100% open!
            r_rim_mid = r_inner_mid + 0.0024
            z_rim = z_center + 0.5 * length - 0.0025
            for ri in range(8):
                ang = 2.0 * math.pi * ri / 8.0
                _add_invisible_col(
                    parent_path,
                    f"{tag}_RolledRim_{ri}",
                    "box",
                    (r_rim_mid * math.cos(ang), r_rim_mid * math.sin(ang), z_rim),
                    (0.0032, 0.0078, 0.0030),
                    keck_pom_mat,
                    quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)),
                )
        if add_floor:
            _add_invisible_col(
                parent_path,
                f"{tag}_SeatStop",
                "box",
                (0.0, 0.0, z_center - 0.5 * length - 0.0020),
                (2.2 * r_inner_mid, 2.2 * r_inner_mid, 0.0040),
                taper_glide_mat,
            )

    def _add_16stave_concave_socket_x(
        parent_path: str,
        tag: str,
        x_center: float,
        length: float,
        r_inner_mid: float,
        mouth_sign: float,
        rot_y_deg: float = 0.0,
        add_rolled_rim: bool = True,
        skip_wedge_quadrants: bool = False,
    ) -> str:
        """Build a 100% hollow conical female socket around local X (`wall_th = 2.4 mm`, NO solid plug!)."""
        sub_xf_p = f"{parent_path}/SocketScope_{tag}"
        sxf = UsdGeom.Xform.Define(stage, sub_xf_p)
        sxf.AddTranslateOp().Set(Gf.Vec3d(float(x_center), 0.0, 0.0))
        ry_r = math.radians(rot_y_deg)
        sxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(0.5 * ry_r), 0.0, math.sin(0.5 * ry_r), 0.0))
        wall_th = 0.0024
        r_mid = r_inner_mid + 0.5 * wall_th
        for i in range(16):
            if skip_wedge_quadrants and (i in (2, 6, 10, 14)):
                continue
            ang = 2.0 * math.pi * i / 16.0
            cy_p, sy_p = math.cos(ang), math.sin(ang)
            qx_h, sx_h = math.cos(0.5 * ang), math.sin(0.5 * ang)
            _add_invisible_col(
                sub_xf_p,
                f"Stave_{i:02d}",
                "box",
                (0.0, r_mid * cy_p, r_mid * sy_p),
                (length, wall_th, 2.0 * math.pi * r_inner_mid / 16.0 * 0.90),
                taper_glide_mat,
                quat_wxyz=(qx_h, sx_h, 0.0, 0.0),
            )
        if add_rolled_rim:
            # Hollow 8-segment outer rolled rim at mouth (`x_sub = mouth_sign * (0.5*length - 0.0025)`) for Keck clip axial locking!
            r_rim_mid = r_inner_mid + 0.0024
            x_rim = mouth_sign * (0.5 * length - 0.0025)
            for ri in range(8):
                ang = 2.0 * math.pi * ri / 8.0
                _add_invisible_col(
                    sub_xf_p,
                    f"RolledRim_{ri}",
                    "box",
                    (x_rim, r_rim_mid * math.cos(ang), r_rim_mid * math.sin(ang)),
                    (0.0030, 0.0032, 0.0078),
                    keck_pom_mat,
                    quat_wxyz=(math.cos(0.5 * ang), math.sin(0.5 * ang), 0.0, 0.0),
                )
        _add_invisible_col(
            sub_xf_p,
            "AxialStop",
            "box",
            (-mouth_sign * (0.5 * length + 0.0020), 0.0, 0.0),
            (0.0040, 2.2 * r_inner_mid, 2.2 * r_inner_mid),
            taper_glide_mat,
        )
        return sub_xf_p

    wedge_pad_joints: list[dict[str, Any]] = []
    manipulated_prims: list[str] = []

    def _add_method_b_micro_spring_wedge_pads(
        host_body_path: str,
        tag: str,
        *,
        socket_axis: str,
        axial_offset: float,
        r_inner_preload: float,
        rot_y_deg: float = 0.0,
        stiffness: float = 3400.0,
        damping: float = 48.0,
        max_force: float = 12.5,
    ) -> list[str]:
        """Method B: 4 internal radially-compliant dynamic sphere wedge pads (`kinematicEnabled = False`!).

        Models the true circumferential elastic wedging ("stacked paper cups" / ground-glass micro-compliance)
        with overdamped `d = 48 N*s/m` and capped `maxForce = 12.5 N` so PhysX never pops or chatters!
        """
        r_sphere = 0.0040
        r_eq_center = r_inner_preload + r_sphere
        host_prim = stage.GetPrimAtPath(host_body_path)
        host_is_rb = host_prim.HasAPI(UsdPhysics.RigidBodyAPI) and bool(
            host_prim.GetAttribute("physics:rigidBodyEnabled").Get()
        )
        mw_host = UsdGeom.Xformable(host_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        q_host = mw_host.ExtractRotationQuat()
        im_h = q_host.GetImaginary()
        q_host_tup = (float(q_host.GetReal()), float(im_h[0]), float(im_h[1]), float(im_h[2]))

        ry_r = math.radians(rot_y_deg)
        q_ry = (math.cos(0.5 * ry_r), 0.0, math.sin(0.5 * ry_r), 0.0)
        R_ry = np.array(
            [[math.cos(ry_r), 0.0, math.sin(ry_r)], [0.0, 1.0, 0.0], [-math.sin(ry_r), 0.0, math.cos(ry_r)]],
            dtype=np.float64,
        )

        created_pads: list[str] = []
        for pi in range(4):
            ang = math.radians(45.0 + 90.0 * pi)
            ca, sa = math.cos(ang), math.sin(ang)
            if socket_axis == "Z":
                p_loc = np.array([r_eq_center * ca, r_eq_center * sa, axial_offset], dtype=np.float64)
                # Joint local +X points radially outward in X-Y plane!
                q_rad_loc = (math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang))
            else:
                # Socket axis is X (tilted by `rot_y_deg`): radial outward is in Y-Z plane (`R_x(ang) @ Ry(-90°)`? Even simpler:
                # use joint axis "Y" rotated around X by `ang` so +Y points along `[0, cos(ang), sin(ang)]`!)
                p_sub = np.array([0.0, r_eq_center * ca, r_eq_center * sa], dtype=np.float64)
                p_loc = np.array([axial_offset, 0.0, 0.0], dtype=np.float64) + R_ry @ p_sub
                q_rx = (math.cos(0.5 * ang), math.sin(0.5 * ang), 0.0, 0.0)
                q_rad_loc = _quat_mul(q_ry, q_rx)

            pw = mw_host.Transform(Gf.Vec3d(float(p_loc[0]), float(p_loc[1]), float(p_loc[2])))
            q_pad_w = _quat_mul(q_host_tup, q_rad_loc)

            pad_p = f"/World/PhysWedgePads/{tag}_WedgePad_{pi}"
            pxf = UsdGeom.Xform.Define(stage, pad_p)
            pxf.AddTranslateOp().Set(Gf.Vec3d(float(pw[0]), float(pw[1]), float(pw[2])))
            pxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*q_pad_w))
            _config_dynamic_rb(pad_p, 0.004, lin_damp=4.0, ang_damp=4.0)
            _add_invisible_col(pad_p, "Sphere", "sphere", (0.0, 0.0, 0.0), (r_sphere,), frosted_wedge_mat)
            created_pads.append(pad_p)

            pj = UsdPhysics.PrismaticJoint.Define(stage, f"/World/PhysWedgePads/{tag}_WedgeJoint_{pi}")
            pj.CreateAxisAttr("X" if socket_axis == "Z" else "Y")
            if host_is_rb:
                pj.CreateBody0Rel().SetTargets([Sdf.Path(host_body_path)])
                pj.CreateLocalPos0Attr().Set(Gf.Vec3f(float(p_loc[0]), float(p_loc[1]), float(p_loc[2])))
                pj.CreateLocalRot0Attr().Set(Gf.Quatf(*map(float, q_rad_loc)))
            else:
                pj.CreateLocalPos0Attr().Set(Gf.Vec3f(float(pw[0]), float(pw[1]), float(pw[2])))
                pj.CreateLocalRot0Attr().Set(Gf.Quatf(*map(float, q_pad_w)))
            pj.CreateBody1Rel().SetTargets([Sdf.Path(pad_p)])
            pj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            pj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            pj.CreateLowerLimitAttr().Set(-0.0002)
            pj.CreateUpperLimitAttr().Set(+0.0025)
            drv = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
            drv.CreateStiffnessAttr().Set(float(stiffness))
            drv.CreateDampingAttr().Set(float(damping))
            drv.CreateMaxForceAttr().Set(float(max_force))
            drv.CreateTargetPositionAttr().Set(0.0)
            wedge_pad_joints.append(
                {
                    "pad_prim": pad_p,
                    "host_prim": host_body_path if host_is_rb else "",
                    "p_rest_w": np.array([float(pw[0]), float(pw[1]), float(pw[2])], dtype=np.float64),
                    "p_loc": p_loc.copy(),
                    "stiffness": float(stiffness),
                    "max_force": float(max_force),
                }
            )
        manipulated_prims.extend(created_pads)
        return created_pads

    def _set_world_joint_frame0(joint: Any, parent_path: str, local_xyz: Sequence[float]) -> None:
        mw = UsdGeom.Xformable(stage.GetPrimAtPath(parent_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pw = mw.Transform(Gf.Vec3d(*map(float, local_xyz)))
        qv = mw.ExtractRotationQuat()
        im = qv.GetImaginary()
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(pw[0]), float(pw[1]), float(pw[2])))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(qv.GetReal()), float(im[0]), float(im[1]), float(im[2])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def _build_bench_pedestal(name: str, xyz: Sequence[float], size_xyz: Sequence[float]) -> None:
        p = f"/World/Environment/{name}"
        c = UsdGeom.Cube.Define(stage, p)
        c.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        xf.AddScaleOp().Set(Gf.Vec3f(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(
            UsdShade.Material(stage.GetPrimAtPath("/World/Looks/BenchTrimDark"))
        )
        UsdPhysics.CollisionAPI.Apply(c.GetPrim()).CreateCollisionEnabledAttr(True)
        physicsUtils.add_physics_material_to_prim(stage, c.GetPrim(), bench_mat)

    def _configure_dynamic_keck_clip(
        keck_root: str,
        staged_xyz: Sequence[float],
        pedestal_name: str = "",
        staged_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> list[str]:
        """Configure all 3 links (`ClipSpine`, `LeftSpringJaw`, `RightSpringJaw`) of an ART-001 Keck clip as 100% dynamic!

        Matches the exact stepped `24/29` geometry (`Male C-claw @ x=-19.5mm`, `Female C-claw @ x=-4.5mm`,
        `9mm` central rim-and-shoulder pocket `x = -17.0..-8.0mm`!).
        """
        if pedestal_name:
            _build_bench_pedestal(
                pedestal_name,
                (staged_xyz[0], staged_xyz[1], 0.5 * (0.750 + staged_xyz[2] - 0.014)),
                (0.044, 0.044, (staged_xyz[2] - 0.014) - 0.750),
            )
        _set_pose(keck_root, staged_xyz, staged_wxyz)

        sp_p = f"{keck_root}/ClipSpine"
        lj_p = f"{keck_root}/LeftSpringJaw"
        rj_p = f"{keck_root}/RightSpringJaw"
        _config_dynamic_rb(sp_p, 0.018, lin_damp=2.6, ang_damp=3.2, com_xyz=(-0.0110, 0.0, 0.0040))
        _config_dynamic_rb(lj_p, 0.006, lin_damp=2.2, ang_damp=2.6)
        _config_dynamic_rb(rj_p, 0.006, lin_damp=2.2, ang_damp=2.6)

        # Dorsal parallel grip tab (`width_y = 0.0100m` at (-0.011, 0, +0.027)) + stepped bridge saddle
        _add_invisible_col(sp_p, "DorsalGripTab", "box", (-0.0110, 0.0, 0.0270), (0.015, 0.0100, 0.014), glass_grip_mat)
        _add_invisible_col(sp_p, "BridgeSaddleFem", "box", (-0.0045, 0.0, 0.0168), (0.008, 0.0100, 0.0036), keck_pom_mat)
        _add_invisible_col(sp_p, "BridgeSaddleMale", "box", (-0.0195, 0.0, 0.0126), (0.006, 0.0090, 0.0036), keck_pom_mat)
        # Positive axial stops straddling the central `x = -17.0 .. -7.5mm` pocket:
        # Rigid bilateral spine ears (`cannot cam open!`, `y_inner_male = 9.7mm > r_neck 9.4mm` & `< r_shoulder 13.0mm`;
        # `y_inner_fem = 14.8mm >= r_socket 14.6mm` & `< r_rolled_rim 17.4mm`)!
        x_mstop = -0.0185 if ("joint_03" in joint_id) else -0.0180
        _add_invisible_col(sp_p, "MaleAxialStopTop", "box", (x_mstop, 0.0, 0.0112), (0.0028, 0.010, 0.0040), keck_pom_mat)
        _add_invisible_col(sp_p, "FemaleRimStopTop", "box", (-0.0068, 0.0, 0.0154), (0.0028, 0.012, 0.0042), keck_pom_mat)
        if variant == "keck_locked_pull":
            for sgn_e, etag in ((+1.0, "L"), (-1.0, "R")):
                _add_invisible_col(sp_p, f"MaleSpineEar_{etag}", "box", (x_mstop, sgn_e * 0.0114, 0.000), (0.0028, 0.0032, 0.016), keck_pom_mat)
                _add_invisible_col(sp_p, f"FemSpineEar_{etag}", "box", (-0.0065, sgn_e * 0.0167, 0.000), (0.0028, 0.0034, 0.018), keck_pom_mat)

        # Stepped Dual-Diameter C-Jaws (`Male C-Claw @ x=-0.0195` & `Female C-Claw @ x=-0.0045`!)
        for j_p, sgn, jname, lims in (
            (lj_p, +1.0, "LeftJawRev", (-4.0, 32.0)),
            (rj_p, -1.0, "RightJawRev", (-32.0, 4.0)),
        ):
            # Female C-claw (`around r_outer = 14.2mm` 16-stave socket; joint hinge is at y = sgn*0.0056, z = +0.0172!)
            _add_invisible_col(j_p, "FemCheek", "box", (-0.0045, sgn * 0.0104, -0.0140), (0.0065, 0.0032, 0.012), keck_pom_mat)
            _add_invisible_col(
                j_p, "FemLowerHook", "box", (-0.0045, sgn * 0.0082, -0.0268), (0.0065, 0.0036, 0.0085), keck_pom_mat,
                quat_wxyz=(math.cos(sgn * (-0.36)), math.sin(sgn * (-0.36)), 0.0, 0.0),
            )
            _add_invisible_col(j_p, "FemAxialStopSide", "box", (-0.0074, sgn * 0.0074, -0.0172), (0.0028, 0.0042, 0.014), keck_pom_mat)

            # Male C-claw (`around r_neck = 9.4mm` male neck & symmetric 3-point catching on `r_shoulder = 11.6mm`!)
            _add_invisible_col(j_p, "MaleCheek", "box", (-0.0195, sgn * 0.0062, -0.0155), (0.0055, 0.0030, 0.010), keck_pom_mat)
            _add_invisible_col(
                j_p, "MaleLowerHook", "box", (-0.0195, sgn * 0.0046, -0.0252), (0.0055, 0.0034, 0.0080), keck_pom_mat,
                quat_wxyz=(math.cos(sgn * (-0.36)), math.sin(sgn * (-0.36)), 0.0, 0.0),
            )
            _add_invisible_col(j_p, "MaleAxialStopSide", "box", (-0.0174, sgn * 0.0048, -0.0172), (0.0028, 0.0038, 0.012), keck_pom_mat)

            rj = UsdPhysics.RevoluteJoint.Define(stage, f"{keck_root}/{jname}")
            rj.CreateAxisAttr("X")
            rj.CreateBody0Rel().SetTargets([Sdf.Path(sp_p)])
            rj.CreateBody1Rel().SetTargets([Sdf.Path(j_p)])
            rj.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, float(sgn * 0.0056), 0.0172))
            rj.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            rj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            rj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            rj.CreateLowerLimitAttr().Set(float(lims[0]))
            rj.CreateUpperLimitAttr().Set(float(lims[1]))
            jdrv = UsdPhysics.DriveAPI.Apply(rj.GetPrim(), "angular")
            jdrv.CreateStiffnessAttr().Set(320.0)
            jdrv.CreateDampingAttr().Set(12.0)
            jdrv.CreateMaxForceAttr().Set(48.0)
            jdrv.CreateTargetPositionAttr().Set(0.0)
        return [sp_p, lj_p, rj_p]

    hose_chain_xfs: dict[str, list[Any]] = {"inlet": [], "outlet": []}
    flask_compliance_prim = ""
    flask_init_z = 0.9400

    # Configure Stage-Accurate Visibility & Full-Dynamic Taper/Keck/Wedge Colliders per Slice
    if joint_id == "joint_01_head_into_flask_24_29_vertical":
        for hp in (
            "/World/Instruments/liebig_condenser_24_29",
            "/World/Instruments/vacuum_tail_adapter",
            "/World/Instruments/erlenmeyer_flask_150ml",
            "/World/Instruments/scissor_jack_white",
            "/World/Instruments/iron_stand_right",
            "/World/Instruments/thermometer_adapter_unit_v08",
            "/World/Mechanisms/condenser_clamp_3prong_art001",
            "/World/Mechanisms/keck_clip_head_cond_art001",
            "/World/Mechanisms/keck_clip_cond_tail_art001",
            "/World/CoolingHoses/Hose_Inlet_Lower",
            "/World/CoolingHoses/Hose_Outlet_Upper",
        ):
            UsdGeom.Imageable(stage.GetPrimAtPath(hp)).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        # Static Flask Female 24/29 12-Stave Windowed Socket + Method B 4-Sphere Micro-Spring Wedge (`z_local = +0.089m`)!
        flask_p = "/World/Instruments/round_bottom_flask_250ml"
        _add_16stave_concave_socket_z(flask_p, "Flask2429", z_center=0.0865, length=0.029, r_inner_mid=0.0122, flare_deg=3.5, add_rolled_rim=True, skip_wedge_quadrants=True)
        _add_method_b_micro_spring_wedge_pads(flask_p, "J01_FlaskWedge", socket_axis="Z", axial_offset=0.0900, r_inner_preload=0.0105)

        # Unscaled Bench Support Rack for Staged Distillation Head at `(-0.335, -0.115, 0.785)` (`0.00°` tilt at t=0!)
        h_rack = UsdGeom.Xform.Define(stage, "/World/Environment/HeadBenchRack")
        h_rack.AddTranslateOp().Set(Gf.Vec3d(-0.335, -0.115, 0.750))
        _add_invisible_col("/World/Environment/HeadBenchRack", "Floor", "box", (0.0, 0.0, 0.0105), (0.052, 0.052, 0.021), bench_mat)
        vis_hb = UsdGeom.Cube.Define(stage, "/World/Environment/HeadBenchRack/VisBase")
        vis_hb.CreateSizeAttr(1.0)
        vhxf = UsdGeom.Xformable(vis_hb)
        vhxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0105))
        vhxf.AddScaleOp().Set(Gf.Vec3f(0.052, 0.052, 0.021))
        UsdShade.MaterialBindingAPI.Apply(vis_hb.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/BenchTrimDark")))
        for si in range(8):
            ang = 2.0 * math.pi * si / 8.0
            _add_invisible_col(
                "/World/Environment/HeadBenchRack", f"Cup_{si}", "box",
                (0.0148 * math.cos(ang), 0.0148 * math.sin(ang), 0.032), (0.004, 0.009, 0.022), bench_mat,
                quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)),
            )

        head_p = "/World/Instruments/distillation_head_75deg"
        _set_pose(head_p, (-0.335, -0.115, 0.785), (1.0, 0.0, 0.0, 0.0))
        _config_dynamic_rb(head_p, 0.095, lin_damp=1.4, ang_damp=2.2, com_xyz=(0.0, 0.0, 0.024))
        _add_invisible_col(head_p, "PlugTip", "cylinder", (0.0, 0.0, -0.009), (0.0102, 0.010), taper_glide_mat, axis="Z")
        _add_invisible_col(head_p, "PlugRamp", "cylinder", (0.0, 0.0, -0.001), (0.0108, 0.006), frosted_wedge_mat, axis="Z")
        _add_invisible_col(head_p, "PlugMid", "cylinder", (0.0, 0.0, +0.006), (0.0114, 0.012), frosted_wedge_mat, axis="Z")
        _add_invisible_col(head_p, "GraspColumn", "cylinder", (0.0, 0.0, +0.034), (0.0126, 0.032), glass_grip_mat, axis="Z")
        manipulated_prims.append(head_p)
        target_body_prim = head_p
        target_seated_pos = np.array([-0.180, 0.000, 1.0260], dtype=np.float64)

    elif joint_id == "joint_02_condenser_to_sidearm_15deg_keck":
        for hp in (
            "/World/Instruments/vacuum_tail_adapter",
            "/World/Instruments/erlenmeyer_flask_150ml",
            "/World/Instruments/scissor_jack_white",
            "/World/Instruments/thermometer_adapter_unit_v08",
            "/World/Mechanisms/keck_clip_cond_tail_art001",
        ):
            UsdGeom.Imageable(stage.GetPrimAtPath(hp)).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        head_p = "/World/Instruments/distillation_head_75deg"
        # Two non-overlapping axial rings (`dx = -5.5mm` & `+5.5mm`, `11mm` baseline >> `6.4mm` sphere diameter -> ZERO overlap!)
        # of Method B micro-spring spheres (`r=3.2mm`, `r_outer=13.0mm` inside `r_in=12.2mm` socket) -> <2.0mm mouth err & <2.0° coaxiality!
        _add_invisible_col(
            head_p, "SidearmKeckShoulder", "cylinder",
            (0.0685 * math.cos(TH15), 0.0, 0.062 - 0.0685 * math.sin(TH15)),
            (0.0118, 0.003), keck_pom_mat, axis="X", quat_wxyz=Q15_WXYZ,
        )
        _add_invisible_col(
            head_p, "SidearmKeckNeck", "cylinder",
            (0.0615 * math.cos(TH15), 0.0, 0.062 - 0.0615 * math.sin(TH15)),
            (0.0104, 0.012), keck_pom_mat, axis="X", quat_wxyz=Q15_WXYZ,
        )
        for ring_idx, x_ax in enumerate((-0.0055, +0.0055)):
            r_out_ring = 0.0131 if ring_idx == 0 else 0.0127  # Tapered 1:10 lead-in (`front ring gentler than rear ring!`)
            r_eq_s = r_out_ring - 0.0032
            for pi in range(4):
                ang_s = math.radians(45.0 + 90.0 * pi)
                ca_s, sa_s = math.cos(ang_s), math.sin(ang_s)
                p_sub_s = np.array([x_ax, r_eq_s * ca_s, r_eq_s * sa_s], dtype=np.float64)
                pw_s = HEAD_SIDEARM_CONE_W + RY15 @ p_sub_s
                q_pad_s = _quat_mul(Q15_WXYZ, (math.cos(0.5 * ang_s), math.sin(0.5 * ang_s), 0.0, 0.0))
                pad_sp = f"/World/PhysWedgePads/J02_SidearmWedgePad_{ring_idx}_{pi}"
                pxf_s = UsdGeom.Xform.Define(stage, pad_sp)
                pxf_s.AddTranslateOp().Set(Gf.Vec3d(float(pw_s[0]), float(pw_s[1]), float(pw_s[2])))
                pxf_s.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*q_pad_s))
                _config_dynamic_rb(pad_sp, 0.016, lin_damp=4.5, ang_damp=4.5)
                _add_invisible_col(pad_sp, "Sphere", "sphere", (0.0, 0.0, 0.0), (0.0032,), frosted_wedge_mat)
                manipulated_prims.append(pad_sp)

                pj_s = UsdPhysics.PrismaticJoint.Define(stage, f"/World/PhysWedgePads/J02_SidearmWedgeJoint_{ring_idx}_{pi}")
                pj_s.CreateAxisAttr("Y")
                pj_s.CreateLocalPos0Attr().Set(Gf.Vec3f(float(pw_s[0]), float(pw_s[1]), float(pw_s[2])))
                pj_s.CreateLocalRot0Attr().Set(Gf.Quatf(*map(float, q_pad_s)))
                pj_s.CreateBody1Rel().SetTargets([Sdf.Path(pad_sp)])
                pj_s.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                pj_s.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                pj_s.CreateLowerLimitAttr().Set(-0.0025)
                pj_s.CreateUpperLimitAttr().Set(+0.0002)
                drv_s = UsdPhysics.DriveAPI.Apply(pj_s.GetPrim(), "linear")
                drv_s.CreateStiffnessAttr().Set(2200.0)
                drv_s.CreateDampingAttr().Set(42.0)
                drv_s.CreateMaxForceAttr().Set(9.0)
                drv_s.CreateTargetPositionAttr().Set(0.0)
                wedge_pad_joints.append(
                    {
                        "pad_prim": pad_sp,
                        "host_prim": "",
                        "p_rest_w": pw_s.copy(),
                        "p_loc": np.zeros((3,), dtype=np.float64),
                        "stiffness": 2200.0,
                        "max_force": 9.0,
                    }
                )

        vrest = UsdGeom.Xform.Define(stage, "/World/Environment/CondenserBenchRest")
        vrest.AddTranslateOp().Set(Gf.Vec3d(0.020, -0.105, 0.750))
        for rx, rtag in ((-0.058, "Neg"), (+0.058, "Pos")):
            _add_invisible_col("/World/Environment/CondenserBenchRest", f"R_{rtag}", "box", (rx, 0.0, 0.0119), (0.010, 0.036, 0.0238), bench_mat)
            _add_invisible_col("/World/Environment/CondenserBenchRest", f"RF_{rtag}", "box", (rx, -0.019, 0.028), (0.010, 0.006, 0.012), bench_mat)
            _add_invisible_col("/World/Environment/CondenserBenchRest", f"RR_{rtag}", "box", (rx, +0.019, 0.028), (0.010, 0.006, 0.012), bench_mat)
        _add_invisible_col("/World/Instruments/heating_stirrer_bath", "BathBody", "cylinder", (0.0, 0.0, 0.064), (0.076, 0.128), bench_mat, axis="Z")

        # Dynamic Liebig Condenser (`100%` hollow 16-stave `CondUpper2429` + 8-segment outer rolled rim — NO solid plug!)
        cond_p = "/World/Instruments/liebig_condenser_24_29"
        _set_pose(cond_p, (0.020, -0.105, 0.790), (1.0, 0.0, 0.0, 0.0))
        _config_dynamic_rb(cond_p, 0.185, lin_damp=1.2, ang_damp=1.8)
        _add_invisible_col(cond_p, "WaterJacket", "cylinder", (0.0, 0.0, 0.0), (0.0162, 0.146), glass_grip_mat, axis="X")
        _add_16stave_concave_socket_x(cond_p, "CondUpper2429", x_center=-0.125, length=0.028, r_inner_mid=0.0123, mouth_sign=-1.0, add_rolled_rim=True)
        manipulated_prims.append(cond_p)

        clamp3_root = "/World/Mechanisms/condenser_clamp_3prong_art001"
        # Place 3-prong clamp at `+0.048 * U15` (`84mm` clear of ArmL grasp at `-0.036 * U15` -> ZERO finger-to-clamp collision!)
        clamp3_pos = COND_SEATED_CENTER_W + 0.048 * U15 - 0.0038 * N15
        _set_pose(clamp3_root, clamp3_pos, Q15_WXYZ)
        _set_pose("/World/Instruments/iron_stand_right", (float(clamp3_pos[0]) + 0.0174, 0.112, 0.750), (1.0, 0.0, 0.0, 0.0))
        cbase_p = f"{clamp3_root}/FixedLowerBase"
        cjaw_p = f"{clamp3_root}/MovableUpperJaw"
        cwing_p = f"{clamp3_root}/TopButterflyWingnut"
        _add_invisible_col(cbase_p, "LowBottom", "box", (0.0, 0.0, -0.0192), (0.042, 0.014, 0.006), clamp_mat)
        _add_invisible_col(cbase_p, "LowFront", "box", (0.0, -0.0155, -0.0138), (0.042, 0.005, 0.012), clamp_mat, quat_wxyz=(math.cos(0.39), -math.sin(0.39), 0.0, 0.0))
        _add_invisible_col(cbase_p, "LowRear", "box", (0.0, +0.0155, -0.0138), (0.042, 0.005, 0.012), clamp_mat, quat_wxyz=(math.cos(0.39), +math.sin(0.39), 0.0, 0.0))

        _config_dynamic_rb(cjaw_p, 0.06, lin_damp=1.5, ang_damp=1.5)
        _add_invisible_col(cjaw_p, "UpperPadTop", "box", (0.0, -0.024, +0.0035), (0.040, 0.018, 0.008), clamp_mat)
        uj = UsdPhysics.RevoluteJoint.Define(stage, f"{clamp3_root}/UpperJawRevolute")
        uj.CreateAxisAttr("X")
        uj.CreateBody1Rel().SetTargets([Sdf.Path(cjaw_p)])
        _set_world_joint_frame0(uj, cbase_p, (0.0, 0.024, 0.014))
        uj.CreateLowerLimitAttr().Set(-46.0)
        uj.CreateUpperLimitAttr().Set(20.0)
        udrv = UsdPhysics.DriveAPI.Apply(uj.GetPrim(), "angular")
        udrv.CreateStiffnessAttr().Set(1400.0)
        udrv.CreateDampingAttr().Set(75.0)
        udrv.CreateMaxForceAttr().Set(140.0)
        udrv.CreateTargetPositionAttr().Set(-42.0)

        _config_dynamic_rb(cwing_p, 0.06, lin_damp=1.0, ang_damp=0.65)
        _add_invisible_col(cwing_p, "TopWingDrum", "cylinder", (0.0, 0.008, 0.0), (0.0070, 0.022), clamp_mat, axis="Y")
        twj = UsdPhysics.RevoluteJoint.Define(stage, f"{clamp3_root}/TopWingRevolute")
        twj.CreateAxisAttr("Y")
        twj.CreateBody1Rel().SetTargets([Sdf.Path(cwing_p)])
        _set_world_joint_frame0(twj, cbase_p, (0.0, 0.048, 0.018))
        UsdPhysics.DriveAPI.Apply(twj.GetPrim(), "angular").CreateDampingAttr().Set(0.65)
        manipulated_prims.extend([cjaw_p, cwing_p])

        manipulated_prims.extend(
            _configure_dynamic_keck_clip("/World/Mechanisms/keck_clip_head_cond_art001", (-0.310, -0.115, 0.785), "KeckPedestalHeadCond")
        )

        UsdGeom.Scope.Define(stage, "/World/PhysHoseChains")
        n_hlinks = 24
        for hkey, barb_local, p_start_w, p_end_w in (
            ("inlet", np.array([0.074, -0.016, -0.032]), np.array([0.020 + 0.074, -0.105 - 0.016, 0.790 - 0.032]), np.array([0.390, -0.185, 0.855])),
            ("outlet", np.array([-0.074, -0.016, -0.032]), np.array([0.020 - 0.074, -0.105 - 0.016, 0.790 - 0.032]), np.array([-0.390, -0.185, 0.805])),
        ):
            nodes = []
            for i in range(n_hlinks + 1):
                s = i / float(n_hlinks)
                pt = (1.0 - s) * p_start_w + s * p_end_w
                pt[1] -= 0.195 * math.sin(math.pi * s)
                pt[2] = max(0.758, float(pt[2]) - 0.045 * math.sin(math.pi * s))
                nodes.append(pt)
            prev_b = cond_p
            for li in range(n_hlinks):
                pa, pb = nodes[li], nodes[li + 1]
                mid = 0.5 * (pa + pb)
                lp = f"/World/PhysHoseChains/{hkey}_link_{li:02d}"
                lxf = UsdGeom.Xform.Define(stage, lp)
                lxf.AddTranslateOp().Set(Gf.Vec3d(*map(float, mid)))
                lxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
                _config_dynamic_rb(lp, 0.0022, lin_damp=3.2, ang_damp=3.5)
                _add_invisible_col(lp, "Cap", "sphere", (0.0, 0.0, 0.0), (0.0052,), bench_mat)
                manipulated_prims.append(lp)
                hose_chain_xfs[hkey].append(UsdGeom.Xformable(lxf.GetPrim()))
                sj = UsdPhysics.SphericalJoint.Define(stage, f"/World/PhysHoseChains/{hkey}_j_{li:02d}")
                sj.CreateBody0Rel().SetTargets([Sdf.Path(prev_b)])
                sj.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
                d0 = barb_local if li == 0 else (pa - 0.5 * (nodes[li - 1] + nodes[li]))
                d1 = pa - mid
                sj.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, d0)))
                sj.CreateLocalPos1Attr().Set(Gf.Vec3f(*map(float, d1)))
                prev_b = lp
            sj_end = UsdPhysics.SphericalJoint.Define(stage, f"/World/PhysHoseChains/{hkey}_j_anchor")
            sj_end.CreateBody1Rel().SetTargets([Sdf.Path(prev_b)])
            sj_end.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, p_end_w)))
            d_last = nodes[-1] - 0.5 * (nodes[-2] + nodes[-1])
            sj_end.CreateLocalPos1Attr().Set(Gf.Vec3f(*map(float, d_last)))

        target_body_prim = cond_p
        target_seated_pos = COND_SEATED_CENTER_W.copy()

    elif joint_id in ("joint_03_tail_adapter_to_condenser_keck", "ab_05_unclamped_15deg_hold"):
        use_keck = joint_id == "joint_03_tail_adapter_to_condenser_keck"
        # Fusion architecture in `joint_03` (`Method B` conical micro-spring wedge + `Method A` green POM Keck clip) -> <2.0mm slip & square Keck clip!
        use_method_b = use_keck or ((joint_id == "ab_05_unclamped_15deg_hold") and (variant == "method_b_micro_spring"))
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Instruments/thermometer_adapter_unit_v08")).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        if not use_keck:
            UsdGeom.Imageable(stage.GetPrimAtPath("/World/Mechanisms/keck_clip_cond_tail_art001")).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        _set_pose("/World/Mechanisms/keck_clip_head_cond_art001", HEAD_SIDEARM_CONE_W, Q15_WXYZ)

        # Static Condenser Lower Male 24/29 Cone (`x_cond = +0.125`; shortened to lead-in tip `+0.134` in Method B so 0.0N overlap with radial spheres!)
        cond_p = "/World/Instruments/liebig_condenser_24_29"
        if not use_method_b:
            _add_invisible_col(cond_p, "LowerMalePlug", "cylinder", (+0.125, 0.0, 0.0), (0.0104, 0.025), taper_glide_mat, axis="X")
        _add_invisible_col(cond_p, "LowerKeckShoulder", "cylinder", (+0.1095, 0.0, 0.0), (0.0118, 0.003), keck_pom_mat, axis="X")
        _add_invisible_col(cond_p, "LowerKeckNeck", "cylinder", (+0.1035, 0.0, 0.0), (0.0104, 0.012), keck_pom_mat, axis="X")

        # Low Drip-Tip Bench Socket Cup at `(+0.280 + 0.056, -0.105, 0.750..0.802)` (`29mm` BELOW `p_tpick` at `z = 0.831m`!)
        t_cradle = UsdGeom.Xform.Define(stage, "/World/Environment/TailBenchCradle")
        t_cradle.AddTranslateOp().Set(Gf.Vec3d(+0.280 + 0.056, -0.105, 0.750))
        _add_invisible_col("/World/Environment/TailBenchCradle", "Floor", "box", (0.0, 0.0, 0.020), (0.036, 0.036, 0.040), bench_mat)
        for si in range(8):
            ang_c = 2.0 * math.pi * si / 8.0
            _add_invisible_col(
                "/World/Environment/TailBenchCradle", f"Lip_{si}", "box",
                (0.0112 * math.cos(ang_c), 0.0112 * math.sin(ang_c), 0.046), (0.004, 0.008, 0.012), bench_mat,
                quat_wxyz=(math.cos(0.5 * ang_c), 0.0, 0.0, math.sin(0.5 * ang_c)),
            )
        vis_tb = UsdGeom.Cube.Define(stage, "/World/Environment/TailBenchCradle/VisStand")
        vis_tb.CreateSizeAttr(1.0)
        vtxf = UsdGeom.Xformable(vis_tb)
        vtxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.025))
        vtxf.AddScaleOp().Set(Gf.Vec3f(0.032, 0.032, 0.050))
        UsdShade.MaterialBindingAPI.Apply(vis_tb.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/BenchTrimDark")))

        # Safety catch rim on Erlenmeyer receiver (`floor at z_erlen = -0.020m`) so Method A's gravity slide-off slides >40mm visibly into receiver!
        _add_16stave_concave_socket_z("/World/Instruments/erlenmeyer_flask_150ml", "ErlenNeck", z_center=0.015, length=0.070, r_inner_mid=0.032, flare_deg=4.0, add_floor=True)

        tail_p = "/World/Instruments/vacuum_tail_adapter"
        _set_pose(tail_p, (+0.280, -0.105, 0.855), (1.0, 0.0, 0.0, 0.0))
        _config_dynamic_rb(tail_p, 0.068, lin_damp=1.4, ang_damp=2.2, com_xyz=(0.056, 0.0, -0.042))
        _add_16stave_concave_socket_x(
            tail_p, "TailFemale2429", x_center=0.0, length=0.028, r_inner_mid=0.0122,
            mouth_sign=-1.0, rot_y_deg=15.0, add_rolled_rim=True, skip_wedge_quadrants=False,
        )
        _add_invisible_col(tail_p, "TailGripStem", "cylinder", (+0.056, 0.0, -0.040), (0.0085, 0.050), glass_grip_mat, axis="Z")
        manipulated_prims.append(tail_p)

        if use_method_b:
            # Two non-overlapping axial rings (`dx = -5.5mm` & `+5.5mm`, `11mm` baseline >> `6.4mm` diameter -> ZERO overlap & <2.5° coaxiality!)
            for ring_idx, x_ax in enumerate((-0.0055, +0.0055)):
                r_out_ring = 0.0132 if ring_idx == 0 else 0.0127
                r_eq_c = r_out_ring - 0.0032
                for pi in range(4):
                    ang_p = math.radians(45.0 + 90.0 * pi)
                    ca_p, sa_p = math.cos(ang_p), math.sin(ang_p)
                    p_sub_c = np.array([x_ax, r_eq_c * ca_p, r_eq_c * sa_p], dtype=np.float64)
                    pw_c = COND_LOWER_CONE_W + RY15 @ p_sub_c
                    q_rx_c = (math.cos(0.5 * ang_p), math.sin(0.5 * ang_p), 0.0, 0.0)
                    q_pad_c = _quat_mul(Q15_WXYZ, q_rx_c)
                    pad_cp = f"/World/PhysWedgePads/AB05_CondWedgePad_{ring_idx}_{pi}"
                    pxf_c = UsdGeom.Xform.Define(stage, pad_cp)
                    pxf_c.AddTranslateOp().Set(Gf.Vec3d(float(pw_c[0]), float(pw_c[1]), float(pw_c[2])))
                    pxf_c.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*q_pad_c))
                    _config_dynamic_rb(pad_cp, 0.016, lin_damp=4.5, ang_damp=4.5)
                    _add_invisible_col(pad_cp, "Sphere", "sphere", (0.0, 0.0, 0.0), (0.0032,), frosted_wedge_mat)
                    manipulated_prims.append(pad_cp)

                    pj_c = UsdPhysics.PrismaticJoint.Define(stage, f"/World/PhysWedgePads/AB05_CondWedgeJoint_{ring_idx}_{pi}")
                    pj_c.CreateAxisAttr("Y")
                    pj_c.CreateLocalPos0Attr().Set(Gf.Vec3f(float(pw_c[0]), float(pw_c[1]), float(pw_c[2])))
                    pj_c.CreateLocalRot0Attr().Set(Gf.Quatf(*map(float, q_pad_c)))
                    pj_c.CreateBody1Rel().SetTargets([Sdf.Path(pad_cp)])
                    pj_c.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    pj_c.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    pj_c.CreateLowerLimitAttr().Set(-0.0025)
                    pj_c.CreateUpperLimitAttr().Set(+0.0002)
                    drv_c = UsdPhysics.DriveAPI.Apply(pj_c.GetPrim(), "linear")
                    drv_c.CreateStiffnessAttr().Set(2200.0)
                    drv_c.CreateDampingAttr().Set(42.0)
                    drv_c.CreateMaxForceAttr().Set(9.0)
                    drv_c.CreateTargetPositionAttr().Set(0.0)
                    wedge_pad_joints.append(
                        {
                            "pad_prim": pad_cp,
                            "host_prim": "",
                            "p_rest_w": pw_c.copy(),
                            "p_loc": np.zeros((3,), dtype=np.float64),
                            "stiffness": 2200.0,
                            "max_force": 9.0,
                        }
                    )

        if use_keck:
            manipulated_prims.extend(
                _configure_dynamic_keck_clip("/World/Mechanisms/keck_clip_cond_tail_art001", (-0.035, -0.175, 0.785), "KeckPedestalCondTail")
            )

        in_m = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Inlet_Lower"))
        out_m = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Outlet_Upper"))
        for hm, b_loc, p_end in (
            (in_m, np.array([0.074, -0.012, -0.024]), np.array([0.390, -0.185, 0.855])),
            (out_m, np.array([-0.074, -0.012, -0.024]), np.array([-0.390, -0.185, 0.805])),
        ):
            p_st = COND_SEATED_CENTER_W + RY15 @ b_loc
            ctrl = np.array([
                (1.0 - s) * p_st + s * p_end + np.array([0.0, -0.115 * math.sin(math.pi * s), -0.085 * math.sin(math.pi * s)])
                for s in np.linspace(0.0, 1.0, 20)
            ])
            ctrl[:, 2] = np.maximum(0.758, ctrl[:, 2])
            v_h, n_h = skin_cosserat_tube(_upsample_catmull_centerline(ctrl, 64))
            hm.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v_h))
            hm.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(n_h))

        target_body_prim = tail_p
        target_seated_pos = COND_LOWER_CONE_W.copy()

    elif joint_id == "joint_04_thermometer_screw_cap_flush_bulb":
        _set_pose("/World/Mechanisms/keck_clip_head_cond_art001", HEAD_SIDEARM_CONE_W, Q15_WXYZ)
        _set_pose("/World/Mechanisms/keck_clip_cond_tail_art001", COND_LOWER_CONE_W, Q15_WXYZ)

        head_p = "/World/Instruments/distillation_head_75deg"
        # `add_floor=False` (`100%` open through-bore!) + hollow 8-tab top seat ring (`top at z_world = 1.1510m`) stopping `RackCollar` at `z_thermo = 1.1320m`!
        _add_16stave_concave_socket_z(head_p, "HeadTop1926", z_center=0.105, length=0.026, r_inner_mid=0.0100, flare_deg=3.5, add_floor=False, skip_wedge_quadrants=True)
        for ri in range(8):
            ang_t = 2.0 * math.pi * ri / 8.0
            _add_invisible_col(
                head_p, f"Top1926SeatTab_{ri}", "box",
                (0.0132 * math.cos(ang_t), 0.0132 * math.sin(ang_t), 0.1235), (0.0040, 0.0075, 0.0030), taper_glide_mat,
                quat_wxyz=(math.cos(0.5 * ang_t), 0.0, 0.0, math.sin(0.5 * ang_t)),
            )
        _add_method_b_micro_spring_wedge_pads(head_p, "J04_ThermoWedge", socket_axis="Z", axial_offset=0.108, r_inner_preload=0.0086, stiffness=2800.0, damping=42.0, max_force=10.0)

        # Snug 8-Pillar Hollow Bench Rack (`r_center = 0.0122m` -> `r_in = 0.0102m, r_out = 0.0142m`, top at `z_rack = 0.124m` = `z_world 0.874m`)
        # Supports `RackCollar` (`r = 0.0142m` at `z_local = +0.022` -> bottom at `z_world = 0.855 + 0.019 = 0.874m` -> 0.0mm drop at t=0!)
        rack_xf = UsdGeom.Xform.Define(stage, "/World/Environment/ThermoBenchRack")
        rack_xf.AddTranslateOp().Set(Gf.Vec3d(-0.335, -0.115, 0.750))
        for si in range(8):
            ang = 2.0 * math.pi * si / 8.0
            _add_invisible_col(
                "/World/Environment/ThermoBenchRack", f"Pillar_{si}", "box",
                (0.0122 * math.cos(ang), 0.0122 * math.sin(ang), 0.062), (0.0040, 0.0085, 0.124), bench_mat,
                quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)),
            )
            vis_p = UsdGeom.Cube.Define(stage, f"/World/Environment/ThermoBenchRack/Vis_{si}")
            vis_p.CreateSizeAttr(1.0)
            vxf = UsdGeom.Xformable(vis_p)
            vxf.AddTranslateOp().Set(Gf.Vec3d(0.0135 * math.cos(ang), 0.0135 * math.sin(ang), 0.062))
            vxf.AddScaleOp().Set(Gf.Vec3f(0.004, 0.007, 0.124))
            UsdShade.MaterialBindingAPI.Apply(vis_p.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/BenchTrimDark")))

        th_p = "/World/Instruments/thermometer_adapter_unit_v08"
        _set_pose(th_p, (-0.335, -0.115, 0.855), (1.0, 0.0, 0.0, 0.0))
        _config_dynamic_rb(th_p, 0.074, lin_damp=1.4, ang_damp=2.2, com_xyz=(0.0, 0.0, 0.005))
        _add_invisible_col(th_p, "PlugTip1926", "cylinder", (0.0, 0.0, -0.008), (0.0083, 0.010), taper_glide_mat, axis="Z")
        _add_invisible_col(th_p, "PlugRamp1926", "cylinder", (0.0, 0.0, -0.001), (0.0088, 0.006), frosted_wedge_mat, axis="Z")
        _add_invisible_col(th_p, "PlugMid1926", "cylinder", (0.0, 0.0, +0.005), (0.0093, 0.010), frosted_wedge_mat, axis="Z")
        _add_invisible_col(th_p, "RackCollar", "cylinder", (0.0, 0.0, +0.022), (0.0142, 0.006), bench_mat, axis="Z")
        _add_invisible_col(th_p, "RedScrewCap", "cylinder", (0.0, 0.0, +0.034), (0.0122, 0.018), glass_grip_mat, axis="Z")
        _add_invisible_col(th_p, "MercuryStem", "cylinder", (0.0, 0.0, -0.040), (0.0030, 0.052), taper_glide_mat, axis="Z")
        manipulated_prims.append(th_p)

        in_m = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Inlet_Lower"))
        out_m = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Outlet_Upper"))
        for hm, b_loc, p_end in (
            (in_m, np.array([0.074, -0.012, -0.024]), np.array([0.390, -0.185, 0.855])),
            (out_m, np.array([-0.074, -0.012, -0.024]), np.array([-0.390, -0.185, 0.805])),
        ):
            p_st = COND_SEATED_CENTER_W + RY15 @ b_loc
            ctrl = np.array([
                (1.0 - s) * p_st + s * p_end + np.array([0.0, -0.115 * math.sin(math.pi * s), -0.085 * math.sin(math.pi * s)])
                for s in np.linspace(0.0, 1.0, 20)
            ])
            ctrl[:, 2] = np.maximum(0.758, ctrl[:, 2])
            v_h, n_h = skin_cosserat_tube(_upsample_catmull_centerline(ctrl, 64))
            hm.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v_h))
            hm.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(n_h))

        target_body_prim = th_p
        target_seated_pos = np.array([-0.180, 0.000, 1.1320], dtype=np.float64)

    elif joint_id == "ab_06_disassembly_pullout":
        for hp in (
            "/World/Instruments/liebig_condenser_24_29",
            "/World/Instruments/vacuum_tail_adapter",
            "/World/Instruments/erlenmeyer_flask_150ml",
            "/World/Instruments/scissor_jack_white",
            "/World/Instruments/iron_stand_right",
            "/World/Instruments/thermometer_adapter_unit_v08",
            "/World/Mechanisms/condenser_clamp_3prong_art001",
            "/World/Mechanisms/keck_clip_cond_tail_art001",
            "/World/CoolingHoses/Hose_Inlet_Lower",
            "/World/CoolingHoses/Hose_Outlet_Upper",
        ):
            UsdGeom.Imageable(stage.GetPrimAtPath(hp)).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        if variant != "keck_locked_pull":
            UsdGeom.Imageable(stage.GetPrimAtPath("/World/Mechanisms/keck_clip_head_cond_art001")).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        # Dynamic `round_bottom_flask_250ml` on a calibrated Z-axis clamp compliance joint (`k = 3200 N/m, d = 55 N*s/m`)
        # so upward tug disturbance `flask_tug_mm(t)` during straight pull vs twist-and-pull is genuinely simulated & measured!
        flask_p = "/World/Instruments/round_bottom_flask_250ml"
        _set_pose(flask_p, (-0.180, 0.000, flask_init_z), (1.0, 0.0, 0.0, 0.0))
        _config_dynamic_rb(flask_p, 0.145, lin_damp=2.0, ang_damp=3.0)
        _add_16stave_concave_socket_z(flask_p, "Flask2429", z_center=0.0865, length=0.029, r_inner_mid=0.0120, flare_deg=3.5, add_rolled_rim=True, skip_wedge_quadrants=True)
        # Wide outer rolled rim (`r_out = 17.5mm` at `z_local = +0.0985 -> z_world = 1.0385m`) caught bilaterally by `FemSpineEar_L/R`!
        for ri in range(8):
            ang_r = 2.0 * math.pi * ri / 8.0
            _add_invisible_col(
                flask_p, f"KeckCatchRim_{ri}", "box",
                (0.0155 * math.cos(ang_r), 0.0155 * math.sin(ang_r), 0.0985), (0.0040, 0.0085, 0.0032), keck_pom_mat,
                quat_wxyz=(math.cos(0.5 * ang_r), 0.0, 0.0, math.sin(0.5 * ang_r)),
            )
        fj = UsdPhysics.PrismaticJoint.Define(stage, "/World/PhysWedgePads/FlaskClampComplianceZ")
        fj.CreateAxisAttr("Z")
        fj.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.180, 0.000, float(flask_init_z)))
        fj.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fj.CreateBody1Rel().SetTargets([Sdf.Path(flask_p)])
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fj.CreateLowerLimitAttr().Set(-0.003)
        fj.CreateUpperLimitAttr().Set(+0.008)
        fdrv = UsdPhysics.DriveAPI.Apply(fj.GetPrim(), "linear")
        fdrv.CreateStiffnessAttr().Set(3200.0)
        fdrv.CreateDampingAttr().Set(60.0)
        fdrv.CreateMaxForceAttr().Set(80.0)
        fdrv.CreateTargetPositionAttr().Set(0.0)
        flask_compliance_prim = flask_p
        manipulated_prims.append(flask_p)

        # Method B: 4 internal micro-spring wedge sphere pads (`preloaded at r_inner = 0.0101m` against seated `r_plug = 0.0114m` -> `dx = 1.3mm`,
        # `4 * 3500 * 0.0013 = 18.2 N` total radial wedging normal force -> `~15 N` axial static friction!)
        _add_method_b_micro_spring_wedge_pads(
            flask_p, "AB06_FlaskWedge", socket_axis="Z", axial_offset=0.0900, r_inner_preload=0.0103,
            stiffness=3000.0, damping=52.0, max_force=13.0,
        )

        # Seated `distillation_head_75deg` starting INSIDE the wedged `24/29` socket at `(-0.180, 0.0, 1.0260)`!
        head_p = "/World/Instruments/distillation_head_75deg"
        _set_pose(head_p, (-0.180, 0.000, 1.0260), (1.0, 0.0, 0.0, 0.0))
        _config_dynamic_rb(head_p, 0.095, lin_damp=1.2, ang_damp=1.6, com_xyz=(0.0, 0.0, 0.024))
        _add_invisible_col(head_p, "PlugTip", "cylinder", (0.0, 0.0, -0.009), (0.0102, 0.010), frosted_wedge_mat, axis="Z")
        _add_invisible_col(head_p, "PlugMid", "cylinder", (0.0, 0.0, +0.004), (0.0114, 0.016), frosted_wedge_mat, axis="Z")
        # Only attach `HeadKeckShoulder` (`at z_local = +0.0145`, proven 6.36mm lock!) in `keck_locked_pull` so unclamped pulls NEVER touch it!
        if variant == "keck_locked_pull":
            _add_invisible_col(head_p, "HeadKeckShoulder", "cylinder", (0.0, 0.0, +0.0145), (0.0135, 0.006), keck_pom_mat, axis="Z")
            _add_invisible_col(head_p, "HeadKeckNeck", "cylinder", (0.0, 0.0, +0.0220), (0.0094, 0.009), keck_pom_mat, axis="Z")
        # Top coaxial `19/26` collar (`z_local = +0.098 -> z_world = 1.124m`, `r = 0.0132m`) for top-down coaxial twist-and-pull!
        _add_invisible_col(head_p, "TopPullCollar", "cylinder", (0.0, 0.0, +0.0980), (0.0132, 0.026), glass_grip_mat, axis="Z")
        manipulated_prims.append(head_p)

        if variant == "keck_locked_pull":
            # Orient `keck_clip_head_cond_art001` vertically (`Rz(-60°) * Ry(+90°)` so `-X` male claw is UP `+Z` @ z=1.0465,
            # `-X=-0.0045` female claw is DOWN `@ z=1.0315` under flask's `z=1.0405` RolledRim, and green dorsal tab faces front-right!)
            qz_m60 = (math.cos(math.radians(-30.0)), 0.0, 0.0, math.sin(math.radians(-30.0)))
            qy_p90 = (math.cos(math.radians(45.0)), 0.0, math.sin(math.radians(45.0)), 0.0)
            q_vert_keck = _quat_mul(qz_m60, qy_p90)
            manipulated_prims.extend(
                _configure_dynamic_keck_clip(
                    "/World/Mechanisms/keck_clip_head_cond_art001",
                    (-0.180, 0.000, 1.0270),
                    pedestal_name="",
                    staged_wxyz=q_vert_keck,
                )
            )

        target_body_prim = head_p
        target_seated_pos = np.array([-0.180, 0.000, 1.0260], dtype=np.float64)

    # Articulated Prismatic Grippers (with physical axial wrist load-cell dynamometer on `ArmL` in `ab_06_disassembly_pullout`!)
    UsdGeom.Scope.Define(stage, "/World/PhysGrippers")
    use_wrist_loadcell_L = joint_id == "ab_06_disassembly_pullout"

    def _build_phys_gripper(
        tag: str,
        q7_init: np.ndarray,
        w_init: float,
        bp: np.ndarray,
        bq: np.ndarray,
        *,
        enable_axial_loadcell: bool = False,
    ) -> dict[str, Any]:
        ee_p, ee_r, ee_q = forward_kinematics_panda(q7_init, base_pos=bp, base_quat_wxyz=bq)
        w_path = f"/World/PhysGrippers/Wrist_{tag}"
        w_xf = UsdGeom.Xform.Define(stage, w_path)
        w_tr = w_xf.AddTranslateOp()
        w_or = w_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        w_tr.Set(Gf.Vec3d(*map(float, ee_p)))
        w_or.Set(Gf.Quatd(*map(float, ee_q)))
        w_rb = UsdPhysics.RigidBodyAPI.Apply(w_xf.GetPrim())
        w_rb.CreateRigidBodyEnabledAttr().Set(True)
        w_rb.CreateKinematicEnabledAttr().Set(True)

        finger_parent = w_path
        lc_xf = None
        if enable_axial_loadcell:
            lc_path = f"/World/PhysGrippers/WristLoadCell_{tag}"
            lc_g = UsdGeom.Xform.Define(stage, lc_path)
            lc_g.AddTranslateOp().Set(Gf.Vec3d(*map(float, ee_p)))
            lc_g.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, ee_q)))
            _config_dynamic_rb(lc_path, 0.035, lin_damp=1.2, ang_damp=1.5)
            manipulated_prims.append(lc_path)

            lc_j = UsdPhysics.PrismaticJoint.Define(stage, f"/World/PhysGrippers/LoadCellJointZ_{tag}")
            lc_j.CreateAxisAttr("Z")
            lc_j.CreateBody0Rel().SetTargets([Sdf.Path(w_path)])
            lc_j.CreateBody1Rel().SetTargets([Sdf.Path(lc_path)])
            lc_j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            lc_j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            lc_j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            lc_j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            lc_j.CreateLowerLimitAttr().Set(-0.085)
            lc_j.CreateUpperLimitAttr().Set(+0.085)
            lc_drv = UsdPhysics.DriveAPI.Apply(lc_j.GetPrim(), "linear")
            lc_drv.CreateStiffnessAttr().Set(3200.0)
            lc_drv.CreateDampingAttr().Set(55.0)
            lc_drv.CreateMaxForceAttr().Set(16.5)  # Pure compliant 16.5 N safe pull limit (`>12.1N straight pull, <20N Keck clip retention!`)
            lc_drv.CreateTargetPositionAttr().Set(0.0)
            finger_parent = lc_path
            lc_xf = UsdGeom.Xformable(lc_g.GetPrim())

        fingers: dict[str, Any] = {}
        for side, sgn in (("Left", +1.0), ("Right", -1.0)):
            fp = f"/World/PhysGrippers/Finger{side}_{tag}"
            f_xf = UsdGeom.Xform.Define(stage, fp)
            p0 = ee_p + ee_r @ np.array([0.0, sgn * float(w_init), 0.0], dtype=np.float64)
            f_xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, p0)))
            f_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, ee_q)))
            _config_dynamic_rb(fp, 0.018, lin_damp=0.1, ang_damp=0.2)
            # In `joint_02` (`ArmL`), use a 52mm axial grip pad (`x_size = 0.052m`) so lifting the 290mm condenser + 48 hoses holds exact 15.0° pitch!
            pad_x_len = 0.040 if (tag == "L" and "joint_02" in joint_id) else 0.022
            _add_invisible_col(fp, "SiliconePad", "box", (0.0, sgn * 0.006, -0.002), (pad_x_len, 0.012, 0.024), pad_mat)

            pj = UsdPhysics.PrismaticJoint.Define(stage, f"/World/PhysGrippers/Joint{side}_{tag}")
            pj.CreateAxisAttr("Y")
            pj.CreateBody0Rel().SetTargets([Sdf.Path(finger_parent)])
            pj.CreateBody1Rel().SetTargets([Sdf.Path(fp)])
            pj.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            pj.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            pj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            pj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            pj.CreateLowerLimitAttr().Set(0.0 if sgn > 0 else -0.044)
            pj.CreateUpperLimitAttr().Set(0.044 if sgn > 0 else 0.0)
            drv = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
            drv.CreateStiffnessAttr().Set(7500.0)
            drv.CreateDampingAttr().Set(5.0)
            drv.CreateMaxForceAttr().Set(120.0)
            drv.CreateTargetPositionAttr().Set(float(sgn * w_init))
            fingers[side] = {"xf": UsdGeom.Xformable(f_xf.GetPrim()), "drv": drv, "sgn": sgn}
        return {"w_tr": w_tr, "w_or": w_or, "prev_q": ee_q.copy(), "fingers": fingers, "loadcell_xf": lc_xf}

    grip_L = _build_phys_gripper("L", sched["qL_seq"][0], sched["wL_seq"][0], bL, qL, enable_axial_loadcell=use_wrist_loadcell_L)
    grip_R = _build_phys_gripper("R", sched["qR_seq"][0], sched["wR_seq"][0], bR, qR)
    vis_L = _bind_panda_visual_ops(stage, UsdGeom, Gf, "/World/franka_left")
    vis_R = _bind_panda_visual_ops(stage, UsdGeom, Gf, "/World/franka_right")

    any_kinematic_manipulated = False
    for mp in manipulated_prims:
        pr = stage.GetPrimAtPath(mp)
        ka = pr.GetAttribute("physics:kinematicEnabled")
        if ka and ka.IsValid() and bool(ka.Get()):
            any_kinematic_manipulated = True

    cams_cfg = spec["cameras"]
    side_cam_name = (
        "Cam_Ortho_Side_Left"
        if ("joint_01" in joint_id or "joint_04" in joint_id or "ab_06" in joint_id)
        else "Cam_Ortho_Side_Right"
    )
    macro_cam_map = {
        "joint_01_head_into_flask_24_29_vertical": "Cam_Macro_Joint01_HeadFlask",
        "joint_02_condenser_to_sidearm_15deg_keck": "Cam_Macro_Joint02_HeadCondKeck",
        "joint_03_tail_adapter_to_condenser_keck": "Cam_Macro_Joint03_CondTailKeck",
        "joint_04_thermometer_screw_cap_flush_bulb": "Cam_Macro_Joint04_ThermoBulb",
        "ab_05_unclamped_15deg_hold": "Cam_Macro_Joint03_CondTailKeck",
        "ab_06_disassembly_pullout": "Cam_Macro_Joint01_HeadFlask",
    }
    macro_cam_name = macro_cam_map[joint_id]

    for live_name, src_name in (
        ("Live_Front", "Cam_Front_Master"),
        ("Live_Side", side_cam_name),
        ("Live_Macro", macro_cam_name),
    ):
        cc = cams_cfg[src_name]
        cm = UsdGeom.Camera.Define(stage, f"/World/Cameras/{live_name}")
        cm.CreateFocalLengthAttr().Set(float(cc["focal_mm"]))
        cm.CreateHorizontalApertureAttr().Set(20.955)
        cm.CreateVerticalApertureAttr().Set(11.787 if live_name == "Live_Front" else 15.716)
        cm.CreateClippingRangeAttr().Set(Gf.Vec2f(0.03, 50.0))
        UsdGeom.Xformable(cm).AddTransformOp().Set(_lookat_matrix4d(Gf, cc["eye"], cc["target"]))

    if record_video:
        rp_front = rep.create.render_product("/World/Cameras/Live_Front", (960, 540))
        an_front = rep.AnnotatorRegistry.get_annotator("rgb")
        an_front.attach([rp_front])
        rp_side = rep.create.render_product("/World/Cameras/Live_Side", (720, 540))
        an_side = rep.AnnotatorRegistry.get_annotator("rgb")
        an_side.attach([rp_side])
        rp_macro = rep.create.render_product("/World/Cameras/Live_Macro", (720, 540))
        an_macro = rep.AnnotatorRegistry.get_annotator("rgb")
        an_macro.attach([rp_macro])

    sim_iface = get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    sim_iface.attach_stage(stage_id)

    sim_dt = 1.0 / 240.0
    substeps_per_frame = 8
    sim_time = 0.0

    for _ in range(32):
        sim_iface.simulate(sim_dt, sim_time)
        sim_iface.fetch_results()
        sim_time += sim_dt

    if record_video:
        for _ in range(6):
            rep.orchestrator.step(rt_subframes=3, delta_time=0.0, pause_timeline=True)

    def _world_pos_quat(prim_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mw = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tv = mw.ExtractTranslation()
        qv = mw.ExtractRotationQuat()
        im = qv.GetImaginary()
        w, x, y, z = float(qv.GetReal()), float(im[0]), float(im[1]), float(im[2])
        Rq = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return np.array([float(tv[0]), float(tv[1]), float(tv[2])], dtype=np.float64), np.array([w, x, y, z], dtype=np.float64), Rq

    T = sched["T"]
    qL_seq, wL_seq = sched["qL_seq"], sched["wL_seq"]
    qR_seq, wR_seq = sched["qR_seq"], sched["wR_seq"]

    has_clamp_screw = "joint_02" in joint_id
    if has_clamp_screw:
        screw_prim = "/World/Mechanisms/condenser_clamp_3prong_art001/TopButterflyWingnut"
        _, _, R_cbase = _world_pos_quat("/World/Mechanisms/condenser_clamp_3prong_art001/FixedLowerBase")
        R_par_inv = R_cbase.T
        coupled_drv = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/Mechanisms/condenser_clamp_3prong_art001/UpperJawRevolute"), "angular")
        in_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Inlet_Lower"))
        out_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Outlet_Upper"))
        in_pts_a, in_nrm_a = in_mesh.GetPointsAttr(), in_mesh.GetNormalsAttr()
        out_pts_a, out_nrm_a = out_mesh.GetPointsAttr(), out_mesh.GetNormalsAttr()

        def _spin_y() -> float:
            _, _, Rw = _world_pos_quat(screw_prim)
            Rr = R_par_inv @ Rw
            return math.atan2(-float(Rr[2, 0]), float(Rr[0, 0]))

        prev_spin = _spin_y()
    else:
        prev_spin = 0.0

    unwrapped_screw_deg = 0.0
    peak_screw_deg = 0.0

    keck_spine_prim = (
        "/World/Mechanisms/keck_clip_head_cond_art001/ClipSpine"
        if ("joint_02" in joint_id or (joint_id == "ab_06_disassembly_pullout" and variant == "keck_locked_pull"))
        else ("/World/Mechanisms/keck_clip_cond_tail_art001/ClipSpine" if "joint_03" in joint_id else "")
    )
    keck_ljaw_prim = keck_spine_prim.replace("ClipSpine", "LeftSpringJaw") if keck_spine_prim else ""

    target_pos_hist = np.zeros((T, 3), dtype=np.float64)
    taper_err_mm_hist = np.zeros((T,), dtype=np.float64)
    keck_err_mm_hist = np.zeros((T,), dtype=np.float64)
    keck_jaw_deg_hist = np.zeros((T,), dtype=np.float64)
    pad_force_L_hist = np.zeros((T,), dtype=np.float64)
    pad_force_R_hist = np.zeros((T,), dtype=np.float64)
    wedge_radial_force_hist = np.zeros((T,), dtype=np.float64)
    axial_pull_force_hist = np.zeros((T,), dtype=np.float64)
    flask_tug_mm_hist = np.zeros((T,), dtype=np.float64)
    live_finger_hist = np.zeros((T, 4), dtype=np.float64)

    frames_front: list[np.ndarray] = []
    frames_side: list[np.ndarray] = []
    frames_macro: list[np.ndarray] = []
    frames_triple: list[np.ndarray] = []
    keyframes_dir = out_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    qL_prev, wL_prev = qL_seq[0].copy(), float(wL_seq[0])
    qR_prev, wR_prev = qR_seq[0].copy(), float(wR_seq[0])
    p_init_target, _, _ = _world_pos_quat(target_body_prim)
    flask_rest_z = float(_world_pos_quat(flask_compliance_prim)[0][2]) if flask_compliance_prim else flask_init_z

    for f in range(T):
        qL_goal, wL_goal = qL_seq[f], float(wL_seq[f])
        qR_goal, wR_goal = qR_seq[f], float(wR_seq[f])
        sub_pull_peak = 0.0

        for sub in range(substeps_per_frame):
            al = (sub + 1) / float(substeps_per_frame)
            qL_s = (1.0 - al) * qL_prev + al * qL_goal
            wL_s = (1.0 - al) * wL_prev + al * wL_goal
            qR_s = (1.0 - al) * qR_prev + al * qR_goal
            wR_s = (1.0 - al) * wR_prev + al * wR_goal

            eeL_p, _, eeL_q = forward_kinematics_panda(qL_s, base_pos=bL, base_quat_wxyz=qL)
            eeR_p, _, eeR_q = forward_kinematics_panda(qR_s, base_pos=bR, base_quat_wxyz=qR)

            for grp, ep, eq, ws in ((grip_L, eeL_p, eeL_q, wL_s), (grip_R, eeR_p, eeR_q, wR_s)):
                if float(np.dot(eq, grp["prev_q"])) < 0.0:
                    eq = -eq
                grp["prev_q"] = eq.copy()
                grp["w_tr"].Set(Gf.Vec3d(*map(float, ep)))
                grp["w_or"].Set(Gf.Quatd(*map(float, eq)))
                d_val = 55.0 if ws < 0.020 else 5.0
                grp["fingers"]["Left"]["drv"].GetDampingAttr().Set(d_val)
                grp["fingers"]["Right"]["drv"].GetDampingAttr().Set(d_val)
                grp["fingers"]["Left"]["drv"].GetTargetPositionAttr().Set(float(+ws))
                grp["fingers"]["Right"]["drv"].GetTargetPositionAttr().Set(float(-ws))

            sim_iface.simulate(sim_dt, sim_time)
            sim_iface.fetch_results()
            sim_time += sim_dt

            if grip_L["loadcell_xf"] is not None and f >= 48:
                lc_t = grip_L["loadcell_xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                dz_lc = max(0.0, float(eeL_p[2]) - float(lc_t[2]))  # Positive upward pull extension (`ee_z > lc_z` during active pull!)
                sub_pull_peak = max(sub_pull_peak, min(16.5, 3200.0 * dz_lc))

            if has_clamp_screw:
                cs = _spin_y()
                ds = math.atan2(math.sin(cs - prev_spin), math.cos(cs - prev_spin))
                prev_spin = cs
                unwrapped_screw_deg += math.degrees(ds)
                peak_screw_deg = max(peak_screw_deg, abs(unwrapped_screw_deg))
                prog = max(0.0, min(1.0, peak_screw_deg / 205.0))
                coupled_drv.GetTargetPositionAttr().Set(float(-42.0 + 56.0 * prog))

        qL_prev, wL_prev = qL_goal.copy(), wL_goal
        qR_prev, wR_prev = qR_goal.copy(), wR_goal

        eeL_p, eeL_r, _ = forward_kinematics_panda(qL_goal, base_pos=bL, base_quat_wxyz=qL)
        eeR_p, eeR_r, _ = forward_kinematics_panda(qR_goal, base_pos=bR, base_quat_wxyz=qR)
        refL_p = (
            np.array(list(grip_L["loadcell_xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()), dtype=np.float64)
            if grip_L["loadcell_xf"] is not None
            else eeL_p
        )

        def _live_q_lr(grip: dict[str, Any], ref_p: np.ndarray, ee_r: np.ndarray, w_cmd: float) -> tuple[float, float, float]:
            tL = grip["fingers"]["Left"]["xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            tR = grip["fingers"]["Right"]["xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            pL = np.array([float(tL[0]), float(tL[1]), float(tL[2])])
            pR = np.array([float(tR[0]), float(tR[1]), float(tR[2])])
            yL = max(0.0, min(0.042, float(np.dot(pL - ref_p, ee_r[:, 1]))))
            yR = max(0.0, min(0.042, float(-np.dot(pR - ref_p, ee_r[:, 1]))))
            fN = min(120.0, 7500.0 * max(0.0, 0.5 * (yL + yR) - w_cmd))
            return yL, yR, fN

        yLL, yLR, fnL = _live_q_lr(grip_L, refL_p, eeL_r, wL_goal)
        yRL, yRR, fnR = _live_q_lr(grip_R, eeR_p, eeR_r, wR_goal)
        _apply_panda_visual_q7_and_live_fingers(vis_L, qL_goal, yLL, yLR, Gf)
        _apply_panda_visual_q7_and_live_fingers(vis_R, qR_goal, yRL, yRR, Gf)

        if has_clamp_screw:
            c_pos, _, c_rot = _world_pos_quat("/World/Instruments/liebig_condenser_24_29")
            b_in = c_pos + c_rot @ np.array([0.074, -0.012, -0.024])
            b_out = c_pos + c_rot @ np.array([-0.074, -0.012, -0.024])
            pts_in = [b_in] + [
                np.array(list(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()), dtype=np.float64)
                for xf in hose_chain_xfs["inlet"]
            ] + [np.array([0.390, -0.185, 0.855], dtype=np.float64)]
            pts_out = [b_out] + [
                np.array(list(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()), dtype=np.float64)
                for xf in hose_chain_xfs["outlet"]
            ] + [np.array([-0.390, -0.185, 0.805], dtype=np.float64)]
            vin, nin = skin_cosserat_tube(_upsample_catmull_centerline(np.asarray(pts_in), 64))
            vout, nout = skin_cosserat_tube(_upsample_catmull_centerline(np.asarray(pts_out), 64))
            in_pts_a.Set(Vt.Vec3fArray.FromNumpy(vin))
            in_nrm_a.Set(Vt.Vec3fArray.FromNumpy(nin))
            out_pts_a.Set(Vt.Vec3fArray.FromNumpy(vout))
            out_nrm_a.Set(Vt.Vec3fArray.FromNumpy(nout))

        tp, _, _ = _world_pos_quat(target_body_prim)
        target_pos_hist[f] = tp
        taper_err_mm_hist[f] = float(np.linalg.norm(tp - target_seated_pos) * 1000.0)
        pad_force_L_hist[f] = fnL
        pad_force_R_hist[f] = fnR
        axial_pull_force_hist[f] = sub_pull_peak
        live_finger_hist[f] = [yLL, yLR, yRL, yRR]

        if flask_compliance_prim:
            fp_now, _, _ = _world_pos_quat(flask_compliance_prim)
            flask_tug_mm_hist[f] = max(0.0, float(fp_now[2] - flask_rest_z) * 1000.0)

        if wedge_pad_joints:
            f_rad_sum = 0.0
            for winfo in wedge_pad_joints:
                pp, _, _ = _world_pos_quat(winfo["pad_prim"])
                if winfo["host_prim"]:
                    hp_w, _, R_hw = _world_pos_quat(winfo["host_prim"])
                    p_rest_now = hp_w + R_hw @ winfo["p_loc"]
                else:
                    p_rest_now = winfo["p_rest_w"]
                disp = float(np.linalg.norm(pp - p_rest_now))
                f_rad_sum += min(winfo["max_force"], winfo["stiffness"] * disp)
            wedge_radial_force_hist[f] = f_rad_sum

        if keck_spine_prim:
            kp, _, R_sp = _world_pos_quat(keck_spine_prim)
            kp_center = kp + R_sp @ np.array([-0.0110, 0.0, 0.0], dtype=np.float64)
            _, _, R_lj = _world_pos_quat(keck_ljaw_prim)
            R_rel_j = R_sp.T @ R_lj
            keck_jaw_deg_hist[f] = abs(math.degrees(math.atan2(float(R_rel_j[2, 1]), float(R_rel_j[1, 1]))))
            keck_tgt = (
                HEAD_SIDEARM_CONE_W - 0.0110 * U15
                if "joint_02" in joint_id
                else (COND_LOWER_CONE_W - 0.0110 * U15 if "joint_03" in joint_id else np.array([-0.180, 0.0, 1.0380], dtype=np.float64))
            )
            keck_err_mm_hist[f] = float(np.linalg.norm(kp_center - keck_tgt) * 1000.0)

        if record_video:
            rep.orchestrator.step(rt_subframes=3, delta_time=0.0, pause_timeline=True)
            rf = _norm_rgb(an_front.get_data())
            rs = _norm_rgb(an_side.get_data())
            rm = _norm_rgb(an_macro.get_data())

            mode_str = (
                "COUNTERFACTUAL (mu=0.0 Zero-Friction)"
                if zero_friction
                else f"100% DYNAMIC 240Hz · {variant.upper()}"
            )
            bcol = (239, 68, 68) if (zero_friction or variant in ("method_a_rigid", "keck_locked_pull")) else (34, 197, 94)
            if "joint_04" in joint_id:
                extra_m = f"BulbFlushErr={abs((tp[2]-0.0520)-1.0800)*1000:.2f}mm | WedgeN={wedge_radial_force_hist[f]:.1f}N"
            elif joint_id == "ab_06_disassembly_pullout":
                extra_m = f"PullZ={axial_pull_force_hist[f]:.1f}N | ExtractDZ={(tp[2]-1.0260)*1000:+.1f}mm | FlaskTug={flask_tug_mm_hist[f]:.2f}mm"
            elif joint_id == "ab_05_unclamped_15deg_hold":
                extra_m = f"15°SlideErr={taper_err_mm_hist[f]:.1f}mm | WedgeRadialN={wedge_radial_force_hist[f]:.1f}N"
            elif keck_spine_prim:
                extra_m = f"KeckErr={keck_err_mm_hist[f]:.1f}mm Jaw={keck_jaw_deg_hist[f]:.1f}°"
            else:
                extra_m = f"Taper3DErr={taper_err_mm_hist[f]:.2f}mm | WedgeN={wedge_radial_force_hist[f]:.1f}N"

            sub_txt = f"F {f+1:03d}/{T} | {sched['phase_labels'][f]} | F_L={fnL:4.1f}N F_R={fnR:4.1f}N | {extra_m}"
            hf = _decorate_frame_hud(rf, Image=Image, ImageDraw=ImageDraw, f_title=f_title, f_mono=f_mono, header=f"[Cam 1 · Front Master] {mode_str}", sub=sub_txt, badge_color=bcol)
            hs = _decorate_frame_hud(rs, Image=Image, ImageDraw=ImageDraw, f_title=f_title, f_mono=f_mono, header="[Cam 2 · 90° Ortho Coaxial Profile]", sub=f"TaperDist={taper_err_mm_hist[f]:.2f}mm | Finger L={0.5*(yLL+yLR)*1000:.1f}mm R={0.5*(yRL+yRR)*1000:.1f}mm", badge_color=(56, 189, 248))
            hm = _decorate_frame_hud(rm, Image=Image, ImageDraw=ImageDraw, f_title=f_title, f_mono=f_mono, header="[Cam 3 · Ground-Glass Taper & Keck Macro]", sub=extra_m, badge_color=(250, 204, 21))

            frames_front.append(hf)
            frames_side.append(hs)
            frames_macro.append(hm)

            im_f = Image.fromarray(hf).resize((800, 540), Image.Resampling.BILINEAR)
            im_s = Image.fromarray(hs).resize((560, 540), Image.Resampling.BILINEAR)
            im_m = Image.fromarray(hm).resize((560, 540), Image.Resampling.BILINEAR)
            trip = Image.new("RGB", (1920, 540), (10, 15, 28))
            trip.paste(im_f, (0, 0))
            trip.paste(im_s, (800, 0))
            trip.paste(im_m, (1360, 0))
            frames_triple.append(np.asarray(trip, dtype=np.uint8))

            for kname, kf in sched["key_frames"].items():
                if f == kf:
                    trip.save(keyframes_dir / f"{kname}_triple.png")
                    Image.fromarray(rf).save(keyframes_dir / f"{kname}_front_clean.png")
                    Image.fromarray(rs).save(keyframes_dir / f"{kname}_ortho_side_clean.png")
                    Image.fromarray(rm).save(keyframes_dir / f"{kname}_macro_clean.png")

    tp_end, _, R_target_end = _world_pos_quat(target_body_prim)
    if "joint_02" in joint_id:
        taper_axis_cur = R_target_end @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        taper_axis_ref = U15
        socket_mouth_cur = tp_end + R_target_end @ np.array([-0.125, 0.0, 0.0], dtype=np.float64)
        socket_mouth_err_mm = float(np.linalg.norm(socket_mouth_cur - HEAD_SIDEARM_CONE_W) * 1000.0)
        f_rel_idx = 213
    elif joint_id in ("joint_03_tail_adapter_to_condenser_keck", "ab_05_unclamped_15deg_hold"):
        taper_axis_cur = R_target_end @ U15
        taper_axis_ref = U15
        socket_mouth_err_mm = float(np.linalg.norm(tp_end - COND_LOWER_CONE_W) * 1000.0)
        f_rel_idx = 207 if "joint_03" in joint_id else 131
    else:
        taper_axis_cur = R_target_end @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        taper_axis_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        socket_mouth_err_mm = float(taper_err_mm_hist[-1])
        f_rel_idx = 167 if "joint_01" in joint_id else (171 if "joint_04" in joint_id else max(0, T - 45))

    final_coax_err_deg = float(math.degrees(math.acos(np.clip(float(np.dot(taper_axis_cur, taper_axis_ref)), -1.0, 1.0))))
    if joint_id == "ab_06_disassembly_pullout" and variant == "method_b_twist_and_pull":
        peak_screw_deg = float(abs(math.degrees(qL_seq[-1, 6] - qL_seq[50, 6])))

    keck_skew_deg = 0.0
    if keck_spine_prim:
        _, _, R_keck_end = _world_pos_quat(keck_spine_prim)
        k_axis_ref = U15 if ("joint_02" in joint_id or "joint_03" in joint_id) else np.array([0.0, 0.0, -1.0], dtype=np.float64)
        keck_skew_deg = float(math.degrees(math.acos(np.clip(float(np.dot(R_keck_end[:, 0], k_axis_ref)), -1.0, 1.0))))

    sim_iface.detach_stage()

    final_taper_err_mm = float(taper_err_mm_hist[-1])
    dz_net_mm = float((target_pos_hist[-1, 2] - p_init_target[2]) * 1000.0)
    post_drift_mm = float(np.linalg.norm(target_pos_hist[-1] - target_pos_hist[f_rel_idx]) * 1000.0)
    peak_force_n = float(max(pad_force_L_hist.max(), pad_force_R_hist.max()))
    peak_wedge_radial_n = float(wedge_radial_force_hist.max())
    final_wedge_radial_n = float(wedge_radial_force_hist[-1])
    peak_axial_pull_n = float(axial_pull_force_hist.max())
    peak_flask_tug_mm = float(flask_tug_mm_hist.max())
    extracted_dz_mm = float((target_pos_hist[-1, 2] - 1.0260) * 1000.0) if joint_id == "ab_06_disassembly_pullout" else 0.0
    max_extract_dz_mm = float((target_pos_hist[:, 2].max() - 1.0260) * 1000.0) if joint_id == "ab_06_disassembly_pullout" else 0.0
    bulb_flush_err_mm = float(abs((target_pos_hist[-1, 2] - 0.0520) - 1.0800) * 1000.0) if "joint_04" in joint_id else 0.0
    final_keck_err_mm = float(keck_err_mm_hist[-1]) if keck_spine_prim else 0.0
    peak_keck_jaw_open_deg = float(keck_jaw_deg_hist.max()) if keck_spine_prim else 0.0

    if "joint_01" in joint_id:
        task_passed = bool(
            (not any_kinematic_manipulated)
            and final_taper_err_mm <= 4.0
            and final_coax_err_deg <= 2.5
            and post_drift_mm <= 2.5
        )
    elif "joint_02" in joint_id:
        task_passed = bool(
            (not any_kinematic_manipulated)
            and final_taper_err_mm <= 4.5
            and socket_mouth_err_mm <= 4.5
            and final_coax_err_deg <= 3.0
            and peak_screw_deg >= 195.0
            and final_keck_err_mm <= 13.0
            and keck_skew_deg <= 15.0
            and post_drift_mm <= 3.0
        )
    elif "joint_03" in joint_id:
        task_passed = bool(
            (not any_kinematic_manipulated)
            and final_taper_err_mm <= 4.5
            and socket_mouth_err_mm <= 4.5
            and final_coax_err_deg <= 3.5
            and final_keck_err_mm <= 15.0
            and keck_skew_deg <= 15.0
            and post_drift_mm <= 3.5
        )
    elif "joint_04" in joint_id:
        task_passed = bool(
            (not any_kinematic_manipulated)
            and final_taper_err_mm <= 4.0
            and final_coax_err_deg <= 2.5
            and bulb_flush_err_mm <= 2.0
            and post_drift_mm <= 2.5
        )
    elif joint_id == "ab_05_unclamped_15deg_hold":
        task_passed = bool(
            (not any_kinematic_manipulated)
            and final_taper_err_mm <= 4.5
            and final_coax_err_deg <= 3.0
            and post_drift_mm <= 3.5
        )
    elif joint_id == "ab_06_disassembly_pullout":
        task_passed = bool((not any_kinematic_manipulated) and extracted_dz_mm >= 42.0)
    else:
        task_passed = False

    videos_out: dict[str, str] = {}
    if record_video and frames_triple:
        vdir = out_dir / "videos"
        vdir.mkdir(parents=True, exist_ok=True)
        for vtag, flist in (
            ("triple_sync", frames_triple),
            ("front_master", frames_front),
            ("ortho_side_90deg", frames_side),
            ("macro_closeup", frames_macro),
        ):
            vp = vdir / f"{vtag}.mp4"
            _encode_h264_mp4(flist, vp, fps=30)
            videos_out[vtag] = vp.as_posix()

    npz_path = out_dir / "telemetry_240hz.npz"
    np.savez_compressed(
        npz_path,
        target_pos=target_pos_hist,
        taper_err_mm=taper_err_mm_hist,
        keck_err_mm=keck_err_mm_hist,
        keck_jaw_deg=keck_jaw_deg_hist,
        pad_force_L_n=pad_force_L_hist,
        pad_force_R_n=pad_force_R_hist,
        wedge_radial_force_n=wedge_radial_force_hist,
        axial_pull_force_n=axial_pull_force_hist,
        flask_tug_mm=flask_tug_mm_hist,
        live_fingers_m=live_finger_hist,
        qL_cmd=qL_seq,
        qR_cmd=qR_seq,
    )

    metrics = {
        "joint_id": joint_id,
        "variant": variant,
        "zero_friction": zero_friction,
        "any_kinematic_manipulated": any_kinematic_manipulated,
        "physx_hz": 240,
        "final_taper_3d_err_mm": round(final_taper_err_mm, 3),
        "final_socket_mouth_err_mm": round(socket_mouth_err_mm, 3),
        "final_taper_coax_err_deg": round(final_coax_err_deg, 3),
        "final_keck_axial_skew_deg": round(keck_skew_deg, 2),
        "target_net_dz_mm": round(dz_net_mm, 2),
        "post_release_drift_mm": round(post_drift_mm, 3),
        "peak_finger_normal_force_n": round(peak_force_n, 2),
        "peak_wedge_radial_force_n": round(peak_wedge_radial_n, 2),
        "final_wedge_radial_force_n": round(final_wedge_radial_n, 2),
        "peak_axial_pull_force_n": round(peak_axial_pull_n, 2),
        "peak_flask_tug_mm": round(peak_flask_tug_mm, 3),
        "extracted_dz_mm": round(extracted_dz_mm, 2),
        "max_extract_dz_mm": round(max_extract_dz_mm, 2),
        "cumulative_screw_twist_deg": round(peak_screw_deg, 2),
        "final_keck_3d_err_mm": round(final_keck_err_mm, 3),
        "peak_keck_jaw_cam_open_deg": round(peak_keck_jaw_open_deg, 2),
        "thermometer_bulb_flush_err_mm": round(bulb_flush_err_mm, 3),
        "task_passed": task_passed,
    }
    payload = {
        "episode_id": str(job["episode_id"]),
        "joint_id": joint_id,
        "variant": variant,
        "zero_friction": zero_friction,
        "metrics": metrics,
        "telemetry_npz": npz_path.as_posix(),
        "videos": videos_out,
        "keyframes_dir": keyframes_dir.as_posix(),
    }
    res_out = job.get("result_json_path")
    if res_out:
        rp_out = Path(str(res_out))
        rp_out.parent.mkdir(parents=True, exist_ok=True)
        rp_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sys.stdout.flush()
    app.close()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    job["result_json_path"] = args.result_json.resolve().as_posix()
    res = execute_v08_job(job)
    if not args.result_json.is_file():
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
