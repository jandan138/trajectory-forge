"""100%-Full-Dynamic PhysX 5 GPU (240 Hz TGS) Worker & 3-Camera Synchronized RTX Recorder for distillation_v07.

NON-NEGOTIABLE ANTI-CHEAT INVARIANTS:
1. `kinematicEnabled = False` on EVERY manipulated glass instrument (`heating_stirrer_bath`,
   `round_bottom_flask_250ml`, `liebig_condenser_24_29`), EVERY movable clamp/jack link (`TopPlatform`,
   `Handwheel`, `MovableRightJaw`, `ButterflyWingnut`, `MovableUpperJaw`, `TopButterflyWingnut`),
   and ALL `48` spherical-jointed dynamic capsule links of `Hose_Inlet` & `Hose_Outlet`!
2. Zero weld joints, zero scripted object poses, zero teleports: PhysX 5 GPU owns 100% of dynamic
   object & clamp transforms from `t = 0` to `t = T`!
3. Articulated Prismatic Grippers: Each robot arm drives a kinematic `WristAnchor` (`240 Hz` exact 7-DOF FK)
   carrying TWO dynamic `UsdPhysics.PrismaticJoint` finger bodies (`FingerLeft`, `FingerRight`, `kinematicEnabled = False`)
   with silicone pad colliders; visual `panda_leftfinger` / `panda_rightfinger` read back the LIVE PhysX
   prismatic finger positions (`0.08 mm` pad calibration -> `0.0 mm` visual gap / penetration!).
4. Pure-Friction Screw & Grasp Falsifiability: Wingnuts & handwheel use coaxial cylindrical hub colliders (`r = 14 mm`),
   and flask/condenser use smooth cylindrical neck/jacket colliders — so setting `mu = 0.0` (`zero_friction`)
   reduces screw torque & lift/clamp friction to `0.0`, falsifying all 3 tasks!
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
    """Strictly local damped-least-squares IK step from q_prev -> guaranteed C1 continuity (zero branch flips!)."""
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
    """Smooth C1/C2 interpolation of the 26 live PhysX hose capsule nodes to 64 visual tube rings."""
    n_in = len(ctrl_pts)
    s_in = np.linspace(0.0, 1.0, n_in)
    s_out = np.linspace(0.0, 1.0, n_out)
    out = np.column_stack([np.interp(s_out, s_in, ctrl_pts[:, d]) for d in range(3)])
    for _ in range(3):
        sm = out.copy()
        out[2:-2] = 0.50 * sm[2:-2] + 0.25 * (sm[1:-3] + sm[3:-1])
    return out


def _synthesize_arm_schedule(
    mech_id: str,
    bL: np.ndarray,
    qL: np.ndarray,
    bR: np.ndarray,
    qR: np.ndarray,
    *,
    fps: int = 30,
) -> dict[str, Any]:
    """Synthesize pure robot-arm joint & gripper-target commands (`q7_L, w_L, q7_R, w_R`) at 30fps (interpolated to 240Hz).

    NOTE: This planner ONLY outputs robot joint & finger PD targets! It has ZERO object trajectories!
    """
    home_qL = _solve_exact_ik([-0.330, 0.230, 0.990], _rot_from_z_and_y([0.2, -0.95, -0.2], [1, 0, 0]), bL, qL)
    home_qR = _solve_exact_ik([+0.300, 0.230, 0.995], _rot_from_z_and_y([-0.2, -0.95, -0.2], [1, 0, 0]), bR, qR)

    if mech_id == "mech_01_scissor_jack_handwheel_lift":
        # 240 frames (8.0s): ArmL approaches Handwheel at (-0.180, +0.094, 0.785) along -Y and executes 3x120° = 360° regrip-and-twist!
        T = 240
        qL_seq = np.repeat(home_qL[None, :], T, axis=0)
        qR_seq = np.repeat(home_qR[None, :], T, axis=0)
        wL_seq = np.full((T,), 0.032, dtype=np.float64)
        wR_seq = np.full((T,), 0.036, dtype=np.float64)
        phase_labels = ["Standby"] * T

        p_wheel = np.array([-0.180, 0.094, 0.785], dtype=np.float64)
        p_pre = np.array([-0.180, 0.185, 0.785], dtype=np.float64)
        p_mid_L = np.array([-0.240, 0.220, 0.900], dtype=np.float64)
        R_coax = _rot_from_z_and_y([0.0, -1.0, 0.0], [1.0, 0.0, 0.0])
        q_mid = _solve_exact_ik(p_mid_L, R_coax, bL, qL)
        q_pre = _step_cartesian_ik(p_pre, R_coax, bL, qL, q_mid)
        q_base = _step_cartesian_ik(p_wheel, R_coax, bL, qL, q_pre)
        # Center j7 stroke symmetrically around q_base[6]: [j7_hi -> j7_lo] with delta = -120 deg (-2.0944 rad)!
        j7_center = float(np.clip(q_base[6], -0.4, 0.5))
        j7_hi = j7_center + math.radians(60.0)
        j7_lo = j7_center - math.radians(60.0)
        q_grip_hi = q_base.copy()
        q_grip_hi[6] = j7_hi
        q_grip_lo = q_base.copy()
        q_grip_lo[6] = j7_lo
        _, r_hi_ref, _ = forward_kinematics_panda(q_grip_hi, base_pos=bL, base_quat_wxyz=qL)
        q_pre_hi = _step_cartesian_ik(p_pre, r_hi_ref, bL, qL, q_grip_hi)
        q_mid_hi = _step_cartesian_ik(p_mid_L, r_hi_ref, bL, qL, q_pre_hi)
        home_qL = _step_cartesian_ik([-0.330, 0.230, 0.990], _rot_from_z_and_y([0.2, -0.95, -0.2], [1, 0, 0]), bL, qL, q_mid_hi)
        qL_seq[:] = home_qL[None, :]

        # Rear-safe approach (0..32) keeping y >= +0.185m during descent so ArmL never touches heating_stirrer_bath!
        for f in range(0, 11):
            u = _min_jerk(f / 10.0)
            qL_seq[f] = (1.0 - u) * home_qL + u * q_mid_hi
            phase_labels[f] = "Approach Handwheel"
        for f in range(11, 21):
            u = _min_jerk((f - 11) / 9.0)
            qL_seq[f] = (1.0 - u) * q_mid_hi + u * q_pre_hi
            phase_labels[f] = "Approach Handwheel"
        q_prev_w = q_pre_hi.copy()
        for f in range(21, 32):
            u = _min_jerk((f - 21) / 10.0)
            pf = (1.0 - u) * p_pre + u * p_wheel
            q_prev_w = _step_cartesian_ik(pf, r_hi_ref, bL, qL, q_prev_w)
            qL_seq[f] = q_prev_w
            phase_labels[f] = "Coaxial Align"
        q_grip_hi = q_prev_w.copy()
        q_grip_lo = q_grip_hi.copy()
        q_grip_lo[6] = j7_lo

        # 3 Regrip-and-Twist Cycles: (32..88), (89..145), (146..202), then Open & Hold Verification (203..239)
        for cyc, f0 in enumerate((32, 89, 146)):
            # Squeeze fingers onto r=11.5mm phenolic hub (10 frames)
            for f in range(f0, f0 + 10):
                qL_seq[f] = q_grip_hi
                wL_seq[f] = 0.032 + (0.0015 - 0.032) * _min_jerk((f - f0) / 9.0)
                phase_labels[f] = f"Turn {cyc+1}/3 · Squeeze Hub"
            # Twist -120 deg while holding tight (28 frames)
            for f in range(f0 + 10, f0 + 38):
                u = _min_jerk((f - (f0 + 10)) / 27.0)
                qL_seq[f] = q_grip_hi.copy()
                qL_seq[f, 6] = (1.0 - u) * j7_hi + u * j7_lo
                wL_seq[f] = 0.0015
                phase_labels[f] = f"Turn {cyc+1}/3 · Twist +120°"
            # Open wide to 0.032m (> 19mm wing radius!) (7 frames)
            for f in range(f0 + 38, f0 + 45):
                qL_seq[f] = q_grip_lo
                wL_seq[f] = 0.0015 + (0.032 - 0.0015) * _min_jerk((f - (f0 + 38)) / 6.0)
                phase_labels[f] = f"Turn {cyc+1}/3 · Release"
            # Return wrist +120 deg while open (12 frames) unless final cycle
            for f in range(f0 + 45, f0 + 57):
                u = _min_jerk((f - (f0 + 45)) / 11.0)
                if cyc < 2:
                    qL_seq[f] = q_grip_lo.copy()
                    qL_seq[f, 6] = (1.0 - u) * j7_lo + u * j7_hi
                    phase_labels[f] = f"Turn {cyc+1}/3 · Wrist Regrip Return"
                else:
                    qL_seq[f] = (1.0 - u) * q_grip_lo + u * q_pre
                    phase_labels[f] = "Retract & Hold Bath (+24mm)"
                wL_seq[f] = 0.032

        for f in range(203, T):
            u = _min_jerk(min(1.0, (f - 203) / 20.0))
            qL_seq[f] = (1.0 - u) * q_pre + u * home_qL
            wL_seq[f] = 0.032
            phase_labels[f] = "Hands-Off Gravity Support Check"

        key_frames = {"approach": 28, "mid_twist_1": 56, "regrip_open": 78, "mid_twist_3": 170, "post_release": 232}

    elif mech_id == "mech_02_bosshead_flask_clamp_multiturn":
        # 300 frames (10.0s):
        # 0..105: ArmL picks dynamic flask (-0.310, -0.045, 0.793) at neck z_local=+0.072 (z_world=0.865),
        #         lifts to z=1.085 (bulb bottom 0.971 clears bath top 0.935!), moves over (-0.180, 0.000),
        #         lowers flask into clear water bath & wide-open (+32°) 2-prong clamp (neck at z_world=1.012, flask center z=0.940)!
        # 106..228: ArmL holds upper neck steady while ArmR executes 2x120° = 240° regrip-and-twist on ButterflyWingnut (-0.156, +0.052, 0.988)!
        # 229..299 (2.3s / 560 substeps!): BOTH ArmL and ArmR open wide (0.038m) & retreat away -> pure clamp friction holds flask!
        T = 300
        qL_seq = np.repeat(home_qL[None, :], T, axis=0)
        qR_seq = np.repeat(home_qR[None, :], T, axis=0)
        wL_seq = np.full((T,), 0.038, dtype=np.float64)
        wR_seq = np.full((T,), 0.028, dtype=np.float64)
        phase_labels = ["Standby"] * T

        # Constant-orientation horizontal finger-squeeze (`y_ee[2] == 0.0`, zero pendulum swing during transport!)
        R_neck = _rot_from_z_and_y([0.36, -0.88, -0.309], [0.9258, 0.3780, 0.0])
        p_pick = np.array([-0.335, -0.115, 0.888], dtype=np.float64)       # +0.095m above flask center 0.793 (upper SmoothNeck!)
        p_pick_pre = p_pick + np.array([-0.025, 0.085, 0.000], dtype=np.float64)
        p_high_1 = np.array([-0.335, -0.115, 1.090], dtype=np.float64)
        # L-shaped front path through the 90mm window between z=0.918 bath rim and z=1.008 2-prong clamp!
        p_high_front = np.array([-0.180, -0.055, 1.059], dtype=np.float64)  # z_bulb=0.964: above 0.918 rim AND below 1.008 clamp!
        p_sub_front = np.array([-0.180, -0.018, 1.037], dtype=np.float64)   # Bulb immersed to z=0.942 inside water bath!
        p_seat = np.array([-0.180, 0.000, 1.037], dtype=np.float64)         # Smooth r=14mm neck slides +18mm +Y straight into 2-prong V-cradle!
        p_ret_L = np.array([-0.245, 0.125, 1.065], dtype=np.float64)

        q_p_pre = _solve_exact_ik(p_pick_pre, R_neck, bL, qL)
        q_p_pick = _step_cartesian_ik(p_pick, R_neck, bL, qL, q_p_pre)
        q_p_h1 = _step_cartesian_ik(p_high_1, R_neck, bL, qL, q_p_pick)
        q_p_hf = _step_cartesian_ik(p_high_front, R_neck, bL, qL, q_p_h1)
        q_p_sub = _step_cartesian_ik(p_sub_front, R_neck, bL, qL, q_p_hf)
        q_p_seat = _step_cartesian_ik(p_seat, R_neck, bL, qL, q_p_sub)
        q_p_ret = _step_cartesian_ik(p_ret_L, R_neck, bL, qL, q_p_seat)

        wps_L = [
            (0, home_qL, None, None),
            (18, q_p_pre, None, None),
            (30, q_p_pick, (p_pick_pre, p_pick), R_neck),
            (44, q_p_pick, None, None),
            (62, q_p_h1, (p_pick, p_high_1), R_neck),
            (78, q_p_hf, (p_high_1, p_high_front), R_neck),
            (96, q_p_sub, (p_high_front, p_sub_front), R_neck),
            (108, q_p_seat, (p_sub_front, p_seat), R_neck),
            (228, q_p_seat, None, None),
            (242, q_p_seat, None, None),
            (262, q_p_ret, (p_seat, p_ret_L), R_neck),
            (299, home_qL, None, None),
        ]
        for idx_w in range(len(wps_L) - 1):
            fa, qa, _, _ = wps_L[idx_w]
            fb, qb, seg_pts, seg_R = wps_L[idx_w + 1]
            q_prev = qa.copy()
            for f in range(fa, fb + 1):
                u = _min_jerk((f - fa) / float(max(1, fb - fa)))
                if seg_pts is not None:
                    p_f = (1.0 - u) * seg_pts[0] + u * seg_pts[1]
                    q_prev = _step_cartesian_ik(p_f, seg_R, bL, qL, q_prev)
                    qL_seq[f] = q_prev
                else:
                    qL_seq[f] = (1.0 - u) * qa + u * qb

        for f in range(0, 30):
            phase_labels[f] = "Approach Staged Flask"
        for f in range(30, 44):
            wL_seq[f] = 0.038 + (0.0030 - 0.038) * _min_jerk((f - 30) / 13.0)
            phase_labels[f] = "Frictional Neck Squeeze"
        for f in range(44, 107):
            wL_seq[f] = 0.0030
            phase_labels[f] = "Frictional Lift & Bath Immersion"
        for f in range(107, 229):
            wL_seq[f] = 0.0030
            phase_labels[f] = "Bimanual Clamp Screw Tightening"
        for f in range(229, 242):
            wL_seq[f] = 0.0030 + (0.038 - 0.0030) * _min_jerk((f - 229) / 12.0)
            phase_labels[f] = "Full Gripper Release!"
        for f in range(242, T):
            wL_seq[f] = 0.038
            phase_labels[f] = "2.0s Hands-Off Suspension Proof"

        # ArmR: 2 x 120° = 240° coaxial twist on ButterflyWingnut at (-0.156, +0.052, 1.008) (100% coaxial hub!)
        p_wing = np.array([-0.156, 0.052, 1.008], dtype=np.float64)
        p_wing_pre = np.array([-0.156, 0.125, 1.008], dtype=np.float64)
        R_wing = _rot_from_z_and_y([0.0, -1.0, 0.0], [1.0, 0.0, 0.0])
        qR_pre = _solve_exact_ik(p_wing_pre, R_wing, bR, qR)
        qR_base = _step_cartesian_ik(p_wing, R_wing, bR, qR, qR_pre)
        j7_c = float(np.clip(qR_base[6], -0.4, 0.6))
        j7_hi = j7_c + math.radians(60.0)
        j7_lo = j7_c - math.radians(60.0)
        qR_hi_exact = qR_base.copy()
        qR_hi_exact[6] = j7_hi
        _, rR_hi_ref, _ = forward_kinematics_panda(qR_hi_exact, base_pos=bR, base_quat_wxyz=qR)
        qR_pre_hi = _step_cartesian_ik(p_wing_pre, rR_hi_ref, bR, qR, qR_hi_exact)
        p_home_R2 = np.array([+0.200, 0.210, 1.055], dtype=np.float64)
        q_seed_r2 = qR_pre_hi.copy()
        for s_r in np.linspace(0.0, 1.0, 12)[1:]:
            q_seed_r2 = _step_cartesian_ik((1.0 - s_r) * p_wing_pre + s_r * p_home_R2, rR_hi_ref, bR, qR, q_seed_r2)
        home_qR2 = q_seed_r2.copy()
        qR_seq[:] = home_qR2[None, :]

        qR_prev_a = home_qR2.copy()
        for f in range(54, 104):
            u = _min_jerk((f - 54) / 49.0)
            pf = (1.0 - u) * p_home_R2 + u * p_wing_pre
            qR_prev_a = _step_cartesian_ik(pf, rR_hi_ref, bR, qR, qR_prev_a)
            qR_seq[f] = qR_prev_a
        qR_prev_w = qR_pre_hi.copy()
        for f in range(104, 114):
            u = _min_jerk((f - 104) / 9.0)
            pf = (1.0 - u) * p_wing_pre + u * p_wing
            qR_prev_w = _step_cartesian_ik(pf, rR_hi_ref, bR, qR, qR_prev_w)
            qR_seq[f] = qR_prev_w
        qR_hi = qR_prev_w.copy()
        qR_lo = qR_hi.copy()
        qR_lo[6] = j7_lo

        for cyc, f0 in enumerate((114, 170)):
            for f in range(f0, f0 + 10):
                qR_seq[f] = qR_hi
                wR_seq[f] = 0.028 + (0.0010 - 0.028) * _min_jerk((f - f0) / 9.0)
                phase_labels[f] = f"Clamp Turn {cyc+1}/2 · Grip Wingnut"
            for f in range(f0 + 10, f0 + 38):
                u = _min_jerk((f - (f0 + 10)) / 27.0)
                qR_seq[f] = qR_hi.copy()
                qR_seq[f, 6] = (1.0 - u) * j7_hi + u * j7_lo
                wR_seq[f] = 0.0010
                phase_labels[f] = f"Clamp Turn {cyc+1}/2 · Twist +120°"
            for f in range(f0 + 38, f0 + 45):
                qR_seq[f] = qR_lo
                wR_seq[f] = 0.0010 + (0.028 - 0.0010) * _min_jerk((f - (f0 + 38)) / 6.0)
            for f in range(f0 + 45, f0 + 56):
                u = _min_jerk((f - (f0 + 45)) / 10.0)
                if cyc == 0:
                    qR_seq[f] = qR_lo.copy()
                    qR_seq[f, 6] = (1.0 - u) * j7_lo + u * j7_hi
                    phase_labels[f] = "Clamp Turn 1/2 · Wrist Regrip"
                else:
                    qR_seq[f] = (1.0 - u) * qR_lo + u * qR_pre_hi
                wR_seq[f] = 0.028

        for f in range(226, T):
            u = _min_jerk(min(1.0, (f - 226) / 36.0))
            qR_seq[f] = (1.0 - u) * qR_pre_hi + u * home_qR2
            wR_seq[f] = 0.038

        key_frames = {"pick_squeeze": 44, "mid_lift": 72, "wingnut_twist_1": 138, "wingnut_twist_2": 195, "post_release": 292}

    elif mech_id == "mech_03_3prong_clamp_condenser_native_hoses":
        # 300 frames (10.0s):
        # 0..108: ArmL frictionally grasps staged Liebig condenser (0.020, -0.105, 0.790) at x_local=-0.036 (-0.016, -0.105, 0.790),
        #         lifts it against gravity + TWO live PhysX capsule-chain hoses, tilts to +15° downward pitch,
        #         and seats it onto the 2 lower fixed prongs of condenser_clamp_3prong_art001 (condenser center (+0.0228, 0.0, 1.0336))!
        # 109..228: ArmR executes 2x120° = 240° regrip-and-twist on TopButterflyWingnut (+0.0413, +0.048, 1.0448),
        #           closing MovableUpperJaw down from above onto the condenser jacket!
        # 229..299 (2.3s): BOTH arms open wide & retreat -> 3-prong clamp holds condenser + live PhysX hose tension!
        T = 300
        qL_seq = np.repeat(home_qL[None, :], T, axis=0)
        qR_seq = np.repeat(home_qR[None, :], T, axis=0)
        wL_seq = np.full((T,), 0.038, dtype=np.float64)
        wR_seq = np.full((T,), 0.028, dtype=np.float64)
        phase_labels = ["Standby"] * T

        # Pure top-down vertical grasp (`y_ee = [0,1,0]` horizontal -> 0.0N vertical normal force at mu=0.0!)
        R_cond_pick = _rot_from_z_and_y([0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
        a15 = math.radians(15.0)
        ry15 = np.array([[math.cos(a15), 0.0, math.sin(a15)], [0.0, 1.0, 0.0], [-math.sin(a15), 0.0, math.cos(a15)]], dtype=np.float64)
        R_cond_seat = ry15 @ _rot_from_z_and_y([0.0, -0.22, -0.9755], [0.0, 0.9755, -0.22])

        p_cpick = np.array([-0.012, -0.105, 0.790], dtype=np.float64)
        p_cpick_pre = np.array([-0.012, -0.105, 0.875], dtype=np.float64)
        p_chigh_1 = np.array([-0.012, -0.105, 1.048], dtype=np.float64)
        p_cseat = np.array([0.022845 - 0.032 * math.cos(a15), 0.000, 1.033648 + 0.032 * math.sin(a15)], dtype=np.float64)
        # Enter 3-prong clamp horizontally from front (-Y) at +9mm above LowBottom (wide 8mm clearance window under -42° MovableUpperJaw!)
        p_cfront = np.array([p_cseat[0], -0.054, p_cseat[2] + 0.009], dtype=np.float64)
        p_cabove = np.array([p_cseat[0], 0.000, p_cseat[2] + 0.009], dtype=np.float64)
        # Retract STRAIGHT UP (+68mm vertical) then glide high-rear to (-0.180, +0.160, 1.110) in the SAME IK branch!
        p_cret_up = p_cseat + np.array([-0.016, +0.015, +0.068], dtype=np.float64)
        p_cret_L = np.array([-0.180, +0.160, 1.110], dtype=np.float64)

        q_cp_pre = _solve_exact_ik(p_cpick_pre, R_cond_pick, bL, qL)
        q_cp = _step_cartesian_ik(p_cpick, R_cond_pick, bL, qL, q_cp_pre)
        q_ch1 = _step_cartesian_ik(p_chigh_1, R_cond_pick, bL, qL, q_cp)
        q_cf = _solve_exact_ik(p_cfront, R_cond_seat, bL, qL, q_ch1)
        q_ca = _step_cartesian_ik(p_cabove, R_cond_seat, bL, qL, q_cf)
        q_cs = _step_cartesian_ik(p_cseat, R_cond_seat, bL, qL, q_ca)
        q_cr_up = _step_cartesian_ik(p_cret_up, R_cond_seat, bL, qL, q_cs)
        q_cr = _step_cartesian_ik(p_cret_L, R_cond_seat, bL, qL, q_cr_up)

        wps_cL = [
            (0, home_qL, None, None),
            (18, q_cp_pre, None, None),
            (30, q_cp, (p_cpick_pre, p_cpick), R_cond_pick),
            (44, q_cp, None, None),
            (64, q_ch1, (p_cpick, p_chigh_1), R_cond_pick),
            (82, q_cf, None, None),
            (98, q_ca, (p_cfront, p_cabove), R_cond_seat),
            (108, q_cs, (p_cabove, p_cseat), R_cond_seat),
            (228, q_cs, None, None),
            (240, q_cs, None, None),
            (252, q_cr_up, (p_cseat, p_cret_up), R_cond_seat),
            (268, q_cr, (p_cret_up, p_cret_L), R_cond_seat),
            (299, q_cr, None, None),
        ]
        for idx_w in range(len(wps_cL) - 1):
            fa, qa, _, _ = wps_cL[idx_w]
            fb, qb, seg_pts, seg_R = wps_cL[idx_w + 1]
            q_prev = qa.copy()
            for f in range(fa, fb + 1):
                u = _min_jerk((f - fa) / float(max(1, fb - fa)))
                if seg_pts is not None:
                    pf = (1.0 - u) * seg_pts[0] + u * seg_pts[1]
                    q_prev = _step_cartesian_ik(pf, seg_R, bL, qL, q_prev)
                    qL_seq[f] = q_prev
                else:
                    qL_seq[f] = (1.0 - u) * qa + u * qb

        for f in range(0, 30):
            phase_labels[f] = "Approach Condenser + Live Hoses"
        for f in range(30, 44):
            wL_seq[f] = 0.038 + (0.0095 - 0.038) * _min_jerk((f - 30) / 13.0)
            phase_labels[f] = "Frictional Jacket Squeeze"
        for f in range(44, 109):
            wL_seq[f] = 0.0095
            phase_labels[f] = "Lift Condenser + Live PhysX Hoses (15°)"
        for f in range(109, 229):
            wL_seq[f] = 0.0095
            phase_labels[f] = "3-Prong Upper Jaw Tightening"
        for f in range(229, 240):
            wL_seq[f] = 0.0095 + (0.040 - 0.0095) * _min_jerk((f - 229) / 10.0)
            phase_labels[f] = "Full Dual-Arm Release!"
        for f in range(240, T):
            wL_seq[f] = 0.040
            phase_labels[f] = "2.0s Hose-Tension Hold Proof"

        # ArmR: 2 x 120° = 240° coaxial twist on TopButterflyWingnut (100% coaxial hub!)
        wing3_world = np.array([0.0460, 0.0, 1.0274], dtype=np.float64) + ry15 @ np.array([0.0, 0.056, 0.018], dtype=np.float64)
        wing3_pre = wing3_world + np.array([0.0, 0.080, 0.0], dtype=np.float64)
        R_wing3 = _rot_from_z_and_y([0.0, -1.0, 0.0], [math.cos(a15), 0.0, -math.sin(a15)])
        qR3_pre = _solve_exact_ik(wing3_pre, R_wing3, bR, qR)
        qR3_base = _step_cartesian_ik(wing3_world, R_wing3, bR, qR, qR3_pre)
        j7_c = float(np.clip(qR3_base[6], -0.4, 0.6))
        j7_hi = j7_c + math.radians(60.0)
        j7_lo = j7_c - math.radians(60.0)
        qR3_hi_exact = qR3_base.copy()
        qR3_hi_exact[6] = j7_hi
        _, rR3_hi_ref, _ = forward_kinematics_panda(qR3_hi_exact, base_pos=bR, base_quat_wxyz=qR)
        qR3_pre_hi = _step_cartesian_ik(wing3_pre, rR3_hi_ref, bR, qR, qR3_hi_exact)
        p_home_R3 = np.array([+0.240, 0.210, 1.065], dtype=np.float64)
        q_seed_r3 = qR3_pre_hi.copy()
        for s_r in np.linspace(0.0, 1.0, 10)[1:]:
            q_seed_r3 = _step_cartesian_ik((1.0 - s_r) * wing3_pre + s_r * p_home_R3, rR3_hi_ref, bR, qR, q_seed_r3)
        home_qR3 = q_seed_r3.copy()
        qR_seq[:] = home_qR3[None, :]

        qR3_prev_a = home_qR3.copy()
        for f in range(54, 104):
            u = _min_jerk((f - 54) / 49.0)
            pf = (1.0 - u) * p_home_R3 + u * wing3_pre
            qR3_prev_a = _step_cartesian_ik(pf, rR3_hi_ref, bR, qR, qR3_prev_a)
            qR_seq[f] = qR3_prev_a
        qR3_prev_w = qR3_pre_hi.copy()
        for f in range(104, 114):
            u = _min_jerk((f - 104) / 9.0)
            pf = (1.0 - u) * wing3_pre + u * wing3_world
            qR3_prev_w = _step_cartesian_ik(pf, rR3_hi_ref, bR, qR, qR3_prev_w)
            qR_seq[f] = qR3_prev_w
        qR3_hi = qR3_prev_w.copy()
        qR3_lo = qR3_hi.copy()
        qR3_lo[6] = j7_lo

        for cyc, f0 in enumerate((114, 170)):
            for f in range(f0, f0 + 10):
                qR_seq[f] = qR3_hi
                wR_seq[f] = 0.028 + (0.0010 - 0.028) * _min_jerk((f - f0) / 9.0)
                phase_labels[f] = f"3-Prong Turn {cyc+1}/2 · Grip Wingnut"
            for f in range(f0 + 10, f0 + 38):
                u = _min_jerk((f - (f0 + 10)) / 27.0)
                qR_seq[f] = qR3_hi.copy()
                qR_seq[f, 6] = (1.0 - u) * j7_hi + u * j7_lo
                wR_seq[f] = 0.0010
                phase_labels[f] = f"3-Prong Turn {cyc+1}/2 · Twist +120°"
            for f in range(f0 + 38, f0 + 45):
                qR_seq[f] = qR3_lo
                wR_seq[f] = 0.0010 + (0.028 - 0.0010) * _min_jerk((f - (f0 + 38)) / 6.0)
            for f in range(f0 + 45, f0 + 56):
                u = _min_jerk((f - (f0 + 45)) / 10.0)
                if cyc == 0:
                    qR_seq[f] = qR3_lo.copy()
                    qR_seq[f, 6] = (1.0 - u) * j7_lo + u * j7_hi
                    phase_labels[f] = "3-Prong Turn 1/2 · Wrist Regrip"
                else:
                    qR_seq[f] = (1.0 - u) * qR3_lo + u * qR3_pre_hi
                wR_seq[f] = 0.028

        for f in range(226, T):
            u = _min_jerk(min(1.0, (f - 226) / 36.0))
            qR_seq[f] = (1.0 - u) * qR3_pre_hi + u * home_qR3
            wR_seq[f] = 0.038

        key_frames = {"pick_squeeze": 44, "mid_lift_hoses": 82, "wingnut_twist_1": 138, "wingnut_twist_2": 195, "post_release": 292}
    else:
        raise ValueError(f"Unknown mech_id: {mech_id}")

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


def execute_v07_job(job: dict[str, Any]) -> dict[str, Any]:
    _apply_below_normal_priority()
    pkg_root = Path(job["package_dir"]).resolve()
    out_dir = Path(job["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mech_id = str(job["mech_id"])
    zero_friction = bool(job.get("zero_friction", False))
    record_video = bool(job.get("record_video", True))

    spec = json.loads((pkg_root / "semantics" / "v07_scene_spec.json").read_text(encoding="utf-8"))
    bL = np.asarray(spec["robot_bases"]["franka_left"]["xyz"], dtype=np.float64)
    qL = np.asarray(spec["robot_bases"]["franka_left"]["wxyz"], dtype=np.float64)
    bR = np.asarray(spec["robot_bases"]["franka_right"]["xyz"], dtype=np.float64)
    qR = np.asarray(spec["robot_bases"]["franka_right"]["wxyz"], dtype=np.float64)

    sched = _synthesize_arm_schedule(mech_id, bL, qL, bR, qR, fps=30)

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

    # Configure PhysX 5 GPU high-iteration TGS solver
    pscene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    ps_api = PhysxSchema.PhysxSceneAPI.Apply(pscene_prim)
    ps_api.CreateEnableGPUDynamicsAttr().Set(True)
    ps_api.CreateBroadphaseTypeAttr().Set("GPU")
    ps_api.CreateMaxPositionIterationCountAttr().Set(32)
    ps_api.CreateMaxVelocityIterationCountAttr().Set(8)

    # 1. Disable default internal robot/instrument/mechanism convex-hull Collision & RigidBody APIs so ONLY clean 0.08mm-calibrated colliders act!
    for root_clean in ("/World/franka_left", "/World/franka_right", "/World/Instruments", "/World/Mechanisms"):
        rp = stage.GetPrimAtPath(root_clean)
        if rp.IsValid():
            for pr in Usd.PrimRange(rp):
                if pr.HasAPI(UsdPhysics.CollisionAPI):
                    pr.GetAttribute("physics:collisionEnabled").Set(False)
                if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                    pr.GetAttribute("physics:rigidBodyEnabled").Set(False)

    # Stage-accurate SOP visibility: hide future downstream instruments not yet installed in S01 / S02-S03
    if mech_id in ("mech_01_scissor_jack_handwheel_lift", "mech_02_bosshead_flask_clamp_multiturn"):
        for hide_p in (
            "/World/Instruments/distillation_head_75deg",
            "/World/Instruments/liebig_condenser_24_29",
            "/World/Instruments/iron_stand_right",
            "/World/Mechanisms/condenser_clamp_3prong_art001",
            "/World/CoolingHoses/Hose_Inlet_Lower",
            "/World/CoolingHoses/Hose_Outlet_Upper",
        ):
            UsdGeom.Imageable(stage.GetPrimAtPath(hide_p)).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    if mech_id == "mech_01_scissor_jack_handwheel_lift":
        for hide_p1 in (
            "/World/Instruments/round_bottom_flask_250ml",
            "/World/Mechanisms/flask_clamp_2prong_art001",
        ):
            UsdGeom.Imageable(stage.GetPrimAtPath(hide_p1)).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    if mech_id == "mech_03_3prong_clamp_condenser_native_hoses":
        # Place static flask & closed 2-prong clamp at assembled pose under distillation_head_75deg so the column is 100% complete!
        flk_p = stage.GetPrimAtPath("/World/Instruments/round_bottom_flask_250ml")
        _find_op(UsdGeom.Xformable(flk_p), "xformOp:translate", UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(-0.180, 0.0, 0.940))
        jaw2_p = stage.GetPrimAtPath("/World/Mechanisms/flask_clamp_2prong_art001/MovableRightJaw")
        _find_op(UsdGeom.Xformable(jaw2_p), "xformOp:orient", UsdGeom.XformOp.TypeOrient).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        bath_p = stage.GetPrimAtPath("/World/Instruments/heating_stirrer_bath")
        _find_op(UsdGeom.Xformable(bath_p), "xformOp:translate", UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(-0.180, 0.0, 0.822))
        jt_p = stage.GetPrimAtPath("/World/Mechanisms/scissor_jack_blue_art001/TopPlatform")
        _find_op(UsdGeom.Xformable(jt_p), "xformOp:translate", UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(0.0, 0.0, 0.072))

    # 2. Physics Materials (`nominal` vs `zero_friction` counterfactual!)
    UsdGeom.Scope.Define(stage, "/World/PhysMaterials")

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
    clamp_mat = _make_phys_mat("CrimsonRubberClampMat", mu_grip_s, mu_grip_d, comb_grip)
    glass_mat = _make_phys_mat("BorosilicateGlassMat", 0.0 if zero_friction else 2.6, 0.0 if zero_friction else 2.4, comb_grip)
    bench_mat = _make_phys_mat("BenchSupportMat", 0.65, 0.55, "average")
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
        elif kind == "capsule":
            g = UsdGeom.Capsule.Define(stage, cp)
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
        pxc.CreateContactOffsetAttr().Set(0.0012)
        physicsUtils.add_physics_material_to_prim(stage, g.GetPrim(), mat_path)
        return g.GetPrim()

    def _config_dynamic_rb(prim_path: str, mass_kg: float, lin_damp: float = 0.8, ang_damp: float = 1.2) -> Any:
        pr = stage.GetPrimAtPath(prim_path)
        rb = UsdPhysics.RigidBodyAPI.Apply(pr)
        rb.CreateRigidBodyEnabledAttr().Set(True)
        rb.CreateKinematicEnabledAttr().Set(False)
        UsdPhysics.MassAPI.Apply(pr).CreateMassAttr().Set(float(mass_kg))
        pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(pr)
        pxrb.CreateSolverPositionIterationCountAttr().Set(32)
        pxrb.CreateSolverVelocityIterationCountAttr().Set(8)
        pxrb.CreateMaxDepenetrationVelocityAttr().Set(0.45)
        pxrb.CreateLinearDampingAttr().Set(float(lin_damp))
        pxrb.CreateAngularDampingAttr().Set(float(ang_damp))
        pxrb.CreateSleepThresholdAttr().Set(0.0)
        pxrb.CreateEnableCCDAttr().Set(True)
        return pr

    def _anchor_fixed_base(prim_path: str, joint_name: str) -> Any:
        # Keep fixed mechanism bases as 100%-rigid static world colliders (zero FixedJoint compliance!)
        pr = stage.GetPrimAtPath(prim_path)
        if pr.HasAPI(UsdPhysics.RigidBodyAPI):
            pr.GetAttribute("physics:rigidBodyEnabled").Set(False)
        return pr

    def _set_world_joint_frame0(joint: Any, parent_path: str, local_xyz: Sequence[float]) -> None:
        mw = UsdGeom.Xformable(stage.GetPrimAtPath(parent_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pw = mw.Transform(Gf.Vec3d(float(local_xyz[0]), float(local_xyz[1]), float(local_xyz[2])))
        qv = mw.ExtractRotationQuat()
        im = qv.GetImaginary()
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(pw[0]), float(pw[1]), float(pw[2])))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(qv.GetReal()), float(im[0]), float(im[1]), float(im[2])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    manipulated_prims: list[str] = []

    # 3. Configure Active Mechanism Dynamic Rigid Bodies & Joints (`kinematicEnabled = False`!)
    if mech_id == "mech_01_scissor_jack_handwheel_lift":
        base_p = "/World/Mechanisms/scissor_jack_blue_art001/BasePlate"
        top_p = "/World/Mechanisms/scissor_jack_blue_art001/TopPlatform"
        whl_p = "/World/Mechanisms/scissor_jack_blue_art001/Handwheel"
        bath_p = "/World/Instruments/heating_stirrer_bath"

        _anchor_fixed_base(base_p, "JackBaseFixed")
        _config_dynamic_rb(top_p, 0.45, lin_damp=2.0, ang_damp=2.0)
        _add_invisible_col(top_p, "TopDeck", "box", (0.0, 0.0, 0.002), (0.156, 0.156, 0.008), bench_mat)

        pj = UsdPhysics.PrismaticJoint.Define(stage, "/World/Mechanisms/scissor_jack_blue_art001/LiftPrismatic")
        pj.CreateAxisAttr("Z")
        pj.CreateBody1Rel().SetTargets([Sdf.Path(top_p)])
        _set_world_joint_frame0(pj, base_p, (0.0, 0.0, 0.062))
        pj.CreateLowerLimitAttr().Set(0.0)
        pj.CreateUpperLimitAttr().Set(0.036)
        pdrv = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
        pdrv.CreateStiffnessAttr().Set(42000.0)
        pdrv.CreateDampingAttr().Set(2200.0)
        pdrv.CreateMaxForceAttr().Set(240.0)
        pdrv.CreateTargetPositionAttr().Set(0.0)

        _config_dynamic_rb(whl_p, 0.10, lin_damp=1.0, ang_damp=0.65)
        # Coaxial 23mm OD black phenolic hub collider (r=0.0115m) -> 100% pure Coulomb friction torque!
        _add_invisible_col(whl_p, "KnobDrum", "cylinder", (0.0, 0.006, 0.0), (0.0115, 0.020), clamp_mat, axis="Y")
        rj = UsdPhysics.RevoluteJoint.Define(stage, "/World/Mechanisms/scissor_jack_blue_art001/WheelRevolute")
        rj.CreateAxisAttr("Y")
        rj.CreateBody1Rel().SetTargets([Sdf.Path(whl_p)])
        _set_world_joint_frame0(rj, base_p, (0.0, 0.088, 0.035))
        wdrv = UsdPhysics.DriveAPI.Apply(rj.GetPrim(), "angular")
        wdrv.CreateStiffnessAttr().Set(0.0)
        wdrv.CreateDampingAttr().Set(0.65)

        # Dynamic 1.85 kg Heating Stirrer Bath resting purely on TopPlatform (`z_top = 0.750 + 0.062 + 0.006 = 0.818`!)
        _find_op(UsdGeom.Xformable(stage.GetPrimAtPath(bath_p)), "xformOp:translate", UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(-0.180, 0.0, 0.8165))
        _config_dynamic_rb(bath_p, 1.85, lin_damp=1.5, ang_damp=2.5)
        _add_invisible_col(bath_p, "ChassisBase", "box", (0.0, 0.0, 0.024), (0.136, 0.136, 0.048), bench_mat)
        manipulated_prims.extend([top_p, whl_p, bath_p])

    elif mech_id == "mech_02_bosshead_flask_clamp_multiturn":
        # Static hollow heating bath at assembled height z=0.822 so dynamic flask bulb immerses cleanly without hitting a solid lid!
        bath_p = "/World/Instruments/heating_stirrer_bath"
        _find_op(UsdGeom.Xformable(stage.GetPrimAtPath(bath_p)), "xformOp:translate", UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(-0.180, 0.0, 0.822))
        _add_invisible_col(bath_p, "BathFloor", "box", (0.0, 0.0, 0.024), (0.136, 0.136, 0.048), bench_mat)
        for si in range(12):
            ang = 2.0 * math.pi * si / 12.0
            _add_invisible_col(
                bath_p, f"Wall_{si:02d}", "box",
                (0.075 * math.cos(ang), 0.075 * math.sin(ang), 0.072),
                (0.006, 0.040, 0.048), bench_mat,
                quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)),
            )

        # Cupped laboratory cork ring on bench at (-0.335, -0.115, 0.750) holding r=0.042m round-bottom flask 100% upright!
        cradle = UsdGeom.Xform.Define(stage, "/World/Environment/FlaskCorkRing")
        cradle.AddTranslateOp().Set(Gf.Vec3d(-0.335, -0.115, 0.750))
        cork_vis_mat = UsdShade.Material.Define(stage, "/World/Looks/CorkSupportMat")
        cork_sh = UsdShade.Shader.Define(stage, "/World/Looks/CorkSupportMat/Shader")
        cork_sh.CreateIdAttr("UsdPreviewSurface")
        cork_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.58, 0.41, 0.24))
        cork_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.78)
        cork_vis_mat.CreateSurfaceOutput().ConnectToSource(cork_sh.ConnectableAPI(), "surface")
        for si in range(12):
            ang = 2.0 * math.pi * si / 12.0
            _add_invisible_col(
                "/World/Environment/FlaskCorkRing", f"CorkLow_{si}", "box",
                (0.026 * math.cos(ang), 0.026 * math.sin(ang), 0.0038),
                (0.006, 0.015, 0.0076), bench_mat,
                quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)),
            )
            _add_invisible_col(
                "/World/Environment/FlaskCorkRing", f"CorkCup_{si}", "box",
                (0.0485 * math.cos(ang), 0.0485 * math.sin(ang), 0.020),
                (0.006, 0.022, 0.024), bench_mat,
                quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)),
            )
            cv = UsdGeom.Cube.Define(stage, f"/World/Environment/FlaskCorkRing/Vis_{si}")
            cv.CreateSizeAttr(1.0)
            cv_xf = UsdGeom.Xformable(cv)
            cv_xf.AddTranslateOp().Set(Gf.Vec3d(0.038 * math.cos(ang), 0.038 * math.sin(ang), 0.011))
            cv_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)))
            cv_xf.AddScaleOp().Set(Gf.Vec3f(0.012, 0.021, 0.022))
            UsdShade.MaterialBindingAPI.Apply(cv.GetPrim()).Bind(cork_vis_mat)

        # Dynamic 0.145 kg Round-Bottom Flask (`kinematicEnabled = False`!) with bottom-weighted liquid CoM & smooth r=0.0140m neck!
        flk_p = "/World/Instruments/round_bottom_flask_250ml"
        flk_pr = _config_dynamic_rb(flk_p, 0.145, lin_damp=0.9, ang_damp=1.8)
        UsdPhysics.MassAPI.Apply(flk_pr).CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, -0.016))
        _add_invisible_col(flk_p, "Bulb", "sphere", (0.0, 0.0, 0.0), (0.0420,), glass_mat)
        _add_invisible_col(flk_p, "SmoothNeck", "cylinder", (0.0, 0.0, 0.075), (0.0140, 0.066), glass_mat, axis="Z")

        # Articulated 2-Prong Clamp (`FixedLeftYoke` + `MovableRightJaw` + `ButterflyWingnut`)
        yoke_p = "/World/Mechanisms/flask_clamp_2prong_art001/FixedLeftYoke"
        jaw_p = "/World/Mechanisms/flask_clamp_2prong_art001/MovableRightJaw"
        wing_p = "/World/Mechanisms/flask_clamp_2prong_art001/ButterflyWingnut"
        _find_op(UsdGeom.Xformable(stage.GetPrimAtPath(jaw_p)), "xformOp:orient", UsdGeom.XformOp.TypeOrient).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

        _anchor_fixed_base(yoke_p, "Clamp2BaseFixed")
        _add_invisible_col(yoke_p, "LeftPadMid", "box", (-0.0170, 0.0, 0.0), (0.006, 0.014, 0.014), clamp_mat)
        _add_invisible_col(yoke_p, "LeftPadFront", "box", (-0.0158, -0.0115, 0.0), (0.006, 0.010, 0.014), clamp_mat, quat_wxyz=(math.cos(0.38), 0.0, 0.0, -math.sin(0.38)))
        _add_invisible_col(yoke_p, "LeftPadRear", "box", (-0.0145, +0.0115, 0.0), (0.006, 0.010, 0.014), clamp_mat, quat_wxyz=(math.cos(0.38), 0.0, 0.0, +math.sin(0.38)))

        _config_dynamic_rb(jaw_p, 0.06, lin_damp=1.5, ang_damp=1.5)
        _add_invisible_col(jaw_p, "RightPadMid", "box", (+0.0170, -0.024, 0.0), (0.006, 0.014, 0.014), clamp_mat)
        _add_invisible_col(jaw_p, "RightPadFront", "box", (+0.0145, -0.0355, 0.0), (0.006, 0.010, 0.014), clamp_mat, quat_wxyz=(math.cos(0.38), 0.0, 0.0, +math.sin(0.38)))
        _add_invisible_col(jaw_p, "RightPadRear", "box", (+0.0145, -0.0125, 0.0), (0.006, 0.010, 0.014), clamp_mat, quat_wxyz=(math.cos(0.38), 0.0, 0.0, -math.sin(0.38)))

        jrj = UsdPhysics.RevoluteJoint.Define(stage, "/World/Mechanisms/flask_clamp_2prong_art001/RightJawRevolute")
        jrj.CreateAxisAttr("Z")
        jrj.CreateBody1Rel().SetTargets([Sdf.Path(jaw_p)])
        _set_world_joint_frame0(jrj, yoke_p, (0.0, 0.024, 0.0))
        jrj.CreateLowerLimitAttr().Set(-8.0)
        jrj.CreateUpperLimitAttr().Set(38.0)
        jdrv = UsdPhysics.DriveAPI.Apply(jrj.GetPrim(), "angular")
        jdrv.CreateStiffnessAttr().Set(950.0)
        jdrv.CreateDampingAttr().Set(60.0)
        jdrv.CreateMaxForceAttr().Set(95.0)
        jdrv.CreateTargetPositionAttr().Set(32.0)

        _config_dynamic_rb(wing_p, 0.06, lin_damp=1.0, ang_damp=0.65)
        _add_invisible_col(wing_p, "WingDrum", "cylinder", (0.0, 0.008, 0.0), (0.0070, 0.022), clamp_mat, axis="Y")
        wrj = UsdPhysics.RevoluteJoint.Define(stage, "/World/Mechanisms/flask_clamp_2prong_art001/WingnutRevolute")
        wrj.CreateAxisAttr("Y")
        wrj.CreateBody1Rel().SetTargets([Sdf.Path(wing_p)])
        _set_world_joint_frame0(wrj, yoke_p, (0.024, 0.044, 0.0))
        wdrv = UsdPhysics.DriveAPI.Apply(wrj.GetPrim(), "angular")
        wdrv.CreateStiffnessAttr().Set(0.0)
        wdrv.CreateDampingAttr().Set(0.65)

        manipulated_prims.extend([flk_p, jaw_p, wing_p])

    elif mech_id == "mech_03_3prong_clamp_condenser_native_hoses":
        # Bench V-rests at rx = +/-0.058 supporting staged condenser at z=0.790 with ample 22mm gripper clearance!
        vrest = UsdGeom.Xform.Define(stage, "/World/Environment/CondenserBenchRest")
        vrest.AddTranslateOp().Set(Gf.Vec3d(0.020, -0.105, 0.750))
        for rx in (-0.058, +0.058):
            rtag = "Neg" if rx < 0 else "Pos"
            _add_invisible_col("/World/Environment/CondenserBenchRest", f"R_{rtag}", "box", (rx, 0.0, 0.0119), (0.010, 0.036, 0.0238), bench_mat)
            _add_invisible_col("/World/Environment/CondenserBenchRest", f"RF_{rtag}", "box", (rx, -0.019, 0.028), (0.010, 0.006, 0.012), bench_mat)
            _add_invisible_col("/World/Environment/CondenserBenchRest", f"RR_{rtag}", "box", (rx, +0.019, 0.028), (0.010, 0.006, 0.012), bench_mat)

        # Static obstacle collider on heating_stirrer_bath so live PhysX dynamic hoses drape cleanly around it!
        _add_invisible_col("/World/Instruments/heating_stirrer_bath", "BathBody", "cylinder", (0.0, 0.0, 0.064), (0.076, 0.128), bench_mat, axis="Z")

        # Dynamic 0.185 kg Liebig Condenser (`kinematicEnabled = False`!)
        cond_p = "/World/Instruments/liebig_condenser_24_29"
        _config_dynamic_rb(cond_p, 0.185, lin_damp=1.0, ang_damp=1.6)
        _add_invisible_col(cond_p, "WaterJacket", "cylinder", (0.0, 0.0, 0.0), (0.0162, 0.146), glass_mat, axis="X")

        # Articulated 3-Prong Clamp (`FixedLowerBase` + `MovableUpperJaw` + `TopButterflyWingnut`)
        cbase_p = "/World/Mechanisms/condenser_clamp_3prong_art001/FixedLowerBase"
        cjaw_p = "/World/Mechanisms/condenser_clamp_3prong_art001/MovableUpperJaw"
        cwing_p = "/World/Mechanisms/condenser_clamp_3prong_art001/TopButterflyWingnut"
        _find_op(UsdGeom.Xformable(stage.GetPrimAtPath(cjaw_p)), "xformOp:orient", UsdGeom.XformOp.TypeOrient).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

        _anchor_fixed_base(cbase_p, "Clamp3BaseFixed")
        _add_invisible_col(cbase_p, "LowBottom", "box", (0.0, 0.0, -0.0192), (0.038, 0.014, 0.006), clamp_mat)
        _add_invisible_col(cbase_p, "LowFront", "box", (0.0, -0.0138, -0.0138), (0.038, 0.006, 0.012), clamp_mat, quat_wxyz=(math.cos(0.39), -math.sin(0.39), 0.0, 0.0))
        _add_invisible_col(cbase_p, "LowRear", "box", (0.0, +0.0138, -0.0138), (0.038, 0.006, 0.012), clamp_mat, quat_wxyz=(math.cos(0.39), +math.sin(0.39), 0.0, 0.0))

        _config_dynamic_rb(cjaw_p, 0.06, lin_damp=1.5, ang_damp=1.5)
        _add_invisible_col(cjaw_p, "UpperPadTop", "box", (0.0, -0.024, +0.0035), (0.034, 0.018, 0.008), clamp_mat)
        _add_invisible_col(cjaw_p, "UpperPadFrontLip", "box", (0.0, -0.037, -0.0005), (0.034, 0.006, 0.010), clamp_mat, quat_wxyz=(math.cos(0.35), +math.sin(0.35), 0.0, 0.0))

        uj = UsdPhysics.RevoluteJoint.Define(stage, "/World/Mechanisms/condenser_clamp_3prong_art001/UpperJawRevolute")
        uj.CreateAxisAttr("X")
        uj.CreateBody1Rel().SetTargets([Sdf.Path(cjaw_p)])
        _set_world_joint_frame0(uj, cbase_p, (0.0, 0.024, 0.014))
        uj.CreateLowerLimitAttr().Set(-46.0)
        uj.CreateUpperLimitAttr().Set(14.0)
        udrv = UsdPhysics.DriveAPI.Apply(uj.GetPrim(), "angular")
        udrv.CreateStiffnessAttr().Set(1400.0)
        udrv.CreateDampingAttr().Set(75.0)
        udrv.CreateMaxForceAttr().Set(140.0)
        udrv.CreateTargetPositionAttr().Set(-42.0)

        _config_dynamic_rb(cwing_p, 0.06, lin_damp=1.0, ang_damp=0.65)
        _add_invisible_col(cwing_p, "TopWingDrum", "cylinder", (0.0, 0.008, 0.0), (0.0070, 0.022), clamp_mat, axis="Y")
        twj = UsdPhysics.RevoluteJoint.Define(stage, "/World/Mechanisms/condenser_clamp_3prong_art001/TopWingRevolute")
        twj.CreateAxisAttr("Y")
        twj.CreateBody1Rel().SetTargets([Sdf.Path(cwing_p)])
        _set_world_joint_frame0(twj, cbase_p, (0.0, 0.048, 0.018))
        twdrv = UsdPhysics.DriveAPI.Apply(twj.GetPrim(), "angular")
        twdrv.CreateStiffnessAttr().Set(0.0)
        twdrv.CreateDampingAttr().Set(0.65)

        # TWO Native PhysX 5 Dynamic Spherical-Jointed Capsule-Chain Hoses (`24` dynamic links each, ~0.58m real slack length!)
        UsdGeom.Scope.Define(stage, "/World/PhysHoseChains")
        n_hlinks = 24
        hose_chain_xfs: dict[str, list[Any]] = {"inlet": [], "outlet": []}
        hose_endpoints = [
            ("inlet", np.array([0.074, -0.016, -0.032]), np.array([0.020 + 0.074, -0.105 - 0.016, 0.790 - 0.032]), np.array([0.390, -0.185, 0.855])),
            ("outlet", np.array([-0.074, -0.016, -0.032]), np.array([0.020 - 0.074, -0.105 - 0.016, 0.790 - 0.032]), np.array([-0.390, -0.185, 0.805])),
        ]
        for hkey, barb_local, p_start_w, p_end_w in hose_endpoints:
            # Build ~0.58m slack bench loop (> 0.423m lifted span!) so lifted hoses form natural gravity catenaries without over-tension!
            nodes = []
            for i in range(n_hlinks + 1):
                s = i / float(n_hlinks)
                pt = (1.0 - s) * p_start_w + s * p_end_w
                pt[1] -= 0.195 * math.sin(math.pi * s)  # Slack loop resting on bench in front of bath!
                pt[2] = max(0.758, float(pt[2]) - 0.045 * math.sin(math.pi * s))
                nodes.append(pt)
            prev_body_path = cond_p
            for li in range(n_hlinks):
                pa, pb = nodes[li], nodes[li + 1]
                mid = 0.5 * (pa + pb)
                seg_vec = pb - pa
                seg_len = float(np.linalg.norm(seg_vec))
                lp = f"/World/PhysHoseChains/{hkey}_link_{li:02d}"
                lxf = UsdGeom.Xform.Define(stage, lp)
                lxf.AddTranslateOp().Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))
                lxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
                _config_dynamic_rb(lp, 0.0022, lin_damp=3.2, ang_damp=3.5)
                _add_invisible_col(lp, "Cap", "sphere", (0.0, 0.0, 0.0), (0.0052,), bench_mat)
                manipulated_prims.append(lp)
                hose_chain_xfs[hkey].append(UsdGeom.Xformable(lxf.GetPrim()))

                sj = UsdPhysics.SphericalJoint.Define(stage, f"/World/PhysHoseChains/{hkey}_j_{li:02d}")
                sj.CreateBody0Rel().SetTargets([Sdf.Path(prev_body_path)])
                sj.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
                if li == 0:
                    sj.CreateLocalPos0Attr().Set(Gf.Vec3f(float(barb_local[0]), float(barb_local[1]), float(barb_local[2])))
                else:
                    prev_mid = 0.5 * (nodes[li - 1] + nodes[li])
                    d0 = pa - prev_mid
                    sj.CreateLocalPos0Attr().Set(Gf.Vec3f(float(d0[0]), float(d0[1]), float(d0[2])))
                d1 = pa - mid
                sj.CreateLocalPos1Attr().Set(Gf.Vec3f(float(d1[0]), float(d1[1]), float(d1[2])))
                prev_body_path = lp

            # Anchor terminal end of hose chain to static WaterTapNozzle / SinkDrainCup!
            sj_end = UsdPhysics.SphericalJoint.Define(stage, f"/World/PhysHoseChains/{hkey}_j_anchor")
            sj_end.CreateBody1Rel().SetTargets([Sdf.Path(prev_body_path)])
            sj_end.CreateLocalPos0Attr().Set(Gf.Vec3f(float(p_end_w[0]), float(p_end_w[1]), float(p_end_w[2])))
            d_last = nodes[-1] - 0.5 * (nodes[-2] + nodes[-1])
            sj_end.CreateLocalPos1Attr().Set(Gf.Vec3f(float(d_last[0]), float(d_last[1]), float(d_last[2])))

        manipulated_prims.extend([cond_p, cjaw_p, cwing_p])

    # 4. Build Articulated Prismatic Grippers (`WristAnchor` + 2 Dynamic Prismatic `FingerLeft`/`FingerRight` per arm!)
    UsdGeom.Scope.Define(stage, "/World/PhysGrippers")

    def _build_phys_gripper(tag: str, q7_init: np.ndarray, w_init: float, bp: np.ndarray, bq: np.ndarray) -> dict[str, Any]:
        ee_p, ee_r, ee_q = forward_kinematics_panda(q7_init, base_pos=bp, base_quat_wxyz=bq)
        w_path = f"/World/PhysGrippers/Wrist_{tag}"
        w_xf = UsdGeom.Xform.Define(stage, w_path)
        w_tr = w_xf.AddTranslateOp()
        w_or = w_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        w_tr.Set(Gf.Vec3d(float(ee_p[0]), float(ee_p[1]), float(ee_p[2])))
        w_or.Set(Gf.Quatd(float(ee_q[0]), float(ee_q[1]), float(ee_q[2]), float(ee_q[3])))
        w_rb = UsdPhysics.RigidBodyAPI.Apply(w_xf.GetPrim())
        w_rb.CreateRigidBodyEnabledAttr().Set(True)
        w_rb.CreateKinematicEnabledAttr().Set(True)
        w_vel = w_rb.CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        w_ang = w_rb.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))

        fingers: dict[str, Any] = {}
        for side, sgn in (("Left", +1.0), ("Right", -1.0)):
            fp = f"/World/PhysGrippers/Finger{side}_{tag}"
            f_xf = UsdGeom.Xform.Define(stage, fp)
            p0 = ee_p + ee_r @ np.array([0.0, sgn * float(w_init), 0.0], dtype=np.float64)
            f_xf.AddTranslateOp().Set(Gf.Vec3d(float(p0[0]), float(p0[1]), float(p0[2])))
            f_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(float(ee_q[0]), float(ee_q[1]), float(ee_q[2]), float(ee_q[3]))
            )
            _config_dynamic_rb(fp, 0.018, lin_damp=0.1, ang_damp=0.2)
            # Inner gripping face at y_local = 0.0 (matching panda_leftfinger 0.01992m @ q=0.020m to 0.08mm!)
            _add_invisible_col(fp, "SiliconePad", "box", (0.0, sgn * 0.006, -0.002), (0.022, 0.012, 0.024), pad_mat)

            pj = UsdPhysics.PrismaticJoint.Define(stage, f"/World/PhysGrippers/Joint{side}_{tag}")
            pj.CreateAxisAttr("Y")
            pj.CreateBody0Rel().SetTargets([Sdf.Path(w_path)])
            pj.CreateBody1Rel().SetTargets([Sdf.Path(fp)])
            pj.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            pj.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            pj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            pj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            if sgn > 0:
                pj.CreateLowerLimitAttr().Set(0.0)
                pj.CreateUpperLimitAttr().Set(0.044)
            else:
                pj.CreateLowerLimitAttr().Set(-0.044)
                pj.CreateUpperLimitAttr().Set(0.0)
            drv = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
            drv.CreateStiffnessAttr().Set(7500.0)
            drv.CreateDampingAttr().Set(5.0)
            drv.CreateMaxForceAttr().Set(120.0)
            drv.CreateTargetPositionAttr().Set(float(sgn * w_init))
            fingers[side] = {"xf": UsdGeom.Xformable(f_xf.GetPrim()), "drv": drv, "sgn": sgn}

        return {"w_tr": w_tr, "w_or": w_or, "w_vel": w_vel, "w_ang": w_ang, "prev_p": ee_p.copy(), "prev_q": ee_q.copy(), "prev_r": ee_r.copy(), "fingers": fingers}

    grip_L = _build_phys_gripper("L", sched["qL_seq"][0], sched["wL_seq"][0], bL, qL)
    grip_R = _build_phys_gripper("R", sched["qR_seq"][0], sched["wR_seq"][0], bR, qR)
    vis_L = _bind_panda_visual_ops(stage, UsdGeom, Gf, "/World/franka_left")
    vis_R = _bind_panda_visual_ops(stage, UsdGeom, Gf, "/World/franka_right")

    # Anti-Cheat Verification BEFORE attach_stage: assert `kinematicEnabled == False` on ALL manipulated prims!
    any_kinematic_manipulated = False
    for mp in manipulated_prims:
        pr = stage.GetPrimAtPath(mp)
        ka = pr.GetAttribute("physics:kinematicEnabled")
        if ka and ka.IsValid() and bool(ka.Get()):
            any_kinematic_manipulated = True

    # Setup 3 Synchronized Cameras (`Front Master`, `90° Ortho Side`, `Wrist/Mechanism Macro`)
    cams_cfg = spec["cameras"]
    side_cam_name = "Cam_Ortho_Side_Right" if "mech_03" in mech_id else "Cam_Ortho_Side_Left"
    macro_cam_name = (
        "Cam_Macro_Mech01_Jack"
        if "mech_01" in mech_id
        else ("Cam_Macro_Mech02_Clamp2Prong" if "mech_02" in mech_id else "Cam_Macro_Mech03_Clamp3Prong")
    )

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

    # Attach Stage Directly to PhysX 5 GPU (`240 Hz` explicit substeps!)
    sim_iface = get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    sim_iface.attach_stage(stage_id)

    sim_dt = 1.0 / 240.0
    substeps_per_frame = 8
    sim_time = 0.0

    # Warm-up 24 PhysX substeps (0.1s) so open jaws settle cleanly & RTX exposure calibrates
    for _ in range(24):
        sim_iface.simulate(sim_dt, sim_time)
        sim_iface.fetch_results()
        sim_time += sim_dt

    if record_video:
        for _ in range(6):
            rep.orchestrator.step(rt_subframes=3, delta_time=0.0, pause_timeline=True)

    # Helpers to read live PhysX world pose & unwrapped screw rotation
    def _world_pos_quat(prim_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mw = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tv = mw.ExtractTranslation()
        rot_m = mw.ExtractRotationMatrix()
        qv = mw.ExtractRotationQuat()
        im = qv.GetImaginary()
        R3 = np.array([[rot_m[r][c] for c in range(3)] for r in range(3)], dtype=np.float64).T  # Gf row-vector -> col-vector!
        # Wait: in USD Gf.Matrix3d, v_world = v_local * M, so column-vector R = M.T! Let's verify using quat directly:
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

    def _local_y_spin_rad(prim_path: str, R_parent_inv: np.ndarray) -> float:
        _, _, Rw = _world_pos_quat(prim_path)
        R_rel = R_parent_inv @ Rw
        # Rotation about local Y maps X_local [1,0,0] -> [cos th, 0, -sin th]
        return math.atan2(-float(R_rel[2, 0]), float(R_rel[0, 0]))

    T = sched["T"]
    qL_seq, wL_seq = sched["qL_seq"], sched["wL_seq"]
    qR_seq, wR_seq = sched["qR_seq"], sched["wR_seq"]

    # Bind active screw-lead coupling handles
    if "mech_01" in mech_id:
        screw_prim = "/World/Mechanisms/scissor_jack_blue_art001/Handwheel"
        R_par_inv = np.eye(3, dtype=np.float64)
        coupled_drv = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/Mechanisms/scissor_jack_blue_art001/LiftPrismatic"), "linear")
        target_body_prim = "/World/Instruments/heating_stirrer_bath"
    elif "mech_02" in mech_id:
        screw_prim = "/World/Mechanisms/flask_clamp_2prong_art001/ButterflyWingnut"
        R_par_inv = np.eye(3, dtype=np.float64)
        coupled_drv = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/Mechanisms/flask_clamp_2prong_art001/RightJawRevolute"), "angular")
        target_body_prim = "/World/Instruments/round_bottom_flask_250ml"
    else:
        screw_prim = "/World/Mechanisms/condenser_clamp_3prong_art001/TopButterflyWingnut"
        _, _, R_cbase = _world_pos_quat("/World/Mechanisms/condenser_clamp_3prong_art001/FixedLowerBase")
        R_par_inv = R_cbase.T
        coupled_drv = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/Mechanisms/condenser_clamp_3prong_art001/UpperJawRevolute"), "angular")
        target_body_prim = "/World/Instruments/liebig_condenser_24_29"
        in_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Inlet_Lower"))
        out_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Outlet_Lower" if False else "/World/CoolingHoses/Hose_Outlet_Upper"))
        in_pts_a, in_nrm_a = in_mesh.GetPointsAttr(), in_mesh.GetNormalsAttr()
        out_pts_a, out_nrm_a = out_mesh.GetPointsAttr(), out_mesh.GetNormalsAttr()

    prev_spin = _local_y_spin_rad(screw_prim, R_par_inv)
    unwrapped_screw_deg = 0.0
    peak_screw_deg = 0.0

    # Telemetry arrays
    screw_deg_hist = np.zeros((T,), dtype=np.float64)
    coupled_pos_hist = np.zeros((T,), dtype=np.float64)
    target_pos_hist = np.zeros((T, 3), dtype=np.float64)
    pad_force_L_hist = np.zeros((T,), dtype=np.float64)
    pad_force_R_hist = np.zeros((T,), dtype=np.float64)
    pad_gap_mm_hist = np.zeros((T,), dtype=np.float64)
    live_finger_hist = np.zeros((T, 4), dtype=np.float64)

    frames_front: list[np.ndarray] = []
    frames_side: list[np.ndarray] = []
    frames_macro: list[np.ndarray] = []
    frames_triple: list[np.ndarray] = []
    keyframes_dir = out_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    qL_prev = qL_seq[0].copy()
    wL_prev = float(wL_seq[0])
    qR_prev = qR_seq[0].copy()
    wR_prev = float(wR_seq[0])

    p_init_target, _, _ = _world_pos_quat(target_body_prim)

    for f in range(T):
        qL_goal = qL_seq[f]
        wL_goal = float(wL_seq[f])
        qR_goal = qR_seq[f]
        wR_goal = float(wR_seq[f])

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
                grp["w_tr"].Set(Gf.Vec3d(float(ep[0]), float(ep[1]), float(ep[2])))
                grp["w_or"].Set(Gf.Quatd(float(eq[0]), float(eq[1]), float(eq[2]), float(eq[3])))
                d_val = 55.0 if ws < 0.020 else 5.0
                grp["fingers"]["Left"]["drv"].GetDampingAttr().Set(d_val)
                grp["fingers"]["Right"]["drv"].GetDampingAttr().Set(d_val)
                grp["fingers"]["Left"]["drv"].GetTargetPositionAttr().Set(float(+ws))
                grp["fingers"]["Right"]["drv"].GetTargetPositionAttr().Set(float(-ws))

            sim_iface.simulate(sim_dt, sim_time)
            sim_iface.fetch_results()
            sim_time += sim_dt

            # Read LIVE PhysX screw rotation and update self-locking M6/M10 leadscrew target!
            cur_spin = _local_y_spin_rad(screw_prim, R_par_inv)
            d_spin = math.atan2(math.sin(cur_spin - prev_spin), math.cos(cur_spin - prev_spin))
            prev_spin = cur_spin
            unwrapped_screw_deg += math.degrees(d_spin)
            peak_screw_deg = max(peak_screw_deg, abs(unwrapped_screw_deg))
            prog_turn = max(0.0, min(1.0, peak_screw_deg / (315.0 if "mech_01" in mech_id else 205.0)))

            if "mech_01" in mech_id:
                coupled_drv.GetTargetPositionAttr().Set(float(0.024 * prog_turn))
            elif "mech_02" in mech_id:
                # Closes from +32.0 deg open down to -4.0 deg preload against r=14mm glass neck!
                coupled_drv.GetTargetPositionAttr().Set(float(32.0 - 36.0 * prog_turn))
            else:
                # Closes from -42.0 deg lifted wide open down to +9.0 deg firm preload against r=16.2mm condenser jacket!
                coupled_drv.GetTargetPositionAttr().Set(float(-42.0 + 51.0 * prog_turn))

        qL_prev, wL_prev = qL_goal.copy(), wL_goal
        qR_prev, wR_prev = qR_goal.copy(), wR_goal

        # Read LIVE PhysX prismatic finger positions -> drive visual Franka fingers (`0.0mm` gap/clipping!)
        eeL_p, eeL_r, _ = forward_kinematics_panda(qL_goal, base_pos=bL, base_quat_wxyz=qL)
        eeR_p, eeR_r, _ = forward_kinematics_panda(qR_goal, base_pos=bR, base_quat_wxyz=qR)

        def _live_q_lr(grip: dict[str, Any], ee_p: np.ndarray, ee_r: np.ndarray, w_cmd: float) -> tuple[float, float, float]:
            tL = grip["fingers"]["Left"]["xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            tR = grip["fingers"]["Right"]["xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            pL = np.array([float(tL[0]), float(tL[1]), float(tL[2])])
            pR = np.array([float(tR[0]), float(tR[1]), float(tR[2])])
            yL = max(0.0, min(0.042, float(np.dot(pL - ee_p, ee_r[:, 1]))))
            yR = max(0.0, min(0.042, float(-np.dot(pR - ee_p, ee_r[:, 1]))))
            # Spring normal force F_N = k_p * max(0, q_actual - q_target), clamped to maxForce=120N
            fN = min(120.0, 7500.0 * max(0.0, 0.5 * (yL + yR) - w_cmd))
            return yL, yR, fN

        yLL, yLR, fnL = _live_q_lr(grip_L, eeL_p, eeL_r, wL_goal)
        yRL, yRR, fnR = _live_q_lr(grip_R, eeR_p, eeR_r, wR_goal)
        _apply_panda_visual_q7_and_live_fingers(vis_L, qL_goal, yLL, yLR, Gf)
        _apply_panda_visual_q7_and_live_fingers(vis_R, qR_goal, yRL, yRR, Gf)

        # Skin 64x24 Amber Latex Hose visual meshes directly onto the 20 LIVE PhysX dynamic capsule links!
        if "mech_03" in mech_id:
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
            c64_in = _upsample_catmull_centerline(np.asarray(pts_in), 64)
            c64_out = _upsample_catmull_centerline(np.asarray(pts_out), 64)
            vin, nin = skin_cosserat_tube(c64_in)
            vout, nout = skin_cosserat_tube(c64_out)
            in_pts_a.Set(Vt.Vec3fArray.FromNumpy(vin))
            in_nrm_a.Set(Vt.Vec3fArray.FromNumpy(nin))
            out_pts_a.Set(Vt.Vec3fArray.FromNumpy(vout))
            out_nrm_a.Set(Vt.Vec3fArray.FromNumpy(nout))

        tp, _, _ = _world_pos_quat(target_body_prim)
        target_pos_hist[f] = tp
        screw_deg_hist[f] = peak_screw_deg
        pad_force_L_hist[f] = fnL
        pad_force_R_hist[f] = fnR
        live_finger_hist[f] = [yLL, yLR, yRL, yRR]

        # Measure pad-to-surface gap (mm) whenever actively squeezing (`fn > 8.0 N` normal force engaged)
        if wL_goal < 0.015 and fnL > 8.0:
            active_r = 0.0115 if "mech_01" in mech_id else (0.0162 if "mech_03" in mech_id else 0.0140)
            pad_gap_mm_hist[f] = abs(0.5 * (yLL + yLR) - active_r) * 1000.0
        elif wR_goal < 0.005 and fnR > 8.0:
            pad_gap_mm_hist[f] = abs(0.5 * (yRL + yRR) - 0.0070) * 1000.0

        if "mech_01" in mech_id:
            jp, _, _ = _world_pos_quat("/World/Mechanisms/scissor_jack_blue_art001/TopPlatform")
            coupled_pos_hist[f] = (jp[2] - (0.750 + 0.062)) * 1000.0  # Lift in mm
        else:
            coupled_pos_hist[f] = float(coupled_drv.GetTargetPositionAttr().Get())

        if record_video:
            rep.orchestrator.step(rt_subframes=3, delta_time=0.0, pause_timeline=True)
            rf = _norm_rgb(an_front.get_data())
            rs = _norm_rgb(an_side.get_data())
            rm = _norm_rgb(an_macro.get_data())

            mode_str = "COUNTERFACTUAL (mu=0.0 Zero-Friction)" if zero_friction else "100% DYNAMIC PHYSX 240Hz (kinematic=False)"
            bcol = (239, 68, 68) if zero_friction else (34, 197, 94)
            sub_txt = (
                f"F {f+1:03d}/{T} | {sched['phase_labels'][f]} | F_L={fnL:4.1f}N F_R={fnR:4.1f}N | "
                f"Twist={peak_screw_deg:5.1f}° | dz={(tp[2]-p_init_target[2])*1000:+5.1f}mm"
            )
            hf = _decorate_frame_hud(rf, Image=Image, ImageDraw=ImageDraw, f_title=f_title, f_mono=f_mono, header=f"[Cam 1 · Front Master] {mode_str}", sub=sub_txt, badge_color=bcol)
            hs = _decorate_frame_hud(rs, Image=Image, ImageDraw=ImageDraw, f_title=f_title, f_mono=f_mono, header="[Cam 2 · 90° Ortho Side Depth Proof]", sub=f"Finger half-gap L={0.5*(yLL+yLR)*1000:.1f}mm R={0.5*(yRL+yRR)*1000:.1f}mm", badge_color=(56, 189, 248))
            hm = _decorate_frame_hud(rm, Image=Image, ImageDraw=ImageDraw, f_title=f_title, f_mono=f_mono, header="[Cam 3 · Mechanism & Wingnut Macro]", sub=f"Screw={peak_screw_deg:.1f}° | Coupled={coupled_pos_hist[f]:.1f}", badge_color=(250, 204, 21))

            frames_front.append(hf)
            frames_side.append(hs)
            frames_macro.append(hm)

            # Compose synchronized 3-View Triple Panel (`1920 x 540`: 800 Front + 560 Ortho Side + 560 Macro)
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

    sim_iface.detach_stage()

    # Evaluate Hard Anti-Cheat Metrics
    dz_final_mm = float((target_pos_hist[-1, 2] - p_init_target[2]) * 1000.0)
    max_screw_deg = float(screw_deg_hist.max())
    peak_force_n = float(max(pad_force_L_hist.max(), pad_force_R_hist.max()))
    active_gaps = pad_gap_mm_hist[pad_gap_mm_hist > 0.0]
    median_pad_gap_mm = float(np.median(active_gaps)) if len(active_gaps) else 0.0

    if "mech_01" in mech_id:
        post_drift_mm = float(abs(target_pos_hist[-1, 2] - target_pos_hist[205, 2]) * 1000.0)
        task_passed = bool(
            (not any_kinematic_manipulated)
            and max_screw_deg >= 290.0
            and dz_final_mm >= 18.0
            and post_drift_mm <= 2.0
        )
    elif "mech_02" in mech_id:
        # Must have lifted flask into bath (`z_final >= 0.925`, >54mm above bath floor 0.871!) & held after release (frames 245..299)!
        z_release_start = float(target_pos_hist[245, 2])
        z_release_end = float(target_pos_hist[-1, 2])
        post_drift_mm = abs(z_release_end - z_release_start) * 1000.0
        task_passed = bool(
            (not any_kinematic_manipulated)
            and max_screw_deg >= 195.0
            and z_release_end >= 0.925
            and post_drift_mm <= 3.0
        )
    else:
        z_release_start = float(target_pos_hist[245, 2])
        z_release_end = float(target_pos_hist[-1, 2])
        post_drift_mm = abs(z_release_end - z_release_start) * 1000.0
        task_passed = bool(
            (not any_kinematic_manipulated)
            and max_screw_deg >= 195.0
            and z_release_end >= 1.010
            and post_drift_mm <= 3.5
        )

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
        screw_deg=screw_deg_hist,
        coupled_pos=coupled_pos_hist,
        target_pos=target_pos_hist,
        pad_force_L_n=pad_force_L_hist,
        pad_force_R_n=pad_force_R_hist,
        pad_gap_mm=pad_gap_mm_hist,
        live_fingers_m=live_finger_hist,
        qL_cmd=qL_seq,
        qR_cmd=qR_seq,
    )

    metrics = {
        "mech_id": mech_id,
        "zero_friction": zero_friction,
        "any_kinematic_manipulated": any_kinematic_manipulated,
        "physx_hz": 240,
        "cumulative_screw_twist_deg": round(max_screw_deg, 2),
        "peak_finger_normal_force_n": round(peak_force_n, 2),
        "median_active_pad_surface_gap_mm": round(median_pad_gap_mm, 3),
        "target_final_z_m": round(float(target_pos_hist[-1, 2]), 4),
        "target_net_dz_mm": round(dz_final_mm, 2),
        "post_release_2s_drift_mm": round(post_drift_mm, 3),
        "task_passed": task_passed,
    }
    result_payload = {
        "episode_id": str(job["episode_id"]),
        "mech_id": mech_id,
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
        rp_out.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    sys.stdout.flush()
    app.close()
    return result_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()

    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    job["result_json_path"] = args.result_json.resolve().as_posix()
    res = execute_v07_job(job)
    if not args.result_json.is_file():
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
