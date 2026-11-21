"""Pure-NumPy forward and inverse kinematics matching panda.urdf for Franka Panda."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


PANDA_ARM_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
PANDA_ARM_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
PANDA_HOME_Q = np.array([0.0, -0.55, 0.0, -2.25, 0.0, 1.75, 0.785398163397], dtype=np.float64)

DEFAULT_PANDA_BASE_POS = np.array([0.0, -0.48, 0.755], dtype=np.float64)
DEFAULT_PANDA_BASE_QUAT_WXYZ = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=np.float64)

# Exact URDF joint chain: (xyz, rpy, is_revolute_z)
_PANDA_CHAIN: tuple[tuple[tuple[float, float, float], tuple[float, float, float], bool], ...] = (
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0), True),                     # joint1
    ((0.0, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0), True),               # joint2
    ((0.0, -0.316, 0.0), (math.pi / 2, 0.0, 0.0), True),             # joint3
    ((0.0825, 0.0, 0.0), (math.pi / 2, 0.0, 0.0), True),             # joint4
    ((-0.0825, 0.384, 0.0), (-math.pi / 2, 0.0, 0.0), True),         # joint5
    ((0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0), True),                # joint6
    ((0.088, 0.0, 0.0), (math.pi / 2, 0.0, 0.0), True),              # joint7
    ((0.0, 0.0, 0.107), (0.0, 0.0, -math.pi / 4), False),            # joint8 + hand_joint
    ((0.0, 0.0, 0.105), (0.0, 0.0, 0.0), False),                     # panda_grasptarget_hand
)


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def quat_wxyz_to_matrix(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64)
    tr = float(np.trace(r))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) or 1.0)


def _se3(rot: np.ndarray, pos: Sequence[float]) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rot
    t[:3, 3] = np.asarray(pos, dtype=np.float64)
    return t


_PRECOMPUTED_FIXED = [_se3(_rpy_matrix(*rpy), xyz) for xyz, rpy, _ in _PANDA_CHAIN]


def forward_kinematics_panda(
    q7: Sequence[float],
    *,
    base_pos: Sequence[float] = DEFAULT_PANDA_BASE_POS,
    base_quat_wxyz: Sequence[float] = DEFAULT_PANDA_BASE_QUAT_WXYZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute world-frame (pos3, rot3x3, quat_wxyz) of panda_grasptarget."""
    t = _se3(quat_wxyz_to_matrix(base_quat_wxyz), base_pos)
    q_arr = np.asarray(q7, dtype=np.float64)
    j_idx = 0
    for tf_fixed, (_, _, is_rev) in zip(_PRECOMPUTED_FIXED, _PANDA_CHAIN):
        t = t @ tf_fixed
        if is_rev:
            t = t @ _se3(_rot_z(float(q_arr[j_idx])), (0.0, 0.0, 0.0))
            j_idx += 1
    pos = t[:3, 3].copy()
    rot = t[:3, :3].copy()
    return pos, rot, matrix_to_quat_wxyz(rot)


def gripper_rot_for_grasp(yaw_rad: float = 0.0) -> np.ndarray:
    """Top-down grasp frame: tool +Z points straight down (-Z world), fingers open along +/-Y when yaw=0."""
    # At yaw_rad=0: tool X -> +X_world, tool Y -> -Y_world, tool Z -> -Z_world
    base_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    return _rot_z(yaw_rad) @ base_down


def solve_ik_panda(
    target_pos: Sequence[float],
    target_rot: np.ndarray | None = None,
    *,
    q_init: Sequence[float] | None = None,
    base_pos: Sequence[float] = DEFAULT_PANDA_BASE_POS,
    base_quat_wxyz: Sequence[float] = DEFAULT_PANDA_BASE_QUAT_WXYZ,
    max_iters: int = 120,
    pos_tol: float = 2e-4,
    rot_tol: float = 2e-3,
    damping: float = 0.04,
) -> np.ndarray:
    """Solve 7-DOF Franka Panda IK for panda_grasptarget using damped least-squares."""
    tgt_p = np.asarray(target_pos, dtype=np.float64)
    tgt_r = gripper_rot_for_grasp(0.0) if target_rot is None else np.asarray(target_rot, dtype=np.float64)
    q = PANDA_HOME_Q.copy() if q_init is None else np.asarray(q_init, dtype=np.float64).copy()
    q = np.clip(q, PANDA_ARM_LOWER, PANDA_ARM_UPPER)

    eps = 1e-4
    for _ in range(max_iters):
        cur_p, cur_r, _ = forward_kinematics_panda(q, base_pos=base_pos, base_quat_wxyz=base_quat_wxyz)
        err_p = tgt_p - cur_p
        r_err_mat = tgt_r @ cur_r.T
        err_w = 0.5 * np.array(
            [
                r_err_mat[2, 1] - r_err_mat[1, 2],
                r_err_mat[0, 2] - r_err_mat[2, 0],
                r_err_mat[1, 0] - r_err_mat[0, 1],
            ],
            dtype=np.float64,
        )
        if np.linalg.norm(err_p) < pos_tol and np.linalg.norm(err_w) < rot_tol:
            break

        err = np.concatenate([err_p, err_w])
        jac = np.zeros((6, 7), dtype=np.float64)
        for j in range(7):
            qp = q.copy()
            qp[j] += eps
            pp, rp, _ = forward_kinematics_panda(qp, base_pos=base_pos, base_quat_wxyz=base_quat_wxyz)
            jac[:3, j] = (pp - cur_p) / eps
            dr = rp @ cur_r.T
            jac[3:, j] = 0.5 * np.array([dr[2, 1] - dr[1, 2], dr[0, 2] - dr[2, 0], dr[1, 0] - dr[0, 1]]) / eps

        jjt = jac @ jac.T + (damping**2) * np.eye(6, dtype=np.float64)
        j_pinv = jac.T @ np.linalg.inv(jjt)
        null_proj = np.eye(7, dtype=np.float64) - j_pinv @ jac
        dq = j_pinv @ err + 0.05 * (null_proj @ (PANDA_HOME_Q - q))
        dq = np.clip(dq, -0.25, 0.25)
        q = np.clip(q + dq, PANDA_ARM_LOWER, PANDA_ARM_UPPER)
    return q


def panda_pad_world_poses(
    q9: Sequence[float],
    *,
    base_pos: Sequence[float] = DEFAULT_PANDA_BASE_POS,
    base_quat_wxyz: Sequence[float] = DEFAULT_PANDA_BASE_QUAT_WXYZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute world (ee_pos, ee_quat_wxyz, left_pad_pos, right_pad_pos, hand_rot) from 9-DOF Panda q9."""
    q_arr = np.asarray(q9, dtype=np.float64)
    ee_pos, hand_rot, ee_quat = forward_kinematics_panda(q_arr[:7], base_pos=base_pos, base_quat_wxyz=base_quat_wxyz)
    f_l = float(q_arr[7])
    f_r = float(q_arr[8])
    left_pad = ee_pos + hand_rot @ np.array([0.0, 0.006 + f_l, -0.0106], dtype=np.float64)
    right_pad = ee_pos + hand_rot @ np.array([0.0, -(0.006 + f_r), -0.0106], dtype=np.float64)
    return ee_pos, ee_quat, left_pad, right_pad, hand_rot
