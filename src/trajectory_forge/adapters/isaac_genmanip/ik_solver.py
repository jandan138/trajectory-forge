"""Pluggable arm kinematics / motion-planning backends (ADR-20261126).

Strictly isolated inside `trajectory_forge.adapters.isaac_genmanip` so pure layers
(`core`, `embodiment`, `sdk`, `solver`, `trajectory`) NEVER import `pinocchio`,
`pink`, `qpsolvers`, or `curobo`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from trajectory_forge.sdk.math3d import Pose3D


@dataclass(frozen=True)
class IKResult:
    q_target: tuple[float, ...]
    converged: bool
    pos_err_m: float
    rot_err_rad: float
    iterations: int
    failure_reason: str | None = None
    nearest_joint_limit: str | None = None


@dataclass(frozen=True)
class TrajectoryResult:
    waypoints_q: tuple[tuple[float, ...], ...]
    converged: bool
    final_pos_err_m: float
    final_rot_err_rad: float
    failure_reason: str | None = None


class KinematicsSolver(Protocol):
    name: Literal["pink_qp", "pyroki", "curobo"]

    def solve_ik(
        self,
        chain: str,
        current_q: np.ndarray,
        target_pose: Pose3D,
        *,
        orientation_weight: float = 1.0,
        position_weight: float = 1.0,
        nullspace_q: np.ndarray | None = None,
        check_collision: bool = False,
        allow_branch_flip: bool = True,
    ) -> IKResult: ...

    def plan_trajectory(
        self,
        chain: str,
        current_q: np.ndarray,
        target_pose: Pose3D,
        *,
        obstacles_enabled: bool = True,
        max_step_rad: float = 0.05,
    ) -> TrajectoryResult: ...


def _quat_wxyz_to_rotmat(wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = wxyz
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=float)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rotmat_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


def _rotation_error_rad(R_cur: np.ndarray, R_tgt: np.ndarray) -> float:
    R_err = R_cur.T @ R_tgt
    cos_theta = np.clip(0.5 * (float(np.trace(R_err)) - 1.0), -1.0, 1.0)
    return float(np.arccos(cos_theta))


def _choose_qp_solver() -> str:
    import qpsolvers  # type: ignore

    avail = set(qpsolvers.available_solvers)
    for pref in ("daqp", "proxqp", "osqp", "quadprog", "ecos", "scs"):
        if pref in avail:
            return pref
    if avail:
        return sorted(avail)[0]
    raise RuntimeError("No QP solver available for qpsolvers (install osqp or daqp).")


class PinkQPIKSolver:
    """Zero-config URDF QP differential IK solver built on Pinocchio + Pink."""

    name: Literal["pink_qp", "pyroki", "curobo"] = "pink_qp"

    def __init__(
        self,
        urdf_path: str | Path,
        chain_joints: dict[str, list[str]],
        chain_frames: dict[str, str],
        *,
        base_pose_world: Pose3D | None = None,
        qp_solver: str | None = None,
        max_iters: int = 80,
        dt: float = 0.05,
        pos_tol_m: float = 2.5e-3,
        rot_tol_rad: float = 3.5e-2,
    ) -> None:
        import pinocchio as pin  # type: ignore

        self._pin = pin
        self._urdf_path = Path(urdf_path).resolve()
        if not self._urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found for PinkQPIKSolver: {self._urdf_path}")

        self._full_model = pin.buildModelFromUrdf(str(self._urdf_path))
        self._chain_joints = {k: list(v) for k, v in chain_joints.items()}
        self._chain_frames = dict(chain_frames)
        self._qp_solver = qp_solver or _choose_qp_solver()
        self._max_iters = int(max_iters)
        self._dt = float(dt)
        self._pos_tol_m = float(pos_tol_m)
        self._rot_tol_rad = float(rot_tol_rad)

        if base_pose_world is None:
            self._base_se3 = pin.SE3.Identity()
        else:
            R_b = _quat_wxyz_to_rotmat(base_pose_world.wxyz)
            t_b = np.asarray(base_pose_world.xyz, dtype=float)
            self._base_se3 = pin.SE3(R_b, t_b)
        self._base_se3_inv = self._base_se3.inverse()

        self._reduced_models: dict[str, Any] = {}
        self._chain_q_order: dict[str, list[str]] = {}
        q_neutral_full = pin.neutral(self._full_model)

        for chain_name, active_joint_names in self._chain_joints.items():
            active_set = set(active_joint_names)
            joints_to_lock: list[int] = []
            for jid in range(1, self._full_model.njoints):
                jname = self._full_model.names[jid]
                if jname not in active_set:
                    joints_to_lock.append(jid)
            reduced = pin.buildReducedModel(self._full_model, joints_to_lock, q_neutral_full)
            if not reduced.existFrame(self._chain_frames[chain_name]):
                raise ValueError(
                    f"Frame {self._chain_frames[chain_name]!r} not found in reduced URDF model for chain {chain_name!r}"
                )
            self._reduced_models[chain_name] = reduced
            self._chain_q_order[chain_name] = [
                reduced.names[jid] for jid in range(1, reduced.njoints)
            ]

    def set_base_pose_world(self, base_pose_world: Pose3D) -> None:
        R_b = _quat_wxyz_to_rotmat(base_pose_world.wxyz)
        t_b = np.asarray(base_pose_world.xyz, dtype=float)
        self._base_se3 = self._pin.SE3(R_b, t_b)
        self._base_se3_inv = self._base_se3.inverse()

    def _world_to_base_se3(self, target_pose: Pose3D) -> Any:
        R_w = _quat_wxyz_to_rotmat(target_pose.wxyz)
        t_w = np.asarray(target_pose.xyz, dtype=float)
        return self._base_se3_inv * self._pin.SE3(R_w, t_w)

    def forward_kinematics(self, chain: str, q_chain: np.ndarray) -> Pose3D:
        model = self._reduced_models[chain]
        data = model.createData()
        q_in = np.asarray(q_chain, dtype=float).reshape(-1)
        self._pin.forwardKinematics(model, data, q_in)
        self._pin.updateFramePlacements(model, data)
        fid = model.getFrameId(self._chain_frames[chain])
        world_se3 = self._base_se3 * data.oMf[fid]
        return Pose3D(
            xyz=(
                float(world_se3.translation[0]),
                float(world_se3.translation[1]),
                float(world_se3.translation[2]),
            ),
            wxyz=_rotmat_to_quat_wxyz(world_se3.rotation),
        )

    def _solve_ik_single_seed(
        self,
        chain: str,
        seed_q: np.ndarray,
        target_pose: Pose3D,
        *,
        orientation_weight: float = 1.0,
        position_weight: float = 1.0,
        nullspace_q: np.ndarray | None = None,
    ) -> IKResult:
        import pink  # type: ignore
        from pink.limits import ConfigurationLimit, VelocityLimit  # type: ignore
        from pink.tasks import FrameTask, PostureTask  # type: ignore

        model = self._reduced_models[chain]
        frame_name = self._chain_frames[chain]
        q0 = np.asarray(seed_q, dtype=float).reshape(-1).copy()
        eps = 1e-4
        q0 = np.clip(q0, model.lowerPositionLimit + eps, model.upperPositionLimit - eps)

        config = pink.Configuration(model, model.createData(), q0)
        target_se3 = self._world_to_base_se3(target_pose)

        frame_task = FrameTask(
            frame_name,
            position_cost=max(float(position_weight), 1e-4) * 2.0,
            orientation_cost=max(float(orientation_weight), 1e-4) * 0.8,
            lm_damping=1e-3,
        )
        frame_task.set_target(target_se3)

        posture_task = PostureTask(cost=1e-3)
        q_ref = (
            np.asarray(nullspace_q, dtype=float).reshape(-1)
            if nullspace_q is not None
            else q0
        )
        posture_task.set_target(q_ref)

        tasks = [frame_task, posture_task]
        limits = [ConfigurationLimit(model), VelocityLimit(model)]

        pos_err = float("inf")
        rot_err = float("inf")
        iters = 0
        failure_reason: str | None = None

        for it in range(1, self._max_iters + 1):
            iters = it
            cur_se3 = config.get_transform_frame_to_world(frame_name)
            pos_err = float(np.linalg.norm(cur_se3.translation - target_se3.translation))
            rot_err = _rotation_error_rad(cur_se3.rotation, target_se3.rotation)
            if pos_err <= self._pos_tol_m and rot_err <= self._rot_tol_rad:
                break
            try:
                vel = pink.solve_ik(
                    config,
                    tasks,
                    self._dt,
                    solver=self._qp_solver,
                    limits=limits,
                    damping=1e-4,
                    safety_break=False,
                )
            except Exception as exc:
                failure_reason = f"qp_solver_error:{type(exc).__name__}"
                break
            config.integrate_inplace(vel, self._dt)

        cur_se3 = config.get_transform_frame_to_world(frame_name)
        pos_err = float(np.linalg.norm(cur_se3.translation - target_se3.translation))
        rot_err = _rotation_error_rad(cur_se3.rotation, target_se3.rotation)
        converged = (
            failure_reason is None
            and pos_err <= self._pos_tol_m * 2.0
            and rot_err <= self._rot_tol_rad * 2.0
        )

        q_final = np.asarray(config.q, dtype=float)
        lo_margin = np.abs(q_final - model.lowerPositionLimit)
        hi_margin = np.abs(model.upperPositionLimit - q_final)
        min_margins = np.minimum(lo_margin, hi_margin)
        worst_idx = int(np.argmin(min_margins))
        nearest_j = self._chain_q_order[chain][worst_idx]
        nearest_margin = float(min_margins[worst_idx])

        if not converged and failure_reason is None:
            if nearest_margin < 0.02:
                failure_reason = f"joint_limit:{nearest_j}"
            elif pos_err > 0.05:
                failure_reason = "out_of_reach"
            else:
                failure_reason = "ik_not_converged"

        return IKResult(
            q_target=tuple(float(v) for v in q_final),
            converged=converged,
            pos_err_m=pos_err,
            rot_err_rad=rot_err,
            iterations=iters,
            failure_reason=failure_reason,
            nearest_joint_limit=f"{nearest_j} (margin={nearest_margin:.4f} rad)",
        )

    def solve_ik(
        self,
        chain: str,
        current_q: np.ndarray,
        target_pose: Pose3D,
        *,
        orientation_weight: float = 1.0,
        position_weight: float = 1.0,
        nullspace_q: np.ndarray | None = None,
        check_collision: bool = False,
        allow_branch_flip: bool = True,
    ) -> IKResult:
        _ = check_collision
        q0 = np.asarray(current_q, dtype=float).reshape(-1)
        best = self._solve_ik_single_seed(
            chain,
            q0,
            target_pose,
            orientation_weight=orientation_weight,
            position_weight=position_weight,
            nullspace_q=nullspace_q,
        )
        if best.converged or not allow_branch_flip:
            return best

        model = self._reduced_models[chain]
        q_mid = 0.5 * (model.lowerPositionLimit + model.upperPositionLimit)
        alt_seeds = [q_mid]
        if q0.size >= 7:
            q_flip1 = q0.copy()
            q_flip1[2] += 0.6
            q_flip1[4] = -q_flip1[4]
            q_flip1[6] = -q_flip1[6]
            q_flip2 = q_mid.copy()
            q_flip2[2] -= 0.6
            alt_seeds.extend([q_flip1, q_flip2])

        for seed in alt_seeds:
            cand = self._solve_ik_single_seed(
                chain,
                seed,
                target_pose,
                orientation_weight=orientation_weight,
                position_weight=position_weight,
                nullspace_q=q0,
            )
            if cand.converged:
                return cand
            if (cand.pos_err_m + 0.1 * cand.rot_err_rad) < (best.pos_err_m + 0.1 * best.rot_err_rad):
                best = cand
        return best

    def plan_trajectory(
        self,
        chain: str,
        current_q: np.ndarray,
        target_pose: Pose3D,
        *,
        obstacles_enabled: bool = True,
        max_step_rad: float = 0.05,
    ) -> TrajectoryResult:
        q0 = np.asarray(current_q, dtype=float).reshape(-1)
        res = self.solve_ik(chain, q0, target_pose, nullspace_q=q0)
        q1 = np.asarray(res.q_target, dtype=float)
        max_delta = float(np.max(np.abs(q1 - q0))) if q0.size else 0.0
        n_steps = max(2, int(np.ceil(max_delta / max(max_step_rad, 1e-3))))
        waypoints = tuple(
            tuple(float(v) for v in (q0 + (q1 - q0) * (k / n_steps)))
            for k in range(1, n_steps + 1)
        )
        return TrajectoryResult(
            waypoints_q=waypoints,
            converged=res.converged,
            final_pos_err_m=res.pos_err_m,
            final_rot_err_rad=res.rot_err_rad,
            failure_reason=res.failure_reason,
        )


class CuRoboIKSolver:
    """Optional CuRobo adapter wrapping existing GenManip-Sim CuRobo planners (e.g. Lift2)."""

    name: Literal["pink_qp", "pyroki", "curobo"] = "curobo"

    def __init__(self, planners_by_chain: dict[str, Any]) -> None:
        self._planners = dict(planners_by_chain)

    def solve_ik(
        self,
        chain: str,
        current_q: np.ndarray,
        target_pose: Pose3D,
        *,
        orientation_weight: float = 1.0,
        position_weight: float = 1.0,
        nullspace_q: np.ndarray | None = None,
        check_collision: bool = False,
        allow_branch_flip: bool = True,
    ) -> IKResult:
        _ = (orientation_weight, position_weight, nullspace_q, check_collision, allow_branch_flip)
        planner = self._planners.get(chain)
        if planner is None:
            return IKResult(
                q_target=tuple(float(v) for v in current_q),
                converged=False,
                pos_err_m=float("inf"),
                rot_err_rad=float("inf"),
                iterations=0,
                failure_reason=f"missing_curobo_planner:{chain}",
            )
        pos = np.asarray(target_pose.xyz, dtype=float)
        ori = np.asarray(target_pose.wxyz, dtype=float)
        q_sol = planner.ik_single(pos, ori, current_q)
        if q_sol is None:
            return IKResult(
                q_target=tuple(float(v) for v in current_q),
                converged=False,
                pos_err_m=float("inf"),
                rot_err_rad=float("inf"),
                iterations=1,
                failure_reason="curobo_ik_failed",
            )
        return IKResult(
            q_target=tuple(float(v) for v in np.asarray(q_sol, dtype=float).reshape(-1)),
            converged=True,
            pos_err_m=0.0,
            rot_err_rad=0.0,
            iterations=1,
        )

    def plan_trajectory(
        self,
        chain: str,
        current_q: np.ndarray,
        target_pose: Pose3D,
        *,
        obstacles_enabled: bool = True,
        max_step_rad: float = 0.05,
    ) -> TrajectoryResult:
        res = self.solve_ik(chain, current_q, target_pose, check_collision=obstacles_enabled)
        if not res.converged:
            return TrajectoryResult(
                waypoints_q=(),
                converged=False,
                final_pos_err_m=res.pos_err_m,
                final_rot_err_rad=res.rot_err_rad,
                failure_reason=res.failure_reason,
            )
        q0 = np.asarray(current_q, dtype=float).reshape(-1)
        q1 = np.asarray(res.q_target, dtype=float)
        max_delta = float(np.max(np.abs(q1 - q0))) if q0.size else 0.0
        n_steps = max(2, int(np.ceil(max_delta / max(max_step_rad, 1e-3))))
        waypoints = tuple(
            tuple(float(v) for v in (q0 + (q1 - q0) * (k / n_steps)))
            for k in range(1, n_steps + 1)
        )
        return TrajectoryResult(
            waypoints_q=waypoints,
            converged=True,
            final_pos_err_m=res.pos_err_m,
            final_rot_err_rad=res.rot_err_rad,
        )
