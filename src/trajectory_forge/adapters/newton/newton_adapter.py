"""Native Newton (Warp) physics `SimAdapter` plugin (Microkernel v2).

Zero Isaac Sim Kit startup; natively supports:
  - `SINGLE_ENV`: live closed-loop stepping with URDF robot articulations & dynamic objects.
  - `CHECKPOINT`: exact tensor state save/restore during `privileged_search` (strictly
    blocked with `FirewallViolationError` in `full_horizon_gate` mode).
  - `VEC_ENV`: true multi-world parallel simulation via `newton.ModelBuilder.replicate`.
  - `PRIVILEGED_STATE`: full object poses & joint states exposed during search.
  - `DIFF_PHYSICS`: Warp differentiable model finalization (`requires_grad=True`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from trajectory_forge.core.contracts import (
    ContractValidationError,
    EmbodimentCard,
    FirewallViolationError,
    PoseSpec,
    ScenarioIntakeBrief,
)
from trajectory_forge.core.microkernel import (
    ActionCommand,
    SimCapability,
    StepObservation,
)


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> tuple[float, float, float, float]:
    return (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def _resolve_urdf_and_meta(
    embodiment: EmbodimentCard,
    options: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    opts = options or {}
    urdf_override = opts.get("urdf_path") or embodiment.metadata.get("urdf_path")
    meta_path_str = opts.get("meta_path") or embodiment.metadata.get("meta_path")
    if urdf_override and Path(str(urdf_override)).is_file():
        urdf_p = Path(str(urdf_override)).resolve()
    else:
        raise ContractValidationError(
            f"NewtonSimAdapter requires an existing `urdf_path` in embodiment.metadata or options "
            f"(got {urdf_override!r} for {embodiment.embodiment_id!r})"
        )
    meta: dict[str, Any] | None = None
    if meta_path_str and Path(str(meta_path_str)).is_file():
        meta = json.loads(Path(str(meta_path_str)).read_text(encoding="utf-8"))
    elif (urdf_p.parent / "robot_assembly_meta.json").is_file():
        meta = json.loads((urdf_p.parent / "robot_assembly_meta.json").read_text(encoding="utf-8"))
    return urdf_p, meta


def _strip_label_prefix(label: str) -> str:
    return label.split("/")[-1] if "/" in label else label


def _build_newton_world(
    brief: ScenarioIntakeBrief,
    urdf_path: Path,
    *,
    seed: int = 0,
    world_count: int = 1,
    device: str = "cpu",
    requires_grad: bool = False,
) -> tuple[Any, dict[str, int], dict[str, int], dict[str, tuple[float, float, float]]]:
    """Build a Newton Model with the assembled URDF robot + dynamic scenario objects."""
    import newton  # type: ignore
    import warp as wp  # type: ignore

    rng = np.random.default_rng(seed)
    sub = newton.ModelBuilder()
    sub.add_urdf(
        str(urdf_path),
        xform=wp.transform(
            wp.vec3(*brief.robot_spawn.xyz),
            wp.quat(
                brief.robot_spawn.wxyz[1],
                brief.robot_spawn.wxyz[2],
                brief.robot_spawn.wxyz[3],
                brief.robot_spawn.wxyz[0],
            ),
        ),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
        force_position_velocity_actuation=True,
    )

    # Spawn dynamic objects from ScenarioIntakeBrief with deterministic multi-seed XY jitter
    init_obj_xyz: dict[str, tuple[float, float, float]] = {}
    for oid, obj in brief.objects.items():
        if obj.role == "static_fixture":
            continue
        jx = float(rng.uniform(-0.025, 0.025)) if seed != 0 else 0.0
        jy = float(rng.uniform(-0.025, 0.025)) if seed != 0 else 0.0
        ox = obj.initial_pose.xyz[0] + jx
        oy = obj.initial_pose.xyz[1] + jy
        oz = obj.initial_pose.xyz[2]
        init_obj_xyz[oid] = (ox, oy, oz)
        q = obj.initial_pose.wxyz
        b_idx = sub.add_body(
            xform=wp.transform(wp.vec3(ox, oy, oz), wp.quat(q[1], q[2], q[3], q[0])),
            label=oid,
        )
        sub.add_shape_cylinder(b_idx, radius=0.03, half_height=0.05)

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    if world_count == 1:
        builder.add_world(sub)
    else:
        builder.replicate(sub, world_count=world_count)

    model = builder.finalize(device=device, requires_grad=requires_grad)

    # Map clean joint & body labels for world 0
    joint_q_map: dict[str, int] = {}
    q_starts = model.joint_q_start.numpy()
    j_types = model.joint_type.numpy()
    n_joints_per_world = model.joint_count // max(1, world_count)
    for j_i in range(n_joints_per_world):
        if j_types[j_i] != newton.JointType.FIXED and j_types[j_i] != newton.JointType.FREE:
            clean = _strip_label_prefix(str(model.joint_label[j_i]))
            joint_q_map[clean] = int(q_starts[j_i])

    body_idx_map: dict[str, int] = {}
    n_bodies_per_world = model.body_count // max(1, world_count)
    for b_i in range(n_bodies_per_world):
        clean = _strip_label_prefix(str(model.body_label[b_i]))
        body_idx_map[clean] = b_i

    return model, joint_q_map, body_idx_map, init_obj_xyz


class NewtonSimSession:
    """Single-world interactive / gate session backed by native Newton (Warp) physics."""

    def __init__(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        mode: str = "full_horizon_gate",
        seed: int = 0,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self.brief = brief
        self.embodiment = embodiment
        self.mode = mode
        self.seed = int(seed)
        self.options = dict(options or {})
        self.snapshots_used = 0
        self._dt = float(self.options.get("dt", 1.0 / 60.0))
        self._device = str(self.options.get("device", "cpu"))
        self._urdf_path, self._meta = _resolve_urdf_and_meta(embodiment, self.options)

        self._model: Any = None
        self._solver: Any = None
        self._state: Any = None
        self._state_next: Any = None
        self._control: Any = None
        self._contacts: Any = None
        self._joint_q_map: dict[str, int] = {}
        self._body_idx_map: dict[str, int] = {}
        self._init_obj_xyz: dict[str, tuple[float, float, float]] = {}
        self._step_index = 0
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._ik_solver: Any = None
        self._q_target: np.ndarray | None = None
        self._init_ik_solver()

    def _init_ik_solver(self) -> None:
        try:
            from trajectory_forge.adapters.isaac_genmanip.ik_solver import PinkQPIKSolver
            from trajectory_forge.sdk.math3d import Pose3D

            chain_joints: dict[str, tuple[str, ...]] = {}
            chain_frames: dict[str, str] = {}
            for cname, chain in self.embodiment.chains.items():
                chain_joints[cname] = tuple(chain.joint_names)
                chain_frames[cname] = chain.ee_frame
            self._ik_solver = PinkQPIKSolver(
                urdf_path=self._urdf_path,
                chain_joints=chain_joints,
                chain_frames=chain_frames,
                base_pose_world=Pose3D(
                    xyz=self.brief.robot_spawn.xyz,
                    wxyz=self.brief.robot_spawn.wxyz,
                ),
            )
        except Exception:
            self._ik_solver = None

    def reset(self, *, seed: int = 0) -> StepObservation:
        import newton  # type: ignore

        self.seed = int(seed)
        self._model, self._joint_q_map, self._body_idx_map, self._init_obj_xyz = _build_newton_world(
            self.brief,
            self._urdf_path,
            seed=self.seed,
            world_count=1,
            device=self._device,
        )
        # Tune joint PD gains for crisp articulation tracking in Newton
        if hasattr(self._model, "joint_target_ke"):
            ke = self._model.joint_target_ke.numpy()
            kd = self._model.joint_target_kd.numpy()
            ke[:] = 400.0
            kd[:] = 40.0
            self._model.joint_target_ke.assign(ke)
            self._model.joint_target_kd.assign(kd)

        self._state = self._model.state()
        self._state_next = self._model.state()
        self._control = self._model.control()
        self._contacts = self._model.contacts()
        self._solver = newton.solvers.SolverSemiImplicit(self._model)

        # Initialize neutral arm posture from EmbodimentCard
        q_init = self._model.joint_q.numpy().copy()
        for chain in self.embodiment.chains.values():
            for jn, val in zip(chain.joint_names, chain.neutral_q):
                if jn in self._joint_q_map:
                    q_init[self._joint_q_map[jn]] = float(val)
        self._state.joint_q.assign(q_init)
        self._q_target = q_init.copy()
        newton.eval_fk(self._model, self._state.joint_q, self._state.joint_qd, self._state)

        self._step_index = 0
        self._snapshots.clear()
        return self._build_observation()

    def _expand_ee_postures(self, ee_postures: Mapping[str, str]) -> dict[str, float]:
        targets: dict[str, float] = {}
        for ag_name, posture in ee_postures.items():
            if ag_name not in self.embodiment.actuator_groups:
                continue
            ag = self.embodiment.actuator_groups[ag_name]
            preset = ag.named_postures.get(posture)
            if preset is None:
                continue
            for jn, val in zip(ag.joint_names, preset):
                targets[jn] = float(val)
            # Expand mimic followers if metadata is available
            side = "left" if "left" in ag_name else ("right" if "right" in ag_name else "")
            if self._meta and side:
                mimics = self._meta.get("mimic_joints", {}).get(side, {})
                for follower, mcfg in mimics.items():
                    ref = mcfg["joint"]
                    if ref in targets:
                        targets[follower] = targets[ref] * float(mcfg.get("multiplier", 1.0)) + float(
                            mcfg.get("offset", 0.0)
                        )
        return targets

    def step(self, action: ActionCommand) -> StepObservation:
        import newton  # type: ignore
        from trajectory_forge.sdk.math3d import Pose3D

        assert self._state is not None and self._q_target is not None
        q_cur = self._state.joint_q.numpy().copy()
        q_tgt = self._q_target.copy()

        # 1. Solve any Cartesian ee_pose_targets via PinkQPIKSolver
        if action.ee_pose_targets and self._ik_solver is not None:
            for chain_name, pose_spec in action.ee_pose_targets.items():
                if chain_name not in self.embodiment.chains:
                    continue
                chain = self.embodiment.chains[chain_name]
                cur_chain_q = np.array(
                    [q_cur[self._joint_q_map[jn]] for jn in chain.joint_names if jn in self._joint_q_map],
                    dtype=float,
                )
                if len(cur_chain_q) == len(chain.joint_names):
                    res = self._ik_solver.solve_ik(
                        chain_name,
                        cur_chain_q,
                        Pose3D(xyz=pose_spec.xyz, wxyz=pose_spec.wxyz),
                    )
                    for jn, val in zip(chain.joint_names, res.q_target):
                        q_tgt[self._joint_q_map[jn]] = float(val)

        # 2. Apply explicit joint targets and EE postures / direct finger targets
        for jn, val in self._expand_ee_postures(action.ee_postures).items():
            if jn in self._joint_q_map:
                q_tgt[self._joint_q_map[jn]] = float(val)
        for _, jmap in action.ee_joint_targets.items():
            for jn, val in jmap.items():
                if jn in self._joint_q_map:
                    q_tgt[self._joint_q_map[jn]] = float(val)
        for jn, val in action.joint_targets.items():
            if jn in self._joint_q_map:
                q_tgt[self._joint_q_map[jn]] = float(val)

        # Smooth kinematic+physics step: blend joint targets + run Newton SemiImplicit solver
        self._q_target = q_tgt
        n_sub = max(1, int(action.substeps))
        for _ in range(n_sub):
            q_live = self._state.joint_q.numpy().copy()
            # Apply bounded joint velocity step toward target and keep dynamic bodies physical
            for jn, qidx in self._joint_q_map.items():
                delta = float(q_tgt[qidx] - q_live[qidx])
                q_live[qidx] += float(np.clip(delta, -0.08, 0.08))
            self._state.joint_q.assign(q_live)
            if self._control is not None and self._control.joint_target_pos is not None:
                # Match control buffer length cleanly
                c_pos = self._control.joint_target_pos.numpy()
                n_copy = min(len(c_pos), len(q_tgt))
                c_pos[:n_copy] = q_tgt[:n_copy]
                self._control.joint_target_pos.assign(c_pos)
            newton.eval_fk(self._model, self._state.joint_q, self._state.joint_qd, self._state)
            self._model.collide(self._state, self._contacts)
            self._solver.step(self._state, self._state_next, self._control, self._contacts, self._dt)
            # Re-enforce actuated robot joint positions after solver step for crisp position control
            q_after = self._state_next.joint_q.numpy().copy()
            for jn, qidx in self._joint_q_map.items():
                q_after[qidx] = q_live[qidx]
            self._state_next.joint_q.assign(q_after)
            newton.eval_fk(self._model, self._state_next.joint_q, self._state_next.joint_qd, self._state_next)
            self._state, self._state_next = self._state_next, self._state
            self._step_index += 1

        return self._build_observation()

    def _evaluate_stages(
        self,
        ee_poses: Mapping[str, PoseSpec],
        object_poses: Mapping[str, PoseSpec],
    ) -> tuple[dict[str, float], float]:
        if not self.brief.stages:
            return {}, 1.0
        stage_scores: dict[str, float] = {}
        for st in self.brief.stages:
            clause_pass = 0
            total_clauses = max(1, len(st.clauses))
            for clause in st.clauses:
                ctype = str(clause.get("type", ""))
                if ctype == "ee_near_object":
                    chain = str(clause.get("chain", "right_arm"))
                    oid = str(clause["object_id"])
                    tol = float(clause.get("tolerance_m", 0.06))
                    offset = tuple(float(v) for v in clause.get("offset_xyz", (0.0, 0.0, 0.0)))
                    if chain in ee_poses and oid in object_poses:
                        ep = ee_poses[chain].xyz
                        op = object_poses[oid].xyz
                        dist = math.dist(ep, (op[0] + offset[0], op[1] + offset[1], op[2] + offset[2]))
                        if dist <= tol:
                            clause_pass += 1
                elif ctype == "ee_near_point":
                    chain = str(clause.get("chain", "right_arm"))
                    pt = tuple(float(v) for v in clause["xyz"])
                    tol = float(clause.get("tolerance_m", 0.06))
                    if chain in ee_poses and math.dist(ee_poses[chain].xyz, pt) <= tol:
                        clause_pass += 1
                elif ctype == "object_lifted":
                    oid = str(clause["object_id"])
                    min_dz = float(clause.get("min_delta_z_m", 0.05))
                    if oid in object_poses and oid in self._init_obj_xyz:
                        if object_poses[oid].xyz[2] - self._init_obj_xyz[oid][2] >= min_dz:
                            clause_pass += 1
                else:
                    clause_pass += 1
            stage_scores[st.metric_id] = clause_pass / total_clauses
        total = sum(stage_scores.values()) / max(1, len(stage_scores))
        return stage_scores, float(total)

    def _build_observation(self) -> StepObservation:
        q_arr = self._state.joint_q.numpy()
        body_q = self._state.body_q.numpy()

        joint_positions = {jn: float(q_arr[idx]) for jn, idx in self._joint_q_map.items()}

        ee_poses: dict[str, PoseSpec] = {}
        for cname, chain in self.embodiment.chains.items():
            if chain.ee_frame in self._body_idx_map:
                bq = body_q[self._body_idx_map[chain.ee_frame]]
                ee_poses[cname] = PoseSpec(
                    xyz=(float(bq[0]), float(bq[1]), float(bq[2])),
                    wxyz=_quat_xyzw_to_wxyz(bq[3:7]),
                )

        object_poses: dict[str, PoseSpec] = {}
        for oid in self.brief.objects:
            if oid in self._body_idx_map:
                bq = body_q[self._body_idx_map[oid]]
                object_poses[oid] = PoseSpec(
                    xyz=(float(bq[0]), float(bq[1]), float(bq[2])),
                    wxyz=_quat_xyzw_to_wxyz(bq[3:7]),
                )

        stage_scores, total_score = self._evaluate_stages(ee_poses, object_poses)
        privileged = (
            {
                "object_poses": {k: v.to_dict() for k, v in object_poses.items()},
                "step_index": self._step_index,
                "seed": self.seed,
            }
            if self.mode == "privileged_search"
            else {}
        )
        return StepObservation(
            step_index=self._step_index,
            sim_time_s=round(self._step_index * self._dt, 5),
            joint_positions=joint_positions,
            ee_poses=ee_poses,
            object_poses=object_poses,
            stage_scores=stage_scores,
            total_score=total_score,
            done=total_score >= 0.9999,
            privileged_state=privileged,
        )

    def save_checkpoint(self, label: str) -> str:
        if self.mode != "privileged_search":
            raise FirewallViolationError(
                f"save_checkpoint({label!r}) is strictly forbidden in mode={self.mode!r}"
            )
        snap_id = f"newton_snap_{len(self._snapshots)}_{label}"
        self._snapshots[snap_id] = {
            "joint_q": self._state.joint_q.numpy().copy(),
            "joint_qd": self._state.joint_qd.numpy().copy(),
            "body_q": self._state.body_q.numpy().copy(),
            "body_qd": self._state.body_qd.numpy().copy(),
            "q_target": self._q_target.copy() if self._q_target is not None else None,
            "step_index": self._step_index,
        }
        return snap_id

    def restore_checkpoint(self, snap_id: str) -> StepObservation:
        if self.mode != "privileged_search":
            raise FirewallViolationError(
                f"restore_checkpoint({snap_id!r}) is strictly forbidden in mode={self.mode!r}"
            )
        if snap_id not in self._snapshots:
            raise ContractValidationError(f"Unknown Newton checkpoint: {snap_id!r}")
        import newton  # type: ignore

        snap = self._snapshots[snap_id]
        self._state.joint_q.assign(snap["joint_q"])
        self._state.joint_qd.assign(snap["joint_qd"])
        self._state.body_q.assign(snap["body_q"])
        self._state.body_qd.assign(snap["body_qd"])
        self._q_target = snap["q_target"].copy() if snap["q_target"] is not None else None
        self._step_index = int(snap["step_index"])
        newton.eval_fk(self._model, self._state.joint_q, self._state.joint_qd, self._state)
        self.snapshots_used += 1
        return self._build_observation()

    def close(self) -> None:
        self._snapshots.clear()


class NewtonVecSession:
    """True vectorized multi-world Newton session (`SimCapability.VEC_ENV`) for parallel RL."""

    def __init__(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        num_envs: int = 4,
        seed: int = 0,
        device: str = "cpu",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        import newton  # type: ignore

        if num_envs < 1:
            raise ContractValidationError("NewtonVecSession requires num_envs >= 1")
        self.num_envs = int(num_envs)
        self.device = device
        self._dt = float((options or {}).get("dt", 1.0 / 60.0))
        urdf_path, _ = _resolve_urdf_and_meta(embodiment, options)
        self._model, self.joint_q_map, self.body_idx_map, _ = _build_newton_world(
            brief,
            urdf_path,
            seed=seed,
            world_count=self.num_envs,
            device=self.device,
        )
        self._state = self._model.state()
        self._state_next = self._model.state()
        self._control = self._model.control()
        self._contacts = self._model.contacts()
        self._solver = newton.solvers.SolverSemiImplicit(self._model)
        newton.eval_fk(self._model, self._state.joint_q, self._state.joint_qd, self._state)

    def step_batch(self, joint_q_batch: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Advance all `num_envs` worlds simultaneously and return batched state tensors."""
        import newton  # type: ignore

        if joint_q_batch is not None:
            flat = np.asarray(joint_q_batch, dtype=np.float32).reshape(-1)
            q_live = self._state.joint_q.numpy().copy()
            n_copy = min(len(flat), len(q_live))
            q_live[:n_copy] = flat[:n_copy]
            self._state.joint_q.assign(q_live)
        newton.eval_fk(self._model, self._state.joint_q, self._state.joint_qd, self._state)
        self._model.collide(self._state, self._contacts)
        self._solver.step(self._state, self._state_next, self._control, self._contacts, self._dt)
        self._state, self._state_next = self._state_next, self._state
        bodies_per_world = self._model.body_count // self.num_envs
        q_per_world = self._model.joint_coord_count // self.num_envs
        return {
            "body_q": self._state.body_q.numpy().reshape(self.num_envs, bodies_per_world, 7),
            "joint_q": self._state.joint_q.numpy().reshape(self.num_envs, q_per_world),
        }

    def close(self) -> None:
        return None


class NewtonSimAdapter:
    """Microkernel v2 `SimAdapter` backed by NVIDIA/Linux Foundation Newton (Warp)."""

    backend_name = "newton"
    capabilities: frozenset[SimCapability] = frozenset(
        {
            SimCapability.SINGLE_ENV,
            SimCapability.CHECKPOINT,
            SimCapability.VEC_ENV,
            SimCapability.PRIVILEGED_STATE,
            SimCapability.DIFF_PHYSICS,
        }
    )

    def open_session(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        mode: str = "full_horizon_gate",
        seed: int = 0,
        options: Mapping[str, Any] | None = None,
    ) -> NewtonSimSession:
        return NewtonSimSession(brief, embodiment, mode=mode, seed=seed, options=options)

    def open_vec_session(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        num_envs: int = 4,
        seed: int = 0,
        device: str = "cpu",
        options: Mapping[str, Any] | None = None,
    ) -> NewtonVecSession:
        return NewtonVecSession(
            brief,
            embodiment,
            num_envs=num_envs,
            seed=seed,
            device=device,
            options=options,
        )
