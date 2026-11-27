from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt
from scipy.spatial.transform import Rotation as R

from trajectory_forge.core.contracts import (
    ALLOWED_EXECUTION_MODES,
    ContractValidationError,
    EmbodimentCard,
    FirewallViolationError,
    MediaArtifactRef,
    RemediationPolicySpec,
    ScenarioIntakeBrief,
    SnapshotHandle,
    SolvabilityDiagnostics,
    StageProgressEntry,
)
from trajectory_forge.sdk.context import (
    AABB3D,
    JointAxisInfo,
    TactileReading,
    WrenchReading,
)
from trajectory_forge.sdk.math3d import (
    Pose3D,
    normalize_quat_wxyz,
    pose_from_named_frame,
    slerp_wxyz,
)
from trajectory_forge.solver.diagnostics import (
    apply_remediation_policy,
    compute_sequential_blocking_stage,
)
from trajectory_forge.adapters.isaac_genmanip.pbd_metrics import (
    evaluate_scene_pbd_containment,
    extract_pbd_goal_specs,
    register_pbd_containment_metric,
    sync_stage_pbd_isosurface_materials,
)
from trajectory_forge.trajectory.conveyor_math import (
    eval_racetrack_pose,
    racetrack_perimeter,
)
from trajectory_forge.trajectory.fluid_helpers import (
    compute_aabb_containment_ratio,
    compute_vessel_fill_bounds,
    scale_ticks_for_physics_dt,
    transform_points_local_to_world,
)


# R5a link6 -> fingertip TCP rotation: R_y(+90 deg) @ R_z(180 deg) (symmetric orthogonal)
_M_TCP_TO_LINK6 = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)
_LINK6_TO_TCP_OFFSET_M = 0.135


def _pose_to_mat(xyz: Sequence[float], wxyz: Sequence[float]) -> NDArray[Any]:
    mat = np.eye(4, dtype=float)
    q_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float)
    mat[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    mat[:3, 3] = np.asarray(xyz, dtype=float)
    return mat


def _mat_to_pose(mat: NDArray[Any]) -> Pose3D:
    q_xyzw = R.from_matrix(mat[:3, :3]).as_quat()
    wxyz = normalize_quat_wxyz((float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])))
    xyz = (float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]))
    return Pose3D(xyz=xyz, wxyz=wxyz)


def tcp_world_to_link6_world(tcp_pose: Pose3D) -> tuple[NDArray[Any], NDArray[Any]]:
    """Convert desired TCP pose (+Z approach, +Y jaw close) to R5a link6 world pose."""
    q_xyzw = np.array([tcp_pose.wxyz[1], tcp_pose.wxyz[2], tcp_pose.wxyz[3], tcp_pose.wxyz[0]], dtype=float)
    rot_tcp = R.from_quat(q_xyzw).as_matrix()
    approach_world = rot_tcp[:, 2]
    p_link6 = np.asarray(tcp_pose.xyz, dtype=float) - _LINK6_TO_TCP_OFFSET_M * approach_world
    rot_link6 = rot_tcp @ _M_TCP_TO_LINK6
    q_l6_xyzw = R.from_matrix(rot_link6).as_quat()
    q_l6_wxyz = np.array([q_l6_xyzw[3], q_l6_xyzw[0], q_l6_xyzw[1], q_l6_xyzw[2]], dtype=float)
    return p_link6, q_l6_wxyz


def link6_world_to_tcp_world(p_link6: NDArray[Any], q_link6_wxyz: NDArray[Any]) -> Pose3D:
    """Convert R5a link6 world pose to fingertip TCP Pose3D."""
    q_xyzw = np.array([q_link6_wxyz[1], q_link6_wxyz[2], q_link6_wxyz[3], q_link6_wxyz[0]], dtype=float)
    rot_link6 = R.from_quat(q_xyzw).as_matrix()
    rot_tcp = rot_link6 @ _M_TCP_TO_LINK6
    approach_world = rot_tcp[:, 2]
    p_tcp = np.asarray(p_link6, dtype=float) + _LINK6_TO_TCP_OFFSET_M * approach_world
    q_tcp_xyzw = R.from_matrix(rot_tcp).as_quat()
    return Pose3D(
        xyz=(float(p_tcp[0]), float(p_tcp[1]), float(p_tcp[2])),
        wxyz=normalize_quat_wxyz(
            (float(q_tcp_xyzw[3]), float(q_tcp_xyzw[0]), float(q_tcp_xyzw[1]), float(q_tcp_xyzw[2]))
        ),
    )


def fix_lift2_finger_collider_offsets(stage: Any, robot_prim_path: str) -> int:
    """Fix R5a finger colliders, dummy_base mass, and friction while preserving mimic-joint gripper physics."""
    updated = 0
    for prim in stage.Traverse():
        pstr = str(prim.GetPath())
        if not pstr.startswith(robot_prim_path):
            continue
        pname = prim.GetName()
        if pname in {"h_link5", "h_link6"} and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            updated += 1
            continue
        if pname in {"dummy_base_x", "dummy_base_y", "dummy_base_rotate"}:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(1.0)
            updated += 1
            continue
        if ("/link7/" in pstr or "/link8/" in pstr) and prim.HasAPI(UsdPhysics.CollisionAPI):
            c_attr = prim.GetAttribute("physxCollision:contactOffset")
            r_attr = prim.GetAttribute("physxCollision:restOffset")
            p_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            c_val = c_attr.Get() if c_attr and c_attr.HasValue() else None
            r_val = r_attr.Get() if r_attr and r_attr.HasValue() else None
            if c_val is None or r_val is None or float(c_val) <= float(r_val) or float(c_val) <= 0.0:
                p_api.CreateContactOffsetAttr().Set(0.004)
                p_api.CreateRestOffsetAttr().Set(0.0005)
                updated += 1
        elif pname == "PhysicsMaterial" or prim.HasAPI(UsdPhysics.MaterialAPI):
            mat_api = UsdPhysics.MaterialAPI.Apply(prim)
            s_attr = prim.GetAttribute("physics:staticFriction")
            d_attr = prim.GetAttribute("physics:dynamicFriction")
            s_val = float(s_attr.Get()) if s_attr and s_attr.HasValue() and s_attr.Get() is not None else 0.0
            d_val = float(d_attr.Get()) if d_attr and d_attr.HasValue() and d_attr.Get() is not None else 0.0
            if s_val < 0.8:
                mat_api.CreateStaticFrictionAttr().Set(0.8)
            if d_val < 0.8:
                mat_api.CreateDynamicFrictionAttr().Set(0.8)
            p_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
            p_mat_api.CreateImprovePatchFrictionAttr().Set(True)
    return updated


def reset_articulation_velocities_and_base(scene: Any) -> None:
    """Zero robot dummy-base DOFs and velocities on all articulations and rigid objects at reset."""
    from genmanip.utils.usd_utils.prim_utils import clean_prim_velocity

    if getattr(scene, "robot_list", None):
        emb = scene.robot_list[0]
        base_idx = list(getattr(emb, "base_dof_indices", ()) or ())
        if base_idx:
            zeros_base = np.zeros(len(base_idx), dtype=float)
            emb.robot.set_joint_positions(zeros_base, joint_indices=base_idx)
            emb.target_base_pose = zeros_base.copy()
        num_dofs = len(np.asarray(emb.robot.get_joint_positions(), dtype=float))
        emb.robot.set_joint_velocities(np.zeros(num_dofs, dtype=float))
    for art in (getattr(scene, "articulation_list", None) or {}).values():
        dof_names = getattr(art, "dof_names", None) or ()
        if len(dof_names) > 0 and hasattr(art, "set_joint_velocities"):
            art.set_joint_velocities(np.zeros(len(dof_names), dtype=float))
    for obj in (getattr(scene, "object_list", None) or {}).values():
        ppath = getattr(obj, "prim_path", None)
        if ppath:
            clean_prim_velocity(str(ppath))


def configure_articulation_drives_at_reset(scene: Any, initial_layout: Mapping[str, Any]) -> None:
    """Lock PD targets ONLY on joints with positive stiffness (e.g. spring buttons); leave passive joints free."""
    stage = scene.world.stage
    for uid, art in scene.articulation_list.items():
        if uid not in initial_layout:
            continue
        positions = np.asarray(art.get_joint_positions(), dtype=float)
        art_prim_path = str(getattr(art, "prim_path", ""))
        stiff_map: dict[str, float] = {}
        if art_prim_path:
            root_prim = stage.GetPrimAtPath(art_prim_path)
            if root_prim and root_prim.IsValid():
                epoch_attr = root_prim.GetAttribute("device:resetEpoch")
                if epoch_attr and epoch_attr.IsValid():
                    cur_epoch = epoch_attr.Get()
                    epoch_attr.Set(int(cur_epoch or 0) + 1)
                for child in Usd.PrimRange(root_prim):
                    if not (child.IsA(UsdPhysics.RevoluteJoint) or child.IsA(UsdPhysics.PrismaticJoint)):
                        c_epoch = child.GetAttribute("device:resetEpoch")
                        if c_epoch and c_epoch.IsValid():
                            cur_epoch = c_epoch.Get()
                            c_epoch.Set(int(cur_epoch or 0) + 1)
                        continue
                    jname = child.GetName()
                    stiff = 0.0
                    for axis_tok in ("angular", "linear"):
                        attr = child.GetAttribute(f"drive:{axis_tok}:physics:stiffness")
                        if attr and attr.HasValue() and attr.Get() is not None:
                            stiff = max(stiff, float(attr.Get()))
                    stiff_map[jname] = stiff

        dof_names = [str(n) for n in getattr(art, "dof_names", ())]
        targets: list[float | None] = []
        any_stiff = False
        for idx, pos in enumerate(positions):
            jname = dof_names[idx] if idx < len(dof_names) else ""
            if stiff_map.get(jname, 0.0) > 0.0:
                targets.append(float(pos))
                any_stiff = True
            else:
                targets.append(None)
        if any_stiff:
            art.apply_action(ArticulationAction(joint_positions=targets))


def _transform_local_pbd_pts(
    pts_local: Sequence[tuple[float, float, float]],
    pos: Sequence[float],
    wxyz: Sequence[float],
) -> list[tuple[float, float, float]]:
    rot = R.from_quat([float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0])])
    arr = rot.apply(np.asarray(pts_local, dtype=float)) + np.asarray(pos, dtype=float).reshape(1, 3)
    return [(float(r[0]), float(r[1]), float(r[2])) for r in arr]


def snapshot_and_reset_stage_pbd_particles(
    scene: Any,
    initial_layout: Mapping[str, Any] | None = None,
    *,
    stop_world: bool = True,
    use_live_host_pose: bool = False,
) -> int:
    """Snapshot t=0 GPU PBD particle points once per scene and restore under host pose on reset."""
    stage = scene.world.stage
    cache: dict[str, tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], str | None]] | None = getattr(
        scene, "_ce_op_t0_pbd_particles", None
    )
    if cache is None:
        cache = {}
        for prim in stage.Traverse():
            if not (prim.IsA(UsdGeom.Points) and prim.HasAPI(PhysxSchema.PhysxParticleSetAPI)):
                continue
            host_attr = prim.GetAttribute("fluid:hostVesselUid")
            host_uid = str(host_attr.Get()) if (host_attr and host_attr.HasValue() and host_attr.Get()) else None
            loc_attr = prim.GetAttribute("fluid:initialLocalPoints")
            if loc_attr and loc_attr.HasValue() and loc_attr.Get():
                pts_val = loc_attr.Get()
            else:
                pts_val = UsdGeom.Points(prim).GetPointsAttr().Get()
            if not pts_val:
                continue
            pts_list = [(float(p[0]), float(p[1]), float(p[2])) for p in pts_val]
            zeros_list = [(0.0, 0.0, 0.0) for _ in pts_list]
            cache[str(prim.GetPath())] = (pts_list, zeros_list, host_uid)
        setattr(scene, "_ce_op_t0_pbd_particles", cache)

    restored = 0
    layout = initial_layout or {}
    if stop_world and cache and layout and hasattr(scene.world, "is_playing") and scene.world.is_playing():
        scene.world.stop()
    for ppath, (pts_list, zeros_list, host_uid) in cache.items():
        prim = stage.GetPrimAtPath(ppath)
        if not prim or not prim.IsValid():
            continue
        pts_target = pts_list
        host_pos = None
        host_ori = None
        if host_uid:
            host_obj = getattr(scene, "object_list", {}).get(host_uid)
            if use_live_host_pose and host_obj is not None and hasattr(host_obj, "get_world_pose"):
                live_pos, live_ori = host_obj.get_world_pose()
                host_pos = [float(x) for x in live_pos]
                host_ori = [float(x) for x in live_ori]
            elif host_uid in layout and isinstance(layout[host_uid], Mapping):
                entry = layout[host_uid]
                if "position" in entry and "orientation" in entry:
                    host_pos = [float(x) for x in entry["position"]]
                    host_ori = [float(x) for x in entry["orientation"]]
                    if host_obj is not None:
                        if hasattr(host_obj, "set_default_state"):
                            host_obj.set_default_state(
                                position=np.asarray(host_pos, dtype=float),
                                orientation=np.asarray(host_ori, dtype=float),
                            )
                        if hasattr(host_obj, "set_world_pose"):
                            host_obj.set_world_pose(host_pos, host_ori)
        if host_pos is not None and host_ori is not None:
            pts_target = _transform_local_pbd_pts(pts_list, host_pos, host_ori)
        vec_pts = Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts_target])
        vec_zeros = Vt.Vec3fArray([Gf.Vec3f(*v) for v in zeros_list])
        pts_geom = UsdGeom.Points(prim)
        if pts_geom.GetPointsAttr():
            pts_geom.GetPointsAttr().Set(vec_pts)
        sim_pts_attr = prim.GetAttribute("physxParticle:simulationPoints")
        if sim_pts_attr and sim_pts_attr.IsValid():
            sim_pts_attr.Set(vec_pts)
        vel_attr = pts_geom.GetVelocitiesAttr()
        if vel_attr and vel_attr.IsValid():
            vel_attr.Set(vec_zeros)
        else:
            pv_attr = prim.GetAttribute("velocities")
            if pv_attr and pv_attr.IsValid():
                pv_attr.Set(vec_zeros)
        restored += 1
    return restored


def prelift_fixed_base_torso_at_reset(
    scene: Any,
    target_lift_m: float = 0.46,
    *,
    physics_dt: float = 1.0 / 60.0,
) -> None:
    """Smoothly elevate the Lift2 torso and settle before t=0 to match online evaluation reset."""
    if not getattr(scene, "robot_list", None):
        return
    emb_obj = scene.robot_list[0]
    fq = np.asarray(emb_obj.robot.get_joint_positions(), dtype=float).copy()
    init_16 = fq[emb_obj.default_dof_indices].copy()
    init_base = fq[emb_obj.base_dof_indices].copy()
    s_lift = float(fq[emb_obj.body_dof_indices[0]])
    ramp_steps = scale_ticks_for_physics_dt(20, physics_dt)
    hold_steps = scale_ticks_for_physics_dt(16, physics_dt)
    for k in range(1, ramp_steps + 1):
        fq[emb_obj.default_dof_indices] = init_16
        fq[emb_obj.body_dof_indices] = [s_lift + (target_lift_m - s_lift) * (k / float(ramp_steps))]
        fq[emb_obj.base_dof_indices] = init_base
        emb_obj.robot_view.set_joint_position_targets(fq.tolist())
        scene.world.step(render=False)
    for _ in range(hold_steps):
        fq[emb_obj.default_dof_indices] = init_16
        fq[emb_obj.body_dof_indices] = [target_lift_m]
        fq[emb_obj.base_dof_indices] = init_base
        emb_obj.robot_view.set_joint_position_targets(fq.tolist())
        scene.world.step(render=False)


def _quat_wxyz_to_rot(q_wxyz: NDArray[Any]) -> R:
    return R.from_quat([float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])])


def _refine_ik_dls(
    planner: Any,
    p_goal: NDArray[Any],
    q_goal_wxyz: NDArray[Any],
    q_init: NDArray[Any],
) -> tuple[NDArray[Any], float]:
    """Refine a 6-DOF arm IK seed to sub-0.05 mm accuracy via batched-FK damped least squares."""
    import torch

    dev = planner.tensor_args.device
    jl = planner.ik_solver.kinematics.get_joint_limits()
    q_lo = jl.position[0, :6].detach().cpu().numpy().astype(np.float64)
    q_hi = jl.position[1, :6].detach().cpu().numpy().astype(np.float64)
    q = np.clip(np.asarray(q_init, dtype=np.float64).copy(), q_lo, q_hi)
    r_goal = _quat_wxyz_to_rot(q_goal_wxyz)
    eps = 1e-4
    eye6 = np.eye(6, dtype=np.float64)
    best_q = q.copy()
    best_cost = 1e9
    for _ in range(16):
        batch = np.vstack([q[None, :], q[None, :] + eps * eye6]).astype(np.float32)
        res = planner.ik_solver.fk(torch.from_numpy(batch).to(dev))
        pos = res.ee_position.detach().cpu().numpy().astype(np.float64)
        quat = res.ee_quaternion.detach().cpu().numpy().astype(np.float64)
        p_err = p_goal - pos[0]
        r0 = _quat_wxyz_to_rot(quat[0])
        r_err = (r_goal * r0.inv()).as_rotvec()
        err = np.concatenate([p_err, r_err])
        cost = float(np.linalg.norm(p_err) + 0.2 * np.linalg.norm(r_err))
        if cost < best_cost:
            best_cost = cost
            best_q = q.copy()
        if np.linalg.norm(p_err) < 4e-5 and np.linalg.norm(r_err) < 4e-4:
            break
        jac = np.zeros((6, 6), dtype=np.float64)
        for j in range(6):
            dp = (pos[j + 1] - pos[0]) / eps
            rj = _quat_wxyz_to_rot(quat[j + 1])
            dr = (rj * r0.inv()).as_rotvec() / eps
            jac[:, j] = np.concatenate([dp, dr])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + (2e-3**2) * eye6, err)
        max_step = float(np.max(np.abs(dq)))
        if max_step > 0.12:
            dq *= 0.12 / max_step
        q = np.clip(q + dq, q_lo, q_hi)
    return best_q, best_cost


def _wrap_goal_leaves(goal: Any) -> Any:
    if isinstance(goal, list):
        return [_wrap_goal_leaves(g) for g in goal]
    if isinstance(goal, dict):
        if "sub_goal_setting" in goal:
            return dict(goal)
        return {
            "type": goal.get("type", "manip/default/sr_based_genmanip_relationship"),
            "skip_steps": goal.get("skip_steps", 1),
            "succ_cnts": goal.get("succ_cnts", 0),
            "never_reset": goal.get("never_reset", False),
            "sub_goal_setting": goal,
        }
    return goal


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class IsaacGenManipSimBackend:
    """Live Isaac Sim 4.1 + GenManip + CuRobo SimBackendProtocol implementation."""

    def __init__(
        self,
        *,
        job_id: str,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        mode: str,
        scene: Any,
        task_data: Mapping[str, Any],
        output_root: Path,
        max_steps: int = 2000,
        render_camera: Any | None = None,
        sensor_cameras: Mapping[str, Any] | None = None,
        camera_stride: int = 4,
        episode_id: str | None = None,
        physics_dt: float = 1.0 / 60.0,
        metric_eval_stride: int = 4,
    ) -> None:
        if mode not in ALLOWED_EXECUTION_MODES:
            raise ContractValidationError(f"Unsupported execution mode: {mode!r}")
        self.job_id = job_id
        self.brief = brief
        self.embodiment = embodiment
        self.mode = mode
        self.scene = scene
        self.task_data = task_data
        self.output_root = output_root
        self.max_steps = max_steps
        self.render_camera = render_camera
        self.sensor_cameras: dict[str, Any] = dict(sensor_cameras or {})
        self.camera_stride = max(1, int(camera_stride))
        self.episode_id = episode_id
        self.physics_dt = physics_dt
        self.metric_eval_stride = max(1, metric_eval_stride)

        self.emb = scene.robot_list[0]
        self.steps_executed = 0
        self.snapshots_used = 0
        self.controller_logs: list[str] = []
        self.saved_snapshots: list[SnapshotHandle] = []
        self._snapshot_blobs: dict[str, dict[str, Any]] = {}

        # Robot target hold registers so stepping one chain/group holds everything else rock steady.
        full_q = np.asarray(self.emb.robot.get_joint_positions(), dtype=float)
        self._full_dof_targets = full_q.copy()
        self._target_16dof = np.asarray(full_q[self.emb.default_dof_indices], dtype=float).copy()
        self._target_lift = float(full_q[self.emb.body_dof_indices[0]])
        if not hasattr(self.emb, "target_base_pose") or self.emb.target_base_pose is None:
            self.emb.target_base_pose = full_q[self.emb.base_dof_indices].copy()
        self._initial_base_pose = np.asarray(self.emb.target_base_pose, dtype=float).copy()

        # Build standalone per-stage leaf evaluators so any stage can be queried independently at any time.
        from genmanip.core.metrics.metrics_manager import MetricsManager
        from genmanip.core.metrics.utils import MetricFactory
        import genmanip.extensions.metrics  # noqa: F401

        register_pbd_containment_metric()
        raw_goal = task_data.get("goal", [])
        self._pbd_goal_specs: list[dict[str, Any]] = extract_pbd_goal_specs(raw_goal)
        self._pbd_containment_t0: list[dict[str, Any]] = [
            evaluate_scene_pbd_containment(scene, spec) for spec in self._pbd_goal_specs
        ]
        self._iso_mesh_logged: set[str] = set()
        wrapped_goal = MetricsManager.ensure_list_nesting(_wrap_goal_leaves(raw_goal), 3)
        self._standalone_stage_metrics: list[list[list[Any]]] = []
        for stage_entry in wrapped_goal:
            unions: list[list[Any]] = []
            for union_list in stage_entry:
                unions.append([MetricFactory.build(cfg["type"], **cfg) for cfg in union_list])
            self._standalone_stage_metrics.append(unions)

        self._bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
            useExtentsHint=False,
        )
        self._guide_bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
            useExtentsHint=False,
        )
        self._xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # Sequential stage tracking
        self._highest_stage_satisfied = -1
        self.stage_progression: list[StageProgressEntry] = []

        # Recording buffers (firewall clean: ONLY non-privileged sensorimotor channels + render media)
        self._traj_proprio: list[NDArray[Any]] = []
        self._traj_action: list[NDArray[Any]] = []
        self._traj_base_target: list[NDArray[Any]] = []
        self._traj_torso_lift: list[float] = []
        self._traj_timestamps: list[float] = []
        self._traj_state_ee_pose: list[NDArray[Any]] = []
        self._traj_action_ee_pose: list[NDArray[Any]] = []
        self._traj_camera_step_indices: list[int] = []
        self._traj_sensor_frames: dict[str, list[NDArray[Any]]] = {
            name: [] for name in self.sensor_cameras
        }
        self._traj_rgb_frames: list[NDArray[Any]] = []
        self._media_keyframes: list[tuple[str, str, NDArray[Any]]] = []
        self._video_frames: list[NDArray[Any]] = []
        self._pbd_dispensers: list[dict[str, Any]] = []
        self._weight_scales: list[dict[str, Any]] = []
        self._bean_grinders: list[dict[str, Any]] = []
        self._conveyors: list[dict[str, Any]] = []
        self._closeup_camera: Any = None
        self._init_pbd_dispensers()
        self._init_weight_scales()
        self._init_bean_grinders()
        self._init_conveyors()
        self._init_closeup_camera()

        rob_pos0, _ = self.emb.robot.get_world_pose()
        p_l0, q_l0 = self.emb.robot_base_left.get_world_pose()
        p_r0, _ = self.emb.robot_base_right.get_world_pose()
        mid_init = 0.5 * (np.asarray(p_l0, dtype=float) + np.asarray(p_r0, dtype=float))
        rot_l_init = R.from_quat([float(q_l0[1]), float(q_l0[2]), float(q_l0[3]), float(q_l0[0])])
        self._mid_to_root_local = rot_l_init.inv().apply(np.asarray(rob_pos0, dtype=float) - mid_init)

        self._t0_snapshot_blob = self._capture_snapshot_blob()
        if self.render_camera is not None:
            rgb0 = self._grab_camera_rgb(accum_renders=4)
            if rgb0 is not None:
                self._media_keyframes.append(("t0_initial", "t0_initial.png", rgb0))
                self._video_frames.append(rgb0)

    def _capture_snapshot_blob(self) -> dict[str, Any]:
        from genmanip.core.metrics.metrics_manager import MetricsManager

        dynamic_objects: dict[str, dict[str, list[float]]] = {}
        for uid, obj in self.scene.object_list.items():
            if uid == "00000000000000000000000000000000":
                continue
            info = self.task_data.get("initial_layout", {}).get(uid, {})
            if isinstance(info, Mapping) and info.get("is_articulation_part", False):
                continue
            if any(uid.startswith(f"{aid}_") for aid in self.scene.articulation_list):
                continue
            pos, ori = obj.get_world_pose()
            dynamic_objects[uid] = {
                "pos": [float(x) for x in np.asarray(pos, dtype=float)],
                "ori": [float(x) for x in np.asarray(ori, dtype=float)],
            }

        articulations: dict[str, dict[str, list[float]]] = {}
        for uid, art in self.scene.articulation_list.items():
            jpos = np.asarray(art.get_joint_positions(), dtype=float)
            articulations[uid] = {
                "joint_positions": [float(x) for x in jpos],
            }

        rob_pos, rob_ori = self.emb.robot.get_world_pose()
        rob_q = np.asarray(self.emb.robot.get_joint_positions(), dtype=float)
        mm = self.scene.metric_manager
        metric_scores_copy = json.loads(json.dumps(mm.metric_score))
        _ = MetricsManager  # keep ref clear
        return {
            "steps_executed": self.steps_executed,
            "highest_stage_satisfied": self._highest_stage_satisfied,
            "dynamic_objects": dynamic_objects,
            "articulations": articulations,
            "robot": {
                "pos": [float(x) for x in np.asarray(rob_pos, dtype=float)],
                "ori": [float(x) for x in np.asarray(rob_ori, dtype=float)],
                "joint_positions": [float(x) for x in rob_q],
                "target_16dof": [float(x) for x in self._target_16dof],
                "target_lift": float(self._target_lift),
                "target_base_pose": [float(x) for x in np.asarray(self.emb.target_base_pose, dtype=float)],
            },
            "metric_long_stride_idx": int(mm.long_stride_idx),
            "metric_score": metric_scores_copy,
        }

    def _apply_snapshot_blob(self, blob: Mapping[str, Any]) -> None:
        from genmanip.core.metrics.metrics_manager import MetricsManager
        from genmanip.utils.usd_utils.prim_utils import clean_prim_velocity

        for uid, pdata in blob.get("dynamic_objects", {}).items():
            if uid in self.scene.object_list:
                self.scene.object_list[uid].set_world_pose(
                    np.asarray(pdata["pos"], dtype=float),
                    np.asarray(pdata["ori"], dtype=float),
                )
                clean_prim_velocity(self.scene.object_list[uid].prim_path)

        for uid, adata in blob.get("articulations", {}).items():
            if uid in self.scene.articulation_list:
                art = self.scene.articulation_list[uid]
                q = np.asarray(adata["joint_positions"], dtype=float)
                art.set_joint_positions(q)
                if hasattr(art, "set_joint_velocities"):
                    art.set_joint_velocities(np.zeros_like(q))
        configure_articulation_drives_at_reset(self.scene, self.task_data.get("initial_layout", {}))

        rob = blob["robot"]
        self.emb.robot.set_world_pose(
            np.asarray(rob["pos"], dtype=float),
            np.asarray(rob["ori"], dtype=float),
        )
        rob_q = np.asarray(rob["joint_positions"], dtype=float)
        self.emb.robot.set_joint_positions(rob_q)
        if hasattr(self.emb.robot, "set_joint_velocities"):
            self.emb.robot.set_joint_velocities(np.zeros_like(rob_q))
        self._full_dof_targets = rob_q.copy()
        self._target_16dof = np.asarray(rob["target_16dof"], dtype=float).copy()
        self._target_lift = float(rob["target_lift"])
        self.emb.target_base_pose = np.asarray(rob["target_base_pose"], dtype=float).copy()
        self._apply_robot_targets()

        # Reconstruct MetricsManager cleanly at exact stride index
        raw_goal = self.task_data.get("goal", [])
        wrapped_goal = _wrap_goal_leaves(raw_goal)
        fresh_mm = MetricsManager(wrapped_goal)
        target_stride = int(blob.get("metric_long_stride_idx", 0))
        while fresh_mm.long_stride_idx < target_stride and fresh_mm.cur_union_metric is not None:
            fresh_mm.cur_union_metric = fresh_mm.get_next_metric()
        if target_stride >= len(fresh_mm.metric_score):
            fresh_mm.cur_union_metric = None
        fresh_mm.metric_score = json.loads(json.dumps(blob.get("metric_score", fresh_mm.metric_score)))
        self.scene.metric_manager = fresh_mm
        self._highest_stage_satisfied = int(blob.get("highest_stage_satisfied", -1))

        # Step a few stabilization ticks with restored targets held
        for _ in range(8):
            self._apply_robot_targets()
            self.scene.world.step(render=False)

    def load_and_restore_snapshot_file(self, snapshot_id: str) -> None:
        if self.mode != "privileged_search":
            raise FirewallViolationError(
                f"resume_from_snapshot_id is forbidden in {self.mode!r} mode"
            )
        snap_path = self.output_root / "snapshots" / f"{snapshot_id}.json"
        if not snap_path.exists():
            raise ContractValidationError(f"Snapshot file not found: {snap_path}")
        blob = json.loads(snap_path.read_text(encoding="utf-8"))
        self._apply_snapshot_blob(blob)
        self.snapshots_used += 1

    def save_snapshot(self, label: str) -> SnapshotHandle:
        if self.mode != "privileged_search":
            raise FirewallViolationError(f"save_snapshot is forbidden in {self.mode!r} mode")
        self.snapshots_used += 1
        snap_id = f"{self.job_id}_snap_{self.snapshots_used}"
        blob = self._capture_snapshot_blob()
        self._snapshot_blobs[snap_id] = blob
        snap_dir = self.output_root / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / f"{snap_id}.json").write_text(
            json.dumps(blob, indent=2),
            encoding="utf-8",
        )
        handle = SnapshotHandle(snapshot_id=snap_id, label=label, step_index=self.steps_executed)
        self.saved_snapshots.append(handle)
        return handle

    def restore_snapshot(self, handle: SnapshotHandle) -> None:
        if self.mode != "privileged_search":
            raise FirewallViolationError(f"restore_snapshot is forbidden in {self.mode!r} mode")
        blob = self._snapshot_blobs.get(handle.snapshot_id)
        if blob is None:
            snap_path = self.output_root / "snapshots" / f"{handle.snapshot_id}.json"
            if not snap_path.exists():
                raise ContractValidationError(f"Unknown snapshot_id: {handle.snapshot_id!r}")
            blob = json.loads(snap_path.read_text(encoding="utf-8"))
        self._apply_snapshot_blob(blob)
        self.snapshots_used += 1

    def reset_to_t0(self) -> None:
        if self.mode != "privileged_search":
            raise FirewallViolationError(f"reset_to_t0 is forbidden in {self.mode!r} mode")
        self._apply_snapshot_blob(self._t0_snapshot_blob)
        self.snapshots_used += 1

    def _find_object_prim_path(self, uid: str) -> str:
        if uid in self.scene.object_list and getattr(self.scene.object_list[uid], "prim_path", None):
            return str(self.scene.object_list[uid].prim_path)
        if uid in self.scene.articulation_list and getattr(self.scene.articulation_list[uid], "prim_path", None):
            return str(self.scene.articulation_list[uid].prim_path)
        if uid in self.scene.articulation_part_list and getattr(self.scene.articulation_part_list[uid], "prim_path", None):
            return str(self.scene.articulation_part_list[uid].prim_path)
        candidate = f"/World/{self.scene.uuid}/obj_{uid}"
        if self.scene.world.stage.GetPrimAtPath(candidate).IsValid():
            return candidate
        raise KeyError(f"Unknown scene entity: {uid!r}")

    def live_object_pose(self, object_id: str) -> Pose3D:
        if object_id in self.scene.articulation_list:
            pos, ori = self.scene.articulation_list[object_id].get_world_pose()
        elif object_id in self.scene.object_list:
            pos, ori = self.scene.object_list[object_id].get_world_pose()
        elif object_id in self.scene.articulation_part_list:
            pos, ori = self.scene.articulation_part_list[object_id].get_world_pose()
        else:
            raise KeyError(f"Unknown object_id: {object_id!r}")
        p = np.asarray(pos, dtype=float)
        q = np.asarray(ori, dtype=float)
        return Pose3D(
            xyz=(float(p[0]), float(p[1]), float(p[2])),
            wxyz=normalize_quat_wxyz((float(q[0]), float(q[1]), float(q[2]), float(q[3]))),
        )

    def _prim_aabb(self, prim_path: str) -> AABB3D:
        prim = self.scene.world.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise KeyError(f"Invalid USD prim path: {prim_path!r}")
        self._bbox_cache.Clear()
        bound = self._bbox_cache.ComputeWorldBound(prim)
        aligned = bound.ComputeAlignedBox()
        if aligned.IsEmpty():
            self._guide_bbox_cache.Clear()
            aligned = self._guide_bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = aligned.GetMin()
        mx = aligned.GetMax()
        return AABB3D(
            min_xyz=(float(mn[0]), float(mn[1]), float(mn[2])),
            max_xyz=(float(mx[0]), float(mx[1]), float(mx[2])),
        )

    def live_object_aabb(self, object_id: str) -> AABB3D:
        return self._prim_aabb(self._find_object_prim_path(object_id))

    def live_part_pose(self, part_id: str) -> Pose3D:
        if part_id in self.scene.articulation_part_list:
            pos, ori = self.scene.articulation_part_list[part_id].get_world_pose()
            p = np.asarray(pos, dtype=float)
            q = np.asarray(ori, dtype=float)
            return Pose3D(
                xyz=(float(p[0]), float(p[1]), float(p[2])),
                wxyz=normalize_quat_wxyz((float(q[0]), float(q[1]), float(q[2]), float(q[3]))),
            )
        return self.live_object_pose(part_id)

    def live_part_aabb(self, part_id: str) -> AABB3D:
        return self._prim_aabb(self._find_object_prim_path(part_id))

    def live_joint_positions(self, object_id: str) -> tuple[float, ...]:
        if object_id not in self.scene.articulation_list:
            raise KeyError(f"Object {object_id!r} is not an articulation")
        vals = np.asarray(self.scene.articulation_list[object_id].get_joint_positions(), dtype=float)
        return tuple(float(v) for v in vals)

    def live_joint_value(self, object_id: str, semantic_joint: str) -> float:
        if object_id not in self.brief.objects:
            raise KeyError(f"Unknown object_id: {object_id!r}")
        obj = self.brief.objects[object_id]
        if semantic_joint not in obj.dofs:
            raise KeyError(f"Object {object_id!r} has no dof {semantic_joint!r}")
        positions = self.live_joint_positions(object_id)
        idx = obj.dofs[semantic_joint].dof_index
        return float(positions[idx])

    def _find_usd_joint_prim(self, object_id: str, joint_prim_hint: str) -> Usd.Prim | None:
        stage = self.scene.world.stage
        art = self.scene.articulation_list.get(object_id)
        if art is None:
            return None
        art_root = str(getattr(art, "prim_path", ""))
        target_name = Path(joint_prim_hint).name
        if joint_prim_hint:
            # Try stripping leading /AssetName or /World/... prefixes and searching under art_root
            for prim in Usd.PrimRange(stage.GetPrimAtPath(art_root)):
                if (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)) and (
                    str(prim.GetPath()).endswith(joint_prim_hint) or prim.GetName() == target_name
                ):
                    return prim
        return None

    def live_joint_info(self, object_id: str, semantic_joint: str) -> JointAxisInfo:
        val = self.live_joint_value(object_id, semantic_joint)
        obj = self.brief.objects[object_id]
        dof = obj.dofs[semantic_joint]
        jprim = self._find_usd_joint_prim(object_id, dof.joint_prim)
        if jprim is None:
            pose = self.live_object_pose(object_id)
            return JointAxisInfo(
                object_id=object_id,
                semantic_joint=semantic_joint,
                dof_index=dof.dof_index,
                joint_type="unknown",
                pivot_world=pose.xyz,
                axis_world=(1.0, 0.0, 0.0),
                lower_limit=-math.pi,
                upper_limit=math.pi,
                current_position=val,
            )

        is_rev = jprim.IsA(UsdPhysics.RevoluteJoint)
        jtype = "revolute" if is_rev else "prismatic"
        axis_tok = str(jprim.GetAttribute("physics:axis").Get() or "X").upper()
        axis_local = {
            "X": np.array([1.0, 0.0, 0.0]),
            "Y": np.array([0.0, 1.0, 0.0]),
            "Z": np.array([0.0, 0.0, 1.0]),
        }.get(axis_tok, np.array([1.0, 0.0, 0.0]))
        lp0 = jprim.GetAttribute("physics:localPos0").Get()
        pos0_local = np.array([float(lp0[0]), float(lp0[1]), float(lp0[2])]) if lp0 is not None else np.zeros(3)
        body0_targets = jprim.GetRelationship("physics:body0").GetTargets()
        self._xform_cache.Clear()
        if body0_targets:
            body0_prim = self.scene.world.stage.GetPrimAtPath(body0_targets[0])
            gf_mat = self._xform_cache.GetLocalToWorldTransform(body0_prim)
        else:
            art_prim = self.scene.world.stage.GetPrimAtPath(str(self.scene.articulation_list[object_id].prim_path))
            gf_mat = self._xform_cache.GetLocalToWorldTransform(art_prim)

        # Note USD GfMatrix4d uses row-vector convention: world_pt = Transform(local_pt), ExtractRotationMatrix() maps column or row
        # Transform point reliably via Gf.Vec3d
        from pxr import Gf

        p_w = gf_mat.Transform(Gf.Vec3d(float(pos0_local[0]), float(pos0_local[1]), float(pos0_local[2])))
        a_w = gf_mat.TransformDir(Gf.Vec3d(float(axis_local[0]), float(axis_local[1]), float(axis_local[2])))
        a_norm = math.sqrt(a_w[0] ** 2 + a_w[1] ** 2 + a_w[2] ** 2) or 1.0
        axis_world = (float(a_w[0] / a_norm), float(a_w[1] / a_norm), float(a_w[2] / a_norm))
        pivot_world = (float(p_w[0]), float(p_w[1]), float(p_w[2]))

        lo_raw = float(jprim.GetAttribute("physics:lowerLimit").Get() or 0.0)
        hi_raw = float(jprim.GetAttribute("physics:upperLimit").Get() or 0.0)
        if is_rev:
            lo_raw = math.radians(lo_raw)
            hi_raw = math.radians(hi_raw)
        return JointAxisInfo(
            object_id=object_id,
            semantic_joint=semantic_joint,
            dof_index=dof.dof_index,
            joint_type=jtype,
            pivot_world=pivot_world,
            axis_world=axis_world,
            lower_limit=lo_raw,
            upper_limit=hi_raw,
            current_position=val,
        )

    def live_stage_scores(self) -> tuple[float, ...]:
        self.scene._frame_pc_cache = {}
        self.scene._frame_hull_cache = {}
        self.scene._frame_tree_cache = {}
        out: list[float] = []
        for unions in self._standalone_stage_metrics:
            union_vals: list[float] = []
            for conj in unions:
                if not conj:
                    continue
                vals = [float(m.weight) if m.check_status(self.scene) else 0.0 for m in conj]
                union_vals.append(sum(vals) / len(vals))
            out.append(round(max(union_vals) if union_vals else 0.0, 5))
        return tuple(out)

    def live_current_score(self) -> float:
        return float(self.scene.metric_manager.calc_overall_score())

    def live_contacts_summary(self) -> dict[str, Any]:
        return {
            "backend": "isaac_genmanip_41",
            "steps_executed": self.steps_executed,
            "stage_scores": list(self.live_stage_scores()),
        }

    def live_fluid_containment_ratio(
        self,
        vessel_id: str,
        *,
        particle_prim_path: str | None = None,
        fallback: float = 0.0,
    ) -> float:
        stage = self.scene.world.stage
        # Support rigid-body granule sets (e.g. /World/CoffeeBeans) for unified containment telemetry
        if particle_prim_path and particle_prim_path.startswith("/World/CoffeeBeans"):
            if self._weight_scales:
                sc = self._weight_scales[0]
                bowl_x, bowl_y, bowl_z = sc["bowl_xyz"]
                n_tot = len(sc["bean_prims"])
                if n_tot == 0:
                    return float(fallback)
                n_in = 0
                if vessel_id == "single_dose_grinder" and self._bean_grinders:
                    gr = self._bean_grinders[0]
                    gx, gy, gz = gr["hopper_xyz"]
                    for bp in sc["bean_prims"]:
                        pos, _ = bp.get_world_pose()
                        r = float(np.hypot(float(pos[0]) - gx, float(pos[1]) - gy))
                        if r <= 0.062 and (gz - 0.130) <= float(pos[2]) <= (gz + 0.090):
                            n_in += 1
                    return float(n_in / n_tot)
                if vessel_id == "bean_weighing_bowl":
                    # Support BOTH static bowl (v5) and dynamic graspable bowl (v6)!
                    try:
                        b_pose = self.live_object_pose("bean_weighing_bowl")
                        bx, by, bz = b_pose.xyz
                    except Exception:
                        bx, by, bz = bowl_x, bowl_y, bowl_z
                    for bp in sc["bean_prims"]:
                        pos, _ = bp.get_world_pose()
                        r = float(np.hypot(float(pos[0]) - bx, float(pos[1]) - by))
                        if r <= 0.056 and (bz - 0.004) <= float(pos[2]) <= (bz + 0.080):
                            n_in += 1
                    return float(n_in / n_tot)
                try:
                    v_pose = self.live_object_pose(vessel_id)
                    vx, vy, vz = v_pose.xyz
                    for bp in sc["bean_prims"]:
                        pos, _ = bp.get_world_pose()
                        r = float(np.hypot(float(pos[0]) - vx, float(pos[1]) - vy))
                        if r <= 0.038 and (vz - 0.005) <= float(pos[2]) <= (vz + 0.095):
                            n_in += 1
                    return float(n_in / n_tot)
                except Exception:
                    return float(fallback)
        target_prim = None
        if particle_prim_path:
            cand = stage.GetPrimAtPath(particle_prim_path)
            if cand and cand.IsValid() and cand.IsA(UsdGeom.Points):
                target_prim = cand
        if target_prim is None:
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Points) and prim.HasAPI(PhysxSchema.PhysxParticleSetAPI):
                    target_prim = prim
                    break
        if target_prim is None:
            return float(fallback)
        pts_raw = UsdGeom.Points(target_prim).GetPointsAttr().Get()
        if not pts_raw:
            return float(fallback)
        self._xform_cache.Clear()
        gf_mat = self._xform_cache.GetLocalToWorldTransform(target_prim)
        sx = float(gf_mat.GetRow3(0).GetLength() or 1.0)
        sy = float(gf_mat.GetRow3(1).GetLength() or 1.0)
        sz = float(gf_mat.GetRow3(2).GetLength() or 1.0)
        mat_ortho = gf_mat.RemoveScaleShear()
        trans = gf_mat.ExtractTranslation()
        rot_q = mat_ortho.ExtractRotationQuat()
        imag = rot_q.GetImaginary()
        world_pose = Pose3D(
            xyz=(float(trans[0]), float(trans[1]), float(trans[2])),
            wxyz=normalize_quat_wxyz(
                (float(rot_q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
            ),
        )
        pts_local = [(float(p[0]) * sx, float(p[1]) * sy, float(p[2]) * sz) for p in pts_raw]
        pts_world = transform_points_local_to_world(pts_local, world_pose)
        try:
            aabb = self.live_object_aabb(vessel_id)
        except KeyError:
            return float(fallback)
        fill_min, fill_max = compute_vessel_fill_bounds(aabb.min_xyz, aabb.max_xyz)
        return float(
            compute_aabb_containment_ratio(
                pts_world,
                box_min=fill_min,
                box_max=fill_max,
            )
        )

    def _arm_name(self, chain: str) -> str:
        if chain == "left_arm":
            return "left"
        if chain == "right_arm":
            return "right"
        raise KeyError(f"Unsupported Lift2 chain: {chain!r}")

    def _planner_and_slice(self, chain: str) -> tuple[Any, slice, str]:
        arm = self._arm_name(chain)
        if arm == "left":
            return self.emb.left_planner, slice(0, 6), "left"
        return self.emb.right_planner, slice(8, 14), "right"

    def ee_pose(self, chain: str) -> Pose3D:
        planner, sl, arm = self._planner_and_slice(chain)
        q_arm = np.asarray(self._cur_16dof()[sl], dtype=float)
        p_rel, q_rel = planner.fk_single(q_arm)
        base_prim = self.emb.robot_base_left if arm == "left" else self.emb.robot_base_right
        base_p, base_q = base_prim.get_world_pose()
        mat_world = _pose_to_mat(base_p, base_q) @ _pose_to_mat(p_rel, q_rel)
        pose_l6 = _mat_to_pose(mat_world)
        return link6_world_to_tcp_world(
            np.asarray(pose_l6.xyz, dtype=float),
            np.asarray(pose_l6.wxyz, dtype=float),
        )

    def _cur_16dof(self) -> NDArray[Any]:
        full_q = np.asarray(self.emb.robot.get_joint_positions(), dtype=float)
        return np.asarray(full_q[self.emb.default_dof_indices], dtype=float)

    def _solve_ik_internal(
        self,
        chain: str,
        target_pose: Pose3D,
        seed_6dof: NDArray[Any] | None = None,
    ) -> NDArray[Any]:
        planner, sl, arm = self._planner_and_slice(chain)
        p_l6, q_l6 = tcp_world_to_link6_world(target_pose)
        p_arm, q_arm = self.emb._transform_goal_pose((p_l6, q_l6), arm=arm)
        p_goal = np.asarray(p_arm, dtype=np.float64)
        q_goal = np.asarray(q_arm, dtype=np.float64)
        pose_7d = np.concatenate([np.asarray(p_arm, dtype=float), np.asarray(q_arm, dtype=float)])
        cur_seed = (
            np.asarray(seed_6dof, dtype=float)
            if seed_6dof is not None
            else np.asarray(self._target_16dof[sl], dtype=float)
        )
        q_loc, c_loc = _refine_ik_dls(planner, p_goal, q_goal, cur_seed)
        if c_loc < 1.5e-3:
            return q_loc
        candidates = [
            cur_seed,
            np.zeros(6, dtype=float),
            np.array([0.0, 0.8, 1.0, 0.0, 0.5, 0.0], dtype=float),
            np.array([0.3 if arm == "left" else -0.3, 1.0, 1.1, 0.0, 0.4, 0.0], dtype=float),
        ]
        refined_pool: list[tuple[NDArray[Any], float]] = []
        for seed in candidates:
            sol = planner.ik_single(pose_7d, seed)
            if sol is not None:
                q_ref, c_ref = _refine_ik_dls(planner, p_goal, q_goal, np.asarray(sol[:6], dtype=np.float64))
                refined_pool.append((q_ref, c_ref))
        if not refined_pool and hasattr(planner, "ik_solver") and hasattr(planner, "tensor_args"):
            from curobo.types.math import Pose
            import torch

            dev = planner.tensor_args.device
            gp = Pose(
                torch.tensor(p_goal, dtype=torch.float32, device=dev).unsqueeze(0),
                torch.tensor(q_goal, dtype=torch.float32, device=dev).unsqueeze(0),
            )
            res = planner.ik_solver.solve_single(gp)
            if bool(res.success.item()):
                sol_raw = res.solution[0, 0, :6].detach().cpu().numpy()
                q_ref, c_ref = _refine_ik_dls(planner, p_goal, q_goal, np.asarray(sol_raw, dtype=np.float64))
                refined_pool.append((q_ref, c_ref))
        if not refined_pool and hasattr(planner, "ik_solver") and hasattr(planner, "tensor_args"):
            s_sign = 1.0 if arm == "left" else -1.0
            dls_seeds = [
                cur_seed,
                np.zeros(6, dtype=float),
                np.array([0.0, 0.6, 0.9, 0.0, 0.5, 0.0], dtype=float),
                np.array([0.4 * s_sign, 0.9, 1.1, 0.0, 0.4, 0.0], dtype=float),
                np.array([-0.4 * s_sign, 0.8, 1.0, 0.5 * s_sign, 0.5, 0.0], dtype=float),
                np.array([0.0, 1.1, 1.3, 0.0, 0.3, 0.0], dtype=float),
                np.array([0.7 * s_sign, 0.7, 0.8, 0.0, 0.6, 0.0], dtype=float),
                np.array([-0.7 * s_sign, 0.7, 0.8, -0.5 * s_sign, 0.6, 0.0], dtype=float),
            ]
            for d_seed in dls_seeds:
                q_ref, c_ref = _refine_ik_dls(planner, p_goal, q_goal, d_seed)
                if c_ref < 2.0e-3:
                    refined_pool.append((q_ref, c_ref))
        if refined_pool:
            accurate = [item for item in refined_pool if item[1] < 5e-3]
            pool = accurate if accurate else refined_pool
            best_q_ref, _ = min(pool, key=lambda item: float(np.linalg.norm(item[0] - cur_seed)))
            return best_q_ref
        raise RuntimeError(
            f"CuRobo IK failed for chain={chain!r} tcp_xyz={target_pose.xyz} "
            f"arm_rel_xyz={tuple(round(float(v), 4) for v in p_arm)}"
        )

    def solve_ik(
        self, chain: str, target_pose: Pose3D, *, check_collision: bool = True
    ) -> tuple[float, ...]:
        _ = check_collision
        sol = self._solve_ik_internal(chain, target_pose)
        return tuple(float(v) for v in sol)

    def _apply_robot_targets(self) -> None:
        if self.embodiment.base_locked:
            self.emb.target_base_pose = self._initial_base_pose.copy()
        self._full_dof_targets[self.emb.default_dof_indices] = self._target_16dof
        self._full_dof_targets[self.emb.body_dof_indices] = [float(self._target_lift)]
        self._full_dof_targets[self.emb.base_dof_indices] = np.asarray(
            self.emb.target_base_pose, dtype=float
        )
        self.emb.robot_view.set_joint_position_targets(self._full_dof_targets.tolist())

    @staticmethod
    def _resize_rgb_wh(rgb: NDArray[Any], target_wh: tuple[int, int] = (640, 480)) -> NDArray[Any]:
        target_w, target_h = target_wh
        if rgb.shape[1] == target_w and rgb.shape[0] == target_h:
            return rgb
        import cv2

        return np.asarray(
            cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR),
            dtype=np.uint8,
        )

    def _extract_camera_rgb(self, camera: Any) -> NDArray[Any] | None:
        if camera is None:
            return None
        from genmanip.utils.usd_utils.camera_utils import get_src

        raw = None
        if getattr(camera, "_rgb_annotator", None) is not None:
            ann = camera._rgb_annotator.get_data()
            if isinstance(ann, np.ndarray) and ann.size > 0:
                raw = ann[:, :, :3]
        if raw is None:
            raw = get_src(camera, "rgb")
        if raw is None:
            return None
        arr = np.asarray(raw)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        rgb = arr[:, :, :3]
        if rgb.dtype == np.uint8:
            return rgb.copy()
        f_arr = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if f_arr.size and float(np.percentile(f_arr, 99.0)) <= 2.0:
            f_arr = f_arr * 255.0
        return np.clip(f_arr, 0.0, 255.0).astype(np.uint8)

    def _sync_pbd_visual_instancers(self) -> None:
        """Ensure native PhysX PBD Isosurface meshes have glossy fluid materials bound
        and raw particle points remain hidden (no coarse PointInstancer beads)."""
        if not hasattr(self, "_iso_mesh_logged"):
            self._iso_mesh_logged = set()
        sync_stage_pbd_isosurface_materials(
            self.scene.world.stage,
            log_sink=self.controller_logs,
            logged_paths=self._iso_mesh_logged,
        )

    def _grab_camera_rgb(
        self, accum_renders: int = 1, camera: Any | None = None
    ) -> NDArray[Any] | None:
        cam = self.render_camera if camera is None else camera
        if cam is None:
            return None
        self._sync_pbd_visual_instancers()
        for _ in range(accum_renders):
            self.scene.world.render()
        rgb = self._extract_camera_rgb(cam)
        if rgb is not None and camera is None:
            return self._compose_pip_frame(rgb)
        return rgb

    def _fk_ee_pose_14d(self, q_16dof: NDArray[Any]) -> NDArray[np.float32]:
        try:
            (p_l, q_l), (p_r, q_r) = self.emb.fk_single(np.asarray(q_16dof, dtype=float))
            vec = np.concatenate(
                [
                    np.asarray(p_l, dtype=np.float32).reshape(3),
                    np.asarray(q_l, dtype=np.float32).reshape(4),
                    np.asarray(p_r, dtype=np.float32).reshape(3),
                    np.asarray(q_r, dtype=np.float32).reshape(4),
                ]
            )
            return np.asarray(vec, dtype=np.float32)
        except Exception:  # noqa: BLE001
            return np.zeros(14, dtype=np.float32)

    @property
    def recorded_sensor_streams(self) -> tuple[str, ...]:
        """Return stream names that captured at least one valid RGB frame."""
        return tuple(k for k, frames in self._traj_sensor_frames.items() if frames)

    def _evaluate_metrics_tick(self, *, force: bool = False) -> None:
        if not force and (self.steps_executed % self.metric_eval_stride != 0):
            return
        mm = self.scene.metric_manager
        score = float(mm.step(self.scene) or 0.0)
        num_stages = len(self.brief.stages)
        satisfied_count = num_stages if mm.cur_union_metric is None else int(mm.long_stride_idx)
        newly_satisfied = satisfied_count - 1
        while self._highest_stage_satisfied < newly_satisfied and self._highest_stage_satisfied + 1 < num_stages:
            self._highest_stage_satisfied += 1
            idx = self._highest_stage_satisfied
            metric_id = self.brief.stages[idx].metric_id if idx < num_stages else f"stage_{idx}"
            self.stage_progression.append(
                StageProgressEntry(
                    stage_index=idx,
                    metric_id=metric_id,
                    score=round(score if idx == newly_satisfied else (idx + 1) / max(1, num_stages), 5),
                    step_index=self.steps_executed,
                )
            )
            if self.render_camera is not None:
                frame = self._grab_camera_rgb(accum_renders=2)
                if frame is not None:
                    self._media_keyframes.append(
                        (f"stage_{idx}_{metric_id}", f"stage_{idx}_{metric_id}.png", frame)
                    )
                    self._video_frames.append(frame)

    def _init_pbd_dispensers(self) -> None:
        stage = self.scene.world.stage
        for prim in stage.Traverse():
            if not (prim.IsA(UsdGeom.Points) and prim.HasAPI(PhysxSchema.PhysxParticleSetAPI)):
                continue
            trig_obj_attr = prim.GetAttribute("fluid:dispenseTriggerObject")
            trig_jnt_attr = prim.GetAttribute("fluid:dispenseTriggerJoint")
            if not (trig_obj_attr and trig_obj_attr.HasValue() and trig_jnt_attr and trig_jnt_attr.HasValue()):
                continue
            thr_attr = prim.GetAttribute("fluid:dispenseTriggerThreshold")
            shutter_attr = prim.GetAttribute("fluid:dispenseShutterPrim")
            shutter_path = str(shutter_attr.Get()) if (shutter_attr and shutter_attr.HasValue() and shutter_attr.Get()) else ""
            shutter_closed_xyz = None
            if shutter_path:
                sp = stage.GetPrimAtPath(shutter_path)
                if sp and sp.IsValid():
                    t_attr = sp.GetAttribute("xformOp:translate")
                    if t_attr and t_attr.HasValue():
                        tv = t_attr.Get()
                        shutter_closed_xyz = (float(tv[0]), float(tv[1]), float(tv[2]))
            self._pbd_dispensers.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "trigger_object": str(trig_obj_attr.Get()),
                    "trigger_joint": str(trig_jnt_attr.Get()),
                    "threshold": float(thr_attr.Get()) if (thr_attr and thr_attr.HasValue()) else 0.0018,
                    "shutter_path": shutter_path,
                    "shutter_closed_xyz": shutter_closed_xyz,
                    "open_progress": 0.0,
                    "triggered": False,
                }
            )

    def _tick_pbd_dispensers(self) -> None:
        stage = self.scene.world.stage
        # Bind visual fluid materials onto runtime-generated PhysX Isosurface meshes once created
        if not getattr(self, "_iso_mats_bound", False):
            from pxr import UsdShade
            bound_any = False
            for sys_p, mat_p in (
                ("/World/PBD_EspressoSystem/Isosurface", "/World/Looks/PBD_EspressoMat"),
                ("/World/PBD_MilkSystem/Isosurface", "/World/Looks/PBD_SteamedMilkMat"),
            ):
                iso_pr = stage.GetPrimAtPath(sys_p)
                mat_pr = stage.GetPrimAtPath(mat_p)
                if iso_pr and iso_pr.IsValid() and mat_pr and mat_pr.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(iso_pr).Bind(UsdShade.Material(mat_pr))
                    bound_any = True
            if bound_any:
                self._iso_mats_bound = True
        if not self._pbd_dispensers:
            return
        for disp in self._pbd_dispensers:
            if not disp["triggered"]:
                try:
                    q = self.live_joint_value(disp["trigger_object"], disp["trigger_joint"])
                except Exception:
                    q = 0.0
                if q >= disp["threshold"]:
                    disp["triggered"] = True
            if disp["triggered"] and disp["shutter_path"] and disp["shutter_closed_xyz"] is not None:
                if disp["open_progress"] < 1.0:
                    disp["open_progress"] = min(1.0, disp["open_progress"] + 0.05)
                    sp = stage.GetPrimAtPath(disp["shutter_path"])
                    if sp and sp.IsValid():
                        t_attr = sp.GetAttribute("xformOp:translate")
                        if t_attr and t_attr.IsValid():
                            cx, cy, cz = disp["shutter_closed_xyz"]
                            # Slide bottom shutter smoothly +0.14 m in +Y to release espresso stream!
                            t_attr.Set(Gf.Vec3d(cx, cy + 0.14 * disp["open_progress"], cz))
        self._tick_weight_scales()

    def _init_weight_scales(self) -> None:
        stage = self.scene.world.stage
        for prim in stage.Traverse():
            pref_attr = prim.GetAttribute("scale:beanPrefix")
            if not (pref_attr and pref_attr.HasValue() and pref_attr.Get()):
                continue
            prefix = str(pref_attr.Get())
            lcd_attr = prim.GetAttribute("scale:lcdTexturePath")
            sh_attr = prim.GetAttribute("scale:lcdShaderPath")
            bowl_attr = prim.GetAttribute("scale:bowlWorldPos")
            gpb_attr = prim.GetAttribute("scale:gramsPerBean")
            if not (lcd_attr and lcd_attr.HasValue() and sh_attr and sh_attr.HasValue() and bowl_attr and bowl_attr.HasValue()):
                continue
            bv = bowl_attr.Get()
            bowl_xyz = (float(bv[0]), float(bv[1]), float(bv[2]))
            gpb = float(gpb_attr.Get()) if (gpb_attr and gpb_attr.HasValue()) else 0.75
            bean_paths = sorted(
                str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith(prefix) and p.IsA(UsdGeom.Xform)
            )
            try:
                from omni.isaac.core.prims import RigidPrim
                bean_prims = [RigidPrim(bp) for bp in bean_paths]
            except Exception:
                bean_prims = []
            if bean_prims:
                self._weight_scales.append(
                    {
                        "lcd_base_path": Path(str(lcd_attr.Get())),
                        "shader_path": str(sh_attr.Get()),
                        "bowl_xyz": bowl_xyz,
                        "grams_per_bean": gpb,
                        "bean_prims": bean_prims,
                        "last_count": -1,
                    }
                )

    def _render_scale_lcd(self, out_path: Path, grams: float, n_beans: int, n_total: int) -> None:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (512, 224), (10, 14, 16))
        d = ImageDraw.Draw(img)
        ready = grams >= 15.0
        border_col = (34, 214, 128) if ready else (60, 120, 96)
        d.rounded_rectangle([8, 8, 504, 216], radius=16, fill=(14, 20, 22), outline=border_col, width=4)
        try:
            f_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            f_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 92)
            f_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except Exception:
            f_sm = f_lg = f_md = ImageFont.load_default()
        d.text((26, 20), "STARBUCKS BARISTA SCALE", fill=(160, 210, 185), font=f_sm)
        badge_txt = "DOSE READY" if ready else f"BEANS {n_beans:02d}/{n_total:02d}"
        badge_col = (24, 180, 102) if ready else (45, 75, 65)
        d.rounded_rectangle([318, 16, 488, 54], radius=10, fill=badge_col)
        d.text((332, 22), badge_txt, fill=(245, 255, 248), font=f_sm)
        val_col = (90, 255, 165) if ready else (225, 252, 238)
        d.text((38, 72), f"{grams:5.1f} g", fill=val_col, font=f_lg)
        d.text((36, 178), f"TARGET: {n_total * 0.75:.1f} g   TARE: 0.0 g", fill=(130, 175, 155), font=f_md)
        img.save(out_path)

    def _tick_weight_scales(self) -> None:
        if not self._weight_scales or (self.steps_executed % 6 != 0):
            return
        stage = self.scene.world.stage
        for sc in self._weight_scales:
            bx, by, bz = sc["bowl_xyz"]
            n_in = 0
            for bp in sc["bean_prims"]:
                try:
                    pos, _ = bp.get_world_pose()
                    r = float(np.hypot(float(pos[0]) - bx, float(pos[1]) - by))
                    if r <= 0.056 and (bz - 0.004) <= float(pos[2]) <= (bz + 0.080):
                        n_in += 1
                except Exception:
                    continue
            if n_in != sc["last_count"]:
                sc["last_count"] = n_in
                grams = n_in * sc["grams_per_bean"]
                out_png = sc["lcd_base_path"].with_name(f"{sc['lcd_base_path'].stem}_{n_in:02d}.png")
                try:
                    self._render_scale_lcd(out_png, grams, n_in, len(sc["bean_prims"]))
                    sh_prim = stage.GetPrimAtPath(sc["shader_path"])
                    if sh_prim and sh_prim.IsValid():
                        file_attr = sh_prim.GetAttribute("inputs:file")
                        if file_attr and file_attr.IsValid():
                            file_attr.Set(str(out_png))
                except Exception:
                    pass
                self.controller_logs.append(
                    f"[BARISTA_SCALE_LIVE] step={self.steps_executed} beans_in_bowl={n_in}/{len(sc['bean_prims'])} weight={grams:.2f}g"
                )
        self._tick_bean_grinders()

    def _init_bean_grinders(self) -> None:
        stage = self.scene.world.stage
        for prim in stage.Traverse():
            pref_attr = prim.GetAttribute("grinder:beanPrefix")
            if not (pref_attr and pref_attr.HasValue() and pref_attr.Get()):
                continue
            prefix = str(pref_attr.Get())
            hop_attr = prim.GetAttribute("grinder:hopperWorldPos")
            sh_attr = prim.GetAttribute("grinder:shutterPrim")
            ring_attr = prim.GetAttribute("grinder:statusRingPrim")
            trig_attr = prim.GetAttribute("grinder:triggerBeans")
            if not (hop_attr and hop_attr.HasValue() and sh_attr and sh_attr.HasValue()):
                continue
            hv = hop_attr.Get()
            hopper_xyz = (float(hv[0]), float(hv[1]), float(hv[2]))
            shutter_path = str(sh_attr.Get())
            shutter_closed_xyz = None
            sp = stage.GetPrimAtPath(shutter_path)
            if sp and sp.IsValid():
                ta = sp.GetAttribute("xformOp:translate")
                if ta and ta.HasValue():
                    tv = ta.Get()
                    shutter_closed_xyz = (float(tv[0]), float(tv[1]), float(tv[2]))
            bean_paths = sorted(
                str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith(prefix) and p.IsA(UsdGeom.Xform)
            )
            try:
                from omni.isaac.core.prims import RigidPrim
                bean_prims = [RigidPrim(bp) for bp in bean_paths]
            except Exception:
                bean_prims = []
            if bean_prims:
                self._bean_grinders.append(
                    {
                        "hopper_xyz": hopper_xyz,
                        "shutter_path": shutter_path,
                        "shutter_closed_xyz": shutter_closed_xyz,
                        "ring_path": str(ring_attr.Get()) if (ring_attr and ring_attr.HasValue()) else "",
                        "trigger_beans": int(trig_attr.Get()) if (trig_attr and trig_attr.HasValue()) else 14,
                        "bean_prims": bean_prims,
                        "state": "WAIT",
                        "triggered_step": -1,
                        "open_progress": 0.0,
                        "max_in_grinder": 0,
                    }
                )

    def _tick_bean_grinders(self) -> None:
        if not self._bean_grinders or (self.steps_executed % 6 != 0):
            return
        stage = self.scene.world.stage
        from pxr import UsdShade
        for gr in self._bean_grinders:
            gx, gy, gz = gr["hopper_xyz"]
            n_in = 0
            for bp in gr["bean_prims"]:
                try:
                    pos, _ = bp.get_world_pose()
                    r = float(np.hypot(float(pos[0]) - gx, float(pos[1]) - gy))
                    if r <= 0.062 and (gz - 0.130) <= float(pos[2]) <= (gz + 0.090):
                        n_in += 1
                except Exception:
                    continue
            if n_in > gr["max_in_grinder"]:
                gr["max_in_grinder"] = n_in
            if gr["state"] == "WAIT" and n_in >= gr["trigger_beans"]:
                gr["state"] = "GRINDING"
                gr["triggered_step"] = self.steps_executed
                ring_pr = stage.GetPrimAtPath(gr["ring_path"]) if gr["ring_path"] else None
                mat_gold = stage.GetPrimAtPath("/World/Looks/V6_GrinderLedGold")
                if ring_pr and ring_pr.IsValid() and mat_gold and mat_gold.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(ring_pr).Bind(
                        UsdShade.Material(mat_gold), bindingStrength=UsdShade.Tokens.strongerThanDescendants
                    )
                self.controller_logs.append(
                    f"[GRINDER_LIVE] step={self.steps_executed} state=GRINDING beans_in_hopper={n_in}/{len(gr['bean_prims'])}"
                )
            elif gr["state"] == "GRINDING":
                elapsed = self.steps_executed - gr["triggered_step"]
                # Hold 48 ticks (0.4 s) so beans are clearly visible in transparent hopper, then open burr shutter!
                if elapsed >= 48 and gr["open_progress"] < 1.0 and gr["shutter_closed_xyz"] is not None:
                    gr["open_progress"] = min(1.0, gr["open_progress"] + 0.12)
                    sp = stage.GetPrimAtPath(gr["shutter_path"])
                    if sp and sp.IsValid():
                        ta = sp.GetAttribute("xformOp:translate")
                        if ta and ta.IsValid():
                            sx, sy, sz = gr["shutter_closed_xyz"]
                            ta.Set(Gf.Vec3d(sx, sy + 0.095 * gr["open_progress"], sz))
                if elapsed >= 180:
                    gr["state"] = "READY"
                    ring_pr = stage.GetPrimAtPath(gr["ring_path"]) if gr["ring_path"] else None
                    mat_grn = stage.GetPrimAtPath("/World/Looks/V6_GrinderLedGreen")
                    if ring_pr and ring_pr.IsValid() and mat_grn and mat_grn.IsValid():
                        UsdShade.MaterialBindingAPI.Apply(ring_pr).Bind(
                            UsdShade.Material(mat_grn), bindingStrength=UsdShade.Tokens.strongerThanDescendants
                        )
                    self.controller_logs.append(
                        f"[GRINDER_LIVE] step={self.steps_executed} state=BREW_READY max_beans_ground={gr['max_in_grinder']}/{len(gr['bean_prims'])}"
                    )

    def _init_conveyors(self) -> None:
        """Discover USD-attribute-driven kinematic racetrack conveyors on the stage."""
        stage = self.scene.world.stage
        for prim in stage.Traverse():
            spd_attr = prim.GetAttribute("conveyor:speedMps")
            pref_attr = prim.GetAttribute("conveyor:slatPrefix")
            if not (spd_attr and spd_attr.HasValue() and pref_attr and pref_attr.HasValue()):
                continue
            slat_prefix = str(pref_attr.Get()).rstrip("/")
            if not slat_prefix:
                continue
            speed_mps = float(spd_attr.Get())
            L_attr = prim.GetAttribute("conveyor:straightHalfLen")
            R_attr = prim.GetAttribute("conveyor:turnRadius")
            ctr_attr = prim.GetAttribute("conveyor:centerWorld")
            z_attr = prim.GetAttribute("conveyor:surfaceZ")
            dir_attr = prim.GetAttribute("conveyor:direction")
            ph_attr = prim.GetAttribute("conveyor:initialPhaseM")

            half_len = float(L_attr.Get()) if (L_attr and L_attr.HasValue()) else 0.42
            turn_r = float(R_attr.Get()) if (R_attr and R_attr.HasValue()) else 0.16
            ctr_val = ctr_attr.Get() if (ctr_attr and ctr_attr.HasValue()) else (0.0, 0.08, 0.0)
            center_xy = (float(ctr_val[0]), float(ctr_val[1]))
            slat_z = float(z_attr.Get()) if (z_attr and z_attr.HasValue()) else 0.773
            direction = float(dir_attr.Get()) if (dir_attr and dir_attr.HasValue()) else 1.0
            phase0 = float(ph_attr.Get()) if (ph_attr and ph_attr.HasValue()) else 0.0
            perim = racetrack_perimeter(half_len, turn_r)

            slats: list[tuple[str, Any]] = []
            for p2 in stage.Traverse():
                pstr = str(p2.GetPath())
                if pstr.startswith(slat_prefix + "/") and pstr.count("/") == slat_prefix.count("/") + 1:
                    slats.append((pstr, p2))
            slats.sort(key=lambda item: item[0])
            if not slats:
                continue

            slat_entries: list[dict[str, Any]] = []
            for idx, (pstr, sp) in enumerate(slats):
                arc_attr = sp.GetAttribute("conveyor:slatArcS")
                base_s = float(arc_attr.Get()) if (arc_attr and arc_attr.HasValue()) else (idx * perim / len(slats))
                t_attr = sp.GetAttribute("xformOp:translate")
                o_attr = sp.GetAttribute("xformOp:orient")
                if not (t_attr and t_attr.IsValid()):
                    t_attr = UsdGeom.Xformable(sp).AddTranslateOp()
                if not (o_attr and o_attr.IsValid()):
                    o_attr = UsdGeom.Xformable(sp).AddOrientOp()
                is_quatd = str(o_attr.GetTypeName()) == "quatd"
                slat_entries.append(
                    {
                        "path": pstr,
                        "base_s": base_s,
                        "t_attr": t_attr,
                        "o_attr": o_attr,
                        "is_quatd": is_quatd,
                    }
                )
            self._conveyors.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "speed_mps": speed_mps,
                    "half_len": half_len,
                    "turn_r": turn_r,
                    "center_xy": center_xy,
                    "slat_z": slat_z,
                    "direction": direction,
                    "phase0": phase0,
                    "slats": slat_entries,
                }
            )
            self.controller_logs.append(
                f"[CONVEYOR_INIT] prim={prim.GetPath()} slats={len(slat_entries)} speed={speed_mps:.4f}m/s dir={direction}"
            )

    def _tick_conveyors(self) -> None:
        if not self._conveyors:
            return
        t_sim = float((self.steps_executed + 1) * self.physics_dt)
        for conv in self._conveyors:
            ds = conv["phase0"] + conv["direction"] * conv["speed_mps"] * t_sim
            half_len = conv["half_len"]
            turn_r = conv["turn_r"]
            center_xy = conv["center_xy"]
            slat_z = conv["slat_z"]
            for sl in conv["slats"]:
                x, y, z, qw, qx, qy, qz = eval_racetrack_pose(
                    sl["base_s"] + ds, half_len, turn_r, center_xy, slat_z
                )
                sl["t_attr"].Set(Gf.Vec3d(x, y, z))
                if sl["is_quatd"]:
                    sl["o_attr"].Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
                else:
                    sl["o_attr"].Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))

    def _init_closeup_camera(self) -> None:
        if self.render_camera is None:
            return
        stage = self.scene.world.stage
        cam_path = "/World/BarCloseupCam"
        if not stage.GetPrimAtPath(cam_path).IsValid():
            return
        try:
            from omni.isaac.sensor import Camera

            cam = Camera(prim_path=cam_path, resolution=(640, 360))
            cam.initialize()
            self._closeup_camera = cam
        except Exception:
            self._closeup_camera = None

    def _compose_pip_frame(self, overview_rgb: NDArray[Any]) -> NDArray[Any]:
        if self._closeup_camera is None:
            return overview_rgb
        close_rgb = self._extract_camera_rgb(self._closeup_camera)
        if close_rgb is None:
            return overview_rgb
        out = overview_rgb.copy()
        oh, ow = out.shape[:2]
        pw, ph = min(440, ow // 3 + 20), min(248, oh // 3 + 12)
        pip = self._resize_rgb_wh(close_rgb, (pw, ph))
        pad = 14
        y0, x0 = pad, ow - pw - pad
        # Crisp Starbucks green border around PiP
        out[max(0, y0 - 3):min(oh, y0 + ph + 3), max(0, x0 - 3):min(ow, x0 + pw + 3)] = (0, 112, 74)
        out[y0:y0 + ph, x0:x0 + pw] = pip
        return out

    def _step_one_tick(self) -> None:
        if self.steps_executed + 1 > self.max_steps:
            raise ContractValidationError(f"Controller exceeded max_steps={self.max_steps}")
        tick_idx = self.steps_executed

        if self.mode == "trajectory_synthesis":
            q_now = self._cur_16dof().astype(np.float32)
            a_cmd = self._target_16dof.astype(np.float32).copy()
            base_cmd = (
                np.zeros(3, dtype=np.float32)
                if self.embodiment.base_locked
                else np.asarray(self.emb.target_base_pose, dtype=np.float32).copy()
            )
            self._traj_proprio.append(q_now)
            self._traj_action.append(a_cmd)
            self._traj_base_target.append(base_cmd)
            self._traj_torso_lift.append(float(self._target_lift))
            self._traj_timestamps.append(float(tick_idx * self.physics_dt))
            self._traj_state_ee_pose.append(self._fk_ee_pose_14d(q_now))
            self._traj_action_ee_pose.append(self._fk_ee_pose_14d(a_cmd))

            if (self.render_camera is not None or self.sensor_cameras) and (
                tick_idx % self.camera_stride == 0
            ):
                self._sync_pbd_visual_instancers()
                self.scene.world.render()
                if self.sensor_cameras:
                    captured: dict[str, NDArray[Any]] = {}
                    for cam_name, cam_obj in self.sensor_cameras.items():
                        rgb = self._extract_camera_rgb(cam_obj)
                        if rgb is not None:
                            captured[cam_name] = self._resize_rgb_wh(rgb, (640, 480))
                    if len(captured) == len(self.sensor_cameras):
                        self._traj_camera_step_indices.append(tick_idx)
                        for cam_name, frame_rgb in captured.items():
                            self._traj_sensor_frames[cam_name].append(frame_rgb)
                if self.render_camera is not None and (
                    tick_idx % max(8, self.camera_stride * 2) == 0
                ):
                    overview_frame = self._extract_camera_rgb(self.render_camera)
                    if overview_frame is not None:
                        overview_frame = self._compose_pip_frame(overview_frame)
                        self._traj_rgb_frames.append(overview_frame)
                        self._video_frames.append(overview_frame)

        self._apply_robot_targets()
        self._tick_pbd_dispensers()
        self._tick_conveyors()
        self.scene.world.step(render=False)
        self.steps_executed += 1
        self._evaluate_metrics_tick(force=False)

    def move_to_pose(
        self,
        chain: str,
        target_pose: Pose3D,
        *,
        steps: int = 40,
        use_motion_gen: bool = False,
        check_collision: bool = True,
    ) -> tuple[float, ...]:
        _ = check_collision
        _, sl, arm = self._planner_and_slice(chain)
        if use_motion_gen:
            try:
                p_l6, q_l6 = tcp_world_to_link6_world(target_pose)
                sim_js = self.emb.robot.get_joints_state()
                plan = self.emb.plan_pose((p_l6, q_l6), sim_js, arm=arm)
                if plan is not None and len(plan) > 0:
                    stride = max(1, len(plan) // max(1, steps))
                    sampled = [np.asarray(wp[:6], dtype=float) for wp in plan[::stride]]
                    if not np.allclose(sampled[-1], plan[-1][:6]):
                        sampled.append(np.asarray(plan[-1][:6], dtype=float))
                    for q_wp in sampled:
                        self._target_16dof[sl] = q_wp
                        self._step_one_tick()
                    self._evaluate_metrics_tick(force=True)
                    return tuple(float(v) for v in self._target_16dof[sl])
            except Exception as exc:  # noqa: BLE001
                self.controller_logs.append(f"motion_gen fallback ({chain}): {exc}")

        start_pose = self.ee_pose(chain)
        num_ik_pts = max(2, min(12, max(2, steps // 5)))
        seed_q = self._target_16dof[sl].copy()
        ik_path: list[NDArray[Any]] = []
        try:
            for idx in range(1, num_ik_pts + 1):
                alpha = idx / num_ik_pts
                xyz = (
                    start_pose.xyz[0] + alpha * (target_pose.xyz[0] - start_pose.xyz[0]),
                    start_pose.xyz[1] + alpha * (target_pose.xyz[1] - start_pose.xyz[1]),
                    start_pose.xyz[2] + alpha * (target_pose.xyz[2] - start_pose.xyz[2]),
                )
                wxyz = slerp_wxyz(start_pose.wxyz, target_pose.wxyz, alpha)
                q_wp = self._solve_ik_internal(chain, Pose3D(xyz=xyz, wxyz=wxyz), seed_q)
                ik_path.append(q_wp)
                seed_q = q_wp
        except RuntimeError:
            q_goal = self._solve_ik_internal(chain, target_pose, self._target_16dof[sl].copy())
            ik_path = [q_goal]

        self._execute_joint_waypoints(sl, ik_path, total_steps=steps)
        return tuple(float(v) for v in self._target_16dof[sl])

    def move_arms_sync(
        self,
        targets: Mapping[str, Pose3D],
        *,
        steps: int = 40,
    ) -> dict[str, tuple[float, ...]]:
        """Move multiple arm chains synchronously tick-by-tick along Cartesian-interpolated IK paths."""
        if not targets:
            return {}
        num_ik_pts = max(2, min(12, max(2, int(steps) // 5)))
        chain_paths: dict[str, tuple[slice, list[NDArray[Any]]]] = {}
        for chain, target_pose in targets.items():
            _, sl, _ = self._planner_and_slice(chain)
            start_pose = self.ee_pose(chain)
            seed_q = self._target_16dof[sl].copy()
            ik_path: list[NDArray[Any]] = []
            try:
                for idx in range(1, num_ik_pts + 1):
                    alpha = idx / num_ik_pts
                    xyz = (
                        start_pose.xyz[0] + alpha * (target_pose.xyz[0] - start_pose.xyz[0]),
                        start_pose.xyz[1] + alpha * (target_pose.xyz[1] - start_pose.xyz[1]),
                        start_pose.xyz[2] + alpha * (target_pose.xyz[2] - start_pose.xyz[2]),
                    )
                    wxyz = slerp_wxyz(start_pose.wxyz, target_pose.wxyz, alpha)
                    q_wp = self._solve_ik_internal(chain, Pose3D(xyz=xyz, wxyz=wxyz), seed_q)
                    ik_path.append(q_wp)
                    seed_q = q_wp
            except RuntimeError:
                q_goal = self._solve_ik_internal(chain, target_pose, self._target_16dof[sl].copy())
                ik_path = [q_goal]
            pts = [self._target_16dof[sl].copy(), *[np.asarray(w, dtype=float) for w in ik_path]]
            chain_paths[chain] = (sl, pts)

        total_steps = max(2, int(steps))
        for s_idx in range(total_steps):
            for _chain, (sl, pts) in chain_paths.items():
                seg_count = len(pts) - 1
                frac = (s_idx + 1) / total_steps * seg_count
                seg = min(seg_count - 1, int(math.floor(frac)))
                local_t = frac - seg
                self._target_16dof[sl] = pts[seg] + local_t * (pts[seg + 1] - pts[seg])
            self._step_one_tick()
        self._evaluate_metrics_tick(force=True)
        return {
            chain: tuple(float(v) for v in self._target_16dof[sl])
            for chain, (sl, _pts) in chain_paths.items()
        }

    def follow_cartesian_path(
        self,
        chain: str,
        poses: Sequence[Pose3D],
        *,
        steps_per_waypoint: int = 3,
    ) -> tuple[float, ...]:
        _, sl, _ = self._planner_and_slice(chain)
        seed_q = self._target_16dof[sl].copy()
        ik_path: list[NDArray[Any]] = []
        for pose in poses:
            q_wp = self._solve_ik_internal(chain, pose, seed_q)
            ik_path.append(q_wp)
            seed_q = q_wp
        self._execute_joint_waypoints(sl, ik_path, total_steps=max(len(ik_path), len(ik_path) * steps_per_waypoint))
        return tuple(float(v) for v in self._target_16dof[sl])

    def _execute_joint_waypoints(
        self,
        sl: slice,
        waypoints: Sequence[NDArray[Any]],
        *,
        total_steps: int,
    ) -> None:
        if not waypoints:
            return
        start_q = self._target_16dof[sl].copy()
        pts = [start_q, *[np.asarray(w, dtype=float) for w in waypoints]]
        seg_count = len(pts) - 1
        steps_left = max(seg_count, int(total_steps))
        for s_idx in range(steps_left):
            frac = (s_idx + 1) / steps_left * seg_count
            seg = min(seg_count - 1, int(math.floor(frac)))
            local_t = frac - seg
            q_interp = pts[seg] + local_t * (pts[seg + 1] - pts[seg])
            self._target_16dof[sl] = q_interp
            self._step_one_tick()
        self._evaluate_metrics_tick(force=True)

    def move_joints(
        self,
        chain: str,
        joint_positions: Sequence[float],
        *,
        steps: int = 30,
    ) -> tuple[float, ...]:
        _, sl, _ = self._planner_and_slice(chain)
        goal = np.asarray(joint_positions, dtype=float)
        self._execute_joint_waypoints(sl, [goal], total_steps=steps)
        return tuple(float(v) for v in self._target_16dof[sl])

    def move_base_world(
        self,
        dx_m: float,
        dy_m: float,
        dyaw_deg: float = 0.0,
        *,
        steps: int = 30,
    ) -> tuple[float, float, float]:
        # Transform world displacement into robot frame; lift2's base_joint_y has inverted local axis sign.
        bp_now = self.base_pose_world()
        q_xyzw = np.array([bp_now.wxyz[1], bp_now.wxyz[2], bp_now.wxyz[3], bp_now.wxyz[0]], dtype=float)
        rot_root = R.from_quat(q_xyzw).as_matrix()
        d_local = rot_root.T @ np.array([dx_m, dy_m, 0.0], dtype=float)
        dx_l, dy_l = float(d_local[0]), -float(d_local[1])
        min_steps = int(
            math.ceil(max(abs(dx_l) / 0.01, abs(dy_l) / 0.01, abs(dyaw_deg) / 0.8, 1.0))
        )
        n_steps = max(int(steps), min_steps)
        for _ in range(n_steps):
            self.emb.delta_move_to(dx_l / n_steps, dy_l / n_steps, dyaw_deg / n_steps)
            self._step_one_tick()
        self._evaluate_metrics_tick(force=True)
        bp = self.base_pose_world()
        return bp.xyz

    def base_pose_world(self) -> Pose3D:
        rob_pos, _ = self.emb.robot.get_world_pose()
        p_l, q_l = self.emb.robot_base_left.get_world_pose()
        p_r, _ = self.emb.robot_base_right.get_world_pose()
        mid_now = 0.5 * (np.asarray(p_l, dtype=float) + np.asarray(p_r, dtype=float))
        rot_l_now = R.from_quat([float(q_l[1]), float(q_l[2]), float(q_l[3]), float(q_l[0])])
        pos_now = mid_now + rot_l_now.apply(self._mid_to_root_local)
        q_tot = rot_l_now.as_quat()
        return Pose3D(
            xyz=(
                float(pos_now[0]),
                float(pos_now[1]),
                float(rob_pos[2]),
            ),
            wxyz=normalize_quat_wxyz((float(q_tot[3]), float(q_tot[0]), float(q_tot[1]), float(q_tot[2]))),
        )

    def set_torso_lift(self, height_m: float, *, steps: int = 20) -> float:
        clamped = float(min(0.46, max(0.0, height_m)))
        start = float(self._target_lift)
        n_steps = max(1, int(steps))
        for idx in range(1, n_steps + 1):
            self._target_lift = start + (clamped - start) * (idx / n_steps)
            self._step_one_tick()
        self._evaluate_metrics_tick(force=True)
        return float(self.torso_lift())

    def torso_lift(self) -> float:
        full_q = np.asarray(self.emb.robot.get_joint_positions(), dtype=float)
        return float(full_q[self.emb.body_dof_indices[0]])

    def command_actuator(
        self,
        group: str,
        targets: tuple[float, ...],
        *,
        control_mode: str = "position",
        steps: int = 16,
    ) -> None:
        _ = control_mode
        grp = self.embodiment.actuator_groups[group]
        for idx, val in zip(grp.dof_indices, targets):
            self._target_16dof[idx] = float(val)
        n_steps = max(1, int(steps))
        for _ in range(n_steps):
            self._step_one_tick()
        self._evaluate_metrics_tick(force=True)

    def command_actuators_sync(
        self,
        group_targets: Mapping[str, tuple[float, ...]],
        *,
        control_mode: str = "position",
        steps: int = 16,
    ) -> None:
        _ = control_mode
        for group, targets in group_targets.items():
            grp = self.embodiment.actuator_groups[group]
            for idx, val in zip(grp.dof_indices, targets):
                self._target_16dof[idx] = float(val)
        n_steps = max(1, int(steps))
        for _ in range(n_steps):
            self._step_one_tick()
        self._evaluate_metrics_tick(force=True)

    def current_proprio(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self._cur_16dof())

    def wrist_wrench(self, chain: str) -> WrenchReading:
        return WrenchReading(chain_id=chain, force_n=(0.0, 0.0, 0.0), torque_nm=(0.0, 0.0, 0.0))

    def tactile(self, group: str) -> TactileReading:
        return TactileReading(
            group_id=group,
            normal_force_n=0.0,
            shear_force_n=(0.0, 0.0),
            slip_detected=False,
            in_contact=False,
        )

    def step(self, steps: int) -> None:
        for _ in range(int(steps)):
            self._step_one_tick()
        self._evaluate_metrics_tick(force=True)

    def _object_has_grasp_handle_or_overhang(
        self, obj_uid: str, dz: float, aabb: AABB3D
    ) -> bool:
        obj_spec = self.brief.objects.get(obj_uid)
        if (
            obj_spec is not None
            and "grasp" in obj_spec.named_frames
            and obj_uid in self.scene.object_list
        ):
            obj_pose = self.live_object_pose(obj_uid)
            grasp_pt = np.asarray(
                pose_from_named_frame(
                    obj_spec, "grasp", object_world_pose=obj_pose
                ).xyz,
                dtype=float,
            )
            prim_path = str(
                getattr(self.scene.object_list[obj_uid], "prim_path", "") or ""
            )
            obj_prim = (
                self.scene.world.stage.GetPrimAtPath(prim_path)
                if prim_path
                else None
            )
            if obj_prim and obj_prim.IsValid():
                self._guide_bbox_cache.Clear()
                for desc in Usd.PrimRange(obj_prim):
                    if not desc.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    en = UsdPhysics.CollisionAPI(desc).GetCollisionEnabledAttr().Get()
                    if en is False:
                        continue
                    box = self._guide_bbox_cache.ComputeWorldBound(desc).ComputeAlignedBox()
                    if box.IsEmpty():
                        continue
                    cmin = np.asarray(box.GetMin(), dtype=float)
                    cmax = np.asarray(box.GetMax(), dtype=float)
                    cdx = float(cmax[0] - cmin[0])
                    cdy = float(cmax[1] - cmin[1])
                    outside = np.maximum(0.0, np.maximum(cmin - grasp_pt, grasp_pt - cmax))
                    if float(np.max(outside)) <= 0.005 and (cdx <= 0.080 or cdy <= 0.080):
                        return True
        if dz <= 0.080 and (
            aabb.min_xyz[0] < -1.020
            or aabb.max_xyz[0] > 1.020
            or aabb.min_xyz[1] < -0.420
            or aabb.max_xyz[1] > 0.420
        ):
            return True
        return False

    def diagnose_blocking_stage(
        self,
        state_summary: Mapping[str, Any],
        remediation_policy: RemediationPolicySpec | None = None,
    ) -> SolvabilityDiagnostics:
        """Classify upstream physical blockers at the first incomplete stage."""
        policy = remediation_policy or RemediationPolicySpec()
        blk = state_summary.get("blocking_stage")
        final_score = float(self.live_current_score())
        default_detail = (
            f"Controller finished with final_score={final_score:.4f}; "
            f"blocking_stage={blk}; stage_scores={state_summary.get('stage_scores')}"
        )
        blk_idx = int(blk) if isinstance(blk, int) else None
        if blk_idx is None or not (0 <= blk_idx < len(self.brief.stages)):
            return SolvabilityDiagnostics(
                failure_code="STAGE_INCOMPLETE",
                stage_index=blk_idx,
                detail=default_detail,
            )

        stage_spec = self.brief.stages[blk_idx]
        for clause in stage_spec.clauses:
            ctype = str(clause.get("type", ""))
            obj_uid = str(clause.get("obj1_uid", ""))
            if ctype == "manip/default/sr_based_genmanip_relationship":
                obj_spec = self.brief.objects.get(obj_uid)
                if obj_spec is None:
                    continue
                status_cfg = clause.get("status", [])
                active_indices: dict[int, tuple[float, float]] = {}
                if isinstance(status_cfg, (list, tuple)):
                    for union_entry in status_cfg:
                        if not isinstance(union_entry, (list, tuple)):
                            continue
                        for and_entry in union_entry:
                            if not isinstance(and_entry, (list, tuple)):
                                continue
                            for idx, band in enumerate(and_entry):
                                if (
                                    isinstance(band, (list, tuple))
                                    and len(band) == 2
                                    and (float(band[0]) > -1e7 or float(band[1]) < 1e7)
                                ):
                                    active_indices[idx] = (float(band[0]), float(band[1]))
                for jname, dof in obj_spec.dofs.items():
                    if dof.dof_index not in active_indices:
                        continue
                    jprim = self._find_usd_joint_prim(obj_uid, dof.joint_prim)
                    if jprim is None:
                        continue
                    if jprim.IsA(UsdPhysics.PrismaticJoint):
                        cur_val = self.live_joint_value(obj_uid, jname)
                        lo, hi = active_indices[dof.dof_index]
                        body1_targets = jprim.GetRelationship("physics:body1").GetTargets()
                        if body1_targets and abs(cur_val) < 0.015 and (lo > 0.05 or hi < -0.05):
                            b1_prim = self.scene.world.stage.GetPrimAtPath(body1_targets[0])
                            if b1_prim and b1_prim.IsValid():
                                self._guide_bbox_cache.Clear()
                                col_mins: list[NDArray[Any]] = []
                                col_maxs: list[NDArray[Any]] = []
                                for desc in Usd.PrimRange(b1_prim):
                                    if desc.HasAPI(UsdPhysics.CollisionAPI):
                                        box = self._guide_bbox_cache.ComputeWorldBound(desc).ComputeAlignedBox()
                                        if not box.IsEmpty():
                                            col_mins.append(np.asarray(box.GetMin(), dtype=float))
                                            col_maxs.append(np.asarray(box.GetMax(), dtype=float))
                                if col_mins:
                                    cmin = np.min(np.stack(col_mins, axis=0), axis=0)
                                    cmax = np.max(np.stack(col_maxs, axis=0), axis=0)
                                    obj_pose = self.live_object_pose(obj_uid)
                                    for fname in obj_spec.named_frames:
                                        if "handle" not in fname.lower():
                                            continue
                                        hp = np.asarray(
                                            pose_from_named_frame(
                                                obj_spec, fname, object_world_pose=obj_pose
                                            ).xyz,
                                            dtype=float,
                                        )
                                        outside = np.maximum(0.0, np.maximum(cmin - hp, hp - cmax))
                                        gap_m = float(np.max(outside))
                                        if gap_m > 0.003:
                                            return apply_remediation_policy(
                                                SolvabilityDiagnostics(
                                                    failure_code="UPSTREAM_MISSING_HANDLE_COLLIDER",
                                                    stage_index=blk_idx,
                                                    detail=(
                                                        f"Stage {blk_idx} prismatic joint '{obj_uid}.{jname}' remained at "
                                                        f"{cur_val:.4f} m (target [{lo:.4f}, {hi:.4f}] m) and named "
                                                        f"frame '{fname}' lies {gap_m * 1000.0:.1f} mm outside "
                                                        f"'{b1_prim.GetName()}' collision AABB."
                                                    ),
                                                    upstream_remediation_hint=(
                                                        f"In upstream asset USD ({obj_uid}), add positive-scale "
                                                        f"UsdPhysics.CollisionAPI box colliders covering '{fname}' on "
                                                        f"link '{b1_prim.GetName()}'."
                                                    ),
                                                ),
                                                policy,
                                            )
                        continue
                    if not jprim.IsA(UsdPhysics.RevoluteJoint):
                        continue
                    damp_attr = jprim.GetAttribute("drive:angular:physics:damping")
                    damping_val = (
                        float(damp_attr.Get())
                        if damp_attr is not None and damp_attr.HasValue()
                        else 0.0
                    )
                    if damping_val > 2.0:
                        si_equiv = damping_val * (180.0 / math.pi)
                        cur_val = self.live_joint_value(obj_uid, jname)
                        lo, hi = active_indices[dof.dof_index]
                        return apply_remediation_policy(
                            SolvabilityDiagnostics(
                                failure_code="UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT",
                                stage_index=blk_idx,
                                detail=(
                                    f"Stage {blk_idx} joint '{obj_uid}.{jname}' ({jprim.GetName()}) has "
                                    f"drive:angular:physics:damping={damping_val:.2f} N*m*s/deg "
                                    f"(~{si_equiv:.1f} N*m*s/rad in PhysX); observed {cur_val:.4f} rad "
                                    f"vs target band [{lo:.4f}, {hi:.4f}] rad."
                                ),
                                upstream_remediation_hint=(
                                    f"In upstream asset USD ({obj_uid}), author UsdPhysics.DriveAPI "
                                    "angular damping in per-degree units (divide SI N*m*s/rad by "
                                    "180/pi = 57.2958, e.g. 0.15-0.45 N*m*s/deg)."
                                ),
                            ),
                            policy,
                        )
            elif ctype == "manip/default/sr_based_genmanip_range":
                if obj_uid in self.brief.objects and obj_uid not in self.scene.articulation_list:
                    try:
                        aabb = self.live_object_aabb(obj_uid)
                    except KeyError:
                        continue
                    dx = float(aabb.max_xyz[0] - aabb.min_xyz[0])
                    dy = float(aabb.max_xyz[1] - aabb.min_xyz[1])
                    dz = float(aabb.max_xyz[2] - aabb.min_xyz[2])
                    if dx > 0.088 and dy > 0.088 and not self._object_has_grasp_handle_or_overhang(
                        obj_uid, dz, aabb
                    ):
                        return apply_remediation_policy(
                            SolvabilityDiagnostics(
                                failure_code="UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE",
                                stage_index=blk_idx,
                                detail=(
                                    f"Stage {blk_idx} object '{obj_uid}' horizontal footprint "
                                    f"({dx:.3f} m x {dy:.3f} m, z={dz:.3f} m) exceeds lift2 max "
                                    "parallel-jaw gripper aperture 0.088 m."
                                ),
                                upstream_remediation_hint=(
                                    f"Present '{obj_uid}' upright in a holder/tray, add a grasp "
                                    "standoff lip <=0.080 m wide, or keep shortest graspable span <=0.080 m."
                                ),
                            ),
                            policy,
                            object_dims_xyz_m=(dx, dy, dz),
                        )

        return SolvabilityDiagnostics(
            failure_code="STAGE_INCOMPLETE",
            stage_index=blk_idx,
            detail=default_detail,
        )

    def finalize_outputs(self) -> tuple[list[MediaArtifactRef], str | None, dict[str, Any]]:
        """Save keyframes, MP4 progression video, optional .npz trajectory, and summary."""
        self._evaluate_metrics_tick(force=True)
        media_refs: list[MediaArtifactRef] = []
        job_dir = self.output_root / "jobs" / self.job_id

        if self.render_camera is not None:
            final_rgb = self._grab_camera_rgb(accum_renders=3)
            if final_rgb is not None:
                self._media_keyframes.append(("final", "final.png", final_rgb))
                for _ in range(8):
                    self._video_frames.append(final_rgb)

            from PIL import Image

            media_dir = job_dir / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            for label, fname, arr in self._media_keyframes:
                fpath = media_dir / fname
                Image.fromarray(arr, mode="RGB").save(fpath, format="PNG")
                media_refs.append(
                    MediaArtifactRef(
                        label=label,
                        relpath=str(fpath),
                        mime_type="image/png",
                        sha256=file_sha256(fpath),
                    )
                )
            if len(self._video_frames) >= 2:
                try:
                    import shutil
                    import subprocess

                    vid_path = media_dir / "progression.mp4"
                    ffmpeg_bin = shutil.which("ffmpeg")
                    h, w = self._video_frames[0].shape[:2]
                    encoded = False
                    if ffmpeg_bin is not None:
                        cmd = [
                            ffmpeg_bin,
                            "-y",
                            "-f",
                            "rawvideo",
                            "-vcodec",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "-s",
                            f"{w}x{h}",
                            "-r",
                            "15",
                            "-i",
                            "-",
                            "-an",
                            "-vcodec",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            str(vid_path),
                        ]
                        raw_bytes = b"".join(np.ascontiguousarray(f[:, :, :3], dtype=np.uint8).tobytes() for f in self._video_frames)
                        proc = subprocess.run(cmd, input=raw_bytes, capture_output=True, check=False)
                        encoded = proc.returncode == 0 and vid_path.exists()
                    if not encoded:
                        import cv2

                        writer = cv2.VideoWriter(
                            str(vid_path),
                            int(getattr(cv2, "VideoWriter_fourcc")(*"mp4v")),
                            15.0,
                            (w, h),
                        )
                        try:
                            for frame in self._video_frames:
                                writer.write(cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR))
                        finally:
                            writer.release()
                        encoded = vid_path.exists()
                    if encoded:
                        media_refs.append(
                            MediaArtifactRef(
                                label="progression_video",
                                relpath=str(vid_path),
                                mime_type="video/mp4",
                                sha256=file_sha256(vid_path),
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    self.controller_logs.append(f"mp4_export_warning: {exc}")

        artifact_relpath: str | None = None
        final_score = self.live_current_score()
        if (
            self.mode == "trajectory_synthesis"
            and self.snapshots_used == 0
            and final_score >= 0.9999
            and self._traj_proprio
        ):
            ep_folder = self.episode_id or f"{self.brief.package_id}_{self.job_id}"
            ep_dir = self.output_root / "episodes" / ep_folder
            ep_dir.mkdir(parents=True, exist_ok=True)
            npz_path = ep_dir / "episode_trajectory.npz"
            save_dict: dict[str, Any] = {
                "proprio": np.stack(self._traj_proprio, axis=0),
                "action": np.stack(self._traj_action, axis=0),
                "base_target": np.stack(self._traj_base_target, axis=0),
                "torso_lift": np.asarray(self._traj_torso_lift, dtype=np.float32).reshape(-1, 1),
                "timestamps_s": np.asarray(self._traj_timestamps, dtype=np.float64),
                "state_ee_pose": np.stack(self._traj_state_ee_pose, axis=0),
                "action_ee_pose": np.stack(self._traj_action_ee_pose, axis=0),
            }
            if self._traj_camera_step_indices:
                save_dict["camera_step_indices"] = np.asarray(
                    self._traj_camera_step_indices, dtype=np.int64
                )
            for cam_name, cam_frames in self._traj_sensor_frames.items():
                if cam_frames:
                    save_dict[cam_name] = np.stack(cam_frames, axis=0)
            if self._traj_rgb_frames:
                save_dict["rgb_head"] = np.stack(self._traj_rgb_frames, axis=0)
            np.savez_compressed(npz_path, **save_dict)
            artifact_relpath = str(npz_path)

        stage_scores = self.live_stage_scores()
        mm = self.scene.metric_manager
        stages_completed, blocking_stage = compute_sequential_blocking_stage(
            num_stages=len(self.brief.stages),
            cur_union_metric_is_none=(mm.cur_union_metric is None),
            long_stride_idx=int(mm.long_stride_idx),
            final_score=final_score,
        )
        end_objects = {}
        for oid in self.brief.objects:
            if oid == "00000000000000000000000000000000":
                continue
            try:
                p = self.live_object_pose(oid)
                end_objects[oid] = {"xyz": [round(v, 5) for v in p.xyz], "wxyz": [round(v, 5) for v in p.wxyz]}
            except KeyError:
                pass
        end_joints = {}
        for oid, obj_spec in self.brief.objects.items():
            if not obj_spec.dofs or oid not in self.scene.articulation_list:
                continue
            vals = self.live_joint_positions(oid)
            end_joints[oid] = {
                jname: {
                    "value": round(float(vals[dof.dof_index]), 5),
                    "bands": {k: [round(b[0], 5), round(b[1], 5)] for k, b in dof.states.items()},
                }
                for jname, dof in obj_spec.dofs.items()
                if dof.dof_index < len(vals)
            }

        pbd_final = [
            evaluate_scene_pbd_containment(self.scene, spec) for spec in getattr(self, "_pbd_goal_specs", [])
        ]
        summary = {
            "backend": "isaac_genmanip_41",
            "stage_scores": list(stage_scores),
            "stages_completed_sequentially": stages_completed,
            "end_of_episode_all_stages_hold": all(s >= 0.9999 for s in stage_scores) if stage_scores else False,
            "blocking_stage": blocking_stage,
            "objects_final": end_objects,
            "articulation_joints_final": end_joints,
            "pbd_containment_t0": list(getattr(self, "_pbd_containment_t0", [])),
            "pbd_containment_final": pbd_final,
            "controller_logs": list(self.controller_logs[-40:]),
        }
        return media_refs, artifact_relpath, summary
