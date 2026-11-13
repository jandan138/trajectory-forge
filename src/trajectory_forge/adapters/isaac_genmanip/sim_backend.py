from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from omni.isaac.core.utils.types import ArticulationAction
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

from trajectory_forge.core.contracts import (
    ALLOWED_EXECUTION_MODES,
    ContractValidationError,
    EmbodimentCard,
    FirewallViolationError,
    MediaArtifactRef,
    ScenarioIntakeBrief,
    SnapshotHandle,
    StageProgressEntry,
)
from trajectory_forge.sdk.context import (
    AABB3D,
    JointAxisInfo,
    TactileReading,
    WrenchReading,
)
from trajectory_forge.sdk.math3d import Pose3D, normalize_quat_wxyz, slerp_wxyz


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


def fix_lift2_finger_collider_offsets(stage: Usd.Stage, robot_prim_path: str) -> int:
    """Fix R5a finger collision restOffset >= contactOffset PhysX warnings and friction in-memory."""
    updated = 0
    for prim in stage.Traverse():
        pstr = str(prim.GetPath())
        if not pstr.startswith(robot_prim_path):
            continue
        if ("/link7/" in pstr or "/link8/" in pstr) and prim.HasAPI(UsdPhysics.CollisionAPI):
            c_attr = prim.GetAttribute("physxCollision:contactOffset")
            r_attr = prim.GetAttribute("physxCollision:restOffset")
            p_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            c_val = c_attr.Get() if c_attr and c_attr.HasValue() else None
            r_val = r_attr.Get() if r_attr and r_attr.HasValue() else None
            if c_val is None or r_val is None or float(c_val) <= float(r_val) or float(c_val) <= 0.0:
                p_api.CreateContactOffsetAttr(0.004)
                p_api.CreateRestOffsetAttr(0.0005)
                updated += 1
        elif prim.GetName() == "PhysicsMaterial" or prim.HasAPI(UsdPhysics.MaterialAPI):
            mat_api = UsdPhysics.MaterialAPI.Apply(prim)
            s_attr = prim.GetAttribute("physics:staticFriction")
            d_attr = prim.GetAttribute("physics:dynamicFriction")
            s_val = float(s_attr.Get()) if s_attr and s_attr.HasValue() and s_attr.Get() is not None else 0.0
            d_val = float(d_attr.Get()) if d_attr and d_attr.HasValue() and d_attr.Get() is not None else 0.0
            if s_val < 0.8:
                mat_api.CreateStaticFrictionAttr(0.8)
            if d_val < 0.8:
                mat_api.CreateDynamicFrictionAttr(0.8)
            p_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
            p_mat_api.CreateImprovePatchFrictionAttr(True)
    return updated


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
                for child in Usd.PrimRange(root_prim):
                    if not (child.IsA(UsdPhysics.RevoluteJoint) or child.IsA(UsdPhysics.PrismaticJoint)):
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

        # Build standalone per-stage leaf evaluators so any stage can be queried independently at any time.
        from genmanip.core.metrics.metrics_manager import MetricsManager
        from genmanip.core.metrics.utils import MetricFactory
        import genmanip.extensions.metrics  # noqa: F401

        raw_goal = task_data.get("goal", [])
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
        self._traj_rgb_frames: list[NDArray[Any]] = []
        self._media_keyframes: list[tuple[str, str, NDArray[Any]]] = []
        self._video_frames: list[NDArray[Any]] = []

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
        pose_7d = np.concatenate([np.asarray(p_arm, dtype=float), np.asarray(q_arm, dtype=float)])
        cur_seed = (
            np.asarray(seed_6dof, dtype=float)
            if seed_6dof is not None
            else np.asarray(self._target_16dof[sl], dtype=float)
        )
        candidates = [
            cur_seed,
            np.zeros(6, dtype=float),
            np.array([0.0, 0.8, 1.0, 0.0, 0.5, 0.0], dtype=float),
            np.array([0.3 if arm == "left" else -0.3, 1.0, 1.1, 0.0, 0.4, 0.0], dtype=float),
        ]
        for seed in candidates:
            sol = planner.ik_single(pose_7d, seed)
            if sol is not None:
                return np.asarray(sol[:6], dtype=float)
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
        self._full_dof_targets[self.emb.default_dof_indices] = self._target_16dof
        self._full_dof_targets[self.emb.body_dof_indices] = [float(self._target_lift)]
        self._full_dof_targets[self.emb.base_dof_indices] = np.asarray(
            self.emb.target_base_pose, dtype=float
        )
        self.emb.robot_view.set_joint_position_targets(self._full_dof_targets.tolist())

    def _grab_camera_rgb(self, accum_renders: int = 1) -> NDArray[Any] | None:
        if self.render_camera is None:
            return None
        from genmanip.utils.usd_utils.camera_utils import get_src

        for _ in range(accum_renders):
            self.scene.world.render()
        raw = None
        if getattr(self.render_camera, "_rgb_annotator", None) is not None:
            ann = self.render_camera._rgb_annotator.get_data()
            if isinstance(ann, np.ndarray) and ann.size > 0:
                raw = ann[:, :, :3]
        if raw is None:
            raw = get_src(self.render_camera, "rgb")
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

    def _step_one_tick(self) -> None:
        if self.steps_executed + 1 > self.max_steps:
            raise ContractValidationError(f"Controller exceeded max_steps={self.max_steps}")
        self._apply_robot_targets()
        need_render_step = self.render_camera is not None and (
            self.mode == "trajectory_synthesis" and self.steps_executed % 8 == 0
        )
        self.scene.world.step(render=need_render_step)
        self.steps_executed += 1

        if self.mode == "trajectory_synthesis":
            self._traj_proprio.append(self._cur_16dof().astype(np.float32))
            self._traj_action.append(self._target_16dof.astype(np.float32).copy())
            self._traj_base_target.append(np.asarray(self.emb.target_base_pose, dtype=np.float32).copy())
            self._traj_torso_lift.append(float(self._target_lift))
            self._traj_timestamps.append(float(self.steps_executed * self.physics_dt))
            if self.render_camera is not None and (self.steps_executed % 8 == 0):
                frame = self._grab_camera_rgb(accum_renders=1)
                if frame is not None:
                    self._traj_rgb_frames.append(frame)
                    self._video_frames.append(frame)

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

    def diagnose_blocking_stage(self, state_summary: Mapping[str, Any]) -> tuple[str, str, str]:
        """Classify upstream physical blockers at the first incomplete stage."""
        blk = state_summary.get("blocking_stage")
        final_score = float(self.live_current_score())
        default_detail = (
            f"Controller finished with final_score={final_score:.4f}; "
            f"blocking_stage={blk}; stage_scores={state_summary.get('stage_scores')}"
        )
        if blk is None or not (0 <= int(blk) < len(self.brief.stages)):
            return ("STAGE_INCOMPLETE", default_detail, "")

        stage_spec = self.brief.stages[int(blk)]
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
                    if jprim is None or not jprim.IsA(UsdPhysics.RevoluteJoint):
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
                        return (
                            "UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT",
                            (
                                f"Stage {blk} joint '{obj_uid}.{jname}' ({jprim.GetName()}) has "
                                f"drive:angular:physics:damping={damping_val:.2f} N*m*s/deg "
                                f"(~{si_equiv:.1f} N*m*s/rad in PhysX); observed {cur_val:.4f} rad "
                                f"vs target band [{lo:.4f}, {hi:.4f}] rad."
                            ),
                            (
                                f"In upstream asset USD ({obj_uid}), author UsdPhysics.DriveAPI "
                                "angular damping in per-degree units (divide SI N*m*s/rad by "
                                "180/pi = 57.2958, e.g. 0.15-0.45 N*m*s/deg matching IngunMA2111Fixture)."
                            ),
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
                    if dx > 0.088 and dy > 0.088:
                        return (
                            "UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE",
                            (
                                f"Stage {blk} object '{obj_uid}' horizontal footprint "
                                f"({dx:.3f} m x {dy:.3f} m, z={dz:.3f} m) exceeds lift2 max "
                                "parallel-jaw gripper aperture 0.088 m."
                            ),
                            (
                                f"Present '{obj_uid}' upright in a holder/tray, add a grasp "
                                "standoff lip <=0.080 m wide, or keep shortest horizontal span "
                                "<=0.080 m (matching pcba_carrier_board)."
                            ),
                        )

        return ("STAGE_INCOMPLETE", default_detail, "")

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
            ep_dir = self.output_root / "episodes" / f"{self.brief.package_id}_{self.job_id}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            npz_path = ep_dir / "episode_trajectory.npz"
            save_dict: dict[str, Any] = {
                "proprio": np.stack(self._traj_proprio, axis=0),
                "action": np.stack(self._traj_action, axis=0),
                "base_target": np.stack(self._traj_base_target, axis=0),
                "torso_lift": np.asarray(self._traj_torso_lift, dtype=np.float32).reshape(-1, 1),
                "timestamps_s": np.asarray(self._traj_timestamps, dtype=np.float64),
            }
            if self._traj_rgb_frames:
                save_dict["rgb_head"] = np.stack(self._traj_rgb_frames, axis=0)
            np.savez_compressed(npz_path, **save_dict)
            artifact_relpath = str(npz_path)

        stage_scores = self.live_stage_scores()
        blocking_stage = next((i for i, s in enumerate(stage_scores) if s < 0.9999), None)
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

        summary = {
            "backend": "isaac_genmanip_41",
            "stage_scores": list(stage_scores),
            "end_of_episode_all_stages_hold": all(s >= 0.9999 for s in stage_scores) if stage_scores else False,
            "blocking_stage": blocking_stage,
            "objects_final": end_objects,
            "articulation_joints_final": end_joints,
            "controller_logs": list(self.controller_logs[-40:]),
        }
        return media_refs, artifact_relpath, summary
