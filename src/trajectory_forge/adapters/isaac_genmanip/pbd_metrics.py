"""GenManip custom 3D PhysX PBD containment & Isosurface material helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trajectory_forge.sdk.math3d import Pose3D, normalize_quat_wxyz
from trajectory_forge.trajectory.fluid_helpers import (
    evaluate_pbd_containment,
    transform_points_local_to_world,
    upright_tilt_deg,
)

METRIC_TYPE_PBD_CONTAINMENT = "manip/default/sr_based_pbd_containment"


def extract_pbd_goal_specs(goal: Any) -> list[dict[str, Any]]:
    """Collect all ``sr_based_pbd_containment`` sub-goal dicts from a nested GenManip goal tree."""
    found: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, Mapping):
            if node.get("type") == METRIC_TYPE_PBD_CONTAINMENT:
                sub = node.get("sub_goal_setting")
                if isinstance(sub, Mapping):
                    found.append(dict(sub))
                else:
                    found.append(dict(node))
            elif isinstance(node.get("sub_goal_setting"), Mapping):
                _walk(node["sub_goal_setting"])

    _walk(goal)
    return found


def resolve_scene_entity_pose(scene: Any, uid: str) -> Pose3D | None:
    """Resolve live world ``Pose3D`` for an object, articulation, or articulation part UID."""
    entity = None
    art_map = getattr(scene, "articulation_list", None) or {}
    obj_map = getattr(scene, "object_list", None) or {}
    part_map = getattr(scene, "articulation_part_list", None) or {}
    if uid in art_map:
        entity = art_map[uid]
    elif uid in obj_map:
        entity = obj_map[uid]
    elif uid in part_map:
        entity = part_map[uid]
    if entity is None or not hasattr(entity, "get_world_pose"):
        return None
    pos, ori = entity.get_world_pose()
    return Pose3D(
        xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
        wxyz=normalize_quat_wxyz(
            (float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]))
        ),
    )


def read_stage_pbd_world_points(
    stage: Any,
    particle_prim_path: str | None = None,
) -> tuple[str | None, tuple[tuple[float, float, float], ...]]:
    """Read live 3D world-space coordinates for a PhysX PBD ``UsdGeom.Points`` prim."""
    from pxr import PhysxSchema, UsdGeom

    target_prim = None
    if particle_prim_path:
        cand = stage.GetPrimAtPath(str(particle_prim_path))
        if cand and cand.IsValid() and cand.IsA(UsdGeom.Points):
            target_prim = cand
    if target_prim is None:
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Points) and prim.HasAPI(PhysxSchema.PhysxParticleSetAPI):
                target_prim = prim
                break
    if target_prim is None:
        return None, ()

    pts_raw = UsdGeom.Points(target_prim).GetPointsAttr().Get()
    if not pts_raw:
        return str(target_prim.GetPath()), ()

    xform_cache = UsdGeom.XformCache()
    gf_mat = xform_cache.GetLocalToWorldTransform(target_prim)
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
    return str(target_prim.GetPath()), pts_world


def evaluate_scene_pbd_containment(
    scene: Any,
    setting: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate 3D PBD containment & external spill against a live GenManip scene."""
    stage = scene.world.stage
    prim_path = setting.get("particle_prim_path")
    found_prim, pts_world = read_stage_pbd_world_points(
        stage,
        particle_prim_path=str(prim_path) if prim_path else None,
    )
    target_uid = str(setting["target_vessel_uid"])
    target_pose = resolve_scene_entity_pose(scene, target_uid)
    target_cavity = setting.get("target_cavity") or {}
    if target_pose is None or not pts_world:
        return {
            "passed": False,
            "particle_prim_path": found_prim,
            "target_vessel_uid": target_uid,
            "total": len(pts_world),
            "target_count": 0,
            "target_ratio": 0.0,
            "allowed_count": 0,
            "spill_count": len(pts_world),
            "spill_ratio": 1.0,
            "target_tilt_deg": None,
        }

    allowed_pairs: list[tuple[Pose3D, Mapping[str, Any]]] = []
    raw_allowed = setting.get("allowed_vessels") or []
    if isinstance(raw_allowed, Sequence) and not isinstance(raw_allowed, (str, bytes)):
        for entry in raw_allowed:
            if not isinstance(entry, Mapping):
                continue
            uid_i = str(entry.get("vessel_uid") or "")
            cav_i = entry.get("cavity")
            if not uid_i or not isinstance(cav_i, Mapping):
                continue
            pose_i = resolve_scene_entity_pose(scene, uid_i)
            if pose_i is not None:
                allowed_pairs.append((pose_i, cav_i))

    stats = evaluate_pbd_containment(
        pts_world,
        target_pose=target_pose,
        target_cavity=target_cavity,
        allowed_vessels=allowed_pairs,
    )
    min_particles = int(setting.get("min_particles", 1))
    min_ratio = float(setting.get("min_containment_ratio", 0.90))
    max_spill = float(setting.get("max_spill_ratio", 0.05))
    max_tilt = setting.get("max_tilt_deg")
    tilt_deg = upright_tilt_deg(target_pose.wxyz)

    passed = (
        int(stats["total"]) >= min_particles
        and float(stats["target_ratio"]) >= min_ratio
        and float(stats["spill_ratio"]) <= max_spill
    )
    if max_tilt is not None:
        passed = passed and (tilt_deg <= float(max_tilt))

    return {
        "passed": bool(passed),
        "particle_prim_path": found_prim,
        "target_vessel_uid": target_uid,
        "total": int(stats["total"]),
        "target_count": int(stats["target_count"]),
        "target_ratio": round(float(stats["target_ratio"]), 5),
        "allowed_count": int(stats["allowed_count"]),
        "spill_count": int(stats["spill_count"]),
        "spill_ratio": round(float(stats["spill_ratio"]), 5),
        "target_tilt_deg": round(float(tilt_deg), 2),
        "min_containment_ratio": min_ratio,
        "max_spill_ratio": max_spill,
    }


def sync_stage_pbd_isosurface_materials(
    stage: Any,
    *,
    log_sink: list[str] | None = None,
    logged_paths: set[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Bind glossy visual materials to runtime-generated PhysX PBD Isosurface meshes
    and keep raw particle point clouds invisible when Isosurface rendering is enabled."""
    from pxr import PhysxSchema, UsdGeom, UsdShade

    iso_stats: dict[str, tuple[int, int]] = {}
    if stage is None:
        return iso_stats

    # 1. Legacy explicit bindings (e.g. dual-phase espresso/milk stages)
    legacy_specs = (
        ("/World/PBD_EspressoSystem/Isosurface", "/World/PBD_Espresso", "/World/Looks/PBD_EspressoMat"),
        ("/World/PBD_MilkSystem/Isosurface", "/World/PBD_SteamedMilk", "/World/Looks/PBD_SteamedMilkMat"),
    )
    for iso_path, pts_path, mat_path in legacy_specs:
        pts_prim = stage.GetPrimAtPath(pts_path)
        if pts_prim and pts_prim.IsValid():
            UsdGeom.Imageable(pts_prim).CreateVisibilityAttr("invisible")
        iso_prim = stage.GetPrimAtPath(iso_path)
        mat_prim = stage.GetPrimAtPath(mat_path)
        if iso_prim and iso_prim.IsValid():
            UsdGeom.Imageable(iso_prim).CreateVisibilityAttr("inherited")
            if mat_prim and mat_prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(iso_prim).Bind(
                    UsdShade.Material(mat_prim),
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )

    # 2. Dynamic discovery for any PhysxParticleSystem with Isosurface enabled
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Points) and prim.HasAPI(PhysxSchema.PhysxParticleSetAPI):
            use_iso = prim.GetAttribute("fluid:useIsosurface")
            if use_iso and use_iso.HasValue() and bool(use_iso.Get()):
                UsdGeom.Imageable(prim).CreateVisibilityAttr("invisible")
            continue

        if prim.GetTypeName() != "PhysxParticleSystem":
            continue
        iso_enabled = prim.GetAttribute("physxParticleIsosurface:isosurfaceEnabled")
        if not (iso_enabled and iso_enabled.HasValue() and bool(iso_enabled.Get())):
            continue
        iso_path = f"{prim.GetPath()}/Isosurface"
        iso_prim = stage.GetPrimAtPath(iso_path)
        if not (iso_prim and iso_prim.IsValid()):
            continue

        UsdGeom.Imageable(iso_prim).CreateVisibilityAttr("inherited")
        mat_prim = None
        mat_attr = prim.GetAttribute("fluid:visualMaterialPath")
        if mat_attr and mat_attr.HasValue() and mat_attr.Get():
            cand = stage.GetPrimAtPath(str(mat_attr.Get()))
            if cand and cand.IsValid():
                mat_prim = cand
        if mat_prim is None:
            binding_api = UsdShade.MaterialBindingAPI(prim)
            direct_mat = binding_api.GetDirectBinding().GetMaterial()
            if direct_mat and direct_mat.GetPrim().IsValid():
                mat_prim = direct_mat.GetPrim()
        if mat_prim is not None:
            UsdShade.MaterialBindingAPI.Apply(iso_prim).Bind(
                UsdShade.Material(mat_prim),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )

        mesh_pts = UsdGeom.Mesh(iso_prim).GetPointsAttr().Get()
        mesh_fvc = UsdGeom.Mesh(iso_prim).GetFaceVertexCountsAttr().Get()
        nv = len(mesh_pts) if mesh_pts else 0
        nf = len(mesh_fvc) if mesh_fvc else 0
        iso_stats[iso_path] = (nv, nf)
        if nv > 0 and log_sink is not None:
            if logged_paths is None or iso_path not in logged_paths:
                if logged_paths is not None:
                    logged_paths.add(iso_path)
                log_sink.append(f"[PBD_ISOSURFACE_LIVE] {iso_path}: verts={nv}, faces={nf}")

    return iso_stats


def register_pbd_containment_metric() -> None:
    """Register ``manip/default/sr_based_pbd_containment`` with GenManip ``MetricFactory``."""
    from genmanip.core.metrics.base import BaseMetric
    from genmanip.core.metrics.utils import MetricFactory

    registry = getattr(MetricFactory, "_registry", None)
    if isinstance(registry, dict) and METRIC_TYPE_PBD_CONTAINMENT in registry:
        return

    @MetricFactory.register(METRIC_TYPE_PBD_CONTAINMENT)
    class SRBasedPBDContainment(BaseMetric):
        def __init__(
            self,
            skip_steps: int = 1,
            succ_cnts: int = 0,
            sub_goal_setting: dict[str, Any] = {},
            never_reset: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                skip_steps=skip_steps,
                succ_cnts=succ_cnts,
                sub_goal_setting=sub_goal_setting,
                never_reset=never_reset,
                **kwargs,
            )
            self.last_eval: dict[str, Any] = {}

        def check_status(self, scene: Any) -> bool:
            res = evaluate_scene_pbd_containment(scene, self.goal_setting)
            self.last_eval = res
            return bool(res["passed"])
