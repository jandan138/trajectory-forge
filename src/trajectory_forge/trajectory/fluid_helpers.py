"""Pure-Python fluid particle geometry, cavity containment, and cadence helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from trajectory_forge.sdk.math3d import (
    Pose3D,
    Vec3,
    normalize_quat_wxyz,
    rotate_vec_by_quat_wxyz,
)


def compute_target_stride(physics_dt: float, target_hz: float = 15.0) -> int:
    """Return integer physics ticks per observation step for a fixed target cadence (e.g. 15 Hz)."""
    dt = max(float(physics_dt), 1e-6)
    hz = max(float(target_hz), 1e-6)
    return max(1, int(round((1.0 / hz) / dt)))


def scale_ticks_for_physics_dt(base_ticks_60hz: int, physics_dt: float) -> int:
    """Scale a 60 Hz baseline tick count to preserve wall-clock duration at arbitrary ``physics_dt``."""
    dt = max(float(physics_dt), 1e-6)
    scale = (1.0 / 60.0) / dt
    return max(1, int(round(float(base_ticks_60hz) * scale)))


def upright_tilt_deg(quat_wxyz: Sequence[float]) -> float:
    """Return tilt angle in degrees between the local +Z axis and world +Z axis."""
    w, x, y, z = normalize_quat_wxyz(
        (float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))
    )
    r22 = 1.0 - 2.0 * (x * x + y * y)
    r22 = max(-1.0, min(1.0, r22))
    return float(math.degrees(math.acos(r22)))


def transform_points_local_to_world(
    points_local: Sequence[Sequence[float]],
    world_pose: Pose3D,
) -> tuple[Vec3, ...]:
    """Transform local-space 3D particle coordinates into world frame using ``Pose3D``."""
    tx, ty, tz = world_pose.xyz
    out: list[Vec3] = []
    for pt in points_local:
        rx, ry, rz = rotate_vec_by_quat_wxyz(
            world_pose.wxyz,
            (float(pt[0]), float(pt[1]), float(pt[2])),
        )
        out.append((float(tx + rx), float(ty + ry), float(tz + rz)))
    return tuple(out)


def transform_points_world_to_local(
    points_world: Sequence[Sequence[float]],
    world_pose: Pose3D,
) -> tuple[Vec3, ...]:
    """Transform world-space 3D points into a vessel's local frame."""
    tx, ty, tz = world_pose.xyz
    w, x, y, z = normalize_quat_wxyz(world_pose.wxyz)
    q_inv = (w, -x, -y, -z)
    out: list[Vec3] = []
    for pt in points_world:
        dx = float(pt[0]) - tx
        dy = float(pt[1]) - ty
        dz = float(pt[2]) - tz
        lx, ly, lz = rotate_vec_by_quat_wxyz(q_inv, (dx, dy, dz))
        out.append((float(lx), float(ly), float(lz)))
    return tuple(out)


def points_in_vessel_cavity_mask(
    points_world: Sequence[Sequence[float]],
    vessel_pose: Pose3D,
    cavity_spec: Mapping[str, Any],
) -> tuple[bool, ...]:
    """Return boolean mask indicating which world-space particles lie inside a vessel cavity."""
    if not points_world:
        return ()
    local_pts = transform_points_world_to_local(points_world, vessel_pose)
    shape = str(cavity_spec.get("shape", "cylinder")).strip().lower()
    margin = float(cavity_spec.get("margin_m", 0.0))
    cx = float(cavity_spec.get("cx", 0.0))
    cy = float(cavity_spec.get("cy", 0.0))
    z_min = float(cavity_spec.get("z_min", 0.0)) - margin
    z_max = float(cavity_spec.get("z_max", 0.2)) + margin
    z_span = max(z_max - z_min, 1e-6)

    mask: list[bool] = []
    if shape == "box":
        x_min = float(cavity_spec["x_min"]) - margin
        x_max = float(cavity_spec["x_max"]) + margin
        y_min = float(cavity_spec["y_min"]) - margin
        y_max = float(cavity_spec["y_max"]) + margin
        for lx, ly, lz in local_pts:
            mask.append(x_min <= lx <= x_max and y_min <= ly <= y_max and z_min <= lz <= z_max)
        return tuple(mask)

    if shape == "elliptical_cylinder":
        rx_bot = float(cavity_spec.get("rx_bot", cavity_spec.get("rx", 0.05))) + margin
        rx_top = float(cavity_spec.get("rx_top", cavity_spec.get("rx", rx_bot - margin))) + margin
        ry_bot = float(cavity_spec.get("ry_bot", cavity_spec.get("ry", 0.05))) + margin
        ry_top = float(cavity_spec.get("ry_top", cavity_spec.get("ry", ry_bot - margin))) + margin
        for lx, ly, lz in local_pts:
            if not (z_min <= lz <= z_max):
                mask.append(False)
                continue
            t = max(0.0, min(1.0, (lz - z_min) / z_span))
            rx_allow = max(1e-6, rx_bot + t * (rx_top - rx_bot))
            ry_allow = max(1e-6, ry_bot + t * (ry_top - ry_bot))
            norm_sq = ((lx - cx) / rx_allow) ** 2 + ((ly - cy) / ry_allow) ** 2
            mask.append(norm_sq <= 1.0)
        return tuple(mask)

    # Default: tapered circular cylinder
    r_bot = float(cavity_spec.get("r_bot", cavity_spec.get("radius", 0.04))) + margin
    r_top = float(cavity_spec.get("r_top", cavity_spec.get("radius", r_bot - margin))) + margin
    for lx, ly, lz in local_pts:
        if not (z_min <= lz <= z_max):
            mask.append(False)
            continue
        t = max(0.0, min(1.0, (lz - z_min) / z_span))
        r_allow = max(1e-6, r_bot + t * (r_top - r_bot))
        mask.append(math.hypot(lx - cx, ly - cy) <= r_allow)
    return tuple(mask)


def evaluate_pbd_containment(
    points_world: Sequence[Sequence[float]],
    *,
    target_pose: Pose3D,
    target_cavity: Mapping[str, Any],
    allowed_vessels: Sequence[tuple[Pose3D, Mapping[str, Any]]] = (),
) -> dict[str, float | int]:
    """Compute target containment and external spill statistics for a PBD fluid particle cloud."""
    total = len(points_world)
    if total == 0:
        return {
            "total": 0,
            "target_count": 0,
            "target_ratio": 0.0,
            "allowed_count": 0,
            "spill_count": 0,
            "spill_ratio": 1.0,
        }
    target_mask = points_in_vessel_cavity_mask(points_world, target_pose, target_cavity)
    target_count = sum(1 for inside in target_mask if inside)

    if allowed_vessels:
        allowed_flags = [False] * total
        for pose_i, cav_i in allowed_vessels:
            m_i = points_in_vessel_cavity_mask(points_world, pose_i, cav_i)
            for idx, inside in enumerate(m_i):
                if inside:
                    allowed_flags[idx] = True
        allowed_count = sum(1 for inside in allowed_flags if inside)
    else:
        allowed_count = target_count

    spill_count = total - allowed_count
    return {
        "total": int(total),
        "target_count": int(target_count),
        "target_ratio": float(target_count / total),
        "allowed_count": int(allowed_count),
        "spill_count": int(spill_count),
        "spill_ratio": float(spill_count / total),
    }


def compute_aabb_containment_ratio(
    points_world: Sequence[Sequence[float]],
    *,
    box_min: Sequence[float],
    box_max: Sequence[float],
) -> float:
    """Return fraction of 3D points lying inside the closed AABB ``[box_min, box_max]``."""
    total = len(points_world)
    if total == 0:
        return 0.0
    bx0, by0, bz0 = float(box_min[0]), float(box_min[1]), float(box_min[2])
    bx1, by1, bz1 = float(box_max[0]), float(box_max[1]), float(box_max[2])
    inside = 0
    for pt in points_world:
        px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
        if bx0 <= px <= bx1 and by0 <= py <= by1 and bz0 <= pz <= bz1:
            inside += 1
    return float(inside / total)


def compute_vessel_fill_bounds(
    box_min: Sequence[float],
    box_max: Sequence[float],
    *,
    top_margin_m: float = 0.015,
    xy_shrink_m: float = 0.004,
) -> tuple[Vec3, Vec3]:
    """Derive interior vessel containment AABB with margin offsets in meters."""
    lo_x = float(box_min[0]) + xy_shrink_m
    hi_x = float(box_max[0]) - xy_shrink_m
    lo_y = float(box_min[1]) + xy_shrink_m
    hi_y = float(box_max[1]) - xy_shrink_m
    if hi_x < lo_x:
        mid_x = 0.5 * (float(box_min[0]) + float(box_max[0]))
        lo_x, hi_x = mid_x, mid_x
    if hi_y < lo_y:
        mid_y = 0.5 * (float(box_min[1]) + float(box_max[1]))
        lo_y, hi_y = mid_y, mid_y
    lo_z = float(box_min[2])
    hi_z = max(lo_z, float(box_max[2]) + top_margin_m)
    return (lo_x, lo_y, lo_z), (hi_x, hi_y, hi_z)
