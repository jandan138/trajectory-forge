"""Shared v0.3 proxy loader, task metrics evaluator, and firewall-pure NPZ persistence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from trajectory_forge.adapters.legacy.deformable_workers.common import (
    apply_low_priority_process_policy,
    compute_damage_score,
    compute_edge_stretch_from_points,
    compute_tet_strains_from_F,
    compute_tet_strains_from_points,
    jitter_for_seed,
    save_episode_bundle,
)
from trajectory_forge.core.contracts import ScenarioIntakeBrief
from trajectory_forge.intake.deformable_v02_intake import load_deformable_v02_package
from trajectory_forge.skills.deformable_primitives_v03 import make_v03_controller


V03_TASKS: tuple[str, ...] = (
    "coffee_croissant_tray_to_plate",
    "coffee_strawberry_tart_topping",
    "office_waffle_towel_corner_fold",
    "office_dust_cover_peel_retrieve",
    "smt_ribbon_cable_clip_route",
    "smt_esd_silicone_boot_press_fit",
    "lab_silicone_tube_manifold_insert",
    "lab_rubber_bulb_squeeze_seat",
)


def orient_tets_positive(pts: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Ensure every tetrahedron (a, b, c, d) has positive signed volume."""
    out = np.asarray(tets, dtype=np.int64).copy()
    for idx, (a, b, c, d_i) in enumerate(out):
        det = float(np.linalg.det(np.column_stack((pts[b] - pts[a], pts[c] - pts[a], pts[d_i] - pts[a]))))
        if det < 0.0:
            out[idx] = (a, c, b, d_i)
    return out


def load_v03_proxy_and_brief(
    package_dir: Path,
    task_id: str,
    mode: str,
    skill_params_override: Mapping[str, Any] | None = None,
) -> tuple[ScenarioIntakeBrief, Any, dict[str, np.ndarray], dict[str, Any]]:
    """Load ScenarioIntakeBrief, controller primitive, localized physics_proxy.npz, and m1_manifest."""
    root = Path(package_dir)
    if (root / "scenario_package" / "manifest.yaml").is_file():
        root = root / "scenario_package"
    brief = load_deformable_v02_package(root)
    def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
    asset_dir = root / "assets" / def_obj.runtime_uid

    with np.load(asset_dir / "physics_proxy.npz", allow_pickle=False) as z:
        proxy = {k: np.asarray(z[k]) for k in z.files}
    m1_manifest = json.loads((asset_dir / "m1_manifest.json").read_text(encoding="utf-8"))
    ctrl_mode = "gentle" if mode == "replay" else mode
    ctrl = make_v03_controller(
        brief,
        task_id,
        mode=ctrl_mode,
        params_override=skill_params_override,
    )
    return brief, ctrl, proxy, m1_manifest


def check_target_obstacle_or_reachability(
    brief: ScenarioIntakeBrief,
    task_id: str,
) -> dict[str, Any]:
    """Extract live fixture poses and check geometric obstacle / workspace reachability."""
    base_xy = np.asarray(brief.robot_spawn.xyz[:2], dtype=np.float64)
    tgt_obj = (
        brief.objects.get("tartlet_pastry_cup")
        or brief.objects.get("serving_plate")
        or brief.objects.get("delivery_caddy")
        or brief.objects.get("smt_u_clip_block")
        or brief.objects.get("pcba_header_boss")
        or brief.objects.get("manifold_fork_saddle")
        or brief.objects.get("bulb_collar_holder")
        or brief.objects.get("smooth_folding_mat")
    )
    tgt_xy = np.asarray(tgt_obj.initial_pose.xyz[:2], dtype=np.float64) if tgt_obj else np.zeros(2, dtype=np.float64)
    tgt_reach_dist = float(np.linalg.norm(tgt_xy - base_xy))
    target_unreachable = bool(tgt_reach_dist > 0.86 or float(tgt_xy[0]) > 0.45)

    obstacle_collision = False
    obstacle_dist_m = None
    if task_id.startswith("smt_") and "ESDTray" in brief.objects:
        esd_xy = np.asarray(brief.objects["ESDTray"].initial_pose.xyz[:2], dtype=np.float64)
        obstacle_dist_m = float(np.linalg.norm(esd_xy - tgt_xy))
        if obstacle_dist_m < 0.14:
            obstacle_collision = True

    return {
        "target_xy": tgt_xy.tolist(),
        "target_reach_dist_m": round(tgt_reach_dist, 4),
        "target_unreachable": target_unreachable,
        "obstacle_collision": obstacle_collision,
        "obstacle_dist_m": round(obstacle_dist_m, 4) if obstacle_dist_m is not None else None,
    }


def evaluate_v03_task_outcome(
    *,
    task_id: str,
    engine_id: str,
    mode: str,
    zero_friction: bool,
    brief: ScenarioIntakeBrief,
    rest_world: np.ndarray,
    pts_hist: np.ndarray,
    elems_or_edges: np.ndarray,
    tac_hist: np.ndarray,
    f_hist: np.ndarray | None = None,
    rigid_target_pose_hist: np.ndarray | None = None,
    pad_touches_obstacle: bool = False,
) -> tuple[dict[str, Any], float]:
    """Compute continuum strain, damage, and task-specific geometric pass/fail across all 8 v0.3 tasks."""
    def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
    def_spec = def_obj.deformable
    assert def_spec is not None
    family = str(def_spec.structural_family)
    finite_ok = bool(np.all(np.isfinite(pts_hist)) and np.all(np.isfinite(rest_world)))

    peak_polar = 0.0
    peak_naive = 0.0
    peak_edge = 0.0
    max_z = float(rest_world[:, 2].mean()) if len(rest_world) else 0.0

    for k in range(len(pts_hist)):
        pts = pts_hist[k]
        if finite_ok and len(pts):
            max_z = max(max_z, float(pts[:, 2].mean()))
        if not finite_ok:
            continue
        if family in ("soft_solid", "rod_cable"):
            if f_hist is not None and len(f_hist) > k:
                ps, ns = compute_tet_strains_from_F(f_hist[k])
            else:
                ps, ns = compute_tet_strains_from_points(rest_world, pts, elems_or_edges)
            peak_polar = max(peak_polar, ps)
            peak_naive = max(peak_naive, ns)
        else:
            es = compute_edge_stretch_from_points(rest_world, pts, elems_or_edges)
            peak_edge = max(peak_edge, es)

    primary_strain = peak_polar if family in ("soft_solid", "rod_cable") else peak_edge
    peak_force_n = float(np.nanmax(np.abs(tac_hist))) if len(tac_hist) else 0.0
    damage = (
        1.0
        if not finite_ok
        else compute_damage_score(
            primary_strain,
            def_spec.strain_onset_a,
            def_spec.strain_failure_b,
            peak_force_n=peak_force_n,
            max_normal_force_n=def_spec.max_normal_force_n,
        )
    )

    scene_chk = check_target_obstacle_or_reachability(brief, task_id)
    fin_pts = pts_hist[-1] if (finite_ok and len(pts_hist)) else rest_world
    fin_com = fin_pts.mean(axis=0) if finite_ok else np.array([np.nan, np.nan, np.nan])
    tgt_xy = np.asarray(scene_chk["target_xy"], dtype=np.float64)
    final_dist = float(np.linalg.norm(fin_com[:2] - tgt_xy)) if finite_ok else 1.0

    extra: dict[str, Any] = {
        "final_com_z_m": round(float(fin_com[2]), 5) if finite_ok else None,
        "obstacle_collision": bool(scene_chk["obstacle_collision"] or pad_touches_obstacle),
        "target_unreachable": bool(scene_chk["target_unreachable"]),
        "target_reach_dist_m": scene_chk["target_reach_dist_m"],
        "final_x_span_m": None,
        "left_strip_mean_x_m": None,
        "cloth_box_xy_dist_m": None,
        "box_max_z_m": None,
        "box_target_dist_m": None,
        "squeeze_ratio": None,
        "rebound_recovery": None,
    }

    if not finite_ok:
        geom_ok = False
    elif task_id == "coffee_croissant_tray_to_plate":
        geom_ok = bool(final_dist <= 0.075 and max_z >= 0.85 and not extra["target_unreachable"])
    elif task_id == "coffee_strawberry_tart_topping":
        geom_ok = bool(final_dist <= 0.045 and float(fin_com[2]) >= 0.770 and max_z >= 0.85)
    elif task_id == "office_waffle_towel_corner_fold":
        x_span = float(np.quantile(fin_pts[:, 0], 0.95) - np.quantile(fin_pts[:, 0], 0.05))
        left_mask = rest_world[:, 0] < (float(rest_world[:, 0].mean()) - 0.04)
        left_mean_x = float(fin_pts[left_mask, 0].mean()) if np.any(left_mask) else float(fin_pts[:, 0].mean())
        extra["final_x_span_m"] = round(x_span, 5)
        extra["left_strip_mean_x_m"] = round(left_mean_x, 5)
        final_dist = max(0.0, x_span - 0.10)
        geom_ok = bool(x_span <= 0.145 and left_mean_x >= -0.025 and max_z >= 0.775)
    elif task_id == "office_dust_cover_peel_retrieve":
        box_obj = brief.objects.get("toner_cartridge_box")
        caddy_obj = brief.objects.get("delivery_caddy")
        box_init_xy = np.asarray(box_obj.initial_pose.xyz[:2] if box_obj else [-0.14, 0.02], dtype=np.float64)
        caddy_xy = np.asarray(caddy_obj.initial_pose.xyz[:2] if caddy_obj else [0.20, -0.02], dtype=np.float64)
        cloth_box_dist = float(np.linalg.norm(fin_com[:2] - box_init_xy))
        if rigid_target_pose_hist is not None and len(rigid_target_pose_hist):
            box_max_z = float(np.max(rigid_target_pose_hist[:, 2]))
            box_fin_xy = rigid_target_pose_hist[-1, :2]
            box_tgt_dist = float(np.linalg.norm(box_fin_xy - caddy_xy))
        else:
            box_max_z = 0.757
            box_tgt_dist = 1.0
        final_dist = box_tgt_dist
        extra["cloth_box_xy_dist_m"] = round(cloth_box_dist, 5)
        extra["box_max_z_m"] = round(box_max_z, 5)
        extra["box_target_dist_m"] = round(box_tgt_dist, 5)
        geom_ok = bool(cloth_box_dist >= 0.11 and float(fin_com[0]) <= -0.18 and box_max_z >= 0.82 and box_tgt_dist <= 0.065)
    elif task_id == "smt_ribbon_cable_clip_route":
        geom_ok = bool(
            final_dist <= 0.055
            and float(fin_com[2]) <= 0.790
            and max_z >= 0.835
            and not extra["obstacle_collision"]
            and not extra["target_unreachable"]
        )
    elif task_id == "smt_esd_silicone_boot_press_fit":
        geom_ok = bool(final_dist <= 0.045 and max_z >= 0.835 and peak_force_n >= 2.0)
    elif task_id == "lab_silicone_tube_manifold_insert":
        geom_ok = bool(
            final_dist <= 0.055
            and float(fin_com[2]) <= 0.795
            and max_z >= 0.835
            and not extra["target_unreachable"]
        )
    elif task_id == "lab_rubber_bulb_squeeze_seat":
        z_rel = rest_world[:, 2] - float(rest_world[:, 2].min())
        belly = (z_rel >= 0.015) & (z_rel <= 0.048)
        if not np.any(belly):
            belly = np.ones((len(rest_world),), dtype=bool)
        w0 = float(np.max(rest_world[belly, 1]) - np.min(rest_world[belly, 1]))
        win_start = min(len(pts_hist) - 1, 148)
        y_widths = [
            float(np.max(pts_hist[k, belly, 1]) - np.min(pts_hist[k, belly, 1]))
            for k in range(win_start, len(pts_hist))
        ]
        min_w = min(y_widths) if y_widths else w0
        w_fin = float(np.max(fin_pts[belly, 1]) - np.min(fin_pts[belly, 1]))
        squeeze_ratio = (w0 - min_w) / max(w0, 1e-6)
        rebound = w_fin / max(w0, 1e-6)
        extra["squeeze_ratio"] = round(squeeze_ratio, 4)
        extra["rebound_recovery"] = round(rebound, 4)
        geom_ok = bool(
            final_dist <= 0.050
            and max_z >= 0.835
            and squeeze_ratio >= 0.18
            and rebound >= 0.88
        )
    else:
        geom_ok = False

    intact = bool(
        finite_ok
        and geom_ok
        and damage < def_spec.damage_threshold
        and peak_force_n <= def_spec.max_normal_force_n
    )

    metrics = {
        "task_id": task_id,
        "structural_family": family,
        "engine_id": engine_id,
        "mode": mode,
        "zero_friction": zero_friction,
        "finite_ok": finite_ok,
        "max_com_z_m": round(max_z, 5) if finite_ok else None,
        "final_com_xyz_m": [round(float(x), 5) for x in fin_com] if finite_ok else None,
        "final_target_dist_m": round(final_dist, 5),
        "polar_stretch_q95": round(peak_polar, 5) if family in ("soft_solid", "rod_cable") else None,
        "naive_f_minus_i_q95": round(peak_naive, 5) if family in ("soft_solid", "rod_cable") else None,
        "edge_stretch_q95": round(peak_edge, 5) if family in ("cloth_sheet", "shell") else None,
        "peak_normal_force_n": round(peak_force_n, 4),
        "damage_score": round(damage, 5),
        "task_geometric_success": geom_ok,
        "intact": intact,
        **extra,
    }
    return metrics, final_dist
