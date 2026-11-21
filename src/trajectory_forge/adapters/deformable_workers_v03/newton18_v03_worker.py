"""Isolated Newton 1.8.0 (Warp SolverVBD) rollout worker for M3 v0.3 deformable tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (str(_SRC_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trajectory_forge.adapters.deformable_workers.common import (  # noqa: E402
    apply_low_priority_process_policy,
    jitter_for_seed,
    save_episode_bundle,
)
from trajectory_forge.adapters.deformable_workers_v03.common_v03 import (  # noqa: E402
    evaluate_v03_task_outcome,
    load_v03_proxy_and_brief,
    orient_tets_positive,
)
from trajectory_forge.sdk.panda_kinematics import panda_pad_world_poses  # noqa: E402


def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> tuple[float, float, float, float]:
    return (float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0]))


def _pad_particle_normal_force(
    pts: np.ndarray,
    pad_pos: np.ndarray,
    hand_rot: np.ndarray,
    half_extents: tuple[float, float, float],
    e_eff: float,
    area_scale: float,
    contact_r: float = 0.0025,
) -> float:
    if len(pts) == 0:
        return 0.0
    rel = (pts - pad_pos[None, :]) @ hand_rot
    hx, hy, hz = half_extents
    inside_x = np.abs(rel[:, 0]) <= hx + contact_r
    inside_z = np.abs(rel[:, 2]) <= hz + contact_r
    pen_y = (hy + contact_r) - np.abs(rel[:, 1])
    mask = inside_x & inside_z & (pen_y > 0.0)
    if not np.any(mask):
        return 0.0
    mean_pen = float(pen_y[mask].mean())
    frac = float(np.sum(mask)) / max(float(len(pts)) * 0.25, 1.0)
    return min(155.0, float(e_eff * area_scale * max(0.0, mean_pen / 0.035) * max(0.25, min(1.5, frac))))


def _approx_tet_volume(pts: np.ndarray, tets: np.ndarray) -> float:
    p0 = pts[tets[:, 0]]
    dm = np.stack((pts[tets[:, 1]] - p0, pts[tets[:, 2]] - p0, pts[tets[:, 3]] - p0), axis=-1)
    return float(np.sum(np.abs(np.linalg.det(dm))) / 6.0)


def run_single_episode(job: dict[str, Any]) -> dict[str, Any]:
    import warp as wp
    import newton

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    pkg_dir = Path(job["package_dir"])
    if (pkg_dir / "scenario_package" / "assets").is_dir():
        pkg_dir = pkg_dir / "scenario_package"
    task_id = str(job["task_id"])
    mode = str(job["mode"])
    seed = int(job.get("seed", 0))
    zero_friction = bool(job.get("zero_friction", False))
    replay_npz = job.get("replay_npz")
    skill_override = job.get("skill_params_override")
    output_dir = Path(job["output_dir"])
    episode_id = str(job["episode_id"])

    brief, ctrl, proxy, m1_manifest = load_v03_proxy_and_brief(
        pkg_dir,
        task_id,
        mode,
        skill_params_override=skill_override,
    )
    replay_actions = None
    if replay_npz:
        with np.load(replay_npz, allow_pickle=False) as z:
            replay_actions = np.asarray(z["action"], dtype=np.float64)

    dx, dy, _ = jitter_for_seed(seed)
    def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
    spec = m1_manifest["source"]
    family = str(spec["structural_family"])
    is_cloth_or_shell = family in ("cloth_sheet", "shell")

    pad_mu = 0.0 if zero_friction else 3.2
    table_mu = 0.45 if task_id == "office_waffle_towel_corner_fold" else (0.14 if family == "cloth_sheet" else 0.55)

    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg_table = newton.ModelBuilder.ShapeConfig(mu=table_mu)
    cfg_tray = newton.ModelBuilder.ShapeConfig(mu=0.65)
    cfg_wall = newton.ModelBuilder.ShapeConfig(mu=0.45)
    cfg_box = newton.ModelBuilder.ShapeConfig(mu=0.0 if zero_friction else 4.8)
    cfg_pad = newton.ModelBuilder.ShapeConfig(mu=pad_mu)

    def _xyz(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
        obj = brief.objects.get(name)
        return tuple(float(v) for v in obj.initial_pose.xyz) if obj else default

    # Single thick non-tunneling support slab (z_top = mat_top when present, else 0.755)
    if "smooth_folding_mat" in brief.objects:
        _, _, zm = _xyz("smooth_folding_mat", (0.0, 0.02, 0.755))
        z_top = zm + 0.002
    else:
        z_top = 0.755
    mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, z_top - 0.035), wp.quat_identity()), hx=1.0, hy=0.45, hz=0.035, cfg=cfg_table)

    if "baking_sheet_tray" in brief.objects:
        x, y, z = _xyz("baking_sheet_tray", (-0.18, -0.02, 0.755))
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.003), wp.quat_identity()), hx=0.095, hy=0.085, hz=0.003, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.0055), wp.quat_identity()), hx=0.084, hy=0.074, hz=0.001, cfg=cfg_tray)
    if "tartlet_pastry_cup" in brief.objects:
        x, y, z = _xyz("tartlet_pastry_cup", (0.18, -0.02, 0.755))
        mb.add_shape_cylinder(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.0025), wp.quat_identity()), radius=0.085, half_height=0.0025, cfg=cfg_tray)
        mb.add_shape_cylinder(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.013), wp.quat_identity()), radius=0.048, half_height=0.008, cfg=cfg_tray)
        mb.add_shape_cylinder(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.021), wp.quat_identity()), radius=0.043, half_height=0.002, cfg=cfg_tray)
    if "garment_folding_board" in brief.objects:
        x, y, z = _xyz("garment_folding_board", (0.0, 0.02, 0.755))
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x + 0.045, y, z + 0.014), wp.quat_identity()), hx=0.095, hy=0.13, hz=0.014, cfg=cfg_tray)

    toner_body_idx = -1
    if "toner_cartridge_box" in brief.objects:
        bx, by, bz = _xyz("toner_cartridge_box", (-0.14, 0.02, 0.757))
        # Dynamic rigid box (coupled with seed jitter dx, dy; high friction resists cloth peel)
        toner_body_idx = mb.add_body(xform=wp.transform(wp.vec3(bx + dx, by + dy, bz), wp.quat_identity()), mass=0.30, is_kinematic=False)
        mb.add_shape_box(body=toner_body_idx, xform=wp.transform(wp.vec3(0.0, 0.0, 0.021), wp.quat_identity()), hx=0.032, hy=0.032, hz=0.021, cfg=cfg_box)
        mb.add_shape_box(body=toner_body_idx, xform=wp.transform(wp.vec3(0.0, 0.0, 0.043), wp.quat_identity()), hx=0.0315, hy=0.0315, hz=0.001, cfg=newton.ModelBuilder.ShapeConfig(mu=0.15))

    if "delivery_caddy" in brief.objects:
        x, y, z = _xyz("delivery_caddy", (0.20, -0.02, 0.757))
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.003), wp.quat_identity()), hx=0.085, hy=0.085, hz=0.003, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y + 0.081, z + 0.010), wp.quat_identity()), hx=0.085, hy=0.004, hz=0.010, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x - 0.081, y, z + 0.010), wp.quat_identity()), hx=0.004, hy=0.085, hz=0.010, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x + 0.081, y, z + 0.010), wp.quat_identity()), hx=0.004, hy=0.085, hz=0.010, cfg=cfg_tray)
    if "cable_carrier_cradle" in brief.objects:
        x, y, z = _xyz("cable_carrier_cradle", (-0.16, -0.03, 0.755))
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.004), wp.quat_identity()), hx=0.105, hy=0.055, hz=0.004, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x - 0.068, y, z + 0.011), wp.quat_identity()), hx=0.012, hy=0.036, hz=0.007, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x + 0.068, y, z + 0.011), wp.quat_identity()), hx=0.012, hy=0.036, hz=0.007, cfg=cfg_tray)
    if "smt_u_clip_block" in brief.objects:
        x, y, z = _xyz("smt_u_clip_block", (0.14, 0.04, 0.755))
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.004), wp.quat_identity()), hx=0.105, hy=0.048, hz=0.004, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y - 0.020, z + 0.014), wp.quat_identity()), hx=0.082, hy=0.005, hz=0.012, cfg=cfg_wall)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y + 0.020, z + 0.014), wp.quat_identity()), hx=0.082, hy=0.005, hz=0.012, cfg=cfg_wall)
    if "pcba_header_boss" in brief.objects:
        x, y, z = _xyz("pcba_header_boss", (0.15, -0.02, 0.755))
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.003), wp.quat_identity()), hx=0.075, hy=0.075, hz=0.003, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.007), wp.quat_identity()), hx=0.026, hy=0.026, hz=0.002, cfg=cfg_tray)
    if "tube_bench_cradle" in brief.objects:
        x, y, z = _xyz("tube_bench_cradle", (-0.17, -0.02, 0.755))
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.004), wp.quat_identity()), hx=0.105, hy=0.055, hz=0.004, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x - 0.068, y, z + 0.011), wp.quat_identity()), hx=0.014, hy=0.032, hz=0.007, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x + 0.068, y, z + 0.011), wp.quat_identity()), hx=0.014, hy=0.032, hz=0.007, cfg=cfg_tray)
    if "manifold_fork_saddle" in brief.objects:
        x, y, z = _xyz("manifold_fork_saddle", (0.16, 0.03, 0.755))
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.004), wp.quat_identity()), hx=0.105, hy=0.055, hz=0.004, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y - 0.022, z + 0.015), wp.quat_identity()), hx=0.085, hy=0.005, hz=0.013, cfg=cfg_wall)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y + 0.022, z + 0.015), wp.quat_identity()), hx=0.085, hy=0.005, hz=0.013, cfg=cfg_wall)
    if "bulb_collar_holder" in brief.objects:
        x, y, z = _xyz("bulb_collar_holder", (0.15, -0.02, 0.755))
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, z + 0.003), wp.quat_identity()), hx=0.065, hy=0.065, hz=0.003, cfg=cfg_tray)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y + 0.034, z + 0.012), wp.quat_identity()), hx=0.018, hy=0.004, hz=0.006, cfg=cfg_wall)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y - 0.034, z + 0.012), wp.quat_identity()), hx=0.018, hy=0.004, hz=0.006, cfg=cfg_wall)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x - 0.034, y, z + 0.012), wp.quat_identity()), hx=0.004, hy=0.018, hz=0.006, cfg=cfg_wall)
        mb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x + 0.034, y, z + 0.012), wp.quat_identity()), hx=0.004, hy=0.018, hz=0.006, cfg=cfg_wall)

    rest_pts = np.asarray(proxy["rest_points"], dtype=np.float64)
    sx, sy, sz = def_obj.initial_pose.xyz
    z_spawn = float(sz) + (0.001 if family == "soft_solid" else 0.0)
    young = float(spec["youngs_modulus_pa"])
    poisson = min(0.42, float(spec["poissons_ratio"]))
    mass_kg = float(spec["mass_kg"])

    if family in ("soft_solid", "rod_cable"):
        tets = orient_tets_positive(rest_pts, np.asarray(proxy["elements"], dtype=np.int64))
        vol = max(_approx_tet_volume(rest_pts, tets), 1e-6)
        density = float(np.clip(mass_kg / vol, 250.0, 1600.0))
        y_vbd = 3.8 * young if family == "soft_solid" else 4.2 * young
        k_mu = y_vbd / (2.0 * (1.0 + poisson))
        k_lam = y_vbd * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        mb.add_soft_mesh(
            pos=wp.vec3(sx + dx, sy + dy, z_spawn),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[(float(v[0]), float(v[1]), float(v[2])) for v in rest_pts],
            indices=[int(i) for tet in tets for i in tet],
            density=density,
            k_mu=k_mu,
            k_lambda=k_lam,
            k_damp=16.0,
            particle_radius=0.0022,
        )
        metrics_elem = tets
    else:
        tris = np.asarray(proxy["triangles"], dtype=np.int64)
        metrics_elem = np.asarray(proxy["edges"], dtype=np.int64)
        is_shell = family == "shell"
        prad = 0.0045 if is_shell else 0.0028
        edge_ke_val = 2200.0 if is_shell else (800.0 if task_id == "office_waffle_towel_corner_fold" else 180.0)
        mb.add_cloth_mesh(
            pos=wp.vec3(sx + dx, sy + dy, z_spawn),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[(float(v[0]), float(v[1]), float(v[2])) for v in rest_pts],
            indices=[int(i) for f in tris for i in f],
            density=0.45 if is_shell else 0.28,
            tri_ke=16000.0 if is_shell else 3600.0,
            tri_ka=11000.0 if is_shell else 2700.0,
            tri_kd=5.0 if is_shell else 3.5,
            edge_ke=edge_ke_val,
            edge_kd=2.2 if is_shell else 0.8,
            particle_radius=prad,
        )

    first_act = (
        replay_actions[0]
        if replay_actions is not None
        else ctrl.step(
            {"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(9), "tactile_normal_force_n": np.zeros(2), "wrist_wrench": np.zeros(6)}
        )
    )
    _, q0_wxyz, lp0, rp0, _ = panda_pad_world_poses(first_act)
    q0_xyzw = wp.quat(*_wxyz_to_xyzw(q0_wxyz))
    pad_l_body = mb.add_body(xform=wp.transform(wp.vec3(*lp0.tolist()), q0_xyzw), mass=0.1, is_kinematic=True)
    pad_r_body = mb.add_body(xform=wp.transform(wp.vec3(*rp0.tolist()), q0_xyzw), mass=0.1, is_kinematic=True)
    pad_l_shape = mb.add_shape_box(body=pad_l_body, xform=wp.transform_identity(), hx=0.018, hy=0.006, hz=0.018, cfg=cfg_pad)
    pad_r_shape = mb.add_shape_box(body=pad_r_body, xform=wp.transform_identity(), hx=0.018, hy=0.006, hz=0.018, cfg=cfg_pad)

    mb.color(include_bending=True)
    model = mb.finalize(device=device)
    model.soft_contact_mu = 0.0 if zero_friction else (0.45 if task_id == "office_dust_cover_peel_retrieve" else 3.5)
    if hasattr(model, "shape_material_mu"):
        mu_host = model.shape_material_mu.numpy().copy()
        mu_host[pad_l_shape] = pad_mu
        mu_host[pad_r_shape] = pad_mu
        model.shape_material_mu.assign(mu_host)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=20,
        particle_enable_self_contact=False,
        integrate_with_external_rigid_solver=False,
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state0 = model.state()
    state1 = model.state()

    rest_world = state0.particle_q.numpy().copy().astype(np.float64)
    substeps = 5
    sub_dt = 0.01 / substeps
    if task_id in ("office_dust_cover_peel_retrieve", "office_waffle_towel_corner_fold"):
        n_set = 150 if task_id == "office_dust_cover_peel_retrieve" else 20
        for _ in range(n_set):
            state0.clear_forces()
            pipeline.collide(state0, contacts)
            solver.step(state0, state1, None, contacts, sub_dt)
            state0, state1 = state1, state0

    total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)
    qpos_hist = np.zeros((total_steps, 9), dtype=np.float64)
    act_hist = np.zeros((total_steps, 9), dtype=np.float64)
    ee_pos_hist = np.zeros((total_steps, 3), dtype=np.float64)
    ee_quat_hist = np.zeros((total_steps, 4), dtype=np.float64)
    tac_hist = np.zeros((total_steps, 2), dtype=np.float64)
    wrench_hist = np.zeros((total_steps, 6), dtype=np.float64)
    pts_hist = np.zeros((total_steps, len(rest_world), 3), dtype=np.float64)
    rigid_pose_hist = np.zeros((total_steps, 7), dtype=np.float64) if toner_body_idx >= 0 else None

    q_cur = np.asarray(first_act, dtype=np.float64).copy()
    prev_lp, prev_rp = lp0.copy(), rp0.copy()
    area_s = 0.0022
    contact_r = 0.0025 if family in ("soft_solid", "rod_cable") else (0.0045 if family == "shell" else 0.0028)
    tactile_e = young if family in ("soft_solid", "rod_cable") else min(young, 28000.0)

    for k in range(total_steps):
        cur_pts = state0.particle_q.numpy().astype(np.float64)
        ee_p, _, lp, rp, hand_r = panda_pad_world_poses(q_cur)
        fn_l = _pad_particle_normal_force(cur_pts, lp, hand_r, (0.018, 0.006, 0.018), tactile_e, area_s, contact_r=contact_r)
        fn_r = _pad_particle_normal_force(cur_pts, rp, hand_r, (0.018, 0.006, 0.018), tactile_e, area_s, contact_r=contact_r)
        if float(ee_p[2]) < 0.761:
            plunge_f = min(150.0, 9000.0 * (0.761 - float(ee_p[2])))
            fn_l = max(fn_l, plunge_f)
            fn_r = max(fn_r, plunge_f)
        if toner_body_idx >= 0 and k >= 105:
            # Stage 2 tactile feedback when gripping dynamic rigid carton (width 64 mm)
            box_p = state0.body_q.numpy()[toner_body_idx, :3]
            if abs(float(ee_p[0] - box_p[0])) < 0.04 and abs(float(ee_p[1] - box_p[1])) < 0.04 and float(q_cur[7]) <= 0.0325:
                pen_box = max(0.0, 0.032 - float(q_cur[7]))
                fb = min(22.0, 4800.0 * pen_box)
                fn_l = max(fn_l, fb)
                fn_r = max(fn_r, fb)
        w_obs = np.array([0.0, 0.0, -(fn_l + fn_r) * 0.5, 0.0, 0.0, 0.0], dtype=np.float64)

        if replay_actions is not None:
            act = replay_actions[k]
        else:
            act = ctrl.step(
                {
                    "step_idx": k,
                    "t_s": k * 0.01,
                    "qpos": q_cur,
                    "ee_pos": ee_p,
                    "tactile_normal_force_n": np.array([fn_l, fn_r], dtype=np.float64),
                    "wrist_wrench": w_obs,
                }
            )

        q_prev_step = q_cur.copy()
        for sub in range(substeps):
            alpha = (sub + 1) / float(substeps)
            q_sub = (1.0 - alpha) * q_prev_step + alpha * act
            _, q_s_wxyz, lp_s, rp_s, _ = panda_pad_world_poses(q_sub)
            q_s_xyzw = _wxyz_to_xyzw(q_s_wxyz)
            v_l = (lp_s - prev_lp) / sub_dt
            v_r = (rp_s - prev_rp) / sub_dt
            prev_lp, prev_rp = lp_s, rp_s

            bq = state0.body_q.numpy().copy()
            bqd = state0.body_qd.numpy().copy()
            bq[pad_l_body, :3] = lp_s
            bq[pad_l_body, 3:] = q_s_xyzw
            bq[pad_r_body, :3] = rp_s
            bq[pad_r_body, 3:] = q_s_xyzw
            bqd[pad_l_body, :3] = v_l
            bqd[pad_l_body, 3:] = 0.0
            bqd[pad_r_body, :3] = v_r
            bqd[pad_r_body, 3:] = 0.0
            state0.body_q.assign(bq)
            state0.body_qd.assign(bqd)

            state0.clear_forces()
            pipeline.collide(state0, contacts)
            solver.step(state0, state1, None, contacts, sub_dt)
            state0, state1 = state1, state0

        q_cur = act.copy()
        cur_pts = state0.particle_q.numpy().astype(np.float64)
        ee_fin, q_fin_wxyz, _, _, _ = panda_pad_world_poses(q_cur)
        qpos_hist[k] = q_cur
        act_hist[k] = act
        ee_pos_hist[k] = ee_fin
        ee_quat_hist[k] = q_fin_wxyz
        tac_hist[k] = np.array([fn_l, fn_r], dtype=np.float64)
        wrench_hist[k] = w_obs
        pts_hist[k] = cur_pts
        if rigid_pose_hist is not None and toner_body_idx >= 0:
            row = state0.body_q.numpy()[toner_body_idx]  # [x,y,z,qx,qy,qz,qw]
            rigid_pose_hist[k, :3] = row[:3]
            rigid_pose_hist[k, 3:] = [row[6], row[3], row[4], row[5]]

    metrics, final_dist = evaluate_v03_task_outcome(
        task_id=task_id,
        engine_id="newton18",
        mode=mode,
        zero_friction=zero_friction,
        brief=brief,
        rest_world=rest_world,
        pts_hist=pts_hist,
        elems_or_edges=metrics_elem,
        tac_hist=tac_hist,
        rigid_target_pose_hist=rigid_pose_hist,
    )
    npz_arrays: dict[str, np.ndarray] = {
        "qpos": qpos_hist,
        "action": act_hist,
        "ee_pos": ee_pos_hist,
        "ee_quat_wxyz": ee_quat_hist,
        "tactile_normal_force_n": tac_hist,
        "wrist_wrench": wrench_hist,
        "deformable_rest_points": rest_world,
        "deformable_points": pts_hist,
        "deformable_elements": metrics_elem,
        "target_dist_m": np.array([final_dist], dtype=np.float64),
        "step_dt": np.array([0.01], dtype=np.float64),
    }
    if rigid_pose_hist is not None:
        npz_arrays["rigid_target_pose"] = rigid_pose_hist

    return save_episode_bundle(
        output_dir=output_dir,
        episode_id=episode_id,
        task_id=task_id,
        engine_id="newton18",
        mode=mode,
        seed=seed,
        zero_friction=zero_friction,
        replay_source=str(replay_npz or ""),
        npz_arrays=npz_arrays,
        summary_metrics=metrics,
    )


def main() -> int:
    apply_low_priority_process_policy()
    parser = argparse.ArgumentParser(description="Newton 1.8.0 M3 v0.3 Deformable Rollout Worker")
    parser.add_argument("--batch-json", required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()

    jobs = json.loads(Path(args.batch_json).read_text(encoding="utf-8"))["jobs"]
    results = [run_single_episode(job) for job in jobs]
    out_path = Path(args.result_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"engine_id": "newton18", "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
