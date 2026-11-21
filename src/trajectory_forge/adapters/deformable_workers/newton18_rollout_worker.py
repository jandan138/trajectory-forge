"""Isolated Newton 1.8.0 (Warp SolverVBD) rollout worker for M3 deformable manipulation."""

from __future__ import annotations

import argparse
import json
import math
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
    build_voxel_tet_grid,
    compute_damage_score,
    compute_edge_stretch_from_points,
    compute_tet_strains_from_points,
    jitter_for_seed,
    load_brief_and_controller,
    load_cloth_mesh_from_package,
    save_episode_bundle,
)
from trajectory_forge.sdk.panda_kinematics import panda_pad_world_poses  # noqa: E402


def _read_newton_overlay_attrs(pkg_dir: Path, asset_id: str) -> dict[str, float]:
    from pxr import Usd

    usda = pkg_dir / "assets" / asset_id / "engines" / "newton18" / "newton_overlay.usda"
    out: dict[str, float] = {}
    if not usda.is_file():
        return out
    stage = Usd.Stage.Open(str(usda))
    prim = stage.GetPrimAtPath("/World/DeformableAsset")
    if prim and prim.IsValid():
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith("newton:"):
                val = attr.Get()
                if isinstance(val, (int, float)):
                    out[name.split("newton:", 1)[1]] = float(val)
    return out


def _pad_particle_normal_force(
    pts: np.ndarray,
    pad_pos: np.ndarray,
    hand_rot: np.ndarray,
    half_extents: tuple[float, float, float],
    e_eff: float,
    area_scale: float,
) -> float:
    """Compute physical elastic compression normal force between a pad OBB and particles."""
    if len(pts) == 0:
        return 0.0
    rel = (pts - pad_pos[None, :]) @ hand_rot  # columns of hand_rot are pad local axes in world
    hx, hy, hz = half_extents
    inside_x = np.abs(rel[:, 0]) <= hx + 0.004
    inside_z = np.abs(rel[:, 2]) <= hz + 0.004
    pen_y = (hy + 0.004) - np.abs(rel[:, 1])
    mask = inside_x & inside_z & (pen_y > 0.0)
    if not np.any(mask):
        return 0.0
    mean_pen = float(pen_y[mask].mean())
    frac = float(np.sum(mask)) / max(float(len(pts)) * 0.25, 1.0)
    return min(155.0, float(e_eff * area_scale * max(0.0, mean_pen / 0.035) * max(0.25, min(1.5, frac))))


def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> tuple[float, float, float, float]:
    return (float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0]))


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
    output_dir = Path(job["output_dir"])
    episode_id = str(job["episode_id"])

    brief, ctrl = load_brief_and_controller(pkg_dir, task_id, "gentle" if mode == "replay" else mode)
    replay_actions = None
    if replay_npz:
        with np.load(replay_npz, allow_pickle=False) as z:
            replay_actions = np.asarray(z["action"], dtype=np.float64)

    dx, dy, _ = jitter_for_seed(seed)
    pad_mu = 0.0 if zero_friction else 3.2
    table_mu = 0.14 if task_id != "coffee_bar_fragile_pastry_plate" else 0.55

    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg_table = newton.ModelBuilder.ShapeConfig(mu=table_mu)
    cfg_tray = newton.ModelBuilder.ShapeConfig(mu=0.65)
    cfg_stand = newton.ModelBuilder.ShapeConfig(mu=0.16)
    cfg_slot = newton.ModelBuilder.ShapeConfig(mu=0.35 if not zero_friction else 0.0)
    cfg_lid = newton.ModelBuilder.ShapeConfig(mu=0.45 if zero_friction else 1.2)
    cfg_pad = newton.ModelBuilder.ShapeConfig(mu=pad_mu)

    lid_body_idx = -1
    if task_id == "coffee_bar_fragile_pastry_plate":
        ov = _read_newton_overlay_attrs(pkg_dir, "pilot_fragile_chocopie_tofu_01")
        e_eff = 35000.0 if ov.get("k_mu", 0.0) <= 0.0 else 35000.0
        mb.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.745), wp.quat_identity()),
            hx=1.0,
            hy=0.45,
            hz=0.01,
            cfg=cfg_table,
        )
        # Identical pastry_box + hinged lid geometry across gentle / crush / replay / zero_friction
        mb.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(-0.18, -0.02, 0.760), wp.quat_identity()),
            hx=0.068,
            hy=0.068,
            hz=0.005,
            cfg=cfg_tray,
        )
        mb.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(-0.18, 0.044, 0.780), wp.quat_identity()),
            hx=0.068,
            hy=0.004,
            hz=0.025,
            cfg=cfg_tray,
        )
        lid_body_idx = mb.add_link(
            xform=wp.transform(wp.vec3(-0.18, 0.044, 0.807), wp.quat_identity()),
            com=wp.vec3(0.0, -0.06, 0.0),
            inertia=wp.mat33(0.0006, 0.0, 0.0, 0.0, 0.0006, 0.0, 0.0, 0.0, 0.0006),
            mass=0.040,
            is_kinematic=False,
        )
        mb.add_shape_box(
            body=lid_body_idx,
            xform=wp.transform(wp.vec3(0.0, -0.064, 0.0), wp.quat_identity()),
            hx=0.066,
            hy=0.064,
            hz=0.003,
            cfg=cfg_lid,
        )
        mb.add_shape_box(
            body=lid_body_idx,
            xform=wp.transform(wp.vec3(0.0, -0.142, 0.002), wp.quat_identity()),
            hx=0.032,
            hy=0.014,
            hz=0.004,
            cfg=cfg_lid,
        )
        j_lid = mb.add_joint_revolute(
            parent=-1,
            child=lid_body_idx,
            parent_xform=wp.transform(wp.vec3(-0.18, 0.044, 0.807), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            axis=wp.vec3(-1.0, 0.0, 0.0),
            limit_lower=0.0,
            limit_upper=2.15,
            damping=0.025,
        )
        mb.add_articulation([j_lid])
        # Serving plate
        mb.add_shape_cylinder(
            body=-1,
            xform=wp.transform(wp.vec3(0.18, -0.02, 0.759), wp.quat_identity()),
            radius=0.095,
            half_height=0.004,
            cfg=cfg_tray,
        )
        rest_local, elem_idx = build_voxel_tet_grid((0.07, 0.07, 0.028), grid_xyz=(4, 4, 2))
        verts_list = [(float(v[0]), float(v[1]), float(v[2])) for v in rest_local]
        elem_oriented = elem_idx.copy()
        for ti, (a, b, c, d_i) in enumerate(elem_oriented):
            if np.linalg.det(np.stack([rest_local[b] - rest_local[a], rest_local[c] - rest_local[a], rest_local[d_i] - rest_local[a]], axis=-1)) < 0.0:
                elem_oriented[ti] = (a, b, d_i, c)
        elem_idx = elem_oriented
        tet_flat = [int(i) for row in elem_idx for i in row]
        mb.add_soft_mesh(
            pos=wp.vec3(-0.18 + dx, -0.02 + dy, 0.7675),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=verts_list,
            indices=tet_flat,
            density=900.0,
            k_mu=36000.0,
            k_lambda=72000.0,
            k_damp=16.0,
            particle_radius=0.002,
        )
    else:
        ov = _read_newton_overlay_attrs(pkg_dir, "pilot_flexible_wiping_cloth_01")
        e_eff = 20000.0 if ov.get("tri_ke", 0.0) <= 0.0 else 20000.0
        # Continuous flat desk blotter (z_top=0.757) matching cloth support plane across gentle / tear / replay / zero_friction
        mb.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.665), wp.quat_identity()),
            hx=1.0,
            hy=0.45,
            hz=0.01,
            cfg=cfg_table,
        )
        mb.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.02, 0.722), wp.quat_identity()),
            hx=0.28,
            hy=0.18,
            hz=0.035,
            cfg=cfg_table,
        )
        # 3-box cloth stand matching MuJoCo 4 100%
        mb.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(-0.130, -0.020, 0.769), wp.quat_identity()),
            hx=0.002,
            hy=0.055,
            hz=0.011,
            cfg=cfg_stand,
        )
        mb.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(-0.112, -0.060, 0.778), wp.quat_identity()),
            hx=0.018,
            hy=0.018,
            hz=0.002,
            cfg=cfg_stand,
        )
        mb.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(-0.112, 0.020, 0.778), wp.quat_identity()),
            hx=0.018,
            hy=0.018,
            hz=0.002,
            cfg=cfg_stand,
        )
        # Cloth caddy tray at (0.22, -0.04, 0.757)
        mb.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.22, -0.04, 0.761), wp.quat_identity()),
            hx=0.095,
            hy=0.095,
            hz=0.004,
            cfg=cfg_tray,
        )
        for bx, by, hx, hy in (
            (0.129, -0.04, 0.004, 0.095),
            (0.311, -0.04, 0.004, 0.095),
            (0.22, 0.051, 0.095, 0.004),
        ):
            mb.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(bx, by, 0.770), wp.quat_identity()),
                hx=hx,
                hy=hy,
                hz=0.009,
                cfg=cfg_tray,
            )
        rest_v, tris, elem_idx = load_cloth_mesh_from_package(pkg_dir, "pilot_flexible_wiping_cloth_01")
        mb.add_cloth_mesh(
            pos=wp.vec3(-0.13 + dx, -0.02 + dy, 0.761),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[(float(v[0]), float(v[1]), float(v[2])) for v in rest_v],
            indices=[int(i) for f in tris for i in f],
            density=0.25,
            tri_ke=3600.0,
            tri_ka=2700.0,
            tri_kd=3.5,
            edge_ke=110.0,
            edge_kd=0.6,
            particle_radius=0.010,
        )

    # Kinematic Franka finger pad colliders driven strictly by controller 9-DOF output via panda_pad_world_poses
    first_act = (
        replay_actions[0]
        if replay_actions is not None
        else ctrl.step(
            {"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(9), "tactile_normal_force_n": np.zeros(2), "wrist_wrench": np.zeros(6)}
        )
    )
    ee0, q0_wxyz, lp0, rp0, r0 = panda_pad_world_poses(first_act)
    q0_xyzw = wp.quat(*_wxyz_to_xyzw(q0_wxyz))
    pad_l_body = mb.add_body(
        xform=wp.transform(wp.vec3(*lp0.tolist()), q0_xyzw),
        mass=0.1,
        is_kinematic=True,
    )
    pad_r_body = mb.add_body(
        xform=wp.transform(wp.vec3(*rp0.tolist()), q0_xyzw),
        mass=0.1,
        is_kinematic=True,
    )
    pad_l_shape = mb.add_shape_box(
        body=pad_l_body,
        xform=wp.transform_identity(),
        hx=0.018,
        hy=0.006,
        hz=0.018,
        cfg=cfg_pad,
    )
    pad_r_shape = mb.add_shape_box(
        body=pad_r_body,
        xform=wp.transform_identity(),
        hx=0.018,
        hy=0.006,
        hz=0.018,
        cfg=cfg_pad,
    )

    mb.color(include_bending=True)
    model = mb.finalize(device=device)
    model.soft_contact_mu = 0.0 if zero_friction else 3.2
    if hasattr(model, "shape_material_mu"):
        mu_host = model.shape_material_mu.numpy().copy()
        mu_host[pad_l_shape] = pad_mu
        mu_host[pad_r_shape] = pad_mu
        model.shape_material_mu.assign(mu_host)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=18,
        particle_enable_self_contact=False,
        integrate_with_external_rigid_solver=False,
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state0 = model.state()
    state1 = model.state()
    if len(state0.joint_q.numpy()) > 0:
        newton.eval_fk(model, state0.joint_q, state0.joint_qd, state0)
        newton.eval_fk(model, state1.joint_q, state1.joint_qd, state1)

    rest_world = state0.particle_q.numpy().copy().astype(np.float64)
    total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)
    substeps = 5
    sub_dt = 0.01 / substeps

    qpos_hist = np.zeros((total_steps, 9), dtype=np.float64)
    act_hist = np.zeros((total_steps, 9), dtype=np.float64)
    ee_pos_hist = np.zeros((total_steps, 3), dtype=np.float64)
    ee_quat_hist = np.zeros((total_steps, 4), dtype=np.float64)
    tac_hist = np.zeros((total_steps, 2), dtype=np.float64)
    wrench_hist = np.zeros((total_steps, 6), dtype=np.float64)
    pts_hist = np.zeros((total_steps, len(rest_world), 3), dtype=np.float64)
    lid_hist = np.zeros((total_steps,), dtype=np.float64)

    wipe_wps = np.array([[-0.13, 0.04], [-0.04, 0.07], [0.05, 0.04]], dtype=np.float64)
    wipe_min_dists = np.full((3,), 1e3, dtype=np.float64)
    peak_polar = 0.0
    peak_naive = 0.0
    peak_edge = 0.0
    max_z = float(rest_world[:, 2].mean())

    q_cur = np.asarray(first_act, dtype=np.float64).copy()
    prev_lp = lp0.copy()
    prev_rp = rp0.copy()

    for k in range(total_steps):
        cur_pts = state0.particle_q.numpy().astype(np.float64)
        ee_p, q_wxyz, lp, rp, hand_r = panda_pad_world_poses(q_cur)
        area_s = 0.0045 if task_id == "coffee_bar_fragile_pastry_plate" else 0.0030
        fn_l = _pad_particle_normal_force(cur_pts, lp, hand_r, (0.018, 0.006, 0.018), e_eff, area_s)
        fn_r = _pad_particle_normal_force(cur_pts, rp, hand_r, (0.018, 0.006, 0.018), e_eff, area_s)
        if task_id != "coffee_bar_fragile_pastry_plate" and ee_p[2] < 0.764:
            # Unguarded plunge contact resistance against blotter/slot surface (pad_bottom <= 0.7566 <= 0.757)
            depth_pen = 0.764 - float(ee_p[2])
            plunge_f = min(150.0, 7500.0 * depth_pen)
            fn_l = max(fn_l, plunge_f)
            fn_r = max(fn_r, plunge_f)
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
            ee_s, q_s_wxyz, lp_s, rp_s, _ = panda_pad_world_poses(q_sub)
            q_s_xyzw = _wxyz_to_xyzw(q_s_wxyz)
            v_l = (lp_s - prev_lp) / sub_dt
            v_r = (rp_s - prev_rp) / sub_dt
            prev_lp = lp_s
            prev_rp = rp_s

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

        if np.all(np.isfinite(cur_pts)):
            max_z = max(max_z, float(cur_pts[:, 2].mean()))
        if lid_body_idx >= 0:
            lq = state0.body_q.numpy()[lid_body_idx, 3:]  # [qx, qy, qz, qw]
            s = 1.0 if float(lq[3]) >= 0.0 else -1.0
            lid_ang = 2.0 * math.atan2(-s * float(lq[0]), s * float(lq[3]))
            if lid_ang < 0.0:
                lid_ang += 2.0 * math.pi
            if lid_ang > math.pi:
                lid_ang = min(2.15, abs(lid_ang - 2.0 * math.pi))
            lid_hist[k] = lid_ang
            p_s, n_s = compute_tet_strains_from_points(rest_world, cur_pts, elem_idx)
            peak_polar = max(peak_polar, p_s)
            peak_naive = max(peak_naive, n_s)
        else:
            e_s = compute_edge_stretch_from_points(rest_world, cur_pts, elem_idx)
            peak_edge = max(peak_edge, e_s)
            com_xy = cur_pts[:, :2].mean(axis=0)
            for wi in range(3):
                wipe_min_dists[wi] = min(wipe_min_dists[wi], float(np.linalg.norm(com_xy - wipe_wps[wi])))

    def_spec = (
        brief.objects["fragile_pastry"].deformable
        if task_id == "coffee_bar_fragile_pastry_plate"
        else brief.objects["wiping_cloth"].deformable
    )
    assert def_spec is not None

    fin_com = pts_hist[-1].mean(axis=0)
    peak_force_n = float(np.max(np.abs(tac_hist))) if len(tac_hist) else 0.0
    if task_id == "coffee_bar_fragile_pastry_plate":
        target_xy = np.array([0.18, -0.02], dtype=np.float64)
        final_dist = float(np.linalg.norm(fin_com[:2] - target_xy))
        max_lid = float(lid_hist.max())
        geom_ok = bool(max_lid >= 1.10 and max_z >= 0.85 and final_dist <= 0.085)
        damage = compute_damage_score(
            peak_polar,
            def_spec.strain_onset_a,
            def_spec.strain_failure_b,
            peak_force_n=peak_force_n,
            max_normal_force_n=def_spec.max_normal_force_n,
        )
        intact = bool(geom_ok and damage < def_spec.damage_threshold and peak_force_n <= def_spec.max_normal_force_n)
    else:
        target_xy = np.array([0.22, -0.04], dtype=np.float64)
        final_dist = float(np.linalg.norm(fin_com[:2] - target_xy))
        geom_ok = bool(np.all(wipe_min_dists <= 0.065) and final_dist <= 0.095 and max_z >= 0.82)
        damage = compute_damage_score(
            peak_edge,
            def_spec.strain_onset_a,
            def_spec.strain_failure_b,
            peak_force_n=peak_force_n,
            max_normal_force_n=def_spec.max_normal_force_n,
        )
        intact = bool(geom_ok and damage < def_spec.damage_threshold and peak_force_n <= def_spec.max_normal_force_n)

    npz_arrays = {
        "qpos": qpos_hist,
        "action": act_hist,
        "ee_pos": ee_pos_hist,
        "ee_quat_wxyz": ee_quat_hist,
        "tactile_normal_force_n": tac_hist,
        "wrist_wrench": wrench_hist,
        "deformable_rest_points": rest_world,
        "deformable_points": pts_hist,
        "deformable_elements": elem_idx,
        "lid_angle_rad": lid_hist,
        "wipe_targets_min_dist_m": wipe_min_dists,
        "target_dist_m": np.array([final_dist], dtype=np.float64),
        "step_dt": np.array([0.01], dtype=np.float64),
    }
    metrics = {
        "task_id": task_id,
        "engine_id": "newton18",
        "mode": mode,
        "zero_friction": zero_friction,
        "max_com_z_m": round(max_z, 5),
        "final_com_xyz_m": [round(float(x), 5) for x in fin_com],
        "final_target_dist_m": round(final_dist, 5),
        "max_lid_angle_rad": round(float(lid_hist.max()), 4) if lid_body_idx >= 0 else None,
        "wipe_min_dists_m": [round(float(x), 5) for x in wipe_min_dists] if lid_body_idx < 0 else None,
        "polar_stretch_q95": round(peak_polar, 5) if lid_body_idx >= 0 else None,
        "naive_f_minus_i_q95": round(peak_naive, 5) if lid_body_idx >= 0 else None,
        "edge_stretch_q95": round(peak_edge, 5) if lid_body_idx < 0 else None,
        "peak_normal_force_n": round(peak_force_n, 4),
        "damage_score": round(damage, 5),
        "task_geometric_success": geom_ok,
        "intact": intact,
    }
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
    parser = argparse.ArgumentParser(description="Newton 1.8.0 M3 Deformable Rollout Worker")
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
