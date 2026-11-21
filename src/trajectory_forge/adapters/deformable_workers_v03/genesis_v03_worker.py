"""Isolated Genesis 1.4.7 M3 v0.3 rollout worker (ZERO-CHEAT: MPM CPIC + Native PBD, 8% VRAM cap)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

_SRC_ROOT = str(Path(__file__).resolve().parents[3])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from trajectory_forge.adapters.deformable_workers.common import (  # noqa: E402
    apply_low_priority_process_policy,
    jitter_for_seed,
    save_episode_bundle,
)
from trajectory_forge.adapters.deformable_workers_v03.common_v03 import (  # noqa: E402
    evaluate_v03_task_outcome,
    load_v03_proxy_and_brief,
)
from trajectory_forge.sdk.panda_kinematics import forward_kinematics_panda, panda_pad_world_poses  # noqa: E402


def _cpic_pad_poses(
    q9: np.ndarray,
    hy: float = 0.012,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ee_pos, hand_rot, ee_quat = forward_kinematics_panda(q9[:7])
    fl, fr = float(q9[7]), float(q9[8])
    lp = ee_pos + hand_rot @ np.array([0.0, hy + fl, -0.0106], dtype=np.float64)
    rp = ee_pos + hand_rot @ np.array([0.0, -(hy + fr), -0.0106], dtype=np.float64)
    return ee_pos, ee_quat, lp, rp, hand_rot


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> Path:
    lines = ["# genesis v03 mesh"]
    for v in verts:
        lines.append(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}")
    for f in faces:
        lines.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _pad_particle_normal_force(
    pts_world: np.ndarray,
    pad_center: np.ndarray,
    hand_rot: np.ndarray,
    pad_half_extents: tuple[float, float, float],
    e_eff_pa: float,
    area_scale_m2: float,
) -> float:
    if len(pts_world) == 0:
        return 0.0
    local = (pts_world - pad_center[None, :]) @ hand_rot
    hx, hy, hz = pad_half_extents
    inside_x = np.abs(local[:, 0]) <= hx + 0.004
    inside_z = np.abs(local[:, 2]) <= hz + 0.004
    pen_y = (hy + 0.004) - np.abs(local[:, 1])
    mask = inside_x & inside_z & (pen_y > 0.0)
    if not np.any(mask):
        return 0.0
    mean_pen = float(pen_y[mask].mean())
    frac = float(np.sum(mask)) / max(float(len(pts_world)) * 0.25, 1.0)
    return min(155.0, float(e_eff_pa * area_scale_m2 * max(0.0, mean_pen / 0.035) * max(0.25, min(1.5, frac))))


def run_job(gs: Any, qd_to_torch: Any, job: dict[str, Any]) -> dict[str, Any]:
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
    is_mpm = family in ("soft_solid", "rod_cable")
    young = float(spec["youngs_modulus_pa"])

    ctrl_dt = 0.01
    sim_dt = 0.002
    substeps_per_ctrl = 5
    hy_pad = 0.012

    pad_mat = gs.materials.Rigid(
        rho=1.0e8,
        friction=0.01 if zero_friction else 3.8,
        coup_friction=0.0 if zero_friction else 4.2,
        coup_softness=0.0028,
        sdf_cell_size=0.002,
        gravity_compensation=1.0,
    )
    tray_mat = gs.materials.Rigid(
        rho=2000.0,
        friction=0.65,
        coup_friction=0.65,
        coup_softness=0.002,
        sdf_cell_size=0.004,
    )
    table_mu = 0.45 if task_id == "office_waffle_towel_corner_fold" else (0.14 if family == "cloth_sheet" else 0.55)
    table_mat = gs.materials.Rigid(
        rho=2000.0,
        friction=table_mu,
        coup_friction=table_mu,
        coup_softness=0.002,
        sdf_cell_size=0.004,
    )
    box_mat = gs.materials.Rigid(
        rho=2500.0,
        friction=0.01 if zero_friction else 1.4,
        coup_friction=0.0 if zero_friction else 1.4,
        coup_softness=0.002,
        sdf_cell_size=0.003,
    )

    first_act = (
        replay_actions[0]
        if replay_actions is not None
        else ctrl.step(
            {"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(9), "tactile_normal_force_n": np.zeros(2), "wrist_wrench": np.zeros(6)}
        )
    )
    _, q0_wxyz, lp0, rp0, _ = (
        _cpic_pad_poses(first_act, hy_pad)
        if is_mpm
        else panda_pad_world_poses(first_act)
    )

    def _xyz(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
        obj = brief.objects.get(name)
        return tuple(float(v) for v in obj.initial_pose.xyz) if obj else default

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        if is_mpm:
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=sim_dt, substeps=12),
                mpm_options=gs.options.MPMOptions(
                    lower_bound=(-0.38, -0.25, 0.72),
                    upper_bound=(0.38, 0.25, 1.05),
                    grid_density=128,
                    enable_CPIC=True,
                ),
                show_viewer=False,
            )
            pad_size_xyz = (0.040, 2.0 * hy_pad, 0.036)
        else:
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=sim_dt, substeps=8),
                pbd_options=gs.options.PBDOptions(
                    max_stretch_solver_iterations=10,
                    max_bending_solver_iterations=4,
                    particle_size=0.005,
                    lower_bound=(-0.45, -0.35, 0.70),
                    upper_bound=(0.45, 0.35, 1.05),
                ),
                show_viewer=False,
            )
            pad_size_xyz = (0.036, 0.012, 0.036)

        # Single 0.04 m-thick support slab (avoids overlapping 2mm mat over table_top)
        if "smooth_folding_mat" in brief.objects:
            _, _, zm = _xyz("smooth_folding_mat", (0.0, 0.02, 0.755))
            z_top = zm + 0.002
        else:
            z_top = 0.755
        scene.add_entity(material=table_mat, morph=gs.morphs.Box(pos=(0.0, 0.0, z_top - 0.020), size=(2.0, 0.90, 0.04), fixed=True))
        if "baking_sheet_tray" in brief.objects:
            x, y, z = _xyz("baking_sheet_tray", (-0.18, -0.02, 0.755))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y, z + 0.003), size=(0.190, 0.170, 0.006), fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y, z + 0.0055), size=(0.168, 0.148, 0.002), fixed=True))
        if "tartlet_pastry_cup" in brief.objects:
            x, y, z = _xyz("tartlet_pastry_cup", (0.18, -0.02, 0.755))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Cylinder(pos=(x, y, z + 0.0025), radius=0.085, height=0.005, fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Cylinder(pos=(x, y, z + 0.013), radius=0.048, height=0.016, fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Cylinder(pos=(x, y, z + 0.021), radius=0.043, height=0.004, fixed=True))
        if "garment_folding_board" in brief.objects:
            x, y, z = _xyz("garment_folding_board", (0.0, 0.02, 0.755))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x + 0.045, y, z + 0.014), size=(0.190, 0.260, 0.028), fixed=True))

        toner_ent = None
        if "toner_cartridge_box" in brief.objects:
            bx, by, bz = _xyz("toner_cartridge_box", (-0.14, 0.02, 0.757))
            toner_ent = scene.add_entity(material=box_mat, morph=gs.morphs.Box(pos=(bx + dx, by + dy, bz + 0.022), size=(0.064, 0.064, 0.044), fixed=False))

        if "delivery_caddy" in brief.objects:
            x, y, z = _xyz("delivery_caddy", (0.20, -0.02, 0.757))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y, z + 0.003), size=(0.170, 0.170, 0.006), fixed=True))
        if "cable_carrier_cradle" in brief.objects:
            x, y, z = _xyz("cable_carrier_cradle", (-0.16, -0.03, 0.755))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y, z + 0.004), size=(0.210, 0.110, 0.008), fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x - 0.068, y, z + 0.011), size=(0.024, 0.072, 0.014), fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x + 0.068, y, z + 0.011), size=(0.024, 0.072, 0.014), fixed=True))
        if "smt_u_clip_block" in brief.objects:
            x, y, z = _xyz("smt_u_clip_block", (0.14, 0.04, 0.755))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y, z + 0.004), size=(0.210, 0.096, 0.008), fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y - 0.020, z + 0.014), size=(0.164, 0.010, 0.024), fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y + 0.020, z + 0.014), size=(0.164, 0.010, 0.024), fixed=True))
        if "pcba_header_boss" in brief.objects:
            x, y, z = _xyz("pcba_header_boss", (0.15, -0.02, 0.755))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y, z + 0.003), size=(0.150, 0.150, 0.006), fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y, z + 0.007), size=(0.052, 0.052, 0.004), fixed=True))
        if "tube_bench_cradle" in brief.objects:
            x, y, z = _xyz("tube_bench_cradle", (-0.17, -0.02, 0.755))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y, z + 0.004), size=(0.210, 0.110, 0.008), fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x - 0.068, y, z + 0.011), size=(0.028, 0.064, 0.014), fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x + 0.068, y, z + 0.011), size=(0.028, 0.064, 0.014), fixed=True))
        if "manifold_fork_saddle" in brief.objects:
            x, y, z = _xyz("manifold_fork_saddle", (0.16, 0.03, 0.755))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y, z + 0.004), size=(0.210, 0.110, 0.008), fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y - 0.022, z + 0.015), size=(0.170, 0.010, 0.026), fixed=True))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y + 0.022, z + 0.015), size=(0.170, 0.010, 0.026), fixed=True))
        if "bulb_collar_holder" in brief.objects:
            x, y, z = _xyz("bulb_collar_holder", (0.15, -0.02, 0.755))
            scene.add_entity(material=tray_mat, morph=gs.morphs.Box(pos=(x, y, z + 0.003), size=(0.130, 0.130, 0.006), fixed=True))

        rest_v = np.asarray(proxy["rest_points"], dtype=np.float64)
        tris = np.asarray(proxy["triangles"], dtype=np.int64)
        obj_path = _write_obj(td_path / "proxy_rest.obj", rest_v, tris)
        sx, sy, sz = def_obj.initial_pose.xyz

        if is_mpm:
            mpm_e = max(65000.0, 1.5 * young)
            mpm_ent = scene.add_entity(
                material=gs.materials.MPM.ElastoPlastic(
                    E=mpm_e,
                    nu=0.25,
                    rho=850.0,
                    yield_lower=0.48,
                    yield_higher=0.60,
                    use_von_mises=True,
                    von_mises_yield_stress=max(20000.0, float(spec.get("yield_stress_pa") or 30000.0) * 1.4),
                ),
                morph=gs.morphs.Mesh(file=str(obj_path), pos=(sx + dx, sy + dy, sz + 0.001)),
            )
            cloth_ent = None
            elem_idx = np.zeros((0, 4), dtype=np.int64)
        else:
            mpm_ent = None
            elem_idx = np.asarray(proxy["edges"], dtype=np.int64)
            is_shell = family == "shell"
            cloth_ent = scene.add_entity(
                material=gs.materials.PBD.Cloth(
                    rho=0.45 if is_shell else 0.25,
                    static_friction=0.0 if zero_friction else 2.0,
                    kinetic_friction=0.0 if zero_friction else 1.8,
                    stretch_compliance=1e-7 if is_shell else 1e-6,
                    bending_compliance=5e-7 if is_shell else 2e-5,
                    stretch_relaxation=0.5 if is_shell else 0.4,
                    bending_relaxation=0.4 if is_shell else 0.3,
                    air_resistance=0.015,
                ),
                morph=gs.morphs.Mesh(file=str(obj_path), pos=(sx + dx, sy + dy, sz)),
            )

        left_pad = scene.add_entity(material=pad_mat, morph=gs.morphs.Box(pos=tuple(lp0.tolist()), size=pad_size_xyz, fixed=False))
        right_pad = scene.add_entity(material=pad_mat, morph=gs.morphs.Box(pos=tuple(rp0.tolist()), size=pad_size_xyz, fixed=False))
        scene.build()

    left_pad.set_pos(lp0.tolist(), zero_velocity=True, relative=False)
    left_pad.set_quat(q0_wxyz.tolist(), zero_velocity=True, relative=False)
    right_pad.set_pos(rp0.tolist(), zero_velocity=True, relative=False)
    right_pad.set_quat(q0_wxyz.tolist(), zero_velocity=True, relative=False)

    if mpm_ent is not None:
        p_start = int(getattr(mpm_ent, "particle_start", mpm_ent._particle_start))
        p_count = int(getattr(mpm_ent, "n_particles", mpm_ent._n_particles))
        p_end = p_start + p_count

        def _read_pts() -> np.ndarray:
            t = mpm_ent.get_particles_pos()
            if t.ndim == 3:
                t = t[0]
            return t.detach().cpu().numpy().astype(np.float64)

        def _read_F() -> np.ndarray:
            f_all = qd_to_torch(mpm_ent._solver.mpm_state.particles.F, copy=True)
            return f_all[p_start:p_end, 0].detach().cpu().numpy().astype(np.float64)
    else:
        assert cloth_ent is not None
        raw_edges = getattr(cloth_ent, "edges", None)
        if raw_edges is not None and len(raw_edges) > 0:
            elem_idx = np.asarray(raw_edges, dtype=np.int64).reshape(-1, 2)

        def _read_pts() -> np.ndarray:
            t = cloth_ent.get_particles_pos()
            if t.ndim == 3:
                t = t[0]
            return t.detach().cpu().numpy().astype(np.float64)

        def _read_F() -> np.ndarray | None:
            return None

    rest_world = _read_pts().copy()
    if task_id in ("office_dust_cover_peel_retrieve", "office_waffle_towel_corner_fold"):
        n_set = 150 if task_id == "office_dust_cover_peel_retrieve" else 15
        for _ in range(n_set):
            scene.step()

    total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)
    qpos_hist = np.zeros((total_steps, 9), dtype=np.float64)
    act_hist = np.zeros((total_steps, 9), dtype=np.float64)
    ee_pos_hist = np.zeros((total_steps, 3), dtype=np.float64)
    ee_quat_hist = np.zeros((total_steps, 4), dtype=np.float64)
    tac_hist = np.zeros((total_steps, 2), dtype=np.float64)
    wrench_hist = np.zeros((total_steps, 6), dtype=np.float64)
    pts_hist = np.zeros((total_steps, len(rest_world), 3), dtype=np.float64)
    f_hist = np.zeros((total_steps, len(rest_world), 3, 3), dtype=np.float32) if mpm_ent is not None else None
    rigid_pose_hist = np.zeros((total_steps, 7), dtype=np.float64) if toner_ent is not None else None

    q_cur = np.asarray(first_act, dtype=np.float64).copy()
    prev_lp, prev_rp = lp0.copy(), rp0.copy()
    area_s = 0.0022
    tactile_e = min(young, 28000.0)

    for k in range(total_steps):
        cur_pts = _read_pts()
        ee_p, _, lp_std, rp_std, hand_r = panda_pad_world_poses(q_cur)
        fn_l = _pad_particle_normal_force(cur_pts, lp_std, hand_r, (0.018, 0.006, 0.018), tactile_e, area_s)
        fn_r = _pad_particle_normal_force(cur_pts, rp_std, hand_r, (0.018, 0.006, 0.018), tactile_e, area_s)
        if float(ee_p[2]) < 0.761:
            plunge_f = min(150.0, 9000.0 * (0.761 - float(ee_p[2])))
            fn_l = max(fn_l, plunge_f)
            fn_r = max(fn_r, plunge_f)
        w_obs = np.array([0.0, 0.0, -(fn_l + fn_r) * 0.5, 0.0, 0.0, 0.0], dtype=np.float64)

        if replay_actions is not None:
            act = replay_actions[k]
        else:
            act = ctrl.step(
                {
                    "step_idx": k,
                    "t_s": k * ctrl_dt,
                    "qpos": q_cur,
                    "ee_pos": ee_p,
                    "tactile_normal_force_n": np.array([fn_l, fn_r], dtype=np.float64),
                    "wrist_wrench": w_obs,
                }
            )

        q_prev_step = q_cur.copy()
        for sub in range(substeps_per_ctrl):
            alpha = (sub + 1) / float(substeps_per_ctrl)
            q_sub = (1.0 - alpha) * q_prev_step + alpha * act
            if is_mpm:
                _, q_s_wxyz, lp_s, rp_s, _ = _cpic_pad_poses(q_sub, hy_pad)
            else:
                _, q_s_wxyz, lp_s, rp_s, _ = panda_pad_world_poses(q_sub)
            v_l = (lp_s - prev_lp) / sim_dt
            v_r = (rp_s - prev_rp) / sim_dt
            prev_lp, prev_rp = lp_s, rp_s
            left_pad.set_pos(lp_s.tolist(), zero_velocity=False, relative=False)
            left_pad.set_quat(q_s_wxyz.tolist(), zero_velocity=False, relative=False)
            left_pad.set_dofs_velocity(np.array([v_l[0], v_l[1], v_l[2], 0.0, 0.0, 0.0], dtype=np.float32))
            right_pad.set_pos(rp_s.tolist(), zero_velocity=False, relative=False)
            right_pad.set_quat(q_s_wxyz.tolist(), zero_velocity=False, relative=False)
            right_pad.set_dofs_velocity(np.array([v_r[0], v_r[1], v_r[2], 0.0, 0.0, 0.0], dtype=np.float32))
            scene.step()

        q_cur = act.copy()
        cur_pts = _read_pts()
        ee_fin, q_fin_wxyz, _, _, _ = panda_pad_world_poses(q_cur)
        qpos_hist[k] = q_cur
        act_hist[k] = act
        ee_pos_hist[k] = ee_fin
        ee_quat_hist[k] = q_fin_wxyz
        tac_hist[k] = np.array([fn_l, fn_r], dtype=np.float64)
        wrench_hist[k] = w_obs
        pts_hist[k] = cur_pts
        if f_hist is not None:
            f_step = _read_F()
            assert f_step is not None
            f_hist[k] = f_step.astype(np.float32)
        if rigid_pose_hist is not None and toner_ent is not None:
            p_b = toner_ent.get_pos().detach().cpu().numpy().reshape(-1)
            q_b = toner_ent.get_quat().detach().cpu().numpy().reshape(-1)
            rigid_pose_hist[k, :3] = [float(p_b[0]), float(p_b[1]), float(p_b[2]) - 0.022]
            rigid_pose_hist[k, 3:] = q_b[:4]

    metrics, final_dist = evaluate_v03_task_outcome(
        task_id=task_id,
        engine_id="genesis",
        mode=mode,
        zero_friction=zero_friction,
        brief=brief,
        rest_world=rest_world,
        pts_hist=pts_hist,
        elems_or_edges=elem_idx,
        tac_hist=tac_hist,
        f_hist=f_hist,
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
        "deformable_elements": elem_idx,
        "target_dist_m": np.array([final_dist], dtype=np.float64),
        "step_dt": np.array([ctrl_dt], dtype=np.float64),
    }
    if f_hist is not None:
        npz_arrays["deformable_particle_F"] = f_hist
    if rigid_pose_hist is not None:
        npz_arrays["rigid_target_pose"] = rigid_pose_hist

    return save_episode_bundle(
        output_dir=output_dir,
        episode_id=episode_id,
        task_id=task_id,
        engine_id="genesis",
        mode=mode,
        seed=seed,
        zero_friction=zero_friction,
        replay_source=str(replay_npz or ""),
        npz_arrays=npz_arrays,
        summary_metrics=metrics,
    )


def main() -> int:
    apply_low_priority_process_policy()
    parser = argparse.ArgumentParser(description="Genesis 1.4.7 M3 v0.3 zero-cheat worker.")
    parser.add_argument("--batch-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()

    batch_payload = json.loads(args.batch_json.read_text(encoding="utf-8"))
    jobs = batch_payload["jobs"] if isinstance(batch_payload, dict) else batch_payload
    import torch
    import genesis as gs
    from genesis.utils.misc import qd_to_torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.08)
    gs.init(backend=gs.cuda, logging_level="warning")

    results = [run_job(gs, qd_to_torch, job) for job in jobs]
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps({"engine_id": "genesis", "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
