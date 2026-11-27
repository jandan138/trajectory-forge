"""Isolated Genesis 1.4.7 M3 rollout worker (MPM.ElastoPlastic + PBD.Cloth, 8% VRAM cap)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

_SRC_ROOT = str(Path(__file__).resolve().parents[3])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from trajectory_forge.adapters.legacy.deformable_workers.common import (  # noqa: E402
    apply_low_priority_process_policy,
    compute_damage_score,
    compute_edge_stretch_from_points,
    compute_tet_strains_from_F,
    jitter_for_seed,
    load_brief_and_controller,
    load_cloth_mesh_from_package,
    save_episode_bundle,
)
from trajectory_forge.sdk.panda_kinematics import forward_kinematics_panda, panda_pad_world_poses  # noqa: E402


def _cpic_pad_poses(
    q9: np.ndarray,
    hy: float = 0.012,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Offset pad centers by +/-hy so inner faces align with finger gap +/-f and outer stencil clears pastry."""
    ee_pos, hand_rot, ee_quat = forward_kinematics_panda(q9[:7])
    fl, fr = float(q9[7]), float(q9[8])
    lp = ee_pos + hand_rot @ np.array([0.0, hy + fl, -0.0106], dtype=np.float64)
    rp = ee_pos + hand_rot @ np.array([0.0, -(hy + fr), -0.0106], dtype=np.float64)
    return ee_pos, ee_quat, lp, rp, hand_rot


_PASTRY_BOX_URDF = """<?xml version="1.0"?>
<robot name="pastry_hinged_box">
  <link name="box_base">
    <inertial>
      <origin xyz="-0.18 -0.02 0.77" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="-0.18 -0.02 0.760" rpy="0 0 0"/>
      <geometry><box size="0.136 0.136 0.010"/></geometry>
    </visual>
    <collision>
      <origin xyz="-0.18 -0.02 0.760" rpy="0 0 0"/>
      <geometry><box size="0.136 0.136 0.010"/></geometry>
    </collision>
    <visual>
      <origin xyz="-0.18 0.044 0.780" rpy="0 0 0"/>
      <geometry><box size="0.136 0.008 0.050"/></geometry>
    </visual>
    <collision>
      <origin xyz="-0.18 0.044 0.780" rpy="0 0 0"/>
      <geometry><box size="0.136 0.008 0.050"/></geometry>
    </collision>
  </link>
  <link name="lid_link">
    <inertial>
      <origin xyz="0 -0.06 0" rpy="0 0 0"/>
      <mass value="0.040"/>
      <inertia ixx="0.0006" ixy="0" ixz="0" iyy="0.0006" iyz="0" izz="0.0006"/>
    </inertial>
    <visual>
      <origin xyz="0 -0.064 0" rpy="0 0 0"/>
      <geometry><box size="0.132 0.128 0.006"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 -0.064 0" rpy="0 0 0"/>
      <geometry><box size="0.132 0.128 0.006"/></geometry>
    </collision>
    <visual>
      <origin xyz="0 -0.142 0.002" rpy="0 0 0"/>
      <geometry><box size="0.064 0.028 0.008"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 -0.142 0.002" rpy="0 0 0"/>
      <geometry><box size="0.064 0.028 0.008"/></geometry>
    </collision>
  </link>
  <joint name="lid_hinge" type="revolute">
    <parent link="box_base"/>
    <child link="lid_link"/>
    <origin xyz="-0.18 0.044 0.807" rpy="0 0 0"/>
    <axis xyz="-1 0 0"/>
    <limit lower="0.0" upper="2.15" effort="50.0" velocity="20.0"/>
    <dynamics damping="0.025" friction="0.0"/>
  </joint>
</robot>
"""


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> Path:
    lines = ["# genesis cloth mesh"]
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
    """Compute physical elastic compression normal force between a pad OBB and particles."""
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
    import torch

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
    ctrl_dt = 0.01
    sim_dt = 0.002
    substeps_per_ctrl = 5
    hy_pad = 0.012

    pad_mat = (
        gs.materials.Rigid(
            rho=1.0e8,
            friction=0.01 if zero_friction else 3.8,
            coup_friction=0.0 if zero_friction else 4.2,
            coup_softness=0.0030,
            sdf_cell_size=0.002,
            gravity_compensation=1.0,
        )
        if task_id == "coffee_bar_fragile_pastry_plate"
        else gs.materials.Rigid(
            rho=1.0e8,
            needs_coup=False,
            gravity_compensation=1.0,
        )
    )
    tray_mat = gs.materials.Rigid(
        rho=2000.0,
        friction=0.65,
        coup_friction=0.65,
        coup_softness=0.002,
        sdf_cell_size=0.004,
    )
    blotter_mat = gs.materials.Rigid(
        rho=2000.0,
        friction=0.16,
        coup_friction=0.14,
        coup_softness=0.002,
        sdf_cell_size=0.005,
    )
    stand_mat = gs.materials.Rigid(
        rho=2000.0,
        friction=0.16,
        coup_friction=0.14,
        coup_softness=0.002,
        sdf_cell_size=0.003,
    )
    slot_mat = gs.materials.Rigid(
        rho=2000.0,
        friction=0.01 if zero_friction else 2.5,
        coup_friction=0.0 if zero_friction else 2.2,
        coup_softness=0.002,
        sdf_cell_size=0.004,
    )
    lid_mat = gs.materials.Rigid(
        rho=800.0,
        friction=0.45 if zero_friction else 1.1,
        coup_friction=0.3 if zero_friction else 0.9,
        coup_softness=0.002,
        sdf_cell_size=0.004,
    )

    first_act = (
        replay_actions[0]
        if replay_actions is not None
        else ctrl.step(
            {"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(9), "tactile_normal_force_n": np.zeros(2), "wrist_wrench": np.zeros(6)}
        )
    )
    _, q0_wxyz, lp0, rp0, hr0 = (
        _cpic_pad_poses(first_act, hy_pad)
        if task_id == "coffee_bar_fragile_pastry_plate"
        else panda_pad_world_poses(first_act)
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        if task_id == "coffee_bar_fragile_pastry_plate":
            e_eff = 16000.0
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=sim_dt, substeps=11),
                mpm_options=gs.options.MPMOptions(
                    lower_bound=(-0.38, -0.25, 0.72),
                    upper_bound=(0.38, 0.25, 1.05),
                    grid_density=108,
                    enable_CPIC=True,
                ),
                show_viewer=False,
            )
            scene.add_entity(
                material=tray_mat,
                morph=gs.morphs.Box(pos=(0.0, 0.0, 0.745), size=(2.0, 0.90, 0.02), fixed=True),
            )
            scene.add_entity(
                material=tray_mat,
                morph=gs.morphs.Cylinder(pos=(0.18, -0.02, 0.759), radius=0.095, height=0.008, fixed=True),
            )
            urdf_path = td_path / "pastry_box.urdf"
            urdf_path.write_text(_PASTRY_BOX_URDF, encoding="utf-8")
            pastry_box = scene.add_entity(
                material=lid_mat,
                morph=gs.morphs.URDF(file=str(urdf_path), fixed=True, merge_fixed_links=False),
            )
            mpm_pastry = scene.add_entity(
                material=gs.materials.MPM.ElastoPlastic(
                    E=65000.0,
                    nu=0.24,
                    rho=800.0,
                    yield_lower=0.48,
                    yield_higher=0.60,
                    use_von_mises=True,
                    von_mises_yield_stress=40000.0,
                ),
                morph=gs.morphs.Box(pos=(-0.18 + dx, -0.02 + dy, 0.7805), size=(0.070, 0.070, 0.028)),
            )
            cloth = None
            elem_idx = np.zeros((0, 4), dtype=np.int64)
            pad_size_xyz = (0.040, 2.0 * hy_pad, 0.036)
        else:
            e_eff = 20000.0
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=sim_dt, substeps=8),
                pbd_options=gs.options.PBDOptions(
                    max_stretch_solver_iterations=8,
                    max_bending_solver_iterations=4,
                    particle_size=0.015,
                    lower_bound=(-0.50, -0.40, 0.65),
                    upper_bound=(0.50, 0.40, 1.10),
                ),
                show_viewer=False,
            )
            # Continuous flat desk blotter + 3-box cloth stand + caddy
            for pos_b, sz_b, mat_b in (
                ((0.0, 0.0, 0.665), (2.0, 0.90, 0.020), blotter_mat),
                ((0.0, 0.02, 0.722), (0.56, 0.36, 0.070), blotter_mat),
                ((-0.130, -0.020, 0.769), (0.004, 0.110, 0.022), stand_mat),
                ((-0.112, -0.060, 0.778), (0.036, 0.036, 0.004), stand_mat),
                ((-0.112, 0.020, 0.778), (0.036, 0.036, 0.004), stand_mat),
                ((0.22, -0.04, 0.761), (0.190, 0.190, 0.008), tray_mat),
                ((0.129, -0.04, 0.770), (0.008, 0.190, 0.018), tray_mat),
                ((0.311, -0.04, 0.770), (0.008, 0.190, 0.018), tray_mat),
                ((0.22, 0.051, 0.770), (0.190, 0.008, 0.018), tray_mat),
            ):
                scene.add_entity(
                    material=mat_b,
                    morph=gs.morphs.Box(pos=pos_b, size=sz_b, fixed=True),
                )
            rest_v, tris, elem_idx = load_cloth_mesh_from_package(pkg_dir, "pilot_flexible_wiping_cloth_01")
            obj_path = _write_obj(td_path / "cloth_rest.obj", rest_v, tris)
            cloth = scene.add_entity(
                material=gs.materials.PBD.Cloth(
                    rho=0.25,
                    static_friction=0.0 if zero_friction else 1.8,
                    kinetic_friction=0.0 if zero_friction else 1.5,
                    stretch_compliance=1e-6,
                    bending_compliance=2e-5,
                    stretch_relaxation=0.4,
                    bending_relaxation=0.3,
                    air_resistance=0.015,
                ),
                morph=gs.morphs.Mesh(file=str(obj_path), pos=(-0.13 + dx, -0.02 + dy, 0.761)),
            )
            pastry_box = None
            mpm_pastry = None
            pad_size_xyz = (0.036, 0.012, 0.036)

        left_pad = scene.add_entity(
            material=pad_mat,
            morph=gs.morphs.Box(pos=tuple(lp0.tolist()), size=pad_size_xyz, fixed=False),
        )
        right_pad = scene.add_entity(
            material=pad_mat,
            morph=gs.morphs.Box(pos=tuple(rp0.tolist()), size=pad_size_xyz, fixed=False),
        )
        scene.build()

    left_pad.set_pos(lp0.tolist(), zero_velocity=True, relative=False)
    left_pad.set_quat(q0_wxyz.tolist(), zero_velocity=True, relative=False)
    right_pad.set_pos(rp0.tolist(), zero_velocity=True, relative=False)
    right_pad.set_quat(q0_wxyz.tolist(), zero_velocity=True, relative=False)

    if mpm_pastry is not None:
        p_start = int(getattr(mpm_pastry, "particle_start", mpm_pastry._particle_start))
        p_count = int(getattr(mpm_pastry, "n_particles", mpm_pastry._n_particles))
        p_end = p_start + p_count

        def _read_pts() -> np.ndarray:
            t = mpm_pastry.get_particles_pos()
            if t.ndim == 3:
                t = t[0]
            return t.detach().cpu().numpy().astype(np.float64)

        def _read_F() -> np.ndarray:
            f_all = qd_to_torch(mpm_pastry._solver.mpm_state.particles.F, copy=True)
            return f_all[p_start:p_end, 0].detach().cpu().numpy().astype(np.float64)
    else:
        assert cloth is not None
        raw_edges = getattr(cloth, "edges", None)
        if raw_edges is not None and len(raw_edges) > 0:
            elem_idx = np.asarray(raw_edges, dtype=np.int64).reshape(-1, 2)

        def _read_pts() -> np.ndarray:
            t = cloth.get_particles_pos()
            if t.ndim == 3:
                t = t[0]
            return t.detach().cpu().numpy().astype(np.float64)

        def _read_F() -> np.ndarray | None:
            return None

    rest_world = _read_pts().copy()
    total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)

    qpos_hist = np.zeros((total_steps, 9), dtype=np.float64)
    act_hist = np.zeros((total_steps, 9), dtype=np.float64)
    ee_pos_hist = np.zeros((total_steps, 3), dtype=np.float64)
    ee_quat_hist = np.zeros((total_steps, 4), dtype=np.float64)
    tac_hist = np.zeros((total_steps, 2), dtype=np.float64)
    wrench_hist = np.zeros((total_steps, 6), dtype=np.float64)
    pts_hist = np.zeros((total_steps, len(rest_world), 3), dtype=np.float64)
    f_hist = np.zeros((total_steps, len(rest_world), 3, 3), dtype=np.float32) if mpm_pastry is not None else None
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
    prev_hr = hr0.copy()
    prev_fl = float(first_act[7])
    mu_coulomb = 0.0 if zero_friction else 3.5
    m_cloth_p = 0.0064 / max(len(rest_world), 1)

    for k in range(total_steps):
        cur_pts = _read_pts()
        ee_p, q_wxyz, lp_std, rp_std, hand_r = panda_pad_world_poses(q_cur)
        area_s = 0.0045 if mpm_pastry is not None else 0.0030
        fn_l = _pad_particle_normal_force(cur_pts, lp_std, hand_r, (0.018, 0.006, 0.018), e_eff, area_s)
        fn_r = _pad_particle_normal_force(cur_pts, rp_std, hand_r, (0.018, 0.006, 0.018), e_eff, area_s)
        if mpm_pastry is None and ee_p[2] < 0.764:
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
            if mpm_pastry is not None:
                _, q_s_wxyz, lp_s, rp_s, _ = _cpic_pad_poses(q_sub, hy_pad)
                v_l = (lp_s - prev_lp) / sim_dt
                v_r = (rp_s - prev_rp) / sim_dt
                prev_lp = lp_s
                prev_rp = rp_s
                left_pad.set_pos(lp_s.tolist(), zero_velocity=False, relative=False)
                left_pad.set_quat(q_s_wxyz.tolist(), zero_velocity=False, relative=False)
                left_pad.set_dofs_velocity(np.array([v_l[0], v_l[1], v_l[2], 0.0, 0.0, 0.0], dtype=np.float32))
                right_pad.set_pos(rp_s.tolist(), zero_velocity=False, relative=False)
                right_pad.set_quat(q_s_wxyz.tolist(), zero_velocity=False, relative=False)
                right_pad.set_dofs_velocity(np.array([v_r[0], v_r[1], v_r[2], 0.0, 0.0, 0.0], dtype=np.float32))
                scene.step()
            else:
                assert cloth is not None
                _, q_s_wxyz, lp_s, rp_s, hand_r_s = panda_pad_world_poses(q_sub)
                v_l = (lp_s - prev_lp) / sim_dt
                v_r = (rp_s - prev_rp) / sim_dt
                pts_before = _read_pts()
                mid_prev = 0.5 * (prev_lp + prev_rp)
                rel_prev = (pts_before - mid_prev[None, :]) @ prev_hr
                fl_s, fr_s = float(q_sub[7]), float(q_sub[8])
                hx_p, hz_p, r_eff = 0.022, 0.022, 0.0125
                pen_prev_l = np.maximum(0.0, r_eff - (prev_fl - rel_prev[:, 1]))
                pen_prev_r = np.maximum(0.0, r_eff - (rel_prev[:, 1] + prev_fl))
                in_prev = (
                    (np.abs(rel_prev[:, 0]) <= hx_p)
                    & (np.abs(rel_prev[:, 2]) <= hz_p)
                    & (np.minimum(pen_prev_l, pen_prev_r) > 0.0)
                )
                prev_lp, prev_rp, prev_hr, prev_fl = lp_s, rp_s, hand_r_s, fl_s
                left_pad.set_pos(lp_s.tolist(), zero_velocity=False, relative=False)
                left_pad.set_quat(q_s_wxyz.tolist(), zero_velocity=False, relative=False)
                right_pad.set_pos(rp_s.tolist(), zero_velocity=False, relative=False)
                right_pad.set_quat(q_s_wxyz.tolist(), zero_velocity=False, relative=False)
                scene.step()
                pts_after = _read_pts()
                mid_cur = 0.5 * (lp_s + rp_s)
                rel_cur = (pts_after - mid_cur[None, :]) @ hand_r_s
                in_cur = (np.abs(rel_cur[:, 0]) <= hx_p) & (np.abs(rel_cur[:, 2]) <= hz_p)
                # Solid steel jaw normal impenetrability: particles pinched at start of substep cannot tunnel through pads when fl_s < r_eff
                can_pinch = (fl_s < r_eff - 0.001) & (fr_s < r_eff - 0.001)
                active = can_pinch & (in_prev | (in_cur & (np.abs(rel_cur[:, 1]) < (r_eff - fl_s))))
                if np.any(active):
                    n_ax = hand_r_s[:, 1]
                    max_y = max(0.0005, fl_s - 0.0065)
                    clamped_y = np.clip(rel_cur[:, 1], -max_y, max_y)
                    pts_clamped = pts_after + (clamped_y - rel_cur[:, 1])[:, None] * n_ax[None, :]
                    fn_pinch = 2.0 * 320.0 * max(0.0, r_eff - fl_s)
                    v_pad = 0.5 * (v_l + v_r)
                    pos_stick = pts_before + v_pad[None, :] * sim_dt
                    drift = pos_stick - pts_clamped
                    drift_t = drift - (drift @ n_ax)[:, None] * n_ax[None, :]
                    d_norm = np.linalg.norm(drift_t, axis=1, keepdims=True)
                    max_hold = (mu_coulomb * fn_pinch / m_cloth_p) * (sim_dt ** 2)
                    scale = np.minimum(1.0, max_hold / np.maximum(d_norm, 1e-12))
                    pos_new = pts_after.copy()
                    pos_new[active] = pts_clamped[active] + (drift_t * scale)[active]
                    vel_t = cloth.get_particles_vel()
                    vel_cur = (vel_t[0] if vel_t.ndim == 3 else vel_t).detach().cpu().numpy().astype(np.float64)
                    vel_new = vel_cur.copy()
                    vel_new[active] = (pos_new[active] - pts_before[active]) / sim_dt
                    cloth.set_particles_pos(torch.as_tensor(pos_new, dtype=gs.tc_float, device=gs.device))
                    cloth.set_velocity(torch.as_tensor(vel_new, dtype=gs.tc_float, device=gs.device))

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

        if np.all(np.isfinite(cur_pts)):
            max_z = max(max_z, float(cur_pts[:, 2].mean()))
        if mpm_pastry is not None and pastry_box is not None:
            dofs = pastry_box.get_dofs_position()
            lid_ang = float(dofs.reshape(-1)[0].detach().cpu().item()) if dofs.numel() > 0 else 0.0
            lid_hist[k] = abs(lid_ang)
            f_step = _read_F()
            assert f_step is not None
            assert f_hist is not None
            f_hist[k] = f_step.astype(np.float32)
            p_s, n_s = compute_tet_strains_from_F(f_step)
            peak_polar = max(peak_polar, p_s)
            peak_naive = max(peak_naive, n_s)
        else:
            e_s = compute_edge_stretch_from_points(rest_world, cur_pts, elem_idx)
            peak_edge = max(peak_edge, e_s)
            if np.all(np.isfinite(cur_pts)):
                com_xy = cur_pts[:, :2].mean(axis=0)
                for wi in range(3):
                    wipe_min_dists[wi] = min(wipe_min_dists[wi], float(np.linalg.norm(com_xy - wipe_wps[wi])))

    def_spec = (
        brief.objects["fragile_pastry"].deformable
        if task_id == "coffee_bar_fragile_pastry_plate"
        else brief.objects["wiping_cloth"].deformable
    )
    assert def_spec is not None

    finite_ok = bool(np.all(np.isfinite(pts_hist)))
    fin_com = pts_hist[-1].mean(axis=0) if finite_ok else np.array([np.nan, np.nan, np.nan])
    peak_force_n = float(np.max(np.abs(tac_hist))) if len(tac_hist) else 0.0
    if mpm_pastry is not None:
        target_xy = np.array([0.18, -0.02], dtype=np.float64)
        final_dist = float(np.linalg.norm(fin_com[:2] - target_xy)) if finite_ok else 1.0
        max_lid = float(lid_hist.max())
        geom_ok = bool(finite_ok and max_lid >= 1.10 and max_z >= 0.85 and final_dist <= 0.085)
        damage = (
            1.0
            if not finite_ok
            else compute_damage_score(
                peak_polar,
                def_spec.strain_onset_a,
                def_spec.strain_failure_b,
                peak_force_n=peak_force_n,
                max_normal_force_n=def_spec.max_normal_force_n,
            )
        )
        intact = bool(geom_ok and damage < def_spec.damage_threshold and peak_force_n <= def_spec.max_normal_force_n)
    else:
        target_xy = np.array([0.22, -0.04], dtype=np.float64)
        final_dist = float(np.linalg.norm(fin_com[:2] - target_xy)) if finite_ok else 1.0
        geom_ok = bool(finite_ok and np.all(wipe_min_dists <= 0.065) and final_dist <= 0.095 and max_z >= 0.82)
        damage = (
            1.0
            if not finite_ok
            else compute_damage_score(
                peak_edge,
                def_spec.strain_onset_a,
                def_spec.strain_failure_b,
                peak_force_n=peak_force_n,
                max_normal_force_n=def_spec.max_normal_force_n,
            )
        )
        intact = bool(geom_ok and damage < def_spec.damage_threshold and peak_force_n <= def_spec.max_normal_force_n)

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
        "lid_angle_rad": lid_hist,
        "wipe_targets_min_dist_m": wipe_min_dists,
        "target_dist_m": np.array([final_dist], dtype=np.float64),
        "step_dt": np.array([ctrl_dt], dtype=np.float64),
    }
    if f_hist is not None:
        npz_arrays["deformable_particle_F"] = f_hist

    metrics = {
        "task_id": task_id,
        "engine_id": "genesis",
        "mode": mode,
        "zero_friction": zero_friction,
        "max_com_z_m": round(max_z, 5) if finite_ok else None,
        "final_com_xyz_m": [round(float(x), 5) for x in fin_com] if finite_ok else None,
        "final_target_dist_m": round(final_dist, 5),
        "max_lid_angle_rad": round(float(lid_hist.max()), 4) if mpm_pastry is not None else None,
        "wipe_min_dists_m": [round(float(x), 5) for x in wipe_min_dists] if mpm_pastry is None else None,
        "polar_stretch_q95": round(peak_polar, 5) if mpm_pastry is not None else None,
        "naive_f_minus_i_q95": round(peak_naive, 5) if mpm_pastry is not None else None,
        "edge_stretch_q95": round(peak_edge, 5) if mpm_pastry is None else None,
        "peak_normal_force_n": round(peak_force_n, 4),
        "damage_score": round(damage, 5),
        "task_geometric_success": geom_ok,
        "intact": intact,
    }
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
    parser = argparse.ArgumentParser(description="Genesis 1.4.7 M3 deformable rollout worker.")
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
