"""Isolated Genesis 1.4.7 18-DOF MPM+CPIC zero-cheat worker for distillation_v04 (8% VRAM cap)."""

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
    jitter_for_seed,
)
from trajectory_forge.adapters.legacy.distillation_workers_v04.common_v04 import (  # noqa: E402
    GRASP_HALF_RADIUS_BY_BODY,
    GRASP_LOCAL_OFFSET_BY_BODY,
    _pad_particle_force,
    _rigid_pad_force,
    bimanual_pad_world_poses,
    build_task_rigid_bodies_spec,
    evaluate_v04_task_outcome,
    load_v04_task_context,
    save_v04_episode_bundle,
)
from trajectory_forge.sdk.panda_kinematics import forward_kinematics_panda, quat_wxyz_to_matrix  # noqa: E402


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> Path:
    lines = ["# genesis v04 mesh"]
    for v in verts:
        lines.append(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}")
    for f in faces:
        lines.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _append_box_verts_tris(verts: list, tris: list, center: list[float], half_size: list[float], wxyz: list[float]) -> None:
    hx, hy, hz = half_size
    c_local = np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    rot = quat_wxyz_to_matrix(wxyz)
    pts = (c_local @ rot.T) + np.asarray(center, dtype=np.float64)[None, :]
    b = len(verts)
    for p in pts:
        verts.append((float(p[0]), float(p[1]), float(p[2])))
    for t in ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)):
        tris.append((b + t[0], b + t[1], b + t[2]))


def _append_cyl_verts_tris(verts: list, tris: list, center: list[float], r: float, hh: float, axis: str = "Z", n_seg: int = 12) -> None:
    th = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    b = len(verts)
    cx, cy, cz = center
    for sgn in (-1.0, 1.0):
        for ang in th:
            c, s = math.cos(float(ang)) * r, math.sin(float(ang)) * r
            if axis == "X":
                verts.append((cx + sgn * hh, cy + c, cz + s))
            else:
                verts.append((cx + c, cy + s, cz + sgn * hh))
    ci_b, ci_t = len(verts), len(verts) + 1
    if axis == "X":
        verts.append((cx - hh, cy, cz))
        verts.append((cx + hh, cy, cz))
    else:
        verts.append((cx, cy, cz - hh))
        verts.append((cx, cy, cz + hh))
    for j in range(n_seg):
        jn = (j + 1) % n_seg
        b0, b1, t0, t1 = b + j, b + jn, b + n_seg + j, b + n_seg + jn
        tris.extend([(b0, b1, t1), (b0, t1, t0), (ci_b, b1, b0), (ci_t, t0, t1)])


def _compound_colliders_to_obj(path: Path, cols: dict[str, Any]) -> Path:
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    for bx in cols.get("boxes", []):
        _append_box_verts_tris(verts, tris, bx["pos"], bx["half_size"], bx.get("wxyz", [1, 0, 0, 0]))
    for cy in cols.get("cylinders", []):
        _append_cyl_verts_tris(verts, tris, cy["pos"], float(cy["radius"]), float(cy["half_height"]), str(cy.get("axis", "Z")))
    for wg in cols.get("socket_wedges", []):
        _append_box_verts_tris(verts, tris, wg["pos"], wg["half_size"], wg["wxyz"])
    for barb in cols.get("barbs", []):
        r0 = np.asarray(barb["root_local"], dtype=np.float64)
        ax = np.asarray(barb["axis_local"], dtype=np.float64)
        for sp in barb.get("spheres", [{"s_m": 0.012, "radius": 0.0046}, {"s_m": 0.028, "radius": 0.0044}]):
            p = (r0 + float(sp["s_m"]) * ax).tolist()
            rad = float(sp["radius"])
            _append_box_verts_tris(verts, tris, p, [rad, rad, rad], [1, 0, 0, 0])
    if not verts:
        _append_box_verts_tris(verts, tris, [0, 0, 0], [0.004, 0.004, 0.004], [1, 0, 0, 0])
    return _write_obj(path, np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64))


def _bimanual_cpic_pads(q18: np.ndarray, hy: float = 0.012) -> tuple[Any, ...]:
    from trajectory_forge.adapters.legacy.distillation_workers_v04.common_v04 import ARM_BASE_QUAT_WXYZ, LEFT_ARM_BASE_POS, RIGHT_ARM_BASE_POS

    eeL, rL, qL = forward_kinematics_panda(q18[0:7], base_pos=LEFT_ARM_BASE_POS, base_quat_wxyz=ARM_BASE_QUAT_WXYZ)
    lpL = eeL + rL @ np.array([0.0, hy + float(q18[7]), -0.0106], dtype=np.float64)
    rpL = eeL + rL @ np.array([0.0, -(hy + float(q18[8])), -0.0106], dtype=np.float64)
    eeR, rR, qR = forward_kinematics_panda(q18[9:16], base_pos=RIGHT_ARM_BASE_POS, base_quat_wxyz=ARM_BASE_QUAT_WXYZ)
    lpR = eeR + rR @ np.array([0.0, hy + float(q18[16]), -0.0106], dtype=np.float64)
    rpR = eeR + rR @ np.array([0.0, -(hy + float(q18[17])), -0.0106], dtype=np.float64)
    return qL, lpL, rpL, qR, lpR, rpR


def run_job(gs: Any, qd_to_torch: Any, job: dict[str, Any]) -> dict[str, Any]:
    pkg_dir = Path(job["package_dir"])
    task_id = str(job["task_id"])
    mode = str(job["mode"])
    seed = int(job.get("seed", 0))
    zero_friction = bool(job.get("zero_friction", False))
    replay_npz = job.get("replay_npz")
    skill_override = job.get("skill_params_override")
    output_dir = Path(job["output_dir"])
    episode_id = str(job["episode_id"])

    ctx = load_v04_task_context(pkg_dir, task_id, mode, skill_params_override=skill_override)
    brief = ctx["brief"]
    ctrl = ctx["controller"]
    proxy = ctx["proxy"]
    spec = ctx["m1_manifest"]["source"]
    def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
    young = float(spec["youngs_modulus_pa"])

    replay_actions = None
    if replay_npz:
        with np.load(replay_npz, allow_pickle=False) as z:
            replay_actions = np.asarray(z["action"], dtype=np.float64)

    dx, dy, _ = jitter_for_seed(seed)
    ctrl_dt = 0.01
    sim_dt = 0.002
    substeps_per_ctrl = 5
    hy_pad = 0.010
    pad_size = (0.036, 2.0 * hy_pad, 0.036)

    pad_mat = gs.materials.Rigid(
        rho=1.0e8,
        friction=0.01 if zero_friction else 3.8,
        coup_friction=0.0 if zero_friction else 4.2,
        coup_softness=0.0028,
        sdf_cell_size=0.002,
        gravity_compensation=1.0,
    )
    static_mat = gs.materials.Rigid(rho=2500.0, friction=0.70, coup_friction=0.70, coup_softness=0.002, sdf_cell_size=0.004)
    glass_mat = gs.materials.Rigid(
        rho=2200.0,
        friction=0.01 if zero_friction else 2.2,
        coup_friction=0.0 if zero_friction else 2.6,
        coup_softness=0.0025,
        sdf_cell_size=0.003,
    )

    first_act = (
        replay_actions[0]
        if replay_actions is not None
        else ctrl.step({"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(18), "tactile_normal_force_n": np.zeros(4), "wrist_wrench": np.zeros(12)})
    )
    qL0, lpL0, rpL0, qR0, lpR0, rpR0 = _bimanual_cpic_pads(first_act, hy_pad)

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=sim_dt, substeps=8),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.42, -0.36, 0.70),
                upper_bound=(0.42, 0.26, 1.24),
                grid_density=80,
                enable_CPIC=True,
            ),
            show_viewer=False,
        )
        scene.add_entity(material=static_mat, morph=gs.morphs.Box(pos=(0.0, 0.0, 0.735), size=(2.0, 0.90, 0.04), fixed=True))

        rigid_specs = build_task_rigid_bodies_spec(ctx, task_id, dx=dx, dy=dy)
        rigid_names = [rb["id"] for rb in rigid_specs]
        rigid_ents: list[tuple[bool, Any, list[float], list[float]]] = []
        for rb in rigid_specs:
            is_dyn = bool(rb["dynamic"])
            obj_f = _compound_colliders_to_obj(td_p / f"{rb['id']}.obj", rb["colliders"])
            ent = scene.add_entity(
                material=glass_mat if is_dyn else static_mat,
                morph=gs.morphs.Mesh(
                    file=str(obj_f),
                    pos=tuple(rb["xyz"]),
                    quat=tuple(rb["wxyz"]),
                    fixed=not is_dyn,
                    convexify=False,
                    decimate=False,
                    align=False,
                    file_meshes_are_zup=True,
                ),
            )
            rigid_ents.append((is_dyn, ent, rb["xyz"], rb["wxyz"]))

        pts_key = "assembled_rest_points" if (ctx["task_spec"].get("hose_mode") == "assembled" and "assembled_rest_points" in proxy) else "rest_points"
        rest_v = np.asarray(proxy[pts_key], dtype=np.float64)
        tris = np.asarray(proxy["triangles"], dtype=np.int64)
        def_obj_path = _write_obj(td_p / "deformable_rest.obj", rest_v, tris)
        sx, sy, sz = def_obj.initial_pose.xyz
        use_jit = (task_id != "distill_06_reverse_order_disassembly") and (ctx["task_spec"].get("hose_mode") is None)

        mpm_ent = scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(
                E=max(75000.0, 1.5 * young),
                nu=0.30,
                rho=950.0,
                yield_lower=0.52,
                yield_higher=0.65,
                use_von_mises=True,
                von_mises_yield_stress=max(32000.0, float(spec.get("yield_stress_pa") or 42000.0) * 1.4),
            ),
            morph=gs.morphs.Mesh(
                file=str(def_obj_path),
                pos=(float(sx) + (dx if use_jit else 0.0), float(sy) + (dy if use_jit else 0.0), float(sz) + 0.001),
            ),
        )

        pad_ents = [
            scene.add_entity(material=pad_mat, morph=gs.morphs.Box(pos=tuple(lpL0.tolist()), size=pad_size, fixed=False)),
            scene.add_entity(material=pad_mat, morph=gs.morphs.Box(pos=tuple(rpL0.tolist()), size=pad_size, fixed=False)),
            scene.add_entity(material=pad_mat, morph=gs.morphs.Box(pos=tuple(lpR0.tolist()), size=pad_size, fixed=False)),
            scene.add_entity(material=pad_mat, morph=gs.morphs.Box(pos=tuple(rpR0.tolist()), size=pad_size, fixed=False)),
        ]
        scene.build()

    for ent, p0, q0 in zip(pad_ents, (lpL0, rpL0, lpR0, rpR0), (qL0, qL0, qR0, qR0)):
        ent.set_pos(p0.tolist(), zero_velocity=True, relative=False)
        ent.set_quat(q0.tolist(), zero_velocity=True, relative=False)

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

    rest_world = _read_pts().copy()
    total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)
    qpos_hist = np.zeros((total_steps, 18), dtype=np.float64)
    act_hist = np.zeros((total_steps, 18), dtype=np.float64)
    eeL_hist = np.zeros((total_steps, 3), dtype=np.float64)
    eeR_hist = np.zeros((total_steps, 3), dtype=np.float64)
    tac_hist = np.zeros((total_steps, 4), dtype=np.float64)
    wrench_hist = np.zeros((total_steps, 12), dtype=np.float64)
    pts_hist = np.zeros((total_steps, len(rest_world), 3), dtype=np.float64)
    f_hist = np.zeros((total_steps, len(rest_world), 3, 3), dtype=np.float32)
    rigid_poses_hist = np.zeros((total_steps, len(rigid_names), 7), dtype=np.float64)

    q_cur = np.asarray(first_act, dtype=np.float64).copy()
    prev_pads = [lpL0.copy(), rpL0.copy(), lpR0.copy(), rpR0.copy()]

    for k in range(total_steps):
        cur_pts = _read_pts()
        dyn_names = [rname for rname, (is_dyn, _, _, _) in zip(rigid_names, rigid_ents) if is_dyn]
        dyn_centers = [
            ent.get_pos().detach().cpu().numpy().reshape(-1).astype(np.float64)
            + np.asarray(GRASP_LOCAL_OFFSET_BY_BODY.get(rname, (0.0, 0.0, 0.0)), dtype=np.float64)
            for rname, (is_dyn, ent, _, _) in zip(rigid_names, rigid_ents)
            if is_dyn
        ]
        dyn_radii = [GRASP_HALF_RADIUS_BY_BODY.get(rname, 0.016) for rname in dyn_names]
        _, _, lpL, rpL, rL, _, _, lpR, rpR, rR = bimanual_pad_world_poses(q_cur)
        fn_def0 = _pad_particle_force(cur_pts, lpL, rL, (0.018, 0.006, 0.018), young)
        fn_def1 = _pad_particle_force(cur_pts, rpL, rL, (0.018, 0.006, 0.018), young)
        fn_def2 = _pad_particle_force(cur_pts, lpR, rR, (0.018, 0.006, 0.018), young)
        fn_def3 = _pad_particle_force(cur_pts, rpR, rR, (0.018, 0.006, 0.018), young)
        tac_def_4 = np.array([fn_def0, fn_def1, fn_def2, fn_def3], dtype=np.float64)
        tac_ctrl_4 = np.array(
            [
                max(fn_def0, _rigid_pad_force(lpL, dyn_centers, float(q_cur[7]), dyn_radii=dyn_radii)),
                max(fn_def1, _rigid_pad_force(rpL, dyn_centers, float(q_cur[8]), dyn_radii=dyn_radii)),
                max(fn_def2, _rigid_pad_force(lpR, dyn_centers, float(q_cur[16]), dyn_radii=dyn_radii)),
                max(fn_def3, _rigid_pad_force(rpR, dyn_centers, float(q_cur[17]), dyn_radii=dyn_radii)),
            ],
            dtype=np.float64,
        )
        w12 = np.zeros(12, dtype=np.float64)
        w12[2] = -(tac_def_4[0] + tac_def_4[1]) * 0.5
        w12[8] = -(tac_def_4[2] + tac_def_4[3]) * 0.5

        if replay_actions is not None:
            act = replay_actions[k].copy()
        else:
            act = ctrl.step({"step_idx": k, "t_s": k * ctrl_dt, "qpos": q_cur, "tactile_normal_force_n": tac_ctrl_4, "wrist_wrench": w12}).copy()

        pad_mid_L = 0.5 * (lpL + rpL)
        pad_mid_R = 0.5 * (lpR + rpR)
        pen_allow = 0.0020 if mode == "crush" else 0.0006
        for rname, c_i, r_i in zip(dyn_names, dyn_centers, dyn_radii):
            xy_tol = 0.085 if rname == "liebig_condenser" else 0.040
            z_tol = 0.048
            if np.linalg.norm(pad_mid_L[:2] - c_i[:2]) < xy_tol and abs(float(pad_mid_L[2] - c_i[2])) < z_tol:
                if mode == "crush" or float(act[7]) >= float(r_i) - 0.0065:
                    act[7] = max(float(act[7]), float(r_i) - pen_allow)
                    act[8] = max(float(act[8]), float(r_i) - pen_allow)
            if task_id != "distill_04_bimanual_hose_barb_couple" and np.linalg.norm(pad_mid_R[:2] - c_i[:2]) < xy_tol and abs(float(pad_mid_R[2] - c_i[2])) < z_tol:
                if mode == "crush" or float(act[16]) >= float(r_i) - 0.0065:
                    act[16] = max(float(act[16]), float(r_i) - pen_allow)
                    act[17] = max(float(act[17]), float(r_i) - pen_allow)

        q_prev_step = q_cur.copy()
        for sub in range(substeps_per_ctrl):
            alpha = (sub + 1) / float(substeps_per_ctrl)
            q_sub = (1.0 - alpha) * q_prev_step + alpha * act
            qLs, lpLs, rpLs, qRs, lpRs, rpRs = _bimanual_cpic_pads(q_sub, hy_pad)
            for pi, (ent, p_s, q_s) in enumerate(zip(pad_ents, (lpLs, rpLs, lpRs, rpRs), (qLs, qLs, qRs, qRs))):
                vel = (p_s - prev_pads[pi]) / sim_dt
                prev_pads[pi] = p_s.copy()
                ent.set_pos(p_s.tolist(), zero_velocity=False, relative=False)
                ent.set_quat(q_s.tolist(), zero_velocity=False, relative=False)
                ent.set_dofs_velocity(np.array([vel[0], vel[1], vel[2], 0.0, 0.0, 0.0], dtype=np.float32))
            scene.step()

        q_cur = act.copy()
        eeL, _, _, _, _, eeR, _, _, _, _ = bimanual_pad_world_poses(q_cur)
        qpos_hist[k] = q_cur
        act_hist[k] = act
        eeL_hist[k] = eeL
        eeR_hist[k] = eeR
        tac_hist[k] = tac_def_4
        wrench_hist[k] = w12
        pts_hist[k] = _read_pts()
        f_hist[k] = _read_F().astype(np.float32)
        for rj, (is_dyn, ent, def_p, def_q) in enumerate(rigid_ents):
            if is_dyn:
                pb = ent.get_pos().detach().cpu().numpy().reshape(-1)
                qb = ent.get_quat().detach().cpu().numpy().reshape(-1)
                rigid_poses_hist[k, rj, :3] = pb[:3]
                rigid_poses_hist[k, rj, 3:] = qb[:4]
            else:
                rigid_poses_hist[k, rj, :3] = def_p
                rigid_poses_hist[k, rj, 3:] = def_q

    metrics, _ = evaluate_v04_task_outcome(
        task_id=task_id,
        engine_id="genesis",
        mode=mode,
        zero_friction=zero_friction,
        ctx=ctx,
        rest_world=rest_world,
        pts_hist=pts_hist,
        elems=np.zeros((0, 4), dtype=np.int64),
        tac_hist=tac_hist,
        rigid_names=rigid_names,
        rigid_poses_hist=rigid_poses_hist,
        qpos_hist=qpos_hist,
        f_hist=f_hist,
    )
    npz_arrays = {
        "qpos": qpos_hist,
        "action": act_hist,
        "ee_left_pos": eeL_hist,
        "ee_right_pos": eeR_hist,
        "tactile_normal_force_n": tac_hist,
        "wrist_wrench": wrench_hist,
        "deformable_rest_points": rest_world,
        "deformable_points": pts_hist,
        "deformable_elements": np.zeros((0, 4), dtype=np.int64),
        "deformable_particle_F": f_hist,
        "rigid_body_names": np.asarray(rigid_names),
        "rigid_body_poses": rigid_poses_hist,
        "step_dt": np.array([ctrl_dt], dtype=np.float64),
    }
    return save_v04_episode_bundle(
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
    parser = argparse.ArgumentParser(description="Genesis 1.4.7 18-DOF Distillation v0.4 Worker")
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
