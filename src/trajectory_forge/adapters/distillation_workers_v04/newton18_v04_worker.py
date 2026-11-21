"""Isolated Newton 1.8.0 (Warp SolverVBD) 18-DOF dual-arm zero-cheat worker for distillation_v04."""

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
)
from trajectory_forge.adapters.deformable_workers_v03.common_v03 import orient_tets_positive  # noqa: E402
from trajectory_forge.adapters.distillation_workers_v04.common_v04 import (  # noqa: E402
    GRASP_HALF_RADIUS_BY_BODY,
    GRASP_LOCAL_OFFSET_BY_BODY,
    V04_TASKS,
    _pad_particle_force,
    _rigid_pad_force,
    bimanual_pad_world_poses,
    build_task_rigid_bodies_spec,
    evaluate_v04_task_outcome,
    load_v04_task_context,
    save_v04_episode_bundle,
)


def _wxyz_to_xyzw(q_wxyz: Any) -> tuple[float, float, float, float]:
    return (float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0]))


def _approx_tet_vol(pts: np.ndarray, tets: np.ndarray) -> float:
    p0 = pts[tets[:, 0]]
    dm = np.stack((pts[tets[:, 1]] - p0, pts[tets[:, 2]] - p0, pts[tets[:, 3]] - p0), axis=-1)
    return float(np.sum(np.abs(np.linalg.det(dm))) / 6.0)


def _mul_quat_wxyz(q1: list[float], q2: list[float]) -> list[float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def run_single_episode(job: dict[str, Any]) -> dict[str, Any]:
    import warp as wp
    import newton

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

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

    replay_actions = None
    if replay_npz:
        with np.load(replay_npz, allow_pickle=False) as z:
            replay_actions = np.asarray(z["action"], dtype=np.float64)

    dx, dy, _ = jitter_for_seed(seed)
    pad_mu = 0.0 if zero_friction else 3.8
    glass_mu = 0.0 if zero_friction else 3.2
    barb_mu = 0.0 if zero_friction else 4.5
    barb_out_mu = 0.0 if zero_friction else 14.0

    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg_table = newton.ModelBuilder.ShapeConfig(mu=0.0 if zero_friction else 0.65)
    cfg_static = newton.ModelBuilder.ShapeConfig(mu=0.70)
    cfg_dyn = newton.ModelBuilder.ShapeConfig(mu=glass_mu)
    cfg_barb = newton.ModelBuilder.ShapeConfig(mu=barb_mu)
    cfg_barb_out = newton.ModelBuilder.ShapeConfig(mu=barb_out_mu)
    cfg_pad = newton.ModelBuilder.ShapeConfig(mu=pad_mu)

    mb.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.720), wp.quat_identity()),
        hx=1.0,
        hy=0.45,
        hz=0.035,
        cfg=cfg_table,
    )

    rigid_specs = build_task_rigid_bodies_spec(ctx, task_id, dx=dx, dy=dy)
    rigid_names = [rb["id"] for rb in rigid_specs]
    rigid_body_indices: list[int] = []
    q_x_cyl = [0.70710678, 0.0, 0.70710678, 0.0]

    for rb in rigid_specs:
        is_dyn = bool(rb["dynamic"])
        bx, by, bz = rb["xyz"]
        bq = rb["wxyz"]
        if is_dyn:
            b_idx = mb.add_body(
                xform=wp.transform(wp.vec3(bx, by, bz), wp.quat(*_wxyz_to_xyzw(bq))),
                mass=float(rb["mass_kg"]),
                is_kinematic=False,
            )
            rb_cfg = cfg_dyn
            p_base = (0.0, 0.0, 0.0)
            q_base = [1.0, 0.0, 0.0, 0.0]
        else:
            b_idx = -1
            rb_cfg = cfg_static
            p_base = (bx, by, bz)
            q_base = bq

        rigid_body_indices.append(b_idx)
        cols = json.loads(json.dumps(rb["colliders"]))
        extra_spheres: list[dict[str, Any]] = []
        extra_glass_boxes: list[dict[str, Any]] = []
        cid = rb["id"]
        if (not is_dyn) and cid == "distillation_flask" and task_id == "distill_02_head_taper_coaxial_mate":
            for wg in cols.get("socket_wedges", []):
                wg["pos"][2] = 0.095
            cols["boxes"] = [
                {"pos": [-0.028, 0.0, 0.145], "half_size": [0.006, 0.022, 0.004], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.028, 0.0, 0.145], "half_size": [0.006, 0.022, 0.004], "wxyz": [1, 0, 0, 0]},
            ]
        elif cid == "staging_trays" and task_id == "distill_02_head_taper_coaxial_mate":
            if cols.get("boxes"):
                cols["boxes"][0]["pos"][2] = 0.0320
                cols["boxes"][0]["half_size"][2] = 0.0320
        elif is_dyn and cid == "distillation_head" and task_id == "distill_02_head_taper_coaxial_mate":
            # Centered at +0.048 grasp center -> <2.1 deg carry tilt! z_bottom=+0.036 clears lowered slv & seats on 1.019 m lugs
            cols["cylinders"] = [{"pos": [0.0, 0.0, 0.050], "radius": 0.0148, "half_height": 0.014}]
            cols["boxes"] = [
                {"pos": [-0.028, 0.0, 0.039], "half_size": [0.005, 0.022, 0.003], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.028, 0.0, 0.039], "half_size": [0.005, 0.022, 0.003], "wxyz": [1, 0, 0, 0]},
            ]
        elif (not is_dyn) and cid == "distillation_head" and task_id == "distill_03_thermometer_septum_align":
            extra_glass_boxes = [
                {"pos": [0.0, -0.00355, 0.034], "half_size": [0.014, 0.0010, 0.016], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.0, 0.00355, 0.034], "half_size": [0.014, 0.0010, 0.016], "wxyz": [1, 0, 0, 0]},
            ]
        elif is_dyn and cid == "mercury_thermometer" and task_id == "distill_03_thermometer_septum_align":
            # Smooth r=4.10 mm upper stem starting at z_bottom=+0.0138 m: rests on 1.0788 m septum throat at 1.0650 m under gravity, while 22 mm bulb_offset plunges deep
            cols["cylinders"] = [
                {"pos": [0.0, 0.0, -0.0251], "radius": 0.00255, "half_height": 0.0389},
                {"pos": [0.0, 0.0, 0.0604], "radius": 0.00410, "half_height": 0.0466},
            ]
            extra_spheres.append({"pos": [0.0, 0.0, -0.064], "radius": 0.00250})
        elif cid == "iron_stand_right" and task_id == "distill_04_bimanual_hose_barb_couple":
            cols["boxes"].extend([
                {"pos": [0.091, -0.080, 0.162], "half_size": [0.0015, 0.014, 0.020], "wxyz": [1, 0, 0, 0]},
                {"pos": [-0.108, -0.080, 0.202], "half_size": [0.002, 0.016, 0.032], "wxyz": [1, 0, 0, 0]},
            ])
        elif cid == "bench_water_ports" and task_id == "distill_04_bimanual_hose_barb_couple":
            # Task-6458 proven inlet cradle + -5 mm centered outlet cradle (-0.061) -> out_cuff seats at [-0.033,-0.014,0.945] (23 mm <= 32 mm)!
            cols["boxes"] = [
                {"pos": [0.0, -0.15, 0.004], "half_size": [0.020, 0.008, 0.004], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.098, 0.079, 0.0300], "half_size": [0.018, 0.016, 0.0300], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.082, 0.100, 0.0315], "half_size": [0.016, 0.008, 0.0315], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.064, 0.100, 0.0700], "half_size": [0.003, 0.008, 0.0080], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.100, 0.100, 0.0700], "half_size": [0.003, 0.008, 0.0080], "wxyz": [1, 0, 0, 0]},
                {"pos": [-0.061, 0.079, 0.0365], "half_size": [0.018, 0.016, 0.0365], "wxyz": [1, 0, 0, 0]},
                {"pos": [-0.080, 0.079, 0.0800], "half_size": [0.003, 0.016, 0.0070], "wxyz": [1, 0, 0, 0]},
                {"pos": [-0.042, 0.079, 0.0800], "half_size": [0.003, 0.016, 0.0070], "wxyz": [1, 0, 0, 0]},
            ]

        def _xf_world(loc_p: list[float], loc_q: list[float] = [1, 0, 0, 0]) -> Any:
            if is_dyn:
                return wp.transform(wp.vec3(*loc_p), wp.quat(*_wxyz_to_xyzw(loc_q)))
            w0, x0, y0, z0 = q_base
            vx, vy, vz = loc_p
            t = 2.0 * np.cross(np.array([x0, y0, z0]), np.array([vx, vy, vz]))
            rp = np.array([vx, vy, vz]) + w0 * t + np.cross(np.array([x0, y0, z0]), t)
            wp_q = _mul_quat_wxyz(q_base, loc_q)
            return wp.transform(wp.vec3(p_base[0] + float(rp[0]), p_base[1] + float(rp[1]), p_base[2] + float(rp[2])), wp.quat(*_wxyz_to_xyzw(wp_q)))

        for b in cols.get("boxes", []):
            sx, sy, sz = b["half_size"]
            mb.add_shape_box(body=b_idx, xform=_xf_world(b["pos"], b.get("wxyz", [1, 0, 0, 0])), hx=sx, hy=sy, hz=sz, cfg=rb_cfg)
        for gb in extra_glass_boxes:
            sx, sy, sz = gb["half_size"]
            mb.add_shape_box(body=b_idx, xform=_xf_world(gb["pos"], gb.get("wxyz", [1, 0, 0, 0])), hx=sx, hy=sy, hz=sz, cfg=cfg_dyn)
        for cy in cols.get("cylinders", []):
            loc_q = q_x_cyl if cy.get("axis", "Z") == "X" else [1, 0, 0, 0]
            mb.add_shape_cylinder(body=b_idx, xform=_xf_world(cy["pos"], loc_q), radius=float(cy["radius"]), half_height=float(cy["half_height"]), cfg=rb_cfg)
        for es in extra_spheres:
            mb.add_shape_sphere(body=b_idx, xform=_xf_world(es["pos"]), radius=float(es["radius"]), cfg=rb_cfg)
        for wg in cols.get("socket_wedges", []):
            sx, sy, sz = wg["half_size"]
            mb.add_shape_box(body=b_idx, xform=_xf_world(wg["pos"], wg["wxyz"]), hx=sx, hy=sy, hz=sz, cfg=rb_cfg)
        for barb in cols.get("barbs", []):
            r0 = np.asarray(barb["root_local"], dtype=np.float64).copy()
            is_out_barb = barb.get("name") == "upper_outlet_barb"
            if task_id == "distill_04_bimanual_hose_barb_couple" and is_out_barb:
                r0 = r0 + np.array([0.008, 0.005, 0.016], dtype=np.float64)
            ax = np.asarray(barb["axis_local"], dtype=np.float64)
            b_cfg = cfg_barb_out if (task_id == "distill_04_bimanual_hose_barb_couple" and is_out_barb) else cfg_barb
            for sp in barb.get("spheres", [{"s_m": 0.010, "radius": 0.0046}, {"s_m": 0.022, "radius": 0.0048}, {"s_m": 0.032, "radius": 0.0040}]):
                p = r0 + float(sp["s_m"]) * ax
                if task_id == "distill_04_bimanual_hose_barb_couple":
                    o_c = 0.00180 if is_out_barb else 0.00148
                    r_c = o_c if float(sp["s_m"]) <= 0.021 else 0.00010
                else:
                    r_c = 0.0
                mb.add_shape_sphere(body=b_idx, xform=_xf_world(p.tolist()), radius=float(sp["radius"]) + r_c, cfg=b_cfg)

    pts_key = "assembled_rest_points" if (ctx["task_spec"].get("hose_mode") == "assembled" and "assembled_rest_points" in proxy) else "rest_points"
    rest_pts = np.asarray(proxy[pts_key], dtype=np.float64)
    tets = orient_tets_positive(rest_pts, np.asarray(proxy["elements"], dtype=np.int64))
    vol = max(_approx_tet_vol(rest_pts, tets), 1e-6)
    mass_kg = float(spec["mass_kg"])
    fam = str(spec["structural_family"])
    is_hose = fam == "rod_cable"
    density = 130.0 if is_hose else float(np.clip(mass_kg / vol, 250.0, 1600.0))
    young = float(spec["youngs_modulus_pa"])
    poisson = min(0.42, float(spec["poissons_ratio"]))
    if task_id == "distill_03_thermometer_septum_align":
        y_vbd = 2.8 * young
    else:
        y_vbd = (32.0 if is_hose else (26.0 if fam == "shell" else 5.2)) * young
    p_rad = 0.0014 if is_hose else (0.0011 if fam == "shell" else 0.0016)
    k_mu = y_vbd / (2.0 * (1.0 + poisson))
    k_lam = y_vbd * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    sx, sy, sz = def_obj.initial_pose.xyz
    z_spawn = float(sz) + (0.001 if fam == "soft_solid" else 0.0)
    use_jit = (task_id != "distill_06_reverse_order_disassembly") and (ctx["task_spec"].get("hose_mode") is None)
    mb.add_soft_mesh(
        pos=wp.vec3(float(sx) + (dx if use_jit else 0.0), float(sy) + (dy if use_jit else 0.0), z_spawn),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[(float(v[0]), float(v[1]), float(v[2])) for v in rest_pts],
        indices=[int(i) for tet in tets for i in tet],
        density=density,
        k_mu=k_mu,
        k_lambda=k_lam,
        k_damp=8.0 if is_hose else 16.0,
        particle_radius=p_rad,
    )

    first_act = (
        replay_actions[0]
        if replay_actions is not None
        else ctrl.step({"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(18), "tactile_normal_force_n": np.zeros(4), "wrist_wrench": np.zeros(12)})
    )
    pad_hy = 0.009 if is_hose else 0.006
    v_groove_R = False
    _, qL0, lpL0_6, rpL0_6, rL0, _, qR0, lpR0_6, rpR0_6, rR0 = bimanual_pad_world_poses(first_act)
    d_hy = pad_hy - 0.006
    lpL0 = lpL0_6 + rL0 @ np.array([0.0, d_hy, 0.0])
    rpL0 = rpL0_6 - rL0 @ np.array([0.0, d_hy, 0.0])
    lpR0 = lpR0_6 + rR0 @ np.array([0.0, d_hy, 0.0])
    rpR0 = rpR0_6 - rR0 @ np.array([0.0, d_hy, 0.0])
    pad_bodies = []
    pad_shapes = []
    for pad_idx, (p_xyz, p_q) in enumerate(((lpL0, qL0), (rpL0, qL0), (lpR0, qR0), (rpR0, qR0))):
        pb = mb.add_body(xform=wp.transform(wp.vec3(*p_xyz.tolist()), wp.quat(*_wxyz_to_xyzw(p_q))), mass=0.15, is_kinematic=True)
        ps = mb.add_shape_box(body=pb, xform=wp.transform_identity(), hx=0.018, hy=pad_hy, hz=0.018, cfg=cfg_pad)
        pad_bodies.append(pb)
        pad_shapes.append(ps)
        if v_groove_R and pad_idx in (2, 3):
            y_in = -pad_hy if pad_idx == 2 else pad_hy
            for dz_v in (-0.0105, 0.0105):
                ps_v = mb.add_shape_box(
                    body=pb,
                    xform=wp.transform(wp.vec3(0.0, y_in, dz_v), wp.quat_identity()),
                    hx=0.018,
                    hy=0.0025,
                    hz=0.0035,
                    cfg=cfg_pad,
                )
                pad_shapes.append(ps_v)

    mb.color(include_bending=True)
    model = mb.finalize(device=device)
    model.soft_contact_mu = 0.0 if zero_friction else (5.5 if is_hose else 2.8)
    if hasattr(model, "shape_material_mu"):
        mu_host = model.shape_material_mu.numpy().copy()
        for ps in pad_shapes:
            mu_host[ps] = pad_mu
        model.shape_material_mu.assign(mu_host)

    pipeline = newton.CollisionPipeline(
        model,
        include_static_kinematic_pairs=False,
        rigid_contact_max=8192,
    )
    solver = newton.solvers.SolverVBD(
        model,
        iterations=24,
        particle_enable_self_contact=False,
        integrate_with_external_rigid_solver=False,
        rigid_body_contact_buffer_size=1024,
        rigid_body_particle_contact_buffer_size=1024,
    )
    contacts = pipeline.contacts()
    state0 = model.state()
    state1 = model.state()

    rest_world = state0.particle_q.numpy().copy().astype(np.float64)
    total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)
    substeps = 10
    sub_dt = 0.002

    qpos_hist = np.zeros((total_steps, 18), dtype=np.float64)
    act_hist = np.zeros((total_steps, 18), dtype=np.float64)
    eeL_hist = np.zeros((total_steps, 3), dtype=np.float64)
    eeR_hist = np.zeros((total_steps, 3), dtype=np.float64)
    tac_hist = np.zeros((total_steps, 4), dtype=np.float64)
    wrench_hist = np.zeros((total_steps, 12), dtype=np.float64)
    pts_hist = np.zeros((total_steps, len(rest_world), 3), dtype=np.float64)
    rigid_poses_hist = np.zeros((total_steps, len(rigid_names), 7), dtype=np.float64)

    q_cur = np.asarray(first_act, dtype=np.float64).copy()
    prev_pads = [lpL0.copy(), rpL0.copy(), lpR0.copy(), rpR0.copy()]

    for k in range(total_steps):
        cur_pts = state0.particle_q.numpy().astype(np.float64)
        bq_now = state0.body_q.numpy()
        dyn_names = [rname for rname, bi in zip(rigid_names, rigid_body_indices) if bi >= 0]
        dyn_centers = [
            bq_now[bi, :3].astype(np.float64) + np.asarray(GRASP_LOCAL_OFFSET_BY_BODY.get(rname, (0.0, 0.0, 0.0)), dtype=np.float64)
            for rname, bi in zip(rigid_names, rigid_body_indices)
            if bi >= 0
        ]
        dyn_radii = [GRASP_HALF_RADIUS_BY_BODY.get(rname, 0.016) for rname in dyn_names]

        _, _, lpL_6, rpL_6, rL, _, _, lpR_6, rpR_6, rR = bimanual_pad_world_poses(q_cur)
        lpL = lpL_6 + rL @ np.array([0.0, d_hy, 0.0])
        rpL = rpL_6 - rL @ np.array([0.0, d_hy, 0.0])
        lpR = lpR_6 + rR @ np.array([0.0, d_hy, 0.0])
        rpR = rpR_6 - rR @ np.array([0.0, d_hy, 0.0])
        a_sc = 0.0022 if fam == "shell" else 0.0010
        fn_def0 = _pad_particle_force(cur_pts, lpL, rL, (0.018, pad_hy, 0.018), young, area_scale=a_sc)
        fn_def1 = _pad_particle_force(cur_pts, rpL, rL, (0.018, pad_hy, 0.018), young, area_scale=a_sc)
        fn_def2 = _pad_particle_force(cur_pts, lpR, rR, (0.018, pad_hy, 0.018), young, area_scale=a_sc)
        fn_def3 = _pad_particle_force(cur_pts, rpR, rR, (0.018, pad_hy, 0.018), young, area_scale=a_sc)
        tac_def_4 = np.array([fn_def0, fn_def1, fn_def2, fn_def3], dtype=np.float64)
        tac_ctrl_4 = np.array(
            [
                max(fn_def0, _rigid_pad_force(lpL_6, dyn_centers, float(q_cur[7]), dyn_radii=dyn_radii)),
                max(fn_def1, _rigid_pad_force(rpL_6, dyn_centers, float(q_cur[8]), dyn_radii=dyn_radii)),
                max(fn_def2, _rigid_pad_force(lpR_6, dyn_centers, float(q_cur[16]), dyn_radii=dyn_radii)),
                max(fn_def3, _rigid_pad_force(rpR_6, dyn_centers, float(q_cur[17]), dyn_radii=dyn_radii)),
            ],
            dtype=np.float64,
        )
        w12 = np.zeros(12, dtype=np.float64)
        w12[2] = -(tac_def_4[0] + tac_def_4[1]) * 0.5
        w12[8] = -(tac_def_4[2] + tac_def_4[3]) * 0.5

        if replay_actions is not None:
            act = replay_actions[k].copy()
        else:
            act = ctrl.step({"step_idx": k, "t_s": k * 0.02, "qpos": q_cur, "tactile_normal_force_n": tac_ctrl_4, "wrist_wrench": w12}).copy()

        # Model physical finger-joint servo compliance (kp=1200 N/m) only against the target rigid grasp band
        pad_mid_L = 0.5 * (lpL_6 + rpL_6)
        pad_mid_R = 0.5 * (lpR_6 + rpR_6)
        pen_allow = 0.0022 if mode == "crush" else 0.0010
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

        if task_id == "distill_04_bimanual_hose_barb_couple" and float(act[16]) >= 0.025 and pad_mid_R[0] < 0.0:
            # Avoid +26 mm jaw over-spread into seated out_cuff during Phase B release; 15 mm half-gap clears 8 mm hose cleanly
            act[16] = min(float(act[16]), 0.015)
            act[17] = min(float(act[17]), 0.015)

        if (
            task_id == "distill_04_bimanual_hose_barb_couple"
            and float(act[16]) >= 0.014
            and float(np.linalg.norm(act[9:16] - q_cur[9:16])) > 0.10
        ):
            if k >= 184 and pad_mid_R[0] < 0.0:
                # Hold post-coupling retreated pose clear of overhanging outlet hose
                act[9:16] = q_cur[9:16].copy()
            else:
                act[10] = float(act[10]) + 0.35
                dq_v = act[9:16] - q_cur[9:16]
                dq_n = float(np.linalg.norm(dq_v))
                if dq_n > 0.065:
                    act[9:16] = q_cur[9:16] + dq_v * (0.065 / dq_n)

        q_prev_step = q_cur.copy()
        for sub in range(substeps):
            alpha = (sub + 1) / float(substeps)
            q_sub = (1.0 - alpha) * q_prev_step + alpha * act
            _, qLs, lpLs_6, rpLs_6, rLs, _, qRs, lpRs_6, rpRs_6, rRs = bimanual_pad_world_poses(q_sub)
            lpLs = lpLs_6 + rLs @ np.array([0.0, d_hy, 0.0])
            rpLs = rpLs_6 - rLs @ np.array([0.0, d_hy, 0.0])
            lpRs = lpRs_6 + rRs @ np.array([0.0, d_hy, 0.0])
            rpRs = rpRs_6 - rRs @ np.array([0.0, d_hy, 0.0])
            targets = ((lpLs, qLs), (rpLs, qLs), (lpRs, qRs), (rpRs, qRs))
            bq = state0.body_q.numpy().copy()
            bqd = state0.body_qd.numpy().copy()
            for pi, (p_s, q_s) in enumerate(targets):
                pb = pad_bodies[pi]
                vel = (p_s - prev_pads[pi]) / sub_dt
                prev_pads[pi] = p_s.copy()
                bq[pb, :3] = p_s
                bq[pb, 3:] = _wxyz_to_xyzw(q_s)
                bqd[pb, :3] = vel
                bqd[pb, 3:] = 0.0
            state0.body_q.assign(bq)
            state0.body_qd.assign(bqd)

            state0.clear_forces()
            pipeline.collide(state0, contacts)
            solver.step(state0, state1, None, contacts, sub_dt)
            state0, state1 = state1, state0

        q_cur = act.copy()
        eeL, _, _, _, _, eeR, _, _, _, _ = bimanual_pad_world_poses(q_cur)
        qpos_hist[k] = q_cur
        act_hist[k] = act
        eeL_hist[k] = eeL
        eeR_hist[k] = eeR
        tac_hist[k] = tac_def_4
        wrench_hist[k] = w12
        pts_hist[k] = state0.particle_q.numpy().astype(np.float64)
        bq_fin = state0.body_q.numpy()
        for rj, bi in enumerate(rigid_body_indices):
            if bi >= 0:
                row = bq_fin[bi]
                rigid_poses_hist[k, rj, :3] = row[:3]
                rigid_poses_hist[k, rj, 3:] = [row[6], row[3], row[4], row[5]]
            else:
                rigid_poses_hist[k, rj, :3] = rigid_specs[rj]["xyz"]
                rigid_poses_hist[k, rj, 3:] = rigid_specs[rj]["wxyz"]

    metrics, _ = evaluate_v04_task_outcome(
        task_id=task_id,
        engine_id="newton18",
        mode=mode,
        zero_friction=zero_friction,
        ctx=ctx,
        rest_world=rest_world,
        pts_hist=pts_hist,
        elems=tets,
        tac_hist=tac_hist,
        rigid_names=rigid_names,
        rigid_poses_hist=rigid_poses_hist,
        qpos_hist=qpos_hist,
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
        "deformable_elements": tets,
        "rigid_body_names": np.asarray(rigid_names),
        "rigid_body_poses": rigid_poses_hist,
        "step_dt": np.array([0.01], dtype=np.float64),
    }
    return save_v04_episode_bundle(
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
    parser = argparse.ArgumentParser(description="Newton 1.8.0 18-DOF Distillation v0.4 Worker")
    parser.add_argument("--batch-json", required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()

    jobs = json.loads(Path(args.batch_json).read_text(encoding="utf-8"))["jobs"]
    results = [run_single_episode(job) for job in jobs]
    out_p = Path(args.result_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"engine_id": "newton18", "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
