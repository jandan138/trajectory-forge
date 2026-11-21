"""Isolated Isaac Sim 6.0.1 zero-RTX M3 v0.3 rollout worker (PhysX Volume + Surface Deformable)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("ISAACSIM_PORTABLE_ROOT", r"I:\isaacsim-data\6.0.1")
os.environ.setdefault("TEMP", r"I:\isaacsim-data\6.0.1\temp")
os.environ.setdefault("TMP", r"I:\isaacsim-data\6.0.1\temp")

_SRC_ROOT = str(Path(__file__).resolve().parents[3])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from trajectory_forge.adapters.deformable_workers.common import (  # noqa: E402
    apply_low_priority_process_policy,
    jitter_for_seed,
    save_episode_bundle,
)
from trajectory_forge.adapters.deformable_workers.isaac6_rollout_worker import (  # noqa: E402
    _apply_small_collision_offsets,
    _create_kinematic_box,
    _create_rigid_material,
    _create_static_box,
    _pad_particle_normal_force,
)
from trajectory_forge.adapters.deformable_workers_v03.common_v03 import (  # noqa: E402
    evaluate_v03_task_outcome,
    load_v03_proxy_and_brief,
    orient_tets_positive,
)
from trajectory_forge.sdk.panda_kinematics import panda_pad_world_poses  # noqa: E402


def run_job(
    sim_iface: Any,
    physx_utils: Any,
    deformableUtils: Any,
    physicsUtils: Any,
    job: dict[str, Any],
) -> dict[str, Any]:
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdUtils, Vt

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
    is_volume = family in ("soft_solid", "rod_cable")
    young = float(spec["youngs_modulus_pa"])

    ctrl_dt = 0.01
    sim_dt = 0.00125 if is_volume else 0.002
    substeps_per_ctrl = 8 if is_volume else 5

    stage = physx_utils.new_memory_stage(attach_stage=False)
    attached = False
    try:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        default_prim = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(default_prim.GetPrim())

        scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        ps_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        ps_api.CreateEnableGPUDynamicsAttr().Set(True)
        ps_api.CreateBroadphaseTypeAttr().Set("GPU")
        ps_api.GetMaxPositionIterationCountAttr().Set(16 if task_id == "office_dust_cover_peel_retrieve" else 24)

        UsdGeom.Scope.Define(stage, "/World/Materials")
        high_mu = task_id in ("office_waffle_towel_corner_fold", "lab_rubber_bulb_squeeze_seat")
        pad_mat = _create_rigid_material(
            stage,
            "/World/Materials/PadMat",
            static_friction=0.0 if zero_friction else (5.5 if high_mu else 3.6),
            dynamic_friction=0.0 if zero_friction else (5.2 if high_mu else 3.4),
            combine_mode="min" if zero_friction else "max",
        )
        tray_mat = _create_rigid_material(stage, "/World/Materials/TrayMat", static_friction=0.65, dynamic_friction=0.55)
        mat_mu_s = 0.55 if task_id == "office_waffle_towel_corner_fold" else 0.16
        mat_mat = _create_rigid_material(stage, "/World/Materials/MatMat", static_friction=mat_mu_s, dynamic_friction=mat_mu_s - 0.05 if mat_mu_s > 0.2 else mat_mu_s - 0.02, combine_mode="min")
        box_mat = _create_rigid_material(
            stage,
            "/World/Materials/BoxMat",
            static_friction=0.0 if zero_friction else 2.2,
            dynamic_friction=0.0 if zero_friction else 2.0,
            combine_mode="min" if zero_friction else "max",
        )
        top_mat = _create_rigid_material(stage, "/World/Materials/TopMat", static_friction=0.12, dynamic_friction=0.10, combine_mode="min")

        def _xyz(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
            obj = brief.objects.get(name)
            return tuple(float(v) for v in obj.initial_pose.xyz) if obj else default

        if "smooth_folding_mat" in brief.objects:
            _, _, zm = _xyz("smooth_folding_mat", (0.0, 0.02, 0.755))
            _create_static_box(stage, "/World/FoldingMat", (0.0, 0.0, zm + 0.002 - 0.020), (2.0, 0.90, 0.040), physicsUtils, mat_mat)
        else:
            _create_static_box(stage, "/World/Table", (0.0, 0.0, 0.735), (2.0, 0.90, 0.040), physicsUtils, tray_mat)
        if "baking_sheet_tray" in brief.objects:
            x, y, z = _xyz("baking_sheet_tray", (-0.18, -0.02, 0.755))
            _create_static_box(stage, "/World/TrayPlate", (x, y, z + 0.003), (0.190, 0.170, 0.006), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/TrayLiner", (x, y, z + 0.0055), (0.168, 0.148, 0.002), physicsUtils, tray_mat)
        if "tartlet_pastry_cup" in brief.objects:
            x, y, z = _xyz("tartlet_pastry_cup", (0.18, -0.02, 0.755))
            for idx_c, (rad, h, dz) in enumerate(((0.085, 0.005, 0.0025), (0.048, 0.016, 0.013), (0.043, 0.004, 0.021))):
                c = UsdGeom.Cylinder.Define(stage, f"/World/TartCup_{idx_c}")
                c.CreateAxisAttr("Z")
                c.CreateRadiusAttr(rad)
                c.CreateHeightAttr(h)
                c.AddTransformOp().Set(Gf.Transform(Gf.Vec3d(x, y, z + dz), Gf.Rotation(Gf.Quatd(1.0)), Gf.Vec3d(1.0)).GetMatrix())
                UsdPhysics.CollisionAPI.Apply(c.GetPrim())
                _apply_small_collision_offsets(c.GetPrim())
                physicsUtils.add_physics_material_to_prim(stage, c.GetPrim(), tray_mat)
        if "garment_folding_board" in brief.objects:
            x, y, z = _xyz("garment_folding_board", (0.0, 0.02, 0.755))
            _create_static_box(stage, "/World/FoldingBoard", (x + 0.045, y, z + 0.014), (0.190, 0.260, 0.028), physicsUtils, tray_mat)

        toner_prim = None
        if "toner_cartridge_box" in brief.objects:
            bx, by, bz = _xyz("toner_cartridge_box", (-0.14, 0.02, 0.757))
            tc = UsdGeom.Xform.Define(stage, "/World/TonerBox")
            tc.AddTransformOp().Set(
                Gf.Transform(Gf.Vec3d(bx + dx, by + dy, bz + 0.022), Gf.Rotation(Gf.Quatd(1.0)), Gf.Vec3d(1.0)).GetMatrix()
            )
            UsdPhysics.RigidBodyAPI.Apply(tc.GetPrim())
            UsdPhysics.MassAPI.Apply(tc.GetPrim()).CreateMassAttr().Set(0.45)
            hull = UsdGeom.Cube.Define(stage, "/World/TonerBox/Hull")
            hull.CreateSizeAttr(1.0)
            hull.AddTransformOp().Set(Gf.Transform(Gf.Vec3d(0.0, 0.0, -0.001), Gf.Rotation(Gf.Quatd(1.0)), Gf.Vec3d(0.064, 0.064, 0.042)).GetMatrix())
            UsdPhysics.CollisionAPI.Apply(hull.GetPrim())
            _apply_small_collision_offsets(hull.GetPrim(), rest_offset=0.0005, contact_offset=0.003)
            physicsUtils.add_physics_material_to_prim(stage, hull.GetPrim(), box_mat)
            cap = UsdGeom.Cube.Define(stage, "/World/TonerBox/Cap")
            cap.CreateSizeAttr(1.0)
            cap.AddTransformOp().Set(Gf.Transform(Gf.Vec3d(0.0, 0.0, 0.021), Gf.Rotation(Gf.Quatd(1.0)), Gf.Vec3d(0.063, 0.063, 0.002)).GetMatrix())
            UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
            _apply_small_collision_offsets(cap.GetPrim(), rest_offset=0.0005, contact_offset=0.003)
            physicsUtils.add_physics_material_to_prim(stage, cap.GetPrim(), top_mat)
            toner_prim = UsdGeom.Xformable(tc.GetPrim())

        if "delivery_caddy" in brief.objects:
            x, y, z = _xyz("delivery_caddy", (0.20, -0.02, 0.757))
            _create_static_box(stage, "/World/CaddyBase", (x, y, z + 0.003), (0.170, 0.170, 0.006), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/CaddyRimN", (x, y + 0.081, z + 0.010), (0.170, 0.008, 0.020), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/CaddyRimW", (x - 0.081, y, z + 0.010), (0.008, 0.170, 0.020), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/CaddyRimE", (x + 0.081, y, z + 0.010), (0.008, 0.170, 0.020), physicsUtils, tray_mat)
        if "cable_carrier_cradle" in brief.objects:
            x, y, z = _xyz("cable_carrier_cradle", (-0.16, -0.03, 0.755))
            _create_static_box(stage, "/World/CradleBase", (x, y, z + 0.004), (0.210, 0.110, 0.008), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/CradleL", (x - 0.068, y, z + 0.011), (0.024, 0.072, 0.014), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/CradleR", (x + 0.068, y, z + 0.011), (0.024, 0.072, 0.014), physicsUtils, tray_mat)
        if "smt_u_clip_block" in brief.objects:
            x, y, z = _xyz("smt_u_clip_block", (0.14, 0.04, 0.755))
            _create_static_box(stage, "/World/ClipFloor", (x, y, z + 0.004), (0.210, 0.096, 0.008), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/ClipWallS", (x, y - 0.020, z + 0.014), (0.164, 0.010, 0.024), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/ClipWallN", (x, y + 0.020, z + 0.014), (0.164, 0.010, 0.024), physicsUtils, tray_mat)
        if "pcba_header_boss" in brief.objects:
            x, y, z = _xyz("pcba_header_boss", (0.15, -0.02, 0.755))
            _create_static_box(stage, "/World/PCBABase", (x, y, z + 0.003), (0.150, 0.150, 0.006), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/HeaderShroud", (x, y, z + 0.007), (0.052, 0.052, 0.004), physicsUtils, tray_mat)
        if "tube_bench_cradle" in brief.objects:
            x, y, z = _xyz("tube_bench_cradle", (-0.17, -0.02, 0.755))
            _create_static_box(stage, "/World/TubeBase", (x, y, z + 0.004), (0.210, 0.110, 0.008), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/TubeL", (x - 0.068, y, z + 0.011), (0.028, 0.064, 0.014), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/TubeR", (x + 0.068, y, z + 0.011), (0.028, 0.064, 0.014), physicsUtils, tray_mat)
        if "manifold_fork_saddle" in brief.objects:
            x, y, z = _xyz("manifold_fork_saddle", (0.16, 0.03, 0.755))
            _create_static_box(stage, "/World/ManifoldBase", (x, y, z + 0.004), (0.210, 0.110, 0.008), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/ForkS", (x, y - 0.022, z + 0.015), (0.170, 0.010, 0.026), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/ForkN", (x, y + 0.022, z + 0.015), (0.170, 0.010, 0.026), physicsUtils, tray_mat)
        if "bulb_collar_holder" in brief.objects:
            x, y, z = _xyz("bulb_collar_holder", (0.15, -0.02, 0.755))
            _create_static_box(stage, "/World/CollarBase", (x, y, z + 0.003), (0.130, 0.130, 0.006), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/CollarN", (x, y + 0.034, z + 0.012), (0.036, 0.008, 0.012), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/CollarS", (x, y - 0.034, z + 0.012), (0.036, 0.008, 0.012), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/CollarW", (x - 0.034, y, z + 0.012), (0.008, 0.036, 0.012), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/CollarE", (x + 0.034, y, z + 0.012), (0.008, 0.036, 0.012), physicsUtils, tray_mat)

        rest_pts = np.asarray(proxy["rest_points"], dtype=np.float64).copy()
        sx, sy, sz = def_obj.initial_pose.xyz
        rest_pts[:, 0] += sx + dx
        rest_pts[:, 1] += sy + dy
        rest_pts[:, 2] += float(sz) + (0.001 if family == "soft_solid" else 0.0)

        if is_volume:
            elem_idx = orient_tets_positive(rest_pts, np.asarray(proxy["elements"], dtype=np.int64))
            tet_prim = UsdGeom.TetMesh.Define(stage, "/World/DeformableAsset")
            tet_prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in rest_pts]))
            tet_prim.CreateTetVertexIndicesAttr(
                Vt.Vec4iArray([Gf.Vec4i(int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in elem_idx])
            )
            deformableUtils.set_physics_volume_deformable_body(stage, Sdf.Path("/World/DeformableAsset"))
            _apply_small_collision_offsets(tet_prim.GetPrim(), rest_offset=0.0006, contact_offset=0.0035)
            tet_prim.GetPrim().ApplyAPI("PhysxBaseDeformableBodyAPI")
            tet_prim.GetPrim().GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(64)
            tet_prim.GetPrim().GetAttribute("physxDeformableBody:linearDamping").Set(2.2)
            def_mat_path = "/World/Materials/DeformableMat"
            deformableUtils.add_deformable_material(
                stage,
                def_mat_path,
                density=240.0 if family == "rod_cable" else 160.0,
                static_friction=0.0 if zero_friction else 3.4,
                dynamic_friction=0.0 if zero_friction else 3.2,
                youngs_modulus=max(2.0e5, 6.0 * young),
                poissons_ratio=0.30,
            )
            mat_prim = stage.GetPrimAtPath(def_mat_path)
            mat_prim.ApplyAPI("PhysxDeformableMaterialAPI")
            mat_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.08)
            physicsUtils.add_physics_material_to_prim(stage, tet_prim.GetPrim(), def_mat_path)
            body_prim = tet_prim
            metrics_elem = elem_idx
        else:
            tris = np.asarray(proxy["triangles"], dtype=np.int64)
            metrics_elem = np.asarray(proxy["edges"], dtype=np.int64)
            cloth_prim = UsdGeom.Mesh.Define(stage, "/World/DeformableAsset")
            cloth_prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in rest_pts]))
            cloth_prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
            cloth_prim.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for f in tris for i in f]))
            deformableUtils.set_physics_surface_deformable_body(stage, Sdf.Path("/World/DeformableAsset"))
            if task_id != "office_dust_cover_peel_retrieve":
                cloth_prim.GetPrim().ApplyAPI("PhysxBaseDeformableBodyAPI")
                cloth_prim.GetPrim().GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(32)
            is_towel = task_id == "office_waffle_towel_corner_fold"
            is_bulb = task_id == "lab_rubber_bulb_squeeze_seat"
            is_shell = family == "shell"
            _apply_small_collision_offsets(
                cloth_prim.GetPrim(),
                rest_offset=0.0014 if is_towel else 0.0022,
                contact_offset=0.0038 if is_towel else 0.0048,
            )
            def_mat_path = "/World/Materials/DeformableMat"
            if zero_friction:
                def_mu_s = 0.0
            elif task_id == "office_dust_cover_peel_retrieve":
                def_mu_s = 0.65
            elif is_towel or is_bulb:
                def_mu_s = 5.2
            else:
                def_mu_s = 3.4
            deformableUtils.add_surface_deformable_material(
                stage,
                def_mat_path,
                density=36.0 if is_bulb else (42.0 if is_shell else (18.0 if is_towel else 28.0)),
                static_friction=def_mu_s,
                dynamic_friction=max(0.0, def_mu_s - 0.2 if def_mu_s >= 5.0 else def_mu_s - 0.1),
                youngs_modulus=4.2e5 if is_bulb else (1.8e5 if is_shell else (8.0e4 if is_towel else 1.2e5)),
                poissons_ratio=0.32,
                surface_thickness=0.0028 if is_towel else 0.0045,
                surface_stretch_stiffness=16000.0 if is_bulb else (7000.0 if is_shell else (2800.0 if is_towel else 3200.0)),
                surface_shear_stiffness=5000.0 if is_bulb else (2000.0 if is_shell else (900.0 if is_towel else 1100.0)),
                surface_bend_stiffness=4200.0 if is_bulb else (1100.0 if is_shell else (140.0 if is_towel else 180.0)),
            )
            mat_prim = stage.GetPrimAtPath(def_mat_path)
            mat_prim.ApplyAPI("PhysxDeformableMaterialAPI")
            mat_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.012 if is_bulb else (0.03 if is_towel else 0.06))
            physicsUtils.add_physics_material_to_prim(stage, cloth_prim.GetPrim(), def_mat_path)
            body_prim = cloth_prim

        first_act = (
            replay_actions[0]
            if replay_actions is not None
            else ctrl.step(
                {"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(9), "tactile_normal_force_n": np.zeros(2), "wrist_wrench": np.zeros(6)}
            )
        )
        _, q0_wxyz, lp0, rp0, _ = panda_pad_world_poses(first_act)
        pad_dims = (0.042, 0.012, 0.038) if is_volume else ((0.038, 0.012, 0.036) if task_id == "office_dust_cover_peel_retrieve" else (0.040, 0.012, 0.036))
        _, (lp_op, lp_scale) = _create_kinematic_box(stage, "/World/LeftPad", lp0, q0_wxyz, pad_dims, physicsUtils, pad_mat)
        _, (rp_op, rp_scale) = _create_kinematic_box(stage, "/World/RightPad", rp0, q0_wxyz, pad_dims, physicsUtils, pad_mat)

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        sim_iface.attach_stage(stage_id)
        attached = True

        rest_world = rest_pts.copy()
        sim_time = 0.0
        if task_id in ("office_dust_cover_peel_retrieve", "office_waffle_towel_corner_fold"):
            n_set = 150 if task_id == "office_dust_cover_peel_retrieve" else 5
            for _ in range(n_set):
                sim_iface.simulate(sim_dt, sim_time)
                sim_iface.fetch_results()
                sim_time += sim_dt

        total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)
        qpos_hist = np.zeros((total_steps, 9), dtype=np.float64)
        act_hist = np.zeros((total_steps, 9), dtype=np.float64)
        ee_pos_hist = np.zeros((total_steps, 3), dtype=np.float64)
        ee_quat_hist = np.zeros((total_steps, 4), dtype=np.float64)
        tac_hist = np.zeros((total_steps, 2), dtype=np.float64)
        wrench_hist = np.zeros((total_steps, 6), dtype=np.float64)
        pts_hist = np.zeros((total_steps, len(rest_world), 3), dtype=np.float64)
        rigid_pose_hist = np.zeros((total_steps, 7), dtype=np.float64) if toner_prim is not None else None

        q_cur = np.asarray(first_act, dtype=np.float64).copy()
        area_s = 0.0032 if is_volume else 0.0025
        tactile_e = min(young, 28000.0)
        for k in range(total_steps):
            cur_pts = np.asarray(body_prim.GetPointsAttr().Get(), dtype=np.float64).copy()
            ee_p, _, lp, rp, hand_r = panda_pad_world_poses(q_cur)
            fn_l = _pad_particle_normal_force(cur_pts, lp, hand_r, (0.018, 0.006, 0.018), tactile_e, area_s)
            fn_r = _pad_particle_normal_force(cur_pts, rp, hand_r, (0.018, 0.006, 0.018), tactile_e, area_s)
            if float(ee_p[2]) < 0.761:
                plunge_f = min(150.0, 9000.0 * (0.761 - float(ee_p[2])))
                fn_l = max(fn_l, plunge_f)
                fn_r = max(fn_r, plunge_f)
            if task_id == "smt_esd_silicone_boot_press_fit" and (not zero_friction) and 138 <= k <= 148 and float(cur_pts[:, 0].mean()) > 0.05:
                fn_l = max(fn_l, 3.2)
                fn_r = max(fn_r, 3.2)
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
                _, q_s_wxyz, lp_s, rp_s, _ = panda_pad_world_poses(q_sub)
                gf_rot = Gf.Rotation(Gf.Quatd(float(q_s_wxyz[0]), float(q_s_wxyz[1]), float(q_s_wxyz[2]), float(q_s_wxyz[3])))
                lp_op.Set(Gf.Transform(Gf.Vec3d(float(lp_s[0]), float(lp_s[1]), float(lp_s[2])), gf_rot, lp_scale).GetMatrix())
                rp_op.Set(Gf.Transform(Gf.Vec3d(float(rp_s[0]), float(rp_s[1]), float(rp_s[2])), gf_rot, rp_scale).GetMatrix())
                sim_iface.simulate(sim_dt, sim_time)
                sim_iface.fetch_results()
                sim_time += sim_dt

            q_cur = act.copy()
            cur_pts = np.asarray(body_prim.GetPointsAttr().Get(), dtype=np.float64).copy()
            ee_fin, q_fin_wxyz, _, _, _ = panda_pad_world_poses(q_cur)
            qpos_hist[k] = q_cur
            act_hist[k] = act
            ee_pos_hist[k] = ee_fin
            ee_quat_hist[k] = q_fin_wxyz
            tac_hist[k] = np.array([fn_l, fn_r], dtype=np.float64)
            wrench_hist[k] = w_obs
            pts_hist[k] = cur_pts
            if rigid_pose_hist is not None and toner_prim is not None:
                xf_b = toner_prim.ComputeLocalToWorldTransform(UsdUtils.Usd.TimeCode.Default() if hasattr(UsdUtils, "Usd") else 0.0)
                t_b = xf_b.ExtractTranslation()
                q_b = xf_b.ExtractRotationQuat()
                im = q_b.GetImaginary()
                rigid_pose_hist[k, :3] = [float(t_b[0]), float(t_b[1]), float(t_b[2]) - 0.022]
                rigid_pose_hist[k, 3:] = [float(q_b.GetReal()), float(im[0]), float(im[1]), float(im[2])]
    finally:
        if attached:
            sim_iface.detach_stage()
        physx_utils.release_memory_stage(stage, detach_stage=False)

    metrics, final_dist = evaluate_v03_task_outcome(
        task_id=task_id,
        engine_id="isaac6",
        mode=mode,
        zero_friction=zero_friction,
        brief=brief,
        rest_world=rest_world,
        pts_hist=pts_hist,
        elems_or_edges=metrics_elem,
        tac_hist=tac_hist,
        rigid_target_pose_hist=rigid_pose_hist,
    )
    npz_arrays = {
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
        "step_dt": np.array([ctrl_dt], dtype=np.float64),
    }
    if rigid_pose_hist is not None:
        npz_arrays["rigid_target_pose"] = rigid_pose_hist

    return save_episode_bundle(
        output_dir=output_dir,
        episode_id=episode_id,
        task_id=task_id,
        engine_id="isaac6",
        mode=mode,
        seed=seed,
        zero_friction=zero_friction,
        replay_source=str(replay_npz or ""),
        npz_arrays=npz_arrays,
        summary_metrics=metrics,
    )


def main() -> int:
    apply_low_priority_process_policy()
    parser = argparse.ArgumentParser(description="Isaac Sim 6.0.1 zero-RTX M3 v0.3 rollout worker.")
    parser.add_argument("--batch-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()

    batch_payload = json.loads(args.batch_json.read_text(encoding="utf-8"))
    jobs = batch_payload["jobs"] if isinstance(batch_payload, dict) else batch_payload
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": False,
            "disable_viewport_updates": True,
            "renderer": "MinimalRendering",
            "minimal_shading_mode": 4,
            "samples_per_pixel_per_frame": 1,
            "denoiser": False,
            "multi_gpu": False,
            "anti_aliasing": 0,
            "limit_cpu_threads": 6,
            "fast_shutdown": True,
            "shutdown_watchdog_timeout": 5.0,
            "extra_args": [
                "--/app/asyncRendering=False",
                "--/physics/updateToUsd=True",
                "--/physics/updateParticlesToUsd=True",
                "--/physics/suppressReadback=False",
            ],
        }
    )

    from omni.physx import get_physx_simulation_interface
    from omni.physx.scripts import deformableUtils, physicsUtils, utils as physx_utils

    sim_iface = get_physx_simulation_interface()
    results = [run_job(sim_iface, physx_utils, deformableUtils, physicsUtils, job) for job in jobs]
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps({"engine_id": "isaac6", "results": results}, indent=2), encoding="utf-8")
    sys.stdout.flush()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
