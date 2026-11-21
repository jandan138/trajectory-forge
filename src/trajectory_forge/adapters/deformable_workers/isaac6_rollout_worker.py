"""Isolated Isaac Sim 6.0.1 zero-RTX M3 rollout worker (PhysX VolumeTet + SurfaceDeformable)."""

from __future__ import annotations

import argparse
import json
import math
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


def _apply_small_collision_offsets(prim: Any, rest_offset: float = 0.001, contact_offset: float = 0.004) -> None:
    from pxr import PhysxSchema

    if not prim or not prim.IsValid():
        return
    col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    col_api.CreateRestOffsetAttr().Set(float(rest_offset))
    col_api.CreateContactOffsetAttr().Set(float(contact_offset))


def _create_rigid_material(
    stage: Any,
    mat_path: str,
    static_friction: float,
    dynamic_friction: float,
    combine_mode: str = "average",
) -> str:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys_mat.CreateStaticFrictionAttr().Set(float(static_friction))
    phys_mat.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    phys_mat.CreateRestitutionAttr().Set(0.02)
    px_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px_mat.CreateFrictionCombineModeAttr().Set(str(combine_mode))
    return mat_path


def _create_static_box(
    stage: Any,
    prim_path: str,
    pos_xyz: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    physicsUtils: Any,
    mat_path: str,
    *,
    rest_offset: float = 0.001,
    contact_offset: float = 0.004,
) -> Any:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    scale_vec = Gf.Vec3d(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2]))
    xform_op = cube.AddTransformOp()
    mat = Gf.Transform(
        Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])),
        Gf.Rotation(Gf.Quatd(1.0)),
        scale_vec,
    ).GetMatrix()
    xform_op.Set(mat)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _apply_small_collision_offsets(cube.GetPrim(), rest_offset=rest_offset, contact_offset=contact_offset)
    physicsUtils.add_physics_material_to_prim(stage, cube.GetPrim(), mat_path)
    return cube.GetPrim()


def _create_kinematic_box(
    stage: Any,
    prim_path: str,
    pos_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
    size_xyz: tuple[float, float, float],
    physicsUtils: Any,
    mat_path: str,
    *,
    rest_offset: float = 0.0005,
    contact_offset: float = 0.004,
) -> tuple[Any, tuple[Any, Any]]:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    scale_vec = Gf.Vec3d(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2]))
    xform_op = cube.AddTransformOp()
    q_gf = Gf.Rotation(Gf.Quatd(float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])))
    mat = Gf.Transform(
        Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])),
        q_gf,
        scale_vec,
    ).GetMatrix()
    xform_op.Set(mat)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _apply_small_collision_offsets(cube.GetPrim(), rest_offset=rest_offset, contact_offset=contact_offset)
    physicsUtils.add_physics_material_to_prim(stage, cube.GetPrim(), mat_path)
    rb = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    rb.CreateKinematicEnabledAttr().Set(True)
    return cube.GetPrim(), (xform_op, scale_vec)


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


def run_job(
    sim_iface: Any,
    physx_utils: Any,
    deformableUtils: Any,
    physicsUtils: Any,
    job: dict[str, Any],
) -> dict[str, Any]:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, Vt

    pkg_dir = Path(job["package_dir"])
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
    sim_dt = 0.001 if task_id == "coffee_bar_fragile_pastry_plate" else 0.002
    substeps_per_ctrl = 10 if task_id == "coffee_bar_fragile_pastry_plate" else 5

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
        ps_api.GetMaxPositionIterationCountAttr().Set(16)

        UsdGeom.Scope.Define(stage, "/World/Materials")
        pad_mat = _create_rigid_material(
            stage,
            "/World/Materials/PadMat",
            static_friction=0.0 if zero_friction else 2.8,
            dynamic_friction=0.0 if zero_friction else 2.6,
            combine_mode="min" if zero_friction else "max",
        )
        tray_mat = _create_rigid_material(
            stage,
            "/World/Materials/TrayMat",
            static_friction=0.65,
            dynamic_friction=0.55,
            combine_mode="average",
        )
        blotter_mat = _create_rigid_material(
            stage,
            "/World/Materials/BlotterMat",
            static_friction=0.16,
            dynamic_friction=0.14,
            combine_mode="min",
        )
        slot_mat = _create_rigid_material(
            stage,
            "/World/Materials/SlotMat",
            static_friction=0.0 if zero_friction else 2.5,
            dynamic_friction=0.0 if zero_friction else 2.2,
            combine_mode="min" if zero_friction else "max",
        )
        lid_mat = _create_rigid_material(
            stage,
            "/World/Materials/LidMat",
            static_friction=0.45 if zero_friction else 1.1,
            dynamic_friction=0.35 if zero_friction else 0.9,
            combine_mode="average",
        )

        lid_prim = None
        if task_id == "coffee_bar_fragile_pastry_plate":
            e_eff = 34000.0
            _create_static_box(stage, "/World/Table", (0.0, 0.0, 0.745), (2.0, 0.90, 0.020), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/PastryBoxBottom", (-0.18, -0.02, 0.760), (0.136, 0.136, 0.010), physicsUtils, tray_mat)
            _create_static_box(stage, "/World/PastryBoxBack", (-0.18, 0.044, 0.780), (0.136, 0.008, 0.050), physicsUtils, tray_mat)

            plate = UsdGeom.Cylinder.Define(stage, "/World/ServingPlate")
            plate.CreateAxisAttr("Z")
            plate.CreateRadiusAttr(0.095)
            plate.CreateHeightAttr(0.008)
            plate.AddTransformOp().Set(
                Gf.Transform(Gf.Vec3d(0.18, -0.02, 0.759), Gf.Rotation(Gf.Quatd(1.0)), Gf.Vec3d(1.0)).GetMatrix()
            )
            UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
            _apply_small_collision_offsets(plate.GetPrim())
            physicsUtils.add_physics_material_to_prim(stage, plate.GetPrim(), tray_mat)

            # Articulated hinged pastry lid pivoting around (-0.18, 0.044, 0.807)
            lid_xf = UsdGeom.Xform.Define(stage, "/World/PastryLid")
            lid_xf.AddTransformOp().Set(
                Gf.Transform(Gf.Vec3d(-0.18, 0.044, 0.807), Gf.Rotation(Gf.Quatd(1.0)), Gf.Vec3d(1.0)).GetMatrix()
            )
            UsdPhysics.RigidBodyAPI.Apply(lid_xf.GetPrim())
            mass_api = UsdPhysics.MassAPI.Apply(lid_xf.GetPrim())
            mass_api.CreateMassAttr().Set(0.040)
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, -0.06, 0.0))
            _create_static_box(
                stage,
                "/World/PastryLid/Panel",
                (0.0, -0.064, 0.0),
                (0.132, 0.128, 0.006),
                physicsUtils,
                lid_mat,
            )
            _create_static_box(
                stage,
                "/World/PastryLid/Tab",
                (0.0, -0.142, 0.002),
                (0.064, 0.028, 0.008),
                physicsUtils,
                lid_mat,
            )
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/LidJoint")
            joint.CreateAxisAttr("X")
            joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/PastryLid")])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.18, 0.044, 0.807))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLowerLimitAttr(-123.2)
            joint.CreateUpperLimitAttr(0.0)
            PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateDampingAttr().Set(0.0004)
            lid_prim = lid_xf.GetPrim()

            rest_local, elem_idx = build_voxel_tet_grid((0.07, 0.07, 0.028), grid_xyz=(2, 2, 1))
            elem_oriented = elem_idx.copy()
            for ti, (a, b, c, d_i) in enumerate(elem_oriented):
                if (
                    np.linalg.det(
                        np.stack(
                            [rest_local[b] - rest_local[a], rest_local[c] - rest_local[a], rest_local[d_i] - rest_local[a]],
                            axis=-1,
                        )
                    )
                    < 0.0
                ):
                    elem_oriented[ti] = (a, b, d_i, c)
            elem_idx = elem_oriented
            rest_pts = rest_local.copy()
            rest_pts[:, 0] += -0.18 + dx
            rest_pts[:, 1] += -0.02 + dy
            rest_pts[:, 2] += 0.7665

            tet_prim = UsdGeom.TetMesh.Define(stage, "/World/DeformableAsset")
            tet_prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in rest_pts]))
            tet_prim.CreateTetVertexIndicesAttr(
                Vt.Vec4iArray([Gf.Vec4i(int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in elem_idx])
            )
            deformableUtils.set_physics_volume_deformable_body(stage, Sdf.Path("/World/DeformableAsset"))
            _apply_small_collision_offsets(tet_prim.GetPrim(), rest_offset=0.0006, contact_offset=0.0035)
            tet_prim.GetPrim().ApplyAPI("PhysxBaseDeformableBodyAPI")
            tet_prim.GetPrim().GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(64)
            tet_prim.GetPrim().GetAttribute("physxDeformableBody:linearDamping").Set(2.5)
            def_mat_path = "/World/Materials/DeformableMat"
            deformableUtils.add_deformable_material(
                stage,
                def_mat_path,
                density=160.0,
                static_friction=0.0 if zero_friction else 3.6,
                dynamic_friction=0.0 if zero_friction else 3.4,
                youngs_modulus=260000.0,
                poissons_ratio=0.28,
            )
            mat_prim = stage.GetPrimAtPath(def_mat_path)
            mat_prim.ApplyAPI("PhysxDeformableMaterialAPI")
            mat_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.08)
            physicsUtils.add_physics_material_to_prim(stage, tet_prim.GetPrim(), def_mat_path)
            body_prim = tet_prim
            is_solid = True
        else:
            e_eff = 20000.0
            for idx_b, (pos_b, sz_b, mat_b) in enumerate(
                (
                    ((0.0, 0.0, 0.665), (2.0, 0.90, 0.020), blotter_mat),
                    ((0.0, 0.02, 0.722), (0.56, 0.36, 0.070), blotter_mat),
                    ((-0.130, -0.020, 0.769), (0.004, 0.110, 0.022), blotter_mat),
                    ((-0.112, -0.060, 0.778), (0.036, 0.036, 0.004), blotter_mat),
                    ((-0.112, 0.020, 0.778), (0.036, 0.036, 0.004), blotter_mat),
                    ((0.22, -0.04, 0.761), (0.190, 0.190, 0.008), tray_mat),
                    ((0.129, -0.04, 0.770), (0.008, 0.190, 0.018), tray_mat),
                    ((0.311, -0.04, 0.770), (0.008, 0.190, 0.018), tray_mat),
                    ((0.22, 0.051, 0.770), (0.190, 0.008, 0.018), tray_mat),
                )
            ):
                _create_static_box(stage, f"/World/DeskProp_{idx_b}", pos_b, sz_b, physicsUtils, mat_b)

            rest_v, tris, elem_idx = load_cloth_mesh_from_package(pkg_dir, "pilot_flexible_wiping_cloth_01")
            rest_pts = rest_v.copy()
            rest_pts[:, 0] += -0.13 + dx
            rest_pts[:, 1] += -0.02 + dy
            rest_pts[:, 2] += 0.761

            cloth_prim = UsdGeom.Mesh.Define(stage, "/World/DeformableAsset")
            cloth_prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in rest_pts]))
            cloth_prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
            cloth_prim.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for f in tris for i in f]))
            deformableUtils.set_physics_surface_deformable_body(stage, Sdf.Path("/World/DeformableAsset"))
            _apply_small_collision_offsets(cloth_prim.GetPrim(), rest_offset=0.0078, contact_offset=0.011)

            def_mat_path = "/World/Materials/DeformableMat"
            deformableUtils.add_surface_deformable_material(
                stage,
                def_mat_path,
                density=32.0,
                static_friction=0.0 if zero_friction else 1.8,
                dynamic_friction=0.0 if zero_friction else 1.6,
                youngs_modulus=2.2e5,
                poissons_ratio=0.30,
                surface_thickness=0.015,
                surface_stretch_stiffness=2400.0,
                surface_shear_stiffness=900.0,
                surface_bend_stiffness=350.0,
            )
            mat_prim = stage.GetPrimAtPath(def_mat_path)
            mat_prim.ApplyAPI("PhysxDeformableMaterialAPI")
            mat_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.05)
            physicsUtils.add_physics_material_to_prim(stage, cloth_prim.GetPrim(), def_mat_path)
            body_prim = cloth_prim
            is_solid = False

        first_act = (
            replay_actions[0]
            if replay_actions is not None
            else ctrl.step(
                {"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(9), "tactile_normal_force_n": np.zeros(2), "wrist_wrench": np.zeros(6)}
            )
        )
        _, q0_wxyz, lp0, rp0, _ = panda_pad_world_poses(first_act)
        pad_dims = (0.044, 0.012, 0.040) if is_solid else (0.036, 0.012, 0.036)
        _, (lp_op, lp_scale) = _create_kinematic_box(
            stage,
            "/World/LeftPad",
            lp0,
            q0_wxyz,
            pad_dims,
            physicsUtils,
            pad_mat,
            rest_offset=0.0005,
            contact_offset=0.004,
        )
        _, (rp_op, rp_scale) = _create_kinematic_box(
            stage,
            "/World/RightPad",
            rp0,
            q0_wxyz,
            pad_dims,
            physicsUtils,
            pad_mat,
            rest_offset=0.0005,
            contact_offset=0.004,
        )

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        sim_iface.attach_stage(stage_id)
        attached = True

        rest_world = rest_pts.copy()
        total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)
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
        sim_time = 0.0

        for k in range(total_steps):
            cur_pts = np.asarray(body_prim.GetPointsAttr().Get(), dtype=np.float64).copy()
            ee_p, q_wxyz, lp, rp, hand_r = panda_pad_world_poses(q_cur)
            area_s = 0.0045 if is_solid else 0.0030
            fn_l = _pad_particle_normal_force(cur_pts, lp, hand_r, (0.018, 0.006, 0.018), e_eff, area_s)
            fn_r = _pad_particle_normal_force(cur_pts, rp, hand_r, (0.018, 0.006, 0.018), e_eff, area_s)
            if not is_solid and ee_p[2] < 0.764:
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

            if np.all(np.isfinite(cur_pts)):
                max_z = max(max_z, float(cur_pts[:, 2].mean()))
            if is_solid and lid_prim is not None:
                lid_xf_world = UsdGeom.Xformable(lid_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                lq = lid_xf_world.ExtractRotationQuat()
                imag = lq.GetImaginary()
                real = lq.GetReal()
                lid_ang = min(2.15, 2.0 * abs(math.atan2(float(imag[0]), float(real))))
                lid_hist[k] = lid_ang
                p_s, n_s = compute_tet_strains_from_points(rest_world, cur_pts, elem_idx)
                peak_polar = max(peak_polar, p_s)
                peak_naive = max(peak_naive, n_s)
            else:
                e_s = compute_edge_stretch_from_points(rest_world, cur_pts, elem_idx)
                peak_edge = max(peak_edge, e_s)
                if np.all(np.isfinite(cur_pts)):
                    com_xy = cur_pts[:, :2].mean(axis=0)
                    for wi in range(3):
                        wipe_min_dists[wi] = min(wipe_min_dists[wi], float(np.linalg.norm(com_xy - wipe_wps[wi])))
    finally:
        if attached:
            sim_iface.detach_stage()
        physx_utils.release_memory_stage(stage, detach_stage=False)

    def_spec = (
        brief.objects["fragile_pastry"].deformable
        if task_id == "coffee_bar_fragile_pastry_plate"
        else brief.objects["wiping_cloth"].deformable
    )
    assert def_spec is not None

    finite_ok = bool(np.all(np.isfinite(pts_hist)))
    fin_com = pts_hist[-1].mean(axis=0) if finite_ok else np.array([np.nan, np.nan, np.nan])
    peak_force_n = float(np.max(np.abs(tac_hist))) if len(tac_hist) else 0.0
    if is_solid:
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
        "step_dt": np.array([ctrl_dt], dtype=np.float64),
    }
    metrics = {
        "task_id": task_id,
        "engine_id": "isaac6",
        "mode": mode,
        "zero_friction": zero_friction,
        "max_com_z_m": round(max_z, 5) if finite_ok else None,
        "final_com_xyz_m": [round(float(x), 5) for x in fin_com] if finite_ok else None,
        "final_target_dist_m": round(final_dist, 5),
        "max_lid_angle_rad": round(float(lid_hist.max()), 4) if is_solid else None,
        "wipe_min_dists_m": [round(float(x), 5) for x in wipe_min_dists] if not is_solid else None,
        "polar_stretch_q95": round(peak_polar, 5) if is_solid else None,
        "naive_f_minus_i_q95": round(peak_naive, 5) if is_solid else None,
        "edge_stretch_q95": round(peak_edge, 5) if not is_solid else None,
        "peak_normal_force_n": round(peak_force_n, 4),
        "damage_score": round(damage, 5),
        "task_geometric_success": geom_ok,
        "intact": intact,
    }
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
    parser = argparse.ArgumentParser(description="Isaac Sim 6.0.1 zero-RTX M3 deformable rollout worker.")
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
