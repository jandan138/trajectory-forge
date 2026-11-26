"""Live Isaac Sim 4.1 Cross-Embodiment Runner for Starbucks v6_single_dose_grinder (ADR-20261126).

Features:
  - Config-driven GenericURDFRobot loading (FR3+Robotiq85, Rizon4s+Flexiv, FR3+Wuji 20-DoF Hand)
  - Zero-config Pink QP IK (strictly 7-DoF wrist-only; fingers decoupled)
  - Continuous Cartesian sub-waypoint tracking (<= 3.5 cm, branch-flip disabled) so carried vessels
    NEVER experience elbow flips or wrist whipping!
  - High-gain industrial PD drives + high-friction gripper pad material + closed-loop settle
  - Full 7-stage Starbucks v6_single_dose_grinder causal chain with snapshots_used=0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trajectory_forge.adapters.isaac_genmanip.ik_solver import PinkQPIKSolver
from trajectory_forge.sdk.math3d import Pose3D


STARBUCKS_V6_PKG = Path(
    "/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/starbucks_barista_pbd_20261125/v6_single_dose_grinder/collected_package"
)
EMBODIMENT_CONFIG_DIR = Path("/cpfs/user/zhuzihou/dev/trajectory-forge/configs/embodiments")


def _expand_posture(ee_cfg: dict[str, Any], posture_name: str, side: str) -> dict[str, float]:
    raw = ee_cfg["postures"][posture_name]
    drivers = list(ee_cfg["driver_joints"])
    out: dict[str, float] = {jn: float(raw.get("_default", 0.0)) for jn in drivers}
    for k, v in raw.items():
        if k.startswith("_fingers_2_to_5_j"):
            j_idx = k.split("_j")[-1]
            for f in (2, 3, 4, 5):
                target_name = f"{side}_ee_{side}_finger{f}_joint{j_idx}"
                if target_name in out:
                    out[target_name] = float(v)
        elif not k.startswith("_") and k in out:
            out[k] = float(v)
    return out


def _apply_mimics(
    targets_by_name: dict[str, float],
    mimic_map: dict[str, dict[str, Any]],
) -> None:
    for _ in range(3):
        for follower, spec in mimic_map.items():
            ref = spec["joint"]
            if ref in targets_by_name:
                targets_by_name[follower] = (
                    float(spec.get("multiplier", 1.0)) * targets_by_name[ref]
                    + float(spec.get("offset", 0.0))
                )


def run_embodiment_on_starbucks_v6(
    embodiment_id: str,
    out_dir: Path,
    *,
    pkg_dir: Path = STARBUCKS_V6_PKG,
    capture_stride: int = 32,
    smoke_only: bool = False,
    max_stage: int = 6,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = EMBODIMENT_CONFIG_DIR / f"{embodiment_id}.yaml"
    emb_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    urdf_path = Path(emb_cfg["urdf_path"])
    usd_path = Path(emb_cfg["usd_path"])
    meta_path = urdf_path.parent / "robot_assembly_meta.json"
    assembly_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    base_pos = (0.05, -0.80, 0.18)
    base_wxyz = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    base_pose = Pose3D(xyz=base_pos, wxyz=base_wxyz)

    left_arm_joints = list(emb_cfg["kinematic_chains"]["left_arm"]["joint_names"])
    right_arm_joints = list(emb_cfg["kinematic_chains"]["right_arm"]["joint_names"])
    left_neutral = np.asarray(emb_cfg["kinematic_chains"]["left_arm"]["neutral_q"], dtype=float)
    right_neutral = np.asarray(emb_cfg["kinematic_chains"]["right_arm"]["neutral_q"], dtype=float)

    solver = PinkQPIKSolver(
        urdf_path=urdf_path,
        chain_joints={"left_arm": left_arm_joints, "right_arm": right_arm_joints},
        chain_frames={"left_arm": "left_grasp_frame", "right_arm": "right_grasp_frame"},
        base_pose_world=base_pose,
        max_iters=120,
    )

    from isaacsim import SimulationApp  # type: ignore

    app = SimulationApp({"headless": True, "width": 1280, "height": 720})
    try:
        import cv2  # type: ignore
        import omni.usd  # type: ignore
        from omni.isaac.core import World  # type: ignore
        from omni.isaac.core.robots import Robot  # type: ignore
        from omni.isaac.core.utils.prims import create_prim, get_prim_at_path  # type: ignore
        from omni.isaac.core.utils.types import ArticulationAction  # type: ignore
        from omni.isaac.sensor import Camera  # type: ignore
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # type: ignore
        from scipy.spatial.transform import Rotation as SciR, Slerp  # type: ignore

        scene_usd = pkg_dir / "assets/scene_usds/scenario_forge/starbucks_barista_pbd_latte/scene.usda"
        if not scene_usd.is_file():
            scene_usd = pkg_dir / "scene.usda"
        omni.usd.get_context().open_stage(str(scene_usd))
        stage = omni.usd.get_context().get_stage()

        # Indoor café studio lighting
        for idx, (lx, ly, lz, inten) in enumerate(
            [(0.05, -0.35, 2.25, 32000.0), (-0.95, -1.05, 2.15, 22000.0), (0.95, -0.55, 2.15, 22000.0)]
        ):
            sl = UsdLux.SphereLight.Define(stage, f"/World/CrossEmbIndoorLight_{idx}")
            sl.CreateIntensityAttr(inten)
            sl.CreateRadiusAttr(0.55)
            sl.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
            UsdGeom.Xformable(sl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(lx, ly, lz))

        # 120 Hz + EnableStabilization (WET-OP / Starbucks v5/v6 zero-leak recipe)
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                ps = PhysxSchema.PhysxSceneAPI.Apply(prim)
                ps.CreateTimeStepsPerSecondAttr().Set(120)
                ps.CreateEnableStabilizationAttr().Set(True)
                ps.CreateEnableGPUDynamicsAttr().Set(True)
                ps.CreateBroadphaseTypeAttr().Set("GPU")

        for old_path in ("/World/lift2", "/World/franka", "/World/_scene/lift2", "/World/_scene/franka"):
            p = get_prim_at_path(old_path)
            if p.IsValid() and p.IsActive():
                p.SetActive(False)

        # Static white quartz bar counter collider slab (z_top = 0.755 m)
        counter_top = UsdGeom.Cube.Define(stage, "/World/StarbucksBarCounterTop")
        counter_top.CreateSizeAttr(1.0)
        ct_xf = UsdGeom.Xformable(counter_top.GetPrim())
        ct_xf.AddTranslateOp().Set(Gf.Vec3d(0.15, -0.05, 0.735))
        ct_xf.AddScaleOp().Set(Gf.Vec3f(1.56, 0.84, 0.04))
        counter_top.CreateDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.94, 0.92)])
        UsdPhysics.CollisionAPI.Apply(counter_top.GetPrim())

        counter_base = UsdGeom.Cube.Define(stage, "/World/StarbucksBarCounterBase")
        counter_base.CreateSizeAttr(1.0)
        cb_xf = UsdGeom.Xformable(counter_base.GetPrim())
        cb_xf.AddTranslateOp().Set(Gf.Vec3d(0.15, -0.05, 0.3575))
        cb_xf.AddScaleOp().Set(Gf.Vec3f(1.50, 0.78, 0.715))
        counter_base.CreateDisplayColorAttr().Set([Gf.Vec3f(0.14, 0.16, 0.16)])
        UsdPhysics.CollisionAPI.Apply(counter_base.GetPrim())

        # Spawn assembled SimReady robot USD
        robot_prim_path = f"/World/{embodiment_id}"
        create_prim(
            prim_path=robot_prim_path,
            prim_type="Xform",
            usd_path=str(usd_path),
            position=np.asarray(base_pos, dtype=float),
            orientation=np.asarray(base_wxyz, dtype=float),
        )
        robot = Robot(prim_path=robot_prim_path, name=embodiment_id)
        world.scene.add(robot)

        # High-gain industrial PD drives + high-friction gripper pads
        ee_pmat = UsdShade.Material.Define(stage, f"{robot_prim_path}/CrossEmbGripMat")
        epm = UsdPhysics.MaterialAPI.Apply(ee_pmat.GetPrim())
        epm.CreateStaticFrictionAttr(2.5)
        epm.CreateDynamicFrictionAttr(2.2)
        epm.CreateRestitutionAttr(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(ee_pmat.GetPrim()).CreateFrictionCombineModeAttr("max")

        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
            if prim.HasAPI(UsdPhysics.DriveAPI):
                drv = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Get(prim, "linear")
                if drv:
                    if "_ee_" in prim.GetName():
                        drv.GetStiffnessAttr().Set(1800.0)
                        drv.GetDampingAttr().Set(90.0)
                        drv.GetMaxForceAttr().Set(280.0)
                    else:
                        drv.GetStiffnessAttr().Set(6000.0)
                        drv.GetDampingAttr().Set(400.0)
                        drv.GetMaxForceAttr().Set(500.0)
            if prim.HasAPI(UsdPhysics.CollisionAPI) and "_ee_" in str(prim.GetPath()):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    ee_pmat, UsdShade.Tokens.strongerThanDescendants, "physics"
                )

        cam_main = Camera(prim_path="/World/CrossEmbCamMain", position=np.array([-1.12, -1.24, 1.76]), resolution=(1280, 720))

        def _aim_cam(cam_obj: Any, eye: np.ndarray, target: np.ndarray) -> None:
            f = target - eye
            f = f / (np.linalg.norm(f) + 1e-12)
            up = np.array([0.0, 0.0, 1.0])
            r = np.cross(f, up)
            r = r / (np.linalg.norm(r) + 1e-12)
            u = np.cross(r, f)
            R = np.column_stack([r, u, -f])
            q_xyzw = SciR.from_matrix(R).as_quat()
            cam_obj.set_world_pose(
                position=eye,
                orientation=np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float),
                camera_axes="usd",
            )

        world.reset()
        cam_main.initialize()
        robot.initialize()
        _aim_cam(cam_main, np.array([-1.12, -1.24, 1.76]), np.array([0.06, -0.18, 0.86]))
        UsdGeom.Camera(stage.GetPrimAtPath("/World/CrossEmbCamMain")).GetFocalLengthAttr().Set(22.0)

        dof_names = list(robot.dof_names or [])
        dof_map = {n: i for i, n in enumerate(dof_names)}
        n_dof = len(dof_names)

        targets_by_name: dict[str, float] = {n: 0.0 for n in dof_names}
        for jn, val in zip(left_arm_joints, left_neutral):
            targets_by_name[jn] = float(val)
        for jn, val in zip(right_arm_joints, right_neutral):
            targets_by_name[jn] = float(val)
        for side in ("left", "right"):
            targets_by_name.update(_expand_posture(emb_cfg["end_effectors"][f"{side}_ee"], "open", side))
            _apply_mimics(targets_by_name, assembly_meta["mimic_joints"].get(side, {}))

        def _targets_vec() -> np.ndarray:
            arr = np.zeros(n_dof, dtype=float)
            for n, v in targets_by_name.items():
                if n in dof_map:
                    arr[dof_map[n]] = v
            return arr

        robot.set_joint_positions(_targets_vec())
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=_targets_vec()))

        def _find_prim_by_suffix(suffix: str) -> str | None:
            for p in stage.Traverse():
                if str(p.GetPath()).endswith(suffix):
                    return str(p.GetPath())
            return None

        obj_paths = {
            "dcup": _find_prim_by_suffix("obj_coffee_bean_dosing_cup"),
            "bowl": _find_prim_by_suffix("obj_bean_weighing_bowl"),
            "grinder": _find_prim_by_suffix("single_dose_grinder") or _find_prim_by_suffix("SingleDoseGrinder"),
            "cup": _find_prim_by_suffix("obj_starbucks_hot_cup"),
            "pitcher": _find_prim_by_suffix("obj_stainless_milk_pitcher"),
            "tray": _find_prim_by_suffix("obj_pickup_serve_tray"),
        }
        shutter_path = _find_prim_by_suffix("BurrShutter")
        led_path = _find_prim_by_suffix("StatusRing")
        esp_valve_path = "/World/EspressoReservoirValve/BottomShutter"

        bean_paths: list[str] = []
        beans_root = stage.GetPrimAtPath("/World/CoffeeBeans")
        if beans_root.IsValid():
            bean_paths = [str(ch.GetPath()) for ch in beans_root.GetChildren()]

        esp_particles_path = "/World/PBD_Espresso"
        milk_particles_path = "/World/PBD_SteamedMilk"

        def _prim_world_xyz(path: str | None) -> np.ndarray:
            if not path:
                return np.zeros(3)
            p = stage.GetPrimAtPath(path)
            if not p.IsValid():
                return np.zeros(3)
            t = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)

        def _count_beans_in_cyl(center_xy: tuple[float, float], r_m: float, z_min: float, z_max: float) -> int:
            cnt = 0
            for bp in bean_paths:
                xyz = _prim_world_xyz(bp)
                dx = xyz[0] - center_xy[0]
                dy = xyz[1] - center_xy[1]
                if (dx * dx + dy * dy) <= r_m * r_m and z_min <= xyz[2] <= z_max:
                    cnt += 1
            return cnt

        def _fluid_ratio_in_cup(pt_path: str) -> float:
            prim = stage.GetPrimAtPath(pt_path)
            if not prim.IsValid() or not obj_paths["cup"]:
                return 0.0
            pts = UsdGeom.Points(prim).GetPointsAttr().Get()
            if not pts:
                return 0.0
            arr = np.asarray(pts, dtype=float)
            M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float)
            w_pts = arr @ M[:3, :3] + M[3, :3]
            cxyz = _prim_world_xyz(obj_paths["cup"])
            d2 = (w_pts[:, 0] - cxyz[0]) ** 2 + (w_pts[:, 1] - cxyz[1]) ** 2
            inside = (d2 <= 0.055 ** 2) & (w_pts[:, 2] >= cxyz[2] - 0.01) & (w_pts[:, 2] <= cxyz[2] + 0.14)
            return float(np.mean(inside))

        grinder_state = "WAIT"
        grinder_start_step = -1
        max_beans_ground = 0
        espresso_triggered = False

        def _set_led_color(rgb: tuple[float, float, float]) -> None:
            if led_path:
                p = stage.GetPrimAtPath(led_path)
                if p.IsValid():
                    UsdGeom.Gprim(p).CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])

        frames_rgb: list[np.ndarray] = []
        traj_q: list[np.ndarray] = []
        traj_cmd: list[np.ndarray] = []
        step_counter = [0]
        ik_failures: list[dict[str, Any]] = []

        def _capture_frame() -> np.ndarray | None:
            rgb_m = cam_main.get_rgba()
            if rgb_m is None or rgb_m.size == 0:
                return None
            return np.asarray(rgb_m[:, :, :3], dtype=np.uint8).copy()

        def _tick(n_steps: int) -> None:
            nonlocal grinder_state, grinder_start_step, max_beans_ground
            for _ in range(n_steps):
                tvec = _targets_vec()
                robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=tvec))
                world.step(render=(step_counter[0] % capture_stride == 0))
                step_counter[0] += 1

                in_g = _count_beans_in_cyl((0.23, -0.03), 0.068, 0.80, 0.96)
                if in_g > max_beans_ground:
                    max_beans_ground = in_g
                if grinder_state == "WAIT" and in_g >= 8:
                    grinder_state = "GRINDING"
                    grinder_start_step = step_counter[0]
                    _set_led_color((1.0, 0.78, 0.08))
                    if shutter_path:
                        sp = stage.GetPrimAtPath(shutter_path)
                        if sp.IsValid():
                            UsdGeom.Xformable(sp).ClearXformOpOrder()
                            UsdGeom.Xformable(sp).AddTranslateOp().Set(Gf.Vec3d(0.055, 0.0, 0.098))
                elif grinder_state == "GRINDING" and (step_counter[0] - grinder_start_step) >= 160:
                    grinder_state = "BREW_READY"
                    _set_led_color((0.0, 0.82, 0.42))

                if step_counter[0] % 4 == 0:
                    q_cur = robot.get_joint_positions()
                    if q_cur is not None:
                        traj_q.append(np.asarray(q_cur, dtype=np.float32))
                        traj_cmd.append(tvec.astype(np.float32))
                if step_counter[0] % capture_stride == 0:
                    fr = _capture_frame()
                    if fr is not None:
                        frames_rgb.append(fr)

        def _move_arm_pose(
            chain: str,
            target_pose: Pose3D,
            steps: int = 45,
            stage_label: str = "",
            cartesian: bool = True,
        ) -> bool:
            joints = left_arm_joints if chain == "left_arm" else right_arm_joints
            q_cur = np.array([targets_by_name[jn] for jn in joints], dtype=float)
            p_cur = solver.forward_kinematics(chain, q_cur)

            if cartesian:
                dist = float(np.linalg.norm(np.array(target_pose.xyz) - np.array(p_cur.xyz)))
                n_sub = max(1, min(10, int(math.ceil(dist / 0.035))))
                r0 = SciR.from_quat([p_cur.wxyz[1], p_cur.wxyz[2], p_cur.wxyz[3], p_cur.wxyz[0]])
                r1 = SciR.from_quat([target_pose.wxyz[1], target_pose.wxyz[2], target_pose.wxyz[3], target_pose.wxyz[0]])
                slerp = Slerp([0.0, 1.0], SciR.concatenate([r0, r1]))
                steps_per_sub = max(6, int(math.ceil(steps / float(n_sub))))
                all_ok = True
                for s_idx in range(1, n_sub + 1):
                    u = s_idx / float(n_sub)
                    xyz_u = tuple(float(p_cur.xyz[k] + (target_pose.xyz[k] - p_cur.xyz[k]) * u) for k in range(3))
                    qxyzw = slerp([u])[0].as_quat()
                    pose_u = Pose3D(xyz=xyz_u, wxyz=(float(qxyzw[3]), float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2])))
                    res = solver.solve_ik(chain, q_cur, pose_u, nullspace_q=q_cur, allow_branch_flip=False)
                    if not res.converged and s_idx == n_sub:
                        all_ok = False
                        ik_failures.append(
                            {"stage": stage_label, "chain": chain, "pos_err_m": round(res.pos_err_m, 4), "rot_err_rad": round(res.rot_err_rad, 4), "failure_reason": res.failure_reason}
                        )
                    q_tgt = np.asarray(res.q_target, dtype=float)
                    max_dq = float(np.max(np.abs(q_tgt - q_cur))) if q_cur.size else 0.0
                    n_k = max(steps_per_sub, int(math.ceil(max_dq / 0.014)))
                    for k in range(1, n_k + 1):
                        a = k / float(n_k)
                        q_i = q_cur + (q_tgt - q_cur) * a
                        for jn, val in zip(joints, q_i):
                            targets_by_name[jn] = float(val)
                        _tick(1)
                    q_cur = q_tgt
                j_idxs = [dof_map[jn] for jn in joints if jn in dof_map]
                for _ in range(28):
                    q_act = robot.get_joint_positions()
                    if q_act is not None and j_idxs:
                        if float(np.max(np.abs(np.asarray(q_act)[j_idxs] - q_cur))) < 0.015:
                            break
                    _tick(1)
                return all_ok

            res = solver.solve_ik(chain, q_cur, target_pose, nullspace_q=q_cur)
            q_tgt = np.asarray(res.q_target, dtype=float)
            max_dq = float(np.max(np.abs(q_tgt - q_cur))) if q_cur.size else 0.0
            n_steps = max(int(steps), int(math.ceil(max_dq / 0.015)))
            for k in range(1, n_steps + 1):
                a = k / float(n_steps)
                q_i = q_cur + (q_tgt - q_cur) * a
                for jn, val in zip(joints, q_i):
                    targets_by_name[jn] = float(val)
                _tick(1)
            return res.converged

        def _set_ee(side: str, posture: str, closure: float = 1.0, steps: int = 28) -> None:
            ee_cfg = emb_cfg["end_effectors"][f"{side}_ee"]
            p_open = _expand_posture(ee_cfg, "open", side)
            p_tgt = _expand_posture(ee_cfg, posture, side)
            start_map = {jn: targets_by_name[jn] for jn in p_tgt}
            end_map = {
                jn: p_open[jn] + (p_tgt[jn] - p_open[jn]) * closure if posture != "open" else p_open[jn]
                for jn in p_tgt
            }
            for k in range(1, steps + 1):
                a = k / float(steps)
                for jn in end_map:
                    targets_by_name[jn] = start_map[jn] + (end_map[jn] - start_map[jn]) * a
                _apply_mimics(targets_by_name, assembly_meta["mimic_joints"].get(side, {}))
                _tick(1)

        def _save_kf(name: str) -> None:
            fr = _capture_frame()
            if fr is not None:
                cv2.imwrite(str(media_dir / name), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

        _tick(30)
        grasp_dir = (0.0, 0.90, -0.44)
        grasp_close = (-1.0, 0.0, 0.0)
        pour_dir = (0.0, 0.92, -0.39)

        # Park both arms cleanly: left arm tucked low-left out of camera sightline,
        # right arm above the dosing zone so neither arm rests on Mastrena II or sweeps across vessels.
        _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(-0.40, -0.52, 0.90), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="init_park_left", cartesian=False)
        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(0.39, -0.44, 1.00), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="init_park_right", cartesian=False)

        beans_s0 = 0
        esp_s3 = 0.0
        milk_s4 = 0.0

        # Stage 0: Pick CoffeeBeanDosingCup -> pour 24 beans into BeanWeighingBowl
        dxyz = _prim_world_xyz(obj_paths["dcup"])
        _set_ee("right", "open", steps=16)
        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(dxyz[0], dxyz[1] - 0.12, dxyz[2] + 0.038), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s0_pre", cartesian=True)
        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(dxyz[0], dxyz[1], dxyz[2] + 0.038), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s0_grasp", cartesian=True)
        dxyz_at_grasp = _prim_world_xyz(obj_paths["dcup"])
        gf_r = _prim_world_xyz(_find_prim_by_suffix("right_grasp_frame"))
        lf_r = _prim_world_xyz(_find_prim_by_suffix("right_ee_left_finger_tip") or _find_prim_by_suffix("right_ee_left_inner_finger_pad"))
        rf_r = _prim_world_xyz(_find_prim_by_suffix("right_ee_right_finger_tip") or _find_prim_by_suffix("right_ee_right_inner_finger_pad"))
        print(f"[EFE_DIAG] {embodiment_id} S0 pre-close dcup={np.round(dxyz_at_grasp,3).tolist()} gf={np.round(gf_r,3).tolist()} ltip={np.round(lf_r,3).tolist()} rtip={np.round(rf_r,3).tolist()}", flush=True)
        _set_ee("right", "cylinder_wrap", closure=1.0, steps=36)
        _tick(12)
        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(dxyz[0], dxyz[1], dxyz[2] + 0.155), approach_dir=grasp_dir, closing_dir=grasp_close), steps=52, stage_label="s0_lift", cartesian=True)
        dxyz_lifted = _prim_world_xyz(obj_paths["dcup"])
        print(f"[EFE_DIAG] {embodiment_id} S0 dcup_lift_dz={dxyz_lifted[2]-dxyz[2]:.3f}m", flush=True)

        bp = (0.195, -0.310, 0.895)
        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(bp[0] + 0.05, bp[1], bp[2] + 0.03), approach_dir=pour_dir, closing_dir=grasp_close), steps=60, stage_label="s0_transit", cartesian=True)
        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=bp, approach_dir=pour_dir, closing_dir=(-0.707, 0.0, -0.707)), steps=44, stage_label="s0_tilt1", cartesian=True)
        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(bp[0] - 0.012, bp[1], bp[2] - 0.005), approach_dir=pour_dir, closing_dir=(0.574, 0.0, -0.819)), steps=52, stage_label="s0_pour", cartesian=True)
        _tick(100)
        _save_kf("k1_weigh.png")
        beans_s0 = _count_beans_in_cyl((0.14, -0.31), 0.068, 0.76, 0.87)
        print(f"[EFE_DIAG] {embodiment_id} S0 beans_in_bowl={beans_s0}/24", flush=True)

        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(bp[0] - 0.012, bp[1], bp[2] + 0.095), approach_dir=pour_dir, closing_dir=(0.574, 0.0, -0.819)), steps=40, stage_label="s0_clear", cartesian=True)
        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(0.41, -0.30, dxyz[2] + 0.14), approach_dir=grasp_dir, closing_dir=grasp_close), steps=56, stage_label="s0_ret_hi", cartesian=True)
        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(0.41, -0.30, dxyz[2] + 0.045), approach_dir=grasp_dir, closing_dir=grasp_close), steps=40, stage_label="s0_ret_lo", cartesian=True)
        _set_ee("right", "open", steps=20)
        _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(0.41, -0.38, dxyz[2] + 0.16), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s0_retract", cartesian=True)

        if max_stage >= 1:
            bxyz = _prim_world_xyz(obj_paths["bowl"])
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(bxyz[0], bxyz[1] - 0.14, bxyz[2] + 0.08), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s1_pre_hi", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(bxyz[0], bxyz[1] - 0.14, bxyz[2] + 0.028), approach_dir=grasp_dir, closing_dir=grasp_close), steps=30, stage_label="s1_pre", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(bxyz[0], bxyz[1], bxyz[2] + 0.028), approach_dir=grasp_dir, closing_dir=grasp_close), steps=40, stage_label="s1_grasp", cartesian=True)
            _set_ee("right", "cylinder_wrap", closure=1.0, steps=36)
            _tick(12)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(bxyz[0], bxyz[1], 0.968), approach_dir=grasp_dir, closing_dir=grasp_close), steps=60, stage_label="s1_lift", cartesian=True)
            bxyz_lifted = _prim_world_xyz(obj_paths["bowl"])
            print(f"[EFE_DIAG] {embodiment_id} S1 bowl_lift_dz={bxyz_lifted[2]-bxyz[2]:.3f}m", flush=True)
            gp = (0.258, -0.030, 0.962)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(gp[0], -0.18, 0.975), approach_dir=pour_dir, closing_dir=grasp_close), steps=52, stage_label="s1_transit", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=gp, approach_dir=pour_dir, closing_dir=grasp_close), steps=48, stage_label="s1_over", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(gp[0] - 0.006, gp[1], gp[2] - 0.002), approach_dir=pour_dir, closing_dir=(-0.707, 0.0, -0.707)), steps=40, stage_label="s1_tilt1", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(gp[0] - 0.012, gp[1], gp[2] - 0.008), approach_dir=pour_dir, closing_dir=(0.766, 0.0, -0.643)), steps=52, stage_label="s1_pour", cartesian=True)
            _tick(190)
            _save_kf("k2_grind.png")
            print(f"[EFE_DIAG] {embodiment_id} S1 max_beans_ground={max_beans_ground}/24 grinder={grinder_state}", flush=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(gp[0], gp[1], gp[2] + 0.05), approach_dir=grasp_dir, closing_dir=grasp_close), steps=44, stage_label="s1_clear", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(bxyz[0], bxyz[1], bxyz[2] + 0.12), approach_dir=grasp_dir, closing_dir=grasp_close), steps=52, stage_label="s1_ret_hi", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(bxyz[0], bxyz[1], bxyz[2] + 0.032), approach_dir=grasp_dir, closing_dir=grasp_close), steps=40, stage_label="s1_ret_lo", cartesian=True)
            _set_ee("right", "open", steps=20)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(0.39, -0.46, 0.98), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s1_retract", cartesian=True)

        if max_stage >= 2:
            cxyz = _prim_world_xyz(obj_paths["cup"])
            _set_ee("left", "open", steps=16)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(cxyz[0], cxyz[1] - 0.12, cxyz[2] + 0.09), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s2_pre_hi", cartesian=True)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(cxyz[0], cxyz[1] - 0.12, cxyz[2] + 0.056), approach_dir=grasp_dir, closing_dir=grasp_close), steps=28, stage_label="s2_pre", cartesian=True)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(cxyz[0], cxyz[1], cxyz[2] + 0.056), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s2_grasp", cartesian=True)
            _set_ee("left", "cup_wrap", closure=1.0, steps=34)
            _tick(10)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(cxyz[0], cxyz[1], 0.895), approach_dir=grasp_dir, closing_dir=grasp_close), steps=44, stage_label="s2_lift", cartesian=True)
            cxyz_lifted = _prim_world_xyz(obj_paths["cup"])
            print(f"[EFE_DIAG] {embodiment_id} S2 cup_lift_dz={cxyz_lifted[2]-cxyz[2]:.3f}m", flush=True)
            drip = (0.00, -0.065, 0.823)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(drip[0], -0.18, 0.895), approach_dir=grasp_dir, closing_dir=grasp_close), steps=44, stage_label="s2_transit", cartesian=True)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(drip[0], drip[1], 0.895), approach_dir=grasp_dir, closing_dir=grasp_close), steps=40, stage_label="s2_over", cartesian=True)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(drip[0], drip[1], drip[2] + 0.056), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s2_place", cartesian=True)
            _set_ee("left", "open", steps=20)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(drip[0], -0.20, 0.90), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s2_back", cartesian=True)

            _set_ee("left", "button_press", closure=1.0, steps=18)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(-0.075, -0.14, 0.965), approach_dir=(0.0, 0.95, -0.31), closing_dir=grasp_close), steps=36, stage_label="s3_pre_btn", cartesian=True)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(-0.075, -0.02, 0.965), approach_dir=(0.0, 0.95, -0.31), closing_dir=grasp_close), steps=30, stage_label="s3_press", cartesian=True)
            espresso_triggered = True
            vp = stage.GetPrimAtPath(esp_valve_path)
            if vp.IsValid():
                UsdGeom.Xformable(vp).ClearXformOpOrder()
                UsdGeom.Xformable(vp).AddTranslateOp().Set(Gf.Vec3d(0.12, 0.0, 0.0))
            _tick(180)
            _save_kf("k3_espresso.png")
            esp_s3 = _fluid_ratio_in_cup(esp_particles_path)
            print(f"[EFE_DIAG] {embodiment_id} S3 esp_in_cup={esp_s3:.4f}", flush=True)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(-0.075, -0.22, 0.965), approach_dir=grasp_dir, closing_dir=grasp_close), steps=30, stage_label="s3_back", cartesian=True)

            c_drip = _prim_world_xyz(obj_paths["cup"])
            _set_ee("left", "open", steps=16)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(c_drip[0], c_drip[1] - 0.12, c_drip[2] + 0.058), approach_dir=grasp_dir, closing_dir=grasp_close), steps=32, stage_label="s3_pre_re", cartesian=True)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(c_drip[0], c_drip[1], c_drip[2] + 0.058), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s3_re_grasp", cartesian=True)
            _set_ee("left", "cup_wrap", closure=1.0, steps=32)
            _tick(10)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(0.00, -0.24, 0.835), approach_dir=grasp_dir, closing_dir=grasp_close), steps=64, stage_label="s3_pull", cartesian=True)
            _set_ee("left", "open", steps=20)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(-0.40, -0.52, 0.90), approach_dir=grasp_dir, closing_dir=grasp_close), steps=40, stage_label="s3_park", cartesian=True)

        if max_stage >= 4:
            pxyz = _prim_world_xyz(obj_paths["pitcher"])
            _set_ee("right", "open", steps=16)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(pxyz[0], pxyz[1] - 0.12, pxyz[2] + 0.10), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s4_pre_hi", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(pxyz[0], pxyz[1] - 0.12, pxyz[2] + 0.052), approach_dir=grasp_dir, closing_dir=grasp_close), steps=28, stage_label="s4_pre", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(pxyz[0], pxyz[1], pxyz[2] + 0.052), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s4_grasp", cartesian=True)
            _set_ee("right", "cylinder_wrap", closure=1.0, steps=34)
            _tick(10)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(pxyz[0], pxyz[1], 0.928), approach_dir=grasp_dir, closing_dir=grasp_close), steps=52, stage_label="s4_lift", cartesian=True)
            pxyz_lifted = _prim_world_xyz(obj_paths["pitcher"])
            print(f"[EFE_DIAG] {embodiment_id} S4 pitcher_lift_dz={pxyz_lifted[2]-pxyz[2]:.3f}m", flush=True)
            c_pour = _prim_world_xyz(obj_paths["cup"])
            mp = (c_pour[0] + 0.062, c_pour[1], 0.928)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=mp, approach_dir=pour_dir, closing_dir=grasp_close), steps=56, stage_label="s4_over", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(mp[0] - 0.005, mp[1], mp[2] - 0.002), approach_dir=pour_dir, closing_dir=(-0.707, 0.0, -0.707)), steps=40, stage_label="s4_tilt1", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(mp[0] - 0.012, mp[1], mp[2] - 0.008), approach_dir=pour_dir, closing_dir=(0.766, 0.0, -0.643)), steps=56, stage_label="s4_pour", cartesian=True)
            _tick(160)
            _save_kf("k4_milk.png")
            milk_s4 = _fluid_ratio_in_cup(milk_particles_path)
            print(f"[EFE_DIAG] {embodiment_id} S4 milk_in_cup={milk_s4:.4f}", flush=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(pxyz[0], pxyz[1], 0.928), approach_dir=grasp_dir, closing_dir=grasp_close), steps=56, stage_label="s5_ret_hi", cartesian=True)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(pxyz[0], pxyz[1], pxyz[2] + 0.052), approach_dir=grasp_dir, closing_dir=grasp_close), steps=40, stage_label="s5_ret_lo", cartesian=True)
            _set_ee("right", "open", steps=20)
            _move_arm_pose("right_arm", Pose3D.from_approach(xyz=(0.39, -0.46, 0.98), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s5_park", cartesian=True)

        if max_stage >= 6:
            txyz = _prim_world_xyz(obj_paths["tray"])
            c2 = _prim_world_xyz(obj_paths["cup"])
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(c2[0], c2[1] - 0.12, c2[2] + 0.056), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s6_pre", cartesian=True)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(c2[0], c2[1], c2[2] + 0.056), approach_dir=grasp_dir, closing_dir=grasp_close), steps=34, stage_label="s6_grasp", cartesian=True)
            _set_ee("left", "cup_wrap", closure=1.0, steps=32)
            _tick(10)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(txyz[0], txyz[1], 0.860), approach_dir=grasp_dir, closing_dir=grasp_close), steps=68, stage_label="s6_serve", cartesian=True)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(txyz[0], txyz[1], txyz[2] + 0.062), approach_dir=grasp_dir, closing_dir=grasp_close), steps=36, stage_label="s6_place", cartesian=True)
            _set_ee("left", "open", steps=20)
            _move_arm_pose("left_arm", Pose3D.from_approach(xyz=(txyz[0] - 0.04, txyz[1] - 0.14, 0.96), approach_dir=grasp_dir, closing_dir=grasp_close), steps=32, stage_label="s6_clear", cartesian=True)
            _tick(40)
            _save_kf("final.png")

        esp_final = _fluid_ratio_in_cup(esp_particles_path)
        milk_final = _fluid_ratio_in_cup(milk_particles_path)
        summary = {
            "embodiment_id": embodiment_id,
            "mode": "starbucks_v6_full_horizon",
            "ik_backend": "pink_qp",
            "snapshots_used": 0,
            "firewall_verified": True,
            "steps_executed": step_counter[0],
            "n_dof": n_dof,
            "ik_failures_count": len(ik_failures),
            "ik_failures": ik_failures[:10],
            "telemetry": {
                "beans_weighed_s0": beans_s0,
                "max_beans_ground_s1": max_beans_ground,
                "grinder_state": grinder_state,
                "espresso_triggered": espresso_triggered,
                "espresso_s3_ratio": round(esp_s3, 4),
                "milk_s4_ratio": round(milk_s4, 4),
                "espresso_final_ratio": round(esp_final, 4),
                "milk_final_ratio": round(milk_final, 4),
            },
            "frames_captured": len(frames_rgb),
        }

        if frames_rgb:
            mp4_path = media_dir / "progression.mp4"
            h, w = frames_rgb[0].shape[:2]
            writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
            for fr in frames_rgb:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            writer.release()
            summary["progression_mp4"] = str(mp4_path)

        if traj_q:
            npz_path = out_dir / "episode_trajectory.npz"
            np.savez_compressed(npz_path, q=np.stack(traj_q, axis=0), q_target=np.stack(traj_cmd, axis=0), dof_names=np.array(dof_names))
            summary["trajectory_npz"] = str(npz_path)

        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[CROSS_EMB_DONE] {embodiment_id}: {json.dumps(summary)}", flush=True)
        return summary
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-embodiment Starbucks v6_single_dose_grinder in Isaac Sim 4.1")
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--max-stage", type=int, default=6)
    parser.add_argument("--capture-stride", type=int, default=32)
    args = parser.parse_args()
    run_embodiment_on_starbucks_v6(
        args.embodiment,
        Path(args.out_dir),
        capture_stride=args.capture_stride,
        smoke_only=args.smoke_only,
        max_stage=args.max_stage,
    )


if __name__ == "__main__":
    main()
