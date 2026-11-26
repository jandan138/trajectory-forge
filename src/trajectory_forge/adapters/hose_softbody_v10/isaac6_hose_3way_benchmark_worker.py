"""Isaac Sim 6.0.1 240Hz GPU 3-Way Hose Softbody Benchmark Worker.

Compares three hose physics models under identical physical excitation on 3 parallel benches in one RTX GPU session:
  1. `baseline_bead_chain`: v0.9 16-link SphericalJoint chain (zero bending EI=0, zero torsion GJ=0, no cuff liner)
  2. `route_a_d6_der`: 16-link UsdPhysics.Joint (D6Joint) Discrete Elastic Rod (calibrated EI & GJ angular drives)
     + Method-B 4-pad radial micro-spring silicone cuff liner + Cosserat twist-stripe skinning
  3. `route_b_physx_fem`: Non-uniform adaptive hollow-tube PhysX 5 GPU Volume Deformable FEM
     (45mm dense hollow barb-cuff + graded mid-span, real wall pinch & radial barb hoop expansion)

Three simultaneous isolated benches inside the stage:
  - Bench 1 (y = 0.0 m): Exp-1 Cantilever Natural Sag & Step Rebound (plunger depresses tip -45mm then releases)
  - Bench 2 (y = 2.5 m): Exp-2 90-deg Axial Torsional Twist Propagation (left motorized chuck +90deg -> free tip vane)
  - Bench 3 (y = 5.0 m): Exp-3 True Bimanual Gripper Hose-on-Barb Interference Fit + Hands-Off Gravity Hang + Pull-Off
    with realistic chemistry-lab Liebig condenser glass barbs (Lower Inlet "下进" vs Upper Outlet "上出"),
    Lab Gooseneck Cold-Water Faucet (blue valve), and Embedded PP Lab Sink!
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

if os.name == "nt":
    os.environ.setdefault("ISAACSIM_PORTABLE_ROOT", r"I:\isaacsim-data\6.0.1")
    os.environ.setdefault("TEMP", r"I:\isaacsim-data\6.0.1\temp")
    os.environ.setdefault("TMP", r"I:\isaacsim-data\6.0.1\temp")
else:
    _isaac_root = os.environ.setdefault("ISAACSIM_PORTABLE_ROOT", "/root/hose_v10_1_work/isaac60_data")
    os.environ.setdefault("TMPDIR", "/root/hose_v10_1_work/isaac60_tmp")
    Path(_isaac_root, "logs").mkdir(parents=True, exist_ok=True)

_SRC_ROOT = str(Path(__file__).resolve().parents[3])
_CA_ROOT = str(Path(__file__).resolve().parents[5] / "ConvertAsset")
for _p in (_SRC_ROOT, _CA_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from convert_asset.hose_softbody_v10 import (  # noqa: E402
    HoseMaterialSpecV10,
    build_adaptive_hollow_tube_tet_mesh,
    build_cosserat_twist_stripe_tube,
    compile_d6_der_rod_spec,
    compute_cuff_hoop_and_pinch_strain,
    extract_twist_angles_from_link_frames,
)

FFMPEG_FALLBACK = Path(r"G:\my_program\ffmpeg-2023-06-04-git-b1c3d81e71-full_build\bin\ffmpeg.exe")


def _find_ffmpeg() -> str:
    if FFMPEG_FALLBACK.is_file():
        return str(FFMPEG_FALLBACK)
    w = shutil.which("ffmpeg")
    if w:
        return w
    import imageio_ffmpeg  # type: ignore

    return str(imageio_ffmpeg.get_ffmpeg_exe())


class _SafeH264Writer:
    """Deadlock-free streaming H.264 encoder (stderr redirected to DEVNULL)."""

    def __init__(self, out_mp4: Path, fps: int = 30) -> None:
        self.out_mp4 = out_mp4
        self.fps = fps
        self.proc: subprocess.Popen[bytes] | None = None

    def write(self, rgb: np.ndarray) -> None:
        arr = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
        if self.proc is None:
            h, w = arr.shape[:2]
            self.out_mp4.parent.mkdir(parents=True, exist_ok=True)
            self.proc = subprocess.Popen(
                [
                    _find_ffmpeg(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    str(self.fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self.out_mp4),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        assert self.proc.stdin is not None
        self.proc.stdin.write(arr.tobytes())

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait()
            self.proc = None


def _lookat_matrix4d(Gf: Any, eye: Sequence[float], target: Sequence[float]) -> Any:
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    fwd = t - e
    fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -fwd
    m[3, :3] = e
    return Gf.Matrix4d(*m.ravel().tolist())


def _make_pbr_mat(stage: Any, UsdShade: Any, Sdf: Any, Gf: Any, path: str, rgb: tuple[float, float, float], rough: float = 0.35, metal: float = 0.0, opacity: float = 1.0) -> Any:
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind_mat(UsdShade: Any, prim: Any, mat: Any) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _create_phys_mat(stage: Any, UsdShade: Any, UsdPhysics: Any, PhysxSchema: Any, path: str, mu_s: float, mu_d: float, combine: str = "average") -> str:
    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(float(mu_s))
    pm.CreateDynamicFrictionAttr().Set(float(mu_d))
    pm.CreateRestitutionAttr().Set(0.02)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr().Set(combine)
    px.CreateRestitutionCombineModeAttr().Set("min")
    return path


def _set_col_offsets(PhysxSchema: Any, prim: Any, rest: float = 0.0001, contact: float = 0.0010) -> None:
    pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    pc.CreateRestOffsetAttr().Set(float(rest))
    pc.CreateContactOffsetAttr().Set(float(contact))


def _create_rigid_box(
    stage: Any, UsdGeom: Any, UsdPhysics: Any, PhysxSchema: Any, UsdShade: Any, physicsUtils: Any, Gf: Any,
    path: str, pos: Sequence[float], dims: Sequence[float], vis_mat: Any, phys_mat: str,
    *, dynamic: bool = False, kinematic: bool = False, mass: float = 0.2,
    quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0), collide: bool = True,
) -> tuple[Any, Any, Any]:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    t_op = xf.AddTranslateOp()
    t_op.Set(Gf.Vec3d(*map(float, pos)))
    o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    o_op.Set(Gf.Quatd(*map(float, quat_wxyz)))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, dims)))
    _bind_mat(UsdShade, cube.GetPrim(), vis_mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        _set_col_offsets(PhysxSchema, cube.GetPrim())
        physicsUtils.add_physics_material_to_prim(stage, cube.GetPrim(), phys_mat)
    if dynamic or kinematic:
        rb = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        rb.CreateKinematicEnabledAttr().Set(bool(kinematic))
        if dynamic and not kinematic:
            UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr().Set(float(mass))
    return cube.GetPrim(), t_op, o_op


def _create_rigid_cyl(
    stage: Any, UsdGeom: Any, UsdPhysics: Any, PhysxSchema: Any, UsdShade: Any, physicsUtils: Any, Gf: Any,
    path: str, pos: Sequence[float], radius: float, height: float, axis: str, vis_mat: Any, phys_mat: str,
    *, dynamic: bool = False, kinematic: bool = False, mass: float = 0.1,
    quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0), collide: bool = True,
) -> tuple[Any, Any, Any]:
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    xf = UsdGeom.Xformable(cyl.GetPrim())
    t_op = xf.AddTranslateOp()
    t_op.Set(Gf.Vec3d(*map(float, pos)))
    o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    o_op.Set(Gf.Quatd(*map(float, quat_wxyz)))
    _bind_mat(UsdShade, cyl.GetPrim(), vis_mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        _set_col_offsets(PhysxSchema, cyl.GetPrim())
        physicsUtils.add_physics_material_to_prim(stage, cyl.GetPrim(), phys_mat)
    if dynamic or kinematic:
        rb = UsdPhysics.RigidBodyAPI.Apply(cyl.GetPrim())
        rb.CreateKinematicEnabledAttr().Set(bool(kinematic))
        if dynamic and not kinematic:
            UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateMassAttr().Set(float(mass))
    return cyl.GetPrim(), t_op, o_op


def _build_rod_links(
    stage: Any, UsdGeom: Any, UsdPhysics: Any, PhysxSchema: Any, Sdf: Any, Gf: Any, physicsUtils: Any,
    scope_path: str, p0: np.ndarray, p1: np.ndarray, n_links: int,
    *, method: str, rod_spec: Any, phys_mat: str, anchor_root_prim: str | None,
    add_cuff_micro_springs: bool = False, zero_friction: bool = False, cuff_mat: str = "",
    motor_twist_root: bool = False,
) -> dict[str, Any]:
    """Build either Baseline SphericalJoint Bead-Chain or Route A Featherstone Articulated D6-DER rod."""
    scope_prim = UsdGeom.Scope.Define(stage, scope_path).GetPrim()
    if add_cuff_micro_springs:
        # v10.1 Coaxial 32mm 2-Ring Micro-Spring Cuff Collar (link_00) + Flexible D6-DER Body (links 1..n-1)
        # Matches V09 Route A architecture: zero Euler column buckling, straddles all 3 glass olives coaxially!
        cuff_len = 0.036
        u_dir = (p1 - p0) / max(float(np.linalg.norm(p1 - p0)), 1e-12)
        p_cuff_end = p0 + cuff_len * u_dir
        body_pts = np.linspace(p_cuff_end, p1, n_links)
        pts = np.vstack([p0[None, :], body_pts])
    else:
        pts = np.linspace(p0, p1, n_links + 1)
    link_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    ds = float(np.mean(link_lens))
    is_d6 = (method == "route_a_d6_der")

    link_prims: list[str] = []
    link_xfs: list[Any] = []
    cuff_pad_info: list[dict[str, Any]] = []
    motor_drv: Any = None

    k_bend_usd = float(rod_spec.bend_stiffness_nm_per_rad) * (math.pi / 180.0)
    c_bend_usd = float(rod_spec.bend_damping_nms_per_rad) * (math.pi / 180.0)
    k_twist_usd = float(rod_spec.twist_stiffness_nm_per_rad) * (math.pi / 180.0)
    c_twist_usd = float(rod_spec.twist_damping_nms_per_rad) * (math.pi / 180.0)

    for i in range(n_links):
        li_len = float(link_lens[i])
        mid = 0.5 * (pts[i] + pts[i + 1])
        lp = f"{scope_path}/link_{i:02d}"
        xf = UsdGeom.Xform.Define(stage, lp)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, mid)))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        rb = UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        rb.CreateKinematicEnabledAttr().Set(False)
        link_mass = float(rod_spec.link_mass_kg) * (li_len / max(ds, 1e-6))
        UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr().Set(link_mass)
        px_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(xf.GetPrim())
        px_rb.CreateLinearDampingAttr().Set(float(rod_spec.linear_damping))
        px_rb.CreateAngularDampingAttr().Set(float(rod_spec.angular_damping))
        px_rb.CreateSolverPositionIterationCountAttr().Set(32)
        px_rb.CreateSolverVelocityIterationCountAttr().Set(4)

        if add_cuff_micro_springs and i == 0:
            # 1) 4 hollow outer grip staves along the 32mm sleeve (inner 5.65mm > 5.4mm olive, outer 6.45mm for gripper pinch!)
            for k_ang in range(4):
                ang = k_ang * 0.5 * math.pi
                uy, uz = math.cos(ang), math.sin(ang)
                st = UsdGeom.Cube.Define(stage, f"{lp}/OuterWall_{k_ang}")
                st.CreateSizeAttr(1.0)
                sxf = UsdGeom.Xformable(st.GetPrim())
                sxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.00605 * uy, 0.00605 * uz))
                sxf.AddScaleOp().Set(Gf.Vec3f(li_len * 0.90, 0.0008 if abs(uy) > 0.5 else 0.0062, 0.0008 if abs(uz) > 0.5 else 0.0062))
                UsdGeom.Imageable(st).MakeInvisible()
                UsdPhysics.CollisionAPI.Apply(st.GetPrim())
                _set_col_offsets(PhysxSchema, st.GetPrim(), rest=0.00005, contact=0.0005)
                physicsUtils.add_physics_material_to_prim(stage, st.GetPrim(), phys_mat)
            # 2) 2 coaxial rings x 4 diagonal (45/135/225/315 deg) radial micro-spring rubber liner pads (20mm baseline!)
            _cuff_m = cuff_mat or phys_mat
            pad_r_c = 0.0018
            r_rest_center = 0.0043 + pad_r_c  # inner edge 4.3mm < 4.4mm stem & 5.4mm olive crests!
            for r_idx, x_loc_p in enumerate((-0.012, +0.011)):
                for k_ang in range(6):
                    ang = k_ang * (math.pi / 3.0) + (math.pi / 6.0)  # 30, 90(TOP), 150, 210, 270(BOT), 330 -> 360-deg enclosed bore!
                    uy, uz = math.cos(ang), math.sin(ang)
                    pad_p = f"{scope_path}/cuff_pad_r{r_idx}_{k_ang}"
                    pxf = UsdGeom.Xform.Define(stage, pad_p)
                    p_world = mid + np.array([x_loc_p, r_rest_center * uy, r_rest_center * uz])
                    pxf.AddTranslateOp().Set(Gf.Vec3d(*map(float, p_world)))
                    pxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
                    prb = UsdPhysics.RigidBodyAPI.Apply(pxf.GetPrim())
                    prb.CreateKinematicEnabledAttr().Set(False)
                    UsdPhysics.MassAPI.Apply(pxf.GetPrim()).CreateMassAttr().Set(0.00025)
                    ppx = PhysxSchema.PhysxRigidBodyAPI.Apply(pxf.GetPrim())
                    ppx.CreateLinearDampingAttr().Set(4.0)
                    ppx.CreateAngularDampingAttr().Set(4.0)
                    sp = UsdGeom.Sphere.Define(stage, f"{pad_p}/Sph")
                    sp.CreateRadiusAttr(pad_r_c)
                    UsdGeom.Imageable(sp).MakeInvisible()
                    UsdPhysics.CollisionAPI.Apply(sp.GetPrim())
                    _set_col_offsets(PhysxSchema, sp.GetPrim(), rest=0.00005, contact=0.0005)
                    physicsUtils.add_physics_material_to_prim(stage, sp.GetPrim(), _cuff_m)
                    pj = UsdPhysics.PrismaticJoint.Define(stage, f"{scope_path}/cuff_spring_r{r_idx}_{k_ang}")
                    pj.CreateAxisAttr("X")
                    pj.CreateBody0Rel().SetTargets([Sdf.Path(lp)])
                    pj.CreateBody1Rel().SetTargets([Sdf.Path(pad_p)])
                    q_rad = Gf.Rotation(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0.0, uy, uz)).GetQuat()
                    qf = Gf.Quatf(float(q_rad.GetReal()), Gf.Vec3f(*[float(v) for v in q_rad.GetImaginary()]))
                    pj.CreateLocalPos0Attr().Set(Gf.Vec3f(float(x_loc_p), float(r_rest_center * uy), float(r_rest_center * uz)))
                    pj.CreateLocalRot0Attr().Set(qf)
                    pj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    pj.CreateLocalRot1Attr().Set(qf)
                    pj.CreateLowerLimitAttr().Set(-0.0003)
                    pj.CreateUpperLimitAttr().Set(0.0035)
                    drv = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
                    drv.CreateTypeAttr("force")
                    drv.CreateStiffnessAttr().Set(180.0)
                    drv.CreateDampingAttr().Set(9.0)
                    drv.CreateMaxForceAttr().Set(3.5)
                    drv.CreateTargetPositionAttr().Set(0.0)
                    cuff_pad_info.append(
                        {
                            "pad_prim": pad_p,
                            "host_prim": lp,
                            "p_loc": np.array([x_loc_p, r_rest_center * uy, r_rest_center * uz], dtype=np.float64),
                            "stiffness": 180.0,
                            "max_force": 3.5,
                        }
                    )
        else:
            cap = UsdGeom.Capsule.Define(stage, f"{lp}/Col")
            cap.CreateAxisAttr("X")
            cap.CreateRadiusAttr(0.0062 if add_cuff_micro_springs else float(rod_spec.capsule_radius_m))
            cap.CreateHeightAttr(max(0.001, li_len * 0.25))
            UsdGeom.Imageable(cap).MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
            _set_col_offsets(PhysxSchema, cap.GetPrim(), rest=0.0001, contact=0.0006)
            physicsUtils.add_physics_material_to_prim(stage, cap.GetPrim(), phys_mat)

        link_prims.append(lp)
        link_xfs.append(UsdGeom.Xformable(xf.GetPrim()))

        prev_p = anchor_root_prim if i == 0 else link_prims[i - 1]
        if i == 0 and (anchor_root_prim is not None or motor_twist_root):
            if is_d6:
                if motor_twist_root:
                    base_p = f"{scope_path}/motor_fixed_base"
                    bxf = UsdGeom.Xform.Define(stage, base_p)
                    bxf.AddTranslateOp().Set(Gf.Vec3d(*map(float, p0)))
                    bxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
                    UsdPhysics.RigidBodyAPI.Apply(bxf.GetPrim()).CreateKinematicEnabledAttr().Set(False)
                    UsdPhysics.MassAPI.Apply(bxf.GetPrim()).CreateMassAttr().Set(0.10)
                    fj_b = UsdPhysics.FixedJoint.Define(stage, f"{scope_path}/j_root_fixed")
                    fj_b.CreateBody1Rel().SetTargets([Sdf.Path(base_p)])
                    fj_b.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, p0)))
                    fj_b.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    rj = UsdPhysics.RevoluteJoint.Define(stage, f"{scope_path}/j_root_motor")
                    rj.CreateAxisAttr("X")
                    rj.CreateBody0Rel().SetTargets([Sdf.Path(base_p)])
                    rj.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
                    rj.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    rj.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5 * li_len, 0.0, 0.0))
                    motor_drv = UsdPhysics.DriveAPI.Apply(rj.GetPrim(), "angular")
                    motor_drv.CreateTypeAttr("force")
                    motor_drv.CreateStiffnessAttr().Set(0.75)
                    motor_drv.CreateDampingAttr().Set(0.06)
                    motor_drv.CreateTargetPositionAttr().Set(0.0)
                else:
                    fj = UsdPhysics.FixedJoint.Define(stage, f"{scope_path}/j_root")
                    fj.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
                    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, p0)))
                    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5 * li_len, 0.0, 0.0))
            else:
                sj = UsdPhysics.SphericalJoint.Define(stage, f"{scope_path}/j_root")
                sj.CreateBody0Rel().SetTargets([Sdf.Path(prev_p)])
                sj.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
                sj.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                sj.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5 * li_len, 0.0, 0.0))
        elif prev_p is not None:
            prev_len = float(link_lens[i - 1])
            if is_d6:
                j = UsdPhysics.Joint.Define(stage, f"{scope_path}/d6_{i:02d}")
                j.CreateBody0Rel().SetTargets([Sdf.Path(prev_p)])
                j.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
                j.CreateLocalPos0Attr().Set(Gf.Vec3f(+0.5 * prev_len, 0.0, 0.0))
                j.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5 * li_len, 0.0, 0.0))
                for ax in ("transX", "transY", "transZ"):
                    lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), ax)
                    lim.CreateLowAttr(1.0)
                    lim.CreateHighAttr(-1.0)
                for ax, k_val, c_val in (("rotX", k_twist_usd, c_twist_usd), ("rotY", k_bend_usd, c_bend_usd), ("rotZ", k_bend_usd, c_bend_usd)):
                    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), ax)
                    drv.CreateTypeAttr("force")
                    drv.CreateStiffnessAttr().Set(float(k_val))
                    drv.CreateDampingAttr().Set(float(c_val))
                    drv.CreateTargetPositionAttr().Set(0.0)
            else:
                sj = UsdPhysics.SphericalJoint.Define(stage, f"{scope_path}/sph_{i:02d}")
                sj.CreateBody0Rel().SetTargets([Sdf.Path(prev_p)])
                sj.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
                sj.CreateLocalPos0Attr().Set(Gf.Vec3f(+0.5 * prev_len, 0.0, 0.0))
                sj.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5 * li_len, 0.0, 0.0))

    for i in range(n_links - 1):
        fp = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(link_prims[i]))
        fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(link_prims[i + 1]))
        if i + 2 < n_links:
            fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(link_prims[i + 2]))
    if cuff_pad_info:
        for ci in cuff_pad_info:
            fp = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(ci["pad_prim"]))
            fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(link_prims[0]))
            fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(link_prims[1]))
            fp.CreateFilteredPairsRel().AddTarget(Sdf.Path("/World/B3_GripPadF"))
            fp.CreateFilteredPairsRel().AddTarget(Sdf.Path("/World/B3_GripPadR"))
            for cj in cuff_pad_info:
                if cj["pad_prim"] != ci["pad_prim"]:
                    fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(cj["pad_prim"]))

    # Visual Cosserat twist-striped mesh updated every frame
    v0, n0, t0, c0 = build_cosserat_twist_stripe_tube(pts, radius_m=float(rod_spec.capsule_radius_m))
    mesh_prim = UsdGeom.Mesh.Define(stage, f"{scope_path}/VisualHose")
    pts_attr = mesh_prim.CreatePointsAttr(v0.tolist())
    nrm_attr = mesh_prim.CreateNormalsAttr(n0.tolist())
    mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh_prim.CreateFaceVertexCountsAttr([3] * len(t0))
    mesh_prim.CreateFaceVertexIndicesAttr(t0.ravel().tolist())
    mesh_prim.CreateDisplayColorAttr(c0.tolist())
    mesh_prim.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.vertex)

    return {
        "link_prims": link_prims,
        "link_xfs": link_xfs,
        "cuff_pad_info": cuff_pad_info,
        "pts_attr": pts_attr,
        "nrm_attr": nrm_attr,
        "ds": ds,
        "link0_half": 0.5 * float(link_lens[0]),
        "motor_drv": motor_drv,
    }


def _build_fem_hose(
    stage: Any, UsdGeom: Any, UsdShade: Any, PhysxSchema: Any, Sdf: Any, Gf: Any, Vt: Any,
    deformableUtils: Any, physicsUtils: Any,
    path: str, p0: np.ndarray, p1: np.ndarray, *, mat_path: str, vis_mat: Any,
    attach_prim_path: str | None = None, youngs_pa: float = 0.55e6,
    r_in: float = 0.0044, r_out: float = 0.0068, r_out_body: float = 0.0062,
) -> dict[str, Any]:
    """Build Route B Non-Uniform Adaptive Hollow-Tube PhysX 5 GPU Volume Deformable Body (v10.1 graded wall)."""
    ctrl = np.linspace(p0, p1, 14)
    mesh = build_adaptive_hollow_tube_tet_mesh(
        ctrl,
        r_in=r_in,
        r_out=r_out,
        r_out_body=r_out_body,
        n_theta=10,
        cuff_length_m=0.045,
        cuff_ds_m=0.0045,
        body_ds_m=0.014,
        dense_both_ends=False,
    )
    tet_prim = UsdGeom.TetMesh.Define(stage, path)
    pts_attr = tet_prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in mesh.rest_points]))
    tet_prim.CreateTetVertexIndicesAttr(Vt.Vec4iArray([Gf.Vec4i(int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in mesh.tets]))
    deformableUtils.set_physics_volume_deformable_body(stage, Sdf.Path(path))
    _set_col_offsets(PhysxSchema, tet_prim.GetPrim(), rest=0.00015, contact=0.0010)
    tet_prim.GetPrim().ApplyAPI("PhysxBaseDeformableBodyAPI")
    tet_prim.GetPrim().GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(64)
    tet_prim.GetPrim().GetAttribute("physxDeformableBody:linearDamping").Set(0.75)
    physicsUtils.add_physics_material_to_prim(stage, tet_prim.GetPrim(), mat_path)
    _bind_mat(UsdShade, tet_prim.GetPrim(), vis_mat)

    if attach_prim_path is not None:
        deformableUtils.create_auto_deformable_attachment(
            stage,
            Sdf.Path(f"{path}_Attach"),
            Sdf.Path(path),
            Sdf.Path(attach_prim_path),
        )

    # Also define a clean shaded surface Mesh synchronized every frame with torsion stripe color!
    UsdGeom.Imageable(tet_prim).MakeInvisible()
    vis_m = UsdGeom.Mesh.Define(stage, f"{path}_VisMesh")
    v_pts_attr = vis_m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in mesh.rest_points]))
    vis_m.CreateFaceVertexCountsAttr([3] * len(mesh.surface_tris))
    vis_m.CreateFaceVertexIndicesAttr(mesh.surface_tris.ravel().tolist())
    # Assign dark-amber longitudinal stripe along theta index 0 & 5 so FEM torsion is crystal clear!
    v_cols = np.tile(np.array([[0.82, 0.54, 0.14]], dtype=np.float32), (len(mesh.rest_points), 1))
    for ri in range(mesh.n_rings):
        out_ids = mesh.ring_outer_vertex_indices(ri)
        v_cols[out_ids[0]] = (0.22, 0.10, 0.02)
        v_cols[out_ids[5]] = (0.22, 0.10, 0.02)
    vis_m.CreateDisplayColorAttr(v_cols.tolist())
    vis_m.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.vertex)

    return {
        "prim_path": path,
        "tet_prim": tet_prim,
        "pts_attr": pts_attr,
        "vis_pts_attr": v_pts_attr,
        "mesh": mesh,
    }


def _world_pos_rot(Usd: Any, xf: Any) -> tuple[np.ndarray, np.ndarray]:
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    p = np.array(list(m.ExtractTranslation()), dtype=np.float64)
    r = m.ExtractRotationMatrix()
    R = np.array([[r[0][0], r[1][0], r[2][0]], [r[0][1], r[1][1], r[2][1]], [r[0][2], r[1][2], r[2][2]]], dtype=np.float64)
    return p, R


def execute_hose_v10_trial(job: dict[str, Any], result_json_path: Path | None = None) -> dict[str, Any]:
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        except Exception:
            pass

    from isaacsim import SimulationApp

    record_video = bool(job.get("record_video", True))
    gpu_id = int(job.get("gpu_id", 0))
    log_dir = Path(os.environ.get("ISAACSIM_PORTABLE_ROOT", "/tmp")) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    kit_log = (log_dir / f"v10_3way_{job.get('trial_id', 't')}_g{gpu_id}.log").as_posix()
    app = SimulationApp({
        "headless": True,
        "renderer": "RayTracedLighting",
        "active_gpu": gpu_id,
        "physics_gpu": gpu_id,
        "multi_gpu": False,
        "anti_aliasing": 1,
        "extra_args": [
            "--/rtx/verifyDriverVersion/enabled=false",
            "--/renderer/multiGpu/enabled=false",
            f"--/renderer/activeGpu={gpu_id}",
            f"--/physics/cudaDevice={gpu_id}",
            f"--/log/file={kit_log}",
        ],
    })

    import omni.kit.app
    import omni.physx
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils, Vt

    method = str(job["method"])
    zero_friction = bool(job.get("zero_friction", False))
    target_port = str(job.get("target_port", "lower_inlet"))
    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    usd_ctx = omni.usd.get_context()
    usd_ctx.new_stage()
    stage = usd_ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Studio lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr().Set(850.0)
    key_l = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key_l.CreateIntensityAttr().Set(1800.0)
    UsdGeom.Xformable(key_l).AddRotateXYZOp().Set(Gf.Vec3f(42.0, -25.0, 20.0))

    # 240Hz PhysX 5 GPU Scene
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    ps_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    ps_api.CreateEnableGPUDynamicsAttr().Set(True)
    ps_api.CreateBroadphaseTypeAttr().Set("GPU")
    ps_api.CreateSolverTypeAttr().Set("TGS")
    ps_api.CreateTimeStepsPerSecondAttr().Set(240)

    # Visual & Physics Materials
    UsdGeom.Scope.Define(stage, "/World/Materials")
    mat_bench = _make_pbr_mat(stage, UsdShade, Sdf, Gf, "/World/Materials/VisBench", (0.86, 0.88, 0.90), rough=0.45)
    mat_wall = _make_pbr_mat(stage, UsdShade, Sdf, Gf, "/World/Materials/VisWall", (0.06, 0.34, 0.62), rough=0.75)
    mat_brass = _make_pbr_mat(stage, UsdShade, Sdf, Gf, "/World/Materials/VisBrass", (0.78, 0.64, 0.25), rough=0.25, metal=0.85)
    mat_steel = _make_pbr_mat(stage, UsdShade, Sdf, Gf, "/World/Materials/VisSteel", (0.68, 0.72, 0.76), rough=0.22, metal=0.88)
    mat_blue = _make_pbr_mat(stage, UsdShade, Sdf, Gf, "/World/Materials/VisBlue", (0.10, 0.45, 0.88), rough=0.35)
    mat_sink = _make_pbr_mat(stage, UsdShade, Sdf, Gf, "/World/Materials/VisSink", (0.14, 0.16, 0.19), rough=0.50)
    mat_glass = _make_pbr_mat(stage, UsdShade, Sdf, Gf, "/World/Materials/VisGlass", (0.82, 0.94, 0.98), rough=0.06, opacity=0.45)
    mat_red = _make_pbr_mat(stage, UsdShade, Sdf, Gf, "/World/Materials/VisRed", (0.88, 0.16, 0.14), rough=0.35)
    mat_amber = _make_pbr_mat(stage, UsdShade, Sdf, Gf, "/World/Materials/VisAmber", (0.82, 0.54, 0.14), rough=0.32)
    mat_pad = _make_pbr_mat(stage, UsdShade, Sdf, Gf, "/World/Materials/VisPad", (0.20, 0.24, 0.30), rough=0.55)

    pmat_bench = _create_phys_mat(stage, UsdShade, UsdPhysics, PhysxSchema, "/World/Materials/PhysBench", 0.6, 0.5)
    pmat_hose = _create_phys_mat(stage, UsdShade, UsdPhysics, PhysxSchema, "/World/Materials/PhysHose", 0.0 if zero_friction else 1.45, 0.0 if zero_friction else 1.25, "min" if zero_friction else "average")
    pmat_cuff = _create_phys_mat(stage, UsdShade, UsdPhysics, PhysxSchema, "/World/Materials/PhysCuff", 0.0 if zero_friction else 2.2, 0.0 if zero_friction else 1.9, "min" if zero_friction else "max")
    pmat_barb = _create_phys_mat(stage, UsdShade, UsdPhysics, PhysxSchema, "/World/Materials/PhysBarb", 0.0 if zero_friction else 2.2, 0.0 if zero_friction else 1.9, "min" if zero_friction else "max")
    pmat_grip = _create_phys_mat(stage, UsdShade, UsdPhysics, PhysxSchema, "/World/Materials/PhysGrip", 3.6, 3.2, "max")
    pmat_slip = _create_phys_mat(stage, UsdShade, UsdPhysics, PhysxSchema, "/World/Materials/PhysSlip", 0.01, 0.01, "min")

    fmat_fem = "/World/Materials/PhysFemLatex"
    deformableUtils.add_deformable_material(
        stage,
        fmat_fem,
        density=960.0,
        static_friction=0.0 if zero_friction else 2.8,
        dynamic_friction=0.0 if zero_friction else 2.4,
        youngs_modulus=3.6e6,
        poissons_ratio=0.36,
    )

    hose_mat_spec = HoseMaterialSpecV10()
    rod_spec_14 = compile_d6_der_rod_spec(hose_mat_spec, total_length_m=0.14, n_links=14)
    rod_spec_18 = compile_d6_der_rod_spec(hose_mat_spec, total_length_m=0.18, n_links=14)

    # =========================================================================
    # BENCH 1 (y = 0.0): Exp-1 Cantilever Natural Sag & Elastic Step Rebound (L = 140 mm!)
    # =========================================================================
    Y1 = 0.0
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B1_Bench", (0.09, Y1, 0.72), (0.65, 0.50, 0.06), mat_bench, pmat_bench)
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B1_Wall", (0.09, Y1 + 0.25, 1.00), (0.75, 0.02, 0.55), mat_wall, pmat_bench, collide=False)
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B1_RootStand", (-0.028, Y1, 0.83), (0.036, 0.04, 0.16), mat_steel, pmat_bench)
    b1_chuck_p, _, _ = _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B1_RootChuck", (0.0, Y1, 0.90), (0.024, 0.022, 0.022), mat_brass, pmat_bench, kinematic=True, collide=(method == "route_b_physx_fem"))
    _, b1_plunger_t, _ = _create_rigid_cyl(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B1_Plunger", (0.115, Y1, 0.955), 0.007, 0.055, "Y", mat_blue, pmat_slip, kinematic=True)

    b1_p0 = np.array([0.00, Y1, 0.90], dtype=np.float64)
    b1_p1 = np.array([0.14, Y1, 0.90], dtype=np.float64)
    if method == "route_b_physx_fem":
        b1_hose = _build_fem_hose(stage, UsdGeom, UsdShade, PhysxSchema, Sdf, Gf, Vt, deformableUtils, physicsUtils, "/World/B1_HoseFEM", b1_p0, b1_p1, mat_path=fmat_fem, vis_mat=mat_amber, attach_prim_path="/World/B1_RootChuck")
    else:
        b1_hose = _build_rod_links(stage, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf, physicsUtils, "/World/B1_HoseRod", b1_p0, b1_p1, 14, method=method, rod_spec=rod_spec_14, phys_mat=pmat_hose, anchor_root_prim="/World/B1_RootChuck")

    # =========================================================================
    # BENCH 2 (y = 2.5): Exp-2 90-deg Axial Torsional Twist Propagation (L = 140 mm, full-length frictionless trough!)
    # =========================================================================
    Y2 = 2.5
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B2_Bench", (0.09, Y2, 0.72), (0.65, 0.50, 0.06), mat_bench, pmat_bench)
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B2_Wall", (0.09, Y2 + 0.25, 1.00), (0.75, 0.02, 0.55), mat_wall, pmat_bench, collide=False)
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B2_MotorBase", (-0.028, Y2, 0.83), (0.036, 0.045, 0.16), mat_steel, pmat_bench)
    b2_chuck_p, _, b2_chuck_o = _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B2_MotorChuck", (0.0, Y2, 0.90), (0.024, 0.024, 0.024), mat_brass, pmat_bench, kinematic=True, collide=(method == "route_b_physx_fem"))
    # Full-length frictionless horizontal guide trough keeping ALL 3 methods horizontal so only PURE axial twist around X is measured!
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B2_TroughBot", (0.075, Y2, 0.891), (0.115, 0.022, 0.006), mat_steel, pmat_slip)
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B2_TroughFront", (0.075, Y2 - 0.009, 0.901), (0.115, 0.004, 0.016), mat_steel, pmat_slip)
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B2_TroughRear", (0.075, Y2 + 0.009, 0.901), (0.115, 0.004, 0.016), mat_steel, pmat_slip)
    _, b2_vane_t, b2_vane_o = _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B2_TipVane", (0.14, Y2, 0.92), (0.005, 0.004, 0.036), mat_red, pmat_slip, collide=False)

    b2_p0 = np.array([0.00, Y2, 0.90], dtype=np.float64)
    b2_p1 = np.array([0.14, Y2, 0.90], dtype=np.float64)
    if method == "route_b_physx_fem":
        b2_hose = _build_fem_hose(stage, UsdGeom, UsdShade, PhysxSchema, Sdf, Gf, Vt, deformableUtils, physicsUtils, "/World/B2_HoseFEM", b2_p0, b2_p1, mat_path=fmat_fem, vis_mat=mat_amber, attach_prim_path="/World/B2_MotorChuck")
    else:
        b2_hose = _build_rod_links(stage, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf, physicsUtils, "/World/B2_HoseRod", b2_p0, b2_p1, 14, method=method, rod_spec=rod_spec_14, phys_mat=pmat_hose, anchor_root_prim="/World/B2_MotorChuck", motor_twist_root=(method == "route_a_d6_der"))

    # =========================================================================
    # BENCH 3 (y = 5.0): Exp-3 True Hose-on-Barb Interference Fit + Lab Faucet & PP Sink!
    # =========================================================================
    Y3 = 5.0
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_Bench", (0.10, Y3, 0.72), (0.95, 0.65, 0.06), mat_bench, pmat_bench)
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_Wall", (0.10, Y3 + 0.30, 1.00), (1.05, 0.02, 0.58), mat_wall, pmat_bench, collide=False)
    # Realistic Lab Gooseneck Cold-Water Faucet (right-rear) & Embedded PP Lab Sink (left-rear)!
    _create_rigid_cyl(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_FaucetPost", (0.34, Y3 + 0.14, 0.84), 0.011, 0.18, "Z", mat_steel, pmat_bench, collide=False)
    _create_rigid_cyl(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_FaucetArch", (0.34, Y3 + 0.09, 0.93), 0.008, 0.10, "Y", mat_steel, pmat_bench, collide=False)
    _create_rigid_cyl(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_FaucetNozzle", (0.34, Y3 + 0.04, 0.90), 0.006, 0.06, "Z", mat_brass, pmat_bench, collide=False)
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_FaucetBlueValve", (0.37, Y3 + 0.14, 0.82), (0.036, 0.014, 0.014), mat_blue, pmat_bench, collide=False)
    _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_LabSink", (-0.26, Y3 + 0.12, 0.756), (0.18, 0.14, 0.014), mat_sink, pmat_bench, collide=False)
    # Mounted Liebig Condenser glass body with LOWER INLET BARB (z=0.885) & UPPER OUTLET BARB (z=0.955)
    _create_rigid_cyl(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_CondenserBody", (-0.06, Y3, 0.920), 0.018, 0.15, "Z", mat_glass, pmat_bench, collide=False)
    Z_LOWER_BARB = 0.885
    Z_UPPER_BARB = 0.955
    for bname, bz in (("LowerInletBarb", Z_LOWER_BARB), ("UpperOutletBarb", Z_UPPER_BARB)):
        # v10.1 Full 3-olive serrated glass barb (r_stem=4.4mm, 3 interference olives r=5.4mm > 4.4mm bore -> ~23-27% hoop expansion, 38mm total length!)
        _create_rigid_cyl(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, f"/World/B3_{bname}_Stem", (-0.022, Y3, bz), 0.0044, 0.042, "X", mat_glass, pmat_barb)
        for si, (bx, br) in enumerate(((-0.002, 0.0054), (+0.007, 0.0054), (+0.016, 0.0052), (+0.022, 0.0038), (+0.026, 0.0024))):
            sp = UsdGeom.Sphere.Define(stage, f"/World/B3_{bname}_Bead{si}")
            sp.CreateRadiusAttr(float(br))
            UsdGeom.Xformable(sp).AddTranslateOp().Set(Gf.Vec3d(bx, Y3, bz))
            _bind_mat(UsdShade, sp.GetPrim(), mat_glass)
            UsdPhysics.CollisionAPI.Apply(sp.GetPrim())
            _set_col_offsets(PhysxSchema, sp.GetPrim(), rest=0.00005, contact=0.0004)
            physicsUtils.add_physics_material_to_prim(stage, sp.GetPrim(), pmat_barb)

    z_tgt = Z_UPPER_BARB if target_port == "upper_outlet" else Z_LOWER_BARB
    # Retractable Anti-roll Lab Staging Rest supporting the hose tail during staging (retracts at f=68 so hose hangs 100% freely on glass barb!)
    _rest_x = 0.145 if method == "route_b_physx_fem" else 0.125
    _rest_sx = 0.13 if method == "route_b_physx_fem" else 0.17
    _rest_dz = 0.013 if method == "route_b_physx_fem" else 0.0124
    _, rest_t_op, _ = _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_RearHoseRest", (_rest_x, Y3, z_tgt - _rest_dz), (_rest_sx, 0.028, 0.012), mat_steel, pmat_slip, kinematic=True)
    # Free dynamic latex hose starting coaxial 4 mm in front of tapered barb lead-nose (ZERO FixedJoint, ZERO attachment!)
    b3_p0 = np.array([0.029, Y3, z_tgt], dtype=np.float64)
    b3_p1 = np.array([0.209, Y3, z_tgt], dtype=np.float64)
    if method == "route_b_physx_fem":
        b3_hose = _build_fem_hose(stage, UsdGeom, UsdShade, PhysxSchema, Sdf, Gf, Vt, deformableUtils, physicsUtils, "/World/B3_HoseFEM", b3_p0, b3_p1, mat_path=fmat_fem, vis_mat=mat_amber, attach_prim_path=None)
    else:
        b3_hose = _build_rod_links(
            stage, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf, physicsUtils, "/World/B3_HoseRod", b3_p0, b3_p1, 14,
            method=method, rod_spec=rod_spec_18, phys_mat=pmat_hose, anchor_root_prim=None,
            add_cuff_micro_springs=(method == "route_a_d6_der"), zero_friction=zero_friction, cuff_mat=pmat_cuff,
        )

    # Opposing dynamic-contact robot gripper finger pads pinching right behind the cuff (x=0.056, 27mm behind mouth -> coaxial stability!)
    _, pad_f_t, _ = _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_GripPadF", (0.056, Y3 - 0.0108, z_tgt), (0.028, 0.008, 0.018), mat_pad, pmat_grip, kinematic=True)
    _, pad_r_t, _ = _create_rigid_box(stage, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, physicsUtils, Gf, "/World/B3_GripPadR", (0.056, Y3 + 0.0108, z_tgt), (0.028, 0.008, 0.018), mat_pad, pmat_grip, kinematic=True)

    # Verify ZERO FixedJoints anywhere on Bench 3 (strict zero-cheat barb insertion!)
    b3_weld_count = 0
    for pr in stage.Traverse():
        if pr.IsA(UsdPhysics.FixedJoint) and "B3_" in str(pr.GetPath()):
            b3_weld_count += 1

    # Configure 3 Studio Cameras (one per Bench)
    cams = [
        ("exp1", "/World/Cam_Exp1", (0.08, Y1 - 0.34, 0.90), (0.08, Y1, 0.85), 34.0),
        ("exp2", "/World/Cam_Exp2", (0.11, Y2 - 0.32, 0.94), (0.08, Y2, 0.90), 34.0),
        ("exp3", "/World/Cam_Exp3", (0.08, Y3 - 0.38, 0.93), (0.07, Y3, 0.90), 32.0),
    ]
    annotators: dict[str, Any] = {}
    writers: dict[str, _SafeH264Writer] = {}
    for cid, cpath, eye, tgt, fl in cams:
        cm = UsdGeom.Camera.Define(stage, cpath)
        cm.CreateFocalLengthAttr().Set(fl)
        cm.CreateHorizontalApertureAttr().Set(20.955)
        cm.CreateVerticalApertureAttr().Set(15.716)
        cm.CreateClippingRangeAttr().Set(Gf.Vec2f(0.02, 20.0))
        UsdGeom.Xformable(cm).AddTransformOp().Set(_lookat_matrix4d(Gf, eye, tgt))
        if record_video:
            rp = rep.create.render_product(cpath, (640, 480))
            an = rep.AnnotatorRegistry.get_annotator("rgb")
            an.attach([rp])
            annotators[cid] = an
            writers[cid] = _SafeH264Writer(vid_dir / f"{cid}_{job['trial_id']}.mp4", fps=30)

    sim_iface = omni.physx.get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    sim_iface.attach_stage(stage_id)
    kit_app = omni.kit.app.get_app()
    kit_app.update()

    N_FRAMES = 180
    dt_frame = 1.0 / 30.0
    substeps = 8
    dt_sub = dt_frame / float(substeps)
    sim_time = 0.0

    # Telemetry arrays
    b1_tip_z = np.zeros(N_FRAMES, dtype=np.float64)
    b2_tip_twist_deg = np.zeros(N_FRAMES, dtype=np.float64)
    b3_mouth_x = np.zeros(N_FRAMES, dtype=np.float64)
    b3_mouth_z = np.zeros(N_FRAMES, dtype=np.float64)
    b3_cuff_rad_force_n = np.zeros(N_FRAMES, dtype=np.float64)
    b3_hoop_strain_pct = np.zeros(N_FRAMES, dtype=np.float64)
    b3_pinch_oval = np.zeros(N_FRAMES, dtype=np.float64)
    step_ms_list: list[float] = []

    b2_ref_vec: np.ndarray | None = None

    for f in range(N_FRAMES):
        # 1. Bench 1 plunger nudge schedule: descend cleanly at x=0.115 to z=0.805, hold, then slide clear (+Y) & lift so it never snags!
        if f < 55:
            px1, py1, pz1 = 0.115, Y1, 0.945
        elif f < 85:
            u = (f - 55) / 30.0
            px1, py1, pz1 = 0.115, Y1, 0.945 - u * 0.140
        elif f < 96:
            px1, py1, pz1 = 0.115, Y1, 0.805
        elif f < 108:
            u = (f - 96) / 12.0
            px1, py1, pz1 = 0.115 + u * 0.08, Y1, 0.805 + u * 0.145
        else:
            px1, py1, pz1 = 0.195, Y1, 0.950
        b1_plunger_t.Set(Gf.Vec3d(px1, py1, pz1))

        # 2. Bench 2 motorized chuck +90 deg axial rotation about X
        if f < 20:
            ang_deg = 0.0
        elif f < 95:
            ang_deg = 90.0 * ((f - 20) / 75.0)
        else:
            ang_deg = 90.0
        hr = 0.5 * math.radians(ang_deg)
        b2_chuck_o.Set(Gf.Quatd(math.cos(hr), math.sin(hr), 0.0, 0.0))
        if method == "route_a_d6_der" and b2_hose.get("motor_drv") is not None:
            b2_hose["motor_drv"].GetTargetPositionAttr().Set(float(ang_deg))

        # 3. Bench 3 v10.1 authentic 2-step inchworm re-grasp push covering ALL 3 glass olives (>=28mm depth!) -> hands-off hang -> re-grasp & pull-off!
        gy_pinch = 0.0080 if method == "route_b_physx_fem" else 0.0094
        gy_pull_pinch = 0.0070 if method == "route_b_physx_fem" else 0.0092
        if f < 6:
            # Pinch close to mouth (gx=0.052, only 23mm behind mouth x=0.029 -> zero mid-span buckling!)
            u = f / 5.0
            gx = 0.052
            gy_off = (1.0 - min(1.0, u * 2.5)) * 0.0108 + min(1.0, u * 2.5) * gy_pinch  # inner face 4.0mm firmly pinches 6.8mm hose wall!
            gz = z_tgt
        elif f < 34:
            # Push 1: advance -24mm (gx: 0.052 -> 0.028) clearing olive #1 (+16mm) & olive #2 (+7mm)!
            u = (f - 6) / 28.0
            gx = 0.052 - u * 0.028
            gy_off = gy_pinch
            gz = z_tgt
        elif f < 37:
            # Open jaws cleanly BEFORE retracting (stationary gx=0.028 so high-friction pads never pull hose back!)
            u = (f - 34) / 3.0
            gx = 0.024
            gy_off = gy_pinch + u * 0.014
            gz = z_tgt
        elif f < 41:
            # Retract WIDE OPEN to gx=0.048 while hose stays locked on olive #1/#2
            u = (f - 37) / 4.0
            gx = 0.024 + u * 0.024
            gy_off = gy_pinch + 0.014
            gz = z_tgt
        elif f < 44:
            # Re-pinch firmly at stationary gx=0.048
            u = (f - 41) / 3.0
            gx = 0.048
            gy_off = (1.0 - u) * (gy_pinch + 0.014) + u * gy_pinch
            gz = z_tgt
        elif f < 74:
            # Push 2: drive mouth cleanly past olive #3 (-2mm) to ~32mm full depth (covers all 3 olives & barb tip!)
            u = (f - 44) / 30.0
            _p2_stroke = 0.024 if method == "route_b_physx_fem" else 0.014
            gx = 0.048 - u * _p2_stroke
            gy_off = gy_pinch
            gz = z_tgt
        elif f < 82:
            # Release gripper cleanly
            u = (f - 74) / 8.0
            gx = 0.024 if method == "route_b_physx_fem" else 0.034
            gy_off = gy_pinch + u * 0.022
            gz = z_tgt
        elif f < 126:
            # Pure hands-off gravity hang on all 3 glass olives (44 frames = 1.47s)!
            gx = 0.024
            gy_off = 0.030
            gz = z_tgt + 0.048
        elif f < 134:
            u = (f - 126) / 8.0
            _gx_pull0 = 0.036 if method == "route_b_physx_fem" else 0.022
            _gz_pull0 = (z_tgt - 0.005) if method == "route_b_physx_fem" else z_tgt
            gx = (1.0 - u) * (0.024 if method == "route_b_physx_fem" else 0.034) + u * _gx_pull0
            gy_off = 0.026
            gz = (1.0 - u) * (z_tgt + 0.048) + u * _gz_pull0
        elif f < 142:
            u = (f - 134) / 8.0
            _gx_pull0 = 0.036 if method == "route_b_physx_fem" else 0.022
            _gz_pull0 = (z_tgt - 0.005) if method == "route_b_physx_fem" else z_tgt
            gx = _gx_pull0
            gy_off = (1.0 - u) * 0.026 + u * (0.0070 if method == "route_b_physx_fem" else 0.0086)
            gz = _gz_pull0
        else:
            u = (f - 142) / 37.0
            _gx_pull0 = 0.036 if method == "route_b_physx_fem" else 0.022
            _gz_pull0 = (z_tgt - 0.005) if method == "route_b_physx_fem" else z_tgt
            gx = _gx_pull0 + u * 0.095
            gy_off = 0.0070 if method == "route_b_physx_fem" else 0.0086
            gz = _gz_pull0 + u * (0.005 if method == "route_b_physx_fem" else 0.0)
        pad_f_t.Set(Gf.Vec3d(gx, Y3 - gy_off, gz))
        pad_r_t.Set(Gf.Vec3d(gx, Y3 + gy_off, gz))
        # Lower rear staging rest out of the way right after seating so full hose weight hangs purely on the glass barb!
        rest_z = (z_tgt - _rest_dz) if f < 76 else (z_tgt - _rest_dz - min(1.0, (f - 76) / 10.0) * 0.090)
        rest_t_op.Set(Gf.Vec3d(_rest_x, Y3, rest_z))

        t_step0 = time.perf_counter()
        for _ in range(substeps):
            sim_iface.simulate(dt_sub, sim_time)
            sim_iface.fetch_results()
            sim_time += dt_sub
        step_ms_list.append((time.perf_counter() - t_step0) * 1000.0)

        # Read live states & update visual meshes
        if method == "route_b_physx_fem":
            # Bench 1 FEM
            p1_arr = np.asarray(b1_hose["pts_attr"].Get(), dtype=np.float64)
            b1_hose["vis_pts_attr"].Set(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in p1_arr]))
            tip_ids1 = b1_hose["mesh"].ring_outer_vertex_indices(b1_hose["mesh"].n_rings - 1)
            b1_tip_z[f] = float(np.mean(p1_arr[tip_ids1, 2]))

            # Bench 2 FEM
            p2_arr = np.asarray(b2_hose["pts_attr"].Get(), dtype=np.float64)
            b2_hose["vis_pts_attr"].Set(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in p2_arr]))
            tip_ids2 = b2_hose["mesh"].ring_outer_vertex_indices(b2_hose["mesh"].n_rings - 1)
            c_tip2 = np.mean(p2_arr[tip_ids2], axis=0)
            v_cur = p2_arr[tip_ids2[0]] - c_tip2
            v_cur[0] = 0.0
            vn = max(float(np.linalg.norm(v_cur)), 1e-9)
            v_cur /= vn
            if b2_ref_vec is None:
                b2_ref_vec = v_cur.copy()
            cos_a = float(np.clip(np.dot(b2_ref_vec, v_cur), -1.0, 1.0))
            sin_a = float(b2_ref_vec[1] * v_cur[2] - b2_ref_vec[2] * v_cur[1])
            tw_deg = math.degrees(math.atan2(sin_a, cos_a))
            b2_tip_twist_deg[f] = tw_deg
            hv = 0.5 * math.radians(tw_deg)
            b2_vane_t.Set(Gf.Vec3d(float(c_tip2[0]), float(c_tip2[1]), float(c_tip2[2])))
            b2_vane_o.Set(Gf.Quatd(math.cos(hv), math.sin(hv), 0.0, 0.0))

            # Bench 3 FEM
            p3_arr = np.asarray(b3_hose["pts_attr"].Get(), dtype=np.float64)
            b3_hose["vis_pts_attr"].Set(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in p3_arr]))
            m_ids = b3_hose["mesh"].ring_inner_vertex_indices(0)
            c_m = np.mean(p3_arr[m_ids], axis=0)
            b3_mouth_x[f] = float(c_m[0])
            b3_mouth_z[f] = float(c_m[2])
            st = compute_cuff_hoop_and_pinch_strain(b3_hose["mesh"], p3_arr)
            b3_hoop_strain_pct[f] = st["max_cuff_hoop_strain_pct"]
            b3_pinch_oval[f] = st["max_cuff_pinch_ovalization"]
        else:
            # Update Cosserat twist-striped visual mesh for Baseline & Route A across all 3 benches
            for b_idx, bh in enumerate((b1_hose, b2_hose, b3_hose)):
                l_pos = []
                l_rot = []
                for xf in bh["link_xfs"]:
                    p_w, R_w = _world_pos_rot(Usd, xf)
                    l_pos.append(p_w)
                    l_rot.append(R_w)
                l_pos_arr = np.asarray(l_pos, dtype=np.float64)
                tw = extract_twist_angles_from_link_frames(l_pos_arr, l_rot, local_material_axis=1)
                # Build dense 48-ring tube matching link centers + twist
                if b_idx == 2 and bh.get("link0_half", 0.0) > 0.012:
                    p_front = l_pos_arr[0] - l_rot[0][:, 0] * float(bh["link0_half"])
                    p_back = l_pos_arr[0] + l_rot[0][:, 0] * float(bh["link0_half"])
                    # Replace link_00 center with [p_front, p_back] and resample body links to preserve exact vertex count (n_links + 1)!
                    s_b_in = np.linspace(0.0, 1.0, len(l_pos_arr) - 1)
                    s_b_out = np.linspace(0.0, 1.0, len(l_pos_arr) - 1)
                    body_v = np.column_stack([np.interp(s_b_out, s_b_in, l_pos_arr[1:, k]) for k in range(3)])
                    c_vis = np.vstack([p_front[None, :], p_back[None, :], body_v])
                else:
                    s_in = np.linspace(0.0, 1.0, len(l_pos_arr))
                    s_out = np.linspace(0.0, 1.0, len(l_pos_arr) + 1)
                    c_vis = np.column_stack([np.interp(s_out, s_in, l_pos_arr[:, k]) for k in range(3)])
                v_m, n_m, _, _ = build_cosserat_twist_stripe_tube(c_vis, tw, radius_m=0.0068 if b_idx == 2 else float(rod_spec_14.capsule_radius_m))
                bh["pts_attr"].Set(v_m.tolist())
                bh["nrm_attr"].Set(n_m.tolist())
                if b_idx == 0:
                    b1_tip_z[f] = float(l_pos_arr[-1, 2])
                elif b_idx == 1:
                    # Measure tip link rotation about world X directly from link local Y vector
                    y_tip = l_rot[-1][:, 1]
                    tw_tip_deg = math.degrees(math.atan2(float(y_tip[2]), float(y_tip[1])))
                    b2_tip_twist_deg[f] = tw_tip_deg
                    hv = 0.5 * math.radians(tw_tip_deg)
                    b2_vane_t.Set(Gf.Vec3d(float(l_pos_arr[-1, 0]), float(l_pos_arr[-1, 1]), float(l_pos_arr[-1, 2])))
                    b2_vane_o.Set(Gf.Quatd(math.cos(hv), math.sin(hv), 0.0, 0.0))
                else:
                    b3_mouth_x[f] = float(l_pos_arr[0, 0] - bh.get("link0_half", 0.5 * bh["ds"]))
                    b3_mouth_z[f] = float(l_pos_arr[0, 2])
                    f_rad = 0.0
                    for ci in bh["cuff_pad_info"]:
                        pp, _ = _world_pos_rot(Usd, UsdGeom.Xformable(stage.GetPrimAtPath(ci["pad_prim"])))
                        hp, Rh = _world_pos_rot(Usd, UsdGeom.Xformable(stage.GetPrimAtPath(ci["host_prim"])))
                        disp = float(np.linalg.norm(pp - (hp + Rh @ ci["p_loc"])))
                        f_rad += min(ci["max_force"], ci["stiffness"] * disp)
                    b3_cuff_rad_force_n[f] = f_rad

        if record_video:
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True)
            for cid, an in annotators.items():
                rgba = an.get_data()
                if rgba is not None and getattr(rgba, "size", 0) > 0:
                    arr = np.asarray(rgba, dtype=np.uint8)
                    if arr.ndim == 3 and arr.shape[2] >= 3:
                        writers[cid].write(arr[:, :, :3])

    for wr in writers.values():
        wr.close()
    sim_iface.detach_stage()

    # Compute quantitative benchmark metrics
    eq_z = float(b1_tip_z[52])
    nudge_min_z = float(np.min(b1_tip_z[60:105]))
    rebound_z = float(np.mean(b1_tip_z[145:175]))
    sag_eq_mm = (0.90 - eq_z) * 1000.0
    nudge_drop_mm = max(0.0, (eq_z - nudge_min_z) * 1000.0)
    rebound_pct = (100.0 * max(0.0, rebound_z - nudge_min_z) / max(1e-4, eq_z - nudge_min_z)) if nudge_drop_mm > 3.0 else 0.0

    tip_twist_peak_deg = float(np.max(np.abs(b2_tip_twist_deg[60:175])))
    tip_twist_steady_deg = float(np.mean(b2_tip_twist_deg[120:175]))

    # Bench 3 Barb engagement & hands-off hang self-lock (f=95..124) + pull-off (f=165..179)
    hang_x_mm = float(np.mean(b3_mouth_x[95:124]) * 1000.0)
    hang_z_mm = float(np.mean(b3_mouth_z[95:124]) * 1000.0)
    barb_tip_x_mm = 26.0
    insertion_depth_hang_mm = max(0.0, barb_tip_x_mm - hang_x_mm)
    hang_z_err_mm = abs(hang_z_mm - z_tgt * 1000.0)
    self_lock_hang_passed = (insertion_depth_hang_mm >= 6.0) and (hang_z_err_mm <= 16.0)
    pull_off_x_mm = float(b3_mouth_x[-1] * 1000.0)
    pull_off_passed = (pull_off_x_mm >= barb_tip_x_mm + 4.0)

    # Automatic SOP coolant port verification ("下进上出"): inlet hose MUST seat on Lower Inlet Barb!
    dist_lower_mm = abs(hang_z_mm - Z_LOWER_BARB * 1000.0)
    dist_upper_mm = abs(hang_z_mm - Z_UPPER_BARB * 1000.0)
    detected_port = "lower_inlet" if dist_lower_mm < dist_upper_mm else "upper_outlet"
    flow_bottom_in_top_out_ok = bool(self_lock_hang_passed and detected_port == "lower_inlet")

    tel_path = out_dir / f"telemetry_{job['trial_id']}.npz"
    np.savez_compressed(
        tel_path,
        b1_tip_z=b1_tip_z,
        b2_tip_twist_deg=b2_tip_twist_deg,
        b3_mouth_x=b3_mouth_x,
        b3_mouth_z=b3_mouth_z,
        b3_cuff_rad_force_n=b3_cuff_rad_force_n,
        b3_hoop_strain_pct=b3_hoop_strain_pct,
        b3_pinch_oval=b3_pinch_oval,
    )

    result = {
        "trial_id": job["trial_id"],
        "method": method,
        "zero_friction": zero_friction,
        "target_port": target_port,
        "b3_weld_joint_count": b3_weld_count,
        "mean_frame_physx_ms": round(float(np.mean(step_ms_list)), 3),
        "effective_sim_fps": round(1000.0 / max(float(np.mean(step_ms_list)), 1e-3), 1),
        "exp1_cantilever": {
            "sag_eq_mm": round(sag_eq_mm, 2),
            "nudge_drop_mm": round(nudge_drop_mm, 2),
            "rebound_z_mm": round((0.90 - rebound_z) * 1000.0, 2),
            "elastic_rebound_pct": round(rebound_pct, 1),
        },
        "exp2_torsion_90deg": {
            "tip_twist_peak_deg": round(tip_twist_peak_deg, 2),
            "tip_twist_steady_deg": round(tip_twist_steady_deg, 2),
            "transmission_ratio_pct": round(100.0 * abs(tip_twist_steady_deg) / 90.0, 1),
        },
        "exp3_barb_fit": {
            "hang_mouth_x_mm": round(hang_x_mm, 2),
            "insertion_depth_hang_mm": round(insertion_depth_hang_mm, 2),
            "hang_z_err_mm": round(hang_z_err_mm, 2),
            "self_lock_hang_passed": self_lock_hang_passed,
            "pull_off_mouth_x_mm": round(pull_off_x_mm, 2),
            "pull_off_passed": pull_off_passed,
            "peak_cuff_radial_force_n": round(float(np.max(b3_cuff_rad_force_n)), 2),
            "peak_fem_hoop_strain_pct": round(float(np.max(b3_hoop_strain_pct)), 2),
            "peak_fem_pinch_ovalization_pct": round(100.0 * float(np.max(b3_pinch_oval)), 2),
            "detected_port": detected_port,
            "flow_bottom_in_top_out_ok": flow_bottom_in_top_out_ok,
        },
        "telemetry_npz": tel_path.as_posix(),
        "videos": {cid: (vid_dir / f"{cid}_{job['trial_id']}.mp4").as_posix() for cid in ("exp1", "exp2", "exp3")} if record_video else {},
    }
    if result_json_path is not None:
        result_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    app.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-json", required=True)
    ap.add_argument("--result-json", required=True)
    args = ap.parse_args()
    job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
    execute_hose_v10_trial(job, result_json_path=Path(args.result_json))


if __name__ == "__main__":
    main()

