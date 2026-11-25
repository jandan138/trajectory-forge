"""100%-Full-Dynamic PhysX 5 GPU (240 Hz TGS) One-Shot S01-S12 Worker for distillation_v09.

Merges:
- `v0.7` Articulated Rotary Mechanisms (`scissor_jack_blue`, `scissor_jack_white`, `flask_clamp_2prong`,
  `condenser_clamp_3prong`) with reversible signed leadscrew coupling (`tighten` in S01/S03/S06/S07 AND `loosen` in S02/S05/S11/S12!)
  + `48` live spherical-jointed dynamic cooling hose links skinned every frame via `skin_cosserat_tube`;
- `v0.8` Standard-Taper Ground-Glass Method B Physics (`ONLY Method B internal radial micro-spring wedges` across ALL 4 joints:
  `Flask 24/29`, `Head 15° Sidearm 24/29`, `Condenser Lower 15° 24/29`, `Head Top 19/26` shared by Funnel & Thermometer)
  + Two 3-link articulated green POM Keck clips (`snap-on` in S06/S07 & `snap-off` in S11!)
  + `0.00 mm` Mercury Thermometer Bulb flush alignment at `z = 1.0800 m`
  + Genuine Coaxial Twist-and-Pull Disassembly (`S11` right-to-left & `S12` top-to-bottom).

NON-NEGOTIABLE ANTI-CHEAT INVARIANTS:
1. `kinematicEnabled = False` on EVERY manipulated glass/lab instrument (`heating_stirrer_bath`, `round_bottom_flask_250ml`,
   `distillation_head_75deg`, `liebig_condenser_24_29`, `vacuum_tail_adapter`, `erlenmeyer_flask_150ml`,
   `long_stem_funnel`, `pour_beaker_100ml`, `thermometer_adapter_unit_v08`), EVERY movable mechanism link,
   EVERY articulated POM Keck clip link, EVERY Method B micro-spring sphere, Both wrist axial load-cells,
   and ALL `48` dynamic hose links!
2. Zero weld joints, zero teleports, zero stage reloads — ONE continuous uncut 240 Hz PhysX 5 GPU session from S01 to S12!
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
import tempfile
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("ISAACSIM_PORTABLE_ROOT", r"I:\isaacsim-data\6.0.1")
os.environ.setdefault("TEMP", r"I:\isaacsim-data\6.0.1\temp")
os.environ.setdefault("TMP", r"I:\isaacsim-data\6.0.1\temp")

_SRC_ROOT = str(Path(__file__).resolve().parents[3])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from trajectory_forge.adapters.distillation_v06.cosserat_hose_solver_v06 import skin_cosserat_tube  # noqa: E402
from trajectory_forge.adapters.distillation_v09.v09_s01_s12_schedule_planner import (  # noqa: E402
    COND_LOWER_CONE_W,
    COND_SEATED_CENTER_W,
    HEAD_SIDEARM_CONE_W,
    N15,
    RY15,
    TH15,
    U15,
    audit_v09_schedule_cpu,
    replan_closed_loop_window,
    synthesize_v09_s01_s12_schedule,
)
from trajectory_forge.sdk.panda_kinematics import forward_kinematics_panda  # noqa: E402

FFMPEG_FALLBACK = Path(r"G:\my_program\ffmpeg-2023-06-04-git-b1c3d81e71-full_build\bin\ffmpeg.exe")
Q15_WXYZ = (math.cos(0.5 * TH15), 0.0, math.sin(0.5 * TH15), 0.0)


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = map(float, a)
    bw, bx, by, bz = map(float, b)
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _apply_below_normal_priority() -> None:
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        except Exception:
            pass


def _find_ffmpeg() -> str:
    if FFMPEG_FALLBACK.is_file():
        return str(FFMPEG_FALLBACK)
    w = shutil.which("ffmpeg")
    if w:
        return w
    import imageio_ffmpeg  # type: ignore

    return str(imageio_ffmpeg.get_ffmpeg_exe())


class _StreamingH264Writer:
    """Zero-RAM-accumulation streaming H.264 MP4 encoder via FFmpeg rawvideo stdin pipe."""

    def __init__(self, out_mp4: Path, fps: int = 30) -> None:
        self.out_mp4 = out_mp4
        self.fps = fps
        self.proc: subprocess.Popen[bytes] | None = None
        self.frame_count = 0

    def write(self, rgb: np.ndarray) -> None:
        arr = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
        if self.proc is None:
            h, w = arr.shape[:2]
            self.out_mp4.parent.mkdir(parents=True, exist_ok=True)
            self.proc = subprocess.Popen(
                [
                    _find_ffmpeg(),
                    "-y",
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
                    "19",
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
        self.frame_count += 1

    def close(self) -> str | None:
        if self.proc is None:
            return None
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        finally:
            rc = self.proc.wait(timeout=180)
            self.proc = None
        if rc != 0:
            raise RuntimeError(f"FFmpeg streaming writer failed rc={rc} for {self.out_mp4}")
        return self.out_mp4.as_posix()


def _lookat_matrix4d(Gf: Any, eye: Sequence[float], target: Sequence[float]) -> Any:
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fwd = (t - e) / max(float(np.linalg.norm(t - e)), 1e-12)
    zc = -fwd
    xc = np.cross(up, zc)
    xc = xc / max(float(np.linalg.norm(xc)), 1e-12)
    yc = np.cross(zc, xc)
    return Gf.Matrix4d(
        float(xc[0]), float(xc[1]), float(xc[2]), 0.0,
        float(yc[0]), float(yc[1]), float(yc[2]), 0.0,
        float(zc[0]), float(zc[1]), float(zc[2]), 0.0,
        float(e[0]), float(e[1]), float(e[2]), 1.0,
    )


def _norm_rgb(raw: Any) -> np.ndarray:
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    arr = np.asarray(raw)[:, :, :3]
    if arr.dtype == np.uint8:
        return arr.copy()
    f = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if f.size and float(np.percentile(f, 99.0)) <= 2.0:
        f = f * 255.0
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


def _find_op(xf: Any, op_name: str, op_type: Any = None) -> Any:
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == op_name or (op_type is not None and op_type == op.GetOpType()):
            return op
    return None


def _bind_panda_visual_ops(stage: Any, UsdGeom: Any, Gf: Any, root_prim_path: str) -> dict[str, Any]:
    link_paths = [
        f"{root_prim_path}/panda_link0" + "".join(f"/panda_link{m}" for m in range(1, j + 1))
        for j in range(1, 8)
    ]
    orient_ops, base_orients = [], []
    for lp in link_paths:
        prim = stage.GetPrimAtPath(lp)
        op = _find_op(UsdGeom.Xformable(prim), "xformOp:orient", UsdGeom.XformOp.TypeOrient)
        orient_ops.append(op)
        base_orients.append(Gf.Quatd(op.Get()))
    hand_p = link_paths[-1] + "/panda_hand"
    lf_op = _find_op(UsdGeom.Xformable(stage.GetPrimAtPath(hand_p + "/panda_leftfinger")), "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
    rf_op = _find_op(UsdGeom.Xformable(stage.GetPrimAtPath(hand_p + "/panda_rightfinger")), "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
    return {"orient_ops": orient_ops, "base_orients": base_orients, "lf_op": lf_op, "rf_op": rf_op}


def _apply_panda_visual_q7_and_live_fingers(arm_ops: dict[str, Any], q7: np.ndarray, lf_m: float, rf_m: float, Gf: Any) -> None:
    for j in range(7):
        half = 0.5 * float(q7[j])
        qz = Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half)))
        arm_ops["orient_ops"][j].Set(arm_ops["base_orients"][j] * qz)
    arm_ops["lf_op"].Set(Gf.Vec3d(0.0, float(lf_m), 0.0584))
    arm_ops["rf_op"].Set(Gf.Vec3d(0.0, -float(rf_m), 0.0584))


def _upsample_catmull_centerline(ctrl_pts: np.ndarray, n_out: int = 64) -> np.ndarray:
    n_in = len(ctrl_pts)
    s_in = np.linspace(0.0, 1.0, n_in)
    s_out = np.linspace(0.0, 1.0, n_out)
    out = np.column_stack([np.interp(s_out, s_in, ctrl_pts[:, d]) for d in range(3)])
    for _ in range(3):
        sm = out.copy()
        out[2:-2] = 0.50 * sm[2:-2] + 0.25 * (sm[1:-3] + sm[3:-1])
    return out


def _decorate_master_hud(
    rgb: np.ndarray,
    *,
    Image: Any,
    ImageDraw: Any,
    f_title: Any,
    f_sub: Any,
    f_mono: Any,
    stage_meta: dict[str, Any],
    phase_txt: str,
    telemetry_txt: str,
    frame_idx: int,
    total_frames: int,
    zero_friction: bool,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    rgba = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)

    bv_s = int(stage_meta["bilibili_start_s"])
    bv_e = int(stage_meta["bilibili_end_s"])
    bv_tag = f"BV17t4y1V7XH {bv_s//60:02d}:{bv_s%60:02d}-{bv_e//60:02d}:{bv_e%60:02d}"
    bcol = (239, 68, 68) if zero_friction else (34, 197, 94)

    dr.rounded_rectangle([(14, 12), (w - 14, 92)], radius=10, fill=(10, 18, 32, 206), outline=(*bcol, 210), width=2)
    dr.rounded_rectangle([(26, 20), (254, 45)], radius=6, fill=(220, 38, 38, 240) if zero_friction else (2, 132, 199, 240))
    dr.text((34, 23), "COUNTERFACTUAL mu=0" if zero_friction else bv_tag, fill=(255, 255, 255, 255), font=f_mono)
    dr.text((266, 19), str(stage_meta["title_zh"]), fill=(248, 250, 252, 255), font=f_title)
    dr.text((28, 50), f"{phase_txt}  |  {telemetry_txt}", fill=(191, 219, 254, 245), font=f_sub)
    dr.text((28, 70), f"Method B All-4-Taper Micro-Spring Wedge + 240Hz PhysX 5 GPU | F {frame_idx+1:04d}/{total_frames}", fill=(148, 163, 184, 245), font=f_mono)
    return np.asarray(Image.alpha_composite(rgba, ov).convert("RGB"), dtype=np.uint8)


def _decorate_sub_hud(
    rgb: np.ndarray,
    *,
    Image: Any,
    ImageDraw: Any,
    f_title: Any,
    f_mono: Any,
    header: str,
    sub: str,
    badge_color: tuple[int, int, int],
) -> np.ndarray:
    h, w = rgb.shape[:2]
    rgba = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    dr.rounded_rectangle([(8, 8), (w - 8, 62)], radius=8, fill=(10, 18, 32, 202), outline=(*badge_color, 210), width=2)
    dr.text((16, 13), header, fill=(248, 250, 252, 255), font=f_title)
    dr.text((16, 37), sub, fill=(191, 219, 254, 245), font=f_mono)
    return np.asarray(Image.alpha_composite(rgba, ov).convert("RGB"), dtype=np.uint8)


def execute_v09_oneshot_job(job: dict[str, Any]) -> dict[str, Any]:
    _apply_below_normal_priority()
    pkg_root = Path(job["package_dir"]).resolve()
    out_dir = Path(job["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    zero_friction = bool(job.get("zero_friction", False))
    record_video = bool(job.get("record_video", True))
    max_frames_override = int(job.get("max_frames", 0))

    spec = json.loads((pkg_root / "semantics" / "v09_scene_spec.json").read_text(encoding="utf-8"))
    bL = np.asarray(spec["robot_bases"]["franka_left"]["xyz"], dtype=np.float64)
    qL = np.asarray(spec["robot_bases"]["franka_left"]["wxyz"], dtype=np.float64)
    bR = np.asarray(spec["robot_bases"]["franka_right"]["xyz"], dtype=np.float64)
    qR = np.asarray(spec["robot_bases"]["franka_right"]["wxyz"], dtype=np.float64)

    sched = synthesize_v09_s01_s12_schedule(bL, qL, bR, qR, fps=30)
    cpu_audit = audit_v09_schedule_cpu(sched, bL, qL, bR, qR)
    if not cpu_audit["passed"]:
        raise RuntimeError(f"CPU IK/Clearance Gate FAILED before GPU launch: {json.dumps(cpu_audit, indent=2)}")

    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    from isaacsim import SimulationApp

    spp = 6 if (record_video and not zero_friction) else (3 if record_video else 1)
    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RayTracedLighting" if record_video else "MinimalRendering",
            "samples_per_pixel_per_frame": spp,
            "denoiser": bool(record_video),
            "anti_aliasing": 3 if record_video else 0,
            "fast_shutdown": True,
            "extra_args": [
                "--/app/asyncRendering=False",
                "--/physics/updateToUsd=True",
                "--/physics/suppressReadback=False",
                "--/rtx/hydra/materialSyncLoads=True",
                "--/rtx/translucency/enabled=True",
                "--/rtx/reflections/enabled=True",
                "--/rtx/translucency/maxRefractionBounces=16",
                "--/rtx/raytracing/maxBounces=16",
                "--/rtx-transient/dlssg/enabled=False",
                "--/persistent/omni/replicator/captureOnPlay=False",
                "--/omni/replicator/captureOnPlay=False",
            ],
        }
    )

    import omni.replicator.core as rep
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from omni.physx.scripts import physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

    try:
        f_title = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 15)
        f_sub = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 12)
        f_mono = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 11)
    except Exception:
        f_title = f_sub = f_mono = ImageFont.load_default()

    usd_path = pkg_root / "scene" / "main.usda"
    omni.usd.get_context().open_stage(str(usd_path))
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    rep.orchestrator.set_capture_on_play(False)

    pscene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    ps_api = PhysxSchema.PhysxSceneAPI.Apply(pscene_prim)
    ps_api.CreateEnableGPUDynamicsAttr().Set(True)
    ps_api.CreateBroadphaseTypeAttr().Set("GPU")
    ps_api.CreateMaxPositionIterationCountAttr().Set(32)
    ps_api.CreateMaxVelocityIterationCountAttr().Set(8)

    # Strip default mesh rigid/collision APIs so ONLY calibrated millimeter-exact colliders act
    for root_clean in ("/World/franka_left", "/World/franka_right", "/World/Instruments", "/World/Mechanisms"):
        rp = stage.GetPrimAtPath(root_clean)
        if rp.IsValid():
            for pr in Usd.PrimRange(rp):
                if pr.HasAPI(UsdPhysics.CollisionAPI):
                    pr.GetAttribute("physics:collisionEnabled").Set(False)
                if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                    pr.GetAttribute("physics:rigidBodyEnabled").Set(False)

    UsdGeom.Scope.Define(stage, "/World/PhysMaterials")
    UsdGeom.Scope.Define(stage, "/World/PhysWedgePads")

    def _make_phys_mat(name: str, mu_s: float, mu_d: float, combine: str = "max") -> str:
        mp = f"/World/PhysMaterials/{name}"
        m = UsdShade.Material.Define(stage, mp)
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr().Set(float(mu_s))
        api.CreateDynamicFrictionAttr().Set(float(mu_d))
        api.CreateRestitutionAttr().Set(0.0)
        pxm = PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim())
        pxm.CreateFrictionCombineModeAttr().Set(combine)
        pxm.CreateRestitutionCombineModeAttr().Set("min")
        return mp

    mu_grip_s = 0.0 if zero_friction else 3.8
    mu_grip_d = 0.0 if zero_friction else 3.5
    comb_grip = "min" if zero_friction else "max"
    pad_mat = _make_phys_mat("SiliconePadMat", mu_grip_s, mu_grip_d, comb_grip)
    clamp_mat = _make_phys_mat("ClampRubberMat", mu_grip_s, mu_grip_d, comb_grip)
    taper_glide_mat = _make_phys_mat("TaperGlideMat", 0.0 if zero_friction else 0.18, 0.0 if zero_friction else 0.15, "average")
    frosted_wedge_mat = _make_phys_mat("FrostedWedgeMat", 0.0 if zero_friction else 0.92, 0.0 if zero_friction else 0.80, comb_grip)
    glass_grip_mat = _make_phys_mat("GlassGripMat", 0.0 if zero_friction else 3.0, 0.0 if zero_friction else 2.7, comb_grip)
    keck_pom_mat = _make_phys_mat("KeckPOMMat", 0.0 if zero_friction else 0.28, 0.0 if zero_friction else 0.22, "average")
    bench_mat = _make_phys_mat("BenchMat", 0.65, 0.55, "average")
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath("/World/Environment/LabBenchTop"), bench_mat)

    def _add_invisible_col(
        parent_path: str,
        cname: str,
        kind: str,
        pos: Sequence[float],
        size_or_rad: Sequence[float],
        mat_path: str,
        *,
        axis: str = "Z",
        quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> Any:
        cp = f"{parent_path}/PhysCol_{cname}"
        if kind == "box":
            g = UsdGeom.Cube.Define(stage, cp)
            g.CreateSizeAttr(1.0)
            sc = Gf.Vec3f(float(size_or_rad[0]), float(size_or_rad[1]), float(size_or_rad[2]))
        elif kind == "cylinder":
            g = UsdGeom.Cylinder.Define(stage, cp)
            g.CreateAxisAttr(axis)
            g.CreateRadiusAttr(float(size_or_rad[0]))
            g.CreateHeightAttr(float(size_or_rad[1]))
            sc = Gf.Vec3f(1.0, 1.0, 1.0)
        elif kind == "capsule":
            g = UsdGeom.Capsule.Define(stage, cp)
            g.CreateAxisAttr(axis)
            g.CreateRadiusAttr(float(size_or_rad[0]))
            g.CreateHeightAttr(float(size_or_rad[1]))
            sc = Gf.Vec3f(1.0, 1.0, 1.0)
        else:
            g = UsdGeom.Sphere.Define(stage, cp)
            g.CreateRadiusAttr(float(size_or_rad[0]))
            sc = Gf.Vec3f(1.0, 1.0, 1.0)
        g.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdGeom.Imageable(g).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        xf = UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))
        )
        xf.AddScaleOp().Set(sc)
        UsdPhysics.CollisionAPI.Apply(g.GetPrim()).CreateCollisionEnabledAttr(True)
        pxc = PhysxSchema.PhysxCollisionAPI.Apply(g.GetPrim())
        pxc.CreateRestOffsetAttr().Set(0.0)
        pxc.CreateContactOffsetAttr().Set(0.0010)
        physicsUtils.add_physics_material_to_prim(stage, g.GetPrim(), mat_path)
        return g.GetPrim()

    def _config_dynamic_rb(
        prim_path: str,
        mass_kg: float,
        lin_damp: float = 1.2,
        ang_damp: float = 1.8,
        com_xyz: Sequence[float] | None = None,
    ) -> Any:
        pr = stage.GetPrimAtPath(prim_path)
        rb = UsdPhysics.RigidBodyAPI.Apply(pr)
        rb.CreateRigidBodyEnabledAttr().Set(True)
        rb.CreateKinematicEnabledAttr().Set(False)
        mapi = UsdPhysics.MassAPI.Apply(pr)
        mapi.CreateMassAttr().Set(float(mass_kg))
        if com_xyz is not None:
            mapi.CreateCenterOfMassAttr().Set(Gf.Vec3f(float(com_xyz[0]), float(com_xyz[1]), float(com_xyz[2])))
        pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(pr)
        pxrb.CreateSolverPositionIterationCountAttr().Set(32)
        pxrb.CreateSolverVelocityIterationCountAttr().Set(8)
        pxrb.CreateMaxDepenetrationVelocityAttr().Set(2.50)
        pxrb.CreateLinearDampingAttr().Set(float(lin_damp))
        pxrb.CreateAngularDampingAttr().Set(float(ang_damp))
        pxrb.CreateSleepThresholdAttr().Set(0.0)
        pxrb.CreateEnableCCDAttr().Set(True)
        return pr

    def _set_world_joint_frame0(joint: Any, parent_path: str, local_xyz: Sequence[float]) -> None:
        mw = UsdGeom.Xformable(stage.GetPrimAtPath(parent_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pw = mw.Transform(Gf.Vec3d(*map(float, local_xyz)))
        qv = mw.ExtractRotationQuat()
        im = qv.GetImaginary()
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(pw[0]), float(pw[1]), float(pw[2])))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(qv.GetReal()), float(im[0]), float(im[1]), float(im[2])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def _add_16stave_concave_socket_z(
        parent_path: str,
        tag: str,
        z_center: float,
        length: float,
        r_inner_mid: float,
        flare_deg: float = 3.2,
        add_floor: bool = True,
        add_rolled_rim: bool = False,
        skip_wedge_quadrants: bool = False,
        add_outer_grip_skin: bool = False,
        stave_mat: str | None = None,
    ) -> None:
        wall_th = 0.0024
        r_mid = r_inner_mid + 0.5 * wall_th
        r_outer_grip = r_inner_mid + wall_th + 0.0022
        flare_rad = math.radians(flare_deg)
        use_mat = stave_mat or taper_glide_mat
        for i in range(16):
            if skip_wedge_quadrants and (i in (2, 6, 10, 14)):
                continue
            ang = 2.0 * math.pi * i / 16.0
            cz, sz = math.cos(0.5 * ang), math.sin(0.5 * ang)
            cy, sy = math.cos(0.5 * flare_rad), math.sin(0.5 * flare_rad)
            qw, qx, qy, qz = cz * cy, -sz * sy, cz * sy, sz * cy
            _add_invisible_col(
                parent_path,
                f"{tag}_Stave_{i:02d}",
                "box",
                (r_mid * math.cos(ang), r_mid * math.sin(ang), z_center),
                (wall_th, 2.0 * math.pi * r_inner_mid / 16.0 * 0.90, length),
                use_mat,
                quat_wxyz=(qw, qx, qy, qz),
            )
            if add_outer_grip_skin:
                _add_invisible_col(
                    parent_path,
                    f"{tag}_OuterGrip_{i:02d}",
                    "box",
                    (r_outer_grip * math.cos(ang), r_outer_grip * math.sin(ang), z_center),
                    (0.0024, 2.0 * math.pi * r_outer_grip / 16.0 * 0.95, length),
                    glass_grip_mat,
                    quat_wxyz=(cz, 0.0, 0.0, sz),
                )
        if add_rolled_rim:
            r_rim_mid = r_inner_mid + 0.0026
            z_rim = z_center + 0.5 * length - 0.0025
            for ri in range(8):
                ang = 2.0 * math.pi * ri / 8.0
                _add_invisible_col(
                    parent_path,
                    f"{tag}_RolledRim_{ri}",
                    "box",
                    (r_rim_mid * math.cos(ang), r_rim_mid * math.sin(ang), z_rim),
                    (0.0034, 0.0078, 0.0030),
                    keck_pom_mat,
                    quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)),
                )
        if add_floor:
            _add_invisible_col(
                parent_path,
                f"{tag}_SeatStop",
                "box",
                (0.0, 0.0, z_center - 0.5 * length - 0.0020),
                (2.2 * r_inner_mid, 2.2 * r_inner_mid, 0.0040),
                taper_glide_mat,
            )

    def _add_16stave_concave_socket_x(
        parent_path: str,
        tag: str,
        x_center: float,
        length: float,
        r_inner_mid: float,
        mouth_sign: float,
        rot_y_deg: float = 0.0,
        flare_deg: float = 4.0,
        add_rolled_rim: bool = True,
    ) -> str:
        sub_xf_p = f"{parent_path}/SocketScope_{tag}"
        sxf = UsdGeom.Xform.Define(stage, sub_xf_p)
        sxf.AddTranslateOp().Set(Gf.Vec3d(float(x_center), 0.0, 0.0))
        ry_r = math.radians(rot_y_deg)
        sxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(0.5 * ry_r), 0.0, math.sin(0.5 * ry_r), 0.0))
        wall_th = 0.0024
        r_mid = r_inner_mid + 0.5 * wall_th
        # Outward flare toward `mouth_sign * X`: at stave `i=0` (`+Y`), mouth `x = mouth_sign*L/2` maps via `Rz(theta)` to `y = r_mid + x*sin(theta)`, requiring `theta = +mouth_sign * flare_rad`!
        rz_flare = +mouth_sign * math.radians(flare_deg)
        cz_f, sz_f = math.cos(0.5 * rz_flare), math.sin(0.5 * rz_flare)
        for i in range(16):
            ang = 2.0 * math.pi * i / 16.0
            cy_p, sy_p = math.cos(ang), math.sin(ang)
            qx_h, sx_h = math.cos(0.5 * ang), math.sin(0.5 * ang)
            q_stave = _quat_mul((qx_h, sx_h, 0.0, 0.0), (cz_f, 0.0, 0.0, sz_f))
            _add_invisible_col(
                sub_xf_p,
                f"Stave_{i:02d}",
                "box",
                (0.0, r_mid * cy_p, r_mid * sy_p),
                (length, wall_th, 2.0 * math.pi * r_inner_mid / 16.0 * 0.90),
                taper_glide_mat,
                quat_wxyz=q_stave,
            )
        if add_rolled_rim:
            r_rim_mid = r_inner_mid + 0.0022
            x_rim = mouth_sign * (0.5 * length - 0.0025)
            for ri in range(8):
                ang = 2.0 * math.pi * ri / 8.0
                _add_invisible_col(
                    sub_xf_p,
                    f"RolledRim_{ri}",
                    "box",
                    (x_rim, r_rim_mid * math.cos(ang), r_rim_mid * math.sin(ang)),
                    (0.0030, 0.0030, 0.0078),
                    keck_pom_mat,
                    quat_wxyz=(math.cos(0.5 * ang), math.sin(0.5 * ang), 0.0, 0.0),
                )
        _add_invisible_col(
            sub_xf_p,
            "AxialStop",
            "box",
            (-mouth_sign * (0.5 * length + 0.0020), 0.0, 0.0),
            (0.0040, 2.2 * r_inner_mid, 2.2 * r_inner_mid),
            taper_glide_mat,
        )
        return sub_xf_p

    wedge_pad_joints: list[dict[str, Any]] = []
    manipulated_prims: list[str] = []

    def _filter_collision_group(paths: Sequence[str]) -> None:
        for idx_a, pa in enumerate(paths):
            fp_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(pa))
            rel = fp_api.CreateFilteredPairsRel()
            for pb in paths[idx_a + 1 :]:
                rel.AddTarget(Sdf.Path(pb))

    def _add_method_b_socket_wedge_ring_z(
        host_body_path: str,
        tag: str,
        *,
        z_local: float,
        r_inner_preload: float,
        r_sphere: float = 0.0036,
        pad_mass: float = 0.0028,
        stiffness: float = 1350.0,
        damping: float = 28.0,
        max_force: float = 10.0,
    ) -> list[str]:
        """Method B internal radial micro-spring sphere ring inside a dynamic Z-axis female socket (`Nyquist-safe!`)."""
        r_eq = r_inner_preload + r_sphere
        host_prim = stage.GetPrimAtPath(host_body_path)
        mw_host = UsdGeom.Xformable(host_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        q_host = mw_host.ExtractRotationQuat()
        im_h = q_host.GetImaginary()
        q_host_tup = (float(q_host.GetReal()), float(im_h[0]), float(im_h[1]), float(im_h[2]))

        created: list[str] = []
        for pi in range(4):
            ang = math.radians(45.0 + 90.0 * pi)
            p_loc = np.array([r_eq * math.cos(ang), r_eq * math.sin(ang), z_local], dtype=np.float64)
            q_rad_loc = (math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang))
            pw = mw_host.Transform(Gf.Vec3d(*map(float, p_loc)))
            q_pad_w = _quat_mul(q_host_tup, q_rad_loc)

            pad_p = f"/World/PhysWedgePads/{tag}_Pad_{pi}"
            pxf = UsdGeom.Xform.Define(stage, pad_p)
            pxf.AddTranslateOp().Set(Gf.Vec3d(float(pw[0]), float(pw[1]), float(pw[2])))
            pxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*q_pad_w))
            _config_dynamic_rb(pad_p, pad_mass, lin_damp=0.6, ang_damp=1.5)
            _add_invisible_col(pad_p, "Sphere", "sphere", (0.0, 0.0, 0.0), (r_sphere,), frosted_wedge_mat)
            created.append(pad_p)

            pj = UsdPhysics.PrismaticJoint.Define(stage, f"/World/PhysWedgePads/{tag}_Joint_{pi}")
            pj.CreateAxisAttr("X")
            pj.CreateBody0Rel().SetTargets([Sdf.Path(host_body_path)])
            pj.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, p_loc)))
            pj.CreateLocalRot0Attr().Set(Gf.Quatf(*map(float, q_rad_loc)))
            pj.CreateBody1Rel().SetTargets([Sdf.Path(pad_p)])
            pj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            pj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            pj.CreateLowerLimitAttr().Set(-0.0002)
            pj.CreateUpperLimitAttr().Set(+0.0026)
            drv = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
            drv.CreateStiffnessAttr().Set(float(stiffness))
            drv.CreateDampingAttr().Set(float(damping))
            drv.CreateMaxForceAttr().Set(float(max_force))
            drv.CreateTargetPositionAttr().Set(0.0)
            wedge_pad_joints.append(
                {
                    "pad_prim": pad_p,
                    "host_prim": host_body_path,
                    "p_loc": p_loc.copy(),
                    "stiffness": float(stiffness),
                    "max_force": float(max_force),
                }
            )
        _filter_collision_group([host_body_path, *created])
        manipulated_prims.extend(created)
        return created

    def _add_method_b_cone_two_ring_x(
        host_body_path: str,
        tag: str,
        *,
        cone_center_local: Sequence[float],
        cone_rot_y_deg: float,
        r_out_front: float = 0.0131,
        r_out_rear: float = 0.0127,
        r_sphere: float = 0.0032,
        pad_mass: float = 0.0024,
        stiffness: float = 1050.0,
        damping: float = 24.0,
        max_force: float = 7.5,
    ) -> list[str]:
        """Method B two-ring 8-sphere micro-spring wedge mounted directly on a dynamic male 24/29 cone (`Nyquist-safe!`)."""
        host_prim = stage.GetPrimAtPath(host_body_path)
        mw_host = UsdGeom.Xformable(host_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        q_host = mw_host.ExtractRotationQuat()
        im_h = q_host.GetImaginary()
        q_host_tup = (float(q_host.GetReal()), float(im_h[0]), float(im_h[1]), float(im_h[2]))

        ry_r = math.radians(cone_rot_y_deg)
        q_ry = (math.cos(0.5 * ry_r), 0.0, math.sin(0.5 * ry_r), 0.0)
        R_ry = np.array(
            [[math.cos(ry_r), 0.0, math.sin(ry_r)], [0.0, 1.0, 0.0], [-math.sin(ry_r), 0.0, math.cos(ry_r)]],
            dtype=np.float64,
        )
        c_loc = np.asarray(cone_center_local, dtype=np.float64)

        created: list[str] = []
        for ring_idx, x_ax in enumerate((-0.0055, +0.0055)):
            r_out = r_out_front if ring_idx == 0 else r_out_rear
            r_eq = r_out - r_sphere
            for pi in range(4):
                ang = math.radians(45.0 + 90.0 * pi)
                p_sub = np.array([x_ax, r_eq * math.cos(ang), r_eq * math.sin(ang)], dtype=np.float64)
                p_loc = c_loc + R_ry @ p_sub
                q_rx = (math.cos(0.5 * ang), math.sin(0.5 * ang), 0.0, 0.0)
                q_rad_loc = _quat_mul(q_ry, q_rx)

                pw = mw_host.Transform(Gf.Vec3d(*map(float, p_loc)))
                q_pad_w = _quat_mul(q_host_tup, q_rad_loc)

                pad_p = f"/World/PhysWedgePads/{tag}_Pad_{ring_idx}_{pi}"
                pxf = UsdGeom.Xform.Define(stage, pad_p)
                pxf.AddTranslateOp().Set(Gf.Vec3d(float(pw[0]), float(pw[1]), float(pw[2])))
                pxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*q_pad_w))
                _config_dynamic_rb(pad_p, pad_mass, lin_damp=0.6, ang_damp=1.5)
                _add_invisible_col(pad_p, "Sphere", "sphere", (0.0, 0.0, 0.0), (r_sphere,), frosted_wedge_mat)
                created.append(pad_p)

                pj = UsdPhysics.PrismaticJoint.Define(stage, f"/World/PhysWedgePads/{tag}_Joint_{ring_idx}_{pi}")
                pj.CreateAxisAttr("Y")
                pj.CreateBody0Rel().SetTargets([Sdf.Path(host_body_path)])
                pj.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, p_loc)))
                pj.CreateLocalRot0Attr().Set(Gf.Quatf(*map(float, q_rad_loc)))
                pj.CreateBody1Rel().SetTargets([Sdf.Path(pad_p)])
                pj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                pj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                pj.CreateLowerLimitAttr().Set(-0.0026)
                pj.CreateUpperLimitAttr().Set(+0.0002)
                drv = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
                drv.CreateStiffnessAttr().Set(float(stiffness))
                drv.CreateDampingAttr().Set(float(damping))
                drv.CreateMaxForceAttr().Set(float(max_force))
                drv.CreateTargetPositionAttr().Set(0.0)
                wedge_pad_joints.append(
                    {
                        "pad_prim": pad_p,
                        "host_prim": host_body_path,
                        "p_loc": p_loc.copy(),
                        "stiffness": float(stiffness),
                        "max_force": float(max_force),
                    }
                )
        _filter_collision_group([host_body_path, *created])
        manipulated_prims.extend(created)
        return created

    # =========================================================================
    # 1. Bench Cradles, Racks & Pedestals (`Zero Overlap` Across Entire Bench!)
    # =========================================================================
    def _build_bench_pedestal(name: str, xyz: Sequence[float], size_xyz: Sequence[float]) -> None:
        p = f"/World/Environment/{name}"
        c = UsdGeom.Cube.Define(stage, p)
        c.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size_xyz)))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/BenchTrimDark")))
        UsdPhysics.CollisionAPI.Apply(c.GetPrim()).CreateCollisionEnabledAttr(True)
        physicsUtils.add_physics_material_to_prim(stage, c.GetPrim(), bench_mat)

    # Flask Cork Ring at `(-0.455, -0.020, 0.750)`
    cork_xf = UsdGeom.Xform.Define(stage, "/World/Environment/FlaskCorkRing")
    cork_xf.AddTranslateOp().Set(Gf.Vec3d(-0.455, -0.020, 0.750))
    cork_mat = UsdShade.Material.Define(stage, "/World/Looks/CorkSupportMat")
    cork_sh = UsdShade.Shader.Define(stage, "/World/Looks/CorkSupportMat/Shader")
    cork_sh.CreateIdAttr("UsdPreviewSurface")
    cork_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.58, 0.41, 0.24))
    cork_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.78)
    cork_mat.CreateSurfaceOutput().ConnectToSource(cork_sh.ConnectableAPI(), "surface")
    for si in range(12):
        ang = 2.0 * math.pi * si / 12.0
        _add_invisible_col("/World/Environment/FlaskCorkRing", f"Low_{si}", "box", (0.026 * math.cos(ang), 0.026 * math.sin(ang), 0.0038), (0.006, 0.015, 0.0076), bench_mat, quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)))
        _add_invisible_col("/World/Environment/FlaskCorkRing", f"Cup_{si}", "box", (0.0485 * math.cos(ang), 0.0485 * math.sin(ang), 0.020), (0.006, 0.022, 0.024), bench_mat, quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)))
        cv = UsdGeom.Cube.Define(stage, f"/World/Environment/FlaskCorkRing/Vis_{si}")
        cv.CreateSizeAttr(1.0)
        cv_xf = UsdGeom.Xformable(cv)
        cv_xf.AddTranslateOp().Set(Gf.Vec3d(0.038 * math.cos(ang), 0.038 * math.sin(ang), 0.011))
        cv_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)))
        cv_xf.AddScaleOp().Set(Gf.Vec3f(0.012, 0.021, 0.022))
        UsdShade.MaterialBindingAPI.Apply(cv.GetPrim()).Bind(cork_mat)

    # Head Bench Rack at `(-0.335, -0.115, 0.750)`
    h_rack = UsdGeom.Xform.Define(stage, "/World/Environment/HeadBenchRack")
    h_rack.AddTranslateOp().Set(Gf.Vec3d(-0.335, -0.115, 0.750))
    _add_invisible_col("/World/Environment/HeadBenchRack", "Floor", "box", (0.0, 0.0, 0.0105), (0.052, 0.052, 0.021), bench_mat)
    vis_hb = UsdGeom.Cube.Define(stage, "/World/Environment/HeadBenchRack/VisBase")
    vis_hb.CreateSizeAttr(1.0)
    vhxf = UsdGeom.Xformable(vis_hb)
    vhxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0105))
    vhxf.AddScaleOp().Set(Gf.Vec3f(0.052, 0.052, 0.021))
    UsdShade.MaterialBindingAPI.Apply(vis_hb.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/BenchTrimDark")))
    for si in range(8):
        ang = 2.0 * math.pi * si / 8.0
        _add_invisible_col("/World/Environment/HeadBenchRack", f"Cup_{si}", "box", (0.0142 * math.cos(ang), 0.0142 * math.sin(ang), 0.034), (0.004, 0.010, 0.026), taper_glide_mat, quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)))
    # Vertical 0.0° anti-tilt & anti-yaw fork under Head's cantilevered +X sidearm so Head stands 0.0° pitch & 0.0° yaw on HeadBenchRack
    _add_invisible_col("/World/Environment/HeadBenchRack", "SidearmRest", "box", (+0.055, 0.0, 0.0492), (0.012, 0.024, 0.045), taper_glide_mat)
    _add_invisible_col("/World/Environment/HeadBenchRack", "SidearmCheekF", "box", (+0.055, -0.0122, 0.068), (0.012, 0.004, 0.022), taper_glide_mat)
    _add_invisible_col("/World/Environment/HeadBenchRack", "SidearmCheekR", "box", (+0.055, +0.0122, 0.068), (0.012, 0.004, 0.022), taper_glide_mat)

    # Thermometer Hollow 8-Pillar Rack at `(-0.335, +0.025, 0.750)` (`top z = 0.8741m -> th_z = 0.8555m with RackCollar at +0.0236m`)
    th_rack = UsdGeom.Xform.Define(stage, "/World/Environment/ThermoBenchRack")
    th_rack.AddTranslateOp().Set(Gf.Vec3d(-0.335, +0.025, 0.750))
    for si in range(8):
        ang = 2.0 * math.pi * si / 8.0
        _add_invisible_col("/World/Environment/ThermoBenchRack", f"Pillar_{si}", "box", (0.0124 * math.cos(ang), 0.0124 * math.sin(ang), 0.06205), (0.0040, 0.0085, 0.1241), taper_glide_mat, quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)))
        vp = UsdGeom.Cube.Define(stage, f"/World/Environment/ThermoBenchRack/Vis_{si}")
        vp.CreateSizeAttr(1.0)
        vxf = UsdGeom.Xformable(vp)
        vxf.AddTranslateOp().Set(Gf.Vec3d(0.0135 * math.cos(ang), 0.0135 * math.sin(ang), 0.06205))
        vxf.AddScaleOp().Set(Gf.Vec3f(0.004, 0.007, 0.1241))
        UsdShade.MaterialBindingAPI.Apply(vp.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/BenchTrimDark")))

    # Funnel Hollow 8-Pillar Rack at `(-0.465, -0.135, 0.750)` (`top z = 0.877m`)
    fn_rack = UsdGeom.Xform.Define(stage, "/World/Environment/FunnelBenchRack")
    fn_rack.AddTranslateOp().Set(Gf.Vec3d(-0.465, -0.135, 0.750))
    for si in range(8):
        ang = 2.0 * math.pi * si / 8.0
        _add_invisible_col("/World/Environment/FunnelBenchRack", f"Pillar_{si}", "box", (0.0126 * math.cos(ang), 0.0126 * math.sin(ang), 0.0635), (0.0040, 0.0085, 0.127), bench_mat, quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)))
        vp = UsdGeom.Cube.Define(stage, f"/World/Environment/FunnelBenchRack/Vis_{si}")
        vp.CreateSizeAttr(1.0)
        vxf = UsdGeom.Xformable(vp)
        vxf.AddTranslateOp().Set(Gf.Vec3d(0.0138 * math.cos(ang), 0.0138 * math.sin(ang), 0.0635))
        vxf.AddScaleOp().Set(Gf.Vec3f(0.004, 0.007, 0.127))
        UsdShade.MaterialBindingAPI.Apply(vp.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/BenchTrimDark")))

    # Condenser V-Rest at `(0.020, -0.105, 0.750)` (`cradles at x=-0.068/+0.058, completely clear of ArmL [-0.057, -0.019] grasp zone!`)
    vrest = UsdGeom.Xform.Define(stage, "/World/Environment/CondenserBenchRest")
    vrest.AddTranslateOp().Set(Gf.Vec3d(0.020, -0.105, 0.750))
    for rx, wx, rtag in ((-0.068, 0.008, "Neg"), (+0.058, 0.012, "Pos")):
        _add_invisible_col("/World/Environment/CondenserBenchRest", f"R_{rtag}", "box", (rx, 0.0, 0.0119), (wx, 0.036, 0.0238), bench_mat)
        _add_invisible_col("/World/Environment/CondenserBenchRest", f"RF_{rtag}", "box", (rx, -0.019, 0.0245), (wx, 0.005, 0.005), bench_mat)
        _add_invisible_col("/World/Environment/CondenserBenchRest", f"RR_{rtag}", "box", (rx, +0.019, 0.0245), (wx, 0.005, 0.005), bench_mat)

    # Tail Adapter Cradle at `(+0.310 + 0.056, +0.015, 0.750)` (`supports TailGripStem bottom -0.040m at z_world = 0.815m -> tail_z = 0.855m!`)
    t_cradle = UsdGeom.Xform.Define(stage, "/World/Environment/TailBenchCradle")
    t_cradle.AddTranslateOp().Set(Gf.Vec3d(+0.366, +0.015, 0.750))
    _add_invisible_col("/World/Environment/TailBenchCradle", "Floor", "box", (0.0, 0.0, 0.0325), (0.038, 0.038, 0.065), bench_mat)
    for si in range(8):
        ang_c = 2.0 * math.pi * si / 8.0
        _add_invisible_col("/World/Environment/TailBenchCradle", f"Lip_{si}", "box", (0.0125 * math.cos(ang_c), 0.0125 * math.sin(ang_c), 0.071), (0.004, 0.008, 0.012), bench_mat, quat_wxyz=(math.cos(0.5 * ang_c), 0.0, 0.0, math.sin(0.5 * ang_c)))
    vis_tb = UsdGeom.Cube.Define(stage, "/World/Environment/TailBenchCradle/VisStand")
    vis_tb.CreateSizeAttr(1.0)
    vtxf = UsdGeom.Xformable(vis_tb)
    vtxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0325))
    vtxf.AddScaleOp().Set(Gf.Vec3f(0.034, 0.034, 0.065))
    UsdShade.MaterialBindingAPI.Apply(vis_tb.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/BenchTrimDark")))

    # Outer-Right Beaker Pedestal (`pure 13-part apparatus faithful to Bilibili BV17t4y1V7XH — zero non-video Keck pedestals!`)
    _build_bench_pedestal("BeakerBenchPedestal", (+0.445, -0.105, 0.5 * (0.750 + 0.792)), (0.064, 0.064, 0.042))

    # =========================================================================
    # 2. Configure 4 Articulated Rotary Lab Mechanisms (`2 Jacks + 2 Clamps`)
    # =========================================================================
    # 2a. Blue & White Scissor Jacks
    for j_id, top_z0, whl_z0, max_lift in (
        ("scissor_jack_blue_art001", 0.062, 0.035, 0.032),
        ("scissor_jack_white_art001", 0.040, 0.026, 0.044),
    ):
        jroot = f"/World/Mechanisms/{j_id}"
        b_p = f"{jroot}/BasePlate"
        t_p = f"{jroot}/TopPlatform"
        w_p = f"{jroot}/Handwheel"
        _config_dynamic_rb(t_p, 0.45, lin_damp=2.5, ang_damp=2.5)
        _add_invisible_col(t_p, "TopDeck", "box", (0.0, 0.0, 0.002), (0.156, 0.156, 0.008), pad_mat)
        pj = UsdPhysics.PrismaticJoint.Define(stage, f"{jroot}/LiftPrismatic")
        pj.CreateAxisAttr("Z")
        pj.CreateBody1Rel().SetTargets([Sdf.Path(t_p)])
        _set_world_joint_frame0(pj, b_p, (0.0, 0.0, top_z0))
        pj.CreateLowerLimitAttr().Set(-0.002)
        pj.CreateUpperLimitAttr().Set(float(max_lift))
        pdrv = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
        pdrv.CreateStiffnessAttr().Set(42000.0)
        pdrv.CreateDampingAttr().Set(2400.0)
        pdrv.CreateMaxForceAttr().Set(260.0)
        pdrv.CreateTargetPositionAttr().Set(0.0)

        _config_dynamic_rb(w_p, 0.10, lin_damp=1.0, ang_damp=0.65)
        _add_invisible_col(w_p, "KnobDrum", "cylinder", (0.0, 0.006, 0.0), (0.0115, 0.020), clamp_mat, axis="Y")
        rj = UsdPhysics.RevoluteJoint.Define(stage, f"{jroot}/WheelRevolute")
        rj.CreateAxisAttr("Y")
        rj.CreateBody1Rel().SetTargets([Sdf.Path(w_p)])
        _set_world_joint_frame0(rj, b_p, (0.0, 0.088, whl_z0))
        UsdPhysics.DriveAPI.Apply(rj.GetPrim(), "angular").CreateDampingAttr().Set(0.65)
        manipulated_prims.extend([t_p, w_p])

    # 2b. 2-Prong Flask Clamp (`starts semi-closed at +10°, pre-opens to +32° in S02, tightens to -2.8° in S03, loosens in S12!`)
    yoke2_p = "/World/Mechanisms/flask_clamp_2prong_art001/FixedLeftYoke"
    jaw2_p = "/World/Mechanisms/flask_clamp_2prong_art001/MovableRightJaw"
    wing2_p = "/World/Mechanisms/flask_clamp_2prong_art001/ButterflyWingnut"
    # Exact proven S03 diagonal entry & anti-roll 2-prong jaw geometry (`flk = [-0.1803, -0.0018, 0.9387]`, < 1° vertical!)
    _add_invisible_col(yoke2_p, "LeftPadMid", "box", (-0.0190, 0.0, 0.0), (0.006, 0.014, 0.014), clamp_mat)
    _add_invisible_col(yoke2_p, "LeftPadFront", "box", (-0.0182, -0.0115, 0.0), (0.006, 0.010, 0.014), clamp_mat, quat_wxyz=(math.cos(0.38), 0.0, 0.0, -math.sin(0.38)))
    _add_invisible_col(yoke2_p, "LeftPadRear", "box", (-0.0182, +0.0115, 0.0), (0.006, 0.010, 0.014), clamp_mat, quat_wxyz=(math.cos(0.38), 0.0, 0.0, +math.sin(0.38)))
    _config_dynamic_rb(jaw2_p, 0.06, lin_damp=1.5, ang_damp=1.5)
    _add_invisible_col(jaw2_p, "RightPadMid", "box", (+0.0175, -0.020, 0.0), (0.006, 0.014, 0.014), clamp_mat)
    _add_invisible_col(jaw2_p, "RightPadFront", "box", (+0.0155, -0.0315, 0.0), (0.006, 0.010, 0.014), clamp_mat, quat_wxyz=(math.cos(0.38), 0.0, 0.0, +math.sin(0.38)))
    _add_invisible_col(jaw2_p, "RightPadRear", "box", (+0.0155, -0.0085, 0.0), (0.006, 0.010, 0.014), clamp_mat, quat_wxyz=(math.cos(0.38), 0.0, 0.0, -math.sin(0.38)))
    jrj2 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Mechanisms/flask_clamp_2prong_art001/RightJawRevolute")
    jrj2.CreateAxisAttr("Z")
    jrj2.CreateBody1Rel().SetTargets([Sdf.Path(jaw2_p)])
    _set_world_joint_frame0(jrj2, yoke2_p, (0.0, 0.024, 0.0))
    jrj2.CreateLowerLimitAttr().Set(-10.0)
    jrj2.CreateUpperLimitAttr().Set(38.0)
    jdrv2 = UsdPhysics.DriveAPI.Apply(jrj2.GetPrim(), "angular")
    jdrv2.CreateStiffnessAttr().Set(2200.0)
    jdrv2.CreateDampingAttr().Set(85.0)
    jdrv2.CreateMaxForceAttr().Set(175.0)
    jdrv2.CreateTargetPositionAttr().Set(12.0)

    _config_dynamic_rb(wing2_p, 0.06, lin_damp=1.0, ang_damp=0.65)
    _add_invisible_col(wing2_p, "WingDrum", "cylinder", (0.0, 0.014, 0.0), (0.0070, 0.030), clamp_mat, axis="Y")
    wrj2 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Mechanisms/flask_clamp_2prong_art001/WingnutRevolute")
    wrj2.CreateAxisAttr("Y")
    wrj2.CreateBody1Rel().SetTargets([Sdf.Path(wing2_p)])
    _set_world_joint_frame0(wrj2, yoke2_p, (0.024, 0.044, 0.0))
    UsdPhysics.DriveAPI.Apply(wrj2.GetPrim(), "angular").CreateDampingAttr().Set(0.65)
    manipulated_prims.extend([jaw2_p, wing2_p])

    # 2c. 3-Prong Condenser Clamp (`starts semi-closed at -14°, pre-opens to -42° in S05, tightens to +12° in S06, loosens in S11!`)
    cbase3_p = "/World/Mechanisms/condenser_clamp_3prong_art001/FixedLowerBase"
    cjaw3_p = "/World/Mechanisms/condenser_clamp_3prong_art001/MovableUpperJaw"
    cwing3_p = "/World/Mechanisms/condenser_clamp_3prong_art001/TopButterflyWingnut"
    _add_invisible_col(cbase3_p, "LowBottom", "box", (0.0, 0.0, -0.0160), (0.042, 0.016, 0.006), clamp_mat)
    _add_invisible_col(cbase3_p, "LowFront", "box", (0.0, -0.0215, -0.0128), (0.042, 0.005, 0.014), clamp_mat, quat_wxyz=(math.cos(0.42), -math.sin(0.42), 0.0, 0.0))
    _add_invisible_col(cbase3_p, "LowRear", "box", (0.0, +0.0215, -0.0128), (0.042, 0.005, 0.014), clamp_mat, quat_wxyz=(math.cos(0.42), +math.sin(0.42), 0.0, 0.0))
    _config_dynamic_rb(cjaw3_p, 0.06, lin_damp=1.5, ang_damp=1.5)
    _add_invisible_col(cjaw3_p, "UpperPadTop", "box", (0.0, -0.024, +0.0035), (0.040, 0.018, 0.008), clamp_mat)
    _add_invisible_col(
        cjaw3_p, "UpperFrontHook", "box",
        (0.0, -0.041, -0.0065), (0.040, 0.006, 0.016), clamp_mat,
        quat_wxyz=(math.cos(0.30), -math.sin(0.30), 0.0, 0.0),
    )
    uj3 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Mechanisms/condenser_clamp_3prong_art001/UpperJawRevolute")
    uj3.CreateAxisAttr("X")
    uj3.CreateBody1Rel().SetTargets([Sdf.Path(cjaw3_p)])
    _set_world_joint_frame0(uj3, cbase3_p, (0.0, 0.024, 0.014))
    uj3.CreateLowerLimitAttr().Set(-58.0)
    uj3.CreateUpperLimitAttr().Set(20.0)
    udrv3 = UsdPhysics.DriveAPI.Apply(uj3.GetPrim(), "angular")
    udrv3.CreateStiffnessAttr().Set(1800.0)
    udrv3.CreateDampingAttr().Set(80.0)
    udrv3.CreateMaxForceAttr().Set(175.0)
    udrv3.CreateTargetPositionAttr().Set(-14.0)

    _config_dynamic_rb(cwing3_p, 0.06, lin_damp=1.0, ang_damp=0.65)
    _add_invisible_col(cwing3_p, "TopWingDrum", "cylinder", (0.0, 0.008, 0.0), (0.0070, 0.022), clamp_mat, axis="Y")
    twj3 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Mechanisms/condenser_clamp_3prong_art001/TopWingRevolute")
    twj3.CreateAxisAttr("Y")
    twj3.CreateBody1Rel().SetTargets([Sdf.Path(cwing3_p)])
    _set_world_joint_frame0(twj3, cbase3_p, (0.0, 0.048, 0.018))
    UsdPhysics.DriveAPI.Apply(twj3.GetPrim(), "angular").CreateDampingAttr().Set(0.65)
    manipulated_prims.extend([cjaw3_p, cwing3_p])

    # =========================================================================
    # 3. Configure ALL 9 Dynamic Glass/Apparatus Bodies + ALL 4 Method B Wedges!
    # =========================================================================
    # 3a. Dynamic Heating Stirrer Bath (`hollow 12-wall cylinder resting on Blue Scissor Jack!`)
    bath_p = "/World/Instruments/heating_stirrer_bath"
    _config_dynamic_rb(bath_p, 1.85, lin_damp=1.8, ang_damp=3.0)
    # True circular cylindrical base (`r=0.095m`, matching real round water bath — eliminates phantom square-box corner at r=0.134m!)
    _add_invisible_col(bath_p, "BathFloor", "cylinder", (0.0, 0.0, 0.024), (0.095, 0.048), bench_mat, axis="Z")
    # Non-slip silicone-lined submerged trivet (`top z_local = 0.0625m -> z_world = 0.8970m`, firmly grounding Bulb at z_flk = 0.9390m for 0° pitch!)
    non_slip_trivet_mat = _make_phys_mat("BathTrivetMat", 0.0 if zero_friction else 0.95, 0.0 if zero_friction else 0.85, comb_grip)
    _add_invisible_col(bath_p, "BathSubmergedTrivet", "box", (0.0, 0.0, 0.0560), (0.130, 0.130, 0.013), non_slip_trivet_mat)
    for si in range(12):
        ang = 2.0 * math.pi * si / 12.0
        _add_invisible_col(
            bath_p, f"Wall_{si:02d}", "box",
            (0.102 * math.cos(ang), 0.102 * math.sin(ang), 0.064), (0.006, 0.054, 0.020), bench_mat,
            quat_wxyz=(math.cos(0.5 * ang), 0.0, 0.0, math.sin(0.5 * ang)),
        )
    manipulated_prims.append(bath_p)

    # 3b. Dynamic 250 mL Round-Bottom Flask + Two-Ring Coaxial Method-B Wedge #1 (`J01_FlaskWedge_Upper/Lower` — ZERO Head Rocking!)
    flask_p = "/World/Instruments/round_bottom_flask_250ml"
    _config_dynamic_rb(flask_p, 0.155, lin_damp=1.2, ang_damp=2.0, com_xyz=(0.0, 0.0, -0.018))
    _add_invisible_col(flask_p, "Bulb", "sphere", (0.0, 0.0, 0.0), (0.0420,), glass_grip_mat)
    _add_invisible_col(flask_p, "SmoothLowerNeck", "cylinder", (0.0, 0.0, +0.0555), (0.0140, 0.031), glass_grip_mat, axis="Z")
    _add_16stave_concave_socket_z(
        flask_p, "Flask2429", z_center=0.0865, length=0.029, r_inner_mid=0.0122,
        flare_deg=3.5, add_floor=True, add_rolled_rim=True, skip_wedge_quadrants=True, add_outer_grip_skin=True,
    )
    for gi in range(8):
        ang_g = 2.0 * math.pi * gi / 8.0
        _add_invisible_col(
            flask_p, f"FlaskGripOct_{gi}", "box",
            (0.0172 * math.cos(ang_g), 0.0172 * math.sin(ang_g), 0.0910), (0.0030, 0.0146, 0.024), glass_grip_mat,
            quat_wxyz=(math.cos(0.5 * ang_g), 0.0, 0.0, math.sin(0.5 * ang_g)),
        )
    _add_method_b_socket_wedge_ring_z(
        flask_p, "J01_FlaskWedge_Upper", z_local=0.0960, r_inner_preload=0.0101,
        r_sphere=0.0036, pad_mass=0.0026, stiffness=1750.0, damping=30.0, max_force=12.5,
    )
    _add_method_b_socket_wedge_ring_z(
        flask_p, "J01_FlaskWedge_Lower", z_local=0.0795, r_inner_preload=0.0100,
        r_sphere=0.0036, pad_mass=0.0026, stiffness=1750.0, damping=30.0, max_force=12.5,
    )
    manipulated_prims.append(flask_p)

    # 3c. Dynamic 75° Distillation Head + Method B Wedge #2 (`J02_SidearmWedge`) + Method B Wedge #4 (`J04_ThermoWedge`)
    head_p = "/World/Instruments/distillation_head_75deg"
    _config_dynamic_rb(head_p, 0.130, lin_damp=1.4, ang_damp=2.2, com_xyz=(0.0, 0.0, 0.028))
    _add_invisible_col(head_p, "PlugLeadTip", "cylinder", (0.0, 0.0, -0.0115), (0.0098, 0.005), taper_glide_mat, axis="Z")
    _add_invisible_col(head_p, "PlugShaft", "cylinder", (0.0, 0.0, +0.0018), (0.0114, 0.0216), frosted_wedge_mat, axis="Z")
    # 100% open-bore top 19/26 socket (`shared by S08 Funnel & S09 Thermometer`, AND grasped on axisymmetric outer glass collar at z=+0.100m!)
    _add_16stave_concave_socket_z(
        head_p, "HeadTop1926", z_center=0.103, length=0.030, r_inner_mid=0.0106,
        flare_deg=5.5, add_floor=False, add_rolled_rim=False, skip_wedge_quadrants=True, add_outer_grip_skin=True,
    )
    for gi in range(8):
        ang_g = 2.0 * math.pi * gi / 8.0
        _add_invisible_col(
            head_p, f"HeadGripOct_{gi}", "box",
            (0.0165 * math.cos(ang_g), 0.0165 * math.sin(ang_g), 0.1000), (0.0032, 0.0142, 0.024), glass_grip_mat,
            quat_wxyz=(math.cos(0.5 * ang_g), 0.0, 0.0, math.sin(0.5 * ang_g)),
        )
    for ri in range(16):
        ang_t = 2.0 * math.pi * ri / 16.0
        _add_invisible_col(
            head_p, f"Top1926SeatTab_{ri:02d}", "box",
            (0.0162 * math.cos(ang_t), 0.0162 * math.sin(ang_t), 0.1170), (0.0044, 0.0068, 0.0160), frosted_wedge_mat,
            quat_wxyz=(math.cos(0.5 * ang_t), 0.0, 0.0, math.sin(0.5 * ang_t)),
        )
    # Rest support neck on Head sidearm (`s = 0.0575m`, `27.5mm` behind J02 cone center so it NEVER blocks CondUpper2429, while resting cleanly on HeadBenchRack/SidearmRest!)
    _add_invisible_col(
        head_p, "SidearmSupportNeck", "cylinder",
        (0.0575 * math.cos(TH15), 0.0, 0.062 - 0.0575 * math.sin(TH15)),
        (0.0098, 0.010), taper_glide_mat, axis="X", quat_wxyz=Q15_WXYZ,
    )
    _add_invisible_col(
        head_p, "SidearmLeadTip", "cylinder",
        (0.0825 * math.cos(TH15), 0.0, 0.062 - 0.0825 * math.sin(TH15)),
        (0.0108, 0.020), taper_glide_mat, axis="X", quat_wxyz=Q15_WXYZ,
    )
    _add_method_b_cone_two_ring_x(
        head_p, "J02_SidearmWedge", cone_center_local=(+0.082104, 0.0, +0.040000),
        cone_rot_y_deg=15.0, r_out_front=0.0129, r_out_rear=0.0125,
        stiffness=1350.0, damping=28.0, max_force=9.5,
    )
    # Two-ring coaxial Method-B micro-spring sleeve inside HeadTop1926 (`14mm` span, 12x softer axial pull than J01 so S09a/S12a never lift Head!)
    _add_method_b_socket_wedge_ring_z(
        head_p, "J04_ThermoWedge_Upper", z_local=0.1115, r_inner_preload=0.0087,
        r_sphere=0.0032, pad_mass=0.0024, stiffness=340.0, damping=12.0, max_force=2.2,
    )
    _add_method_b_socket_wedge_ring_z(
        head_p, "J04_ThermoWedge_Lower", z_local=0.0975, r_inner_preload=0.0086,
        r_sphere=0.0032, pad_mass=0.0024, stiffness=340.0, damping=12.0, max_force=2.2,
    )
    manipulated_prims.append(head_p)

    # 3d. Dynamic Liebig Condenser + Method B Wedge #3 (`J03_CondWedge`) + 48 Live Dynamic Hose Links (`true outward +mouth_sign 4.2° flare + 10.9mm anti-droop pin!`)
    cond_p = "/World/Instruments/liebig_condenser_24_29"
    _config_dynamic_rb(cond_p, 0.185, lin_damp=1.2, ang_damp=1.8)
    _add_invisible_col(cond_p, "WaterJacket", "cylinder", (0.0, 0.0, 0.0), (0.0162, 0.146), glass_grip_mat, axis="X")
    _add_16stave_concave_socket_x(cond_p, "CondUpper2429", x_center=-0.125, length=0.028, r_inner_mid=0.0121, mouth_sign=-1.0, flare_deg=4.2, add_rolled_rim=False)
    _add_invisible_col(cond_p, "LowerConeLeadTip", "cylinder", (+0.1245, 0.0, 0.0), (0.0111, 0.024), taper_glide_mat, axis="X")
    _add_method_b_cone_two_ring_x(
        cond_p, "J03_CondWedge", cone_center_local=(+0.125000, 0.0, 0.000000),
        cone_rot_y_deg=0.0, r_out_front=0.0121, r_out_rear=0.0119,
        stiffness=1500.0, damping=30.0, max_force=11.0,
    )
    manipulated_prims.append(cond_p)

    # Upgraded Route A D6-DER Elastic Cooling Hoses (`2 x 24` links with calibrated bending/torsion drives
    # compiled from ConvertAsset/hose_softbody_v10 -> replaces zero-stiffness SphericalJoint bead chain!)
    # Far ends terminate cleanly on the Lab Cold-Water Faucet Brass Barb (+0.390,-0.245,0.855) & Embedded PP Sink (-0.390,-0.245,0.798)!
    UsdGeom.Scope.Define(stage, "/World/PhysHoseChains")
    hose_chain_xfs: dict[str, list[Any]] = {"inlet": [], "outlet": []}
    n_hlinks = 24
    k_hose_bend_d6 = 0.00024  # Soft amber-latex D6-DER elastic bending (prevents roll on rest & whip during S06 15-deg tilt!)
    c_hose_bend_d6 = 0.00010
    k_hose_twist_d6 = 0.00048
    c_hose_twist_d6 = 0.00014
    for hkey, barb_local, p_start_w, p_end_w in (
        ("inlet", np.array([0.074, -0.016, -0.032]), np.array([0.020 + 0.074, -0.105 - 0.016, 0.790 - 0.032]), np.array([0.390, -0.245, 0.855])),
        ("outlet", np.array([-0.074, -0.016, -0.032]), np.array([0.020 - 0.074, -0.105 - 0.016, 0.790 - 0.032]), np.array([-0.390, -0.245, 0.798])),
    ):
        nodes = []
        for i in range(n_hlinks + 1):
            s = i / float(n_hlinks)
            pt = (1.0 - s) * p_start_w + s * p_end_w
            # Natural double-fold bench slack (arc ~0.563 m > 0.499 m max S06 lift reach while staying y >= -0.353 m on bench!)
            pt[1] -= 0.115 * math.sin(math.pi * s) - 0.055 * math.sin(3.0 * math.pi * s)
            pt[2] = max(0.758, float(pt[2]) - 0.045 * math.sin(math.pi * s))
            nodes.append(pt)
        prev_b = cond_p
        for li in range(n_hlinks):
            pa, pb = nodes[li], nodes[li + 1]
            mid = 0.5 * (pa + pb)
            lp = f"/World/PhysHoseChains/{hkey}_link_{li:02d}"
            lxf = UsdGeom.Xform.Define(stage, lp)
            lxf.AddTranslateOp().Set(Gf.Vec3d(*map(float, mid)))
            lxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
            _config_dynamic_rb(lp, 0.0022, lin_damp=3.2, ang_damp=3.5)
            _add_invisible_col(lp, "Cap", "sphere", (0.0, 0.0, 0.0), (0.0052,), bench_mat)
            manipulated_prims.append(lp)
            hose_chain_xfs[hkey].append(UsdGeom.Xformable(lxf.GetPrim()))
            d0 = barb_local if li == 0 else (pa - 0.5 * (nodes[li - 1] + nodes[li]))
            d1 = pa - mid
            if li == 0:
                sj = UsdPhysics.SphericalJoint.Define(stage, f"/World/PhysHoseChains/{hkey}_j_{li:02d}")
                sj.CreateBody0Rel().SetTargets([Sdf.Path(prev_b)])
                sj.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
                sj.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, d0)))
                sj.CreateLocalPos1Attr().Set(Gf.Vec3f(*map(float, d1)))
            else:
                # Route A D6-DER elastic joint (locked translations + soft-latex bending & torsional angular drives!)
                dj = UsdPhysics.Joint.Define(stage, f"/World/PhysHoseChains/{hkey}_d6_{li:02d}")
                dj.CreateBody0Rel().SetTargets([Sdf.Path(prev_b)])
                dj.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
                dj.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, d0)))
                dj.CreateLocalPos1Attr().Set(Gf.Vec3f(*map(float, d1)))
                for ax in ("transX", "transY", "transZ"):
                    lim = UsdPhysics.LimitAPI.Apply(dj.GetPrim(), ax)
                    lim.CreateLowAttr(1.0)
                    lim.CreateHighAttr(-1.0)
                for ax, kv, cv in (("rotX", k_hose_twist_d6, c_hose_twist_d6), ("rotY", k_hose_bend_d6, c_hose_bend_d6), ("rotZ", k_hose_bend_d6, c_hose_bend_d6)):
                    drv = UsdPhysics.DriveAPI.Apply(dj.GetPrim(), ax)
                    drv.CreateTypeAttr("force")
                    drv.CreateStiffnessAttr().Set(float(kv))
                    drv.CreateDampingAttr().Set(float(cv))
                    drv.CreateMaxForceAttr().Set(0.0022)
                    drv.CreateTargetPositionAttr().Set(0.0)
            prev_b = lp
        sj_end = UsdPhysics.SphericalJoint.Define(stage, f"/World/PhysHoseChains/{hkey}_j_anchor")
        sj_end.CreateBody1Rel().SetTargets([Sdf.Path(prev_b)])
        sj_end.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, p_end_w)))
        d_last = nodes[-1] - 0.5 * (nodes[-2] + nodes[-1])
        sj_end.CreateLocalPos1Attr().Set(Gf.Vec3f(*map(float, d_last)))

    # 3e. Dynamic Vacuum Tail Adapter (`2.4° concentric 24/29 female sleeve + upright CoM over TailGripStem + non-colliding drip clearance!`)
    tail_p = "/World/Instruments/vacuum_tail_adapter"
    _config_dynamic_rb(tail_p, 0.068, lin_damp=1.4, ang_damp=2.2, com_xyz=(0.056, 0.0, -0.025))
    tail_sock_scope = _add_16stave_concave_socket_x(tail_p, "TailFemale2429", x_center=0.0, length=0.028, r_inner_mid=0.0116, mouth_sign=-1.0, rot_y_deg=15.0, flare_deg=2.2, add_rolled_rim=False)
    _add_invisible_col(tail_p, "TailGripStem", "cylinder", (+0.056, 0.0, -0.025), (0.0085, 0.030), glass_grip_mat, axis="Z")
    manipulated_prims.append(tail_p)

    # 3f. Dynamic 150 mL Erlenmeyer Receiver (`hollow 16-stave mouth so tail drip tip enters without collision!`)
    erl_p = "/World/Instruments/erlenmeyer_flask_150ml"
    _config_dynamic_rb(erl_p, 0.115, lin_damp=1.5, ang_damp=2.4, com_xyz=(0.0, 0.0, 0.024))
    _add_invisible_col(erl_p, "BaseDisc", "cylinder", (0.0, 0.0, +0.016), (0.034, 0.032), bench_mat, axis="Z")
    _add_16stave_concave_socket_z(
        erl_p, "HollowMouth", z_center=0.082, length=0.034, r_inner_mid=0.0145,
        flare_deg=1.5, add_floor=False, add_rolled_rim=False, stave_mat=glass_grip_mat,
    )
    manipulated_prims.append(erl_p)

    # 3g. Dynamic Long-Stem Funnel (`two-ring non-jamming 19/26 plug spanning both J04_Lower & J04_Upper!`)
    fn_p = "/World/Instruments/long_stem_funnel"
    _config_dynamic_rb(fn_p, 0.055, lin_damp=1.8, ang_damp=2.8, com_xyz=(0.0, 0.0, -0.006))
    _add_invisible_col(fn_p, "Stem", "cylinder", (0.0, 0.0, -0.038), (0.0036, 0.052), taper_glide_mat, axis="Z")
    _add_invisible_col(fn_p, "FunnelLeadTip", "cylinder", (0.0, 0.0, -0.018), (0.0078, 0.008), taper_glide_mat, axis="Z")
    _add_invisible_col(fn_p, "FunnelConePlug", "cylinder", (0.0, 0.0, -0.0035), (0.0091, 0.025), frosted_wedge_mat, axis="Z")
    _add_invisible_col(fn_p, "FunnelSeatCollar", "cylinder", (0.0, 0.0, +0.021), (0.0172, 0.010), frosted_wedge_mat, axis="Z")
    _add_invisible_col(fn_p, "FunnelGripRim", "cylinder", (0.0, 0.0, +0.034), (0.0210, 0.016), glass_grip_mat, axis="Z")
    manipulated_prims.append(fn_p)

    # 3h. Dynamic 100 mL Pour Beaker
    bkr_p = "/World/Instruments/pour_beaker_100ml"
    _config_dynamic_rb(bkr_p, 0.088, lin_damp=1.4, ang_damp=2.2, com_xyz=(0.0, 0.0, 0.025))
    _add_invisible_col(bkr_p, "BeakerBase", "cylinder", (0.0, 0.0, +0.012), (0.0255, 0.024), bench_mat, axis="Z")
    _add_invisible_col(bkr_p, "BeakerGripBody", "cylinder", (0.0, 0.0, +0.040), (0.0255, 0.032), glass_grip_mat, axis="Z")
    manipulated_prims.append(bkr_p)

    # 3i. Dynamic Calibrated Screw-Cap Mercury Thermometer Unit (`0.00 mm` bulb flush @ z = 1.0800 m!)
    th_p = "/World/Instruments/thermometer_adapter_unit_v08"
    _config_dynamic_rb(th_p, 0.074, lin_damp=1.4, ang_damp=2.2, com_xyz=(0.0, 0.0, 0.005))
    _add_invisible_col(th_p, "PlugTip1926", "cylinder", (0.0, 0.0, -0.016), (0.0080, 0.010), taper_glide_mat, axis="Z")
    _add_invisible_col(th_p, "PlugMid1926", "cylinder", (0.0, 0.0, -0.001), (0.0092, 0.024), frosted_wedge_mat, axis="Z")
    _add_invisible_col(th_p, "RackCollar", "cylinder", (0.0, 0.0, +0.0236), (0.0172, 0.010), frosted_wedge_mat, axis="Z")
    _add_invisible_col(th_p, "RedScrewCap", "cylinder", (0.0, 0.0, +0.036), (0.0145, 0.018), glass_grip_mat, axis="Z")
    _add_invisible_col(th_p, "MercuryStem", "cylinder", (0.0, 0.0, -0.040), (0.0030, 0.052), taper_glide_mat, axis="Z")
    manipulated_prims.append(th_p)

    # =========================================================================
    # 4. Articulated Prismatic Grippers + Dual Axial Wrist Load-Cells (`kinematicEnabled = False`!)
    # =========================================================================
    UsdGeom.Scope.Define(stage, "/World/PhysGrippers")

    def _build_phys_gripper(tag: str, q7_init: np.ndarray, w_init: float, bp: np.ndarray, bq: np.ndarray) -> dict[str, Any]:
        ee_p, ee_r, ee_q = forward_kinematics_panda(q7_init, base_pos=bp, base_quat_wxyz=bq)
        w_path = f"/World/PhysGrippers/Wrist_{tag}"
        w_xf = UsdGeom.Xform.Define(stage, w_path)
        w_tr = w_xf.AddTranslateOp()
        w_or = w_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        w_tr.Set(Gf.Vec3d(*map(float, ee_p)))
        w_or.Set(Gf.Quatd(*map(float, ee_q)))
        w_rb = UsdPhysics.RigidBodyAPI.Apply(w_xf.GetPrim())
        w_rb.CreateRigidBodyEnabledAttr().Set(True)
        w_rb.CreateKinematicEnabledAttr().Set(True)

        lc_path = f"/World/PhysGrippers/WristLoadCell_{tag}"
        lc_g = UsdGeom.Xform.Define(stage, lc_path)
        lc_g.AddTranslateOp().Set(Gf.Vec3d(*map(float, ee_p)))
        lc_g.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, ee_q)))
        _config_dynamic_rb(lc_path, 0.220, lin_damp=0.0, ang_damp=0.0)
        manipulated_prims.append(lc_path)

        lc_j = UsdPhysics.PrismaticJoint.Define(stage, f"/World/PhysGrippers/LoadCellJointZ_{tag}")
        lc_j.CreateAxisAttr("Z")
        lc_j.CreateBody0Rel().SetTargets([Sdf.Path(w_path)])
        lc_j.CreateBody1Rel().SetTargets([Sdf.Path(lc_path)])
        lc_j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        lc_j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        lc_j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        lc_j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        lc_j.CreateLowerLimitAttr().Set(-0.006)
        lc_j.CreateUpperLimitAttr().Set(+0.006)
        lc_drv = UsdPhysics.DriveAPI.Apply(lc_j.GetPrim(), "linear")
        lc_drv.CreateStiffnessAttr().Set(28000.0)
        lc_drv.CreateDampingAttr().Set(180.0)
        lc_drv.CreateMaxForceAttr().Set(160.0)
        lc_drv.CreateTargetPositionAttr().Set(0.0)
        lc_xf = UsdGeom.Xformable(lc_g.GetPrim())

        fingers: dict[str, Any] = {}
        pad_x_len = 0.038 if tag == "L" else 0.026
        for side, sgn in (("Left", +1.0), ("Right", -1.0)):
            fp = f"/World/PhysGrippers/Finger{side}_{tag}"
            f_xf = UsdGeom.Xform.Define(stage, fp)
            p0 = ee_p + ee_r @ np.array([0.0, sgn * float(w_init), 0.0], dtype=np.float64)
            f_xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, p0)))
            f_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, ee_q)))
            _config_dynamic_rb(fp, 0.024, lin_damp=0.0, ang_damp=0.0)
            _add_invisible_col(fp, "SiliconePad", "box", (0.0, sgn * 0.006, -0.002), (pad_x_len, 0.012, 0.024), pad_mat)
            manipulated_prims.append(fp)

            pj = UsdPhysics.PrismaticJoint.Define(stage, f"/World/PhysGrippers/Joint{side}_{tag}")
            pj.CreateAxisAttr("Y")
            pj.CreateBody0Rel().SetTargets([Sdf.Path(lc_path)])
            pj.CreateBody1Rel().SetTargets([Sdf.Path(fp)])
            pj.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            pj.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            pj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            pj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            pj.CreateLowerLimitAttr().Set(0.0 if sgn > 0 else -0.044)
            pj.CreateUpperLimitAttr().Set(0.044 if sgn > 0 else 0.0)
            drv = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
            drv.CreateStiffnessAttr().Set(14000.0)
            drv.CreateDampingAttr().Set(18.0)
            drv.CreateMaxForceAttr().Set(165.0)
            drv.CreateTargetPositionAttr().Set(float(sgn * w_init))
            fingers[side] = {"xf": UsdGeom.Xformable(f_xf.GetPrim()), "drv": drv, "sgn": sgn}
        return {"w_tr": w_tr, "w_or": w_or, "prev_q": ee_q.copy(), "fingers": fingers, "loadcell_xf": lc_xf}

    grip_L = _build_phys_gripper("L", sched["qL_seq"][0], sched["wL_seq"][0], bL, qL)
    grip_R = _build_phys_gripper("R", sched["qR_seq"][0], sched["wR_seq"][0], bR, qR)
    # Internal bore micro-spring wedge spheres only contact glass plugs inside frosted sockets
    for f_prim in (
        "/World/PhysGrippers/FingerLeft_L",
        "/World/PhysGrippers/FingerRight_L",
        "/World/PhysGrippers/FingerLeft_R",
        "/World/PhysGrippers/FingerRight_R",
    ):
        fp_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f_prim))
        rel = fp_api.CreateFilteredPairsRel()
        for wp in wedge_pad_joints:
            rel.AddTarget(Sdf.Path(wp["pad_prim"]))
        if f_prim.endswith("_R"):
            for ts_child in stage.GetPrimAtPath(tail_sock_scope).GetChildren():
                rel.AddTarget(ts_child.GetPath())
    vis_L = _bind_panda_visual_ops(stage, UsdGeom, Gf, "/World/franka_left")
    vis_R = _bind_panda_visual_ops(stage, UsdGeom, Gf, "/World/franka_right")

    # Anti-Cheat Gate BEFORE `attach_stage`: verify `kinematicEnabled == False` on ALL manipulated prims AND zero Weld/Fixed joints on glassware!
    any_kinematic_manipulated = False
    for mp in manipulated_prims:
        pr = stage.GetPrimAtPath(mp)
        ka = pr.GetAttribute("physics:kinematicEnabled")
        if ka and ka.IsValid() and bool(ka.Get()):
            any_kinematic_manipulated = True

    weld_joint_count = 0
    for pr in stage.Traverse():
        if pr.IsA(UsdPhysics.FixedJoint):
            weld_joint_count += 1

    # =========================================================================
    # 5. Configure Synchronized Multi-Camera Suite (`Front Master + Ortho Side + Active Macro`)
    # =========================================================================
    cams_cfg = spec["cameras"]
    fc_m = cams_cfg["Cam_Front_Master"]
    cam_front = UsdGeom.Camera.Define(stage, "/World/Cameras/Live_Front")
    cam_front.CreateFocalLengthAttr().Set(float(fc_m["focal_mm"]))
    cam_front.CreateHorizontalApertureAttr().Set(20.955)
    cam_front.CreateVerticalApertureAttr().Set(11.787)
    cam_front.CreateClippingRangeAttr().Set(Gf.Vec2f(0.04, 50.0))
    UsdGeom.Xformable(cam_front).AddTransformOp().Set(_lookat_matrix4d(Gf, fc_m["eye"], fc_m["target"]))

    cam_side = UsdGeom.Camera.Define(stage, "/World/Cameras/Live_Side")
    cam_side.CreateFocalLengthAttr().Set(35.0)
    cam_side.CreateHorizontalApertureAttr().Set(20.955)
    cam_side.CreateVerticalApertureAttr().Set(15.716)
    cam_side.CreateClippingRangeAttr().Set(Gf.Vec2f(0.03, 50.0))
    side_tf_op = UsdGeom.Xformable(cam_side).AddTransformOp()
    side_focal_attr = cam_side.GetFocalLengthAttr()

    cam_macro = UsdGeom.Camera.Define(stage, "/World/Cameras/Live_Macro")
    cam_macro.CreateFocalLengthAttr().Set(38.0)
    cam_macro.CreateHorizontalApertureAttr().Set(20.955)
    cam_macro.CreateVerticalApertureAttr().Set(15.716)
    cam_macro.CreateClippingRangeAttr().Set(Gf.Vec2f(0.03, 25.0))
    macro_tf_op = UsdGeom.Xformable(cam_macro).AddTransformOp()
    macro_focal_attr = cam_macro.GetFocalLengthAttr()

    stage_side_cam_map = {
        0: "Cam_Ortho_Side_Left",
        1: "Cam_Ortho_Side_Left",
        2: "Cam_Ortho_Side_Left",
        3: "Cam_Ortho_Side_Left",
        4: "Cam_Ortho_Side_Right",
        5: "Cam_Ortho_Side_Right",
        6: "Cam_Ortho_Side_Right",
        7: "Cam_Ortho_Side_Left",
        8: "Cam_Ortho_Side_Left",
        9: "Cam_Ortho_Side_Right",
        10: "Cam_Ortho_Side_Right",
        11: "Cam_Ortho_Side_Left",
    }
    stage_macro_cam_map = {
        0: "Cam_JackBlue_Close",
        1: "Cam_Bosshead_Close",
        2: "Cam_Bosshead_Close",
        3: "Cam_HeadFlask_Close",
        4: "Cam_Clamp3Prong_Close",
        5: "Cam_Clamp3Prong_Close",
        6: "Cam_Receiver_Drip",
        7: "Cam_FunnelPour_Close",
        8: "Cam_Thermometer_Bulb",
        9: "Cam_Receiver_Drip",
        10: "Cam_Clamp3Prong_Close",
        11: "Cam_HeadFlask_Close",
    }

    if record_video:
        rp_front = rep.create.render_product("/World/Cameras/Live_Front", (1280, 720))
        an_front = rep.AnnotatorRegistry.get_annotator("rgb")
        an_front.attach([rp_front])

        rp_side = rep.create.render_product("/World/Cameras/Live_Side", (720, 540))
        an_side = rep.AnnotatorRegistry.get_annotator("rgb")
        an_side.attach([rp_side])

        rp_macro = rep.create.render_product("/World/Cameras/Live_Macro", (720, 540))
        an_macro = rep.AnnotatorRegistry.get_annotator("rgb")
        an_macro.attach([rp_macro])

    # Attach Stage to PhysX 5 GPU (`240 Hz` explicit substeps!)
    sim_iface = get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    sim_iface.attach_stage(stage_id)

    sim_dt = 1.0 / 240.0
    substeps_per_frame = 8
    sim_time = 0.0

    # Warm-up 24 PhysX substeps (0.1s) so all 9 staged bodies & 48 hose links settle cleanly
    for _ in range(24):
        sim_iface.simulate(sim_dt, sim_time)
        sim_iface.fetch_results()
        sim_time += sim_dt

    def _world_pos_quat(prim_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mw = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tv = mw.ExtractTranslation()
        qv = mw.ExtractRotationQuat()
        im = qv.GetImaginary()
        w, x, y, z = float(qv.GetReal()), float(im[0]), float(im[1]), float(im[2])
        Rq = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return np.array([float(tv[0]), float(tv[1]), float(tv[2])], dtype=np.float64), np.array([w, x, y, z], dtype=np.float64), Rq

    def _local_y_spin_rad(prim_path: str, R_parent_inv: np.ndarray) -> float:
        _, _, Rw = _world_pos_quat(prim_path)
        R_rel = R_parent_inv @ Rw
        return math.atan2(-float(R_rel[2, 0]), float(R_rel[0, 0]))

    # Bind Reversible Signed Leadscrew Trackers on all 4 Rotary Mechanisms!
    _, _, R_cbase3 = _world_pos_quat(cbase3_p)
    leadscrew_trackers: dict[str, dict[str, Any]] = {
        "jack_blue": {
            "prim": "/World/Mechanisms/scissor_jack_blue_art001/Handwheel",
            "R_inv": np.eye(3),
            "drv": UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/Mechanisms/scissor_jack_blue_art001/LiftPrismatic"), "linear"),
            "deg": 0.0,
            "deg_bounds": (0.0, 340.0),
            "map_fn": lambda d: 0.018 * max(0.0, min(1.0, d / 290.0)),
        },
        "jack_white": {
            "prim": "/World/Mechanisms/scissor_jack_white_art001/Handwheel",
            "R_inv": np.eye(3),
            "drv": UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/Mechanisms/scissor_jack_white_art001/LiftPrismatic"), "linear"),
            "deg": 0.0,
            "deg_bounds": (0.0, 260.0),
            "map_fn": lambda d: 0.028 * max(0.0, min(1.0, d / 195.0)),
        },
        "clamp_2prong": {
            "prim": wing2_p,
            "R_inv": np.eye(3),
            "drv": jdrv2,
            "deg": 95.0,  # Starts semi-closed (`+14.2°`); -120° in S02 -> 0° (`+32°` wide open); +240° in S03 -> 215° (`-4.5°` centered V-pocket lock!)
            "deg_bounds": (0.0, 230.0),
            "map_fn": lambda d: 32.0 - 36.5 * max(0.0, min(1.0, d / 195.0)),
        },
        "clamp_3prong": {
            "prim": cwing3_p,
            "R_inv": R_cbase3.T,
            "drv": udrv3,
            "deg": 95.0,  # Starts semi-closed (`-22.8°`); -120° in S05 -> 0° (`-54°` wide open); +240° in S06 -> 215° (`+10.0°` slip-proof coaxial 15° lock!)
            "deg_bounds": (0.0, 230.0),
            "map_fn": lambda d: -54.0 + 64.0 * max(0.0, min(1.0, d / 195.0)),
        },
    }
    for trk in leadscrew_trackers.values():
        trk["prev_spin"] = _local_y_spin_rad(trk["prim"], trk["R_inv"])

    in_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Inlet_Lower"))
    out_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Outlet_Upper"))
    in_pts_a, in_nrm_a = in_mesh.GetPointsAttr(), in_mesh.GetNormalsAttr()
    out_pts_a, out_nrm_a = out_mesh.GetPointsAttr(), out_mesh.GetNormalsAttr()

    flk_pool_p = stage.GetPrimAtPath("/World/DynamicFluids/FlaskPurplePool")
    flk_pool_sc = _find_op(UsdGeom.Xformable(flk_pool_p), "xformOp:scale", UsdGeom.XformOp.TypeScale)
    flk_pool_tr = _find_op(UsdGeom.Xformable(flk_pool_p), "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
    rec_pool_p = stage.GetPrimAtPath("/World/DynamicFluids/ReceiverClearPool")
    rec_pool_sc = _find_op(UsdGeom.Xformable(rec_pool_p), "xformOp:scale", UsdGeom.XformOp.TypeScale)
    rec_pool_tr = _find_op(UsdGeom.Xformable(rec_pool_p), "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
    bulb_drop_p = stage.GetPrimAtPath("/World/DynamicFluids/BulbCondensateDrop")
    pour_drops = [stage.GetPrimAtPath(f"/World/DynamicFluids/PourDrop_{i:02d}") for i in range(18)]
    pour_drop_trs = [_find_op(UsdGeom.Xformable(p), "xformOp:translate", UsdGeom.XformOp.TypeTranslate) for p in pour_drops]
    dist_drops = [stage.GetPrimAtPath(f"/World/DynamicFluids/DistillateDrop_{i:02d}") for i in range(6)]
    dist_drop_trs = [_find_op(UsdGeom.Xformable(p), "xformOp:translate", UsdGeom.XformOp.TypeTranslate) for p in dist_drops]

    if record_video:
        sc0 = cams_cfg[stage_side_cam_map[0]]
        side_focal_attr.Set(float(sc0["focal_mm"]))
        side_tf_op.Set(_lookat_matrix4d(Gf, sc0["eye"], sc0["target"]))
        mc0 = cams_cfg[stage_macro_cam_map[0]]
        macro_focal_attr.Set(float(mc0["focal_mm"]))
        macro_tf_op.Set(_lookat_matrix4d(Gf, mc0["eye"], mc0["target"]))
        for _ in range(6):
            rep.orchestrator.step(rt_subframes=3, delta_time=0.0, pause_timeline=True)

    T_full = int(sched["T"])
    T = min(T_full, max_frames_override) if max_frames_override > 0 else T_full
    qL_seq, wL_seq = sched["qL_seq"], sched["wL_seq"]
    qR_seq, wR_seq = sched["qR_seq"], sched["wR_seq"]
    stage_idx_seq = sched["stage_index"]
    stages_meta = [
        {
            **st_m,
            "title_zh": str(st_m.get("title_zh", ""))
            .replace("、绿色标准口夹", "、方案B裸磨口自锁")
            .replace("+ 标准口绿夹 ", "+ 裸磨口方案B自锁 "),
        }
        for st_m in spec["stages"]
    ]
    stage_rep_frames = sched["stage_rep_frames"]
    macro_schedule = sched["macro_schedule"]
    frame_to_sop_macros: dict[int, list[tuple[str, str]]] = {}
    for mkey, (cname, flist, _) in macro_schedule.items():
        for fn in flist:
            frame_to_sop_macros.setdefault(fn, []).append((mkey, cname))

    tracked_bodies = (
        "heating_stirrer_bath",
        "round_bottom_flask_250ml",
        "distillation_head_75deg",
        "liebig_condenser_24_29",
        "vacuum_tail_adapter",
        "erlenmeyer_flask_150ml",
        "long_stem_funnel",
        "pour_beaker_100ml",
        "thermometer_adapter_unit_v08",
    )
    body_pos_hist = {b: np.zeros((T, 3), dtype=np.float64) for b in tracked_bodies}
    body_rot_hist = {b: np.zeros((T, 3, 3), dtype=np.float64) for b in tracked_bodies}
    screw_deg_hist = {k: np.zeros((T,), dtype=np.float64) for k in leadscrew_trackers}
    wedge_force_hist = np.zeros((T,), dtype=np.float64)
    axial_force_L_hist = np.zeros((T,), dtype=np.float64)
    axial_force_R_hist = np.zeros((T,), dtype=np.float64)
    pad_force_L_hist = np.zeros((T,), dtype=np.float64)
    pad_force_R_hist = np.zeros((T,), dtype=np.float64)
    hose_min_z_hist = np.zeros((T,), dtype=np.float64)

    vdir = out_dir / "videos"
    stream_writers: dict[str, _StreamingH264Writer] = {}
    sop_macro_writers: dict[str, _StreamingH264Writer] = {}
    if record_video:
        vdir.mkdir(parents=True, exist_ok=True)
        for vtag in (
            "master_front_hud_v09",
            "master_front_clean_v09",
            "ortho_side_uncut_v09",
            "active_macro_uncut_v09",
            "triple_sync_uncut_v09",
        ):
            stream_writers[vtag] = _StreamingH264Writer(vdir / f"{vtag}.mp4", fps=30)
        for mkey in macro_schedule:
            sop_macro_writers[mkey] = _StreamingH264Writer(vdir / f"{mkey}.mp4", fps=30)
    keyframes_dir = out_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    servo_triggers: dict[int, list[tuple[str, str, str | None]]] = {}
    for wname, wspec in sched.get("servo_windows", {}).items():
        fa_s, _, b_id = wspec[0], wspec[1], wspec[2]
        h_id = wspec[3] if len(wspec) >= 4 and wspec[3] else None
        servo_triggers.setdefault(int(fa_s), []).append((wname, f"/World/Instruments/{b_id}", f"/World/Instruments/{h_id}" if h_id else None))

    qL_prev, wL_prev = qL_seq[0].copy(), float(wL_seq[0])
    qR_prev, wR_prev = qR_seq[0].copy(), float(wR_seq[0])
    flask_purple_level = 0.0
    receiver_clear_level = 0.0

    for f in range(T):
        if (not zero_friction) and (f in servo_triggers):
            for wname, b_prim, h_prim in servo_triggers[f]:
                lp_w, _, lR_w = _world_pos_quat(b_prim)
                hp_w, _, hR_w = _world_pos_quat(h_prim) if h_prim else (None, None, None)
                replan_closed_loop_window(sched, wname, lp_w, lR_w, bL, qL, bR, qR, held_pos=hp_w, held_rot=hR_w)

        qL_goal, wL_goal = qL_seq[f], float(wL_seq[f])
        qR_goal, wR_goal = qR_seq[f], float(wR_seq[f])
        peak_ax_L = 0.0
        peak_ax_R = 0.0

        for sub in range(substeps_per_frame):
            al = (sub + 1) / float(substeps_per_frame)
            qL_s = (1.0 - al) * qL_prev + al * qL_goal
            wL_s = (1.0 - al) * wL_prev + al * wL_goal
            qR_s = (1.0 - al) * qR_prev + al * qR_goal
            wR_s = (1.0 - al) * wR_prev + al * wR_goal

            eeL_p, _, eeL_q = forward_kinematics_panda(qL_s, base_pos=bL, base_quat_wxyz=qL)
            eeR_p, _, eeR_q = forward_kinematics_panda(qR_s, base_pos=bR, base_quat_wxyz=qR)

            for grp, ep, eq, ws in ((grip_L, eeL_p, eeL_q, wL_s), (grip_R, eeR_p, eeR_q, wR_s)):
                if float(np.dot(eq, grp["prev_q"])) < 0.0:
                    eq = -eq
                grp["prev_q"] = eq.copy()
                grp["w_tr"].Set(Gf.Vec3d(*map(float, ep)))
                grp["w_or"].Set(Gf.Quatd(*map(float, eq)))
                d_val = 58.0 if ws < 0.020 else 5.0
                grp["fingers"]["Left"]["drv"].GetDampingAttr().Set(d_val)
                grp["fingers"]["Right"]["drv"].GetDampingAttr().Set(d_val)
                grp["fingers"]["Left"]["drv"].GetTargetPositionAttr().Set(float(+ws))
                grp["fingers"]["Right"]["drv"].GetTargetPositionAttr().Set(float(-ws))

            sim_iface.simulate(sim_dt, sim_time)
            sim_iface.fetch_results()
            sim_time += sim_dt

            # Update all 4 reversible signed leadscrew couplings from LIVE PhysX screw rotation!
            for trk in leadscrew_trackers.values():
                cs = _local_y_spin_rad(trk["prim"], trk["R_inv"])
                ds = math.atan2(math.sin(cs - trk["prev_spin"]), math.cos(cs - trk["prev_spin"]))
                trk["prev_spin"] = cs
                lo_b, hi_b = trk["deg_bounds"]
                trk["deg"] = float(np.clip(trk["deg"] + math.degrees(ds), lo_b, hi_b))
                trk["drv"].GetTargetPositionAttr().Set(float(trk["map_fn"](trk["deg"])))

            if wL_s < 0.012:
                lcL_t = grip_L["loadcell_xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                dispL = float(np.linalg.norm(np.array([float(lcL_t[0]), float(lcL_t[1]), float(lcL_t[2])]) - eeL_p))
                peak_ax_L = max(peak_ax_L, min(160.0, 28000.0 * dispL))
            if wR_s < 0.012:
                lcR_t = grip_R["loadcell_xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                dispR = float(np.linalg.norm(np.array([float(lcR_t[0]), float(lcR_t[1]), float(lcR_t[2])]) - eeR_p))
                peak_ax_R = max(peak_ax_R, min(160.0, 28000.0 * dispR))

        qL_prev, wL_prev = qL_goal.copy(), wL_goal
        qR_prev, wR_prev = qR_goal.copy(), wR_goal

        eeL_p, eeL_r, _ = forward_kinematics_panda(qL_goal, base_pos=bL, base_quat_wxyz=qL)
        eeR_p, eeR_r, _ = forward_kinematics_panda(qR_goal, base_pos=bR, base_quat_wxyz=qR)
        refL_p = np.array(list(grip_L["loadcell_xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()), dtype=np.float64)
        refR_p = np.array(list(grip_R["loadcell_xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()), dtype=np.float64)

        def _live_q_lr(grip: dict[str, Any], ref_p: np.ndarray, ee_r: np.ndarray, w_cmd: float) -> tuple[float, float, float]:
            tL = grip["fingers"]["Left"]["xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            tR = grip["fingers"]["Right"]["xf"].ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            pL = np.array([float(tL[0]), float(tL[1]), float(tL[2])])
            pR = np.array([float(tR[0]), float(tR[1]), float(tR[2])])
            yL = max(0.0, min(0.042, float(np.dot(pL - ref_p, ee_r[:, 1]))))
            yR = max(0.0, min(0.042, float(-np.dot(pR - ref_p, ee_r[:, 1]))))
            fN = min(135.0, 8500.0 * max(0.0, 0.5 * (yL + yR) - w_cmd))
            return yL, yR, fN

        yLL, yLR, fnL = _live_q_lr(grip_L, refL_p, eeL_r, wL_goal)
        yRL, yRR, fnR = _live_q_lr(grip_R, refR_p, eeR_r, wR_goal)
        _apply_panda_visual_q7_and_live_fingers(vis_L, qL_goal, yLL, yLR, Gf)
        _apply_panda_visual_q7_and_live_fingers(vis_R, qR_goal, yRL, yRR, Gf)

        # Skin 64x24 Amber Latex Hose visual meshes directly onto the 48 LIVE PhysX dynamic links!
        c_pos, _, c_rot = _world_pos_quat(cond_p)
        b_in = c_pos + c_rot @ np.array([0.074, -0.012, -0.024])
        b_out = c_pos + c_rot @ np.array([-0.074, -0.012, -0.024])
        pts_in = [b_in] + [
            np.array(list(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()), dtype=np.float64)
            for xf in hose_chain_xfs["inlet"]
        ] + [np.array([0.390, -0.245, 0.855], dtype=np.float64)]
        pts_out = [b_out] + [
            np.array(list(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()), dtype=np.float64)
            for xf in hose_chain_xfs["outlet"]
        ] + [np.array([-0.390, -0.245, 0.798], dtype=np.float64)]
        c64_in = _upsample_catmull_centerline(np.asarray(pts_in), 64)
        c64_out = _upsample_catmull_centerline(np.asarray(pts_out), 64)
        vin, nin = skin_cosserat_tube(c64_in)
        vout, nout = skin_cosserat_tube(c64_out)
        in_pts_a.Set(Vt.Vec3fArray.FromNumpy(vin))
        in_nrm_a.Set(Vt.Vec3fArray.FromNumpy(nin))
        out_pts_a.Set(Vt.Vec3fArray.FromNumpy(vout))
        out_nrm_a.Set(Vt.Vec3fArray.FromNumpy(nout))
        hose_min_z_hist[f] = float(min(vin[:, 2].min(), vout[:, 2].min()))

        # Read live poses of all 9 dynamic instruments
        for bname in tracked_bodies:
            bp_w, _, R_bw = _world_pos_quat(f"/World/Instruments/{bname}")
            body_pos_hist[bname][f] = bp_w
            body_rot_hist[bname][f] = R_bw

        for k_s, trk in leadscrew_trackers.items():
            screw_deg_hist[k_s][f] = trk["deg"]

        f_rad_sum = 0.0
        for winfo in wedge_pad_joints:
            pp, _, _ = _world_pos_quat(winfo["pad_prim"])
            hp_w, _, R_hw = _world_pos_quat(winfo["host_prim"])
            disp = float(np.linalg.norm(pp - (hp_w + R_hw @ winfo["p_loc"])))
            f_rad_sum += min(winfo["max_force"], winfo["stiffness"] * disp)
        wedge_force_hist[f] = f_rad_sum
        axial_force_L_hist[f] = peak_ax_L
        axial_force_R_hist[f] = peak_ax_R
        pad_force_L_hist[f] = fnL
        pad_force_R_hist[f] = fnR

        # Update collision-free visual fluid pools riding inside dynamic Flask & dynamic Erlenmeyer!
        if float(sched["pour_active"][f]) > 0.5 and not zero_friction:
            flask_purple_level = min(1.0, flask_purple_level + 1.0 / 28.0)
            bkr_pos = body_pos_hist["pour_beaker_100ml"][f]
            fn_pos = body_pos_hist["long_stem_funnel"][f]
            for di, (dp, dtr) in enumerate(zip(pour_drops, pour_drop_trs)):
                UsdGeom.Imageable(dp).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                u_d = (f * 0.14 + di / 18.0) % 1.0
                if u_d < 0.42:
                    s_u = u_d / 0.42
                    dx = float((1.0 - s_u) * (bkr_pos[0] - 0.026) + s_u * fn_pos[0])
                    dz = float((bkr_pos[2] + 0.028) - 0.035 * (s_u * s_u))
                else:
                    s_u = (u_d - 0.42) / 0.58
                    dx = float(fn_pos[0])
                    dz = float((fn_pos[2] + 0.025) - 0.210 * s_u)
                dtr.Set(Gf.Vec3d(dx, float(fn_pos[1]), dz))
        else:
            for dp in pour_drops:
                UsdGeom.Imageable(dp).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        if flask_purple_level > 0.01:
            UsdGeom.Imageable(flk_pool_p).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            fp_now = body_pos_hist["round_bottom_flask_250ml"][f]
            flk_pool_tr.Set(Gf.Vec3d(float(fp_now[0]), float(fp_now[1]), float(fp_now[2] - 0.015)))
            flk_pool_sc.Set(Gf.Vec3f(0.92, 0.92, max(0.08, 0.58 * flask_purple_level)))

        th_now = body_pos_hist["thermometer_adapter_unit_v08"][f]
        UsdGeom.Imageable(bulb_drop_p).GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if (float(sched["bulb_drop_vis"][f]) > 0.5 and not zero_friction) else UsdGeom.Tokens.invisible
        )
        if float(sched["bulb_drop_vis"][f]) > 0.5:
            _find_op(UsdGeom.Xformable(bulb_drop_p), "xformOp:translate", UsdGeom.XformOp.TypeTranslate).Set(
                Gf.Vec3d(float(th_now[0]), float(th_now[1] - 0.0025), float(th_now[2] - 0.0675))
            )

        if float(sched["distill_drip_active"][f]) > 0.5 and not zero_friction:
            receiver_clear_level = min(1.0, receiver_clear_level + 1.0 / 70.0)
            tail_now = body_pos_hist["vacuum_tail_adapter"][f]
            tail_R_now = body_rot_hist["vacuum_tail_adapter"][f]
            drip_xyz = tail_now + tail_R_now @ np.array([+0.0714, 0.0, -0.0793], dtype=np.float64)
            for di, (dp, dtr) in enumerate(zip(dist_drops, dist_drop_trs)):
                ph = (f * 0.085 + di / 6.0) % 1.0
                if ph < 0.65:
                    UsdGeom.Imageable(dp).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                    fall = (ph / 0.65) ** 2
                    dtr.Set(Gf.Vec3d(float(drip_xyz[0]), float(drip_xyz[1]), float(drip_xyz[2] - 0.082 * fall)))
                else:
                    UsdGeom.Imageable(dp).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        else:
            for dp in dist_drops:
                UsdGeom.Imageable(dp).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        if receiver_clear_level > 0.01:
            UsdGeom.Imageable(rec_pool_p).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            ep_now = body_pos_hist["erlenmeyer_flask_150ml"][f]
            rec_pool_tr.Set(Gf.Vec3d(float(ep_now[0]), float(ep_now[1]), float(ep_now[2] + 0.005 + 0.005 * receiver_clear_level)))
            rec_pool_sc.Set(Gf.Vec3f(0.94, 0.94, max(0.12, 1.05 * receiver_clear_level)))

        if record_video:
            st_idx = int(stage_idx_seq[f])
            sc_cfg = cams_cfg[stage_side_cam_map[st_idx]]
            side_focal_attr.Set(float(sc_cfg["focal_mm"]))
            side_tf_op.Set(_lookat_matrix4d(Gf, sc_cfg["eye"], sc_cfg["target"]))

            active_sop_macros = frame_to_sop_macros.get(f, [])
            m_cam_name = active_sop_macros[0][1] if active_sop_macros else stage_macro_cam_map[st_idx]
            mc_cfg = cams_cfg[m_cam_name]
            macro_focal_attr.Set(float(mc_cfg["focal_mm"]))
            macro_tf_op.Set(_lookat_matrix4d(Gf, mc_cfg["eye"], mc_cfg["target"]))

            rep.orchestrator.step(rt_subframes=3, delta_time=0.0, pause_timeline=True)
            rf = _norm_rgb(an_front.get_data())
            rs = _norm_rgb(an_side.get_data())
            rm = _norm_rgb(an_macro.get_data())

            bulb_err_now = abs((th_now[2] - 0.0520) - 1.0800) * 1000.0
            tel_txt = f"F_L={fnL:4.1f}N F_R={fnR:4.1f}N | WedgeN={f_rad_sum:4.1f}N | Pull={max(peak_ax_L, peak_ax_R):4.1f}N | BulbErr={bulb_err_now:4.2f}mm"
            hf = _decorate_master_hud(
                rf,
                Image=Image,
                ImageDraw=ImageDraw,
                f_title=f_title,
                f_sub=f_sub,
                f_mono=f_mono,
                stage_meta=stages_meta[st_idx],
                phase_txt=sched["phase_labels"][f],
                telemetry_txt=tel_txt,
                frame_idx=f,
                total_frames=T,
                zero_friction=zero_friction,
            )
            hs = _decorate_sub_hud(
                rs,
                Image=Image,
                ImageDraw=ImageDraw,
                f_title=f_title,
                f_mono=f_mono,
                header=f"[Cam 2 · 90° Ortho Profile] {stage_side_cam_map[st_idx]}",
                sub=f"JackB={screw_deg_hist['jack_blue'][f]:.0f}° JackW={screw_deg_hist['jack_white'][f]:.0f}° | C2={screw_deg_hist['clamp_2prong'][f]:.0f}° C3={screw_deg_hist['clamp_3prong'][f]:.0f}°",
                badge_color=(56, 189, 248),
            )
            hm = _decorate_sub_hud(
                rm,
                Image=Image,
                ImageDraw=ImageDraw,
                f_title=f_title,
                f_mono=f_mono,
                header=f"[Cam 3 · SOP Macro] {m_cam_name}",
                sub=f"WedgeRadial={f_rad_sum:.1f}N | HoseMinZ={hose_min_z_hist[f]:.3f}m (>=0.750m)",
                badge_color=(250, 204, 21),
            )

            stream_writers["master_front_clean_v09"].write(rf)
            stream_writers["master_front_hud_v09"].write(hf)
            stream_writers["ortho_side_uncut_v09"].write(hs)
            stream_writers["active_macro_uncut_v09"].write(hm)

            im_f = Image.fromarray(hf).resize((840, 540), Image.Resampling.BILINEAR)
            im_s = Image.fromarray(hs).resize((540, 540), Image.Resampling.BILINEAR)
            im_m = Image.fromarray(hm).resize((540, 540), Image.Resampling.BILINEAR)
            trip = Image.new("RGB", (1920, 540), (10, 15, 28))
            trip.paste(im_f, (0, 0))
            trip.paste(im_s, (840, 0))
            trip.paste(im_m, (1380, 0))
            stream_writers["triple_sync_uncut_v09"].write(np.asarray(trip, dtype=np.uint8))

            if f == stage_rep_frames.get(st_idx):
                sid = stages_meta[st_idx]["stage_id"]
                Image.fromarray(rf).save(keyframes_dir / f"{sid}_clean.png")
                Image.fromarray(hf).save(keyframes_dir / f"{sid}_hud.png")
                trip.save(keyframes_dir / f"{sid}_triple.png")

            if active_sop_macros:
                mkey0, _ = active_sop_macros[0]
                sop_macro_writers[mkey0].write(rm)
                if f == macro_schedule[mkey0][2]:
                    Image.fromarray(rm).save(keyframes_dir / f"{mkey0}.png")
                for mkey_ex, cname_ex in active_sop_macros[1:]:
                    mc_ex = cams_cfg[cname_ex]
                    macro_focal_attr.Set(float(mc_ex["focal_mm"]))
                    macro_tf_op.Set(_lookat_matrix4d(Gf, mc_ex["eye"], mc_ex["target"]))
                    rep.orchestrator.step(rt_subframes=3, delta_time=0.0, pause_timeline=True)
                    rm_ex = _norm_rgb(an_macro.get_data())
                    sop_macro_writers[mkey_ex].write(rm_ex)
                    if f == macro_schedule[mkey_ex][2]:
                        Image.fromarray(rm_ex).save(keyframes_dir / f"{mkey_ex}.png")

    sim_iface.detach_stage()

    # =========================================================================
    # 6. Compute Stage-by-Stage S01..S12 Steady-State & Disassembly Gate Metrics!
    # =========================================================================
    f_s01 = min(T - 1, int(sched.get("f_s01_done", 165)))
    f_s08 = min(T - 1, int(sched.get("f_s08_pour_mid", 1416)))
    f_s10 = min(T - 1, int(sched.get("f_s10_mid", 1630)))
    f_s12_start = min(T - 1, int(sched.get("f_s12_start", 2125)))
    f_end = T - 1

    bath_dz_s01_mm = float((body_pos_hist["heating_stirrer_bath"][f_s01, 2] - body_pos_hist["heating_stirrer_bath"][0, 2]) * 1000.0)
    flk_p10 = body_pos_hist["round_bottom_flask_250ml"][f_s10]
    flk_R10 = body_rot_hist["round_bottom_flask_250ml"][f_s10]
    flask_s10_err_mm = float(np.linalg.norm(flk_p10 - np.array([-0.180, 0.0, 0.940])) * 1000.0)
    flask_tilt_s10_deg = float(math.degrees(math.acos(np.clip(float(flk_R10[2, 2]), -1.0, 1.0))))

    hd_p10 = body_pos_hist["distillation_head_75deg"][f_s10]
    hd_R10 = body_rot_hist["distillation_head_75deg"][f_s10]
    head_s10_err_mm = float(np.linalg.norm(hd_p10 - np.array([-0.180, 0.0, 1.0260])) * 1000.0)
    head_upright_tilt_s10_deg = float(math.degrees(math.acos(np.clip(float(hd_R10[2, 2]), -1.0, 1.0))))
    j01_rel_err_mm = float(np.linalg.norm((flk_p10 + flk_R10 @ np.array([0.0, 0.0, 0.0860])) - hd_p10) * 1000.0)
    j01_rel_coax_deg = float(math.degrees(math.acos(np.clip(float(np.dot(flk_R10[:, 2], hd_R10[:, 2])), -1.0, 1.0))))

    c_p10 = body_pos_hist["liebig_condenser_24_29"][f_s10]
    c_R10 = body_rot_hist["liebig_condenser_24_29"][f_s10]
    hd_sidearm_cone_w10 = hd_p10 + hd_R10 @ np.array([+0.082104, 0.0, +0.040000], dtype=np.float64)
    hd_sidearm_axis_w10 = hd_R10 @ U15
    hd_sidearm_axis_w10 = hd_sidearm_axis_w10 / max(float(np.linalg.norm(hd_sidearm_axis_w10)), 1e-12)
    cond_upper_w10 = c_p10 + c_R10 @ np.array([-0.125, 0.0, 0.0], dtype=np.float64)
    cond_mouth_s10_err_mm = float(np.linalg.norm(cond_upper_w10 - HEAD_SIDEARM_CONE_W) * 1000.0)
    cond_coax_s10_deg = float(math.degrees(math.acos(np.clip(float(np.dot(c_R10[:, 0], U15)), -1.0, 1.0))))
    j02_rel_err_mm = float(np.linalg.norm(cond_upper_w10 - hd_sidearm_cone_w10) * 1000.0)
    j02_rel_coax_deg = float(math.degrees(math.acos(np.clip(float(np.dot(c_R10[:, 0], hd_sidearm_axis_w10)), -1.0, 1.0))))

    t_p10 = body_pos_hist["vacuum_tail_adapter"][f_s10]
    t_R10 = body_rot_hist["vacuum_tail_adapter"][f_s10]
    cond_lower_w10 = c_p10 + c_R10 @ np.array([+0.125, 0.0, 0.0], dtype=np.float64)
    tail_s10_err_mm = float(np.linalg.norm(t_p10 - COND_LOWER_CONE_W) * 1000.0)
    tail_coax_s10_deg = float(math.degrees(math.acos(np.clip(float(np.dot(t_R10 @ U15, U15)), -1.0, 1.0))))
    j03_rel_err_mm = float(np.linalg.norm(t_p10 - cond_lower_w10) * 1000.0)
    j03_rel_coax_deg = float(math.degrees(math.acos(np.clip(float(np.dot(t_R10 @ U15, c_R10[:, 0])), -1.0, 1.0))))

    erl_p10 = body_pos_hist["erlenmeyer_flask_150ml"][f_s10]
    erl_R10 = body_rot_hist["erlenmeyer_flask_150ml"][f_s10]
    erl_dz_s10_mm = float((erl_p10[2] - body_pos_hist["erlenmeyer_flask_150ml"][0, 2]) * 1000.0)
    erlenmeyer_tilt_s10_deg = float(math.degrees(math.acos(np.clip(float(erl_R10[2, 2]), -1.0, 1.0))))
    tail_drip_w10 = t_p10 + t_R10 @ np.array([+0.0714, 0.0, -0.0793], dtype=np.float64)
    drip_in_mouth_xy_mm = float(np.linalg.norm(tail_drip_w10[:2] - erl_p10[:2]) * 1000.0)

    hd_p08 = body_pos_hist["distillation_head_75deg"][f_s08]
    hd_R08 = body_rot_hist["distillation_head_75deg"][f_s08]
    hd_top08_w = hd_p08 + hd_R08 @ np.array([0.0, 0.0, 0.1100], dtype=np.float64)
    funnel_pour_err_mm = float(np.linalg.norm(body_pos_hist["long_stem_funnel"][f_s08, :2] - hd_top08_w[:2]) * 1000.0)

    hd_top10_w = hd_p10 + hd_R10 @ np.array([0.0, 0.0, 0.1060], dtype=np.float64)
    hd_lip10_z = float((hd_p10 + hd_R10 @ np.array([0.0, 0.0, 0.0540], dtype=np.float64))[2])
    th_p10 = body_pos_hist["thermometer_adapter_unit_v08"][f_s10]
    th_R10 = body_rot_hist["thermometer_adapter_unit_v08"][f_s10]
    thermo_s10_err_mm = float(np.linalg.norm(th_p10 - hd_top10_w) * 1000.0)
    j04_rel_coax_deg = float(math.degrees(math.acos(np.clip(float(np.dot(th_R10[:, 2], hd_R10[:, 2])), -1.0, 1.0))))
    bulb_top10_z = float(th_p10[2] - 0.0520)
    bulb_flush_s10_mm = float(max(abs(bulb_top10_z - 1.0800), abs(bulb_top10_z - hd_lip10_z)) * 1000.0)
    max_hose_bench_penetration_mm = float(max(0.0, (0.750 - float(hose_min_z_hist.min())) * 1000.0))

    # Disassembly Return-to-Bench Verification (`S11 & S12`)
    erl_dismount_dist_mm = float(np.linalg.norm(body_pos_hist["erlenmeyer_flask_150ml"][f_end, :2] - np.array([+0.215, 0.0])) * 1000.0)
    tail_dismount_dist_mm = float(np.linalg.norm(body_pos_hist["vacuum_tail_adapter"][f_end] - COND_LOWER_CONE_W) * 1000.0)
    cond_dismount_dist_mm = float(np.linalg.norm(body_pos_hist["liebig_condenser_24_29"][f_end] - COND_SEATED_CENTER_W) * 1000.0)
    thermo_dismount_dist_mm = float(np.linalg.norm(body_pos_hist["thermometer_adapter_unit_v08"][f_end] - np.array([-0.180, 0.0, 1.1320])) * 1000.0)
    head_dismount_dist_mm = float(np.linalg.norm(body_pos_hist["distillation_head_75deg"][f_end] - np.array([-0.180, 0.0, 1.0260])) * 1000.0)
    flask_dismount_dist_mm = float(np.linalg.norm(body_pos_hist["round_bottom_flask_250ml"][f_end] - np.array([-0.180, 0.0, 0.940])) * 1000.0)

    s10_assembled_passed = bool(
        (not any_kinematic_manipulated)
        and weld_joint_count == 0
        and bath_dz_s01_mm >= 12.0
        and flask_s10_err_mm <= 5.5
        and flask_tilt_s10_deg <= 3.5
        and head_upright_tilt_s10_deg <= 3.0
        and j01_rel_err_mm <= 2.5
        and j01_rel_coax_deg <= 2.8
        and j02_rel_err_mm <= 2.5
        and j02_rel_coax_deg <= 3.0
        and j03_rel_err_mm <= 2.5
        and j03_rel_coax_deg <= 3.0
        and thermo_s10_err_mm <= 2.2
        and j04_rel_coax_deg <= 2.2
        and erl_dz_s10_mm >= 65.0
        and erlenmeyer_tilt_s10_deg <= 1.5
        and drip_in_mouth_xy_mm <= 6.0
        and funnel_pour_err_mm <= 3.0
        and bulb_flush_s10_mm <= 1.5
        and max_hose_bench_penetration_mm <= 1.0
    )
    s12_disassembled_passed = bool(
        erl_dismount_dist_mm >= 120.0
        and tail_dismount_dist_mm >= 100.0
        and cond_dismount_dist_mm >= 120.0
        and thermo_dismount_dist_mm >= 100.0
        and head_dismount_dist_mm >= 100.0
        and flask_dismount_dist_mm >= 120.0
    )
    task_passed = bool(s10_assembled_passed and s12_disassembled_passed)

    videos_out: dict[str, str] = {}
    sop_macros_out: dict[str, str] = {}
    if record_video:
        for vtag, wr in stream_writers.items():
            vp_s = wr.close()
            if vp_s:
                videos_out[vtag] = vp_s
        for mkey, mwr in sop_macro_writers.items():
            mp_s = mwr.close()
            if mp_s:
                sop_macros_out[mkey] = mp_s

    npz_path = out_dir / "telemetry_v09_240hz.npz"
    np.savez_compressed(
        npz_path,
        **{f"pos_{k}": v for k, v in body_pos_hist.items()},
        **{f"rot_{k}": v for k, v in body_rot_hist.items()},
        **{f"screw_{k}_deg": v for k, v in screw_deg_hist.items()},
        wedge_radial_force_n=wedge_force_hist,
        axial_force_L_n=axial_force_L_hist,
        axial_force_R_n=axial_force_R_hist,
        pad_force_L_n=pad_force_L_hist,
        pad_force_R_n=pad_force_R_hist,
        hose_min_z_m=hose_min_z_hist,
    )

    metrics = {
        "zero_friction": zero_friction,
        "any_kinematic_manipulated": any_kinematic_manipulated,
        "weld_joint_count": weld_joint_count,
        "physx_hz": 240,
        "total_frames": T,
        "duration_s": round(T / 30.0, 2),
        "cpu_ik_audit": cpu_audit,
        "s10_steady_state": {
            "bath_lift_dz_mm": round(bath_dz_s01_mm, 2),
            "flask_3d_err_mm": round(flask_s10_err_mm, 3),
            "flask_tilt_deg": round(flask_tilt_s10_deg, 3),
            "head_24_29_err_mm": round(head_s10_err_mm, 3),
            "head_upright_tilt_deg": round(head_upright_tilt_s10_deg, 3),
            "j01_flask_head_rel_err_mm": round(j01_rel_err_mm, 3),
            "j01_flask_head_coax_deg": round(j01_rel_coax_deg, 3),
            "condenser_mouth_err_mm": round(cond_mouth_s10_err_mm, 3),
            "condenser_coax_err_deg": round(cond_coax_s10_deg, 3),
            "j02_head_cond_rel_err_mm": round(j02_rel_err_mm, 3),
            "j02_head_cond_coax_deg": round(j02_rel_coax_deg, 3),
            "tail_24_29_err_mm": round(tail_s10_err_mm, 3),
            "tail_coax_err_deg": round(tail_coax_s10_deg, 3),
            "j03_cond_tail_rel_err_mm": round(j03_rel_err_mm, 3),
            "j03_cond_tail_coax_deg": round(j03_rel_coax_deg, 3),
            "receiver_lift_dz_mm": round(erl_dz_s10_mm, 2),
            "erlenmeyer_tilt_deg": round(erlenmeyer_tilt_s10_deg, 3),
            "drip_in_mouth_xy_err_mm": round(drip_in_mouth_xy_mm, 3),
            "funnel_s08_xy_err_mm": round(funnel_pour_err_mm, 2),
            "thermometer_19_26_err_mm": round(thermo_s10_err_mm, 3),
            "j04_head_thermo_coax_deg": round(j04_rel_coax_deg, 3),
            "thermometer_bulb_flush_err_mm": round(bulb_flush_s10_mm, 3),
            "s10_wedge_radial_force_n": round(float(wedge_force_hist[f_s10]), 2),
            "max_hose_bench_penetration_mm": round(max_hose_bench_penetration_mm, 4),
            "passed": s10_assembled_passed,
        },
        "s11_s12_disassembly": {
            "erlenmeyer_dismount_dist_mm": round(erl_dismount_dist_mm, 2),
            "tail_dismount_dist_mm": round(tail_dismount_dist_mm, 2),
            "condenser_dismount_dist_mm": round(cond_dismount_dist_mm, 2),
            "thermometer_dismount_dist_mm": round(thermo_dismount_dist_mm, 2),
            "head_dismount_dist_mm": round(head_dismount_dist_mm, 2),
            "flask_dismount_dist_mm": round(flask_dismount_dist_mm, 2),
            "peak_s12_twist_pull_force_n": round(float(axial_force_L_hist[f_s12_start:].max()), 2),
            "passed": s12_disassembled_passed,
        },
        "peak_wedge_radial_force_n": round(float(wedge_force_hist.max()), 2),
        "peak_finger_normal_force_n": round(float(max(pad_force_L_hist.max(), pad_force_R_hist.max())), 2),
        "task_passed": task_passed,
    }
    payload = {
        "episode_id": str(job.get("episode_id", "v09_oneshot")),
        "zero_friction": zero_friction,
        "metrics": metrics,
        "telemetry_npz": npz_path.as_posix(),
        "videos": videos_out,
        "sop_macro_videos": sop_macros_out,
        "keyframes_dir": keyframes_dir.as_posix(),
    }
    res_out = job.get("result_json_path")
    if res_out:
        rp_out = Path(str(res_out))
        rp_out.parent.mkdir(parents=True, exist_ok=True)
        rp_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sys.stdout.flush()
    app.close()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    job["result_json_path"] = args.result_json.resolve().as_posix()
    res = execute_v09_oneshot_job(job)
    if not args.result_json.is_file():
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

