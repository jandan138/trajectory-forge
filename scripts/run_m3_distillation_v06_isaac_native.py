"""Execute M3 distillation_v06 closed-loop 12-stage trajectory & multi-camera RTX OmniGlass recording in Isaac Sim 6.0.1."""

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

os.environ.setdefault("ISAACSIM_PORTABLE_ROOT", r"I:\isaacsim-data\6.0.1")
os.environ.setdefault("TEMP", r"I:\isaacsim-data\6.0.1\temp")
os.environ.setdefault("TMP", r"I:\isaacsim-data\6.0.1\temp")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

DEFAULT_PKG_ROOT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\distillation_v06_package\scenario_package"
)
DEFAULT_OUT_DIR = _REPO_ROOT / "external_artifacts" / "distillation_v06_m3"
FFMPEG_FALLBACK = Path(r"G:\my_program\ffmpeg-2023-06-04-git-b1c3d81e71-full_build\bin\ffmpeg.exe")


def _apply_below_normal_priority() -> None:
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                0x00004000,  # BELOW_NORMAL_PRIORITY_CLASS
            )
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


def _encode_h264_mp4(frames_rgb: list[Any], out_mp4: Path, fps: int = 30) -> None:
    from PIL import Image  # type: ignore

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = _find_ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for idx, fr in enumerate(frames_rgb):
            Image.fromarray(fr, mode="RGB").save(td_p / f"f_{idx:05d}.png", compress_level=1)
        subprocess.check_call(
            [
                ffmpeg_bin,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(td_p / "f_%05d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_mp4),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _lookat_matrix4d(Gf: Any, eye: Sequence[float], target: Sequence[float]) -> Any:
    import numpy as np

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


def _find_op(xf: Any, op_name: str, op_type: Any = None) -> Any:
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == op_name or (op_type is not None and op_type == op.GetOpType()):
            return op
    return None


def _bind_panda_arm_ops(stage: Any, UsdGeom: Any, Gf: Any, root_prim_path: str) -> dict[str, Any]:
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


def _apply_panda_arm_q9(arm_ops: dict[str, Any], q9: Any, Gf: Any) -> None:
    for j in range(7):
        half = 0.5 * float(q9[j])
        qz = Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half)))
        arm_ops["orient_ops"][j].Set(arm_ops["base_orients"][j] * qz)
    arm_ops["lf_op"].Set(Gf.Vec3d(0.0, float(q9[7]), 0.0584))
    arm_ops["rf_op"].Set(Gf.Vec3d(0.0, -float(q9[8]), 0.0584))


def _norm_rgb(raw: Any) -> Any:
    import numpy as np

    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    arr = np.asarray(raw)[:, :, :3]
    if arr.dtype == np.uint8:
        return arr.copy()
    f = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if f.size and float(np.percentile(f, 99.0)) <= 2.0:
        f = f * 255.0
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


def _decorate_hud(
    rgb: Any,
    *,
    Image: Any,
    ImageDraw: Any,
    f_title: Any,
    f_sub: Any,
    f_mono: Any,
    stage_meta: dict[str, Any],
    frame_idx: int,
    total_frames: int,
) -> Any:
    import numpy as np

    h, w = rgb.shape[:2]
    rgba = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)

    bv_s = int(stage_meta["bilibili_start_s"])
    bv_e = int(stage_meta["bilibili_end_s"])
    bv_tag = f"BV17t4y1V7XH {bv_s//60:02d}:{bv_s%60:02d}-{bv_e//60:02d}:{bv_e%60:02d}"

    dr.rounded_rectangle([(16, 14), (w - 16, 86)], radius=10, fill=(10, 20, 36, 198), outline=(56, 189, 248, 185), width=2)
    dr.rounded_rectangle([(28, 24), (254, 49)], radius=6, fill=(2, 132, 199, 240))
    dr.text((36, 27), bv_tag, fill=(255, 255, 255, 255), font=f_mono)
    dr.text((266, 23), str(stage_meta["title_zh"]), fill=(248, 250, 252, 255), font=f_title)
    dr.text((30, 56), str(stage_meta["sop_rule_zh"]), fill=(191, 219, 254, 245), font=f_sub)

    prog = (frame_idx + 1) / float(max(1, total_frames))
    bx0, by0, bx1, by1 = w - 280, 26, w - 32, 40
    dr.rounded_rectangle([(bx0, by0), (bx1, by1)], radius=6, fill=(30, 41, 59, 230))
    fw = int(round((bx1 - bx0) * prog))
    if fw > 4:
        dr.rounded_rectangle([(bx0, by0), (bx0 + fw, by1)], radius=6, fill=(34, 197, 94, 255))
    dr.text((bx0, 48), f"Isaac Sim 6.0.1 Native 30fps | F {frame_idx+1:03d}/{total_frames}", fill=(148, 163, 184, 245), font=f_mono)
    return np.asarray(Image.alpha_composite(rgba, ov).convert("RGB"), dtype=np.uint8)


def run_plan_and_optional_isaac(pkg_root: Path, out_dir: Path, *, plan_only: bool = False) -> dict[str, Any]:
    import numpy as np
    from trajectory_forge.adapters.legacy.distillation_v06.isaac6_native_v06_director import (
        synthesize_v06_12stage_closed_loop_plan,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    scene_spec = json.loads((pkg_root / "semantics" / "v06_scene_spec.json").read_text(encoding="utf-8"))
    cad_specs = scene_spec["cad_specs"]
    cameras_cfg = scene_spec["cameras"]

    plan = synthesize_v06_12stage_closed_loop_plan(cad_specs, fps=30, frames_per_stage=60)
    tel = plan["telemetry"]
    audit = plan["audit"]

    npz_path = out_dir / "v06_trajectory_telemetry.npz"
    np.savez_compressed(npz_path, **tel)
    audit_path = out_dir / "mesh_penetration_report_v06.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if not audit["overall_passed"]:
        raise RuntimeError(f"Zero-penetration / coaxial SOP gate FAILED before rendering: {json.dumps(audit, indent=2)}")

    if plan_only:
        return {"mode": "plan_only", "audit": audit, "telemetry_npz": str(npz_path)}

    # --- Full Isaac Sim 6.0.1 Native PhysX 5 GPU + RTX OmniGlass Execution & Multi-Camera Recording ---
    _apply_below_normal_priority()
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RayTracedLighting",
            "samples_per_pixel_per_frame": 8,
            "denoiser": True,
            "anti_aliasing": 3,
            "extra_args": [
                "--/app/asyncRendering=False",
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
    import omni.timeline
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    try:
        f_title = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 16)
        f_sub = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 13)
        f_mono = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 12)
    except Exception:
        f_title = f_sub = f_mono = ImageFont.load_default()

    usd_path = pkg_root / "scene" / "main.usda"
    omni.usd.get_context().open_stage(str(usd_path))
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    rep.orchestrator.set_capture_on_play(False)

    # Keep PhysX 5 GPU scene LIVE with kinematic rigid body tracking so SDF/convex colliders step natively!
    for pr in stage.Traverse():
        if pr.HasAPI(UsdPhysics.RigidBodyAPI):
            pr.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    # Set exact lookAt matrices on cameras + create switchable Macro camera for single-pass multi-cam capture
    m_cfg = cameras_cfg["Cam_Master_Front"]
    cam_master = UsdGeom.Camera.Define(stage, "/World/Cameras/Cam_Master_Live")
    cam_master.CreateFocalLengthAttr().Set(float(m_cfg["focal_mm"]))
    cam_master.CreateHorizontalApertureAttr().Set(20.955)
    cam_master.CreateVerticalApertureAttr().Set(11.787)
    cam_master.CreateClippingRangeAttr().Set(Gf.Vec2f(0.04, 50.0))
    UsdGeom.Xformable(cam_master).AddTransformOp().Set(_lookat_matrix4d(Gf, m_cfg["eye"], m_cfg["target"]))

    cam_macro = UsdGeom.Camera.Define(stage, "/World/Cameras/Cam_Macro_Live")
    cam_macro.CreateFocalLengthAttr().Set(36.0)
    cam_macro.CreateHorizontalApertureAttr().Set(20.955)
    cam_macro.CreateVerticalApertureAttr().Set(15.716)  # 4:3 macro framing (960x720)
    cam_macro.CreateClippingRangeAttr().Set(Gf.Vec2f(0.03, 25.0))
    macro_tf_op = UsdGeom.Xformable(cam_macro).AddTransformOp()
    macro_focal_attr = cam_macro.GetFocalLengthAttr()

    # Tone bench diffuse slightly to (0.76, 0.79, 0.83) so clear glass & white scissor jack never wash out!
    bench_sh = stage.GetPrimAtPath("/World/Looks/BenchWhite/Shader")
    if bench_sh and bench_sh.IsValid():
        bench_sh.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(0.76, 0.79, 0.83))
        bench_sh.GetAttribute("inputs:roughness").Set(0.46)

    arm_left = _bind_panda_arm_ops(stage, UsdGeom, Gf, "/World/franka_left")
    arm_right = _bind_panda_arm_ops(stage, UsdGeom, Gf, "/World/franka_right")

    inst_names = [str(x) for x in tel["instrument_names"]]
    inst_ops = []
    for aid in inst_names:
        ip = stage.GetPrimAtPath(f"/World/Instruments/{aid}")
        ixf = UsdGeom.Xformable(ip)
        inst_ops.append(
            (
                _find_op(ixf, "xformOp:translate", UsdGeom.XformOp.TypeTranslate),
                _find_op(ixf, "xformOp:orient", UsdGeom.XformOp.TypeOrient),
            )
        )

    hose_in_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Inlet_Lower"))
    hose_out_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/CoolingHoses/Hose_Outlet_Upper"))
    in_pts_attr, in_nrm_attr = hose_in_mesh.GetPointsAttr(), hose_in_mesh.GetNormalsAttr()
    out_pts_attr, out_nrm_attr = hose_out_mesh.GetPointsAttr(), hose_out_mesh.GetNormalsAttr()

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

    rp_master = rep.create.render_product("/World/Cameras/Cam_Master_Live", (1280, 720))
    annot_master = rep.AnnotatorRegistry.get_annotator("rgb")
    annot_master.attach([rp_master])

    rp_macro = rep.create.render_product("/World/Cameras/Cam_Macro_Live", (960, 720))
    annot_macro = rep.AnnotatorRegistry.get_annotator("rgb")
    annot_macro.attach([rp_macro])

    qpos_seq = tel["qpos"]
    inst_poses = tel["instrument_poses"]
    hose_in_v = tel["hose_inlet_verts"]
    hose_in_n = tel["hose_inlet_norms"]
    hose_out_v = tel["hose_outlet_verts"]
    hose_out_n = tel["hose_outlet_norms"]
    flask_fill = tel["flask_purple_fill"]
    rec_fill = tel["receiver_clear_fill"]
    pour_act = tel["pour_active"]
    bulb_vis = tel["bulb_drop_vis"]
    drip_act = tel["distill_drip_active"]
    stage_idx_seq = tel["stage_index"]
    stages_meta = cad_specs["stages"]
    idx_flask = inst_names.index("round_bottom_flask_250ml")
    idx_erl = inst_names.index("erlenmeyer_flask_150ml")

    # Macro clip frame windows (keyframes chosen when arms are clear of macro lens -> 0% gripper occlusion!)
    macro_schedule: dict[str, tuple[str, list[int], int]] = {
        "macro_01_bosshead_open_up": ("Cam_Bosshead_Close", list(range(60, 125)), 119),
        "macro_02_clamp3prong_hoses": ("Cam_Clamp3Prong_Close", list(range(245, 362)), 359),
        "macro_03_thermometer_bulb_flush": ("Cam_Thermometer_Bulb", list(range(485, 595)), 568),
        "macro_04_receiver_drip": ("Cam_Receiver_Drip", list(range(395, 420)) + list(range(540, 598)), 572),
    }
    frame_to_macros: dict[int, list[tuple[str, str]]] = {}
    for mkey, (cname, flist, _) in macro_schedule.items():
        for fn in flist:
            frame_to_macros.setdefault(fn, []).append((mkey, cname))

    def _apply_frame_state(f: int) -> None:
        _apply_panda_arm_q9(arm_left, qpos_seq[f, 0:9], Gf)
        _apply_panda_arm_q9(arm_right, qpos_seq[f, 9:18], Gf)
        for ii, (top, oop) in enumerate(inst_ops):
            px, py, pz, qw, qx, qy, qz = inst_poses[f, ii]
            top.Set(Gf.Vec3d(float(px), float(py), float(pz)))
            oop.Set(Gf.Quatd(float(qw), float(qx), float(qy), float(qz)))

        in_pts_attr.Set(Vt.Vec3fArray.FromNumpy(hose_in_v[f]))
        in_nrm_attr.Set(Vt.Vec3fArray.FromNumpy(hose_in_n[f]))
        out_pts_attr.Set(Vt.Vec3fArray.FromNumpy(hose_out_v[f]))
        out_nrm_attr.Set(Vt.Vec3fArray.FromNumpy(hose_out_n[f]))

        # Purple KMnO4 solution inside round-bottom flask (moves with flask!)
        ff = float(flask_fill[f])
        if ff > 0.01:
            UsdGeom.Imageable(flk_pool_p).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            fx, fy, fz = inst_poses[f, idx_flask, :3]
            flk_pool_tr.Set(Gf.Vec3d(float(fx), float(fy), float(fz - 0.015)))
            flk_pool_sc.Set(Gf.Vec3f(0.92, 0.92, max(0.08, 0.58 * ff)))
        else:
            UsdGeom.Imageable(flk_pool_p).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        # Pouring vivid royal-purple KMnO4 stream drops in Stage S08
        if float(pour_act[f]) > 0.5:
            for di, (dp, dtr) in enumerate(zip(pour_drops, pour_drop_trs)):
                UsdGeom.Imageable(dp).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                u_d = ((f * 0.14 + di / 18.0) % 1.0)
                if u_d < 0.42:
                    # Arc from tilted beaker spout (-0.155, -0.004, 1.202) into funnel cone (-0.180, 0.0, 1.168)
                    s_u = u_d / 0.42
                    dx = -0.154 + (-0.180 - (-0.154)) * s_u
                    dz = 1.202 - 0.036 * (s_u * s_u)
                else:
                    # Stream down long funnel stem into round-bottom flask!
                    s_u = (u_d - 0.42) / 0.58
                    dx = -0.180
                    dz = 1.166 - 0.215 * s_u
                dtr.Set(Gf.Vec3d(dx, -0.002, dz))
        else:
            for dp in pour_drops:
                UsdGeom.Imageable(dp).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        # Glistening condensate drop on mercury bulb in Stage S10
        UsdGeom.Imageable(bulb_drop_p).GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if float(bulb_vis[f]) > 0.5 else UsdGeom.Tokens.invisible
        )

        # Falling clear distillate drops & rising receiver pool in Stage S10
        rf = float(rec_fill[f])
        if rf > 0.01:
            UsdGeom.Imageable(rec_pool_p).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            ex, ey, ez = inst_poses[f, idx_erl, :3]
            rec_pool_tr.Set(Gf.Vec3d(float(ex), float(ey), float(ez + 0.004 + 0.005 * rf)))
            rec_pool_sc.Set(Gf.Vec3f(0.95, 0.95, max(0.12, 1.10 * rf)))
        else:
            UsdGeom.Imageable(rec_pool_p).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        if float(drip_act[f]) > 0.5:
            for di, (dp, dtr) in enumerate(zip(dist_drops, dist_drop_trs)):
                phase = (f * 0.085 + di / 6.0) % 1.0
                if phase < 0.65:
                    UsdGeom.Imageable(dp).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                    fall = (phase / 0.65) ** 2
                    dtr.Set(Gf.Vec3d(0.215, 0.0, 0.922 - 0.082 * fall))
                else:
                    UsdGeom.Imageable(dp).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        else:
            for dp in dist_drops:
                UsdGeom.Imageable(dp).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Warm-up shaders & auto-calibrate exposure on Frame 0 so p95_luma is guaranteed in [165, 238]!
    _apply_frame_state(0)
    cc0 = cameras_cfg["Cam_Bosshead_Close"]
    macro_focal_attr.Set(float(cc0["focal_mm"]))
    macro_tf_op.Set(_lookat_matrix4d(Gf, cc0["eye"], cc0["target"]))
    for _ in range(8):
        rep.orchestrator.step(rt_subframes=4, delta_time=1.0 / 30.0, pause_timeline=False)
    test_rgb = _norm_rgb(annot_master.get_data())
    p95 = float(np.percentile(test_rgb, 95))
    if p95 > 240.0 or p95 < 140.0:
        scale = 205.0 / max(p95, 1.0)
        for pr in stage.Traverse():
            if "Light" in pr.GetTypeName():
                ia = pr.GetAttribute("inputs:intensity")
                if ia and ia.IsValid() and ia.Get() is not None:
                    ia.Set(float(ia.Get()) * scale)
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=4, delta_time=1.0 / 30.0, pause_timeline=False)

    T = int(qpos_seq.shape[0])
    master_hud_frames: list[Any] = []
    master_clean_frames: list[Any] = []
    macro_frames_map: dict[str, list[Any]] = {k: [] for k in macro_schedule}
    keyframes_dir = out_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    rep_arr = tel.get("stage_rep_frames")
    if rep_arr is not None:
        stage_rep_frames = {si: int(rep_arr[si]) for si in range(12)}
    else:
        stage_rep_frames = {si: si * 60 + 48 for si in range(12)}

    for f in range(T):
        _apply_frame_state(f)
        active_macros = frame_to_macros.get(f, [])
        if active_macros:
            # Set macro camera to first scheduled macro for this frame
            mkey0, cname0 = active_macros[0]
            mc = cameras_cfg[cname0]
            macro_focal_attr.Set(float(mc["focal_mm"]))
            macro_tf_op.Set(_lookat_matrix4d(Gf, mc["eye"], mc["target"]))

        rep.orchestrator.step(rt_subframes=4, delta_time=1.0 / 30.0, pause_timeline=False)
        raw_master = _norm_rgb(annot_master.get_data())
        st_idx = int(stage_idx_seq[f])
        hud_master = _decorate_hud(
            raw_master,
            Image=Image,
            ImageDraw=ImageDraw,
            f_title=f_title,
            f_sub=f_sub,
            f_mono=f_mono,
            stage_meta=stages_meta[st_idx],
            frame_idx=f,
            total_frames=T,
        )
        master_clean_frames.append(raw_master)
        master_hud_frames.append(hud_master)

        if f == stage_rep_frames.get(st_idx):
            sid = stages_meta[st_idx]["stage_id"]
            Image.fromarray(raw_master).save(keyframes_dir / f"{sid}_clean.png")
            Image.fromarray(hud_master).save(keyframes_dir / f"{sid}_hud.png")

        if active_macros:
            raw_mac = _norm_rgb(annot_macro.get_data())
            mkey0, cname0 = active_macros[0]
            macro_frames_map[mkey0].append(raw_mac)
            if f == macro_schedule[mkey0][2]:
                Image.fromarray(raw_mac).save(keyframes_dir / f"{mkey0}.png")
            # If a second macro shares this frame (e.g. S10 has both Bulb & Drip), step macro cam once for the 2nd angle
            for mkey_extra, cname_extra in active_macros[1:]:
                mc_e = cameras_cfg[cname_extra]
                macro_focal_attr.Set(float(mc_e["focal_mm"]))
                macro_tf_op.Set(_lookat_matrix4d(Gf, mc_e["eye"], mc_e["target"]))
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
                raw_mac_e = _norm_rgb(annot_macro.get_data())
                macro_frames_map[mkey_extra].append(raw_mac_e)
                if f == macro_schedule[mkey_extra][2]:
                    Image.fromarray(raw_mac_e).save(keyframes_dir / f"{mkey_extra}.png")

    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    master_mp4 = videos_dir / "master_full_sop_v06.mp4"
    master_clean_mp4 = videos_dir / "master_full_sop_v06_clean.mp4"
    _encode_h264_mp4(master_hud_frames, master_mp4, fps=30)
    _encode_h264_mp4(master_clean_frames, master_clean_mp4, fps=30)

    macro_mp4s: dict[str, str] = {}
    for mkey, mframes in macro_frames_map.items():
        mp = videos_dir / f"{mkey}.mp4"
        _encode_h264_mp4(mframes, mp, fps=30)
        macro_mp4s[mkey] = str(mp)

    timeline.stop()
    manifest = {
        "schema_version": "distillation-v06-isaac-native-recording/v1.0",
        "engine": "isaac_sim_6.0.1_physx5_rtx_omniglass",
        "total_frames": T,
        "fps": 30,
        "stride": 1,
        "duration_s": round(T / 30.0, 2),
        "master_video_mp4": str(master_mp4),
        "master_clean_video_mp4": str(master_clean_mp4),
        "macro_videos_mp4": macro_mp4s,
        "keyframes_dir": str(keyframes_dir),
        "penetration_audit": audit,
    }
    (out_dir / "v06_recording_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    app.close()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-root", type=Path, default=DEFAULT_PKG_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    res = run_plan_and_optional_isaac(args.pkg_root, args.out_dir, plan_only=args.plan_only)
    print(json.dumps({"overall_passed": res.get("audit", res.get("penetration_audit", {})).get("overall_passed"), "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

