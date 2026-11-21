"""Offline RTX showcase renderer for M2+M3+M4 v0.3 dual-mesh PBR deformable tasks (strictly gated)."""

from __future__ import annotations

import argparse
import ctypes
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Sequence


DEFAULT_M2_ROOT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\deformable_v03_packages"
)
DEFAULT_M3_ROOT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge\external_artifacts\deformable_v03_m3"
)
DEFAULT_M4_REPORT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\embodied-eval-os\external_artifacts\deformable_v03_m4\eos_v03_10task_certification_report.json"
)
DEFAULT_MEDIA_OUT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge\external_artifacts\deformable_v03_media"
)
ISAAC_PYTHON_BAT = Path(r"I:\isaacsim\6.0.1\python.bat")

V03_TASKS: tuple[str, ...] = (
    "coffee_croissant_tray_to_plate",
    "coffee_strawberry_tart_topping",
    "office_waffle_towel_corner_fold",
    "office_dust_cover_peel_retrieve",
    "smt_ribbon_cable_clip_route",
    "smt_esd_silicone_boot_press_fit",
    "lab_silicone_tube_manifold_insert",
    "lab_rubber_bulb_squeeze_seat",
)

TASK_TITLES_ZH: dict[str, tuple[str, str]] = {
    "coffee_croissant_tray_to_plate": ("soft_solid", "酥皮牛角包防压溃夹持与+30°旋转装盘"),
    "coffee_strawberry_tart_topping": ("soft_solid", "超脆弱(<7.5N)鲜草莓精准轻放奶油挞杯"),
    "office_waffle_towel_corner_fold": ("cloth_sheet", "无托架纯摩擦华夫格方巾揉捏提角对折"),
    "office_dust_cover_peel_retrieve": ("cloth_sheet", "防尘亚麻布揭离+动态碳粉盒抓取入框"),
    "smt_ribbon_cable_clip_route": ("rod_cable", "柔性彩色排线顺应绕障装配至PCBA U型槽"),
    "smt_esd_silicone_boot_press_fit": ("shell", "防静电硅胶套对齐排针凸台定量弹性压装"),
    "lab_silicone_tube_manifold_insert": ("rod_cable", "医用硅胶软管大挠度弯曲嵌入歧管卡口"),
    "lab_rubber_bulb_squeeze_seat": ("shell", "中空橡胶洗耳球入座与>=18%定量挤压回弹"),
}


def _apply_low_priority() -> None:
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                0x00000040,  # IDLE_PRIORITY_CLASS
            )
        except Exception:
            pass
    else:
        try:
            os.nice(19)
        except Exception:
            pass


def check_render_readiness(
    *,
    m2_root: Path = DEFAULT_M2_ROOT,
    m3_root: Path = DEFAULT_M3_ROOT,
    m4_report: Path = DEFAULT_M4_REPORT,
) -> dict[str, Any]:
    """Verify M2 OpenUSD + PBR + skinning, M3 NPZ trajectories, and M4 EOS certification at 0% GPU."""
    import numpy as np
    import yaml

    task_checks: dict[str, Any] = {}
    all_ready = True
    for task_id in V03_TASKS:
        pkg = m2_root / task_id / "scenario_package"
        usd_path = pkg / "scene" / "main.usda"
        usd_ok = usd_path.is_file()
        skin_ok = False
        tex_ok = False
        if (pkg / "scene" / "instances.yaml").is_file():
            inst = yaml.safe_load((pkg / "scene" / "instances.yaml").read_text(encoding="utf-8"))
            def_entries = [x for x in inst.get("instances", []) if x.get("role") == "manipulated_deformable"]
            if def_entries:
                aid = str(def_entries[0]["asset_id"])
                skin_ok = (pkg / "assets" / aid / "skinning.npz").is_file()
                tex_ok = all(
                    (pkg / "assets" / aid / "textures" / t).is_file()
                    for t in ("albedo.png", "normal.png", "roughness.png")
                )

        engine_npz: dict[str, dict[str, Any]] = {}
        for eng in ("mujoco4", "newton18", "genesis", "isaac6"):
            npz_path = m3_root / task_id / eng / f"{task_id}__{eng}__gentle_s0.npz"
            info: dict[str, Any] = {"exists": npz_path.is_file(), "path": str(npz_path)}
            if npz_path.is_file():
                with np.load(npz_path, allow_pickle=False) as z:
                    info["steps"] = int(z["action"].shape[0])
                    info["n_verts"] = int(z["deformable_rest_points"].shape[0])
            engine_npz[eng] = info

        mj_ok = engine_npz["mujoco4"]["exists"]
        if not (usd_ok and skin_ok and tex_ok and mj_ok):
            all_ready = False
        task_checks[task_id] = {
            "usd_exists": usd_ok,
            "skinning_exists": skin_ok,
            "pbr_textures_exists": tex_ok,
            "engines": engine_npz,
        }

    m4_ok = m4_report.is_file()
    m4_tiers: dict[str, str] = {}
    if m4_ok:
        rep = json.loads(m4_report.read_text(encoding="utf-8"))
        for task_id in V03_TASKS:
            trep = (rep.get("tasks") or {}).get(task_id, {})
            tier = str(trep.get("certification_tier", "Unverified"))
            m4_tiers[task_id] = tier
            if tier not in ("Gold", "Silver"):
                all_ready = False
    else:
        all_ready = False

    return {
        "schema_version": "deformable-v03-media-readiness/v0.1",
        "ready": bool(all_ready and len(m4_tiers) == len(V03_TASKS)),
        "m4_report_exists": m4_ok,
        "m4_certification_tiers": m4_tiers,
        "tasks": task_checks,
    }


def _find_ffmpeg_executable() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        cand = imageio_ffmpeg.get_ffmpeg_exe()
        if cand and Path(cand).is_file():
            return str(cand)
    except Exception:
        pass
    return None


def _encode_mp4(frames_rgb: list[Any], out_mp4: Path, fps: int = 30) -> None:
    from PIL import Image  # type: ignore

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = _find_ffmpeg_executable()
    if ffmpeg_bin is not None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for idx, fr in enumerate(frames_rgb):
                Image.fromarray(fr, mode="RGB").save(td_path / f"frame_{idx:05d}.png", compress_level=1)
            subprocess.check_call(
                [
                    ffmpeg_bin,
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(td_path / "frame_%05d.png"),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "19",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(out_mp4),
                ]
            )
            return

    import imageio.v2 as iio  # type: ignore

    with iio.get_writer(str(out_mp4), fps=fps, codec="libx264", pixelformat="yuv420p", quality=8) as writer:
        for fr in frames_rgb:
            writer.append_data(fr)


def _compute_vertex_normals(pts: Any, tris: Any) -> Any:
    import numpy as np

    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(pts, dtype=np.float64)
    for col in (0, 1, 2):
        np.add.at(vn, tris[:, col], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return vn / norms


def _tri_local_frame(p0: Any, p1: Any, p2: Any) -> Any:
    import numpy as np

    e1 = p1 - p0
    t1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
    n_vec = np.cross(e1, p2 - p0)
    n_ax = n_vec / max(float(np.linalg.norm(n_vec)), 1e-9)
    t2 = np.cross(n_ax, t1)
    t2 = t2 / max(float(np.linalg.norm(t2)), 1e-9)
    return np.column_stack((t1, t2, n_ax))


def _eval_skinned_visual_pts(skin_data: dict[str, Any], cur_phys_pts: Any) -> Any:
    import numpy as np

    mode = str(skin_data["mode"][0])
    phys = np.asarray(cur_phys_pts, dtype=np.float64)
    if mode == "tet_bary":
        return np.asarray(skin_data["w_skin"], dtype=np.float64) @ phys
    tr = np.asarray(skin_data["tris"], dtype=np.int64)
    tri_ids = np.asarray(skin_data["tri_ids"], dtype=np.int64)
    bw = np.asarray(skin_data["bary_weights"], dtype=np.float64)
    offsets = np.asarray(skin_data["local_offsets"], dtype=np.float64)
    frames = np.stack([_tri_local_frame(phys[t[0]], phys[t[1]], phys[t[2]]) for t in tr], axis=0)
    bound_tris = tr[tri_ids]
    base_pts = (
        bw[:, 0:1] * phys[bound_tris[:, 0]]
        + bw[:, 1:2] * phys[bound_tris[:, 1]]
        + bw[:, 2:3] * phys[bound_tris[:, 2]]
    )
    world_off = np.einsum("vij,vj->vi", frames[tri_ids], offsets)
    return base_pts + world_off


def _create_lookat_matrix(Gf: Any, eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> Any:
    import numpy as np

    eye_a = np.asarray(eye, dtype=np.float64)
    tgt_a = np.asarray(target, dtype=np.float64)
    up_a = np.asarray(up, dtype=np.float64)
    fwd = tgt_a - eye_a
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-12)
    z_cam = -fwd
    x_cam = np.cross(up_a, z_cam)
    x_cam = x_cam / max(float(np.linalg.norm(x_cam)), 1e-12)
    y_cam = np.cross(z_cam, x_cam)
    return Gf.Matrix4d(
        float(x_cam[0]), float(x_cam[1]), float(x_cam[2]), 0.0,
        float(y_cam[0]), float(y_cam[1]), float(y_cam[2]), 0.0,
        float(z_cam[0]), float(z_cam[1]), float(z_cam[2]), 0.0,
        float(eye_a[0]), float(eye_a[1]), float(eye_a[2]), 1.0,
    )


def _pick_font(ImageFont: Any, candidates: Sequence[str], size: int) -> Any:
    for cand in candidates:
        if Path(cand).is_file():
            try:
                return ImageFont.truetype(cand, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _normalize_rgb_uint8(raw_value: Any) -> Any:
    import numpy as np

    if isinstance(raw_value, dict) and "data" in raw_value:
        raw_value = raw_value["data"]
    arr = np.asarray(raw_value)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise RuntimeError(f"Unexpected RGB frame shape: {arr.shape}")
    rgb = arr[:, :, :3]
    if rgb.dtype == np.uint8:
        return rgb.copy()
    f_arr = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if f_arr.size and float(np.percentile(f_arr, 99.0)) <= 2.0:
        f_arr = f_arr * 255.0
    return np.clip(f_arr, 0.0, 255.0).astype(np.uint8)


def _find_op(xf: Any, op_name: str, op_type: Any = None) -> Any:
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == op_name or (op_type is not None and op.GetOpType() == op_type):
            return op
    return None


def _decorate_v03_frame(
    rgb_1080p: Any,
    *,
    Image: Any,
    ImageDraw: Any,
    font_title: Any,
    font_sub: Any,
    font_mono: Any,
    badge_text: str,
    title_text: str,
    readout_text: str,
    tier_label: str,
    progress: float,
) -> Any:
    import numpy as np

    base = Image.fromarray(rgb_1080p[:, :, :3].astype("uint8"), mode="RGB").resize((1280, 720), Image.Resampling.LANCZOS)
    rgba = base.convert("RGBA")
    overlay = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    draw.rounded_rectangle([(20, 16), (880, 94)], radius=12, fill=(12, 24, 42, 212), outline=(56, 189, 178, 185), width=2)
    draw.rounded_rectangle([(34, 28), (242, 54)], radius=6, fill=(13, 148, 136, 240))
    draw.text((44, 32), badge_text, fill=(255, 255, 255, 255), font=font_mono)
    draw.text((254, 27), title_text, fill=(248, 250, 252, 255), font=font_title)
    draw.text((36, 63), readout_text, fill=(196, 214, 232, 248), font=font_sub)

    outline_col = (234, 179, 8, 190) if "Gold" in tier_label else (148, 163, 184, 210)
    txt_col = (250, 204, 21, 255) if "Gold" in tier_label else (226, 232, 240, 255)
    draw.rounded_rectangle([(930, 16), (1260, 94)], radius=12, fill=(12, 24, 42, 212), outline=outline_col, width=2)
    draw.text((948, 26), f"EOS {tier_label} · INTACT", fill=txt_col, font=font_mono)
    bar_x0, bar_y0, bar_x1, bar_y1 = 948, 62, 1242, 76
    draw.rounded_rectangle([(bar_x0, bar_y0), (bar_x1, bar_y1)], radius=7, fill=(36, 54, 76, 235))
    fill_w = int(round((bar_x1 - bar_x0) * max(0.0, min(1.0, progress))))
    if fill_w > 4:
        draw.rounded_rectangle([(bar_x0, bar_y0), (bar_x0 + fill_w, bar_y1)], radius=7, fill=(20, 184, 166, 255))

    return np.asarray(Image.alpha_composite(rgba, overlay).convert("RGB"), dtype=np.uint8)


def _render_rtx_showcase(
    *,
    m2_root: Path,
    m3_root: Path,
    m4_report: Path,
    out_dir: Path,
    only_task: str | None = None,
) -> dict[str, Any]:
    _apply_low_priority()
    os.environ.setdefault("ISAACSIM_PORTABLE_ROOT", r"I:\isaacsim-data\6.0.1")

    _src_root = str(Path(__file__).resolve().parents[1] / "src")
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

    import numpy as np
    import yaml
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    from isaacsim import SimulationApp

    m4_data = json.loads(m4_report.read_text(encoding="utf-8")) if m4_report.is_file() else {}
    out_dir.mkdir(parents=True, exist_ok=True)
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
                "--/rtx-transient/dlssg/enabled=False",
                "--/persistent/omni/replicator/captureOnPlay=False",
                "--/omni/replicator/captureOnPlay=False",
            ],
        }
    )

    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt

    font_title = _pick_font(ImageFont, [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc"], 17)
    font_sub = _pick_font(ImageFont, [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"], 13)
    font_mono = _pick_font(ImageFont, [r"C:\Windows\Fonts\consolab.ttf", r"C:\Windows\Fonts\consola.ttf"], 13)

    manifest_entries: dict[str, Any] = {}
    contact_items: list[tuple[str, Path]] = []
    active_tasks: tuple[str, ...] = (only_task,) if only_task else V03_TASKS

    try:
        timeline = omni.timeline.get_timeline_interface()
        for task_id in active_tasks:
            pkg = m2_root / task_id / "scenario_package"
            usd_path = pkg / "scene" / "main.usda"
            npz_path = m3_root / task_id / "mujoco4" / f"{task_id}__mujoco4__gentle_s0.npz"
            inst = yaml.safe_load((pkg / "scene" / "instances.yaml").read_text(encoding="utf-8"))
            def_entry = next(x for x in inst["instances"] if x["role"] == "manipulated_deformable")
            aid = str(def_entry["asset_id"])
            def_prim_path = str(def_entry["source_prim_path"])

            with np.load(pkg / "assets" / aid / "skinning.npz", allow_pickle=False) as zs:
                skin_data = {k: np.asarray(zs[k]) for k in zs.files}
            with np.load(pkg / "assets" / aid / "physics_proxy.npz", allow_pickle=False) as zp:
                vis_tris = np.asarray(zp["visual_triangles"], dtype=np.int64)
            with np.load(npz_path, allow_pickle=False) as zn:
                pts_seq = np.asarray(zn["deformable_points"], dtype=np.float64)
                qpos_seq = np.asarray(zn["qpos"], dtype=np.float64)
                tac_seq = np.asarray(zn["tactile_normal_force_n"], dtype=np.float64)
                rigid_seq = np.asarray(zn["rigid_target_pose"], dtype=np.float64) if "rigid_target_pose" in zn else None

            omni.usd.get_context().open_stage(str(usd_path))
            timeline.stop()
            stage = omni.usd.get_context().get_stage()
            stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
            rep.orchestrator.set_capture_on_play(False)
            for pr in stage.Traverse():
                if pr.HasAPI(UsdPhysics.ArticulationRootAPI):
                    pr.CreateAttribute("physics:articulationEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                    pr.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                if pr.IsA(UsdPhysics.Joint):
                    pr.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(False)

            # Bind 7 revolute arm links + 2 prismatic fingers on /World/franka_panda
            link_paths = [
                "/World/franka_panda/panda_link0" + "".join(f"/panda_link{m}" for m in range(1, j + 1))
                for j in range(1, 8)
            ]
            orient_ops = []
            base_orients = []
            for lp in link_paths:
                prim = stage.GetPrimAtPath(lp)
                op = _find_op(UsdGeom.Xformable(prim), "xformOp:orient", UsdGeom.XformOp.TypeOrient)
                if op is None:
                    raise RuntimeError(f"Missing xformOp:orient on {lp}")
                orient_ops.append(op)
                base_orients.append(Gf.Quatd(op.Get()))

            hand_path = link_paths[-1] + "/panda_hand"
            lf_op = _find_op(UsdGeom.Xformable(stage.GetPrimAtPath(hand_path + "/panda_leftfinger")), "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
            rf_op = _find_op(UsdGeom.Xformable(stage.GetPrimAtPath(hand_path + "/panda_rightfinger")), "xformOp:translate", UsdGeom.XformOp.TypeTranslate)

            # Drive high-poly UV PBR deformable mesh directly in world coords on SessionLayer
            def_mesh_prim = stage.GetPrimAtPath(def_prim_path)
            def_xf = UsdGeom.Xformable(def_mesh_prim)
            def_t_op = _find_op(def_xf, "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
            def_o_op = _find_op(def_xf, "xformOp:orient", UsdGeom.XformOp.TypeOrient)
            if def_t_op is not None:
                def_t_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            if def_o_op is not None:
                def_o_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

            def_mesh = UsdGeom.Mesh(def_mesh_prim)
            def_mesh.CreateNormalsAttr()
            def_mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            pts_attr = def_mesh.GetPointsAttr()
            nrm_attr = def_mesh.GetNormalsAttr()
            vis_tris_flat = vis_tris.reshape(-1)

            toner_t_op = None
            toner_o_op = None
            if rigid_seq is not None:
                toner_prim = stage.GetPrimAtPath("/World/toner_cartridge_box")
                if toner_prim and toner_prim.IsValid():
                    txf = UsdGeom.Xformable(toner_prim)
                    toner_t_op = _find_op(txf, "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
                    toner_o_op = _find_op(txf, "xformOp:orient", UsdGeom.XformOp.TypeOrient)

            # Studio lights + exposure normalization across all rooms
            tunable_exp_attrs = []
            for pr in stage.Traverse():
                if "Light" in pr.GetTypeName() and not str(pr.GetPath()).startswith("/World/ShowcaseVisual"):
                    exp_attr = pr.GetAttribute("inputs:exposure") or pr.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float)
                    exp_attr.Set(float(exp_attr.Get() or 0.0) + 1.5)
                    tunable_exp_attrs.append(exp_attr)
            studio_specs = [
                ("/World/ShowcaseVisual/StudioKey", (0.95, -1.05, 1.85), 0.28, 3200.0, (1.0, 0.97, 0.92)),
                ("/World/ShowcaseVisual/StudioFill", (-0.85, -0.75, 1.85), 0.32, 1600.0, (0.92, 0.96, 1.0)),
                ("/World/ShowcaseVisual/StudioTop", (0.0, 0.05, 2.05), 0.30, 2000.0, (1.0, 0.98, 0.95)),
            ]
            UsdGeom.Scope.Define(stage, "/World/ShowcaseVisual")
            for s_path, s_pos, s_rad, s_int, s_col in studio_specs:
                sl = UsdLux.SphereLight.Define(stage, s_path)
                UsdGeom.Xformable(sl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*s_pos))
                sl.CreateRadiusAttr(float(s_rad))
                sl.CreateNormalizeAttr(True)
                sl.CreateIntensityAttr(float(s_int))
                sl.CreateColorAttr(Gf.Vec3f(*s_col))
                s_exp = sl.CreateExposureAttr(0.0)
                sl.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
                tunable_exp_attrs.append(s_exp)

            cam_prim = UsdGeom.Camera.Define(stage, "/World/ShowcaseCamera")
            cam_prim.CreateFocalLengthAttr().Set(15.0)
            cam_prim.CreateHorizontalApertureAttr().Set(20.955)
            cam_prim.CreateVerticalApertureAttr().Set(11.787)
            cam_prim.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
            UsdGeom.Xformable(cam_prim.GetPrim()).AddTransformOp().Set(
                _create_lookat_matrix(Gf, (0.82, -0.84, 1.32), (0.0, 0.01, 0.82), (0.0, 0.0, 1.0))
            )

            rp = rep.create.render_product("/World/ShowcaseCamera", (1920, 1080))
            rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_annot.attach([rp])

            def _apply_step(k: int) -> None:
                q_row = qpos_seq[k]
                for j in range(7):
                    half = 0.5 * float(q_row[j])
                    qz = Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half)))
                    orient_ops[j].Set(base_orients[j] * qz)
                lf_op.Set(Gf.Vec3d(0.0, float(q_row[7]), 0.0584))
                rf_op.Set(Gf.Vec3d(0.0, -float(q_row[8]), 0.0584))
                if rigid_seq is not None and toner_t_op is not None:
                    rx, ry, rz, rqw, rqx, rqy, rqz = rigid_seq[k]
                    toner_t_op.Set(Gf.Vec3d(float(rx), float(ry), float(rz)))
                    if toner_o_op is not None:
                        toner_o_op.Set(Gf.Quatd(float(rqw), float(rqx), float(rqy), float(rqz)))
                vis_pts = _eval_skinned_visual_pts(skin_data, pts_seq[k])
                vis_nrm = _compute_vertex_normals(vis_pts, vis_tris)
                fv_nrm = vis_nrm[vis_tris_flat]
                pts_attr.Set(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in vis_pts]))
                nrm_attr.Set(Vt.Vec3fArray([Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in fv_nrm]))

            _apply_step(0)
            for _ in range(25):
                app.update()
            rep.orchestrator.step(rt_subframes=16, delta_time=0.0, pause_timeline=False)
            warm_u8 = _normalize_rgb_uint8(rgb_annot.get_data(device="cpu"))
            roi_mean = float(warm_u8[220:980, 200:1720].mean())
            if roi_mean < 85.0 or roi_mean > 145.0:
                delta_ev = max(-3.5, min(6.0, math.log2((112.0 / max(roi_mean, 3.0)) ** 2.2)))
                for exp_attr in tunable_exp_attrs:
                    exp_attr.Set(float(exp_attr.Get() or 0.0) + delta_ev)
                for _ in range(8):
                    app.update()
                rep.orchestrator.step(rt_subframes=16, delta_time=0.0, pause_timeline=False)
                warm_u8 = _normalize_rgb_uint8(rgb_annot.get_data(device="cpu"))
            if float(warm_u8.std()) < 8.0:
                raise RuntimeError(f"Degenerate RTX warmup frame in {task_id}: std={float(warm_u8.std()):.2f}")

            m4_task = (m4_data.get("tasks") or {}).get(task_id, {})
            tier = str(m4_task.get("certification_tier", "Certified"))
            n_pass = int(m4_task.get("passing_engine_count", 4))
            tier_label = f"{tier} ({n_pass}/4)"
            mj_sum = (m4_task.get("engine_summaries") or {}).get("mujoco4", {})
            fam_label, title_zh = TASK_TITLES_ZH[task_id]

            n_steps = len(pts_seq)
            mid_k = n_steps // 2
            sample_ids = list(range(0, n_steps, 2))
            if (n_steps - 1) not in sample_ids:
                sample_ids.append(n_steps - 1)
            if mid_k not in sample_ids:
                sample_ids.append(mid_k)
                sample_ids.sort()

            frames_rgb: list[Any] = []
            task_kfs: list[dict[str, str]] = []
            task_contact_items: list[tuple[str, Path]] = []
            for k in sample_ids:
                _apply_step(k)
                app.update()
                subf = 4 if k in (0, mid_k, n_steps - 1) else 2
                rep.orchestrator.step(rt_subframes=subf, delta_time=0.0, pause_timeline=False)
                rgb_1080 = _normalize_rgb_uint8(rgb_annot.get_data(device="cpu"))
                if k in (0, mid_k, n_steps - 1):
                    label_key = "t0" if k == 0 else ("mid" if k == mid_k else "final")
                    kf_path = out_dir / f"{task_id}_{label_key}.png"
                    Image.fromarray(rgb_1080, mode="RGB").save(kf_path, format="PNG")
                    task_kfs.append({"label": label_key, "path": str(kf_path)})
                    task_contact_items.append((f"{task_id} | {label_key}", kf_path))

                fn_val = float(max(abs(tac_seq[k, 0]), abs(tac_seq[k, 1])))
                dec = _decorate_v03_frame(
                    rgb_1080,
                    Image=Image,
                    ImageDraw=ImageDraw,
                    font_title=font_title,
                    font_sub=font_sub,
                    font_mono=font_mono,
                    badge_text=f"{fam_label.upper()} · {k}/{n_steps - 1}",
                    title_text=title_zh,
                    readout_text=(
                        f"F_n={fn_val:.2f}N | dist={float(mj_sum.get('final_target_dist_m', 0.0))*1000:.1f}mm | "
                        f"dmg={float(mj_sum.get('damage_score', 0.0)):.3f} | 2K PBR Skinned"
                    ),
                    tier_label=tier_label,
                    progress=k / float(max(n_steps - 1, 1)),
                )
                repeat = 15 if k == 0 else (20 if k == n_steps - 1 else 1)
                for _ in range(repeat):
                    frames_rgb.append(dec)

            mp4_path = out_dir / f"{task_id}.mp4"
            _encode_mp4(frames_rgb, mp4_path, fps=30)
            mp4_bytes = mp4_path.read_bytes()
            manifest_entries[task_id] = {
                "mp4_path": str(mp4_path),
                "sha256": sha256(mp4_bytes).hexdigest(),
                "bytes": len(mp4_bytes),
                "frames": len(frames_rgb),
                "fps": 30,
                "keyframes": task_kfs,
            }
            (out_dir / f"{task_id}_entry.json").write_text(
                json.dumps({"entry": manifest_entries[task_id], "contact_items": [(l, str(p)) for l, p in task_contact_items]}, indent=2),
                encoding="utf-8",
            )
            contact_items.extend(task_contact_items)
            rgb_annot.detach([rp])
            rp.destroy()
            rep.orchestrator.stop()
            app.update()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    app.close()
    return manifest_entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or (with --confirm-rtx-render) run v0.3 8-task RTX rendering.")
    parser.add_argument("--m2-root", type=Path, default=DEFAULT_M2_ROOT)
    parser.add_argument("--m3-root", type=Path, default=DEFAULT_M3_ROOT)
    parser.add_argument("--m4-report", type=Path, default=DEFAULT_M4_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MEDIA_OUT)
    parser.add_argument("--only-task", choices=V03_TASKS, default=None)
    parser.add_argument("--check-ready", action="store_true")
    parser.add_argument("--confirm-rtx-render", action="store_true")
    args = parser.parse_args()

    readiness = check_render_readiness(m2_root=args.m2_root, m3_root=args.m3_root, m4_report=args.m4_report)
    if args.check_ready or not args.confirm_rtx_render:
        print(json.dumps(readiness, indent=2))
        if not args.check_ready and not args.confirm_rtx_render:
            print("\n[GUARD] Heavy Isaac RTX rendering blocked without --confirm-rtx-render.", file=sys.stderr)
            return 2 if not readiness["ready"] else 0
        return 0 if readiness["ready"] else 1

    if not readiness["ready"]:
        print(json.dumps(readiness, indent=2), file=sys.stderr)
        return 1

    if args.only_task is None:
        _apply_low_priority()
        env = os.environ.copy()
        env["ISAACSIM_PORTABLE_ROOT"] = r"I:\isaacsim-data\6.0.1"
        env["TEMP"] = r"I:\isaacsim-data\6.0.1\temp"
        env["TMP"] = r"I:\isaacsim-data\6.0.1\temp"
        cflags = 0x00000040 if sys.platform.startswith("win") else 0
        for tid in V03_TASKS:
            proc = subprocess.run(
                [
                    str(ISAAC_PYTHON_BAT),
                    str(Path(__file__).resolve()),
                    "--confirm-rtx-render",
                    "--m2-root", str(args.m2_root),
                    "--m3-root", str(args.m3_root),
                    "--m4-report", str(args.m4_report),
                    "--out-dir", str(args.out_dir),
                    "--only-task", tid,
                ],
                env=env,
                creationflags=cflags,
            )
            if proc.returncode != 0:
                return int(proc.returncode)

        manifest_entries: dict[str, Any] = {}
        for tid in V03_TASKS:
            ep = json.loads((args.out_dir / f"{tid}_entry.json").read_text(encoding="utf-8"))
            manifest_entries[tid] = ep["entry"]
        (args.out_dir / "media_manifest.json").write_text(json.dumps({"tasks": manifest_entries}, indent=2), encoding="utf-8")
        return 0

    if importlib.util.find_spec("isaacsim") is None:
        _apply_low_priority()
        env = os.environ.copy()
        env["ISAACSIM_PORTABLE_ROOT"] = r"I:\isaacsim-data\6.0.1"
        env["TEMP"] = r"I:\isaacsim-data\6.0.1\temp"
        env["TMP"] = r"I:\isaacsim-data\6.0.1\temp"
        cflags = 0x00000040 if sys.platform.startswith("win") else 0
        proc = subprocess.run([str(ISAAC_PYTHON_BAT), str(Path(__file__).resolve()), *sys.argv[1:]], env=env, creationflags=cflags)
        return int(proc.returncode)

    _render_rtx_showcase(
        m2_root=args.m2_root,
        m3_root=args.m3_root,
        m4_report=args.m4_report,
        out_dir=args.out_dir,
        only_task=args.only_task,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

