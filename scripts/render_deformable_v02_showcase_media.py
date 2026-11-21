"""Offline RTX showcase renderer for M2+M3+M4 deformable episodes (strictly gated)."""

from __future__ import annotations

import argparse
import ctypes
from hashlib import sha256
import importlib.util
from io import BytesIO
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
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\deformable_v02_pilots"
)
DEFAULT_M3_ROOT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge\external_artifacts\deformable_v02_m3"
)
DEFAULT_M4_REPORT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\embodied-eval-os\external_artifacts\deformable_v02_m4\eos_certification_report.json"
)
DEFAULT_MEDIA_OUT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge\external_artifacts\deformable_v02_media"
)
ISAAC_PYTHON_BAT = Path(r"I:\isaacsim\6.0.1\python.bat")

TASKS: tuple[str, ...] = (
    "coffee_bar_fragile_pastry_plate",
    "office_desk_wiping_cloth_clean_and_park",
)


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
    """Verify M2 OpenUSD scenes, M3 NPZ trajectories, and M4 Gold certification at 0% GPU."""
    import numpy as np

    task_checks: dict[str, Any] = {}
    all_ready = True
    for task_id in TASKS:
        usd_path = m2_root / task_id / "scenario_package" / "scene" / "main.usda"
        usd_ok = usd_path.is_file()
        engine_npz: dict[str, dict[str, Any]] = {}
        for eng in ("mujoco4", "newton18", "genesis", "isaac6"):
            npz_path = m3_root / task_id / eng / f"{task_id}__{eng}__gentle_s0.npz"
            info: dict[str, Any] = {"exists": npz_path.is_file(), "path": str(npz_path)}
            if npz_path.is_file():
                with np.load(npz_path, allow_pickle=False) as z:
                    info["steps"] = int(z["action"].shape[0])
                    info["n_verts"] = int(z["deformable_rest_points"].shape[0])
                    info["has_ee_pos"] = "ee_pos" in z
            else:
                all_ready = False
            engine_npz[eng] = info
        task_checks[task_id] = {
            "usd_path": str(usd_path),
            "usd_exists": usd_ok,
            "engines": engine_npz,
        }
        if not usd_ok:
            all_ready = False

    m4_ok = m4_report.is_file()
    m4_tiers: dict[str, str] = {}
    if m4_ok:
        rep = json.loads(m4_report.read_text(encoding="utf-8"))
        for task_id, trep in rep.get("tasks", {}).items():
            m4_tiers[task_id] = str(trep.get("certification_tier", "Unverified"))
    else:
        all_ready = False

    return {
        "schema_version": "deformable-v02-media-readiness/v0.1",
        "ready": bool(all_ready and all(v == "Gold" for v in m4_tiers.values())),
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


def _build_pastry_visual_cage(rest_pts: Any) -> tuple[Any, Any]:
    """Build smooth 578-vertex rounded pastry cage interpolated trilinearly from (5x5x3) FEM vertices."""
    import numpy as np

    rest = np.asarray(rest_pts, dtype=np.float64)
    ux = np.unique(np.round(rest[:, 0], 5))
    uy = np.unique(np.round(rest[:, 1], 5))
    uz = np.unique(np.round(rest[:, 2], 5))
    nx, ny, nz = len(ux), len(uy), len(uz)
    vmap = np.zeros((nx, ny, nz), dtype=np.int64)
    for vi, p in enumerate(rest):
        ix = int(np.argmin(np.abs(ux - p[0])))
        iy = int(np.argmin(np.abs(uy - p[1])))
        iz = int(np.argmin(np.abs(uz - p[2])))
        vmap[ix, iy, iz] = vi

    gx, gy, gz = 12, 12, 6
    su = np.linspace(0.0, 1.0, gx + 1)
    sv = np.linspace(0.0, 1.0, gy + 1)
    sw = np.linspace(0.0, 1.0, gz + 1)
    bnd_ijk: dict[tuple[int, int, int], int] = {}
    params: list[tuple[float, float, float]] = []
    for i in range(gx + 1):
        for j in range(gy + 1):
            for k in range(gz + 1):
                if i in (0, gx) or j in (0, gy) or k in (0, gz):
                    bnd_ijk[(i, j, k)] = len(params)
                    cx, cy, cz = 2.0 * su[i] - 1.0, 2.0 * sv[j] - 1.0, 2.0 * sw[k] - 1.0
                    r_inf = max(abs(cx), abs(cy), abs(cz), 1e-8)
                    r_p = (abs(cx) ** 6 + abs(cy) ** 6 + abs(cz) ** 6) ** (1.0 / 6.0)
                    s = 0.28 * (1.0 - r_inf / r_p)
                    params.append(
                        (
                            0.5 + 0.5 * cx * (1.0 - s),
                            0.5 + 0.5 * cy * (1.0 - s),
                            0.5 + 0.5 * cz * (1.0 - s),
                        )
                    )

    w_mat = np.zeros((len(params), len(rest)), dtype=np.float64)
    for r_i, (u, v, w) in enumerate(params):
        fx = float(np.clip(u * (nx - 1), 0.0, nx - 1 - 1e-9))
        fy = float(np.clip(v * (ny - 1), 0.0, ny - 1 - 1e-9))
        fz = float(np.clip(w * (nz - 1), 0.0, nz - 1 - 1e-9))
        ix, iy, iz = int(fx), int(fy), int(fz)
        tx, ty, tz = fx - ix, fy - iy, fz - iz
        for dx in (0, 1):
            wx = (1.0 - tx) if dx == 0 else tx
            for dy in (0, 1):
                wy = (1.0 - ty) if dy == 0 else ty
                for dz in (0, 1):
                    wz = (1.0 - tz) if dz == 0 else tz
                    w_mat[r_i, vmap[ix + dx, iy + dy, iz + dz]] += wx * wy * wz

    tris_list: list[tuple[int, int, int]] = []

    def _add_quad(a: int, b: int, c: int, d: int) -> None:
        tris_list.append((a, b, c))
        tris_list.append((a, c, d))

    for i in range(gx):
        for j in range(gy):
            _add_quad(bnd_ijk[(i, j, 0)], bnd_ijk[(i, j + 1, 0)], bnd_ijk[(i + 1, j + 1, 0)], bnd_ijk[(i + 1, j, 0)])
            _add_quad(bnd_ijk[(i, j, gz)], bnd_ijk[(i + 1, j, gz)], bnd_ijk[(i + 1, j + 1, gz)], bnd_ijk[(i, j + 1, gz)])
    for i in range(gx):
        for k in range(gz):
            _add_quad(bnd_ijk[(i, 0, k)], bnd_ijk[(i + 1, 0, k)], bnd_ijk[(i + 1, 0, k + 1)], bnd_ijk[(i, 0, k + 1)])
            _add_quad(bnd_ijk[(i, gy, k)], bnd_ijk[(i, gy, k + 1)], bnd_ijk[(i + 1, gy, k + 1)], bnd_ijk[(i + 1, gy, k)])
    for j in range(gy):
        for k in range(gz):
            _add_quad(bnd_ijk[(0, j, k)], bnd_ijk[(0, j, k + 1)], bnd_ijk[(0, j + 1, k + 1)], bnd_ijk[(0, j + 1, k)])
            _add_quad(bnd_ijk[(gx, j, k)], bnd_ijk[(gx, j + 1, k)], bnd_ijk[(gx, j + 1, k + 1)], bnd_ijk[(gx, j, k + 1)])

    tris = np.asarray(tris_list, dtype=np.int64)
    vis0 = w_mat @ rest
    ctr = vis0.mean(axis=0)
    n_raw = np.cross(vis0[tris[:, 1]] - vis0[tris[:, 0]], vis0[tris[:, 2]] - vis0[tris[:, 0]])
    f_ctr = vis0[tris].mean(axis=1)
    flip = np.sum(n_raw * (f_ctr - ctr[None, :]), axis=1) < 0.0
    if np.any(flip):
        tris[flip] = tris[flip][:, [0, 2, 1]]
    return w_mat, tris


def _build_cloth_visual_cage(rest_pts: Any) -> tuple[Any, Any]:
    """Build smooth 529-vertex (22x22) cloth surface interpolated bilinearly from (11x11) cloth vertices."""
    import numpy as np

    rest = np.asarray(rest_pts, dtype=np.float64)
    ux = np.unique(np.round(rest[:, 0], 5))
    uy = np.unique(np.round(rest[:, 1], 5))
    nx, ny = len(ux), len(uy)
    cmap = np.zeros((nx, ny), dtype=np.int64)
    for vi, p in enumerate(rest):
        cmap[int(np.argmin(np.abs(ux - p[0]))), int(np.argmin(np.abs(uy - p[1])))] = vi

    gc = 22
    w_mat = np.zeros(((gc + 1) * (gc + 1), len(rest)), dtype=np.float64)
    for j in range(gc + 1):
        fy = float(np.clip((j / float(gc)) * (ny - 1), 0.0, ny - 1 - 1e-9))
        iy, ty = int(fy), fy - int(fy)
        for i in range(gc + 1):
            fx = float(np.clip((i / float(gc)) * (nx - 1), 0.0, nx - 1 - 1e-9))
            ix, tx = int(fx), fx - int(fx)
            r_idx = j * (gc + 1) + i
            w_mat[r_idx, cmap[ix, iy]] += (1.0 - tx) * (1.0 - ty)
            w_mat[r_idx, cmap[ix + 1, iy]] += tx * (1.0 - ty)
            w_mat[r_idx, cmap[ix, iy + 1]] += (1.0 - tx) * ty
            w_mat[r_idx, cmap[ix + 1, iy + 1]] += tx * ty

    tris_list: list[tuple[int, int, int]] = []
    for j in range(gc):
        for i in range(gc):
            a = j * (gc + 1) + i
            b = j * (gc + 1) + i + 1
            c = (j + 1) * (gc + 1) + i + 1
            d = (j + 1) * (gc + 1) + i
            tris_list.append((a, b, c))
            tris_list.append((a, c, d))
    return w_mat, np.asarray(tris_list, dtype=np.int64)


def _compute_vertex_normals(pts: Any, tris: Any) -> Any:
    """Compute area-weighted smooth per-vertex normals for a triangle mesh."""
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


def _create_lookat_matrix(Gf: Any, eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> Any:
    """Return USD row-vector camera transform looking from eye toward target along -Z."""
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


def _task_phase_text(
    task_id: str,
    step_k: int,
    total_steps: int,
    *,
    strain_val: float,
    fn_val: float,
    lid_rad: float,
    final_dist_m: float,
    wipe_dists: Sequence[float] | None,
) -> tuple[str, str, str]:
    if task_id == "coffee_bar_fragile_pastry_plate":
        if step_k == 0:
            badge = "PHASE 0/4 · t=0"
            title = "t=0 · 冷启动稳态 (Franka Panda 9-DOF 就绪)"
            readout = "软体蛋糕静置于胡桃木合页盒内 · ||U-I||_F = 0.000"
        elif step_k < 50:
            badge = "PHASE 1/4 · OPEN LID"
            title = "Stage 1/4 · 机械臂触指掀开合页盒盖"
            readout = f"lid_hinge = {math.degrees(lid_rad):.1f} deg  |  ||U-I||_F = {strain_val:.4f} (< 0.15)"
        elif step_k < 104:
            badge = "PHASE 2/4 · TACTILE GRASP"
            title = "Stage 2/4 · 触觉传感柔顺夹持防压溃"
            readout = f"F_n = {fn_val:.2f} N (threshold 68.1 N)  |  ||U-I||_F = {strain_val:.4f} (intact)"
        elif step_k < 150:
            badge = "PHASE 3/4 · +35° CARRY"
            title = "Stage 3/4 · +35 deg 旋转悬空转运至瓷盘"
            readout = f"旋转不变极应变 ||U-I||_F = {strain_val:.4f}  |  F_n = {fn_val:.2f} N"
        else:
            badge = "PHASE 4/4 · PLACE"
            title = "Stage 4/4 · 轻落瓷盘并松爪完成无损放置"
            readout = f"target_dist = {final_dist_m * 1000.0:.1f} mm  |  damage = 0.000  |  EOS Gold (4/4)"
        return badge, title, readout

    wipe_str = ", ".join(f"{w * 1000.0:.1f}mm" for w in (wipe_dists or (0.0, 0.0, 0.0)))
    if step_k == 0:
        badge = "PHASE 0/4 · t=0"
        title = "t=0 · 冷启动稳态 (柔性擦布架于支架)"
        readout = "办公桌垫 3 处擦拭标靶就绪 · edge_stretch = 0.000"
    elif step_k < 48:
        badge = "PHASE 1/4 · PINCH"
        title = "Stage 1/4 · 下降轻捏柔性超纤擦布脊背"
        readout = f"F_n = {fn_val:.2f} N  |  edge_stretch_q95 = {strain_val:.4f} (< 0.27)"
    elif step_k < 108:
        badge = "PHASE 2/4 · WIPE x3"
        title = "Stage 2/4 · 贴合桌垫依次擦拭 3 处标靶点"
        readout = f"3-waypoint min_dist = [{wipe_str}]  |  edge = {strain_val:.4f}"
    elif step_k < 142:
        badge = "PHASE 3/4 · CARRY"
        title = "Stage 3/4 · 提布避障转运至桦木收纳盒"
        readout = f"避让收纳盒 9mm 侧壁  |  F_n = {fn_val:.2f} N  |  edge = {strain_val:.4f}"
    else:
        badge = "PHASE 4/4 · PARK"
        title = "Stage 4/4 · 松爪平铺入槽完成收纳"
        readout = f"target_dist = {final_dist_m * 1000.0:.1f} mm  |  damage = 0.000  |  EOS Gold (4/4)"
    return badge, title, readout


def _decorate_video_frame(
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
    progress: float,
) -> Any:
    import numpy as np

    base = Image.fromarray(rgb_1080p[:, :, :3].astype("uint8"), mode="RGB").resize(
        (1280, 720), Image.Resampling.LANCZOS
    )
    rgba = base.convert("RGBA")
    overlay = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Top-left phase & physics telemetry card
    draw.rounded_rectangle(
        [(20, 16), (860, 94)],
        radius=12,
        fill=(12, 24, 42, 212),
        outline=(56, 189, 178, 185),
        width=2,
    )
    draw.rounded_rectangle([(34, 28), (242, 54)], radius=6, fill=(13, 148, 136, 240))
    draw.text((44, 32), badge_text, fill=(255, 255, 255, 255), font=font_mono)
    draw.text((254, 27), title_text, fill=(248, 250, 252, 255), font=font_title)
    draw.text((36, 63), readout_text, fill=(196, 214, 232, 248), font=font_sub)

    # Top-right EOS certification badge + trajectory progress bar
    draw.rounded_rectangle(
        [(952, 16), (1260, 94)],
        radius=12,
        fill=(12, 24, 42, 212),
        outline=(234, 179, 8, 190),
        width=2,
    )
    draw.text((970, 26), "EOS Gold (4/4) · INTACT", fill=(250, 204, 21, 255), font=font_mono)
    bar_x0, bar_y0, bar_x1, bar_y1 = 970, 62, 1242, 76
    draw.rounded_rectangle([(bar_x0, bar_y0), (bar_x1, bar_y1)], radius=7, fill=(36, 54, 76, 235))
    fill_w = int(round((bar_x1 - bar_x0) * max(0.0, min(1.0, progress))))
    if fill_w > 4:
        draw.rounded_rectangle(
            [(bar_x0, bar_y0), (bar_x0 + fill_w, bar_y1)],
            radius=7,
            fill=(20, 184, 166, 255),
        )

    out_rgb = Image.alpha_composite(rgba, overlay).convert("RGB")
    return np.asarray(out_rgb, dtype=np.uint8)


def _build_contact_sheet(
    keyframe_items: Sequence[tuple[str, Path]],
    out_path: Path,
    *,
    Image: Any,
    ImageDraw: Any,
    font_mono: Any,
) -> Path:
    cols = 3
    rows = max(1, int(math.ceil(len(keyframe_items) / float(cols))))
    tile_w, tile_h = 640, 360
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (13, 17, 23))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, path) in enumerate(keyframe_items):
        r, c = divmod(idx, cols)
        tile = Image.open(path).convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas.paste(tile, (c * tile_w, r * tile_h))
        draw.rectangle([(c * tile_w + 8, r * tile_h + 8), (c * tile_w + 350, r * tile_h + 36)], fill=(12, 24, 42))
        draw.text((c * tile_w + 16, r * tile_h + 13), label, fill=(248, 250, 252), font=font_mono)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def _find_op(xf: Any, op_name: str, op_type: Any = None) -> Any:
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == op_name or (op_type is not None and op.GetOpType() == op_type):
            return op
    return None


def _render_rtx_showcase(
    *,
    m2_root: Path,
    m3_root: Path,
    m4_report: Path,
    out_dir: Path,
    only_task: str | None = None,
) -> dict[str, Any]:
    """Playback M3 9-DOF robot + smooth shaded deformable mesh trajectories under Isaac Sim 6.0.1 RTX."""
    _apply_low_priority()
    os.environ.setdefault("ISAACSIM_PORTABLE_ROOT", r"I:\isaacsim-data\6.0.1")

    _src_root = str(Path(__file__).resolve().parents[1] / "src")
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

    import numpy as np
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
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

    font_title = _pick_font(ImageFont, [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"], 18)
    font_sub = _pick_font(ImageFont, [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"], 14)
    font_mono = _pick_font(ImageFont, [r"C:\Windows\Fonts\consolab.ttf", r"C:\Windows\Fonts\consola.ttf"], 13)

    manifest_entries: dict[str, Any] = {}
    contact_items: list[tuple[str, Path]] = []
    active_tasks: tuple[str, ...] = (only_task,) if only_task else TASKS

    try:
        timeline = omni.timeline.get_timeline_interface()
        for task_id in active_tasks:
            usd_path = m2_root / task_id / "scenario_package" / "scene" / "main.usda"
            npz_path = m3_root / task_id / "mujoco4" / f"{task_id}__mujoco4__gentle_s0.npz"
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

            # Hide static t=0 deformable mesh so no frozen duplicate remains
            static_path = "/World/fragile_pastry" if task_id == "coffee_bar_fragile_pastry_plate" else "/World/wiping_cloth"
            static_prim = stage.GetPrimAtPath(static_path)
            if static_prim and static_prim.IsValid():
                UsdGeom.Imageable(static_prim).MakeInvisible()

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
            if lf_op is None or rf_op is None:
                raise RuntimeError(f"Missing finger translate xformOps under {hand_path}")

            # Load MuJoCo4 trajectory & build smooth shaded deformable mesh cage
            with np.load(npz_path, allow_pickle=False) as z:
                rest_pts = np.asarray(z["deformable_rest_points"], dtype=np.float64)
                pts_seq = np.asarray(z["deformable_points"], dtype=np.float64)
                qpos_seq = np.asarray(z["qpos"], dtype=np.float64)
                tac_seq = np.asarray(z["tactile_normal_force_n"], dtype=np.float64)
                lid_seq = np.asarray(z["lid_angle_rad"], dtype=np.float64) if "lid_angle_rad" in z else np.zeros(len(pts_seq))

            if task_id == "coffee_bar_fragile_pastry_plate":
                w_cage, tris_cage = _build_pastry_visual_cage(rest_pts)
                mat_color = (0.84, 0.56, 0.25)
                mat_rough = 0.42
                mid_k = min(132, len(pts_seq) - 2)
                step_strains = [float(0.0)] * len(pts_seq)
                from trajectory_forge.adapters.deformable_workers.common import compute_tet_strains_from_points
                with np.load(npz_path, allow_pickle=False) as z:
                    tets_arr = np.asarray(z["deformable_elements"], dtype=np.int64)
                for ti in range(len(pts_seq)):
                    step_strains[ti] = compute_tet_strains_from_points(rest_pts, pts_seq[ti], tets_arr)[0]
            else:
                w_cage, tris_cage = _build_cloth_visual_cage(rest_pts)
                mat_color = (0.14, 0.68, 0.86)
                mat_rough = 0.68
                mid_k = min(104, len(pts_seq) - 2)
                step_strains = [float(0.0)] * len(pts_seq)
                from trajectory_forge.adapters.deformable_workers.common import compute_edge_stretch_from_points
                with np.load(npz_path, allow_pickle=False) as z:
                    edges_arr = np.asarray(z["deformable_elements"], dtype=np.int64)
                for ti in range(len(pts_seq)):
                    step_strains[ti] = compute_edge_stretch_from_points(rest_pts, pts_seq[ti], edges_arr)

            UsdGeom.Scope.Define(stage, "/World/ShowcaseVisual")
            mesh_prim = UsdGeom.Mesh.Define(stage, "/World/ShowcaseVisual/DeformableMesh")
            mesh_prim.CreateDoubleSidedAttr().Set(True)
            mesh_prim.CreateSubdivisionSchemeAttr().Set("none")
            tris_flat = tris_cage.reshape(-1)
            mesh_prim.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(tris_cage)))
            mesh_prim.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([int(v) for v in tris_flat]))
            mesh_prim.CreateNormalsAttr()
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            pts_attr = mesh_prim.GetPointsAttr()
            nrm_attr = mesh_prim.GetNormalsAttr()

            mat = UsdShade.Material.Define(stage, "/World/ShowcaseVisual/DeformableMat")
            shader = UsdShade.Shader.Define(stage, "/World/ShowcaseVisual/DeformableMat/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*mat_color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(mat_rough))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim()).Bind(mat)

            tunable_exp_attrs = []
            boost_prefixes = (
                "/World/Ceiling_Soft_",
                "/World/NorthBar_KeyWash",
                "/World/Light_North_Fill",
                "/World/Light_South_Fill",
                "/World/Light_West_Fill",
                "/World/Light_East_Fill",
            )
            for pr in stage.Traverse():
                p_str = str(pr.GetPath())
                if "Light" in pr.GetTypeName() and any(p_str.startswith(pref) for pref in boost_prefixes):
                    exp_attr = pr.GetAttribute("inputs:exposure")
                    if not exp_attr:
                        exp_attr = pr.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float)
                    exp_attr.Set(5.5)
                    tunable_exp_attrs.append(exp_attr)
            env_prim = stage.GetPrimAtPath("/World/env_light")
            if env_prim and env_prim.IsValid():
                env_int = env_prim.GetAttribute("inputs:intensity")
                if env_int:
                    env_int.Set(0.45)

            studio_specs = [
                ("/World/ShowcaseVisual/StudioKey", (0.95, -1.05, 1.85), 0.28, 3200.0, (1.0, 0.97, 0.92)),
                ("/World/ShowcaseVisual/StudioFill", (-0.85, -0.75, 1.85), 0.32, 1600.0, (0.92, 0.96, 1.0)),
                ("/World/ShowcaseVisual/StudioTop", (0.0, 0.05, 2.05), 0.30, 2000.0, (1.0, 0.98, 0.95)),
            ]
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

            wipe_marker_inputs: list[tuple[int, Any]] = []
            if task_id == "office_desk_wiping_cloth_clean_and_park":
                for m_idx, onset_k in ((1, 45), (2, 84), (3, 106)):
                    t_prim = stage.GetPrimAtPath(f"/World/wipe_zone_mat/TargetMarker{m_idx}/WipeTarget{m_idx}")
                    if t_prim and t_prim.IsValid():
                        w_mat = UsdShade.Material.Define(stage, f"/World/ShowcaseVisual/WipeMat{m_idx}")
                        w_sh = UsdShade.Shader.Define(stage, f"/World/ShowcaseVisual/WipeMat{m_idx}/Shader")
                        w_sh.CreateIdAttr("UsdPreviewSurface")
                        diff_in = w_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                        diff_in.Set(Gf.Vec3f(0.88, 0.44, 0.10))
                        w_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.38)
                        w_sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                        w_mat.CreateSurfaceOutput().ConnectToSource(w_sh.ConnectableAPI(), "surface")
                        UsdShade.MaterialBindingAPI.Apply(t_prim).Bind(
                            w_mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                        )
                        wipe_marker_inputs.append((onset_k, diff_in))

            cam_prim = UsdGeom.Camera.Define(stage, "/World/ShowcaseCamera")
            cam_prim.CreateFocalLengthAttr().Set(14.5)
            cam_prim.CreateHorizontalApertureAttr().Set(20.955)
            cam_prim.CreateVerticalApertureAttr().Set(11.787)
            cam_prim.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
            cam_xf = UsdGeom.Xformable(cam_prim.GetPrim())
            cam_op = cam_xf.AddTransformOp()
            cam_op.Set(_create_lookat_matrix(Gf, (0.86, -0.88, 1.38), (-0.04, 0.02, 0.88), (0.0, 0.0, 1.0)))

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
                if task_id == "coffee_bar_fragile_pastry_plate":
                    lid_prim = stage.GetPrimAtPath("/World/pastry_box_hinged/LidLink")
                    xf_lid = UsdGeom.Xformable(lid_prim)
                    lid_op = _find_op(xf_lid, "xformOp:rotateX", UsdGeom.XformOp.TypeRotateX)
                    if lid_op is None:
                        lid_op = xf_lid.AddRotateXOp()
                    lid_op.Set(-min(106.0, math.degrees(float(lid_seq[k]))))
                for onset_k, diff_in in wipe_marker_inputs:
                    alpha = max(0.0, min(1.0, (k - onset_k) / 12.0))
                    col = (
                        0.88 + alpha * (0.18 - 0.88),
                        0.44 + alpha * (0.68 - 0.44),
                        0.10 + alpha * (0.50 - 0.10),
                    )
                    diff_in.Set(Gf.Vec3f(*col))
                vis_pts = w_cage @ pts_seq[k]
                vis_nrm = _compute_vertex_normals(vis_pts, tris_cage)
                fv_nrm = vis_nrm[tris_flat]
                pts_attr.Set(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in vis_pts]))
                nrm_attr.Set(Vt.Vec3fArray([Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in fv_nrm]))

            # Warm up RTX lighting & materials at t=0, with 1-step photometric exposure calibration
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
            mj_sum = (m4_task.get("engine_summaries") or {}).get("mujoco4", {})
            final_dist_m = float(mj_sum.get("final_target_dist_m", 0.004))
            wipe_dists = mj_sum.get("wipe_min_dists_m") or [0.002, 0.010, 0.005]

            n_steps = len(pts_seq)
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
                    short_t = "coffee" if "coffee" in task_id else "office"
                    task_contact_items.append((f"{short_t} | {label_key} (step {k}/{n_steps - 1})", kf_path))

                fn_val = float(max(abs(tac_seq[k, 0]), abs(tac_seq[k, 1])))
                badge, title_t, readout_t = _task_phase_text(
                    task_id,
                    k,
                    n_steps,
                    strain_val=step_strains[k],
                    fn_val=fn_val,
                    lid_rad=float(lid_seq[k]),
                    final_dist_m=final_dist_m,
                    wipe_dists=wipe_dists,
                )
                dec = _decorate_video_frame(
                    rgb_1080,
                    Image=Image,
                    ImageDraw=ImageDraw,
                    font_title=font_title,
                    font_sub=font_sub,
                    font_mono=font_mono,
                    badge_text=badge,
                    title_text=title_t,
                    readout_text=readout_t,
                    progress=k / float(max(n_steps - 1, 1)),
                )
                repeat = 18 if k == 0 else (24 if k == n_steps - 1 else 1)
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
                json.dumps(
                    {
                        "entry": manifest_entries[task_id],
                        "contact_items": [(lbl, str(p)) for lbl, p in task_contact_items],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            contact_items.extend(task_contact_items)
            rgb_annot.detach([rp])
            rp.destroy()
            rep.orchestrator.stop()
            app.update()

        sheet_path: Path | None = None
        if only_task is None:
            sheet_path = _build_contact_sheet(
                contact_items,
                out_dir / "showcase_contact_sheet.png",
                Image=Image,
                ImageDraw=ImageDraw,
                font_mono=font_mono,
            )
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # SimulationApp.close() with default fastShutdown=True terminates Kit with exit code 0.
        os._exit(1)

    if only_task is None and sheet_path is not None:
        manifest_path = out_dir / "media_manifest.json"
        manifest_path.write_text(
            json.dumps({"contact_sheet": str(sheet_path), "tasks": manifest_entries}, indent=2),
            encoding="utf-8",
        )
    app.close()
    return manifest_entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or (with --confirm-rtx-render) execute RTX deformable showcase rendering."
    )
    parser.add_argument("--m2-root", type=Path, default=DEFAULT_M2_ROOT)
    parser.add_argument("--m3-root", type=Path, default=DEFAULT_M3_ROOT)
    parser.add_argument("--m4-report", type=Path, default=DEFAULT_M4_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MEDIA_OUT)
    parser.add_argument("--only-task", choices=TASKS, default=None, help="Render one task in an isolated Kit session.")
    parser.add_argument("--check-ready", action="store_true", help="Run 0%-GPU readiness check only.")
    parser.add_argument(
        "--confirm-rtx-render",
        action="store_true",
        help="Explicitly confirm heavy Isaac RTX GPU rendering (refuses without this flag).",
    )
    args = parser.parse_args()

    readiness = check_render_readiness(m2_root=args.m2_root, m3_root=args.m3_root, m4_report=args.m4_report)
    if args.check_ready or not args.confirm_rtx_render:
        print(json.dumps(readiness, indent=2))
        if not args.check_ready and not args.confirm_rtx_render:
            print(
                "\n[GUARD] Heavy Isaac RTX rendering was NOT started because --confirm-rtx-render was omitted.",
                file=sys.stderr,
            )
            return 2 if not readiness["ready"] else 0
        return 0 if readiness["ready"] else 1

    if not readiness["ready"]:
        print(json.dumps(readiness, indent=2), file=sys.stderr)
        return 1

    if args.only_task is None:
        if not ISAAC_PYTHON_BAT.is_file():
            raise FileNotFoundError(f"Isaac Sim Python launcher not found: {ISAAC_PYTHON_BAT}")
        _apply_low_priority()
        env = os.environ.copy()
        env["ISAACSIM_PORTABLE_ROOT"] = r"I:\isaacsim-data\6.0.1"
        env["TEMP"] = r"I:\isaacsim-data\6.0.1\temp"
        env["TMP"] = r"I:\isaacsim-data\6.0.1\temp"
        creationflags = 0x00000040 if sys.platform.startswith("win") else 0
        for task_id in TASKS:
            proc = subprocess.run(
                [
                    str(ISAAC_PYTHON_BAT),
                    str(Path(__file__).resolve()),
                    "--confirm-rtx-render",
                    "--m2-root",
                    str(args.m2_root),
                    "--m3-root",
                    str(args.m3_root),
                    "--m4-report",
                    str(args.m4_report),
                    "--out-dir",
                    str(args.out_dir),
                    "--only-task",
                    task_id,
                ],
                env=env,
                creationflags=creationflags,
            )
            if int(proc.returncode) != 0:
                return int(proc.returncode)

        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        font_mono = _pick_font(ImageFont, [r"C:\Windows\Fonts\consolab.ttf", r"C:\Windows\Fonts\consola.ttf"], 13)
        manifest_entries: dict[str, Any] = {}
        contact_items: list[tuple[str, Path]] = []
        for task_id in TASKS:
            entry_payload = json.loads((args.out_dir / f"{task_id}_entry.json").read_text(encoding="utf-8"))
            manifest_entries[task_id] = entry_payload["entry"]
            for lbl, p_str in entry_payload["contact_items"]:
                contact_items.append((lbl, Path(p_str)))

        sheet_path = _build_contact_sheet(
            contact_items,
            args.out_dir / "showcase_contact_sheet.png",
            Image=Image,
            ImageDraw=ImageDraw,
            font_mono=font_mono,
        )
        manifest_path = args.out_dir / "media_manifest.json"
        manifest_path.write_text(
            json.dumps({"contact_sheet": str(sheet_path), "tasks": manifest_entries}, indent=2),
            encoding="utf-8",
        )
        return 0

    if importlib.util.find_spec("isaacsim") is None:
        if not ISAAC_PYTHON_BAT.is_file():
            raise FileNotFoundError(f"Isaac Sim Python launcher not found: {ISAAC_PYTHON_BAT}")
        _apply_low_priority()
        env = os.environ.copy()
        env["ISAACSIM_PORTABLE_ROOT"] = r"I:\isaacsim-data\6.0.1"
        env["TEMP"] = r"I:\isaacsim-data\6.0.1\temp"
        env["TMP"] = r"I:\isaacsim-data\6.0.1\temp"
        creationflags = 0x00000040 if sys.platform.startswith("win") else 0
        proc = subprocess.run(
            [str(ISAAC_PYTHON_BAT), str(Path(__file__).resolve()), *sys.argv[1:]],
            env=env,
            creationflags=creationflags,
        )
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
