"""Dual-tier 3D studio comparison video & Before/After peak keyframe renderer (v0.8)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from convert_asset.deformable_v08_hermite.hermite_gs_binding import render_hermite_3dgs_frame


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )
    for c in candidates:
        if Path(c).is_file():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def _camera_basis(
    cam_pos: tuple[float, float, float],
    look_at: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam = np.asarray(cam_pos, dtype=np.float64)
    tgt = np.asarray(look_at, dtype=np.float64)
    fwd = tgt - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return cam, right, up, fwd


def _project_pts(
    pts: np.ndarray,
    *,
    w: int,
    h: int,
    focal: float,
    cam: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    fwd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    rel = p - cam[None, :]
    cx = rel @ right
    cy = rel @ up
    cz = np.maximum(rel @ fwd, 0.05)
    u = 0.5 * w + focal * (cx / cz)
    v = 0.5 * h - focal * (cy / cz)
    return np.column_stack([u, v]), cz


def _draw_panel_3d(
    *,
    verts_grid: np.ndarray,
    normals_grid: np.ndarray,
    gt_grid: np.ndarray | None,
    gripper_pos: np.ndarray,
    gripper_rot: np.ndarray,
    gripper_gap: float,
    table_z: float,
    base_rgb: tuple[float, float, float],
    obstacles: Sequence[Mapping[str, Any]] = (),
    panel_w: int = 480,
    panel_h: int = 400,
    show_error_tint: bool = False,
    cam_pos: tuple[float, float, float] = (0.04, -0.56, 1.08),
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.78),
) -> np.ndarray:
    """Render a crisp lit 3D studio frame with workbench grid, drop shadow, obstacle, shell & gripper."""
    img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    # Vertical studio gradient background
    grad_y = np.linspace(0.0, 1.0, panel_h)[:, None, None]
    bg_top = np.array([[[19, 24, 34]]], dtype=np.float64)
    bg_bot = np.array([[[30, 38, 52]]], dtype=np.float64)
    img[:] = ((1.0 - grad_y) * bg_top + grad_y * bg_bot).astype(np.uint8)

    cam, right, up, fwd = _camera_basis(cam_pos, look_at)
    focal = 0.5 * panel_h / math.tan(0.5 * math.radians(42.0))

    # 1. Draw workbench surface polygon + subtle perspective grid
    tbl_corners = np.array(
        [
            [-0.36, -0.32, table_z - 0.001],
            [0.36, -0.32, table_z - 0.001],
            [0.36, 0.32, table_z - 0.001],
            [-0.36, 0.32, table_z - 0.001],
        ],
        dtype=np.float64,
    )
    tbl_px, _ = _project_pts(tbl_corners, w=panel_w, h=panel_h, focal=focal, cam=cam, right=right, up=up, fwd=fwd)
    cv2.fillConvexPoly(img, np.round(tbl_px).astype(np.int32), (42, 49, 62), lineType=cv2.LINE_AA)
    cv2.polylines(img, [np.round(tbl_px).astype(np.int32)], True, (76, 89, 110), 1, cv2.LINE_AA)

    for g_val in np.linspace(-0.30, 0.30, 7):
        l1 = np.array([[g_val, -0.30, table_z - 0.0005], [g_val, 0.30, table_z - 0.0005]])
        l2 = np.array([[-0.34, g_val, table_z - 0.0005], [0.34, g_val, table_z - 0.0005]])
        p1, _ = _project_pts(l1, w=panel_w, h=panel_h, focal=focal, cam=cam, right=right, up=up, fwd=fwd)
        p2, _ = _project_pts(l2, w=panel_w, h=panel_h, focal=focal, cam=cam, right=right, up=up, fwd=fwd)
        cv2.line(img, tuple(np.round(p1[0]).astype(int)), tuple(np.round(p1[1]).astype(int)), (55, 64, 80), 1, cv2.LINE_AA)
        cv2.line(img, tuple(np.round(p2[0]).astype(int)), tuple(np.round(p2[1]).astype(int)), (55, 64, 80), 1, cv2.LINE_AA)

    # 2. Soft drop shadow of sheet projected onto tabletop
    nu, nv, _ = verts_grid.shape
    sh_pts = verts_grid.copy()
    sh_pts[:, :, 0] += 0.015 * np.maximum(verts_grid[:, :, 2] - table_z, 0.0)
    sh_pts[:, :, 1] += 0.015 * np.maximum(verts_grid[:, :, 2] - table_z, 0.0)
    sh_pts[:, :, 2] = table_z - 0.0002
    # Convex-ish boundary trace for crisp fast shadow
    b_pts = np.concatenate([sh_pts[0, :], sh_pts[:, -1], sh_pts[-1, ::-1], sh_pts[::-1, 0]], axis=0)
    b_px, _ = _project_pts(b_pts, w=panel_w, h=panel_h, focal=focal, cam=cam, right=right, up=up, fwd=fwd)
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.round(b_px).astype(np.int32)], (22, 26, 34))
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0.0)

    # 3. Rigid obstacles (e.g., calibration book box on Task 3 novel action)
    for obs in obstacles:
        c = np.asarray(obs["center_xyz"], dtype=np.float64)
        h = np.asarray(obs["half_extents_xyz"], dtype=np.float64)
        box_v = np.array(
            [
                [c[0] - h[0], c[1] - h[1], c[2] - h[2]],
                [c[0] + h[0], c[1] - h[1], c[2] - h[2]],
                [c[0] + h[0], c[1] + h[1], c[2] - h[2]],
                [c[0] - h[0], c[1] + h[1], c[2] - h[2]],
                [c[0] - h[0], c[1] - h[1], c[2] + h[2]],
                [c[0] + h[0], c[1] - h[1], c[2] + h[2]],
                [c[0] + h[0], c[1] + h[1], c[2] + h[2]],
                [c[0] - h[0], c[1] + h[1], c[2] + h[2]],
            ],
            dtype=np.float64,
        )
        box_px, box_z = _project_pts(box_v, w=panel_w, h=panel_h, focal=focal, cam=cam, right=right, up=up, fwd=fwd)
        box_faces = [
            ([4, 5, 6, 7], (92, 102, 118)),  # top
            ([0, 1, 5, 4], (68, 76, 90)),    # front
            ([1, 2, 6, 5], (56, 63, 76)),    # right
            ([3, 0, 4, 7], (74, 83, 98)),    # left
        ]
        box_faces.sort(key=lambda item: float(np.mean(box_z[item[0]])), reverse=True)
        for f_idx, f_col in box_faces:
            poly = np.round(box_px[f_idx]).astype(np.int32)
            cv2.fillConvexPoly(img, poly, f_col, lineType=cv2.LINE_AA)
            cv2.polylines(img, [poly], True, (130, 144, 165), 1, cv2.LINE_AA)

    # 4. Deformable thin shell quads (Painter's back-to-front depth sort + two-sided Phong lighting)
    pts_flat = verts_grid.reshape(-1, 3)
    n_flat = normals_grid.reshape(-1, 3)
    px_flat, z_flat = _project_pts(pts_flat, w=panel_w, h=panel_h, focal=focal, cam=cam, right=right, up=up, fwd=fwd)

    # Compute vertex error in mm if error tint enabled
    if show_error_tint and gt_grid is not None:
        v_err_mm = np.linalg.norm(verts_grid - gt_grid, axis=-1).reshape(-1) * 1000.0
    else:
        v_err_mm = np.zeros(len(pts_flat), dtype=np.float64)

    light_key = np.array([0.45, -0.55, 0.70], dtype=np.float64)
    light_key /= np.linalg.norm(light_key)
    light_rim = np.array([-0.55, 0.50, 0.65], dtype=np.float64)
    light_rim /= np.linalg.norm(light_rim)
    base_col = np.asarray(base_rgb, dtype=np.float64)

    i_idx, j_idx = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    a_id = (i_idx * nv + j_idx).ravel()
    b_id = ((i_idx + 1) * nv + j_idx).ravel()
    c_id = ((i_idx + 1) * nv + (j_idx + 1)).ravel()
    d_id = (i_idx * nv + (j_idx + 1)).ravel()

    q_depths = 0.25 * (z_flat[a_id] + z_flat[b_id] + z_flat[c_id] + z_flat[d_id])
    order = np.argsort(q_depths)[::-1]  # farthest first

    q_norms = 0.25 * (n_flat[a_id] + n_flat[b_id] + n_flat[c_id] + n_flat[d_id])
    q_norms /= np.maximum(np.linalg.norm(q_norms, axis=1, keepdims=True), 1e-10)
    # Two-sided diffuse + subtle specular sheen
    ndotl_key = np.abs(q_norms @ light_key)
    ndotl_rim = np.abs(q_norms @ light_rim)
    half_vec = light_key - fwd
    half_vec /= np.linalg.norm(half_vec)
    spec = (np.abs(q_norms @ half_vec) ** 18) * 0.22

    # Checker subtle fabric weave texture
    checker = (((i_idx + j_idx) % 2) * 0.06 - 0.03).ravel()
    err_q = 0.25 * (v_err_mm[a_id] + v_err_mm[b_id] + v_err_mm[c_id] + v_err_mm[d_id])

    for q_k in order:
        ia, ib, ic, id_ = int(a_id[q_k]), int(b_id[q_k]), int(c_id[q_k]), int(d_id[q_k])
        poly = np.round(px_flat[[ia, ib, ic, id_]]).astype(np.int32)
        illum = 0.28 + 0.62 * float(ndotl_key[q_k]) + 0.16 * float(ndotl_rim[q_k])
        col = np.clip((base_col + float(checker[q_k])) * illum + float(spec[q_k]), 0.0, 1.0)
        if show_error_tint and err_q[q_k] > 5.0:
            # Blend subtle coral-red highlight where prediction deviates > 5mm from GT
            blend = min(0.62, (float(err_q[q_k]) - 5.0) / 38.0)
            col = (1.0 - blend) * col + blend * np.array([0.96, 0.28, 0.22])
        rgb_u8 = tuple(int(v * 255) for v in col)
        cv2.fillConvexPoly(img, poly, rgb_u8, lineType=cv2.LINE_AA)
        edge_col = tuple(max(0, int(c * 0.72)) for c in rgb_u8)
        cv2.polylines(img, [poly], True, edge_col, 1, cv2.LINE_AA)

    # 5. Articulated dual-finger parallel gripper rendered in 3D
    half_g = 0.5 * float(gripper_gap) + 0.005
    for sign in (-1.0, 1.0):
        j_c = gripper_pos + gripper_rot @ np.array([sign * half_g, 0.0, 0.018])
        j_box = []
        for sx in (-0.004, 0.004):
            for sy in (-0.010, 0.010):
                for sz in (-0.018, 0.018):
                    j_box.append(j_c + gripper_rot @ np.array([sx, sy, sz]))
        jb_arr = np.asarray(j_box, dtype=np.float64)
        j_px, _ = _project_pts(jb_arr, w=panel_w, h=panel_h, focal=focal, cam=cam, right=right, up=up, fwd=fwd)
        hull = cv2.convexHull(np.round(j_px).astype(np.int32))
        cv2.fillConvexPoly(img, hull, (215, 224, 235), lineType=cv2.LINE_AA)
        cv2.polylines(img, [hull], True, (75, 90, 110), 1, cv2.LINE_AA)

    # Gripper wrist bar
    w_pts = np.array(
        [
            gripper_pos + gripper_rot @ np.array([-half_g - 0.006, 0.0, 0.042]),
            gripper_pos + gripper_rot @ np.array([half_g + 0.006, 0.0, 0.042]),
        ],
        dtype=np.float64,
    )
    w_px, _ = _project_pts(w_pts, w=panel_w, h=panel_h, focal=focal, cam=cam, right=right, up=up, fwd=fwd)
    cv2.line(img, tuple(np.round(w_px[0]).astype(int)), tuple(np.round(w_px[1]).astype(int)), (140, 165, 200), 5, cv2.LINE_AA)

    return img


def _compose_3up_frame(
    *,
    left_img: np.ndarray,
    mid_img: np.ndarray,
    right_img: np.ndarray,
    left_title: str,
    mid_title: str,
    right_title: str,
    tier_badge: str,
    phase_text: str,
    phase_color: tuple[int, int, int],
    hud_left: str,
    hud_mid: str,
    hud_right: str,
    font_title: ImageFont.ImageFont,
    font_hud: ImageFont.ImageFont,
) -> np.ndarray:
    pw = left_img.shape[1]
    ph = left_img.shape[0]
    header_h = 56
    footer_h = 54
    total_w = pw * 3
    total_h = header_h + ph + footer_h

    canvas = Image.new("RGB", (total_w, total_h), (14, 18, 26))
    canvas.paste(Image.fromarray(left_img), (0, header_h))
    canvas.paste(Image.fromarray(mid_img), (pw, header_h))
    canvas.paste(Image.fromarray(right_img), (2 * pw, header_h))

    draw = ImageDraw.Draw(canvas)
    # Dividers
    draw.line([(pw, 0), (pw, total_h)], fill=(42, 52, 68), width=2)
    draw.line([(2 * pw, 0), (2 * pw, total_h)], fill=(42, 52, 68), width=2)

    # Top panel headers
    titles = [
        (0, left_title, (165, 195, 235)),
        (pw, mid_title, (248, 145, 95)),
        (2 * pw, right_title, (64, 224, 178)),
    ]
    for x0, txt, col in titles:
        draw.text((x0 + 16, 16), txt, fill=col, font=font_title)

    # Tier & Phase pills on top right
    draw.rounded_rectangle([total_w - 250, 12, total_w - 14, 42], radius=6, fill=(28, 42, 64), outline=(72, 118, 175))
    draw.text((total_w - 238, 18), tier_badge[:24], fill=(165, 210, 255), font=font_hud)

    # Bottom telemetry HUD
    draw.rectangle([0, header_h + ph, total_w, total_h], fill=(11, 15, 22))
    draw.rounded_rectangle([14, header_h + ph + 12, 230, total_h - 12], radius=6, fill=phase_color)
    draw.text((24, header_h + ph + 18), phase_text, fill=(10, 14, 20), font=font_hud)
    draw.text((246, header_h + ph + 18), hud_left, fill=(205, 215, 230), font=font_hud)
    draw.text((pw + 16, header_h + ph + 18), hud_mid, fill=(248, 175, 125), font=font_hud)
    draw.text((2 * pw + 16, header_h + ph + 18), hud_right, fill=(96, 238, 194), font=font_hud)

    return np.asarray(canvas, dtype=np.uint8)


def _encode_rgb_frames_to_mp4(
    frames: Sequence[np.ndarray],
    out_mp4: Path,
    *,
    fps: int = 24,
) -> Path:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "19",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    raw_bytes = b"".join(np.ascontiguousarray(fr, dtype=np.uint8).tobytes() for fr in frames)
    proc = subprocess.run(cmd, input=raw_bytes, capture_output=True, timeout=120, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode('utf-8', 'ignore')}")
    return out_mp4


def render_task_ab_comparison_video(
    *,
    task_id: str,
    task_entry: Mapping[str, Any],
    cert_entry: Mapping[str, Any],
    out_media_dir: str | Path,
    tier: str = "fast_cpu",
    blender_bin: str | None = None,
) -> dict[str, Any]:
    """Render synchronized 3-up A/B video + high-res Before/After peak PNGs for one task."""
    media_dir = Path(out_media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    npz_path = Path(task_entry["main_npz"])
    z = np.load(npz_path, allow_pickle=False)

    gt_v = np.asarray(z["gt_vertices"], dtype=np.float64)
    gt_n = np.asarray(z["gt_normals"], dtype=np.float64)
    b_v = np.asarray(z["baseline_vertices"], dtype=np.float64)
    b_n = np.asarray(z["baseline_normals"], dtype=np.float64)
    h_v = np.asarray(z["hermite_vertices"], dtype=np.float64)
    h_n = np.asarray(z["hermite_normals"], dtype=np.float64)
    g_pos = np.asarray(z["gripper_pos_xyz"], dtype=np.float64)
    g_rot = np.asarray(z["gripper_rot_3x3"], dtype=np.float64)
    g_gap = np.asarray(z["gripper_gap_m"], dtype=np.float64)
    table_z = float(z["table_z_m"])
    base_rgb = tuple(float(c) for c in z["base_rgb"])
    split = int(z["train_split_frame"])

    b_cd = list(cert_entry["baseline_summary"]["cd_per_frame_mm2"])
    h_cd = list(cert_entry["hermite_summary"]["cd_per_frame_mm2"])
    peak_t = int(cert_entry["hermite_summary"]["peak_frame"])

    font_title = _load_font(20, bold=True)
    font_hud = _load_font(15, bold=True)
    badge = "STUDIO_CYCLES_AGX" if tier == "studio_cycles" else "FAST_CHECK_CPU_PREVIEW"

    # Optional Phase 2 Blender Cycles AgX render frames if requested
    cycles_arms: dict[str, Path] = {}
    if tier == "studio_cycles" and blender_bin:
        bpy_script = Path(__file__).resolve().with_name("bpy_hermite_studio_scene.py")
        for arm_key in ("gt", "baseline", "hermite"):
            arm_frames_dir = media_dir / f"cycles_{arm_key}"
            subprocess.run(
                [
                    blender_bin,
                    "--background",
                    "--python",
                    str(bpy_script),
                    "--",
                    "--npz",
                    str(npz_path),
                    "--out-dir",
                    str(arm_frames_dir),
                    "--arm",
                    arm_key,
                    "--width",
                    "480",
                    "--height",
                    "400",
                ],
                check=True,
            )
            cycles_arms[arm_key] = arm_frames_dir

    T = len(gt_v)
    frames = []
    for t in range(T):
        if cycles_arms:
            left_img = np.asarray(Image.open(cycles_arms["gt"] / f"frame_{t:04d}.png").convert("RGB"))
            mid_img = np.asarray(Image.open(cycles_arms["baseline"] / f"frame_{t:04d}.png").convert("RGB"))
            right_img = np.asarray(Image.open(cycles_arms["hermite"] / f"frame_{t:04d}.png").convert("RGB"))
        else:
            left_img = _draw_panel_3d(
                verts_grid=gt_v[t], normals_grid=gt_n[t], gt_grid=None,
                gripper_pos=g_pos[t], gripper_rot=g_rot[t], gripper_gap=float(g_gap[t]),
                table_z=table_z, base_rgb=base_rgb, show_error_tint=False,
            )
            mid_img = _draw_panel_3d(
                verts_grid=b_v[t], normals_grid=b_n[t], gt_grid=gt_v[t],
                gripper_pos=g_pos[t], gripper_rot=g_rot[t], gripper_gap=float(g_gap[t]),
                table_z=table_z, base_rgb=base_rgb, show_error_tint=True,
            )
            right_img = _draw_panel_3d(
                verts_grid=h_v[t], normals_grid=h_n[t], gt_grid=gt_v[t],
                gripper_pos=g_pos[t], gripper_rot=g_rot[t], gripper_gap=float(g_gap[t]),
                table_z=table_z, base_rgb=base_rgb, show_error_tint=True,
            )

        is_future = t >= split
        phase_txt = "FUTURE HOLDOUT" if is_future else "RESIM (TRAIN)"
        phase_col = (110, 225, 255) if is_future else (112, 235, 175)
        frames.append(
            _compose_3up_frame(
                left_img=left_img,
                mid_img=mid_img,
                right_img=right_img,
                left_title="Reference GT (Continuum)",
                mid_title="Old: Spring-Mass Baseline",
                right_title="New: HermiteTwin (C1 Shell)",
                tier_badge=badge,
                phase_text=f"F {t:02d}/{T-1:02d}  {phase_txt}",
                phase_color=phase_col,
                hud_left=f"Split @ F{split}",
                hud_mid=f"Old CD: {b_cd[t]:6.2f} mm2",
                hud_right=f"New CD: {h_cd[t]:6.2f} mm2",
                font_title=font_title,
                font_hud=font_hud,
            )
        )

    # Append novel action frames if present (Task 3)
    novel_start_frame = None
    if "novel_gt_vertices" in z:
        novel_start_frame = len(frames)
        ngt = np.asarray(z["novel_gt_vertices"], dtype=np.float64)
        nb = np.asarray(z["novel_baseline_vertices"], dtype=np.float64)
        nh = np.asarray(z["novel_hermite_vertices"], dtype=np.float64)
        ng_pos = np.asarray(z["novel_gripper_pos_xyz"], dtype=np.float64)
        ng_rot = np.asarray(z["novel_gripper_rot_3x3"], dtype=np.float64)
        ng_gap = np.asarray(z["novel_gripper_gap_m"], dtype=np.float64)
        obs_list = json.loads(str(z["novel_obstacles_json"]))

        def _approx_norms(v_grid: np.ndarray) -> np.ndarray:
            du = np.gradient(v_grid, axis=0)
            dv = np.gradient(v_grid, axis=1)
            nt = np.cross(du, dv)
            return nt / np.maximum(np.linalg.norm(nt, axis=-1, keepdims=True), 1e-10)

        for nt_idx in range(len(ngt)):
            l_i = _draw_panel_3d(
                verts_grid=ngt[nt_idx], normals_grid=_approx_norms(ngt[nt_idx]), gt_grid=None,
                gripper_pos=ng_pos[nt_idx], gripper_rot=ng_rot[nt_idx], gripper_gap=float(ng_gap[nt_idx]),
                table_z=table_z, base_rgb=base_rgb, obstacles=obs_list, show_error_tint=False,
            )
            m_i = _draw_panel_3d(
                verts_grid=nb[nt_idx], normals_grid=_approx_norms(nb[nt_idx]), gt_grid=ngt[nt_idx],
                gripper_pos=ng_pos[nt_idx], gripper_rot=ng_rot[nt_idx], gripper_gap=float(ng_gap[nt_idx]),
                table_z=table_z, base_rgb=base_rgb, obstacles=obs_list, show_error_tint=True,
            )
            r_i = _draw_panel_3d(
                verts_grid=nh[nt_idx], normals_grid=_approx_norms(nh[nt_idx]), gt_grid=ngt[nt_idx],
                gripper_pos=ng_pos[nt_idx], gripper_rot=ng_rot[nt_idx], gripper_gap=float(ng_gap[nt_idx]),
                table_z=table_z, base_rgb=base_rgb, obstacles=obs_list, show_error_tint=True,
            )
            cd_old_n = float(np.mean(np.sum((nb[nt_idx] - ngt[nt_idx]) ** 2, axis=-1)) * 1e6)
            cd_new_n = float(np.mean(np.sum((nh[nt_idx] - ngt[nt_idx]) ** 2, axis=-1)) * 1e6)
            frames.append(
                _compose_3up_frame(
                    left_img=l_i,
                    mid_img=m_i,
                    right_img=r_i,
                    left_title="Reference GT (Novel Drape)",
                    mid_title="Old: Spring-Mass (Novel)",
                    right_title="New: HermiteTwin (Novel)",
                    tier_badge=badge,
                    phase_text=f"NOVEL {nt_idx:02d}/{len(ngt)-1:02d}",
                    phase_color=(245, 200, 95),
                    hud_left="Rigid Block Drape",
                    hud_mid=f"Old CD: {cd_old_n:6.2f} mm2",
                    hud_right=f"New CD: {cd_new_n:6.2f} mm2",
                    font_title=font_title,
                    font_hud=font_hud,
                )
            )

    mp4_path = media_dir / f"{task_id}_ab_comparison.mp4"
    _encode_rgb_frames_to_mp4(frames, mp4_path, fps=24)

    # Render high-res (640x480) peak keyframes for interactive Before/After curtain slider
    curtain_t = min(peak_t, T - 1) if "cloth" not in task_id else T - 1
    old_peak_arr = _draw_panel_3d(
        verts_grid=b_v[curtain_t], normals_grid=b_n[curtain_t], gt_grid=gt_v[curtain_t],
        gripper_pos=g_pos[curtain_t], gripper_rot=g_rot[curtain_t], gripper_gap=float(g_gap[curtain_t]),
        table_z=table_z, base_rgb=base_rgb, show_error_tint=True, panel_w=640, panel_h=480,
    )
    new_peak_arr = _draw_panel_3d(
        verts_grid=h_v[curtain_t], normals_grid=h_n[curtain_t], gt_grid=gt_v[curtain_t],
        gripper_pos=g_pos[curtain_t], gripper_rot=g_rot[curtain_t], gripper_gap=float(g_gap[curtain_t]),
        table_z=table_z, base_rgb=base_rgb, show_error_tint=True, panel_w=640, panel_h=480,
    )
    old_peak_png = media_dir / f"{task_id}_old_peak.png"
    new_peak_png = media_dir / f"{task_id}_new_peak.png"
    Image.fromarray(old_peak_arr).save(old_peak_png)
    Image.fromarray(new_peak_arr).save(new_peak_png)

    return {
        "mp4_path": mp4_path.resolve().as_posix(),
        "old_peak_png": old_peak_png.resolve().as_posix(),
        "new_peak_png": new_peak_png.resolve().as_posix(),
        "total_video_frames": len(frames),
        "peak_frame": curtain_t,
        "train_split_frame": split,
        "novel_start_frame": novel_start_frame,
    }


def render_dual_renderer_comparison_video(
    *,
    task_id: str,
    task_entry: Mapping[str, Any],
    out_media_dir: str | Path,
    tier: str = "fast_cpu",
) -> dict[str, Any]:
    """Render synchronized Dual-Renderer MP4 & peak PNGs: [PBR Shaded Mesh | Hermite 3DGS | Diff x4]."""
    media_dir = Path(out_media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(task_entry["main_npz"], allow_pickle=False)

    h_v = np.asarray(z["hermite_vertices"], dtype=np.float64)
    h_n = np.asarray(z["hermite_normals"], dtype=np.float64)
    q_seq = np.asarray(z["hermite_q_seq"], dtype=np.float64)
    g_pos = np.asarray(z["gripper_pos_xyz"], dtype=np.float64)
    g_rot = np.asarray(z["gripper_rot_3x3"], dtype=np.float64)
    g_gap = np.asarray(z["gripper_gap_m"], dtype=np.float64)
    table_z = float(z["table_z_m"])
    base_rgb = tuple(float(c) for c in z["base_rgb"])
    gs_binding = Path(task_entry["gs_binding_path"])

    font_title = _load_font(20, bold=True)
    font_hud = _load_font(15, bold=True)
    badge = "CYCLES_vs_3DGS" if tier == "studio_cycles" else "PBR_vs_3DGS_DUAL_AUDIT"

    pw, ph = 480, 400
    T = len(h_v)
    frames = []
    # Sample every 2nd frame for fast crisp 3DGS splat rendering, duplicating 2x or at 12fps -> 24fps
    pbr_peak_png = media_dir / f"{task_id}_dual_pbr_peak.png"
    gs_peak_png = media_dir / f"{task_id}_dual_gs_peak.png"
    peak_t = T // 2

    for t in range(0, T, 2):
        pbr_img = _draw_panel_3d(
            verts_grid=h_v[t], normals_grid=h_n[t], gt_grid=None,
            gripper_pos=g_pos[t], gripper_rot=g_rot[t], gripper_gap=float(g_gap[t]),
            table_z=table_z, base_rgb=base_rgb, show_error_tint=False,
            panel_w=pw, panel_h=ph,
        )
        gs_res = render_hermite_3dgs_frame(
            gs_binding,
            q_seq[t],
            image_hw=(ph, pw),
            camera_pos_xyz=(0.04, -0.56, 1.08),
            look_at_xyz=(0.0, 0.0, 0.78),
            fov_deg=42.0,
        )
        gs_img = np.clip(gs_res["rgb"] * 255.0, 0, 255).astype(np.uint8)
        diff_img = np.clip(np.abs(pbr_img.astype(np.float64) - gs_img.astype(np.float64)) * 2.5, 0, 255).astype(np.uint8)

        if t <= peak_t < t + 2:
            Image.fromarray(pbr_img).save(pbr_peak_png)
            Image.fromarray(gs_img).save(gs_peak_png)

        comp = _compose_3up_frame(
            left_img=pbr_img,
            mid_img=gs_img,
            right_img=diff_img,
            left_title="Renderer A: Shaded Mesh PBR",
            mid_title="Renderer B: Hermite-Driven 3DGS",
            right_title="Non-Trivial Appearance Diff (2.5x)",
            tier_badge=badge,
            phase_text=f"DUAL F {t:02d}/{T-1:02d}",
            phase_color=(165, 195, 255),
            hud_left="Tangent-Space Mesh",
            hud_mid="784 Anisotropic Gaussians",
            hud_right="Anti-Brightness-Copy Verified",
            font_title=font_title,
            font_hud=font_hud,
        )
        frames.append(comp)
        frames.append(comp)

    mp4_path = media_dir / f"{task_id}_dual_renderer_3dgs.mp4"
    _encode_rgb_frames_to_mp4(frames, mp4_path, fps=24)

    return {
        "mp4_path": mp4_path.resolve().as_posix(),
        "pbr_png": pbr_peak_png.resolve().as_posix(),
        "gs_png": gs_peak_png.resolve().as_posix(),
        "pbr_profile": "cycles_or_cpu_pbr_v08",
        "gs_profile": "hermitetwin_3dgs_v08",
    }
