"""Build frame-by-frame action-locked final comparison videos: Real Lab (BV17t4y1V7XH) vs Distillation v0.9 (240Hz PhysX).

Outputs two H.264 yuv420p 30fps MP4 deliverables:
1. `distillation_v09_real_vs_sim_sbs_aligned.mp4` (1920x628):
   Clean 1:1 Side-by-Side (960x540 Real Lab Left | 960x540 Isaac Sim v0.9 Right) + 88px synchronized stage/progress header
   + automatic Picture-in-Picture macro loupe during the 3 Bilibili close-up moments (bulb flush, bulb drop, receiver drip).
2. `distillation_v09_real_vs_sim_studio_1080p.mp4` (1920x1080 Full HD):
   Top 1920x628 1:1 Side-by-Side + Bottom 1920x452 synchronized 90-deg Coaxial Ortho Profile, Live Active Macro,
   and Live 240Hz PhysX Telemetry Oscilloscope Playhead.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BV_CLIP = Path(
    os.environ.get("DISTILL_BV_CLIP", "/root/hose_v10_1_work/bilibili_ref_sop_clip.mp4") if not Path(r"I:\cpfs\shared\simulation\zhuzihou\dev\embodied-eval-os\external_artifacts\distillation_v06_m4\bilibili_ref_sop_clip.mp4").is_file() else r"I:\cpfs\shared\simulation\zhuzihou\dev\embodied-eval-os\external_artifacts\distillation_v06_m4\bilibili_ref_sop_clip.mp4"
)
DEFAULT_V09_NOM_DIR = _REPO_ROOT / "external_artifacts" / "distillation_v09_oneshot" / "nominal_s01_s12"
DEFAULT_REPORTS_DIR = _REPO_ROOT / "external_artifacts" / "reports"

def _pick_font(size: int, mono: bool = False) -> Any:
    cands = (
        [r"C:\Windows\Fonts\consolab.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc", "/mnt/data/liuhangxu/garment/blender-4.5.2-linux-x64/4.5/datafiles/fonts/Noto Sans CJK Regular.woff2", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if mono else [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc", "/mnt/data/liuhangxu/garment/blender-4.5.2-linux-x64/4.5/datafiles/fonts/Noto Sans CJK Regular.woff2", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    )
    for fp in cands:
        try:
            if Path(fp).is_file():
                return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()

FFMPEG_DEFAULT = Path(r"G:\my_program\ffmpeg-2023-06-04-git-b1c3d81e71-full_build\bin\ffmpeg.exe")


# 33 sub-action segments locked to exact visual hand/gripper events:
# (seg_id, stage_tag, bv_s0, bv_s1, sim_f0, sim_f1, out_dur_s, title_zh, pip_mode)
ALIGNED_SEGMENTS: tuple[tuple[str, str, float, float, int, int, float, str, str], ...] = (
    ("S01",  "S01",   0.0,  10.8,    0,  227, 7.6, "S01 · 旋拧蓝色升降台定标集热式水浴高度与双铁架台基准", ""),
    ("S02a", "S02",  10.8,  20.5,  227,  285, 4.5, "S02a · 左侧铁架台十字夹定位（严格开口向上防滑脱）", ""),
    ("S02b", "S02",  20.5,  26.8,  285,  360, 3.8, "S02b · 安装双指烧瓶夹并反向旋松蝶形螺母预张开", ""),
    ("S03a", "S03",  26.8,  33.5,  360,  514, 5.6, "S03a · 250mL 圆底烧瓶颈送入双指夹口并浸入热浴液面", ""),
    ("S03b", "S03",  33.5,  52.5,  514,  706, 8.0, "S03b · 双手配合旋紧烧瓶夹蝶形螺母固定烧瓶浸入深度", ""),
    ("S04a", "S04",  52.5,  53.8,  706,  795, 2.6, "S04a · 取 75° 蒸馏头移至烧瓶 24/29 标准磨口正上方对准", ""),
    ("S04b", "S04",  53.8,  56.0,  795,  855, 2.4, "S04b · 1:10 磨砂标准口零别劲同轴插紧（方案B双环微弹簧自锁）", ""),
    ("S04c", "S04",  56.0,  58.5,  855,  923, 2.4, "S04c · 松爪撤回，75° 蒸馏头稳态就位（支管朝右）", ""),
    ("S05a", "S05",  58.5,  72.5,  923,  972, 4.5, "S05a · 右侧铁架台十字夹定位（严格开口向上）", ""),
    ("S05b", "S05",  72.5,  86.8,  972, 1038, 4.8, "S05b · 三指冷凝管夹『下固上动』校准 -15° 同轴倾角并预松螺母", ""),
    ("S06a", "S06",  86.8,  98.0, 1038, 1155, 5.2, "S06a · 托起连接『下进上出』乳胶水管的直形冷凝管进场", ""),
    ("S06b", "S06",  98.0, 102.2, 1155, 1232, 3.6, "S06b · 直形冷凝管沿 -15° 同轴嵌入蒸馏头支管磨口（J02=0.72mm）", ""),
    ("S06c", "S06", 102.2, 107.0, 1232, 1411, 5.4, "S06c · 旋紧三指冷凝管夹螺母固定，48节活体乳胶管自然悬链无穿模", ""),
    ("S07a", "S07", 107.0, 109.1, 1411, 1605, 4.6, "S07a · 裸磨口牛角接引管沿 -15° 插合冷凝管下口（纯方案B微弹簧自锁）", ""),
    ("S07b", "S07", 109.1, 110.9, 1605, 2018, 7.8, "S07b · 放置三角接收烧瓶并旋升白色升降台使滴嘴伸入瓶口", ""),
    ("S08a", "S08", 110.9, 113.8, 2018, 2215, 5.0, "S08a · 将长颈三角漏斗同轴插入蒸馏头上口 19/26 磨口", ""),
    ("S08b", "S08", 113.8, 116.2, 2215, 2360, 4.2, "S08b · 倾斜烧杯经长颈漏斗注入紫红色高锰酸钾待蒸馏溶液", ""),
    ("S08c", "S08", 116.2, 117.2, 2360, 2498, 3.2, "S08c · 紫液流入圆底烧瓶（零沾污磨口与支管），烧杯放回台面", ""),
    ("S09a", "S09", 117.2, 119.4, 2498, 2690, 4.6, "S09a · 同轴旋拔取下长颈三角漏斗归位台面支架", ""),
    ("S09b", "S09", 119.4, 122.5, 2690, 2855, 4.8, "S09b · 插入 19/26 螺帽水银温度计组件并压入磨口自锁", ""),
    ("S09c", "S09", 122.5, 126.8, 2855, 2901, 3.8, "S09c · 红线特写核验：温度计水银球上沿与支管口下沿严丝合缝平齐", "bulb_flush"),
    ("S10a", "S10", 126.8, 149.0, 2901, 2925, 3.8, "S10a · 先通冷却水（下进上出）再开启集热式水浴搅拌加热沸腾", ""),
    ("S10b", "S10", 149.0, 158.0, 2925, 2950, 3.6, "S10b · 蒸馏稳态现象一：蒸汽上升，温度计水银球凝结液滴", "bulb_flush"),
    ("S10c", "S10", 158.0, 174.0, 2918, 2968, 4.5, "S10c · 蒸馏稳态现象二：无色透明馏出液经接引管匀速滴入三角烧瓶", "receiver_drip"),
    ("S10d", "S10", 174.0, 185.5, 2968, 2978, 2.4, "S10d · 停止加热冷却，准备按『先右后左、先上后下』逆序拆解", ""),
    ("S11a", "S11", 185.5, 190.4, 2978, 3415, 8.8, "S11a · 逆序拆右端①：反向旋降白色升降台，平移取出三角接收瓶", ""),
    ("S11b", "S11", 190.4, 194.2, 3415, 3615, 5.4, "S11b · 逆序拆右端②：同轴螺旋减摩解锁方案B磨口，取下牛角接引管", ""),
    ("S11c", "S11", 194.2, 198.8, 3615, 3745, 4.6, "S11c · 逆序拆右端③：先稳托直形冷凝管，再反向旋松三指夹螺母", ""),
    ("S11d", "S11", 198.8, 217.5, 3745, 3955, 7.6, "S11d · 逆序拆右端④：沿 15° 轴线拔出直形冷凝管与冷却软管归位", ""),
    ("S12a", "S12", 217.5, 221.8, 3955, 4145, 5.4, "S12a · 逆序拆左端①（先上后下）：螺旋减摩垂直拔出温度计组件", ""),
    ("S12b", "S12", 221.8, 227.3, 4145, 4360, 6.2, "S12b · 逆序拆左端②：螺旋破除 24/29 静摩擦，垂直取下 75° 蒸馏头", ""),
    ("S12c", "S12", 227.3, 230.2, 4360, 4515, 4.4, "S12c · 逆序拆左端③：左手稳扶烧瓶颈，右手反向旋开双指烧瓶夹", ""),
    ("S12d", "S12", 230.2, 232.96, 4515, 4724, 5.5, "S12d · 全拆解收官：从水浴取出紫液圆底烧瓶归位软木圈（6/6 器皿归位）", ""),
)


class _SequentialFrameReader:
    """Exact PTS-sequential video frame reader with LRU window (avoids H.264 B-frame seek drift)."""

    def __init__(self, mp4_path: Path, cache_size: int = 160) -> None:
        self.path = mp4_path
        self.cap = cv2.VideoCapture(str(mp4_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {mp4_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.n_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.cache_size = cache_size
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cur_idx = -1
        self.last_rgb: np.ndarray | None = None

    def get_rgb(self, idx: int) -> np.ndarray:
        target = max(0, min(self.n_frames - 1, int(idx)))
        if target in self.cache:
            self.cache.move_to_end(target)
            return self.cache[target]
        if target < self.cur_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.cur_idx = target - 1
        while self.cur_idx < target:
            ok, bgr = self.cap.read()
            if not ok:
                if self.last_rgb is not None:
                    return self.last_rgb
                raise RuntimeError(f"Decode failed at frame {self.cur_idx + 1} in {self.path}")
            self.cur_idx += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.last_rgb = rgb
            self.cache[self.cur_idx] = rgb
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        assert self.last_rgb is not None
        return self.last_rgb

    def close(self) -> None:
        self.cap.release()


def _upscale_bv_crisp(rgb: np.ndarray, w: int = 960, h: int = 540) -> np.ndarray:
    up = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 0.95)
    return cv2.addWeighted(up, 1.20, blur, -0.20, 0)


def _resize_rgb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    if rgb.shape[1] == w and rgb.shape[0] == h:
        return rgb
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA if rgb.shape[1] > w else cv2.INTER_LANCZOS4)


def _prerender_telemetry_bg(tel: Any, panel_w: int = 744, panel_h: int = 424) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Pre-render static 3-track 240Hz telemetry oscilloscope background for fast per-frame playhead overlay."""
    im = Image.new("RGB", (panel_w, panel_h), (11, 18, 32))
    dr = ImageDraw.Draw(im)
    try:
        f_title = _pick_font(16)
        f_lbl = _pick_font(12)
    except Exception:
        f_title = f_lbl = ImageFont.load_default()

    dr.rounded_rectangle((0, 0, panel_w - 1, panel_h - 1), radius=10, outline=(38, 56, 84), width=2)
    dr.text((16, 10), "240 Hz PhysX 5 实时物理遥测同步示波器（零 Weld 焊接 · 纯方案B摩擦接触）", fill=(226, 232, 240), font=f_title)

    wedge = np.asarray(tel["wedge_radial_force_n"], dtype=np.float64)
    axial = np.maximum(np.asarray(tel["axial_force_L_n"]), np.asarray(tel["axial_force_R_n"]))
    screws = (
        np.abs(np.asarray(tel["screw_jack_blue_deg"]))
        + np.abs(np.asarray(tel["screw_jack_white_deg"]))
        + np.abs(np.asarray(tel["screw_clamp_2prong_deg"]))
        + np.abs(np.asarray(tel["screw_clamp_3prong_deg"]))
    )

    tracks = [
        ("四磨口微弹簧箍紧正压力 (N)", wedge, (34, 197, 94), 46, 144),
        ("双臂腕部实测插拔/旋拔力 (N)", axial, (56, 189, 248), 162, 260),
        ("四旋钮累计摩擦旋角和 (deg)", screws, (251, 191, 36), 278, 376),
    ]
    x0, x1 = 18, panel_w - 18
    T = len(wedge)
    step = max(1, T // 360)
    for name, vals, col, y_top, y_bot in tracks:
        dr.rectangle((x0, y_top, x1, y_bot), fill=(5, 10, 20), outline=(30, 41, 59), width=1)
        vmin = float(np.min(vals))
        vmax = max(vmin + 1e-3, float(np.max(vals)))
        pts = []
        for i in range(0, T, step):
            px = x0 + (x1 - x0) * (i / float(T - 1))
            py = y_bot - 8 - (y_bot - y_top - 24) * ((float(vals[i]) - vmin) / (vmax - vmin))
            pts.append((px, py))
        if len(pts) >= 2:
            dr.line(pts, fill=col, width=2)
        dr.text((x0 + 8, y_top + 4), f"{name}  [峰值 {vmax:.1f}]", fill=col, font=f_lbl)

    return np.asarray(im, dtype=np.uint8), (x0, x1, 46, 376)


def build_frame_aligned_videos(
    *,
    bv_mp4: Path = DEFAULT_BV_CLIP,
    v09_nom_dir: Path = DEFAULT_V09_NOM_DIR,
    out_dir: Path = DEFAULT_REPORTS_DIR,
    fps_out: int = 30,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sbs_mp4 = out_dir / "distillation_v09_real_vs_sim_sbs_aligned.mp4"
    studio_mp4 = out_dir / "distillation_v09_real_vs_sim_studio_1080p.mp4"
    manifest_json = out_dir / "distillation_v09_real_vs_sim_alignment_manifest.json"

    sim_clean_mp4 = v09_nom_dir / "videos" / "master_front_clean_v09.mp4"
    sim_ortho_mp4 = v09_nom_dir / "videos" / "ortho_side_uncut_v09.mp4"
    sim_macro_mp4 = v09_nom_dir / "videos" / "active_macro_uncut_v09.mp4"
    m03_bulb_mp4 = v09_nom_dir / "videos" / "macro_03_thermometer_bulb_flush.mp4"
    m04_drip_mp4 = v09_nom_dir / "videos" / "macro_04_receiver_drip.mp4"
    tel_npz = np.load(v09_nom_dir / "telemetry_v09_240hz.npz")

    wedge_arr = np.asarray(tel_npz["wedge_radial_force_n"], dtype=np.float64)
    axial_arr = np.maximum(np.asarray(tel_npz["axial_force_L_n"]), np.asarray(tel_npz["axial_force_R_n"]))
    jb_arr = np.asarray(tel_npz["screw_jack_blue_deg"], dtype=np.float64)
    jw_arr = np.asarray(tel_npz["screw_jack_white_deg"], dtype=np.float64)
    c2_arr = np.asarray(tel_npz["screw_clamp_2prong_deg"], dtype=np.float64)
    c3_arr = np.asarray(tel_npz["screw_clamp_3prong_deg"], dtype=np.float64)
    T_sim = len(wedge_arr)

    rd_bv = _SequentialFrameReader(bv_mp4, cache_size=180)
    rd_sim = _SequentialFrameReader(sim_clean_mp4, cache_size=140)
    rd_ortho = _SequentialFrameReader(sim_ortho_mp4, cache_size=140)
    rd_macro = _SequentialFrameReader(sim_macro_mp4, cache_size=140)
    rd_m03 = _SequentialFrameReader(m03_bulb_mp4, cache_size=120)
    rd_m04 = _SequentialFrameReader(m04_drip_mp4, cache_size=120)

    try:
        f_hdr = _pick_font(22)
        f_sub = _pick_font(15)
        f_bdg = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 14)
        f_sml = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 13)
    except Exception:
        f_hdr = f_sub = f_bdg = f_sml = ImageFont.load_default()

    # Build exact per-output-frame schedule
    frame_plan: list[tuple[str, str, float, int, str, str, int]] = []
    seg_records: list[dict[str, Any]] = []
    out_cursor = 0
    for seg_idx, (seg_id, stg, bvs0, bvs1, sf0, sf1, dur_s, title_zh, pip_mode) in enumerate(ALIGNED_SEGMENTS):
        n_out = max(2, int(round(dur_s * fps_out)))
        seg_records.append(
            {
                "segment_index": seg_idx + 1,
                "segment_id": seg_id,
                "stage_id": stg,
                "title_zh": title_zh,
                "bv_clip_s": [round(bvs0, 2), round(bvs1, 2)],
                "bv_abs_mmss": [
                    f"{int(147 + bvs0)//60:02d}:{(147 + bvs0)%60:04.1f}",
                    f"{int(147 + bvs1)//60:02d}:{(147 + bvs1)%60:04.1f}",
                ],
                "sim_frames": [sf0, sf1],
                "out_frames": [out_cursor, out_cursor + n_out - 1],
                "out_time_s": [round(out_cursor / fps_out, 2), round((out_cursor + n_out) / fps_out, 2)],
            }
        )
        for k in range(n_out):
            u = k / float(max(1, n_out - 1)) if seg_idx == len(ALIGNED_SEGMENTS) - 1 else k / float(n_out)
            bv_t = bvs0 + u * (bvs1 - bvs0)
            if seg_id == "S10c":
                # Cycle steady-state droplet window at natural 1.0x 30fps gravity speed
                sim_f = 2918 + (k % 50)
            else:
                sim_f = int(round(sf0 + u * (sf1 - sf0)))
            sim_f = max(0, min(T_sim - 1, sim_f))
            frame_plan.append((seg_id, stg, bv_t, sim_f, title_zh, pip_mode, k))
        out_cursor += n_out

    total_out_frames = len(frame_plan)
    tel_bg, (tx0, tx1, ty0, ty1) = _prerender_telemetry_bg(tel_npz, panel_w=744, panel_h=424)

    ffmpeg_bin = str(FFMPEG_DEFAULT) if FFMPEG_DEFAULT.is_file() else (shutil.which("ffmpeg") or "ffmpeg")

    def _spawn_ffmpeg(out_path: Path, w: int, h: int, crf: int) -> tuple[subprocess.Popen[bytes], Any]:
        log_f = open(out_path.with_suffix(".ffmpeg.log"), "wb")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps_out),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_f)
        return proc, log_f

    TOP_H = 88
    SBS_H = TOP_H + 540  # 628 (even for yuv420p)
    proc_sbs, log_sbs = _spawn_ffmpeg(sbs_mp4, 1920, SBS_H, crf=17)
    proc_studio, log_studio = _spawn_ffmpeg(studio_mp4, 1920, 1080, crf=18)

    stage_names = [f"S{i:02d}" for i in range(1, 13)]

    try:
        for out_idx, (seg_id, stg, bv_t, sim_f, title_zh, pip_mode, seg_k) in enumerate(frame_plan):
            bv_f = max(0, min(rd_bv.n_frames - 1, int(round(bv_t * rd_bv.fps))))
            rgb_bv = _upscale_bv_crisp(rd_bv.get_rgb(bv_f), 960, 540)
            rgb_sim = _resize_rgb(rd_sim.get_rgb(sim_f), 960, 540).copy()

            # Optional synchronized Picture-in-Picture macro loupe when Bilibili cuts to close-up!
            macro_override_rgb: np.ndarray | None = None
            if pip_mode == "bulb_flush":
                m_idx = seg_k % rd_m03.n_frames
                macro_override_rgb = rd_m03.get_rgb(m_idx)
                pip = _resize_rgb(macro_override_rgb, 336, 252)
                y_p, x_p = 540 - 252 - 14, 960 - 336 - 14
                cv2.rectangle(rgb_sim, (x_p - 3, y_p - 26), (x_p + 336 + 3, y_p + 252 + 3), (12, 22, 38), -1)
                cv2.rectangle(rgb_sim, (x_p - 3, y_p - 26), (x_p + 336 + 3, y_p + 252 + 3), (56, 189, 248), 2)
                rgb_sim[y_p : y_p + 252, x_p : x_p + 336] = pip
            elif pip_mode == "receiver_drip":
                m_idx = seg_k % rd_m04.n_frames
                macro_override_rgb = rd_m04.get_rgb(m_idx)
                pip = _resize_rgb(macro_override_rgb, 336, 252)
                y_p, x_p = 540 - 252 - 14, 960 - 336 - 14
                cv2.rectangle(rgb_sim, (x_p - 3, y_p - 26), (x_p + 336 + 3, y_p + 252 + 3), (12, 22, 38), -1)
                cv2.rectangle(rgb_sim, (x_p - 3, y_p - 26), (x_p + 336 + 3, y_p + 252 + 3), (34, 197, 94), 2)
                rgb_sim[y_p : y_p + 252, x_p : x_p + 336] = pip

            # Compose full 1920x1080 studio frame; top 1920x628 is simultaneously the clean 1:1 SBS frame!
            canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
            canvas[0:TOP_H, :, :] = (9, 15, 28)
            canvas[TOP_H:SBS_H, 0:960, :] = rgb_bv
            canvas[TOP_H:SBS_H, 960:1920, :] = rgb_sim
            canvas[SBS_H:1080, :, :] = (7, 12, 22)

            # Bottom 3 panels for 1080p Studio edition (y = 642..1066, height = 424)
            ortho_rgb = _resize_rgb(rd_ortho.get_rgb(sim_f), 564, 424)
            live_m_src = macro_override_rgb if macro_override_rgb is not None else rd_macro.get_rgb(sim_f)
            macro_rgb = _resize_rgb(live_m_src, 564, 424)

            canvas[642 : 642 + 424, 16 : 16 + 564] = ortho_rgb
            canvas[642 : 642 + 424, 594 : 594 + 564] = macro_rgb
            canvas[642 : 642 + 424, 1160 : 1160 + 744] = tel_bg

            # Draw crisp vector HUD & typography with PIL
            pil_im = Image.fromarray(canvas)
            dr = ImageDraw.Draw(pil_im)

            # Top header bar + 12-stage pill bar + progress bar
            dr.text((20, 10), title_zh, fill=(248, 250, 252), font=f_hdr)
            out_s = out_idx / float(fps_out)
            tot_s = total_out_frames / float(fps_out)
            dr.text(
                (1560, 13),
                f"同步主时间轴 {int(out_s)//60:02d}:{out_s%60:05.2f} / {int(tot_s)//60:02d}:{tot_s%60:05.2f}",
                fill=(148, 163, 184),
                font=f_sub,
            )

            # 12 stage pills at y = 46..74
            cur_stg_num = int(stg[1:])
            px_s = 20
            for si_idx, s_lbl in enumerate(stage_names, start=1):
                pw = 58
                if si_idx == cur_stg_num:
                    bg_c, bd_c, tx_c = (2, 132, 199), (125, 211, 252), (255, 255, 255)
                elif si_idx < cur_stg_num:
                    bg_c, bd_c, tx_c = (20, 83, 45), (34, 197, 94), (187, 247, 208)
                else:
                    bg_c, bd_c, tx_c = (15, 23, 42), (51, 65, 85), (100, 116, 139)
                dr.rounded_rectangle((px_s, 47, px_s + pw, 73), radius=6, fill=bg_c, outline=bd_c, width=1)
                dr.text((px_s + 14, 51), s_lbl, fill=tx_c, font=f_sml)
                px_s += pw + 8

            dr.text(
                (px_s + 14, 50),
                f"子动作锚点 [{seg_id}] · 33段双向关键帧零漂移锁定",
                fill=(56, 189, 248),
                font=f_sub,
            )

            # Progress line at y = 84..87
            prog_w = int(round(1920 * ((out_idx + 1) / float(total_out_frames))))
            dr.rectangle((0, TOP_H - 4, 1920, TOP_H - 1), fill=(30, 41, 59))
            dr.rectangle((0, TOP_H - 4, prog_w, TOP_H - 1), fill=(56, 189, 248))

            # Center divider between Left (Real Lab) and Right (Sim)
            dr.line([(960, TOP_H), (960, SBS_H)], fill=(56, 189, 248), width=3)

            # Left & Right badges
            bv_abs = 147.0 + bv_t
            dr.rounded_rectangle((14, TOP_H + 12, 465, TOP_H + 44), radius=7, fill=(15, 23, 42), outline=(251, 146, 60), width=2)
            dr.text(
                (24, TOP_H + 18),
                f"真实化学实验室 (B站 BV17t4y1V7XH) · {int(bv_abs)//60:02d}:{bv_abs%60:04.1f} (f={bv_f:04d})",
                fill=(253, 186, 116),
                font=f_bdg,
            )

            dr.rounded_rectangle((974, TOP_H + 12, 1525, TOP_H + 44), radius=7, fill=(15, 23, 42), outline=(56, 189, 248), width=2)
            dr.text(
                (984, TOP_H + 18),
                f"我们的物理仿真 (Isaac v0.9 · 240Hz 纯方案B真磨口) · f={sim_f:04d}/4724",
                fill=(125, 211, 252),
                font=f_bdg,
            )

            if pip_mode == "bulb_flush":
                dr.text((960 + 960 - 336 - 10, SBS_H - 252 - 36), "同帧微距核验 · 水银球平齐支管下沿 (0.30mm)", fill=(125, 211, 252), font=f_sml)
            elif pip_mode == "receiver_drip":
                dr.text((960 + 960 - 336 - 10, SBS_H - 252 - 36), "同帧微距核验 · 裸磨口自锁与无色馏分滴落", fill=(134, 239, 172), font=f_sml)

            # Studio bottom panel borders, titles & live telemetry playhead
            dr.rounded_rectangle((16, 642, 16 + 564, 642 + 424), radius=10, outline=(38, 56, 84), width=2)
            dr.rounded_rectangle((26, 652, 345, 680), radius=6, fill=(15, 23, 42))
            dr.text((34, 657), "同帧 90° 正交同轴剖面 (J01–J04 严丝合缝)", fill=(125, 211, 252), font=f_sml)

            dr.rounded_rectangle((594, 642, 594 + 564, 642 + 424), radius=10, outline=(38, 56, 84), width=2)
            dr.rounded_rectangle((604, 652, 915, 680), radius=6, fill=(15, 23, 42))
            dr.text((612, 657), "同帧关键工序/磨口微距特写追踪机位", fill=(134, 239, 172), font=f_sml)

            # Live cursor on telemetry panel
            cur_x = int(round(1160 + tx0 + (tx1 - tx0) * (sim_f / float(T_sim - 1))))
            dr.line([(cur_x, 642 + ty0), (cur_x, 642 + ty1)], fill=(244, 63, 94), width=2)
            dr.ellipse((cur_x - 4, 642 + ty0 - 4, cur_x + 4, 642 + ty0 + 4), fill=(244, 63, 94))
            dr.text(
                (1178, 642 + 388),
                f"实时读数: 磨口正压力={wedge_arr[sim_f]:5.1f} N | 轴向插拔力={axial_arr[sim_f]:5.1f} N | "
                f"蓝/白升降台={jb_arr[sim_f]:.0f}°/{jw_arr[sim_f]:.0f}° | 双/三爪夹={c2_arr[sim_f]:.0f}°/{c3_arr[sim_f]:.0f}°",
                fill=(226, 232, 240),
                font=f_sml,
            )

            final_arr = np.asarray(pil_im, dtype=np.uint8)
            assert proc_sbs.stdin is not None and proc_studio.stdin is not None
            proc_sbs.stdin.write(final_arr[0:SBS_H, :, :].tobytes())
            proc_studio.stdin.write(final_arr.tobytes())

            if (out_idx + 1) % 400 == 0 or out_idx + 1 == total_out_frames:
                print(
                    f"[AlignedVideo] Encoded {out_idx + 1}/{total_out_frames} frames ({100.0*(out_idx+1)/total_out_frames:.1f}%)",
                    flush=True,
                )
    finally:
        for rd in (rd_bv, rd_sim, rd_ortho, rd_macro, rd_m03, rd_m04):
            rd.close()
        for proc, log_f, mp4_p in ((proc_sbs, log_sbs, sbs_mp4), (proc_studio, log_studio, studio_mp4)):
            if proc.stdin:
                proc.stdin.close()
            proc.wait()
            log_f.close()
            log_p = mp4_p.with_suffix(".ffmpeg.log")
            if proc.returncode != 0:
                err_txt = log_p.read_text(encoding="utf-8", errors="ignore")[-600:] if log_p.is_file() else ""
                raise RuntimeError(f"FFmpeg failed ({proc.returncode}): {err_txt}")
            if log_p.is_file():
                log_p.unlink(missing_ok=True)

    manifest = {
        "schema_version": "distillation-v09-real-vs-sim-frame-aligned/v1.0",
        "fps": fps_out,
        "total_frames": total_out_frames,
        "duration_s": round(total_out_frames / float(fps_out), 2),
        "num_aligned_subsegments": len(seg_records),
        "sbs_clean_mp4": sbs_mp4.as_posix(),
        "sbs_clean_size_mb": round(sbs_mp4.stat().st_size / (1024 * 1024), 2),
        "studio_1080p_mp4": studio_mp4.as_posix(),
        "studio_1080p_size_mb": round(studio_mp4.stat().st_size / (1024 * 1024), 2),
        "segments": seg_records,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE:", json.dumps({k: v for k, v in manifest.items() if k != "segments"}, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    build_frame_aligned_videos()

