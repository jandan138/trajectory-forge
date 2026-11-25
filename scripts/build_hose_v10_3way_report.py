"""Build 3-way synchronized H.264 comparison videos & 100%-offline HTML report for Hose Softbody v1.0."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V10_DIR = _REPO_ROOT / "external_artifacts" / "hose_softbody_v10_3way"
DEFAULT_REPORTS_DIR = _REPO_ROOT / "external_artifacts" / "reports"
FFMPEG_DEFAULT = Path(r"G:\my_program\ffmpeg-2023-06-04-git-b1c3d81e71-full_build\bin\ffmpeg.exe")


def _find_ffmpeg() -> str:
    if FFMPEG_DEFAULT.is_file():
        return str(FFMPEG_DEFAULT)
    return shutil.which("ffmpeg") or "ffmpeg"


def _b64_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _compose_3way_mp4(
    mp4_paths: tuple[Path, Path, Path],
    labels: tuple[str, str, str],
    colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    header_zh: str,
    out_mp4: Path,
    fps: int = 30,
) -> Path:
    caps = [cv2.VideoCapture(str(p)) for p in mp4_paths]
    n_frames = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
    pw, ph = 640, 480
    hdr_h = 68
    W, H = pw * 3, ph + hdr_h  # 1920 x 548 (even)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
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
            f"{W}x{H}",
            "-r",
            str(fps),
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
            str(out_mp4),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        f_hdr = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 20)
        f_lbl = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 15)
    except Exception:
        f_hdr = f_lbl = ImageFont.load_default()

    try:
        for f in range(n_frames):
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            canvas[:hdr_h, :, :] = (10, 17, 32)
            for idx, cap in enumerate(caps):
                ok, bgr = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[1] != pw or rgb.shape[0] != ph:
                    rgb = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
                canvas[hdr_h:, idx * pw : (idx + 1) * pw, :] = rgb
            im = Image.fromarray(canvas)
            dr = ImageDraw.Draw(im)
            dr.text((18, 10), header_zh, fill=(248, 250, 252), font=f_hdr)
            dr.text((W - 220, 14), f"240Hz PhysX 5 · f={f:03d}/{n_frames-1}", fill=(148, 163, 184), font=f_lbl)
            for idx, (lbl, col) in enumerate(zip(labels, colors)):
                ox = idx * pw
                dr.rounded_rectangle((ox + 10, 40, ox + pw - 10, 64), radius=5, fill=(18, 28, 48), outline=col, width=2)
                dr.text((ox + 20, 42), lbl, fill=col, font=f_lbl)
                if idx > 0:
                    dr.line([(ox, hdr_h), (ox, H)], fill=(51, 65, 85), width=2)
            assert proc.stdin is not None
            proc.stdin.write(np.asarray(im, dtype=np.uint8).tobytes())
    finally:
        for cap in caps:
            cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed on {out_mp4}")
    return out_mp4


def _svg_multi_curve(
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    unit: str,
) -> str:
    w, h, pad = 640, 185, 30
    all_v = np.concatenate([s[1] for s in series])
    vmin = float(np.min(all_v))
    vmax = max(vmin + 1e-3, float(np.max(all_v)))
    lines = []
    leg = []
    for name, vals, col in series:
        n = len(vals)
        pts = []
        for i in range(n):
            x = pad + (w - 2 * pad) * (i / float(max(1, n - 1)))
            y = h - pad - (h - 2 * pad) * ((float(vals[i]) - vmin) / (vmax - vmin))
            pts.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.4" points="{" ".join(pts)}" />')
        leg.append(f'<span style="color:{col};margin-right:14px;font-weight:700;">■ {html.escape(name)}</span>')
    return f"""
    <div class="card">
      <div class="card-hdr">{html.escape(title)} <span class="muted">({vmin:.1f} .. {vmax:.1f} {html.escape(unit)})</span></div>
      <div style="font-size:12px;margin-bottom:6px;">{"".join(leg)}</div>
      <svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;background:#020617;border-radius:8px;border:1px solid #1e293b;">
        {"".join(lines)}
      </svg>
    </div>
    """


def build_hose_v10_3way_report(
    v10_dir: Path = DEFAULT_V10_DIR,
    out_dir: Path = DEFAULT_REPORTS_DIR,
) -> dict[str, Any]:
    summary = json.loads((v10_dir / "hose_v10_3way_master_summary.json").read_text(encoding="utf-8"))
    tr = summary["trials"]
    base = tr["baseline_bead_chain"]
    ra = tr["route_a_d6_der"]
    rb = tr["route_b_physx_fem"]
    ra_z = tr["route_a_zero_friction"]
    rb_z = tr["route_b_zero_friction"]
    ra_w = tr["route_a_wrong_port"]
    rb_w = tr["route_b_wrong_port"]

    vids_dir = v10_dir / "composed_3way_videos"
    v_e1 = _compose_3way_mp4(
        (Path(base["videos"]["exp1"]), Path(ra["videos"]["exp1"]), Path(rb["videos"]["exp1"])),
        (
            f"Baseline v0.9 球铰珠链 (挠度 {base['exp1_cantilever']['sag_eq_mm']:.1f}mm / 回弹 {base['exp1_cantilever']['elastic_rebound_pct']:.0f}%)",
            f"路线A D6-DER 弹性杆 (挠度 {ra['exp1_cantilever']['sag_eq_mm']:.1f}mm / 回弹 {ra['exp1_cantilever']['elastic_rebound_pct']:.0f}%)",
            f"路线B PhysX GPU FEM 中空管 (挠度 {rb['exp1_cantilever']['sag_eq_mm']:.1f}mm / 回弹 {rb['exp1_cantilever']['elastic_rebound_pct']:.0f}%)",
        ),
        ((248, 113, 113), (56, 189, 248), (34, 197, 94)),
        "实验 1 · 140mm 水平悬臂自然下挠与阶跃拨动弹性回弹三方同条件对照",
        vids_dir / "hose_v10_exp1_cantilever_3way.mp4",
    )
    v_e2 = _compose_3way_mp4(
        (Path(base["videos"]["exp2"]), Path(ra["videos"]["exp2"]), Path(rb["videos"]["exp2"])),
        (
            f"Baseline v0.9 球铰珠链 (末端扭转 {base['exp2_torsion_90deg']['tip_twist_steady_deg']:.1f}°)",
            f"路线A D6-DER 弹性杆 (末端扭转 {ra['exp2_torsion_90deg']['tip_twist_steady_deg']:.1f}°)",
            f"路线B PhysX GPU FEM 中空管 (末端扭转 {rb['exp2_torsion_90deg']['tip_twist_steady_deg']:.1f}°)",
        ),
        ((248, 113, 113), (56, 189, 248), (34, 197, 94)),
        "实验 2 · 左端 +90° 轴向扭转力矩沿管身传递三方对照（观察红白指针与纵纹）",
        vids_dir / "hose_v10_exp2_torsion_3way.mp4",
    )
    v_e3 = _compose_3way_mp4(
        (Path(base["videos"]["exp3"]), Path(ra["videos"]["exp3"]), Path(rb["videos"]["exp3"])),
        (
            f"Baseline 珠链 (自锁={base['exp3_barb_fit']['self_lock_hang_passed']}, 掉落失败)",
            f"路线A 微弹簧套筒 (套深 {ra['exp3_barb_fit']['insertion_depth_hang_mm']:.1f}mm · 箍力 {ra['exp3_barb_fit']['peak_cuff_radial_force_n']:.1f}N)",
            f"路线B GPU FEM 中空管 (套深 {rb['exp3_barb_fit']['insertion_depth_hang_mm']:.1f}mm · 胀大 {rb['exp3_barb_fit']['peak_fem_hoop_strain_pct']:.1f}%)",
        ),
        ((248, 113, 113), (56, 189, 248), (34, 197, 94)),
        "实验 3 · 机械臂捏管套入玻璃刺嘴 + 撒手悬吊自锁（零FixedJoint）+ 拔管 + 真实水龙头/化验水槽",
        vids_dir / "hose_v10_exp3_barb_fit_3way.mp4",
    )
    v_cf = _compose_3way_mp4(
        (Path(ra_z["videos"]["exp3"]), Path(rb_z["videos"]["exp3"]), Path(rb_w["videos"]["exp3"])),
        (
            f"路线A mu=0 零摩擦反事实 (自锁={ra_z['exp3_barb_fit']['self_lock_hang_passed']} 滑脱证伪)",
            f"路线B FEM mu=0 零摩擦反事实 (自锁={rb_z['exp3_barb_fit']['self_lock_hang_passed']} 滑脱证伪)",
            f"SOP 接反测试：进水管误接上口 (检出={not rb_w['exp3_barb_fit']['flow_bottom_in_top_out_ok']})",
        ),
        ((251, 146, 60), (251, 146, 60), (192, 132, 252)),
        "实验 4 · 物理反事实证伪（mu=0 零摩擦必滑脱）& 化学 SOP『上进下出接反』自动判错检出",
        vids_dir / "hose_v10_counterfactuals_3way.mp4",
    )

    tb = np.load(Path(base["telemetry_npz"]))
    ta = np.load(Path(ra["telemetry_npz"]))
    tbm = np.load(Path(rb["telemetry_npz"]))
    svg1 = _svg_multi_curve(
        [
            ("Baseline 珠链", (0.90 - tb["b1_tip_z"]) * 1000.0, "#f87171"),
            ("路线A D6-DER", (0.90 - ta["b1_tip_z"]) * 1000.0, "#38bdf8"),
            ("路线B GPU FEM", (0.90 - tbm["b1_tip_z"]) * 1000.0, "#22c55e"),
        ],
        "Exp-1 悬臂末端下挠位移曲线（越小越挺括，f=96 后回弹）",
        "mm",
    )
    svg2 = _svg_multi_curve(
        [
            ("Baseline 珠链", tb["b2_tip_twist_deg"], "#f87171"),
            ("路线A D6-DER", ta["b2_tip_twist_deg"], "#38bdf8"),
            ("路线B GPU FEM", tbm["b2_tip_twist_deg"], "#22c55e"),
        ],
        "Exp-2 右端自由指针实测跟随扭转角（目标 +90°）",
        "deg",
    )
    svg3 = _svg_multi_curve(
        [
            ("路线A 方案B四球微弹簧箍紧力 (N)", ta["b3_cuff_rad_force_n"], "#38bdf8"),
            ("路线B FEM 管口环向撑大应变 (%)", tbm["b3_hoop_strain_pct"], "#22c55e"),
        ],
        "Exp-3 软管套上玻璃刺嘴期间真实径向过盈响应",
        "N / %",
    )

    out_html = out_dir / "hose_softbody_v10_3way_ab_report_20261125.html"
    rows = []
    for lbl, r in (
        ("Baseline (v0.9 球铰珠链)", base),
        ("路线 A (D6-DER + 微弹簧刺嘴套筒)", ra),
        ("路线 B (自适应中空四面体 GPU FEM)", rb),
        ("反事实 · 路线 A (mu = 0.0)", ra_z),
        ("反事实 · 路线 B FEM (mu = 0.0)", rb_z),
        ("SOP接反 · 路线 A (误接上出水口)", ra_w),
        ("SOP接反 · 路线 B (误接上出水口)", rb_w),
    ):
        e1, e2, e3 = r["exp1_cantilever"], r["exp2_torsion_90deg"], r["exp3_barb_fit"]
        rows.append(
            f"<tr><td><b>{html.escape(lbl)}</b></td>"
            f"<td>{e1['sag_eq_mm']:.1f} mm / {e1['elastic_rebound_pct']:.0f}%</td>"
            f"<td>{e2['tip_twist_steady_deg']:.1f}° ({e2['transmission_ratio_pct']:.0f}%)</td>"
            f"<td>{e3['insertion_depth_hang_mm']:.1f} mm / z误差 {e3['hang_z_err_mm']:.2f} mm</td>"
            f"<td>{'PASS' if e3['self_lock_hang_passed'] else 'FAIL'} / {'PASS' if e3['pull_off_passed'] else 'FAIL'}</td>"
            f"<td>{e3['peak_cuff_radial_force_n']:.1f} N / 胀大 {e3['peak_fem_hoop_strain_pct']:.1f}% / 捏扁 {e3['peak_fem_pinch_ovalization_pct']:.1f}%</td>"
            f"<td>{e3['detected_port']} ({'合规下进' if e3['flow_bottom_in_top_out_ok'] else '拦截/接反'})</td>"
            f"<td>{r['mean_frame_physx_ms']:.1f} ms</td></tr>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<title>Hose Softbody v1.0 · 三方同条件 240Hz PhysX 5 软管物理对比报告（旧珠链 vs 路线A D6-DER vs 路线B GPU FEM）</title>
<style>
  body {{ margin:0; font-family:"Microsoft YaHei",system-ui,sans-serif; background:#070d19; color:#f8fafc; line-height:1.55; }}
  .wrap {{ max-width:1520px; margin:0 auto; padding:24px; }}
  .hero {{ background:linear-gradient(135deg,#0f172a,#0c243d); border:1px solid #1d4ed8; border-radius:16px; padding:22px 26px; margin-bottom:22px; }}
  .hero h1 {{ margin:0 0 8px 0; font-size:24px; }}
  .muted {{ color:#94a3b8; font-size:13px; }}
  .card {{ background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:14px; margin-bottom:18px; }}
  .card-hdr {{ font-weight:700; font-size:15px; margin-bottom:8px; color:#e2e8f0; }}
  video {{ width:100%; border-radius:10px; background:#000; border:1px solid #334155; }}
  .grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ border:1px solid #1e293b; padding:9px 10px; text-align:left; }}
  th {{ background:#0b1325; color:#38bdf8; }}
</style></head><body><div class="wrap">
  <div class="hero">
    <h1>蒸馏冷却软管物理升级实测报告 · 旧珠链 v0.9 vs 路线 A（D6-DER 弹性杆+微弹簧套筒）vs 路线 B（自适应中空四面体 PhysX GPU FEM）</h1>
    <div class="muted">7 个独立冷启动 OS 进程（PIDs = {summary['distinct_cold_start_pids']}）· 240 Hz PhysX 5 GPU · Exp-3 零 FixedJoint · 含实验室鹅颈水龙头 &amp; PP 化验水槽 &amp; 上进下出接反自动检出</div>
  </div>
  <div class="card">
    <div class="card-hdr">定量对比总表（同 240 Hz GPU 预算 · RTX 4090）</div>
    <table><thead><tr>
      <th>方案 / 试验组</th><th>Exp-1 悬臂挠度 / 弹性回弹</th><th>Exp-2 90°扭转传递</th><th>Exp-3 撒手悬吊套深 / 偏心</th><th>撒手自锁 / 拔管</th><th>径向过盈力 / FEM撑大 / 捏扁</th><th>冷却水口 SOP 判别</th><th>PhysX 单帧耗时</th>
    </tr></thead><tbody>{"".join(rows)}</tbody></table>
  </div>
  <div class="grid-3">{svg1}{svg2}{svg3}</div>
  <div class="card"><div class="card-hdr">1. Exp-1 悬臂自然下挠与阶跃拨动弹性回弹（三方同框同步）</div><video controls loop muted playsinline src="{_b64_uri(v_e1, 'video/mp4')}"></video></div>
  <div class="card"><div class="card-hdr">2. Exp-2 90° 轴向扭转力矩传递（观察红白指针与 Cosserat 纵纹螺旋）</div><video controls loop muted playsinline src="{_b64_uri(v_e2, 'video/mp4')}"></video></div>
  <div class="card"><div class="card-hdr">3. Exp-3 机械臂真捏管套玻璃刺嘴 + 撒手悬吊自锁 + 拔管（零 FixedJoint）</div><video controls loop muted playsinline src="{_b64_uri(v_e3, 'video/mp4')}"></video></div>
  <div class="card"><div class="card-hdr">4. 反事实证伪（mu=0 零摩擦滑脱）&amp; 进水管误接上口（上进下出接反检出）</div><video controls loop muted playsinline src="{_b64_uri(v_cf, 'video/mp4')}"></video></div>
</div></body></html>"""
    out_html.write_text(doc, encoding="utf-8")
    res = {
        "out_html": out_html.as_posix(),
        "out_html_mb": round(out_html.stat().st_size / (1024 * 1024), 2),
        "videos": {
            "exp1": v_e1.as_posix(),
            "exp2": v_e2.as_posix(),
            "exp3": v_e3.as_posix(),
            "counterfactuals": v_cf.as_posix(),
        },
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    build_hose_v10_3way_report()

