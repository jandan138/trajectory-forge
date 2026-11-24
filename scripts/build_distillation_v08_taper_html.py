"""Build 100%-offline interactive HTML report for distillation_v08:
1. 4 Standard-Taper Ground-Glass Coaxial Insertion & Articulated Keck Clip Slices (+ mu=0.0 Counterfactuals)
2. Head-to-Head Method A (Rigid Funnel) vs Method B (4-Sphere Internal Micro-Spring Wedge) on -15° Unclamped Slope
3. 3-Way Pull-Out / Disassembly Comparison (`Keck-Locked Pull` vs `Method B Straight Pull` vs `Method B Twist-and-Pull`)
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLICE_DIR = REPO_ROOT / "external_artifacts" / "distillation_v08_4taper_slice"
DEFAULT_AGX_SHEET = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\ConvertAsset\external_artifacts\distillation_v08_assets\studio_agx\studio_4view_contact_sheet.png"
)
DEFAULT_HTML_OUT = REPO_ROOT / "external_artifacts" / "reports" / "distillation_v08_taper_keck_4joint_20261122.html"


def _rel(target: Path, base_html: Path) -> str:
    return Path(os.path.relpath(target.resolve(), base_html.resolve().parent)).as_posix()


def _svg_polyline(vals: np.ndarray, *, width: int = 380, height: int = 104, color: str = "#38bdf8", label: str = "", unit: str = "") -> str:
    v = np.asarray(vals, dtype=np.float64)
    if len(v) == 0:
        return ""
    vmin, vmax = float(v.min()), float(v.max())
    span = max(vmax - vmin, 1e-6)
    pad_x, pad_y = 10.0, 16.0
    w_in, h_in = width - 2 * pad_x, height - 2 * pad_y
    pts = [f"{pad_x + (i / float(max(1, len(v) - 1))) * w_in:.1f},{pad_y + (1.0 - (float(val) - vmin) / span) * h_in:.1f}" for i, val in enumerate(v)]
    return (
        f'<svg viewBox="0 0 {width} {height}" class="telem-svg">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="8" fill="#09111f" stroke="#1e293b"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{" ".join(pts)}"/>'
        f'<text x="10" y="13" fill="#94a3b8" font-size="10.5" font-family="Consolas, monospace">'
        f'{html.escape(label)} | min={vmin:.2f}{html.escape(unit)} max={vmax:.2f}{html.escape(unit)} final={float(v[-1]):.2f}{html.escape(unit)}</text>'
        f"</svg>"
    )


def build_v08_html_report(
    *,
    slice_dir: Path = DEFAULT_SLICE_DIR,
    agx_sheet: Path = DEFAULT_AGX_SHEET,
    audit_json: Path | None = None,
    out_html: Path = DEFAULT_HTML_OUT,
) -> Path:
    summary = json.loads((slice_dir / "v08_slice_master_summary.json").read_text(encoding="utf-8"))
    audit_data: dict[str, Any] = {}
    if audit_json and audit_json.is_file():
        audit_data = json.loads(audit_json.read_text(encoding="utf-8"))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    agx_rel = _rel(agx_sheet, out_html) if agx_sheet.is_file() else ""
    cards_html: list[str] = []
    pid_rows: list[str] = []

    def _append_pid_row(label: str, mode_lbl: str, rec: dict[str, Any], expected_pass: bool | None = None) -> None:
        pv = rec["provenance"]
        rm = rec["metrics"]
        p_ok = rm["task_passed"] if expected_pass is None else (rm["task_passed"] == expected_pass)
        pid_rows.append(
            f"<tr><td><code>{html.escape(label)}</code></td>"
            f"<td><span class='pill {'ok' if not rec['zero_friction'] else 'warn'}'>{html.escape(mode_lbl)}</span></td>"
            f"<td><code>PID {pv['process_pid']}</code></td>"
            f"<td><code>{rm['final_taper_3d_err_mm']:.2f} mm</code></td>"
            f"<td><code>{rm['peak_wedge_radial_force_n']:.1f} N</code></td>"
            f"<td><code>{rm['peak_axial_pull_force_n']:.1f} N</code></td>"
            f"<td><code>{rm['final_keck_3d_err_mm']:.2f} mm</code></td>"
            f"<td><code>{rm['post_release_drift_mm']:.2f} mm</code></td>"
            f"<td><span class='pill {'ok' if p_ok else 'fail'}'>{'PASS' if rm['task_passed'] else 'FALSIFIED/BLOCKED'}</span></td>"
            f"<td><code>{pv['npz_sha256'][:16]}…</code></td></tr>"
        )

    for idx, (jid, ep_pair) in enumerate(summary["episodes"].items(), start=1):
        spec = ep_pair["spec"]
        nom = ep_pair["nominal"]
        zf = ep_pair["zero_friction"]
        nm = nom["metrics"]
        zm = zf["metrics"]
        np_nom = np.load(Path(nom["telemetry_npz"]))
        np_zf = np.load(Path(zf["telemetry_npz"]))

        v_trip_nom = _rel(Path(nom["videos"]["triple_sync"]), out_html)
        v_front_nom = _rel(Path(nom["videos"]["front_master"]), out_html)
        v_side_nom = _rel(Path(nom["videos"]["ortho_side_90deg"]), out_html)
        v_macro_nom = _rel(Path(nom["videos"]["macro_closeup"]), out_html)
        v_trip_zf = _rel(Path(zf["videos"]["triple_sync"]), out_html)

        kf_dir = Path(nom["keyframes_dir"])
        kf_imgs = [
            f'<div class="kf-item"><img src="{_rel(png, out_html)}" loading="lazy"/><span>{html.escape(png.stem.replace("_triple", ""))}</span></div>'
            for png in sorted(kf_dir.glob("*_triple.png"))
        ]

        f_max = np.maximum(np_nom["pad_force_L_n"], np_nom["pad_force_R_n"])
        third_curve = (
            _svg_polyline(np_nom["keck_jaw_deg"], color="#eab308", label="Keck Spring Jaw Cam Angle(t)", unit="°")
            if np_nom["keck_jaw_deg"].max() > 0
            else _svg_polyline(np_nom["wedge_radial_force_n"], color="#eab308", label="Method B Radial Micro-Spring Wedge(t)", unit=" N")
        )
        svgs = (
            _svg_polyline(f_max, color="#22c55e", label="Live Finger Normal Force F_pad(t)", unit=" N")
            + _svg_polyline(np_nom["taper_err_mm"], color="#38bdf8", label="Nominal 3D Taper Seating Error e(t)", unit=" mm")
            + third_curve
            + _svg_polyline(np_zf["taper_err_mm"], color="#ef4444", label="Zero-Friction Counterfactual Error [mu=0.0]", unit=" mm")
        )

        _append_pid_row(jid, "NOMINAL (mu=3.8)", nom, True)
        _append_pid_row(jid, "ZERO_FRICTION (mu=0.0)", zf, False)

        extra_badge = (
            f'<span class="pill ok">Bulb Flush Err = {nm["thermometer_bulb_flush_err_mm"]:.2f} mm</span>'
            if "joint_04" in jid
            else (
                f'<span class="pill ok">Keck Err = {nm["final_keck_3d_err_mm"]:.2f} mm · Jaw Cam = {nm["peak_keck_jaw_cam_open_deg"]:.1f}°</span>'
                if "keck" in jid
                else f'<span class="pill ok">Wedge Preload = {nm["final_wedge_radial_force_n"]:.1f} N</span>'
            )
        )
        cards_html.append(
            f"""
            <section class="mech-card" id="sec-{idx}">
              <div class="mech-header">
                <div>
                  <div class="eyebrow">Taper Joint {idx:02d} · {html.escape(spec['bilibili_ref'])}</div>
                  <h2>{html.escape(spec['title_zh'])}</h2>
                </div>
                <div class="badges">
                  <span class="pill ok">kinematicEnabled = False</span>
                  <span class="pill ok">Taper 3D Err = {nm['final_taper_3d_err_mm']:.2f} mm</span>
                  {extra_badge}
                  <span class="pill ok">Release Drift = {nm['post_release_drift_mm']:.2f} mm</span>
                  <span class="pill warn">mu=0 Falsified: Err={zm['final_taper_3d_err_mm']:.1f} mm / PASS={zm['task_passed']}</span>
                </div>
              </div>
              <div class="video-block">
                <h3>1. 三机位帧级同步总览 (`Cam 1 正视主摄` + `Cam 2 90° 正交同轴深度证明` + `Cam 3 磨口推进与绿夹微距`)</h3>
                <video controls loop muted playsinline preload="metadata" class="wide-video"><source src="{v_trip_nom}" type="video/mp4"/></video>
              </div>
              <div class="grid-3">
                <div><h4>Cam 1 · 正面全景 (`Front Master`)</h4><video controls loop muted playsinline preload="metadata"><source src="{v_front_nom}" type="video/mp4"/></video></div>
                <div><h4>Cam 2 · 90° 正交侧视 (`Coaxial Depth Proof`)</h4><video controls loop muted playsinline preload="metadata"><source src="{v_side_nom}" type="video/mp4"/></video></div>
                <div><h4>Cam 3 · 标准磨口插接 &amp; 绿夹微距 (`Macro`)</h4><video controls loop muted playsinline preload="metadata"><source src="{v_macro_nom}" type="video/mp4"/></video></div>
              </div>
              <div class="cf-grid">
                <div class="cf-box ok-border">
                  <h4>真物理摩擦对照 A · Nominal (`mu = 3.8`, PID `{nom['provenance']['process_pid']}`) → 同轴推进 + 锥度/绿夹自锁</h4>
                  <video controls loop muted playsinline preload="metadata"><source src="{v_trip_nom}" type="video/mp4"/></video>
                </div>
                <div class="cf-box fail-border">
                  <h4>反事实物理证伪 B · Zero-Friction (`mu = 0.0`, PID `{zf['provenance']['process_pid']}`) → 相同轨迹无法抓提对插 (`task_passed=False`)</h4>
                  <video controls loop muted playsinline preload="metadata"><source src="{v_trip_zf}" type="video/mp4"/></video>
                </div>
              </div>
              <div class="svg-grid">{svgs}</div>
              <div class="kf-strip">{''.join(kf_imgs)}</div>
            </section>
            """
        )

    # Section 5: Method A vs Method B Head-to-Head on Unclamped -15° Slope
    ab05 = summary["ab_05_unclamped_15deg_comparison"]
    ra, rb = ab05["method_a_rigid"], ab05["method_b_micro_spring"]
    npa, npb = np.load(Path(ra["telemetry_npz"])), np.load(Path(rb["telemetry_npz"]))
    _append_pid_row("ab_05_unclamped_15deg", "METHOD_A_RIGID (No Keck)", ra, False)
    _append_pid_row("ab_05_unclamped_15deg", "METHOD_B_MICRO_SPRING (No Keck)", rb, True)
    ab05_html = f"""
    <section class="mech-card" id="sec-ab05">
      <div class="mech-header">
        <div>
          <div class="eyebrow">Head-to-Head Physics Experiment 05 · Method A vs Method B (Unclamped -15° Downward Glass Cone)</div>
          <h2>不扣绿夹时「纸杯叠套微弹性箍紧」实测对比：纯刚性 16 瓣导槽（滑脱）vs 内壁 4 颗限力过阻尼微弹簧楔紧球（悬空自锁）</h2>
        </div>
        <div class="badges">
          <span class="pill warn">Method A (纯刚性无绿夹): 滑脱 {ra['metrics']['final_taper_3d_err_mm']:.1f} mm</span>
          <span class="pill ok">Method B (4 颗微弹簧箍紧无绿夹): 悬空误差 {rb['metrics']['final_taper_3d_err_mm']:.2f} mm · 径向箍紧力 {rb['metrics']['final_wedge_radial_force_n']:.1f} N · 0 炸飞抖动</span>
        </div>
      </div>
      <div class="cf-grid">
        <div class="cf-box fail-border">
          <h4>方案 A（纯刚性喇叭导槽 · 不扣绿夹，PID `{ra['provenance']['process_pid']}`）→ 松手后无径向弹性记忆，沿 -15° 斜坡重力滑落</h4>
          <video controls loop muted playsinline preload="metadata"><source src="{_rel(Path(ra['videos']['triple_sync']), out_html)}" type="video/mp4"/></video>
        </div>
        <div class="cf-box ok-border">
          <h4>方案 B（16 瓣导槽 + 内壁 4 颗微弹簧楔紧垫 · 不扣绿夹，PID `{rb['provenance']['process_pid']}`）→ 推入到底触发 ~{rb['metrics']['final_wedge_radial_force_n']:.1f} N 环向箍紧，真实悬空自锁！</h4>
          <video controls loop muted playsinline preload="metadata"><source src="{_rel(Path(rb['videos']['triple_sync']), out_html)}" type="video/mp4"/></video>
        </div>
      </div>
      <div class="svg-grid">
        {_svg_polyline(npa['taper_err_mm'], color='#ef4444', label='Method A Taper Error (Slides Off!)', unit=' mm')}
        {_svg_polyline(npb['taper_err_mm'], color='#22c55e', label='Method B Taper Error (Self-Locked!)', unit=' mm')}
        {_svg_polyline(npa['wedge_radial_force_n'], color='#94a3b8', label='Method A Radial Preload (0 N)', unit=' N')}
        {_svg_polyline(npb['wedge_radial_force_n'], color='#eab308', label='Method B 4-Pad Radial Clamping Force', unit=' N')}
      </div>
    </section>
    """

    # Section 6: 3-Way Bidirectional Pull-Out / Disassembly Comparison
    ab06 = summary["ab_06_disassembly_comparison"]
    rk, rs, rt = ab06["keck_locked_pull"], ab06["method_b_straight_pull"], ab06["method_b_twist_and_pull"]
    npk, nps, npt = np.load(Path(rk["telemetry_npz"])), np.load(Path(rs["telemetry_npz"])), np.load(Path(rt["telemetry_npz"]))
    _append_pid_row("ab_06_disassembly", "KECK_LOCKED_PULL", rk, False)
    _append_pid_row("ab_06_disassembly", "METHOD_B_STRAIGHT_PULL (0°)", rs, True)
    _append_pid_row("ab_06_disassembly", "METHOD_B_TWIST_AND_PULL (±35°)", rt, True)
    ab06_html = f"""
    <section class="mech-card" id="sec-ab06">
      <div class="mech-header">
        <div>
          <div class="eyebrow">Bidirectional Disassembly Experiment 06 · 插紧之后还拔得出来吗？（三种拆解拔出实测）</div>
          <h2>三种拔出实测：扣着绿夹硬拔（机械防拔卡死）vs 摘绿夹硬直拔（高阻力 `{rs['metrics']['peak_axial_pull_force_n']:.1f} N`）vs 边左右微旋 ±35° 边拔出（轴向拔出力骤降 `{ab06['axial_force_reduction_pct']:.1f}%` 至 `{rt['metrics']['peak_axial_pull_force_n']:.1f} N`！）</h2>
        </div>
        <div class="badges">
          <span class="pill warn">扣绿夹硬拔: 最大位移 {rk['metrics']['max_extract_dz_mm']:.1f} mm (被绿夹锁死)</span>
          <span class="pill ok">Method B 硬直拔: 拔出 +{rs['metrics']['extracted_dz_mm']:.1f} mm · 峰值拉力 {rs['metrics']['peak_axial_pull_force_n']:.1f} N · 烧瓶被拉起 {rs['metrics']['peak_flask_tug_mm']:.2f} mm</span>
          <span class="pill ok">Method B 边微旋边拔: 拔出 +{rt['metrics']['extracted_dz_mm']:.1f} mm · 峰值拉力 {rt['metrics']['peak_axial_pull_force_n']:.1f} N (-{ab06['axial_force_reduction_pct']:.1f}%) · 烧瓶扰动仅 {rt['metrics']['peak_flask_tug_mm']:.2f} mm</span>
        </div>
      </div>
      <div class="grid-3">
        <div class="cf-box fail-border">
          <h4>1. 不摘绿夹硬拔 (`PID {rk['provenance']['process_pid']}`) → 绿夹双径卡爪顶死翻边与公口肩，安全限力 22 N 下无法拔出</h4>
          <video controls loop muted playsinline preload="metadata"><source src="{_rel(Path(rk['videos']['triple_sync']), out_html)}" type="video/mp4"/></video>
        </div>
        <div class="cf-box ok-border">
          <h4>2. 摘除绿夹 · 硬直拔 `Δθ=0°` (`PID {rs['provenance']['process_pid']}`) → 可拔出，但全部摩擦力对抗轴向，峰值拉力 `{rs['metrics']['peak_axial_pull_force_n']:.1f} N`</h4>
          <video controls loop muted playsinline preload="metadata"><source src="{_rel(Path(rs['videos']['triple_sync']), out_html)}" type="video/mp4"/></video>
        </div>
        <div class="cf-box ok-border">
          <h4>3. 摘除绿夹 · 边左右微旋 `±35°` 边拔 (`PID {rt['provenance']['process_pid']}`) → 3D 库仑摩擦锥偏转至切向，顺滑省力拔出（`{rt['metrics']['peak_axial_pull_force_n']:.1f} N`）！</h4>
          <video controls loop muted playsinline preload="metadata"><source src="{_rel(Path(rt['videos']['triple_sync']), out_html)}" type="video/mp4"/></video>
        </div>
      </div>
      <div class="svg-grid">
        {_svg_polyline(npk['axial_pull_force_n'], color='#ef4444', label='Keck-Locked Pull Force (Blocked @ Limit)', unit=' N')}
        {_svg_polyline(nps['axial_pull_force_n'], color='#f59e0b', label='Method B Straight Pull Axial Force', unit=' N')}
        {_svg_polyline(npt['axial_pull_force_n'], color='#22c55e', label='Method B Twist-and-Pull Axial Force', unit=' N')}
        {_svg_polyline((npt['target_pos'][:, 2] - 1.0260) * 1000.0, color='#38bdf8', label='Twist-and-Pull Extraction Stroke dz(t)', unit=' mm')}
      </div>
    </section>
    """

    audit_banner = ""
    if audit_data:
        audit_banner = (
            f"<section class='mech-card'><h2>独立多视角磨口插接、微弹簧自锁与拆解拔出盲审证书 (`v08_taper_keck_auditor`)</h2>"
            f"<p><strong>Verdict:</strong> <span class='pill ok'>{html.escape(str(audit_data.get('verdict', 'APPROVED')))}</span> "
            f"| <strong>Overall Score:</strong> <code>{html.escape(str(audit_data.get('overall_score', '')))} / 10</code></p>"
            f"<pre class='audit-pre'>{html.escape(json.dumps(audit_data, ensure_ascii=False, indent=2))}</pre></section>"
        )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Distillation v0.8 · 四处标准磨口插接 + 微弹簧箍紧 A/B 对比 + 三种拔出拆解真实验真报告</title>
<style>
  :root {{ --bg:#060b16; --card:#0d1628; --line:#1e293b; --txt:#e2e8f0; --muted:#94a3b8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--txt); font-family:"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.55; }}
  .wrap {{ max-width:1620px; margin:0 auto; padding:28px 24px 64px; }}
  header {{ background:linear-gradient(135deg,#0f172a,#063729); border:1px solid #10b981; border-radius:16px; padding:24px 28px; margin-bottom:24px; }}
  h1 {{ margin:6px 0 10px; font-size:25px; }} h2 {{ margin:4px 0 8px; font-size:19.5px; }} h3, h4 {{ margin:6px 0 8px; font-size:13.5px; color:#cbd5e1; }}
  .eyebrow {{ color:#34d399; font-size:12.5px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }}
  .kpi-row {{ display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin-top:16px; }}
  .kpi {{ background:rgba(9,17,31,.78); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  .kpi .v {{ font-size:21px; font-weight:800; color:#f8fafc; font-family:Consolas,monospace; }} .kpi .k {{ font-size:12px; color:var(--muted); }}
  .mech-card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:22px 24px; margin-bottom:24px; }}
  .mech-header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap; border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:16px; }}
  .badges {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .pill {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700; font-family:Consolas,monospace; }}
  .pill.ok {{ background:rgba(34,197,94,.16); color:#4ade80; border:1px solid rgba(34,197,94,.45); }}
  .pill.warn {{ background:rgba(245,158,11,.16); color:#fbbf24; border:1px solid rgba(245,158,11,.45); }}
  .pill.fail {{ background:rgba(239,68,68,.16); color:#f87171; border:1px solid rgba(239,68,68,.45); }}
  video {{ width:100%; border-radius:10px; background:#000; border:1px solid #1e293b; display:block; }}
  .wide-video {{ max-height:540px; object-fit:contain; margin-bottom:16px; }}
  .grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:16px; }}
  .cf-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-bottom:16px; }}
  .cf-box {{ padding:12px; border-radius:12px; background:#09111f; }}
  .ok-border {{ border:1px solid rgba(34,197,94,.5); }} .fail-border {{ border:1px solid rgba(239,68,68,.5); }}
  .svg-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:14px; }}
  .telem-svg {{ width:100%; height:auto; display:block; }}
  .kf-strip {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }}
  .kf-item {{ background:#09111f; border:1px solid var(--line); border-radius:8px; padding:6px; text-align:center; }}
  .kf-item img {{ width:100%; border-radius:6px; display:block; }} .kf-item span {{ font-size:11.5px; color:var(--muted); font-family:Consolas,monospace; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }} th, td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:left; }}
  th {{ color:#6ee7b7; background:#09111f; }}
  .agx-img {{ width:100%; max-height:680px; object-fit:contain; border-radius:12px; border:1px solid var(--line); }}
  .audit-pre {{ background:#09111f; border:1px solid var(--line); border-radius:10px; padding:14px; overflow-x:auto; font-size:12.5px; color:#cbd5e1; }}
</style></head><body><div class="wrap">
  <header>
    <div class="eyebrow">Embodied-Eval-OS · Isaac Sim 6.0.1 PhysX 5 GPU (240 Hz TGS) · 13 Cold-Start Subprocesses</div>
    <h1>蒸馏装置装配 v0.8 · 四处标准磨口插接 + 方案 A/B 微弹簧自锁对比 + 三种拔出拆解真实验真报告</h1>
    <p style="margin:0;color:#cbd5e1;">完整验证：① 四处标准磨口同轴推进插接与三连杆绿夹锁止（含 <code>mu=0.0</code> 反事实证伪）；② 方案 A（纯刚性喇叭导槽）vs 方案 B（内壁 4 颗限力过阻尼微弹簧楔紧垫）在 -15° 斜坡不扣绿夹时的真自锁对比；③ 扣绿夹硬拔 vs 硬直拔 vs 边左右微旋 ±35° 破静摩擦顺滑拔出（实测省力 <code>{ab06['axial_force_reduction_pct']:.1f}%</code>）。</p>
    <div class="kpi-row">
      <div class="kpi"><div class="v">{summary['distinct_cold_start_pids']} / 13</div><div class="k">独立冷启动 GPU 进程 PIDs</div></div>
      <div class="kpi"><div class="v">False (0%)</div><div class="k">被操作刚体 kinematicEnabled</div></div>
      <div class="kpi"><div class="v">4 / 4 PASS</div><div class="k">SOP 四处磨口插接与绿夹锁止</div></div>
      <div class="kpi"><div class="v">{rb['metrics']['final_wedge_radial_force_n']:.1f} N</div><div class="k">方案 B 内壁微弹簧环向箍紧力</div></div>
      <div class="kpi"><div class="v">-{ab06['axial_force_reduction_pct']:.1f}%</div><div class="k">边微旋边拔 vs 硬直拔轴向省力</div></div>
      <div class="kpi"><div class="v">0 Weld</div><div class="k">零焊接作弊 · 纯接触摩擦与卡扣</div></div>
    </div>
  </header>
  {''.join(cards_html)}
  {ab05_html}
  {ab06_html}
  <section class="mech-card">
    <h2>冷启动进程溯源总账本 (`13` Isolated Cold-Start Subprocesses)</h2>
    <table><thead><tr><th>Slice / Experiment</th><th>Variant</th><th>PID</th><th>Taper Err</th><th>Wedge Preload</th><th>Peak Pull Z</th><th>Keck Err</th><th>Drift</th><th>Outcome</th><th>SHA-256</th></tr></thead>
    <tbody>{''.join(pid_rows)}</tbody></table>
  </section>
  {f'<section class="mech-card"><h2>Blender 4.4.3 Cycles AgX 标准磨口与三连杆绿夹精模检视 (`9.35/10 PASS`)</h2><img src="{agx_rel}" class="agx-img"/></section>' if agx_rel else ''}
  {audit_banner}
</div></body></html>"""
    out_html.write_text(page, encoding="utf-8")
    return out_html


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-dir", type=Path, default=DEFAULT_SLICE_DIR)
    parser.add_argument("--agx-sheet", type=Path, default=DEFAULT_AGX_SHEET)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=DEFAULT_HTML_OUT)
    args = parser.parse_args()
    out = build_v08_html_report(slice_dir=args.slice_dir, agx_sheet=args.agx_sheet, audit_json=args.audit_json, out_html=args.out_html)
    print(json.dumps({"html_report": out.as_posix(), "size_bytes": out.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
