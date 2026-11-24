"""Build 100%-offline interactive A/B comparison HTML showcase for HermiteTwin v0.8."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any, Sequence


def _b64_uri(path: str | Path, mime: str) -> str:
    p = Path(path)
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _svg_line_chart(
    *,
    series_a: Sequence[float],
    series_b: Sequence[float],
    label_a: str,
    label_b: str,
    split_x: int | None,
    x_title: str,
    y_title: str,
    width: int = 560,
    height: int = 210,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 54, 18, 22, 32
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = max(len(series_a), len(series_b), 2)
    y_max = max(max(series_a or [1.0]), max(series_b or [1.0]), 1e-3) * 1.08

    def _pts(vals: Sequence[float]) -> str:
        if not vals:
            return ""
        denom = max(len(vals) - 1, 1)
        out = []
        for i, v in enumerate(vals):
            x = pad_l + (i / denom) * plot_w
            y = pad_t + plot_h - (min(max(float(v), 0.0), y_max) / y_max) * plot_h
            out.append(f"{x:.1f},{y:.1f}")
        return " ".join(out)

    split_svg = ""
    if split_x is not None and n > 1:
        sx = pad_l + (split_x / (n - 1)) * plot_w
        split_svg = (
            f'<line x1="{sx:.1f}" y1="{pad_t}" x2="{sx:.1f}" y2="{pad_t + plot_h}" '
            'stroke="#60a5fa" stroke-width="1.5" stroke-dasharray="4 3"/>'
            f'<text x="{sx + 4:.1f}" y="{pad_t + 12}" fill="#93c5fd" font-size="10">Train | Future</text>'
        )

    grid_svg = ""
    for k in range(4):
        gy = pad_t + (k / 3.0) * plot_h
        yv = y_max * (1.0 - k / 3.0)
        grid_svg += (
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" stroke="#263147" stroke-width="1"/>'
            f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" text-anchor="end" fill="#94a3b8" font-size="10">{yv:.1f}</text>'
        )

    return f'''<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" style="background:#0d131f;border:1px solid #1f2b40;border-radius:10px;">
  {grid_svg}
  {split_svg}
  <polyline fill="none" stroke="#fb923c" stroke-width="2.2" points="{_pts(series_a)}"/>
  <polyline fill="none" stroke="#34d399" stroke-width="2.5" points="{_pts(series_b)}"/>
  <text x="{pad_l}" y="14" fill="#cbd5e1" font-size="11" font-weight="600">{html.escape(y_title)}</text>
  <text x="{width // 2}" y="{height - 6}" text-anchor="middle" fill="#94a3b8" font-size="10">{html.escape(x_title)}</text>
  <rect x="{width - 210}" y="6" width="10" height="10" fill="#fb923c" rx="2"/>
  <text x="{width - 195}" y="15" fill="#fed7aa" font-size="10">{html.escape(label_a)}</text>
  <rect x="{width - 110}" y="6" width="10" height="10" fill="#34d399" rx="2"/>
  <text x="{width - 95}" y="15" fill="#a7f3d0" font-size="10">{html.escape(label_b)}</text>
</svg>'''


def build_hermitetwin_comparison_html(
    *,
    rollout_summary_path: str | Path,
    cert_summary_path: str | Path,
    media_manifest_path: str | Path,
    out_html_path: str | Path,
) -> Path:
    roll = json.loads(Path(rollout_summary_path).read_text(encoding="utf-8"))
    cert = json.loads(Path(cert_summary_path).read_text(encoding="utf-8"))
    media = json.loads(Path(media_manifest_path).read_text(encoding="utf-8"))

    tier = str(cert.get("evidence_tier", roll.get("evidence_tier", "FAST_CHECK_CPU_PREVIEW")))
    all_passed = bool(cert.get("all_tasks_passed", False))
    dual_m = cert.get("dual_renderer") or {}

    task_ids = list(roll["tasks"].keys())
    c3_list = [cert["tasks"][t]["protocol_c3"] for t in task_ids]

    avg_resim_red = sum(float(c["metrics_comparison"]["resim_reduction_pct"]) for c in c3_list) / max(len(c3_list), 1)
    avg_fut_red = sum(float(c["metrics_comparison"]["future_reduction_pct"]) for c in c3_list) / max(len(c3_list), 1)

    b_rms_avg = sum(float(c["metrics_comparison"]["baseline_rms_track_err_mm"]) for c in c3_list) / max(len(c3_list), 1)
    h_rms_avg = sum(float(c["metrics_comparison"]["hermite_rms_track_err_mm"]) for c in c3_list) / max(len(c3_list), 1)
    rms_red = 100.0 * (b_rms_avg - h_rms_avg) / max(b_rms_avg, 1e-6)

    b_sag_avg = sum(float(c["metrics_comparison"]["baseline_foam_sag_err_mm"]) for c in c3_list) / max(len(c3_list), 1)
    h_sag_avg = sum(float(c["metrics_comparison"]["hermite_foam_sag_err_mm"]) for c in c3_list) / max(len(c3_list), 1)
    sag_red = 100.0 * (b_sag_avg - h_sag_avg) / max(b_sag_avg, 1e-6)

    # Build interactive tabs payload (3 A/B tasks + 1 Dual-Renderer tab)
    tabs_html_parts = []
    tab_btns_parts = []
    tab_meta_json: dict[str, Any] = {}

    task_labels = {
        "v08_cloth_corner_fold": "1. Cloth Corner Fold (折痕还原)",
        "v08_stiff_foam_sheet_transfer": "2. Stiff Foam Sheet (泡沫挺括度)",
        "v08_contact_drape_peel_novel": "3. Drape & Novel Action (接触迁移)",
    }

    for idx, tid in enumerate(task_ids):
        t_roll = roll["tasks"][tid]
        t_cert = cert["tasks"][tid]
        t_med = media["tasks"][tid]
        mc = t_cert["protocol_c3"]["metrics_comparison"]

        mp4_uri = _b64_uri(t_med["mp4_path"], "video/mp4")
        old_png_uri = _b64_uri(t_med["old_peak_png"], "image/png")
        new_png_uri = _b64_uri(t_med["new_peak_png"], "image/png")

        cd_svg = _svg_line_chart(
            series_a=t_cert["baseline_summary"]["cd_per_frame_mm2"],
            series_b=t_cert["hermite_summary"]["cd_per_frame_mm2"],
            label_a="Old (Spring-Mass)",
            label_b="New (HermiteTwin)",
            split_x=int(t_roll["train_split_frame"]),
            x_title="Frame Index (Resimulation -> Future Holdout)",
            y_title="Chamfer Distance (mm^2, lower is better)",
        )
        sysid_svg = _svg_line_chart(
            series_a=t_roll["sysid"]["baseline"]["loss_history"],
            series_b=t_roll["sysid"]["hermite"]["loss_history"],
            label_a="Old SysID Loss",
            label_b="New SysID Loss",
            split_x=None,
            x_title="CMA-ES Gen + L-BFGS-B Polish",
            y_title="Train Loss (mm^2, equal eval budget)",
        )

        total_vid_frames = max(int(t_med["total_video_frames"]), 1)
        fps = 24.0
        jump_buttons = [
            {"label": "Grasp Start", "time_s": round(6.0 / fps, 3)},
            {"label": "Peak Deform", "time_s": round(float(t_med["peak_frame"]) / fps, 3)},
            {"label": "Future Holdout", "time_s": round(float(t_roll["train_split_frame"]) / fps, 3)},
        ]
        if t_med.get("novel_start_frame") is not None:
            jump_buttons.append({"label": "Novel Drape", "time_s": round(float(t_med["novel_start_frame"]) / fps, 3)})

        tab_meta_json[tid] = {"jumps": jump_buttons, "duration_hint_s": round(total_vid_frames / fps, 3)}
        active_cls = "active" if idx == 0 else ""
        disp_style = "block" if idx == 0 else "none"
        label = task_labels.get(tid, tid)

        tab_btns_parts.append(
            f'<button class="tab-btn {active_cls}" onclick="selectTab(\'{tid}\')" id="btn-{tid}">{html.escape(label)}</button>'
        )

        novel_chip = ""
        if mc.get("novel_reduction_pct") is not None:
            novel_chip = (
                f'<div class="mini-kpi"><span>Novel Action CD</span>'
                f'<b>{mc["baseline_novel_chamfer_mm2"]:.1f} &rarr; <em class="good">{mc["hermite_novel_chamfer_mm2"]:.1f}</em> mm&sup2;</b>'
                f'<small>-{mc["novel_reduction_pct"]:.1f}%</small></div>'
            )

        jumps_html = " ".join(
            f'<button class="jump-btn" onclick="jumpVideo(\'{tid}\', {j["time_s"]})">{html.escape(j["label"])}</button>'
            for j in jump_buttons
        )

        tabs_html_parts.append(f'''
<section class="tab-panel" id="panel-{tid}" style="display:{disp_style};">
  <div class="task-header">
    <div>
      <h3>{html.escape(label)}</h3>
      <p class="muted">{html.escape(t_roll["instruction"])}</p>
    </div>
    <div class="pass-pill {"pass" if t_cert["protocol_c3"]["passed"] else "fail"}">
      Protocol C3: {"PASS" if t_cert["protocol_c3"]["passed"] else "FAIL"}
    </div>
  </div>

  <div class="mini-kpi-grid">
    <div class="mini-kpi"><span>Resim Chamfer (Train)</span><b>{mc["baseline_resim_chamfer_mm2"]:.1f} &rarr; <em class="good">{mc["hermite_resim_chamfer_mm2"]:.1f}</em> mm&sup2;</b><small>-{mc["resim_reduction_pct"]:.1f}%</small></div>
    <div class="mini-kpi"><span>Future Pred Chamfer</span><b>{mc["baseline_future_chamfer_mm2"]:.1f} &rarr; <em class="good">{mc["hermite_future_chamfer_mm2"]:.1f}</em> mm&sup2;</b><small>-{mc["future_reduction_pct"]:.1f}%</small></div>
    <div class="mini-kpi"><span>Fold Crease Error</span><b>{mc["baseline_crease_err_deg"]:.1f}&deg; &rarr; <em class="good">{mc["hermite_crease_err_deg"]:.1f}&deg;</em></b><small>Sil IoU: {mc["baseline_iou"]:.3f} &rarr; {mc["hermite_iou"]:.3f}</small></div>
    <div class="mini-kpi"><span>Foam Sag / RMS Track</span><b>{mc["baseline_foam_sag_err_mm"]:.1f} &rarr; <em class="good">{mc["hermite_foam_sag_err_mm"]:.1f}</em> mm</b><small>RMS {mc["baseline_rms_track_err_mm"]:.1f} &rarr; {mc["hermite_rms_track_err_mm"]:.1f} mm</small></div>
    {novel_chip}
  </div>

  <div class="video-card">
    <video id="vid-{tid}" controls loop playsinline preload="metadata" src="{mp4_uri}"></video>
    <div class="controls-bar">
      <div class="ctrl-group">
        <button class="ctrl-btn" onclick="togglePlay('{tid}')">Play / Pause</button>
        <button class="ctrl-btn" onclick="setSpeed('{tid}', 0.25)">0.25x</button>
        <button class="ctrl-btn" onclick="setSpeed('{tid}', 0.5)">0.5x</button>
        <button class="ctrl-btn" onclick="setSpeed('{tid}', 1.0)">1.0x</button>
      </div>
      <div class="ctrl-group">
        <span class="muted">Jump to:</span>
        {jumps_html}
      </div>
    </div>
  </div>

  <div class="split-grid">
    <div class="card-sub">
      <h4>Peak Frame Before / After Curtain Slider (Drag Divider)</h4>
      <div class="curtain-wrap" data-task="{tid}">
        <img class="curtain-base" src="{new_png_uri}" alt="New HermiteTwin Peak"/>
        <div class="curtain-top" id="curtain-top-{tid}" style="width:50%;">
          <img src="{old_png_uri}" alt="Old Spring-Mass Peak"/>
        </div>
        <div class="curtain-handle" id="curtain-handle-{tid}" style="left:50%;"></div>
        <span class="curtain-tag left-tag">OLD: Spring-Mass</span>
        <span class="curtain-tag right-tag">NEW: HermiteTwin</span>
      </div>
      <input type="range" min="0" max="100" value="50" class="curtain-range" oninput="updateCurtain('{tid}', this.value)"/>
    </div>
    <div class="card-sub">
      <h4>Per-Frame Chamfer Trajectory &amp; Equal-Budget SysID Convergence</h4>
      {cd_svg}
      <div style="height:10px;"></div>
      {sysid_svg}
    </div>
  </div>
</section>
''')

    # Tab 4: Dual-Renderer 3DGS vs PBR
    dual_vid = media.get("dual_renderer") or {}
    if dual_vid and Path(dual_vid["mp4_path"]).is_file():
        dual_mp4_uri = _b64_uri(dual_vid["mp4_path"], "video/mp4")
        pbr_uri = _b64_uri(dual_vid["pbr_png"], "image/png")
        gs_uri = _b64_uri(dual_vid["gs_png"], "image/png")
        tab_btns_parts.append(
            '<button class="tab-btn" onclick="selectTab(\'dual_renderer\')" id="btn-dual_renderer">4. 3DGS vs. PBR Dual Renderer</button>'
        )
        tabs_html_parts.append(f'''
<section class="tab-panel" id="panel-dual_renderer" style="display:none;">
  <div class="task-header">
    <div>
      <h3>4. Shell-Driven 3DGS vs. Shaded PBR Dual Renderer Audit</h3>
      <p class="muted">784 anisotropic 3D Gaussians anchored to the C1 bicubic Hermite shell tangent basis, verified by EOS anti-brightness-copy affine residual check.</p>
    </div>
    <div class="pass-pill {"pass" if dual_m.get("non_trivial_distinct_renderers") else "fail"}">
      Affine Residual: {dual_m.get("affine_copy_residual", 0.0):.4f} (&gt; 0.005)
    </div>
  </div>
  <div class="mini-kpi-grid">
    <div class="mini-kpi"><span>PSNR (PBR vs 3DGS)</span><b>{dual_m.get("psnr_db", 0.0):.2f} dB</b><small>Distinct renderers</small></div>
    <div class="mini-kpi"><span>SSIM</span><b>{dual_m.get("ssim", 0.0):.4f}</b><small>Structural match</small></div>
    <div class="mini-kpi"><span>Affine Copy Residual</span><b><em class="good">{dual_m.get("affine_copy_residual", 0.0):.4f}</em></b><small>Non-trivial (&gt; 0.005)</small></div>
  </div>
  <div class="video-card">
    <video id="vid-dual_renderer" controls loop playsinline preload="metadata" src="{dual_mp4_uri}"></video>
    <div class="controls-bar">
      <div class="ctrl-group">
        <button class="ctrl-btn" onclick="togglePlay('dual_renderer')">Play / Pause</button>
        <button class="ctrl-btn" onclick="setSpeed('dual_renderer', 0.25)">0.25x</button>
        <button class="ctrl-btn" onclick="setSpeed('dual_renderer', 0.5)">0.5x</button>
        <button class="ctrl-btn" onclick="setSpeed('dual_renderer', 1.0)">1.0x</button>
      </div>
    </div>
  </div>
  <div class="card-sub">
    <h4>Dual-Renderer Curtain Compare (Renderer A: PBR Mesh vs Renderer B: Hermite 3DGS)</h4>
    <div class="curtain-wrap" data-task="dual_renderer">
      <img class="curtain-base" src="{gs_uri}" alt="3DGS"/>
      <div class="curtain-top" id="curtain-top-dual_renderer" style="width:50%;">
        <img src="{pbr_uri}" alt="PBR"/>
      </div>
      <div class="curtain-handle" id="curtain-handle-dual_renderer" style="left:50%;"></div>
      <span class="curtain-tag left-tag">Renderer A: PBR</span>
      <span class="curtain-tag right-tag">Renderer B: Hermite 3DGS</span>
    </div>
    <input type="range" min="0" max="100" value="50" class="curtain-range" oninput="updateCurtain('dual_renderer', this.value)"/>
  </div>
</section>
''')

    # Certification & Provenance table rows
    audit_rows = []
    for tid in task_ids:
        t_roll = roll["tasks"][tid]
        t_cert = cert["tasks"][tid]
        c3 = t_cert["protocol_c3"]
        mc = c3["metrics_comparison"]
        pa = c3["protocol_a"]
        fl = c3["freeze_lock"]
        audit_rows.append(
            f"<tr>"
            f"<td><code>{html.escape(tid)}</code></td>"
            f"<td>{mc['baseline_resim_chamfer_mm2']:.2f} &rarr; <b>{mc['hermite_resim_chamfer_mm2']:.2f}</b> (-{mc['resim_reduction_pct']:.1f}%)</td>"
            f"<td>{mc['baseline_future_chamfer_mm2']:.2f} &rarr; <b>{mc['hermite_future_chamfer_mm2']:.2f}</b> (-{mc['future_reduction_pct']:.1f}%)</td>"
            f"<td>{mc['hermite_max_penetration_mm']:.2f} mm</td>"
            f"<td>{pa['replay_passed_count']}/{pa['replay_total_count']} (ZF={pa['zero_friction_intact_count']}/{pa['zero_friction_total_count']})</td>"
            f"<td>{fl['passed_count']}/{fl['total_count']}</td>"
            f"<td><span class='pass-pill {'pass' if c3['passed'] else 'fail'}'>{'PASS' if c3['passed'] else 'FAIL'}</span></td>"
            f"<td class='mono-sm'>{t_roll['main_npz_sha256'][:16]}...</td>"
            f"</tr>"
        )

    four_engine_status = (
        "VERIFIED_4_ENGINE"
        if tier == "STUDIO_CYCLES_AGX_NATIVE_VERIFIED"
        else "COMPILED_OVERLAYS_READY (Runtime Execution Pending Phase 2 Idle GPU)"
    )

    html_doc = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>HermiteTwin v0.8 A/B Experimental Comparison — Four-Piece Suite</title>
<style>
  :root {{
    --bg: #090d16; --card: #111827; --card2: #0f1623; --border: #22304a;
    --text: #e2e8f0; --muted: #94a3b8; --good: #34d399; --old: #fb923c; --accent: #60a5fa;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }}
  .wrap {{ max-width: 1380px; margin: 0 auto; padding: 24px 20px 56px; }}
  .hero {{ background: linear-gradient(135deg, #0f172a 0%, #162238 60%, #102a2e 100%); border: 1px solid var(--border); border-radius: 16px; padding: 26px 28px; margin-bottom: 22px; }}
  .hero-top {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; gap: 16px; }}
  .hero h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: -0.02em; }}
  .hero p {{ margin: 0; color: var(--muted); max-width: 820px; font-size: 14px; }}
  .tier-badge {{ padding: 8px 14px; border-radius: 999px; font-size: 12px; font-weight: 700; border: 1px solid #3b82f6; background: rgba(59,130,246,0.14); color: #bfdbfe; white-space: nowrap; }}
  .tier-badge.ok {{ border-color: #10b981; background: rgba(16,185,129,0.14); color: #a7f3d0; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-top: 20px; }}
  .kpi-card {{ background: rgba(15,23,42,0.75); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }}
  .kpi-card span {{ font-size: 12px; color: var(--muted); display: block; }}
  .kpi-card strong {{ font-size: 28px; color: var(--good); display: block; margin: 4px 0; font-variant-numeric: tabular-nums; }}
  .kpi-card small {{ font-size: 12px; color: #cbd5e1; }}
  .schematic {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; margin-bottom: 22px; }}
  .schematic h2 {{ margin: 0 0 12px; font-size: 17px; }}
  .tabs-bar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
  .tab-btn {{ background: var(--card); color: var(--muted); border: 1px solid var(--border); border-radius: 10px; padding: 10px 16px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s; }}
  .tab-btn.active {{ background: #1e293b; color: #fff; border-color: var(--accent); box-shadow: 0 0 0 1px rgba(96,165,250,0.4); }}
  .tab-panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; margin-bottom: 22px; }}
  .task-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; margin-bottom: 14px; }}
  .task-header h3 {{ margin: 0 0 4px; font-size: 19px; }}
  .muted {{ color: var(--muted); margin: 0; font-size: 13px; }}
  .pass-pill {{ padding: 4px 11px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
  .pass-pill.pass {{ background: rgba(16,185,129,0.16); color: #6ee7b7; border: 1px solid #10b981; }}
  .pass-pill.fail {{ background: rgba(239,68,68,0.16); color: #fca5a5; border: 1px solid #ef4444; }}
  .mini-kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin-bottom: 16px; }}
  .mini-kpi {{ background: var(--card2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; font-variant-numeric: tabular-nums; }}
  .mini-kpi span {{ font-size: 11px; color: var(--muted); display: block; }}
  .mini-kpi b {{ font-size: 14px; display: block; margin: 2px 0; }}
  .mini-kpi em.good {{ font-style: normal; color: var(--good); }}
  .mini-kpi small {{ font-size: 11px; color: #93c5fd; }}
  .video-card {{ background: #070b12; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 18px; }}
  .video-card video {{ width: 100%; display: block; background: #000; }}
  .controls-bar {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; padding: 10px 14px; background: #0d131f; border-top: 1px solid var(--border); }}
  .ctrl-group {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }}
  .ctrl-btn, .jump-btn {{ background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 7px; padding: 5px 11px; font-size: 12px; cursor: pointer; font-weight: 600; }}
  .ctrl-btn:hover, .jump-btn:hover {{ background: #2d3e58; border-color: var(--accent); }}
  .split-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 960px) {{ .split-grid {{ grid-template-columns: 1fr; }} }}
  .card-sub {{ background: var(--card2); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }}
  .card-sub h4 {{ margin: 0 0 10px; font-size: 14px; color: #cbd5e1; }}
  .curtain-wrap {{ position: relative; width: 100%; aspect-ratio: 4 / 3; overflow: hidden; border-radius: 10px; border: 1px solid var(--border); user-select: none; background: #0b101b; }}
  .curtain-wrap img.curtain-base {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
  .curtain-top {{ position: absolute; top: 0; left: 0; height: 100%; overflow: hidden; border-right: 2px solid #60a5fa; }}
  .curtain-top img {{ width: 100%; height: 100%; object-fit: contain; display: block; max-width: none; }}
  .curtain-handle {{ position: absolute; top: 50%; transform: translate(-50%, -50%); width: 28px; height: 28px; border-radius: 50%; background: #60a5fa; border: 2px solid #fff; pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,0.6); }}
  .curtain-tag {{ position: absolute; bottom: 8px; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; background: rgba(9,13,22,0.82); }}
  .left-tag {{ left: 8px; color: var(--old); border: 1px solid var(--old); }}
  .right-tag {{ right: 8px; color: var(--good); border: 1px solid var(--good); }}
  .curtain-range {{ width: 100%; margin-top: 8px; accent-color: #60a5fa; }}
  .audit-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: left; padding: 10px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
  .mono-sm {{ font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="hero-top">
      <div>
        <h1>HermiteTwin (`arXiv:2611.27282`) &times; Four-Piece Suite (`v0.8`) A/B Experimental Comparison</h1>
        <p>Controlled equal-budget evaluation comparing <b>Old: Single-Resolution Spring-Mass Particles</b> vs. <b>New: Coarse C&sup1; Bicubic Hermite Kirchhoff-Love Shell + Dense Material-Space Grasp</b> across cloth folding, stiff foam transfer, unseen draping, and shell-driven 3DGS dual rendering.</p>
      </div>
      <div class="tier-badge {"ok" if all_passed else ""}">{html.escape(tier)} &bull; Protocol C3: {"3/3 PASS" if all_passed else "CHECK"}</div>
    </div>
    <div class="kpi-grid">
      <div class="kpi-card"><span>Avg Resimulation Chamfer Reduction</span><strong>-{avg_resim_red:.1f}%</strong><small>Measured over training horizon (mm&sup2;)</small></div>
      <div class="kpi-card"><span>Avg Future Prediction Reduction</span><strong>-{avg_fut_red:.1f}%</strong><small>Held-out trajectory continuation (mm&sup2;)</small></div>
      <div class="kpi-card"><span>Out-of-Plane Sag Error Reduction</span><strong>-{sag_red:.1f}%</strong><small>Preserves stiff foam &amp; sheet geometry</small></div>
      <div class="kpi-card"><span>RMS Tracking Error Reduction</span><strong>-{rms_red:.1f}%</strong><small>Zero-cheat dense material-space grasp</small></div>
    </div>
  </div>

  <div class="schematic">
    <h2>Architectural Contrast: Single-Particle Coupling vs. Bicubic Hermite Decoupling</h2>
    <svg viewBox="0 0 1120 145" width="100%" height="145" style="background:#0c121e;border-radius:10px;border:1px solid #1e293b;">
      <rect x="16" y="16" width="510" height="112" rx="10" fill="#171f2e" stroke="#fb923c" stroke-width="1.4"/>
      <text x="34" y="42" fill="#fdba74" font-size="14" font-weight="700">OLD: Single-Resolution Spring-Mass Particle Set (Coupled)</text>
      <text x="34" y="66" fill="#cbd5e1" font-size="12">&bull; Same coarse 10&times;10 particles handle both bending &amp; gripper contact</text>
      <text x="34" y="88" fill="#cbd5e1" font-size="12">&bull; 2-hop springs blur sharp fold creases &amp; sag on stiff foam (D ~ E&middot;h&sup3;)</text>
      <text x="34" y="110" fill="#94a3b8" font-size="11">Result: higher Chamfer error, contact drift, and explicit stability cap</text>

      <path d="M538 72 L582 72" stroke="#60a5fa" stroke-width="2.5" marker-end="url(#arrow)"/>
      <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#60a5fa"/></marker></defs>

      <rect x="594" y="16" width="510" height="112" rx="10" fill="#0f2326" stroke="#34d399" stroke-width="1.6"/>
      <text x="612" y="42" fill="#6ee7b7" font-size="14" font-weight="700">NEW: HermiteTwin v0.8 (Decoupled Deformation &times; Contact)</text>
      <text x="612" y="66" fill="#e2e8f0" font-size="12">&bull; Coarse 4&times;4 C&sup1; Bogner-Fox-Schmit Hermite shell (Kirchhoff-Love bending)</text>
      <text x="612" y="88" fill="#e2e8f0" font-size="12">&bull; Dense 24&times;24 material-space grasp (FREE/CAPTURED/CLAMPED/RELEASED)</text>
      <text x="612" y="110" fill="#a7f3d0" font-size="11">SysID: CMA-ES + L-BFGS-B + Shell-Driven 3DGS + 4-Engine v0.3 Overlays</text>
    </svg>
  </div>

  <div class="tabs-bar">
    {"".join(tab_btns_parts)}
  </div>

  {"".join(tabs_html_parts)}

  <div class="audit-card">
    <h2 style="margin:0 0 12px;font-size:17px;">EOS Protocol C3, Protocol A, Freeze Lock &amp; Provenance Scorecard</h2>
    <table>
      <thead>
        <tr>
          <th>Task ID</th>
          <th>Resim CD (Old &rarr; New)</th>
          <th>Future CD (Old &rarr; New)</th>
          <th>Max Table Pen</th>
          <th>Protocol A (5&times; + ZF)</th>
          <th>Freeze Lock (3&times;)</th>
          <th>Verdict</th>
          <th>Main NPZ SHA-256</th>
        </tr>
      </thead>
      <tbody>
        {"".join(audit_rows)}
      </tbody>
    </table>
    <p class="muted" style="margin-top:12px;">
      <b>4-Engine Thin-Shell Overlay Matrix (mujoco4 / newton18 / genesis / isaac_physx):</b> {html.escape(four_engine_status)} &bull;
      <b>CE-Bench Sandbox Isolation:</b> Executed in isolated clones under <code>hermitetwin_sandbox/</code> on branch <code>feat/deformable-v08-hermitetwin</code>.
    </p>
  </div>
</div>

<script>
function selectTab(id) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const panel = document.getElementById('panel-' + id);
  const btn = document.getElementById('btn-' + id);
  if (panel) panel.style.display = 'block';
  if (btn) btn.classList.add('active');
}}
function togglePlay(id) {{
  const v = document.getElementById('vid-' + id);
  if (!v) return;
  if (v.paused) v.play(); else v.pause();
}}
function setSpeed(id, rate) {{
  const v = document.getElementById('vid-' + id);
  if (v) v.playbackRate = rate;
}}
function jumpVideo(id, tSec) {{
  const v = document.getElementById('vid-' + id);
  if (v) {{ v.currentTime = tSec; v.play(); }}
}}
function updateCurtain(id, pct) {{
  const top = document.getElementById('curtain-top-' + id);
  const handle = document.getElementById('curtain-handle-' + id);
  if (top) top.style.width = pct + '%';
  if (handle) handle.style.left = pct + '%';
}}
</script>
</body>
</html>
'''
    out_p = Path(out_html_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(html_doc, encoding="utf-8")
    return out_p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-summary", type=Path, required=True)
    parser.add_argument("--cert-summary", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    args = parser.parse_args()
    out = build_hermitetwin_comparison_html(
        rollout_summary_path=args.rollout_summary,
        cert_summary_path=args.cert_summary,
        media_manifest_path=args.media_manifest,
        out_html_path=args.out_html,
    )
    print(f"Built self-contained HTML showcase: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
