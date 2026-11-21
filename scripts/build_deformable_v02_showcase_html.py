"""Build self-contained HTML certification & showcase dashboard from M2-M4 deformable artifacts."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import html
from io import BytesIO
import json
from pathlib import Path
from typing import Any


DEFAULT_M4_REPORT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\embodied-eval-os\external_artifacts\deformable_v02_m4\eos_certification_report.json"
)
DEFAULT_MEDIA_DIR = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge\external_artifacts\deformable_v02_media"
)
DEFAULT_HTML_OUT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge\external_artifacts\reports\deformable_v02_showcase_20261119.html"
)


def _inline_mp4_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if len(raw) < 12 or raw[4:8] != b"ftyp":
        return None
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def _inline_keyframe_jpg_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        from PIL import Image  # type: ignore

        buf = BytesIO()
        with Image.open(path) as img:
            rgb = img.convert("RGB").resize((640, 360), Image.Resampling.LANCZOS)
            rgb.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"


def _esc(val: Any) -> str:
    return html.escape(str(val), quote=True)


def render_deformable_v02_showcase_html(
    report: dict[str, Any],
    *,
    media_dir: Path | None = None,
) -> str:
    """Render 100% offline self-contained HTML dashboard (no external CDN or URLs)."""
    tasks = report.get("tasks", {})
    ts = report.get("generated_at_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards_html: list[str] = []

    kf_specs = (
        ("t0", "t=0 初始稳态"),
        ("mid", "Mid 柔顺操作过程"),
        ("final", "Final 无损落位终态"),
    )

    for task_id, trep in tasks.items():
        tier = str(trep.get("certification_tier", "Unverified"))
        badge_cls = "badge-gold" if tier == "Gold" else ("badge-silver" if tier == "Silver" else "badge-fail")
        n_pass = int(trep.get("passing_engine_count", 0))
        n_tot = int(trep.get("total_engines", 4))
        pa = trep.get("protocol_a", {})
        fl = trep.get("freeze_lock", {})
        cf_list = trep.get("counterfactual_checks", [])
        cf_route = (trep.get("counterfactual_routing") or {}).get("target_repo", "—")

        rows: list[str] = []
        for eng, esum in trep.get("engine_summaries", {}).items():
            intact = bool(esum.get("intact_task_success", False))
            strain_key = "polar_stretch_q95" if esum.get("structural_family") == "soft_solid" else "edge_stretch_q95"
            strain_val = esum.get(strain_key)
            naive_val = esum.get("naive_f_minus_i_q95")
            strain_str = f"{strain_val:.4f}" if isinstance(strain_val, (int, float)) else "—"
            naive_str = f"{naive_val:.4f}" if isinstance(naive_val, (int, float)) else "—"
            rows.append(
                f"<tr><td><code>{_esc(eng)}</code></td>"
                f"<td>{'PASS' if intact else 'FAIL'}</td>"
                f"<td><code>{_esc(strain_key)}={strain_str}</code></td>"
                f"<td>{naive_str}</td>"
                f"<td>{float(esum.get('peak_normal_force_n', 0.0)):.2f} N</td>"
                f"<td>{float(esum.get('damage_score', 0.0)):.3f}</td>"
                f"<td>{float(esum.get('final_target_dist_m', 0.0)):.4f} m</td></tr>"
            )

        video_html = "<div class='pending-media'>RTX Showcase Media Pending Explicit User Confirmation (<code>--confirm-rtx-render</code>)</div>"
        keyframe_strip_html = ""
        if media_dir is not None:
            mp4_uri = _inline_mp4_if_exists(media_dir / f"{task_id}.mp4")
            if mp4_uri:
                video_html = (
                    f"<video controls autoplay loop muted playsinline preload='metadata' src='{mp4_uri}'></video>"
                )
            kf_figs: list[str] = []
            for kf_key, kf_caption in kf_specs:
                kf_uri = _inline_keyframe_jpg_if_exists(media_dir / f"{task_id}_{kf_key}.png")
                if kf_uri:
                    kf_figs.append(
                        f"<figure class='kf-item'><img src='{kf_uri}' alt='{_esc(task_id)}_{_esc(kf_key)}'/><figcaption>{_esc(kf_caption)}</figcaption></figure>"
                    )
            if kf_figs:
                keyframe_strip_html = f"<div class='keyframe-grid'>{''.join(kf_figs)}</div>"

        cf_summary = "; ".join(
            f"{_esc(c.get('episode_id'))}: dmg={float(c.get('damage_score', 0.0)):.2f}, intact={c.get('intact_task_success')}"
            for c in cf_list
        )
        cards_html.append(
            f"""
            <section class="task-card">
              <div class="card-head">
                <h2><code>{_esc(task_id)}</code></h2>
                <span class="badge {badge_cls}">{_esc(tier)} ({n_pass}/{n_tot})</span>
              </div>
              <div class="pill-row">
                <span>Protocol A: {'PASS' if pa.get('passed') else 'FAIL'} (Replay {pa.get('replay_passed_count', 0)}/{pa.get('replay_total_count', 0)}, 0-Friction Intact {pa.get('zero_friction_intact_count', 0)}/{pa.get('zero_friction_total_count', 0)})</span>
                <span>Freeze Lock: {'PASS' if fl.get('passed') else 'FAIL'} ({fl.get('passed_count', 0)}/{fl.get('total_count', 0)} seeds)</span>
                <span>CF Route: <code>{_esc(cf_route)}</code></span>
              </div>
              <table>
                <thead>
                  <tr><th>Engine</th><th>Intact Gate</th><th>Rotation-Invariant Strain</th><th>Naive ||F-I||</th><th>Peak Normal Force</th><th>Damage D</th><th>Target Dist</th></tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
              <p class="cf-line"><strong>Counterfactuals:</strong> {cf_summary or 'None'}</p>
              <div class="media-box">{video_html}{keyframe_strip_html}</div>
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>v0.2 Four-Engine Deformable + EOS Closed-Loop Certification Showcase</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 24px auto; max-width: 1180px; background: #0d1117; color: #e6edf3; line-height: 1.5; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 4px; }}
  .meta {{ color: #8b949e; margin-bottom: 24px; font-size: 0.92rem; }}
  .task-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px 20px; margin-bottom: 22px; }}
  .card-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
  .card-head h2 {{ margin: 0; font-size: 1.2rem; }}
  .badge {{ padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }}
  .badge-gold {{ background: #3b2e00; color: #f2cc60; border: 1px solid #d29922; }}
  .badge-silver {{ background: #1f2937; color: #d1d5db; border: 1px solid #6b7280; }}
  .badge-fail {{ background: #490b0b; color: #ffa198; border: 1px solid #f85149; }}
  .pill-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; font-size: 0.86rem; }}
  .pill-row span {{ background: #21262d; border: 1px solid #30363d; padding: 4px 10px; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-bottom: 12px; }}
  th, td {{ border-bottom: 1px solid #30363d; padding: 8px 10px; text-align: left; }}
  th {{ color: #8b949e; font-weight: 600; }}
  code {{ background: #21262d; padding: 2px 5px; border-radius: 4px; font-size: 0.86em; }}
  .cf-line {{ font-size: 0.88rem; color: #c9d1d9; }}
  .pending-media {{ padding: 14px; border: 1px dashed #484f58; border-radius: 6px; color: #8b949e; text-align: center; font-size: 0.88rem; }}
  video {{ width: 100%; max-height: 520px; border-radius: 6px; background: #000; }}
  .keyframe-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }}
  .kf-item {{ margin: 0; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }}
  .kf-item img {{ width: 100%; display: block; }}
  .kf-item figcaption {{ padding: 6px 8px; font-size: 0.82rem; color: #8b949e; text-align: center; }}
</style>
</head>
<body>
  <h1>v0.2 Four-Engine OpenUSD Deformable &amp; EOS Closed-Loop Showcase</h1>
  <div class="meta">Generated: {_esc(ts)} &middot; Engines: mujoco4, newton18, genesis, isaac6 &middot; 100% Offline Self-Contained Bundle</div>
  {''.join(cards_html)}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline v0.2 deformable showcase HTML.")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_M4_REPORT)
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--out-html", type=Path, default=DEFAULT_HTML_OUT)
    args = parser.parse_args()

    rep = json.loads(args.report_json.read_text(encoding="utf-8"))
    html_text = render_deformable_v02_showcase_html(rep, media_dir=args.media_dir)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html_text, encoding="utf-8")
    print(json.dumps({"out_html": str(args.out_html), "bytes": len(html_text.encode("utf-8"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
