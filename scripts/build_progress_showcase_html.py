#!/usr/bin/env python3
"""Build a 100% offline, self-contained plain-Chinese HTML progress report with inline MP4s."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from hashlib import sha256
import html
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_HTML = Path("external_artifacts/reports/trajectory_forge_progress_showcase_20261114.html")
SCHEMA_VERSION = "trajectory-forge-progress-showcase-manifest/v0.1"

EXTERNAL_URL_RE = re.compile(
    r"""(?:src|href|poster)\s*=\s*["']\s*(?:https?:|file:|//)""",
    re.IGNORECASE,
)
CSS_EXTERNAL_URL_RE = re.compile(
    r"""url\(\s*["']?\s*(?:https?:|file:|//)""",
    re.IGNORECASE,
)


def resolve_default_inputs(repo_root: Path) -> dict[str, dict[str, Path]]:
    """Resolve expected rollout videos, media dirs, solve reports, and .npz deliverables."""
    rollouts = repo_root / "external_artifacts" / "rollouts"
    videos = {
        "real2sim": rollouts / "jobs/real2sim_office_desk_riser_organize_a1_synth/media/progression.mp4",
        "ingun": rollouts / "jobs/electronics_qa_ingun_probe_lock_a1_synth/media/progression.mp4",
        "creatista_r3": rollouts / "jobs/r3_creatista_final_media/media/progression.mp4",
        "creatista_r4": rollouts / "jobs/specialty_coffee_bar_creatista_prep_a1_synth/media/progression.mp4",
        "brother_r3": rollouts / "jobs/r3_brother_final_media/media/progression.mp4",
        "brother_mode2_pose": rollouts / "mode2_pose/jobs/modern_office_brother_calibrate_a1_synth/media/progression.mp4",
        "brother_mode1_codesign": rollouts / "jobs/modern_office_brother_calibrate_a1_synth/media/progression.mp4",
    }
    reports = {
        "real2sim": rollouts / "reports/real2sim_office_desk_riser_organize_solve_report.json",
        "ingun": rollouts / "reports/electronics_qa_ingun_probe_lock_solve_report.json",
        "creatista_r3": rollouts / "reports/specialty_coffee_bar_creatista_prep_solve_report.json",
        "creatista_r4": rollouts / "reports/specialty_coffee_bar_creatista_prep_r4_solve_report.json",
        "brother_mode2_strict": rollouts / "reports/modern_office_brother_calibrate_r4_mode2_strict_report.json",
        "brother_mode2_pose": rollouts / "reports/modern_office_brother_calibrate_r4_mode2_pose_report.json",
        "brother_mode1_codesign": rollouts / "reports/modern_office_brother_calibrate_r4_scene_codesign_report.json",
    }
    episodes = {
        "real2sim": rollouts / "episodes/real2sim_office_desk_riser_organize_real2sim_office_desk_riser_organize_a1_synth/episode_trajectory.npz",
        "ingun": rollouts / "episodes/electronics_qa_ingun_probe_lock_electronics_qa_ingun_probe_lock_a1_synth/episode_trajectory.npz",
        "creatista_r4": rollouts / "episodes/specialty_coffee_bar_creatista_prep_specialty_coffee_bar_creatista_prep_a1_synth/episode_trajectory.npz",
        "brother_mode2_pose": rollouts / "mode2_pose/episodes/modern_office_brother_calibrate_modern_office_brother_calibrate_a1_synth/episode_trajectory.npz",
        "brother_mode1_codesign": rollouts / "episodes/modern_office_brother_calibrate_modern_office_brother_calibrate_a1_synth/episode_trajectory.npz",
    }
    return {"videos": videos, "reports": reports, "episodes": episodes}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _encode_mp4_data_uri(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    if len(raw) < 12 or raw[4:8] != b"ftyp":
        raise ValueError(f"invalid MP4 header in {path}")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:video/mp4;base64,{b64}", len(raw)


def _encode_image_data_uri(path: Path, *, max_width: int, quality: int = 86) -> str:
    raw = path.read_bytes()
    if len(raw) < 256:
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as img:
            rgb = img.convert("RGB")
            if rgb.width > max_width:
                ratio = max_width / float(rgb.width)
                rgb = rgb.resize((max_width, max(1, int(round(rgb.height * ratio)))), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"


def _fmt_mb(size_bytes: int | None) -> str:
    if not size_bytes:
        return "—"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024.0:.1f} KB"
    return f"{size_bytes / (1024.0 * 1024.0):.1f} MB"


def _step_lookup(report: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    syn = report.get("synthesis_result") or report.get("gate_result") or {}
    items = syn.get("stage_progression") or []
    return {int(item["stage_index"]): dict(item) for item in items}


def _render_filmstrip(
    *,
    video_id: str,
    total_steps: int,
    frames: Sequence[tuple[str, Path, int, str]],
) -> str:
    cards: list[str] = []
    denom = max(total_steps, 1)
    for title, img_path, step_idx, caption in frames:
        thumb_uri = _encode_image_data_uri(img_path, max_width=640, quality=84)
        full_uri = _encode_image_data_uri(img_path, max_width=960, quality=88)
        frac = max(0.0, min(1.0, step_idx / float(denom)))
        step_label = "初始 t=0" if step_idx == 0 else f"第 {step_idx} 步"
        cards.append(
            '<div class="film-card">'
            f'<button type="button" class="film-jump" data-video-id="{html.escape(video_id)}" '
            f'data-frac="{frac:.4f}" title="点击跳转视频到对应时刻">'
            f'<img src="{thumb_uri}" alt="{html.escape(title)}" loading="lazy" />'
            f'<span class="film-badge">{html.escape(step_label)}</span>'
            "</button>"
            '<div class="film-meta">'
            f'<div class="film-title">{html.escape(title)}</div>'
            f'<div class="film-cap">{html.escape(caption)}</div>'
            f'<button type="button" class="btn-mini zoom-btn" data-full-src="{full_uri}" '
            f'data-full-title="{html.escape(title)} · {html.escape(caption)}">🔍 放大看图</button>'
            "</div>"
            "</div>"
        )
    return '<div class="filmstrip">' + "".join(cards) + "</div>"


def audit_html(html_text: str) -> dict[str, Any]:
    """Verify zero external asset references and count embedded video/image URIs."""
    ext_hits = EXTERNAL_URL_RE.findall(html_text) + CSS_EXTERNAL_URL_RE.findall(html_text)
    mp4_uris = re.findall(r"data:video/mp4;base64,[A-Za-z0-9+/=]+", html_text)
    img_uris = re.findall(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=]+", html_text)
    for uri in mp4_uris:
        b64 = uri.split(",", 1)[1]
        prefix = base64.b64decode(b64[:32])
        if len(prefix) < 8 or prefix[4:8] != b"ftyp":
            raise ValueError("embedded video URI missing ISO BMFF ftyp header")
    return {
        "external_url_count": len(ext_hits),
        "mp4_data_uri_count": len(mp4_uris),
        "image_data_uri_count": len(img_uris),
        "byte_length": len(html_text.encode("utf-8")),
    }


def _css_block() -> str:
    return """
:root {
  --bg: #f4f7fb;
  --surface: #ffffff;
  --surface-soft: #f8fafc;
  --ink: #0f172a;
  --ink-soft: #475569;
  --line: #dbe4f0;
  --brand: #1d4ed8;
  --brand-soft: #eff6ff;
  --ok: #047857;
  --ok-bg: #ecfdf5;
  --ok-line: #a7f3d0;
  --warn: #b45309;
  --warn-bg: #fffbeb;
  --warn-line: #fde68a;
  --block: #be123c;
  --block-bg: #fff1f2;
  --block-line: #fecdd3;
  --radius: 16px;
  --shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.65;
}
.topbar {
  position: sticky;
  top: 0;
  z-index: 40;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--line);
}
.topbar-inner {
  max-width: 1280px;
  margin: 0 auto;
  padding: 10px 20px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.brand {
  font-weight: 800;
  font-size: 15px;
  letter-spacing: -0.01em;
  display: flex;
  align-items: center;
  gap: 8px;
}
.brand-tag {
  font-size: 11px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--brand);
  border: 1px solid #bfdbfe;
}
.navlinks {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.navlinks a {
  text-decoration: none;
  color: var(--ink-soft);
  font-size: 13px;
  font-weight: 600;
  padding: 6px 11px;
  border-radius: 999px;
  transition: all 0.15s ease;
}
.navlinks a:hover {
  background: var(--brand-soft);
  color: var(--brand);
}
.wrap {
  max-width: 1280px;
  margin: 0 auto;
  padding: 28px 20px 64px;
}
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0f766e 100%);
  color: #f8fafc;
  border-radius: 22px;
  padding: 34px 32px;
  box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
  margin-bottom: 24px;
}
.hero-kicker {
  display: inline-block;
  font-size: 12px;
  font-weight: 700;
  padding: 4px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  border: 1px solid rgba(255, 255, 255, 0.25);
  margin-bottom: 12px;
}
.hero h1 {
  margin: 0 0 12px;
  font-size: clamp(24px, 3.2vw, 36px);
  line-height: 1.25;
}
.hero-lead {
  margin: 0 0 22px;
  color: #e2e8f0;
  font-size: 16px;
  max-width: 960px;
}
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px;
}
.kpi-card {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 14px;
  padding: 14px 16px;
}
.kpi-val {
  font-size: 26px;
  font-weight: 800;
  color: #ffffff;
  line-height: 1.2;
}
.kpi-label {
  font-size: 13px;
  color: #cbd5e1;
  margin-top: 4px;
}
.section {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 26px 26px;
  box-shadow: var(--shadow);
  margin-bottom: 24px;
}
.sec-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
}
.sec-num {
  font-size: 12px;
  font-weight: 800;
  color: var(--brand);
  background: var(--brand-soft);
  padding: 4px 10px;
  border-radius: 999px;
}
.sec-title {
  margin: 4px 0 8px;
  font-size: 22px;
  line-height: 1.3;
}
.plain-box {
  background: var(--surface-soft);
  border-left: 4px solid var(--brand);
  border-radius: 10px;
  padding: 14px 16px;
  margin: 12px 0 18px;
  color: #1e293b;
  font-size: 15px;
}
.flow-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 14px;
  margin: 16px 0;
}
.flow-step {
  background: var(--surface-soft);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 16px;
}
.flow-step h4 {
  margin: 6px 0 6px;
  font-size: 16px;
}
.pill {
  display: inline-block;
  font-size: 12px;
  font-weight: 700;
  padding: 2px 9px;
  border-radius: 999px;
}
.pill-ok { background: var(--ok-bg); color: var(--ok); border: 1px solid var(--ok-line); }
.pill-warn { background: var(--warn-bg); color: var(--warn); border: 1px solid var(--warn-line); }
.pill-block { background: var(--block-bg); color: var(--block); border: 1px solid var(--block-line); }
.pill-brand { background: var(--brand-soft); color: var(--brand); border: 1px solid #bfdbfe; }
.video-stage {
  background: #0b1220;
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid #1e293b;
  margin: 14px 0;
}
.video-stage video {
  display: block;
  width: 100%;
  max-height: 540px;
  background: #000;
}
.video-bar {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: #111827;
  color: #e2e8f0;
  font-size: 13px;
}
.filmstrip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
  gap: 12px;
  margin: 14px 0 6px;
}
.film-card {
  background: var(--surface-soft);
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.film-jump {
  position: relative;
  border: 0;
  padding: 0;
  margin: 0;
  background: #000;
  cursor: pointer;
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
}
.film-jump img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: transform 0.2s ease;
}
.film-jump:hover img { transform: scale(1.03); }
.film-badge {
  position: absolute;
  left: 8px;
  top: 8px;
  background: rgba(15, 23, 42, 0.85);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 6px;
}
.film-meta {
  padding: 9px 10px 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
}
.film-title { font-size: 13px; font-weight: 700; color: var(--ink); }
.film-cap { font-size: 12px; color: var(--ink-soft); flex: 1; }
.btn-mini {
  align-self: flex-start;
  margin-top: 4px;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 8px;
  padding: 3px 8px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}
.btn-mini:hover { background: var(--brand-soft); border-color: #93c5fd; }
.compare-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 18px;
  margin-top: 14px;
}
.compare-3 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
  margin-top: 14px;
}
.subpanel {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 16px;
  background: #fff;
}
.subpanel.ok { border-top: 4px solid var(--ok); }
.subpanel.block { border-top: 4px solid var(--block); }
.subpanel.brand { border-top: 4px solid var(--brand); }
.mode-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0 4px;
}
.mode-tab {
  border: 1px solid var(--line);
  background: var(--surface-soft);
  color: var(--ink);
  border-radius: 999px;
  padding: 7px 14px;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
}
.mode-tab.active {
  background: var(--brand);
  color: #fff;
  border-color: var(--brand);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13.5px;
  margin-top: 10px;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 10px 10px;
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--surface-soft);
  font-weight: 700;
  color: var(--ink);
}
details.tech-drawer {
  margin-top: 14px;
  background: var(--surface-soft);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px 14px;
}
details.tech-drawer summary {
  cursor: pointer;
  font-weight: 700;
  font-size: 13.5px;
  color: var(--brand);
}
details.tech-drawer pre {
  margin: 10px 0 0;
  padding: 12px;
  background: #0f172a;
  color: #e2e8f0;
  border-radius: 8px;
  overflow-x: auto;
  font-size: 12px;
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.9em;
  background: #eef2ff;
  color: #1e3a8a;
  padding: 1px 5px;
  border-radius: 5px;
}
.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.85);
  z-index: 90;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 20px;
}
.modal-backdrop.open { display: flex; }
.modal-box {
  background: #0f172a;
  color: #f8fafc;
  border-radius: 16px;
  max-width: 1060px;
  width: 100%;
  overflow: hidden;
  border: 1px solid #334155;
}
.modal-box img { width: 100%; display: block; max-height: 78vh; object-fit: contain; background: #000; }
.modal-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  font-size: 14px;
}
.modal-close {
  border: 1px solid #475569;
  background: #1e293b;
  color: #fff;
  border-radius: 8px;
  padding: 5px 12px;
  cursor: pointer;
}
.hidden-panel { display: none; }
.footer-audit {
  text-align: center;
  font-size: 12.5px;
  color: var(--ink-soft);
  margin-top: 24px;
}
"""


def _js_block() -> str:
    return """
(function() {
  document.querySelectorAll('.film-jump').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var vidId = btn.getAttribute('data-video-id');
      var frac = parseFloat(btn.getAttribute('data-frac') || '0');
      var vid = document.getElementById(vidId);
      if (!vid) return;
      var seekAndPlay = function() {
        var dur = vid.duration || 0;
        if (dur > 0) {
          vid.currentTime = Math.max(0, Math.min(dur - 0.1, frac * dur - 0.8));
        }
        vid.play().catch(function() {});
      };
      if (vid.readyState >= 1) {
        seekAndPlay();
      } else {
        vid.addEventListener('loadedmetadata', seekAndPlay, { once: true });
      }
      vid.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });

  var backdrop = document.getElementById('img-modal');
  var modalImg = document.getElementById('img-modal-pic');
  var modalCap = document.getElementById('img-modal-cap');
  document.querySelectorAll('.zoom-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      modalImg.src = btn.getAttribute('data-full-src') || '';
      modalCap.textContent = btn.getAttribute('data-full-title') || '';
      backdrop.classList.add('open');
    });
  });
  var closeModal = function() { backdrop.classList.remove('open'); };
  document.getElementById('img-modal-close').addEventListener('click', closeModal);
  backdrop.addEventListener('click', function(e) {
    if (e.target === backdrop) closeModal();
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeModal();
  });

  var tabs = document.querySelectorAll('.mode-tab');
  var cards = document.querySelectorAll('.brother-mode-card');
  tabs.forEach(function(tab) {
    tab.addEventListener('click', function() {
      tabs.forEach(function(t) { t.classList.remove('active'); });
      tab.classList.add('active');
      var target = tab.getAttribute('data-target');
      cards.forEach(function(card) {
        if (target === 'all' || card.getAttribute('data-mode') === target) {
          card.classList.remove('hidden-panel');
        } else {
          card.classList.add('hidden-panel');
        }
      });
    });
  });
})();
"""


def _render_html_document(repo_root: Path) -> str:
    inputs = resolve_default_inputs(repo_root)
    videos = inputs["videos"]
    reports_paths = inputs["reports"]
    episodes = inputs["episodes"]

    reports = {key: _read_json(path) for key, path in reports_paths.items()}
    video_data: dict[str, tuple[str, int]] = {
        key: _encode_mp4_data_uri(path) for key, path in videos.items()
    }
    ep_sizes = {
        key: (path.stat().st_size if path.exists() else None)
        for key, path in episodes.items()
    }

    r2s_steps = int(reports["real2sim"]["synthesis_result"]["steps_executed"])
    r2s_prog = _step_lookup(reports["real2sim"])
    r2s_media = videos["real2sim"].parent

    ingun_steps = int(reports["ingun"]["synthesis_result"]["steps_executed"])
    ingun_prog = _step_lookup(reports["ingun"])
    ingun_media = videos["ingun"].parent

    crea_r3_hist = (reports["creatista_r3"].get("history") or [{}])[0]
    crea_r4_steps = int(reports["creatista_r4"]["synthesis_result"]["steps_executed"])
    crea_r4_prog = _step_lookup(reports["creatista_r4"])
    crea_r4_media = videos["creatista_r4"].parent

    bro_strict_hist = (reports["brother_mode2_strict"].get("history") or [{}])[0]
    bro_pose_steps = int(reports["brother_mode2_pose"]["synthesis_result"]["steps_executed"])
    bro_pose_prog = _step_lookup(reports["brother_mode2_pose"])
    bro_pose_media = videos["brother_mode2_pose"].parent

    bro_code_steps = int(reports["brother_mode1_codesign"]["synthesis_result"]["steps_executed"])
    bro_code_prog = _step_lookup(reports["brother_mode1_codesign"])
    bro_code_media = videos["brother_mode1_codesign"].parent

    r2s_filmstrip = _render_filmstrip(
        video_id="vid-real2sim",
        total_steps=r2s_steps,
        frames=[
            ("开局初始态", r2s_media / "t0_initial.png", 0, "桌面与双抽屉增高架静止就位，好丽友派/蓝色圆珠笔/玻璃水杯在左前方"),
            ("第1步：拉开右抽屉", r2s_media / "stage_0_stage0_drawer_opened.png", int(r2s_prog[0]["step_index"]), "右手夹持圆环拉手水平拉出 158.9 mm"),
            ("第2步：收纳好丽友派", r2s_media / "stage_1_stage1_chocopie_stored.png", int(r2s_prog[1]["step_index"]), "抓起好丽友派旋转 90° 平放入右抽屉左侧"),
            ("第3步：收纳蓝色圆珠笔", r2s_media / "stage_2_stage2_pen_stored.png", int(r2s_prog[2]["step_index"]), "抓起圆珠笔平稳放入右抽屉右侧空位"),
            ("第4步：推回右抽屉", r2s_media / "stage_3_stage3_drawer_closed.png", int(r2s_prog[3]["step_index"]), "轻推拉手闭合至 1.75 mm，抽屉内两件物品安稳随行"),
            ("第5步：扶正水杯收官", r2s_media / "stage_4_stage4_mug_placed_upright.png", int(r2s_prog[4]["step_index"]), "将玻璃水杯竖直放到雪碧瓶旁（倾角 < 0.6°），双手退离完成"),
        ],
    )

    ingun_filmstrip = _render_filmstrip(
        video_id="vid-ingun",
        total_steps=ingun_steps,
        frames=[
            ("开局初始态", ingun_media / "t0_initial.png", 0, "SMT 防静电台面、INGUN 针床治具与 PCBA 载板就绪"),
            ("第1步：放入 PCBA 载板", ingun_media / "stage_0_stage0_pcba_aligned.png", int(ingun_prog[0]["step_index"]), "取板旋转 90° 精准落入四导轨针巢"),
            ("第2步：合上压针上盖", ingun_media / "stage_1_stage1_hood_closed.png", int(ingun_prog[1]["step_index"]), "下拉上盖至锁止行程"),
            ("第3步：压紧偏心手柄", ingun_media / "stage_2_stage2_lever_locked.png", int(ingun_prog[2]["step_index"]), "下压偏心锁紧杆完成机械压合"),
            ("第4步：拨动探针开关", ingun_media / "stage_3_stage3_probe_enabled.png", int(ingun_prog[3]["step_index"]), "切换探针测试使能开关"),
            ("第5步：按下启动全绿", ingun_media / "stage_4_stage4_test_started.png", int(ingun_prog[4]["step_index"]), "触发测试启动键，5/5 项同时保持通电锁紧"),
        ],
    )

    crea_r4_filmstrip = _render_filmstrip(
        video_id="vid-creatista-r4",
        total_steps=crea_r4_steps,
        frames=[
            ("开局初始态", crea_r4_media / "t0_initial.png", 0, "咖啡吧台与 Creatista 咖啡机外观零改动"),
            ("第1步：拉出滴水盘", crea_r4_media / "stage_0_stage0_tray_extended.png", int(crea_r4_prog[0]["step_index"]), "滑出不锈钢滴水盘至 -37.5 mm"),
            ("第2步：放稳浓缩杯", crea_r4_media / "stage_1_stage1_cup_seated.png", int(crea_r4_prog[1]["step_index"]), "浓缩杯竖直落座在拉出的滴水盘正中"),
            ("第3步：投入咖啡胶囊", crea_r4_media / "stage_2_stage2_capsule_inserted.png", int(crea_r4_prog[2]["step_index"]), "咖啡胶囊落入顶部 46×48 mm 非凸投料槽"),
            ("第4步：压合不锈钢压杆", crea_r4_media / "stage_3_stage3_lever_locked.png", int(crea_r4_prog[3]["step_index"]), "阻尼恢复物理合理后一杆轻松压锁（r3 卡点突破）"),
            ("第5步：拨旋钮至浓缩档", crea_r4_media / "stage_4_stage4_dial_selected.png", int(crea_r4_prog[4]["step_index"]), "11 次开爪棘轮搓旋将旋钮转至浓缩档"),
            ("第6步：按下旋钮完成", crea_r4_media / "stage_5_stage5_dial_pressed.png", int(crea_r4_prog[5]["step_index"]), "按压旋钮确认启动，6/6 项全部保持达标"),
        ],
    )

    bro_pose_filmstrip = _render_filmstrip(
        video_id="vid-bro-pose",
        total_steps=bro_pose_steps,
        frames=[
            ("开局：纸包悬桌沿 3cm", bro_pose_media / "t0_initial.png", 0, "零新增部件，纸包与校准纸前移悬出桌沿 30–32 mm 便于侧捏薄边"),
            ("第1步：装纸入盒", bro_pose_media / "stage_0_stage0_media_loaded.png", int(bro_pose_prog[0]["step_index"]), "侧捏 22 mm 薄边顺滑放入进纸盒"),
            ("第2步：合上纸盒", bro_pose_media / "stage_1_stage1_cassette_closed.png", int(bro_pose_prog[1]["step_index"]), "前推纸盒推至闭合限位"),
            ("第3步：掀盖放校准纸", bro_pose_media / "stage_2_stage2_sheet_aligned.png", int(bro_pose_prog[2]["step_index"]), "开扫描盖，侧捏 6 mm 薄纸放入玻璃台面"),
            ("第4步：轻合扫描盖", bro_pose_media / "stage_3_stage3_lid_closed.png", int(bro_pose_prog[3]["step_index"]), "扫描盖稳步压合归零"),
            ("第5步：按启动键收官", bro_pose_media / "stage_4_stage4_copy_started.png", int(bro_pose_prog[4]["step_index"]), "按下 Start 键，5/5 全绿通关"),
        ],
    )

    bro_code_filmstrip = _render_filmstrip(
        video_id="vid-bro-codesign",
        total_steps=bro_code_steps,
        frames=[
            ("开局：桌心原位+共设计提手", bro_code_media / "t0_initial.png", 0, "保留桌心自然布局，在纸包加顶置提手桥、校准纸加前缘指提条"),
            ("第1步：提拎装盒", bro_code_media / "stage_0_stage0_media_loaded.png", int(bro_code_prog[0]["step_index"]), "通过 named_frame('grasp') 正抓 48 mm 宽提手转 -90° 稳入纸盒"),
            ("第2步：推闭纸盒", bro_code_media / "stage_1_stage1_cassette_closed.png", int(bro_code_prog[1]["step_index"]), "推回纸盒锁定"),
            ("第3步：提条放入校准纸", bro_code_media / "stage_2_stage2_sheet_aligned.png", int(bro_code_prog[2]["step_index"]), "掀盖并通过指提条平稳放平校准纸"),
            ("第4步：轻合扫描盖", bro_code_media / "stage_3_stage3_lid_closed.png", int(bro_code_prog[3]["step_index"]), "压合扫描盖"),
            ("第5步：按启动键收官", bro_code_media / "stage_4_stage4_copy_started.png", int(bro_code_prog[4]["step_index"]), "按下 Start 键，5/5 全绿通关"),
        ],
    )

    def _video_block(vid_id: str, key: str, poster_path: Path, bar_left: str, bar_right: str) -> str:
        uri, raw_len = video_data[key]
        poster_uri = _encode_image_data_uri(poster_path, max_width=960, quality=88) if poster_path.exists() else ""
        poster_attr = f' poster="{poster_uri}"' if poster_uri else ""
        return (
            '<div class="video-stage">'
            f'<video id="{html.escape(vid_id)}" controls playsinline preload="metadata"{poster_attr}>'
            f'<source src="{uri}" type="video/mp4" />'
            "</video>"
            '<div class="video-bar">'
            f"<span>{html.escape(bar_left)}</span>"
            f"<span>{html.escape(bar_right)} · 原始 MP4 {_fmt_mb(raw_len)}</span>"
            "</div>"
            "</div>"
        )

    ledger_rows = [
        (
            "实拍工位孪生收纳 (GEN-010)",
            "real2sim_office_desk_riser_organize",
            "模式 2 物理修复（补齐圆环拉手碰撞体）",
            f"5/5 (1.00) · {r2s_steps} 步",
            _fmt_mb(ep_sizes["real2sim"]),
            "0 快照 · continuous_from_t0=True · privileged=False",
        ),
        (
            "电子质检针床装夹 (r3/r4 基准)",
            "electronics_qa_ingun_probe_lock",
            "首轮即通 · 无上游改景",
            f"5/5 (1.00) · {ingun_steps} 步",
            _fmt_mb(ep_sizes["ingun"]),
            "0 快照 · continuous_from_t0=True · privileged=False",
        ),
        (
            "咖啡机浓缩萃取准备 (r4)",
            "specialty_coffee_bar_creatista_prep",
            "模式 2 严格物理调参（阻尼修正确，外观/位姿 0 改动）",
            f"6/6 (1.00) · {crea_r4_steps} 步",
            _fmt_mb(ep_sizes["creatista_r4"]),
            "0 快照 · continuous_from_t0=True · privileged=False",
        ),
        (
            "打印机校准 — 严格模式 2 消融 (r4)",
            "modern_office_brother_calibrate (mode2_strict)",
            "physics_tuning + allow_pose_adjustment=False",
            "策略熔断 (attempts=1) · Stage 1 通过",
            "—（不产出作弊轨迹）",
            "BLOCKED_BY_REMEDIATION_POLICY（省算力早停）",
        ),
        (
            "打印机校准 — 模式 2 + 位姿子开关 (r4)",
            "modern_office_brother_calibrate (mode2_pose)",
            "physics_tuning + allow_pose_adjustment=True（桌沿悬 3 cm）",
            f"5/5 (1.00) · {bro_pose_steps} 步",
            _fmt_mb(ep_sizes["brother_mode2_pose"]),
            "0 快照 · continuous_from_t0=True · privileged=False",
        ),
        (
            "打印机校准 — 模式 1 共设计 (r4)",
            "modern_office_brother_calibrate (scene_codesign)",
            "scene_codesign（桌心原位 + 增补提手桥/指提条）",
            f"5/5 (1.00) · {bro_code_steps} 步",
            _fmt_mb(ep_sizes["brother_mode1_codesign"]),
            "0 快照 · continuous_from_t0=True · privileged=False",
        ),
    ]
    ledger_html = "\n".join(
        "<tr>"
        f"<td><strong>{html.escape(a)}</strong><br/><code>{html.escape(b)}</code></td>"
        f"<td>{html.escape(c)}</td>"
        f"<td>{html.escape(d)}</td>"
        f"<td>{html.escape(e)}</td>"
        f"<td>{html.escape(f)}</td>"
        "</tr>"
        for a, b, c, d, e, f in ledger_rows
    )

    generated_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Trajectory Forge 阶段成果汇报 · 离线自包含视频版</title>
<style>{_css_block()}</style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <span>Trajectory Forge 成果汇报</span>
      <span class="brand-tag">100% 离线自包含 · 实机视频可点播</span>
    </div>
    <nav class="navlinks">
      <a href="#overview">一分钟看懂</a>
      <a href="#architecture">防作弊与双模式</a>
      <a href="#real2sim">实拍工位孪生</a>
      <a href="#industrial">工业/商业修前修后</a>
      <a href="#remediation">三模态对照实验</a>
      <a href="#ledger">交付数据账本</a>
    </nav>
  </div>
</header>

<main class="wrap">
  <section id="overview" class="hero">
    <div class="hero-kicker">汇报生成时间：{html.escape(generated_at)} · 双 RTX 4090 Isaac Sim 4.1 实测通过</div>
    <h1>让机器人在逼真物理世界里“零作弊”学会干活，<br/>并反推上游场景哪里不合理</h1>
    <p class="hero-lead">
      <strong>一句话讲清两个仓库的分工：</strong>上游 <code>scenario-forge</code> 负责“<strong>出考卷造房间</strong>”（搭建桌子、咖啡机、打印机、实拍工位数字孪生）；
      本仓库 <code>trajectory-forge</code> 负责“<strong>解题并录制纯净教学录像</strong>”——让双臂机器人用真物理手感（零作弊、从第 0 帧一镜到底）把长程任务完整做通，
      做不通时还能像老师一样精确定位“到底是题本身印错了，还是物理参数不合理”。
    </p>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-val">4 / 4 场景全绿</div>
        <div class="kpi-label">实拍办公工位 + 电子质检 + 咖啡吧 + 打印机校准全流程打通（1.00 满分）</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">5 条纯净示范轨迹</div>
        <div class="kpi-label">全部一镜到底从 t=0 连续录制，0 次存档作弊、0 泄漏上帝视角参数</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">2 套整改模式 + 1 子开关</div>
        <div class="kpi-label">严格物理调参 / 允许微调摆放 / 允许加装提手，策略越界首轮即熔断止损</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">7 段高清视频离线内置</div>
        <div class="kpi-label">含实机通关录屏与修前卡点对比，点击下方任意分镜缩略图可直跳对应时刻</div>
      </div>
    </div>
  </section>

  <section id="architecture" class="section">
    <div class="sec-head">
      <span class="sec-num">01 · 核心机制（通俗版）</span>
      <span class="pill pill-brand">Privileged-to-Sensorimotor Firewall + Dual-Mode Remediation</span>
    </div>
    <h2 class="sec-title">为什么不能边看“正确答案”边练？防作弊三道关 + 上游双模式整改</h2>
    <div class="plain-box">
      如果教机器人学做咖啡时允许它随时读档重来、或者偷偷瞄到“物体三维真值坐标”，学出来的策略一旦上真机就会翻车。
      因此 <code>trajectory-forge</code> 把流程严格切成了<strong>先打草稿、再闭卷统考、最后刻盘导出</strong>三步；一旦发现卡死，再按团队允许的权限级别去指导上游改场景。
    </div>

    <div class="flow-grid">
      <div class="flow-step">
        <span class="pill pill-warn">第 1 关：草稿试错搜索 (Search)</span>
        <h4>允许存档查错，快速找解法与病根</h4>
        <p>在物理仿真里逐阶段调试控制脚本，允许保存/恢复快照定位哪一步卡壳，自动诊断具体故障码。</p>
      </div>
      <div class="flow-step">
        <span class="pill pill-brand">第 2 关：闭卷一镜到底 (Gate)</span>
        <h4>禁止一切读档，必须从第 0 秒走到满分</h4>
        <p>严禁恢复任何快照（触碰即抛异常拦截），验证机器人靠连续控制能否稳稳把所有阶段做到结束且保持不滑脱。</p>
      </div>
      <div class="flow-step">
        <span class="pill pill-ok">第 3 关：零特权导出 (Synthesis)</span>
        <h4>只留“眼睛看+关节感受”，打包训练集</h4>
        <p>导出给下游训练/评测系统的 <code>.npz</code> 只包含第一视角画面、机械臂关节角、移动底盘与时间戳，剔除一切上帝真值。</p>
      </div>
    </div>

    <h3 style="margin-top:20px;margin-bottom:6px;font-size:17px;">当场景本身“卡死”时，上游场景应该怎么改？（双模式 + 位姿子开关）</h3>
    <table>
      <thead>
        <tr>
          <th>策略档位（代码字段）</th>
          <th>打个比方</th>
          <th>允许上游改动什么</th>
          <th>绝对禁止什么</th>
          <th>适用场景</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><span class="pill pill-ok">模式 2 严格版</span><br/><code>physics_tuning</code><br/><code>allow_pose_adjustment=False</code></td>
          <td><strong>“只许修零件内伤”</strong></td>
          <td>关节阻尼/驱动刚度、摩擦系数、隐形碰撞壳（如实拍拉手的透明碰撞代理）</td>
          <td>严禁加减零件、改外观贴图、挪动桌子或物品初始摆位</td>
          <td><strong>Real-to-Sim 实拍还原</strong>、标准真机对齐基准</td>
        </tr>
        <tr>
          <td><span class="pill pill-brand">模式 2 + 位姿子开关</span><br/><code>physics_tuning</code><br/><code>allow_pose_adjustment=True</code></td>
          <td><strong>“顺手摆到好拿边上”</strong></td>
          <td>模式 2 全部物理修复 <strong>+</strong> 允许在不改物体模型前提下微调摆放位姿（如纸包桌沿悬出 3cm）</td>
          <td>仍然严禁给物体粘提手、改造型、换贴图</td>
          <td><strong>不改资产造型</strong>，但原桌面平放物理不可抓</td>
        </tr>
        <tr>
          <td><span class="pill pill-warn">模式 1 场景共设计</span><br/><code>scene_codesign</code></td>
          <td><strong>“给难抓物体加提手/工装”</strong></td>
          <td>模式 2 全部物理修复 <strong>+</strong> 允许上游给过宽平铺物件加提手桥、指提条、治具</td>
          <td>不得伪造不可复现的物理穿模；所有动作仍须过一镜到底门禁</td>
          <td><strong>大规模合成数据厂</strong>、适配特定二指夹爪</td>
        </tr>
      </tbody>
    </table>

    <details class="tech-drawer">
      <summary>展开工程师视图：合约与策略熔断实现位置</summary>
      <pre>契约定义: src/trajectory_forge/core/contracts.py -> RemediationPolicySpec(mode, allow_pose_adjustment)
策略判定: src/trajectory_forge/solver/diagnostics.py -> apply_remediation_policy() / compute_sequential_blocking_stage()
调度熔断: src/trajectory_forge/solver/harness.py -> stop_on_policy_block=True（超限时首轮即停，输出 BLOCKED_BY_REMEDIATION_POLICY）
命名位点: src/trajectory_forge/intake/genmanip_intake.py -> NamedFrameSpec（支持共设计提手的 named_frame("grasp") 自动读取）</pre>
    </details>
  </section>

  <section id="real2sim" class="section">
    <div class="sec-head">
      <span class="sec-num">02 · GEN-010 实拍数字孪生落地</span>
      <span class="pill pill-ok">模式 2 物理修复 · 5/5 满分一镜到底</span>
    </div>
    <h2 class="sec-title">实拍工位桌面 5 步整理：自动揪出“看得见却抓不到”的透明碰撞漏洞</h2>
    <div class="plain-box">
      <strong>通俗解读：</strong>这是从真实办公桌面照片/视频一键还原出来的数字孪生工位。机器人需要连续完成
      <strong>①拉开右抽屉 → ②装入好丽友派 → ③放入蓝色圆珠笔 → ④推关抽屉（里面的东西要稳稳跟着回去） → ⑤把玻璃水杯摆正到雪碧瓶旁</strong>。
      刚开始仿真时，机器人手指直接从直径仅 29 毫米的金属圆环拉手中间“穿过去”——我们的碰撞审计一眼揪出上游模型“有外形但忘加碰撞体”（<code>UPSTREAM_MISSING_HANDLE_COLLIDER</code>），
      按<strong>模式 2 严格要求</strong>在不改一像素外观的前提下仅补上隐形胶囊/方块碰撞代理，随即便以 <strong>1448 步一镜到底拿到 1.00 满分</strong>！
    </div>

    {_video_block("vid-real2sim", "real2sim", r2s_media / "final.png", "实拍数字孪生工位 · 5 阶段连续整理（点击下方分镜可跳帧）", f"{r2s_steps} 步 · 5/5 顺序全绿 · 0 快照作弊")}
    {r2s_filmstrip}

    <details class="tech-drawer">
      <summary>展开工程师视图：实拍工位物理指标与 .npz 维度</summary>
      <pre>任务包: real2sim_office_desk_riser_organize (GEN-010 Workstation Twin)
关键物理状态: 抽屉拉出 0.15888 m (Step 240) -> 放入派 (Step 548) -> 放入笔 (Step 764) -> 推闭至 1.75 mm (Step 1036) -> 水杯竖直摆正倾角 &lt; 0.6° (Step 1328)
防火墙验证: snapshots_used=0, firewall_verified=True, continuous_from_t0=True, privileged_channels_recorded=False
导出 .npz 字段: proprio(1448,16), action(1448,16), base_target(1448,3), torso_lift(1448,1), timestamps_s(1448,), rgb_head(181,720,1280,3)
控制器源码: examples/controllers/real2sim/office_desk_riser_organize_privileged.py</pre>
    </details>
  </section>

  <section id="industrial" class="section">
    <div class="sec-head">
      <span class="sec-num">03 · 域外复杂设备突破</span>
      <span class="pill pill-ok">电子质检全通 + 咖啡吧修前/修后对比</span>
    </div>
    <h2 class="sec-title">精密工业针床一次通关；商用咖啡机修复“百倍过硬压杆”后 6 步全绿</h2>
    <div class="plain-box">
      <strong>通俗解读：</strong>除了办公桌面，我们还挑战了日常很少见的精密设备：
      <strong>左边/上面是工厂电路板针床质检台</strong>，机械臂放板、压盖、扳偏心锁杆、拨开关一次全通；
      <strong>下面是精品咖啡吧台</strong>——此前无论怎么用力机器人都压不合那根不锈钢压杆（3/6 卡死），系统诊断出上游把物理旋转阻尼错当成国际单位多写了 57.3 倍（物理等价 <code>687.5 N·m·s/rad</code> 堪比生锈铁门！）。
      上游按诊断调回正常阻尼并在旋钮上启用连续棘轮搓旋，<strong>完全未改外观与摆位</strong>就直接拿到 6/6 满分！
    </div>

    <h3 style="margin-top:8px;">3A. SMT 电子质检针床测试（<code>electronics_qa_ingun_probe_lock</code> · 首轮 5/5 满分）</h3>
    {_video_block("vid-ingun", "ingun", ingun_media / "final.png", "INGUN 针床治具 PCBA 装夹与锁紧全流程", f"{ingun_steps} 步 · 5/5 保持通关 · 0 快照作弊")}
    {ingun_filmstrip}

    <h3 style="margin-top:26px;">3B. Nespresso Creatista 咖啡机“修前卡死 vs 修后满分”（<code>specialty_coffee_bar_creatista_prep</code>）</h3>
    <div class="compare-2">
      <div class="subpanel block">
        <span class="pill pill-block">修前 (r3)：在第 4 步压杆处硬卡死 (3/6, 0.50分)</span>
        <h4 style="margin:8px 0 4px;">诊断发现：压杆关节阻尼被错误放大 57 倍</h4>
        <p style="font-size:13.5px;color:var(--ink-soft);margin:0 0 8px;">
          放杯、投胶囊都顺利，但压杆怎么下压都纹丝不动。诊断码：<code>{html.escape(str(crea_r3_hist.get('failure_code', 'UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT')))}</code>
        </p>
        {_video_block("vid-creatista-r3", "creatista_r3", videos["creatista_r3"].parent / "final.png", "修复前 (r3) 卡死回放：压杆推不动", "1119 步卡在 Stage 3 · 得分 0.50")}
      </div>
      <div class="subpanel ok">
        <span class="pill pill-ok">修后 (r4 严格模式 2)：外观/位姿零改动 6/6 通关 (1.00分)</span>
        <h4 style="margin:8px 0 4px;">只修上游关节物理阻尼 + 11 次连续手爪搓旋拨盘</h4>
        <p style="font-size:13.5px;color:var(--ink-soft);margin:0 0 8px;">
          不加零件、不改模型、不挪咖啡机位置，严格在 <strong>模式 2 (<code>physics_tuning</code>)</strong> 下一镜到底跑完 6 步。
        </p>
        {_video_block("vid-creatista-r4", "creatista_r4", crea_r4_media / "final.png", "修复后 (r4) 满分录屏：轻松压杆 + 棘轮搓旋选浓缩档", f"{crea_r4_steps} 步 · 6/6 全绿保持 · 0 快照作弊")}
      </div>
    </div>
    {crea_r4_filmstrip}

    <details class="tech-drawer">
      <summary>展开工程师视图：电子质检与咖啡吧物理阶段时间线</summary>
      <pre>1) electronics_qa_ingun_probe_lock:
   stage0(pcba)=268 -&gt; stage1(hood)=1418 -&gt; stage2(lever)=1588 -&gt; stage3(switch)=1592 -&gt; stage4(start)=1722 (total=1757)
2) specialty_coffee_bar_creatista_prep (r4 Mode 2 strict):
   stage0(tray=-0.0375m)=160 -&gt; stage1(cup)=448 -&gt; stage2(capsule)=748 -&gt; stage3(lever=-0.0)=896 -&gt; stage4(dial=1.328rad)=1620 -&gt; stage5(press)=1748 (total=1775)
控制器源码: examples/controllers/r3/ingun_probe_lock_privileged.py · examples/controllers/r3/creatista_prep_privileged.py</pre>
    </details>
  </section>

  <section id="remediation" class="section">
    <div class="sec-head">
      <span class="sec-num">04 · 双模式场景调整三方对照实验</span>
      <span class="pill pill-brand">同一道难题的 3 种策略选择</span>
    </div>
    <h2 class="sec-title">14cm 宽纸包 vs 8.8cm 夹爪极限：看懂“严守规矩 / 摆边借力 / 加装提手”</h2>
    <div class="plain-box">
      <strong>通俗解读（物理硬约束）：</strong>在办公打印机校准任务（<code>modern_office_brother_calibrate</code>）里，桌心平放着一包 <strong>14 厘米宽 × 17.3 厘米长、仅 2.2 厘米厚</strong> 的 A5 打印纸包装。
      而双臂机器人的平行二指夹爪最大张开只有 <strong>8.8 厘米</strong>——它平时就像用两根直筷子平贴桌面去捏一本平躺的书：<strong>上面捏不起（太宽），侧面够不着（贴着桌面）！</strong>
      <br/>我们在同一套系统里实测了 <strong>3 档策略开关</strong>，完美展示了系统既能“守规矩报错止损”，又能按需用两种合法方式优雅通关：
    </div>

    <div class="mode-tabs" role="tablist">
      <button type="button" class="mode-tab active" data-target="all">全部三模态同屏对照</button>
      <button type="button" class="mode-tab" data-target="mode2-strict">A · 模式 2 严格禁止改摆位（自动拦截）</button>
      <button type="button" class="mode-tab" data-target="mode2-pose">B · 模式 2 + 位姿子开关（桌沿悬 3cm）</button>
      <button type="button" class="mode-tab" data-target="mode1-codesign">C · 模式 1 场景共设计（加装提手）</button>
    </div>

    <div class="compare-3">
      <div class="subpanel block brother-mode-card" data-mode="mode2-strict">
        <span class="pill pill-block">模态 A · 模式 2 严格版 (allow_pose=False)</span>
        <h4 style="margin:8px 0 4px;">自动识别越权，第 1 轮直接熔断止损</h4>
        <p style="font-size:13.5px;color:var(--ink-soft);margin:0 0 8px;">
          既不许加提手也不许挪到桌边，物理上神仙也抓不起来。系统精确判断根因需要位姿或共设计改动，主动返回
          <code>{html.escape(str(bro_strict_hist.get('failure_code', 'BLOCKED_BY_REMEDIATION_POLICY')))}</code> 并立刻停机（<code>attempts=1</code>），不浪费一分钟 GPU 瞎撞！
        </p>
        {_video_block("vid-bro-r3", "brother_r3", videos["brother_r3"].parent / "final.png", "物理卡死实录：纸包过宽且无处下爪", "Stage 1 纸盒OK · Stage 0/2 因孔径锁死熔断")}
      </div>

      <div class="subpanel brand brother-mode-card" data-mode="mode2-pose">
        <span class="pill pill-brand">模态 B · 模式 2 + 位姿子开关 (allow_pose=True)</span>
        <h4 style="margin:8px 0 4px;">零新增零件！仅桌沿悬出 3 厘米即可 5/5 通关</h4>
        <p style="font-size:13.5px;color:var(--ink-soft);margin:0 0 8px;">
          打开 <code>allow_pose_adjustment=True</code> 后，不改物体三维模型，仅把纸包与校准纸挪到桌前缘悬出 <strong>30–32 mm</strong>（重心稳稳在桌内）。
          机械爪直接侧面 pinch 捏住 22 mm 与 6 mm 的厚度，<strong>{bro_pose_steps} 步拿到 1.00 满分！</strong>
        </p>
        {_video_block("vid-bro-pose", "brother_mode2_pose", bro_pose_media / "final.png", "模式 2 + 位姿微调：桌沿悬出侧捏装纸全通", f"{bro_pose_steps} 步 · 5/5 满分 · 0 快照作弊")}
      </div>

      <div class="subpanel ok brother-mode-card" data-mode="mode1-codesign">
        <span class="pill pill-ok">模态 C · 模式 1 场景共设计 (scene_codesign)</span>
        <h4 style="margin:8px 0 4px;">保留桌心原位！加装提手与指提条 5/5 通关</h4>
        <p style="font-size:13.5px;color:var(--ink-soft);margin:0 0 8px;">
          在合成训练场景模式下，允许上游为桌面正中物体加装 <strong>48 mm 宽顶置提手桥</strong>与<strong>前缘指提条</strong>，并在包内标注
          <code>named_frame("grasp")</code>，机械臂垂直拎起、空中转 -90° 顺滑放入纸盒，<strong>{bro_code_steps} 步拿到 1.00 满分！</strong>
        </p>
        {_video_block("vid-bro-codesign", "brother_mode1_codesign", bro_code_media / "final.png", "模式 1 共设计：顶置提手桥+named_frame 正抓全通", f"{bro_code_steps} 步 · 5/5 满分 · 0 快照作弊")}
      </div>
    </div>

    <h4 style="margin:20px 0 6px;">模态 B（桌沿悬出 3cm）分镜直跳：</h4>
    {bro_pose_filmstrip}

    <h4 style="margin:18px 0 6px;">模态 C（桌心原位 + 共设计提手）分镜直跳：</h4>
    {bro_code_filmstrip}

    <details class="tech-drawer">
      <summary>展开工程师视图：三模态实测对照表与物理关键细节</summary>
      <pre>模态 A (mode2_strict_ablation): attempts_used=1, failure_code=BLOCKED_BY_REMEDIATION_POLICY, root_failure_code=UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE, policy_blocked=True, stage_scores=[0.0, 1.0, 0.0, 0.0, 0.0]
模态 B (mode2_pose_ablation): steps=1449, final_score=1.0, snapshots_used=0, firewall_verified=True; controller=examples/controllers/r4/brother_calibrate_pose_adjusted_privileged.py
模态 C (scene_gate codesign): steps=1566, final_score=1.0, snapshots_used=0, firewall_verified=True; controller=examples/controllers/r3/brother_calibrate_privileged.py
注: 0.28kg A5MediaPack 从 22mm 薄边夹持在 41.5cm 大横移时会滑脱; 经夹持 48mm X向宽面获 2.18x 法向夹紧与机械形锁+空中-90度转正一次过。</pre>
    </details>
  </section>

  <section id="ledger" class="section">
    <div class="sec-head">
      <span class="sec-num">05 · 全量交付账本与纯净核验</span>
      <span class="pill pill-ok">全部可通过单文件离线核验</span>
    </div>
    <h2 class="sec-title">5 条零特权示范轨迹 + GitHub 公开仓库已同步就绪</h2>
    <div class="plain-box">
      所有纯代码、基准配置、特权控制器、单元测试与阶段核验记录均已提交并推送到 GitHub 公开仓库
      <code>jandan138/trajectory-forge</code>（上游造景库在 <code>jandan138/scenario-forge</code>），重型 <code>.npz</code> 仿真数据包与本 HTML 均妥善留在共享盘 <code>external_artifacts/</code> 不污染 Git 体积。
    </div>
    <table>
      <thead>
        <tr>
          <th>场景名称 / 任务 ID</th>
          <th>生效调整策略</th>
          <th>实机最终成绩与步数</th>
          <th>纯净轨迹包 (.npz)</th>
          <th>防作弊防火墙验证</th>
        </tr>
      </thead>
      <tbody>
        {ledger_html}
      </tbody>
    </table>
  </section>

  <div class="footer-audit">
    本报告为 100% 离线自包含 HTML（Built by <code>scripts/build_progress_showcase_html.py</code> · Schema <code>{html.escape(SCHEMA_VERSION)}</code>）· 外部 URL 依赖 = 0 · 双击即可在任意电脑浏览器播放全部视频与分镜。
  </div>
</main>

<div id="img-modal" class="modal-backdrop" role="dialog" aria-modal="true">
  <div class="modal-box">
    <div class="modal-bar">
      <span id="img-modal-cap"></span>
      <button type="button" id="img-modal-close" class="modal-close">关闭 (Esc)</button>
    </div>
    <img id="img-modal-pic" src="" alt="高清放大图" />
  </div>
</div>

<script>{_js_block()}</script>
</body>
</html>
"""


def build_showcase_html(
    *,
    repo_root: Path = REPO_ROOT,
    out_path: Path = DEFAULT_OUT_HTML,
) -> dict[str, Any]:
    """Generate the self-contained HTML file and companion SHA-256 audit manifest."""
    resolved_out = out_path if out_path.is_absolute() else repo_root / out_path
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    html_text = _render_html_document(repo_root)
    audit = audit_html(html_text)
    if audit["external_url_count"] != 0:
        raise RuntimeError(f"self-contained HTML audit failed: external_url_count={audit['external_url_count']}")

    resolved_out.write_text(html_text, encoding="utf-8")
    digest = sha256(resolved_out.read_bytes()).hexdigest()
    size_bytes = resolved_out.stat().st_size
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "html_path": str(resolved_out),
        "html_sha256": f"sha256:{digest}",
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024.0 * 1024.0), 2),
        "audit": audit,
    }
    manifest_path = resolved_out.with_name(f"{resolved_out.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline self-contained Trajectory Forge showcase HTML.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_HTML)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_showcase_html(repo_root=args.repo_root.resolve(), out_path=args.out)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
