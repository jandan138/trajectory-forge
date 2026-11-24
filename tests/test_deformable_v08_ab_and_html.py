"""Tests for trajectory-forge v0.8 HermiteTwin A/B worker, video renderer, and offline HTML."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import sys
import numpy as np

from embodied_eval_os.core.reports.deformable_v08_router import certify_v08_rollout_dir
from scenario_forge.adapters.deformable_v08 import build_v08_ab_scenario_package
from trajectory_forge.adapters.deformable_workers_v08 import (
    render_dual_renderer_comparison_video,
    render_task_ab_comparison_video,
    run_single_v08_ab_task,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_hermitetwin_v08_comparison_html import build_hermitetwin_comparison_html


def test_v08_single_task_ab_rollout_video_and_offline_html(tmp_path: Path) -> None:
    pkg_receipt = build_v08_ab_scenario_package("v08_cloth_corner_fold", tmp_path / "pkgs")
    roll_dir = tmp_path / "rollouts"
    task_out = run_single_v08_ab_task(
        tmp_path / "pkgs" / "v08_cloth_corner_fold",
        roll_dir / "v08_cloth_corner_fold",
        eval_nu=15,
        eval_nv=15,
    )
    assert pkg_receipt["geometry_and_trajectory_frozen"] is True
    assert task_out["firewall_clean"] is True
    with np.load(task_out["main_npz"], allow_pickle=False) as z:
        assert bool(z["privileged_channels_recorded"]) is False

    media_dir = tmp_path / "media"
    dual_res = render_dual_renderer_comparison_video(
        task_id="v08_cloth_corner_fold",
        task_entry=task_out,
        out_media_dir=media_dir,
        tier="fast_cpu",
    )

    roll_summary = {
        "schema_version": "trajectory-forge-v08-ab-rollouts/v0.1",
        "evidence_tier": "FAST_CHECK_CPU_PREVIEW",
        "tasks": {"v08_cloth_corner_fold": task_out},
        "dual_renderer": dual_res,
    }
    roll_json = roll_dir / "ab_rollout_summary.json"
    roll_json.write_text(json.dumps(roll_summary, indent=2), encoding="utf-8")

    cert_json = tmp_path / "cert.json"
    cert = certify_v08_rollout_dir(roll_dir, cert_json)
    assert cert["all_tasks_passed"] is True, f"Protocol C3 failed: {cert}"

    vid_res = render_task_ab_comparison_video(
        task_id="v08_cloth_corner_fold",
        task_entry=task_out,
        cert_entry=cert["tasks"]["v08_cloth_corner_fold"],
        out_media_dir=media_dir,
        tier="fast_cpu",
    )
    # Verify H.264 video via ffprobe
    for mp4_p in (vid_res["mp4_path"], dual_res["mp4_path"]):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,nb_frames", "-of", "json", mp4_p],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(probe.stdout)["streams"][0]
        assert info["codec_name"] == "h264"
        assert info["pix_fmt"] == "yuv420p"
        assert int(info["nb_frames"]) > 10

    media_manifest = tmp_path / "media_manifest.json"
    media_manifest.write_text(
        json.dumps({"tasks": {"v08_cloth_corner_fold": vid_res}, "dual_renderer": dual_res}),
        encoding="utf-8",
    )
    html_p = build_hermitetwin_comparison_html(
        rollout_summary_path=roll_json,
        cert_summary_path=cert_json,
        media_manifest_path=media_manifest,
        out_html_path=tmp_path / "showcase.html",
    )
    text = html_p.read_text(encoding="utf-8")
    assert "data:video/mp4;base64," in text
    assert "data:image/png;base64," in text
    assert len(re.findall(r"https?://", text)) == 0
