"""Cross-platform end-to-end HermiteTwin v0.8 A/B pipeline & comparison HTML release runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from convert_asset.deformable_v08_hermite import EvidenceTierV08
from embodied_eval_os.core.reports.deformable_v08_router import certify_v08_rollout_dir
from scenario_forge.adapters.deformable_v08 import build_all_v08_ab_packages
from trajectory_forge.adapters.legacy.deformable_workers_v08 import (
    render_dual_renderer_comparison_video,
    render_task_ab_comparison_video,
    run_all_v08_ab_experiments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hermitetwin_v08_comparison_html import build_hermitetwin_comparison_html


def run_v08_release(
    *,
    sandbox_root: Path,
    tier: str = "fast_cpu",
    blender_bin: str | None = None,
) -> dict[str, Any]:
    sf_pkgs = sandbox_root / "scenario-forge" / "external_artifacts" / "deformable_v08_ab_packages"
    tf_rollouts = sandbox_root / "trajectory-forge" / "external_artifacts" / "deformable_v08_rollouts"
    tf_media = sandbox_root / "trajectory-forge" / "external_artifacts" / "deformable_v08_media"
    eos_cert_json = sandbox_root / "embodied-eval-os" / "external_artifacts" / "deformable_v08_certification" / "v08_hermitetwin_certification_summary.json"
    html_out = sandbox_root / "trajectory-forge" / "external_artifacts" / "reports" / "hermitetwin_v08_comparison_showcase.html"

    ev_tier = (
        EvidenceTierV08.STUDIO_CYCLES_AGX_NATIVE_VERIFIED.value
        if tier == "studio_cycles"
        else EvidenceTierV08.FAST_CHECK_CPU_PREVIEW.value
    )

    # 1. scenario-forge frozen-hash A/B scenario packages
    build_all_v08_ab_packages(sf_pkgs)

    # 2. trajectory-forge A/B rollouts + CMA-ES/adjoint SysID + Protocol A/Freeze Lock/ZF CF
    roll_summary = run_all_v08_ab_experiments(
        packages_root=sf_pkgs,
        out_root=tf_rollouts,
        evidence_tier=ev_tier,
    )

    # 3. Render Dual-Renderer (PBR vs Hermite 3DGS) video & PNGs for EOS affine residual audit
    dual_res = render_dual_renderer_comparison_video(
        task_id="v08_cloth_corner_fold",
        task_entry=roll_summary["tasks"]["v08_cloth_corner_fold"],
        out_media_dir=tf_media,
        tier=tier,
    )
    roll_summary["dual_renderer"] = dual_res
    roll_summary_path = tf_rollouts / "ab_rollout_summary.json"
    roll_summary_path.write_text(json.dumps(roll_summary, indent=2), encoding="utf-8")

    # 4. embodied-eval-os Protocol C3 + Protocol A + Freeze Lock certification
    cert_summary = certify_v08_rollout_dir(tf_rollouts, eos_cert_json)

    # 5. Render synchronized 3-up A/B videos + peak Before/After curtain PNGs
    media_tasks: dict[str, Any] = {}
    for tid, t_entry in roll_summary["tasks"].items():
        media_tasks[tid] = render_task_ab_comparison_video(
            task_id=tid,
            task_entry=t_entry,
            cert_entry=cert_summary["tasks"][tid],
            out_media_dir=tf_media,
            tier=tier,
            blender_bin=blender_bin,
        )
    media_manifest = {
        "schema_version": "trajectory-forge-v08-media/v0.1",
        "tier": tier,
        "tasks": media_tasks,
        "dual_renderer": dual_res,
    }
    media_manifest_path = tf_media / "media_manifest.json"
    media_manifest_path.write_text(json.dumps(media_manifest, indent=2), encoding="utf-8")

    # 6. Build single-file offline interactive HTML showcase
    out_html = build_hermitetwin_comparison_html(
        rollout_summary_path=roll_summary_path,
        cert_summary_path=eos_cert_json,
        media_manifest_path=media_manifest_path,
        out_html_path=html_out,
    )

    return {
        "evidence_tier": ev_tier,
        "all_tasks_passed": cert_summary["all_tasks_passed"],
        "html_path": out_html.resolve().as_posix(),
        "html_bytes": out_html.stat().st_size,
        "cert_json": eos_cert_json.resolve().as_posix(),
        "media_manifest": media_manifest_path.resolve().as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sandbox-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Root directory containing ConvertAsset, scenario-forge, trajectory-forge, embodied-eval-os.",
    )
    parser.add_argument("--tier", choices=("fast_cpu", "studio_cycles"), default="fast_cpu")
    parser.add_argument("--blender-bin", type=str, default=None)
    args = parser.parse_args()
    res = run_v08_release(sandbox_root=args.sandbox_root, tier=args.tier, blender_bin=args.blender_bin)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
