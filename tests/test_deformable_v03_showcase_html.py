"""Pure-CPU tests for v0.3 offline 10-task HTML showcase builder & media readiness gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


_SCRIPTS_DIR = Path(r"I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge\scripts")


def _load_mod(filename: str):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), _SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_deformable_v03_showcase_html_zero_external_urls(tmp_path: Path) -> None:
    html_mod = _load_mod("build_deformable_v03_showcase_html.py")
    cert = {
        "generated_at_utc": "2026-11-20T02:00:00Z",
        "tasks": {
            "coffee_croissant_tray_to_plate": {
                "structural_family": "soft_solid",
                "certification_tier": "Gold",
                "passing_engine_count": 4,
                "total_engines": 4,
                "protocol_a": {"passed": True, "replay_passed_count": 5, "replay_total_count": 5, "zero_friction_intact_count": 0, "zero_friction_total_count": 1},
                "freeze_lock": {"passed": True, "passed_count": 3, "total_count": 3},
                "engine_summaries": {
                    eng: {"intact_task_success": True, "damage_score": 0.0, "peak_normal_force_n": 3.1, "polar_stretch_q95": 0.08}
                    for eng in ("mujoco4", "newton18", "genesis", "isaac6")
                },
            }
        },
    }
    cl = {
        "cases": [
            {
                "case_id": "case_01_smt_obstacle_remediation",
                "task_id": "smt_ribbon_cable_clip_route",
                "defect_type": "scene_defect",
                "v0_failed": {"summary": {"intact_task_success": False, "obstacle_collision": True, "damage_score": 0.0, "peak_normal_force_n": 1.0}},
                "eos_diagnosis": {"target_repo": "scenario-forge", "reason_code": "scene_obstacle_blocking_target"},
                "auto_remediation": {"actions": ["Relocated ESDTray"]},
                "v1_certified": {"summary": {"intact_task_success": True, "damage_score": 0.0, "peak_normal_force_n": 2.4, "final_target_dist_m": 0.012}},
            }
        ]
    }
    out_html = html_mod.render_deformable_v03_showcase_html(
        cert,
        cl,
        v02_media=tmp_path,
        v03_media=tmp_path,
        m2_v03=tmp_path,
        m3_v03=tmp_path,
    )
    assert "CLOSED-LOOP HEALED" in out_html
    assert "Gold (4/4)" in out_html
    assert "http://" not in out_html and "https://" not in out_html


def test_render_v03_media_readiness_0pct_gpu(tmp_path: Path) -> None:
    media_mod = _load_mod("render_deformable_v03_showcase_media.py")
    m2 = tmp_path / "m2"
    m3 = tmp_path / "m3"
    for tid in media_mod.V03_TASKS:
        pkg = m2 / tid / "scenario_package"
        (pkg / "scene").mkdir(parents=True, exist_ok=True)
        (pkg / "scene" / "main.usda").write_text("#usda 1.0\n", encoding="utf-8")
        (pkg / "scene" / "instances.yaml").write_text(
            "instances:\n  - role: manipulated_deformable\n    asset_id: dummy_def\n",
            encoding="utf-8",
        )
        tex_dir = pkg / "assets" / "dummy_def" / "textures"
        tex_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pkg / "assets" / "dummy_def" / "skinning.npz", mode=np.array(["tet_bary"]))
        for tf in ("albedo.png", "normal.png", "roughness.png"):
            (tex_dir / tf).write_bytes(b"PNG")
        npz = m3 / tid / "mujoco4" / f"{tid}__mujoco4__gentle_s0.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, action=np.zeros((5, 9)), deformable_rest_points=np.zeros((8, 3)))

    rep = tmp_path / "cert.json"
    rep.write_text(json.dumps({"tasks": {t: {"certification_tier": "Gold"} for t in media_mod.V03_TASKS}}), encoding="utf-8")
    res = media_mod.check_render_readiness(m2_root=m2, m3_root=m3, m4_report=rep)
    assert res["ready"] is True

