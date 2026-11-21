"""Pure-CPU fixture tests for v0.2 deformable showcase HTML & media readiness gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


_SCRIPTS_DIR = Path(r"I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge\scripts")


def _load_script_module(filename: str):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), _SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_deformable_v02_showcase_html_offline_fixture(tmp_path: Path) -> None:
    html_mod = _load_script_module("build_deformable_v02_showcase_html.py")
    synthetic_report = {
        "generated_at_utc": "2026-11-19T12:00:00Z",
        "tasks": {
            "coffee_bar_fragile_pastry_plate": {
                "certification_tier": "Gold",
                "passing_engine_count": 4,
                "total_engines": 4,
                "protocol_a": {
                    "passed": True,
                    "replay_passed_count": 5,
                    "replay_total_count": 5,
                    "zero_friction_intact_count": 0,
                    "zero_friction_total_count": 1,
                },
                "freeze_lock": {"passed": True, "passed_count": 3, "total_count": 3},
                "engine_summaries": {
                    eng: {
                        "structural_family": "soft_solid",
                        "intact_task_success": True,
                        "polar_stretch_q95": 0.10,
                        "naive_f_minus_i_q95": 0.87,
                        "peak_normal_force_n": 3.2,
                        "damage_score": 0.0,
                        "final_target_dist_m": 0.01,
                    }
                    for eng in ("mujoco4", "newton18", "genesis", "isaac6")
                },
                "counterfactual_checks": [
                    {"episode_id": "crush_s0", "damage_score": 1.0, "intact_task_success": False}
                ],
                "counterfactual_routing": {"target_repo": "trajectory-forge"},
            }
        },
    }
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "coffee_bar_fragile_pastry_plate.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isomstub")
    html_out = html_mod.render_deformable_v02_showcase_html(synthetic_report, media_dir=media_dir)
    assert "Gold (4/4)" in html_out
    assert "data:video/mp4;base64," in html_out
    assert "https://" not in html_out and "http://" not in html_out


def test_render_media_check_ready_0pct_gpu(tmp_path: Path) -> None:
    media_mod = _load_script_module("render_deformable_v02_showcase_media.py")
    m2 = tmp_path / "m2"
    m3 = tmp_path / "m3"
    for tid in media_mod.TASKS:
        usd = m2 / tid / "scenario_package" / "scene" / "main.usda"
        usd.parent.mkdir(parents=True, exist_ok=True)
        usd.write_text("#usda 1.0\n", encoding="utf-8")
        for eng in ("mujoco4", "newton18", "genesis", "isaac6"):
            npz = m3 / tid / eng / f"{tid}__{eng}__gentle_s0.npz"
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz,
                action=np.zeros((5, 9), dtype=np.float64),
                deformable_rest_points=np.zeros((8, 3), dtype=np.float64),
                ee_pos=np.zeros((5, 3), dtype=np.float64),
            )
    rep = tmp_path / "eos_report.json"
    rep.write_text(json.dumps({"tasks": {t: {"certification_tier": "Gold"} for t in media_mod.TASKS}}), encoding="utf-8")
    res = media_mod.check_render_readiness(m2_root=m2, m3_root=m3, m4_report=rep)
    assert res["ready"] is True

