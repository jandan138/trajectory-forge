"""Unit tests for M3 distillation_v06 12-stage closed-loop trajectory, zero-penetration gate, and multi-camera MP4s."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

M3_OUT = _REPO_ROOT / "external_artifacts" / "distillation_v06_m3"


def test_v06_zero_penetration_and_12stage_coaxial_telemetry() -> None:
    audit_p = M3_OUT / "mesh_penetration_report_v06.json"
    npz_p = M3_OUT / "v06_trajectory_telemetry.npz"
    assert audit_p.is_file() and npz_p.is_file(), "Run scripts/run_m3_distillation_v06_isaac_native.py first"

    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    assert audit["overall_passed"] is True
    assert audit["stride"] == 1
    assert audit["fps"] == 30
    assert audit["max_penetration_mm"] == 0.0
    assert audit["hose_table_penetration_mm"] == 0.0
    assert audit["hose_obstacle_penetration_mm"] == 0.0
    assert audit["hose_robot_arm_penetration_mm"] == 0.0
    assert audit["s10_steady_state_metrics"]["thermometer_bulb_flush_err_mm"] <= 1.0
    assert 14.0 <= audit["s10_steady_state_metrics"]["condenser_downward_pitch_deg"] <= 16.0

    with np.load(npz_p, allow_pickle=False) as z:
        assert z["qpos"].shape == (720, 18)
        assert z["hose_inlet_verts"].shape == (720, 64 * 24, 3)
        assert z["hose_outlet_verts"].shape == (720, 64 * 24, 3)


def test_v06_isaac_native_master_and_4_macro_videos_exist() -> None:
    rec_p = M3_OUT / "v06_recording_manifest.json"
    if not rec_p.is_file():
        pytest.skip("Run full GPU Isaac Sim 6.0.1 recording first")
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    assert Path(rec["master_video_mp4"]).is_file()
    assert Path(rec["master_video_mp4"]).stat().st_size > 100_000
    assert len(rec["macro_videos_mp4"]) == 4
    for mp in rec["macro_videos_mp4"].values():
        assert Path(mp).is_file() and Path(mp).stat().st_size > 30_000

