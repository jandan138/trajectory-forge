"""Anti-cheat provenance & counterfactual falsification tests for distillation_v07 3-Mechanism Dynamic Slice."""

from __future__ import annotations

import json
from pathlib import Path


SLICE_DIR = Path(__file__).resolve().parents[1] / "external_artifacts" / "distillation_v07_3mech_slice"


def test_v07_full_dynamic_provenance_and_zero_friction_falsification() -> None:
    sp = SLICE_DIR / "v07_slice_master_summary.json"
    assert sp.is_file(), "Run scripts/run_v07_hardcore_3mech_slice.py first"
    s = json.loads(sp.read_text(encoding="utf-8"))

    assert s["all_passed"] is True
    assert s["distinct_cold_start_pids"] == 6, "All 6 rollouts (3 nominal + 3 zero_friction) must run in isolated OS PIDs"

    for mid in (
        "mech_01_scissor_jack_handwheel_lift",
        "mech_02_bosshead_flask_clamp_multiturn",
        "mech_03_3prong_clamp_condenser_native_hoses",
    ):
        ep = s["episodes"][mid]
        nom = ep["nominal"]
        zf = ep["zero_friction"]

        assert nom["metrics"]["any_kinematic_manipulated"] is False
        assert nom["metrics"]["task_passed"] is True
        assert nom["metrics"]["median_active_pad_surface_gap_mm"] <= 2.0
        assert nom["metrics"]["peak_finger_normal_force_n"] >= 8.0
        assert zf["metrics"]["task_passed"] is False
        assert ep["counterfactual_falsified"] is True
        assert int(nom["provenance"]["process_pid"]) != int(zf["provenance"]["process_pid"])
        assert Path(nom["provenance"]["npz_path"]).is_file()
        assert Path(zf["provenance"]["npz_path"]).is_file()
        assert Path(nom["videos"]["triple_sync"]).is_file()

