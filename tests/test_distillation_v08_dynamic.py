"""Anti-cheat provenance, counterfactual falsification, Method A/B wedge & pull-out disassembly tests for distillation_v08."""

from __future__ import annotations

import json
from pathlib import Path


SLICE_DIR = Path(__file__).resolve().parents[1] / "external_artifacts" / "distillation_v08_4taper_slice"


def test_v08_taper_insertion_keck_ab_wedge_and_disassembly() -> None:
    sp = SLICE_DIR / "v08_slice_master_summary.json"
    assert sp.is_file(), "Run scripts/run_v08_taper_keck_slice.py first"
    s = json.loads(sp.read_text(encoding="utf-8"))

    assert s["all_passed"] is True
    assert s["distinct_cold_start_pids"] == 13

    for jid in (
        "joint_01_head_into_flask_24_29_vertical",
        "joint_02_condenser_to_sidearm_15deg_keck",
        "joint_03_tail_adapter_to_condenser_keck",
        "joint_04_thermometer_screw_cap_flush_bulb",
    ):
        ep = s["episodes"][jid]
        nom = ep["nominal"]
        zf = ep["zero_friction"]

        assert nom["metrics"]["any_kinematic_manipulated"] is False
        assert nom["metrics"]["task_passed"] is True
        assert zf["metrics"]["task_passed"] is False
        assert ep["counterfactual_falsified"] is True
        assert int(nom["provenance"]["process_pid"]) != int(zf["provenance"]["process_pid"])
        assert Path(nom["videos"]["triple_sync"]).is_file()

    assert s["episodes"]["joint_04_thermometer_screw_cap_flush_bulb"]["nominal"]["metrics"]["thermometer_bulb_flush_err_mm"] <= 2.5

    ab05 = s["ab_05_unclamped_15deg_comparison"]
    assert ab05["passed"] is True
    assert ab05["method_a_rigid"]["metrics"]["task_passed"] is False
    assert ab05["method_a_rigid"]["metrics"]["final_taper_3d_err_mm"] >= 25.0
    assert ab05["method_b_micro_spring"]["metrics"]["task_passed"] is True
    assert ab05["method_b_micro_spring"]["metrics"]["peak_wedge_radial_force_n"] >= 6.0

    ab06 = s["ab_06_disassembly_comparison"]
    assert ab06["passed"] is True
    assert ab06["keck_locked_pull"]["metrics"]["task_passed"] is False
    assert ab06["keck_locked_pull"]["metrics"]["max_extract_dz_mm"] <= 12.0
    assert ab06["method_b_straight_pull"]["metrics"]["task_passed"] is True
    assert ab06["method_b_twist_and_pull"]["metrics"]["task_passed"] is True
    assert ab06["axial_force_reduction_pct"] >= 20.0
