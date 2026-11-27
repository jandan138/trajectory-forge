"""Pytest verification for distillation_v09 Full S01-S12 One-Shot Method-B Assembly & Twist-Pull Disassembly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_SPEC = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\distillation_v09_package\scenario_package\semantics\v09_scene_spec.json"
)
SUMMARY_JSON = REPO_ROOT / "external_artifacts" / "distillation_v09_oneshot" / "v09_master_summary.json"


def test_v09_cpu_schedule_ik_and_clearance_gate() -> None:
    from trajectory_forge.adapters.legacy.distillation_v09.v09_s01_s12_schedule_planner import (
        audit_v09_schedule_cpu,
        synthesize_v09_s01_s12_schedule,
    )

    spec = json.loads(PKG_SPEC.read_text(encoding="utf-8"))
    bL = np.asarray(spec["robot_bases"]["franka_left"]["xyz"], dtype=np.float64)
    qL = np.asarray(spec["robot_bases"]["franka_left"]["wxyz"], dtype=np.float64)
    bR = np.asarray(spec["robot_bases"]["franka_right"]["xyz"], dtype=np.float64)
    qR = np.asarray(spec["robot_bases"]["franka_right"]["wxyz"], dtype=np.float64)

    sched = synthesize_v09_s01_s12_schedule(bL, qL, bR, qR, fps=30)
    audit = audit_v09_schedule_cpu(sched, bL, qL, bR, qR)
    assert audit["passed"] is True, f"CPU IK/Clearance gate failed: {audit}"
    assert audit["max_dqL_rad_per_frame"] <= 0.55
    assert audit["max_dqR_rad_per_frame"] <= 0.55
    assert audit["min_arm_to_arm_dist_mm"] >= 80.0


def test_v09_oneshot_gpu_master_summary() -> None:
    assert SUMMARY_JSON.is_file(), "Run scripts/run_v09_s01_s12_oneshot.py first"
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    assert summary["all_passed"] is True
    assert summary["nominal_passed"] is True
    assert summary["counterfactual_falsified"] is True
    assert summary["distinct_cold_start_pids"] >= 2

    nom_m = summary["episodes"]["nominal"]["metrics"]
    assert nom_m["any_kinematic_manipulated"] is False
    assert nom_m["weld_joint_count"] == 0
    sm = nom_m["s10_steady_state"]
    assert sm["passed"] is True
    assert sm["j01_flask_head_rel_err_mm"] <= 2.5
    assert sm["j01_flask_head_coax_deg"] <= 2.8
    assert sm["j02_head_cond_rel_err_mm"] <= 2.5
    assert sm["j02_head_cond_coax_deg"] <= 3.0
    assert sm["j03_cond_tail_rel_err_mm"] <= 2.5
    assert sm["j03_cond_tail_coax_deg"] <= 3.0
    assert sm["erlenmeyer_tilt_deg"] <= 1.5
    assert sm["drip_in_mouth_xy_err_mm"] <= 6.0
    assert sm["thermometer_bulb_flush_err_mm"] <= 1.5
    assert sm["max_hose_bench_penetration_mm"] <= 1.0
    assert nom_m["s11_s12_disassembly"]["passed"] is True

