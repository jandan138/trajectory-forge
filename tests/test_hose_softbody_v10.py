"""Regression test verifying the 3-way Hose Softbody v1.0 cold-start benchmark summary & offline HTML report."""

from __future__ import annotations

import json
from pathlib import Path
import re


def test_hose_softbody_v10_3way_gates_and_offline_html() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sum_p = repo_root / "external_artifacts" / "hose_softbody_v10_3way" / "hose_v10_3way_master_summary.json"
    html_p = repo_root / "external_artifacts" / "reports" / "hose_softbody_v10_3way_ab_report_20261125.html"
    assert sum_p.is_file(), f"Missing 3-way summary: {sum_p}"
    assert html_p.is_file(), f"Missing 3-way offline HTML: {html_p}"

    s = json.loads(sum_p.read_text(encoding="utf-8"))
    assert int(s["distinct_cold_start_pids"]) >= 7
    tr = s["trials"]

    # Baseline bead chain must fail elastic rebound, 90-deg torsion, and barb self-lock
    assert tr["baseline_bead_chain"]["exp1_cantilever"]["elastic_rebound_pct"] < 5.0
    assert abs(tr["baseline_bead_chain"]["exp2_torsion_90deg"]["tip_twist_steady_deg"]) < 5.0
    assert tr["baseline_bead_chain"]["exp3_barb_fit"]["self_lock_hang_passed"] is False

    # Route A (D6-DER + Method-B micro-spring cuff) must pass all 3 physics tests + reversal detection
    ra = tr["route_a_d6_der"]
    assert ra["b3_weld_joint_count"] == 0
    assert ra["exp1_cantilever"]["elastic_rebound_pct"] >= 85.0
    assert ra["exp2_torsion_90deg"]["transmission_ratio_pct"] >= 90.0
    assert ra["exp3_barb_fit"]["self_lock_hang_passed"] is True
    assert ra["exp3_barb_fit"]["pull_off_passed"] is True
    assert ra["exp3_barb_fit"]["peak_cuff_radial_force_n"] >= 8.0
    assert ra["exp3_barb_fit"]["flow_bottom_in_top_out_ok"] is True
    assert tr["route_a_wrong_port"]["exp3_barb_fit"]["flow_bottom_in_top_out_ok"] is False

    # Route B (Adaptive Hollow PhysX GPU FEM) must pass all 3 physics tests + real hoop expansion + mu=0 falsification + reversal
    rb = tr["route_b_physx_fem"]
    assert rb["b3_weld_joint_count"] == 0
    assert rb["exp1_cantilever"]["elastic_rebound_pct"] >= 85.0
    assert rb["exp2_torsion_90deg"]["transmission_ratio_pct"] >= 95.0
    assert rb["exp3_barb_fit"]["self_lock_hang_passed"] is True
    assert rb["exp3_barb_fit"]["pull_off_passed"] is True
    assert rb["exp3_barb_fit"]["peak_fem_hoop_strain_pct"] >= 15.0
    assert rb["exp3_barb_fit"]["peak_fem_pinch_ovalization_pct"] >= 30.0
    assert rb["exp3_barb_fit"]["flow_bottom_in_top_out_ok"] is True
    assert tr["route_b_zero_friction"]["exp3_barb_fit"]["self_lock_hang_passed"] is False
    assert tr["route_b_wrong_port"]["exp3_barb_fit"]["flow_bottom_in_top_out_ok"] is False

    # Offline self-contained HTML check
    txt = html_p.read_text(encoding="utf-8")
    assert "data:video/mp4;base64," in txt
    assert len(re.findall(r"https?://", txt)) == 0

