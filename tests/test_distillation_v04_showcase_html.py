"""Pure-CPU verification tests for the 100%-offline distillation_v04 HTML showcase and RTX videos."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = REPO_ROOT / "external_artifacts" / "reports" / "distillation_v04_sop_showcase_20261120.html"
MEDIA_DIR = REPO_ROOT / "external_artifacts" / "distillation_v04_media"

V04_TASKS = (
    "distill_01_burner_ring_flask_mount",
    "distill_02_head_taper_coaxial_mate",
    "distill_03_thermometer_septum_align",
    "distill_04_bimanual_hose_barb_couple",
    "distill_05_condenser_adapter_receiver_train",
    "distill_06_reverse_order_disassembly",
    "distill_07_full_assembly_disassembly_sop",
)


def test_distillation_v04_media_manifest_and_mp4s() -> None:
    manifest_p = MEDIA_DIR / "media_manifest_v04.json"
    assert manifest_p.is_file(), f"Missing RTX media manifest: {manifest_p}"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert Path(manifest["contact_sheet"]).is_file()
    for tid in V04_TASKS:
        assert tid in manifest["videos"], f"Missing video entry for {tid}"
        mp4_p = Path(manifest["videos"][tid]["mp4_path"])
        assert mp4_p.is_file() and mp4_p.stat().st_size > 20_000


def test_distillation_v04_html_100_percent_offline_and_complete() -> None:
    assert HTML_PATH.is_file(), f"Missing HTML showcase: {HTML_PATH}"
    txt = HTML_PATH.read_text(encoding="utf-8")
    assert "http://" not in txt, "External http:// link forbidden in offline HTML!"
    assert "https://" not in txt, "External https:// link forbidden in offline HTML!"
    assert txt.count("data:video/mp4;base64,") == 7
    for tid in V04_TASKS:
        assert tid in txt
    for cid in (
        "case_01_water_flow_reversed",
        "case_02_nozzle_bending_fracture",
        "case_03_thermometer_bulb_offset",
        "case_04_taper_joint_tilt_jam",
        "case_05_sop_order_violation",
    ):
        assert cid in txt

