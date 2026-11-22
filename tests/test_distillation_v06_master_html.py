"""Verify the 100%-offline synchronized dual-video HTML master report for distillation_v06."""

from __future__ import annotations

from pathlib import Path
import re

import pytest


HTML_PATH = (
    Path(__file__).resolve().parents[1]
    / "external_artifacts"
    / "reports"
    / "distillation_v06_bilibili_aligned_master_20261121.html"
)


def test_v06_master_html_is_100_percent_offline_and_embeds_dual_videos_and_16_cards() -> None:
    if not HTML_PATH.is_file():
        pytest.skip("Run scripts/build_distillation_v06_master_html.py first")
    txt = HTML_PATH.read_text(encoding="utf-8")
    # Zero external http:// or https:// URLs anywhere in the offline report
    assert re.search(r"https?://", txt) is None
    assert txt.count("data:video/mp4;base64,") >= 6  # 1 Bilibili + 1 Isaac Master + 4 Macro close-ups
    assert txt.count("data:image/png;base64,") >= 19  # 3 Cycles AgX sheets + 16 Bilibili-vs-Isaac cards
    assert "APPROVED" in txt
    assert "jumpStage(" in txt

