from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _write_fake_mp4(path: Path, payload: bytes = b"synthetic-mp4-stream") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    path.write_bytes(header + payload)


def _write_fake_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    path.write_bytes(png_1x1)


def _write_fake_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK\x03\x04synthetic-npz-payload")


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _populate_synthetic_repo(root: Path) -> None:
    media_dirs = {
        "real2sim": (
            root / "external_artifacts/rollouts/jobs/real2sim_office_desk_riser_organize_a1_synth/media",
            [
                "t0_initial.png",
                "stage_0_stage0_drawer_opened.png",
                "stage_1_stage1_chocopie_stored.png",
                "stage_2_stage2_pen_stored.png",
                "stage_3_stage3_drawer_closed.png",
                "stage_4_stage4_mug_placed_upright.png",
                "final.png",
            ],
        ),
        "ingun": (
            root / "external_artifacts/rollouts/jobs/electronics_qa_ingun_probe_lock_a1_synth/media",
            [
                "t0_initial.png",
                "stage_0_stage0_pcba_aligned.png",
                "stage_1_stage1_hood_closed.png",
                "stage_2_stage2_lever_locked.png",
                "stage_3_stage3_probe_enabled.png",
                "stage_4_stage4_test_started.png",
                "final.png",
            ],
        ),
        "creatista_r3": (
            root / "external_artifacts/rollouts/jobs/r3_creatista_final_media/media",
            ["t0_initial.png", "final.png"],
        ),
        "creatista_r4": (
            root / "external_artifacts/rollouts/jobs/specialty_coffee_bar_creatista_prep_a1_synth/media",
            [
                "t0_initial.png",
                "stage_0_stage0_tray_extended.png",
                "stage_1_stage1_cup_seated.png",
                "stage_2_stage2_capsule_inserted.png",
                "stage_3_stage3_lever_locked.png",
                "stage_4_stage4_dial_selected.png",
                "stage_5_stage5_dial_pressed.png",
                "final.png",
            ],
        ),
        "brother_r3": (
            root / "external_artifacts/rollouts/jobs/r3_brother_final_media/media",
            ["t0_initial.png", "final.png"],
        ),
        "brother_mode2_pose": (
            root / "external_artifacts/rollouts/mode2_pose/jobs/modern_office_brother_calibrate_a1_synth/media",
            [
                "t0_initial.png",
                "stage_0_stage0_media_loaded.png",
                "stage_1_stage1_cassette_closed.png",
                "stage_2_stage2_sheet_aligned.png",
                "stage_3_stage3_lid_closed.png",
                "stage_4_stage4_copy_started.png",
                "final.png",
            ],
        ),
        "brother_mode1_codesign": (
            root / "external_artifacts/rollouts/jobs/modern_office_brother_calibrate_a1_synth/media",
            [
                "t0_initial.png",
                "stage_0_stage0_media_loaded.png",
                "stage_1_stage1_cassette_closed.png",
                "stage_2_stage2_sheet_aligned.png",
                "stage_3_stage3_lid_closed.png",
                "stage_4_stage4_copy_started.png",
                "final.png",
            ],
        ),
    }
    for tag, (mdir, pngs) in media_dirs.items():
        _write_fake_mp4(mdir / "progression.mp4", tag.encode("utf-8"))
        for png in pngs:
            _write_fake_png(mdir / png)

    npz_paths = [
        root / "external_artifacts/rollouts/episodes/real2sim_office_desk_riser_organize_real2sim_office_desk_riser_organize_a1_synth/episode_trajectory.npz",
        root / "external_artifacts/rollouts/episodes/electronics_qa_ingun_probe_lock_electronics_qa_ingun_probe_lock_a1_synth/episode_trajectory.npz",
        root / "external_artifacts/rollouts/episodes/specialty_coffee_bar_creatista_prep_specialty_coffee_bar_creatista_prep_a1_synth/episode_trajectory.npz",
        root / "external_artifacts/rollouts/mode2_pose/episodes/modern_office_brother_calibrate_modern_office_brother_calibrate_a1_synth/episode_trajectory.npz",
        root / "external_artifacts/rollouts/episodes/modern_office_brother_calibrate_modern_office_brother_calibrate_a1_synth/episode_trajectory.npz",
    ]
    for npz in npz_paths:
        _write_fake_npz(npz)

    reports = root / "external_artifacts/rollouts/reports"
    _write_json(
        reports / "real2sim_office_desk_riser_organize_solve_report.json",
        {
            "package_id": "real2sim_office_desk_riser_organize",
            "succeeded": True,
            "attempts_used": 1,
            "synthesis_result": {
                "steps_executed": 1448,
                "final_score": 1.0,
                "snapshots_used": 0,
                "firewall_verified": True,
                "stage_progression": [
                    {"stage_index": 0, "metric_id": "stage0_drawer_opened", "score": 0.2, "step_index": 240},
                    {"stage_index": 1, "metric_id": "stage1_chocopie_stored", "score": 0.4, "step_index": 548},
                    {"stage_index": 2, "metric_id": "stage2_pen_stored", "score": 0.6, "step_index": 764},
                    {"stage_index": 3, "metric_id": "stage3_drawer_closed", "score": 0.8, "step_index": 1036},
                    {"stage_index": 4, "metric_id": "stage4_mug_placed_upright", "score": 1.0, "step_index": 1328},
                ],
                "trajectory_manifest": {
                    "schema_version": "trajectory-forge-episode/v0.1",
                    "continuous_from_t0": True,
                    "privileged_channels_recorded": False,
                    "recorded_channels": ["proprio", "action", "base_target", "torso_lift", "timestamps_s", "rgb_head"],
                },
            },
        },
    )
    _write_json(
        reports / "electronics_qa_ingun_probe_lock_solve_report.json",
        {
            "package_id": "electronics_qa_ingun_probe_lock",
            "succeeded": True,
            "attempts_used": 1,
            "synthesis_result": {
                "steps_executed": 1757,
                "final_score": 1.0,
                "snapshots_used": 0,
                "firewall_verified": True,
                "stage_progression": [
                    {"stage_index": 0, "metric_id": "stage0_pcba_aligned", "score": 0.2, "step_index": 268},
                    {"stage_index": 1, "metric_id": "stage1_hood_closed", "score": 0.4, "step_index": 1418},
                    {"stage_index": 2, "metric_id": "stage2_lever_locked", "score": 0.6, "step_index": 1588},
                    {"stage_index": 3, "metric_id": "stage3_probe_enabled", "score": 0.8, "step_index": 1592},
                    {"stage_index": 4, "metric_id": "stage4_test_started", "score": 1.0, "step_index": 1722},
                ],
                "trajectory_manifest": {
                    "schema_version": "trajectory-forge-episode/v0.1",
                    "continuous_from_t0": True,
                    "privileged_channels_recorded": False,
                    "recorded_channels": ["proprio", "action", "base_target", "torso_lift", "timestamps_s", "rgb_head"],
                },
            },
        },
    )
    _write_json(
        reports / "specialty_coffee_bar_creatista_prep_solve_report.json",
        {
            "package_id": "specialty_coffee_bar_creatista_prep",
            "succeeded": False,
            "attempts_used": 1,
            "history": [
                {
                    "status": "failed",
                    "final_score": 0.5,
                    "steps_executed": 1119,
                    "snapshots_used": 5,
                    "blocking_stage": 3,
                    "failure_code": "UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT",
                }
            ],
        },
    )
    _write_json(
        reports / "specialty_coffee_bar_creatista_prep_r4_solve_report.json",
        {
            "package_id": "specialty_coffee_bar_creatista_prep",
            "succeeded": True,
            "attempts_used": 1,
            "remediation_policy": {"mode": "physics_tuning", "allow_pose_adjustment": False},
            "synthesis_result": {
                "steps_executed": 1775,
                "final_score": 1.0,
                "snapshots_used": 0,
                "firewall_verified": True,
                "stage_progression": [
                    {"stage_index": 0, "metric_id": "stage0_tray_extended", "score": 0.16667, "step_index": 160},
                    {"stage_index": 1, "metric_id": "stage1_cup_seated", "score": 0.33333, "step_index": 448},
                    {"stage_index": 2, "metric_id": "stage2_capsule_inserted", "score": 0.5, "step_index": 748},
                    {"stage_index": 3, "metric_id": "stage3_lever_locked", "score": 0.66667, "step_index": 896},
                    {"stage_index": 4, "metric_id": "stage4_dial_selected", "score": 0.83333, "step_index": 1620},
                    {"stage_index": 5, "metric_id": "stage5_dial_pressed", "score": 1.0, "step_index": 1748},
                ],
                "trajectory_manifest": {
                    "schema_version": "trajectory-forge-episode/v0.1",
                    "continuous_from_t0": True,
                    "privileged_channels_recorded": False,
                    "recorded_channels": ["proprio", "action", "base_target", "torso_lift", "timestamps_s", "rgb_head"],
                },
            },
        },
    )
    _write_json(
        reports / "modern_office_brother_calibrate_r4_mode2_strict_report.json",
        {
            "package_id": "modern_office_brother_calibrate",
            "succeeded": False,
            "attempts_used": 1,
            "remediation_policy": {"mode": "physics_tuning", "allow_pose_adjustment": False},
            "history": [
                {
                    "status": "failed",
                    "final_score": 0.0,
                    "steps_executed": 1280,
                    "snapshots_used": 4,
                    "blocking_stage": 0,
                    "failure_code": "BLOCKED_BY_REMEDIATION_POLICY",
                    "root_failure_code": "UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE",
                    "remediation_category": "pose_adjustment",
                    "required_remediation_mode": "scene_codesign",
                    "requires_pose_adjustment": True,
                    "policy_blocked": True,
                }
            ],
        },
    )
    _write_json(
        reports / "modern_office_brother_calibrate_r4_mode2_pose_report.json",
        {
            "package_id": "modern_office_brother_calibrate",
            "succeeded": True,
            "attempts_used": 1,
            "remediation_policy": {"mode": "physics_tuning", "allow_pose_adjustment": True},
            "synthesis_result": {
                "steps_executed": 1449,
                "final_score": 1.0,
                "snapshots_used": 0,
                "firewall_verified": True,
                "stage_progression": [
                    {"stage_index": 0, "metric_id": "stage0_media_loaded", "score": 0.2, "step_index": 348},
                    {"stage_index": 1, "metric_id": "stage1_cassette_closed", "score": 0.4, "step_index": 552},
                    {"stage_index": 2, "metric_id": "stage2_sheet_aligned", "score": 0.6, "step_index": 919},
                    {"stage_index": 3, "metric_id": "stage3_lid_closed", "score": 0.8, "step_index": 1364},
                    {"stage_index": 4, "metric_id": "stage4_copy_started", "score": 1.0, "step_index": 1420},
                ],
                "trajectory_manifest": {
                    "schema_version": "trajectory-forge-episode/v0.1",
                    "continuous_from_t0": True,
                    "privileged_channels_recorded": False,
                    "recorded_channels": ["proprio", "action", "base_target", "torso_lift", "timestamps_s", "rgb_head"],
                },
            },
        },
    )
    _write_json(
        reports / "modern_office_brother_calibrate_r4_scene_codesign_report.json",
        {
            "package_id": "modern_office_brother_calibrate",
            "succeeded": True,
            "attempts_used": 1,
            "remediation_policy": {"mode": "scene_codesign", "allow_pose_adjustment": False},
            "synthesis_result": {
                "steps_executed": 1566,
                "final_score": 1.0,
                "snapshots_used": 0,
                "firewall_verified": True,
                "stage_progression": [
                    {"stage_index": 0, "metric_id": "stage0_media_loaded", "score": 0.2, "step_index": 404},
                    {"stage_index": 1, "metric_id": "stage1_cassette_closed", "score": 0.4, "step_index": 660},
                    {"stage_index": 2, "metric_id": "stage2_sheet_aligned", "score": 0.6, "step_index": 1068},
                    {"stage_index": 3, "metric_id": "stage3_lid_closed", "score": 0.8, "step_index": 1462},
                    {"stage_index": 4, "metric_id": "stage4_copy_started", "score": 1.0, "step_index": 1536},
                ],
                "trajectory_manifest": {
                    "schema_version": "trajectory-forge-episode/v0.1",
                    "continuous_from_t0": True,
                    "privileged_channels_recorded": False,
                    "recorded_channels": ["proprio", "action", "base_target", "torso_lift", "timestamps_s", "rgb_head"],
                },
            },
        },
    )


def test_build_showcase_html_with_synthetic_bundle(tmp_path: Path) -> None:
    from build_progress_showcase_html import audit_html, build_showcase_html

    _populate_synthetic_repo(tmp_path)
    out_html = tmp_path / "external_artifacts/reports/showcase.html"
    manifest = build_showcase_html(repo_root=tmp_path, out_path=out_html)

    assert out_html.exists()
    manifest_path = out_html.with_name(f"{out_html.stem}_manifest.json")
    assert manifest_path.exists()
    assert manifest["schema_version"] == "trajectory-forge-progress-showcase-manifest/v0.1"
    assert manifest["audit"]["external_url_count"] == 0
    assert manifest["audit"]["mp4_data_uri_count"] == 7
    assert manifest["audit"]["image_data_uri_count"] >= 30

    html_text = out_html.read_text(encoding="utf-8")
    assert audit_html(html_text)["external_url_count"] == 0
    for expected_token in (
        "防火墙",
        "模式 2",
        "模式 1",
        "allow_pose_adjustment",
        "BLOCKED_BY_REMEDIATION_POLICY",
        "real2sim_office_desk_riser_organize",
        "specialty_coffee_bar_creatista_prep",
        "modern_office_brother_calibrate",
        "electronics_qa_ingun_probe_lock",
    ):
        assert expected_token in html_text

    # Every embedded MP4 must decode to a valid ISO BMFF ftyp header.
    encoded_videos = re.findall(r"data:video/mp4;base64,([A-Za-z0-9+/=]+)", html_text)
    assert len(encoded_videos) == 7
    for payload_b64 in encoded_videos:
        raw = base64.b64decode(payload_b64)
        assert raw[4:8] == b"ftyp"


@pytest.mark.local_artifacts
def test_live_showcase_artifacts_present() -> None:
    from build_progress_showcase_html import resolve_default_inputs

    rollouts_root = REPO_ROOT / "external_artifacts" / "rollouts"
    if not rollouts_root.exists():
        pytest.skip("live rollout artifacts not present")

    inputs = resolve_default_inputs(REPO_ROOT)
    assert len(inputs["videos"]) == 7
    for path in inputs["videos"].values():
        assert path.exists(), f"missing video: {path}"
    for path in inputs["reports"].values():
        assert path.exists(), f"missing report: {path}"
