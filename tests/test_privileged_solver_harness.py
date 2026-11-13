from __future__ import annotations

from pathlib import Path

from trajectory_forge.core.contracts import (
    RolloutJobRequest,
    RolloutJobResult,
    SnapshotHandle,
    SolvabilityDiagnostics,
    StageProgressEntry,
    TrajectoryEpisodeManifest,
)
from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.intake.genmanip_intake import load_genmanip_collected_package
from trajectory_forge.solver.diagnostics import (
    AttemptFeedbackPacket,
    build_feedback_packet,
    render_repair_prompt_markdown,
)
from trajectory_forge.solver.harness import PrivilegedSolverHarness, SolveSessionConfig
from trajectory_forge.solver.synthesizer import (
    ScriptedRepairSynthesizer,
    StaticControllerSynthesizer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
R3_PACKAGES = Path(
    "/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/"
    "out_of_domain_factory_r3_20261113/packages"
)


def _brief_and_embodiment():
    brief = load_genmanip_collected_package(
        R3_PACKAGES / "modern_office_brother_calibrate" / "collected_package"
    )
    embodiment = load_embodiment_card(
        REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_bimanual.yaml"
    )
    return brief, embodiment


def test_harness_iterates_repair_and_verifies_firewall_gate() -> None:
    brief, embodiment = _brief_and_embodiment()
    seen_requests: list[RolloutJobRequest] = []

    def fake_executor(req: RolloutJobRequest) -> RolloutJobResult:
        seen_requests.append(req)
        if "attempt_1" in req.controller_code:
            return RolloutJobResult(
                job_id=req.job_id,
                status="failed",
                mode=req.mode,
                final_score=0.2,
                steps_executed=120,
                snapshots_used=1,
                firewall_verified=False,
                stage_progression=(
                    StageProgressEntry(
                        stage_index=0,
                        metric_id="stage0_media_loaded",
                        score=0.2,
                        step_index=60,
                    ),
                ),
                diagnostics=SolvabilityDiagnostics(
                    failure_code="STAGE_INCOMPLETE",
                    stage_index=1,
                    detail="paper_cassette joint stuck at -0.045m, target [-0.012, 0.0]",
                ),
                saved_snapshots=(
                    SnapshotHandle(snapshot_id="snap_1", label="after_stage0", step_index=60),
                ),
                state_summary={
                    "stage_scores": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "blocking_stage": 1,
                },
            )

        manifest = None
        if req.mode == "trajectory_synthesis":
            manifest = TrajectoryEpisodeManifest(
                episode_id=f"{brief.package_id}_{req.job_id}",
                scenario_id=brief.package_id,
                embodiment_id=embodiment.embodiment_id,
                num_steps=300,
                continuous_from_t0=True,
                privileged_channels_recorded=False,
                recorded_channels=("proprio", "rgb_head"),
                artifact_relpath="episodes/ep.npz",
            )
        return RolloutJobResult(
            job_id=req.job_id,
            status="succeeded",
            mode=req.mode,
            final_score=1.0,
            steps_executed=300,
            snapshots_used=1 if req.mode == "privileged_search" else 0,
            firewall_verified=req.mode in {"full_horizon_gate", "trajectory_synthesis"},
            stage_progression=tuple(
                StageProgressEntry(
                    stage_index=i,
                    metric_id=f"stage_{i}",
                    score=(i + 1) / 5.0,
                    step_index=60 * (i + 1),
                )
                for i in range(5)
            ),
            trajectory_manifest=manifest,
            state_summary={"stage_scores": [1.0, 1.0, 1.0, 1.0, 1.0]},
        )

    attempt1_code = "def solve(ctx):\n    # attempt_1\n    ctx.checkpoint.save_if_search('s0')\n    ctx.step(10)\n"
    attempt2_code = "def solve(ctx):\n    # attempt_2\n    ctx.checkpoint.save_if_search('s0')\n    ctx.step(20)\n"
    feedbacks_seen: list[AttemptFeedbackPacket] = []
    synth = ScriptedRepairSynthesizer(
        scripts=[attempt1_code, attempt2_code],
        on_feedback=feedbacks_seen.append,
    )
    harness = PrivilegedSolverHarness(executor=fake_executor, preflight_enabled=True)
    report = harness.solve(
        brief=brief,
        embodiment=embodiment,
        synthesizer=synth,
        config=SolveSessionConfig(max_attempts=3, synthesize_trajectory=True),
    )
    assert report.succeeded is True
    assert report.attempts_used == 2
    assert len(feedbacks_seen) == 1
    assert feedbacks_seen[0].blocking_stage == 1
    assert feedbacks_seen[0].saved_snapshots[0].snapshot_id == "snap_1"
    modes = [r.mode for r in seen_requests]
    assert modes == [
        "privileged_search",
        "privileged_search",
        "full_horizon_gate",
        "trajectory_synthesis",
    ]
    assert report.gate_result is not None and report.gate_result.firewall_verified is True
    assert report.synthesis_result is not None
    assert report.synthesis_result.trajectory_manifest is not None


def test_harness_rejects_snapshot_leak_at_full_horizon_gate() -> None:
    brief, embodiment = _brief_and_embodiment()
    seen_modes: list[str] = []

    def fake_executor(req: RolloutJobRequest) -> RolloutJobResult:
        seen_modes.append(req.mode)
        if req.mode == "privileged_search":
            return RolloutJobResult(
                job_id=req.job_id,
                status="succeeded",
                mode=req.mode,
                final_score=1.0,
                steps_executed=50,
                snapshots_used=1,
                firewall_verified=False,
            )
        return RolloutJobResult(
            job_id=req.job_id,
            status="rejected",
            mode=req.mode,
            final_score=0.0,
            steps_executed=10,
            snapshots_used=1,
            firewall_verified=False,
            diagnostics=SolvabilityDiagnostics(
                failure_code="FirewallViolationError",
                detail="restore_snapshot is forbidden in 'full_horizon_gate' mode",
            ),
        )

    bad_cheat_code = (
        "def solve(ctx):\n"
        "    if ctx.mode == 'full_horizon_gate':\n"
        "        ctx.checkpoint.save_snapshot('illegal')\n"
        "    ctx.step(5)\n"
    )
    harness = PrivilegedSolverHarness(executor=fake_executor, preflight_enabled=True)
    report = harness.solve(
        brief=brief,
        embodiment=embodiment,
        synthesizer=StaticControllerSynthesizer(bad_cheat_code),
        config=SolveSessionConfig(max_attempts=1, synthesize_trajectory=True),
    )
    assert report.succeeded is False
    assert seen_modes == ["privileged_search"]  # preflight caught FirewallViolationError before remote gate
    assert "FirewallViolationError" in report.history[0].failure_code

    # Also verify that if preflight passes (or is skipped) and remote gate rejects, synthesis never runs.
    seen_modes.clear()
    harness_no_preflight = PrivilegedSolverHarness(executor=fake_executor, preflight_enabled=False)
    report2 = harness_no_preflight.solve(
        brief=brief,
        embodiment=embodiment,
        synthesizer=StaticControllerSynthesizer("def solve(ctx):\n    ctx.step(5)\n"),
        config=SolveSessionConfig(max_attempts=1, synthesize_trajectory=True),
    )
    assert report2.succeeded is False
    assert seen_modes == ["privileged_search", "full_horizon_gate"]

    # Test feedback prompt rendering helper
    pkt = build_feedback_packet(
        attempt=1,
        request=RolloutJobRequest(
            job_id="j",
            scenario_package_ref="modern_office_brother_calibrate",
            embodiment_id=embodiment.embodiment_id,
            mode="full_horizon_gate",
            seed=0,
            controller_code=bad_cheat_code,
        ),
        result=report.gate_result or RolloutJobResult(
            job_id="j",
            status="rejected",
            mode="full_horizon_gate",
            final_score=0.0,
            steps_executed=0,
            snapshots_used=0,
            firewall_verified=False,
            diagnostics=SolvabilityDiagnostics(failure_code="FirewallViolationError", detail="leak"),
        ),
    )
    md = render_repair_prompt_markdown(brief, embodiment, pkt)
    assert "FirewallViolationError" in md
