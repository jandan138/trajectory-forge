from __future__ import annotations

from pathlib import Path

import pytest

from trajectory_forge.core.contracts import (
    ArticulationDofSpec,
    FirewallViolationError,
    ObjectContractSpec,
    PoseSpec,
    RobotSpawnSpec,
    RolloutJobRequest,
    RolloutJobResult,
    ScenarioIntakeBrief,
    SnapshotHandle,
    SolvabilityDiagnostics,
    StageContractSpec,
    StageProgressEntry,
    TrajectoryEpisodeManifest,
)
from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.intake.genmanip_intake import load_genmanip_collected_package
from trajectory_forge.sdk.context import run_controller_preflight
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
R3_CONTROLLERS_DIR = REPO_ROOT / "examples" / "controllers" / "r3"
R3_PACKAGES = Path(
    "/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/"
    "out_of_domain_factory_r3_20261113/packages"
)


def _make_dofs(specs: tuple[tuple[str, str], ...]) -> dict[str, ArticulationDofSpec]:
    return {
        name: ArticulationDofSpec(semantic_joint=name, dof_index=idx, joint_prim=prim)
        for idx, (name, prim) in enumerate(specs)
    }


def _synthetic_r3_brief(package_id: str) -> ScenarioIntakeBrief:
    specs: dict[str, tuple[tuple[str, ...], dict[str, ObjectContractSpec]]] = {
        "electronics_qa_ingun_probe_lock": (
            (
                "stage0_pcba_aligned",
                "stage1_hood_closed",
                "stage2_lever_locked",
                "stage3_probe_enabled",
                "stage4_test_started",
            ),
            {
                "pcba_carrier_board": ObjectContractSpec(
                    object_id="pcba_carrier_board",
                    runtime_uid="pcba_carrier_board",
                    role="manipulable",
                    state_prim_path="/World/pcba_carrier_board",
                    initial_pose=PoseSpec(xyz=(-0.41, -0.15, 0.78)),
                ),
                "ingun_fixture": ObjectContractSpec(
                    object_id="ingun_fixture",
                    runtime_uid="ingun_fixture",
                    role="hero_articulated",
                    state_prim_path="/World/ingun_fixture",
                    initial_pose=PoseSpec(xyz=(0.0, 0.025, 0.755)),
                    dofs=_make_dofs(
                        (
                            ("pressure_hood", "/World/ingun_fixture/PressureHoodJoint"),
                            ("clamp_lever", "/World/ingun_fixture/ClampLeverJoint"),
                            ("mode_toggle", "/World/ingun_fixture/ModeToggleJoint"),
                            ("test_start_button", "/World/ingun_fixture/TestStartButtonJoint"),
                        )
                    ),
                    reset_joint_positions=(-1.0472, -1.39626, 0.34907, 0.0),
                ),
            },
        ),
        "specialty_coffee_bar_creatista_prep": (
            (
                "stage0_tray_extended",
                "stage1_cup_seated",
                "stage2_capsule_inserted",
                "stage3_lever_locked",
                "stage4_dial_selected",
                "stage5_dial_pressed",
            ),
            {
                "espresso_cup": ObjectContractSpec(
                    object_id="espresso_cup",
                    runtime_uid="espresso_cup",
                    role="manipulable",
                    state_prim_path="/World/espresso_cup",
                    initial_pose=PoseSpec(xyz=(-0.22, -0.18, 0.755)),
                ),
                "coffee_capsule": ObjectContractSpec(
                    object_id="coffee_capsule",
                    runtime_uid="coffee_capsule",
                    role="manipulable",
                    state_prim_path="/World/coffee_capsule",
                    initial_pose=PoseSpec(xyz=(0.24, -0.18, 0.755)),
                ),
                "creatista_machine": ObjectContractSpec(
                    object_id="creatista_machine",
                    runtime_uid="creatista_machine",
                    role="hero_articulated",
                    state_prim_path="/World/creatista_machine",
                    initial_pose=PoseSpec(xyz=(0.0, -0.02, 0.755)),
                    dofs=_make_dofs(
                        (
                            ("drip_tray", "/World/creatista_machine/DripTrayJoint"),
                            ("brew_lever", "/World/creatista_machine/BrewLeverJoint"),
                            ("select_dial", "/World/creatista_machine/SelectDialJoint"),
                            ("dial_press", "/World/creatista_machine/DialPressJoint"),
                        )
                    ),
                    reset_joint_positions=(0.0, -1.22173, 0.0, 0.0),
                ),
            },
        ),
        "modern_office_brother_calibrate": (
            (
                "stage0_media_loaded",
                "stage1_cassette_closed",
                "stage2_sheet_aligned",
                "stage3_lid_closed",
                "stage4_copy_started",
            ),
            {
                "a5_media_pack": ObjectContractSpec(
                    object_id="a5_media_pack",
                    runtime_uid="a5_media_pack",
                    role="manipulable",
                    state_prim_path="/World/a5_media_pack",
                    initial_pose=PoseSpec(xyz=(-0.415, -0.12, 0.755)),
                ),
                "calibration_sheet": ObjectContractSpec(
                    object_id="calibration_sheet",
                    runtime_uid="calibration_sheet",
                    role="manipulable",
                    state_prim_path="/World/calibration_sheet",
                    initial_pose=PoseSpec(xyz=(0.415, -0.10, 0.755)),
                ),
                "brother_printer": ObjectContractSpec(
                    object_id="brother_printer",
                    runtime_uid="brother_printer",
                    role="hero_articulated",
                    state_prim_path="/World/brother_printer",
                    initial_pose=PoseSpec(xyz=(0.0, 0.0, 0.755)),
                    dofs=_make_dofs(
                        (
                            ("paper_cassette", "/World/brother_printer/PaperCassetteJoint"),
                            ("scanner_lid", "/World/brother_printer/ScannerLidJoint"),
                            ("button_start", "/World/brother_printer/ButtonStartJoint"),
                        )
                    ),
                    reset_joint_positions=(-0.165, -1.0821, 0.0),
                ),
            },
        ),
    }
    stage_names, objects = specs[package_id]
    return ScenarioIntakeBrief(
        package_id=package_id,
        episode_name="0",
        instruction=f"Offline synthetic brief for {package_id}",
        scene_usd_relpath="scene.usda",
        robot_profile="manip/lift2/R5a",
        robot_spawn=RobotSpawnSpec(
            robot_id="lift2_r5a_bimanual",
            xyz=(0.0, -1.02, 0.31),
            wxyz=(0.7071067812, 0.0, 0.0, 0.7071067812),
            joint_positions=(0.0,) * 16,
        ),
        objects=objects,
        stages=tuple(
            StageContractSpec(stage_index=i, metric_id=name, clauses=())
            for i, name in enumerate(stage_names)
        ),
    )


def _brief_and_embodiment():
    brief = _synthetic_r3_brief("modern_office_brother_calibrate")
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


_R3_CONTROLLER_MAP = (
    ("electronics_qa_ingun_probe_lock", "ingun_probe_lock_privileged.py"),
    ("specialty_coffee_bar_creatista_prep", "creatista_prep_privileged.py"),
    ("modern_office_brother_calibrate", "brother_calibrate_privileged.py"),
)


def test_r3_controllers_offline_preflight_search_and_firewall_gate() -> None:
    embodiment = load_embodiment_card(
        REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_bimanual.yaml"
    )
    for package_id, script_name in _R3_CONTROLLER_MAP:
        brief = _synthetic_r3_brief(package_id)
        code = (R3_CONTROLLERS_DIR / script_name).read_text(encoding="utf-8")
        search_res = run_controller_preflight(
            code,
            brief,
            embodiment,
            mode="privileged_search",
            max_steps=2500,
        )
        assert search_res.status == "succeeded"
        assert search_res.snapshots_used > 0
        assert search_res.firewall_verified is False

        gate_res = run_controller_preflight(
            code,
            brief,
            embodiment,
            mode="full_horizon_gate",
            max_steps=2500,
        )
        assert gate_res.status == "succeeded"
        assert gate_res.snapshots_used == 0
        assert gate_res.firewall_verified is True
        assert gate_res.trajectory_manifest is not None
        assert gate_res.trajectory_manifest.continuous_from_t0 is True
        assert gate_res.trajectory_manifest.privileged_channels_recorded is False


@pytest.mark.local_artifacts
def test_r3_controllers_preflight_against_real_collected_packages() -> None:
    if not R3_PACKAGES.exists():
        pytest.skip(f"r3 collected packages not present at {R3_PACKAGES}")
    embodiment = load_embodiment_card(
        REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_bimanual.yaml"
    )
    for package_id, script_name in _R3_CONTROLLER_MAP:
        brief = load_genmanip_collected_package(
            R3_PACKAGES / package_id / "collected_package"
        )
        code = (R3_CONTROLLERS_DIR / script_name).read_text(encoding="utf-8")
        search_res = run_controller_preflight(
            code,
            brief,
            embodiment,
            mode="privileged_search",
            max_steps=2500,
        )
        assert search_res.status == "succeeded"
        assert search_res.snapshots_used > 0

        gate_res = run_controller_preflight(
            code,
            brief,
            embodiment,
            mode="full_horizon_gate",
            max_steps=2500,
        )
        assert gate_res.status == "succeeded"
        assert gate_res.snapshots_used == 0
        assert gate_res.firewall_verified is True
        assert gate_res.trajectory_manifest is not None
        assert gate_res.trajectory_manifest.privileged_channels_recorded is False


def test_firewall_forbids_snapshots_and_resume_in_gate_and_synthesis() -> None:
    brief, embodiment = _brief_and_embodiment()
    save_code = "def solve(ctx):\n    ctx.checkpoint.save_snapshot('leak')\n"
    restore_code = (
        "from trajectory_forge.core.contracts import SnapshotHandle\n"
        "def solve(ctx):\n"
        "    h = ctx.checkpoint.save_snapshot('ok')\n"
        "    ctx.checkpoint.restore_snapshot(h)\n"
    )
    restore_direct_code = (
        "def solve(ctx):\n"
        "    ctx.checkpoint.restore_snapshot('snap_1')\n"
    )
    for mode in ("full_horizon_gate", "trajectory_synthesis"):
        with pytest.raises(FirewallViolationError):
            run_controller_preflight(save_code, brief, embodiment, mode=mode)
        with pytest.raises(FirewallViolationError):
            run_controller_preflight(restore_direct_code, brief, embodiment, mode=mode)
        with pytest.raises(FirewallViolationError):
            RolloutJobRequest(
                job_id=f"resume_{mode}",
                scenario_package_ref=brief.package_id,
                embodiment_id=embodiment.embodiment_id,
                mode=mode,
                seed=0,
                controller_code="def solve(ctx):\n    ctx.step(1)\n",
                resume_from_snapshot_id="snap_1",
            )
    _ = restore_code


def test_upstream_solvability_diagnostics_propagate_to_solve_report() -> None:
    embodiment = load_embodiment_card(
        REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_bimanual.yaml"
    )
    cases = (
        (
            "specialty_coffee_bar_creatista_prep",
            3,
            "UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT",
            "creatista_machine.brew_lever damping=12.00 N*m*s/deg",
            "Divide SI N*m*s/rad by 57.2958 in upstream asset USD.",
        ),
        (
            "modern_office_brother_calibrate",
            0,
            "UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE",
            "a5_media_pack 0.148m x 0.210m exceeds 0.088m R5a jaw aperture",
            "Add recessed grip cutout or reduce thickness <= 0.085m.",
        ),
    )
    for package_id, blocking_stage, failure_code, detail, hint in cases:
        brief = _synthetic_r3_brief(package_id)

        def fake_exec(req: RolloutJobRequest) -> RolloutJobResult:
            return RolloutJobResult(
                job_id=req.job_id,
                status="failed",
                mode=req.mode,
                final_score=0.5 if blocking_stage == 3 else 0.0,
                steps_executed=400,
                snapshots_used=1,
                firewall_verified=False,
                diagnostics=SolvabilityDiagnostics(
                    failure_code=failure_code,
                    stage_index=blocking_stage,
                    detail=detail,
                    upstream_remediation_hint=hint,
                ),
                state_summary={"blocking_stage": blocking_stage},
            )

        harness = PrivilegedSolverHarness(executor=fake_exec, preflight_enabled=True)
        report = harness.solve(
            brief=brief,
            embodiment=embodiment,
            synthesizer=StaticControllerSynthesizer("def solve(ctx):\n    ctx.step(5)\n"),
            config=SolveSessionConfig(max_attempts=1, synthesize_trajectory=True),
        )
        assert report.succeeded is False
        assert len(report.history) == 1
        entry = report.history[0]
        assert entry.failure_code == failure_code
        assert entry.blocking_stage == blocking_stage
        assert entry.upstream_remediation_hint == hint

        pkt = build_feedback_packet(
            attempt=1,
            request=RolloutJobRequest(
                job_id="diag_check",
                scenario_package_ref=package_id,
                embodiment_id=embodiment.embodiment_id,
                mode="privileged_search",
                seed=0,
                controller_code="def solve(ctx):\n    ctx.step(5)\n",
            ),
            result=fake_exec(
                RolloutJobRequest(
                    job_id="diag_check",
                    scenario_package_ref=package_id,
                    embodiment_id=embodiment.embodiment_id,
                    mode="privileged_search",
                    seed=0,
                    controller_code="def solve(ctx):\n    ctx.step(5)\n",
                )
            ),
        )
        md = render_repair_prompt_markdown(brief, embodiment, pkt)
        assert failure_code in md
        assert hint in md
