from __future__ import annotations

from pathlib import Path

import pytest

from trajectory_forge.core.contracts import (
    ALLOWED_REMEDIATION_MODES,
    ArticulationDofSpec,
    ContractValidationError,
    FirewallViolationError,
    NamedFrameSpec,
    ObjectContractSpec,
    PoseSpec,
    RemediationPolicySpec,
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
    apply_remediation_policy,
    build_feedback_packet,
    compute_sequential_blocking_stage,
    render_repair_prompt_markdown,
)
from trajectory_forge.solver.harness import PrivilegedSolverHarness, SolveSessionConfig
from trajectory_forge.solver.synthesizer import (
    ScriptedRepairSynthesizer,
    StaticControllerSynthesizer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
R3_CONTROLLERS_DIR = REPO_ROOT / "examples" / "controllers" / "r3"
REAL2SIM_CONTROLLERS_DIR = REPO_ROOT / "examples" / "controllers" / "real2sim"
R3_PACKAGES = Path(
    "/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/"
    "out_of_domain_factory_r3_20261113/packages"
)
REAL2SIM_PACKAGES = Path(
    "/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/"
    "real2sim_office_desk_20261113/scene_gate/packages"
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
                    initial_pose=PoseSpec(xyz=(-0.415, -0.152, 0.757)),
                    named_frames={
                        "grasp": NamedFrameSpec(
                            name="grasp", pose_local=PoseSpec(xyz=(0.0, 0.0, 0.030))
                        )
                    },
                ),
                "calibration_sheet": ObjectContractSpec(
                    object_id="calibration_sheet",
                    runtime_uid="calibration_sheet",
                    role="manipulable",
                    state_prim_path="/World/calibration_sheet",
                    initial_pose=PoseSpec(xyz=(0.415, -0.142, 0.757)),
                    named_frames={
                        "grasp": NamedFrameSpec(
                            name="grasp", pose_local=PoseSpec(xyz=(0.0, -0.054, 0.0125))
                        )
                    },
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
        "real2sim_office_desk_riser_organize": (
            (
                "stage0_right_drawer_opened",
                "stage1_chocopie_in_right_drawer",
                "stage2_pen_in_right_drawer",
                "stage3_right_drawer_closed",
                "stage4_mug_organized_next_to_sprite",
            ),
            {
                "chocopie_pack": ObjectContractSpec(
                    object_id="chocopie_pack",
                    runtime_uid="chocopie_pack",
                    role="manipulable",
                    state_prim_path="/World/chocopie_pack",
                    initial_pose=PoseSpec(xyz=(0.355, -0.015, 0.752)),
                ),
                "blue_ballpoint_pen": ObjectContractSpec(
                    object_id="blue_ballpoint_pen",
                    runtime_uid="blue_ballpoint_pen",
                    role="manipulable",
                    state_prim_path="/World/blue_ballpoint_pen",
                    initial_pose=PoseSpec(xyz=(0.305, -0.100, 0.752)),
                ),
                "glass_water_mug": ObjectContractSpec(
                    object_id="glass_water_mug",
                    runtime_uid="glass_water_mug",
                    role="manipulable",
                    state_prim_path="/World/glass_water_mug",
                    initial_pose=PoseSpec(xyz=(0.438, -0.195, 0.752)),
                ),
                "sprite_bottle_500ml": ObjectContractSpec(
                    object_id="sprite_bottle_500ml",
                    runtime_uid="sprite_bottle_500ml",
                    role="context",
                    state_prim_path="/World/sprite_bottle_500ml",
                    initial_pose=PoseSpec(xyz=(0.578, -0.111, 0.752)),
                ),
                "desktop_riser_rig": ObjectContractSpec(
                    object_id="desktop_riser_rig",
                    runtime_uid="desktop_riser_rig",
                    role="hero_articulated",
                    state_prim_path="/World/desktop_riser_rig",
                    initial_pose=PoseSpec(xyz=(-0.045, 0.125, 0.750)),
                    named_frames={
                        "handle_right": NamedFrameSpec(
                            name="handle_right",
                            pose_local=PoseSpec(xyz=(0.121, -0.122, 0.030)),
                        ),
                        "drawer_right_cavity": NamedFrameSpec(
                            name="drawer_right_cavity",
                            pose_local=PoseSpec(xyz=(0.121, -0.158, 0.0175)),
                        ),
                    },
                    dofs=_make_dofs(
                        (
                            ("drawer_left", "/World/desktop_riser_rig/drawer_left"),
                            ("drawer_right", "/World/desktop_riser_rig/drawer_right"),
                            ("laptop_lid_hinge", "/World/desktop_riser_rig/laptop_lid_hinge"),
                        )
                    ),
                    reset_joint_positions=(0.0, 0.0, 1.7802358),
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
        (
            "real2sim_office_desk_riser_organize",
            0,
            "UPSTREAM_MISSING_HANDLE_COLLIDER",
            "desktop_riser_rig.handle_right lies 5.0mm outside DrawerRight collision AABB",
            "Add positive-scale PhysicsCollisionAPI box colliders for the drawer pull handle.",
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


def test_compute_sequential_blocking_stage_handles_non_monotonic_cyclic_stages() -> None:
    # Stage 0 opened drawer_right, Stage 3 closed drawer_right -> stage_scores[0] flips back to 0.0
    completed, blk = compute_sequential_blocking_stage(
        num_stages=5,
        cur_union_metric_is_none=True,
        long_stride_idx=5,
        final_score=1.0,
    )
    assert completed == 5
    assert blk is None

    completed_mid, blk_mid = compute_sequential_blocking_stage(
        num_stages=5,
        cur_union_metric_is_none=False,
        long_stride_idx=3,
        final_score=0.6,
    )
    assert completed_mid == 3
    assert blk_mid == 3

    req = RolloutJobRequest(
        job_id="cyclic_ok",
        scenario_package_ref="real2sim_office_desk_riser_organize",
        embodiment_id="lift2_r5a_bimanual",
        mode="full_horizon_gate",
        seed=0,
        controller_code="def solve(ctx):\n    ctx.step(1)\n",
    )
    res = RolloutJobResult(
        job_id="cyclic_ok",
        status="succeeded",
        mode="full_horizon_gate",
        final_score=1.0,
        steps_executed=1800,
        snapshots_used=0,
        firewall_verified=True,
        state_summary={
            "stage_scores": [0.0, 1.0, 1.0, 1.0, 1.0],
            "stages_completed_sequentially": 5,
            "blocking_stage": None,
        },
    )
    pkt = build_feedback_packet(attempt=1, request=req, result=res)
    assert pkt.blocking_stage is None


def test_real2sim_controller_offline_preflight_search_and_firewall_gate() -> None:
    embodiment = load_embodiment_card(
        REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_bimanual.yaml"
    )
    brief = _synthetic_r3_brief("real2sim_office_desk_riser_organize")
    code = (REAL2SIM_CONTROLLERS_DIR / "office_desk_riser_organize_privileged.py").read_text(
        encoding="utf-8"
    )
    search_res = run_controller_preflight(
        code,
        brief,
        embodiment,
        mode="privileged_search",
        max_steps=2600,
    )
    assert search_res.status == "succeeded"
    assert search_res.snapshots_used > 0
    assert search_res.firewall_verified is False

    gate_res = run_controller_preflight(
        code,
        brief,
        embodiment,
        mode="full_horizon_gate",
        max_steps=2600,
    )
    assert gate_res.status == "succeeded"
    assert gate_res.snapshots_used == 0
    assert gate_res.firewall_verified is True


@pytest.mark.local_artifacts
def test_real2sim_controller_preflight_against_real_collected_package() -> None:
    pkg_dir = REAL2SIM_PACKAGES / "real2sim_office_desk_riser_organize" / "collected_package"
    if not pkg_dir.exists():
        pytest.skip(f"real2sim collected package not present at {pkg_dir}")
    embodiment = load_embodiment_card(
        REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_bimanual.yaml"
    )
    brief = load_genmanip_collected_package(pkg_dir)
    code = (REAL2SIM_CONTROLLERS_DIR / "office_desk_riser_organize_privileged.py").read_text(
        encoding="utf-8"
    )
    gate_res = run_controller_preflight(
        code,
        brief,
        embodiment,
        mode="full_horizon_gate",
        max_steps=2600,
    )
    assert gate_res.status == "succeeded"
    assert gate_res.snapshots_used == 0
    assert gate_res.firewall_verified is True


def test_remediation_policy_spec_validation_and_roundtrip() -> None:
    assert ALLOWED_REMEDIATION_MODES == frozenset({"physics_tuning", "scene_codesign"})
    strict_policy = RemediationPolicySpec()
    assert strict_policy.mode == "physics_tuning"
    assert strict_policy.allow_pose_adjustment is False
    assert strict_policy.allows_pose_adjustment is False
    assert strict_policy.allows_structural_parts is False

    pose_policy = RemediationPolicySpec(mode="physics_tuning", allow_pose_adjustment=True)
    assert pose_policy.allows_pose_adjustment is True
    assert pose_policy.allows_structural_parts is False

    codesign_policy = RemediationPolicySpec(mode="scene_codesign", allow_pose_adjustment=False)
    assert codesign_policy.allows_pose_adjustment is True
    assert codesign_policy.allows_structural_parts is True

    with pytest.raises(ContractValidationError):
        RemediationPolicySpec(mode="invalid_mode")

    req_legacy = RolloutJobRequest.from_dict(
        {
            "job_id": "legacy",
            "scenario_package_ref": "pkg",
            "embodiment_id": "lift2_r5a_bimanual",
            "mode": "privileged_search",
            "seed": 0,
            "controller_code": "def solve(ctx):\n    ctx.step(1)\n",
        }
    )
    assert req_legacy.remediation_policy == strict_policy

    req_pose = RolloutJobRequest(
        job_id="pose_mode",
        scenario_package_ref="pkg",
        embodiment_id="lift2_r5a_bimanual",
        mode="privileged_search",
        seed=0,
        controller_code="def solve(ctx):\n    ctx.step(1)\n",
        remediation_policy=pose_policy,
    )
    req_rt = RolloutJobRequest.from_dict(req_pose.to_dict())
    assert req_rt.remediation_policy == pose_policy

    diag = SolvabilityDiagnostics(
        failure_code="BLOCKED_BY_REMEDIATION_POLICY",
        stage_index=0,
        detail="blocked",
        upstream_remediation_hint="switch mode",
        remediation_category="affordance_part",
        required_remediation_mode="scene_codesign",
        requires_pose_adjustment=True,
        policy_blocked=True,
        root_failure_code="UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE",
    )
    diag_rt = SolvabilityDiagnostics.from_dict(diag.to_dict())
    assert diag_rt == diag


def test_apply_remediation_policy_modes_and_pose_subswitch() -> None:
    damp_diag = SolvabilityDiagnostics(
        failure_code="UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT",
        stage_index=3,
        detail="damping too high",
        upstream_remediation_hint="Lower angular damping to 0.22 N*m*s/deg.",
    )
    strict = RemediationPolicySpec(mode="physics_tuning", allow_pose_adjustment=False)
    out_damp = apply_remediation_policy(damp_diag, strict)
    assert out_damp.policy_blocked is False
    assert out_damp.remediation_category == "joint_drive"
    assert out_damp.required_remediation_mode == "physics_tuning"
    assert out_damp.failure_code == "UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT"

    proxy_diag = SolvabilityDiagnostics(
        failure_code="UPSTREAM_MISSING_HANDLE_COLLIDER",
        stage_index=0,
        detail="handle collider missing",
        upstream_remediation_hint="Add positive-scale box collider on handle.",
    )
    out_proxy = apply_remediation_policy(proxy_diag, strict)
    assert out_proxy.policy_blocked is False
    assert out_proxy.remediation_category == "collision_proxy"
    assert out_proxy.required_remediation_mode == "physics_tuning"

    aperture_diag = SolvabilityDiagnostics(
        failure_code="UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE",
        stage_index=0,
        detail="a5_media_pack horizontal footprint (0.156 m x 0.116 m, z=0.022 m) exceeds 0.088 m",
        upstream_remediation_hint="Add grasp handle <=0.080 m or stand upright.",
    )
    blocked = apply_remediation_policy(
        aperture_diag, strict, object_dims_xyz_m=(0.156, 0.116, 0.022)
    )
    assert blocked.policy_blocked is True
    assert blocked.failure_code == "BLOCKED_BY_REMEDIATION_POLICY"
    assert blocked.root_failure_code == "UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE"
    assert blocked.required_remediation_mode == "scene_codesign"
    assert blocked.requires_pose_adjustment is True

    pose_ok = apply_remediation_policy(
        aperture_diag,
        RemediationPolicySpec(mode="physics_tuning", allow_pose_adjustment=True),
        object_dims_xyz_m=(0.156, 0.116, 0.022),
    )
    assert pose_ok.policy_blocked is False
    assert pose_ok.failure_code == "UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE"
    assert pose_ok.remediation_category == "pose_adjustment"
    assert pose_ok.required_remediation_mode == "physics_tuning"
    assert pose_ok.requires_pose_adjustment is True
    assert "without modifying visual" in pose_ok.upstream_remediation_hint.lower()

    codesign_ok = apply_remediation_policy(
        aperture_diag,
        RemediationPolicySpec(mode="scene_codesign"),
        object_dims_xyz_m=(0.156, 0.116, 0.022),
    )
    assert codesign_ok.policy_blocked is False
    assert codesign_ok.remediation_category == "affordance_part"
    assert codesign_ok.required_remediation_mode == "scene_codesign"

    unpinchable_pose_blocked = apply_remediation_policy(
        aperture_diag,
        RemediationPolicySpec(mode="physics_tuning", allow_pose_adjustment=True),
        object_dims_xyz_m=(0.120, 0.120, 0.120),
    )
    assert unpinchable_pose_blocked.policy_blocked is True
    assert unpinchable_pose_blocked.failure_code == "BLOCKED_BY_REMEDIATION_POLICY"
    assert unpinchable_pose_blocked.root_failure_code == "UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE"
    assert unpinchable_pose_blocked.required_remediation_mode == "scene_codesign"


def test_solver_harness_early_stops_on_policy_blocked_diagnostic() -> None:
    brief, embodiment = _brief_and_embodiment()
    seen_policies: list[RemediationPolicySpec] = []

    def blocking_executor(req: RolloutJobRequest) -> RolloutJobResult:
        seen_policies.append(req.remediation_policy)
        return RolloutJobResult(
            job_id=req.job_id,
            status="failed",
            mode=req.mode,
            final_score=0.0,
            steps_executed=50,
            snapshots_used=0,
            firewall_verified=False,
            diagnostics=SolvabilityDiagnostics(
                failure_code="BLOCKED_BY_REMEDIATION_POLICY",
                stage_index=0,
                detail="blocked by strict physics_tuning policy",
                upstream_remediation_hint="Re-run with scene_codesign or allow_pose_adjustment.",
                remediation_category="affordance_part",
                required_remediation_mode="scene_codesign",
                requires_pose_adjustment=True,
                policy_blocked=True,
                root_failure_code="UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE",
            ),
        )

    harness = PrivilegedSolverHarness(executor=blocking_executor, preflight_enabled=False)
    policy = RemediationPolicySpec(mode="physics_tuning", allow_pose_adjustment=False)
    report = harness.solve(
        brief=brief,
        embodiment=embodiment,
        synthesizer=StaticControllerSynthesizer("def solve(ctx):\n    ctx.step(1)\n"),
        config=SolveSessionConfig(max_attempts=4, remediation_policy=policy),
    )
    assert report.succeeded is False
    assert report.attempts_used == 1
    assert len(report.history) == 1
    assert report.history[0].policy_blocked is True
    assert report.history[0].failure_code == "BLOCKED_BY_REMEDIATION_POLICY"
    assert report.history[0].root_failure_code == "UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE"
    assert seen_policies == [policy]
    md = render_repair_prompt_markdown(brief, embodiment, report.history[0])
    assert "BLOCKED_BY_REMEDIATION_POLICY" in md
    assert "physics_tuning" in md
