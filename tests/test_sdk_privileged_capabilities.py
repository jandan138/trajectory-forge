from __future__ import annotations

from pathlib import Path
import math

import pytest

from trajectory_forge.core.contracts import (
    FirewallViolationError,
    RolloutJobRequest,
    RolloutJobResult,
    SnapshotHandle,
)
from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.intake.genmanip_intake import load_genmanip_collected_package
from trajectory_forge.sdk.context import run_controller_preflight
from trajectory_forge.sdk.math3d import Pose3D, compose_pose, invert_pose


REPO_ROOT = Path(__file__).resolve().parents[1]
R3_PACKAGES = Path(
    "/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/"
    "out_of_domain_factory_r3_20261113/packages"
)


def test_pose3d_from_approach_orthonormal_and_composition() -> None:
    pose = Pose3D.from_approach(
        xyz=(0.1, -0.2, 0.8),
        approach_dir=(0.0, 0.0, -1.0),
        closing_dir=(1.0, 0.0, 0.0),
    )
    local = Pose3D(xyz=(0.0, 0.0, 0.1))
    world_tip = compose_pose(pose, local)
    assert world_tip.xyz == pytest.approx((0.1, -0.2, 0.7), abs=1e-6)
    rt = compose_pose(pose, invert_pose(pose))
    assert rt.xyz == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert abs(rt.wxyz[0]) == pytest.approx(1.0, abs=1e-6)


def test_contracts_roundtrip_new_privileged_fields() -> None:
    req = RolloutJobRequest(
        job_id="j1",
        scenario_package_ref="modern_office_brother_calibrate",
        embodiment_id="lift2_r5a_bimanual",
        mode="privileged_search",
        seed=0,
        controller_code="def solve(ctx):\n    ctx.step(1)\n",
        render_media=True,
    )
    req2 = RolloutJobRequest.from_dict(req.to_dict())
    assert req2.render_media is True

    res = RolloutJobResult(
        job_id="j1",
        status="succeeded",
        mode="privileged_search",
        final_score=1.0,
        steps_executed=12,
        snapshots_used=1,
        firewall_verified=False,
        saved_snapshots=(SnapshotHandle(snapshot_id="snap_1", label="s0", step_index=5),),
        state_summary={"end_of_episode_all_stages_hold": True},
    )
    res2 = RolloutJobResult.from_dict(res.to_dict())
    assert len(res2.saved_snapshots) == 1
    assert res2.saved_snapshots[0].snapshot_id == "snap_1"
    assert res2.state_summary["end_of_episode_all_stages_hold"] is True


@pytest.mark.local_artifacts
def test_intake_preserves_initial_layout_orientation_and_part_ids() -> None:
    pkg = R3_PACKAGES / "electronics_qa_ingun_probe_lock" / "collected_package"
    brief = load_genmanip_collected_package(pkg)
    pcba = brief.objects["pcba_carrier_board"]
    assert pcba.initial_pose.wxyz[0] == pytest.approx(math.sqrt(0.5), abs=1e-4)
    assert abs(pcba.initial_pose.wxyz[3]) == pytest.approx(math.sqrt(0.5), abs=1e-4)
    assert "ingun_fixture_pressure_hood" in brief.part_ids


def test_sdk_expanded_apis_and_save_if_search_firewall() -> None:
    embodiment = load_embodiment_card(
        REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_bimanual.yaml"
    )
    pkg = R3_PACKAGES / "modern_office_brother_calibrate" / "collected_package"
    brief = load_genmanip_collected_package(pkg)

    code = """
from trajectory_forge.sdk.math3d import Pose3D

def solve(ctx):
    assert ctx.mode in ("privileged_search", "full_horizon_gate", "trajectory_synthesis")
    assert "brother_printer" in ctx.world.object_ids()
    _ = ctx.world.object_aabb("brother_printer")
    jval = ctx.world.joint_value("brother_printer", "paper_cassette")
    assert isinstance(jval, float)
    _ = ctx.world.joint_positions("brother_printer")
    _ = ctx.world.joint_info("brother_printer", "paper_cassette")
    scores = ctx.world.stage_scores()
    assert len(scores) == 5
    assert isinstance(ctx.world.current_score(), float)
    assert ctx.world.is_stage_satisfied(0) is False
    ee = ctx.kinematics.ee_pose("left_arm")
    assert isinstance(ee, Pose3D)
    target = Pose3D.from_approach(
        xyz=(0.0, -0.3, 0.9),
        approach_dir=(0.0, 0.0, -1.0),
        closing_dir=(0.0, 1.0, 0.0),
    )
    ctx.kinematics.move_to_pose("left_arm", target, steps=4)
    ctx.end_effector.set_posture("left_gripper", "closed", steps=2)
    ctx.kinematics.move_base_world(0.01, 0.02, 0.0, steps=2)
    ctx.kinematics.set_torso_lift(0.46, steps=2)
    assert isinstance(ctx.kinematics.torso_lift(), float)
    assert len(ctx.sensors.gripper_widths("left_gripper")) == 2
    snap = ctx.checkpoint.save_if_search("stage0_done")
    if ctx.mode == "privileged_search":
        assert snap is not None
    else:
        assert snap is None
    print("controller_ok", ctx.mode)
    ctx.step(2)
"""
    res_search = run_controller_preflight(code, brief, embodiment, mode="privileged_search")
    assert res_search.snapshots_used == 1
    assert len(res_search.saved_snapshots) == 1
    assert "controller_ok privileged_search" in res_search.state_summary.get("controller_logs", [])

    res_gate = run_controller_preflight(code, brief, embodiment, mode="full_horizon_gate")
    assert res_gate.snapshots_used == 0
    assert res_gate.firewall_verified is True
    assert res_gate.saved_snapshots == ()

    bad_code = """
def solve(ctx):
    ctx.checkpoint.save_snapshot("illegal")
"""
    with pytest.raises(FirewallViolationError):
        run_controller_preflight(bad_code, brief, embodiment, mode="full_horizon_gate")


def test_fluid_containment_ratio_query_and_pure_particle_helpers() -> None:
    from trajectory_forge.trajectory.fluid_helpers import (
        compute_aabb_containment_ratio,
        transform_points_local_to_world,
    )

    embodiment = load_embodiment_card(
        REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_fixed_base_16dof.yaml"
    )
    pkg = R3_PACKAGES / "modern_office_brother_calibrate" / "collected_package"
    brief = load_genmanip_collected_package(pkg)

    code = """
def solve(ctx):
    ratio = ctx.world.fluid_containment_ratio("missing_tank", fallback=0.0)
    assert isinstance(ratio, float)
    assert ratio == 0.0
    ctx.step(1)
"""
    res = run_controller_preflight(code, brief, embodiment, mode="full_horizon_gate")
    assert res.status == "succeeded"

    pts_local = ((0.0, 0.0, 0.0), (0.01, 0.0, 0.02), (0.50, 0.50, 0.50))
    world_pose = Pose3D(xyz=(1.0, -0.5, 0.8), wxyz=(1.0, 0.0, 0.0, 0.0))
    pts_world = transform_points_local_to_world(pts_local, world_pose)
    assert pts_world[0] == pytest.approx((1.0, -0.5, 0.8), abs=1e-6)
    inside = compute_aabb_containment_ratio(
        pts_world,
        box_min=(0.9, -0.6, 0.7),
        box_max=(1.1, -0.4, 0.9),
    )
    assert inside == pytest.approx(2.0 / 3.0, abs=1e-6)

