from __future__ import annotations

import math
from pathlib import Path

import pytest

from trajectory_forge.core.contracts import (
    ArticulationDofSpec,
    ControllerSafetyError,
    FirewallViolationError,
    NamedFrameSpec,
    ObjectContractSpec,
    PoseSpec,
    RobotSpawnSpec,
    ScenarioIntakeBrief,
    StageContractSpec,
)
from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.sdk.context import run_controller_preflight
from trajectory_forge.sdk.math3d import (
    Pose3D,
    pose_from_named_frame,
    sample_prismatic_sweep_poses,
    sample_revolute_arc_poses,
)
from trajectory_forge.trajectory.manifest import build_trajectory_episode_manifest
from trajectory_forge.worker.safety import validate_controller_source


REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_brief() -> ScenarioIntakeBrief:
    return ScenarioIntakeBrief(
        package_id="synthetic_coffee_test",
        episode_name="101",
        instruction="Test revolute lever and tactile push.",
        scene_usd_relpath="assets/scene_usds/synthetic/scene.usda",
        robot_profile="manip/lift2/R5a",
        robot_spawn=RobotSpawnSpec(
            robot_id="lift2",
            xyz=(0.0, -1.02, 0.31),
            wxyz=(0.7071067812, 0.0, 0.0, 0.7071067812),
            joint_positions=(0.0,) * 16,
        ),
        objects={
            "machine": ObjectContractSpec(
                object_id="machine",
                runtime_uid="machine",
                role="articulated_appliance",
                state_prim_path="/World/test/obj_machine",
                initial_pose=PoseSpec(xyz=(0.0, 0.0, 0.755), wxyz=(1.0, 0.0, 0.0, 0.0)),
                named_frames={
                    "capsule_chute": NamedFrameSpec(
                        name="capsule_chute",
                        pose_local=PoseSpec(xyz=(0.0, -0.105, 0.252), wxyz=(1.0, 0.0, 0.0, 0.0)),
                    )
                },
                dofs={
                    "brew_lever": ArticulationDofSpec(
                        semantic_joint="brew_lever",
                        dof_index=0,
                        joint_prim="/World/Machine/Joints/BrewLeverJoint",
                        states={"open_load": (-1.309, -0.960), "closed_locked": (-0.087, 0.0)},
                    )
                },
            )
        },
        stages=(
            StageContractSpec(stage_index=0, metric_id="stage0_tray", clauses=({"type": "rel"},)),
            StageContractSpec(stage_index=1, metric_id="stage1_lever", clauses=({"type": "rel"},)),
        ),
    )


def test_pose_from_named_frame_and_arc_sweep() -> None:
    brief = _synthetic_brief()
    chute_world = pose_from_named_frame(brief.objects["machine"], "capsule_chute")
    assert chute_world.xyz == pytest.approx((0.0, -0.105, 1.007), abs=1e-6)

    start_pose = Pose3D(xyz=(0.0, -0.15, 1.05), wxyz=(1.0, 0.0, 0.0, 0.0))
    pivot = (0.0, 0.0, 1.05)
    axis = (1.0, 0.0, 0.0)
    arc = sample_revolute_arc_poses(
        start_pose=start_pose,
        hinge_pivot_world=pivot,
        hinge_axis_world=axis,
        delta_angle_rad=math.pi / 2.0,
        num_waypoints=5,
    )
    assert len(arc) == 5
    # Radius from pivot must stay constant (0.15 m) at every arc waypoint
    for wp in arc:
        dist = math.sqrt(
            (wp.xyz[0] - pivot[0]) ** 2
            + (wp.xyz[1] - pivot[1]) ** 2
            + (wp.xyz[2] - pivot[2]) ** 2
        )
        assert dist == pytest.approx(0.15, abs=1e-6)
    assert arc[-1].xyz == pytest.approx((0.0, 0.0, 0.90), abs=1e-5)

    sweep = sample_prismatic_sweep_poses(
        start_pose=start_pose,
        direction_world=(0.0, -1.0, 0.0),
        distance_m=0.18,
        num_waypoints=4,
    )
    assert len(sweep) == 4
    assert sweep[-1].xyz == pytest.approx((0.0, -0.33, 1.05), abs=1e-6)
    assert sweep[-1].wxyz == pytest.approx(start_pose.wxyz, abs=1e-6)


def test_controller_ast_safety_validator() -> None:
    with pytest.raises(ControllerSafetyError):
        validate_controller_source("import subprocess\ndef solve(ctx):\n    pass\n")
    with pytest.raises(ControllerSafetyError):
        validate_controller_source("def wrong_entry(ctx):\n    pass\n")
    validate_controller_source("import math\ndef solve(ctx):\n    return None\n")


def test_preflight_search_vs_firewall_modes() -> None:
    brief = _synthetic_brief()
    card = load_embodiment_card(REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_bimanual.yaml")

    search_code = """
import math

def solve(ctx):
    target = ctx.world.named_frame("machine", "capsule_chute")
    arc = ctx.kinematics.plan_revolute_arc(
        chain="right_arm",
        start_pose=target,
        hinge_pivot_world=(0.0, 0.0, 1.007),
        hinge_axis_world=(1.0, 0.0, 0.0),
        delta_angle_rad=math.pi / 4.0,
        num_waypoints=4,
    )
    ctx.end_effector.set_posture("right_gripper", "closed")
    snap = ctx.checkpoint.save_snapshot("stage0_done")
    ctx.checkpoint.restore_snapshot(snap)
    for wp in arc:
        ctx.kinematics.solve_ik("right_arm", wp)
        tactile = ctx.sensors.tactile("right_gripper")
        if tactile.slip_detected:
            break
        ctx.step(1)
"""
    res = run_controller_preflight(search_code, brief, card, mode="privileged_search")
    assert res.status == "succeeded"
    assert res.snapshots_used >= 1

    with pytest.raises(FirewallViolationError):
        run_controller_preflight(search_code, brief, card, mode="full_horizon_gate")

    with pytest.raises(FirewallViolationError):
        run_controller_preflight(search_code, brief, card, mode="trajectory_synthesis")


def test_trajectory_manifest_enforces_sensorimotor_firewall() -> None:
    with pytest.raises(FirewallViolationError):
        build_trajectory_episode_manifest(
            episode_id="ep_001",
            scenario_id="specialty_coffee_bar_creatista_prep",
            embodiment_id="lift2_r5a_bimanual",
            num_steps=100,
            continuous_from_t0=False,
            privileged_channels_recorded=False,
            recorded_channels=("proprio", "rgb_head"),
        )

    with pytest.raises(FirewallViolationError):
        build_trajectory_episode_manifest(
            episode_id="ep_002",
            scenario_id="specialty_coffee_bar_creatista_prep",
            embodiment_id="lift2_r5a_bimanual",
            num_steps=100,
            continuous_from_t0=True,
            privileged_channels_recorded=True,
            recorded_channels=("proprio", "privileged_object_poses"),
        )

    manifest = build_trajectory_episode_manifest(
        episode_id="ep_003",
        scenario_id="specialty_coffee_bar_creatista_prep",
        embodiment_id="lift2_r5a_bimanual",
        num_steps=120,
        continuous_from_t0=True,
        privileged_channels_recorded=False,
        recorded_channels=("proprio", "rgb_head", "rgb_left_wrist"),
    )
    assert manifest.schema_version == "trajectory-forge-episode/v0.1"
    assert manifest.continuous_from_t0 is True
    assert manifest.privileged_channels_recorded is False
