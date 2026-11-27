from __future__ import annotations

import math
from pathlib import Path

from trajectory_forge.core.contracts import RobotSpawnSpec, ScenarioIntakeBrief
from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.sdk.context import ControllerContext, DryRunSimBackend
from trajectory_forge.sdk.math3d import Pose3D
from trajectory_forge.trajectory.conveyor_math import (
    eval_racetrack_pose,
    racetrack_perimeter,
)


def _make_dummy_brief() -> ScenarioIntakeBrief:
    return ScenarioIntakeBrief(
        package_id="dummy_bimanual_test",
        episode_name="0",
        scene_usd_relpath="scene.usda",
        robot_profile="manip/lift2/R5a",
        robot_spawn=RobotSpawnSpec(
            robot_id="lift2",
            xyz=(0.0, -0.96, 0.0),
            wxyz=(0.70710678, 0.0, 0.0, 0.70710678),
            joint_positions=(),
        ),
        instruction="Test bimanual sync",
        objects={},
        stages=(),
        part_ids=(),
    )


def test_racetrack_perimeter_and_continuity() -> None:
    L = 0.42
    R = 0.16
    perim = racetrack_perimeter(L, R)
    assert math.isclose(perim, 4.0 * L + 2.0 * math.pi * R, rel_tol=1e-9)

    # Check periodicity at s=0 and s=perim
    p0 = eval_racetrack_pose(0.0, L, R, (0.0, 0.08), 0.778)
    p1 = eval_racetrack_pose(perim, L, R, (0.0, 0.08), 0.778)
    for a, b in zip(p0[:3], p1[:3]):
        assert math.isclose(a, b, abs_tol=1e-7)

    # Check front straight midpoint (s = L -> x=0.0, y=0.08 - R = -0.08)
    pf = eval_racetrack_pose(L, L, R, (0.0, 0.08), 0.778)
    assert math.isclose(pf[0], 0.0, abs_tol=1e-7)
    assert math.isclose(pf[1], -0.08, abs_tol=1e-7)
    assert math.isclose(pf[2], 0.778, abs_tol=1e-7)

    # Check back straight midpoint (s = 3L + pi*R -> x=0.0, y=0.08 + R = 0.24)
    pb = eval_racetrack_pose(3.0 * L + math.pi * R, L, R, (0.0, 0.08), 0.778)
    assert math.isclose(pb[0], 0.0, abs_tol=1e-7)
    assert math.isclose(pb[1], 0.24, abs_tol=1e-7)

    # Check step-to-step C0 continuity around entire loop
    n = 400
    ds = perim / n
    prev = eval_racetrack_pose(0.0, L, R, (0.0, 0.08), 0.778)
    for i in range(1, n + 1):
        cur = eval_racetrack_pose(i * ds, L, R, (0.0, 0.08), 0.778)
        dist = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        assert abs(dist - ds) < 5e-4
        prev = cur


def test_bimanual_sync_apis_in_dryrun() -> None:
    card_path = Path(__file__).resolve().parents[1] / "configs/embodiments/lift2_r5a_bimanual.yaml"
    emb = load_embodiment_card(card_path)
    brief = _make_dummy_brief()
    backend = DryRunSimBackend(brief=brief, embodiment=emb, mode="full_horizon_gate")
    ctx = ControllerContext(brief=brief, embodiment=emb, backend=backend, mode="full_horizon_gate")

    p_left = Pose3D(xyz=(-0.10, -0.10, 0.85), wxyz=(1.0, 0.0, 0.0, 0.0))
    p_right = Pose3D(xyz=(0.10, -0.10, 0.85), wxyz=(1.0, 0.0, 0.0, 0.0))

    steps_before = backend.steps_executed
    ctx.kinematics.move_arms_sync(
        {"left_arm": p_left, "right_arm": p_right},
        steps=20,
    )
    assert backend.steps_executed == steps_before + 20
    assert ctx.kinematics.ee_pose("left_arm").xyz == p_left.xyz
    assert ctx.kinematics.ee_pose("right_arm").xyz == p_right.xyz

    steps_before_g = backend.steps_executed
    res = ctx.end_effector.set_postures_sync(
        {"left_gripper": "closed", "right_gripper": "closed"},
        steps=12,
    )
    assert backend.steps_executed == steps_before_g + 12
    assert res["left_gripper"] == tuple(emb.actuator_groups["left_gripper"].named_postures["closed"])
    assert res["right_gripper"] == tuple(emb.actuator_groups["right_gripper"].named_postures["closed"])
