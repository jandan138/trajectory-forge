"""Unit tests for Pink QP IK, two-tier EE control, pure-layer isolation, and EFE diagnostics."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from trajectory_forge.core.contracts import (
    ActuatorGroupSpec,
    EmbodimentCard,
    KinematicChainSpec,
    RobotSpawnSpec,
    RolloutJobRequest,
    RolloutJobResult,
    ScenarioIntakeBrief,
    SolvabilityDiagnostics,
)
from trajectory_forge.sdk.math3d import Pose3D
from trajectory_forge.sdk.context import EndEffectorAPI, KinematicsAPI
from trajectory_forge.solver.diagnostics import (
    build_feedback_packet,
    render_repair_prompt_markdown,
)


FORBIDDEN_PURE_IMPORTS = {
    "pxr",
    "omni",
    "isaacsim",
    "genmanip",
    "curobo",
    "pinocchio",
    "pink",
    "qpsolvers",
    "torch",
}


def test_pure_layers_never_import_sim_or_ik_sdks() -> None:
    src_root = Path("/cpfs/user/zhuzihou/dev/trajectory-forge/src/trajectory_forge")
    pure_dirs = ("core", "intake", "embodiment", "sdk", "solver", "worker", "trajectory")
    violations: list[str] = []
    for sub in pure_dirs:
        sub_path = src_root / sub
        if not sub_path.is_dir():
            continue
        for py_file in sub_path.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in FORBIDDEN_PURE_IMPORTS:
                            violations.append(f"{py_file}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    if top in FORBIDDEN_PURE_IMPORTS:
                        violations.append(f"{py_file}: from {node.module} import ...")
    assert not violations, f"Pure-layer import boundary violated: {violations}"


class _DummyBackend:
    def __init__(self, n_dof: int = 27) -> None:
        self.proprio = [0.0] * n_dof
        self.commands: list[tuple[str, tuple[float, ...], str, int]] = []
        self.joint_moves: list[tuple[str, tuple[float, ...], int]] = []

    def current_proprio(self) -> tuple[float, ...]:
        return tuple(self.proprio)

    def command_actuator(
        self, group: str, targets: tuple[float, ...], *, control_mode: str = "position", steps: int = 16
    ) -> None:
        self.commands.append((group, targets, control_mode, steps))

    def move_joints(
        self, chain: str, joint_positions: tuple[float, ...], *, steps: int = 30
    ) -> tuple[float, ...]:
        self.joint_moves.append((chain, joint_positions, steps))
        return joint_positions


def test_two_tier_end_effector_and_chain_escape_hatch() -> None:
    finger_names = tuple(f"right_ee_right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5))
    emb = EmbodimentCard(
        embodiment_id="bimanual_fr3_wuji_hand",
        runtime_robot_type="generic_urdf",
        action_dim=27,
        chains={
            "right_arm": KinematicChainSpec(
                chain_id="right_arm",
                base_frame="pedestal_base_link",
                ee_frame="right_grasp_frame",
                dof_indices=tuple(range(7)),
                joint_names=tuple(f"right_arm_fr3_joint{i}" for i in range(1, 8)),
            )
        },
        actuator_groups={
            "right_ee": ActuatorGroupSpec(
                group_id="right_ee",
                kind="dexterous_hand",
                dof_indices=tuple(range(7, 27)),
                joint_names=finger_names,
                control_modes=("position",),
                named_postures={
                    "open": tuple([0.0] * 20),
                    "cylinder_wrap": tuple([0.8] * 20),
                },
            )
        },
        sensor_streams=(),
    )
    backend = _DummyBackend(27)
    ee_api = EndEffectorAPI(emb, backend)  # type: ignore[arg-type]
    kin_api = KinematicsAPI(emb, backend)  # type: ignore[arg-type]

    # Tier 1: interpolated posture closure
    out_mid = ee_api.set_posture("right_ee", "cylinder_wrap", closure=0.5, steps=10)
    assert len(out_mid) == 20
    assert all(abs(v - 0.4) < 1e-6 for v in out_mid)

    # Tier 2: suffix-matched individual finger override
    out_custom = ee_api.set_joint_targets("right_ee", {"finger1_joint1": 0.42, "finger2_joint3": 0.91}, steps=12)
    assert abs(out_custom[0] - 0.42) < 1e-6
    assert abs(out_custom[6] - 0.91) < 1e-6

    # Tier 2: direct arm chain joint targets
    q_arm = kin_api.set_chain_joint_targets("right_arm", [0.1] * 7, steps=15)
    assert len(q_arm) == 7
    assert backend.joint_moves[-1][2] == 15


def test_diagnostics_packet_carries_multimodal_ik_and_contact_fields() -> None:
    brief = ScenarioIntakeBrief(
        package_id="starbucks_v6",
        episode_name="starbucks_barista_pbd_latte",
        instruction="Make latte",
        scene_usd_relpath="scene.usda",
        robot_profile="bimanual_fr3_wuji_hand",
        robot_spawn=RobotSpawnSpec(
            robot_id="bimanual_fr3_wuji_hand",
            xyz=(-0.46, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            joint_positions=(),
        ),
        objects={},
        stages=(),
    )
    emb = EmbodimentCard(
        embodiment_id="bimanual_fr3_wuji_hand",
        runtime_robot_type="generic_urdf",
        action_dim=54,
        chains={},
        actuator_groups={},
        sensor_streams=(),
    )
    req = RolloutJobRequest(
        job_id="j1",
        scenario_package_ref="/tmp/pkg",
        embodiment_id=emb.embodiment_id,
        mode="privileged_search",
        seed=0,
        controller_code="def solve(ctx): pass",
        max_steps=8000,
    )
    res = RolloutJobResult(
        job_id="j1",
        status="failed",
        mode="privileged_search",
        final_score=0.42,
        steps_executed=500,
        snapshots_used=1,
        firewall_verified=False,
        stage_progression=(),
        diagnostics=SolvabilityDiagnostics(failure_code="IK_LIMIT", detail="Right arm hit joint4 limit"),
        media=(),
        trajectory_manifest=None,
        saved_snapshots=(),
        state_summary={
            "ik_failures": [{"stage": 1, "chain": "right_arm", "failure_reason": "joint_limit:right_arm_fr3_joint4"}],
            "ee_contact_metrics": {"ee_group": "right_ee", "slip_detected": True, "wrist_residual_xyz": [0.01, 0.0, 0.02]},
        },
    )
    pkt = build_feedback_packet(attempt=1, request=req, result=res)
    assert len(pkt.ik_failures) == 1
    assert pkt.ee_contact_metrics["slip_detected"] is True
    md = render_repair_prompt_markdown(brief, emb, pkt)
    assert "IK Failures" in md
    assert "EE Contact & Residual Metrics" in md


@pytest.mark.parametrize(
    "robot_id,arm_prefix,flange_suffix,neutral_q",
    [
        ("bimanual_fr3_robotiq85", "right_arm_fr3_joint", "right_grasp_frame", [0.0, -0.55, 0.0, -2.15, 0.0, 1.65, 0.785]),
        ("bimanual_rizon4s_flexiv", "right_arm_joint", "right_grasp_frame", [0.0, -0.55, 0.0, 1.55, 0.0, 0.65, 0.0]),
        ("bimanual_fr3_wuji_hand", "right_arm_fr3_joint", "right_grasp_frame", [0.0, -0.55, 0.0, -2.15, 0.0, 1.65, 0.785]),
    ],
)
def test_pink_qp_ik_converges_on_all_three_assembled_bimanual_urdfs(
    robot_id: str,
    arm_prefix: str,
    flange_suffix: str,
    neutral_q: list[float],
) -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("pink")
    from trajectory_forge.adapters.isaac_genmanip.ik_solver import PinkQPIKSolver

    urdf = Path(f"/cpfs/user/zhuzihou/dev/ConvertAsset/external_artifacts/cross_embodiment_robots_v1/{robot_id}/{robot_id}.urdf")
    if not urdf.is_file():
        pytest.skip(f"Assembled URDF not found: {urdf}")

    right_joints = [f"{arm_prefix}{i}" for i in range(1, 8)]
    left_prefix = arm_prefix.replace("right_", "left_")
    left_joints = [f"{left_prefix}{i}" for i in range(1, 8)]

    solver = PinkQPIKSolver(
        urdf_path=urdf,
        chain_joints={"left_arm": left_joints, "right_arm": right_joints},
        chain_frames={"left_arm": "left_grasp_frame", "right_arm": flange_suffix},
        base_pose_world=Pose3D(xyz=(-0.46, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0)),
    )
    # Hard constraint: reduced model for right_arm MUST have strictly 7 DoF (even for 20-DoF Wuji Hand!)
    assert solver._reduced_models["right_arm"].nq == 7

    q0 = np.asarray(neutral_q, dtype=float)
    p0 = solver.forward_kinematics("right_arm", q0)
    # Perturb target pose by (+6 cm, +4 cm, -5 cm) in world space
    target = Pose3D(
        xyz=(p0.xyz[0] + 0.06, p0.xyz[1] + 0.04, p0.xyz[2] - 0.05),
        wxyz=p0.wxyz,
    )
    res = solver.solve_ik("right_arm", q0, target)
    assert res.converged, f"{robot_id} Pink QP IK failed: {res}"
    assert res.pos_err_m < 0.005
    assert res.rot_err_rad < 0.05
