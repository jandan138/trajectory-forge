"""End-to-end tests for Microkernel v2: Intake Registry, Producers, FullHorizonGate & Newton SimAdapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trajectory_forge.adapters import (
    DEFAULT_SIM_ADAPTER_REGISTRY,
    IsaacGenManipSimAdapter,
    NewtonSimAdapter,
)
from trajectory_forge.core.contracts import (
    ContractValidationError,
    FirewallViolationError,
    PoseSpec,
)
from trajectory_forge.core.microkernel import (
    ActionCommand,
    CheckpointPolicy,
    CompositePolicy,
    RelativeWaypoint,
    SimCapability,
    StepObservation,
    WaypointPolicy,
)
from trajectory_forge.embodiment import load_embodiment_card
from trajectory_forge.intake import (
    USD_DIRECT_SCHEMA,
    load_scenario_package,
    render_agent_brief_markdown,
)
from trajectory_forge.producers import (
    DEFAULT_PRODUCER_REGISTRY,
    CodeToControlProducer,
    DirectControlProducer,
    DirectControlSession,
    RLPolicyProducer,
)
from trajectory_forge.verifier import FullHorizonGate, GateConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
FR3_CARD_PATH = REPO_ROOT / "configs" / "embodiments" / "bimanual_fr3_robotiq85.yaml"
FR3_URDF_PATH = Path(
    "/cpfs/user/zhuzihou/dev/ConvertAsset/external_artifacts/cross_embodiment_robots_v1/"
    "bimanual_fr3_robotiq85/bimanual_fr3_robotiq85.urdf"
)


def _write_usd_direct_scenario(tmp_path: Path) -> Path:
    spec = {
        "schema": USD_DIRECT_SCHEMA,
        "package_id": "newton_cup_reach_v1",
        "episode_name": "ep_000",
        "instruction": "Reach right_grasp_frame cleanly above the target_cup across randomized seeds.",
        "scene_usd": "",
        "robot_profile": "generic_urdf",
        "robot_spawn": {"robot_id": "bimanual_fr3_robotiq85", "xyz": [0.0, 0.0, 0.0], "wxyz": [1.0, 0.0, 0.0, 0.0]},
        "objects": [
            {
                "object_id": "target_cup",
                "role": "manipulable",
                "initial_pose": {"xyz": [0.46, -0.22, 1.05], "wxyz": [1.0, 0.0, 0.0, 0.0]},
            }
        ],
        "stages": [
            {
                "stage_index": 0,
                "metric_id": "reach_target_cup",
                "clauses": [
                    {
                        "type": "ee_near_object",
                        "chain": "right_arm",
                        "object_id": "target_cup",
                        "offset_xyz": [0.0, 0.0, 0.06],
                        "tolerance_m": 0.055,
                    }
                ],
            }
        ],
    }
    scenario_dir = tmp_path / "newton_cup_reach_v1"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "scenario_direct.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return scenario_dir


def test_registry_intake_and_capability_checks(tmp_path: Path) -> None:
    scenario_dir = _write_usd_direct_scenario(tmp_path)
    brief = load_scenario_package(scenario_dir)
    assert brief.package_id == "newton_cup_reach_v1"
    assert "target_cup" in brief.objects

    embodiment = load_embodiment_card(FR3_CARD_PATH)
    md = render_agent_brief_markdown(brief, embodiment)
    assert "newton_cup_reach_v1" in md
    assert "target_cup" in md

    assert set(DEFAULT_SIM_ADAPTER_REGISTRY.list_names()) >= {"isaac_genmanip", "newton"}
    assert set(DEFAULT_PRODUCER_REGISTRY.list_names()) >= {
        "code_to_control",
        "direct_control",
        "rl_policy",
    }

    # RLPolicyProducer must fail fast on single-env-only IsaacGenManipSimAdapter
    isaac_adapter = IsaacGenManipSimAdapter()
    assert SimCapability.VEC_ENV not in isaac_adapter.capabilities
    with pytest.raises(ContractValidationError, match="vec_env"):
        RLPolicyProducer().produce(brief, embodiment, isaac_adapter, trainer_fn=lambda *_: None)  # type: ignore[arg-type]


@pytest.mark.skipif(not FR3_URDF_PATH.is_file(), reason="Assembled FR3+Robotiq85 URDF not present")
def test_newton_checkpoint_firewall_enforcement(tmp_path: Path) -> None:
    brief = load_scenario_package(_write_usd_direct_scenario(tmp_path))
    embodiment = load_embodiment_card(FR3_CARD_PATH)
    adapter = NewtonSimAdapter()

    # In privileged_search, save & restore checkpoints work and restore exact step index
    s_search = adapter.open_session(brief, embodiment, mode="privileged_search", seed=0)
    try:
        obs0 = s_search.reset(seed=0)
        snap = s_search.save_checkpoint("s0")
        obs1 = s_search.step(ActionCommand(ee_postures={"right_gripper": "closed"}, substeps=3))
        assert obs1.step_index == 3
        obs_restored = s_search.restore_checkpoint(snap)
        assert obs_restored.step_index == obs0.step_index == 0
        assert s_search.snapshots_used == 1
    finally:
        s_search.close()

    # In full_horizon_gate, checkpoint save/restore is strictly blocked
    s_gate = adapter.open_session(brief, embodiment, mode="full_horizon_gate", seed=0)
    try:
        s_gate.reset(seed=0)
        with pytest.raises(FirewallViolationError):
            s_gate.save_checkpoint("illegal")
    finally:
        s_gate.close()


@pytest.mark.skipif(not FR3_URDF_PATH.is_file(), reason="Assembled FR3+Robotiq85 URDF not present")
def test_newton_end_to_end_direct_control_and_code_to_control_pass_multi_seed_gate(
    tmp_path: Path,
) -> None:
    brief = load_scenario_package(_write_usd_direct_scenario(tmp_path))
    embodiment = load_embodiment_card(FR3_CARD_PATH)
    adapter = NewtonSimAdapter()
    gate = FullHorizonGate()

    # 1. Direct Control interactive probing -> compiles object-relative WaypointPolicy -> Multi-seed Gate
    def _interactive_probe(dc: DirectControlSession, _b: Any, _e: Any) -> None:
        snap = dc.save_checkpoint("before_probe")
        cup_xyz = dc.last_obs.object_poses["target_cup"].xyz
        dc.step_action(
            ActionCommand(
                ee_pose_targets={
                    "right_arm": PoseSpec(
                        xyz=(cup_xyz[0], cup_xyz[1], cup_xyz[2] + 0.06),
                        wxyz=(0.0, 1.0, 0.0, 0.0),
                    )
                },
                ee_postures={"right_gripper": "open"},
                substeps=8,
            ),
            label="probe_hover",
        )
        dc.restore_checkpoint(snap)
        dc.record_waypoint(
            RelativeWaypoint(
                label="hover_above_cup",
                target_object="target_cup",
                offset_xyz=(0.0, 0.0, 0.06),
                target_wxyz=(0.0, 1.0, 0.0, 0.0),
                chain="right_arm",
                ee_postures={"right_gripper": "open"},
                hold_steps=35,
            )
        )

    wp_policy = DirectControlProducer().produce(
        brief, embodiment, adapter, interactive_driver=_interactive_probe, seed=0
    )
    rep_direct = gate.verify(
        policy=wp_policy,
        adapter=adapter,
        brief=brief,
        embodiment=embodiment,
        output_dir=tmp_path / "gate_direct",
        config=GateConfig(seeds=(0, 1), max_steps=60, success_threshold=0.99),
    )
    assert rep_direct.passed is True, rep_direct.to_dict()
    assert rep_direct.firewall_verified is True
    assert rep_direct.seeds_passed == 2
    assert rep_direct.sim_backend == "newton"
    assert rep_direct.producer_name == "direct_control"
    npz_path = Path(rep_direct.outcomes[0].trajectory_npz or "")
    assert npz_path.is_file()
    data = np.load(npz_path, allow_pickle=True)
    assert bool(data["privileged_channels_recorded"]) is False
    assert bool(data["continuous_from_t0"]) is True
    assert int(data["snapshots_used"]) == 0

    # 2. Code-to-Control Producer -> PythonControllerPolicy -> FullHorizonGate
    code = """
def act(obs, state):
    cup = obs.object_poses["target_cup"].xyz
    return ActionCommand(
        ee_pose_targets={
            "right_arm": PoseSpec(
                xyz=(cup[0], cup[1], cup[2] + 0.06),
                wxyz=(0.0, 1.0, 0.0, 0.0),
            )
        },
        ee_postures={"right_gripper": "open"},
        substeps=2,
    )
"""
    code_policy = CodeToControlProducer().produce(
        brief, embodiment, adapter, source_code=code, max_steps=60
    )
    rep_code = gate.verify(
        policy=code_policy,
        adapter=adapter,
        brief=brief,
        embodiment=embodiment,
        output_dir=tmp_path / "gate_code",
        config=GateConfig(seeds=(0, 2), max_steps=60, success_threshold=0.99),
    )
    assert rep_code.passed is True, rep_code.to_dict()
    assert rep_code.producer_name == "code_to_control"
    assert rep_code.policy_kind == "python_controller"


@pytest.mark.skipif(not FR3_URDF_PATH.is_file(), reason="Assembled FR3+Robotiq85 URDF not present")
def test_newton_vec_env_rl_producer_and_composite_policy_gate(tmp_path: Path) -> None:
    brief = load_scenario_package(_write_usd_direct_scenario(tmp_path))
    embodiment = load_embodiment_card(FR3_CARD_PATH)
    adapter = NewtonSimAdapter()

    # Exercise real 4-world vectorized Newton stepping inside RLPolicyProducer
    def _mock_vec_rl_trainer(b: Any, e: Any, ad: Any, cfg: dict[str, Any]) -> CheckpointPolicy:
        vec = ad.open_vec_session(b, e, num_envs=int(cfg.get("num_envs", 4)), seed=0, device="cpu")
        try:
            batch_out = vec.step_batch()
            assert batch_out["body_q"].shape[0] == 4
            assert batch_out["joint_q"].shape[0] == 4
        finally:
            vec.close()

        def _policy_infer(obs: StepObservation, state: dict[str, Any]) -> ActionCommand:
            cup = obs.object_poses["target_cup"].xyz
            if obs.total_score >= 0.99:
                state["done"] = True
            return ActionCommand(
                ee_pose_targets={
                    "right_arm": PoseSpec(
                        xyz=(cup[0], cup[1], cup[2] + 0.06),
                        wxyz=(0.0, 1.0, 0.0, 0.0),
                    )
                },
                ee_postures={"right_gripper": "closed"},
                substeps=2,
            )

        return CheckpointPolicy(
            checkpoint_path=str(tmp_path / "dummy_rl_skill.pt"),
            inference_fn=_policy_infer,
            producer_name="rl_policy",
            max_steps=60,
            metadata={"num_envs_trained": 4},
        )

    rl_policy = RLPolicyProducer().produce(
        brief, embodiment, adapter, trainer_fn=_mock_vec_rl_trainer, train_config={"num_envs": 4}
    )

    # Compose a short pre-position waypoint + RL skill inside CompositePolicy and pass FullHorizonGate
    pre_wp = WaypointPolicy(
        (
            RelativeWaypoint(
                label="pre_approach",
                target_object="target_cup",
                offset_xyz=(0.0, 0.0, 0.14),
                target_wxyz=(0.0, 1.0, 0.0, 0.0),
                chain="right_arm",
                hold_steps=6,
            ),
        ),
        producer_name="direct_control",
    )
    composite = CompositePolicy(
        [("pre_approach", pre_wp), ("rl_reach_skill", rl_policy)],
        producer_name="composite_code_plus_rl",
    )
    rep = FullHorizonGate().verify(
        policy=composite,
        adapter=adapter,
        brief=brief,
        embodiment=embodiment,
        output_dir=tmp_path / "gate_composite",
        config=GateConfig(seeds=(0,), max_steps=80, success_threshold=0.99),
    )
    assert rep.passed is True, rep.to_dict()
    assert rep.policy_kind == "composite"
    assert rep.firewall_verified is True
