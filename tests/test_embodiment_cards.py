from __future__ import annotations

from pathlib import Path

import pytest

from trajectory_forge.core.contracts import (
    ActuatorGroupSpec,
    EmbodimentCard,
    EmbodimentValidationError,
    KinematicChainSpec,
    SensorStreamSpec,
)
from trajectory_forge.embodiment.registry import load_embodiment_card


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_lift2_r5a_bimanual_card() -> None:
    card = load_embodiment_card(REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_bimanual.yaml")
    assert card.schema_version == "trajectory-forge-embodiment-card/v0.1"
    assert card.embodiment_id == "lift2_r5a_bimanual"
    assert card.action_dim == 16
    assert set(card.chains.keys()) == {"left_arm", "right_arm"}
    assert set(card.actuator_groups.keys()) == {"left_gripper", "right_gripper"}
    assert card.actuator_groups["left_gripper"].kind == "parallel_gripper"
    assert set(card.actuator_groups["left_gripper"].named_postures.keys()) >= {"open", "closed"}
    assert "proprio" in card.sensor_streams


def test_load_dexterous_tactile_template_card() -> None:
    card = load_embodiment_card(
        REPO_ROOT / "configs" / "embodiments" / "template_bimanual_dexterous_tactile.yaml"
    )
    assert card.schema_version == "trajectory-forge-embodiment-card/v0.1"
    assert card.actuator_groups["right_hand"].kind == "dexterous_hand"
    assert "impedance" in card.actuator_groups["right_hand"].control_modes
    assert set(card.actuator_groups["right_hand"].named_postures.keys()) >= {
        "pre_pinch",
        "tripod_dial",
        "index_poke",
    }
    assert "right_tactile_summary" in card.sensor_streams
    assert "right_tactile_array" in card.sensor_streams


def test_invalid_embodiment_card_raises_validation_error() -> None:
    with pytest.raises(EmbodimentValidationError):
        EmbodimentCard(
            embodiment_id="bad_robot",
            runtime_robot_type="manip/bad/v1",
            action_dim=4,
            chains={
                "left_arm": KinematicChainSpec(
                    chain_id="left_arm",
                    base_frame="base_link",
                    ee_frame="ee_link",
                    dof_indices=(0, 1),
                    joint_names=("j0", "j1"),
                )
            },
            actuator_groups={
                "left_hand": ActuatorGroupSpec(
                    group_id="left_hand",
                    kind="dexterous_hand",
                    dof_indices=(2, 3),
                    joint_names=("h0", "h1"),
                    control_modes=("unsupported_mode",),  # invalid mode
                    named_postures={"open": (0.0, 0.0)},
                )
            },
            sensor_streams={
                "proprio": SensorStreamSpec(
                    stream_id="proprio",
                    modality="proprio",
                    rate_hz=30.0,
                    shape=(4,),
                )
            },
        )

    with pytest.raises(EmbodimentValidationError):
        EmbodimentCard(
            embodiment_id="bad_posture_len",
            runtime_robot_type="manip/bad/v2",
            action_dim=4,
            chains={
                "left_arm": KinematicChainSpec(
                    chain_id="left_arm",
                    base_frame="base_link",
                    ee_frame="ee_link",
                    dof_indices=(0, 1),
                    joint_names=("j0", "j1"),
                )
            },
            actuator_groups={
                "left_gripper": ActuatorGroupSpec(
                    group_id="left_gripper",
                    kind="parallel_gripper",
                    dof_indices=(2, 3),
                    joint_names=("g0", "g1"),
                    control_modes=("position",),
                    named_postures={"open": (0.044,)},  # mismatched length
                )
            },
            sensor_streams={},
        )
