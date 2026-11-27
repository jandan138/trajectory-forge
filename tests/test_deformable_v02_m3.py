"""Pure-CPU tests for M3 contracts, Franka Panda kinematics, intake, controllers, and firewall."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from examples.controllers.deformable_v02 import (
    CoffeePastryCrushController,
    CoffeePastryGentleController,
    OfficeClothTearController,
    OfficeClothWipeGentleController,
)
from trajectory_forge.adapters.legacy.deformable_workers.common import (
    build_voxel_tet_grid,
    compute_damage_score,
    compute_edge_stretch_from_points,
    compute_tet_strains_from_points,
    jitter_for_seed,
    save_episode_bundle,
)
from trajectory_forge.core.contracts import (
    ContractValidationError,
    DeformableContractSpec,
    EmbodimentCard,
    FirewallViolationError,
)
from trajectory_forge.trajectory.manifest import build_trajectory_episode_manifest
from trajectory_forge.intake.deformable_v02_intake import load_deformable_v02_package
from trajectory_forge.sdk.panda_kinematics import (
    forward_kinematics_panda,
    gripper_rot_for_grasp,
    panda_pad_world_poses,
    solve_ik_panda,
)


_COFFEE_PKG = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\deformable_v02_pilots\coffee_bar_fragile_pastry_plate\scenario_package"
)
_OFFICE_PKG = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\deformable_v02_pilots\office_desk_wiping_cloth_clean_and_park\scenario_package"
)


def test_embodiment_card_and_kinematics_ik() -> None:
    import yaml

    cfg = yaml.safe_load(
        Path(
            r"I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge\configs\embodiments\franka_panda_tabletop.yaml"
        ).read_text(encoding="utf-8")
    )
    card = EmbodimentCard.from_dict(cfg)
    assert card.embodiment_id == "franka_panda_tabletop"
    assert card.action_dim == 9

    for target in ([-0.18, -0.02, 0.78], [0.18, -0.02, 0.78], [-0.13, 0.04, 0.78], [0.22, -0.04, 0.78]):
        tgt = np.asarray(target, dtype=np.float64)
        rot = gripper_rot_for_grasp(math.radians(35.0))
        q7 = solve_ik_panda(tgt, rot)
        pos, r_out, _ = forward_kinematics_panda(q7)
        assert float(np.linalg.norm(pos - tgt)) < 1e-3
        assert float(np.linalg.norm(r_out - rot)) < 2e-2

    q9 = np.concatenate([solve_ik_panda(np.array([-0.18, -0.02, 0.78]), gripper_rot_for_grasp(0.0)), [0.04, 0.04]])
    ee_p, _, lp, rp, _ = panda_pad_world_poses(q9)
    assert abs(float(lp[2] - ee_p[2]) - 0.0106) < 1e-3
    assert abs(float(np.linalg.norm(lp[:2] - rp[:2])) - 0.092) < 1e-3


def test_m2_intake_and_controllers_emit_valid_9dof_actions() -> None:
    brief_coffee = load_deformable_v02_package(_COFFEE_PKG)
    assert "fragile_pastry" in brief_coffee.objects
    pastry_def = brief_coffee.objects["fragile_pastry"].deformable
    assert isinstance(pastry_def, DeformableContractSpec)
    assert pastry_def.structural_family == "soft_solid"

    brief_office = load_deformable_v02_package(_OFFICE_PKG)
    cloth_def = brief_office.objects["wiping_cloth"].deformable
    assert isinstance(cloth_def, DeformableContractSpec)
    assert cloth_def.structural_family == "cloth_sheet"

    for cls, brief in (
        (CoffeePastryGentleController, brief_coffee),
        (CoffeePastryCrushController, brief_coffee),
        (OfficeClothWipeGentleController, brief_office),
        (OfficeClothTearController, brief_office),
    ):
        ctrl = cls(brief)
        act = ctrl.step(
            {
                "step_idx": 0,
                "t_s": 0.0,
                "qpos": np.zeros(9),
                "tactile_normal_force_n": np.zeros(2),
                "wrist_wrench": np.zeros(6),
            }
        )
        assert act.shape == (9,)
        assert np.all(np.isfinite(act))
        assert 0.0 <= act[7] <= 0.041
        assert 0.0 <= act[8] <= 0.041


def test_continuum_polar_stretch_vs_naive_f_and_firewall(tmp_path: Path) -> None:
    rest, tets = build_voxel_tet_grid((0.07, 0.07, 0.028), (3, 3, 2))
    theta = math.radians(35.0)
    rot_y = np.array(
        [[math.cos(theta), 0.0, math.sin(theta)], [0.0, 1.0, 0.0], [-math.sin(theta), 0.0, math.cos(theta)]],
        dtype=np.float64,
    )
    rotated = rest @ rot_y.T
    polar, naive = compute_tet_strains_from_points(rest, rotated, tets)
    assert naive > 0.55
    assert polar < 1e-6

    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    stretched = rest.copy()
    stretched[1] += np.array([0.02, 0.0, 0.0])
    assert compute_edge_stretch_from_points(rest, stretched, edges) > 0.0
    assert compute_damage_score(0.50, 0.15, 0.45) == 1.0
    assert jitter_for_seed(0) == (0.0, 0.0, 0.0)
    assert jitter_for_seed(1) != (0.0, 0.0, 0.0)

    bundle = save_episode_bundle(
        output_dir=tmp_path,
        episode_id="ep_test",
        task_id="coffee_bar_fragile_pastry_plate",
        engine_id="mujoco4",
        mode="gentle",
        seed=0,
        zero_friction=False,
        replay_source="",
        npz_arrays={
            "qpos": np.zeros((4, 9), dtype=np.float64),
            "action": np.zeros((4, 9), dtype=np.float64),
            "tactile_normal_force_n": np.zeros((4, 2), dtype=np.float64),
            "wrist_wrench": np.zeros((4, 6), dtype=np.float64),
            "deformable_rest_points": rest,
            "deformable_points": np.stack([rest] * 4, axis=0),
            "deformable_elements": tets,
        },
        summary_metrics={"intact": True},
    )
    assert bundle["manifest"]["privileged_channels_recorded"] is False
    try:
        build_trajectory_episode_manifest(
            episode_id="ep_bad",
            scenario_id="coffee_bar_fragile_pastry_plate",
            embodiment_id="franka_panda_tabletop",
            num_steps=4,
            continuous_from_t0=True,
            privileged_channels_recorded=False,
            recorded_channels=("qpos", "privileged_gt_state"),
        )
        raise AssertionError("Expected ContractValidationError for privileged channel")
    except (ContractValidationError, FirewallViolationError):
        pass


def test_compute_tet_strains_from_F_rotation_invariant() -> None:
    from trajectory_forge.adapters.legacy.deformable_workers.common import compute_tet_strains_from_F

    theta = math.radians(35.0)
    r_y = np.array(
        [[math.cos(theta), 0.0, math.sin(theta)], [0.0, 1.0, 0.0], [-math.sin(theta), 0.0, math.cos(theta)]],
        dtype=np.float64,
    )
    f_rot = np.broadcast_to(r_y[None, :, :], (16, 3, 3)).copy()
    p_rot, n_rot = compute_tet_strains_from_F(f_rot)
    assert p_rot < 1e-12
    assert n_rot > 0.75

    f_crush = np.broadcast_to((r_y @ np.diag([1.0, 1.0, 0.45]))[None, :, :], (16, 3, 3)).copy()
    p_crush, n_crush = compute_tet_strains_from_F(f_crush)
    assert abs(p_crush - 0.55) < 1e-6
    assert n_crush > 0.75
