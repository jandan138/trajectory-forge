"""M4 verification tests for v0.5 Material-Morph Firewall and tactile-adaptive skill controller."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from trajectory_forge.core.material_morph_firewall import (
    MaterialMorphFirewallError,
    MaterialMorphFirewallProxy,
    audit_controller_material_firewall,
    validate_exported_morph_manifest,
)
from trajectory_forge.skills import tactile_adaptive_v05
from trajectory_forge.skills.tactile_adaptive_v05 import (
    AdaptiveDeformableSkillController,
    run_tactile_adaptive_rollout,
)


def test_m4_material_morph_firewall_blocks_privileged_peeking_and_cheats() -> None:
    proxy = MaterialMorphFirewallProxy({"preload_tactile_forces_n": [0.5, 1.0]}, mode="protocol_c1")
    with pytest.raises(MaterialMorphFirewallError):
        _ = proxy["delta_q"]
    with pytest.raises(MaterialMorphFirewallError):
        _ = proxy.morphed_youngs_modulus_pa

    bad_code = "def solve(ctx):\n    dq = ctx.delta_q\n    return '<weld/>'\n"
    violations = audit_controller_material_firewall(bad_code)
    assert len(violations) >= 2

    # The v0.5 adaptive controller itself must be 100% clean under static AST firewall audit
    ctrl_src = inspect.getsource(AdaptiveDeformableSkillController)
    assert audit_controller_material_firewall(ctrl_src) == []


def test_m4_zero_edit_tactile_adaptive_rollout_across_stiffness_sweep_and_counterfactuals(
    tmp_path: Path,
) -> None:
    sweep = [
        ("nominal", 14000.0, 0.28),
        ("soften_2x", 7000.0, 0.28),
        ("stiffen_2x", 28000.0, 0.28),
        ("poisson_high", 15000.0, 0.42),
    ]
    for var_id, e_pa, nu in sweep:
        res = run_tactile_adaptive_rollout(
            episode_id=f"ep_{var_id}",
            task_id="coffee_croissant_tray_to_plate",
            engine_id="mujoco4",
            variant_id=var_id,
            youngs_modulus_pa=e_pa,
            poissons_ratio=nu,
            dimensions_xyz_m=(0.09, 0.055, 0.045),
            mass_kg=0.065,
            max_normal_force_n=4.5,
            out_npz_path=tmp_path / f"{var_id}.npz",
        )
        validate_exported_morph_manifest(res["manifest"])
        assert res["metrics"]["lifted"] is True
        assert res["metrics"]["final_target_dist_m"] < 0.01
        assert res["metrics"]["peak_tactile_normal_force_n"] <= 4.5

    # Zero-friction falsification must fail lift/placement
    zf = run_tactile_adaptive_rollout(
        episode_id="ep_zero_friction",
        task_id="coffee_croissant_tray_to_plate",
        engine_id="mujoco4",
        variant_id="nominal",
        youngs_modulus_pa=14000.0,
        poissons_ratio=0.28,
        dimensions_xyz_m=(0.09, 0.055, 0.045),
        mass_kg=0.065,
        max_normal_force_n=4.5,
        out_npz_path=tmp_path / "zero_friction.npz",
        zero_friction=True,
    )
    assert zf["metrics"]["lifted"] is False
    assert zf["metrics"]["final_target_dist_m"] > 0.15
    assert hasattr(tactile_adaptive_v05, "AdaptiveDeformableSkillController")

