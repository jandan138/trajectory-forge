"""Pure-CPU tests for M3 v0.3 skill primitives, zero-cheat audit, and MuJoCo4 rollouts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from trajectory_forge.adapters.deformable_workers_v03.common_v03 import V03_TASKS  # noqa: E402
from trajectory_forge.intake.deformable_v02_intake import load_deformable_v02_package  # noqa: E402
from trajectory_forge.skills.deformable_primitives_v03 import make_v03_controller  # noqa: E402


M2_ROOT = Path(r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\deformable_v03_packages")
M3_ROOT = _REPO_ROOT / "external_artifacts" / "deformable_v03_m3"

COUNTERFACTUAL_MODES: dict[str, str] = {
    "coffee_croissant_tray_to_plate": "crush",
    "coffee_strawberry_tart_topping": "crush",
    "office_waffle_towel_corner_fold": "tear",
    "office_dust_cover_peel_retrieve": "tear",
    "smt_ribbon_cable_clip_route": "crush",
    "smt_esd_silicone_boot_press_fit": "crush",
    "lab_silicone_tube_manifold_insert": "crush",
    "lab_rubber_bulb_squeeze_seat": "crush",
}


def test_v03_zero_cheat_source_and_skill_firewall_audit() -> None:
    workers_dir = _SRC_ROOT / "trajectory_forge" / "adapters" / "deformable_workers_v03"
    forbidden = ("<weld", "<connect", "PhysxPhysicsAttachment", "cloth_spine", "set_particles_pos")
    py_files = list(workers_dir.glob("*.py")) + [_SRC_ROOT / "trajectory_forge" / "skills" / "deformable_primitives_v03.py"]
    assert len(py_files) >= 6
    for pyf in py_files:
        txt = pyf.read_text(encoding="utf-8")
        for bad in forbidden:
            assert bad not in txt, f"Forbidden grasping cheat {bad!r} found in {pyf.name}"

    for task_id in V03_TASKS:
        brief = load_deformable_v02_package(M2_ROOT / task_id / "scenario_package")
        ctrl = make_v03_controller(brief, task_id, mode="gentle")
        q0 = ctrl.step({"step_idx": 0, "t_s": 0.0, "qpos": [0.0] * 9, "tactile_normal_force_n": [0.0, 0.0]})
        assert len(q0) == 9


def test_v03_mujoco4_all_8_tasks_intact_and_falsified_counterfactuals() -> None:
    for task_id in V03_TASKS:
        mj_dir = M3_ROOT / task_id / "mujoco4"
        g_sum = json.loads((mj_dir / f"{task_id}__mujoco4__gentle_s0_summary.json").read_text(encoding="utf-8"))
        cf_mode = COUNTERFACTUAL_MODES[task_id]
        cf_sum = json.loads((mj_dir / f"{task_id}__mujoco4__{cf_mode}_s0_summary.json").read_text(encoding="utf-8"))
        zf_sum = json.loads((mj_dir / f"{task_id}__mujoco4__zero_friction_s0_summary.json").read_text(encoding="utf-8"))

        assert g_sum["manifest"]["privileged_channels_recorded"] is False
        assert g_sum["metrics"]["intact"] is True, f"{task_id} gentle failed: {g_sum['metrics']}"
        assert cf_sum["metrics"]["intact"] is False, f"{task_id} counterfactual unexpectedly intact: {cf_sum['metrics']}"
        assert zf_sum["metrics"]["intact"] is False, f"{task_id} zero_friction cheated pass: {zf_sum['metrics']}"

