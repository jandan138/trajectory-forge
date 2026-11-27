"""Pure-CPU tests for M3 v0.4 18-DOF skill primitives, zero-cheat audit, and MuJoCo4 rollouts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from trajectory_forge.adapters.legacy.distillation_workers_v04.common_v04 import (  # noqa: E402
    V04_TASKS,
    load_v04_task_context,
)


M2_ROOT = Path(r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\distillation_v04_packages")
M3_ROOT = _REPO_ROOT / "external_artifacts" / "distillation_v04_m3"

COUNTERFACTUAL_MODES: dict[str, str] = {
    "distill_01_burner_ring_flask_mount": "crush",
    "distill_02_head_taper_coaxial_mate": "taper_jam",
    "distill_03_thermometer_septum_align": "bulb_offset",
    "distill_04_bimanual_hose_barb_couple": "nozzle_snap",
    "distill_05_condenser_adapter_receiver_train": "taper_jam",
    "distill_06_reverse_order_disassembly": "wrong_order",
    "distill_07_full_assembly_disassembly_sop": "wrong_order",
}


def test_v04_zero_cheat_source_and_18dof_controller_audit() -> None:
    workers_dir = _SRC_ROOT / "trajectory_forge" / "adapters" / "distillation_workers_v04"
    forbidden = (
        "<weld",
        "<connect",
        "PhysxPhysicsAttachment",
        "OmniPhysicsVtxXformAttachment",
        "cloth_spine",
        "set_particles_pos",
    )
    py_files = list(workers_dir.glob("*.py")) + [
        _SRC_ROOT / "trajectory_forge" / "skills" / "distillation_bimanual_v04.py"
    ]
    assert len(py_files) >= 6
    for pyf in py_files:
        txt = pyf.read_text(encoding="utf-8")
        for bad in forbidden:
            assert bad not in txt, f"Forbidden grasping cheat {bad!r} found in {pyf.name}"

    for tid in V04_TASKS:
        ctx = load_v04_task_context(M2_ROOT / tid / "scenario_package", tid, "gentle")
        ctrl = ctx["controller"]
        q0 = ctrl.step(
            {"step_idx": 0, "t_s": 0.0, "qpos": [0.0] * 18, "tactile_normal_force_n": [0.0] * 4}
        )
        assert len(q0) == 18


def test_v04_mujoco4_all_7_tasks_intact_and_falsified_counterfactuals() -> None:
    for tid in V04_TASKS:
        mj_dir = M3_ROOT / tid / "mujoco4"
        g_sum = json.loads((mj_dir / f"{tid}__mujoco4__gentle_s0_summary.json").read_text(encoding="utf-8"))
        cf_mode = COUNTERFACTUAL_MODES[tid]
        cf_sum = json.loads((mj_dir / f"{tid}__mujoco4__{cf_mode}_s0_summary.json").read_text(encoding="utf-8"))
        zf_sum = json.loads((mj_dir / f"{tid}__mujoco4__zero_friction_s0_summary.json").read_text(encoding="utf-8"))

        assert g_sum["manifest"]["privileged_channels_recorded"] is False
        assert g_sum["metrics"]["intact"] is True, f"{tid} gentle failed: {g_sum['metrics']}"
        assert cf_sum["metrics"]["intact"] is False, f"{tid} counterfactual unexpectedly intact: {cf_sum['metrics']}"
        assert zf_sum["metrics"]["intact"] is False, f"{tid} zero_friction cheated pass: {zf_sum['metrics']}"

