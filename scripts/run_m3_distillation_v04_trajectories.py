"""Execute M3 v0.4 dual-arm 18-DOF distillation rollout campaigns per engine (strictly serialized GPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (str(_SRC_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trajectory_forge.adapters.distillation_workers_v04.common_v04 import V04_TASKS  # noqa: E402
from trajectory_forge.adapters.distillation_workers_v04.runner_v04 import run_v04_engine_batch  # noqa: E402


DEFAULT_PACKAGES_ROOT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\distillation_v04_packages"
)
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "external_artifacts" / "distillation_v04_m3"

COUNTERFACTUAL_MODES: dict[str, str] = {
    "distill_01_burner_ring_flask_mount": "crush",
    "distill_02_head_taper_coaxial_mate": "taper_jam",
    "distill_03_thermometer_septum_align": "bulb_offset",
    "distill_04_bimanual_hose_barb_couple": "nozzle_snap",
    "distill_05_condenser_adapter_receiver_train": "taper_jam",
    "distill_06_reverse_order_disassembly": "wrong_order",
    "distill_07_full_assembly_disassembly_sop": "wrong_order",
}


def build_jobs_for_engine(
    *,
    engine_id: str,
    tasks: tuple[str, ...],
    modes: set[str],
    packages_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for task_id in tasks:
        pkg_dir = packages_root / task_id / "scenario_package"
        out_dir = output_root / task_id / engine_id
        gentle_npz = out_dir / f"{task_id}__{engine_id}__gentle_s0.npz"

        if "gentle" in modes:
            jobs.append(
                {
                    "episode_id": f"{task_id}__{engine_id}__gentle_s0",
                    "task_id": task_id,
                    "package_dir": str(pkg_dir),
                    "output_dir": str(out_dir),
                    "mode": "gentle",
                    "seed": 0,
                    "zero_friction": False,
                }
            )
        if "counterfactual" in modes:
            cf_mode = COUNTERFACTUAL_MODES[task_id]
            jobs.append(
                {
                    "episode_id": f"{task_id}__{engine_id}__{cf_mode}_s0",
                    "task_id": task_id,
                    "package_dir": str(pkg_dir),
                    "output_dir": str(out_dir),
                    "mode": cf_mode,
                    "seed": 0,
                    "zero_friction": False,
                }
            )
        if "zero_friction" in modes:
            jobs.append(
                {
                    "episode_id": f"{task_id}__{engine_id}__zero_friction_s0",
                    "task_id": task_id,
                    "package_dir": str(pkg_dir),
                    "output_dir": str(out_dir),
                    "mode": "gentle",
                    "seed": 0,
                    "zero_friction": True,
                }
            )
        if "replay" in modes and engine_id == "mujoco4":
            for ri in range(5):
                jobs.append(
                    {
                        "episode_id": f"{task_id}__{engine_id}__replay_r{ri}",
                        "task_id": task_id,
                        "package_dir": str(pkg_dir),
                        "output_dir": str(out_dir),
                        "mode": "replay",
                        "seed": 0,
                        "zero_friction": False,
                        "replay_npz": str(gentle_npz),
                    }
                )
        if "seeds" in modes and engine_id == "mujoco4":
            for s_idx in (1, 2, 3):
                jobs.append(
                    {
                        "episode_id": f"{task_id}__{engine_id}__gentle_s{s_idx}",
                        "task_id": task_id,
                        "package_dir": str(pkg_dir),
                        "output_dir": str(out_dir),
                        "mode": "gentle",
                        "seed": s_idx,
                        "zero_friction": False,
                    }
                )
    return jobs


def _chunk_size_for_engine(engine_id: str) -> int:
    if engine_id == "newton18":
        return 4
    if engine_id in ("genesis", "isaac6"):
        return 7
    return 14


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M3 v0.4 dual-arm distillation rollouts per engine.")
    parser.add_argument("--engines", default="mujoco4", help="Comma-separated: mujoco4,newton18,genesis,isaac6")
    parser.add_argument("--tasks", default="all", help="Comma-separated task IDs or 'all'")
    parser.add_argument("--modes", default="gentle,counterfactual,zero_friction", help="Comma-separated modes")
    parser.add_argument("--packages-root", type=Path, default=DEFAULT_PACKAGES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    tasks = V04_TASKS if args.tasks == "all" else tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    modes = {m.strip() for m in args.modes.split(",") if m.strip()}

    summary_rows: list[dict[str, Any]] = []
    for eng in engines:
        chunk_sz = _chunk_size_for_engine(eng)
        base_modes = modes & {"gentle", "counterfactual", "zero_friction"}
        if base_modes:
            jobs = build_jobs_for_engine(
                engine_id=eng,
                tasks=tasks,
                modes=base_modes,
                packages_root=args.packages_root,
                output_root=args.output_root,
            )
            for idx in range(0, len(jobs), chunk_sz):
                results = run_v04_engine_batch(eng, jobs[idx : idx + chunk_sz], timeout_s=900)
                for r in results:
                    summary_rows.append({"episode_id": r["episode_id"], **r["metrics"]})

        follow_modes = modes & {"replay", "seeds"}
        if follow_modes:
            jobs2 = build_jobs_for_engine(
                engine_id=eng,
                tasks=tasks,
                modes=follow_modes,
                packages_root=args.packages_root,
                output_root=args.output_root,
            )
            for idx in range(0, len(jobs2), chunk_sz):
                results2 = run_v04_engine_batch(eng, jobs2[idx : idx + chunk_sz], timeout_s=900)
                for r in results2:
                    summary_rows.append({"episode_id": r["episode_id"], **r["metrics"]})

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"m3_v04_{'_'.join(engines)}_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    for row in summary_rows:
        print(
            f"{row['episode_id']:58s} intact={str(row['intact']):5s} geom={str(row['geom_ok']):5s} "
            f"dmg={row['damage_score']:.3f} fn={row['peak_normal_force_n']:.2f}N "
            f"haz={row.get('hazard_events', [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

