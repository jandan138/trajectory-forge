"""Strictly serialized 4-engine M3 deformable rollout orchestrator."""

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

from trajectory_forge.adapters.deformable_workers.runner import (  # noqa: E402
    ENGINE_ORDER,
    run_deformable_batch_serialized,
)


DEFAULT_M2_ROOT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\deformable_v02_pilots"
)
DEFAULT_OUT_ROOT = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge\external_artifacts\deformable_v02_m3"
)

TASKS: tuple[tuple[str, str, str], ...] = (
    ("coffee_bar_fragile_pastry_plate", "gentle", "crush"),
    ("office_desk_wiping_cloth_clean_and_park", "gentle", "tear"),
)


def build_standard_jobs(m2_root: Path, out_root: Path, engines: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    jobs_by_engine: dict[str, list[dict[str, Any]]] = {e: [] for e in engines}
    for task_id, ok_mode, cf_mode in TASKS:
        pkg_dir = m2_root / task_id / "scenario_package"
        for eng in engines:
            eng_dir = out_root / task_id / eng
            jobs_by_engine[eng].extend(
                [
                    {
                        "episode_id": f"{task_id}__{eng}__{ok_mode}_s0",
                        "package_dir": str(pkg_dir),
                        "task_id": task_id,
                        "mode": ok_mode,
                        "seed": 0,
                        "zero_friction": False,
                        "output_dir": str(eng_dir),
                    },
                    {
                        "episode_id": f"{task_id}__{eng}__{cf_mode}_s0",
                        "package_dir": str(pkg_dir),
                        "task_id": task_id,
                        "mode": cf_mode,
                        "seed": 0,
                        "zero_friction": False,
                        "output_dir": str(eng_dir),
                    },
                    {
                        "episode_id": f"{task_id}__{eng}__zero_friction_s0",
                        "package_dir": str(pkg_dir),
                        "task_id": task_id,
                        "mode": ok_mode,
                        "seed": 0,
                        "zero_friction": True,
                        "output_dir": str(eng_dir),
                    },
                ]
            )
    return jobs_by_engine


def main() -> int:
    parser = argparse.ArgumentParser(description="Run serialized 4-engine M3 deformable trajectories.")
    parser.add_argument("--m2-root", type=Path, default=DEFAULT_M2_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--engines", nargs="+", default=list(ENGINE_ORDER))
    args = parser.parse_args()

    engines = tuple(args.engines)
    jobs = build_standard_jobs(args.m2_root, args.out_root, engines)
    results = run_deformable_batch_serialized(jobs_by_engine=jobs, engines=engines, timeout_s=480)

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_root / "m3_rollout_summary.json"
    summary_path.write_text(json.dumps({"engines": list(engines), "results": results}, indent=2), encoding="utf-8")

    for eng, items in results.items():
        for item in items:
            m = item["metrics"]
            print(
                f"{eng:8s} | {item['task_id']:40s} | {item['mode']:6s} | zf={str(item['zero_friction']):5s} | "
                f"intact={str(m['intact']):5s} | dmg={m['damage_score']:.3f} | dist={m['final_target_dist_m']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

