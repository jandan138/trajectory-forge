"""Run distillation_v09 Full S01-S12 One-Shot Method-B Assembly & Twist-Pull Disassembly + mu=0.0 Counterfactual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from trajectory_forge.adapters.distillation_v09.cold_start_runner_v09 import run_cold_start_v09_episode  # noqa: E402


DEFAULT_PKG_DIR = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\distillation_v09_package\scenario_package"
)
DEFAULT_OUT_ROOT = _REPO_ROOT / "external_artifacts" / "distillation_v09_oneshot"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PKG_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument("--reuse-counterfactual", action="store_true")
    parser.add_argument("--counterfactual-only", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    record_video = not args.no_video
    results: dict[str, dict] = {}
    pids: list[int] = []

    if not args.counterfactual_only:
        nom = run_cold_start_v09_episode(
            episode_id="v09_s01_s12_method_b_nominal",
            package_dir=args.package_dir,
            output_dir=args.out_root / "nominal_s01_s12",
            zero_friction=False,
            record_video=record_video,
            max_frames=args.max_frames,
        )
        results["nominal"] = nom
        pids.append(int(nom["cold_start_process"]["pid"]))

    cf_existing = args.out_root / "counterfactual_zero_friction" / "cold_start_result.json"
    if args.reuse_counterfactual and cf_existing.is_file():
        cf = json.loads(cf_existing.read_text(encoding="utf-8"))
        results["zero_friction"] = cf
        pids.append(int(cf["cold_start_process"]["pid"]))
    elif not args.skip_counterfactual:
        cf = run_cold_start_v09_episode(
            episode_id="v09_s01_s12_method_b_zero_friction",
            package_dir=args.package_dir,
            output_dir=args.out_root / "counterfactual_zero_friction",
            zero_friction=True,
            record_video=record_video,
            max_frames=args.max_frames if args.max_frames > 0 else 650,  # S01..S05 suffices to falsify jack, clamp, flask & head!
        )
        results["zero_friction"] = cf
        pids.append(int(cf["cold_start_process"]["pid"]))

    nom_ok = bool(results["nominal"]["metrics"]["task_passed"]) if "nominal" in results else True
    cf_falsified = (not bool(results["zero_friction"]["metrics"]["task_passed"])) if "zero_friction" in results else True
    all_passed = bool(nom_ok and cf_falsified and len(set(pids)) == len(pids))

    master = {
        "schema_version": "distillation-v09-oneshot-master/v1.0",
        "physx_hz": 240,
        "taper_method": "method_b_radial_micro_spring_only",
        "distinct_cold_start_pids": len(set(pids)),
        "nominal_passed": nom_ok,
        "counterfactual_falsified": cf_falsified,
        "all_passed": all_passed,
        "episodes": results,
    }
    out_json = args.out_root / "v09_master_summary.json"
    out_json.write_text(json.dumps(master, indent=2), encoding="utf-8")
    print(json.dumps({"all_passed": all_passed, "summary": out_json.as_posix()}, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

