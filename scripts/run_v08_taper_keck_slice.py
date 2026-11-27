"""Run the distillation_v08 4-Taper-Joint + Method A/B + 3-Way Pull-Out Disassembly Suite (`13` Cold-Start PIDs)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trajectory_forge.adapters.legacy.distillation_v08.cold_start_runner_v08 import (  # noqa: E402
    run_v08_cold_start_episode,
    run_v08_full_4joint_slice_with_counterfactuals,
)

DEFAULT_PKG = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\distillation_v08_package\scenario_package"
)
DEFAULT_OUT = _REPO_ROOT / "external_artifacts" / "distillation_v08_4taper_slice"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-dir", type=Path, default=DEFAULT_PKG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--single-joint", type=str, default="")
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--zero-friction", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    if args.single_joint:
        res = run_v08_cold_start_episode(
            package_dir=args.pkg_dir,
            joint_id=args.single_joint,
            zero_friction=args.zero_friction,
            variant=args.variant,
            out_dir=args.out_dir,
            record_video=not args.no_video,
        )
        print(json.dumps({"episode_id": res["episode_id"], "metrics": res["metrics"], "pid": res["provenance"]["process_pid"]}, indent=2))
        return 0

    summary = run_v08_full_4joint_slice_with_counterfactuals(
        package_dir=args.pkg_dir,
        out_dir=args.out_dir,
        record_video=not args.no_video,
    )
    print(
        json.dumps(
            {
                "all_passed": summary["all_passed"],
                "distinct_cold_start_pids": summary["distinct_cold_start_pids"],
                "ab_05_passed": summary["ab_05_unclamped_15deg_comparison"]["passed"],
                "ab_06_passed": summary["ab_06_disassembly_comparison"]["passed"],
                "ab_06_force_reduction_pct": summary["ab_06_disassembly_comparison"]["axial_force_reduction_pct"],
                "summary_json": (args.out_dir / "v08_slice_master_summary.json").as_posix(),
            },
            indent=2,
        )
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
