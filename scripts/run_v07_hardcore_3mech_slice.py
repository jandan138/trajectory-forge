"""Run the complete distillation_v07 3-Mechanism Full-Dynamic Slice (`nominal` + `zero_friction` = 6 Cold-Start PIDs)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trajectory_forge.adapters.distillation_v07.cold_start_runner_v07 import (  # noqa: E402
    run_v07_cold_start_episode,
    run_v07_full_3mech_slice_with_counterfactuals,
)

DEFAULT_PKG = Path(
    r"I:\cpfs\shared\simulation\zhuzihou\dev\scenario-forge\external_artifacts\distillation_v07_package\scenario_package"
)
DEFAULT_OUT = _REPO_ROOT / "external_artifacts" / "distillation_v07_3mech_slice"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-dir", type=Path, default=DEFAULT_PKG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--single-mech", type=str, default="")
    parser.add_argument("--zero-friction", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    if args.single_mech:
        res = run_v07_cold_start_episode(
            package_dir=args.pkg_dir,
            mech_id=args.single_mech,
            zero_friction=args.zero_friction,
            out_dir=args.out_dir,
            record_video=not args.no_video,
        )
        print(json.dumps({"episode_id": res["episode_id"], "metrics": res["metrics"], "pid": res["provenance"]["process_pid"]}, indent=2))
        return 0

    summary = run_v07_full_3mech_slice_with_counterfactuals(
        package_dir=args.pkg_dir,
        out_dir=args.out_dir,
        record_video=not args.no_video,
    )
    print(
        json.dumps(
            {
                "all_passed": summary["all_passed"],
                "distinct_cold_start_pids": summary["distinct_cold_start_pids"],
                "summary_json": (args.out_dir / "v07_slice_master_summary.json").as_posix(),
            },
            indent=2,
        )
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

