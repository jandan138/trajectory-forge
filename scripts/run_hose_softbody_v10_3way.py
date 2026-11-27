"""CLI entrypoint to run the 3-way Isaac Sim 6.0.1 240Hz Hose Softbody v1.0 benchmark suite."""

from __future__ import annotations

import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trajectory_forge.adapters.legacy.hose_softbody_v10 import run_full_3way_hose_v10_suite


def main() -> None:
    out_root = _REPO_ROOT / "external_artifacts" / "hose_softbody_v10_3way"
    summary = run_full_3way_hose_v10_suite(out_root, record_video=True)
    print(json.dumps({k: v["exp1_cantilever"] for k, v in summary["trials"].items()}, indent=2))
    print(json.dumps({k: v["exp2_torsion_90deg"] for k, v in summary["trials"].items()}, indent=2))
    print(json.dumps({k: v["exp3_barb_fit"] for k, v in summary["trials"].items()}, indent=2))


if __name__ == "__main__":
    main()

