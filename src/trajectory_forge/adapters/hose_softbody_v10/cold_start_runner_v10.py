"""Cold-start isolated OS subprocess runner for Hose Softbody v1.0 3-way Isaac Sim 6.0.1 GPU benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ISAAC_PYTHON = Path(r"I:\isaacsim\6.0.1\python.bat")
WORKER_SCRIPT = Path(__file__).resolve().parent / "isaac6_hose_3way_benchmark_worker.py"
CONVERT_ASSET_ROOT = Path(r"I:\cpfs\shared\simulation\zhuzihou\dev\ConvertAsset")


def run_cold_start_hose_v10_trial(
    *,
    trial_id: str,
    method: str,
    output_dir: Path,
    zero_friction: bool = False,
    target_port: str = "lower_inlet",
    record_video: bool = True,
    timeout_s: int = 900,
) -> dict[str, Any]:
    """Run one isolated OS cold-start Isaac Sim 6.0.1 240Hz GPU trial across Exp-1, Exp-2, and Exp-3."""
    output_dir.mkdir(parents=True, exist_ok=True)
    job_json = output_dir / "trial_job.json"
    res_json = output_dir / "trial_result.json"
    log_file = output_dir / "trial_stdout.log"
    if res_json.exists():
        res_json.unlink()

    payload = {
        "trial_id": trial_id,
        "method": method,
        "output_dir": output_dir.resolve().as_posix(),
        "zero_friction": bool(zero_friction),
        "target_port": str(target_port),
        "record_video": bool(record_video),
    }
    job_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["ISAACSIM_PORTABLE_ROOT"] = r"I:\isaacsim-data\6.0.1"
    tf_src = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = f"{tf_src};{CONVERT_ASSET_ROOT}"

    t0 = time.perf_counter()
    creationflags = 0x00004000 if os.name == "nt" else 0
    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [str(ISAAC_PYTHON), str(WORKER_SCRIPT), "--job-json", str(job_json), "--result-json", str(res_json)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )
        child_pid = proc.pid
        rc = proc.wait(timeout=timeout_s)

    wall_s = round(time.perf_counter() - t0, 2)
    if not res_json.is_file():
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"Cold-start hose v1.0 trial {trial_id!r} failed (rc={rc}):\n{tail}")

    res = json.loads(res_json.read_text(encoding="utf-8"))
    res["cold_start_process"] = {
        "pid": child_pid,
        "return_code": rc,
        "wall_time_s": wall_s,
        "isolated_os_process": True,
    }
    res_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def run_full_3way_hose_v10_suite(
    out_root: Path,
    *,
    record_video: bool = True,
) -> dict[str, Any]:
    """Run the complete controlled 3-way suite (Baseline vs Route A vs Route B + Counterfactuals)."""
    out_root.mkdir(parents=True, exist_ok=True)
    trials_spec = [
        ("baseline_bead_chain", "baseline_bead_chain", False, "lower_inlet", record_video),
        ("route_a_d6_der", "route_a_d6_der", False, "lower_inlet", record_video),
        ("route_b_physx_fem", "route_b_physx_fem", False, "lower_inlet", record_video),
        ("route_a_zero_friction", "route_a_d6_der", True, "lower_inlet", record_video),
        ("route_b_zero_friction", "route_b_physx_fem", True, "lower_inlet", record_video),
        ("route_a_wrong_port", "route_a_d6_der", False, "upper_outlet", record_video),
        ("route_b_wrong_port", "route_b_physx_fem", False, "upper_outlet", record_video),
    ]
    results: dict[str, Any] = {}
    pids: set[int] = set()
    for tid, meth, zf, port, rec in trials_spec:
        print(f"[HoseV10Suite] Launching cold-start trial: {tid} (method={meth}, mu0={zf}, port={port})", flush=True)
        r = run_cold_start_hose_v10_trial(
            trial_id=tid,
            method=meth,
            output_dir=out_root / tid,
            zero_friction=zf,
            target_port=port,
            record_video=rec,
        )
        results[tid] = r
        pids.add(int(r["cold_start_process"]["pid"]))

    summary = {
        "schema_version": "hose-softbody-v10-3way-suite/v1.0",
        "physx_hz": 240,
        "distinct_cold_start_pids": len(pids),
        "trials": results,
    }
    (out_root / "hose_v10_3way_master_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

