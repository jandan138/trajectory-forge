"""Cold-start isolated OS subprocess runner for Hose Softbody v10.1 3-Way Isaac Sim 6.0 GPU Benchmark."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def _resolve_isaac_python() -> Path:
    env_py = os.environ.get("ISAAC_PYTHON")
    if env_py and Path(env_py).is_file():
        return Path(env_py)
    for cand in (
        Path("/mnt/data/ganwanshui/isaacsim60/python.sh"),
        Path(r"I:\isaacsim\6.0.1\python.bat"),
    ):
        if cand.is_file():
            return cand
    return Path(r"I:\isaacsim\6.0.1\python.bat")


ISAAC_PYTHON = _resolve_isaac_python()
WORKER_SCRIPT = Path(__file__).resolve().parent / "isaac6_hose_3way_benchmark_worker.py"
CONVERT_ASSET_ROOT = Path(__file__).resolve().parents[5] / "ConvertAsset"


def run_cold_start_hose_v10_trial(
    *,
    trial_id: str,
    method: str,
    output_dir: Path,
    zero_friction: bool = False,
    target_port: str = "lower_inlet",
    record_video: bool = True,
    gpu_id: int = 0,
    timeout_s: int = 900,
) -> dict[str, Any]:
    """Run one isolated OS cold-start Isaac Sim 6.0 240Hz GPU trial across Exp-1, Exp-2, and Exp-3."""
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
        "gpu_id": int(gpu_id),
    }
    job_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    env = os.environ.copy()
    if os.name == "nt":
        env.setdefault("ISAACSIM_PORTABLE_ROOT", r"I:\isaacsim-data\6.0.1")
    else:
        isaac_data = Path("/root/hose_v10_1_work/isaac60_data")
        isaac_data.mkdir(parents=True, exist_ok=True)
        env.setdefault("ISAACSIM_PORTABLE_ROOT", str(isaac_data))
        env.setdefault("XDG_CACHE_HOME", str(isaac_data / "cache"))
        env.setdefault("XDG_DATA_HOME", str(isaac_data / "data"))
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        env.setdefault("ACCEPT_EULA", "Y")
        env.setdefault("PRIVACY_CONSENT", "Y")
        env.pop("CUDA_VISIBLE_DEVICES", None)

    tf_src = str(Path(__file__).resolve().parents[3])
    ca_root = str(CONVERT_ASSET_ROOT)
    env["PYTHONPATH"] = os.pathsep.join([tf_src, ca_root])

    isaac_py = _resolve_isaac_python()
    t0 = time.perf_counter()
    creationflags = 0x00004000 if os.name == "nt" else 0
    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [str(isaac_py), str(WORKER_SCRIPT), "--job-json", str(job_json), "--result-json", str(res_json)],
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
        raise RuntimeError(f"Cold-start hose v10.1 trial {trial_id!r} failed (rc={rc}):\n{tail}")

    res = json.loads(res_json.read_text(encoding="utf-8"))
    res["cold_start_process"] = {
        "pid": child_pid,
        "return_code": rc,
        "wall_time_s": wall_s,
        "isolated_os_process": True,
        "gpu_id": int(gpu_id),
    }
    res_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def run_full_3way_hose_v10_suite(
    out_root: Path,
    *,
    record_video: bool = True,
    gpu_ids: tuple[int, ...] = (0,),
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
    n_workers = max(1, len(gpu_ids))

    def _run_one(idx_item: tuple[int, tuple[str, str, bool, str, bool]]) -> tuple[str, dict[str, Any]]:
        idx, (tid, meth, zf, port, rec) = idx_item
        gid = int(gpu_ids[idx % len(gpu_ids)])
        print(f"[HoseV10.1Suite] Launching cold-start trial: {tid} on GPU {gid} (method={meth}, mu0={zf}, port={port})", flush=True)
        r = run_cold_start_hose_v10_trial(
            trial_id=tid,
            method=meth,
            output_dir=out_root / tid,
            zero_friction=zf,
            target_port=port,
            record_video=rec,
            gpu_id=gid,
        )
        return tid, r

    if n_workers == 1:
        for pair in enumerate(trials_spec):
            tid, r = _run_one(pair)
            results[tid] = r
            pids.add(int(r["cold_start_process"]["pid"]))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for tid, r in pool.map(_run_one, enumerate(trials_spec)):
                results[tid] = r
                pids.add(int(r["cold_start_process"]["pid"]))

    summary = {
        "schema_version": "hose-softbody-v10.1-3way-suite/v1.1",
        "physx_hz": 240,
        "distinct_cold_start_pids": len(pids),
        "trials": results,
    }
    (out_root / "hose_v10_3way_master_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
