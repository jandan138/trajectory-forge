"""Cold-start isolated OS subprocess runner for distillation_v09 S01-S12 One-Shot (cross-platform Windows/Linux)."""

from __future__ import annotations

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
WORKER_SCRIPT = Path(__file__).resolve().parent / "isaac6_v09_s01_s12_oneshot_worker.py"


def run_cold_start_v09_episode(
    *,
    episode_id: str,
    package_dir: Path,
    output_dir: Path,
    zero_friction: bool = False,
    record_video: bool = True,
    max_frames: int = 0,
    hose_mode: str = "d6_der",
    gpu_id: int = 0,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    job_json = output_dir / "cold_start_job.json"
    res_json = output_dir / "cold_start_result.json"
    log_file = output_dir / "cold_start_stdout.log"
    if res_json.exists():
        res_json.unlink()

    job_payload = {
        "episode_id": episode_id,
        "package_dir": package_dir.resolve().as_posix(),
        "output_dir": output_dir.resolve().as_posix(),
        "zero_friction": bool(zero_friction),
        "record_video": bool(record_video),
        "max_frames": int(max_frames),
        "hose_mode": str(hose_mode),
        "gpu_id": int(gpu_id),
    }
    job_json.write_text(json.dumps(job_payload, indent=2), encoding="utf-8")

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

    tf_src = Path(__file__).resolve().parents[3]
    dev_root = tf_src.parents[1]
    ca_root = dev_root / "ConvertAsset"
    env["PYTHONPATH"] = os.pathsep.join([str(tf_src), str(ca_root)])

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
        raise RuntimeError(f"Cold-start v0.9 worker exited rc={rc} without result JSON:\n{tail}")

    result = json.loads(res_json.read_text(encoding="utf-8"))
    result["cold_start_process"] = {
        "pid": child_pid,
        "return_code": rc,
        "wall_time_s": wall_s,
        "isolated_os_process": True,
        "gpu_id": int(gpu_id),
    }
    res_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
