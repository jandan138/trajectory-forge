"""Cold-start isolated OS subprocess runner for distillation_v09 S01-S12 One-Shot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ISAAC_PYTHON = Path(r"I:\isaacsim\6.0.1\python.bat")
WORKER_SCRIPT = Path(__file__).resolve().parent / "isaac6_v09_s01_s12_oneshot_worker.py"


def run_cold_start_v09_episode(
    *,
    episode_id: str,
    package_dir: Path,
    output_dir: Path,
    zero_friction: bool = False,
    record_video: bool = True,
    max_frames: int = 0,
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
    }
    job_json.write_text(json.dumps(job_payload, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["ISAACSIM_PORTABLE_ROOT"] = r"I:\isaacsim-data\6.0.1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])

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
        raise RuntimeError(f"Cold-start v0.9 worker exited rc={rc} without result JSON:\n{tail}")

    result = json.loads(res_json.read_text(encoding="utf-8"))
    result["cold_start_process"] = {
        "pid": child_pid,
        "return_code": rc,
        "wall_time_s": wall_s,
        "isolated_os_process": True,
    }
    res_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

