"""Isolated subprocess batch dispatcher for M3 v0.4 dual-arm distillation workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


ENGINE_EXECUTABLES: dict[str, Path] = {
    "mujoco4": Path(r"C:\Users\Administrator\Miniconda3\envs\sim-mujoco4-py312\python.exe"),
    "newton18": Path(r"C:\Users\Administrator\Miniconda3\envs\sim-newton18-py312\python.exe"),
    "genesis": Path(r"C:\Users\Administrator\Miniconda3\envs\sim-genesis-py312\python.exe"),
    "isaac6": Path(r"I:\isaacsim\6.0.1\python.bat"),
}

WORKER_SCRIPTS: dict[str, str] = {
    "mujoco4": "mujoco4_v04_worker.py",
    "newton18": "newton18_v04_worker.py",
    "genesis": "genesis_v04_worker.py",
    "isaac6": "isaac6_v04_worker.py",
}


def run_v04_engine_batch(
    engine_id: str,
    jobs: list[dict[str, Any]],
    *,
    timeout_s: int = 900,
) -> list[dict[str, Any]]:
    """Run v0.4 rollout jobs inside the dedicated isolated simulator process and parse JSON results."""
    if engine_id not in ENGINE_EXECUTABLES:
        raise ValueError(f"Unsupported engine_id: {engine_id!r}")
    exe = ENGINE_EXECUTABLES[engine_id]
    if not exe.is_file():
        raise FileNotFoundError(f"Simulator runtime missing for {engine_id}: {exe}")
    worker = Path(__file__).resolve().parent / WORKER_SCRIPTS[engine_id]

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        batch_json = td_p / f"{engine_id}_v04_batch.json"
        result_json = td_p / f"{engine_id}_v04_result.json"
        batch_json.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")

        env = os.environ.copy()
        if engine_id == "isaac6":
            env["ISAACSIM_PORTABLE_ROOT"] = r"I:\isaacsim-data\6.0.1"
            env["TEMP"] = r"I:\isaacsim-data\6.0.1\temp"
            env["TMP"] = r"I:\isaacsim-data\6.0.1\temp"

        creationflags = 0x00000040 if os.name == "nt" else 0  # IDLE_PRIORITY_CLASS
        subprocess.run(
            [str(exe), str(worker), "--batch-json", str(batch_json), "--result-json", str(result_json)],
            check=True,
            env=env,
            timeout=timeout_s,
            creationflags=creationflags,
        )
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        return list(payload["results"])

