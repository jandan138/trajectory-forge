"""Strictly serialized multi-engine rollout dispatcher for M3 deformable tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence


ENGINE_ORDER: tuple[str, ...] = ("mujoco4", "newton18", "genesis", "isaac6")

ENGINE_PYTHON_EXECUTABLES: dict[str, str] = {
    "mujoco4": r"C:\Users\Administrator\Miniconda3\envs\sim-mujoco4-py312\python.exe",
    "newton18": r"C:\Users\Administrator\Miniconda3\envs\sim-newton18-py312\python.exe",
    "genesis": r"C:\Users\Administrator\Miniconda3\envs\sim-genesis-py312\python.exe",
    "isaac6": r"I:\isaacsim\6.0.1\python.bat",
}

ENGINE_WORKER_SCRIPTS: dict[str, Path] = {
    "mujoco4": Path(__file__).resolve().with_name("mujoco4_rollout_worker.py"),
    "newton18": Path(__file__).resolve().with_name("newton18_rollout_worker.py"),
    "genesis": Path(__file__).resolve().with_name("genesis_rollout_worker.py"),
    "isaac6": Path(__file__).resolve().with_name("isaac6_rollout_worker.py"),
}


def run_deformable_batch_serialized(
    *,
    jobs_by_engine: dict[str, list[dict[str, Any]]],
    engines: Sequence[str] = ENGINE_ORDER,
    timeout_s: int = 360,
) -> dict[str, list[dict[str, Any]]]:
    """Run deformable worker batches strictly one engine subprocess at a time."""
    outputs: dict[str, list[dict[str, Any]]] = {}
    creationflags = 0x00000040 if sys.platform.startswith("win") else 0  # IDLE_PRIORITY_CLASS

    for engine_id in engines:
        jobs = jobs_by_engine.get(engine_id, [])
        if not jobs:
            outputs[engine_id] = []
            continue
        exe = ENGINE_PYTHON_EXECUTABLES[engine_id]
        worker_script = ENGINE_WORKER_SCRIPTS[engine_id]
        if not worker_script.is_file():
            raise FileNotFoundError(f"Worker script missing for engine {engine_id!r}: {worker_script}")

        env = os.environ.copy()
        if engine_id == "isaac6":
            env.setdefault("ISAACSIM_PORTABLE_ROOT", r"I:\isaacsim-data\6.0.1")
            env["TEMP"] = r"I:\isaacsim-data\6.0.1\temp"
            env["TMP"] = r"I:\isaacsim-data\6.0.1\temp"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            batch_json = td_path / f"{engine_id}_batch.json"
            result_json = td_path / f"{engine_id}_result.json"
            batch_json.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")

            proc = subprocess.run(
                [exe, str(worker_script), "--batch-json", str(batch_json), "--result-json", str(result_json)],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                creationflags=creationflags,
            )
            if proc.returncode != 0 or not result_json.is_file():
                tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-35:])
                raise RuntimeError(f"Engine worker {engine_id!r} failed (rc={proc.returncode}):\n{tail}")
            outputs[engine_id] = json.loads(result_json.read_text(encoding="utf-8"))["results"]
    return outputs

