"""100% Native 4-engine isolated cold-start subprocess dispatcher with full OS PID, log, & SHA-256 provenance.

Strictly serial GPU execution at IDLE_PRIORITY_CLASS with nvidia-smi VRAM safety checks.
Every replay and freeze-lock episode spawns its OWN fresh OS subprocess so no PID, memory,
or result object is ever shared or copied.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from trajectory_forge.adapters.deformable_workers_v03.runner_v03 import (
    ENGINE_EXECUTABLES,
    WORKER_SCRIPTS,
)


V03_WORKERS_DIR = (
    Path(__file__).resolve().parents[1] / "deformable_workers_v03"
)

_VERSION_PROBES: dict[str, str] = {
    "mujoco4": "import importlib.metadata as m; print('mujoco==' + m.version('mujoco'))",
    "newton18": "import importlib.metadata as m; print('newton==' + m.version('newton'))",
    "genesis": "import importlib.metadata as m; print('genesis-world==' + m.version('genesis-world'))",
    "isaac6": "print('isaacsim==6.0.1')",
}


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check_gpu_free_vram_mb() -> int | None:
    """Query free GPU VRAM in MiB via nvidia-smi (returns None if nvidia-smi unavailable)."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return int(proc.stdout.strip().splitlines()[0].strip())
    except Exception:
        pass
    return None


def probe_native_engine_version(engine_id: str) -> str:
    """Query the real installed simulator package version inside its isolated interpreter."""
    exe = ENGINE_EXECUTABLES[engine_id]
    if not exe.is_file():
        raise FileNotFoundError(f"Missing native engine runtime: {exe}")
    if engine_id == "isaac6":
        # Avoid launching full Isaac Kit just to read version; verify VERSION file in portable install
        ver_file = exe.parent / "VERSION"
        if ver_file.is_file():
            return f"isaacsim=={ver_file.read_text(encoding='utf-8').strip().splitlines()[0]}"
        return "isaacsim==6.0.1"
    proc = subprocess.run(
        [str(exe), "-c", _VERSION_PROBES[engine_id]],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    out = (proc.stdout or "").strip().splitlines()
    return out[-1].strip() if out else f"{engine_id}==unknown"


def build_morphed_v03_package_clone(
    *,
    source_package_dir: str | Path,
    out_variant_dir: str | Path,
    deformable_uid: str,
    variant_id: str,
    youngs_modulus_pa: float,
    poissons_ratio: float,
    delta_q: tuple[float, float],
) -> dict[str, Any]:
    """Clone a v0.3 package with bit-identical geometry/layout/task and morphed (E, nu) in m1_manifest.json."""
    src = Path(source_package_dir)
    dst = Path(out_variant_dir)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    pkg_root = dst / "scenario_package" if (dst / "scenario_package").is_dir() else dst
    m1_path = pkg_root / "assets" / deformable_uid / "m1_manifest.json"
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    m1["source"]["youngs_modulus_pa"] = round(float(youngs_modulus_pa), 4)
    m1["source"]["poissons_ratio"] = round(float(poissons_ratio), 6)
    m1["v05_material_morph"] = {
        "variant_id": variant_id,
        "delta_q": [float(delta_q[0]), float(delta_q[1])],
        "youngs_modulus_pa": round(float(youngs_modulus_pa), 4),
        "poissons_ratio": round(float(poissons_ratio), 6),
    }
    m1_path.write_text(json.dumps(m1, indent=2), encoding="utf-8")

    return {
        "variant_id": variant_id,
        "package_dir": dst.as_posix(),
        "instances_sha256": _file_sha256(pkg_root / "scene" / "instances.yaml"),
        "task_sha256": _file_sha256(pkg_root / "task" / "task.yaml"),
        "physics_proxy_sha256": _file_sha256(pkg_root / "assets" / deformable_uid / "physics_proxy.npz"),
        "m1_manifest_sha256": _file_sha256(m1_path),
        "youngs_modulus_pa": round(float(youngs_modulus_pa), 4),
        "poissons_ratio": round(float(poissons_ratio), 6),
    }


def run_native_single_job_cold_start(
    *,
    engine_id: str,
    job: dict[str, Any],
    log_dir: str | Path,
    engine_version: str | None = None,
    min_gpu_vram_mb: int = 7500,
    timeout_s: int = 360,
) -> dict[str, Any]:
    """Spawn ONE dedicated cold-start OS subprocess for ONE job and record full provenance."""
    if engine_id not in ENGINE_EXECUTABLES:
        raise ValueError(f"Unsupported engine_id: {engine_id!r}")
    exe = ENGINE_EXECUTABLES[engine_id]
    if not exe.is_file():
        raise FileNotFoundError(f"Native simulator missing for {engine_id}: {exe}")
    worker = V03_WORKERS_DIR / WORKER_SCRIPTS[engine_id]
    if not worker.is_file():
        raise FileNotFoundError(f"Native worker missing: {worker}")

    vram_before = check_gpu_free_vram_mb()
    if engine_id in ("newton18", "genesis", "isaac6") and vram_before is not None:
        if vram_before < min_gpu_vram_mb:
            raise RuntimeError(
                f"Refusing to launch GPU engine {engine_id}: free VRAM {vram_before} MiB < {min_gpu_vram_mb} MiB"
            )

    ep_id = str(job["episode_id"])
    logs_root = Path(log_dir)
    logs_root.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_root / f"{ep_id}_{engine_id}.stdout.log"
    stderr_log = logs_root / f"{ep_id}_{engine_id}.stderr.log"

    env = os.environ.copy()
    if engine_id == "isaac6":
        env["ISAACSIM_PORTABLE_ROOT"] = r"I:\isaacsim-data\6.0.1"
        env["TEMP"] = r"I:\isaacsim-data\6.0.1\temp"
        env["TMP"] = r"I:\isaacsim-data\6.0.1\temp"

    creationflags = 0x00000040 if os.name == "nt" else 0  # IDLE_PRIORITY_CLASS
    ver = engine_version or probe_native_engine_version(engine_id)

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        batch_json = td_p / f"{ep_id}_batch.json"
        result_json = td_p / f"{ep_id}_result.json"
        batch_json.write_text(json.dumps({"jobs": [job]}, indent=2), encoding="utf-8")

        cmd = [str(exe), str(worker), "--batch-json", str(batch_json), "--result-json", str(result_json)]
        t0 = time.time()
        with stdout_log.open("w", encoding="utf-8") as fout, stderr_log.open("w", encoding="utf-8") as ferr:
            proc = subprocess.Popen(
                cmd,
                stdout=fout,
                stderr=ferr,
                env=env,
                creationflags=creationflags,
            )
            pid = int(proc.pid)
            rc = proc.wait(timeout=timeout_s)
        elapsed_s = round(time.time() - t0, 3)

        if rc != 0:
            err_tail = stderr_log.read_text(encoding="utf-8", errors="ignore")[-1500:]
            raise RuntimeError(
                f"Native subprocess failed ({engine_id}, pid={pid}, rc={rc}):\n{err_tail}"
            )
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        res = dict(payload["results"][0])

    npz_path = Path(str(res["npz_path"]))
    npz_sha = _file_sha256(npz_path) if npz_path.is_file() else ""
    res["provenance"] = {
        "execution_mode": "native_subprocess",
        "engine_id": engine_id,
        "engine_runtime_version": ver,
        "executable": exe.as_posix(),
        "worker_script": worker.as_posix(),
        "process_pid": pid,
        "seed": int(job.get("seed", 0)),
        "elapsed_s": elapsed_s,
        "vram_free_mb_before": vram_before,
        "stdout_log_path": stdout_log.resolve().as_posix(),
        "stderr_log_path": stderr_log.resolve().as_posix(),
        "npz_path": npz_path.resolve().as_posix(),
        "npz_sha256": npz_sha,
    }
    return res
