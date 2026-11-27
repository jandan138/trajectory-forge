"""Isolated Cold-Start Subprocess Dispatcher for distillation_v07 True-Dynamic 3-Mechanism Slice.

Follows the `deformable_v05.1` zero-compromise provenance contract:
- Every single rollout (`nominal` and `zero_friction` counterfactual) runs in its OWN fresh OS subprocess
  (`I:\\isaacsim\\6.0.1\\python.bat`) at `IDLE_PRIORITY_CLASS` after verifying free GPU VRAM via `nvidia-smi`.
- Records real `process_pid`, `engine_runtime_version`, `stdout_log_path`, `stderr_log_path`,
  `npz_path`, `npz_sha256`, and synchronized 3-camera MP4 paths (`Front + 90-deg Ortho Side + Macro`).
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any


ISAAC6_PYTHON_BAT = Path(r"I:\isaacsim\6.0.1\python.bat")
WORKER_SCRIPT = Path(__file__).resolve().parent / "isaac6_v07_full_dynamic_worker.py"

V07_MECHANISM_SLICE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "mech_id": "mech_01_scissor_jack_handwheel_lift",
        "title_zh": "核心机构 01 · 蓝色升降台蝶形手轮 3×120° 换把旋拧顶升 1.85 kg 集热式热浴",
        "bilibili_ref": "BV17t4y1V7XH 02:27-02:40 (S01 升降台定标)",
        "has_zero_friction_counterfactual": True,
    },
    {
        "mech_id": "mech_02_bosshead_flask_clamp_multiturn",
        "title_zh": "核心机构 02 · 纯摩擦抓提圆底烧瓶入水浴 + 双手协同 2×120° 拧紧烧瓶夹 + 松爪悬空抗重力验证",
        "bilibili_ref": "BV17t4y1V7XH 02:56-03:18 (S02-S03 十字夹开口向上与烧瓶夹紧)",
        "has_zero_friction_counterfactual": True,
    },
    {
        "mech_id": "mech_03_3prong_clamp_condenser_native_hoses",
        "title_zh": "核心机构 03 · 带双路 PhysX 动态软管的直形冷凝管 15° 摩擦托举 + 下固上动三指夹 2×120° 锁紧悬空验证",
        "bilibili_ref": "BV17t4y1V7XH 03:32-04:13 (S05-S06 三指夹下固上动与乳胶管悬链)",
        "has_zero_friction_counterfactual": True,
    },
)


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check_gpu_free_vram_mb() -> int | None:
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


def probe_isaac6_version() -> str:
    ver_file = ISAAC6_PYTHON_BAT.parent / "VERSION"
    if ver_file.is_file():
        return f"isaacsim=={ver_file.read_text(encoding='utf-8').strip().splitlines()[0]}"
    return "isaacsim==6.0.1"


def run_v07_cold_start_episode(
    *,
    package_dir: Path,
    mech_id: str,
    zero_friction: bool,
    out_dir: Path,
    record_video: bool = True,
    min_gpu_vram_mb: int = 7500,
    timeout_s: int = 600,
) -> dict[str, Any]:
    """Spawn ONE fresh cold-start Isaac Sim 6.0.1 OS subprocess for ONE mechanism rollout."""
    if not ISAAC6_PYTHON_BAT.is_file():
        raise FileNotFoundError(f"Missing Isaac Sim 6.0.1 runtime: {ISAAC6_PYTHON_BAT}")
    if not WORKER_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing v0.7 dynamic worker: {WORKER_SCRIPT}")

    vram_before = check_gpu_free_vram_mb()
    if vram_before is not None and vram_before < min_gpu_vram_mb:
        raise RuntimeError(
            f"Refusing to launch Isaac Sim 6.0.1 GPU subprocess: free VRAM {vram_before} MiB < {min_gpu_vram_mb} MiB"
        )

    mode_tag = "zero_friction" if zero_friction else "nominal"
    episode_id = f"{mech_id}__{mode_tag}"
    ep_out = (out_dir / episode_id).resolve()
    ep_out.mkdir(parents=True, exist_ok=True)
    logs_dir = (out_dir / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_dir / f"{episode_id}.stdout.log"
    stderr_log = logs_dir / f"{episode_id}.stderr.log"

    env = os.environ.copy()
    env["ISAACSIM_PORTABLE_ROOT"] = r"I:\isaacsim-data\6.0.1"
    env["TEMP"] = r"I:\isaacsim-data\6.0.1\temp"
    env["TMP"] = r"I:\isaacsim-data\6.0.1\temp"
    src_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = src_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    creationflags = 0x00000040 if os.name == "nt" else 0  # IDLE_PRIORITY_CLASS
    ver = probe_isaac6_version()

    job = {
        "episode_id": episode_id,
        "mech_id": mech_id,
        "zero_friction": bool(zero_friction),
        "package_dir": package_dir.resolve().as_posix(),
        "output_dir": ep_out.as_posix(),
        "record_video": bool(record_video),
    }

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        job_json = td_p / f"{episode_id}_job.json"
        res_json = td_p / f"{episode_id}_result.json"
        job_json.write_text(json.dumps(job, indent=2), encoding="utf-8")

        cmd = [str(ISAAC6_PYTHON_BAT), str(WORKER_SCRIPT), "--job-json", str(job_json), "--result-json", str(res_json)]
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
            err_tail = stderr_log.read_text(encoding="utf-8", errors="ignore")[-2000:]
            out_tail = stdout_log.read_text(encoding="utf-8", errors="ignore")[-2000:]
            raise RuntimeError(
                f"v0.7 cold-start subprocess failed ({episode_id}, pid={pid}, rc={rc}):\nSTDOUT:\n{out_tail}\nSTDERR:\n{err_tail}"
            )
        res = json.loads(res_json.read_text(encoding="utf-8"))

    npz_p = Path(str(res["telemetry_npz"]))
    res["provenance"] = {
        "execution_mode": "native_cold_start_subprocess",
        "engine_id": "isaac6_physx5_gpu_240hz",
        "engine_runtime_version": ver,
        "executable": ISAAC6_PYTHON_BAT.as_posix(),
        "worker_script": WORKER_SCRIPT.as_posix(),
        "process_pid": pid,
        "elapsed_s": elapsed_s,
        "vram_free_mb_before": vram_before,
        "stdout_log_path": stdout_log.as_posix(),
        "stderr_log_path": stderr_log.as_posix(),
        "npz_path": npz_p.as_posix(),
        "npz_sha256": _file_sha256(npz_p),
    }
    (ep_out / "episode_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def run_v07_full_3mech_slice_with_counterfactuals(
    *,
    package_dir: Path,
    out_dir: Path,
    record_video: bool = True,
) -> dict[str, Any]:
    """Execute all 3 core mechanisms (`nominal` + `zero_friction` counterfactuals = 6 isolated cold-start PIDs!)."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes: dict[str, Any] = {}
    pids: list[int] = []

    for mspec in V07_MECHANISM_SLICE_SPECS:
        mid = mspec["mech_id"]
        nom = run_v07_cold_start_episode(
            package_dir=package_dir,
            mech_id=mid,
            zero_friction=False,
            out_dir=out_dir,
            record_video=record_video,
        )
        zf = run_v07_cold_start_episode(
            package_dir=package_dir,
            mech_id=mid,
            zero_friction=True,
            out_dir=out_dir,
            record_video=record_video,
        )
        pids.extend([int(nom["provenance"]["process_pid"]), int(zf["provenance"]["process_pid"])])
        episodes[mid] = {
            "spec": mspec,
            "nominal": nom,
            "zero_friction": zf,
            "counterfactual_falsified": bool(nom["metrics"]["task_passed"] and (not zf["metrics"]["task_passed"])),
        }

    all_passed = all(
        ep["nominal"]["metrics"]["task_passed"]
        and (not ep["nominal"]["metrics"]["any_kinematic_manipulated"])
        and ep["counterfactual_falsified"]
        for ep in episodes.values()
    )
    summary = {
        "schema_version": "distillation-v07-3mech-slice-summary/v1.0",
        "all_passed": all_passed,
        "distinct_cold_start_pids": len(set(pids)),
        "all_pids": pids,
        "episodes": episodes,
    }
    (out_dir / "v07_slice_master_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

