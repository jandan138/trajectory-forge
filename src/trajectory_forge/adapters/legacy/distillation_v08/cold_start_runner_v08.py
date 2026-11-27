"""Isolated Cold-Start Subprocess Dispatcher for distillation_v08:
1. `8` Cold-Start PIDs for the 4 Standard-Taper Ground-Glass Coaxial Insertion & Keck Clip Slices (`nominal` + `zero_friction`)
2. `2` Cold-Start PIDs for Method A (`Rigid Funnel`) vs Method B (`4-Capsule/Sphere Micro-Spring Wedge`) on `-15°` Unclamped Slope
3. `3` Cold-Start PIDs for 3-Way Bidirectional Pull-Out / Disassembly (`Keck-Locked` vs `Method B Straight Pull` vs `Method B Twist-and-Pull`)
Total = `13` isolated cold-start OS subprocesses (`13` distinct PIDs, `13` SHA-256 telemetry archives, `39` synchronized RTX MP4 streams).
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
WORKER_SCRIPT = Path(__file__).resolve().parent / "isaac6_v08_taper_keck_worker.py"

V08_TAPER_JOINT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "joint_id": "joint_01_head_into_flask_24_29_vertical",
        "title_zh": "磨口插接 01 · 75° 蒸馏头下公口 → 圆底烧瓶 24/29 母口垂直同轴推进（16 瓣导槽 + 4 颗微弹簧箍紧自锁）",
        "bilibili_ref": "BV17t4y1V7XH 03:19-03:31 (S04 蒸馏头磨口安装)",
    },
    {
        "joint_id": "joint_02_condenser_to_sidearm_15deg_keck",
        "title_zh": "磨口插接 02 · 直形冷凝管母口 → 蒸馏头 75° 支管公口 15° 斜向同轴推进 + 三指夹锁紧 + 三连杆绿夹卡扣",
        "bilibili_ref": "BV17t4y1V7XH 03:32-04:19 (S05 冷凝管磨口对接与绿色塑料夹)",
    },
    {
        "joint_id": "joint_03_tail_adapter_to_condenser_keck",
        "title_zh": "磨口插接 03 · 牛角接引管母口 → 冷凝管下公口 15° 斜向同轴套合 + 三连杆绿夹锁扣悬空抗滑脱验证",
        "bilibili_ref": "BV17t4y1V7XH 04:48-05:06 (S07 接引管对接与标准口夹防脱落)",
    },
    {
        "joint_id": "joint_04_thermometer_screw_cap_flush_bulb",
        "title_zh": "磨口插接 04 · 螺帽温度计 19/26 垂直同轴穿管插合 + 水银球上沿与支管口下沿 0.0 mm 平齐红线验证",
        "bilibili_ref": "BV17t4y1V7XH 05:07-05:26 (S09 温度计水银球平齐核心考点)",
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


def run_v08_cold_start_episode(
    *,
    package_dir: Path,
    joint_id: str,
    zero_friction: bool = False,
    variant: str = "",
    out_dir: Path,
    record_video: bool = True,
    min_gpu_vram_mb: int = 7500,
    timeout_s: int = 600,
) -> dict[str, Any]:
    if not ISAAC6_PYTHON_BAT.is_file():
        raise FileNotFoundError(f"Missing Isaac Sim 6.0.1 runtime: {ISAAC6_PYTHON_BAT}")
    if not WORKER_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing v0.8 dynamic worker: {WORKER_SCRIPT}")

    vram_before = check_gpu_free_vram_mb()
    if vram_before is not None and vram_before < min_gpu_vram_mb:
        raise RuntimeError(f"Insufficient free GPU VRAM: {vram_before} MiB < {min_gpu_vram_mb} MiB")

    eff_variant = variant if variant else ("zero_friction" if zero_friction else "nominal")
    episode_id = f"{joint_id}__{eff_variant}"
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

    creationflags = 0x00000040 if os.name == "nt" else 0
    ver = probe_isaac6_version()

    job = {
        "episode_id": episode_id,
        "joint_id": joint_id,
        "variant": eff_variant,
        "zero_friction": bool(zero_friction or eff_variant == "zero_friction"),
        "record_video": bool(record_video),
        "package_dir": str(package_dir.resolve()),
        "output_dir": str(ep_out),
    }

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        job_json = td_p / "job.json"
        res_json = td_p / "result.json"
        job_json.write_text(json.dumps(job, indent=2), encoding="utf-8")

        t0 = time.perf_counter()
        with stdout_log.open("w", encoding="utf-8") as fo, stderr_log.open("w", encoding="utf-8") as fe:
            proc = subprocess.Popen(
                [str(ISAAC6_PYTHON_BAT), str(WORKER_SCRIPT), "--job-json", str(job_json), "--result-json", str(res_json)],
                stdout=fo,
                stderr=fe,
                env=env,
                creationflags=creationflags,
            )
            pid = int(proc.pid)
            rc = proc.wait(timeout=timeout_s)
        elapsed_s = round(time.perf_counter() - t0, 3)

        if rc != 0 and not res_json.is_file():
            err_tail = stderr_log.read_text(encoding="utf-8", errors="replace")[-1600:]
            raise RuntimeError(f"v0.8 worker failed (rc={rc}, pid={pid}) for {episode_id}:\n{err_tail}")

        res = json.loads(res_json.read_text(encoding="utf-8"))

    npz_p = Path(res["telemetry_npz"])
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
    (ep_out / "episode_provenance.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def run_v08_full_4joint_slice_with_counterfactuals(
    *,
    package_dir: Path,
    out_dir: Path,
    record_video: bool = True,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes: dict[str, Any] = {}
    all_pids: list[int] = []
    all_passed = True

    # Part 1: 4 Standard-Taper Insertion & Keck-Clip Slices (`8` Cold-Start PIDs)
    for spec in V08_TAPER_JOINT_SPECS:
        jid = spec["joint_id"]
        nom = run_v08_cold_start_episode(
            package_dir=package_dir,
            joint_id=jid,
            zero_friction=False,
            variant="nominal",
            out_dir=out_dir,
            record_video=record_video,
        )
        zf = run_v08_cold_start_episode(
            package_dir=package_dir,
            joint_id=jid,
            zero_friction=True,
            variant="zero_friction",
            out_dir=out_dir,
            record_video=record_video,
        )
        all_pids.extend([int(nom["provenance"]["process_pid"]), int(zf["provenance"]["process_pid"])])
        cf_ok = (nom["metrics"]["task_passed"] is True) and (zf["metrics"]["task_passed"] is False)
        if not cf_ok:
            all_passed = False
        episodes[jid] = {
            "spec": spec,
            "nominal": nom,
            "zero_friction": zf,
            "counterfactual_falsified": cf_ok,
        }

    # Part 2: Method A vs Method B on Unclamped `-15°` Downward Taper (`2` Cold-Start PIDs)
    ab05_a = run_v08_cold_start_episode(
        package_dir=package_dir,
        joint_id="ab_05_unclamped_15deg_hold",
        variant="method_a_rigid",
        out_dir=out_dir,
        record_video=record_video,
    )
    ab05_b = run_v08_cold_start_episode(
        package_dir=package_dir,
        joint_id="ab_05_unclamped_15deg_hold",
        variant="method_b_micro_spring",
        out_dir=out_dir,
        record_video=record_video,
    )
    all_pids.extend([int(ab05_a["provenance"]["process_pid"]), int(ab05_b["provenance"]["process_pid"])])
    ab05_ok = (
        (ab05_a["metrics"]["task_passed"] is False)
        and (ab05_a["metrics"]["final_taper_3d_err_mm"] >= 25.0)
        and (ab05_b["metrics"]["task_passed"] is True)
        and (ab05_b["metrics"]["peak_wedge_radial_force_n"] >= 6.0)
    )
    if not ab05_ok:
        all_passed = False

    # Part 3: 3-Way Bidirectional Pull-Out / Disassembly Comparison (`3` Cold-Start PIDs)
    ab06_keck = run_v08_cold_start_episode(
        package_dir=package_dir,
        joint_id="ab_06_disassembly_pullout",
        variant="keck_locked_pull",
        out_dir=out_dir,
        record_video=record_video,
    )
    ab06_straight = run_v08_cold_start_episode(
        package_dir=package_dir,
        joint_id="ab_06_disassembly_pullout",
        variant="method_b_straight_pull",
        out_dir=out_dir,
        record_video=record_video,
    )
    ab06_twist = run_v08_cold_start_episode(
        package_dir=package_dir,
        joint_id="ab_06_disassembly_pullout",
        variant="method_b_twist_and_pull",
        out_dir=out_dir,
        record_video=record_video,
    )
    all_pids.extend(
        [
            int(ab06_keck["provenance"]["process_pid"]),
            int(ab06_straight["provenance"]["process_pid"]),
            int(ab06_twist["provenance"]["process_pid"]),
        ]
    )
    f_straight = float(ab06_straight["metrics"]["peak_axial_pull_force_n"])
    f_twist = float(ab06_twist["metrics"]["peak_axial_pull_force_n"])
    ab06_ok = (
        (ab06_keck["metrics"]["task_passed"] is False)
        and (ab06_keck["metrics"]["max_extract_dz_mm"] <= 12.0)
        and (ab06_straight["metrics"]["task_passed"] is True)
        and (ab06_twist["metrics"]["task_passed"] is True)
        and (f_twist <= 0.80 * f_straight)
    )
    if not ab06_ok:
        all_passed = False

    summary = {
        "schema_version": "distillation-v08-4taper-ab-disassembly-summary/v2.0",
        "all_passed": all_passed and (len(set(all_pids)) == 13),
        "distinct_cold_start_pids": len(set(all_pids)),
        "all_pids": all_pids,
        "episodes": episodes,
        "ab_05_unclamped_15deg_comparison": {
            "passed": ab05_ok,
            "method_a_rigid": ab05_a,
            "method_b_micro_spring": ab05_b,
        },
        "ab_06_disassembly_comparison": {
            "passed": ab06_ok,
            "axial_force_reduction_pct": round((1.0 - f_twist / max(f_straight, 1e-6)) * 100.0, 2),
            "keck_locked_pull": ab06_keck,
            "method_b_straight_pull": ab06_straight,
            "method_b_twist_and_pull": ab06_twist,
        },
    }
    (out_dir / "v08_slice_master_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
