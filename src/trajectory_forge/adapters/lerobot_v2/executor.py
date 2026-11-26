from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from trajectory_forge.core.contracts import ContractValidationError
from trajectory_forge.trajectory.lerobot_converter import (
    LeRobotEpisodeInputSpec,
    write_lerobot_staging_manifest,
)


DEFAULT_LEROBOT_PYTHON = Path("/mnt/data/zouyude/openpi/.venv/bin/python")


@dataclass(frozen=True)
class LeRobotExportConfig:
    """Subprocess configuration for writing LeRobot v2.1 datasets."""

    lerobot_python: Path = DEFAULT_LEROBOT_PYTHON
    fps: int = 15
    robot_type: str = "lift2"
    repo_id: str | None = None
    overwrite: bool = False
    timeout_s: float = 1800.0


@dataclass(frozen=True)
class LeRobotExportResult:
    """Summary returned after LeRobot v2.1 export finishes."""

    repo_id: str
    output_dir: Path
    fps: int
    total_episodes: int
    total_frames: int


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[3]


def build_lerobot_export_env(
    *,
    trajectory_forge_src: Path,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    src_str = str(trajectory_forge_src.resolve())
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_str}{os.pathsep}{inherited}" if inherited else src_str
    )
    return env


def build_lerobot_export_command(
    *,
    lerobot_python: Path,
    staging_manifest: Path,
    output_dir: Path,
    summary_out: Path,
    config: LeRobotExportConfig,
) -> list[str]:
    cmd = [
        str(lerobot_python),
        "-u",
        "-m",
        "trajectory_forge.adapters.lerobot_v2.writer_entry",
        "--staging-manifest",
        str(staging_manifest.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--fps",
        str(int(config.fps)),
        "--robot-type",
        config.robot_type,
        "--summary-out",
        str(summary_out.resolve()),
    ]
    if config.repo_id:
        cmd.extend(["--repo-id", config.repo_id])
    if config.overwrite:
        cmd.append("--overwrite")
    return cmd


def export_lerobot_dataset(
    *,
    episodes: Sequence[LeRobotEpisodeInputSpec],
    output_dir: Path,
    config: LeRobotExportConfig | None = None,
    staging_dir: Path | None = None,
) -> LeRobotExportResult:
    """Validate and export verified episodes to a LeRobot v2.1 dataset in an isolated venv."""
    cfg = config or LeRobotExportConfig()
    if not cfg.lerobot_python.is_file():
        raise ContractValidationError(
            f"LeRobot Python interpreter not found: {cfg.lerobot_python}"
        )
    resolved_out = output_dir.resolve()
    work_dir = staging_dir or (resolved_out.parent / f".{resolved_out.name}_staging")
    work_dir.mkdir(parents=True, exist_ok=True)
    staging_manifest = write_lerobot_staging_manifest(
        work_dir / "staging_episodes.json", episodes
    )
    summary_out = work_dir / "export_summary.json"
    cmd = build_lerobot_export_command(
        lerobot_python=cfg.lerobot_python,
        staging_manifest=staging_manifest,
        output_dir=resolved_out,
        summary_out=summary_out,
        config=cfg,
    )
    env = build_lerobot_export_env(trajectory_forge_src=_repo_src())
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=cfg.timeout_s,
    )
    if proc.returncode != 0 or not summary_out.is_file():
        raise ContractValidationError(
            f"LeRobot v2.1 export failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    data = json.loads(summary_out.read_text(encoding="utf-8"))
    return LeRobotExportResult(
        repo_id=str(data["repo_id"]),
        output_dir=Path(data["output_dir"]),
        fps=int(data["fps"]),
        total_episodes=int(data["total_episodes"]),
        total_frames=int(data["total_frames"]),
    )
