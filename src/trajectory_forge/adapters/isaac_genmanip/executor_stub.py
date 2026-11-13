from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from trajectory_forge.core.contracts import RolloutJobRequest, RolloutJobResult


@dataclass(frozen=True)
class IsaacGenManipWorkerConfig:
    """Subprocess boundary configuration for the local RTX 4090 Isaac Sim 4.1 worker."""

    isaac_python: Path
    genmanip_root: Path
    package_store_root: Path
    output_root: Path


class IsaacGenManipSubprocessStub:
    """Adapter boundary delegating live physics rollouts to the Isaac Sim Python runtime.

    Pure layers never import `isaacsim`, `omni.*`, or `pxr`; instead, the worker invokes
    an isolated Isaac Sim subprocess using `IsaacGenManipWorkerConfig`.
    """

    def __init__(self, config: IsaacGenManipWorkerConfig) -> None:
        self.config = config

    def build_command(self, request_json: Path, result_json: Path) -> list[str]:
        return [
            str(self.config.isaac_python),
            "-m",
            "trajectory_forge.adapters.isaac_genmanip.runner_entry",
            "--request",
            str(request_json),
            "--result",
            str(result_json),
            "--genmanip-root",
            str(self.config.genmanip_root),
            "--package-store",
            str(self.config.package_store_root),
        ]

    def execute(self, request: RolloutJobRequest) -> RolloutJobResult:
        raise NotImplementedError(
            "Live Isaac Sim 4.1 + GenManip runner_entry is wired on the RTX 4090 worker host "
            f"using {self.config.isaac_python} (subprocess boundary ready)."
        )

    @staticmethod
    def check_isaac_python_present(isaac_python: Path) -> bool:
        if not isaac_python.exists():
            return False
        proc = subprocess.run(
            [str(isaac_python), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0
