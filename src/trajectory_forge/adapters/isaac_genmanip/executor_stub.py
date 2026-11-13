from __future__ import annotations

from pathlib import Path
import subprocess

from trajectory_forge.adapters.isaac_genmanip.executor import (
    IsaacGenManipWorkerConfig,
    LocalIsaacGenManipExecutor,
)
from trajectory_forge.core.contracts import RolloutJobRequest, RolloutJobResult


class IsaacGenManipSubprocessStub:
    """Backward-compatible wrapper delegating live physics rollouts to LocalIsaacGenManipExecutor."""

    def __init__(self, config: IsaacGenManipWorkerConfig) -> None:
        self.config = config
        self._executor = LocalIsaacGenManipExecutor(config)

    def build_command(self, request_json: Path, result_json: Path) -> list[str]:
        return self._executor.build_command(request_json, result_json)

    def execute(self, request: RolloutJobRequest) -> RolloutJobResult:
        return self._executor.execute(request)

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
