from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess

from trajectory_forge.adapters.isaac_genmanip.executor import (
    DEFAULT_CUROBO_SRC,
    DEFAULT_GENMANIP_ROOT,
    DEFAULT_ISAAC_PYTHON,
    DEFAULT_PACKAGE_STORE_ROOT,
)
from trajectory_forge.core.contracts import (
    ContractValidationError,
    RolloutJobRequest,
    RolloutJobResult,
    SolvabilityDiagnostics,
)


@dataclass(frozen=True)
class SSHWorkerConfig:
    """SSH dispatch configuration for executing rollouts on a GPU node sharing /cpfs."""

    ssh_host: str = "10.150.168.71"
    ssh_user: str = "root"
    ssh_port: int = 22
    ssh_key_path: Path = Path("~/.ssh/id_ed25519_github").expanduser()
    remote_isaac_python: Path = DEFAULT_ISAAC_PYTHON
    remote_genmanip_root: Path = DEFAULT_GENMANIP_ROOT
    remote_curobo_src: Path = DEFAULT_CUROBO_SRC
    remote_repo_root: Path = Path("/cpfs/user/zhuzihou/dev/trajectory-forge")
    package_store_root: Path = DEFAULT_PACKAGE_STORE_ROOT
    output_root: Path = Path("/cpfs/user/zhuzihou/dev/trajectory-forge/external_artifacts/rollouts")
    timeout_s: float = 900.0


class SSHIsaacGenManipExecutor:
    """Executes RolloutJobRequest via SSH on a remote RTX 4090 host sharing `/cpfs`."""

    def __init__(self, config: SSHWorkerConfig) -> None:
        self.config = config

    def build_remote_shell_command(self, request_json: Path, result_json: Path, job_id: str) -> str:
        out_root = self.config.output_root.resolve()
        prefix = self.config.remote_isaac_python.parent.parent
        libs = ":".join(
            [
                str(prefix / "lib/python3.10/site-packages/torch/lib"),
                str(prefix / "lib/python3.10/site-packages/nvidia/cuda_runtime/lib"),
                str(prefix / "lib/python3.10/site-packages/isaacsim/extscache/omni.cuda.libs/bin"),
            ]
        )
        py_path = ":".join(
            [
                str(self.config.remote_curobo_src),
                str(self.config.remote_repo_root / "src"),
                str(self.config.remote_genmanip_root),
            ]
        )
        scratch = f"/tmp/tf_omni_job_{job_id}"
        shared_cache = "/tmp/tf_omni_shared_cache"
        runner_cmd = " ".join(
            shlex.quote(arg)
            for arg in (
                str(self.config.remote_isaac_python),
                "-B",
                "-u",
                "-m",
                "trajectory_forge.adapters.isaac_genmanip.runner_entry",
                "--request",
                str(request_json),
                "--result",
                str(result_json),
                "--genmanip-root",
                str(self.config.remote_genmanip_root),
                "--package-store",
                str(self.config.package_store_root),
                "--output-root",
                str(out_root),
            )
        )
        adapter_dir = self.config.remote_repo_root / "src/trajectory_forge/adapters/isaac_genmanip"
        core_dir = self.config.remote_repo_root / "src/trajectory_forge/core"
        solver_dir = self.config.remote_repo_root / "src/trajectory_forge/solver"
        intake_dir = self.config.remote_repo_root / "src/trajectory_forge/intake"
        return (
            f"ls -la {shlex.quote(str(adapter_dir))} {shlex.quote(str(core_dir))} {shlex.quote(str(solver_dir))} {shlex.quote(str(intake_dir))} >/dev/null 2>&1 || true; "
            f"mkdir -p {shlex.quote(scratch)} {shared_cache}/isaac {shared_cache}/xdg; "
            f"trap 'rm -rf {shlex.quote(scratch)}' EXIT; "
            f"PYTHONDONTWRITEBYTECODE=1 "
            f"OMNI_USER_DATA_DIR={shlex.quote(scratch + '/omni')} "
            f"ISAACSIM_CACHE_DIR={shared_cache}/isaac "
            f"XDG_CACHE_HOME={shared_cache}/xdg "
            f"PYTHONPATH={shlex.quote(py_path)} "
            f"LD_LIBRARY_PATH={shlex.quote(libs)}:${{LD_LIBRARY_PATH:-}} "
            f"{runner_cmd}"
        )

    def build_ssh_command(self, remote_shell_cmd: str) -> list[str]:
        return [
            "ssh",
            "-i",
            str(self.config.ssh_key_path.expanduser()),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            str(self.config.ssh_port),
            f"{self.config.ssh_user}@{self.config.ssh_host}",
            remote_shell_cmd,
        ]

    def execute(self, request: RolloutJobRequest) -> RolloutJobResult:
        out_root = self.config.output_root.resolve()
        job_dir = out_root / "jobs" / request.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        request_json = job_dir / "request.json"
        result_json = job_dir / "result.json"
        log_path = job_dir / "remote_worker.log"
        if result_json.exists():
            result_json.unlink()
        request_json.write_text(
            json.dumps(request.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        remote_cmd = self.build_remote_shell_command(request_json, result_json, request.job_id)
        ssh_cmd = self.build_ssh_command(remote_cmd)
        try:
            proc = subprocess.run(
                ssh_cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise ContractValidationError(
                f"Remote SSH rollout {request.job_id!r} timed out after {self.config.timeout_s}s"
            ) from exc

        log_path.write_text(
            f"=== STDOUT ===\n{proc.stdout}\n=== STDERR ===\n{proc.stderr}\n",
            encoding="utf-8",
        )
        if result_json.exists():
            payload = json.loads(result_json.read_text(encoding="utf-8"))
            return RolloutJobResult.from_dict(payload)
        return RolloutJobResult(
            job_id=request.job_id,
            status="failed",
            mode=request.mode,
            final_score=0.0,
            steps_executed=0,
            snapshots_used=0,
            firewall_verified=False,
            diagnostics=SolvabilityDiagnostics(
                failure_code="REMOTE_RUNNER_NO_RESULT",
                detail=(
                    f"SSH runner on {self.config.ssh_host} exited rc={proc.returncode} "
                    f"without writing {result_json}; see {log_path}"
                ),
            ),
            state_summary={"worker_log": str(log_path), "returncode": proc.returncode},
        )

    __call__ = execute
