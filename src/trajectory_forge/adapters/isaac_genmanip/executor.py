from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from trajectory_forge.core.contracts import (
    BatchSynthesisReport,
    BatchSynthesisRequest,
    ContractValidationError,
    RolloutJobRequest,
    RolloutJobResult,
    SolvabilityDiagnostics,
)


DEFAULT_ISAAC_PYTHON = Path(
    "/cpfs/shared/simulation/zhuzihou/dev/conda-managed/envs/"
    "embodied-eval-os-sim-isaacsim41-genmanip-py310/bin/python"
)
DEFAULT_GENMANIP_ROOT = Path("/cpfs/shared/simulation/zhuzihou/dev/GenManip")
DEFAULT_CUROBO_SRC = Path("/cpfs/shared/simulation/mamengchen/curobo-wbc-backup/src")
DEFAULT_PACKAGE_STORE_ROOT = Path(
    "/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/"
    "out_of_domain_factory_r3_20261113/packages"
)


@dataclass(frozen=True)
class IsaacGenManipWorkerConfig:
    """Subprocess boundary configuration for the Isaac Sim 4.1 + GenManip worker."""

    isaac_python: Path = DEFAULT_ISAAC_PYTHON
    genmanip_root: Path = DEFAULT_GENMANIP_ROOT
    package_store_root: Path = DEFAULT_PACKAGE_STORE_ROOT
    output_root: Path = Path("external_artifacts/rollouts")
    curobo_src: Path = DEFAULT_CUROBO_SRC
    timeout_s: float = 900.0


def build_isaac_worker_env(
    *,
    isaac_python: Path,
    trajectory_forge_src: Path,
    genmanip_root: Path,
    curobo_src: Path | None = None,
    omni_scratch_dir: Path | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Construct environment variables matching the Isaac Sim 4.1 + CuRobo runtime."""
    env = dict(os.environ if base_env is None else base_env)
    py_parts = [str(trajectory_forge_src.resolve()), str(genmanip_root.resolve())]
    if curobo_src is not None and curobo_src.is_dir():
        py_parts.insert(0, str(curobo_src.resolve()))
    inherited_py = env.get("PYTHONPATH", "")
    if inherited_py:
        py_parts.append(inherited_py)
    env["PYTHONPATH"] = os.pathsep.join(py_parts)

    env["PATH"] = os.pathsep.join([str(isaac_python.parent), env.get("PATH", "")])
    prefix = isaac_python.parent.parent
    libs = (
        prefix / "lib/python3.10/site-packages/torch/lib",
        prefix / "lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
        prefix / "lib/python3.10/site-packages/isaacsim/extscache/omni.cuda.libs/bin",
    )
    inherited_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            *[str(p) for p in libs if p.is_dir()],
            *([inherited_ld] if inherited_ld else []),
        ]
    )
    if omni_scratch_dir is not None:
        env["OMNI_USER_DATA_DIR"] = str(omni_scratch_dir / "omni_data")
    raw_gpu = (env.get("CE_SIM_GPU_ID") or env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if raw_gpu.isdigit():
        env["CE_SIM_GPU_ID"] = raw_gpu
        env.pop("CUDA_VISIBLE_DEVICES", None)
    gpu_tag = env.get("CE_SIM_GPU_ID", "0")
    shared_cache = Path(f"/tmp/tf_omni_shared_cache_gpu{gpu_tag}")
    env["ISAACSIM_CACHE_DIR"] = str(shared_cache / "isaac")
    env["XDG_CACHE_HOME"] = str(shared_cache / "xdg")
    return env


class LocalIsaacGenManipExecutor:
    """Runs RolloutJobRequest inside an isolated local Isaac Sim 4.1 subprocess."""

    def __init__(self, config: IsaacGenManipWorkerConfig) -> None:
        self.config = config

    @staticmethod
    def _repo_src() -> Path:
        return Path(__file__).resolve().parents[3]

    def _build_entrypoint_command(
        self,
        module_name: str,
        request_json: Path,
        result_json: Path,
        *,
        output_root: Path | None = None,
    ) -> list[str]:
        out_root = (output_root or self.config.output_root).resolve()
        return [
            str(self.config.isaac_python),
            "-u",
            "-m",
            module_name,
            "--request",
            str(request_json),
            "--result",
            str(result_json),
            "--genmanip-root",
            str(self.config.genmanip_root),
            "--package-store",
            str(self.config.package_store_root),
            "--output-root",
            str(out_root),
        ]

    def build_command(
        self,
        request_json: Path,
        result_json: Path,
        *,
        output_root: Path | None = None,
    ) -> list[str]:
        return self._build_entrypoint_command(
            "trajectory_forge.adapters.isaac_genmanip.runner_entry",
            request_json,
            result_json,
            output_root=output_root,
        )

    def build_batch_command(
        self,
        request_json: Path,
        result_json: Path,
        *,
        output_root: Path | None = None,
    ) -> list[str]:
        return self._build_entrypoint_command(
            "trajectory_forge.adapters.isaac_genmanip.batch_runner_entry",
            request_json,
            result_json,
            output_root=output_root,
        )

    def execute_batch(self, request: BatchSynthesisRequest) -> BatchSynthesisReport:
        """Execute a warm multi-episode trajectory synthesis batch inside one Isaac Sim worker."""
        out_root = self.config.output_root.resolve()
        batch_dir = out_root / "batches" / request.batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        request_json = batch_dir / "batch_request.json"
        result_json = batch_dir / "batch_report.json"
        log_path = batch_dir / "batch_worker.log"
        request_json.write_text(
            json.dumps(request.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        omni_scratch = Path(tempfile.mkdtemp(prefix=f"tf_batch_{request.batch_id}_"))
        env = build_isaac_worker_env(
            isaac_python=self.config.isaac_python,
            trajectory_forge_src=self._repo_src(),
            genmanip_root=self.config.genmanip_root,
            curobo_src=self.config.curobo_src,
            omni_scratch_dir=omni_scratch,
        )
        cmd = self.build_batch_command(request_json, result_json, output_root=out_root)
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=self.config.timeout_s,
            )
            log_path.write_text(
                f"=== STDOUT ===\n{proc.stdout}\n=== STDERR ===\n{proc.stderr}\n",
                encoding="utf-8",
            )
            if result_json.exists():
                payload = json.loads(result_json.read_text(encoding="utf-8"))
                return BatchSynthesisReport.from_dict(payload)
            raise ContractValidationError(
                f"Warm batch runner exited rc={proc.returncode} without writing {result_json}; "
                f"see {log_path}"
            )
        except subprocess.TimeoutExpired as exc:
            raise ContractValidationError(
                f"Isaac Sim batch {request.batch_id!r} timed out after {self.config.timeout_s}s"
            ) from exc
        finally:
            shutil.rmtree(omni_scratch, ignore_errors=True)

    def execute(self, request: RolloutJobRequest) -> RolloutJobResult:
        out_root = self.config.output_root.resolve()
        job_dir = out_root / "jobs" / request.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        request_json = job_dir / "request.json"
        result_json = job_dir / "result.json"
        log_path = job_dir / "worker.log"
        request_json.write_text(
            json.dumps(request.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        omni_scratch = Path(tempfile.mkdtemp(prefix=f"tf_omni_{request.job_id}_"))
        env = build_isaac_worker_env(
            isaac_python=self.config.isaac_python,
            trajectory_forge_src=self._repo_src(),
            genmanip_root=self.config.genmanip_root,
            curobo_src=self.config.curobo_src,
            omni_scratch_dir=omni_scratch,
        )
        cmd = self.build_command(request_json, result_json, output_root=out_root)
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=self.config.timeout_s,
            )
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
                    failure_code="RUNNER_NO_RESULT",
                    detail=(
                        f"Isaac Sim runner exited rc={proc.returncode} without writing result.json; "
                        f"see {log_path}"
                    ),
                ),
                state_summary={"worker_log": str(log_path), "returncode": proc.returncode},
            )
        except subprocess.TimeoutExpired as exc:
            raise ContractValidationError(
                f"Isaac Sim rollout {request.job_id!r} timed out after {self.config.timeout_s}s"
            ) from exc
        finally:
            shutil.rmtree(omni_scratch, ignore_errors=True)

    __call__ = execute
