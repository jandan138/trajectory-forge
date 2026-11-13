from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
from pathlib import Path
from typing import Callable

from trajectory_forge.core.contracts import (
    ContractValidationError,
    ControllerSafetyError,
    FirewallViolationError,
    RobotSpawnSpec,
    RolloutJobRequest,
    RolloutJobResult,
    ScenarioIntakeBrief,
    SolvabilityDiagnostics,
    StageContractSpec,
)
from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.intake.genmanip_intake import load_genmanip_collected_package
from trajectory_forge.sdk.context import run_controller_preflight


RolloutExecutor = Callable[[RolloutJobRequest], RolloutJobResult]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class DryRunRolloutExecutor:
    """Default pure-Python worker executor for cloud/CI validation and RPC smoke tests."""

    def __init__(self, repo_root: Path | None = None, backend_name: str = "dry_run_preflight") -> None:
        self.repo_root = repo_root if repo_root is not None else _default_repo_root()
        self.backend_name = backend_name

    def _resolve_brief(self, req: RolloutJobRequest) -> ScenarioIntakeBrief:
        ref = Path(req.scenario_package_ref)
        if not ref.is_absolute():
            candidate = self.repo_root / ref
            if candidate.exists():
                ref = candidate
        if ref.is_dir() and (
            (ref / "package_manifest.json").exists()
            or (ref / "collected_package" / "package_manifest.json").exists()
        ):
            return load_genmanip_collected_package(ref)
        return ScenarioIntakeBrief(
            package_id=ref.stem or "dry_run_task",
            episode_name=str(req.seed),
            instruction="Dry-run preflight scenario brief.",
            scene_usd_relpath="",
            robot_profile=req.embodiment_id,
            robot_spawn=RobotSpawnSpec(
                robot_id="robot",
                xyz=(0.0, -1.02, 0.31),
                wxyz=(0.7071067812, 0.0, 0.0, 0.7071067812),
                joint_positions=(0.0,) * 16,
            ),
            objects={},
            stages=(StageContractSpec(stage_index=0, metric_id="stage0_dry_run", clauses=()),),
        )

    def __call__(self, req: RolloutJobRequest) -> RolloutJobResult:
        embodiment_path = self.repo_root / "configs" / "embodiments" / f"{req.embodiment_id}.yaml"
        card = load_embodiment_card(embodiment_path)
        brief = self._resolve_brief(req)
        return run_controller_preflight(
            req.controller_code,
            brief,
            card,
            mode=req.mode,
            max_steps=req.max_steps,
            job_id=req.job_id,
        )


class RolloutWorkerHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server hosting `/v1/health` and `/v1/rollout`."""

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        auth_token: str | None = None,
        executor: RolloutExecutor | None = None,
        backend_name: str = "dry_run_preflight",
    ) -> None:
        self.auth_token = auth_token
        self.executor = executor if executor is not None else DryRunRolloutExecutor(
            backend_name=backend_name
        )
        self.backend_name = backend_name
        super().__init__(server_address, _WorkerRequestHandler)


class _WorkerRequestHandler(BaseHTTPRequestHandler):
    server: RolloutWorkerHTTPServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.server.auth_token
        if not expected:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        provided = header[len("Bearer ") :]
        return hmac.compare_digest(provided, expected)

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if self.path == "/v1/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "backend": self.server.backend_name,
                    "schema_version": "trajectory-forge-rollout-rpc/v0.1",
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if self.path != "/v1/rollout":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            req = RolloutJobRequest.from_dict(payload)
            result = self.server.executor(req)
        except (ControllerSafetyError, FirewallViolationError, ContractValidationError) as exc:
            err_result = RolloutJobResult(
                job_id=str(payload.get("job_id", "unknown")) if isinstance(locals().get("payload"), dict) else "unknown",
                status="rejected",
                mode=str(payload.get("mode", "unknown")) if isinstance(locals().get("payload"), dict) else "unknown",
                final_score=0.0,
                steps_executed=0,
                snapshots_used=0,
                firewall_verified=False,
                diagnostics=SolvabilityDiagnostics(
                    failure_code=type(exc).__name__,
                    detail=str(exc),
                ),
            )
            self._send_json(HTTPStatus.BAD_REQUEST, err_result.to_dict())
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.OK, result.to_dict())


def create_worker_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    auth_token: str | None = None,
    executor: RolloutExecutor | None = None,
    backend_name: str = "dry_run_preflight",
) -> RolloutWorkerHTTPServer:
    """Create a configured RolloutWorkerHTTPServer instance."""
    return RolloutWorkerHTTPServer(
        (host, port),
        auth_token=auth_token,
        executor=executor,
        backend_name=backend_name,
    )
