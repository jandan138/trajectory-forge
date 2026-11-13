from __future__ import annotations

import threading
import urllib.error
import urllib.request

import pytest

from trajectory_forge.core.contracts import RolloutJobRequest
from trajectory_forge.worker.client import RolloutWorkerClient
from trajectory_forge.worker.server import create_worker_server


CONTROLLER_SAMPLE = """
def solve(ctx):
    ctx.end_effector.set_posture("left_gripper", "open")
    ctx.step(2)
"""


def test_worker_http_rpc_roundtrip_and_bearer_auth() -> None:
    token = "test-secret-token"
    server = create_worker_server(host="127.0.0.1", port=0, auth_token=token)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"

        # Unauthorized request must fail with 401
        req_unauth = urllib.request.Request(f"{base_url}/v1/health", method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_unauth, timeout=5)
        assert exc_info.value.code == 401

        client = RolloutWorkerClient(base_url=base_url, auth_token=token)
        health = client.health()
        assert health["status"] == "ok"
        assert health["backend"] == "dry_run_preflight"

        job = RolloutJobRequest(
            job_id="job-001",
            scenario_package_ref="configs/benchmarks/r3_out_of_domain_suite.yaml",
            embodiment_id="lift2_r5a_bimanual",
            mode="full_horizon_gate",
            seed=101,
            controller_code=CONTROLLER_SAMPLE,
            max_steps=50,
            resume_from_snapshot_id=None,
        )
        result = client.submit_rollout(job)
        assert result.job_id == "job-001"
        assert result.status == "succeeded"
        assert result.firewall_verified is True
        assert result.trajectory_manifest is not None
        assert result.trajectory_manifest.continuous_from_t0 is True
    finally:
        server.shutdown()
        server.server_close()
