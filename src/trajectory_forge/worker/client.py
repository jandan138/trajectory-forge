from __future__ import annotations

import json
from typing import Any
import urllib.request

from trajectory_forge.core.contracts import RolloutJobRequest, RolloutJobResult


class RolloutWorkerClient:
    """Stdlib HTTP client used by the cloud brain to submit jobs to the Sim Worker."""

    def __init__(
        self,
        *,
        base_url: str,
        auth_token: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def health(self) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/v1/health",
            headers=self._headers(),
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return dict(json.loads(resp.read().decode("utf-8")))

    def submit_rollout(self, job: RolloutJobRequest) -> RolloutJobResult:
        body = json.dumps(job.to_dict()).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/rollout",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return RolloutJobResult.from_dict(payload)
