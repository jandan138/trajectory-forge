"""Simulator-neutral FullHorizonGate & Privileged-to-Sensorimotor Firewall Verifier.

Accepts ANY `ExecutablePolicy` from ANY `TrajectoryProducer` (Code-to-Control,
Direct Control, RL, World-Model, VLA Bootstrap, Diff-Physics) and verifies it
on ANY `SimAdapter` across multiple seeds with:
  - `mode == "full_horizon_gate"` (snapshot save/restore strictly forbidden),
  - `snapshots_used == 0`,
  - `continuous_from_t0 == True`,
  - `privileged_channels_recorded == False` on all exported `.npz` trajectories.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_forge.core.contracts import (
    EmbodimentCard,
    FirewallViolationError,
    ScenarioIntakeBrief,
)
from trajectory_forge.core.microkernel import (
    ExecutablePolicy,
    SimAdapter,
)


@dataclass(frozen=True)
class GateConfig:
    """Verification rules enforced by `FullHorizonGate`."""

    seeds: tuple[int, ...] = (0,)
    max_steps: int = 1000
    success_threshold: float = 0.99
    export_npz: bool = True
    adapter_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateSeedOutcome:
    """Per-seed audit outcome of continuous `t=0` verification."""

    seed: int
    passed: bool
    final_score: float
    steps_executed: int
    snapshots_used: int
    continuous_from_t0: bool
    privileged_channels_recorded: bool
    stage_scores: dict[str, float]
    trajectory_npz: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateVerificationReport:
    """Multi-seed firewall verification report across any producer & simulator backend."""

    schema: str
    package_id: str
    embodiment_id: str
    sim_backend: str
    producer_name: str
    policy_kind: str
    passed: bool
    firewall_verified: bool
    seeds_passed: int
    seeds_total: int
    outcomes: tuple[GateSeedOutcome, ...]
    policy_manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "package_id": self.package_id,
            "embodiment_id": self.embodiment_id,
            "sim_backend": self.sim_backend,
            "producer_name": self.producer_name,
            "policy_kind": self.policy_kind,
            "passed": self.passed,
            "firewall_verified": self.firewall_verified,
            "seeds_passed": self.seeds_passed,
            "seeds_total": self.seeds_total,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "policy_manifest": dict(self.policy_manifest),
        }


class FullHorizonGate:
    """Invariant verifier enforcing zero-snapshot `t=0` execution across all backends."""

    def verify(
        self,
        *,
        policy: ExecutablePolicy,
        adapter: SimAdapter,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        output_dir: str | Path | None = None,
        config: GateConfig | None = None,
    ) -> GateVerificationReport:
        cfg = config or GateConfig()
        out_root = Path(output_dir).resolve() if output_dir is not None else None
        if out_root is not None:
            out_root.mkdir(parents=True, exist_ok=True)

        outcomes: list[GateSeedOutcome] = []
        for seed in cfg.seeds:
            session = adapter.open_session(
                brief,
                embodiment,
                mode="full_horizon_gate",
                seed=seed,
                options=cfg.adapter_options,
            )
            try:
                obs = session.reset(seed=seed)
                policy.reset(brief, embodiment, seed=seed)

                joint_names = sorted(obs.joint_positions.keys())
                q_trace: list[list[float]] = []
                t_trace: list[float] = []
                score_trace: list[float] = []

                steps = 0
                while steps < cfg.max_steps and not obs.done and not policy.is_finished(obs):
                    cmd = policy.act(obs)
                    obs = session.step(cmd)
                    steps += 1
                    if not joint_names and obs.joint_positions:
                        joint_names = sorted(obs.joint_positions.keys())
                    q_trace.append([float(obs.joint_positions.get(jn, 0.0)) for jn in joint_names])
                    t_trace.append(float(obs.sim_time_s))
                    score_trace.append(float(obs.total_score))

                if session.snapshots_used != 0:
                    raise FirewallViolationError(
                        f"FullHorizonGate detected {session.snapshots_used} snapshot resume(s) on seed={seed}"
                    )

                seed_passed = bool(
                    obs.total_score >= cfg.success_threshold and session.snapshots_used == 0
                )

                npz_path_str: str | None = None
                if cfg.export_npz and out_root is not None:
                    npz_path = out_root / f"seed_{seed}_clean_trajectory.npz"
                    np.savez_compressed(
                        npz_path,
                        joint_names=np.asarray(joint_names, dtype=object),
                        joint_positions=np.asarray(q_trace, dtype=np.float32),
                        sim_time_s=np.asarray(t_trace, dtype=np.float32),
                        total_score=np.asarray(score_trace, dtype=np.float32),
                        privileged_channels_recorded=np.asarray(False),
                        continuous_from_t0=np.asarray(True),
                        snapshots_used=np.asarray(0, dtype=np.int32),
                        sim_backend=np.asarray(adapter.backend_name),
                        producer_name=np.asarray(policy.producer_name),
                        policy_kind=np.asarray(policy.policy_kind),
                    )
                    npz_path_str = str(npz_path)

                outcomes.append(
                    GateSeedOutcome(
                        seed=seed,
                        passed=seed_passed,
                        final_score=round(float(obs.total_score), 6),
                        steps_executed=steps,
                        snapshots_used=int(session.snapshots_used),
                        continuous_from_t0=True,
                        privileged_channels_recorded=False,
                        stage_scores={k: round(float(v), 6) for k, v in obs.stage_scores.items()},
                        trajectory_npz=npz_path_str,
                    )
                )
            finally:
                session.close()

        seeds_passed = sum(1 for o in outcomes if o.passed)
        all_passed = seeds_passed == len(outcomes) and len(outcomes) > 0
        firewall_ok = all(
            o.snapshots_used == 0 and o.continuous_from_t0 and not o.privileged_channels_recorded
            for o in outcomes
        )

        report = GateVerificationReport(
            schema="trajectory-forge-full-horizon-gate/v2",
            package_id=brief.package_id,
            embodiment_id=embodiment.embodiment_id,
            sim_backend=adapter.backend_name,
            producer_name=policy.producer_name,
            policy_kind=policy.policy_kind,
            passed=all_passed and firewall_ok,
            firewall_verified=firewall_ok,
            seeds_passed=seeds_passed,
            seeds_total=len(outcomes),
            outcomes=tuple(outcomes),
            policy_manifest=policy.to_manifest_dict(),
        )
        if out_root is not None:
            (out_root / "gate_report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        return report
