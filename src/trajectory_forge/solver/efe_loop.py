"""Embodied First Exam (EFE) Privileged Auto-Debug Loop (ADR-20261126 Layer 4).

Strictly pure Python (no simulator/IK SDK imports). Orchestrates:
  1. `privileged_search`: iterative Coding Agent repair with stage checkpoints
     (`ctx.checkpoint.save_if_search` / `restore_snapshot`) and multimodal
     `AttemptFeedbackPacket` (IK failures, EE contact/residual metrics, keyframes).
  2. `full_horizon_gate`: mandatory continuous-from-t0 verification with
     `snapshots_used == 0` and `firewall_verified == True`.
  3. `trajectory_synthesis`: clean unprivileged demonstration export.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trajectory_forge.core.contracts import EmbodimentCard, ScenarioIntakeBrief
from trajectory_forge.solver.diagnostics import AttemptFeedbackPacket
from trajectory_forge.solver.harness import (
    PrivilegedSolverHarness,
    SolveSessionConfig,
    SolveSessionReport,
)
from trajectory_forge.solver.synthesizer import ControllerSynthesizer


@dataclass(frozen=True)
class EFELoopConfig:
    """Configuration for the EFE Privileged Auto-Debug Loop."""

    max_search_attempts: int = 8
    target_score: float = 1.0
    require_zero_snapshot_gate: bool = True
    synthesize_on_pass: bool = True
    seed: int = 0
    step_budget: int = 8000
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EFELoopResult:
    """Auditable outcome of an EFE Privileged Auto-Debug Loop."""

    embodiment_id: str
    task_id: str
    succeeded: bool
    search_attempts: int
    gate_score: float
    gate_snapshots_used: int
    firewall_verified: bool
    winning_controller_code: str
    feedback_history: tuple[AttemptFeedbackPacket, ...]
    report: SolveSessionReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "embodiment_id": self.embodiment_id,
            "task_id": self.task_id,
            "succeeded": self.succeeded,
            "search_attempts": self.search_attempts,
            "gate_score": self.gate_score,
            "gate_snapshots_used": self.gate_snapshots_used,
            "firewall_verified": self.firewall_verified,
            "feedback_count": len(self.feedback_history),
            "report": self.report.to_dict(),
        }


class EFEPrivilegedLoop:
    """High-level EFE Coding Agent driver wrapping `PrivilegedSolverHarness`."""

    def __init__(
        self,
        harness: PrivilegedSolverHarness,
        synthesizer: ControllerSynthesizer,
        config: EFELoopConfig | None = None,
    ) -> None:
        self._harness = harness
        self._synthesizer = synthesizer
        self._config = config or EFELoopConfig()

    def run(
        self,
        *,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        output_dir: Path,
    ) -> EFELoopResult:
        session_cfg = SolveSessionConfig(
            max_search_attempts=self._config.max_search_attempts,
            target_score=self._config.target_score,
            run_full_horizon_gate=self._config.require_zero_snapshot_gate,
            run_synthesis=self._config.synthesize_on_pass,
            seed=self._config.seed,
            step_budget=self._config.step_budget,
        )
        report = self._harness.solve(
            brief=brief,
            embodiment=embodiment,
            synthesizer=self._synthesizer,
            output_dir=output_dir,
            config=session_cfg,
        )
        gate_res = report.gate_result
        gate_score = float(gate_res.final_score) if gate_res is not None else 0.0
        gate_snaps = int(gate_res.snapshots_used) if gate_res is not None else 0
        firewall_ok = bool(
            gate_res is not None
            and gate_res.snapshots_used == 0
            and gate_res.final_score >= self._config.target_score - 1e-4
        )
        return EFELoopResult(
            embodiment_id=embodiment.embodiment_id,
            task_id=brief.task_id,
            succeeded=bool(report.succeeded and firewall_ok),
            search_attempts=len(report.search_attempts),
            gate_score=gate_score,
            gate_snapshots_used=gate_snaps,
            firewall_verified=firewall_ok,
            winning_controller_code=report.winning_controller_code,
            feedback_history=tuple(report.feedback_history),
            report=report,
        )
