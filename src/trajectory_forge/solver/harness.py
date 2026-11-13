from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from trajectory_forge.core.contracts import (
    ContractValidationError,
    ControllerSafetyError,
    EmbodimentCard,
    FirewallViolationError,
    RolloutJobRequest,
    RolloutJobResult,
    ScenarioIntakeBrief,
    SolvabilityDiagnostics,
)
from trajectory_forge.sdk.context import run_controller_preflight
from trajectory_forge.solver.diagnostics import (
    AttemptFeedbackPacket,
    build_feedback_packet,
)
from trajectory_forge.solver.synthesizer import ControllerSynthesizer


RolloutExecutor = Callable[[RolloutJobRequest], RolloutJobResult]


@dataclass(frozen=True)
class SolveSessionConfig:
    """Configuration governing one Embodied-First-Exam Privileged solve session."""

    max_attempts: int = 5
    max_steps: int = 2000
    seed: int = 0
    synthesize_trajectory: bool = True
    render_gate_media: bool = False
    render_synthesis_media: bool = True
    success_threshold: float = 0.9999

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ContractValidationError("SolveSessionConfig.max_attempts must be > 0")
        if self.max_steps <= 0:
            raise ContractValidationError("SolveSessionConfig.max_steps must be > 0")


@dataclass(frozen=True)
class SolveSessionReport:
    """Complete audit report for a privileged solve + gate + synthesis session."""

    package_id: str
    embodiment_id: str
    succeeded: bool
    attempts_used: int
    final_code: str
    history: tuple[AttemptFeedbackPacket, ...]
    gate_result: RolloutJobResult | None = None
    synthesis_result: RolloutJobResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "embodiment_id": self.embodiment_id,
            "succeeded": self.succeeded,
            "attempts_used": self.attempts_used,
            "final_code": self.final_code,
            "history": [h.to_dict() for h in self.history],
            "gate_result": self.gate_result.to_dict() if self.gate_result is not None else None,
            "synthesis_result": (
                self.synthesis_result.to_dict() if self.synthesis_result is not None else None
            ),
            "config_summary": asdict(
                SolveSessionConfig(max_attempts=max(1, self.attempts_used))
            ),
        }


class PrivilegedSolverHarness:
    """Closed-loop Embodied-First-Exam Privileged Mode solver and firewall verifier."""

    def __init__(
        self,
        *,
        executor: RolloutExecutor,
        preflight_enabled: bool = True,
    ) -> None:
        self.executor = executor
        self.preflight_enabled = preflight_enabled

    def _run_preflight_if_enabled(
        self,
        *,
        code: str,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        mode: str,
        max_steps: int,
        job_id: str,
    ) -> RolloutJobResult | None:
        if not self.preflight_enabled:
            return None
        try:
            run_controller_preflight(
                code,
                brief,
                embodiment,
                mode=mode,
                max_steps=max_steps,
                job_id=f"{job_id}_preflight",
            )
            return None
        except (ControllerSafetyError, FirewallViolationError, ContractValidationError) as exc:
            return RolloutJobResult(
                job_id=job_id,
                status="rejected",
                mode=mode,
                final_score=0.0,
                steps_executed=0,
                snapshots_used=0,
                firewall_verified=False,
                diagnostics=SolvabilityDiagnostics(
                    failure_code=type(exc).__name__,
                    detail=str(exc),
                ),
            )

    def solve(
        self,
        *,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        synthesizer: ControllerSynthesizer,
        package_ref: str | None = None,
        config: SolveSessionConfig | None = None,
    ) -> SolveSessionReport:
        cfg = config or SolveSessionConfig()
        ref = package_ref or brief.package_id
        history: list[AttemptFeedbackPacket] = []

        code = synthesizer.propose_initial(brief, embodiment)
        gate_result: RolloutJobResult | None = None
        synthesis_result: RolloutJobResult | None = None

        for attempt in range(1, cfg.max_attempts + 1):
            search_job_id = f"{brief.package_id}_a{attempt}_search"
            search_req = RolloutJobRequest(
                job_id=search_job_id,
                scenario_package_ref=ref,
                embodiment_id=embodiment.embodiment_id,
                mode="privileged_search",
                seed=cfg.seed,
                controller_code=code,
                max_steps=cfg.max_steps,
                render_media=False,
            )
            preflight_err = self._run_preflight_if_enabled(
                code=code,
                brief=brief,
                embodiment=embodiment,
                mode="privileged_search",
                max_steps=cfg.max_steps,
                job_id=search_job_id,
            )
            if preflight_err is not None:
                pkt = build_feedback_packet(
                    attempt=attempt, request=search_req, result=preflight_err
                )
                history.append(pkt)
                if attempt < cfg.max_attempts:
                    code = synthesizer.repair(brief, embodiment, code, pkt)
                continue

            search_res = self.executor(search_req)
            if search_res.final_score < cfg.success_threshold:
                pkt = build_feedback_packet(attempt=attempt, request=search_req, result=search_res)
                history.append(pkt)
                if attempt < cfg.max_attempts:
                    code = synthesizer.repair(brief, embodiment, code, pkt)
                continue

            # Passed privileged_search -> mandatory zero-snapshot full_horizon_gate
            gate_job_id = f"{brief.package_id}_a{attempt}_gate"
            gate_req = RolloutJobRequest(
                job_id=gate_job_id,
                scenario_package_ref=ref,
                embodiment_id=embodiment.embodiment_id,
                mode="full_horizon_gate",
                seed=cfg.seed,
                controller_code=code,
                max_steps=cfg.max_steps,
                render_media=cfg.render_gate_media,
            )
            gate_preflight_err = self._run_preflight_if_enabled(
                code=code,
                brief=brief,
                embodiment=embodiment,
                mode="full_horizon_gate",
                max_steps=cfg.max_steps,
                job_id=gate_job_id,
            )
            if gate_preflight_err is not None:
                gate_result = gate_preflight_err
                pkt = build_feedback_packet(
                    attempt=attempt, request=gate_req, result=gate_preflight_err
                )
                history.append(pkt)
                if attempt < cfg.max_attempts:
                    code = synthesizer.repair(brief, embodiment, code, pkt)
                continue

            gate_res = self.executor(gate_req)
            gate_result = gate_res
            if (
                gate_res.final_score < cfg.success_threshold
                or gate_res.snapshots_used != 0
                or not gate_res.firewall_verified
            ):
                pkt = build_feedback_packet(attempt=attempt, request=gate_req, result=gate_res)
                history.append(pkt)
                if attempt < cfg.max_attempts:
                    code = synthesizer.repair(brief, embodiment, code, pkt)
                continue

            if cfg.synthesize_trajectory:
                synth_job_id = f"{brief.package_id}_a{attempt}_synth"
                synth_req = RolloutJobRequest(
                    job_id=synth_job_id,
                    scenario_package_ref=ref,
                    embodiment_id=embodiment.embodiment_id,
                    mode="trajectory_synthesis",
                    seed=cfg.seed,
                    controller_code=code,
                    max_steps=cfg.max_steps,
                    render_media=cfg.render_synthesis_media,
                )
                synth_res = self.executor(synth_req)
                synthesis_result = synth_res
                if (
                    synth_res.final_score < cfg.success_threshold
                    or synth_res.snapshots_used != 0
                    or not synth_res.firewall_verified
                    or synth_res.trajectory_manifest is None
                ):
                    pkt = build_feedback_packet(
                        attempt=attempt, request=synth_req, result=synth_res
                    )
                    history.append(pkt)
                    if attempt < cfg.max_attempts:
                        code = synthesizer.repair(brief, embodiment, code, pkt)
                    continue

            history.append(build_feedback_packet(attempt=attempt, request=gate_req, result=gate_res))
            return SolveSessionReport(
                package_id=brief.package_id,
                embodiment_id=embodiment.embodiment_id,
                succeeded=True,
                attempts_used=attempt,
                final_code=code,
                history=tuple(history),
                gate_result=gate_result,
                synthesis_result=synthesis_result,
            )

        return SolveSessionReport(
            package_id=brief.package_id,
            embodiment_id=embodiment.embodiment_id,
            succeeded=False,
            attempts_used=cfg.max_attempts,
            final_code=code,
            history=tuple(history),
            gate_result=gate_result,
            synthesis_result=synthesis_result,
        )

