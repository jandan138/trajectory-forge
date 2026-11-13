from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from trajectory_forge.core.contracts import (
    EmbodimentCard,
    RolloutJobRequest,
    RolloutJobResult,
    ScenarioIntakeBrief,
    SnapshotHandle,
)
from trajectory_forge.intake.genmanip_intake import render_agent_brief_markdown


@dataclass(frozen=True)
class AttemptFeedbackPacket:
    """Structured failure/progress packet fed back to the agent after a rollout."""

    attempt: int
    mode: str
    status: str
    final_score: float
    steps_executed: int
    snapshots_used: int
    blocking_stage: int | None
    failure_code: str
    detail: str
    upstream_remediation_hint: str = ""
    stage_scores: tuple[float, ...] = ()
    saved_snapshots: tuple[SnapshotHandle, ...] = ()
    state_summary: dict[str, Any] = field(default_factory=dict)
    media_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "saved_snapshots": [s.to_dict() for s in self.saved_snapshots],
        }


def build_feedback_packet(
    *,
    attempt: int,
    request: RolloutJobRequest,
    result: RolloutJobResult,
) -> AttemptFeedbackPacket:
    """Summarize a RolloutJobResult into an actionable repair packet."""
    _ = request
    raw_scores = result.state_summary.get("stage_scores", ())
    stage_scores = tuple(float(x) for x in raw_scores) if isinstance(raw_scores, (list, tuple)) else ()

    blocking_stage = result.diagnostics.stage_index
    if blocking_stage is None and "blocking_stage" in result.state_summary:
        raw_blk = result.state_summary.get("blocking_stage")
        if isinstance(raw_blk, int):
            blocking_stage = raw_blk
    if blocking_stage is None and stage_scores:
        for idx, val in enumerate(stage_scores):
            if val < 0.9999:
                blocking_stage = idx
                break

    return AttemptFeedbackPacket(
        attempt=attempt,
        mode=result.mode,
        status=result.status,
        final_score=result.final_score,
        steps_executed=result.steps_executed,
        snapshots_used=result.snapshots_used,
        blocking_stage=blocking_stage,
        failure_code=result.diagnostics.failure_code,
        detail=result.diagnostics.detail,
        upstream_remediation_hint=result.diagnostics.upstream_remediation_hint,
        stage_scores=stage_scores,
        saved_snapshots=result.saved_snapshots,
        state_summary=dict(result.state_summary),
        media_paths=tuple(m.relpath for m in result.media),
    )


def render_repair_prompt_markdown(
    brief: ScenarioIntakeBrief,
    embodiment: EmbodimentCard,
    feedback: AttemptFeedbackPacket,
) -> str:
    """Render an Embodied-First-Exam-style Markdown repair prompt for a controller agent."""
    lines = [
        render_agent_brief_markdown(brief, embodiment).rstrip(),
        "",
        f"## Rollout Feedback (Attempt {feedback.attempt}, mode=`{feedback.mode}`)",
        "",
        f"- **Status**: `{feedback.status}` | **Final Score**: `{feedback.final_score:.4f}` | **Steps**: `{feedback.steps_executed}`",
        f"- **Blocking Stage**: `{feedback.blocking_stage}`",
        f"- **Failure Code**: `{feedback.failure_code or 'NONE'}`",
        f"- **Detail**: {feedback.detail or '(none)'}",
    ]
    if feedback.stage_scores:
        lines.append(f"- **Per-Stage Scores**: `{list(feedback.stage_scores)}`")
    if feedback.saved_snapshots:
        snaps = ", ".join(
            f"{s.snapshot_id}({s.label}@step={s.step_index})" for s in feedback.saved_snapshots
        )
        lines.append(f"- **Saved Stage Snapshots**: {snaps}")
    if feedback.upstream_remediation_hint:
        lines.append(
            f"- **Upstream Remediation Hint**: {feedback.upstream_remediation_hint}"
        )
    if feedback.media_paths:
        lines.append(f"- **Media Artifacts**: `{list(feedback.media_paths)}`")
    if feedback.state_summary:
        lines.extend(
            [
                "",
                "### Live State Snapshot Summary",
                "",
                "```json",
                json.dumps(feedback.state_summary, indent=2, ensure_ascii=False)[:4000],
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "### Required Controller Contract",
            "",
            "Return a complete Python module defining `def solve(ctx):` using only allowed imports "
            "(`math`, `numpy`, `trajectory_forge`, `typing`). Use `ctx.world.*`, `ctx.kinematics.*`, "
            "`ctx.end_effector.*`, `ctx.sensors.*`, and optional `ctx.checkpoint.save_if_search(label)` "
            "(never `restore_snapshot` if targeting continuous `full_horizon_gate`).",
        ]
    )
    return "\n".join(lines) + "\n"

