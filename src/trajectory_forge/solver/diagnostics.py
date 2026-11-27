from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from trajectory_forge.core.contracts import (
    EmbodimentCard,
    RemediationPolicySpec,
    RolloutJobRequest,
    RolloutJobResult,
    ScenarioIntakeBrief,
    SnapshotHandle,
    SolvabilityDiagnostics,
)
from trajectory_forge.intake.registry import render_agent_brief_markdown


_PHYSICS_FAILURE_CATEGORIES: dict[str, str] = {
    "UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT": "joint_drive",
    "UPSTREAM_MISSING_HANDLE_COLLIDER": "collision_proxy",
    "UPSTREAM_CONTACT_FRICTION_OR_OFFSET": "friction_or_offset",
}


def apply_remediation_policy(
    diagnostics: SolvabilityDiagnostics,
    policy: RemediationPolicySpec,
    *,
    object_dims_xyz_m: tuple[float, float, float] | None = None,
    max_graspable_span_m: float = 0.080,
) -> SolvabilityDiagnostics:
    """Enforce dual-mode upstream remediation permissions on raw physical diagnostics."""
    code = diagnostics.failure_code
    if code in _PHYSICS_FAILURE_CATEGORIES:
        return SolvabilityDiagnostics(
            failure_code=diagnostics.failure_code,
            stage_index=diagnostics.stage_index,
            detail=diagnostics.detail,
            colliding_pairs=diagnostics.colliding_pairs,
            upstream_remediation_hint=diagnostics.upstream_remediation_hint,
            remediation_category=_PHYSICS_FAILURE_CATEGORIES[code],
            required_remediation_mode="physics_tuning",
            requires_pose_adjustment=False,
            policy_blocked=False,
            root_failure_code="",
        )

    if code == "UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE":
        can_solve_via_pose = (
            object_dims_xyz_m is not None
            and min(float(v) for v in object_dims_xyz_m) <= max_graspable_span_m
        )
        if policy.mode == "scene_codesign":
            hint = diagnostics.upstream_remediation_hint or (
                f"In upstream scenario-forge under scene_codesign, add an affordance handle "
                f"or holder <= {max_graspable_span_m:.3f} m wide so the parallel-jaw gripper can pinch it."
            )
            return SolvabilityDiagnostics(
                failure_code=code,
                stage_index=diagnostics.stage_index,
                detail=diagnostics.detail,
                colliding_pairs=diagnostics.colliding_pairs,
                upstream_remediation_hint=hint,
                remediation_category="affordance_part",
                required_remediation_mode="scene_codesign",
                requires_pose_adjustment=can_solve_via_pose,
                policy_blocked=False,
                root_failure_code="",
            )

        if policy.allow_pose_adjustment and can_solve_via_pose:
            min_dim = min(float(v) for v in object_dims_xyz_m)  # type: ignore[union-attr]
            return SolvabilityDiagnostics(
                failure_code=code,
                stage_index=diagnostics.stage_index,
                detail=diagnostics.detail,
                colliding_pairs=diagnostics.colliding_pairs,
                upstream_remediation_hint=(
                    f"Adjust initial tabletop pose/orientation without modifying visual meshes or adding parts "
                    f"(minimum thickness {min_dim:.3f} m <= {max_graspable_span_m:.3f} m: overhang edge past table front "
                    f"with CoM supported on tabletop, or orient upright)."
                ),
                remediation_category="pose_adjustment",
                required_remediation_mode="physics_tuning",
                requires_pose_adjustment=True,
                policy_blocked=False,
                root_failure_code="",
            )

        if can_solve_via_pose and not policy.allow_pose_adjustment:
            blocked_hint = (
                "Active policy (physics_tuning, allow_pose_adjustment=False) forbids both structural part edits "
                "and initial pose changes. Enable allow_pose_adjustment=True (edge overhang without visual edits), "
                "switch to scene_codesign (add graspable handle/holder), or use a wider-aperture/suction end-effector."
            )
            category = "pose_adjustment"
        else:
            blocked_hint = (
                f"All three object dimensions exceed {max_graspable_span_m:.3f} m; initial pose changes alone "
                "cannot expose a pinchable edge. Active policy physics_tuning forbids structural part additions; "
                "switch to scene_codesign or change embodiment."
            )
            category = "affordance_part"

        return SolvabilityDiagnostics(
            failure_code="BLOCKED_BY_REMEDIATION_POLICY",
            stage_index=diagnostics.stage_index,
            detail=(
                f"{diagnostics.detail} [Blocked under policy mode={policy.mode!r}, "
                f"allow_pose_adjustment={policy.allow_pose_adjustment}]"
            ),
            colliding_pairs=diagnostics.colliding_pairs,
            upstream_remediation_hint=blocked_hint,
            remediation_category=category,
            required_remediation_mode="scene_codesign",
            requires_pose_adjustment=can_solve_via_pose,
            policy_blocked=True,
            root_failure_code=code,
        )

    return diagnostics


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
    remediation_policy: dict[str, Any] = field(default_factory=dict)
    remediation_category: str = "none"
    required_remediation_mode: str = ""
    requires_pose_adjustment: bool = False
    policy_blocked: bool = False
    root_failure_code: str = ""
    stage_scores: tuple[float, ...] = ()
    saved_snapshots: tuple[SnapshotHandle, ...] = ()
    state_summary: dict[str, Any] = field(default_factory=dict)
    media_paths: tuple[str, ...] = ()
    ik_failures: tuple[dict[str, Any], ...] = ()
    ee_contact_metrics: dict[str, Any] = field(default_factory=dict)
    failure_keyframes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "saved_snapshots": [s.to_dict() for s in self.saved_snapshots],
        }


def compute_sequential_blocking_stage(
    *,
    num_stages: int,
    cur_union_metric_is_none: bool,
    long_stride_idx: int,
    final_score: float,
) -> tuple[int, int | None]:
    """Return (stages_completed_sequentially, blocking_stage) for sequential goal rubrics."""
    if num_stages <= 0:
        return (0, None)
    if cur_union_metric_is_none or final_score >= 0.9999:
        return (num_stages, None)
    completed = max(0, min(num_stages, int(long_stride_idx)))
    blocking = completed if completed < num_stages else None
    return (completed, blocking)


def build_feedback_packet(
    *,
    attempt: int,
    request: RolloutJobRequest,
    result: RolloutJobResult,
) -> AttemptFeedbackPacket:
    """Summarize a RolloutJobResult into an actionable repair packet."""
    raw_scores = result.state_summary.get("stage_scores", ())
    stage_scores = tuple(float(x) for x in raw_scores) if isinstance(raw_scores, (list, tuple)) else ()

    blocking_stage = result.diagnostics.stage_index
    has_explicit_blocking_key = "blocking_stage" in result.state_summary
    if blocking_stage is None and has_explicit_blocking_key:
        raw_blk = result.state_summary.get("blocking_stage")
        if isinstance(raw_blk, int):
            blocking_stage = raw_blk
    if (
        blocking_stage is None
        and not has_explicit_blocking_key
        and stage_scores
        and result.final_score < 0.9999
    ):
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
        remediation_policy=request.remediation_policy.to_dict(),
        remediation_category=result.diagnostics.remediation_category,
        required_remediation_mode=result.diagnostics.required_remediation_mode,
        requires_pose_adjustment=result.diagnostics.requires_pose_adjustment,
        policy_blocked=result.diagnostics.policy_blocked,
        root_failure_code=result.diagnostics.root_failure_code,
        stage_scores=stage_scores,
        saved_snapshots=result.saved_snapshots,
        state_summary=dict(result.state_summary),
        media_paths=tuple(m.relpath for m in result.media),
        ik_failures=tuple(
            x for x in (result.state_summary.get("ik_failures") or ()) if isinstance(x, dict)
        ),
        ee_contact_metrics=dict(result.state_summary.get("ee_contact_metrics") or {}),
        failure_keyframes=tuple(
            m.relpath
            for m in result.media
            if m.relpath.endswith((".png", ".jpg", ".jpeg"))
        ),
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
        f"- **Failure Code**: `{feedback.failure_code or 'NONE'}`"
        + (f" (root: `{feedback.root_failure_code}`)" if feedback.root_failure_code else ""),
        f"- **Detail**: {feedback.detail or '(none)'}",
    ]
    if feedback.remediation_policy or feedback.remediation_category != "none" or feedback.policy_blocked:
        lines.append(
            f"- **Remediation Policy**: `{json.dumps(feedback.remediation_policy)}` | "
            f"**Category**: `{feedback.remediation_category}` | "
            f"**Required Mode**: `{feedback.required_remediation_mode or 'none'}` | "
            f"**Policy Blocked**: `{feedback.policy_blocked}`"
        )
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
    if feedback.ik_failures:
        lines.append(f"- **IK Failures**: `{json.dumps(list(feedback.ik_failures)[:5], ensure_ascii=False)}`")
    if feedback.ee_contact_metrics:
        lines.append(
            f"- **EE Contact & Residual Metrics**: `{json.dumps(feedback.ee_contact_metrics, ensure_ascii=False)}`"
        )
    if feedback.failure_keyframes:
        lines.append(f"- **Failure Keyframes**: `{list(feedback.failure_keyframes)}`")
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

