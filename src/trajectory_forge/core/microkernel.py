"""Simulator-neutral Microkernel v2 contracts (pure Python, zero simulator SDK imports).

Architecture:
  - Upper plugins (`trajectory_forge.producers.*`): trajectory generation paradigms
    (`code_to_control`, `direct_control`, `rl_policy`, future world-model / VLA / diff-physics).
  - Microkernel (`core`, `intake`, `verifier`, `trajectory`): invariant contracts,
    zero-snapshot multi-seed firewall gate, and unprivileged trajectory recording.
  - Lower plugins (`trajectory_forge.adapters.*`): physics engines declaring explicit
    `SimCapability` sets (`isaac_genmanip`, `newton`, future `mujoco` / `isaaclab`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from trajectory_forge.core.contracts import (
    ContractValidationError,
    EmbodimentCard,
    PoseSpec,
    ScenarioIntakeBrief,
)


class SimCapability(str, Enum):
    """Explicit capabilities declared by a `SimAdapter`."""

    SINGLE_ENV = "single_env"
    CHECKPOINT = "checkpoint"
    VEC_ENV = "vec_env"
    DIFF_PHYSICS = "diff_physics"
    RGB_RENDER = "rgb_render"
    PRIVILEGED_STATE = "privileged_state"


@dataclass(frozen=True)
class ActionCommand:
    """Simulator-neutral closed-loop action emitted by any `ExecutablePolicy`."""

    joint_targets: dict[str, float] = field(default_factory=dict)
    ee_postures: dict[str, str] = field(default_factory=dict)
    ee_joint_targets: dict[str, dict[str, float]] = field(default_factory=dict)
    ee_pose_targets: dict[str, PoseSpec] = field(default_factory=dict)
    substeps: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_targets": dict(self.joint_targets),
            "ee_postures": dict(self.ee_postures),
            "ee_joint_targets": {k: dict(v) for k, v in self.ee_joint_targets.items()},
            "ee_pose_targets": {k: v.to_dict() for k, v in self.ee_pose_targets.items()},
            "substeps": int(self.substeps),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StepObservation:
    """Unified observation packet returned after each simulator step.

    `privileged_state` is visible during `privileged_search`, but strictly stripped
    before recording any exported demonstration trajectory.
    """

    step_index: int
    sim_time_s: float
    joint_positions: dict[str, float]
    ee_poses: dict[str, PoseSpec] = field(default_factory=dict)
    object_poses: dict[str, PoseSpec] = field(default_factory=dict)
    stage_scores: dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    done: bool = False
    privileged_state: dict[str, Any] = field(default_factory=dict)

    def unprivileged_dict(self) -> dict[str, Any]:
        """Sensorimotor-only projection (guaranteed zero privileged state leakage)."""
        return {
            "step_index": int(self.step_index),
            "sim_time_s": float(self.sim_time_s),
            "joint_positions": dict(self.joint_positions),
            "ee_poses": {k: v.to_dict() for k, v in self.ee_poses.items()},
            "stage_scores": dict(self.stage_scores),
            "total_score": float(self.total_score),
            "done": bool(self.done),
        }


@runtime_checkable
class ExecutablePolicy(Protocol):
    """Unified closed-loop deliverable accepted by `FullHorizonGate`.

    Every trajectory generation paradigm (Code-to-Control, Direct Control, RL,
    World Model, VLA Bootstrap, Diff-Physics) delivers an `ExecutablePolicy` that
    can be verified across multiple seeds from `t=0` with zero snapshot resumes.
    """

    policy_kind: str
    producer_name: str

    def reset(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        seed: int = 0,
    ) -> None: ...

    def act(self, obs: StepObservation) -> ActionCommand: ...

    def is_finished(self, obs: StepObservation) -> bool: ...

    def to_manifest_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RelativeWaypoint:
    """Closed-loop waypoint anchored either in world frame or relative to a scene object."""

    label: str
    target_object: str | None = None
    offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    chain: str = "right_arm"
    joint_targets: dict[str, float] = field(default_factory=dict)
    ee_postures: dict[str, str] = field(default_factory=dict)
    ee_joint_targets: dict[str, dict[str, float]] = field(default_factory=dict)
    hold_steps: int = 20
    pos_tol_m: float = 0.025


class WaypointPolicy:
    """Closed-loop multi-seed waypoint policy (typically compiled from Direct Control)."""

    policy_kind = "waypoint_closed_loop"

    def __init__(
        self,
        waypoints: Sequence[RelativeWaypoint],
        *,
        producer_name: str = "direct_control",
    ) -> None:
        if not waypoints:
            raise ContractValidationError("WaypointPolicy requires at least one RelativeWaypoint")
        self.waypoints = tuple(waypoints)
        self.producer_name = producer_name
        self._wp_idx = 0
        self._wp_step = 0

    def reset(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        seed: int = 0,
    ) -> None:
        _ = brief, embodiment, seed
        self._wp_idx = 0
        self._wp_step = 0

    def act(self, obs: StepObservation) -> ActionCommand:
        idx = min(self._wp_idx, len(self.waypoints) - 1)
        wp = self.waypoints[idx]

        ee_poses: dict[str, PoseSpec] = {}
        if wp.target_object is not None and wp.target_object in obs.object_poses:
            base = obs.object_poses[wp.target_object].xyz
            ee_poses[wp.chain] = PoseSpec(
                xyz=(
                    base[0] + wp.offset_xyz[0],
                    base[1] + wp.offset_xyz[1],
                    base[2] + wp.offset_xyz[2],
                ),
                wxyz=wp.target_wxyz,
            )
        elif not wp.joint_targets and wp.offset_xyz != (0.0, 0.0, 0.0):
            ee_poses[wp.chain] = PoseSpec(xyz=wp.offset_xyz, wxyz=wp.target_wxyz)

        self._wp_step += 1
        reached = False
        if wp.chain in ee_poses and wp.chain in obs.ee_poses:
            cur = obs.ee_poses[wp.chain].xyz
            tgt = ee_poses[wp.chain].xyz
            dist = ((cur[0] - tgt[0]) ** 2 + (cur[1] - tgt[1]) ** 2 + (cur[2] - tgt[2]) ** 2) ** 0.5
            if dist <= wp.pos_tol_m and self._wp_step >= max(1, wp.hold_steps // 2):
                reached = True
        if self._wp_step >= wp.hold_steps or reached:
            if self._wp_idx < len(self.waypoints):
                self._wp_idx += 1
                self._wp_step = 0

        return ActionCommand(
            joint_targets=dict(wp.joint_targets),
            ee_postures=dict(wp.ee_postures),
            ee_joint_targets={k: dict(v) for k, v in wp.ee_joint_targets.items()},
            ee_pose_targets=ee_poses,
            metadata={"waypoint_label": wp.label, "waypoint_index": idx},
        )

    def is_finished(self, obs: StepObservation) -> bool:
        _ = obs
        return self._wp_idx >= len(self.waypoints)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": self.policy_kind,
            "producer_name": self.producer_name,
            "num_waypoints": len(self.waypoints),
            "waypoints": [asdict(w) for w in self.waypoints],
        }


class PythonControllerPolicy:
    """Closed-loop policy backed by an AST-validated Python `def act(obs, state):` script."""

    policy_kind = "python_controller"

    def __init__(
        self,
        source_code: str,
        *,
        producer_name: str = "code_to_control",
        max_steps: int = 2000,
    ) -> None:
        from trajectory_forge.sdk.safety import validate_controller_source

        validate_controller_source(source_code, required_fn="act")
        self.source_code = source_code
        self.producer_name = producer_name
        self.max_steps = int(max_steps)
        self._namespace: dict[str, Any] = {}
        self._state: dict[str, Any] = {}
        self._finished = False

    def reset(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        seed: int = 0,
    ) -> None:
        import math

        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
        ns: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "math": math,
            "ActionCommand": ActionCommand,
            "PoseSpec": PoseSpec,
        }
        exec(compile(self.source_code, "<python_controller_policy>", "exec"), ns, ns)
        if "act" not in ns or not callable(ns["act"]):
            raise ContractValidationError(
                "PythonControllerPolicy source must define callable `def act(obs, state):`"
            )
        self._namespace = ns
        self._state = {"brief": brief, "embodiment": embodiment, "seed": seed, "done": False}
        self._finished = False
        if "on_reset" in ns and callable(ns["on_reset"]):
            ns["on_reset"](brief, embodiment, self._state)

    def act(self, obs: StepObservation) -> ActionCommand:
        raw = self._namespace["act"](obs, self._state)
        if self._state.get("done"):
            self._finished = True
        if isinstance(raw, ActionCommand):
            return raw
        if isinstance(raw, Mapping):
            ee_poses = {
                k: (v if isinstance(v, PoseSpec) else PoseSpec.from_dict(v))
                for k, v in (raw.get("ee_pose_targets") or {}).items()
            }
            return ActionCommand(
                joint_targets={k: float(v) for k, v in (raw.get("joint_targets") or {}).items()},
                ee_postures={k: str(v) for k, v in (raw.get("ee_postures") or {}).items()},
                ee_joint_targets={
                    k: {jn: float(jv) for jn, jv in v.items()}
                    for k, v in (raw.get("ee_joint_targets") or {}).items()
                },
                ee_pose_targets=ee_poses,
                substeps=int(raw.get("substeps", 1)),
                metadata=dict(raw.get("metadata") or {}),
            )
        raise ContractValidationError(f"Unsupported action return type from act(): {type(raw)!r}")

    def is_finished(self, obs: StepObservation) -> bool:
        return self._finished or obs.step_index >= self.max_steps

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": self.policy_kind,
            "producer_name": self.producer_name,
            "max_steps": self.max_steps,
            "source_length": len(self.source_code),
        }


class CheckpointPolicy:
    """Closed-loop policy wrapper for learned RL / VLA / World-Model checkpoints."""

    policy_kind = "learned_checkpoint"

    def __init__(
        self,
        checkpoint_path: str,
        inference_fn: Callable[[StepObservation, dict[str, Any]], ActionCommand],
        *,
        producer_name: str = "rl_policy",
        max_steps: int = 1000,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self._inference_fn = inference_fn
        self.producer_name = producer_name
        self.max_steps = int(max_steps)
        self.metadata = dict(metadata or {})
        self._state: dict[str, Any] = {}

    def reset(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        seed: int = 0,
    ) -> None:
        self._state = {"brief": brief, "embodiment": embodiment, "seed": seed, "done": False}

    def act(self, obs: StepObservation) -> ActionCommand:
        return self._inference_fn(obs, self._state)

    def is_finished(self, obs: StepObservation) -> bool:
        return bool(self._state.get("done")) or obs.step_index >= self.max_steps

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": self.policy_kind,
            "producer_name": self.producer_name,
            "checkpoint_path": self.checkpoint_path,
            "max_steps": self.max_steps,
            "metadata": dict(self.metadata),
        }


class CompositePolicy:
    """Sequentially composes heterogeneous ExecutablePolicies (e.g. Code + RL skill)."""

    policy_kind = "composite"

    def __init__(
        self,
        stages: Sequence[tuple[str, ExecutablePolicy]],
        *,
        producer_name: str = "composite",
    ) -> None:
        if not stages:
            raise ContractValidationError("CompositePolicy requires at least one (label, policy) stage")
        self.stages = tuple(stages)
        self.producer_name = producer_name
        self._stage_idx = 0

    def reset(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        seed: int = 0,
    ) -> None:
        self._stage_idx = 0
        for _, pol in self.stages:
            pol.reset(brief, embodiment, seed=seed)

    def act(self, obs: StepObservation) -> ActionCommand:
        while self._stage_idx < len(self.stages) - 1 and self.stages[self._stage_idx][1].is_finished(obs):
            self._stage_idx += 1
        label, pol = self.stages[self._stage_idx]
        cmd = pol.act(obs)
        merged_meta = dict(cmd.metadata)
        merged_meta["composite_stage"] = label
        return ActionCommand(
            joint_targets=cmd.joint_targets,
            ee_postures=cmd.ee_postures,
            ee_joint_targets=cmd.ee_joint_targets,
            ee_pose_targets=cmd.ee_pose_targets,
            substeps=cmd.substeps,
            metadata=merged_meta,
        )

    def is_finished(self, obs: StepObservation) -> bool:
        return (
            self._stage_idx >= len(self.stages) - 1
            and self.stages[-1][1].is_finished(obs)
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": self.policy_kind,
            "producer_name": self.producer_name,
            "stages": [
                {"label": label, "policy": pol.to_manifest_dict()}
                for label, pol in self.stages
            ],
        }


@runtime_checkable
class SimSession(Protocol):
    """Live simulator session opened by a `SimAdapter`."""

    mode: str
    snapshots_used: int

    def reset(self, *, seed: int = 0) -> StepObservation: ...

    def step(self, action: ActionCommand) -> StepObservation: ...

    def save_checkpoint(self, label: str) -> str: ...

    def restore_checkpoint(self, snap_id: str) -> StepObservation: ...

    def close(self) -> None: ...


@runtime_checkable
class SimAdapter(Protocol):
    """Lower-layer simulator plugin declaring capabilities and opening live sessions."""

    backend_name: str
    capabilities: frozenset[SimCapability]

    def open_session(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        mode: str = "full_horizon_gate",
        seed: int = 0,
        options: Mapping[str, Any] | None = None,
    ) -> SimSession: ...


@runtime_checkable
class TrajectoryProducer(Protocol):
    """Upper-layer trajectory generation paradigm plugin."""

    producer_name: str
    required_capabilities: frozenset[SimCapability]

    def produce(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        adapter: SimAdapter,
        **kwargs: Any,
    ) -> ExecutablePolicy: ...
