from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trajectory_forge.core.contracts import (
    ALLOWED_EXECUTION_MODES,
    ContractValidationError,
    EmbodimentCard,
    FirewallViolationError,
    RolloutJobResult,
    ScenarioIntakeBrief,
    SnapshotHandle,
    SolvabilityDiagnostics,
    StageProgressEntry,
)
from trajectory_forge.sdk.math3d import (
    Pose3D,
    Vec3,
    pose_from_named_frame,
    sample_prismatic_sweep_poses,
    sample_revolute_arc_poses,
)
from trajectory_forge.trajectory.manifest import build_trajectory_episode_manifest
from trajectory_forge.sdk.safety import validate_controller_source


@dataclass(frozen=True)
class TactileReading:
    """Standardized tactile pad summary reserved for tactile grippers & dexterous hands."""

    group_id: str
    normal_force_n: float
    shear_force_n: tuple[float, float]
    slip_detected: bool
    in_contact: bool


@dataclass(frozen=True)
class WrenchReading:
    """6-axis wrist force-torque reading."""

    chain_id: str
    force_n: Vec3
    torque_nm: Vec3


class WorldAPI:
    """Privileged world inspection API exposed to the agent controller."""

    def __init__(self, brief: ScenarioIntakeBrief) -> None:
        self._brief = brief

    @property
    def instruction(self) -> str:
        return self._brief.instruction

    def object_pose(self, object_id: str) -> Pose3D:
        if object_id not in self._brief.objects:
            raise KeyError(f"Unknown object_id: {object_id!r}")
        return Pose3D.from_spec(self._brief.objects[object_id].initial_pose)

    def named_frame(self, object_id: str, frame_name: str) -> Pose3D:
        if object_id not in self._brief.objects:
            raise KeyError(f"Unknown object_id: {object_id!r}")
        return pose_from_named_frame(self._brief.objects[object_id], frame_name)

    def joint_band(self, object_id: str, semantic_joint: str, state_name: str) -> tuple[float, float]:
        if object_id not in self._brief.objects:
            raise KeyError(f"Unknown object_id: {object_id!r}")
        obj = self._brief.objects[object_id]
        if semantic_joint not in obj.dofs:
            raise KeyError(f"Object {object_id!r} has no dof {semantic_joint!r}")
        dof = obj.dofs[semantic_joint]
        if state_name not in dof.states:
            raise KeyError(f"DOF {semantic_joint!r} has no state {state_name!r}")
        return dof.states[state_name]


class KinematicsAPI:
    """Chain-agnostic kinematics and mechanism arc/sweep trajectory helpers."""

    def __init__(self, embodiment: EmbodimentCard, backend: DryRunSimBackend) -> None:
        self._embodiment = embodiment
        self._backend = backend

    def solve_ik(
        self,
        chain: str,
        target_pose: Pose3D,
        check_collision: bool = True,
    ) -> tuple[float, ...]:
        if chain not in self._embodiment.chains:
            raise KeyError(f"Unknown kinematic chain: {chain!r}")
        spec = self._embodiment.chains[chain]
        self._backend.record_chain_command(chain, target_pose, check_collision=check_collision)
        return tuple(0.0 for _ in spec.dof_indices)

    def plan_revolute_arc(
        self,
        *,
        chain: str,
        start_pose: Pose3D,
        hinge_pivot_world: Vec3,
        hinge_axis_world: Vec3,
        delta_angle_rad: float,
        num_waypoints: int = 8,
    ) -> list[Pose3D]:
        if chain not in self._embodiment.chains:
            raise KeyError(f"Unknown kinematic chain: {chain!r}")
        return sample_revolute_arc_poses(
            start_pose=start_pose,
            hinge_pivot_world=hinge_pivot_world,
            hinge_axis_world=hinge_axis_world,
            delta_angle_rad=delta_angle_rad,
            num_waypoints=num_waypoints,
        )

    def plan_linear_sweep(
        self,
        *,
        chain: str,
        start_pose: Pose3D,
        direction_world: Vec3,
        distance_m: float,
        num_waypoints: int = 8,
    ) -> list[Pose3D]:
        if chain not in self._embodiment.chains:
            raise KeyError(f"Unknown kinematic chain: {chain!r}")
        return sample_prismatic_sweep_poses(
            start_pose=start_pose,
            direction_world=direction_world,
            distance_m=distance_m,
            num_waypoints=num_waypoints,
        )


class EndEffectorAPI:
    """Cross-embodiment actuator group API (parallel grippers and dexterous hands)."""

    def __init__(self, embodiment: EmbodimentCard, backend: DryRunSimBackend) -> None:
        self._embodiment = embodiment
        self._backend = backend

    def set_posture(
        self,
        group: str,
        posture_name: str,
        control_mode: str = "position",
    ) -> tuple[float, ...]:
        if group not in self._embodiment.actuator_groups:
            raise KeyError(f"Unknown actuator group: {group!r}")
        grp = self._embodiment.actuator_groups[group]
        if control_mode not in grp.control_modes:
            raise ContractValidationError(
                f"Actuator group {group!r} does not support mode {control_mode!r}"
            )
        if posture_name not in grp.named_postures:
            raise KeyError(f"Actuator group {group!r} has no posture {posture_name!r}")
        targets = grp.named_postures[posture_name]
        self._backend.record_actuator_command(group, targets, control_mode)
        return targets

    def command_targets(
        self,
        group: str,
        targets: Sequence[float],
        control_mode: str = "position",
    ) -> tuple[float, ...]:
        if group not in self._embodiment.actuator_groups:
            raise KeyError(f"Unknown actuator group: {group!r}")
        grp = self._embodiment.actuator_groups[group]
        if control_mode not in grp.control_modes:
            raise ContractValidationError(
                f"Actuator group {group!r} does not support mode {control_mode!r}"
            )
        if len(targets) != len(grp.dof_indices):
            raise ContractValidationError(
                f"Actuator group {group!r} expected {len(grp.dof_indices)} targets, "
                f"got {len(targets)}"
            )
        vec = tuple(float(v) for v in targets)
        self._backend.record_actuator_command(group, vec, control_mode)
        return vec


class SensorsAPI:
    """Standardized sensorimotor feedback streams (proprio, wrench, tactile)."""

    def __init__(self, embodiment: EmbodimentCard, backend: DryRunSimBackend) -> None:
        self._embodiment = embodiment
        self._backend = backend

    def proprio(self) -> tuple[float, ...]:
        return self._backend.current_joint_state

    def wrist_wrench(self, chain: str) -> WrenchReading:
        if chain not in self._embodiment.chains:
            raise KeyError(f"Unknown kinematic chain: {chain!r}")
        return WrenchReading(
            chain_id=chain,
            force_n=(0.0, 0.0, 0.0),
            torque_nm=(0.0, 0.0, 0.0),
        )

    def tactile(self, group: str) -> TactileReading:
        if group not in self._embodiment.actuator_groups:
            raise KeyError(f"Unknown actuator group: {group!r}")
        return TactileReading(
            group_id=group,
            normal_force_n=0.0,
            shear_force_n=(0.0, 0.0),
            slip_detected=False,
            in_contact=False,
        )


class CheckpointAPI:
    """Snapshot save/restore API strictly gated to privileged_search mode."""

    def __init__(self, mode: str, backend: DryRunSimBackend) -> None:
        self._mode = mode
        self._backend = backend

    def save_snapshot(self, label: str) -> SnapshotHandle:
        if self._mode != "privileged_search":
            raise FirewallViolationError(
                f"save_snapshot is forbidden in {self._mode!r} mode"
            )
        return self._backend.save_snapshot(label)

    def restore_snapshot(self, handle: SnapshotHandle) -> None:
        if self._mode != "privileged_search":
            raise FirewallViolationError(
                f"restore_snapshot is forbidden in {self._mode!r} mode"
            )
        self._backend.restore_snapshot(handle)


class ControllerContext:
    """Top-level context injected into `solve(ctx)`."""

    def __init__(
        self,
        *,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        mode: str,
        backend: DryRunSimBackend,
    ) -> None:
        self.world = WorldAPI(brief)
        self.kinematics = KinematicsAPI(embodiment, backend)
        self.end_effector = EndEffectorAPI(embodiment, backend)
        self.sensors = SensorsAPI(embodiment, backend)
        self.checkpoint = CheckpointAPI(mode, backend)
        self._backend = backend

    def step(self, steps: int = 1) -> None:
        if steps <= 0:
            raise ContractValidationError("ctx.step(steps) requires steps > 0")
        self._backend.step(steps)


class DryRunSimBackend:
    """Lightweight pure-Python cloud preflight backend for controller validation."""

    def __init__(
        self,
        *,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        mode: str,
        max_steps: int = 2000,
    ) -> None:
        if mode not in ALLOWED_EXECUTION_MODES:
            raise ContractValidationError(f"Unsupported execution mode: {mode!r}")
        self.brief = brief
        self.embodiment = embodiment
        self.mode = mode
        self.max_steps = max_steps
        self.steps_executed = 0
        self.snapshots_used = 0
        self._snapshots: dict[str, int] = {}
        self.chain_commands: list[tuple[str, Pose3D, bool]] = []
        self.actuator_commands: list[tuple[str, tuple[float, ...], str]] = []
        init_len = embodiment.action_dim
        self.current_joint_state: tuple[float, ...] = tuple(0.0 for _ in range(init_len))

    def record_chain_command(self, chain: str, pose: Pose3D, *, check_collision: bool) -> None:
        self.chain_commands.append((chain, pose, check_collision))

    def record_actuator_command(
        self, group: str, targets: tuple[float, ...], control_mode: str
    ) -> None:
        self.actuator_commands.append((group, targets, control_mode))

    def save_snapshot(self, label: str) -> SnapshotHandle:
        self.snapshots_used += 1
        snap_id = f"snap_{self.snapshots_used}"
        self._snapshots[snap_id] = self.steps_executed
        return SnapshotHandle(snapshot_id=snap_id, label=label, step_index=self.steps_executed)

    def restore_snapshot(self, handle: SnapshotHandle) -> None:
        if handle.snapshot_id not in self._snapshots:
            raise ContractValidationError(f"Unknown snapshot_id: {handle.snapshot_id!r}")
        self.snapshots_used += 1

    def step(self, steps: int) -> None:
        if self.steps_executed + steps > self.max_steps:
            raise ContractValidationError(
                f"Controller exceeded max_steps={self.max_steps}"
            )
        self.steps_executed += steps


def _safe_import(
    name: str,
    globals: Mapping[str, Any] | None = None,
    locals: Mapping[str, Any] | None = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> Any:
    root = name.split(".")[0]
    if root not in {"math", "numpy", "trajectory_forge", "typing"}:
        raise ImportError(f"Import of {name!r} is forbidden in controller sandbox")
    return __import__(name, globals, locals, fromlist, level)


def _restricted_globals() -> dict[str, Any]:
    safe_builtins = {
        "__import__": _safe_import,
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
        "Exception": Exception,
        "ValueError": ValueError,
        "RuntimeError": RuntimeError,
    }
    return {"__builtins__": safe_builtins}


def run_controller_preflight(
    controller_code: str,
    brief: ScenarioIntakeBrief,
    embodiment: EmbodimentCard,
    *,
    mode: str = "privileged_search",
    max_steps: int = 2000,
    job_id: str = "preflight",
) -> RolloutJobResult:
    """Validate and dry-run a Python controller script against ScenarioIntakeBrief."""
    validate_controller_source(controller_code)
    backend = DryRunSimBackend(
        brief=brief,
        embodiment=embodiment,
        mode=mode,
        max_steps=max_steps,
    )
    ctx = ControllerContext(
        brief=brief,
        embodiment=embodiment,
        mode=mode,
        backend=backend,
    )
    ns = _restricted_globals()
    compiled = compile(controller_code, "<controller>", "exec")
    exec(compiled, ns, ns)
    solve_fn = ns.get("solve")
    if not callable(solve_fn):
        raise ContractValidationError("Controller missing callable solve(ctx)")
    solve_fn(ctx)

    steps = max(1, backend.steps_executed)
    stages: list[StageProgressEntry] = []
    num_stages = len(brief.stages)
    for idx, st in enumerate(brief.stages):
        score = round((idx + 1) / num_stages, 4) if num_stages > 0 else 1.0
        stages.append(
            StageProgressEntry(
                stage_index=st.stage_index,
                metric_id=st.metric_id,
                score=score,
                step_index=steps,
            )
        )

    firewall_verified = mode in {"full_horizon_gate", "trajectory_synthesis"} and (
        backend.snapshots_used == 0
    )
    manifest = None
    if firewall_verified:
        channels = tuple(
            sid
            for sid, stream in embodiment.sensor_streams.items()
            if not stream.privileged
        )
        manifest = build_trajectory_episode_manifest(
            episode_id=f"{brief.package_id}_{job_id}",
            scenario_id=brief.package_id,
            embodiment_id=embodiment.embodiment_id,
            num_steps=steps,
            continuous_from_t0=True,
            privileged_channels_recorded=False,
            recorded_channels=channels or ("proprio",),
        )

    return RolloutJobResult(
        job_id=job_id,
        status="succeeded",
        mode=mode,
        final_score=1.0,
        steps_executed=steps,
        snapshots_used=backend.snapshots_used,
        firewall_verified=firewall_verified,
        stage_progression=tuple(stages),
        diagnostics=SolvabilityDiagnostics(),
        trajectory_manifest=manifest,
    )
