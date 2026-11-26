from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

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
from trajectory_forge.sdk.safety import validate_controller_source
from trajectory_forge.trajectory.manifest import build_trajectory_episode_manifest


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


@dataclass(frozen=True)
class AABB3D:
    """World-space axis-aligned bounding box in meters."""

    min_xyz: Vec3
    max_xyz: Vec3

    @property
    def center_xyz(self) -> Vec3:
        return (
            0.5 * (self.min_xyz[0] + self.max_xyz[0]),
            0.5 * (self.min_xyz[1] + self.max_xyz[1]),
            0.5 * (self.min_xyz[2] + self.max_xyz[2]),
        )

    @property
    def extents_xyz(self) -> Vec3:
        return (
            self.max_xyz[0] - self.min_xyz[0],
            self.max_xyz[1] - self.min_xyz[1],
            self.max_xyz[2] - self.min_xyz[2],
        )


@dataclass(frozen=True)
class JointAxisInfo:
    """Live world-space hinge/slide geometry and drive bounds for an articulated DOF."""

    object_id: str
    semantic_joint: str
    dof_index: int
    joint_type: str
    pivot_world: Vec3
    axis_world: Vec3
    lower_limit: float
    upper_limit: float
    current_position: float


class SimBackendProtocol(Protocol):
    """Backend protocol implemented by both DryRunSimBackend and IsaacGenManipSimBackend."""

    brief: ScenarioIntakeBrief
    embodiment: EmbodimentCard
    mode: str
    steps_executed: int
    snapshots_used: int
    controller_logs: list[str]

    def live_object_pose(self, object_id: str) -> Pose3D: ...
    def live_object_aabb(self, object_id: str) -> AABB3D: ...
    def live_part_pose(self, part_id: str) -> Pose3D: ...
    def live_part_aabb(self, part_id: str) -> AABB3D: ...
    def live_joint_positions(self, object_id: str) -> tuple[float, ...]: ...
    def live_joint_value(self, object_id: str, semantic_joint: str) -> float: ...
    def live_joint_info(self, object_id: str, semantic_joint: str) -> JointAxisInfo: ...
    def live_stage_scores(self) -> tuple[float, ...]: ...
    def live_current_score(self) -> float: ...
    def live_contacts_summary(self) -> dict[str, Any]: ...
    def live_fluid_containment_ratio(
        self,
        vessel_id: str,
        *,
        particle_prim_path: str | None = None,
        fallback: float = 0.0,
    ) -> float: ...
    def ee_pose(self, chain: str) -> Pose3D: ...
    def solve_ik(
        self, chain: str, target_pose: Pose3D, *, check_collision: bool = True
    ) -> tuple[float, ...]: ...
    def move_to_pose(
        self,
        chain: str,
        target_pose: Pose3D,
        *,
        steps: int = 40,
        use_motion_gen: bool = False,
        check_collision: bool = True,
    ) -> tuple[float, ...]: ...
    def follow_cartesian_path(
        self,
        chain: str,
        poses: Sequence[Pose3D],
        *,
        steps_per_waypoint: int = 3,
    ) -> tuple[float, ...]: ...
    def move_joints(
        self,
        chain: str,
        joint_positions: Sequence[float],
        *,
        steps: int = 30,
    ) -> tuple[float, ...]: ...
    def move_base_world(
        self,
        dx_m: float,
        dy_m: float,
        dyaw_deg: float = 0.0,
        *,
        steps: int = 30,
    ) -> tuple[float, float, float]: ...
    def base_pose_world(self) -> Pose3D: ...
    def set_torso_lift(self, height_m: float, *, steps: int = 20) -> float: ...
    def torso_lift(self) -> float: ...
    def command_actuator(
        self,
        group: str,
        targets: tuple[float, ...],
        *,
        control_mode: str = "position",
        steps: int = 16,
    ) -> None: ...
    def current_proprio(self) -> tuple[float, ...]: ...
    def wrist_wrench(self, chain: str) -> WrenchReading: ...
    def tactile(self, group: str) -> TactileReading: ...
    def save_snapshot(self, label: str) -> SnapshotHandle: ...
    def restore_snapshot(self, handle: SnapshotHandle) -> None: ...
    def reset_to_t0(self) -> None: ...
    def step(self, steps: int) -> None: ...


class WorldAPI:
    """Privileged live simulator state API exposed to the agent controller."""

    def __init__(self, brief: ScenarioIntakeBrief, backend: SimBackendProtocol) -> None:
        self._brief = brief
        self._backend = backend

    @property
    def instruction(self) -> str:
        return self._brief.instruction

    def object_ids(self) -> tuple[str, ...]:
        return tuple(self._brief.objects.keys())

    def part_ids(self) -> tuple[str, ...]:
        return self._brief.part_ids

    def object_pose(self, object_id: str) -> Pose3D:
        return self._backend.live_object_pose(object_id)

    def object_aabb(self, object_id: str) -> AABB3D:
        return self._backend.live_object_aabb(object_id)

    def part_pose(self, part_id: str) -> Pose3D:
        return self._backend.live_part_pose(part_id)

    def part_aabb(self, part_id: str) -> AABB3D:
        return self._backend.live_part_aabb(part_id)

    def named_frame(self, object_id: str, frame_name: str) -> Pose3D:
        if object_id not in self._brief.objects:
            raise KeyError(f"Unknown object_id: {object_id!r}")
        live_obj_pose = self._backend.live_object_pose(object_id)
        return pose_from_named_frame(
            self._brief.objects[object_id],
            frame_name,
            object_world_pose=live_obj_pose,
        )

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

    def joint_positions(self, object_id: str) -> tuple[float, ...]:
        return self._backend.live_joint_positions(object_id)

    def joint_value(self, object_id: str, semantic_joint: str) -> float:
        return self._backend.live_joint_value(object_id, semantic_joint)

    def joint_info(self, object_id: str, semantic_joint: str) -> JointAxisInfo:
        return self._backend.live_joint_info(object_id, semantic_joint)

    def stage_scores(self) -> tuple[float, ...]:
        return self._backend.live_stage_scores()

    def is_stage_satisfied(self, stage_index: int) -> bool:
        scores = self._backend.live_stage_scores()
        if stage_index < 0 or stage_index >= len(scores):
            raise IndexError(f"stage_index {stage_index} out of range (0..{len(scores) - 1})")
        return scores[stage_index] >= 0.9999

    def current_score(self) -> float:
        return self._backend.live_current_score()

    def contacts_summary(self) -> dict[str, Any]:
        return self._backend.live_contacts_summary()

    def fluid_containment_ratio(
        self,
        vessel_id: str,
        *,
        particle_prim_path: str | None = None,
        fallback: float = 0.0,
    ) -> float:
        return float(
            self._backend.live_fluid_containment_ratio(
                vessel_id,
                particle_prim_path=particle_prim_path,
                fallback=float(fallback),
            )
        )


class KinematicsAPI:
    """Chain-agnostic kinematics, IK, Cartesian motion, and base/torso control."""

    def __init__(self, embodiment: EmbodimentCard, backend: SimBackendProtocol) -> None:
        self._embodiment = embodiment
        self._backend = backend

    def _validate_chain(self, chain: str) -> None:
        if chain not in self._embodiment.chains:
            raise KeyError(f"Unknown kinematic chain: {chain!r}")

    def ee_pose(self, chain: str) -> Pose3D:
        self._validate_chain(chain)
        return self._backend.ee_pose(chain)

    def solve_ik(
        self,
        chain: str,
        target_pose: Pose3D,
        check_collision: bool = True,
    ) -> tuple[float, ...]:
        self._validate_chain(chain)
        return self._backend.solve_ik(chain, target_pose, check_collision=check_collision)

    def move_to_pose(
        self,
        chain: str,
        target_pose: Pose3D,
        *,
        steps: int = 40,
        use_motion_gen: bool = False,
        check_collision: bool = True,
    ) -> tuple[float, ...]:
        self._validate_chain(chain)
        if steps <= 0:
            raise ContractValidationError("move_to_pose requires steps > 0")
        return self._backend.move_to_pose(
            chain,
            target_pose,
            steps=steps,
            use_motion_gen=use_motion_gen,
            check_collision=check_collision,
        )

    def follow_cartesian_path(
        self,
        chain: str,
        poses: Sequence[Pose3D],
        *,
        steps_per_waypoint: int = 3,
    ) -> tuple[float, ...]:
        self._validate_chain(chain)
        if not poses:
            raise ContractValidationError("follow_cartesian_path requires non-empty poses")
        if steps_per_waypoint <= 0:
            raise ContractValidationError("steps_per_waypoint must be > 0")
        return self._backend.follow_cartesian_path(
            chain, poses, steps_per_waypoint=steps_per_waypoint
        )

    def move_joints(
        self,
        chain: str,
        joint_positions: Sequence[float],
        *,
        steps: int = 30,
    ) -> tuple[float, ...]:
        self._validate_chain(chain)
        spec = self._embodiment.chains[chain]
        if len(joint_positions) != len(spec.dof_indices):
            raise ContractValidationError(
                f"Chain {chain!r} expects {len(spec.dof_indices)} joints, got {len(joint_positions)}"
            )
        return self._backend.move_joints(
            chain, tuple(float(v) for v in joint_positions), steps=steps
        )

    def move_base_world(
        self,
        dx_m: float,
        dy_m: float,
        dyaw_deg: float = 0.0,
        *,
        steps: int = 30,
    ) -> tuple[float, float, float]:
        if self._embodiment.base_locked and (
            abs(dx_m) > 1e-9 or abs(dy_m) > 1e-9 or abs(dyaw_deg) > 1e-9
        ):
            raise ContractValidationError(
                f"Embodiment {self._embodiment.embodiment_id!r} has base_locked=True; "
                "move_base_world with non-zero displacement is forbidden"
            )
        if steps <= 0:
            raise ContractValidationError("move_base_world requires steps > 0")
        return self._backend.move_base_world(
            float(dx_m), float(dy_m), float(dyaw_deg), steps=steps
        )

    def base_pose_world(self) -> Pose3D:
        return self._backend.base_pose_world()

    def set_torso_lift(self, height_m: float, *, steps: int = 20) -> float:
        if (
            self._embodiment.base_locked
            and "torso" not in self._embodiment.chains
            and abs(float(height_m) - 0.46) > 1e-4
        ):
            raise ContractValidationError(
                f"Embodiment {self._embodiment.embodiment_id!r} locks torso lift at 0.46 m "
                f"under base_locked=True (got {float(height_m):.4f} m)"
            )
        if steps <= 0:
            raise ContractValidationError("set_torso_lift requires steps > 0")
        return self._backend.set_torso_lift(float(height_m), steps=steps)

    def torso_lift(self) -> float:
        return self._backend.torso_lift()

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
        self._validate_chain(chain)
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
        self._validate_chain(chain)
        return sample_prismatic_sweep_poses(
            start_pose=start_pose,
            direction_world=direction_world,
            distance_m=distance_m,
            num_waypoints=num_waypoints,
        )


class EndEffectorAPI:
    """Cross-embodiment actuator group API (parallel grippers and dexterous hands)."""

    def __init__(self, embodiment: EmbodimentCard, backend: SimBackendProtocol) -> None:
        self._embodiment = embodiment
        self._backend = backend

    def set_posture(
        self,
        group: str,
        posture_name: str,
        control_mode: str = "position",
        *,
        steps: int = 16,
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
        self._backend.command_actuator(group, targets, control_mode=control_mode, steps=steps)
        return targets

    def command_targets(
        self,
        group: str,
        targets: Sequence[float],
        control_mode: str = "position",
        *,
        steps: int = 16,
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
        self._backend.command_actuator(group, vec, control_mode=control_mode, steps=steps)
        return vec


class SensorsAPI:
    """Standardized sensorimotor feedback streams (proprio, wrench, tactile)."""

    def __init__(self, embodiment: EmbodimentCard, backend: SimBackendProtocol) -> None:
        self._embodiment = embodiment
        self._backend = backend

    def proprio(self) -> tuple[float, ...]:
        return self._backend.current_proprio()

    def gripper_widths(self, group: str) -> tuple[float, ...]:
        if group not in self._embodiment.actuator_groups:
            raise KeyError(f"Unknown actuator group: {group!r}")
        grp = self._embodiment.actuator_groups[group]
        q = self._backend.current_proprio()
        return tuple(float(q[idx]) for idx in grp.dof_indices)

    def wrist_wrench(self, chain: str) -> WrenchReading:
        if chain not in self._embodiment.chains:
            raise KeyError(f"Unknown kinematic chain: {chain!r}")
        return self._backend.wrist_wrench(chain)

    def tactile(self, group: str) -> TactileReading:
        if group not in self._embodiment.actuator_groups:
            raise KeyError(f"Unknown actuator group: {group!r}")
        return self._backend.tactile(group)


class CheckpointAPI:
    """Snapshot save/restore API strictly gated to privileged_search mode."""

    def __init__(self, mode: str, backend: SimBackendProtocol) -> None:
        self._mode = mode
        self._backend = backend

    def save_snapshot(self, label: str) -> SnapshotHandle:
        if self._mode != "privileged_search":
            raise FirewallViolationError(
                f"save_snapshot is forbidden in {self._mode!r} mode"
            )
        return self._backend.save_snapshot(label)

    def save_if_search(self, label: str) -> SnapshotHandle | None:
        """Save a checkpoint during privileged_search; harmlessly no-op in continuous gate/synthesis."""
        if self._mode != "privileged_search":
            return None
        return self._backend.save_snapshot(label)

    def restore_snapshot(self, handle: SnapshotHandle) -> None:
        if self._mode != "privileged_search":
            raise FirewallViolationError(
                f"restore_snapshot is forbidden in {self._mode!r} mode"
            )
        self._backend.restore_snapshot(handle)

    def reset_to_t0(self) -> None:
        if self._mode != "privileged_search":
            raise FirewallViolationError(
                f"reset_to_t0 is forbidden in {self._mode!r} mode"
            )
        self._backend.reset_to_t0()


class ControllerContext:
    """Top-level context injected into `solve(ctx)`."""

    def __init__(
        self,
        *,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        mode: str,
        backend: SimBackendProtocol,
    ) -> None:
        self.mode = mode
        self.world = WorldAPI(brief, backend)
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
        self.controller_logs: list[str] = []
        self._snapshots: dict[str, tuple[int, tuple[float, ...], float, tuple[float, float, float]]] = {}
        self.saved_snapshots: list[SnapshotHandle] = []
        self.chain_commands: list[tuple[str, Pose3D, bool]] = []
        self.actuator_commands: list[tuple[str, tuple[float, ...], str]] = []
        init_len = embodiment.action_dim
        self.current_joint_state: list[float] = [0.0 for _ in range(init_len)]
        if brief.robot_spawn.joint_positions and len(brief.robot_spawn.joint_positions) == init_len:
            self.current_joint_state = [float(v) for v in brief.robot_spawn.joint_positions]
        self._torso_lift_m = 0.46
        self._base_xyz = list(brief.robot_spawn.xyz)
        self._ee_poses: dict[str, Pose3D] = {
            cid: Pose3D(xyz=(0.0 if cid == "left_arm" else 0.25, -0.45, 0.95))
            for cid in embodiment.chains
        }

    def live_object_pose(self, object_id: str) -> Pose3D:
        if object_id not in self.brief.objects:
            raise KeyError(f"Unknown object_id: {object_id!r}")
        return Pose3D.from_spec(self.brief.objects[object_id].initial_pose)

    def live_object_aabb(self, object_id: str) -> AABB3D:
        p = self.live_object_pose(object_id).xyz
        return AABB3D(
            min_xyz=(p[0] - 0.05, p[1] - 0.05, p[2] - 0.05),
            max_xyz=(p[0] + 0.05, p[1] + 0.05, p[2] + 0.05),
        )

    def live_part_pose(self, part_id: str) -> Pose3D:
        if part_id not in self.brief.part_ids and part_id not in self.brief.objects:
            raise KeyError(f"Unknown part_id: {part_id!r}")
        return Pose3D(xyz=(0.0, 0.0, 0.8))

    def live_part_aabb(self, part_id: str) -> AABB3D:
        p = self.live_part_pose(part_id).xyz
        return AABB3D(
            min_xyz=(p[0] - 0.03, p[1] - 0.03, p[2] - 0.03),
            max_xyz=(p[0] + 0.03, p[1] + 0.03, p[2] + 0.03),
        )

    def live_joint_positions(self, object_id: str) -> tuple[float, ...]:
        if object_id not in self.brief.objects:
            raise KeyError(f"Unknown object_id: {object_id!r}")
        obj = self.brief.objects[object_id]
        if obj.reset_joint_positions:
            return obj.reset_joint_positions
        n_dofs = max((d.dof_index + 1 for d in obj.dofs.values()), default=0)
        return tuple(0.0 for _ in range(n_dofs))

    def live_joint_value(self, object_id: str, semantic_joint: str) -> float:
        if object_id not in self.brief.objects:
            raise KeyError(f"Unknown object_id: {object_id!r}")
        obj = self.brief.objects[object_id]
        if semantic_joint not in obj.dofs:
            raise KeyError(f"Object {object_id!r} has no dof {semantic_joint!r}")
        positions = self.live_joint_positions(object_id)
        idx = obj.dofs[semantic_joint].dof_index
        return float(positions[idx]) if idx < len(positions) else 0.0

    def live_joint_info(self, object_id: str, semantic_joint: str) -> JointAxisInfo:
        val = self.live_joint_value(object_id, semantic_joint)
        obj = self.brief.objects[object_id]
        dof = obj.dofs[semantic_joint]
        p = self.live_object_pose(object_id).xyz
        return JointAxisInfo(
            object_id=object_id,
            semantic_joint=semantic_joint,
            dof_index=dof.dof_index,
            joint_type="revolute",
            pivot_world=p,
            axis_world=(1.0, 0.0, 0.0),
            lower_limit=-1.5,
            upper_limit=1.5,
            current_position=val,
        )

    def live_stage_scores(self) -> tuple[float, ...]:
        return tuple(0.0 for _ in self.brief.stages)

    def live_current_score(self) -> float:
        scores = self.live_stage_scores()
        return sum(scores) / len(scores) if scores else 0.0

    def live_contacts_summary(self) -> dict[str, Any]:
        return {"backend": "dry_run_preflight", "active_contacts": []}

    def live_fluid_containment_ratio(
        self,
        vessel_id: str,
        *,
        particle_prim_path: str | None = None,
        fallback: float = 0.0,
    ) -> float:
        _ = (vessel_id, particle_prim_path)
        return float(fallback)

    def ee_pose(self, chain: str) -> Pose3D:
        return self._ee_poses[chain]

    def solve_ik(
        self, chain: str, target_pose: Pose3D, *, check_collision: bool = True
    ) -> tuple[float, ...]:
        self.chain_commands.append((chain, target_pose, check_collision))
        spec = self.embodiment.chains[chain]
        return tuple(self.current_joint_state[i] for i in spec.dof_indices)

    def move_to_pose(
        self,
        chain: str,
        target_pose: Pose3D,
        *,
        steps: int = 40,
        use_motion_gen: bool = False,
        check_collision: bool = True,
    ) -> tuple[float, ...]:
        _ = use_motion_gen
        q = self.solve_ik(chain, target_pose, check_collision=check_collision)
        self._ee_poses[chain] = target_pose
        self.step(steps)
        return q

    def follow_cartesian_path(
        self,
        chain: str,
        poses: Sequence[Pose3D],
        *,
        steps_per_waypoint: int = 3,
    ) -> tuple[float, ...]:
        last_q: tuple[float, ...] = ()
        for pose in poses:
            last_q = self.solve_ik(chain, pose, check_collision=True)
            self._ee_poses[chain] = pose
            self.step(steps_per_waypoint)
        return last_q

    def move_joints(
        self,
        chain: str,
        joint_positions: Sequence[float],
        *,
        steps: int = 30,
    ) -> tuple[float, ...]:
        spec = self.embodiment.chains[chain]
        for idx, val in zip(spec.dof_indices, joint_positions):
            self.current_joint_state[idx] = float(val)
        self.step(steps)
        return tuple(float(v) for v in joint_positions)

    def move_base_world(
        self,
        dx_m: float,
        dy_m: float,
        dyaw_deg: float = 0.0,
        *,
        steps: int = 30,
    ) -> tuple[float, float, float]:
        _ = dyaw_deg
        self._base_xyz[0] += float(dx_m)
        self._base_xyz[1] += float(dy_m)
        self.step(steps)
        return (self._base_xyz[0], self._base_xyz[1], 0.0)

    def base_pose_world(self) -> Pose3D:
        return Pose3D(
            xyz=(self._base_xyz[0], self._base_xyz[1], self._base_xyz[2]),
            wxyz=self.brief.robot_spawn.wxyz,
        )

    def set_torso_lift(self, height_m: float, *, steps: int = 20) -> float:
        self._torso_lift_m = float(height_m)
        self.step(steps)
        return self._torso_lift_m

    def torso_lift(self) -> float:
        return self._torso_lift_m

    def command_actuator(
        self,
        group: str,
        targets: tuple[float, ...],
        *,
        control_mode: str = "position",
        steps: int = 16,
    ) -> None:
        self.actuator_commands.append((group, targets, control_mode))
        grp = self.embodiment.actuator_groups[group]
        for idx, val in zip(grp.dof_indices, targets):
            self.current_joint_state[idx] = float(val)
        self.step(steps)

    def current_proprio(self) -> tuple[float, ...]:
        return tuple(self.current_joint_state)

    def wrist_wrench(self, chain: str) -> WrenchReading:
        return WrenchReading(chain_id=chain, force_n=(0.0, 0.0, 0.0), torque_nm=(0.0, 0.0, 0.0))

    def tactile(self, group: str) -> TactileReading:
        return TactileReading(
            group_id=group,
            normal_force_n=0.0,
            shear_force_n=(0.0, 0.0),
            slip_detected=False,
            in_contact=False,
        )

    def save_snapshot(self, label: str) -> SnapshotHandle:
        self.snapshots_used += 1
        snap_id = f"snap_{self.snapshots_used}"
        self._snapshots[snap_id] = (
            self.steps_executed,
            tuple(self.current_joint_state),
            self._torso_lift_m,
            (self._base_xyz[0], self._base_xyz[1], self._base_xyz[2]),
        )
        handle = SnapshotHandle(snapshot_id=snap_id, label=label, step_index=self.steps_executed)
        self.saved_snapshots.append(handle)
        return handle

    def restore_snapshot(self, handle: SnapshotHandle) -> None:
        if handle.snapshot_id not in self._snapshots:
            raise ContractValidationError(f"Unknown snapshot_id: {handle.snapshot_id!r}")
        _, joints, lift_m, base_xyz = self._snapshots[handle.snapshot_id]
        self.current_joint_state = list(joints)
        self._torso_lift_m = lift_m
        self._base_xyz = list(base_xyz)
        self.snapshots_used += 1

    def reset_to_t0(self) -> None:
        self.snapshots_used += 1
        self._torso_lift_m = 0.46
        self._base_xyz = list(self.brief.robot_spawn.xyz)
        if self.brief.robot_spawn.joint_positions:
            self.current_joint_state = [float(v) for v in self.brief.robot_spawn.joint_positions]
        else:
            self.current_joint_state = [0.0 for _ in range(self.embodiment.action_dim)]

    def step(self, steps: int) -> None:
        if self.steps_executed + steps > self.max_steps:
            raise ContractValidationError(f"Controller exceeded max_steps={self.max_steps}")
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


def restricted_globals(log_sink: list[str] | None = None) -> dict[str, Any]:
    """Build restricted execution globals for agent-authored `solve(ctx)` scripts."""

    def _bounded_print(*args: object, sep: str = " ", end: str = "") -> None:
        if log_sink is None or len(log_sink) >= 200:
            return
        msg = (sep.join(str(a) for a in args) + end).strip()
        if msg:
            log_sink.append(msg[:500])

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
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": _bounded_print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "AssertionError": AssertionError,
        "Exception": Exception,
        "KeyError": KeyError,
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
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
    ns = restricted_globals(backend.controller_logs)
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
        saved_snapshots=tuple(backend.saved_snapshots),
        state_summary={
            "backend": "dry_run_preflight",
            "controller_logs": list(backend.controller_logs),
        },
    )
