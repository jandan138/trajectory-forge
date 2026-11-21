from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


ALLOWED_CONTROL_MODES: frozenset[str] = frozenset(
    {"position", "velocity", "effort", "impedance"}
)
ALLOWED_EXECUTION_MODES: frozenset[str] = frozenset(
    {"privileged_search", "full_horizon_gate", "trajectory_synthesis"}
)
ALLOWED_REMEDIATION_MODES: frozenset[str] = frozenset(
    {"physics_tuning", "scene_codesign"}
)
ALLOWED_REMEDIATION_CATEGORIES: frozenset[str] = frozenset(
    {
        "none",
        "joint_drive",
        "collision_proxy",
        "friction_or_offset",
        "pose_adjustment",
        "affordance_part",
        "fixture_or_layout",
    }
)
ALLOWED_SENSOR_MODALITIES: frozenset[str] = frozenset(
    {"proprio", "wrench", "tactile_summary", "tactile_array", "rgb", "depth"}
)
ALLOWED_ACTUATOR_KINDS: frozenset[str] = frozenset(
    {"parallel_gripper", "dexterous_hand", "suction_end_effector", "custom"}
)


class ContractValidationError(ValueError):
    """Raised when a scenario intake, rollout RPC, or episode contract is invalid."""


class EmbodimentValidationError(ValueError):
    """Raised when an EmbodimentCard specification violates structural invariants."""


class FirewallViolationError(RuntimeError):
    """Raised when privileged snapshots or state leak across the sensorimotor firewall."""


class ControllerSafetyError(ValueError):
    """Raised when an agent-authored Python controller fails static safety checks."""


def _validate_vec(values: tuple[float, ...], expected_len: int, label: str) -> None:
    if len(values) != expected_len:
        raise ContractValidationError(f"{label} must have length {expected_len}, got {len(values)}")


@dataclass(frozen=True)
class PoseSpec:
    """Rigid pose in meters and wxyz quaternion convention."""

    xyz: tuple[float, float, float]
    wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _validate_vec(self.xyz, 3, "PoseSpec.xyz")
        _validate_vec(self.wxyz, 4, "PoseSpec.wxyz")

    def to_dict(self) -> dict[str, Any]:
        return {"xyz": list(self.xyz), "wxyz": list(self.wxyz)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PoseSpec:
        xyz = tuple(float(v) for v in data["xyz"])
        wxyz = tuple(float(v) for v in data.get("wxyz", (1.0, 0.0, 0.0, 0.0)))
        return cls(xyz=(xyz[0], xyz[1], xyz[2]), wxyz=(wxyz[0], wxyz[1], wxyz[2], wxyz[3]))


@dataclass(frozen=True)
class NamedFrameSpec:
    """Authoritative functional coordinate frame attached to a scene object."""

    name: str
    pose_local: PoseSpec
    parent_prim: str = ""
    authoritative: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pose_local": self.pose_local.to_dict(),
            "parent_prim": self.parent_prim,
            "authoritative": self.authoritative,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NamedFrameSpec:
        return cls(
            name=str(data["name"]),
            pose_local=PoseSpec.from_dict(data["pose_local"]),
            parent_prim=str(data.get("parent_prim", "")),
            authoritative=bool(data.get("authoritative", True)),
        )


@dataclass(frozen=True)
class ArticulationDofSpec:
    """Semantic articulation joint mapping and target state bands."""

    semantic_joint: str
    dof_index: int
    joint_prim: str
    states: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_joint": self.semantic_joint,
            "dof_index": self.dof_index,
            "joint_prim": self.joint_prim,
            "states": {k: list(v) for k, v in self.states.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArticulationDofSpec:
        raw_states = data.get("states", {})
        states = {
            str(k): (float(v[0]), float(v[1]))
            for k, v in raw_states.items()
        }
        return cls(
            semantic_joint=str(data["semantic_joint"]),
            dof_index=int(data["dof_index"]),
            joint_prim=str(data["joint_prim"]),
            states=states,
        )


@dataclass(frozen=True)
class DeformableContractSpec:
    """Deformable integrity & cross-engine material contract for an object."""

    asset_id: str
    structural_family: str
    youngs_modulus_pa: float
    poissons_ratio: float
    strain_metric: str
    strain_onset_a: float
    strain_failure_b: float
    damage_threshold: float = 0.25
    max_normal_force_n: float = 60.0
    mass_kg: float = 0.1
    dimensions_m: tuple[float, float, float] = (0.07, 0.07, 0.028)
    engine_configs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ContractValidationError("DeformableContractSpec.asset_id must not be empty")
        if self.structural_family not in {"soft_solid", "cloth_sheet", "rod_cable", "shell"}:
            raise ContractValidationError(
                f"Unsupported structural_family {self.structural_family!r}; expected one of soft_solid, cloth_sheet, rod_cable, shell"
            )
        if self.strain_onset_a < 0.0 or self.strain_failure_b <= self.strain_onset_a:
            raise ContractValidationError(
                "DeformableContractSpec requires 0 <= strain_onset_a < strain_failure_b"
            )
        if not (0.0 < self.damage_threshold <= 1.0):
            raise ContractValidationError("DeformableContractSpec.damage_threshold must lie in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "structural_family": self.structural_family,
            "youngs_modulus_pa": self.youngs_modulus_pa,
            "poissons_ratio": self.poissons_ratio,
            "strain_metric": self.strain_metric,
            "strain_onset_a": self.strain_onset_a,
            "strain_failure_b": self.strain_failure_b,
            "damage_threshold": self.damage_threshold,
            "max_normal_force_n": self.max_normal_force_n,
            "mass_kg": self.mass_kg,
            "dimensions_m": list(self.dimensions_m),
            "engine_configs": dict(self.engine_configs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeformableContractSpec:
        dims = tuple(float(v) for v in data.get("dimensions_m", (0.07, 0.07, 0.028)))
        return cls(
            asset_id=str(data["asset_id"]),
            structural_family=str(data["structural_family"]),
            youngs_modulus_pa=float(data["youngs_modulus_pa"]),
            poissons_ratio=float(data["poissons_ratio"]),
            strain_metric=str(data["strain_metric"]),
            strain_onset_a=float(data["strain_onset_a"]),
            strain_failure_b=float(data["strain_failure_b"]),
            damage_threshold=float(data.get("damage_threshold", 0.25)),
            max_normal_force_n=float(data.get("max_normal_force_n", 60.0)),
            mass_kg=float(data.get("mass_kg", 0.1)),
            dimensions_m=(dims[0], dims[1], dims[2]),
            engine_configs={str(k): str(v) for k, v in data.get("engine_configs", {}).items()},
        )


@dataclass(frozen=True)
class ObjectContractSpec:
    """Scene object or articulated hero asset parsed from runtime contract."""

    object_id: str
    runtime_uid: str
    role: str
    state_prim_path: str
    initial_pose: PoseSpec
    named_frames: dict[str, NamedFrameSpec] = field(default_factory=dict)
    dofs: dict[str, ArticulationDofSpec] = field(default_factory=dict)
    reset_joint_positions: tuple[float, ...] = ()
    deformable: DeformableContractSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "object_id": self.object_id,
            "runtime_uid": self.runtime_uid,
            "role": self.role,
            "state_prim_path": self.state_prim_path,
            "initial_pose": self.initial_pose.to_dict(),
            "named_frames": {k: v.to_dict() for k, v in self.named_frames.items()},
            "dofs": {k: v.to_dict() for k, v in self.dofs.items()},
            "reset_joint_positions": list(self.reset_joint_positions),
        }
        if self.deformable is not None:
            payload["deformable"] = self.deformable.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObjectContractSpec:
        raw_def = data.get("deformable")
        return cls(
            object_id=str(data["object_id"]),
            runtime_uid=str(data["runtime_uid"]),
            role=str(data["role"]),
            state_prim_path=str(data["state_prim_path"]),
            initial_pose=PoseSpec.from_dict(data["initial_pose"]),
            named_frames={
                str(k): NamedFrameSpec.from_dict(v)
                for k, v in data.get("named_frames", {}).items()
            },
            dofs={
                str(k): ArticulationDofSpec.from_dict(v)
                for k, v in data.get("dofs", {}).items()
            },
            reset_joint_positions=tuple(float(v) for v in data.get("reset_joint_positions", ())),
            deformable=(
                DeformableContractSpec.from_dict(raw_def)
                if raw_def is not None
                else None
            ),
        )


@dataclass(frozen=True)
class RobotSpawnSpec:
    """Initial robot spawn pose and joint configuration."""

    robot_id: str
    xyz: tuple[float, float, float]
    wxyz: tuple[float, float, float, float]
    joint_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_vec(self.xyz, 3, "RobotSpawnSpec.xyz")
        _validate_vec(self.wxyz, 4, "RobotSpawnSpec.wxyz")

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "xyz": list(self.xyz),
            "wxyz": list(self.wxyz),
            "joint_positions": list(self.joint_positions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RobotSpawnSpec:
        xyz = tuple(float(v) for v in data["xyz"])
        wxyz = tuple(float(v) for v in data["wxyz"])
        return cls(
            robot_id=str(data["robot_id"]),
            xyz=(xyz[0], xyz[1], xyz[2]),
            wxyz=(wxyz[0], wxyz[1], wxyz[2], wxyz[3]),
            joint_positions=tuple(float(v) for v in data.get("joint_positions", ())),
        )


@dataclass(frozen=True)
class StageContractSpec:
    """One sequential evaluation stage from GenManip MetricsManager goal."""

    stage_index: int
    metric_id: str
    clauses: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "metric_id": self.metric_id,
            "clauses": list(self.clauses),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageContractSpec:
        return cls(
            stage_index=int(data["stage_index"]),
            metric_id=str(data["metric_id"]),
            clauses=tuple(dict(c) for c in data.get("clauses", ())),
        )


@dataclass(frozen=True)
class ScenarioIntakeBrief:
    """Normalized intake contract parsed from a scenario-forge collected package."""

    package_id: str
    episode_name: str
    instruction: str
    scene_usd_relpath: str
    robot_profile: str
    robot_spawn: RobotSpawnSpec
    objects: dict[str, ObjectContractSpec]
    stages: tuple[StageContractSpec, ...]
    part_ids: tuple[str, ...] = ()
    schema_version: str = "trajectory-forge-intake-brief/v0.1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "episode_name": self.episode_name,
            "instruction": self.instruction,
            "scene_usd_relpath": self.scene_usd_relpath,
            "robot_profile": self.robot_profile,
            "robot_spawn": self.robot_spawn.to_dict(),
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "stages": [s.to_dict() for s in self.stages],
            "part_ids": list(self.part_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScenarioIntakeBrief:
        return cls(
            schema_version=str(
                data.get("schema_version", "trajectory-forge-intake-brief/v0.1")
            ),
            package_id=str(data["package_id"]),
            episode_name=str(data["episode_name"]),
            instruction=str(data["instruction"]),
            scene_usd_relpath=str(data["scene_usd_relpath"]),
            robot_profile=str(data["robot_profile"]),
            robot_spawn=RobotSpawnSpec.from_dict(data["robot_spawn"]),
            objects={
                str(k): ObjectContractSpec.from_dict(v)
                for k, v in data.get("objects", {}).items()
            },
            stages=tuple(StageContractSpec.from_dict(s) for s in data.get("stages", ())),
            part_ids=tuple(str(p) for p in data.get("part_ids", ())),
        )


@dataclass(frozen=True)
class KinematicChainSpec:
    """Serial arm or finger kinematic chain specification."""

    chain_id: str
    base_frame: str
    ee_frame: str
    dof_indices: tuple[int, ...]
    joint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.dof_indices:
            raise EmbodimentValidationError(f"Chain {self.chain_id!r} must declare dof_indices")
        if len(self.dof_indices) != len(self.joint_names):
            raise EmbodimentValidationError(
                f"Chain {self.chain_id!r} dof_indices/joint_names length mismatch"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KinematicChainSpec:
        return cls(
            chain_id=str(data["chain_id"]),
            base_frame=str(data["base_frame"]),
            ee_frame=str(data["ee_frame"]),
            dof_indices=tuple(int(i) for i in data["dof_indices"]),
            joint_names=tuple(str(n) for n in data["joint_names"]),
        )


@dataclass(frozen=True)
class ActuatorGroupSpec:
    """End-effector actuator group (parallel gripper, dexterous hand, or custom)."""

    group_id: str
    kind: str
    dof_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    control_modes: tuple[str, ...]
    named_postures: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ALLOWED_ACTUATOR_KINDS:
            raise EmbodimentValidationError(
                f"ActuatorGroup {self.group_id!r} invalid kind {self.kind!r}"
            )
        if not self.dof_indices:
            raise EmbodimentValidationError(
                f"ActuatorGroup {self.group_id!r} must declare dof_indices"
            )
        if len(self.dof_indices) != len(self.joint_names):
            raise EmbodimentValidationError(
                f"ActuatorGroup {self.group_id!r} dof_indices/joint_names length mismatch"
            )
        if not self.control_modes:
            raise EmbodimentValidationError(
                f"ActuatorGroup {self.group_id!r} must declare at least one control_mode"
            )
        for mode in self.control_modes:
            if mode not in ALLOWED_CONTROL_MODES:
                raise EmbodimentValidationError(
                    f"ActuatorGroup {self.group_id!r} invalid control mode {mode!r}"
                )
        expected_len = len(self.dof_indices)
        for posture_name, vec in self.named_postures.items():
            if len(vec) != expected_len:
                raise EmbodimentValidationError(
                    f"ActuatorGroup {self.group_id!r} posture {posture_name!r} "
                    f"expected length {expected_len}, got {len(vec)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "kind": self.kind,
            "dof_indices": list(self.dof_indices),
            "joint_names": list(self.joint_names),
            "control_modes": list(self.control_modes),
            "named_postures": {k: list(v) for k, v in self.named_postures.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActuatorGroupSpec:
        return cls(
            group_id=str(data["group_id"]),
            kind=str(data["kind"]),
            dof_indices=tuple(int(i) for i in data["dof_indices"]),
            joint_names=tuple(str(n) for n in data["joint_names"]),
            control_modes=tuple(str(m) for m in data["control_modes"]),
            named_postures={
                str(k): tuple(float(x) for x in v)
                for k, v in data.get("named_postures", {}).items()
            },
        )


@dataclass(frozen=True)
class SensorStreamSpec:
    """Declarative sensor stream channel (proprio, wrench, tactile, rgb, depth)."""

    stream_id: str
    modality: str
    rate_hz: float
    shape: tuple[int, ...]
    attached_group: str = ""
    privileged: bool = False

    def __post_init__(self) -> None:
        if self.modality not in ALLOWED_SENSOR_MODALITIES:
            raise EmbodimentValidationError(
                f"SensorStream {self.stream_id!r} invalid modality {self.modality!r}"
            )
        if self.rate_hz <= 0.0:
            raise EmbodimentValidationError(
                f"SensorStream {self.stream_id!r} rate_hz must be > 0"
            )
        if self.privileged:
            raise EmbodimentValidationError(
                f"SensorStream {self.stream_id!r} cannot be marked privileged in EmbodimentCard"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SensorStreamSpec:
        return cls(
            stream_id=str(data["stream_id"]),
            modality=str(data["modality"]),
            rate_hz=float(data["rate_hz"]),
            shape=tuple(int(d) for d in data["shape"]),
            attached_group=str(data.get("attached_group", "")),
            privileged=bool(data.get("privileged", False)),
        )


@dataclass(frozen=True)
class EmbodimentCard:
    """Declarative robot capability card supporting arms, grippers, dexterous hands, and tactile."""

    embodiment_id: str
    runtime_robot_type: str
    action_dim: int
    chains: dict[str, KinematicChainSpec]
    actuator_groups: dict[str, ActuatorGroupSpec]
    sensor_streams: dict[str, SensorStreamSpec]
    schema_version: str = "trajectory-forge-embodiment-card/v0.1"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise EmbodimentValidationError("EmbodimentCard.action_dim must be > 0")
        seen_dofs: set[int] = set()
        for chain in self.chains.values():
            for idx in chain.dof_indices:
                if idx < 0 or idx >= self.action_dim:
                    raise EmbodimentValidationError(
                        f"Chain {chain.chain_id!r} dof index {idx} out of [0, {self.action_dim})"
                    )
                if idx in seen_dofs:
                    raise EmbodimentValidationError(f"Duplicate DOF index {idx} in EmbodimentCard")
                seen_dofs.add(idx)
        for grp in self.actuator_groups.values():
            for idx in grp.dof_indices:
                if idx < 0 or idx >= self.action_dim:
                    raise EmbodimentValidationError(
                        f"ActuatorGroup {grp.group_id!r} dof index {idx} out of range"
                    )
                if idx in seen_dofs:
                    raise EmbodimentValidationError(f"Duplicate DOF index {idx} in EmbodimentCard")
                seen_dofs.add(idx)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "embodiment_id": self.embodiment_id,
            "runtime_robot_type": self.runtime_robot_type,
            "action_dim": self.action_dim,
            "chains": {k: v.to_dict() for k, v in self.chains.items()},
            "actuator_groups": {k: v.to_dict() for k, v in self.actuator_groups.items()},
            "sensor_streams": {k: v.to_dict() for k, v in self.sensor_streams.items()},
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EmbodimentCard:
        return cls(
            schema_version=str(
                data.get("schema_version", "trajectory-forge-embodiment-card/v0.1")
            ),
            embodiment_id=str(data["embodiment_id"]),
            runtime_robot_type=str(data["runtime_robot_type"]),
            action_dim=int(data["action_dim"]),
            chains={
                str(k): KinematicChainSpec.from_dict(v)
                for k, v in data.get("chains", {}).items()
            },
            actuator_groups={
                str(k): ActuatorGroupSpec.from_dict(v)
                for k, v in data.get("actuator_groups", {}).items()
            },
            sensor_streams={
                str(k): SensorStreamSpec.from_dict(v)
                for k, v in data.get("sensor_streams", {}).items()
            },
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class SnapshotHandle:
    """Opaque reference to a stage physics snapshot during privileged search."""

    snapshot_id: str
    label: str
    step_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SnapshotHandle:
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            label=str(data["label"]),
            step_index=int(data["step_index"]),
        )


@dataclass(frozen=True)
class StageProgressEntry:
    """Recorded stage completion state at a simulation step."""

    stage_index: int
    metric_id: str
    score: float
    step_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageProgressEntry:
        return cls(
            stage_index=int(data["stage_index"]),
            metric_id=str(data["metric_id"]),
            score=float(data["score"]),
            step_index=int(data["step_index"]),
        )


@dataclass(frozen=True)
class RemediationPolicySpec:
    """Policy governing which upstream scene edits may be recommended."""

    mode: str = "physics_tuning"
    allow_pose_adjustment: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ALLOWED_REMEDIATION_MODES:
            raise ContractValidationError(
                f"Unsupported remediation mode: {self.mode!r}"
            )

    @property
    def allows_pose_adjustment(self) -> bool:
        return self.mode == "scene_codesign" or self.allow_pose_adjustment

    @property
    def allows_structural_parts(self) -> bool:
        return self.mode == "scene_codesign"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allow_pose_adjustment": self.allow_pose_adjustment,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> RemediationPolicySpec:
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            mode=str(data.get("mode", "physics_tuning")),
            allow_pose_adjustment=bool(data.get("allow_pose_adjustment", False)),
        )


@dataclass(frozen=True)
class SolvabilityDiagnostics:
    """Structured failure diagnostics fed back to the agent or upstream scenario-forge."""

    failure_code: str = ""
    stage_index: int | None = None
    detail: str = ""
    colliding_pairs: tuple[tuple[str, str], ...] = ()
    upstream_remediation_hint: str = ""
    remediation_category: str = "none"
    required_remediation_mode: str = ""
    requires_pose_adjustment: bool = False
    policy_blocked: bool = False
    root_failure_code: str = ""

    def __post_init__(self) -> None:
        if self.remediation_category not in ALLOWED_REMEDIATION_CATEGORIES:
            raise ContractValidationError(
                f"Unsupported remediation category: {self.remediation_category!r}"
            )
        if self.required_remediation_mode not in ({""} | ALLOWED_REMEDIATION_MODES):
            raise ContractValidationError(
                f"Unsupported required remediation mode: {self.required_remediation_mode!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_code": self.failure_code,
            "stage_index": self.stage_index,
            "detail": self.detail,
            "colliding_pairs": [list(p) for p in self.colliding_pairs],
            "upstream_remediation_hint": self.upstream_remediation_hint,
            "remediation_category": self.remediation_category,
            "required_remediation_mode": self.required_remediation_mode,
            "requires_pose_adjustment": self.requires_pose_adjustment,
            "policy_blocked": self.policy_blocked,
            "root_failure_code": self.root_failure_code,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SolvabilityDiagnostics:
        return cls(
            failure_code=str(data.get("failure_code", "")),
            stage_index=data.get("stage_index"),
            detail=str(data.get("detail", "")),
            colliding_pairs=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in data.get("colliding_pairs", ())
            ),
            upstream_remediation_hint=str(data.get("upstream_remediation_hint", "")),
            remediation_category=str(data.get("remediation_category", "none")),
            required_remediation_mode=str(data.get("required_remediation_mode", "")),
            requires_pose_adjustment=bool(data.get("requires_pose_adjustment", False)),
            policy_blocked=bool(data.get("policy_blocked", False)),
            root_failure_code=str(data.get("root_failure_code", "")),
        )


@dataclass(frozen=True)
class MediaArtifactRef:
    """Reference to an image or video diagnostic artifact produced by a rollout."""

    label: str
    relpath: str
    mime_type: str
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MediaArtifactRef:
        return cls(
            label=str(data["label"]),
            relpath=str(data["relpath"]),
            mime_type=str(data["mime_type"]),
            sha256=str(data.get("sha256", "")),
        )


@dataclass(frozen=True)
class TrajectoryEpisodeManifest:
    """Firewall-checked metadata manifest for a synthesized demonstration episode."""

    episode_id: str
    scenario_id: str
    embodiment_id: str
    num_steps: int
    continuous_from_t0: bool
    privileged_channels_recorded: bool
    recorded_channels: tuple[str, ...]
    schema_version: str = "trajectory-forge-episode/v0.1"
    artifact_relpath: str = ""

    def __post_init__(self) -> None:
        if not self.continuous_from_t0:
            raise FirewallViolationError(
                "TrajectoryEpisodeManifest requires continuous_from_t0=True (no snapshot splicing)"
            )
        if self.privileged_channels_recorded:
            raise FirewallViolationError(
                "TrajectoryEpisodeManifest forbids recording privileged state channels"
            )
        if self.num_steps <= 0:
            raise ContractValidationError("TrajectoryEpisodeManifest.num_steps must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "embodiment_id": self.embodiment_id,
            "num_steps": self.num_steps,
            "continuous_from_t0": self.continuous_from_t0,
            "privileged_channels_recorded": self.privileged_channels_recorded,
            "recorded_channels": list(self.recorded_channels),
            "artifact_relpath": self.artifact_relpath,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrajectoryEpisodeManifest:
        return cls(
            schema_version=str(data.get("schema_version", "trajectory-forge-episode/v0.1")),
            episode_id=str(data["episode_id"]),
            scenario_id=str(data["scenario_id"]),
            embodiment_id=str(data["embodiment_id"]),
            num_steps=int(data["num_steps"]),
            continuous_from_t0=bool(data["continuous_from_t0"]),
            privileged_channels_recorded=bool(data["privileged_channels_recorded"]),
            recorded_channels=tuple(str(c) for c in data.get("recorded_channels", ())),
            artifact_relpath=str(data.get("artifact_relpath", "")),
        )


@dataclass(frozen=True)
class RolloutJobRequest:
    """RPC job submitted from the cloud brain to a local or dry-run rollout worker."""

    job_id: str
    scenario_package_ref: str
    embodiment_id: str
    mode: str
    seed: int
    controller_code: str
    max_steps: int = 2000
    resume_from_snapshot_id: str | None = None
    render_media: bool = False
    remediation_policy: RemediationPolicySpec = field(
        default_factory=RemediationPolicySpec
    )
    schema_version: str = "trajectory-forge-rollout-rpc/v0.1"

    def __post_init__(self) -> None:
        if self.mode not in ALLOWED_EXECUTION_MODES:
            raise ContractValidationError(f"Unsupported execution mode: {self.mode!r}")
        if self.max_steps <= 0:
            raise ContractValidationError("RolloutJobRequest.max_steps must be > 0")
        if self.mode in {"full_horizon_gate", "trajectory_synthesis"}:
            if self.resume_from_snapshot_id is not None:
                raise FirewallViolationError(
                    f"Mode {self.mode!r} forbids resume_from_snapshot_id"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "scenario_package_ref": self.scenario_package_ref,
            "embodiment_id": self.embodiment_id,
            "mode": self.mode,
            "seed": self.seed,
            "controller_code": self.controller_code,
            "max_steps": self.max_steps,
            "resume_from_snapshot_id": self.resume_from_snapshot_id,
            "render_media": self.render_media,
            "remediation_policy": self.remediation_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RolloutJobRequest:
        return cls(
            schema_version=str(
                data.get("schema_version", "trajectory-forge-rollout-rpc/v0.1")
            ),
            job_id=str(data["job_id"]),
            scenario_package_ref=str(data["scenario_package_ref"]),
            embodiment_id=str(data["embodiment_id"]),
            mode=str(data["mode"]),
            seed=int(data["seed"]),
            controller_code=str(data["controller_code"]),
            max_steps=int(data.get("max_steps", 2000)),
            resume_from_snapshot_id=data.get("resume_from_snapshot_id"),
            render_media=bool(data.get("render_media", False)),
            remediation_policy=RemediationPolicySpec.from_dict(
                data.get("remediation_policy")
            ),
        )


@dataclass(frozen=True)
class RolloutJobResult:
    """RPC response returned by a rollout worker after running a controller script."""

    job_id: str
    status: str
    mode: str
    final_score: float
    steps_executed: int
    snapshots_used: int
    firewall_verified: bool
    stage_progression: tuple[StageProgressEntry, ...] = ()
    diagnostics: SolvabilityDiagnostics = field(default_factory=SolvabilityDiagnostics)
    media: tuple[MediaArtifactRef, ...] = ()
    trajectory_manifest: TrajectoryEpisodeManifest | None = None
    saved_snapshots: tuple[SnapshotHandle, ...] = ()
    state_summary: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "trajectory-forge-rollout-rpc/v0.1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "status": self.status,
            "mode": self.mode,
            "final_score": self.final_score,
            "steps_executed": self.steps_executed,
            "snapshots_used": self.snapshots_used,
            "firewall_verified": self.firewall_verified,
            "stage_progression": [s.to_dict() for s in self.stage_progression],
            "diagnostics": self.diagnostics.to_dict(),
            "media": [m.to_dict() for m in self.media],
            "trajectory_manifest": (
                self.trajectory_manifest.to_dict()
                if self.trajectory_manifest is not None
                else None
            ),
            "saved_snapshots": [s.to_dict() for s in self.saved_snapshots],
            "state_summary": dict(self.state_summary),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RolloutJobResult:
        raw_manifest = data.get("trajectory_manifest")
        raw_state = data.get("state_summary", {})
        return cls(
            schema_version=str(
                data.get("schema_version", "trajectory-forge-rollout-rpc/v0.1")
            ),
            job_id=str(data["job_id"]),
            status=str(data["status"]),
            mode=str(data["mode"]),
            final_score=float(data["final_score"]),
            steps_executed=int(data["steps_executed"]),
            snapshots_used=int(data["snapshots_used"]),
            firewall_verified=bool(data["firewall_verified"]),
            stage_progression=tuple(
                StageProgressEntry.from_dict(s) for s in data.get("stage_progression", ())
            ),
            diagnostics=SolvabilityDiagnostics.from_dict(data.get("diagnostics", {})),
            media=tuple(MediaArtifactRef.from_dict(m) for m in data.get("media", ())),
            trajectory_manifest=(
                TrajectoryEpisodeManifest.from_dict(raw_manifest)
                if isinstance(raw_manifest, Mapping)
                else None
            ),
            saved_snapshots=tuple(
                SnapshotHandle.from_dict(s) for s in data.get("saved_snapshots", ())
            ),
            state_summary=dict(raw_state) if isinstance(raw_state, Mapping) else {},
        )
