from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from trajectory_forge.core.contracts import (
    ActuatorGroupSpec,
    EmbodimentCard,
    EmbodimentValidationError,
    KinematicChainSpec,
    SensorStreamSpec,
)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EmbodimentValidationError(f"{label} must be a mapping")
    return value


def _load_v02_embodiment_card(root: Mapping[str, Any]) -> EmbodimentCard:
    """Parse cross-embodiment v0.2 YAML (`kinematic_chains` / `end_effectors` / `urdf_path`)."""
    chains_raw = _require_mapping(root.get("kinematic_chains", {}), "kinematic_chains")
    chains: dict[str, KinematicChainSpec] = {}
    next_dof = 0
    for cid, entry in chains_raw.items():
        c_map = _require_mapping(entry, f"kinematic_chains.{cid}")
        jnames = tuple(str(n) for n in c_map.get("joint_names", ()))
        dof_indices = tuple(range(next_dof, next_dof + len(jnames)))
        next_dof += len(jnames)
        chains[str(cid)] = KinematicChainSpec(
            chain_id=str(cid),
            base_frame=str(c_map.get("base_frame", "pedestal_base_link")),
            ee_frame=str(c_map.get("ee_link") or c_map.get("ee_frame") or f"{cid}_ee"),
            dof_indices=dof_indices,
            joint_names=jnames,
            neutral_q=tuple(float(v) for v in c_map.get("neutral_q", ())),
        )

    ees_raw = _require_mapping(root.get("end_effectors", {}), "end_effectors")
    actuator_groups: dict[str, ActuatorGroupSpec] = {}
    for gid, entry in ees_raw.items():
        e_map = _require_mapping(entry, f"end_effectors.{gid}")
        jnames = tuple(
            str(n)
            for n in (e_map.get("driver_joints") or e_map.get("all_joints") or ())
        )
        dof_indices = tuple(range(next_dof, next_dof + len(jnames)))
        next_dof += len(jnames)
        named_postures: dict[str, tuple[float, ...]] = {}
        for pname, pval in (e_map.get("postures") or {}).items():
            if isinstance(pval, Mapping):
                named_postures[str(pname)] = tuple(
                    float(pval.get(jn, 0.0)) for jn in jnames
                )
            elif isinstance(pval, (list, tuple)) and len(pval) == len(jnames):
                named_postures[str(pname)] = tuple(float(v) for v in pval)
        actuator_groups[str(gid)] = ActuatorGroupSpec(
            group_id=str(gid),
            kind=str(e_map.get("type", "parallel_gripper")),
            dof_indices=dof_indices,
            joint_names=jnames,
            control_modes=tuple(str(m) for m in e_map.get("control_modes", ("position", "effort"))),
            named_postures=named_postures,
        )

    action_dim = int(root.get("action_dim", max(1, next_dof)))
    sensor_streams = {
        "proprio": SensorStreamSpec(
            stream_id="proprio",
            modality="proprio",
            rate_hz=30.0,
            shape=(action_dim,),
        )
    }
    metadata = {
        "display_name": str(root.get("display_name", root["embodiment_id"])),
        "ik_backend": str(root.get("ik_backend", "pink_qp")),
        "urdf_path": str(root.get("urdf_path", "")),
        "usd_path": str(root.get("usd_path", "")),
        "base_pose_world": dict(root.get("base_pose_world") or {}),
    }
    return EmbodimentCard(
        schema_version=str(root.get("schema_version", "0.2")),
        embodiment_id=str(root["embodiment_id"]),
        runtime_robot_type=str(root.get("runtime_robot_type") or root.get("robot_family") or "generic_urdf"),
        action_dim=action_dim,
        base_locked=bool(root.get("base_locked", True)),
        chains=chains,
        actuator_groups=actuator_groups,
        sensor_streams=sensor_streams,
        notes=str(root.get("notes") or root.get("display_name") or ""),
        metadata=metadata,
    )


def load_embodiment_card(path: Path) -> EmbodimentCard:
    """Load and validate an EmbodimentCard YAML file (supports v0.1 and v0.2 schemas)."""
    if not path.exists():
        raise EmbodimentValidationError(f"EmbodimentCard not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _require_mapping(raw, f"EmbodimentCard({path})")

    if "kinematic_chains" in root or "end_effectors" in root:
        return _load_v02_embodiment_card(root)

    chains_raw = _require_mapping(root.get("chains", {}), "chains")
    chains: dict[str, KinematicChainSpec] = {}
    for cid, entry in chains_raw.items():
        c_map = dict(_require_mapping(entry, f"chains.{cid}"))
        c_map.setdefault("chain_id", cid)
        chains[str(cid)] = KinematicChainSpec.from_dict(c_map)

    groups_raw = _require_mapping(root.get("actuator_groups", {}), "actuator_groups")
    actuator_groups: dict[str, ActuatorGroupSpec] = {}
    for gid, entry in groups_raw.items():
        g_map = dict(_require_mapping(entry, f"actuator_groups.{gid}"))
        g_map.setdefault("group_id", gid)
        actuator_groups[str(gid)] = ActuatorGroupSpec.from_dict(g_map)

    streams_raw = _require_mapping(root.get("sensor_streams", {}), "sensor_streams")
    sensor_streams: dict[str, SensorStreamSpec] = {}
    for sid, entry in streams_raw.items():
        s_map = dict(_require_mapping(entry, f"sensor_streams.{sid}"))
        s_map.setdefault("stream_id", sid)
        sensor_streams[str(sid)] = SensorStreamSpec.from_dict(s_map)

    return EmbodimentCard(
        schema_version=str(
            root.get("schema_version", "trajectory-forge-embodiment-card/v0.1")
        ),
        embodiment_id=str(root["embodiment_id"]),
        runtime_robot_type=str(root["runtime_robot_type"]),
        action_dim=int(root["action_dim"]),
        base_locked=bool(root.get("base_locked", False)),
        chains=chains,
        actuator_groups=actuator_groups,
        sensor_streams=sensor_streams,
        notes=str(root.get("notes", "")),
        metadata=dict(root.get("metadata") or {}),
    )
