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


def load_embodiment_card(path: Path) -> EmbodimentCard:
    """Load and validate an EmbodimentCard YAML file."""
    if not path.exists():
        raise EmbodimentValidationError(f"EmbodimentCard not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _require_mapping(raw, f"EmbodimentCard({path})")

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
    )
