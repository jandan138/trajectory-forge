"""Simulator-neutral direct OpenUSD/URDF scenario intake (`usd_direct/v1`).

Allows `trajectory-forge` to consume tasks natively (e.g., for Newton, MuJoCo, or
standalone OpenUSD tasks) without depending on GenManip collected-package layouts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trajectory_forge.core.contracts import (
    ContractValidationError,
    NamedFrameSpec,
    ObjectContractSpec,
    PoseSpec,
    RobotSpawnSpec,
    ScenarioIntakeBrief,
    StageContractSpec,
)

USD_DIRECT_SCHEMA = "trajectory-forge-usd-direct-scenario/v1"


def load_usd_direct_scenario(
    scenario_path: str | Path,
    *,
    episode_name: str | None = None,
) -> ScenarioIntakeBrief:
    """Load a simulator-neutral `scenario_direct.json` spec into `ScenarioIntakeBrief`."""
    p = Path(scenario_path).resolve()
    spec_file = p / "scenario_direct.json" if p.is_dir() else p
    if not spec_file.is_file():
        raise ContractValidationError(f"Missing USD direct scenario file: {spec_file}")

    raw = json.loads(spec_file.read_text(encoding="utf-8"))
    schema = str(raw.get("schema", ""))
    if schema != USD_DIRECT_SCHEMA:
        raise ContractValidationError(
            f"Expected schema {USD_DIRECT_SCHEMA!r}, got {schema!r}"
        )

    pkg_root = spec_file.parent
    package_id = str(raw.get("package_id") or pkg_root.name)
    ep_name = episode_name or str(raw.get("episode_name") or "ep_000")
    instruction = str(raw.get("instruction") or f"Complete {package_id}")

    rspawn_raw: Mapping[str, Any] = raw.get("robot_spawn") or {}
    xyz_r = tuple(float(v) for v in rspawn_raw.get("xyz", (0.0, 0.0, 0.0)))
    wxyz_r = tuple(float(v) for v in rspawn_raw.get("wxyz", (1.0, 0.0, 0.0, 0.0)))
    robot_spawn = RobotSpawnSpec(
        robot_id=str(rspawn_raw.get("robot_id", "generic_robot")),
        xyz=(xyz_r[0], xyz_r[1], xyz_r[2]),
        wxyz=(wxyz_r[0], wxyz_r[1], wxyz_r[2], wxyz_r[3]),
        joint_positions=tuple(float(v) for v in rspawn_raw.get("joint_positions", ())),
    )

    objects: dict[str, ObjectContractSpec] = {}
    for raw_obj in raw.get("objects") or []:
        oid = str(raw_obj["object_id"])
        pose = PoseSpec.from_dict(raw_obj.get("initial_pose") or {"xyz": [0.0, 0.0, 0.0]})
        nframes: dict[str, NamedFrameSpec] = {}
        for fname, fval in (raw_obj.get("named_frames") or {}).items():
            fpose = PoseSpec.from_dict(fval.get("pose_local") if isinstance(fval, Mapping) and "pose_local" in fval else fval)
            nframes[str(fname)] = NamedFrameSpec(name=str(fname), pose_local=fpose)
        objects[oid] = ObjectContractSpec(
            object_id=oid,
            runtime_uid=str(raw_obj.get("runtime_uid") or oid),
            role=str(raw_obj.get("role") or "manipulable"),
            initial_pose=pose,
            state_prim_path=str(raw_obj.get("prim_path") or f"/World/{oid}"),
            named_frames=nframes,
            dofs={},
        )

    stages: list[StageContractSpec] = []
    for idx, st in enumerate(raw.get("stages") or []):
        stages.append(
            StageContractSpec(
                stage_index=int(st.get("stage_index", idx)),
                metric_id=str(st.get("metric_id", f"stage_{idx}")),
                clauses=tuple(dict(c) for c in (st.get("clauses") or [])),
            )
        )

    return ScenarioIntakeBrief(
        package_id=package_id,
        episode_name=ep_name,
        instruction=instruction,
        scene_usd_relpath=str(raw.get("scene_usd") or ""),
        robot_profile=str(raw.get("robot_profile") or "generic_urdf"),
        robot_spawn=robot_spawn,
        objects=objects,
        stages=tuple(stages),
        part_ids=tuple(str(x) for x in (raw.get("part_ids") or [])),
    )
