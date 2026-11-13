from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trajectory_forge.core.contracts import (
    ArticulationDofSpec,
    ContractValidationError,
    EmbodimentCard,
    NamedFrameSpec,
    ObjectContractSpec,
    PoseSpec,
    RobotSpawnSpec,
    ScenarioIntakeBrief,
    StageContractSpec,
)


SUPPORTED_PACKAGE_SCHEMA = "scenario-forge-genmanip-collected-package/v0.1"


def _resolve_json_pointer(doc: Any, pointer: str) -> Any:
    if not pointer or pointer == "/":
        return doc
    if not pointer.startswith("/"):
        raise ContractValidationError(f"Invalid JSON pointer: {pointer!r}")
    cur = doc
    for raw_seg in pointer.lstrip("/").split("/"):
        seg = raw_seg.replace("~1", "/").replace("~0", "~")
        if not isinstance(cur, Mapping) or seg not in cur:
            raise ContractValidationError(
                f"JSON pointer {pointer!r} failed at segment {seg!r}"
            )
        cur = cur[seg]
    return cur


def _resolve_package_root(package_dir: Path) -> tuple[Path, Path]:
    direct = package_dir / "package_manifest.json"
    if direct.exists():
        return package_dir, direct
    nested = package_dir / "collected_package" / "package_manifest.json"
    if nested.exists():
        return package_dir / "collected_package", nested
    raise ContractValidationError(
        f"Could not find package_manifest.json in {package_dir} or {package_dir / 'collected_package'}"
    )


def _parse_pose(
    raw: Mapping[str, Any] | None,
    layout_entry: Mapping[str, Any] | None = None,
) -> PoseSpec:
    if not isinstance(raw, Mapping) and not isinstance(layout_entry, Mapping):
        return PoseSpec(xyz=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0))
    raw_map = raw if isinstance(raw, Mapping) else {}
    layout_map = layout_entry if isinstance(layout_entry, Mapping) else {}
    xyz_raw = (
        raw_map.get("xyz")
        or raw_map.get("translation_parent_local_m")
        or layout_map.get("position")
        or (0.0, 0.0, 0.0)
    )
    wxyz_raw = (
        raw_map.get("wxyz")
        or raw_map.get("rotation_parent_local_wxyz")
        or layout_map.get("orientation")
        or (1.0, 0.0, 0.0, 0.0)
    )
    return PoseSpec(
        xyz=(float(xyz_raw[0]), float(xyz_raw[1]), float(xyz_raw[2])),
        wxyz=(float(wxyz_raw[0]), float(wxyz_raw[1]), float(wxyz_raw[2]), float(wxyz_raw[3])),
    )


def _parse_object_contract(
    raw: Mapping[str, Any],
    initial_layout: Mapping[str, Any] | None = None,
) -> ObjectContractSpec:
    object_id = str(raw.get("scenario_object_id") or raw.get("runtime_uid") or "")
    if not object_id:
        raise ContractValidationError("Runtime contract object missing scenario_object_id")
    runtime_uid = str(raw.get("runtime_uid") or object_id)
    role = str(raw.get("role", "object"))
    state_prim_path = str(raw.get("state_prim_path", ""))
    layout_entry = (
        initial_layout.get(runtime_uid)
        if isinstance(initial_layout, Mapping)
        else None
    )
    initial_pose = _parse_pose(raw.get("initial_pose"), layout_entry)

    named_frames: dict[str, NamedFrameSpec] = {}
    articulation = raw.get("articulation")
    art_frames = articulation.get("named_frames", {}) if isinstance(articulation, Mapping) else {}
    top_frames = raw.get("named_frames", {})
    for frame_name, frame_data in dict(top_frames).items():
        if isinstance(frame_data, Mapping):
            auth_info = art_frames.get(frame_name, {})
            named_frames[str(frame_name)] = NamedFrameSpec(
                name=str(frame_name),
                pose_local=_parse_pose(frame_data),
                parent_prim=str(auth_info.get("parent_prim", "")),
                authoritative=bool(auth_info.get("authoritative", True)),
            )
    for frame_name, frame_data in dict(art_frames).items():
        if str(frame_name) not in named_frames and isinstance(frame_data, Mapping):
            named_frames[str(frame_name)] = NamedFrameSpec(
                name=str(frame_name),
                pose_local=_parse_pose(frame_data),
                parent_prim=str(frame_data.get("parent_prim", "")),
                authoritative=bool(frame_data.get("authoritative", True)),
            )

    dofs: dict[str, ArticulationDofSpec] = {}
    reset_joints: tuple[float, ...] = ()
    if isinstance(articulation, Mapping):
        states_map = articulation.get("states", {})
        for mapping in articulation.get("dof_mapping", ()):
            sem_name = str(mapping["semantic_joint"])
            joint_states = states_map.get(sem_name, {}) if isinstance(states_map, Mapping) else {}
            dofs[sem_name] = ArticulationDofSpec(
                semantic_joint=sem_name,
                dof_index=int(mapping["dof_index"]),
                joint_prim=str(mapping["joint_prim"]),
                states={
                    str(sname): (float(bounds[0]), float(bounds[1]))
                    for sname, bounds in joint_states.items()
                },
            )
        reset_joints = tuple(float(v) for v in articulation.get("reset_joint_positions", ()))

    return ObjectContractSpec(
        object_id=object_id,
        runtime_uid=runtime_uid,
        role=role,
        state_prim_path=state_prim_path,
        initial_pose=initial_pose,
        named_frames=named_frames,
        dofs=dofs,
        reset_joint_positions=reset_joints,
    )


def _parse_robot_spawn(initial_layout: Mapping[str, Any]) -> RobotSpawnSpec:
    for uid, entry in initial_layout.items():
        if isinstance(entry, Mapping) and entry.get("type") == "robot":
            pos = entry.get("position", (0.0, 0.0, 0.0))
            quat = entry.get("orientation", (1.0, 0.0, 0.0, 0.0))
            return RobotSpawnSpec(
                robot_id=str(uid),
                xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
                wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                joint_positions=tuple(float(v) for v in entry.get("joint_positions", ())),
            )
    raise ContractValidationError("episode_metadata.task_data.initial_layout missing robot entry")


def _parse_stages(
    goal: Any,
    runtime_contract: Mapping[str, Any],
) -> tuple[StageContractSpec, ...]:
    execution = runtime_contract.get("execution", {})
    rubric = execution.get("progress_rubric", {}) if isinstance(execution, Mapping) else {}
    metric_ids = list(rubric.get("unevaluated_metric_ids", ()))
    if not isinstance(goal, list):
        return ()

    stages: list[StageContractSpec] = []
    for idx, stage_block in enumerate(goal):
        clauses: list[dict[str, Any]] = []
        if isinstance(stage_block, list):
            for conj in stage_block:
                if isinstance(conj, list):
                    for clause in conj:
                        if isinstance(clause, Mapping):
                            clauses.append(dict(clause))
                elif isinstance(conj, Mapping):
                    clauses.append(dict(conj))
        metric_id = str(metric_ids[idx]) if idx < len(metric_ids) else f"stage_{idx}"
        stages.append(
            StageContractSpec(
                stage_index=idx,
                metric_id=metric_id,
                clauses=tuple(clauses),
            )
        )
    return tuple(stages)


def load_genmanip_collected_package(package_dir: Path) -> ScenarioIntakeBrief:
    """Parse a scenario-forge GenManip collected package into ScenarioIntakeBrief."""
    pkg_root, manifest_path = _resolve_package_root(package_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SUPPORTED_PACKAGE_SCHEMA:
        raise ContractValidationError(
            f"Unsupported package schema_version: {manifest.get('schema_version')!r}"
        )

    entrypoints = manifest.get("entrypoints", {})
    semantic_cfg = manifest.get("semantic_contract", {})
    ep_relpath = semantic_cfg.get("path") or entrypoints.get("episode_metadata")
    if not ep_relpath:
        raise ContractValidationError("package_manifest.json missing episode_metadata path")

    ep_path = pkg_root / str(ep_relpath)
    if not ep_path.exists():
        raise ContractValidationError(f"episode_metadata.json not found: {ep_path}")
    ep_doc = json.loads(ep_path.read_text(encoding="utf-8"))

    pointer = str(semantic_cfg.get("json_pointer", "/task_data/scenario_forge_runtime_contract"))
    runtime_contract = _resolve_json_pointer(ep_doc, pointer)
    if not isinstance(runtime_contract, Mapping):
        raise ContractValidationError("Resolved scenario_forge_runtime_contract is not a mapping")

    task_data = ep_doc.get("task_data", {})
    if not isinstance(task_data, Mapping):
        raise ContractValidationError("episode_metadata missing task_data mapping")

    initial_layout = task_data.get("initial_layout", {})
    robot_spawn = _parse_robot_spawn(initial_layout)
    objects: dict[str, ObjectContractSpec] = {}
    for obj_raw in runtime_contract.get("objects", ()):
        obj_spec = _parse_object_contract(obj_raw, initial_layout=initial_layout)
        objects[obj_spec.object_id] = obj_spec

    art_uids = {
        uid
        for uid, entry in initial_layout.items()
        if isinstance(entry, Mapping) and entry.get("type") == "articulation"
    }
    part_ids = tuple(
        sorted(
            str(uid)
            for uid, entry in initial_layout.items()
            if isinstance(entry, Mapping)
            and entry.get("type") == "object"
            and uid not in objects
            and (
                bool(entry.get("is_articulation_part", False))
                or any(str(uid).startswith(f"{aid}_") for aid in art_uids)
            )
        )
    )

    stages = _parse_stages(task_data.get("goal", []), runtime_contract)
    runtime_reqs = manifest.get("runtime_requirements", {})

    return ScenarioIntakeBrief(
        package_id=str(manifest["package_id"]),
        episode_name=str(ep_doc.get("episode_name", "")),
        instruction=str(task_data.get("instruction", "")),
        scene_usd_relpath=str(entrypoints.get("scene_usd", "")),
        robot_profile=str(runtime_reqs.get("runtime_robot_type", "")),
        robot_spawn=robot_spawn,
        objects=objects,
        stages=stages,
        part_ids=part_ids,
    )


def render_agent_brief_markdown(
    brief: ScenarioIntakeBrief,
    embodiment: EmbodimentCard | None = None,
) -> str:
    """Render a compact Markdown brief for privileged solver agents."""
    lines = [
        f"# Privileged Task Brief: `{brief.package_id}` (Episode `{brief.episode_name}`)",
        "",
        f"- **Instruction**: {brief.instruction}",
        f"- **Scene USD**: `{brief.scene_usd_relpath}`",
        f"- **Robot Spawn**: `{brief.robot_spawn.robot_id}` at `xyz={list(brief.robot_spawn.xyz)}`",
    ]
    if embodiment is not None:
        lines.append(
            f"- **EmbodimentCard**: `{embodiment.embodiment_id}` "
            f"(chains={list(embodiment.chains.keys())}, "
            f"actuators={list(embodiment.actuator_groups.keys())})"
        )
    lines.extend(["", "## Stages", ""])
    for st in brief.stages:
        lines.append(f"- Stage {st.stage_index}: `{st.metric_id}` ({len(st.clauses)} clause(s))")

    lines.extend(["", "## Objects, Named Frames & Articulated DOFs", ""])
    for obj in brief.objects.values():
        lines.append(
            f"- `{obj.object_id}` (`{obj.role}`, uid=`{obj.runtime_uid}`): "
            f"xyz=`{list(obj.initial_pose.xyz)}`"
        )
        if obj.named_frames:
            frames_str = ", ".join(
                f"{fn}@{list(f.pose_local.xyz)}" for fn, f in obj.named_frames.items()
            )
            lines.append(f"  - named_frames: {frames_str}")
        if obj.dofs:
            dofs_str = ", ".join(
                f"{dn}(idx={d.dof_index}, states={list(d.states.keys())})"
                for dn, d in obj.dofs.items()
            )
            lines.append(f"  - dofs: {dofs_str}")

    return "\n".join(lines) + "\n"
