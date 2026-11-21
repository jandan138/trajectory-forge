"""Pure-layer intake loader for M2 deformable v0.2 scenario packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trajectory_forge.core.contracts import (
    ArticulationDofSpec,
    ContractValidationError,
    DeformableContractSpec,
    NamedFrameSpec,
    ObjectContractSpec,
    PoseSpec,
    RobotSpawnSpec,
    ScenarioIntakeBrief,
    StageContractSpec,
)


def _pose_from_yaml(raw: dict[str, Any] | None) -> PoseSpec:
    data = raw or {}
    xyz = data.get("xyz", [0.0, 0.0, 0.0])
    wxyz = data.get("wxyz", [1.0, 0.0, 0.0, 0.0])
    return PoseSpec(
        xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        wxyz=(float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])),
    )


def load_deformable_v02_package(package_dir: str | Path) -> ScenarioIntakeBrief:
    """Parse a scenario-package/v0.2 deformable package without simulator imports."""
    root = Path(package_dir)
    if (root / "scenario_package" / "manifest.yaml").is_file():
        root = root / "scenario_package"
    manifest_path = root / "manifest.yaml"
    if not manifest_path.is_file():
        raise ContractValidationError(f"Missing v0.2 manifest.yaml under {package_dir}")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    entrypoints = manifest["entrypoints"]
    instances_doc = yaml.safe_load((root / entrypoints["scene_instances"]).read_text(encoding="utf-8"))
    robot_doc = yaml.safe_load((root / entrypoints["robot"]).read_text(encoding="utf-8"))
    task_doc = yaml.safe_load((root / entrypoints["task"]).read_text(encoding="utf-8"))
    predicates_doc = yaml.safe_load((root / entrypoints["predicates"]).read_text(encoding="utf-8"))

    spawn_raw = robot_doc["spawn"]
    robot_spawn = RobotSpawnSpec(
        robot_id="franka_panda",
        xyz=(float(spawn_raw["xyz"][0]), float(spawn_raw["xyz"][1]), float(spawn_raw["xyz"][2])),
        wxyz=(
            float(spawn_raw["wxyz"][0]),
            float(spawn_raw["wxyz"][1]),
            float(spawn_raw["wxyz"][2]),
            float(spawn_raw["wxyz"][3]),
        ),
        joint_positions=(0.0, -0.55, 0.0, -2.25, 0.0, 1.75, 0.785398163397, 0.04, 0.04),
    )

    objects: dict[str, ObjectContractSpec] = {}
    for entry in instances_doc.get("instances", []):
        obj_id = str(entry["id"])
        frames = {
            str(fname): NamedFrameSpec(name=str(fname), pose_local=_pose_from_yaml(fval))
            for fname, fval in (entry.get("named_frames") or {}).items()
        }
        dofs: dict[str, ArticulationDofSpec] = {}
        init_state = entry.get("initial_state") or {}
        if obj_id == "pastry_box_hinged":
            dofs["lid_hinge"] = ArticulationDofSpec(
                semantic_joint="lid_hinge",
                dof_index=0,
                joint_prim="/World/pastry_box_hinged/lid_hinge",
                states={"closed": (-0.05, 0.15), "open_wide": (1.15, 1.66)},
            )
        deformable = None
        if isinstance(entry.get("deformable"), dict):
            raw_def = dict(entry["deformable"])
            raw_def.setdefault("asset_id", str(entry.get("asset_id", obj_id)))
            deformable = DeformableContractSpec.from_dict(raw_def)
        objects[obj_id] = ObjectContractSpec(
            object_id=obj_id,
            runtime_uid=str(entry.get("asset_id", obj_id)),
            role=str(entry["role"]),
            state_prim_path=str(entry["source_prim_path"]),
            initial_pose=_pose_from_yaml(entry.get("pose")),
            named_frames=frames,
            dofs=dofs,
            deformable=deformable,
        )

    stages: list[StageContractSpec] = []
    for pred in predicates_doc.get("success_predicates", []):
        stages.append(
            StageContractSpec(
                stage_index=int(pred.get("sequence_index", len(stages))),
                metric_id=str(pred["id"]),
                clauses=(
                    {
                        "predicate": str(pred["type"]),
                        "parameters": dict(pred.get("parameters") or {}),
                    },
                ),
            )
        )

    return ScenarioIntakeBrief(
        package_id=str(manifest["package_id"]),
        episode_name="deformable_v02_pilot",
        instruction=str(task_doc.get("instruction", "")),
        scene_usd_relpath=str(entrypoints["scene_usd"]),
        robot_profile=str(robot_doc.get("profile_ref", "franka_panda_tabletop")),
        robot_spawn=robot_spawn,
        objects=objects,
        stages=tuple(stages),
        part_ids=(),
    )
