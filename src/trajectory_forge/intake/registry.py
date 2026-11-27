"""Registry-driven simulator-neutral scenario intake (`intake/registry.py`).

Decouples `trajectory-forge` core & solver layers from any single scenario package
format (supports `genmanip_package`, `usd_direct`, and future `mjcf` / `isaaclab`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from trajectory_forge.core.contracts import (
    ContractValidationError,
    EmbodimentCard,
    ScenarioIntakeBrief,
)
from trajectory_forge.intake.genmanip_intake import (
    SUPPORTED_PACKAGE_SCHEMA as GENMANIP_PACKAGE_SCHEMA,
    load_genmanip_collected_package,
)
from trajectory_forge.intake.usd_direct import (
    USD_DIRECT_SCHEMA,
    load_usd_direct_scenario,
)

IntakeLoader = Callable[..., ScenarioIntakeBrief]


class IntakeRegistry:
    """Pluggable registry mapping scenario package format names to loader functions."""

    def __init__(self) -> None:
        self._loaders: dict[str, IntakeLoader] = {}

    def register(self, format_name: str, loader: IntakeLoader) -> None:
        self._loaders[format_name] = loader

    def list_formats(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaders.keys()))

    def load(
        self,
        source: str | Path,
        *,
        format_name: str | None = None,
        **kwargs: Any,
    ) -> ScenarioIntakeBrief:
        fmt = format_name or detect_scenario_format(source)
        if fmt not in self._loaders:
            raise ContractValidationError(
                f"Unknown scenario intake format {fmt!r}. Registered: {self.list_formats()}"
            )
        return self._loaders[fmt](source, **kwargs)


def detect_scenario_format(source: str | Path) -> str:
    """Auto-detect scenario package format from directory/file layout & schema field."""
    p = Path(source).resolve()
    if p.is_file() and p.suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        schema = str(raw.get("schema", ""))
        if schema == USD_DIRECT_SCHEMA:
            return "usd_direct"
        if schema == GENMANIP_PACKAGE_SCHEMA:
            return "genmanip_package"
    if p.is_dir():
        if (p / "scenario_direct.json").is_file():
            return "usd_direct"
        if (p / "package_manifest.json").is_file() or (
            p / "collected_package" / "package_manifest.json"
        ).is_file():
            return "genmanip_package"
    raise ContractValidationError(f"Could not detect scenario intake format for: {p}")


def render_agent_brief_markdown(
    brief: ScenarioIntakeBrief,
    embodiment: EmbodimentCard | None = None,
) -> str:
    """Render a compact simulator-neutral Markdown brief for coding agents."""
    lines = [
        f"# Privileged Task Brief: `{brief.package_id}` (Episode `{brief.episode_name}`)",
        "",
        f"- **Instruction**: {brief.instruction}",
        f"- **Scene Asset**: `{brief.scene_usd_relpath or '<procedural/direct>'}`",
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


DEFAULT_INTAKE_REGISTRY = IntakeRegistry()
DEFAULT_INTAKE_REGISTRY.register("genmanip_package", load_genmanip_collected_package)
DEFAULT_INTAKE_REGISTRY.register("usd_direct", load_usd_direct_scenario)


def load_scenario_package(
    source: str | Path,
    *,
    format_name: str | None = None,
    **kwargs: Any,
) -> ScenarioIntakeBrief:
    """Unified simulator-neutral scenario package loader."""
    return DEFAULT_INTAKE_REGISTRY.load(source, format_name=format_name, **kwargs)
