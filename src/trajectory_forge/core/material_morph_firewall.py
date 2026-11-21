"""Pure-layer Material-Morph Firewall for v0.5 (M4).

Prevents controllers in full_horizon_gate, trajectory_synthesis, and EOS Protocol C1 from
reading ground-truth delta_q, morphed (E, nu), carrier stiffnesses, particle deformation
gradient F, or FractureMask buffers, and audits exported .npz manifests.
"""

from __future__ import annotations

import ast
from typing import Any, Mapping


FORBIDDEN_PRIVILEGED_MATERIAL_TOKENS: frozenset[str] = frozenset(
    {
        "delta_q",
        "morphed_youngs_modulus_pa",
        "morphed_poissons_ratio",
        "carrier_stiffness",
        "particle_F",
        "deformation_gradient",
        "fracture_mask",
        "material_morph_spec",
    }
)

FORBIDDEN_CHEAT_ATTACHMENT_TOKENS: frozenset[str] = frozenset(
    {
        "<weld",
        "<connect",
        "PhysxPhysicsAttachment",
        "OmniPhysicsVtxXformAttachment",
        "cloth_spine",
    }
)


class MaterialMorphFirewallError(PermissionError):
    """Raised when a controller or manifest violates the v0.5 Material-Morph Firewall."""


class MaterialMorphFirewallProxy:
    """Read-only observation proxy that raises on any privileged material-morph attribute access."""

    def __init__(self, sensor_obs: Mapping[str, Any], *, mode: str = "protocol_c1") -> None:
        self._mode = str(mode)
        self._obs = dict(sensor_obs)

    def get(self, key: str, default: Any = None) -> Any:
        if key in FORBIDDEN_PRIVILEGED_MATERIAL_TOKENS:
            raise MaterialMorphFirewallError(
                f"MaterialMorphFirewall violation in mode={self._mode!r}: "
                f"controller attempted to read privileged key {key!r}"
            )
        return self._obs.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if key in FORBIDDEN_PRIVILEGED_MATERIAL_TOKENS:
            raise MaterialMorphFirewallError(
                f"MaterialMorphFirewall violation in mode={self._mode!r}: "
                f"controller attempted to read privileged key {key!r}"
            )
        return self._obs[key]

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_PRIVILEGED_MATERIAL_TOKENS:
            raise MaterialMorphFirewallError(
                f"MaterialMorphFirewall violation in mode={self._mode!r}: "
                f"controller attempted to access privileged attribute {name!r}"
            )
        raise AttributeError(name)


def audit_controller_material_firewall(source_code: str) -> list[str]:
    """Static AST + token audit ensuring a controller neither cheats nor peeks at delta_q."""
    violations: list[str] = []
    for cheat in sorted(FORBIDDEN_CHEAT_ATTACHMENT_TOKENS):
        if cheat in source_code:
            violations.append(f"forbidden zero-cheat token {cheat!r}")

    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_PRIVILEGED_MATERIAL_TOKENS:
            violations.append(f"forbidden attribute read .{node.attr} at line {node.lineno}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in FORBIDDEN_PRIVILEGED_MATERIAL_TOKENS:
                violations.append(f"forbidden privileged key literal {node.value!r} at line {node.lineno}")
    return violations


def validate_exported_morph_manifest(manifest: Mapping[str, Any]) -> None:
    """Assert that an exported v0.5 trajectory manifest obeys all firewall invariants."""
    if int(manifest.get("snapshots_used", -1)) != 0:
        raise MaterialMorphFirewallError("Exported manifest must have snapshots_used == 0")
    if manifest.get("continuous_from_t0") is not True:
        raise MaterialMorphFirewallError("Exported manifest must have continuous_from_t0 == True")
    if manifest.get("privileged_channels_recorded") is not False:
        raise MaterialMorphFirewallError(
            "Exported manifest must have privileged_channels_recorded == False"
        )
    if manifest.get("material_morph_privileged_visible") is not False:
        raise MaterialMorphFirewallError(
            "Exported manifest must have material_morph_privileged_visible == False"
        )

