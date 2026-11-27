"""Lower-layer simulator adapter registry (`adapters/base.py`)."""
from __future__ import annotations

from trajectory_forge.core.contracts import ContractValidationError
from trajectory_forge.core.microkernel import SimAdapter


class SimAdapterRegistry:
    """Registry of simulator backend plugins (`isaac_genmanip`, `newton`, future `mujoco`/`isaaclab`)."""

    def __init__(self) -> None:
        self._adapters: dict[str, SimAdapter] = {}

    def register(self, adapter: SimAdapter) -> None:
        self._adapters[adapter.backend_name] = adapter

    def get(self, name: str) -> SimAdapter:
        if name not in self._adapters:
            raise ContractValidationError(
                f"Unknown SimAdapter {name!r}. Registered: {self.list_names()}"
            )
        return self._adapters[name]

    def list_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters.keys()))
