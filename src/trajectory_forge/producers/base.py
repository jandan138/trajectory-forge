"""Upper-layer pluggable trajectory generation paradigm registry (pure Python)."""
from __future__ import annotations

from trajectory_forge.core.contracts import ContractValidationError
from trajectory_forge.core.microkernel import (
    SimAdapter,
    SimCapability,
    TrajectoryProducer,
)


def ensure_adapter_capabilities(
    adapter: SimAdapter,
    required: frozenset[SimCapability],
    *,
    producer_name: str,
) -> None:
    """Validate that `adapter` provides every `SimCapability` needed by `producer_name`."""
    missing = sorted(cap.value for cap in required if cap not in adapter.capabilities)
    if missing:
        raise ContractValidationError(
            f"Producer {producer_name!r} requires capabilities {missing}, "
            f"but SimAdapter {adapter.backend_name!r} only declares "
            f"{sorted(c.value for c in adapter.capabilities)}"
        )


class ProducerRegistry:
    """Registry of peer trajectory generation paradigms (Code / Direct / RL / future WM/VLA)."""

    def __init__(self) -> None:
        self._producers: dict[str, TrajectoryProducer] = {}

    def register(self, producer: TrajectoryProducer) -> None:
        self._producers[producer.producer_name] = producer

    def get(self, name: str) -> TrajectoryProducer:
        if name not in self._producers:
            raise ContractValidationError(
                f"Unknown TrajectoryProducer {name!r}. Registered: {self.list_names()}"
            )
        return self._producers[name]

    def list_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._producers.keys()))
