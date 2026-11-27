"""Code-to-Control TrajectoryProducer plugin (pure Python)."""
from __future__ import annotations

from typing import Any

from trajectory_forge.core.contracts import (
    ContractValidationError,
    EmbodimentCard,
    ScenarioIntakeBrief,
)
from trajectory_forge.core.microkernel import (
    PythonControllerPolicy,
    SimAdapter,
    SimCapability,
)
from trajectory_forge.producers.base import ensure_adapter_capabilities


class CodeToControlProducer:
    """Produces an AST-validated closed-loop `PythonControllerPolicy`."""

    producer_name = "code_to_control"
    required_capabilities: frozenset[SimCapability] = frozenset({SimCapability.SINGLE_ENV})

    def produce(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        adapter: SimAdapter,
        **kwargs: Any,
    ) -> PythonControllerPolicy:
        _ = brief, embodiment
        ensure_adapter_capabilities(
            adapter, self.required_capabilities, producer_name=self.producer_name
        )
        source_code = kwargs.get("source_code")
        if not isinstance(source_code, str) or not source_code.strip():
            raise ContractValidationError(
                "CodeToControlProducer.produce requires non-empty `source_code` defining `def act(obs, state):`"
            )
        max_steps = int(kwargs.get("max_steps", 2000))
        return PythonControllerPolicy(
            source_code=source_code,
            producer_name=self.producer_name,
            max_steps=max_steps,
        )
