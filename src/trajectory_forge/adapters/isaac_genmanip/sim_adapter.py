"""Microkernel v2 `SimAdapter` declaration for the Isaac Sim 4.1 + GenManip backend."""
from __future__ import annotations

from typing import Any, Mapping

from trajectory_forge.core.contracts import (
    ContractValidationError,
    EmbodimentCard,
    ScenarioIntakeBrief,
)
from trajectory_forge.core.microkernel import SimCapability, SimSession


class IsaacGenManipSimAdapter:
    """Declares Isaac Sim 4.1 + GenManip capabilities cleanly for Microkernel v2.

    Note: intentionally does NOT declare `SimCapability.VEC_ENV` (single-env Isaac Sim 4.1),
    so producers requiring GPU vectorized rollouts (e.g. `RLPolicyProducer`) fail fast with
    a clear capability mismatch instead of crashing mid-run.
    """

    backend_name = "isaac_genmanip"
    capabilities: frozenset[SimCapability] = frozenset(
        {
            SimCapability.SINGLE_ENV,
            SimCapability.CHECKPOINT,
            SimCapability.RGB_RENDER,
            SimCapability.PRIVILEGED_STATE,
        }
    )

    def open_session(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        *,
        mode: str = "full_horizon_gate",
        seed: int = 0,
        options: Mapping[str, Any] | None = None,
    ) -> SimSession:
        _ = brief, embodiment, mode, seed, options
        raise ContractValidationError(
            "IsaacGenManipSimAdapter.open_session requires an active Isaac Sim 4.1 SimulationApp "
            "process (launch via `trajectory_forge.adapters.isaac_genmanip.runner_entry` or "
            "`cross_embodiment_runner`)."
        )
