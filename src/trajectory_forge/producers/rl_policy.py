"""RL / Learned Policy TrajectoryProducer plugin (requires `SimCapability.VEC_ENV`).

Pure-Python microkernel contract for the EFE `Train a Policy (RL)` paradigm:
  - Enforces that the target `SimAdapter` declares `SimCapability.VEC_ENV` (e.g. Newton
    or Isaac Lab GPU parallel worlds) before launching training.
  - Accepts any pluggable trainer callable (PPO / SAC / Evolution / Diff-Physics) that
    produces a closed-loop `CheckpointPolicy` deliverable for `FullHorizonGate`.
"""
from __future__ import annotations

from typing import Any, Callable

from trajectory_forge.core.contracts import (
    ContractValidationError,
    EmbodimentCard,
    ScenarioIntakeBrief,
)
from trajectory_forge.core.microkernel import (
    CheckpointPolicy,
    SimAdapter,
    SimCapability,
)
from trajectory_forge.producers.base import ensure_adapter_capabilities


RLTrainerFn = Callable[
    [ScenarioIntakeBrief, EmbodimentCard, SimAdapter, dict[str, Any]],
    CheckpointPolicy,
]


class RLPolicyProducer:
    """Produces a closed-loop `CheckpointPolicy` trained on a vectorized `SimAdapter`."""

    producer_name = "rl_policy"
    required_capabilities: frozenset[SimCapability] = frozenset(
        {SimCapability.VEC_ENV, SimCapability.PRIVILEGED_STATE}
    )

    def produce(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        adapter: SimAdapter,
        **kwargs: Any,
    ) -> CheckpointPolicy:
        ensure_adapter_capabilities(
            adapter, self.required_capabilities, producer_name=self.producer_name
        )
        trainer_fn: RLTrainerFn | None = kwargs.get("trainer_fn")
        if trainer_fn is None:
            raise ContractValidationError(
                "RLPolicyProducer.produce requires a `trainer_fn(brief, embodiment, adapter, cfg) -> CheckpointPolicy`"
            )
        train_cfg = dict(kwargs.get("train_config") or {})
        policy = trainer_fn(brief, embodiment, adapter, train_cfg)
        if not isinstance(policy, CheckpointPolicy):
            raise ContractValidationError(
                f"RL trainer_fn must return CheckpointPolicy, got {type(policy)!r}"
            )
        return policy
