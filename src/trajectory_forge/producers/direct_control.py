"""Direct Control interactive probing session & WaypointPolicy compiler (pure Python).

Implements the EFE `Direct Control` paradigm cleanly inside the microkernel:
  - Agents interact step-by-step with a live `SimSession` (`privileged_search` mode),
    inspecting live observations, saving/restoring stage checkpoints, and recording
    relative object-anchored waypoints.
  - Calling `compile_to_policy()` converts the discovered interactive waypoints into
    a closed-loop `WaypointPolicy` that generalizes across seeds inside `FullHorizonGate`.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from trajectory_forge.core.contracts import (
    ContractValidationError,
    EmbodimentCard,
    ScenarioIntakeBrief,
)
from trajectory_forge.core.microkernel import (
    ActionCommand,
    RelativeWaypoint,
    SimAdapter,
    SimCapability,
    SimSession,
    StepObservation,
    WaypointPolicy,
)
from trajectory_forge.producers.base import ensure_adapter_capabilities


class DirectControlSession:
    """Interactive agent probing session over any `SimAdapter`."""

    def __init__(
        self,
        session: SimSession,
        *,
        initial_obs: StepObservation,
    ) -> None:
        self._session = session
        self.last_obs = initial_obs
        self.recorded_waypoints: list[RelativeWaypoint] = []
        self.step_history: list[dict[str, Any]] = []

    def step_action(self, command: ActionCommand, *, label: str = "") -> StepObservation:
        obs = self._session.step(command)
        self.last_obs = obs
        self.step_history.append(
            {
                "label": label,
                "step_index": obs.step_index,
                "total_score": obs.total_score,
                "action": command.to_dict(),
            }
        )
        return obs

    def save_checkpoint(self, label: str) -> str:
        return self._session.save_checkpoint(label)

    def restore_checkpoint(self, snap_id: str) -> StepObservation:
        obs = self._session.restore_checkpoint(snap_id)
        self.last_obs = obs
        return obs

    def record_waypoint(self, waypoint: RelativeWaypoint) -> None:
        self.recorded_waypoints.append(waypoint)

    def compile_to_policy(self) -> WaypointPolicy:
        if not self.recorded_waypoints:
            raise ContractValidationError(
                "DirectControlSession has no recorded waypoints to compile into WaypointPolicy"
            )
        return WaypointPolicy(
            tuple(self.recorded_waypoints),
            producer_name="direct_control",
        )

    def close(self) -> None:
        self._session.close()


class DirectControlProducer:
    """Runs an interactive Direct Control agent callback and compiles a closed-loop policy."""

    producer_name = "direct_control"
    required_capabilities: frozenset[SimCapability] = frozenset(
        {SimCapability.SINGLE_ENV, SimCapability.PRIVILEGED_STATE}
    )

    def produce(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        adapter: SimAdapter,
        **kwargs: Any,
    ) -> WaypointPolicy:
        ensure_adapter_capabilities(
            adapter, self.required_capabilities, producer_name=self.producer_name
        )
        waypoints: Sequence[RelativeWaypoint] | None = kwargs.get("waypoints")
        if waypoints is not None:
            return WaypointPolicy(tuple(waypoints), producer_name=self.producer_name)

        driver: Callable[[DirectControlSession, ScenarioIntakeBrief, EmbodimentCard], None] | None = (
            kwargs.get("interactive_driver")
        )
        if driver is None:
            raise ContractValidationError(
                "DirectControlProducer requires either `interactive_driver` or `waypoints`"
            )
        seed = int(kwargs.get("seed", 0))
        options = kwargs.get("adapter_options")
        raw_session = adapter.open_session(
            brief, embodiment, mode="privileged_search", seed=seed, options=options
        )
        try:
            obs0 = raw_session.reset(seed=seed)
            dc_session = DirectControlSession(raw_session, initial_obs=obs0)
            driver(dc_session, brief, embodiment)
            return dc_session.compile_to_policy()
        finally:
            raw_session.close()
