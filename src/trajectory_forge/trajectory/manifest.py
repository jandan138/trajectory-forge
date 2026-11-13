from __future__ import annotations

from trajectory_forge.core.contracts import (
    FirewallViolationError,
    TrajectoryEpisodeManifest,
)


FORBIDDEN_CHANNEL_PREFIXES = ("privileged_", "ground_truth_", "oracle_")


def build_trajectory_episode_manifest(
    *,
    episode_id: str,
    scenario_id: str,
    embodiment_id: str,
    num_steps: int,
    continuous_from_t0: bool,
    privileged_channels_recorded: bool,
    recorded_channels: tuple[str, ...],
    artifact_relpath: str = "",
) -> TrajectoryEpisodeManifest:
    """Construct a firewall-checked demonstration trajectory manifest."""
    for ch in recorded_channels:
        if any(ch.startswith(prefix) for prefix in FORBIDDEN_CHANNEL_PREFIXES):
            raise FirewallViolationError(
                f"Channel {ch!r} is forbidden in exported sensorimotor trajectories"
            )
    return TrajectoryEpisodeManifest(
        episode_id=episode_id,
        scenario_id=scenario_id,
        embodiment_id=embodiment_id,
        num_steps=num_steps,
        continuous_from_t0=continuous_from_t0,
        privileged_channels_recorded=privileged_channels_recorded,
        recorded_channels=recorded_channels,
        artifact_relpath=artifact_relpath,
    )
