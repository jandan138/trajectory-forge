from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from trajectory_forge.core.contracts import (
    ContractValidationError,
    TrajectoryEpisodeManifest,
)
from trajectory_forge.trajectory.manifest import build_trajectory_episode_manifest


# GenManip ARX Lift2 default_dof_indices interleaved layout:
# [left_arm(0..5), left_gripper(6..7), right_arm(8..13), right_gripper(14..15)]
INTERLEAVED_16DOF_ARM_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13)
INTERLEAVED_16DOF_GRIPPER_INDICES: tuple[int, ...] = (6, 7, 14, 15)


@dataclass(frozen=True)
class SynchronizedEpisodeStreams:
    """Firewall-verified, frame-synchronized sensorimotor episode arrays ready for dataset export."""

    episode_id: str
    scenario_id: str
    instruction: str
    num_frames: int
    timestamps_s: NDArray[np.float64]
    state_joints: NDArray[np.float32]
    action_joints: NDArray[np.float32]
    action_joints_delta: NDArray[np.float32]
    state_gripper: NDArray[np.float32]
    action_gripper: NDArray[np.float32]
    state_base: NDArray[np.float32]
    action_base: NDArray[np.float32]
    action_base_delta: NDArray[np.float32]
    state_ee_pose: NDArray[np.float32]
    action_ee_pose: NDArray[np.float32]
    action_ee_pose_delta: NDArray[np.float32]
    observation_state_16dof: NDArray[np.float32]
    action_16dof: NDArray[np.float32]
    rgb_overlook: NDArray[np.uint8]
    rgb_left_wrist: NDArray[np.uint8]
    rgb_right_wrist: NDArray[np.uint8]


@dataclass(frozen=True)
class LeRobotEpisodeInputSpec:
    """Pure-Python descriptor pointing to one firewall-verified episode artifact for export."""

    episode_id: str
    npz_path: str
    manifest: TrajectoryEpisodeManifest
    instruction: str
    base_locked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "npz_path": self.npz_path,
            "manifest": self.manifest.to_dict(),
            "instruction": self.instruction,
            "base_locked": self.base_locked,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LeRobotEpisodeInputSpec:
        return cls(
            episode_id=str(data["episode_id"]),
            npz_path=str(data["npz_path"]),
            manifest=TrajectoryEpisodeManifest.from_dict(data["manifest"]),
            instruction=str(data["instruction"]),
            base_locked=bool(data.get("base_locked", True)),
        )


def split_interleaved_16dof(
    vec: NDArray[Any],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Split interleaved 16-DoF [L_arm(6), L_grip(2), R_arm(6), R_grip(2)] into (arm_12, grip_4)."""
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape[-1] != 16:
        raise ContractValidationError(
            f"Expected last dimension 16 for interleaved DOF vector, got {arr.shape}"
        )
    joints_12 = arr[..., INTERLEAVED_16DOF_ARM_INDICES].astype(np.float32, copy=True)
    gripper_4 = arr[..., INTERLEAVED_16DOF_GRIPPER_INDICES].astype(np.float32, copy=True)
    return joints_12, gripper_4


def reconstruct_interleaved_16dof(split_vec: NDArray[Any]) -> NDArray[np.float32]:
    """Reconstruct interleaved 16-DoF vector from split [arm_12, grip_4, ...base] ordering."""
    arr = np.asarray(split_vec, dtype=np.float32)
    if arr.shape[-1] < 16:
        raise ContractValidationError(
            f"Expected last dimension >= 16 for split DOF vector, got {arr.shape}"
        )
    joined: NDArray[np.float32] = np.concatenate(
        [
            arr[..., 0:6],
            arr[..., 12:14],
            arr[..., 6:12],
            arr[..., 14:16],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    return joined


def _validate_rgb_stream(
    arr: Any,
    *,
    name: str,
    expected_frames: int | None,
    expected_height: int,
    expected_width: int,
) -> NDArray[np.uint8]:
    rgb = np.asarray(arr)
    if rgb.ndim != 4 or rgb.shape[3] != 3:
        raise ContractValidationError(
            f"Camera stream {name!r} must have shape (N, H, W, 3), got {rgb.shape}"
        )
    if rgb.dtype != np.uint8:
        raise ContractValidationError(
            f"Camera stream {name!r} must have uint8 dtype, got {rgb.dtype}"
        )
    if rgb.shape[1] != expected_height or rgb.shape[2] != expected_width:
        raise ContractValidationError(
            f"Camera stream {name!r} resolution must be {(expected_height, expected_width)}, "
            f"got {(rgb.shape[1], rgb.shape[2])}"
        )
    if rgb.shape[0] <= 0:
        raise ContractValidationError(f"Camera stream {name!r} has zero frames")
    if expected_frames is not None and rgb.shape[0] != expected_frames:
        raise ContractValidationError(
            f"Camera stream {name!r} frame count mismatch: expected {expected_frames}, got {rgb.shape[0]}"
        )
    return rgb


def _resolve_camera_indices(
    npz_data: Mapping[str, Any],
    *,
    num_ticks: int,
    num_frames: int,
) -> NDArray[np.int64]:
    if "camera_step_indices" in npz_data:
        idx = np.asarray(npz_data["camera_step_indices"], dtype=np.int64)
        if idx.ndim != 1 or idx.shape[0] != num_frames:
            raise ContractValidationError(
                f"camera_step_indices shape mismatch: expected ({num_frames},), got {idx.shape}"
            )
        if np.any(idx < 0) or np.any(idx >= num_ticks):
            raise ContractValidationError(
                f"camera_step_indices out of bounds [0, {num_ticks})"
            )
        if num_frames > 1 and np.any(np.diff(idx) <= 0):
            raise ContractValidationError("camera_step_indices must be strictly increasing")
        return idx
    if num_ticks == num_frames:
        return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, num_ticks - 1, num=num_frames, dtype=np.int64)


def extract_synchronized_episode_streams(
    *,
    npz_path: Path,
    manifest: TrajectoryEpisodeManifest,
    instruction: str,
    base_locked: bool = True,
    expected_height: int = 480,
    expected_width: int = 640,
) -> SynchronizedEpisodeStreams:
    """Validate firewall invariants and extract synchronized RGB + bimanual state/action arrays."""
    checked_manifest = build_trajectory_episode_manifest(
        episode_id=manifest.episode_id,
        scenario_id=manifest.scenario_id,
        embodiment_id=manifest.embodiment_id,
        num_steps=manifest.num_steps,
        continuous_from_t0=manifest.continuous_from_t0,
        privileged_channels_recorded=manifest.privileged_channels_recorded,
        recorded_channels=manifest.recorded_channels,
        artifact_relpath=manifest.artifact_relpath,
    )
    if not instruction.strip():
        raise ContractValidationError("Episode instruction must be non-empty")
    if not npz_path.is_file():
        raise ContractValidationError(f"Episode artifact does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as archive:
        required_keys = ("proprio", "action", "rgb_overlook", "rgb_left_wrist", "rgb_right_wrist")
        missing = [k for k in required_keys if k not in archive]
        if missing:
            raise ContractValidationError(
                f"Episode archive {npz_path} missing required channels: {missing}"
            )

        proprio_all = np.asarray(archive["proprio"], dtype=np.float32)
        action_all = np.asarray(archive["action"], dtype=np.float32)
        if proprio_all.ndim != 2 or proprio_all.shape[1] != 16:
            raise ContractValidationError(
                f"Expected proprio shape (T, 16), got {proprio_all.shape}"
            )
        if action_all.shape != proprio_all.shape:
            raise ContractValidationError(
                f"Action shape {action_all.shape} does not match proprio {proprio_all.shape}"
            )
        num_ticks = proprio_all.shape[0]
        if num_ticks <= 0:
            raise ContractValidationError("Episode contains zero control ticks")

        rgb_overlook = _validate_rgb_stream(
            archive["rgb_overlook"],
            name="rgb_overlook",
            expected_frames=None,
            expected_height=expected_height,
            expected_width=expected_width,
        )
        num_frames = int(rgb_overlook.shape[0])
        rgb_left = _validate_rgb_stream(
            archive["rgb_left_wrist"],
            name="rgb_left_wrist",
            expected_frames=num_frames,
            expected_height=expected_height,
            expected_width=expected_width,
        )
        rgb_right = _validate_rgb_stream(
            archive["rgb_right_wrist"],
            name="rgb_right_wrist",
            expected_frames=num_frames,
            expected_height=expected_height,
            expected_width=expected_width,
        )

        cam_idx = _resolve_camera_indices(archive, num_ticks=num_ticks, num_frames=num_frames)
        proprio_sync = proprio_all[cam_idx].astype(np.float32, copy=True)
        action_sync = action_all[cam_idx].astype(np.float32, copy=True)
        state_joints, state_gripper = split_interleaved_16dof(proprio_sync)
        action_joints, action_gripper = split_interleaved_16dof(action_sync)
        action_joints_delta = (action_joints - state_joints).astype(np.float32)

        if base_locked or "base_target" not in archive:
            state_base = np.zeros((num_frames, 3), dtype=np.float32)
            action_base = np.zeros((num_frames, 3), dtype=np.float32)
            action_base_delta = np.zeros((num_frames, 3), dtype=np.float32)
        else:
            base_all = np.asarray(archive["base_target"], dtype=np.float32)
            if base_all.shape != (num_ticks, 3):
                raise ContractValidationError(
                    f"Expected base_target shape {(num_ticks, 3)}, got {base_all.shape}"
                )
            base_sync = base_all[cam_idx].astype(np.float32, copy=True)
            state_base = base_sync
            action_base = base_sync.copy()
            action_base_delta = np.zeros_like(base_sync)
            if num_frames > 1:
                action_base_delta[1:] = base_sync[1:] - base_sync[:-1]

        if "state_ee_pose" in archive and "action_ee_pose" in archive:
            state_ee_all = np.asarray(archive["state_ee_pose"], dtype=np.float32)
            action_ee_all = np.asarray(archive["action_ee_pose"], dtype=np.float32)
            if state_ee_all.shape != (num_ticks, 14) or action_ee_all.shape != (num_ticks, 14):
                raise ContractValidationError(
                    "state_ee_pose/action_ee_pose must match shape (T, 14)"
                )
            state_ee_pose = state_ee_all[cam_idx].astype(np.float32, copy=True)
            action_ee_pose = action_ee_all[cam_idx].astype(np.float32, copy=True)
            action_ee_pose_delta = (action_ee_pose - state_ee_pose).astype(np.float32)
        else:
            state_ee_pose = np.zeros((num_frames, 14), dtype=np.float32)
            action_ee_pose = np.zeros((num_frames, 14), dtype=np.float32)
            action_ee_pose_delta = np.zeros((num_frames, 14), dtype=np.float32)

        if "timestamps_s" in archive:
            ts_all = np.asarray(archive["timestamps_s"], dtype=np.float64)
            if ts_all.shape != (num_ticks,):
                raise ContractValidationError(
                    f"Expected timestamps_s shape {(num_ticks,)}, got {ts_all.shape}"
                )
            timestamps_s = ts_all[cam_idx].astype(np.float64, copy=True)
        else:
            timestamps_s = np.arange(num_frames, dtype=np.float64) / 15.0

    return SynchronizedEpisodeStreams(
        episode_id=checked_manifest.episode_id,
        scenario_id=checked_manifest.scenario_id,
        instruction=instruction,
        num_frames=num_frames,
        timestamps_s=timestamps_s,
        state_joints=state_joints,
        action_joints=action_joints,
        action_joints_delta=action_joints_delta,
        state_gripper=state_gripper,
        action_gripper=action_gripper,
        state_base=state_base,
        action_base=action_base,
        action_base_delta=action_base_delta,
        state_ee_pose=state_ee_pose,
        action_ee_pose=action_ee_pose,
        action_ee_pose_delta=action_ee_pose_delta,
        observation_state_16dof=proprio_sync,
        action_16dof=action_sync,
        rgb_overlook=rgb_overlook,
        rgb_left_wrist=rgb_left,
        rgb_right_wrist=rgb_right,
    )


def write_lerobot_staging_manifest(
    path: Path,
    episodes: Sequence[LeRobotEpisodeInputSpec],
) -> Path:
    """Serialize validated episode input descriptors for downstream dataset conversion."""
    if not episodes:
        raise ContractValidationError("At least one episode is required for LeRobot export")
    payload = {
        "schema_version": "trajectory-forge-lerobot-staging/v0.1",
        "episodes": [ep.to_dict() for ep in episodes],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_lerobot_staging_manifest(path: Path) -> tuple[LeRobotEpisodeInputSpec, ...]:
    """Load validated episode input descriptors from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("episodes", [])
    if not isinstance(raw, list) or not raw:
        raise ContractValidationError(f"Empty or invalid staging manifest: {path}")
    return tuple(LeRobotEpisodeInputSpec.from_dict(item) for item in raw)


_UNUSED_KEEP = asdict
