from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from trajectory_forge.trajectory.lerobot_converter import (
    SynchronizedEpisodeStreams,
    extract_synchronized_episode_streams,
    load_lerobot_staging_manifest,
)


CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
JOINT_NAMES_12 = [f"joint_{i}" for i in range(12)]
GRIPPER_NAMES_4 = ["gripper_1_left", "gripper_1_right", "gripper_2_left", "gripper_2_right"]
BASE_NAMES_3 = ["base_x", "base_y", "base_theta"]
EE_POSE_NAMES_14 = [
    "ee0_x",
    "ee0_y",
    "ee0_z",
    "ee0_qw",
    "ee0_qx",
    "ee0_qy",
    "ee0_qz",
    "ee1_x",
    "ee1_y",
    "ee1_z",
    "ee1_qw",
    "ee1_qx",
    "ee1_qy",
    "ee1_qz",
]
STATE_16_NAMES = [f"dof_{i}" for i in range(16)]


def build_lerobot_features() -> dict[str, dict[str, Any]]:
    """Build LeRobot v2.1 feature schema carrying both split and interleaved 16-DoF channels."""
    features: dict[str, dict[str, Any]] = {
        "state.joints": {"dtype": "float32", "shape": (12,), "names": [JOINT_NAMES_12]},
        "action.joints": {"dtype": "float32", "shape": (12,), "names": [JOINT_NAMES_12]},
        "action.joints_delta": {"dtype": "float32", "shape": (12,), "names": [JOINT_NAMES_12]},
        "state.gripper": {"dtype": "float32", "shape": (4,), "names": [GRIPPER_NAMES_4]},
        "action.gripper": {"dtype": "float32", "shape": (4,), "names": [GRIPPER_NAMES_4]},
        "state.base": {"dtype": "float32", "shape": (3,), "names": [BASE_NAMES_3]},
        "action.base": {"dtype": "float32", "shape": (3,), "names": [BASE_NAMES_3]},
        "action.base_delta": {"dtype": "float32", "shape": (3,), "names": [BASE_NAMES_3]},
        "state.ee_pose": {"dtype": "float32", "shape": (14,), "names": [EE_POSE_NAMES_14]},
        "action.ee_pose": {"dtype": "float32", "shape": (14,), "names": [EE_POSE_NAMES_14]},
        "action.ee_pose_delta": {
            "dtype": "float32",
            "shape": (14,),
            "names": [EE_POSE_NAMES_14],
        },
        "observation.state": {"dtype": "float32", "shape": (16,), "names": [STATE_16_NAMES]},
        "action": {"dtype": "float32", "shape": (16,), "names": [STATE_16_NAMES]},
    }
    for cam_key in (
        "video.overlook_camera_view",
        "video.left_camera_view",
        "video.right_camera_view",
    ):
        features[cam_key] = {
            "dtype": "video",
            "shape": (3, CAMERA_HEIGHT, CAMERA_WIDTH),
            "names": ["channels", "height", "width"],
        }
    return features


def _append_episode(dataset: Any, streams: SynchronizedEpisodeStreams) -> None:
    for idx in range(streams.num_frames):
        frame = {
            "state.joints": streams.state_joints[idx],
            "action.joints": streams.action_joints[idx],
            "action.joints_delta": streams.action_joints_delta[idx],
            "state.gripper": streams.state_gripper[idx],
            "action.gripper": streams.action_gripper[idx],
            "state.base": streams.state_base[idx],
            "action.base": streams.action_base[idx],
            "action.base_delta": streams.action_base_delta[idx],
            "state.ee_pose": streams.state_ee_pose[idx],
            "action.ee_pose": streams.action_ee_pose[idx],
            "action.ee_pose_delta": streams.action_ee_pose_delta[idx],
            "observation.state": streams.observation_state_16dof[idx],
            "action": streams.action_16dof[idx],
            "video.overlook_camera_view": streams.rgb_overlook[idx],
            "video.left_camera_view": streams.rgb_left_wrist[idx],
            "video.right_camera_view": streams.rgb_right_wrist[idx],
            "task": streams.instruction,
        }
        dataset.add_frame(frame)
    dataset.save_episode()


def export_from_staging_manifest(
    *,
    staging_manifest: Path,
    output_dir: Path,
    repo_id: str | None = None,
    fps: int = 15,
    robot_type: str = "lift2",
    overwrite: bool = False,
    summary_out: Path | None = None,
    image_writer_threads: int = 4,
) -> dict[str, Any]:
    specs = load_lerobot_staging_manifest(staging_manifest)
    resolved_out = output_dir.resolve()
    if resolved_out.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {resolved_out}")
        shutil.rmtree(resolved_out)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=repo_id or resolved_out.name,
        fps=int(fps),
        root=resolved_out,
        robot_type=robot_type,
        features=build_lerobot_features(),
        use_videos=True,
        tolerance_s=1e-4,
        image_writer_processes=0,
        image_writer_threads=max(1, int(image_writer_threads)),
        video_backend="ffmpeg",
    )
    total_frames = 0
    try:
        for spec in specs:
            streams = extract_synchronized_episode_streams(
                npz_path=Path(spec.npz_path),
                manifest=spec.manifest,
                instruction=spec.instruction,
                base_locked=spec.base_locked,
                expected_height=CAMERA_HEIGHT,
                expected_width=CAMERA_WIDTH,
            )
            _append_episode(dataset, streams)
            total_frames += streams.num_frames
    finally:
        dataset.stop_image_writer()

    summary = {
        "schema_version": "trajectory-forge-lerobot-export/v0.1",
        "repo_id": dataset.repo_id,
        "output_dir": str(resolved_out),
        "fps": int(fps),
        "total_episodes": len(specs),
        "total_frames": total_frames,
    }
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export verified trajectory-forge .npz episodes to LeRobot v2.1."
    )
    parser.add_argument("--staging-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--robot-type", default="lift2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-out", default=None, type=Path)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    args = parser.parse_args(argv)

    summary = export_from_staging_manifest(
        staging_manifest=args.staging_manifest,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        overwrite=args.overwrite,
        summary_out=args.summary_out,
        image_writer_threads=args.image_writer_threads,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
