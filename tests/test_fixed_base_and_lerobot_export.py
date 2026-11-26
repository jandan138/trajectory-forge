from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trajectory_forge.core.contracts import (
    BatchEpisodeOutcome,
    BatchEpisodeSpec,
    BatchSynthesisReport,
    BatchSynthesisRequest,
    ContractValidationError,
    FirewallViolationError,
    RobotSpawnSpec,
    ScenarioIntakeBrief,
    StageContractSpec,
    TrajectoryEpisodeManifest,
)
from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.sdk.context import run_controller_preflight
from trajectory_forge.trajectory.lerobot_converter import (
    INTERLEAVED_16DOF_ARM_INDICES,
    INTERLEAVED_16DOF_GRIPPER_INDICES,
    extract_synchronized_episode_streams,
    reconstruct_interleaved_16dof,
    split_interleaved_16dof,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_fixed_base_16dof_embodiment_card_and_base_lock() -> None:
    card = load_embodiment_card(
        REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_fixed_base_16dof.yaml"
    )
    assert card.schema_version == "trajectory-forge-embodiment-card/v0.1"
    assert card.embodiment_id == "lift2_r5a_fixed_base_16dof"
    assert card.action_dim == 16
    assert card.base_locked is True
    assert set(card.sensor_streams.keys()) >= {
        "proprio",
        "rgb_overlook",
        "rgb_left_wrist",
        "rgb_right_wrist",
    }
    for stream_name in ("rgb_overlook", "rgb_left_wrist", "rgb_right_wrist"):
        stream = card.sensor_streams[stream_name]
        assert stream.modality == "rgb"
        assert stream.rate_hz == 15.0
        assert stream.shape == (480, 640, 3)

    brief = ScenarioIntakeBrief(
        package_id="fixed_base_smoke",
        episode_name="000",
        instruction="Press button without base translation.",
        scene_usd_relpath="",
        robot_profile=card.runtime_robot_type,
        robot_spawn=RobotSpawnSpec(
            robot_id=card.embodiment_id,
            xyz=(0.0, -0.88, 0.31),
            wxyz=(0.7071067812, 0.0, 0.0, 0.7071067812),
            joint_positions=(0.0,) * 16,
        ),
        objects={},
        stages=(StageContractSpec(stage_index=0, metric_id="stage0", clauses=()),),
    )

    illegal_base_controller = """
def solve(ctx):
    ctx.kinematics.move_base_world(0.10, 0.0, 0.0, steps=10)
"""
    with pytest.raises(ContractValidationError, match="base_locked"):
        run_controller_preflight(
            illegal_base_controller,
            brief,
            card,
            mode="trajectory_synthesis",
        )

    illegal_torso_controller = """
def solve(ctx):
    ctx.kinematics.set_torso_lift(0.30, steps=10)
"""
    with pytest.raises(ContractValidationError, match="locks torso lift at 0.46"):
        run_controller_preflight(
            illegal_torso_controller,
            brief,
            card,
            mode="trajectory_synthesis",
        )

    stationary_controller = """
def solve(ctx):
    ctx.kinematics.set_torso_lift(0.46, steps=4)
    ctx.end_effector.set_posture("left_gripper", "open", steps=4)
    ctx.step(4)
"""
    res = run_controller_preflight(
        stationary_controller,
        brief,
        card,
        mode="trajectory_synthesis",
    )
    assert res.status == "succeeded"
    assert res.firewall_verified is True


def test_batch_synthesis_contracts_enforce_firewall_invariants() -> None:
    req = BatchSynthesisRequest(
        batch_id="ce_pilot_batch_001",
        scenario_package_ref="/tmp/pkg",
        embodiment_id="lift2_r5a_fixed_base_16dof",
        controller_code="def solve(ctx):\n    ctx.step(4)\n",
        episodes=(
            BatchEpisodeSpec(episode_id="ep_000", seed=0),
            BatchEpisodeSpec(
                episode_id="ep_001",
                seed=1,
                episode_metadata_relpath="tasks/t/001/episode_metadata.json",
                instruction_override="Set dial to 180 C",
            ),
        ),
        max_steps=1600,
        camera_stride=4,
        capture_wrist_cameras=True,
    )
    roundtrip = BatchSynthesisRequest.from_dict(req.to_dict())
    assert roundtrip == req

    with pytest.raises(FirewallViolationError):
        BatchSynthesisRequest(
            batch_id="bad_mode",
            scenario_package_ref="/tmp/pkg",
            embodiment_id="lift2_r5a_fixed_base_16dof",
            controller_code="def solve(ctx):\n    ctx.step(4)\n",
            episodes=(BatchEpisodeSpec(episode_id="ep_000", seed=0),),
            mode="privileged_search",
        )

    good_manifest = TrajectoryEpisodeManifest(
        episode_id="ep_000",
        scenario_id="ce_01",
        embodiment_id="lift2_r5a_fixed_base_16dof",
        num_steps=16,
        continuous_from_t0=True,
        privileged_channels_recorded=False,
        recorded_channels=(
            "proprio",
            "action",
            "rgb_overlook",
            "rgb_left_wrist",
            "rgb_right_wrist",
        ),
        artifact_relpath="/tmp/ep_000/episode_trajectory.npz",
    )
    with pytest.raises(FirewallViolationError):
        BatchEpisodeOutcome(
            episode_id="ep_000",
            seed=0,
            status="succeeded",
            final_score=1.0,
            steps_executed=16,
            snapshots_used=1,
            firewall_verified=False,
            trajectory_manifest=good_manifest,
        )

    report = BatchSynthesisReport(
        batch_id="ce_pilot_batch_001",
        scenario_package_ref="/tmp/pkg",
        embodiment_id="lift2_r5a_fixed_base_16dof",
        outcomes=(
            BatchEpisodeOutcome(
                episode_id="ep_000",
                seed=0,
                status="succeeded",
                final_score=1.0,
                steps_executed=16,
                snapshots_used=0,
                firewall_verified=True,
                trajectory_manifest=good_manifest,
            ),
            BatchEpisodeOutcome(
                episode_id="ep_001",
                seed=1,
                status="failed",
                final_score=0.5,
                steps_executed=20,
                snapshots_used=0,
                firewall_verified=False,
                failure_code="STAGE_INCOMPLETE",
            ),
        ),
        accepted_count=1,
        failed_count=1,
    )
    assert BatchSynthesisReport.from_dict(report.to_dict()) == report


def test_lerobot_channel_bijection_and_synchronized_extraction(tmp_path: Path) -> None:
    assert INTERLEAVED_16DOF_ARM_INDICES == (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13)
    assert INTERLEAVED_16DOF_GRIPPER_INDICES == (6, 7, 14, 15)

    vec16 = np.arange(32, dtype=np.float32).reshape(2, 16) * 0.1
    joints12, gripper4 = split_interleaved_16dof(vec16)
    assert joints12.shape == (2, 12)
    assert gripper4.shape == (2, 4)
    np.testing.assert_allclose(joints12[:, :6], vec16[:, 0:6])
    np.testing.assert_allclose(joints12[:, 6:12], vec16[:, 8:14])
    np.testing.assert_allclose(gripper4[:, :2], vec16[:, 6:8])
    np.testing.assert_allclose(gripper4[:, 2:4], vec16[:, 14:16])

    split19 = np.concatenate([joints12, gripper4, np.zeros((2, 3), dtype=np.float32)], axis=-1)
    rebuilt16 = reconstruct_interleaved_16dof(split19)
    np.testing.assert_allclose(rebuilt16, vec16)

    # 16 ticks @ 60 Hz, camera frames every 4 ticks at [3, 7, 11, 15] -> 4 synchronized 15 Hz frames
    t_steps = 16
    cam_steps = np.array([3, 7, 11, 15], dtype=np.int64)
    proprio = (np.arange(t_steps * 16, dtype=np.float32).reshape(t_steps, 16)) * 0.01
    action = proprio + 0.05
    base_target = np.zeros((t_steps, 3), dtype=np.float32)
    timestamps_s = (np.arange(t_steps, dtype=np.float64) + 1) / 60.0
    rng = np.random.default_rng(7)
    rgb_overlook = rng.integers(20, 220, size=(4, 480, 640, 3), dtype=np.uint8)
    rgb_left = rng.integers(20, 220, size=(4, 480, 640, 3), dtype=np.uint8)
    rgb_right = rng.integers(20, 220, size=(4, 480, 640, 3), dtype=np.uint8)
    state_ee = np.zeros((t_steps, 14), dtype=np.float32)
    action_ee = np.ones((t_steps, 14), dtype=np.float32) * 0.25

    npz_path = tmp_path / "episode_trajectory.npz"
    np.savez_compressed(
        npz_path,
        proprio=proprio,
        action=action,
        base_target=base_target,
        timestamps_s=timestamps_s,
        camera_step_indices=cam_steps,
        rgb_overlook=rgb_overlook,
        rgb_left_wrist=rgb_left,
        rgb_right_wrist=rgb_right,
        state_ee_pose=state_ee,
        action_ee_pose=action_ee,
    )
    manifest = TrajectoryEpisodeManifest(
        episode_id="ce_01_seed_0",
        scenario_id="ce_01_espresso_capsule_brew",
        embodiment_id="lift2_r5a_fixed_base_16dof",
        num_steps=t_steps,
        continuous_from_t0=True,
        privileged_channels_recorded=False,
        recorded_channels=(
            "proprio",
            "action",
            "base_target",
            "timestamps_s",
            "camera_step_indices",
            "rgb_overlook",
            "rgb_left_wrist",
            "rgb_right_wrist",
            "state_ee_pose",
            "action_ee_pose",
        ),
        artifact_relpath=str(npz_path),
    )
    streams = extract_synchronized_episode_streams(
        npz_path=npz_path,
        manifest=manifest,
        instruction="Open slider, insert capsule, and brew espresso.",
        base_locked=True,
    )
    assert streams.num_frames == 4
    assert streams.instruction == "Open slider, insert capsule, and brew espresso."
    assert streams.state_joints.shape == (4, 12)
    assert streams.action_joints.shape == (4, 12)
    assert streams.action_joints_delta.shape == (4, 12)
    assert streams.state_gripper.shape == (4, 4)
    assert streams.action_gripper.shape == (4, 4)
    assert streams.state_base.shape == (4, 3)
    assert streams.action_base.shape == (4, 3)
    assert streams.action_base_delta.shape == (4, 3)
    assert streams.observation_state_16dof.shape == (4, 16)
    assert streams.action_16dof.shape == (4, 16)
    assert streams.rgb_overlook.shape == (4, 480, 640, 3)
    assert streams.rgb_left_wrist.shape == (4, 480, 640, 3)
    assert streams.rgb_right_wrist.shape == (4, 480, 640, 3)
    np.testing.assert_allclose(streams.observation_state_16dof, proprio[cam_steps])
    np.testing.assert_allclose(streams.action_16dof, action[cam_steps])
    np.testing.assert_allclose(
        streams.action_joints_delta,
        streams.action_joints - streams.state_joints,
    )
    np.testing.assert_allclose(streams.state_base, 0.0)
    np.testing.assert_allclose(streams.action_base, 0.0)

    # Reject mismatched camera frame counts
    bad_npz = tmp_path / "bad_cam_count.npz"
    np.savez_compressed(
        bad_npz,
        proprio=proprio,
        action=action,
        base_target=base_target,
        timestamps_s=timestamps_s,
        camera_step_indices=cam_steps,
        rgb_overlook=rgb_overlook,
        rgb_left_wrist=rgb_left[:3],
        rgb_right_wrist=rgb_right,
    )
    with pytest.raises(ContractValidationError, match="frame count"):
        extract_synchronized_episode_streams(
            npz_path=bad_npz,
            manifest=manifest,
            instruction="bad",
            base_locked=True,
        )

    # Reject privileged prefix in manifest
    priv_manifest_dict = manifest.to_dict()
    priv_manifest_dict["recorded_channels"] = ["proprio", "privileged_object_pose"]
    manifest_json = tmp_path / "priv.json"
    manifest_json.write_text(json.dumps(priv_manifest_dict), encoding="utf-8")
    priv_manifest = TrajectoryEpisodeManifest.from_dict(
        json.loads(manifest_json.read_text(encoding="utf-8"))
    )
    with pytest.raises(FirewallViolationError, match="privileged_"):
        extract_synchronized_episode_streams(
            npz_path=npz_path,
            manifest=priv_manifest,
            instruction="priv",
            base_locked=True,
        )


def test_executor_batch_command_and_live_lerobot_v2_export_roundtrip(tmp_path: Path) -> None:
    from trajectory_forge import cli
    from trajectory_forge.adapters.isaac_genmanip.executor import (
        IsaacGenManipWorkerConfig,
        LocalIsaacGenManipExecutor,
    )
    from trajectory_forge.adapters.lerobot_v2.executor import DEFAULT_LEROBOT_PYTHON

    cfg = IsaacGenManipWorkerConfig(output_root=tmp_path / "out")
    executor = LocalIsaacGenManipExecutor(cfg)
    batch_cmd = executor.build_batch_command(tmp_path / "breq.json", tmp_path / "bres.json")
    assert batch_cmd[1:4] == [
        "-u",
        "-m",
        "trajectory_forge.adapters.isaac_genmanip.batch_runner_entry",
    ]

    # Build a synthetic verified BatchSynthesisReport pointing to 2 synthetic .npz episodes
    rng = np.random.default_rng(11)
    t_steps = 16
    cam_steps = np.array([0, 4, 8, 12], dtype=np.int64)
    outcomes: list[BatchEpisodeOutcome] = []
    for idx in range(2):
        proprio = np.full((t_steps, 16), 0.1 * (idx + 1), dtype=np.float32)
        action = proprio + 0.02
        npz_path = tmp_path / f"ep_{idx}.npz"
        np.savez_compressed(
            npz_path,
            proprio=proprio,
            action=action,
            base_target=np.zeros((t_steps, 3), dtype=np.float32),
            timestamps_s=np.arange(t_steps, dtype=np.float64) / 60.0,
            camera_step_indices=cam_steps,
            rgb_overlook=rng.integers(30, 200, size=(4, 480, 640, 3), dtype=np.uint8),
            rgb_left_wrist=rng.integers(30, 200, size=(4, 480, 640, 3), dtype=np.uint8),
            rgb_right_wrist=rng.integers(30, 200, size=(4, 480, 640, 3), dtype=np.uint8),
            state_ee_pose=np.zeros((t_steps, 14), dtype=np.float32),
            action_ee_pose=np.zeros((t_steps, 14), dtype=np.float32),
        )
        ep_id = f"smoke_ep_{idx:04d}"
        manifest = TrajectoryEpisodeManifest(
            episode_id=ep_id,
            scenario_id="ce_01_smoke",
            embodiment_id="lift2_r5a_fixed_base_16dof",
            num_steps=t_steps,
            continuous_from_t0=True,
            privileged_channels_recorded=False,
            recorded_channels=(
                "proprio",
                "action",
                "base_target",
                "timestamps_s",
                "state_ee_pose",
                "action_ee_pose",
                "camera_step_indices",
                "rgb_overlook",
                "rgb_left_wrist",
                "rgb_right_wrist",
            ),
            artifact_relpath=str(npz_path),
        )
        outcomes.append(
            BatchEpisodeOutcome(
                episode_id=ep_id,
                seed=idx,
                status="succeeded",
                final_score=1.0,
                steps_executed=t_steps,
                snapshots_used=0,
                firewall_verified=True,
                trajectory_manifest=manifest,
            )
        )

    report = BatchSynthesisReport(
        batch_id="smoke_batch",
        scenario_package_ref="ce_01_smoke",
        embodiment_id="lift2_r5a_fixed_base_16dof",
        outcomes=tuple(outcomes),
        accepted_count=2,
        failed_count=0,
    )
    report_path = tmp_path / "batch_report.json"
    report_path.write_text(json.dumps(report.to_dict()), encoding="utf-8")

    if not DEFAULT_LEROBOT_PYTHON.is_file():
        pytest.skip(f"LeRobot python not installed: {DEFAULT_LEROBOT_PYTHON}")

    lerobot_dir = tmp_path / "lerobot_ds"
    rc = cli.main(
        [
            "export-lerobot",
            "--batch-report",
            str(report_path),
            "--output-dir",
            str(lerobot_dir),
            "--repo-id",
            "ce_op_smoke_ds",
            "--default-instruction",
            "Brew espresso using slider and start button.",
            "--overwrite",
        ]
    )
    assert rc == 0
    assert (lerobot_dir / "meta" / "info.json").is_file()
    assert (lerobot_dir / "meta" / "episodes.jsonl").is_file()
    assert (lerobot_dir / "meta" / "tasks.jsonl").is_file()
    assert (lerobot_dir / "meta" / "episodes_stats.jsonl").is_file()
    assert (lerobot_dir / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    assert (
        lerobot_dir
        / "videos"
        / "chunk-000"
        / "video.overlook_camera_view"
        / "episode_000000.mp4"
    ).is_file()

    # Verify loading in LeRobotDataset (openpi venv)
    import subprocess

    check_code = (
        "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; "
        f"ds = LeRobotDataset('ce_op_smoke_ds', root={str(lerobot_dir)!r}); "
        "assert len(ds) == 8; "
        "sample = ds[0]; "
        "assert tuple(sample['video.overlook_camera_view'].shape) == (3, 480, 640); "
        "assert tuple(sample['video.left_camera_view'].shape) == (3, 480, 640); "
        "assert tuple(sample['video.right_camera_view'].shape) == (3, 480, 640); "
        "assert tuple(sample['state.joints'].shape) == (12,); "
        "assert tuple(sample['action.joints'].shape) == (12,); "
        "assert tuple(sample['state.gripper'].shape) == (4,); "
        "assert tuple(sample['state.base'].shape) == (3,); "
        "assert tuple(sample['observation.state'].shape) == (16,); "
        "assert tuple(sample['action'].shape) == (16,); "
        "print('LEROBOT_LOAD_OK', len(ds), sample['task'])"
    )
    proc = subprocess.run(
        [str(DEFAULT_LEROBOT_PYTHON), "-c", check_code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "LEROBOT_LOAD_OK 8" in proc.stdout

