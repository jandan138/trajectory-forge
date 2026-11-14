# 2026-11-14 `r4` Out-of-Domain Suite & Dual-Mode Remediation Live Verification Record

## Verification Scope & Execution Environment

- **Control Plane (CPU)**: `/cpfs/user/zhuzihou/dev/trajectory-forge` using `/cpfs/user/zhuzihou/conda-managed/envs/embodied-eval-os-py310/bin/python`.
- **GPU Simulation Nodes**:
  - Local RTX 4090 GPU via `--backend isaac-genmanip` (Isaac Sim `4.1` + `GenManip` + CUDA CuRobo).
  - Remote RTX 4090 GPU `root@10.150.168.71 -p 22` via `--backend ssh-isaac-genmanip`.
- **Upstream `r4` Dataset (`2026-11-13`)**:
  - Canonical `scene_gate`: `/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/out_of_domain_factory_r4_20261113/scene_gate/packages`
  - Mode 2 strict ablation: `/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/out_of_domain_factory_r4_20261113/mode2_strict_ablation/packages`
  - Mode 2 pose-adjusted ablation: `/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/out_of_domain_factory_r4_20261113/mode2_pose_ablation/packages`
- **Embodiment**: `lift2_r5a_bimanual` (`configs/embodiments/lift2_r5a_bimanual.yaml`, `runtime_robot_type: manip/lift2/R5a`, `16` DOFs).
- **Suite Definition**: [r4_out_of_domain_suite.yaml](../../configs/benchmarks/r4_out_of_domain_suite.yaml) (alongside prior `r3` baseline [2026-11-13-r3-privileged-mode-live-verification.md](2026-11-13-r3-privileged-mode-live-verification.md) and `GEN-010` workstation [2026-11-14-real2sim-workstation-live-verification.md](2026-11-14-real2sim-workstation-live-verification.md)).

---

## Dual-Mode Remediation Policy Matrix

| Run | Task & Package Tier | Policy (`mode`, `allow_pose_adjustment`) | Controller | Stages Passed | Final Score | Steps | Snapshots Used | Firewall Verified | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| Run 1 | `specialty_coffee_bar_creatista_prep` (`r4` `scene_gate`) | `physics_tuning`, `False` | [creatista_prep_privileged.py](../../examples/controllers/r3/creatista_prep_privileged.py) | `6/6` (`hold=True`) | `1.0` | `1775` | `0` (gate & synth) | `True` | **`succeeded=True`** |
| Run 2 | `modern_office_brother_calibrate` (`r4` `mode2_strict_ablation`) | `physics_tuning`, `False` | [brother_calibrate_privileged.py](../../examples/controllers/r3/brother_calibrate_privileged.py) | Stage `1` only (`0.0` seq) | `0.0` | `1280` | `4` (`privileged_search`) | `False` | **`BLOCKED_BY_REMEDIATION_POLICY`** (`attempts_used=1`) |
| Run 3 | `modern_office_brother_calibrate` (`r4` `mode2_pose_ablation`) | `physics_tuning`, `True` | [brother_calibrate_pose_adjusted_privileged.py](../../examples/controllers/r4/brother_calibrate_pose_adjusted_privileged.py) | `5/5` | `1.0` | `1449` | `0` (gate & synth) | `True` | **`succeeded=True`** |
| Run 4 | `modern_office_brother_calibrate` (`r4` `scene_gate`) | `scene_codesign`, `False` | [brother_calibrate_privileged.py](../../examples/controllers/r3/brother_calibrate_privileged.py) | `5/5` | `1.0` | `1566` | `0` (gate & synth) | `True` | **`succeeded=True`** |

---

## 1. Run 1: `specialty_coffee_bar_creatista_prep` — Mode 2 Strict (`physics_tuning`, `allow_pose_adjustment=False`)

Executed via `trajectory_forge.cli solve --backend isaac-genmanip --remediation-mode physics_tuning --max-steps 2600 --max-attempts 1 --render-gate-media`. Zero visual, part, or initial pose edits were needed.

- **Report**: `external_artifacts/rollouts/reports/specialty_coffee_bar_creatista_prep_r4_solve_report.json` (`succeeded: true`, `attempts_used: 1`).
- **`full_horizon_gate` (`specialty_coffee_bar_creatista_prep_a1_gate`)**: `status: "succeeded"`, `final_score: 1.0`, `steps_executed: 1775`, `snapshots_used: 0`, `firewall_verified: true`.
- **`trajectory_synthesis` (`specialty_coffee_bar_creatista_prep_a1_synth`)**:
  - `status: "succeeded"`, `final_score: 1.0`, `stages_completed_sequentially: 6`, `end_of_episode_all_stages_hold: true`, `blocking_stage: null`.
  - `steps_executed: 1775`, `snapshots_used: 0`, `firewall_verified: true`.
  - Stage progression (`step_index`):
    - Stage `0` (`stage0_tray_extended`, `drip_tray = -0.03754 m`): step `160` (`score = 0.16667`)
    - Stage `1` (`stage1_cup_seated`, `espresso_cup.xyz = [0.01031, -0.19363, 0.79545]`): step `448` (`score = 0.33333`)
    - Stage `2` (`stage2_capsule_inserted`, `coffee_capsule.xyz = [0.00083, -0.12945, 1.007]`): step `748` (`score = 0.5`)
    - Stage `3` (`stage3_lever_locked`, `brew_lever = -0.0 rad`): step `896` (`score = 0.66667`)
    - Stage `4` (`stage4_dial_selected`, 11 open-jaw ratchet strokes to `select_dial = 1.32764 rad`): step `1620` (`score = 0.83333`)
    - Stage `5` (`stage5_dial_pressed`, `dial_press = 0.00187 m`): step `1748` (`score = 1.0`)
  - **`TrajectoryEpisodeManifest`**: `continuous_from_t0: true`, `privileged_channels_recorded: false`, `num_steps: 1775`, `recorded_channels: ["proprio", "action", "base_target", "torso_lift", "timestamps_s", "rgb_head"]`.
  - **Exported `.npz`**: `external_artifacts/rollouts/episodes/specialty_coffee_bar_creatista_prep_specialty_coffee_bar_creatista_prep_a1_synth/episode_trajectory.npz` (`384,090,465` bytes):
    - `proprio`: `(1775, 16)`, `float32`; `action`: `(1775, 16)`, `float32`; `base_target`: `(1775, 3)`, `float32`; `torso_lift`: `(1775, 1)`, `float32`; `timestamps_s`: `(1775,)`, `float64`; `rgb_head`: `(221, 720, 1280, 3)`, `uint8`.
  - **Rendered Media**: `external_artifacts/rollouts/jobs/specialty_coffee_bar_creatista_prep_a1_synth/media/progression.mp4` (`h264`, `1280x720`, `15/1`, `236` frames, `15.733333s`, `1,357,476` bytes) and `final.png` visually inspected.

---

## 2. Run 2: `modern_office_brother_calibrate` (`mode2_strict_ablation`) — Mode 2 Strict (`physics_tuning`, `allow_pose_adjustment=False`)

Tests strict Mode 2 policy enforcement when `a5_media_pack` (`[-0.41503, -0.151817, 0.757]`, `0.140 m x 0.173 m x 0.022 m`) and `calibration_sheet` (`[0.415431, -0.142128, 0.757]`) sit flat in the desk center with `grasp = (0, 0, 0.011)` / `(0, 0, 0.003)` and no handle colliders or pose overhang.

- **Report**: `external_artifacts/rollouts/reports/modern_office_brother_calibrate_r4_mode2_strict_report.json` (`succeeded: false`, `attempts_used: 1`, `stop_on_policy_block: true`).
- **`history[0]` (`modern_office_brother_calibrate_a1_search`)**:
  - `status: "failed"`, `final_score: 0.0`, `steps_executed: 1280`, `snapshots_used: 4`, `blocking_stage: 0`, `stage_scores: [0.0, 1.0, 0.0, 0.0, 0.0]` (`paper_cassette = -0.0031 m`).
  - `failure_code: "BLOCKED_BY_REMEDIATION_POLICY"`, `root_failure_code: "UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE"`.
  - `remediation_category: "pose_adjustment"`, `required_remediation_mode: "scene_codesign"`, `requires_pose_adjustment: true`, `policy_blocked: true`.
- Confirmed that `PrivilegedSolverHarness` halts cleanly on Attempt 1 without wasting additional search attempts once `policy_blocked=True` is raised.

---

## 3. Run 3: `modern_office_brother_calibrate` (`mode2_pose_ablation`) — Mode 2 With Pose Sub-Switch (`physics_tuning`, `allow_pose_adjustment=True`)

Executed via `trajectory_forge.cli solve --backend ssh-isaac-genmanip --remediation-mode physics_tuning --allow-pose-adjustment --max-steps 2600 --max-attempts 1 --render-gate-media`. Zero added meshes or handles: initial poses shifted to front tabletop overhang (`a5_media_pack` at `[-0.36, -0.372, 0.757]`, `calibration_sheet` at `[0.38, -0.367, 0.757]`) so the parallel-jaw gripper pinches the `0.022 m` and `0.006 m` Z-thickness edges directly.

- **Report**: `external_artifacts/rollouts/reports/modern_office_brother_calibrate_r4_mode2_pose_report.json` (`succeeded: true`, `attempts_used: 1`).
- **`full_horizon_gate` (`modern_office_brother_calibrate_a1_gate`)**: `status: "succeeded"`, `final_score: 1.0`, `steps_executed: 1449`, `snapshots_used: 0`, `firewall_verified: true`.
- **`trajectory_synthesis` (`modern_office_brother_calibrate_a1_synth`)**:
  - `status: "succeeded"`, `final_score: 1.0`, `stages_completed_sequentially: 5`, `blocking_stage: null`.
  - `steps_executed: 1449`, `snapshots_used: 0`, `firewall_verified: true`.
  - Stage progression (`step_index`):
    - Stage `0` (`stage0_media_loaded`, `a5_media_pack.xyz = [-0.0029, -0.1836, 0.769]`): step `348` (`score = 0.2`)
    - Stage `1` (`stage1_cassette_closed`, `paper_cassette = -0.00311 m`): step `552` (`score = 0.4`)
    - Stage `2` (`stage2_sheet_aligned`, `calibration_sheet.xyz = [0.00575, 0.07824, 1.034]`): step `919` (`score = 0.6`)
    - Stage `3` (`stage3_lid_closed`, `scanner_lid = -0.0 rad`): step `1364` (`score = 0.8`)
    - Stage `4` (`stage4_copy_started`, `button_start = -0.00226 m`): step `1420` (`score = 1.0`)
  - **`TrajectoryEpisodeManifest`**: `continuous_from_t0: true`, `privileged_channels_recorded: false`, `num_steps: 1449`, `recorded_channels: ["proprio", "action", "base_target", "torso_lift", "timestamps_s", "rgb_head"]`.
  - **Exported `.npz`**: `external_artifacts/rollouts/mode2_pose/episodes/modern_office_brother_calibrate_modern_office_brother_calibrate_a1_synth/episode_trajectory.npz` (`308,625,181` bytes):
    - `proprio`: `(1449, 16)`, `float32`; `action`: `(1449, 16)`, `float32`; `base_target`: `(1449, 3)`, `float32`; `torso_lift`: `(1449, 1)`, `float32`; `timestamps_s`: `(1449,)`, `float64`; `rgb_head`: `(181, 720, 1280, 3)`, `uint8`.
  - **Rendered Media**: `external_artifacts/rollouts/mode2_pose/jobs/modern_office_brother_calibrate_a1_synth/media/progression.mp4` (`h264`, `1280x720`, `15/1`, `195` frames, `13.000000s`, `1,852,482` bytes) and `final.png` visually inspected.

---

## 4. Run 4: `modern_office_brother_calibrate` (`scene_gate`) — Mode 1 (`scene_codesign`)

Executed via `trajectory_forge.cli solve --backend isaac-genmanip --remediation-mode scene_codesign --max-steps 2600 --max-attempts 1 --render-gate-media` on canonical desk-center placements (`a5_media_pack` at `[-0.41503, -0.151817, 0.757]`, `calibration_sheet` at `[0.415431, -0.142128, 0.757]`) with upstream co-designed grasp handles (`named_frame("grasp")` at `(0, 0, 0.030)` and `(0, -0.054, 0.0125)`).

- **Report**: `external_artifacts/rollouts/reports/modern_office_brother_calibrate_r4_scene_codesign_report.json` (`succeeded: true`, `attempts_used: 1`).
- **`full_horizon_gate` (`modern_office_brother_calibrate_a1_gate`)**: `status: "succeeded"`, `final_score: 1.0`, `steps_executed: 1566`, `snapshots_used: 0`, `firewall_verified: true`.
- **`trajectory_synthesis` (`modern_office_brother_calibrate_a1_synth`)**:
  - `status: "succeeded"`, `final_score: 1.0`, `stages_completed_sequentially: 5`, `blocking_stage: null`.
  - `steps_executed: 1566`, `snapshots_used: 0`, `firewall_verified: true`.
  - Stage progression (`step_index`):
    - Stage `0` (`stage0_media_loaded`, `a5_media_pack.xyz = [0.0127, -0.1726, 0.769]`): step `404` (`score = 0.2`)
    - Stage `1` (`stage1_cassette_closed`, `paper_cassette = -0.00309 m`): step `660` (`score = 0.4`)
    - Stage `2` (`stage2_sheet_aligned`, `calibration_sheet.xyz = [0.00133, 0.07454, 1.034]`): step `1068` (`score = 0.6`)
    - Stage `3` (`stage3_lid_closed`, `scanner_lid = -0.0 rad`): step `1462` (`score = 0.8`)
    - Stage `4` (`stage4_copy_started`, `button_start = -0.00226 m`): step `1536` (`score = 1.0`)
  - **`TrajectoryEpisodeManifest`**: `continuous_from_t0: true`, `privileged_channels_recorded: false`, `num_steps: 1566`, `recorded_channels: ["proprio", "action", "base_target", "torso_lift", "timestamps_s", "rgb_head"]`.
  - **Exported `.npz`**: `external_artifacts/rollouts/episodes/modern_office_brother_calibrate_modern_office_brother_calibrate_a1_synth/episode_trajectory.npz` (`336,695,201` bytes):
    - `proprio`: `(1566, 16)`, `float32`; `action`: `(1566, 16)`, `float32`; `base_target`: `(1566, 3)`, `float32`; `torso_lift`: `(1566, 1)`, `float32`; `timestamps_s`: `(1566,)`, `float64`; `rgb_head`: `(195, 720, 1280, 3)`, `uint8`.
  - **Rendered Media**: `external_artifacts/rollouts/jobs/modern_office_brother_calibrate_a1_synth/media/progression.mp4` (`h264`, `1280x720`, `15/1`, `209` frames, `13.933333s`, `1,857,157` bytes) and `final.png` visually inspected.
