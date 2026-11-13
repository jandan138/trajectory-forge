# 2026-11-13 `r3` Out-of-Domain Privileged Mode Live Verification Record

## Verification Scope & Execution Environment

- **Control Plane (CPU)**: `/cpfs/user/zhuzihou/dev/trajectory-forge` using `/cpfs/user/zhuzihou/conda-managed/envs/embodied-eval-os-py310/bin/python`.
- **GPU Simulation Node**: `root@10.150.168.71 -p 22` (RTX 4090, Isaac Sim `4.1` + `GenManip` + CUDA CuRobo, via `--backend ssh-isaac-genmanip`).
- **Upstream Dataset**: `/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/out_of_domain_factory_r3_20261113/packages` (`r3`, unmodified).
- **Embodiment**: `lift2_r5a_bimanual` (`configs/embodiments/lift2_r5a_bimanual.yaml`, `runtime_robot_type: manip/lift2/R5a`, `16` DOFs).

---

## Summary Matrix

| Task | Controller | Stages Passed | Final Score | Steps | Snapshots Used | Firewall Verified | Diagnostics (`failure_code`) |
|---|---|---|---|---|---|---|---|
| `electronics_qa_ingun_probe_lock` | [ingun_probe_lock_privileged.py](../../examples/controllers/r3/ingun_probe_lock_privileged.py) | `5/5` (`[1.0, 1.0, 1.0, 1.0, 1.0]`, hold=`True`) | `1.0` | `1757` | `0` (gate & synth) | `True` | None (`succeeded=True`) |
| `specialty_coffee_bar_creatista_prep` | [creatista_prep_privileged.py](../../examples/controllers/r3/creatista_prep_privileged.py) | Stages `0–2` (`[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]`) | `0.5` | `1119` | `5` (`privileged_search`) | `False` | `UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT` (Stage `3`) |
| `modern_office_brother_calibrate` | [brother_calibrate_privileged.py](../../examples/controllers/r3/brother_calibrate_privileged.py) | Stages `1` & `4` (`[0.0, 1.0, 0.0, 0.0, 1.0]`) | `0.0` | `967` | `4` (`privileged_search`) | `False` | `UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE` (Stage `0`) |

---

## 1. `electronics_qa_ingun_probe_lock` — Full Closed-Loop Solve & Trajectory Synthesis

Executed via `trajectory_forge.cli solve --backend ssh-isaac-genmanip --max-steps 2500 --max-attempts 1 --render-gate-media`.

- **Report**: `external_artifacts/rollouts/reports/electronics_qa_ingun_probe_lock_solve_report.json` (`succeeded: true`, `attempts_used: 1`).
- **`full_horizon_gate` (`electronics_qa_ingun_probe_lock_a1_gate`)**:
  - `status: "succeeded"`, `final_score: 1.0`, `stage_scores: [1.0, 1.0, 1.0, 1.0, 1.0]`, `end_of_episode_all_stages_hold: true`.
  - `steps_executed: 1757`, `snapshots_used: 0`, `firewall_verified: true`.
- **`trajectory_synthesis` (`electronics_qa_ingun_probe_lock_a1_synth`)**:
  - `status: "succeeded"`, `final_score: 1.0`, `stage_scores: [1.0, 1.0, 1.0, 1.0, 1.0]`, `end_of_episode_all_stages_hold: true`.
  - `steps_executed: 1757`, `snapshots_used: 0`, `firewall_verified: true`.
  - Final state observed: `pcba_carrier_board.xyz = [-0.00056, -0.01425, 0.883]`, `pressure_hood = -0.07651`, `clamp_lever = -0.07765`, `mode_toggle = -0.41227`, `test_start_button = -0.0021`.
  - **`TrajectoryEpisodeManifest`**: `continuous_from_t0: true`, `privileged_channels_recorded: false`, `num_steps: 1757`, `recorded_channels: ["proprio", "action", "base_target", "torso_lift", "timestamps_s", "rgb_head"]`.
  - **Exported `.npz`**: `external_artifacts/rollouts/episodes/electronics_qa_ingun_probe_lock_electronics_qa_ingun_probe_lock_a1_synth/episode_trajectory.npz` (`217,111,860` bytes):
    - `proprio`: `(1757, 16)`, `float32`
    - `action`: `(1757, 16)`, `float32`
    - `base_target`: `(1757, 3)`, `float32`
    - `torso_lift`: `(1757, 1)`, `float32`
    - `timestamps_s`: `(1757,)`, `float64`
    - `rgb_head`: `(219, 720, 1280, 3)`, `uint8`
  - **Rendered Media**: `external_artifacts/rollouts/jobs/electronics_qa_ingun_probe_lock_a1_synth/media/progression.mp4` (`h264`, `1280x720`, `15/1`, `233` frames, `15.534s`, `1,410,651` bytes) and `final.png` visually inspected.

---

## 2. `specialty_coffee_bar_creatista_prep` — Partial Solve + Upstream Physics Blocker

Executed via `cli solve` (`reports/specialty_coffee_bar_creatista_prep_solve_report.json`) and rendered `rollout run --mode privileged_search --render-media` (`job_id=r3_creatista_final_media`, `reports/specialty_coffee_bar_creatista_prep_media_rollout.json`).

- **Physically Verified Stages**:
  - Stage 0 (`drip_tray = -0.03756` in `[-0.045, -0.035]`) -> `1.0`
  - Stage 1 (`espresso_cup` at `[0.01744, -0.19154, 0.79579]`) -> `1.0`
  - Stage 2 (`coffee_capsule` at `[0.00051, -0.1296, 1.007]`) -> `1.0`
  - Stage 5 final attempt: `dial_press = 0.00318` (vs pressed `[0.0014, 0.0031]`).
- **Automated `SolvabilityDiagnostics`**:
  - `failure_code`: `UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT` (`stage_index: 3`)
  - `detail`: `Stage 3 joint 'creatista_machine.brew_lever' (BrewLeverJoint) has drive:angular:physics:damping=12.00 N*m*s/deg (~687.5 N*m*s/rad in PhysX); observed -1.2172 rad vs target band [-0.0873, 0.0000] rad.`
- **Rendered Media**: `external_artifacts/rollouts/jobs/r3_creatista_final_media/media/progression.mp4` (`h264`, `1280x720`, `15/1`, `12` frames, `0.800s`, `162,038` bytes) and `final.png` visually inspected.

---

## 3. `modern_office_brother_calibrate` — Partial Solve + Upstream Aperture & Damping Blockers

Executed via `cli solve` (`reports/modern_office_brother_calibrate_solve_report.json`) and rendered `rollout run --mode privileged_search --render-media` (`job_id=r3_brother_final_media`, `reports/modern_office_brother_calibrate_media_rollout.json`).

- **Physically Verified Stages**:
  - Stage 1 (`paper_cassette = -0.0031 m` in `[-0.012, 0.0]`) -> `1.0`
  - Stage 4 (`button_start = -0.00226 m` in `[-0.0025, -0.0011]`) -> `1.0`
  - `scanner_lid` remains at `-1.0821 rad` (Stage 3, `ScannerLidJoint` `drive:angular:physics:damping = 20.0 N*m*s/deg`).
- **Automated `SolvabilityDiagnostics`**:
  - `failure_code`: `UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE` (`stage_index: 0`)
  - `detail`: `Stage 0 object 'a5_media_pack' horizontal footprint (0.122 m x 0.160 m, z=0.022 m) exceeds lift2 max parallel-jaw gripper aperture 0.088 m.` (`calibration_sheet` similarly spans `0.130 m x 0.180 m x 0.0061 m` in USD.)
- **Rendered Media**: `external_artifacts/rollouts/jobs/r3_brother_final_media/media/progression.mp4` (`h264`, `1280x720`, `15/1`, `9` frames, `0.600s`, `175,310` bytes) and `final.png` visually inspected.
