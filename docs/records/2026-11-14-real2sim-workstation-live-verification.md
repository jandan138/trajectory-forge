# 2026-11-14 `GEN-010` Real-to-Sim Open-Office Workstation Live Verification Record

## Verification Scope & Execution Environment

- **Control Plane (CPU)**: `/cpfs/user/zhuzihou/dev/trajectory-forge` using `/cpfs/user/zhuzihou/conda-managed/envs/embodied-eval-os-py310/bin/python`.
- **GPU Simulation Node**: `root@10.150.168.71 -p 22` (RTX 4090, Isaac Sim `4.1` + `GenManip` + CUDA CuRobo, via `--backend ssh-isaac-genmanip`).
- **Upstream Dataset (`GEN-010`)**: `/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/real2sim_office_desk_20261113/scene_gate/packages/real2sim_office_desk_riser_organize` (`DesktopDualDrawerRiserRig` updated with positive-scale `ColHandleNeck` + `ColHandleRing` omega ring-pull colliders, re-qualified via `ConvertAsset (isaac41)` and `verify_genmanip_predicate_canary.py --profile real2sim` `0.00 -> [0.2, 0.4, 0.6, 0.8, 1.0]`, `max_drift = 0.01 mm`).
- **Embodiment**: `lift2_r5a_bimanual` (`configs/embodiments/lift2_r5a_bimanual.yaml`, `runtime_robot_type: manip/lift2/R5a`, `16` DOFs).

---

## Summary Matrix

| Task | Controller | Sequential Stages Passed | Final Score | Steps | Snapshots Used | Firewall Verified | Diagnostics (`failure_code`) |
|---|---|---|---|---|---|---|---|
| `real2sim_office_desk_riser_organize` | [office_desk_riser_organize_privileged.py](../../examples/controllers/real2sim/office_desk_riser_organize_privileged.py) | `5/5` (`240 -> 548 -> 764 -> 1036 -> 1328`) | `1.0` | `1448` | `0` (gate & synth) | `True` | None (`succeeded=True`) |

---

## 1. Upstream Physical Handle Collider Remediation (`DesktopDualDrawerRiserRig`)

During pre-rollout USD collision inspection, `DesktopDualDrawerRiserRig`'s `Ø29 mm` `CircularOmegaRingPull` (`handle_right` at `[0.121, -0.122, 0.030]`) had visual geometry (`Visual/CircularOmegaRingPull`) but no `UsdPhysics.CollisionAPI` collider under `Links/DrawerRight/Collision` (`ColWallFront` recessed `6 mm` inside `ColMidShelf` with `4.0 mm` vertical clearance).

- Added two positive-scale `purpose="guide"` box colliders (`ColHandleNeck` + `ColHandleRing`, flush at `y = -0.1246 m`) on `DrawerLeft` and `DrawerRight` in `scenario-forge`'s `build_desktop_dual_drawer_riser_rig.py`, re-qualified via `ConvertAsset (isaac41)`, and re-exported `real2sim_office_desk_riser_organize`.
- Added automated `UPSTREAM_MISSING_HANDLE_COLLIDER` detection in `IsaacGenManipBackend.diagnose_blocking_stage()` and sequential `MetricsManager`-based `blocking_stage` calculation (`compute_sequential_blocking_stage()`) so cyclic open-then-close drawer tasks report accurate sequential progress.

---

## 2. `real2sim_office_desk_riser_organize` — Full Closed-Loop Solve & Trajectory Synthesis

Executed via `trajectory_forge.cli solve --backend ssh-isaac-genmanip --max-steps 2600 --max-attempts 1 --render-gate-media`.

- **Report**: `external_artifacts/rollouts/reports/real2sim_office_desk_riser_organize_solve_report.json` (`succeeded: true`, `attempts_used: 1`).
- **`full_horizon_gate` (`real2sim_office_desk_riser_organize_a1_gate`)**:
  - `status: "succeeded"`, `final_score: 1.0`, `stages_completed_sequentially: 5`, `blocking_stage: null`.
  - `steps_executed: 1448`, `snapshots_used: 0`, `firewall_verified: true`.
- **`trajectory_synthesis` (`real2sim_office_desk_riser_organize_a1_synth`)**:
  - `status: "succeeded"`, `final_score: 1.0`, `stages_completed_sequentially: 5`, `blocking_stage: null`.
  - `steps_executed: 1448`, `snapshots_used: 0`, `firewall_verified: true`.
  - Stage progression timestamps (`step_index`):
    - Stage `0` (`stage0_drawer_opened`, `drawer_right = 0.15888 m` in `[0.120, 0.165]`): step `240` (`score = 0.2`)
    - Stage `1` (`stage1_chocopie_stored`, `chocopie_pack` on drawer floor `z = 0.766 m`): step `548` (`score = 0.4`)
    - Stage `2` (`stage2_pen_stored`, `blue_ballpoint_pen` on drawer floor `z = 0.766 m`): step `764` (`score = 0.6`)
    - Stage `3` (`stage3_drawer_closed`, `drawer_right = 0.00175 m` in `[0.0, 0.025]`, both stored items carried inside at `z = 0.766 m`): step `1036` (`score = 0.8`)
    - Stage `4` (`stage4_mug_placed_upright`, `glass_water_mug.xyz = [0.46349, -0.11178, 0.75]`, tilt `< 0.6 deg` beside `sprite_bottle_500ml` `[0.578, -0.111, 0.75]`): step `1328` (`score = 1.0`)
  - **`TrajectoryEpisodeManifest`**: `continuous_from_t0: true`, `privileged_channels_recorded: false`, `num_steps: 1448`, `recorded_channels: ["proprio", "action", "base_target", "torso_lift", "timestamps_s", "rgb_head"]`.
  - **Exported `.npz`**: `external_artifacts/rollouts/episodes/real2sim_office_desk_riser_organize_real2sim_office_desk_riser_organize_a1_synth/episode_trajectory.npz` (`301,668,664` bytes):
    - `proprio`: `(1448, 16)`, `float32`
    - `action`: `(1448, 16)`, `float32`
    - `base_target`: `(1448, 3)`, `float32`
    - `torso_lift`: `(1448, 1)`, `float32`
    - `timestamps_s`: `(1448,)`, `float64`
    - `rgb_head`: `(181, 720, 1280, 3)`, `uint8`
  - **Rendered Media**: `external_artifacts/rollouts/jobs/real2sim_office_desk_riser_organize_a1_synth/media/progression.mp4` (`h264`, `1280x720`, `15/1`, `195` frames, `13.000s`, `1,547,427` bytes) and stage keyframe PNGs (`t0_initial.png` .. `final.png`) visually inspected.
