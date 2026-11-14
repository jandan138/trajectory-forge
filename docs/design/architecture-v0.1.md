# Trajectory Forge `v0.1` Architecture & Privileged Mode Contract

## 1. Benchmark Baseline: *Embodied First Exam* Privileged Mode

`trajectory-forge` adopts the **Privileged Setting** from *Embodied First Exam* as its solver benchmark:

- **Information Surface**: During `privileged_search`, the agent receives full simulator-authored semantic geometry (`scenario_forge_runtime_contract`: `named_frames`, `dof_mapping`, `states`, initial poses, and `MetricsManager` stage goals) plus unlimited stage checkpoint resets.
- **Purpose**:
  1. **Solvability Certification (Upper Bound)**: Certify whether an upstream `scenario-forge` package is physically and kinematically solvable by an articulated robot (`0.00 -> 1.00` without teleportation).
  2. **Structured Diagnostics**: Distinguish scene/asset defects (`ik_infeasible`, `gripper_collision_with_hood`, `insufficient_friction`) from planner defects and route actionable hints back to `scenario-forge`.
  3. **Demonstration Synthesis**: Once a Python controller solves a task, run it continuously from `t=0` under `trajectory_synthesis` across randomized seeds to produce clean sensorimotor datasets.

### Three Execution Modes & The Privileged-to-Sensorimotor Firewall

| Mode | Privileged `world` Reads | `ctx.checkpoint` Snapshots | Purpose | Allowed to Emit Training Trajectories? |
|---|---|---|---|---|
| `privileged_search` | Yes | Allowed (`save_snapshot` / `restore_snapshot`) | Rapid per-stage failure repair without replaying earlier stages | **No** |
| `full_horizon_gate` | Yes | **Forbidden** (`FirewallViolationError`) | Zero-snapshot continuous `t=0 -> 1.00` verification | Manifest verification only |
| `trajectory_synthesis` | Coarse approach + sensorimotor closed loop | **Forbidden** (`FirewallViolationError`) | Multi-seed rollout recording strictly non-privileged streams | **Yes** (`continuous_from_t0=True`, `privileged_channels_recorded=False`) |

`TrajectoryEpisodeManifest` (`trajectory-forge-episode/v0.1`) enforces at construction time that `continuous_from_t0` is `True`, `privileged_channels_recorded` is `False`, and no channel name uses reserved prefixes (`privileged_`, `ground_truth_`, `oracle_`).

---

## 2. Why Full Python Scripts Instead of Fixed High-Level Primitives

Hardcoded primitives (`grasp(obj)`, `pull_drawer()`) bake in **two-finger parallel gripper** and **open-loop position trajectory** assumptions. Authoring complete `def solve(ctx):` Python scripts over a thin 4-module SDK prevents interface lock-in:

1. **`ctx.world` (`WorldAPI`)**: Queries `instruction`, `object_pose`, authoritative `named_frame(obj, frame)`, and `joint_band(obj, joint, state)`.
2. **`ctx.kinematics` (`KinematicsAPI`)**: Chain-agnostic `solve_ik(chain, target_pose)` plus exact SE(3) constrained mechanism helpers:
   - `plan_revolute_arc(...)`: Keeps constant lever radius around `hinge_pivot_world` while rotating end-effector orientation along `hinge_axis_world` (handles coffee brew levers, fixture hoods, scanner lids without manual quaternion math).
   - `plan_linear_sweep(...)`: Generates straight-line prismatic trajectories along `direction_world` (drip trays, printer paper cassettes).
3. **`ctx.end_effector` (`EndEffectorAPI`)**: Commands arbitrary `ActuatorGroupSpec` groups via `set_posture(group, posture_name, control_mode)` or `command_targets(group, targets, control_mode)`.
4. **`ctx.sensors` (`SensorsAPI`)**: Provides per-step `proprio()`, `wrist_wrench(chain)`, and `tactile(group)` (`TactileReading` with `normal_force_n`, `shear_force_n`, `slip_detected`, `in_contact`) so scripts can write reactive `while` / `if` guarded moves.

---

## 3. Cross-Embodiment, Dexterous Hand & Tactile Forward Compatibility

While `v0.1` targets `lift2_r5a_bimanual` (16-DOF dual 6-DOF arms + dual 2-finger parallel grippers), every contract is generalized via `EmbodimentCard` (`configs/embodiments/`):

- **No scalar gripper width**: Parallel grippers and 16-DOF+ five-finger dexterous hands both implement `ActuatorGroupSpec` (`kind`: `parallel_gripper | dexterous_hand | suction_end_effector | custom`, `control_modes`: `position | velocity | effort | impedance`, and `named_postures` such as `pre_pinch`, `tripod_dial`, `index_poke`).
- **Extensible sensor streams**: `SensorStreamSpec` declares `proprio`, `wrench`, `tactile_summary`, `tactile_array`, `rgb`, and `depth` shapes and rates (`configs/embodiments/template_bimanual_dexterous_tactile.yaml` validates through the exact same parser as `lift2`).
- **Causal alignment for tactile data**: When synthesizing tactile trajectories, controllers combine privileged coarse approach with sensor-guarded contact loops (`while not tactile.in_contact` / `if tactile.slip_detected`) under millimeter pose perturbations so recorded actions exhibit genuine sensorimotor causality.

---

## 4. `v0.1` Closed-Loop Solver, Diagnostics & `r3` Live Verification

- **`PrivilegedSolverHarness` (`cli solve`)**: Orchestrates iterative controller synthesis/repair (`privileged_search`, snapshot checkpointing permitted) -> zero-snapshot continuous `t=0` gate (`full_horizon_gate`, `FirewallViolationError` on any snapshot save/restore/resume) -> sensorimotor export (`trajectory_synthesis`, writing `episode_trajectory.npz` with `privileged_channels_recorded=False`, `proprio`, `action`, `base_target`, `torso_lift`, `timestamps_s`, `rgb_head`).
- **Executor Backends**: Supports `dry-run`, `isaac-genmanip`, `ssh-isaac-genmanip` (`SSHIsaacGenManipExecutor` running real Isaac Sim 4.1 + `GenManip` + CUDA CuRobo remotely over SSH), and `http-worker`.
- **Automated `SolvabilityDiagnostics`**: Distinguishes physical USD/asset blockers from controller errors (`UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT` when `drive:angular:physics:damping > 2.0 N*m*s/deg` causes >114.6 N*m*s/rad PhysX torque; `UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE` when an ungraspable object's horizontal footprint exceeds `0.088 m`).

Tracked in `configs/benchmarks/r3_out_of_domain_suite.yaml` (detailed live run metrics in [2026-11-13-r3-privileged-mode-live-verification.md](../records/2026-11-13-r3-privileged-mode-live-verification.md)):

| Task ID | Domain | Hero Object & Key DOFs | Target Stages | Verified Live Status (Isaac Sim 4.1) |
|---|---|---|---|---|
| `electronics_qa_ingun_probe_lock` | `electronics_qa_station` | `ingun_fixture` (`pressure_hood`, `clamp_lever`, `mode_toggle`, `test_start_button`) | 5 | **Solved `1.00` (`5/5` hold)**, `1757` steps, `snapshots_used=0`, `firewall_verified=True` |
| `specialty_coffee_bar_creatista_prep` | `specialty_coffee_bar` | `creatista_machine` (`drip_tray`, `brew_lever`, `select_dial`, `dial_press`) | 6 | Stages `0–2` pass (`0.50`); Stage `3` blocked (`UPSTREAM_USD_OVERDAMPED_REVOLUTE_JOINT`) |
| `modern_office_brother_calibrate` | `modern_office_print_hub` | `brother_printer` (`paper_cassette`, `scanner_lid`, `button_start`) | 5 | Stages `1` & `4` pass physically; Stage `0` blocked (`UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE`) |

---

## 5. Dual-Mode Upstream Scene Remediation Contract & `r4` Live Verification

When downstream simulation discovers that an upstream `scenario-forge` package is physically blocked, `trajectory-forge` governs allowable upstream scene changes via `RemediationPolicySpec` (`src/trajectory_forge/core/contracts.py`):

| Policy Mode | `allow_pose_adjustment` | Permitted Upstream Remediations | Behavior on Structural Blockers (`UPSTREAM_OBJECT_EXCEEDS_GRIPPER_APERTURE`) |
|---|---|---|---|
| `physics_tuning` (default) | `False` (strict) | Joint drive damping/stiffness/effort (`joint_drive`), missing collision proxies (`collision_proxy`), contact friction/offset (`friction_or_offset`). Zero visual/mesh edits, zero added parts, zero initial pose changes. | If the object has no pinchable edge `<= 0.080 m` without moving or adding parts, `apply_remediation_policy()` emits `BLOCKED_BY_REMEDIATION_POLICY` (`policy_blocked=True`, preserving `root_failure_code`) and `PrivilegedSolverHarness` halts cleanly after Attempt 1 (`stop_on_policy_block=True`). |
| `physics_tuning` | `True` (sub-switch) | All Mode 2 physics adjustments **plus** non-destructive initial pose/overhang adjustments (`pose_adjustment`, e.g., tabletop front-edge overhang when `min(dims_xyz) <= 0.080 m`) with zero visual mesh edits or added parts. | Allowed when `min(dims_xyz) <= 0.080 m` (annotated `remediation_category="pose_adjustment"`, `required_remediation_mode="physics_tuning"`); blocked only if all three dimensions exceed the gripper span. |
| `scene_codesign` | `True` / `False` | All Mode 2 adjustments **plus** structural co-design (`affordance_part`, `fixture_or_layout`), such as adding a tabletop grasp handle / carry-bar with `named_frame("grasp")` and positive-scale `purpose="guide"` proxy collider. | Allowed (`remediation_category="affordance_part"`, `required_remediation_mode="scene_codesign"`), consumed directly by controllers via `ctx.world.named_frame(obj_id, "grasp")`. |

Verified across `r4` packages (`configs/benchmarks/r4_out_of_domain_suite.yaml`) and `GEN-010` in Isaac Sim `4.1` (full metrics in [2026-11-14-out-of-domain-r4-full-suite-verification.md](../records/2026-11-14-out-of-domain-r4-full-suite-verification.md)):

| Task ID | Remediation Policy Tested | Stages Passed | Final Score | Steps | Snapshots Used | Firewall Verified | Outcome / Failure Code |
|---|---|---|---|---|---|---|---|
| `electronics_qa_ingun_probe_lock` (`r3`/`r4`) | `physics_tuning` (strict) | `5/5` (`hold=True`) | `1.0` | `1757` | `0` | `True` | `succeeded=True` (`r3` baseline) |
| `real2sim_office_desk_riser_organize` (`GEN-010`) | `physics_tuning` (`collision_proxy`) | `5/5` | `1.0` | `1448` | `0` | `True` | `succeeded=True` ([record](../records/2026-11-14-real2sim-workstation-live-verification.md)) |
| `specialty_coffee_bar_creatista_prep` (`r4`) | `physics_tuning` (strict) | `6/6` (`hold=True`) | `1.0` | `1775` | `0` | `True` | `succeeded=True` |
| `modern_office_brother_calibrate` (`r4` `mode2_strict_ablation`) | `physics_tuning` (`allow_pose_adjustment=False`) | Stage `1` only | `0.0` | `1280` | `4` (search) | `False` | `BLOCKED_BY_REMEDIATION_POLICY` (`policy_blocked=True`) |
| `modern_office_brother_calibrate` (`r4` `mode2_pose_ablation`) | `physics_tuning` (`allow_pose_adjustment=True`) | `5/5` | `1.0` | `1449` | `0` | `True` | `succeeded=True` |
| `modern_office_brother_calibrate` (`r4` `scene_gate`) | `scene_codesign` | `5/5` | `1.0` | `1566` | `0` | `True` | `succeeded=True` |
