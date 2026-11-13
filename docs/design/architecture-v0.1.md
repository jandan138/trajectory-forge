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

## 4. `v0.1` Acceptance Target: `scenario-forge` `r3` Out-of-Domain Suite

Tracked in `configs/benchmarks/r3_out_of_domain_suite.yaml`:

| Task ID | Domain | Hero Object & Key DOFs | Target Stages | Goal |
|---|---|---|---|---|
| `specialty_coffee_bar_creatista_prep` | `specialty_coffee_bar` | `creatista_machine` (`drip_tray`, `brew_lever`, `select_dial`, `dial_press`) | 6 | Continuous `0.00 -> 1.00` on `lift2` |
| `electronics_qa_ingun_probe_lock` | `electronics_qa_station` | `ingun_fixture` (`pressure_hood`, `clamp_lever`, `mode_toggle`) | 5 | Continuous `0.00 -> 1.00` on `lift2` |
| `modern_office_brother_calibrate` | `modern_office_print_hub` | `brother_printer` (`paper_cassette`, `scanner_lid`) | 5 | Continuous `0.00 -> 1.00` on `lift2` |
