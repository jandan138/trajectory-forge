# Agent Guide

Trajectory Forge is the downstream **Privileged Manipulation Solver & Demonstration Trajectory Factory**. Keep the repository narrow and strictly decoupled from scene authoring and black-box policy benchmarking.

## Hard Boundaries

- **Identity neutrality**: Do not use `lab` in repository/package naming or public schema identifiers, and do not hardcode external model names (e.g., `astra`) in core package contracts.
- **Pure-layer isolation**: Pure modules under `src/trajectory_forge/core`, `intake`, `embodiment`, `sdk`, `worker`, and `trajectory` must never import heavy simulator, planner, or ML SDKs (`pxr`, `omni`, `isaacsim`, `genmanip`, `curobo`, `mujoco`, `torch`). Simulator bindings belong strictly under `src/trajectory_forge/adapters/`.
- **Upstream/downstream ownership**:
  - Do not add USD/scene compilation or asset conversion here (`scenario-forge` and `ConvertAsset` own those).
  - Do not add black-box policy evaluation leaderboards or benchmark reporting here (`embodied-eval-os` owns those).
- **Privileged-to-Sensorimotor Firewall**:
  - `privileged_search` allows full world state reads and stage snapshot save/restore for rapid local failure repair.
  - `full_horizon_gate` and `trajectory_synthesis` strictly forbid snapshot resumption (`resume_from_snapshot_id` must be `None` and `checkpoint.restore_snapshot()` must raise `FirewallViolationError`).
  - Exported demonstration manifests (`trajectory-forge-episode/v0.1`) must verify `continuous_from_t0=True` and `privileged_channels_recorded=False`.

## Cross-Embodiment & Future Hardware Policy

- Never hardcode single scalar gripper width assumptions in core contracts. Represent end effectors via `ActuatorGroupSpec` (supporting `parallel_gripper` as well as multi-DOF `dexterous_hand` with `position | velocity | effort | impedance` control modes and named synergy postures).
- Represent sensors via extensible `SensorStreamSpec` channels (`proprio`, `wrench`, `tactile_summary`, `tactile_array`, `rgb`, `depth`) so tactile guarded-move control (`while` loops in Python scripts) upgrades without breaking `v0.1` contracts.

## Expected Workflow

1. Write a failing test for contract, SDK, intake, or worker behavior.
2. Implement the minimal typed pure-Python code that passes.
3. Run `make ci-check` (and `make check` when `/cpfs` `scenario-forge` `r3` packages are mounted).
4. Keep documentation synchronized with any contract or worker protocol change.
