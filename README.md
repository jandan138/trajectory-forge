# Trajectory Forge

> Privileged manipulation solver and demonstration trajectory factory for embodied scenario packages.

`trajectory-forge` is the downstream sibling of `scenario-forge`. It consumes self-contained collected scenario packages (`scenario-forge-genmanip-collected-package/v0.1`), runs **Embodied First Exam Privileged Mode** searches where agents write closed-loop Python controllers, verifies zero-snapshot continuous `t=0` execution, and synthesizes sensorimotor demonstration trajectories across robot embodiments.

## Ecosystem Positioning

```text
ConvertAsset
  owns: normalized USD/MDL/mesh/collider asset packages

scenario-forge
  owns: scene compilation, layout constraints, ScenarioSpec, MetricsManager goals,
        GenManip collected packages + scenario_forge_runtime_contract

trajectory-forge (this repo)
  owns: privileged manipulation solving, Embodied First Exam Privileged Mode gate,
        cross-embodiment Python controller SDK, stage snapshot search,
        full-horizon continuous replay verification, demonstration synthesis

embodied-eval-os
  owns: black-box model/policy evaluation, trace capture, leaderboards
```

## Key Architectural Principles

1. **Pure-Layer Isolation**: `core`, `intake`, `embodiment`, `sdk`, `worker`, and `trajectory` never import simulator, physics, or ML SDKs (`pxr`, `omni`, `isaacsim`, `genmanip`, `curobo`, `mujoco`, `torch`). Simulator execution is isolated under `src/trajectory_forge/adapters/`.
2. **Agent Python Controller SDK**: Instead of rigid parallel-gripper primitives, agents author full Python `def solve(ctx):` scripts backed by `world`, `kinematics`, `end_effector`, `sensors`, and `checkpoint`.
3. **Cross-Embodiment & Tactile Reservation**: `EmbodimentCard` (`trajectory-forge-embodiment-card/v0.1`) abstracts kinematic chains, multi-DOF actuator groups (`parallel_gripper`, `dexterous_hand` with `position | velocity | effort | impedance`), and extensible sensor channels (`proprio`, `wrench`, `tactile_summary`, `tactile_array`, `rgb`, `depth`).
4. **Privileged-to-Sensorimotor Firewall**:
   - `privileged_search` permits stage snapshot save/restore (`ctx.checkpoint`).
   - `full_horizon_gate` and `trajectory_synthesis` strictly forbid snapshot resumption (`FirewallViolationError`) and enforce `continuous_from_t0=True`, `privileged_channels_recorded=False` on exported manifests.
5. **Cloud Brain / Local 4090 Worker Split**: Cloud agent orchestration, brief parsing, AST safety checks, and `<50 ms` `DryRunSimBackend` preflight run cleanly in Google AI Ultra workspaces; heavy Isaac Sim 4.1 RTX/PhysX rollouts execute on a remote GPU worker via `/v1/rollout`.

## Quick Start

Validate embodiment cards and run a controller preflight locally:

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=src python -m trajectory_forge.cli embodiment check configs/embodiments/lift2_r5a_bimanual.yaml
PYTHONPATH=src python -m trajectory_forge.cli embodiment check configs/embodiments/template_bimanual_dexterous_tactile.yaml
PYTHONPATH=src python -m trajectory_forge.cli controller preflight \
  --controller examples/sample_privileged_controller.py \
  --embodiment configs/embodiments/lift2_r5a_bimanual.yaml \
  --mode full_horizon_gate
```

Inspect an upstream `scenario-forge` `r3` package:

```bash
PYTHONPATH=src python -m trajectory_forge.cli intake inspect \
  --package /cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/out_of_domain_factory_r3_20261113/packages/specialty_coffee_bar_creatista_prep \
  --embodiment configs/embodiments/lift2_r5a_bimanual.yaml
```

Start the HTTP worker server:

```bash
TF_WORKER_TOKEN=change-me PYTHONPATH=src python -m trajectory_forge.cli worker serve --host 127.0.0.1 --port 8765
```

## Developer Checks

```bash
make ci-check   # pure unit, boundary, RPC, ruff lint, and CLI smoke tests
make check      # additionally runs @pytest.mark.local_artifacts tests against r3 packages
```

See also:
- [v0.1 Architecture & Privileged Mode Contract](docs/design/architecture-v0.1.md)
- [Google AI Ultra Cloud Brain to RTX 4090 Sim Worker Workflow](docs/operations/ultra-cloud-to-4090-worker-workflow.md)
