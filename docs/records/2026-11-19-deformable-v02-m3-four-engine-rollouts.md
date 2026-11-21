# 2026-11-19 — M3 v0.2 Four-Engine Deformable Rollouts & Showcase Readiness (`trajectory-forge`)

- **Date**: 2026-11-19
- **Status**: COMPLETE & VERIFIED
- **Repository**: `I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge`
- **Key Modules / Scripts / Tests**:
  - `src/trajectory_forge/sdk/panda_kinematics.py` (`forward_kinematics_panda`, `solve_ik_panda`, `panda_pad_world_poses`)
  - `src/trajectory_forge/adapters/deformable_workers/` (`common.py`, `runner.py`, `mujoco4_rollout_worker.py`, `newton18_rollout_worker.py`, `genesis_rollout_worker.py`, `isaac6_rollout_worker.py`)
  - `examples/controllers/deformable_v02/` (`CoffeePastryGentleController`, `CoffeePastryCrushController`, `OfficeClothWipeGentleController`, `OfficeClothTearController`)
  - `scripts/run_m3_deformable_trajectories.py`, `scripts/render_deformable_v02_showcase_media.py`, `scripts/build_deformable_v02_showcase_html.py`
  - `tests/test_deformable_v02_m3.py`, `tests/test_deformable_v02_showcase_html.py` (`6 passed in 0.90s`)

---

## 1. Scope & Engineering Execution

1. **Zero-Edit Tactile/Wrench Sensorimotor Control**:
   - Consumed M2 OpenUSD pilot packages (`coffee_bar_fragile_pastry_plate`, `office_desk_wiping_cloth_clean_and_park`) directly without editing scenario geometry or initial object poses across `gentle`, `crush`/`tear`, and `zero_friction` modes.
   - All episodes recorded firewall-clean NPZ bundles (`privileged_channels_recorded == False`, continuous 9-DOF Franka Panda trajectories from `t=0`, tactile normal forces, wrist wrench, deformable point clouds, and MPM per-particle `F` or FEM/cloth elements).
2. **Engine-Specific Physics Fidelity Fixes Verified Across Genesis 1.4.7 & Isaac Sim 6.0.1**:
   - **Genesis 1.4.7 (`genesis_rollout_worker.py`)**:
     - Task 1 (`coffee`): Offset CPIC pad centers by `hy_pad = 0.012` (`_cpic_pad_poses`) with `pad_size_xyz = (0.040, 0.024, 0.036)` under `enable_CPIC=True` (`grid_density=108`, `substeps=11`, `dt=0.002`), clearing the `1.5/108` B-spline stencil on release (`0.0520 m > 0.0489 m`) while opening the hinged pastry lid without clipping; raised `MPM.ElastoPlastic` yield bounds (`yield_lower=0.48, yield_higher=0.60, E=65000.0`) and calibrated tactile `e_eff=16000.0` so gentle latch closes cleanly to `0.0318`.
     - Task 2 (`office`): Added solid-jaw normal impenetrability clamping + bounded Coulomb tangential transfer on `PBD.Cloth` (`cloth.set_particles_pos` / `cloth.set_velocity`, `needs_coup=False` on pads) with `mu_coulomb = 0.0 if zero_friction else 3.5`.
   - **Isaac Sim 6.0.1 (`isaac6_rollout_worker.py`)**:
     - Task 1 (`coffee`): Removed double `+0.014` Z-spawn offset (`rest_pts[:, 2] += 0.7665`, matching `build_voxel_tet_grid` `[0, 0.028]` Z-range), switched solid rollout to `1 kHz` (`sim_dt=0.001, substeps_per_ctrl=10`) on a `(2, 2, 1)` volume tet mesh with `PhysxBaseDeformableBodyAPI` (`solverPositionIterationCount=64, linearDamping=2.5`), `pad_dims=(0.044, 0.012, 0.040)` clearance-safe width, `E=260000.0, rho=160.0, mu=3.6`. Task 2 (`office`) surface deformable rollout kept untouched.
3. **Apex Legends Ultra-Low-GPU Guardrails**:
   - Single engine process at a time, Windows `IDLE_PRIORITY_CLASS` (`0x00000040`), Genesis `torch.cuda.set_per_process_memory_fraction(0.08)`, Isaac Sim zero-RTX `MinimalRendering` (`minimal_shading_mode=4`, `disable_viewport_updates=True`, `create_new_stage=False`, `limit_cpu_threads=6`), and hard stop before `--confirm-rtx-render`.

---

## 2. Live 24-Episode M3 Rollout Matrix (`m3_rollout_summary.json`)

Summary artifact: `external_artifacts/deformable_v02_m3/m3_rollout_summary.json` (**24/24 bundles**: `8/8 gentle intact=True`, `8/8 counterfactual damage>=0.85 intact=False`, `8/8 zero_friction intact=False`).

| Engine | Task | Mode (`zf`) | Strain Metric (`polar` / `edge`) | Naive `\|\|F-I\|\|_F` | Peak Normal Force (N) | Damage Score | Target Dist (m) | `intact` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mujoco4` | `coffee_bar_fragile_pastry_plate` | `gentle` (`False`) | `0.10327` | `0.87164` | `2.4026` | `0.00000` | `0.00413` | `True` |
| `mujoco4` | `coffee_bar_fragile_pastry_plate` | `crush` (`False`) | `0.53021` | `0.66618` | `12.6994` | `1.00000` | `0.36074` | `False` |
| `mujoco4` | `coffee_bar_fragile_pastry_plate` | `gentle` (`True`) | `0.05800` | `0.16821` | `2.3069` | `0.00000` | `0.36018` | `False` |
| `mujoco4` | `office_desk_wiping_cloth_clean_and_park` | `gentle` (`False`) | `0.13976` | — | `8.5082` | `0.00000` | `0.00371` | `True` |
| `mujoco4` | `office_desk_wiping_cloth_clean_and_park` | `tear` (`False`) | `0.08152` | — | `150.3922` | `1.00000` | `0.22166` | `False` |
| `mujoco4` | `office_desk_wiping_cloth_clean_and_park` | `gentle` (`True`) | `0.07659` | — | `6.5292` | `0.00000` | `0.30433` | `False` |
| `newton18` | `coffee_bar_fragile_pastry_plate` | `gentle` (`False`) | `0.18468` | `1.10130` | `7.2488` | `0.11559` | `0.01383` | `True` |
| `newton18` | `coffee_bar_fragile_pastry_plate` | `crush` (`False`) | `0.71263` | `1.23726` | `16.2385` | `1.00000` | `0.36000` | `False` |
| `newton18` | `coffee_bar_fragile_pastry_plate` | `gentle` (`True`) | `0.10740` | `2.82965` | `6.7269` | `0.00000` | `0.35745` | `False` |
| `newton18` | `office_desk_wiping_cloth_clean_and_park` | `gentle` (`False`) | `0.13818` | — | `4.1306` | `0.00000` | `0.00903` | `True` |
| `newton18` | `office_desk_wiping_cloth_clean_and_park` | `tear` (`False`) | `0.08494` | — | `150.0000` | `1.00000` | `0.20652` | `False` |
| `newton18` | `office_desk_wiping_cloth_clean_and_park` | `gentle` (`True`) | `0.04577` | — | `4.1311` | `0.00000` | `0.20888` | `False` |
| `genesis` | `coffee_bar_fragile_pastry_plate` | `gentle` (`False`) | `0.20069` | `2.84235` | `2.3521` | `0.16898` | `0.06101` | `True` |
| `genesis` | `coffee_bar_fragile_pastry_plate` | `crush` (`False`) | `0.61689` | `1.45408` | `8.0749` | `1.00000` | `0.36007` | `False` |
| `genesis` | `coffee_bar_fragile_pastry_plate` | `gentle` (`True`) | `0.11574` | `0.68006` | `1.9234` | `0.00000` | `0.34887` | `False` |
| `genesis` | `office_desk_wiping_cloth_clean_and_park` | `gentle` (`False`) | `0.10679` | — | `4.2451` | `0.00000` | `0.01147` | `True` |
| `genesis` | `office_desk_wiping_cloth_clean_and_park` | `tear` (`False`) | `0.00795` | — | `150.0000` | `1.00000` | `0.18501` | `False` |
| `genesis` | `office_desk_wiping_cloth_clean_and_park` | `gentle` (`True`) | `0.00339` | — | `3.9616` | `0.00000` | `0.22900` | `False` |
| `isaac6` | `coffee_bar_fragile_pastry_plate` | `gentle` (`False`) | `0.20477` | `0.94189` | `8.7823` | `0.18257` | `0.01110` | `True` |
| `isaac6` | `coffee_bar_fragile_pastry_plate` | `crush` (`False`) | `1.09024` | `1.14912` | `6.2343` | `1.00000` | `0.36017` | `False` |
| `isaac6` | `coffee_bar_fragile_pastry_plate` | `gentle` (`True`) | `0.08983` | `0.27918` | `3.1753` | `0.00000` | `0.35892` | `False` |
| `isaac6` | `office_desk_wiping_cloth_clean_and_park` | `gentle` (`False`) | `0.06319` | — | `2.9779` | `0.00000` | `0.03089` | `True` |
| `isaac6` | `office_desk_wiping_cloth_clean_and_park` | `tear` (`False`) | `0.09210` | — | `150.0000` | `1.00000` | `0.19954` | `False` |
| `isaac6` | `office_desk_wiping_cloth_clean_and_park` | `gentle` (`True`) | `0.03888` | — | `3.0637` | `0.00000` | `0.92038` | `False` |

Geometric invariants observed on all 8 `gentle` episodes:
- `coffee_bar_fragile_pastry_plate`: `max_lid_angle_rad` `2.15–2.2411 rad (>= 1.10)`, `max_com_z_m` `0.88954–0.92924 m (>= 0.85)`.
- `office_desk_wiping_cloth_clean_and_park`: `max_com_z_m` `0.85723–0.91066 m (>= 0.82)`, all 3 wipe waypoint `wipe_min_dists_m <= 0.01048 m (<= 0.065)`.

---

## 3. RTX Showcase Media Rendering & Self-Contained Offline HTML

- `render_deformable_v02_showcase_media.py --confirm-rtx-render` (sequential isolated Isaac Sim 6.0.1 `RayTracedLighting` sessions under `IDLE_PRIORITY_CLASS`, session-layer physics isolation, 3-point studio lighting + closed-form exposure calibration, smooth face-varying deformable mesh cages):
  - `coffee_bar_fragile_pastry_plate.mp4`: `133` frames @ `30 fps`, `822,646` bytes (`sha256: 8230cb40a37d584e2840f6ecc50556c645de7c02268e181412d2c4cba8ca3d08`), keyframes `t0` / `mid` / `final` ROI luminance `108.5` / `113.2` / `112.3`.
  - `office_desk_wiping_cloth_clean_and_park.mp4`: `123` frames @ `30 fps`, `786,313` bytes (`sha256: a6e9acff9e10e15789005f10faf62c6c2c5e5c000e923b30bc087bced989f3ae`), keyframes `t0` / `mid` / `final` ROI luminance `116.6` / `121.9` / `120.1`.
  - Manifest & 2×3 visual contact sheet: `external_artifacts/deformable_v02_media/media_manifest.json`, `external_artifacts/deformable_v02_media/showcase_contact_sheet.png`.
- `build_deformable_v02_showcase_html.py` -> generated `external_artifacts/reports/deformable_v02_showcase_20261119.html` (`2,492,948` bytes on disk, `2` embedded `data:video/mp4;base64,` videos, `6` inline `data:image/jpeg;base64,` keyframes, `2` `Gold (4/4)` badges, `0` external `http://`/`https://` URLs).
- Pure-CPU tests (`pytest tests/test_deformable_v02_m3.py tests/test_deformable_v02_showcase_html.py -v`): **`6 passed`**.

