# 2026-11-20 — M3 v0.3 Four-Engine Zero-Cheat Rollouts & Offline Showcase (`trajectory-forge`)

- **Date**: 2026-11-20
- **Status**: COMPLETE & VERIFIED (All 10 RTX Videos Rendered & Embedded in Offline HTML)
- **Repository**: `I:\cpfs\shared\simulation\zhuzihou\dev\trajectory-forge`
- **Key Modules / Scripts / Tests**:
  - `src/trajectory_forge/skills/deformable_primitives_v03.py` (`CompliantPinchPlaceController`, `ClothEdgeFoldAndPeelController`, `FlexibleRodRouteClipController`, `HollowShellSqueezeFitController`, `auto_tune_skill_params`)
  - `src/trajectory_forge/adapters/deformable_workers_v03/` (`common_v03.py`, `runner_v03.py`, `mujoco4_v03_worker.py`, `newton18_v03_worker.py`, `genesis_v03_worker.py`, `isaac6_v03_worker.py`)
  - `scripts/run_m3_deformable_v03_trajectories.py`, `scripts/render_deformable_v03_showcase_media.py`, `scripts/build_deformable_v03_showcase_html.py`
  - `tests/test_deformable_v03_m3.py`, `tests/test_deformable_v03_showcase_html.py`, `tests/test_deformable_v02_m3.py`, `tests/test_deformable_v02_showcase_html.py` (**`10 passed`**, latest run `1.73s`)

---

## 1. Zero-Cheat Policy & Cross-Engine Calibration

- **Zero-Cheat Redline**: Zero `<weld>`, `<connect>`, `PhysxPhysicsAttachment`, `OmniPhysicsVtxXformAttachment`, `cloth_spine`, or Python vertex clamping across all 4 workers and 4 skill primitives (`test_v03_zero_cheat_source_and_skill_firewall_audit` PASSED).
- **Newton 1.8.0 (`newton18_v03_worker.py`)**: Uncapped `e_eff` in `_pad_particle_normal_force` for volumetric `soft_solid`/`rod_cable` assets (fixing `95 kPa` ribbon cable tactile latching) and calibrated `y_vbd = 4.2 * young` for `rod_cable`.
- **Isaac Sim 6.0.1 (`isaac6_v03_worker.py`)**: Calibrated `office_dust_cover_peel_retrieve` with scene `MaxPositionIterationCount=16`, default cloth solver iterations, `def_mu_s=0.65`, and `(0.038, 0.012, 0.036)` pads so stage-1 peel clears to `-0.193 m` (`finX <= -0.18 m`) before stage-2 `0.45 kg` `TonerBox` pick & caddy placement.

---

## 2. Observed 4-Engine M3 Rollout Matrix (`8 v0.3 Tasks`)

- **MuJoCo 4 (`m3_v03_mujoco4_summary.json`, 88 episodes)**: `8/8` `gentle_s0` `intact=True`, `8/8` counterfactual `intact=False`, `8/8` `zero_friction_s0` `intact=False`, `40/40` deterministic `replay_r0..r4` `intact=True`, `24/24` perturbed `gentle_s1..s3` `intact=True`.
- **Cross-Engine Summary (`m3_v03_cross_engine_summary.json`, 72 `newton18`/`genesis`/`isaac6` episodes)**:
  - **Zero-friction falsification (`zero_friction_s0`)**: `32/32` (`intact=False` across all 8 tasks x 4 engines).
  - **Counterfactual falsification (`crush_s0`/`tear_s0`)**: `32/32` (`intact=False` across all 8 tasks x 4 engines).
  - **All 8/8 `v0.3` tasks pass on `>= 2/4` engines** (`7` tasks at `2/4`, `1` task at `3/4`):

| Task ID | Passing (`N/4`) | `mujoco4` | `newton18` | `genesis` | `isaac6` |
| --- | --- | --- | --- | --- | --- |
| `coffee_croissant_tray_to_plate` | `2/4` | `PASS` (`dmg=0.000, fn=3.25N, dist=0.0086`) | `PASS` (`dmg=0.061, fn=2.57N, dist=0.0030`) | `FAIL` (`dmg=0.262, fn=2.35N, dist=0.0037`) | `FAIL` (`dmg=0.993, fn=5.08N, dist=0.0120`) |
| `coffee_strawberry_tart_topping` | `2/4` | `PASS` (`dmg=0.000, fn=1.73N, dist=0.0030`) | `PASS` (`dmg=0.188, fn=0.71N, dist=0.0002`) | `FAIL` (`dmg=0.545, fn=0.68N, dist=0.0190`) | `FAIL` (`dmg=1.000, fn=2.00N, dist=0.0041`) |
| `office_waffle_towel_corner_fold` | `2/4` | `PASS` (`dmg=0.000, fn=0.48N, dist=0.0255`) | `FAIL` (`dmg=0.000, fn=3.82N, dist=0.0999`) | `FAIL` (`dmg=0.000, fn=2.45N, dist=0.1046`) | `PASS` (`dmg=0.000, fn=3.00N, dist=0.0000`) |
| `office_dust_cover_peel_retrieve` | `2/4` | `PASS` (`dmg=0.000, fn=17.17N, dist=0.0100`) | `FAIL` (`dmg=0.000, fn=2.81N, dist=0.4938`) | `FAIL` (`dmg=1.000, fn=1.66N, dist=0.0005`) | `PASS` (`dmg=0.000, fn=3.54N, dist=0.0071`) |
| `smt_ribbon_cable_clip_route` | `2/4` | `PASS` (`dmg=0.000, fn=3.74N, dist=0.0043`) | `PASS` (`dmg=0.000, fn=5.32N, dist=0.0009`) | `FAIL` (`dmg=0.113, fn=2.11N, dist=0.3066`) | `FAIL` (`dmg=1.000, fn=2.61N, dist=0.0083`) |
| `smt_esd_silicone_boot_press_fit` | `2/4` | `PASS` (`dmg=0.000, fn=3.99N, dist=0.0019`) | `FAIL` (`dmg=0.000, fn=4.12N, dist=0.3128`) | `FAIL` (`dmg=0.511, fn=0.87N, dist=0.3144`) | `PASS` (`dmg=0.000, fn=5.29N, dist=0.0009`) |
| `lab_silicone_tube_manifold_insert` | `3/4` | `PASS` (`dmg=0.000, fn=3.77N, dist=0.0043`) | `PASS` (`dmg=0.075, fn=1.57N, dist=0.0025`) | `PASS` (`dmg=0.169, fn=1.71N, dist=0.0075`) | `FAIL` (`dmg=1.000, fn=1.97N, dist=0.0061`) |
| `lab_rubber_bulb_squeeze_seat` | `2/4` | `PASS` (`dmg=0.000, fn=12.15N, dist=0.0024`) | `FAIL` (`dmg=0.000, fn=5.13N, dist=0.1846`) | `FAIL` (`dmg=1.000, fn=0.00N, dist=0.1257`) | `PASS` (`dmg=0.000, fn=19.94N, dist=0.0241`) |

---

## 3. RTX Showcase Media Rendering & 10-Video Self-Contained Offline HTML

- `render_deformable_v03_showcase_media.py --confirm-rtx-render` (8 sequential isolated Isaac Sim 6.0.1 `RayTracedLighting` sessions under `IDLE_PRIORITY_CLASS`, `1280x720` HUD overlay + `ffmpeg libx264` @ `30 fps`):
  - Manifest: `external_artifacts/deformable_v03_media/media_manifest.json`
  - Visual QA `8x3` contact sheet: `external_artifacts/deformable_v03_media/showcase_v03_contact_sheet.png` (`3,523,517` bytes)

| Task ID | Bytes | Frames | `t0 / mid / final` ROI Mean (`std`) | SHA-256 |
| --- | --- | --- | --- | --- |
| `coffee_croissant_tray_to_plate` | `706,440` | `124` | `110.9 (71.3) / 116.3 (73.8) / 114.0 (72.5)` | `5a2e41b20b5baa8ac594efc0652e3968c2afdba37984f61f26877059d1d82994` |
| `coffee_strawberry_tart_topping` | `702,011` | `124` | `110.9 (71.4) / 116.5 (73.9) / 115.9 (72.6)` | `8f2cf4182a5a11927ef24cd1ed0f2cf2a20a3cc3dd2053fafdb5ac2e183cd2ef` |
| `office_waffle_towel_corner_fold` | `629,192` | `124` | `122.6 (71.5) / 126.6 (72.3) / 126.3 (71.9)` | `d47e3c52e30969ea0bc4b0250b55a91a19697287112afb5b81ea49b3ff6f1a6f` |
| `office_dust_cover_peel_retrieve` | `814,040` | `156` | `121.9 (72.3) / 126.5 (73.6) / 126.1 (71.3)` | `cb785f11b328eb206361e4d0ebf5258eea937861d59b0e77008eea5d880d131a` |
| `smt_ribbon_cable_clip_route` | `629,561` | `124` | `119.1 (72.8) / 124.0 (73.1) / 123.4 (72.2)` | `218af4cac677d646406dcedf1e9cf499ba54998238b6dc6d33c30f6d323575dd` |
| `smt_esd_silicone_boot_press_fit` | `617,496` | `122` | `119.0 (72.4) / 124.1 (72.9) / 121.9 (72.2)` | `fc54099031811d5d77bb44b638b32da9e588de39f1d5439a814594dc78a749d1` |
| `lab_silicone_tube_manifold_insert` | `803,772` | `124` | `159.3 (45.5) / 164.2 (44.5) / 162.1 (44.5)` | `f163cd074463cd0750d3302bb4ce4ff00daaaeaf47696956b3b667064079cde2` |
| `lab_rubber_bulb_squeeze_seat` | `848,542` | `136` | `159.8 (45.5) / 163.4 (44.6) / 162.6 (44.7)` | `6d1942f18f2fc647f5303ff2f82f7b095aff642e52dd7bc333b43fb1dda316c1` |

- `build_deformable_v03_showcase_html.py` -> rebuilt offline 10-task report `external_artifacts/reports/deformable_v03_10task_showcase_20261120.html` (`10,151,587` bytes, **`10` embedded `data:video/mp4;base64,` players**, `0` pending placeholders, `0` external `http://`/`https://` URLs).

