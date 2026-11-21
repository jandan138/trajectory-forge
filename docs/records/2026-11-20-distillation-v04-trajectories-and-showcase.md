# Engineering Record — Distillation Apparatus v0.4 18-DOF Bimanual Trajectories, RTX Glassware Showcase & Offline HTML (M3 & M5)

- **Date**: 2026-11-20
- **Repository**: `trajectory-forge`
- **Scope**: Milestones 3 & 5 (`distillation_v04` — 《蒸馏装置的装配与拆除》)

## Summary
Implemented zero-cheat 18-DOF dual-arm Franka Panda bimanual skill primitives, 4-engine physics workers (`mujoco4`, `newton18`, `genesis`, `isaac6`), `140` multi-engine rollout episodes, Isaac Sim 6.0.1 RTX ray-traced borosilicate glassware showcase videos, and a 100%-offline self-contained HTML report:

1. **18-DOF Bimanual Skill Primitives & Zero-Cheat Frictional Physics (`src/trajectory_forge/skills/distillation_bimanual_v04.py`, `src/trajectory_forge/adapters/distillation_workers_v04/`)**:
   - Strictly zero grasping/attachment cheats (`<weld>`, `<connect>`, `PhysxPhysicsAttachment`, `OmniPhysicsVtxXformAttachment`, `cloth_spine`, `set_particles_pos` audited absent by `test_v04_zero_cheat_source_and_18dof_controller_audit`).
   - All `140` episodes generated across `mujoco4` (`77` episodes: `7` gentle + `7` counterfactual + `7` zero-friction + `35` cold-start replays + `21` perturbed seeds), `newton18` (`21` episodes), `genesis` (`21` episodes), and `isaac6` (`21` episodes) under strict single-GPU serialization.
   - Multi-engine certification achieved on all `7/7` SOP tasks (`28/28` counterfactuals falsified, `28/28` zero-friction falsified):
     - `distill_01_burner_ring_flask_mount`: `mujoco4` + `newton18` (`2/4 Silver`).
     - `distill_02_head_taper_coaxial_mate`: `mujoco4` + `newton18` (`2/4 Silver`, `tilt <= 2.10° <= 2.50°`, `F <= 1.28 N`).
     - `distill_03_thermometer_septum_align`: `mujoco4` + `newton18` (`2/4 Silver`, `|dz| = 0.32..1.21 mm <= 2.00 mm`).
     - `distill_04_bimanual_hose_barb_couple`: `mujoco4` + `newton18` (`2/4 Silver`, left-arm root brace `M_nozzle <= 0.17 N·m <= 0.45 N·m`, right-arm `"下进上出"` dual-hose coupling).
     - `distill_05_condenser_adapter_receiver_train`: `mujoco4` + `isaac6` (`2/4 Silver`).
     - `distill_06_reverse_order_disassembly`: `mujoco4` + `newton18` + `isaac6` (`3/4 Silver`, `"先右后左、先上后下"`).
     - `distill_07_full_assembly_disassembly_sop`: `mujoco4` + `isaac6` (`2/4 Silver`, `"从下到上、从左到右"` -> `55%` dwell -> `"先右后左、先上后下"`).

2. **Isaac Sim 6.0.1 RTX Borosilicate Glass Showcase & 100%-Offline HTML (`external_artifacts/distillation_v04_media/`, `external_artifacts/reports/distillation_v04_sop_showcase_20261120.html`)**:
   - Rendered 7 H.264 MP4 showcase videos with ray-traced translucency/refraction (`maxRefractionBounces=6`), barycentric-skinned 3D deformable meshes with 2K PBR maps, dual 18-DOF Franka arms, and Chinese HUD telemetry, plus `showcase_v04_contact_sheet.png`.
   - Built 100%-offline self-contained HTML showcase embedding all 7 base64 MP4 videos, PBR swatches, 4-engine certification matrix, and 5 EOS self-healing closed-loop cards with zero external `http://` or `https://` URLs.

## Verification
- `pytest tests/test_distillation_v04_m3.py tests/test_distillation_v04_showcase_html.py tests/test_deformable_v03_m3.py -v` -> `6 passed`.

