# 2026-11-21 — `v0.5` MorphTwin (`arXiv:2611.25963`) Absorption Across Four Repos (`M1`–`M5`)

- **Date**: 2026-11-21
- **Status**: COMPLETE & VERIFIED (`17 passed` across all 4 repos + `v0.2`/`v0.3` regression checks; Pilot 1 & Pilot 2 certified)
- **Design Spec**: [architecture-v0.5-morphtwin-deformable-real2sim-and-morph-loop.md](../design/architecture-v0.5-morphtwin-deformable-real2sim-and-morph-loop.md)
- **Workspace Co-Existence**: Zero edits to active `distillation_v04` files and zero GPU contention with running local jobs.

---

## 1. Delivered Modules Across Four Repositories

### `ConvertAsset` (`M1` + `M2`)
- `convert_asset/deformable_v05/contracts_v05.py`: `multi_engine_deformable_bundle/v0.2` (backward-compatible with `v0.1`), `MorphCalibrationRecord`, `VolumetricShearProbeSummary`, `FractureBudgetV02`.
- `convert_asset/deformable_v05/response_carriers.py`: MorphTwin Relaxable Response Carriers (bulk centroid + paired circumsphere shear auxiliary nodes), reduced elastic energy $\overline{\mathcal{E}}(\mathbf{E})$ relaxation over $\mathbf{E}_{\text{vol}}$ and 5 orthonormal deviatoric strains, $2 \times 2$ Jacobian $\mathbf{A} = \partial\mathbf{q}/\partial\mathbf{u}$, explicit spring fracture, and `gs_binding.npz` export.
- `convert_asset/deformable_v05/morph_compiler.py`: 4-engine decoupled volumetric/shear calibration + `apply_material_morph()` + `transfer_material_response()`.
- `convert_asset/video_deformable_intake/` (`contracts.py`, `synthetic_proxy.py`, `remote_worker.py`): `deformable_video_observation/v0.1`, offline synthetic Spring-Gaus CI proxy, `import_morphtwin_export()`, and lazy-import Colab Pro+ / RTX 4090 worker entrypoint.
- `tests/test_deformable_v05_morphtwin.py`

### `scenario-forge` (`M3`)
- `src/scenario_forge/adapters/deformable_v05/material_morph_compiler.py`: `MaterialMorphSpec`, `build_morphed_scenario_package()`, `build_material_morph_suite()` (verifying frozen geometry/pose/task SHA-256 invariance across variants + `real2sim-result/v0.1` schema output).
- `tests/test_deformable_v05_morph_packages_m3.py`

### `trajectory-forge` (`M4`)
- `src/trajectory_forge/core/material_morph_firewall.py`: `MaterialMorphFirewallError`, `MaterialMorphFirewallProxy`, AST audit `audit_controller_material_firewall()`, and `validate_exported_morph_manifest()`.
- `src/trajectory_forge/skills/tactile_adaptive_v05.py`: `TactileComplianceProbe` + zero-edit `AdaptiveDeformableSkillController` + `run_tactile_adaptive_rollout()`.
- `tests/test_architecture_boundaries.py`: expanded `FORBIDDEN_IMPORT_MARKERS` with `gsplat`, `sam2`, `cotracker`, `trellis`.
- `tests/test_deformable_v05_m4.py`

### `embodied-eval-os` (`M5`)
- `src/embodied_eval_os/core/evaluators/deformable_integrity.py`: additive rotation-invariant volumetric/deviatoric polar strain split (`compute_volumetric_and_deviatoric_strain_from_F`), monotonic `compute_fracture_mask_from_F_seq`, and `evaluate_v05_npz_with_vol_dev_fracture()`.
- `src/embodied_eval_os/renderers/gaussian_twin.py`: pluggable `GaussianTwinRendererAdapter(RendererAdapter)` with surviving-edge fracture masking + `compute_dual_renderer_metrics()` (RTX PBR vs. 3DGS).
- `src/embodied_eval_os/core/reports/deformable_v05_router.py`: `evaluate_protocol_c1_material_morph()`, `evaluate_protocol_c2_video_fidelity()`, `route_deformable_v05_failure()`, and `certify_task_v05()`.
- `scripts/run_m5_morphtwin_v05_certification.py` & `tests/test_deformable_v05_protocol_c_m5.py`

---

## 2. Observed Verification Results

- **Unified 4-Repo Pytest Suite (`17 passed` in `75.97s`)**:
  - `ConvertAsset/tests/test_deformable_v05_morphtwin.py` (`3 passed`)
  - `scenario-forge/tests/test_deformable_v05_morph_packages_m3.py` (`2 passed`)
  - `trajectory-forge/tests/test_architecture_boundaries.py` + `tests/test_deformable_v05_m4.py` (`4 passed`)
  - `embodied-eval-os/tests/test_deformable_v05_protocol_c_m5.py` + `test_deformable_integrity_and_router_m4.py` + `test_deformable_v03_m4.py` (`8 passed`)
- **One-Command Pilot 1 & Pilot 2 Certification (`scripts/run_m5_morphtwin_v05_certification.py`)**:
  - Output receipt: `embodied-eval-os/external_artifacts/deformable_v05_morphtwin_certification/v05_morphtwin_certification_summary.json`
  - **Pilot 1 (`real2sim_video_sponge_pick_place`)**: Badges = `["ProtocolB_Gold", "ProtocolC2_Real2SimVideoFidelity"]`; Poisson MAE = `0.000439` (MorphTwin carrier) vs. `0.1300` (central-force baseline).
  - **Pilot 2 (`coffee_croissant_tray_to_plate` + `smt_esd_silicone_boot_v05` transfer)**: Badges = `["ProtocolB_Gold", "ProtocolC1_MaterialGeneralization"]` across all 6 frozen-geometry variants (`nominal`, `soften_2x`, `stiffen_2x`, `poisson_low`, `poisson_high`, `transfer_smt_esd_silicone_boot_v05`), plus verified counterfactual fracture & 3-repo deterministic failure routing.

