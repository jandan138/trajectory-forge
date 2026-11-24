# 2026-11-24 — `v0.8` HermiteTwin (`arXiv:2611.27282`) Four-Repo Integration & A/B Comparison Showcase

- **Date**: 2026-11-24
- **Branch**: `feat/deformable-v08-hermitetwin` (isolated across `ConvertAsset`, `scenario-forge`, `trajectory-forge`, `embodied-eval-os`)
- **Evidence Tier (Phase 1)**: `FAST_CHECK_CPU_PREVIEW` (pure-CPU `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `nice -n 19`, zero GPU contention with active CE-Bench `ce_op_home_v2_phase3_20261120`)
- **Offline Interactive HTML**: `external_artifacts/reports/hermitetwin_v08_comparison_showcase.html` (`3,758,530` bytes, `external_urls = 0`, 4 inline H.264 MP4s + Before/After peak curtain PNGs)

## 1. Four-Repo Architecture (`v0.8`)

1. **`ConvertAsset` (`convert_asset/deformable_v08_hermite/`)**:
   - `bicubic_hermite_shell.py`: $C^1$ Bogner–Fox–Schmit bicubic Hermite Kirchhoff–Love thin shell (`4×4`, 300 scalar DOFs) with exact $SO(3)$-invariant membrane & bending energies, analytical gradients, tensile geometric stiffness, and implicit backward-Euler stepping (`~6.9–7.7 ms/step` on 4 CPU threads).
   - `material_space_grasp.py`: Dense material-space gripper grasp state machine (`FREE`/`CAPTURED`/`CLAMPED`/`RELEASED` + Coulomb friction cone) & unilateral tabletop/obstacle contact projected via $\mathbf{W}^\top$.
   - `spring_mass_baseline.py`: Controlled single-resolution `10×10` particle/spring-mass baseline (`Old Method`) with CFL-safe adaptive substeps and unilateral tabletop/obstacle projection.
   - `sysid_cmaes_adjoint.py`: Equal-budget two-stage CMA-ES + sensitivity L-BFGS-B parameter identification.
   - `hermite_gs_binding.py`, `four_engine_shell_compiler.py`, & `real_video_cloth_intake.py`: Shell-tangent 3DGS binding (`gs_binding.npz`, 784 anisotropic Gaussians), 4-engine `multi_engine_deformable_bundle/v0.3` overlays (`mujoco4`, `newton18`, `genesis`, `isaac_physx`), and real-video intake manifest.
2. **`scenario-forge` (`src/scenario_forge/adapters/deformable_v08/`)**:
   - Frozen-hash A/B packages for `v08_cloth_corner_fold`, `v08_stiff_foam_sheet_transfer`, and `v08_contact_drape_peel_novel`.
3. **`embodied-eval-os` (`thin_shell_integrity.py` & `deformable_v08_router.py`)**:
   - Bidirectional Chamfer (`mm²`), silhouette IoU, fold crease dihedral angle error, foam out-of-plane sag/flatness retention, contact penetration, Protocol C3 A/B superiority gate, and deterministic failure routing.
4. **`trajectory-forge` (`adapters/deformable_workers_v08/` & `scripts/build_hermitetwin_v08_comparison_html.py`)**:
   - Fair equal-budget A/B rollout + SysID runner, dual-tier studio comparison video renderer (`fast_cpu` vs. `studio_cycles` Blender 4.4.3 `AgX`), and single-file self-contained HTML showcase.

## 2. Observed Phase 1 Quantitative A/B Results (`Protocol C3: 3/3 PASS`)

| Task ID | Resim CD (Old → New, ↓%) | Future CD (Old → New, ↓%) | Silhouette IoU (Old → New) | Shape Metric (Old → New) | Novel Action CD | Max Pen | Protocol A & Freeze Lock |
|---|---|---|---|---|---|---|---|
| `v08_cloth_corner_fold` | `1161.35 → 203.65 mm²` (**`-82.46%`**) | `6157.57 → 1027.68 mm²` (**`-83.31%`**) | `0.4989 → 0.7190` | Crease Err `51.98° → 26.55°` | — | `2.16 mm` | `5/5` (ZF `0/1`), `3/3` |
| `v08_stiff_foam_sheet_transfer` | `4460.62 → 118.69 mm²` (**`-97.34%`**) | `7037.80 → 129.95 mm²` (**`-98.15%`**) | `0.3246 → 0.8419` | Foam Sag Err `28.47 → 0.003 mm` | — | `0.60 mm` | `5/5` (ZF `0/1`), `3/3` |
| `v08_contact_drape_peel_novel` | `530.43 → 111.41 mm²` (**`-79.00%`**) | `5936.29 → 64.22 mm²` (**`-98.92%`**) | `0.5279 → 0.8623` | RMS Track `118.94 → 12.69 mm` | `650.49 → 256.84 mm²` (**`-60.52%`**) | `2.20 mm` | `5/5` (ZF `0/1`), `3/3` |

- **Dual-Renderer Audit (`PBR Mesh` vs. `Hermite 3DGS`)**: `PSNR = 19.82 dB`, `SSIM = 0.5721`, `affine_copy_residual = 0.087486 (> 0.005)`, `non_trivial_distinct_renderers = True`.

## 3. Phase 2 Studio Cycles `AgX` + Four-Engine Parity Command (Idle 4090 / Windows-WSL2)

Once on an idle RTX 4090 workstation with Blender 4.4.3+ and the four physics engines installed, checkout `feat/deformable-v08-hermitetwin` in all four repos and run:

```bash
python trajectory-forge/scripts/run_v08_hermitetwin_studio_release.py \
  --sandbox-root /path/to/four_repos_root \
  --tier studio_cycles \
  --blender-bin /path/to/blender
```
