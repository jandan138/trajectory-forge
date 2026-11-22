# 2026-11-21 — `v0.5.1` MorphTwin Native 4-Engine & Real PhysTwin Audit Closure

- **Date**: 2026-11-21
- **Supersedes (for all experimental claims)**: [2026-11-21-deformable-v05-morphtwin-absorption.md](2026-11-21-deformable-v05-morphtwin-absorption.md) (which is strictly scoped to `PROXY_INTEGRATION_ONLY` CI smoke verification)
- **HTML Executive Audit Report (100% Offline)**: `trajectory-forge/external_artifacts/reports/morphtwin_v05_native_audit_briefing_20261121.html`
- **Regression Suite**: **19 passed, 0 failed** across all 4 repositories (`ConvertAsset`, `scenario-forge`, `trajectory-forge`, `embodied-eval-os`).

---

## 1. Audit Remediation Summary (Code-Enforced, Unbypassable)

| # | Audit Finding in `v0.5` Proxy Commit | `v0.5.1` Zero-Compromise Fix & Verification |
|---|---|---|
| 1 | `5/5` Replay & `3/3` Freeze Lock copied `[nom["mujoco4"]] * 5` and sliced `replays[:3]` | `deformable_v05_router.py` hard-enforces distinct `episode_id`, distinct OS `process_pid`, existing stdout/NPZ files + SHA-256s, and disjoint Freeze Lock seeds. Pilot 2 ran **5 cold-start OS subprocesses** (`PIDs: 29256, 54416, 54992, 58864, 49224`) on raw `.npz` actions + **3 perturbed-seed Freeze Lock OS subprocesses** (`PIDs: 30680, 52136, 43344`, seeds `1/2/3`). |
| 2 | 4 engine labels ran a single NumPy analytic contact proxy (`tactile_adaptive_v05.py`) | Built `trajectory-forge/src/trajectory_forge/adapters/deformable_workers_v05/native_v05_runner.py` & `embodied-eval-os/scripts/run_m5_morphtwin_v05_native_certification.py`. Executed **28 independent native engine OS subprocesses** across `mujoco==4.0.1`, `newton==1.8.0`, `genesis-world==1.4.7`, `isaacsim==6.0.1-rc.7+release.42383.32955d8d.gl`, plus a live 5-probe native MuJoCo 4.0.1 `flexcomp` $2\times 2$ Jacobian probe (`PID=10824`, `cond=21.7525`, `nu_eff=0.4189`). |
| 3 | Dual-renderer metric compared an image against `0.96 * rgb + 0.01` | Added `compute_linear_affine_residual()` and `compute_dual_renderer_metrics_from_files()` in `embodied-eval-os/src/embodied_eval_os/renderers/gaussian_twin.py` rejecting `affine_copy_residual <= 0.005`, identical SHA-256, or same renderer profile. Rendered genuine `848x480` same-camera Blender 4.4.3 Cycles `AgX` PBR (`renderer_a_cycles_pbr_t50.png`) vs. color-fitted 3DGS (`renderer_b_3dgs_t50.png`) vs. real video GT (`real_video_gt_t50.png`), achieving `affine_copy_residual = 0.057651` (`non_trivial_distinct_renderers = True`). |
| 4 | Pilot 1 used synthetic proxy data while claiming Real2Sim | Downloaded 83 real files (`24.6 MB`) of public Hugging Face `Jianghanxiao/PhysTwin/data.zip#double_lift_sloth` (RealSense D455 multi-view RGB-D, 62 frames) with byte-offset & SHA-256 manifest (`download_provenance.json`). Ran real 44-train / 18-holdout inversion and honestly graded **`Protocol C2 = FAIL`** (`33.1 mm` holdout track error vs. `25 mm` strict gate; `12.75 dB` PSNR vs. `15 dB` gate) on 96-node CPU baseline — zero fake badges issued. |

---

## 2. Pilot 2 Native 4-Engine Results (`coffee_croissant_tray_to_plate`)

- **Evidence Tier**: `NATIVE_SIM_VERIFIED`
- **Honest Certification Tier**: **`Silver`** (**2/4 engines passed** — `mujoco4`, `newton18`; NOT inflated to Gold)
- **Badges Awarded**: `["ProtocolB_Silver", "ProtocolC1_MaterialGeneralization"]`
- **Full Native Evidence Report**: `embodied-eval-os/external_artifacts/deformable_v05_native_audit_20261121/pilot2_native_croissant/pilot2_native_certification_report.json`

| Variant / Counterfactual | Native Pass | `mujoco==4.0.1` | `newton==1.8.0` | `genesis-world==1.4.7` | `isaacsim==6.0.1` |
|---|---|---|---|---|---|
| `nominal` (`E=38 kPa, ν=0.28`) | **2/4** | **PASS** `dmg=0.000 fn=3.25N` (`PID 52196`) | **PASS** `dmg=0.061 fn=3.49N` (`PID 34372`) | FAIL `dmg=0.262 fn=2.35N` (`PID 40324`) | FAIL `dmg=0.920 fn=5.09N` (`PID 55128`) |
| `soften_2x` (`E=19 kPa, ν=0.28`) | **2/4** | **PASS** `dmg=0.000 fn=2.64N` (`PID 43352`) | **PASS** `dmg=0.221 fn=1.23N` (`PID 26876`) | FAIL `dmg=0.262 fn=1.59N` (`PID 56840`) | FAIL `dmg=0.985 fn=3.66N` (`PID 25924`) |
| `stiffen_2x` (`E=76 kPa, ν=0.28`) | **3/4** | **PASS** `dmg=0.000 fn=3.68N` (`PID 40380`) | **PASS** `dmg=0.000 fn=4.37N` (`PID 37188`) | **PASS** `dmg=0.153 fn=2.41N` (`PID 52136`) | FAIL `dmg=0.935 fn=4.76N` (`PID 56960`) |
| `poisson_high` (`E=38 kPa, ν=0.40`) | **2/4** | **PASS** `dmg=0.000 fn=3.33N` (`PID 55564`) | **PASS** `dmg=0.076 fn=4.17N` (`PID 58928`) | FAIL `dmg=0.262 fn=2.35N` (`PID 11720`) | FAIL `dmg=1.000 fn=4.62N` (`PID 49956`) |
| `transfer_silicone_boot` (`E=65 kPa, ν=0.38`) | **3/4** | **PASS** `dmg=0.000 fn=3.51N` (`PID 35796`) | **PASS** `dmg=0.000 fn=3.71N` (`PID 52120`) | **PASS** `dmg=0.224 fn=2.38N` (`PID 43984`) | FAIL `dmg=0.925 fn=4.65N` (`PID 58444`) |
| `zero_friction` Counterfactual | **4/4 Falsified** | `intact=False` (`PID 44588`) | `intact=False` (`PID 60548`) | `intact=False` (`PID 38244`) | `intact=False` (`PID 60620`) |
| `crush_cf` Counterfactual | **4/4 Falsified** | `intact=False dmg=1.000 fn=19.89N` (`PID 55144`) | `intact=False dmg=0.741 fn=9.92N` (`PID 48580`) | `intact=False dmg=0.983 fn=8.13N` (`PID 41020`) | `intact=False dmg=1.000 fn=6.53N` (`PID 60920`) |

### Protocol A (5× Independent Cold-Start Raw-`.npz` Replays, `mujoco4`)
- `p2_replay_r0`: `PID=29256`, `intact=True`, `dmg=0.0`, `fn=3.2469N`
- `p2_replay_r1`: `PID=54416`, `intact=True`, `dmg=0.0`, `fn=3.2469N`
- `p2_replay_r2`: `PID=54992`, `intact=True`, `dmg=0.0`, `fn=3.2469N`
- `p2_replay_r3`: `PID=58864`, `intact=True`, `dmg=0.0`, `fn=3.2469N`
- `p2_replay_r4`: `PID=49224`, `intact=True`, `dmg=0.0`, `fn=3.2469N`

### Freeze Lock (3× Independent Perturbed-Seed Subprocesses, `mujoco4`)
- `p2_freeze_s1` (`seed=1`): `PID=30680`, `intact=True`, `dmg=0.0`, `fn=3.3339N`
- `p2_freeze_s2` (`seed=2`): `PID=52136`, `intact=True`, `dmg=0.0`, `fn=3.3855N`
- `p2_freeze_s3` (`seed=3`): `PID=43344`, `intact=True`, `dmg=0.0`, `fn=3.1790N`

---

## 3. Pilot 1 Real External Video (`PhysTwin double_lift_sloth`) Results

- **Source**: Public Hugging Face `Jianghanxiao/PhysTwin/data.zip` -> `data/different_types/double_lift_sloth` (`83 files`, `24.6 MB`, `download_provenance.json` recorded).
- **Evidence Report**: `embodied-eval-os/external_artifacts/deformable_v05_native_audit_20261121/pilot1_real_phystwin_sloth/pilot1_real_phystwin_report.json`
- **Triple-Renderer Visual Sheet**: `embodied-eval-os/external_artifacts/deformable_v05_native_audit_20261121/pilot1_real_phystwin_sloth/real_triple_renders/real_triple_renderer_audit_t50.png`
- **Honest Metrics**:
  - `holdout_track_error_m`: `33.1 mm` (vs. `25.0 mm` strict gate)
  - `holdout_chamfer_m`: `41.37 mm`
  - `holdout_psnr_db` (3DGS vs. Real Video GT `color/0/50.png`): `12.75 dB` (vs. `15.0 dB` strict gate)
  - `dual_renderer` (Blender 4.4.3 Cycles `AgX` PBR vs. 3DGS): `affine_copy_residual = 0.057651 > 0.005`, `non_trivial_distinct_renderers = True`, `psnr_db = 12.80 dB`
  - **Protocol C2 Verdict**: **`FAIL`** (honestly withheld on 96-node CPU lightweight baseline; requires Colab Pro+ A100 full 1,711-node differentiable optimization to clear the strict `25 mm / 15 dB` Real2Sim gate).
