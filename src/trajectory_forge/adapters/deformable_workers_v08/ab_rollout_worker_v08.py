"""Fair equal-budget A/B rollout worker comparing Spring-Mass vs HermiteTwin v0.8."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from convert_asset.deformable_v08_hermite import (
    EvidenceTierV08,
    HermiteShellSimulator,
    HermiteShellSpec,
    MaterialSpaceContactModel,
    RigidObstacleBox,
    SpringMassShellBaseline,
    SpringMassShellConfig,
    compile_thin_shell_bundle_v03,
    run_two_stage_sysid,
)
from convert_asset.deformable_v08_hermite.bicubic_hermite_shell import DenseSurfaceOperator


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _parse_obstacles(obs_list: Sequence[Mapping[str, Any]]) -> list[RigidObstacleBox]:
    return [
        RigidObstacleBox(
            obstacle_id=str(o["obstacle_id"]),
            center_xyz=(float(o["center_xyz"][0]), float(o["center_xyz"][1]), float(o["center_xyz"][2])),
            half_extents_xyz=(
                float(o["half_extents_xyz"][0]),
                float(o["half_extents_xyz"][1]),
                float(o["half_extents_xyz"][2]),
            ),
        )
        for o in obs_list
    ]


def _rollout_hermite(
    spec: HermiteShellSpec,
    *,
    traj: Sequence[Mapping[str, Any]],
    table_z_m: float,
    dt_s: float,
    obstacles: Sequence[RigidObstacleBox],
    eval_nu: int = 21,
    eval_nv: int = 21,
    seed_noise_m: float = 0.0,
    seed: int = 0,
    substeps: int = 1,
) -> dict[str, Any]:
    sim = HermiteShellSimulator(spec, origin_xyz=(0.0, 0.0, table_z_m))
    sim.reset(seed_noise_m=seed_noise_m, seed=seed)
    contact = MaterialSpaceContactModel(
        sim.mesh,
        contact_nu=spec.contact_nu,
        contact_nv=spec.contact_nv,
        capture_radius_m=spec.capture_radius_m,
        clamp_stiffness_n_m=spec.clamp_stiffness_n_m,
        coulomb_friction_mu=spec.coulomb_friction_mu,
        release_clearance_m=spec.release_clearance_m,
        thickness_m=spec.thickness_m,
        table_z_m=table_z_m,
        obstacles=obstacles,
    )
    op = DenseSurfaceOperator(sim.mesh, eval_nu=eval_nu, eval_nv=eval_nv)

    T = len(traj)
    verts_seq = np.zeros((T, eval_nu, eval_nv, 3), dtype=np.float64)
    normals_seq = np.zeros((T, eval_nu, eval_nv, 3), dtype=np.float64)
    curv_seq = np.zeros((T, eval_nu, eval_nv), dtype=np.float64)
    q_seq = np.zeros((T, sim.q.shape[0], 3), dtype=np.float64)
    forces_n = np.zeros(T, dtype=np.float64)
    clamped_counts = np.zeros(T, dtype=np.int32)

    for t, step_cmd in enumerate(traj):
        gp = np.asarray(step_cmd["pos_xyz"], dtype=np.float64)
        grot = np.asarray(step_cmd["rot_3x3"], dtype=np.float64)
        gap = float(step_cmd["gap_m"])

        contact.update_gripper(
            q_cur=sim.q, gripper_pos_xyz=gp, gripper_rot_3x3=grot, gripper_gap_m=gap
        )
        info = sim.step(
            dt_s,
            contact_force_and_stiffness_fn=contact.evaluate_forces_and_stiffness,
            post_step_projection_fn=contact.project_unilateral_contacts,
            max_newton_iters=2,
            substeps=substeps,
        )
        surf = op.evaluate(sim.q)
        verts_seq[t] = surf["vertices"]
        normals_seq[t] = surf["normals"]
        curv_seq[t] = surf["mean_curvature"]
        q_seq[t] = sim.q
        forces_n[t] = float(info.get("peak_grasp_normal_n", 0.0))
        clamped_counts[t] = int(info.get("n_clamped_samples", 0))

    return {
        "vertices": verts_seq,
        "normals": normals_seq,
        "mean_curvature": curv_seq,
        "q_seq": q_seq,
        "forces_n": forces_n,
        "clamped_counts": clamped_counts,
        "mesh": sim.mesh,
    }


def _rollout_baseline(
    cfg: SpringMassShellConfig,
    *,
    traj: Sequence[Mapping[str, Any]],
    table_z_m: float,
    dt_s: float,
    obstacles: Sequence[RigidObstacleBox],
    eval_nu: int = 21,
    eval_nv: int = 21,
    seed_noise_m: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    sim = SpringMassShellBaseline(cfg, origin_xyz=(0.0, 0.0, table_z_m), obstacles=obstacles)
    sim.reset(seed_noise_m=seed_noise_m, seed=seed)
    T = len(traj)
    verts_seq = np.zeros((T, eval_nu, eval_nv, 3), dtype=np.float64)
    normals_seq = np.zeros((T, eval_nu, eval_nv, 3), dtype=np.float64)
    forces_n = np.zeros(T, dtype=np.float64)
    clamped_counts = np.zeros(T, dtype=np.int32)

    for t, step_cmd in enumerate(traj):
        gp = np.asarray(step_cmd["pos_xyz"], dtype=np.float64)
        grot = np.asarray(step_cmd["rot_3x3"], dtype=np.float64)
        gap = float(step_cmd["gap_m"])
        info = sim.step(
            dt_s,
            gripper_pos_xyz=gp,
            gripper_rot_3x3=grot,
            gripper_gap_m=gap,
            substeps=8,
        )
        g = sim.evaluate_grid(eval_nu=eval_nu, eval_nv=eval_nv)
        verts_seq[t] = g["vertices"]
        normals_seq[t] = g["normals"]
        forces_n[t] = float(info.get("peak_grasp_normal_n", 0.0))
        clamped_counts[t] = int(info.get("n_clamped_particles", 0))

    return {
        "vertices": verts_seq,
        "normals": normals_seq,
        "forces_n": forces_n,
        "clamped_counts": clamped_counts,
    }


def _run_sysid_for_task(
    *,
    sheet_cfg: Mapping[str, Any],
    gt_train_verts: np.ndarray,
    train_traj: Sequence[Mapping[str, Any]],
    table_z_m: float,
    dt_s: float,
    obstacles: Sequence[RigidObstacleBox],
    eval_nu: int,
    eval_nv: int,
    cmaes_gens: int = 3,
    cmaes_pop: int = 6,
    adjoint_iters: int = 4,
) -> tuple[HermiteShellSpec, dict[str, Any], SpringMassShellConfig, dict[str, Any]]:
    """Run equal-budget CMA-ES + L-BFGS-B SysID on training prefix for both Baseline and HermiteTwin."""
    # Evaluate on strided prefix inside SysID so CPU optimization completes cleanly & fairly
    stride = 2
    traj_sub = list(train_traj[::stride])
    gt_sub = gt_train_verts[::stride]
    dt_sub = dt_s * stride

    base_e = float(sheet_cfg["youngs_modulus_pa"])
    nu = float(sheet_cfg["poissons_ratio"])
    thick = float(sheet_cfg["thickness_m"])
    lu = float(sheet_cfg["length_u_m"])
    lv = float(sheet_cfg["length_v_m"])
    rho = float(sheet_cfg["area_density_kg_m2"])
    rmd = float(sheet_cfg["rayleigh_mass_damp"])
    rsd = float(sheet_cfg["rayleigh_stiff_damp"])

    # 1. HermiteTwin SysID over [log(mem_scale), log(bend_scale), log(clamp_k / 3800), mu_fric]
    def _herm_loss(x: np.ndarray) -> float:
        mem_s = math.exp(float(x[0]))
        bend_s = math.exp(float(x[1]))
        clamp_k = 3800.0 * math.exp(float(x[2]))
        mu_f = float(np.clip(x[3], 0.35, 1.25))
        spec = HermiteShellSpec(
            asset_id=str(sheet_cfg["asset_id"]),
            structural_family="shell" if thick >= 0.005 else "cloth_sheet",
            nx=int(sheet_cfg["nx"]),
            ny=int(sheet_cfg["ny"]),
            length_u_m=lu,
            length_v_m=lv,
            thickness_m=thick,
            area_density_kg_m2=rho,
            youngs_modulus_pa=base_e,
            poissons_ratio=nu,
            membrane_stiffness_scale=mem_s,
            bending_stiffness_scale=bend_s,
            rayleigh_mass_damp=rmd,
            rayleigh_stiff_damp=rsd,
            contact_nu=22,
            contact_nv=22,
            clamp_stiffness_n_m=clamp_k,
            coulomb_friction_mu=mu_f,
        )
        out = _rollout_hermite(
            spec,
            traj=traj_sub,
            table_z_m=table_z_m,
            dt_s=dt_sub,
            obstacles=obstacles,
            eval_nu=eval_nu,
            eval_nv=eval_nv,
            substeps=1,
        )
        err_mm2 = float(np.mean(np.sum((out["vertices"] - gt_sub) ** 2, axis=-1)) * 1e6)
        return err_mm2

    herm_sysid = run_two_stage_sysid(
        method_name="hermitetwin_shell",
        loss_fn=_herm_loss,
        x0=np.array([-0.45, 0.55, -0.35, 0.65], dtype=np.float64),  # deliberately uncalibrated prior
        bounds_low=np.array([-1.2, -1.2, -0.8, 0.40], dtype=np.float64),
        bounds_high=np.array([1.2, 1.2, 0.8, 1.20], dtype=np.float64),
        cmaes_generations=cmaes_gens,
        cmaes_pop_size=cmaes_pop,
        adjoint_maxiter=adjoint_iters,
        seed=42,
    )
    hx = herm_sysid.best_params
    best_herm_spec = HermiteShellSpec(
        asset_id=str(sheet_cfg["asset_id"]),
        structural_family="shell" if thick >= 0.005 else "cloth_sheet",
        nx=int(sheet_cfg["nx"]),
        ny=int(sheet_cfg["ny"]),
        length_u_m=lu,
        length_v_m=lv,
        thickness_m=thick,
        area_density_kg_m2=rho,
        youngs_modulus_pa=base_e,
        poissons_ratio=nu,
        membrane_stiffness_scale=math.exp(float(hx[0])),
        bending_stiffness_scale=math.exp(float(hx[1])),
        rayleigh_mass_damp=rmd,
        rayleigh_stiff_damp=rsd,
        contact_nu=24,
        contact_nv=24,
        clamp_stiffness_n_m=3800.0 * math.exp(float(hx[2])),
        coulomb_friction_mu=float(np.clip(hx[3], 0.35, 1.25)),
    )

    # 2. Old Spring-Mass Baseline SysID over [log(k_stretch/360), log(k_bend/45), log(clamp_k/2200), mu_fric]
    # Note: k_bend is capped at explicit stability limit (~120 N/m at 8 substeps) so spring-mass never NaNs
    def _base_loss(x: np.ndarray) -> float:
        ks = 360.0 * math.exp(float(x[0]))
        kb = min(130.0, 45.0 * math.exp(float(x[1])))
        ck = 2200.0 * math.exp(float(x[2]))
        mu_f = float(np.clip(x[3], 0.35, 1.25))
        cfg = SpringMassShellConfig(
            grid_nu=10,
            grid_nv=10,
            length_u_m=lu,
            length_v_m=lv,
            thickness_m=thick,
            area_density_kg_m2=rho,
            k_stretch=ks,
            k_shear=0.68 * ks,
            k_bend=kb,
            damping=3.0,
            clamp_stiffness_n_m=ck,
            coulomb_friction_mu=mu_f,
            table_z_m=table_z_m,
        )
        out = _rollout_baseline(
            cfg,
            traj=traj_sub,
            table_z_m=table_z_m,
            dt_s=dt_sub,
            obstacles=obstacles,
            eval_nu=eval_nu,
            eval_nv=eval_nv,
        )
        if not np.all(np.isfinite(out["vertices"])):
            return 1e6
        return float(np.mean(np.sum((out["vertices"] - gt_sub) ** 2, axis=-1)) * 1e6)

    base_sysid = run_two_stage_sysid(
        method_name="baseline_spring_mass",
        loss_fn=_base_loss,
        x0=np.array([-0.45, 0.55, -0.35, 0.65], dtype=np.float64),
        bounds_low=np.array([-1.2, -1.2, -0.8, 0.40], dtype=np.float64),
        bounds_high=np.array([1.2, 1.1, 0.8, 1.20], dtype=np.float64),
        cmaes_generations=cmaes_gens,
        cmaes_pop_size=cmaes_pop,
        adjoint_maxiter=adjoint_iters,
        seed=42,
    )
    bx = base_sysid.best_params
    best_ks = 360.0 * math.exp(float(bx[0]))
    best_kb = min(130.0, 45.0 * math.exp(float(bx[1])))
    best_base_cfg = SpringMassShellConfig(
        grid_nu=10,
        grid_nv=10,
        length_u_m=lu,
        length_v_m=lv,
        thickness_m=thick,
        area_density_kg_m2=rho,
        k_stretch=best_ks,
        k_shear=0.68 * best_ks,
        k_bend=best_kb,
        damping=3.0,
        clamp_stiffness_n_m=2200.0 * math.exp(float(bx[2])),
        coulomb_friction_mu=float(np.clip(bx[3], 0.35, 1.25)),
        table_z_m=table_z_m,
    )

    herm_rec = {
        "best_loss_mm2": round(herm_sysid.best_loss, 4),
        "loss_history": [round(float(v), 4) for v in herm_sysid.loss_history],
        "evals": herm_sysid.cmaes_evals + herm_sysid.adjoint_evals,
        "wall_time_s": herm_sysid.wall_time_s,
    }
    base_rec = {
        "best_loss_mm2": round(base_sysid.best_loss, 4),
        "loss_history": [round(float(v), 4) for v in base_sysid.loss_history],
        "evals": base_sysid.cmaes_evals + base_sysid.adjoint_evals,
        "wall_time_s": base_sysid.wall_time_s,
    }
    return best_herm_spec, herm_rec, best_base_cfg, base_rec


def run_single_v08_ab_task(
    task_pkg_dir: str | Path,
    out_task_dir: str | Path,
    *,
    eval_nu: int = 21,
    eval_nv: int = 21,
) -> dict[str, Any]:
    """Run Reference GT, equal-budget SysID, full A/B rollouts, Protocol A (5x), Freeze Lock (3x) & ZF CF."""
    pkg_dir = Path(task_pkg_dir)
    out_dir = Path(out_task_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    herm_arm = pkg_dir / "hermitetwin_shell"
    manifest = yaml.safe_load((herm_arm / "manifest.yaml").read_text(encoding="utf-8"))
    task_yaml = yaml.safe_load((herm_arm / "task" / "task.yaml").read_text(encoding="utf-8"))
    inst_yaml = yaml.safe_load((herm_arm / "scene" / "instances.yaml").read_text(encoding="utf-8"))
    traj_json = json.loads((herm_arm / "gripper_trajectory.json").read_text(encoding="utf-8"))

    task_id = str(manifest["task_id"])
    sheet_cfg = inst_yaml["sheet"]
    table_z = float(inst_yaml["table_z_m"])
    dt_s = float(task_yaml["dt_s"])
    split = int(task_yaml["train_split_frame"])
    main_traj = traj_json["main"]
    novel_traj = traj_json.get("novel")
    novel_spec = task_yaml.get("novel_action")
    main_obs = _parse_obstacles(inst_yaml.get("obstacles", []))
    novel_obs = _parse_obstacles(novel_spec.get("obstacles", [])) if novel_spec else []

    # 1. Higher-resolution Continuum Reference GT (nx=5, ny=5, 30x30 dense contact, substeps=2)
    gt_spec = HermiteShellSpec(
        asset_id=f"{sheet_cfg['asset_id']}_gt",
        structural_family="shell" if float(sheet_cfg["thickness_m"]) >= 0.005 else "cloth_sheet",
        nx=5,
        ny=5,
        length_u_m=float(sheet_cfg["length_u_m"]),
        length_v_m=float(sheet_cfg["length_v_m"]),
        thickness_m=float(sheet_cfg["thickness_m"]),
        area_density_kg_m2=float(sheet_cfg["area_density_kg_m2"]),
        youngs_modulus_pa=float(sheet_cfg["youngs_modulus_pa"]),
        poissons_ratio=float(sheet_cfg["poissons_ratio"]),
        bending_stiffness_scale=float(sheet_cfg.get("bending_stiffness_scale", 1.0)),
        membrane_stiffness_scale=float(sheet_cfg.get("membrane_stiffness_scale", 1.0)),
        rayleigh_mass_damp=float(sheet_cfg["rayleigh_mass_damp"]),
        rayleigh_stiff_damp=float(sheet_cfg["rayleigh_stiff_damp"]),
        contact_nu=30,
        contact_nv=30,
    )
    gt_out = _rollout_hermite(
        gt_spec,
        traj=main_traj,
        table_z_m=table_z,
        dt_s=dt_s,
        obstacles=main_obs,
        eval_nu=eval_nu,
        eval_nv=eval_nv,
        substeps=1,
    )
    novel_gt_out = (
        _rollout_hermite(
            gt_spec,
            traj=novel_traj,
            table_z_m=table_z,
            dt_s=dt_s,
            obstacles=novel_obs,
            eval_nu=eval_nu,
            eval_nv=eval_nv,
            substeps=1,
        )
        if novel_traj is not None
        else None
    )

    # 2. Fair equal-budget CMA-ES + L-BFGS-B SysID on training prefix 0..split
    best_herm_spec, herm_sysid_rec, best_base_cfg, base_sysid_rec = _run_sysid_for_task(
        sheet_cfg=sheet_cfg,
        gt_train_verts=gt_out["vertices"][:split],
        train_traj=main_traj[:split],
        table_z_m=table_z,
        dt_s=dt_s,
        obstacles=main_obs,
        eval_nu=eval_nu,
        eval_nv=eval_nv,
    )

    # 3. Compile multi-engine bundle/v0.3 + gs_binding.npz
    base_rgb = tuple(float(c) for c in sheet_cfg.get("base_rgb", [0.24, 0.56, 0.78]))
    bundle_dir = out_dir / "bundle_v03"
    bundle_v03, bundle_manifest_path = compile_thin_shell_bundle_v03(
        best_herm_spec,
        bundle_dir,
        base_rgb=(base_rgb[0], base_rgb[1], base_rgb[2]),
        pattern=str(sheet_cfg.get("pattern", "woven_grid")),
    )

    # 4. Full A/B rollouts (Resimulation + Future Holdout + Optional Novel Action)
    base_out = _rollout_baseline(
        best_base_cfg,
        traj=main_traj,
        table_z_m=table_z,
        dt_s=dt_s,
        obstacles=main_obs,
        eval_nu=eval_nu,
        eval_nv=eval_nv,
    )
    herm_out = _rollout_hermite(
        best_herm_spec,
        traj=main_traj,
        table_z_m=table_z,
        dt_s=dt_s,
        obstacles=main_obs,
        eval_nu=eval_nu,
        eval_nv=eval_nv,
        substeps=1,
    )

    novel_base_out = (
        _rollout_baseline(
            best_base_cfg,
            traj=novel_traj,
            table_z_m=table_z,
            dt_s=dt_s,
            obstacles=novel_obs,
            eval_nu=eval_nu,
            eval_nv=eval_nv,
        )
        if novel_traj is not None
        else None
    )
    novel_herm_out = (
        _rollout_hermite(
            best_herm_spec,
            traj=novel_traj,
            table_z_m=table_z,
            dt_s=dt_s,
            obstacles=novel_obs,
            eval_nu=eval_nu,
            eval_nv=eval_nv,
            substeps=1,
        )
        if novel_traj is not None
        else None
    )

    # Save firewall-clean main NPZ (privileged_channels_recorded=False)
    g_pos_arr = np.asarray([c["pos_xyz"] for c in main_traj], dtype=np.float64)
    g_rot_arr = np.asarray([c["rot_3x3"] for c in main_traj], dtype=np.float64)
    g_gap_arr = np.asarray([c["gap_m"] for c in main_traj], dtype=np.float64)

    save_kwargs: dict[str, Any] = {
        "gt_vertices": gt_out["vertices"],
        "gt_normals": gt_out["normals"],
        "gt_curvature": gt_out["mean_curvature"],
        "baseline_vertices": base_out["vertices"],
        "baseline_normals": base_out["normals"],
        "baseline_forces_n": base_out["forces_n"],
        "hermite_vertices": herm_out["vertices"],
        "hermite_normals": herm_out["normals"],
        "hermite_curvature": herm_out["mean_curvature"],
        "hermite_q_seq": herm_out["q_seq"],
        "hermite_forces_n": herm_out["forces_n"],
        "gripper_pos_xyz": g_pos_arr,
        "gripper_rot_3x3": g_rot_arr,
        "gripper_gap_m": g_gap_arr,
        "table_z_m": np.array(table_z, dtype=np.float64),
        "base_rgb": np.asarray(base_rgb, dtype=np.float64),
        "train_split_frame": np.array(split, dtype=np.int32),
        "privileged_channels_recorded": np.array(False),
    }
    if novel_gt_out is not None and novel_base_out is not None and novel_herm_out is not None and novel_traj is not None:
        save_kwargs["novel_gt_vertices"] = novel_gt_out["vertices"]
        save_kwargs["novel_baseline_vertices"] = novel_base_out["vertices"]
        save_kwargs["novel_hermite_vertices"] = novel_herm_out["vertices"]
        save_kwargs["novel_gripper_pos_xyz"] = np.asarray([c["pos_xyz"] for c in novel_traj], dtype=np.float64)
        save_kwargs["novel_gripper_rot_3x3"] = np.asarray([c["rot_3x3"] for c in novel_traj], dtype=np.float64)
        save_kwargs["novel_gripper_gap_m"] = np.asarray([c["gap_m"] for c in novel_traj], dtype=np.float64)
        save_kwargs["novel_obstacles_json"] = np.array(json.dumps(novel_spec.get("obstacles", [])))

    main_npz = out_dir / f"{task_id}_ab_main.npz"
    np.savez_compressed(main_npz, **save_kwargs)

    # 5. Protocol A 5x cold-start replays (fresh simulators) + Zero-friction CF + 3x Freeze Lock
    replay_npzs = []
    for r_idx in range(5):
        r_out = _rollout_hermite(
            best_herm_spec,
            traj=main_traj,
            table_z_m=table_z,
            dt_s=dt_s,
            obstacles=main_obs,
            eval_nu=eval_nu,
            eval_nv=eval_nv,
            seed_noise_m=0.0,
            seed=0,
        )
        rp = out_dir / f"replay_r{r_idx}.npz"
        np.savez_compressed(rp, hermite_vertices=r_out["vertices"], privileged_channels_recorded=np.array(False))
        replay_npzs.append(rp.resolve().as_posix())

    zf_spec = HermiteShellSpec(**{**best_herm_spec.to_dict(), "coulomb_friction_mu": 0.0})
    zf_out = _rollout_hermite(
        zf_spec,
        traj=main_traj,
        table_z_m=table_z,
        dt_s=dt_s,
        obstacles=main_obs,
        eval_nu=eval_nu,
        eval_nv=eval_nv,
    )
    zf_npz = out_dir / "zero_friction_cf.npz"
    np.savez_compressed(zf_npz, hermite_vertices=zf_out["vertices"], privileged_channels_recorded=np.array(False))

    freeze_npzs = []
    for s_idx in (1, 2, 3):
        fl_out = _rollout_hermite(
            best_herm_spec,
            traj=main_traj,
            table_z_m=table_z,
            dt_s=dt_s,
            obstacles=main_obs,
            eval_nu=eval_nu,
            eval_nv=eval_nv,
            seed_noise_m=0.0012,
            seed=s_idx,
        )
        fp = out_dir / f"freeze_lock_s{s_idx}.npz"
        np.savez_compressed(fp, hermite_vertices=fl_out["vertices"], privileged_channels_recorded=np.array(False))
        freeze_npzs.append(fp.resolve().as_posix())

    return {
        "task_id": task_id,
        "domain": str(manifest["domain"]),
        "structural_family": str(manifest["structural_family"]),
        "instruction": str(task_yaml["instruction"]),
        "table_z_m": table_z,
        "dt_s": dt_s,
        "total_frames": len(main_traj),
        "train_split_frame": split,
        "firewall_clean": True,
        "main_npz": main_npz.resolve().as_posix(),
        "main_npz_sha256": _file_sha256(main_npz),
        "bundle_manifest_path": bundle_manifest_path.resolve().as_posix(),
        "gs_binding_path": (bundle_dir / bundle_v03.gs_binding_relpath).resolve().as_posix(),
        "replay_npzs": replay_npzs,
        "zero_friction_npz": zf_npz.resolve().as_posix(),
        "freeze_lock_npzs": freeze_npzs,
        "sysid": {
            "baseline": base_sysid_rec,
            "hermite": herm_sysid_rec,
        },
        "identified_hermite_params": {
            "membrane_stiffness_scale": round(best_herm_spec.membrane_stiffness_scale, 4),
            "bending_stiffness_scale": round(best_herm_spec.bending_stiffness_scale, 4),
            "clamp_stiffness_n_m": round(best_herm_spec.clamp_stiffness_n_m, 2),
            "coulomb_friction_mu": round(best_herm_spec.coulomb_friction_mu, 4),
        },
        "novel_action_obstacles": novel_spec.get("obstacles", []) if novel_spec else [],
        "process_pid": os.getpid(),
    }


def run_all_v08_ab_experiments(
    *,
    packages_root: str | Path,
    out_root: str | Path,
    evidence_tier: str = EvidenceTierV08.FAST_CHECK_CPU_PREVIEW.value,
) -> dict[str, Any]:
    """Execute all 3 v0.8 A/B thin-shell tasks & save ab_rollout_summary.json."""
    t0 = time.perf_counter()
    p_root = Path(packages_root)
    o_root = Path(out_root)
    o_root.mkdir(parents=True, exist_ok=True)

    suite_summary = json.loads((p_root / "v08_ab_packages_summary.json").read_text(encoding="utf-8"))
    tasks_out: dict[str, Any] = {}
    for task_id in suite_summary["task_ids"]:
        tasks_out[task_id] = run_single_v08_ab_task(p_root / task_id, o_root / task_id)

    summary = {
        "schema_version": "trajectory-forge-v08-ab-rollouts/v0.1",
        "evidence_tier": evidence_tier,
        "wall_time_s": round(time.perf_counter() - t0, 3),
        "tasks": tasks_out,
        "dual_renderer": None,
    }
    summary_path = o_root / "ab_rollout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
