"""Isolated MuJoCo 4.0.1 CPU rollout worker for M3 deformable manipulation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (str(_SRC_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trajectory_forge.adapters.deformable_workers.common import (  # noqa: E402
    apply_low_priority_process_policy,
    build_voxel_tet_grid,
    compute_damage_score,
    compute_edge_stretch_from_points,
    compute_tet_strains_from_points,
    jitter_for_seed,
    load_brief_and_controller,
    load_cloth_mesh_from_package,
    save_episode_bundle,
)


def _patch_panda_xml(pkg_dir: Path, finger_mu: float, condim: int = 3) -> tuple[str, str]:
    meshes_dir = (pkg_dir / "assets" / "franka_panda" / "meshes").resolve().as_posix()
    panda_raw = (pkg_dir / "assets" / "franka_panda" / "panda.xml").read_text(encoding="utf-8")
    wb_inner = panda_raw.split("<worldbody>")[1].split("</worldbody>")[0]
    wb_inner = re.sub(r'<body (name="panda_[^"]+")', r'<body \1 gravcomp="1"', wb_inner)
    wb_inner = wb_inner.replace(
        '<body name="panda_link0" gravcomp="1" pos="0 0 0">',
        '<body name="panda_link0" gravcomp="1" childclass="panda" pos="0 -0.48 0.755" euler="0 0 1.57079632679">',
    )
    f_str = f"{finger_mu:.6g} 0.02 0.002"
    wb_inner = wb_inner.replace(
        'pos="0 0.008 0.036" size="0.010 0.008 0.018" rgba="0.72 0.74 0.76 1" solref="0.004 1"',
        f'pos="0 0.006 0.036" size="0.018 0.006 0.018" rgba="0.72 0.74 0.76 1" condim="{condim}" friction="{f_str}" solref="0.005 1"',
    )
    wb_inner = wb_inner.replace(
        'pos="0 -0.008 0.036" size="0.010 0.008 0.018" rgba="0.72 0.74 0.76 1" solref="0.004 1"',
        f'pos="0 -0.006 0.036" size="0.018 0.006 0.018" rgba="0.72 0.74 0.76 1" condim="{condim}" friction="{f_str}" solref="0.005 1"',
    )
    return meshes_dir, wb_inner


def _build_scene_xml(
    *,
    pkg_dir: Path,
    task_id: str,
    zero_friction: bool,
    seed: int,
    mode: str,
) -> tuple[str, np.ndarray, np.ndarray]:
    dx, dy, _ = jitter_for_seed(seed)
    condim = 1 if zero_friction else 3
    finger_mu = 0.0 if zero_friction else 1.25
    def_f = "0.0 0.0 0.0" if zero_friction else "1.25 0.02 0.002"
    meshes_dir, wb_inner = _patch_panda_xml(pkg_dir, finger_mu, condim=condim)

    if task_id == "coffee_bar_fragile_pastry_plate":
        rest_pts, elem_idx = build_voxel_tet_grid((0.07, 0.07, 0.028), grid_xyz=(4, 4, 2))
        px = -0.18 + dx
        py = -0.02 + dy
        task_bodies = f"""
    <body name="pastry_box" pos="-0.18 -0.02 0.755">
      <geom type="box" pos="0 0 0.005" size="0.068 0.068 0.005" friction="0.6 0.02 0.002"/>
      <geom type="box" pos="0 0.064 0.025" size="0.068 0.004 0.025"/>
      <body name="lid_link" pos="0 0.064 0.052">
        <inertial pos="0 -0.06 0" mass="0.040" diaginertia="0.0006 0.0006 0.0006"/>
        <joint name="lid_hinge" type="hinge" axis="-1 0 0" range="0 2.15" damping="0.025"/>
        <geom name="lid_plate" type="box" pos="0 -0.064 0" size="0.066 0.064 0.003"/>
        <geom name="lid_tab" type="box" pos="0 -0.142 0.002" size="0.032 0.014 0.004"/>
      </body>
    </body>
    <body name="serving_plate" pos="0.18 -0.02 0.755">
      <geom name="plate_base" type="cylinder" pos="0 0 0.004" size="0.095 0.004" friction="0.8 0.02 0.002"/>
    </body>
    <flexcomp name="fragile_pastry" type="grid" count="5 5 3" spacing="0.0175 0.0175 0.014" pos="{px:.5f} {py:.5f} 0.7815" dim="3" mass="0.13034" radius="0.002">
      <contact friction="{def_f}" condim="{condim}" selfcollide="none" solref="0.005 1"/>
      <edge equality="false"/>
      <elasticity young="35000" poisson="0.32" damping="0.05"/>
    </flexcomp>"""
    else:
        cloth_obj = (pkg_dir / "assets" / "pilot_flexible_wiping_cloth_01" / "engines" / "mujoco4" / "mesh.obj").resolve().as_posix()
        rest_pts, _, elem_idx = load_cloth_mesh_from_package(pkg_dir, "pilot_flexible_wiping_cloth_01")
        cx = -0.13 + dx
        cy = -0.02 + dy
        # Identical scene across gentle / tear / replay / zero_friction:
        # blotter_left supports t=0 cloth (x in [-0.22, -0.04]), recessed drain slot at x in [+0.002, +0.034]
        # with high-friction slot lips (x=-0.006, +0.042) snags/overloads unguarded plunges while lifted wipe clears.
        task_bodies = f"""
    <body name="wipe_zone_mat" pos="0 0 0">
      <geom name="blotter_left" type="box" pos="-0.139 0.02 0.722" size="0.141 0.18 0.035" friction="0.35 0.01 0.001"/>
      <geom name="blotter_right" type="box" pos="0.157 0.02 0.722" size="0.123 0.18 0.035" friction="0.35 0.01 0.001"/>
      <geom name="cloth_spine" type="box" pos="-0.130 -0.020 0.769" size="0.002 0.055 0.011" friction="1.5 0.02 0.002"/>
      <geom name="cloth_wing_f" type="box" pos="-0.112 -0.060 0.778" size="0.018 0.018 0.002" friction="1.5 0.02 0.002"/>
      <geom name="cloth_wing_b" type="box" pos="-0.112  0.020 0.778" size="0.018 0.018 0.002" friction="1.5 0.02 0.002"/>
      <geom name="slot_lip_l" type="box" pos="-0.006 0.04 0.7555" size="0.006 0.12 0.0025" friction="6.0 0.05 0.01"/>
      <geom name="slot_lip_r" type="box" pos="0.042 0.04 0.7555" size="0.006 0.12 0.0025" friction="6.0 0.05 0.01"/>
    </body>
    <body name="cloth_caddy_tray" pos="0.22 -0.04 0.757">
      <geom name="caddy_base" type="box" pos="0 0 0.004" size="0.095 0.095 0.004" friction="0.7 0.02 0.002"/>
      <geom name="caddy_rim_l" type="box" pos="-0.091 0 0.013" size="0.004 0.095 0.009"/>
      <geom name="caddy_rim_r" type="box" pos="0.091 0 0.013" size="0.004 0.095 0.009"/>
      <geom name="caddy_rim_b" type="box" pos="0 0.091 0.013" size="0.095 0.004 0.009"/>
    </body>
    <flexcomp name="wiping_cloth" type="mesh" file="{cloth_obj}" pos="{cx:.5f} {cy:.5f} 0.761" dim="2" mass="0.045" radius="0.003">
      <contact friction="{def_f}" condim="{condim}" selfcollide="none" solref="0.005 1"/>
      <edge equality="false"/>
      <elasticity young="20000" poisson="0.30" thickness="0.010" damping="0.04" elastic2d="both"/>
    </flexcomp>"""

    table_xml = (
        '<geom name="table_top" type="box" pos="0 0 0.745" size="1.0 0.45 0.01" friction="0.5 0.02 0.002" solref="0.005 1"/>'
        if task_id == "coffee_bar_fragile_pastry_plate"
        else '<geom name="table_top" type="box" pos="0 0 0.665" size="1.0 0.45 0.01" friction="0.5 0.02 0.002" solref="0.005 1"/>'
    )
    full_xml = f"""<mujoco model="{task_id}_mj4">
  <compiler angle="radian" meshdir="{meshes_dir}" autolimits="true"/>
  <option timestep="0.002" integrator="discrete" iterations="80"><flag speculative="enable"/></option>
  <asset>
    <mesh name="link0_m" file="collision/link0.obj"/>
    <mesh name="link1_m" file="visual/link1.obj"/>
    <mesh name="link2_m" file="visual/link2.obj"/>
    <mesh name="link3_m" file="visual/link3.obj"/>
    <mesh name="link4_m" file="visual/link4.obj"/>
    <mesh name="link5_m" file="visual/link5.obj"/>
    <mesh name="link6_m" file="visual/link6.obj"/>
    <mesh name="link7_m" file="collision/link7.obj"/>
    <mesh name="hand_m" file="visual/hand.obj"/>
    <mesh name="finger_m" file="visual/finger.obj"/>
  </asset>
  <default>
    <default class="panda">
      <joint armature="0.08" damping="1.0"/>
      <geom friction="0.6 0.02 0.002"/>
    </default>
  </default>
  <worldbody>
    {table_xml}
    {task_bodies}
    {wb_inner}
  </worldbody>
  <actuator>
    <position name="act_j1" joint="panda_joint1" kp="1400" kv="95" ctrlrange="-2.9671 2.9671"/>
    <position name="act_j2" joint="panda_joint2" kp="1400" kv="95" ctrlrange="-1.8326 1.8326"/>
    <position name="act_j3" joint="panda_joint3" kp="1400" kv="95" ctrlrange="-2.9671 2.9671"/>
    <position name="act_j4" joint="panda_joint4" kp="1400" kv="95" ctrlrange="-3.1416 0.0"/>
    <position name="act_j5" joint="panda_joint5" kp="600" kv="45" ctrlrange="-2.9671 2.9671"/>
    <position name="act_j6" joint="panda_joint6" kp="600" kv="45" ctrlrange="-0.0873 3.8223"/>
    <position name="act_j7" joint="panda_joint7" kp="450" kv="35" ctrlrange="-2.9671 2.9671"/>
    <position name="act_f1" joint="panda_finger_joint1" kp="1100" kv="45" ctrlrange="0.0 0.04"/>
    <position name="act_f2" joint="panda_finger_joint2" kp="1100" kv="45" ctrlrange="0.0 0.04"/>
  </actuator>
</mujoco>"""
    return full_xml, rest_pts, elem_idx


def _extract_finger_forces(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    pad_l: int,
    pad_r: int,
    desk_geoms: tuple[int, ...] = (),
) -> tuple[float, float, np.ndarray]:
    vec_l = np.zeros(3, dtype=np.float64)
    vec_r = np.zeros(3, dtype=np.float64)
    f6 = np.zeros(6, dtype=np.float64)
    for ci in range(d.ncon):
        c = d.contact[ci]
        if c.geom1 in (pad_l, pad_r) or c.geom2 in (pad_l, pad_r):
            other = c.geom2 if c.geom1 in (pad_l, pad_r) else c.geom1
            if not (other < 0 or other in desk_geoms):
                continue
            mujoco.mj_contactForce(m, d, ci, f6)
            # Project contact-frame force to world 3D resultant vector per pad to avoid vertex over-counting
            f_local = np.array([min(150.0, abs(float(f6[0]))), float(f6[1]), float(f6[2])], dtype=np.float64)
            f_world = np.asarray(c.frame, dtype=np.float64).reshape(3, 3).T @ f_local
            if c.geom1 == pad_l or c.geom2 == pad_l:
                vec_l += f_world
            if c.geom1 == pad_r or c.geom2 == pad_r:
                vec_r += f_world
    fn_l = min(160.0, float(np.linalg.norm(vec_l)))
    fn_r = min(160.0, float(np.linalg.norm(vec_r)))
    fz = min(160.0, abs(float(vec_l[2])) + abs(float(vec_r[2])))
    wrist_w = np.array([0.0, 0.0, -(fn_l + fn_r) * 0.3 - fz, 0.0, 0.0, 0.0], dtype=np.float64)
    return fn_l, fn_r, wrist_w


def run_single_episode(job: dict[str, Any]) -> dict[str, Any]:
    pkg_dir = Path(job["package_dir"])
    if (pkg_dir / "scenario_package" / "assets").is_dir():
        pkg_dir = pkg_dir / "scenario_package"
    task_id = str(job["task_id"])
    mode = str(job["mode"])
    seed = int(job.get("seed", 0))
    zero_friction = bool(job.get("zero_friction", False))
    replay_npz = job.get("replay_npz")
    output_dir = Path(job["output_dir"])
    episode_id = str(job["episode_id"])

    brief, ctrl = load_brief_and_controller(pkg_dir, task_id, "gentle" if mode == "replay" else mode)
    replay_actions = None
    if replay_npz:
        with np.load(replay_npz, allow_pickle=False) as z:
            replay_actions = np.asarray(z["action"], dtype=np.float64)

    xml_str, rest_pts_local, elem_idx = _build_scene_xml(
        pkg_dir=pkg_dir,
        task_id=task_id,
        zero_friction=zero_friction,
        seed=seed,
        mode=mode,
    )
    m = mujoco.MjModel.from_xml_string(xml_str)
    d = mujoco.MjData(m)

    q_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
        "panda_finger_joint1", "panda_finger_joint2",
    ]
    q_idx = [m.joint(n).qposadr[0] for n in q_names]
    pad_l = m.geom("left_finger_pad").id
    pad_r = m.geom("right_finger_pad").id
    ee_site = m.site("panda_grasptarget").id
    lid_adr = m.joint("lid_hinge").qposadr[0] if task_id == "coffee_bar_fragile_pastry_plate" else -1
    desk_names = ("blotter_left", "blotter_right", "slot_lip_l", "slot_lip_r", "table_top")
    desk_geoms = tuple(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
        for nm in desk_names
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm) >= 0 and lid_adr < 0
    )

    # Initialize arm near step 0 target so t=0 begins settled over workspace
    first_act = replay_actions[0] if replay_actions is not None else ctrl.step(
        {"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(9), "tactile_normal_force_n": np.zeros(2), "wrist_wrench": np.zeros(6)}
    )
    for i, qi in enumerate(q_idx):
        d.qpos[qi] = first_act[i]
    mujoco.mj_forward(m, d)
    rest_world = d.flexvert_xpos.copy().astype(np.float64)

    total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)
    substeps = 5

    qpos_hist = np.zeros((total_steps, 9), dtype=np.float64)
    act_hist = np.zeros((total_steps, 9), dtype=np.float64)
    ee_pos_hist = np.zeros((total_steps, 3), dtype=np.float64)
    ee_quat_hist = np.zeros((total_steps, 4), dtype=np.float64)
    tac_hist = np.zeros((total_steps, 2), dtype=np.float64)
    wrench_hist = np.zeros((total_steps, 6), dtype=np.float64)
    pts_hist = np.zeros((total_steps, len(rest_world), 3), dtype=np.float64)
    lid_hist = np.zeros((total_steps,), dtype=np.float64)

    wipe_wps = np.array([[-0.13, 0.04], [-0.04, 0.07], [0.05, 0.04]], dtype=np.float64)
    wipe_min_dists = np.full((3,), 1e3, dtype=np.float64)
    peak_polar = 0.0
    peak_naive = 0.0
    peak_edge = 0.0
    max_z = float(rest_world[:, 2].mean())

    for k in range(total_steps):
        q_obs = np.array([d.qpos[qi] for qi in q_idx], dtype=np.float64)
        ee_p = d.site_xpos[ee_site].copy()
        fn_l, fn_r, w_obs = _extract_finger_forces(m, d, pad_l, pad_r, desk_geoms)
        ee_r = d.site_xmat[ee_site].reshape(3, 3)

        if replay_actions is not None:
            act = replay_actions[k]
        else:
            act = ctrl.step(
                {
                    "step_idx": k,
                    "t_s": k * 0.01,
                    "qpos": q_obs,
                    "ee_pos": ee_p,
                    "tactile_normal_force_n": np.array([fn_l, fn_r], dtype=np.float64),
                    "wrist_wrench": w_obs,
                }
            )
        d.ctrl[:] = act
        for _ in range(substeps):
            mujoco.mj_step(m, d)

        cur_pts = d.flexvert_xpos.copy().astype(np.float64)
        qpos_hist[k] = np.array([d.qpos[qi] for qi in q_idx], dtype=np.float64)
        act_hist[k] = act
        ee_pos_hist[k] = d.site_xpos[ee_site].copy()
        # Extract wxyz quaternion from site_xmat
        mat = d.site_xmat[ee_site].reshape(3, 3)
        q_wxyz = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q_wxyz, mat.reshape(9))
        ee_quat_hist[k] = q_wxyz
        tac_hist[k] = np.array([fn_l, fn_r], dtype=np.float64)
        wrench_hist[k] = w_obs
        pts_hist[k] = cur_pts

        max_z = max(max_z, float(cur_pts[:, 2].mean()))
        if lid_adr >= 0:
            lid_hist[k] = float(d.qpos[lid_adr])
            p_s, n_s = compute_tet_strains_from_points(rest_world, cur_pts, elem_idx)
            peak_polar = max(peak_polar, p_s)
            peak_naive = max(peak_naive, n_s)
        else:
            e_s = compute_edge_stretch_from_points(rest_world, cur_pts, elem_idx)
            peak_edge = max(peak_edge, e_s)
            com_xy = cur_pts[:, :2].mean(axis=0)
            for wi in range(3):
                wipe_min_dists[wi] = min(wipe_min_dists[wi], float(np.linalg.norm(com_xy - wipe_wps[wi])))

    def_spec = (
        brief.objects["fragile_pastry"].deformable
        if task_id == "coffee_bar_fragile_pastry_plate"
        else brief.objects["wiping_cloth"].deformable
    )
    assert def_spec is not None

    fin_com = pts_hist[-1].mean(axis=0)
    peak_force_n = float(np.max(np.abs(tac_hist))) if len(tac_hist) else 0.0
    if task_id == "coffee_bar_fragile_pastry_plate":
        target_xy = np.array([0.18, -0.02], dtype=np.float64)
        final_dist = float(np.linalg.norm(fin_com[:2] - target_xy))
        max_lid = float(lid_hist.max())
        geom_ok = bool(max_lid >= 1.10 and max_z >= 0.85 and final_dist <= 0.085)
        damage = compute_damage_score(
            peak_polar,
            def_spec.strain_onset_a,
            def_spec.strain_failure_b,
            peak_force_n=peak_force_n,
            max_normal_force_n=def_spec.max_normal_force_n,
        )
        intact = bool(geom_ok and damage < def_spec.damage_threshold and peak_force_n <= def_spec.max_normal_force_n)
    else:
        target_xy = np.array([0.22, -0.04], dtype=np.float64)
        final_dist = float(np.linalg.norm(fin_com[:2] - target_xy))
        geom_ok = bool(np.all(wipe_min_dists <= 0.065) and final_dist <= 0.095 and max_z >= 0.82)
        damage = compute_damage_score(
            peak_edge,
            def_spec.strain_onset_a,
            def_spec.strain_failure_b,
            peak_force_n=peak_force_n,
            max_normal_force_n=def_spec.max_normal_force_n,
        )
        intact = bool(geom_ok and damage < def_spec.damage_threshold and peak_force_n <= def_spec.max_normal_force_n)

    npz_arrays = {
        "qpos": qpos_hist,
        "action": act_hist,
        "ee_pos": ee_pos_hist,
        "ee_quat_wxyz": ee_quat_hist,
        "tactile_normal_force_n": tac_hist,
        "wrist_wrench": wrench_hist,
        "deformable_rest_points": rest_world,
        "deformable_points": pts_hist,
        "deformable_elements": elem_idx,
        "lid_angle_rad": lid_hist,
        "wipe_targets_min_dist_m": wipe_min_dists,
        "target_dist_m": np.array([final_dist], dtype=np.float64),
        "step_dt": np.array([0.01], dtype=np.float64),
    }
    metrics = {
        "task_id": task_id,
        "engine_id": "mujoco4",
        "mode": mode,
        "zero_friction": zero_friction,
        "max_com_z_m": round(max_z, 5),
        "final_com_xyz_m": [round(float(x), 5) for x in fin_com],
        "final_target_dist_m": round(final_dist, 5),
        "max_lid_angle_rad": round(float(lid_hist.max()), 4) if lid_adr >= 0 else None,
        "wipe_min_dists_m": [round(float(x), 5) for x in wipe_min_dists] if lid_adr < 0 else None,
        "polar_stretch_q95": round(peak_polar, 5) if lid_adr >= 0 else None,
        "naive_f_minus_i_q95": round(peak_naive, 5) if lid_adr >= 0 else None,
        "edge_stretch_q95": round(peak_edge, 5) if lid_adr < 0 else None,
        "peak_normal_force_n": round(peak_force_n, 4),
        "damage_score": round(damage, 5),
        "task_geometric_success": geom_ok,
        "intact": intact,
    }
    return save_episode_bundle(
        output_dir=output_dir,
        episode_id=episode_id,
        task_id=task_id,
        engine_id="mujoco4",
        mode=mode,
        seed=seed,
        zero_friction=zero_friction,
        replay_source=str(replay_npz or ""),
        npz_arrays=npz_arrays,
        summary_metrics=metrics,
    )


def main() -> int:
    apply_low_priority_process_policy()
    parser = argparse.ArgumentParser(description="MuJoCo 4.0.1 M3 Deformable Rollout Worker")
    parser.add_argument("--batch-json", required=True, help="JSON file containing list of rollout jobs.")
    parser.add_argument("--result-json", required=True, help="Output JSON summary path.")
    args = parser.parse_args()

    jobs = json.loads(Path(args.batch_json).read_text(encoding="utf-8"))["jobs"]
    results = [run_single_episode(job) for job in jobs]
    out_path = Path(args.result_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"engine_id": "mujoco4", "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
