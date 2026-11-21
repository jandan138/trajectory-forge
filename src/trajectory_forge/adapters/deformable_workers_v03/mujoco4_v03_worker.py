"""Isolated MuJoCo 4.0.1 CPU rollout worker for M3 v0.3 full-family deformable tasks (0% GPU)."""

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
    jitter_for_seed,
    save_episode_bundle,
)
from trajectory_forge.adapters.deformable_workers_v03.common_v03 import (  # noqa: E402
    evaluate_v03_task_outcome,
    load_v03_proxy_and_brief,
    orient_tets_positive,
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


def _fixture_xml_for_task(
    brief: Any,
    task_id: str,
    zero_friction: bool,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> str:
    blocks: list[str] = []

    def _xyz(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
        obj = brief.objects.get(name)
        return tuple(float(v) for v in obj.initial_pose.xyz) if obj else default

    if "baking_sheet_tray" in brief.objects:
        x, y, z = _xyz("baking_sheet_tray", (-0.18, -0.02, 0.755))
        blocks.append(
            f"""
    <body name="baking_sheet_tray" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="tray_plate" type="box" pos="0 0 0.003" size="0.095 0.085 0.003" friction="0.6 0.02 0.002"/>
      <geom name="liner_pad" type="box" pos="0 0 0.0055" size="0.084 0.074 0.001" friction="0.55 0.02 0.002"/>
      <geom name="rim_n" type="box" pos="0 0.081 0.007" size="0.095 0.004 0.007"/>
      <geom name="rim_s" type="box" pos="0 -0.081 0.007" size="0.095 0.004 0.007"/>
    </body>"""
        )

    if "tartlet_pastry_cup" in brief.objects:
        x, y, z = _xyz("tartlet_pastry_cup", (0.18, -0.02, 0.755))
        blocks.append(
            f"""
    <body name="tartlet_pastry_cup" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="saucer_base" type="cylinder" pos="0 0 0.0025" size="0.085 0.0025" friction="0.8 0.02 0.002"/>
      <geom name="tart_shell" type="cylinder" pos="0 0 0.013" size="0.048 0.008" friction="0.8 0.02 0.002"/>
      <geom name="pastry_cream" type="cylinder" pos="0 0 0.021" size="0.043 0.002" friction="0.85 0.02 0.002"/>
    </body>"""
        )

    if "smooth_folding_mat" in brief.objects:
        x, y, z = _xyz("smooth_folding_mat", (0.0, 0.02, 0.755))
        mat_f = "0.45 0.01 0.001" if task_id == "office_waffle_towel_corner_fold" else "0.14 0.005 0.001"
        blocks.append(
            f"""
    <body name="smooth_folding_mat" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="mat_slab" type="box" pos="0 0 0.001" size="0.24 0.18 0.001" friction="{mat_f}" solref="0.004 1"/>
    </body>"""
        )

    if "garment_folding_board" in brief.objects:
        x, y, z = _xyz("garment_folding_board", (0.0, 0.02, 0.755))
        blocks.append(
            f"""
    <body name="garment_folding_board" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="board_slab" type="box" pos="0.045 0 0.014" size="0.095 0.13 0.014" friction="0.65 0.02 0.002" solref="0.004 1"/>
    </body>"""
        )

    if "toner_cartridge_box" in brief.objects:
        x, y, z = _xyz("toner_cartridge_box", (-0.14, 0.02, 0.757))
        box_mu = 0.0 if zero_friction else 1.4
        box_cd = 1 if zero_friction else 3
        blocks.append(
            f"""
    <body name="toner_cartridge_box" pos="{x + dx:.5f} {y + dy:.5f} {z:.5f}">
      <freejoint name="toner_box_free"/>
      <inertial pos="0 0 0.022" mass="0.45" diaginertia="0.00038 0.00038 0.00048"/>
      <geom name="toner_box_hull" type="box" pos="0 0 0.021" size="0.032 0.032 0.021" condim="{box_cd}" friction="{box_mu:.4g} 0.02 0.002" solref="0.004 1"/>
      <geom name="toner_box_topcap" type="box" pos="0 0 0.043" size="0.0315 0.0315 0.001" condim="3" friction="0.15 0.005 0.001"/>
    </body>"""
        )

    if "delivery_caddy" in brief.objects:
        x, y, z = _xyz("delivery_caddy", (0.20, -0.02, 0.757))
        blocks.append(
            f"""
    <body name="delivery_caddy" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="caddy_base" type="box" pos="0 0 0.003" size="0.085 0.085 0.003" friction="0.8 0.02 0.002"/>
      <geom name="caddy_rim_n" type="box" pos="0 0.081 0.010" size="0.085 0.004 0.010"/>
      <geom name="caddy_rim_w" type="box" pos="-0.081 0 0.010" size="0.004 0.085 0.010"/>
      <geom name="caddy_rim_e" type="box" pos="0.081 0 0.010" size="0.004 0.085 0.010"/>
    </body>"""
        )

    if "cable_carrier_cradle" in brief.objects:
        x, y, z = _xyz("cable_carrier_cradle", (-0.16, -0.03, 0.755))
        blocks.append(
            f"""
    <body name="cable_carrier_cradle" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="cradle_base" type="box" pos="0 0 0.004" size="0.105 0.055 0.004"/>
      <geom name="cradle_l" type="box" pos="-0.068 0 0.011" size="0.012 0.036 0.007" friction="0.55 0.02 0.002"/>
      <geom name="cradle_r" type="box" pos="0.068 0 0.011" size="0.012 0.036 0.007" friction="0.55 0.02 0.002"/>
    </body>"""
        )

    if "smt_u_clip_block" in brief.objects:
        x, y, z = _xyz("smt_u_clip_block", (0.14, 0.04, 0.755))
        blocks.append(
            f"""
    <body name="smt_u_clip_block" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="clip_floor" type="box" pos="0 0 0.004" size="0.105 0.048 0.004" friction="0.8 0.02 0.002"/>
      <geom name="clip_wall_s" type="box" pos="0 -0.020 0.014" size="0.082 0.005 0.012" friction="0.45 0.02 0.002"/>
      <geom name="clip_wall_n" type="box" pos="0 0.020 0.014" size="0.082 0.005 0.012" friction="0.45 0.02 0.002"/>
    </body>"""
        )

    if "pcba_header_boss" in brief.objects:
        x, y, z = _xyz("pcba_header_boss", (0.15, -0.02, 0.755))
        blocks.append(
            f"""
    <body name="pcba_header_boss" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="pcba_base" type="box" pos="0 0 0.003" size="0.075 0.075 0.003" friction="0.85 0.02 0.002"/>
      <geom name="header_shroud" type="box" pos="0 0 0.007" size="0.026 0.026 0.002" friction="0.85 0.02 0.002"/>
      <geom name="guide_rail_w" type="box" pos="-0.042 0 0.009" size="0.004 0.036 0.005"/>
      <geom name="guide_rail_e" type="box" pos="0.042 0 0.009" size="0.004 0.036 0.005"/>
    </body>"""
        )

    if "tube_bench_cradle" in brief.objects:
        x, y, z = _xyz("tube_bench_cradle", (-0.17, -0.02, 0.755))
        blocks.append(
            f"""
    <body name="tube_bench_cradle" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="tube_cradle_base" type="box" pos="0 0 0.004" size="0.105 0.055 0.004"/>
      <geom name="tube_cradle_l" type="box" pos="-0.068 0 0.011" size="0.014 0.032 0.007" friction="0.55 0.02 0.002"/>
      <geom name="tube_cradle_r" type="box" pos="0.068 0 0.011" size="0.014 0.032 0.007" friction="0.55 0.02 0.002"/>
    </body>"""
        )

    if "manifold_fork_saddle" in brief.objects:
        x, y, z = _xyz("manifold_fork_saddle", (0.16, 0.03, 0.755))
        blocks.append(
            f"""
    <body name="manifold_fork_saddle" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="manifold_base" type="box" pos="0 0 0.004" size="0.105 0.055 0.004" friction="0.8 0.02 0.002"/>
      <geom name="fork_s" type="box" pos="0 -0.022 0.015" size="0.085 0.005 0.013" friction="0.5 0.02 0.002"/>
      <geom name="fork_n" type="box" pos="0 0.022 0.015" size="0.085 0.005 0.013" friction="0.5 0.02 0.002"/>
    </body>"""
        )

    if "bulb_collar_holder" in brief.objects:
        x, y, z = _xyz("bulb_collar_holder", (0.15, -0.02, 0.755))
        blocks.append(
            f"""
    <body name="bulb_collar_holder" pos="{x:.5f} {y:.5f} {z:.5f}">
      <geom name="stand_base" type="box" pos="0 0 0.003" size="0.065 0.065 0.003" friction="0.85 0.02 0.002"/>
      <geom name="collar_n" type="box" pos="0 0.034 0.012" size="0.018 0.004 0.006"/>
      <geom name="collar_s" type="box" pos="0 -0.034 0.012" size="0.018 0.004 0.006"/>
      <geom name="collar_w" type="box" pos="-0.034 0 0.012" size="0.004 0.018 0.006"/>
      <geom name="collar_e" type="box" pos="0.034 0 0.012" size="0.004 0.018 0.006"/>
    </body>"""
        )

    if task_id.startswith("smt_") and "ESDTray" in brief.objects:
        ex, ey, ez = _xyz("ESDTray", (-0.42, -0.18, 0.757))
        blocks.append(
            f"""
    <body name="esd_tray_prop" pos="{ex:.5f} {ey:.5f} {ez:.5f}">
      <geom name="esd_tray_obstacle" type="box" pos="0 0 0.024" size="0.115 0.085 0.024" friction="0.7 0.02 0.002"/>
    </body>"""
        )

    return "\n".join(blocks)


def _build_scene_xml(
    *,
    pkg_dir: Path,
    brief: Any,
    proxy: dict[str, np.ndarray],
    m1_manifest: dict[str, Any],
    task_id: str,
    zero_friction: bool,
    seed: int,
) -> tuple[str, np.ndarray, np.ndarray]:
    dx, dy, _ = jitter_for_seed(seed)
    condim = 1 if zero_friction else 3

    def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
    spec = m1_manifest["source"]
    family = str(spec["structural_family"])

    if zero_friction:
        finger_mu = 0.0
        def_mu = 0.0
    elif task_id == "office_dust_cover_peel_retrieve":
        finger_mu = 3.6
        def_mu = 0.35
    elif task_id == "office_waffle_towel_corner_fold":
        finger_mu = 3.4
        def_mu = 3.2
    elif family == "shell":
        finger_mu = 2.4
        def_mu = 2.2
    else:
        finger_mu = 1.45
        def_mu = 1.45
    def_f = f"{def_mu:.4g} 0.02 0.002"
    meshes_dir, wb_inner = _patch_panda_xml(pkg_dir, finger_mu, condim=condim)

    sx, sy, sz = def_obj.initial_pose.xyz
    # Ensure zero initial interpenetration above liner/cradle/mat
    z_spawn = float(sz) + (0.001 if family == "soft_solid" else 0.0)
    sx += dx
    sy += dy

    pts = np.asarray(proxy["rest_points"], dtype=np.float64)
    p_str = "  ".join(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" for p in pts)
    mass_kg = float(spec["mass_kg"])
    young = float(spec["youngs_modulus_pa"])
    poisson = float(spec["poissons_ratio"])

    if family in ("soft_solid", "rod_cable"):
        dim = 3
        elem_arr = orient_tets_positive(pts, np.asarray(proxy["elements"], dtype=np.int64))
        e_str = "  ".join(f"{t[0]} {t[1]} {t[2]} {t[3]}" for t in elem_arr)
        el_tag = f'<elasticity young="{young:.1f}" poisson="{poisson:.3f}" damping="0.06"/>'
        self_col = "none"
        flex_rad = 0.0018 if family == "rod_cable" else 0.0022
        metrics_elem = elem_arr
    else:
        dim = 2
        tris = np.asarray(proxy["triangles"], dtype=np.int64)
        e_str = "  ".join(f"{t[0]} {t[1]} {t[2]}" for t in tris)
        el2d = "both"
        if family == "cloth_sheet":
            # Prevent cloth vertices crossing through one another during X-axis folding
            self_col = "auto" if task_id == "office_waffle_towel_corner_fold" else "none"
            thick = 0.0065 if task_id == "office_waffle_towel_corner_fold" else 0.0055
            damp = 0.04
            flex_rad = 0.0025
        else:
            self_col = "none"
            thick = 0.0042 if task_id == "lab_rubber_bulb_squeeze_seat" else 0.0075
            damp = 0.05
            flex_rad = 0.0025
        el_tag = f'<elasticity young="{young:.1f}" poisson="{poisson:.3f}" thickness="{thick}" damping="{damp}" elastic2d="{el2d}"/>'
        metrics_elem = np.asarray(proxy["edges"], dtype=np.int64)

    flex_xml = f"""
    <flexcomp name="{def_obj.object_id}" type="direct" dim="{dim}" pos="{sx:.5f} {sy:.5f} {z_spawn:.5f}" point="{p_str}" element="{e_str}" mass="{mass_kg:.5f}" radius="{flex_rad}">
      <contact friction="{def_f}" condim="{condim}" selfcollide="{self_col}" solref="0.004 1"/>
      <edge equality="false"/>
      {el_tag}
    </flexcomp>"""

    fixtures_xml = _fixture_xml_for_task(brief, task_id, zero_friction, dx=dx, dy=dy)
    f_kp = 2600 if task_id == "lab_rubber_bulb_squeeze_seat" else 3200
    f_kv = 70 if task_id == "lab_rubber_bulb_squeeze_seat" else 80
    table_xml = '<geom name="table_top" type="box" pos="0 0 0.745" size="1.0 0.45 0.01" friction="0.45 0.02 0.002" solref="0.004 1"/>'

    full_xml = f"""<mujoco model="{task_id}_v03_mj4">
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
    {fixtures_xml}
    {flex_xml}
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
    <position name="act_f1" joint="panda_finger_joint1" kp="{f_kp}" kv="{f_kv}" ctrlrange="0.0 0.04"/>
    <position name="act_f2" joint="panda_finger_joint2" kp="{f_kp}" kv="{f_kv}" ctrlrange="0.0 0.04"/>
  </actuator>
</mujoco>"""
    return full_xml, pts, metrics_elem


def _extract_finger_forces(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    pad_l: int,
    pad_r: int,
    include_rigid_geoms: tuple[int, ...] = (),
    obstacle_geom_id: int = -1,
) -> tuple[float, float, np.ndarray, bool]:
    vec_l = np.zeros(3, dtype=np.float64)
    vec_r = np.zeros(3, dtype=np.float64)
    f6 = np.zeros(6, dtype=np.float64)
    touched_obstacle = False
    for ci in range(d.ncon):
        c = d.contact[ci]
        if c.geom1 in (pad_l, pad_r) or c.geom2 in (pad_l, pad_r):
            other = c.geom2 if c.geom1 in (pad_l, pad_r) else c.geom1
            if obstacle_geom_id >= 0 and other == obstacle_geom_id:
                touched_obstacle = True
            if not (other < 0 or other in include_rigid_geoms):
                continue
            mujoco.mj_contactForce(m, d, ci, f6)
            f_local = np.array([min(150.0, abs(float(f6[0]))), float(f6[1]), float(f6[2])], dtype=np.float64)
            f_world = np.asarray(c.frame, dtype=np.float64).reshape(3, 3).T @ f_local
            if c.geom1 == pad_l or c.geom2 == pad_l:
                vec_l += f_world
            if c.geom1 == pad_r or c.geom2 == pad_r:
                vec_r += f_world
        elif obstacle_geom_id >= 0 and (c.geom1 == obstacle_geom_id or c.geom2 == obstacle_geom_id):
            other = c.geom2 if c.geom1 == obstacle_geom_id else c.geom1
            if other < 0:
                touched_obstacle = True
    fn_l = min(160.0, float(np.linalg.norm(vec_l)))
    fn_r = min(160.0, float(np.linalg.norm(vec_r)))
    fz = min(160.0, abs(float(vec_l[2])) + abs(float(vec_r[2])))
    wrist_w = np.array([0.0, 0.0, -(fn_l + fn_r) * 0.3 - fz, 0.0, 0.0, 0.0], dtype=np.float64)
    return fn_l, fn_r, wrist_w, touched_obstacle


def run_single_episode(job: dict[str, Any]) -> dict[str, Any]:
    pkg_dir = Path(job["package_dir"])
    if (pkg_dir / "scenario_package" / "assets").is_dir():
        pkg_dir = pkg_dir / "scenario_package"
    task_id = str(job["task_id"])
    mode = str(job["mode"])
    seed = int(job.get("seed", 0))
    zero_friction = bool(job.get("zero_friction", False))
    replay_npz = job.get("replay_npz")
    skill_override = job.get("skill_params_override")
    output_dir = Path(job["output_dir"])
    episode_id = str(job["episode_id"])

    brief, ctrl, proxy, m1_manifest = load_v03_proxy_and_brief(
        pkg_dir,
        task_id,
        mode,
        skill_params_override=skill_override,
    )
    replay_actions = None
    if replay_npz:
        with np.load(replay_npz, allow_pickle=False) as z:
            replay_actions = np.asarray(z["action"], dtype=np.float64)

    xml_str, _, metrics_elem = _build_scene_xml(
        pkg_dir=pkg_dir,
        brief=brief,
        proxy=proxy,
        m1_manifest=m1_manifest,
        task_id=task_id,
        zero_friction=zero_friction,
        seed=seed,
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

    box_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "toner_cartridge_box")
    obstacle_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "esd_tray_obstacle")
    contact_rigid_names = (
        "toner_box_hull",
        "header_shroud",
        "esd_tray_obstacle",
        "clip_floor",
        "manifold_base",
        "stand_base",
        "board_slab",
        "mat_slab",
        "table_top",
    )
    contact_rigid_geoms = tuple(
        gid
        for nm in contact_rigid_names
        if (gid := mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)) >= 0
    )

    first_act = (
        replay_actions[0]
        if replay_actions is not None
        else ctrl.step(
            {"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(9), "tactile_normal_force_n": np.zeros(2), "wrist_wrench": np.zeros(6)}
        )
    )
    for i, qi in enumerate(q_idx):
        d.qpos[qi] = first_act[i]
    mujoco.mj_forward(m, d)
    rest_world = d.flexvert_xpos.copy().astype(np.float64)

    # Settle cloth before k=0 so dust cover drapes over box and waffle towel overhangs board edge
    if task_id == "office_dust_cover_peel_retrieve":
        d.ctrl[:] = first_act
        for _ in range(175):  # 35 control equiv steps = 0.35 s
            mujoco.mj_step(m, d)
    elif task_id == "office_waffle_towel_corner_fold":
        d.ctrl[:] = first_act
        for _ in range(50):
            mujoco.mj_step(m, d)

    total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)
    substeps = 5

    qpos_hist = np.zeros((total_steps, 9), dtype=np.float64)
    act_hist = np.zeros((total_steps, 9), dtype=np.float64)
    ee_pos_hist = np.zeros((total_steps, 3), dtype=np.float64)
    ee_quat_hist = np.zeros((total_steps, 4), dtype=np.float64)
    tac_hist = np.zeros((total_steps, 2), dtype=np.float64)
    wrench_hist = np.zeros((total_steps, 6), dtype=np.float64)
    pts_hist = np.zeros((total_steps, len(rest_world), 3), dtype=np.float64)
    rigid_pose_hist = np.zeros((total_steps, 7), dtype=np.float64) if box_body_id >= 0 else None

    pad_touched_obstacle = False
    for k in range(total_steps):
        q_obs = np.array([d.qpos[qi] for qi in q_idx], dtype=np.float64)
        ee_p = d.site_xpos[ee_site].copy()
        fn_l, fn_r, w_obs, touched_obs = _extract_finger_forces(
            m, d, pad_l, pad_r, contact_rigid_geoms, obstacle_geom_id
        )
        pad_touched_obstacle = pad_touched_obstacle or touched_obs

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
        mat = d.site_xmat[ee_site].reshape(3, 3)
        q_wxyz = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q_wxyz, mat.reshape(9))
        ee_quat_hist[k] = q_wxyz
        tac_hist[k] = np.array([fn_l, fn_r], dtype=np.float64)
        wrench_hist[k] = w_obs
        pts_hist[k] = cur_pts
        if rigid_pose_hist is not None and box_body_id >= 0:
            rigid_pose_hist[k, :3] = d.xpos[box_body_id]
            rigid_pose_hist[k, 3:] = d.xquat[box_body_id]

    metrics, final_dist = evaluate_v03_task_outcome(
        task_id=task_id,
        engine_id="mujoco4",
        mode=mode,
        zero_friction=zero_friction,
        brief=brief,
        rest_world=rest_world,
        pts_hist=pts_hist,
        elems_or_edges=metrics_elem,
        tac_hist=tac_hist,
        rigid_target_pose_hist=rigid_pose_hist,
        pad_touches_obstacle=pad_touched_obstacle,
    )

    npz_arrays: dict[str, np.ndarray] = {
        "qpos": qpos_hist,
        "action": act_hist,
        "ee_pos": ee_pos_hist,
        "ee_quat_wxyz": ee_quat_hist,
        "tactile_normal_force_n": tac_hist,
        "wrist_wrench": wrench_hist,
        "deformable_rest_points": rest_world,
        "deformable_points": pts_hist,
        "deformable_elements": metrics_elem,
        "target_dist_m": np.array([final_dist], dtype=np.float64),
        "step_dt": np.array([0.01], dtype=np.float64),
    }
    if rigid_pose_hist is not None:
        npz_arrays["rigid_target_pose"] = rigid_pose_hist

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
    parser = argparse.ArgumentParser(description="MuJoCo 4.0.1 M3 v0.3 Deformable Rollout Worker")
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
