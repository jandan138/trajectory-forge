"""Isolated MuJoCo 4.0.1 CPU 18-DOF dual-arm zero-cheat rollout worker for distillation_v04 (0% GPU)."""

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
)
from trajectory_forge.adapters.deformable_workers_v03.common_v03 import orient_tets_positive  # noqa: E402
from trajectory_forge.adapters.distillation_workers_v04.common_v04 import (  # noqa: E402
    V04_TASKS,
    bimanual_pad_world_poses,
    build_task_rigid_bodies_spec,
    evaluate_v04_task_outcome,
    load_v04_task_context,
    save_v04_episode_bundle,
)


# Physical time dilation: 14 x 2 ms MuJoCo steps per 18-DOF ctrl tick (28 ms, ~35.7 Hz) -> realistic <=0.5 m/s pad speeds
SUBSTEPS_PER_CTRL = 14
CTRL_DT = 0.002 * SUBSTEPS_PER_CTRL

# Optional offline diagnostics (stderr only, never persisted): V04_DEBUG_BODIES="alcohol_lamp,iron_ring_gauze"
_DEBUG_BODIES = tuple(b for b in (__import__("os").environ.get("V04_DEBUG_BODIES", "").split(",")) if b)
_DEBUG_EVERY = int(__import__("os").environ.get("V04_DEBUG_EVERY", "5"))


def _debug_dump_contacts(m: Any, d: Any, k: int, bodies: tuple[str, ...]) -> None:
    if k % _DEBUG_EVERY:
        return
    f6 = np.zeros(6, dtype=np.float64)
    for bname in bodies:
        try:
            bid = int(m.body(bname).id)
        except Exception:
            continue
        rows = []
        for ci in range(d.ncon):
            c = d.contact[ci]
            b1 = int(m.geom_bodyid[c.geom1]) if c.geom1 >= 0 else -1
            b2 = int(m.geom_bodyid[c.geom2]) if c.geom2 >= 0 else -1
            if bid not in (b1, b2):
                continue
            other = c.geom2 if b1 == bid else c.geom1
            oname = m.geom(other).name if other >= 0 else "flex"
            if not oname:
                oname = f"g{other}:{m.body(int(m.geom_bodyid[other])).name}"
            mujoco.mj_contactForce(m, d, ci, f6)
            rows.append(f"{oname}(d={c.dist*1000:.1f}mm,F={f6[0]:.1f})")
        p = d.xpos[bid]
        print(f"[dbg k={k}] {bname} p=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) contacts={rows[:8]}", file=sys.stderr)


def _quat_to_euler_xyz(wxyz: list[float]) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in wxyz]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _patch_dual_panda_xml(
    pkg_dir: Path,
    finger_mu: float,
    condim: int,
    *,
    finger_kp_L: float = 1200.0,
    finger_kp_R: float = 1200.0,
    finger_kv: float = 60.0,
    pad_hy: float = 0.006,
    v_groove_R: bool = False,
) -> tuple[str, str, str]:
    meshes_dir = (pkg_dir / "assets" / "franka_panda" / "meshes").resolve().as_posix()
    panda_raw = (pkg_dir / "assets" / "franka_panda" / "panda.xml").read_text(encoding="utf-8")
    wb_raw = panda_raw.split("<worldbody>")[1].split("</worldbody>")[0]
    wb_raw = re.sub(r'<body (name="panda_[^"]+")', r'<body \1 gravcomp="1"', wb_raw)

    f_str = f"{finger_mu:.6g} 0.02 0.002"
    # Elliptic 4-D pad contact: 36x36 mm pad patch resists rod pendulum swing about the finger axis (torsional mu*Fn*0.02 m)
    pad_condim = 1 if condim == 1 else 4
    wb_raw = wb_raw.replace(
        'pos="0 0.008 0.036" size="0.010 0.008 0.018" rgba="0.72 0.74 0.76 1" solref="0.004 1"',
        f'pos="0 {pad_hy:.4g} 0.036" size="0.018 {pad_hy:.4g} 0.018" rgba="0.72 0.74 0.76 1" condim="{pad_condim}" friction="{f_str}" solref="0.004 1"',
    )
    wb_raw = wb_raw.replace(
        'pos="0 -0.008 0.036" size="0.010 0.008 0.018" rgba="0.72 0.74 0.76 1" solref="0.004 1"',
        f'pos="0 -{pad_hy:.4g} 0.036" size="0.018 {pad_hy:.4g} 0.018" rgba="0.72 0.74 0.76 1" condim="{pad_condim}" friction="{f_str}" solref="0.004 1"',
    )

    arm_blocks: list[str] = []
    act_blocks: list[str] = []
    for pfx, base_xyz in (("L", "-0.32 -0.46 0.755"), ("R", "0.32 -0.46 0.755")):
        f_kp = finger_kp_L if pfx == "L" else finger_kp_R
        wb_arm = wb_raw.replace(
            '<body name="panda_link0" gravcomp="1" pos="0 0 0">',
            f'<body name="{pfx}_panda_link0" gravcomp="1" childclass="panda" pos="{base_xyz}" euler="0 0 1.57079632679">',
        )
        wb_arm = re.sub(r'name="panda_', f'name="{pfx}_panda_', wb_arm)
        wb_arm = wb_arm.replace('name="left_finger_pad"', f'name="{pfx}_left_finger_pad"')
        wb_arm = wb_arm.replace('name="right_finger_pad"', f'name="{pfx}_right_finger_pad"')
        if pfx == "R" and v_groove_R:
            # Prismatic V-jaw chamfer lips along hand_X at z=0.036+/-0.0105 keep cylindrical tubing bore circular without crushing
            wb_arm = wb_arm.replace(
                f'name="R_left_finger_pad" type="box" pos="0 0.009 0.036" size="0.018 0.009 0.018" rgba="0.72 0.74 0.76 1" condim="{pad_condim}" friction="{f_str}" solref="0.004 1"/>',
                f'name="R_left_finger_pad" type="box" pos="0 0.009 0.036" size="0.018 0.009 0.018" rgba="0.72 0.74 0.76 1" condim="{pad_condim}" friction="{f_str}" solref="0.004 1"/>\n'
                f'                    <geom name="R_left_vlip_lo" type="box" pos="0 0.0062 0.0255" size="0.018 0.009 0.0025" rgba="0.72 0.74 0.76 1" condim="{pad_condim}" friction="{f_str}" solref="0.004 1"/>\n'
                f'                    <geom name="R_left_vlip_hi" type="box" pos="0 0.0062 0.0465" size="0.018 0.009 0.0025" rgba="0.72 0.74 0.76 1" condim="{pad_condim}" friction="{f_str}" solref="0.004 1"/>',
            )
            wb_arm = wb_arm.replace(
                f'name="R_right_finger_pad" type="box" pos="0 -0.009 0.036" size="0.018 0.009 0.018" rgba="0.72 0.74 0.76 1" condim="{pad_condim}" friction="{f_str}" solref="0.004 1"/>',
                f'name="R_right_finger_pad" type="box" pos="0 -0.009 0.036" size="0.018 0.009 0.018" rgba="0.72 0.74 0.76 1" condim="{pad_condim}" friction="{f_str}" solref="0.004 1"/>\n'
                f'                    <geom name="R_right_vlip_lo" type="box" pos="0 -0.0062 0.0255" size="0.018 0.009 0.0025" rgba="0.72 0.74 0.76 1" condim="{pad_condim}" friction="{f_str}" solref="0.004 1"/>\n'
                f'                    <geom name="R_right_vlip_hi" type="box" pos="0 -0.0062 0.0465" size="0.018 0.009 0.0025" rgba="0.72 0.74 0.76 1" condim="{pad_condim}" friction="{f_str}" solref="0.004 1"/>',
            )
        arm_blocks.append(wb_arm)
        act_blocks.append(
            f"""
    <position name="{pfx}_act_j1" joint="{pfx}_panda_joint1" kp="3600" kv="120" ctrlrange="-2.9671 2.9671"/>
    <position name="{pfx}_act_j2" joint="{pfx}_panda_joint2" kp="3600" kv="120" ctrlrange="-1.8326 1.8326"/>
    <position name="{pfx}_act_j3" joint="{pfx}_panda_joint3" kp="3600" kv="120" ctrlrange="-2.9671 2.9671"/>
    <position name="{pfx}_act_j4" joint="{pfx}_panda_joint4" kp="3600" kv="120" ctrlrange="-3.1416 0.0"/>
    <position name="{pfx}_act_j5" joint="{pfx}_panda_joint5" kp="1600" kv="60" ctrlrange="-2.9671 2.9671"/>
    <position name="{pfx}_act_j6" joint="{pfx}_panda_joint6" kp="1600" kv="60" ctrlrange="-0.0873 3.8223"/>
    <position name="{pfx}_act_j7" joint="{pfx}_panda_joint7" kp="1100" kv="42" ctrlrange="-2.9671 2.9671"/>
    <position name="{pfx}_act_f1" joint="{pfx}_panda_finger_joint1" kp="{f_kp:.6g}" kv="{finger_kv:.6g}" ctrlrange="0.0 0.04"/>
    <position name="{pfx}_act_f2" joint="{pfx}_panda_finger_joint2" kp="{f_kp:.6g}" kv="{finger_kv:.6g}" ctrlrange="0.0 0.04"/>"""
        )
    return meshes_dir, "\n".join(arm_blocks), "\n".join(act_blocks)


def _rigid_bodies_xml(rigid_specs: list[dict[str, Any]], zero_friction: bool, *, assembled_hoses: bool = False) -> str:
    glass_mu = 0.0 if zero_friction else 2.2
    barb_mu = 0.0 if zero_friction else (2.2 if assembled_hoses else 4.5)
    condim = 1 if zero_friction else 3
    lines: list[str] = []
    for rb in rigid_specs:
        cid = rb["id"]
        x, y, z = rb["xyz"]
        qw, qx, qy, qz = rb["wxyz"]
        mass = float(rb["mass_kg"])
        cols = rb["colliders"]
        mu = 0.7 if not rb["dynamic"] else glass_mu
        cd = 3 if not rb["dynamic"] else condim
        geoms: list[str] = []
        gi = 0
        for b in cols.get("boxes", []):
            bx, by, bz = b["pos"]
            sx, sy, sz = b["half_size"]
            bw = b.get("wxyz", [1, 0, 0, 0])
            geoms.append(
                f'      <geom name="{cid}_b{gi}" type="box" pos="{bx:.5f} {by:.5f} {bz:.5f}" '
                f'quat="{bw[0]:.6f} {bw[1]:.6f} {bw[2]:.6f} {bw[3]:.6f}" '
                f'size="{sx:.5f} {sy:.5f} {sz:.5f}" condim="{cd}" friction="{mu:.4g} 0.02 0.002" solref="0.004 1"/>'
            )
            gi += 1
        for cy in cols.get("cylinders", []):
            cx, ccy, cz = cy["pos"]
            r, hh = float(cy["radius"]), float(cy["half_height"])
            ax = cy.get("axis", "Z")
            quat_str = "0.707107 0 0.707107 0" if ax == "X" else "1 0 0 0"
            geoms.append(
                f'      <geom name="{cid}_c{gi}" type="cylinder" pos="{cx:.5f} {ccy:.5f} {cz:.5f}" '
                f'quat="{quat_str}" size="{r:.5f} {hh:.5f}" condim="{cd}" friction="{mu:.4g} 0.02 0.002" solref="0.004 1"/>'
            )
            gi += 1
        for wg in cols.get("socket_wedges", []):
            wx, wy, wz = wg["pos"]
            sx, sy, sz = wg["half_size"]
            ww = wg["wxyz"]
            geoms.append(
                f'      <geom name="{cid}_w{gi}" type="box" pos="{wx:.5f} {wy:.5f} {wz:.5f}" '
                f'quat="{ww[0]:.6f} {ww[1]:.6f} {ww[2]:.6f} {ww[3]:.6f}" '
                f'size="{sx:.5f} {sy:.5f} {sz:.5f}" condim="{cd}" friction="{mu:.4g} 0.02 0.002" solref="0.004 1"/>'
            )
            gi += 1
        for barb in cols.get("barbs", []):
            r0 = np.asarray(barb["root_local"], dtype=np.float64)
            ax = np.asarray(barb["axis_local"], dtype=np.float64)
            for sp in barb.get("spheres", [{"s_m": 0.010, "radius": 0.0046}, {"s_m": 0.022, "radius": 0.0048}, {"s_m": 0.032, "radius": 0.0040}]):
                p = r0 + float(sp["s_m"]) * ax
                geoms.append(
                    f'      <geom name="{cid}_barb{gi}" type="sphere" pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" '
                    f'size="{float(sp["radius"]):.5f}" condim="{condim}" friction="{barb_mu:.4g} 0.02 0.002" solref="0.004 1"/>'
                )
                gi += 1

        fj = f'      <freejoint name="{cid}_free"/>\n      <inertial pos="0 0 0.02" mass="{mass:.4f}" diaginertia="0.0005 0.0005 0.0005"/>' if rb["dynamic"] else ""
        # Give Liebig condenser high passive rotational/linear damping when sitting in stand cradle
        if not geoms:
            continue
        lines.append(
            f'    <body name="{cid}" pos="{x:.5f} {y:.5f} {z:.5f}" quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">\n{fj}\n'
            + "\n".join(geoms)
            + "\n    </body>"
        )
    return "\n".join(lines)


def _pad_forces_4ch(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    pad_ids: tuple[int, int, int, int],
    include_geom_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctrl_vecs = [np.zeros(3, dtype=np.float64) for _ in range(4)]
    def_vecs = [np.zeros(3, dtype=np.float64) for _ in range(4)]
    f6 = np.zeros(6, dtype=np.float64)
    for ci in range(d.ncon):
        c = d.contact[ci]
        for idx, pid in enumerate(pad_ids):
            if c.geom1 == pid or c.geom2 == pid:
                other = int(c.geom2 if c.geom1 == pid else c.geom1)
                if include_geom_ids is not None and other >= 0 and other not in include_geom_ids:
                    continue
                mujoco.mj_contactForce(m, d, ci, f6)
                fl = np.array([min(150.0, abs(float(f6[0]))), float(f6[1]), float(f6[2])], dtype=np.float64)
                fw = np.asarray(c.frame, dtype=np.float64).reshape(3, 3).T @ fl
                ctrl_vecs[idx] += fw
                if other < 0:
                    def_vecs[idx] += fw
    fn_ctrl = np.array([min(155.0, float(np.linalg.norm(v))) for v in ctrl_vecs], dtype=np.float64)
    fn_def = np.array([min(155.0, float(np.linalg.norm(v))) for v in def_vecs], dtype=np.float64)
    w12 = np.zeros(12, dtype=np.float64)
    w12[2] = -(fn_ctrl[0] + fn_ctrl[1]) * 0.5
    w12[8] = -(fn_ctrl[2] + fn_ctrl[3]) * 0.5
    return fn_ctrl, fn_def, w12


def run_single_episode(job: dict[str, Any]) -> dict[str, Any]:
    pkg_dir = Path(job["package_dir"])
    task_id = str(job["task_id"])
    mode = str(job["mode"])
    seed = int(job.get("seed", 0))
    zero_friction = bool(job.get("zero_friction", False))
    replay_npz = job.get("replay_npz")
    skill_override = job.get("skill_params_override")
    output_dir = Path(job["output_dir"])
    episode_id = str(job["episode_id"])

    ctx = load_v04_task_context(pkg_dir, task_id, mode, skill_params_override=skill_override)
    brief = ctx["brief"]
    ctrl = ctx["controller"]
    proxy = ctx["proxy"]
    spec = ctx["m1_manifest"]["source"]
    def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")

    replay_actions = None
    if replay_npz:
        with np.load(replay_npz, allow_pickle=False) as z:
            replay_actions = np.asarray(z["action"], dtype=np.float64)

    dx, dy, _ = jitter_for_seed(seed)
    condim = 1 if zero_friction else 3
    finger_mu = 0.0 if zero_friction else 3.2
    def_mu = 0.0 if zero_friction else 2.6
    finger_kp_L = 3200.0 if mode == "crush" else 1200.0
    finger_kp_R = 3200.0 if mode == "crush" else (2400.0 if spec["structural_family"] == "rod_cable" else 1200.0)
    finger_kv = 80.0 if (mode == "crush" or spec["structural_family"] == "rod_cable") else 60.0

    meshes_dir, arms_xml, acts_xml = _patch_dual_panda_xml(
        ctx["package_root"],
        finger_mu,
        condim,
        finger_kp_L=finger_kp_L,
        finger_kp_R=finger_kp_R,
        finger_kv=finger_kv,
        pad_hy=0.009 if spec["structural_family"] == "rod_cable" else 0.006,
        v_groove_R=(task_id == "distill_04_bimanual_hose_barb_couple"),
    )
    rigid_specs = build_task_rigid_bodies_spec(ctx, task_id, dx=dx, dy=dy)
    for rb in rigid_specs:
        if rb["id"] == "bench_water_ports" and ctx["task_spec"].get("hose_mode") != "assembled":
            rb["colliders"] = {
                "boxes": [
                    {"pos": [0.0, -0.15, 0.004], "half_size": [0.020, 0.008, 0.004], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.090, 0.089, 0.0300], "half_size": [0.018, 0.006, 0.0300], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.073, 0.089, 0.0670], "half_size": [0.003, 0.006, 0.0070], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.107, 0.089, 0.0670], "half_size": [0.003, 0.006, 0.0070], "wxyz": [1, 0, 0, 0]},
                    {"pos": [-0.056, 0.089, 0.0425], "half_size": [0.018, 0.006, 0.0425], "wxyz": [1, 0, 0, 0]},
                    {"pos": [-0.073, 0.089, 0.0920], "half_size": [0.003, 0.006, 0.0070], "wxyz": [1, 0, 0, 0]},
                    {"pos": [-0.039, 0.089, 0.0920], "half_size": [0.003, 0.006, 0.0070], "wxyz": [1, 0, 0, 0]},
                ],
                "cylinders": [],
                "socket_wedges": [],
            }
        elif (not rb["dynamic"]) and rb["id"] == "distillation_head" and task_id == "distill_03_thermometer_septum_align":
            rb["colliders"] = {
                "boxes": list(rb["colliders"].get("boxes", []))[:12],
                "cylinders": [],
                "socket_wedges": [],
            }
    rigids_xml = _rigid_bodies_xml(rigid_specs, zero_friction, assembled_hoses=(ctx["task_spec"].get("hose_mode") == "assembled"))

    pts_key = "assembled_rest_points" if (ctx["task_spec"].get("hose_mode") == "assembled" and "assembled_rest_points" in proxy) else "rest_points"
    pts = np.asarray(proxy[pts_key], dtype=np.float64)
    tets = orient_tets_positive(pts, np.asarray(proxy["elements"], dtype=np.int64))
    p_str = "  ".join(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" for p in pts)
    e_str = "  ".join(f"{t[0]} {t[1]} {t[2]} {t[3]}" for t in tets)

    sx, sy, sz = def_obj.initial_pose.xyz
    z_spawn = float(sz) + (0.001 if spec["structural_family"] == "soft_solid" else 0.0)
    use_jit = (task_id != "distill_06_reverse_order_disassembly") and (ctx["task_spec"].get("hose_mode") is None)
    px, py = float(sx) + (dx if use_jit else 0.0), float(sy) + (dy if use_jit else 0.0)

    young_eff = (2.6 if spec["structural_family"] == "rod_cable" else 1.0) * float(spec["youngs_modulus_pa"])
    mass_eff = (1.5 if spec["structural_family"] == "rod_cable" else 1.0) * float(spec["mass_kg"])
    flex_xml = f"""
    <flexcomp name="{def_obj.object_id}" type="direct" dim="3" pos="{px:.5f} {py:.5f} {z_spawn:.5f}" point="{p_str}" element="{e_str}" mass="{mass_eff:.5f}" radius="0.0018">
      <contact friction="{def_mu:.4g} 0.02 0.002" condim="{condim}" selfcollide="none" solref="0.004 1"/>
      <edge equality="false"/>
      <elasticity young="{young_eff:.1f}" poisson="{float(spec['poissons_ratio']):.3f}" damping="0.08"/>
    </flexcomp>"""

    imp_str = ' impratio="27"' if task_id == "distill_04_bimanual_hose_barb_couple" else ""
    tbl_f = "0 0 0" if zero_friction else "0.35 0.01 0.001"
    full_xml = f"""<mujoco model="{task_id}_v04_mj4">
  <compiler angle="radian" meshdir="{meshes_dir}" autolimits="true"/>
  <option timestep="0.002" integrator="discrete" iterations="55"{imp_str}><flag speculative="enable"/></option>
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
      <joint armature="0.08" damping="1.2"/>
      <geom friction="0.7 0.02 0.002"/>
    </default>
  </default>
  <worldbody>
    <geom name="table_top" type="box" pos="0 0 0.745" size="1.0 0.45 0.01" priority="1" condim="{condim}" friction="{tbl_f}" solref="0.004 1"/>
    {arms_xml}
    {rigids_xml}
    {flex_xml}
  </worldbody>
  <actuator>
    {acts_xml}
  </actuator>
</mujoco>"""

    m = mujoco.MjModel.from_xml_string(full_xml)
    d = mujoco.MjData(m)

    j_names = [
        f"{pfx}_panda_joint{i}" for pfx in ("L", "R") for i in range(1, 8)
    ]
    # Note: assemble L_0..8 then R_0..8 order matching q18
    q_names_18 = (
        [f"L_panda_joint{i}" for i in range(1, 8)] + ["L_panda_finger_joint1", "L_panda_finger_joint2"]
        + [f"R_panda_joint{i}" for i in range(1, 8)] + ["R_panda_finger_joint1", "R_panda_finger_joint2"]
    )
    _ = j_names
    q_idx = [int(m.jnt_qposadr[m.joint(jn).id]) for jn in q_names_18]
    a_names_18 = (
        [f"L_act_j{i}" for i in range(1, 8)] + ["L_act_f1", "L_act_f2"]
        + [f"R_act_j{i}" for i in range(1, 8)] + ["R_act_f1", "R_act_f2"]
    )
    a_idx = [int(m.actuator(an).id) for an in a_names_18]

    pad_ids = (
        int(m.geom("L_left_finger_pad").id),
        int(m.geom("L_right_finger_pad").id),
        int(m.geom("R_left_finger_pad").id),
        int(m.geom("R_right_finger_pad").id),
    )

    rigid_names = [rb["id"] for rb in rigid_specs]
    rigid_bids = [int(m.body(rn).id) for rn in rigid_names]
    dyn_bids = {int(m.body(rb["id"]).id) for rb in rigid_specs if rb["dynamic"]}
    dyn_geom_ids = {gi for gi in range(int(m.ngeom)) if int(m.geom_bodyid[gi]) in dyn_bids}

    first_act = (
        replay_actions[0]
        if replay_actions is not None
        else ctrl.step({"step_idx": 0, "t_s": 0.0, "qpos": np.zeros(18), "tactile_normal_force_n": np.zeros(4), "wrist_wrench": np.zeros(12)})
    )
    for ii, qi in enumerate(q_idx):
        d.qpos[qi] = float(first_act[ii])
    for ii, ai in enumerate(a_idx):
        d.ctrl[ai] = float(first_act[ii])
    mujoco.mj_forward(m, d)
    rest_world = np.asarray(d.flexvert_xpos, dtype=np.float64).copy()

    total_steps = len(replay_actions) if replay_actions is not None else int(ctrl.total_steps)
    qpos_hist = np.zeros((total_steps, 18), dtype=np.float64)
    act_hist = np.zeros((total_steps, 18), dtype=np.float64)
    eeL_hist = np.zeros((total_steps, 3), dtype=np.float64)
    eeR_hist = np.zeros((total_steps, 3), dtype=np.float64)
    tac_hist = np.zeros((total_steps, 4), dtype=np.float64)
    wrench_hist = np.zeros((total_steps, 12), dtype=np.float64)
    pts_hist = np.zeros((total_steps, len(rest_world), 3), dtype=np.float64)
    rigid_poses_hist = np.zeros((total_steps, len(rigid_names), 7), dtype=np.float64)

    for k in range(total_steps):
        q18_now = np.array([float(d.qpos[qi]) for qi in q_idx], dtype=np.float64)
        fn4_ctrl, fn4_def, w12 = _pad_forces_4ch(m, d, pad_ids, include_geom_ids=dyn_geom_ids)
        if replay_actions is not None:
            act = replay_actions[k]
        else:
            act = ctrl.step(
                {
                    "step_idx": k,
                    "t_s": k * CTRL_DT,
                    "qpos": q18_now,
                    "tactile_normal_force_n": fn4_ctrl,
                    "wrist_wrench": w12,
                }
            )
        for ii, ai in enumerate(a_idx):
            d.ctrl[ai] = float(act[ii])
        for _ in range(SUBSTEPS_PER_CTRL):
            mujoco.mj_step(m, d)
        if _DEBUG_BODIES:
            _debug_dump_contacts(m, d, k, _DEBUG_BODIES)

        q18_fin = np.array([float(d.qpos[qi]) for qi in q_idx], dtype=np.float64)
        eeL, _, _, _, _, eeR, _, _, _, _ = bimanual_pad_world_poses(q18_fin)
        qpos_hist[k] = q18_fin
        act_hist[k] = act
        eeL_hist[k] = eeL
        eeR_hist[k] = eeR
        tac_hist[k] = fn4_def
        wrench_hist[k] = w12
        pts_hist[k] = np.asarray(d.flexvert_xpos, dtype=np.float64).copy()
        for rj, bid in enumerate(rigid_bids):
            rigid_poses_hist[k, rj, :3] = np.asarray(d.xpos[bid], dtype=np.float64)
            rigid_poses_hist[k, rj, 3:] = np.asarray(d.xquat[bid], dtype=np.float64)

    metrics, _ = evaluate_v04_task_outcome(
        task_id=task_id,
        engine_id="mujoco4",
        mode=mode,
        zero_friction=zero_friction,
        ctx=ctx,
        rest_world=rest_world,
        pts_hist=pts_hist,
        elems=tets,
        tac_hist=tac_hist,
        rigid_names=rigid_names,
        rigid_poses_hist=rigid_poses_hist,
        qpos_hist=qpos_hist,
    )
    npz_arrays = {
        "qpos": qpos_hist,
        "action": act_hist,
        "ee_left_pos": eeL_hist,
        "ee_right_pos": eeR_hist,
        "tactile_normal_force_n": tac_hist,
        "wrist_wrench": wrench_hist,
        "deformable_rest_points": rest_world,
        "deformable_points": pts_hist,
        "deformable_elements": tets,
        "rigid_body_names": np.asarray(rigid_names),
        "rigid_body_poses": rigid_poses_hist,
        "step_dt": np.array([CTRL_DT], dtype=np.float64),
    }
    return save_v04_episode_bundle(
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
    parser = argparse.ArgumentParser(description="MuJoCo 4.0.1 18-DOF Distillation v0.4 Worker")
    parser.add_argument("--batch-json", required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()

    jobs = json.loads(Path(args.batch_json).read_text(encoding="utf-8"))["jobs"]
    results = [run_single_episode(job) for job in jobs]
    out_p = Path(args.result_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"engine_id": "mujoco4", "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
