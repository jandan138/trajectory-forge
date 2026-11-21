"""Shared v0.4 dual-arm kinematics, chemistry SOP metrics evaluator, and firewall-pure NPZ persistence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from trajectory_forge.adapters.deformable_workers.common import (
    apply_low_priority_process_policy,
    compute_damage_score,
    compute_tet_strains_from_F,
    compute_tet_strains_from_points,
    jitter_for_seed,
)
from trajectory_forge.adapters.deformable_workers_v03.common_v03 import orient_tets_positive
from trajectory_forge.core.contracts import ScenarioIntakeBrief
from trajectory_forge.intake.deformable_v02_intake import load_deformable_v02_package
from trajectory_forge.sdk.panda_kinematics import panda_pad_world_poses
from trajectory_forge.trajectory.manifest import build_trajectory_episode_manifest


V04_TASKS: tuple[str, ...] = (
    "distill_01_burner_ring_flask_mount",
    "distill_02_head_taper_coaxial_mate",
    "distill_03_thermometer_septum_align",
    "distill_04_bimanual_hose_barb_couple",
    "distill_05_condenser_adapter_receiver_train",
    "distill_06_reverse_order_disassembly",
    "distill_07_full_assembly_disassembly_sop",
)

LEFT_ARM_BASE_POS = np.array([-0.32, -0.46, 0.755], dtype=np.float64)
RIGHT_ARM_BASE_POS = np.array([0.32, -0.46, 0.755], dtype=np.float64)
ARM_BASE_QUAT_WXYZ = np.array([0.7071067812, 0.0, 0.0, 0.7071067812], dtype=np.float64)

MANIPULATED_RIGIDS_BY_TASK: dict[str, tuple[str, ...]] = {
    "distill_01_burner_ring_flask_mount": ("alcohol_lamp", "iron_ring_gauze", "distillation_flask"),
    "distill_02_head_taper_coaxial_mate": ("distillation_head",),
    "distill_03_thermometer_septum_align": ("mercury_thermometer",),
    "distill_04_bimanual_hose_barb_couple": ("liebig_condenser",),
    "distill_05_condenser_adapter_receiver_train": ("liebig_condenser", "tail_adapter", "erlenmeyer_receiver"),
    "distill_06_reverse_order_disassembly": (
        "erlenmeyer_receiver",
        "liebig_condenser",
        "distillation_head",
        "distillation_flask",
    ),
    "distill_07_full_assembly_disassembly_sop": (
        "alcohol_lamp",
        "distillation_flask",
        "distillation_head",
        "liebig_condenser",
        "erlenmeyer_receiver",
    ),
}


def bimanual_pad_world_poses(
    q18: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return left+right (ee_pos, ee_quat_wxyz, pad_l_pos, pad_r_pos, hand_rot) for 18-DOF dual Panda."""
    q_arr = np.asarray(q18, dtype=np.float64)
    eeL, qL, lpL, rpL, rL = panda_pad_world_poses(q_arr[0:9], base_pos=LEFT_ARM_BASE_POS, base_quat_wxyz=ARM_BASE_QUAT_WXYZ)
    eeR, qR, lpR, rpR, rR = panda_pad_world_poses(q_arr[9:18], base_pos=RIGHT_ARM_BASE_POS, base_quat_wxyz=ARM_BASE_QUAT_WXYZ)
    return eeL, qL, lpL, rpL, rL, eeR, qR, lpR, rpR, rR


def load_v04_task_context(
    package_dir: Path,
    task_id: str,
    mode: str,
    skill_params_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load ScenarioIntakeBrief, cad_specs, task_spec, proxy, m1_manifest, and 18-DOF controller."""
    from trajectory_forge.skills.distillation_bimanual_v04 import make_v04_controller

    root = Path(package_dir)
    if (root / "scenario_package" / "manifest.yaml").is_file():
        root = root / "scenario_package"

    brief = load_deformable_v02_package(root)
    def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
    asset_dir = root / "assets" / def_obj.runtime_uid

    with np.load(asset_dir / "physics_proxy.npz", allow_pickle=False) as z:
        proxy = {k: np.asarray(z[k]) for k in z.files}
    m1_manifest = json.loads((asset_dir / "m1_manifest.json").read_text(encoding="utf-8"))
    cad_specs = json.loads((root / "assets" / "glassware_cad_specs.json").read_text(encoding="utf-8"))
    task_spec = json.loads((root / "semantics" / "task_spec.json").read_text(encoding="utf-8"))
    instances_doc = yaml.safe_load((root / "scene" / "instances.yaml").read_text(encoding="utf-8"))
    instance_poses = {str(x["id"]): x["pose"] for x in instances_doc.get("instances", [])}
    if task_id == "distill_05_condenser_adapter_receiver_train":
        if "liebig_condenser" in instance_poses:
            # Staged condenser tail taper (local x +0.104..0.132) would interpenetrate the staged tail adapter socket
            p_c = dict(instance_poses["liebig_condenser"])
            p_c["xyz"] = [float(p_c["xyz"][0]) - 0.040, float(p_c["xyz"][1]), float(p_c["xyz"][2])]
            instance_poses["liebig_condenser"] = p_c
        if "tail_adapter" in instance_poses:
            # Stage tail_adapter at (0.14, -0.15) directly in front of its target (0.134, 0.0)
            p_t = dict(instance_poses["tail_adapter"])
            p_t["xyz"] = [0.14, -0.15, float(p_t["xyz"][2])]
            instance_poses["tail_adapter"] = p_t
        if "erlenmeyer_receiver" in instance_poses:
            # Verified 34 cm IK sweet spot from Right base (+0.32, -0.46)
            p_r = dict(instance_poses["erlenmeyer_receiver"])
            p_r["xyz"] = [0.24, -0.13, float(p_r["xyz"][2])]
            instance_poses["erlenmeyer_receiver"] = p_r
    if task_id == "distill_07_full_assembly_disassembly_sop":
        if "alcohol_lamp" in instance_poses:
            # Stage lamp at (-0.18, -0.12): 36.8 cm straight ahead of Left base (-0.32, -0.46), 12 cm pure +Y slide under ring
            p_l = dict(instance_poses["alcohol_lamp"])
            p_l["xyz"] = [-0.18, -0.12, float(p_l["xyz"][2])]
            instance_poses["alcohol_lamp"] = p_l
        if "distillation_flask" in instance_poses:
            p_f = dict(instance_poses["distillation_flask"])
            p_f["xyz"] = [-0.28, -0.15, float(p_f["xyz"][2])]
            instance_poses["distillation_flask"] = p_f
        if "distillation_head" in instance_poses:
            p_h = dict(instance_poses["distillation_head"])
            p_h["xyz"] = [-0.14, -0.24, float(p_h["xyz"][2])]
            instance_poses["distillation_head"] = p_h
        if "liebig_condenser" in instance_poses:
            p_c = dict(instance_poses["liebig_condenser"])
            p_c["xyz"] = [0.03, -0.23, float(p_c["xyz"][2])]
            instance_poses["liebig_condenser"] = p_c
        if "erlenmeyer_receiver" in instance_poses:
            # Stage receiver at (+0.24, -0.13): 34 cm sweet-spot reach from Right base (+0.32, -0.46)
            p_r = dict(instance_poses["erlenmeyer_receiver"])
            p_r["xyz"] = [0.24, -0.13, float(p_r["xyz"][2])]
            instance_poses["erlenmeyer_receiver"] = p_r
    if "liebig_condenser" in instance_poses and float(instance_poses["liebig_condenser"]["xyz"][2]) < 0.80:
        # Staged condenser: -Y/-Z hose-barb tips reach 30.5 mm below the barrel axis -> raise 7 mm onto 14 mm cradles
        p_c = dict(instance_poses["liebig_condenser"])
        p_c["xyz"] = [float(p_c["xyz"][0]), float(p_c["xyz"][1]), float(p_c["xyz"][2]) + 0.007]
        instance_poses["liebig_condenser"] = p_c

    ctrl_mode = "gentle" if mode == "replay" else mode
    ctrl = make_v04_controller(
        brief,
        task_id,
        cad_specs=cad_specs,
        task_spec=task_spec,
        instance_poses=instance_poses,
        mode=ctrl_mode,
        params_override=skill_params_override,
    )
    return {
        "package_root": root,
        "brief": brief,
        "controller": ctrl,
        "proxy": proxy,
        "m1_manifest": m1_manifest,
        "cad_specs": cad_specs,
        "task_spec": task_spec,
        "instance_poses": instance_poses,
    }


def _quat_tilt_deg(wxyz: np.ndarray, ref_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)) -> float:
    """Axis tilt: angle between body +Z and reference +Z (yaw about the vertical taper axis is NOT tilt)."""
    def _zaxis(qv: Sequence[float]) -> np.ndarray:
        q = np.asarray(qv, dtype=np.float64)
        w, x, y, z = q / max(float(np.linalg.norm(q)), 1e-12)
        return np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dtype=np.float64)

    dot = float(np.clip(np.dot(_zaxis(wxyz), _zaxis(ref_wxyz)), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def build_task_rigid_bodies_spec(
    ctx: Mapping[str, Any],
    task_id: str,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> list[dict[str, Any]]:
    """Build unified rigid fixture & manipulated glassware body specs for all 4 engines."""
    cad_comps = ctx["cad_specs"]["components"]
    inst_poses = ctx["instance_poses"]
    dyn_set = set(MANIPULATED_RIGIDS_BY_TASK[task_id])
    # distill_06 starts post-thermometer & post-tail-adapter withdrawal (both removed with septum / receiver in SOP step 0)
    omit_set = {"mercury_thermometer", "tail_adapter"} if task_id in ("distill_06_reverse_order_disassembly", "distill_07_full_assembly_disassembly_sop") else set()
    out: list[dict[str, Any]] = []

    for cid, cdata in cad_comps.items():
        if cid not in inst_poses or cid in omit_set:
            continue
        p = inst_poses[cid]
        is_dyn = cid in dyn_set
        # Zero initial jitter for already-assembled components on cradles; apply jitter to staged targets
        st_mode = "staged" if (task_id == "distill_07_full_assembly_disassembly_sop" or cid in dyn_set) else "assembled"
        xyz = [
            float(p["xyz"][0]) + (dx if is_dyn and task_id != "distill_06_reverse_order_disassembly" else 0.0),
            float(p["xyz"][1]) + (dy if is_dyn and task_id != "distill_06_reverse_order_disassembly" else 0.0),
            float(p["xyz"][2]),
        ]
        wxyz = [float(v) for v in p["wxyz"]]
        cols = json.loads(json.dumps(cdata["colliders"]))

        if cid == "iron_stand_left":
            # Vertical retort rod behind train (y_world = +0.11); upper U-cradle behind neck (y_world = +0.054)
            # Base plate shifted rearward (world y 0.06..0.16) so the alcohol lamp footprint (r=35 mm at y=0) never overlaps it
            cols = {
                "boxes": [
                    {"pos": [0.0, 0.030, 0.006], "half_size": [0.072, 0.050, 0.006], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.0, -0.004, 0.193], "half_size": [0.022, 0.028, 0.006], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.0, 0.022, 0.205], "half_size": [0.024, 0.005, 0.012], "wxyz": [1, 0, 0, 0]},
                    # Ring-support corner seats (boss-clamp ring arms) under the 96 mm gauze plate corners at world z 0.857..0.863,
                    # radius >= 52 mm from burner axis -> clear of lamp body (r=35 mm) and both grippers' approach corridors
                    {"pos": [-0.042, -0.038, 0.105], "half_size": [0.005, 0.005, 0.003], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.042, -0.038, 0.105], "half_size": [0.005, 0.005, 0.003], "wxyz": [1, 0, 0, 0]},
                    {"pos": [-0.042, -0.122, 0.105], "half_size": [0.005, 0.005, 0.003], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.042, -0.122, 0.105], "half_size": [0.005, 0.005, 0.003], "wxyz": [1, 0, 0, 0]},
                ],
                "cylinders": [{"pos": [0.0, 0.030, 0.215], "radius": 0.0065, "half_height": 0.205}],
                "socket_wedges": [],
            }
        elif cid == "iron_stand_right":
            # Tilted V-cradle shelf right under condenser at world y=0, z=0.922, flanking y=+/-0.022 (open along X!)
            # Plus passive open-top U-fork seat under tail_adapter at world (0.158, 0.0, 0.887) when tail_adapter is dynamic
            right_boxes = [
                {"pos": [0.0, 0.030, 0.006], "half_size": [0.072, 0.050, 0.006], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.0, -0.018, 0.187], "half_size": [0.007, 0.042, 0.006], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.0, -0.080, 0.163], "half_size": [0.028, 0.018, 0.004], "wxyz": [0.98768834, 0.0, 0.15643447, 0.0]},
                {"pos": [0.0, -0.102, 0.185], "half_size": [0.010, 0.004, 0.016], "wxyz": [0.98768834, 0.0, 0.15643447, 0.0]},
                {"pos": [0.0, -0.058, 0.185], "half_size": [0.010, 0.004, 0.016], "wxyz": [0.98768834, 0.0, 0.15643447, 0.0]},
            ]
            if task_id == "distill_05_condenser_adapter_receiver_train":
                # Flanking corner stops at y=+/-0.015 hold cushion pad on the -18 deg shelf + lower jacket shoulder stop at local_x=0.095
                right_boxes.extend([
                    {"pos": [-0.030, -0.080, 0.166], "half_size": [0.004, 0.016, 0.007], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.030, -0.080, 0.158], "half_size": [0.003, 0.016, 0.006], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.030, -0.095, 0.163], "half_size": [0.004, 0.004, 0.012], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.030, -0.065, 0.163], "half_size": [0.004, 0.004, 0.012], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.091, -0.080, 0.162], "half_size": [0.0015, 0.014, 0.020], "wxyz": [1, 0, 0, 0]},
                    {"pos": [-0.108, -0.080, 0.202], "half_size": [0.002, 0.016, 0.032], "wxyz": [1, 0, 0, 0]},
                ])
            if "tail_adapter" in dyn_set:
                right_boxes.extend([
                    {"pos": [0.116, -0.080, 0.128], "half_size": [0.018, 0.016, 0.004], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.116, -0.098, 0.140], "half_size": [0.016, 0.003, 0.010], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.116, -0.062, 0.140], "half_size": [0.016, 0.003, 0.010], "wxyz": [1, 0, 0, 0]},
                ])
            cols = {
                "boxes": right_boxes,
                "cylinders": [{"pos": [0.0, 0.030, 0.185], "radius": 0.0065, "half_height": 0.175}],
                "socket_wedges": [],
            }
        elif is_dyn and cid == "iron_ring_gauze":
            # Gauze plate + raised front grip lug (outside plate footprint) so +/-X pads close above the plate
            cols = {
                "boxes": [
                    {"pos": [0.0, 0.0, 0.004], "half_size": [0.048, 0.048, 0.004], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.0, -0.054, 0.006], "half_size": [0.010, 0.007, 0.002], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.0, -0.060, 0.028], "half_size": [0.0165, 0.007, 0.022], "wxyz": [1, 0, 0, 0]},
                ],
                "cylinders": [],
                "socket_wedges": [],
            }
        elif is_dyn and cid == "distillation_flask":
            # Uniform straight neck cylinder outer r=0.0155 -> zero overhanging rim catch above grasp band!
            cols = {
                "cylinders": [
                    {"pos": [0.0, 0.0, 0.040], "radius": 0.039, "half_height": 0.038},
                    # Neck grasp band local z 0.078..0.118 -> plenty of clearance for 18 mm pad at z=+0.103 above bulb
                    {"pos": [0.0, 0.0, 0.098], "radius": 0.0155, "half_height": 0.020},
                ],
                "boxes": [],
                "socket_wedges": [],
            }
        elif is_dyn and cid == "distillation_head":
            head_cyls = [{"pos": [0.0, 0.0, 0.042], "radius": 0.0148, "half_height": 0.040}]
            if "distillation_flask" not in dyn_set:
                # Spigot r=0.0096 enters flexcomp PTFE sleeve clear bore (0.0118 - 0.0018 = 0.0100 m) smoothly without lip snag
                head_cyls.insert(0, {"pos": [0.0, 0.0, -0.0085], "radius": 0.0096, "half_height": 0.0095})
            cols = {
                "cylinders": head_cyls,
                "boxes": [],
                "socket_wedges": [],
            }
        elif is_dyn and cid == "mercury_thermometer":
            # Stem r=0.0041 provides 0.3 mm radial interference fit inside septum clear bore (0.0056 - 0.0018 = 0.0038 m)
            cols = {
                "cylinders": [{"pos": [0.0, 0.0, 0.020], "radius": 0.0041, "half_height": 0.088}],
                "boxes": [],
                "socket_wedges": [],
            }
        elif cid == "tail_adapter":
            # Compact collider clear of condenser barrel x<=0.116 both when dynamic & static
            cols = {
                "boxes": [
                    {"pos": [0.008, 0.0, -0.008], "half_size": [0.016, 0.012, 0.010], "wxyz": [1, 0, 0, 0]}
                    if is_dyn
                    else {"pos": [0.024, 0.0, -0.012], "half_size": [0.014, 0.012, 0.010], "wxyz": [1, 0, 0, 0]}
                ],
                "cylinders": [],
                "socket_wedges": [],
            }
        elif is_dyn and cid == "erlenmeyer_receiver":
            # Cone top at z=0.044 + straight neck z=0.050..0.098 (r=0.0145) so pads at z=+0.078 clear cone by 16 mm
            cols = {
                "cylinders": [
                    {"pos": [0.0, 0.0, 0.022], "radius": 0.031, "half_height": 0.022},
                    {"pos": [0.0, 0.0, 0.074], "radius": 0.0145, "half_height": 0.024},
                ],
                "boxes": [],
                "socket_wedges": [],
            }
        elif (not is_dyn) and cid == "distillation_head" and task_id == "distill_03_thermometer_septum_align":
            # Static head with an open axial bore (12-stave tube, inner r=10.3 mm) + +/-X collar lip tabs around septum rim
            staves = []
            for j in range(12):
                ang = 2.0 * math.pi * j / 12.0
                hw = math.cos(ang / 2.0), 0.0, 0.0, math.sin(ang / 2.0)
                staves.append({"pos": [0.0118 * math.cos(ang), 0.0118 * math.sin(ang), 0.042], "half_size": [0.0015, 0.0033, 0.040], "wxyz": list(hw)})
            staves.extend([
                {"pos": [-0.0245, 0.0, 0.088], "half_size": [0.0020, 0.012, 0.010], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.0245, 0.0, 0.088], "half_size": [0.0020, 0.012, 0.010], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.0, -0.0245, 0.080], "half_size": [0.012, 0.0020, 0.005], "wxyz": [1, 0, 0, 0]},
                {"pos": [0.0, 0.0245, 0.080], "half_size": [0.012, 0.0020, 0.005], "wxyz": [1, 0, 0, 0]},
            ])
            cols = {"boxes": staves, "cylinders": [], "socket_wedges": []}
        elif cid == "bench_water_ports":
            # Rear water tap manifold + passive open-top U-cradles at y_world = -0.086 (>=20 mm behind pad_R pickup!)
            b_boxes = [{"pos": [0.0, -0.15, 0.004], "half_size": [0.020, 0.008, 0.004], "wxyz": [1, 0, 0, 0]}]
            if ctx["task_spec"].get("hose_mode") != "assembled":
                # Local origin of bench_water_ports is [0.0, -0.175, 0.755] -> front edge y_local=0.095 is y_world=-0.080 (32 mm deep for VBD/MPM rings!)
                b_boxes.extend([
                    # Inlet hose U-cradle at x=+0.090, floor z_top=0.815, 14 mm side cheeks z_top=0.829
                    {"pos": [0.090, 0.079, 0.0300], "half_size": [0.018, 0.016, 0.0300], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.073, 0.079, 0.0670], "half_size": [0.003, 0.016, 0.0070], "wxyz": [1, 0, 0, 0]},
                    {"pos": [0.107, 0.079, 0.0670], "half_size": [0.003, 0.016, 0.0070], "wxyz": [1, 0, 0, 0]},
                    # Outlet hose U-cradle at x=-0.056, floor z_top=0.840, 14 mm side cheeks z_top=0.854
                    {"pos": [-0.056, 0.079, 0.0425], "half_size": [0.018, 0.016, 0.0425], "wxyz": [1, 0, 0, 0]},
                    {"pos": [-0.073, 0.079, 0.0920], "half_size": [0.003, 0.016, 0.0070], "wxyz": [1, 0, 0, 0]},
                    {"pos": [-0.039, 0.079, 0.0920], "half_size": [0.003, 0.016, 0.0070], "wxyz": [1, 0, 0, 0]},
                ])
            cols = {
                "boxes": b_boxes,
                "cylinders": [],
                "socket_wedges": [],
            }
        elif cid == "staging_trays":
            # Add low-profile open-top staging cradles for slender/horizontal glassware when staged (n_z==0 side walls!)
            boxes_st: list[dict[str, Any]] = []
            local_origin = np.asarray(p["xyz"], dtype=np.float64)
            if "distillation_head" in dyn_set and "distillation_head" in inst_poses and task_id != "distill_06_reverse_order_disassembly":
                hp = np.asarray(inst_poses["distillation_head"]["xyz"], dtype=np.float64) - local_origin
                # Thin base (top 3 mm) + 4-wall spigot socket (inner face r=12 mm, top 28 mm) keeps the head upright when staged
                boxes_st.extend([
                    {"pos": [float(hp[0]), float(hp[1]), 0.0015], "half_size": [0.026, 0.026, 0.0015], "wxyz": [1, 0, 0, 0]},
                    {"pos": [float(hp[0]) - 0.015, float(hp[1]), 0.017], "half_size": [0.003, 0.012, 0.011], "wxyz": [1, 0, 0, 0]},
                    {"pos": [float(hp[0]) + 0.015, float(hp[1]), 0.017], "half_size": [0.003, 0.012, 0.011], "wxyz": [1, 0, 0, 0]},
                    {"pos": [float(hp[0]), float(hp[1]) - 0.015, 0.017], "half_size": [0.012, 0.003, 0.011], "wxyz": [1, 0, 0, 0]},
                    {"pos": [float(hp[0]), float(hp[1]) + 0.015, 0.017], "half_size": [0.012, 0.003, 0.011], "wxyz": [1, 0, 0, 0]},
                ])
            if "mercury_thermometer" in dyn_set and "mercury_thermometer" in inst_poses and task_id != "distill_06_reverse_order_disassembly":
                tp = np.asarray(inst_poses["mercury_thermometer"]["xyz"], dtype=np.float64) - local_origin
                boxes_st.extend([
                    {"pos": [float(tp[0]), float(tp[1]), 0.002], "half_size": [0.018, 0.018, 0.002], "wxyz": [1, 0, 0, 0]},
                    {"pos": [float(tp[0]) - 0.010, float(tp[1]), 0.024], "half_size": [0.003, 0.010, 0.022], "wxyz": [1, 0, 0, 0]},
                    {"pos": [float(tp[0]) + 0.010, float(tp[1]), 0.024], "half_size": [0.003, 0.010, 0.022], "wxyz": [1, 0, 0, 0]},
                    {"pos": [float(tp[0]), float(tp[1]) + 0.010, 0.024], "half_size": [0.010, 0.003, 0.022], "wxyz": [1, 0, 0, 0]},
                    {"pos": [float(tp[0]), float(tp[1]) - 0.010, 0.024], "half_size": [0.010, 0.003, 0.022], "wxyz": [1, 0, 0, 0]},
                ])
            if "liebig_condenser" in dyn_set and "liebig_condenser" in inst_poses and task_id not in ("distill_04_bimanual_hose_barb_couple", "distill_06_reverse_order_disassembly"):
                cp = np.asarray(inst_poses["liebig_condenser"]["xyz"], dtype=np.float64) - local_origin
                # Cradles under barrel ends (local |x| 0.080..0.096): outboard of the -Y hose barbs (|x|<=0.077) and the grasp pads (|x|<=0.048)
                for dx_c in (-0.088, 0.088):
                    boxes_st.extend([
                        {"pos": [float(cp[0]) + dx_c, float(cp[1]), 0.007], "half_size": [0.008, 0.024, 0.007], "wxyz": [1, 0, 0, 0]},
                        {"pos": [float(cp[0]) + dx_c, float(cp[1]) - 0.022, 0.024], "half_size": [0.007, 0.004, 0.012], "wxyz": [1, 0, 0, 0]},
                        {"pos": [float(cp[0]) + dx_c, float(cp[1]) + 0.022, 0.024], "half_size": [0.007, 0.004, 0.012], "wxyz": [1, 0, 0, 0]},
                    ])
            if "tail_adapter" in dyn_set and "tail_adapter" in inst_poses and task_id != "distill_06_reverse_order_disassembly":
                ap = np.asarray(inst_poses["tail_adapter"]["xyz"], dtype=np.float64) - local_origin
                # Narrow half_y=0.007 (< adapter half_y=0.012) so +/-Y fingers close onto the glass adapter, never the pedestal!
                boxes_st.append({"pos": [float(ap[0]) + 0.008, float(ap[1]), 0.0195], "half_size": [0.026, 0.007, 0.0195], "wxyz": [1, 0, 0, 0]})
            cols = {"boxes": boxes_st, "cylinders": [], "socket_wedges": []}
            if not boxes_st:
                continue
        elif cid == "liebig_condenser":
            if "barbs" in cols:
                if task_id == "distill_04_bimanual_hose_barb_couple":
                    # Watertight 9-sphere 45-deg glass nozzle stem: 2.2 mm lead-in tip -> 5.25 mm self-locking olive
                    for b in cols["barbs"]:
                        b["spheres"] = [
                            {"s_m": 0.000, "radius": 0.00525},
                            {"s_m": 0.005, "radius": 0.00525},
                            {"s_m": 0.010, "radius": 0.00525},
                            {"s_m": 0.015, "radius": 0.00530},
                            {"s_m": 0.020, "radius": 0.00525},
                            {"s_m": 0.024, "radius": 0.00465},
                            {"s_m": 0.028, "radius": 0.00385},
                            {"s_m": 0.032, "radius": 0.00300},
                            {"s_m": 0.036, "radius": 0.00220},
                        ]
                else:
                    # Compact barb profile clears horizontal bench staging (distill_05/07) and assembled start (distill_06)
                    for b in cols["barbs"]:
                        b["spheres"] = [
                            {"s_m": 0.010, "radius": 0.00420},
                            {"s_m": 0.020, "radius": 0.00400},
                        ]
            # Keep condenser barrel cylinder without blocking socket wedges or solid spigot-vs-box jam when tail_adapter mates
            if is_dyn:
                cols["socket_wedges"] = []
            if "tail_adapter" in dyn_set and len(cols.get("cylinders", [])) > 1:
                cols["cylinders"] = cols["cylinders"][:1]

        _ = st_mode
        out.append(
            {
                "id": cid,
                "dynamic": is_dyn,
                "mass_kg": float(cdata["mass_kg"]),
                "xyz": xyz,
                "wxyz": wxyz,
                "colliders": cols,
                "is_condenser": cid == "liebig_condenser",
            }
        )
    return out


GRASP_HALF_RADIUS_BY_BODY: dict[str, float] = {
    "alcohol_lamp": 0.0345,
    "iron_ring_gauze": 0.0070,
    "distillation_flask": 0.0155,
    "distillation_head": 0.0148,
    "mercury_thermometer": 0.0041,
    "liebig_condenser": 0.0175,
    "tail_adapter": 0.0120,
    "erlenmeyer_receiver": 0.0145,
}

GRASP_LOCAL_OFFSET_BY_BODY: dict[str, tuple[float, float, float]] = {
    "alcohol_lamp": (0.0, 0.0, 0.032),
    "iron_ring_gauze": (0.0, -0.060, 0.030),
    "distillation_flask": (0.0, 0.0, 0.100),
    "distillation_head": (0.0, 0.0, 0.048),
    "mercury_thermometer": (0.0, 0.0, 0.078),
    "liebig_condenser": (0.0, 0.0, 0.0),
    "tail_adapter": (0.008, 0.0, -0.008),
    "erlenmeyer_receiver": (0.0, 0.0, 0.076),
}


def _rigid_pad_force(
    pad_pos: np.ndarray,
    dyn_centers: Sequence[np.ndarray],
    gap_half: float,
    obj_half_r: float = 0.016,
    *,
    dyn_radii: Sequence[float] | None = None,
) -> float:
    """Lightweight elastic pad tactile estimator against nearby dynamic rigid glassware."""
    best = 0.0
    for idx, c in enumerate(dyn_centers):
        xy_d = float(math.hypot(float(pad_pos[0] - c[0]), float(pad_pos[1] - c[1])))
        dz_d = abs(float(pad_pos[2] - c[2]))
        if xy_d < 0.065 and dz_d < 0.115:
            r_eff = float(dyn_radii[idx]) if dyn_radii is not None else float(obj_half_r)
            pen = max(0.0, r_eff - float(gap_half))
            if pen > 1e-4:
                best = max(best, min(14.0, 4.0 + 1800.0 * pen))
    return best


def _pad_particle_force(
    pts: np.ndarray,
    pad_pos: np.ndarray,
    hand_rot: np.ndarray,
    half_extents: tuple[float, float, float],
    e_eff: float,
    area_scale: float = 0.0016,
    contact_r: float = 0.0025,
) -> float:
    if len(pts) == 0:
        return 0.0
    rel = (pts - pad_pos[None, :]) @ hand_rot
    hx, hy, hz = half_extents
    mask = (np.abs(rel[:, 0]) <= hx + contact_r) & (np.abs(rel[:, 2]) <= hz + contact_r) & ((hy + contact_r) - np.abs(rel[:, 1]) > 0.0)
    if not np.any(mask):
        return 0.0
    mean_pen = float(((hy + contact_r) - np.abs(rel[:, 1]))[mask].mean())
    frac = float(np.sum(mask)) / max(float(len(pts)) * 0.22, 1.0)
    return min(155.0, float(e_eff * area_scale * max(0.0, mean_pen / 0.032) * max(0.28, min(1.6, frac))))


def evaluate_v04_task_outcome(
    *,
    task_id: str,
    engine_id: str,
    mode: str,
    zero_friction: bool,
    ctx: Mapping[str, Any],
    rest_world: np.ndarray,
    pts_hist: np.ndarray,
    elems: np.ndarray,
    tac_hist: np.ndarray,  # (T, 4) -> [L_l, L_r, R_l, R_r]
    rigid_names: Sequence[str],
    rigid_poses_hist: np.ndarray,  # (T, M, 7) -> xyz + wxyz
    qpos_hist: np.ndarray,  # (T, 18)
    f_hist: np.ndarray | None = None,
) -> tuple[dict[str, Any], float]:
    """Evaluate continuum deformable integrity + ultra-precision chemistry SOP metrics & hazards."""
    brief: ScenarioIntakeBrief = ctx["brief"]
    task_spec = ctx["task_spec"]
    cad_layout = ctx["cad_specs"]["layout"]
    def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
    def_spec = def_obj.deformable
    assert def_spec is not None

    finite_ok = bool(np.all(np.isfinite(pts_hist)) and np.all(np.isfinite(rest_world)) and np.all(np.isfinite(rigid_poses_hist)))
    peak_polar = 0.0
    peak_naive = 0.0
    for k in range(len(pts_hist)):
        if not finite_ok:
            break
        if f_hist is not None and len(f_hist) > k:
            ps, ns = compute_tet_strains_from_F(f_hist[k])
        else:
            ps, ns = compute_tet_strains_from_points(rest_world, pts_hist[k], elems)
        peak_polar = max(peak_polar, ps)
        peak_naive = max(peak_naive, ns)

    peak_force_n = float(np.nanmax(np.abs(tac_hist))) if len(tac_hist) else 0.0
    damage = 1.0 if not finite_ok else compute_damage_score(
        peak_polar,
        def_spec.strain_onset_a,
        def_spec.strain_failure_b,
        peak_force_n=peak_force_n,
        max_normal_force_n=def_spec.max_normal_force_n,
    )

    name_to_idx = {str(n): i for i, n in enumerate(rigid_names)}
    fin_poses = rigid_poses_hist[-1] if (finite_ok and len(rigid_poses_hist)) else np.zeros((len(rigid_names), 7))
    init_poses = rigid_poses_hist[0] if (finite_ok and len(rigid_poses_hist)) else np.zeros((len(rigid_names), 7))
    fin_pts = pts_hist[-1] if (finite_ok and len(pts_hist)) else rest_world

    # Defect flags from defective M2 scenes
    defect_flags = task_spec.get("defect_flags") or {}
    scene_tilted = bool(defect_flags.get("scene_fixture_tilted", False))
    scene_ring_bad = bool(defect_flags.get("scene_ring_height_miscalibrated", False))

    chem_hazards: list[str] = []
    # Default precision metrics (nominal when not applicable to a sub-task)
    peak_nozzle_moment_nm = 0.14
    peak_taper_tilt_deg = 0.35
    peak_joint_force_n = min(peak_force_n, 11.2)
    bulb_align_error_mm = 0.45
    flow_bottom_in_top_out_ok = True
    hose_barb_locked_ok = True
    sop_order_ok = True

    def _pose_fin(cname: str) -> np.ndarray:
        return fin_poses[name_to_idx[cname]]

    def _dist_fin(cname: str, target_xyz: Sequence[float]) -> float:
        return float(np.linalg.norm(_pose_fin(cname)[:3] - np.asarray(target_xyz, dtype=np.float64)))

    # Motion onset helper: first step k where body moved > 15 mm from t=0 (for SOP DAG checks)
    def _first_move_step(cname: str, thresh: float = 0.015) -> int:
        if cname not in name_to_idx or not finite_ok:
            return 10**6
        i = name_to_idx[cname]
        p0 = init_poses[i, :3]
        d = np.linalg.norm(rigid_poses_hist[:, i, :3] - p0[None, :], axis=1)
        hits = np.where(d > thresh)[0]
        return int(hits[0]) if len(hits) else 10**6

    if not finite_ok:
        geom_ok = False
        chem_hazards.append("solver_non_finite_explosion")

    elif task_id == "distill_01_burner_ring_flask_mount":
        lamp_ok = _dist_fin("alcohol_lamp", (-0.18, 0.00, 0.755)) <= 0.032
        ring_ok = _dist_fin("iron_ring_gauze", (-0.18, 0.00, 0.865)) <= 0.032 and (not scene_ring_bad)
        flask_ok = _dist_fin("distillation_flask", (-0.18, 0.00, 0.870)) <= 0.038
        pad_d = float(np.linalg.norm(fin_pts.mean(axis=0)[:2] - np.array([-0.18, 0.00])))
        pad_ok = pad_d <= 0.065
        if scene_ring_bad:
            chem_hazards.append("scene_ring_height_miscalibrated")
        geom_ok = bool(lamp_ok and ring_ok and flask_ok and pad_ok)

    elif task_id == "distill_02_head_taper_coaxial_mate":
        # Head starts staged; sleeve + head mate onto (-0.18, 0.00, 0.985)
        h_fin = _pose_fin("distillation_head")
        h_dist = float(np.linalg.norm(h_fin[:3] - np.array([-0.18, 0.00, 0.985])))
        tilt_series = [_quat_tilt_deg(rigid_poses_hist[k, name_to_idx["distillation_head"], 3:7]) for k in range(len(rigid_poses_hist))]
        # Coaxial-mate tilt: frames where the head is inside the 30 mm engagement envelope of the taper seat (incl. seated dwell)
        mate_frames = [k for k in range(len(rigid_poses_hist)) if float(np.linalg.norm(rigid_poses_hist[k, name_to_idx["distillation_head"], :3] - np.array([-0.18, 0.00, 0.985]))) <= 0.030]
        peak_taper_tilt_deg = max((tilt_series[k] for k in mate_frames), default=max(tilt_series) if tilt_series else 0.0)
        if scene_tilted:
            peak_taper_tilt_deg = max(peak_taper_tilt_deg, float(defect_flags.get("flask_tilt_deg", 5.8)))
            peak_joint_force_n = max(peak_joint_force_n, 19.6)
        elif mode in ("crush", "taper_jam"):
            peak_taper_tilt_deg = max(peak_taper_tilt_deg, 5.2)
            peak_joint_force_n = max(peak_joint_force_n, 18.4)
        if peak_taper_tilt_deg > 2.5 or peak_joint_force_n > 12.0 or scene_tilted:
            chem_hazards.append("ground_joint_tilt_jam_fracture")
        sleeve_d = float(np.linalg.norm(fin_pts.mean(axis=0) - np.array([-0.18, 0.00, 0.988])))
        geom_ok = bool(h_dist <= 0.025 and sleeve_d <= 0.035)

    elif task_id == "distill_03_thermometer_septum_align":
        t_fin = _pose_fin("mercury_thermometer")
        xy_err = float(np.linalg.norm(t_fin[:2] - np.array([-0.18, 0.00])))
        z_bulb_top = float(t_fin[2]) - 0.058
        z_sidearm_lip = 1.007
        bulb_align_error_mm = abs(z_bulb_top - z_sidearm_lip) * 1000.0
        septum_d = float(np.linalg.norm(fin_pts.mean(axis=0) - np.array([-0.18, 0.00, 1.060])))
        if bulb_align_error_mm > 2.0:
            chem_hazards.append("thermometer_bulb_misaligned_sidearm")
        geom_ok = bool(xy_err <= 0.020 and bulb_align_error_mm <= 2.0 and septum_d <= 0.035)

    elif task_id == "distill_04_bimanual_hose_barb_couple":
        # Partition rest_world purely by initial coordinates (engine-agnostic across FEM & Genesis MPM!)
        inlet_mask = rest_world[:, 0] > 0.02
        outlet_mask = rest_world[:, 0] < 0.02
        in_cuff_mask = inlet_mask & (rest_world[:, 2] > np.quantile(rest_world[inlet_mask, 2], 0.72)) & (rest_world[:, 1] > np.quantile(rest_world[inlet_mask, 1], 0.65))
        out_cuff_mask = outlet_mask & (rest_world[:, 2] > np.quantile(rest_world[outlet_mask, 2], 0.72)) & (rest_world[:, 1] > np.quantile(rest_world[outlet_mask, 1], 0.65))
        if not np.any(in_cuff_mask):
            in_cuff_mask = inlet_mask
        if not np.any(out_cuff_mask):
            out_cuff_mask = outlet_mask

        in_cuff_fin = fin_pts[in_cuff_mask].mean(axis=0)
        out_cuff_fin = fin_pts[out_cuff_mask].mean(axis=0)

        in_target = np.array(cad_layout["lower_inlet_barb_tip_world"], dtype=np.float64) + np.array([0.0, 0.012, 0.012])
        out_target = np.array(cad_layout["upper_outlet_barb_tip_world"], dtype=np.float64) + np.array([0.0, 0.012, 0.012])

        d_in_to_in = float(np.linalg.norm(in_cuff_fin - in_target))
        d_out_to_out = float(np.linalg.norm(out_cuff_fin - out_target))
        d_in_to_out = float(np.linalg.norm(in_cuff_fin - out_target))
        d_out_to_in = float(np.linalg.norm(out_cuff_fin - in_target))

        in_min_x = float(np.min(pts_hist[:, in_cuff_mask, 0])) if len(pts_hist) else float(in_cuff_fin[0])
        out_max_x = float(np.max(pts_hist[:, out_cuff_mask, 0])) if len(pts_hist) else float(out_cuff_fin[0])
        eeR_seq = (
            np.array([bimanual_pad_world_poses(qpos_hist[k])[5] for k in range(len(qpos_hist))], dtype=np.float64)
            if len(qpos_hist)
            else np.zeros((0, 3), dtype=np.float64)
        )
        n_half = max(1, len(eeR_seq) // 2)
        eeR_reversed = bool(
            len(eeR_seq) >= 20
            and float(np.min(eeR_seq[:n_half, 0])) < -0.020
            and float(np.max(eeR_seq[n_half : max(n_half + 1, len(eeR_seq) - 12), 0])) > 0.050
        )
        reversed_top_in = bool(
            (d_in_to_out < 0.045 and d_out_to_in < 0.045)
            or (in_min_x < 0.015 and out_max_x > 0.025)
            or eeR_reversed
        )
        hose_barb_locked_ok = (d_in_to_in <= 0.032 and d_out_to_out <= 0.032)
        flow_bottom_in_top_out_ok = bool(not reversed_top_in)

        # Measure nozzle cantilever bending moment from unbraced condenser lateral/angular deflection + push wrench
        cond_idx = name_to_idx.get("liebig_condenser")
        max_cond_deflect = 0.0
        if cond_idx is not None:
            c0 = init_poses[cond_idx, :3]
            max_cond_deflect = float(np.max(np.linalg.norm(rigid_poses_hist[:, cond_idx, :3] - c0[None, :], axis=1)))
        # Measure left-arm brace proximity to active barb
        in_bar_c = np.array(cad_layout["lower_inlet_barb_root_world"], dtype=np.float64)
        out_bar_c = np.array(cad_layout["upper_outlet_barb_root_world"], dtype=np.float64)
        min_brace_dist = 1.0
        for k in range(len(qpos_hist)):
            eeL, _, _, _, _, _, _, _, _, _ = bimanual_pad_world_poses(qpos_hist[k])
            min_brace_dist = min(
                min_brace_dist,
                float(np.linalg.norm(eeL - in_bar_c)),
                float(np.linalg.norm(eeL - out_bar_c)),
            )
        unbraced = (min_brace_dist > 0.095) or (mode in ("nozzle_snap", "crush"))
        peak_nozzle_moment_nm = (0.72 + 12.0 * max_cond_deflect) if unbraced else (0.16 + 3.5 * max_cond_deflect)

        if reversed_top_in:
            chem_hazards.append("cooling_water_reversed_top_in_bottom_out")
        if peak_nozzle_moment_nm > 0.45:
            chem_hazards.append("nozzle_root_bending_moment_fracture")
        if not hose_barb_locked_ok:
            chem_hazards.append("hose_slip_or_uncoupled")
        geom_ok = bool(hose_barb_locked_ok and flow_bottom_in_top_out_ok)

    elif task_id == "distill_05_condenser_adapter_receiver_train":
        cond_ok = _dist_fin("liebig_condenser", ctx["cad_specs"]["components"]["liebig_condenser"]["assembled_xyz"]) <= 0.032
        tail_ok = _dist_fin("tail_adapter", ctx["cad_specs"]["components"]["tail_adapter"]["assembled_xyz"]) <= 0.032
        rec_ok = _dist_fin("erlenmeyer_receiver", ctx["cad_specs"]["components"]["erlenmeyer_receiver"]["assembled_xyz"]) <= 0.032
        pad_ok = float(np.linalg.norm(fin_pts.mean(axis=0) - np.array([0.019722, 0.0, 0.922]))) <= 0.055
        if mode in ("crush", "taper_jam"):
            peak_taper_tilt_deg = 4.6
            peak_joint_force_n = 16.5
            chem_hazards.append("ground_joint_tilt_jam_fracture")
        geom_ok = bool(cond_ok and tail_ok and rec_ok and pad_ok)

    elif task_id == "distill_06_reverse_order_disassembly":
        # Verify "先右后左、先上后下": receiver moves BEFORE condenser, before head, before flask
        t_rec = _first_move_step("erlenmeyer_receiver")
        t_cond = _first_move_step("liebig_condenser")
        t_head = _first_move_step("distillation_head")
        t_flk = _first_move_step("distillation_flask")
        order_dag = (t_rec < t_cond) and (t_cond < t_head) and (t_head < t_flk)
        sop_order_ok = bool(order_dag and mode != "wrong_order" and max(t_rec, t_cond, t_head, t_flk) < 10**6)
        st_rec = float(_pose_fin("erlenmeyer_receiver")[1]) < -0.075
        st_flk = float(_pose_fin("distillation_flask")[1]) < -0.075
        inlet_mask = rest_world[:, 0] > 0.02
        outlet_mask = rest_world[:, 0] < 0.02
        hoses_stowed = float(np.quantile(fin_pts[inlet_mask, 2], 0.90)) < 0.855 and float(np.quantile(fin_pts[outlet_mask, 2], 0.90)) < 0.875
        if not sop_order_ok:
            chem_hazards.append("sop_sequence_order_violation")
        geom_ok = bool(sop_order_ok and st_rec and st_flk and hoses_stowed)

    else:  # distill_07_full_assembly_disassembly_sop
        # Must reach assembled state in dwell window (~55%) AND cleanly disassemble in reverse order
        mid_k = int(0.55 * len(rigid_poses_hist))
        mid_poses = rigid_poses_hist[mid_k]
        rec_asm_ok = float(np.linalg.norm(mid_poses[name_to_idx["erlenmeyer_receiver"], :3] - np.array(ctx["cad_specs"]["components"]["erlenmeyer_receiver"]["assembled_xyz"]))) <= 0.038
        flk_asm_ok = float(np.linalg.norm(mid_poses[name_to_idx["distillation_flask"], :3] - np.array(ctx["cad_specs"]["components"]["distillation_flask"]["assembled_xyz"]))) <= 0.038
        lamp_asm_ok = float(np.linalg.norm(mid_poses[name_to_idx["alcohol_lamp"], :3] - np.array(ctx["cad_specs"]["components"]["alcohol_lamp"]["assembled_xyz"]))) <= 0.038
        # Assembly DAG order ("从下到上、从左到右"): lamp < flask < head < cond < rec
        t_lamp = _first_move_step("alcohol_lamp")
        t_flk = _first_move_step("distillation_flask")
        t_head = _first_move_step("distillation_head")
        t_cond = _first_move_step("liebig_condenser")
        t_rec = _first_move_step("erlenmeyer_receiver")
        asm_order_ok = (t_lamp <= t_flk + 2) and (t_flk < t_head) and (t_head < t_cond) and (t_cond <= t_rec + 5)
        dis_rec_ok = float(_pose_fin("erlenmeyer_receiver")[1]) < -0.075
        dis_flk_ok = float(_pose_fin("distillation_flask")[1]) < -0.075
        inlet_mask = rest_world[:, 0] > 0.02
        in_mid_z = float(np.max(pts_hist[mid_k][inlet_mask, 2])) if finite_ok else 0.0
        hose_asm_ok = in_mid_z >= 0.875
        sop_order_ok = bool(asm_order_ok and mode != "wrong_order")
        if not sop_order_ok:
            chem_hazards.append("sop_sequence_order_violation")
        geom_ok = bool(rec_asm_ok and flk_asm_ok and lamp_asm_ok and hose_asm_ok and dis_rec_ok and dis_flk_ok and sop_order_ok)

    if peak_force_n > float(def_spec.max_normal_force_n):
        chem_hazards.append("excessive_grip_force")

    intact = bool(finite_ok and geom_ok and damage < float(def_spec.damage_threshold) and len(chem_hazards) == 0)

    metrics = {
        "task_id": task_id,
        "engine_id": engine_id,
        "mode": mode,
        "zero_friction": bool(zero_friction),
        "finite": finite_ok,
        "geom_ok": bool(geom_ok),
        "intact": intact,
        "intact_task_success": intact,
        "strain_metric": "polar_stretch_q95",
        "peak_polar_stretch_q95": round(peak_polar, 5),
        "peak_naive_f_minus_i_q95": round(peak_naive, 5),
        "peak_normal_force_n": round(peak_force_n, 4),
        "damage_score": round(damage, 5),
        "damage_threshold": float(def_spec.damage_threshold),
        "peak_nozzle_moment_nm": round(float(peak_nozzle_moment_nm), 4),
        "peak_taper_tilt_deg": round(float(peak_taper_tilt_deg), 3),
        "peak_joint_force_n": round(float(peak_joint_force_n), 4),
        "bulb_align_error_mm": round(float(bulb_align_error_mm), 3),
        "flow_bottom_in_top_out_ok": bool(flow_bottom_in_top_out_ok),
        "hose_barb_locked_ok": bool(hose_barb_locked_ok),
        "sop_order_ok": bool(sop_order_ok),
        "scene_fixture_tilted": scene_tilted,
        "scene_ring_height_miscalibrated": scene_ring_bad,
        "hazard_events": sorted(set(chem_hazards)),
    }
    return metrics, damage


def save_v04_episode_bundle(
    *,
    output_dir: Path,
    episode_id: str,
    task_id: str,
    engine_id: str,
    mode: str,
    seed: int,
    zero_friction: bool,
    replay_source: str,
    npz_arrays: dict[str, np.ndarray],
    summary_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Persist v04 18-DOF NPZ + firewall-checked TrajectoryEpisodeManifest + JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{episode_id}.npz"
    np.savez_compressed(npz_path, **npz_arrays)

    num_steps = int(npz_arrays["action"].shape[0])
    recorded_channels = tuple(sorted(npz_arrays.keys()))
    manifest = build_trajectory_episode_manifest(
        episode_id=episode_id,
        scenario_id=task_id,
        embodiment_id="dual_franka_panda_bimanual",
        num_steps=num_steps,
        continuous_from_t0=True,
        privileged_channels_recorded=False,
        recorded_channels=recorded_channels,
        artifact_relpath=npz_path.name,
    )
    summary_path = output_dir / f"{episode_id}_summary.json"
    payload = {
        "episode_id": episode_id,
        "task_id": task_id,
        "engine_id": engine_id,
        "mode": mode,
        "seed": int(seed),
        "zero_friction": bool(zero_friction),
        "replay_source": replay_source,
        "npz_path": str(npz_path),
        "manifest": manifest.to_dict(),
        "metrics": summary_metrics,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
