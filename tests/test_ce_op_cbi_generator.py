from __future__ import annotations

import copy
import json
from pathlib import Path
import importlib.util
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_ce_op_cbi_episodes.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_ce_op_cbi_episodes", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_collected_fixture(tmp_path: Path, package_id: str) -> Path:
    pkg = tmp_path / package_id
    pkg.mkdir(parents=True, exist_ok=True)
    ep_rel = f"tasks/scenario_forge/{package_id}/ep_0.json"
    ep_path = pkg / ep_rel
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "genmanip-collected-package/v0.1",
        "package_id": package_id,
        "entrypoints": {
            "episode_metadata": ep_rel,
            "task_config": "tasks/task_config.yaml",
            "scene_usd": "assets/scene.usd",
            "camera_config": "cameras/fixed_camera_lift2.yml",
        },
    }
    (pkg / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Initial tabletop layout with hero appliance, two target consumables, and two near-field clutter props.
    ep_doc = {
        "task_name": package_id,
        "task_data": {
            "instruction": "Base instruction.",
            "initial_layout": {
                "hero_rig": {
                    "type": "object",
                    "position": [0.0, -0.08, 0.755],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                    "footprint_xy": [0.22, 0.36],
                },
                "cup": {
                    "type": "object",
                    "position": [-0.23, -0.24, 0.757],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                    "footprint_xy": [0.08, 0.08],
                },
                "near_jam_jar": {
                    "type": "object",
                    "position": [-0.14, 0.04, 0.757],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                    "footprint_xy": [0.07, 0.07],
                },
                "near_candle": {
                    "type": "object",
                    "position": [0.22, -0.02, 0.757],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                    "footprint_xy": [0.075, 0.075],
                },
            },
            "goal": [
                [
                    [
                        {
                            "type": "manip/default/sr_based_genmanip_relationship",
                            "sub_goal_setting": {
                                "type": "manip/default/range",
                                "obj1_uid": "cup",
                                "x_type": "relative",
                                "y_type": "relative",
                                "x_range": [-0.02, 0.02],
                                "y_range": [-0.21, -0.17],
                                "z_range": [0.03, 0.05],
                            },
                        }
                    ]
                ],
                [
                    [
                        {
                            "type": "manip/default/sr_based_genmanip_relationship",
                            "sub_goal_setting": {
                                "type": "manip/default/range",
                                "obj1_uid": "near_jam_jar",
                                "x_type": "absolute",
                                "y_type": "absolute",
                                "z_type": "absolute",
                                "x_range": [-0.175, -0.105],
                                "y_range": [0.005, 0.075],
                                "z_range": [0.727, 0.787],
                            },
                        },
                        {
                            "type": "manip/default/sr_based_genmanip_relationship",
                            "sub_goal_setting": {
                                "type": "manip/default/range",
                                "obj1_uid": "near_candle",
                                "x_type": "absolute",
                                "y_type": "absolute",
                                "z_type": "absolute",
                                "x_range": [0.185, 0.255],
                                "y_range": [-0.055, 0.015],
                                "z_range": [0.727, 0.787],
                            },
                        },
                    ]
                ],
            ],
            "scenario_forge_runtime_contract": {
                "objects": [
                    {"id": "hero_rig", "role": "articulated_appliance"},
                    {"id": "cup", "role": "target_container"},
                    {"id": "near_jam_jar", "role": "context_prop"},
                    {"id": "near_candle", "role": "context_prop"},
                ]
            },
        },
    }
    ep_path.write_text(json.dumps(ep_doc, indent=2), encoding="utf-8")
    return pkg


def test_cbi_shifts_absolute_anti_knockover_goal_bounds_and_keeps_clearance(tmp_path: Path) -> None:
    mod = _load_script_module()
    assert "ce_01_home_espresso_brew_v2" in mod.INSTRUCTION_PARAPHRASES
    assert "ce_12_home_wireless_uv_dock_v2" in mod.INSTRUCTION_PARAPHRASES

    pkg = _make_collected_fixture(tmp_path, "ce_01_home_espresso_brew_v2")
    out_dir = tmp_path / "cbi_out"
    mod.generate_variants(
        collected_root=pkg,
        out_dir=out_dir,
        num_episodes=3,
        start_seed=0,
    )

    ep0 = json.loads((out_dir / "ce_01_home_espresso_brew_v2" / "seed_0000" / "episode_metadata.json").read_text())
    ep1 = json.loads((out_dir / "ce_01_home_espresso_brew_v2" / "seed_0001" / "episode_metadata.json").read_text())

    for oid in ("near_jam_jar", "near_candle"):
        p0 = ep0["task_data"]["initial_layout"][oid]["position"]
        p1 = ep1["task_data"]["initial_layout"][oid]["position"]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]

        leaf0 = next(
            leaf["sub_goal_setting"]
            for stage in ep0["task_data"]["goal"]
            for union in stage
            for leaf in union
            if leaf["sub_goal_setting"]["obj1_uid"] == oid
        )
        leaf1 = next(
            leaf["sub_goal_setting"]
            for stage in ep1["task_data"]["goal"]
            for union in stage
            for leaf in union
            if leaf["sub_goal_setting"]["obj1_uid"] == oid
        )
        assert abs((leaf1["x_range"][0] - leaf0["x_range"][0]) - dx) < 1e-6
        assert abs((leaf1["x_range"][1] - leaf0["x_range"][1]) - dx) < 1e-6
        assert abs((leaf1["y_range"][0] - leaf0["y_range"][0]) - dy) < 1e-6
        assert abs((leaf1["y_range"][1] - leaf0["y_range"][1]) - dy) < 1e-6

        # Relative goal leaves for cup must NOT be shifted when cup moves? Wait: cup relative goal stays identical;
        # only absolute initial_pose goals shift! Check cup relative range unchanged.
    cup_leaf0 = copy.deepcopy(ep0["task_data"]["goal"][0][0][0]["sub_goal_setting"])
    cup_leaf1 = ep1["task_data"]["goal"][0][0][0]["sub_goal_setting"]
    assert cup_leaf1["x_range"] == cup_leaf0["x_range"]
    assert cup_leaf1["y_range"] == cup_leaf0["y_range"]

    # Check AABB non-overlap across all pairs in seed_0001 and seed_0002
    for s in ("seed_0001", "seed_0002"):
        ep = json.loads((out_dir / "ce_01_home_espresso_brew_v2" / s / "episode_metadata.json").read_text())
        layout = ep["task_data"]["initial_layout"]
        ids = ["hero_rig", "cup", "near_jam_jar", "near_candle"]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = layout[ids[i]], layout[ids[j]]
                ax, ay = a["position"][:2]
                bx, by = b["position"][:2]
                ha = a["footprint_xy"]
                hb = b["footprint_xy"]
                gap_x = abs(ax - bx) - 0.5 * (ha[0] + hb[0])
                gap_y = abs(ay - by) - 0.5 * (ha[1] + hb[1])
                assert max(gap_x, gap_y) >= 0.015


def test_split_modes_isolate_c_b_i_axes_and_hold_out_prompts(tmp_path: Path) -> None:
    mod = _load_script_module()
    assert len(mod.INSTRUCTION_PARAPHRASES["ce_01_home_espresso_brew_v2"]) >= 6
    assert len(mod.INSTRUCTION_PARAPHRASES["ce_12_home_wireless_uv_dock_v2"]) >= 6

    pkg = _make_collected_fixture(tmp_path, "ce_01_home_espresso_brew_v2")
    base_ep = json.loads((pkg / "tasks/scenario_forge/ce_01_home_espresso_brew_v2/ep_0.json").read_text())["task_data"]
    train_prompts = set(mod.INSTRUCTION_PARAPHRASES["ce_01_home_espresso_brew_v2"][:4])
    test_prompts = set(mod.INSTRUCTION_PARAPHRASES["ce_01_home_espresso_brew_v2"][4:])
    assert train_prompts.isdisjoint(test_prompts)

    out_c = tmp_path / "split_c"
    mod.generate_variants(collected_root=pkg, out_dir=out_c, num_episodes=2, start_seed=1000, split_mode="test_c_only")
    ep_c = json.loads((out_c / "ce_01_home_espresso_brew_v2/seed_1000/episode_metadata.json").read_text())["task_data"]
    assert ep_c["initial_layout"]["near_jam_jar"]["position"] == base_ep["initial_layout"]["near_jam_jar"]["position"]
    assert ep_c["initial_layout"]["cup"]["position"] != base_ep["initial_layout"]["cup"]["position"]
    assert ep_c["instruction"] in train_prompts

    out_b = tmp_path / "split_b"
    mod.generate_variants(collected_root=pkg, out_dir=out_b, num_episodes=2, start_seed=1005, split_mode="test_b_only")
    ep_b = json.loads((out_b / "ce_01_home_espresso_brew_v2/seed_1005/episode_metadata.json").read_text())["task_data"]
    assert ep_b["initial_layout"]["cup"]["position"] == base_ep["initial_layout"]["cup"]["position"]
    assert ep_b["initial_layout"]["near_jam_jar"]["position"] != base_ep["initial_layout"]["near_jam_jar"]["position"]
    assert ep_b["instruction"] in train_prompts

    out_i = tmp_path / "split_i"
    mod.generate_variants(collected_root=pkg, out_dir=out_i, num_episodes=2, start_seed=1010, split_mode="test_i_only")
    ep_i = json.loads((out_i / "ce_01_home_espresso_brew_v2/seed_1010/episode_metadata.json").read_text())["task_data"]
    assert ep_i["initial_layout"]["cup"]["position"] == base_ep["initial_layout"]["cup"]["position"]
    assert ep_i["initial_layout"]["near_jam_jar"]["position"] == base_ep["initial_layout"]["near_jam_jar"]["position"]
    assert ep_i["instruction"] in test_prompts
