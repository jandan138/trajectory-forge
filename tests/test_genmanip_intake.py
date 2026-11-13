from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.intake.genmanip_intake import (
    load_genmanip_collected_package,
    render_agent_brief_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
R3_PACKAGES_ROOT = Path(
    "/cpfs/user/zhuzihou/dev/scenario-forge/external_artifacts/"
    "out_of_domain_factory_r3_20261113/packages"
)


def test_load_synthetic_collected_package(tmp_path: Path) -> None:
    pkg = tmp_path / "collected_package"
    ep_path = pkg / "tasks" / "scenario_forge" / "demo_task" / "101" / "episode_metadata.json"
    ep_path.parent.mkdir(parents=True)

    manifest = {
        "schema_version": "scenario-forge-genmanip-collected-package/v0.1",
        "package_id": "demo_task",
        "entrypoints": {
            "episode_metadata": "tasks/scenario_forge/demo_task/101/episode_metadata.json",
            "scene_usd": "assets/scene_usds/scenario_forge/demo_task/scene.usda",
        },
        "runtime_requirements": {"runtime_robot_type": "manip/lift2/R5a"},
        "semantic_contract": {
            "authority": "episode_metadata",
            "json_pointer": "/task_data/scenario_forge_runtime_contract",
            "path": "tasks/scenario_forge/demo_task/101/episode_metadata.json",
        },
    }
    (pkg / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    episode_meta = {
        "episode_name": "101",
        "task_data": {
            "instruction": "Pull drawer and place cup.",
            "goal": [
                [[{"type": "manip/default/sr_based_genmanip_relationship", "obj1_uid": "hero"}]],
                [[{"type": "manip/default/sr_based_genmanip_range", "obj1_uid": "cup"}]],
            ],
            "initial_layout": {
                "lift2": {
                    "type": "robot",
                    "position": [0.0, -1.02, 0.31],
                    "orientation": [0.70710678, 0.0, 0.0, 0.70710678],
                    "joint_positions": [0.0] * 16,
                }
            },
            "scenario_forge_runtime_contract": {
                "execution": {
                    "progress_rubric": {
                        "unevaluated_metric_ids": ["stage0_drawer", "stage1_cup"]
                    }
                },
                "objects": [
                    {
                        "scenario_object_id": "hero",
                        "runtime_uid": "hero",
                        "role": "articulated_appliance",
                        "state_prim_path": "/World/demo_task/obj_hero",
                        "initial_pose": {"xyz": [0.0, 0.0, 0.75], "wxyz": [1.0, 0.0, 0.0, 0.0]},
                        "named_frames": {
                            "tray_target": {"xyz": [0.0, -0.1, 0.05], "wxyz": [1.0, 0.0, 0.0, 0.0]}
                        },
                        "articulation": {
                            "dof_mapping": [
                                {
                                    "dof_index": 0,
                                    "joint_prim": "/World/Hero/Joints/DrawerJoint",
                                    "semantic_joint": "drawer",
                                }
                            ],
                            "states": {"drawer": {"open": [-0.18, -0.14], "closed": [-0.01, 0.0]}},
                        },
                    }
                ],
            },
        },
    }
    ep_path.write_text(json.dumps(episode_meta), encoding="utf-8")

    brief = load_genmanip_collected_package(tmp_path)
    assert brief.package_id == "demo_task"
    assert brief.episode_name == "101"
    assert len(brief.stages) == 2
    assert brief.stages[0].metric_id == "stage0_drawer"
    assert "drawer" in brief.objects["hero"].dofs
    assert "tray_target" in brief.objects["hero"].named_frames

    card = load_embodiment_card(REPO_ROOT / "configs" / "embodiments" / "lift2_r5a_bimanual.yaml")
    md = render_agent_brief_markdown(brief, card)
    assert "demo_task" in md
    assert "lift2_r5a_bimanual" in md
    assert "tray_target" in md


@pytest.mark.local_artifacts
def test_intake_real_r3_out_of_domain_benchmark_packages() -> None:
    suite_cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "benchmarks" / "r3_out_of_domain_suite.yaml").read_text(
            encoding="utf-8"
        )
    )
    tasks = suite_cfg["tasks"]
    assert len(tasks) == 3

    expected = {
        "specialty_coffee_bar_creatista_prep": (
            6,
            "creatista_machine",
            {"brew_lever", "drip_tray", "select_dial", "dial_press"},
            {"capsule_chute", "cup_spout_target"},
        ),
        "electronics_qa_ingun_probe_lock": (
            5,
            "ingun_fixture",
            {"pressure_hood", "clamp_lever", "mode_toggle"},
            {"pcba_nest"},
        ),
        "modern_office_brother_calibrate": (
            5,
            "brother_printer",
            {"paper_cassette", "scanner_lid"},
            {"platen_center"},
        ),
    }

    for entry in tasks:
        pkg_id = entry["package_id"]
        pkg_dir = R3_PACKAGES_ROOT / pkg_id
        assert pkg_dir.exists(), f"missing r3 package: {pkg_dir}"
        brief = load_genmanip_collected_package(pkg_dir)
        exp_stages, hero_id, exp_dofs, exp_frames = expected[pkg_id]
        assert brief.package_id == pkg_id
        assert len(brief.stages) == exp_stages
        assert len(brief.robot_spawn.joint_positions) == 16
        assert hero_id in brief.objects
        hero = brief.objects[hero_id]
        assert exp_dofs <= set(hero.dofs.keys())
        assert exp_frames <= set(hero.named_frames.keys())
