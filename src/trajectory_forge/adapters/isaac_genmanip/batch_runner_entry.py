from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys
import threading
import traceback
from typing import Any, Sequence

import numpy as np
import yaml

from trajectory_forge.adapters.isaac_genmanip.runner_entry import (
    prepare_camera_config_file,
    resolve_collected_root,
    resolve_sensor_cameras,
    rewrite_collected_asset_paths,
    setup_overview_camera,
    setup_studio_lighting,
    warmup_camera_renders,
    write_json_atomic,
)
from trajectory_forge.core.contracts import (
    BatchEpisodeOutcome,
    BatchSynthesisReport,
    BatchSynthesisRequest,
    ContractValidationError,
    FirewallViolationError,
)
from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.intake.genmanip_intake import load_genmanip_collected_package
from trajectory_forge.sdk.context import ControllerContext, restricted_globals
from trajectory_forge.sdk.safety import validate_controller_source
from trajectory_forge.trajectory.fluid_helpers import (
    compute_target_stride,
    scale_ticks_for_physics_dt,
)
from trajectory_forge.trajectory.manifest import build_trajectory_episode_manifest


def _build_report(
    req: BatchSynthesisRequest,
    outcomes: list[BatchEpisodeOutcome],
) -> BatchSynthesisReport:
    accepted = sum(1 for item in outcomes if item.status == "succeeded")
    failed = len(outcomes) - accepted
    return BatchSynthesisReport(
        batch_id=req.batch_id,
        scenario_package_ref=req.scenario_package_ref,
        embodiment_id=req.embodiment_id,
        outcomes=tuple(outcomes),
        accepted_count=accepted,
        failed_count=failed,
    )


def run_batch(
    *,
    request_path: Path,
    result_path: Path,
    genmanip_root: Path,
    package_store_root: Path,
    output_root: Path,
) -> int:
    req_data = json.loads(request_path.read_text(encoding="utf-8"))
    req = BatchSynthesisRequest.from_dict(req_data)
    validate_controller_source(req.controller_code)
    compiled_ctrl = compile(req.controller_code, f"<batch:{req.batch_id}>", "exec")

    repo_root = Path(__file__).resolve().parents[4]
    emb_path = repo_root / "configs" / "embodiments" / f"{req.embodiment_id}.yaml"
    embodiment = load_embodiment_card(emb_path)
    collected_root = resolve_collected_root(req.scenario_package_ref, package_store_root)
    base_brief = load_genmanip_collected_package(collected_root)

    manifest = json.loads(
        (collected_root / "package_manifest.json").read_text(encoding="utf-8")
    )
    entrypoints = manifest["entrypoints"]
    task_config = yaml.safe_load(
        (collected_root / entrypoints["task_config"]).read_text(encoding="utf-8")
    )
    base_episode = json.loads(
        (collected_root / entrypoints["episode_metadata"]).read_text(encoding="utf-8")
    )
    task_name = str(base_episode["task_name"])
    matches = [
        copy.deepcopy(c)
        for c in task_config["evaluation_configs"]
        if c.get("task_name") == task_name
    ]
    if len(matches) != 1:
        raise ContractValidationError(
            f"Expected 1 evaluation_config for {task_name!r}, got {len(matches)}"
        )
    evaluation = matches[0]
    evaluation["usd_name"] = str(
        (collected_root / entrypoints["scene_usd"]).with_suffix("")
    )

    want_wrist = bool(
        req.capture_wrist_cameras
        or (
            "rgb_left_wrist" in embodiment.sensor_streams
            and "rgb_right_wrist" in embodiment.sensor_streams
        )
    )
    batch_scratch = output_root / "batches" / req.batch_id
    evaluation["domain_randomization"]["cameras"]["config_path"] = str(
        prepare_camera_config_file(
            collected_root / entrypoints["camera_config"],
            need_wrist_cameras=want_wrist,
            scratch_dir=batch_scratch,
        )
    )

    if str(genmanip_root.resolve()) not in sys.path:
        sys.path.insert(0, str(genmanip_root.resolve()))

    # Preload PyTorch/Pydantic before SimulationApp so Isaac Kit LD_LIBRARY_PATH hooks
    # do not break dynamic symbols inside container runtime mode.
    import numpy  # noqa: F401
    import pydantic  # noqa: F401
    from pydantic import BaseModel, Field  # noqa: F401
    import torch  # noqa: F401
    import typing_extensions  # noqa: F401
    from isaacsim import SimulationApp

    sim_gpu_raw = os.environ.get("CE_SIM_GPU_ID", "0").strip()
    sim_gpu = int(sim_gpu_raw) if sim_gpu_raw.isdigit() else 0
    if torch.cuda.is_available() and sim_gpu < torch.cuda.device_count():
        torch.cuda.set_device(sim_gpu)
    else:
        sim_gpu = 0
    try:
        import curobo.types.base as _curobo_base

        if getattr(_curobo_base.TensorDeviceType.__init__, "__defaults__", None):
            _curobo_base.TensorDeviceType.__init__.__defaults__ = (
                torch.device("cuda", sim_gpu),
                *_curobo_base.TensorDeviceType.__init__.__defaults__[1:],
            )
    except Exception:  # noqa: BLE001
        pass
    simulation_app = SimulationApp(
        {
            "headless": True,
            "multi_gpu": False,
            "active_gpu": sim_gpu,
            "physics_gpu": sim_gpu,
        }
    )
    import carb.settings

    _carb_settings = carb.settings.get_settings()
    _carb_settings.set_bool("/app/omni.graph.scriptnode/enable_opt_in", False)
    _carb_settings.set_bool("/app/omni.graph.scriptnode/opt_in", True)
    scene = None
    outcomes: list[BatchEpisodeOutcome] = []
    rc = 2
    try:
        from genmanip.core.scene.scene import Scene
        from genmanip.core.scene.scene_config import SceneConfig
        from genmanip.utils.loader.domain_randomization import reset_scene
        from genmanip.utils.loader.scene import recovery_scene
        from genmanip.utils.standalone.file_utils import load_default_config
        import genmanip.extensions.metrics  # noqa: F401

        from trajectory_forge.adapters.isaac_genmanip.pbd_metrics import (
            register_pbd_containment_metric,
            sync_stage_pbd_isosurface_materials,
        )
        from trajectory_forge.adapters.isaac_genmanip.sim_backend import (
            IsaacGenManipSimBackend,
            configure_articulation_drives_at_reset,
            fix_lift2_finger_collider_offsets,
            prelift_fixed_base_torso_at_reset,
            reset_articulation_velocities_and_base,
            snapshot_and_reset_stage_pbd_particles,
        )

        register_pbd_containment_metric()
        default_config = load_default_config(
            str(genmanip_root.resolve()), "__trajectory_forge__.json", "local"
        )
        scene = Scene(SceneConfig(**evaluation))
        physics_dt = float(evaluation.get("physics_dt", 1.0 / 60.0))
        target_15hz_stride = compute_target_stride(physics_dt, target_hz=15.0)
        effective_camera_stride = (
            target_15hz_stride
            if (req.camera_stride <= 0 or (req.camera_stride == 4 and target_15hz_stride != 4))
            else int(req.camera_stride)
        )
        settle_ticks = scale_ticks_for_physics_dt(40, physics_dt)
        scene.initialize(
            default_config,
            physics_dt=physics_dt,
            rendering_dt=float(evaluation.get("rendering_dt", physics_dt)),
            is_render=True,
            only_color_rep_for_camera=True,
        )
        scene.post_initialize()
        setup_studio_lighting(scene)
        overview_cam = (
            setup_overview_camera(scene, evaluation) if req.render_media else None
        )
        sensor_cams = resolve_sensor_cameras(
            scene,
            embodiment,
            capture_wrist_cameras=want_wrist,
        )

        for ep_idx, ep in enumerate(req.episodes):
            random.seed(ep.seed)
            np.random.seed(ep.seed)
            ep_doc: dict[str, Any]
            ep_brief = base_brief
            if ep.episode_metadata_relpath:
                cand = Path(ep.episode_metadata_relpath)
                ep_meta_path = cand if cand.is_absolute() else (collected_root / cand)
                ep_doc = json.loads(ep_meta_path.read_text(encoding="utf-8"))
                ep_brief = load_genmanip_collected_package(
                    collected_root, episode_metadata_path=ep_meta_path
                )
            else:
                ep_doc = copy.deepcopy(base_episode)

            task_data = copy.deepcopy(ep_doc["task_data"])
            rewrite_collected_asset_paths(
                task_data, collected_root, str(manifest["package_id"])
            )
            if ep.instruction_override:
                task_data["instruction"] = ep.instruction_override
                ep_brief = replace(ep_brief, instruction=ep.instruction_override)

            if hasattr(scene.world, "is_playing") and scene.world.is_playing():
                scene.world.stop()
            if getattr(scene, "robot_list", None):
                fix_lift2_finger_collider_offsets(
                    scene.world.stage, scene.robot_list[0].robot.prim_path
                )
            snapshot_and_reset_stage_pbd_particles(
                scene, task_data.get("initial_layout", {})
            )
            scene.world.reset()
            for robot, robot_cfg in zip(scene.robot_list, scene.scene_config.robots):
                robot.initialize(default_joint_positions=robot_cfg.default_joint_positions)
            for key, art in scene.articulation_list.items():
                if scene.articulation_data[key]["is_articulated"]:
                    art._articulation_view.initialize()

            reset_scene(scene)
            recovered_task_data = copy.deepcopy(task_data)
            for pid in tuple(scene.articulation_part_list):
                recovered_task_data["initial_layout"].setdefault(
                    pid, {"type": "articulation_part"}
                )
            recovery_scene(scene, recovered_task_data, task_name, default_config)

            reset_articulation_velocities_and_base(scene)
            configure_articulation_drives_at_reset(
                scene, task_data.get("initial_layout", {})
            )
            snapshot_and_reset_stage_pbd_particles(
                scene,
                task_data.get("initial_layout", {}),
                stop_world=False,
                use_live_host_pose=True,
            )
            for _ in range(settle_ticks):
                scene.world.step(render=False)
            if embodiment.embodiment_id == "lift2_r5a_fixed_base_16dof":
                prelift_fixed_base_torso_at_reset(scene, 0.46, physics_dt=physics_dt)

            goal = task_data.get("goal", [])
            scene.build_metrics_manager(goal)
            sync_stage_pbd_isosurface_materials(scene.world.stage)
            warmup_camera_renders(scene, 8)

            job_tag = f"{req.batch_id}_{ep.episode_id}"
            backend = IsaacGenManipSimBackend(
                job_id=job_tag,
                brief=ep_brief,
                embodiment=embodiment,
                mode="trajectory_synthesis",
                scene=scene,
                task_data=task_data,
                output_root=output_root,
                max_steps=req.max_steps,
                render_camera=overview_cam,
                sensor_cameras=sensor_cams,
                camera_stride=effective_camera_stride,
                episode_id=ep.episode_id,
                physics_dt=physics_dt,
                metric_eval_stride=target_15hz_stride,
            )
            ctx = ControllerContext(
                brief=ep_brief,
                embodiment=embodiment,
                mode="trajectory_synthesis",
                backend=backend,
            )
            ns = restricted_globals(backend.controller_logs)
            status = "succeeded"
            failure_code = ""
            failure_detail = ""
            try:
                exec(compiled_ctrl, ns, ns)
                solve_fn = ns.get("solve")
                if not callable(solve_fn):
                    raise ContractValidationError("Controller missing callable solve(ctx)")
                solve_fn(ctx)
            except FirewallViolationError as exc:
                status = "rejected"
                failure_code = "FirewallViolationError"
                failure_detail = str(exc)
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                failure_code = type(exc).__name__
                failure_detail = f"{exc}\n{traceback.format_exc(limit=4)}"

            _, artifact_relpath, state_summary = backend.finalize_outputs()
            final_score = float(backend.live_current_score())
            if status == "succeeded" and final_score < 0.9999:
                status = "failed"
                failure_code = "STAGE_INCOMPLETE"
                telemetry = {
                    "stage_scores": [round(float(s), 4) for s in state_summary.get("stage_scores", [])],
                    "joints": {
                        oid: {j: d.get("value") for j, d in jmap.items()}
                        for oid, jmap in (state_summary.get("articulation_joints_final") or {}).items()
                    },
                    "objs": {
                        oid: o.get("xyz")
                        for oid, o in (state_summary.get("objects_final") or {}).items()
                    },
                    "pbd_t0": state_summary.get("pbd_containment_t0") or [],
                    "pbd_final": state_summary.get("pbd_containment_final") or [],
                }
                failure_detail = (
                    f"final_score={final_score:.4f}; "
                    f"blocking_stage={state_summary.get('blocking_stage')}; "
                    f"telemetry={json.dumps(telemetry, separators=(',', ':'))}"
                )
            elif status == "failed" and failure_detail:
                telemetry = {
                    "stage_scores": [round(float(s), 4) for s in state_summary.get("stage_scores", [])],
                    "joints": {
                        oid: {j: d.get("value") for j, d in jmap.items()}
                        for oid, jmap in (state_summary.get("articulation_joints_final") or {}).items()
                    },
                    "objs": {
                        oid: o.get("xyz")
                        for oid, o in (state_summary.get("objects_final") or {}).items()
                    },
                    "pbd_t0": state_summary.get("pbd_containment_t0") or [],
                    "pbd_final": state_summary.get("pbd_containment_final") or [],
                }
                failure_detail = (
                    f"{failure_detail}\n"
                    f"telemetry={json.dumps(telemetry, separators=(',', ':'))}"
                )

            if (
                status == "succeeded"
                and want_wrist
                and not {"rgb_overlook", "rgb_left_wrist", "rgb_right_wrist"}.issubset(
                    set(backend.recorded_sensor_streams)
                )
            ):
                status = "failed"
                failure_code = "MISSING_RGB_STREAMS"
                failure_detail = f"recorded_sensor_streams={backend.recorded_sensor_streams}"

            firewall_verified = (
                status == "succeeded"
                and backend.snapshots_used == 0
                and final_score >= 0.9999
                and artifact_relpath is not None
            )
            traj_manifest = None
            if firewall_verified and artifact_relpath is not None:
                channels = [
                    "proprio",
                    "action",
                    "base_target",
                    "torso_lift",
                    "timestamps_s",
                    "state_ee_pose",
                    "action_ee_pose",
                ]
                if backend.recorded_sensor_streams:
                    channels.append("camera_step_indices")
                    channels.extend(sorted(backend.recorded_sensor_streams))
                if overview_cam is not None:
                    channels.append("rgb_head")
                traj_manifest = build_trajectory_episode_manifest(
                    episode_id=ep.episode_id,
                    scenario_id=ep_brief.package_id,
                    embodiment_id=embodiment.embodiment_id,
                    num_steps=max(1, backend.steps_executed),
                    continuous_from_t0=True,
                    privileged_channels_recorded=False,
                    recorded_channels=tuple(channels),
                    artifact_relpath=artifact_relpath,
                )
            elif artifact_relpath is not None:
                Path(artifact_relpath).unlink(missing_ok=True)

            outcomes.append(
                BatchEpisodeOutcome(
                    episode_id=ep.episode_id,
                    seed=ep.seed,
                    status=status,
                    final_score=round(final_score, 6),
                    steps_executed=backend.steps_executed,
                    snapshots_used=backend.snapshots_used,
                    firewall_verified=firewall_verified,
                    trajectory_manifest=traj_manifest,
                    failure_code=failure_code,
                    detail=failure_detail,
                )
            )
            write_json_atomic(result_path, _build_report(req, outcomes).to_dict())
            sys.stdout.flush()
            sys.stderr.flush()

        rc = 0
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        rc = 2
        return 2
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        _close_watchdog = threading.Timer(15.0, lambda: os._exit(rc))
        _close_watchdog.daemon = True
        _close_watchdog.start()
        try:
            try:
                if scene is not None and getattr(scene, "world", None) is not None:
                    scene.world.stop()
                    scene.world.clear_instance()
            except Exception:  # noqa: BLE001
                pass
            simulation_app.close()
        finally:
            _close_watchdog.cancel()
        if rc != 0:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(rc)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Isaac Sim 4.1 + GenManip warm batch synthesis runner."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--genmanip-root", required=True, type=Path)
    parser.add_argument("--package-store", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_batch(
        request_path=args.request,
        result_path=args.result,
        genmanip_root=args.genmanip_root,
        package_store_root=args.package_store,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
