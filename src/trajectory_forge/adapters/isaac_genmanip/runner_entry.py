from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import yaml

from trajectory_forge.core.contracts import (
    ContractValidationError,
    ControllerSafetyError,
    FirewallViolationError,
    RolloutJobRequest,
    RolloutJobResult,
    SolvabilityDiagnostics,
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


def resolve_collected_root(package_ref: str, package_store: Path) -> Path:
    ref_path = Path(package_ref)
    if ref_path.is_dir():
        if (ref_path / "package_manifest.json").exists():
            return ref_path.resolve()
        if (ref_path / "collected_package" / "package_manifest.json").exists():
            return (ref_path / "collected_package").resolve()
    cand = package_store / package_ref / "collected_package"
    if (cand / "package_manifest.json").exists():
        return cand.resolve()
    cand_direct = package_store / package_ref
    if (cand_direct / "package_manifest.json").exists():
        return cand_direct.resolve()
    raise ContractValidationError(
        f"Could not resolve collected package {package_ref!r} under {package_store}"
    )


def rewrite_collected_asset_paths(
    task_data: dict[str, Any],
    collected_root: Path,
    package_id: str,
) -> None:
    prefix = f"collected_packages/{package_id}/"
    for entry in task_data.get("initial_layout", {}).values():
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path")
        if isinstance(raw_path, str) and raw_path.startswith(prefix):
            rel = raw_path[len(prefix) :]
            entry["path"] = str((collected_root / rel).resolve())


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _wrist_camera_spec(name: str, prim_path: str) -> dict[str, Any]:
    return {
        "exists": True,
        "frequency": 60,
        "name": name,
        "camera_params": [433.89, 433.38, 322.79, 243.14],
        "position": [0.07, 0.01, 0.08],
        "orientation": [0.62, 0.33, -0.33, -0.62],
        "prim_path": prim_path,
        "resolution": [640, 480],
        "pixel_size": 3.0,
        "f_number": 2.0,
        "focus_distance": 0.6,
        "with_distance": False,
        "with_semantic": False,
        "with_bbox2d": False,
        "with_bbox3d": False,
        "with_motion_vector": False,
        "camera_axes": "usd",
    }


def prepare_camera_config_file(
    orig_cfg_path: Path,
    *,
    need_wrist_cameras: bool,
    scratch_dir: Path,
) -> Path:
    """Return a camera YAML guaranteed to contain wrist cameras when requested."""
    if not need_wrist_cameras:
        return orig_cfg_path.resolve()
    raw = yaml.safe_load(orig_cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return orig_cfg_path.resolve()
    if "left_camera" in raw and "right_camera" in raw:
        return orig_cfg_path.resolve()
    augmented = copy.deepcopy(raw)
    augmented.setdefault(
        "left_camera",
        _wrist_camera_spec("left_camera", "/lift2/lift2/lift2/fl/link6/Camera"),
    )
    augmented.setdefault(
        "right_camera",
        _wrist_camera_spec("right_camera", "/lift2/lift2/lift2/fr/link6/Camera"),
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_path = scratch_dir / "three_camera_lift2.yml"
    out_path.write_text(yaml.safe_dump(augmented, sort_keys=False), encoding="utf-8")
    return out_path.resolve()


def _stage_uses_warm_home_lighting(stage: Any) -> bool:
    for prim in stage.Traverse():
        attr = prim.GetAttribute("home:warmInteriorLighting")
        if attr and attr.HasValue() and bool(attr.Get()):
            return True
    return False


def setup_studio_lighting(scene: Any) -> None:
    """Author neutral dome and 3-point sphere lights once on the active USD stage (warm-adjusted for home scenes)."""
    from pxr import Gf, UsdLux

    stage = scene.world.stage
    warm_home = _stage_uses_warm_home_lighting(stage)
    if not stage.GetPrimAtPath("/World/TrajectoryForgeStudioDome").IsValid():
        dome = UsdLux.DomeLight.Define(stage, "/World/TrajectoryForgeStudioDome")
        if warm_home:
            dome.CreateIntensityAttr(540.0)
            dome.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
        else:
            dome.CreateIntensityAttr(1100.0)
            dome.CreateColorAttr(Gf.Vec3f(1.0, 0.99, 0.97))

    rig = (
        (
            ("KeyFrontRight", (0.95, -1.25, 2.55), 4800.0, 0.55, (1.0, 0.94, 0.86)),
            ("FillFrontLeft", (-0.95, -1.15, 2.55), 3600.0, 0.60, (1.0, 0.96, 0.92)),
            ("WashRear", (0.0, 1.95, 2.75), 5500.0, 0.70, (1.0, 0.95, 0.88)),
        )
        if warm_home
        else (
            ("KeyFrontRight", (0.95, -1.25, 2.55), 11500.0, 0.50, (1.0, 0.98, 0.95)),
            ("FillFrontLeft", (-0.95, -1.15, 2.55), 8500.0, 0.55, (0.96, 0.98, 1.0)),
            ("WashRear", (0.0, 1.95, 2.75), 14500.0, 0.65, (1.0, 0.98, 0.95)),
        )
    )
    for name, pos, intensity, radius, color in rig:
        prim_path = f"/World/TrajectoryForge_{name}"
        if not stage.GetPrimAtPath(prim_path).IsValid():
            light = UsdLux.SphereLight.Define(stage, prim_path)
            light.CreateIntensityAttr(float(intensity))
            light.CreateRadiusAttr(float(radius))
            light.CreateColorAttr(Gf.Vec3f(*color))
            light.AddTranslateOp().Set(Gf.Vec3d(*pos))


def setup_overview_camera(scene: Any, evaluation: Mapping[str, Any]) -> Any:
    """Attach optional 3rd-person overview camera for MP4 progression diagnostics."""
    import numpy as np

    from genmanip.utils.loader.scene import create_camera_list
    from genmanip.utils.usd_utils.camera_utils import set_camera_look_at

    setup_studio_lighting(scene)
    cam_cfg = {
        "exists": False,
        "frequency": 60,
        "name": "trajectory_forge_overview_cam",
        "position": [0.0, 0.0, 1.0],
        "orientation": [1.0, 0.0, 0.0, 0.0],
        "prim_path": "_tf_overview_cam",
        "resolution": [1280, 720],
        "focal_length": 9.8,
        "horizontal_aperture": 10.0,
        "vertical_aperture": 5.625,
        "with_distance": False,
        "with_semantic": False,
        "with_bbox2d": False,
        "with_bbox3d": False,
        "with_motion_vector": False,
    }
    cam_map = create_camera_list(
        {"overview_cam": cam_cfg},
        scene.uuid,
        float(evaluation.get("rendering_dt", 1.0 / 60.0)),
        only_color_rep_for_camera=True,
    )
    camera = cam_map["overview_cam"]
    camera._frequency = -1
    set_camera_look_at(
        camera,
        np.array([0.0, -0.08, 0.92], dtype=float),
        distance=2.25,
        elevation=30.0,
        azimuth=-34.0,
    )
    return camera


def resolve_sensor_cameras(
    scene: Any,
    embodiment: Any,
    *,
    capture_wrist_cameras: bool,
) -> dict[str, Any]:
    """Bind scene RGB cameras according to EmbodimentCard stream names."""
    cam_list = getattr(scene, "camera_list", None) or {}
    want_wrist = capture_wrist_cameras or (
        "rgb_left_wrist" in embodiment.sensor_streams
        and "rgb_right_wrist" in embodiment.sensor_streams
    )
    want_overlook = "rgb_overlook" in embodiment.sensor_streams or want_wrist
    bound: dict[str, Any] = {}
    if want_overlook and "overlook_camera" in cam_list:
        cam = cam_list["overlook_camera"]
        cam._frequency = -1
        bound["rgb_overlook"] = cam
    if want_wrist:
        if "left_camera" in cam_list:
            cam_l = cam_list["left_camera"]
            cam_l._frequency = -1
            bound["rgb_left_wrist"] = cam_l
        if "right_camera" in cam_list:
            cam_r = cam_list["right_camera"]
            cam_r._frequency = -1
            bound["rgb_right_wrist"] = cam_r
    return bound


def warmup_camera_renders(scene: Any, count: int = 8) -> None:
    for _ in range(count):
        scene.world.render()


def run_job(
    *,
    request_path: Path,
    result_path: Path,
    genmanip_root: Path,
    package_store_root: Path,
    output_root: Path,
) -> int:
    req_data = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        req = RolloutJobRequest.from_dict(req_data)
        validate_controller_source(req.controller_code)
    except (ControllerSafetyError, FirewallViolationError, ContractValidationError) as exc:
        err = RolloutJobResult(
            job_id=str(req_data.get("job_id", "unknown")),
            status="rejected",
            mode=str(req_data.get("mode", "unknown")),
            final_score=0.0,
            steps_executed=0,
            snapshots_used=0,
            firewall_verified=False,
            diagnostics=SolvabilityDiagnostics(failure_code=type(exc).__name__, detail=str(exc)),
        )
        write_json_atomic(result_path, err.to_dict())
        return 0

    repo_root = Path(__file__).resolve().parents[4]
    emb_path = repo_root / "configs" / "embodiments" / f"{req.embodiment_id}.yaml"
    embodiment = load_embodiment_card(emb_path)
    collected_root = resolve_collected_root(req.scenario_package_ref, package_store_root)
    brief = load_genmanip_collected_package(collected_root)

    manifest = json.loads((collected_root / "package_manifest.json").read_text(encoding="utf-8"))
    entrypoints = manifest["entrypoints"]
    task_config = yaml.safe_load((collected_root / entrypoints["task_config"]).read_text(encoding="utf-8"))
    episode = json.loads((collected_root / entrypoints["episode_metadata"]).read_text(encoding="utf-8"))
    task_name = str(episode["task_name"])
    task_data = copy.deepcopy(episode["task_data"])
    rewrite_collected_asset_paths(task_data, collected_root, str(manifest["package_id"]))

    matches = [
        copy.deepcopy(c) for c in task_config["evaluation_configs"] if c.get("task_name") == task_name
    ]
    if len(matches) != 1:
        raise ContractValidationError(f"Expected 1 evaluation_config for {task_name!r}, got {len(matches)}")
    evaluation = matches[0]
    evaluation["usd_name"] = str((collected_root / entrypoints["scene_usd"]).with_suffix(""))
    want_wrist = bool(
        req.capture_wrist_cameras
        or (
            "rgb_left_wrist" in embodiment.sensor_streams
            and "rgb_right_wrist" in embodiment.sensor_streams
        )
    )
    want_sensor_cams = bool(
        req.mode == "trajectory_synthesis"
        and (want_wrist or "rgb_overlook" in embodiment.sensor_streams)
    )
    job_scratch = output_root / "jobs" / req.job_id
    evaluation["domain_randomization"]["cameras"]["config_path"] = str(
        prepare_camera_config_file(
            collected_root / entrypoints["camera_config"],
            need_wrist_cameras=want_sensor_cams and want_wrist,
            scratch_dir=job_scratch,
        )
    )

    if str(genmanip_root.resolve()) not in sys.path:
        sys.path.insert(0, str(genmanip_root.resolve()))

    import torch  # noqa: F401
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
        is_render = bool(req.render_media or req.mode == "trajectory_synthesis")
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
            is_render=is_render,
            only_color_rep_for_camera=True,
        )
        scene.post_initialize()
        if hasattr(scene.world, "is_playing") and scene.world.is_playing():
            scene.world.stop()
        if getattr(scene, "robot_list", None):
            fix_lift2_finger_collider_offsets(scene.world.stage, scene.robot_list[0].robot.prim_path)
        snapshot_and_reset_stage_pbd_particles(scene, task_data.get("initial_layout", {}))
        scene.world.reset()
        for robot, robot_cfg in zip(scene.robot_list, scene.scene_config.robots):
            robot.initialize(default_joint_positions=robot_cfg.default_joint_positions)
        for key, art in scene.articulation_list.items():
            if scene.articulation_data[key]["is_articulated"]:
                art._articulation_view.initialize()
        reset_scene(scene)

        recovered_task_data = copy.deepcopy(task_data)
        for pid in tuple(scene.articulation_part_list):
            recovered_task_data["initial_layout"].setdefault(pid, {"type": "articulation_part"})
        recovery_scene(scene, recovered_task_data, task_name, default_config)

        reset_articulation_velocities_and_base(scene)
        configure_articulation_drives_at_reset(scene, task_data.get("initial_layout", {}))
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
        if is_render:
            setup_studio_lighting(scene)
        render_cam = setup_overview_camera(scene, evaluation) if is_render else None
        sensor_cams = (
            resolve_sensor_cameras(scene, embodiment, capture_wrist_cameras=want_wrist)
            if want_sensor_cams
            else {}
        )
        if is_render:
            sync_stage_pbd_isosurface_materials(scene.world.stage)
            warmup_camera_renders(scene, 12)

        backend = IsaacGenManipSimBackend(
            job_id=req.job_id,
            brief=brief,
            embodiment=embodiment,
            mode=req.mode,
            scene=scene,
            task_data=task_data,
            output_root=output_root,
            max_steps=req.max_steps,
            render_camera=render_cam,
            sensor_cameras=sensor_cams,
            camera_stride=effective_camera_stride,
            physics_dt=physics_dt,
            metric_eval_stride=target_15hz_stride,
        )
        if req.resume_from_snapshot_id:
            backend.load_and_restore_snapshot_file(req.resume_from_snapshot_id)

        ctx = ControllerContext(
            brief=brief,
            embodiment=embodiment,
            mode=req.mode,
            backend=backend,
        )
        ns = restricted_globals(backend.controller_logs)
        failure_code = ""
        failure_detail = ""
        status = "succeeded"
        try:
            compiled = compile(req.controller_code, f"<controller:{req.job_id}>", "exec")
            exec(compiled, ns, ns)
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

        media_refs, artifact_relpath, state_summary = backend.finalize_outputs()
        final_score = float(backend.live_current_score())
        diagnostics = SolvabilityDiagnostics(
            failure_code=failure_code,
            stage_index=(
                state_summary.get("blocking_stage") if status != "succeeded" else None
            ),
            detail=failure_detail,
        )
        if status == "succeeded" and final_score < 0.9999:
            status = "failed"
            diagnostics = backend.diagnose_blocking_stage(
                state_summary, remediation_policy=req.remediation_policy
            )

        firewall_verified = (
            req.mode in {"full_horizon_gate", "trajectory_synthesis"}
            and backend.snapshots_used == 0
            and req.resume_from_snapshot_id is None
            and status == "succeeded"
        )
        traj_manifest = None
        if (
            req.mode == "trajectory_synthesis"
            and firewall_verified
            and artifact_relpath is not None
        ):
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
            if render_cam is not None:
                channels.append("rgb_head")
            traj_manifest = build_trajectory_episode_manifest(
                episode_id=f"{brief.package_id}_{req.job_id}",
                scenario_id=brief.package_id,
                embodiment_id=embodiment.embodiment_id,
                num_steps=max(1, backend.steps_executed),
                continuous_from_t0=True,
                privileged_channels_recorded=False,
                recorded_channels=tuple(channels),
                artifact_relpath=artifact_relpath,
            )

        result = RolloutJobResult(
            job_id=req.job_id,
            status=status,
            mode=req.mode,
            final_score=round(final_score, 6),
            steps_executed=backend.steps_executed,
            snapshots_used=backend.snapshots_used,
            firewall_verified=firewall_verified,
            stage_progression=tuple(backend.stage_progression),
            diagnostics=diagnostics,
            media=tuple(media_refs),
            trajectory_manifest=traj_manifest,
            saved_snapshots=tuple(backend.saved_snapshots),
            state_summary=state_summary,
        )
        write_json_atomic(result_path, result.to_dict())
        sys.stdout.flush()
        sys.stderr.flush()
        return 0
    except BaseException as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        err = RolloutJobResult(
            job_id=req.job_id,
            status="failed",
            mode=req.mode,
            final_score=0.0,
            steps_executed=0,
            snapshots_used=0,
            firewall_verified=False,
            diagnostics=SolvabilityDiagnostics(
                failure_code=type(exc).__name__,
                detail=f"{exc}\n{tb}",
            ),
        )
        write_json_atomic(result_path, err.to_dict())
        sys.stdout.flush()
        sys.stderr.flush()
        return 2
    finally:
        try:
            if scene is not None and getattr(scene, "world", None) is not None:
                scene.world.stop()
                scene.world.clear_instance()
        except Exception:  # noqa: BLE001
            pass
        simulation_app.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isaac Sim 4.1 + GenManip rollout runner.")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--genmanip-root", required=True, type=Path)
    parser.add_argument("--package-store", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_job(
        request_path=args.request,
        result_path=args.result,
        genmanip_root=args.genmanip_root,
        package_store_root=args.package_store,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
