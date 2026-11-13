from __future__ import annotations

import argparse
import copy
import json
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
from trajectory_forge.trajectory.manifest import build_trajectory_episode_manifest


def _resolve_collected_root(package_ref: str, package_store: Path) -> Path:
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


def _rewrite_collected_asset_paths(
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


def _write_result_atomic(result_path: Path, result: RolloutJobResult) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = result_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(result_path)


def _setup_studio_render_camera(scene: Any, evaluation: Mapping[str, Any]) -> Any:
    import numpy as np
    from pxr import Gf, UsdLux

    from genmanip.utils.loader.scene import create_camera_list
    from genmanip.utils.usd_utils.camera_utils import set_camera_look_at

    stage = scene.world.stage
    dome = UsdLux.DomeLight.Define(stage, "/World/TrajectoryForgeStudioDome")
    dome.CreateIntensityAttr(1100.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.99, 0.97))

    rig = (
        ("KeyFrontRight", (0.95, -1.25, 2.55), 11500.0, 0.50, (1.0, 0.98, 0.95)),
        ("FillFrontLeft", (-0.95, -1.15, 2.55), 8500.0, 0.55, (0.96, 0.98, 1.0)),
        ("WashRear", (0.0, 1.95, 2.75), 14500.0, 0.65, (1.0, 0.98, 0.95)),
    )
    for name, pos, intensity, radius, color in rig:
        light = UsdLux.SphereLight.Define(stage, f"/World/TrajectoryForge_{name}")
        light.CreateIntensityAttr(float(intensity))
        light.CreateRadiusAttr(float(radius))
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.AddTranslateOp().Set(Gf.Vec3d(*pos))

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
    for _ in range(35):
        scene.world.render()
    return camera


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
        _write_result_atomic(result_path, err)
        return 0

    repo_root = Path(__file__).resolve().parents[4]
    emb_path = repo_root / "configs" / "embodiments" / f"{req.embodiment_id}.yaml"
    embodiment = load_embodiment_card(emb_path)
    collected_root = _resolve_collected_root(req.scenario_package_ref, package_store_root)
    brief = load_genmanip_collected_package(collected_root)

    manifest = json.loads((collected_root / "package_manifest.json").read_text(encoding="utf-8"))
    entrypoints = manifest["entrypoints"]
    task_config = yaml.safe_load((collected_root / entrypoints["task_config"]).read_text(encoding="utf-8"))
    episode = json.loads((collected_root / entrypoints["episode_metadata"]).read_text(encoding="utf-8"))
    task_name = str(episode["task_name"])
    task_data = copy.deepcopy(episode["task_data"])
    _rewrite_collected_asset_paths(task_data, collected_root, str(manifest["package_id"]))

    matches = [
        copy.deepcopy(c) for c in task_config["evaluation_configs"] if c.get("task_name") == task_name
    ]
    if len(matches) != 1:
        raise ContractValidationError(f"Expected 1 evaluation_config for {task_name!r}, got {len(matches)}")
    evaluation = matches[0]
    evaluation["usd_name"] = str((collected_root / entrypoints["scene_usd"]).with_suffix(""))
    evaluation["domain_randomization"]["cameras"]["config_path"] = str(
        (collected_root / entrypoints["camera_config"]).resolve()
    )

    if str(genmanip_root.resolve()) not in sys.path:
        sys.path.insert(0, str(genmanip_root.resolve()))

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "multi_gpu": False})
    scene = None
    try:
        from genmanip.core.scene.scene import Scene
        from genmanip.core.scene.scene_config import SceneConfig
        from genmanip.utils.loader.domain_randomization import reset_scene
        from genmanip.utils.loader.scene import recovery_scene
        from genmanip.utils.standalone.file_utils import load_default_config
        import genmanip.extensions.metrics  # noqa: F401

        from trajectory_forge.adapters.isaac_genmanip.sim_backend import (
            IsaacGenManipSimBackend,
            configure_articulation_drives_at_reset,
            fix_lift2_finger_collider_offsets,
        )

        is_render = bool(req.render_media or req.mode == "trajectory_synthesis")
        default_config = load_default_config(
            str(genmanip_root.resolve()), "__trajectory_forge__.json", "local"
        )
        scene = Scene(SceneConfig(**evaluation))
        physics_dt = float(evaluation.get("physics_dt", 1.0 / 60.0))
        scene.initialize(
            default_config,
            physics_dt=physics_dt,
            rendering_dt=float(evaluation.get("rendering_dt", 1.0 / 60.0)),
            is_render=is_render,
            only_color_rep_for_camera=True,
        )
        scene.post_initialize()
        reset_scene(scene)

        recovered_task_data = copy.deepcopy(task_data)
        for pid in tuple(scene.articulation_part_list):
            recovered_task_data["initial_layout"].setdefault(pid, {"type": "articulation_part"})
        recovery_scene(scene, recovered_task_data, task_name, default_config)

        # In-memory USD fix for R5a finger collision restOffset >= contactOffset warnings
        if getattr(scene, "robot_list", None):
            fix_lift2_finger_collider_offsets(scene.world.stage, scene.robot_list[0].robot.prim_path)
        configure_articulation_drives_at_reset(scene, task_data.get("initial_layout", {}))

        for _ in range(40):
            scene.world.step(render=False)

        goal = task_data.get("goal", [])
        scene.build_metrics_manager(goal)
        render_cam = _setup_studio_render_camera(scene, evaluation) if is_render else None

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
            physics_dt=physics_dt,
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
        remediation_hint = ""
        if status == "succeeded" and final_score < 0.9999:
            status = "failed"
            failure_code, failure_detail, remediation_hint = backend.diagnose_blocking_stage(
                state_summary
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
            channels = ["proprio", "action", "base_target", "torso_lift", "timestamps_s"]
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
            diagnostics=SolvabilityDiagnostics(
                failure_code=failure_code,
                stage_index=state_summary.get("blocking_stage") if status != "succeeded" else None,
                detail=failure_detail,
                upstream_remediation_hint=remediation_hint,
            ),
            media=tuple(media_refs),
            trajectory_manifest=traj_manifest,
            saved_snapshots=tuple(backend.saved_snapshots),
            state_summary=state_summary,
        )
        _write_result_atomic(result_path, result)
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
        _write_result_atomic(result_path, err)
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
