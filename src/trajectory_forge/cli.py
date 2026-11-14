from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from trajectory_forge.adapters.isaac_genmanip.executor import (
    DEFAULT_CUROBO_SRC,
    DEFAULT_GENMANIP_ROOT,
    DEFAULT_ISAAC_PYTHON,
    DEFAULT_PACKAGE_STORE_ROOT,
    IsaacGenManipWorkerConfig,
    LocalIsaacGenManipExecutor,
)
from trajectory_forge.adapters.isaac_genmanip.ssh_executor import (
    SSHIsaacGenManipExecutor,
    SSHWorkerConfig,
)
from trajectory_forge.core.contracts import (
    RemediationPolicySpec,
    RobotSpawnSpec,
    RolloutJobRequest,
    ScenarioIntakeBrief,
    StageContractSpec,
)
from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.intake.genmanip_intake import (
    load_genmanip_collected_package,
    render_agent_brief_markdown,
)
from trajectory_forge.sdk.context import run_controller_preflight
from trajectory_forge.solver import (
    ControllerSynthesizer,
    OpenAICompatibleSynthesizer,
    PrivilegedSolverHarness,
    SolveSessionConfig,
    StaticControllerSynthesizer,
)
from trajectory_forge.worker.client import RolloutWorkerClient
from trajectory_forge.worker.server import (
    DryRunRolloutExecutor,
    RolloutExecutor,
    create_worker_server,
)


def _cmd_intake_inspect(args: argparse.Namespace) -> int:
    brief = load_genmanip_collected_package(Path(args.package))
    embodiment = (
        load_embodiment_card(Path(args.embodiment)) if args.embodiment else None
    )
    if args.format == "json":
        print(json.dumps(brief.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render_agent_brief_markdown(brief, embodiment))
    return 0


def _cmd_embodiment_check(args: argparse.Namespace) -> int:
    card = load_embodiment_card(Path(args.card))
    print(
        json.dumps(
            {
                "status": "ok",
                "embodiment_id": card.embodiment_id,
                "action_dim": card.action_dim,
                "chains": list(card.chains.keys()),
                "actuator_groups": list(card.actuator_groups.keys()),
                "sensor_streams": list(card.sensor_streams.keys()),
            },
            indent=2,
        )
    )
    return 0


def _cmd_controller_preflight(args: argparse.Namespace) -> int:
    card = load_embodiment_card(Path(args.embodiment))
    code = Path(args.controller).read_text(encoding="utf-8")
    if args.package:
        brief = load_genmanip_collected_package(Path(args.package))
    else:
        brief = ScenarioIntakeBrief(
            package_id="preflight_smoke",
            episode_name="0",
            instruction="Smoke preflight brief.",
            scene_usd_relpath="",
            robot_profile=card.runtime_robot_type,
            robot_spawn=RobotSpawnSpec(
                robot_id=card.embodiment_id,
                xyz=(0.0, -1.02, 0.31),
                wxyz=(0.7071067812, 0.0, 0.0, 0.7071067812),
                joint_positions=(0.0,) * card.action_dim,
            ),
            objects={},
            stages=(StageContractSpec(stage_index=0, metric_id="stage0_smoke", clauses=()),),
        )
    result = run_controller_preflight(
        code,
        brief,
        card,
        mode=args.mode,
        max_steps=args.max_steps,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _build_executor_from_args(args: argparse.Namespace) -> RolloutExecutor:
    backend = getattr(args, "backend", "dry-run")
    if backend == "dry-run":
        return DryRunRolloutExecutor()
    if backend == "isaac-genmanip":
        return LocalIsaacGenManipExecutor(
            IsaacGenManipWorkerConfig(
                isaac_python=Path(args.isaac_python),
                genmanip_root=Path(args.genmanip_root),
                package_store_root=Path(args.package_store),
                output_root=Path(args.output_root).resolve(),
                curobo_src=Path(args.curobo_src),
                timeout_s=float(args.timeout),
            )
        )
    if backend == "ssh-isaac-genmanip":
        return SSHIsaacGenManipExecutor(
            SSHWorkerConfig(
                ssh_host=args.ssh_host,
                ssh_user=args.ssh_user,
                ssh_port=int(args.ssh_port),
                ssh_key_path=Path(args.ssh_key).expanduser(),
                remote_isaac_python=Path(args.isaac_python),
                remote_genmanip_root=Path(args.genmanip_root),
                remote_curobo_src=Path(args.curobo_src),
                package_store_root=Path(args.package_store),
                output_root=Path(args.output_root).resolve(),
                timeout_s=float(args.timeout),
            )
        )
    if backend == "http-worker":
        if not getattr(args, "worker_url", None):
            raise SystemExit("--worker-url is required when --backend http-worker")
        token = args.token if getattr(args, "token", None) is not None else os.environ.get("TF_WORKER_TOKEN")
        client = RolloutWorkerClient(
            base_url=args.worker_url,
            auth_token=token or None,
            timeout_s=float(args.timeout),
        )
        return client.submit_rollout
    raise SystemExit(f"Unsupported backend: {backend}")


def _add_backend_arguments(parser: argparse.ArgumentParser, *, allow_http_worker: bool) -> None:
    choices = (
        ("dry-run", "isaac-genmanip", "ssh-isaac-genmanip", "http-worker")
        if allow_http_worker
        else ("dry-run", "isaac-genmanip", "ssh-isaac-genmanip")
    )
    parser.add_argument("--backend", choices=choices, default="dry-run")
    parser.add_argument("--ssh-host", default="10.150.168.71")
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519_github")
    parser.add_argument("--isaac-python", default=str(DEFAULT_ISAAC_PYTHON))
    parser.add_argument("--genmanip-root", default=str(DEFAULT_GENMANIP_ROOT))
    parser.add_argument("--curobo-src", default=str(DEFAULT_CUROBO_SRC))
    parser.add_argument("--package-store", default=str(DEFAULT_PACKAGE_STORE_ROOT))
    parser.add_argument("--output-root", default="external_artifacts/rollouts")
    parser.add_argument("--timeout", type=float, default=900.0)
    if allow_http_worker:
        parser.add_argument("--worker-url", default=None, help="HTTP worker base URL for http-worker.")
        parser.add_argument("--token", default=None, help="Bearer token (or set TF_WORKER_TOKEN).")


def _cmd_worker_serve(args: argparse.Namespace) -> int:
    token = args.token if args.token is not None else os.environ.get("TF_WORKER_TOKEN")
    executor = _build_executor_from_args(args)
    backend_name = (
        "dry_run_preflight"
        if args.backend == "dry-run"
        else args.backend.replace("-", "_")
    )
    server = create_worker_server(
        host=args.host,
        port=args.port,
        auth_token=token or None,
        executor=executor,
        backend_name=backend_name,
    )
    print(f"trajectory-forge worker ({backend_name}) listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _cmd_rollout_run(args: argparse.Namespace) -> int:
    card = load_embodiment_card(Path(args.embodiment))
    code = Path(args.controller).read_text(encoding="utf-8")
    pkg_path = Path(args.package)
    package_ref = str(pkg_path.resolve()) if pkg_path.exists() else args.package
    job_id = args.job_id or f"{pkg_path.name}_{args.mode}"
    req = RolloutJobRequest(
        job_id=job_id,
        scenario_package_ref=package_ref,
        embodiment_id=card.embodiment_id,
        mode=args.mode,
        seed=int(args.seed),
        controller_code=code,
        max_steps=int(args.max_steps),
        resume_from_snapshot_id=args.resume_snapshot,
        render_media=bool(args.render_media),
        remediation_policy=RemediationPolicySpec(
            mode=str(args.remediation_mode),
            allow_pose_adjustment=bool(args.allow_pose_adjustment),
        ),
    )
    executor = _build_executor_from_args(args)
    result = executor(req)
    rendered = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result.status == "succeeded" else 1


def _cmd_solve(args: argparse.Namespace) -> int:
    card = load_embodiment_card(Path(args.embodiment))
    pkg_path = Path(args.package)
    brief = load_genmanip_collected_package(pkg_path)
    package_ref = str(pkg_path.resolve())
    synthesizer: ControllerSynthesizer
    if args.llm:
        synthesizer = OpenAICompatibleSynthesizer(
            api_base=args.llm_api_base,
            model=args.llm_model,
        )
    elif args.controller:
        code = Path(args.controller).read_text(encoding="utf-8")
        synthesizer = StaticControllerSynthesizer(code)
    else:
        raise SystemExit("Either --controller or --llm must be provided to `solve`.")

    executor = _build_executor_from_args(args)
    harness = PrivilegedSolverHarness(
        executor=executor,
        preflight_enabled=(args.backend != "dry-run"),
    )
    cfg = SolveSessionConfig(
        max_attempts=int(args.max_attempts),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        synthesize_trajectory=not bool(args.no_synthesize),
        render_gate_media=bool(args.render_gate_media),
        render_synthesis_media=not bool(args.no_render_synthesis_media),
        remediation_policy=RemediationPolicySpec(
            mode=str(args.remediation_mode),
            allow_pose_adjustment=bool(args.allow_pose_adjustment),
        ),
    )
    report = harness.solve(
        brief=brief,
        embodiment=card,
        synthesizer=synthesizer,
        package_ref=package_ref,
        config=cfg,
    )
    rendered = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.succeeded else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trajectory-forge",
        description="Privileged manipulation solver and trajectory factory CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    intake = sub.add_parser("intake", help="Inspect upstream scenario-forge packages.")
    intake_sub = intake.add_subparsers(dest="subcommand", required=True)
    inspect_p = intake_sub.add_parser("inspect", help="Render scenario intake brief.")
    inspect_p.add_argument("--package", required=True, help="Collected package directory.")
    inspect_p.add_argument("--embodiment", help="Optional EmbodimentCard YAML.")
    inspect_p.add_argument("--format", choices=("markdown", "json"), default="markdown")
    inspect_p.set_defaults(func=_cmd_intake_inspect)

    emb = sub.add_parser("embodiment", help="Validate EmbodimentCard definitions.")
    emb_sub = emb.add_subparsers(dest="subcommand", required=True)
    emb_check = emb_sub.add_parser("check", help="Validate an EmbodimentCard YAML file.")
    emb_check.add_argument("card", help="Path to EmbodimentCard YAML.")
    emb_check.set_defaults(func=_cmd_embodiment_check)

    ctrl = sub.add_parser("controller", help="Validate and preflight Python controllers.")
    ctrl_sub = ctrl.add_subparsers(dest="subcommand", required=True)
    preflight = ctrl_sub.add_parser("preflight", help="AST-check and dry-run a controller.")
    preflight.add_argument("--controller", required=True, help="Path to controller .py file.")
    preflight.add_argument("--embodiment", required=True, help="Path to EmbodimentCard YAML.")
    preflight.add_argument("--package", help="Optional collected package directory.")
    preflight.add_argument(
        "--mode",
        choices=("privileged_search", "full_horizon_gate", "trajectory_synthesis"),
        default="full_horizon_gate",
    )
    preflight.add_argument("--max-steps", type=int, default=2000)
    preflight.set_defaults(func=_cmd_controller_preflight)

    worker = sub.add_parser("worker", help="Run the remote rollout worker HTTP service.")
    worker_sub = worker.add_subparsers(dest="subcommand", required=True)
    serve = worker_sub.add_parser("serve", help="Start HTTP worker server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--token", default=None, help="Bearer token (or set TF_WORKER_TOKEN).")
    _add_backend_arguments(serve, allow_http_worker=False)
    serve.set_defaults(func=_cmd_worker_serve)

    rollout = sub.add_parser("rollout", help="Run a single simulation or dry-run rollout.")
    rollout_sub = rollout.add_subparsers(dest="subcommand", required=True)
    run_p = rollout_sub.add_parser("run", help="Execute a RolloutJobRequest.")
    run_p.add_argument("--controller", required=True, help="Path to controller .py file.")
    run_p.add_argument("--package", required=True, help="Collected package directory or ref.")
    run_p.add_argument("--embodiment", required=True, help="Path to EmbodimentCard YAML.")
    run_p.add_argument(
        "--mode",
        choices=("privileged_search", "full_horizon_gate", "trajectory_synthesis"),
        default="privileged_search",
    )
    run_p.add_argument("--max-steps", type=int, default=2000)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--job-id", default=None)
    run_p.add_argument("--resume-snapshot", default=None)
    run_p.add_argument("--render-media", action="store_true")
    run_p.add_argument(
        "--remediation-mode",
        choices=("physics_tuning", "scene_codesign"),
        default="physics_tuning",
        help="Upstream scene remediation policy tier when diagnosing blockers.",
    )
    run_p.add_argument(
        "--allow-pose-adjustment",
        action="store_true",
        help="Under physics_tuning, allow non-destructive initial pose/overhang adjustments.",
    )
    run_p.add_argument("--out", default=None, help="Optional output path for result JSON.")
    _add_backend_arguments(run_p, allow_http_worker=True)
    run_p.set_defaults(func=_cmd_rollout_run)

    solve_p = sub.add_parser(
        "solve",
        help="Run closed-loop Embodied-First-Exam Privileged solve + gate + synthesis.",
    )
    solve_p.add_argument("--package", required=True, help="Collected package directory.")
    solve_p.add_argument("--embodiment", required=True, help="Path to EmbodimentCard YAML.")
    solve_p.add_argument("--controller", default=None, help="Static controller .py path.")
    solve_p.add_argument("--llm", action="store_true", help="Use OpenAICompatibleSynthesizer.")
    solve_p.add_argument("--llm-model", default=None)
    solve_p.add_argument("--llm-api-base", default=None)
    solve_p.add_argument("--max-attempts", type=int, default=5)
    solve_p.add_argument("--max-steps", type=int, default=2000)
    solve_p.add_argument("--seed", type=int, default=0)
    solve_p.add_argument("--render-gate-media", action="store_true")
    solve_p.add_argument("--no-render-synthesis-media", action="store_true")
    solve_p.add_argument("--no-synthesize", action="store_true")
    solve_p.add_argument(
        "--remediation-mode",
        choices=("physics_tuning", "scene_codesign"),
        default="physics_tuning",
        help="Upstream scene remediation policy tier when diagnosing blockers.",
    )
    solve_p.add_argument(
        "--allow-pose-adjustment",
        action="store_true",
        help="Under physics_tuning, allow non-destructive initial pose/overhang adjustments.",
    )
    solve_p.add_argument("--report-out", default=None, help="Optional report JSON path.")
    _add_backend_arguments(solve_p, allow_http_worker=True)
    solve_p.set_defaults(func=_cmd_solve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
