from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from trajectory_forge.core.contracts import (
    RobotSpawnSpec,
    ScenarioIntakeBrief,
    StageContractSpec,
)
from trajectory_forge.embodiment.registry import load_embodiment_card
from trajectory_forge.intake.genmanip_intake import (
    load_genmanip_collected_package,
    render_agent_brief_markdown,
)
from trajectory_forge.sdk.context import run_controller_preflight
from trajectory_forge.worker.server import create_worker_server


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


def _cmd_worker_serve(args: argparse.Namespace) -> int:
    token = args.token if args.token is not None else os.environ.get("TF_WORKER_TOKEN")
    server = create_worker_server(
        host=args.host,
        port=args.port,
        auth_token=token or None,
    )
    print(f"trajectory-forge worker listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


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
    serve.set_defaults(func=_cmd_worker_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
