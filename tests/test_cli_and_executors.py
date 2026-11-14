from __future__ import annotations

import json
from pathlib import Path

from trajectory_forge import cli
from trajectory_forge.adapters.isaac_genmanip.executor import (
    IsaacGenManipWorkerConfig,
    LocalIsaacGenManipExecutor,
    build_isaac_worker_env,
)
from trajectory_forge.adapters.isaac_genmanip.executor_stub import IsaacGenManipSubprocessStub
from trajectory_forge.adapters.isaac_genmanip.ssh_executor import (
    SSHIsaacGenManipExecutor,
    SSHWorkerConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_local_executor_build_command_and_env(tmp_path: Path) -> None:
    isaac_py = tmp_path / "conda" / "bin" / "python"
    isaac_py.parent.mkdir(parents=True)
    isaac_py.write_text("", encoding="utf-8")
    genmanip_root = tmp_path / "GenManip"
    genmanip_root.mkdir()
    curobo_src = tmp_path / "curobo" / "src"
    curobo_src.mkdir(parents=True)

    cfg = IsaacGenManipWorkerConfig(
        isaac_python=isaac_py,
        genmanip_root=genmanip_root,
        package_store_root=tmp_path / "pkgs",
        output_root=tmp_path / "out",
        curobo_src=curobo_src,
    )
    executor = LocalIsaacGenManipExecutor(cfg)
    req_json = tmp_path / "req.json"
    res_json = tmp_path / "res.json"
    cmd = executor.build_command(req_json, res_json)
    assert cmd[0] == str(isaac_py)
    assert cmd[1:4] == ["-u", "-m", "trajectory_forge.adapters.isaac_genmanip.runner_entry"]
    assert str(req_json) in cmd
    assert str(res_json) in cmd

    stub = IsaacGenManipSubprocessStub(cfg)
    assert stub.build_command(req_json, res_json) == cmd

    env = build_isaac_worker_env(
        isaac_python=isaac_py,
        trajectory_forge_src=REPO_ROOT / "src",
        genmanip_root=genmanip_root,
        curobo_src=curobo_src,
        omni_scratch_dir=tmp_path / "scratch",
        base_env={"PATH": "/usr/bin"},
    )
    assert str(curobo_src.resolve()) in env["PYTHONPATH"]
    assert str(genmanip_root.resolve()) in env["PYTHONPATH"]
    assert str((REPO_ROOT / "src").resolve()) in env["PYTHONPATH"]
    assert env["OMNI_USER_DATA_DIR"].endswith("omni_data")


def test_ssh_executor_build_commands(tmp_path: Path) -> None:
    cfg = SSHWorkerConfig(
        ssh_host="10.150.168.71",
        ssh_user="root",
        ssh_port=22,
        output_root=tmp_path / "rollouts",
    )
    executor = SSHIsaacGenManipExecutor(cfg)
    remote = executor.build_remote_shell_command(
        tmp_path / "req.json", tmp_path / "res.json", "job42"
    )
    assert "OMNI_USER_DATA_DIR=" in remote
    assert "ISAACSIM_CACHE_DIR=" in remote
    assert "trajectory_forge.adapters.isaac_genmanip.runner_entry" in remote
    assert "trap 'rm -rf" in remote

    ssh_cmd = executor.build_ssh_command(remote)
    assert ssh_cmd[0] == "ssh"
    assert "root@10.150.168.71" in ssh_cmd
    assert "BatchMode=yes" in ssh_cmd
    assert ssh_cmd[-1] == remote


def test_cli_parser_new_subcommands_and_dry_run_rollout(tmp_path: Path) -> None:
    parser = cli.build_parser()
    serve_args = parser.parse_args(
        ["worker", "serve", "--backend", "ssh-isaac-genmanip", "--port", "9999"]
    )
    assert serve_args.backend == "ssh-isaac-genmanip"
    assert serve_args.port == 9999

    solve_args = parser.parse_args(
        [
            "solve",
            "--package",
            "dummy_pkg",
            "--embodiment",
            str(REPO_ROOT / "configs/embodiments/lift2_r5a_bimanual.yaml"),
            "--controller",
            str(REPO_ROOT / "examples/sample_privileged_controller.py"),
            "--backend",
            "ssh-isaac-genmanip",
            "--max-attempts",
            "3",
            "--remediation-mode",
            "scene_codesign",
            "--allow-pose-adjustment",
        ]
    )
    assert solve_args.backend == "ssh-isaac-genmanip"
    assert solve_args.max_attempts == 3
    assert solve_args.remediation_mode == "scene_codesign"
    assert solve_args.allow_pose_adjustment is True

    rollout_args = parser.parse_args(
        [
            "rollout",
            "run",
            "--controller",
            str(REPO_ROOT / "examples/sample_privileged_controller.py"),
            "--package",
            "dummy_pkg",
            "--embodiment",
            str(REPO_ROOT / "configs/embodiments/lift2_r5a_bimanual.yaml"),
            "--remediation-mode",
            "physics_tuning",
            "--allow-pose-adjustment",
        ]
    )
    assert rollout_args.remediation_mode == "physics_tuning"
    assert rollout_args.allow_pose_adjustment is True

    out_json = tmp_path / "rollout_res.json"
    rc = cli.main(
        [
            "rollout",
            "run",
            "--controller",
            str(REPO_ROOT / "examples/sample_privileged_controller.py"),
            "--package",
            "dry_pkg",
            "--embodiment",
            str(REPO_ROOT / "configs/embodiments/lift2_r5a_bimanual.yaml"),
            "--mode",
            "full_horizon_gate",
            "--backend",
            "dry-run",
            "--out",
            str(out_json),
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["mode"] == "full_horizon_gate"
    assert payload["firewall_verified"] is True

