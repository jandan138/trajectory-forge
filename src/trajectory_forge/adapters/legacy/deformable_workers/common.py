"""Shared numpy-only kinematics, strain diagnostics, and firewall-checked NPZ export."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from trajectory_forge.core.contracts import ScenarioIntakeBrief
from trajectory_forge.intake.deformable_v02_intake import load_deformable_v02_package
from trajectory_forge.trajectory.manifest import build_trajectory_episode_manifest


def apply_low_priority_process_policy() -> None:
    """Keep Apex Legends responsive while running local simulation workers."""
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                0x00000040,  # IDLE_PRIORITY_CLASS
            )
        except Exception:
            pass
    else:
        try:
            import os

            os.nice(19)
        except Exception:
            pass


def compute_tet_strains_from_points(
    rest_points: np.ndarray,
    cur_points: np.ndarray,
    tets: np.ndarray,
) -> tuple[float, float]:
    """Return (polar_stretch_q95 = q95(||U-I||_F), naive_f_minus_i_q95 = q95(||F-I||_F))."""
    if len(tets) == 0 or len(rest_points) == 0:
        return 0.0, 0.0
    p0_r = rest_points[tets[:, 0]]
    dm = np.stack(
        [
            rest_points[tets[:, 1]] - p0_r,
            rest_points[tets[:, 2]] - p0_r,
            rest_points[tets[:, 3]] - p0_r,
        ],
        axis=-1,
    )
    dets = np.linalg.det(dm)
    valid = np.abs(dets) > 1e-12
    if not np.any(valid):
        return 0.0, 0.0
    dm = dm[valid]
    tets_v = tets[valid]
    p0_c = cur_points[tets_v[:, 0]]
    ds = np.stack(
        [
            cur_points[tets_v[:, 1]] - p0_c,
            cur_points[tets_v[:, 2]] - p0_c,
            cur_points[tets_v[:, 3]] - p0_c,
        ],
        axis=-1,
    )
    F = ds @ np.linalg.inv(dm)
    ident = np.eye(3, dtype=np.float64)[None, :, :]
    naive = np.linalg.norm(F - ident, axis=(1, 2))
    # Polar stretch U via singular values of F: ||U - I||_F^2 = sum_k (s_k - 1)^2
    svals = np.linalg.svd(F, compute_uv=False)
    polar = np.linalg.norm(svals - 1.0, axis=1)
    return float(np.quantile(polar, 0.95)), float(np.quantile(naive, 0.95))


def compute_tet_strains_from_F(F_mats: np.ndarray) -> tuple[float, float]:
    """Compute (polar_stretch_q95, naive_f_minus_i_q95) directly from per-particle F (N,3,3)."""
    if F_mats is None or len(F_mats) == 0:
        return 0.0, 0.0
    f_arr = np.asarray(F_mats, dtype=np.float64)
    finite_mask = np.all(np.isfinite(f_arr), axis=(1, 2))
    if not np.any(finite_mask):
        return 1.0, 1.0
    f_arr = f_arr[finite_mask]
    ident = np.eye(3, dtype=np.float64)[None, :, :]
    naive = np.linalg.norm(f_arr - ident, axis=(1, 2))
    svals = np.linalg.svd(f_arr, compute_uv=False)
    polar = np.linalg.norm(svals - 1.0, axis=1)
    return float(np.quantile(polar, 0.95)), float(np.quantile(naive, 0.95))


def compute_edge_stretch_from_points(
    rest_points: np.ndarray,
    cur_points: np.ndarray,
    edges: np.ndarray,
) -> float:
    """Return 95th-percentile relative edge stretch |l - l0| / l0 across cloth mesh edges."""
    if len(edges) == 0 or len(rest_points) == 0:
        return 0.0
    l0 = np.linalg.norm(rest_points[edges[:, 1]] - rest_points[edges[:, 0]], axis=1)
    valid = l0 > 1e-9
    if not np.any(valid):
        return 0.0
    l1 = np.linalg.norm(cur_points[edges[valid, 1]] - cur_points[edges[valid, 0]], axis=1)
    stretch = np.abs(l1 - l0[valid]) / l0[valid]
    return float(np.quantile(stretch, 0.95))


def compute_damage_score(
    strain: float,
    onset_a: float,
    failure_b: float,
    *,
    peak_force_n: float = 0.0,
    max_normal_force_n: float = 68.0,
) -> float:
    """Map peak continuum strain and normal contact force to multi-modal damage D_total in [0, 1]."""
    if failure_b <= onset_a:
        d_strain = 1.0 if strain > onset_a else 0.0
    else:
        d_strain = float(np.clip((strain - onset_a) / (failure_b - onset_a), 0.0, 1.0))
    d_force = (
        float(np.clip((peak_force_n - max_normal_force_n) / max(0.5 * max_normal_force_n, 1.0), 0.0, 1.0))
        if peak_force_n > max_normal_force_n
        else 0.0
    )
    return max(d_strain, d_force)


def build_voxel_tet_grid(
    dimensions_xyz: tuple[float, float, float],
    grid_xyz: tuple[int, int, int] = (4, 4, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Create canonical 5-tet-per-hex mesh matching ConvertAsset / MuJoCo flexcomp."""
    sx, sy, sz = dimensions_xyz
    nx, ny, nz = grid_xyz
    xs = np.linspace(-0.5 * sx, 0.5 * sx, nx + 1)
    ys = np.linspace(-0.5 * sy, 0.5 * sy, ny + 1)
    zs = np.linspace(0.0, sz, nz + 1)
    verts = np.array(
        [(x, y, z) for z in zs for y in ys for x in xs],
        dtype=np.float64,
    )

    def vid(ix: int, iy: int, iz: int) -> int:
        return iz * ((ny + 1) * (nx + 1)) + iy * (nx + 1) + ix

    tets: list[tuple[int, int, int, int]] = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                p000 = vid(ix, iy, iz)
                p100 = vid(ix + 1, iy, iz)
                p010 = vid(ix, iy + 1, iz)
                p110 = vid(ix + 1, iy + 1, iz)
                p001 = vid(ix, iy, iz + 1)
                p101 = vid(ix + 1, iy, iz + 1)
                p011 = vid(ix, iy + 1, iz + 1)
                p111 = vid(ix + 1, iy + 1, iz + 1)
                if (ix + iy + iz) % 2 == 0:
                    tets.extend(
                        [
                            (p000, p100, p110, p101),
                            (p000, p110, p010, p011),
                            (p000, p001, p101, p011),
                            (p111, p110, p101, p011),
                            (p000, p110, p101, p011),
                        ]
                    )
                else:
                    tets.extend(
                        [
                            (p100, p110, p010, p111),
                            (p000, p100, p010, p001),
                            (p100, p101, p111, p001),
                            (p010, p111, p011, p001),
                            (p100, p010, p111, p001),
                        ]
                    )
    return verts, np.asarray(tets, dtype=np.int64)


def load_cloth_mesh_from_package(
    package_dir: Path,
    asset_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load localized self-contained cloth OBJ vertices, triangles, and edges."""
    root = Path(package_dir)
    if (root / "scenario_package" / "manifest.yaml").is_file():
        root = root / "scenario_package"
    obj_path = root / "assets" / asset_id / "engines" / "mujoco4" / "mesh.obj"
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    for raw in obj_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "f":
            a, b, c = [int(tok.split("/")[0]) - 1 for tok in parts[1:4]]
            tris.append((a, b, c))
    edge_set: set[tuple[int, int]] = set()
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_set.add((min(u, v), max(u, v)))
    return (
        np.asarray(verts, dtype=np.float64),
        np.asarray(tris, dtype=np.int64),
        np.asarray(sorted(edge_set), dtype=np.int64),
    )


def jitter_for_seed(seed: int) -> tuple[float, float, float]:
    """Deterministic small initial pose jitter (±2 mm XY, ±0.02 rad yaw) for seed > 0."""
    if int(seed) <= 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(int(seed) + 1013)
    dx = float(rng.uniform(-0.002, 0.002))
    dy = float(rng.uniform(-0.002, 0.002))
    dyaw = float(rng.uniform(-0.02, 0.02))
    return (dx, dy, dyaw)


def load_brief_and_controller(
    package_dir: Path,
    task_id: str,
    mode: str,
) -> tuple[ScenarioIntakeBrief, Any]:
    brief = load_deformable_v02_package(package_dir)
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.controllers.deformable_v02 import build_controller_for_mode

    ctrl = build_controller_for_mode(task_id, mode, brief)
    return brief, ctrl


def save_episode_bundle(
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
    """Persist NPZ arrays + firewall-checked TrajectoryEpisodeManifest + JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{episode_id}.npz"
    np.savez_compressed(npz_path, **npz_arrays)

    num_steps = int(npz_arrays["action"].shape[0])
    recorded_channels = tuple(sorted(npz_arrays.keys()))
    manifest = build_trajectory_episode_manifest(
        episode_id=episode_id,
        scenario_id=task_id,
        embodiment_id="franka_panda_tabletop",
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
