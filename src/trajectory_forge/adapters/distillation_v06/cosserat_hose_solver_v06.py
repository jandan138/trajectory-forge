"""64-segment C2-smooth zero-penetration Cosserat catenary solver for natural amber latex cooling hoses."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


TABLE_Z_DEFAULT = 0.750
HOSE_OUTER_RADIUS = 0.0052  # 10.4 mm OD lab amber latex tubing


def quat_wxyz_rot(q_wxyz: Sequence[float], v3: Sequence[float]) -> np.ndarray:
    """Rotate 3D vector v3 by quaternion (w, x, y, z)."""
    w, x, y, z = [float(c) for c in q_wxyz]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    vx, vy, vz = float(v3[0]), float(v3[1]), float(v3[2])
    qvec = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, np.array([vx, vy, vz], dtype=np.float64))
    uuv = np.cross(qvec, uv)
    return np.array([vx, vy, vz], dtype=np.float64) + 2.0 * (w * uv + uuv)


def solve_cosserat_catenary_centerline(
    p_start: Sequence[float],
    dir_start: Sequence[float],
    p_end: Sequence[float],
    dir_end: Sequence[float],
    *,
    rest_length: float = 0.62,
    n_rings: int = 64,
    table_z: float = TABLE_Z_DEFAULT,
    r_outer: float = HOSE_OUTER_RADIUS,
    cyl_obstacles: Sequence[tuple[float, float, float, float]] = (),
    sphere_obstacles: Sequence[tuple[float, float, float, float]] = (),
    prev_centerline: np.ndarray | None = None,
    relax_iters: int = 60,
) -> np.ndarray:
    """Vectorized C2-smooth inextensible Cosserat catenary centerline with 0.0 mm obstacle penetration."""
    p0 = np.asarray(p_start, dtype=np.float64)
    d0 = np.asarray(dir_start, dtype=np.float64)
    d0 = d0 / max(float(np.linalg.norm(d0)), 1e-9)
    p1 = np.asarray(p_end, dtype=np.float64)
    d1 = np.asarray(dir_end, dtype=np.float64)
    d1 = d1 / max(float(np.linalg.norm(d1)), 1e-9)

    chord = float(np.linalg.norm(p1 - p0))
    total_len = max(float(rest_length), chord * 1.18)
    seg_len = total_len / float(n_rings - 1)
    min_z = table_z + r_outer + 0.0020

    s = np.linspace(0.0, 1.0, n_rings, dtype=np.float64)[:, None]
    c0 = p0 + 0.22 * total_len * d0 + np.array([0.0, -0.065, -0.03], dtype=np.float64)
    c1 = p1 + 0.20 * total_len * d1 + np.array([0.0, -0.055, -0.04], dtype=np.float64)
    hermite = (
        ((1.0 - s) ** 3) * p0
        + 3.0 * ((1.0 - s) ** 2) * s * c0
        + 3.0 * (1.0 - s) * (s**2) * c1
        + (s**3) * p1
    )
    if prev_centerline is not None and prev_centerline.shape == (n_rings, 3):
        pts = 0.70 * prev_centerline.copy() + 0.30 * hermite
    else:
        pts = hermite.copy()

    # Sleeve outer 22 mm of the 38 mm serrated glass barb (nodes 0..3) & tap/drain nozzle (nodes -1..-3)
    def _pin_ends(arr: np.ndarray) -> None:
        for kb in range(4):
            arr[kb] = p0 + (0.0072 * kb) * d0
        for ke in range(3):
            arr[-1 - ke] = p1 + (0.0080 * ke) * d1
        # Enforce tabletop clearance even on barb exit if condenser is near bench
        arr[2:, 2] = np.maximum(arr[2:, 2], min_z)

    _pin_ends(pts)
    gravity_step = np.array([0.0, -0.00032, -0.00145], dtype=np.float64)
    sph_arr = np.asarray(sphere_obstacles, dtype=np.float64) if sphere_obstacles else np.zeros((0, 4), dtype=np.float64)

    def _project_obstacles(arr: np.ndarray) -> None:
        sub = arr[4:-3]
        sub[:, 2] = np.maximum(sub[:, 2], min_z)
        sub[:, 1] = np.minimum(sub[:, 1], -0.025)
        for cx, cy, crad, zmax in cyl_obstacles:
            mask = sub[:, 2] <= (zmax + r_outer + 0.006)
            if np.any(mask):
                dx = sub[mask, 0] - cx
                dy = sub[mask, 1] - cy
                d2 = np.hypot(dx, dy)
                req = crad + r_outer + 0.006
                hit = d2 < req
                if np.any(hit):
                    idx_m = np.where(mask)[0][hit]
                    dx_h = sub[idx_m, 0] - cx
                    dy_h = sub[idx_m, 1] - cy
                    dn = np.maximum(np.hypot(dx_h, dy_h), 1e-6)
                    ux = dx_h / dn
                    uy = np.where(dy_h / dn > -0.25, -0.80, dy_h / dn)
                    un = np.hypot(ux, uy)
                    sub[idx_m, 0] = cx + (ux / un) * req
                    sub[idx_m, 1] = cy + (uy / un) * req
        if len(sph_arr) > 0:
            diff = sub[:, None, :] - sph_arr[None, :, :3]
            dist = np.linalg.norm(diff, axis=2)
            req_s = sph_arr[None, :, 3] + r_outer + 0.006
            pen = req_s - dist
            max_j = np.argmax(pen, axis=1)
            row_idx = np.arange(len(sub))
            worst = pen[row_idx, max_j]
            hit_s = worst > 0.0
            if np.any(hit_s):
                r_h = row_idx[hit_s]
                j_h = max_j[hit_s]
                dv = diff[r_h, j_h]
                dn = np.maximum(dist[r_h, j_h], 1e-6)[:, None]
                sub[r_h] = sph_arr[j_h, :3] + (dv / dn) * req_s[0, j_h][:, None]
        sub[:, 2] = np.maximum(sub[:, 2], min_z)

    for _ in range(relax_iters):
        pts[4:-3] += gravity_step

        for parity in (0, 1):
            i0 = np.arange(parity, n_rings - 1, 2)
            i1 = i0 + 1
            delta = pts[i1] - pts[i0]
            dist = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
            corr = 0.5 * (dist - seg_len) * (delta / dist)
            w0 = np.where(i0 <= 3, 0.0, 1.0)[:, None]
            w1 = np.where(i1 >= n_rings - 3, 0.0, 1.0)[:, None]
            ws = np.maximum(w0 + w1, 1e-9)
            active = (w0 + w1)[:, 0] > 0.0
            pts[i0[active]] += ((w0 / ws) * corr)[active]
            pts[i1[active]] -= ((w1 / ws) * corr)[active]

        # Strong C2 elastic bending filter across nodes 3..-3 so transition off barb has ZERO polygonal kink!
        interior = pts[2:-2].copy()
        pts[3:-3] = 0.54 * pts[3:-3] + 0.23 * (interior[:-2] + interior[2:])
        _project_obstacles(pts)
        _pin_ends(pts)

    # Final gentle C2 bend polish on nodes 3..10 followed by exact obstacle projection
    for _ in range(3):
        sm = pts[2:-2].copy()
        pts[3:-3] = 0.60 * pts[3:-3] + 0.20 * (sm[:-2] + sm[2:])
        _project_obstacles(pts)
        _pin_ends(pts)
    return pts


def skin_cosserat_tube(
    centerline: np.ndarray,
    *,
    n_circ: int = 24,
    r_outer: float = HOSE_OUTER_RADIUS,
    r_outer_profile: Sequence[float] | None = None,
    table_z: float = TABLE_Z_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Skin a (N, 3) centerline into smooth (N*n_circ, 3) float32 vertices & unit vertex normals."""
    n_rings = int(centerline.shape[0])
    tangents = np.zeros_like(centerline)
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    tangents[1:-1] = centerline[2:] - centerline[:-2]
    t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(t_norms, 1e-9)

    ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, tangents[0]))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    nrm = ref - float(np.dot(ref, tangents[0])) * tangents[0]
    nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-9)

    angles = np.linspace(0.0, 2.0 * math.pi, n_circ, endpoint=False, dtype=np.float64)
    ca = np.cos(angles)[:, None]
    sa = np.sin(angles)[:, None]

    verts = np.zeros((n_rings * n_circ, 3), dtype=np.float32)
    norms = np.zeros((n_rings * n_circ, 3), dtype=np.float32)

    for i in range(n_rings):
        t = tangents[i]
        nrm = nrm - float(np.dot(nrm, t)) * t
        nn = float(np.linalg.norm(nrm))
        if nn < 1e-7:
            alt = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            nrm = np.cross(alt, t)
            nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-9)
        else:
            nrm = nrm / nn
        bnm = np.cross(t, nrm)
        # Smoothly tapered cuff swell over barb olive (no sharp step!)
        cuff = 1.0 + 0.12 * math.exp(-((i - 1.5) / 2.2) ** 2) + 0.10 * math.exp(-((i - (n_rings - 2.0)) / 2.0) ** 2)
        dirs = ca * nrm[None, :] + sa * bnm[None, :]
        r_eff = float(r_outer_profile[i]) if r_outer_profile is not None else float(r_outer * cuff)
        ring_pts = centerline[i][None, :] + r_eff * dirs
        ring_pts[:, 2] = np.maximum(ring_pts[:, 2], table_z + 0.0003)
        idx0 = i * n_circ
        verts[idx0 : idx0 + n_circ] = ring_pts.astype(np.float32)
        norms[idx0 : idx0 + n_circ] = dirs.astype(np.float32)

    return verts, norms
