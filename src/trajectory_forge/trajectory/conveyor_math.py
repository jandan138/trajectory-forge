from __future__ import annotations

import math


def racetrack_perimeter(straight_half_len: float, turn_radius: float) -> float:
    """Return total closed perimeter of a stadium/racetrack curve in meters."""
    half_len = max(0.0, float(straight_half_len))
    radius = max(1e-6, float(turn_radius))
    return 4.0 * half_len + 2.0 * math.pi * radius


def eval_racetrack_pose(
    arc_s: float,
    straight_half_len: float,
    turn_radius: float,
    center_xy: tuple[float, float],
    z: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Evaluate (x, y, z, qw, qx, qy, qz) at arc-length ``arc_s`` along a closed racetrack.

    Segment layout (counter-clockwise in XY):
      - Segment 0 [0, 2L):                Front straight (-Y side), moving +X, yaw = 0
      - Segment 1 [2L, 2L + pi*R):        Right semicircle around (cx+L, cy), yaw = theta + pi/2
      - Segment 2 [2L + pi*R, 4L + pi*R): Back straight (+Y side), moving -X, yaw = pi
      - Segment 3 [4L + pi*R, P):         Left semicircle around (cx-L, cy), yaw = theta + pi/2
    """
    half_len = max(0.0, float(straight_half_len))
    radius = max(1e-6, float(turn_radius))
    cx, cy = float(center_xy[0]), float(center_xy[1])
    perim = racetrack_perimeter(half_len, radius)
    s = float(arc_s) % perim

    seg0 = 2.0 * half_len
    seg1 = seg0 + math.pi * radius
    seg2 = seg1 + 2.0 * half_len

    if s < seg0:
        x = cx - half_len + s
        y = cy - radius
        yaw = 0.0
    elif s < seg1:
        theta = -0.5 * math.pi + (s - seg0) / radius
        x = cx + half_len + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        yaw = theta + 0.5 * math.pi
    elif s < seg2:
        x = cx + half_len - (s - seg1)
        y = cy + radius
        yaw = math.pi
    else:
        theta = 0.5 * math.pi + (s - seg2) / radius
        x = cx - half_len + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        yaw = theta + 0.5 * math.pi

    half_yaw = 0.5 * yaw
    return (
        float(x),
        float(y),
        float(z),
        float(math.cos(half_yaw)),
        0.0,
        0.0,
        float(math.sin(half_yaw)),
    )
