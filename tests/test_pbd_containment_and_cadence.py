from __future__ import annotations

import math

import pytest

from trajectory_forge.sdk.math3d import Pose3D
from trajectory_forge.trajectory.fluid_helpers import (
    compute_target_stride,
    evaluate_pbd_containment,
    points_in_vessel_cavity_mask,
    scale_ticks_for_physics_dt,
    upright_tilt_deg,
)


def test_compute_target_stride_and_tick_scaling() -> None:
    assert compute_target_stride(1.0 / 60.0, target_hz=15.0) == 4
    assert compute_target_stride(1.0 / 120.0, target_hz=15.0) == 8
    assert compute_target_stride(1.0 / 30.0, target_hz=15.0) == 2

    assert scale_ticks_for_physics_dt(40, 1.0 / 60.0) == 40
    assert scale_ticks_for_physics_dt(40, 1.0 / 120.0) == 80
    assert scale_ticks_for_physics_dt(20, 1.0 / 120.0) == 40
    assert scale_ticks_for_physics_dt(16, 1.0 / 120.0) == 32


def test_cylinder_and_elliptical_cavity_containment_with_rotation() -> None:
    # 90 deg rotation around Z swaps local X and Y in world space
    q_z90 = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    vessel_pose = Pose3D(xyz=(1.0, -0.5, 0.75), wxyz=q_z90)

    # Elliptical cavity: rx=0.070 (along local X -> world +Y), ry=0.050 (along local Y -> world -X)
    cavity = {
        "shape": "elliptical_cylinder",
        "rx": 0.070,
        "ry": 0.050,
        "z_min": 0.010,
        "z_max": 0.080,
    }
    pts = [
        (1.0, -0.5, 0.78),  # center -> inside
        (1.0, -0.5 + 0.062, 0.78),  # local x=+0.062 <= 0.070 -> inside
        (1.0 - 0.062, -0.5, 0.78),  # local y=+0.062 > 0.050 -> outside
        (1.0, -0.5, 0.74),  # local z=-0.010 < 0.010 -> outside (below floor)
    ]
    mask = points_in_vessel_cavity_mask(pts, vessel_pose, cavity)
    assert mask == (True, True, False, False)

    summary = evaluate_pbd_containment(
        pts,
        target_pose=vessel_pose,
        target_cavity=cavity,
    )
    assert summary["total"] == 4
    assert summary["target_count"] == 2
    assert summary["target_ratio"] == pytest.approx(0.5)
    assert summary["spill_count"] == 2
    assert summary["spill_ratio"] == pytest.approx(0.5)


def test_box_and_pour_allowed_vessels_spill_accounting() -> None:
    tank_pose = Pose3D(xyz=(0.0, 0.0, 0.75), wxyz=(1.0, 0.0, 0.0, 0.0))
    tank_cavity = {
        "shape": "box",
        "x_min": -0.095,
        "x_max": 0.095,
        "y_min": -0.060,
        "y_max": 0.080,
        "z_min": 0.090,
        "z_max": 0.270,
    }
    carafe_pose = Pose3D(xyz=(0.25, -0.15, 0.75), wxyz=(1.0, 0.0, 0.0, 0.0))
    carafe_cavity = {
        "shape": "cylinder",
        "r_bot": 0.036,
        "r_top": 0.040,
        "z_min": 0.012,
        "z_max": 0.160,
    }
    pts = [
        # 8 in tank
        *[(0.0, 0.0, 0.85 + 0.01 * i) for i in range(8)],
        # 1 left in carafe (not spilled!)
        (0.25, -0.15, 0.80),
        # 1 spilled on tabletop
        (0.12, -0.10, 0.755),
    ]
    summary = evaluate_pbd_containment(
        pts,
        target_pose=tank_pose,
        target_cavity=tank_cavity,
        allowed_vessels=((tank_pose, tank_cavity), (carafe_pose, carafe_cavity)),
    )
    assert summary["total"] == 10
    assert summary["target_count"] == 8
    assert summary["target_ratio"] == pytest.approx(0.8)
    assert summary["allowed_count"] == 9
    assert summary["spill_count"] == 1
    assert summary["spill_ratio"] == pytest.approx(0.1)


def test_upright_tilt_deg() -> None:
    assert upright_tilt_deg((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-6)
    # 30 deg tilt about X
    q_x30 = (math.cos(math.radians(15.0)), math.sin(math.radians(15.0)), 0.0, 0.0)
    assert upright_tilt_deg(q_x30) == pytest.approx(30.0, abs=1e-5)
