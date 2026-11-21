"""Unprotected crush counterfactual controller for Task 1 (coffee_bar_fragile_pastry_plate)."""

from __future__ import annotations

import math
import numpy as np

from trajectory_forge.core.contracts import ScenarioIntakeBrief
from trajectory_forge.sdk.panda_kinematics import (
    DEFAULT_PANDA_BASE_POS,
    DEFAULT_PANDA_BASE_QUAT_WXYZ,
    PANDA_HOME_Q,
    gripper_rot_for_grasp,
    solve_ik_panda,
)


def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    t = float(min(max(alpha, 0.0), 1.0))
    return (1.0 - t) * a + t * b


class CoffeePastryCrushController:
    """Deliberately omits tactile guard: over-closes fingers (0.008m) & compresses pastry."""

    controller_id = "coffee_pastry_crush_counterfactual"
    total_steps = 145
    dt_ctrl = 0.01

    def __init__(self, brief: ScenarioIntakeBrief) -> None:
        self.brief = brief
        self.base_pos = np.asarray(brief.robot_spawn.xyz or DEFAULT_PANDA_BASE_POS, dtype=np.float64)
        self.base_quat = np.asarray(brief.robot_spawn.wxyz or DEFAULT_PANDA_BASE_QUAT_WXYZ, dtype=np.float64)
        pastry = brief.objects.get("fragile_pastry")
        self.pastry_xyz = np.asarray(pastry.initial_pose.xyz if pastry else [-0.18, -0.02, 0.766], dtype=np.float64)
        self._q_prev = PANDA_HOME_Q.copy()

    def step(self, obs: Mapping[str, object] | dict[str, object]) -> np.ndarray:
        idx = int(obs["step_idx"])
        p0 = self.pastry_xyz
        y_hinge = p0[1] + 0.064
        z_hinge = 0.807

        if idx < 10:
            a = idx / 9.0
            pos = _lerp(np.array([p0[0], p0[1] - 0.150, 0.835]), np.array([p0[0], p0[1] - 0.118, 0.782]), a)
            finger = 0.002
        elif idx < 18:
            a = (idx - 10) / 7.0
            pos = _lerp(np.array([p0[0], p0[1] - 0.118, 0.782]), np.array([p0[0], p0[1] - 0.090, 0.782]), a)
            finger = 0.002
        elif idx < 42:
            a = (idx - 18) / 23.0
            theta = a * math.radians(108.0)
            y_cmd = y_hinge - 0.134 * math.cos(theta) - 0.012 * math.sin(theta)
            z_cmd = z_hinge + 0.134 * math.sin(theta) - 0.025 * math.cos(theta)
            pos = np.array([p0[0], y_cmd, z_cmd], dtype=np.float64)
            finger = 0.002
        elif idx < 50:
            a = (idx - 42) / 7.0
            arc_end = np.array([p0[0], y_hinge - 0.134 * math.cos(math.radians(108.0)) - 0.012 * math.sin(math.radians(108.0)), z_hinge + 0.134 * math.sin(math.radians(108.0)) - 0.025 * math.cos(math.radians(108.0))])
            pos = _lerp(arc_end, np.array([p0[0], p0[1], 0.875]), a)
            finger = 0.040
        elif idx < 60:
            pos = np.array([p0[0], p0[1], 0.855])
            finger = 0.040
        elif idx < 76:
            a = (idx - 60) / 15.0
            pos = _lerp(np.array([p0[0], p0[1], 0.855]), np.array([p0[0], p0[1], 0.779]), a)
            finger = 0.040
        elif idx < 82:
            pos = np.array([p0[0], p0[1], 0.779])
            finger = 0.040
        else:
            a = (idx - 82) / 62.0
            # No tactile force guard -> crushes radially down to 0.006 (1.2 cm gap on 7 cm pastry!)
            pos = np.array([p0[0], p0[1], 0.779])
            finger = 0.040 - a * 0.034

        q7 = solve_ik_panda(
            pos,
            gripper_rot_for_grasp(0.0),
            q_init=self._q_prev,
            base_pos=self.base_pos,
            base_quat_wxyz=self.base_quat,
            max_iters=45,
        )
        self._q_prev = q7
        return np.concatenate([q7, np.array([finger, finger], dtype=np.float64)])
