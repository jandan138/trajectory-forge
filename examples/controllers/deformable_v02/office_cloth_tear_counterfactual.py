"""Unprotected tear/over-stretch counterfactual controller for Task 2."""

from __future__ import annotations

import math
from typing import Mapping

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


class OfficeClothTearController:
    """Deliberately snags & yanks opposite cloth edge without wrench guard -> edge_stretch >= 0.48."""

    controller_id = "office_cloth_tear_counterfactual"
    total_steps = 115
    dt_ctrl = 0.01

    def __init__(self, brief: ScenarioIntakeBrief) -> None:
        self.brief = brief
        self.base_pos = np.asarray(brief.robot_spawn.xyz or DEFAULT_PANDA_BASE_POS, dtype=np.float64)
        self.base_quat = np.asarray(brief.robot_spawn.wxyz or DEFAULT_PANDA_BASE_QUAT_WXYZ, dtype=np.float64)
        cloth = brief.objects.get("wiping_cloth")
        self.cloth_xyz = np.asarray(cloth.initial_pose.xyz if cloth else [-0.13, -0.02, 0.758], dtype=np.float64)
        self._q_prev = PANDA_HOME_Q.copy()

    def step(self, obs: Mapping[str, object]) -> np.ndarray:
        idx = int(obs["step_idx"])
        c0 = self.cloth_xyz
        yaw = 0.5 * math.pi

        if idx < 18:
            a = idx / 17.0
            pos = _lerp(np.array([c0[0], c0[1] - 0.04, 0.87]), np.array([c0[0], c0[1], c0[2] + 0.015]), a)
            finger = 0.036
        elif idx < 38:
            a = (idx - 18) / 19.0
            pos = np.array([c0[0], c0[1], c0[2] + 0.015])
            finger = 0.036 - a * 0.029  # pinch cloth crest cleanly before lifting over slot
        elif idx < 52:
            a = (idx - 38) / 13.0
            pos = _lerp(np.array([c0[0], c0[1], c0[2] + 0.015]), np.array([c0[0], c0[1], 0.855]), a)
            finger = 0.007
        elif idx < 74:
            a = (idx - 52) / 21.0
            pos = _lerp(np.array([c0[0], c0[1], 0.855]), np.array([0.040, 0.04, 0.788]), a)
            finger = 0.007
        else:
            a = min(1.0, (idx - 74) / 35.0)
            # Unguarded downward plunge onto high-friction slot lip & blotter (peak force >> 68 N)
            pos = _lerp(np.array([0.040, 0.04, 0.788]), np.array([0.040, 0.04, 0.742]), a)
            finger = 0.001

        q7 = solve_ik_panda(
            pos,
            gripper_rot_for_grasp(yaw),
            q_init=self._q_prev,
            base_pos=self.base_pos,
            base_quat_wxyz=self.base_quat,
            max_iters=45,
        )
        self._q_prev = q7
        return np.concatenate([q7, np.array([finger, finger], dtype=np.float64)])
