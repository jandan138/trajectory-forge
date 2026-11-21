"""Zero-edit wrench-guarded cloth wipe & park controller for Task 2."""

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


class OfficeClothWipeGentleController:
    """Grasp arched microfiber cloth, wipe 3 targets with bounded downward wrench, park in caddy."""

    controller_id = "office_cloth_wipe_gentle_controller"
    total_steps = 165
    dt_ctrl = 0.01

    def __init__(self, brief: ScenarioIntakeBrief) -> None:
        self.brief = brief
        self.base_pos = np.asarray(brief.robot_spawn.xyz or DEFAULT_PANDA_BASE_POS, dtype=np.float64)
        self.base_quat = np.asarray(brief.robot_spawn.wxyz or DEFAULT_PANDA_BASE_QUAT_WXYZ, dtype=np.float64)
        cloth = brief.objects.get("wiping_cloth")
        caddy = brief.objects.get("cloth_caddy_tray")
        self.cloth_xyz = np.asarray(cloth.initial_pose.xyz if cloth else [-0.13, -0.02, 0.758], dtype=np.float64)
        self.caddy_xyz = np.asarray(caddy.initial_pose.xyz if caddy else [0.22, -0.04, 0.755], dtype=np.float64)
        self.wipe_pts = [
            np.array([-0.13, 0.04, self.cloth_xyz[2] + 0.018], dtype=np.float64),
            np.array([-0.04, 0.07, self.cloth_xyz[2] + 0.018], dtype=np.float64),
            np.array([0.05, 0.04, self.cloth_xyz[2] + 0.018], dtype=np.float64),
        ]
        self._q_prev = PANDA_HOME_Q.copy()
        self._locked_finger_q: float | None = None

    def step(self, obs: Mapping[str, object]) -> np.ndarray:
        idx = int(obs["step_idx"])
        wrench = np.asarray(obs.get("wrist_wrench", np.zeros(6)), dtype=np.float64)
        tactile = np.asarray(obs.get("tactile_normal_force_n", [0.0, 0.0]), dtype=np.float64)
        avg_fn = float(0.5 * (tactile[0] + tactile[1]))
        downward_force = abs(float(wrench[2]))

        c0 = self.cloth_xyz
        cad = self.caddy_xyz
        # Align fingers along +/-X across cloth arch ridge (yaw=+90 deg)
        yaw = 0.5 * math.pi
        finger = 0.008

        if idx < 18:
            a = idx / 17.0
            pos = _lerp(np.array([c0[0], c0[1] - 0.04, 0.87]), np.array([c0[0], c0[1], c0[2] + 0.015]), a)
            finger = 0.036
        elif idx < 38:
            a = (idx - 18) / 19.0
            pos = np.array([c0[0], c0[1], c0[2] + 0.015])
            cmd_f = 0.036 - a * 0.029  # 0.007 gentle gap -> true friction grasp around stand spine (0.002)
            if avg_fn >= 6.0 and self._locked_finger_q is None:
                q_obs = np.asarray(obs.get("qpos", np.zeros(9)), dtype=np.float64)
                self._locked_finger_q = float(max(0.006, min(0.010, 0.5 * (q_obs[7] + q_obs[8]))))
            finger = self._locked_finger_q if self._locked_finger_q is not None else cmd_f
        elif idx < 52:
            a = (idx - 38) / 13.0
            # Lift straight up vertically BEFORE lateral sweep
            pos = _lerp(np.array([c0[0], c0[1], c0[2] + 0.015]), np.array([c0[0], c0[1], 0.855]), a)
            finger = self._locked_finger_q if self._locked_finger_q is not None else 0.007
        elif idx < 64:
            a = (idx - 52) / 11.0
            z_bias = 0.004 if downward_force > 12.0 else 0.0
            pos = _lerp(np.array([c0[0], c0[1], 0.855]), self.wipe_pts[0] + np.array([0.0, 0.0, z_bias]), a)
            finger = self._locked_finger_q if self._locked_finger_q is not None else 0.007
        elif idx < 82:
            a = (idx - 64) / 17.0
            z_bias = 0.004 if downward_force > 12.0 else 0.0
            pos = _lerp(self.wipe_pts[0], self.wipe_pts[1], a) + np.array([0.0, 0.0, z_bias])
            finger = self._locked_finger_q if self._locked_finger_q is not None else 0.007
        elif idx < 100:
            a = (idx - 82) / 17.0
            z_bias = 0.004 if downward_force > 12.0 else 0.0
            pos = _lerp(self.wipe_pts[1], self.wipe_pts[2], a) + np.array([0.0, 0.0, z_bias])
            finger = self._locked_finger_q if self._locked_finger_q is not None else 0.007
        elif idx < 118:
            a = (idx - 100) / 17.0
            pos = _lerp(self.wipe_pts[2], np.array([self.wipe_pts[2][0], self.wipe_pts[2][1], 0.905]), a)
            finger = self._locked_finger_q if self._locked_finger_q is not None else 0.007
        elif idx < 138:
            a = (idx - 118) / 19.0
            xy = _lerp(self.wipe_pts[2][:2], cad[:2], a)
            pos = np.array([xy[0], xy[1], 0.905 - 0.065 * a])
            finger = self._locked_finger_q if self._locked_finger_q is not None else 0.007
        elif idx < 150:
            a = (idx - 138) / 11.0
            pos = _lerp(np.array([cad[0], cad[1], 0.84]), np.array([cad[0], cad[1], cad[2] + 0.025]), a)
            finger = self._locked_finger_q if self._locked_finger_q is not None else 0.007
        elif idx < 157:
            pos = np.array([cad[0], cad[1], cad[2] + 0.025])
            finger = 0.036
        else:
            a = (idx - 157) / 7.0
            pos = _lerp(np.array([cad[0], cad[1], cad[2] + 0.025]), np.array([cad[0], cad[1], 0.89]), a)
            finger = 0.036

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
