"""Zero-edit tactile-guarded controller for Task 1 (coffee_bar_fragile_pastry_plate)."""

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


class CoffeePastryGentleController:
    """Open pastry box hinged lid, tactile-guarded gentle grasp, +35 deg lift, place on plate."""

    controller_id = "coffee_pastry_gentle_controller"
    total_steps = 185
    dt_ctrl = 0.01

    def __init__(self, brief: ScenarioIntakeBrief) -> None:
        self.brief = brief
        self.base_pos = np.asarray(brief.robot_spawn.xyz or DEFAULT_PANDA_BASE_POS, dtype=np.float64)
        self.base_quat = np.asarray(brief.robot_spawn.wxyz or DEFAULT_PANDA_BASE_QUAT_WXYZ, dtype=np.float64)
        pastry = brief.objects.get("fragile_pastry")
        plate = brief.objects.get("serving_plate")
        self.pastry_xyz = np.asarray(pastry.initial_pose.xyz if pastry else [-0.18, -0.02, 0.766], dtype=np.float64)
        self.plate_xyz = np.asarray(plate.initial_pose.xyz if plate else [0.18, -0.02, 0.755], dtype=np.float64)
        self._q_prev = PANDA_HOME_Q.copy()
        self._locked_finger_q: float | None = None

    def step(self, obs: Mapping[str, object]) -> np.ndarray:
        idx = int(obs["step_idx"])
        tactile = np.asarray(obs.get("tactile_normal_force_n", [0.0, 0.0]), dtype=np.float64)
        avg_fn = float(0.5 * (tactile[0] + tactile[1]))

        p0 = self.pastry_xyz
        pl = self.plate_xyz

        # Waypoint schedule over 150 ticks (1.5 s):
        # 0..24: move behind front lid tab and flip box lid wide open
        # 25..44: move above open box & descend around pastry (open jaws 0.04)
        # 45..64: tactile-guarded squeeze onto pastry sides (~0.0328 target, latched by tactile force)
        # 65..104: lift + carry across to plate while rotating wrist +35 deg yaw
        # 105..124: lower gently onto serving plate
        # 125..149: open gripper & retract overhead
        yaw = 0.0
        finger = 0.040
        y_hinge = p0[1] + 0.064
        z_hinge = 0.807
        if idx < 10:
            a = idx / 9.0
            pos = _lerp(np.array([p0[0], p0[1] - 0.150, 0.835]), np.array([p0[0], p0[1] - 0.118, 0.782]), a)
            finger = 0.002
        elif idx < 18:
            a = (idx - 10) / 7.0
            # Slide closed paddle under front lid_tab (y in [-0.112, -0.084], z in [0.805, 0.813])
            pos = _lerp(np.array([p0[0], p0[1] - 0.118, 0.782]), np.array([p0[0], p0[1] - 0.090, 0.782]), a)
            finger = 0.002
        elif idx < 42:
            a = (idx - 18) / 23.0
            # Follow circular hinge arc from 0 -> 108 deg so lid tips past 90 deg and rests wide open
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
            # Settle directly above pastry center with fingers full-open (0.040) before descent
            pos = np.array([p0[0], p0[1], 0.855])
            finger = 0.040
        elif idx < 76:
            a = (idx - 60) / 15.0
            pos = _lerp(np.array([p0[0], p0[1], 0.855]), np.array([p0[0], p0[1], 0.779]), a)
            finger = 0.040
        elif idx < 82:
            # Brief bottom dwell so both Y and Z track cleanly centered before squeezing
            pos = np.array([p0[0], p0[1], 0.779])
            finger = 0.040
        elif idx < 104:
            a = (idx - 82) / 21.0
            pos = np.array([p0[0], p0[1], 0.779])
            cmd_f = 0.040 - a * 0.0085  # gently closes toward 0.0315 unless latched earlier by tactile guard
            if avg_fn >= 2.2 and self._locked_finger_q is None:
                q_obs = np.asarray(obs.get("qpos", np.zeros(9)), dtype=np.float64)
                self._locked_finger_q = float(min(0.0338, max(0.0318, 0.5 * (q_obs[7] + q_obs[8]))))
            finger = self._locked_finger_q if self._locked_finger_q is not None else cmd_f
        elif idx < 122:
            a = (idx - 104) / 17.0
            pos = _lerp(np.array([p0[0], p0[1], 0.779]), np.array([p0[0], p0[1], 0.895]), a)
            yaw = math.radians(35.0) * a
            finger = self._locked_finger_q if self._locked_finger_q is not None else 0.0318
        elif idx < 150:
            a = (idx - 122) / 27.0
            mid_z = 0.895 + 0.03 * math.sin(math.pi * a)
            xy = _lerp(p0[:2], pl[:2], a)
            pos = np.array([xy[0], xy[1], mid_z])
            yaw = math.radians(35.0)
            finger = self._locked_finger_q if self._locked_finger_q is not None else 0.0318
        elif idx < 166:
            a = (idx - 150) / 15.0
            pos = _lerp(np.array([pl[0], pl[1], 0.895]), np.array([pl[0], pl[1], pl[2] + 0.022]), a)
            yaw = math.radians(35.0)
            finger = self._locked_finger_q if self._locked_finger_q is not None else 0.0318
        elif idx < 174:
            pos = np.array([pl[0], pl[1], pl[2] + 0.022])
            yaw = math.radians(35.0)
            finger = 0.040
        else:
            a = (idx - 174) / 10.0
            pos = _lerp(np.array([pl[0], pl[1], pl[2] + 0.022]), np.array([pl[0], pl[1], 0.90]), a)
            yaw = math.radians(35.0)
            finger = 0.040

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
