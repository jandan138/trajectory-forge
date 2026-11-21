"""Composable firewall-pure deformable manipulation skill primitives (M3 v0.3)."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

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


V03_DEFAULT_SKILL_PARAMS: dict[str, dict[str, Any]] = {
    "coffee_croissant_tray_to_plate": {
        "skill": "CompliantPinchPlaceController",
        "grasp_z": 0.770,
        "lift_z": 0.890,
        "place_z_offset": 0.032,
        "open_gap": 0.040,
        "min_finger_gap": 0.0195,
        "force_latch_n": 2.4,
        "place_yaw": math.radians(30.0),
    },
    "coffee_strawberry_tart_topping": {
        "skill": "CompliantPinchPlaceController",
        "grasp_z": 0.774,
        "lift_z": 0.895,
        "place_z_offset": 0.035,
        "open_gap": 0.040,
        "min_finger_gap": 0.0225,
        "force_latch_n": 1.6,
        "place_yaw": 0.0,
    },
    "office_waffle_towel_corner_fold": {
        "skill": "ClothEdgeFoldAndPeelController",
        "submode": "fold",
        "x_pick": -0.055,
        "y_pick": 0.020,
        "grasp_z": 0.7665,
        "scrunch_steps": 32,
        "lift_steps": 22,
        "lift_z": 0.855,
        "x_place": 0.072,
        "place_z": 0.778,
        "fold_steps": 54,
        "open_gap": 0.034,
        "min_finger_gap": 0.0002,
    },
    "office_dust_cover_peel_retrieve": {
        "skill": "ClothEdgeFoldAndPeelController",
        "submode": "peel_retrieve",
        "peel_target_xy": [-0.34, 0.16],
        "cloth_pick_dx": -0.070,
        "cloth_pick_z": 0.767,
        "cloth_open_gap": 0.022,
        "cloth_pinch_gap": 0.0002,
        "box_grasp_gap": 0.0302,
        "force_latch_n": 6.5,
    },
    "smt_ribbon_cable_clip_route": {
        "skill": "FlexibleRodRouteClipController",
        "grasp_z": 0.773,
        "lift_z": 0.875,
        "place_z": 0.771,
        "open_gap": 0.028,
        "min_finger_gap": 0.0085,
        "force_latch_n": 3.5,
    },
    "smt_esd_silicone_boot_press_fit": {
        "skill": "HollowShellSqueezeFitController",
        "submode": "press_fit",
        "grasp_z": 0.773,
        "lift_z": 0.870,
        "press_z": 0.769,
        "open_gap": 0.040,
        "min_finger_gap": 0.0235,
        "force_latch_n": 3.5,
    },
    "lab_silicone_tube_manifold_insert": {
        "skill": "FlexibleRodRouteClipController",
        "grasp_z": 0.777,
        "lift_z": 0.875,
        "place_z": 0.774,
        "open_gap": 0.028,
        "min_finger_gap": 0.0082,
        "force_latch_n": 3.2,
    },
    "lab_rubber_bulb_squeeze_seat": {
        "skill": "HollowShellSqueezeFitController",
        "submode": "squeeze_seat",
        "grasp_z": 0.778,
        "lift_z": 0.875,
        "seat_z": 0.777,
        "open_gap": 0.038,
        "min_finger_gap": 0.0205,
        "squeeze_gap": 0.0010,
        "force_latch_n": 4.2,
    },
}


class _BaseV03PandaController:
    dt_ctrl: float = 0.01

    def __init__(self, brief: ScenarioIntakeBrief, *, mode: str = "gentle", params: Mapping[str, Any] | None = None) -> None:
        self.brief = brief
        self.mode = str(mode)
        self.params = dict(params or {})
        self.base_pos = np.asarray(brief.robot_spawn.xyz or DEFAULT_PANDA_BASE_POS, dtype=np.float64)
        self.base_quat = np.asarray(brief.robot_spawn.wxyz or DEFAULT_PANDA_BASE_QUAT_WXYZ, dtype=np.float64)
        self._q_prev = PANDA_HOME_Q.copy()
        self._locked_finger_q: float | None = None

    def _ik(self, pos: np.ndarray, yaw: float, finger: float) -> np.ndarray:
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


class CompliantPinchPlaceController(_BaseV03PandaController):
    """Skill Primitive 1: Tactile-guarded compliant pinch pick-and-place for fragile sculpted solids."""

    controller_id = "compliant_pinch_place_v03"
    total_steps = 180

    def __init__(self, brief: ScenarioIntakeBrief, *, mode: str = "gentle", params: Mapping[str, Any] | None = None) -> None:
        super().__init__(brief, mode=mode, params=params)
        def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
        tgt_obj = brief.objects.get("tartlet_pastry_cup") or brief.objects.get("serving_plate")
        self.pick_xyz = np.asarray(def_obj.initial_pose.xyz, dtype=np.float64)
        self.place_xyz = np.asarray(tgt_obj.initial_pose.xyz if tgt_obj else [0.18, -0.02, 0.755], dtype=np.float64)

    def step(self, obs: Mapping[str, object]) -> np.ndarray:
        idx = int(obs["step_idx"])
        tac = np.asarray(obs.get("tactile_normal_force_n", [0.0, 0.0]), dtype=np.float64)
        avg_fn = float(0.5 * (abs(tac[0]) + abs(tac[1])))
        q_obs = np.asarray(obs.get("qpos", np.zeros(9)), dtype=np.float64)

        p0, pl = self.pick_xyz, self.place_xyz
        grasp_z = float(self.params.get("grasp_z", 0.772))
        lift_z = float(self.params.get("lift_z", 0.890))
        place_off = float(self.params.get("place_z_offset", 0.032))
        open_gap = float(self.params.get("open_gap", 0.040))
        min_gap = float(self.params.get("min_finger_gap", 0.020))
        latch_n = float(self.params.get("force_latch_n", 2.0))
        place_yaw = float(self.params.get("place_yaw", 0.0))

        is_crush = self.mode in ("crush", "v0_overforce")
        if is_crush:
            min_gap = 0.0055
            latch_n = 999.0

        yaw = 0.0
        finger = open_gap
        if idx < 16:
            pos = _lerp(np.array([p0[0], p0[1], lift_z]), np.array([p0[0], p0[1], 0.840]), idx / 15.0)
        elif idx < 34:
            pos = _lerp(np.array([p0[0], p0[1], 0.840]), np.array([p0[0], p0[1], grasp_z]), (idx - 16) / 17.0)
        elif idx < 64:
            a = (idx - 34) / 29.0
            pos = np.array([p0[0], p0[1], grasp_z])
            cmd_f = open_gap - a * (open_gap - min_gap)
            if avg_fn >= latch_n and self._locked_finger_q is None:
                self._locked_finger_q = float(min(open_gap - 0.004, max(min_gap, 0.5 * (q_obs[7] + q_obs[8]))))
            finger = self._locked_finger_q if self._locked_finger_q is not None else cmd_f
        elif idx < 88:
            a = (idx - 64) / 23.0
            pos = _lerp(np.array([p0[0], p0[1], grasp_z]), np.array([p0[0], p0[1], lift_z]), a)
            yaw = place_yaw * a
            finger = self._locked_finger_q if self._locked_finger_q is not None else min_gap
        elif idx < 126:
            a = (idx - 88) / 37.0
            xy = _lerp(p0[:2], pl[:2], a)
            pos = np.array([xy[0], xy[1], lift_z + 0.025 * math.sin(math.pi * a)])
            yaw = place_yaw
            finger = self._locked_finger_q if self._locked_finger_q is not None else min_gap
        elif idx < 150:
            a = (idx - 126) / 23.0
            pos = _lerp(np.array([pl[0], pl[1], lift_z]), np.array([pl[0], pl[1], pl[2] + place_off]), a)
            yaw = place_yaw
            finger = self._locked_finger_q if self._locked_finger_q is not None else min_gap
        elif idx < 164:
            pos = np.array([pl[0], pl[1], pl[2] + place_off])
            yaw = place_yaw
            finger = open_gap
        else:
            a = (idx - 164) / max(1, self.total_steps - 165)
            pos = _lerp(np.array([pl[0], pl[1], pl[2] + place_off]), np.array([pl[0], pl[1], lift_z]), a)
            yaw = place_yaw
            finger = open_gap

        return self._ik(pos, yaw, finger)


class ClothEdgeFoldAndPeelController(_BaseV03PandaController):
    """Skill Primitive 2: Stand-free cloth scrunch fold OR 2-stage cloth-peel + rigid retrieve."""

    controller_id = "cloth_edge_fold_peel_v03"

    def __init__(self, brief: ScenarioIntakeBrief, *, mode: str = "gentle", params: Mapping[str, Any] | None = None) -> None:
        super().__init__(brief, mode=mode, params=params)
        self.submode = str(self.params.get("submode", "fold"))
        self.total_steps = 180 if self.submode == "fold" else 242
        def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
        self.cloth_xyz = np.asarray(def_obj.initial_pose.xyz, dtype=np.float64)
        box_obj = brief.objects.get("toner_cartridge_box")
        caddy_obj = brief.objects.get("delivery_caddy")
        self.box_xyz = np.asarray(box_obj.initial_pose.xyz if box_obj else [-0.14, 0.02, 0.757], dtype=np.float64)
        self.caddy_xyz = np.asarray(caddy_obj.initial_pose.xyz if caddy_obj else [0.20, -0.02, 0.757], dtype=np.float64)

    def step(self, obs: Mapping[str, object]) -> np.ndarray:
        idx = int(obs["step_idx"])
        if self.submode == "peel_retrieve":
            return self._step_peel_retrieve(idx, obs)
        return self._step_fold(idx, obs)

    def _step_fold(self, idx: int, obs: Mapping[str, object]) -> np.ndarray:
        xp = float(self.params.get("x_pick", -0.055))
        yp = float(self.params.get("y_pick", 0.020))
        gz = float(self.params.get("grasp_z", 0.7665))
        scrunch_n = int(self.params.get("scrunch_steps", 32))
        lift_n = int(self.params.get("lift_steps", 22))
        fold_n = int(self.params.get("fold_steps", 54))
        lz = float(self.params.get("lift_z", 0.855))
        xpl = float(self.params.get("x_place", 0.072))
        pz = float(self.params.get("place_z", 0.778))
        open_gap = float(self.params.get("open_gap", 0.034))
        min_gap = float(self.params.get("min_finger_gap", 0.0002))

        if self.mode == "v0_fast_slip":
            scrunch_n = 8
            fold_n = 16
            lz = 0.805
            gz = 0.792
            min_gap = max(min_gap, 0.022)
        elif self.mode == "tear":
            gz = 0.751

        t0 = 22
        t1 = t0 + scrunch_n
        t_lift = t1 + lift_n
        t_fold = t_lift + fold_n
        t_rel = t_fold + 24
        yaw = math.pi / 2.0  # Fingers open along world X for X-axis cloth fold!
        if idx < t0:
            a = idx / float(t0 - 1)
            pos = _lerp(np.array([xp, yp, 0.860]), np.array([xp, yp, gz]), a)
            finger = open_gap
        elif idx < t1:
            a = (idx - t0) / max(1.0, float(t1 - t0 - 1))
            pos = np.array([xp, yp, gz])
            finger = open_gap - a * (open_gap - min_gap)
        elif idx < t_lift:
            a = (idx - t1) / max(1.0, float(t_lift - t1 - 1))
            if self.mode == "tear":
                pos = _lerp(np.array([xp, yp, 0.750]), np.array([0.22, yp, 0.750]), a)
            else:
                pos = _lerp(np.array([xp, yp, gz]), np.array([xp + 0.010, yp, lz]), a)
            finger = min_gap
        elif idx < t_fold:
            a = (idx - t_lift) / max(1.0, float(t_fold - t_lift - 1))
            if self.mode == "tear":
                pos = np.array([0.22, yp, 0.750])
            else:
                pos = _lerp(np.array([xp + 0.010, yp, lz]), np.array([xpl, yp, pz]), a)
            finger = min_gap
        elif idx < t_rel:
            a = (idx - t_fold) / max(1.0, float(t_rel - t_fold - 1))
            pos = np.array([xpl, yp, pz])
            finger = 0.036 if a > 0.35 else min_gap
        else:
            a = (idx - t_rel) / max(1.0, float(self.total_steps - t_rel - 1))
            pos = _lerp(np.array([xpl, yp, pz]), np.array([xpl, yp, 0.875]), a)
            finger = 0.036
        return self._ik(pos, yaw, finger)

    def _step_peel_retrieve(self, idx: int, obs: Mapping[str, object]) -> np.ndarray:
        bx, by, bz = self.box_xyz
        cx, cy, cz = self.caddy_xyz
        px_t, py_t = self.params.get("peel_target_xy", [-0.34, 0.16])
        c_gap = float(self.params.get("cloth_pinch_gap", 0.0002))
        c_open = float(self.params.get("cloth_open_gap", 0.022))
        b_gap = 0.012 if self.mode == "tear" else float(self.params.get("box_grasp_gap", 0.0302))
        px_pick = bx + float(self.params.get("cloth_pick_dx", -0.070))
        z_pick = float(self.params.get("cloth_pick_z", 0.767))

        # Stage 1 (0..117): pinch -X hanging cloth skirt clear of box -> peel to (-0.34, +0.16) -> vertical release
        # Stage 2 (118..241): grasp uncovered toner_cartridge_box -> lift & place in caddy (0.20, -0.02)
        if idx < 24:
            a = idx / 23.0
            pos = _lerp(np.array([px_pick, by, 0.885]), np.array([px_pick, by, z_pick]), a)
            yaw, finger = math.pi / 2.0, c_open
        elif idx < 54:
            a = (idx - 24) / 29.0
            pos = np.array([px_pick, by, z_pick])
            yaw = math.pi / 2.0
            finger = c_open - a * (c_open - c_gap)
        elif idx < 80:
            a = (idx - 54) / 25.0
            if self.mode == "tear":
                pos = _lerp(np.array([px_pick, by, 0.749]), np.array([0.05, by, 0.749]), a)
            else:
                pos = _lerp(np.array([px_pick, by, z_pick]), np.array([-0.24, 0.10, 0.920]), a)
            yaw, finger = math.pi / 2.0, c_gap
        elif idx < 104:
            a = (idx - 80) / 23.0
            if self.mode == "tear":
                pos = np.array([0.05, by, 0.749])
            else:
                pos = _lerp(np.array([-0.24, 0.10, 0.920]), np.array([px_t, py_t, 0.810]), a)
            yaw, finger = math.pi / 2.0, c_gap
        elif idx < 118:
            a = (idx - 104) / 13.0
            pos = _lerp(np.array([px_t, py_t, 0.810]), np.array([px_t, py_t, 0.910]), a)
            yaw, finger = math.pi / 2.0, 0.040
        elif idx < 134:
            a = (idx - 118) / 15.0
            pos = _lerp(np.array([px_t, py_t, 0.910]), np.array([bx, by, 0.890]), a)
            yaw = (1.0 - a) * (math.pi / 2.0)
            finger = 0.039
        elif idx < 154:
            a = (idx - 134) / 19.0
            pos = _lerp(np.array([bx, by, 0.890]), np.array([bx, by, 0.779]), a)
            yaw, finger = 0.0, 0.039
        elif idx < 176:
            a = (idx - 154) / 21.0
            pos = np.array([bx, by, 0.779])
            yaw, finger = 0.0, 0.039 - a * (0.039 - b_gap)
        elif idx < 194:
            a = (idx - 176) / 17.0
            pos = _lerp(np.array([bx, by, 0.779]), np.array([bx, by, 0.890]), a)
            yaw, finger = 0.0, b_gap
        elif idx < 220:
            a = (idx - 194) / 25.0
            xy = _lerp(np.array([bx, by]), np.array([cx, cy]), a)
            pos = np.array([xy[0], xy[1], 0.890 + 0.02 * math.sin(math.pi * a)])
            yaw, finger = 0.0, b_gap
        elif idx < 232:
            a = (idx - 220) / 11.0
            pos = _lerp(np.array([cx, cy, 0.890]), np.array([cx, cy, cz + 0.025]), a)
            yaw, finger = 0.0, b_gap
        else:
            a = (idx - 232) / max(1.0, float(self.total_steps - 233))
            pos = _lerp(np.array([cx, cy, cz + 0.025]), np.array([cx, cy, 0.890]), a)
            yaw, finger = 0.0, 0.039
        return self._ik(pos, yaw, finger)


class FlexibleRodRouteClipController(_BaseV03PandaController):
    """Skill Primitive 3: Compliant routing and snap-seating for 1D ribbons and silicone tubes."""

    controller_id = "flexible_rod_route_clip_v03"
    total_steps = 180

    def __init__(self, brief: ScenarioIntakeBrief, *, mode: str = "gentle", params: Mapping[str, Any] | None = None) -> None:
        super().__init__(brief, mode=mode, params=params)
        def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
        tgt_obj = brief.objects.get("smt_u_clip_block") or brief.objects.get("manifold_fork_saddle")
        self.pick_xyz = np.asarray(def_obj.initial_pose.xyz, dtype=np.float64)
        self.place_xyz = np.asarray(tgt_obj.initial_pose.xyz if tgt_obj else [0.15, 0.03, 0.755], dtype=np.float64)

    def step(self, obs: Mapping[str, object]) -> np.ndarray:
        idx = int(obs["step_idx"])
        tac = np.asarray(obs.get("tactile_normal_force_n", [0.0, 0.0]), dtype=np.float64)
        avg_fn = float(0.5 * (abs(tac[0]) + abs(tac[1])))
        q_obs = np.asarray(obs.get("qpos", np.zeros(9)), dtype=np.float64)

        p0, pl = self.pick_xyz, self.place_xyz
        gz = float(self.params.get("grasp_z", 0.775))
        lz = float(self.params.get("lift_z", 0.875))
        pz = float(self.params.get("place_z", 0.772))
        open_gap = float(self.params.get("open_gap", 0.028))
        min_gap = float(self.params.get("min_finger_gap", 0.0085))
        latch_n = float(self.params.get("force_latch_n", 3.5))
        is_crush = self.mode in ("crush", "tear")
        if is_crush:
            min_gap = 0.0005
            latch_n = 999.0
            pz = 0.750

        finger = open_gap
        if idx < 16:
            pos = _lerp(np.array([p0[0], p0[1], lz]), np.array([p0[0], p0[1], 0.840]), idx / 15.0)
        elif idx < 32:
            pos = _lerp(np.array([p0[0], p0[1], 0.840]), np.array([p0[0], p0[1], gz]), (idx - 16) / 15.0)
        elif idx < 60:
            a = (idx - 32) / 27.0
            pos = np.array([p0[0], p0[1], gz])
            cmd_f = open_gap - a * (open_gap - min_gap)
            if avg_fn >= latch_n and self._locked_finger_q is None:
                self._locked_finger_q = float(min(open_gap - 0.005, max(min_gap, 0.5 * (q_obs[7] + q_obs[8]))))
            finger = self._locked_finger_q if self._locked_finger_q is not None else cmd_f
        elif idx < 84:
            a = (idx - 60) / 23.0
            pos = _lerp(np.array([p0[0], p0[1], gz]), np.array([p0[0], p0[1], lz]), a)
            finger = self._locked_finger_q if self._locked_finger_q is not None else min_gap
        elif idx < 126:
            a = (idx - 84) / 41.0
            xy = _lerp(p0[:2], pl[:2], a)
            pos = np.array([xy[0], xy[1], lz + 0.02 * math.sin(math.pi * a)])
            finger = self._locked_finger_q if self._locked_finger_q is not None else min_gap
        elif idx < 154:
            a = (idx - 126) / 27.0
            pos = _lerp(np.array([pl[0], pl[1], lz]), np.array([pl[0], pl[1], pz]), a)
            finger = self._locked_finger_q if self._locked_finger_q is not None else min_gap
        elif idx < 166:
            pos = np.array([pl[0], pl[1], pz])
            finger = open_gap
        else:
            a = (idx - 166) / max(1.0, float(self.total_steps - 167))
            pos = _lerp(np.array([pl[0], pl[1], pz]), np.array([pl[0], pl[1], lz]), a)
            finger = open_gap
        return self._ik(pos, 0.0, finger)


class HollowShellSqueezeFitController(_BaseV03PandaController):
    """Skill Primitive 4: Elastic shell press-fit onto shroud OR collar seat + quantitative squeeze-rebound."""

    controller_id = "hollow_shell_squeeze_fit_v03"

    def __init__(self, brief: ScenarioIntakeBrief, *, mode: str = "gentle", params: Mapping[str, Any] | None = None) -> None:
        super().__init__(brief, mode=mode, params=params)
        self.submode = str(self.params.get("submode", "press_fit"))
        self.total_steps = 175 if self.submode == "press_fit" else 205
        def_obj = next(o for o in brief.objects.values() if o.role == "manipulated_deformable")
        tgt_obj = brief.objects.get("pcba_header_boss") or brief.objects.get("bulb_collar_holder")
        self.pick_xyz = np.asarray(def_obj.initial_pose.xyz, dtype=np.float64)
        self.place_xyz = np.asarray(tgt_obj.initial_pose.xyz if tgt_obj else [0.15, -0.02, 0.755], dtype=np.float64)

    def step(self, obs: Mapping[str, object]) -> np.ndarray:
        idx = int(obs["step_idx"])
        tac = np.asarray(obs.get("tactile_normal_force_n", [0.0, 0.0]), dtype=np.float64)
        avg_fn = float(0.5 * (abs(tac[0]) + abs(tac[1])))
        q_obs = np.asarray(obs.get("qpos", np.zeros(9)), dtype=np.float64)

        p0, pl = self.pick_xyz, self.place_xyz
        gz = float(self.params.get("grasp_z", 0.774))
        lz = float(self.params.get("lift_z", 0.870))
        open_gap = float(self.params.get("open_gap", 0.040))
        min_gap = float(self.params.get("min_finger_gap", 0.0215))
        latch_n = float(self.params.get("force_latch_n", 3.5))
        is_crush = self.mode in ("crush", "tear")
        if is_crush:
            min_gap = 0.001
            latch_n = 999.0

        finger = open_gap
        if idx < 16:
            pos = _lerp(np.array([p0[0], p0[1], lz]), np.array([p0[0], p0[1], 0.840]), idx / 15.0)
        elif idx < 32:
            pos = _lerp(np.array([p0[0], p0[1], 0.840]), np.array([p0[0], p0[1], gz]), (idx - 16) / 15.0)
        elif idx < 60:
            a = (idx - 32) / 27.0
            pos = np.array([p0[0], p0[1], gz])
            cmd_f = open_gap - a * (open_gap - min_gap)
            if avg_fn >= latch_n and self._locked_finger_q is None:
                self._locked_finger_q = float(min(open_gap - 0.005, max(min_gap, 0.5 * (q_obs[7] + q_obs[8]))))
            finger = self._locked_finger_q if self._locked_finger_q is not None else cmd_f
        elif idx < 82:
            a = (idx - 60) / 21.0
            pos = _lerp(np.array([p0[0], p0[1], gz]), np.array([p0[0], p0[1], lz]), a)
            finger = self._locked_finger_q if self._locked_finger_q is not None else min_gap
        elif idx < 116:
            a = (idx - 82) / 33.0
            xy = _lerp(p0[:2], pl[:2], a)
            pos = np.array([xy[0], xy[1], lz + 0.02 * math.sin(math.pi * a)])
            finger = self._locked_finger_q if self._locked_finger_q is not None else min_gap
        elif self.submode == "press_fit":
            pz = 0.750 if is_crush else float(self.params.get("press_z", 0.769))
            if idx < 142:
                a = (idx - 116) / 25.0
                pos = _lerp(np.array([pl[0], pl[1], lz]), np.array([pl[0], pl[1], pz]), a)
                finger = self._locked_finger_q if self._locked_finger_q is not None else min_gap
            elif idx < 156:
                # Hold firm compliant downward seat over PCBA header shroud
                pos = np.array([pl[0], pl[1], pz - 0.002])
                finger = (self._locked_finger_q if self._locked_finger_q is not None else min_gap) if idx < 148 else open_gap
            else:
                a = (idx - 156) / max(1.0, float(self.total_steps - 157))
                pos = _lerp(np.array([pl[0], pl[1], pz]), np.array([pl[0], pl[1], lz]), a)
                finger = open_gap
        else:
            # squeeze_seat: lower into 4-post collar -> open slightly -> quantitative squeeze pulse -> release & rebound
            sz = 0.749 if is_crush else float(self.params.get("seat_z", 0.776))
            sq_gap = 0.001 if is_crush else float(self.params.get("squeeze_gap", 0.0025))
            if idx < 138:
                a = (idx - 116) / 21.0
                pos = _lerp(np.array([pl[0], pl[1], lz]), np.array([pl[0], pl[1], sz]), a)
                finger = self._locked_finger_q if self._locked_finger_q is not None else min_gap
            elif idx < 148:
                # Settle inside collar pocket before radial squeeze pulse
                pos = np.array([pl[0], pl[1], sz])
                finger = 0.032
            elif idx < 168:
                a = (idx - 148) / 19.0
                pos = np.array([pl[0], pl[1], sz + 0.002])
                finger = 0.032 - a * (0.032 - sq_gap)
            elif idx < 176:
                pos = np.array([pl[0], pl[1], sz + 0.002])
                finger = sq_gap
            elif idx < 192:
                # Release open and hold stationary to measure elastic rebound recovery
                pos = np.array([pl[0], pl[1], sz + 0.002])
                finger = open_gap
            else:
                a = (idx - 192) / max(1.0, float(self.total_steps - 193))
                pos = _lerp(np.array([pl[0], pl[1], sz + 0.002]), np.array([pl[0], pl[1], lz]), a)
                finger = open_gap
        return self._ik(pos, 0.0, finger)


def make_v03_controller(
    brief: ScenarioIntakeBrief,
    task_id: str,
    *,
    mode: str = "gentle",
    params_override: Mapping[str, Any] | None = None,
) -> _BaseV03PandaController:
    base_p = copy.deepcopy(V03_DEFAULT_SKILL_PARAMS[task_id])
    if params_override:
        base_p.update(dict(params_override))
    cls_map = {
        "CompliantPinchPlaceController": CompliantPinchPlaceController,
        "ClothEdgeFoldAndPeelController": ClothEdgeFoldAndPeelController,
        "FlexibleRodRouteClipController": FlexibleRodRouteClipController,
        "HollowShellSqueezeFitController": HollowShellSqueezeFitController,
    }
    return cls_map[str(base_p["skill"])](brief, mode=mode, params=base_p)


def auto_tune_skill_params(
    task_id: str,
    current_params: Mapping[str, Any],
    eos_advice: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically remediate policy skill parameters in response to EOS failure diagnostics."""
    tuned = dict(copy.deepcopy(V03_DEFAULT_SKILL_PARAMS[task_id]))
    tuned.update(dict(current_params))
    reason = str(eos_advice.get("reason_code", ""))
    changes: list[str] = []

    if reason == "policy_damage_or_force_budget_exceeded":
        old_latch = float(tuned.get("force_latch_n", 4.0))
        old_gap = float(tuned.get("min_finger_gap", 0.015))
        new_latch = round(max(1.4, min(old_latch * 0.4, float(V03_DEFAULT_SKILL_PARAMS[task_id].get("force_latch_n", 1.8)))), 3)
        new_gap = round(max(old_gap, float(V03_DEFAULT_SKILL_PARAMS[task_id].get("min_finger_gap", 0.022))), 4)
        tuned["force_latch_n"] = new_latch
        tuned["min_finger_gap"] = new_gap
        changes.append(f"Reduced tactile force_latch_n {old_latch} -> {new_latch} N; widened min_finger_gap {old_gap} -> {new_gap} m")

    elif reason in ("policy_slip_or_fold_incomplete", "task_geometric_goal_unmet"):
        default_p = V03_DEFAULT_SKILL_PARAMS[task_id]
        for key in ("scrunch_steps", "lift_steps", "fold_steps", "lift_z", "min_finger_gap", "grasp_z", "place_z"):
            if key in default_p:
                tuned[key] = default_p[key]
        changes.append("Restored scrunch_steps/lift_steps/fold_steps/grasp_z dwell & lift_z arc clearance from EOS slip diagnosis")

    return tuned, changes
