from __future__ import annotations

from dataclasses import dataclass
import math

from trajectory_forge.core.contracts import (
    ContractValidationError,
    ObjectContractSpec,
    PoseSpec,
)


Vec3 = tuple[float, float, float]
QuatWXYZ = tuple[float, float, float, float]


def normalize_vec3(v: Vec3) -> Vec3:
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if norm <= 1e-12:
        raise ContractValidationError("Cannot normalize near-zero 3D vector")
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def normalize_quat_wxyz(q: QuatWXYZ) -> QuatWXYZ:
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if norm <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


def quat_conjugate_wxyz(q: QuatWXYZ) -> QuatWXYZ:
    w, x, y, z = normalize_quat_wxyz(q)
    return (w, -x, -y, -z)


def quat_multiply_wxyz(a: QuatWXYZ, b: QuatWXYZ) -> QuatWXYZ:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return normalize_quat_wxyz(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def quat_from_axis_angle_wxyz(axis: Vec3, angle_rad: float) -> QuatWXYZ:
    ax, ay, az = normalize_vec3(axis)
    half = 0.5 * angle_rad
    s = math.sin(half)
    return normalize_quat_wxyz((math.cos(half), ax * s, ay * s, az * s))


def rotate_vec_by_quat_wxyz(q: QuatWXYZ, v: Vec3) -> Vec3:
    qn = normalize_quat_wxyz(q)
    vq: QuatWXYZ = (0.0, v[0], v[1], v[2])
    q_inv = (qn[0], -qn[1], -qn[2], -qn[3])
    # Pure quaternion product without re-normalizing the vector quaternion magnitude
    aw, ax, ay, az = qn
    bw, bx, by, bz = vq
    tmp = (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )
    cw, cx, cy, cz = tmp
    dw, dx, dy, dz = q_inv
    rx = cw * dx + cx * dw + cy * dz - cz * dy
    ry = cw * dy - cx * dz + cy * dw + cz * dx
    rz = cw * dz + cx * dy - cy * dx + cz * dw
    return (rx, ry, rz)


def slerp_wxyz(q0: QuatWXYZ, q1: QuatWXYZ, t: float) -> QuatWXYZ:
    a = normalize_quat_wxyz(q0)
    b = normalize_quat_wxyz(q1)
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if dot < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
        dot = -dot
    if dot > 0.9995:
        lerp = (
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
            a[3] + t * (b[3] - a[3]),
        )
        return normalize_quat_wxyz(lerp)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return normalize_quat_wxyz(
        (
            s0 * a[0] + s1 * b[0],
            s0 * a[1] + s1 * b[1],
            s0 * a[2] + s1 * b[2],
            s0 * a[3] + s1 * b[3],
        )
    )


@dataclass(frozen=True)
class Pose3D:
    """SE(3) pose in meters and wxyz quaternion."""

    xyz: Vec3
    wxyz: QuatWXYZ = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.xyz) != 3 or len(self.wxyz) != 4:
            raise ContractValidationError("Pose3D requires 3D xyz and 4D wxyz")

    @classmethod
    def from_spec(cls, spec: PoseSpec) -> Pose3D:
        return cls(xyz=spec.xyz, wxyz=normalize_quat_wxyz(spec.wxyz))

    @classmethod
    def from_approach(
        cls,
        *,
        xyz: Vec3,
        approach_dir: Vec3,
        closing_dir: Vec3,
    ) -> Pose3D:
        """Build an orthonormal gripper TCP pose with +Z along approach_dir and +Y along closing_dir."""
        z_axis = normalize_vec3(approach_dir)
        # Gram-Schmidt orthogonalize closing_dir against approach_dir -> +Y axis, +X = +Y x +Z
        dot = closing_dir[0] * z_axis[0] + closing_dir[1] * z_axis[1] + closing_dir[2] * z_axis[2]
        y_raw = (
            closing_dir[0] - dot * z_axis[0],
            closing_dir[1] - dot * z_axis[1],
            closing_dir[2] - dot * z_axis[2],
        )
        y_norm_sq = y_raw[0] * y_raw[0] + y_raw[1] * y_raw[1] + y_raw[2] * y_raw[2]
        if y_norm_sq <= 1e-10:
            raise ContractValidationError("approach_dir and closing_dir must not be collinear")
        y_axis = normalize_vec3(y_raw)
        x_axis = (
            y_axis[1] * z_axis[2] - y_axis[2] * z_axis[1],
            y_axis[2] * z_axis[0] - y_axis[0] * z_axis[2],
            y_axis[0] * z_axis[1] - y_axis[1] * z_axis[0],
        )
        wxyz = quat_from_rotation_matrix(x_axis, y_axis, z_axis)
        return cls(xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])), wxyz=wxyz)

    def to_spec(self) -> PoseSpec:
        return PoseSpec(xyz=self.xyz, wxyz=normalize_quat_wxyz(self.wxyz))


def quat_from_rotation_matrix(col_x: Vec3, col_y: Vec3, col_z: Vec3) -> QuatWXYZ:
    """Convert 3 orthonormal basis column vectors into a unit wxyz quaternion."""
    m00, m10, m20 = col_x
    m01, m11, m21 = col_y
    m02, m12, m22 = col_z
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return normalize_quat_wxyz((w, x, y, z))


def quat_angle_deg(q0: QuatWXYZ, q1: QuatWXYZ) -> float:
    """Shortest angular difference in degrees between two unit quaternions."""
    a = normalize_quat_wxyz(q0)
    b = normalize_quat_wxyz(q1)
    dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def compose_pose(parent: Pose3D, local: Pose3D) -> Pose3D:
    """Compute world_from_child = world_from_parent * parent_from_child."""
    rotated = rotate_vec_by_quat_wxyz(parent.wxyz, local.xyz)
    xyz = (
        parent.xyz[0] + rotated[0],
        parent.xyz[1] + rotated[1],
        parent.xyz[2] + rotated[2],
    )
    wxyz = quat_multiply_wxyz(parent.wxyz, local.wxyz)
    return Pose3D(xyz=xyz, wxyz=wxyz)


def invert_pose(pose: Pose3D) -> Pose3D:
    """Compute inverse SE(3) transform."""
    q_inv = quat_conjugate_wxyz(pose.wxyz)
    t_inv = rotate_vec_by_quat_wxyz(
        q_inv, (-pose.xyz[0], -pose.xyz[1], -pose.xyz[2])
    )
    return Pose3D(xyz=t_inv, wxyz=q_inv)


def pose_from_named_frame(
    obj: ObjectContractSpec,
    frame_name: str,
    object_world_pose: Pose3D | None = None,
) -> Pose3D:
    """Resolve an object's named_frame into world coordinates."""
    if frame_name not in obj.named_frames:
        raise KeyError(f"Object {obj.object_id!r} has no named_frame {frame_name!r}")
    base = object_world_pose if object_world_pose is not None else Pose3D.from_spec(obj.initial_pose)
    local = Pose3D.from_spec(obj.named_frames[frame_name].pose_local)
    return compose_pose(base, local)


def sample_revolute_arc_poses(
    *,
    start_pose: Pose3D,
    hinge_pivot_world: Vec3,
    hinge_axis_world: Vec3,
    delta_angle_rad: float,
    num_waypoints: int,
) -> list[Pose3D]:
    """Generate rigid-constraint circular arc waypoints around a revolute hinge."""
    if num_waypoints < 2:
        raise ContractValidationError("sample_revolute_arc_poses requires num_waypoints >= 2")
    rel_start = (
        start_pose.xyz[0] - hinge_pivot_world[0],
        start_pose.xyz[1] - hinge_pivot_world[1],
        start_pose.xyz[2] - hinge_pivot_world[2],
    )
    waypoints: list[Pose3D] = []
    for i in range(num_waypoints):
        frac = i / (num_waypoints - 1)
        q_delta = quat_from_axis_angle_wxyz(hinge_axis_world, delta_angle_rad * frac)
        rel_rot = rotate_vec_by_quat_wxyz(q_delta, rel_start)
        xyz = (
            hinge_pivot_world[0] + rel_rot[0],
            hinge_pivot_world[1] + rel_rot[1],
            hinge_pivot_world[2] + rel_rot[2],
        )
        wxyz = quat_multiply_wxyz(q_delta, start_pose.wxyz)
        waypoints.append(Pose3D(xyz=xyz, wxyz=wxyz))
    return waypoints


def sample_prismatic_sweep_poses(
    *,
    start_pose: Pose3D,
    direction_world: Vec3,
    distance_m: float,
    num_waypoints: int,
) -> list[Pose3D]:
    """Generate straight-line prismatic translation waypoints preserving orientation."""
    if num_waypoints < 2:
        raise ContractValidationError("sample_prismatic_sweep_poses requires num_waypoints >= 2")
    dx, dy, dz = normalize_vec3(direction_world)
    waypoints: list[Pose3D] = []
    for i in range(num_waypoints):
        s = distance_m * (i / (num_waypoints - 1))
        waypoints.append(
            Pose3D(
                xyz=(
                    start_pose.xyz[0] + dx * s,
                    start_pose.xyz[1] + dy * s,
                    start_pose.xyz[2] + dz * s,
                ),
                wxyz=start_pose.wxyz,
            )
        )
    return waypoints
