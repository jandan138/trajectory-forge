"""Blender 4.4.3 Cycles AgX studio scene renderer for HermiteTwin v0.8 Phase 2 release.

Invoked by `studio_video_renderer_v08.py --tier studio_cycles`:
  blender --background --python bpy_hermite_studio_scene.py -- \
    --npz <task_main.npz> --out-dir <frames_dir> --arm ground_truth|baseline|hermite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--arm", choices=("gt", "baseline", "hermite"), required=True)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p.parse_args(argv)


def main() -> None:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Matrix, Vector  # type: ignore[import-not-found]

    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.npz, allow_pickle=False)
    key_map = {
        "gt": "gt_vertices",
        "baseline": "baseline_vertices",
        "hermite": "hermite_vertices",
    }
    verts_seq = np.asarray(z[key_map[args.arm]], dtype=np.float64)
    g_pos_seq = np.asarray(z["gripper_pos_xyz"], dtype=np.float64)
    g_rot_seq = np.asarray(z["gripper_rot_3x3"], dtype=np.float64)
    g_gap_seq = np.asarray(z["gripper_gap_m"], dtype=np.float64)
    table_z = float(z["table_z_m"])
    base_rgb = tuple(float(c) for c in z["base_rgb"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(args.samples)
    scene.cycles.use_denoising = True
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = int(args.width)
    scene.render.resolution_y = int(args.height)
    scene.render.image_settings.file_format = "PNG"

    # Studio environment & softbox lighting
    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.04, 0.05, 0.07, 1.0)
        bg.inputs[1].default_value = 0.7

    def _add_area_light(name: str, loc: tuple[float, float, float], rot_euler: tuple[float, float, float], energy: float, size: float) -> None:
        ldata = bpy.data.lights.new(name=name, type="AREA")
        ldata.energy = energy
        ldata.size = size
        lobj = bpy.data.objects.new(name=name, object_data=ldata)
        scene.collection.objects.link(lobj)
        lobj.location = loc
        lobj.rotation_euler = rot_euler

    _add_area_light("KeySoftbox", (0.45, -0.55, 1.45), (0.65, 0.35, 0.4), 55.0, 0.8)
    _add_area_light("RimLight", (-0.50, 0.45, 1.25), (-0.7, -0.35, 0.0), 32.0, 0.6)

    # Workbench table
    bpy.ops.mesh.primitive_plane_add(size=1.4, location=(0.0, 0.0, table_z - 0.001))
    table = bpy.context.active_object
    mat_tab = bpy.data.materials.new("TableMat")
    mat_tab.use_nodes = True
    bsdf_t = mat_tab.node_tree.nodes["Principled BSDF"]
    bsdf_t.inputs["Base Color"].default_value = (0.16, 0.17, 0.19, 1.0)
    bsdf_t.inputs["Roughness"].default_value = 0.48
    table.data.materials.append(mat_tab)

    # Camera matching evaluation viewport
    cam_data = bpy.data.cameras.new("StudioCam")
    cam_data.lens = 35.0
    cam_obj = bpy.data.objects.new("StudioCam", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -0.56, 1.06)
    direction = Vector((0.0, 0.0, 0.78)) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam_obj

    # Deformable shell mesh object
    T, nu, nv, _ = verts_seq.shape
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = (i + 1) * nv + j
            c = (i + 1) * nv + (j + 1)
            d = i * nv + (j + 1)
            faces.append((a, b, c, d))
    mesh_data = bpy.data.meshes.new("ThinShellMesh")
    mesh_data.from_pydata(verts_seq[0].reshape(-1, 3).tolist(), [], faces)
    shell_obj = bpy.data.objects.new("ThinShell", mesh_data)
    scene.collection.objects.link(shell_obj)
    bpy.context.view_layer.objects.active = shell_obj
    bpy.ops.object.shade_smooth()
    mod_solid = shell_obj.modifiers.new("Solidify", "SOLIDIFY")
    mod_solid.thickness = 0.0025

    mat_sh = bpy.data.materials.new("ShellPBR")
    mat_sh.use_nodes = True
    bsdf_s = mat_sh.node_tree.nodes["Principled BSDF"]
    bsdf_s.inputs["Base Color"].default_value = (base_rgb[0], base_rgb[1], base_rgb[2], 1.0)
    bsdf_s.inputs["Roughness"].default_value = 0.65
    if "Sheen Weight" in bsdf_s.inputs:
        bsdf_s.inputs["Sheen Weight"].default_value = 0.45
    shell_obj.data.materials.append(mat_sh)

    # Dual parallel-jaw gripper
    def _make_jaw(name: str) -> object:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = (0.008, 0.020, 0.036)
        return obj

    jaw_l = _make_jaw("JawLeft")
    jaw_r = _make_jaw("JawRight")

    for t in range(T):
        pts = verts_seq[t].reshape(-1, 3)
        for idx, v in enumerate(mesh_data.vertices):
            v.co = pts[idx]
        mesh_data.update()

        gp = Vector(g_pos_seq[t].tolist())
        rot_m = Matrix(g_rot_seq[t].tolist()).to_4x4()
        half_g = 0.5 * float(g_gap_seq[t]) + 0.004
        jaw_l.matrix_world = Matrix.Translation(gp + (rot_m.to_3x3() @ Vector((-half_g, 0.0, 0.018)))) @ rot_m
        jaw_r.matrix_world = Matrix.Translation(gp + (rot_m.to_3x3() @ Vector((half_g, 0.0, 0.018)))) @ rot_m
        jaw_l.scale = (0.008, 0.020, 0.036)
        jaw_r.scale = (0.008, 0.020, 0.036)

        scene.render.filepath = str(args.out_dir / f"frame_{t:04d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
