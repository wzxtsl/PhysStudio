"""Refractive studio rendering of cached multi-fluid states and tea concentration."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache/blender"))
os.environ.setdefault("OPTIX_CACHE_PATH", str(ROOT / ".cache/optix"))
sys.path.insert(0, str(ROOT))
import bpy
import numpy as np
from mathutils import Vector
from implicit_mixture.scenes import motion

SCALE = 5.0


def material(name, color, rough=0.3, metal=0.0, transmission=0.0, ior=1.45):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bs = mat.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Base Color"].default_value = (*color, 1)
    bs.inputs["Roughness"].default_value = rough
    bs.inputs["Metallic"].default_value = metal
    bs.inputs["Transmission Weight"].default_value = transmission
    bs.inputs["IOR"].default_value = ior
    return mat


def mesh_object(name, vertices, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    for p in mesh.polygons:
        p.use_smooth = True
    return obj


def lathe(name, profile, mat, n=192):
    verts = [
        (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z)
        for r, z in profile
        for i in range(n)
    ]
    faces = []
    for j in range(len(profile) - 1):
        for i in range(n):
            faces.append(
                (
                    j * n + i,
                    j * n + (i + 1) % n,
                    (j + 1) * n + (i + 1) % n,
                    (j + 1) * n + i,
                )
            )
    obj = mesh_object(name, verts, faces, mat)
    normals = []
    for j in range(len(profile) - 1):
        dr = profile[j + 1][0] - profile[j][0]
        dz = profile[j + 1][1] - profile[j][1]
        norm = max(math.hypot(dr, dz), 1.0e-12)
        for i in range(n):
            for q in [i, (i + 1) % n, (i + 1) % n, i]:
                theta = 2 * math.pi * q / n
                normals.append(
                    (
                        dz * math.cos(theta) / norm,
                        dz * math.sin(theta) / norm,
                        -dr / norm,
                    )
                )
    obj.data.normals_split_custom_set(normals)
    return obj


def cube(name, center, size, mat, bevel=0.05):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    o = bpy.context.object
    o.name = name
    o.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    if bevel:
        b = o.modifiers.new("Soft edges", "BEVEL")
        b.width = bevel
        b.segments = 4
        o.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return o


def tube(name, points, radius, mat):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points):
        p.co = (*co, 1)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def lamp(name, pos, target, power, size, size_y):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = power
    data.shape = "RECTANGLE"
    data.size = size
    data.size_y = size_y
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = pos
    obj.rotation_euler = (
        (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )


def volume_material(name, density, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    volume = nodes.new("ShaderNodeVolumeAbsorption")
    volume.inputs["Color"].default_value = (*color, 1)
    volume.inputs["Density"].default_value = density
    mat.node_tree.links.new(volume.outputs["Volume"], out.inputs["Volume"])
    return mat


def setup(cfg, args, levels):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = True
    pref = bpy.context.preferences.addons["cycles"].preferences
    pref.compute_device_type = "OPTIX"
    pref.get_devices()
    for device in pref.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.max_bounces = 16
    scene.cycles.transmission_bounces = 12
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.volume_bounces = 1
    scene.cycles.volume_step_rate = 1.0
    scene.cycles.adaptive_threshold = 0.025
    scene.cycles.seed = 37
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.45
    world = bpy.data.worlds.new("Neutral daylight")
    world.use_nodes = True
    scene.world = world
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.48,
        0.50,
        0.53,
        1,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.65
    white = material("Porcelain", (0.65, 0.67, 0.68), 0.26)
    stone = material("Neutral stone", (0.32, 0.34, 0.35), 0.60)
    glass = material(
        "Borosilicate", (0.98, 0.99, 1.0), 0.025, transmission=1.0, ior=1.46
    )
    water = material("Water", (0.98, 0.995, 1.0), 0.025, transmission=1.0, ior=1.333)
    metal = material("Brushed alloy", (0.28, 0.31, 0.33), 0.23, 0.85)
    cube("Table", (0, 0, -0.55), (200, 200, 0.5), stone, 0.0)
    objects = {}
    bag = None
    rod = None
    string = None
    tag = None
    if cfg["scene"] == "tea":
        r = cfg["bounds"][1][0] * SCALE
        height = cfg["bounds"][1][2] * SCALE
        lathe(
            "Glass cup",
            [
                (0, -0.12),
                (r + 0.12, -0.12),
                (r + 0.12, height),
                (r, height),
                (r, 0),
                (0, 0),
            ],
            glass,
        )
        angles = np.linspace(-math.pi / 2, math.pi / 2, 90)
        handle = [
            (r + 0.03 + 1.12 * math.cos(t), 0, height * 0.48 + 1.05 * math.sin(t))
            for t in angles
        ]
        tube("Glass handle", handle, 0.12, glass)
        lathe(
            "Saucer",
            [
                (0, -0.20),
                (r + 0.75, -0.20),
                (r + 0.8, -0.08),
                (r + 0.55, -0.06),
                (r + 0.05, -0.16),
                (0, -0.16),
            ],
            white,
        )
        tea = material("Woven tea paper", (0.35, 0.25, 0.13), 0.82, transmission=0.2)
        nodes = tea.node_tree.nodes
        links = tea.node_tree.links
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 165
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.3
        bump.inputs["Distance"].default_value = 0.018
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], nodes["Principled BSDF"].inputs["Normal"])
        stain = nodes.new("ShaderNodeTexNoise")
        stain.inputs["Scale"].default_value = 38.0
        stain.inputs["Detail"].default_value = 3.0
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (0.045, 0.022, 0.006, 1)
        ramp.color_ramp.elements[1].color = (0.28, 0.18, 0.075, 1)
        links.new(stain.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
        bag = cube(
            "Tea sachet", (0, 0, 0), np.array(cfg["bag_half"]) * 2 * SCALE, tea, 0.065
        )
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=cfg["rod_radius"] * SCALE,
            depth=2 * cfg["rod_half"] * SCALE,
        )
        rod = bpy.context.object
        rod.name = "Glass stirring rod"
        rod.data.materials.append(glass)
        bevel = rod.modifiers.new("Rounded rod tips", "BEVEL")
        bevel.width = 0.035
        bevel.segments = 4
        for poly in rod.data.polygons:
            poly.use_smooth = True
        string = tube(
            "Tea string",
            [(0, 0, 0), (0, 0, 1)],
            0.014,
            material("Cotton", (0.7, 0.68, 0.61), 0.75),
        )
        tag = cube(
            "Tea tag",
            (0, 0, 0),
            (0.3, 0.04, 0.42),
            material("Tea label", (0.14, 0.23, 0.22), 0.65),
            0.018,
        )
        last = 0.0
        for i, level in enumerate(levels):
            objects[f"tea_{i}"] = mesh_object(
                f"Tea concentration {level}",
                [],
                [],
                volume_material(
                    f"Tea absorption {i}",
                    (level - last) * args.tea_absorption,
                    args.tea_color,
                ),
            )
            last = level
    else:
        lo, hi = np.array(cfg["bounds"]) * SCALE
        length, width = hi[:2] - lo[:2]
        height = hi[2]
        cube("Basin floor", (0, 0, -0.12), (length + 0.24, width + 0.24, 0.24), white)
        for sign in [-1, 1]:
            cube(
                "Long glass wall",
                (0, sign * (width / 2 + 0.035), height / 2),
                (length + 0.14, 0.07, height),
                glass,
                0.018,
            )
            cube(
                "Short glass wall",
                (sign * (length / 2 + 0.035), 0, height / 2),
                (0.07, width + 0.14, height),
                glass,
                0.018,
            )
            cube(
                "Edge rail",
                (0, sign * (width / 2 + 0.035), -0.04),
                (length + 0.22, 0.07, 0.07),
                metal,
                0.015,
            )
        scene.view_settings.exposure = -0.15
        water = material(
            "Colored fluid", (0.4, 0.5, 0.6), 0.10, transmission=0.06, ior=1.333
        )
        attr = water.node_tree.nodes.new("ShaderNodeVertexColor")
        attr.layer_name = "PhaseColor"
        water.node_tree.links.new(
            attr.outputs["Color"],
            water.node_tree.nodes["Principled BSDF"].inputs["Base Color"],
        )
    objects["fluid"] = mesh_object("Mixture free surface", [], [], water)
    lamp("Window", (-5, -5, 10), (0, 0, 1), 1800, 6.0, 7.0)
    lamp("Vertical reflection", (5, 3, 7), (0, 0, 2), 1600, 3.0, 6.0)
    lamp("Soft fill", (-4, 3, 4), (0, 0, 2), 750, 4.0, 4.0)
    lamp("Narrow reflection", (2, -5, 6), (0, 0, 2), 700, 1.0, 5.0)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    camera.data.lens = 55
    camera.data.clip_start = 0.03
    camera.data.clip_end = 300
    camera.data.dof.use_dof = False
    return scene, objects, camera, bag, rod, string, tag


def update_mesh(obj, vertices, faces, fractions=None, colors=None):
    obj.hide_render = len(faces) == 0
    mesh = obj.data
    mesh.clear_geometry()
    if len(faces):
        mesh.from_pydata((vertices * SCALE).tolist(), [], faces.tolist())
        mesh.update()
        for p in mesh.polygons:
            p.use_smooth = True
        if fractions is not None:
            attr = mesh.color_attributes.get("PhaseColor") or mesh.color_attributes.new(
                name="PhaseColor", type="FLOAT_COLOR", domain="POINT"
            )
            rgba = np.c_[fractions @ colors, np.ones(len(vertices))].astype(np.float32)
            attr.data.foreach_set("color", rgba.ravel())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--meshes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--frames", type=int, nargs="*")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tea-absorption", type=float, default=270.0)
    p.add_argument("--tea-color", type=float, nargs=3, default=[0.78, 0.50, 0.10])
    p.add_argument("--tea-camera-height", type=float, default=10.0)
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir(exist_ok=True)
    cfg = json.loads((args.input / "config.json").read_text())
    manifest = json.loads((args.meshes / "manifest.json").read_text())
    indices = args.frames if args.frames is not None else list(range(cfg["frames"]))
    assert set(indices).issubset(manifest["frames"])
    settings = dict(
        input=str(args.input.resolve()),
        meshes=str(args.meshes.resolve()),
        frames=indices,
        width=args.width,
        height=args.height,
        samples=args.samples,
        tea_absorption=args.tea_absorption,
        tea_color=args.tea_color,
        tea_camera_height=args.tea_camera_height,
    )
    settings["source_hash"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    previous = args.output / "render_config.json"
    if previous.exists():
        if not args.resume:
            raise FileExistsError(f"Use a fresh render directory: {args.output}")
        if json.loads(previous.read_text()) != settings:
            raise ValueError("Resume requires identical rendering settings and code")
    (args.output / "render_config.json").write_text(json.dumps(settings, indent=2))
    scene, objects, camera, bag, rod, string, tag = setup(
        cfg, args, manifest["tea_levels"]
    )
    start = time.monotonic()
    for index in indices:
        dest = args.output / "frames" / f"{index:05d}.png"
        if args.resume and dest.exists():
            continue
        data = np.load(args.meshes / f"{index:05d}.npz")
        for name, obj in objects.items():
            colored = name == "fluid" and cfg["scene"] != "tea"
            update_mesh(
                obj,
                data[name + "_vertices"],
                data[name + "_faces"],
                data["fractions"] if colored else None,
                np.power(np.array(cfg["colors"]), 2.2),
            )
        t = index / cfg["fps"]
        progress = index / max(cfg["frames"] - 1, 1)
        if bag:
            state = np.load(args.input / "states" / f"{index:05d}.npz")
            pose = state if "bag" in state else motion(cfg, t)
            bag.location = np.array(pose["bag"]) * SCALE
            rod.location = np.array(pose["rod"]) * SCALE
            rod.hide_render = t < 4.7 or not cfg.get("stirring", True)
            top = bag.location + Vector((0, 0, cfg["bag_half"][2] * SCALE))
            end = Vector(
                (
                    top.x - 0.12,
                    top.y,
                    max(top.z + 0.70, (cfg["bounds"][1][2] + 0.08) * SCALE),
                )
            )
            string.data.splines[0].points[0].co = (*top, 1)
            string.data.splines[0].points[1].co = (*end, 1)
            tag.location = end + Vector((0, 0, 0.18))
        angle = math.radians(-108 + 38 * progress)
        radius = 20.0 if cfg["scene"] == "collision" else 17.5
        camera.location = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            args.tea_camera_height if cfg["scene"] == "tea" else 10.5,
        )
        target = Vector(np.array(cfg["camera_target"]) * SCALE)
        if cfg["scene"] == "collision":
            target.z = 0.43 * SCALE
        camera.rotation_euler = (
            (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        scene.render.filepath = str(dest)
        bpy.ops.render.render(write_still=True)
        print(
            json.dumps(dict(frame=index, elapsed=time.monotonic() - start)), flush=True
        )


if __name__ == "__main__":
    main()
