"""Render plant development as a botanical cutaway with a moving camera."""
import argparse, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache/blender"))
os.environ.setdefault("OPTIX_CACHE_PATH", str(ROOT / ".cache/optix"))
import bpy
import numpy as np
from mathutils import Vector
from ice_melting import render_blender as art


def mesh_object(name, vertices, faces, mat, smooth=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(mat)
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    return obj


def wood_material(name, colors, root=False, birch=False):
    mat = art.material(name, colors[0], 0.82)
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 1.0
    tex.inputs["Detail"].default_value = 4.0
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    scale = nt.nodes.new("ShaderNodeVectorMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = (
        (35.0, 35.0, 5.0) if not root else (55.0, 55.0, 14.0)
    )
    nt.links.new(geo.outputs["Position"], scale.inputs[0])
    nt.links.new(scale.outputs[0], tex.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.23
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[0].color = (*colors[0], 1)
    ramp.color_ramp.elements[1].color = (*colors[1], 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    bump.inputs["Distance"].default_value = 0.012 if not root else 0.004
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    if birch:
        scale.inputs[1].default_value = (4.0, 4.0, 42.0)
        ramp.color_ramp.elements[0].position = 0.38
        ramp.color_ramp.elements[1].position = 0.48
    return mat


def leaf_material(cfg):
    mat = art.material("Living leaves", cfg["leaf_color"], 0.40)
    nt = mat.node_tree
    p = nt.nodes.get("Principled BSDF")
    p.inputs["Subsurface Weight"].default_value = 0.055
    p.inputs["Coat Weight"].default_value = 0.15
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 14.0
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    nt.links.new(geo.outputs["Position"], tex.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    color = np.array(cfg["leaf_color"])
    ramp.color_ramp.elements[0].color = (*tuple(color * 0.35), 1)
    ramp.color_ramp.elements[1].color = (*tuple(color * 1.7), 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], p.inputs["Base Color"])
    translucent = nt.nodes.new("ShaderNodeBsdfTranslucent")
    nt.links.new(ramp.outputs[0], translucent.inputs[0])
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.22
    nt.links.new(p.outputs[0], mix.inputs[1])
    nt.links.new(translucent.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes.get("Material Output").inputs["Surface"])
    return mat


def replace_mesh(obj, v, f):
    old = obj.data
    mesh = bpy.data.meshes.new(obj.name)
    mesh.from_pydata(v, [], f)
    mesh.update()
    for mat in old.materials:
        mesh.materials.append(mat)
    obj.data = mesh
    bpy.data.meshes.remove(old)
    for poly in mesh.polygons:
        poly.use_smooth = True


def branches(data, kind):
    p = data["p"]
    par = data["parent"]
    r = data["radius"]
    ids = np.where((data["kind"] == kind) & (par >= 0))[0]
    ids = ids[np.linalg.norm(p[ids] - p[par[ids]], axis=1) > 1.0e-6]
    if len(ids) == 0:
        return [], []
    start = p[par[ids]]
    end = p[ids]
    d = end - start
    d /= np.linalg.norm(d, axis=1)[:, None]
    ref = np.tile([0.0, 1.0, 0.0], (len(ids), 1))
    ref[np.abs(d[:, 1]) > 0.92] = [1, 0, 0]
    u = np.cross(d, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    angles = np.arange(8) * 2 * math.pi / 8
    ring = (
        np.cos(angles)[None, :, None] * u[:, None, :]
        + np.sin(angles)[None, :, None] * v[:, None, :]
    )
    # Slight overlap at branch joints closes the union without animated remeshing.
    rr = np.minimum(r[par[ids]], r[ids] * 1.7)
    vertices = np.stack(
        [
            start[:, None, :]
            + ring * rr[:, None, None]
            - d[:, None, :] * r[ids, None, None] * 0.2,
            end[:, None, :]
            + ring * r[ids, None, None]
            + d[:, None, :] * r[ids, None, None] * 0.15,
        ],
        axis=1,
    ).reshape(-1, 3)
    base = np.arange(len(ids)) * 16
    faces = []
    for j in range(8):
        nxt = (j + 1) % 8
        faces.extend(
            np.stack([base + j, base + nxt, base + 8 + nxt, base + 8 + j], 1).tolist()
        )
    return vertices.tolist(), faces


def leaves(data, cfg, simtime):
    ids = np.where(
        (data["kind"] == 0)
        & ((data["order"] >= 2) | ((data["order"] == 1) & (data["progress"] > 14)))
    )[0]
    age = np.maximum(0, simtime - data["birth"][ids])
    ids = ids[age > 0.05]
    if len(ids) == 0:
        return [], []
    ids = np.repeat(ids, 2)
    side = np.tile([-1, 1], len(ids) // 2)
    n = len(ids)
    age = np.maximum(0, simtime - data["birth"][ids])
    unfold = np.minimum(1.0, age / 7.0)
    unfold = unfold * unfold * (3 - 2 * unfold)
    roll = ids * 2.399963 + side * 1.35
    length = cfg["leaf_size"] * (0.8 + 0.25 * np.sin(ids * 11.73)) * unfold
    direction = np.stack(
        [np.cos(roll), np.sin(roll), 0.18 + 0.22 * np.sin(ids * 1.4)], 1
    )
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    across = np.stack([-np.sin(roll), np.cos(roll), np.zeros(n)], 1)
    start = data["p"][ids] + direction * 0.025 * unfold[:, None]
    # A folded, lobed blade with a raised midrib and an asymmetric droop.
    t = np.linspace(0, 1, 9)
    vertices = []
    for j, tt in enumerate(t):
        width = math.sin(math.pi * tt) ** 0.8 * 0.34
        if cfg["leaf_shape"] == "oak":
            width *= 0.72 + 0.28 * math.cos(tt * math.pi * 9)
        if cfg["leaf_shape"] == "maple":
            width = (
                math.sin(math.pi * tt) ** 0.7
                * 0.50
                * (0.65 + 0.35 * math.cos(tt * math.pi * 7))
            )
        if cfg["leaf_shape"] == "birch":
            width *= 1.0 + 0.10 * math.cos(tt * math.pi * 15)
        center = start + direction * (length * tt)[:, None]
        center[:, 2] += (0.10 * math.sin(tt * math.pi) - 0.14 * tt * tt) * length
        for sign in [-1, 0, 1]:
            point = center + across * (length * width * sign)[:, None]
            point[:, 2] -= abs(sign) * length * 0.055
            vertices.append(point)
    vertices = np.stack(vertices, 1).reshape(-1, 3)
    base = np.arange(n) * 27
    faces = []
    for j in range(8):
        for half in range(2):
            a = j * 3 + half
            b = a + 3
            faces.extend(
                np.stack([base + a, base + a + 1, base + b + 1, base + b], 1).tolist()
            )
    return vertices.tolist(), faces


def interpolated_state(path, time_value):
    a = int(math.floor(time_value))
    b = int(math.ceil(time_value))
    f = time_value - a
    with np.load(path / "states" / f"{a:05d}.npz") as data:
        prev = {k: data[k] for k in data.files}
    if a == b:
        return prev
    with np.load(path / "states" / f"{b:05d}.npz") as data:
        nxt = {k: data[k] for k in data.files}
    count = len(prev["p"])
    nxt["p"][:count] = prev["p"] * (1 - f) + nxt["p"][:count] * f
    nxt["radius"][:count] = prev["radius"] * (1 - f) + nxt["radius"][:count] * f
    for i in range(count, len(nxt["p"])):
        par = nxt["parent"][i]
        nxt["p"][i] = nxt["p"][par] + (nxt["p"][i] - nxt["p"][par]) * f
        nxt["radius"][i] *= max(0.001, f)
    return nxt


def setup(cfg, args):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.denoising_use_gpu = True
    pref = bpy.context.preferences.addons["cycles"].preferences
    pref.compute_device_type = "OPTIX"
    pref.get_devices()
    for dev in pref.devices:
        dev.use = dev.type == "OPTIX"
    sc.cycles.device = "GPU"
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 8
    sc.cycles.seed = 31
    sc.cycles.use_animated_seed = False
    sc.render.resolution_x = args.width
    sc.render.resolution_y = args.height
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = 0.0
    world = bpy.data.worlds.new("Soft daylight")
    world.use_nodes = True
    sc.world = world
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.38,
        0.43,
        0.47,
        1,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    floor = art.material("Slate studio", (0.055, 0.068, 0.072), 0.8)
    bottom = -cfg["_display_depth"]
    art.cube("Infinite ground", (0, 0, bottom - 0.15), (200, 200, 0.2), floor, 0)
    bark = wood_material("Bark", [(0.09, 0.045, 0.018), (0.28, 0.17, 0.085)])
    if cfg["scene"] == "birch_lateral":
        bark = wood_material(
            "Birch bark", [(0.065, 0.052, 0.035), (0.75, 0.73, 0.65)], birch=True
        )
    root = wood_material(
        "Root bark", [(0.16, 0.068, 0.025), (0.46, 0.26, 0.11)], root=True
    )
    leaf = leaf_material(cfg)
    # An open-front geological section exposes the entire foreground root system.
    earth = wood_material(
        "Soil texture", [(0.012, 0.007, 0.004), (0.052, 0.028, 0.012)], root=True
    )
    sand = wood_material(
        "Subsoil", [(0.028, 0.014, 0.006), (0.090, 0.047, 0.018)], root=True
    )
    stone = wood_material(
        "Bedrock", [(0.035, 0.039, 0.038), (0.13, 0.14, 0.12)], root=True
    )
    for mat in [earth, sand, stone]:
        for node in mat.node_tree.nodes:
            if node.bl_idname == "ShaderNodeVectorMath":
                node.inputs[1].default_value = (13.0, 13.0, 13.0)
            if node.bl_idname == "ShaderNodeBump":
                node.inputs["Distance"].default_value = 0.022
    split = -1.5 if cfg["scene"] == "birch_lateral" else -2.0
    for name, z, h, mat in [
        ("Humus", -0.20, 0.40, earth),
        ("Sandy loam", (-0.4 + split) / 2.0, -0.4 - split, sand),
        ("Deep soil", (split + bottom) / 2.0, split - bottom, earth),
    ]:
        art.cube(name, (0, 2.85, z), (7.2, 0.65, h), mat, 0.035)
    art.cube("Surface soil", (0, 1.35, -0.09), (7.2, 3.65, 0.18), earth, 0.025)
    art.cube(
        "Cutaway plinth",
        (0, 0.4, bottom - 0.02),
        (7.5, 5.6, 0.13),
        art.material("Charcoal base", (0.035, 0.043, 0.045), 0.4),
        0.04,
    )
    for rock in cfg["rocks"]:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=40, ring_count=24, location=rock["center"]
        )
        obj = bpy.context.object
        obj.name = "Impermeable rock"
        obj.scale = rock["radii"]
        obj.data.materials.append(stone)
        for vertex in obj.data.vertices:
            q = vertex.co
            factor = 1.0 - 0.065 * (
                0.5 + 0.5 * math.sin(5 * q.x + 3 * q.y) * math.cos(4 * q.z - 2 * q.x)
            )
            vertex.co *= factor
        for face in obj.data.polygons:
            face.use_smooth = True
    # Small stones give the soil section scale without hiding the roots.
    rng = np.random.default_rng(173)
    for i in range(38):
        pos = [
            rng.uniform(-3.35, 3.35),
            rng.uniform(2.48, 2.55),
            rng.uniform(bottom + 0.2, -0.2),
        ]
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=1, radius=rng.uniform(0.025, 0.09), location=pos
        )
        ob = bpy.context.object
        ob.scale = (1.0, 0.55, 0.7)
        ob.data.materials.append(stone)
    art.area("Daylight", (-4, -6, 9), (0, 0, 0.4), 2300, 6, 6, (1, 0.96, 0.88))
    art.area("Leaf edge", (3, 3, 8), (0, 0, 2), 1900, 5, 5, (0.91, 0.96, 1))
    art.area("Root softbox", (-1, -4, -1), (0, 0, -2), 350, 5, 4, (1, 0.97, 0.91))
    art.area("Right fill", (5, -1, 3), (0, 0, 0), 1100, 4, 6)
    objects = [
        mesh_object("Growing trunk", [], [], bark),
        mesh_object("Growing roots", [], [], root),
        mesh_object("Growing foliage", [], [], leaf),
    ]
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    sc.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 11.0
    camera.data.lens = 55
    camera.data.clip_end = 200
    return sc, objects, camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--indices", type=int, nargs="*")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--samples", type=int, default=24)
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    cfg = json.loads((args.input / "config.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    final = np.load(args.input / "states" / f"{cfg['steps']:05d}.npz")
    cfg["_display_depth"] = max(2.35, float(-final["p"][:, 2].min()) + 0.42)
    top = float(final["p"][:, 2].max()) + 0.40
    bottom = -cfg["_display_depth"] - 0.1
    center = (top + bottom) / 2
    vertical = top - bottom + 1.6
    sc, objects, camera = setup(cfg, args)
    start = time.monotonic()
    indices = args.indices if args.indices is not None else range(args.frames)
    for frame in indices:
        t = frame / max(1, args.frames - 1)
        development = min(1.0, t / 0.90) * cfg["steps"]
        state = interpolated_state(args.input, development)
        for kind in [0, 1]:
            replace_mesh(objects[kind], *branches(state, kind))
        replace_mesh(objects[2], *leaves(state, cfg, development))
        angle = math.radians(cfg["camera_start"] + cfg["camera_orbit"] * t)
        camera.location = (18 * math.cos(angle), 18 * math.sin(angle), center + 3.0)
        target = Vector((0, 0, center))
        camera.rotation_euler = (
            (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        # Establishment begins closer and opens to the complete root/crown system.
        camera.data.ortho_scale = vertical * args.width / args.height
        sc.render.filepath = str(
            (args.output / "frames" / f"{frame:05d}.png").resolve()
        )
        bpy.ops.render.render(write_still=True)
        print(
            json.dumps(
                dict(
                    frame=frame, elapsed=time.monotonic() - start, nodes=len(state["p"])
                )
            ),
            flush=True,
        )
    (args.output / "render.json").write_text(
        json.dumps(
            dict(
                frames=args.frames,
                width=args.width,
                height=args.height,
                samples=args.samples,
                input=str(args.input),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
