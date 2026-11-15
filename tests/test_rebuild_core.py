"""Headless tests for the Mesh Repair tab's Mesh Rebuild, Normals and UV tools (`magic_fit.resurface`, from
Resurface): each button's operator on a UV sphere, and that they sit next to a standalone Resurface without
clashing.

Run from the repository root:
    blender -b --factory-startup --python tests/test_rebuild_core.py
"""

import math
import os
import shutil
import struct
import sys
import tempfile
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import magic_fit  # noqa: E402
from magic_fit.resurface import blender_io, operators, ui, uvtools  # noqa: E402
from magic_fit.resurface.core import tex  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def run(op, what, **kwargs):
    try:
        result = op(**kwargs)
    except RuntimeError as ex:
        result = str(ex)
    check(result == {'FINISHED'}, "{:s} ({!r})".format(what, result))


def sphere():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    obj = bpy.context.active_object
    obj.select_set(True)
    return obj


def test_names():
    """Its operators are Magic Fit's and its settings Scene.magic_fit_repair: none is named like Resurface's."""
    magic_fit.register()
    try:
        idnames = [cls.bl_idname for module in (operators, uvtools) for cls in module.classes]
        check(idnames and all(idname.startswith("magic_fit.") for idname in idnames),
              "operators are magic_fit.* ({:d})".format(len(idnames)))
        check(all(hasattr(bpy.ops.magic_fit, idname.split(".")[1]) for idname in idnames), "operators registered")
        check("magic_fit.reset_rebuild_settings" in idnames, "Reset to Defaults is there")
        check(hasattr(bpy.context.scene, "magic_fit_repair") and not hasattr(bpy.context.scene, "resurface"),
              "settings are Scene.magic_fit_repair, Resurface's Scene.resurface is left free")
        check(not hasattr(bpy.types.WindowManager, "resurface_tab"), "no tab property of Resurface's own")
        panels = {cls.__name__ for cls in ui.classes}
        check(all(cls.bl_category == "Magic Fit" and cls.tab == 'REPAIR' for cls in ui.classes),
              "the panels are on the Magic Fit tab's Mesh Repair tab")
        check(not any(name.startswith("RESURFACE_") for name in panels), "no panel is named like Resurface's")
        check({sub for cls in ui.classes for sub in cls.subtabs} == {'REBUILD', 'NORMALS', 'UV'},
              "they're on the Mesh Rebuild, Normals and UV sub-tabs")
    finally:
        magic_fit.unregister()
    check(not hasattr(bpy.types.Scene, "magic_fit_repair"), "unregistering removes the settings")


def test_tools():
    """Every button's operator runs on a sphere, as in Resurface's own smoke test."""
    magic_fit.register()
    try:
        obj = sphere()
        faces_before = len(obj.data.polygons)
        settings = bpy.context.scene.magic_fit_repair

        run(bpy.ops.magic_fit.analyze_mesh, "Analyze")
        check("triangles" in settings.report_mesh, "Analyze reports the triangles")

        run(bpy.ops.magic_fit.rebuild_mesh, "Rebuild Mesh")
        mesh = obj.data
        check(len(mesh.polygons) > 100, "the rebuilt mesh has faces ({:d})".format(len(mesh.polygons)))
        check(all(len(p.vertices) == 3 for p in mesh.polygons), "the rebuilt mesh is all triangles")
        check(len(mesh.uv_layers) > 0, "UVs transferred")
        check(blender_io.original_name(obj), "the original is kept")
        check("Rebuilt 1 object" in settings.report_mesh, "Rebuild reports what it did")

        run(bpy.ops.magic_fit.smooth_normals, "Smooth Normals")
        check(settings.report_normals.startswith("Smoothed normals"), "Smooth Normals reports")

        # A flat normal map: the check can't tell, so Fix only flips it when asked to.
        material = bpy.data.materials.new("Normal Map Test")
        material.use_nodes = True
        tree = material.node_tree
        flat = bpy.data.images.new("Flat Normal", 64, 64)
        flat.generated_color = (0.5, 0.5, 1.0, 1.0)
        image_node = tree.nodes.new("ShaderNodeTexImage")
        image_node.image = flat
        normal_node = tree.nodes.new("ShaderNodeNormalMap")
        tree.links.new(image_node.outputs["Color"], normal_node.inputs["Color"])
        bsdf = next(node for node in tree.nodes if node.type == 'BSDF_PRINCIPLED')
        tree.links.new(normal_node.outputs["Normal"], bsdf.inputs["Normal"])
        obj.data.materials.append(material)
        run(bpy.ops.magic_fit.check_normal_maps, "Check Normal Maps")
        check("Flat Normal" in settings.report_normalmap, "the check reports the map")
        run(bpy.ops.magic_fit.fix_normal_maps, "Fix Normal Maps")
        check(image_node.image == flat, "a map the check can't judge is left alone")
        run(bpy.ops.magic_fit.fix_normal_maps, "Fix Normal Maps (also unclear ones)", include_unclear=True)
        check(image_node.image.name == "Flat Normal_green_up", "a flipped copy is put in place")

        # From File: the texture the game uses, read on the active material's faces instead of its own map.
        folder = tempfile.mkdtemp()
        game = bpy.data.images.new("game_n", 64, 64)
        game.generated_color = (0.5, 0.3, 0.9, 1.0)
        game.filepath_raw = os.path.join(folder, "game_n.png")
        game.file_format = 'PNG'
        game.save()
        bpy.data.images.remove(game)
        in_material = image_node.image
        images_before = len(bpy.data.images)
        settings.normalmap_file = os.path.join(folder, "game_n.png")
        run(bpy.ops.magic_fit.check_normal_maps, "Check Normal Maps from a file")
        check("game_n.png (on Normal Map Test)" in settings.report_normalmap,
              "the check reports the file and the material it was read on")
        run(bpy.ops.magic_fit.fix_normal_maps, "Fix Normal Maps from a file", include_unclear=True)
        flipped = os.path.join(folder, "game_n_green_up.png")
        check(os.path.isfile(flipped), "Fix saves <name>_green_up.png next to the file")
        check(image_node.image == in_material and len(bpy.data.images) == images_before,
              "the materials and the .blend's images are left alone")
        if os.path.isfile(flipped):
            pixels = blender_io.read_image_file(flipped)
            check(abs(float(pixels[..., 1].mean()) - 0.7) < 2 / 255 and abs(float(pixels[..., 2].mean()) - 0.9) < 2 / 255,
                  "green is flipped and blue kept ({:.3f}, {:.3f})".format(pixels[..., 1].mean(), pixels[..., 2].mean()))

        # A .tex from the mod's folder: read as the game stores it (row 0 at the top), fixed as a .tex.
        rgba = np.zeros((64, 32, 4), np.uint8)
        rgba[..., 1], rgba[..., 2], rgba[..., 3] = 77, 230, 255
        rgba[:8, :, 0] = 255                                    # the top rows of the texture are red
        tex.write_tex(os.path.join(folder, "game_n.tex"), rgba, mips=4)
        pixels = blender_io.read_image_file(os.path.join(folder, "game_n.tex"))
        check(pixels.shape == (64, 32, 4) and pixels[-1, 0, 0] == 1.0 and pixels[0, 0, 0] == 0.0,
              ".tex pixels arrive the way Blender stores images (row 0 at the bottom)")
        settings.normalmap_file = os.path.join(folder, "game_n.tex")
        run(bpy.ops.magic_fit.check_normal_maps, "Check Normal Maps from a .tex")
        run(bpy.ops.magic_fit.fix_normal_maps, "Fix Normal Maps from a .tex", include_unclear=True)
        flipped = os.path.join(folder, "game_n_green_up.tex")
        check(os.path.isfile(flipped), "Fix saves <name>_green_up.tex next to the .tex")
        if os.path.isfile(flipped):
            with open(flipped, "rb") as f:
                data = f.read()
            header = tex.read_header(data)
            back, _ = tex.read_tex(data)
            check(header["format"] == tex.B8G8R8A8 and header["mips"] == 4,
                  "it's uncompressed, with the original's 4 mip levels ({:#x}, {:d})".format(
                      header["format"], header["mips"]))
            expected = rgba.copy()
            expected[..., 1] = 255 - 77
            check(np.array_equal(back, expected), "only green is flipped, exactly, and nothing moved")

        # A compressed one (BC1, 8x8: red, blue / blue, green blocks) goes through Blender's OpenImageIO.
        def block(c565):
            return struct.pack("<HHI", c565, c565, 0)
        bc1 = struct.pack("<IIHHHBB", tex.TYPE_2D, tex.BC1, 8, 8, 1, 1, 0)
        bc1 += struct.pack("<3I", 0, 0, 0) + struct.pack("<13I", 80, *([0] * 12))
        bc1 += block(0xF800) + block(0x001F) + block(0x001F) + block(0x07E0)
        with open(os.path.join(folder, "bc1.tex"), "wb") as f:
            f.write(bc1)
        pixels = blender_io.read_image_file(os.path.join(folder, "bc1.tex"))
        check(pixels[7, 0, 0] > 0.8 and pixels[0, 7, 1] > 0.8 and pixels[7, 7, 2] > 0.8,
              "a BC1 .tex decodes, top left red, bottom right green")
        settings.normalmap_file = ""
        shutil.rmtree(folder, ignore_errors=True)

        run(bpy.ops.magic_fit.rebuild_uvs, "Rebuild UVs")
        check(mesh.uv_layers.get("Old UVs") is not None, "the Old UVs layer is kept")

        settings.idx_width = settings.idx_height = 256
        run(bpy.ops.magic_fit.index_map, "Index Map Generator")
        check(any(image.size[0] == 256 for image in bpy.data.images), "the index map image is made")

        run(bpy.ops.magic_fit.resize_uv_canvas, "Resize Canvas")

        run(bpy.ops.magic_fit.clear_repair_report, "Clear a result", prop="report_mesh")
        check(settings.report_mesh == "", "the result's X hides it")

        run(bpy.ops.magic_fit.restore_original, "Restore Original")
        check(len(obj.data.polygons) == faces_before, "the original mesh is back")
    finally:
        magic_fit.unregister()


def shape(pieces, name="Shape"):
    """One object from pieces: lists of (verts, faces) in metres, triangulated."""
    import bmesh
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    bm = bmesh.new()
    for verts, faces in pieces:
        made = [bm.verts.new(v) for v in verts]
        for f in faces:
            bm.faces.new([made[i] for i in f])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def box(sx, sy, sz, cuts=10, bend=False):
    """A closed box sx x sy x sz (metres) centred at 0, each side a grid of quads. bend: bent 90° along x."""
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=True)
    R = sx / (math.pi / 2)
    verts = []
    for v in bm.verts:
        x, y, z = v.co.x * sx, v.co.y * sy, v.co.z * sz
        if bend:
            t = x / R
            x, z = (R + z) * math.sin(t), (R + z) * math.cos(t) - R
        verts.append((x, y, z))
    faces = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    return verts, faces


def folded_sheet(width, depth, flap, offset=0.0, n=20):
    """An open sheet facing up, `width` x `depth`, with its far long border folded down by `flap`,
    moved down and back by `offset`."""
    rows = [i * depth / n for i in range(n + 1)] + [depth + flap * k / 4 for k in range(1, 5)]
    verts = []
    for y in rows:
        for i in range(n + 1):
            x = -width / 2 + width * i / n
            yy, z = (y, 0.0) if y <= depth else (depth, depth - y)
            verts.append((x, yy - offset, z - offset))
    m = n + 1
    faces = [[r * m + i, r * m + i + 1, (r + 1) * m + i + 1, (r + 1) * m + i]
             for r in range(len(rows) - 1) for i in range(n)]
    return verts, faces


def rebuild(obj, **kw):
    from magic_fit.resurface.pipeline import RebuildJob, RebuildSettings
    before = sum(p.area for p in obj.data.polygons)
    report = RebuildJob([obj], RebuildSettings(**kw)).run()
    return sum(p.area for p in obj.data.polygons) / before, report


def test_thin_rim_shell():
    """A closed shell 0.4 mm thick, bent like a cup, keeps its shape with default settings."""
    kept, report = rebuild(shape([box(0.1, 0.04, 0.0004, bend=True)]))
    check(kept >= 0.97, "the thin shell keeps its area ({:.3f})".format(kept))
    check(report["features"].get("thin_rim_edges", 0) > 0, "its rim is found")


def test_thin_rim_nested():
    """A thin shell inside another thin shell, joined into one object: both rims are found."""
    pieces = [box(0.1, 0.04, 0.001, bend=True), box(0.096, 0.036, 0.0004, bend=True)]
    kept, _ = rebuild(shape(pieces))
    kept_sharp, _ = rebuild(shape(pieces), preserve_sharp=True)
    check(abs(kept - kept_sharp) <= 0.01,
          "nested shells keep as much as with Sharp Edges ({:.3f}, {:.3f})".format(kept, kept_sharp))


def test_thin_rim_thick_box():
    """A 100 x 40 x 20 mm box keeps no thin rims: its other side is farther than a triangle."""
    _, report = rebuild(shape([box(0.1, 0.04, 0.02)]))
    check(report["features"].get("thin_rim_edges", -1) == 0, "no thin rims on a thick box")


def test_thin_rim_same_facing():
    """One object made of two open 100 x 40 mm sheets 0.4 mm apart, both facing up, each with one long
    border folded down 90°: the sheet below is not the other side of the sheet above."""
    obj = shape([folded_sheet(0.1, 0.04, 0.005), folded_sheet(0.1, 0.04, 0.005, offset=0.0004)])
    _, report = rebuild(obj)
    check(report["features"].get("thin_rim_edges", -1) == 0, "same-facing layers are not thin rims")


def rounded_step(width=0.1, depth=0.05, step=0.004, turn=65.0, n=50, flip_from=None):
    """An open sheet facing up, `width` x `depth`, with one step down by `step` across it. Each of the two
    step corners turns `turn` degrees over 3 segments of 0.5 mm, like the waist folds of a sculpted shirt.
    flip_from: faces past this y are wound the other way. Returns (verts, faces, corner ys)."""
    seg, flat = 0.0005, 0.002
    t = math.radians(turn)
    drop_corners = 2 * sum(seg * math.sin(t * k / 3) for k in (1, 2, 3))
    slope = max((step - drop_corners) / math.sin(t), 0.0)
    prof = [(0.0, 0.0)]

    def walk(angle, length):
        y, z = prof[-1]
        prof.append((y + length * math.cos(angle), z - length * math.sin(angle)))

    while prof[-1][0] < 0.4 * depth:
        walk(0.0, flat)
    corners = [prof[-1][0]]
    for k in (1, 2, 3):
        walk(t * k / 3, seg)
    walk(t, slope)
    for k in (2, 1, 0):
        walk(t * k / 3, seg)
    corners.append(prof[-1][0])
    while prof[-1][0] < depth:
        walk(0.0, flat)
    verts = [(-width / 2 + width * i / n, y, z) for y, z in prof for i in range(n + 1)]
    m = n + 1
    faces = []
    for r in range(len(prof) - 1):
        for i in range(n):
            f = [r * m + i, r * m + i + 1, (r + 1) * m + i + 1, (r + 1) * m + i]
            faces.append(f[::-1] if flip_from is not None and prof[r][0] >= flip_from else f)
    return verts, faces, corners


def fold_cut(obj, **kw):
    """Rebuild obj and return (largest distance in metres from a source vertex within 2 mm of a step
    corner to the new surface, new triangle count). The corners are in obj["corners"]."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    near = [v.co.copy() for v in obj.data.vertices
            if min(abs(v.co.y - c) for c in obj["corners"]) < 0.002]
    rebuild(obj, **kw)
    me = obj.data
    tree = BVHTree.FromPolygons([v.co for v in me.vertices], [p.vertices[:] for p in me.polygons])
    return max(tree.find_nearest(Vector(p))[3] for p in near), len(me.polygons)


def step_object(**kw):
    verts, faces, corners = rounded_step(**kw)
    obj = shape([(verts, faces)])
    obj["corners"] = corners
    return obj


def tri_count(obj, **kw):
    rebuild(obj, **kw)
    return len(obj.data.polygons)


def test_normal_refine_fold():
    """A rounded fold corner (65° over 1.5 mm, no edge over 22°) is followed much more closely with
    Normal Tolerance 15° than with the distance check alone, for at most three times the triangles."""
    common = dict(density=0.4, smooth_normals=False)
    cut0, tris0 = fold_cut(step_object(), normal_tolerance=0, **common)
    cut15, tris15 = fold_cut(step_object(), normal_tolerance=15, **common)
    check(cut15 <= 0.5 * cut0, "the fold corner is followed ({:.3f} mm, was {:.3f} mm)".format(cut15 * 1e3, cut0 * 1e3))
    check(tris15 <= 3 * tris0, "for at most 3x the triangles ({:,} -> {:,})".format(tris0, tris15))


def test_normal_refine_flat():
    """A flat grid and a sphere have no folds: Normal Tolerance adds (almost) no triangles."""
    import bmesh
    n = 50
    grid = ([(0.1 * i / n, 0.1 * j / n, 0.0) for j in range(n + 1) for i in range(n + 1)],
            [[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i]
             for j in range(n) for i in range(n)])
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=0.1)
    ball = ([v.co[:] for v in bm.verts], [[v.index for v in f.verts] for f in bm.faces])
    bm.free()
    for name, piece in (("grid", grid), ("sphere", ball)):
        t0 = tri_count(shape([piece]), normal_tolerance=0, smooth_normals=False)
        t15 = tri_count(shape([piece]), normal_tolerance=15, smooth_normals=False)
        check(abs(t15 - t0) <= 0.02 * t0, "{}: no extra triangles ({:,} -> {:,})".format(name, t0, t15))


def test_normal_refine_noisy_source():
    """Decimation noise is not a fold: a flat grid of 2 mm squares whose points jitter up and down by up to
    0.5 mm gets at most 15% more triangles from Normal Tolerance 20° (single source faces: 43% more)."""
    import random
    rnd = random.Random(7)
    verts, faces = flat_grid(n=50)
    noisy = ([(x, y, rnd.uniform(-0.0005, 0.0005)) for x, y, _ in verts], faces)
    t0 = tri_count(shape([noisy]), normal_tolerance=0, smooth_normals=False)
    t20 = tri_count(shape([noisy]), normal_tolerance=20, smooth_normals=False)
    check(t20 <= 1.15 * t0, "noise adds few triangles ({:,} -> {:,})".format(t0, t20))


def test_normal_refine_low_poly():
    """A low-poly source (a 16 x 8 sphere, 22.5° between facets) is followed facet by facet: the new
    triangles lie on its flat facets, so Normal Tolerance 20° adds at most 15% triangles."""
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=0.1)
    ball = ([v.co[:] for v in bm.verts], [[v.index for v in f.verts] for f in bm.faces])
    bm.free()
    t0 = tri_count(shape([ball]), normal_tolerance=0, smooth_normals=False)
    t20 = tri_count(shape([ball]), normal_tolerance=20, smooth_normals=False)
    check(t20 <= 1.15 * t0, "a low-poly sphere adds few triangles ({:,} -> {:,})".format(t0, t20))


def pillow(size=0.06, h=0.001, n=12):
    """A thin closed shell like a bra cup: a top sheet facing up and a bottom sheet facing down, up to
    2*h apart in the middle and meeting at a sharp rim along their shared border."""
    verts, faces = [], []
    index = {}
    for side, sign in (("top", 1.0), ("bottom", -1.0)):
        for j in range(n + 1):
            for i in range(n + 1):
                rim = i in (0, n) or j in (0, n)
                key = ("rim", i, j) if rim else (side, i, j)
                if key not in index:
                    z = sign * h * math.sin(math.pi * i / n) * math.sin(math.pi * j / n)
                    index[key] = len(verts)
                    verts.append((size * i / n, size * j / n, z))
        for j in range(n):
            for i in range(n):
                q = [index[("rim", a, b) if a in (0, n) or b in (0, n) else (side, a, b)]
                     for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
                faces.append(q if sign > 0 else q[::-1])
    return verts, faces


def test_normal_refine_thin_shell():
    """The sharp rim of a thin closed shell is not a fold to refine: its two sides point opposite ways,
    and averaging them gives a sideways normal. Normal Tolerance 20° adds at most 15% triangles."""
    t0 = tri_count(shape([pillow()]), normal_tolerance=0, smooth_normals=False)
    t20 = tri_count(shape([pillow()]), normal_tolerance=20, smooth_normals=False)
    check(t20 <= 1.15 * t0, "the thin shell's rim adds few triangles ({:,} -> {:,})".format(t0, t20))


def test_normal_refine_flipped_source():
    """Faces wound the other way are not folds: the orientation of the source does not count."""
    plain = tri_count(step_object(), normal_tolerance=15, smooth_normals=False)
    flipped = tri_count(step_object(flip_from=0.04), normal_tolerance=15, smooth_normals=False)
    check(flipped <= 1.1 * plain, "a reversed part adds no triangles ({:,} -> {:,})".format(plain, flipped))


def test_normal_refine_adaptive_off():
    """With Adaptive Detail off, Normal Tolerance does nothing."""
    t0 = tri_count(step_object(), adaptive=False, normal_tolerance=0, smooth_normals=False)
    t15 = tri_count(step_object(), adaptive=False, normal_tolerance=15, smooth_normals=False)
    check(t0 == t15, "same triangles with Adaptive Detail off ({:,}, {:,})".format(t0, t15))


def flat_grid(n=20, size=0.1, x0=0.0):
    return ([(x0 + size * i / n, size * j / n, 0.0) for j in range(n + 1) for i in range(n + 1)],
            [[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i]
             for j in range(n) for i in range(n)])


def grid_object(name, x0=0.0, broken=None):
    """A flat grid facing up as its own object. broken: share of corners whose custom normal points down
    (None: no custom normals, 0: custom normals equal to the face normals)."""
    verts, faces = flat_grid(x0=x0)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    if broken is not None:
        nl = len(mesh.loops)
        mesh.normals_split_custom_set([(0.0, 0.0, -1.0) if i < broken * nl else (0.0, 0.0, 1.0) for i in range(nl)])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_normals_broken():
    """Normals count as broken only when custom normals point far from their faces."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for broken, expect, what in ((None, False, "no custom normals"), (0.0, False, "custom normals along the faces"),
                                 (0.03, False, "3% of the corners reversed (clean game meshes reach ~2%)"),
                                 (0.1, True, "10% of the corners reversed")):
        obj = grid_object("G", broken=broken)
        got = blender_io.normals_broken(blender_io.ObjectData(obj))
        check(got == expect, "{}: broken is {}".format(what, got))
        bpy.data.objects.remove(obj)
    obj = grid_object("Turned", broken=0.0)
    obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.context.view_layer.update()
    check(not blender_io.normals_broken(blender_io.ObjectData(obj)), "a turned object with clean normals is not broken")
    bpy.data.objects.remove(obj)
    obj = grid_object("Mirrored", broken=0.0)
    obj.scale = (-1.0, 1.0, 1.0)
    bpy.context.view_layer.update()
    check(not blender_io.normals_broken(blender_io.ObjectData(obj)), "a mirrored object with clean normals is not broken")
    bpy.data.objects.remove(obj)


def test_saved_smooth_normals_off():
    """A file saved with the old Smooth Normals toggle off opens with Smooth Normals: Off, not Auto."""
    from magic_fit.resurface import props
    magic_fit.register()
    try:
        check(props.migrate_rebuild_normals in bpy.app.handlers.load_post, "the migration runs when a file loads")
        s = bpy.context.scene.magic_fit_repair
        for old, expect in ((0, "OFF"), (1, "ON")):
            s.rebuild_normals_mode = "AUTO"
            s.bl_system_properties_get(do_create=True)["rebuild_normals"] = old
            props.migrate_rebuild_normals(None)
            check(s.rebuild_normals_mode == expect, "old toggle {} opens as {} ({})".format(old, expect, s.rebuild_normals_mode))
            check("rebuild_normals" not in s.bl_system_properties_get(), "the old value is dropped")
        props.migrate_rebuild_normals(None)
        check(s.rebuild_normals_mode == "ON", "without an old value nothing changes")
        s.property_unset("rebuild_normals_mode")
    finally:
        magic_fit.unregister()


def test_auto_normals_per_object():
    """Auto smooths only the object whose source normals are broken; the other keeps its shading."""
    from magic_fit.resurface.pipeline import RebuildJob, RebuildSettings
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    clean = grid_object("Clean")
    broken = grid_object("Broken", x0=0.2, broken=0.1)
    report = RebuildJob([clean, broken], RebuildSettings(smooth_normals="AUTO")).run()
    check(report.get("normals", {}).get("auto_smoothed") == ["Broken"],
          "Auto smoothed only the broken one ({})".format(report.get("normals")))
    ups = [n.vector.z for n in clean.data.corner_normals]
    check(min(ups) > math.cos(math.radians(1.0)), "the clean one keeps its shading (min z {:.4f})".format(min(ups)))


def test_export_estimate_cube():
    """A flat-shaded cube with one UV island per face exports 24 vertices."""
    import bmesh
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    uv = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        for k, loop in enumerate(f.loops):
            loop[uv].uv = (f.index / 6 + (k in (1, 2)) / 7, (k in (2, 3)) / 7)
    mesh = bpy.data.meshes.new("Cube")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = False
    obj = bpy.data.objects.new("Cube", mesh)
    bpy.context.scene.collection.objects.link(obj)
    n = blender_io.export_vertex_estimate(obj)
    check(n == 24, "cube: 24 vertices after export ({})".format(n))


def test_export_estimate_smooth():
    """A smooth UV sphere splits only where its UVs split."""
    obj = sphere()
    me = obj.data
    for p in me.polygons:
        p.use_smooth = True
    uv = me.uv_layers.active.data
    expect = len({(l.vertex_index, round(uv[l.index].uv[0], 4), round(uv[l.index].uv[1], 4)) for l in me.loops})
    n = blender_io.export_vertex_estimate(obj)
    check(n == expect, "sphere: {} vertices after export ({} expected, {} in Blender)".format(n, expect, len(me.vertices)))


def test_export_estimate_no_uv():
    """A smooth mesh without UV maps counts its vertices and does not fail."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    obj = grid_object("NoUV")
    for p in obj.data.polygons:
        p.use_smooth = True
    n = blender_io.export_vertex_estimate(obj)
    check(n == len(obj.data.vertices), "no UVs: {} vertices after export ({})".format(n, len(obj.data.vertices)))


def test_rebuild_defaults():
    """New scenes rebuild with Density 0.5 and Normal Tolerance 20°."""
    magic_fit.register()
    try:
        settings = bpy.context.scene.magic_fit_repair
        check(abs(settings.density - 0.5) < 1e-6, "Density defaults to 0.5 ({:.2f})".format(settings.density))
        check(abs(settings.normal_tolerance - math.radians(20.0)) < 1e-6, "Normal Tolerance defaults to 20°")
        check(abs(settings.to_settings().normal_tolerance - 20.0) < 1e-4, "the rebuild gets it in degrees")
        check(settings.rebuild_normals_mode == "AUTO" and settings.to_settings().smooth_normals == "AUTO",
              "Smooth Normals defaults to Auto")
    finally:
        magic_fit.unregister()


def test_reset_rebuild_settings():
    """Reset to Defaults puts the Rebuild Mesh settings back and leaves the other tools' settings alone."""
    magic_fit.register()
    try:
        s = bpy.context.scene.magic_fit_repair
        s.density, s.tolerance, s.preserve_sharp = 1.5, 10.0, True
        s.normals_angle, s.canvas_new_w = math.radians(30.0), 512
        run(bpy.ops.magic_fit.reset_rebuild_settings, "Reset to Defaults")
        rna = s.bl_rna.properties
        check(s.density == rna["density"].default and s.tolerance == rna["tolerance"].default
              and s.preserve_sharp == rna["preserve_sharp"].default, "the Rebuild Mesh settings are back")
        check(abs(s.normals_angle - math.radians(30.0)) < 1e-6 and s.canvas_new_w == 512,
              "the Smooth Normals and Resize Canvas settings are unchanged")
    finally:
        magic_fit.unregister()


def main():
    tests =[value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception:
            FAILURES.append(test.__name__ + " raised")
            traceback.print_exc()
    print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
    sys.exit(1 if FAILURES else 0)


main()
