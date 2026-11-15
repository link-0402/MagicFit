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
