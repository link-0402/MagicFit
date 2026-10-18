"""Squish a game model's clothing into its own skin, everywhere at once, and report how it went.

    blender -b --factory-startup --python tools/squish_check.py -- MODEL.mdl OUT_DIR [--press M] [--bulge M] [--width M]
        [--seam M]

The model's submeshes drawn with the game's skin material are the body, the others the clothing. Prints clothing
vertices inside the body (more than 0.5 mm, before and after), the deepest press, the skin's move at the fabric
edge, the highest bulge, and triangles turned over; renders the model before and after (front, side, and close-ups of the
lowest hem) with Workbench into OUT_DIR. Never saves a .blend.
"""

import argparse
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magic_fit import facedata, squishing  # noqa: E402


def joined(parts):
    co, tris, first = [], [], 0
    for part in parts:
        co.append(np.asarray(part.points, dtype=np.float64))
        tris.append(np.asarray(part.tris, dtype=np.int64).reshape(-1, 3) + first)
        first += len(part.points)
    return np.concatenate(co), np.concatenate(tris)


def normals(co, tris):
    n = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def turned_over(before, after, tris):
    big = np.linalg.norm(np.cross(before[tris[:, 1]] - before[tris[:, 0]], before[tris[:, 2]] - before[tris[:, 0]]),
                         axis=1) > 1e-12
    return int(((np.einsum("ij,ij->i", normals(before, tris), normals(after, tris)) < 0.0) & big).sum())


def inside(body_co, body_tris, cloth_co):
    distance = squishing._surface(body_co, body_tris).nearest(cloth_co).distance
    return int((distance < -0.0005).sum())


def crossings(body_co, body_tris, cloth_co, cloth_tris):
    """How many pairs of skin and clothing triangles cut through each other."""
    from mathutils.bvhtree import BVHTree
    body = BVHTree.FromPolygons(body_co.tolist(), body_tris.tolist(), all_triangles=True)
    cloth = BVHTree.FromPolygons(cloth_co.tolist(), cloth_tris.tolist(), all_triangles=True)
    return len(body.overlap(cloth))


def make_object(name, co, tris, color):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(co.tolist(), [], tris.tolist())
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    obj.color = color
    bpy.context.scene.collection.objects.link(obj)
    return obj


def render(scene, path, target, direction, scale):
    camera = scene.camera
    direction = Vector(direction).normalized()
    camera.location = Vector(target) + direction * 3.0
    camera.rotation_euler = (-direction).to_track_quat('-Z', 'Z').to_euler()
    camera.data.ortho_scale = scale
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    # The Chest preset.
    parser.add_argument("--press", type=float, default=0.003)
    parser.add_argument("--bulge", type=float, default=0.009)
    parser.add_argument("--width", type=float, default=0.07)
    # The brush's Seam Distance.
    parser.add_argument("--seam", type=float, default=0.0001)
    args = parser.parse_args(argv)

    with open(args.model, "rb") as f:
        parts = [part for part in facedata.model_parts(f.read()) if len(part.tris)]
    skin = [part for part in parts if "b0001" in (part.material or "")]
    clothing = [part for part in parts if "b0001" not in (part.material or "")]
    body_co, body_tris = joined(skin)
    cloth_co, cloth_tris = joined(clothing)
    print("body: {} vertices in {} parts, clothing: {} vertices in {} parts".format(
        len(body_co), len(skin), len(cloth_co), len(clothing)))

    stroke = squishing.SquishStroke(
        body_co, body_co, body_tris, np.ones(len(body_co), bool), cloth_co, cloth_co, cloth_tris,
        np.ones(len(cloth_co), bool), press=args.press, bulge=args.bulge, width=args.width, weld_distance=args.seam,
    )
    stroke.dab(body_co.mean(axis=0), 10.0, 1.0, 'CONSTANT')
    n = stroke.body_count
    body_after, cloth_after = stroke.co[:n], stroke.co[n:]

    field = stroke.field
    move = np.einsum("ij,ij->i", (body_after - body_co)[stroke.welded_rep], field.normals)
    near_edge = np.abs(field.s) < 0.005
    print("fabric edge: {} of {} skin vertices covered, {} within 5 mm of an edge".format(
        int(field.covered.sum()), len(field.covered), int(near_edge.sum())))
    print("deepest press: {:.2f} mm, at the edge: {:.2f} mm, highest bulge: {:.2f} mm".format(
        -1000.0 * move.min(), 1000.0 * float(np.median(move[near_edge])) if near_edge.any() else 0.0,
        1000.0 * move.max()))
    print("clothing inside the skin by more than 0.5 mm: {} before, {} after".format(
        inside(body_co, body_tris, cloth_co), inside(body_after, body_tris, cloth_after)))
    print("skin and clothing triangles cutting through each other: {} before, {} after".format(
        crossings(body_co, body_tris, cloth_co, cloth_tris), crossings(body_after, body_tris, cloth_after, cloth_tris)))
    print("triangles turned over: body {}, clothing {}".format(
        turned_over(body_co, body_after, body_tris), turned_over(cloth_co, cloth_after, cloth_tris)))

    # Renders: the lowest stretch of fabric edge at the front, and the whole model, before and after.
    edge = stroke.body.co[near_edge & (stroke.body.co[:, 1] < np.median(stroke.body.co[:, 1]))]
    hem = edge[np.argsort(edge[:, 2])[:max(1, len(edge) // 20)]].mean(axis=0) if len(edge) else body_co.mean(axis=0)
    middle = body_co.mean(axis=0)
    os.makedirs(args.out, exist_ok=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = scene.render.resolution_y = 900
    shading = scene.display.shading
    shading.light = 'STUDIO'
    shading.color_type = 'OBJECT'
    shading.show_cavity = True
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.type = 'ORTHO'
    camera.data.clip_end = 10.0
    scene.collection.objects.link(camera)
    scene.camera = camera
    name = os.path.splitext(os.path.basename(args.model))[0].replace(" ", "_")
    for state, (bco, cco) in (("before", (body_co, cloth_co)), ("after", (body_after, cloth_after))):
        body = make_object("Body", bco, body_tris, (0.85, 0.66, 0.55, 1.0))
        cloth = make_object("Clothing", cco, cloth_tris, (0.25, 0.25, 0.3, 1.0))
        for view, target, direction, scale in (
                ("front", middle, (0.0, -1.0, 0.0), 0.6),
                ("side", middle, (1.0, 0.0, 0.0), 0.6),
                ("hem", hem, (0.35, -1.0, 0.1), 0.12),
                ("hem_side", hem, (1.0, -0.35, 0.0), 0.12),
        ):
            path = os.path.join(args.out, "{}_{}_{}.png".format(name, view, state))
            render(scene, path, target, direction, scale)
            print("rendered", path)
        bpy.data.objects.remove(body)
        bpy.data.objects.remove(cloth)


main()
