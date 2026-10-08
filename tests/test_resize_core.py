"""Headless tests for Resize (the Body Fit panel's Resize section).

Run from the repository root:
    blender -b --factory-startup --python tests/test_resize_core.py
"""

import os
import sys
import traceback

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import resize  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import api, fitting, lineup, operators, properties, resizing, sizing  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def signed_distance(points, co, faces):
    """Distance of points to the surface (``co``, ``faces``), negative behind it."""
    tree = BVHTree.FromPolygons(np.asarray(co).tolist(), np.asarray(faces).tolist())
    result = np.empty(len(points))
    for k, point in enumerate(points.tolist()):
        location, normal, _index, distance = tree.find_nearest(point)
        result[k] = distance if (Vector(point) - location).dot(normal) >= 0.0 else -distance
    return result


def nearest_point_transfer(points, source, target, tris):
    """Each point moved like its closest point on the body: Data Transfer or a shape key transferred by
    closest points does this (Surface Deform, bound to the closest faces, much the same)."""
    tree = BVHTree.FromPolygons(source.tolist(), tris.tolist(), all_triangles=True)
    moved = np.empty_like(points)
    for k, point in enumerate(points.tolist()):
        location, _normal, index, _distance = tree.find_nearest(point)
        a, b, c = (source[i] for i in tris[index])
        v0, v1, v2 = b - a, c - a, np.array(location) - a
        d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        bary = np.array((1.0 - v - w, v, w))
        moved[k] = points[k] + bary @ (target[tris[index]] - source[tris[index]])
    return moved


def change():
    large, small, quads = resize.body_shapes()
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return resizing.ShapeChange(large, small, tris), large, small, quads


def garment_edges(objects):
    """The edges of ``objects``, numbered across them in order (as the operator numbers them)."""
    result, first = [], 0
    for obj in objects:
        result.append(resize.edges(obj) + first)
        first += len(obj.data.vertices)
    return np.concatenate(result)


def reset_settings():
    settings = bpy.context.scene.body_fit_brush
    for prop in ("resize_from_object", "resize_from_key", "resize_to_object", "resize_to_key", "resize_match",
                 "resize_radius", "resize_bridge", "resize_rigid_size", "resize_keep_close", "resize_key_name",
                 "resize_use_keys_state", "resize_use_sizes", "resize_sizes_problem"):
        settings.property_unset(prop)
    settings.resize_sizes.clear()
    return settings


def add_group(obj, name, members):
    """Give mesh ``obj`` a vertex group ``name`` holding the vertices ``members`` (a mask) at weight 1."""
    obj.vertex_groups.new(name=name).add(np.flatnonzero(members).tolist(), 1.0, 'REPLACE')


# -----------------------------------------------------------------------------
# Body shapes from shape keys

def test_rest_shape_mix():
    co = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    obj = resize.mesh_object("Keys", co, [(0, 1, 2)])
    obj.shape_key_add(name="Basis", from_mix=False)
    first = obj.shape_key_add(name="First", from_mix=False)
    first.data.foreach_set("co", (co + (0.0, 0.0, 1.0)).astype(np.float32).ravel())
    second = obj.shape_key_add(name="Second", from_mix=False)
    second.data.foreach_set("co", (co + (0.0, 0.0, 1.0) + (1.0, 0.0, 0.0)).astype(np.float32).ravel())
    second.relative_key = first
    muted = obj.shape_key_add(name="Muted", from_mix=False)
    muted.data.foreach_set("co", (co + (0.0, 5.0, 0.0)).astype(np.float32).ravel())
    muted.mute = True
    muted.value = 1.0
    group = obj.vertex_groups.new(name="Half")
    group.add([0, 1, 2], 0.5, 'REPLACE')
    second.vertex_group = "Half"
    first.value = 0.5
    second.value = 1.0

    mixed = resizing.rest_shape(obj)
    expected = co + (0.0, 0.0, 0.5) + (0.5, 0.0, 0.0)
    check(np.allclose(mixed, expected), "mix: values, relative keys and a key's vertex group count like Blender's")
    obj.data.update()
    bpy.context.view_layer.update()
    shown = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    check(np.allclose([v.co[:] for v in shown.vertices], expected, atol=1e-6), "mix: matches what Blender shows")
    obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
    check(np.allclose(resizing.rest_shape(obj, {"Muted": 1.0, "First": 0.0}), co + (0.5, 5.0, 0.0)),
          "mix: given values override, even for muted keys")
    check([block.name for block in resizing.contributing_keys(obj)] == ["First", "Second"],
          "contributing keys: not the basis, not muted ones")
    bpy.data.objects.remove(obj)


def test_rest_shape_missing_group():
    # Blender applies a key fully when its vertex group doesn't exist or the mesh has no weights at all,
    # but not at all when the mesh has weights and none in the group.
    co = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    obj = resize.mesh_object("Unweighted", co, [(0, 1, 2)])
    obj.shape_key_add(name="Basis", from_mix=False)
    key = obj.shape_key_add(name="Key", from_mix=False)
    key.data.foreach_set("co", (co + (0.0, 0.0, 1.0)).astype(np.float32).ravel())
    key.value = 1.0

    def shown():
        obj.data.update()
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        result = np.array([v.co[:] for v in evaluated.to_mesh().vertices])
        evaluated.to_mesh_clear()
        return result

    key.vertex_group = "Renamed"
    check(np.allclose(resizing.rest_shape(obj), co + (0.0, 0.0, 1.0)) and np.allclose(shown(), co + (0.0, 0.0, 1.0)),
          "missing group: the key counts fully, like Blender's")
    obj.vertex_groups.new(name="Renamed")
    check(np.allclose(resizing.rest_shape(obj), co + (0.0, 0.0, 1.0)) and np.allclose(shown(), co + (0.0, 0.0, 1.0)),
          "a mesh without weights: the key counts fully, like Blender's")
    obj.vertex_groups.new(name="Other").add([0], 1.0, 'REPLACE')
    check(np.allclose(resizing.rest_shape(obj), co) and np.allclose(shown(), co),
          "an empty group of a weighted mesh: the key counts for nothing, like Blender's")
    bpy.data.objects.remove(obj)


def test_body_shapes():
    body, _garments = resize.build()
    large, small, quads = resize.body_shapes()
    shapes = resizing.body_shapes(body, "", None, "Small")
    check(np.allclose(shapes.source, large, atol=1e-6) and np.allclose(shapes.target, small, atol=1e-6),
          "shapes: From the basis, To the key")
    check(len(shapes.tris) == 2 * len(quads) and shapes.how == 'VERTICES', "shapes: the body's triangles")
    body.data.shape_keys.key_blocks["Small"].value = 1.0
    shapes = resizing.body_shapes(body, "", None, "Small")
    check(np.allclose(shapes.source, large, atol=1e-6), "shapes: a key picked for To is off in From, whatever it shows")
    shapes = resizing.body_shapes(body, "Small", None, "Basis")
    check(np.allclose(shapes.source, small, atol=1e-6) and np.allclose(shapes.target, large, atol=1e-6),
          "shapes: the other way round, the basis naming no key")
    body.data.shape_keys.key_blocks["Small"].value = 0.0

    body.location = (0.0, 0.0, 1.0)
    bpy.context.view_layer.update()
    shapes = resizing.body_shapes(body, "", None, "Small")
    check(np.allclose(shapes.source, large + (0.0, 0.0, 1.0), atol=1e-6), "shapes: in world space")
    copy = resize.mesh_object("Small Body", small, quads)
    copy.location = (1.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    shapes = resizing.body_shapes(body, "", copy, "")
    check(np.allclose(shapes.target, small + (0.0, 0.0, 1.0), atol=1e-6),
          "shapes: another object's shape, placed like the body wherever it stands")

    tri = resize.mesh_object("Tri", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    messages = {
        "same": resizing.check_shapes(body, "", None, "Basis"),
        "missing": resizing.check_shapes(body, "Nope", None, "Small"),
        "keyless": resizing.check_shapes(copy, "", None, ""),
        "no uv": resizing.check_shapes(body, "", tri, "", 'UV'),
        "other vertices": resizing.check_shapes(body, "", tri, "", 'VERTICES'),
        "none": resizing.check_shapes(None, "", None, ""),
    }
    check(messages["same"] == "From and To are the same shape", "check: the same shape twice")
    check(messages["missing"] == "'Body' has no shape key 'Nope'", "check: a missing key")
    check("no shape keys" in messages["keyless"], "check: a body without keys needs a To body")
    check(messages["no uv"] == "'Tri' has no UV map to match it to 'Body' by",
          "check: matched by UV map, a different To body needs one")
    check(messages["other vertices"] == "'Body' and 'Tri' have different vertices: match them by UV map or shape",
          "check: matched by vertices, a different To body needs the same ones")
    check(resizing.check_shapes(body, "", tri, "") is None and resizing.check_shapes(body, "", tri, "", 'SHAPE') is None,
          "check: Auto and Shape take any To body")
    check(messages["none"] == "Pick the From Body", "check: no body")
    check(resizing.check_shapes(body, "", copy, "") is None and resizing.matching(body, copy) == 'VERTICES',
          "check: two objects with the same vertices work, vertex by vertex")
    check(resizing.matching(body, None, 'SHAPE') == 'VERTICES' and resizing.matching(body, body, 'UV') == 'VERTICES',
          "check: one body's own shapes are always matched vertex by vertex")


def test_matched_bodies():
    body, garments = resize.build()
    coarse = resize.coarse_body()
    coarse_large = resize.positions(coarse)
    shapes = resizing.body_shapes(coarse, "", body, "Small")
    check(resizing.check_shapes(coarse, "", body, "Small") is None and resizing.matching(coarse, body) == 'AUTO'
          and shapes.how == 'UV', "matched: a different body sharing the UV layout is matched by the UV maps")
    source, target, tris = shapes.source, shapes.target, shapes.tris
    expected = resize.shrink(coarse_large)
    error = np.linalg.norm(target - expected, axis=1)
    check(len(source) == len(coarse_large) and np.allclose(source, coarse_large) and len(tris) == 2 * 46 * 36,
          "matched: From on the From body's own vertices, all found")
    # Up to a couple of mm off in the sharp crease at the breasts' base, where the To body's flat
    # triangles cut the curve.
    check(np.median(error) < 0.0001 and error.max() < 0.003,
          "matched: To lands where the To body's key puts the same spot (median {:.3f}, at most {:.2f} mm off)".format(
              np.median(error) * 1000, error.max() * 1000))

    # Resizing with the matched bodies comes out like with the To body's own key.
    direct = resizing.body_shapes(body, "", None, "Small").change()
    matched = shapes.change()
    suit = resize.positions(garments["Suit"])
    diff = np.linalg.norm(matched.apply(suit) - direct.apply(suit), axis=1)
    check(np.median(diff) < 0.0002 and diff.max() < 0.001,
          "matched: the suit resizes the same as with one body's key (median {:.2f}, at most {:.2f} mm apart)".format(
              np.median(diff) * 1000, diff.max() * 1000))

    # UV islands sharing texture space (hands and body can): the one landing close wins.
    large, small, quads = resize.body_shapes()
    far = (1.0, 0.0, 0.0)
    twin = resize.body_object("Twin", np.concatenate([large + far, large]), np.concatenate([small + far, small]),
                              np.concatenate([quads, quads + len(large)]), np.tile(resize.body_uvs(large), (2, 1)))
    twin_target = resizing.body_shapes(coarse, "", twin, "Small", 'UV').target
    check(np.allclose(twin_target, target), "matched: overlapping UV islands, the close one is picked")

    # A part the To body lacks (hands, say) with UVs overlapping its chest's, 30 cm away from it: landing
    # on the chest, it's left out.
    c_large, c_small, c_quads = resize.body_shapes(resize.COARSE_XS, resize.COARSE_ZS)
    stray, stray_quads = resize.grid(np.linspace(-0.05, 0.05, 11), np.linspace(-0.13, -0.10, 4), resize.body_front)
    stray = stray + (0.0, -0.3, 0.0)
    extra = resize.body_object("Coarse And Stray", np.concatenate([c_large, stray]), np.concatenate([c_small, stray]),
                               np.concatenate([c_quads, stray_quads + len(c_large)]),
                               np.concatenate([resize.body_uvs(c_large), resize.body_uvs(stray)]))
    extra_shapes = resizing.body_shapes(extra, "", body, "Small")
    check(len(extra_shapes.source) == len(c_large) and np.allclose(extra_shapes.source, source)
          and np.allclose(extra_shapes.target, target) and len(extra_shapes.tris) == len(tris),
          "matched: a part the To body lacks is left out")

    quarter = resize.body_shapes(resize.XS[:35], resize.ZS[:27])
    mirrored = resize.body_object("Mirrored", large, small, quads, resize.body_uvs(large) * (-1.0, 1.0) + (1.0, 0.0))
    for label, other in (
            ("a mirrored UV map", mirrored),
            ("a quarter of the body", resize.body_object("Quarter", *quarter, resize.body_uvs(quarter[0]))),
    ):
        try:
            resizing.body_shapes(coarse, "", other, "Small", 'UV')
            check(False, "matched: refused with " + label)
        except resizing.ResizeError as error:
            check("UV" in str(error), "matched: refused with {:s} ({:s})".format(label, str(error)))
    check(resizing.body_shapes(coarse, "", mirrored, "Small").how == 'SHAPE',
          "matched: Auto matches bodies whose UV maps don't match by shape")
    # A UV map a whole square away is the same layout (Bibo+ keeps its legs a square below Neolithe's), and so is
    # one spanning two squares, where the triangles crossing between them reach over the side.
    for label, from_shift, to_shift in (("a square below", None, (0.0, -1.0)),
                                        ("across two squares", (0.5, 0.0), (-0.5, 0.0))):
        moved = resize.body_object("Moved", large, small, quads, resize.body_uvs(large) + to_shift)
        start = coarse if from_shift is None else resize.coarse_body(
            "Shifted", uv=lambda co, shift=from_shift: resize.body_uvs(co) + shift)
        moved_shapes = resizing.body_shapes(start, "", moved, "Small", 'UV')
        check(len(moved_shapes.source) == len(source) and np.abs(moved_shapes.target - target).max() < 1e-6,
              "matched: a UV map {:s} lands the same ({:.2e} m off)".format(
                  label, np.abs(moved_shapes.target - target).max() if len(moved_shapes.target) == len(target) else -1))
        bpy.data.objects.remove(moved)
        if start != coarse:
            bpy.data.objects.remove(start)

    # A triangle without area in the UV map (Neolithe's neck opening is a ring of them, two corners sharing a UV)
    # is a line in it: a spot on it lies on the side closest to the vertex, not in the triangle's middle.
    line = np.array([[0.5, 0.5], [0.5, 0.5], [0.6, 0.5]])
    corners = np.array([[0.0, 0.0, 0.0], [0.0, 0.03, 0.0], [0.1, 0.0, 0.0]])
    triangle, weights = resizing.match_by_uv(np.array([[0.55, 0.5]]), np.array([[0.05, 0.014, 0.0]]),
                                             np.array([[0, 1, 2]]), line[None], corners)
    check(triangle[0] == 0 and np.allclose(weights[0], (0.0, 0.5, 0.5)),
          "matched: a spot on a triangle without area in the UV map lies on its closest side ({!s})".format(
              np.round(weights[0], 3).tolist()))

    # An island of the UV map the To body lacks altogether (a gen3 body's extra detail in texture space no other
    # layout uses) doesn't count against the share matched, up to a third of the body.
    c_uv = resize.body_uvs(c_large) * 0.7
    for label, rows, accepted in (("a fifth", np.linspace(-0.13, -0.05, 21), True),
                                  ("half", np.linspace(-0.13, 0.12, 60), False)):
        island, island_quads = resize.grid(np.linspace(-0.05, 0.05, 21), rows, resize.body_front)
        island = island + (0.0, -0.3, 0.0)
        island_uv = 0.8 + 0.15 * resize.body_uvs(island)
        own = resize.body_object("Own Island", np.concatenate([c_large, island]), np.concatenate([c_small, island]),
                                 np.concatenate([c_quads, island_quads + len(c_large)]),
                                 np.concatenate([c_uv, island_uv]))
        lacking = resize.body_object("Lacking", large, small, quads, resize.body_uvs(large) * 0.7)
        share = len(island) / (len(island) + len(c_large))
        try:
            island_shapes = resizing.body_shapes(own, "", lacking, "Small", 'UV')
            check(accepted and len(island_shapes.source) == len(c_large) and np.allclose(island_shapes.source, source),
                  "matched: an island the To body lacks, {:s} of the body ({:.0f}%), is left out".format(label, 100 * share))
        except resizing.ResizeError as error:
            check(not accepted, "matched: an island the To body lacks, {:s} of the body ({:.0f}%), is refused ({:s})".format(
                label, 100 * share, str(error)))
        bpy.data.objects.remove(own)
        bpy.data.objects.remove(lacking)

    # The same vertices joined the same way are a copy, matched vertex by vertex whatever their UVs.
    copy = resize.coarse_body("Copy", uv=lambda co: np.zeros((len(co), 2)))
    check(resizing.matching(coarse, copy) == 'VERTICES' and resizing.body_shapes(coarse, "", copy, "Small").how == 'VERTICES',
          "matched: a copy of the From body isn't matched by UV")


def surface_distances(points, co, tris):
    """Unsigned distance of points to the surface (``co``, triangles ``tris``)."""
    find = BVHTree.FromPolygons(np.asarray(co).tolist(), np.asarray(tris).tolist(), all_triangles=True).find_nearest
    return np.array([find(point)[3] for point in np.asarray(points).tolist()])


def test_shape_matched_bodies():
    body, garments = resize.build()
    large, small, quads = resize.body_shapes()
    # Another body without a UV map (another game's), lying on the body.
    plain = resize.coarse_body("Plain Body", uv=None)
    plain_large = resize.positions(plain)
    check(resizing.check_shapes(plain, "", body, "Small") is None, "shape: Auto takes a body without a UV map")
    shapes = resizing.body_shapes(plain, "", body, "Small")
    match = shapes.match
    check(shapes.how == 'SHAPE' and len(shapes.source) == len(plain_large) and np.allclose(shapes.source, plain_large)
          and len(shapes.tris) == 2 * 46 * 36, "shape: Auto matches it by shape, on all of the From body's vertices")
    check(match.known.all() and match.matched.all() and match.matched_share > 0.999 and match.folded == 0.0,
          "shape: all of it found on the To body ({:.1f}% matched, {:.1f}% folded)".format(
              100 * match.matched_share, 100 * match.folded))
    on = surface_distances(shapes.target, small, resize.triangles(body))
    check(on.max() < 1e-5, "shape: the matched shape lies on the To shape ({:.4f} mm off at most)".format(1000 * on.max()))
    # Closest spots aren't where the key takes each spot: the matched shape slides along the body, a
    # little where the breasts shrink a lot.
    error = np.linalg.norm(shapes.target - resize.shrink(plain_large), axis=1)
    fronts = [np.argmin(np.linalg.norm(plain_large[:, [0, 2]] - (cx, cz), axis=1)) for cx, _cy, cz in resize.CENTERS]
    check(np.median(error) < 0.0005 and error[fronts].max() < 0.001,
          "shape: mostly where the To body's key puts the same spot, the breasts' fronts too (median {:.2f}, "
          "fronts {:.2f} mm off; {:.0f} mm at most)".format(
              1000 * np.median(error), 1000 * error[fronts].max(), 1000 * error.max()))
    check(abs(match.apart) < 1e-4 and 0.03 < match.spread < 0.04,
          "shape: the bodies differ only at the breasts (by {:.0f} mm, typically {:.2f})".format(
              1000 * match.spread, 1000 * match.apart))

    # The suit resized with it keeps its distance to the body, as with the key.
    suit = garments["Suit"]
    before = resize.positions(suit)
    after = shapes.change(bridge=resizing.BRIDGE_RADIUS).apply(before, edges=resize.edges(suit))
    h0 = signed_distance(before, large, quads)
    h1 = signed_distance(after, small, quads)
    dh = np.abs(h1 - h0)
    check(np.median(dh) < 0.001 and np.percentile(dh, 99) < 0.003,
          "shape: the suit stays about as far from the body (median {:.2f}, 99% within {:.2f} mm)".format(
              np.median(dh) * 1000, np.percentile(dh, 99) * 1000))
    check(resize.flipped(before, after, resize.triangles(suit)) == 0 and float(h1.min()) > 0.0,
          "shape: no face of the suit turned over, nothing sank in ({:.2f} mm closest)".format(h1.min() * 1000))


def test_shape_refused():
    body, _garments = resize.build()
    large, small, quads = resize.body_shapes()
    plain = resize.coarse_body("Plain Body", uv=None)
    # Matched by shape, the bodies must lie on each other; by UV map they can stand anywhere.
    body.location = (0.0, -0.3, 0.0)
    bpy.context.view_layer.update()
    try:
        resizing.body_shapes(plain, "", body, "Small")
        check(False, "shape: refused bodies standing apart")
    except resizing.ResizeError as error:
        check(str(error) == "'Plain Body' and 'Body' don't lie on each other (typically 30 cm apart): line them up first",
              "shape: refused bodies standing apart ({:s})".format(str(error)))
    check(resizing.body_shapes(resize.coarse_body(), "", body, "Small").how == 'UV',
          "UV map: the To body can stand anywhere")
    body.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    quarter = resize.body_shapes(resize.XS[:35], resize.ZS[:27])
    try:
        resizing.body_shapes(plain, "", resize.body_object("Quarter", *quarter, None), "Small")
        check(False, "shape: refused a To body that's a quarter of it")
    except resizing.ResizeError as error:
        check(str(error).startswith("Only 26% of 'Plain Body' lies over 'Quarter'"),
              "shape: refused a To body that's a quarter of it ({:s})".format(str(error)))


def test_shape_skipped_parts():
    body, _garments = resize.build()
    plain = resize.coarse_body("Plain Body", uv=None)
    co = resize.positions(plain)
    # A hand resting on the side of the chest, where the body doesn't change. Thighs make the groups tell
    # the limbs.
    def groups(obj, points):
        hand = points[:, 0] < -0.152
        thigh = (points[:, 2] < -0.13) & (np.abs(points[:, 0]) > 0.02) & (np.abs(points[:, 0]) < 0.1)
        add_group(obj, "Spine", ~hand)
        add_group(obj, "LeftHand", hand)
        add_group(obj, "LeftThigh", thigh & (points[:, 0] < 0.0))
        add_group(obj, "RightThigh", thigh & (points[:, 0] > 0.0))
        add_group(obj, "Yiggle", np.ones(len(points), dtype=bool))  # a devkit's mask, telling no part
        return hand
    hand = groups(plain, co)
    labels = resizing.part_labels(plain)
    check(labels is not None and np.all(labels[hand] == lineup.PART_INDEX["hand_l"])
          and np.all(labels[~hand] != lineup.PART_INDEX["hand_l"]), "skipped: parts from the groups named after them")
    match = resizing.body_shapes(plain, "", body, "Small").match
    check(np.array_equal(match.skipped, hand) and not match.known[hand].any() and not match.matched[hand].any(),
          "skipped: a hand isn't matched by shape")
    # Where the body doesn't change, it stays on it, sliding along with the parts around it.
    off = surface_distances(match.target[hand], resize.body_shapes()[1], resize.triangles(body))
    slide = np.linalg.norm(match.target[hand] - co[hand], axis=1)
    check(off.max() < 1e-5 and slide.max() < 0.001, "skipped: it follows the parts around it, on the body "
          "({:.4f} mm off it, {:.2f} mm along it)".format(1000 * off.max(), 1000 * slide.max()))
    # Without groups of its own, the To body's tell: what lies on its hand.
    bare = resize.coarse_body("Bare Body", uv=None)
    check(resizing.part_labels(bare) is None, "skipped: a body without groups tells no parts")
    groups(body, resize.positions(body))
    match = resizing.body_shapes(bare, "", body, "Small").match
    check(np.array_equal(match.skipped, hand), "skipped: without groups, what lies on the To body's hand")


# -----------------------------------------------------------------------------
# The engine

def test_tight_clothing_keeps_its_distance():
    shape_change, large, small, quads = change()
    resize.build()
    suit = bpy.data.objects["Suit"]
    before = resize.positions(suit)
    after = shape_change.apply(before)
    h0 = signed_distance(before, large, quads)
    h1 = signed_distance(after, small, quads)
    dh = np.abs(h1 - h0)
    moved = np.linalg.norm(after - before, axis=1)
    check(moved.max() > 0.03, "suit: the breasts' fronts moved back ({:.0f} mm)".format(moved.max() * 1000))
    check(np.median(dh) < 0.001 and np.percentile(dh, 99) < 0.003,
          "suit: stays about as far from the body (median {:.2f}, 99% within {:.2f} mm)".format(
              np.median(dh) * 1000, np.percentile(dh, 99) * 1000))
    check(resize.flipped(before, after, resize.triangles(suit)) == 0, "suit: no face turned over")
    check(float(h1.min()) > -0.001, "suit: nothing sinks into the body ({:.2f} mm closest)".format(h1.min() * 1000))


def test_bridge_stays_whole():
    shape_change, large, small, quads = change()
    resize.build()
    bridge = bpy.data.objects["Bridge"]
    before = resize.positions(bridge)
    after = shape_change.apply(before)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    nearest = nearest_point_transfer(before, large, small, tris)
    edge_list = resize.edges(bridge)
    ours, theirs = resize.stretch(before, after, edge_list), resize.stretch(before, nearest, edge_list)
    check(theirs > 1.5 * ours, "bridge: stretched {:.1f}x at most, following each closest point {:.1f}x".format(ours, theirs))
    check(resize.flipped(before, after, resize.triangles(bridge)) == 0, "bridge: no face turned over")
    check(float(signed_distance(after, small, quads).min()) > 0.0, "bridge: stays in front of the body")


def test_layers_and_seams():
    shape_change, _large, _small, _quads = change()
    resize.build()
    parts = [bpy.data.objects["Strap A"], bpy.data.objects["Strap B"]]
    before = np.concatenate([resize.positions(obj) for obj in parts])
    after = shape_change.apply(before)
    count_a = len(parts[0].data.vertices)
    half_a, half_b = count_a // 2, len(parts[1].data.vertices) // 2
    inner = np.concatenate([after[:half_a], after[count_a:count_a + half_b]])
    outer = np.concatenate([after[half_a:count_a], after[count_a + half_b:]])
    thickness = np.linalg.norm(outer - inner, axis=1)
    check(abs(np.median(thickness) - 0.004) < 0.0002 and np.percentile(thickness, 90) < 0.0047
          and 0.0027 < thickness.min() and thickness.max() < 0.0065,
          "strap: its layers stay 4 mm apart (median {:.2f}; {:.2f} to {:.2f} mm at its corners over the breast's base)".format(
              np.median(thickness) * 1000, thickness.min() * 1000, thickness.max() * 1000))
    # Strap A's last column of vertices lies on Strap B's first, in both layers.
    rows = len(resize.STRAP_ZS)
    seam_a = np.concatenate([layer * half_a + (len(resize.STRAP_XS[0]) - 1) * rows + np.arange(rows) for layer in (0, 1)])
    seam_b = count_a + np.concatenate([layer * half_b + np.arange(rows) for layer in (0, 1)])
    same = np.array_equal(before[seam_a], before[seam_b])
    gap = np.abs(after[seam_a] - after[seam_b]).max()
    check(same and gap == 0.0, "strap: the seam between its two objects stays closed")


def test_rigid_parts():
    shape_change, _large, _small, _quads = change()
    resize.build()
    pendant = bpy.data.objects["Pendant"]
    before = resize.positions(pendant)
    edge_list = resize.edges(pendant)
    free = shape_change.apply(before)
    after = shape_change.apply(before, rigid=np.zeros(len(before), dtype=np.int64))
    check(resize.stretch(before, free, edge_list) > 1.1, "pendant: the space around it changes shape ({:.2f}x)".format(
        resize.stretch(before, free, edge_list)))
    check(resize.stretch(before, after, edge_list) < 1.0 + 1e-9, "pendant: rigid, it keeps its shape")
    check(np.allclose(after.mean(axis=0), free.mean(axis=0)), "pendant: and goes where the space around it goes")

    # A piece lying on the body (a ring, a chain link) grows and shrinks with it, keeping its shape.
    _large, small, quads = resize.body_shapes()
    plate, plate_quads = resize.offset_sheet(np.linspace(-0.085, -0.055, 7), np.linspace(-0.015, 0.015, 7), 0.002)
    plate_edges = np.concatenate([plate_quads[:, [0, 1]], plate_quads[:, [1, 2]], plate_quads[:, [2, 3]],
                                  plate_quads[:, [3, 0]]])
    stiff = shape_change.apply(plate, rigid=np.zeros(len(plate), dtype=np.int64))
    l0 = np.linalg.norm(plate[plate_edges[:, 0]] - plate[plate_edges[:, 1]], axis=1)
    l1 = np.linalg.norm(stiff[plate_edges[:, 0]] - stiff[plate_edges[:, 1]], axis=1)
    ratio = l1 / l0
    unscaled = shape_change.apply(plate)
    resizing.keep_rigid(plate, unscaled, np.zeros(len(plate), dtype=np.int64))
    off =np.abs(signed_distance(stiff, small, quads))
    off_unscaled = np.abs(signed_distance(unscaled, small, quads))
    check(ratio.max() - ratio.min() < 1e-6 and ratio.mean() < 0.9,
          "plate on the breast: rigid, it shrinks with it as one piece ({:.2f}x)".format(ratio.mean()))
    check(np.median(off) < 0.75 * np.median(off_unscaled),
          "plate on the breast: and stays closer to it than without shrinking ({:.1f} mm, {:.1f} mm)".format(
              np.median(off) * 1000, np.median(off_unscaled) * 1000))

    co = np.concatenate([before, before + (0.1, 0.0, 0.0), before + (0.0, 0.0, 0.2)])
    edge_list = np.concatenate([edge_list, edge_list + 8, edge_list + 16])
    parts, count = resizing.loose_parts(co, edge_list)
    check(count == 3 and len(set(parts[:8])) == 1, "loose parts: one per cube")
    co[16:] = co[:8]
    parts, count = resizing.loose_parts(co, edge_list)
    check(count == 2 and parts[0] == parts[16], "loose parts: parts at the same spot are one (split seams)")
    sizes = resizing.part_sizes(co, parts, count)
    check(np.allclose(sizes, 0.02 * np.sqrt(3.0)), "loose parts: sized by their bounding box")


def test_skin_follows_body():
    shape_change, large, small, quads = change()
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    resize.build()
    suit = bpy.data.objects["Suit"]
    before = resize.positions(suit)
    # Skin: every third vertex of the body (a mod's copy of the body's skin) and the centres of its triangles.
    skin = np.concatenate([large[::3], large[tris].mean(axis=1)])
    expected = np.concatenate([small[::3], small[tris].mean(axis=1)])
    check(resizing.is_skin(skin, large, tris) and not resizing.is_skin(before, large, tris),
          "skin: the body's own vertices are skin, a suit 3 mm off it isn't")
    points = np.concatenate([before, skin])
    bound = np.concatenate([np.zeros(len(before), dtype=bool), np.ones(len(skin), dtype=bool)])
    edge_list = resize.edges(suit)
    after = shape_change.apply(points, edges=edge_list, bound=bound)
    gap = np.abs(after[len(before):] - expected).max()
    check(gap < 1e-6, "skin: lands on the small shape ({:.2e} m off)".format(gap))
    alone = shape_change.apply(before, edges=edge_list)
    off = np.abs(signed_distance(after[:len(before)], small, quads) - 0.003)
    off_alone = np.abs(signed_distance(alone, small, quads) - 0.003)
    check(np.percentile(off, 95) <= np.percentile(off_alone, 95) + 1e-4,
          "skin: the suit over it keeps its 3 mm from the body as well as without it ({:.2f}, {:.2f} mm, 95th percentile)".format(
              np.percentile(off, 95) * 1000, np.percentile(off_alone, 95) * 1000))
    loose = shape_change.apply(skin, edges=np.zeros((0, 2), dtype=np.int64))
    check(np.abs(loose - expected).max() > 0.001, "skin: moved like clothing, it would miss the body ({:.1f} mm)".format(
        np.abs(loose - expected).max() * 1000))

    # Through the API, a skin mesh is told apart from the clothing.
    body, garments = resize.devkit_body()
    skin_obj = resize.mesh_object("Skin", resizing.rest_shape(body)[::2], [])
    settings = reset_settings()
    settings.resize_from_object = body
    settings.resize_use_sizes = True
    bpy.ops.magic_fit.resize_pick(setting="chest_size", option=1)
    sizes = api.picked_sizes()
    runs, _notes = api.resize_meshes([garments["Suit"], skin_obj], sizes, api.Job(settings), ["Medium"])
    check(runs[0].skin == [skin_obj], "api: the skin mesh listed ({!s})".format(runs[0].skin))
    medium_body = resizing.rest_shape(body, sizing.size_values(body, sizes[0].values))
    check(np.abs(resize.positions(skin_obj, "Medium") - medium_body[::2]).max() < 1e-6,
          "api: its key is the body's shape at the size")
    bpy.data.objects.remove(skin_obj)


def test_sculpted_skin():
    shape_change, large, small, quads = change()
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    resize.build()
    check(all(resizing.is_skin_material(name) for name in (
        "mt_c0201b0001_bibo.mtrl", "/mt_c0201b0001_bibo.mtrl", "mt_c1401b0001_b.mtrl [faf614cb]", "mt_c0201b0001_a",
        "mt_c0201b0101_eve.mtrl.001")) and not any(resizing.is_skin_material(name) for name in (
        "mt_c0201b0001_piercings.mtrl", "mt_c0201b0001_bibopube.mtrl", "mt_c0201e0239_top_a.mtrl",
        "mt_c0201b0001_a_nails.mtrl", "Material", "")),
          "skin material: the game's skin, not piercings, pubes, nails or clothing")

    # Skin a top's author lifted 2 mm off the breast (too far off to count as a copy of the body's), drawn with the
    # skin material, and a lining 0.5 mm over it.
    xs, zs = np.linspace(-0.11, -0.03, 21), np.linspace(-0.04, 0.04, 21)
    skin, skin_quads = resize.offset_sheet(xs, zs, 0.002)
    check(not resizing.is_skin(skin, large, tris), "sculpted skin: 2 mm off the body, it isn't a copy of it")
    obj = resize.mesh_object("Lifted Skin", skin, skin_quads)
    obj.data.materials.append(bpy.data.materials.new("Cloth"))
    obj.data.materials.append(bpy.data.materials.new("mt_c0201b0001_bibo.mtrl"))
    half = np.zeros(len(obj.data.polygons), dtype=np.int64)
    half[: len(half) // 2] = 1
    obj.data.polygons.foreach_set("material_index", half)
    marked = resizing.skin_vertices(obj)
    used = np.unique(skin_quads[: len(half) // 2])
    check(marked[used].all() and marked.sum() == len(used), "skin vertices: those of the faces drawn with skin")
    for polygon in obj.data.polygons:
        polygon.material_index = 1
    check(resizing.skin_vertices(obj).all(), "skin vertices: the whole mesh drawn with skin")
    bpy.data.objects.remove(obj)

    far = resize.offset_sheet(xs, zs, 0.0045)[0][:3]
    farther = resize.offset_sheet(xs, zs, 0.007)[0][:3]
    shares = resizing.skin_shares(np.concatenate([skin[:3], far, farther]), large, tris, np.ones(9, dtype=bool))
    check(np.allclose(shares[:3], 1.0) and np.all((shares[3:6] > 0.3) & (shares[3:6] < 0.7)) and np.allclose(shares[6:], 0.0),
          "skin shares: skin follows the body fully when close, fades out from 3 to 6 mm off ({:s})".format(
              str(np.round(shares, 2).tolist())))

    # Growing breasts, which the smooth move lags behind (see Keep Close): the skin and the lining laid over the small
    # ones instead.
    grow = resizing.ShapeChange(small, large, tris)
    surface = resizing._surface(small, tris)
    normals = surface.normals[surface._weld]
    columns = np.flatnonzero((resize.XS >= -0.11) & (resize.XS <= -0.03))
    rows = np.flatnonzero((resize.ZS >= -0.04) & (resize.ZS <= 0.04))
    region = np.add.outer(columns * len(resize.ZS), rows).ravel()
    sheet_quads = resize.grid(np.arange(len(columns)), np.arange(len(rows)), lambda x, z: 0.0 * x)[1]
    skin = small[region] + 0.002 * normals[region]
    lining = small[region] + 0.0025 * normals[region]
    points = np.concatenate([skin, lining])
    bound = np.concatenate([np.ones(len(skin)), np.zeros(len(lining))])
    edge_list = np.concatenate([np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]])
                                for q in (sheet_quads, sheet_quads + len(skin))])
    faces = np.concatenate([sheet_quads[:, [0, 1, 2]], sheet_quads[:, [0, 2, 3]]])
    after = grow.apply(points, edges=edge_list, bound=bound, faces=faces)
    # The lining moved on its own, the skin following the body: what Resize did before it knew about the skin.
    alone = grow.apply(lining, edges=np.concatenate([sheet_quads[:, [0, 1]], sheet_quads[:, [1, 2]],
                                                     sheet_quads[:, [2, 3]], sheet_quads[:, [3, 0]]]))
    without = np.concatenate([grow.bind(skin), alone])
    skin_off = signed_distance(after[:len(skin)], large, quads)
    # About: its spot on the breast keeps its offset along the normal there, which curves less on the large breast.
    check(np.abs(skin_off - 0.002).max() < 0.00075,
          "sculpted skin: keeps its 2 mm off the grown breast ({:.2f} to {:.2f} mm)".format(
              skin_off.min() * 1000, skin_off.max() * 1000))
    # Over the skin, not past its edges.
    inner = np.add.outer(np.arange(2, len(columns) - 2) * len(rows), np.arange(2, len(rows) - 2)).ravel()
    gap = signed_distance(after[len(skin):][inner], after[:len(skin)], sheet_quads)
    gap_without = signed_distance(without[len(skin):][inner], without[:len(skin)], sheet_quads)
    check(gap.min() > 0.0005 - 0.00015,
          "lining: stays over its own skin ({:.2f} mm closest, authored 0.5 mm)".format(gap.min() * 1000))
    check(gap_without.min() < gap.min() - 0.0002,
          "lining: moved without the skin, it would sink into it ({:.2f} mm)".format(
              gap_without.min() * 1000))


def test_round_trip():
    shape_change, large, small, quads = change()
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    back = resizing.ShapeChange(small, large, tris)
    resize.build()
    before = resize.positions(bpy.data.objects["Suit"])
    again = back.apply(shape_change.apply(before))
    error = np.linalg.norm(again - before, axis=1)
    check(np.median(error) < 0.001 and error.max() < 0.004,
          "round trip: back to the large breasts within {:.2f} mm (median {:.2f})".format(error.max() * 1000, np.median(error) * 1000))


def test_bridged_shape():
    large, _small, quads = resize.body_shapes()
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    lifted = resizing.bridged(large, tris, resizing.BRIDGE_RADIUS)
    lift = lifted - large
    x, z = large[:, 0], large[:, 2]
    cleft = np.argmin(np.hypot(x, z))
    fronts = [np.argmin(np.hypot(x - cx, z - cz)) for cx, _cy, cz in resize.CENTERS]
    check(np.linalg.norm(lift[cleft]) > 0.01,
          "bridged: the cleavage is filled ({:.1f} mm)".format(1000 * np.linalg.norm(lift[cleft])))
    check(max(np.linalg.norm(lift[i]) for i in fronts) < 1e-4, "bridged: the breasts' fronts stay where they are")
    outward = np.einsum("ij,ij->i", lift, resize.front_normals(x, z))
    check(outward.min() > -1e-6, "bridged: the skin is only ever lifted outward")
    loose = np.concatenate([large, [(0.0, -0.3, 0.0)]])
    check(np.array_equal(resizing.bridged(loose, tris, resizing.BRIDGE_RADIUS)[-1], loose[-1]),
          "bridged: a vertex of no face stays where it is")


def small_bridge():
    """A sheet stretched across the chest with small breasts, 8 mm in front of it (spanning the cleavage),
    the chest's triangles and shapes, and the sheet's edges; the same sheet over the large breasts."""
    large, small, quads = resize.body_shapes()
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    find = BVHTree.FromPolygons(small.tolist(), tris.tolist(), all_triangles=True).ray_cast

    def small_front(xs, zs):
        return np.array([find((x, -1.0, z), (0.0, 1.0, 0.0))[0][1] for x, z in zip(xs, zs)])

    def sheet(front):
        def fn(x, z):
            y = np.empty_like(x)
            for j in range(x.shape[1]):
                y[:, j] = resize.envelope(front(x[:, j], z[:, j])) - 0.008
            return y
        return fn
    xs = np.arange(-0.15, 0.15 + 1e-9, 0.004)
    zs = np.arange(-0.10, 0.09 + 1e-9, 0.004)
    co, quads_sheet = resize.grid(xs, zs, sheet(small_front))
    over_large, _ = resize.grid(xs, zs, sheet(resize.body_front))
    sheet_edges = np.concatenate([quads_sheet[:, [0, 1]], quads_sheet[:, [1, 2]], quads_sheet[:, [2, 3]],
                                  quads_sheet[:, [3, 0]]])
    return co, over_large, sheet_edges, np.concatenate([quads_sheet[:, [0, 1, 2]], quads_sheet[:, [0, 2, 3]]]), large, small, tris


def test_bridge_creases():
    co, over_large, sheet_edges, sheet_tris, large, small, tris = small_bridge()
    middle = (np.abs(co[:, 0]) < 0.012) & (np.abs(co[:, 2]) < 0.03)
    grown = {}
    for bridge in (0.0, resizing.BRIDGE_RADIUS):
        grown[bridge] = resizing.ShapeChange(small, large, tris, bridge=bridge).apply(co, edges=sheet_edges)
    sunk = {bridge: float(np.median(out[middle, 1] - over_large[middle, 1])) for bridge, out in grown.items()}
    check(sunk[0.0] - sunk[resizing.BRIDGE_RADIUS] > 0.01,
          "bridge creases: a sheet across a growing cleavage sinks into it {:.0f} mm less ({:.0f} mm, {:.0f} mm "
          "following the skin)".format(1000 * (sunk[0.0] - sunk[resizing.BRIDGE_RADIUS]),
                                       1000 * sunk[resizing.BRIDGE_RADIUS], 1000 * sunk[0.0]))
    check(resize.flipped(co, grown[resizing.BRIDGE_RADIUS], sheet_tris) == 0, "bridge creases: no face turned over")
    plain = resizing.ShapeChange(small, large, tris).apply(co, edges=sheet_edges)
    check(np.array_equal(plain, grown[0.0]), "bridge creases: 0 follows the skin, as without it")

    # Skin-tight clothing keeps following the skin, there and back.
    resize.build()
    suit = bpy.data.objects["Suit"]
    before = resize.positions(suit)
    suit_edges = resize.edges(suit)
    down = resizing.ShapeChange(large, small, tris, bridge=resizing.BRIDGE_RADIUS)
    up = resizing.ShapeChange(small, large, tris, bridge=resizing.BRIDGE_RADIUS)
    error = np.linalg.norm(up.apply(down.apply(before, edges=suit_edges), edges=suit_edges) - before, axis=1)
    check(np.median(error) < 0.001 and error.max() < 0.005,
          "bridge creases: a skin-tight suit comes back within {:.1f} mm (median {:.2f})".format(
              1000 * error.max(), 1000 * np.median(error)))


def test_shared():
    """Changes from one source shape to several targets, sharing what they can (resizing.Shared), move clothing as
    each does alone; with more first-step pairs than kept, too."""
    co, _over_large, sheet_edges, _sheet_tris, large, small, tris = small_bridge()
    resize.build()
    suit = bpy.data.objects["Suit"]
    clothing = ((resize.positions(suit), resize.edges(suit)), (co, sheet_edges))
    rigid = np.full(len(co), -1, dtype=np.int64)
    rigid[:40] = 0
    targets = (large, small + 0.5 * (large - small), large)
    for limit in (resizing.CACHED_PAIRS, 1000):
        saved, resizing.CACHED_PAIRS = resizing.CACHED_PAIRS, limit
        try:
            shared = resizing.Shared()
            worst, entries = 0.0, []
            for target in targets:
                for bridge in (0.0, resizing.BRIDGE_RADIUS):
                    for (points, edges), labels in zip(clothing, (None, rigid)):
                        alone = resizing.ShapeChange(small, target, tris, bridge=bridge).apply(
                            points, rigid=labels, edges=edges)
                        together = resizing.ShapeChange(small, target, tris, bridge=bridge, shared=shared).apply(
                            points, rigid=labels, edges=edges)
                        worst = max(worst, float(np.abs(together - alone).max()))
                entries.append(len(shared._entries))
        finally:
            resizing.CACHED_PAIRS = saved
        check(worst < 1e-6 and entries[0] == entries[-1] == 6,
              "shared: as each change alone ({:.2g} m off, {!s} entries{:s})".format(
                  worst, entries, ", pairs over the limit" if limit < saved else ""))


def test_steps_and_limits():
    shape_change, large, small, _quads = change()
    largest = float(np.linalg.norm(small - large, axis=1).max())
    check(shape_change.steps == int(np.ceil(largest / (resizing.STEP_RADII * resizing.DEFAULT_RADIUS))),
          "steps: a couple of radii of change each ({:d} for {:.0f} mm)".format(shape_change.steps, largest * 1000))
    tris = shape_change.tris
    same = resizing.ShapeChange(large, large, tris)
    points = np.array([(0.0, -0.2, 0.0), (0.1, -0.1, 0.05)])
    check(same.steps == 1 and np.array_equal(same.apply(points), points), "no change: nothing moves")
    far = np.array([(0.0, -2.0, 0.0)])
    moved = shape_change.apply(far)
    check(np.all(np.isfinite(moved)) and np.linalg.norm(moved - far) < 0.05, "far away: moves little, and sanely")
    for args, message in (
            ((large, large[:-1], tris), "different vertex counts"),
            ((large, large, np.zeros((0, 3), dtype=np.int64)), "no faces"),
    ):
        try:
            resizing.ShapeChange(*args)
            check(False, "refused: " + message)
        except resizing.ResizeError:
            check(True, "refused: " + message)


def test_keep_close():
    # A flat body that doesn't change, and a sheet 2 mm over it split in two along x = 0 (both halves have
    # the column of vertices there). The move left a bag 6 mm out on one side and a dent 3 mm in (1 mm
    # into the body) on the other.
    xs = np.linspace(-0.1, 0.1, 41)
    body, quads = resize.grid(xs, xs, lambda X, Z: np.zeros_like(X))
    body_tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    # A flat body has no inside: heights count along the normal the body surface takes as outward.
    normal = fitting.BodySurface(body, body_tris, BVHTree.FromPolygons(body.tolist(), body_tris.tolist())).normals[0]
    halves = [resize.grid(xs[xs <= 0.0], xs, lambda X, Z: np.zeros_like(X)),
              resize.grid(xs[xs >= 0.0], xs, lambda X, Z: np.zeros_like(X))]
    start = np.concatenate([halves[0][0], halves[1][0]]) + 0.002 * normal
    count = len(halves[0][0])
    faces = np.concatenate([halves[0][1], halves[1][1] + count])
    edge_list = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 3]], faces[:, [3, 0]]])
    # Smooth bumps 3 cm across, like the bags a smooth move leaves.
    bump_bag = np.cos(np.minimum(np.linalg.norm(start[:, [0, 2]] - (-0.05, 0.0), axis=1) / 0.03, 1.0) * np.pi / 2) ** 2
    bump_dent = np.cos(np.minimum(np.linalg.norm(start[:, [0, 2]] - (0.05, 0.0), axis=1) / 0.03, 1.0) * np.pi / 2) ** 2
    bag, dent = bump_bag > 0.5, bump_dent > 0.5
    moved = start + (0.006 * bump_bag - 0.003 * bump_dent)[:, None] * normal
    kept = resizing.keep_close(start, moved, edge_list, body, body, body_tris)
    height = kept @ normal
    far = np.linalg.norm(start[:, [0, 2]] - (-0.05, 0.0), axis=1) > 0.06
    far &= np.linalg.norm(start[:, [0, 2]] - (0.05, 0.0), axis=1) > 0.06
    print("      bag {:.2f} mm high at most (was 8), dent {:.2f} mm at least (was -1)".format(
        height[bag].max() * 1000, height[dent].min() * 1000))
    check(height[bag].max() < 0.002 + resizing.PULL_SLACK + 0.0005,
          "keep close: a bag comes back to within the slack ({:.2f} mm high at most)".format(height[bag].max() * 1000))
    check(height[dent].min() > 0.002 - resizing.PUSH_SLACK - 0.0005,
          "keep close: a dent is pushed back out ({:.2f} mm at least)".format(height[dent].min() * 1000))
    check(np.abs(kept - moved)[far].max() < 0.0002, "keep close: cloth far from both is left alone ({:.3f} mm)".format(
        np.abs(kept - moved)[far].max() * 1000))
    seam_a = np.flatnonzero(np.isclose(start[:count, 0], 0.0))
    seam_b = count + np.flatnonzero(np.isclose(start[count:, 0], 0.0))
    check(np.array_equal(kept[seam_a], kept[seam_b]), "keep close: the seam between the halves stays closed")
    fine = resizing.keep_close(start, start, edge_list, body, body, body_tris)
    check(np.abs(fine - start).max() < 1e-12, "keep close: cloth already at its distance doesn't move")
    loose = start + 0.01 * normal
    check(np.array_equal(resizing.keep_close(loose, loose + 0.01 * normal, edge_list, body, body, body_tris),
                         loose + 0.01 * normal), "keep close: loose cloth (1 cm off) isn't held to the body")

    # The To body as a mesh of its own (a finer one of the same plane), as when matched by shape.
    other, other_quads = resize.grid(np.linspace(-0.1, 0.1, 61), np.linspace(-0.1, 0.1, 61), lambda X, Z: np.zeros_like(X))
    other_tris = np.concatenate([other_quads[:, [0, 1, 2]], other_quads[:, [0, 2, 3]]])
    apart = resizing.keep_close(start, moved, edge_list, body, other, body_tris, target_tris=other_tris)
    check(np.abs(apart - kept).max() < 1e-9, "keep close: against another mesh of the To body, the same")
    unknown = resizing.keep_close(start, moved, edge_list, body, other, body_tris, target_tris=other_tris,
                                  known=np.zeros(len(body), dtype=bool))
    check(np.array_equal(unknown, moved), "keep close: nothing is held over parts not known on the To body")
    partly = resizing.keep_close(start, moved, edge_list, body, other, body_tris, target_tris=other_tris,
                                 known=body[:, 0] < 0.0) @ normal
    check(partly[bag].max() < 0.002 + resizing.PULL_SLACK + 0.0005 and partly[dent].min() < -0.0005,
          "keep close: only over the parts known (the bag comes back, the dent stays {:.2f} mm deep)".format(
              -1000 * partly[dent].min()))


def test_keep_close_suit():
    shape_change, large, small, quads = change()
    resize.build()
    suit = bpy.data.objects["Suit"]
    before = resize.positions(suit)
    plain = shape_change.apply(before)
    kept = shape_change.apply(before, edges=resize.edges(suit))
    h0 = signed_distance(before, large, quads)
    off_plain = np.abs(signed_distance(plain, small, quads) - h0)
    off_kept = np.abs(signed_distance(kept, small, quads) - h0)
    check(off_kept.max() <= off_plain.max() + 1e-6 and np.percentile(off_kept, 99) <= np.percentile(off_plain, 99) + 1e-6,
          "keep close: the suit stays at least as close to its distance (at most {:.2f} mm off, {:.2f} mm without)".format(
              off_kept.max() * 1000, off_plain.max() * 1000))
    check(resize.flipped(before, kept, resize.triangles(suit)) == 0, "keep close: no face of the suit turned over")

    parts = [bpy.data.objects["Strap A"], bpy.data.objects["Strap B"]]
    strap = np.concatenate([resize.positions(obj) for obj in parts])
    count_a = len(parts[0].data.vertices)
    strap_edges = np.concatenate([resize.edges(parts[0]), resize.edges(parts[1]) + count_a])
    moved = shape_change.apply(strap, edges=strap_edges)
    rows = len(resize.STRAP_ZS)
    half_a, half_b = count_a // 2, len(parts[1].data.vertices) // 2
    seam_a = np.concatenate([layer * half_a + (len(resize.STRAP_XS[0]) - 1) * rows + np.arange(rows) for layer in (0, 1)])
    seam_b = count_a + np.concatenate([layer * half_b + np.arange(rows) for layer in (0, 1)])
    check(np.array_equal(moved[seam_a], moved[seam_b]), "keep close: the strap's seam between its objects stays closed")


# -----------------------------------------------------------------------------
# The operator

def test_operator():
    body, garments = resize.build()
    settings = reset_settings()
    scene = bpy.context.scene
    scene.magic_fit.target = body  # Body Fit's body isn't Resize's
    check(operators.check_resize_ready(bpy.context) == "Pick the From Body",
          "button: needs a From body")
    scene.magic_fit.target = None
    settings.resize_from_object = body
    check(operators.check_resize_ready(bpy.context) == "Pick a To Body, or turn on Sizes or Use Shapekeys",
          "button: needs two shapes")
    settings.resize_use_keys = True
    check(operators.check_resize_ready(bpy.context) == "From and To are the same shape",
          "button: needs two shapes, with shapekeys too")
    settings.resize_to_key = "Small"
    check(operators.check_resize_ready(bpy.context) is None, "button: ready")
    check(operators.resize_key_name(bpy.context) == "Small", "name: after the To key")
    before = {name: resize.positions(obj) for name, obj in garments.items()}
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    check(operators.check_resize_ready(bpy.context) == "Select the meshes to resize", "button: needs meshes")
    for obj in garments.values():
        obj.select_set(True)
    body.select_set(True)
    check(bpy.ops.magic_fit.resize() == {'FINISHED'}, "button: finishes")

    check(abs(settings.resize_bridge - resizing.BRIDGE_RADIUS) < 1e-6, "button: bridges creases by default")
    shape_change = resizing.body_shapes(body, "", None, "Small").change(radius=settings.resize_radius,
                                                                       bridge=settings.resize_bridge)
    rigid = np.full(sum(len(co) for co in before.values()), -1)
    rigid[-8:] = 0  # the pendant is the only part smaller than Rigid Parts
    expected = shape_change.apply(np.concatenate(list(before.values())), rigid=rigid,
                                  edges=garment_edges(garments.values()))
    got = np.concatenate([resize.positions(obj, "Small") for obj in garments.values()])
    check(np.abs(got - expected).max() < 1e-5, "button: every selected mesh got the resized shape, kept close ({:.2g} m off)".format(
        np.abs(got - expected).max()))
    check(body.data.shape_keys.key_blocks.get("Small") is not None and len(body.data.shape_keys.key_blocks) == 2,
          "button: the body gets no key")
    suit = garments["Suit"]
    keys = suit.data.shape_keys.key_blocks
    check([block.name for block in keys] == ["Basis", "Small"] and keys["Small"].value == 1.0
          and suit.active_shape_key.name == "Small" and keys["Small"].relative_key == keys["Basis"],
          "button: a key relative to the basis, fully on and active")
    check(np.array_equal(resize.positions(suit, "Basis"), before["Suit"]), "button: the basis stays as it was")
    check(body.data.shape_keys.key_blocks["Small"].value == 0.0, "button: the body is left as it was (undo couldn't restore it)")
    check(not resizing.shows_to_shape(body, "", None, "Small"), "hint: the body doesn't show the To shape")
    body.data.shape_keys.key_blocks["Small"].value = 1.0
    check(resizing.shows_to_shape(body, "", None, "Small"), "hint: now it does")

    settings.resize_from_key = "Small"
    settings.resize_to_key = ""
    settings.resize_key_name = "Large"
    bpy.ops.magic_fit.resize()
    keys = suit.data.shape_keys.key_blocks
    back = resize.positions(suit, "Large")
    check([block.name for block in keys] == ["Basis", "Small", "Large"] and keys["Small"].value == 0.0
          and keys["Large"].value == 1.0, "again: a new key, from the one shown, which is turned off")
    check(np.median(np.linalg.norm(back - before["Suit"], axis=1)) < 0.001, "again: back to the large breasts")
    settings.resize_key_name = ""
    check(operators.resize_key_name(bpy.context) == "Resized", "name: 'Resized' when To is the body's basis")


def test_operator_keep_close_off():
    body, garments = resize.build()
    settings = reset_settings()
    check(settings.resize_keep_close, "keep close: on by default")
    settings.resize_from_object = body
    settings.resize_to_key = "Small"
    settings.resize_keep_close = False
    before = np.concatenate([resize.positions(obj) for obj in garments.values()])
    check(bpy.ops.magic_fit.resize() == {'FINISHED'}, "keep close off: finishes")
    shape_change = resizing.body_shapes(body, "", None, "Small").change(radius=settings.resize_radius,
                                                                       bridge=settings.resize_bridge)
    rigid = np.full(len(before), -1)
    rigid[-8:] = 0
    expected = shape_change.apply(before, rigid=rigid)
    got = np.concatenate([resize.positions(obj, "Small") for obj in garments.values()])
    check(np.abs(got - expected).max() < 1e-5, "keep close off: just the smooth move")


def test_operator_matched_bodies():
    body, garments = resize.build()
    coarse = resize.coarse_body()
    settings = reset_settings()
    settings.resize_from_object = coarse
    settings.resize_to_object = body
    settings.resize_to_key = "Small"
    for obj in bpy.context.view_layer.objects:
        obj.select_set(True)
    check(operators.check_resize_ready(bpy.context) is None, "matched button: ready")
    check({obj.name for obj in operators.resize_targets(bpy.context)} == set(garments),
          "matched button: never resizes the From or To body")
    check(operators.resize_key_name(bpy.context) == "Small", "matched button: named after the To key")
    before = np.concatenate([resize.positions(obj) for obj in garments.values()])
    check(bpy.ops.magic_fit.resize() == {'FINISHED'}, "matched button: finishes")
    shape_change = resizing.body_shapes(coarse, "", body, "Small").change(radius=settings.resize_radius,
                                                                         bridge=settings.resize_bridge)
    rigid = np.full(len(before), -1)
    rigid[-8:] = 0  # the pendant
    expected = shape_change.apply(before, rigid=rigid, edges=garment_edges(garments.values()))
    got = np.concatenate([resize.positions(obj, "Small") for obj in garments.values()])
    check(np.abs(got - expected).max() < 1e-5, "matched button: the meshes got the resized shape")
    check(not resizing.shows_to_shape(coarse, "", body, "Small"), "matched hint: the To body doesn't show its To key")
    body.data.shape_keys.key_blocks["Small"].value = 1.0
    check(resizing.shows_to_shape(coarse, "", body, "Small"), "matched hint: now it does")


def test_operator_shape_matched():
    body, garments = resize.build()
    plain = resize.coarse_body("Plain Body", uv=None)
    settings = reset_settings()
    check(settings.resize_match == 'AUTO', "match by: Auto by default")
    settings.resize_from_object = plain
    settings.resize_to_object = body
    settings.resize_to_key = "Small"
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj.name in garments)
    check(operators.check_resize_ready(bpy.context) is None, "shape button: ready with a body of another kind")
    settings.resize_match = 'VERTICES'
    check(operators.check_resize_ready(bpy.context) == "'Plain Body' and 'Body' have different vertices: match them "
                                                      "by UV map or shape", "shape button: not vertex by vertex")
    settings.resize_match = 'UV'
    check(operators.check_resize_ready(bpy.context) == "'Plain Body' has no UV map to match it to 'Body' by",
          "shape button: not by UV map")
    settings.resize_match = 'SHAPE'
    before = np.concatenate([resize.positions(obj) for obj in garments.values()])
    check(bpy.ops.magic_fit.resize() == {'FINISHED'}, "shape button: finishes")
    shapes = resizing.body_shapes(plain, "", body, "Small", 'SHAPE')
    change = shapes.change(radius=settings.resize_radius, bridge=settings.resize_bridge)
    check(change.to_body is not None and change.known is shapes.match.known,
          "shape button: keeps clothing close to the To body itself")
    rigid = np.full(len(before), -1)
    rigid[-8:] = 0  # the pendant
    expected = change.apply(before, rigid=rigid, edges=garment_edges(garments.values()))
    got = np.concatenate([resize.positions(obj, "Small") for obj in garments.values()])
    check(np.abs(got - expected).max() < 1e-5, "shape button: the meshes got the resized shape ({:.2g} m off)".format(
        np.abs(got - expected).max()))


def test_operator_edit_mode_and_drivers():
    body, garments = resize.build()
    settings = reset_settings()
    settings.resize_from_object = body
    settings.resize_to_key = "Small"
    # A key of the mesh that a driver holds on can't be turned off: it's reported, and stays.
    suit = garments["Suit"]
    suit.shape_key_add(name="Basis", from_mix=False)
    held = suit.shape_key_add(name="Held", from_mix=False)
    held.value = 1.0
    curve = held.driver_add("value")
    curve.driver.expression = "1.0"
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj == garments["Suit"])
    bpy.context.view_layer.objects.active = garments["Suit"]
    bpy.ops.object.mode_set(mode='EDIT')
    check(operators.resize_targets(bpy.context) == [garments["Suit"]], "edit mode: resizes the meshes in Edit Mode")
    check(bpy.ops.magic_fit.resize() == {'FINISHED'}, "edit mode: finishes")
    check(bpy.context.object.mode == 'EDIT', "edit mode: back in Edit Mode")
    bpy.ops.object.mode_set(mode='OBJECT')
    check(suit.data.shape_keys.key_blocks.get("Small") is not None, "edit mode: the key was added")
    _block, stuck = resizing.add_resized_key(suit, "Again", resize.positions(suit, "Basis"))
    check(stuck == ["Held"] and suit.data.shape_keys.key_blocks["Held"].value == 1.0,
          "driven: a key held by a driver is reported, and stays on")

    keyless = resize.mesh_object("Keyless", *resize.cube((0, 0, 0), 1.0))
    settings.resize_from_object = keyless
    settings.resize_to_key = ""
    settings.resize_use_keys = True
    check("no shape keys" in operators.check_resize_ready(bpy.context), "button: a body without keys needs a To body")


def test_use_shapekeys():
    body, _garments = resize.build()
    settings = reset_settings()
    check(not settings.resize_use_keys, "shapekeys: off by default")
    settings.resize_from_object = body
    check(operators.check_resize_ready(bpy.context) == "Pick a To Body, or turn on Sizes or Use Shapekeys",
          "shapekeys: off, the From Body alone is one shape")
    settings.resize_use_keys = True
    settings.resize_to_key = "Small"
    check(operators.check_resize_ready(bpy.context) is None and operators.resize_keys(settings) == ("", "Small")
          and operators.resize_key_name(bpy.context) == "Small", "shapekeys: on, the To Shapekey is used")
    settings.resize_use_keys = False
    check(operators.resize_keys(settings) == ("", "") and operators.resize_key_name(bpy.context) == "Resized"
          and settings.resize_to_key == "Small", "shapekeys: off again, the key stays picked but isn't used")
    settings.property_unset("resize_use_keys_state")
    check(settings.resize_use_keys, "shapekeys: never set, as in files from before it: on where a key is picked")


# -----------------------------------------------------------------------------
# Sizes

def key_states(obj):
    return [(block.name, block.value, block.mute) for block in obj.data.shape_keys.key_blocks]


def devkit_state():
    state = bpy.context.scene.ya_devkit_props.mannequin_state
    return [getattr(state, name) for name in ("chest_size", "leg_size", "rue", "buff", "lavabod", "alt_hips")]


def size_error(body, size):
    try:
        sizing.size_values(body, size)
    except resizing.ResizeError as error:
        return str(error)
    return None


def listed(body):
    settings, problems = sizing.body_settings(body)
    return {setting.name: setting for setting in settings}, problems


def select_only(objects):
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj in objects)


def test_size_values():
    body, _garments = resize.devkit_body()
    keys = body.data.shape_keys
    state = bpy.context.scene.ya_devkit_props.mannequin_state
    settings, problems = listed(body)
    check(set(settings) == {"chest_size", "leg_size", "rue", "buff", "lavabod", "alt_hips"} and not problems
          and sizing.body_settings(None) == ([], []), "sizes: the settings driving the body's keys")
    chest = settings["chest_size"]
    check(chest.kind == 'ENUM' and [option.label for option in chest.options] == ["Large", "Medium", "Small", "Masc"]
          and [option.value for option in chest.options] == [0.0, 1.0, 2.0, 3.0],
          "sizes: an enum's items, listed by a function too, by number")
    check(settings["rue"].kind == 'BOOL' and settings["rue"].label == "Rue" and settings["leg_size"].label == "Legs",
          "sizes: switches, and labels for properties without names")
    check(settings["leg_size"].option(3.0).word == "Masc Legs", "sizes: the devkit's words for leg sizes")
    state.chest_size = "2"
    state.leg_size = "1"
    names = list(settings)
    check(sizing.now_values(body, names) == {"chest_size": 2.0, "leg_size": 1.0, "rue": 0.0, "buff": 0.0,
                                             "lavabod": 0.0, "alt_hips": 0.0}, "sizes: the settings now, by number")
    state.chest_size = "0"
    state.leg_size = "0"
    before, settings_before = key_states(body), devkit_state()

    check(sizing.size_values(body, {"chest_size": 1}) == {"MEDIUM": 1.0, "- Squeeze": 0.0},
          "sizes: Medium turns MEDIUM on and mutes the Large sub-key")
    check(sizing.size_values(body, {"chest_size": 0, "rue": 0}) == {}, "sizes: the sizes it shows change nothing")
    check(sizing.size_values(body, {"rue": 1}) == {"Rue": 1.0, "Rue/Follow": 1.0},
          "sizes: Rue, and the key that follows it")
    check(sizing.size_values(body, {"rue": 1, "buff": 1}) == {"Rue": 1.0, "Buff": 1.0, "Rue/Buff": 1.0,
                                                              "Rue/Follow": 1.0},
          "sizes: Rue with Buff adds their correction")
    check(sizing.size_values(body, {"lavabod": 1, "chest_size": 1}) == {"Lavabod": 1.0, "- Squeeze": 0.0},
          "sizes: Lavabod's medium chest isn't MEDIUM")
    check(sizing.size_values(body, {"lavabod": 1, "chest_size": 3}) == sizing.size_values(
        body, {"lavabod": 1, "chest_size": 2}), "sizes: Lavabod has no Masc chest, it takes Small")
    check(sizing.size_values(body, {"lavabod": 1}) == {"Lavabod": 1.0, "- Squeeze": 0.0},
          "sizes: switching Lavabod on the first time takes '- Squeeze' off, as the devkit does")
    kept = state.lava_keys.add()
    kept.name, kept.value = "- Squeeze", 0.5
    check(sizing.size_values(body, {"lavabod": 1}) == {"Lavabod": 1.0, "- Squeeze": 0.5},
          "sizes: ...or gives the sub-keys the values kept from last time")
    state.alt_hips = True
    keys.key_blocks["Hip Dips (for YAB)"].value = 1.0
    check(sizing.size_values(body, {"leg_size": 1}) == {"Skull Crushers": 1.0},
          "sizes: hip dips stay with the first leg sizes")
    check(sizing.size_values(body, {"leg_size": 3}) == {"Hip Dips (for YAB)": 0.0},
          "sizes: ...and go with the others (the devkit turns Alt Hips off)")
    check(sizing.size_values(body, {"rue": 1}).get("Hip Dips (for YAB)") == 0.0, "sizes: ...and with Rue")
    state.alt_hips = False
    keys.key_blocks["Hip Dips (for YAB)"].value = 0.0
    state.lava_keys.clear()
    check(key_states(body) == before and devkit_state() == settings_before,
          "sizes: working the sizes out changes neither the body nor the devkit")

    rest = resizing.rest_shape(body, sizing.size_values(body, {"chest_size": 1, "rue": 1}))
    expected = resizing.rest_shape(body, {"MEDIUM": 1.0, "- Squeeze": 0.0, "Rue": 1.0, "Rue/Follow": 1.0})
    check(np.abs(rest - expected).max() == 0.0, "sizes: the To shape mixes the keys at those values")

    resize.drive(keys, "Buff", 'mute', "size == 1 and nope == 1", (("size", "chest_size"), ("nope", "nope")))
    unreadable = size_error(body, {"chest_size": 1})
    check(unreadable is not None and "is the devkit's script running" in unreadable,
          "sizes: a setting that can't be read is refused ({!s})".format(unreadable))
    check(listed(body)[1] == ["'ya_devkit_props.mannequin_state.nope' can't be read: is the devkit's script running?"],
          "sizes: ...and said when listing them")
    keys.key_blocks["Buff"].driver_remove("mute")
    resize.drive(keys, "Buff", 'mute', "__import__('os').getpid() > 0 and size", (("size", "chest_size"),))
    unsafe = size_error(body, {"chest_size": 1})
    check(unsafe is not None and "can't be worked out" in unsafe, "sizes: expressions that could run code aren't run")
    keys.key_blocks["Buff"].driver_remove("mute")

    settings_list, _problems = sizing.body_settings(body)
    labels = {
        "Medium": ({"chest_size": 1}, {"chest_size": 0}),
        "Rue Buff": ({"rue": 1, "buff": 1, "chest_size": 0}, {"rue": 0, "buff": 0, "chest_size": 0}),
        "Teardrop Lavabod": ({"chest_size": 1, "lavabod": 1}, {"chest_size": 0, "lavabod": 0}),
        "Masc Legs": ({"leg_size": 3}, {"leg_size": 0}),
        "Large YAB": ({"chest_size": 0, "lavabod": 0}, {"chest_size": 0, "lavabod": 1}),
        "Rue": ({"lavabod": 0, "rue": 1}, {"lavabod": 1, "rue": 0}),
        "No Buff": ({"buff": 0}, {"buff": 1}),
        "": ({"chest_size": 2}, {"chest_size": 2}),
    }
    for label, (size, now) in labels.items():
        check(sizing.size_label(settings_list, size, now) == label, "size label: {!r}".format(label))
    now = {"chest_size": 0.0, "leg_size": 0.0, "rue": 0.0, "buff": 0.0, "lavabod": 0.0, "alt_hips": 0.0}
    sizes = sizing.combinations(settings_list, {"chest_size": {0.0, 1.0, 2.0}, "rue": {0.0, 1.0}}, now)
    named = [sizing.size_label(settings_list, size, now, {"chest_size", "rue"}) for size in sizes]
    check(sorted(named) == ["Large", "Large Rue", "Medium", "Medium Rue", "Small", "Small Rue"],
          "combinations: every one of the values picked, named by what's picked ({!s})".format(named))
    sizes = sizing.combinations(settings_list, {"chest_size": {2.0, 3.0}, "lavabod": {1.0}}, now)
    check(len(sizes) == 1 and sizes[0]["chest_size"] == 2.0, "combinations: Masc Lavabod is Small Lavabod, given once")


def test_body_parts():
    body, _garments = resize.devkit_body()
    part = resize.devkit_part(body)
    names = [setting.name for setting in sizing.body_settings(body)[0]]
    check(sizing.follows(part, names) and not sizing.follows(resize.coarse_body(), names),
          "parts: a body part follows the body's settings, from its own group")
    check(sizing.size_values(part, {"chest_size": 1, "rue": 1}) == {"MEDIUM": 1.0, "Rue": 1.0},
          "parts: its keys worked out with the same sizes")
    copy = body.copy()
    copy.data = body.data.copy()
    copy.data.shape_keys.animation_data_clear()
    bpy.context.scene.collection.objects.link(copy)
    body_values = sizing.size_values(body, {"chest_size": 1, "rue": 1})
    check(sizing.follows(copy, names, body_values)
          and sizing.size_values(copy, {"chest_size": 1, "rue": 1}, body_values) == body_values,
          "parts: a copy without drivers takes the body's keys by name")


def test_custom_property_settings():
    body, _garments, driver = resize.driver_body()
    settings, problems = listed(body)
    check(set(settings) == {"Buff", "Heels", "Nipple Extra"} and not problems,
          "custom properties: an object's properties driving the keys are settings")
    heels = settings["Heels"]
    check(heels.kind == 'NUMBER' and [option.value for option in heels.options] == [0.0, 1.0, 2.0, 3.0]
          and [option.label for option in heels.options] == ["None", "Heels", "Cinderella", "3"],
          "custom properties: numbers in their range, named after the keys they turn on ({!s})".format(
              [option.label for option in heels.options]))
    check([option.label for option in settings["Nipple Extra"].options] == ["Default Type", "Inverted", "Flatter"],
          "custom properties: ...or as their description names them")
    check(settings["Buff"].kind == 'BOOL', "custom properties: a switch")
    check(sizing.size_values(body, {"Heels": 2, "Buff": 1}) == {"Cinderella": 1.0, "Buff": 1.0},
          "custom properties: worked out without switching them")
    check(driver["Heels"] == 0 and not driver["Buff"], "custom properties: left as they were")
    driver["Heels"] = 2
    check(sizing.now_values(body, ["Heels"]) == {"Heels": 2.0}, "custom properties: read as they are now")
    driver["Heels"] = 0


def test_plain_key_sizes():
    body, garments = resize.build()
    settings, _problems = listed(body)
    check(list(settings) == ["key:Small"] and settings["key:Small"].kind == 'KEY'
          and settings["key:Small"].label == "Small", "plain keys: a body whose keys nothing drives has them as sizes")
    check(sizing.size_values(body, {"key:Small": 1.0}) == {"Small": 1.0}, "plain keys: turned on")
    brush = reset_settings()
    brush.resize_from_object = body
    brush.resize_use_sizes = True
    check([item.name for item in brush.resize_sizes] == ["key:Small"], "plain keys: listed in the panel")
    bpy.ops.magic_fit.resize_pick(setting="key:Small", option=1)
    select_only(garments.values())
    check(operators.check_resize_ready(bpy.context) is None and operators.resize_key_name(bpy.context) == "Small",
          "plain keys: ready, named after the key")
    before = np.concatenate([resize.positions(obj) for obj in garments.values()])
    check(bpy.ops.magic_fit.resize() == {'FINISHED'}, "plain keys: resized")
    change = resizing.body_shapes(body, "", None, "Small").change(radius=brush.resize_radius, bridge=brush.resize_bridge)
    rigid = np.full(len(before), -1)
    rigid[-8:] = 0
    expected = change.apply(before, rigid=rigid, edges=garment_edges(garments.values()))
    got = np.concatenate([resize.positions(obj, "Small") for obj in garments.values()])
    check(np.abs(got - expected).max() < 1e-5, "plain keys: as resizing to the key ({:.2g} m off)".format(
        np.abs(got - expected).max()))


def test_sizes_operator():
    body, garments = resize.devkit_body()
    settings = reset_settings()
    settings.resize_from_object = body
    check(operators.check_resize_ready(bpy.context) == "Pick a To Body, or turn on Sizes or Use Shapekeys",
          "sizes: offered for a body with sizes")
    state = bpy.context.scene.ya_devkit_props.mannequin_state
    skull = body.data.shape_keys.key_blocks["Skull Crushers"]
    state.leg_size = "1"
    skull.value = 1.0  # as its driver sets it
    settings.resize_use_sizes = True
    check(properties.picked_of(settings.resize_sizes["leg_size"]) == {1.0}
          and properties.picked_of(settings.resize_sizes["chest_size"]) == {0.0},
          "sizes: turned on, they start from the body's")
    check(operators.check_resize_ready(bpy.context) == "The body shows these sizes now: pick others",
          "sizes: the same as now is nothing to do")
    state.leg_size = "0"
    skull.value = 0.0
    bpy.ops.magic_fit.resize_pick(setting="leg_size", option=0)
    bpy.ops.magic_fit.resize_pick(setting="chest_size", option=1)
    check(operators.check_resize_ready(bpy.context) is None and operators.resize_key_name(bpy.context) == "Medium",
          "sizes: ready, named after what changes")
    bpy.ops.magic_fit.resize_pick(setting="rue", option=1)
    check(operators.resize_key_name(bpy.context) == "Medium Rue", "sizes: named after every size that changes")

    before = {name: resize.positions(obj) for name, obj in garments.items()}
    keys_before, settings_before = key_states(body), devkit_state()
    check(bpy.ops.magic_fit.resize() == {'FINISHED'}, "sizes: resized")
    check(key_states(body) == keys_before and devkit_state() == settings_before,
          "sizes: neither the body nor the devkit was switched")

    # As before: a copy of the body without its drivers, its keys set by hand.
    copy = body.copy()
    copy.data = body.data.copy()
    copy.data.shape_keys.animation_data_clear()
    bpy.context.scene.collection.objects.link(copy)
    blocks = copy.data.shape_keys.key_blocks
    blocks["MEDIUM"].value = blocks["Rue"].value = blocks["Rue/Follow"].value = 1.0
    blocks["- Squeeze"].mute = True
    shape_change = resizing.body_shapes(body, "", copy, "").change(radius=settings.resize_radius,
                                                                   bridge=settings.resize_bridge)
    rigid = np.full(sum(len(co) for co in before.values()), -1)
    rigid[-8:] = 0
    expected = shape_change.apply(np.concatenate(list(before.values())), rigid=rigid,
                                  edges=garment_edges(garments.values()))
    got = np.concatenate([resize.positions(obj, "Medium Rue") for obj in garments.values()])
    check(np.abs(got - expected).max() < 1e-5, "sizes: as resizing to a hand-set copy ({:.2g} m off)".format(
        np.abs(got - expected).max()))
    suit = garments["Suit"]
    check(suit.data.shape_keys.key_blocks["Medium Rue"].value == 1.0, "sizes: one size is shown")

    # Several at once: Shift-click picks more. The suit shows Medium Rue now, but each starts from its own shape.
    bpy.ops.magic_fit.resize_pick(setting="chest_size", option=2, extend=True)
    picks = operators.resize_sizes(settings)
    check([size.name for size in picks.changed] == ["Medium Rue", "Small Rue"]
          and operators.check_resize_ready(bpy.context) is None, "several: every combination of the values picked")
    check(bpy.ops.magic_fit.resize() == {'FINISHED'}, "several: resized")
    blocks = suit.data.shape_keys.key_blocks
    check([block.name for block in blocks] == ["Basis", "Medium Rue", "Small Rue"],
          "several: a key per size, the one made before kept")
    small = resizing.body_shapes(body, "", None, "", 'AUTO', sizing.size_values(body, picks.changed[1].values))
    expected = small.change(radius=settings.resize_radius, bridge=settings.resize_bridge).apply(
        np.concatenate(list(before.values())), rigid=rigid, edges=garment_edges(garments.values()))
    got = np.concatenate([resize.positions(obj, "Small Rue") for obj in garments.values()])
    check(np.abs(got - expected).max() < 1e-5, "several: from the clothing's own shape, not the size it shows "
                                               "({:.2g} m off)".format(np.abs(got - expected).max()))

    part = resize.devkit_part(body)
    select_only(list(garments.values()) + [part])
    check(bpy.ops.magic_fit.resize() == {'FINISHED'} and part.data.shape_keys.key_blocks.get("Medium Rue") is None,
          "several: a body part selected along isn't resized")

    settings.resize_to_object = copy
    picks = operators.resize_sizes(settings)
    check(picks is not None and all(setting.kind == 'KEY' for setting in picks.settings),
          "sizes: a To body nothing drives has its keys as sizes")
    settings.resize_to_object = None
    settings.resize_use_sizes = False
    check(operators.resize_sizes(settings) is None, "sizes: off")


def test_reused_keys():
    body, garments = resize.devkit_body()
    settings = reset_settings()
    settings.resize_from_object = body
    settings.resize_use_sizes = True
    bpy.ops.magic_fit.resize_pick(setting="chest_size", option=1)
    select_only(garments.values())
    suit = garments["Suit"]
    bpy.ops.magic_fit.resize()
    block = suit.data.shape_keys.key_blocks["Medium"]
    made = resize.positions(suit, "Medium")
    bridge_made = resize.positions(garments["Bridge"], "Medium")
    record = resizing.resized_keys(suit)["Medium"]
    check(record["label"] == "Medium" and record["count"] == len(suit.data.vertices) and record["mix"] == {},
          "records: what the key was made from")
    check(resizing.key_state(suit, "Medium", record["inputs"]) == (block, 'FRESH'), "records: fresh")
    bpy.ops.magic_fit.resize()
    check(np.array_equal(resize.positions(suit, "Medium"), made) and len(suit.data.shape_keys.key_blocks) == 2,
          "again: the key made from the same shapes is kept, not made twice")

    settings.resize_radius = 0.012
    bpy.ops.magic_fit.resize()
    remade = resize.positions(suit, "Medium")
    check(not np.array_equal(remade, made) and len(suit.data.shape_keys.key_blocks) == 2,
          "changed: made again in place when the options change")
    # Fixed by hand, then the options change: the fix stays.
    fixed = remade.copy()
    fixed[:, 1] -= 0.001
    suit.data.shape_keys.key_blocks["Medium"].data.foreach_set("co", fixed.astype(np.float32).ravel())
    settings.resize_radius = 0.01
    bpy.ops.magic_fit.resize()
    check(np.allclose(resize.positions(suit, "Medium"), fixed, atol=1e-6), "edited: a key fixed by hand is kept")
    check(np.array_equal(resize.positions(garments["Bridge"], "Medium"), bridge_made),
          "edited: the other meshes' keys were made again, as at first")


def test_resize_collection():
    body, garments = resize.devkit_body()
    part = resize.devkit_part(body)
    settings = reset_settings()
    settings.resize_from_object = body
    settings.resize_use_sizes = True
    bpy.ops.magic_fit.resize_pick(setting="chest_size", option=0)
    bpy.ops.magic_fit.resize_pick(setting="chest_size", option=1, extend=True)
    bpy.ops.magic_fit.resize_pick(setting="rue", option=1)
    collection = bpy.data.collections.new("Outfit")
    bpy.context.scene.collection.children.link(collection)
    for obj in list(garments.values()) + [part, body]:
        collection.objects.link(obj)
    sizes = api.picked_sizes()
    check([(size.name, size.now) for size in sizes] == [("Large Rue", False), ("Medium Rue", False)],
          "api: the sizes picked ({!s})".format(sizes))
    results = api.resize_collection(collection)
    check([result.name for result in results] == ["Large Rue", "Medium Rue"], "api: a result per size")
    medium = results[1]
    check(set(medium.keys) == set(garments.values()) and part in medium.parts and body not in medium.parts,
          "api: clothing resized, the body part set, the body left out")
    # Its keys the torso's settings drive are worked out; the others take the body's values by name.
    check(medium.parts.get(part) == {"MEDIUM": 1.0, "Rue": 1.0, "- Squeeze": 0.0, "Rue/Follow": 1.0},
          "api: the part's keys at the size")
    suit = garments["Suit"]
    blocks = suit.data.shape_keys.key_blocks
    check([block.value for block in blocks][1:] == [0.0, 0.0], "api: the clothing shows its own shape still")
    part_keys = part.data.shape_keys
    drivers_before = [curve.mute for curve in part_keys.animation_data.drivers]
    states_before = key_states(part), key_states(suit)
    with medium.shown():
        check(blocks[medium.keys[suit]].value == 1.0 and blocks[results[0].keys[suit]].value == 0.0,
              "shown: the clothing's key for the size on")
        check(part_keys.key_blocks["MEDIUM"].value == 1.0 and part_keys.key_blocks["Rue"].value == 1.0
              and all(curve.mute for curve in part_keys.animation_data.drivers
                      if curve.data_path.split('"')[1] in ("MEDIUM", "Rue")),
              "shown: the part's keys set, their drivers muted meanwhile")
        bpy.context.view_layer.update()
        evaluated = suit.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        co = np.empty(len(evaluated.vertices) * 3)
        evaluated.vertices.foreach_get("co", co)
        check(np.allclose(co.reshape(-1, 3), resize.positions(suit, medium.keys[suit]), atol=1e-6),
              "shown: what an export reads is the size")
    check((key_states(part), key_states(suit)) == states_before
          and [curve.mute for curve in part_keys.animation_data.drivers] == drivers_before,
          "shown: everything put back afterwards")
    try:
        with medium.shown():
            raise RuntimeError("export failed")
    except RuntimeError:
        pass
    check((key_states(part), key_states(suit)) == states_before, "shown: put back after an error too")

    again = api.resize_collection(list(garments.values()) + [part], [{"chest_size": 2}, {"chest_size": 0}])
    check([(result.name, result.size.now) for result in again] == [("Small", False), ("Large", True)]
          and all(name is None for name in again[1].keys.values()) and not again[1].parts,
          "api: sizes given, the one the body shows kept as it is ({!s})".format(again))
    check(sum(1 for block in blocks if block.name == "Small") == 1, "api: the key for the size given")


def test_resize_collection_as_is():
    """The size the body shows exports the clothing as it fits the body, also while it shows another size."""
    body, garments = resize.devkit_body()
    settings = reset_settings()
    settings.resize_from_object = body
    settings.resize_use_sizes = True
    bpy.ops.magic_fit.resize_pick(setting="chest_size", option=0)
    bpy.ops.magic_fit.resize_pick(setting="chest_size", option=1, extend=True)
    suit = garments["Suit"]

    def evaluated():
        bpy.context.view_layer.update()
        mesh = suit.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        return co.reshape(-1, 3)

    own = evaluated()
    large, medium = api.resize_collection(list(garments.values()))
    check(large.size.now and large.keys[suit] is None and medium.keys[suit] is not None,
          "as is: the size the body shows keeps the clothing as it is")
    blocks = suit.data.shape_keys.key_blocks
    # Looking at a size, as the Resize button leaves one size on.
    blocks[medium.keys[suit]].value = 1.0
    large, medium = api.resize_collection(list(garments.values()))
    before = key_states(suit)
    with large.shown():
        check(np.allclose(evaluated(), own, atol=1e-6) and blocks[medium.keys[suit]].value == 0.0,
              "as is: shown as it fits the body, not the size it shows")
    check(key_states(suit) == before, "as is: the size it shows put back afterwards")
    with medium.shown():
        check(np.allclose(evaluated(), resize.positions(suit, medium.keys[suit]), atol=1e-6),
              "as is: the other size still shows its key")


def test_resizer():
    """A size at a time (api.Resizer) makes the keys resize_collection makes."""
    body, garments = resize.devkit_body()
    settings = reset_settings()
    settings.resize_from_object = body
    settings.resize_use_sizes = True
    bpy.ops.magic_fit.resize_pick(setting="chest_size", option=1)
    bpy.ops.magic_fit.resize_pick(setting="chest_size", option=2, extend=True)
    bpy.ops.magic_fit.resize_pick(setting="rue", option=1)
    part = resize.devkit_part(body)
    meshes = list(garments.values()) + [part]
    resizer = api.Resizer(meshes)
    names = [size.name for size in resizer.sizes]
    check(names == [size.name for size in api.picked_sizes()] == ["Medium Rue", "Small Rue"]
          and resizer.parts == [part] and set(resizer.clothing) == set(garments.values()),
          "resizer: the sizes picked, the body part left out")
    progress = []
    one = [resizer.resize(size, progress=progress.append) for size in resizer.sizes]
    check(progress and progress[-1] == 1.0 and all(0.0 <= done <= 1.0 for done in progress), "resizer: progress")

    def made(results):
        return {(sized.name, obj.name): resize.positions(obj, name)
                for sized in results for obj, name in sized.keys.items() if name is not None}
    first = made(one)
    for obj in garments.values():
        for name in resizing.resized_keys(obj):
            obj.shape_key_remove(obj.data.shape_keys.key_blocks[name])
        del obj.data.shape_keys[resizing.RECORD]
    together = api.resize_collection(meshes)
    again = made(together)
    check(set(first) == set(again) and len(first) == 2 * len(garments)
          and max(float(np.abs(first[key] - again[key]).max()) for key in first) < 1e-6
          and [sized.parts == other.parts for sized, other in zip(one, together)] == [True, True],
          "resizer: the keys and body part values resize_collection gives")
    small = resizer.resize({"chest_size": 2, "rue": 1})
    check(small.name == "Small Rue" and small.keys == together[1].keys, "resizer: a size given as values, kept")
    check(api.VERSION >= 2, "resizer: VERSION 2 has it")


def test_registered():
    check(hasattr(bpy.ops.magic_fit, "resize"), "the Resize operator is registered")
    settings = bpy.context.scene.body_fit_brush
    for prop in ("resize_from_object", "resize_from_key", "resize_to_object", "resize_to_key", "resize_match",
                 "resize_radius", "resize_bridge", "resize_rigid_size", "resize_keep_close", "resize_key_name",
                 "resize_use_keys", "resize_use_sizes", "resize_sizes"):
        check(prop in settings.bl_rna.properties, "setting: " + prop)
    check(hasattr(bpy.ops.magic_fit, "resize_pick") and hasattr(bpy.ops.magic_fit, "resize_sizes_read"),
          "the Sizes buttons are registered")
    check(api.VERSION >= 1 and callable(api.resize_collection) and callable(api.picked_sizes),
          "the API for other add-ons is there")


def main():
    magic_fit.register()
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception:
            FAILURES.append(test.__name__ + " raised")
            traceback.print_exc()
        finally:
            if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
    print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
    sys.exit(1 if FAILURES else 0)


main()
