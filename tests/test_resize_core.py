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
from magic_fit import fitting, lineup, operators, properties, resizing  # noqa: E402

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
                 "resize_use_keys_state", "resize_use_sizes", "resize_chest", "resize_legs", "resize_rue",
                 "resize_buff", "resize_lavabod"):
        settings.property_unset(prop)
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

    co = np.concatenate([before, before + (0.1, 0.0, 0.0), before + (0.0, 0.0, 0.2)])
    edge_list = np.concatenate([edge_list, edge_list + 8, edge_list + 16])
    parts, count = resizing.loose_parts(co, edge_list)
    check(count == 3 and len(set(parts[:8])) == 1, "loose parts: one per cube")
    co[16:] = co[:8]
    parts, count = resizing.loose_parts(co, edge_list)
    check(count == 2 and parts[0] == parts[16], "loose parts: parts at the same spot are one (split seams)")
    sizes = resizing.part_sizes(co, parts, count)
    check(np.allclose(sizes, 0.02 * np.sqrt(3.0)), "loose parts: sized by their bounding box")


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
    check(operators.check_resize_ready(bpy.context) == "Pick a To Body, or turn on Use Shapekeys",
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
    check(operators.check_resize_ready(bpy.context) == "Pick a To Body, or turn on Use Shapekeys",
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
# Devkit Sizes

def key_states(obj):
    return [(block.name, block.value, block.mute) for block in obj.data.shape_keys.key_blocks]


def devkit_state():
    state = bpy.context.scene.ya_devkit_props.mannequin_state
    return [getattr(state, name) for name in ("chest_size", "leg_size", "rue", "buff", "lavabod", "alt_hips")]


def devkit_error(body, sizes):
    try:
        resizing.devkit_values(body, sizes)
    except resizing.ResizeError as error:
        return str(error)
    return None


def test_devkit_values():
    body, _garments = resize.devkit_body()
    keys = body.data.shape_keys
    state = bpy.context.scene.ya_devkit_props.mannequin_state
    check(resizing.devkit_settings(body) == {"chest_size", "leg_size", "rue", "buff", "lavabod"}
          and resizing.devkit_settings(resize.coarse_body()) == set() and resizing.devkit_settings(None) == set(),
          "devkit: the settings driving the body's keys")
    state.chest_size = "2"
    state.leg_size = "1"
    check(resizing.devkit_state(body) == {"chest_size": 2, "leg_size": 1, "rue": 0, "buff": 0, "lavabod": 0},
          "devkit: its settings now, sizes by number (listed by a function too)")
    state.chest_size = "0"
    state.leg_size = "0"
    before, settings_before = key_states(body), devkit_state()

    check(resizing.devkit_values(body, {"chest_size": 1}) == {"MEDIUM": 1.0, "- Squeeze": 0.0},
          "devkit: Medium turns MEDIUM on and mutes the Large sub-key")
    check(resizing.devkit_values(body, {"chest_size": 0, "rue": 0}) == {}, "devkit: the sizes it shows change nothing")
    check(resizing.devkit_values(body, {"rue": 1}) == {"Rue": 1.0, "Rue/Follow": 1.0},
          "devkit: Rue, and the key that follows it")
    check(resizing.devkit_values(body, {"rue": 1, "buff": 1}) == {"Rue": 1.0, "Buff": 1.0, "Rue/Buff": 1.0,
                                                                  "Rue/Follow": 1.0},
          "devkit: Rue with Buff adds their correction")
    check(resizing.devkit_values(body, {"lavabod": 1, "chest_size": 1}) == {"Lavabod": 1.0, "- Squeeze": 0.0},
          "devkit: Lavabod's medium chest isn't MEDIUM")
    check(resizing.devkit_values(body, {"lavabod": 1, "chest_size": 3}) == resizing.devkit_values(
        body, {"lavabod": 1, "chest_size": 2}), "devkit: Lavabod has no Masc chest, it takes Small")
    check(resizing.devkit_values(body, {"lavabod": 1}) == {"Lavabod": 1.0, "- Squeeze": 0.0},
          "devkit: switching Lavabod on the first time takes '- Squeeze' off, as the devkit does")
    kept = state.lava_keys.add()
    kept.name, kept.value = "- Squeeze", 0.5
    check(resizing.devkit_values(body, {"lavabod": 1}) == {"Lavabod": 1.0, "- Squeeze": 0.5},
          "devkit: ...or gives the sub-keys the values kept from last time")
    state.alt_hips = True
    keys.key_blocks["Hip Dips (for YAB)"].value = 1.0
    check(resizing.devkit_values(body, {"leg_size": 1}) == {"Skull Crushers": 1.0},
          "devkit: hip dips stay with the first leg sizes")
    check(resizing.devkit_values(body, {"leg_size": 3}) == {"Hip Dips (for YAB)": 0.0},
          "devkit: ...and go with the others (the devkit turns Alt Hips off)")
    check(resizing.devkit_values(body, {"rue": 1}).get("Hip Dips (for YAB)") == 0.0, "devkit: ...and with Rue")
    state.alt_hips = False
    keys.key_blocks["Hip Dips (for YAB)"].value = 0.0
    state.lava_keys.clear()
    check(key_states(body) == before and devkit_state() == settings_before,
          "devkit: working the sizes out changes neither the body nor the devkit")

    rest = resizing.rest_shape(body, resizing.devkit_values(body, {"chest_size": 1, "rue": 1}))
    expected = resizing.rest_shape(body, {"MEDIUM": 1.0, "- Squeeze": 0.0, "Rue": 1.0, "Rue/Follow": 1.0})
    check(np.abs(rest - expected).max() == 0.0, "devkit: the To shape mixes the keys at those values")

    resize.drive(keys, "Buff", 'mute', "size == 1 and nope == 1", (("size", "chest_size"), ("nope", "nope")))
    unreadable = devkit_error(body, {"chest_size": 1})
    check(unreadable is not None and "is the devkit's script running" in unreadable,
          "devkit: a setting that can't be read is refused ({!s})".format(unreadable))
    keys.key_blocks["Buff"].driver_remove("mute")
    resize.drive(keys, "Buff", 'mute', "__import__('os').getpid() > 0 and size", (("size", "chest_size"),))
    unsafe = devkit_error(body, {"chest_size": 1})
    check(unsafe is not None and "can't be worked out" in unsafe, "devkit: expressions that could run code aren't run")
    keys.key_blocks["Buff"].driver_remove("mute")

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
    for label, (sizes, now) in labels.items():
        check(resizing.devkit_label(sizes, now) == label, "devkit label: {!r}".format(label))


def test_devkit_sizes_operator():
    body, garments = resize.devkit_body()
    settings = reset_settings()
    settings.resize_from_object = body
    check(operators.check_resize_ready(bpy.context) == "Pick a To Body, or turn on Devkit Sizes or Use Shapekeys",
          "sizes: offered for a devkit body")
    state = bpy.context.scene.ya_devkit_props.mannequin_state
    skull = body.data.shape_keys.key_blocks["Skull Crushers"]
    state.leg_size = "1"
    skull.value = 1.0  # as its driver sets it
    settings.resize_use_sizes = True
    check((settings.resize_chest, settings.resize_legs, settings.resize_rue) == ('LARGE', 'SKULL', False),
          "sizes: turned on, they start from the devkit's")
    check(operators.check_resize_ready(bpy.context) == "The devkit shows these sizes now: pick others",
          "sizes: the same as now is nothing to do")
    state.leg_size = "0"
    skull.value = 0.0
    settings.resize_legs = 'MELON'
    settings.resize_chest = 'MEDIUM'
    check(operators.resize_sizes(settings) == {"chest_size": 1, "leg_size": 0, "rue": 0, "buff": 0, "lavabod": 0}
          and operators.check_resize_ready(bpy.context) is None and operators.resize_key_name(bpy.context) == "Medium",
          "sizes: ready, named after what changes")

    settings.resize_chest = 'MASC'
    settings.resize_lavabod = True
    check(settings.resize_chest == 'SMALL', "sizes: Lavabod takes Small for Masc, as in the devkit")
    check([item[1] for item in properties._chest_items(settings, None)][:3] == ["Omoi", "Teardrop", "Cupcake"],
          "sizes: with Lavabod, its own chest names")
    settings.resize_chest = 'MASC'
    check(not settings.resize_lavabod, "sizes: Masc turns Lavabod off")
    settings.resize_chest = 'MEDIUM'
    settings.resize_rue = True
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

    settings.resize_to_object = copy
    check(operators.resize_sizes(settings) is None and operators.check_resize_ready(bpy.context) is None,
          "sizes: not for a To body the devkit doesn't drive")
    settings.resize_to_object = None
    settings.resize_use_sizes = False
    check(operators.resize_sizes(settings) is None, "sizes: off")


def test_registered():
    check(hasattr(bpy.ops.magic_fit, "resize"), "the Resize operator is registered")
    settings = bpy.context.scene.body_fit_brush
    for prop in ("resize_from_object", "resize_from_key", "resize_to_object", "resize_to_key", "resize_match",
                 "resize_radius", "resize_bridge", "resize_rigid_size", "resize_keep_close", "resize_key_name",
                 "resize_use_keys", "resize_use_sizes", "resize_chest", "resize_legs", "resize_rue", "resize_buff",
                 "resize_lavabod"):
        check(prop in settings.bl_rna.properties, "setting: " + prop)


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
