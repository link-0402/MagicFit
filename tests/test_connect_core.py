"""Headless tests for Connect Parts (the Connect Parts button of the Weights tab's General sub-tab).

Run from the repository root:
    blender -b --factory-startup --python tests/test_connect_core.py
"""

import math
import os
import sys
import traceback

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import skirt  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import connecting, operators  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


# -----------------------------------------------------------------------------
# The scene: a skirt on the rig of the Skirt tests, with a belt, a belt loop, a strap and a patch on it

SKIRT_R = 0.13
BELT_R = SKIRT_R + 0.003    # 3 mm off the skirt
LOOP_R = BELT_R + 0.003     # 3 mm off the belt
STRAP_R = SKIRT_R + 0.001
FAR_R = SKIRT_R + 0.015     # farther than Distance
SKIRT_TOP = 1.10
FRONT = -0.5 * math.pi


def ring(radius, heights, columns, seam=False):
    """A tube around the Z axis, as (co, faces): ring i at heights[i], column 0 at the front. With
    ``seam``, column 0 is repeated as a last column that isn't connected to it (a split seam)."""
    count = columns + 1 if seam else columns
    angles = FRONT + 2.0 * math.pi * np.arange(count) / columns
    co = [(radius * math.cos(a), radius * math.sin(a), z) for z in heights for a in angles]
    faces = [(i * count + j, i * count + (j + 1) % count, (i + 1) * count + (j + 1) % count, (i + 1) * count + j)
             for i in range(len(heights) - 1) for j in range(columns)]
    return co, faces


def patch(radius, heights, angles):
    """A piece of a tube around the Z axis, as (co, faces)."""
    n = len(angles)
    co = [(radius * math.cos(a), radius * math.sin(a), z) for z in heights for a in angles]
    faces = [(i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j)
             for i in range(len(heights) - 1) for j in range(n - 1)]
    return co, faces


def pieces():
    """{name: (co, faces, weights)} of the outfit's loose parts: the skirt weighted like the body under it
    (the hips handing over to the thighs from 0.96 down to 0.86), and the others weighted on their own."""
    result = {}
    co, faces = ring(SKIRT_R, np.linspace(SKIRT_TOP, 0.80, 31), 64, seam=True)
    result["skirt"] = (co, faces, skirt.body_weights(np.array(co)))
    co, faces = ring(BELT_R, np.linspace(0.97, 0.91, 7), 64)
    result["belt"] = (co, faces, {"j_kosi": np.full(len(co), 1.0)})
    co, faces = patch(LOOP_R, np.linspace(0.985, 0.895, 10), FRONT + np.radians([-4.0, 0.0, 4.0]))
    result["loop"] = (co, faces, {"j_kosi": np.full(len(co), 0.7), "j_sebo_a": np.full(len(co), 0.3)})
    co, faces = patch(STRAP_R, np.linspace(1.06, 1.24, 19), FRONT + np.radians([30.0, 34.0, 38.0]))
    result["strap"] = (co, faces, {"j_sebo_b": np.full(len(co), 1.0)})
    co, faces = patch(FAR_R, np.linspace(0.88, 0.84, 5), FRONT + np.radians([150.0, 155.0, 160.0]))
    result["far"] = (co, faces, {"j_kosi": np.full(len(co), 1.0)})
    return result


def add_weights(obj, weights, start=0):
    for name, values in weights.items():
        group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        for i, weight in enumerate(values.tolist()):
            if weight > 1e-4:
                group.add([start + i], weight, 'REPLACE')


def new_mesh(name, parts, rig):
    co, faces, starts = [], [], {}
    for part, (part_co, part_faces, _weights) in parts.items():
        starts[part] = (len(co), len(co) + len(part_co))
        faces.extend(tuple(len(co) + k for k in face) for face in part_faces)
        co.extend(part_co)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(co, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for part, (_co, _faces, weights) in parts.items():
        add_weights(obj, weights, starts[part][0])
    obj.modifiers.new("Armature", 'ARMATURE').object = rig
    return obj, starts


def build(separate=False):
    """The rig and the outfit, in Object Mode with nothing but the outfit selected. Returns (rig, {part:
    (object, (first vertex, end))}): one mesh for every part, or one each with ``separate``."""
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    rig = skirt.build_rig("Rig", chains={})
    parts = pieces()
    result = {}
    if separate:
        for part, data in parts.items():
            obj, starts = new_mesh(part.title(), {part: data}, rig)
            result[part] = (obj, starts[part])
    else:
        obj, starts = new_mesh("Outfit", parts, rig)
        result = {part: (obj, starts[part]) for part in parts}
    rig.select_set(False)
    for obj, _span in result.values():
        obj.select_set(True)
    bpy.context.view_layer.objects.active = result["skirt"][0]
    use_settings()
    return rig, result


def use_settings(**values):
    settings = bpy.context.scene.magic_fit
    for prop in ("connect_distance", "connect_blend", "connect_max_groups"):
        settings.property_unset(prop)
    settings.target = None
    for name, value in values.items():
        setattr(settings, name, value)
    return settings


def by_name(obj):
    """{group name: (vertices,) weights}."""
    table, names = skirt.weights(obj)
    return {name: table[:, k] for k, name in enumerate(names)}


def rows(weights, span, names=None):
    """The (vertices, groups) table of ``weights`` for the vertices in ``span``, in the order of ``names``."""
    names = sorted(weights) if names is None else names
    count = len(next(iter(weights.values())))
    zero = np.zeros(count)
    return np.stack([weights.get(name, zero)[span[0]:span[1]] for name in names], axis=1)


def same(a, b, names, tolerance):
    return float(np.abs(rows(a, (0, 10 ** 9), names) - rows(b, (0, 10 ** 9), names)).max()) <= tolerance


def at_closest(source, span, points):
    """``source``'s weights (by name) at its closest points to ``points``, on its triangles within ``span``."""
    obj = source
    mesh = obj.data
    mesh.calc_loop_triangles()
    co = skirt.rest_co(obj)
    tris = [tuple(tri.vertices) for tri in mesh.loop_triangles if span[0] <= tri.vertices[0] < span[1]]
    tree = BVHTree.FromPolygons(co.tolist(), tris, all_triangles=True)
    weights = by_name(obj)
    names = sorted(weights)
    result = np.zeros((len(points), len(names)))
    gaps = np.zeros(len(points))
    for k, point in enumerate(points):
        location, _normal, index, gap = tree.find_nearest(point)
        tri = tris[index]
        factors = poly_3d_calc([Vector(co[v]) for v in tri], location)
        for v, factor in zip(tri, factors):
            result[k] += factor * np.array([weights[name][v] for name in names])
        gaps[k] = gap
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12), names, gaps


def run():
    return bpy.ops.magic_fit.connect_parts()


# -----------------------------------------------------------------------------
# What it does

def test_parts_take_the_weights_of_what_they_lie_on():
    _rig, parts = build()
    obj = parts["skirt"][0]
    start = by_name(obj)
    check(run() == {'FINISHED'}, "button: finishes")
    after = by_name(obj)
    names = sorted(set(start) | set(after))
    co = skirt.rest_co(obj)

    span = parts["skirt"][1]
    check(float(np.abs(rows(after, span, names) - rows(start, span, names)).max()) == 0.0,
          "the biggest part keeps its weights")
    span = parts["far"][1]
    check(float(np.abs(rows(after, span, names) - rows(start, span, names)).max()) == 0.0,
          "a part farther than Distance isn't connected")

    span = parts["belt"][1]
    expected, expected_names, _gaps = at_closest(obj, parts["skirt"][1], co[span[0]:span[1]])
    worst = float(np.abs(rows(after, span, expected_names) - expected).max())
    check(worst < 2e-3, "the belt takes the skirt's weights at its closest points ({:.1e})".format(worst))
    totals = rows(after, span, names).sum(axis=1)
    check(np.allclose(totals, 1.0, atol=1e-4), "the belt's vertices keep their total weight")

    # The loop lies on the belt (3 mm), and past the belt's edges 6 mm off the skirt. Where it lies on the
    # belt, it ends up with the skirt's weights through it (a little off where the belt's closest point
    # isn't the loop's); past the edges, it carries them on, and none of its own are left.
    span = parts["loop"][1]
    loop_co = co[span[0]:span[1]]
    on_belt = (loop_co[:, 2] > 0.91) & (loop_co[:, 2] < 0.97)
    expected, expected_names, _gaps = at_closest(obj, parts["skirt"][1], loop_co)
    worst = float(np.abs(rows(after, span, expected_names)[on_belt] - expected[on_belt]).max())
    check(worst < 0.03, "a loop on the belt takes the skirt's weights through it ({:.3f})".format(worst))
    check(not after["j_sebo_a"][span[0]:span[1]].any() and np.allclose(rows(after, span).sum(axis=1), 1.0, atol=1e-4),
          "the loop's ends past the belt carry them on")

    # The strap lies on the skirt below its top, and reaches past it: there it keeps its own weights,
    # from Distance plus Blend beyond the skirt on, and blends in between.
    span = parts["strap"][1]
    strap_co = co[span[0]:span[1]]
    expected, expected_names, gaps = at_closest(obj, parts["skirt"][1], strap_co)
    table = rows(after, span, expected_names)
    settings = bpy.context.scene.magic_fit
    lying = gaps <= settings.connect_distance
    check(lying.sum() >= 6 and float(np.abs(table[lying] - expected[lying]).max()) < 2e-3,
          "the strap takes the skirt's weights where it lies on it")
    beyond = gaps > settings.connect_distance + settings.connect_blend
    sebo_b = rows(after, span, ["j_sebo_b"])[:, 0]
    check(beyond.sum() >= 6 and np.allclose(sebo_b[beyond], 1.0), "the strap keeps its own weights farther along")
    between = ~lying & ~beyond
    order = np.argsort(strap_co[:, 2])
    column = sebo_b[order][np.isin(order, np.flatnonzero(strap_co[:, 0] == strap_co[0, 0]))]
    check(between.sum() >= 3 and (sebo_b[between] > 0.02).all() and (sebo_b[between] < 0.98).all(),
          "the strap blends between them")
    check(bool(np.all(np.diff(column) >= -1e-6)), "the strap's own weights come back steadily along it")


def height_change(obj, outer, inner, posed):
    """How far each vertex of ``outer`` (a span of ``obj``) lies above ``inner`` along its normal in the rest
    pose, less how far it does posed (the most over the vertices), in metres."""
    rest = skirt.rest_co(obj)
    mesh = obj.data
    mesh.calc_loop_triangles()
    tris = np.array([tuple(tri.vertices) for tri in mesh.loop_triangles if inner[0] <= tri.vertices[0] < inner[1]])
    tree = BVHTree.FromPolygons(rest.tolist(), tris.tolist(), all_triangles=True)
    worst = 0.0
    for v in range(*outer):
        location, _normal, index, _gap = tree.find_nearest(rest[v])
        a, b, c = tris[index]
        factors = poly_3d_calc([Vector(rest[k]) for k in (a, b, c)], location)

        def height(co):
            normal = np.cross(co[b] - co[a], co[c] - co[a])
            normal /= np.linalg.norm(normal)
            return float(normal @ (co[v] - sum(f * co[k] for f, k in zip(factors, (a, b, c)))))

        worst = max(worst, abs(height(posed) - height(rest)))
    return worst


def test_parts_move_together():
    rig, parts = build()
    obj = parts["skirt"][0]
    skirt.scale_thighs(rig, 1.4)
    before = height_change(obj, parts["belt"][1], parts["skirt"][1], skirt.displayed(obj))
    skirt.scale_thighs(rig, 1.0)
    run()
    skirt.scale_thighs(rig, 1.4)
    after = height_change(obj, parts["belt"][1], parts["skirt"][1], skirt.displayed(obj))
    skirt.scale_thighs(rig, 1.0)
    # What's left is the 3 mm gap growing with the thighs.
    check(before > 0.002 and after < 0.0015 and after < 0.3 * before,
          "with thighs 40 % larger the belt stays on the skirt ({:.2f} mm, before {:.2f} mm)".format(
              after * 1000, before * 1000))


def test_again_changes_nothing():
    _rig, parts = build()
    obj = parts["skirt"][0]
    run()
    first = by_name(obj)
    check(run() == {'CANCELLED'}, "a second time: nothing to do")
    check(same(first, by_name(obj), sorted(first), 0.0), "a second time: the weights stay as they are")


def test_separate_meshes():
    _rig, parts = build(separate=True)
    body = parts["far"][0]
    settings = use_settings(target=body)
    belt = parts["belt"][0]
    belt_start = by_name(belt)
    skirt_start = by_name(parts["skirt"][0])
    body_start = by_name(body)
    check(run() == {'FINISHED'}, "separate meshes: finishes")
    after = by_name(belt)
    check({"j_asi_a_l", "j_asi_a_r"} <= set(after) and set(belt_start) == {"j_kosi"},
          "separate meshes: the belt gets the groups the skirt is weighted to")
    expected, names, _gaps = at_closest(parts["skirt"][0], (0, 10 ** 9), skirt.rest_co(belt))
    worst = float(np.abs(rows(after, (0, 10 ** 9), names) - expected).max())
    check(worst < 2e-3, "separate meshes: the belt takes the skirt's weights ({:.1e})".format(worst))
    check(same(skirt_start, by_name(parts["skirt"][0]), sorted(skirt_start), 0.0), "separate meshes: the skirt stays")
    check(same(body_start, by_name(body), sorted(body_start), 0.0), "separate meshes: the body is left out")
    strap = by_name(parts["strap"][0])
    check("j_kosi" in strap, "separate meshes: the strap connects too")
    settings.target = None


def test_locked_hidden_and_masked():
    # A locked group keeps its weight, and the others share the rest like the skirt's.
    _rig, parts = build()
    obj = parts["skirt"][0]
    span = parts["loop"][1]
    obj.vertex_groups["j_sebo_a"].lock_weight = True
    run()
    after = by_name(obj)
    check(np.allclose(after["j_sebo_a"][span[0]:span[1]], 0.3, atol=1e-5), "locked: the group keeps its weight")
    totals = rows(after, span).sum(axis=1)
    check(np.allclose(totals, 1.0, atol=1e-4), "locked: the vertices keep their total")
    check(float(after["j_kosi"][span[0]:span[1]].min()) < 0.69, "locked: the other groups change")

    # Hidden vertices are left alone.
    _rig, parts = build()
    obj = parts["skirt"][0]
    span = parts["belt"][1]
    start = by_name(obj)
    hidden = np.zeros(len(obj.data.vertices), dtype=bool)
    hidden[span[0]:(span[0] + span[1]) // 2] = True
    obj.data.vertices.foreach_set("hide", hidden)
    run()
    after = by_name(obj)
    names = sorted(set(start) | set(after))
    changed = np.abs(rows(after, (0, len(hidden)), names) - rows(start, (0, len(hidden)), names)).max(axis=1) > 0.0
    check(changed[span[0]:span[1]].any() and not changed[hidden].any(), "hidden vertices are left alone")

    # Weight Paint mode with vertex selection masking: only the selected vertices change.
    _rig, parts = build()
    obj = parts["skirt"][0]
    span = parts["belt"][1]
    start = by_name(obj)
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    mesh = obj.data
    select = np.zeros(len(mesh.vertices), dtype=bool)
    select[span[0]:(span[0] + span[1]) // 2] = True
    mesh.vertices.foreach_set("select", select)
    mesh.use_paint_mask_vertex = True
    check(bpy.ops.magic_fit.connect_parts.poll(), "Weight Paint mode: ready")
    run()
    after = by_name(obj)
    changed = np.abs(rows(after, (0, len(select)), names) - rows(start, (0, len(select)), names)).max(axis=1) > 0.0
    check(changed.any() and not changed[~select].any(), "Weight Paint mode: selection masking is respected")
    bpy.ops.object.mode_set(mode='OBJECT')


def test_unweighted_parts_are_filled_in():
    _rig, parts = build()
    obj = parts["skirt"][0]
    span = parts["strap"][1]
    for group in obj.vertex_groups:
        group.remove(list(range(*span)))
    run()
    after = by_name(obj)
    table = rows(after, span)
    check(np.allclose(table.sum(axis=1), 1.0, atol=1e-4), "an unweighted strap gets weights all along")
    co = skirt.rest_co(obj)[span[0]:span[1]]
    top = co[:, 2] > SKIRT_TOP + 0.05
    expected, names, _gaps = at_closest(obj, parts["skirt"][1], co)
    worst = float(np.abs(rows(after, span, names)[top] - expected[top]).max())
    check(worst < 0.02, "far from the skirt, it takes what the skirt has where it's sewn on ({:.3f})".format(worst))


def test_settings_change_what_connects():
    # Blend 0: the strap keeps its own weights right past Distance.
    _rig, parts = build()
    obj = parts["skirt"][0]
    use_settings(connect_blend=0.0)
    run()
    span = parts["strap"][1]
    after = by_name(obj)
    _expected, _names, gaps = at_closest(obj, parts["skirt"][1], skirt.rest_co(obj)[span[0]:span[1]])
    sebo_b = after["j_sebo_b"][span[0]:span[1]]
    check(np.allclose(sebo_b[gaps > 0.0051], 1.0) and np.allclose(sebo_b[gaps < 0.0049], 0.0),
          "Blend 0: a sharp change at Distance")

    # A Distance shorter than the gaps: the belt and the loop aren't connected, the strap is.
    _rig, parts = build()
    obj = parts["skirt"][0]
    start = by_name(obj)
    use_settings(connect_distance=0.002)
    run()
    after = by_name(obj)
    names = sorted(set(start) | set(after))
    for part in ("belt", "loop"):
        span = parts[part][1]
        check(float(np.abs(rows(after, span, names) - rows(start, span, names)).max()) == 0.0,
              "Distance 2 mm: the {:s} 3 mm off isn't connected".format(part))
    span = parts["strap"][1]
    check(float(np.abs(rows(after, span, names) - rows(start, span, names)).max()) > 0.5,
          "Distance 2 mm: the strap 1 mm off is")

    # The group limit holds.
    _rig, parts = build()
    obj = parts["skirt"][0]
    use_settings(connect_max_groups=2)
    run()
    used = sum((values > 0.0).astype(int) for values in by_name(obj).values())
    for part in ("belt", "loop", "strap"):
        span = parts[part][1]
        check(int(used[span[0]:span[1]].max()) <= 2, "Max Groups 2: the {:s} follows at most 2 bones".format(part))


def test_readiness():
    _rig, parts = build()
    obj = parts["skirt"][0]
    check(operators.check_connect_ready(bpy.context) is None, "ready in Object Mode")
    bpy.ops.object.mode_set(mode='EDIT')
    check(operators.check_connect_ready(bpy.context) == "Connect parts in Object or Weight Paint mode",
          "refused in Edit Mode")
    bpy.ops.object.mode_set(mode='OBJECT')
    use_settings(target=obj)
    for other in bpy.context.selected_objects:
        other.select_set(other == obj)
    check(operators.check_connect_ready(bpy.context) == "Select meshes other than the body", "the body alone is refused")
    use_settings()
    obj.vertex_groups.clear()
    check(operators.check_connect_ready(bpy.context) == "No weights to connect",
          "a mesh without weights is refused")
    obj.select_set(False)
    bpy.context.view_layer.objects.active = None
    check(operators.check_connect_ready(bpy.context) == "Select the meshes to connect",
          "says to select meshes")
    check(not bpy.ops.magic_fit.connect_parts.poll(), "the button is off then")


def test_defaults():
    settings = bpy.context.scene.magic_fit
    rna = settings.bl_rna.properties
    expected = {"connect_distance": 0.005, "connect_blend": 0.02, "connect_max_groups": 8}
    check(all(abs(rna[name].default - value) < 1e-6 for name, value in expected.items()),
          "defaults: the settings start at the chosen values")
    import inspect
    engine = inspect.signature(connecting.connect).parameters
    check(all(abs(engine[name[8:]].default - expected[name]) < 1e-6 for name in ("connect_distance", "connect_blend")),
          "defaults: the engine's match the settings'")
    check(inspect.signature(connecting.ConnectStroke.__init__).parameters["max_groups"].default == 8,
          "defaults: the group limit matches")


def test_registered():
    check(hasattr(bpy.ops.magic_fit, "connect_parts"), "the Connect Parts operator is registered")


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
