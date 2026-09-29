"""Headless tests for the Weight Brushes' Smooth mode.

Run from the repository root:
    blender -b --factory-startup --python tests/test_smooth_core.py
"""

import inspect
import os
import sys
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "deps"))  # SciPy, for the direct solver

import smooth  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import operators, properties, smoothing  # noqa: E402

FAILURES = []
GAP = 0.0005
SEAM_A = 0.1           # the last column of the first part
SEAM_B = 0.1 + GAP     # the first column of the second


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def new_stroke(obj, **kwargs):
    return smoothing.SmoothStroke(obj, smooth.rest_co(obj), **kwargs)


def difference(a, b):
    return max(float(np.abs(a[name] - b[name]).max()) for name in a)


def use_settings(**values):
    settings = bpy.context.scene.magic_fit
    for prop in ("smooth_reach", "smooth_merge_range", "smooth_max_groups", "use_accumulate", "group_mode"):
        settings.property_unset(prop)
    settings.target = None
    settings.mode = 'SMOOTH'
    for name, value in values.items():
        setattr(settings, name, value)
    return settings


def profile(sheet, weights, row=0.05):
    """(x, weight) along the sheet's row at height ``row``, by x."""
    co = smooth.rest_co(sheet)
    rows = np.flatnonzero(np.abs(co[:, 1] - row) < 1e-6)
    order = np.argsort(co[rows, 0])
    return co[rows[order], 0], weights[rows[order]]


def at(x, samples, offset):
    """A profile's value ``offset`` from the seam (the seam is at SEAM_A and SEAM_B)."""
    seam = SEAM_A if offset <= 0 else SEAM_B
    return float(np.interp(seam + offset, *samples))


# -----------------------------------------------------------------------------
# The graph

def test_graph():
    _rig, sheet, _owner = smooth.build()
    stroke = new_stroke(sheet, merge_range=0.002)
    graph = stroke.graph
    check(graph.count == len(sheet.data.vertices) and int(graph.parts.max()) + 1 == 2, "graph: two loose parts")
    check(graph.links == 11, "graph: one link per row of the seam, {:d}".format(graph.links))
    rows = np.repeat(np.arange(graph.count), np.diff(graph.pointers))
    across = graph.parts[rows] != graph.parts[graph.neighbours]
    check(int(across.sum()) == 2 * graph.links, "graph: only the links join different parts")
    check(bool((np.abs(smooth.rest_co(sheet)[stroke.weld[0], 0] - stroke.rest[0, 0]) < 1e-9)), "graph: nodes are at the vertices")

    for merge, expected in ((0.0, 0), (0.0002, 0), (0.001, 11)):
        found = new_stroke(sheet, merge_range=merge).graph.links
        check(found == expected, "merge range {:.4f}: {:d} links (expected {:d})".format(merge, found, expected))

    wide = new_stroke(sheet, merge_range=0.05).graph
    rows = np.repeat(np.arange(wide.count), np.diff(wide.pointers))
    across = wide.parts[rows] != wide.parts[wide.neighbours]
    per_node = np.bincount(rows[across], minlength=wide.count)
    check(int(per_node.max()) <= smoothing.MAX_LINKS, "graph: a node links to at most {:d} others".format(smoothing.MAX_LINKS))
    check(int(across.sum()) == 2 * wide.links and wide.links > 11, "graph: a wide range still only links different parts")


def test_graph_cache():
    _rig, sheet, _owner = smooth.build()
    first = new_stroke(sheet, merge_range=0.002).graph
    check(new_stroke(sheet, merge_range=0.002).graph is first, "cache: the next stroke reuses the graph")
    check(new_stroke(sheet, merge_range=0.001).graph is not first, "cache: another merge range builds another")
    _rig, sheet, _owner = smooth.build()
    check(new_stroke(sheet, merge_range=0.001).graph is not first, "cache: another mesh builds another")


def test_close_pairs():
    rng = np.random.default_rng(3)
    points = rng.random((400, 3)) * 0.05
    a, b = smoothing.close_pairs(points, 0.006)
    delta = points[:, None, :] - points[None, :, :]
    near = np.linalg.norm(delta, axis=2) <= 0.006
    expected = {(i, j) for i in range(400) for j in range(i + 1, 400) if near[i, j]}
    check(set(zip(a.tolist(), b.tolist())) == expected and bool((a < b).all()),
          "close pairs: exactly the pairs within range ({:d})".format(len(expected)))
    check(len(smoothing.close_pairs(points[:1], 0.1)[0]) == 0 and len(smoothing.close_pairs(points, 0.0)[0]) == 0,
          "close pairs: none for one point or no range")


# -----------------------------------------------------------------------------
# Smoothing across gaps

def test_gap_needs_merge_range():
    _rig, sheet, owner = smooth.build()
    start = smooth.by_name(sheet)
    stroke = new_stroke(sheet, merge_range=0.0)
    smooth.apply_all(stroke, sheet)
    check(difference(smooth.by_name(sheet), start) < 1e-6 and not stroke.modified,
          "no merge range: parts are smoothed on their own, so hard steps between them stay")


def test_smooth_across_gap():
    _rig, sheet, owner = smooth.build()
    stroke = new_stroke(sheet, merge_range=0.002, reach=0.02)
    smooth.apply_all(stroke, sheet)
    after = smooth.by_name(sheet)
    a = profile(sheet, after["j_a"])
    check(bool(np.all(np.diff(a[1]) <= 1e-6)), "across the gap: j_a falls steadily from one part to the other")
    left, right = float(np.interp(SEAM_A, *a)), float(np.interp(SEAM_B, *a))
    check(0.4 < left < 0.6 and 0.4 < right < 0.6 and abs(left - right) < 0.05,
          "across the gap: parts that touch are smoothed as if merged, the seam is even ({:.2f} | {:.2f})".format(left, right))
    check(abs(left + right - 1.0) < 0.03, "across the gap: the sides mirror each other ({:.3f})".format(left + right))
    check(float(a[1][0]) > 0.99 and float(a[1][-1]) < 0.01,
          "across the gap: far from the seam the weights are as they were")
    total = after["j_a"] + after["j_b"] + after["j_c"]
    check(float(np.abs(total - 1.0).max()) < 1e-4, "across the gap: totals kept")
    check(not after["j_c"].any(), "across the gap: groups nothing has aren't invented")


def test_welded_split_seam():
    _rig, sheet, _owner = smooth.build(parts=((0.0, 0.1), (0.1, 0.2)))
    stroke = new_stroke(sheet, merge_range=0.0)
    check(stroke.weld_count == len(sheet.data.vertices) - 11 and stroke.graph.links == 0,
          "split seam: the copies are one node, no links needed")
    smooth.apply_all(stroke, sheet)
    after = smooth.by_name(sheet)
    co = smooth.rest_co(sheet)
    seam = np.flatnonzero(np.abs(co[:, 0] - 0.1) < 1e-6)
    check(len(seam) == 22, "split seam: two copies per row")
    keys = {}
    for vert in seam.tolist():
        keys.setdefault(round(float(co[vert, 1]), 6), []).append(vert)
    same = all(np.array_equal(after["j_a"][pair[0]], after["j_a"][pair[1]]) for pair in keys.values())
    check(same and all(0.3 < after["j_a"][pair[0]] < 0.7 for pair in keys.values()),
          "split seam: both copies get the same, blended weights")


def test_all_groups_at_once():
    parts = ((0.0, 0.1), (0.1005, 0.2005), (0.2010, 0.3010))

    def weights(part, co):
        return {smooth.BONES[part]: np.ones(len(co))}

    _rig, sheet, _owner = smooth.build(parts=parts, weights=weights)
    stroke = new_stroke(sheet)
    smooth.apply_all(stroke, sheet)
    after = smooth.by_name(sheet)
    co = smooth.rest_co(sheet)
    middle = np.abs(co[:, 1] - 0.05) < 1e-6
    b = profile(sheet, after["j_b"])
    check(float(np.interp(0.1005, *b)) < 0.6 and float(np.interp(0.2005, *b)) < 0.6 and float(np.interp(0.15, *b)) > 0.9,
          "all groups: the middle part is blended on both sides in one stroke")
    check(bool(after["j_a"][middle].max() > 0.9 and after["j_c"][middle].max() > 0.9 and after["j_c"].min() == 0.0),
          "all groups: the outer parts keep their own")
    check(float(np.abs(after["j_a"] + after["j_b"] + after["j_c"] - 1.0).max()) < 1e-4, "all groups: totals kept")
    check(after["j_a"][co[:, 0] > 0.2].max() < 0.05 and after["j_c"][co[:, 0] < 0.1].max() < 0.01,
          "all groups: j_a doesn't reach across the middle part, j_c neither")


def test_totals_kept():
    def weights(part, co):
        return {"j_a" if part == 0 else "j_b": np.full(len(co), 0.5 if part == 0 else 0.25)}

    _rig, sheet, owner = smooth.build(weights=weights)
    start = smooth.by_name(sheet)
    stroke = new_stroke(sheet)
    smooth.apply_all(stroke, sheet)
    after = smooth.by_name(sheet)
    total = after["j_a"] + after["j_b"]
    expected = start["j_a"] + start["j_b"]
    check(float(np.abs(total - expected).max()) < 1e-4 and stroke.modified,
          "totals: a vertex with 0.5 or 0.25 keeps that, blended between the two bones")


def test_unweighted_filled():
    def weights(part, co):
        if part == 1:
            return {}
        return {"j_a": np.where((co[:, 0] > 0.045) & (co[:, 0] < 0.075), 0.0, 1.0)}

    _rig, sheet, owner = smooth.build(weights=weights)
    co = smooth.rest_co(sheet)
    start = smooth.by_name(sheet)
    band = (owner == 0) & (co[:, 0] > 0.045) & (co[:, 0] < 0.075)
    check(band.any() and float(start["j_a"][band].max()) == 0.0 and float(start["j_b"].max()) == 0.0, "unweighted: the setup has holes")
    stroke = new_stroke(sheet)
    smooth.apply_all(stroke, sheet)
    after = smooth.by_name(sheet)
    check(float(np.abs(after["j_a"][band] - 1.0).max()) < 1e-4, "unweighted: vertices without weights take on those around them")
    second = owner == 1
    check(float(after["j_a"][second].max()) > 0.0,
          "unweighted: a part without weights takes on those of the part it touches")


# -----------------------------------------------------------------------------
# Groups

def test_locked_groups():
    def weights(part, co):
        return {"j_a" if part == 0 else "j_b": np.full(len(co), 0.7), "j_c": np.full(len(co), 0.3)}

    _rig, sheet, _owner = smooth.build(weights=weights)
    sheet.vertex_groups["j_c"].lock_weight = True
    start = smooth.by_name(sheet)
    stroke = new_stroke(sheet)
    smooth.apply_all(stroke, sheet)
    after = smooth.by_name(sheet)
    check(np.array_equal(after["j_c"], start["j_c"]), "locked: the locked group isn't changed")
    check(float(np.abs(after["j_a"] + after["j_b"] - 0.7).max()) < 1e-4, "locked: the others share what it leaves")
    a = profile(sheet, after["j_a"])
    check(0.15 < float(np.interp(SEAM_A, *a)) < 0.65 and stroke.modified, "locked: the rest is smoothed as usual")

    for group in sheet.vertex_groups:
        group.lock_weight = True
    stroke = new_stroke(sheet)
    smooth.apply_all(stroke, sheet)
    check(not stroke.modified, "locked: nothing changes when every group is locked")


def test_active_group():
    _rig, sheet, _owner = smooth.build()
    everything = new_stroke(sheet)
    smooth.apply_all(everything, sheet)
    reference = smooth.by_name(sheet)
    everything.restore()
    sheet.data.update()

    sheet.vertex_groups.active_index = sheet.vertex_groups["j_a"].index
    stroke = new_stroke(sheet, group_mode='ACTIVE')
    smooth.apply_all(stroke, sheet)
    after = smooth.by_name(sheet)
    check(float(np.abs(after["j_a"] - reference["j_a"]).max()) < 1e-3,
          "current group: the active group gets what it gets in All Groups mode")
    check(float(np.abs(after["j_a"] + after["j_b"] - 1.0).max()) < 1e-4 and not after["j_c"].any(),
          "current group: the other bones make room for it")

    stroke.restore()
    sheet.vertex_groups.active_index = sheet.vertex_groups["j_c"].index
    stroke = new_stroke(sheet, group_mode='ACTIVE')
    smooth.apply_all(stroke, sheet)
    check(not stroke.modified, "current group: a group nothing has stays where it is")


def test_group_limit():
    def weights(part, co):
        if part == 0:
            return {name: np.full(len(co), value) for name, value in (("j_a", 0.4), ("j_b", 0.3), ("j_c", 0.2),
                                                                     ("j_d", 0.06), ("j_e", 0.04))}
        return {"j_e": np.ones(len(co))}

    _rig, sheet, _owner = smooth.build(weights=weights, armature=False, groups=smooth.BONES + ("j_d", "j_e"))
    start_total = sum(smooth.by_name(sheet).values())
    stroke = new_stroke(sheet, max_groups=3)
    smooth.apply_all(stroke, sheet)
    after = smooth.by_name(sheet)
    counts = sum((values > 1e-4).astype(int) for values in after.values())
    check(int(counts.max()) <= 3, "group limit: at most 3 weights per vertex ({:d})".format(int(counts.max())))
    check(float(np.abs(sum(after.values()) - start_total).max()) < 1e-4, "group limit: the total is kept")
    dust = [values[(values > 1e-4) & (values < 0.0015)] for values in after.values()]
    check(all(not len(each) for each in dust), "group limit: no weight smaller than a byte's worth is left")


def test_no_armature():
    _rig, sheet, _owner = smooth.build(armature=False, groups=smooth.BONES + ("Mask",))
    group = sheet.vertex_groups["Mask"]
    group.add(list(range(len(sheet.data.vertices))), 0.5, 'REPLACE')
    modifier = sheet.modifiers.new("Displace", 'DISPLACE')
    modifier.vertex_group = "Mask"
    smoothed = smoothing.smoothed_groups(sheet)
    check(smoothed == {sheet.vertex_groups[name].index for name in smooth.BONES},
          "no armature: every group counts but the ones a modifier uses")
    start = smooth.by_name(sheet)
    stroke = new_stroke(sheet)
    smooth.apply_all(stroke, sheet)
    after = smooth.by_name(sheet)
    check(np.array_equal(after["Mask"], start["Mask"]) and stroke.modified and
          float(np.abs(after["j_a"] + after["j_b"] - 1.0).max()) < 1e-4,
          "no armature: the mask is left alone, the rest is smoothed")


# -----------------------------------------------------------------------------
# The brush

def test_solvers_agree():
    if smoothing._direct_solver() is None:
        print("skip: SciPy isn't there, so there's no second solver to compare (put it in deps/)")
        return
    results = {}
    for use_scipy in (True, False):
        smoothing.USE_SCIPY = use_scipy
        try:
            for edge in (0.01, 0.0025):
                _rig, sheet, _owner = smooth.build(edge=edge)
                stroke = new_stroke(sheet, reach=0.02)
                smooth.apply_all(stroke, sheet)
                results[(use_scipy, edge)] = smooth.by_name(sheet)
        finally:
            smoothing.USE_SCIPY = True
    for edge in (0.01, 0.0025):
        error = difference(results[(True, edge)], results[(False, edge)])
        check(error < 0.005, "solvers: SciPy's and the conjugate gradients agree on {:.2f} cm edges ({:.5f})".format(edge * 100, error))


def test_reach_is_a_distance():
    profiles = []
    for edge in (0.01, 0.005, 0.0025):
        _rig, sheet, _owner = smooth.build(edge=edge)
        stroke = new_stroke(sheet, reach=0.02)
        smooth.apply_all(stroke, sheet)
        samples = profile(sheet, smooth.by_name(sheet)["j_a"])
        profiles.append([at(0.0, samples, offset) for offset in (-0.03, -0.02, -0.01, 0.01, 0.02, 0.03)])
    spread = np.ptp(np.array(profiles), axis=0).max()
    check(spread < 0.02, "reach: the same profile on meshes of 1, 0.5 and 0.25 cm edges (spread {:.3f})".format(spread))


def test_reach_widens():
    values = []
    for reach in (0.01, 0.02, 0.04):
        _rig, sheet, _owner = smooth.build()
        stroke = new_stroke(sheet, reach=reach)
        smooth.apply_all(stroke, sheet)
        values.append(at(0.0, profile(sheet, smooth.by_name(sheet)["j_a"]), 0.03))
    check(values[0] < values[1] < values[2] and values[0] < 0.05 and values[2] > 0.1,
          "reach: the larger it is, the farther the weights spread ({:s})".format(", ".join("{:.3f}".format(v) for v in values)))


def test_strength():
    _rig, sheet, _owner = smooth.build()
    start = smooth.by_name(sheet)
    stroke = new_stroke(sheet)
    smooth.apply_all(stroke, sheet, 1.0)
    full = smooth.by_name(sheet)
    stroke.restore()
    sheet.data.update()

    stroke = new_stroke(sheet)
    smooth.apply_all(stroke, sheet, 0.5)
    half = smooth.by_name(sheet)
    expected = {name: start[name] + 0.5 * (full[name] - start[name]) for name in start}
    check(difference(half, expected) < 0.02, "strength: 0.5 goes half of the way to the smoothed weights")
    smooth.apply_all(stroke, sheet, 0.5)
    check(difference(smooth.by_name(sheet), half) < 1e-6,
          "strength: passing over the same spot again doesn't go further, within a stroke")
    smooth.apply_all(stroke, sheet, 0.8)
    expected = {name: start[name] + 0.8 * (full[name] - start[name]) for name in start}
    check(difference(smooth.by_name(sheet), expected) < 0.02, "strength: a stronger dab makes up the difference")

    again = new_stroke(sheet)
    before = smooth.by_name(sheet)
    smooth.apply_all(again, sheet, 1.0)
    check(again.modified and difference(smooth.by_name(sheet), before) > 0.01,
          "strength: the next stroke smooths what the last left")


def test_accumulate():
    _rig, sheet, _owner = smooth.build()
    stroke = new_stroke(sheet, accumulate=True)
    smooth.apply_all(stroke, sheet, 0.5)
    once = smooth.by_name(sheet)
    smooth.apply_all(stroke, sheet, 0.5)
    twice = smooth.by_name(sheet)
    check(difference(twice, once) > 0.01, "accumulate: every dab smooths further")
    check(float(twice["j_a"].min()) >= 0.0 and float(np.abs(twice["j_a"] + twice["j_b"] - 1.0).max()) < 1e-4,
          "accumulate: weights stay valid")
    ranges = []
    for _ in range(12):
        smooth.apply_all(stroke, sheet, 1.0)
        a = profile(sheet, smooth.by_name(sheet)["j_a"])[1]
        ranges.append(float(a.max() - a.min()))
        if float(np.diff(a).max()) > 1e-4:
            ranges.append(float("inf"))
    check(all(later < earlier for earlier, later in zip(ranges, ranges[1:])) and ranges[-1] < 0.5 * ranges[0],
          "accumulate: keeps evening out the profile, without ever overshooting ({:.2f} -> {:.2f})".format(ranges[0], ranges[-1]))


def test_path_independence():
    _rig, sheet, _owner = smooth.build()
    co = smooth.rest_co(sheet)
    stroke = new_stroke(sheet)
    smooth.apply_all(stroke, sheet)
    whole = smooth.by_name(sheet)
    stroke.restore()
    sheet.data.update()

    stroke = new_stroke(sheet)
    for low, high in ((0.09, 0.12), (0.05, 0.09), (0.12, 0.16), (0.0, 0.06), (0.14, 0.21), (0.0, 0.21)):
        rows = np.flatnonzero((co[:, 0] >= low) & (co[:, 0] <= high)).tolist()
        stroke.apply(rows, [1.0] * len(rows))
    sheet.data.update()
    pieces = smooth.by_name(sheet)
    error = difference(pieces, whole)
    check(error < 0.04, "path: dabs in any order and size give the same weights as one ({:.4f})".format(error))


def test_split_dabs_touch_only_their_vertices():
    _rig, sheet, owner = smooth.build()
    co = smooth.rest_co(sheet)
    start = smooth.by_name(sheet)
    stroke = new_stroke(sheet)
    rows = np.flatnonzero((co[:, 0] > 0.09) & (co[:, 0] < 0.11)).tolist()
    stroke.apply(rows, [1.0] * len(rows))
    sheet.data.update()
    after = smooth.by_name(sheet)
    changed = np.flatnonzero(np.abs(after["j_a"] - start["j_a"]) > 1e-6)
    check(len(changed) > 0 and set(changed.tolist()) <= set(rows), "dab: only the vertices under it change")


def test_restore():
    _rig, sheet, _owner = smooth.build()
    start = smooth.by_name(sheet)
    stroke = new_stroke(sheet)
    smooth.apply_all(stroke, sheet)
    check(stroke.modified and difference(smooth.by_name(sheet), start) > 0.1, "restore: the stroke changed weights")
    stroke.restore()
    sheet.data.update()
    check(difference(smooth.by_name(sheet), start) < 1e-6, "restore: cancelling puts the weights back")
    smooth.apply_all(stroke, sheet)
    check(difference(smooth.by_name(sheet), start) > 0.1, "restore: and the stroke can go on")


def test_rest_pose():
    rig, sheet, _owner = smooth.build()
    settings = use_settings()
    calm = smoothing.build_stroke(bpy.context, sheet, settings)
    rest = calm.rest.copy()
    smooth.apply_all(calm, sheet)
    reference = smooth.by_name(sheet)
    calm.restore()
    sheet.data.update()

    bone = rig.pose.bones["j_a"]
    bone.rotation_mode = 'XYZ'
    bone.rotation_euler = (1.2, 0.0, 0.4)
    bpy.context.view_layer.update()
    posed = smoothing.build_stroke(bpy.context, sheet, settings)
    check(np.allclose(posed.rest, rest, atol=1e-6) and rig.data.pose_position == 'POSE',
          "rest pose: distances are measured before posing, and the pose comes back")
    smooth.apply_all(posed, sheet)
    check(difference(smooth.by_name(sheet), reference) < 1e-6, "rest pose: posing doesn't change the result")


def test_odd_meshes():
    _rig, sheet, _owner = smooth.build()
    mesh = sheet.data
    count = len(mesh.vertices)
    mesh.vertices.add(3)  # a loose vertex, and two more under a triangle without area
    mesh.vertices[count].co = (0.5, 0.5, 0.0)
    mesh.vertices[count + 1].co = (0.6, 0.5, 0.0)
    mesh.vertices[count + 2].co = (0.7, 0.5, 0.0)
    mesh.edges.add(1)
    mesh.edges[len(mesh.edges) - 1].vertices = (count + 1, count + 2)
    mesh.update()
    stroke = new_stroke(sheet, merge_range=0.002)
    total = len(mesh.vertices)
    stroke.apply(list(range(total)), [1.0] * total)
    after = smooth.by_name(sheet)
    check(stroke.modified and float(np.abs(after["j_a"][count:]).max()) == 0.0,
          "odd meshes: loose vertices and wires are fine, and have no weights to give")

    empty = bpy.data.objects.new("Empty", bpy.data.meshes.new("Empty"))
    bpy.context.scene.collection.objects.link(empty)
    empty.vertex_groups.new(name="j_a")
    check(smoothing.SmoothStroke(empty, np.zeros((0, 3))).weld_count == 0, "odd meshes: an empty mesh is fine")


# -----------------------------------------------------------------------------
# Readiness and settings

def test_readiness():
    _rig, sheet, _owner = smooth.build()
    settings = use_settings()
    check(operators.check_ready(bpy.context, settings) is None, "ready: Smooth needs no body")
    settings.group_mode = 'ACTIVE'
    sheet.vertex_groups.active_index = sheet.vertex_groups["j_a"].index
    check(operators.check_ready(bpy.context, settings) is None, "ready: with an active group")
    sheet.vertex_groups["j_a"].lock_weight = True
    check("locked" in operators.check_ready(bpy.context, settings), "ready: the active group is locked")
    sheet.vertex_groups["j_a"].lock_weight = False
    settings.group_mode = 'ALL'
    for group in sheet.vertex_groups:
        group.lock_weight = True
    check("locked" in operators.check_ready(bpy.context, settings), "ready: every group is locked")
    for group in sheet.vertex_groups:
        group.lock_weight = False
        group.name = "other " + group.name
    check("No vertex group matches a bone" in operators.check_ready(bpy.context, settings), "ready: the groups match no bone")
    settings.group_mode = 'ACTIVE'
    check(operators.check_ready(bpy.context, settings) is not None, "ready: the active group isn't a bone")
    sheet.vertex_groups.clear()
    settings.group_mode = 'ALL'
    check(operators.check_ready(bpy.context, settings) is not None, "ready: no groups at all")
    try:
        smoothing.build_stroke(bpy.context, sheet, settings)
        raised = False
    except smoothing.SmoothError as error:
        raised = "no vertex groups" in str(error)
    check(raised, "build: says why it can't smooth")


def test_registered():
    settings = bpy.context.scene.magic_fit
    modes = [item.identifier for item in settings.bl_rna.properties["mode"].enum_items]
    general = [item.identifier for item in settings.bl_rna.properties["general_mode"].enum_items]
    check(modes[-1] == 'SMOOTH' and general == ['COPY', 'STRAIGHTEN', 'SMOOTH'],
          "modes: Smooth comes last, after Face (files store their positions)")
    settings.mode = 'COPY'
    settings.mode = 'SMOOTH'
    check(settings.weights_tab == 'GENERAL' and settings.general_mode == 'SMOOTH',
          "modes: Smooth is on the General sub-tab, which brings it back")
    settings.weights_tab = 'HEELS'
    settings.weights_tab = 'GENERAL'
    check(settings.mode == 'SMOOTH', "modes: General brings Smooth back after another sub-tab")
    check('SMOOTH' in properties.GENERAL_MODES, "modes: it counts as a General mode")
    engine = inspect.signature(smoothing.SmoothStroke.__init__).parameters
    check(abs(engine["reach"].default - settings.bl_rna.properties["smooth_reach"].default) < 1e-9 and
          abs(engine["merge_range"].default - settings.bl_rna.properties["smooth_merge_range"].default) < 1e-9 and
          engine["max_groups"].default == settings.bl_rna.properties["smooth_max_groups"].default,
          "defaults: the engine's match the settings'")
    settings.mode = 'COPY'


def main():
    magic_fit.register()
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    # Everything runs with SciPy's direct solver, and again with the conjugate gradients that are used
    # where SciPy isn't there.
    for use_scipy in (True, False):
        smoothing.USE_SCIPY = use_scipy
        print("== " + ("solving with SciPy (when it's there)" if use_scipy else "solving with conjugate gradients"))
        for test in tests:
            try:
                test()
            except Exception:
                FAILURES.append(test.__name__ + " raised")
                traceback.print_exc()
            finally:
                if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
    smoothing.USE_SCIPY = True
    print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
    sys.exit(1 if FAILURES else 0)


main()
