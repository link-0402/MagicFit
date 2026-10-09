"""Headless tests for Close Rips (`magic_fit.mending`, and the Find Rips and Close Rips buttons on the Mesh
Repair tab), with rips torn by Blender's own Move with Connected Only.

Run from the repository root:
    blender -b --factory-startup --python tests/test_rips_core.py
"""

import math
import os
import sys
import traceback

import bmesh
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import magic_fit  # noqa: E402
from magic_fit import mending  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


STEP = 0.004
MAX_GAP = 0.01
SPREAD = 0.03


def grid(columns, rows, place):
    """A grid of quads with ``columns`` x ``rows`` vertices at ``place(i, j)``: (positions, quads, index)."""
    co = np.array([place(i, j) for i in range(columns) for j in range(rows)], dtype=np.float64)
    index = np.arange(len(co)).reshape(columns, rows)
    quads = np.stack((index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(),
                      index[:-1, 1:].ravel()), axis=1)
    return co, quads, index


def join(*pieces):
    """Several (positions, quads) as one mesh, the vertices of each after the last's."""
    co, quads, first = [], [], 0
    for piece_co, piece_quads in pieces:
        co.append(piece_co)
        quads.append(piece_quads + first)
        first += len(piece_co)
    return np.concatenate(co), np.concatenate(quads)


def tris_of(quads):
    return np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))


def split_sheet():
    """A sheet in the XY plane, 10 x 6 cm, split down x = 0 like a UV seam: (positions, quads, the seam's
    vertex pairs (left copy, right copy))."""
    left_co, left_quads, left = grid(14, 16, lambda i, j: ((i - 13) * STEP, j * STEP, 0.0))
    right_co, right_quads, right = grid(14, 16, lambda i, j: (i * STEP, j * STEP, 0.0))
    co, quads = join((left_co, left_quads), (right_co, right_quads))
    return co, quads, np.stack((left[-1], right[0] + len(left_co)), axis=1)


def mesh_object(co, quads, name="Ripped"):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(co.tolist(), [], quads.tolist())
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def blender_move(co, quads, selected, translation, size):
    """Blender's Move with proportional editing, Connected Only: the positions afterwards."""
    obj = mesh_object(co, quads)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.select_mode = {'VERT'}
    for vert in bm.verts:
        vert.select = bool(selected[vert.index])
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.transform.translate(value=translation, use_proportional_edit=True, use_proportional_connected=True,
                                proportional_size=size, proportional_edit_falloff='SMOOTH')
    bpy.ops.object.mode_set(mode='OBJECT')
    return np.array([vert.co[:] for vert in obj.data.vertices])


def nearest(co, point):
    return int(np.argmin(np.linalg.norm(co - np.array(point), axis=1)))


def torn_sheet():
    """The split sheet with its seam torn open by moving the left side near it 5 mm away and down."""
    co, quads, seam = split_sheet()
    selected = np.zeros(len(co), dtype=bool)
    selected[nearest(co, (-2 * STEP, 0.03, 0.0))] = True
    ripped = blender_move(co, quads, selected, (-0.004, 0.0, -0.003), 0.025)
    return co, quads, seam, ripped


def gaps(co, pairs):
    return np.linalg.norm(co[pairs[:, 0]] - co[pairs[:, 1]], axis=1)


def everything(co):
    return np.ones(len(co), dtype=bool)


# -----------------------------------------------------------------------------
# Finding and closing

def test_torn_seam_closes():
    co, quads, seam, ripped = torn_sheet()
    torn = gaps(ripped, seam) > 1e-4
    check(torn.sum() >= 5 and gaps(ripped, seam).max() > 0.003,
          "Connected Only tears the split seam open ({:d} vertices, up to {:.1f} mm)".format(
              int(torn.sum()), gaps(ripped, seam).max() * 1000))
    rips = mending.find(ripped, tris_of(quads), everything(co), MAX_GAP)
    check(len(rips.pairs) >= torn.sum() - 2,
          "its vertices are paired one to one ({:d} pairs for {:d} torn)".format(len(rips.pairs), int(torn.sum())))
    nodes = rips.node
    right_pairs = {tuple(sorted(pair)) for pair in nodes[seam].tolist()}
    found = {tuple(sorted(pair)) for pair in rips.pairs.tolist()}
    check(found <= right_pairs, "each with its own copy ({:d} of {:d} right)".format(len(found & right_pairs),
                                                                                    len(found)))
    closed = mending.close(rips, everything(co), SPREAD)
    check(gaps(closed, seam).max() < 1e-5, "Close Rips closes it ({:.1e} m left)".format(gaps(closed, seam).max()))
    far = np.linalg.norm(co[:, :2] - np.array([0.0, 0.03]), axis=1) > 0.06
    check(np.abs(closed[far] - ripped[far]).max() < 1e-9, "the mesh away from the rip doesn't move")
    error = np.linalg.norm(closed - co, axis=1)[seam[torn].ravel()].max()
    before = np.linalg.norm(ripped - co, axis=1)[seam[torn].ravel()].max()
    check(error < 0.65 * before, "the sides meet about halfway ({:.1f} mm off the original, the tear was {:.1f} mm)"
          .format(error * 1000, before * 1000))
    # Smooth: no vertex near the seam moves much further than its neighbours.
    moved = np.linalg.norm(closed - ripped, axis=1)
    edges = np.concatenate((quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]))
    jump = np.abs(moved[edges[:, 0]] - moved[edges[:, 1]]).max()
    check(jump < 0.0015, "the moves fade out smoothly ({:.2f} mm at most between neighbours)".format(jump * 1000))


def test_closed_seam_untouched():
    co, quads, seam = split_sheet()
    rips = mending.find(co, tris_of(quads), everything(co), MAX_GAP)
    check(rips.count == 0, "an intact split seam isn't a rip")
    check(np.array_equal(mending.close(rips, everything(co), SPREAD), co), "and nothing moves")


def coarse_and_dense(offset):
    """Two sheets meeting along a curve, z = 6 * x^2 at y = 0, one with an edge every 2 mm along it and one with
    an edge every 8 mm, the coarse one moved ``offset`` (a crack between parts that weren't made together)."""
    def curve(x):
        return (x, 0.0, 6.0 * x * x)

    dense_co, dense_quads, dense = grid(41, 6, lambda i, j: tuple(np.add(curve(-0.04 + 0.002 * i), (0, -0.004 * (5 - j), 0))))
    coarse_co, coarse_quads, coarse = grid(11, 6, lambda i, j: tuple(np.add(curve(-0.04 + 0.008 * i), (0, 0.004 * j, 0))))
    coarse_co = coarse_co + np.array(offset)
    co, quads = join((dense_co, dense_quads), (coarse_co, coarse_quads))
    return co, quads, dense[:, -1], coarse[:, 0] + len(dense_co), curve


def test_crack_between_coarse_and_dense():
    co, quads, dense_rim, coarse_rim, curve = coarse_and_dense((0.0, 0.004, 0.001))
    rips = mending.find(co, tris_of(quads), everything(co), MAX_GAP)
    check(len(rips.point_node) >= 9 and not len(rips.pairs),
          "the coarse side's vertices are put on the dense side ({:d})".format(len(rips.point_node)))
    check(set(rips.node[coarse_rim].tolist()) >= set(rips.point_node.tolist()),
          "only the coarse side's vertices are put on the other side")
    closed = mending.close(rips, everything(co), SPREAD)
    a, b = closed[dense_rim[:-1]], closed[dense_rim[1:]]
    worst = 0.0
    for point in closed[coarse_rim[1:-1]]:
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
        worst = max(worst, np.linalg.norm(point - (a + t[:, None] * ab), axis=1).min())
    check(worst < 1e-5, "the coarse side lies on the dense one ({:.1e} m off)".format(worst))
    # The dense side keeps its curve: no kinks where the coarse vertices are.
    bend = np.linalg.norm(np.diff(closed[dense_rim], 2, axis=0), axis=1)
    bend_before = np.linalg.norm(np.diff(co[dense_rim], 2, axis=0), axis=1)
    check(bend.max() < 3.0 * bend_before.max() + 2e-5, "the dense side stays smooth (bends {:.3f} mm, was {:.3f})"
          .format(bend.max() * 1000, bend_before.max() * 1000))


def test_crease_closes():
    # Two sheets at right angles along x, with their corner torn open: the floor moved 4 mm off the wall.
    floor_co, floor_quads, floor = grid(16, 10, lambda i, j: (i * STEP, -(9 - j) * STEP, 0.0))
    wall_co, wall_quads, wall = grid(16, 10, lambda i, j: (i * STEP, 0.0, j * STEP))
    floor_co[:, 1] -= 0.004
    co, quads = join((floor_co, floor_quads), (wall_co, wall_quads))
    rips = mending.find(co, tris_of(quads), everything(co), MAX_GAP)
    check(len(rips.pairs) >= 14, "a crease torn open is a rip ({:d} pairs)".format(len(rips.pairs)))
    closed = mending.close(rips, everything(co), SPREAD)
    gap = np.linalg.norm(closed[floor[:, -1]] - closed[wall[:, 0] + len(floor_co)], axis=1).max()
    check(gap < 1e-5, "and closes ({:.1e} m)".format(gap))


def test_ribbon_not_a_rip():
    # The two sides of a 6 mm ribbon face each other too, but are only 6 mm apart over its faces.
    co, quads, _ = grid(30, 4, lambda i, j: (i * STEP, j * 0.002, 0.0))
    rips = mending.find(co, tris_of(quads), everything(co), MAX_GAP)
    check(rips.count == 0, "a ribbon's two sides aren't a rip ({:d})".format(rips.count))


def test_layers_not_a_rip():
    # A trim lying 3 mm over a panel, its edge along the panel's: they run the same way.
    panel_co, panel_quads, _ = grid(20, 10, lambda i, j: (i * STEP, j * STEP, 0.0))
    trim_co, trim_quads, _ = grid(20, 4, lambda i, j: (i * STEP, 0.036 - j * STEP + 0.001, 0.003))
    co, quads = join((panel_co, panel_quads), (trim_co, trim_quads[:, ::-1]))
    rips = mending.find(co, tris_of(quads), everything(co), MAX_GAP)
    check(rips.count == 0, "a layer lying on another isn't a rip ({:d})".format(rips.count))


def tube(center, radius=0.06, height=0.03, segments=48):
    def place(i, j):
        angle = 2.0 * math.pi * i / segments
        return (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle), j * STEP)
    co, quads, index = grid(segments + 1, int(height / STEP) + 1, place)
    # Close it round: the last column is the first.
    co = co[:-index.shape[1]]
    quads = np.where(quads >= len(co), quads - len(co), quads)
    return co, quads


def test_touching_hems_not_a_rip():
    # The hems of two trouser legs 5 mm apart at the inner thighs: closest in the middle, apart further on.
    co, quads = join(tube((-0.0625, 0.0)), tube((0.0625, 0.0)))
    rips = mending.find(co, tris_of(quads), everything(co), MAX_GAP)
    check(rips.count == 0, "trouser legs touching aren't a rip ({:d})".format(rips.count))


def test_max_gap():
    co, quads, seam = split_sheet()
    right = np.arange(len(co)) >= len(co) // 2
    ripped = co.copy()
    ripped[right, 0] += 0.015
    rips = mending.find(ripped, tris_of(quads), everything(co), MAX_GAP)
    check(rips.count == 0, "a gap wider than Max Gap isn't a rip")
    rips = mending.find(ripped, tris_of(quads), everything(co), 0.02)
    check(len(rips.pairs) == len(seam), "with a larger Max Gap it is, all along ({:d} of {:d} pairs)".format(
        len(rips.pairs), len(seam)))
    closed = mending.close(rips, everything(co), SPREAD)
    check(gaps(closed, seam).max() < 1e-5, "and closes")


def test_both_sides_must_be_selected():
    co, quads, seam, ripped = torn_sheet()
    left = np.arange(len(co)) < len(co) // 2   # with its copies of the seam
    rips = mending.find(ripped, tris_of(quads), left, MAX_GAP)
    check(rips.count == 0, "one side selected finds nothing")
    near = np.linalg.norm(co[:, :2] - np.array([0.0, 0.03]), axis=1) < 0.02
    rips = mending.find(ripped, tris_of(quads), near, MAX_GAP)
    check(rips.count > 0, "selecting around the rip finds it")


def test_hidden_side_stays():
    co, quads, seam, ripped = torn_sheet()
    rips = mending.find(ripped, tris_of(quads), everything(co), MAX_GAP)
    movable = np.arange(len(co)) < len(co) // 2   # the left half, its copies of the seam too
    closed = mending.close(rips, movable, SPREAD)
    right = ~movable
    check(np.abs(closed[right] - ripped[right]).max() == 0.0, "vertices that may not move stay")
    check(gaps(closed, seam).max() < 1e-5, "and the other side goes all the way ({:.1e} m)".format(
        gaps(closed, seam).max()))


# -----------------------------------------------------------------------------
# Welding

def bmesh_of(co, quads, groups=None):
    bm = bmesh.new()
    deform = bm.verts.layers.deform.verify()
    verts = [bm.verts.new(p) for p in co.tolist()]
    if groups is not None:
        for vert, weight in zip(verts, groups):
            vert[deform][0] = weight
    uv = bm.loops.layers.uv.verify()
    for quad in quads.tolist():
        face = bm.faces.new([verts[k] for k in quad])
        for loop in face.loops:
            loop[uv].uv = (loop.vert.co.x, loop.vert.co.y)   # the UVs follow the positions before closing
    bm.verts.ensure_lookup_table()
    return bm


def open_edges(bm):
    return sum(1 for edge in bm.edges if edge.is_boundary)


def test_weld_torn_seam():
    co, quads, seam, ripped = torn_sheet()
    rips = mending.find(ripped, tris_of(quads), everything(co), MAX_GAP)
    closed = mending.close(rips, everything(co), SPREAD)
    bm = bmesh_of(closed, quads)
    before = open_edges(bm)
    merged = mending.weld(bm, mending.weld_plan(rips, closed))
    check(merged == len(rips.pairs), "Weld merges each pair ({:d} of {:d})".format(merged, len(rips.pairs)))
    check(open_edges(bm) == before - 2 * (len(rips.pairs) - 1) or open_edges(bm) < before,
          "the torn stretch is no longer open ({:d} open edges, were {:d})".format(open_edges(bm), before))
    bm.free()


def test_weld_crack():
    co, quads, dense_rim, coarse_rim, _curve = coarse_and_dense((0.0, 0.003, 0.0))
    groups = np.where(np.arange(len(co)) < coarse_rim[0], 0.2, 1.0)
    rips = mending.find(co, tris_of(quads), everything(co), MAX_GAP)
    closed = mending.close(rips, everything(co), SPREAD)
    bm = bmesh_of(closed, quads, groups)
    before = open_edges(bm)
    merged = mending.weld(bm, mending.weld_plan(rips, closed))
    seam_edges = [edge for edge in bm.edges if edge.is_boundary and
                  all(abs(vert.co.y) < 0.0005 and abs(vert.co.x) < 0.039 for vert in edge.verts)]
    check(merged >= len(dense_rim) - 4, "Weld joins the dense side's vertices to the coarse side ({:d})".format(merged))
    check(not seam_edges, "nothing stays open along the crack ({:d} open edges there, {:d} were open in all)".format(
        len(seam_edges), before))
    deform = bm.verts.layers.deform.active
    weights = [vert[deform].get(0, 0.0) for vert in bm.verts if abs(vert.co.y) < 0.0005 and abs(vert.co.x) < 0.039]
    check(all(0.15 < w <= 1.0 for w in weights), "the vertices it adds take weights from their edge")
    uv = bm.loops.layers.uv.active
    off = max(abs(loop[uv].uv.x - loop.vert.co.x) for face in bm.faces for loop in face.loops)
    check(off < 0.002, "and UVs ({:.2f} mm off their position along the crack at most)".format(off * 1000))
    bm.free()


# -----------------------------------------------------------------------------
# The buttons

def test_buttons():
    magic_fit.register()
    try:
        co, quads, seam, ripped = torn_sheet()
        obj = mesh_object(ripped, quads)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        settings = bpy.context.scene.close_rips
        settings.max_gap, settings.spread, settings.use_weld = MAX_GAP, SPREAD, False
        result = bpy.ops.magic_fit.find_rips()
        check(result == {'FINISHED'}, "Find Rips finds the rip")
        bm = bmesh.from_edit_mesh(obj.data)
        selected = np.array([vert.select for vert in bm.verts])
        torn = seam[gaps(ripped, seam) > 1e-4].ravel()
        check(selected[torn].all() and selected.sum() <= 3 * len(torn),
              "and selects its sides ({:d} selected, {:d} torn)".format(int(selected.sum()), len(torn)))
        result = bpy.ops.magic_fit.close_rips()
        check(result == {'FINISHED'}, "Close Rips closes it")
        bpy.ops.object.mode_set(mode='OBJECT')
        now = np.array([vert.co[:] for vert in obj.data.vertices])
        check(gaps(now, seam).max() < 1e-5, "the seam is closed in the mesh ({:.1e} m)".format(gaps(now, seam).max()))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        result = bpy.ops.magic_fit.close_rips()
        check(result == {'CANCELLED'}, "without a selection, Close Rips does nothing")
        # Weld, from the ripped mesh again.
        obj = mesh_object(ripped, quads)
        count = len(obj.data.vertices)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        settings.use_weld = True
        result = bpy.ops.magic_fit.close_rips()
        bpy.ops.object.mode_set(mode='OBJECT')
        check(result == {'FINISHED'} and len(obj.data.vertices) < count,
              "with Weld, the torn vertices are merged ({:d} -> {:d})".format(count, len(obj.data.vertices)))
        settings.use_weld = False
    finally:
        if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        magic_fit.unregister()


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception:
            FAILURES.append(test.__name__ + " raised")
            traceback.print_exc()
    print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
    sys.exit(1 if FAILURES else 0)


main()
