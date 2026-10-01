"""Headless tests for Fit Move's engine (`magic_fit.moving`), next to Blender's own Move.

Run from the repository root:
    blender -b --factory-startup --python tests/test_move_core.py
"""

import os
import sys
import traceback

import bmesh
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from magic_fit import moving  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def grid_tris(index):
    """Triangles of a grid whose vertex at (column, row) is ``index[column, row]``."""
    a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    return np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))


GAP = 0.012
LEG = 0.1
STEP = 0.004


def trouser_strips(split_at=None):
    """Two strips of cloth hanging GAP apart, like trouser legs on both inner thighs, joined over the gap at the
    top (the crotch). Returns positions, triangles, and each vertex's distance along the cloth from the
    bottom of the left strip. ``split_at``: that column of the left strip is split, like a UV seam (its
    vertices doubled, one copy for each side)."""
    us = np.arange(round((2 * LEG + GAP) / STEP) + 1) * STEP
    vs = np.linspace(-0.03, 0.03, 13)

    def place(u, v):
        if u <= LEG:
            return (-GAP / 2.0, v, u - LEG)
        if u <= LEG + GAP:
            return (-GAP / 2.0 + (u - LEG), v, 0.0)
        return (GAP / 2.0, v, -(u - LEG - GAP))

    co = [place(u, v) for u in us for v in vs]
    along = list(np.repeat(us, len(vs)))
    index = np.arange(len(co)).reshape(len(us), len(vs))
    if split_at is not None:
        column = int(np.argmin(np.abs(us - split_at)))
        copies = np.arange(len(co), len(co) + len(vs))
        co += [co[i] for i in index[column]]
        along += [along[i] for i in index[column]]
        right = index.copy()
        right[column] = copies
        tris = np.concatenate((grid_tris(index[:column + 1]), grid_tris(right[column:])))
    else:
        tris = grid_tris(index)
    return np.array(co), tris, np.array(along)


def edges_of(tris):
    edges = np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def blender_move(co, tris, selected, hidden, translation, **proportional):
    """Blender's own Move on a mesh of ``co`` and ``tris``: the positions afterwards."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    mesh = bpy.data.meshes.new("Strips")
    mesh.from_pydata(co.tolist(), [], tris.tolist())
    obj = bpy.data.objects.new("Strips", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.select_mode = {'VERT'}
    for vert in bm.verts:
        vert.select = bool(selected[vert.index])
    bm.select_flush_mode()
    for face in bm.faces:
        face.hide = any(hidden[vert.index] for vert in face.verts)
    for edge in bm.edges:
        edge.hide = any(hidden[vert.index] for vert in edge.verts)
    for vert in bm.verts:
        vert.hide = bool(hidden[vert.index])
    bmesh.update_edit_mesh(mesh)
    bpy.ops.transform.translate(value=translation, use_proportional_edit=True, **proportional)
    bpy.ops.object.mode_set(mode='OBJECT')
    return np.array([vert.co[:] for vert in mesh.vertices])


def pick(co, along, u, v=0.0):
    return int(np.argmin(np.abs(along - u) + np.abs(co[:, 1] - v)))


MOVE = np.array([-0.005, 0.0, 0.0])  # away from the gap


def test_matches_blender_connected_only():
    # Where there are no seams or hidden parts, the falloff is Blender's own, distances straight across the
    # triangles and all.
    co, tris, along = trouser_strips()
    selected = np.zeros(len(co), bool)
    selected[pick(co, along, 0.048)] = True
    for falloff in ('SMOOTH', 'SPHERE', 'ROOT', 'INVERSE_SQUARE', 'SHARP', 'LINEAR'):
        theirs = blender_move(co, tris, selected, np.zeros(len(co), bool), MOVE, proportional_size=0.032,
                              proportional_edit_falloff=falloff, use_proportional_connected=True)
        move = moving.FitMove(co, edges_of(tris), selected, np.ones(len(co), bool), tris=tris,
                              proportional_size=0.032, falloff=falloff)
        ours = move.positions(MOVE, np.arange(len(co)))
        check(np.abs(ours - theirs).max() < 1e-5, "{}: moves like Blender's Connected Only (off by {:.1e} m)".format(
            falloff, np.abs(ours - theirs).max()))


def test_other_leg_stays():
    co, tris, along = trouser_strips()
    selected = np.zeros(len(co), bool)
    selected[pick(co, along, 0.048)] = True
    other = along > LEG + GAP
    theirs = blender_move(co, tris, selected, np.zeros(len(co), bool), MOVE, proportional_size=0.032)
    check(np.abs(theirs[other] - co[other]).max() > 0.002,
          "Blender's Move pulls the other leg along ({:.1f} mm)".format(np.abs(theirs[other] - co[other]).max() * 1000))
    move = moving.FitMove(co, edges_of(tris), selected, np.ones(len(co), bool), tris=tris, proportional_size=0.032)
    check(not np.isin(np.flatnonzero(other), move.indices).any(), "Fit Move leaves the other leg alone")
    check(abs(move.share[pick(co, along, 0.064)] - 0.5) < 0.01,
          "halfway out along its own leg, a vertex moves half as far ({:.2f})".format(move.share[pick(co, along, 0.064)]))


def test_split_seam_stays_closed():
    co, tris, along = trouser_strips(split_at=0.06)
    seam = np.flatnonzero(np.abs(along - 0.06) < 1e-9)
    pairs = seam.reshape(2, -1).T  # each seam vertex and its copy
    selected = np.zeros(len(co), bool)
    selected[pick(co, along, 0.048)] = True
    theirs = blender_move(co, tris, selected, np.zeros(len(co), bool), MOVE, proportional_size=0.032,
                          use_proportional_connected=True)
    opened = np.linalg.norm(theirs[pairs[:, 0]] - theirs[pairs[:, 1]], axis=1).max()
    check(opened > 0.001, "Blender's Connected Only opens the split seam ({:.1f} mm)".format(opened * 1000))
    move = moving.FitMove(co, edges_of(tris), selected, np.ones(len(co), bool), tris=tris, proportional_size=0.032)
    ours = move.positions(MOVE, np.arange(len(co)))
    gap = np.linalg.norm(ours[pairs[:, 0]] - ours[pairs[:, 1]], axis=1).max()
    check(gap < 1e-12, "Fit Move keeps it closed ({:.1e} m)".format(gap))
    beyond = along > 0.062
    check(np.abs(ours[beyond] - co[beyond]).max() > 0.001, "and reaches across it")


def test_fade_at_hidden():
    co, tris, along = trouser_strips()
    hidden = along >= 0.08 - 1e-9  # the top of the left leg and all beyond
    border = np.flatnonzero(np.abs(along - 0.076) < 1e-9)  # the last visible row
    selected = np.zeros(len(co), bool)
    selected[pick(co, along, 0.048)] = True
    theirs = blender_move(co, tris, selected, hidden, MOVE, proportional_size=0.04)
    check(np.abs(theirs[border] - co[border]).max() > 0.0005,
          "Blender's Move pulls the last visible row off the hidden one ({:.2f} mm)".format(
              np.abs(theirs[border] - co[border]).max() * 1000))
    move = moving.FitMove(co, edges_of(tris), selected, ~hidden, tris=tris, proportional_size=0.04, fade_distance=0.02)
    ours = move.positions(MOVE, np.arange(len(co)))
    check(np.array_equal(ours[hidden], co[hidden]), "Fit Move leaves hidden vertices alone")
    check(np.array_equal(ours[border], co[border]), "and the vertices next to them")
    rows = np.array([move.share[pick(co, along, u)] for u in np.arange(0.048, 0.08, STEP)])
    check(rows[0] > 0.99 and (np.diff(rows) <= 1e-9).all() and np.abs(np.diff(rows)).max() < 0.4,
          "it eases in toward them ({})".format(", ".join("{:.2f}".format(r) for r in rows)))
    again = moving.FitMove(ours, edges_of(tris), selected, ~hidden, tris=tris, proportional_size=0.04, fade_distance=0.02)
    check(np.array_equal(again.positions(MOVE, border), co[border]), "moving again doesn't wear the fade down")


def test_welded_selection_moves_whole():
    # Selecting one copy of a split seam vertex moves both, without proportional editing too.
    co, tris, along = trouser_strips(split_at=0.06)
    seam = np.flatnonzero(np.abs(along - 0.06) < 1e-9).reshape(2, -1)
    selected = np.zeros(len(co), bool)
    selected[seam[0]] = True
    move = moving.FitMove(co, edges_of(tris), selected, np.ones(len(co), bool), tris=tris)
    check(np.array_equal(np.sort(move.indices), np.sort(seam.ravel())), "both copies of the seam move, nothing else")


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
