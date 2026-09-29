"""Headless tests for the Straighten weight brush engine.

Run from the repository root:
    blender -b --factory-startup --python tests/test_straighten_core.py
"""

import os
import sys
import time
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import cleavage  # noqa: E402
from magic_fit import fitting, painting, straighten  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def new_stroke(obj, body, **kwargs):
    """A stroke on ``obj`` as displayed now, the way the brush starts one."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    posed = painting.read_geometry(obj, depsgraph, source='INDEXED').co
    surface = fitting.BodySurface.from_object(body, depsgraph)
    rest, = straighten.rest_geometry(bpy.context, (obj,), ('INDEXED',))
    return straighten.StraightenStroke(obj, rest.co, posed, surface, **kwargs)


def click(stroke, obj, center, radius, strength=1.0):
    """One dab (smooth falloff) at ``center`` as displayed, then the end of the stroke."""
    co = cleavage.displayed(obj)
    t = np.linalg.norm(co - center, axis=1) / radius
    verts = np.flatnonzero(t < 1.0)
    stroke.apply(verts.tolist(), (painting.falloff_weights('SMOOTH', t[verts]) * strength).tolist(), radius=radius)
    stroke.finish()
    obj.data.update()


def middle_of(xs, zs):
    return np.argmin(np.abs(xs)) * len(zs) + np.argmin(np.abs(zs))


def test_cotangent_blend_is_linear():
    # A flat strip between two supports: blended weights go evenly from one to the other.
    xs = np.linspace(0.0, 1.0, 11)
    co = np.array([(x, z, 0.0) for x in xs for z in (0.0, 0.1)])
    tris = []
    for i in range(10):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        tris += [(a, c, d), (a, d, b)]
    edges, weights = straighten.cotangent_weights(co, np.array(tris))
    check(weights.min() > 0.0, "cotangent weights are positive")
    n = len(co)
    fixed = np.zeros(n, bool)
    fixed[[0, 1, n - 2, n - 1]] = True
    value = np.where(np.arange(n) >= n - 2, 1.0, 0.0)
    L = np.zeros((n, n))
    for (i, j), w in zip(edges.tolist(), weights.tolist()):
        L[i, j] -= w
        L[j, i] -= w
        L[i, i] += w
        L[j, j] += w
    free = ~fixed
    solution = np.linalg.solve(L[np.ix_(free, free)], -L[np.ix_(free, fixed)] @ value[fixed])
    check(np.abs(solution - co[free, 0]).max() < 1e-3, "harmonic blend over a flat strip is linear ({:.1e})".format(
        np.abs(solution - co[free, 0]).max()))


def test_solver_zero_column():
    # A bone found only on vertices that don't hold the span has nothing on the right-hand side of the
    # held blend: its column is 0 right away, and doesn't keep the others iterating.
    size = 50
    calls = [0]

    def multiply(x):
        calls[0] += 1
        grid = x.reshape(size, size, -1)
        result = 4.0 * grid
        result[1:] -= grid[:-1]
        result[:-1] -= grid[1:]
        result[:, 1:] -= grid[:, :-1]
        result[:, :-1] -= grid[:, 1:]
        return result.reshape(size * size, -1)
    rhs = np.zeros((size * size, 2))
    rhs[:size, 0] = 1.0
    scale = np.full(size * size, 0.25)
    one = straighten._conjugate_gradient(multiply, rhs[:, :1], np.full((size * size, 1), 0.3), scale)
    alone = calls[0]
    calls[0] = 0
    both = straighten._conjugate_gradient(multiply, rhs, np.full((size * size, 2), 0.3), scale)
    check(np.array_equal(both[:, 1], np.zeros(size * size)) and np.abs(both[:, 0] - one[:, 0]).max() < 1e-9,
          "solver: a column without a right-hand side is 0")
    check(calls[0] == alone, "solver: it takes no extra iterations ({} vs {})".format(calls[0], alone))


def test_sinking():
    # A V-shaped profile extruded: the bottom of the V lies below the hull by its depth, the rims on it.
    points, normals = [], []
    for x in np.linspace(-0.1, 0.1, 21):
        for z in (0.0, 0.05):
            points.append((x, 0.02 * (1.0 - abs(x) / 0.1), z))   # y: rims at 0, bottom at 0.02 (deeper = +y)
            normals.append((0.0, -1.0, 0.0))
    points, normals = np.array(points), np.array(normals)
    sink = straighten.sinking(points, normals, np.arange(len(points)))
    bottom = np.abs(points[:, 0]) < 1e-9
    rims = np.abs(np.abs(points[:, 0]) - 0.1) < 1e-9
    check(np.abs(sink[bottom] - 0.02).max() < 1e-4 and sink[rims].max() < 1e-4,
          "sinking: the bottom of a V lies its depth below the hull, the rims on it ({:.4f})".format(
              sink[bottom].max()))


def test_needs_a_pose():
    rig, body, garment, xs, zs = cleavage.build()
    start = cleavage.weights(garment)
    stroke = new_stroke(garment, body)
    check(stroke.unposed, "rest pose: the stroke knows nothing is posed")
    click(stroke, garment, cleavage.displayed(garment)[middle_of(xs, zs)], 0.05)
    check(np.array_equal(cleavage.weights(garment), start), "rest pose: nothing changes")


def test_straighten_cleavage():
    rig, body, garment, xs, zs = cleavage.build()
    cleavage.scale_breasts(rig, 1.0)
    rest = cleavage.displayed(garment)
    cleavage.scale_breasts(rig, 1.4)
    posed = cleavage.displayed(garment)
    before_fold = cleavage.fold(posed, xs, zs)
    before_notch = cleavage.notch(posed, xs, zs)
    before = cleavage.distortion(posed, rest, xs, zs)
    check(before_fold > 0.02, "copied weights: the fabric folds over in the cleavage when scaled ({:.1f} mm)".format(
        before_fold * 1000))
    check(before_notch > 0.005, "copied weights: seen from above, the outline dips at the cleavage ({:.1f} mm)".format(
        before_notch * 1000))
    start = cleavage.weights(garment)

    stroke = new_stroke(garment, body)
    check(bpy.context.scene.objects["Rig"].data.pose_position == 'POSE', "the pose is back after reading the rest pose")
    middle = middle_of(xs, zs)
    stroke.apply([middle], [1.0], radius=0.05)
    check(np.array_equal(cleavage.weights(garment), start), "nothing changes before the stroke ends")
    t0 = time.time()
    click(stroke, garment, posed[middle], 0.05)
    seconds = time.time() - t0
    after_points = cleavage.displayed(garment)
    after_fold = cleavage.fold(after_points, xs, zs)
    after_notch = cleavage.notch(after_points, xs, zs)
    after = cleavage.distortion(after_points, rest, xs, zs)
    check(after_fold < 0.003, "straightened: the fold is gone ({:.1f} -> {:.1f} mm)".format(
        before_fold * 1000, after_fold * 1000))
    check(after_notch < 0.0025, "straightened: the outline runs (nearly) straight across ({:.1f} -> {:.1f} mm)".format(
        before_notch * 1000, after_notch * 1000))
    check(after < 0.8 * before, "straightened: the shape stays closer to the rest shape ({:.2f} -> {:.2f} mm)".format(
        before * 1000, after * 1000))
    check(seconds < 2.0, "straightened in {:.2f} s".format(seconds))

    weights = cleavage.weights(garment)
    changed = np.abs(weights - start).max(axis=1) > 1e-6
    check(stroke.span_size == int(changed.sum()) and changed.any(), "the span is what changed ({} vertices)".format(
        stroke.span_size))
    X = rest[:, 0]
    check(not changed[np.abs(X) > 0.1].any(), "the parts beside the breasts keep their weights")
    check(np.abs(weights.sum(axis=1) - start.sum(axis=1)).max() < 1e-4, "every vertex keeps its total weight")
    check(abs(weights[middle, 0] - weights[middle, 1]) < 0.05 and weights[middle, :2].sum() > 0.8,
          "the middle of the bridge follows both breasts alike ({})".format(weights[middle].round(2)))

    # Another profile: the weights don't depend on the pose they were worked out in.
    cleavage.scale_breasts(rig, 1.2)
    other = cleavage.displayed(garment)
    check(cleavage.fold(other, xs, zs) < 0.001 and cleavage.notch(other, xs, zs) < 0.0015,
          "straight at 120 % too ({:.1f} / {:.1f} mm)".format(
              cleavage.fold(other, xs, zs) * 1000, cleavage.notch(other, xs, zs) * 1000))
    cleavage.scale_breasts(rig, 1.0)
    check(np.abs(cleavage.displayed(garment) - rest).max() < 1e-6, "the rest shape doesn't change")
    cleavage.scale_breasts(rig, 1.4)

    stroke.restore()
    garment.data.update()
    check(np.array_equal(cleavage.weights(garment), start), "cancel: every weight restored exactly")


def test_strap_to_contact():
    # A strap from the front of the breast down to the chest: clicking just under the breast straightens
    # all of it, down to where it touches the chest again.
    rig, body, garment, xs, zs = cleavage.build()
    garment.hide_set(True)
    strap = cleavage.add_strap(rig, body)
    cleavage.scale_breasts(rig, 1.0)
    rest_sag = cleavage.strap_sag(cleavage.displayed(strap))
    cleavage.scale_breasts(rig, 1.4)
    posed = cleavage.displayed(strap)
    before = cleavage.strap_sag(posed)
    check(rest_sag < 0.001, "the strap runs straight at rest ({:.1f} mm)".format(rest_sag * 1000))
    check(before > 0.005, "scaled: the copied weights make it sag under the breast ({:.1f} mm)".format(before * 1000))
    start = cleavage.weights(strap)
    szs = cleavage.STRAP_ZS
    under = 1 * len(szs) + int(np.argmin(np.abs(szs + 0.07)))   # middle line, 7 cm below the breast center
    stroke = new_stroke(strap, body)
    click(stroke, strap, posed[under], 0.02)
    after = cleavage.strap_sag(cleavage.displayed(strap))
    check(after < 0.35 * before, "one click straightens the strap ({:.1f} -> {:.1f} mm)".format(
        before * 1000, after * 1000))
    changed = np.abs(cleavage.weights(strap) - start).max(axis=1) > 1e-6
    reached = szs[np.unique(np.flatnonzero(changed) % len(szs))]
    check(reached.min() < -0.1, "the click reached along the strap, far past the brush ({:.0f} to {:.0f} mm)".format(
        reached.min() * 1000, reached.max() * 1000))


def test_nothing_sags_where_it_rests():
    rig, body, garment, xs, zs = cleavage.build()
    cleavage.scale_breasts(rig, 1.4)
    start = cleavage.weights(garment)
    posed = cleavage.displayed(garment)
    cup = np.argmin(np.abs(xs + 0.1)) * len(zs) + np.argmin(np.abs(zs))   # the outer side of the left cup
    stroke = new_stroke(garment, body)
    click(stroke, garment, posed[cup], 0.01)
    check(stroke.found_nothing, "clicking the side of a cup: nothing there sags")
    check(np.array_equal(cleavage.weights(garment), start), "clicking the side of a cup changes nothing")


def test_strength():
    rig, body, garment, xs, zs = cleavage.build()
    cleavage.scale_breasts(rig, 1.4)
    start = cleavage.weights(garment)
    center = cleavage.displayed(garment)[middle_of(xs, zs)]
    full = new_stroke(garment, body)
    click(full, garment, center, 0.05)
    target = cleavage.weights(garment)
    full.restore()
    garment.data.update()
    half = new_stroke(garment, body)
    click(half, garment, center, 0.05, strength=0.5)
    weights = cleavage.weights(garment)
    moved = np.abs(target - start).max(axis=1) > 0.05
    check(moved.any() and np.abs(weights[moved] - 0.5 * (start[moved] + target[moved])).max() < 0.02,
          "half strength goes half the way")


def test_locked_group_kept():
    rig, body, garment, xs, zs = cleavage.build()
    cleavage.scale_breasts(rig, 1.4)
    garment.vertex_groups["j_sebo_c"].lock_weight = True
    start = cleavage.weights(garment)
    stroke = new_stroke(garment, body)
    click(stroke, garment, cleavage.displayed(garment)[middle_of(xs, zs)], 0.05)
    weights = cleavage.weights(garment)
    spine = garment.vertex_groups["j_sebo_c"].index
    check(np.array_equal(weights[:, spine], start[:, spine]), "locked: the locked group is untouched")
    check(np.abs(weights - start).max() > 0.1, "locked: the other groups still changed")
    check(np.abs(weights.sum(axis=1) - start.sum(axis=1)).max() < 1e-4, "locked: totals kept")


def test_current_group_only():
    rig, body, garment, xs, zs = cleavage.build()
    cleavage.scale_breasts(rig, 1.4)
    garment.vertex_groups.active_index = garment.vertex_groups["j_mune_l"].index
    start = cleavage.weights(garment)
    stroke = new_stroke(garment, body, group_mode='ACTIVE')
    click(stroke, garment, cleavage.displayed(garment)[middle_of(xs, zs)], 0.05)
    weights = cleavage.weights(garment)
    l, r, s = (garment.vertex_groups[name].index for name in cleavage.BONES)
    changed = np.abs(weights - start).max(axis=0)
    check(changed[l] > 0.1, "current group: the active group changed ({:.2f})".format(changed[l]))
    both = (start[:, r] > 1e-3) & (start[:, s] > 1e-3) & (weights[:, r] > 1e-3) & (weights[:, s] > 1e-3)
    ratio_before = start[both, r] / start[both, s]
    ratio_after = weights[both, r] / weights[both, s]
    check(both.any() and np.abs(ratio_after / ratio_before - 1.0).max() < 1e-3,
          "current group: the other bones kept their proportions")


def test_selection_limits_the_span():
    # With selection masking, only the selected vertices may change: the span stops at the others.
    rig, body, garment, xs, zs = cleavage.build()
    cleavage.scale_breasts(rig, 1.4)
    start = cleavage.weights(garment)
    X = cleavage.displayed(garment)[:, 0]
    allowed = X < 0.0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    posed = painting.read_geometry(garment, depsgraph, source='INDEXED').co
    surface = fitting.BodySurface.from_object(body, depsgraph)
    rest, = straighten.rest_geometry(bpy.context, (garment,), ('INDEXED',))
    stroke = straighten.StraightenStroke(garment, rest.co, posed, surface, allowed=allowed)
    click(stroke, garment, posed[middle_of(xs, zs)] - np.array([0.01, 0.0, 0.0]), 0.05)
    changed = np.abs(cleavage.weights(garment) - start).max(axis=1) > 1e-6
    check(changed.any() and not changed[~allowed].any(),
          "selection masking: only selected vertices changed ({} changed, none unselected)".format(int(changed.sum())))


def test_no_bones_around():
    # A sagging part with nothing around it that has bone weights: nothing to blend from.
    rig, body, garment, xs, zs = cleavage.build()
    cleavage.scale_breasts(rig, 1.4)
    for group in garment.vertex_groups:
        group.lock_weight = True
    start = cleavage.weights(garment)
    stroke = new_stroke(garment, body)
    click(stroke, garment, cleavage.displayed(garment)[middle_of(xs, zs)], 0.05)
    check(stroke.unsupported, "all groups locked: the stroke says there's nothing to blend from")
    check(np.array_equal(cleavage.weights(garment), start), "all groups locked: nothing changes")


def main():
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
