"""Headless tests for the body fitting engine.

Run from the repository root:
    blender -b --factory-startup --python tests/test_fit_core.py
"""

import math
import os
import sys
import traceback
from types import SimpleNamespace

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from magic_fit import clipping, fitting, operators  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def uv_sphere(radius, segments=64, rings=32, radial=None, split_seam=False):
    """Triangulated UV sphere; ``radial(theta, phi)`` scales the radius per vertex."""
    columns = segments + 1 if split_seam else segments
    co = [(0.0, 0.0, radius)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        for j in range(columns):
            phi = 2.0 * math.pi * j / segments
            r = radius * (radial(theta, phi) if radial else 1.0)
            co.append((r * math.sin(theta) * math.cos(phi), r * math.sin(theta) * math.sin(phi), r * math.cos(theta)))
    co.append((0.0, 0.0, -radius))
    last = len(co) - 1

    def ring(i, j):
        return 1 + i * columns + (j if split_seam else j % segments)

    tris = []
    for j in range(segments):
        tris.append((0, ring(0, j), ring(0, j + 1)))
        for i in range(rings - 2):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            tris += [(a, c, d), (a, d, b)]
        tris.append((ring(rings - 2, j), last, ring(rings - 2, j + 1)))
    return np.array(co, dtype=np.float64), np.array(tris, dtype=np.int64)


def body_surface(co, tris):
    bvh = BVHTree.FromPolygons(co.tolist(), tris.tolist(), all_triangles=True)
    return fitting.BodySurface(co, tris, bvh)


def bumpy_body(amplitude=0.008):
    """A sphere with pseudo-random radial noise on every vertex, like a lumpy low-poly body."""
    co, tris = uv_sphere(1.0)
    noise = np.modf(np.sin(np.arange(len(co)) * 12.9898) * 43758.5453)[0]
    return body_surface(co * (1.0 + amplitude * noise)[:, None], tris)


def edges_of(tris):
    edges = np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def roughness(values, edges, subset):
    """Mean distance of ``values`` from the average of their neighbours, over ``subset``."""
    total = np.zeros(len(values))
    count = np.zeros(len(values))
    np.add.at(total, edges[:, 0], values[edges[:, 1]])
    np.add.at(total, edges[:, 1], values[edges[:, 0]])
    np.add.at(count, edges[:, 0], 1.0)
    np.add.at(count, edges[:, 1], 1.0)
    return np.abs(values - total / np.maximum(count, 1.0))[subset].mean()


def shell(inner, outer, radial=None, segments=48, rings=24):
    """Two layers with matching vertices: index k of the inner layer lies below index k + n."""
    a, _ = uv_sphere(inner, segments, rings, radial)
    b, _ = uv_sphere(outer, segments, rings, radial)
    return np.concatenate((a, b)), len(a)


def paint(stroke, center, radius, dabs=40, strength=1.0):
    for _ in range(dabs):
        stroke.dab(center, radius, strength)


def angle_from(co, direction):
    unit = co / np.linalg.norm(co, axis=1, keepdims=True)
    return np.arccos(np.clip(unit @ direction, -1.0, 1.0))


FRONT = np.array([0.0, -1.0, 0.0])


def test_push_keeps_thickness():
    body = body_surface(*uv_sphere(1.0))
    co, n = shell(0.97, 1.00)          # 3 cm thick, inner layer 3 cm inside the body
    start = co.copy()
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='PUSH', offset=0.01, layer_radius=0.2)
    paint(stroke, FRONT * 1.0, 0.5)

    angle = angle_from(start[:n], FRONT)
    center = angle < 0.15
    inner_r = np.linalg.norm(co[:n], axis=1)
    outer_r = np.linalg.norm(co[n:], axis=1)
    thickness = outer_r - inner_r
    check(center.sum() > 3, "push: vertices in the brush center ({})".format(center.sum()))
    check(inner_r[center].min() > 1.0 + 0.01 - 0.002,
          "push: inner layer cleared the body ({:.4f})".format(inner_r[center].min()))
    check(np.abs(thickness[center] - 0.03).max() < 0.001,
          "push: thickness kept at the center (max error {:.5f})".format(np.abs(thickness[center] - 0.03).max()))
    check(np.abs(thickness - 0.03).max() < 0.002,
          "push: thickness kept everywhere (max error {:.5f})".format(np.abs(thickness - 0.03).max()))
    far = angle > 1.0
    check(np.abs(co[:n][far] - start[:n][far]).max() == 0.0 and np.abs(co[n:][far] - start[n:][far]).max() == 0.0,
          "push: vertices outside the brush untouched")
    # A shrinkwrap would have put both layers on the same offset surface.
    check(thickness[center].min() > 0.02, "push: layers did not collapse")


def test_push_mismatched_layers():
    # Real clothing layers don't line up vertex for vertex: different density, rotated.
    body = body_surface(*uv_sphere(1.0))
    inner, _ = uv_sphere(0.97, 48, 24)
    outer, _ = uv_sphere(1.00, 37, 19)
    turn = 0.05
    outer = outer @ np.array([[math.cos(turn), -math.sin(turn), 0], [math.sin(turn), math.cos(turn), 0], [0, 0, 1]]).T
    co = np.concatenate((inner, outer))
    n = len(inner)
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='PUSH', offset=0.01, layer_radius=0.2)
    paint(stroke, FRONT * 1.0, 0.5)
    inner_r = np.linalg.norm(co[:n], axis=1)[angle_from(co[:n], FRONT) < 0.15]
    outer_r = np.linalg.norm(co[n:], axis=1)[angle_from(co[n:], FRONT) < 0.15]
    check(np.abs(inner_r - 1.01).max() < 0.001, "mismatched: inner layer at the offset ({:.4f}..{:.4f})".format(inner_r.min(), inner_r.max()))
    check(np.abs(outer_r - 1.04).max() < 0.002, "mismatched: outer layer kept its distance ({:.4f}..{:.4f})".format(outer_r.min(), outer_r.max()))


def test_push_leaves_clear_mesh_alone():
    body = body_surface(*uv_sphere(1.0))
    co, n = shell(1.02, 1.05)
    start = co.copy()
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='PUSH', offset=0.01, layer_radius=0.2)
    paint(stroke, FRONT * 1.05, 0.5, dabs=10)
    check(np.array_equal(co, start), "push: a mesh already clear of the body does not move")


def test_tighten_keeps_thickness():
    body = body_surface(*uv_sphere(1.0))
    co, n = shell(1.10, 1.13)
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='TIGHTEN', offset=0.01, layer_radius=0.2,
                               max_distance=0.5)
    paint(stroke, FRONT * 1.13, 0.5, dabs=80)
    center = angle_from(co[:n], FRONT) < 0.15
    inner_r = np.linalg.norm(co[:n], axis=1)
    thickness = np.linalg.norm(co[n:], axis=1) - inner_r
    check(np.abs(inner_r[center] - 1.01).max() < 0.003,
          "tighten: inner layer pulled to the offset ({:.4f}..{:.4f})".format(inner_r[center].min(), inner_r[center].max()))
    check(np.abs(thickness - 0.03).max() < 0.002,
          "tighten: thickness kept (max error {:.5f})".format(np.abs(thickness - 0.03).max()))


def test_tighten_respects_max_distance():
    body = body_surface(*uv_sphere(1.0))
    co, n = shell(1.30, 1.33)
    start = co.copy()
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='TIGHTEN', offset=0.01, layer_radius=0.2,
                               max_distance=0.1)
    paint(stroke, FRONT * 1.33, 0.5, dabs=10)
    check(np.array_equal(co, start), "tighten: layers farther than Max Distance are left alone")


def test_details_move_as_one():
    # Small folds on both layers: a dab must shift them, not iron them out.
    folds = lambda theta, phi: 1.0 + 0.004 * math.sin(40 * phi) * math.sin(40 * theta)  # noqa: E731
    body = body_surface(*uv_sphere(1.0))
    co, n = shell(0.98, 1.00, radial=folds, segments=96, rings=48)
    start = co.copy()
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='PUSH', offset=0.01, layer_radius=0.2)
    paint(stroke, FRONT * 1.0, 0.5)
    center = angle_from(start, FRONT) < 0.12
    moved = np.linalg.norm(co, axis=1) - np.linalg.norm(start, axis=1)
    spread = moved[center].max() - moved[center].min()
    check(moved[center].mean() > 0.02, "folds: pushed out ({:.4f})".format(moved[center].mean()))
    check(spread < 0.1 * moved[center].mean(),
          "folds: all vertices at the center moved alike (spread {:.5f} of {:.4f})".format(spread, moved[center].mean()))


def test_movable_mask():
    body = body_surface(*uv_sphere(1.0))
    co, n = shell(0.97, 1.00)
    start = co.copy()
    movable = np.ones(len(co), bool)
    movable[n:] = False  # e.g. the outer layer hidden or unselected
    stroke = fitting.FitStroke(body, co, movable, mode='PUSH', offset=0.01, layer_radius=0.2)
    paint(stroke, FRONT * 1.0, 0.5, dabs=5)
    check(np.array_equal(co[n:], start[n:]), "mask: vertices that may not move stay put")
    check(not np.array_equal(co[:n], start[:n]), "mask: the others moved")


def test_inside_out_body():
    co_b, tris = uv_sphere(1.0)
    body = body_surface(co_b, tris[:, ::-1].copy())
    co, n = shell(0.97, 1.00)
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='PUSH', offset=0.01, layer_radius=0.2)
    paint(stroke, FRONT * 1.0, 0.5)
    center = angle_from(co[:n], FRONT) < 0.15
    check(np.linalg.norm(co[:n], axis=1)[center].min() > 1.005, "inside-out body: still pushed outward")


def test_split_seam_body():
    # Duplicate vertices along a seam must not leave a crease in the fitted mesh.
    body = body_surface(*uv_sphere(1.0, split_seam=True))
    co, n = shell(0.97, 1.00)
    start = co.copy()
    side = np.array([1.0, 0.0, 0.0])  # the seam of the body sphere is at +X
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='PUSH', offset=0.01, layer_radius=0.2)
    paint(stroke, side * 1.0, 0.5)
    center = angle_from(start[:n], side) < 0.15
    inner_r = np.linalg.norm(co[:n], axis=1)
    check(inner_r[center].min() > 1.008 and inner_r[center].max() - inner_r[center].min() < 0.004,
          "seam: even push across the seam ({:.4f}..{:.4f})".format(inner_r[center].min(), inner_r[center].max()))


def test_fit_both_ways():
    body = body_surface(*uv_sphere(1.0))
    # One half of the shell floats, the other half is inside the body.
    radial = lambda theta, phi: 1.05 if math.cos(phi) > 0 else 0.97  # noqa: E731
    co, n = shell(1.0, 1.03, radial=lambda t, p: radial(t, p))
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='FIT', offset=0.01, layer_radius=0.05,
                               max_distance=0.5)
    for direction in (np.array([1.0, -0.2, 0.0]), np.array([-1.0, -0.2, 0.0])):
        direction /= np.linalg.norm(direction)
        paint(stroke, direction * 1.05, 0.3, dabs=80)
        center = angle_from(co[:n], direction) < 0.1
        inner_r = np.linalg.norm(co[:n], axis=1)[center]
        check(np.abs(inner_r - 1.01).max() < 0.004,
              "fit: inner layer at the offset on side {} ({:.4f}..{:.4f})".format(direction.round(2), inner_r.min(), inner_r.max()))


# Auto Smooth is judged at a realistic scale: a torso-sized body (15 cm radius, ~5 mm edges), an 8 mm
# thick garment, default settings and strokes that move back and forth across the area.
TORSO = 0.15


def torso(radial=None):
    return body_surface(*uv_sphere(TORSO, 192, 96, radial))


def garment(inner, radial=None, thickness=0.008):
    """Two matching layers ``thickness`` apart, and the edges of both."""
    a, tris = uv_sphere(inner, 144, 72, radial)
    b, _ = uv_sphere(inner + thickness, 144, 72, radial)
    edges = edges_of(tris)
    return np.concatenate((a, b)), len(a), np.concatenate((edges, edges + len(a)))


def gaussian(direction, width, height):
    """Radial scale with a smooth bump (or dent, for a negative ``height``) around ``direction``."""
    def radial(theta, phi):
        point = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
        angle = math.acos(max(-1.0, min(1.0, point @ direction)))
        return 1.0 + height * math.exp(-(angle / width) ** 2)
    return radial


def lumps(amplitude=0.02, width=0.1, count=60, seed=3):
    """Radial scale with smooth, muscle-like lumps all over."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(count, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    heights = amplitude * rng.choice((-1.0, 1.0), size=count)

    def radial(theta, phi):
        point = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
        return 1.0 + (heights * np.exp(-(np.arccos(np.clip(centers @ point, -1, 1)) / width) ** 2)).sum()
    return radial


def stroke_across(stroke, radius=0.05, strength=0.5, passes=3):
    """Dabs 15 % of the radius apart, back and forth across the front."""
    angles = np.arange(-0.35, 0.35 + 1e-9, radius * 0.15 / (TORSO + 0.01))
    for i in range(passes):
        for angle in (angles if i % 2 == 0 else angles[::-1]):
            stroke.dab(np.array([math.sin(angle), -math.cos(angle), 0.0]) * (TORSO + 0.01), radius, strength)


def garment_tris(n):
    """The triangles of both layers of `garment` (``n`` vertices each), as the operator passes them."""
    _, tris = uv_sphere(TORSO, 144, 72)
    return np.concatenate((tris, tris + n))


def compare_auto_smooth(surface, co, n, edges, mode, tris=None):
    """(roughness, closest gap to the body, worst thickness error) without and with full Auto Smooth,
    over the band the strokes went over. With ``tris``, the stroke gets the garment's triangles."""
    unit = co[:n] / np.linalg.norm(co[:n], axis=1, keepdims=True)
    band = np.flatnonzero((np.abs(unit[:, 2]) < 0.08) & (unit[:, 1] < 0.0) & (np.abs(unit[:, 0]) < math.sin(0.3)))
    results = []
    for smooth in (0.0, 1.0):
        points = co.copy()
        stroke = fitting.FitStroke(surface, points, np.ones(len(points), bool), mode=mode, offset=0.002,
                                   layer_radius=0.02, max_distance=0.05, edges=edges, tris=tris, smooth=smooth)
        stroke_across(stroke)
        radius = np.linalg.norm(points, axis=1)
        thickness = radius[n:][band] - radius[:n][band]
        results.append((roughness(radius, edges, band), surface.nearest(points[:n][band]).distance.min(),
                        np.abs(thickness - 0.008).max()))
    return results


def check_auto_smooth(label, results, factor):
    (plain, _, _), (smooth, gap, thickness) = results
    # With triangles, a fit without Auto Smooth can already be smooth to within a hundredth of a millimetre
    # (onto lumps), and then the ratio only compares how smooth the lumps are.
    check(smooth < factor * plain or smooth < 1e-5,
          "auto smooth, {}: smoother ({:.3f} mm vs {:.3f} mm)".format(label, smooth * 1000, plain * 1000))
    check(gap > 0.002 - 5e-5, "auto smooth, {}: never closer than the offset ({:.2f} mm)".format(label, gap * 1000))
    check(thickness < 0.001, "auto smooth, {}: thickness kept (max error {:.2f} mm)".format(label, thickness * 1000))


def test_auto_smooth_pushed_dent():
    # A dent poking 13 mm into the body: pushed out, it used to leave a rim.
    co, n, edges = garment(TORSO + 0.002, gaussian(FRONT, 0.1, -0.013 / (TORSO + 0.002)))
    for tris, label in ((None, "pushed dent"), (garment_tris(n), "pushed dent, triangles")):
        check_auto_smooth(label, compare_auto_smooth(torso(), co, n, edges, 'PUSH', tris), 0.6)


def test_auto_smooth_uneven_push():
    # A garment hovering inside the offset with 1 mm of unevenness: pushed out in patches.
    co, n, edges = garment(TORSO + 0.0005)
    wobble = np.modf(np.sin(np.arange(n) * 91.17) * 43758.5453)[0] * 0.001
    co *= (1.0 + np.concatenate((wobble, wobble)) / TORSO)[:, None]
    for tris, label in ((None, "uneven push"), (garment_tris(n), "uneven push, triangles")):
        check_auto_smooth(label, compare_auto_smooth(torso(), co, n, edges, 'PUSH', tris), 0.2)


def test_auto_smooth_tighten_onto_lumps():
    # Tightened onto a lumpy body, the garment used to copy every lump.
    co, n, edges = garment(TORSO + 0.012)
    body = torso(lumps())
    for tris, label in ((None, "tighten onto lumps"), (garment_tris(n), "tighten onto lumps, triangles")):
        check_auto_smooth(label, compare_auto_smooth(body, co, n, edges, 'TIGHTEN', tris), 0.5)


def test_auto_smooth_keeps_seams_closed():
    # Vertices split along a seam (as in game meshes) must end up exactly together.
    body = bumpy_body()
    inner, tris = uv_sphere(0.97, split_seam=True)
    outer, _ = uv_sphere(1.00, split_seam=True)
    co = np.concatenate((inner, outer))
    edges = edges_of(tris)
    edges = np.concatenate((edges, edges + len(inner)))
    start = co.copy()
    side = np.array([1.0, 0.0, 0.0])  # the seam is at +X
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='PUSH', offset=0.01, layer_radius=0.02,
                               edges=edges, smooth=1.0)
    paint(stroke, side * 1.0, 0.5, dabs=60)
    key = np.round(start / 1e-7).astype(np.int64)
    _, group = np.unique(key, axis=0, return_inverse=True)
    group = group.reshape(-1)

    def widest_gap(points):
        gap = 0.0
        for g in np.flatnonzero(np.bincount(group) > 1):
            members = points[group == g]
            gap = max(gap, np.abs(members - members[0]).max())
        return gap

    check(not np.array_equal(co, start), "seam: the brush moved the mesh")
    check(widest_gap(co) <= widest_gap(start) + 1e-12,
          "auto smooth: split seam vertices stayed together (gap {:.1e}, before {:.1e})".format(widest_gap(co), widest_gap(start)))


def test_weld_groups_match_brute_force():
    rng = np.random.default_rng(7)
    base = rng.uniform(-0.1, 0.1, size=(300, 3))
    # Exact duplicates, near duplicates (some on either side of any rounding grid) and chains.
    co = np.concatenate((
        base,
        base[:40],
        base[40:80] + rng.normal(scale=2e-5, size=(40, 3)),
        base[80:90] + np.array([6e-5, 0.0, 0.0]),
        base[80:90] + np.array([12e-5, 0.0, 0.0]),
    ))
    distance = 1e-4
    weld, count = fitting.weld_groups(co, distance)
    close = np.linalg.norm(co[:, None, :] - co[None, :, :], axis=2) <= distance
    labels = np.arange(len(co))
    while True:
        spread = np.array([labels[np.flatnonzero(row)].min() for row in close])
        if np.array_equal(spread, labels):
            break
        labels = spread
    _, expected = np.unique(labels, return_inverse=True)
    same = weld[:, None] == weld[None, :]
    check(count == len(np.unique(expected)) and np.array_equal(same, expected[:, None] == expected[None, :]),
          "weld groups: same groups as a brute-force search ({} groups of {} points)".format(count, len(co)))
    check(fitting.weld_groups(co, 0.0)[1] == 300 + 40 + 20,
          "weld groups: distance 0 only joins exact duplicates")


def valley_body():
    """Two overlapping spheres side by side: a valley between them, like a cleavage."""
    co, tris = uv_sphere(0.08, 96, 48)
    shift = np.array([0.07, 0.0, 0.0])
    return body_surface(np.concatenate((co - shift, co + shift)), np.concatenate((tris, tris + len(co))))


def strap(gap):
    """A strip across the front of `valley_body`, 3.5 cm inside it, in two parts (like two objects) whose
    touching columns at x = 0 are ``gap`` apart. Returns positions, edges and the pairs of touching vertices."""
    xs = np.linspace(0.0, 0.1, 41)
    zs = np.linspace(-0.02, 0.02, 17)
    parts, edges = [], []
    for side in (-1.0, 1.0):
        x = side * (xs + gap / 2.0)
        grid = np.array([(xi, -0.035, zi) for xi in x for zi in zs])
        index = np.arange(len(grid)).reshape(len(xs), len(zs)) + sum(len(p) for p in parts)
        edges += [np.stack((index[:-1].ravel(), index[1:].ravel()), axis=1),
                  np.stack((index[:, :-1].ravel(), index[:, 1:].ravel()), axis=1)]
        parts.append(grid)
    co = np.concatenate(parts)
    half = len(parts[0])
    pairs = np.stack((np.arange(len(zs)), half + np.arange(len(zs))), axis=1)
    return co, np.concatenate(edges), pairs


def sweep_strap(stroke):
    for i in range(4):
        xs = np.linspace(-0.08, 0.08, 33)
        for x in (xs if i % 2 == 0 else xs[::-1]):
            stroke.dab(np.array([x, -0.045, 0.0]), 0.03, 0.5)


def test_seams_move_together():
    body = valley_body()
    for gap, smooth in ((0.0, 0.0), (6e-5, 0.0), (6e-5, 1.0)):
        label = "{} seam{}".format("touching" if gap == 0.0 else "0.06 mm", ", auto smooth" if smooth else "")
        results = []
        for weld_distance in (0.0, 1e-4):
            co, edges, pairs = strap(gap)
            start = co.copy()
            stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='PUSH', offset=0.002,
                                       layer_radius=0.01, edges=edges, smooth=smooth, weld_distance=weld_distance)
            sweep_strap(stroke)
            before = np.linalg.norm(start[pairs[:, 0]] - start[pairs[:, 1]], axis=1)
            after = np.linalg.norm(co[pairs[:, 0]] - co[pairs[:, 1]], axis=1)
            moved = np.linalg.norm(co[pairs[:, 0]] - start[pairs[:, 0]], axis=1).min()
            results.append((np.abs(after - before).max(), moved))
        (loose, _), (kept, moved) = results
        check(moved > 0.002, "seams, {}: the seam was pushed out ({:.1f} mm)".format(label, moved * 1000))
        check(kept < 1e-9, "seams, {}: welded, the seam kept its gap (change {:.1e} m)".format(label, kept))
        if gap:
            check(loose > 1e-4, "seams, {}: not welded, it would open ({:.2f} mm)".format(label, loose * 1000))


def test_seams_pinned_by_unmovable_side():
    # One part hidden (or unselected): the seam may not open, so the vertices on it stay put.
    body = valley_body()
    co, edges, pairs = strap(6e-5)
    start = co.copy()
    movable = np.ones(len(co), bool)
    movable[len(co) // 2:] = False
    stroke = fitting.FitStroke(body, co, movable, mode='PUSH', offset=0.002, layer_radius=0.01, edges=edges,
                               weld_distance=1e-4)
    sweep_strap(stroke)
    check(np.array_equal(co[len(co) // 2:], start[len(co) // 2:]), "seams: the unmovable part stayed put")
    check(np.array_equal(co[pairs[:, 0]], start[pairs[:, 0]]), "seams: its partners on the seam stayed with it")
    check(not np.array_equal(co[:len(co) // 2], start[:len(co) // 2]), "seams: the rest of the movable part moved")


def crease_strip():
    """A flat strip 2.5 mm apart in x, spanning the valley of `valley_body` 3 cm above its crease and
    reaching into both breasts. Returns positions, edges and the vertex index per (x, z)."""
    xs = np.linspace(-0.1, 0.1, 81)
    zs = np.linspace(-0.02, 0.02, 9)
    co = np.array([(x, -0.07, z) for x in xs for z in zs])
    index = np.arange(len(co)).reshape(len(xs), len(zs))
    edges = np.concatenate((np.stack((index[:-1].ravel(), index[1:].ravel()), axis=1),
                            np.stack((index[:, :-1].ravel(), index[:, 1:].ravel()), axis=1)))
    return co, edges, index


def test_crease_keeps_vertices_together():
    # Across a crease the closest body point jumps from one side to the other, and moving along its
    # normal slid the vertices on either side apart (Tighten) or into each other (Push Out).
    body = valley_body()
    for mode in ('PUSH', 'TIGHTEN', 'FIT'):
        co, edges, index = crease_strip()
        start = co.copy()
        stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode=mode, offset=0.001, layer_radius=0.02,
                                   max_distance=0.05, edges=edges, smooth=0.5)
        for i in range(4):
            xs = np.linspace(-0.06, 0.06, 41)
            for x in (xs if i % 2 == 0 else xs[::-1]):
                stroke.dab(np.array([x, -0.075, 0.0]), 0.03, 0.5)
        middle = len(index) // 2
        spacing = np.diff(co[index[middle - 4:middle + 5], 0], axis=0)  # between columns 1 cm around it
        slid = np.abs(co[index[middle], 0]).max()
        uneven = np.abs(np.diff(spacing, axis=0)).max()
        check(np.abs(co - start).max() > 0.002, "crease, {}: the strip moved".format(mode))
        check(slid < 1e-4, "crease, {}: the middle stayed over the crease ({:.2f} mm off)".format(mode, slid * 1000))
        check(uneven < 0.0008, "crease, {}: vertices stayed evenly spaced across it (gaps {:.2f} to {:.2f} mm)".format(
            mode, spacing.min() * 1000, spacing.max() * 1000))


THIGH = 0.07
LEG_GAP = 0.012


def tube(radius, x, segments, zs, caps=True, radial=None):
    """A triangulated vertical tube around (x, 0), closed with fans when ``caps``. ``radial(phi, z)``
    is added to its radius, ``phi`` measured from the direction toward x = 0."""
    toward = -math.copysign(1.0, x)
    co = []
    for z in zs:
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            r = radius + (radial(phi, z) if radial else 0.0)
            co.append((x + toward * r * math.cos(phi), r * math.sin(phi), z))
    index = np.arange(len(co)).reshape(len(zs), segments)
    a, b = index[:-1], np.roll(index[:-1], -1, axis=1)
    c, d = index[1:], np.roll(index[1:], -1, axis=1)
    tris = np.concatenate((np.stack((a, b, d), -1).reshape(-1, 3), np.stack((a, d, c), -1).reshape(-1, 3)))
    if caps:
        co += [(x, 0.0, zs[0]), (x, 0.0, zs[-1])]
        bottom, top = len(co) - 2, len(co) - 1
        tris = np.concatenate((tris, np.stack((np.full(segments, bottom), np.roll(index[0], -1), index[0]), -1),
                               np.stack((np.full(segments, top), index[-1], np.roll(index[-1], -1)), -1)))
    return np.array(co), np.asarray(tris, dtype=np.int64)


def legs():
    """Two thighs, upright cylinders LEG_GAP apart, with edges of about 6.5 mm like the devkit Mannequin's.
    The second tube is mirrored, so its triangles are the other way round (inside out, see
    `test_body_parts_face_out`)."""
    parts = [tube(THIGH, side * (THIGH + LEG_GAP / 2.0), 68, np.linspace(-0.12, 0.12, 37)) for side in (-1.0, 1.0)]
    return body_surface(np.concatenate((parts[0][0], parts[1][0])),
                        np.concatenate((parts[0][1], parts[1][1] + len(parts[0][0]))))


def trouser_legs(thickness=0.0):
    """Coarse trouser legs (11 mm edges) made for other thighs: toward the gap between the legs they clip
    up to 10 mm deep lower down and float 5 mm off higher up. With ``thickness``, a second layer that much
    farther out (a separate part whose vertices match the first's). Returns positions, edges, triangles,
    the leg of each vertex (-1 or 1), its angle from the gap and its layer (0 inner, 1 outer)."""
    def radial(phi, z):
        facing = math.exp(-(math.atan2(math.sin(phi), math.cos(phi)) / 0.7) ** 2)
        return 0.001 + facing * (0.005 - 0.015 * 0.5 * (1.0 - math.tanh(z / 0.015)))
    co, edges, tris, side, angle, layer = [], [], [], [], [], []
    zs = np.linspace(-0.08, 0.08, 15)
    phi = 2.0 * math.pi * np.arange(32) / 32
    for s in (-1.0, 1.0):
        for k, extra in enumerate((0.0, thickness) if thickness else (0.0,)):
            points, t = tube(THIGH, s * (THIGH + LEG_GAP / 2.0), 32, zs, caps=False,
                             radial=lambda p, z, e=extra: radial(p, z) + e)
            first = sum(len(p) for p in co)
            edges.append(edges_of(t) + first)
            tris.append(t + first)
            co.append(points)
            side.append(np.full(len(points), s))
            angle.append(np.tile(np.arctan2(np.sin(phi), np.cos(phi)), len(zs)))
            layer.append(np.full(len(points), k))
    return (np.concatenate(co), np.concatenate(edges), np.concatenate(tris), np.concatenate(side),
            np.concatenate(angle), np.concatenate(layer))


def stroke_down_the_gap(stroke):
    """Dabs down both sides of the gap between `legs`."""
    for i in range(4):
        zs = np.linspace(-0.05, 0.05, 41)
        for z in (zs if i % 2 == 0 else zs[::-1]):
            for s in (-1.0, 1.0):
                for y in (-0.015, 0.015):
                    stroke.dab(np.array([s * (LEG_GAP / 2.0 + 0.001), y, z]), 0.02, 0.5)


def fit_trousers(layer_radius=0.02, mode='FIT', thickness=0.0, smooth=0.5):
    """Fit `trouser_legs` onto `legs` down both sides of the gap, with their triangles. Returns the
    body, the positions before and after, and each vertex's leg, angle from the gap and layer."""
    body = legs()
    co, edges, tris, side, angle, layer = trouser_legs(thickness)
    start = co.copy()
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode=mode, offset=0.001, layer_radius=layer_radius,
                               max_distance=0.05, edges=edges, tris=tris, smooth=smooth)
    stroke_down_the_gap(stroke)
    return body, start, co, side, angle, layer


def flipped_faces(start, co, tris):
    """How many of ``tris`` turned over between the positions ``start`` and ``co``."""
    def normals(points):
        return np.cross(points[tris[:, 1]] - points[tris[:, 0]], points[tris[:, 2]] - points[tris[:, 0]])
    return int((np.einsum("ij,ij->i", normals(start), normals(co)) < 0.0).sum())


def test_body_parts_face_out():
    # legs() joins two tubes, the second mirrored, so its triangles are the other way round. One volume
    # sign for the whole body left that thigh facing inward: points in the gap measured as inside it.
    # Each closed part is now turned by the volume it encloses itself.
    body = legs()
    for s in (-1.0, 1.0):
        surface = s * LEG_GAP / 2.0  # the thigh's side facing the gap
        points = np.array([[surface + s * 0.004, 0.0, 0.0], [surface - s * 0.004, 0.0, 0.0]])
        distance = body.nearest(points).distance
        check(abs(distance[0] + 0.004) < 2e-4 and abs(distance[1] - 0.004) < 2e-4,
              "body parts: the {} thigh faces out ({:.2f} mm inside it, {:.2f} mm in the gap)".format(
                  "left" if s < 0 else "right", distance[0] * 1000, distance[1] * 1000))


def test_leg_gap_keep_together():
    # With the mesh's triangles, goals are shared along the mesh by a closing: clothing that clips in one
    # place and floats a little farther on fits in both, so at the default Keep Together (2 cm) nothing is
    # pushed over to the other leg (with goals shared by the largest, 6 of 82 vertices were, 3.6 mm off on
    # average). The other thigh, facing back across the gap, doesn't turn the directions either.
    for layer_radius in (0.02, 0.003):
        body, start, co, side, angle, layer = fit_trousers(layer_radius)
        gap = np.flatnonzero((np.abs(angle) < 0.35) & (np.abs(start[:, 2]) < 0.04))
        before = body.nearest(start[gap])
        near = body.nearest(co[gap])
        error = np.abs(near.distance - 0.001)
        crossed = int((np.sign(near.location[:, 0]) != side[gap]).sum())
        move = co[gap] - start[gap]
        slide = np.linalg.norm(move - np.einsum("ij,ij->i", move, before.normal)[:, None] * before.normal, axis=1)
        label = "leg gap, Keep Together {:.0f} mm".format(layer_radius * 1000)
        check(crossed == 0, "{}: every vertex stayed on its own leg ({} crossed)".format(label, crossed))
        check(error.mean() < 0.0005 and error.max() < 0.0012, "{}: fitted to the offset (off by {:.2f} mm on average, "
              "{:.2f} mm at most)".format(label, error.mean() * 1000, error.max() * 1000))
        check(slide.max() < 0.0002, "{}: nothing slid toward the middle ({:.3f} mm at most)".format(label, slide.max() * 1000))


def test_leg_gap_two_layers():
    # A second layer 3 mm out, a separate part: in the 12 mm gap its closest body point is on the other
    # thigh, so stacked by the body it moved with the wrong leg (thickness off by 3.4 mm, 20 of its vertices
    # over the other leg). Found on the mesh, it rides on the layer under it.
    body, start, co, side, angle, layer = fit_trousers(thickness=0.003)
    gap = np.flatnonzero((np.abs(angle) < 0.35) & (np.abs(start[:, 2]) < 0.04) & (layer == 0))
    outer = gap + len(co) // 4  # the outer layer of each leg follows its inner one
    center = np.stack((side[gap] * (THIGH + LEG_GAP / 2.0), np.zeros(len(gap))), axis=1)
    thickness = np.linalg.norm(co[outer, :2] - center, axis=1) - np.linalg.norm(co[gap, :2] - center, axis=1)
    inner = np.abs(body.nearest(co[gap]).distance - 0.001)
    crossed = int((np.sign(body.nearest(co[outer]).location[:, 0]) != side[gap]).sum())
    check(np.abs(thickness - 0.003).max() < 0.0002,
          "two layers in the gap: thickness kept (off by {:.3f} mm at most)".format(np.abs(thickness - 0.003).max() * 1000))
    check(inner.mean() < 0.0005, "two layers in the gap: inner layer at the offset ({:.2f} mm off)".format(inner.mean() * 1000))
    check(crossed == 0, "two layers in the gap: the outer layer stayed over its own leg ({} crossed)".format(crossed))


def test_modes_keep_to_their_side():
    # The fit itself: Push Out never pulls in and Tighten never pushes out. (Auto Smooth evens out bumps
    # either way in every mode, never closer to the body than the offset.)
    body = legs()
    for mode in ('PUSH', 'TIGHTEN'):
        _, start, co, side, angle, layer = fit_trousers(mode=mode, smooth=0.0)
        before, after = body.nearest(start).distance, body.nearest(co).distance
        gap = (np.abs(angle) < 0.35) & (np.abs(start[:, 2]) < 0.04)
        if mode == 'PUSH':
            wrong = np.maximum(before - after, 0.0)[before > 0.001].max()
            fitted = np.abs(after - 0.001)[gap & (before < 0.001)].mean()
        else:
            wrong = np.maximum(after - before, 0.0)[before < 0.001].max()
            fitted = np.abs(after - 0.001)[gap & (before > 0.001)].mean()
        check(wrong < 1e-6, "{}: nothing moved the other way ({:.4f} mm)".format(mode, wrong * 1000))
        check(fitted < 0.0006, "{}: what it fixes got to the offset ({:.2f} mm off)".format(mode, fitted * 1000))


def folded_trousers(solid):
    """Single trouser legs with a 2 cm hem folded back inside them, 1.5 mm in, through a sharp fold; or,
    ``solid``, two layers 3 mm apart closed into a solid at both ends. Their triangles continue through the
    folds, so the hem's and the inner layer's face the body. Returns positions, edges, triangles, each
    vertex's leg, angle from the gap and kind (0 outer layer, 1 hem or inner layer)."""
    zs = np.linspace(-0.08, 0.08, 15)
    count = 32
    phi = 2.0 * math.pi * np.arange(count) / count

    def radial(p, z):
        facing = math.exp(-(math.atan2(math.sin(p), math.cos(p)) / 0.7) ** 2)
        return 0.001 + facing * (0.005 - 0.015 * 0.5 * (1.0 - math.tanh(z / 0.015)))
    co, tris, side, angle, kind = [], [], [], [], []
    for s in (-1.0, 1.0):
        if solid:
            profile = [(z, 0.003, 0) for z in zs] + [(z, 0.0, 1) for z in zs[::-1]] + [(zs[0], 0.003, 0)]
        else:
            profile = [(z, 0.0, 0) for z in zs] + [(z, -0.0015, 1) for z in np.linspace(zs[-1], zs[-1] - 0.02, 5)]
        x = s * (THIGH + LEG_GAP / 2.0)
        rings = [[(x - s * (THIGH + radial(p, z) + extra) * math.cos(p), (THIGH + radial(p, z) + extra) * math.sin(p), z)
                  for p in phi] for z, extra, _ in profile]
        index = np.arange(len(rings) * count).reshape(len(rings), count) + sum(len(p) for p in co)
        a, b = index[:-1], np.roll(index[:-1], -1, axis=1)
        c, d = index[1:], np.roll(index[1:], -1, axis=1)
        tris.append(np.concatenate((np.stack((a, b, d), -1).reshape(-1, 3), np.stack((a, d, c), -1).reshape(-1, 3))))
        co.append(np.array(rings).reshape(-1, 3))
        side.append(np.full(len(rings) * count, s))
        angle.append(np.tile(np.arctan2(np.sin(phi), np.cos(phi)), len(rings)))
        kind.append(np.repeat([k for _, _, k in profile], count))
    co, tris = np.concatenate(co), np.concatenate(tris)
    return co, edges_of(tris), tris, np.concatenate(side), np.concatenate(angle), np.concatenate(kind)


def test_folded_layers():
    # A hem folded back inside and a closed solid: the folded part is a sheet of its own, which faces away
    # from the body too, so the layer over it rides on it and keeps its distance. One side per mesh part
    # (a closed solid votes both ways) misjudged the solid: 24 vertices ended inside the body.
    body = legs()
    for solid in (False, True):
        co, edges, tris, side, angle, kind = folded_trousers(solid)
        start = co.copy()
        stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='FIT', offset=0.001, layer_radius=0.02,
                                   max_distance=0.05, edges=edges, tris=tris, smooth=0.5, weld_distance=1e-5)
        stroke_down_the_gap(stroke)
        label = "solid" if solid else "hem folded inside"
        gap = np.abs(angle) < 0.35
        rise = np.linalg.norm(co[:, :2] - np.stack((side * (THIGH + LEG_GAP / 2.0), np.zeros(len(co))), axis=1), axis=1)
        rise -= np.linalg.norm(start[:, :2] - np.stack((side * (THIGH + LEG_GAP / 2.0), np.zeros(len(co))), axis=1), axis=1)
        # How much farther each vertex of the outer layer lying over the other part moved than that part
        # under it (at the same angle, interpolated to its height).
        over = np.flatnonzero(gap & (kind == 0) & ((start[:, 2] >= 0.06 - 1e-9) | solid))
        change = []
        for v in over:
            under = np.flatnonzero((kind == 1) & (side == side[v]) & (np.abs(angle - angle[v]) < 1e-6))
            under = under[np.argsort(start[under, 2])]
            change.append(abs(rise[v] - np.interp(start[v, 2], start[under, 2], rise[under])))
        change = max(change)
        inside = int((body.nearest(co[gap & (np.abs(start[:, 2]) < 0.04)]).distance < 0.0).sum())
        check(len(over) > 10 and change < 0.0002, "{}: layers kept their distance (changed {:.3f} mm at most, {} vertices)".format(
            label, change * 1000, len(over)))
        check(inside == 0, "{}: nothing stroked ended inside the body ({})".format(label, inside))


def test_glove_finger():
    # Around a limb narrower than Keep Together, a ray toward the body passes through it to the far side of
    # the glove: that isn't a layer under the near side (both sides rode on each other and stayed put).
    body = body_surface(*tube(0.007, 1e-9, 24, np.linspace(-0.04, 0.04, 33)))
    co, tris = tube(0.007, 1e-9, 24, np.linspace(-0.03, 0.03, 25), caps=False, radial=lambda p, z: 0.0025)
    start = co.copy()
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='FIT', offset=0.001, layer_radius=0.02,
                               max_distance=0.05, edges=edges_of(tris), tris=tris, smooth=0.5)
    for _ in range(3):
        for z in np.linspace(-0.02, 0.02, 21):
            for k in range(8):
                stroke.dab(np.array([0.008 * math.cos(k * math.pi / 4), 0.008 * math.sin(k * math.pi / 4), z]), 0.008, 0.5)
    middle = np.abs(start[:, 2]) < 0.015
    error = np.abs(body.nearest(co[middle]).distance - 0.001)
    check(error.mean() < 0.0002, "glove: fitted onto the finger ({:.2f} mm off on average, 1.5 mm before)".format(error.mean() * 1000))


def wrinkled_garment(base):
    """A single layer on `torso` with wrinkles ±2.5 mm deep and 2 cm apart, ``base`` from its middle."""
    k = 2.0 * math.pi * TORSO / 0.02
    co, tris = uv_sphere(base, 288, 96, lambda theta, phi: 1.0 + 0.0025 / base * math.sin(k * phi))
    unit = co / np.linalg.norm(co, axis=1, keepdims=True)
    band = (np.abs(unit[:, 2]) < 0.05) & (unit[:, 1] < 0.0) & (np.abs(unit[:, 0]) < math.sin(0.2))
    return co, tris, band


def test_auto_smooth_keeps_wrinkles():
    # Auto Smooth smoothed the mesh's height every dab, so wrinkles wore down stroke after stroke (2 cm ones
    # kept 18 to 24 % of their depth) and repeated Push Out strokes crept outward (3.7, 8.1, 11.6, then
    # 15.1 mm from the body). It now smooths the shape the fit heads for, once, by Taubin's method.
    surface = torso()
    for smooth, keep in ((0.5, 0.9), (1.0, 0.85)):
        co, tris, band = wrinkled_garment(TORSO - 0.003)
        start = co.copy()
        stroke = fitting.FitStroke(surface, co, np.ones(len(co), bool), mode='PUSH', offset=0.002, layer_radius=0.02,
                                   max_distance=0.05, edges=edges_of(tris), tris=tris, smooth=smooth)
        stroke_across(stroke)
        r0, r1 = np.linalg.norm(start[band], axis=1), np.linalg.norm(co[band], axis=1)
        kept = (r1.max() - r1.min()) / (r0.max() - r0.min())
        check(kept > keep, "auto smooth {}: wrinkles kept {:.0f} % of their depth".format(smooth, kept * 100))
    co, tris, band = wrinkled_garment(TORSO - 0.003)
    gaps = []
    for _ in range(4):
        stroke = fitting.FitStroke(surface, co, np.ones(len(co), bool), mode='PUSH', offset=0.002, layer_radius=0.02,
                                   max_distance=0.05, edges=edges_of(tris), tris=tris, smooth=0.5)
        stroke_across(stroke)
        co = stroke.co
        gaps.append(surface.nearest(co[band]).distance.min())
    # Push Out never pulls the wrinkles' crests in, so smoothing them raises their troughs a little each
    # stroke (0.14 mm).
    check(gaps[-1] < 0.0026, "auto smooth: four Push Out strokes stay near the offset ({} mm)".format(
        ", ".join("{:.2f}".format(gap * 1000) for gap in gaps)))


def wobbly_garment():
    """`garment` hovering inside the offset with 1 mm of per-vertex unevenness, and its triangles."""
    co, n, edges = garment(TORSO + 0.0005)
    wobble = np.modf(np.sin(np.arange(n) * 91.17) * 43758.5453)[0] * 0.001
    co *= (1.0 + np.concatenate((wobble, wobble)) / TORSO)[:, None]
    _, tris = uv_sphere(TORSO + 0.0005, 144, 72)
    return co, n, edges, np.concatenate((tris, tris + n))


def test_auto_smooth_amount_is_continuous():
    # Auto Smooth ran a whole number of Taubin pairs, so the slider had four steps (0.05 smoothed as much as
    # 0.25). The last pair now runs in part: every bit more smooths a bit more.
    surface = torso()
    rough = []
    for smooth in (0.0, 0.05, 0.25, 0.5, 1.0):
        co, n, edges, tris = wobbly_garment()
        stroke = fitting.FitStroke(surface, co, np.ones(len(co), bool), mode='PUSH', offset=0.002, layer_radius=0.02,
                                   max_distance=0.05, edges=edges, tris=tris, smooth=smooth)
        stroke_across(stroke)
        unit = co[:n] / np.linalg.norm(co[:n], axis=1, keepdims=True)
        band = np.flatnonzero((np.abs(unit[:, 2]) < 0.08) & (unit[:, 1] < 0.0) & (np.abs(unit[:, 0]) < math.sin(0.3)))
        rough.append(roughness(np.linalg.norm(co, axis=1), edges, band))
    check(all(b < 0.95 * a for a, b in zip(rough, rough[1:])), "auto smooth: every step of the slider smooths more ({} mm)".format(
        ", ".join("{:.3f}".format(r * 1000) for r in rough)))


def test_crease_bridged_cleanly():
    # Fitted with the mesh's triangles, the strip spanning the valley is fitted onto both breasts and
    # bridges the cleavage instead of being pulled into it, without turning any face over.
    body = valley_body()
    for mode in ('TIGHTEN', 'FIT'):
        co, edges, index = crease_strip()
        a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
        tris = np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))
        start = co.copy()
        stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode=mode, offset=0.001, layer_radius=0.02,
                                   max_distance=0.05, edges=edges, tris=tris, smooth=0.5)
        for i in range(4):
            xs = np.linspace(-0.06, 0.06, 41)
            for x in (xs if i % 2 == 0 else xs[::-1]):
                stroke.dab(np.array([x, -0.075, 0.0]), 0.03, 0.5)
        middle = len(index) // 2
        distance = body.nearest(co).distance
        on_breasts = (np.abs(start[:, 0]) > 0.0245) & (np.abs(start[:, 0]) < 0.0285)
        flipped = flipped_faces(start, co, tris)
        check(flipped == 0, "crease, {}: no face turned over ({})".format(mode, flipped))
        check(np.abs(distance[on_breasts] - 0.001).max() < 0.0003, "crease, {}: fitted onto the breasts ({:.2f} to {:.2f} mm)".format(
            mode, distance[on_breasts].min() * 1000, distance[on_breasts].max() * 1000))
        check(distance[index[middle]].min() > 0.005, "crease, {}: the cleavage is bridged ({:.1f} mm over it)".format(
            mode, distance[index[middle]].min() * 1000))
        check(np.abs(co[index[middle], 0]).max() < 1e-4, "crease, {}: the middle stayed over the crease".format(mode))


def test_dab_reaching_only_riders():
    # A fine outer shell over a coarse lining (triangles about 6 cm across): a big dab aims everything, then
    # a small one at the same spot, or one 2 cm off the shell, reaches only shell vertices riding on lining
    # triangles out of its reach. That raised an IndexError (after the stroke's moves were written).
    surface = torso()
    step = 2.0 * math.pi / 16
    theta, phi = math.pi / 2 + step / 2, 1.5 * math.pi + step / 2
    spot = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]) * (TORSO + 0.012)
    for segments, dabs in ((16, ((spot, 0.05), (spot, 0.01))), (144, ((FRONT * (TORSO + 0.012), 0.04), (FRONT * (TORSO + 0.032), 0.005)))):
        a, inner = uv_sphere(TORSO + 0.002, segments, segments // 2)
        b, outer = uv_sphere(TORSO + 0.012, 144, 72)
        co = np.concatenate((a, b))
        tris = np.concatenate((inner, outer + len(a)))
        stroke = fitting.FitStroke(surface, co, np.ones(len(co), bool), mode='FIT', offset=0.002, layer_radius=0.02,
                                   max_distance=0.05, edges=edges_of(tris), tris=tris, smooth=0.5)
        error = None
        try:
            for center, radius in dabs:
                stroke.dab(center, radius, 0.5)
        except Exception as caught:
            error = repr(caught)
        check(error is None, "riders only, lining of {} segments: the dabs went through{}".format(
            segments, "" if error is None else " ({})".format(error)))


def loose_leg(bulge):
    """A single left trouser leg whose side facing the gap bulges ``bulge`` (plus 1 mm) off its thigh, with
    its triangles and each vertex's angle from the gap."""
    def radial(phi, z):
        return 0.001 + bulge * math.exp(-(math.atan2(math.sin(phi), math.cos(phi)) / 0.5) ** 2)
    zs = np.linspace(-0.08, 0.08, 15)
    co, tris = tube(THIGH, -(THIGH + LEG_GAP / 2.0), 32, zs, caps=False, radial=radial)
    phi = 2.0 * math.pi * np.arange(32) / 32
    return co, tris, np.tile(np.arctan2(np.sin(phi), np.cos(phi)), len(zs))


def test_leg_past_the_middle_of_the_gap():
    # A loose trouser leg whose side facing the gap lies past its middle (10 mm off its own thigh, 2 mm off
    # the other one) or has sunk 2 mm into the other thigh. Its closest body point is on the other thigh:
    # with Keep Together at a few millimetres its direction came from there, and sunk in, it slid up to
    # 19 mm sideways and stayed inside. Fit and Tighten now settle it on its own thigh at any Keep
    # Together; Push Out, which never pulls in, gets it out of the other thigh the shortest way.
    body = legs()
    own_axis = np.array([-(THIGH + LEG_GAP / 2.0), 0.0])
    for bulge, mode, layer_radius in ((0.009, 'FIT', 0.003), (0.009, 'FIT', 0.02), (0.013, 'FIT', 0.02),
                                      (0.013, 'TIGHTEN', 0.005), (0.013, 'PUSH', 0.02)):
        co, tris, angle = loose_leg(bulge)
        start = co.copy()
        stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode=mode, offset=0.001, layer_radius=layer_radius,
                                   max_distance=0.05, edges=edges_of(tris), tris=tris, smooth=0.5)
        for i in range(4):
            zs = np.linspace(-0.05, 0.05, 41)
            for z in (zs if i % 2 == 0 else zs[::-1]):
                for y in (-0.015, 0.0, 0.015):
                    stroke.dab(np.array([-(LEG_GAP / 2.0 + 0.001), y, z]), 0.02, 0.5)
        gap = np.flatnonzero((np.abs(angle) < 0.35) & (np.abs(start[:, 2]) < 0.04))
        near = body.nearest(co[gap])
        inside = int((near.distance < 0.0).sum())
        label = "leg {:.0f} mm past its thigh, {}, Keep Together {:.0f} mm".format((bulge + 0.001) * 1000, mode, layer_radius * 1000)
        check(inside == 0, "{}: nothing left inside the body ({})".format(label, inside))
        if mode == 'PUSH':
            slide = np.abs(co[gap, 1] - start[gap, 1]).max()
            check(slide < 0.0005, "{}: out the shortest way ({:.2f} mm sideways)".format(label, slide * 1000))
            continue
        own = np.linalg.norm(co[gap, :2] - own_axis, axis=1) - THIGH
        crossed = int((near.location[:, 0] > 0.0).sum())
        check(crossed == 0 and np.abs(own - 0.001).max() < 0.001, "{}: on its own thigh ({:.2f} to {:.2f} mm off it, {} "
              "over the other one)".format(label, own.min() * 1000, own.max() * 1000, crossed))


def test_keep_together_is_a_distance():
    # Keep Together's closing counted rings of the median edge of everything in Edit Mode: a dense detail
    # anywhere (a 1 mm grid far from the stroke) made it reach about 10 cm over coarse trouser legs, 2.3 mm
    # off in the gap instead of 0.3. Now it reaches as far along every part of the mesh.
    errors = []
    for dense in (False, True):
        body = legs()
        co, edges, tris, side, angle, layer = trouser_legs()
        count = len(co)
        if dense:
            spacing = np.arange(70) * 0.001
            grid = np.array([(0.3 + x, 0.3 + y, 0.3) for x in spacing for y in spacing])
            index = np.arange(len(grid)).reshape(70, 70)
            a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
            tris = np.concatenate((tris, np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1))) + count))
            co = np.concatenate((co, grid))
        start = co.copy()
        stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='FIT', offset=0.001, layer_radius=0.02,
                                   max_distance=0.05, edges=edges_of(tris), tris=tris, smooth=0.5)
        stroke_down_the_gap(stroke)
        gap = np.flatnonzero((np.abs(angle) < 0.35) & (np.abs(start[:count, 2]) < 0.04))
        errors.append(np.abs(body.nearest(co[gap]).distance - 0.001).mean())
    check(abs(errors[1] - errors[0]) < 1e-5 and errors[1] < 0.0005, "keep together: a dense detail elsewhere changes "
          "nothing ({:.3f} mm off without it, {:.3f} mm with it)".format(errors[0] * 1000, errors[1] * 1000))


def test_loose_bulge_tightened():
    # Keep Together's closing kept loose bulges up to 2 to 3 times as wide as it as if they were folds
    # (a bulge 4 cm wide at 2 cm: 4.2 of its 5 mm left). It now keeps dips up to about 1.5 times as wide.
    surface = torso()
    left = []
    for width in (0.015, 0.04):
        base = TORSO + 0.001
        sigma = width / (2.0 * math.sqrt(math.log(2.0))) / base
        co, tris = uv_sphere(base, 288, 144, gaussian(FRONT, sigma, 0.005 / base))
        stroke = fitting.FitStroke(surface, co, np.ones(len(co), bool), mode='FIT', offset=0.001, layer_radius=0.02,
                                   max_distance=0.05, edges=edges_of(tris), tris=tris, smooth=0.5, weld_distance=1e-4)
        stroke_across(stroke)
        peak = np.flatnonzero(angle_from(co, FRONT) < 0.3 * sigma + 1e-3)
        left.append(surface.nearest(co[peak]).distance.max() - 0.001)
    check(left[1] < 0.002, "loose bulge 4 cm wide: tightened ({:.1f} of 5 mm left)".format(left[1] * 1000))
    check(left[0] > 0.003, "fold 1.5 cm wide: kept ({:.1f} of 5 mm)".format(left[0] * 1000))


def rounded_solid_trousers(thickness, rim_segments):
    """Trouser legs closed into solids ``thickness`` thick (the inner wall where `trouser_legs`'s are), their
    rims rounded over ``rim_segments`` segments: one smooth sheet per leg, whose inside faces the body and
    outside away. Returns positions, triangles, each vertex's leg and angle from the gap, and whether it lies
    on the inner wall."""
    def radial(p, z):
        facing = math.exp(-(math.atan2(math.sin(p), math.cos(p)) / 0.7) ** 2)
        return 0.001 + facing * (0.005 - 0.015 * 0.5 * (1.0 - math.tanh(z / 0.015)))
    zs = np.linspace(-0.08, 0.08, 15)
    turns = [math.pi * i / rim_segments for i in range(1, rim_segments)]
    # Up the outer wall, over the top rim, down the inner wall and round the bottom rim: (z, extra, inner).
    profile = [(z, thickness, False) for z in zs]
    profile += [(zs[-1] + 0.5 * thickness * math.sin(t), 0.5 * thickness * (1.0 + math.cos(t)), False) for t in turns]
    profile += [(z, 0.0, True) for z in zs[::-1]]
    profile += [(zs[0] - 0.5 * thickness * math.sin(t), 0.5 * thickness * (1.0 - math.cos(t)), False) for t in turns]
    count = 32
    phi = 2.0 * math.pi * np.arange(count) / count
    co, tris, side, angle, inner = [], [], [], [], []
    for s in (-1.0, 1.0):
        x = s * (THIGH + LEG_GAP / 2.0)
        rings = [[(x - s * (THIGH + radial(p, z) + extra) * math.cos(p), (THIGH + radial(p, z) + extra) * math.sin(p), z)
                  for p in phi] for z, extra, _ in profile]
        first = sum(len(c) for c in co)
        index = np.arange(len(rings) * count).reshape(len(rings), count) + first
        index = np.concatenate((index, index[:1]))  # round to the first ring again
        a, b = index[:-1], np.roll(index[:-1], -1, axis=1)
        c, d = index[1:], np.roll(index[1:], -1, axis=1)
        tris.append(np.concatenate((np.stack((a, b, d), -1).reshape(-1, 3), np.stack((a, d, c), -1).reshape(-1, 3))))
        co.append(np.array(rings).reshape(-1, 3))
        side.append(np.full(len(rings) * count, s))
        angle.append(np.tile(np.arctan2(np.sin(phi), np.cos(phi)), len(rings)))
        inner.append(np.repeat([on for _, _, on in profile], count))
    return np.concatenate(co), np.concatenate(tris), np.concatenate(side), np.concatenate(angle), np.concatenate(inner)


def test_rounded_solid():
    # An 8 mm solid trouser leg whose rims are rounded is one smooth sheet: its inside faces the body and its
    # outside away, so it faced no way in particular, and the gap fell back to the old directions (2.1 mm
    # off on average, 4 of 42 vertices over the other leg). Its walls now decide the way each faces.
    body = legs()
    for segments in (1, 8):
        co, tris, side, angle, inner = rounded_solid_trousers(0.008, segments)
        start = co.copy()
        stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='FIT', offset=0.001, layer_radius=0.02,
                                   max_distance=0.05, edges=edges_of(tris), tris=tris, smooth=0.5)
        stroke_down_the_gap(stroke)
        gap = np.flatnonzero((np.abs(angle) < 0.35) & (np.abs(start[:, 2]) < 0.04) & inner)
        near = body.nearest(co[gap])
        error = np.abs(near.distance - 0.001)
        crossed = int((np.sign(near.location[:, 0]) != side[gap]).sum())
        check(error.mean() < 0.0005 and crossed == 0, "solid with rims of {} segment(s): inner wall fitted ({:.2f} mm off, "
              "{} over the other leg)".format(segments, error.mean() * 1000, crossed))


def test_layer_far_over_a_sunk_panel():
    # An outer layer 2 mm over its lining, but where the lining has sunk 11 mm into the body (an overlapping
    # panel pushed in): there the gap between them isn't the clothing's thickness. Pushing the lining out
    # carried the outer layer 12 mm farther off the skin; now it only keeps twice the layers' usual
    # thickness over the lining, and moves with the part of it riding on the lining around it.
    surface = torso()
    lining, tris = uv_sphere(TORSO + 0.001, 144, 72, gaussian(FRONT, 0.08, -0.012 / (TORSO + 0.001)))
    outer, _ = uv_sphere(TORSO + 0.003, 144, 72)
    co = np.concatenate((lining, outer))
    count = len(lining)
    tris = np.concatenate((tris, tris + count))
    start = co.copy()
    stroke = fitting.FitStroke(surface, co, np.ones(len(co), bool), mode='PUSH', offset=0.001, layer_radius=0.02,
                               max_distance=0.05, edges=edges_of(tris), tris=tris, smooth=0.5)
    stroke_across(stroke)
    over = np.flatnonzero(angle_from(start[:count], FRONT) < 0.1)
    rise = np.linalg.norm(co[count:][over], axis=1) - np.linalg.norm(start[count:][over], axis=1)
    apart = np.linalg.norm(co[count:][over], axis=1) - np.linalg.norm(co[:count][over], axis=1)
    lowest = surface.nearest(co[:count][over]).distance.min()
    check(lowest > 0.0009, "sunk panel: pushed out of the body ({:.2f} mm)".format(lowest * 1000))
    check(rise.max() < 0.006 and apart.min() > 0.0019, "sunk panel: the layer over it stays near the skin and clear of "
          "it (rose {:.1f} mm at most, {:.1f} mm over it at least)".format(rise.max() * 1000, apart.min() * 1000))


def test_hidden_inner_layer_holds_the_outer_one():
    # With the inner layer hidden, the outer one was fitted on its own and pulled 4 mm through it. It now
    # stays clear of the layer under it, though that one may not move.
    surface = torso()
    co, count, edges = garment(TORSO + 0.006)
    _, tris = uv_sphere(TORSO + 0.006, 144, 72)
    movable = np.zeros(len(co), bool)
    movable[count:] = True
    start = co.copy()
    stroke = fitting.FitStroke(surface, co, movable, mode='FIT', offset=0.002, layer_radius=0.02, max_distance=0.05,
                               edges=edges, tris=np.concatenate((tris, tris + count)), smooth=0.5)
    stroke_across(stroke)
    unit = start[:count] / np.linalg.norm(start[:count], axis=1, keepdims=True)
    band = np.flatnonzero((np.abs(unit[:, 2]) < 0.08) & (unit[:, 1] < 0.0) & (np.abs(unit[:, 0]) < math.sin(0.3)))
    apart = np.linalg.norm(co[count:][band], axis=1) - np.linalg.norm(co[:count][band], axis=1)
    check(apart.min() > 0.0075, "hidden inner layer: the outer one isn't pulled through it ({:.2f} mm over it)".format(
        apart.min() * 1000))


def test_open_cup_keeps_its_side():
    # A separate open cup sunk into the body, facing its hollow (a lining of a hollow of the body), was
    # turned inside out by its own volume: points in the hollow read as inside the body. Only closed parts
    # are turned by their own volume.
    sphere, sphere_tris = uv_sphere(0.15, 96, 48)
    center, radius, segments, rings = np.array([0.0, -0.15, 0.0]), 0.02, 32, 8
    co = [center + np.array([0.0, radius, 0.0])]
    for i in range(1, rings + 1):
        theta = 0.5 * math.pi * i / rings
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            co.append(center + radius * np.array([math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi)]))
    tris = []
    for j in range(segments):
        tris.append((0, 1 + (j + 1) % segments, 1 + j))
        for i in range(rings - 1):
            a, b = 1 + i * segments + j, 1 + i * segments + (j + 1) % segments
            tris += [(a, b, b + segments), (a, b + segments, a + segments)]
    co, tris = np.array(co), np.array(tris)
    normals = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    if np.einsum("ij,ij->", normals, center - co[tris].mean(axis=1)) < 0.0:
        tris = tris[:, ::-1]
    body = body_surface(np.concatenate((sphere, co)), np.concatenate((sphere_tris, tris + len(sphere))))
    distance = body.nearest(np.array([[0.0, -0.135, 0.0], [0.004, -0.136, 0.003]])).distance
    check((distance > 0.0).all(), "open cup: points in its hollow lie outside the body ({} mm)".format(
        ", ".join("{:.2f}".format(d * 1000) for d in distance)))


# Round two: clothing lying along the body slides less, loose clothing spans valleys. Round three: neither
# tears the mesh apart nor bunches it up at a crease.


def valley_side_patch(height=0.006):
    """A patch lying along the right sphere of `valley_body`, ``height`` off it, from just beside the valley
    (5 mm right of its middle) to the front of the sphere: it faces the way the sphere does under it, while
    the left sphere's side lies within Keep Together of its edge by the valley. Returns positions, edges
    and triangles."""
    center, radius = np.array([0.07, 0.0, 0.0]), 0.08 + height
    lowest = math.acos((0.005 - 0.07) / radius)
    azimuths = -np.linspace(lowest, 0.5 * math.pi, 25)
    elevations = np.linspace(-0.2, 0.2, 9)
    co = np.array([center + radius * np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
                   for a in azimuths for e in elevations])
    index = np.arange(len(co)).reshape(len(azimuths), len(elevations))
    a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    tris = np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))
    return co, edges_of(tris), tris


def test_mesh_along_the_body_slides_less():
    # Clothing lying along the body (it faces the way the body under it does) has no crease to span: its
    # vertices turn from the body's normal smoothed over Keep Together toward the normal at their closest
    # points, as far as that moves where each settles by a tenth of its edges' length. Along the smoothed
    # normal alone, this patch slid up to 1.07 mm toward the valley; now 0.73 mm. (Round two followed the
    # closest normal fully and let such vertices settle below the smoothed body: the patch moved straight
    # in, but clothing rounding off or dipping into a crease tore at the crease line, see
    # test_dress_dipping_into_a_cleft.) The smoothed body, which fills the valley, still holds the patch's
    # edge by the valley up to 1.1 mm short of the offset; from 1 cm beside it on, it settles at the offset.
    body = valley_body()
    co, edges, tris = valley_side_patch()
    start = co.copy()
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='FIT', offset=0.001, layer_radius=0.02,
                               max_distance=0.05, edges=edges, tris=tris, smooth=0.5)
    for i in range(3):
        for p in (start[4::9] if i % 2 == 0 else start[4::9][::-1]):
            stroke.dab(p, 0.03, 0.5)
    before = body.nearest(start)
    after = body.nearest(co)
    move = co - start
    slide = np.linalg.norm(move - np.einsum("ij,ij->i", move, before.normal)[:, None] * before.normal, axis=1)
    error = np.abs(after.distance - 0.001)
    beside = start[:, 0] > 0.01
    check(error[beside].max() < 0.0003, "along the body: settled at the offset beside the valley (off by {:.2f} mm "
          "at most)".format(error[beside].max() * 1000))
    check(error.max() < 0.0015, "along the body: held up by the valley at most {:.2f} mm".format(error.max() * 1000))
    check(slide.max() < 0.0009, "along the body: slid less toward the valley ({:.2f} mm sideways at most)".format(
        slide.max() * 1000))

def skirt_loop(float_=0.004, spacing=0.006):
    """A skirt around both of `legs`' thighs, ``float_`` off their fronts and backs: a loop of straight
    panels over both thighs joined by half circles around their outer sides, 16 cm tall, with its
    triangles. Returns positions, edges, triangles and the vertex index per (row, column); column 0 is in
    front of the middle of the gap."""
    c = THIGH + LEG_GAP / 2.0
    rs = THIGH + float_
    line, arc = 2.0 * c, math.pi * rs
    total = 2.0 * line + 2.0 * arc
    n = int(round(total / spacing / 2.0)) * 2
    outline = []
    for t in (np.arange(n) * total / n + c) % total:
        if t < line:
            outline.append((-c + t, -rs))
        elif t < line + arc:
            a = -math.pi / 2 + (t - line) / rs
            outline.append((c + rs * math.cos(a), rs * math.sin(a)))
        elif t < 2 * line + arc:
            outline.append((c - (t - line - arc), rs))
        else:
            a = math.pi / 2 + (t - 2 * line - arc) / rs
            outline.append((-c + rs * math.cos(a), rs * math.sin(a)))
    zs = np.linspace(-0.08, 0.08, 21)
    co = np.array([(x, y, z) for z in zs for x, y in outline])
    index = np.arange(len(co)).reshape(len(zs), n)
    a, b = index[:-1], np.roll(index[:-1], -1, axis=1)
    c2, d = index[1:], np.roll(index[1:], -1, axis=1)
    tris = np.concatenate((np.stack((a, b, d), -1).reshape(-1, 3), np.stack((a, d, c2), -1).reshape(-1, 3)))
    normals = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    if np.einsum("ij,ij->", normals[:, :2], co[tris].mean(axis=1)[:, :2]) < 0.0:
        tris = tris[:, ::-1].copy()
    return co, edges_of(tris), tris, index


def test_skirt_spans_the_leg_gap():
    # A skirt's front panel spans the 15 cm wide valley between the fronts of both thighs. Fitted onto
    # them, it was pulled 20 mm into the gap, the vertices beside the middle sliding 16 mm apart toward
    # either thigh, and its edges over the middle of the gap stretched to 3.06 times their length; it
    # settles on the thighs and stays nearer the line over them across the gap. (Round two kept those
    # edges under 2.2 times by averaging the body under clothing far off it over as far again, which made
    # other skirts sag and slide more: up to 2.33 now.) At Keep Together 4 cm, the panel held up over the
    # gap held the skirt up to 3.7 mm off the thighs beside it (0.46 mm on average).
    body = legs()
    for mode, layer_radius in (('FIT', 0.02), ('TIGHTEN', 0.02), ('FIT', 0.04)):
        label = mode if layer_radius == 0.02 else "{}, Keep Together {:.0f} mm".format(mode, layer_radius * 1000)
        co, edges, tris, index = skirt_loop()
        start = co.copy()
        stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode=mode, offset=0.001,
                                   layer_radius=layer_radius, max_distance=0.05, edges=edges, tris=tris, smooth=0.5)
        for i in range(3):
            for z in np.arange(-0.04, 0.04 + 1e-9, 0.02):
                xs = np.arange(-0.09, 0.09 + 1e-9, 0.025 * 0.15)
                for x in (xs if i % 2 == 0 else xs[::-1]):
                    stroke.dab(np.array([x, -(THIGH + 0.002), z]), 0.025, 0.5)
        front = (start[:, 1] < -0.9 * THIGH) & (np.abs(start[:, 2]) < 0.045)
        gap = front & (np.abs(start[:, 0]) < LEG_GAP / 2.0 + 0.01)
        thighs = front & (np.abs(start[:, 0]) > LEG_GAP / 2.0 + 0.03) & (np.abs(start[:, 0]) < THIGH + LEG_GAP / 2.0 + 0.02)
        sag = co[gap, 1] + THIGH + 0.001
        distance = body.nearest(co[thighs]).distance
        check(sag.max() < 0.015, "skirt, {}: spans the gap ({:.1f} mm behind the line over the thighs)".format(
            label, sag.max() * 1000))
        check(np.abs(distance - 0.001).mean() < 0.0002, "skirt, {}: settles on the thighs ({:.2f} mm off on average)".format(
            label, np.abs(distance - 0.001).mean() * 1000))
        row = index[len(index) // 2]
        row = row[(start[row, 1] < -0.9 * THIGH) & (np.abs(start[row, 0]) < LEG_GAP / 2.0 + 0.02)]
        row = row[np.argsort(start[row, 0])]
        stretch = (np.linalg.norm(np.diff(co[row], axis=0), axis=1) / np.linalg.norm(np.diff(start[row], axis=0), axis=1)).max()
        check(stretch < 2.5, "skirt, {}: not torn apart over the gap (edges up to {:.2f} times as long)".format(label, stretch))

def valley_front(x):
    """How far in front of its middle (-y) the front of `valley_body` lies at ``x``."""
    return -math.sqrt(max(0.08 ** 2 - (abs(x) - 0.07) ** 2, 0.0))


def tight_crease_strip(height=0.004):
    """A strip following the valley of `valley_body` down into it and up both breasts, ``height`` off them
    (along the spheres' normals) and 12 cm wide, with its triangles. Returns positions, edges and
    triangles."""
    xs = np.linspace(-0.06, 0.06, 161)
    zs = np.linspace(-0.01, 0.01, 9)
    co = []
    for x in xs:
        p = np.array([x, valley_front(x), 0.0])
        normal = p - np.array([math.copysign(0.07, x), 0.0, 0.0])
        normal /= np.linalg.norm(normal)
        co += [(x, p[1] - height / abs(normal[1]), z) for z in zs]
    co = np.array(co)
    index = np.arange(len(co)).reshape(len(xs), len(zs))
    a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    tris = np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))
    return co, edges_of(tris), tris


def test_tight_suit_in_a_crease():
    # Clothing that follows a crease 4 mm off it, crease and all: its sides lie along the breasts and turn
    # toward their normals, so they slide less toward the crease (up to 0.54 mm along the body's normal
    # smoothed over Keep Together 2 cm, now 0.39 mm; at Keep Together 5 mm they move straight in). Round two
    # let them follow those normals fully and settle below the body smoothed over Keep Together, which
    # fills the crease: the bottom settled at the offset, but the strip was pulled apart at the crease
    # line, its edges there stretched to 2.62 times their length (2.83 at Keep Together 5 mm; 1.74 and 2.36
    # without following at all, 1.79 and 2.42 now). Now the smoothed body still holds the bottom up about
    # 0.9 mm at Keep Together 2 cm; at 5 mm it settles within 0.25 mm.
    body = valley_body()
    for layer_radius, held, slid in ((0.02, 0.001, 0.00045), (0.005, 0.0005, 0.0001)):
        co, edges, tris = tight_crease_strip()
        start = co.copy()
        stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='FIT', offset=0.001,
                                   layer_radius=layer_radius, max_distance=0.05, edges=edges, tris=tris, smooth=0.5)
        for i in range(4):
            for x in (np.linspace(-0.04, 0.04, 33) if i % 2 == 0 else np.linspace(0.04, -0.04, 33)):
                stroke.dab(np.array([x, valley_front(x) - 0.002, 0.0]), 0.02, 0.5)
        before = body.nearest(start)
        move = co - start
        slide = np.linalg.norm(move - np.einsum("ij,ij->i", move, before.normal)[:, None] * before.normal, axis=1)
        bottom = np.abs(start[:, 0]) < 0.012
        sides = (np.abs(start[:, 0]) > 0.006) & (np.abs(start[:, 0]) < 0.03)
        error = np.abs(body.nearest(co[bottom]).distance - 0.001)
        stretch = (np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1) /
                   np.linalg.norm(start[edges[:, 0]] - start[edges[:, 1]], axis=1)).max()
        label = "tight crease, Keep Together {:.0f} mm".format(layer_radius * 1000)
        check(error.mean() < held, "{}: settled toward it ({:.2f} mm off on average)".format(label, error.mean() * 1000))
        check(slide[sides].max() < slid, "{}: its sides slid little toward it ({:.2f} mm sideways at most)".format(
            label, slide[sides].max() * 1000))
        check(stretch < 2.5, "{}: not pulled apart at the crease line (edges up to {:.2f} times as long)".format(
            label, stretch))


def cleft_dress(gap=0.003, dip=0.015, spacing=0.004):
    """A dress panel over the fronts of `valley_body`'s spheres, ``gap`` off them, level across the cleft
    between them but for a dip ``dip`` deep into it (2 cm wide), with its triangles facing front. Returns
    positions, edges and triangles."""
    xs = np.arange(-0.13, 0.13 + 1e-9, spacing)
    zs = np.arange(-0.05, 0.05 + 1e-9, spacing)
    co = []
    for z in zs:
        rz = math.sqrt(max(0.08 ** 2 - z * z, 0.0))
        for x in xs:
            dx = abs(x) - 0.07
            front = -math.sqrt(max(rz * rz - dx * dx, 0.0)) if abs(dx) < rz else 0.0
            y = min(front - gap, -(rz + gap)) if abs(x) > 0.07 else -(rz + gap)
            if abs(x) < 0.07:
                y += dip * math.exp(-(x / 0.02) ** 2)
            co.append((x, y, z))
    co = np.array(co)
    index = np.arange(len(co)).reshape(len(zs), len(xs))
    a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    tris = np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))
    normals = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    if normals[:, 1].sum() > 0.0:
        tris = tris[:, ::-1].copy()
    return co, edges_of(tris), tris


def test_dress_dipping_into_a_cleft():
    # The sides of the dip face the way the spheres under them do. Round two let them follow the normals at
    # their closest points fully, and their neighbours a little farther from the cleft less: edges at the
    # bottom of the dip were pulled to up to 2.51 times their length. Following only as far as that moves
    # where each vertex settles by a tenth of its edges' length, up to 1.95 times (1.90 without following
    # at all).
    body = valley_body()
    co, edges, tris = cleft_dress()
    start = co.copy()
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='FIT', offset=0.001, layer_radius=0.02,
                               max_distance=0.05, edges=edges, tris=tris, smooth=0.5)
    for i in range(3):
        for z in (-0.02, 0.0, 0.02):
            xs = np.arange(-0.10, 0.10 + 1e-9, 0.03 * 0.15)
            for x in (xs if i % 2 == 0 else xs[::-1]):
                stroke.dab(np.array([x, -0.085, z]), 0.03, 0.5)
    middle = (np.abs(start[:, 2]) < 0.03) & (np.abs(start[:, 0]) < 0.1)
    inner = edges[middle[edges].any(axis=1)]
    ratio = (np.linalg.norm(co[inner[:, 0]] - co[inner[:, 1]], axis=1) /
             np.linalg.norm(start[inner[:, 0]] - start[inner[:, 1]], axis=1))
    contact = (np.abs(start[:, 2]) < 0.03) & (np.abs(np.abs(start[:, 0]) - 0.07) < 0.02)
    distance = body.nearest(co[contact]).distance
    check(ratio.max() < 2.2, "dress dipping into a cleft: not torn at its bottom (edges up to {:.2f} times as "
          "long)".format(ratio.max()))
    check(np.abs(distance - 0.001).mean() < 0.0002, "dress dipping into a cleft: settles on the spheres ({:.2f} mm "
          "off on average)".format(np.abs(distance - 0.001).mean() * 1000))


def test_sunk_strip_in_a_crease():
    # The crease strip sunk 4 mm into the valley: getting out along the normals at their closest points,
    # vertices on both sides of the crease head into each other. Round two had them follow those normals
    # once they were aimed, and get out along the normal of the side they were in every dab: edges at the
    # crease were squeezed to 0.11 times their length. Along the body's normal smoothed over Keep Together,
    # they keep 0.61 of it.
    body = valley_body()
    for mode in ('PUSH', 'FIT'):
        co, edges, tris = tight_crease_strip(-0.004)
        start = co.copy()
        stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode=mode, offset=0.001, layer_radius=0.02,
                                   max_distance=0.05, edges=edges, tris=tris, smooth=0.5)
        for i in range(4):
            for x in (np.linspace(-0.06, 0.06, 49) if i % 2 == 0 else np.linspace(0.06, -0.06, 49)):
                stroke.dab(np.array([x, valley_front(x) - 0.002, 0.0]), 0.02, 0.5)
        middle = np.abs(start[:, 0]) < 0.04
        inner = edges[middle[edges].all(axis=1)]
        ratio = (np.linalg.norm(co[inner[:, 0]] - co[inner[:, 1]], axis=1) /
                 np.linalg.norm(start[inner[:, 0]] - start[inner[:, 1]], axis=1))
        inside = int((body.nearest(co[middle]).distance < 0.0).sum())
        check(ratio.min() > 0.5, "sunk crease strip, {}: not bunched up at the crease (edges down to {:.2f} times as "
              "long)".format(mode, ratio.min()))
        check(inside <= 10, "sunk crease strip, {}: got out of the body ({} of {} left inside)".format(
            mode, inside, int(middle.sum())))


SLEEVE_ROWS = np.linspace(-0.08, 0.08, 33)  # 5 mm apart
SLEEVE_SEGMENTS = 48


def hidden_top_sleeve(mode, fade_distance, depth, outer=False, strokes=2):
    """A sleeve around a thigh, ``depth`` from its skin, with its rows above z = 2 cm hidden, sculpted all
    over in ``strokes`` strokes (``outer``: with a separate layer 3 mm farther out over the rest). Returns how far
    each row moved (in m) after every stroke, whether the hidden rows stayed put, and the thickness range
    between the layers where the inner one shows."""
    body = body_surface(*tube(THIGH, 0.0, 68, np.linspace(-0.12, 0.12, 37)))
    co, tris = tube(THIGH + depth, 0.0, SLEEVE_SEGMENTS, SLEEVE_ROWS, caps=False)
    count = len(co)
    edges = edges_of(tris)
    shown = SLEEVE_ROWS.repeat(SLEEVE_SEGMENTS) < 0.0201
    movable = shown.copy()
    if outer:
        layer, tris = tube(THIGH + depth + 0.003, 0.0, SLEEVE_SEGMENTS, SLEEVE_ROWS[SLEEVE_ROWS < 0.0201], caps=False)
        co = np.concatenate((co, layer))
        edges = np.concatenate((edges, edges_of(tris) + count))
        movable = np.concatenate((movable, np.ones(len(layer), bool)))
    start = co.copy()
    moved = []
    for _ in range(strokes):
        stroke = fitting.FitStroke(body, co, movable, mode=mode, offset=0.001, layer_radius=0.02, max_distance=0.05,
                                   edges=edges, smooth=0.5, fade_distance=fade_distance)
        for z in np.linspace(-0.1, 0.05, 31):
            for k in range(24):
                phi = 2.0 * math.pi * k / 24
                stroke.dab(np.array([THIGH * math.cos(phi), THIGH * math.sin(phi), z]), 0.02, 0.5)
        co = stroke.co
        change = np.linalg.norm(co[:count, :2], axis=1) - np.linalg.norm(start[:count, :2], axis=1)
        moved.append(change.reshape(len(SLEEVE_ROWS), SLEEVE_SEGMENTS).mean(axis=1))
    kept = np.array_equal(co[:count][~shown], start[:count][~shown])
    thickness = None
    if outer:
        gap = np.linalg.norm(co[count:, :2], axis=1) - np.linalg.norm(co[:count][shown, :2], axis=1)
        thickness = (gap.min(), gap.max())
    return moved, kept, thickness


def test_fade_at_hidden():
    # Hiding part of a mesh keeps it from moving, but the brush fitted the visible part right up to it,
    # leaving a step at its edge. Faded, the visible part eases from the hidden edge's height into the fit.
    edge = np.flatnonzero(SLEEVE_ROWS < 0.0201)[-1]  # the last shown row, next to the hidden ones
    # Past the fade (3 cm) and Keep Together (2 cm): Tighten shares the least tightening that far.
    fitted = SLEEVE_ROWS < -0.0249
    moved, kept, _ = hidden_top_sleeve('FIT', 0.0, -0.004, strokes=1)
    check(moved[0][edge] > 0.004, "fade: without it, the row next to the hidden ones moved off {:.2f} mm".format(
        moved[0][edge] * 1000))
    for mode, depth in (('FIT', -0.004), ('PUSH', -0.004), ('TIGHTEN', 0.006)):
        moved, kept, thickness = hidden_top_sleeve(mode, 0.03, depth, outer=mode == 'TIGHTEN')
        label = "fade, {}".format(mode)
        full = abs(depth - 0.001)
        last = moved[-1]
        check(kept, label + ": hidden rows stayed put")
        check(abs(last[edge]) < 0.1 * full, "{}: the row next to them moved {:.2f} of {:.1f} mm".format(
            label, abs(last[edge]) * 1000, full * 1000))
        check(np.abs(np.abs(last[fitted]) - full).max() < 0.0002, label + ": fitted fully farther away ({:.2f}..{:.2f} mm)".format(
            np.abs(last[fitted]).min() * 1000, np.abs(last[fitted]).max() * 1000))
        steps = np.diff(np.abs(last[:edge + 2]))
        check(np.abs(steps).max() < 0.3 * full and (steps < 0.0001).all(),
              label + ": it eases in row by row (at most {:.2f} mm between rows)".format(np.abs(steps).max() * 1000))
        check(abs(moved[-1][edge] - moved[0][edge]) < 0.0001,
              label + ": more strokes don't wear the fade down ({:.2f}, then {:.2f} mm)".format(
                  abs(moved[0][edge]) * 1000, abs(moved[-1][edge]) * 1000))
        if thickness is not None:
            check(abs(thickness[0] - 0.003) < 0.0002 and abs(thickness[1] - 0.003) < 0.0002,
                  label + ": the separate layer over it kept its distance ({:.2f}..{:.2f} mm)".format(
                      thickness[0] * 1000, thickness[1] * 1000))


def open_torso(columns=48, rings=13):
    """An open tube like a torso on its own, from its waist edge at z 0 up to z 0.6. It widens toward
    the waist edge, so its normals there tilt up, as on the devkit's torso."""
    zs = np.linspace(0.6, 0.0, rings)
    angles = 2.0 * math.pi * np.arange(columns) / columns
    co = np.array([(r * math.cos(a), r * math.sin(a), z)
                   for z, r in zip(zs, 0.20 + 0.1 * (0.6 - zs) / 0.6) for a in angles])
    tris = []
    for i in range(rings - 1):
        for j in range(columns):
            a, b = i * columns + j, i * columns + (j + 1) % columns
            tris += [(a, b, b + columns), (a, b + columns, a + columns)]
    return co, np.array(tris)


def ring(radius, z, count=48):
    angles = 2.0 * math.pi * (np.arange(count) + 0.3) / count
    return np.stack((radius * np.cos(angles), radius * np.sin(angles), np.full(count, z)), axis=1)


def test_strokes_stopping_short():
    # Strokes that stop short of the mesh's end fit the rows past them only part of the way, and every
    # stroke measures its goals anew. Shared over Keep Together, what those rows still needed once pushed the
    # part fitted next to them out again with every stroke: 7.4 mm after 2 strokes where 5 mm was right,
    # 11.6 after 5, 31.8 after 20.
    body = body_surface(*tube(THIGH, 0.0, 68, np.linspace(-0.12, 0.12, 37)))
    rows = SLEEVE_ROWS.repeat(SLEEVE_SEGMENTS)
    fitted = (rows > -0.0501) & (rows < 0.0401)
    for mode, smooth, strokes in (('PUSH', 0.0, 4), ('FIT', 0.5, 3)):
        co, tris = tube(THIGH - 0.004, 0.0, SLEEVE_SEGMENTS, SLEEVE_ROWS, caps=False)
        edges = edges_of(tris)
        start = co.copy()
        moved = []
        for _ in range(strokes):
            stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode=mode, offset=0.001, layer_radius=0.02,
                                       max_distance=0.05, edges=edges, tris=tris, smooth=smooth)
            for _ in range(2):
                for z in np.linspace(-0.06, 0.05, 23):
                    for k in range(24):
                        phi = 2.0 * math.pi * k / 24
                        stroke.dab(np.array([THIGH * math.cos(phi), THIGH * math.sin(phi), z]), 0.02, 0.5)
            co = stroke.co
            moved.append(np.linalg.norm(co[:, :2], axis=1) - np.linalg.norm(start[:, :2], axis=1))
        label = "strokes stopping short, {}{}".format(mode, " with Auto Smooth" if smooth else "")
        beyond = body.nearest(co).distance.max() - 0.001
        check(beyond < 0.0004, label + ": nothing past the offset after {} strokes (at most {:.2f} mm)".format(
            strokes, beyond * 1000))
        check(np.abs(moved[-1][fitted] - 0.005).max() < 0.0003,
              label + ": the part in the strokes moved 5 mm ({:.2f}..{:.2f} mm)".format(
                  moved[-1][fitted].min() * 1000, moved[-1][fitted].max() * 1000))
        growth = np.abs(moved[-1] - moved[1])[fitted].max()
        check(growth < 0.0001, label + ": strokes after the second didn't move it any more ({:.3f} mm)".format(
            growth * 1000))


def test_open_body_has_no_inside_past_its_edge():
    body = body_surface(*open_torso())
    check(len(body.open_keys) == 2 * 48 and body.open_vertex.sum() == 2 * 48, "open body: its two rims are open")
    closed = body_surface(*uv_sphere(1.0))
    check(not len(closed.open_keys), "closed body: nothing is open")

    # Below the waist edge, inside and outside its radius (0.3): past the edge, never inside.
    below = np.concatenate([ring(r, z) for r in (0.22, 0.28, 0.35) for z in (-0.05, -0.1, -0.4)])
    near = body.nearest(below)
    check(near.beyond.all(), "open body: points below its edge are past it ({} of {})".format(near.beyond.sum(), len(below)))
    to_edge = np.linalg.norm(below - near.location, axis=1)
    check(np.allclose(near.distance, to_edge), "open body: their distance is the distance to the edge, outside")
    # Above the edge, inside the tube and outside it, and just inside its rim: as before.
    inner, outer, rim = body.nearest(ring(0.1, 0.3)), body.nearest(ring(0.4, 0.3)), body.nearest(ring(0.28, 0.005))
    check(not inner.beyond.any() and (inner.distance < 0).all(), "open body: inside the tube is inside")
    check(not outer.beyond.any() and (outer.distance > 0).all(), "open body: outside the tube is outside")
    check(not rim.beyond.any() and (rim.distance < 0).all(), "open body: just inside its edge is still inside")
    level = body.nearest(ring(0.22, -0.02))  # 2 cm below the edge, 8 cm in: in line with its normal there
    check(not level.beyond.any() and (level.distance < 0).all(), "open body: straight in from its edge is inside")
    rng = np.random.default_rng(5)
    points = rng.uniform(-1.5, 1.5, (2000, 3))
    check(not closed.nearest(points).beyond.any(), "closed body: nothing is past an edge")


def test_push_leaves_parts_past_an_open_body_alone():
    body = body_surface(*open_torso())
    # A narrow skirt below the waist edge (inside its radius), and a band clipping 3 cm into the torso.
    below = ring(0.26, -0.05)
    above = ring(0.24, 0.2)
    co = np.concatenate((below, above))
    start = co.copy()
    stroke = fitting.FitStroke(body, co, np.ones(len(co), bool), mode='PUSH', offset=0.01, layer_radius=0.02)
    paint(stroke, (0.0, -0.27, 0.07), 0.3)
    n = len(below)
    check(np.array_equal(co[:n], start[:n]), "open body: the skirt below its edge isn't pushed")
    front = start[n:, 1] < -0.2
    radius = np.linalg.norm(co[n:, :2], axis=1)
    check(front.any() and radius[front].min() > 0.27, "open body: the band clipping into it is pushed out ({:.3f})".format(radius[front].min()))


def same_surface(a, b):
    """Whether two body surfaces are alike, however their welded vertices are numbered."""
    points = np.random.default_rng(3).uniform(-1.3, 1.3, (500, 3))
    near_a, near_b = a.nearest(points), b.nearest(points)
    return (
        np.allclose(a.co[a._weld], b.co[b._weld]) and np.allclose(a.normals[a._weld], b.normals[b._weld]) and
        np.allclose(a.area_normals[a._weld], b.area_normals[b._weld]) and
        np.array_equal(a.open_vertex[a._weld], b.open_vertex[b._weld]) and len(a.edges) == len(b.edges) and
        math.isclose(a.edge_length, b.edge_length) and np.array_equal(near_a.beyond, near_b.beyond) and
        all(np.allclose(getattr(near_a, name), getattr(near_b, name)) for name in ("location", "normal", "distance"))
    )


def test_changed_shape_keeps_topology():
    # A posed or reshaped body keeps what its triangles decide, and comes out as if built from scratch.
    fitting.BodySurface.clear_cache()
    co, tris = uv_sphere(1.0, split_seam=True)
    first = fitting.BodySurface.from_geometry("changing body", SimpleNamespace(co=co, tris=tris))
    posed = co * np.array([1.1, 0.9, 1.0]) + 0.01 * np.sin(5.0 * co[:, [1, 2, 0]])
    second = fitting.BodySurface.from_geometry("changing body", SimpleNamespace(co=posed, tris=tris))
    check(second is not first and second.edges is first.edges and second.corners is first.corners,
          "changed shape: what the triangles decide is reused")
    check(same_surface(second, body_surface(posed, tris)), "changed shape: same as built from scratch")
    # Opening the split seam (the column at 360 degrees) welds the vertices differently: built again.
    opened = posed.copy()
    seam = 1 + 65 * np.arange(31) + 64
    opened[seam] *= 1.001
    third = fitting.BodySurface.from_geometry("changing body", SimpleNamespace(co=opened, tris=tris))
    check(third.edges is not second.edges and len(third.co) == len(second.co) + len(seam) and
          same_surface(third, body_surface(opened, tris)), "opened seam: built again, as from scratch")
    fitting.BodySurface.clear_cache()


def mesh_object(name, co, tris):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(co.tolist(), [], tris.tolist())
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def remove_objects(*objects):
    for obj in objects:
        data = obj.data
        bpy.data.objects.remove(obj)
        if isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
        else:
            bpy.data.armatures.remove(data)


def clipping_scene():
    """A body and a garment around it, which the scaled bone of its armature pulls into the body from
    frame 1 to 10, like an animation. Only the garment is selected."""
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    body = mesh_object("Clip Body", *uv_sphere(1.0, 32, 16))
    garment = mesh_object("Clip Garment", *uv_sphere(1.02, 32, 16))
    rig = bpy.data.objects.new("Clip Rig", bpy.data.armatures.new("Clip Rig"))
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    bone = rig.data.edit_bones.new("Root")
    bone.head, bone.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 0.3)
    bpy.ops.object.mode_set(mode='OBJECT')
    garment.vertex_groups.new(name="Root").add(list(range(len(garment.data.vertices))), 1.0, 'REPLACE')
    garment.modifiers.new("Armature", 'ARMATURE').object = rig
    pose = rig.pose.bones["Root"]
    pose.keyframe_insert("scale", frame=1)
    pose.scale = (0.95, 0.95, 0.95)
    pose.keyframe_insert("scale", frame=10)
    garment.select_set(True)
    bpy.context.scene.frame_set(1)
    return body, garment, rig


def measure_clipping(state, body):
    while clipping._refresh(state, bpy.context, body, None):
        pass
    return state.marks


def test_clipping_follows_frame_changes():
    # Scrubbing or playing an animation sends no depsgraph update: the marks still follow the new frame.
    body, garment, rig = clipping_scene()
    clipping.register()
    try:
        state = clipping._state(bpy.context)
        inside = int((measure_clipping(state, body)[garment.name].distance < 0.0).sum())
        check(inside == 0, "clipping: nothing clips at frame 1 ({} vertices)".format(inside))
        bpy.context.scene.frame_set(10)
        inside = int((measure_clipping(state, body)[garment.name].distance < 0.0).sum())
        check(inside == len(garment.data.vertices),
              "clipping: at frame 10 the posed garment clips ({} of {} vertices)".format(inside, len(garment.data.vertices)))
    finally:
        clipping.unregister()
        bpy.context.scene.frame_set(1)
        remove_objects(body, garment, rig)
        fitting.BodySurface.clear_cache()


def test_clipping_per_view_layer():
    # Windows showing another view layer, or another scene with its marks off, leave the marks alone.
    body, garment, rig = clipping_scene()
    scene = bpy.context.scene
    layer = scene.view_layers.new("Clip Layer")
    garment.select_set(False, view_layer=layer)
    other_scene = bpy.data.scenes.new("Clip Scene")
    try:
        state = clipping._state(bpy.context)
        marks = measure_clipping(state, body)[garment.name]
        with bpy.context.temp_override(view_layer=layer):
            other = measure_clipping(clipping._state(bpy.context), body)
        with bpy.context.temp_override(scene=other_scene, view_layer=other_scene.view_layers[0]):
            clipping._draw()
        check(not other and clipping._state(bpy.context) is state and state.marks.get(garment.name) is marks,
              "clipping: the marks of each view layer are kept apart")
        check(not clipping._refresh(state, bpy.context, body, None) and not marks.pending.any(),
              "clipping: nothing had to be measured again")
    finally:
        clipping.clear()
        bpy.data.scenes.remove(other_scene)
        scene.view_layers.remove(layer)
        remove_objects(body, garment, rig)
        fitting.BodySurface.clear_cache()


def test_read_edited_meshes():
    # Body Fit and Texture Relax read the meshes in Edit Mode in bulk: the positions of the active shape
    # key and the UVs, as the edit mesh holds them.
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8)
    obj = bpy.context.object
    obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="Wide")
    for point in key.data:
        point.co *= 1.5
    try:
        for active in (1, 0):
            obj.active_shape_key_index = active
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            bm.verts[3].co.x += 0.25
            layer = bm.loops.layers.uv.active
            bm.faces[2].loops[1][layer].uv = (0.9, 0.1)
            bmesh.update_edit_mesh(obj.data)
            edited = operators.read_edited_meshes(bpy.context, None, uvs=True)
            positions = np.array([vert.co[:] for vert in bm.verts], dtype=np.float32)
            uv = np.array([loop[layer].uv[:] for face in bm.faces for loop in face.loops])
            tri_loops = np.empty(len(obj.data.loop_triangles) * 3, dtype=np.int32)
            obj.data.loop_triangles.foreach_get("loops", tri_loops)
            check(np.array_equal(edited.start, positions),
                  "edited meshes: positions of shape key {} as edited".format(key.name if active else "Basis"))
            check(np.array_equal(edited.tri_uvs, uv[tri_loops.reshape(-1, 3)]), "edited meshes: UVs as edited")
            bpy.ops.object.mode_set(mode='OBJECT')
    finally:
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        remove_objects(obj)


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
