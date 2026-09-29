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


def compare_auto_smooth(surface, co, n, edges, mode):
    """(roughness, closest gap to the body, worst thickness error) without and with full Auto Smooth,
    over the band the strokes went over."""
    unit = co[:n] / np.linalg.norm(co[:n], axis=1, keepdims=True)
    band = np.flatnonzero((np.abs(unit[:, 2]) < 0.08) & (unit[:, 1] < 0.0) & (np.abs(unit[:, 0]) < math.sin(0.3)))
    results = []
    for smooth in (0.0, 1.0):
        points = co.copy()
        stroke = fitting.FitStroke(surface, points, np.ones(len(points), bool), mode=mode, offset=0.002,
                                   layer_radius=0.02, max_distance=0.05, edges=edges, smooth=smooth)
        stroke_across(stroke)
        radius = np.linalg.norm(points, axis=1)
        thickness = radius[n:][band] - radius[:n][band]
        results.append((roughness(radius, edges, band), surface.nearest(points[:n][band]).distance.min(),
                        np.abs(thickness - 0.008).max()))
    return results


def check_auto_smooth(label, results, factor):
    (plain, _, _), (smooth, gap, thickness) = results
    check(smooth < factor * plain,
          "auto smooth, {}: smoother ({:.3f} mm vs {:.3f} mm)".format(label, smooth * 1000, plain * 1000))
    check(gap > 0.002 - 5e-5, "auto smooth, {}: never closer than the offset ({:.2f} mm)".format(label, gap * 1000))
    check(thickness < 0.001, "auto smooth, {}: thickness kept (max error {:.2f} mm)".format(label, thickness * 1000))


def test_auto_smooth_pushed_dent():
    # A dent poking 13 mm into the body: pushed out, it used to leave a rim.
    co, n, edges = garment(TORSO + 0.002, gaussian(FRONT, 0.1, -0.013 / (TORSO + 0.002)))
    check_auto_smooth("pushed dent", compare_auto_smooth(torso(), co, n, edges, 'PUSH'), 0.6)


def test_auto_smooth_uneven_push():
    # A garment hovering inside the offset with 1 mm of unevenness: pushed out in patches.
    co, n, edges = garment(TORSO + 0.0005)
    wobble = np.modf(np.sin(np.arange(n) * 91.17) * 43758.5453)[0] * 0.001
    co *= (1.0 + np.concatenate((wobble, wobble)) / TORSO)[:, None]
    check_auto_smooth("uneven push", compare_auto_smooth(torso(), co, n, edges, 'PUSH'), 0.2)


def test_auto_smooth_tighten_onto_lumps():
    # Tightened onto a lumpy body, the garment used to copy every lump.
    co, n, edges = garment(TORSO + 0.012)
    check_auto_smooth("tighten onto lumps", compare_auto_smooth(torso(lumps()), co, n, edges, 'TIGHTEN'), 0.5)


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
    check(np.allclose(near.distance, to_edge) and np.allclose(near.height, to_edge),
          "open body: their distance is the distance to the edge, outside")
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
        np.array_equal(a.open_vertex[a._weld], b.open_vertex[b._weld]) and len(a.edges) == len(b.edges) and
        math.isclose(a.edge_length, b.edge_length) and np.array_equal(near_a.beyond, near_b.beyond) and
        all(np.allclose(getattr(near_a, name), getattr(near_b, name)) for name in ("location", "normal", "distance", "height"))
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
