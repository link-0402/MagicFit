"""Headless tests for the Texture Relax engine.

Run from the repository root:
    blender -b --factory-startup --python tests/test_relax_core.py
"""

import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from magic_fit import relaxing  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def grid(nx=16, ny=16, size=0.16, *, split=False, uv_scale=(1.0, 1.0)):
    """A triangulated grid in the XY plane with a matching UV map, ``size`` wide.

    With ``split``, the column in the middle is duplicated (a split seam): the right half has its
    own copies of those vertices, at the same positions, and its own UV island.
    """
    columns = nx + 1
    x = np.linspace(-size / 2.0, size / 2.0, columns)
    y = np.linspace(-size / 2.0, size / 2.0, ny + 1)
    co = np.array([(xi, yj, 0.0) for yj in y for xi in x])
    uv = np.array([(0.5 + xi / size, 0.5 + yj / size) for yj in y for xi in x]) * np.array(uv_scale)

    def index(i, j):
        return j * columns + i

    tris, right = [], []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)
            tris += [(a, b, c), (a, c, d)]
            right += [i >= nx // 2] * 2
    tris = np.array(tris)
    right = np.array(right)
    tri_uvs = uv[tris]
    if split:
        middle = nx // 2
        copies = np.arange(len(co), len(co) + ny + 1)
        co = np.concatenate((co, co[[index(middle, j) for j in range(ny + 1)]]))
        on_seam = right[:, None] & (tris % columns == middle)
        tris = np.where(on_seam, copies[tris // columns], tris)
        tri_uvs[right] += np.array([0.6, 0.0])
    return co, tris, tri_uvs


def distort(co, *, center=(0.0, 0.0), radius=0.05, amount=0.4):
    """Push vertices sideways, unevenly, within ``radius`` of ``center``: a bump in the mapping.
    Nothing beyond 1.5 x ``radius`` moves at all."""
    delta = co[:, :2] - np.array(center)
    r = np.linalg.norm(delta, axis=1)
    window = np.clip(1.0 - (r / (1.5 * radius)) ** 2, 0.0, 1.0)
    push = amount * r * np.exp(-(r / radius) ** 2 * 2.0) * window * window
    out = co.copy()
    out[:, :2] += delta * (push / np.maximum(r, 1e-12))[:, None]
    return out


def stretch(co, tris, tri_uvs):
    """Texture distortion per triangle: (anisotropy, log texel scale)."""
    e1 = co[tris[:, 1]] - co[tris[:, 0]]
    e2 = co[tris[:, 2]] - co[tris[:, 0]]
    u1 = tri_uvs[:, 1] - tri_uvs[:, 0]
    u2 = tri_uvs[:, 2] - tri_uvs[:, 0]
    # Jacobian of UV -> 3D: [e1 e2] = J [u1 u2].
    uv = np.stack((u1, u2), axis=2)
    e = np.stack((e1, e2), axis=2)
    jacobian = e @ np.linalg.inv(uv)
    s = np.linalg.svd(jacobian, compute_uv=False)
    return s[:, 0] / np.maximum(s[:, 1], 1e-12) - 1.0, np.log(np.maximum(s[:, 0] * s[:, 1], 1e-30)) / 2.0


def unevenness(co, tris, tri_uvs, subset=None):
    """One number for how distorted the texture is: mean anisotropy plus the spread of the texel scale."""
    anisotropy, scale = stretch(co, tris, tri_uvs)
    if subset is not None:
        anisotropy, scale = anisotropy[subset], scale[subset]
    return float(anisotropy.mean() + scale.std())


def relax(stroke, center=(0.0, 0.0, 0.0), radius=1.0, dabs=400, strength=1.0, falloff='CONSTANT'):
    for _ in range(dabs):
        stroke.dab(np.array(center), radius, strength, falloff)


def test_flat_grid_repaired():
    co, tris, tri_uvs = grid()
    rest = co.copy()
    co = distort(co)
    before = unevenness(co, tris, tri_uvs)
    check(before > 0.1, "distorted: the texture is uneven ({:.3f})".format(before))
    stroke = relaxing.RelaxStroke(co, tris, tri_uvs, np.ones(len(co), bool))
    relax(stroke)
    after = unevenness(co, tris, tri_uvs)
    check(after < 0.05 * before, "relax: the texture is even again ({:.3f} -> {:.4f})".format(before, after))
    check(np.abs(co[:, 2]).max() == 0.0, "relax: the flat surface stayed flat")
    check(np.abs(co - rest).max() < 1e-4, "relax: vertices went back to where the texture wants them ({:.2f} mm off)".format(
        np.abs(co - rest).max() * 1000))
    boundary = (np.abs(np.abs(rest[:, 0]) - 0.08) < 1e-9) | (np.abs(np.abs(rest[:, 1]) - 0.08) < 1e-9)
    on_edge = (np.abs(np.abs(co[boundary, 0]) - 0.08) < 1e-9) | (np.abs(np.abs(co[boundary, 1]) - 0.08) < 1e-9)
    check(on_edge.all(), "relax: open edges kept their line")


def test_even_stretch_left_alone():
    # An evenly stretched texture (the UVs squashed along one axis) is not a distortion.
    co, tris, tri_uvs = grid(uv_scale=(1.0, 0.4))
    start = co.copy()
    stroke = relaxing.RelaxStroke(co, tris, tri_uvs, np.ones(len(co), bool))
    relax(stroke, dabs=10)
    check(np.abs(co - start).max() < 1e-9, "even stretch: nothing moves ({:.2e})".format(np.abs(co - start).max()))
    check(not stroke.moved.any(), "even stretch: no vertex counts as moved")


def test_brush_falloff_and_reach():
    co, tris, tri_uvs = grid()
    co = distort(co)
    start = co.copy()
    stroke = relaxing.RelaxStroke(co, tris, tri_uvs, np.ones(len(co), bool))
    relax(stroke, radius=0.03, dabs=5, strength=0.5, falloff='SMOOTH')
    far = np.linalg.norm(start[:, :2], axis=1) > 0.03
    check(np.array_equal(co[far], start[far]), "falloff: vertices outside the brush untouched")
    check((co[~far] != start[~far]).any(), "falloff: vertices inside the brush moved")
    moved = np.flatnonzero(stroke.moved)
    check(len(moved) and not far[moved].any(), "falloff: moved marks only the vertices inside")


def bend_over_cylinder(points, radius=0.06):
    """The XY plane wrapped around a cylinder of ``radius`` along y (x becomes the arc length)."""
    out = points.copy()
    angle = points[:, 0] / radius
    out[:, 0] = radius * np.sin(angle)
    out[:, 2] = radius * (np.cos(angle) - 1.0)
    return out


def test_curved_surface_keeps_shape():
    # The grid bent over a cylinder: sliding must follow the curve, not a tangent plane.
    co, tris, tri_uvs = grid(nx=24, ny=24)
    rest = bend_over_cylinder(co)
    co = bend_over_cylinder(distort(co, amount=0.35))
    before = unevenness(co, tris, tri_uvs)
    stroke = relaxing.RelaxStroke(co, tris, tri_uvs, np.ones(len(co), bool))
    relax(stroke, dabs=800)
    after = unevenness(co, tris, tri_uvs)
    check(after < 0.1 * before, "cylinder: the texture is even again ({:.3f} -> {:.4f})".format(before, after))
    # The vertices stay on the flat triangles of the start surface, which sag below the true cylinder.
    off = np.abs(np.sqrt(co[:, 0] ** 2 + (co[:, 2] + 0.06) ** 2) - 0.06).max()
    check(off < 2e-4, "cylinder: the surface kept its shape ({:.3f} mm off the cylinder)".format(off * 1000))
    check(np.abs(co - rest).max() < 5e-4, "cylinder: vertices close to their undistorted spots ({:.2f} mm)".format(
        np.abs(co - rest).max() * 1000))


def test_curved_undistorted_left_alone():
    # An even texture on a curved surface is a fixed point, whatever the open edges do and however
    # the quads are split: neither the fit's inward pull nor one-sided fans may creep sideways.
    for mode in ('SLIDE', 'FREE', 'FIXED'):
        co, tris, tri_uvs = grid(nx=24, ny=24)
        co = bend_over_cylinder(co)
        start = co.copy()
        stroke = relaxing.RelaxStroke(co, tris, tri_uvs, np.ones(len(co), bool), boundary=mode)
        relax(stroke, dabs=200)
        drift = np.abs(co - start).max()
        check(drift < 1e-6, "cylinder, {}: an even texture doesn't move ({:.4f} mm)".format(mode.lower(), drift * 1000))


def test_split_seam_stays_closed():
    co, tris, tri_uvs = grid(split=True)
    original_count = 17 * 17
    copies = np.arange(original_count, len(co))
    originals = np.array([8 + 17 * j for j in range(17)])
    right = tri_uvs[:, :, 0].min(axis=1) > 0.55
    co = distort(co, center=(0.01, 0.0))
    check(np.abs(co[copies] - co[originals]).max() == 0.0, "split: the seam starts closed")
    before = max(unevenness(co, tris, tri_uvs, ~right), unevenness(co, tris, tri_uvs, right))
    stroke = relaxing.RelaxStroke(co, tris, tri_uvs, np.ones(len(co), bool))
    relax(stroke)
    after = max(unevenness(co, tris, tri_uvs, ~right), unevenness(co, tris, tri_uvs, right))
    check(after < 0.1 * before, "split: the texture is even on each island ({:.3f} -> {:.4f})".format(before, after))
    check(np.abs(co[copies] - co[originals]).max() == 0.0, "split: the seam is still closed")
    check(stroke.moved[originals].any(), "split: the seam itself moved")
    # The two islands' texel densities aren't tied together by the UV map, so the seam may settle a
    # little off its original line, with each island evenly stretched to meet it.
    check(np.abs(co[originals[8], 0]) < 3e-3, "split: the seam settled near its line ({:.2f} mm off)".format(
        abs(co[originals[8], 0]) * 1000))


def test_boundary_modes():
    for mode in ('SLIDE', 'FREE', 'FIXED'):
        co, tris, tri_uvs = grid()
        rest = co.copy()
        # Distort near the top edge, so the edge itself is uneven.
        co = distort(co, center=(0.0, 0.08), radius=0.04)
        edge = np.abs(rest[:, 1] - 0.08) < 1e-9
        start = co.copy()
        stroke = relaxing.RelaxStroke(co, tris, tri_uvs, np.ones(len(co), bool), boundary=mode)
        relax(stroke, center=(0.0, 0.08, 0.0), radius=0.06, dabs=40)
        if mode == 'FIXED':
            check(np.array_equal(co[edge], start[edge]), "fixed: the open edge didn't move")
        elif mode == 'SLIDE':
            check((co[edge] != start[edge]).any(), "slide: the open edge's vertices moved")
            check(np.abs(co[edge, 1] - 0.08).max() < 1e-9, "slide: they stayed on the edge's line")
            check(np.abs(co[edge, 0] - rest[edge, 0]).max() < 1e-3, "slide: spaced as the texture wants ({:.2f} mm)".format(
                np.abs(co[edge, 0] - rest[edge, 0]).max() * 1000))
        else:
            check((co[edge] != start[edge]).any(), "free: the open edge's vertices moved")
        interior = ~edge & (np.abs(rest[:, 0]) < 0.079) & (rest[:, 1] > -0.079)
        check(np.abs(co[:, 2]).max() < 1e-9, mode.lower() + ": the surface stayed flat")


def test_free_edge_beside_flat_triangle():
    # A triangle without area on the open edge, as game meshes often have, has no plane to move past
    # the edge along: the vertex beside it moves out as it would without that triangle.
    moved = []
    for flat in (False, True):
        co, tris, tri_uvs = grid()
        top = 16 * 17 + np.arange(6, 9)
        if flat:
            flat_uvs = np.array([tri_uvs[tris == v][0] for v in top])
            tris = np.concatenate((top[None], tris))
            tri_uvs = np.concatenate((flat_uvs[None], tri_uvs))
        # The texture wants the first of them past the edge.
        tri_uvs[tris == top[0]] += np.array([0.0, 0.03])
        start = co.copy()
        stroke = relaxing.RelaxStroke(co, tris, tri_uvs, np.ones(len(co), bool), boundary='FREE')
        step = 0.0
        for _ in range(20):
            before = co.copy()
            stroke.dab(np.array([-0.02, 0.08, 0.0]), 0.05, 1.0, 'CONSTANT')
            step = max(step, np.linalg.norm(co - before, axis=1).max())
        moved.append(co[top[0]] - start[top[0]])
        check(step < 0.005, "free{}: no dab moved a vertex half an edge ({:.2f} mm)".format(
            ", flat triangle" if flat else "", step * 1000))
    check(moved[0][1] > 0.003 and np.abs(moved[1] - moved[0]).max() < 2e-4,
          "free, flat triangle: the vertex beside it moved out as without it ({:.2f} mm, {:.2f} mm)".format(
              moved[1][1] * 1000, moved[0][1] * 1000))


def test_slide_next_to_irregular_corner():
    # Two grids touching at one corner, where four open edges meet: it has no single edge to slide
    # on, and a vertex that slides more than halfway to it must still keep to its own edge.
    co, tris, tri_uvs = grid()
    n = len(co)
    co = np.concatenate((co, co + np.array([0.16, 0.16, 0.0])))
    tris = np.concatenate((tris, tris + n))
    tri_uvs = np.concatenate((tri_uvs, tri_uvs + np.array([2.0, 0.0])))
    # The texture wants the vertex next to the corner closer to it, and past the edge.
    vertex = 16 * 17 + 15
    tri_uvs[tris == vertex] += np.array([0.04, 0.05])
    edge = np.abs(co[:, 1] - 0.08) < 1e-9
    stroke = relaxing.RelaxStroke(co, tris, tri_uvs, np.ones(len(co), bool), boundary='SLIDE')
    relax(stroke, center=(0.07, 0.08, 0.0), radius=0.02, dabs=40)
    check(co[vertex, 0] > 0.075, "slide: the vertex went more than halfway to the corner ({:.2f} mm)".format(
        co[vertex, 0] * 1000))
    check(np.abs(co[edge, 1] - 0.08).max() < 1e-9, "slide: it stayed on the edge's line ({:.2f} mm off)".format(
        np.abs(co[edge, 1] - 0.08).max() * 1000))


def test_two_layers_stay_apart():
    # Two layers a hair apart: neither may be pulled onto the other while sliding.
    co, tris, tri_uvs = grid()
    gap = 0.0005
    upper = co + np.array([0.0, 0.0, gap])
    both = np.concatenate((distort(co), distort(upper, amount=0.2)))
    tris_both = np.concatenate((tris, tris + len(co)))
    uvs_both = np.concatenate((tri_uvs, tri_uvs))
    stroke = relaxing.RelaxStroke(both, tris_both, uvs_both, np.ones(len(both), bool))
    relax(stroke)
    check(np.abs(both[:len(co), 2]).max() == 0.0 and np.abs(both[len(co):, 2] - gap).max() == 0.0,
          "layers: every vertex stayed on its own layer")
    after = unevenness(both, tris_both, uvs_both)
    check(after < 0.02, "layers: both textures are even ({:.4f})".format(after))


def test_missing_uvs_and_hidden():
    co, tris, tri_uvs = grid()
    co = distort(co)
    start = co.copy()
    no_uvs = np.full_like(tri_uvs, np.nan)
    stroke = relaxing.RelaxStroke(co, tris, no_uvs, np.ones(len(co), bool))
    relax(stroke, dabs=3)
    check(np.array_equal(co, start), "no UV map: nothing moves")
    movable = np.linalg.norm(start[:, :2], axis=1) > 0.02
    stroke = relaxing.RelaxStroke(co, tris, tri_uvs, movable)
    relax(stroke, dabs=3)
    check(np.array_equal(co[~movable], start[~movable]), "hidden: unmovable vertices stay put")
    check((co[movable] != start[movable]).any(), "hidden: the others move")


def test_closest_on_triangles():
    a = np.array([[0.0, 0.0, 0.0]] * 4)
    b = np.array([[1.0, 0.0, 0.0]] * 4)
    c = np.array([[0.0, 1.0, 0.0]] * 4)
    p = np.array([[0.25, 0.25, 1.0], [2.0, -1.0, 0.0], [-1.0, -1.0, 0.5], [1.0, 1.0, 0.0]])
    point, bary, plane, has_plane = relaxing._closest_on_triangles(p, a, b, c)
    expected = np.array([[0.25, 0.25, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    check(np.allclose(point, expected), "closest point on a triangle: inside, corner, corner, edge")
    check(np.allclose(bary.sum(axis=1), 1.0) and np.allclose(bary[3], [0.0, 0.5, 0.5]), "closest point: barycentric coordinates")
    check(has_plane.all() and np.allclose(plane[0], [0.25, 0.25, 0.0]) and np.allclose(plane[1], [2.0, -1.0, 0.0]),
          "closest point: plane projection")
    point, bary, plane, has_plane = relaxing._closest_on_triangles(
        np.array([[0.5, 0.5, 0.0]]), a[:1], b[:1], np.array([[2.0, 0.0, 0.0]]))
    check(not has_plane[0] and np.allclose(point[0], [0.5, 0.0, 0.0]) and np.allclose(plane[0], point[0]),
          "closest point: a triangle without area has no plane, its closest point stands in")


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
