"""Headless tests for the Squish brush's engine (`magic_fit.squishing`).

Run from the repository root:
    blender -b --factory-startup --python tests/test_squish_core.py
"""

import os
import sys
import time
import traceback

import numpy as np
from mathutils.bvhtree import BVHTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from magic_fit import fitting, squishing  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


# The Chest preset: Press, Bulge and Bulge Width.
P, H, W = 0.003, 0.009, 0.07
# A smoothing radius for the edge tests.
R = 0.02
# The tube's hem, between two rows of the torso.
HEM = 0.0025


def grid_tris(index):
    """Triangles of a grid whose vertex at (column, row) is ``index[column, row]``."""
    a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    return np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))


def rings(radii, zs, step):
    """Positions and triangles of rings around Z at ``radii`` and heights ``zs`` (one per row), joined row to
    row, facing out. The ring is split where it closes, like a UV seam: its first and last columns lie on top of
    each other."""
    count = max(3, round(2.0 * np.pi * max(radii) / step))
    angles = np.linspace(0.0, 2.0 * np.pi, count + 1)
    co = np.array([(r * np.cos(a), r * np.sin(a), z) for a in angles for r, z in zip(radii, zs)])
    index = np.arange(len(co)).reshape(len(angles), len(zs))
    return co, grid_tris(index)


def torso(radius=0.12, height=0.4, step=0.005):
    """An open cylinder around Z, from -height / 2 to height / 2, with a split seam."""
    zs = np.linspace(-height / 2.0, height / 2.0, round(height / step) + 1)
    return rings(np.full(len(zs), radius), zs, step)


def tube(radius=0.123, z0=-0.21, z1=HEM, step=0.005, fold=0.0):
    """A tube of cloth over the torso from ``z0`` to its hem at ``z1``. With ``fold``, the hem turns in and runs
    back down that far inside the tube (and outside the skin), so the hem has no open border."""
    zs = list(np.linspace(z0, z1, round((z1 - z0) / step) + 1))
    radii = [radius] * len(zs)
    if fold > 0.0:
        inner = list(np.linspace(z1, z1 - fold, round(fold / step) + 1))
        zs += inner
        radii += [radius - 0.0015] * len(inner)
    return rings(radii, zs, step)


def joined(*meshes):
    """Positions and triangles of the meshes (pairs of them) put together."""
    co, tris, first = [], [], 0
    for mesh_co, mesh_tris in meshes:
        co.append(mesh_co)
        tris.append(mesh_tris + first)
        first += len(mesh_co)
    return np.concatenate(co), np.concatenate(tris)


def squish(body, cloth, body_movable=None, cloth_movable=None, **settings):
    """A stroke over ``body`` and ``cloth`` (positions and triangles each), both at their Basis."""
    (body_co, body_tris), (cloth_co, cloth_tris) = body, cloth
    if body_movable is None:
        body_movable = np.ones(len(body_co), dtype=bool)
    if cloth_movable is None:
        cloth_movable = np.ones(len(cloth_co), dtype=bool)
    return squishing.SquishStroke(
        body_co, body_co, body_tris, body_movable, cloth_co, cloth_co, cloth_tris, cloth_movable, **settings,
    )


def radial(co):
    """Unit directions straight out from the Z axis."""
    out = np.array(co, dtype=np.float64)
    out[:, 2] = 0.0
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def surface(co, tris):
    return fitting.BodySurface(co, tris, BVHTree.FromPolygons(co.tolist(), tris.tolist(), all_triangles=True))


def edge_checks(field, body, hem, label):
    z = body.co[:, 2]
    check(np.array_equal(field.covered, z < hem), label + ": covered exactly below the hem")
    near = np.isfinite(field.s)
    check(np.allclose(field.s[near], z[near] - hem, atol=0.003), label + ": s is the height above the hem")
    check(np.all(np.isinf(field.s[np.abs(z - hem) > 0.035])), label + ": beyond reach, infinite")
    radial = body.co.copy()
    radial[:, 2] = 0.0
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    check(np.einsum("ij,ij->i", field.normals, radial).min() > 0.999, label + ": normals point out of the torso")


def test_profile_shape():
    # Measured on Cultist's Rue Medium tops: the edge rides on the bulge (2-5 mm out), the bulge peaks about 2 cm
    # past the edge, the press is full from about 2 cm under it.
    def at(*s):
        return squishing.profile(np.array(s, dtype=np.float64), P, H, W)

    check(abs(at(0.3 * W)[0] - H) < 1e-12, "the bulge peaks at Bulge, 0.3 widths past the edge")
    check(abs(at(0.0)[0] - 0.5 * H) < 1e-12, "half way up at the edge")
    check(abs(at(0.65 * W)[0] - 0.5 * H) < 1e-12, "and half way down 0.65 widths past it")
    check(0.25 * H < at(0.0)[0] < 0.6 * H, "the edge is pushed out, not creased ({:.1f} mm)".format(1000 * at(0.0)[0]))
    check(abs(at(-W)[0] + P) < 1e-12, "fully pressed in under the fabric")
    check(np.all(np.abs(at(W, 2 * W, -2 * W, np.inf, -np.inf)) < 1e-12), "nothing past the bulge, deep under, or at infinity")
    s = np.linspace(-2.2 * W, 1.2 * W, 34001)
    p = at(*s)
    check(np.abs(np.diff(p)).max() < 1e-5, "smooth all the way (no step)")
    check(p[(s > -0.2 * W) & (s < W)].min() > -0.5 * P, "no dent at the edge")


def test_edge_on_tube():
    body = surface(*torso())
    field = squishing.fabric_edge(body, *tube(), max_gap=0.03, reach=0.03, smooth_radius=0.01)
    edge_checks(field, body, HEM, "open hem")


def test_edge_on_folded_hem():
    body = surface(*torso())
    field = squishing.fabric_edge(body, *tube(fold=0.02), max_gap=0.03, reach=0.03, smooth_radius=0.01)
    edge_checks(field, body, HEM, "turned-in hem")


def test_edge_across_rows():
    # A hem cutting the torso's rows at an angle: the tube tilted 15 degrees about X. The edge is found between the
    # torso's vertices, so the roll doesn't step along its rows.
    tilt = np.radians(15.0)
    co, tris = tube()
    # The tube's top row is cut by a plane tilted about X; the rows below it are spread out evenly again.
    rows = round((HEM + 0.21) / 0.005) + 1
    top = (HEM + co[:, 1] * np.sin(tilt)) / np.cos(tilt)
    row = np.arange(len(co)) % rows
    co[:, 2] = -0.21 + (top + 0.21) * row / (rows - 1)
    body = surface(*torso())
    field = squishing.fabric_edge(body, co, tris, max_gap=0.03, reach=0.03, smooth_radius=0.01)
    x, y, z = body.co.T
    hem_z = (HEM + y * np.sin(tilt)) / np.cos(tilt)
    slope = np.tan(tilt) * x / np.hypot(x, y)  # of the hem along the torso's circumference
    expected = (z - hem_z) / np.sqrt(1.0 + slope ** 2)
    near = np.abs(expected) < 0.015
    error = np.abs(field.s[near] - expected[near]).max()
    check(error < 0.001, "the edge is found between the rows (s off by {:.4f} m at most)".format(error))


def test_edge_reused_between_strokes():
    body, cloth = torso(), tube()
    first = squish(body, cloth).field
    check(squish(body, cloth).field is first, "the next stroke over the same Basis reuses the fabric edge")
    check(squish(body, cloth, max_gap=0.02).field is not first, "not with other settings")
    moved = (cloth[0] + [0.0, 0.0, 0.001], cloth[1])
    check(squish(body, moved).field is not first, "nor with the clothing's Basis changed")


def test_clipping_cloth_counts_as_cover():
    body = surface(*torso())
    inside = squishing.fabric_edge(body, *tube(radius=0.119), max_gap=0.03, reach=0.03, smooth_radius=0.01)
    check(inside.covered[body.co[:, 2] < HEM].all(), "a tube 1 mm inside the skin covers it")
    far = squishing.fabric_edge(body, *tube(radius=0.17), max_gap=0.03, reach=0.03, smooth_radius=0.01)
    check(not far.covered.any(), "a tube 5 cm out (past Max Gap) covers nothing")


def test_strap_two_edges():
    body = surface(*torso())
    field = squishing.fabric_edge(body, *tube(z0=-0.0075, z1=0.0075), max_gap=0.03, reach=0.03, smooth_radius=0.01)
    z = body.co[:, 2]
    strap = np.abs(z) < 0.0075
    check(np.array_equal(field.covered, strap), "covered only under the strap")
    check(np.all(field.s[strap] < 0) and np.all(field.s[~strap & (np.abs(z) < 0.03)] > 0),
          "pressed under the strap, out past both edges")
    check(abs(field.s[np.abs(z) < 1e-9].min() + 0.0075) < 0.003, "the middle is a strap's half-width from an edge")


def test_body_goal_follows_profile():
    body = torso()
    stroke = squish(body, tube())
    basis = body[0]
    move = stroke.goal[:stroke.body_count] - basis
    out = radial(basis)
    along = np.einsum("ij,ij->i", move, out)
    aside = np.abs(move - along[:, None] * out).max()
    check(aside < 1e-4, "the skin moves straight out or in ({:.2e} m aside)".format(aside))
    expected = squishing.profile(basis[:, 2] - HEM, P, H, W)
    check(np.abs(along - expected).max() < 0.0003, "by the profile of its height above the hem")
    check(along.min() < -0.9 * P and along.max() > 0.9 * H, "pressed in and bulging out")
    seam = np.flatnonzero((np.abs(basis[:, 1]) < 1e-9) & (basis[:, 0] > 0))
    pairs = seam.reshape(2, -1) if len(seam) % 2 == 0 else None
    check(pairs is not None and np.allclose(move[pairs[0]], move[pairs[1]], rtol=0.0, atol=1e-12), "both copies of the seam move alike")


def test_cloth_rides_skin():
    body, cloth = torso(), tube()
    stroke = squish(body, cloth)
    n = stroke.body_count
    before = surface(*body).nearest(cloth[0]).distance
    after = surface(stroke.goal[:n], body[1]).nearest(stroke.goal[n:]).distance
    check(np.abs(after - before).max() < 0.0001, "the gap between the skin and the clothing is kept")
    check(np.abs(stroke.goal[n:] - cloth[0]).max() > 0.8 * P, "and the clothing moves with the skin")


def test_no_cloth_inside_goal():
    for label, cloth in (("open hem", tube()), ("turned-in hem", tube(fold=0.02))):
        body = torso()
        stroke = squish(body, cloth)
        n = stroke.body_count
        distance = surface(stroke.goal[:n], body[1]).nearest(stroke.goal[n:]).distance
        check(distance.min() >= -1e-5, label + ": no clothing inside the squished skin")


def test_loose_cloth_stays():
    body = torso()
    tight, loose = tube(), tube(radius=0.19)
    stroke = squish(body, joined(tight, loose))
    rows = stroke.body_count + len(tight[0]) + np.arange(len(loose[0]))
    check(np.array_equal(stroke.goal[rows], loose[0]), "clothing 7 cm out stays put")


def test_cut_away_skin():
    # Skin under the fabric cut away below z = -0.02, as game models often do.
    zs = np.linspace(-0.02, 0.2, 45)
    body = rings(np.full(len(zs), 0.12), zs, 0.005)
    cloth = tube()
    stroke = squish(body, cloth)
    # Past the cut, the press goes on as if the skin were there: to twice the Bulge Width under the hem.
    below = np.flatnonzero(cloth[0][:, 2] < HEM - 2.0 * W - 0.005)
    check(np.array_equal(stroke.goal[stroke.body_count + below], cloth[0][below]),
          "clothing farther under the fabric than the press reaches stays put")
    check(np.abs(stroke.goal[:stroke.body_count] - body[0]).max() > 0.9 * P, "the skin there is still squished")


def clothing_inside(stroke, body_tris):
    """How deep the clothing is inside the skin now, at most (0 when it isn't)."""
    n = stroke.body_count
    return max(0.0, -float(surface(stroke.co[:n], body_tris).nearest(stroke.co[n:]).distance.min()))


def full_dab(stroke):
    return stroke.dab(np.zeros(3), 1.0, 1.0, 'CONSTANT')


def test_dab_settles():
    body, cloth = torso(), tube()
    stroke = squish(body, cloth)
    moved = full_dab(stroke)
    check(np.abs(stroke.co - stroke.goal).max() < 1e-9, "a full dab reaches the squish")
    check(len(moved) and stroke.moved[moved].all(), "and reports what it moved")
    before = stroke.co.copy()
    for _ in range(10):
        full_dab(stroke)
    check(np.abs(stroke.co - before).max() < 1e-12, "going over it again changes nothing")


def test_rim_never_clips():
    body, cloth = torso(), tube()
    stroke = squish(body, cloth)
    for k in range(6):
        angle = 0.1 * k
        stroke.dab(np.array([0.123 * np.cos(angle), 0.123 * np.sin(angle), HEM]), 0.03, 0.5, 'SMOOTH')
        check(clothing_inside(stroke, body[1]) < 1e-5, "no clothing inside the skin after dab {:d}".format(k + 1))


def test_erase():
    body, cloth = torso(), tube()
    stroke = squish(body, cloth)
    full_dab(stroke)
    n = stroke.body_count
    eraser = squishing.SquishStroke(
        stroke.co[:n], body[0], body[1], np.ones(n, bool), stroke.co[n:], cloth[0], cloth[1],
        np.ones(len(cloth[0]), bool), invert=True,
    )
    full_dab(eraser)
    check(np.abs(eraser.co - np.concatenate((body[0], cloth[0]))).max() < 1e-9, "Ctrl puts everything back at Basis")


def test_nothing_without_edge():
    body, cloth = torso(), tube()
    stroke = squish(body, cloth)
    before = stroke.co.copy()
    # The press reaches twice the Bulge Width under the fabric: 14 cm.
    stroke.dab(np.array([0.123, 0.0, -0.19]), 0.02, 1.0, 'SMOOTH')
    stroke.dab(np.array([0.12, 0.0, 0.15]), 0.03, 1.0, 'SMOOTH')
    check(np.abs(stroke.co - before).max() < 1e-12, "far under the fabric and on bare skin nothing moves")


def test_hidden_and_fade():
    body, cloth = torso(), tube()
    z = body[0][:, 2]
    movable = z > -0.01 - 1e-9
    stroke = squish(body, cloth, body_movable=movable)
    full_dab(stroke)
    n = stroke.body_count
    check(np.array_equal(stroke.co[:n][~movable], body[0][~movable]), "hidden skin stays put")
    low = np.flatnonzero(cloth[0][:, 2] < -0.0151)
    check(np.array_equal(stroke.co[n + low], cloth[0][low]), "and so does the clothing over it")
    check(np.abs(stroke.co[:n][movable] - body[0][movable]).max() > 0.9 * P, "the rest is squished")

    faded = squish(body, cloth, body_movable=movable, fade_distance=0.02)
    full_dab(faded)
    row = np.flatnonzero(np.abs(z + 0.005) < 1e-9)
    full = np.linalg.norm(stroke.co[row] - body[0][row], axis=1)
    part = np.linalg.norm(faded.co[row] - body[0][row], axis=1)
    check(np.all(part > 1e-6) and np.all(part < full - 1e-6), "1 cm from the hidden skin, the move is faded")
    before = surface(*body).nearest(cloth[0]).distance
    after = surface(faded.co[:n], body[1]).nearest(faded.co[n:]).distance
    check(clothing_inside(faded, body[1]) < 1e-5 and np.abs(after - before).max() < 0.0003,
          "the clothing over faded skin fades alike")


def crossing_pairs(stroke, body_tris, cloth_tris, positions):
    """How many pairs of skin and clothing triangles cut through each other at ``positions`` (all rows)."""
    n = stroke.body_count
    body = BVHTree.FromPolygons(positions[:n].tolist(), body_tris.tolist(), all_triangles=True)
    cloth = BVHTree.FromPolygons(positions[n:].tolist(), cloth_tris.tolist(), all_triangles=True)
    return len(body.overlap(cloth))


def test_skin_border_held():
    # The skin of a top ends near its hem, where it meets the skin of other models: its open borders stay put where
    # they show.
    body, cloth = torso(height=0.03), tube()
    stroke = squish(body, cloth)
    z = body[0][:, 2]
    move = np.linalg.norm(stroke.goal[:stroke.body_count] - body[0], axis=1)
    check(move[np.abs(z - 0.015) < 1e-9].max() == 0.0, "the skin's open border past the fabric doesn't move")
    check(move[np.abs(z) < 1e-9].min() > 0.1 * P, "the edge 1.5 cm from it is still squished")


def test_skin_cut_under_fabric_pressed():
    # Skin cut away 7.5 mm under the hem: that border is hidden under the fabric, so the skin next to it is squished
    # fully, and the clothing past the cut goes on as if the skin were there.
    zs = np.linspace(-0.0075, 0.2, 84)
    body = rings(np.full(len(zs), 0.12), zs, 0.005)
    cloth = tube()
    stroke = squish(body, cloth)
    z = body[0][:, 2]
    move = np.einsum("ij,ij->i", stroke.goal[:stroke.body_count] - body[0], radial(body[0]))
    near_cut = z < 0.01
    error = np.abs(move[near_cut] - squishing.profile(z[near_cut] - HEM, P, H, W)).max()
    check(error < 0.0003, "the skin by the cut is squished fully (off by {:.4f} m)".format(error))
    n = stroke.body_count
    cloth_move = np.einsum("ij,ij->i", stroke.goal[n:] - cloth[0], radial(cloth[0]))
    cz = cloth[0][:, 2]
    past = cz < -0.01
    error = np.abs(cloth_move[past] - squishing.profile(cz[past] - HEM, P, H, W)).max()
    check(error < 0.0005, "the clothing past the cut is pressed as if the skin were there (off by {:.4f} m)".format(error))


def test_seam_welded_at_seam_distance():
    # The torso's seam split by 0.05 mm, more than the body's own weld but less than Seam Distance: the seam stays
    # closed, and isn't held like an open border of the skin.
    body = torso()
    co = body[0].copy()
    rows = round(0.4 / 0.005) + 1
    co[-rows:, 1] += 5e-5
    body = (co, body[1])
    stroke = squish(body, tube(), weld_distance=1e-4)
    move = stroke.goal[:stroke.body_count] - co
    first, last = np.arange(rows), len(co) - rows + np.arange(rows)
    check(np.allclose(move[first], move[last], rtol=0.0, atol=1e-9), "both sides of the seam move alike")
    along = np.einsum("ij,ij->i", move, radial(co))
    peak = np.argmin(np.abs(co[:rows, 2] - (HEM + squishing.BULGE_PEAK * W)))
    check(along[peak] > 0.9 * H, "the seam bulges out like the rest ({:.2f} mm)".format(1000 * along[peak]))


def test_bulge_fits_before_border():
    # The skin of a top ending 3 cm above its hem, where it meets other models' skin: the border stays put, and the
    # bulge fits between it and the hem, still all but its full height, with no step. Rows 0.8 mm apart, so a
    # step stands out from the steep but smooth fall.
    zs = 0.0325 - 0.0008 * np.arange(166)
    body = rings(np.full(len(zs), 0.12), zs, 0.005)
    stroke = squish(body, tube())
    z = body[0][:, 2]
    move = np.einsum("ij,ij->i", stroke.goal[:stroke.body_count] - body[0], radial(body[0]))
    rim = np.abs(z - zs[0]) < 1e-9
    check(np.abs(stroke.goal[:stroke.body_count][rim] - body[0][rim]).max() == 0.0, "the border doesn't move")
    check(move.max() > 0.9 * H, "the bulge still rises to {:.2f} mm".format(1000 * move.max()))
    row = np.array([move[np.abs(z - height) < 1e-9].mean() for height in zs[::-1]])
    steps = np.abs(np.diff(row))
    bends = np.abs(np.diff(row, 2))
    check(steps.max() < 0.15 * H and bends.max() < 0.05 * H,
          "no step ({:.2f} mm a row, bends {:.2f} mm at most)".format(1000 * steps.max(), 1000 * bends.max()))


def wavy_torso(step=0.005):
    """A torso whose radius goes in and out along it, 1 cm either way every 10 cm, and a tube 3 mm out over it
    to a hem 0.7 mm under the row at z = 0, where the torso is steepest."""
    zs = np.linspace(-0.2, 0.2, 81)
    radius = 0.12 + 0.01 * np.sin(2.0 * np.pi * zs / 0.1)
    body = rings(radius, zs, step)
    cloth_zs = np.concatenate((np.linspace(-0.21, -0.005, 42), [-0.0007]))
    cloth = rings(0.123 + 0.01 * np.sin(2.0 * np.pi * cloth_zs / 0.1), cloth_zs, step)
    return body, cloth


def test_cover_whatever_width():
    # Where the fabric covers the skin doesn't change with Bulge Width.
    body, cloth = wavy_torso()
    narrow = squish(body, cloth, width=0.02).field.covered.copy()
    wide = squish(body, cloth, width=0.15).field.covered
    check(np.array_equal(narrow, wide), "the same skin is covered at 2 cm and 15 cm ({} vertices differ)".format(
        int((narrow != wide).sum())))


def test_strap_stays_pressed():
    # A strap 2.25 cm wide: pressed in all across, flat along its middle rather than folded into a V there.
    body = torso(step=0.0025)
    cloth = tube(z0=-0.01125, z1=0.01125, step=0.0025)
    stroke = squish(body, cloth)
    z = body[0][:, 2]
    move = np.einsum("ij,ij->i", stroke.goal[:stroke.body_count] - body[0], radial(body[0]))
    row = {k: move[np.abs(z - 0.0025 * k) < 1e-9].mean() for k in range(-4, 5)}
    check(row[0] < -0.5 * P, "the middle of the strap is pressed in ({:.2f} mm)".format(1000 * row[0]))
    check(row[0] <= min(row.values()) + 1e-9, "and is its lowest")
    near, next_ = abs(row[1] - row[0]), abs(row[2] - row[1])
    check(near < 0.5 * next_, "flat along the middle, not a V ({:.2f} then {:.2f} mm a row)".format(
        1000 * near, 1000 * next_))


def test_loose_cloth_eases_out():
    # Loose clothing 4 to 7 cm off the skin, over the bulge 2 cm past a hem: it moves less and less the farther out
    # it is, not all the way up to twice Max Gap and then not at all.
    gaps = np.arange(0.04, 0.0701, 0.0025)
    body = torso()
    meshes = [tube()] + [tube(radius=0.12 + g, z0=0.012, z1=0.028) for g in gaps]
    stroke = squish(body, joined(*meshes))
    n = stroke.body_count
    first = n + len(meshes[0][0])
    moves = []
    for mesh in meshes[1:]:
        rows = np.arange(first, first + len(mesh[0]))
        moves.append(np.linalg.norm(stroke.goal[rows] - stroke.basis[rows], axis=1).max())
        first += len(mesh[0])
    moves = np.array(moves)
    steps = np.abs(np.diff(moves))
    check(steps.max() < 0.0015, "no step from one ring to the next ({:.2f} mm at most)".format(1000 * steps.max()))
    check(moves[gaps >= 0.06 - 1e-9].max() < 1e-9, "nothing from twice Max Gap")


def test_smoothing_scales():
    # A wide Roll: smoothing mustn't look at every pair of vertices in reach (here, about 4000 per vertex).
    body = surface(*torso())
    start = time.perf_counter()
    normals = squishing.smoothed_normals(body, 0.1)
    took = time.perf_counter() - start
    check(took < 1.0, "{} vertices smoothed over 10 cm in {:.2f} s".format(len(body.co), took))
    out = radial(body.co)
    middle = np.abs(body.co[:, 2]) < 0.1
    check(np.einsum("ij,ij->i", normals[middle], out[middle]).min() > 0.99, "and still facing out")


def test_close_cloth_never_crosses():
    # Coarse fabric 0.5 mm off the skin: straight between its vertices, it would cut under the skin where the skin
    # bends at the roll.
    body, cloth = torso(), tube(radius=0.1205, step=0.02)
    stroke = squish(body, cloth)
    check(crossing_pairs(stroke, body[1], cloth[1], stroke.basis) == 0, "no crossings at Basis")
    crossed = crossing_pairs(stroke, body[1], cloth[1], stroke.goal)
    check(crossed == 0, "none once squished ({} pairs)".format(crossed))


def grooved_torso(depth=0.003, width=0.003):
    """A torso meshed finely (1 mm rows) around a groove ``depth`` deep just above the hem, like a navel in a body
    that is coarse elsewhere."""
    zs = np.concatenate((np.linspace(-0.2, 0.0, 41)[:-1], np.linspace(0.0, 0.03, 31), np.linspace(0.03, 0.2, 35)[1:]))
    co, tris = rings(np.full(len(zs), 0.12), zs, 0.005)
    dent = depth * np.exp(-((co[:, 2] - 0.012) / width) ** 2)
    co[:, :2] *= (1.0 - dent / 0.12)[:, None]
    return co, tris


def test_normals_even_where_dense():
    body = surface(*grooved_torso())
    normals = squishing.smoothed_normals(body, R)
    groove = np.abs(body.co[:, 2] - 0.012) < 0.006
    check(np.abs(normals[groove, 2]).max() < 0.2, "normals smoothed over the groove too (max lean {:.2f})".format(
        np.abs(normals[groove, 2]).max()))


def test_no_skin_turns_over():
    for label, body in (("groove", grooved_torso()), ("deep narrow groove", grooved_torso(0.01, 0.0015))):
        stroke = squish(body, tube())
        full_dab(stroke)
        tris = body[1]
        n0 = np.cross(body[0][tris[:, 1]] - body[0][tris[:, 0]], body[0][tris[:, 2]] - body[0][tris[:, 0]])
        after = stroke.co[:stroke.body_count]
        n1 = np.cross(after[tris[:, 1]] - after[tris[:, 0]], after[tris[:, 2]] - after[tris[:, 0]])
        turned = int((np.einsum("ij,ij->i", n0, n1) < 0.0).sum())
        check(turned == 0, "{}: no skin triangle turns over ({})".format(label, turned))


def test_keep_facing():
    index = np.arange(21 * 21).reshape(21, 21)
    co = np.array([(x * 0.001, y * 0.001, 0.0) for x in range(21) for y in range(21)])
    tris = grid_tris(index)
    move = np.zeros_like(co)
    move[:, 2] = 0.002
    move[index[10, 10]] = (0.005, 0.0, 0.002)  # across its neighbours
    kept = squishing.keep_facing(co, tris, move)
    after = co + kept
    n = np.cross(after[tris[:, 1]] - after[tris[:, 0]], after[tris[:, 2]] - after[tris[:, 0]])
    check((n[:, 2] > 0.0).all(), "no triangle turns over")
    far = np.ones(len(co), dtype=bool)
    far[index[7:14, 7:14].ravel()] = False
    check(np.array_equal(kept[far], move[far]), "moves away from it are kept")


def test_keep_facing_after_holding():
    # A sliver (0.1 micrometres high) turned over by any part of its move, so its corners are held at last; holding
    # them must not turn over the triangle next to it, whose far corner moves down past them.
    co = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, -1e-7, 0.0), (0.5, 0.01, 0.0)])
    tris = np.array([(0, 2, 1), (0, 1, 3)])
    move = np.array([(0.0, -0.02, 0.0), (0.0, -0.02, 0.0), (0.0, 0.02, 0.0), (0.0, -0.0103, 0.0)])
    kept = squishing.keep_facing(co, tris, move)
    after = co + kept
    n = np.cross(after[tris[:, 1]] - after[tris[:, 0]], after[tris[:, 2]] - after[tris[:, 0]])
    check((n[:, 2] > 0.0).all(), "no triangle turns over once the sliver is held ({})".format(n[:, 2].tolist()))


def test_invert_goal_is_basis():
    body, cloth = torso(), tube()
    stroke = squish(body, cloth, invert=True)
    check(np.array_equal(stroke.goal, np.concatenate((body[0], cloth[0]))), "erasing aims at Basis")


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
