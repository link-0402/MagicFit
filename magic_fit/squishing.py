# SPDX-License-Identifier: GPL-3.0-or-later

"""Skin squished by the edge of tight clothing, for the Squish brush.

Tight fabric squeezes the skin it covers, and the flesh spills over its edge: under the fabric the skin is pressed
in, and around the edge it bulges out in a broad, smooth swell that the fabric's edge rides on, peaking a little past
the edge (see `profile`). Clothing over the skin moves exactly as the skin under it does, so the gap
between them is kept and nothing clips. Everything is aimed at one squished shape worked out from the unsquished
shape (the meshes' Basis), so going over a spot again completes the squish instead of piling it up.

Like `fitting`, nothing here depends on Blender's data: positions are world space.
"""

import hashlib

import numpy as np
from mathutils.bvhtree import BVHTree

from .fitting import WELD_DISTANCE, BodySurface, _mesh_distance, weld_groups
from .moving import FitMove
from .painting import falloff_weights

# The squish's shape, in Bulge Widths (see `profile`), from Cultist's squished Rue Medium tops (at their 7 cm width):
# pressed in fully from 2.8 cm under the fabric to 8.4 cm, and not at all from 14 cm; the bulge rising from 2.1 cm
# under the fabric, half way up at the edge, at its peak 2.1 cm past it, half way down at 4.5 cm and gone at 7 cm.
PRESS_FULL = 0.4
PRESS_END = 1.2
BULGE_START = 0.3
BULGE_PEAK = 0.3
# Body normals are smoothed over this many Bulge Widths for the way the skin moves.
SMOOTH_WIDTHS = 0.4
# And over this distance for the rays that find what the fabric covers, so the fabric edge is found in the same
# place whatever the Bulge Width.
COVER_SMOOTH = 0.02

# Clothing the squished skin comes closer to than its gap, and closer than this, is lifted back to the smaller of the
# two; by less than `LIFT_TOLERANCE`, it's left.
LIFT_KEEP = 0.003
LIFT_TOLERANCE = 1e-4
# Clothing whose triangles cut through the squished skin is pushed out by this much at a time, at most this many
# times (see `_uncross`).
UNCROSS_STEP = 0.0005
UNCROSS_PASSES = 20
# The skin's open borders stay put, and the bulge is fitted into the room before them (see `fit_room`). In case
# something is left there, the squish also eases in over this distance from them.
BORDER_FADE = 0.005
# Where the squish would turn skin triangles over, their corners move this much less at a time, at most this many
# times before they don't move at all (see `keep_facing`).
FACING_CUT = 0.7
FACING_PASSES = 10

# The fabric edge last found, by what it was found from (see `fabric_edge`).
_last_edge = (None, None)

# Clothing this far inside the skin (clipping a little already) still covers it.
COVER_INSIDE = 0.002
# Rays that miss the clothing are cast again from this far beside where they started, in case they slipped
# between two triangles.
COVER_NUDGE = 1e-4
# At most this many passes of setting vertices that disagree with most of their neighbours right (see `fabric_edge`).
VOTE_PASSES = 10
# Halvings of a body edge to find where the fabric edge crosses it: 1/64 of the edge.
EDGE_STEPS = 6
# Smoothed normals that add up to less than this share of the normals around (opposite sides of something thin
# cancelling out) are left as the vertex's own (see `smoothed_normals`).
CANCEL = 0.3
# Cells (half the smoothing radius wide) this many along each axis either way can hold points within the radius.
CELL_REACH = 2


class EdgeField:
    """Where the fabric edge lies on the body, per welded body vertex (see `fabric_edge`).

    ``normals`` are the body's normals smoothed along it (unit length), ``covered`` marks the vertices with clothing
    over them, and ``s`` is the distance from the fabric edge along the body: negative under the fabric, infinite
    beyond reach.
    """

    __slots__ = ("normals", "covered", "s")


class _Cells:
    """Points (n, 3) gathered in cubes ``size`` wide, to find what lies around other points fast, however many points
    that is. Each cell stands at the middle of its points, by their ``weights``."""

    def __init__(self, co, size, weights):
        self.size = size
        keys, member = np.unique(np.floor(co / size).astype(np.int64), axis=0, return_inverse=True)
        self.member = member.ravel()
        self.count = len(keys)
        plain = np.stack([np.bincount(self.member, co[:, k], minlength=self.count) for k in range(3)], axis=1)
        plain /= np.bincount(self.member, minlength=self.count)[:, None]
        total = self.sum(weights)
        middle = np.stack([self.sum(weights * co[:, k]) for k in range(3)], axis=1)
        self.middle = np.where(total[:, None] > 0.0, middle / np.maximum(total, 1e-30)[:, None], plain)
        # Cells by a code of their place.
        self._low = keys.min(axis=0)
        self._span = keys.max(axis=0) - self._low + 1
        codes = self._code(keys)
        self._order = np.argsort(codes)
        self._codes = codes[self._order]

    def _code(self, places):
        places = places - self._low
        inside = np.all((places >= 0) & (places < self._span), axis=1)
        code = (places[:, 0] * self._span[1] + places[:, 1]) * self._span[2] + places[:, 2]
        return np.where(inside, code, -1)

    def sum(self, values):
        """``values`` per point, added up per cell."""
        return np.bincount(self.member, values, minlength=self.count)

    def around(self, points, reach):
        """For every cell up to ``reach`` cells along each axis from each of ``points``: (the points, their cells)."""
        cell = np.floor(points / self.size).astype(np.int64)
        steps = range(-reach, reach + 1)
        for offset in ((x, y, z) for x in steps for y in steps for z in steps):
            wanted = self._code(cell + offset)
            at = np.minimum(np.searchsorted(self._codes, wanted), self.count - 1)
            there = np.flatnonzero((self._codes[at] == wanted) & (wanted >= 0))
            yield there, self._order[at[there]]


def narrowed(co, s, covered, cut, full):
    """``s`` (per point at ``co``) stretched under narrow strips of fabric (a strap, a garter), so the middle of the
    strip lies ``full`` under the edge, where the press is full and flat, rather than on a ridge between its two
    edges where the squish from both sides would meet in a V (a strip 2 cm wide would hardly be pressed, and fold
    lengthwise). Each covered point takes the strip's half-width as the farthest any covered point within ``full``
    lies from an edge, counted less from up to twice that far, so it changes smoothly from point to point. Where the
    skin was ``cut`` away under the fabric, the fabric goes on: that counts as wide."""
    under = covered & np.isfinite(s)
    if not under.any():
        return s
    depth = -s[under]
    cells = _Cells(co[under], full, np.ones(len(depth)))
    deepest = np.zeros(cells.count)
    np.maximum.at(deepest, cells.member, np.where(cut[under], 2.0 * full, depth))
    half = np.zeros(len(depth))
    for there, other in cells.around(co[under], 2):
        distance = np.linalg.norm(co[under][there] - cells.middle[other], axis=1)
        reach = deepest[other] * (1.0 - smoothstep((distance - full) / full))
        np.maximum.at(half, there, reach)
    half = np.maximum(half, depth)
    result = s.copy()
    result[under] = -depth * np.maximum(full / np.maximum(half, 1e-9), 1.0)
    return result


def smoothed_normals(body, radius):
    """The body's normals (a `fitting.BodySurface`, welded) averaged over ``radius`` around each vertex (in space,
    not along the body), so neighbours move along nearly the same line and never cross, however finely the body is
    meshed there. Each counts by its area, less with distance. Both sides of a pit or a crease (a navel, under the
    bust) count, so they move the same way, out of it, instead of into each other; where they nearly cancel out
    (something thin), a vertex keeps its own normal."""
    co = body.co
    count = len(co)
    # The normals are gathered in cells half the radius wide, each standing at the middle of its area, so this
    # takes as long however many vertices lie within the radius.
    area = np.linalg.norm(body.area_normals, axis=1)
    cells = _Cells(co, 0.5 * radius, area)
    cell_normal = np.stack([cells.sum(body.area_normals[:, k]) for k in range(3)], axis=1)
    cell_area = cells.sum(area)
    total = np.zeros((count, 3))
    whole = np.zeros(count)
    for there, other in cells.around(co, CELL_REACH):
        distance = np.linalg.norm(co[there] - cells.middle[other], axis=1)
        weight = np.clip(1.0 - (distance / radius) ** 2, 0.0, None) ** 2
        total[there] += weight[:, None] * cell_normal[other]
        whole[there] += weight * cell_area[other]
    size = np.linalg.norm(total, axis=1)
    kept = size > CANCEL * whole
    normals = body.normals.copy()
    normals[kept] = total[kept] / size[kept, None]
    return normals


def fabric_edge(body, cloth_co, cloth_tris, *, max_gap, reach, smooth_radius, cover_radius=COVER_SMOOTH):
    """Find the fabric edge on ``body`` (a `fitting.BodySurface`) under the clothing (world space positions and
    triangles), see `EdgeField`; its normals are smoothed over ``smooth_radius``.

    A body vertex is covered when a ray out along its normal smoothed over ``cover_radius`` meets the clothing
    between `COVER_INSIDE` inside the skin and ``max_gap`` out. The edge is found on the body, where covered vertices
    border uncovered ones, so a hem turned in, or layers of a part, count as well as an open border of the mesh.
    Along each of the body's edges that cross it, the same test finds where it lies between their ends, so the
    distances don't step along the body's rows where the edge cuts across them. Distances from it go along the
    body, up to ``reach``.
    """
    global _last_edge
    cloth_co = np.asarray(cloth_co, dtype=np.float64)
    cloth_tris = np.asarray(cloth_tris, dtype=np.int64).reshape(-1, 3)
    # Stroke after stroke over the same meshes, everything here is the same: it's all measured at Basis.
    key = hashlib.sha1(b"".join(np.ascontiguousarray(array).tobytes() for array in (
        body.co, body.corners, cloth_co, cloth_tris,
        np.array([max_gap, reach, smooth_radius, cover_radius])))).digest()
    if _last_edge[0] != key:
        _last_edge = (key, _find_edge(body, cloth_co, cloth_tris, max_gap, reach, smooth_radius, cover_radius))
    return _last_edge[1]


def clear_cache():
    global _last_edge
    _last_edge = (None, None)


def _find_edge(body, cloth_co, cloth_tris, max_gap, reach, smooth_radius, cover_radius):
    """`fabric_edge`, worked out."""
    field = EdgeField()
    field.normals = smoothed_normals(body, smooth_radius)
    rays = field.normals if cover_radius == smooth_radius else smoothed_normals(body, cover_radius)
    count = len(body.co)
    covers = _cover_test(cloth_co, cloth_tris, max_gap)
    covered = covers(body.co, rays)
    edges = body.edges
    if len(edges):
        # A ray can slip between two triangles along their shared edge, and rays along a seam of the clothing can
        # all do it, missing a whole line of vertices. A vertex most of whose neighbours disagree with it goes with
        # them, so no line of vertices makes edges of its own in the middle of the fabric. (On a real edge, at
        # most half of a vertex's neighbours lie across it.) Repeated, since the end of such a line only goes once
        # the vertex next to it has.
        tail = np.concatenate((edges[:, 0], edges[:, 1]))
        head = np.concatenate((edges[:, 1], edges[:, 0]))
        degree = np.bincount(tail, minlength=count)
        for _ in range(VOTE_PASSES):
            flip = 2 * np.bincount(tail, covered[head] != covered[tail], minlength=count) > degree
            if not flip.any():
                break
            covered ^= flip
    field.covered = covered

    crossing = edges[covered[edges[:, 0]] != covered[edges[:, 1]]] if len(edges) else edges
    if not len(crossing):
        field.s = np.where(covered, -np.inf, np.inf)
        return field
    # Where along each crossing edge (from its covered end) the edge lies, by halving the stretch it's in.
    crossing = np.where(covered[crossing[:, 0]][:, None], crossing, crossing[:, ::-1])
    a, b = body.co[crossing[:, 0]], body.co[crossing[:, 1]]
    na, nb = rays[crossing[:, 0]], rays[crossing[:, 1]]
    low, high = np.zeros(len(crossing)), np.ones(len(crossing))
    for _ in range(EDGE_STEPS):
        t = 0.5 * (low + high)
        normal = na + t[:, None] * (nb - na)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
        inside = covers(a + t[:, None] * (b - a), normal)
        low = np.where(inside, t, low)
        high = np.where(inside, high, t)
    t = 0.5 * (low + high)
    lengths = np.linalg.norm(b - a, axis=1)
    # Each vertex next to the edge starts from its distance to the closest point found on its edges.
    start = np.full(count, np.inf)
    np.minimum.at(start, crossing[:, 0], t * lengths)
    np.minimum.at(start, crossing[:, 1], (1.0 - t) * lengths)
    border = np.isfinite(start)
    start = np.where(border, start, 0.0)
    inside, _ = _mesh_distance(body.co, edges, border & covered, reach, start=start)
    outside, _ = _mesh_distance(body.co, edges, border & ~covered, reach, start=start)
    field.s = np.where(covered, -inside, outside)
    return field


def _cover_test(cloth_co, cloth_tris, max_gap):
    """A test whether the clothing covers body points (n, 3) with normals (n, 3), see `fabric_edge`: returns a
    mask."""
    if not len(cloth_tris):
        return lambda points, normals: np.zeros(len(points), dtype=bool)
    tree = BVHTree.FromPolygons(np.asarray(cloth_co).tolist(), np.asarray(cloth_tris).tolist(), all_triangles=True)
    cast = tree.ray_cast
    limit = max_gap + COVER_INSIDE

    def covers(points, normals):
        origins = points - COVER_INSIDE * normals
        result = np.array([cast(origin, normal, limit)[0] is not None
                           for origin, normal in zip(origins.tolist(), normals.tolist())], dtype=bool)
        # A ray can slip between two triangles along their shared edge, or through a seam whose sides don't quite
        # meet: missed rays are cast again from just beside where they started.
        missed = np.flatnonzero(~result)
        if len(missed):
            side = np.cross(normals[missed], np.where(np.abs(normals[missed, :1]) < 0.9, [[1.0, 0.0, 0.0]],
                                                      [[0.0, 1.0, 0.0]]))
            side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-30)
            other = np.cross(normals[missed], side)
            for offset in (side, -side, other, -other):
                for k, origin, normal in zip(missed.tolist(), (origins[missed] + COVER_NUDGE * offset).tolist(),
                                             normals[missed].tolist()):
                    if not result[k] and cast(origin, normal, limit)[0] is not None:
                        result[k] = True
        return result

    return covers


def smoothstep(x):
    """0 at 0 and below, 1 at 1 and above, easing in and out between."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def profile(s, press, bulge, width):
    """How far the skin moves out (negative: in) at distances ``s`` from the fabric edge along the body: negative
    under the fabric, positive past it.

    Tight fabric squeezes what it covers and the flesh spills over its edge (as measured on hand-made squish): the
    covered skin is pressed in by ``press``, fully from `PRESS_FULL` x ``width`` under the edge, and back to nothing
    by twice the width under it. A bulge ``bulge`` high rises from `BULGE_START` x ``width`` under the edge, so the
    edge itself rides on it rather than in a dent, peaks `BULGE_PEAK` x ``width`` past the edge, and is flat again
    ``width`` past it. Infinite distances (no edge in reach) don't move.
    """
    s = np.asarray(s, dtype=np.float64)
    result = np.zeros(s.shape)
    known = np.isfinite(s)
    under = known & (s < 0.0)
    depth = -s[under]
    result[under] = -press * smoothstep(depth / (PRESS_FULL * width)) * (
        1.0 - smoothstep((depth - PRESS_END * width) / ((2.0 - PRESS_END) * width)))
    rising = known & (s > -BULGE_START * width) & (s <= BULGE_PEAK * width)
    result[rising] += bulge * smoothstep((s[rising] + BULGE_START * width) / ((BULGE_START + BULGE_PEAK) * width))
    falling = known & (s > BULGE_PEAK * width) & (s < width)
    result[falling] += bulge * (1.0 - smoothstep((s[falling] - BULGE_PEAK * width) / ((1.0 - BULGE_PEAK) * width)))
    return result


def fit_room(s, from_border, width):
    """``s`` (distances from the fabric edge, see `profile`) stretched past the edge where an open border of the
    skin that stays put lies closer to the edge than the bulge reaches (``width``): the bulge is fitted into the
    room it has, so it still rises to its full height and is flat again right at the border, instead of being cut
    off there. The room is how far the edge lies from the border through each point (``from_border`` its distance
    from the border). The stretch grows with the square of the distance from the edge, so at the edge the bulge
    is still half way up and rises as steeply as without it."""
    s = np.asarray(s, dtype=np.float64)
    result = s.copy()
    past = np.flatnonzero(np.isfinite(s) & np.isfinite(from_border) & (s > 0.0))
    room = s[past] + from_border[past]
    tight, room = past[room < width], room[room < width]
    result[tight] += (width - room) * (s[tight] / room) ** 2
    return result


def keep_facing(co, tris, move):
    """``move`` (per vertex of ``co``), cut back where it would turn triangles over: the corners of each such
    triangle move less and less, a step at a time, until none turns over (or, at last, not at all). A navel or other
    small pit meshed far finer than the squish is wide can't take it whole. Corners that don't move at all can turn
    over the triangles next to them in turn (a sliver held, its neighbour's far corner moving past it): those don't
    move either, until none turns over."""
    move = move.copy()
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    before = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    tris = tris[np.linalg.norm(before, axis=1) > 0.0]
    before = before[np.linalg.norm(before, axis=1) > 0.0]
    # Each step past the cut ones holds at least one more vertex (a triangle whose corners are all held is as it
    # was), so this ends.
    step = 0
    while True:
        after = co + move
        turned = np.einsum("ij,ij->i", before,
                           np.cross(after[tris[:, 1]] - after[tris[:, 0]], after[tris[:, 2]] - after[tris[:, 0]])) <= 0.0
        if not turned.any():
            break
        corners = np.unique(tris[turned].ravel())
        move[corners] *= FACING_CUT if step < FACING_PASSES else 0.0
        step += 1
    return move


def _crossings(body_co, body_tris, cloth_co, cloth_tris):
    """The pairs of skin and clothing triangles that cut through each other, as a set of (skin, clothing)."""
    body = BVHTree.FromPolygons(body_co.tolist(), body_tris.tolist(), all_triangles=True)
    cloth = BVHTree.FromPolygons(cloth_co.tolist(), cloth_tris.tolist(), all_triangles=True)
    return set(body.overlap(cloth))


def _uncross(body_basis, body_goal, body_tris, cloth_basis, cloth_goal, cloth_tris, normals):
    """Push the clothing out of the squished skin where their triangles cut through each other, though their
    vertices don't: fabric whose triangles are larger than the skin's, close over it, runs straight under the skin
    where it bends. The clothing's vertices of such triangles go out along the skin's ``normals`` under them
    (changes ``cloth_goal``), a little at a time. Crossings already there at Basis are left as they are."""
    body_tris = np.asarray(body_tris, dtype=np.int64).reshape(-1, 3)
    before = _crossings(body_basis, body_tris, cloth_basis, cloth_tris)
    for _ in range(UNCROSS_PASSES):
        crossed = _crossings(body_goal, body_tris, cloth_goal, cloth_tris) - before
        if not crossed:
            return
        vertices = np.unique(cloth_tris[[cloth for _body, cloth in crossed]].ravel())
        cloth_goal[vertices] += UNCROSS_STEP * normals[vertices]


def _surface(co, tris):
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    return BodySurface(co, tris, BVHTree.FromPolygons(co.tolist(), tris.tolist(), all_triangles=True))


def _welded_surface(co, tris, distance):
    """The body (positions ``co``, triangles ``tris``) as a `fitting.BodySurface` whose vertices closer than
    ``distance`` are welded, so a seam split wider than the surface's own weld still stays closed and isn't taken
    for an open border of the skin. Returns (the surface, vertex -> welded vertex of the surface)."""
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if distance <= WELD_DISTANCE:
        body = _surface(co, tris)
        return body, body._weld
    groups, count = weld_groups(co, distance)
    merged = np.zeros((count, 3))
    np.add.at(merged, groups, co)
    merged /= np.bincount(groups, minlength=count)[:, None]
    tris = groups[tris]
    # Triangles with corners welded together have nothing left to them.
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])]
    body = _surface(merged, tris)
    return body, body._weld[groups]


class SquishStroke:
    """Squishes the body's skin along the fabric edge of the clothing over it, dab by dab.

    Positions are world space: the body's and the clothing's as they are now (``body_co``, ``cloth_co``) and at
    their Basis (``body_basis``, ``cloth_basis``), with their triangles and which vertices may move. The goal, the
    fully squished shape, is worked out once from Basis (see `fabric_edge` and `profile`): the skin moves along its
    smoothed normal by the profile of its distance from the fabric edge. Each clothing vertex is bound to its
    closest point on the skin and moves as the skin there does, all the way within ``max_gap`` of the skin and
    less and less up to twice that, so loose clothing isn't dragged along. Clothing whose closest point lies past
    an open edge of the body (where the skin under the fabric was cut away) is squished as if the skin went on.
    Clothing the squished skin would come closer to than its gap and `LIFT_KEEP` is lifted back to the smaller of
    the two. Under narrow strips of fabric the profile is stretched (see `narrowed`), and so it is before the skin's
    open borders that show, which stay put (see `fit_room`). Body vertices closer than ``weld_distance`` move as one.
    With ``invert``, the goal is Basis.

    ``press``, ``bulge`` and ``width`` shape the squish (see `profile`).

    ``co`` holds the body's positions, then the clothing's; ``goal`` likewise; ``moved`` flags the vertices moved so
    far, and ``body_count`` is where the clothing's begin.
    """

    def __init__(
            self, body_co, body_basis, body_tris, body_movable, cloth_co, cloth_basis, cloth_tris, cloth_movable, *,
            press=0.003, bulge=0.009, width=0.07, max_gap=0.03, body_edges=None, cloth_edges=None,
            weld_distance=WELD_DISTANCE, fade_distance=0.0, invert=False,
    ):
        body_basis = np.asarray(body_basis, dtype=np.float64).reshape(-1, 3)
        cloth_basis = np.asarray(cloth_basis, dtype=np.float64).reshape(-1, 3)
        cloth_tris = np.asarray(cloth_tris, dtype=np.int64).reshape(-1, 3)
        self.body_count = len(body_basis)
        self.basis = np.concatenate((body_basis, cloth_basis))
        self.co = np.concatenate((np.asarray(body_co, dtype=np.float64).reshape(-1, 3),
                                  np.asarray(cloth_co, dtype=np.float64).reshape(-1, 3)))
        self.movable = np.concatenate((np.asarray(body_movable, dtype=bool), np.asarray(cloth_movable, dtype=bool)))
        self.moved = np.zeros(len(self.co), dtype=bool)

        body, weld = _welded_surface(body_basis, body_tris, weld_distance)
        self.body = body
        self.field = fabric_edge(body, cloth_basis, cloth_tris, max_gap=max_gap, reach=2.0 * width,
                                 smooth_radius=SMOOTH_WIDTHS * width)
        # The skin's move, per welded vertex.
        s = narrowed(body.co, self.field.s, self.field.covered, body.open_vertex & self.field.covered,
                     PRESS_FULL * width)
        held = body.open_vertex & ~self.field.covered
        from_border = None
        if held.any():
            # The skin's open borders stay put where they show: there it meets the skin of other models (a top's
            # skin ends at the waist, the neck and the wrists), which would come apart. Under the fabric, where the
            # skin was cut away, nobody sees it.
            from_border, _ = _mesh_distance(body.co, body.edges, held, width)
            s = fit_room(s, from_border, width)
        move = profile(s, press, bulge, width)[:, None] * self.field.normals
        if from_border is not None:
            move *= smoothstep(from_border / BORDER_FADE)[:, None]
        move = keep_facing(body.co, body.corners, move)
        body_goal = body_basis + move[weld]

        # Each clothing vertex rides on its closest point of the skin.
        near = body.nearest(cloth_basis)
        gap = np.abs(near.distance)
        share = 1.0 - smoothstep((gap - max_gap) / max_gap)
        share[~near.found] = 0.0
        normals = np.einsum("ij,ijk->ik", near.bary, self.field.normals[near.corners])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        # Past where the skin was cut away (under the fabric), there's no skin to ride on: the clothing is squished
        # as if the skin went on, by the distance along it from the cut.
        past = np.flatnonzero(near.found & near.beyond)
        extra = np.zeros_like(cloth_basis)
        if len(past):
            corners = s[near.corners[past]]
            known = np.isfinite(corners).all(axis=1)
            at_cut = np.where(known, np.einsum("ij,ij->i", near.bary[past], np.where(np.isfinite(corners), corners, 0.0)),
                              -np.inf)
            offset = cloth_basis[past] - near.location[past]
            along = np.linalg.norm(offset - np.einsum("ij,ij->i", offset, normals[past])[:, None] * normals[past], axis=1)
            beyond = np.where(at_cut <= 0.0, at_cut - along, at_cut + along)
            extra[past] = profile(beyond, press, bulge, width)[:, None] * normals[past]
            share[past] = 0.0
        self.bind_corners = near.corners
        self.bind_bary = near.bary
        self.bind_share = share
        cloth_goal = cloth_basis + extra + share[:, None] * np.einsum("ij,ijk->ik", near.bary, move[near.corners])

        if len(cloth_basis) and (share > 0.0).any():
            # Where the squished skin would come closer than it was, and closer than `LIFT_KEEP`, the clothing is lifted
            # back to that: clothing close on the skin keeps its gap, and loose clothing farther out is left as it
            # rides, less and less with its gap.
            after = _surface(body_goal, body_tris).nearest(cloth_goal)
            keep = np.minimum(near.distance, LIFT_KEEP)
            lift = (share > 0.0) & (near.distance > 0.0) & after.found & (after.distance < keep - LIFT_TOLERANCE)
            cloth_goal[lift] += (keep[lift] - after.distance[lift])[:, None] * after.normal[lift]
            cloth_goal = cloth_basis + keep_facing(cloth_basis, cloth_tris, cloth_goal - cloth_basis)
            _uncross(body_basis, body_goal, body_tris, cloth_basis, cloth_goal, cloth_tris, normals)

        self.goal = self.basis.copy() if invert else np.concatenate((body_goal, cloth_goal))

        # Welded skin moves only when all of it may.
        self.body_weld = weld
        self.welded_movable = np.ones(len(body.co), dtype=bool)
        np.logical_and.at(self.welded_movable, weld, self.movable[:self.body_count])
        # Clothing only moves where all of the skin it rides on may.
        self.bind_movable = self.welded_movable[near.corners].all(axis=1) & self.movable[self.body_count:]
        # How much of a dab each vertex takes near the vertices that may not move (see `moving.FitMove`).
        self.fade = np.ones(len(self.co))
        if fade_distance > 0.0:
            for rows, basis, tris, edges in (
                    (slice(0, self.body_count), body_basis, body_tris, body_edges),
                    (slice(self.body_count, None), cloth_basis, cloth_tris, cloth_edges),
            ):
                movable = self.movable[rows]
                if not len(basis) or movable.all():
                    continue
                tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
                if edges is None:
                    edges = np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))
                move = FitMove(basis, np.asarray(edges, dtype=np.int64), np.zeros(len(basis), dtype=bool), movable,
                               tris=tris, weld_distance=weld_distance, fade_distance=fade_distance)
                self.fade[rows] = move.fade[move.weld]
        # The skin's fade, per welded vertex: the clothing over it takes it too.
        self.welded_fade = np.ones(len(body.co))
        np.minimum.at(self.welded_fade, weld, self.fade[:self.body_count])

        # The clothing rides on the skin: it moves as the skin under it has moved since the stroke began (a vertex
        # of each welded group stands for it). What its goal adds to that (lifting it back to its gap) comes in step
        # by step, by the weight of the skin under it.
        self.start = self.co.copy()
        self.welded_rep = np.zeros(len(body.co), dtype=np.int64)
        self.welded_rep[weld] = np.arange(self.body_count)
        self.residual = self.goal[self.body_count:] - self.start[self.body_count:] - self._ride(self.goal)
        self.progress = np.zeros(len(cloth_basis))

    def _ride(self, positions):
        """How far each clothing vertex rides along with the skin, for the skin at the body rows of ``positions``."""
        rep = self.welded_rep
        moved = positions[rep] - self.start[rep]
        return self.bind_share[:, None] * np.einsum("ij,ijk->ik", self.bind_bary, moved[self.bind_corners])

    def dab(self, center, radius, strength, falloff='SMOOTH'):
        """Apply one dab around the world space point ``center``: everything under it moves ``strength`` x the
        brush ``falloff`` of the remaining way to its goal. Returns the indices of the vertices it moved.

        The falloff is measured from the skin at Basis. The clothing moves exactly as the skin under it has (see
        `_ride`), so the skin and the fabric over it always move alike, even at the brush's rim.
        """
        nothing = np.empty(0, dtype=np.int64)
        if radius <= 0.0 or strength <= 0.0:
            return nothing
        n = self.body_count
        center = np.asarray(center, dtype=np.float64)
        distance = np.linalg.norm(self.body.co - center, axis=1) / radius
        welded = np.where((distance < 1.0) & self.welded_movable, falloff_weights(falloff, distance), 0.0)
        welded *= self.welded_fade
        weight = welded[self.body_weld]
        body_rows = np.flatnonzero(weight > 0.0)
        self.co[body_rows] += (strength * weight[body_rows])[:, None] * (self.goal[body_rows] - self.co[body_rows])

        cloth_weight = np.einsum("ij,ij->i", self.bind_bary, welded[self.bind_corners])
        cloth_rows = np.flatnonzero((cloth_weight > 0.0) & self.movable[n:])
        if len(cloth_rows):
            take = np.where(self.bind_movable[cloth_rows], strength * cloth_weight[cloth_rows], 0.0)
            self.progress[cloth_rows] += take * (1.0 - self.progress[cloth_rows])
            ride = self._ride(self.co)[cloth_rows]
            # The clothing's own fade, toward its own vertices that may not move.
            fade = self.fade[n + cloth_rows, None]
            self.co[n + cloth_rows] = self.start[n + cloth_rows] + fade * (
                ride + self.progress[cloth_rows, None] * self.residual[cloth_rows])
        rows = np.concatenate((body_rows, n + cloth_rows))
        self.moved[rows] = True
        return rows
