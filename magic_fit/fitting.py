# SPDX-License-Identifier: GPL-3.0-or-later

"""Fitting a mesh around a body without flattening it.

Shrinkwrap-like tools send every vertex to its own spot on the surface, which collapses the inner
and outer layers of thick clothing into one shell. Here the mesh is treated as layers instead: only
the innermost layer is fitted, and the layers over it move with it, so thickness, folds and other
details ride along unchanged.

With the mesh's triangles, the layers are found on the mesh itself: a vertex with another layer of
the mesh under it rides on that layer, and the innermost layer shares its corrections along its own
edges, only as far as its shape needs. A dip in what it needs up to about one and a half times Keep
Together wide (a wrinkle, a small fold) moves with what's around it, while a misfit that changes over a
wider stretch is fitted where it is, so shallow parts don't go as far as the deepest one beside them,
and a loose bulge is pulled in. Without the triangles, the layers are stacked on the body instead: for
each spot *of the body*, a correction is worked out from the innermost layer above it, and every layer
over that spot gets the same displacement.

Vertices move along the body's normal smoothed over Keep Together: over creases, where the body's
normal flips from one side to the other, the directions turn smoothly, so a mesh spanning a crease
isn't pulled apart, and it bridges creases narrower than that rather than being pulled into them.
The mesh's normals tell which parts of the body it lies on: clothing that wraps a thigh settles on
that thigh rather than on the one across the gap, even where it lies closer to that one, or has sunk
into it. And where the mesh lies along the body, facing the way the body does under it, it heads more
directly for its closest point, as far as that keeps it together, so it slides less across the body
on its way. Loose clothing spans the valleys of the body under it, up to about twice Keep Together
wide, as far as it doesn't dip into them itself: a skirt's front stays level across the gap between
the thighs instead of sagging into it.

Like `painting`, nothing here depends on the 3D view: `FitStroke` works on world space positions.
"""

import math

import numpy as np

from mathutils.bvhtree import BVHTree

from .painting import falloff_weights, read_geometry, surface_bvh

# Vertices closer than this are welded for vertex graphs (split normals, UV seams).
WELD_DISTANCE = 1e-5
# A connected part of the body is turned to face out of the volume it encloses itself when that is at
# least this much of the area around it (volume / area^1.5; a ball has 0.094, the devkit Mannequin's
# body and nails 0.027 to 0.051, a flat piece none), see `BodySurface._outward`...
ENCLOSED = 0.01
# ...and it is closed, but for holes whose rims add up to no more than this share of the square root of
# its area (a sphere 10 cm across may miss a triangle or two). Open wider, its inside is anybody's guess:
# a cup sunk into the body may line a hollow of the body as well as cap a bump.
CLOSED = 0.1
# A closest body point this close to an open edge of the body (in barycentric coordinates; the search
# tree works in single precision) lies on that edge.
OPEN_EDGE_TOLERANCE = 1e-4
# Points whose closest body point lies on an open edge are beyond the body there (next to its edge, not
# in front of or behind it) when the direction to them is farther than this (a cosine, about 11 degrees)
# from the body's normal.
BEYOND_COSINE = 0.98
# No dab moves a vertex farther than this fraction of the brush radius.
MAX_STEP = 0.1
# Limit for how many edges away from a body vertex layers are still grouped with it (without the mesh's
# triangles), and for how many steps the closing along the mesh takes (with them).
MAX_RINGS = 8
# With the mesh's triangles, the closing that shares what its parts need reaches this many times Keep
# Together each way along the mesh: dips up to about twice that wide are filled (wrinkles, small folds),
# while wider ones, loose bulges, are fitted.
CLOSE_REACH = 0.75
# With them, a part of the mesh lying over a valley of the body up to about twice this many times Keep
# Together wide (between the thighs, under a skirt) isn't pulled deeper into it than the mesh around it
# spans it (see `FitStroke._bridged`)...
BRIDGE_REACH = 1.0
# ...seen from the brush: spots moving farther than this from the way it looks (a cosine, 60 degrees) are
# left out...
BRIDGE_COSINE = 0.5
# ...where it lies farther from the body than Keep Together, or this (in metres) if that is farther, and partly
# from half that on: nearer, clothing settles into the valleys it lies over (a tight suit, into a crease)...
BRIDGE_LOOSE = 0.02
# ...and once the middle of the brush, this share of its radius, has passed over it, it stays held up for the
# rest of the stroke.
BRIDGE_CORE = 0.25
# Without the mesh's triangles, Auto Smooth drapes goals over up to this many times the Keep Together
# distance...
SMOOTH_REACH = 2.0
# ...in at most this many smoothing passes per dab (passes grow with the square of the reach)...
MAX_SMOOTH_PASSES = 150
# ...and smooths the mesh's height above the body, moving each vertex up to this fraction of the way
# to its neighbours' average per dab.
SMOOTH_RATE = 0.5
# With them, Auto Smooth takes bumps out of the shape the fit heads for in up to this many pairs of
# passes, at full Auto Smooth (Taubin's smoothing, which leaves larger shapes such as wrinkles and the
# body's curvature alone)...
SMOOTH_PAIRS = 6
# ...each a step this far toward the neighbours' average and one this far back, past where it started,
# so the shape doesn't shrink.
TAUBIN_SHRINK = 0.6
TAUBIN_INFLATE = -0.66
# A vertex moves no farther than 1 / this per unit of distance it gains from the body: its direction is
# turned toward the body's normal until it gains at least this much (a cosine, about 75 degrees)...
MIN_GAIN = 0.25
# ...and with the mesh's triangles, a vertex that is fitted rather than riding on another layer of the
# mesh no farther than twice what it gains (60 degrees): it doesn't slide far across the body for that.
FIT_GAIN = 0.5
# Points per block when averaging the body around them (`BodySurface.smoothed`).
SMOOTHED_CHUNK = 128
# Mesh triangles turning farther than this from a neighbour (a cosine, 90 degrees) start another sheet,
# which faces its own way (see `FitStroke._sheets`): a hem folded back inside, a copy for the back side,
# the inside of a solid...
SHEET_COSINE = 0.0
# ...and so do triangles bending around a tighter radius than this (in metres) between them: a hem folded
# back through a rounded fold. Clothing over a body bends far less sharply...
FOLD_RADIUS = 0.0025
# ...while triangles flatter than this (a cosine, about 6 degrees) never do: even a quarter of a millimetre
# wide, they'd pass.
FLAT_COSINE = 0.995
# A sheet faces away from the body on the side its normals face the body's normals under it when they
# agree on this share of the vote more than they disagree (cosine-weighted, 0..1)...
SHEET_CLEAR = 0.3
# ...voted at this many spots spread over it.
SIDE_SAMPLES = 64
# A sheet less clear than that (a closed solid: its inside faces the body, its outside away from it) is
# split into walls where it bends around a tighter radius than this (its rims), and each wall faces the
# way its own vote says, or no way in particular.
SOLID_FOLD_RADIUS = 0.01
# The way a mesh faces is smoothed over this many rings of its triangles, so slivers along a seam, a tiny
# ledge or a twisted triangle don't turn it...
FACING_RINGS = 2
# ...and where the sheets around a vertex face every which way (a sharp ridge, a loose tip), so that their
# sum is shorter than this share of their lengths, it faces no way in particular.
FACING_CLARITY = 0.25
# Body parts count toward a vertex's direction by how much they face the way the mesh faces there: not at
# all when they face back at it, from across a gap (the other thigh, seen from trousers between the legs),
# beyond this cosine (120 degrees)...
FACING_NONE = -0.5
# ...and fully from this one (90 degrees) on, like both walls of a crease the mesh spans or dips into.
FACING_FULL = 0.0
# A vertex whose closest body point faces away from its direction by more than this (a cosine, 90
# degrees) lies closer to some other part of the body (the other thigh, across the gap) than to the
# part under it, or has sunk into that other part: it's measured from where its own line meets the body
# instead...
FACING_COSINE = 0.0
# ...up to this far (in metres) past the closest body point, or Keep Together if that is farther.
GAP_REACH = 0.03
# Push Out gets such a vertex out of the part it has sunk into the shortest way, along that part's normal,
# but only where that leads to at least this share of the offset clear of the rest of the body. In a
# crease it doesn't: out of one side leads into the other, and the vertex would be kicked back and forth
# across the crease line, bunching the mesh up there.
EXIT_CLEARANCE = 0.5
# Where the mesh faces the way of the body's normal at a vertex's closest point within this cosine (about
# 37 degrees; a mesh spanning a right-angled crease faces its sides at 45), it lies along the body rather
# than spanning a crease: the vertex turns from the normal smoothed over Keep Together toward that one,
# toward its closest point (see `FitStroke._follow`)...
FOLLOW_FROM = 0.8
# ...fully from this cosine on (about 18 degrees)...
FOLLOW_FULL = 0.95
# ...but only as far as that moves where it settles by no more than this share of the length of its edges:
# however much the vertices around it turn, the mesh isn't pulled apart or bunched up.
FOLLOW_SHIFT = 0.1
# A dab that would turn a triangle of the mesh over moves its corners half as far, up to this many
# times, then leaves them where they are.
FOLD_TRIES = 4
# Looking for the layer of the mesh under a vertex, a ray passes at most this many triangles that don't
# count (the vertex's own, other garments facing the other way)...
LAYER_HITS = 8
# ...and at most this many of the body's, on its way to where it leaves the body.
BODY_HITS = 4
# A vertex rides on the layer under it when it lies no farther from it than this many times the middle of
# the distances at which its part lies on layers (its thickness). Farther, the gap between them isn't the
# clothing's thickness (an overlapping panel sunk deep into the body under it): the vertex is fitted on
# its own, just never closer to that layer than that.
LAYER_SPREAD = 2.0


def _barycentric(p, a, b, c):
    """Barycentric coordinates of points ``p`` in triangles ``a, b, c`` (all (n, 3) arrays)."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    ok = np.abs(denom) > 1e-30
    denom = np.where(ok, denom, 1.0)
    v = np.where(ok, (d11 * d20 - d01 * d21) / denom, 1.0 / 3.0)
    w = np.where(ok, (d00 * d21 - d01 * d20) / denom, 1.0 / 3.0)
    bary = np.clip(np.stack((1.0 - v - w, v, w), axis=1), 0.0, None)
    return bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)


def _spread_max(values, edges):
    """Each vertex takes the largest value among itself and its graph neighbours."""
    spread = values.copy()
    np.maximum.at(spread, edges[:, 0], values[edges[:, 1]])
    np.maximum.at(spread, edges[:, 1], values[edges[:, 0]])
    return spread


def _close(values, edges, costs, steps):
    """Morphological closing over ``steps`` units of distance along the graph's ``edges``, each edge
    costing ``costs`` units (whole numbers, at least 1): each vertex takes the largest value within
    that distance, then the smallest of those within it (-inf: no value, which doesn't count).

    The result is never below a vertex's own value. Dips narrower than twice the distance are filled
    to the level of what surrounds them, while a value that rises or falls over a wider stretch, or a
    peak of any width, stays where it is: unlike spreading the largest value alone, a deep spot doesn't
    raise the shallow ones beside it. As the distance is measured along the edges by their lengths, a
    dense part of the mesh reaches no farther than a coarse one.
    """
    usable = costs <= steps
    if not steps or not usable.any():
        return values
    count = len(values)
    # Every usable edge both ways, sorted by cost: the vertex it leads to, and where what it brings lies
    # in the rows of what each vertex took from within k units (one row of ``count`` per unit, see
    # below), less the row being filled.
    costs = np.concatenate((costs[usable], costs[usable]))
    order = np.argsort(costs, kind='stable')
    costs = costs[order]
    head = np.concatenate((edges[usable, 1], edges[usable, 0]))[order]
    tail = np.concatenate((edges[usable, 0], edges[usable, 1]))[order] - costs * count
    # Edges that fit within each number of units.
    fits = np.searchsorted(costs, np.arange(steps + 1), side='right')

    def spread(start, take):
        # What each vertex takes from within k units, for k = 0..steps: from within k - 1 units, and
        # from what its neighbours took from within k less the edge's cost.
        reached = np.empty((steps + 1) * count)
        reached[:count] = start
        for k in range(1, steps + 1):
            row = reached[k * count:(k + 1) * count]
            row[:] = reached[(k - 1) * count:k * count]
            n = fits[k]
            if n:
                take.at(row, head[:n], reached[tail[:n] + k * count])
        return reached[steps * count:]

    spread_out = spread(values, np.maximum)
    known = np.isfinite(spread_out)
    closed = spread(np.where(known, spread_out, np.inf), np.minimum)
    return np.where(known, closed, -np.inf)


def _out_of(distance, rise):
    """How much of a step that raises points ``distance`` from a surface (negative inside) by ``rise`` it
    takes to get them out (0..1; 0 for points outside, or steps that don't raise them)."""
    return np.where((distance < 0.0) & (rise > 0.0), np.minimum(-distance / np.maximum(rise, 1e-30), 1.0), 0.0)


def _connected(count, edges):
    """Connected parts of a graph of ``count`` vertices: (vertex -> part index, part count)."""
    parent = np.arange(count)
    if len(edges):
        a, b = edges[:, 0], edges[:, 1]
        # Every edge whose ends lie in different trees hangs the higher root under the lower one, then
        # every vertex points straight at its root; a few rounds join everything.
        while True:
            roots_a, roots_b = parent[a], parent[b]
            apart = roots_a != roots_b
            if not apart.any():
                break
            a, b, roots_a, roots_b = a[apart], b[apart], roots_a[apart], roots_b[apart]
            np.minimum.at(parent, np.maximum(roots_a, roots_b), np.minimum(roots_a, roots_b))
            while True:
                above = parent[parent]
                if np.array_equal(above, parent):
                    break
                parent = above
    _, labels = np.unique(parent, return_inverse=True)
    labels = labels.reshape(-1)
    return labels, (int(labels.max()) + 1 if len(labels) else 0)


def _both_ways(edges):
    """Every edge both ways: (tail, head)."""
    return np.concatenate((edges[:, 0], edges[:, 1])), np.concatenate((edges[:, 1], edges[:, 0]))


def _neighbour_mean(values, tail, head, degree):
    """The average of each vertex's graph neighbours' ``values`` (n, 3), or its own without neighbours."""
    count = len(values)
    total = np.stack([np.bincount(tail, values[head, k], minlength=count) for k in range(values.shape[1])], axis=1)
    return np.where(degree[:, None] > 0, total / np.maximum(degree, 1)[:, None], values)


def _across(p, a, b, da, db):
    """Distance to points ``p`` from a source that is ``da`` from ``a`` and ``db`` from ``b``, in the plane
    of each triangle (p, a, b) and on the far side of a-b, or inf where the straight line from there to
    ``p`` doesn't cross a-b (Blender's geodesic_distance_propagate_across_triangle)."""
    ab = b - a
    length = np.linalg.norm(ab, axis=1)
    u = ab / np.maximum(length, 1e-30)[:, None]
    ap = p - a
    x = np.einsum("ij,ij->i", ap, u)
    y = np.linalg.norm(ap - x[:, None] * u, axis=1)
    share = 0.5 * (1.0 + (da * da - db * db) / np.maximum(length * length, 1e-30))
    hh = da * da - share * share * length * length
    h = np.sqrt(np.maximum(hh, 0.0))
    sx = share * length
    cross = sx + h * (x - sx) / np.maximum(y + h, 1e-30)
    ok = (length > 1e-30) & (hh > 0.0) & (cross >= 0.0) & (cross <= length)
    return np.where(ok, np.hypot(x - sx, y + h), np.inf)


def _rows(keys, count):
    """``keys`` (0..count - 1) grouped: the order that sorts them and where each key's run starts."""
    order = np.argsort(keys, kind='stable')
    start = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=count), out=start[1:])
    return order, start


def _rows_of(order, start, keys):
    """The positions (in the grouped array, see `_rows`) of every entry under ``keys``."""
    first = start[keys]
    lengths = start[keys + 1] - first
    total = int(lengths.sum())
    if not total:
        return np.empty(0, dtype=np.int64)
    return order[np.repeat(first - np.cumsum(lengths) + lengths, lengths) + np.arange(total)]


def _mesh_distance(co, edges, sources, limit, tris=None):
    """Shortest distance along ``edges`` (index pairs into ``co``) from the nearest of the ``sources``
    (a mask), for vertices up to ``limit`` away (inf for the others), and which source that is (-1).
    With ``tris``, distances also cross the triangles in straight lines, like Blender's Connected Only:
    along edges alone, they'd come out up to about 15 % too long between them."""
    count = len(co)
    distance = np.where(sources, 0.0, np.inf)
    nearest = np.where(sources, np.arange(count), -1)
    if not len(edges):
        return distance, nearest
    # Every edge both ways, found by the vertex it leaves.
    tail = np.concatenate((edges[:, 0], edges[:, 1]))
    head = np.concatenate((edges[:, 1], edges[:, 0]))
    length = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
    length = np.concatenate((length, length))
    edge_rows = _rows(tail, count)
    if tris is not None:
        tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])]
        # The triangles around each vertex.
        tri_rows = _rows(tris.reshape(-1), count)
    # Vertices right at the limit count (less rounding).
    limit *= 1.0 + 1e-9
    closer = np.flatnonzero(sources)
    # Out from the vertices that just came closer, until none does. Only they and their neighbours are
    # looked at, so this takes about as long as the area it reaches.
    while len(closer):
        out = _rows_of(*edge_rows, closer)
        reach = distance[tail[out]] + length[out]
        keep = reach <= limit
        out, reach = out[keep], reach[keep]
        to = head[out]
        best = np.full(count, np.inf)
        np.minimum.at(best, to, reach)
        via = np.full(count, -1)
        won = reach == best[to]
        via[to[won]] = nearest[tail[out[won]]]
        if tris is not None and len(tris):
            # Across the triangles around them: to each corner from the other two.
            near = tris[np.unique(_rows_of(*tri_rows, closer) // 3)]
            for k in range(3):
                p, a, b = near[:, k], near[:, (k + 1) % 3], near[:, (k + 2) % 3]
                da, db = distance[a], distance[b]
                known = np.isfinite(da) & np.isfinite(db) & (da > 0.0) & (db > 0.0)
                p, a, b = p[known], a[known], b[known]
                reach = _across(co[p], co[a], co[b], da[known], db[known])
                keep = (reach <= limit) & (reach < best[p])
                np.minimum.at(best, p[keep], reach[keep])
                won = keep & (reach == best[p])
                via[p[won]] = nearest[np.where(da[known] <= db[known], a, b)[won]]
        # Tiny gains don't count, so rounding can't keep it going.
        closer = np.flatnonzero(best < distance - 1e-9 * limit)
        distance[closer] = best[closer]
        nearest[closer] = via[closer]
    return distance, nearest


def _smooth_above(values, floor, known, edges, iterations):
    """Average each known value with its known graph neighbours, ``iterations`` times, never
    letting it drop below ``floor``. Like a cloth draped over the floor: smooth, but resting on it."""
    a, b = edges[:, 0], edges[:, 1]
    weight = known.astype(np.float64)
    count = weight.copy()
    np.add.at(count, a, weight[b])
    np.add.at(count, b, weight[a])
    count = np.maximum(count, 1.0)
    floor = np.where(known, floor, -np.inf)
    values = np.where(known, values, 0.0)
    for _ in range(iterations):
        total = values.copy()
        np.add.at(total, a, values[b])
        np.add.at(total, b, values[a])
        values = np.where(known, np.maximum(total / count, floor), 0.0)
    return values


# Points are sorted along this direction to find the ones close together; any direction works, this
# one is just unlikely to line up with a mesh's rows of vertices.
_SWEEP = np.array([0.8219, 0.4471, 0.3528]) / np.linalg.norm([0.8219, 0.4471, 0.3528])


def _unique_rows(co):
    """Distinct rows of ``co`` and the row -> distinct row index (faster than np.unique(axis=0))."""
    if not len(co):
        return co, np.zeros(0, dtype=np.int64)
    order = np.lexsort(co.T[::-1])
    ordered = co[order]
    new = np.ones(len(co), dtype=bool)
    new[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    inverse = np.empty(len(co), dtype=np.int64)
    inverse[order] = np.cumsum(new) - 1
    return ordered[new], inverse


def weld_groups(co, distance):
    """Group points closer than ``distance`` to each other (chains of close points form one group),
    like Merge by Distance. Returns (point -> group index, group count)."""
    unique, inverse = _unique_rows(co)
    count = len(unique)
    if distance <= 0.0 or count < 2:
        return inverse, count
    # Points closer than `distance` are also closer than that along the sweep direction, so after
    # sorting along it, each point only needs comparing with the next few.
    order = np.argsort(unique @ _SWEEP, kind='stable')
    points = unique[order]
    key = points @ _SWEEP
    first, second = [], []
    shift = 1
    while shift < count:
        close = np.flatnonzero(key[shift:] - key[:-shift] <= distance)
        if not len(close):
            break
        delta = points[close + shift] - points[close]
        close = close[np.einsum("ij,ij->i", delta, delta) <= distance * distance]
        first.append(order[close])
        second.append(order[close + shift])
        shift += 1
    if not first or not sum(len(pair) for pair in first):
        return inverse, count
    labels, groups = _connected(count, np.stack((np.concatenate(first), np.concatenate(second)), axis=1))
    return labels[inverse], groups


def _weld(co, faces, distance=WELD_DISTANCE):
    """Weld vertices closer than ``distance``.

    ``faces`` holds triangles or edges. Returns (vertex -> welded index, welded count, welded edges).
    """
    weld, count = weld_groups(co, distance)
    return weld, count, _welded_edges(weld, count, faces)


def _welded_edges(weld, count, faces):
    """The distinct edges of ``faces`` (triangles or edges) between the welded vertices, see `_weld`."""
    faces = np.asarray(faces, dtype=np.int64)
    if not faces.size:
        return np.empty((0, 2), dtype=np.int64)
    corners = weld[faces]
    if corners.shape[1] == 3:
        corners = np.concatenate((corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]))
    edges = np.sort(corners[corners[:, 0] != corners[:, 1]], axis=1)
    keys = np.unique(edges[:, 0] * count + edges[:, 1])
    return np.stack((keys // count, keys % count), axis=1)


def _same_groups(a, b, count):
    """Whether the group indices ``a`` and ``b`` (point -> group, ``count`` groups each) group the
    points alike, whatever order the groups are in."""
    if len(a) != len(b):
        return False
    group = np.zeros(count, dtype=np.int64)
    group[a] = b
    return np.array_equal(group[a], b)


class Nearest:
    """Closest body points for a set of query points, see `BodySurface.nearest`."""

    __slots__ = ("found", "location", "normal", "distance", "corners", "bary", "beyond")

    def take(self, mask):
        result = Nearest()
        for name in self.__slots__:
            setattr(result, name, getattr(self, name)[mask])
        return result

    def put(self, mask, other):
        """Replace the points under ``mask`` (bool) with those of ``other`` (as many as it selects)."""
        for name in self.__slots__:
            getattr(self, name)[mask] = getattr(other, name)


class BodySurface:
    """The body a mesh is fitted around: nearest points, smooth outward normals and a vertex graph.

    Vertices sharing a position (split normals, UV seams) are welded, so normals and values flow
    across seams. Normals point away from the enclosed volume even if the mesh, or some of its parts,
    are inside out (see `_outward`). A body can be open, like a torso on its own: past its open edges
    (used by one triangle) it has no inside.
    """

    _cache = {}

    def __init__(self, co, tris, bvh, like=None):
        """``like`` is an earlier surface with the same ``tris``: what they decide is taken from it
        when the vertices weld together the same way."""
        self._bvh = bvh
        self.tris = tris
        self._tri_list = None  # see `from_geometry`

        weld, count = weld_groups(co, WELD_DISTANCE)
        if like is not None and len(like.co) == count and _same_groups(like._weld, weld, count):
            # The same body posed or reshaped: only the positions changed.
            weld = like._weld
            self.edges, self.corners = like.edges, like.corners
            self.open_keys, self.open_vertex = like.open_keys, like.open_vertex
            self.parts = like.parts
        else:
            self.edges = _welded_edges(weld, count, tris)
            self.corners = weld[tris]
            # Open edges (used by one triangle, as keys low * count + high, sorted) and their vertices.
            sides = np.sort(np.concatenate([self.corners[:, [0, 1]], self.corners[:, [1, 2]], self.corners[:, [2, 0]]]), axis=1)
            sides = sides[sides[:, 0] != sides[:, 1]]
            keys, uses = np.unique(sides[:, 0] * count + sides[:, 1], return_counts=True)
            self.open_keys = keys[uses == 1]
            self.open_vertex = np.zeros(count, dtype=bool)
            self.open_vertex[self.open_keys // count] = True
            self.open_vertex[self.open_keys % count] = True
            # The connected parts of the body (vertex -> part), see `_outward`.
            self.parts = _connected(count, self.edges)[0]
        self._weld = weld
        members = np.bincount(weld, minlength=count).astype(np.float64)
        self.co = np.zeros((count, 3))
        np.add.at(self.co, weld, co)
        self.co /= members[:, None]

        a, b, c = (self.co[self.corners[:, k]] for k in range(3))
        face_normals = np.cross(b - a, c - a)
        face_normals *= self._outward(a, b, c, face_normals)[:, None]
        # Unit normals of the triangles, facing out.
        self.tri_normals = face_normals / np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-30)
        corners = self.corners.reshape(-1)
        normals = np.stack([
            np.bincount(corners, np.repeat(face_normals[:, k], 3), minlength=count) for k in range(3)
        ], axis=1)
        # Normals scaled by the area around each vertex, for averages that don't depend on the mesh density.
        self.area_normals = normals
        self.normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

        lengths = np.linalg.norm(self.co[self.edges[:, 0]] - self.co[self.edges[:, 1]], axis=1)
        self.edge_length = float(np.median(lengths)) if len(lengths) else 0.0

    def _outward(self, a, b, c, face_normals):
        """Per triangle (corners ``a, b, c``, ``face_normals`` their cross products), 1 or -1: whether its
        normal faces out of the volume the body encloses, or has to be turned around.

        Each connected part of the body that is closed (see ``CLOSED``) and encloses a volume of its own
        (see ``ENCLOSED``) is turned by it, so a body joined from pieces of which some are inside out (a
        leg mirrored by hand) faces out everywhere. The other parts (flat ones, or ones open too wide to
        tell) go by the body as a whole: the volume of all of it, around its middle.
        """
        part = self.parts[self.corners[:, 0]]
        count = int(self.parts.max()) + 1
        members = np.bincount(self.parts, minlength=count).astype(np.float64)
        middle = np.stack([np.bincount(self.parts, self.co[:, k], count) for k in range(3)], axis=1)
        middle = (middle / members[:, None])[part]
        # Six times the volume of each part, by the cones from its middle to its triangles.
        own = np.bincount(part, np.einsum("ij,ij->i", a - middle, np.cross(b - middle, c - middle)), count)
        area = 0.5 * np.bincount(part, np.linalg.norm(face_normals, axis=1), count)
        # How long the open edges of each part are, all told.
        low, high = self.open_keys // len(self.co), self.open_keys % len(self.co)
        rims = np.bincount(self.parts[low], np.linalg.norm(self.co[low] - self.co[high], axis=1), count)
        enclosed = (rims <= CLOSED * np.sqrt(area)) & (np.abs(own) > 6.0 * ENCLOSED * area ** 1.5)
        center = self.co.mean(axis=0)
        whole = np.einsum("ij,ij->i", a - center, np.cross(b - center, c - center))[~enclosed[part]].sum()
        whole += np.abs(own[enclosed]).sum()
        return np.where(enclosed, np.sign(own), -1.0 if whole < 0.0 else 1.0)[part]

    @classmethod
    def from_object(cls, body, depsgraph):
        """The body as displayed, or None when it has no faces. Reused while its shape is unchanged."""
        return cls.from_geometry(body.as_pointer(), read_geometry(body, depsgraph, source='DISPLAYED'))

    @classmethod
    def from_geometry(cls, key, geometry):
        """The body for ``geometry`` (see `read_geometry`), or None when it has no faces. Reused while
        the geometry under ``key`` is unchanged, and so is what its triangles decide while only its
        positions change (a posed body)."""
        if not len(geometry.tris):
            return None
        co, tris = geometry.co, geometry.tris
        entry = cls._cache.get(key)
        like = None
        if entry is not None and np.array_equal(entry[1], tris):
            if np.array_equal(entry[0], co):
                return entry[2]
            like = entry[2]
        # A body that changes shape is rebuilt again and again: keep its triangles ready for the search
        # tree (as tuples, which the garbage collector leaves alone).
        tri_list = None if like is None else (like._tri_list or list(map(tuple, tris.tolist())))
        surface = cls(co, tris, surface_bvh(key, co, tris, tri_list), like)
        surface._tri_list = tri_list
        cls._cache.pop(key, None)
        while len(cls._cache) >= 2:
            del cls._cache[next(iter(cls._cache))]
        cls._cache[key] = (geometry.co, geometry.tris, surface)
        return surface

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()

    def nearest(self, points):
        """Closest body point, smooth normal and signed distance (negative inside) for (n, 3) points.

        ``beyond`` marks points past an open edge of the body (below a torso's waist, say): their closest
        body point is on the edge and they lie off to its side. The body doesn't reach them, so they are
        never inside it; their distance is the (positive) distance to the edge.
        """
        count = len(points)
        location = np.zeros((count, 3))
        tri = np.full(count, -1, dtype=np.int64)
        find = self._bvh.find_nearest
        for k, point in enumerate(points.tolist()):
            hit, _normal, index, _distance = find(point)
            if hit is not None:
                location[k] = hit
                tri[k] = index

        result = Nearest()
        result.found = tri >= 0
        result.location = location
        result.corners = self.corners[np.maximum(tri, 0)]
        a, b, c = (self.co[result.corners[:, k]] for k in range(3))
        result.bary = _barycentric(location, a, b, c)
        normal = np.einsum("ij,ijk->ik", result.bary, self.normals[result.corners])
        result.normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        offset = points - location
        distance = np.linalg.norm(offset, axis=1)
        result.beyond = self._beyond(result, offset, distance)
        inside = (np.einsum("ij,ij->i", offset, result.normal) < 0.0) & ~result.beyond
        result.distance = np.where(inside, -distance, distance)
        return result

    def leave(self, points, directions, limits):
        """How far each of (n, 3) ``points`` goes along its (unit) direction until it leaves the body,
        through a part of it facing that way (in and out again, for points outside it), or its limit
        (``limits``) where it doesn't."""
        result = np.array(limits, dtype=np.float64)
        cast = self._bvh.ray_cast
        rays = zip(range(len(result)), points.tolist(), directions.tolist(), result.tolist())
        for k, point, direction, limit in rays:
            origin, left = point, limit
            for _ in range(BODY_HITS):
                hit, _normal, index, distance = cast(origin, direction, left)
                if hit is None:
                    break
                if self.tri_normals[index] @ direction > 0.0:
                    result[k] = limit - left + distance
                    break
                left -= distance + 1e-6
                if left <= 0.0:
                    break
                origin = (hit[0] + 1e-6 * direction[0], hit[1] + 1e-6 * direction[1], hit[2] + 1e-6 * direction[2])
        return result

    def along(self, points, directions, limits, through=False):
        """The body under (n, 3) ``points`` along their (unit) ``directions`` away from it, as `nearest`
        gives it for the closest point: where the line back from each point enters the body (or, for a
        point inside it, where the line leaves it), within ``limits``. ``distance`` is the signed
        distance from the body's tangent plane there and ``found`` marks the points whose line meets
        the body from the right side.

        With ``through``, the line back from a point inside the body goes on past where it leaves it, to
        where it enters the body again: the part the point faces, beyond the one it has sunk into (one
        leg's clothing pushed into the other thigh). Such points aren't looked for forward.
        """
        count = len(points)
        location = np.zeros((count, 3))
        tri = np.full(count, -1, dtype=np.int64)
        sunk = np.zeros(count, dtype=bool)
        cast = self._bvh.ray_cast
        for sign in (-1.0, 1.0):
            # Back from outside first; then forward, for the points the line back from left the body.
            rows = np.flatnonzero((tri < 0) & ~sunk) if sign > 0.0 else np.arange(count)
            for k, point, direction, limit in zip(
                    rows.tolist(), points[rows].tolist(), (sign * directions[rows]).tolist(), limits[rows].tolist()):
                hit, _normal, index, distance = cast(point, direction, limit)
                if through and sign < 0.0 and hit is not None and self.tri_normals[index] @ direction > 0.0:
                    # Leaving the body on the way back: on to where the line enters it again.
                    sunk[k] = True
                    left = limit - distance - 1e-6
                    origin = (hit[0] + 1e-6 * direction[0], hit[1] + 1e-6 * direction[1], hit[2] + 1e-6 * direction[2])
                    hit, _normal, index, _distance = cast(origin, direction, left) if left > 0.0 else (None,) * 4
                if hit is not None:
                    location[k] = hit
                    tri[k] = index
            if sign < 0.0:
                # Hitting the body's inside on the way back means the point is inside it.
                corners = self.corners[np.maximum(tri, 0)]
                bary = _barycentric(location, *(self.co[corners[:, k]] for k in range(3)))
                normal = np.einsum("ij,ijk->ik", bary, self.normals[corners])
                tri[np.einsum("ij,ij->i", normal, directions) <= 0.0] = -1

        result = Nearest()
        result.location = location
        result.corners = self.corners[np.maximum(tri, 0)]
        a, b, c = (self.co[result.corners[:, k]] for k in range(3))
        result.bary = _barycentric(location, a, b, c)
        normal = np.einsum("ij,ijk->ik", result.bary, self.normals[result.corners])
        result.normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        result.found = (tri >= 0) & (np.einsum("ij,ij->i", result.normal, directions) > 0.0)
        result.distance = np.einsum("ij,ij->i", points - location, result.normal)
        result.beyond = np.zeros(count, dtype=bool)
        return result

    def smoothed(self, points, near, spread, facing=None):
        """The body under (n, 3) ``points`` (``near`` from `nearest`, or `along`) smoothed over ``spread``:
        directions away from it that turn smoothly over creases, the points they start from, and which
        points found any body to average (the others keep ``near``'s normal and point).

        Both are averages over the body around each point, by area, each part weighted down the
        farther it is than the body point ``near`` gives (smoothly, to nothing ``spread`` farther). The
        closest body point jumps from one side of a crease to the other halfway across it, and so does
        its normal; these directions turn gradually instead, pointing straight out of the crease
        halfway, and the points lie between both sides. On smooth parts of the body the directions are
        its normal, and a low-poly body's facets don't show in them.

        With ``facing`` (a unit direction per point, the way the mesh faces there, or zero for points
        without one), each part of the body also counts by how much its normal faces that way (see
        ``FACING_NONE``), and not at all when it faces back: a mesh that wraps one thigh isn't turned by
        the other thigh's side across the gap, while one that spans a crease faces out of it like both
        of its sides, which still count.
        """
        count = len(points)
        directions = near.normal.copy()
        bases = near.location.copy()
        if not count or spread <= 0.0:
            return directions, bases, np.ones(count, dtype=bool)
        gap = np.abs(near.distance)
        center = points.mean(axis=0)
        points = points - center
        co = self.co - center
        span = math.sqrt(np.einsum("ij,ij->i", points, points).max()) + gap.max() + spread
        nearby = np.flatnonzero(np.einsum("ij,ij->i", co, co) < span * span)
        if not len(nearby):
            return directions, bases, np.zeros(count, dtype=bool)
        co = co[nearby]
        normals = self.area_normals[nearby]
        area = np.linalg.norm(normals, axis=1)
        if facing is not None:
            facing = facing.astype(np.float32)
            aware = np.einsum("ij,ij->i", facing, facing) > 0.0
            units = self.normals[nearby].astype(np.float32)
        # Per point, weighted: normals, positions times area, and area.
        sums = np.concatenate((normals, co * area[:, None], area[:, None]), axis=1).astype(np.float32)
        # A body vertex at distance d weighs (1 - t)², where t = (d² - gap²) / ((gap + spread)² - gap²)
        # runs from 0 at the closest body point to 1 at ``spread`` farther.
        scale = (1.0 / (spread * (2.0 * gap + spread))).astype(np.float32)
        gap_squared = np.square(gap).astype(np.float32)
        total = np.zeros((count, 7))
        # Blocks of points close together (sorted by cells ``spread`` wide), each against the body
        # around it only.
        order = np.lexsort(np.floor(points / spread).T[::-1])
        for first in range(0, count, SMOOTHED_CHUNK):
            rows = order[first:first + SMOOTHED_CHUNK]
            part = points[rows]
            middle = part.mean(axis=0)
            part = part - middle
            reach = math.sqrt(np.einsum("ij,ij->i", part, part).max()) + gap[rows].max() + spread
            around = co - middle
            squared = np.einsum("ij,ij->i", around, around)
            close = np.flatnonzero(squared < reach * reach)
            if not len(close):
                continue
            part = part.astype(np.float32)
            weight = part @ around[close].astype(np.float32).T
            weight *= -2.0
            weight += (np.einsum("ij,ij->i", part, part) - gap_squared[rows])[:, None]
            weight += squared[close].astype(np.float32)[None, :]
            weight *= scale[rows, None]
            np.subtract(1.0, weight, out=weight)
            np.clip(weight, 0.0, 1.0, out=weight)
            weight *= weight
            if facing is not None:
                agree = facing[rows] @ units[close].T
                agree -= FACING_NONE
                agree *= 1.0 / (FACING_FULL - FACING_NONE)
                np.clip(agree, 0.0, 1.0, out=agree)
                agree[~aware[rows]] = 1.0
                weight *= agree
            total[rows] = weight @ sums[close]
        length = np.linalg.norm(total[:, :3], axis=1)
        found = (length > 1e-30) & (total[:, 6] > 1e-30)
        directions[found] = total[found, :3] / length[found, None]
        bases[found] = total[found, 3:6] / total[found, 6:] + center
        return directions, bases, found

    def _beyond(self, near, offset, distance):
        """Which points lie past an open edge of the body: their closest body point (``near``) is on the
        edge (or at one of its ends), and ``offset`` from it leaves the body sideways, rather than along
        its normal."""
        count = len(self.co)
        beyond = np.zeros(len(offset), dtype=bool)
        if not len(self.open_keys) or not len(offset):
            return beyond
        corners, bary = near.corners, near.bary
        on_open = np.zeros(len(offset), dtype=bool)
        for k in range(3):
            # The edge across from corner k, which the point is on when its share is (about) 0.
            a, b = corners[:, (k + 1) % 3], corners[:, (k + 2) % 3]
            keys = np.minimum(a, b) * count + np.maximum(a, b)
            slot = np.minimum(np.searchsorted(self.open_keys, keys), len(self.open_keys) - 1)
            on_open |= (bary[:, k] < OPEN_EDGE_TOLERANCE) & (self.open_keys[slot] == keys)
        rows = np.arange(len(offset))
        at = np.argmax(bary, axis=1)
        on_open |= (bary[rows, at] > 1.0 - OPEN_EDGE_TOLERANCE) & self.open_vertex[corners[rows, at]]
        if not on_open.any():
            return beyond
        a, b, c = (self.co[corners[:, k]] for k in range(3))
        face = np.cross(b - a, c - a)
        face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-30)
        along = np.abs(np.einsum("ij,ij->i", offset, face)) / np.maximum(distance, 1e-30)
        return near.found & on_open & (distance > 0.0) & (along < BEYOND_COSINE)


class FitStroke:
    """Moves a mesh so the innermost of its layers sits ``offset`` away from the body, dab by dab.

    ``mode``: 'PUSH' only moves layers that are too close (or inside), 'TIGHTEN' only pulls in layers
    that are too far, 'FIT' does both.

    Every vertex of the innermost layer heads for a goal: how far it has to move away from the body,
    measured from its reference position (where the stroke found it). Goals don't change while
    vertices travel toward them, so dabs settle instead of piling up; each dab covers strength x
    falloff of the remaining way.

    With the mesh's triangles (``tris``), the layers are found on the mesh (see `_layers`): a vertex
    that has another layer of the mesh under it, within ``layer_radius`` toward the body, rides on
    that layer. It rises as much as the spot of the layer under it, so thickness is kept however the
    layers' vertices line up, and a layer is never pulled in farther than the one under it allows
    (`_stacked`). A vertex lying much farther from the layer under it than its part's layers usually
    do (see ``LAYER_SPREAD``) is fitted on its own instead, so a panel sunk deep into the body doesn't
    lift the clothing over it off the skin. The innermost layer shares goals along its own edges by a
    closing over ``CLOSE_REACH`` x ``layer_radius``, measured along the mesh (`_close_along`): a vertex
    that needs less than the vertices on both sides of it within that distance moves with them, so
    small folds and wrinkles move as one piece, but where what's needed changes over a wider stretch
    (clothing that clips in one place and floats a little farther on, as between the thighs; a loose
    bulge), every part moves as far as it needs, instead of the shallow parts going as far as the
    deepest one. Nothing is pulled in deeper than the offset from the body smoothed over
    ``layer_radius``, so the mesh bridges creases narrower than that, and loose clothing isn't pulled
    into a valley up to about twice that wide deeper than it spans it (see `_bridged`): a skirt's front
    stays level across the gap between the thighs.

    Without triangles, the layers are stacked on the body: a vertex counts for the corners of its
    closest body triangle, each body vertex's stack follows the layer that needs the most, and the
    largest goal spreads along the body's edges as far as ``layer_radius``, so layers whose vertices
    don't line up and small folds move as one piece. Where the misfit changes fast (between the
    thighs), that moves the shallow parts as far as the deepest one.

    Edges of the mesh longer than its closing reaches share nothing, and without triangles, nor does a
    ``layer_radius`` shorter than the body's edges: then each vertex fits on its own (counting only for
    the body vertex closest to it).

    Vertices move in straight lines, along the body's normal averaged over ``layer_radius`` and picked
    where the stroke first reaches them (see `_aim`): over a crease, vertices on either side of it
    don't slide apart. With triangles, the body counts by how much it faces the same way as the mesh
    (see `_facing`), so a trouser leg moves toward its own thigh, never toward the other one across the
    gap, even from closer to the other thigh than to its own; and a vertex whose closest body point
    faces away from it is measured from where its line meets the body, past the other thigh if it has
    sunk into it (see `_contact`). Where the mesh lies along the body, facing the way the body's normal
    at its closest point does, and isn't loose, its direction turns toward that normal, so it slides
    less across the body, but only as far as that moves where it settles by a tenth of its edges' length
    (see `_follow`): over a crease, neighbours still don't pull apart or bunch up. Only a vertex inside the
    body where the mesh faces into it (a seam allowance folded over inside) doesn't keep its line: it
    gets out the shortest way, along the normal at its closest point. A dab turns no triangle of the
    mesh over, unless that's what it takes to get clothing out of the body (see `_unfolded`).

    Auto Smooth (``smooth`` 0..1) keeps the result smooth, scaled by it:

    - With triangles, bumps are taken out of the shape the fit heads for (`_smoothing`), by Taubin's
      method, which leaves larger shapes, wrinkles and the body's curvature as they are. That shape
      doesn't change while vertices travel toward it, so going over a spot again doesn't wear it down
      further. It works either way in every mode, but never moves a vertex closer to the body than
      the offset (or than the fit puts it), and it keeps vertices split along seams together. The closing leaves no rim to
      ease out.
    - Without them, goals are draped: instead of the area moved as one piece dropping off steeply at
      its edge (a rim), goals ease out over up to ``SMOOTH_REACH`` x ``layer_radius``, like cloth
      draped over what needs to move, never below what a stack needs. And the mesh's height above the
      body is smoothed (needs the mesh's ``edges``, see `_auto_smooth`).

    Vertices closer together than ``weld_distance`` (split seams, separate parts or objects that
    touch) always move by exactly the same amount, so seams never open up. Such a group only moves
    when all of its vertices may.

    With ``fade_distance``, the brush fades out toward the vertices that may not move (hidden or
    unselected ones), along the mesh (needs ``edges``), so the part fitted stays joined to them instead
    of leaving a step or a kink at their edge. Next to them a vertex keeps the distance from the body
    of the closest one (its anchor); ``fade_distance`` away along the mesh it is fitted fully, and in
    between it sits at a smooth blend of both. That is a shape rather than a share of the way, so
    stroke after stroke leaves it as it is instead of wearing the fade down. Push Out never keeps a
    vertex lower than the offset for its anchor and Tighten never higher. Layers over the same spot
    follow their most faded one.
    """

    # The triangles of the mesh last stroked, and as a list for its search tree (see `_layers`): stroke
    # after stroke over the same mesh, building that list took half the time to build the tree.
    _tri_list = (None, None)

    @classmethod
    def clear_cache(cls):
        cls._tri_list = (None, None)

    def __init__(
            self, body, co, movable, *,
            mode='PUSH', offset=0.002, layer_radius=0.02, max_distance=None, edges=None, tris=None, smooth=0.0,
            weld_distance=WELD_DISTANCE, fade_distance=0.0,
    ):
        self.body = body
        # World space positions, updated as vertices move.
        self.co = co
        # Reference positions the fit displacement is measured from: where the stroke found each vertex.
        self.start = co.copy()
        self.mode = mode
        self.offset = offset
        self.layer_radius = layer_radius
        self.max_distance = max_distance
        self.moved = np.zeros(len(co), dtype=bool)
        # The direction each vertex moves in, picked where the stroke first reached it (see `_aim`).
        self.direction = np.zeros((len(co), 3))
        # ...and the point of the body smoothed over Keep Together it moves away from (see `_aim`).
        self.base = np.zeros((len(co), 3))
        self.aimed = np.zeros(len(co), dtype=bool)
        # Whether each vertex bridges creases: it faces some way, and found body facing that way (see `_aim`).
        self.bridging = np.zeros(len(co), dtype=bool)
        self.smooth = smooth
        edge = body.edge_length
        if edge > 0.0:
            # Keep Together shorter than the body's edges: every vertex fits on its own (see `dab`).
            self.rings = 0 if layer_radius < edge else min(MAX_RINGS, math.ceil(layer_radius / edge))
            # Repeated neighbour averaging spreads about sqrt(passes) edges.
            reach = smooth * SMOOTH_REACH * layer_radius / edge
            self.passes = self.rings + min(MAX_SMOOTH_PASSES, math.ceil(1.2 * reach * reach))
        else:
            self.rings = self.passes = 1
        self.margin = layer_radius + math.sqrt(self.passes) * edge

        if edges is None:
            edges = np.empty((0, 2), dtype=np.int64)
        tris = None if tris is None or not len(tris) else np.asarray(tris, dtype=np.int64)
        if not len(edges) and tris is not None:
            edges = tris
        self.weld, self.weld_count, self.weld_edges = _weld(co, edges, weld_distance)
        # Whether any vertices were welded together.
        self.linked = self.weld_count < len(co)
        if self.linked:
            together = np.ones(self.weld_count, dtype=bool)
            np.logical_and.at(together, self.weld, movable)
            movable = together[self.weld]
        self.movable = movable
        # Every welded edge both ways, found by the welded vertex it leaves (see `_local_edges`).
        self.tail, self.head = _both_ways(self.weld_edges)
        self.edge_rows = _rows(self.tail, self.weld_count)

        # With triangles: which way the mesh faces (see `_sheets`), its connected parts (as the mesh
        # itself joins them, not welded) and the layer of the mesh each vertex rides on (see `_layers`).
        self.tris = tris
        self.layered = tris is not None
        # Auto Smooth: with triangles, Taubin passes per dab (in pairs, the last one maybe in part); without,
        # the mesh's height above the body is smoothed (see `_auto_smooth`).
        self.pairs = SMOOTH_PAIRS * smooth if len(self.weld_edges) else 0.0
        self.smooth_mesh = not self.layered and smooth > 0.0 and len(self.weld_edges) > 0
        if self.layered:
            # The welded vertices where the stroke found them.
            self.points = np.empty((self.weld_count, 3))
            self.points[self.weld] = co
            self._sheets()
            links = edges if edges.shape[1] == 2 else _welded_edges(np.arange(len(co)), len(co), tris)
            self.part, self.part_count = _connected(len(co), links)
            # The triangles around each vertex (see `_unfolded`).
            self.tri_rows = _rows(tris.reshape(-1), len(co))
            # The members of each welded group, and how many there are (see `_unfolded`).
            self.weld_rows = _rows(self.weld, self.weld_count)
            self.weld_size = np.bincount(self.weld, minlength=self.weld_count).astype(np.float64)
            # The way the mesh faces at each vertex once aimed (see `_aim`), the connected parts of the
            # welded mesh, and how far each welded vertex may be pulled in at most over a valley, once the
            # middle of the brush has passed over it (see `_bridged`).
            self.facing = np.zeros((len(co), 3))
            self.weld_part = _connected(self.weld_count, self.weld_edges)[0]
            # How long each welded vertex's edges are on average (see `_follow`).
            length = np.linalg.norm(self.points[self.weld_edges[:, 0]] - self.points[self.weld_edges[:, 1]], axis=1)
            ends = self.weld_edges.reshape(-1)
            self.spacing = np.bincount(ends, np.repeat(length, 2), self.weld_count) / np.maximum(
                np.bincount(ends, minlength=self.weld_count), 1)
            self.bridge = np.full(self.weld_count, -np.inf)
            # The triangle each vertex rides on (-1: none) and where on it, once looked for.
            self.carrier = np.full(len(co), -1, dtype=np.int64)
            self.carrier_bary = np.zeros((len(co), 3))
            # Whether that triangle lies under the vertex (toward the body) rather than over it, and how far
            # the vertex was from it.
            self.carrier_below = np.zeros(len(co), dtype=bool)
            self.carrier_gap = np.zeros(len(co))
            # How many vertices of each part were found right above another part, as {part * parts + other:
            # count}: where parts cross each other (a strap's edge dipping under the top it lies on), a
            # part rides on another only if more of its vertices lie on that part than the other way round.
            self.on_part = {}
            # How far the vertices of each part found right above a layer were from it, and the middle of
            # those distances (NaN: none yet), the thickness its layers keep (see `_riding`).
            self.part_gaps = {}
            self.thickness = np.full(self.part_count, np.nan)
            self.mesh_bvh = None

        # Near vertices that may not move: how much each vertex is fitted (0..1) rather than kept at the
        # height of the closest of them (its anchor), see `dab`. None when nothing fades.
        self.fade = self.anchor_height = None
        if fade_distance > 0.0 and len(self.weld_edges) and not movable.all():
            # Per welded vertex, whose members all may move or none.
            fixed = np.ones(self.weld_count, dtype=bool)
            fixed[self.weld[movable]] = False
            points = np.empty((self.weld_count, 3))
            points[self.weld] = co
            # Edges between fixed vertices lead nowhere (a mesh mostly hidden has lots of them).
            edges = self.weld_edges[~fixed[self.weld_edges].all(axis=1)]
            distance, nearest = _mesh_distance(points, edges, fixed, fade_distance)
            band = np.flatnonzero(~fixed & (nearest >= 0))
            if len(band):
                t = np.minimum(distance / fade_distance, 1.0)
                self.fade = (t * t * (3.0 - 2.0 * t))[self.weld]
                # The anchors' distances from the body. Push Out never keeps a vertex lower than the
                # offset for them, Tighten never higher.
                anchors, which = np.unique(nearest[band], return_inverse=True)
                height = np.full(self.weld_count, float(offset))
                height[band] = body.nearest(points[anchors]).distance[which.reshape(-1)]
                if mode == 'PUSH':
                    height = np.minimum(height, offset)
                elif mode == 'TIGHTEN':
                    height = np.maximum(height, offset)
                self.anchor_height = height[self.weld]

    def dab(self, center, radius, strength, falloff='SMOOTH'):
        """Apply one dab around the world space point ``center``. Returns the indices of moved vertices."""
        nothing = np.empty(0, dtype=np.int64)
        if radius <= 0.0 or strength <= 0.0:
            return nothing
        body = self.body
        center = np.asarray(center, dtype=np.float64)
        hub = body.nearest(center[None, :])
        if not hub.found[0]:
            return nothing
        anchor = hub.location[0]

        # Vertices near the brush, with a margin so layers just outside it still count.
        reach = radius + self.layer_radius
        delta = self.co - center
        in_reach = np.einsum("ij,ij->i", delta, delta) < reach * reach
        if self.linked:
            # Whole welded groups, so they can move together.
            touched = np.zeros(self.weld_count, dtype=bool)
            touched[self.weld[in_reach]] = True
            in_reach = touched[self.weld]
        candidates = np.flatnonzero(self.movable & in_reach)
        if not len(candidates):
            return nothing
        near = body.nearest(self.co[candidates])
        # Parts past an open edge of the body (below a torso on its own, say) aren't over it: left alone.
        over = near.found & ~near.beyond
        candidates = candidates[over]
        near = near.take(over)
        if self.linked and not self.layered:
            # With triangles, welded vertices move alike already (see `_unfolded`).
            moving_set = candidates
            before = self.co[candidates].copy()
            # Auto Smooth moves the reference positions too (see `_auto_smooth`).
            before_start = self.start[candidates].copy()

        # The falloff is measured along the body, so all layers over a spot get the same strength.
        under_brush = np.linalg.norm(near.location - anchor, axis=1) < radius
        if not under_brush.any():
            return nothing
        direction = self._aim(candidates, near, under_brush | self.moved[candidates])
        riders = None
        alone = np.zeros(len(candidates), dtype=bool)
        if self.layered:
            # Measured from the part of the body each vertex lies over (see `_contact`). Vertices lying on
            # another layer of the mesh ride on it (see `_ride`); the rest is fitted, but for those too far
            # from the part of the body they face to tell where it is.
            near, direction, alone, lost = self._contact(candidates, near, direction)
            riding = self._riding(candidates)
            riders, carried, heading = candidates[riding], near.take(riding), direction[riding]
            fitted = ~riding & ~lost
            candidates, near, direction, alone = candidates[fitted], near.take(fitted), direction[fitted], alone[fitted]
            # Those inside the part of the body they're measured from where the mesh faces into it (a seam
            # allowance folded over inside the body) get out of it the shortest way, along its normal. Their
            # directions, which follow how the mesh faces, would lead them along the body, or deeper.
            folded = (near.distance < 0.0) & (np.einsum("ij,ij->i", self.facing[candidates], near.normal) < 0.0)
            direction = np.where(folded[:, None], near.normal, direction)
            if not len(candidates):
                # Riders only, whose layers lie out of the brush's reach: nothing moves them.
                return nothing
        t = np.linalg.norm(near.location - anchor, axis=1) / radius
        inside = t < 1.0
        weight = np.zeros(len(t))
        weight[inside] = falloff_weights(falloff, t[inside])
        alpha = weight * strength
        # With the mesh's triangles, fitted vertices travel no farther than twice what they gain (see FIT_GAIN).
        direction, gain = self._turn(direction, near.normal, FIT_GAIN if self.layered else MIN_GAIN)

        # How far each vertex has come from its reference position along its direction, and how far it
        # has to rise from there to sit at the offset: measured from its current distance minus what
        # it gained, which also corrects for the distance gained per step differing from the step
        # (curved or tilted body surfaces).
        came = self.co[candidates] - self.start[candidates]
        travel = np.einsum("ij,ij->i", came, direction)
        start_distance = near.distance - np.einsum("ij,ij->i", came, near.normal)
        need = self.offset - start_distance
        if self.layered:
            # Pulled in no deeper than the offset from the body smoothed over Keep Together (its point
            # under each vertex along its direction, see `_aim`): over a crease narrower than that,
            # clothing bridges it instead of being pulled into it, while on the rest of the body that
            # smoothed body lies under the body itself and changes nothing.
            bridging = self.bridging[candidates] & ~alone
            above = self.start[candidates] - self.base[candidates]
            height = np.einsum("ij,ij->i", above, self.direction[candidates])
            need = np.where(bridging, np.maximum(need, np.minimum(self.offset - height, 0.0)), need)
            need = np.where(alone, need, self._stacked(candidates, need, riders, carried))

        # Where goals are worked out and shared: ``count`` spots along ``edges``, each vertex counting for
        # its ``spots`` and heading for what its ``corners`` share, by ``share`` (see the class). With
        # triangles, the spots are the welded vertices of the fitted layer, each counting for itself, but
        # for the vertices on their own; without, the body's vertices.
        if self.layered:
            unique, inverse = np.unique(self.weld[candidates], return_inverse=True)
            corners = inverse.reshape(-1, 1)
            share = np.ones((len(candidates), 1))
            edges = self._local_edges(unique)
            # No draping: the closing doesn't leave a rim to ease out (see `_smoothing` instead).
            count, passes = len(unique), 0
            stacked = np.where(alone, -np.inf, need)
            rows, spots = np.arange(len(candidates)), corners.reshape(-1)
        else:
            # Body vertices around the brush and the edges between them.
            margin = reach + self.margin + 2.0 * body.edge_length
            delta = body.co - anchor
            region = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) < margin * margin)
            local = np.full(len(body.co), -1, dtype=np.int64)
            local[region] = np.arange(len(region))
            edges = local[body.edges]
            edges = edges[(edges >= 0).all(axis=1)]
            corners, share = local[near.corners], near.bary
            count, passes = len(region), self.passes
            stacked = need
            if self.rings:
                rows, spots = np.repeat(np.arange(len(candidates)), 3), corners.reshape(-1)
            else:
                # Then a vertex counts only for the closest corner: it doesn't take on what the vertices a
                # triangle away need, which on a mesh crossing the body at a slant is a lot more (or less).
                rows = np.arange(len(candidates))
                spots = corners[rows, np.argmax(near.bary, axis=1)]
        # Per spot: the stack above it follows its innermost layer, which needs the most.
        ok = spots >= 0
        stack = np.full(count, -np.inf)
        inner = np.full(count, np.inf)
        np.maximum.at(stack, spots[ok], stacked[rows[ok]])
        np.minimum.at(inner, spots[ok], start_distance[rows[ok]])
        floor = None
        if self.layered and self.mode != 'PUSH':
            # How far loose clothing may be pulled into a valley at most: no deeper than it spans it (see
            # `_bridged`).
            floor, looseness = self._bridged(
                stack, unique, edges, candidates, spots, direction, weight, inner, center, radius)
        loose = None
        if self.max_distance is not None and self.mode != 'PUSH':
            # Loose parts (skirts, capes...) are not pulled in.
            loose = np.isfinite(inner) & (inner > self.max_distance)
            stack[loose & (stack < 0.0)] = 0.0
        # What the mode lets each stack do: Push Out only lifts, Tighten only lowers.
        has_stack = np.isfinite(stack)
        if self.mode == 'PUSH':
            stack[has_stack] = np.maximum(stack[has_stack], 0.0)
        elif self.mode == 'TIGHTEN':
            stack[has_stack] = np.minimum(stack[has_stack], 0.0)

        # Near vertices that may not move, stacks follow their most faded layer: how much it is fitted,
        # and how far it has to move to sit at its anchor's height instead (see `__init__`).
        faded = None
        if self.fade is not None:
            fade = self.fade[candidates]
            layer, spot = rows[ok], spots[ok]
            order = np.lexsort((fade[layer], spot))
            layer, spot = layer[order], spot[order]
            first = np.ones(len(spot), dtype=bool)
            first[1:] = spot[1:] != spot[:-1]
            layer, spot = layer[first], spot[first]
            spot_fade = np.ones(count)
            spot_fade[spot] = fade[layer]
            to_anchor = np.zeros(count)
            to_anchor[spot] = self.anchor_height[candidates[layer]] - start_distance[layer]
            faded = spot_fade < 1.0

        # Stacks near each other share their targets, never below a stack's own target: that way
        # nothing ends up inside the body or pulled past the offset. Faded stacks share none: what they
        # need but are kept from doing would push their neighbours too far, or keep them from
        # tightening.
        source = stack if faded is None else np.where(faded, -np.inf, stack)
        if self.layered:
            # Along the mesh, by a closing (see `_close_along`). Spots held up over a valley (but faded ones)
            # share that with the other loose spots only: clothing lying on the body beside them still settles
            # onto it.
            hold = None if floor is None else np.where(np.isfinite(source), floor, -np.inf)
            if hold is None or not np.isfinite(hold).any():
                target = self._close_along(source, unique, edges)
            elif (looseness > 0.0).all():
                target = self._close_along(np.maximum(source, hold), unique, edges)
            else:
                target, raised = self._close_along(np.stack((source, np.maximum(source, hold))), unique, edges)
                target = np.where(looseness > 0.0, raised, target)
            if len(riders):
                # Riders count too, at what the spot they ride on heads for (see `_held`): a fitted part of
                # a layer that needs more than the part of it lying on another layer all around (a strap's
                # edge just past the top it lies on) moves with that part instead of tearing away from it.
                held, value = self._held(riders, unique, target)
                if len(held):
                    nodes = np.union1d(unique, held)
                    fitted = np.searchsorted(nodes, unique)
                    both = np.full(len(nodes), -np.inf)
                    both[fitted], both[np.searchsorted(nodes, held)] = source, value
                    target = np.maximum(target, self._close_along(both, nodes, self._local_edges(nodes))[fitted])
        else:
            # Along the body, the largest target spreads as far as Keep Together, so layers whose
            # vertices don't line up and small folds move as one piece; then it's draped with Auto
            # Smooth.
            target = source
            for _ in range(self.rings):
                target = _spread_max(target, edges)
        known = np.isfinite(target)
        target = _smooth_above(target, source, known, edges, passes)
        if faded is not None:
            # Faded stacks take on only part of what their neighbours share, and head for between that
            # and their anchor's height. Going over them again doesn't move them farther.
            own = stack[faded] if floor is None else np.maximum(stack[faded], floor[faded])
            # Spots with nothing but vertices on their own (see `_contact`) have no stack: they head for the
            # offset alone and take nothing from here, but the blend stays a number.
            own = np.where(np.isfinite(own), own, 0.0)
            part = spot_fade[faded]
            shared = np.where(known[faded], target[faded], own)
            blend = part * (own + part * np.maximum(shared - own, 0.0)) + (1.0 - part) * to_anchor[faded]
            if loose is not None:
                blend[loose[faded] & (blend < 0.0)] = 0.0
            if self.mode == 'PUSH':
                blend = np.maximum(blend, 0.0)
            elif self.mode == 'TIGHTEN':
                blend = np.minimum(blend, 0.0)
            target[faded] = blend
            known |= faded

        # Each vertex heads for the target of the spots it counts for, along its direction, so the
        # layers over a spot move alike. Over a crease, where its direction leans away from the body's
        # normal and gains distance more slowly, a vertex that is too close itself goes as far as it
        # needs to. Corners without a stack (only the closest corners have one when every vertex fits on
        # its own, without triangles) don't count.
        weights = np.where((corners >= 0) & known[np.maximum(corners, 0)], share, 0.0)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        goal = (weights * np.where(known, target, 0.0)[np.maximum(corners, 0)]).sum(axis=1)
        # The distance from the body each vertex sits at: the offset, or for a faded one, its own blend
        # of the offset and its anchor's height.
        level = self.offset
        # Auto Smooth leaves faded stacks alone: their shape is the blend, already smooth, and Push Out and
        # Tighten couldn't undo what smoothing does the other way, so stroke after stroke it would wear the
        # fade down.
        still = np.zeros(len(candidates), dtype=bool)
        if self.fade is not None:
            level = fade * self.offset + (1.0 - fade) * self.anchor_height[candidates]
            under = np.where(weights > 0.0, spot_fade[np.maximum(corners, 0)], 1.0).min(axis=1)
            still = (fade < 1.0) | (under < 1.0)
        own = travel + (level - near.distance) / gain
        # Vertices on their own head for the offset alone.
        aim = np.where(alone, own, np.maximum(goal, own))
        if self.layered:
            # Where the fit alone takes each vertex from its reference position: Push Out only lifts,
            # Tighten only lowers.
            fit = aim
            if self.mode == 'PUSH':
                fit = np.maximum(aim, 0.0)
            elif self.mode == 'TIGHTEN':
                fit = np.minimum(aim, 0.0)
            # Auto Smooth's share: how far smoothing the shape the fit heads for moves each vertex. It works
            # either way in every mode (a bumpy garment clear of the body evens out with Push Out too), but
            # never takes a vertex closer to the body than the offset, or than the fit puts it. Both are
            # measured from the reference positions, so going over a spot again doesn't smooth it further.
            shift = self._smoothing(candidates, direction, fit, (weight > 0.0) & ~still & ~alone, riders)
            aim = np.maximum(fit + shift, np.minimum(fit, own))
            step = alpha * (aim - travel)
        else:
            step = alpha * (aim - travel)
            if self.mode == 'PUSH':
                step = np.maximum(step, 0.0)
            elif self.mode == 'TIGHTEN':
                step = np.minimum(step, 0.0)
        limit = MAX_STEP * radius
        step = np.clip(step, -limit, limit)
        moving = np.abs(step) > 1e-9
        indices = candidates[moving]
        motion = step[moving, None] * direction[moving]
        if self.layered:
            # How much of its step each vertex needs to get out of the body (see `_unfolded`).
            least = _out_of(near.distance[moving], step[moving] * gain[moving])
            if len(riders):
                moved, motion, out = self._ride(riders, carried, heading, indices, motion)
                indices = np.concatenate((indices, moved))
                least = np.concatenate((least, out))
            if len(indices):
                indices, motion = self._unfolded(indices, motion, least)
        self.co[indices] += motion

        if self.smooth_mesh:
            # Without triangles, Auto Smooth smooths the mesh's height above the body (see `_auto_smooth`).
            distance = near.distance + np.where(moving, step * gain, 0.0)
            height = np.einsum("ij,ij->i", self.start[candidates] - self.base[candidates], direction)
            amount = SMOOTH_RATE * self.smooth * np.where(still, 0.0, weight)
            indices = np.union1d(indices, self._auto_smooth(candidates, direction, gain, height, distance, amount))
        if self.linked and not self.layered and len(indices):
            indices = self._move_together(moving_set, before, before_start)
        if not len(indices):
            return nothing
        self.moved[indices] = True
        return indices

    def _aim(self, candidates, near, needed):
        """The directions ``candidates`` move in. Only the ``needed`` ones get their own (those that may
        move); the others haven't moved and take the normal at their closest body point (``near``).

        A vertex's direction is picked where the stroke first reaches it and kept, so it moves in a
        straight line. It is the body's normal smoothed over the Keep Together distance
        (`BodySurface.smoothed`): across a crease, the closest body point jumps from one side to the
        other, and moving along its normal would pull neighbouring vertices apart (or push them into
        each other). The point of that smoothed body it moves away from is kept too (``self.base``):
        over a crease it lies between both sides, so it tells how deep the mesh may be pulled in (see
        `dab`). With triangles, the body counts by how much it faces the same way as the mesh there
        (`_facing`): where a trouser leg wraps a thigh, the other thigh's side across the gap faces
        the other way and doesn't count, while both sides of a crease that the mesh spans do. A vertex
        closer to the other thigh than to its own, or sunk into it, is averaged around where the line
        back along its facing meets its own thigh instead (`_line_contact`), and one that finds no body
        facing its way at all moves along its facing. Where the mesh lies along the body instead, the
        direction turns toward the normal at the closest body point, so the vertex slides less across the
        body toward a crease beside it (see `_follow`). Only vertices that face some way and found body
        facing that way bridge creases (``self.bridging``). Then the vertex also looks for a layer of the
        mesh under it (`_layers`).
        """
        fresh = needed & ~self.aimed[candidates]
        if fresh.any():
            new = candidates[fresh]
            under = near.take(fresh)
            facing = None
            if self.layered:
                facing = self._facing(new)
                self.facing[new] = facing
                # How much the mesh faces the way of the normal at each closest body point. `_line_contact`
                # leaves those of the vertices lying along the body as they are.
                agree = np.einsum("ij,ij->i", facing, under.normal)
                under = self._line_contact(new, under, facing)[0]
            directions, bases, found = self.body.smoothed(self.co[new], under, self.layer_radius, facing)
            if facing is not None:
                faces = np.einsum("ij,ij->i", facing, facing) > 0.0
                directions[faces & ~found] = facing[faces & ~found]
                # Where the mesh lies along the body: turned toward the closest body point.
                follow = self._follow(new, under, agree, directions)
                directions += follow[:, None] * (under.normal - directions)
                directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
                self.bridging[new] = faces & found
            self.direction[new], self.base[new] = directions, bases
            self.aimed[new] = True
            if self.layered:
                self._layers(new, facing)
        return np.where(self.aimed[candidates, None], self.direction[candidates], near.normal)

    def _loose(self, gap):
        """How loose clothing ``gap`` off the body is (0..1): not at all up to half Keep Together (or of
        ``BRIDGE_LOOSE`` if that is farther) off it, fully from all of it on, in a smooth step between."""
        t = np.clip(2.0 * gap / max(self.layer_radius, BRIDGE_LOOSE) - 1.0, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _follow(self, vertices, near, agree, directions):
        """How far each of ``vertices`` turns from its direction (``directions``, the body's normal smoothed
        over Keep Together) toward the normal at its closest body point (``near``), 0..1.

        Where the mesh lies along the body, facing the way of that normal (``agree``: the cosine between
        them, see ``FOLLOW_FROM``), there is no crease for it to span, and moving along that normal takes
        the vertex straight to where it settles instead of sliding it across the body toward a crease
        beside it. But the closest body points of neighbouring vertices can lie on either side of a
        crease, and neighbours can follow their normals more or less: each vertex turns only as far as
        that moves where it settles by ``FOLLOW_SHIFT`` of the length of its edges, so the mesh isn't
        pulled apart or bunched up, whatever the vertices around it do. Clothing inside the body doesn't
        turn (on both sides of a crease, the normals lead into each other), and loose clothing turns only
        as far as it isn't loose (see `_loose`): it spans the valleys of the body rather than lying along it.
        """
        follow = np.clip((agree - FOLLOW_FROM) / (FOLLOW_FULL - FOLLOW_FROM), 0.0, 1.0)
        follow *= follow * (3.0 - 2.0 * follow)
        follow *= np.where(near.distance < 0.0, 0.0, 1.0 - self._loose(near.distance))
        # Along a direction at an angle a from the normal, a vertex rising ``need`` settles need x tan a
        # sideways of its closest point (the body taken as flat there); along a blend f of it and the
        # normal, need x (1 - f) sin a / (cos a + f (1 - cos a)). That moves by no more than ``room`` x need
        # for f up to the limit below.
        cosine = np.einsum("ij,ij->i", directions, near.normal)
        sine = np.sqrt(np.maximum(1.0 - cosine * cosine, 0.0))
        tangent = sine / np.maximum(cosine, 1e-12)
        need = np.maximum(np.abs(near.distance - self.offset), 1e-12)
        room = FOLLOW_SHIFT * self.spacing[self.weld[vertices]] / need
        limit = room * cosine / np.maximum(sine + (tangent - room) * (1.0 - cosine), 1e-12)
        return np.minimum(follow, np.clip(np.where(tangent <= room, 1.0, limit), 0.0, 1.0))

    def _unfolded(self, indices, motion, least):
        """``motion`` (one row per vertex in ``indices``), shortened where it would turn a triangle of the
        mesh over against how it faced where the stroke found it: halved for the corners that move, up to
        ``FOLD_TRIES`` times, then not taken at all, but never below the share of it each vertex needs to
        get out of the body (``least``): clothing clipping into the bottom of a narrow crease is pushed
        out even where that folds it. Triangles turned over already are left to themselves. Slivers along
        a seam, whose corners head different ways, keep their side that way.

        Welded vertices move alike, all of a group as its members do on average (those that don't move
        count as standing still), so seams never open up, and the check sees how they really move.
        Returns the vertices that move and their motion."""
        if self.linked:
            groups, inverse = np.unique(self.weld[indices], return_inverse=True)
            inverse = inverse.reshape(-1)
            total = np.zeros((len(groups), 3))
            np.add.at(total, inverse, motion)
            needs = np.zeros(len(groups))
            np.maximum.at(needs, inverse, least)
            indices = _rows_of(*self.weld_rows, groups)
            group = np.searchsorted(groups, self.weld[indices])
            motion = (total / self.weld_size[groups, None])[group]
            least = needs[group]
        tris = np.unique(_rows_of(*self.tri_rows, indices) // 3)
        corners = self.tris[tris]
        facing = self.tri_unit[tris]

        def upright(points, rows):
            a, b, c = (points[rows[:, k]] for k in range(3))
            return np.einsum("ij,ij->i", np.cross(b - a, c - a), facing) > 0.0
        keep = upright(self.co, corners)
        corners, facing = corners[keep], facing[keep]
        if not len(corners):
            return indices, motion
        # Only the corners of these triangles are looked at, so the moves are kept for those alone (the
        # mesh can be a lot larger than the brush).
        verts, rows = np.unique(corners, return_inverse=True)
        rows = rows.reshape(corners.shape)
        base = self.co[verts]
        slot = np.minimum(np.searchsorted(verts, indices), len(verts) - 1)
        there = verts[slot] == indices
        slot = slot[there]
        moves = np.zeros((len(verts), 3))
        scale = np.ones(len(indices))
        for attempt in range(FOLD_TRIES + 1):
            moves[slot] = (scale[:, None] * motion)[there]
            over = ~upright(base + moves, rows)
            if not over.any():
                break
            # Whole welded groups halve their steps.
            folding = np.zeros(self.weld_count, dtype=bool)
            folding[self.weld[corners[over].reshape(-1)]] = True
            scale[folding[self.weld[indices]]] *= 0.5 if attempt < FOLD_TRIES else 0.0
            scale = np.maximum(scale, least)
        return indices, scale[:, None] * motion

    def _local_edges(self, welded):
        """The welded edges between the ``welded`` vertices (sorted, as from np.unique), in their order."""
        out = _rows_of(*self.edge_rows, welded)
        tail, head = self.tail[out], self.head[out]
        slot = np.minimum(np.searchsorted(welded, head), len(welded) - 1)
        keep = (welded[slot] == head) & (tail < head)
        return np.stack((np.searchsorted(welded, tail[keep]), slot[keep]), axis=1)

    def _close_along(self, values, welded, edges):
        """`_close` of ``values`` on the ``welded`` vertices (sorted) along ``edges`` between them (see
        `_local_edges`), reaching ``CLOSE_REACH`` x Keep Together each way along the mesh as the stroke
        found it: dips in what the mesh needs up to about twice that wide are filled. Edges count by their
        length, in at most ``MAX_RINGS`` steps, so edges longer than the reach share nothing. ``values``
        may hold several rows, each closed on its own."""
        reach = CLOSE_REACH * self.layer_radius
        if not len(edges) or reach <= 0.0:
            return values
        length = np.linalg.norm(self.points[welded[edges[:, 0]]] - self.points[welded[edges[:, 1]]], axis=1)
        costs = np.maximum(np.rint(length * (MAX_RINGS / reach)), 1.0).astype(np.int64)
        if values.ndim == 1:
            return _close(values, edges, costs, MAX_RINGS)
        # All rows in one go, as copies of the graph side by side.
        rows, count = values.shape
        edges = np.concatenate([edges + k * count for k in range(rows)])
        return _close(values.reshape(-1), edges, np.tile(costs, rows), MAX_RINGS).reshape(rows, count)

    def _bridged(self, stack, welded, edges, candidates, spots, direction, weight, gap, center, radius):
        """How far each of the ``welded`` vertices may be pulled in at most (-inf: as far as it needs), where
        loose clothing would be pulled into a valley of the body deeper than it spans it, and how loose each
        is (0..1). ``stack`` is what they need (``spots``: which of them each of ``candidates`` is, see
        `dab`), ``gap`` how far the innermost layer over each lies from the body.

        Seen from the brush (along the way the mesh faces under it, averaged by the falloff ``weight``
        over each connected part of the mesh), every spot lies at some height and heads for another. A
        closing of each over ``BRIDGE_REACH`` x Keep Together along the mesh fills their dips narrower than
        about twice that: of where the spots head for, the valleys of the body under the mesh (the gap
        between the thighs, seen from a skirt's front), and of where they lie, the dips of the mesh
        itself. No spot heads deeper below the first than it lies below the second: a skirt's front stays
        about level with where it settles on the thighs instead of sagging into the gap, while clothing
        that dips into a crease itself still settles into it, and a loose bulge (which only rises above
        what's around it) is pulled in. Only loose clothing is held: from Keep Together off the body, or
        ``BRIDGE_LOOSE`` if that is farther, and partly from half that on. Nearer, a tight suit settles
        into the creases it lies over.

        Only spots under the brush are held, which see that far past them on every side within what the
        dab looks at, and those under the middle of the brush (``BRIDGE_CORE``) stay held that far for the
        rest of the stroke: a dab off to one side of a spot sees less of the valley's far side. Spots
        moving farther than ``BRIDGE_COSINE`` from the way the brush looks don't count. `dab` shares how
        far spots are held among the loose ones only, so a spot held over the valley doesn't hold up the
        clothing lying on the body beside it.
        """
        loose = self._loose(gap)
        floor = np.where(np.isfinite(stack), self.bridge[welded], -np.inf)
        reach = BRIDGE_REACH * self.layer_radius
        points = self.points[welded]
        away = np.linalg.norm(points - center, axis=1)
        # The spots that may be held: loose ones being pulled in, under the brush.
        held = (loose > 0.0) & (stack < 0.0) & (away <= radius + self.layer_radius - reach)
        if not len(edges) or not held.any() or reach <= 0.0:
            return floor, loose
        count = len(welded)
        # The way the brush looks, per part of the mesh.
        part, which = np.unique(self.weld_part[welded], return_inverse=True)
        which = which.reshape(-1)
        facing = self.facing[candidates] * weight[:, None]
        up = np.stack([np.bincount(which[spots], facing[:, k], len(part)) for k in range(3)], axis=1)
        up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-30)
        up = up[which]
        heading = np.zeros((count, 3))
        heading[spots] = direction
        # How much a spot rises, seen from the brush, per unit it moves along its direction.
        lift = np.einsum("ij,ij->i", heading, up)
        seen = (lift > BRIDGE_COSINE) & np.isfinite(stack)
        lift = np.where(seen, lift, 1.0)
        need = np.where(seen, stack, 0.0)
        ground = np.where(seen, np.einsum("ij,ij->i", points, up), -np.inf)
        # Where the spots head for and where they lie, closed together (edges never join two parts).
        edges = edges[seen[edges].all(axis=1)]
        length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        costs = np.maximum(np.rint(length * (MAX_RINGS / reach)), 1.0).astype(np.int64)
        closed = _close(np.concatenate((ground + need * lift, ground)), np.concatenate((edges, edges + count)),
                        np.concatenate((costs, costs)), MAX_RINGS)
        span, dip = closed[:count], closed[count:]
        held &= seen & np.isfinite(span) & np.isfinite(dip)
        rise = np.minimum((np.where(held, span, 0.0) - np.where(held, dip, 0.0)) / lift, 0.0)
        floor = np.maximum(floor, np.where(held, need + loose * np.maximum(rise - need, 0.0), -np.inf))
        # Held under the middle of the brush: kept for the rest of the stroke.
        core = held & (away <= BRIDGE_CORE * radius)
        self.bridge[welded[core]] = floor[core]
        return floor, loose

    def _sheets(self):
        """Read the mesh's triangles for which way it faces, see `_facing`.

        Triangles next to each other form sheets, which face one way: away from the body under most of
        them. A sheet ends where the mesh turns more than ``SHEET_COSINE`` or bends more sharply than
        ``FOLD_RADIUS``, and at edges more than two triangles share (parts touching along them), so a hem
        folded back inside, the inside of a solid strap or a copy for the back side is a sheet of its
        own, facing the body too, though its triangles are the other way round; so are parts joined
        with their triangles the other way round (mirrored legs). The shape the mesh had where the
        stroke found it decides, so a crease the stroke gets to later still pulls both ways alike.
        """
        tris = self.tris
        a, b, c = (self.co[tris[:, k]] for k in range(3))
        normals = np.cross(b - a, c - a)
        unit = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        middle = (a + b + c) / 3.0
        # Two triangles sharing an edge (welded across seams) are in one sheet unless they turn too far.
        # An edge of more than two joins nothing.
        corners = self.weld[tris]
        first = corners.T.reshape(-1)
        second = corners[:, [1, 2, 0]].T.reshape(-1)
        low, high = np.minimum(first, second), np.maximum(first, second)
        face = np.flatnonzero(low != high)
        keys = low[face] * self.weld_count + high[face]
        order = np.argsort(keys)
        keys, face = keys[order], face[order] % len(tris)
        same = np.concatenate(([False], keys[1:] == keys[:-1], [False]))
        pair = np.flatnonzero(same[1:-1] & ~same[:-2] & ~same[2:])
        one, other = face[pair], face[pair + 1]
        cosine = np.einsum("ij,ij->i", unit[one], unit[other])
        keep = (one != other) & (cosine > SHEET_COSINE)
        # How sharply those that turn bend: the turn between their normals over how far their middles are
        # from the edge between them. (Flatter than about 6 degrees, triangles a quarter of a millimetre
        # wide would still pass.)
        bent = np.flatnonzero(keep & (cosine < FLAT_COSINE))
        points = self.points
        end = points[keys[pair[bent]] // self.weld_count]
        along = points[keys[pair[bent]] % self.weld_count] - end
        along /= np.maximum(np.linalg.norm(along, axis=1, keepdims=True), 1e-30)
        width = sum(np.linalg.norm(np.cross(middle[f[bent]] - end, along), axis=1) for f in (one, other))
        turn = np.arccos(cosine[bent])
        keep[bent] = width >= FOLD_RADIUS * turn
        sheet, count = _connected(len(tris), np.stack((one[keep], other[keep]), axis=1))
        # Each sheet's walls: its pieces between where it bends around SOLID_FOLD_RADIUS, the rims of a
        # closed solid (see `_vote`).
        gentle = keep.copy()
        gentle[bent] &= width >= SOLID_FOLD_RADIUS * turn
        wall, walls = _connected(len(tris), np.stack((one[gentle], other[gentle]), axis=1))
        # Per welded vertex and sheet around it (a "side"): the normals of the sheet's triangles there, by
        # area, as they were and smoothed over FACING_RINGS more rings of the sheet. Most vertices lie in
        # one sheet and are a side of their own; only around the others are the sheets told apart.
        flat = corners.T.reshape(-1)
        sheets = np.tile(sheet, 3).astype(np.float64)
        uses = np.bincount(flat, minlength=self.weld_count)
        total = np.bincount(flat, sheets, self.weld_count)
        mixed = total * total != np.bincount(flat, sheets * sheets, self.weld_count) * uses
        plain = np.flatnonzero((uses > 0) & ~mixed)
        split = mixed[flat]
        apart, inverse = np.unique(flat[split] * count + np.tile(sheet, 3)[split], return_inverse=True)
        side_of = np.full(self.weld_count, -1, dtype=np.int64)
        side_of[plain] = np.arange(len(plain))
        flat = side_of[flat]
        flat[split] = len(plain) + inverse.reshape(-1)
        side = flat.reshape(3, -1).T
        side_vertex = np.concatenate((plain, apart // count))
        side_sheet = np.concatenate(((total[plain] / uses[plain]).round().astype(np.int64), apart % count))
        sides = len(side_vertex)
        # The wall of one of each side's triangles (on a rim, either does).
        side_wall = np.zeros(sides, dtype=np.int64)
        side_wall[flat] = np.tile(wall, 3)

        def gather(values):
            """Per side, the sum of ``values`` (per triangle) over its triangles."""
            return np.stack([np.bincount(flat, np.tile(values[:, k], 3), minlength=sides) for k in range(3)], axis=1)
        self.side_normal = gather(normals)
        smooth = self.side_normal
        for _ in range(FACING_RINGS):
            smooth = gather(smooth[side[:, 0]] + smooth[side[:, 1]] + smooth[side[:, 2]])
        self.side_facing = smooth
        self.side_vertex, self.side_sheet, self.side_wall = side_vertex, side_sheet, side_wall
        self.side_rows = _rows(self.side_vertex, self.weld_count)
        self.sheet_rows = _rows(self.side_sheet, count)
        self.wall_rows = _rows(self.side_wall, walls)
        # Per triangle: its sheet, its wall and its unit normal; per wall, its sheet.
        self.tri_sheet, self.tri_wall, self.tri_unit = sheet, wall, unit
        self.wall_sheet = np.zeros(walls, dtype=np.int64)
        self.wall_sheet[wall] = sheet
        # Per sheet and per wall: +1 or -1 where its triangles' normals face away from the body or toward
        # it, 0 where it faces no way in particular, NaN until a stroke reaches it.
        self.sheet_sign = np.full(count, np.nan)
        self.wall_sign = np.full(walls, np.nan)

    def _facing(self, candidates):
        """The way the mesh faces at ``candidates`` (unit vectors, or zero where it faces no way in
        particular): the smoothed normals of the sheets around each (see `_sheets`), added up across
        welded seams, every sheet turned away from the body.

        A sheet decides which way that is the first time a stroke reaches it, by how its normals face
        the body's under up to ``SIDE_SAMPLES`` spots spread over it (by area, cosine-weighted): the
        side most of it faces. So the parts of a trouser leg closer to the other thigh, across the gap,
        still face the way the rest of the leg does. A sheet that doesn't clearly face one way (by
        ``SHEET_CLEAR``: a closed solid, a tip sticking out) is decided wall by wall (see `_sheets`):
        a closed solid's inner wall faces the body and its outer wall away, though its rims join them in
        smooth curves. Walls that don't clearly face one way either face no way in particular.
        """
        welded = self.weld[candidates]
        order, start = self.side_rows
        slots = _rows_of(order, start, welded)
        rows = np.repeat(np.arange(len(candidates)), start[welded + 1] - start[welded])
        sheet = self.side_sheet[slots]
        self._vote(sheet)
        normals = self.side_facing[slots] * self._sign(sheet, self.side_wall[slots])[:, None]
        sums = np.stack([np.bincount(rows, weights, minlength=len(candidates)) for weights in (
            normals[:, 0], normals[:, 1], normals[:, 2], np.linalg.norm(normals, axis=1))], axis=1)
        length = np.linalg.norm(sums[:, :3], axis=1)
        # Where the sheets around a vertex face every which way (a sharp ridge, a loose tip), it faces no
        # way in particular.
        clear = length > FACING_CLARITY * sums[:, 3]
        return np.where(clear[:, None], sums[:, :3] / np.maximum(length, 1e-30)[:, None], 0.0)

    def _sign(self, sheets, walls):
        """+1 or -1 where the triangles of the ``sheets`` (with their ``walls``, where a sheet faces no way in
        particular) face away from the body or toward it, 0 where they face no way in particular."""
        sign = self.sheet_sign[sheets]
        return np.where(sign == 0.0, self.wall_sign[walls], sign)

    def _vote(self, sheets):
        """Decide which way those of the ``sheets`` (indices, repeats allowed) that haven't yet face, and
        the walls of those that face no way in particular, see `_facing`."""
        new = np.unique(sheets[np.isnan(self.sheet_sign[sheets])])
        if not len(new):
            return
        self.sheet_sign[new] = self._ballot(new, self.sheet_rows, self.side_sheet)
        unclear = np.flatnonzero(np.isin(self.wall_sheet, new[self.sheet_sign[new] == 0.0]))
        if len(unclear):
            self.wall_sign[unclear] = self._ballot(unclear, self.wall_rows, self.side_wall)

    def _ballot(self, groups, rows, group_of):
        """+1, -1 or 0 per group of sides (``groups``, grouped by ``rows`` from ``group_of``, see `_rows`):
        whether their normals clearly face away from the body's under up to ``SIDE_SAMPLES`` spots spread
        over each, or toward them, or neither."""
        order, start = rows
        samples = []
        for one in groups.tolist():
            members = order[start[one]:start[one + 1]]
            spaced = np.linspace(0, len(members) - 1, min(SIDE_SAMPLES, len(members)))
            samples.append(members[spaced.astype(np.int64)])
        samples = np.concatenate(samples)
        near = self.body.nearest(self.points[self.side_vertex[samples]])
        vote = np.einsum("ij,ij->i", self.side_normal[samples], near.normal)
        vote[~near.found | near.beyond] = 0.0
        count = len(start) - 1
        signed = np.bincount(group_of[samples], vote, count)[groups]
        asked = np.bincount(group_of[samples], np.abs(vote), count)[groups]
        return np.where(signed > SHEET_CLEAR * asked, 1.0, np.where(signed < -SHEET_CLEAR * asked, -1.0, 0.0))

    def _layers(self, vertices, facing):
        """Look for a layer of the mesh under each of ``vertices``, as the mesh was where the stroke
        started: the first triangle hit going from its reference position against its direction,
        within Keep Together and before that line leaves the body again, other than its own. Failing
        that, the first triangle of another part of the mesh hit the other way, within Keep Together:
        the vertex dips under the part it lies on (a strap's edge pressed into the top under it).
        Triangles of other parts that face the other way than the vertex (its ``facing``, see
        `_facing`) don't count: they belong to another garment crossing this one, like trouser legs
        overlapping between the legs. The vertex rides on that layer from then on (see `_riding` and
        `_ride`), unless it lies farther from it than its part's layers usually do, or no corner of it
        may move: then it is fitted on its own, though never pulled in through it (see `_stacked`).
        How far the vertex lies from the layer under it counts toward its part's thickness."""
        if self.mesh_bvh is None:
            last, tri_list = FitStroke._tri_list
            if last is None or not np.array_equal(last, self.tris):
                tri_list = list(map(tuple, self.tris.tolist()))
                FitStroke._tri_list = (self.tris, tri_list)
            self.mesh_bvh = BVHTree.FromPolygons(self.start.tolist(), tri_list, all_triangles=True)
        cast = self.mesh_bvh.ray_cast
        welded, tris, part = self.weld, self.tris, self.part
        found = np.full(len(vertices), -1, dtype=np.int64)
        below = np.zeros(len(vertices), dtype=bool)
        location = np.zeros((len(vertices), 3))
        # Under a vertex, only as far as the body: past it lies the other side of the limb (a glove's
        # palm, seen from the back of the hand).
        depth = self.body.leave(
            self.start[vertices], -self.direction[vertices], np.full(len(vertices), self.layer_radius))
        points, backs = self.start[vertices].tolist(), (-self.direction[vertices]).tolist()
        rays = zip(range(len(vertices)), vertices.tolist(), points, backs, depth.tolist())
        for k, vertex, point, back, deep in rays:
            own = welded[vertex]
            for sign in (1.0, -1.0):
                ray = (sign * back[0], sign * back[1], sign * back[2])
                origin, left = point, (deep if sign > 0.0 else self.layer_radius)
                # Past the vertex's own triangles (a few, where the mesh curves), above it past its own
                # part, and past other parts facing the other way.
                for _ in range(LAYER_HITS):
                    hit, _normal, index, distance = cast(origin, ray, left)
                    if hit is None:
                        break
                    other = part[tris[index, 0]] != part[vertex]
                    if own not in welded[tris[index]] and (sign > 0.0 or other) and (
                            not other or self._faces(index, facing[k]) >= 0.0):
                        found[k] = index
                        location[k] = hit
                        break
                    left -= distance + 1e-6
                    if left <= 0.0:
                        break
                    origin = (hit[0] + 1e-6 * ray[0], hit[1] + 1e-6 * ray[1], hit[2] + 1e-6 * ray[2])
                if found[k] >= 0:
                    below[k] = sign > 0.0
                    break
        rides = found >= 0
        corners = tris[np.maximum(found, 0)]
        a, b, c = (self.start[corners[:, k]] for k in range(3))
        self.carrier[vertices[rides]] = found[rides]
        self.carrier_bary[vertices[rides]] = _barycentric(location, a, b, c)[rides]
        self.carrier_below[vertices[rides]] = below[rides]
        # How far each vertex lies from that layer, square to it.
        gaps = np.abs(np.einsum("ij,ij->i", self.start[vertices] - location, self.tri_unit[np.maximum(found, 0)]))
        self.carrier_gap[vertices[rides]] = gaps[rides]
        parts = self.part_count
        rides &= below
        keys, counts = np.unique(part[vertices[rides]] * parts + part[corners[rides, 0]], return_counts=True)
        for key, count in zip(keys.tolist(), counts.tolist()):
            self.on_part[key] = self.on_part.get(key, 0) + count
        # The thickness of each part's layers: the middle of the distances at which its vertices lie on one.
        own = part[vertices[rides]]
        gaps = gaps[rides]
        for one in np.unique(own).tolist():
            known = self.part_gaps.get(one)
            more = gaps[own == one]
            self.part_gaps[one] = more if known is None else np.concatenate((known, more))
            self.thickness[one] = np.median(self.part_gaps[one])

    def _faces(self, tri, facing):
        """How much triangle ``tri`` faces the way of ``facing`` (a cosine, 0 where either faces no way in
        particular), with its sheet turned away from the body (see `_facing`)."""
        sheet = self.tri_sheet[tri]
        if np.isnan(self.sheet_sign[sheet]):
            self._vote(np.array([sheet]))
        return float(self._sign(sheet, self.tri_wall[tri])) * float(self.tri_unit[tri] @ facing)

    def _riding(self, candidates):
        """Which ``candidates`` ride on the layer found for them (see `_layers`): one that may move, on
        their own part (a fold) or on a part that more of their part's vertices lie on than the other way
        round, and that they lie no farther from than their part's layers usually do (see `_slack`)."""
        riding = self.carrier[candidates] >= 0
        if not riding.any():
            return riding
        rows = np.flatnonzero(riding)
        vertices = candidates[rows]
        corners = self.tris[self.carrier[vertices]]
        own = self.part[vertices]
        under = self.part[corners[:, 0]]
        parts = self.part_count
        pairs, inverse = np.unique(own * parts + under, return_inverse=True)
        over = np.array([self.on_part.get(key, 0) for key in pairs.tolist()])
        back = np.array([self.on_part.get((key % parts) * parts + key // parts, 0) for key in pairs.tolist()])
        voted = ((over > back) | (pairs // parts == pairs % parts))[inverse.reshape(-1)]
        riding[rows] = voted & self.movable[corners].any(axis=1) & (self._slack(vertices) <= 0.0)
        return riding

    def _slack(self, vertices):
        """How much farther each of ``vertices`` lies from the layer under it than ``LAYER_SPREAD`` times
        its part's thickness (0 for a layer over it, or none)."""
        below = (self.carrier[vertices] >= 0) & self.carrier_below[vertices]
        usual = np.nan_to_num(LAYER_SPREAD * self.thickness[self.part[vertices]], nan=np.inf)
        return np.where(below, np.maximum(self.carrier_gap[vertices] - usual, 0.0), 0.0)

    def _stacked(self, fitted, need, riders, near):
        """``need`` of the ``fitted`` vertices, raised to what the layer of the mesh under each needs, where
        one was found under it (see `_layers`) that it doesn't ride on: the innermost layer over a spot
        decides how far everything over it moves, so a layer isn't pulled in through another one lying
        under it (a strap's edge pressed under the top it lies on, which rides on that top; a layer that
        may not move). A vertex lying farther from that layer than its part's layers usually do needs
        that much less (see `_slack`): the layer rises that far before it pushes the vertex. ``riders``
        need what their own distance from the body (``near``, their closest points) says, and layers
        that may not move nothing. Twice over, for layers on layers."""
        rows = np.flatnonzero((self.carrier[fitted] >= 0) & self.carrier_below[fitted])
        if not len(rows):
            return need
        values = np.full(len(self.co), np.nan)
        values[fitted] = need
        if len(riders):
            came = self.co[riders] - self.start[riders]
            values[riders] = self.offset - (near.distance - np.einsum("ij,ij->i", came, near.normal))
        corners = self.tris[self.carrier[fitted[rows]]]
        # Layers that may not move stay where they are.
        values[corners[~self.movable[corners]]] = 0.0
        bary = self.carrier_bary[fitted[rows]]
        slack = self._slack(fitted[rows])
        need = need.copy()
        for _ in range(2):
            under = values[corners]
            weights = np.where(np.isnan(under), 0.0, bary)
            total = weights.sum(axis=1)
            under = (weights * np.nan_to_num(under)).sum(axis=1) / np.maximum(total, 1e-12) - slack
            need[rows] = np.where(total > 0.0, np.maximum(need[rows], under), need[rows])
            values[fitted[rows]] = need[rows]
        return need

    def _held(self, riders, fitted, target):
        """The welded vertices of ``riders`` that aren't among the ``fitted`` welded vertices (sorted), and
        the target of the spot each rides on: the ``target`` of the fitted corners of the triangle under
        it, weighted as there (welded members share the largest)."""
        if not len(fitted):
            return np.empty(0, dtype=np.int64), np.empty(0)
        corners = np.searchsorted(fitted, self.weld[self.tris[self.carrier[riders]]])
        corners = np.minimum(corners, len(fitted) - 1)
        ok = (fitted[corners] == self.weld[self.tris[self.carrier[riders]]]) & np.isfinite(target[corners])
        weights = np.where(ok, self.carrier_bary[riders], 0.0)
        total = weights.sum(axis=1)
        value = (weights * np.where(ok, target[corners], 0.0)).sum(axis=1) / np.maximum(total, 1e-12)
        welded = self.weld[riders]
        keep = (total > 0.0) & ~np.isin(welded, fitted)
        held, inverse = np.unique(welded[keep], return_inverse=True)
        values = np.full(len(held), -np.inf)
        np.maximum.at(values, inverse.reshape(-1), value[keep])
        return held, values

    def _ride(self, riders, near, direction, indices, motion):
        """How the ``riders`` move this dab: as far as the spots they ride on rise from the body, so their
        distance from the layer under them stays as it is. That is interpolated from the corners of the
        triangle under each rider, which moved by ``motion`` (one row per vertex in ``indices``), twice
        over for layers riding on layers. Like the fitted vertices, riders move straight along their
        ``direction`` (see `_turn`), rather than sliding along with a layer that moves at a slant, and
        never closer to the body (``near``, their closest points) than the offset, or than they already
        are. A rider whose direction faces away from the body under it (an outer layer starting over the
        other thigh) moves just as the spot it rides on does instead, rather than being turned sideways
        toward that body. Returns the riders that moved, their motion appended to ``motion``, and how
        much of it each needs to get out of the body (see `_out_of`)."""
        corners = self.tris[self.carrier[riders]]
        bary = self.carrier_bary[riders]
        # Moves are kept for the vertices that matter: those that move, the riders and what they ride on.
        verts = np.unique(np.concatenate((indices, riders, corners.reshape(-1))))
        moves = np.zeros((len(verts), 3))
        moves[np.searchsorted(verts, indices)] = motion
        corners, here = np.searchsorted(verts, corners), np.searchsorted(verts, riders)
        for _ in range(2):
            moves[here] = np.einsum("ij,ijk->ik", bary, moves[corners])
        rise = np.einsum("ij,ij->i", moves[here], near.normal)
        rise = np.maximum(rise, np.minimum(self.offset - near.distance, 0.0))
        astray = np.einsum("ij,ij->i", direction, near.normal) < FACING_COSINE
        direction, gain = self._turn(direction, near.normal)
        carried = np.where(astray[:, None], moves[here], (rise / gain)[:, None] * direction)
        moving = np.abs(carried).max(axis=1) > 1e-12
        out = _out_of(near.distance[moving], np.einsum("ij,ij->i", carried[moving], near.normal[moving]))
        return riders[moving], np.concatenate((motion, carried[moving])), out

    def _contact(self, candidates, near, direction):
        """Where on the body each of ``candidates`` is measured from, and which way it moves this dab:
        its closest body point (``near``) and its ``direction``, except where that point faces away from
        the direction (see `_line_contact`). Then the vertex lies closer to some other part of the body
        (the other thigh, across a narrow gap) than to the part it faces, or has sunk into that other
        part, and is measured from where its line back meets the part it faces, past the one it has sunk
        into. Returns (near, direction, alone, lost):

        - ``alone``: vertices closer to that other part than the offset, or in it, that their line finds
          no part of the body to measure from, or any with Push Out (which never pulls them in toward
          the part they face), unless the shortest way out of the other part leads into another one (a
          crease, see ``EXIT_CLEARANCE``). They move out of the other part the shortest way, along its
          normal, on their own: they share nothing with the vertices around them, which head the other way.
        - ``lost``: the others that their line finds no part of the body for. They are too far from the
          part they face to tell where it is, and are left where they are.
        """
        near, away, found = self._line_contact(candidates, near, direction, self.mode == 'PUSH')
        alone = np.zeros(len(candidates), dtype=bool)
        lost = np.zeros(len(candidates), dtype=bool)
        missed = away[~found]
        close = near.distance[missed] < self.offset
        alone[missed[close]] = True
        lost[missed[~close]] = True
        if alone.any():
            direction = np.where(alone[:, None], near.normal, direction)
        return near, direction, alone, lost

    def _line_contact(self, vertices, near, directions, clear=False):
        """``near`` (the closest body points of ``vertices``), but for those whose closest point faces
        away from their (unit) ``directions`` by more than ``FACING_COSINE``: they lie closer to some
        other part of the body than to the part they face (the other thigh, across a narrow gap), or have
        sunk into that other part. They take where the line back from them meets the body from the side
        they face (past the part they have sunk into, see `BodySurface.along`) if that is within
        ``GAP_REACH`` (or Keep Together) of the closest point and doesn't graze it (see ``MIN_GAIN``).
        With ``clear``, only those at least the offset clear of the other part do, and those the shortest
        way out of it (along its normal) doesn't lead clear of the body (see ``EXIT_CLEARANCE``). Directions
        of zero length face no way and keep ``near``. Returns (near, the rows of those facing away, which
        of them found the body along their line)."""
        facing = np.einsum("ij,ij->i", directions, directions) > 0.0
        away = np.flatnonzero(facing & (np.einsum("ij,ij->i", directions, near.normal) < FACING_COSINE))
        if not len(away):
            return near, away, np.zeros(0, dtype=bool)
        limits = np.abs(near.distance[away]) + max(self.layer_radius, GAP_REACH)
        line = self.body.along(self.co[vertices[away]], directions[away], limits, through=True)
        found = line.found & (np.einsum("ij,ij->i", directions[away], line.normal) >= MIN_GAIN)
        if clear:
            unclear = np.flatnonzero(found & (near.distance[away] < self.offset))
            if len(unclear):
                found[unclear] = self._exit_blocked(vertices[away[unclear]], near.take(away[unclear]))
        if found.any():
            near = near.take(np.ones(len(vertices), dtype=bool))
            mask = np.zeros(len(vertices), dtype=bool)
            mask[away[found]] = True
            near.put(mask, line.take(found))
        return near, away, found

    def _exit_blocked(self, vertices, near):
        """Which of ``vertices``, closer to the body than the offset (``near``: their closest points), the
        shortest way out of the part they are closest to, along its normal, would take less than
        ``EXIT_CLEARANCE`` of the offset clear of the body: into another part of it, as in a crease."""
        lift = (self.offset - near.distance)[:, None] * near.normal
        there = self.body.nearest(self.co[vertices] + lift)
        return there.found & (there.distance < self.offset * EXIT_CLEARANCE - 1e-6)

    @staticmethod
    def _turn(direction, normal, least=MIN_GAIN):
        """``direction`` turned toward the body's ``normal`` where it gains less than ``least`` (a very
        narrow crease, or the vertex has moved to where the body faces another way), and how fast each
        gains distance from the body along it (the cosine to the normal)."""
        gain = np.einsum("ij,ij->i", direction, normal)
        low = gain < least
        if low.any():
            # Keep the direction's sideways part, at ``least`` from the normal.
            side = direction[low] - gain[low, None] * normal[low]
            length = np.linalg.norm(side, axis=1, keepdims=True)
            side = np.where(length > 1e-12, side / np.maximum(length, 1e-12), 0.0)
            turned = least * normal[low] + math.sqrt(1.0 - least * least) * side
            turned[length[:, 0] <= 1e-12] = normal[low][length[:, 0] <= 1e-12]
            direction = direction.copy()
            direction[low] = turned
            gain = np.einsum("ij,ij->i", direction, normal)
        return direction, gain

    def _move_together(self, candidates, before, before_start):
        """Without triangles, give every member of a welded group the group's average movement of this
        dab (and of its reference position, see `_auto_smooth`), so their distances to each other stay
        exactly as they were (with triangles, `_unfolded` moves them alike). Returns the indices of moved
        vertices."""
        unique, inverse = np.unique(self.weld[candidates], return_inverse=True)
        inverse = inverse.reshape(-1)
        members = np.bincount(inverse, minlength=len(unique)).astype(np.float64)[:, None]
        for positions, old in ((self.co, before), (self.start, before_start)):
            total = np.zeros((len(unique), 3))
            np.add.at(total, inverse, positions[candidates] - old)
            positions[candidates] = old + (total / members)[inverse]
        return candidates[(self.co[candidates] != before).any(axis=1)]

    def _auto_smooth(self, candidates, direction, gain, height, distance, amount):
        """Without triangles, smooth the height of the reference positions above the body, ``amount``
        (0..1) per candidate.

        ``height`` is each candidate's reference height along its ``direction`` above the smoothed body
        (see `_aim`), ``distance`` its current distance to the body, and ``gain`` how much of that it
        gains per unit along its direction. Heights rather than positions are smoothed, so the body's
        own curvature is left alone (no shrinking), and vertices only move along their directions, the
        same way the fit does. Over the smoothed body, a mesh spanning a crease isn't pulled into it.
        Smoothing never takes a vertex closer to the body than the offset (or than it already is).
        Returns the indices of the vertices that moved.
        """
        active = amount > 0.0
        if not active.any():
            return np.empty(0, dtype=np.int64)
        candidates = candidates[active]
        direction = direction[active]
        unique, inverse = np.unique(self.weld[candidates], return_inverse=True)
        inverse = inverse.reshape(-1)
        count = len(unique)
        edges = self._local_edges(unique)
        a, b = edges[:, 0], edges[:, 1]

        # Per welded vertex (members of a weld share one position and so one height).
        value = np.zeros(count)
        value[inverse] = height[active]
        rate = np.zeros(count)
        rate[inverse] = amount[active]
        allowed = np.zeros(count)
        allowed[inverse] = np.minimum(self.offset - distance[active], 0.0) / gain[active]

        total = np.zeros(count)
        degree = np.zeros(count)
        np.add.at(total, a, value[b])
        np.add.at(total, b, value[a])
        np.add.at(degree, a, 1.0)
        np.add.at(degree, b, 1.0)
        change = np.where(degree > 0.0, rate * (total / np.maximum(degree, 1.0) - value), 0.0)
        change = np.maximum(change, allowed)[inverse]

        moving = np.abs(change) > 1e-9
        indices = candidates[moving]
        offset = change[moving, None] * direction[moving]
        self.start[indices] += offset
        self.co[indices] += offset
        return indices

    def _smoothing(self, candidates, direction, aim, active, riders=None):
        """How far each of ``candidates`` would move along its ``direction`` for the shape the fit heads
        for (each reference position moved ``aim`` along its direction) to be smoothed (0 for those not
        ``active``).

        That shape is smoothed over the mesh's edges by ``self.pairs`` pairs of Taubin passes (the last
        one only in part, for Auto Smooth between whole pairs): one moves each vertex ``TAUBIN_SHRINK``
        of the way to its neighbours' average, the next slightly more than that back out. Bumps a few
        edges wide shrink, while larger shapes (wrinkles, the curvature of the body under the mesh)
        come out unchanged instead of shrinking. The shape doesn't change while vertices
        travel toward it, so going over a spot again doesn't wear it down further. ``riders`` stay where
        the layer they ride on takes them, so the fitted part of a layer eases into the part of it
        lying on another layer. Members of a weld share one position and so move alike.
        """
        shift = np.zeros(len(candidates))
        if not active.any() or self.pairs <= 0.0:
            return shift
        # Where each vertex heads for, from its reference position: as far as it has come, for those
        # that aren't candidates. Kept for the vertices that matter: the candidates, the riders and what
        # they ride on (the mesh can be a lot larger than the brush).
        members = candidates
        corners = None
        if riders is not None and len(riders):
            corners = self.tris[self.carrier[riders]]
            members = np.concatenate((candidates, riders))
        verts = np.unique(members if corners is None else np.concatenate((members, corners.reshape(-1))))
        heading = self.co[verts] - self.start[verts]
        heading[np.searchsorted(verts, candidates)] = aim[:, None] * direction
        if corners is not None:
            bary = self.carrier_bary[riders]
            corners, here = np.searchsorted(verts, corners), np.searchsorted(verts, riders)
            for _ in range(2):
                heading[here] = np.einsum("ij,ijk->ik", bary, heading[corners])
        unique, inverse = np.unique(self.weld[members], return_inverse=True)
        inverse = inverse.reshape(-1)
        edges = self._local_edges(unique)
        if not len(edges):
            return shift
        points = np.zeros((len(unique), 3))
        points[inverse] = self.start[members] + heading[np.searchsorted(verts, members)]
        held = np.ones(len(unique), dtype=bool)
        held[inverse[:len(candidates)]] = False
        original = points.copy()
        tail, head = _both_ways(edges)
        degree = np.bincount(tail, minlength=len(unique)).astype(np.float64)
        whole = math.floor(self.pairs)
        for pair in range(math.ceil(self.pairs)):
            last = points
            for rate in (TAUBIN_SHRINK, TAUBIN_INFLATE):
                points = points + rate * (_neighbour_mean(points, tail, head, degree) - points)
                points[held] = original[held]
            if pair == whole:
                points = last + (self.pairs - whole) * (points - last)
        change = (points - original)[inverse[:len(candidates)]]
        return np.where(active, np.einsum("ij,ij->i", change, direction), 0.0)
