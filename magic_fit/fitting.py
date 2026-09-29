# SPDX-License-Identifier: GPL-3.0-or-later

"""Fitting a mesh around a body without flattening it.

Shrinkwrap-like tools send every vertex to its own spot on the surface, which collapses the inner
and outer layers of thick clothing into one shell. Here the mesh is treated as layers stacked on
the body instead: for each spot *of the body*, a correction is worked out from the innermost layer
above it, and every layer over that spot gets the same displacement along the body's normal (with
the same brush falloff, which is measured along the body as well). Thickness, folds and other
details ride along unchanged.

Like `painting`, nothing here depends on the 3D view: `FitStroke` works on world space positions.
"""

import math

import numpy as np

from .painting import falloff_weights, read_geometry, surface_bvh

# Vertices closer than this are welded for vertex graphs (split normals, UV seams).
WELD_DISTANCE = 1e-5
# A closest body point this close to an open edge of the body (in barycentric coordinates; the search
# tree works in single precision) lies on that edge.
OPEN_EDGE_TOLERANCE = 1e-4
# Points whose closest body point lies on an open edge are beyond the body there (next to its edge, not
# in front of or behind it) when the direction to them is farther than this (a cosine, about 11 degrees)
# from the body's normal.
BEYOND_COSINE = 0.98
# No dab moves a vertex farther than this fraction of the brush radius.
MAX_STEP = 0.1
# Limit for how many edges away from a body vertex layers are still grouped with it.
MAX_RINGS = 8
# Auto Smooth drapes goals over up to this many times the Keep Together distance...
SMOOTH_REACH = 2.0
# ...in at most this many smoothing passes per dab (passes grow with the square of the reach).
MAX_SMOOTH_PASSES = 150
# ...and smooths the mesh's height above the body, moving each vertex up to this fraction of the way
# to its neighbours' average per dab.
SMOOTH_RATE = 0.5
# Heights are measured to the body with its triangles bent this far toward their vertex normals
# (Phong tessellation). 0.5 matches round shapes: flat triangles sag, the full bend overshoots.
PHONG_SHAPE = 0.5


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
    a = np.concatenate(first)
    b = np.concatenate(second)
    # Connected components: every point takes the lowest label in reach until nothing changes.
    labels = np.arange(count)
    while True:
        low = np.minimum(labels[a], labels[b])
        if np.array_equal(labels[a], low) and np.array_equal(labels[b], low):
            break
        np.minimum.at(labels, a, low)
        np.minimum.at(labels, b, low)
        labels = labels[labels]
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(-1)
    return labels[inverse], int(labels.max()) + 1


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

    __slots__ = ("found", "location", "normal", "distance", "height", "corners", "bary", "beyond")

    def take(self, mask):
        result = Nearest()
        for name in self.__slots__:
            setattr(result, name, getattr(self, name)[mask])
        return result


class BodySurface:
    """The body a mesh is fitted around: nearest points, smooth outward normals and a vertex graph.

    Vertices sharing a position (split normals, UV seams) are welded, so normals and values flow
    across seams. Normals point away from the enclosed volume even if the mesh is inside out. A body
    can be open, like a torso on its own: past its open edges (used by one triangle) it has no inside.
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
        self._weld = weld
        members = np.bincount(weld, minlength=count).astype(np.float64)
        self.co = np.zeros((count, 3))
        np.add.at(self.co, weld, co)
        self.co /= members[:, None]

        a, b, c = (self.co[self.corners[:, k]] for k in range(3))
        face_normals = np.cross(b - a, c - a)
        corners = self.corners.reshape(-1)
        normals = np.stack([
            np.bincount(corners, np.repeat(face_normals[:, k], 3), minlength=count) for k in range(3)
        ], axis=1)
        center = self.co.mean(axis=0)
        if np.einsum("ij,ij->", a - center, np.cross(b - center, c - center)) < 0.0:
            normals = -normals
        self.normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

        lengths = np.linalg.norm(self.co[self.edges[:, 0]] - self.co[self.edges[:, 1]], axis=1)
        self.edge_length = float(np.median(lengths)) if len(lengths) else 0.0

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

        ``height`` is the signed distance to a smooth version of the body instead (Phong
        tessellation: each flat triangle bent between its vertex normals), so a low-poly body's
        facets don't show up in it.

        ``beyond`` marks points past an open edge of the body (below a torso's waist, say): their closest
        body point is on the edge and they lie off to its side. The body doesn't reach them, so they are
        never inside it; their distance and height are the (positive) distance to the edge.
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

        corner_co = self.co[result.corners]
        corner_normals = self.normals[result.corners]
        along = np.einsum("ijk,ijk->ij", location[:, None, :] - corner_co, corner_normals)
        projected = location[:, None, :] - along[:, :, None] * corner_normals
        bulge = np.einsum("ij,ijk->ik", result.bary, projected) - location
        # Half of the full Phong bulge, which follows a round surface best (the full one overshoots).
        smooth_point = location + PHONG_SHAPE * bulge
        result.height = np.where(result.beyond, distance, np.einsum("ij,ij->i", points - smooth_point, result.normal))
        return result

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

    Every vertex heads for a goal: how far its stack's innermost layer has to move along the body
    normal, measured from its reference position (where the stroke found it). Stacks within
    ``layer_radius`` of each other share the largest (safest) goal, so layers lying beside each other
    and small folds move as one piece. Goals don't change while vertices travel toward them, so dabs
    settle instead of piling up; each dab covers strength x falloff of the remaining way.

    Auto Smooth (``smooth`` 0..1) keeps the result smooth in two ways, both scaled by it:

    - Goals are draped: instead of the area moved as one piece dropping off steeply at its edge (a
      rim), goals ease out over up to ``SMOOTH_REACH`` x ``layer_radius``, like cloth draped over
      what needs to move, never below what a stack needs.
    - The mesh under the brush is smoothed like the Auto-Smooth of Blender's sculpt brushes (needs the
      mesh's ``edges``): the height of the reference positions above the body is smoothed, so the
      body's curvature doesn't shrink the mesh and layers keep their distance. It never moves a vertex
      closer to the body than the offset and keeps vertices split along seams together. The fit
      carries on from the smoothed shape.

    Vertices closer together than ``weld_distance`` (split seams, separate parts or objects that
    touch) always move by exactly the same amount, so seams never open up. Such a group only moves
    when all of its vertices may.
    """

    def __init__(
            self, body, co, movable, *,
            mode='PUSH', offset=0.002, layer_radius=0.02, max_distance=None, edges=None, smooth=0.0,
            weld_distance=WELD_DISTANCE,
    ):
        self.body = body
        # World space positions, updated as vertices move.
        self.co = co
        # Reference positions the fit displacement is measured from: where the stroke found each
        # vertex, as smoothed by Auto Smooth.
        self.start = co.copy()
        self.mode = mode
        self.offset = offset
        self.layer_radius = layer_radius
        self.max_distance = max_distance
        self.moved = np.zeros(len(co), dtype=bool)
        self.smooth = smooth
        edge = body.edge_length
        if edge > 0.0:
            self.rings = min(MAX_RINGS, max(1, math.ceil(layer_radius / edge)))
            # Repeated neighbour averaging spreads about sqrt(passes) edges.
            reach = smooth * SMOOTH_REACH * layer_radius / edge
            self.passes = self.rings + min(MAX_SMOOTH_PASSES, math.ceil(1.2 * reach * reach))
        else:
            self.rings = self.passes = 1
        self.margin = layer_radius + math.sqrt(self.passes) * edge

        if edges is None:
            edges = np.empty((0, 2), dtype=np.int64)
        self.weld, self.weld_count, self.weld_edges = _weld(co, edges, weld_distance)
        # Whether any vertices were welded together.
        self.linked = self.weld_count < len(co)
        if self.linked:
            together = np.ones(self.weld_count, dtype=bool)
            np.logical_and.at(together, self.weld, movable)
            movable = together[self.weld]
        self.movable = movable
        self.smooth_mesh = smooth > 0.0 and len(self.weld_edges) > 0

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
        if self.linked:
            before = self.co[candidates].copy()
            before_start = self.start[candidates].copy()

        # The falloff is measured along the body, so all layers over a spot get the same strength.
        t = np.linalg.norm(near.location - anchor, axis=1) / radius
        inside = t < 1.0
        if not inside.any():
            return nothing
        weight = np.zeros(len(t))
        weight[inside] = falloff_weights(falloff, t[inside])
        alpha = weight * strength

        # Body vertices around the brush and the edges between them.
        margin = reach + self.margin + 2.0 * body.edge_length
        delta = body.co - anchor
        region = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) < margin * margin)
        local = np.full(len(body.co), -1, dtype=np.int64)
        local[region] = np.arange(len(region))
        edges = local[body.edges]
        edges = edges[(edges >= 0).all(axis=1)]
        corners = local[near.corners]

        # How far each vertex has to travel, from its reference position, to sit at the offset.
        # Measured from its current distance minus how far it came, which also corrects for the
        # distance gained per step differing from the step (curved or tilted body surfaces).
        travel = np.einsum("ij,ij->i", self.co[candidates] - self.start[candidates], near.normal)
        start_distance = near.distance - travel
        need = self.offset - start_distance

        # Per body vertex: the stack above it follows its innermost layer, which needs the most.
        stack = np.full(len(region), -np.inf)
        inner = np.full(len(region), np.inf)
        for k in range(3):
            ok = corners[:, k] >= 0
            np.maximum.at(stack, corners[ok, k], need[ok])
            np.minimum.at(inner, corners[ok, k], start_distance[ok])
        if self.max_distance is not None and self.mode != 'PUSH':
            # Loose parts (skirts, capes...) are not pulled in.
            stack[np.isfinite(inner) & (inner > self.max_distance) & (stack < 0.0)] = 0.0
        # What the mode lets each stack do: Push Out only lifts, Tighten only lowers.
        has_stack = np.isfinite(stack)
        if self.mode == 'PUSH':
            stack[has_stack] = np.maximum(stack[has_stack], 0.0)
        elif self.mode == 'TIGHTEN':
            stack[has_stack] = np.minimum(stack[has_stack], 0.0)

        # Stacks near each other share the largest target, then the result is smoothed (draped
        # further with Auto Smooth). Never below a stack's own target: that way nothing ends up
        # inside the body or pulled past the offset.
        target = stack
        for _ in range(self.rings):
            target = _spread_max(target, edges)
        known = np.isfinite(target)
        target = _smooth_above(target, stack, known, edges, self.passes)

        # Each vertex heads for the target of the spot below it, along the body normal.
        weights = np.where(corners >= 0, near.bary, 0.0)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        goal = (weights * np.where(known, target, 0.0)[np.maximum(corners, 0)]).sum(axis=1)
        step = alpha * (goal - travel)
        if self.mode == 'PUSH':
            step = np.maximum(step, 0.0)
        elif self.mode == 'TIGHTEN':
            step = np.minimum(step, 0.0)
        limit = MAX_STEP * radius
        step = np.clip(step, -limit, limit)
        moving = np.abs(step) > 1e-9
        indices = candidates[moving]
        self.co[indices] += step[moving, None] * near.normal[moving]

        if self.smooth_mesh:
            distance = near.distance + np.where(moving, step, 0.0)
            smoothed = self._auto_smooth(
                candidates, near.normal, near.height - travel, distance, SMOOTH_RATE * self.smooth * weight,
            )
            indices = np.union1d(indices, smoothed)
        if self.linked and len(indices):
            indices = self._move_together(candidates, before, before_start)
        if not len(indices):
            return nothing
        self.moved[indices] = True
        return indices

    def _move_together(self, candidates, before, before_start):
        """Give every member of a welded group the group's average movement of this dab, so their
        distances to each other stay exactly as they were. Returns the indices of moved vertices."""
        unique, inverse = np.unique(self.weld[candidates], return_inverse=True)
        inverse = inverse.reshape(-1)
        members = np.bincount(inverse, minlength=len(unique)).astype(np.float64)[:, None]
        for positions, old in ((self.co, before), (self.start, before_start)):
            total = np.zeros((len(unique), 3))
            np.add.at(total, inverse, positions[candidates] - old)
            positions[candidates] = old + (total / members)[inverse]
        return candidates[(self.co[candidates] != before).any(axis=1)]

    def _auto_smooth(self, candidates, normals, height, distance, amount):
        """Smooth the height of the reference positions above the body, ``amount`` (0..1) per candidate.

        ``height`` is each candidate's reference height above the (smoothed) body, ``distance`` its current
        one. Heights rather than positions are smoothed, so the body's own curvature is left alone
        (no shrinking) and vertices only move along the body normal, the same way the fit does.
        Smoothing never takes a vertex closer to the body than the offset (or than it already is).
        Returns the indices of the vertices that moved.
        """
        active = amount > 0.0
        if not active.any():
            return np.empty(0, dtype=np.int64)
        candidates = candidates[active]
        normals = normals[active]
        unique, inverse = np.unique(self.weld[candidates], return_inverse=True)
        inverse = inverse.reshape(-1)
        count = len(unique)
        local = np.full(self.weld_count, -1, dtype=np.int64)
        local[unique] = np.arange(count)
        edges = local[self.weld_edges]
        edges = edges[(edges >= 0).all(axis=1)]
        a, b = edges[:, 0], edges[:, 1]

        # Per welded vertex (members of a weld share one position and so one height).
        value = np.zeros(count)
        value[inverse] = height[active]
        rate = np.zeros(count)
        rate[inverse] = amount[active]
        allowed = np.zeros(count)
        allowed[inverse] = np.minimum(self.offset - distance[active], 0.0)

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
        offset = change[moving, None] * normals[moving]
        self.start[indices] += offset
        self.co[indices] += offset
        return indices
