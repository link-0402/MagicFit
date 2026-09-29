# SPDX-License-Identifier: GPL-3.0-or-later

"""Resizing meshes from one body shape to another.

A devkit turns one body into another with a shape key (Rue into RueXB, YAB into Rue...), which moves
every vertex of the body. Resizing moves a mesh fitted to the first shape along with it: every point of
space moves by a smooth average of how the body around it moves, over a patch of the body ``radius``
wide at the skin and wider the farther a point is from the body. Close to the body that is how the
skin under it moves, so tight clothing keeps hugging it; farther out, the average spans more of the
body, so loose parts keep their shape. Being one smooth function of space, it keeps layers and
thickness, keeps seams between parts and objects closed, and doesn't tear clothing spanning a crease
the way following each vertex's closest point on the body does (or Surface Deform, which binds to it).

Where the body changes a lot within a short distance (under large breasts, whose underside rises by
several centimetres while the ribs just below stay put), one smooth step can still fold the cloth over
itself. So the change is made in steps of a couple of radii at most, each averaged around the body as
it is by then, like the body growing or shrinking under the cloth.

Creases are bridged: the movement is taken from crease-free versions of both shapes, in which the skin
that a large ball rolled over the body can't reach (the underbust, the cleavage) is lifted onto the
balls. A dress spanning the underbust then keeps spanning it when the breasts grow, moving out with the
breast above it, instead of sinking into the deeper crease with the skin. Clothing that lay on the skin
is fitted back to the real body afterwards (Keep Close).

The mesh keeps its relation to the first shape: a part clipping into it by 2 mm clips into the second
one about as much. Body Fit fixes that afterwards.

The two shapes can also come from two different bodies (a devkit's RueXB body and another's Rue one):
their vertices are then matched by their UV maps, which body mods made for the same skin textures share.

Like `painting`, the engine (`ShapeChange`) works on world space arrays and doesn't need the 3D view.
"""

import math

import numpy as np
from mathutils.bvhtree import BVHTree

# Width of the average at the skin, in world units...
DEFAULT_RADIUS = 0.01
# ...but at least this fraction of the body's typical edge, or the average would follow its vertices.
MIN_RADIUS_EDGES = 0.5
# The average widens by this fraction of a point's distance from the body.
SPREAD = 0.5
# Each body sample weighs (1 - (d / R)^2)^3 at distance d from a point, up to a reach R that meets the
# body this many widths around the point's foot on it. That spreads like a Gaussian of the width (the
# same variance over a flat body), but fades out smoothly instead of trailing off.
KERNEL_REACH = math.sqrt(10.0)
# The change is made in steps that move the body at most this many radii each...
STEP_RADII = 2.25
# ...but in no more than this many.
MAX_STEPS = 12
# Levels of coarser samples for wide averages: each clusters the body into cubes twice as large.
MAX_LEVELS = 12
# Points are grouped by width, this many groups per doubling, so each group's search cubes fit it.
BINS_PER_OCTAVE = 4
# Points handled at once (bounds the memory for their pairs of samples).
CHUNK = 4096

# Keep Close: afterwards, points that lay this close to the source shape (tight clothing)...
KEEP_CLOSE_DISTANCE = 0.005
# ...are brought back to their old distance from the target shape where they ended up more than this
# farther out (bags under breasts that shrink a lot)...
PULL_SLACK = 0.002
# ...or more than this farther in (sinking into it).
PUSH_SLACK = 0.0005
# Corrections are spread over the clothing's edges, so it doesn't crease: in each pass a point that lay
# close keeps this share of its own correction and takes the rest from its neighbours' average, and
# other points take their neighbours' average, fading by KEEP_FADE per edge.
KEEP_SHARE = 0.05
KEEP_FADE = 0.9
KEEP_PASSES = 80
# Rounds of measuring and correcting (each measures the distances anew).
KEEP_ROUNDS = 8

# Bridge Creases: clothing off the skin moves with crease-free versions of both shapes, in which the skin
# that a ball of this radius rolled over the body can't reach is lifted onto the balls (the setting's
# default).
BRIDGE_RADIUS = 0.08
# A ball fits where no part of the body comes closer to its center than its radius less this.
BRIDGE_TOLERANCE = 0.0015
# The lifts are averaged over neighbouring vertices across about this distance, so they change gradually
# where the closest ball switches from one side of a crease to the other...
BRIDGE_SMOOTHING = 0.014
# ...in at most this many passes.
BRIDGE_PASSES = 100
# Clothing closer than this to the source shape lies on the skin and follows it (a skin-tight suit keeps
# following a crease that deepens); from this much farther out it moves with the crease-free shapes, in
# between with a blend of both...
BRIDGE_HUG = 0.003
BRIDGE_BLEND = 0.003
# ...averaged over the clothing around each point across this distance, so neighbouring points (and
# layers) blend alike and the clothing doesn't tear where the two moves differ.
BRIDGE_BLEND_WIDTH = 0.015


class ResizeError(Exception):
    """A resize that can't be done; the message is meant for the user."""


# -----------------------------------------------------------------------------
# Engine

def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _vertex_areas(co, tris):
    """A third of the area of each triangle around each vertex: the surface a vertex stands for."""
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 6.0
    result = np.zeros(len(co))
    for k in range(3):
        np.add.at(result, tris[:, k], area)
    return result


def _cells(co, size):
    """Integer coordinates of the cubes of edge ``size`` that points are in."""
    return np.floor(co / size).astype(np.int64)


def _cell_keys(cells):
    # Cube coordinates stay well inside +-2^20 for any mesh measured in meters.
    shifted = cells + (1 << 20)
    return (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]


# Cubes half as large as the reach: every sample within reach of a point is in the 5 x 5 x 5 cubes
# around the point's (one more cube away, it's at least two cubes, the whole reach, away).
_STENCIL = np.array([(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)], dtype=np.int64)


class _Grid:
    """Samples sorted into cubes half as wide as ``reach``, to find every sample within ``reach`` of a point."""

    def __init__(self, co, reach):
        self.size = 0.5 * reach
        keys = _cell_keys(_cells(co, self.size))
        self.order = np.argsort(keys, kind='stable')
        self.keys, self.starts, self.counts = np.unique(keys[self.order], return_index=True, return_counts=True)

    def pairs(self, points):
        """(point index, sample index) for every sample in the cubes around each point."""
        cells = _cells(points, self.size)
        query = _cell_keys((cells[:, None, :] + _STENCIL[None, :, :]).reshape(-1, 3))
        slot = np.minimum(np.searchsorted(self.keys, query), len(self.keys) - 1)
        found = np.flatnonzero(self.keys[slot] == query)
        counts = self.counts[slot[found]]
        run = np.repeat(np.arange(len(found)), counts)
        offsets = np.arange(len(run)) - (np.cumsum(counts) - counts)[run]
        return found[run] // len(_STENCIL), self.order[self.starts[slot[found]][run] + offsets]


class _Levels:
    """The body's vertices, and coarser clusters of them for wide averages.

    Level 0 holds the vertices themselves; level ``l`` clusters them into cubes of ``radius x 2^(l-1)``.
    Clusters are made once, from the source shape, and follow their vertices through the steps.
    """

    def __init__(self, co, radius):
        self.members = [None]
        for level in range(1, MAX_LEVELS):
            _keys, inverse = np.unique(_cell_keys(_cells(co, radius * 2.0 ** (level - 1))), return_inverse=True)
            self.members.append(inverse.reshape(-1))
            if len(_keys) <= 1:
                break

    def samples(self, level, co, area, displacement):
        """Positions, areas and displacements of the samples at ``level`` for the body at ``co``."""
        if level == 0:
            return co, area, displacement
        members = self.members[min(level, len(self.members) - 1)]
        count = int(members.max()) + 1
        total = np.bincount(members, weights=area, minlength=count)
        keep = total > 0.0
        weight = area / np.maximum(total[members], 1e-300)

        def mean(values):
            return np.stack([np.bincount(members, weights=weight * values[:, k], minlength=count) for k in range(3)], 1)
        return mean(co)[keep], total[keep], mean(displacement)[keep]


def _average(points, reach, samples):
    """For each point, the average displacement of the samples within ``reach`` of it, each weighted by
    its area and (1 - (distance / reach)^2)^3. Points without any sample in reach don't move."""
    co, area, displacement = samples
    result = np.zeros_like(points)
    if not len(co):
        return result
    grid = _Grid(co, float(reach.max()))
    # Pairs are worked out in single precision, relative to each point, which is plenty for weights.
    co = co.astype(np.float32)
    area = area.astype(np.float32)
    displacement = displacement.astype(np.float32)
    inverse = (1.0 / reach ** 2).astype(np.float32)
    for first in range(0, len(points), CHUNK):
        chunk = slice(first, first + CHUNK)
        p = points[chunk].astype(np.float32)
        owner, sample = grid.pairs(points[chunk])
        delta = co[sample] - p[owner]
        falloff = np.maximum(1.0 - np.einsum("ij,ij->i", delta, delta) * inverse[chunk][owner], 0.0)
        weight = area[sample] * falloff * falloff * falloff
        total = np.bincount(owner, weights=weight, minlength=len(p))
        moved = displacement[sample]
        for k in range(3):
            result[chunk, k] = np.bincount(owner, weights=weight * moved[:, k], minlength=len(p))
        result[chunk] /= np.maximum(total, 1e-300)[:, None]
    return result


def bridged(co, tris, radius):
    """The body shape ``co`` (triangles ``tris``) with its creases filled: each vertex that a ball of
    ``radius`` rolled over the outside of the body can't reach is lifted onto the closest ball.

    There is a ball at every vertex, touching it along the vertex normal; it fits where no other part of
    the body comes closer to its center. The lifts are smoothed over the body, so they change gradually
    between vertices. Returns positions for the same vertices; those of no face stay where they are.
    """
    from mathutils.kdtree import KDTree

    from .fitting import _weld

    co = np.asarray(co, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    # Vertices at the same spot (split seams) are one, so lifts flow across seams.
    weld, count, pairs = _weld(co, tris)
    members = np.maximum(np.bincount(weld, minlength=count), 1).astype(np.float64)
    welded = np.stack([np.bincount(weld, weights=co[:, k], minlength=count) for k in range(3)], 1) / members[:, None]
    corners = weld[tris]
    a, b, c = (welded[corners[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    normals = np.zeros((count, 3))
    for k in range(3):
        np.add.at(normals, corners[:, k], face)
    center = welded.mean(axis=0)
    # Normals point away from the enclosed volume, even when the mesh is inside out.
    if np.einsum("ij,ij->", a - center, np.cross(b - center, c - center)) < 0.0:
        normals = -normals
    length = np.linalg.norm(normals, axis=1)
    faced = np.flatnonzero(length > 0.0)
    if not len(faced):
        return co.copy()
    normals[faced] /= length[faced, None]
    centers = welded[faced] + radius * normals[faced]
    find = BVHTree.FromPolygons(co.tolist(), tris.tolist(), all_triangles=True).find_nearest
    gap = np.array([find(point)[3] for point in centers.tolist()])
    fits = centers[gap >= radius - BRIDGE_TOLERANCE]
    if not len(fits):
        return co.copy()
    tree = KDTree(len(fits))
    for index, point in enumerate(fits.tolist()):
        tree.insert(point, index)
    tree.balance()
    closest = [tree.find(point) for point in welded[faced].tolist()]
    ball = np.array([hit[0] for hit in closest])
    distance = np.array([hit[2] for hit in closest])
    lift = np.zeros((count, 3))
    reached = distance <= radius
    lift[faced] = np.where(reached[:, None], 0.0, (ball - welded[faced]) * (1.0 - radius / np.maximum(distance, 1e-12))[:, None])
    if len(pairs):
        first, second = pairs[:, 0], pairs[:, 1]
        edge = float(np.median(np.linalg.norm(welded[first] - welded[second], axis=1)))
        passes = min(BRIDGE_PASSES, int(math.ceil(2.0 * (BRIDGE_SMOOTHING / edge) ** 2))) if edge > 0.0 else 0
        degree = np.maximum(np.bincount(first, minlength=count) + np.bincount(second, minlength=count), 1)[:, None]
        for _ in range(passes):
            average = np.stack([
                np.bincount(first, weights=lift[second, k], minlength=count)
                + np.bincount(second, weights=lift[first, k], minlength=count)
                for k in range(3)
            ], axis=1) / degree
            lift = 0.5 * (lift + average)
    unfaced = np.ones(count, dtype=bool)
    unfaced[faced] = False
    lift[unfaced] = 0.0
    return co + lift[weld]


class ShapeChange:
    """How a body changes from one shape to another, to move meshes along with it.

    ``source`` and ``target`` are the body's vertex positions in both shapes (world space, the same
    vertices), ``tris`` its triangles. ``radius`` is how wide the body's movement is averaged at the
    skin: details of the change smaller than that are smoothed out, and so are sharp differences
    between surfaces closer than that (the two sides of a crease). With ``bridge`` (a ball radius, see
    `bridged`), clothing that doesn't lie on the skin moves with crease-free versions of both shapes;
    0 follows the skin everywhere.
    """

    def __init__(self, source, target, tris, radius=DEFAULT_RADIUS, bridge=0.0):
        self.source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
        self.target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
        self.tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        if self.source.shape != self.target.shape:
            raise ResizeError("The two body shapes have different vertex counts")
        area = _vertex_areas(self.source, self.tris) if len(self.tris) else np.zeros(len(self.source))
        if area.sum() <= 0.0:
            raise ResizeError("The body has no faces")
        edges = np.linalg.norm(self.source[self.tris] - self.source[np.roll(self.tris, 1, axis=1)], axis=2)
        self.radius = max(float(radius), MIN_RADIUS_EDGES * float(np.median(edges)), 1e-6)
        # Vertices of no face (loose ones) stand for no surface and are left out.
        self._used = area > 0.0
        self._levels = _Levels(self.source[self._used], self.radius)
        self._tree_tris = self.tris.tolist()
        moves = np.linalg.norm(self.target - self.source, axis=1)[self._used]
        # Bridge Creases: crease-free versions of both shapes (and their clusters), for clothing that
        # doesn't lie on the skin.
        self.bridge = max(float(bridge), 0.0)
        self._bridged = None
        if self.bridge > 0.0:
            bridged_from = bridged(self.source, self.tris, self.bridge)
            bridged_to = bridged(self.target, self.tris, self.bridge)
            self._bridged = (bridged_from, bridged_to, _Levels(bridged_from[self._used], self.radius))
            moves = np.maximum(moves, np.linalg.norm(bridged_to - bridged_from, axis=1)[self._used])
            # Where both shapes of a vertex are unlifted the two moves agree; they differ only within reach
            # of the lifted ones, in either shape.
            lifted = self._used & ((np.abs(bridged_from - self.source).max(axis=1) > 1e-6)
                                   | (np.abs(bridged_to - self.target).max(axis=1) > 1e-6))
            self._lifted = np.concatenate([self.source[lifted], self.target[lifted]])
        self.largest = float(moves.max()) if len(moves) else 0.0
        self.steps = min(MAX_STEPS, max(1, math.ceil(self.largest / (STEP_RADII * self.radius))))

    def _step(self, points, source, target, levels, step):
        """How ``points`` move in step ``step`` of the change from shape ``source`` to ``target``,
        whose vertices ``levels`` clusters."""
        before = source + (target - source) * (step / self.steps)
        after = source + (target - source) * ((step + 1) / self.steps)
        used = self._used
        distance = _distances(points, before, self._tree_tris)
        width = np.maximum(self.radius, SPREAD * distance)
        reach = np.sqrt(distance * distance + (KERNEL_REACH * width) ** 2)
        # A point averages over one level of samples, or two neighbouring ones in between, so the
        # average changes smoothly with the width.
        octave = np.log2(width / self.radius)
        groups = np.floor(octave * BINS_PER_OCTAVE).astype(np.int64)

        co = before[used]
        area = _vertex_areas(before, self.tris)[used]
        displacement = (after - before)[used]
        samples = {}
        moved = np.zeros_like(points)
        for group in np.unique(groups):
            which = np.flatnonzero(groups == group)
            low = int(group) // BINS_PER_OCTAVE
            blend = octave[which] - low
            for level, share in ((low, 1.0 - blend), (low + 1, blend)):
                active = share > 0.0
                if not active.any():
                    continue
                if level not in samples:
                    samples[level] = levels.samples(level, co, area, displacement)
                chosen = which[active]
                moved[chosen] += share[active, None] * _average(points[chosen], reach[chosen], samples[level])
        return moved

    def apply(self, points, progress=None, rigid=None, edges=None):
        """Where ``points`` (n x 3, world space, fitted to the source shape) go with the target shape.

        ``rigid`` optionally labels points that move together without changing shape (a label per
        point, -1 for none): each such group is moved the rigid way closest to how the others would
        move it. ``edges`` (pairs of point indices: the clothing's edges), when given, turn on Keep
        Close, see `keep_close`. ``progress(fraction)`` is called after every step.
        """
        start = np.array(points, dtype=np.float64).reshape(-1, 3)
        if not len(start) or self.largest <= 0.0:
            return start
        # Points at the same spot (split seams, touching parts) go the same way: work each out once.
        points, copies = np.unique(start, axis=0, return_inverse=True)
        copies = copies.reshape(-1)
        first = points
        rounds = KEEP_ROUNDS if edges is not None else 0
        total = self.steps + rounds
        tris = self.tris
        # Clothing on the skin follows it; farther out it moves with the crease-free shapes.
        free = None
        if self._bridged is not None:
            distance = _distances(points, self.source, self._tree_tris)
            free = _blend_shares(points, _smoothstep(BRIDGE_HUG, BRIDGE_HUG + BRIDGE_BLEND, distance))
            # Points whose averages never reach a lifted vertex move alike either way: work out one move.
            spanning = np.flatnonzero(free > 0.0)
            if len(spanning):
                if len(self._lifted):
                    from mathutils.kdtree import KDTree
                    tree = KDTree(len(self._lifted))
                    for index, point in enumerate(self._lifted.tolist()):
                        tree.insert(point, index)
                    tree.balance()
                    gap = np.array([tree.find(point)[2] for point in points[spanning].tolist()])
                else:
                    gap = np.full(len(spanning), np.inf)
                farthest = distance[spanning] + self.largest
                reach = np.sqrt(farthest ** 2 + (KERNEL_REACH * np.maximum(self.radius, SPREAD * farthest)) ** 2)
                free[spanning[gap > reach + self.largest]] = 0.0
        for step in range(self.steps):
            if free is None:
                moved = self._step(points, self.source, self.target, self._levels, step)
            else:
                moved = np.zeros_like(points)
                on_skin = np.flatnonzero(free < 1.0)
                if len(on_skin):
                    moved[on_skin] = (1.0 - free[on_skin, None]) * self._step(
                        points[on_skin], self.source, self.target, self._levels, step)
                spanning = np.flatnonzero(free > 0.0)
                if len(spanning):
                    bridged_from, bridged_to, levels = self._bridged
                    moved[spanning] += free[spanning, None] * self._step(
                        points[spanning], bridged_from, bridged_to, levels, step)
            points = points + moved
            if progress is not None:
                progress((step + 1) / total)
        if edges is not None:
            loose = np.ones(len(points), dtype=bool)
            if rigid is not None:
                loose[copies[np.asarray(rigid) >= 0]] = False

            def keep_progress(done):
                if progress is not None:
                    progress((self.steps + done * rounds) / total)
            points = keep_close(first, points, copies[np.asarray(edges, dtype=np.int64).reshape(-1, 2)],
                                self.source, self.target, tris, anchors=loose, progress=keep_progress)
        result = points[copies]
        if rigid is not None:
            _keep_rigid(start, result, np.asarray(rigid))
        return result


def _blend_shares(points, shares):
    """``shares`` (one per point) averaged over the points around each across BRIDGE_BLEND_WIDTH. The
    points are first gathered into cubes of half that, so dense clothing averages quickly."""
    _keys, cube = np.unique(_cell_keys(_cells(points, 0.5 * BRIDGE_BLEND_WIDTH)), return_inverse=True)
    cube = cube.reshape(-1)
    count = np.bincount(cube).astype(np.float64)
    center = np.stack([np.bincount(cube, weights=points[:, k]) for k in range(3)], 1) / count[:, None]
    values = np.zeros_like(center)
    values[:, 0] = np.bincount(cube, weights=shares) / count
    reach = np.full(len(points), KERNEL_REACH * BRIDGE_BLEND_WIDTH)
    shares = _average(points, reach, (center, count, values))[:, 0]
    # A point barely blended takes one move alone, which saves working out the other.
    return np.where(shares < 0.01, 0.0, np.where(shares > 0.99, 1.0, shares))


def keep_close(start, moved, edges, source, target, tris, anchors=None, progress=None):
    """Bring tight clothing back to its old distance from the body after a smooth move (Keep Close).

    ``start`` are the points fitted to the body's ``source`` shape, ``moved`` where the move took them
    for its ``target`` shape (the body's triangles ``tris``), ``edges`` pairs of points joined by an
    edge. Points closer than KEEP_CLOSE_DISTANCE to the source shape (and in ``anchors``, when given)
    that ended up more than PULL_SLACK farther from the target shape, or PUSH_SLACK closer, are moved
    back along the body's normal by the difference beyond that. The corrections are spread over the
    edges, so the rest of the clothing comes along smoothly and it doesn't crease; points at the same
    spot always move alike. Returns the new positions.
    """
    from .fitting import BodySurface

    def surface(co):
        return BodySurface(co, tris, BVHTree.FromPolygons(co.tolist(), tris.tolist(), all_triangles=True))

    start = np.asarray(start, dtype=np.float64)
    moved = np.asarray(moved, dtype=np.float64)
    # Points at the same spot are one.
    _spots, weld = np.unique(start, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    count = int(weld.max()) + 1 if len(weld) else 0
    first = np.zeros(count, dtype=np.int64)
    first[weld[::-1]] = np.arange(len(weld))[::-1]
    points = moved[first]
    near = surface(np.asarray(source, dtype=np.float64)).nearest(start[first])
    height = near.distance
    close = near.found & ~near.beyond & (np.abs(height) < KEEP_CLOSE_DISTANCE)
    if anchors is not None:
        free = np.ones(count, dtype=bool)
        np.logical_and.at(free, weld, np.asarray(anchors, dtype=bool))
        close &= free
    pairs = np.sort(weld[np.asarray(edges, dtype=np.int64).reshape(-1, 2)], axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    a, b = pairs[:, 0], pairs[:, 1]
    degree = np.maximum(np.bincount(a, minlength=count) + np.bincount(b, minlength=count), 1)[:, None]
    body = surface(np.asarray(target, dtype=np.float64))
    which = np.flatnonzero(close)
    for done in range(KEEP_ROUNDS if len(which) else 0):
        near = body.nearest(points[which])
        ok = near.found & ~near.beyond
        need = np.where(ok, height[which] - near.distance, 0.0)
        need = np.where(need > 0.0, np.maximum(need - PUSH_SLACK, 0.0), np.minimum(need + PULL_SLACK, 0.0))
        own = np.zeros((count, 3))
        own[which] = need[:, None] * near.normal
        held = np.zeros(count, dtype=bool)
        held[which[ok]] = True
        spread = own
        for _ in range(KEEP_PASSES):
            average = np.stack([
                np.bincount(a, weights=spread[b, k], minlength=count) + np.bincount(b, weights=spread[a, k], minlength=count)
                for k in range(3)
            ], axis=1) / degree
            spread = np.where(held[:, None], KEEP_SHARE * own + (1.0 - KEEP_SHARE) * average, KEEP_FADE * average)
        points = points + spread
        if progress is not None:
            progress((done + 1) / KEEP_ROUNDS)
    return points[weld]


def _keep_rigid(before, after, groups):
    """Move each group of points (label >= 0 in ``groups``) from ``before`` the rigid way (rotation and
    translation) that comes closest to where they are in ``after``, in place."""
    members = np.flatnonzero(groups >= 0)
    if not len(members):
        return
    _labels, owner = np.unique(groups[members], return_inverse=True)
    owner = owner.reshape(-1)
    count = np.bincount(owner).astype(np.float64)[:, None]
    center_before = np.stack([np.bincount(owner, weights=before[members, k]) for k in range(3)], 1) / count
    center_after = np.stack([np.bincount(owner, weights=after[members, k]) for k in range(3)], 1) / count
    p = before[members] - center_before[owner]
    q = after[members] - center_after[owner]
    # Kabsch: the rotation that best turns p into q, per group.
    covariance = np.zeros((len(count), 3, 3))
    np.add.at(covariance, owner, p[:, :, None] * q[:, None, :])
    u, _s, vt = np.linalg.svd(covariance)
    turn = np.transpose(vt, (0, 2, 1)) @ np.transpose(u, (0, 2, 1))
    mirror = np.linalg.det(turn) < 0.0
    if mirror.any():
        flip = np.ones((mirror.sum(), 3))
        flip[:, 2] = -1.0
        v = np.transpose(vt[mirror], (0, 2, 1)) * flip[:, None, :]
        turn[mirror] = v @ np.transpose(u[mirror], (0, 2, 1))
    after[members] = center_after[owner] + np.einsum("nij,nj->ni", turn[owner], p)


def _components(count, a, b):
    """Group ``count`` items joined in pairs (``a[i]``, ``b[i]``): the lowest item of each one's group."""
    labels = np.arange(count)
    while len(a):
        low = np.minimum(labels[a], labels[b])
        if np.array_equal(labels[a], low) and np.array_equal(labels[b], low):
            break
        np.minimum.at(labels, a, low)
        np.minimum.at(labels, b, low)
        labels = labels[labels]
    return labels


def loose_parts(co, edges):
    """Label each vertex with its loose part (connected by ``edges``, or at the same spot as a vertex
    of the part: split seams). Returns (labels, part count)."""
    from .fitting import WELD_DISTANCE, weld_groups
    weld, count = weld_groups(co, WELD_DISTANCE)
    labels = _components(count, weld[edges[:, 0]], weld[edges[:, 1]])
    _unique, labels = np.unique(labels[weld], return_inverse=True)
    labels = labels.reshape(-1)
    return labels, int(labels.max()) + 1 if len(labels) else 0


def part_sizes(co, labels, count):
    """Length of the diagonal of each part's bounding box."""
    low = np.full((count, 3), np.inf)
    high = np.full((count, 3), -np.inf)
    np.minimum.at(low, labels, co)
    np.maximum.at(high, labels, co)
    return np.linalg.norm(high - low, axis=1)


def _distances(points, co, tris):
    """Distance from each point to the surface (``co``, ``tris`` as a list)."""
    find = BVHTree.FromPolygons(co.tolist(), tris, all_triangles=True).find_nearest
    distance = np.zeros(len(points))
    for k, point in enumerate(points.tolist()):
        hit = find(point)
        if hit[0] is not None:
            distance[k] = hit[3]
    return distance


# -----------------------------------------------------------------------------
# Matching two different bodies

# Two bodies with different vertices are matched by their UV maps: body mods made for the same skin
# textures (YAB, Rue and RueXB all use bibo's) share a UV layout, so a spot of the texture is the same
# spot of either body. A vertex is placed on the other body's triangles within this distance of its
# spot in the UV map...
UV_TOLERANCE = 1e-3
# ...unless that lands it farther than this from where it is: a part the other body lacks, whose UVs
# overlap some other part's. (Whole islands of the UV map can't be judged by how far they land: the
# chest of RueXB lands several centimetres from Rue's.)
MAX_MATCH_DISTANCE = 0.15
# Bodies are refused as not sharing a layout when fewer of the vertices than this find a spot, or when
# half of those land farther than this from where they are.
MIN_MATCHED = 0.9
MAX_MEDIAN_GAP = 0.03


def _uv_islands(tri_verts, tri_uv):
    """UV island of each triangle: triangles sharing an edge with the same UVs at both ends are in one."""
    count = len(tri_verts)
    uv = np.round(tri_uv * (1 << 20)).astype(np.int64)
    rows = []
    for k in range(3):
        a, b = k, (k + 1) % 3
        swap = tri_verts[:, a] > tri_verts[:, b]
        rows.append(np.column_stack((
            np.where(swap, tri_verts[:, b], tri_verts[:, a]), np.where(swap, tri_verts[:, a], tri_verts[:, b]),
            np.where(swap[:, None], uv[:, b], uv[:, a]), np.where(swap[:, None], uv[:, a], uv[:, b]),
        )))
    _edges, edge = np.unique(np.concatenate(rows), axis=0, return_inverse=True)
    edge = edge.reshape(-1)
    owner = np.tile(np.arange(count), 3)
    order = np.argsort(edge, kind='stable')
    shared = edge[order][1:] == edge[order][:-1]
    return _components(count, owner[order][:-1][shared], owner[order][1:][shared])


def _barycentric(points, corners):
    """Barycentric weights of 2D ``points`` (n x 2) in triangles ``corners`` (n x 3 x 2)."""
    a = corners[:, 0]
    e1, e2, p = corners[:, 1] - a, corners[:, 2] - a, points - a
    d11 = np.einsum("ij,ij->i", e1, e1)
    d12 = np.einsum("ij,ij->i", e1, e2)
    d22 = np.einsum("ij,ij->i", e2, e2)
    p1 = np.einsum("ij,ij->i", p, e1)
    p2 = np.einsum("ij,ij->i", p, e2)
    denominator = d11 * d22 - d12 * d12
    flat = np.abs(denominator) <= 1e-30
    denominator = np.where(flat, 1.0, denominator)
    # A triangle without area in the UV map gives its three corners the same weight.
    v = np.where(flat, 1.0 / 3.0, (d22 * p1 - d12 * p2) / denominator)
    w = np.where(flat, 1.0 / 3.0, (d11 * p2 - d12 * p1) / denominator)
    return np.clip(np.column_stack((1.0 - v - w, v, w)), 0.0, 1.0)


def match_by_uv(uv, co, tri_verts, tri_uv, other_co):
    """Where each vertex (its spot ``uv`` in the UV map, nan for none, and position ``co``) is on another
    mesh with triangles ``tri_verts``, their corners' UVs ``tri_uv`` and positions ``other_co``.

    Returns the triangle and its barycentric weights per vertex; triangle -1 where no triangle is
    within UV_TOLERANCE of its spot, or the spot lands more than MAX_MATCH_DISTANCE away. Where
    islands of the UV map overlap (hands and body can share texture space), the island landing closest
    to the vertex wins.
    """
    island = _uv_islands(tri_verts, tri_uv)
    flat = np.concatenate((tri_uv, np.zeros(tri_uv.shape[:2] + (1,))), axis=2).reshape(-1, 3)
    find = BVHTree.FromPolygons(flat.tolist(), np.arange(len(flat)).reshape(-1, 3).tolist(),
                                all_triangles=True).find_nearest_range
    vertex, tri, spot, gap = [], [], [], []
    for k, (u, v) in enumerate(uv.tolist()):
        if u != u:
            continue
        for location, _normal, index, distance in find((u, v, 0.0), UV_TOLERANCE):
            vertex.append(k)
            tri.append(index)
            spot.append((location[0], location[1]))
            gap.append(distance)
    triangle = np.full(len(uv), -1, dtype=np.int64)
    weights = np.zeros((len(uv), 3))
    if not vertex:
        return triangle, weights
    vertex, tri, gap = np.array(vertex), np.array(tri), np.array(gap)
    bary = _barycentric(np.array(spot), tri_uv[tri])
    miss = np.linalg.norm(np.einsum("ij,ijk->ik", bary, other_co[tri_verts[tri]]) - co[vertex], axis=1)
    # For each vertex and island the triangle closest in the UV map, then the island landing closest.
    island = island[tri]
    order = np.lexsort((gap, island, vertex))
    first = np.ones(len(order), dtype=bool)
    first[1:] = (vertex[order][1:] != vertex[order][:-1]) | (island[order][1:] != island[order][:-1])
    best = order[first]
    best = best[np.lexsort((miss[best], vertex[best]))]
    first = np.ones(len(best), dtype=bool)
    first[1:] = vertex[best][1:] != vertex[best][:-1]
    best = best[first & (miss[best] <= MAX_MATCH_DISTANCE)]
    triangle[vertex[best]] = tri[best]
    weights[vertex[best]] = bary[best]
    return triangle, weights


# -----------------------------------------------------------------------------
# Shapes of Blender meshes

def _key_positions(block):
    co = np.empty(len(block.data) * 3, dtype=np.float32)
    block.data.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def _group_weights(obj, name):
    """Weight of each vertex of ``obj`` in its vertex group ``name`` (0 where it isn't in it), or None
    when Blender applies a shape key limited to that group fully: without such a group, or when the mesh
    was never given any weights."""
    group = obj.vertex_groups.get(name)
    if group is None:
        return None
    # Read through a BMesh: reading a vertex's groups gives a mesh never given weights a weight layer,
    # which changes how Blender applies the key.
    import bmesh
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        layer = bm.verts.layers.deform.active
        if layer is None:
            return None
        index = group.index
        return np.array([vert[layer].get(index, 0.0) for vert in bm.verts], dtype=np.float64)
    finally:
        bm.free()


def contributing_keys(obj):
    """The shape keys of ``obj`` that change its rest shape right now: not the basis, not muted, not 0."""
    keys = obj.data.shape_keys
    if keys is None:
        return []
    return [block for block in keys.key_blocks
            if block != keys.reference_key and not block.mute and block.value != 0.0]


def rest_shape(obj, values=None):
    """Vertex positions (object space) of mesh ``obj`` shaped by its shape keys, before any modifier.

    Keys are mixed at their current values, like Blender does, except for those named in ``values``
    (name -> value), which are used at the value given there even if they're muted.
    """
    mesh = obj.data
    keys = mesh.shape_keys
    if keys is None:
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        return co.reshape(-1, 3).astype(np.float64)
    if not keys.use_relative:
        raise ResizeError("'{:s}' uses absolute shape keys, which Resize can't mix".format(obj.name))
    values = values or {}
    reference = keys.reference_key
    positions = {}

    def co_of(block):
        if block.name not in positions:
            positions[block.name] = _key_positions(block)
        return positions[block.name]

    result = co_of(reference).copy()
    for block in keys.key_blocks:
        if block == reference:
            continue
        value = values[block.name] if block.name in values else (0.0 if block.mute else block.value)
        if value == 0.0:
            continue
        delta = co_of(block) - co_of(block.relative_key)
        weights = _group_weights(obj, block.vertex_group) if block.vertex_group else None
        if weights is not None:
            delta *= weights[:, None]
        result += value * delta
    return result


def _transform(co, matrix):
    m = np.array(matrix, dtype=np.float64)
    return co @ m[:3, :3].T + m[:3, 3]


def _tris(mesh, attribute="vertices"):
    """The triangles of ``mesh``: their vertices, or with ``attribute`` "loops", their corners."""
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get(attribute, tris)
    return tris.reshape(-1, 3)


def _corner_uvs(mesh):
    """UV of each face corner of ``mesh`` in its active UV map (None without one)."""
    layer = mesh.uv_layers.active
    if layer is None:
        return None
    uv = np.empty(len(mesh.loops) * 2)
    layer.data.foreach_get("uv", uv)
    return uv.reshape(-1, 2)


def _vertex_uvs(mesh, corners):
    """A spot in the UV map per vertex, from one of its corners (nan for vertices of no face)."""
    vertex = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", vertex)
    uv = np.full((len(mesh.vertices), 2), np.nan)
    uv[vertex] = corners
    return uv


def _edge_set(mesh):
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", edges)
    edges = np.sort(edges.reshape(-1, 2), axis=1)
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))]


def _same_vertices(mesh, other):
    """Whether two meshes have the same vertices, joined the same way (copies of one mesh)."""
    if mesh == other:
        return True
    if len(mesh.vertices) != len(other.vertices) or len(mesh.edges) != len(other.edges):
        return False
    return np.array_equal(_edge_set(mesh), _edge_set(other))


def matches_by_uv(body, to_body):
    """Whether Resize matches ``body`` and ``to_body`` by their UV maps: they're different meshes."""
    return to_body is not None and not _same_vertices(body.data, to_body.data)


def _key_name(obj, name):
    """``name`` when ``obj`` has such a shape key (or ``name`` is empty), else None."""
    if not name:
        return ""
    keys = obj.data.shape_keys
    return name if keys is not None and keys.key_blocks.get(name) is not None else None


def check_shapes(body, from_key, to_body, to_key):
    """Why the From and To shapes can't be used, or None when they can. ``to_body`` None means ``body``."""
    if body is None:
        return "Pick the From Body"
    if body.type != 'MESH':
        return "The From Body must be a mesh"
    other = to_body if to_body is not None else body
    if other.type != 'MESH':
        return "The To Body must be a mesh"
    if _key_name(body, from_key) is None:
        return "'{:s}' has no shape key '{:s}'".format(body.name, from_key)
    if _key_name(other, to_key) is None:
        return "'{:s}' has no shape key '{:s}'".format(other.name, to_key)
    if len(other.data.vertices) != len(body.data.vertices):
        # A different body: it's matched by the UV maps (the same vertex count is taken for a copy).
        for obj, match in ((body, other), (other, body)):
            if obj.data.uv_layers.active is None:
                return "'{:s}' has no UV map to match it to '{:s}' by".format(obj.name, match.name)
    for obj in {body, other}:
        keys = obj.data.shape_keys
        if keys is not None and not keys.use_relative:
            return "'{:s}' uses absolute shape keys, which Resize can't mix".format(obj.name)
    if other == body:
        keys = body.data.shape_keys
        if keys is None:
            return "'{:s}' has no shape keys: pick a To Body with the other shape".format(body.name)
        # The basis adds nothing, like no key at all.
        reference = keys.reference_key.name
        if (from_key if from_key != reference else "") == (to_key if to_key != reference else ""):
            return "From and To are the same shape"
    return None


def body_shapes(body, from_key, to_body, to_key):
    """The From and To shapes of the body (world space, on the vertices of ``body``) and its triangles.

    From is ``body`` with its shape key ``from_key`` fully on, To is ``to_body`` (``body`` when None)
    with ``to_key`` fully on; empty names add no key. The other shape keys stay at their current
    values, except that when both are the same object, the key picked for one shape is off in the other.
    The To shape is taken in its own object space and placed like ``body``, so the To body can stand
    anywhere. A To body with other vertices is matched to ``body`` by their UV maps, and only the
    vertices of ``body`` found on it are returned.
    """
    problem = check_shapes(body, from_key, to_body, to_key)
    if problem is not None:
        raise ResizeError(problem)
    other = to_body if to_body is not None else body
    source_values, target_values = {}, {}
    if from_key:
        source_values[from_key] = 1.0
    if to_key:
        target_values[to_key] = 1.0
    if other == body:
        if from_key and from_key not in target_values:
            target_values[from_key] = 0.0
        if to_key and to_key not in source_values:
            source_values[to_key] = 0.0
    source = rest_shape(body, source_values)
    target = rest_shape(other, target_values)
    tris = _tris(body.data)
    if matches_by_uv(body, to_body):
        source, target, tris = _match_bodies(body, source, other, target, tris)
    return _transform(source, body.matrix_world), _transform(target, body.matrix_world), tris


def _match_bodies(body, source, other, target, tris):
    """The shape ``target`` of ``other`` on the vertices of ``body`` (shape ``source``, triangles
    ``tris``), matched by their UV maps: (the vertices of ``body`` found on ``other``, in ``source``
    and in ``target``, and the triangles of ``body`` among them)."""
    corners = _corner_uvs(body.data)
    other_corners = _corner_uvs(other.data)
    for obj, uv, match in ((body, corners, other), (other, other_corners, body)):
        if uv is None:
            raise ResizeError("'{:s}' has no UV map to match it to '{:s}' by".format(obj.name, match.name))
    uv = _vertex_uvs(body.data, corners)
    other_tris = _tris(other.data)
    triangle, weights = match_by_uv(uv, source, other_tris, other_corners[_tris(other.data, "loops")], target)
    found = triangle >= 0
    share = found.sum() / max(int((~np.isnan(uv[:, 0])).sum()), 1)
    if share < MIN_MATCHED:
        raise ResizeError("Only {:.0f}% of '{:s}' could be matched to '{:s}' by their UV maps: the bodies need the "
                          "same UV layout".format(100.0 * share, body.name, other.name))
    placed = np.einsum("ij,ijk->ik", weights[found], target[other_tris[triangle[found]]])
    gap = float(np.median(np.linalg.norm(placed - source[found], axis=1)))
    if gap > MAX_MEDIAN_GAP:
        raise ResizeError("'{:s}' and '{:s}' don't line up by their UV maps (typically {:.0f} cm apart): the bodies "
                          "need the same UV layout".format(body.name, other.name, 100.0 * gap))
    index = np.cumsum(found) - 1
    return source[found], placed, index[tris[found[tris].all(axis=1)]]


def shape_label(obj, key, fallback_obj):
    """How to call a shape: its key, else its object when that isn't ``fallback_obj``, else ''."""
    if key:
        keys = obj.data.shape_keys
        if keys is None or key != keys.reference_key.name:
            return key
    return obj.name if obj != fallback_obj else ""


def garment_shape(obj):
    """World space positions of ``obj`` as its shape keys show it (before modifiers)."""
    return _transform(rest_shape(obj), obj.matrix_world)


def add_resized_key(obj, name, world_co):
    """Add a shape key named ``name`` (Blender makes it unique) holding ``world_co`` to ``obj``, turn it
    fully on and make it the active key. The keys that shaped the mesh before are turned off, so it
    shows the new shape alone; those that can't be (driven ones) are returned."""
    before = contributing_keys(obj)
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    block = obj.shape_key_add(name=name, from_mix=False)
    inverse = np.linalg.inv(np.array(obj.matrix_world, dtype=np.float64))
    block.data.foreach_set("co", _transform(world_co, inverse).astype(np.float32).ravel())
    block.relative_key = obj.data.shape_keys.reference_key
    block.value = 1.0
    stuck = []
    for other in before:
        if _driven(obj.data.shape_keys, other, "value"):
            stuck.append(other.name)
        else:
            other.value = 0.0
    obj.active_shape_key_index = list(obj.data.shape_keys.key_blocks).index(block)
    obj.data.update()
    return block, stuck


def _driven(keys, block, prop):
    animation = keys.animation_data
    if animation is None:
        return False
    path = 'key_blocks["{:s}"].{:s}'.format(block.name.replace('"', '\\"'), prop)
    return any(curve.data_path == path for curve in animation.drivers)


def shows_to_shape(body, from_key, to_body, to_key):
    """Whether the To body (``to_body``, else ``body``) shows the To shape right now: ``to_key`` on and,
    when it's ``body`` itself, ``from_key`` off. (Resize doesn't switch it: Blender wouldn't undo that.)"""
    other = to_body if to_body is not None else body
    keys = other.data.shape_keys
    if keys is None:
        return True
    wanted = ((to_key, 1.0),) if other != body else ((from_key, 0.0), (to_key, 1.0))
    for name, value in wanted:
        block = keys.key_blocks.get(name) if name else None
        if block is None or block == keys.reference_key:
            continue
        if (0.0 if block.mute else block.value) != value:
            return False
    return True
