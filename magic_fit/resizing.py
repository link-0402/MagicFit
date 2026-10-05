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
Bodies of any kind (an outfit's own body from another game, lined up on the Mannequin with it) are
matched by shape instead: each vertex goes to the closest spot of the other body, where that faces the
same way, and the parts that can't be matched like that (hands, feet, a head) follow the rest.

A Yet Another Devkit body can also be resized to its other sizes (chest, legs, Rue, Buff, Lavabod) without
switching the devkit: its settings drive the body's shape keys, and those drivers are worked out again with
the sizes picked (`devkit_values`), which gives the To shape the devkit would show.

Like `painting`, the engine (`ShapeChange`) works on world space arrays and doesn't need the 3D view.
"""

import ast
import functools
import math
import operator
import re

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

    ``to_body`` (positions and triangles) is the To body itself when ``target`` was matched onto it by
    shape, and ``known`` which vertices of ``target`` lie on it (see `match_by_shape`): Keep Close then
    brings clothing back to its distance from the To body itself, over those vertices only.
    """

    def __init__(self, source, target, tris, radius=DEFAULT_RADIUS, bridge=0.0, to_body=None, known=None):
        self.source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
        self.target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
        self.tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        self.to_body, self.known = to_body, known
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
            to_co, to_tris = self.to_body if self.to_body is not None else (self.target, None)
            points = keep_close(first, points, copies[np.asarray(edges, dtype=np.int64).reshape(-1, 2)],
                                self.source, to_co, tris, anchors=loose, progress=keep_progress,
                                target_tris=to_tris, known=self.known)
        result = points[copies]
        if rigid is not None:
            keep_rigid(start, result, np.asarray(rigid))
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


def keep_close(start, moved, edges, source, target, tris, anchors=None, progress=None, target_tris=None,
               known=None):
    """Bring tight clothing back to its old distance from the body after a smooth move (Keep Close).

    ``start`` are the points fitted to the body's ``source`` shape, ``moved`` where the move took them
    for its ``target`` shape (the body's triangles ``tris``), ``edges`` pairs of points joined by an
    edge. Points closer than KEEP_CLOSE_DISTANCE to the source shape (and in ``anchors``, when given)
    that ended up more than PULL_SLACK farther from the target shape, or PUSH_SLACK closer, are moved
    back along the body's normal by the difference beyond that. The corrections are spread over the
    edges, so the rest of the clothing comes along smoothly and it doesn't crease; points at the same
    spot always move alike. Returns the new positions.

    With ``target_tris``, ``target`` is another mesh (the To body a body of other vertices was matched
    to by shape), and with ``known`` (per vertex of ``source``), only points over source triangles most
    of whose corners are known count (where the matched shape lies on that mesh: over a heel left to
    follow the leg, Keep Close would press a boot onto the To body's flat foot).
    """
    from .fitting import BodySurface

    def surface(co, faces=tris):
        return BodySurface(co, faces, BVHTree.FromPolygons(co.tolist(), faces.tolist(), all_triangles=True))

    start = np.asarray(start, dtype=np.float64)
    moved = np.asarray(moved, dtype=np.float64)
    # Points at the same spot are one.
    _spots, weld = np.unique(start, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    count = int(weld.max()) + 1 if len(weld) else 0
    first = np.zeros(count, dtype=np.int64)
    first[weld[::-1]] = np.arange(len(weld))[::-1]
    points = moved[first]
    source_surface = surface(np.asarray(source, dtype=np.float64))
    near = source_surface.nearest(start[first])
    height = near.distance
    close = near.found & ~near.beyond & (np.abs(height) < KEEP_CLOSE_DISTANCE)
    if known is not None:
        welded = np.zeros(len(source_surface.co), dtype=bool)
        welded[source_surface._weld] = np.asarray(known, dtype=bool)
        close &= welded[near.corners].sum(axis=1) >= 2
    if anchors is not None:
        free = np.ones(count, dtype=bool)
        np.logical_and.at(free, weld, np.asarray(anchors, dtype=bool))
        close &= free
    pairs = np.sort(weld[np.asarray(edges, dtype=np.int64).reshape(-1, 2)], axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    a, b = pairs[:, 0], pairs[:, 1]
    degree = np.maximum(np.bincount(a, minlength=count) + np.bincount(b, minlength=count), 1)[:, None]
    body = surface(np.asarray(target, dtype=np.float64),
                   tris if target_tris is None else np.asarray(target_tris, dtype=np.int64).reshape(-1, 3))
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


def keep_rigid(before, after, groups):
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


def loose_parts(co, edges):
    """Label each vertex with its loose part (connected by ``edges``, or at the same spot as a vertex
    of the part: split seams). Returns (labels, part count)."""
    from .fitting import WELD_DISTANCE, _connected, weld_groups
    weld, count = weld_groups(co, WELD_DISTANCE)
    labels, parts = _connected(count, np.stack((weld[edges[:, 0]], weld[edges[:, 1]]), axis=1))
    return labels[weld], parts


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
    from .fitting import _connected
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
    return _connected(count, np.stack((owner[order][:-1][shared], owner[order][1:][shared]), axis=1))[0]


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


# Bodies of any kind are matched by shape, lying on each other (an outfit's own body from another game and
# the Mannequin, after Line Up): each vertex of one goes to the closest spot of the other no farther than
# this...
SHAPE_REACH = 0.10
# ...that faces within about 60 degrees of the way it faces...
SHAPE_FACING = 0.5
# ...and that it lies off mostly along that spot's normal (as Fit Shape's matches do), unless closer than
# this (bodies lined up within a few millimetres).
SHAPE_ALONG = 0.6
SHAPE_SLACK = 0.003
# Parts never matched by shape (fingers count with the hands): toes, heels and fingers find the wrong
# spots, many on one (heeled feet on the Mannequin's flat ones), and FFXIV bodies have no head. They follow
# the parts around them.
SHAPE_SKIPPED = ("head", "hand", "foot")
# The moves found are smoothed over the body across about this distance (Resize averages over a centimetre
# anyway), and fade out over this distance from what matched into what didn't.
SHAPE_SMOOTH = 0.01
SHAPE_FADE = 0.15
# Then the vertices of the smoothed shape that match again are put onto the other body where it's closer
# than this, so the matched shape is that body's surface itself, not a smoothed copy of it.
SHAPE_PROJECT = 0.02
# A body is refused when less than this share of its area lies over the other at all (not past its open
# edges, not in a skipped part); MIN_MATCHED and MAX_MEDIAN_GAP count as for UV maps, over that area.
SHAPE_MIN_OVER = 0.3
# Where the bodies differ by more than this (for the farthest 5 % of what matched), clothing tied to the
# skin's landmarks (the underbust, a piercing) slides and can crease: Resize warns.
SHAPE_DIFFERENT = 0.03


class ShapeMatch:
    """Where each vertex of a body goes on another, matched by shape (see `match_by_shape`).

    ``target`` is its position on the other body (n x 3). ``matched`` marks the vertices that match it
    (those closer than SHAPE_PROJECT were put onto it), ``skipped`` those in parts never matched by shape,
    ``beyond`` those past its open edges (it has nothing there) that don't match, and ``known`` those
    neither: the others follow them. ``over`` is the share of
    the body's area known, ``matched_share`` the share of that area that matched, ``folded`` the share of
    the known triangles turned over in the matched shape (pressed many on one), ``apart`` the median
    distance between the bodies over the known vertices, and ``spread`` the 95th percentile of it over the
    matched ones.
    """

    def __init__(self):
        self.target = self.matched = self.skipped = self.beyond = self.known = None
        self.over = self.matched_share = self.folded = self.apart = self.spread = 0.0


def skipped_parts():
    """Indices into lineup.PARTS of the parts never matched by shape (SHAPE_SKIPPED)."""
    from . import lineup
    chosen = [index for index, part in enumerate(lineup.PARTS)
              if (part.rsplit("_", 1)[0] if part.endswith(("_l", "_r")) else part) in SHAPE_SKIPPED
              or part in lineup.DIGIT_PARTS]
    return np.array(chosen, dtype=np.int64)


def _screened_fill(values, weights, edges, count, stiffness, hold, fixed=None, start=None, iterations=2000,
                   tolerance=1e-7):
    """Solve (W + stiffness L + hold I) x = W values over a graph of ``count`` vertices (``edges``: pairs; L
    its uniform Laplacian, W the diagonal of ``weights``), each column by conjugate gradients with a Jacobi
    preconditioner, from ``start`` (else ``values`` where weighted, 0 elsewhere). That smooths ``values``
    where they're weighted, and fills in the rest smoothly, fading to 0 over about sqrt(stiffness / hold)
    edges from the weighted ones. Where ``fixed`` (a mask), x is ``values`` exactly, and the rest is solved
    around it."""
    a, b = edges[:, 0], edges[:, 1]
    values = np.asarray(values, dtype=np.float64).reshape(count, -1)
    weights = np.asarray(weights, dtype=np.float64)
    free = np.ones((count, 1)) if fixed is None else (~np.asarray(fixed, dtype=bool)).astype(np.float64)[:, None]
    degree = (np.bincount(a, minlength=count) + np.bincount(b, minlength=count)).astype(np.float64)
    diagonal = (weights + stiffness * degree + hold)[:, None]

    def around(x):
        return np.stack([np.bincount(a, weights=x[b, k], minlength=count) + np.bincount(b, weights=x[a, k], minlength=count)
                         for k in range(x.shape[1])], axis=1)

    def times(x):
        # Over the free vertices only: the fixed ones' values are in the right-hand side.
        return free * (diagonal * x - stiffness * around(x))

    held = (1.0 - free) * values
    rhs = free * (weights[:, None] * values + stiffness * around(held))
    x = free * (np.where(weights[:, None] > 0.0, values, 0.0) if start is None else np.reshape(start, (count, -1)))
    r = rhs - times(x)
    z = r / diagonal
    p = z.copy()
    rz = np.einsum("ij,ij->j", r, z)
    norm = max(float(np.linalg.norm(rhs)), 1e-30)
    for _ in range(iterations):
        if float(np.linalg.norm(r)) <= tolerance * norm:
            break
        q = times(p)
        step = rz / np.maximum(np.einsum("ij,ij->j", p, q), 1e-300)
        x += step * p
        r -= step * q
        z = r / diagonal
        rz_new = np.einsum("ij,ij->j", r, z)
        p = z + (rz_new / np.maximum(rz, 1e-300)) * p
        rz = rz_new
    return x + held


def match_by_shape(co, tris, to_co, to_tris, labels=None, to_labels=None):
    """Where each vertex of a body (``co``, ``tris``) goes on another body of any vertices (``to_co``,
    ``to_tris``) lying on it, matched by shape.

    Each vertex takes the closest spot of the other body within SHAPE_REACH that faces about the same way
    and that it lies off along that spot's normal. Those moves are smoothed over the body and filled in
    where nothing matched (fading out over SHAPE_FADE), and the vertices that match again from there are
    put onto the other body. ``labels`` (per vertex, its part in lineup.PARTS, -1 for none) mark the parts
    never matched (SHAPE_SKIPPED); without them, ``to_labels`` (the same for the other body) mark the
    vertices whose closest spot is in such a part. Returns a ShapeMatch.
    """
    from .fitting import BodySurface

    def surface(points, faces):
        return BodySurface(points, faces, BVHTree.FromPolygons(points.tolist(), faces.tolist(), all_triangles=True))

    body = surface(np.asarray(co, dtype=np.float64), np.asarray(tris, dtype=np.int64).reshape(-1, 3))
    other = surface(np.asarray(to_co, dtype=np.float64), np.asarray(to_tris, dtype=np.int64).reshape(-1, 3))
    # Vertices at the same spot (split seams) are one.
    weld, count, corners = body._weld, len(body.co), body.corners
    skip = skipped_parts()
    skipped = np.zeros(count, dtype=bool)
    if labels is not None:
        np.logical_or.at(skipped, weld, np.isin(labels, skip))
    near = other.nearest(body.co)
    if labels is None and to_labels is not None:
        skipping = np.zeros(len(other.co), dtype=bool)
        np.logical_or.at(skipping, other._weld, np.isin(to_labels, skip))
        closest = near.corners[np.arange(count), np.argmax(near.bary, axis=1)]
        skipped = near.found & skipping[closest]

    def accept(points, facing, near):
        """Which of ``points`` (facing ``facing``) match their closest spots ``near``, and how far those are."""
        offset = near.location - points
        length = np.linalg.norm(offset, axis=1)
        ok = near.found & ~near.beyond & ~skipped & (length <= SHAPE_REACH)
        ok &= np.einsum("ij,ij->i", facing, near.normal) > SHAPE_FACING
        ok &= np.abs(np.einsum("ij,ij->i", offset, near.normal)) >= SHAPE_ALONG * length - SHAPE_SLACK
        return ok, length

    # Each triangle turned the way the body's outward normals face it, for the normals of the moved shape.
    a, b, c = (body.co[corners[:, k]] for k in range(3))
    before = np.cross(b - a, c - a)
    turn = np.where(np.einsum("ij,ij->i", before, body.tri_normals) < 0.0, -1.0, 1.0)

    def normals(points):
        a, b, c = (points[corners[:, k]] for k in range(3))
        face = np.cross(b - a, c - a) * turn[:, None]
        summed = np.stack([np.bincount(corners.reshape(-1), np.repeat(face[:, k], 3), minlength=count)
                           for k in range(3)], axis=1)
        return summed / np.maximum(np.linalg.norm(summed, axis=1, keepdims=True), 1e-12)

    found, gap = accept(body.co, body.normals, near)
    beyond = near.beyond.copy()
    edge = max(body.edge_length, 1e-6)
    stiffness = (SHAPE_SMOOTH / edge) ** 2
    hold = stiffness * (edge / SHAPE_FADE) ** 2
    moved = body.co + _screened_fill(near.location - body.co, found, body.edges, count, stiffness, hold)
    again = other.nearest(moved)
    matched, off = accept(moved, normals(moved), again)
    put = matched & (off < SHAPE_PROJECT)
    moved[put] = again.location[put]
    if put.any() and not put.all():
        # The rest follows what was put onto the other body from where it was put, not from the smoothed
        # moves, which would leave a step there: a hand would end up off the wrist it hangs from.
        moves = moved - body.co
        moved = body.co + _screened_fill(moves, np.zeros(count), body.edges, count, stiffness, hold, fixed=put,
                                         start=moves)
    beyond &= ~matched
    known = ~skipped & ~beyond

    result = ShapeMatch()
    area = np.linalg.norm(body.area_normals, axis=1)
    known_area = float(area[known].sum())
    result.over = known_area / max(float(area.sum()), 1e-30)
    result.matched_share = float(area[known & matched].sum()) / max(known_area, 1e-30)
    result.apart = float(np.median(gap[known])) if known.any() else math.inf
    result.spread = float(np.percentile(gap[found], 95)) if found.any() else 0.0
    ta, tb, tc = (moved[corners[:, k]] for k in range(3))
    counted = known[corners].all(axis=1) & (np.linalg.norm(before, axis=1) > 1e-14)
    turned = counted & (np.einsum("ij,ij->i", before, np.cross(tb - ta, tc - ta)) < 0.0)
    result.folded = float(turned.sum()) / max(int(counted.sum()), 1)
    result.target = moved[weld]
    result.matched, result.skipped, result.beyond, result.known = matched[weld], skipped[weld], beyond[weld], known[weld]
    return result


def check_shape_match(match, name, other_name):
    """Why a match by shape (``match``, a ShapeMatch) of the body ``name`` to ``other_name`` can't be used (a
    message), or None."""
    if match.over < SHAPE_MIN_OVER:
        return "Only {:.0f}% of '{:s}' lies over '{:s}': line them up on each other first".format(
            100.0 * match.over, name, other_name)
    if match.apart > MAX_MEDIAN_GAP:
        return "'{:s}' and '{:s}' don't lie on each other (typically {:.0f} cm apart): line them up first".format(
            name, other_name, 100.0 * match.apart)
    if match.matched_share < MIN_MATCHED:
        return "Only {:.0f}% of '{:s}' could be matched to '{:s}' by shape: the bodies are too different".format(
            100.0 * match.matched_share, name, other_name)
    return None


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


def matching(body, to_body, match='AUTO'):
    """How Resize matches the From body ``body`` with the To body (``to_body``, else ``body`` itself), with
    Match By ``match`` ('AUTO', 'VERTICES', 'UV' or 'SHAPE'): one body's shapes and, for 'AUTO', copies of
    one mesh go vertex by vertex ('VERTICES'); other bodies as ``match`` says, 'AUTO' meaning by UV map where
    they share a layout, else by shape."""
    if to_body is None or to_body == body:
        return 'VERTICES'
    if match == 'AUTO' and _same_vertices(body.data, to_body.data):
        return 'VERTICES'
    return match


def part_labels(obj):
    """Body part (index into lineup.PARTS) of each vertex of mesh ``obj``, -1 for none, by its weights in
    the deform bones of its armature, else in its vertex groups named after body parts. None when they
    don't tell its limbs."""
    from . import lineup
    parts = None
    armature = lineup.armature_of(obj)
    if armature is not None:
        _parts, deform, _joints = lineup.armature_rig(armature)
        parts = {group.name: deform[group.name] for group in obj.vertex_groups if group.name in deform}
        if not lineup.has_half(parts):
            parts = None
    if parts is None:
        # Groups whose names say nothing about a body part (a devkit's 'Buff' or 'Yiggle' masks) would join
        # the part they share the most weight with, and outweigh its bones.
        names = [group.name for group in obj.vertex_groups if lineup.classify(group.name) != (None, None)]
        parts, _named = lineup.group_parts_by_name(names, lineup.mesh_weights(obj, names))
        if not lineup.has_half(parts):
            return None
    part_weights = lineup.mesh_weights(obj, list(parts)).by_part(parts)
    return np.where(part_weights.sum(axis=1) > 0.0, np.argmax(part_weights, axis=1), -1)


def _key_name(obj, name):
    """``name`` when ``obj`` has such a shape key (or ``name`` is empty), else None."""
    if not name:
        return ""
    keys = obj.data.shape_keys
    return name if keys is not None and keys.key_blocks.get(name) is not None else None


def check_shapes(body, from_key, to_body, to_key, match='AUTO', to_values=None):
    """Why the From and To shapes can't be used, or None when they can. ``to_body`` None means ``body``;
    ``match`` is how different bodies are matched (see `matching`); ``to_values`` are key values of the To
    shape, as for `body_shapes` (or just true when there are some)."""
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
    if other != body:
        if match == 'VERTICES' and len(other.data.vertices) != len(body.data.vertices):
            return "'{:s}' and '{:s}' have different vertices: match them by UV map or shape".format(
                body.name, other.name)
        if match == 'UV':
            for obj, match_to in ((body, other), (other, body)):
                if obj.data.uv_layers.active is None:
                    return "'{:s}' has no UV map to match it to '{:s}' by".format(obj.name, match_to.name)
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
        if (from_key if from_key != reference else "") == (to_key if to_key != reference else "") and not to_values:
            return "From and To are the same shape"
    return None


class BodyShapes:
    """The From and To shapes Resize moves meshes between, in world space: ``source`` and ``target`` on the
    same vertices, ``tris`` their triangles, and ``how`` the bodies were matched ('VERTICES', 'UV' or
    'SHAPE'). A match by shape also keeps the To body itself (``to_co``, ``to_tris``) and the ShapeMatch
    (``match``), whose known vertices lie on it."""

    def __init__(self, source, target, tris, how):
        self.source, self.target, self.tris, self.how = source, target, tris, how
        self.to_co = self.to_tris = self.match = None

    def change(self, radius=DEFAULT_RADIUS, bridge=0.0):
        """The ShapeChange between the two shapes."""
        if self.match is None:
            return ShapeChange(self.source, self.target, self.tris, radius=radius, bridge=bridge)
        return ShapeChange(self.source, self.target, self.tris, radius=radius, bridge=bridge,
                           to_body=(self.to_co, self.to_tris), known=self.match.known)


def body_shapes(body, from_key, to_body, to_key, match='AUTO', to_values=None):
    """The From and To shapes of the body (a BodyShapes, on the vertices of ``body``).

    From is ``body`` with its shape key ``from_key`` fully on, To is ``to_body`` (``body`` when None)
    with ``to_key`` fully on; empty names add no key. ``to_values`` (key name -> value, like
    `devkit_values` gives them) set more keys of the To shape. The other shape keys stay at their current
    values, except that when both are the same object, the key picked for one shape is off in the other.
    A To body with other vertices is matched to ``body`` as ``match`` says (see `matching`): by their UV
    maps, keeping only the vertices of ``body`` found on it, or by shape. Matched by vertices or UV map,
    the To shape is taken in its own object space and placed like ``body``, so the To body can stand
    anywhere; matched by shape, the bodies must lie on each other.
    """
    problem = check_shapes(body, from_key, to_body, to_key, match, to_values)
    if problem is not None:
        raise ResizeError(problem)
    other = to_body if to_body is not None else body
    source_values, target_values = {}, dict(to_values or {})
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
    way = matching(body, to_body, match)
    if way in ('UV', 'AUTO'):
        try:
            matched = _match_bodies(body, source, other, target, tris)
        except ResizeError:
            if way == 'UV':
                raise
        else:
            return BodyShapes(_transform(matched[0], body.matrix_world), _transform(matched[1], body.matrix_world),
                              matched[2], 'UV')
    if way in ('SHAPE', 'AUTO'):
        return _shape_bodies(body, source, other, target, tris)
    return BodyShapes(_transform(source, body.matrix_world), _transform(target, body.matrix_world), tris, 'VERTICES')


def _shape_bodies(body, source, other, target, tris):
    """The shape ``target`` of ``other`` matched by shape to ``body`` (shape ``source``, triangles ``tris``),
    both where they stand: a BodyShapes."""
    source = _transform(source, body.matrix_world)
    to_co = _transform(target, other.matrix_world)
    to_tris = _tris(other.data)
    labels = part_labels(body)
    match = match_by_shape(source, tris, to_co, to_tris, labels, part_labels(other) if labels is None else None)
    problem = check_shape_match(match, body.name, other.name)
    if problem is not None:
        raise ResizeError(problem)
    result = BodyShapes(source, match.target, tris, 'SHAPE')
    result.to_co, result.to_tris, result.match = to_co, to_tris, match
    return result


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


# -----------------------------------------------------------------------------
# Devkit sizes

# Yet Another Devkit drives its bodies' shape keys from its settings, kept in Scene.ya_devkit_props in a
# group per body (mannequin_state for the Mannequin): the chest size turns on the MEDIUM, SMALL or MASC key
# and mutes the other sizes' sub-keys, Rue turns on the Rue key and, with Buff, the Rue/Buff correction,
# and so on. Resize goes to other sizes without switching the devkit: it works those drivers out again with
# the sizes picked.
DEVKIT_PATH = re.compile(r"ya_devkit_props\.(\w+)\.(\w+)")
# The settings it can pick, as the drivers read them: the chest and leg sizes by number (in the orders
# below), the switches 0 or 1.
DEVKIT_SIZES = ("chest_size", "leg_size", "rue", "buff", "lavabod")
DEVKIT_CHEST = ("Large", "Medium", "Small", "Masc")
# With Lavabod, the chest sizes are these instead.
DEVKIT_LAVA_CHEST = ("Omoi", "Teardrop", "Cupcake", "Masc")
DEVKIT_LEGS = ("Melon", "Skull", "Yanilla", "Masc", "Lavabod", "Mini")
# Alt Hips (the devkit turns it off with the other leg sizes) only goes with these first ones.
DEVKIT_HIP_LEGS = 3
# Switching Lavabod, the devkit gives the chest's sub-keys ('- Squeeze', '-- Sag'...) the values they had
# the last time it was switched the other way, kept in these lists of the settings' group; the first time
# Lavabod comes on, it takes this key off.
DEVKIT_KEPT_KEYS = {1: "lava_keys", 0: "yab_keys"}
DEVKIT_LAVA_OFF = "- Squeeze"

_KEY_PATH = re.compile(r'key_blocks\["((?:[^"\\]|\\.)*)"\]\.(value|mute)')
_FUNCTIONS = {"min": min, "max": max, "abs": abs, "round": round, "int": int, "float": float, "bool": bool,
              "floor": math.floor, "ceil": math.ceil, "sqrt": math.sqrt}
_CONSTANTS = {"True": True, "False": False, "pi": math.pi}
_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
               ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_COMPARISONS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
                ast.Gt: operator.gt, ast.GtE: operator.ge}


@functools.lru_cache(maxsize=512)
def _parsed(expression):
    return ast.parse(expression.strip(), mode="eval").body


def _expression_value(node, names):
    """The value of a driver's expression (``node``, parsed) with its variables at ``names``. Only
    arithmetic, comparisons, logic and a few functions, so nothing in a file can run code here."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError("it reads '{:s}'".format(node.id))
    if isinstance(node, ast.BoolOp):
        # As in Python: the first value that decides, else the last.
        for part in node.values:
            value = _expression_value(part, names)
            if bool(value) == isinstance(node.op, ast.Or):
                return value
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
        value = _expression_value(node.operand, names)
        return not value if isinstance(node.op, ast.Not) else -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        return _ARITHMETIC[type(node.op)](_expression_value(node.left, names), _expression_value(node.right, names))
    if isinstance(node, ast.Compare) and all(type(op) in _COMPARISONS for op in node.ops):
        left = _expression_value(node.left, names)
        for op, part in zip(node.ops, node.comparators):
            right = _expression_value(part, names)
            if not _COMPARISONS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _expression_value(node.body if _expression_value(node.test, names) else node.orelse, names)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
            and not node.keywords):
        return _FUNCTIONS[node.func.id](*(_expression_value(part, names) for part in node.args))
    raise ValueError("Resize can't work out '{:s}' in it".format(ast.unparse(node)))


def _unescape(name):
    return re.sub(r"\\(.)", r"\1", name)


def _key_drivers(keys):
    """The drivers of shape keys ``keys`` (a Key) that set a key's value or mute: {(key name, 'value' or
    'mute'): F-curve}."""
    animation = keys.animation_data
    if animation is None:
        return {}
    result = {}
    for curve in animation.drivers:
        found = _KEY_PATH.fullmatch(curve.data_path)
        if found is not None and not curve.mute:
            result[(_unescape(found.group(1)), found.group(2))] = curve
    return result


def _devkit_setting(var):
    """(scene, group, name) of the devkit setting driver variable ``var`` reads, or None."""
    if var.type != 'SINGLE_PROP':
        return None
    target = var.targets[0]
    if target.id_type != 'SCENE' or target.id is None:
        return None
    found = DEVKIT_PATH.fullmatch(target.data_path)
    return None if found is None else (target.id, found.group(1), found.group(2))


def _devkit_groups(keys):
    """The devkit settings the drivers of ``keys`` read: {name: (scene, group)}."""
    result = {}
    for curve in _key_drivers(keys).values():
        for var in curve.driver.variables:
            setting = _devkit_setting(var)
            if setting is not None:
                result.setdefault(setting[2], setting[:2])
    return result


def _mesh_keys(obj):
    return obj.data.shape_keys if obj is not None and obj.type == 'MESH' else None


def devkit_settings(obj):
    """The settings among DEVKIT_SIZES of Yet Another Devkit that drive shape keys of ``obj`` (a set)."""
    keys = _mesh_keys(obj)
    return set() if keys is None else set(_devkit_groups(keys)) & set(DEVKIT_SIZES)


def _property_value(owner, path):
    """The number a driver reads from property ``path`` of ``owner`` (an enum by its item's number), or None
    when there's no such property."""
    try:
        value = owner.path_resolve(path)
    except ValueError:
        return None
    if isinstance(value, (bool, int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    holder, _dot, name = path.rpartition(".")
    try:
        prop = (owner.path_resolve(holder) if holder else owner).bl_rna.properties[name]
    except (ValueError, KeyError):
        return None
    if prop.type != 'ENUM':
        return None
    item = prop.enum_items.get(value)
    if item is not None:
        return float(item.value)
    # Items made by a function aren't listed; the devkit's are named after their numbers.
    return float(value) if value.isdigit() else None


def _devkit_now(groups, name):
    scene, group = groups[name]
    return _property_value(scene, "ya_devkit_props.{:s}.{:s}".format(group, name))


def devkit_state(obj):
    """How the settings among DEVKIT_SIZES of Yet Another Devkit that drive shape keys of ``obj`` are set
    now: {name: number}, as the drivers read them. Those that can't be read (the devkit's script isn't
    running) are left out."""
    keys = _mesh_keys(obj)
    if keys is None:
        return {}
    groups = _devkit_groups(keys)
    result = {}
    for name in DEVKIT_SIZES:
        if name in groups:
            value = _devkit_now(groups, name)
            if value is not None:
                result[name] = int(value)
    return result


def devkit_rules(sizes):
    """``sizes`` (see `devkit_state`) as the devkit would set them: Lavabod has no Masc chest (it takes
    Small instead)."""
    sizes = dict(sizes)
    if sizes.get("lavabod") and sizes.get("chest_size") == 3:
        sizes["chest_size"] = 2
    return sizes


def devkit_values(obj, sizes):
    """The shape keys of mesh ``obj`` that Yet Another Devkit would change, with its settings set as
    ``sizes`` says ({name: number}, see `devkit_state`): {key name: value}, 0 for a key it would mute.
    Nothing is switched: the drivers are worked out here. Raises ResizeError when one involved can't be."""
    keys = _mesh_keys(obj)
    if keys is None or not sizes:
        return {}
    groups = _devkit_groups(keys)
    sizes = devkit_rules({name: value for name, value in sizes.items() if name in groups})
    if "alt_hips" in groups and sizes.get("leg_size", 0) >= DEVKIT_HIP_LEGS:
        sizes["alt_hips"] = 0
    blocks = keys.key_blocks
    worked = {}
    lavabod = sizes.get("lavabod")
    if lavabod is not None and lavabod != _devkit_now(groups, "lavabod"):
        scene, group = groups["lavabod"]
        try:
            kept = getattr(scene.path_resolve("ya_devkit_props." + group), DEVKIT_KEPT_KEYS[int(bool(lavabod))])
        except (ValueError, AttributeError):
            kept = ()
        values = {item.name: item.value for item in kept} or ({DEVKIT_LAVA_OFF: 0.0} if lavabod else {})
        for name, value in values.items():
            if blocks.get(name) is not None:
                worked[(name, 'value')] = value

    drivers = _key_drivers(keys)

    def variable(var):
        # (Its value, whether that's one of the sizes or of the keys worked out.)
        setting = _devkit_setting(var)
        if setting is not None and setting[2] in sizes:
            return float(sizes[setting[2]]), True
        if var.type != 'SINGLE_PROP':
            raise ValueError("it has a {:s} variable".format(var.type.replace("_", " ").lower()))
        target = var.targets[0]
        if target.id == keys:
            found = _KEY_PATH.fullmatch(target.data_path)
            if found is not None and (_unescape(found.group(1)), found.group(2)) in worked:
                return float(worked[(_unescape(found.group(1)), found.group(2))]), True
        value = None if target.id is None else _property_value(target.id, target.data_path)
        if value is None:
            raise ValueError("'{:s}' can't be read: is the devkit's script running?".format(target.data_path))
        return value, False

    # Each round works out the drivers that read a size or a key changed in the rounds before.
    for _round in range(len(drivers) + 1):
        changed = False
        for (name, prop), curve in drivers.items():
            driver = curve.driver
            try:
                read = [(var.name, variable(var)) for var in driver.variables]
                if not any(involved for _name, (_value, involved) in read):
                    continue
                values = [value for _name, (value, _involved) in read]
                if driver.type == 'SCRIPTED':
                    value = float(_expression_value(_parsed(driver.expression),
                                                    {var: value for var, (value, _involved) in read}))
                elif not values:
                    value = 0.0
                else:
                    value = {'AVERAGE': sum(values) / len(values), 'SUM': sum(values), 'MIN': min(values),
                             'MAX': max(values)}[driver.type]
            except (ValueError, SyntaxError, TypeError, KeyError, ArithmeticError) as error:
                raise ResizeError("The driver of '{:s}' on '{:s}' can't be worked out: {:s}".format(
                    name, obj.name, str(error))) from error
            if len(curve.keyframe_points) or len(curve.modifiers):
                value = curve.evaluate(value)
            # Blender turns a boolean on from 1.
            value = value > 1.0 - 1e-6 if prop == 'mute' else value
            if worked.get((name, prop)) != value:
                worked[(name, prop)] = value
                changed = True
        if not changed:
            break

    result = {}
    for name in {name for name, _prop in worked}:
        block = blocks.get(name)
        if block is None or block == keys.reference_key:
            continue
        value = min(max(worked.get((name, 'value'), block.value), block.slider_min), block.slider_max)
        value = 0.0 if worked.get((name, 'mute'), block.mute) else value
        if value != (0.0 if block.mute else block.value):
            result[name] = value
    return result


def devkit_label(sizes, now):
    """A name for the devkit ``sizes`` (see `devkit_state`): those that differ from ``now``, like
    'Medium Rue'; empty when none do."""
    sizes, now = devkit_rules(sizes), devkit_rules(now)
    words = []

    def differs(name):
        return name in sizes and sizes[name] != now.get(name)

    lavabod = sizes.get("lavabod", now.get("lavabod", 0))
    if "chest_size" in sizes and (differs("chest_size") or differs("lavabod")):
        chest = DEVKIT_LAVA_CHEST if lavabod else DEVKIT_CHEST
        words.append(chest[min(sizes["chest_size"], len(chest) - 1)])
    if differs("leg_size"):
        words.append(("Melon", "Skull", "Yanilla", "Masc Legs", "Lava Legs", "Mini")[sizes["leg_size"]])
    if differs("lavabod") or differs("rue"):
        body = [word for name, word in (("lavabod", "Lavabod"), ("rue", "Rue"))
                if sizes.get(name, now.get(name, 0))]
        words.extend(body or ["YAB"])
    if differs("buff"):
        words.append("Buff" if sizes["buff"] else "No Buff")
    return " ".join(words)
