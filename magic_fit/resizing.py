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
is fitted back to the real body afterwards (Keep Close), and where the body would come out between the
clothing's vertices (a pointier breast under a low-poly top), the faces over it are pushed back out (Keep
Covered). `refining` cuts such faces beforehand, so the clothing can take the body's shape there.

The mesh keeps its relation to the first shape: a part clipping into it by 2 mm clips into the second
one about as much. Body Fit fixes that afterwards.

The two shapes can also come from two different bodies (a devkit's RueXB body and another's Rue one):
their vertices are then matched by their UV maps, which body mods made for the same skin textures share.
Bodies of any kind (an outfit's own body from another game, lined up on the Mannequin with it) are
matched by shape instead: each vertex goes to the closest spot of the other body, where that faces the
same way, and the parts that can't be matched like that (hands, feet, a head) follow the rest.

A devkit body can also be resized to its other sizes (chest, legs, Rue...) without switching the devkit: its
settings drive the body's shape keys, and those drivers are worked out again with the sizes picked (see
`sizing`), which gives the To shape the devkit would show. Each size becomes a shape key of each mesh, which
remembers what it was made from (`ResizedKeys`), so the next resize starts from the mesh's own shape and
an unchanged key can be kept.

Like `painting`, the engine (`ShapeChange`) works on world space arrays and doesn't need the 3D view.
"""

import hashlib
import math
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
# Pairs of points and samples kept for the first step of each size, at most (12 bytes each).
CACHED_PAIRS = 20_000_000

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

# Keep Covered: last, clothing that came apart from the body is pushed back out over it. The clothing over a vertex of
# the body is the first triangle its line out meets within this distance...
COVER_REACH = 0.02
# ...where that faces along the line at least this much (the cosine; the side of a strap covers nothing) and lies over
# this part of the body: at least this share as far from the body as from the vertex.
COVER_FACING = 0.5
COVER_OVER = 0.5
# Clothing is kept as far over the body as it was, up to this much (a cape far out over a breast only stays out of
# it), less Keep Close's slack (PUSH_SLACK): what Keep Close lets sink, this does too.
COVER_KEEP = 0.002
# A vertex of the body counts as out through a triangle within this share of the triangle past its edges.
COVER_EDGE = 0.2
# Rounds of measuring and pushing (a push can uncover another vertex); each push is spread to the clothing around it
# this far away, fading smoothly, as far as this many edges reach (its own piece).
COVER_ROUNDS = 2
COVER_SPREAD = 0.02
COVER_RINGS = 16
# A push that would turn triangles over is halved at their corners up to this many times, then left out.
COVER_HALVINGS = 4

# Rigid parts with this share of their points closer to the body than this lie on it: they follow its surface, as
# skin does (see ShapeChange.bind), and grow or shrink with it (see keep_rigid).
RIGID_ON_BODY = 0.005
RIGID_ON_BODY_SHARE = 0.9
# Rigid parts this close to each other touch. Touching parts no larger together than this (a piercing's bar and balls)
# lie on the body, or partly, as one part. In a larger group (a chain), the parts lying on the body follow its surface by
# moving along, and the parts touching them take that move in turn, fading over about this many parts: a chain hanging
# into the cleavage, its links bound to the skin one by one, came apart between those that were and those that weren't.
RIGID_TOUCH = 0.0005
RIGID_GROUP = 0.04
RIGID_CHAIN_FADE = 4.0

# Skin: a mesh with this share of its vertices within this distance of the From shape (a mod's own skin, a
# copy of the body's) follows the body's surface exactly (see ShapeChange.bind) instead of moving like
# clothing, which only approximates it.
SKIN_DISTANCE = 0.001
SKIN_SHARE = 0.9
# So does what's drawn with the game's skin material (mt_c0201b0001_bibo, _b, _a or _eve for the bibo, gen3, gen2
# and Eve layouts: the skin a top's author reshaped under it, lifted or pressed in by a few millimetres)...
SKIN_MATERIAL = re.compile(r"mt_c\d{4}b\d{4}_(bibo|eve|b|a)\b", re.IGNORECASE)
# ...fully within this distance of the From shape, fading into moving like clothing by this distance (skin standing
# farther off the body isn't its surface, and following the body's surface from there isn't undone going back).
SKIN_NEAR = 0.003
SKIN_FAR = 0.006
# Keep Close keeps clothing lying on the mesh's own skin (closer to it than to the body) as far out from that skin as
# it was, but for rounding (the skin's slack, in either direction): skin and clothing are drawn together, and any
# clothing sinking into it, or lifting off it, shows.
SKIN_PUSH_SLACK = 0.0001
SKIN_PULL_SLACK = 0.0001
# Clothing this close to the mesh's own skin takes the skin's correction (how far following the body's surface moves
# it beyond the smooth move) in full, fading out by this distance, averaged over this many skin points around it
# within this distance more than the nearest: clothing made over the skin keeps lying over it, at the gap it was
# made with.
SKIN_FOLLOW_NEAR = 0.005
SKIN_FOLLOW_FAR = 0.015
SKIN_FOLLOW_COUNT = 16
SKIN_FOLLOW_WIDTH = 0.01

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
# Clothing down in a crease, under the crease-free surface, moves with the crease-free shapes only from this share of
# the way up from the skin to that surface, fully at it (the height averaged like the blend): a corset's front and
# what hangs on it at the bottom of the cleavage lie on the breastbone, which stays put while the crease-free surface
# over it falls back with smaller breasts; moved with it, they sank and Keep Close pushed the layers through each
# other.
BRIDGE_DEPTH = 0.5


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
    return np.bincount(tris.T.reshape(-1), weights=np.tile(area, 3), minlength=len(co))


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
        self.co = co
        self.size = 0.5 * reach
        keys = _cell_keys(_cells(co, self.size))
        self.order = np.argsort(keys, kind='stable')
        self.keys, self.starts, self.counts = np.unique(keys[self.order], return_index=True, return_counts=True)

    def pairs(self, points, reach):
        """(point index, sample index) for the samples in the cubes around each point, by point, then cube,
        leaving out those farther than ``reach`` (per point) from every point of the point's own cube."""
        if not len(points):
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        cells = _cells(points, self.size)
        # Points in one cube look in the same cubes around it: look once per cube.
        _keys, cube = np.unique(_cell_keys(cells), return_inverse=True)
        cube = cube.reshape(-1)
        cubes = int(cube.max()) + 1
        first = np.zeros(cubes, dtype=np.int64)
        first[cube[::-1]] = np.arange(len(cube))[::-1]
        own = cells[first]
        query = _cell_keys((own[:, None, :] + _STENCIL[None, :, :]).reshape(-1, 3))
        slot = np.minimum(np.searchsorted(self.keys, query), len(self.keys) - 1)
        found = np.flatnonzero(self.keys[slot] == query)
        counts = self.counts[slot[found]]
        run = np.repeat(np.arange(len(found)), counts)
        offsets = np.arange(len(run)) - (np.cumsum(counts) - counts)[run]
        holder = found[run] // len(_STENCIL)
        sample = self.order[self.starts[slot[found]][run] + offsets]
        # Samples out of reach of the whole cube (by a margin over single precision's rounding) weigh nothing.
        farthest = np.zeros(cubes)
        np.maximum.at(farthest, cube, reach)
        low = own[holder] * self.size
        outside = np.maximum(np.maximum(low - self.co[sample], self.co[sample] - (low + self.size)), 0.0)
        limit = farthest[holder] * 1.001 + 1e-6
        near = np.einsum("ij,ij->i", outside, outside) <= limit * limit
        holder, sample = holder[near], sample[near]
        # Then each point takes its cube's samples.
        start = np.searchsorted(holder, np.arange(cubes + 1))
        counts = (start[1:] - start[:-1])[cube]
        run = np.repeat(np.arange(len(cube)), counts)
        offsets = np.arange(len(run)) - (np.cumsum(counts) - counts)[run]
        return run, sample[start[:-1][cube][run] + offsets]


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

    def _mean(self, level, area):
        members = self.members[min(level, len(self.members) - 1)]
        count = int(members.max()) + 1
        total = np.bincount(members, weights=area, minlength=count)
        weight = area / np.maximum(total[members], 1e-300)

        def mean(values):
            return np.stack([np.bincount(members, weights=weight * values[:, k], minlength=count) for k in range(3)], 1)
        return mean, total, total > 0.0

    def samples(self, level, co, area, displacement):
        """Positions, areas and displacements of the samples at ``level`` for the body at ``co``."""
        if level == 0:
            return co, area, displacement
        mean, total, keep = self._mean(level, area)
        return mean(co)[keep], total[keep], mean(displacement)[keep]

    def moves(self, level, area, displacement):
        """The displacements alone of `samples`."""
        if level == 0:
            return displacement
        mean, _total, keep = self._mean(level, area)
        return mean(displacement)[keep]


def _weights(points, reach, co, area):
    """The samples at ``co`` (areas ``area``) within ``reach`` of each of ``points``, weighted by their area
    and (1 - (distance / reach)^2)^3. Yields, per chunk of the points, (its first point, its size, point
    in the chunk, sample, weight) for every pair weighing anything, by point."""
    grid = _Grid(co, float(reach.max()))
    # Pairs are worked out in single precision, relative to each point, which is plenty for weights.
    co = co.astype(np.float32)
    area = area.astype(np.float32)
    inverse = (1.0 / reach ** 2).astype(np.float32)
    for first in range(0, len(points), CHUNK):
        chunk = slice(first, first + CHUNK)
        p = points[chunk].astype(np.float32)
        owner, sample = grid.pairs(points[chunk], reach[chunk])
        delta = co[sample] - p[owner]
        falloff = np.maximum(1.0 - np.einsum("ij,ij->i", delta, delta) * inverse[chunk][owner], 0.0)
        near = np.flatnonzero(falloff > 0.0)
        owner, sample, falloff = owner[near], sample[near], falloff[near]
        yield first, len(p), owner, sample, area[sample] * falloff * falloff * falloff


def _average(points, reach, samples):
    """For each point, the average displacement of the samples within ``reach`` of it, each weighted by
    its area and (1 - (distance / reach)^2)^3. Points without any sample in reach don't move."""
    co, area, displacement = samples
    result = np.zeros_like(points)
    if not len(co):
        return result
    displacement = displacement.astype(np.float32)
    for first, size, owner, sample, weight in _weights(points, reach, co, area):
        chunk = slice(first, first + size)
        total = np.bincount(owner, weights=weight, minlength=size)
        moved = displacement[sample]
        for k in range(3):
            result[chunk, k] = np.bincount(owner, weights=weight * moved[:, k], minlength=size)
        result[chunk] /= np.maximum(total, 1e-300)[:, None]
    return result


class _FirstStep:
    """Which samples each point averages in the first step of a change (see `ShapeChange._step`), and how
    much each weighs. That step starts from the From shape, so these are the same whatever the To shape:
    `Shared` keeps them for every size, and only the samples' displacements are worked out again.

    ``points`` are the points, ``chosen`` those to work out (indices), ``distance`` their distances from
    the body, which has vertices ``co`` and areas ``area`` (those used) clustered by ``levels``. ``full``
    is False when the pairs would take more than CACHED_PAIRS: the step is then worked out as usual."""

    def __init__(self, points, chosen, distance, radius, co, area, levels):
        self.count = len(points)
        self.levels = []
        offsets, samples, size, kept = {}, {}, 0, 0
        rows, cols, weights = [], [], []
        reach, octave, groups = _reaches(distance, radius)
        self.full = True
        for level, which, share in _shares(groups, octave):
            if level not in offsets:
                # (The displacements, here the positions again, don't matter yet.)
                samples[level] = levels.samples(level, co, area, co)
                offsets[level] = size
                size += len(samples[level][0])
                self.levels.append(level)
            level_co, level_area, _ = samples[level]
            if not len(level_co):
                continue
            for first, count, owner, sample, weight in _weights(points[chosen[which]], reach[which], level_co,
                                                                level_area):
                total = np.bincount(owner, weights=weight, minlength=count)
                rows.append(chosen[which[first + owner]].astype(np.int32))
                cols.append((offsets[level] + sample).astype(np.int32))
                weights.append((share[first + owner] * weight / np.maximum(total, 1e-300)[owner]).astype(np.float32))
                kept += len(owner)
                if kept > CACHED_PAIRS:
                    self.full = False
                    return
        rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int32)
        order = np.argsort(rows, kind='stable')
        self.rows = rows[order]
        self.cols = np.concatenate(cols)[order] if cols else np.zeros(0, dtype=np.int32)
        self.weights = np.concatenate(weights)[order] if weights else np.zeros(0, dtype=np.float32)
        self.starts = np.concatenate(([0], np.cumsum(np.bincount(self.rows, minlength=self.count))))

    def moves(self, which, area, displacement, levels):
        """How the points ``which`` (indices) move in the first step, the body's vertices (those used, with
        areas ``area``) moving by ``displacement``."""
        values = np.concatenate([levels.moves(level, area, displacement) for level in self.levels]) \
            if self.levels else np.zeros((0, 3))
        if len(which) == self.count:
            rows, cols, weights = self.rows, self.cols, self.weights
        else:
            start = self.starts[which]
            counts = self.starts[which + 1] - start
            rows = np.repeat(np.arange(len(which)), counts)
            entries = np.arange(len(rows)) - (np.cumsum(counts) - counts)[rows] + start[rows]
            cols, weights = self.cols[entries], self.weights[entries]
        moved = values[cols]
        return np.stack([np.bincount(rows, weights=weights * moved[:, k], minlength=len(which))
                         for k in range(3)], axis=1)


def _reaches(distance, radius):
    """For points at ``distance`` from the body: how far their averages reach, their octave (doublings of the
    width over ``radius``) and their group by width (see BINS_PER_OCTAVE)."""
    width = np.maximum(radius, SPREAD * distance)
    reach = np.sqrt(distance * distance + (KERNEL_REACH * width) ** 2)
    octave = np.log2(width / radius)
    return reach, octave, np.floor(octave * BINS_PER_OCTAVE).astype(np.int64)


def _shares(groups, octave):
    """(level, points, their shares) for the levels of samples that points in ``groups`` average over: one
    level, or two neighbouring ones in between, so the average changes smoothly with the width."""
    for group in np.unique(groups):
        which = np.flatnonzero(groups == group)
        low = int(group) // BINS_PER_OCTAVE
        blend = octave[which] - low
        for level, share in ((low, 1.0 - blend), (low + 1, blend)):
            active = share > 0.0
            if active.any():
                yield level, which[active], share[active]


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
    normals = np.stack([np.bincount(corners.T.reshape(-1), weights=np.tile(face[:, k], 3), minlength=count)
                        for k in range(3)], axis=1)
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
        # Each edge both ways, and the lifts by axis (rows), for quick sums over neighbours.
        into, out_of = np.concatenate((first, second)), np.concatenate((second, first))
        degree = np.maximum(np.bincount(into, minlength=count), 1)
        lift = np.ascontiguousarray(lift.T)
        total = np.empty((3, count))
        for _ in range(passes):
            for k in range(3):
                total[k] = np.bincount(into, weights=lift[k][out_of], minlength=count)
            lift = 0.5 * (lift + total / degree)
        lift = lift.T.copy()
    unfaced = np.ones(count, dtype=bool)
    unfaced[faced] = False
    lift[unfaced] = 0.0
    return co + lift[weld]


class Shared:
    """What resizes from one From shape share, whatever their To shapes, kept for a batch of them (sizes):
    pass one to each of their ShapeChanges. That's the From shape's clusters and crease-free copy, and for
    the same clothing, its distances from the body, how it blends the two moves (Bridge Creases), the
    first step's averages and Keep Close's measurements. Each is found by the exact values it's made
    from, so changed inputs just make another."""

    def __init__(self):
        self._entries = {}

    def get(self, key, make):
        """What's kept under ``key``, made by ``make()`` the first time."""
        entry = self._entries.get(key)
        if entry is None:
            entry = self._entries[key] = make()
        return entry


def _shared(shared, key, make):
    return make() if shared is None else shared.get(key, make)


def _digest(*parts):
    """A key for ``parts``: arrays by their exact values, anything else by its repr."""
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(repr((part.dtype.str, part.shape)).encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(repr(part).encode())
        digest.update(b"\0")
    return digest.hexdigest()


class _FromShape:
    """What a ShapeChange takes from its source shape alone (see `Shared`)."""

    def __init__(self, source, tris, radius, bridge):
        area = _vertex_areas(source, tris) if len(tris) else np.zeros(len(source))
        if area.sum() <= 0.0:
            raise ResizeError("The body has no faces")
        edges = np.linalg.norm(source[tris] - source[np.roll(tris, 1, axis=1)], axis=2)
        self.radius = max(float(radius), MIN_RADIUS_EDGES * float(np.median(edges)), 1e-6)
        # Vertices of no face (loose ones) stand for no surface and are left out.
        self.used = area > 0.0
        self.area = area[self.used]
        self.levels = _Levels(source[self.used], self.radius)
        self.tree_tris = tris.tolist()
        self.bridged = None
        if bridge > 0.0:
            bridged_from = bridged(source, tris, bridge)
            self.bridged = (bridged_from, _vertex_areas(bridged_from, tris)[self.used],
                            _Levels(bridged_from[self.used], self.radius))
        self._surface = self._bridged_surface = None

    def surface(self, source, tris):
        """The source shape as a BodySurface (made when first asked for), for binding skin to it."""
        if self._surface is None:
            self._surface = _surface(source, tris)
        return self._surface

    def bridged_surface(self, source, tris):
        """The source shape's crease-free version as a BodySurface (made when first asked for)."""
        if self._bridged_surface is None:
            self._bridged_surface = _surface(self.bridged[0], tris, like=self.surface(source, tris))
        return self._bridged_surface


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

    ``shared`` (a Shared) keeps what changes from the same source shape share, for several target shapes.
    """

    def __init__(self, source, target, tris, radius=DEFAULT_RADIUS, bridge=0.0, to_body=None, known=None,
                 shared=None):
        self.source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
        self.target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
        self.tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        self.to_body, self.known = to_body, known
        if self.source.shape != self.target.shape:
            raise ResizeError("The two body shapes have different vertex counts")
        self.bridge = max(float(bridge), 0.0)
        self._shared = shared
        self._key = _digest(self.source, self.tris, float(radius), self.bridge)
        self._from = _shared(shared, ("from", self._key),
                             lambda: _FromShape(self.source, self.tris, float(radius), self.bridge))
        self.radius = self._from.radius
        self._used = self._from.used
        self._levels = self._from.levels
        self._tree_tris = self._from.tree_tris
        moves = np.linalg.norm(self.target - self.source, axis=1)[self._used]
        # Bridge Creases: crease-free versions of both shapes (and their clusters), for clothing that
        # doesn't lie on the skin.
        self._bridged = None
        if self.bridge > 0.0:
            bridged_from, _area, levels = self._from.bridged
            bridged_to = bridged(self.target, self.tris, self.bridge)
            self._bridged = (bridged_from, bridged_to, levels)
            moves = np.maximum(moves, np.linalg.norm(bridged_to - bridged_from, axis=1)[self._used])
            # Where both shapes of a vertex are unlifted the two moves agree; they differ only within reach
            # of the lifted ones, in either shape.
            lifted = self._used & ((np.abs(bridged_from - self.source).max(axis=1) > 1e-6)
                                   | (np.abs(bridged_to - self.target).max(axis=1) > 1e-6))
            self._lifted = np.concatenate([self.source[lifted], self.target[lifted]])
        self.largest = float(moves.max()) if len(moves) else 0.0
        self.steps = min(MAX_STEPS, max(1, math.ceil(self.largest / (STEP_RADII * self.radius))))

    def _path(self, crease_free):
        """The source and target shapes clothing moves between, their clusters and the source's areas (of
        the vertices used): the body's, or with ``crease_free``, its crease-free versions'."""
        if crease_free:
            bridged_from, bridged_to, levels = self._bridged
            return bridged_from, bridged_to, levels, self._from.bridged[1]
        return self.source, self.target, self._levels, self._from.area

    def _step(self, points, which, crease_free, step, clothing):
        """How ``points[which]`` move in step ``step`` of the change, between the body's shapes or their
        crease-free versions (``crease_free``). ``clothing`` (a _Clothing of ``points`` as they were before
        the first step) keeps that step's averages, for other changes sharing it."""
        source, target, levels, area = self._path(crease_free)
        before = source + (target - source) * (step / self.steps)
        after = source + (target - source) * ((step + 1) / self.steps)
        used = self._used
        displacement = (after - before)[used]
        if step == 0 and self._shared is not None:
            first = clothing.first_step(self, crease_free)
            if first.full:
                return first.moves(which, area, displacement, levels)
        else:
            area = _vertex_areas(before, self.tris)[used]
        points = points[which]
        distance = _distances(points, before, self._tree_tris)
        reach, octave, groups = _reaches(distance, self.radius)
        co = before[used]
        samples = {}
        moved = np.zeros_like(points)
        for level, chosen, share in _shares(groups, octave):
            if level not in samples:
                samples[level] = levels.samples(level, co, area, displacement)
            moved[chosen] += share[:, None] * _average(points[chosen], reach[chosen], samples[level])
        return moved

    def apply(self, points, progress=None, rigid=None, edges=None, bound=None, faces=None, skin=None, follow=None,
              cover=None):
        """Where ``points`` (n x 3, world space, fitted to the source shape) go with the target shape.

        ``rigid`` optionally labels points that move together without changing shape (a label per
        point, -1 for none): each such group is moved the rigid way closest to how the others would
        move it. ``edges`` (pairs of point indices: the clothing's edges), when given, turn on Keep
        Close, see `keep_close`. ``bound`` (a mask, or a share from 0 to 1, per point) marks skin: points
        lying on the source shape that follow its surface exactly (see `bind`), taking no part in the
        smooth move, in Keep Close or in rigid groups; a share between blends the two moves (see
        `skin_shares`). Clothing near the skin takes the skin's correction (see SKIN_FOLLOW_NEAR).
        ``faces`` (triples of point indices: the meshes' triangles) give Keep Close the skin's surface, so
        clothing lying on it keeps its distance from it (see `keep_close`). ``cover`` (the triangles of all the
        meshes, likewise) turns on Keep Covered: clothing the body came out through, between its points or at them,
        is pushed back out over it (see `keep_covered`). ``progress(fraction)`` is called after every step. ``skin``
        (that surface, as `keep_close` takes it) and ``follow`` (the skin's points, where they go and their shares) are
        for the move without the skin.
        """
        start = np.array(points, dtype=np.float64).reshape(-1, 3)
        if bound is not None:
            bound = np.clip(np.asarray(bound, dtype=np.float64).reshape(-1), 0.0, 1.0)
            if bound.any():
                held = np.flatnonzero(bound > 0.0)
                skin_at = self.bind(start[held])
                if edges is not None and faces is not None:
                    # The skin's surface where it follows the body fully: elsewhere it's partly clothing.
                    index = np.full(len(start), -1, dtype=np.int64)
                    index[held] = np.where(bound[held] >= 1.0, np.arange(len(held)), -1)
                    skin_tris = index[np.asarray(faces, dtype=np.int64).reshape(-1, 3)]
                    skin_tris = skin_tris[(skin_tris >= 0).all(axis=1)]
                    if len(skin_tris):
                        skin = (start[held], skin_at, skin_tris)
                if rigid is not None:
                    rigid = np.array(rigid, dtype=np.int64).reshape(-1)
                    rigid[held] = -1
                # Everything moves smoothly, the skin too, so the clothing around it can take its correction.
                result = self.apply(start, progress, rigid, edges, skin=skin, follow=(held, skin_at, bound[held]))
                share = bound[held, None]
                result[held] = share * skin_at + (1.0 - share) * result[held]
                if cover is not None:
                    # The skin stays on the body.
                    result = self.keep_covered(start, result, cover, rigid, fixed=bound >= 0.5)
                return result
        if not len(start) or self.largest <= 0.0:
            return start
        clothing = _shared(self._shared, ("clothing", self._key, _digest(start)), lambda: _Clothing(self, start))
        points, copies = clothing.points, clothing.copies
        first = points
        rounds = KEEP_ROUNDS if edges is not None else 0
        total = self.steps + rounds
        # Clothing on the skin follows it; farther out it moves with the crease-free shapes.
        free = None
        if self._bridged is not None:
            distance = clothing.distance
            free = clothing.free.copy()
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
                moved = self._step(points, np.arange(len(points)), False, step, clothing)
            else:
                moved = np.zeros_like(points)
                on_skin = np.flatnonzero(free < 1.0)
                if len(on_skin):
                    moved[on_skin] = (1.0 - free[on_skin, None]) * self._step(points, on_skin, False, step, clothing)
                spanning = np.flatnonzero(free > 0.0)
                if len(spanning):
                    moved[spanning] += free[spanning, None] * self._step(points, spanning, True, step, clothing)
            points = points + moved
            if progress is not None:
                progress((step + 1) / total)
        if follow is not None:
            # The skin's correction, carried over to the clothing around it.
            held, skin_at, held_share = follow
            spots = copies[held]
            correction = (np.asarray(skin_at) - points[spots]) * np.asarray(held_share)[:, None]
            rows, cols, weights = clothing.followers(spots)
            carried = np.zeros_like(points)
            np.add.at(carried, rows, weights[:, None] * correction[cols])
            points = points + carried
            points[spots] = points[spots] - carried[spots] + correction
        if edges is not None:
            loose = np.ones(len(points), dtype=bool)
            if rigid is not None:
                loose[copies[np.asarray(rigid) >= 0]] = False
            if follow is not None:
                loose[copies[follow[0]]] = False

            def keep_progress(done):
                if progress is not None:
                    progress((self.steps + done * rounds) / total)
            to_co, to_tris = self.to_body if self.to_body is not None else (self.target, None)
            pairs = copies[np.asarray(edges, dtype=np.int64).reshape(-1, 2)]
            points = keep_close(first, points, pairs, self.source, to_co, self.tris, anchors=loose,
                                progress=keep_progress, target_tris=to_tris, known=self.known, skin=skin,
                                measured=clothing.keep_close(self, pairs, loose, skin))
        result = points[copies]
        if rigid is not None:
            # Rigid parts lying on the body (piercings, studs, a ring on the skin) follow its surface exactly, as
            # the mesh's skin does (see `bind`), rather than the smooth move, which lags it. Only whole: a part
            # partly off the body, aimed two ways at once, can come out turned over. Small groups of touching parts
            # count as one part; in a chain, the links on the body pull the others along (see RIGID_TOUCH).
            rigid = np.asarray(rigid)
            on_body = np.zeros(len(rigid), dtype=bool)
            members = np.flatnonzero(rigid >= 0)
            owner = near = small = chained = None
            if len(members):
                _labels, part = np.unique(rigid[members], return_inverse=True)
                part = part.reshape(-1)
                from .fitting import _connected
                touching = _touching(start[members], part)
                group = _connected(int(part.max()) + 1, touching)[0]
                _labels, owner = np.unique(group[part], return_inverse=True)
                owner = owner.reshape(-1)
                near = clothing.near(self)[copies][members] <= RIGID_ON_BODY
                groups = int(owner.max()) + 1
                # Groups no larger than RIGID_GROUP, or of a single part, aren't chains.
                parts_in = np.bincount(np.unique(np.stack((owner, part), axis=1), axis=0)[:, 0], minlength=groups)
                small = (part_sizes(start[members], owner, groups) <= RIGID_GROUP) | (parts_in == 1)
                chained = ~small[owner]
                # A small group by its share of points on the body, a chain's link by its own.
                group_share = np.bincount(owner, weights=near) / np.bincount(owner)
                part_share = np.bincount(part, weights=near) / np.bincount(part)
                on_body[members] = (group_share[owner] >= RIGID_ON_BODY_SHARE) & small[owner]
                linked = (part_share[part] >= RIGID_ON_BODY_SHARE) & chained
            if on_body.any():
                result[on_body] = self.bind(start[on_body])
            keep_rigid(start, result, rigid, on_body=on_body)
            if chained is not None and linked.any():
                # Links of a chain lying on the body move along, without turning or growing, as far as they fall short
                # of following it, and the other links take that move, fading away from them along the chain. Bound
                # to the body one by one, links lying in a crease, their points bound to both sides of it, grew or
                # turned as the sides parted.
                count = int(part.max()) + 1
                rows = members[linked]
                short = self.bind(start[rows]) - result[rows]
                values = np.stack([np.bincount(part[linked], weights=short[:, k], minlength=count)
                                   for k in range(3)], axis=1)
                fixed = np.bincount(part[linked], minlength=count) > 0
                values[fixed] /= np.bincount(part[linked], minlength=count)[fixed, None]
                shift = _screened_fill(values, np.zeros(count), touching, count, 1.0, 1.0 / RIGID_CHAIN_FADE ** 2,
                                       fixed=fixed)
                result[members[chained]] += shift[part[chained]]
            # A part (or small group) partly on the body is moved, without turning, as far as its points on the body
            # fall short of following it.
            partly = near & ~on_body[members] & ~chained if near is not None else None
            if partly is not None and partly.any():
                rows = members[partly]
                short = self.bind(start[rows]) - result[rows]
                shift = np.stack([np.bincount(owner[partly], weights=short[:, k], minlength=groups)
                                  for k in range(3)], axis=1)
                shift /= np.maximum(np.bincount(owner[partly], minlength=groups), 1)[:, None]
                moved = ~on_body[members] & ~chained
                result[members[moved]] += shift[owner[moved]]
        if cover is not None:
            result = self.keep_covered(start, result, cover, rigid)
        return result

    def keep_covered(self, start, moved, cover, rigid=None, fixed=None):
        """``moved`` (where `apply` took ``start``) with the clothing pushed back out over the body (Keep Covered, see
        `keep_covered`). ``cover`` are the triangles of all the meshes (point indices), ``rigid`` labels rigid parts as
        `apply` takes it, and ``fixed`` (a mask) marks points that stay where they are (skin)."""
        tris = np.asarray(cover, dtype=np.int64).reshape(-1, 3)
        if not len(tris) or not len(start) or self.largest <= 0.0:
            return moved
        source = self._from.surface(self.source, self.tris)
        fixed = np.zeros(len(start), dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool)
        found = _shared(self._shared, ("cover", self._key, _digest(start, tris, fixed)),
                        lambda: _Cover(start, tris, source, fixed))
        return keep_covered(found, moved, _surface(self.target, self.tris, like=source), rigid)

    def bind(self, points):
        """Where ``points`` lying on the source shape (skin) go with the target shape: each keeps its spot
        on its closest triangle of the source and its (small) distance from it along the normal there.

        That is exact for a copy of the body's skin, which the smooth move only approximates: it averages
        the body's move around each point, and Keep Close leaves tight clothing some slack. A point past
        an open edge of the body keeps its offset from the edge.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not len(points) or self.largest <= 0.0:
            return points.copy()
        source = self._from.surface(self.source, self.tris)
        near = _shared(self._shared, ("bound", self._key, _digest(points)), lambda: source.nearest(points))
        target = _surface(self.target, self.tris, like=source)
        normal = np.einsum("ij,ijk->ik", near.bary, target.normals[near.corners])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        location = np.einsum("ij,ijk->ik", near.bary, target.co[near.corners])
        offset = np.where(near.beyond[:, None], points - near.location, near.distance[:, None] * normal)
        return np.where(near.found[:, None], location + offset, points)


class _Clothing:
    """What ShapeChange.apply works out for clothing at ``start`` on the source shape of ``change``, the same
    whatever the target shape (see `Shared`)."""

    def __init__(self, change, start):
        # Points at the same spot (split seams, touching parts) go the same way: work each out once.
        self.points, copies = np.unique(start, axis=0, return_inverse=True)
        self.copies = copies.reshape(-1)
        self.distance = self.free = None
        if change.bridge > 0.0:
            self.distance = _distances(self.points, change.source, change._tree_tris)
            self.free = _blend_shares(self.points, _smoothstep(BRIDGE_HUG, BRIDGE_HUG + BRIDGE_BLEND, self.distance))
            # Clothing down in a crease, nearer the skin than the crease-free surface over it, follows the skin.
            spanning = np.flatnonzero(self.free > 0.0)
            if len(spanning):
                surface = change._from.bridged_surface(change.source, change.tris)
                under = np.maximum(-surface.nearest(self.points[spanning]).distance, 0.0)
                height = np.ones(len(self.points))
                height[spanning] = self.distance[spanning] / np.maximum(self.distance[spanning] + under, 1e-12)
                self.free = self.free * _smoothstep(BRIDGE_DEPTH, 1.0, _blend_shares(self.points, height))
        self._first = {}
        self._kept = {}
        self._followers = {}

    def followers(self, spots):
        """For each point within SKIN_FOLLOW_FAR of the skin (``spots``, indices of these points), the skin points
        around it whose correction it takes and their weights (see SKIN_FOLLOW_NEAR), as (point, index into
        ``spots``, weight) rows."""
        key = _digest(spots)
        if key not in self._followers:
            from mathutils.kdtree import KDTree
            tree = KDTree(len(spots))
            for index, point in enumerate(self.points[spots].tolist()):
                tree.insert(point, index)
            tree.balance()
            rows, cols, weights = [], [], []
            # Only points within SKIN_FOLLOW_FAR of the skin's box can be that near the skin.
            skin = self.points[spots]
            low, high = skin.min(axis=0) - SKIN_FOLLOW_FAR, skin.max(axis=0) + SKIN_FOLLOW_FAR
            candidates = np.flatnonzero(np.all((self.points >= low) & (self.points <= high), axis=1)) \
                if len(spots) else np.zeros(0, dtype=np.int64)
            for k, point in zip(candidates.tolist(), self.points[candidates].tolist()):
                # The nearest alone first: most points are too far to take anything.
                if tree.find(point)[2] >= SKIN_FOLLOW_FAR:
                    continue
                found = tree.find_n(point, SKIN_FOLLOW_COUNT)
                if not found or found[0][2] >= SKIN_FOLLOW_FAR:
                    continue
                fade = 1.0 - _smoothstep(SKIN_FOLLOW_NEAR, SKIN_FOLLOW_FAR, found[0][2])
                # A smooth average over the skin around (as wide as the nearest is far, and SKIN_FOLLOW_WIDTH more),
                # so the clothing takes the skin's correction without its small bumps.
                reach = found[0][2] + SKIN_FOLLOW_WIDTH
                near = [max(1.0 - (distance / reach) ** 2, 0.0) ** 3 for _co, _index, distance in found]
                total = sum(near)
                for (_co, index, _distance), weight in zip(found, near):
                    rows.append(k)
                    cols.append(index)
                    weights.append(fade * weight / total)
            self._followers[key] = (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
                                    np.array(weights, dtype=np.float64))
        return self._followers[key]

    def near(self, change):
        """How far each point is from the source shape of ``change``."""
        if self.distance is None:
            self.distance = _distances(self.points, change.source, change._tree_tris)
        return self.distance

    def first_step(self, change, crease_free):
        """The first step's averages (a _FirstStep) of the points that may move with the body's shapes (all
        of them), or with ``crease_free``, with their crease-free versions (those blending towards them)."""
        if crease_free not in self._first:
            source, _target, levels, area = change._path(crease_free)
            if crease_free:
                chosen = np.flatnonzero(self.free > 0.0)
                distance = _distances(self.points[chosen], source, change._tree_tris)
            else:
                chosen = np.arange(len(self.points))
                distance = self.distance if self.distance is not None else \
                    _distances(self.points, source, change._tree_tris)
            self._first[crease_free] = _FirstStep(self.points, chosen, distance, change.radius,
                                                  source[change._used], area, levels)
        return self._first[crease_free]

    def keep_close(self, change, edges, anchors, skin=None):
        """Keep Close's measurements (a _KeepClose) of these points, joined by ``edges``, held where not in
        ``anchors``, with the mesh's own ``skin`` (see `keep_close`)."""
        key = _digest(edges, anchors, *(() if skin is None else (skin[0], skin[2])))
        if key not in self._kept:
            self._kept[key] = _KeepClose(self.points, edges, change.source, change.tris, anchors,
                                         None if skin is None else (skin[0], skin[2]))
        return self._kept[key]


def _turned_like(surface, body, turns=None):
    """Turn the normals of ``surface`` (a mesh's own skin: open, so which side is out can't be told from it) per
    connected part to face the way those of the ``body`` surface do where it lies, or as ``turns`` (whether to turn
    each of the points it was made from) say. Returns ``turns``."""
    if turns is None:
        near = body.nearest(surface.co)
        vote = np.einsum("ij,ij->i", surface.normals, near.normal) * near.found
        turns = (np.bincount(surface.parts, weights=vote, minlength=int(surface.parts.max()) + 1) < 0.0)[
            surface.parts[surface._weld]]
    # By the points (a surface of them moved can weld differently).
    sign = np.ones(len(surface.co))
    sign[surface._weld[turns]] = -1.0
    surface.normals = surface.normals * sign[:, None]
    surface.area_normals = surface.area_normals * sign[:, None]
    surface.tri_normals = surface.tri_normals * sign[surface.corners[:, 0], None]
    return turns


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


def _surface(co, tris, like=None):
    from .fitting import BodySurface
    return BodySurface(co, tris, BVHTree.FromPolygons(co.tolist(), tris.tolist(), all_triangles=True), like)


class _KeepClose:
    """What Keep Close measures of clothing on the source shape, the same whatever the target shape (see
    `keep_close` for the arguments)."""

    def __init__(self, start, edges, source, tris, anchors=None, skin=None):
        start = np.asarray(start, dtype=np.float64)
        # Points at the same spot are one.
        _spots, weld = np.unique(start, axis=0, return_inverse=True)
        self.weld = weld.reshape(-1)
        count = self.count = int(self.weld.max()) + 1 if len(self.weld) else 0
        self.first = np.zeros(count, dtype=np.int64)
        self.first[self.weld[::-1]] = np.arange(len(self.weld))[::-1]
        source_surface = self.surface = _surface(np.asarray(source, dtype=np.float64),
                                                 np.asarray(tris, dtype=np.int64))
        near = source_surface.nearest(start[self.first])
        self.height = near.distance
        self.close = near.found & ~near.beyond & (np.abs(self.height) < KEEP_CLOSE_DISTANCE)
        self.corners, self.source_weld, self.source_count = near.corners, source_surface._weld, len(source_surface.co)
        # Points closer to the mesh's own skin (its points on the source shape and its triangles) than to the body
        # keep their distance from the skin.
        self.on_skin = np.zeros(count, dtype=bool)
        self.skin_surface = self.skin_turns = None
        if skin is not None:
            skin_surface = self.skin_surface = _surface(np.asarray(skin[0], dtype=np.float64),
                                                        np.asarray(skin[1], dtype=np.int64))
            self.skin_turns = _turned_like(skin_surface, source_surface)
            on = skin_surface.nearest(start[self.first])
            body_gap = np.where(near.found & ~near.beyond, np.abs(self.height), np.inf)
            self.on_skin = on.found & ~on.beyond & (np.abs(on.distance) < body_gap)
            self.height = np.where(self.on_skin, on.distance, self.height)
            self.close = (self.close & ~self.on_skin) | (self.on_skin & (np.abs(self.height) < KEEP_CLOSE_DISTANCE))
        if anchors is not None:
            free = np.ones(count, dtype=bool)
            np.logical_and.at(free, self.weld, np.asarray(anchors, dtype=bool))
            self.close &= free
        pairs = np.sort(self.weld[np.asarray(edges, dtype=np.int64).reshape(-1, 2)], axis=1)
        pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
        # Each edge both ways, for sums over each point's neighbours.
        self.into = np.concatenate((pairs[:, 0], pairs[:, 1]))
        self.out_of = np.concatenate((pairs[:, 1], pairs[:, 0]))
        self.degree = np.maximum(np.bincount(self.into, minlength=count), 1)


def keep_close(start, moved, edges, source, target, tris, anchors=None, progress=None, target_tris=None,
               known=None, measured=None, skin=None):
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

    With ``skin`` (the mesh's own skin: its points on the source shape, where they went, and its triangles), points
    closer to the skin than to the body keep their distance from the skin instead, and never come closer to it
    than they were (SKIN_PUSH_SLACK): a garment's skin is drawn with it, and clothing sinking into it shows.

    ``measured`` is a _KeepClose of the same ``start``, ``edges``, ``source``, ``tris``, ``anchors`` and ``skin``,
    made before for another target.
    """
    tris = np.asarray(tris, dtype=np.int64)
    kept = measured if measured is not None else _KeepClose(
        start, edges, source, tris, anchors, None if skin is None else (skin[0], skin[2]))
    count, into, out_of, degree = kept.count, kept.into, kept.out_of, kept.degree
    points = np.asarray(moved, dtype=np.float64)[kept.first]
    height, close = kept.height, kept.close
    if known is not None:
        welded = np.zeros(kept.source_count, dtype=bool)
        welded[kept.source_weld] = np.asarray(known, dtype=bool)
        close = close & (kept.on_skin | (welded[kept.corners].sum(axis=1) >= 2))
    skin_surface = None
    if kept.skin_surface is not None and skin is not None:
        skin_surface = _surface(np.asarray(skin[1], dtype=np.float64), kept.skin_surface.tris,
                                like=kept.skin_surface)
        _turned_like(skin_surface, None, kept.skin_turns)
    if target_tris is None:
        # The source's triangles: what they decide is the source's, where the vertices weld alike.
        body = _surface(np.asarray(target, dtype=np.float64), tris, like=kept.surface)
    else:
        body = _surface(np.asarray(target, dtype=np.float64), np.asarray(target_tris, dtype=np.int64).reshape(-1, 3))
    which = np.flatnonzero(close)
    on_skin = kept.on_skin[which] if skin_surface is not None else np.zeros(len(which), dtype=bool)
    push_slack = np.where(on_skin, SKIN_PUSH_SLACK, PUSH_SLACK)
    pull_slack = np.where(on_skin, SKIN_PULL_SLACK, PULL_SLACK)
    settled = False
    for done in range(KEEP_ROUNDS if len(which) else 0):
        if not settled:
            near = body.nearest(points[which])
            if on_skin.any():
                on = skin_surface.nearest(points[which[on_skin]])
                for name in ("found", "beyond", "distance", "normal"):
                    getattr(near, name)[on_skin] = getattr(on, name)
            ok = near.found & ~near.beyond
            need = np.where(ok, height[which] - near.distance, 0.0)
            need = np.where(need > 0.0, np.maximum(need - push_slack, 0.0), np.minimum(need + pull_slack, 0.0))
            # Nothing to correct: nor will there be in the rounds left.
            settled = not need.any()
        if not settled:
            # By axis (rows), for quick sums over neighbours.
            own = np.zeros((3, count))
            own[:, which] = (need[:, None] * near.normal).T
            held = np.zeros(count, dtype=bool)
            held[which[ok]] = True
            keep = np.where(held, KEEP_SHARE, 0.0) * own
            scale = np.where(held, 1.0 - KEEP_SHARE, KEEP_FADE) / degree
            spread = own
            total = np.empty((3, count))
            for _ in range(KEEP_PASSES):
                for k in range(3):
                    total[k] = np.bincount(into, weights=spread[k][out_of], minlength=count)
                spread = keep + scale * total
            points = points + spread.T
        if progress is not None:
            progress((done + 1) / KEEP_ROUNDS)
    return points[kept.weld]


class _Cover:
    """What Keep Covered finds of clothing at ``start`` (n x 3) on the source shape (``body``, a BodySurface), with
    the meshes' triangles ``tris`` and their ``fixed`` points (skin, a mask), the same whatever the target shape: the
    body under each point and the clothing over each vertex of the body (see `keep_covered`)."""

    def __init__(self, start, tris, body, fixed=None):
        start = np.asarray(start, dtype=np.float64)
        # Points at the same spot are one.
        _spots, weld = np.unique(start, axis=0, return_inverse=True)
        self.weld = weld.reshape(-1)
        count = self.count = int(self.weld.max()) + 1
        self.first = np.zeros(count, dtype=np.int64)
        self.first[self.weld[::-1]] = np.arange(len(self.weld))[::-1]
        points = start[self.first]
        tris = self.weld[tris]
        tris = self.tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])]
        # How the triangles face (unscaled), which pushes mustn't turn over.
        self.normals = np.cross(points[tris[:, 1]] - points[tris[:, 0]], points[tris[:, 2]] - points[tris[:, 0]])
        self.fixed = np.zeros(count, dtype=bool)
        if fixed is not None:
            np.logical_or.at(self.fixed, self.weld, np.asarray(fixed, dtype=bool))
        # How far out over the body each point is.
        near = body.nearest(points)
        height = self.height = np.where(near.found & ~near.beyond, near.distance, np.nan)
        # Over each vertex of the body: the first triangle of clothing its line out meets within COVER_REACH (not the
        # skin's, which lies on it; nor one along the line, the side of a strap), where that lies over this part of
        # the body: about as far from the body as from the vertex, not closer to another part (a sleeve on the arm,
        # seen from the side of a breast).
        self.over = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
        clothing = ~self.fixed[tris].all(axis=1)
        if clothing.any():
            cloth = np.flatnonzero(clothing)
            cast = BVHTree.FromPolygons(points.tolist(), tris[cloth].tolist(), all_triangles=True).ray_cast
            rows, found, distances = [], [], []
            for k, origin, ray in zip(range(len(body.co)), (body.co + 1e-6 * body.normals).tolist(),
                                      body.normals.tolist()):
                hit, facing, index, distance = cast(origin, ray, COVER_REACH)
                if hit is not None and abs(facing.dot(ray)) >= COVER_FACING:
                    rows.append(k)
                    found.append(cloth[index])
                    distances.append(distance)
            rows, found, distances = (np.array(rows, dtype=np.int64), np.array(found, dtype=np.int64),
                                      np.array(distances))
            if len(rows):
                lies = np.nan_to_num(height, nan=0.0)[tris[found]].mean(axis=1) >= COVER_OVER * distances
                self.over = rows[lies], found[lies], distances[lies]
        # The meshes' edges (welded), each both ways, for spreading pushes.
        pairs = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
        pairs = np.unique(pairs, axis=0)
        self.into = np.concatenate((pairs[:, 0], pairs[:, 1]))
        self.out_of = np.concatenate((pairs[:, 1], pairs[:, 0]))


def _kept(height):
    """How far over the body clothing that lay ``height`` over it is kept: as far, up to COVER_KEEP, less PUSH_SLACK."""
    return np.minimum(height, COVER_KEEP) - PUSH_SLACK


def keep_covered(found, moved, body, rigid=None):
    """Push clothing back out over the body after a move (Keep Covered).

    ``found`` (a _Cover) holds, from the source shape, the body under each point of the clothing and the clothing's
    triangle over each vertex of the body. ``moved`` are the points after the move and ``body`` the target shape (a
    BodySurface with the source's welding). The move and Keep Close keep the clothing's points out of the body, but
    not its faces: a low-poly top over a breast that grew pointier, or a nipple, lets the body out between its points.
    So a point that sank closer to the body than it lay (up to COVER_KEEP, less PUSH_SLACK) is pushed back out, and
    where a vertex of the body comes out through the triangle that covered it (inside its edges, seen along its
    normal), the triangle's corners are pushed out until it covers the vertex as it did, each by its share of the
    spot over the vertex. Pushes go along the body's normal, and are spread to the clothing around within
    COVER_SPREAD so it comes along smoothly; rigid parts (``rigid``, a label per point, -1 for none) move whole, as far
    as their point that needs it most, and fixed points (skin) never move. Only ever pushes out. Returns the new
    positions.
    """
    from .fitting import _barycentric
    weld, first, count, tris = found.weld, found.first, found.count, found.tris
    points = np.asarray(moved, dtype=np.float64)[first]
    still = found.fixed
    group = np.full(count, -1, dtype=np.int64)
    if rigid is not None:
        np.maximum.at(group, weld, np.asarray(rigid, dtype=np.int64))
    # Points that lay over the body, near enough to sink into it.
    on_body = np.flatnonzero(np.isfinite(found.height) & (found.height > 0.0) & (found.height < COVER_REACH) & ~still)
    body_kept = _kept(found.height[on_body])
    rows, over_tri, distances = found.over
    corners, under, outward = tris[over_tri], body.co[rows], body.normals[rows]
    over_kept = _kept(distances)
    total = np.zeros((count, 3))
    for _ in range(COVER_ROUNDS):
        now = points + total
        need = np.zeros(count)
        direction = np.zeros((count, 3))

        def offer(which, amount, way):
            """Push ``which`` by ``amount`` along ``way``, where that's more than they're pushed already."""
            more = amount > need[which]
            which, amount, way = which[more], amount[more], way[more]
            # Of repeats, the largest (assigned last) wins.
            order = np.argsort(amount)
            need[which[order]] = amount[order]
            direction[which[order]] = way[order]

        # Points sunk toward the body, from where it is closest now.
        near = body.nearest(now[on_body])
        offer(on_body, np.where(near.found & ~near.beyond, body_kept - near.distance, 0.0), near.normal)
        # Vertices of the body out through the triangles over them: where the line out along the vertex's normal meets
        # the triangle's plane (within its edges), how far over the vertex that is.
        if len(rows):
            a, b, c = (now[corners[:, k]] for k in range(3))
            unit = np.cross(b - a, c - a)
            unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-30)
            facing = np.einsum("ij,ij->i", unit, outward)
            across = np.abs(facing) >= COVER_FACING
            height = np.einsum("ij,ij->i", a - under, unit) / np.where(across, facing, 1.0)
            bary = _barycentric(under + height[:, None] * outward, a, b, c)
            within = (bary.min(axis=1) >= -COVER_EDGE) & across
            short = np.where(within, over_kept - height, 0.0)
            share = np.clip(bary, 0.0, None)
            share /= np.maximum(share.sum(axis=1, keepdims=True), 1e-12)
            share /= np.maximum((share * share).sum(axis=1, keepdims=True), 1e-12)
            for k in range(3):
                offer(corners[:, k], short * share[:, k], outward)
        need[still] = 0.0
        # A push farther than clothing is looked for is clothing lost, not body coming out.
        need[need > COVER_REACH] = 0.0
        pushed = need > 1e-6
        if not pushed.any():
            break
        push = np.where(pushed[:, None], need[:, None] * direction, 0.0)
        step = np.zeros((count, 3))
        # Rigid parts move whole: along their points' pushes, as far as the one that needs it most.
        members = np.flatnonzero((group >= 0) & pushed)
        if len(members):
            labels, owner = np.unique(group[members], return_inverse=True)
            owner = owner.reshape(-1)
            way = np.stack([np.bincount(owner, weights=push[members, k], minlength=len(labels)) for k in range(3)], 1)
            way /= np.maximum(np.linalg.norm(way, axis=1, keepdims=True), 1e-12)
            reach = need[members] / np.maximum(np.einsum("ij,ij->i", direction[members], way[owner]), 0.3)
            far = np.zeros(len(labels))
            np.maximum.at(far, owner, reach)
            whole = np.flatnonzero(np.isin(group, labels))
            step[whole] = (way * far[:, None])[np.searchsorted(labels, group[whole])]
        # The rest: each pushed point as far as it can without turning triangles over, the points around it along,
        # fading with the distance (a push held back isn't spread, or the clothing around would rise in its place).
        loose = pushed & (group < 0)
        if loose.any():
            own = _unflipped(tris, found.normals, now, np.where(loose[:, None], push, 0.0))
            loose &= np.einsum("ij,ij->i", own, push) > 1e-12
            if loose.any():
                step += _spread_pushes(found, own, loose, still | (group >= 0), now)
        total += _unflipped(tris, found.normals, now, step)
    return np.asarray(moved, dtype=np.float64) + total[weld]


def _unflipped(tris, normals, points, step):
    """``step`` (a move of ``points``), less where it would turn triangles over from how they faced (``normals``, on the
    source shape): the corners of such triangles move half as far, up to COVER_HALVINGS times, then not at all (pushed
    corners of a triangle spanning a crease can go two ways). Triangles the move turned over already don't count."""

    def turned(at):
        faces = np.cross(at[tris[:, 1]] - at[tris[:, 0]], at[tris[:, 2]] - at[tris[:, 0]])
        return np.einsum("ij,ij->i", normals, faces) <= 0.0
    # Triangles with no area have no side to turn to.
    sided = (np.einsum("ij,ij->i", normals, normals) > 1e-20) & ~turned(points)
    scale = np.ones(len(points))
    for halving in range(COVER_HALVINGS + len(tris)):
        over = sided & turned(points + step * scale[:, None])
        if not over.any():
            break
        corners = np.unique(tris[over])
        scale[corners] = scale[corners] * 0.5 if halving < COVER_HALVINGS else 0.0
    return step * scale[:, None]


def _spread_pushes(found, push, pushed, still, points):
    """``push`` (per welded point, at ``points``) at the ``pushed`` points, spread to the points around them within
    COVER_SPREAD (weighted along a smooth bump): those of the same piece of clothing (within COVER_RINGS edges of
    ``found``, a _Cover), and those of any piece lying farther out than the pushed point (a strap over a cup that's
    pushed out comes along), not those under it. ``still`` points take none."""
    from mathutils.kdtree import KDTree
    into, out_of, count = found.into, found.out_of, found.count
    piece = pushed.copy()
    for _ in range(COVER_RINGS):
        reached = np.bincount(into, weights=piece[out_of], minlength=count) > 0.0
        if not (reached & ~piece).any():
            break
        piece |= reached
    sources = np.flatnonzero(pushed)
    tree = KDTree(len(sources))
    for index, point in enumerate(points[sources].tolist()):
        tree.insert(point, index)
    tree.balance()
    # Only points within reach of the pushes' box are looked at.
    low = points[sources].min(axis=0) - COVER_SPREAD
    high = points[sources].max(axis=0) + COVER_SPREAD
    near = (~still | pushed) & (points >= low).all(axis=1) & (points <= high).all(axis=1)
    height = np.nan_to_num(found.height, nan=-np.inf)
    rows, cols, weights = [], [], []
    for row, point in zip(np.flatnonzero(near).tolist(), points[near].tolist()):
        for _co, index, distance in tree.find_range(point, COVER_SPREAD):
            source = sources[index]
            if piece[row] or height[row] >= height[source]:
                rows.append(row)
                cols.append(source)
                weights.append((1.0 - (distance / COVER_SPREAD) ** 2) ** 2)
    rows, cols, weights = np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(weights)
    # The pushes around each point, by their weights, as a share of one at most: a lone push fades out from its
    # point, and among many the point takes their average.
    total = np.maximum(np.bincount(rows, weights=weights, minlength=count), 1.0)
    result = np.stack([np.bincount(rows, weights=weights * push[cols, k], minlength=count) for k in range(3)], 1)
    result /= total[:, None]
    # A pushed point goes at least as far as it needs itself.
    own = np.einsum("ij,ij->i", result[pushed], push[pushed]) / np.maximum(np.einsum("ij,ij->i", push[pushed],
                                                                                     push[pushed]), 1e-30)
    result[pushed] += np.maximum(1.0 - own, 0.0)[:, None] * push[pushed]
    return result


def keep_rigid(before, after, groups, on_body=None):
    """Move each group of points (label >= 0 in ``groups``) from ``before`` the rigid way (rotation and
    translation) that comes closest to where they are in ``after``, in place.

    With ``on_body`` (a mask per point: those lying on the body), each group also scales by the share of its points
    on the body, uniformly, as much as the space it is in grows or shrinks (the least-squares scale). A chain link,
    a ring or a band on the body grows with it, so a chain doesn't come apart between its links, and a pendant
    hanging off it keeps its size."""
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
    turned = np.einsum("nij,nj->ni", turn[owner], p)
    if on_body is not None:
        spread = np.bincount(owner, weights=np.einsum("ij,ij->i", turned, turned), minlength=len(count))
        scale = np.bincount(owner, weights=np.einsum("ij,ij->i", turned, q), minlength=len(count))
        scale = np.where(spread > 1e-18, scale / np.maximum(spread, 1e-30), 1.0)
        share = np.bincount(owner, weights=np.asarray(on_body, dtype=np.float64)[members], minlength=len(count))
        scale = 1.0 + (share / count[:, 0]) * (scale - 1.0)
        turned *= scale[owner, None]
    after[members] = center_after[owner] + turned


def _touching(points, parts):
    """Pairs of rigid parts touching each other, closer than RIGID_TOUCH, for ``points`` of them (``parts``, from 0, per
    point)."""
    from .fitting import weld_groups
    if not len(points):
        return np.zeros((0, 2), dtype=np.int64)
    weld, groups = weld_groups(points, RIGID_TOUCH)
    first = np.zeros(groups, dtype=np.int64)
    first[weld[::-1]] = parts[::-1]
    pairs = np.stack((parts, first[weld]), axis=1)
    pairs = np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1)
    return np.unique(pairs, axis=0) if len(pairs) else np.zeros((0, 2), dtype=np.int64)


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


def is_skin_material(name):
    """Whether a material (by ``name``) is the game's skin (see SKIN_MATERIAL)."""
    return bool(SKIN_MATERIAL.search(name or ""))


def skin_vertices(obj):
    """Which vertices of mesh ``obj`` belong to faces drawn with the game's skin material."""
    mesh = obj.data
    skin = np.array([slot.material is not None and is_skin_material(slot.material.name)
                     for slot in obj.material_slots] + [False], dtype=bool)
    marked = np.zeros(len(mesh.vertices), dtype=bool)
    if not skin.any() or not len(mesh.polygons):
        return marked
    index = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("material_index", index)
    sizes = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_total", sizes)
    vertex = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", vertex)
    marked[vertex[np.repeat(skin[np.minimum(index, len(skin) - 1)], sizes)]] = True
    return marked


def skin_shares(points, co, tris, marked):
    """How much each of ``points`` (``marked`` as skin) follows the body (``co``, ``tris``) as skin (see
    ShapeChange.bind) rather than moving like clothing: fully within SKIN_NEAR of it, fading out by SKIN_FAR."""
    shares = np.zeros(len(points))
    marked = np.asarray(marked, dtype=bool)
    if marked.any():
        distance = _distances(np.asarray(points, dtype=np.float64)[marked], np.asarray(co, dtype=np.float64),
                              np.asarray(tris, dtype=np.int64).tolist())
        shares[marked] = 1.0 - _smoothstep(SKIN_NEAR, SKIN_FAR, distance)
    return shares


def is_skin(points, co, tris):
    """Whether a mesh (its vertices ``points``) is skin lying on the body (``co``, triangles ``tris``): SKIN_SHARE of
    its vertices within SKIN_DISTANCE of it, like a mod's copy of the body's skin. Such a mesh follows the body's
    surface exactly (see ShapeChange.bind)."""
    if not len(points):
        return False
    distance = _distances(np.asarray(points, dtype=np.float64), np.asarray(co, dtype=np.float64),
                          np.asarray(tris, dtype=np.int64).tolist())
    return bool((distance <= SKIN_DISTANCE).mean() >= SKIN_SHARE)


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
# A texture repeats, so a spot of the UV map a whole square away from another is the same spot (Bibo+ keeps its
# legs a square below where Neolithe keeps them): spots are compared within one square, and those this close to
# its sides are looked for across them too, in triangles reaching over from the next square.
UV_WRAP_MARGIN = 0.1
# Islands of the UV map that the other body lacks altogether (less than this share of their vertices found, like
# a gen3 body's extra crotch detail in texture space no other layout uses) don't count against MIN_MATCHED, as
# long as they're at most this share of the body; the parts of the body around them carry the clothing there.
ISLAND_FOUND = 0.05
MAX_MISSING_ISLANDS = 1.0 / 3.0


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
    # A triangle without area in the UV map gives its three corners the same weight (see `_along_flat` for
    # what match_by_uv makes of one).
    v = np.where(flat, 1.0 / 3.0, (d22 * p1 - d12 * p2) / denominator)
    w = np.where(flat, 1.0 / 3.0, (d11 * p2 - d12 * p1) / denominator)
    return np.clip(np.column_stack((1.0 - v - w, v, w)), 0.0, 1.0)


def _flat_in_uv(corners):
    """Which triangles (``corners``, n x 3 x 2 UVs) have no area in the UV map: lines, or a point."""
    e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    cross = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    size = np.maximum(np.einsum("ij,ij->i", e1, e1), np.einsum("ij,ij->i", e2, e2))
    return cross <= 1e-9 * np.maximum(size, 1e-30)


def _along_flat(spot, corners, points, near):
    """Barycentric weights of the UV ``spot`` in a triangle with no area in the UV map (``corners``, its
    corners' UVs, ``points`` their positions), whose spot is closest to the position ``near``.

    Such a triangle is a line in the UV map (two corners sharing a UV: Neolithe's neck opening is a ring of
    them), and a spot on it is a spot of each of its sides that pass there: on the side between the two corners
    sharing a UV, anywhere. Equal weights would put a vertex in the triangle's middle, several centimetres off.
    """
    best, weights = np.inf, np.full(3, 1.0 / 3.0)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        side = corners[b] - corners[a]
        length = float(side @ side)
        if length <= 1e-18:
            # Both ends at one spot of the UV map: the spot is anywhere along the side.
            offset, span = points[b] - points[a], float((points[b] - points[a]) @ (points[b] - points[a]))
            ts = (min(max(float((near - points[a]) @ offset) / span, 0.0), 1.0),) if span > 0.0 else (0.0,)
        else:
            ts = (min(max(float((spot - corners[a]) @ side) / length, 0.0), 1.0),)
        for t in ts:
            candidate = np.zeros(3)
            candidate[a], candidate[b] = 1.0 - t, t
            off = float(np.linalg.norm(candidate @ points - near))
            if off < best:
                best, weights = off, candidate
    return weights


def _flat_strips(tri_uv):
    """Strips UV_TOLERANCE wide around the sides of the triangles without area in the UV map (``tri_uv``, n x 3 x 2),
    two triangles each (m x 3 x 2), and the triangle each belongs to."""
    flat = np.flatnonzero(_flat_in_uv(tri_uv))
    if not len(flat):
        return np.zeros((0, 3, 2)), np.zeros(0, dtype=np.int64)
    start = tri_uv[flat].reshape(-1, 2)
    end = tri_uv[flat][:, [1, 2, 0]].reshape(-1, 2)
    along = end - start
    length = np.linalg.norm(along, axis=1, keepdims=True)
    # A side whose ends share a spot is a point: a square around it.
    along = np.where(length > 1e-12, along / np.maximum(length, 1e-30), (1.0, 0.0)) * (0.5 * UV_TOLERANCE)
    across = along[:, ::-1] * (-1.0, 1.0)
    a, b = start - along - across, end + along - across
    c, d = end + along + across, start - along + across
    strips = np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))
    owner = np.tile(np.repeat(flat, 3), 2)
    return strips, owner


def _wrapped_triangles(tri_uv):
    """``tri_uv`` (n x 3 x 2) each moved by whole squares of the UV map into the square at its middle's spot."""
    return tri_uv - np.floor(tri_uv.mean(axis=1))[:, None, :]


def _wrapped_spots(u, v):
    """The spots to look for UV (``u``, ``v``) at: in the first square, and across its sides near them (see
    UV_WRAP_MARGIN)."""
    u, v = u - math.floor(u), v - math.floor(v)
    us = (u,) + ((u + 1.0,) if u < UV_WRAP_MARGIN else ()) + ((u - 1.0,) if u > 1.0 - UV_WRAP_MARGIN else ())
    vs = (v,) + ((v + 1.0,) if v < UV_WRAP_MARGIN else ()) + ((v - 1.0,) if v > 1.0 - UV_WRAP_MARGIN else ())
    return [(a, b) for a in us for b in vs]


def match_by_uv(uv, co, tri_verts, tri_uv, other_co):
    """Where each vertex (its spot ``uv`` in the UV map, nan for none, and position ``co``) is on another
    mesh with triangles ``tri_verts``, their corners' UVs ``tri_uv`` and positions ``other_co``.

    Returns the triangle and its barycentric weights per vertex; triangle -1 where no triangle is
    within UV_TOLERANCE of its spot, or the spot lands more than MAX_MATCH_DISTANCE away. Spots are
    compared within one square of the UV map (see UV_WRAP_MARGIN). Where islands of the UV map overlap
    (hands and body can share texture space), the island landing closest to the vertex wins.
    """
    island = _uv_islands(tri_verts, tri_uv)
    tri_uv = _wrapped_triangles(tri_uv)
    # Triangles without area in the UV map are lines in it, which a BVH tree never finds: each side is looked for
    # as a thin strip around it instead (see `_flat_strips`).
    strips, owner = _flat_strips(tri_uv)
    searched = np.concatenate((tri_uv, strips))
    flat = np.concatenate((searched, np.zeros(searched.shape[:2] + (1,))), axis=2).reshape(-1, 3)
    find = BVHTree.FromPolygons(flat.tolist(), np.arange(len(flat)).reshape(-1, 3).tolist(),
                                all_triangles=True).find_nearest_range
    owner = np.concatenate((np.arange(len(tri_uv)), owner))
    vertex, tri, spot, gap = [], [], [], []
    for k, (u, v) in enumerate(uv.tolist()):
        if u != u:
            continue
        for au, av in _wrapped_spots(u, v):
            for location, _normal, index, distance in find((au, av, 0.0), UV_TOLERANCE):
                vertex.append(k)
                tri.append(index)
                spot.append((location[0], location[1]))
                gap.append(distance)
    triangle = np.full(len(uv), -1, dtype=np.int64)
    weights = np.zeros((len(uv), 3))
    if not vertex:
        return triangle, weights
    vertex, tri, gap = np.array(vertex), owner[np.array(tri)], np.array(gap)
    spot = np.array(spot)
    bary = _barycentric(spot, tri_uv[tri])
    for row in np.flatnonzero(_flat_in_uv(tri_uv[tri])):
        bary[row] = _along_flat(spot[row], tri_uv[tri[row]], other_co[tri_verts[tri[row]]], co[vertex[row]])
    miss = np.linalg.norm(np.einsum("ij,ijk->ik", bary, other_co[tri_verts[tri]]) - co[vertex], axis=1)
    # For each vertex and island the triangles closest in the UV map, then of those the landing closest. Where
    # several are as close (an island meets itself across a side of the square), they're all landings.
    island = island[tri]
    pair = np.unique(np.stack((vertex, island), axis=1), axis=0, return_inverse=True)[1].reshape(-1)
    closest = np.full(pair.max() + 1, np.inf)
    np.minimum.at(closest, pair, gap)
    best = np.flatnonzero(gap <= closest[pair] + 1e-7)
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


def rest_shape(obj, values=None, groups=None):
    """Vertex positions (object space) of mesh ``obj`` shaped by its shape keys, before any modifier.

    Keys are mixed at their current values, like Blender does, except for those named in ``values``
    (name -> value), which are used at the value given there even if they're muted. ``groups`` (a dict)
    keeps the weights of the vertex groups keys are limited to, read for one shape, for the next (of
    ``obj`` with the same weights).
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
        weights = None
        if block.vertex_group:
            if groups is None:
                weights = _group_weights(obj, block.vertex_group)
            else:
                if block.vertex_group not in groups:
                    groups[block.vertex_group] = _group_weights(obj, block.vertex_group)
                weights = groups[block.vertex_group]
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

    def change(self, radius=DEFAULT_RADIUS, bridge=0.0, shared=None):
        """The ShapeChange between the two shapes (``shared``: see ShapeChange)."""
        if self.match is None:
            return ShapeChange(self.source, self.target, self.tris, radius=radius, bridge=bridge, shared=shared)
        return ShapeChange(self.source, self.target, self.tris, radius=radius, bridge=bridge,
                           to_body=(self.to_co, self.to_tris), known=self.match.known, shared=shared)


def body_shapes(body, from_key, to_body, to_key, match='AUTO', to_values=None, shapes=None):
    """The From and To shapes of the body (a BodyShapes, on the vertices of ``body``).

    From is ``body`` with its shape key ``from_key`` fully on, To is ``to_body`` (``body`` when None)
    with ``to_key`` fully on; empty names add no key. ``to_values`` (key name -> value, like
    `devkit_values` gives them) set more keys of the To shape. The other shape keys stay at their current
    values, except that when both are the same object, the key picked for one shape is off in the other.
    A To body with other vertices is matched to ``body`` as ``match`` says (see `matching`): by their UV
    maps, keeping only the vertices of ``body`` found on it, or by shape. Matched by vertices or UV map,
    the To shape is taken in its own object space and placed like ``body``, so the To body can stand
    anywhere; matched by shape, the bodies must lie on each other. ``shapes`` are the From and To shapes as
    `rest_shape` gives them (in their objects' space), when they've been read already.
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
    if shapes is not None:
        source, target = shapes
    else:
        source, target = rest_shape(body, source_values), rest_shape(other, target_values)
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


def _matched_share(found, mapped, tris, tri_uv):
    """The share of the vertices with a spot in the UV map (``mapped``) that were ``found`` on the other body,
    leaving out the islands of the UV map (of triangles ``tris``, their corners' UVs ``tri_uv``) it lacks
    altogether, while those are at most MAX_MISSING_ISLANDS of them (see ISLAND_FOUND)."""
    total = max(int(mapped.sum()), 1)
    share = found.sum() / total
    if share >= MIN_MATCHED or not len(tris):
        return share
    island = np.full(len(found), -1, dtype=np.int64)
    tri_island = _uv_islands(tris, tri_uv)
    for k in range(3):
        island[tris[:, k]] = tri_island
    counted = mapped & (island >= 0)
    members = np.bincount(island[counted], minlength=int(tri_island.max()) + 1)
    hits = np.bincount(island[counted & found], minlength=len(members))
    missing = (hits < ISLAND_FOUND * members)[np.maximum(island, 0)] & counted
    if missing.sum() > MAX_MISSING_ISLANDS * total:
        return share
    return found[mapped & ~missing].sum() / max(int((mapped & ~missing).sum()), 1)


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
    share = _matched_share(found, ~np.isnan(uv[:, 0]), tris, corners[_tris(body.data, "loops")])
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
    return _put_key(obj, None, name, world_co, show=True)


def _put_key(obj, block, name, world_co, show):
    """Put ``world_co`` into shape key ``block`` of ``obj`` (None: a new one named ``name``), relative to the basis.
    With ``show``, turn it fully on and make it active, and turn the keys that shaped the mesh before off; returns
    (the key, those that can't be: driven ones)."""
    before = contributing_keys(obj)
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    if block is None:
        block = obj.shape_key_add(name=name, from_mix=False)
        block.value = 0.0
    inverse = np.linalg.inv(np.array(obj.matrix_world, dtype=np.float64))
    block.data.foreach_set("co", _transform(world_co, inverse).astype(np.float32).ravel())
    block.relative_key = obj.data.shape_keys.reference_key
    stuck = []
    if show:
        block.value = 1.0
        block.mute = False
        for other in before:
            if other == block:
                continue
            if _driven(obj.data.shape_keys, other, "value"):
                stuck.append(other.name)
            else:
                other.value = 0.0
        obj.active_shape_key_index = list(obj.data.shape_keys.key_blocks).index(block)
    obj.data.update()
    return block, stuck


# -----------------------------------------------------------------------------
# Resized keys

# Each shape key Resize makes remembers what it was made from, in its Key's custom properties under this name:
# the size it's for ("label"), fingerprints of the From body's shape ("from"), of everything the result depends
# on ("inputs") and of the key as made ("result"), the mesh's vertex count ("count"), and the values of the mesh's
# other keys that gave the shape it was resized from ("mix").
RECORD = "magic_fit_resized"


def fingerprint(*parts):
    """A short hash of ``parts``: arrays (as float32, as shape keys keep them) and anything else by its repr."""
    digest = hashlib.blake2b(digest_size=12)
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(np.ascontiguousarray(part, dtype=np.float32).tobytes())
        else:
            digest.update(repr(part).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def resized_keys(obj):
    """{key name: record} of the shape keys of mesh ``obj`` that Resize made (see RECORD), those still there."""
    keys = obj.data.shape_keys if obj.type == 'MESH' else None
    stored = keys.get(RECORD) if keys is not None else None
    if stored is None or not hasattr(stored, "to_dict"):
        return {}
    return {name: record for name, record in stored.to_dict().items()
            if isinstance(record, dict) and keys.key_blocks.get(name) is not None}


def resized_key(obj, label):
    """The shape key of ``obj`` Resize made for the size ``label``, or None."""
    for name, record in resized_keys(obj).items():
        if record.get("label") == label:
            return obj.data.shape_keys.key_blocks[name]
    return None


def note_edited_keys(obj):
    """Note on the keys Resize made on ``obj`` that were changed by hand since, before the mesh gets other vertices
    (Subdivide for Sizes): then it can't be told anymore, and they stay as they are (see key_state)."""
    records = resized_keys(obj)
    noted = False
    for name, record in records.items():
        if (record.get("count") == len(obj.data.vertices) and not record.get("edited") and
                fingerprint(_key_positions(obj.data.shape_keys.key_blocks[name])) != record.get("result")):
            record["edited"] = True
            noted = True
    if noted:
        obj.data.shape_keys[RECORD] = records


def key_state(obj, label, inputs):
    """Whether the key Resize made for ``label`` on ``obj`` can stay, for a resize whose inputs have the fingerprint
    ``inputs``: (the key or None, 'NEW' without one, 'FRESH' when made from the same inputs, 'STALE' when made from
    others and not changed since, 'EDITED' when made from others and changed by hand since). Once the mesh has other
    vertices, a hand change only shows where `note_edited_keys` noted it before."""
    block = resized_key(obj, label)
    if block is None:
        return None, 'NEW'
    record = resized_keys(obj)[block.name]
    if record.get("count") != len(obj.data.vertices):
        return block, 'EDITED' if record.get("edited") else 'STALE'
    if record.get("inputs") == inputs:
        return block, 'FRESH'
    return block, 'STALE' if fingerprint(_key_positions(block)) == record.get("result") else 'EDITED'


def from_values(obj, from_print):
    """The shape key values giving ``obj`` the shape it fits the From body in, the body's shape having the
    fingerprint ``from_print``: its keys as they are, but those Resize made from that same shape off and, when one
    of those is on (the mesh shows a size), the others as they were when it was made. A key made from another
    shape (a resize before, to the body it fits now) counts as it is.

    Returns ({key name: value}, for rest_shape; the values of the other keys that aren't 0: the mix to remember).
    """
    keys = obj.data.shape_keys if obj.type == 'MESH' else None
    if keys is None:
        return {}, {}
    records = resized_keys(obj)
    ours = [name for name, record in records.items() if record.get("from") == from_print]
    values = {name: 0.0 for name in ours}
    showing = [name for name in ours if not keys.key_blocks[name].mute and keys.key_blocks[name].value != 0.0]
    if showing:
        values.update(records[showing[0]].get("mix", {}))
    mix = {}
    for block in keys.key_blocks:
        if block == keys.reference_key or block.name in ours:
            continue
        value = values.get(block.name, 0.0 if block.mute else block.value)
        if value:
            mix[block.name] = float(value)
    return values, mix


def from_shape(obj, from_print):
    """World space positions of ``obj`` in the shape it fits the From body in (see from_values), and the mix that
    gives it."""
    values, mix = from_values(obj, from_print)
    return _transform(rest_shape(obj, values), obj.matrix_world), mix


def write_resized_key(obj, label, world_co, record, show=False):
    """Put ``world_co`` (world space) into the shape key Resize made for the size ``label`` on ``obj``, else into a
    new key named ``label`` (Blender makes it unique), and remember ``record`` for it (see RECORD). With ``show``,
    the key is turned fully on and made active, and the keys that shaped the mesh before are turned off, so it
    shows the new shape alone. Returns (the key, keys that couldn't be turned off: driven ones)."""
    block, stuck = _put_key(obj, resized_key(obj, label), label, world_co, show)
    keys = obj.data.shape_keys
    records = resized_keys(obj)
    records[block.name] = dict(record, label=label, count=len(obj.data.vertices),
                               result=fingerprint(_key_positions(block)))
    keys[RECORD] = records
    return block, stuck


def remove_resized_keys(obj):
    """Remove the shape keys Resize made on mesh ``obj`` (see RECORD), those changed by hand since too, and their
    records. When one of them shows, the mesh's other keys go back to the values they had when it was made, so the
    mesh shows the shape it was resized from. Keys nothing drives only; the basis stays. Must be out of Edit Mode.
    Returns (keys removed, how many of them were changed by hand)."""
    records = resized_keys(obj)
    keys = obj.data.shape_keys
    if not records:
        if keys is not None and RECORD in keys:
            del keys[RECORD]
        return 0, 0
    blocks = keys.key_blocks
    edited = sum(1 for name, record in records.items()
                 if record.get("edited") or (record.get("count") == len(obj.data.vertices) and
                                             fingerprint(_key_positions(blocks[name])) != record.get("result")))
    showing = next((name for name in records if not blocks[name].mute and blocks[name].value != 0.0), None)
    if showing is not None:
        for name, value in records[showing].get("mix", {}).items():
            block = blocks.get(name)
            if block is not None and name not in records and not _driven(keys, block, "value"):
                block.value = value
    for name in records:
        obj.shape_key_remove(blocks[name])
    keys = obj.data.shape_keys
    if keys is not None and RECORD in keys:
        del keys[RECORD]
    obj.active_shape_key_index = 0
    obj.data.update()
    return len(records), edited


def show_key(obj, block):
    """Turn shape key ``block`` of ``obj`` fully on and the others that shape the mesh off, and make it active.
    Returns the keys that couldn't be turned off (driven ones)."""
    stuck = []
    for other in contributing_keys(obj):
        if other == block:
            continue
        if _driven(obj.data.shape_keys, other, "value"):
            stuck.append(other.name)
        else:
            other.value = 0.0
    block.value = 1.0
    block.mute = False
    obj.active_shape_key_index = list(obj.data.shape_keys.key_blocks).index(block)
    obj.data.update()
    return stuck


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
