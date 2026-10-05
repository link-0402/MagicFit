# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoothing weights: an averaging brush that works on every bone group at once and reaches across gaps.

Blender's own weight Smooth brush averages one vertex group at a time, along the edges of the mesh. A
garment imported from FFXIV is full of places where that stops: its parts are loose (sleeves, panels,
collars, layers), its seams are split, and a smoothed bone leaves the others as they were. This brush
smooths the whole bone weight of a vertex, and treats parts that touch as one mesh.

The mesh becomes a graph of nodes (vertices at the same spot, the two sides of a split seam, are one
node). Its edges are the mesh's edges, with the cotangent weights of a Laplacian so that irregular
triangles don't bias the average. Weights are averaged by letting them diffuse over that graph for a time
of Reach squared, with the area around each node as its mass, so Reach is a distance over the surface and
works the same on dense and coarse meshes: what counts for a vertex is the weights within about Reach of
it, the nearer the more. That is a single implicit heat step, solved for the region around the dab with a
sparse factorization (SciPy, which Weight Transfer needs too) or, where SciPy isn't there, with conjugate
gradients, which slow down as Reach grows over the mesh's edges.

Across gaps, links hold the mesh together wherever surfaces lie within the Merge Range of each other:
each node links to the closest point of every other sheet near it (see `fitting.mesh_sheets`), shared out
among the corners of the triangle there, so a dense layer over a coarse one holds on everywhere, not only
where vertices happen to meet. Sheets of other loose parts count (a sleeve against a torso, a belt loop on a
waistband), and so do the folds of a node's own part (a hem folded back, a cuff, a lining), unless
`join_folds` is off: then lips and eyelids that touch can still open. Where one smooth sheet comes close to
itself (trouser legs at the crotch, fingers), nothing links, so it still moves apart. A link holds as hard
as `BIND` times the node's area, whatever Reach is: layers in contact end up with the same weights, as if
merged, so they move together however the bones scale, even when Reach is too small to even out anything
else.

What diffuses is each vertex's share of every group (its bone weights, as fractions of their sum) times how
much it has, so vertices without weights add nothing to their neighbours and take on the weights around
them. A vertex keeps its own total. Locked groups are never changed; Current Group mode only moves the
active group's share, the others make room for it (see `goals.GoalStroke`). Weights below `CUTOFF` are
dropped and only the largest `max_groups` are kept.

Without Accumulate a stroke moves each vertex at most Strength of the way to the weights it started with
smoothed, so the result doesn't depend on how the brush got there: a second stroke smooths further. With
Accumulate every dab smooths the weights the last ones left.

Like `painting`, nothing here depends on the 3D view.
"""

import numpy as np
from mathutils.bvhtree import BVHTree

from . import fitting, goals, hairing, straighten
from .goals import DEFAULT_MAX_GROUPS
from .painting import MIN_CHANGE, deform_bone_names
from .relaxing import _closest_on_triangles
from .robust_transfer import dependencies

# The weights around a vertex count out to this many times Reach; beyond that, the region a dab looks at
# ends (the influence of weights farther away is small, and the farther the region goes, the slower a dab).
# With Accumulate, where nothing has to add up across dabs, it ends sooner.
MARGIN = 3.0
MARGIN_ACCUMULATING = 2.0
# Shares of a smoothed vertex below this are dropped: FFXIV stores weights in bytes, and smaller ones
# only take up one of the eight slots.
CUTOFF = 0.002
# A node links across gaps to at most this many sheets (or, without triangles, nodes of other parts).
MAX_LINKS = 4
# A link holds a node to the surface across the gap with this many times the node's area, whatever Reach
# is: two layers linked all over end up within about 1 / (1 + 2 * BIND) of each other's weights.
BIND = 50.0
# Past the Merge Range, links fade out over this share of it, so a surface drifting away lets go gradually.
FADE = 0.25
# Cotangent weights are kept between these multiples of their mean: a skinny triangle would make the
# system stiff, and an obtuse one would make it negative (weights would leave the range of their neighbours).
MIN_CONDUCTANCE = 1e-3
MAX_CONDUCTANCE = 8.0
# The area around a node is at least this share of the median, so single points and wires still mix.
MIN_MASS = 0.02
# Conjugate gradients, when SciPy isn't there.
SOLVE_ITERATIONS = 150
SOLVE_TOLERANCE = 1e-5
# Solve with SciPy's sparse LU where it can be imported (the tests turn it off to try the fallback).
USE_SCIPY = True
_direct = []  # [(scipy.sparse, splu)] once imported, [None] when there's no SciPy


def _direct_solver():
    """(scipy.sparse, scipy.sparse.linalg.splu), or None when SciPy can't be used."""
    if not USE_SCIPY:
        return None
    if not _direct:
        try:
            dependencies.add_dll_directories()
            from scipy import sparse
            from scipy.sparse.linalg import splu
            _direct.append((sparse, splu))
        except ImportError:
            _direct.append(None)
    return _direct[0]


class SmoothError(Exception):
    """Why the smoothing can't be set up; the message is meant for the user."""


def smoothed_groups(obj):
    """Indices of the vertex groups of ``obj`` that Smooth changes: the deforming bones' when an armature
    deforms it, else every group but the ones that mask something (see `goals.mask_groups`)."""
    bones = deform_bone_names(obj)
    if bones:
        return {group.index for group in obj.vertex_groups if group.name in bones}
    masks = goals.mask_groups(obj)
    return {group.index for group in obj.vertex_groups if group.name not in masks}


# -----------------------------------------------------------------------------
# Compressed rows: ``values[pointers[k]:pointers[k + 1]]`` belong to row k

def _rows(keys, count):
    """Row pointers and the order of the entries that put them in rows by ``keys``."""
    order = np.argsort(keys, kind='stable')
    pointers = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=count), out=pointers[1:])
    return pointers, order


def _span(pointers, keys):
    """Positions of the entries of rows ``keys``, and for each the position in ``keys`` of its row."""
    starts = pointers[keys]
    lengths = pointers[keys + 1] - starts
    owner = np.repeat(np.arange(len(keys)), lengths)
    offsets = np.arange(len(owner)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return starts[owner] + offsets, owner


def close_pairs(points, radius):
    """Pairs of ``points`` (n, 3) at most ``radius`` apart, as index arrays (first, second), first < second."""
    empty = np.zeros(0, dtype=np.int64)
    if len(points) < 2 or radius <= 0.0:
        return empty, empty
    cell = np.floor((points - points.min(axis=0)) / radius).astype(np.int64) + 1
    span = cell.max(axis=0) + 2
    strides = (int(span[1] * span[2]), int(span[2]), 1)
    key = cell[:, 0] * strides[0] + cell[:, 1] * strides[1] + cell[:, 2]
    order = np.argsort(key, kind='stable')
    ordered = key[order]
    everyone = np.arange(len(points))
    firsts, seconds = [], []
    # Each pair of neighbouring cells once (the offsets that point "forward"), and each cell with itself.
    for dx in (0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                offset = dx * strides[0] + dy * strides[1] + dz
                if offset < 0:
                    continue
                low = np.searchsorted(ordered, key + offset, side='left')
                high = np.searchsorted(ordered, key + offset, side='right')
                count = high - low
                total = int(count.sum())
                if not total:
                    continue
                first = np.repeat(everyone, count)
                second = order[np.arange(total) - np.repeat(np.cumsum(count) - count, count) + np.repeat(low, count)]
                if offset == 0:
                    keep = second > first
                    first, second = first[keep], second[keep]
                delta = points[first] - points[second]
                near = np.einsum("ij,ij->i", delta, delta) <= radius * radius
                firsts.append(first[near])
                seconds.append(second[near])
    if not firsts:
        return empty, empty
    first, second = np.concatenate(firsts), np.concatenate(seconds)
    return np.minimum(first, second), np.maximum(first, second)


def _read_edges(mesh):
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    return edges.reshape(-1, 2)


def _read_triangles(mesh):
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris)
    return tris.reshape(-1, 3)


# -----------------------------------------------------------------------------
# The graph

def _fade(distance, merge_range):
    """How hard links across gaps of ``distance`` hold (0..1): fully up to ``merge_range``, then less and
    less, not at all `FADE` of it farther."""
    return goals.smoothstep((merge_range * (1.0 + FADE) - distance) / (FADE * merge_range))


def _ranks(keys, distance):
    """Each entry's place (0, 1...) among the entries with the same key, nearest first."""
    order = np.lexsort((distance, keys))
    starts = np.flatnonzero(np.diff(keys[order], prepend=-1))
    rank = np.empty(len(keys), dtype=np.int64)
    rank[order] = np.arange(len(keys)) - np.repeat(starts, np.diff(starts, append=len(keys)))
    return rank


class SmoothGraph:
    """The nodes of a mesh, the edges and links between them and the area around each.

    ``rest`` are the node positions (n, 3), ``edges`` (m, 2) and ``tris`` (t, 3) hold node indices.
    Surfaces within ``merge_range`` of each other are linked across the gap (see the module docs and
    `_surfaces`): other loose parts (parts are what the edges and triangles connect), and with
    ``join_folds`` the folds of a node's own part. Nodes without triangles link to the nearest of
    them in other parts.

    ``pointers`` / ``neighbours`` / ``length`` are the edges and links in compressed rows, both ways round;
    ``conductance`` how much an edge passes on per time (0 for links) and ``binding`` how hard a link
    holds (0 for edges); ``mass`` the area around each node; ``parts`` the loose part of each node;
    ``links`` the number of links across gaps (a node's link to the closest point of a sheet counts once).
    """

    def __init__(self, rest, edges, tris, merge_range, join_folds=True):
        count = len(rest)
        self.count = count
        edges = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
        edges = edges[edges[:, 0] != edges[:, 1]]
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]

        # Cotangent weights where there are triangles, the mean where there are only edges.
        cotangents = straighten.cotangent_weights(rest, tris) if len(tris) else (edges[:0], np.zeros(0))
        keys = np.unique(np.concatenate((edges[:, 0] * count + edges[:, 1],
                                         cotangents[0][:, 0] * count + cotangents[0][:, 1])))
        pairs = np.stack((keys // count, keys % count), axis=1)
        mean = float(cotangents[1].mean()) if len(cotangents[1]) else 1.0
        conductance = np.full(len(keys), mean)
        conductance[np.searchsorted(keys, cotangents[0][:, 0] * count + cotangents[0][:, 1])] = cotangents[1]
        conductance = np.clip(conductance, MIN_CONDUCTANCE * mean, MAX_CONDUCTANCE * mean)

        self.parts = hairing.components(count, pairs)
        area = hairing.vertex_areas(rest, tris)
        self.mass = np.maximum(area, MIN_MASS * (float(np.median(area[area > 0.0])) if (area > 0.0).any() else 1e-6))

        link_a, link_b, binding = self._links(rest, tris, merge_range, join_folds)
        pair_a = np.concatenate((pairs[:, 0], link_a))
        pair_b = np.concatenate((pairs[:, 1], link_b))
        pair_conductance = np.concatenate((conductance, np.zeros(len(link_a))))
        pair_binding = np.concatenate((np.zeros(len(pairs)), binding))
        length = np.linalg.norm(rest[pair_a] - rest[pair_b], axis=1)
        rows = np.concatenate((pair_a, pair_b))
        self.pointers, order = _rows(rows, count)
        self.neighbours = np.concatenate((pair_b, pair_a))[order]
        self.conductance = np.concatenate((pair_conductance, pair_conductance))[order]
        self.binding = np.concatenate((pair_binding, pair_binding))[order]
        self.length = np.concatenate((length, length))[order]

    def _links(self, rest, tris, merge_range, join_folds):
        """The links across gaps: (first nodes, second nodes, how hard each holds)."""
        empty = np.zeros(0, dtype=np.int64)
        self.links = 0
        if merge_range <= 0.0 or not self.count:
            return empty, empty, np.zeros(0)
        reach = merge_range * (1.0 + FADE)
        firsts, seconds, holds = [], [], []
        if len(tris):
            # Each node to the closest point of each sheet across a gap, shared out among its corners.
            nodes, corners, bary, distance = self._surfaces(rest, tris, reach, join_folds)
            hold = BIND * self.mass[nodes] * _fade(distance, merge_range)
            for k in range(3):
                use = (bary[:, k] > 0.0) & (hold > 0.0)
                firsts.append(nodes[use])
                seconds.append(corners[use, k])
                holds.append(hold[use] * bary[use, k])
            self.links += int((hold > 0.0).sum())
        # Nodes without triangles (points, wires) have no surface: they link to the nearest such nodes of
        # other parts, as hard as the smaller of them, shared out among their links.
        bare = np.ones(self.count, dtype=bool)
        bare[tris.reshape(-1)] = False
        if bare.sum() > 1:
            a, b = close_pairs(rest, reach)
            use = bare[a] & bare[b] & (self.parts[a] != self.parts[b])
            a, b = a[use], b[use]
            distance = np.linalg.norm(rest[a] - rest[b], axis=1)
            rank = _ranks(np.concatenate((a, b)), np.concatenate((distance, distance)))
            keep = (rank[:len(a)] < MAX_LINKS) & (rank[len(a):] < MAX_LINKS)
            a, b, distance = a[keep], b[keep], distance[keep]
            shared = np.bincount(np.concatenate((a, b)), minlength=self.count)
            hold = (BIND * np.minimum(self.mass[a], self.mass[b]) * _fade(distance, merge_range)
                    / np.maximum(np.maximum(shared[a], shared[b]), 1))
            use = hold > 0.0
            firsts.append(a[use])
            seconds.append(b[use])
            holds.append(hold[use])
            self.links += int(use.sum())
        if not firsts:
            return empty, empty, np.zeros(0)
        return np.concatenate(firsts), np.concatenate(seconds), np.concatenate(holds)

    def _surfaces(self, rest, tris, reach, join_folds):
        """For each node, the closest points within ``reach`` of up to `MAX_LINKS` sheets across gaps from
        it, nearest first: (nodes, their triangles' corners (k, 3), barycentric coordinates there (k, 3),
        distances).

        Sheets are the mesh's surfaces as `fitting.mesh_sheets` tells them apart: a fold tighter than
        ``fitting.FOLD_RADIUS`` or sharper than 90 degrees (a hem, a cuff, the rim of a lining) ends one, so
        the layers it joins are sheets of their own, while one smooth surface (both sides of a double-sided
        one) stays one sheet wherever it comes close to itself. The sheets a node lies on are never across a
        gap from it, other sheets of its own part only with ``join_folds``.
        """
        count = self.count
        sheet, sheets = fitting.mesh_sheets(rest, tris)
        tree = BVHTree.FromPolygons(rest.tolist(), tris.tolist(), all_triangles=True)
        found, hits = [], []
        for point in rest.tolist():
            near = tree.find_nearest_range(point, reach)
            found.append(len(near))
            hits.extend([hit[2] for hit in near])
        node = np.repeat(np.arange(count), found)
        tri = np.array(hits, dtype=np.int64)
        lying_on = np.unique(tris.T.reshape(-1) * sheets + np.tile(sheet, 3))
        key = node * sheets + sheet[tri]
        keep = lying_on[np.minimum(np.searchsorted(lying_on, key), len(lying_on) - 1)] != key
        node, tri = node[keep], tri[keep]
        corners = tris[tri]
        point, bary, _plane, _flat = _closest_on_triangles(
            rest[node], rest[corners[:, 0]], rest[corners[:, 1]], rest[corners[:, 2]])
        distance = np.linalg.norm(point - rest[node], axis=1)
        keep = distance <= reach
        if not join_folds:
            keep &= self.parts[corners[:, 0]] != self.parts[node]
        node, tri, corners, bary, distance = node[keep], tri[keep], corners[keep], bary[keep], distance[keep]
        # The closest point of each sheet, of the nearest sheets.
        first = _ranks(node * sheets + sheet[tri], distance) == 0
        node, corners, bary, distance = node[first], corners[first], bary[first], distance[first]
        first = _ranks(node, distance) < MAX_LINKS
        return node[first], corners[first], bary[first], distance[first]

    def region(self, nodes, radius):
        """The nodes within ``radius`` (along the links) of ``nodes``, in order."""
        reached = np.full(self.count, np.inf)
        frontier = np.unique(nodes)
        reached[frontier] = 0.0
        while len(frontier):
            positions, owner = _span(self.pointers, frontier)
            around = self.neighbours[positions]
            distance = reached[frontier][owner] + self.length[positions]
            better = (distance <= radius) & (distance < reached[around])
            if not better.any():
                break
            np.minimum.at(reached, around[better], distance[better])
            frontier = np.unique(around[better])
        return np.flatnonzero(np.isfinite(reached))

    def smooth(self, region, values, reach):
        """``values`` (a row per node of ``region``, sorted) after diffusing over the edges for a time of
        ``reach`` squared: every row becomes an average of the rows around it, with weights that fall off
        over about ``reach``, and rows linked across gaps end up alike whatever ``reach`` is. Nodes outside
        ``region`` don't take part."""
        size = len(region)
        local = np.full(self.count, -1, dtype=np.int64)
        local[region] = np.arange(size)
        positions, owner = _span(self.pointers, region)
        inside = local[self.neighbours[positions]] >= 0
        positions, owner = positions[inside], owner[inside]
        columns = local[self.neighbours[positions]]
        weight = reach * reach * self.conductance[positions] + self.binding[positions]
        mass = self.mass[region]
        diagonal = mass + np.bincount(owner, weights=weight, minlength=size)
        rhs = mass[:, None] * values
        solver = _direct_solver()
        if solver is not None:
            sparse, splu = solver
            diagonal_index = np.arange(size)
            matrix = sparse.csc_matrix(
                (np.concatenate((diagonal, -weight)),
                 (np.concatenate((diagonal_index, owner)), np.concatenate((diagonal_index, columns)))),
                shape=(size, size),
            )
            try:
                return splu(matrix, permc_spec='COLAMD').solve(rhs)
            except RuntimeError:
                pass  # (not singular unless something is very wrong) conjugate gradients then

        starts = np.flatnonzero(np.diff(owner, prepend=-1))
        linked = owner[starts]

        def multiply(x):
            result = diagonal[:, None] * x
            if len(columns):
                result[linked] -= np.add.reduceat(x[columns] * weight[:, None], starts, axis=0)
            return result

        return straighten._conjugate_gradient(
            multiply, rhs, values.copy(), 1.0 / diagonal, SOLVE_ITERATIONS, SOLVE_TOLERANCE,
        )


# The graph of the last mesh smoothed is kept for the next stroke, until the mesh changes.
_graph_cache = None


def _cached_graph(key, rest, edges, tris, merge_range, join_folds):
    global _graph_cache
    entry = _graph_cache
    if (entry is not None and entry[0] == (key, merge_range, join_folds) and np.array_equal(entry[1], rest) and
            np.array_equal(entry[2], edges) and np.array_equal(entry[3], tris)):
        return entry[4]
    graph = SmoothGraph(rest, edges, tris, merge_range, join_folds)
    _graph_cache = ((key, merge_range, join_folds), rest, edges, tris, graph)
    return graph


def clear_cache():
    global _graph_cache
    _graph_cache = None


# -----------------------------------------------------------------------------
# The brush

class SmoothStroke(goals.GoalStroke):
    """Blends a mesh's weights toward the same weights smoothed over the mesh, dab by dab.

    ``rest_co`` are the world space vertex positions in the rest pose; ``reach`` and ``merge_range``
    are distances, and ``join_folds`` whether folds of a part hold together like separate parts, see the
    module docs. See `goals.GoalStroke` for how weights change toward the goals.
    """

    def __init__(
            self, obj, rest_co, *,
            reach=0.02,
            merge_range=0.002,
            join_folds=True,
            accumulate=False,
            max_groups=DEFAULT_MAX_GROUPS,
            group_mode='ALL',
    ):
        super().__init__(obj, rest_co, max_groups=max_groups, group_mode=group_mode)
        self.reach = max(float(reach), 1e-6)
        self.accumulate = accumulate
        self.margin = (MARGIN_ACCUMULATING if accumulate else MARGIN) * self.reach
        # Groups that count: `GoalStroke` takes the bones of an armature; without one, all but the masks.
        self.bone_groups = smoothed_groups(obj)
        self.nothing_to_do = self.active is not None and not (
            0 <= self.active < len(self.groups) and
            self.active in self.bone_groups and self.active not in self.locked
        )
        self.columns = sorted(self.bone_groups - self.locked)
        self.column_of = {group: column for column, group in enumerate(self.columns)}

        mesh = self.mesh
        edges, tris = _read_edges(mesh), _read_triangles(mesh)
        self.graph = _cached_graph(
            obj.as_pointer(), self.rest, self.weld[edges], self.weld[tris], max(float(merge_range), 0.0),
            bool(join_folds),
        )
        self.member_pointers, self.members = _rows(self.weld, self.weld_count)
        # What is smoothed, per node, filled in as the stroke reaches it: its bone weights as shares of
        # their sum times how much of them it has (up to 1).
        self.table = np.zeros((self.weld_count, len(self.columns)), dtype=np.float32)
        self.confidence = np.zeros(self.weld_count, dtype=np.float32)
        self.known = np.zeros(self.weld_count, dtype=bool)
        self.smoothed = np.zeros(self.weld_count, dtype=bool)
        self.smoothed_weights = {}  # node -> {group: share} or None, see `_smooth`
        self.coverage = {}          # strongest blend factor applied per vertex (without Accumulate)

    # -------------------------------------------------------------------------
    # Public API

    def apply(self, verts, factors):
        """Move ``verts`` toward their smoothed weights by ``factors`` (strength times falloff, 0..1)."""
        verts = np.asarray(verts, dtype=np.int64).reshape(-1)
        factors = np.asarray(factors, dtype=np.float64).reshape(-1)
        keep = factors > 0.0
        verts, factors = verts[keep], factors[keep]
        if self.nothing_to_do or not len(verts):
            return
        nodes = self.weld[verts]
        fresh = np.unique(nodes if self.accumulate else nodes[~self.smoothed[nodes]])
        if len(fresh):
            self._smooth(fresh)

        removals = {}
        coverage = self.coverage
        for vert, node, factor in zip(verts.tolist(), nodes.tolist(), factors.tolist()):
            goal = self.smoothed_weights[node]
            if goal is None:
                continue
            if not self.accumulate:
                # Within a stroke a vertex moves at most as far as the strongest dab that touched it.
                if factor <= coverage.get(vert, 0.0) + MIN_CHANGE:
                    continue
                coverage[vert] = factor
            self._read(vert)
            weights = self._blend(vert, node, self.current[vert] if self.accumulate else self.start[vert],
                                  goal, factor)
            if weights is not None:
                self._write(vert, weights, removals)
        self._remove(removals)
        if self.accumulate:
            self._fill(np.unique(nodes))  # the next dab smooths what this one left

    def restore(self):
        super().restore()
        self.coverage.clear()
        self.known[:] = False
        self.smoothed[:] = False

    # -------------------------------------------------------------------------
    # Internals

    def _goal(self, vert, group, param, shares):
        return [(1.0, param)]

    def _compose(self, parts, budget, limit):
        if self.active is not None:
            return parts[0][1]  # only the active group's share counts, and the others' proportions
        return super()._compose(parts, budget, limit)

    def _fill(self, nodes):
        """Work out the table rows of ``nodes`` from the weights of their vertices."""
        positions, owner = _span(self.member_pointers, nodes)
        size = np.bincount(owner, minlength=len(nodes))
        source = self.current if self.accumulate else self.start
        column_of = self.column_of
        rows, columns, values = [], [], []
        confidence = np.zeros(len(nodes))
        for vert, row in zip(self.members[positions].tolist(), owner.tolist()):
            self._read(vert)
            entries = [(column_of[group], weight) for group, weight in source[vert].items()
                       if weight > 0.0 and group in column_of]
            total = sum(weight for _column, weight in entries)
            if total <= 1e-9:
                continue
            have = min(total, 1.0) / size[row]
            confidence[row] += have
            for column, weight in entries:
                rows.append(row)
                columns.append(column)
                values.append(weight / total * have)
        block = np.zeros((len(nodes), len(self.columns)))
        np.add.at(block, (np.array(rows, dtype=np.int64), np.array(columns, dtype=np.int64)), values)
        self.table[nodes] = block
        self.confidence[nodes] = confidence
        self.known[nodes] = True

    def _smooth(self, nodes):
        """Work out the smoothed weights of ``nodes``: what the weights around them make of them."""
        graph = self.graph
        region = graph.region(nodes, self.margin)
        unknown = region[~self.known[region]]
        if len(unknown):
            self._fill(unknown)
        block = self.table[region]
        used = np.flatnonzero(block.any(axis=0))
        values = np.empty((len(region), len(used) + 1))
        values[:, :-1] = block[:, used]
        values[:, -1] = self.confidence[region]
        smoothed = graph.smooth(region, values, self.reach)[np.searchsorted(region, nodes)]
        groups = [self.columns[column] for column in used.tolist()]
        for node, row in zip(nodes.tolist(), smoothed.tolist()):
            share = np.maximum(row[:-1], 0.0)
            if row[-1] <= 1e-6 or not len(share):
                self.smoothed_weights[node] = None
                continue
            share = share / row[-1]
            big = share > CUTOFF
            if not big.any():
                big[int(np.argmax(share))] = True
            kept = share[big]
            self.smoothed_weights[node] = dict(zip(
                (groups[column] for column in np.flatnonzero(big).tolist()), (kept / kept.sum()).tolist(),
            ))
        self.smoothed[nodes] = True


def stroke_params(settings):
    """`SmoothStroke` keyword arguments from the brush settings (`Scene.magic_fit`)."""
    return dict(
        reach=settings.smooth_reach,
        merge_range=settings.smooth_merge_range,
        join_folds=settings.smooth_use_folds,
        accumulate=settings.use_accumulate,
        max_groups=settings.smooth_max_groups,
        group_mode=settings.group_mode,
    )


def build_stroke(context, obj, settings):
    """A `SmoothStroke` for ``obj`` with the brush settings, measured in the rest pose.

    Raises `SmoothError` when it can't be set up.
    """
    if not smoothed_groups(obj):
        raise SmoothError("The mesh has no vertex groups to smooth")
    rest, = straighten.rest_geometry(context, (obj,), ('INDEXED',))
    return SmoothStroke(obj, rest.co, **stroke_params(settings))
