# SPDX-License-Identifier: GPL-3.0-or-later

"""Connect Parts: making the loose parts of a garment that lie on each other move together.

A garment is full of loose parts lying on others: a belt and its loops on a skirt, a flap over a panel, a
buckle on the belt, pockets, trims, rivets. Each got its weights on its own (from the body, by a transfer,
by hand), so where one lies on another their weights differ a little, and so do their moves. A belt
weighted to the hips alone, on a skirt that also follows the thighs, sinks into the skirt or lifts off it
as the legs bend, and more so when Customize+ scales the thighs and the butt.

Connect Parts gives each part the weights of the bigger parts it lies on. Its vertices within Distance of
them take the weights at their closest point there (barycentric, like the Copy brush); those farther than
Distance plus Blend keep their own; in between, over Blend, the weights are filled in smoothly along the
part from both sides (harmonic interpolation, cotangent weights). A part whose vertices all lie within
reach follows what it lies on entirely, however far it stands off in places (a buckle, a flap's free
edge); one that only touches it somewhere (a strap sewn to a dress) follows it there and keeps its own
weights farther along. Vertices without bone weights are filled in from the rest of their part, wherever
they are. As nothing depends on the weights that get replaced, connecting again changes nothing.

Parts go from the biggest down, by area, so a loop on a belt on a skirt takes the belt's new weights,
which are the skirt's. The biggest part keeps its weights, and a part that comes no closer to a bigger one
than Distance isn't connected at all. Parts are what a mesh's edges join, with vertices at the same spot
welded (split seams). The parts of every mesh given connect to each other, matching their vertex groups by
name: a belt can be a mesh of its own. Everything is measured in the rest pose, where weights apply. Only
bone weights change (see `goals.GoalStroke`): each vertex keeps their total, locked groups keep theirs, and
the group limit holds.

Like `painting`, nothing here depends on the 3D view.
"""

import numpy as np
from mathutils.bvhtree import BVHTree

from . import fitting, goals, smoothing, straighten
from .goals import DEFAULT_MAX_GROUPS

# Shares of a filled-in vertex below this are dropped, as Smooth does (FFXIV stores weights in bytes).
CUTOFF = smoothing.CUTOFF
# A vertex whose weights would all change by less than this is left as it is. Connecting again works the
# same weights out, but for rounding: the rest pose read through the Armature modifier moves by about
# 1e-8 once the weights change, and the search trees work in single precision.
MIN_CHANGE = 1e-5
# Conjugate gradients for the fill, when SciPy isn't there.
SOLVE_ITERATIONS = 2000
SOLVE_TOLERANCE = 1e-16


class ConnectError(Exception):
    """Why parts can't be connected; the message is meant for the user."""


class ConnectStroke(goals.GoalStroke):
    """Writes one mesh's new weights: the goal of each welded vertex, worked out by `connect`.

    ``rest_co`` are the world space vertex positions in the rest pose. Groups count as bone weights as
    for Smooth (`smoothing.smoothed_groups`): the deforming bones' when an armature deforms the mesh, else
    every group but the masks. ``bone_names`` are the groups it may add (the bones the parts it lies on
    are weighted to), when no armature deforms it.
    """

    def __init__(self, obj, rest_co, *, bone_names=(), max_groups=DEFAULT_MAX_GROUPS):
        super().__init__(obj, rest_co, max_groups=max_groups)
        self.bone_groups = set(smoothing.smoothed_groups(obj))
        if not self.bone_names:
            self.bone_names = set(bone_names)
        self.goals = {}  # welded vertex -> {group name: share}
        self.member_order, self.member_start = fitting._rows(self.weld, self.weld_count)

    def members(self, node):
        """The vertices welded into ``node``."""
        return self.member_order[self.member_start[node]:self.member_start[node + 1]]

    def node_shares(self, node):
        """The bone weights of welded vertex ``node`` as {group name: fraction of their sum}, from the
        first of its vertices that has any, or None."""
        for vert in self.members(node).tolist():
            self._read(vert)
            weights = {group: weight for group, weight in self.current[vert].items()
                       if weight > 0.0 and group in self.bone_groups}
            total = sum(weights.values())
            if total > 1e-9:
                return {self.group_names[group]: weight / total for group, weight in weights.items()}
        return None

    def keyed(self, shares):
        """``shares`` by group name as a goal of this mesh (None if it can't take any): its own groups, or
        names of bones it may add; groups it can't take (locked, not bones) are left out and the others
        make up for them. Called as each vertex is written, so a group added for one is found by the next."""
        goal = {}
        for name, share in shares.items():
            key = self._key(name)
            if key is not None and share > 0.0:
                goal[key] = goal.get(key, 0.0) + share
        total = sum(goal.values())
        return {key: share / total for key, share in goal.items()} if total > 1e-9 else None

    def _prepare(self, groups):
        return [self.goals.get(group) for group in groups.tolist()]

    def _leave(self, current, param):
        return param is None

    def _goal(self, vert, group, param, shares):
        goal = self.keyed(param)
        return [(1.0, goal)] if goal is not None else None

    def _write(self, vert, weights, removals):
        current = self.current[vert]
        if all(abs(weight - current.get(group, 0.0)) <= MIN_CHANGE for group, weight in weights.items()):
            return
        super()._write(vert, weights, removals)


class _Mesh:
    """A mesh's loose parts (welded vertices of each, and their area), and the links of its welded vertex
    graph with their cotangent weights."""

    def __init__(self, stroke, tris, edges):
        weld, count = stroke.weld, stroke.weld_count
        tris = weld[tris]
        tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])]
        self.tris = tris
        links, weights = straighten.cotangent_weights(stroke.rest, tris) if len(tris) else (
            np.empty((0, 2), dtype=np.int64), np.zeros(0))
        # Edges without triangles (wires) link as strongly as the triangles' edges do on average.
        wires = fitting._welded_edges(weld, count, edges)
        if len(wires):
            keys = links[:, 0] * count + links[:, 1]
            wires = wires[~np.isin(wires[:, 0] * count + wires[:, 1], keys)]
            links = np.concatenate((links, wires))
            weights = np.concatenate((weights, np.full(len(wires), weights.mean() if len(weights) else 1.0)))
        self.links, self.weights = links, weights
        labels, parts = fitting._connected(count, links)
        self.labels = labels
        self.area = np.zeros(parts)
        if len(tris):
            co = stroke.rest[tris]
            np.add.at(self.area, labels[tris[:, 0]],
                      0.5 * np.linalg.norm(np.cross(co[:, 1] - co[:, 0], co[:, 2] - co[:, 0]), axis=1))
        order, start = fitting._rows(labels, parts)
        self.nodes = [order[start[k]:start[k + 1]] for k in range(parts)]
        order, start = fitting._rows(labels[links[:, 0]], parts)
        self.part_links = [order[start[k]:start[k + 1]] for k in range(parts)]


def connect(strokes, tris, edges, *, distance=0.005, blend=0.02, allowed=None):
    """Give the loose parts of the meshes of ``strokes`` (`ConnectStroke`s) the weights of the bigger parts
    they lie on, see the module docs.

    ``tris`` and ``edges`` are each mesh's triangles and edges (vertex indices, (n, 3) and (m, 2));
    ``allowed`` the vertices of each mesh that may change (a bool mask, or None for all). Returns how many
    parts of each mesh were connected; the strokes hold what changed.
    """
    distance = max(float(distance), 0.0)
    reach = distance + max(float(blend), 0.0)
    if allowed is None:
        allowed = [None] * len(strokes)
    meshes = [_Mesh(stroke, np.asarray(t, dtype=np.int64).reshape(-1, 3), np.asarray(e, dtype=np.int64).reshape(-1, 2))
              for stroke, t, e in zip(strokes, tris, edges)]
    allowed_nodes = []
    for stroke, mask in zip(strokes, allowed):
        nodes = np.ones(stroke.weld_count, dtype=bool)
        if mask is not None:
            nodes[:] = False
            nodes[stroke.weld[np.asarray(mask, dtype=bool)]] = True
        allowed_nodes.append(nodes)

    # Every part of every mesh, the biggest first (equal ones in a fixed order).
    parts = [(index, part) for index, mesh in enumerate(meshes) for part in range(len(mesh.nodes))]
    order = sorted(range(len(parts)), key=lambda k: -meshes[parts[k][0]].area[parts[k][1]])

    # Every triangle, in the welded vertices of all the meshes, with the rank of its part.
    offsets = np.cumsum([0] + [stroke.weld_count for stroke in strokes])
    points = np.concatenate([stroke.rest for stroke in strokes]) if strokes else np.empty((0, 3))
    node_mesh = np.repeat(np.arange(len(strokes)), np.diff(offsets))
    node_rank = np.empty(len(points), dtype=np.int64)
    for rank, k in enumerate(order):
        index, part = parts[k]
        node_rank[offsets[index] + meshes[index].nodes[part]] = rank
    all_tris = np.concatenate([offsets[index] + mesh.tris for index, mesh in enumerate(meshes)] or
                              [np.empty((0, 3), dtype=np.int64)])
    tri_rank = node_rank[all_tris[:, 0]]
    corners = points[all_tris]
    tri_low, tri_high = corners.min(axis=1), corners.max(axis=1)

    # Each welded vertex's bone weights as shares by group name, read when first needed (and again
    # once its part has changed).
    shares = {}

    def shares_of(node):
        if node not in shares:
            index = int(node_mesh[node])
            shares[node] = strokes[index].node_shares(node - int(offsets[index]))
        return shares[node]

    connected = [0] * len(strokes)
    for rank, k in enumerate(order):
        index, part = parts[k]
        mesh, stroke = meshes[index], strokes[index]
        nodes = mesh.nodes[part]
        rest = stroke.rest[nodes]
        near = ((tri_rank < rank) & (tri_high >= rest.min(axis=0) - reach).all(axis=1) &
                (tri_low <= rest.max(axis=0) + reach).all(axis=1))
        if not near.any():
            continue
        found = _closest(points, all_tris[near], rest, reach)
        if found is None:
            continue
        hit, location, tri, gap = found
        if not (gap <= distance).any():
            continue  # it doesn't lie on a bigger part
        tri = np.flatnonzero(near)[tri]
        bary = fitting._barycentric(location, *(points[all_tris[tri, c]] for c in range(3)))

        # What each vertex of the part takes: the weights where it lies on a bigger part, its own (kept
        # as they are), or filled in between them (unknown).
        count = len(nodes)
        may_change = allowed_nodes[index][nodes]
        values = [None] * count
        base = int(offsets[index])
        own = [shares_of(base + node) for node in nodes.tolist()]
        lies_on = np.zeros(count, dtype=bool)
        within = np.zeros(count, dtype=bool)
        within[hit] = True
        for row, corner_row, weight_row, apart in zip(hit.tolist(), all_tris[tri].tolist(), bary.tolist(),
                                                     gap.tolist()):
            if apart > distance or not may_change[row]:
                continue
            values[row] = _mix([(weight, shares_of(corner)) for corner, weight in zip(corner_row, weight_row)])
            lies_on[row] = values[row] is not None
        if not lies_on.any():
            continue
        has_own = np.array([value is not None for value in own])
        unknown = may_change & ~lies_on & (within | ~has_own)
        keep = ~lies_on & ~unknown & has_own
        for row in np.flatnonzero(keep).tolist():
            values[row] = own[row]
        filled = _fill(count, *_local_links(mesh, part, nodes), values, unknown)

        for row in np.flatnonzero(lies_on).tolist():
            stroke.goals[int(nodes[row])] = values[row]
        for row, value in filled.items():
            stroke.goals[int(nodes[row])] = value
        changing = np.zeros(stroke.weld_count, dtype=bool)
        changing[nodes[lies_on]] = True
        changing[nodes[list(filled)]] = True
        verts = np.flatnonzero(changing[stroke.weld])
        if allowed[index] is not None:
            verts = verts[np.asarray(allowed[index], dtype=bool)[verts]]
        modified = stroke.modified
        stroke.modified = False
        stroke.apply(verts, np.ones(len(verts)))
        if stroke.modified:
            connected[index] += 1
            for node in (base + nodes).tolist():
                shares.pop(node, None)
        stroke.modified = stroke.modified or modified
    return connected


def _mix(weighted):
    """The sum of shares ({name: share}) times their weights, over those that aren't None, divided by the
    weights that count; None when none does."""
    mixed = {}
    total = 0.0
    for weight, value in weighted:
        if value is None or weight <= 0.0:
            continue
        total += weight
        for name, share in value.items():
            mixed[name] = mixed.get(name, 0.0) + weight * share
    if total <= 1e-12:
        return None
    return {name: share / total for name, share in mixed.items()}


def _local_links(mesh, part, nodes):
    """The links of ``part`` of ``mesh`` in the part's own numbering (the order of ``nodes``), and their
    weights."""
    rows = mesh.part_links[part]
    local = np.full(len(mesh.labels), -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    return local[mesh.links[rows]], mesh.weights[rows]


def _fill(count, links, weights, values, unknown):
    """Harmonic interpolation over a graph of ``count`` nodes, ``links`` (index pairs) with ``weights``: the
    ``unknown`` ones (a mask) take a weighted average of their neighbours, the others have ``values`` (shares
    by name; None: not part of it). Returns {node: shares} for the unknown nodes linked to a known one."""
    known = np.array([value is not None for value in values]) & ~unknown
    a, b = links[:, 0], links[:, 1]
    use = (unknown[a] | unknown[b]) & (unknown[a] | known[a]) & (unknown[b] | known[b])
    a, b, weights = a[use], b[use], weights[use]
    # Unknown nodes cut off from every known one stay as they are.
    inner = unknown[a] & unknown[b]
    labels, groups = fitting._connected(count, np.stack((a[inner], b[inner]), axis=1))
    reached = np.zeros(groups, dtype=bool)
    reached[labels[np.where(unknown[a], a, b)[~inner]]] = True
    solved = np.flatnonzero(unknown & reached[labels])
    if not len(solved):
        return {}
    local = np.full(count, -1, dtype=np.int64)
    local[solved] = np.arange(len(solved))
    names = sorted({name for value in values if value is not None for name in value})
    column = {name: k for k, name in enumerate(names)}
    table = np.zeros((count, len(names)))
    for row, value in enumerate(values):
        if value is not None and known[row]:
            for name, share in value.items():
                table[row, column[name]] = share

    size = len(solved)
    diagonal = np.zeros(size)
    rhs = np.zeros((size, len(names)))
    first, second, off = [], [], []
    for tail, head in ((a, b), (b, a)):
        rows = local[tail]
        mine = rows >= 0
        np.add.at(diagonal, rows[mine], weights[mine])
        other = local[head]
        inside = mine & (other >= 0)
        first.append(rows[inside])
        second.append(other[inside])
        off.append(weights[inside])
        border = mine & (other < 0)
        np.add.at(rhs, rows[border], weights[border, None] * table[head[border]])
    first, second, off = np.concatenate(first), np.concatenate(second), np.concatenate(off)

    solver = smoothing._direct_solver()
    result = None
    if solver is not None:
        sparse, splu = solver
        matrix = sparse.csc_matrix(
            (np.concatenate((diagonal, -off)), (np.concatenate((np.arange(size), first)),
                                               np.concatenate((np.arange(size), second)))),
            shape=(size, size))
        try:
            result = splu(matrix, permc_spec='COLAMD').solve(rhs)
        except RuntimeError:
            result = None
    if result is None:
        order = np.argsort(first, kind='stable')
        first, second, off = first[order], second[order], off[order]
        starts = np.flatnonzero(np.diff(first, prepend=-1))

        def multiply(x):
            product = diagonal[:, None] * x
            if len(first):
                product[first[starts]] -= np.add.reduceat(x[second] * off[:, None], starts, axis=0)
            return product

        result = straighten._conjugate_gradient(multiply, rhs, np.zeros_like(rhs), 1.0 / diagonal,
                                                SOLVE_ITERATIONS, SOLVE_TOLERANCE)

    filled = {}
    for row, shares_row in zip(solved.tolist(), np.maximum(result, 0.0).tolist()):
        kept = {names[k]: share for k, share in enumerate(shares_row) if share > CUTOFF}
        if not kept:
            best = int(np.argmax(shares_row))
            kept = {names[best]: 1.0}
        total = sum(kept.values())
        filled[row] = {name: share / total for name, share in kept.items()}
    return filled


def _closest(points, tris, queries, reach):
    """The closest points of the triangles ``tris`` (indices into ``points``) to ``queries``, up to
    ``reach`` away: (which queries found one, the points, which triangle, how far), or None."""
    used, local = np.unique(tris, return_inverse=True)
    tree = BVHTree.FromPolygons(points[used].tolist(), local.reshape(-1, 3).tolist(), all_triangles=True)
    hit, location, tri, gap = [], [], [], []
    for k, query in enumerate(queries.tolist()):
        found = tree.find_nearest(query, reach)
        if found[0] is None:
            continue
        hit.append(k)
        location.append(found[0])
        tri.append(found[2])
        gap.append(found[3])
    if not hit:
        return None
    return np.array(hit), np.array(location), np.array(tri, dtype=np.int64), np.array(gap)


def stroke_params(settings):
    """`connect` keyword arguments from the settings (`Scene.magic_fit`)."""
    return dict(distance=settings.connect_distance, blend=settings.connect_blend)


def build(context, objects, settings):
    """A `ConnectStroke` for each of ``objects`` (meshes) with the settings, and their triangles and edges
    (for `connect`), measured in the rest pose.

    Raises `ConnectError` when they can't be connected.
    """
    counted = [smoothing.smoothed_groups(obj) for obj in objects]
    if not any(counted):
        raise ConnectError("No weights to connect")
    names = {obj.vertex_groups[index].name for obj, indices in zip(objects, counted) for index in indices}
    geometries = straighten.rest_geometry(context, objects, ('INDEXED',) * len(objects))
    strokes, tris, edges = [], [], []
    for obj, geometry in zip(objects, geometries):
        strokes.append(ConnectStroke(obj, geometry.co, bone_names=names, max_groups=settings.connect_max_groups))
        tris.append(geometry.tris)
        edge_array = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
        obj.data.edges.foreach_get("vertices", edge_array)
        edges.append(edge_array.reshape(-1, 2))
    return strokes, tris, edges
