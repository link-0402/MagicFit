# SPDX-License-Identifier: GPL-3.0-or-later

"""Close Rips: finding where a mesh has been pulled apart, and pulling it back together.

Blender's proportional editing with Connected Only measures its reach along the mesh. Where a mesh is in
pieces that only touch (split seams, the cracks of models whose parts were made with different edges),
that reach stops at the edge, so moving one side leaves the other where it was and the mesh tears open.

A rip shows as two open edges that face each other across a narrow gap. To be taken as one, two edges
have to:

- run opposite ways (`FACING`). The faces on the two sides of a seam go round it in opposite directions,
  whatever angle the surface turns there (a flat seam, a crease, a fold), while a layer lying on another
  runs the same way as what it lies on;
- be far apart along the mesh compared with the gap (`AROUND`): the two sides of a ribbon, a strap or a
  narrow notch face each other too, but are only their width apart over their faces;
- choose each other (each side's closest such edge leads back to the other side), over a stretch of
  several vertices longer than the gap (`MIN_RUN`, `RUN_LENGTH`), so that a part that merely passes by
  the rip isn't pulled in;
- not merely touch: edges that touch somewhere (the hems of two trouser legs) bend away from each other,
  their gap growing like a parabola from where they're closest, while a rip narrows toward its tips or
  keeps its width (`BENDING_APART`, `NEVER_MEETS`).

Vertices at the same spot (`SEAM_DISTANCE`) are taken as one throughout, so split seams that are still
closed count as joined and stay closed, and a seam torn open shows as a single gap however many split
seams meet along it. Where both sides of a stretch have as many vertices, they are paired one to one, in
order: they were the two copies of the same vertices (`Rips.pairs`). Elsewhere a vertex goes to its
closest point of the other side (`Rips.points`), unless it's much denser than that side (`DENSER`): the
denser side keeps its own curve, and the vertices of the coarser side are put on it, so straight coarse
edges don't flatten it.

`close` then moves both sides halfway, as smoothly as it can: the moves fade out over Spread along the
mesh around the rip (a harmonic displacement, the rest of the mesh held where it is), so nothing else
needs fixing after. Which side was pulled away can't be told from the mesh, so both move. `weld_plan`
lists what to merge to join the sides for good: paired vertices are merged, and the edges of the coarser
side are split where the vertices of the other side lie on them.

Nothing is known about the shape before it tore, so cracks the model came with close too, where they
qualify. Find Rips shows what would close.

Like `fitting`, nothing here depends on the 3D view: positions are world space.
"""

import heapq

import bmesh
import numpy as np

from . import fitting, smoothing, straighten

# Vertices closer than this are one (split seams, and the copies that join several parts).
SEAM_DISTANCE = 1e-4
# Two open edges only face each other when they run at least this much opposite ways (cosine).
FACING = -0.5
# The other side of a rip is at least this many times the gap away along the mesh: a ribbon's sides, a
# notch's or a strap's are about their width apart over their faces.
AROUND = 2.5
# A rip runs along at least this many vertices on each side, over a stretch at least this many times as
# long as its gap.
MIN_RUN = 3
RUN_LENGTH = 2.0
# Along a run, the points matched on the other side may step back or jump by this many times the gap
# (plus `RUN_SLACK`) before the run breaks.
RUN_STEP = 4.0
RUN_SLACK = 0.03
# Two open edges that merely touch (the hems of trouser legs) bend away from each other: their gap grows
# like a parabola away from where they're closest, and never gets small. A rip either narrows to its tips or
# keeps its width. A run isn't a rip when its gap, fitted with a parabola, rises by at least `BENDING_APART`
# of its narrowest gap over its length, and that narrowest gap is at least `NEVER_MEETS` of its widest.
BENDING_APART = 0.4
NEVER_MEETS = 0.3
# A vertex whose edges are this much shorter than the edge it faces isn't put on that edge: the denser
# side keeps its curve, and the coarser side's vertices go on it instead.
DENSER = 0.6
# Open edges are sampled this many times per gap to find the edges near each vertex.
SAMPLES_PER_GAP = 4
# Matched points are held together this many times harder than the mesh holds its shape.
HOLD = 1e4
# Conjugate gradients for the solve, when SciPy isn't there.
SOLVE_ITERATIONS = 3000
SOLVE_TOLERANCE = 1e-14
# Weld: a vertex within this share of an edge from one of its ends is merged with that end instead of
# splitting the edge there.
WELD_END = 0.2
# Weld: edges shorter than this between merged vertices are dissolved.
DEGENERATE = 1e-6


class Rips:
    """The rips `find` found in a mesh.

    Vertex ``k``, at ``co[k]``, belongs to node ``node[k]`` (vertices closer than `SEAM_DISTANCE` share
    one), at ``node_co``. ``tris`` are the mesh's triangles over the nodes, and ``tail``, ``head`` its open edges
    (in their face's winding), with ``tail_vertex`` and ``head_vertex`` the vertices they were found at.

    ``pairs`` (n, 2) are nodes that belong at the same spot, ``points`` nodes that belong on an open edge:
    node ``point_node[i]`` on edge ``point_edge[i]``, at ``point_t[i]`` of the way from its tail to its
    head. ``faced`` are the edge matches of every rip vertex, also the ones of denser sides, for welding.
    """

    __slots__ = ("co", "node", "node_co", "tris", "tail", "head", "tail_vertex", "head_vertex", "pairs",
                 "point_node", "point_edge", "point_t", "faced")

    @property
    def count(self):
        return len(self.pairs) + len(self.point_node)

    def nodes(self):
        """The nodes on the found rips."""
        parts = [self.pairs.reshape(-1), self.point_node, self.tail[self.point_edge], self.head[self.point_edge]]
        return np.unique(np.concatenate(parts)) if any(len(p) for p in parts) else np.zeros(0, dtype=np.int64)

    def vertices(self):
        """The vertices on the found rips (every copy of a node)."""
        found = np.zeros(len(self.node_co), dtype=bool)
        found[self.nodes()] = True
        return np.flatnonzero(found[self.node])


# -----------------------------------------------------------------------------
# The mesh as nodes

def _nodes(co):
    node, count = fitting.weld_groups(co, SEAM_DISTANCE)
    node_co = np.zeros((count, 3))
    np.add.at(node_co, node, co)
    node_co /= np.bincount(node, minlength=count)[:, None]
    return node, node_co


def _open_edges(tris, node):
    """The open edges of the triangles ``tris`` once their vertices are nodes: (tail, head) nodes in
    their face's winding, and the vertices they come from. Edges with three or more faces count as closed."""
    corners = node[tris]
    keep = (corners[:, 0] != corners[:, 1]) & (corners[:, 1] != corners[:, 2]) & (corners[:, 2] != corners[:, 0])
    tris, corners = tris[keep], corners[keep]
    tail = corners.reshape(-1)
    head = corners[:, [1, 2, 0]].reshape(-1)
    count = int(node.max()) + 1 if len(node) else 0
    key = np.minimum(tail, head).astype(np.int64) * count + np.maximum(tail, head)
    _, inverse, uses = np.unique(key, return_inverse=True, return_counts=True)
    single = uses[inverse] == 1
    return (tail[single], head[single], tris.reshape(-1)[single], tris[:, [1, 2, 0]].reshape(-1)[single], corners)


def _graph_edges(tris):
    edges = np.sort(np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])), axis=1)
    return np.unique(edges, axis=0)


class _Reach:
    """Shortest distances along the edges of a graph from one vertex, as far as a limit: the regions are
    small (a few times a gap), so a heap beats sweeping the whole mesh."""

    def __init__(self, co, edges):
        count = len(co)
        tail = np.concatenate((edges[:, 0], edges[:, 1]))
        head = np.concatenate((edges[:, 1], edges[:, 0]))
        order = np.argsort(tail, kind='stable')
        start = np.zeros(count + 1, dtype=np.int64)
        np.cumsum(np.bincount(tail, minlength=count), out=start[1:])
        self.start = start.tolist()
        self.head = head[order].tolist()
        self.length = np.linalg.norm(co[tail[order]] - co[head[order]], axis=1).tolist()

    def around(self, source, limit):
        distance = {source: 0.0}
        heap = [(0.0, source)]
        start, head, length = self.start, self.head, self.length
        while heap:
            reached, vertex = heapq.heappop(heap)
            if reached > distance[vertex]:
                continue
            for k in range(start[vertex], start[vertex + 1]):
                other = head[k]
                further = reached + length[k]
                if further <= limit and further < distance.get(other, np.inf):
                    distance[other] = further
                    heapq.heappush(heap, (further, other))
        return distance


def _loops(tail, head, co):
    """Each open edge's loop and where along it the edge starts (arc length from the loop's first edge),
    and each loop's length. A chain that doesn't close (it meets other open edges at a vertex) is a loop of
    its own."""
    count = len(tail)
    leaving = {}
    for index, vertex in enumerate(tail.tolist()):
        leaving.setdefault(vertex, []).append(index)
    length = np.linalg.norm(co[head] - co[tail], axis=1)
    loop = np.full(count, -1, dtype=np.int64)
    arc = np.zeros(count)
    totals = []
    heads = head.tolist()
    # Chains that don't close are walked from where they start, so that each is one loop.
    starts = np.bincount(head, minlength=len(co))[tail] == 0
    for first in np.concatenate((np.flatnonzero(starts), np.flatnonzero(~starts))).tolist():
        if loop[first] >= 0:
            continue
        edge, along, number = first, 0.0, len(totals)
        while edge is not None and loop[edge] < 0:
            loop[edge] = number
            arc[edge] = along
            along += length[edge]
            nxt = [k for k in leaving.get(heads[edge], ()) if loop[k] < 0]
            edge = nxt[0] if len(nxt) == 1 else None
        totals.append(along)
    return loop, arc, np.array(totals), length


# -----------------------------------------------------------------------------
# Finding rips

def _closest_on_edges(points, a, b):
    ab = b - a
    t = np.einsum("ij,ij->i", points - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30)
    t = np.clip(t, 0.0, 1.0)
    return t, np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _facing_edges(co, tail, head, nodes, max_gap, reach):
    """For each of ``nodes`` (on open edges), the closest open edge that faces it within ``max_gap``:
    (node, edge, t, gap) arrays, sorted by node."""
    length = np.linalg.norm(co[head] - co[tail], axis=1)
    direction = (co[head] - co[tail]) / np.maximum(length, 1e-30)[:, None]
    along = np.zeros_like(co)
    np.add.at(along, tail, direction)
    np.add.at(along, head, direction)
    along /= np.maximum(np.linalg.norm(along, axis=1), 1e-30)[:, None]
    # Points along every open edge, at most a quarter of the gap apart: an edge within the gap of a vertex
    # has one within 1.25 times the gap.
    steps = np.maximum(np.ceil(length * SAMPLES_PER_GAP / max_gap).astype(np.int64), 1)
    owner = np.repeat(np.arange(len(tail)), steps)
    share = (np.arange(len(owner)) - np.repeat(np.cumsum(steps) - steps, steps) + 0.5) / steps[owner]
    samples = co[tail[owner]] + share[:, None] * (co[head[owner]] - co[tail[owner]])
    first, second = smoothing.close_pairs(np.concatenate((co[nodes], samples)),
                                          (1.0 + 1.0 / SAMPLES_PER_GAP) * max_gap)
    mixed = (first < len(nodes)) != (second < len(nodes))
    first, second = first[mixed], second[mixed]
    which = np.where(first < len(nodes), first, second)
    edge = owner[np.where(first < len(nodes), second, first) - len(nodes)]
    key = np.unique(which.astype(np.int64) * len(tail) + edge)
    vertex, edge = nodes[key // len(tail)], key % len(tail)
    keep = (tail[edge] != vertex) & (head[edge] != vertex)
    vertex, edge = vertex[keep], edge[keep]
    t, gap = _closest_on_edges(co[vertex], co[tail[edge]], co[head[edge]])
    keep = (gap <= max_gap) & (np.einsum("ij,ij->i", along[vertex], direction[edge]) <= FACING)
    vertex, edge, t, gap = vertex[keep], edge[keep], t[keep], gap[keep]
    order = np.lexsort((gap, vertex))
    vertex, edge, t, gap = vertex[order], edge[order], t[order], gap[order]
    # The closest edge that's far enough along the mesh.
    found = []
    starts = np.flatnonzero(np.diff(vertex, prepend=-1)).tolist() + [len(vertex)]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        source = int(vertex[s0])
        near = reach.around(source, AROUND * float(gap[s1 - 1]))
        for k in range(s0, s1):
            da = near.get(int(tail[edge[k]]), np.inf)
            db = near.get(int(head[edge[k]]), np.inf)
            closest = (1.0 - t[k]) * da + t[k] * db if np.isfinite(da) and np.isfinite(db) else min(da, db)
            if closest >= AROUND * gap[k]:
                found.append(k)
                break
    found = np.array(found, dtype=np.int64)
    return vertex[found], edge[found], t[found], gap[found]


def _mutual(co, tail, head, loop, length, vertex, edge, t):
    """Which matches are chosen back: an end of the edge a vertex faces is matched to the vertex's own
    loop, near it."""
    count = len(co)
    vertex_loop = np.full(count, -1, dtype=np.int64)
    vertex_loop[tail] = loop
    match_edge = np.full(count, -1, dtype=np.int64)
    match_edge[vertex] = edge
    match_point = np.zeros((count, 3))
    match_point[vertex] = co[tail[edge]] + t[:, None] * (co[head[edge]] - co[tail[edge]])
    keep = np.zeros(len(vertex), dtype=bool)
    for k in range(len(vertex)):
        for end in (tail[edge[k]], head[edge[k]]):
            back = match_edge[end]
            if (back >= 0 and loop[back] == vertex_loop[vertex[k]] and
                    np.linalg.norm(match_point[end] - co[vertex[k]]) <= length[edge[k]] + length[back]):
                keep[k] = True
                break
    return keep


def _touching(points, gaps):
    """Whether a run along ``points`` with these gaps is two edges that bend away from each other."""
    if gaps.min() < NEVER_MEETS * gaps.max():
        return False
    along = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if along[-1] <= 0.0:
        return False
    bend = np.polyfit(along / along[-1], gaps, 2)[0]
    return bend >= BENDING_APART * gaps.min()


def _runs(co, tail, head, loop, arc, totals, length, vertex, edge, t, gap):
    """Keep the matches that make runs: at least `MIN_RUN` vertices in a row along an open loop, each matched
    a little further back along the same loop, over a stretch at least `RUN_LENGTH` times their gap long.
    Also returns the vertex pairs of runs whose other side has as many vertices (one to one)."""
    count = len(co)
    match_edge = np.full(count, -1, dtype=np.int64)
    match_edge[vertex] = edge
    match_pos = np.zeros(count)
    match_pos[vertex] = arc[edge] + t * length[edge]
    match_gap = np.zeros(count)
    match_gap[vertex] = gap
    keep = np.zeros(count, dtype=bool)
    pairs = []
    order = np.lexsort((arc, loop))
    bounds = np.flatnonzero(np.diff(loop[order], prepend=-1)).tolist() + [len(order)]
    loops = {}
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        edges = order[s0:s1]
        number = int(loop[edges[0]])
        loops[number] = (tail[edges], arc[edges], totals[number], head[edges[-1]] == tail[edges[0]])

    def label(node):
        e = match_edge[node]
        return None if e < 0 else (int(loop[e]), float(match_pos[node]))

    def same(a, b, step_gap):
        return (a is not None and b is not None and a[0] == b[0] and
                abs(a[1] - b[1]) <= RUN_STEP * step_gap + RUN_SLACK)

    for number, (nodes, positions, total, closed) in loops.items():
        if not (match_edge[nodes] >= 0).any():
            continue
        labels = [label(node) for node in nodes]
        size = len(nodes)
        start = 0
        if closed:
            for k in range(size):
                if not same(labels[k - 1], labels[k], match_gap[nodes[k]]):
                    start = k
                    break
        sequence = [(start + k) % size for k in range(size)] if closed else list(range(size))
        run = []

        def finish(run):
            if len(run) < MIN_RUN:
                return
            members = nodes[run]
            stretch = np.linalg.norm(np.diff(co[members], axis=0), axis=1).sum()
            gaps = match_gap[members]
            if stretch < RUN_LENGTH * np.median(gaps):
                return
            if _touching(co[members], gaps):
                return
            keep[members] = True
            other, other_pos, other_total, other_closed = loops[labels[run[0]][0]]
            matched = np.array([labels[k][1] for k in run])
            if other_closed and other_total > 0.0:
                matched = np.unwrap(matched * (2.0 * np.pi / other_total)) * (other_total / (2.0 * np.pi))
            half = 0.5 * float(np.median(length[match_edge[members]]))
            low, high = matched.min() - half, matched.max() + half
            shifts = (-other_total, 0.0, other_total) if other_closed else (0.0,)
            inside = sorted((position + shift, int(node)) for shift in shifts
                            for node, position in zip(other.tolist(), other_pos.tolist())
                            if low <= position + shift <= high)
            # Both sides can be on the same loop (a crack within one part): the run's own nodes aren't partners.
            if len(inside) == len(run) and not set(node for _, node in inside) & set(members.tolist()):
                partners = [node for _, node in inside]
                if matched[-1] < matched[0]:
                    partners.reverse()
                pairs.extend(zip(members.tolist(), partners))

        for k in sequence:
            if run and same(labels[run[-1]], labels[k], match_gap[nodes[k]]):
                run.append(k)
            else:
                finish(run)
                run = [k] if labels[k] is not None else []
        finish(run)
    # Both sides' runs find the same pairs.
    pairs = np.unique(np.sort(np.array(pairs, dtype=np.int64).reshape(-1, 2), axis=1), axis=0)
    return keep[vertex], pairs


def find(co, tris, allowed, max_gap):
    """The rips of a mesh (``co`` world space positions, ``tris`` its triangles) whose two sides both have
    vertices in ``allowed``, with gaps up to ``max_gap``, as `Rips`."""
    rips = Rips()
    rips.co = co
    rips.node, rips.node_co = _nodes(co)
    node, node_co = rips.node, rips.node_co
    allowed_node = np.zeros(len(node_co), dtype=bool)
    allowed_node[node[allowed]] = True
    tail, head, tail_vertex, head_vertex, rips.tris = _open_edges(tris, node)
    rips.tail, rips.head, rips.tail_vertex, rips.head_vertex = tail, head, tail_vertex, head_vertex
    empty = np.zeros(0, dtype=np.int64)
    rips.pairs = np.zeros((0, 2), dtype=np.int64)
    rips.point_node, rips.point_edge, rips.point_t = empty, empty, np.zeros(0)
    rips.faced = (empty, empty, np.zeros(0))
    nodes = np.unique(tail)
    nodes = nodes[allowed_node[nodes]]
    if not len(nodes):
        return rips
    loop, arc, totals, length = _loops(tail, head, node_co)
    reach = _Reach(node_co, _graph_edges(rips.tris))
    vertex, edge, t, gap = _facing_edges(node_co, tail, head, nodes, max_gap, reach)
    # The edge faced is on the other side when one of its ends is (the other can be where the rip ends).
    other = allowed_node[tail[edge]] | allowed_node[head[edge]]
    vertex, edge, t, gap = vertex[other], edge[other], t[other], gap[other]
    keep = _mutual(node_co, tail, head, loop, length, vertex, edge, t)
    vertex, edge, t, gap = vertex[keep], edge[keep], t[keep], gap[keep]
    keep, pairs = _runs(node_co, tail, head, loop, arc, totals, length, vertex, edge, t, gap)
    vertex, edge, t = vertex[keep], edge[keep], t[keep]
    rips.faced = (vertex, edge, t)
    rips.pairs = pairs
    if len(pairs):
        paired = np.zeros(len(node_co), dtype=bool)
        paired[pairs.reshape(-1)] = True
        keep = ~paired[vertex]
        vertex, edge, t = vertex[keep], edge[keep], t[keep]
    # The denser side keeps its curve.
    spacing = np.zeros(len(node_co))
    edges_at = np.zeros(len(node_co))
    for ends in (tail, head):
        np.add.at(spacing, ends, length)
        np.add.at(edges_at, ends, 1.0)
    spacing /= np.maximum(edges_at, 1.0)
    keep = spacing[vertex] >= DENSER * length[edge]
    rips.point_node, rips.point_edge, rips.point_t = vertex[keep], edge[keep], t[keep]
    return rips


# -----------------------------------------------------------------------------
# Closing them

def _solve(count, rows, columns, entries, rhs):
    """Solve A x = ``rhs`` (each column), A the symmetric positive definite ``count`` square matrix with
    ``entries`` at ``rows``, ``columns`` (repeats adding up): SciPy's sparse LU, else conjugate gradients."""
    solver = smoothing._direct_solver()
    if solver is not None:
        sparse, splu = solver
        matrix = sparse.csc_matrix((entries, (rows, columns)), shape=(count, count))
        return splu(matrix, permc_spec='COLAMD').solve(rhs)
    diagonal = np.bincount(rows[rows == columns], entries[rows == columns], minlength=count)
    order = np.argsort(rows, kind='stable')
    rows, columns, entries = rows[order], columns[order], entries[order]
    starts = np.flatnonzero(np.diff(rows, prepend=-1))

    def multiply(x):
        product = np.zeros_like(x)
        product[rows[starts]] = np.add.reduceat(x[columns] * entries[:, None], starts, axis=0)
        return product

    return straighten._conjugate_gradient(multiply, rhs, np.zeros_like(rhs), 1.0 / diagonal, SOLVE_ITERATIONS,
                                          SOLVE_TOLERANCE)


def close(rips, movable, spread):
    """Where every vertex goes when the rips close (world space, one row per vertex): both sides meet
    halfway, and the moves fade out over ``spread`` along the mesh. Vertices that aren't ``movable`` stay,
    and so does every other copy of their node. Every copy of a node moves alike, from where it is."""
    node, node_co = rips.node, rips.node_co
    count = len(node_co)
    fixed = np.zeros(count, dtype=bool)
    fixed[node[~movable]] = True
    # Each constraint: sum of coefficient * position = 0, over three nodes.
    pairs = rips.pairs
    tail, head = rips.tail[rips.point_edge], rips.head[rips.point_edge]
    nodes = np.concatenate((np.stack((pairs[:, 0], pairs[:, 1], pairs[:, 1]), axis=1),
                            np.stack((rips.point_node, tail, head), axis=1)))
    weights = np.concatenate((np.tile([1.0, -1.0, 0.0], (len(pairs), 1)),
                              np.stack((np.ones(len(tail)), rips.point_t - 1.0, -rips.point_t), axis=1)))
    if not len(nodes):
        return np.array(rips.co)
    offset = np.einsum("km,kmj->kj", weights, node_co[nodes])
    open_ = np.linalg.norm(offset, axis=1) > 1e-9
    if not open_.any():
        return np.array(rips.co)
    edges = _graph_edges(rips.tris)
    sources = np.zeros(count, dtype=bool)
    sources[nodes[open_].reshape(-1)] = True
    distance, _ = fitting._mesh_distance(node_co, edges, sources, spread, rips.tris)
    free = np.isfinite(distance) & ~fixed
    # The other side of a constraint moves as well, even when it's out of reach along the mesh.
    free[nodes[free[nodes].any(axis=1)].reshape(-1)] = True
    free &= ~fixed
    index = np.full(count, -1, dtype=np.int64)
    index[free] = np.arange(int(free.sum()))
    size = int(free.sum())
    if not size:
        return np.array(rips.co)
    # The mesh holds its shape: each edge resists its two ends moving apart, the harder the shorter it is.
    length = np.linalg.norm(node_co[edges[:, 0]] - node_co[edges[:, 1]], axis=1)
    stiffness = 1.0 / np.maximum(length, 1e-9)
    stiffness /= stiffness.mean()
    rows, columns, entries = [np.arange(size)], [np.arange(size)], [np.full(size, 1e-9)]
    for a, b in ((0, 1), (1, 0)):
        here = free[edges[:, a]]
        rows.append(index[edges[here, a]])
        columns.append(index[edges[here, a]])
        entries.append(stiffness[here])
        both = here & free[edges[:, b]]
        rows.append(index[edges[both, a]])
        columns.append(index[edges[both, b]])
        entries.append(-stiffness[both])
    rhs = np.zeros((size, 3))
    for k in range(3):
        for m in range(3):
            both = free[nodes[:, k]] & free[nodes[:, m]] & (weights[:, k] != 0.0) & (weights[:, m] != 0.0)
            rows.append(index[nodes[both, k]])
            columns.append(index[nodes[both, m]])
            entries.append(HOLD * weights[both, k] * weights[both, m])
        here = free[nodes[:, k]] & (weights[:, k] != 0.0)
        np.add.at(rhs, index[nodes[here, k]], -HOLD * weights[here, k][:, None] * offset[here])
    move = np.zeros((count, 3))
    move[free] = _solve(size, np.concatenate(rows), np.concatenate(columns), np.concatenate(entries), rhs)
    return rips.co + move[node]


def weld_plan(rips, co):
    """What joins the sides of the rips for good, with the vertices at ``co`` (after `close`): a list of
    (vertex, tail vertex, head vertex, t), each to merge the vertex with the point ``t`` of the way along the
    open edge from its tail to its head vertex (t = 0 or 1: with that vertex itself). Paired vertices merge
    with each other; the vertices of the other side are put on the coarser side's edges."""
    node = rips.node
    # A vertex of each node on the open edges, to merge.
    rim = np.full(len(rips.node_co), -1, dtype=np.int64)
    rim[rips.tail] = rips.tail_vertex
    rim[rips.head] = rips.head_vertex
    plan = [(int(rim[a]), int(rim[b]), int(rim[b]), 0.0) for a, b in rips.pairs.tolist()]
    paired = np.zeros(len(rips.node_co), dtype=bool)
    paired[rips.pairs.reshape(-1)] = True
    vertex, edge, _ = rips.faced
    for v, e in zip(vertex.tolist(), edge.tolist()):
        if paired[v]:
            continue
        a, b = int(rips.tail_vertex[e]), int(rips.head_vertex[e])
        point = co[rim[v]]
        t, _gap = _closest_on_edges(point[None], co[a][None], co[b][None])
        t = float(t[0])
        if t < WELD_END:
            t = 0.0
        elif t > 1.0 - WELD_END:
            t = 1.0
        plan.append((int(rim[v]), a, b, t))
    return plan


def weld(bm, plan):
    """Carry out a `weld_plan` (vertex indices of ``bm``) on the BMesh ``bm``: split the edges, then merge.
    Merges that would collapse an edge or a face, or give an edge more than two faces (where the sides
    overlap), are left out. Returns how many vertices were merged away."""
    bm.verts.ensure_lookup_table()
    verts = list(bm.verts)
    merges = []
    splits = {}
    for vertex, tail, head, t in plan:
        if t <= 0.0:
            merges.append((verts[vertex], verts[tail]))
        elif t >= 1.0:
            merges.append((verts[vertex], verts[head]))
        else:
            splits.setdefault((tail, head), []).append((t, vertex))
    for (tail, head), items in splits.items():
        start, end = verts[tail], verts[head]
        done = 0.0
        for t, vertex in sorted(items):
            if t <= done:
                # At the same spot as the vertex before it.
                merges.append((verts[vertex], start))
                continue
            edge = bm.edges.get((start, end))
            if edge is None:
                break
            # The new vertex lands this share of the way from `start`, along what's left of the edge.
            _edge, made = bmesh.utils.edge_split(edge, start, (t - done) / (1.0 - done))
            merges.append((verts[vertex], made))
            start, done = made, t
    # Merge in groups, each into one of its vertices, as long as no edge ends up with more than two faces
    # (and nothing collapses).
    root = {}
    members = {}

    def find_root(vert):
        while root.get(vert, vert) is not vert:
            vert = root[vert]
        return vert

    def group(vert):
        return members.get(vert, (vert,))

    def faces_between(first, second):
        others = set(group(second))
        return sum(len(edge.link_faces) for vert in group(first) for edge in vert.link_edges
                   if edge.other_vert(vert) in others)

    def can_merge(ra, rb):
        others = set(group(rb))
        if any(edge.other_vert(vert) in others for vert in group(ra) for edge in vert.link_edges):
            return False
        around = {find_root(edge.other_vert(vert)) for vert in group(ra) for edge in vert.link_edges}
        return all(faces_between(ra, rx) + faces_between(rb, rx) <= 2 for rx in around - {ra, rb})

    for a, b in merges:
        ra, rb = find_root(a), find_root(b)
        if ra is not rb and can_merge(ra, rb):
            root[ra] = rb
            members[rb] = group(rb) + group(ra)
    targetmap = {}
    for vert in root:
        target = find_root(vert)
        if target is not vert:
            targetmap[vert] = target
    if targetmap:
        targets = set(targetmap.values())
        bmesh.ops.weld_verts(bm, targetmap=targetmap)
        # Vertices that couldn't merge (they share a face) can still end up at the same spot: dissolve the
        # edges between them, so no face folds onto itself.
        edges = {edge for vert in targets if vert.is_valid for edge in vert.link_edges}
        bmesh.ops.dissolve_degenerate(bm, dist=DEGENERATE, edges=list(edges))
    return len(targetmap)
