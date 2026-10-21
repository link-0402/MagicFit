# SPDX-License-Identifier: GPL-3.0-or-later

"""Hair rig: a hair skeleton built for the hair itself, one chain of bones per lock of hair.

The game's hair skeletons are made for the game's hairstyles; new hair often hangs where none of them
has a chain. Own Skeleton builds the chains from the mesh. It first splits the hanging hair into locks:

- What hangs: vertices whose hanging share (`hairing.hanging_share`, with the settings' Hang Start and
  Hang Length) is above LOCK_START, just past Hang Start. The scalp stays with the head.
- The hanging vertices fall into regions, the connected pieces of the mesh (welded) they make. A region's
  roots are its vertices next to the scalp, where the hair leaves the head; a region that hangs entirely
  (a strand off the head) has its roots where it comes closest to the head.
- Every vertex is measured along the mesh from the nearest root (a Dijkstra from all of them at once): how
  far down the hair it is, and which root it hangs from.
- Where a vertex is around the head is its arc: its angle around the head's vertical axis (through the
  head bone, as `hairing.chain_terms` measures it), from the region's mean angle, times the mean distance
  of the roots from the axis. A vertex has two: its own, and its root's (where its hair left the scalp).
- A loose part that mostly hangs and leaves the head across no more than WHOLE_WIDTH (a ponytail, a braid,
  a strand) stays one lock. Anything else is cut into locks about LOCK_WIDTH wide, counting both how wide
  its roots are and, at HANG_DIRECTION_WEIGHT, how wide it hangs: hair leaving the head in one piece and
  parting as it falls makes two locks. The cut is a k-means on those two arcs, by area, so the cuts run
  down the hair as hair parts.
- Locks smaller than MIN_LOCK_AREA join the lock whose root is nearest, and every lock is measured again
  from its own roots.

Like `hairing`, nothing here depends on the 3D view.
"""

import heapq
import math

import numpy as np

from . import fitting
from .hairing import HEAD, components, hanging_share, unique_triangles, vertex_areas

# A vertex belongs to a lock when its hanging share (the settings' Hang Start and Hang Length) is above
# this: just past Hang Start, so a lock's root is where the hair leaves the head.
LOCK_START = 0.05
# The lock width aimed at (metres): a region is cut into max(1, round(span / LOCK_WIDTH)) locks, where
# span = max(root span, HANG_DIRECTION_WEIGHT * own span).
LOCK_WIDTH = 0.05
# A region whose loose part hangs for at least half its area, with a root span up to this (metres), stays
# one lock: a ponytail, braid or single strand.
WHOLE_WIDTH = 0.10
# Weight of where a vertex hangs (its own arc) next to where it left the scalp (its root's arc) when
# counting and cutting locks.
HANG_DIRECTION_WEIGHT = 0.5
# Smaller locks (m²) join the lock with the nearest root.
MIN_LOCK_AREA = 0.0004
# A region that hangs entirely has its roots where it is within this (metres) of its closest point to the
# head.
ROOT_BAND = 0.01
# Lloyd steps of the k-means that cuts a region into locks.
CUT_STEPS = 10


class Lock:
    """A lock of hair: ``vertices`` (indices of the points), ``distance`` (metres along the mesh from the
    lock's roots, one per vertex; 0 at the roots), ``area`` (m²), ``root`` ((3,), world space: the
    area-weighted mean of its roots), ``whole`` (a loose part kept whole, see WHOLE_WIDTH) and ``angle``
    (of the root around the head's vertical axis, degrees: 0 straight in front, 180 straight behind)."""

    __slots__ = ("vertices", "distance", "area", "root", "whole", "angle")

    def __init__(self, vertices, distance, area, root, whole, angle):
        self.vertices = vertices
        self.distance = distance
        self.area = area
        self.root = root
        self.whole = whole
        self.angle = angle


def _dijkstra(count, edges, lengths, sources):
    """Distance along ``edges`` ((m, 2)) of ``lengths`` from the nearest of ``sources``, for each of
    ``count`` nodes, and which source that is: (distances, sources); inf and -1 where none reaches."""
    a = np.concatenate((edges[:, 0], edges[:, 1]))
    b = np.concatenate((edges[:, 1], edges[:, 0]))
    w = np.concatenate((lengths, lengths))
    order = np.argsort(a, kind='stable')
    ptr = np.searchsorted(a[order], np.arange(count + 1)).tolist()
    to, wl = b[order].tolist(), w[order].tolist()
    dist = [math.inf] * count
    source = [-1] * count
    heap = []
    for s in np.asarray(sources).tolist():
        dist[s] = 0.0
        source[s] = s
        heap.append((0.0, s))
    heapq.heapify(heap)
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        for j in range(ptr[v], ptr[v + 1]):
            u = to[j]
            nd = d + wl[j]
            if nd < dist[u]:
                dist[u] = nd
                source[u] = source[v]
                heapq.heappush(heap, (nd, u))
    return np.array(dist), np.array(source, dtype=np.int64)


def _closest_to_head(nodes, head):
    """Those of ``nodes`` within ROOT_BAND of their closest to the head."""
    return nodes[head[nodes] <= head[nodes].min() + ROOT_BAND]


def _cut(features, weights, seed, k):
    """Labels (0..k-1) of an area-weighted k-means of ``features`` ((n, 2)), starting from k equal bins of
    column ``seed``, then CUT_STEPS Lloyd steps."""
    column = features[:, seed]
    low, high = float(column.min()), float(column.max())
    labels = np.clip(((column - low) / max(high - low, 1e-12) * k).astype(np.int64), 0, k - 1)
    centers = np.zeros((k, 2))
    for c in range(k):
        mine = labels == c
        if mine.any():
            centers[c] = np.average(features[mine], axis=0, weights=weights[mine])
        else:
            centers[c] = np.average(features, axis=0, weights=weights)
            centers[c, seed] = low + (c + 0.5) * (high - low) / k
    for _ in range(CUT_STEPS):
        labels = np.argmin(((features[:, None, :] - centers[None]) ** 2).sum(2), axis=1)
        for c in range(k):
            mine = labels == c
            if mine.any():
                centers[c] = np.average(features[mine], axis=0, weights=weights[mine])
    return np.argmin(((features[:, None, :] - centers[None]) ** 2).sum(2), axis=1)


def find_locks(points, tris, rig, *, hang_start=0.025, hang_length=0.07, weld=None):
    """The locks of hair ``points`` (world space, rest pose, (n, 3)) with triangles ``tris`` hanging from
    ``rig`` (`hairing.HairRig`), sorted by angle and then by the root's X. ``weld`` (vertex -> welded group)
    joins vertices sharing a position; by default they are found here."""
    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if weld is None:
        weld, count = fitting.weld_groups(points, fitting.WELD_DISTANCE)
    else:
        weld = np.asarray(weld, dtype=np.int64)
        count = int(weld.max()) + 1 if len(weld) else 0
    if not count or not len(tris):
        return []
    rest = np.zeros((count, 3))
    rest[weld] = points
    model = rig.model(rest)
    t = unique_triangles(weld[tris])
    if not len(t):
        return []
    edges = np.unique(np.sort(np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(model[edges[:, 0]] - model[edges[:, 1]], axis=1)
    area = vertex_areas(model, t)
    weight = area + 1e-12
    head = rig.head_distance(rest)
    used = np.bincount(t.ravel(), minlength=count) > 0
    hanging = used & (hanging_share(head, hang_start, hang_length) > LOCK_START)
    if not hanging.any():
        return []

    # Around the head's vertical axis: 0 straight in front (-Y), positive toward +X.
    axis_y = rig.by_name[HEAD].head[1]
    theta = np.arctan2(model[:, 0], -(model[:, 1] - axis_y))
    radius = np.hypot(model[:, 0], model[:, 1] - axis_y)

    part = components(count, edges)
    part_area = np.bincount(part, area)
    part_hanging = np.bincount(part, area * hanging)
    # Roots: hanging vertices next to the scalp.
    across = hanging[edges[:, 0]] != hanging[edges[:, 1]]
    is_root = np.zeros(count, dtype=bool)
    is_root[np.where(hanging[edges[across, 0]], edges[across, 0], edges[across, 1])] = True
    inside = hanging[edges[:, 0]] & hanging[edges[:, 1]]
    region = components(count, edges[inside])
    hanging_nodes = np.flatnonzero(hanging)
    order = np.argsort(region[hanging_nodes], kind='stable')
    hanging_nodes = hanging_nodes[order]
    starts = np.flatnonzero(np.diff(region[hanging_nodes], prepend=-1))

    regions = np.split(hanging_nodes, starts[1:])
    region_roots = []
    for nodes in regions:
        roots = nodes[is_root[nodes]]
        region_roots.append(roots if len(roots) else _closest_to_head(nodes, head))
    # The regions don't touch, so one Dijkstra over all of them measures each from its own roots.
    region_distance, source = _dijkstra(count, edges[inside], lengths[inside], np.concatenate(region_roots))

    drafts = []   # (nodes, roots, whole)
    arc = np.zeros(count)
    for nodes, roots in zip(regions, region_roots):
        mean = math.atan2(float((weight[nodes] * np.sin(theta[nodes])).sum()),
                          float((weight[nodes] * np.cos(theta[nodes])).sum()))
        scale = float(np.average(radius[roots], weights=weight[roots]))
        arc[nodes] = ((theta[nodes] - mean + math.pi) % (2.0 * math.pi) - math.pi) * scale
        own = arc[nodes]
        root_arc = arc[source[nodes]]
        root_span = float(np.ptp(arc[roots]))
        own_span = float(np.ptp(own))
        p = part[nodes[0]]
        if part_hanging[p] >= 0.5 * part_area[p] and root_span <= WHOLE_WIDTH:
            drafts.append((nodes, roots, True))
            continue
        span = max(root_span, HANG_DIRECTION_WEIGHT * own_span)
        k = max(1, int(round(span / LOCK_WIDTH)))
        if k == 1:
            drafts.append((nodes, roots, False))
            continue
        features = np.column_stack((root_arc, HANG_DIRECTION_WEIGHT * own))
        labels = _cut(features, weight[nodes], 0 if root_span >= HANG_DIRECTION_WEIGHT * own_span else 1, k)
        for c in range(k):
            mine = nodes[labels == c]
            if not len(mine):
                continue
            own_roots = mine[np.isin(mine, roots)]
            drafts.append((mine, own_roots if len(own_roots) else _closest_to_head(mine, head), False))

    # Small locks join the lock with the nearest root, the smallest first. Each lock's root is kept as the
    # weighted sum of its roots' positions and their weight.
    owner = np.full(count, -1, dtype=np.int64)
    is_lock_root = np.zeros(count, dtype=bool)
    for index, (nodes, roots, _whole) in enumerate(drafts):
        owner[nodes] = index
        is_lock_root[roots] = True
    alive = np.ones(len(drafts), dtype=bool)
    sizes = np.array([area[nodes].sum() for nodes, _roots, _whole in drafts])
    root_sum = np.array([(weight[roots, None] * rest[roots]).sum(0) for _nodes, roots, _whole in drafts]).reshape(-1, 3)
    root_weight = np.array([weight[roots].sum() for _nodes, roots, _whole in drafts])
    while alive.sum() > 1:
        small = int(np.argmin(np.where(alive, sizes, np.inf)))
        if sizes[small] >= MIN_LOCK_AREA:
            break
        alive[small] = False
        at = root_sum / root_weight[:, None]
        gap = np.linalg.norm(at - at[small], axis=1)
        into = int(np.argmin(np.where(alive, gap, np.inf)))
        owner[owner == small] = into
        sizes[into] += sizes[small]
        root_sum[into] += root_sum[small]
        root_weight[into] += root_weight[small]

    # Every lock measured again from its own roots, along itself; a piece its roots don't reach that way (a
    # cut runs around it) keeps its distance from the region's roots.
    within = (owner[edges[:, 0]] == owner[edges[:, 1]]) & (owner[edges[:, 0]] >= 0)
    distance, _source = _dijkstra(count, edges[within], lengths[within], np.flatnonzero(is_lock_root))
    distance = np.where(np.isfinite(distance), distance, region_distance)
    point_owner = owner[weld]
    by_owner = np.argsort(point_owner, kind='stable')
    bounds = np.searchsorted(point_owner[by_owner], np.arange(len(drafts) + 1))
    locks = []
    for index in np.flatnonzero(alive):
        vertices = by_owner[bounds[index]:bounds[index + 1]]
        root = root_sum[index] / root_weight[index]
        at = rig.model(root)
        angle = math.degrees(math.atan2(abs(at[0]), -(at[1] - axis_y)))
        whole = drafts[index][2]
        locks.append(Lock(vertices, distance[weld[vertices]], float(sizes[index]), root, whole, angle))
    locks.sort(key=lambda lock: (lock.angle, lock.root[0]))
    return locks
