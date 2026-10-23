# SPDX-License-Identifier: GPL-3.0-or-later

"""Hair rig: a hair skeleton built for the hair itself, one chain of bones per lock of hair.

The game's hair skeletons are made for the game's hairstyles; new hair often hangs where none of them
has a chain. Own Skeleton builds the chains from the mesh. It first splits the hanging hair into locks:

- What hangs, by the head's scalp map (`hairing.HairRig.scalp`) and the hair's cap depth (`hairing.cap_depth`,
  how far its cap lies off the head): vertices over the face (a scalp share below SCALP_HALF), however close
  to the skin, and vertices over the scalp standing off the cap (more than STAND_MARGIN beyond the cap
  depth). The hair lying on the cap stays with the head.
- The hanging vertices fall into regions, the connected pieces of the mesh (welded) they make. A region's
  roots are its vertices next to the held hair, where the hair leaves the scalp: on the hairline for hair
  over the face, on the cap for hair standing off it. A region that hangs entirely (a strand off the head)
  has its roots where it comes closest to the scalp (`hairing.HairRig.scalp_gap`).
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
- Locks smaller than MIN_LOCK_AREA (crumbs: a card corner that barely hangs) join the nearest lock (by
  root) of at least that size, whichever loose part it is, keeping that lock's root; when there is none
  they are dropped, and their hair goes to the nearest chain. Crumbs never join each other. No chain starts
  on the neck or face: a lock without roots next to held hair (rooted where it comes closest to the scalp)
  whose root (the area-weighted mean of its roots) lies farther than ROOT_REACH from the scalp beyond the cap
  depth (an inner layer lying wholly on the neck) joins the same way. A lock rooted next to held hair leaves
  the scalp or the cap there and keeps its chain. Every lock is then measured again from its roots.
- A lock is light when most of its roots (by area) are over the scalp: it stands off the cap (a bun) rather
  than leaving the scalp at the hairline. Its chain is light too; two locks merged are light when both are.

Then it plans a chain of bones down each lock:

- The lock's spine is its middle line down the hair: its root, then the centroid of its vertices in each
  SPINE_STEP band of distance from the root, smoothed over SPINE_SMOOTHING samples (on a coarse mesh a band
  mixes neighbouring rows, and the centroids zig-zag). A spine shorter than MIN_CHAIN_LENGTH gets no chain.
- The chain has a bone per BONE_SPACING of spine (2 to 5), its joints spread evenly along the spine from the
  root, the last bone ending at the spine's end.
- Its kind (bangs, side, back or ponytail) comes from where the lock leaves the head and how long it is; only
  a round lock (PONYTAIL_ROUND) is a ponytail, a flat card lying on back hair is back hair.
- Then locks merge. Twin layers always do: of two locks whose roots lie within TWIN_REACH of each other (a
  mesh doubled for its back faces), the smaller merges into the larger. Locks of one kind whose roots lie
  closer than the chain spacing (CHAIN_SPACING, the Chain Spacing setting) merge too: the smallest such lock
  into the nearest of its kind (by root). Kinds don't mix here, so the fringe keeps its chain next to the
  side hair. While the rig still takes more than MAX_HAIR_BONES bones, the smallest lock that shares its
  kind with another merges into the nearest of that kind, so a lone kind (one fringe among back hair) keeps
  its chain; only when no two locks share a kind does the smallest merge into the nearest of any kind.
- A merge (other than of twins) stays on the hair: it is refused when the merged lock's chain has a joint or
  its tip farther than SPINE_REACH from the lock's own hair (two locks far apart, its spine running through
  the space between them), and the next nearest is tried. A lock with no partner that keeps it on the hair
  stays as it is; over the budget the next smallest lock merges instead, and only when no lock has a merge
  that stays on the hair does the smallest merge into the nearest after all.
- The merged lock hangs from the roots of both (its root is their mean) and is planned again. It is kept
  whole only when both locks were and one's roots lie within the other's, no wider than WHOLE_WIDTH: layers
  of one ponytail. Strands or cards merged side by side make back hair.

The chains become bones on the hair's armature (`build_armature_rig`), j_ex_hair_01_a and on, hanging from
j_kao. They are ordinary Blender bones, head on a joint and tail on the next, so they draw along the hair
and the user can move, delete or add some in Edit Mode; `sync_rig` reads them back as they are and names
new ones; the rig state keeps a hash of the bones (`bones_hash`) to tell whether the user edited them since,
and for the weights the hair's cap depth and which chains are light (kept by chain number, like the kinds).
Each bone's Z axis points away from the head's axis, and its bone colour is its chain's kind's
(`hairview.color_bones`). Every hair bone, the user's new ones too, is in the armature's CHAIN_COLLECTION bone
collection, and only they are (`collect_hair_bones`); Show Chains solos it. The armature carries the rig as RIG_PROPERTY, written from
`Object.magic_fit_hair_rig` after every change, for exporters (Instant Edit) to make the hair skeleton. A
hair bone points along its X axis in the game and along its Y axis in Blender: ALONG_X turns one into the
other, here (`custom_rig`) and in the exporter.

Like `hairing`, nothing here depends on the 3D view.
"""

import hashlib
import heapq
import json
import math
import re

import numpy as np

from . import fitting, hairing
from .hairing import HEAD, SCALP_HALF, STAND_MARGIN, HairError, components, unique_triangles, vertex_areas

# A lock without roots next to held hair whose root lies farther than this (metres) from the scalp, beyond the
# hair's cap depth (`hairing.HairRig.scalp_gap`), gets no chain: its hair joins the nearest lock rooted at the
# scalp.
ROOT_REACH = 0.03
# The lock width aimed at (metres): a region is cut into max(1, round(span / LOCK_WIDTH)) locks, where
# span = max(root span, HANG_DIRECTION_WEIGHT * own span).
LOCK_WIDTH = 0.05
# A region whose loose part hangs for at least half its area, with a root span up to this (metres), stays
# one lock: a ponytail, braid or single strand.
WHOLE_WIDTH = 0.10
# Weight of where a vertex hangs (its own arc) next to where it left the scalp (its root's arc) when
# counting and cutting locks.
HANG_DIRECTION_WEIGHT = 0.5
# Smaller locks (m²) join the nearest lock (by root) of at least this size rooted at the scalp (ROOT_REACH),
# across loose parts if need be; with none, they are dropped (their hair goes to the nearest chain). They
# never join each other.
MIN_LOCK_AREA = 0.0004
# A region that hangs entirely has its roots where it is within this (metres) of its closest point to the
# scalp.
ROOT_BAND = 0.01
# Lloyd steps of the k-means that cuts a region into locks.
CUT_STEPS = 10
# Spine samples along the geodesic distance from the root (metres): one per band this wide.
SPINE_STEP = 0.01
# Samples in the centred moving average that smooths the spine (the first and last stay as they are): on a
# coarse mesh one band holds the back of one ring and the front of the next, and the samples zig-zag.
SPINE_SMOOTHING = 5
# A shorter spine (metres) gets no chain; its hair goes to the nearest chain.
MIN_CHAIN_LENGTH = 0.03
# Bones per chain: clamp(ceil(length / BONE_SPACING), CHAIN_BONES), evenly along the spine (metres).
BONE_SPACING = 0.10
# The fewest and most bones in a chain.
CHAIN_BONES = (2, 5)
# The budget of the whole rig: locks merge until the chains fit (the smallest into the nearest of its kind,
# see plan_rig).
MAX_HAIR_BONES = 48
# Two locks whose roots lie within this (metres) of each other are twin layers (a mesh doubled for its back
# faces): they always merge, the smaller into the larger.
TWIN_REACH = 0.005
# Locks of one kind whose roots lie closer than this (metres) merge, budget or not: the chains follow the
# hair's layout, not every card (plan_rig's ``spacing``, the Chain Spacing setting).
CHAIN_SPACING = 0.10
# A merge is refused when the merged lock's chain has a joint or its tip farther than this (metres) from the
# lock's own hair: its spine would run through the space between the two locks.
SPINE_REACH = 0.03
# Two whole locks merged are still whole (WHOLE_WIDTH) only when one's roots lie within the other's around the
# head, give or take this (radians): layers of one ponytail. Strands or cards side by side make back hair.
WITHIN_SLACK = 1e-6
# The kinds of chain, by where the lock leaves the head and how long it is (see chain_kind).
KINDS = ("bangs", "side", "back", "ponytail")
# Bangs: a root this close to straight in front (degrees around the head's axis, 0 in front)...
BANGS_ANGLE = 45.0
# ...and a spine no longer than this (metres).
BANGS_LENGTH = 0.20
# Side: a root this close to the front (degrees); farther round it is back hair.
SIDE_ANGLE = 110.0
# Ponytail: a lock kept whole (WHOLE_WIDTH) with a spine at least this long (metres)...
PONYTAIL_LENGTH = 0.30
# ...and round: its hair spreads at least this far (metres, the square root of the second eigenvalue of its
# vertices about the spine) across the spine both ways. A flat card lying on back hair is back hair: Maeve's
# centre cards spread at most 0.0074, a tube ponytail 0.017.
PONYTAIL_ROUND = 0.01
# The custom property on the armature object that holds the rig for exporters, as JSON: {"race": "c0801",
# "chains": [{"kind": "back", "bones": ["j_ex_hair_01_a", ...]}]}, each chain's bones root to tip.
RIG_PROPERTY = "xiv_hair_rig"
# The hair's own bones are named BONE_PREFIX, the chain's number (01 on) and the bone's letter (a from the
# root): j_ex_hair_01_a. Exporters accept no other names (BONE_PATTERN).
BONE_PREFIX = "j_ex_hair_"
BONE_PATTERN = re.compile(r"j_ex_hair_([0-9]{2})_([a-z])")
# The bone collection on the armature that holds every hair bone (BONE_PREFIX), made when missing; Show Chains
# solos it (`hairview`).
CHAIN_COLLECTION = "Hair Chains"
# The most chains the names allow.
MAX_CHAINS = 99
# Decimals (of metres, and of the bones' Z axes) `bones_hash` rounds the hair's bones to.
HASH_PRECISION = 6
# Chains built again when a part joins the hair keep the kind of the chain before whose first bone starts
# within this of theirs (half a lock): the same lock, so a kind the user picked in the list stays.
KIND_REACH = 0.5 * LOCK_WIDTH
# The game's frame of a hair bone from the Blender bone's (rest @ ALONG_X): the game's X (along which a
# hair bone points) is the bone's +Y (head to tail), the game's Y its -X, and Z stays.
ALONG_X = np.array(((0.0, -1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0)))


class Lock:
    """A lock of hair: ``vertices`` (indices of the points), ``distance`` (metres from the lock's root, one
    per vertex: along the mesh from its roots, 0 there; for a joined crumb's vertices, in a straight line
    from ``root``), ``area`` (m²), ``root`` ((3,), world space: the area-weighted mean of its own roots, not
    those of crumbs that joined it), ``whole`` (a loose part kept whole, see WHOLE_WIDTH), ``angle`` (of the
    root around the head's vertical axis, degrees: 0 straight in front, 180 straight behind) and ``joined``
    (bool per vertex: it came from a crumb that joined the lock, see MIN_LOCK_AREA). The chain's spine
    skips the ``joined`` vertices; they only take its weights. ``areas`` (m² per vertex, a welded vertex's
    area shared among its points) weighs the spine; None weighs every vertex alike. ``measured`` (bool per
    vertex) marks the vertices whose distance was measured along the lock from its roots, the others being
    joined or out of their reach; None when that isn't known. A ``light`` lock has most of its roots (by
    area) over the scalp: it stands off the cap rather than leaving the scalp at the hairline."""

    __slots__ = ("vertices", "distance", "area", "root", "whole", "angle", "joined", "areas", "measured", "light")

    def __init__(self, vertices, distance, area, root, whole, angle, joined, areas=None, measured=None, light=False):
        self.vertices = vertices
        self.distance = distance
        self.area = area
        self.root = root
        self.whole = whole
        self.angle = angle
        self.joined = joined
        self.areas = areas
        self.measured = measured
        self.light = light


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


def _closest_to_scalp(nodes, gap):
    """Those of ``nodes`` within ROOT_BAND of their closest to the scalp (``gap``, per node: its distance from
    the scalp, `hairing.HairRig.scalp_gap`)."""
    return nodes[gap[nodes] <= gap[nodes].min() + ROOT_BAND]


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


def _welded(points, tris, weld):
    """The mesh of ``points`` and ``tris`` with welded vertices made one: (weld, count, rest, triangles,
    edges), where ``rest`` ((count, 3)) is where each welded vertex is; None when there is no mesh. ``weld``
    (vertex -> welded group) is found here when None."""
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if weld is None:
        weld, count = fitting.weld_groups(points, fitting.WELD_DISTANCE)
    else:
        weld = np.asarray(weld, dtype=np.int64)
        count = int(weld.max()) + 1 if len(weld) else 0
    if not count or not len(tris):
        return None
    rest = np.zeros((count, 3))
    rest[weld] = points
    t = unique_triangles(weld[tris])
    if not len(t):
        return None
    edges = np.unique(np.sort(np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1), axis=0)
    return weld, count, rest, t, edges


def _angle(rig, root):
    """The angle (degrees) of world ``root`` around the head's vertical axis in ``rig``'s model space: 0
    straight in front (-Y), 180 straight behind, the same on either side."""
    at = rig.model(root)
    return math.degrees(math.atan2(abs(at[0]), -(at[1] - rig.by_name[HEAD].head[1])))


def _hanging(share, head, cap):
    """Whether hair with scalp shares ``share`` (`hairing.HairRig.scalp`) and head distances ``head`` hangs,
    for a hair whose cap lies ``cap`` off the head (`hairing.cap_depth`): over the face (a share below
    SCALP_HALF), or standing off the cap (more than STAND_MARGIN beyond it)."""
    return (share < SCALP_HALF) | (head > cap + STAND_MARGIN)


def hangs(points, rig, *, cap=None):
    """Whether any of hair ``points`` (world space, rest pose) hangs from ``rig``'s head (`_hanging`), for a
    hair whose cap lies ``cap`` off the head (`hairing.cap_depth`; None: CAP_FALLBACK)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cap = hairing.CAP_FALLBACK if cap is None else cap
    return bool(_hanging(rig.scalp(points), rig.head_distance(points), cap).any())


def find_locks(points, tris, rig, *, cap=None, weld=None):
    """The locks of hair ``points`` (world space, rest pose, (n, 3)) with triangles ``tris`` hanging from
    ``rig`` (`hairing.HairRig`), sorted by angle and then by the root's X. ``cap`` is how far the hair's cap
    lies off the head (`hairing.cap_depth`, measured from these points when None). ``weld`` (vertex ->
    welded group) joins vertices sharing a position; by default they are found here."""
    points = np.asarray(points, dtype=np.float64)
    mesh = _welded(points, tris, weld)
    if mesh is None:
        return []
    weld, count, rest, t, edges = mesh
    model = rig.model(rest)
    lengths = np.linalg.norm(model[edges[:, 0]] - model[edges[:, 1]], axis=1)
    area = vertex_areas(model, t)
    weight = area + 1e-12
    # A welded vertex's area shared among the points on it.
    point_area = (area / np.maximum(np.bincount(weld, minlength=count), 1))[weld]
    head = rig.head_distance(rest)
    share = rig.scalp(rest)
    if cap is None:
        cap = hairing.cap_depth(points, tris, rig, share=share[weld])
    used = np.bincount(t.ravel(), minlength=count) > 0
    hanging = used & _hanging(share, head, cap)
    if not hanging.any():
        return []
    # How far the hanging hair is from the scalp, for roots where none is next to held hair.
    gap = np.full(count, np.inf)
    gap[hanging] = rig.scalp_gap(rest[hanging])

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
        region_roots.append(roots if len(roots) else _closest_to_scalp(nodes, gap))
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
        # A region is connected and holds its roots, so the Dijkstra reached every vertex of it (a source of
        # -1 would read the last vertex's arc).
        assert (source[nodes] >= 0).all(), "a hanging region's vertex not reached from its roots"
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
            drafts.append((mine, own_roots if len(own_roots) else _closest_to_scalp(mine, gap), False))

    # Small locks (crumbs) join the nearest lock (by root) of at least MIN_LOCK_AREA, across loose parts if
    # need be, without moving its root; with none to join they are dropped. Crumbs never join each other. So
    # does a lock without roots next to held hair (rooted where it comes closest to the scalp) whose root lies
    # farther than ROOT_REACH from the scalp: a layer lying wholly on the neck. No chain starts on the neck or
    # face.
    owner = np.full(count, -1, dtype=np.int64)
    for index, (nodes, _roots, _whole) in enumerate(drafts):
        owner[nodes] = index
    sizes = np.array([area[nodes].sum() for nodes, _roots, _whole in drafts])
    roots_at = np.array([np.average(rest[roots], axis=0, weights=weight[roots]) for _nodes, roots, _whole in drafts])
    natural = np.array([bool(is_root[roots].any()) for _nodes, roots, _whole in drafts])
    alive = (sizes >= MIN_LOCK_AREA) & (natural | (rig.scalp_gap(roots_at, cap) <= ROOT_REACH))
    target = np.where(alive, np.arange(len(drafts)), -1)
    if alive.any():
        for small in np.flatnonzero(~alive):
            gap = np.linalg.norm(roots_at - roots_at[small], axis=1)
            into = int(np.argmin(np.where(alive, gap, np.inf)))
            target[small] = into
            sizes[into] += sizes[small]
    joined = (owner >= 0) & ~alive[np.maximum(owner, 0)]
    owner = np.where(owner >= 0, target[np.maximum(owner, 0)], -1)

    # Every lock measured again from its own roots, along itself; a piece its roots don't reach that way (a
    # cut runs around it) keeps its distance from the region's roots. A joined crumb, often on another loose
    # part, is as far from the lock's root as it is in a straight line.
    is_lock_root = np.zeros(count, dtype=bool)
    for index in np.flatnonzero(alive):
        is_lock_root[drafts[index][1]] = True
    within = (owner[edges[:, 0]] == owner[edges[:, 1]]) & (owner[edges[:, 0]] >= 0) & ~joined[edges[:, 0]] \
        & ~joined[edges[:, 1]]
    distance, _source = _dijkstra(count, edges[within], lengths[within], np.flatnonzero(is_lock_root))
    measured = np.isfinite(distance)
    distance = np.where(measured, distance, region_distance)
    point_owner = owner[weld]
    by_owner = np.argsort(point_owner, kind='stable')
    bounds = np.searchsorted(point_owner[by_owner], np.arange(len(drafts) + 1))
    locks = []
    for index in np.flatnonzero(alive):
        vertices = by_owner[bounds[index]:bounds[index + 1]]
        root = roots_at[index]
        crumb = joined[weld[vertices]]
        lock_distance = distance[weld[vertices]]
        lock_distance[crumb] = np.linalg.norm(rest[weld[vertices[crumb]]] - root, axis=1)
        angle = _angle(rig, root)
        _nodes, roots, whole = drafts[index]
        # Light: most of its roots (by area) over the scalp, standing off the cap.
        light = bool(weight[roots][share[roots] >= SCALP_HALF].sum() > 0.5 * weight[roots].sum())
        locks.append(Lock(vertices, lock_distance, float(sizes[index]), root, whole, angle, crumb,
                          point_area[vertices], measured[weld[vertices]] & ~crumb, light))
    # Mirrored locks have the same angle but for float noise, which mustn't decide their order (chain names
    # come from it): the lower root X first.
    locks.sort(key=lambda lock: (round(lock.angle, 6), lock.root[0]))
    return locks


# -----------------------------------------------------------------------------
# Chains

class ChainPlan:
    """The chain of bones planned for a lock: its ``kind`` (one of KINDS), ``joints`` ((n, 3), world space,
    the bones' heads from the root on, CHAIN_BONES[0] <= n <= CHAIN_BONES[1]), ``tip`` ((3,), world space,
    the lock's end, where the last bone ends), and the lock's ``area`` (m²), ``whole``, ``angle`` (degrees),
    ``vertices`` (indices of the points) and ``light`` (see `Lock`)."""

    __slots__ = ("kind", "joints", "tip", "area", "whole", "angle", "vertices", "light")

    def __init__(self, kind, joints, tip, area, whole, angle, vertices, light=False):
        self.kind = kind
        self.joints = joints
        self.tip = tip
        self.area = area
        self.whole = whole
        self.angle = angle
        self.vertices = vertices
        self.light = light


def chain_kind(angle, length, whole):
    """The kind of a chain whose lock leaves the head at ``angle`` (degrees around the head's vertical axis,
    0 straight in front, 180 straight behind), with a spine ``length`` (metres) long, kept ``whole`` or not
    (WHOLE_WIDTH): bangs in front if short, side hair up to SIDE_ANGLE, then a long whole lock is a ponytail
    and anything else back hair."""
    if angle <= BANGS_ANGLE and length <= BANGS_LENGTH:
        return "bangs"
    if angle <= SIDE_ANGLE:
        return "side"
    if whole and length >= PONYTAIL_LENGTH:
        return "ponytail"
    return "back"


def _smooth(line):
    """``line`` ((m, 3)) smoothed by a centred moving average of SPINE_SMOOTHING samples, narrower near the
    ends so it stays centred; the first and last samples stay as they are."""
    half = SPINE_SMOOTHING // 2
    count = len(line)
    index = np.arange(count)
    sums = np.vstack((np.zeros((1, 3)), np.cumsum(line, axis=0)))
    reach = np.minimum(np.minimum(index, count - 1 - index), half)
    low, high = index - reach, index + reach + 1
    smooth = (sums[high] - sums[low]) / (high - low)[:, None]
    smooth[[0, -1]] = line[[0, -1]]
    return smooth


def _bands(points, lock, step):
    """The lock's own (not joined) vertices, their areas, which ``step``-wide band of distance from the root
    each is in, and each band's area-weighted centroid (nearest band first, empty bands skipped)."""
    own = ~np.asarray(lock.joined, dtype=bool)
    vertices = lock.vertices[own]
    weight = (np.ones(len(vertices)) if lock.areas is None else lock.areas[own]) + 1e-12
    bands, band = np.unique(np.floor(lock.distance[own] / step).astype(np.int64), return_inverse=True)
    total = np.bincount(band, weight, minlength=len(bands))
    centroids = np.column_stack([np.bincount(band, weight * points[vertices, k], minlength=len(bands)) / total
                                 for k in range(3)])
    return vertices, weight, band, centroids


def spine(points, lock, step=SPINE_STEP):
    """The middle line of ``lock`` down the hair ((m, 3), world space like ``points``, root to end): the
    lock's root, then the area-weighted centroid of its vertices in each ``step``-wide band of distance from
    the root, nearest first (empty bands are skipped), smoothed (SPINE_SMOOTHING). Joined vertices (crumbs,
    measured in a straight line) are left out: they would pull the line toward wherever the crumb is."""
    points = np.asarray(points, dtype=np.float64)
    root = np.asarray(lock.root, dtype=np.float64).reshape(1, 3)
    vertices, _weight, _band, centroids = _bands(points, lock, step)
    if not len(vertices):
        return root
    return _smooth(np.vstack((root, centroids)))


def _round(points, lock):
    """Whether ``lock``'s own hair spreads at least PONYTAIL_ROUND across its spine both ways: the square root
    of the second eigenvalue of its vertices about their bands' centroids (`spine`). A tube is round, a flat
    card or a sheet of cards isn't."""
    points = np.asarray(points, dtype=np.float64)
    vertices, weight, band, centroids = _bands(points, lock, SPINE_STEP)
    if not len(vertices):
        return False
    offsets = points[vertices] - centroids[band]
    spread = np.linalg.eigvalsh((offsets * weight[:, None]).T @ offsets / weight.sum())
    return bool(spread[1] >= PONYTAIL_ROUND * PONYTAIL_ROUND)


def plan_chain(points, lock, rig=None):
    """The chain for ``lock`` (`ChainPlan`), or None when its spine is shorter than MIN_CHAIN_LENGTH. Its
    clamp(ceil(length / BONE_SPACING), CHAIN_BONES) joints are spread evenly along the spine by length: the
    first on the root, the next ones one bone length further each, the last one bone length short of the
    lock's end (the tip). With ``rig`` (`hairing.HairRig`), side hair whose tip lies behind SIDE_ANGLE is back
    hair: it falls back over the shoulder, and as side hair it would keep a chain of its own there."""
    line = spine(points, lock)
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    length = float(seg.sum())
    if length < MIN_CHAIN_LENGTH:
        return None
    # Points on one spot would make the interpolation's steps ambiguous.
    line = line[np.concatenate(([True], seg > 0.0))]
    at = np.concatenate(([0.0], np.cumsum(seg[seg > 0.0])))
    count = int(np.clip(math.ceil(length / BONE_SPACING), *CHAIN_BONES))
    marks = at[-1] * np.arange(count) / count
    joints = np.column_stack([np.interp(marks, at, line[:, k]) for k in range(3)])
    # Only a round lock can be a ponytail (PONYTAIL_ROUND); a flat card stays whole for merging all the same.
    kind = chain_kind(lock.angle, length, lock.whole and _round(points, lock))
    if kind == "side" and rig is not None and _angle(rig, line[-1]) > SIDE_ANGLE:
        kind = "back"
    return ChainPlan(kind, joints, line[-1].copy(), lock.area, lock.whole,
                     lock.angle, lock.vertices, lock.light)


def _roots(lock):
    """Positions in ``lock.vertices`` of its roots: its own vertices at distance 0 (the nearest one if
    none is)."""
    own = np.flatnonzero(~lock.joined)
    roots = own[lock.distance[own] <= 0.0]
    return roots if len(roots) else own[[int(np.argmin(lock.distance[own]))]]


def _graph(mesh):
    """The edges of ``mesh`` (`_welded`) by node: (ptr, to, edge), node v's neighbours being to[ptr[v]:ptr[v + 1]]
    along the mesh's edges edge[ptr[v]:ptr[v + 1]]."""
    _weld, count, _rest, _tris, edges = mesh
    a = np.concatenate((edges[:, 0], edges[:, 1]))
    order = np.argsort(a, kind='stable')
    to = np.concatenate((edges[:, 1], edges[:, 0]))[order]
    edge = np.concatenate((np.arange(len(edges)), np.arange(len(edges))))[order]
    return np.searchsorted(a[order], np.arange(count + 1)), to, edge


def _edges_from(graph, nodes, inside):
    """The numbers of the edges (`_graph`) from ``nodes`` (unique) to nodes where ``inside`` (bool per node)
    is set, in no order; an edge between two of ``nodes`` comes twice."""
    ptr, to, edge = graph
    starts = ptr[nodes]
    counts = ptr[nodes + 1] - starts
    at = np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(int(counts.sum()))
    return edge[at[inside[to[at]]]]


def _root_arcs(rig, at, weights):
    """The angles (radians) of world points ``at`` around the head's vertical axis in ``rig``'s model space,
    about their ``weights``-weighted circular mean, and their weighted mean distance from the axis (metres):
    an angle times that distance is an arc, as find_locks measures root spans."""
    model = rig.model(at)
    axis_y = rig.by_name[HEAD].head[1]
    theta = np.arctan2(model[:, 0], -(model[:, 1] - axis_y))
    mean = math.atan2(float((weights * np.sin(theta)).sum()), float((weights * np.cos(theta)).sum()))
    radius = np.hypot(model[:, 0], model[:, 1] - axis_y)
    return (theta - mean + math.pi) % (2.0 * math.pi) - math.pi, float(np.average(radius, weights=weights))


def _merge(target, small, mesh, points, rig, graph=None):
    """The lock ``small`` merged into ``target`` (`Lock`s) on ``mesh`` (`_welded`, with its `_graph`, found
    here when None): both locks' vertices and area, ``whole`` when both are and one's roots lie within the
    other's (WITHIN_SLACK), spanning at most WHOLE_WIDTH around the head (`_root_arcs`). The roots of both
    are the merged lock's roots: its root is their area-weighted mean, and its distances are measured again
    from them along the hair of both, so the small lock's own vertices are measured from its own roots when
    it is another loose part. Vertices of the target those roots don't reach keep their distance, as
    find_locks measured it; those of the small lock join like a crumb. Crumbs are measured in a straight line
    from the new root. The merged lock is light when both are.

    When the two locks share no edge (separate loose parts), each one's distances from its own roots stand
    as they are; otherwise they are measured again on the merged lock's own part of the mesh, not the whole
    mesh. Both give the same distances: a merge runs in time with the size of the locks, not of the hair."""
    weld, count, rest, _tris, edges = mesh
    if graph is None:
        graph = _graph(mesh)
    vertices = np.concatenate((target.vertices, small.vertices))
    order = np.argsort(vertices, kind='stable')
    vertices = vertices[order]
    joined = np.concatenate((target.joined, small.joined))[order]
    distance = np.concatenate((target.distance, small.distance)).astype(np.float64)[order]
    from_small = (np.arange(len(order)) >= len(target.vertices))[order]
    areas = None
    if target.areas is not None and small.areas is not None:
        areas = np.concatenate((target.areas, small.areas))[order]
    roots = np.argsort(order)[np.concatenate((_roots(target), len(target.vertices) + _roots(small)))]
    weight = (np.ones(len(vertices)) if areas is None else areas)[roots] + 1e-12
    root = np.average(points[vertices[roots]], axis=0, weights=weight)
    nodes = weld[vertices]
    own = ~joined
    target_free = np.zeros(count, dtype=bool)
    target_free[weld[target.vertices[~target.joined]]] = True
    small_free = np.unique(weld[small.vertices[~small.joined]])
    if target.measured is not None and small.measured is not None \
            and not len(_edges_from(graph, small_free, target_free)):
        # Nothing joins the two: measured from the roots of both, each keeps its distances from its own.
        reached = np.concatenate((target.measured, small.measured))[order] & own
    else:
        free_nodes = np.unique(nodes[own])
        free = target_free
        free[small_free] = True
        sub = np.unique(_edges_from(graph, free_nodes, free))
        lengths = np.linalg.norm(rest[edges[sub, 0]] - rest[edges[sub, 1]], axis=1)
        local, _source = _dijkstra(len(free_nodes), np.searchsorted(free_nodes, edges[sub]).reshape(-1, 2), lengths,
                                   np.searchsorted(free_nodes, np.unique(nodes[roots])))
        measured = np.full(len(vertices), math.inf)
        measured[own] = local[np.searchsorted(free_nodes, nodes[own])]
        reached = own & np.isfinite(measured)
        distance[reached] = measured[reached]
    joined = joined | (from_small & ~reached)
    distance[joined] = np.linalg.norm(points[vertices[joined]] - root, axis=1)
    # Whole when both are, their roots span at most WHOLE_WIDTH together, and one's roots lie within the
    # other's (around the head): layers of one ponytail, not strands or cards side by side.
    whole = bool(target.whole and small.whole)
    if whole:
        arcs, scale = _root_arcs(rig, points[vertices[roots]], weight)
        mine = ~from_small[roots]
        span = float(np.ptp(arcs))
        whole = span * scale <= WHOLE_WIDTH and span <= max(float(np.ptp(arcs[mine])), float(np.ptp(arcs[~mine]))) \
            + WITHIN_SLACK
    return Lock(vertices, distance, target.area + small.area, root, whole, _angle(rig, root), joined, areas, reached,
                target.light and small.light)


def _on_hair(points, lock, chain):
    """Whether ``chain`` (`ChainPlan`, or None) runs down ``lock``'s own hair: its joints and tip all within
    SPINE_REACH of the lock's own (not joined) vertices."""
    own = points[lock.vertices[~lock.joined]]
    if chain is None or not len(own):
        return False
    line = np.vstack((chain.joints, chain.tip))
    gap = np.min(((own[None, :, :] - line[:, None, :]) ** 2).sum(axis=2), axis=1)
    return bool(gap.max() <= SPINE_REACH * SPINE_REACH)


def plan_rig(points, tris, rig, *, cap=None, weld=None, max_bones=MAX_HAIR_BONES, spacing=CHAIN_SPACING):
    """The chains of a hair skeleton for hair ``points`` (world space, rest pose) with triangles ``tris``
    hanging from ``rig`` (`hairing.HairRig`), its cap ``cap`` off the head (see `find_locks`): one
    `ChainPlan` per lock (`find_locks`) whose spine is long enough, in name order (by angle, then by the
    root's X). Locks merge (both hang from both roots, see `_merge`) and the merged lock is planned again:
    twin layers (roots within TWIN_REACH) always, the smaller into the larger; then, while any two of one
    kind have roots closer than ``spacing`` (metres), the smallest of them into the nearest (by root) of its
    kind. While the chains then take more than ``max_bones`` bones, the smallest lock that shares its kind
    with another merges into the nearest (by root) of that kind; when no two share a kind, the smallest into
    the nearest of any kind. A merge other than of twins whose chain leaves the hair (`_on_hair`) is refused
    and the next nearest tried; when none is left, the lock stays as it is. Over the budget the next smallest
    lock is tried then, and only when no lock has a merge that stays on the hair does the smallest merge into
    its nearest after all."""
    points = np.asarray(points, dtype=np.float64)
    mesh = _welded(points, tris, weld)
    if mesh is None:
        return []
    locks = find_locks(points, tris, rig, cap=cap, weld=mesh[0])
    planned = [(lock, plan_chain(points, lock, rig)) for lock in locks]
    planned = [(lock, chain) for lock, chain in planned if chain is not None]
    # The locks stay in their places in ``planned``, the merged ones marked gone (not ``alive``), so the
    # smallest and the nearest are the first ones in the same order as among the locks left.
    alive = np.ones(len(planned), dtype=bool)
    areas = np.array([lock.area for lock, _chain in planned], dtype=np.float64)
    roots = np.array([lock.root for lock, _chain in planned], dtype=np.float64).reshape(-1, 3)
    kinds = np.array([KINDS.index(chain.kind) for _lock, chain in planned], dtype=np.int64)
    left = len(planned)
    bones = sum(len(chain.joints) for _lock, chain in planned)
    graph = _graph(mesh) if left > 1 else None
    # Each lock's nearest other lock (by root), to find twins, and its nearest of its own kind, for the
    # spacing, kept up to date as locks merge.
    near = np.full(len(planned), math.inf)
    near_to = np.zeros(len(planned), dtype=np.int64)
    kin = np.full(len(planned), math.inf)
    kin_to = np.zeros(len(planned), dtype=np.int64)
    rows = np.arange(len(planned))
    for start in range(0, len(planned) if left > 1 else 0, 512):
        gap = np.linalg.norm(roots[start:start + 512, None, :] - roots[None, :, :], axis=2)
        at = rows[:len(gap)]
        gap[at, start + at] = math.inf
        near_to[start:start + 512] = np.argmin(gap, axis=1)
        near[start:start + 512] = gap[at, near_to[start:start + 512]]
        gap[kinds[start:start + 512, None] != kinds[None, :]] = math.inf
        kin_to[start:start + 512] = np.argmin(gap, axis=1)
        kin[start:start + 512] = gap[at, kin_to[start:start + 512]]
    # Locks no merge within the spacing keeps on the hair: they stay, until a merge brings another lock near.
    settled = np.zeros(len(planned), dtype=bool)
    # How many locks merged into each.
    version = np.zeros(len(planned), dtype=np.int64)

    def gaps(k):
        """The distances of the locks left from lock ``k``'s root (infinite to itself and the merged ones)."""
        gap = np.linalg.norm(roots - roots[k], axis=1)
        gap[k] = math.inf
        gap[~alive] = math.inf
        return gap

    def find_near(k):
        gap = gaps(k)
        near_to[k] = int(np.argmin(gap))
        near[k] = gap[near_to[k]]
        same = np.where(kinds == kinds[k], gap, math.inf)
        kin_to[k] = int(np.argmin(same))
        kin[k] = same[kin_to[k]]
        return gap

    def try_merge(small, into):
        merged = _merge(planned[into][0], planned[small][0], mesh, points, rig, graph)
        return merged, plan_chain(points, merged, rig)

    def join(small, into, merged, chain):
        """Lock ``small`` merged into ``into`` as ``merged``, planned as ``chain``."""
        nonlocal left, bones
        bones -= len(planned[into][1].joints) + len(planned[small][1].joints)
        planned[into] = (merged, chain)
        version[into] += 1
        alive[small] = False
        left -= 1
        if chain is None:
            alive[into] = False
            left -= 1
        else:
            bones += len(chain.joints)
            areas[into] = merged.area
            roots[into] = merged.root
            kinds[into] = KINDS.index(chain.kind)
        # The nearest locks again: of those whose nearest was one of the two, of the merged lock, and of the
        # others where the merged lock's new root is nearer.
        stale = alive & ((near_to == small) | (near_to == into) | (kin_to == small) | (kin_to == into))
        stale[into] = False
        for k in np.flatnonzero(stale):
            find_near(k)
        if alive[into]:
            gap = find_near(into)
            closer = alive & (gap < near)
            near[closer] = gap[closer]
            near_to[closer] = into
            closer = alive & (kinds == kinds[into]) & (gap < kin)
            kin[closer] = gap[closer]
            kin_to[closer] = into
            # The merged lock is new: it, and the locks of its kind within the spacing, look for partners again.
            settled[into] = False
            settled[alive & (kinds == kinds[into]) & (gap < spacing)] = False

    def merge_twins():
        """Merges the smallest twin into its twin, the larger; False when there are none."""
        twins = alive & (near <= TWIN_REACH)
        if not twins.any():
            return False
        # The smallest twin's nearest is a twin too.
        small = int(np.argmin(np.where(twins, areas, math.inf)))
        into = int(near_to[small])
        join(small, into, *try_merge(small, into))
        return True

    while left > 1:
        if merge_twins():
            continue
        ready = alive & ~settled & (kin < spacing)
        if not ready.any():
            break
        small = int(np.argmin(np.where(ready, areas, math.inf)))
        gap = gaps(small)
        gap[kinds != kinds[small]] = math.inf
        settled[small] = True
        for into in np.argsort(gap, kind='stable')[:int((gap < spacing).sum())]:
            merged, chain = try_merge(small, int(into))
            if _on_hair(points, merged, chain):
                join(small, int(into), merged, chain)
                break
    # Merges over the budget found to leave the hair, as (small, its version, into, its version): a lock's
    # version counts the merges into it, so a pair is tried again only once one of the two changed.
    refused = set()
    while left > 1 and bones > max_bones:
        if merge_twins():
            continue
        shared = alive & (np.bincount(kinds[alive], minlength=len(KINDS))[kinds] >= 2)
        eligible = np.flatnonzero(shared if shared.any() else alive)
        # The smallest lock first; when none of its merges keeps it on the hair, the next smallest.
        order = eligible[np.argsort(areas[eligible], kind='stable')]
        done = False
        for small in order.tolist():
            gap = gaps(small)
            if shared.any():
                gap[kinds != kinds[small]] = math.inf
            for into in np.argsort(gap, kind='stable')[:int(np.isfinite(gap).sum())].tolist():
                key = (small, int(version[small]), into, int(version[into]))
                if key in refused:
                    continue
                merged, chain = try_merge(small, into)
                if _on_hair(points, merged, chain):
                    join(small, into, merged, chain)
                    done = True
                    break
                refused.add(key)
            if done:
                break
        if not done:
            # No lock has a merge that keeps it on the hair: the budget wins, the smallest into its nearest.
            small = int(order[0])
            gap = gaps(small)
            if shared.any():
                gap[kinds != kinds[small]] = math.inf
            into = int(np.argmin(gap))
            join(small, into, *try_merge(small, into))
    planned = [pair for pair, keep in zip(planned, alive) if keep]
    # A merged lock's root moved: back in name order, as find_locks sorts.
    planned.sort(key=lambda pair: (round(pair[0].angle, 6), pair[0].root[0]))
    return [chain for _lock, chain in planned]


# -----------------------------------------------------------------------------
# Armature

def bone_name(chain, bone):
    """The name of bone ``bone`` (0-based, from the root) of chain ``chain`` (0-based): (0, 0) is
    j_ex_hair_01_a."""
    return "{:s}{:02d}_{:s}".format(BONE_PREFIX, chain + 1, chr(ord("a") + bone))


def _parse(name):
    """(chain number, from 1; bone, from 0) of a hair bone's name, or None when it isn't one (BONE_PATTERN)."""
    match = BONE_PATTERN.fullmatch(name)
    return None if match is None else (int(match.group(1)), ord(match.group(2)) - ord("a"))


def _away(point, axis):
    """The horizontal direction (world space) from the head's vertical axis through ``axis`` to ``point``;
    straight behind when the point is on the axis."""
    out = np.asarray(point, dtype=np.float64) - axis
    out[2] = 0.0
    length = float(np.linalg.norm(out))
    return out / length if length > 1e-6 else np.array((0.0, 1.0, 0.0))


def _transform(matrix, point):
    return matrix[:3, :3] @ np.asarray(point, dtype=np.float64) + matrix[:3, 3]


def _read_property(armature_object):
    """The rig RIG_PROPERTY holds (a dict), or None when there is none or it isn't JSON."""
    text = armature_object.get(RIG_PROPERTY) if armature_object is not None else None
    if not isinstance(text, str):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def has_rig(armature_object, race):
    """Whether ``armature_object`` carries a rig (RIG_PROPERTY) for ``race``."""
    data = _read_property(armature_object)
    return data is not None and data.get("race") == race


def write_rig_property(armature_object):
    """Write RIG_PROPERTY on ``armature_object`` from its rig state (`Object.magic_fit_hair_rig`)."""
    state = armature_object.magic_fit_hair_rig
    armature_object[RIG_PROPERTY] = json.dumps({"race": state.race, "chains": [
        {"kind": chain.kind, "bones": [name for name in chain.bones.split(",") if name]}
        for chain in state.chains]})
    armature_object.id_properties_ui(RIG_PROPERTY).update(
        description="The hair's own skeleton: its race and chains. Set by Magic Fit's Hair Weights")


def bones_hash(armature_object):
    """A hash of the rests of the hair's own bones on ``armature_object`` (named BONE_PREFIX, or below one):
    their names, parents, heads, tails and rolls (their Z axes) in armature space, rounded to HASH_PRECISION.
    The rig state keeps it (`HairRigState.bones_hash`) to tell whether the user edited the bones since Hair
    Weights built or read them (`rig_edited`)."""
    rows = []
    for bone in sorted(armature_object.data.bones, key=lambda bone: bone.name):
        if not any(b.name.startswith(BONE_PREFIX) for b in [bone] + list(bone.parent_recursive)):
            continue
        rest = np.concatenate((bone.head_local, bone.tail_local, np.array(bone.matrix_local)[:3, 2]))
        # Adding 0.0 turns -0.0 into 0.0.
        rows.append([bone.name, bone.parent.name if bone.parent is not None else "",
                     (np.round(rest, HASH_PRECISION) + 0.0).tolist()])
    return hashlib.sha1(json.dumps(rows).encode("utf-8")).hexdigest()


def rig_edited(armature_object):
    """Whether the hair's own bones on ``armature_object`` changed since Hair Weights built or read them last
    (`bones_hash`); so it seems when it has no hash of them."""
    stored = armature_object.magic_fit_hair_rig.bones_hash
    return not stored or stored != bones_hash(armature_object)


def keep_kinds(armature_object, plans):
    """``plans`` (`ChainPlan`s, world space) with the kinds of the chains on ``armature_object`` (its rig
    state) whose first bone starts within KIND_REACH of theirs, nearest first, each chain's kind given once:
    built again over the hair, the same locks keep the kinds the user picked. Others keep their own."""
    world = np.array(armature_object.matrix_world, dtype=np.float64)
    bones = armature_object.data.bones
    old = []
    for chain in armature_object.magic_fit_hair_rig.chains:
        first = bones.get(chain.bones.split(",")[0])
        if first is not None and chain.kind in KINDS:
            old.append((_transform(world, first.head_local), chain.kind))
    pairs = sorted((float(np.linalg.norm(plan.joints[0] - root)), p, o)
                   for p, plan in enumerate(plans) for o, (root, _kind) in enumerate(old))
    given, taken = set(), set()
    for distance, p, o in pairs:
        if distance > KIND_REACH:
            break
        if p in given or o in taken:
            continue
        plans[p].kind = old[o][1]
        given.add(p)
        taken.add(o)
    return plans


def counted(count, word):
    """``count`` ``word``s, as UI and report texts count chains and bones: "1 chain", "2 chains"."""
    return "{:d} {:s}{:s}".format(count, word, "" if count == 1 else "s")


def _fill_state(armature_object, race, chains, cap_depth=None):
    """Fill the rig state of ``armature_object`` with ``race``, ``chains`` ((kind, bone names, light) each),
    the hair's ``cap_depth`` (None keeps the one stored) and the hash of its bones (`bones_hash`), write
    RIG_PROPERTY and colour the chains' bones by kind (`hairview.color_bones`)."""
    # hairview imports this module.
    from . import hairview
    state = armature_object.magic_fit_hair_rig
    state.race = race
    state.bones_hash = bones_hash(armature_object)
    if cap_depth is not None:
        state.cap_depth = cap_depth
    state.chains.clear()
    for kind, names, light in chains:
        chain = state.chains.add()
        chain.bones = ",".join(names)
        chain.light = light
        chain.kind = kind
    state.active = min(max(state.active, 0), max(len(state.chains) - 1, 0))
    write_rig_property(armature_object)
    hairview.color_bones(armature_object)
    collect_hair_bones(armature_object)
    hairview.chains_changed(armature_object)


def collect_hair_bones(armature_object):
    """Put every hair bone (BONE_PREFIX) of ``armature_object`` into its CHAIN_COLLECTION bone collection (made
    when missing), its edit bones while the armature is in Edit Mode, and take the others out of it. No other
    collection changes."""
    data = armature_object.data
    collection = data.collections_all.get(CHAIN_COLLECTION)
    if collection is None:
        collection = data.collections.new(CHAIN_COLLECTION)
    bones = data.edit_bones if armature_object.mode == 'EDIT' else data.bones
    for bone in bones:
        inside = bone.collections.get(CHAIN_COLLECTION) is not None
        if bone.name.startswith(BONE_PREFIX) and not inside:
            collection.assign(bone)
        elif inside and not bone.name.startswith(BONE_PREFIX):
            collection.unassign(bone)


def build_armature_rig(context, armature_object, plans, race, cap_depth=-1.0):
    """Replace every hair bone (BONE_PREFIX) of ``armature_object`` with a bone per joint of ``plans``
    (`ChainPlan`s, world space): head on its joint, tail on the next joint (the last one's on the plan's
    tip), its Z axis away from the head's vertical axis, the first hanging from j_kao and each next one from
    the one before. The armature must have j_kao (`hairing.ensure_bones`). Fills the rig state for ``race``,
    with the plans' kinds and light flags and the hair's ``cap_depth`` (`hairing.cap_depth`; -1: unknown),
    and writes RIG_PROPERTY. Returns the new bone names."""
    from mathutils import Vector
    world = np.array(armature_object.matrix_world, dtype=np.float64)
    to_armature = np.array(armature_object.matrix_world.inverted_safe(), dtype=np.float64)
    names, chains = [], []
    with hairing.editing_bones(context, armature_object) as edit_bones:
        head = edit_bones.get(HEAD)
        if head is None:
            raise HairError("Armature '{:s}' has no {:s} bone".format(armature_object.name, HEAD))
        for edit in [edit for edit in edit_bones if edit.name.startswith(BONE_PREFIX)]:
            edit_bones.remove(edit)
        axis = _transform(world, head.head)
        for c, plan in enumerate(plans):
            ends = list(plan.joints[1:]) + [plan.tip]
            parent = head
            chain = []
            for b, (joint, end) in enumerate(zip(plan.joints, ends)):
                edit = edit_bones.new(bone_name(c, b))
                edit.head = Vector(_transform(to_armature, joint).tolist())
                edit.tail = Vector(_transform(to_armature, end).tolist())
                edit.align_roll(Vector((to_armature[:3, :3] @ _away(joint, axis)).tolist()))
                edit.parent = parent
                edit.use_connect = b > 0
                edit.use_deform = True
                parent = edit
                chain.append(edit.name)
            names += chain
            chains.append((plan.kind, chain, plan.light))
    _fill_state(armature_object, race, chains, cap_depth)
    return names


def _hair_bones(bones):
    """Those of ``bones`` (edit bones) that belong to the hair's own skeleton: named BONE_PREFIX, or below
    one that is."""
    return [bone for bone in bones
            if any(b.name.startswith(BONE_PREFIX) for b in [bone] + list(bone.parent_recursive))]


def _walk_chains(bones):
    """The chains through ``bones`` (the hair's edit bones, `_hair_bones`): (names root to tip, fresh) each,
    no bone in two chains. A fresh chain takes a new number and all its bones new names.

    A chain starts at each bone whose parent isn't one of them (in name order, hair bone names first) and
    runs down its children: those named for its own chain number in name order, then those not named for any
    (BONE_PATTERN). Where it goes on down the first, each other one starts a fresh chain (a branch). A child
    named for another chain number starts a chain of its own, which keeps its names (a branch made before)
    unless that number is taken already. Branches come after all the others."""
    names = {bone.name for bone in bones}
    children = {name: [] for name in names}
    for bone in bones:
        if bone.parent is not None and bone.parent.name in names:
            children[bone.parent.name].append(bone)
    tops = sorted((bone for bone in bones if bone.parent is None or bone.parent.name not in names),
                  key=lambda bone: (_parse(bone.name) is None, bone.name))
    pending = [(bone, False) for bone in tops]
    claimed = set()
    chains = []
    index = 0
    while index < len(pending):
        bone, fresh = pending[index]
        index += 1
        parsed = _parse(bone.name)
        fresh = fresh or (parsed is not None and parsed[0] in claimed)
        number = None if fresh or parsed is None else parsed[0]
        if number is not None:
            claimed.add(number)
        chain = [bone.name]
        while True:
            kids = sorted(children[bone.name], key=lambda kid: kid.name)
            numbered = [(kid, _parse(kid.name)) for kid in kids]
            own = [kid for kid, at in numbered if at is not None and number is not None and at[0] == number]
            own += [kid for kid, at in numbered if at is None]
            taken = {kid.name for kid in own}
            pending.extend((kid, False) for kid, _at in numbered if kid.name not in taken)
            if not own:
                break
            pending.extend((kid, True) for kid in own[1:])
            bone = own[0]
            chain.append(bone.name)
        chains.append((chain, fresh))
    return chains


def _known_kinds(armature_object):
    """{chain number: kind} of the chains in the rig state; when it has none (a file from another tool),
    of the chains in RIG_PROPERTY. A chain's number is its first bone's (by its place when that isn't a
    hair bone's name)."""
    state = armature_object.magic_fit_hair_rig
    if len(state.chains):
        listed = [(chain.kind, chain.bones.split(",")) for chain in state.chains]
    else:
        data = _read_property(armature_object)
        chains = data.get("chains") if data is not None else None
        listed = [(chain.get("kind"), chain.get("bones")) for chain in (chains if isinstance(chains, list) else ())
                  if isinstance(chain, dict)]
    kinds = {}
    for index, (kind, bones) in enumerate(listed):
        if kind not in KINDS:
            continue
        first = _parse(bones[0]) if isinstance(bones, list) and bones and isinstance(bones[0], str) else None
        kinds[first[0] if first is not None else index + 1] = kind
    return kinds


def _known_light(armature_object):
    """The chain numbers of the light chains in the rig state (numbered like `_known_kinds`). RIG_PROPERTY
    doesn't list light chains: a rig read from it alone has none."""
    light = set()
    for index, chain in enumerate(armature_object.magic_fit_hair_rig.chains):
        if chain.light:
            first = _parse(chain.bones.split(",")[0])
            light.add(first[0] if first is not None else index + 1)
    return light


def _free_name(used, number):
    """The next free bone name in chain ``number`` (from 1): the letter after its last one taken in
    ``used`` (names), else the first one free."""
    taken = {parsed[1] for parsed in map(_parse, used) if parsed is not None and parsed[0] == number}
    after = max(taken) + 1 if taken else 0
    free = [k for k in range(after, 26)] or [k for k in range(26) if k not in taken]
    if not free:
        raise HairError("Hair chain {:02d} has more than 26 bones".format(number))
    return bone_name(number - 1, free[0])


def sync_rig(context, armature_object, race):
    """Read the chains of ``armature_object``'s hair bones as they are now (the user may have moved, deleted
    or added some in Edit Mode) into its rig state for ``race``, and write RIG_PROPERTY.

    The hair bones are those named BONE_PREFIX and every bone below them; their chains (`_walk_chains`) are
    ordered by their first bone's name. A bone whose name isn't a hair bone's (BONE_PATTERN) gets the next
    free name in its chain (Blender renames the vertex groups of the meshes the armature deforms with it); a
    chain whose first bone has no such name gets the next free number. A new branch (a bone's second child
    on) starts a chain of its own: the next free number, its bones named a, b, ... down it; a branch made
    before keeps its chain. Chains keep their kinds and light flags by number (`_known_kinds`,
    `_known_light`); new ones get `chain_kind` and aren't light. The hair's cap depth stays as stored. Returns
    the renamed bones as "old → new"."""
    kinds = _known_kinds(armature_object)
    light = _known_light(armature_object)
    base = hairing.Reference.get().rig(race, 0)
    renamed = []
    with hairing.editing_bones(context, armature_object) as edit_bones:
        world = np.array(armature_object.matrix_world, dtype=np.float64)
        head = edit_bones.get(HEAD)
        offset = _transform(world, head.head) - base.by_name[HEAD].head if head is not None else np.zeros(3)
        rig = base.moved(offset)
        bones = _hair_bones(edit_bones)
        ends = {edit.name: (_transform(world, edit.head), _transform(world, edit.tail)) for edit in bones}
        walked = _walk_chains(bones)
        used = {edit.name for edit in edit_bones}
        numbers = {parsed[0] for parsed in map(_parse, used) if parsed is not None}
        fresh = set()
        rename = {}
        for names, fresh_chain in walked:
            first = None if fresh_chain else _parse(names[0])
            if first is None:
                number = max(numbers, default=0) + 1
                if number > MAX_CHAINS:
                    raise HairError("The hair has more than {:d} chains".format(MAX_CHAINS))
                numbers.add(number)
                fresh.add(number)
            else:
                number = first[0]
            for name in names:
                # A fresh chain's bones are all named for it.
                if not fresh_chain and _parse(name) is not None:
                    continue
                rename[name] = _free_name(used, number)
                used.add(rename[name])
        chains = [names for names, _fresh in walked]
        for old, new in rename.items():
            edit_bones[old].name = new
            renamed.append("{:s} → {:s}".format(old, new))
    result = []
    for old in chains:
        names = [rename.get(name, name) for name in old]
        number = _parse(names[0])[0]
        # A new chain may take the number of one the user deleted: it doesn't take that one's kind or light.
        kind = kinds.get(number) if number not in fresh else None
        if kind is None:
            length = sum(float(np.linalg.norm(ends[name][1] - ends[name][0])) for name in old)
            kind = chain_kind(_angle(rig, ends[old[0]][0]), length, False)
        result.append((kind, names, number in light and number not in fresh))
    result.sort(key=lambda chain: chain[1][0])
    _fill_state(armature_object, race, result)
    return renamed


def custom_rig(armature_object, race, reference=None):
    """The `hairing.HairRig` of the hair's own skeleton on ``armature_object``: the body's bones of ``race``
    (``reference.rig(race, 0)``) and the armature's hair bones (BONE_PREFIX), the latter in the game's frame
    (ALONG_X: each one's X runs from its head to its tail). Moved by the armature's offset
    (`hairing.armature_offset`), like a vanilla rig, and ``custom``: the body's hair bones take none of the
    hair. It carries the rig state's cap depth (None when unknown) and light chains (their first bones)."""
    reference = hairing.Reference.get() if reference is None else reference
    base = reference.rig(race, 0)
    offset = hairing.armature_offset(armature_object, base)
    world = np.array(armature_object.matrix_world, dtype=np.float64)
    bones = list(base.bones)
    have = {bone.name for bone in bones}
    skeleton = []
    for bone in sorted(armature_object.data.bones, key=lambda bone: bone.name):
        if not bone.name.startswith(BONE_PREFIX) or bone.name in have:
            continue
        rest = world @ np.array(bone.matrix_local, dtype=np.float64) @ ALONG_X
        # Rest bones hold no scale.
        rest[:3, :3] /= np.maximum(np.linalg.norm(rest[:3, :3], axis=0), 1e-12)[None, :]
        rest[:3, 3] -= offset
        bones.append(hairing.Bone(bone.name, bone.parent.name if bone.parent is not None else None, rest))
        skeleton.append(bone.name)
    state = armature_object.magic_fit_hair_rig
    light = {chain.bones.split(",")[0] for chain in state.chains if chain.light}
    return hairing.HairRig(race, 0, bones, base.head_field, skeleton, custom=True,
                           cap_depth=state.cap_depth if state.cap_depth >= 0.0 else None, light=light).moved(offset)
