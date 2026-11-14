# SPDX-License-Identifier: GPL-3.0-or-later

"""Hair rig: a hair skeleton built for the hair itself, chains of bones down the hair.

The game's hair skeletons are made for the game's hairstyles; new hair often hangs where none of them
has a chain. Own Skeleton builds the chains from the mesh.

- What hangs, by the head's scalp map (`hairing.HairRig.scalp`) and the hair's cap depth (`hairing.cap_depth`,
  how far its cap lies off the head): vertices over the face (a scalp share below SCALP_HALF), however close
  to the skin, and vertices over the scalp standing off the cap (more than STAND_MARGIN beyond the cap
  depth) (`_hanging`, `hangs`). The hair lying on the cap stays with the head.
- The chains are planned by `hairlanes` from the shape of the hanging hair, area by area (back, left and right
  side, bangs; the tail with Hair Type Ponytail): each area cut into lanes side by side, as many as its chain count
  (`ChainCounts`), a chain down each (`plan_layout`, `plan_rig`). Detect proposes the counts (`detect_counts`).

The user can also add a chain along hair they select (`plan_selection`, Add Chain): the selected vertices make one
lock (`Lock`), rooted in each loose part where they come closest to the scalp and measured along the whole mesh from
there. Its spine is its middle line down the hair: its root, then the centroid of its vertices in each SPINE_STEP
band of distance from the root, smoothed over SPINE_SMOOTHING samples; a spine shorter than MIN_CHAIN_LENGTH gets no
chain. The chain has a bone per BONE_SPACING of spine (CHAIN_BONES), or as many as the user asks, its joints spread
evenly along the lock's own middle line. Its kind (bangs, side, back or ponytail, `chain_kind`) comes from where the
lock leaves the head and how long it is; only a round lock (PONYTAIL_ROUND) is a ponytail, and where the hair ends
moves it between side and back (SIDE_ANGLE). It is light when most of its roots (by area) are over the scalp: it
stands off the cap (a bun) rather than leaving the scalp at the hairline.

The chains become bones on the hair's armature (`build_armature_rig`), j_ex_hair_01_a and on, hanging from
j_kao. They are ordinary Blender bones, head on a joint and tail on the next, so they draw along the hair
and the user can move, delete or add some in Edit Mode; `sync_rig` reads them back as they are and names
new ones; the rig state keeps a hash of the bones (`bones_hash`) to tell whether the user edited them since,
for the weights the hair's cap depth and which chains are light, and each planned chain's sheet and order (kept
by chain number, like the kinds).
Each bone's Z axis points away from the head's axis, and its bone colour is its chain's kind's
(`hairview.color_bones`). Every hair bone, the user's new ones too, is in the armature's CHAIN_COLLECTION bone
collection, and only they are (`collect_hair_bones`); Show Chains solos it. The armature carries the rig as RIG_PROPERTY, written from
`Object.magic_fit_hair_rig` after every change, for exporters (Instant Edit) to make the hair skeleton; its
version 2 lists each planned chain's sheet (its area) and order (Instant Edit's exporter reads neither). A
hair bone points along its X axis in the game and along its Y axis in Blender: ALONG_X turns one into the
other, here (`custom_rig`) and in the exporter. A chain the user adds along a selection (`add_chain`) takes the
next free number and hangs from j_kao; deleting one from the list (`delete_chain`) hands what hangs from it to its
parent. Both leave the stored hash as it was, so they count as the user's edits and Hair Weights keeps them.

Like `hairing`, nothing here depends on the 3D view.
"""

import hashlib
import heapq
import json
import math
import re
from collections import namedtuple

import numpy as np
from mathutils.kdtree import KDTree

from . import fitting, hairing
from .hairing import HEAD, SCALP_HALF, STAND_MARGIN, HairError, components, unique_triangles, vertex_areas

# Add Chain roots the selection in each of its loose parts that comes within this (metres, beyond the hair's cap
# depth, `hairing.HairRig.scalp_gap`) as near the scalp as the nearest one does (`plan_selection`).
ROOT_REACH = 0.03
# The width of a lock of hair (metres): chains built again keep the kinds of the chains within half of it
# (KIND_REACH).
LOCK_WIDTH = 0.05
# A selection whose roots span no more than this (metres) around the head is kept whole: a ponytail, braid or
# single strand (`plan_selection`, PONYTAIL_LENGTH).
WHOLE_WIDTH = 0.10
# Hair within this (metres) of its closest point to the scalp is rooted there (`_closest_to_scalp`).
ROOT_BAND = 0.01
# Spine samples along the geodesic distance from the root (metres): one per band this wide.
SPINE_STEP = 0.01
# Samples in the centred moving average that smooths the spine (the first and last stay as they are): on a
# coarse mesh one band holds the back of one ring and the front of the next, and the samples zig-zag.
SPINE_SMOOTHING = 5
# A shorter chain (metres) isn't built: a selection's spine (`plan_chain`), a lane's chain (`hairlanes`).
MIN_CHAIN_LENGTH = 0.03
# Points of the arc a chain's joints are spread along (`_arc`).
ARC_SAMPLES = 65
# Bones per chain: clamp(ceil(length / BONE_SPACING), CHAIN_BONES), evenly along the spine (metres).
BONE_SPACING = 0.10
# The fewest and most bones in a chain.
CHAIN_BONES = (2, 5)
# The budget of the whole rig: over it the bones get longer (BONE_GROWTH, `hairlanes.plan`).
MAX_HAIR_BONES = 48
# How wide a chain's share of the hair is (metres) when Detect counts the chains an area gets (`hairlanes.detect`;
# plan_layout's ``spacing``, the Chain Spacing setting).
CHAIN_SPACING = 0.08
# The chains per area the user sets (the Back, Ponytail, Side and Bang Chains settings; plan_layout's ``counts``):
# ``back`` chains side by side across the back hair, ``ponytail`` across the tail Hair Type Ponytail ties, ``left``
# and ``right`` across the side hair on the character's left (+X) and right, ``bangs`` across the bangs (around the
# whole head on short hair, `hairlanes.areas`).
ChainCounts = namedtuple("ChainCounts", "back ponytail left right bangs")
# The fewest and most chains of each area (`ChainCounts`'s fields).
COUNT_RANGES = {"back": (1, 4), "ponytail": (1, 3), "left": (1, 2), "right": (1, 2), "bangs": (1, 6)}
# A rig over its budget (MAX_HAIR_BONES) keeps its chains and lengthens their bones: BONE_SPACING grows by this
# and every chain is planned again, until the rig fits or every chain has CHAIN_BONES[0] bones (user ruling: scale
# the bones up rather than drop back chains).
BONE_GROWTH = 1.25
# The kinds of chain, by where the lock leaves the head and how long it is (see chain_kind).
KINDS = ("bangs", "side", "back", "ponytail")
# Bangs: a root this close to straight in front (degrees around the head's axis, 0 in front)...
BANGS_ANGLE = 45.0
# ...and a spine no longer than this (metres).
BANGS_LENGTH = 0.20
# Side: a root this close to the front (degrees); farther round it is back hair. A tip on the other side of it moves
# hanging hair across (plan_chain): side hair falling back over the shoulder is back hair, and back hair falling
# forward is side hair.
SIDE_ANGLE = 110.0
# Ponytail: a lock kept whole (WHOLE_WIDTH) with a spine at least this long (metres)...
PONYTAIL_LENGTH = 0.30
# ...and round: its hair spreads at least this far (metres, the square root of the second eigenvalue of its
# vertices about the spine) across the spine both ways. A flat card lying on back hair is back hair: Maeve's
# centre cards spread at most 0.0074, a tube ponytail 0.017.
PONYTAIL_ROUND = 0.01
# The custom property on the armature object that holds the rig for exporters, as JSON: {"race": "c0801",
# "version": 2, "chains": [{"kind": "back", "bones": ["j_ex_hair_01_a", ...], "sheet": 0, "order": 1}]}, each
# chain's bones root to tip. Only a planned chain has "sheet" (from 0, shared by one area's chains, `hairlanes.plan`)
# and "order" (its place among that sheet's chains, from 0, across the area; a chain the user deleted leaves a gap); a
# chain the user added has neither. Version 1 had no "version", no sheets and no orders.
RIG_PROPERTY = "xiv_hair_rig"
RIG_VERSION = 2
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
    (bool per vertex: it came from a crumb that joined the lock; `plan_selection` joins none). The chain's spine
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


# -----------------------------------------------------------------------------
# Chains

class ChainPlan:
    """The chain of bones planned for a lock: its ``kind`` (one of KINDS), ``joints`` ((n, 3), world space,
    the bones' heads from the root on, CHAIN_BONES[0] <= n <= CHAIN_BONES[1]), ``tip`` ((3,), world space,
    the lock's end, where the last bone ends), and the lock's ``area`` (m²), ``whole``, ``angle`` (degrees),
    ``vertices`` (indices of the points) and ``light`` (see `Lock`). The joints lie on the spine's arc (`_arc`);
    ``middle`` ((n, 3), the joints when None) is where they would lie on the spine itself, down the middle of
    the lock's hair. A chain of `hairlanes.plan` has the ``sheet`` it is in (from 0, one per area, the sheets
    numbered in the order of their first chain) and its ``order`` among that sheet's chains (from 0, across the
    area); a chain the user selects (`plan_selection`) has -1 for both. ``length`` is the spine's (metres). A chain
    of `hairlanes` counts in ``moved`` how many of its joints and tip its check moved onto its hair, and in
    ``lifted`` how many of those off their row; 0 for both here."""

    __slots__ = ("kind", "joints", "tip", "area", "whole", "angle", "vertices", "light", "middle", "sheet", "order",
                 "length", "moved", "lifted")

    def __init__(self, kind, joints, tip, area, whole, angle, vertices, light=False, middle=None, sheet=-1,
                 order=-1, length=0.0, moved=0, lifted=0):
        self.kind = kind
        self.joints = joints
        self.tip = tip
        self.area = area
        self.whole = whole
        self.angle = angle
        self.vertices = vertices
        self.light = light
        self.middle = joints if middle is None else middle
        self.sheet = sheet
        self.order = order
        self.length = length
        self.moved = moved
        self.lifted = lifted


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


def _arc(line, seg):
    """The arc the chain runs along for a spine ``line`` ((m, 3)) with segments ``seg`` ((m - 1,), not all 0):
    from the spine's first sample to its last, bending one way only, ARC_SAMPLES points. It is
    root + (end - root) s + bend s (1 - s) at s from 0 to 1, ``bend`` fitted by least squares to the spine's
    samples (each at its share of the spine's length): the spine's general line, without its waves."""
    share = np.concatenate(([0.0], np.cumsum(seg))) / float(seg.sum())
    root, end = line[0], line[-1]
    shape = share * (1.0 - share)
    bend = shape @ (line - root - np.outer(share, end - root)) / max(float(shape @ shape), 1e-12)
    s = np.linspace(0.0, 1.0, ARC_SAMPLES)
    curve = root + np.outer(s, end - root) + np.outer(s * (1.0 - s), bend)
    curve[-1] = end
    return curve


def _spread(line, count):
    """``count`` points spread evenly by length along ``line`` ((m, 3), not all on one spot): the first on its
    start, the next ones a ``count``-th of its length further each."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    # Points on one spot would make the interpolation's steps ambiguous.
    line = line[np.concatenate(([True], seg > 0.0))]
    at = np.concatenate(([0.0], np.cumsum(seg[seg > 0.0])))
    marks = at[-1] * np.arange(count) / count
    return np.column_stack([np.interp(marks, at, line[:, k]) for k in range(3)])


def plan_chain(points, lock, rig=None, bone_length=BONE_SPACING):
    """The chain for ``lock`` (`ChainPlan`), or None when its spine is shorter than MIN_CHAIN_LENGTH. It has a
    bone per ``bone_length`` (metres) of spine: its clamp(ceil(length / bone_length), CHAIN_BONES) joints are
    spread evenly by length along the spine's arc
    (`_arc`): the first on the root, the next ones one bone length further each, the last one bone length short
    of the lock's end (the tip). With ``rig`` (`hairing.HairRig`), side hair whose tip lies behind SIDE_ANGLE is back
    hair: it falls back over the shoulder, and as side hair it would keep a chain of its own there. Back hair hanging
    down to a tip in front of SIDE_ANGLE is side hair: it falls forward over the shoulder. Back hair standing up (its
    tip above its root: a bun on the crown, near the head's axis, where a 1 cm lean turns the angle round) stays back
    hair."""
    line = spine(points, lock)
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    length = float(seg.sum())
    if length < MIN_CHAIN_LENGTH:
        return None
    count = int(np.clip(math.ceil(length / bone_length), *CHAIN_BONES))
    # Only a round lock can be a ponytail (PONYTAIL_ROUND). Roundness is measured only where it decides (chain_kind's
    # last test).
    kind = chain_kind(lock.angle, length, lock.whole and lock.angle > SIDE_ANGLE and length >= PONYTAIL_LENGTH
                      and _round(points, lock))
    if kind == "side" and rig is not None and _angle(rig, line[-1]) > SIDE_ANGLE:
        kind = "back"
    elif kind == "back" and rig is not None and line[-1][2] < line[0][2] and _angle(rig, line[-1]) <= SIDE_ANGLE:
        kind = "side"
    return ChainPlan(kind, _spread(_arc(line, seg), count), line[-1].copy(), lock.area, lock.whole,
                     lock.angle, lock.vertices, lock.light, _spread(line, count), length=length)


def plan_selection(points, tris, rig, selected, *, cap=None, bones=0, weld=None):
    """The chain (`ChainPlan`) along the hair the user selected: ``selected`` (indices of ``points``, world
    space, rest pose, with triangles ``tris``) made one lock, or None when nothing is selected or its spine is
    shorter than MIN_CHAIN_LENGTH. In each loose part holding selected vertices that comes within ROOT_REACH as
    near the scalp as the nearest one does, its selected vertices within ROOT_BAND of its closest to the scalp are
    root candidates (hair lying on the cap is all as close), and those of them within ROOT_BAND of the farthest from
    the selection's tip are roots. Distances run along the whole welded mesh from them, so a sparse selection (one
    vertex row) still runs root to tip; the other parts (pieces lower down the lock) are measured on from the nearest
    measured selected hair. Its kind is planned like any lock's (`plan_chain`),
    whole when its roots span no more than WHOLE_WIDTH, light as `Lock` has it. ``bones`` joints (0: by its
    length, as `plan_chain` counts them) lie on the selection's own middle line (`spine`), waves included, not on
    the arc: the user picked the hair it follows (user ruling 2026-11-13). ``cap`` is the hair's cap depth
    (`hairing.cap_depth`, measured from these points when None)."""
    points = np.asarray(points, dtype=np.float64)
    selected = np.unique(np.asarray(selected, dtype=np.int64))
    if not len(selected):
        return None
    mesh = _welded(points, tris, weld)
    if mesh is None:
        return None
    weld, count, rest, t, edges = mesh
    if cap is None:
        cap = hairing.cap_depth(points, tris, rig)
    model = rig.model(rest)
    lengths = np.linalg.norm(model[edges[:, 0]] - model[edges[:, 1]], axis=1)
    area = vertex_areas(model, t)
    weight = area + 1e-12
    point_area = (area / np.maximum(np.bincount(weld, minlength=count), 1))[weld]
    nodes = np.unique(weld[selected])
    gap = np.full(count, np.inf)
    gap[nodes] = rig.scalp_gap(rest[nodes], cap)
    part = components(count, edges)
    groups = [nodes[part[nodes] == p] for p in np.unique(part[nodes])]
    # Only parts that start as near the scalp as the nearest one, give or take ROOT_REACH, are rooted: a lock built
    # of pieces one below the other hangs from its top piece.
    lowest = min(float(gap[group].min()) for group in groups)
    rooted = [group for group in groups if float(gap[group].min()) <= lowest + ROOT_REACH]
    pending = [group for group in groups if float(gap[group].min()) > lowest + ROOT_REACH]
    near = [_closest_to_scalp(group, gap) for group in rooted]
    # Hair lying on the cap is all as near the scalp: of those, the roots are the ones at the selection's end,
    # farthest in a straight line from its tip (the selected vertex farthest along the mesh from them; along the
    # mesh, a card's two top corners lie up to its width apart from it). The parts don't touch, so one Dijkstra
    # measures each from its own.
    distance, _source = _dijkstra(count, edges, lengths, np.concatenate(near))
    roots = []
    for group, candidates in zip(rooted, near):
        tip = group[int(np.argmax(distance[group]))]
        if distance[tip] <= 0.0:
            # All of it is as near the scalp: no end to pick.
            roots.append(candidates)
            continue
        far = np.linalg.norm(rest[candidates] - rest[tip], axis=1)
        roots.append(candidates[far >= far.max() - ROOT_BAND])
    roots = np.concatenate(roots)
    distance, _source = _dijkstra(count, edges, lengths, roots)
    # The other parts are measured on from the measured selected hair nearest them, as if joined there, the
    # nearest part first.
    while pending:
        reached = nodes[np.isfinite(distance[nodes])]
        tree = KDTree(len(reached))
        for k, node in enumerate(reached.tolist()):
            tree.insert(rest[node], k)
        tree.balance()
        apart, k, a, b = min((found[2], k, node, int(reached[found[1]]))
                             for k, group in enumerate(pending) for node in group.tolist()
                             for found in [tree.find(rest[node])])
        edges = np.vstack((edges, [[a, b]]))
        lengths = np.append(lengths, apart)
        pending.pop(k)
        distance, _source = _dijkstra(count, edges, lengths, roots)
    root = np.average(rest[roots], axis=0, weights=weight[roots])
    # The roots' span around the head's vertical axis.
    axis_y = rig.by_name[HEAD].head[1]
    theta = np.arctan2(model[roots, 0], -(model[roots, 1] - axis_y))
    radius = np.hypot(model[roots, 0], model[roots, 1] - axis_y)
    mean = math.atan2(float((weight[roots] * np.sin(theta)).sum()), float((weight[roots] * np.cos(theta)).sum()))
    arc = ((theta - mean + math.pi) % (2.0 * math.pi) - math.pi) * float(np.average(radius, weights=weight[roots]))
    share = rig.scalp(rest[roots])
    light = bool(weight[roots][share >= SCALP_HALF].sum() > 0.5 * weight[roots].sum())
    lock = Lock(selected, distance[weld[selected]], float(point_area[selected].sum()), root,
                float(np.ptp(arc)) <= WHOLE_WIDTH, _angle(rig, root), np.zeros(len(selected), dtype=bool),
                point_area[selected], None, light)
    plan = plan_chain(points, lock, rig)
    if plan is None:
        return None
    plan.joints = plan.middle = _spread(spine(points, lock), bones or len(plan.joints))
    return plan


AREAS = ("back", "ponytail", "left", "right", "bangs")


def plan_rig(points, tris, rig, **options):
    """The chains of `plan_layout` (the same arguments)."""
    return plan_layout(points, tris, rig, **options)[0]


def plan_layout(points, tris, rig, *, cap=None, weld=None, max_bones=MAX_HAIR_BONES, spacing=CHAIN_SPACING,
                ponytail=False, counts=None):
    """(chains, made, short): the chains (`ChainPlan`, world space) of a hair skeleton for hair ``points`` (world
    space, rest pose) with triangles ``tris`` hanging from ``rig`` (`hairing.HairRig`), its cap ``cap`` off the head
    (`hairing.cap_depth`, measured from these points when None), how many each area got ({name in AREAS: count}) and
    whether the hair is short, as `hairlanes.plan` plans them: each area cut into as many lanes side by side as
    ``counts`` (`ChainCounts`) gives it, without counts as many as Detect proposes at ``spacing`` (metres, above 0);
    ``ponytail`` is Hair Type Ponytail, ``max_bones`` the budget."""
    if spacing <= 0.0:
        raise ValueError("The chain spacing must be above 0")
    # hairlanes imports this module.
    from . import hairlanes
    return hairlanes.plan(points, tris, rig, cap=cap, weld=weld, max_bones=max_bones, spacing=spacing,
                          ponytail=ponytail, counts=counts)


def detect_counts(points, tris, rig, *, cap=None, weld=None, spacing=CHAIN_SPACING, ponytail=False):
    """The chain counts Detect proposes (`hairlanes.detect`, a chain per ``spacing`` metres, above 0, of each area's
    width; ``ponytail``: Hair Type Ponytail): ({name in AREAS: count, clamped to COUNT_RANGES}, short). An area
    without a chain to run is left out. Short hair has one count, its chains around the whole head: "bangs"."""
    if spacing <= 0.0:
        raise ValueError("The chain spacing must be above 0")
    # hairlanes imports this module.
    from . import hairlanes
    return hairlanes.detect(points, tris, rig, cap=cap, weld=weld, spacing=spacing, ponytail=ponytail)


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
    chains = []
    for chain in state.chains:
        listed = {"kind": chain.kind, "bones": [name for name in chain.bones.split(",") if name]}
        if chain.sheet >= 0:
            listed.update(sheet=chain.sheet, order=chain.order)
        chains.append(listed)
    armature_object[RIG_PROPERTY] = json.dumps({"race": state.race, "version": RIG_VERSION, "chains": chains})
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
    """Fill the rig state of ``armature_object`` with ``race``, ``chains`` ((kind, bone names, light, sheet,
    order) each), the hair's ``cap_depth`` (None keeps the one stored) and the hash of its bones (`bones_hash`),
    write RIG_PROPERTY and colour the chains' bones by kind (`hairview.color_bones`)."""
    # hairview imports this module.
    from . import hairview
    state = armature_object.magic_fit_hair_rig
    state.race = race
    state.bones_hash = bones_hash(armature_object)
    if cap_depth is not None:
        state.cap_depth = cap_depth
    state.chains.clear()
    for kind, names, light, sheet, order in chains:
        chain = state.chains.add()
        # The kind before the bones: its update callback skips a chain without bones.
        chain.kind = kind
        chain.bones = ",".join(names)
        chain.light = light
        chain.sheet = sheet
        chain.order = order
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
    with the plans' kinds, light flags, sheets and orders and the hair's ``cap_depth`` (`hairing.cap_depth`;
    -1: unknown),
    and writes RIG_PROPERTY. Returns the new bone names."""
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
            chain = _make_chain(edit_bones, plan, c + 1, head, axis, to_armature)
            names += chain
            chains.append((plan.kind, chain, plan.light, int(plan.sheet), int(plan.order)))
    _fill_state(armature_object, race, chains, cap_depth)
    return names


def _make_chain(edit_bones, plan, number, parent, axis, to_armature):
    """Add chain ``number`` (from 1) of ``plan`` (`ChainPlan`, world space) to ``edit_bones``: a bone per joint,
    head on it and tail on the next joint (the last one's on the tip), its Z axis away from the head's vertical
    axis through ``axis`` (world space), the first hanging from ``parent`` (an edit bone) and each next one from
    the one before. ``to_armature`` is the armature's inverted world matrix. Returns the bone names."""
    from mathutils import Vector
    ends = list(plan.joints[1:]) + [plan.tip]
    chain = []
    for b, (joint, end) in enumerate(zip(plan.joints, ends)):
        edit = edit_bones.new(bone_name(number - 1, b))
        edit.head = Vector(_transform(to_armature, joint).tolist())
        edit.tail = Vector(_transform(to_armature, end).tolist())
        edit.align_roll(Vector((to_armature[:3, :3] @ _away(joint, axis)).tolist()))
        edit.parent = parent
        edit.use_connect = b > 0
        edit.use_deform = True
        parent = edit
        chain.append(edit.name)
    return chain


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


def _known_columns(armature_object):
    """{chain number: (sheet, order)} of the planned chains (sheet 0 and on) in the rig state (numbered like
    `_known_kinds`)."""
    columns = {}
    for index, chain in enumerate(armature_object.magic_fit_hair_rig.chains):
        if chain.sheet >= 0:
            first = _parse(chain.bones.split(",")[0])
            columns[first[0] if first is not None else index + 1] = (chain.sheet, chain.order)
    return columns


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
    before keeps its chain. Chains keep their kinds, light flags, sheets and orders by number (`_known_kinds`,
    `_known_light`, `_known_columns`); new ones get `chain_kind`, aren't light and are in no sheet. The hair's
    cap depth stays as stored. Returns the renamed bones as "old → new"."""
    kinds = _known_kinds(armature_object)
    light = _known_light(armature_object)
    columns = _known_columns(armature_object)
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
        # A new chain may take the number of one the user deleted: it doesn't take that one's kind, light or
        # sheet.
        kind = kinds.get(number) if number not in fresh else None
        if kind is None:
            length = sum(float(np.linalg.norm(ends[name][1] - ends[name][0])) for name in old)
            kind = chain_kind(_angle(rig, ends[old[0]][0]), length, False)
        sheet, order = columns.get(number, (-1, -1)) if number not in fresh else (-1, -1)
        result.append((kind, names, number in light and number not in fresh, sheet, order))
    result.sort(key=lambda chain: chain[1][0])
    _fill_state(armature_object, race, result)
    return renamed


def add_chain(context, armature_object, plan, race, cap_depth=None):
    """Add ``plan`` (`ChainPlan`, world space; `plan_selection`) to ``armature_object``'s hair bones as a new
    chain hanging from j_kao, numbered one past the highest chain number there or in the rig state, with the
    plan's kind and light flag and in no sheet; it is picked in the list. Bones the user edited since are read
    back first (`sync_rig`). ``cap_depth`` is stored when the rig state has none. The stored `bones_hash` stays
    as it was, so the chain counts as a user edit (`rig_edited`): Hair Weights reads it back rather than building
    the chains again. Returns the new bone names."""
    from . import hairview
    stored = armature_object.magic_fit_hair_rig.bones_hash
    if any(bone.name.startswith(BONE_PREFIX) for bone in armature_object.data.bones):
        sync_rig(context, armature_object, race)
    state = armature_object.magic_fit_hair_rig
    listed = [name for chain in state.chains for name in chain.bones.split(",")[:1]]
    numbers = {parsed[0] for parsed in map(_parse, list(armature_object.data.bones.keys()) + listed)
               if parsed is not None}
    number = max(numbers, default=0) + 1
    if number > MAX_CHAINS:
        raise HairError("The hair has more than {:d} chains".format(MAX_CHAINS))
    world = np.array(armature_object.matrix_world, dtype=np.float64)
    to_armature = np.array(armature_object.matrix_world.inverted_safe(), dtype=np.float64)
    with hairing.editing_bones(context, armature_object) as edit_bones:
        head = edit_bones.get(HEAD)
        if head is None:
            raise HairError("Armature '{:s}' has no {:s} bone".format(armature_object.name, HEAD))
        names = _make_chain(edit_bones, plan, number, head, _transform(world, head.head), to_armature)
    sync_rig(context, armature_object, race)
    state = armature_object.magic_fit_hair_rig
    index = next(i for i, chain in enumerate(state.chains) if chain.bones.split(",")[0] == names[0])
    chain = state.chains[index]
    chain.kind = plan.kind
    chain.light = plan.light
    chain.sheet = chain.order = -1
    state.active = index
    if cap_depth is not None and state.cap_depth < 0.0:
        state.cap_depth = cap_depth
    write_rig_property(armature_object)
    hairview.color_bones(armature_object)
    state.bones_hash = stored
    return names


def delete_chain(context, armature_object, index, race):
    """Remove chain ``index`` (0-based, in the rig state's list) from ``armature_object``: bones the user edited
    since are read back first (`sync_rig`; a bone extruded from the chain is in it then), the chain's bones go, and
    a bone hanging from one of them hangs from the chain's first bone's parent instead (its chain stays a chain).
    The rest is read back again, and the stored `bones_hash` stays as it was, so the deletion counts as a
    user edit (`rig_edited`), like `add_chain`. Returns the chain's number ("02")."""
    state = armature_object.magic_fit_hair_rig
    if not 0 <= index < len(state.chains):
        raise HairError("No chain {:d} in the list".format(index + 1))
    stored = state.bones_hash
    # Bones the user edited since are read back first: one extruded from the chain goes with it.
    first_name = state.chains[index].bones.split(",")[0]
    sync_rig(context, armature_object, race)
    state = armature_object.magic_fit_hair_rig
    if not len(state.chains):
        state.bones_hash = stored
        raise HairError("No chain {:d} in the list".format(index + 1))
    index = next((i for i, chain in enumerate(state.chains) if chain.bones.split(",")[0] == first_name),
                 min(index, len(state.chains) - 1))
    names = [name for name in state.chains[index].bones.split(",") if name]
    first = _parse(names[0]) if names else None
    number = "{:02d}".format(first[0] if first is not None else index + 1)
    with hairing.editing_bones(context, armature_object) as edit_bones:
        gone = [edit_bones[name] for name in names if edit_bones.get(name) is not None]
        if gone:
            parent = gone[0].parent
            doomed = {edit.name for edit in gone}
            for edit in edit_bones:
                if edit.name not in doomed and edit.parent is not None and edit.parent.name in doomed:
                    edit.use_connect = False
                    edit.parent = parent
            for edit in gone:
                edit_bones.remove(edit)
    sync_rig(context, armature_object, race)
    state = armature_object.magic_fit_hair_rig
    state.active = max(min(index, len(state.chains) - 1), 0)
    state.bones_hash = stored
    return number


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
