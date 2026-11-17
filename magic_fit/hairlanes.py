# SPDX-License-Identifier: GPL-3.0-or-later

"""Hair lanes: Own Skeleton's chains planned from the shape of the hair, area by area (`hairrig` builds them).

The hanging hair is measured first (`measure`), one entry per hanging vertex, in the rig's model space:

- What hangs is `hairrig._hanging`, by the head's scalp map and the hair's cap depth (`hairing.cap_depth`).
- Where it hangs around the head: its angle around the head's vertical axis (through j_kao: 0 straight in front,
  positive toward +X, the character's left) and its distance from the axis.
- How far down the hair it is: its drop below the top hanging vertex, and its distance from the scalp beyond the
  cap depth (`hairing.HairRig.scalp_gap`). The scalp's lower edge in its direction (`hairing.HeadField.scalp_floor`)
  and its scalp share go with it.

Then every hanging vertex gets one area (`areas`), by where it hangs:

- Beyond SIDE_ANGLE (`hairrig`) from the front it is back hair; any other is side hair, left (+X) or right by the
  sign of its X.
- A loose part (a connected piece of the welded mesh: a card, a strand, a braid) goes whole to one area when that
  area holds at least WHOLE_SHARE of its hanging hair, each vertex counted by its area times DEPTH_BIAS + its depth
  (0 at the part's top hanging vertex, 1 at its lowest). So a card rooted behind the ear that falls forward over the
  shoulder is side hair, and one that falls back is back hair. A part spread over several areas (a one-piece mesh)
  is split vertex by vertex, unless it hangs across the middle in front, no wider than MIDDLE_WIDTH (a lock centered
  on the face): that one goes whole to the side of its mean X.
- Bangs: a whole part that isn't back hair, whose hanging hair lies within FRONT_ANGLE of straight in front on
  average (by area) and ends no lower than the neck pivot (j_kubi) + BANGS_ABOVE. In a split part, its vertices
  within FRONT_ANGLE of the front at that height or higher.
- With Hair Type Ponytail the back area is the tail: the back hair, and every whole part rooted behind the ears,
  wherever it then hangs (of its top ROOT_SHARE by height, most lies beyond ROOT_BEHIND from the front). A split part
  stays split; its back vertices are the tail.
- The hair is short when under SHORT_SHARE of its hanging hair (by area) lies more than SHORT_BELOW below the neck
  pivot + BANGS_ABOVE.

Each area is then cut into lanes side by side, a chain down each (`plan_area`, a `Lane` each):

- Levels: hair that hangs is measured by its drop, so one level is one height whatever layer a vertex is in; an area
  that doesn't hang (it spreads more than HANG_RATIO times as far from the scalp as in height: a bun, a tail standing
  out) by its distance from the scalp. Its chains run between the first and last level whose band (LEVEL_STEP apart,
  LEVEL_BAND either side) holds THIN of its median band, and on hanging hair start no higher than the scalp's lower
  edge when BELOW_EDGE of the area hangs below it.
- Lanes: a vertex's place across the area is an arc around the head (`across_back`, `across_side`, `across_front`).
  At every level the lane edges are the area-weighted k / count quantiles of it, smoothed down the levels
  (EDGE_SMOOTHING); below the widest level they go straight down once a lane would narrow past NARROW. A vertex on
  an edge (ON_EDGE) counts half in the chains of both lanes beside it.
- Rows: the area's middle line (MIDDLE_SMOOTHING) gets clamp(ceil(length / bone length), `hairrig.CHAIN_BONES`)
  bones, and its rows cut it into equal lengths. Every chain has its joints on those rows: the middle of its lane's
  hair there within FOLLOW across of the joint above (on straight from the two above over a hole in that hair, or,
  with one joint above, the middle of that hair nearest the row); the first joint is the middle of the clump
  nearest the lane's middle when none of its hair lies within FOLLOW of it.
  When none of the lane's hair within FOLLOW across of the joint above reaches the next row, but the hair that
  joint was made of runs on into the row's band, joined by the lane's hair between the two rows (`_reached`: through
  the mesh, or touching, closer than the lane's median edge and LEVEL_STEP), it swept sideways: the chain goes on with
  that piece's hair within FOLLOW across of where it comes nearest the joint above. From then on each joint's hair is
  the part of it joined to the hair the chain followed (over a hole, the followed hair nearest the row below it), and
  the hair under a joint is the hair joined to that below it (a second strand beside it is never joined).
  A chain ends with the hair it follows (its lane's hair within FOLLOW across of the joint above), keeping the rows
  at least `hairrig.MIN_CHAIN_LENGTH` above that, so a lane that runs out early, or a clump that ends early, has
  fewer bones on the same rows. At its fewest bones (`hairrig.CHAIN_BONES`), when none of that hair lies under its
  last joint and the tip going on from there would lie beyond CHECK_REACH of the lane's hair, the hair it follows is
  too short for two rows: the second joint goes halfway down it, the tip to its end. Its joints are eased sideways
  toward their neighbors.
- The check: a joint or tip farther than CHECK_REACH from its lane's hair moves onto the hair it follows: on its
  row (ROW_REACH) within FOLLOW + ROW_REACH across of it, else at any level within FOLLOW (lifted); the whole lane's
  hair only when neither holds any.
- Bone length (`_spaced`): after the check no bone is shorter than MIN_BONE. A joint closer than that to the joint
  above it, or to the tip, goes (two joints the check put on one vertex, two rows of hair measured from the scalp
  on one spot); a chain left with one joint gets its second halfway from the root to the tip, and one whose root
  and tip then lie under `hairrig.MIN_CHAIN_LENGTH` apart is not built.
- Detect (`lane_count`): a lane per Chain Spacing of the area's median width (between the SPREAD quantiles of each
  level), 1 for hair wider than long.

The whole rig (`plan`; `detect` proposes its counts):

- Areas in order: bangs (`across_front`), left and right (`across_side`), the back or with Hair Type Ponytail the
  tail (`across_back`), each cut into as many lanes as its count (Detect's without counts); the chains' kind is their
  area's (bangs, side, back, ponytail). An area with no chain to run builds none and Detect leaves it out.
- Short hair is cut into Bang Chains equal sectors around the whole head, the first centered straight in front, a
  bangs chain each with the distance from the scalp as its levels; Hair Type Ponytail ties nothing on it. Detect
  gives it the arc its hanging hair covers (RING_BIN) per Chain Spacing.
- Over the budget (`hairrig.MAX_HAIR_BONES`) the bones get longer by `hairrig.BONE_GROWTH` and every area is planned
  again, until the rig fits or every chain has two bones; nothing merges.
- The chains are in name order, by their first joint's angle around the head, then its X. Each area is one sheet,
  numbered by its first chain; a chain's order is its place across its area.

Like `hairing`, nothing here depends on the 3D view.
"""

import math

import numpy as np

from . import hairing, hairrig

# The areas of the hanging hair (`areas`): back (the tail with Hair Type Ponytail), the character's left (+X) and
# right side, and bangs.
BACK, LEFT, RIGHT, BANGS = 0, 1, 2, 3
# Bangs lie within this (degrees around the head's axis) of straight in front...
FRONT_ANGLE = 50.0
# ...no lower than this (metres) above the neck pivot (j_kubi).
BANGS_ABOVE = 0.02
# A loose part goes whole to the area holding at least this share of its hanging hair, each vertex counted by its
# area times DEPTH_BIAS + its depth down the part (0 at its top, 1 at its bottom): where a part ends decides more
# than where it starts.
WHOLE_SHARE = 0.7
DEPTH_BIAS = 0.2
# A loose part hanging across the middle in front (at least WHOLE_SHARE of it, counted so, outside the back, but
# neither side holding WHOLE_SHARE: a lock centered on the face) goes whole to the side of its mean X when its hanging
# hair is no wider in X than this (metres, between the SPREAD quantiles by area). A wider one stays split: left and
# right never share a part.
MIDDLE_WIDTH = 0.08
# With Hair Type Ponytail, a whole part is rooted behind the ears, and tied into the tail, when most (by area) of
# its top ROOT_SHARE (of its height) lies beyond ROOT_BEHIND (degrees) from the front.
ROOT_BEHIND = 90.0
ROOT_SHARE = 0.2
# Short hair: less than SHORT_SHARE of the hanging hair (by area) lies more than SHORT_BELOW (metres) below the neck
# pivot + BANGS_ABOVE.
SHORT_BELOW = 0.05
SHORT_SHARE = 0.05
# An area's levels (`plan_area`) are looked at every LEVEL_STEP (metres), each holding the hair within LEVEL_BAND of
# it (half a 4 cm band).
LEVEL_STEP = 0.01
LEVEL_BAND = 0.02
# An area's hair runs from the first to the last level whose band holds at least THIN (by area) of its median band
# with hair: a thin strand above or below doesn't stretch the chains.
THIN = 0.15
# An area doesn't hang (a bun, a tail standing out) when its hair spreads more than HANG_RATIO times as far in its
# distance from the scalp as in height, both between the SPREAD quantiles of its area: its levels are then its
# distance from the scalp, else its drop. Detect (`lane_count`) measures an area's width between them too.
HANG_RATIO = 1.5
SPREAD = (0.02, 0.98)
# Hanging hair starts its chains no higher than the scalp's lower edge in the area's direction (the area-weighted
# median of `Hair.edge`) when at least BELOW_EDGE of it (by area) hangs below that edge: hair standing off the cap
# higher up stays with the head.
BELOW_EDGE = 0.6
# Across an area, a vertex's place is an arc around the head: its angle times its distance from the head's axis, no
# less than MIN_RADIUS (metres).
MIN_RADIUS = 0.08
# The lane edges are smoothed down the levels over EDGE_SMOOTHING levels, the area's middle line over
# MIDDLE_SMOOTHING.
EDGE_SMOOTHING = 7
MIDDLE_SMOOTHING = 5
# Below the area's widest level, once a lane would get narrower than NARROW of its width there, the lane edges go
# straight down: chains fan out with hair that widens and run parallel where it narrows.
NARROW = 0.7
# A vertex within ON_EDGE (metres across) of the edge between two lanes counts half in the chains of both: the middle
# column of mirrored hair lies on the middle edge, give or take how its triangles weigh its sides.
ON_EDGE = 0.001
# A joint is the middle of its lane's hair at its row within FOLLOW (metres across) of the joint above: a lane whose
# hair parts into clumps doesn't put a joint between them, and its chain ends with the clump it follows.
FOLLOW = 0.04
# An area with less hanging hair than this (m², `hairing.vertex_areas`) builds nothing.
MIN_AREA = 0.0015
# The check: a joint or tip farther than CHECK_REACH (metres) from its lane's hair moves onto the nearest of its
# lane's vertices within ROW_REACH of its level and FOLLOW + ROW_REACH across of it, or when that row holds none, the
# nearest within FOLLOW across at any level (lifted); onto the whole lane's hair the same way when neither has any.
CHECK_REACH = 0.03
ROW_REACH = 0.02
# No bone is shorter than this (metres, a level step): a joint closer than it to the joint above, or to the tip,
# goes (`_spaced`). The check can put two joints on one vertex, and on hair measured from the scalp two rows can lie
# on one spot.
MIN_BONE = LEVEL_STEP
# Detect on short hair: its Bang Chains are the arc it covers around the head per Chain Spacing, the arc being the
# RING_BIN (degrees) directions around the head's axis that hold hanging hair times its median distance from the axis.
RING_BIN = 15.0


class Hair:
    """The hanging hair of a mesh (`measure`), one entry per hanging vertex: ``index`` (into the mesh's points),
    ``at`` ((m, 3), the rig's model space), ``area`` (m², `hairing.vertex_areas`), ``phi`` (radians around the head's
    vertical axis through j_kao: 0 straight in front, positive toward +X, +-pi behind), ``far`` (metres from that
    axis), ``part`` (its loose part, a component of the welded mesh), ``drop`` (metres below the top hanging vertex),
    ``gap`` (metres from the scalp beyond ``cap``, `hairing.HairRig.scalp_gap`), ``edge`` (the height of the scalp's
    lower edge in its direction, `hairing.HeadField.scalp_floor`; -inf where none) and ``share`` (its scalp share,
    `hairing.HairRig.scalp`), ``group`` (its welded vertex). For the whole hair: ``floor`` (the neck pivot's height +
    BANGS_ABOVE), ``cap`` (the hair's cap depth) and ``edges`` ((m, 2), the mesh's edges between welded vertices)."""

    __slots__ = ("index", "at", "area", "phi", "far", "part", "drop", "gap", "edge", "share", "floor", "cap", "group",
                 "edges")

    def __init__(self, index, at, area, phi, far, part, drop, gap, edge, share, floor, cap, group=None, edges=None):
        self.index = index
        self.at = at
        self.area = area
        self.phi = phi
        self.far = far
        self.part = part
        self.drop = drop
        self.gap = gap
        self.edge = edge
        self.share = share
        self.floor = floor
        self.cap = cap
        self.group = group
        self.edges = edges


def measure(points, tris, rig, cap=None, weld=None):
    """The hanging hair (`Hair`) of hair ``points`` (world space, rest pose, (n, 3)) with triangles ``tris`` hanging
    from ``rig`` (`hairing.HairRig`); None when nothing hangs or there is no mesh. ``cap`` is how far the hair's cap
    lies off the head (`hairing.cap_depth`, measured from these points when None). ``weld`` (vertex -> welded group)
    joins vertices sharing a position into one loose part; by default they are found here."""
    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    mesh = hairrig._welded(points, tris, weld)
    if mesh is None:
        return None
    weld, count, _rest, _t, edges = mesh
    share = rig.scalp(points)
    if cap is None:
        cap = hairing.cap_depth(points, tris, rig, share=share)
    used = np.zeros(len(points), dtype=bool)
    used[tris.ravel()] = True
    index = np.flatnonzero(used & hairrig._hanging(share, rig.head_distance(points), cap))
    if not len(index):
        return None
    at = rig.model(points[index])
    axis_y = rig.by_name[hairing.HEAD].head[1]
    field = rig.head_field
    return Hair(index=index, at=at, area=hairing.vertex_areas(points, tris)[index],
                phi=np.arctan2(at[:, 0], -(at[:, 1] - axis_y)), far=np.hypot(at[:, 0], at[:, 1] - axis_y),
                part=hairing.components(count, edges)[weld[index]], drop=at[:, 2].max() - at[:, 2],
                gap=rig.scalp_gap(points[index], cap), edge=field.scalp_floor[field._bins(at)], share=share[index],
                floor=float(rig.by_name[hairing.NECK].head[2]) + BANGS_ABOVE, cap=cap, group=weld[index], edges=edges)


def areas(hair, ponytail=False):
    """(label, short): the area (BACK, LEFT, RIGHT or BANGS) of each of ``hair``'s vertices (`measure`), and whether
    the hair is short. With ``ponytail`` (Hair Type Ponytail) BACK is the tail."""
    z = hair.at[:, 2]
    weight = hair.area + 1e-12
    turn = np.abs(hair.phi)
    zone = np.where(turn > math.radians(hairrig.SIDE_ANGLE), BACK, np.where(hair.at[:, 0] >= 0.0, LEFT, RIGHT))
    front = turn <= math.radians(FRONT_ANGLE)
    label = zone.copy()
    order = np.argsort(hair.part, kind="stable")
    starts = np.flatnonzero(np.diff(hair.part[order], prepend=-1))
    for mine in np.split(order, starts[1:]):
        top, bottom = float(z[mine].max()), float(z[mine].min())
        depth = (top - z[mine]) / max(top - bottom, 1e-6)
        heavy = weight[mine] * (DEPTH_BIAS + depth)
        shares = np.bincount(zone[mine], heavy, minlength=3) / heavy.sum()
        best = int(np.argmax(shares))
        if shares[best] < WHOLE_SHARE:
            x = hair.at[mine, 0]
            if 1.0 - shares[BACK] < WHOLE_SHARE or np.ptp(_quantiles(x, weight[mine], SPREAD)) > MIDDLE_WIDTH:
                # Split vertex by vertex: its front above the floor is bangs.
                label[mine[front[mine] & (z[mine] >= hair.floor)]] = BANGS
                continue
            # Narrow, across the middle in front: one side's, by its mean X.
            best = LEFT if np.average(x, weights=weight[mine]) >= 0.0 else RIGHT
        label[mine] = best
        if ponytail:
            # Rooted behind the ears: most of its top lies behind.
            root = depth <= ROOT_SHARE
            behind = turn[mine] > math.radians(ROOT_BEHIND)
            if weight[mine][root & behind].sum() > 0.5 * weight[mine][root].sum():
                label[mine] = BACK
                continue
        ahead = np.average(turn[mine], weights=weight[mine]) <= math.radians(FRONT_ANGLE)
        if best != BACK and ahead and bottom >= hair.floor:
            label[mine] = BANGS
    short = float(weight[z < hair.floor - SHORT_BELOW].sum()) < SHORT_SHARE * float(weight.sum())
    return label, short


class Lane:
    """A chain of one area (`plan_area`): ``joints`` ((n, 3), the rig's model space) and ``tip`` ((3,)), ``members``
    (indices into `Hair`'s arrays of the lane's vertices), ``number`` (its place across the area, 0 first), ``light``
    (most of its hair at its first row lies over the scalp, and it doesn't start at the scalp's lower edge: a light
    chain, as `hairrig.ChainPlan`'s), ``moved`` (how many of its joints and tip the check moved onto its hair) and
    ``lifted`` (how many of those it moved off their row)."""

    __slots__ = ("joints", "tip", "members", "number", "light", "moved", "lifted")

    def __init__(self, joints, tip, members, number, light, moved, lifted):
        self.joints = joints
        self.tip = tip
        self.members = members
        self.number = number
        self.light = light
        self.moved = moved
        self.lifted = lifted


def _quantiles(values, weights, shares):
    """The weighted quantiles ``shares`` of ``values``, each value counted at the middle of its weight."""
    order = np.argsort(values, kind="stable")
    weights = weights[order]
    return np.interp(shares, (np.cumsum(weights) - 0.5 * weights) / weights.sum(), values[order])


def across_back(hair):
    """Each of ``hair``'s vertices' place across the back (metres): its angle from straight behind times its
    distance from the head's axis (MIN_RADIUS at least), +X lowest."""
    return (hair.phi % (2.0 * math.pi) - math.pi) * np.maximum(hair.far, MIN_RADIUS)


def across_side(hair):
    """Each of ``hair``'s vertices' place across a side (metres): its angle from the front times its distance from
    the head's axis (MIN_RADIUS at least), the front lowest."""
    return np.abs(hair.phi) * np.maximum(hair.far, MIN_RADIUS)


def across_front(hair):
    """Each of ``hair``'s vertices' place across the front (metres): its angle from the front times its distance from
    the head's axis (MIN_RADIUS at least), -X lowest."""
    return hair.phi * np.maximum(hair.far, MIN_RADIUS)


def _bands(level, weight, grid):
    """The hair within LEVEL_BAND of each level of ``grid``: (its weight, the vertices by level, and each band's
    first and end place in them)."""
    order = np.argsort(level, kind="stable")
    ordered = level[order]
    total = np.concatenate(([0.0], np.cumsum(weight[order])))
    low = np.searchsorted(ordered, grid - LEVEL_BAND)
    high = np.searchsorted(ordered, grid + LEVEL_BAND, side="right")
    return total[high] - total[low], order, low, high


def _span(level, weight):
    """The first and last level (LEVEL_STEP apart, within the hair's own) whose band holds at least THIN of the
    median band with hair."""
    grid = np.arange(float(level.min()), float(level.max()) + 0.5 * LEVEL_STEP, LEVEL_STEP)
    bands = _bands(level, weight, grid)[0]
    on = np.flatnonzero(bands >= THIN * float(np.median(bands[bands > 0.0])))
    return float(grid[on[0]]), float(grid[on[-1]])


def _levels(hair, members, weight, from_scalp=False):
    """(level, hangs, reach, edged) of an area's vertices ``members``: each one's level, whether the area hangs (never
    ``from_scalp``), the first and last level its chains run between (None when they lie under MIN_CHAIN_LENGTH
    apart), and whether they start at the scalp's lower edge (BELOW_EDGE)."""
    drop, gap = hair.drop[members], hair.gap[members]
    hangs = not from_scalp and (np.ptp(_quantiles(gap, weight, SPREAD))
                                <= HANG_RATIO * np.ptp(_quantiles(drop, weight, SPREAD)))
    level = drop if hangs else gap
    first, last = _span(level, weight)
    if last - first < hairrig.MIN_CHAIN_LENGTH:
        return level, hangs, None, False
    edged = False
    if hangs:
        edge = hair.edge[members]
        known = np.isfinite(edge)
        if known.any():
            edge = float(_quantiles(edge[known], weight[known], 0.5))
            if weight[hair.at[members, 2] < edge].sum() >= BELOW_EDGE * weight.sum():
                # The level of the edge's height: a drop is measured from the top hanging vertex.
                top = float(hair.at[:, 2].max())
                first = max(first, min(top - edge, last - hairrig.MIN_CHAIN_LENGTH))
                edged = True
    return level, hangs, (first, last), edged


def _fill(values):
    """``values`` ((n, k)) with the rows holding NaN taken from their neighbors, column by column."""
    index = np.arange(len(values))
    for k in range(values.shape[1]):
        known = np.isfinite(values[:, k])
        values[:, k] = np.interp(index, index[known], values[known, k])
    return values


def _smooth(values, window):
    """The moving average of ``values`` ((n, k)) over ``window`` rows, the first and last row repeated past the
    ends."""
    half = window // 2
    padded = np.concatenate((np.repeat(values[:1], half, axis=0), values, np.repeat(values[-1:], half, axis=0)))
    kernel = np.full(window, 1.0 / window)
    return np.column_stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(values.shape[1])])


def _middle(xyz, weight, on, level, top):
    """The middle (by area) of the hair ``on`` at ``level``; on hanging hair (``top``, the height of level 0) at the
    level's height."""
    point = np.average(xyz[on], axis=0, weights=weight[on])
    if top is not None:
        point[2] = top - level
    return point


def _touching(points, reach):
    """The pairs (i < j) of ``points`` closer together than ``reach``, found cell by cell."""
    cell = np.floor(points / reach).astype(np.int64)
    cell -= cell.min(axis=0) - 1
    size = cell.max(axis=0) + 2
    key = (cell[:, 0] * size[1] + cell[:, 1]) * size[2] + cell[:, 2]
    order = np.argsort(key, kind="stable")
    keys, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
    pairs = []
    for offset in np.array(np.meshgrid((-1, 0, 1), (-1, 0, 1), (-1, 0, 1))).T.reshape(-1, 3):
        other = key + (offset[0] * size[1] + offset[1]) * size[2] + offset[2]
        found = np.minimum(np.searchsorted(keys, other), len(keys) - 1)
        count = np.where(keys[found] == other, counts[found], 0)
        first = np.repeat(np.arange(len(points)), count)
        within = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
        second = order[np.repeat(starts[found], count) + within]
        near = (first < second) & (np.linalg.norm(points[first] - points[second], axis=1) < reach)
        pairs.append(np.column_stack((first[near], second[near])))
    return np.vstack(pairs)


def _pieces(s, low, high, links):
    """The pieces of a lane's hair between levels ``low`` and ``high``: (whether each vertex lies between them, the
    piece of each vertex). ``links`` is the lane's mesh as `_links` gives it. Hair is one piece where the mesh joins
    it, or where it touches: closer than the lane's median edge, LEVEL_STEP at most (cards laid over each other, a
    braid's strands)."""
    group, edges, points, reach = links
    inside = (s >= low) & (s <= high)
    held = np.zeros(int(group.max()) + 1, dtype=bool)
    held[group[inside]] = True
    joined = [edges[held[edges[:, 0]] & held[edges[:, 1]]]]
    which = np.flatnonzero(inside)
    if reach > 0.0 and len(which) > 1:
        joined.append(group[which[_touching(points[which], reach)]])
    piece = hairing.components(len(held), np.vstack(joined))
    return inside, piece[group]


def _reached(seed, s, low, high, links):
    """The lane's hair between levels ``low`` and ``high`` joined through the mesh, by hair between them, to the
    vertices ``seed`` (the hair a joint was made of)."""
    inside, piece = _pieces(s, low, high, links)
    return inside & np.isin(piece, piece[seed & inside])


def _onto(point, level, place, xyz, levels, across):
    """The check (CHECK_REACH, ROW_REACH) of a joint or tip ``point`` at ``level`` and ``place`` across against its
    lane's hair ``xyz`` with ``levels`` and places ``across``: (where it goes, 0 when it stays, 1 when it moved on
    its row, 2 when it was lifted off it). It keeps to the hair it follows: on its row within FOLLOW + ROW_REACH
    across of it, else at any level within FOLLOW, else the whole lane's hair as the same two steps."""
    gap = np.linalg.norm(xyz - point, axis=1)
    if gap.min() <= CHECK_REACH:
        return point, 0
    off = np.abs(across - place)
    on_row = np.abs(levels - level) <= ROW_REACH
    # Keeping a joint on its row comes first, so the row step reaches farther across: the hair beside a hole the
    # chain runs through lies just beyond FOLLOW of its middle.
    row, near = on_row & (off <= FOLLOW + ROW_REACH), off <= FOLLOW
    if not row.any() and not near.any():
        row, near = on_row, np.ones(len(xyz), dtype=bool)
    if row.any():
        row = np.flatnonzero(row)
        return xyz[row[np.argmin(gap[row])]].copy(), 1
    near = np.flatnonzero(near)
    return xyz[near[np.argmin(gap[near])]].copy(), 2


def _spaced(line, state):
    """A chain's ``line`` (joints then tip) and each point's ``state`` (`_onto`) with no bone shorter than MIN_BONE:
    from the root down, a joint closer than that to the last one kept goes, then the last ones kept that close to
    the tip; the root and the tip stay. A chain left with under `hairrig.CHAIN_BONES`[0] joints gets its fewest
    bones, equal, straight from the root to the tip. (None, None) when those two are under
    `hairrig.MIN_CHAIN_LENGTH` apart."""
    keep = [0]
    for i in range(1, len(line) - 1):
        if np.linalg.norm(line[i] - line[keep[-1]]) >= MIN_BONE:
            keep.append(i)
    while len(keep) > 1 and np.linalg.norm(line[-1] - line[keep[-1]]) < MIN_BONE:
        del keep[-1]
    fewest = hairrig.CHAIN_BONES[0]
    if len(keep) >= fewest:
        keep.append(len(line) - 1)
        return line[keep], state[keep]
    if np.linalg.norm(line[-1] - line[0]) < hairrig.MIN_CHAIN_LENGTH:
        return None, None
    return (np.linspace(line[0], line[-1], fewest + 1),
            np.concatenate((state[:1], np.zeros(fewest - 1, dtype=state.dtype), state[-1:])))


def _chain(xyz, weight, u, s, rows, top, links=None):
    """The chain of one lane: its hair ``xyz`` with areas ``weight``, places across ``u`` and levels ``s``, on the
    area's ``rows`` (the last is the area's end); ``top`` as `_middle`; ``links`` the lane's mesh as `_pieces`
    takes it (None: no chain follows hair that sweeps sideways). (joints, tip, first level, moved, lifted), or None
    when it is under MIN_CHAIN_LENGTH long."""
    least = hairrig.MIN_CHAIN_LENGTH
    end = min(_span(s, weight)[1], rows[-1])
    if end - rows[0] < least:
        return None
    levels = [row for row in rows[:-1] if row <= end - least]
    if len(levels) < hairrig.CHAIN_BONES[0]:
        levels = list(np.linspace(rows[0], end, hairrig.CHAIN_BONES[0] + 1)[:-1])
    joints, follows, seeds = [], [], []
    follow = None
    # The first joint placed on hair that swept sideways (None: none is).
    sweep_at = None
    i = 0
    while i < len(levels):
        row = levels[i]
        on = np.abs(s - row) <= LEVEL_BAND
        seed = None
        if follow is not None:
            near = np.abs(u - follow) <= FOLLOW
            if links is not None and not (on & near).any() and not (near & (s > row)).any():
                # No hair within FOLLOW across of the joint above lies at or below this row. When the hair that
                # joint was made of runs on, joined through the mesh by hair between the two rows, into this row's
                # band, it swept sideways: the chain goes on with the piece of it there nearest the joint above.
                reached = _reached(seeds[-1], s, levels[i - 1] - LEVEL_BAND, row + LEVEL_BAND, links) & on
                if reached.any():
                    # Of that piece, the hair within FOLLOW across of where it comes nearest the joint above.
                    piece = _pieces(s, row - LEVEL_BAND, row + LEVEL_BAND, links)[1]
                    nearest = np.flatnonzero(reached)[np.argmin(np.abs(u[reached] - follow))]
                    reached &= (piece == piece[nearest]) & (np.abs(u - u[nearest]) <= FOLLOW)
                    follow = float(np.average(u[reached], weights=weight[reached]))
                    near = np.abs(u - follow) <= FOLLOW
                    sweep_at = i if sweep_at is None else sweep_at
                    seed = reached
            if not (on & near).any() and not (near & (s > row)).any():
                # The hair it follows ends above this row: so does the chain. With too little of it for two rows,
                # its second joint goes halfway down that hair.
                if i >= hairrig.CHAIN_BONES[0]:
                    del levels[i:]
                    break
                under = near & (s > levels[i - 1])
                row = levels[i] = 0.5 * (levels[i - 1] + (_span(s[under], weight[under])[1] if under.any()
                                                          else levels[i - 1]))
                del levels[i + 1:]
                on = np.abs(s - row) <= LEVEL_BAND
            if (on & near).any():
                on &= near
            elif i >= 2:
                # A hole in the hair it follows, which goes on below: on straight from the two joints above.
                joints.append(joints[-1] + (joints[-1] - joints[-2]) * (row - levels[i - 1])
                              / max(levels[i - 1] - levels[i - 2], 1e-9))
                follows.append(follow)
                if sweep_at is not None:
                    # On a swept chain, its hair is the followed hair nearest this row below the hole.
                    seed = _reached(seeds[-1], s, levels[i - 1] - LEVEL_BAND, np.inf, links) & (s > levels[i - 1])
                    if seed.any():
                        seed &= np.abs(s - row) <= np.abs(s[seed] - row).min() + LEVEL_BAND
                seeds.append(seed if sweep_at is not None and seed.any() else seeds[-1])
                i += 1
                continue
            elif near.any():
                # A hole at the second row, with one joint above: the hair it follows nearest that row.
                on = near & (np.abs(s - row) <= np.abs(s[near] - row).min() + LEVEL_BAND)
        if not on.any():
            on = np.abs(s - row) <= np.abs(s - row).min() + LEVEL_BAND
        point = _middle(xyz, weight, on, row, top)
        follow = float(np.average(u[on], weights=weight[on]))
        if i == 0 and not (on & (np.abs(u - follow) <= FOLLOW)).any():
            # The lane's hair parts into clumps around its middle: the first joint is the middle of the clump
            # nearest it, and the chain follows that one down.
            nearest = np.flatnonzero(on)[np.argmin(np.linalg.norm(xyz[on] - point, axis=1))]
            on &= np.abs(u - u[nearest]) <= FOLLOW
            point = _middle(xyz, weight, on, row, top)
            follow = float(np.average(u[on], weights=weight[on]))
        joints.append(point)
        follows.append(follow)
        if seed is None and sweep_at is not None:
            # On a swept chain, the joint's hair joined to the hair the chain followed (a card beside it within
            # FOLLOW across that isn't joined counts in its middle, but the chain doesn't follow it down).
            seed = _reached(seeds[-1], s, levels[i - 1] - LEVEL_BAND, row + LEVEL_BAND, links) & on
        elif seed is None:
            # Before a sweep, the hair a joint follows lies within FOLLOW across of it, not all its row's band.
            seed = on & (np.abs(u - follow) <= FOLLOW)
        seeds.append(seed if seed is not None and seed.any() else on)
        i += 1

    found = {}

    def below(j):
        # The hair under joint ``j`` that the chain follows: within FOLLOW across of it, or on a chain that followed
        # hair sweeping sideways, from the first joint placed on it on, the hair below it joined through the mesh
        # to the hair it was made of (kept, by the joint and its level: the trim loop and the tip ask again). A
        # joint made before the sweep follows the hair within FOLLOW across of it, as it was placed.
        k = j % len(levels)
        if sweep_at is None or k < sweep_at:
            return (s > levels[j]) & (np.abs(u - follows[j]) <= FOLLOW)
        key = (k, levels[j])
        if key not in found:
            found[key] = _reached(seeds[j], s, levels[j] - LEVEL_BAND, np.inf, links) & (s > levels[j])
        return found[key]

    # The chain keeps its rows at least MIN_CHAIN_LENGTH above where the hair under its last joint ends.
    while len(levels) > hairrig.CHAIN_BONES[0]:
        tail = below(-1)
        if tail.any() and _span(s[tail], weight[tail])[1] - levels[-1] >= least:
            break
        del levels[-1], joints[-1], follows[-1], seeds[-1]
    # At its fewest bones with none of the hair it follows under its last joint, a tip going on the way the last bone
    # ran (below) can lie beyond CHECK_REACH of the lane's hair, and the check would pull it back up past that joint.
    # Then the hair it follows is too short for two rows: the second joint goes halfway down it and the tip to its
    # end, as when that hair ends above the second row.
    tail = below(-1)
    off = False
    if len(levels) == hairrig.CHAIN_BONES[0] and not tail.any():
        along = joints[-1] - joints[-2]
        length = max(float(np.linalg.norm(along)), 1e-9)
        going = joints[-1] + along * (min(max(end - levels[-1], least), length) / length)
        off = float(np.linalg.norm(xyz - going, axis=1).min()) > CHECK_REACH
    if off:
        # The first joint is never on swept hair (a chain sweeps at its second joint at the earliest): the hair under
        # it as `below` measures it, within FOLLOW across.
        under = below(0)
        near = np.abs(u - follows[0]) <= FOLLOW
        row = 0.5 * (levels[0] + _span(s[under], weight[under])[1]) if under.any() else levels[1]
        if row < levels[1] and near.any():
            levels[1] = row
            on = near & (np.abs(s - row) <= np.abs(s[near] - row).min() + LEVEL_BAND)
            joints[1] = _middle(xyz, weight, on, row, top)
            follows[1] = float(np.average(u[on], weights=weight[on]))
            seeds[1] = on
            # The second joint is now on the hair the first follows: no joint is on swept hair.
            sweep_at = None
    # The tip: where the hair it follows under the last joint ends, or on the way the last bone ran, one row at most,
    # when none hangs there.
    tail = below(-1)
    if tail.any():
        end = max(min(end, _span(s[tail], weight[tail])[1]), levels[-1] + least)
        on = tail & (np.abs(s - end) <= np.abs(s[tail] - end).min() + LEVEL_BAND)
        if sweep_at is not None and sweep_at < len(levels):
            # Of the hair joined to it there, that within FOLLOW across of its end (of the hair nearest the end's
            # level, the one nearest the last joint across).
            ends = np.flatnonzero(on & (np.abs(s - end) == np.abs(s[on] - end).min()))
            nearest = ends[np.argmin(np.abs(u[ends] - follows[-1]))]
            on &= np.abs(u - u[nearest]) <= FOLLOW
        tip, tip_level = _middle(xyz, weight, on, end, top), end
        follows.append(float(np.average(u[on], weights=weight[on])))
    else:
        along = joints[-1] - joints[-2]
        length = max(float(np.linalg.norm(along)), 1e-9)
        reach = min(max(end - levels[-1], least), length)
        tip, tip_level = joints[-1] + along * (reach / length), levels[-1] + reach
        follows.append(follows[-1] + (follows[-1] - follows[-2]) * (reach / length))
    line = np.vstack(joints + [tip])
    across = np.array(follows)
    # The joints between the root and the tip eased halfway toward their neighbors' middle, sideways only (their
    # places across with them).
    line[1:-1, :2] = 0.25 * line[:-2, :2] + 0.5 * line[1:-1, :2] + 0.25 * line[2:, :2]
    across[1:-1] = 0.25 * across[:-2] + 0.5 * across[1:-1] + 0.25 * across[2:]
    if float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()) < least:
        return None
    state = np.zeros(len(line), dtype=np.int64)
    for i, level in enumerate(levels + [tip_level]):
        line[i], state[i] = _onto(line[i], level, across[i], xyz, s, u)
    line, state = _spaced(line, state)
    if line is None:
        return None
    return line[:-1], line[-1], levels[0], int((state > 0).sum()), int((state == 2).sum())


def _links(hair, vertices):
    """The mesh of ``hair``'s vertices ``vertices`` as `_pieces` takes it: (each one's welded vertex, numbered from 0
    among them; the mesh's edges between those; the vertices' places, `Hair.at`; the median length of those edges).
    None without the mesh."""
    if hair.group is None or hair.edges is None:
        return None
    local = np.unique(hair.group[vertices], return_inverse=True)[1].reshape(-1)
    groups = hair.group[vertices]
    number = np.full(int(max(hair.group.max(), hair.edges.max(initial=0))) + 1, -1, dtype=np.int64)
    number[groups] = local
    edges = number[hair.edges]
    edges = edges[(edges >= 0).all(axis=1)]
    points = hair.at[vertices]
    first = np.zeros(int(local.max()) + 1, dtype=np.int64)
    first[local] = np.arange(len(local))
    reach = float(np.median(np.linalg.norm(points[first[edges[:, 0]]] - points[first[edges[:, 1]]], axis=1))) if len(
        edges) else 0.0
    # Never more than a level step: pieces closer than that can't be told apart by level, and nothing farther may
    # join two strands.
    return local, edges, points, min(reach, LEVEL_STEP)


def plan_area(hair, members, across, count, bone_length=hairrig.BONE_SPACING, from_scalp=False):
    """The chains of one area of ``hair`` (`measure`): its vertices ``members`` (indices into `Hair`'s arrays) cut
    into ``count`` lanes side by side by ``across`` (one per hanging vertex, its place across the area, `across_back`,
    `across_side`, `across_front`), a `Lane` each, lane 0 lowest across. Lanes too short for a chain are left out;
    none when the area holds under MIN_AREA. ``bone_length`` is how long its bones are aimed to be. ``from_scalp``
    measures its levels by the distance from the scalp, whether it hangs or not (short hair's ring)."""
    members = np.asarray(members, dtype=np.int64)
    if not len(members) or float(hair.area[members].sum()) < MIN_AREA:
        return []
    weight = hair.area[members] + 1e-12
    level, hangs, reach, edged = _levels(hair, members, weight, from_scalp)
    if reach is None:
        return []
    xyz, u = hair.at[members], np.asarray(across, dtype=np.float64)[members]
    top = float(hair.at[:, 2].max()) if hangs else None
    grid = np.arange(reach[0], reach[1] + 0.5 * LEVEL_STEP, LEVEL_STEP)
    _weight, order, low, high = _bands(level, weight, grid)
    edges = np.full((len(grid), count + 1), np.nan)
    middle = np.full((len(grid), 3), np.nan)
    for j in range(len(grid)):
        band = order[low[j]:high[j]]
        if len(band) < 3:
            continue
        edges[j] = _quantiles(u[band], weight[band], np.arange(count + 1) / count)
        middle[j] = np.average(xyz[band], axis=0, weights=weight[band])
    if not np.isfinite(edges[:, 0]).any():
        return []
    edges = _smooth(_fill(edges), EDGE_SMOOTHING)
    middle = _smooth(_fill(middle), MIDDLE_SMOOTHING)
    # Below the widest level the edges follow the hair until a lane would narrow past NARROW; from there straight.
    width = np.diff(edges, axis=1)
    widest = int(np.argmax(edges[:, -1] - edges[:, 0]))
    for j in range(widest + 1, len(grid)):
        if (width[j] < NARROW * width[widest]).any():
            edges[j:] = edges[j - 1]
            break
    cut = np.array([np.interp(level, grid, edges[:, k]) for k in range(1, count)]).reshape(count - 1, len(members)).T
    lane = (u[:, None] > cut).sum(axis=1)
    # A vertex on the edge between two lanes (the middle column of mirrored hair) is in one lane, but both chains
    # count it at half its area, as the quantiles cut it: mirrored hair gets mirrored chains.
    on_edge = np.abs(u[:, None] - cut) <= ON_EDGE
    # Rows: the levels cutting the area's middle line into equal lengths, as many as it has bones.
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(middle, axis=0), axis=1))))
    bones = int(np.clip(math.ceil(arc[-1] / bone_length), *hairrig.CHAIN_BONES))
    rows = np.interp(arc[-1] * np.arange(bones + 1) / bones, arc, grid)
    lanes = []
    for number in range(count):
        if not (lane == number).any():
            continue
        edge = np.zeros(len(members), dtype=bool)
        if number:
            edge |= on_edge[:, number - 1]
        if number < count - 1:
            edge |= on_edge[:, number]
        mine = np.flatnonzero((lane == number) | edge)
        share = weight[mine] * np.where(edge[mine], 0.5, 1.0)
        chain = _chain(xyz[mine], share, u[mine], level[mine], rows, top, _links(hair, members[mine]))
        if chain is None:
            continue
        joints, tip, first, moved, lifted = chain
        # Light: most of its hair at its first row lies over the scalp; never when it starts at the scalp's edge.
        at = np.abs(level[mine] - first) <= LEVEL_BAND
        over = hair.share[members[mine]] >= hairing.SCALP_HALF
        light = not edged and share[at & over].sum() > 0.5 * share[at].sum()
        lanes.append(Lane(joints, tip, members[lane == number], number, bool(light), moved, lifted))
    return lanes


def lane_count(hair, members, across, spacing, most):
    """How many lanes Detect gives an area (its vertices ``members`` of ``hair``, places ``across`` as `plan_area`):
    one per ``spacing`` of its median width over its levels (between the SPREAD quantiles of each level's band),
    rounded, 1 to ``most``; 1 for hair wider than it is long."""
    members = np.asarray(members, dtype=np.int64)
    if not len(members):
        return 1
    weight = hair.area[members] + 1e-12
    level, _hangs, reach, _edged = _levels(hair, members, weight)
    if reach is None:
        return 1
    u = np.asarray(across, dtype=np.float64)[members]
    _weight, order, low, high = _bands(level, weight, np.arange(reach[0], reach[1] + 0.5 * LEVEL_STEP, LEVEL_STEP))
    widths = [float(np.ptp(_quantiles(u[order[a:b]], weight[order[a:b]], SPREAD))) for a, b in zip(low, high)
              if b - a >= 3]
    if not widths:
        return 1
    width = float(np.median(widths))
    if width > reach[1] - reach[0]:
        return 1
    return int(np.clip(round(width / spacing), 1, most))


def _rig_areas(ponytail):
    """The areas a rig's chains are planned in, in order: (name in `hairrig.AREAS`, label (`areas`), kind, across);
    with ``ponytail`` (Hair Type Ponytail) the back is the tail."""
    tail = "ponytail" if ponytail else "back"
    return (("bangs", BANGS, "bangs", across_front), ("left", LEFT, "side", across_side),
            ("right", RIGHT, "side", across_side), (tail, BACK, tail, across_back))


def _ring(hair, count):
    """Short hair cut into ``count`` equal sectors around the head, the first centered straight in front: (each of
    ``hair``'s vertices' sector, from 0 toward +X, and its place across it: its angle from the sector's middle times
    its distance from the head's axis, MIN_RADIUS at least)."""
    size = 2.0 * math.pi / count
    sector = np.rint(hair.phi / size).astype(np.int64)
    return sector % count, (hair.phi - sector * size) * np.maximum(hair.far, MIN_RADIUS)


def _detect(hair, label, short, spacing, ponytail):
    """The counts Detect gives ``hair`` (`measure`) with areas ``label`` (`areas`): see `detect`."""
    if short:
        bins = np.unique(np.floor(np.degrees(hair.phi) % 360.0 / RING_BIN))
        arc = len(bins) * math.radians(RING_BIN) * float(_quantiles(hair.far, hair.area + 1e-12, 0.5))
        return {"bangs": int(np.clip(round(arc / spacing), 1, hairrig.COUNT_RANGES["bangs"][1]))}
    found = {}
    for name, number, _kind, across in _rig_areas(ponytail):
        members = np.flatnonzero(label == number)
        # An area with no chain to run (too little hair, or too short) is left out, as `plan_area` builds none.
        if float(hair.area[members].sum()) < MIN_AREA or _levels(hair, members, hair.area[members] + 1e-12)[2] is None:
            continue
        found[name] = lane_count(hair, members, across(hair), spacing, hairrig.COUNT_RANGES[name][1])
    return found


def detect(points, tris, rig, *, cap=None, weld=None, spacing=hairrig.CHAIN_SPACING, ponytail=False):
    """The chain counts Detect proposes for hair ``points`` (world space, rest pose) with triangles ``tris`` hanging
    from ``rig`` (``cap`` and ``weld`` as `measure`; ``ponytail``: Hair Type Ponytail): ({name in `hairrig.AREAS`:
    count, within `hairrig.COUNT_RANGES`}, short). Each area with a chain to run gets `lane_count`'s lanes per
    ``spacing`` (Chain Spacing); the others are left out. Short hair (`areas`) has only "bangs": the arc its hanging
    hair covers around the head (RING_BIN) per ``spacing``, rounded."""
    hair = measure(points, tris, rig, cap, weld)
    if hair is None:
        return {}, False
    label, short = areas(hair, ponytail)
    return _detect(hair, label, short, spacing, ponytail), short


def plan(points, tris, rig, *, cap=None, weld=None, max_bones=hairrig.MAX_HAIR_BONES, spacing=hairrig.CHAIN_SPACING,
         ponytail=False, counts=None):
    """(chains, made, short): the chains (`hairrig.ChainPlan`, world space) of a hair skeleton for hair ``points``
    (world space, rest pose) with triangles ``tris`` hanging from ``rig`` (``cap`` and ``weld`` as `measure`), how
    many each area got ({name in `hairrig.AREAS`: count}) and whether the hair is short (`areas`).

    Each area is cut into as many lanes as ``counts`` (`hairrig.ChainCounts`) gives it (`plan_area`), without counts
    as many as `detect` proposes at ``spacing``: bangs across the front, left and right across the sides, the back,
    or with ``ponytail`` (Hair Type Ponytail) the tail, across the back. The chains' kind is their area's. Short hair
    is instead cut into Bang Chains equal sectors around the whole head, the first centered straight in front, one
    bangs chain each, measured from the scalp; nothing is tied into a tail. Over ``max_bones`` bones the bones get
    longer (`hairrig.BONE_SPACING` grows by `hairrig.BONE_GROWTH`) and every area is planned again, until the rig
    fits or every chain has `hairrig.CHAIN_BONES`[0] bones; nothing merges.

    The chains are in name order: by their first joint's angle around the head (`hairrig._angle`), then its X. Each
    area is one sheet, the sheets numbered from 0 in the order of their first chain; a chain's order is its place in
    its area: +X first behind, front to back at the sides, -X first across the front, from the front toward +X
    around short hair."""
    made = dict.fromkeys(hairrig.AREAS, 0)
    hair = measure(points, tris, rig, cap, weld)
    if hair is None:
        return [], made, False
    label, short = areas(hair, ponytail)
    if counts is None:
        found = _detect(hair, label, short, spacing, ponytail)
        counts = hairrig.ChainCounts(**{name: found.get(name, 0) for name in hairrig.ChainCounts._fields})
    every = _rig_areas(ponytail)
    kinds = {name: kind for name, _number, kind, _across in every}
    if short:
        sector, ring = _ring(hair, max(counts.bangs, 1))
        kinds = {"bangs": "bangs"}

    def lanes(bone_length):
        """(area name, place in the area, `Lane`) of every chain, its bones aimed at ``bone_length``."""
        if short:
            return [("bangs", k, lane) for k in range(counts.bangs)
                    for lane in plan_area(hair, np.flatnonzero(sector == k), ring, 1, bone_length, from_scalp=True)]
        return [(name, lane.number, lane) for name, number, _kind, across in every if getattr(counts, name) > 0
                for lane in plan_area(hair, np.flatnonzero(label == number), across(hair), getattr(counts, name),
                                      bone_length)]

    bone_length = hairrig.BONE_SPACING
    planned = lanes(bone_length)
    while sum(len(lane.joints) for _name, _place, lane in planned) > max_bones \
            and any(len(lane.joints) > hairrig.CHAIN_BONES[0] for _name, _place, lane in planned):
        bone_length *= hairrig.BONE_GROWTH
        planned = lanes(bone_length)
    chains = []
    for name, place, lane in planned:
        joints, tip = lane.joints + rig.offset, lane.tip + rig.offset
        length = float(np.linalg.norm(np.diff(np.vstack((joints, tip)), axis=0), axis=1).sum())
        chains.append((name, place, hairrig.ChainPlan(
            kinds[name], joints, tip, float(hair.area[lane.members].sum()), False, hairrig._angle(rig, joints[0]),
            np.sort(hair.index[lane.members]), light=lane.light, length=length, moved=lane.moved, lifted=lane.lifted)))
    chains.sort(key=lambda item: (round(item[2].angle, 6), item[2].joints[0][0]))
    sheets = {}
    for name, _place, chain in chains:
        chain.sheet = sheets.setdefault(name, len(sheets))
        made[name] += 1
    for name in sheets:
        mine = sorted(((place, chain) for other, place, chain in chains if other == name), key=lambda item: item[0])
        for order, (_place, chain) in enumerate(mine):
            chain.order = order
    return [chain for _name, _place, chain in chains], made, short
