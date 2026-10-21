# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless tests for the hair rig Hair Weights builds for the hair itself (Own Skeleton): its locks, the
chains planned on them (kinds, joints, the bone budget) and the bones they make on the hair's armature.

Run from the repository root:
    blender -b --factory-startup --python tests/test_hair_rig_core.py
"""

import json
import math
import os
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import hair  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import fitting, hairing, hairrig  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def join(parts):
    """Vertices and triangles of (vertices, quads) parts as one mesh, and each part's vertex indices."""
    co, tris, ranges = [], [], []
    start = 0
    for verts, quads in parts:
        co.append(np.asarray(verts, dtype=np.float64))
        for quad in quads:
            a, b, c, d = (start + k for k in quad)
            tris += [(a, b, c), (a, c, d)]
        ranges.append(np.arange(start, start + len(verts)))
        start += len(verts)
    return np.vstack(co), np.array(tris, dtype=np.int64), ranges


def rig():
    return hairing.Reference.get().rig(hair.RACE, 0)


def sheet(tops, bottoms, rows=20):
    """A sheet of hair hanging from the points ``tops`` (a row, in order) to ``bottoms``: (vertices, quads)."""
    tops, bottoms = np.asarray(tops, dtype=np.float64), np.asarray(bottoms, dtype=np.float64)
    columns = len(tops)
    verts = [tops + (bottoms - tops) * (i / rows) for i in range(rows + 1)]
    quads = [(i * columns + j, i * columns + j + 1, (i + 1) * columns + j + 1, (i + 1) * columns + j)
             for i in range(rows) for j in range(columns - 1)]
    return np.vstack(verts), quads


def back_ring(segments):
    """`hair.cap` with ``segments`` around, its lowest ring's vertex indices (by azimuth) and each vertex's
    azimuth around the skull's middle in degrees (0 straight behind, positive toward +X)."""
    points, faces = hair.cap(segments=segments)
    offset = points - hair.CENTER
    polar = np.degrees(np.arccos(offset[:, 2] / np.linalg.norm(offset, axis=1)))
    azimuth = np.degrees(np.arctan2(offset[:, 0], offset[:, 1]))
    ring = np.flatnonzero(polar > 99.0)
    return points, faces, ring[np.argsort(azimuth[ring])], azimuth


def one_piece_long_hair():
    """Long hair in one piece: the scalp cap, and a sheet hanging 40 cm from its lowest ring behind the head
    (82.5 degrees to either side, short of the ears), spreading out from the head to 30 cm wide at the
    bottom, sharing the ring's vertices. Returns (points, triangles, {"cap": indices, "sheet": indices});
    the shared vertices weld the two into one loose part."""
    cap, faces, ring, azimuth = back_ring(48)
    tops = cap[ring[np.abs(azimuth[ring]) <= 82.5 + 1e-6]]
    axis = np.array((0.0, rig().by_name[hairing.HEAD].head[1]))
    out = tops[:, :2] - axis
    out /= np.linalg.norm(out, axis=1)[:, None]
    bottoms = np.column_stack((axis + out * 0.15 / np.abs(out[:, 0]).max(),
                               np.full(len(tops), tops[:, 2].mean() - 0.40)))
    co, tris, (cap_ids, sheet_ids) = join([(cap, faces), sheet(tops, bottoms)])
    return co, tris, {"cap": cap_ids, "sheet": sheet_ids}


def long_hair():
    """`hair.parts(long=True)` as one mesh: (points, triangles, part index of each point, part names)."""
    named = hair.parts(long=True)
    co, tris, ranges = join([(verts, faces) for _name, verts, faces in named])
    part = np.zeros(len(co), dtype=np.int64)
    for index, indices in enumerate(ranges):
        part[indices] = index
    return co, tris, part, [name for name, _verts, _faces in named]


def bangs_hair():
    """The scalp cap and one 4 cm bangs strip from the forehead's hairline, hanging forward and down 12 cm,
    clear of the forehead (its hanging share passes LOCK_START 2 cm down and is 0.57 at the end):
    (points, triangles, {"cap": indices, "bangs": indices})."""
    co, tris, (cap_ids, bangs_ids) = join([hair.cap(), hair.strip((0.0, -0.10, 1.625), (0.0, -0.19, 1.54), 0.04,
                                                                   (1.0, 0.0, 0.0))])
    return co, tris, {"cap": cap_ids, "bangs": bangs_ids}


def spine_length(co, lock):
    line = hairrig.spine(co, lock)
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def lock_of(chain, locks):
    """The lock (of `find_locks`) a chain was planned from: the one with its vertices."""
    return next((lock for lock in locks if np.array_equal(lock.vertices, chain.vertices)), None)


def along(line, point):
    """How far along the polyline ``line`` its closest point to ``point`` is."""
    a, b = line[:-1], line[1:]
    seg = b - a
    lengths = np.linalg.norm(seg, axis=1)
    t = np.clip(((point - a) * seg).sum(1) / np.maximum(lengths ** 2, 1e-18), 0.0, 1.0)
    gap = np.linalg.norm(a + seg * t[:, None] - point, axis=1)
    best = int(np.argmin(gap))
    return float(np.concatenate(([0.0], np.cumsum(lengths)))[best] + t[best] * lengths[best])


def main_part(chain, part, area):
    """The part holding most of a chain's area."""
    return int(np.argmax(np.bincount(part[chain.vertices], area[chain.vertices])))


def describe_chains(chains, co, locks):
    for chain in chains:
        lock = lock_of(chain, locks)
        print("      chain: {:9s} {:d} bones, angle {:5.1f}, area {:.4f} m2, whole {!s:5}, spine {:s}, root {}, "
              "tip {}".format(chain.kind, len(chain.joints), chain.angle, chain.area, chain.whole,
                              "{:.3f} m".format(spine_length(co, lock)) if lock is not None else "(merged)",
                              np.round(chain.joints[0], 3), np.round(chain.tip, 3)))


def phi(points, model_rig):
    """Angle (degrees) of world ``points`` around the head's vertical axis: 0 in front, 180 behind."""
    model = model_rig.model(points)
    axis_y = model_rig.by_name[hairing.HEAD].head[1]
    return np.degrees(np.arctan2(np.abs(model[:, 0]), -(model[:, 1] - axis_y)))


def root_span(lock, co, model_rig):
    """How far around the head a lock's roots (its vertices at distance 0) reach: their angles around the
    head's axis, about their mean, times their mean distance from it."""
    roots = model_rig.model(co[lock.vertices[lock.distance <= 1e-9]])
    axis_y = model_rig.by_name[hairing.HEAD].head[1]
    angle = np.arctan2(roots[:, 0], -(roots[:, 1] - axis_y))
    mean = math.atan2(np.sin(angle).sum(), np.cos(angle).sum())
    relative = (angle - mean + math.pi) % (2.0 * math.pi) - math.pi
    radius = np.hypot(roots[:, 0], roots[:, 1] - axis_y).mean()
    return float((relative.max() - relative.min()) * radius)


def describe(locks, co, model_rig):
    for lock in locks:
        print("      lock: {:4d} vertices, angle {:5.1f}, area {:.4f} m2, whole {!s:5}, root span {:.3f} m, "
              "length {:.3f} m".format(len(lock.vertices), lock.angle, lock.area, lock.whole,
                                       root_span(lock, co, model_rig), float(lock.distance.max())))


def test_long_hair_locks():
    model_rig = rig()
    named = hair.parts(long=True)
    co, tris, ranges = join([(verts, faces) for _name, verts, faces in named])
    names = [name for name, _verts, _faces in named]
    part = np.zeros(len(co), dtype=np.int64)
    for index, indices in enumerate(ranges):
        part[indices] = index
    started = time.perf_counter()
    locks = hairrig.find_locks(co, tris, model_rig)
    print("      find_locks: {:.3f} s for {:d} vertices".format(time.perf_counter() - started, len(co)))
    describe(locks, co, model_rig)
    lock_parts = [set(part[lock.vertices].tolist()) for lock in locks]
    check(all(lock.area >= hairrig.MIN_LOCK_AREA for lock in locks), "long hair: no lock under MIN_LOCK_AREA")
    # The outer bangs strips each hang at two vertices only (a crumb under MIN_LOCK_AREA), and a crumb joins
    # the nearest lock (by root) of at least MIN_LOCK_AREA: the temple lock on its side. Beyond such crumbs,
    # no lock mixes two parts.
    area = hairing.vertex_areas(co, tris)
    mixed, main = [], []
    for lock in locks:
        shares = np.bincount(part[lock.vertices], area[lock.vertices], minlength=len(names))
        main.append(int(np.argmax(shares)))
        shares[main[-1]] = 0.0
        mixed.append(float(shares.max()))
    check(max(mixed) < hairrig.MIN_LOCK_AREA, "long hair: no lock mixes two parts beyond crumbs ({}, {})".format(
        [sorted(names[i] for i in p) for p in lock_parts], [round(m, 5) for m in mixed]))
    of = {index: [lock for lock, p in zip(locks, lock_parts) if index in p] for index in range(len(names))}
    check(not of[names.index("cap")], "long hair: the cap gives no lock")
    for name in ("lock_l", "lock_r"):
        mine = of[names.index(name)]
        check(len(mine) == 1 and 45.0 <= mine[0].angle <= 110.0,
              "long hair: {:s} gives one lock at the side ({})".format(name, [round(lock.angle, 1) for lock in mine]))
    mine = of[names.index("ponytail")]
    check(len(mine) == 1 and mine[0].whole and mine[0].angle > 110.0,
          "long hair: the ponytail gives one whole lock behind ({})".format(
              [(round(lock.angle, 1), lock.whole) for lock in mine]))
    if len(mine) == 1:
        length = float(mine[0].distance.max())
        check(0.35 < length < 0.5, "long hair: the ponytail's distance runs down its 42 cm ({:.3f} m)".format(length))
    for index, name in enumerate(names):
        if name != "bangs":
            continue
        mine = of[index]
        check(not any(names[m] == "bangs" for m in main), "long hair: bangs strip {:d} gives no lock of its own".format(
            index))
        side = float(np.sign(named[index][1][:, 0].mean()))
        if side == 0.0:
            check(not mine, "long hair: the middle bangs strip doesn't hang ({:d} locks)".format(len(mine)))
        else:
            temple = "lock_l" if side > 0 else "lock_r"
            check(len(mine) == 1 and names[main[locks.index(mine[0])]] == temple,
                  "long hair: bangs strip {:d}'s crumb joins {:s}, the nearest lock ({})".format(
                      index, temple, [names[main[locks.index(lock)]] for lock in mine]))
            if len(mine) != 1:
                continue
            # The crumb's vertices are marked joined and measured from the lock's root, about 10 cm away; the
            # lock's own vertices nearest its root are still at its start.
            lock = mine[0]
            crumb = part[lock.vertices] == index
            own = ~lock.joined
            nearest = np.argsort(np.linalg.norm(co[lock.vertices[own]] - lock.root, axis=1))[:2]
            check(np.array_equal(lock.joined, crumb) and lock.distance[crumb].min() > 0.05
                  and lock.distance[own][nearest].max() < 0.01,
                  "long hair: ...as joined vertices {:.3f} m from its root, its own nearest at {:.3f} m".format(
                      float(lock.distance[crumb].min()), float(lock.distance[own][nearest].max())))
    # Each lock's root lies where its (main) part leaves the head: the part's hanging vertex closest to the
    # head; crumbs that joined it don't move it. The ponytail's root is the middle of the ring where its 7 cm
    # tube leaves the head, and that vertex is on the ring's side: within the tube's radius.
    distance = model_rig.head_distance(co)
    hanging = hairing.hanging_share(distance, 0.025, 0.07) > hairrig.LOCK_START
    for lock, index in zip(locks, main):
        candidates = ranges[index][hanging[ranges[index]]]
        leaves = co[candidates[np.argmin(distance[candidates])]]
        gap = float(np.linalg.norm(lock.root - leaves))
        reach = 0.035 if names[index] == "ponytail" else 0.02
        check(gap <= reach, "long hair: the {:s} lock's root is where it leaves the head ({:.3f} m off)".format(
            names[index], gap))


def test_one_piece_long_hair():
    model_rig = rig()
    co, tris, _parts = one_piece_long_hair()
    started = time.perf_counter()
    locks = hairrig.find_locks(co, tris, model_rig)
    print("      find_locks: {:.3f} s for {:d} vertices".format(time.perf_counter() - started, len(co)))
    describe(locks, co, model_rig)
    check(5 <= len(locks) <= 8, "one piece: the sheet down the back is cut into 5 to 8 locks ({:d})".format(len(locks)))
    check(all(lock.angle > 110.0 for lock in locks), "one piece: all of them behind ({})".format(
        [round(lock.angle, 1) for lock in locks]))
    spans = [root_span(lock, co, model_rig) for lock in locks]
    check(all(span <= 0.08 for span in spans), "one piece: each lock's roots span 8 cm at most ({})".format(
        [round(span, 3) for span in spans]))
    angles = [lock.angle for lock in locks]
    check(angles == sorted(angles), "one piece: the locks come sorted by angle")
    check(all(a.root[0] < b.root[0] for a, b in zip(locks, locks[1:]) if abs(a.angle - b.angle) < 1e-6),
          "one piece: mirrored locks come lower root X first")
    weld = fitting.weld_groups(co, fitting.WELD_DISTANCE)[0]
    given = hairrig.find_locks(co, tris, model_rig, weld=weld)
    check(len(given) == len(locks) and all(np.array_equal(a.vertices, b.vertices) for a, b in zip(given, locks)),
          "one piece: the same locks with the weld given")
    check(not any(lock.joined.any() for lock in locks), "one piece: no crumbs joined")
    # Hair whose armature has its head elsewhere: the same locks, their roots moved along.
    offset = np.array((0.05, -0.1, 0.2))
    moved = hairrig.find_locks(co + offset, tris, model_rig.moved(offset))
    check(len(moved) == len(locks) and all(
        abs(a.angle - b.angle) < 1e-6 and np.allclose(a.root, b.root + offset)
        and np.array_equal(a.vertices, b.vertices) for a, b in zip(moved, locks)),
        "one piece: hair moved with its head gets the same locks")


def test_separate_strands():
    model_rig = rig()
    # Five 2 cm strands 6 cm apart, hanging 30 cm from just behind the head.
    strands = [hair.strip((x, 0.08, 1.53), (x, 0.10, 1.23), 0.02, (1.0, 0.0, 0.0))
               for x in (-0.12, -0.06, 0.0, 0.06, 0.12)]
    co, tris, ranges = join(strands)
    locks = hairrig.find_locks(co, tris, model_rig)
    describe(locks, co, model_rig)
    check(len(locks) == 5 and all(lock.whole for lock in locks),
          "strands: five separate strands give five whole locks ({:d})".format(len(locks)))
    strand = np.concatenate([np.full(len(indices), index) for index, indices in enumerate(ranges)])
    owners = [set(strand[lock.vertices].tolist()) for lock in locks]
    check(sorted(min(o) for o in owners if len(o) == 1) == list(range(5)), "strands: one lock per strand")


def test_crumbs():
    # Two tiny pieces of hair (1 x 2 cm) hanging behind the head, nothing else: no lock of 4 cm² to join, so
    # both are dropped.
    crumbs = [hair.strip((x, 0.12, 1.47), (x, 0.13, 1.45), 0.01, (1.0, 0.0, 0.0), rows=2) for x in (-0.05, 0.05)]
    co, tris, _ranges = join(crumbs)
    hanging = hairing.hanging_share(rig().head_distance(co), 0.025, 0.07) > hairrig.LOCK_START
    locks = hairrig.find_locks(co, tris, rig())
    check(hanging.all() and not locks, "crumbs: two tiny hanging crumbs give no lock ({:d})".format(len(locks)))


def test_parted_locks():
    model_rig = rig()
    cap, faces, ring, azimuth = back_ring(96)

    def at(angle):
        return cap[ring[np.argmin(np.abs(azimuth[ring] - angle))]]

    # Two strips of about 3 cm, 4 cm apart, each hanging from two of the cap's ring vertices, one to the
    # left and one to the right: their bottoms 20 cm apart.
    strips = [(cap, faces)]
    for sign in (-1.0, 1.0):
        inner, outer = at(sign * 3.75), at(sign * 22.5)
        side = outer - inner
        width = float(np.linalg.norm(side))
        top = 0.5 * (inner + outer)
        strips.append(hair.strip(top, (sign * 0.10, 0.08, top[2] - 0.25), width, side / width))
    co, tris, _ranges = join(strips)
    locks = hairrig.find_locks(co, tris, model_rig)
    describe(locks, co, model_rig)
    check(len(locks) == 2, "parted: two strips parting left and right give two locks ({:d})".format(len(locks)))
    check(len(locks) == 2 and locks[0].root[0] * locks[1].root[0] < 0.0, "parted: one on each side")

    # The cap between the two strips doesn't hang, so they are two regions already. The same width as one
    # sheet is one region: hanging straight down it is one lock (its roots span 7 cm); its two sides parting
    # left and right (the bottom 23 cm wide) make it two, by where they hang.
    tops = cap[ring[np.abs(azimuth[ring]) <= 22.5 + 1e-6]]
    for spread, expected in ((None, 1), (0.23, 2)):
        bottoms = tops + np.array((0.0, 0.01, -0.25))
        if spread is not None:
            bottoms[:, 0] = np.linspace(-0.5 * spread, 0.5 * spread, len(tops))
        co, tris, _ranges = join([(cap, faces), sheet(tops, bottoms)])
        locks = hairrig.find_locks(co, tris, model_rig)
        describe(locks, co, model_rig)
        check(len(locks) == expected, "parted: a 7 cm sheet {:s} gives {:d} lock(s) ({:d})".format(
            "hanging straight down" if spread is None else "parting left and right", expected, len(locks)))


def test_long_hair_chains():
    model_rig = rig()
    co, tris, part, names = long_hair()
    started = time.perf_counter()
    chains = hairrig.plan_rig(co, tris, model_rig)
    print("      plan_rig: {:.3f} s for {:d} vertices".format(time.perf_counter() - started, len(co)))
    locks = hairrig.find_locks(co, tris, model_rig)
    describe_chains(chains, co, locks)
    check(all(lock_of(chain, locks) is not None for chain in chains),
          "long hair chains: each from one lock (nothing over the budget to merge)")
    planned = [lock for lock in locks if lock_of(lock, chains) is not None]
    check([lock_of(chain, locks) for chain in chains] == planned, "long hair chains: in the locks' order")
    area = hairing.vertex_areas(co, tris)
    mains = [names[main_part(chain, part, area)] for chain in chains]
    for name in ("lock_l", "lock_r"):
        mine = [chain for chain, m in zip(chains, mains) if m == name]
        length = spine_length(co, lock_of(mine[0], locks)) if len(mine) == 1 else math.nan
        check(len(mine) == 1 and mine[0].kind == "side" and 0.20 <= length <= 0.26 and len(mine[0].joints) == 3,
              "long hair chains: {:s} is one side chain, its spine 0.20-0.26 m, 3 bones ({})".format(
                  name, [(chain.kind, len(chain.joints)) for chain in mine] + [round(length, 3)]))
    mine = [chain for chain, m in zip(chains, mains) if m == "ponytail"]
    length = spine_length(co, lock_of(mine[0], locks)) if len(mine) == 1 else math.nan
    check(len(mine) == 1 and mine[0].kind == "ponytail" and len(mine[0].joints) in (4, 5),
          "long hair chains: the ponytail is one ponytail chain of 4 or 5 bones ({})".format(
              [(chain.kind, len(chain.joints)) for chain in mine]))
    # The tube's rows are 2 cm apart and its roots on one side of its top rows: a 1 cm band holds the back of
    # one row and the front of the next. Smoothed, the spine runs down its 42 cm instead of zig-zagging.
    check(0.36 <= length <= 0.48, "long hair chains: the ponytail's spine runs down its 42 cm ({:.3f} m)".format(
        length))
    # The fixture's bangs barely hang: their crumbs join the temple locks (test_long_hair_locks), so there
    # are no others here; test_bangs_chain has bangs that hang.
    others = [(m, chain.kind, len(chain.joints)) for chain, m in zip(chains, mains)
              if m not in ("lock_l", "lock_r", "ponytail")]
    check(all(other == ("bangs", "bangs", 2) for other in others),
          "long hair chains: every other chain is a bangs strip's, bangs with 2 bones ({})".format(others))


def test_bangs_chain():
    model_rig = rig()
    co, tris, parts = bangs_hair()
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(chains, co, hairrig.find_locks(co, tris, model_rig))
    check(len(chains) == 1 and chains[0].kind == "bangs" and len(chains[0].joints) == 2,
          "bangs: a strip hanging over the forehead is one bangs chain of 2 bones ({})".format(
              [(chain.kind, len(chain.joints)) for chain in chains]))
    check(len(chains) == 1 and np.isin(chains[0].vertices, parts["bangs"]).all(), "bangs: ...made of the strip")


def chain_fixtures():
    """(name, points, triangles) of the hair the chain tests plan."""
    co, tris, _part, _names = long_hair()
    one_co, one_tris, _parts = one_piece_long_hair()
    bangs_co, bangs_tris, _parts = bangs_hair()
    return (("long hair", co, tris), ("one piece", one_co, one_tris), ("bangs", bangs_co, bangs_tris))


def test_pivot_at_root():
    model_rig = rig()
    for name, co, tris in chain_fixtures():
        locks = hairrig.find_locks(co, tris, model_rig)
        chains = hairrig.plan_rig(co, tris, model_rig)
        gaps = []
        for chain in chains:
            lock = lock_of(chain, locks)
            first = chain.joints[0]
            gaps.append((float(np.linalg.norm(first - lock.root)), abs(float(first[2] - lock.root[2])))
                        if lock is not None else (math.inf, math.inf))
        check(chains and all(gap <= 0.02 and height <= 0.02 for gap, height in gaps),
              "pivot: {:s}: every chain's first joint at its lock's root, at its height ({})".format(
                  name, [(round(gap, 4), round(height, 4)) for gap, height in gaps]))


def test_even_joints():
    model_rig = rig()
    for name, co, tris in chain_fixtures():
        locks = hairrig.find_locks(co, tris, model_rig)
        chains = hairrig.plan_rig(co, tris, model_rig)
        spreads, short = [], []
        for chain in chains:
            lock = lock_of(chain, locks)
            if lock is None:
                spreads.append(math.inf)
                continue
            line = hairrig.spine(co, lock)
            length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
            at = [along(line, joint) for joint in chain.joints] + [along(line, chain.tip)]
            gaps = np.diff(at)
            spreads.append(float((gaps.max() - gaps.min()) / gaps.max()))
            short.append(bool(at[-2] < length - 1e-6 and np.linalg.norm(chain.tip - line[-1]) < 1e-9
                              and np.linalg.norm(chain.tip - chain.joints[-1]) > 0.01))
        check(chains and all(spread < 0.25 for spread in spreads) and all(short),
              "even joints: {:s}: joint gaps along the spine within 25 %, the last joint short of the tip ({}, {})"
              .format(name, [round(spread, 3) for spread in spreads], short))


def test_kinds():
    for angle, length, whole, kind in ((10, 0.1, False, "bangs"), (10, 0.3, False, "side"), (80, 0.5, False, "side"),
                                       (150, 0.4, True, "ponytail"), (150, 0.4, False, "back"),
                                       (150, 0.2, True, "back")):
        got = hairrig.chain_kind(angle, length, whole)
        check(got == kind, "kinds: ({}, {}, {}) is {:s} ({:s})".format(angle, length, whole, kind, got))
    check(set(hairrig.KINDS) == {"bangs", "side", "back", "ponytail"}, "kinds: the four kinds")


def test_budget():
    model_rig = rig()
    co, tris, _parts = one_piece_long_hair()
    free = hairrig.plan_rig(co, tris, model_rig)
    chains = hairrig.plan_rig(co, tris, model_rig, max_bones=8)
    describe_chains(chains, co, hairrig.find_locks(co, tris, model_rig))
    total = sum(len(chain.joints) for chain in chains)
    check(sum(len(chain.joints) for chain in free) > 8, "budget: the sheet takes more than 8 bones unbudgeted ({:d})"
          .format(sum(len(chain.joints) for chain in free)))
    check(chains and total <= 8 and all(len(chain.joints) >= 2 for chain in chains),
          "budget: at most 8 bones, at least 2 per chain ({})".format([len(chain.joints) for chain in chains]))
    check(np.array_equal(np.sort(np.concatenate([chain.vertices for chain in chains])),
                         np.sort(np.concatenate([chain.vertices for chain in free]))),
          "budget: merged chains keep every vertex of the hair")
    check(len(set(np.concatenate([chain.vertices for chain in chains]).tolist()))
          == sum(len(chain.vertices) for chain in chains), "budget: no vertex in two chains")
    keys = [(round(chain.angle, 6), chain.joints[0][0]) for chain in chains]
    check(keys == sorted(keys), "budget: the chains stay in name order")
    check(abs(sum(chain.area for chain in chains) - sum(chain.area for chain in free)) <= 1e-9,
          "budget: merged chains keep the hair's area")
    # A merged lock hangs from the roots of both locks: its chain runs down the hair, not across the sheet.
    sideways = [max(abs(float(chain.joints[0][0] - end[0])) for end in (chain.joints[-1], chain.tip))
                for chain in chains]
    check(chains and all(side <= 2.0 * hairrig.LOCK_WIDTH for side in sideways),
          "budget: no chain runs across the sheet (first joint to last joint and tip {} m sideways)".format(
              [round(side, 3) for side in sideways]))


def card_hair(count, seed=0):
    """`hair.cards` as one mesh: (points, triangles)."""
    co, tris, _ranges = join(hair.cards(count, seed=seed))
    return co, tris


def test_card_hair_time():
    # Card hair of 3000 cards (66,000 vertices): about 2950 budget merges, each measuring only the locks it
    # merges, not the whole hair.
    model_rig = rig()
    co, tris = card_hair(3000)
    started = time.perf_counter()
    chains = hairrig.plan_rig(co, tris, model_rig)
    took = time.perf_counter() - started
    print("      plan_rig: {:.3f} s for 3000 cards, {:d} vertices".format(took, len(co)))
    check(chains and sum(len(chain.joints) for chain in chains) <= hairrig.MAX_HAIR_BONES,
          "card hair time: the cards fit the budget ({:d} chains)".format(len(chains)))
    check(took < 5.0, "card hair time: 3000 cards planned in under 5 s ({:.2f} s)".format(took))


def test_moved_hair():
    model_rig = rig()
    co, tris, _part, _names = long_hair()
    offset = np.array((0.3, 0.1, 0.05))
    chains = hairrig.plan_rig(co, tris, model_rig)
    moved = hairrig.plan_rig(co + offset, tris, model_rig.moved(offset))
    check(chains and len(moved) == len(chains) and all(
        a.kind == b.kind and len(a.joints) == len(b.joints)
        and np.allclose(a.joints, b.joints + offset, rtol=0.0, atol=1e-6)
        and np.allclose(a.tip, b.tip + offset, rtol=0.0, atol=1e-6) for a, b in zip(moved, chains)),
        "moved hair: the same kinds and bone counts, the joints moved by the offset ({} vs {})".format(
            [(chain.kind, len(chain.joints)) for chain in moved],
            [(chain.kind, len(chain.joints)) for chain in chains]))


# -----------------------------------------------------------------------------
# The rig on the armature

def rig_scene():
    """The long hair (`hair.build`) deformed by a new armature with c0201's j_kao, j_kubi and j_sebo_c (made by
    `hairing.ensure_bones`), and its chains built on the armature: (hair object, armature, plans, bone
    names, points, triangles, part index of each point, part names)."""
    hair.clear_scene()
    co, tris, part, names = long_hair()
    obj, _ranges = hair.build()
    armature = hairing.new_armature(bpy.context, "Armature", bpy.context.scene.collection)
    hairing.ensure_bones(bpy.context, armature, rig(), [hairing.HEAD, hairing.NECK, hairing.UPPER_BACK])
    hairing.add_armature_modifier(obj, armature)
    plans = hairrig.plan_rig(co, tris, rig())
    bones = hairrig.build_armature_rig(bpy.context, armature, plans, hair.RACE)
    return obj, armature, plans, bones, co, tris, part, names


def world(armature, local):
    return np.array(armature.matrix_world @ Vector(local))


def rig_json(armature):
    return json.loads(armature[hairrig.RIG_PROPERTY])


def state_chains(armature):
    """(kind, bone names) of each chain in the armature's rig state."""
    return [(chain.kind, chain.bones.split(",")) for chain in armature.magic_fit_hair_rig.chains]


def json_matches_state(armature):
    state = armature.magic_fit_hair_rig
    return rig_json(armature) == {"race": state.race, "chains": [{"kind": kind, "bones": bones}
                                                                 for kind, bones in state_chains(armature)]}


def test_armature_rig():
    obj, armature, plans, bones, _co, _tris, _part, _names = rig_scene()
    check(hairrig.bone_name(0, 0) == "j_ex_hair_01_a" and hairrig.bone_name(11, 2) == "j_ex_hair_12_c",
          "armature rig: bone names count chains from 01 and bones from a")
    expected = [hairrig.bone_name(c, b) for c, plan in enumerate(plans) for b in range(len(plan.joints))]
    check(plans and bones == expected, "armature rig: bones j_ex_hair_01_a and on, one per joint ({})".format(bones))
    have = sorted(bone.name for bone in armature.data.bones if bone.name.startswith(hairrig.BONE_PREFIX))
    check(have == sorted(expected), "armature rig: the armature has exactly those hair bones")
    firsts = [armature.data.bones.get(hairrig.bone_name(c, 0)) for c in range(len(plans))]
    check(all(first is not None and first.parent is not None and first.parent.name == hairing.HEAD
              for first in firsts), "armature rig: every chain's first bone hangs from j_kao")
    gaps, chained, deform = [], True, True
    for c, plan in enumerate(plans):
        ends = list(plan.joints[1:]) + [plan.tip]
        for b, (joint, end) in enumerate(zip(plan.joints, ends)):
            bone = armature.data.bones[hairrig.bone_name(c, b)]
            gaps.append(max(float(np.linalg.norm(world(armature, bone.head_local) - joint)),
                            float(np.linalg.norm(world(armature, bone.tail_local) - end))))
            if b:
                chained &= bone.parent is not None and bone.parent.name == hairrig.bone_name(c, b - 1)
            deform &= bone.use_deform
    check(max(gaps) <= 1e-5, "armature rig: each bone from its joint to the next joint or the tip ({:.2e})".format(
        max(gaps)))
    check(chained and deform, "armature rig: each bone hangs from the one before it and deforms")
    # The roll: each bone's Z axis points away from the head's vertical axis.
    axis = world(armature, armature.data.bones[hairing.HEAD].head_local)
    away = []
    for name in bones:
        bone = armature.data.bones[name]
        z = np.array(armature.matrix_world.to_3x3() @ bone.matrix_local.to_3x3().col[2])
        out = world(armature, bone.head_local)[:2] - axis[:2]
        away.append(float(np.dot(z[:2], out / max(np.linalg.norm(out), 1e-9))))
    check(min(away) > 0.0, "armature rig: every bone's Z axis points away from the head's axis ({:.3f})".format(
        min(away)))
    check(rig_json(armature) == {"race": hair.RACE, "chains": [
        {"kind": plan.kind, "bones": [hairrig.bone_name(c, b) for b in range(len(plan.joints))]}
        for c, plan in enumerate(plans)]}, "armature rig: the rig property lists the race, kinds and bones")
    check(json_matches_state(armature) and armature.magic_fit_hair_rig.race == hair.RACE,
          "armature rig: the state holds the same")
    check(hairrig.has_rig(armature, hair.RACE) and not hairrig.has_rig(armature, "c0801")
          and not hairrig.has_rig(obj, hair.RACE), "armature rig: has_rig for its race only")

    custom = hairrig.custom_rig(armature, hair.RACE)
    hair_bones = custom.hair_bones()
    check(custom.custom and custom.moved(np.zeros(3)).custom and not rig().custom,
          "custom rig: marked custom, also when moved; vanilla rigs aren't")
    check(sorted(bone.name for bone in hair_bones) == sorted(bones) and sorted(custom.skeleton) == sorted(bones),
          "custom rig: its hair bones are exactly the j_ex_hair_ bones, no j_kami_*")
    along = []
    for bone in hair_bones:
        data = armature.data.bones[bone.name]
        direction = world(armature, data.tail_local) - world(armature, data.head_local)
        along.append(float(np.linalg.norm(bone.x - direction / np.linalg.norm(direction))))
    check(max(along) <= 1e-5, "custom rig: each bone's X runs from its head to its tail ({:.2e})".format(max(along)))
    check(np.allclose(custom.offset, 0.0) and all(np.allclose(custom.by_name[bone.name].head, bone.head)
                                                   for bone in rig().bones),
          "custom rig: the body's bones as the race has them, not moved")
    # An armature with its head elsewhere: the rig moves with it, the bones stay where they are on it.
    armature.location = (0.3, -0.1, 0.05)
    bpy.context.view_layer.update()
    moved = hairrig.custom_rig(armature, hair.RACE)
    check(np.allclose(moved.offset, (0.3, -0.1, 0.05)) and all(
        np.allclose(moved.by_name[name].head, custom.by_name[name].head, atol=1e-6) for name in bones),
        "custom rig: an armature moved by an offset gives the same rig, moved by it")
    armature.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    # Building again replaces every hair bone.
    again = hairrig.build_armature_rig(bpy.context, armature, plans[:1], hair.RACE)
    have = sorted(bone.name for bone in armature.data.bones if bone.name.startswith(hairrig.BONE_PREFIX))
    check(have == sorted(again) == sorted(expected[:len(plans[0].joints)])
          and len(rig_json(armature)["chains"]) == 1, "armature rig: building again replaces the hair bones")


def test_weights_follow_locks():
    obj, armature, plans, bones, co, tris, part, names = rig_scene()
    custom = hairrig.custom_rig(armature, hair.RACE)
    bone_names, weights = hairing.hair_weights(co, tris, custom)
    column = {name: i for i, name in enumerate(bone_names)}
    kami = [i for i, name in enumerate(bone_names) if name.startswith("j_kami_")]
    check(not kami or float(np.abs(weights[:, kami]).max()) <= 1e-9,
          "follow locks: no weight on j_kami_* ({:.3g})".format(float(np.abs(weights[:, kami]).max()) if kami else 0.0))
    area = hairing.vertex_areas(co, tris)
    hanging = hairing.hanging_share(custom.head_distance(co), 0.025, 0.07) > hairrig.LOCK_START
    hair_columns = [column[name] for name in bones]
    for name in ("lock_l", "lock_r", "ponytail"):
        index = names.index(name)
        mine = [c for c, plan in enumerate(plans) if main_part(plan, part, area) == index]
        if len(mine) != 1:
            check(False, "follow locks: {:s} has one chain ({})".format(name, mine))
            continue
        own = [column[hairrig.bone_name(mine[0], b)] for b in range(len(plans[mine[0]].joints))]
        vertices = np.flatnonzero((part == index) & hanging)
        total = weights[np.ix_(vertices, hair_columns)].sum(1)
        ratio = weights[np.ix_(vertices, own)].sum(1) / np.maximum(total, 1e-12)
        check(len(vertices) and float(ratio.mean()) >= 0.8,
              "follow locks: {:s}'s hanging hair has {:.3f} of its chain weight on its own chain".format(
                  name, float(ratio.mean())))


def test_edited_rig_is_kept():
    obj, armature, plans, bones, _co, _tris, _part, _names = rig_scene()
    before = state_chains(armature)
    obj.vertex_groups.new(name="Bone").add([0], 1.0, 'REPLACE')
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    edit_bones.remove(edit_bones[before[0][1][-1]])
    last = edit_bones[before[1][1][-1]]
    added = edit_bones.new("Bone")
    added.head = last.tail
    added.tail = last.tail - Vector((0.0, 0.0, 0.05))
    added.parent = last
    # A new chain behind the head, its first bone named for the hair, the second not.
    top = edit_bones.new("j_ex_hair_new")
    top.head, top.tail = (0.0, 0.10, 1.55), (0.0, 0.12, 1.42)
    top.parent = edit_bones[hairing.HEAD]
    below = edit_bones.new("Bone.001")
    below.head, below.tail = top.tail, (0.0, 0.13, 1.30)
    below.parent = top
    renamed = hairrig.sync_rig(bpy.context, armature, hair.RACE)
    print("      renamed:", renamed)
    after = state_chains(armature)
    new_name = "j_ex_hair_02_" + chr(ord("a") + len(before[1][1]))
    count = len(before) + 1
    check(armature.mode == 'OBJECT', "edited rig: the armature is back in Object Mode")
    check(len(after) == count and after[0][1] == before[0][1][:-1],
          "edited rig: chain 01 has one bone fewer ({})".format(after[0][1] if after else None))
    check(len(after) == count and after[1][1] == before[1][1] + [new_name],
          "edited rig: chain 02 has one more, named {:s} ({})".format(new_name, after[1][1] if len(after) > 1 else None))
    check(after[2:-1] == before[2:], "edited rig: the other chains are as they were")
    new_chain = ["j_ex_hair_{:02d}_a".format(count), "j_ex_hair_{:02d}_b".format(count)]
    check(after[-1] == ("back", new_chain), "edited rig: the new chain is {:s}, back hair ({})".format(
        ", ".join(new_chain), after[-1]))
    check(sorted(renamed) == sorted(["Bone → " + new_name, "j_ex_hair_new → " + new_chain[0],
                                     "Bone.001 → " + new_chain[1]]), "edited rig: the renames are listed")
    check(obj.vertex_groups.get(new_name) is not None and obj.vertex_groups.get("Bone") is None,
          "edited rig: the hair's vertex group Bone took the bone's new name")
    check([kind for kind, _bones in after[:len(before)]] == [kind for kind, _bones in before],
          "edited rig: the chains keep their kinds")
    check(json_matches_state(armature) and [chain["bones"] for chain in rig_json(armature)["chains"]]
          == [bones for _kind, bones in after], "edited rig: the rig property lists the chains as they are now")
    head = armature.data.bones[new_chain[0]]
    check(head.parent is not None and head.parent.name == hairing.HEAD, "edited rig: the new chain hangs from j_kao")

    # A rig from another tool: no state, only the property. Its kinds are kept, by chain number.
    data = rig_json(armature)
    data["chains"][0]["kind"] = "ponytail"
    armature.magic_fit_hair_rig.chains.clear()
    armature[hairrig.RIG_PROPERTY] = json.dumps(data)
    renamed = hairrig.sync_rig(bpy.context, armature, hair.RACE)
    check(not renamed and [kind for kind, _bones in state_chains(armature)]
          == [chain["kind"] for chain in data["chains"]], "edited rig: with no state, the property's kinds are kept")


def chains_are_clean(armature):
    """Whether every hair bone of the armature is in exactly one chain of the rig property, and each chain
    lists its bones root to tip, each bone's parent the one before it."""
    chains = rig_json(armature)["chains"]
    listed = [name for chain in chains for name in chain["bones"]]
    have = sorted(bone.name for bone in armature.data.bones if bone.name.startswith(hairrig.BONE_PREFIX))
    chained = all(armature.data.bones[name].parent is not None and armature.data.bones[name].parent.name == parent
                  for chain in chains for parent, name in zip(chain["bones"], chain["bones"][1:]))
    return sorted(listed) == have and len(set(listed)) == len(listed) and chained


def test_branch_starts_chain():
    _obj, armature, _plans, _bones, _co, _tris, _part, _names = rig_scene()
    before = state_chains(armature)
    first = before[0][1][0]
    # Two bones branching off chain 01's first bone, one below the other, 5 cm each, 3 cm off its chain.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    root = edit_bones[first]
    upper = edit_bones.new("Branch")
    upper.head = root.tail + Vector((0.03, 0.0, 0.0))
    upper.tail = upper.head - Vector((0.0, 0.0, 0.05))
    upper.parent = root
    lower = edit_bones.new("Branch.001")
    lower.head, lower.tail = upper.tail, upper.tail - Vector((0.0, 0.0, 0.05))
    lower.parent = upper
    renamed = hairrig.sync_rig(bpy.context, armature, hair.RACE)
    print("      renamed:", renamed)
    after = state_chains(armature)
    number = len(before) + 1
    branch = ["j_ex_hair_{:02d}_a".format(number), "j_ex_hair_{:02d}_b".format(number)]
    check(len(after) == len(before) + 1 and after[:len(before)] == before,
          "branch: one more chain, the others as they were ({:d} -> {:d})".format(len(before), len(after)))
    check(after and after[-1][1] == branch and after[-1][0] in hairrig.KINDS
          and armature.data.bones.get(branch[0]) is not None
          and armature.data.bones[branch[0]].parent.name == first,
          "branch: the branch is chain {:02d}, {:s}, hanging from {:s} ({})".format(
              number, ", ".join(branch), first, after[-1] if after else None))
    check(chains_are_clean(armature) and json_matches_state(armature),
          "branch: every hair bone in exactly one chain, each chain root to tip")

    # One more bone off the same bone (a third child): one more chain again; the first child keeps its chain.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    root = edit_bones[first]
    twig = edit_bones.new("Twig")
    twig.head = root.tail - Vector((0.03, 0.0, 0.0))
    twig.tail = twig.head - Vector((0.0, 0.0, 0.05))
    twig.parent = root
    hairrig.sync_rig(bpy.context, armature, hair.RACE)
    again = state_chains(armature)
    twig_name = "j_ex_hair_{:02d}_a".format(number + 1)
    check(len(again) == len(after) + 1 and again[:len(after)] == after and again[-1][1] == [twig_name],
          "branch: a second branch off the same bone is chain {:02d} ({})".format(number + 1, again[-1]))
    check(chains_are_clean(armature) and json_matches_state(armature),
          "branch: still every hair bone in exactly one chain")


def test_kind_change_writes_json():
    _obj, armature, _plans, _bones, _co, _tris, _part, _names = rig_scene()
    chains = armature.magic_fit_hair_rig.chains
    check(len(chains) > 1 and chains[1].kind != "back", "kind change: chain 02 isn't back hair yet")
    chains[1].kind = "back"
    check(rig_json(armature)["chains"][1]["kind"] == "back" and json_matches_state(armature),
          "kind change: the rig property follows")


def main():
    magic_fit.register()
    try:
        for test in (test_long_hair_locks, test_one_piece_long_hair, test_separate_strands, test_crumbs,
                     test_parted_locks, test_long_hair_chains, test_bangs_chain, test_pivot_at_root, test_even_joints,
                     test_kinds, test_budget, test_card_hair_time, test_moved_hair, test_armature_rig, test_weights_follow_locks,
                     test_edited_rig_is_kept, test_branch_starts_chain, test_kind_change_writes_json):
            print("==", test.__name__)
            try:
                test()
            except Exception:
                traceback.print_exc()
                FAILURES.append(test.__name__ + ": exception")
    finally:
        magic_fit.unregister()
    print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
    sys.stdout.flush()
    os._exit(1 if FAILURES else 0)


main()
