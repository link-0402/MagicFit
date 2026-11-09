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
sys.path.insert(0, os.path.join(ROOT, "tools"))

import hair  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import fitting, hairing, hairrig  # noqa: E402

FAILURES = []
# The head the scalp fixtures hang from: the scalp map is tuned on the Miqo'te, Au Ra and Viera women.
MIQOTE = "c0801"
# The real test hair (DOUX Maeve, c0801 hair 108), read only; the test is skipped without it.
MAEVE = os.environ.get("MAGIC_FIT_MAEVE", r"G:\Penumbra\DOUX Maeve\chara\human\c0801\obj\hair\h0108\model"
                                          r"\c0801h0108_hir.mdl")
# Another real test hair (Briar, c0801 hair 137), read only; the test is skipped without it.
BRIAR = os.environ.get("MAGIC_FIT_BRIAR", r"G:\Penumbra\Briar Hair with custom physics\chara\human\c0801\obj\hair"
                                          r"\h0137\model\c0801h0137_hir.mdl")
# Where the neck fixtures' sheet ends (metres, world Z), far below the nape hairline (about 1.42).
NECK_BOTTOM = 1.15
# plan_rig on 3000 cards (test_card_hair_time) before sheets were cut into columns, in seconds (measured
# 2026-11-09): the columns may cost at most as much again, plus a second.
CARDS_TIME = 3.863


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


def rig(race=hair.RACE):
    return hairing.Reference.get().rig(race, 0)


def hanging(co, tris, model_rig):
    """Which of hair ``co`` hang (go into locks), by the scalp and the hair's cap depth."""
    return hairrig._hanging(model_rig.scalp(co), model_rig.head_distance(co), hairing.cap_depth(co, tris, model_rig))


def sheet(tops, bottoms, rows=20):
    """A sheet of hair hanging from the points ``tops`` (a row, in order) to ``bottoms``: (vertices, quads)."""
    tops, bottoms = np.asarray(tops, dtype=np.float64), np.asarray(bottoms, dtype=np.float64)
    columns = len(tops)
    verts = [tops + (bottoms - tops) * (i / rows) for i in range(rows + 1)]
    quads = [(i * columns + j, i * columns + j + 1, (i + 1) * columns + j + 1, (i + 1) * columns + j)
             for i in range(rows) for j in range(columns - 1)]
    return np.vstack(verts), quads


def back_ring(segments, race=hair.RACE):
    """`hair.cap` with ``segments`` around, its lowest ring's vertex indices (by azimuth) and each vertex's
    azimuth around the skull's middle in degrees (0 straight behind, positive toward +X)."""
    points, faces = hair.cap(segments=segments, race=race)
    offset = points - hair.center(race)
    polar = np.degrees(np.arccos(offset[:, 2] / np.linalg.norm(offset, axis=1)))
    azimuth = np.degrees(np.arctan2(offset[:, 0], offset[:, 1]))
    ring = np.flatnonzero(polar > 99.0)
    return points, faces, ring[np.argsort(azimuth[ring])], azimuth


def back_sheet(half_width=0.15, depth=0.40, shape=None, sweep=0.0):
    """Long hair in one piece: the scalp cap, and a sheet hanging ``depth`` from its lowest ring behind the head
    (82.5 degrees to either side, short of the ears), spreading out from the head to ``half_width`` either side
    at the bottom, sharing the ring's vertices. The bottom row is ``shape(x)`` (of its X, None: 0) higher, then
    moved ``sweep`` along X. Returns (points, triangles, {"cap": indices, "sheet": indices}); the shared
    vertices weld the two into one loose part."""
    cap, faces, ring, azimuth = back_ring(48)
    tops = cap[ring[np.abs(azimuth[ring]) <= 82.5 + 1e-6]]
    axis = np.array((0.0, rig().by_name[hairing.HEAD].head[1]))
    out = tops[:, :2] - axis
    out /= np.linalg.norm(out, axis=1)[:, None]
    ends = axis + out * half_width / np.abs(out[:, 0]).max()
    lift = np.zeros(len(tops)) if shape is None else np.array([shape(x) for x in ends[:, 0]], dtype=np.float64)
    bottoms = np.column_stack((ends, tops[:, 2].mean() - depth + lift))
    bottoms[:, 0] += sweep
    co, tris, (cap_ids, sheet_ids) = join([(cap, faces), sheet(tops, bottoms)])
    return co, tris, {"cap": cap_ids, "sheet": sheet_ids}


def one_piece_long_hair():
    """`back_sheet` as it comes: 40 cm long, 30 cm wide at the bottom."""
    return back_sheet()


def side_strip(azimuth, length, width, race=hair.RACE, polar=95.0):
    """A strip of side hair (`hair.strip`) ``width`` wide from 1 cm off the head at ``polar`` degrees from the top
    and ``azimuth`` around the skull's middle (0 straight behind, as `hair.patch`), hanging ``length`` down and
    leaning 4 cm out per ``length``, its width running around the head: (vertices, quads)."""
    p, a = math.radians(polar), math.radians(azimuth)
    top = hair._onto_head(np.array(((math.sin(p) * math.sin(a), math.sin(p) * math.cos(a), math.cos(p)),)), 0.01,
                          race)[0][0]
    out = np.array((math.sin(a), math.cos(a), 0.0))
    d = 0.04 * out - np.array((0.0, 0.0, length))
    return hair.strip(top, top + d * length / np.linalg.norm(d), width, np.array((math.cos(a), -math.sin(a), 0.0)))


def columns_by_sheet(chains):
    """{sheet: its chains by order} of the column chains (sheet 0 and on) among ``chains``."""
    sheets = {}
    for chain in chains:
        if chain.sheet >= 0:
            sheets.setdefault(chain.sheet, []).append(chain)
    return {sheet: sorted(mine, key=lambda chain: chain.order) for sheet, mine in sheets.items()}


def crossed(a, b):
    """Whether chains ``a`` and ``b`` cross around the head: the sign of a's X less b's at their first joints
    differs from its sign at a joint both have, or at their tips."""
    sign = np.sign(a.joints[0][0] - b.joints[0][0])
    later = [a.joints[k][0] - b.joints[k][0] for k in range(1, min(len(a.joints), len(b.joints)))]
    return any(np.sign(x) != sign for x in later + [a.tip[0] - b.tip[0]])


def any_crossed(chains):
    """Whether any two neighbors of ``chains`` (one sheet's, by order) cross."""
    return any(crossed(a, b) for a, b in zip(chains, chains[1:]))


def drift(chain):
    """How far a chain's joints and tip lie at most from its first joint's X (metres)."""
    return max(abs(float(point[0] - chain.joints[0][0])) for point in np.vstack((chain.joints, chain.tip)))


def describe_columns(chains):
    for chain in chains:
        print("      column: {:9s} sheet {:2d} order {:2d}, {:d} bones, root {}, tip {}, drift {:.3f}".format(
            chain.kind, chain.sheet, chain.order, len(chain.joints), np.round(chain.joints[0], 3),
            np.round(chain.tip, 3), drift(chain)))


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
    clear of the forehead (it stands off the cap from about 2 cm down):
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
    # The bangs strips run down over the forehead, below the hairline: each hangs there however close to the
    # skin, a lock of its own (too short for a chain, see test_long_hair_chains).
    for index, name in enumerate(names):
        if name != "bangs":
            continue
        mine = of[index]
        check(len(mine) == 1 and set(part[mine[0].vertices].tolist()) == {index} and not mine[0].joined.any(),
              "long hair: bangs strip {:d} gives a lock of its own ({:d} locks)".format(index, len(mine)))
    # Each lock's root lies where its (main) part leaves the head: the part's hanging vertex closest to the
    # head; crumbs that joined it don't move it. The ponytail's root is the middle of the ring where its 7 cm
    # tube leaves the head, and that vertex is on the ring's side: within the tube's radius.
    distance = model_rig.head_distance(co)
    hangs = hanging(co, tris, model_rig)
    for lock, index in zip(locks, main):
        candidates = ranges[index][hangs[ranges[index]]]
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
    hangs = hanging(co, tris, rig())
    locks = hairrig.find_locks(co, tris, rig())
    check(hangs.all() and not locks, "crumbs: two tiny hanging crumbs give no lock ({:d})".format(len(locks)))


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
    # The temple locks hang from the side hairline (about z 1.45) down to z 1.28.
    for name in ("lock_l", "lock_r"):
        mine = [chain for chain, m in zip(chains, mains) if m == name]
        length = spine_length(co, lock_of(mine[0], locks)) if len(mine) == 1 else math.nan
        check(len(mine) == 1 and mine[0].kind == "side" and 0.15 <= length <= 0.20 and len(mine[0].joints) == 2,
              "long hair chains: {:s} is one side chain, its spine 0.15-0.20 m, 2 bones ({})".format(
                  name, [(chain.kind, len(chain.joints)) for chain in mine] + [round(length, 3)]))
    sides = [chain.sheet for chain in chains if chain.kind == "side"]
    check(len(sides) == 2 and len(set(sides)) == 2,
          "long hair chains: the two side chains are in different sheets ({})".format(sides))
    mine = [chain for chain, m in zip(chains, mains) if m == "ponytail"]
    length = spine_length(co, lock_of(mine[0], locks)) if len(mine) == 1 else math.nan
    check(len(mine) == 1 and mine[0].kind == "ponytail" and len(mine[0].joints) in (4, 5),
          "long hair chains: the ponytail is one ponytail chain of 4 or 5 bones ({})".format(
              [(chain.kind, len(chain.joints)) for chain in mine]))
    # The tube's rows are 2 cm apart and its roots on one side of its top rows: a 1 cm band holds the back of
    # one row and the front of the next. Smoothed, the spine runs down its 42 cm instead of zig-zagging.
    check(0.36 <= length <= 0.48, "long hair chains: the ponytail's spine runs down its 42 cm ({:.3f} m)".format(
        length))
    # The fixture's bangs hang only below the hairline, too short for chains (test_long_hair_locks), so there
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


def planned_alone(co, locks, model_rig):
    """The chain of each of ``locks`` on its own (`hairrig.plan_chain`), those too short for one left out."""
    return [chain for chain in (hairrig.plan_chain(co, lock, model_rig) for lock in locks) if chain is not None]


def test_pivot_at_root():
    model_rig = rig()
    for name, co, tris in chain_fixtures():
        locks = hairrig.find_locks(co, tris, model_rig)
        # Each chain from one lock.
        chains = planned_alone(co, locks, model_rig)
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
        chains = planned_alone(co, locks, model_rig)
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


def wavy_strip(amplitude=0.03, wavelength=0.30, rows=60):
    """A 3 cm strip hanging 40 cm down behind the head, waving ``amplitude`` (metres) from side to side once every
    ``wavelength`` down it, like a wavy ribbon card: (points, triangles)."""
    top, bottom = np.array((0.0, 0.08, 1.53)), np.array((0.0, 0.10, 1.13))
    verts = []
    for i in range(rows + 1):
        centre = top + (bottom - top) * (i / rows)
        centre[0] += amplitude * math.sin(2.0 * math.pi * float(np.linalg.norm(bottom - top)) * (i / rows) / wavelength)
        verts += [centre - (0.015, 0.0, 0.0), centre + (0.015, 0.0, 0.0)]
    co, tris, _ranges = join([(np.array(verts), [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(rows)])])
    return co, tris


def test_wavy_lock_straight():
    # A wavy card swings 3 cm from side to side as it hangs; its chain runs down the card's general line in one
    # arc, as the game's chains do (none bends over 23 degrees between bones, none crosses the line from its root
    # to its tip), not from crest to crest.
    model_rig = rig()
    co, tris = wavy_strip()
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(chains, co, [])
    check(len(chains) == 1, "wavy lock: one chain ({:d})".format(len(chains)))
    if len(chains) != 1:
        return
    chain = chains[0]
    line = np.vstack((chain.joints, chain.tip))
    chord = (line[-1] - line[0]) / np.linalg.norm(line[-1] - line[0])
    offsets = line[1:-1] - line[0]
    offsets -= np.outer(offsets @ chord, chord)
    widest = offsets[int(np.argmax(np.linalg.norm(offsets, axis=1)))]
    across = offsets @ (widest / max(float(np.linalg.norm(widest)), 1e-12))
    bones = np.diff(line, axis=0)
    bones /= np.linalg.norm(bones, axis=1)[:, None]
    bends = [math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0)))) for a, b in zip(bones[:-1], bones[1:])]
    check(float(across.min()) >= -0.005 and max(bends) <= 25.0,
          "wavy lock: the joints stay on one side of the line from root to tip, bends up to 25 degrees "
          "(offsets {} m, bends {})".format(np.round(across, 3).tolist(), [round(bend, 1) for bend in bends]))
    check(off_hair(chain, co) <= hairrig.SPINE_REACH,
          "wavy lock: the chain stays on its hair ({:.4f} m)".format(off_hair(chain, co)))


def test_kinds():
    for angle, length, whole, kind in ((10, 0.1, False, "bangs"), (10, 0.3, False, "side"), (80, 0.5, False, "side"),
                                       (150, 0.4, True, "ponytail"), (150, 0.4, False, "back"),
                                       (150, 0.2, True, "back")):
        got = hairrig.chain_kind(angle, length, whole)
        check(got == kind, "kinds: ({}, {}, {}) is {:s} ({:s})".format(angle, length, whole, kind, got))
    check(set(hairrig.KINDS) == {"bangs", "side", "back", "ponytail"}, "kinds: the four kinds")


def test_columns():
    r, w = 0.1, 0.08
    side_areas = (0.002, 0.02, 0.02, 0.02)
    for case, kind, arcs, areas, expected in (
            ("wide back", "back", (-0.12, -0.05, 0.0, 0.03, 0.06, 0.20), None, [[0, 1], [2, 3], [4, 5]]),
            ("narrow back", "back", (-0.02, 0.0, 0.03), None, [[0, 1, 2]]),
            ("bangs, no limit", "bangs", (-0.17, -0.08, 0.0, 0.09, 0.16), None, [[0], [1], [2], [3], [4]]),
            ("small edge column", "back", (-0.08, 0.0, 0.08), (0.002, 0.02, 0.02), [[0, 1], [2]]),
            ("tie goes to the larger", "back", (-0.08, 0.0, 0.08), (0.02, 0.002, 0.03), [[0], [1, 2]]),
            ("a sheet's only column", "back", (0.0,), (0.001,), [[0]]),
            ("stray before 3 side columns", "side", (0.01, 0.09, 0.17, 0.25), side_areas, [[0, 1], [2, 3]]),
            ("the other side", "side", (-0.01, -0.09, -0.17, -0.25), side_areas, [[0, 1], [2, 3]])):
        theta = np.array(arcs) / r + (math.pi if kind == "back" else 0.0)
        got = hairrig._columns(kind, theta, np.full(len(arcs), 0.01) if areas is None else np.array(areas), r, w)
        check(got == expected, "columns: {:s} gives {} ({})".format(case, expected, got))
    model_rig = rig()
    head = np.asarray(model_rig.by_name[hairing.HEAD].head, dtype=np.float64) + model_rig.offset
    theta, radius = hairrig._polar(model_rig, np.array((head + (0.0, 0.1, 0.0), head + (0.1, 0.0, 0.0))))
    # The head bone is stored in single precision, its X a little off 0.
    check(abs(abs(float(theta[0])) - math.pi) < 1e-5 and abs(float(radius[0]) - 0.1) < 1e-5
          and abs(float(theta[1]) - 0.5 * math.pi) < 1e-5,
          "columns: a point behind the head is at |theta| pi, radius 0.1, one to +X at pi/2 ({}, {})".format(
              np.round(theta, 6).tolist(), np.round(radius, 6).tolist()))


def back_columns(name, chains):
    """The back chains of ``chains`` if they are 3 in one sheet, by order (else None), checked as such."""
    back = [chain for chain in chains if chain.kind == "back"]
    sheets = {chain.sheet for chain in back}
    ok = len(back) == 3 and len(sheets) == 1 and min(sheets) >= 0
    check(ok and [chain.order for chain in columns_by_sheet(back)[min(sheets)]] == [0, 1, 2],
          "{:s}: 3 back chains in one sheet, orders 0-2 ({})".format(
              name, [(chain.kind, chain.sheet, chain.order) for chain in chains]))
    return columns_by_sheet(back)[min(sheets)] if ok else None


def test_u_bottom():
    # A sheet whose sides are 20 cm shorter than its middle: the middle column's chain hangs straight down, and
    # the outer ones don't swing in under it.
    model_rig = rig()
    co, tris, _parts = back_sheet(depth=0.45, shape=lambda x: 0.20 * (x / 0.15) ** 2)
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_columns(chains)
    back = back_columns("u bottom", chains)
    if back is None:
        return
    middle = back[1]
    check(drift(middle) <= hairrig.CHAIN_SPACING / 2, "u bottom: the middle chain hangs straight down ({:.3f})".format(
        drift(middle)))
    check(not any_crossed(back), "u bottom: no neighbors cross")
    apart = [(abs(float(chain.tip[0] - middle.tip[0])), abs(float(chain.joints[0][0] - middle.joints[0][0])))
             for chain in (back[0], back[2])]
    check(all(tip >= 0.5 * root for tip, root in apart),
          "u bottom: each outer tip at least half its root gap from the middle's tip, along X ({})".format(
              [(round(tip, 3), round(root, 3)) for tip, root in apart]))


def test_mirrored_sheet():
    model_rig = rig()
    co, tris, _parts = one_piece_long_hair()
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_columns(chains)
    back = back_columns("mirrored", chains)
    if back is None:
        return
    left, middle, right = back
    check(abs(float(middle.joints[0][0])) <= 0.005 and abs(float(middle.tip[0])) <= 0.02,
          "mirrored: the middle chain straight behind (first joint X {:.3f}, tip X {:.3f})".format(
              float(middle.joints[0][0]), float(middle.tip[0])))
    pairs = ((left.joints[0], right.joints[0]), (left.tip, right.tip))
    check(all(abs(float(a[0] + b[0])) <= 0.02 and abs(float(a[1] - b[1])) <= 0.02 and abs(float(a[2] - b[2])) <= 0.02
              for a, b in pairs),
          "mirrored: the outer chains mirror each other within 2 cm (roots {}, {}; tips {}, {})".format(
              np.round(left.joints[0], 3), np.round(right.joints[0], 3), np.round(left.tip, 3), np.round(right.tip, 3)))


def test_back_width():
    # A sheet whose roots span 21 cm around the back, 50 cm wide at the bottom: still 3 back chains; a strip 3 cm
    # wide, 1.
    model_rig = rig()
    co, tris, _parts = back_sheet(half_width=0.25)
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_columns(chains)
    back = back_columns("back width", chains)
    check(back is not None and not any_crossed(back), "back width: none crossed")
    co, tris = wavy_strip()
    chains = hairrig.plan_rig(co, tris, model_rig)
    check(len(chains) == 1, "back width: a 3 cm strip gives 1 chain ({:d})".format(len(chains)))


def test_swept_sheet():
    # The sheet's bottom swept 12 cm to the side: the column chains lean along, none across another.
    model_rig = rig()
    co, tris, _parts = back_sheet(sweep=0.12)
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_columns(chains)
    back = back_columns("swept", chains)
    check(back is not None and not any_crossed(back), "swept: none crossed (roots at X {})".format(
        [round(float(chain.joints[0][0]), 3) for chain in back] if back is not None else None))


def test_ponytail_on_sheet():
    # A ponytail tube hanging over the back sheet: a separate piece, its own chain, and the sheet's columns.
    model_rig = rig()
    co, tris, _parts = back_sheet()
    tube_co, tube_tris, (tube_ids,) = join([hair.tube((0.0, 0.12, 1.50), (0.0, 0.22, 1.10), 0.035)])
    tube_ids = tube_ids + len(co)
    co, tris = np.vstack((co, tube_co)), np.vstack((tris, tube_tris + len(co)))
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_columns(chains)
    tails = [chain for chain in chains if chain.kind == "ponytail"]
    check(len(tails) == 1 and tails[0].sheet == -1 and np.isin(tails[0].vertices, tube_ids).all(),
          "ponytail on sheet: one ponytail chain, a separate piece of the tube only ({})".format(
              [(chain.kind, chain.sheet) for chain in tails]))
    back = back_columns("ponytail on sheet", chains)
    check(back is not None and len(chains) == 4, "ponytail on sheet: ...and the sheet's 3 ({:d} chains)".format(
        len(chains)))


def test_side_columns():
    # On each side three strips and a narrow stray in front of them, mirrored: at the slider's minimum spacing,
    # each side is cut into two columns, the frontmost with the stray and the first strip.
    model_rig = rig()
    plus = [side_strip(142.0, 0.22, 0.01)] + [side_strip(azimuth, 0.30, 0.04) for azimuth in (128.0, 113.0, 99.0)]
    minus = [(verts * np.array((-1.0, 1.0, 1.0)), [tuple(reversed(quad)) for quad in quads]) for verts, quads in plus]
    co, tris, ranges = join(plus + minus)
    chains = hairrig.plan_rig(co, tris, model_rig, spacing=0.02)
    describe_columns(chains)
    sheets = columns_by_sheet(chains)
    sides = {}
    for sheet, mine in sheets.items():
        sides[np.sign(float(mine[0].joints[0][0]))] = mine
    check(len(chains) == 4 and len(sheets) == 2 and set(sides) == {-1.0, 1.0}
          and all(len(mine) == 2 and all(chain.kind == "side" for chain in mine) for mine in sides.values()),
          "side columns: each side one sheet of 2 side chains ({})".format(
              [(chain.kind, chain.sheet, chain.order) for chain in chains]))
    if set(sides) != {-1.0, 1.0} or any(len(mine) != 2 for mine in sides.values()):
        return
    for sign, strips in ((1.0, ranges[:4]), (-1.0, ranges[4:])):
        held = [[int(np.isin(chain.vertices, strip).sum()) for strip in strips] for chain in sides[sign]]
        check(all(count > 0 for count in held[0][:2]) and not any(held[0][2:])
              and all(count > 0 for count in held[1][2:]) and not any(held[1][:2])
              and all(sum(row) == len(chain.vertices) for row, chain in zip(held, sides[sign])),
              "side columns: on the {:s} side the front chain holds the stray and the first strip, the other the "
              "last two (vertices of each: {})".format("+X" if sign > 0 else "-X", held))
    pairs = [(a.joints[0], b.joints[0]) for a, b in zip(sides[1.0], sides[-1.0])]
    check(all(abs(float(a[0] + b[0])) <= 0.02 and abs(float(a[1] - b[1])) <= 0.02 and abs(float(a[2] - b[2])) <= 0.02
              for a, b in pairs),
          "side columns: the two sides' first joints mirror within 2 cm ({})".format(
              [(np.round(a, 3).tolist(), np.round(b, 3).tolist()) for a, b in pairs]))


def test_forward_strand():
    # Back hair, a side strip in front of the ear and a strand rooted behind the ear that falls forward over the
    # shoulder onto the chest (40 degrees from the front): the strand is side hair, the side sheet's rear column, and
    # the back's 3 columns keep to the back hair. As back hair it would join the outer back column and pull that
    # column's chain off the back.
    model_rig = rig()
    co, tris, _parts = back_sheet()
    a, p = math.radians(45.0), math.radians(95.0)
    top = hair._onto_head(np.array(((math.sin(p) * math.sin(a), math.sin(p) * math.cos(a), math.cos(p)),)), 0.01)[0][0]
    head = np.asarray(model_rig.by_name[hairing.HEAD].head, dtype=np.float64) + model_rig.offset
    ahead = math.radians(40.0)
    bottom = np.array((head[0] + 0.14 * math.sin(ahead), head[1] - 0.14 * math.cos(ahead), top[2] - 0.38))
    strand = hair.strip(top, bottom, 0.03, np.array((math.cos(a), -math.sin(a), 0.0)))
    extra_co, extra_tris, (front_ids, strand_ids) = join([side_strip(118.0, 0.30, 0.04), strand])
    front_ids, strand_ids = front_ids + len(co), strand_ids + len(co)
    co, tris = np.vstack((co, extra_co)), np.vstack((tris, extra_tris + len(co)))
    locks = hairrig.find_locks(co, tris, model_rig)
    mine = [lock for lock in locks if np.isin(lock.vertices, strand_ids).any()]
    ends = [(round(hairrig._angle(model_rig, lock.root), 1), round(hairrig._angle(model_rig, hairrig.spine(co, lock)[-1]),
                                                                    1)) for lock in mine]
    check(len(mine) == 1 and np.isin(mine[0].vertices, strand_ids).all() and ends[0][0] > hairrig.SIDE_ANGLE
          and ends[0][1] <= hairrig.SIDE_ANGLE,
          "forward strand: the strand is one lock, rooted behind SIDE_ANGLE and ending in front of it (root and tip "
          "angles {})".format(ends))
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_columns(chains)
    held = [chain for chain in chains if np.isin(chain.vertices, strand_ids).any()]
    side = columns_by_sheet(chains).get(held[0].sheet, []) if len(held) == 1 and held[0].sheet >= 0 else []
    check(len(held) == 1 and held[0].kind == "side" and np.isin(held[0].vertices, strand_ids).all()
          and len(side) == 2 and side[1] is held[0] and np.isin(side[0].vertices, front_ids).all(),
          "forward strand: the strand is side hair, its side sheet's rear column behind the front strip ({})".format(
              [(chain.kind, chain.sheet, chain.order, int(np.isin(chain.vertices, strand_ids).sum())) for chain in chains]))
    back = back_columns("forward strand", chains)
    far = [round(off_hair(chain, co), 3) for chain in back] if back is not None else None
    check(back is not None and not any(np.isin(chain.vertices, strand_ids).any() for chain in back)
          and max(far) <= hairrig.SPINE_REACH,
          "forward strand: the back columns hold none of the strand and keep to their hair ({} m off)".format(far))
    check(len(chains) == 5, "forward strand: 5 chains, 3 back and 2 side ({:d})".format(len(chains)))


def test_light_column():
    # Two flat strips standing up from the crown: two light locks, one column, and a column's chain isn't light.
    model_rig = rig(MIQOTE)
    crown = hair.top(MIQOTE)
    co, tris, _ranges = join([hair.cap(race=MIQOTE)] + [
        hair.strip((x, 0.01, crown + 0.01), (x, 0.03, crown + 0.11), 0.02, (1.0, 0.0, 0.0)) for x in (-0.012, 0.012)])
    locks = hairrig.find_locks(co, tris, model_rig)
    check(len(locks) == 2 and all(lock.light for lock in locks), "light column: two light locks ({})".format(
        [lock.light for lock in locks]))
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_columns(chains)
    check(len(chains) == 1 and not chains[0].light, "light column: one chain, not light ({})".format(
        [(chain.kind, chain.light) for chain in chains]))


def test_budget():
    # The sheet's 3 column chains take 14 bones. With room for 10, the bones get longer: still 3 chains, each
    # with the same hair as unbudgeted.
    model_rig = rig()
    co, tris, _parts = one_piece_long_hair()
    budget = 10
    free = hairrig.plan_rig(co, tris, model_rig)
    chains = hairrig.plan_rig(co, tris, model_rig, max_bones=budget)
    describe_columns(chains)
    check(len(free) == 3 and sum(len(chain.joints) for chain in free) == 14,
          "budget: unbudgeted, 3 chains of 14 bones ({})".format([len(chain.joints) for chain in free]))
    back = back_columns("budget", chains)
    check(back is not None and len(chains) == 3, "budget: ...and no other chain ({:d})".format(len(chains)))
    check(chains and sum(len(chain.joints) for chain in chains) <= budget
          and all(len(chain.joints) >= 2 for chain in chains),
          "budget: at most {:d} bones, at least 2 per chain ({})".format(budget, [len(chain.joints) for chain in chains]))
    check(len(chains) == len(free) and all(np.array_equal(np.sort(a.vertices), np.sort(b.vertices))
                                           for a, b in zip(chains, free)),
          "budget: each chain keeps its unbudgeted twin's vertices")
    check(abs(sum(chain.area for chain in chains) - sum(chain.area for chain in free)) <= 1e-9,
          "budget: the chains keep the hair's area")
    keys = [(round(chain.angle, 6), chain.joints[0][0]) for chain in chains]
    check(keys == sorted(keys), "budget: the chains stay in name order")
    sideways = [max(abs(float(chain.joints[0][0] - end[0])) for end in (chain.joints[-1], chain.tip))
                for chain in chains]
    check(chains and all(side <= 2.0 * hairrig.LOCK_WIDTH for side in sideways),
          "budget: no chain runs across the sheet (first joint to last joint and tip {} m sideways)".format(
              [round(side, 3) for side in sideways]))


def test_lone_kind_survives():
    # A bun on the crown (a light back chain, the only one of its kind) and two ponytails: over the budget the
    # bones get longer until every chain has 2 (6 bones), then one ponytail merges into the other, and the bun
    # keeps its chain.
    model_rig = rig(MIQOTE)
    crown = hair.top(MIQOTE)
    co, tris, (_cap, bun_ids, _left, _right) = join(
        [hair.cap(race=MIQOTE), hair.tube((0.0, 0.0, crown + 0.01), (0.0, 0.0, crown + 0.11), 0.03)]
        + [hair.tube((sign * 0.08, 0.09, 1.50), (sign * 0.10, 0.14, 1.10), 0.035) for sign in (-1.0, 1.0)])
    free = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(free, co, [])
    shape = sorted((chain.kind, len(chain.joints)) for chain in free)
    check(shape == [("back", 2), ("ponytail", 5), ("ponytail", 5)]
          and all(chain.light for chain in free if chain.kind == "back"),
          "lone kind: unbudgeted, two ponytails of 5 bones and the bun's light back chain of 2 ({})".format(
              [(chain.kind, len(chain.joints), chain.light) for chain in free]))
    chains = hairrig.plan_rig(co, tris, model_rig, max_bones=5)
    describe_chains(chains, co, [])
    bun = [chain for chain in chains if np.isin(chain.vertices, bun_ids).any()]
    check(len(chains) == 2 and sum(len(chain.joints) for chain in chains) <= 5 and len(bun) == 1
          and np.isin(bun[0].vertices, bun_ids).all() and bun[0].light,
          "lone kind: with room for 5 bones, 2 chains, the bun's light and only the bun ({})".format(
              [(chain.kind, len(chain.joints), chain.light) for chain in chains]))


def card_hair(count, seed=0):
    """`hair.cards` as one mesh: (points, triangles)."""
    co, tris, _ranges = join(hair.cards(count, seed=seed))
    return co, tris


def test_card_hair_kinds():
    # Card hair: 300 cards, each its own loose part kept whole (a single strand), 4 bones each: flat, so a sheet
    # cut into columns, slices of the hair around the back, not ponytails.
    model_rig = rig()
    co, tris = card_hair(300)
    spaced = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(spaced, co, [])
    far = [off_hair(chain, co) for chain in spaced]
    check(spaced and sum(len(chain.joints) for chain in spaced) <= hairrig.MAX_HAIR_BONES
          and max(far) <= hairrig.SPINE_REACH,
          "card hair: spaced out, within the budget and on the hair ({:d} chains, up to {:.4f} m off; {})".format(
              len(spaced), max(far, default=math.nan), [chain.kind for chain in spaced]))
    # Their kinds: side hair toward the ears, back hair behind, no bangs. A card is flat, never a ponytail
    # (PONYTAIL_ROUND).
    card = len(card_hair(1)[0])
    kinds = [(chain.kind, round(chain.angle, 1), len(chain.vertices)) for chain in spaced]
    print("      spaced kinds (kind, angle, vertices; {:d} per card):".format(card), kinds)
    check(spaced and all((kind == "side") == (angle <= hairrig.SIDE_ANGLE) for kind, angle, _count in kinds)
          and all(kind in ("side", "back") for kind, _angle, _count in kinds),
          "card hair: spaced out, side hair to the side and back hair behind ({})".format(kinds))
    behind = [chain for chain in spaced if chain.angle > hairrig.SIDE_ANGLE]
    check(behind and all(chain.kind == "back" for chain in behind),
          "card hair: every chain behind the head is back hair ({})".format([chain.kind for chain in behind]))
    # At the slider's minimum spacing, narrower columns: every chain still on the hair.
    dense = hairrig.plan_rig(co, tris, model_rig, spacing=0.02)
    far = [off_hair(chain, co) for chain in dense]
    check(dense and sum(len(chain.joints) for chain in dense) <= hairrig.MAX_HAIR_BONES
          and max(far) <= hairrig.SPINE_REACH,
          "card hair: at spacing 0.02 within the budget and on the hair ({:d} chains, up to {:.4f} m off)".format(
              len(dense), max(far, default=math.nan)))
    # A ponytail in two layers, a tube in a tube (two loose parts, each kept whole, their roots twins): merged
    # unbudgeted, it is still one ponytail.
    co, tris, _ranges = join([hair.cap(), hair.tube((0.0, 0.105, 1.53), (0.0, 0.16, 1.11), 0.035),
                              hair.tube((0.0, 0.105, 1.53), (0.0, 0.16, 1.11), 0.028)])
    free = hairrig.plan_rig(co, tris, model_rig)
    chains = hairrig.plan_rig(co, tris, model_rig, max_bones=5)
    describe_chains(chains, co, [])
    check(len(free) == 1 and free[0].kind == "ponytail" and free[0].whole and len(chains) == 1
          and chains[0].kind == "ponytail" and chains[0].whole,
          "card hair: a ponytail in two layers merged stays one ponytail ({}, {})".format(
              [chain.kind for chain in free], [chain.kind for chain in chains]))


def test_card_hair_time():
    # Card hair of 3000 cards (66,000 vertices): the cards merge one at a time into a few column locks, each merge
    # measuring only the locks it merges, not the whole hair.
    model_rig = rig()
    co, tris = card_hair(3000)
    started = time.perf_counter()
    chains = hairrig.plan_rig(co, tris, model_rig)
    took = time.perf_counter() - started
    print("      plan_rig: {:.3f} s for 3000 cards, {:d} vertices".format(took, len(co)))
    check(chains and sum(len(chain.joints) for chain in chains) <= hairrig.MAX_HAIR_BONES,
          "card hair time: the cards fit the budget ({:d} chains)".format(len(chains)))
    check(took <= 2.0 * CARDS_TIME + 1.0, "card hair time: at most twice as long as before the columns, plus a "
          "second ({:.2f} s, {:.3f} s before)".format(took, CARDS_TIME))


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
# What hangs (by the scalp) and where locks start

def root_gaps(chains, co, tris, model_rig):
    """How far each chain's first joint is from the scalp, beyond the hair's cap depth."""
    cap = hairing.cap_depth(co, tris, model_rig)
    return [float(model_rig.scalp_gap(chain.joints[:1], cap)[0]) for chain in chains]


def root_gaps_by_kind(locks, co, tris, model_rig):
    """The root gaps (from the scalp, beyond the hair's cap depth) of the locks (of `find_locks`): (natural, others),
    natural being those rooted next to held hair (exempt from ROOT_REACH), the others rooted where they come
    closest to the scalp (within ROOT_REACH)."""
    cap = hairing.cap_depth(co, tris, model_rig)
    weld, count = fitting.weld_groups(co, fitting.WELD_DISTANCE)
    held = np.ones(count, dtype=bool)
    held[weld] = ~hairrig._hanging(model_rig.scalp(co), model_rig.head_distance(co), cap)
    t = weld[np.asarray(tris, dtype=np.int64)]
    edges = np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))
    next_to_held = np.zeros(count, dtype=bool)
    next_to_held[edges[held[edges[:, 1]], 0]] = True
    next_to_held[edges[held[edges[:, 0]], 1]] = True
    natural, others = [], []
    for lock in locks:
        own = ~lock.joined
        roots = weld[lock.vertices[own][lock.distance[own] <= 1e-9]]
        gap = float(model_rig.scalp_gap(lock.root[None], cap)[0])
        (natural if (next_to_held[roots] & ~held[roots]).any() else others).append(gap)
    return natural, others


def flat_fringe():
    """The scalp cap on c0801 and a fringe lying flat on the forehead 1.2 cm off it, its own loose part: from
    high on the front of the head (35 degrees from the top) down to the brow (115 degrees), 30 degrees wide.
    Returns (points, triangles, {"cap": indices, "fringe": indices})."""
    co, tris, (cap_ids, fringe_ids) = join([hair.cap(race=MIQOTE),
                                            hair.patch(np.arange(35.0, 116.0, 5.0), np.arange(165.0, 196.0, 7.5),
                                                       0.012, MIQOTE)])
    return co, tris, {"cap": cap_ids, "fringe": fringe_ids}


def neck_hair(loose=False, bottom=NECK_BOTTOM, rows=30):
    """The scalp cap on c0801 and a sheet sharing its lowest ring behind the head (60 degrees to either side),
    hanging down the neck 1 cm off it to ``bottom``; with ``loose``, also the sheet's lower half as its own
    loose part 5 mm inside it, an inner layer lying wholly on the neck. Returns (points, triangles, {"cap",
    "sheet", "layer": indices})."""
    co, tris, ranges = join(neck_parts(loose, bottom, rows))
    return co, tris, dict(zip(("cap", "sheet", "layer"), ranges))


def neck_parts(loose=False, bottom=NECK_BOTTOM, rows=30):
    """The parts of `neck_hair`, (vertices, quads) each."""
    cap, faces, ring, azimuth = back_ring(24, MIQOTE)
    tops = cap[ring[np.abs(azimuth[ring]) <= 60.0 + 1e-6]]
    verts, quads = hair.down_neck(tops, bottom, rows, 0.01, MIQOTE)
    parts = [(cap, faces), (verts, quads)]
    if loose:
        start = rows // 2 * len(tops)
        layer = verts[start:].copy()
        inward = layer[:, :2] - rig(MIQOTE).by_name[hairing.NECK].head[:2]
        layer[:, :2] -= 0.005 * inward / np.linalg.norm(inward, axis=1)[:, None]
        parts.append((layer, [tuple(k - start for k in quad) for quad in quads if min(quad) >= start]))
    return parts


def bun_hair():
    """The scalp cap on c0801 and a bun: a tube 3 cm wide standing 10 cm up from the crown, its own loose
    part. Returns (points, triangles, {"cap": indices, "bun": indices})."""
    crown = hair.top(MIQOTE)
    co, tris, (cap_ids, bun_ids) = join([hair.cap(race=MIQOTE),
                                         hair.tube((0.0, 0.0, crown + 0.01), (0.0, 0.0, crown + 0.11), 0.03)])
    return co, tris, {"cap": cap_ids, "bun": bun_ids}


def test_flat_fringe():
    model_rig = rig(MIQOTE)
    co, tris, parts = flat_fringe()
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(chains, co, hairrig.find_locks(co, tris, model_rig))
    gaps = root_gaps(chains, co, tris, model_rig)
    fringe = [(chain, gap) for chain, gap in zip(chains, gaps) if np.isin(chain.vertices, parts["fringe"]).all()]
    check(any(chain.kind == "bangs" for chain, _gap in fringe),
          "flat fringe: a fringe lying 1.2 cm off the forehead gets a bangs chain ({})".format(
              [chain.kind for chain in chains]))
    check(fringe and all(gap <= hairrig.ROOT_REACH for _chain, gap in fringe),
          "flat fringe: ...rooted at the hairline ({} m from the scalp)".format([round(gap, 4) for _c, gap in fringe]))
    check(len(fringe) == len(chains) and all(not chain.light for chain in chains),
          "flat fringe: no other chain, none light ({})".format([(chain.kind, chain.light) for chain in chains]))


def test_neck_layer():
    model_rig = rig(MIQOTE)
    co, tris, parts = neck_hair()
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(chains, co, hairrig.find_locks(co, tris, model_rig))
    gaps = root_gaps(chains, co, tris, model_rig)
    back = [(chain, gap) for chain, gap in zip(chains, gaps) if chain.kind == "back"]
    check(len(back) >= 1 and len(back) == len(chains) and all(gap <= hairrig.ROOT_REACH for _chain, gap in back),
          "neck layer: back chains start at the scalp ({})".format(
              [(chain.kind, round(gap, 4)) for chain, gap in zip(chains, gaps)]))
    heights = [float(chain.joints[0][2]) for chain in chains]
    check(chains and min(heights) >= 1.36, "neck layer: ...at the nape hairline, not low on the neck (z {})".format(
        [round(z, 3) for z in heights]))


def test_neck_only_layer():
    model_rig = rig(MIQOTE)
    co, tris, parts = neck_hair(loose=True)
    layer = parts["layer"]
    print("      the loose layer is {:.3f} m from the scalp at its closest".format(
        float(model_rig.scalp_gap(co[layer], hairing.cap_depth(co, tris, model_rig)).min())))
    locks = hairrig.find_locks(co, tris, model_rig)
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(chains, co, locks)
    without = hairrig.plan_rig(*neck_hair()[:2], model_rig)
    check(chains and len(chains) == len(without) and all(
        np.allclose(a.joints[0], b.joints[0]) for a, b in zip(chains, without)),
        "neck only layer: no chain of its own, the chains start where they do without it ({:d}, {:d})".format(
            len(chains), len(without)))
    holding = [lock for lock in locks if np.isin(lock.vertices, layer).any()]
    joined = all(lock.joined[np.isin(lock.vertices, layer)].all() for lock in holding)
    kinds = [None if hairrig.plan_chain(co, lock) is None else hairrig.plan_chain(co, lock).kind for lock in holding]
    held = sum(int(np.isin(lock.vertices, layer).sum()) for lock in holding)
    check(holding and held == len(layer) and joined and all(kind == "back" for kind in kinds),
          "neck only layer: joins a back chain's lock ({:d} of {:d} vertices, joined {!s}, {})".format(
              held, len(layer), joined, kinds))


def test_cap_rim():
    model_rig = rig(MIQOTE)
    co, tris, _ranges = join([hair.cap(race=MIQOTE)])
    check(not hairrig.find_locks(co, tris, model_rig) and not hairrig.plan_rig(co, tris, model_rig),
          "cap rim: a cap alone has no lock and no chain")
    # The cap's rim behind the head reaching just below the nape hairline: it hangs, but too little for a chain.
    cap, faces, ring, azimuth = back_ring(24, MIQOTE)
    tops = cap[ring[np.abs(azimuth[ring]) <= 90.0 + 1e-6]]
    co, tris, (_cap_ids, rim_ids) = join([(cap, faces), hair.down_neck(tops, 1.41, 7, 0.01, MIQOTE)])
    rim = rim_ids[-len(tops):]
    share = model_rig.scalp(co[rim])
    print("      the rim's lowest row: z {:.3f}, scalp share {:.2f} to {:.2f}".format(
        float(co[rim, 2].mean()), float(share.min()), float(share.max())))
    locks = hairrig.find_locks(co, tris, model_rig)
    describe(locks, co, model_rig)
    with_chain = [lock for lock in locks if hairrig.plan_chain(co, lock) is not None]
    check(share.min() < hairing.SCALP_HALF and not hairrig.plan_rig(co, tris, model_rig)
          and not any(np.isin(lock.vertices, rim).any() for lock in with_chain),
          "cap rim: a rim reaching below the hairline gets no chain ({:d} locks)".format(len(locks)))


def test_bun():
    model_rig = rig(MIQOTE)
    co, tris, parts = bun_hair()
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(chains, co, hairrig.find_locks(co, tris, model_rig))
    gaps = root_gaps(chains, co, tris, model_rig)
    check(len(chains) == 1 and chains[0].light and np.isin(chains[0].vertices, parts["bun"]).all(),
          "bun: the bun gets one light chain ({}, {} m from the scalp)".format(
              [(chain.kind, chain.light) for chain in chains], [round(gap, 4) for gap in gaps]))
    # Rooted just behind the head's axis and leaning forward over it, the bun's tip lies in front of SIDE_ANGLE: it
    # stands up, so it stays back hair.
    head = np.asarray(model_rig.by_name[hairing.HEAD].head, dtype=np.float64) + model_rig.offset
    crown = hair.top(MIQOTE, 0.0, head[1] + 0.035)
    co, tris, _ranges = join([hair.cap(race=MIQOTE), hair.tube((0.0, head[1] + 0.035, crown + 0.01),
                                                               (0.0, head[1] - 0.03, crown + 0.11), 0.02,
                                                               bend=(0.0, 0.0, 0.0))])
    chains = hairrig.plan_rig(co, tris, model_rig)
    ends = [(round(chain.angle, 1), round(hairrig._angle(model_rig, chain.tip), 1)) for chain in chains]
    check(chains and all(chain.kind == "back" for chain in chains)
          and any(root > hairrig.SIDE_ANGLE and tip <= hairrig.SIDE_ANGLE for root, tip in ends),
          "bun: leaning forward over the head's axis, rooted behind SIDE_ANGLE and its tip in front, it stays back "
          "hair ({}; root and tip angles {})".format([chain.kind for chain in chains], ends))


def test_roots_near_scalp():
    model_rig = rig()
    co, tris, _part, _names = long_hair()
    one_co, one_tris, _parts = one_piece_long_hair()
    card_co, card_tris = card_hair(300)
    strands = join([hair.strip((x, 0.08, 1.53), (x, 0.10, 1.23), 0.02, (1.0, 0.0, 0.0))
                    for x in (-0.12, -0.06, 0.0, 0.06, 0.12)])[:2]
    for name, points, triangles in (("long hair", co, tris), ("one piece", one_co, one_tris),
                                    ("card hair", card_co, card_tris), ("strands", *strands)):
        chains = hairrig.plan_rig(points, triangles, model_rig)
        natural, others = root_gaps_by_kind(hairrig.find_locks(points, triangles, model_rig), points, triangles,
                                            model_rig)
        check(chains and all(gap <= hairrig.ROOT_REACH for gap in others),
              "roots near the scalp: {:s}: locks not rooted next to held hair within ROOT_REACH ({}; rooted next "
              "to held hair: {:d}, up to {:.4f} m)".format(name, [round(gap, 4) for gap in others], len(natural),
                                                           max(natural, default=math.nan)))


def ring_strip(azimuths, bottom, outward=0.0):
    """A strip hanging down the neck (`hair.down_neck`) from the c0801 cap's lowest ring at ``azimuths``
    (degrees, the ring's vertices nearest them) to ``bottom``, moved ``outward`` (metres) away from the neck's
    axis: (vertices, quads)."""
    cap, _faces, ring, azimuth = back_ring(24, MIQOTE)
    tops = cap[[ring[np.argmin(np.abs(azimuth[ring] - a))] for a in azimuths]]
    verts, quads = hair.down_neck(tops, bottom, 20, 0.01, MIQOTE)
    out = verts[:, :2] - rig(MIQOTE).by_name[hairing.NECK].head[:2]
    verts[:, :2] += outward * out / np.linalg.norm(out, axis=1)[:, None]
    return verts, quads


def off_hair(chain, co):
    """How far a chain's joints and tip lie from its hair at most (metres)."""
    own = co[chain.vertices]
    return max(float(np.min(np.linalg.norm(own - point, axis=1))) for point in np.vstack((chain.joints, chain.tip)))


def test_twins_merge():
    # The neck sheet (two back locks), a strip down the side doubled 0.3 mm apart (twin layers, each its own
    # loose part) and a shorter strip on the other side: the twins merge, one chain holds both layers, and the
    # short strip joins a column rather than keeping a chain of its own.
    model_rig = rig(MIQOTE)
    parts = neck_parts() + [ring_strip((75.0, 90.0), 1.15), ring_strip((75.0, 90.0), 1.15, 0.0003),
                            ring_strip((-75.0, -90.0), 1.30)]
    co, tris, (_cap, _sheet, twin_ids, other_ids, short_ids) = join(parts)
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_columns(chains)
    both = [chain for chain in chains if np.isin(chain.vertices, twin_ids).any()
            and np.isin(chain.vertices, other_ids).any()]
    alone = [chain for chain in chains if np.isin(chain.vertices, short_ids).all()]
    check(len(chains) == 2 and len(both) == 1 and not alone,
          "twins: the twin layers in one chain, the short strip in no chain of its own ({:d} chains)".format(
              len(chains)))
    # A card doubled for its back faces, alone on the cap: one chain, not two.
    co, tris, _ranges = join(neck_parts()[:1] + [ring_strip((75.0, 90.0), 1.15),
                                                 ring_strip((75.0, 90.0), 1.15, 0.0003)])
    chains = hairrig.plan_rig(co, tris, model_rig)
    check(len(chains) == 1, "twins: a doubled card is one chain ({:d})".format(len(chains)))


def test_merge_stays_on_hair():
    # Three round tubes behind the head, separate pieces: a narrow one straight down, one rooted 3 cm from it
    # swinging out to the side, and one rooted 5 cm from it on the other side hanging alongside it. A round
    # tube's root follows it out to the side, so the swinging one lies farther by root (8.7 cm, 7 mm beyond the
    # spacing) than the one alongside (5.5 cm): spaced out, the narrow tube merges with that one. (No merge here
    # is refused; test_hollow_ponytail's second half has one.)
    model_rig = rig()
    co, tris, (narrow, swing, beside) = join([
        hair.tube((0.0, 0.08, 1.53), (0.0, 0.10, 1.33), 0.035 * 0.86),
        hair.tube((-0.03, 0.08, 1.53), (-0.22, 0.10, 1.33), 0.035),
        hair.tube((0.05, 0.08, 1.53), (0.05, 0.10, 1.33), 0.035)])
    locks = hairrig.find_locks(co, tris, model_rig)
    check(len(locks) == 3 and all(lock.whole and hairrig._round(co, lock) for lock in locks),
          "on the hair: three round locks ({})".format([(lock.whole, hairrig._round(co, lock)) for lock in locks]))
    free = hairrig.plan_rig(co, tris, model_rig, spacing=0.01)
    describe_chains(free, co, locks)
    mine = {name: [chain for chain in free if np.isin(chain.vertices, ids).all()]
            for name, ids in (("narrow", narrow), ("swing", swing), ("beside", beside))}
    ready = len(free) == 3 and all(len(v) == 1 for v in mine.values())
    gaps = {name: float(np.linalg.norm(mine[name][0].joints[0] - mine["narrow"][0].joints[0])) if ready else math.nan
            for name in ("swing", "beside")}
    check(ready and all(chain.kind == "back" and chain.sheet == -1 for chain in free)
          and min(free, key=lambda chain: chain.area) is mine["narrow"][0]
          and sorted(len(chain.joints) for chain in free) == [2, 2, 3],
          "on the hair: at spacing 0.01 three back pieces, the narrow the smallest, 3, 2 and 2 bones ({}, {}, "
          "{} m by root)".format([chain.kind for chain in free], [len(chain.joints) for chain in free],
                                 {name: round(gap, 4) for name, gap in gaps.items()}))
    # With room for 5 bones the bones get longer until every chain has 2 (6 bones), then the narrow tube, the
    # smallest, merges with its nearest by root, the one alongside.
    chains = hairrig.plan_rig(co, tris, model_rig, spacing=0.01, max_bones=5)
    describe_chains(chains, co, locks)
    merged = [chain for chain in chains if np.isin(chain.vertices, narrow).any()]
    far = [off_hair(chain, co) for chain in chains]
    check(len(chains) == 2 and sum(len(chain.joints) for chain in chains) <= 5 and len(merged) == 1
          and np.isin(merged[0].vertices, beside).any() and not np.isin(merged[0].vertices, swing).any()
          and max(far) <= hairrig.SPINE_REACH,
          "on the hair: with room for 5 bones the narrow tube merges with the one alongside, the swinging one stays "
          "apart, every chain on the hair ({}, {} m)".format([len(chain.joints) for chain in chains],
                                                             [round(f, 4) for f in far]))
    spaced = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(spaced, co, locks)
    merged = [chain for chain in spaced if np.isin(chain.vertices, narrow).any()]
    far = [off_hair(chain, co) for chain in spaced]
    check(len(spaced) == 2 and len(merged) == 1 and np.isin(merged[0].vertices, beside).any()
          and not np.isin(merged[0].vertices, swing).any() and max(far) <= hairrig.SPINE_REACH,
          "on the hair: spaced out, the narrow tube merges with the one alongside, the swinging one stays apart, "
          "every chain on the hair ({:d} chains, {} m)".format(len(spaced), [round(f, 4) for f in far]))


def hollow_tail(count=12):
    """A hollow ponytail of ``count`` 1.2 cm strips, each its own loose part, in a cone behind the head: their
    tops on a 1.2 cm ring, their bottoms 41 cm lower on a 5 cm ring. (points, triangles, each strip's indices)."""
    strips = []
    for k in range(count):
        a = 2.0 * math.pi * k / count
        ring = np.array((math.cos(a), math.sin(a), 0.0))
        side = np.array((-math.sin(a), math.cos(a), 0.0))
        strips.append(hair.strip(np.array((0.0, 0.10, 1.53)) + 0.012 * ring,
                                 np.array((0.0, 0.17, 1.12)) + 0.05 * ring, 0.012, side, rows=20))
    return join(strips)


def hollow_tubes(count=6):
    """A hollow ponytail of ``count`` round tubes (`hair.tube`, 0.035), each its own loose part, in a cone behind
    the head: their tops on a 4.5 cm ring, their bottoms 41 cm lower on an 8 cm ring. (points, triangles, each
    tube's indices)."""
    tubes = []
    for k in range(count):
        a = 2.0 * math.pi * k / count
        ring = np.array((math.cos(a), math.sin(a), 0.0))
        tubes.append(hair.tube(np.array((0.0, 0.10, 1.53)) + 0.045 * ring, np.array((0.0, 0.17, 1.12)) + 0.080 * ring,
                               0.035))
    return join(tubes)


def test_hollow_ponytail():
    # Merged from all its strips, the tail's chain runs down its hollow middle, up to 5 cm from the nearest
    # strip: farther than SPINE_REACH, but the strips enclose it. One chain, at the default spacing too.
    model_rig = rig()
    co, tris, ranges = hollow_tail()
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(chains, co, [])
    check(len(chains) == 1 and all(np.isin(strip, chains[0].vertices).any() for strip in ranges)
          and off_hair(chains[0], co) > hairrig.SPINE_REACH,
          "hollow ponytail: one chain down the middle of all its strips ({:d} chains, {} m off the hair)".format(
              len(chains), [round(off_hair(chain, co), 4) for chain in chains]))
    # Round tubes are separate pieces, merged by the spacing only where the merge stays on the hair: the tubes
    # enclose their middle.
    co, tris, ranges = hollow_tubes()
    chains = hairrig.plan_rig(co, tris, model_rig)
    describe_chains(chains, co, [])
    check(len(chains) == 1 and all(np.isin(tube, chains[0].vertices).any() for tube in ranges),
          "hollow ponytail: six round tubes in a ring make one chain down their middle ({:d} chains, {} m off the "
          "hair)".format(len(chains), [round(off_hair(chain, co), 4) for chain in chains]))
    old = hairrig.ENCLOSED_GAP
    try:
        hairrig.ENCLOSED_GAP = 0.0
        apart = hairrig.plan_rig(co, tris, model_rig)
    finally:
        hairrig.ENCLOSED_GAP = old
    check(len(apart) >= 2, "hollow ponytail: ...where only SPINE_REACH would keep it in {:d} chains".format(len(apart)))


def test_maeve_chains():
    """The real test hair: a chain for the fringe, every chain rooted at the scalp, none on the neck or face."""
    if not os.path.isfile(MAEVE):
        print("skip: no Maeve hair at " + MAEVE)
        return
    import hair_models
    with open(MAEVE, "rb") as handle:
        points, tris, _held = hair_models.decode(handle.read())
    maeve = rig(MIQOTE)
    cap = hairing.cap_depth(points, tris, maeve)
    started = time.perf_counter()
    chains = hairrig.plan_rig(points, tris, maeve, cap=cap)
    took = time.perf_counter() - started
    top = float(maeve.head_field.surface[:, 2].max())
    gaps = [float(gap) for gap in maeve.scalp_gap(np.array([chain.joints[0] for chain in chains]).reshape(-1, 3), cap)]
    print("      plan_rig: {:.3f} s for {:d} vertices; cap depth {:.4f} m, head top z {:.4f}".format(
        took, len(points), cap, top))
    chain_table(chains, gaps)
    fringe = [chain for chain in chains if chain.angle <= hairrig.BANGS_ANGLE
              and top - 0.20 <= chain.joints[0][2] <= top - 0.08]
    check(fringe, "maeve: the fringe gets a chain, rooted in front on the forehead ({:d})".format(len(fringe)))
    natural, others = root_gaps_by_kind(hairrig.find_locks(points, tris, maeve, cap=cap), points, tris, maeve)
    check(chains and all(gap <= hairrig.ROOT_REACH for gap in others),
          "maeve: every lock not rooted next to held hair within ROOT_REACH of the scalp ({:d}, up to {:.4f} m; "
          "{:d} rooted next to held hair, up to {:.4f} m)".format(len(others), max(others, default=math.nan),
                                                                  len(natural), max(natural, default=math.nan)))
    lowest = min(float(chain.joints[0][2]) for chain in chains) if chains else math.nan
    check(chains and lowest >= 1.36, "maeve: no chain starts below z 1.36 ({:.3f})".format(lowest))
    # Spaced out (CHAIN_SPACING) and on the hair: no chain runs through the hair from one lock to another.
    far = [off_hair(chain, points) for chain in chains]
    check(chains and max(far) <= hairrig.SPINE_REACH,
          "maeve: every chain's joints and tip within SPINE_REACH of its hair (up to {:.4f} m)".format(
              max(far, default=math.nan)))
    check(any(chain.kind == "bangs" for chain in chains), "maeve: a bangs chain ({})".format(
        [chain.kind for chain in chains]))
    print("      {:d} chains, {:d} bones".format(len(chains), sum(len(chain.joints) for chain in chains)))
    # The back is one sheet cut into 3 columns, the middle one straight down behind, and its flat centre cards
    # are back hair, not a ponytail swinging on its own.
    back = back_columns("maeve", chains)
    check(back is not None and not any_crossed(back) and drift(back[1]) <= hairrig.CHAIN_SPACING / 2,
          "maeve: none crossed, the middle chain hangs straight down ({})".format(
              "drift {:.3f}, X {:.3f} to {:.3f}".format(drift(back[1]), float(back[1].joints[0][0]),
                                                       float(back[1].tip[0])) if back is not None else None))
    sides = [len(mine) for mine in columns_by_sheet(chains).values() if mine[0].kind == "side"]
    check(all(count <= 2 for count in sides), "maeve: at most 2 chains per side sheet ({})".format(sides))
    check(not any(chain.kind == "ponytail" for chain in chains), "maeve: no ponytail")
    check(not any(chain.light for chain in chains if chain.sheet >= 0), "maeve: no column chain is light")
    # A lock rooted by the ear that falls back over the shoulder is back hair: as side hair its own chain would
    # swing a piece of the back.
    tips = [round(hairrig._angle(maeve, chain.tip), 1) for chain in chains if chain.kind == "side"]
    check(all(angle <= hairrig.SIDE_ANGLE for angle in tips), "maeve: every side chain ends to the side ({})".format(tips))


def chain_table(chains, gaps):
    """Print ``chains`` as a table, with ``gaps`` (each first joint's distance from the scalp)."""
    print("      chain  kind      light  sheet  order  bones  root x   root y   root z   gap      angle  length")
    for number, (chain, gap) in enumerate(zip(chains, gaps)):
        line = np.vstack((chain.joints, chain.tip))
        print("      {:02d}     {:9s} {!s:6} {:5d}  {:5d}  {:d}      {:7.4f}  {:7.4f}  {:7.4f}  {:.4f}  {:5.1f}  "
              "{:.3f}".format(number + 1, chain.kind, chain.light, chain.sheet, chain.order, len(chain.joints),
                              *chain.joints[0], gap, chain.angle,
                              float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())))


def test_briar_chains():
    """Another real hair: its back one sheet of 3 columns, at most 2 chains per side, within the budget."""
    if not os.path.isfile(BRIAR):
        print("skip: no Briar hair at " + BRIAR)
        return
    import hair_models
    with open(BRIAR, "rb") as handle:
        points, tris, _held = hair_models.decode(handle.read())
    briar = rig(MIQOTE)
    cap = hairing.cap_depth(points, tris, briar)
    started = time.perf_counter()
    chains = hairrig.plan_rig(points, tris, briar, cap=cap)
    took = time.perf_counter() - started
    gaps = [float(gap) for gap in briar.scalp_gap(np.array([chain.joints[0] for chain in chains]).reshape(-1, 3), cap)]
    print("      plan_rig: {:.3f} s for {:d} vertices; cap depth {:.4f} m".format(took, len(points), cap))
    chain_table(chains, gaps)
    bones = sum(len(chain.joints) for chain in chains)
    print("      {:d} chains, {:d} bones".format(len(chains), bones))
    back = back_columns("briar", chains)
    check(back is not None and not any_crossed(back), "briar: none of the back chains crossed")
    sides = [len(mine) for mine in columns_by_sheet(chains).values() if mine[0].kind == "side"]
    check(all(count <= 2 for count in sides), "briar: at most 2 chains per side sheet ({})".format(sides))
    check(bones <= hairrig.MAX_HAIR_BONES, "briar: the bones fit the budget ({:d})".format(bones))


def test_maeve_weights():
    """The real test hair's weights from its own chains: the fringe hangs, the cap above it stays."""
    if not os.path.isfile(MAEVE):
        print("skip: no Maeve hair at " + MAEVE)
        return
    import hair_models
    with open(MAEVE, "rb") as handle:
        points, tris, _held = hair_models.decode(handle.read())
    maeve = rig(MIQOTE)
    cap = hairing.cap_depth(points, tris, maeve)
    chains = hairrig.plan_rig(points, tris, maeve, cap=cap)
    custom = own_rig(maeve, chains, cap)
    started = time.perf_counter()
    names, weights = hairing.hair_weights(points, tris, custom)
    took = time.perf_counter() - started
    own = own_weight(names, weights)
    top = float(maeve.head_field.surface[:, 2].max())
    below = top - points[:, 2]
    front = (np.abs(points[:, 0]) < 0.05) & (points[:, 1] < maeve.by_name[hairing.HEAD].head[1] - 0.04)
    print("      hair_weights: {:.3f} s for {:d} vertices, {:d} chains ({:d} light)".format(
        took, len(points), len(chains), sum(chain.light for chain in chains)))
    print("      front rows (cm below the head's top): vertices, mean weight on the chains, on j_kao")
    for start in range(0, 24, 2):
        row = front & (below >= start / 100.0) & (below < (start + 2) / 100.0)
        if row.any():
            print("      {:2d}-{:2d}  {:5d}  {:.3f}  {:.3f}".format(start, start + 2, int(row.sum()), float(own[row].mean()),
                                                                   float(weights[row, names.index(hairing.HEAD)].mean())))
    # Past the ramp of the nearest bangs chain: more than half the ramp below its first joint.
    bangs = [chain for chain in hairing.chains_of(custom.hair_bones())
             if chains[hairrig._parse(chain.root)[0] - 1].kind == "bangs"]
    model = custom.model(points)
    nearest = np.argmin(np.column_stack([chain.project(model)[0] for chain in bangs]), axis=1) if bangs else None
    past = np.zeros(len(points), dtype=bool)
    for c, chain in enumerate(bangs):
        past |= (nearest == c) & (chain.along(model) > 0.5 * hairing.RAMP_SHARE * first_bone(chain))
    past &= front
    print("      {:d} bangs chain(s); {:d} front vertices past the ramp".format(len(bangs), int(past.sum())))
    check(past.any() and float(own[past].mean()) >= 0.7,
          "maeve weights: the fringe past its bangs chain's ramp hangs from the chains ({:.3f} on average)".format(
              float(own[past].mean()) if past.any() else math.nan))
    fringe = front & (below >= 0.12) & (below <= 0.16)
    cap_rows = front & (below >= 0.0) & (below <= 0.08)
    check(fringe.any() and float(own[fringe].mean()) >= 0.4,
          "maeve weights: the fringe 12-16 cm below the top hangs from the chains ({:.3f} on average)".format(
              float(own[fringe].mean()) if fringe.any() else math.nan))
    check(cap_rows.any() and float(own[cap_rows].mean()) <= 0.05,
          "maeve weights: the front 0-8 cm below the top stays with the head ({:.3f} on the chains on average)".format(
              float(own[cap_rows].mean()) if cap_rows.any() else math.nan))


# -----------------------------------------------------------------------------
# The rig on the armature

def rig_scene(fixture=None):
    """The long hair (`hair.build`) deformed by a new armature with c0201's j_kao, j_kubi and j_sebo_c (made by
    `hairing.ensure_bones`), and its chains built on the armature with the hair's cap depth, as Hair Weights
    builds them: (hair object, armature, plans, bone names, points, triangles, part index of each point, part
    names). With ``fixture`` (a c0801 fixture: `bun_hair`, `flat_fringe` or `neck_hair`), that hair instead,
    on c0801's bones."""
    hair.clear_scene()
    race = hair.RACE if fixture is None else MIQOTE
    if fixture is not None:
        co, tris, ids = fixture()
        names = list(ids)
        part = np.zeros(len(co), dtype=np.int64)
        for index, name in enumerate(names):
            part[ids[name]] = index
        mesh = bpy.data.meshes.new(fixture.__name__)
        mesh.from_pydata([tuple(p) for p in co], [], [tuple(int(k) for k in t) for t in tris])
        obj = bpy.data.objects.new(fixture.__name__, mesh)
        bpy.context.scene.collection.objects.link(obj)
    else:
        co, tris, part, names = long_hair()
        obj, _ranges = hair.build()
    armature = hairing.new_armature(bpy.context, "Armature", bpy.context.scene.collection)
    hairing.ensure_bones(bpy.context, armature, rig(race), [hairing.HEAD, hairing.NECK, hairing.UPPER_BACK])
    hairing.add_armature_modifier(obj, armature)
    cap = hairing.cap_depth(co, tris, rig(race))
    plans = hairrig.plan_rig(co, tris, rig(race), cap=cap)
    bones = hairrig.build_armature_rig(bpy.context, armature, plans, race, cap_depth=cap)
    return obj, armature, plans, bones, co, tris, part, names


def world(armature, local):
    return np.array(armature.matrix_world @ Vector(local))


def rig_json(armature):
    return json.loads(armature[hairrig.RIG_PROPERTY])


def state_chains(armature):
    """(kind, bone names) of each chain in the armature's rig state."""
    return [(chain.kind, chain.bones.split(",")) for chain in armature.magic_fit_hair_rig.chains]


def chain_json(kind, bones, sheet, order):
    """A chain as the rig property lists it: its sheet and order only when it is a column's (sheet >= 0)."""
    return dict({"kind": kind, "bones": bones}, **({"sheet": sheet, "order": order} if sheet >= 0 else {}))


def json_matches_state(armature):
    state = armature.magic_fit_hair_rig
    return rig_json(armature) == {"race": state.race, "version": 2, "chains": [
        chain_json(chain.kind, chain.bones.split(","), chain.sheet, chain.order) for chain in state.chains]}


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
    check(rig_json(armature) == {"race": hair.RACE, "version": 2, "chains": [
        chain_json(plan.kind, [hairrig.bone_name(c, b) for b in range(len(plan.joints))], plan.sheet, plan.order)
        for c, plan in enumerate(plans)]}, "armature rig: the rig property lists the race, kinds, bones and columns")
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
    hangs = hanging(co, tris, custom)
    hair_columns = [column[name] for name in bones]
    for name in ("lock_l", "lock_r", "ponytail"):
        index = names.index(name)
        mine = [c for c, plan in enumerate(plans) if main_part(plan, part, area) == index]
        if len(mine) != 1:
            check(False, "follow locks: {:s} has one chain ({})".format(name, mine))
            continue
        own = [column[hairrig.bone_name(mine[0], b)] for b in range(len(plans[mine[0]].joints))]
        vertices = np.flatnonzero((part == index) & hangs)
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


def collection_bones(armature, name):
    """The names of the bones in ``armature``'s bone collection ``name`` (None when it has none), in Object
    Mode."""
    collection = armature.data.collections_all.get(name)
    return None if collection is None else sorted(bone.name for bone in collection.bones)


def test_chain_collection():
    _obj, armature, plans, bones, _co, _tris, _part, _names = rig_scene()
    check(collection_bones(armature, hairrig.CHAIN_COLLECTION) == sorted(bones),
          "chain collection: after the build, exactly the hair bones are in it ({})".format(
              collection_bones(armature, hairrig.CHAIN_COLLECTION)))
    # The body's bones in a collection of their own, as an imported skeleton has them: built again, the hair
    # bones are in Hair Chains, and the body's collection keeps its bones.
    body = armature.data.collections.new("Body")
    for name in (hairing.HEAD, hairing.NECK):
        body.assign(armature.data.bones[name])
    bones = hairrig.build_armature_rig(bpy.context, armature, plans, hair.RACE)
    body_bones = collection_bones(armature, "Body")
    check(collection_bones(armature, hairrig.CHAIN_COLLECTION) == sorted(bones)
          and [name for name in body_bones if not name.startswith(hairrig.BONE_PREFIX)]
          == sorted((hairing.HEAD, hairing.NECK)),
          "chain collection: built again, the hair bones are in it, Body keeps its bones ({})".format(body_bones))
    # A bone the user adds below chain 01, and chain 02's first bone renamed to something else.
    first = state_chains(armature)[0][1]
    second = state_chains(armature)[1][1][0]
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    added = edit_bones.new("Strand")
    added.head = edit_bones[first[-1]].tail
    added.tail = added.head - Vector((0.0, 0.0, 0.05))
    added.parent = edit_bones[first[-1]]
    edit_bones[second].name = "My Bone"
    hairrig.sync_rig(bpy.context, armature, hair.RACE)
    bpy.ops.object.mode_set(mode='OBJECT')
    have = collection_bones(armature, hairrig.CHAIN_COLLECTION)
    hair_bones = sorted(bone.name for bone in armature.data.bones if bone.name.startswith(hairrig.BONE_PREFIX))
    new_name = state_chains(armature)[0][1][-1]
    check(have == hair_bones and new_name not in first and new_name in have and "My Bone" not in have,
          "chain collection: sync_rig adds the user's new chain bone ({:s}) and drops the renamed one".format(
              new_name))
    check(collection_bones(armature, "Body") == body_bones, "chain collection: sync_rig leaves Body as it was")


def test_bones_hash():
    _obj, armature, _plans, bones, _co, _tris, _part, _names = rig_scene()
    check(armature.magic_fit_hair_rig.bones_hash == hairrig.bones_hash(armature) and not hairrig.rig_edited(armature),
          "bones hash: the state keeps the hash of the bones built")
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')
    check(not hairrig.rig_edited(armature), "bones hash: Edit Mode in and out changes nothing")
    for change in ("roll", "tail", "parent"):
        bpy.ops.object.mode_set(mode='EDIT')
        edit = armature.data.edit_bones[bones[1]]
        if change == "roll":
            edit.roll += 0.1
        elif change == "tail":
            edit.tail = edit.tail + Vector((0.0, 0.0, 0.001))
        else:
            edit.use_connect = False
            edit.parent = armature.data.edit_bones[hairing.HEAD]
        bpy.ops.object.mode_set(mode='OBJECT')
        check(hairrig.rig_edited(armature), "bones hash: a changed {:s} counts as an edit".format(change))
        hairrig.sync_rig(bpy.context, armature, hair.RACE)
        check(not hairrig.rig_edited(armature), "bones hash: ...until the rig is read again")
    armature.magic_fit_hair_rig.bones_hash = ""
    check(hairrig.rig_edited(armature), "bones hash: no hash counts as edited")


def test_kind_change_writes_json():
    _obj, armature, _plans, _bones, _co, _tris, _part, _names = rig_scene()
    chains = armature.magic_fit_hair_rig.chains
    check(len(chains) > 1 and chains[1].kind != "back", "kind change: chain 02 isn't back hair yet")
    chains[1].kind = "back"
    check(rig_json(armature)["chains"][1]["kind"] == "back" and json_matches_state(armature),
          "kind change: the rig property follows")


def test_state_keeps_light():
    _obj, armature, plans, _bones, co, tris, _part, _names = rig_scene(bun_hair)
    state = armature.magic_fit_hair_rig
    cap = hairing.cap_depth(co, tris, rig(MIQOTE))
    first = hairrig.bone_name(0, 0)

    def kept(when):
        custom = hairrig.custom_rig(armature, MIQOTE)
        check(abs(state.cap_depth - cap) < 1e-6 and cap > 0.0 and [chain.light for chain in state.chains] == [True],
              "light: the state keeps the cap depth and the bun's light chain, {:s} ({:.4f}, {})".format(
                  when, state.cap_depth, [chain.light for chain in state.chains]))
        check(custom.light == {first} and custom.cap_depth is not None and abs(custom.cap_depth - cap) < 1e-6
              and custom.moved(np.zeros(3)).light == {first} and custom.moved(np.zeros(3)).cap_depth == custom.cap_depth,
              "light: custom_rig carries them, also moved, {:s} ({}, {})".format(when, custom.light, custom.cap_depth))

    check(len(plans) == 1 and plans[0].light, "light: the bun's one chain is planned light")
    kept("built")
    check(json.dumps(rig_json(armature)).find("light") < 0 and json_matches_state(armature),
          "light: the rig property doesn't list it")
    hairrig.sync_rig(bpy.context, armature, MIQOTE)
    kept("after sync_rig")
    # A chain the user adds (its first bone named for the hair) is not light; the bun's stays light.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    added = edit_bones.new("j_ex_hair_new")
    added.head, added.tail = (0.0, 0.10, 1.55), (0.0, 0.12, 1.42)
    added.parent = edit_bones[hairing.HEAD]
    hairrig.sync_rig(bpy.context, armature, MIQOTE)
    check([chain.light for chain in state.chains] == [True, False] and abs(state.cap_depth - cap) < 1e-6,
          "light: a chain the user adds isn't light ({})".format([chain.light for chain in state.chains]))
    # The bun's chain deleted and a new one added in its place: the new one takes its number, not its light.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    for edit in [edit for edit in edit_bones if edit.name.startswith(hairrig.BONE_PREFIX)]:
        edit_bones.remove(edit)
    added = edit_bones.new("j_ex_hair_new")
    added.head, added.tail = (0.0, 0.10, 1.55), (0.0, 0.12, 1.42)
    added.parent = edit_bones[hairing.HEAD]
    hairrig.sync_rig(bpy.context, armature, MIQOTE)
    check(state_chains(armature) == [("back", [first])] and [chain.light for chain in state.chains] == [False],
          "light: a new chain taking a deleted light chain's number isn't light ({}, {})".format(
              state_chains(armature), [chain.light for chain in state.chains]))
    # Built again over the hair (a part joining), keeping the kinds.
    again = hairrig.keep_kinds(armature, hairrig.plan_rig(co, tris, rig(MIQOTE), cap=cap))
    hairrig.build_armature_rig(bpy.context, armature, again, MIQOTE, cap_depth=cap)
    kept("after the rebuild")


def state_columns(armature):
    """{first bone's name: (sheet, order)} of each chain in the armature's rig state."""
    return {chain.bones.split(",")[0]: (chain.sheet, chain.order) for chain in armature.magic_fit_hair_rig.chains}


def json_columns(armature):
    """{first bone's name: (sheet, order)} of the chains the rig property lists with a sheet."""
    return {chain["bones"][0]: (chain["sheet"], chain["order"]) for chain in rig_json(armature)["chains"]
            if "sheet" in chain or "order" in chain}


def test_state_keeps_columns():
    _obj, armature, plans, _bones, co, tris, _part, _names = rig_scene()
    describe_columns(plans)
    planned = {hairrig.bone_name(c, 0): (plan.sheet, plan.order) for c, plan in enumerate(plans)}
    columns = {name: column for name, column in planned.items() if column[0] >= 0}
    check(columns and any(plan.kind == "ponytail" and (plan.sheet, plan.order) == (-1, -1) for plan in plans),
          "columns: the long hair has column chains and a ponytail without a sheet ({})".format(planned))
    check(state_columns(armature) == planned, "columns: each chain keeps its plan's sheet and order ({})".format(
        state_columns(armature)))
    check(json_columns(armature) == columns and json_matches_state(armature),
          "columns: the rig property lists sheet and order for exactly the column chains ({})".format(
              json_columns(armature)))
    hairrig.sync_rig(bpy.context, armature, hair.RACE)
    check(state_columns(armature) == planned and json_columns(armature) == columns and json_matches_state(armature),
          "columns: sync_rig changes nothing ({})".format(state_columns(armature)))
    # A column chain's kind set in the list: the rig property is written again, its sheet and order still there.
    state = armature.magic_fit_hair_rig
    index = next(i for i, chain in enumerate(state.chains) if chain.sheet >= 0)
    state.chains[index].kind = "side" if state.chains[index].kind != "side" else "back"
    check(rig_json(armature)["chains"][index]["kind"] == state.chains[index].kind
          and json_columns(armature) == columns and json_matches_state(armature),
          "columns: a kind changed in the list keeps the sheets and orders in the rig property")
    # A chain the user adds in Edit Mode: no sheet; the others keep theirs.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    added = edit_bones.new("j_ex_hair_new")
    added.head, added.tail = (0.0, 0.10, 1.55), (0.0, 0.12, 1.42)
    added.parent = edit_bones[hairing.HEAD]
    hairrig.sync_rig(bpy.context, armature, hair.RACE)
    new_name = hairrig.bone_name(len(plans), 0)
    check(state_columns(armature) == dict(planned, **{new_name: (-1, -1)}) and json_columns(armature) == columns
          and json_matches_state(armature),
          "columns: a chain the user adds has no sheet, the others keep theirs ({})".format(state_columns(armature)))
    # One column chain's bones deleted: the others keep their sheets and orders, a gap in the orders included.
    # The long hair's side locks are a sheet each: put them in one sheet, orders 0 and 1, and delete order 0.
    state = armature.magic_fit_hair_rig
    for order, name in enumerate(sorted(columns)):
        chain = next(chain for chain in state.chains if chain.bones.split(",")[0] == name)
        chain.sheet, chain.order = 0, order
    hairrig.write_rig_property(armature)
    gone = sorted(columns)[0]
    before = state_columns(armature)
    check(len(columns) >= 2 and json_matches_state(armature), "columns: {:d} column chains in one sheet ({})".format(
        len(columns), before))
    gone_bones = next(chain.bones.split(",") for chain in armature.magic_fit_hair_rig.chains
                      if chain.bones.split(",")[0] == gone)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    for name in gone_bones:
        edit_bones.remove(edit_bones[name])
    hairrig.sync_rig(bpy.context, armature, hair.RACE)
    left = {name: column for name, column in before.items() if name != gone}
    check(state_columns(armature) == left and json_columns(armature) == {
        name: column for name, column in left.items() if column[0] >= 0} and json_matches_state(armature),
        "columns: with chain {:s} deleted the others keep their sheets and orders ({})".format(
            gone, state_columns(armature)))
    # Built again over the hair: the new plans' sheets and orders.
    again = hairrig.keep_kinds(armature, hairrig.plan_rig(co, tris, rig(), cap=hairing.cap_depth(co, tris, rig())))
    hairrig.build_armature_rig(bpy.context, armature, again, hair.RACE)
    check(state_columns(armature) == {hairrig.bone_name(c, 0): (plan.sheet, plan.order) for c, plan in enumerate(again)}
          and json_matches_state(armature), "columns: built again, the chains take the new plans' sheets and orders")


def test_old_rig_state():
    _obj, armature, plans, _bones, _co, _tris, _part, _names = rig_scene()
    # Built without a cap depth (as before it was kept).
    hairrig.build_armature_rig(bpy.context, armature, plans, hair.RACE)
    state = armature.magic_fit_hair_rig
    check(state.cap_depth == -1.0, "old rig: built without a cap depth, the state has none ({})".format(
        state.cap_depth))
    # A rig state from a file made before the light flags.
    for chain in state.chains:
        chain.light = False
    custom = hairrig.custom_rig(armature, hair.RACE)
    check(custom.cap_depth is None and not custom.light, "old rig: unknown cap, nothing light ({}, {})".format(
        custom.cap_depth, custom.light))
    hairrig.sync_rig(bpy.context, armature, hair.RACE)
    check(state.cap_depth == -1.0 and not any(chain.light for chain in state.chains),
          "old rig: ...also after sync_rig")
    # A rig state from a file made before the columns: version 2, no chain lists a sheet or order.
    for chain in state.chains:
        chain.sheet = chain.order = -1
    hairrig.write_rig_property(armature)

    def no_columns(when):
        data = rig_json(armature)
        check(data["version"] == 2 and not any("sheet" in chain or "order" in chain for chain in data["chains"])
              and json_matches_state(armature), "old rig: no sheets or orders in the rig property, {:s}".format(when))

    no_columns("written")
    hairrig.sync_rig(bpy.context, armature, hair.RACE)
    no_columns("after sync_rig")
    # A rig from another tool (no state, only the property): nothing is light.
    state.chains.clear()
    hairrig.sync_rig(bpy.context, armature, hair.RACE)
    check(len(state.chains) == len(plans) and not any(chain.light for chain in state.chains),
          "old rig: a rig read from the property has no light chain")


# -----------------------------------------------------------------------------
# Weights from the chains

def body_weight(names, weights):
    """Each vertex's weight on the bones what stays with the body goes to: j_kao, j_kubi and j_sebo_c."""
    return weights[:, [names.index(name) for name in (hairing.HEAD, hairing.NECK, hairing.UPPER_BACK)]].sum(1)


def own_weight(names, weights, chain=None):
    """Each vertex's weight on the hair's own bones (BONE_PREFIX), or on those of chain number ``chain`` (from 1)."""
    prefix = hairrig.BONE_PREFIX if chain is None else "{:s}{:02d}_".format(hairrig.BONE_PREFIX, chain)
    return weights[:, [k for k, name in enumerate(names) if name.startswith(prefix)]].sum(1)


def chain_along(custom, co, number=None):
    """How far down a chain of ``custom`` each of ``co`` is (`hairing.Chain.along`): down its nearest chain, or
    down chain ``number`` (from 1); and that chain's first bone's length."""
    chains = hairing.chains_of(custom.hair_bones())
    model = custom.model(co)
    if number is not None:
        chain = next(chain for chain in chains if chain.root == hairrig.bone_name(number - 1, 0))
        return chain.along(model), np.full(len(co), first_bone(chain))
    nearest = np.argmin(np.column_stack([chain.project(model)[0] for chain in chains]), axis=1)
    along = np.column_stack([chain.along(model) for chain in chains])[np.arange(len(co)), nearest]
    return along, np.array([first_bone(chain) for chain in chains])[nearest]


def first_bone(chain):
    return float(chain.seg[0]) if len(chain.seg) else hairing.BONE_LENGTH


def own_rig(base, plans, cap):
    """The hair's own skeleton of ``plans`` on ``base`` (c0801's body, not moved) as `hairrig.custom_rig` reads it
    from the bones `hairrig.build_armature_rig` makes, without Blender: a bone per joint, its X axis to the next
    joint (the last one's to the tip), each hanging from the one before, the first from j_kao."""
    bones, skeleton, light = list(base.bones), [], set()
    for c, plan in enumerate(plans):
        parent = hairing.HEAD
        for b, (joint, end) in enumerate(zip(plan.joints, list(plan.joints[1:]) + [plan.tip])):
            rest = np.eye(4)
            rest[:3, 0] = (end - joint) / np.linalg.norm(end - joint)
            rest[:3, 3] = joint
            name = hairrig.bone_name(c, b)
            bones.append(hairing.Bone(name, parent, rest))
            skeleton.append(name)
            parent = name
        if plan.light:
            light.add(hairrig.bone_name(c, 0))
    return hairing.HairRig(base.race, 0, bones, base.head_field, skeleton, custom=True, cap_depth=cap, light=light)


def test_chain_along():
    rest = np.eye(4)
    rest[:3, 0] = (0.0, 0.0, -1.0)
    rest[:3, 3] = (0.0, 0.0, 1.5)
    second = rest.copy()
    second[:3, 3] = (0.0, 0.0, 1.4)
    chain = hairing.Chain([hairing.Bone("a", None, rest), hairing.Bone("b", "a", second)])
    points = np.array(((0.01, 0.0, 1.53), (0.01, 0.0, 1.45), (0.0, 0.01, 1.35), (0.0, 0.0, 1.2)))
    along = chain.along(points)
    _distance, down = chain.project(points)
    check(np.allclose(along, (-0.03, 0.05, 0.15, 0.30)) and np.allclose(down[1:], along[1:]) and down[0] == 0.0,
          "chain along: down the chain as project measures it, negative above its first joint ({})".format(
              np.round(along, 4).tolist()))


def test_ramp_from_first_joint():
    _obj, armature, plans, _bones, co, tris, part, names = rig_scene(flat_fringe)
    custom = hairrig.custom_rig(armature, MIQOTE)
    # The rule itself, without holding layers together (which smooths the ramp over about LAYER_SMOOTH).
    bone_names, weights = hairing.hair_weights(co, tris, custom, layer_range=0.0)
    along, first = chain_along(custom, co)
    # The ramp is centred on the first joint, RAMP_SHARE of the first bone wide.
    half = 0.5 * hairing.RAMP_SHARE * first
    fringe = part == names.index("fringe")
    above = fringe & (along < -half)
    past = fringe & (along > half)
    body, own = body_weight(bone_names, weights), own_weight(bone_names, weights)
    print("      {:d} chains; fringe: {:d} vertices above the ramp, {:d} past it".format(
        len(plans), int(above.sum()), int(past.sum())))
    check(above.any() and float(body[above].min()) >= 0.95,
          "ramp: the fringe above its chain's ramp stays on j_kao ({:.3f} at least)".format(
              float(body[above].min()) if above.any() else math.nan))
    check(past.any() and float(own[past].min()) >= 0.95,
          "ramp: the fringe past the ramp hangs from the chain ({:.3f} at least)".format(
              float(own[past].min()) if past.any() else math.nan))
    _names, planned = hairing.hair_weights(co, tris, own_rig(rig(MIQOTE), plans, armature.magic_fit_hair_rig.cap_depth),
                                           layer_range=0.0)
    # The armature keeps its bones in single precision.
    check(_names == bone_names and float(np.abs(planned - weights).max()) < 1e-4,
          "ramp: the plans' rig without Blender weights as the armature's ({:.2e})".format(
              float(np.abs(planned - weights).max()) if _names == bone_names else math.inf))


def test_cap_stays():
    for fixture in (flat_fringe, neck_hair, bun_hair):
        _obj, armature, _plans, _bones, co, tris, part, names = rig_scene(fixture)
        custom = hairrig.custom_rig(armature, MIQOTE)
        # The rule itself: holding layers together gives the cap under the bun's foot some of the bun's weights.
        bone_names, weights = hairing.hair_weights(co, tris, custom, layer_range=0.0)
        held = (part == names.index("cap")) & ~hairrig._hanging(custom.scalp(co), custom.head_distance(co),
                                                               custom.cap_depth)
        own = own_weight(bone_names, weights)
        # The c0801 cap covers the cat ears' roots: Ears gives them to the ear bones.
        ears = weights[:, [bone_names.index(name) for name in hairing.EARS]].sum(1)
        check(held.any() and float(own[held].max()) <= 1e-6
              and float((body_weight(bone_names, weights) + ears)[held].min()) >= 1.0 - 1e-6,
              "cap stays: {:s}: the held cap only on j_kao, the neck, upper back and ears ({:d} vertices, {:.2e} on "
              "the chains at most)".format(fixture.__name__, int(held.sum()),
                                           float(own[held].max()) if held.any() else math.nan))


def test_bun_light():
    _obj, armature, _plans, _bones, co, tris, part, _names = rig_scene(bun_hair)
    custom = hairrig.custom_rig(armature, MIQOTE)
    # The rule itself; holding layers together changes it a little.
    bone_names, weights = hairing.hair_weights(co, tris, custom, layer_range=0.0)
    bun = np.flatnonzero(part == 1)
    own = own_weight(bone_names, weights)
    top = bun[co[bun, 2] >= co[bun, 2].max() - 1e-6]
    held_names, held = hairing.hair_weights(co, tris, custom)
    print("      scalp share over the bun: {:.3f} to {:.3f}; with layers held, {:.4f} on the chain at most".format(
        float(custom.scalp(co[bun]).min()), float(custom.scalp(co[bun]).max()),
        float(own_weight(held_names, held)[bun].max())))
    check(custom.light and float(own[bun].max()) <= hairing.LIGHT_SHARE + 1e-6,
          "bun light: the bun's light chain takes at most a third ({:.4f})".format(float(own[bun].max())))
    check(float(own[top].min()) > 0.2, "bun light: ...and its top moves ({:.4f})".format(float(own[top].min())))


def test_own_neck_weight():
    _obj, armature, _plans, _bones, co, tris, _part, _names = rig_scene(neck_hair)
    custom = hairrig.custom_rig(armature, MIQOTE)
    held = ~hairrig._hanging(custom.scalp(co), custom.head_distance(co), custom.cap_depth)
    below = held & (co[:, 2] < custom.head_height(hairing.HEAD))
    kubi = {}
    for neck_weight in (0.3, 0.0):
        bone_names, weights = hairing.hair_weights(co, tris, custom, neck_weight=neck_weight)
        kubi[neck_weight] = weights[below, bone_names.index(hairing.NECK)]
    check(below.any() and float(kubi[0.3].mean()) > 0.02 and float(kubi[0.0].max()) == 0.0,
          "own neck weight: the held hair below j_kao gets j_kubi weight with Neck Weight 0.3 ({:.3f} on average, "
          "{:d} vertices), none with 0 ({:.3f})".format(float(kubi[0.3].mean()) if below.any() else math.nan,
                                                        int(below.sum()), float(kubi[0.0].max()) if below.any() else 0.0))


def test_ramp_follows_edited_bone():
    _obj, armature, _plans, _bones, co, tris, _part, _names = rig_scene(neck_hair)
    before = hairrig.custom_rig(armature, MIQOTE)
    old_names, old = hairing.hair_weights(co, tris, before, neck_weight=0.0, layer_range=0.0)
    old_along, first = chain_along(before, co, 1)
    # The vertices chain 01 takes (the sheet's other chain takes some of those near it).
    chains = hairing.chains_of(before.hair_bones())
    ones = hairing.chain_shares(before.model(co), before, chains)[:, [chain.root for chain in chains].index(
        hairrig.bone_name(0, 0))] >= 0.95
    # The first joint moved 3 cm down its bone.
    name = hairrig.bone_name(0, 0)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit = armature.data.edit_bones[name]
    edit.head = edit.head + (edit.tail - edit.head).normalized() * 0.03
    hairrig.sync_rig(bpy.context, armature, MIQOTE)
    after = hairrig.custom_rig(armature, MIQOTE)
    new_names, new = hairing.hair_weights(co, tris, after, neck_weight=0.0, layer_range=0.0)
    new_along, new_first = chain_along(after, co, 1)
    # Past the old first joint (half hanging or more), above the new one's ramp.
    between = ones & (old_along > 0.0) & (new_along < -0.5 * hairing.RAMP_SHARE * new_first)
    # Chain 01's own weight there: the moved joint also moves chain 01's share, so some of that hair goes to
    # the other chain, whose ramp didn't move.
    one_old, one_new = own_weight(old_names, old, 1), own_weight(new_names, new, 1)
    print("      first bone {:.3f} m; {:d} vertices between the old and the new joint".format(float(first[0]),
                                                                                       int(between.sum())))
    check(between.any() and float(one_new[between].max()) <= 1e-6 and float(one_old[between].mean()) >= 0.3,
          "edited bone: the hair between the old and the new first joint doesn't hang from chain 01 now ({:.3f} at "
          "most, {:.3f} on average before)".format(float(one_new[between].max()) if between.any() else math.nan,
                                                    float(one_old[between].mean()) if between.any() else math.nan))


def main():
    magic_fit.register()
    try:
        for test in (test_long_hair_locks, test_one_piece_long_hair, test_separate_strands, test_crumbs,
                     test_parted_locks, test_long_hair_chains, test_bangs_chain, test_pivot_at_root, test_even_joints,
                     test_wavy_lock_straight, test_kinds, test_columns, test_u_bottom, test_mirrored_sheet,
                     test_back_width, test_swept_sheet, test_ponytail_on_sheet, test_side_columns, test_forward_strand,
                     test_light_column,
                     test_budget, test_lone_kind_survives, test_card_hair_kinds, test_card_hair_time,
                     test_moved_hair,
                     test_flat_fringe, test_neck_layer, test_neck_only_layer, test_cap_rim, test_bun,
                     test_roots_near_scalp, test_twins_merge, test_merge_stays_on_hair, test_hollow_ponytail,
                     test_maeve_chains, test_briar_chains,
                     test_maeve_weights, test_armature_rig, test_weights_follow_locks,
                     test_edited_rig_is_kept, test_branch_starts_chain, test_chain_collection, test_bones_hash,
                     test_kind_change_writes_json,
                     test_state_keeps_light, test_state_keeps_columns, test_old_rig_state, test_chain_along,
                     test_ramp_from_first_joint,
                     test_cap_stays, test_bun_light, test_own_neck_weight, test_ramp_follows_edited_bone):
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
