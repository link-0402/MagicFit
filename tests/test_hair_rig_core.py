# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless tests for the hair rig Hair Weights builds for the hair itself (Own Skeleton): the hair's areas, the
chains planned in them (`hairlanes`: lanes, rows, kinds, the bone budget), the bones they make on the hair's armature,
and the chains the user adds and deletes.

Run from the repository root:
    blender -b --factory-startup --python tests/test_hair_rig_core.py
"""

import json
import math
import os
import sys
import time
import traceback
import warnings

import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import hair  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import hairing, hairlanes, hairrig  # noqa: E402

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
# The lock planner's plan_rig on 3000 cards before it cut sheets into columns, in seconds (measured 2026-11-09):
# the lanes (test_lanes_card_hair_time) may cost at most as much again, plus a second.
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
    """Which of hair ``co`` hang (go into chains), by the scalp and the hair's cap depth."""
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


def card_back(ranges, depth=0.40, lean=0.03):
    """Back hair in flat cards: the scalp cap, and for each (low, high) of ``ranges`` (degrees around the skull's
    middle, 0 straight behind, positive toward +X) a card hanging ``depth`` from the cap's lowest ring between
    those azimuths, straight down and ``lean`` out from the head's axis. Each card is a lock of its own. Returns
    (points, triangles)."""
    cap, faces, ring, azimuth = back_ring(48)
    axis = np.array((0.0, rig().by_name[hairing.HEAD].head[1]))
    bottom = cap[ring, 2].mean() - depth
    parts = [(cap, faces)]
    for low, high in ranges:
        tops = cap[ring[(azimuth[ring] >= low - 1e-6) & (azimuth[ring] <= high + 1e-6)]]
        out = tops[:, :2] - axis
        out /= np.linalg.norm(out, axis=1)[:, None]
        bottoms = np.column_stack((tops[:, :2] + out * lean, np.full(len(tops), bottom)))
        parts.append(sheet(tops, bottoms))
    co, tris, _ranges = join(parts)
    return co, tris


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


def main_part(chain, part, area):
    """The part holding most of a chain's area."""
    return int(np.argmax(np.bincount(part[chain.vertices], area[chain.vertices])))


def wavy_strip(amplitude=0.03, wavelength=0.30, rows=60):
    """A 3 cm strip hanging 40 cm down behind the head, waving ``amplitude`` (meters) from side to side once every
    ``wavelength`` down it, like a wavy ribbon card: (points, triangles)."""
    top, bottom = np.array((0.0, 0.08, 1.53)), np.array((0.0, 0.10, 1.13))
    verts = []
    for i in range(rows + 1):
        center = top + (bottom - top) * (i / rows)
        center[0] += amplitude * math.sin(2.0 * math.pi * float(np.linalg.norm(bottom - top)) * (i / rows) / wavelength)
        verts += [center - (0.015, 0.0, 0.0), center + (0.015, 0.0, 0.0)]
    co, tris, _ranges = join([(np.array(verts), [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(rows)])])
    return co, tris


def counts(back=2, ponytail=2, left=2, right=2, bangs=2):
    """The chain counts (`hairrig.ChainCounts`), the settings' defaults but those given."""
    return hairrig.ChainCounts(back, ponytail, left, right, bangs)


def kinds_of(chains):
    return [(chain.kind, chain.sheet, chain.order) for chain in chains]


def bangs_strip(x=0.0, width=0.04):
    """`bangs_hair`'s strip, moved ``x`` along X and ``width`` wide: (vertices, quads)."""
    return hair.strip((x, -0.10, 1.625), (x, -0.19, 1.54), width, (1.0, 0.0, 0.0))


def short_hair():
    """Short hair: the scalp cap, `bangs_hair`'s strip and five strips 12 cm long around the sides and the back:
    (points, triangles)."""
    parts = [hair.cap(), bangs_strip()]
    parts += [side_strip(azimuth, 0.12, 0.04) for azimuth in (130.0, -130.0, 60.0, -60.0, 0.0)]
    co, tris, _ranges = join(parts)
    return co, tris


def around(angle, z, gap=0.01):
    """The point ``gap`` off the head at height ``z`` in the direction ``angle`` (degrees around the head's vertical
    axis through j_kao, 0 straight in front, positive toward +X): bisection along the ray from the axis."""
    field = hair.head()
    a = math.radians(angle)
    start = np.array((0.0, rig().by_name[hairing.HEAD].head[1], z))
    direction = np.array((math.sin(a), -math.cos(a), 0.0))
    lo, hi = 0.0, 0.3
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if field.distance(start + direction * mid)[0] > gap:
            hi = mid
        else:
            lo = mid
    return start + direction * hi


def swung(root, swing, length=0.38, width=0.03, tube=False, rows=12, segments=8):
    """Hair rooted 1 cm off the head at ear height ``root`` degrees around the head's axis (as `around`), hanging
    ``length`` down: its top third there, moving out to 14 cm from the axis, its lower two thirds swung round to
    ``swing`` degrees. A flat strip ``width`` wide running around the head, or with ``tube`` a tube ``width``
    across. One loose part: (vertices, quads)."""
    top = around(root, hair.CENTER[2] - 0.01, 0.01 + (width / 2.0 if tube else 0.0))
    axis_y = rig().by_name[hairing.HEAD].head[1]
    reach = math.hypot(top[0], top[1] - axis_y)
    verts = []
    for i in range(rows + 1):
        a = math.radians(root if 3 * i <= rows else swing)
        r = reach + (0.14 - reach) * min(1.0, 3.0 * i / rows)
        c = np.array((r * math.sin(a), axis_y - r * math.cos(a), top[2] - length * i / rows))
        if tube:
            verts += [c + width / 2.0 * np.array((math.cos(b), math.sin(b), 0.0))
                      for b in np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)]
        else:
            side = np.array((math.cos(a), math.sin(a), 0.0))
            verts += [c - side * width / 2.0, c + side * width / 2.0]
    ring = segments if tube else 2
    quads = [(i * ring + j, i * ring + (j + 1) % ring, (i + 1) * ring + (j + 1) % ring, (i + 1) * ring + j)
             for i in range(rows) for j in range(ring if tube else 1)]
    return np.array(verts), quads


def wrapped_hair(depth=0.40):
    """Long hair in one piece round the back and both sides: the scalp cap, and a sheet hanging ``depth`` from its
    lowest ring 120 degrees to either side of straight behind (around the skull's middle, as `back_ring`), sharing
    the ring's vertices; its bottom row lies 12 cm from the head's axis, its ends fanned forward to 50 degrees from
    the front (as `around`). One loose part spread over the back and the sides (`one_piece_long_hair` is all back
    hair): (points, triangles, {"cap": indices, "sheet": indices})."""
    cap, faces, ring, azimuth = back_ring(48)
    keep = ring[np.abs(azimuth[ring]) <= 120.0 + 1e-6]
    tops = cap[keep]
    a = np.radians(np.where(azimuth[keep] >= 0.0, 1.0, -1.0) * (180.0 - 130.0 * np.abs(azimuth[keep]) / 120.0))
    axis_y = rig().by_name[hairing.HEAD].head[1]
    bottoms = np.column_stack((0.12 * np.sin(a), axis_y - 0.12 * np.cos(a),
                               np.full(len(tops), tops[:, 2].mean() - depth)))
    co, tris, (cap_ids, sheet_ids) = join([(cap, faces), sheet(tops, bottoms)])
    return co, tris, {"cap": cap_ids, "sheet": sheet_ids}


def front_lock(bottom):
    """A 3 cm strip hanging from 1 cm off the face 20 degrees from the front (as `around`) at height 1.57 down to
    height ``bottom``: (vertices, quads)."""
    top = around(20.0, 1.57)
    return hair.strip(top, np.array((top[0], top[1] - 0.02, bottom)), 0.03, np.array((1.0, 0.0, 0.0)))


def lanes_of(measured, label, ids):
    """The areas (`hairlanes.areas`'s ``label``) of those of vertices ``ids`` that hang, and their angles (degrees,
    as `around`)."""
    on = np.isin(measured.index, ids)
    return label[on], np.degrees(measured.phi[on])


def lane_share(measured, label, ids, *names):
    """The share (by area) of the hanging hair of vertices ``ids`` that is in one of the areas ``names``."""
    on = np.isin(measured.index, ids)
    return float(measured.area[on & np.isin(label, names)].sum() / max(measured.area[on].sum(), 1e-18))


def describe_lanes(name, measured, label, ids):
    mine, angles = lanes_of(measured, label, ids)
    shares = [round(lane_share(measured, label, ids, k), 3) for k in range(4)]
    print("      {:s}: {:d} hanging, angles {:.1f} to {:.1f}, back/left/right/bangs {}".format(
        name, len(mine), float(angles.min()) if len(mine) else 0.0, float(angles.max()) if len(mine) else 0.0, shares))


def test_lanes_areas_by_where_it_hangs():
    # Areas by where the hair hangs around the head: a strip 60 degrees from the front is side hair, left (+X) or
    # right by its side, all of it; a sheet down the back is back hair, a strip over the forehead bangs.
    model_rig = rig()
    co, tris, (_cap_ids, plus_ids, minus_ids) = join([hair.cap(), side_strip(120.0, 0.30, 0.04),
                                                       side_strip(-120.0, 0.30, 0.04)])
    measured = hairlanes.measure(co, tris, model_rig)
    label, _short = hairlanes.areas(measured)
    describe_lanes("+X strip", measured, label, plus_ids)
    describe_lanes("-X strip", measured, label, minus_ids)
    check(lane_share(measured, label, plus_ids, hairlanes.LEFT) == 1.0
          and lane_share(measured, label, minus_ids, hairlanes.RIGHT) == 1.0,
          "lanes areas: the +X side strip is left, the -X one right, all of each")
    # Hair whose armature has its head elsewhere: the same areas.
    offset = np.array((0.05, -0.1, 0.2))
    moved = hairlanes.measure(co + offset, tris, model_rig.moved(offset))
    moved_label, _short = hairlanes.areas(moved)
    check(np.array_equal(moved.index, measured.index) and np.array_equal(moved_label, label)
          and np.allclose(moved.at, measured.at),
          "lanes areas: hair moved with its head gets the same areas")
    co, tris, ids = back_sheet()
    measured = hairlanes.measure(co, tris, model_rig)
    label, _short = hairlanes.areas(measured)
    describe_lanes("back sheet", measured, label, ids["sheet"])
    check(lane_share(measured, label, ids["sheet"], hairlanes.BACK) == 1.0, "lanes areas: a back sheet is all back")
    co, tris, ids = bangs_hair()
    measured = hairlanes.measure(co, tris, model_rig)
    label, _short = hairlanes.areas(measured)
    describe_lanes("bangs", measured, label, ids["bangs"])
    check(lane_share(measured, label, ids["bangs"], hairlanes.BANGS) == 1.0, "lanes areas: a bangs strip is all bangs")
    co, tris, _part, _names = long_hair()
    _label, short = hairlanes.areas(hairlanes.measure(co, tris, model_rig))
    check(not short, "lanes areas: long hair isn't short")


def test_lanes_forward_card_is_side():
    # A strip rooted behind the ear (125 degrees from the front) whose lower two thirds fall forward over the
    # shoulder (to 60) is side hair, all of it; the same strip rooted in front of the ear (95) falling back (to 150)
    # is back hair, all of it.
    model_rig = rig()
    for root, swing, names, what in ((125.0, 60.0, (hairlanes.LEFT, hairlanes.RIGHT), "side"),
                                     (95.0, 150.0, (hairlanes.BACK,), "back")):
        co, tris, (_cap_ids, strip_ids) = join([hair.cap(), swung(root, swing)])
        measured = hairlanes.measure(co, tris, model_rig)
        label, _short = hairlanes.areas(measured)
        describe_lanes("strip {:.0f} to {:.0f}".format(root, swing), measured, label, strip_ids)
        mine, _angles = lanes_of(measured, label, strip_ids)
        check(len(mine) > 0 and len(set(mine.tolist())) == 1 and lane_share(measured, label, strip_ids, *names) == 1.0,
              "lanes forward card: rooted at {:.0f} and hanging at {:.0f}, it is {:s} hair, all of it ({})".format(
                  root, swing, what, sorted(set(mine.tolist()))))


def test_lanes_one_piece_is_split():
    # Long hair in one piece round the back and the sides is split vertex by vertex: back hair beyond SIDE_ANGLE,
    # side hair in front of it.
    model_rig = rig()
    co, tris, ids = wrapped_hair()
    measured = hairlanes.measure(co, tris, model_rig)
    label, _short = hairlanes.areas(measured)
    describe_lanes("one piece", measured, label, ids["sheet"])
    mine, angles = lanes_of(measured, label, ids["sheet"])
    check(len(set(measured.part[np.isin(measured.index, ids["sheet"])].tolist())) == 1,
          "lanes one piece: the sheet is one loose part")
    check((mine == hairlanes.BACK).any() and np.isin(mine, (hairlanes.LEFT, hairlanes.RIGHT)).any(),
          "lanes one piece: it has back hair and side hair ({})".format(sorted(set(mine.tolist()))))
    check((mine[np.abs(angles) > hairrig.SIDE_ANGLE] == hairlanes.BACK).all(),
          "lanes one piece: every vertex beyond SIDE_ANGLE is back hair")


def test_lanes_long_front_lock_is_side():
    # A lock 20 degrees from the front hanging 10 cm below the neck pivot is side hair, not bangs; cut to end 3 cm
    # above the neck pivot it is bangs.
    model_rig = rig()
    neck = float(model_rig.by_name[hairing.NECK].head[2])
    for bottom, names, what in ((neck - 0.10, (hairlanes.LEFT, hairlanes.RIGHT), "side hair"),
                                (neck + 0.03, (hairlanes.BANGS,), "bangs")):
        co, tris, (_cap_ids, lock_ids) = join([hair.cap(), front_lock(bottom)])
        measured = hairlanes.measure(co, tris, model_rig)
        label, _short = hairlanes.areas(measured)
        describe_lanes("front lock to {:.3f}".format(bottom), measured, label, lock_ids)
        check(lane_share(measured, label, lock_ids, *names) == 1.0,
              "lanes front lock: ending {:+.2f} m from the neck pivot it is {:s}".format(bottom - neck, what))


def test_lanes_middle_part():
    # A loose part hanging across the middle in front, most of it in neither side: narrow, it goes whole to one side
    # (a long lock centered on the face is one side's hair, not bangs and not halved); wide, it stays split (a 14 cm
    # fringe is bangs, a curtain from one side round the front to the other is left on +X and right on -X).
    model_rig = rig()
    neck = float(model_rig.by_name[hairing.NECK].head[2])
    top = around(0.0, 1.57)
    lock = hair.strip(top, np.array((top[0], top[1] - 0.02, neck - 0.10)), 0.03, np.array((1.0, 0.0, 0.0)))
    co, tris, (_cap_ids, lock_ids) = join([hair.cap(), lock])
    measured = hairlanes.measure(co, tris, model_rig)
    label, _short = hairlanes.areas(measured)
    describe_lanes("centered lock", measured, label, lock_ids)
    mine, _angles = lanes_of(measured, label, lock_ids)
    check(len(mine) > 0 and len(set(mine.tolist())) == 1 and int(mine[0]) in (hairlanes.LEFT, hairlanes.RIGHT),
          "lanes middle part: a long lock centered on the face is one side's hair, all of it ({})".format(
              sorted(set(mine.tolist()))))
    co, tris, (_cap_ids, fringe_ids) = join([hair.cap(), bangs_strip(0.0, 0.14)])
    measured = hairlanes.measure(co, tris, model_rig)
    label, _short = hairlanes.areas(measured)
    describe_lanes("14 cm fringe", measured, label, fringe_ids)
    check(lane_share(measured, label, fringe_ids, hairlanes.BANGS) == 1.0,
          "lanes middle part: a 14 cm wide fringe is bangs, all of it")
    tops = np.array([around(angle, 1.52, 0.02) for angle in np.arange(-70.0, 71.0, 10.0)])
    co, tris, (_cap_ids, curtain_ids) = join([hair.cap(), sheet(tops, tops - (0.0, 0.0, 0.30))])
    measured = hairlanes.measure(co, tris, model_rig)
    label, _short = hairlanes.areas(measured)
    describe_lanes("curtain", measured, label, curtain_ids)
    mine, _angles = lanes_of(measured, label, curtain_ids)
    x = measured.at[np.isin(measured.index, curtain_ids), 0]
    check((mine == hairlanes.LEFT).any() and (mine == hairlanes.RIGHT).any()
          and (x[mine == hairlanes.LEFT] >= 0.0).all() and (x[mine == hairlanes.RIGHT] < 0.0).all(),
          "lanes middle part: a curtain joining both sides is split, left on +X and right on -X")


def test_lanes_short():
    # Short hair: hardly any of it hangs below the chin.
    model_rig = rig()
    co, tris = short_hair()
    _label, short = hairlanes.areas(hairlanes.measure(co, tris, model_rig))
    check(short, "lanes short: short hair is short")


def test_lanes_ponytail_areas():
    # Hair Type Ponytail: the tail is the back hair and every whole loose part rooted behind the ears, wherever it
    # hangs: a tube rooted behind the head (170 degrees from the front) falling forward over the shoulder (to 40) is
    # all tail, and side hair without Hair Type Ponytail. A side strip stays side hair, and a one-piece mesh stays
    # split: only its hair beyond SIDE_ANGLE is the tail.
    model_rig = rig()
    co, tris, (_cap_ids, tube_ids, strip_ids) = join([hair.cap(), swung(170.0, 40.0, 0.40, 0.06, tube=True),
                                                      side_strip(120.0, 0.30, 0.04)])
    measured = hairlanes.measure(co, tris, model_rig)
    loose, _short = hairlanes.areas(measured)
    label, _short = hairlanes.areas(measured, ponytail=True)
    describe_lanes("tube, loose", measured, loose, tube_ids)
    describe_lanes("tube, ponytail", measured, label, tube_ids)
    describe_lanes("side strip, ponytail", measured, label, strip_ids)
    check(lane_share(measured, loose, tube_ids, hairlanes.LEFT, hairlanes.RIGHT) == 1.0,
          "lanes ponytail: without Hair Type Ponytail the tube is side hair")
    check(lane_share(measured, label, tube_ids, hairlanes.BACK) == 1.0, "lanes ponytail: the tube is all tail")
    check(lane_share(measured, label, strip_ids, hairlanes.LEFT, hairlanes.RIGHT) == 1.0,
          "lanes ponytail: the side strip stays side hair")
    co, tris, ids = wrapped_hair()
    measured = hairlanes.measure(co, tris, model_rig)
    label, _short = hairlanes.areas(measured, ponytail=True)
    mine, angles = lanes_of(measured, label, ids["sheet"])
    check(np.array_equal(mine == hairlanes.BACK, np.abs(angles) > hairrig.SIDE_ANGLE),
          "lanes ponytail: of the one-piece mesh only the hair beyond SIDE_ANGLE is the tail")


def test_lanes_nothing_hangs():
    co, tris, _ranges = join([hair.cap()])
    check(hairlanes.measure(co, tris, rig()) is None, "lanes nothing hangs: a cap alone measures as None")


# Lopsided back cards: two on the right, one across the middle reaching well over to the left, one at the left edge
# (degrees around the skull's middle, as `card_back`).
LOPSIDED = ((-75.0, -45.0), (-37.5, -22.5), (-15.0, 37.5), (45.0, 75.0))


def back_lanes(co, tris, count, model_rig=None):
    """The hanging hair of a mesh (`hairlanes.measure`), its areas, and its back area cut into ``count`` lanes:
    (measured, label, lanes)."""
    measured = hairlanes.measure(co, tris, model_rig or rig())
    label, _short = hairlanes.areas(measured)
    lanes = hairlanes.plan_area(measured, np.flatnonzero(label == hairlanes.BACK), hairlanes.across_back(measured),
                                count)
    return measured, label, lanes


def rows(lane):
    """The heights of a lane's joints."""
    return lane.joints[:, 2]


def lane_line(lane):
    """A lane's chain: its joints and tip, (n + 1, 3)."""
    return np.vstack((lane.joints, lane.tip))


def describe_lane_plan(lanes):
    for lane in lanes:
        print("      lane {:d}: {:d} joints, {:d} vertices, first {}, last {}, tip {}, light {!s}, moved {:d}, lifted "
              "{:d}".format(lane.number, len(lane.joints), len(lane.members), np.round(lane.joints[0], 3),
                            np.round(lane.joints[-1], 3), np.round(lane.tip, 3), lane.light, lane.moved, lane.lifted))


def x_at(line, heights):
    """The X of the polyline ``line`` at ``heights``."""
    order = np.argsort(line[:, 2], kind="stable")
    return np.interp(heights, line[order, 2], line[order, 0])


def lanes_crossed(a, b):
    """Whether the chains of lanes ``a`` and ``b`` swap their order in X at any of 20 heights both cover."""
    line_a, line_b = lane_line(a), lane_line(b)
    low = max(line_a[:, 2].min(), line_b[:, 2].min())
    high = min(line_a[:, 2].max(), line_b[:, 2].max())
    if high <= low:
        return False
    heights = np.linspace(low, high, 20)
    side = np.sign(x_at(line_a, heights) - x_at(line_b, heights))
    return bool((side != side[0]).any())


def nearest(points, hair_points):
    """How far each of ``points`` lies from the nearest of ``hair_points``."""
    return np.array([float(np.min(np.linalg.norm(hair_points - point, axis=1))) for point in points])


def flat_sheet(scale, depth=0.40):
    """A flat sheet of hair 15 cm behind the head's axis, 20 cm wide at its top (height 1.47, below the nape's
    hairline) and ``scale`` times as wide at its bottom ``depth`` lower: (points, triangles)."""
    axis_y = rig().by_name[hairing.HEAD].head[1]
    x = np.linspace(-0.10, 0.10, 11)
    tops = np.column_stack((x, np.full(len(x), axis_y + 0.15), np.full(len(x), 1.47)))
    bottoms = np.column_stack((x * scale, np.full(len(x), axis_y + 0.15), np.full(len(x), 1.47 - depth)))
    co, tris, _ranges = join([sheet(tops, bottoms)])
    return co, tris


def crown_layer_hair():
    """`back_sheet` and a layer standing 6 cm off the head over the back of the crown, its own loose part: 20 to 70
    degrees from the top, 60 degrees to either side of straight behind (as `hair.patch`). Returns (points,
    triangles, {"cap", "sheet", "layer": indices})."""
    co, tris, ids = back_sheet()
    layer_co, layer_tris, (layer_ids,) = join([hair.patch(np.arange(20.0, 71.0, 5.0), np.arange(-60.0, 61.0, 7.5),
                                                          0.06)])
    return (np.vstack((co, layer_co)), np.vstack((tris, layer_tris + len(co))),
            dict(ids, layer=layer_ids + len(co)))


def test_lanes_count_and_order():
    # The back sheet cut into 1 to 4 lanes: as many as asked, numbered from +X, every back vertex in one lane.
    co, tris, _ids = back_sheet()
    for count in range(1, 5):
        measured, label, lanes = back_lanes(co, tris, count)
        describe_lane_plan(lanes)
        back = np.flatnonzero(label == hairlanes.BACK)
        firsts = [float(lane.joints[0][0]) for lane in lanes]
        check(len(lanes) == count and [lane.number for lane in lanes] == list(range(count))
              and all(a > b for a, b in zip(firsts, firsts[1:])),
              "lanes count: {:d} asked, {:d} lanes numbered {}, first joints' X {} falling".format(
                  count, len(lanes), [lane.number for lane in lanes], [round(x, 3) for x in firsts]))
        members = np.concatenate([lane.members for lane in lanes]) if lanes else np.zeros(0, dtype=np.int64)
        check(len(members) == len(set(members.tolist())) and set(members.tolist()) == set(back.tolist()),
              "lanes count: with {:d} lanes every back vertex is in one lane ({:d} in lanes, {:d} back)".format(
                  count, len(members), len(back)))


def test_lanes_never_cross():
    # The sheet swept 12 cm to the side, and lopsided cards: 3 lanes, no two neighbors swapping sides.
    swept_co, swept_tris, _ids = back_sheet(sweep=0.12)
    for name, (co, tris) in (("swept sheet", (swept_co, swept_tris)), ("lopsided cards", card_back(LOPSIDED))):
        _measured, _label, lanes = back_lanes(co, tris, 3)
        describe_lane_plan(lanes)
        crossed = [lanes_crossed(a, b) for a, b in zip(lanes, lanes[1:])]
        check(len(lanes) == 3 and not any(crossed),
              "lanes never cross: {:s}: 3 lanes ({:d}), neighbors crossed {}".format(name, len(lanes), crossed))


def test_lanes_rows_shared():
    # The back sheet's straight bottom: 3 lanes with the same number of joints, on the same rows.
    co, tris, _ids = back_sheet()
    _measured, _label, lanes = back_lanes(co, tris, 3)
    describe_lane_plan(lanes)
    counts = [len(lane.joints) for lane in lanes]
    apart = max((float(np.abs(rows(a) - rows(b)).max()) for a in lanes for b in lanes
                 if len(a.joints) == len(b.joints)), default=math.inf)
    check(len(lanes) == 3 and len(set(counts)) == 1 and apart <= 0.02,
          "lanes rows: 3 lanes ({:d}) of equal joints ({}), rows within 2 cm ({:.4f} m apart)".format(
              len(lanes), counts, apart))


def test_lanes_mirrored():
    # The back sheet is mirror symmetric: so are its lanes.
    co, tris, _ids = back_sheet()
    for count in (2, 3):
        _measured, _label, lanes = back_lanes(co, tris, count)
        describe_lane_plan(lanes)
        apart = [float(np.linalg.norm(lane.joints * (-1.0, 1.0, 1.0) - other.joints, axis=1).max())
                 if lane.joints.shape == other.joints.shape else math.inf
                 for lane, other in zip(lanes, lanes[::-1])]
        check(len(lanes) == count and max(apart) <= 0.005,
              "lanes mirrored: {:d} lanes, each lane's joints mirrored within 5 mm of its twin's ({})".format(
                  count, [round(a, 4) for a in apart]))


def test_lanes_u_bottom():
    # A sheet whose outer parts (beyond 3 cm either side of the middle) end 20 cm above its middle (45 cm): the outer
    # lanes run out of hair first. Their chains end there with fewer joints, on the middle chain's rows, the last bone
    # under the last joint; the middle chain reaches lowest. On a U (middle 45 cm, edges 25 cm, a
    # parabola in between) the outer lanes' hair within FOLLOW across of their last joints runs on nearly as far as
    # the middle's (to 0.36 m below the top against the middle's 0.40, measured 2026-11-14), so their chains keep
    # the middle's joints; only the rest holds there. Both start at the scalp's lower edge: no chain is light.
    for name, shape, fewer in (("stepped", lambda x: 0.20 if abs(x) > 0.03 else 0.0, True),
                               ("parabola", lambda x: 0.20 * (x / 0.15) ** 2, False)):
        co, tris, _ids = back_sheet(depth=0.45, shape=shape)
        _measured, _label, lanes = back_lanes(co, tris, 3)
        describe_lane_plan(lanes)
        if len(lanes) != 3:
            check(False, "u bottom lanes: {:s}: 3 lanes ({:d})".format(name, len(lanes)))
            continue
        outer, middle = (lanes[0], lanes[2]), lanes[1]
        check(not any(lane.light for lane in lanes),
              "u bottom lanes: {:s}: started at the scalp's edge, none light ({})".format(
                  name, [lane.light for lane in lanes]))
        if fewer:
            check(all(len(lane.joints) < len(middle.joints) for lane in outer),
                  "u bottom lanes: {:s}: the outer lanes have fewer joints than the middle one ({})".format(
                      name, [len(lane.joints) for lane in lanes]))
        apart = [float(np.abs(rows(lane) - rows(middle)[:len(lane.joints)]).max()) if len(lane.joints) <= len(
            middle.joints) else math.inf for lane in outer]
        check(max(apart) <= 0.02, "u bottom lanes: {:s}: the outer joints on the middle lane's first rows ({})".format(
            name, [round(a, 4) for a in apart]))
        swing = [abs(float(lane.tip[0] - lane.joints[-1][0])) for lane in outer]
        check(max(swing) <= 0.03,
              "u bottom lanes: {:s}: the outer tips' X within 3 cm of their last joint's ({})".format(
                  name, [round(s, 3) for s in swing]))
        check(middle.tip[2] < min(lane.tip[2] for lane in outer),
              "u bottom lanes: {:s}: the middle lane's tip lowest ({})".format(
                  name, [round(float(lane.tip[2]), 3) for lane in lanes]))


def test_lanes_fan_and_narrow():
    # A flat sheet twice as wide at the bottom as at the top: the outer chains fan out with it. One half as wide at
    # the bottom: the outer chains stop narrowing with it partway down (NARROW).
    co, tris = flat_sheet(2.0)
    _measured, _label, lanes = back_lanes(co, tris, 3)
    describe_lane_plan(lanes)
    out = [(abs(float(lane.joints[0][0])), abs(float(lane.tip[0]))) for lane in (lanes[0], lanes[-1])] if lanes else []
    check(len(lanes) == 3 and all(tip > root for root, tip in out),
          "lanes fan: the outer tips lie farther out in X than their roots ({})".format(
              [(round(root, 3), round(tip, 3)) for root, tip in out]))
    co, tris = flat_sheet(0.5)
    _measured, _label, lanes = back_lanes(co, tris, 3)
    describe_lane_plan(lanes)
    # The sheet is half as wide at its bottom: chains following it all the way would end at half their roots' X.
    # They stop narrowing at NARROW of the lane's width, so the outer tips stay beyond 0.55 of it (measured 0.62).
    kept = [abs(float(lane.tip[0])) / abs(float(lane.joints[0][0])) for lane in (lanes[0], lanes[-1])] if lanes else []
    check(len(lanes) == 3 and min(kept) >= 0.55 and max(kept) < 1.0,
          "lanes narrow: the outer tips' X between 0.55 of their roots' and all of it ({})".format(
              [round(k, 3) for k in kept]))


def test_lanes_clumps():
    # One lane over two strips 12 cm apart starting at the same height, one 20 cm longer: no joint between them, and
    # the chain stays on the strip it follows (it ends with it), not moved across by the check. So too with a hole
    # cut across the followed strip around its second joint.
    model_rig = rig()
    axis_y = model_rig.by_name[hairing.HEAD].head[1]
    y, top, width = axis_y + 0.15, 1.47, 0.03
    strips = ((-0.06, 0.30), (0.06, 0.50))

    def off_strip(points, x, length):
        # How far ``points`` lie from the flat strip at ``x`` hanging ``length`` from ``top`` (model space is world
        # space for this rig).
        side = np.maximum(np.abs(points[:, 0] - x) - width / 2.0, 0.0)
        down = np.maximum(np.maximum(points[:, 2] - top, top - length - points[:, 2]), 0.0)
        return np.sqrt(side ** 2 + (points[:, 1] - y) ** 2 + down ** 2)

    def one_strip(name, lanes):
        describe_lane_plan(lanes)
        line = [off_strip(lane_line(lane), x, length) for lane in lanes for x, length in strips]
        check(len(lanes) == 1 and any((o <= 0.01).all() for o in line),
              "lanes clumps: {:s}: every joint and the tip within 1 cm of the same strip ({})".format(
                  name, [np.round(o, 3).tolist() for o in line]))

    co, tris, ranges = join([hair.strip((x, y, top), (x, y, top - length), width, (1.0, 0.0, 0.0),
                                        rows=int(round(length / 0.02))) for x, length in strips])
    _measured, _label, lanes = back_lanes(co, tris, 1)
    off = [off_strip(lane.joints, x, length) for lane in lanes for x, length in strips]
    print("      joints off the short strip {}, off the long strip {}".format(
        *[np.round(o, 3).tolist() for o in off]))
    check(len(lanes) == 1 and (np.minimum(off[0], off[1]) <= 0.01).all(),
          "lanes clumps: one lane ({:d}), every joint within 1 cm of a strip".format(len(lanes)))
    one_strip("strips", lanes)
    if len(lanes) != 1 or len(lanes[0].joints) < 2:
        return
    # The followed strip (by the first joint's side) with a hole 8 and 10 cm high around the second joint.
    followed = ranges[int(lanes[0].joints[0][0] > 0.0)]
    center = float(lanes[0].joints[1][2])
    middle = co[tris].mean(axis=1)
    for half in (0.04, 0.05):
        hole = np.isin(tris, followed).all(axis=1) & (np.abs(middle[:, 2] - center) < half)
        one_strip("a {:.0f} cm hole at the second joint".format(200.0 * half), back_lanes(co, tris[~hole], 1)[2])
    # The strips with vertex rows 6 cm apart: a joint on a strip between two rows can lie 3 cm from every vertex, so
    # the check fires there; it keeps to the strip the chain follows. Measured across (X) only: every joint and the
    # tip within 1 cm of one strip's sides.
    co, tris, _ranges = join([hair.strip((x, y, top), (x, y, top - length), width, (1.0, 0.0, 0.0),
                                         rows=int(round(length / 0.06))) for x, length in strips])
    lanes = back_lanes(co, tris, 1)[2]
    describe_lane_plan(lanes)
    across = [np.maximum(np.abs(lane_line(lane)[:, 0] - x) - width / 2.0, 0.0) for lane in lanes for x, _l in strips]
    check(len(lanes) == 1 and any((a <= 0.01).all() for a in across),
          "lanes clumps: vertex rows 6 cm apart: every joint and the tip within 1 cm across of the same strip "
          "({})".format([np.round(a, 3).tolist() for a in across]))


def test_lanes_start_at_scalp_edge():
    # A layer standing off the cap at the back of the crown over a sheet hanging from the nape: the back lanes start
    # at the scalp's lower edge behind the head, not on the crown, and aren't light.
    co, tris, ids = crown_layer_hair()
    measured, label, lanes = back_lanes(co, tris, 3)
    describe_lane_plan(lanes)
    back = np.flatnonzero(label == hairlanes.BACK)
    layer = np.isin(measured.index[back], ids["layer"])
    edge = measured.edge[back]
    edge = float(np.median(edge[np.isfinite(edge)]))
    print("      the scalp's edge behind at {:.3f}, the crown layer from {:.3f} to {:.3f} ({:d} back vertices)".format(
        edge, float(measured.at[back[layer], 2].min()), float(measured.at[back[layer], 2].max()), int(layer.sum())))
    firsts = [round(float(lane.joints[0][2]), 3) for lane in lanes]
    check(len(lanes) == 3 and layer.any() and all(abs(z - edge) <= 0.03 for z in firsts),
          "lanes scalp edge: 3 lanes ({:d}) starting within 3 cm of the scalp's edge (first joints at {})".format(
              len(lanes), firsts))
    check(lanes and not any(lane.light for lane in lanes), "lanes scalp edge: none light")


def test_lanes_bun():
    # A bun standing up from the crown: one lane, starting in the bun, light.
    model_rig = rig(MIQOTE)
    co, tris, _ids = bun_hair()
    measured, label, lanes = back_lanes(co, tris, 1, model_rig)
    describe_lane_plan(lanes)
    edge = measured.edge[label == hairlanes.BACK]
    edge = float(edge[np.isfinite(edge)].max())
    check(len(lanes) == 1 and lanes[0].joints[0][2] > edge and lanes[0].light,
          "lanes bun: one lane ({:d}), first joint above the scalp's edge ({:.3f}; tip at {:.3f}; edge {:.3f}), "
          "light".format(len(lanes), float(lanes[0].joints[0][2]) if lanes else math.nan,
                         float(lanes[0].tip[2]) if lanes else math.nan, edge))


def test_lanes_check_keeps_rows():
    # One lane over a flat strip 20 cm wide, with a hole 10 cm across and 8 cm high cut around its middle row: the
    # check moves that joint onto the hair beside the hole, on its row. Without the hole it moves nothing.
    model_rig = rig()
    co, tris = flat_sheet(1.0)
    _measured, _label, plain = back_lanes(co, tris, 1)
    describe_lane_plan(plain)
    if len(plain) != 1 or len(plain[0].joints) < 3 or plain[0].moved:
        check(False, "lanes check: one lane of 3 joints or more without the hole, none moved ({})".format(
            [(len(lane.joints), lane.moved) for lane in plain]))
        return
    row = len(plain[0].joints) // 2
    center = plain[0].joints[row]
    middle = model_rig.model(co)[tris].mean(axis=1)
    hole = (np.abs(middle[:, 0] - center[0]) < 0.05) & (np.abs(middle[:, 2] - center[2]) < 0.04)
    measured, _label, lanes = back_lanes(co, tris[~hole], 1)
    describe_lane_plan(lanes)
    if len(lanes) != 1 or len(lanes[0].joints) != len(plain[0].joints):
        check(False, "lanes check: one lane of {:d} joints with the hole ({})".format(
            len(plain[0].joints), [len(lane.joints) for lane in lanes]))
        return
    lane = lanes[0]
    joint = lane.joints[row]
    off = float(nearest(joint[None], measured.at[lane.members])[0])
    print("      row {:d}: at {} without the hole, {} with it, {:.4f} m off the hair".format(
        row, np.round(center, 3), np.round(joint, 3), off))
    check(lane.moved >= 1 and lane.lifted == 0 and abs(float(joint[2] - center[2])) <= 0.02
          and off <= hairlanes.CHECK_REACH,
          "lanes check: the joint in the hole moved onto the hair (moved {:d}, lifted {:d}), its height within 2 cm of "
          "its row ({:.4f} m)".format(lane.moved, lane.lifted, abs(float(joint[2] - center[2]))))


def test_lanes_short_followed_hair():
    # One lane over two strips behind the head: a wide one 8 cm long, which the chain follows, and a narrow one 12 cm
    # across running on to 30 cm, which sets the area's rows. The wide strip ends just above the second row, so with
    # two bones no hair lies under the second joint: that strip is too short for two rows and the chain splits its
    # length in two, the tip at its end. Before, the tip went on past the strip and the check lifted it back above the
    # second joint (measured 2026-11-14: a 178 degree fold, as on c0801h0120's left chain).
    model_rig = rig()
    axis_y = model_rig.by_name[hairing.HEAD].head[1]
    y, top = axis_y + 0.15, 1.47
    strips = ((0.0, 0.04, 0.08), (0.12, 0.01, 0.30))
    co, tris, _ranges = join([hair.strip((x, y, top), (x, y, top - length), width, (1.0, 0.0, 0.0),
                                         rows=int(round(length / 0.01))) for x, width, length in strips])
    measured, _label, lanes = back_lanes(co, tris, 1, model_rig)
    describe_lane_plan(lanes)
    x, width, length = strips[0]
    if len(lanes) != 1 or abs(float(lanes[0].joints[0][0]) - x) > width / 2.0 + hairlanes.CHECK_REACH:
        check(False, "lanes short followed hair: one lane, starting on the wide strip ({})".format(
            [np.round(lane.joints[0], 3).tolist() for lane in lanes]))
        return
    lane = lanes[0]
    line = lane_line(lane)
    side = np.maximum(np.abs(line[:, 0] - x) - width / 2.0, 0.0)
    down = np.maximum(np.maximum(line[:, 2] - top, top - length - line[:, 2]), 0.0)
    off = np.sqrt(side ** 2 + (line[:, 1] - y) ** 2 + down ** 2)
    bones = np.diff(line, axis=0)
    bend = [round(math.degrees(math.acos(float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0)))), 1)
            for a, b in zip(bones, bones[1:])]
    print("      off the wide strip {}, bends {}".format(np.round(off, 3).tolist(), bend))
    check(len(lane.joints) == 2 and lane.tip[2] < lane.joints[-1][2] and lane.moved == 0
          and (off <= 0.02).all(),
          "lanes short followed hair: two joints ({:d}), the tip below the last joint ({:.3f} < {:.3f}), none moved "
          "({:d}), every joint and the tip within 2 cm of the wide strip ({})".format(
              len(lane.joints), float(lane.tip[2]), float(lane.joints[-1][2]), lane.moved, np.round(off, 3).tolist()))
    # The split, not the area's second row (0.0836 below the top): the second joint halfway between the first and the
    # strip's end, give or take half a level band (the end is where the strip's last band holds hair).
    halfway = 0.5 * (float(lane.joints[0][2]) + top - length)
    check(len(lane.joints) == 2 and abs(float(lane.joints[1][2]) - halfway) <= hairlanes.LEVEL_BAND / 2.0,
          "lanes short followed hair: the second joint halfway down the wide strip ({:.3f}, halfway {:.3f})".format(
              float(lane.joints[-1][2]), halfway))


def bone_lengths(joints, tip):
    """The lengths of a chain's bones (meters)."""
    return np.linalg.norm(np.diff(np.vstack((joints, tip)), axis=0), axis=1)


def test_lanes_joints_on_one_spot():
    # One lane over a strip 4 cm wide that hangs 14 cm, stops, and goes on from 48 to 54 cm: the rows at 20 and 30 cm
    # lie in the gap, with no hair on their row, so the check lifts both joints onto the strip's end above, the same
    # vertex (c0801h0130's back chains, measured 2026-11-15: two joints on one vertex, a bone of length 0). The
    # second of them goes: no bone is shorter than MIN_BONE, and the chain still runs to the lower hair.
    model_rig = rig()
    axis_y = model_rig.by_name[hairing.HEAD].head[1]
    y, top = axis_y + 0.15, 1.47
    co, tris, _ranges = join([hair.strip((0.0, y, top - start), (0.0, y, top - end), 0.04, (1.0, 0.0, 0.0),
                                         rows=int(round((end - start) / 0.01)))
                              for start, end in ((0.0, 0.14), (0.48, 0.54))])
    _measured, _label, lanes = back_lanes(co, tris, 1, model_rig)
    describe_lane_plan(lanes)
    if len(lanes) != 1:
        check(False, "lanes joints on one spot: one lane ({:d})".format(len(lanes)))
        return
    lane = lanes[0]
    bones = bone_lengths(lane.joints, lane.tip)
    print("      joints at {}, bones {} m".format(np.round(lane.joints[:, 2], 3).tolist(), np.round(bones, 4).tolist()))
    check(lane.lifted >= 1 and float(bones.min()) >= hairlanes.MIN_BONE,
          "lanes joints on one spot: a joint was lifted ({:d}), no bone under {:.3f} m ({:.4f})".format(
              lane.lifted, hairlanes.MIN_BONE, float(bones.min())))
    check(len(lane.joints) >= 3 and lane.tip[2] < top - 0.48,
          "lanes joints on one spot: 3 joints or more ({:d}), the tip on the lower hair (z {:.3f})".format(
              len(lane.joints), float(lane.tip[2])))


def test_lanes_bone_length():
    # No bone of any chain is shorter than MIN_BONE: `_spaced` on a line by hand, then the whole rig on the fixtures.
    line = np.array(((0.0, 0.0, 1.0), (0.0, 0.0, 0.9), (0.02, 0.0, 0.85), (0.02, 0.0, 0.85), (0.024, 0.0, 0.848),
                     (0.0, 0.0, 0.7)))
    kept, state = hairlanes._spaced(line, np.array((0, 0, 2, 2, 1, 0)))
    check(np.array_equal(kept, line[[0, 1, 2, 5]]) and state.tolist() == [0, 0, 2, 0],
          "lanes bone length: joints on one spot and 4.5 mm from it leave the first of them ({})".format(
              np.round(kept, 3).tolist()))
    # At two bones nothing can go: a second joint 3.5 mm from the first goes halfway to the tip instead (short hair's
    # ring, c0801h0138), and one 5 mm above the tip too.
    for name, second in (("near the root", (0.003, 0.0, 1.002)), ("near the tip", (0.0, 0.004, 0.953))):
        line = np.array(((0.0, 0.0, 1.0), second, (0.0, 0.0, 0.95)))
        kept, state = hairlanes._spaced(line, np.zeros(3, dtype=np.int64))
        check(len(kept) == 3 and np.allclose(kept[1], (0.0, 0.0, 0.975)) and np.array_equal(kept[[0, 2]], line[[0, 2]]),
              "lanes bone length: two bones, the second joint {:s} goes halfway ({})".format(
                  name, np.round(kept, 4).tolist()))
    # Root and tip under MIN_CHAIN_LENGTH apart with nothing between them to keep: no chain.
    line = np.array(((0.0, 0.0, 1.0), (0.0, 0.0, 0.995), (0.0, 0.0, 0.98)))
    check(hairlanes._spaced(line, np.zeros(3, dtype=np.int64))[0] is None,
          "lanes bone length: root and tip 2 cm apart with the joint on the root: no chain")
    model_rig = rig()
    full = counts(back=4, left=2, right=2, bangs=6)
    fixtures = (("rig counts hair", rig_counts_hair(), full, model_rig), ("card hair", card_hair(300), full, model_rig),
                ("short hair, 4", short_hair(), counts(bangs=4), model_rig),
                ("short hair, 6", short_hair(), counts(bangs=6), model_rig),
                ("bun", bun_hair()[:2], counts(back=1), rig(MIQOTE)),
                ("fringe and neck", fringe_and_neck()[:2], None, rig(MIQOTE)))
    for name, (co, tris), asked, fixture_rig in fixtures:
        chains, _made, _short = hairlanes.plan(co, tris, fixture_rig, counts=asked)
        shortest = min((float(bone_lengths(chain.joints, chain.tip).min()) for chain in chains), default=math.nan)
        check(chains and shortest >= hairlanes.MIN_BONE,
              "lanes bone length: {:s}: {:d} chains, no bone under {:.3f} m ({:.4f})".format(
                  name, len(chains), hairlanes.MIN_BONE, shortest))


def test_lanes_sweeping_hair():
    # One lane over a strand behind the head that hangs 10 cm, sweeps 12 cm sideways over the next 6 cm, and hangs on
    # to 45 cm (a braid swept over the shoulder, c0801h0153): the chain follows it to its end, though the strand
    # moves farther than FOLLOW across between two rows. Before, it ended above the sweep (measured 2026-11-14).
    model_rig = rig()
    axis_y = model_rig.by_name[hairing.HEAD].head[1]
    y, top = axis_y + 0.15, 1.47
    bends = ((0.0, top), (0.0, top - 0.10), (0.12, top - 0.16), (0.12, top - 0.45))
    co, tris, _ranges = join([hair.strip((a[0], y, a[1]), (b[0], y, b[1]), 0.03, (0.0, 1.0, 0.0), rows=10)
                              for a, b in zip(bends, bends[1:])])
    measured, _label, lanes = back_lanes(co, tris, 1, model_rig)
    describe_lane_plan(lanes)
    if len(lanes) != 1:
        check(False, "lanes sweeping hair: one lane ({:d})".format(len(lanes)))
        return
    lane = lanes[0]
    line = lane_line(lane)
    off = nearest(line, measured.at[lane.members])
    bottom = np.array((0.12, y, top - 0.45))
    # The strand itself turns by atan(12 / 6) at each end of its sweep: the chain turns no sharper.
    turn = math.degrees(math.atan2(0.12, 0.06))
    bend = line_bends(line)
    # And its joints lie on the strand: within 2 cm of its middle line (the strip is 3 cm deep).
    along = off_line(line, [(x, top - z) for x, z in bends], y, top)
    print("      joints off the strand {}, off its line {}, tip {} from its bottom, bends {} (the strand turns "
          "{:.1f})".format(np.round(off, 3).tolist(), np.round(along, 3).tolist(),
                           round(float(np.linalg.norm(lane.tip - bottom)), 3), bend, turn))
    check(float(np.linalg.norm(lane.tip - bottom)) <= hairlanes.CHECK_REACH and lane.lifted == 0 and max(bend) <= turn
          and (along <= 0.02).all(),
          "lanes sweeping hair: the tip within 3 cm of the strand's bottom ({}), none lifted ({:d}), no bend sharper "
          "than the strand's ({}), every joint within 2 cm of its line ({})".format(
              np.round(lane.tip, 3).tolist(), lane.lifted, bend, np.round(along, 3).tolist()))


def strand(bends, y, top, width=0.01, step=0.01):
    """A strand of hair behind the head through ``bends`` ((x, depth below ``top``) pairs) at ``y``: flat strips
    ``width`` deep in Y, a vertex row every ``step`` at most, sharing their vertices at the bends. (points, triangles,
    its vertex indices)."""
    parts = [hair.strip((a[0], y, top - a[1]), (b[0], y, top - b[1]), width, (0.0, 1.0, 0.0),
                        rows=max(1, int(round(max(abs(b[1] - a[1]), abs(b[0] - a[0])) / step))))
             for a, b in zip(bends, bends[1:])]
    co, tris, ranges = join(parts)
    return co, tris, np.concatenate(ranges)


def off_line(points, bends, y, top):
    """How far each of ``points`` lies from the middle line of a `strand` through ``bends``."""
    corners = np.array([(x, y, top - depth) for x, depth in bends])
    starts, spans = corners[:-1], np.diff(corners, axis=0)
    offset = points[:, None, :] - starts[None]
    t = np.clip((offset * spans[None]).sum(axis=2) / (spans * spans).sum(axis=1)[None], 0.0, 1.0)
    return np.linalg.norm(offset - t[:, :, None] * spans[None], axis=2).min(axis=1)


def card(left, right, start, end, y, top, step=0.01):
    """A flat card behind the head at ``y``, from ``left`` to ``right`` in X and ``start`` to ``end`` below ``top``,
    its vertices on a ``step`` grid: (points, triangles)."""
    x = np.linspace(left, right, int(round((right - left) / step)) + 1)
    co, tris, _ranges = join([sheet(np.column_stack((x, np.full(len(x), y), np.full(len(x), top - start))),
                                    np.column_stack((x, np.full(len(x), y), np.full(len(x), top - end))),
                                    rows=int(round((end - start) / step)))])
    return co, tris


def together(*meshes):
    """The (points, triangles) of ``meshes`` as one mesh, not joined."""
    co, tris, start = [], [], 0
    for points, triangles in meshes:
        co.append(points)
        tris.append(triangles + start)
        start += len(points)
    return np.vstack(co), np.vstack(tris)


def line_bends(line):
    """The angle between each two bones of the polyline ``line`` (degrees)."""
    bones = np.diff(line, axis=0)
    cos = (bones[:-1] * bones[1:]).sum(axis=1) / (np.linalg.norm(bones[:-1], axis=1)
                                                  * np.linalg.norm(bones[1:], axis=1))
    return [round(math.degrees(math.acos(float(c))), 1) for c in np.clip(cos, -1.0, 1.0)]


def test_lanes_separate_strand():
    # One lane over strand A, sweeping from x 0 to 4.7 cm and ending 17 cm down, and a second strand of its own at
    # x 7.9 cm: "jump" from 12 to 40 cm down (beside A's end, 3.2 cm across of it), "void" only from 30 to 40 cm down.
    # Neither is joined to A through the mesh, so the chain ends with A (a clump of its own, as test_lanes_clumps).
    # Before, the sweep's walk took the middle of all hair within FOLLOW, slid onto the second strand and ran on down
    # it (measured 2026-11-14: tips at 0.397 and 0.396).
    model_rig = rig()
    y, top = model_rig.by_name[hairing.HEAD].head[1] + 0.15, 1.47
    a_co, a_tris, _a = strand(((0.0, 0.0), (0.047, 0.17)), y, top)
    for name, start in (("jump", 0.12), ("void", 0.30)):
        b_co, b_tris, _b = strand(((0.079, start), (0.079, 0.40)), y, top)
        co, tris = np.vstack((a_co, b_co)), np.vstack((a_tris, b_tris + len(a_co)))
        _measured, _label, lanes = back_lanes(co, tris, 1, model_rig)
        describe_lane_plan(lanes)
        off = nearest(lane_line(lanes[0]), model_rig.model(a_co)) if len(lanes) == 1 else np.full(1, math.inf)
        check(len(lanes) == 1 and (off <= 0.02).all(),
              "lanes separate strand: {:s}: every joint and the tip within 2 cm of strand A ({})".format(
                  name, np.round(off, 3).tolist()))


def test_lanes_sweep_to_end():
    # One lane over a strand behind the head that hangs 10 cm, then sweeps sideways to its end: as far as it drops
    # (slope 1.0, 30 cm) or twice as far (slope 2.0, 24 cm over 12 cm). The chain follows it to its end, also under its
    # last row. Before, the trim loop measured the hair under the last joint within FOLLOW across of it and cut the
    # chain back to two joints (measured 2026-11-14 on slope 1.0: tip 25 cm above the strand's end).
    model_rig = rig()
    y, top = model_rig.by_name[hairing.HEAD].head[1] + 0.15, 1.47
    for slope, end_x, depth in ((1.0, 0.30, 0.40), (2.0, 0.24, 0.22)):
        co, tris, _ids = strand(((0.0, 0.0), (0.0, 0.10), (end_x, depth)), y, top)
        measured, _label, lanes = back_lanes(co, tris, 1, model_rig)
        describe_lane_plan(lanes)
        if len(lanes) != 1:
            check(False, "lanes sweep to end: slope {:.1f}: one lane ({:d})".format(slope, len(lanes)))
            continue
        lane = lanes[0]
        end = model_rig.model(np.array([(end_x, y, top - depth)]))[0]
        off = nearest(lane_line(lane), measured.at[lane.members])
        print("      joints off the strand {}, tip {} from its end".format(
            np.round(off, 3).tolist(), round(float(np.linalg.norm(lane.tip - end)), 3)))
        check(float(np.linalg.norm(lane.tip - end)) <= hairlanes.CHECK_REACH and lane.lifted == 0
              and (off <= 0.02).all(),
              "lanes sweep to end: slope {:.1f}: the tip within 3 cm of the strand's end ({}), none lifted ({:d}), "
              "every joint within 2 cm of the strand ({})".format(slope, np.round(lane.tip, 3).tolist(), lane.lifted,
                                                                np.round(off, 3).tolist()))


def test_lanes_coarse_strands_apart():
    # Two strands with vertex rows 4 cm apart, 3.2 cm apart across (strand A of test_lanes_separate_strand and one at
    # x 7.9 cm): their median edge is 4 cm, but touching never reaches past LEVEL_STEP, so they are two pieces.
    # Before, touching reached the median edge and joined them (a chain on A ran on down the other, measured
    # 2026-11-14 in the reviewer's probe).
    model_rig = rig()
    y, top = model_rig.by_name[hairing.HEAD].head[1] + 0.15, 1.47
    a_co, a_tris, _a = strand(((0.0, 0.0), (0.047, 0.17)), y, top, step=0.04)
    b_co, b_tris, _b = strand(((0.079, 0.12), (0.079, 0.40)), y, top, step=0.04)
    co, tris = together((a_co, a_tris), (b_co, b_tris))
    measured = hairlanes.measure(co, tris, model_rig)
    links = hairlanes._links(measured, np.arange(len(measured.index)))
    _inside, piece = hairlanes._pieces(measured.drop, -1.0, np.inf, links)
    on_a = measured.index < len(a_co)
    apart = set(piece[on_a].tolist()).isdisjoint(piece[~on_a].tolist())
    check(on_a.any() and (~on_a).any() and apart and links[3] <= hairlanes.LEVEL_STEP,
          "lanes coarse strands apart: two pieces ({}), touching reach {:.3f} m".format(apart, links[3]))


def test_lanes_card_behind():
    # A strand 6 cm deep that hangs 10 cm, sweeps 12 cm sideways over 6 cm and hangs on to 45 cm, and 3 cm behind it
    # (never touching) a card from x 10 to 40 cm, 15 to 45 cm down. The card lies within FOLLOW across of the strand's
    # joints, so it counts in their middles, but the chain follows the strand: its tip at the strand's end. Before, the
    # hair under a joint was everything joined to all its hair, card included (measured 2026-11-14 in the reviewer's
    # probe: tip at x 0.24).
    model_rig = rig()
    y, top = model_rig.by_name[hairing.HEAD].head[1] + 0.15, 1.47
    bends = ((0.0, 0.0), (0.0, 0.10), (0.12, 0.16), (0.12, 0.45))
    co, tris = together(strand(bends, y, top, width=0.06)[:2], card(0.10, 0.40, 0.15, 0.45, y + 0.06, top))
    _measured, _label, lanes = back_lanes(co, tris, 1, model_rig)
    describe_lane_plan(lanes)
    end = model_rig.model(np.array([(0.12, y, top - 0.45)]))[0]
    gap = float(np.linalg.norm(lanes[0].tip - end)) if len(lanes) == 1 else math.inf
    check(gap <= hairlanes.CHECK_REACH, "lanes card behind: one lane ({:d}), its tip {:.3f} m from the strand's "
          "end".format(len(lanes), gap))


def test_lanes_touching_cards():
    # A swept strand in two cards that aren't joined: the upper one hangs 10 cm and starts its sweep, 6 cm sideways to
    # 13 cm down; the lower one, half a centimetre behind (closer than its edges and LEVEL_STEP), overlaps it from
    # 12.5 cm down, sweeps on to x 12 cm at 16 cm down and hangs to 45 cm. They touch, so they are one piece and the
    # chain follows both to the end. With touching left out it ends with the upper card (measured 2026-11-14: tip
    # 13.7 cm down).
    model_rig = rig()
    y, top = model_rig.by_name[hairing.HEAD].head[1] + 0.15, 1.47
    co, tris = together(strand(((0.0, 0.0), (0.0, 0.10), (0.06, 0.13)), y, top)[:2],
                        strand(((0.06, 0.125), (0.12, 0.16), (0.12, 0.45)), y + 0.005, top)[:2])
    _measured, _label, lanes = back_lanes(co, tris, 1, model_rig)
    describe_lane_plan(lanes)
    end = model_rig.model(np.array([(0.12, y, top - 0.45)]))[0]
    gap = float(np.linalg.norm(lanes[0].tip - end)) if len(lanes) == 1 else math.inf
    check(gap <= hairlanes.CHECK_REACH, "lanes touching cards: one lane ({:d}), its tip {:.3f} m from the lower card's "
          "end".format(len(lanes), gap))


def test_lanes_split_before_sweep():
    # A bar of hair 20 cm across behind the head, 2 cm high (the first row's band); a strand that leaves its middle,
    # sweeps 5.5 cm sideways to 6 cm down and hangs to 8 cm, inside the second row's band (10 cm down); and two
    # strands at the bar's ends hanging to 30 cm (they set the rows). The chain sweeps at its second row onto the
    # strand, which ends there, so at its fewest bones it splits the hair its first joint follows (within FOLLOW
    # across of it) in two: the second joint halfway down, the tip on that hair, nothing moved or lifted. Before, the
    # hair under the first joint was all hair joined to its whole row band, the outer strands too, and the chain ran
    # down the strand 10 cm across (measured 2026-11-14; so c1401h0128's Detect bangs went from 42 to 70 degrees).
    model_rig = rig()
    y, top = model_rig.by_name[hairing.HEAD].head[1] + 0.15, 1.47
    co, tris = together(card(-0.10, 0.10, 0.0, 0.02, y, top),
                        strand(((0.0, 0.025), (0.055, 0.06), (0.055, 0.08)), y, top)[:2],
                        strand(((-0.10, 0.025), (-0.10, 0.30)), y, top)[:2],
                        strand(((0.10, 0.025), (0.10, 0.30)), y, top)[:2])
    measured, _label, lanes = back_lanes(co, tris, 1, model_rig)
    describe_lane_plan(lanes)
    if len(lanes) != 1:
        check(False, "lanes split before sweep: one lane ({:d})".format(len(lanes)))
        return
    lane = lanes[0]
    line = lane_line(lane)
    depth = top - line[:, 2]
    off = nearest(line, measured.at[lane.members])
    halfway = 0.5 * (depth[0] + depth[-1])
    check(len(lane.joints) == 2 and lane.moved == 0 and depth[-1] > depth[1]
          and abs(depth[1] - halfway) <= hairlanes.LEVEL_BAND / 2.0 and (np.abs(line[:, 0]) < 0.08).all()
          and (off <= hairlanes.CHECK_REACH).all(),
          "lanes split before sweep: two joints ({:d}), none moved ({:d}), the tip below the second joint, that joint "
          "halfway down ({} m down, halfway {:.3f}), off the outer strands (X {}), on the hair ({})".format(
              len(lane.joints), lane.moved, np.round(depth, 3).tolist(), halfway, np.round(line[:, 0], 3).tolist(),
              np.round(off, 3).tolist()))


def test_lanes_swept_time():
    # Dense swept hair: 100 layers 1.5 mm apart of a ribbon 2 cm deep (5 vertices across, rows 5 mm apart) that hangs
    # 10 cm, sweeps 12 cm sideways over 6 cm and hangs on to 45 cm. Its chain follows the sweep to the end, within the
    # time test_lanes_card_hair_time allows for 3000 cards.
    model_rig = rig()
    y, top = model_rig.by_name[hairing.HEAD].head[1] + 0.15, 1.47
    bends = ((0.0, 0.0), (0.0, 0.10), (0.12, 0.16), (0.12, 0.45))
    layers = []
    for k in range(100):
        depth = np.linspace(-0.01, 0.01, 5) + y + 0.0015 * k
        layers.append(join([sheet(np.column_stack((np.full(5, a[0]), depth, np.full(5, top - a[1]))),
                                  np.column_stack((np.full(5, b[0]), depth, np.full(5, top - b[1]))),
                                  rows=int(round(max(abs(b[1] - a[1]), abs(b[0] - a[0])) / 0.005)))
                            for a, b in zip(bends, bends[1:])])[:2])
    co, tris = together(*layers)
    measured = hairlanes.measure(co, tris, model_rig)
    label, _short = hairlanes.areas(measured)
    back = np.flatnonzero(label == hairlanes.BACK)
    started = time.perf_counter()
    lanes = hairlanes.plan_area(measured, back, hairlanes.across_back(measured), 1)
    took = time.perf_counter() - started
    measured.group = None
    started = time.perf_counter()
    hairlanes.plan_area(measured, back, hairlanes.across_back(measured), 1)
    plain = time.perf_counter() - started
    # The hair's end, in the middle of its layers.
    end = model_rig.model(np.array([(0.12, y + 0.0015 * 99 / 2.0, top - 0.45)]))[0]
    gap = float(np.linalg.norm(lanes[0].tip - end)) if len(lanes) == 1 else math.inf
    most = 2.0 * CARDS_TIME + 1.0
    check(gap <= hairlanes.CHECK_REACH and took <= most,
          "lanes swept time: {:d} vertices, the chain's tip {:.3f} m from the hair's end, {:.2f} s (without the mesh "
          "{:.2f} s, at most {:.2f} s)".format(len(co), gap, took, plain, most))


def test_lanes_flat_ring():
    # Cards all between two heights 2 cm apart: too short for a chain, nothing built, and no warning on the way.
    model_rig = rig()
    axis_y = model_rig.by_name[hairing.HEAD].head[1]
    cards = []
    for angle in np.radians(np.arange(130.0, 231.0, 10.0)):
        at = np.array((0.13 * math.sin(angle), axis_y - 0.13 * math.cos(angle), 0.0))
        cards.append(hair.strip(at + (0.0, 0.0, 1.42), at + (0.0, 0.0, 1.40), 0.03,
                                np.array((math.cos(angle), math.sin(angle), 0.0)), rows=2))
    co, tris, _ranges = join(cards)
    measured = hairlanes.measure(co, tris, model_rig)
    label, _short = hairlanes.areas(measured)
    back = np.flatnonzero(label == hairlanes.BACK)
    area = float(measured.area[back].sum())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lanes = hairlanes.plan_area(measured, back, hairlanes.across_back(measured), 3)
    check(area >= hairlanes.MIN_AREA and lanes == [] and not caught,
          "lanes flat ring: {:.4f} m² of back hair 2 cm high gives no lane ({:d}), no warning ({})".format(
              area, len(lanes), [str(w.message) for w in caught]))


def test_lanes_double_sided():
    # Every triangle doubled back to back: the same lanes.
    co, tris, _ids = back_sheet()
    _measured, _label, single = back_lanes(co, tris, 3)
    _measured, _label, double = back_lanes(co, np.vstack((tris, tris[:, ::-1])), 3)
    apart = [float(np.abs(a.joints - b.joints).max()) if a.joints.shape == b.joints.shape else math.inf
             for a, b in zip(single, double)]
    check(len(single) == len(double) == 3 and max(apart) <= 0.001,
          "lanes double sided: the same 3 lanes, joints within 1 mm ({})".format([round(a, 5) for a in apart]))


def test_lanes_detect_width():
    # Detect: a lane per Chain Spacing of the area's width, one for a narrow strip and for hair wider than long.
    co, tris, _ids = back_sheet(half_width=0.15)
    measured = hairlanes.measure(co, tris, rig())
    label, _short = hairlanes.areas(measured)
    count = hairlanes.lane_count(measured, np.flatnonzero(label == hairlanes.BACK), hairlanes.across_back(measured),
                                 0.08, 4)
    check(count in (3, 4), "lanes detect: the back sheet gets 3 or 4 lanes ({:d})".format(count))
    co, tris, _ranges = join([hair.cap(), side_strip(120.0, 0.30, 0.06)])
    measured = hairlanes.measure(co, tris, rig())
    label, _short = hairlanes.areas(measured)
    count = hairlanes.lane_count(measured, np.flatnonzero(label == hairlanes.LEFT), hairlanes.across_side(measured),
                                 0.08, 2)
    check(count == 1, "lanes detect: a 6 cm side strip gets 1 lane ({:d})".format(count))
    co, tris, _ids = bun_hair()
    measured = hairlanes.measure(co, tris, rig(MIQOTE))
    label, _short = hairlanes.areas(measured)
    count = hairlanes.lane_count(measured, np.flatnonzero(label == hairlanes.BACK), hairlanes.across_back(measured),
                                 0.08, 4)
    check(count == 1, "lanes detect: a bun, wider than long, gets 1 lane ({:d})".format(count))


def rig_counts_hair():
    """`long_hair` with `back_sheet` and a `side_strip` 60 degrees from the front on each side, 30 cm long and 4 cm
    wide: (points, triangles)."""
    co, tris, _part, _names = long_hair()
    sheet_co, sheet_tris, _ids = back_sheet()
    strips_co, strips_tris, _ranges = join([side_strip(azimuth, 0.30, 0.04) for azimuth in (120.0, -120.0)])
    return (np.vstack((co, sheet_co, strips_co)),
            np.vstack((tris, sheet_tris + len(co), strips_tris + len(co) + len(sheet_co))))


def by(chains, kind):
    """The chains of ``kind`` among ``chains``, by order."""
    return sorted((chain for chain in chains if chain.kind == kind), key=lambda chain: chain.order)


def test_lanes_rig_counts():
    # The whole rig with counts: as many chains of each area as asked, the left side's on +X, and with Hair Type
    # Loose no ponytail.
    co, tris = rig_counts_hair()
    chains, made, short = hairlanes.plan(co, tris, rig(), counts=counts(back=3, left=2, right=1, bangs=1))
    describe_columns(chains)
    sides = by(chains, "side")
    left = [chain for chain in sides if chain.joints[0][0] > 0.0]
    right = [chain for chain in sides if chain.joints[0][0] < 0.0]
    check(not short and len(by(chains, "back")) == 3 and len(left) == 2 and len(right) == 1 and len(sides) == 3,
          "lanes rig counts: 3 back, 2 side on +X, 1 side on -X ({})".format(kinds_of(chains)))
    bangs = len(by(chains, "bangs"))
    check(made == {"back": 3, "ponytail": 0, "left": 2, "right": 1, "bangs": bangs} and bangs <= 1,
          "lanes rig counts: made counts them ({}; {:d} bangs built)".format(made, bangs))
    check(chains and all(chain.kind in hairrig.KINDS for chain in chains) and not by(chains, "ponytail"),
          "lanes rig counts: every kind known, no ponytail ({})".format([chain.kind for chain in chains]))


def test_lanes_left_right_apart():
    # One chain per side: no side chain holds hair of both sides.
    model_rig = rig()
    co, tris = rig_counts_hair()
    chains, _made, _short = hairlanes.plan(co, tris, model_rig, counts=counts(left=1, right=1))
    area = hairing.vertex_areas(co, tris)
    x = model_rig.model(co)[:, 0]
    mixed = []
    for chain in by(chains, "side"):
        own, at = area[chain.vertices], x[chain.vertices]
        mixed.append(float(min(own[at > 0.0].sum(), own[at < 0.0].sum()) / own.sum()))
    check(len(mixed) == 2 and max(mixed) <= 0.02,
          "lanes left and right: 2 side chains, each with at most 2 % of its area across X = 0 ({})".format(
              [round(m, 4) for m in mixed]))


def test_lanes_order_and_sheets():
    # Name order by the first joint's angle, then its X; an area's chains share a sheet, the sheets numbered in that
    # order; within a sheet the order runs from 0, +X first behind, front to back at the sides.
    model_rig = rig()
    co, tris = rig_counts_hair()
    chains, _made, _short = hairlanes.plan(co, tris, model_rig, counts=counts(back=3, left=2, right=1, bangs=1))
    describe_columns(chains)
    keys = [(round(hairrig._angle(model_rig, chain.joints[0]), 6), chain.joints[0][0]) for chain in chains]
    check(chains and keys == sorted(keys), "lanes order: the chains in name order")
    areas = {}
    for chain in chains:
        side = float(np.sign(chain.joints[0][0])) if chain.kind == "side" else 0.0
        areas.setdefault((chain.kind, side), []).append(chain)
    sheets = [{chain.sheet for chain in mine} for mine in areas.values()]
    check(all(len(mine) == 1 and min(mine) >= 0 for mine in sheets) and len(set.union(*sheets)) == len(areas),
          "lanes order: each area one sheet of its own ({})".format(kinds_of(chains)))
    seen = []
    for chain in chains:
        if chain.sheet not in seen:
            seen.append(chain.sheet)
    check(seen == list(range(len(seen))), "lanes order: the sheets numbered in list order ({})".format(seen))
    check(all(sorted(chain.order for chain in mine) == list(range(len(mine))) for mine in areas.values()),
          "lanes order: each sheet's order runs 0 to n - 1 ({})".format(kinds_of(chains)))
    back = by(chains, "back")
    check(len(back) == 3 and all(a.joints[0][0] > b.joints[0][0] for a, b in zip(back, back[1:])),
          "lanes order: behind, +X first ({})".format([round(float(chain.joints[0][0]), 3) for chain in back]))
    left = sorted(areas.get(("side", 1.0), []), key=lambda chain: chain.order)
    angles = [hairrig._angle(model_rig, chain.joints[0]) for chain in left]
    check(len(left) == 2 and angles == sorted(angles),
          "lanes order: on a side, front first ({})".format([round(a, 1) for a in angles]))


def test_lanes_area_without_hair():
    # The back sheet alone: the sides and bangs asked for build nothing.
    co, tris, _ids = back_sheet()
    chains, made, _short = hairlanes.plan(co, tris, rig(), counts=counts(left=2, right=2, bangs=2))
    check(made["left"] == made["right"] == made["bangs"] == 0 and made["back"] == len(by(chains, "back")) > 0
          and len(chains) == made["back"],
          "lanes without hair: no side or bangs chain, the back's built ({}, {})".format(made, kinds_of(chains)))


def test_lanes_short_ring():
    # Short hair: Bang Chains sectors around the whole head, a bangs chain each at most; Hair Type Ponytail ties
    # nothing.
    model_rig = rig()
    co, tris = short_hair()
    chains, made, short = hairlanes.plan(co, tris, model_rig, counts=counts(bangs=4))
    describe_columns(chains)
    print("      first joints' angles {}".format([round(hairrig._angle(model_rig, chain.joints[0]), 1)
                                                  for chain in chains]))
    check(short and all(chain.kind == "bangs" for chain in chains)
          and made["bangs"] == len(chains) == 4 and made["back"] == 0,
          "lanes short ring: short, 4 bangs chains ({}, {})".format(kinds_of(chains), made))
    # One sector each, the first straight in front, the next toward +X, then behind, then -X.
    ring = by(chains, "bangs")
    turn = [round(hairrig._angle(model_rig, chain.joints[0]), 1) for chain in ring]
    side = [round(float(model_rig.model(chain.joints[0][None])[0][0]), 3) for chain in ring]
    check(len(ring) == 4 and [chain.order for chain in ring] == [0, 1, 2, 3]
          and turn[0] < 45.0 and turn[2] > 135.0 and side[1] > 0.0 > side[3],
          "lanes short ring: in order the front, +X, behind, -X (angles from the front {}, X {})".format(turn, side))
    tied, tied_made, tied_short = hairlanes.plan(co, tris, model_rig, ponytail=True, counts=counts(bangs=4))
    check(tied_short and tied_made == made and len(tied) == len(chains)
          and all(a.kind == b.kind and np.array_equal(a.joints, b.joints) for a, b in zip(tied, chains)),
          "lanes short ring: Hair Type Ponytail ties nothing, the same chains ({})".format(kinds_of(tied)))


def test_lanes_loose_ties_nothing():
    # Hair Type Loose: the tube down the back (`long_hair`'s ponytail) is back hair.
    co, tris, _part, _names = long_hair()
    chains, made, _short = hairlanes.plan(co, tris, rig(), counts=counts(back=2))
    describe_columns(chains)
    check(not by(chains, "ponytail") and len(by(chains, "back")) == 2 and made["back"] == 2 and made["ponytail"] == 0,
          "lanes loose: no ponytail, 2 back chains ({}, {})".format(kinds_of(chains), made))


def test_lanes_ponytail():
    # Hair Type Ponytail: the tube down the back is the tail, its chains on it; two side by side, not crossing.
    co, tris, part, names = long_hair()
    tube = co[part == names.index("ponytail")]
    for n in (1, 2):
        chains, made, _short = hairlanes.plan(co, tris, rig(), ponytail=True, counts=counts(ponytail=n))
        describe_columns(chains)
        tails = by(chains, "ponytail")
        off = [float(nearest(np.vstack((chain.joints, chain.tip)), tube).max()) for chain in tails]
        check(len(tails) == n and made["ponytail"] == n and made["back"] == 0 and max(off, default=math.inf) <= 0.03,
              "lanes ponytail: {:d} asked, {:d} ponytail chains within 3 cm of the tube ({} m; {})".format(
                  n, len(tails), [round(o, 4) for o in off], made))
        if n == 2 and len(tails) == 2:
            check(not lanes_crossed(*tails) and tails[0].joints[0][0] > tails[1].joints[0][0],
                  "lanes ponytail: 2 side by side, +X first, not crossing ({})".format(
                      [round(float(chain.joints[0][0]), 3) for chain in tails]))


def test_lanes_budget():
    # Over the budget the bones get longer: as many chains, the bones within it or 2 per chain.
    model_rig = rig()
    co, tris = card_hair(300)
    asked = counts(back=4, left=2, right=2, bangs=6)
    free, _made, _short = hairlanes.plan(co, tris, model_rig, counts=asked)
    most = sum(len(chain.joints) for chain in free)
    # A budget halfway between the bones without one and two per chain: it bites, and it can be met.
    budget = (most + 2 * len(free)) // 2
    chains, _made, _short = hairlanes.plan(co, tris, model_rig, counts=asked, max_bones=budget)
    bones = [len(chain.joints) for chain in chains]
    check(2 * len(free) < budget < most and len(chains) == len(free) and min(bones, default=0) >= 2
          and sum(bones) <= budget,
          "lanes budget: {:d} bones without a budget, {:d} allowed: {:d} chains as without it ({:d}), bones {} ({:d}) "
          "within it".format(most, budget, len(chains), len(free), bones, sum(bones)))
    # A budget no rig can meet: every chain at two bones, none dropped.
    chains, _made, _short = hairlanes.plan(co, tris, model_rig, counts=asked, max_bones=len(free))
    bones = [len(chain.joints) for chain in chains]
    check(len(chains) == len(free) and all(b == 2 for b in bones),
          "lanes budget: {:d} allowed: {:d} chains, 2 bones each ({})".format(len(free), len(chains), bones))


def test_lanes_detect():
    # Detect: within the ranges, 3 or 4 back, 1 a side, only the areas with hair; short hair only bangs.
    model_rig = rig()
    co, tris = rig_counts_hair()
    found, short = hairlanes.detect(co, tris, model_rig)
    check(not short and all(hairrig.COUNT_RANGES[name][0] <= count <= hairrig.COUNT_RANGES[name][1]
                            for name, count in found.items())
          and found.get("back") in (3, 4) and found.get("left") == 1 and found.get("right") == 1
          and "ponytail" not in found,
          "lanes detect: 3 or 4 back, 1 a side, no ponytail ({})".format(found))
    co, tris, _ids = back_sheet()
    found, _short = hairlanes.detect(co, tris, model_rig)
    check(set(found) == {"back"}, "lanes detect: the back sheet alone has only back ({})".format(found))
    co, tris = short_hair()
    found, short = hairlanes.detect(co, tris, model_rig)
    check(short and set(found) == {"bangs"}, "lanes detect: short hair, only bangs ({}, {})".format(found, short))


def test_lanes_no_counts():
    # Without counts every area gets as many chains as Detect gives it.
    model_rig = rig()
    co, tris = rig_counts_hair()
    found, _short = hairlanes.detect(co, tris, model_rig)
    _chains, made, _short = hairlanes.plan(co, tris, model_rig)
    check(made == {name: found.get(name, 0) for name in hairrig.AREAS},
          "lanes no counts: as many chains as Detect gives ({} against {})".format(made, found))


def test_lanes_empty():
    # Nothing hangs, or no mesh: no chains, nothing detected.
    model_rig = rig()
    nothing = ([], dict.fromkeys(hairrig.AREAS, 0), False)
    cap_co, cap_tris, _ranges = join([hair.cap()])
    for name, (co, tris) in (("a cap alone", (cap_co, cap_tris)), ("no triangles", (cap_co, np.zeros((0, 3), int)))):
        check(hairlanes.plan(co, tris, model_rig) == nothing and hairlanes.detect(co, tris, model_rig) == ({}, False),
              "lanes empty: {:s} plans and detects nothing".format(name))


def test_lanes_moved_rig():
    # Hair whose armature has its head elsewhere: the same chains, moved.
    model_rig = rig()
    co, tris = rig_counts_hair()
    offset = np.array((0.3, -0.2, 0.1))
    asked = counts(back=3, left=2, right=1, bangs=1)
    chains, _made, _short = hairlanes.plan(co, tris, model_rig, counts=asked)
    moved, _made, _short = hairlanes.plan(co + offset, tris, model_rig.moved(offset), counts=asked)
    check(chains and len(moved) == len(chains) and all(
        a.kind == b.kind and a.joints.shape == b.joints.shape
        and np.allclose(a.joints, b.joints + offset, rtol=0.0, atol=1e-6)
        and np.allclose(a.tip, b.tip + offset, rtol=0.0, atol=1e-6) for a, b in zip(moved, chains)),
        "lanes moved rig: the same chains, moved by the offset ({} vs {})".format(kinds_of(moved), kinds_of(chains)))


def test_lanes_card_hair_time():
    # 3000 cards (66,000 vertices), counts detected: no slower than CARDS_TIME allows.
    model_rig = rig()
    co, tris = card_hair(3000)
    started = time.perf_counter()
    chains, _made, _short = hairlanes.plan(co, tris, model_rig)
    took = time.perf_counter() - started
    print("      plan: {:.3f} s for 3000 cards, {:d} vertices, {:d} chains".format(took, len(co), len(chains)))
    check(chains and took <= 2.0 * CARDS_TIME + 1.0,
          "lanes card hair time: {:.2f} s (at most {:.2f} s)".format(took, 2.0 * CARDS_TIME + 1.0))


def test_plan_layout_is_lanes():
    # plan_layout, plan_rig and detect_counts are the lane planner's; the lock planner is gone.
    model_rig = rig()
    co, tris = rig_counts_hair()
    asked = counts(back=3)
    chains, made, short = hairrig.plan_layout(co, tris, model_rig, counts=asked)
    lanes, lanes_made, lanes_short = hairlanes.plan(co, tris, model_rig, counts=asked)
    check(chains and len(chains) == len(lanes) and made == lanes_made and short == lanes_short and all(
        a.kind == b.kind and a.sheet == b.sheet and a.order == b.order and a.light == b.light
        and np.array_equal(a.joints, b.joints) and np.array_equal(a.tip, b.tip) for a, b in zip(chains, lanes)),
        "plan layout: the lane planner's chains, made and short ({}, {})".format(kinds_of(chains), made))
    first = hairrig.plan_rig(co, tris, model_rig, counts=asked)
    check(len(first) == len(chains) and all(np.array_equal(a.joints, b.joints) for a, b in zip(first, chains)),
          "plan layout: plan_rig gives its chains")
    found = hairrig.detect_counts(co, tris, model_rig)
    check(found == hairlanes.detect(co, tris, model_rig), "plan layout: detect_counts is Detect's ({})".format(found))
    check(not hasattr(hairrig, "find_locks"), "plan layout: the lock planner is gone")
    try:
        hairrig.plan_layout(co, tris, model_rig, spacing=0.0)
        refused = None
    except ValueError as error:
        refused = str(error)
    check(refused == "The chain spacing must be above 0", "plan layout: refuses a spacing of 0 ({})".format(refused))


def real_back_lanes(name, path):
    """A real hair (`hair_models.decode`) planned with Back 3 and Hair Type Loose: its 3 back chains of equal bones,
    rows within 2 cm, none crossing, and every chain's joints within 3 cm of its hair."""
    if not os.path.isfile(path):
        print("skip: no {:s} hair at {:s}".format(name, path))
        return
    import hair_models
    with open(path, "rb") as handle:
        points, tris, _held = hair_models.decode(handle.read())
    model_rig = rig(MIQOTE)
    cap = hairing.cap_depth(points, tris, model_rig)
    started = time.perf_counter()
    chains, made, _short = hairlanes.plan(points, tris, model_rig, cap=cap, counts=counts(back=3))
    took = time.perf_counter() - started
    gaps = [float(gap) for gap in model_rig.scalp_gap(np.array([chain.joints[0] for chain in chains]).reshape(-1, 3),
                                                      cap)]
    print("      plan: {:.3f} s for {:d} vertices; {}".format(took, len(points), made))
    chain_table(chains, gaps)
    back = by(chains, "back")
    bones = [len(chain.joints) for chain in back]
    apart = max((float(np.abs(a.joints[:, 2] - b.joints[:, 2]).max()) for a in back for b in back
                 if len(a.joints) == len(b.joints)), default=math.inf)
    check(len(back) == 3 and len(set(bones)) == 1 and apart <= 0.02,
          "{:s}: 3 back chains ({:d}) of equal bones ({}), rows within 2 cm ({:.4f} m)".format(
              name.lower(), len(back), bones, apart))
    check(len(back) == 3 and not any(lanes_crossed(a, b) for a, b in zip(back, back[1:])),
          "{:s}: no back chains crossed".format(name.lower()))
    far = [off_hair(chain, points) for chain in chains]
    check(chains and max(far) <= 0.03, "{:s}: every joint within 3 cm of its hair (up to {:.4f} m)".format(
        name.lower(), max(far, default=math.nan)))


def test_lanes_maeve():
    real_back_lanes("Maeve", MAEVE)


def test_lanes_briar():
    real_back_lanes("Briar", BRIAR)


def card_hair(count, seed=0):
    """`hair.cards` as one mesh: (points, triangles)."""
    co, tris, _ranges = join(hair.cards(count, seed=seed))
    return co, tris


# -----------------------------------------------------------------------------
# What hangs (by the scalp) and where chains start

def chain_starts(chains, co, tris, model_rig):
    """Where each chain starts: (its first joint's distance from the scalp beyond the hair's cap depth, how far in Z
    its first joint lies from where its hair may start), and whether that is at the scalp. Its hair may start at the
    lower of its own hair's top and the scalp's lower edge in its hair's direction (the area-weighted median of
    `hairlanes.Hair.edge` over its vertices, the finite ones): hanging hair's chains start no higher than that edge
    (the design's section 2, BELOW_EDGE), and hair that never comes within ROOT_REACH of the scalp starts at its top.
    A chain starts at the scalp when its first joint lies within ROOT_REACH of the scalp or within 3 cm in Z of
    that start."""
    cap = hairing.cap_depth(co, tris, model_rig)
    measured = hairlanes.measure(co, tris, model_rig, cap)
    starts = []
    for chain in chains:
        gap = float(model_rig.scalp_gap(chain.joints[:1], cap)[0])
        on = np.isin(measured.index, chain.vertices)
        start = float(measured.at[on, 2].max())
        edge, weight = measured.edge[on], measured.area[on] + 1e-12
        known = np.isfinite(edge)
        if known.any():
            start = min(start, float(hairlanes._quantiles(edge[known], weight[known], 0.5)))
        down = abs(float(model_rig.model(chain.joints[:1])[0][2]) - start)
        starts.append((gap, down, gap <= hairrig.ROOT_REACH or down <= 0.03))
    return starts


def describe_starts(starts):
    return [(round(gap, 4), round(down, 4)) for gap, down, _ok in starts]


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


def fringe_and_neck():
    """`neck_hair` (no loose layer) with `flat_fringe`'s fringe in front, its own loose part: hair that isn't short,
    so its fringe hangs by its drop. Returns (points, triangles, {"cap", "sheet", "fringe": indices})."""
    fringe = hair.patch(np.arange(35.0, 116.0, 5.0), np.arange(165.0, 196.0, 7.5), 0.012, MIQOTE)
    co, tris, ranges = join(neck_parts() + [fringe])
    return co, tris, dict(zip(("cap", "sheet", "fringe"), ranges))


def bun_hair():
    """The scalp cap on c0801 and a bun: a tube 3 cm wide standing 10 cm up from the crown, its own loose
    part. Returns (points, triangles, {"cap": indices, "bun": indices})."""
    crown = hair.top(MIQOTE)
    co, tris, (cap_ids, bun_ids) = join([hair.cap(race=MIQOTE),
                                         hair.tube((0.0, 0.0, crown + 0.01), (0.0, 0.0, crown + 0.11), 0.03)])
    return co, tris, {"cap": cap_ids, "bun": bun_ids}


def off_hair(chain, co):
    """How far a chain's joints and tip lie from its hair at most (meters)."""
    own = co[chain.vertices]
    return max(float(np.min(np.linalg.norm(own - point, axis=1))) for point in np.vstack((chain.joints, chain.tip)))


def test_lanes_flat_fringe():
    # A fringe lying 1.2 cm off the forehead, with the neck sheet behind it, gets a bangs chain made of it, rooted at
    # the hairline (`chain_starts`). Alone on the cap the fringe is short hair, whose levels are the distance from
    # the scalp: its middle line runs 2.99 cm, under MIN_CHAIN_LENGTH, so it gets no chain (the planner's rule, not
    # a fault). That is 0.1 mm under the limit, so the case is only checked to be short hair and to plan without
    # raising, not to build nothing.
    model_rig = rig(MIQOTE)
    alone, made, short = hairlanes.plan(*flat_fringe()[:2], model_rig)
    check(short and sum(made.values()) == made["bangs"] == len(alone),
          "lanes flat fringe: the fringe alone is short hair, nothing but bangs planned ({})".format(made))
    co, tris, parts = fringe_and_neck()
    chains, made, _short = hairlanes.plan(co, tris, model_rig)
    describe_columns(chains)
    bangs = [chain for chain in chains if chain.kind == "bangs"]
    check(made["bangs"] == len(bangs) >= 1 and all(np.isin(chain.vertices, parts["fringe"]).all() for chain in bangs),
          "lanes flat fringe: the fringe gets a bangs chain made of it ({}, {})".format(kinds_of(chains), made))
    starts = chain_starts(bangs, co, tris, model_rig)
    check(bangs and all(ok for _gap, _down, ok in starts),
          "lanes flat fringe: ...rooted at the hairline (from the scalp, from its start: {} m)".format(
              describe_starts(starts)))


def test_lanes_neck_layer():
    # A sheet down the neck from the nape: its back chains start at the scalp (`chain_starts`) at the nape hairline,
    # not low on the neck.
    model_rig = rig(MIQOTE)
    co, tris, _parts = neck_hair()
    chains, made, _short = hairlanes.plan(co, tris, model_rig)
    describe_columns(chains)
    starts = chain_starts(chains, co, tris, model_rig)
    check(chains and made["back"] == len(chains) and all(ok for _gap, _down, ok in starts),
          "lanes neck layer: back chains start at the scalp ({}, {} m)".format(made, describe_starts(starts)))
    heights = [float(chain.joints[0][2]) for chain in chains]
    check(chains and min(heights) >= 1.36, "lanes neck layer: ...at the nape hairline, not low on the neck (z {})".format(
        [round(z, 3) for z in heights]))


def test_lanes_neck_only_layer():
    # The neck sheet with its lower half doubled 5 mm inside it, a loose layer lying wholly on the neck: no chain
    # starts on it. As many chains as without the layer, each starting at the scalp at the nape hairline.
    model_rig = rig(MIQOTE)
    co, tris, parts = neck_hair(loose=True)
    cap = hairing.cap_depth(co, tris, model_rig)
    print("      the loose layer is {:.3f} m from the scalp at its closest".format(
        float(model_rig.scalp_gap(co[parts["layer"]], cap).min())))
    chains, _made, _short = hairlanes.plan(co, tris, model_rig)
    describe_columns(chains)
    without, _made, _short = hairlanes.plan(*neck_hair()[:2], model_rig)
    starts = chain_starts(chains, co, tris, model_rig)
    heights = [float(chain.joints[0][2]) for chain in chains]
    check(chains and len(chains) == len(without) and all(ok for _gap, _down, ok in starts) and min(heights) >= 1.36,
          "lanes neck only layer: no chain of its own ({:d}, {:d} without), every chain at the scalp ({} m) at the "
          "nape hairline (z {})".format(len(chains), len(without), describe_starts(starts),
                                        [round(z, 3) for z in heights]))
    check(len(chains) == len(without) and all(np.allclose(a.joints[0], b.joints[0]) for a, b in zip(chains, without)),
          "lanes neck only layer: the chains start where they do without it ({})".format(
              [np.round(a.joints[0] - b.joints[0], 4).tolist() for a, b in zip(chains, without)]))
    held = np.isin(parts["layer"], np.concatenate([chain.vertices for chain in chains if chain.kind == "back"]
                                                  or [np.zeros(0, dtype=np.int64)]))
    check(bool(held.all()), "lanes neck only layer: the layer's hair is in the back chains ({:d} of {:d} vertices)".format(
        int(held.sum()), len(held)))


def test_lanes_cap_rim():
    # A cap alone, and a cap whose rim behind the head reaches just below the nape hairline (it hangs, but too little
    # for a chain): no chain on the neck.
    model_rig = rig(MIQOTE)
    co, tris, _ranges = join([hair.cap(race=MIQOTE)])
    check(not hairlanes.plan(co, tris, model_rig)[0], "lanes cap rim: a cap alone has no chain")
    cap, faces, ring, azimuth = back_ring(24, MIQOTE)
    tops = cap[ring[np.abs(azimuth[ring]) <= 90.0 + 1e-6]]
    co, tris, (_cap_ids, rim_ids) = join([(cap, faces), hair.down_neck(tops, 1.41, 7, 0.01, MIQOTE)])
    rim = rim_ids[-len(tops):]
    share = model_rig.scalp(co[rim])
    chains, _made, _short = hairlanes.plan(co, tris, model_rig)
    check(share.min() < hairing.SCALP_HALF and not chains,
          "lanes cap rim: a rim reaching below the hairline (scalp share {:.2f} to {:.2f}) gets no chain ({})".format(
              float(share.min()), float(share.max()), kinds_of(chains)))


def test_lanes_roots_near_scalp():
    # Long hair, long hair in one piece, card hair and five strands: every chain starts at the scalp (`chain_starts`:
    # within ROOT_REACH of it, or where its hair may start), none on the neck or face.
    model_rig = rig()
    co, tris, _part, _names = long_hair()
    one_co, one_tris, _parts = one_piece_long_hair()
    card_co, card_tris = card_hair(300)
    strands = join([hair.strip((x, 0.08, 1.53), (x, 0.10, 1.23), 0.02, (1.0, 0.0, 0.0))
                    for x in (-0.12, -0.06, 0.0, 0.06, 0.12)])[:2]
    for name, points, triangles in (("long hair", co, tris), ("one piece", one_co, one_tris),
                                    ("card hair", card_co, card_tris), ("strands", *strands)):
        chains, _made, _short = hairlanes.plan(points, triangles, model_rig)
        starts = chain_starts(chains, points, triangles, model_rig)
        check(chains and all(ok for _gap, _down, ok in starts),
              "lanes roots near the scalp: {:s}: every chain starts at the scalp (from the scalp, from its start: {} m; "
              "{})".format(name, describe_starts(starts), kinds_of(chains)))


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


def chain_table(chains, gaps):
    """Print ``chains`` as a table, with ``gaps`` (each first joint's distance from the scalp)."""
    print("      chain  kind      light  sheet  order  bones  root x   root y   root z   gap      angle  length")
    for number, (chain, gap) in enumerate(zip(chains, gaps)):
        line = np.vstack((chain.joints, chain.tip))
        print("      {:02d}     {:9s} {!s:6} {:5d}  {:5d}  {:d}      {:7.4f}  {:7.4f}  {:7.4f}  {:.4f}  {:5.1f}  "
              "{:.3f}".format(number + 1, chain.kind, chain.light, chain.sheet, chain.order, len(chain.joints),
                              *chain.joints[0], gap, chain.angle,
                              float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())))


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
    names). With ``fixture`` (a c0801 fixture: `bun_hair`, `fringe_and_neck` or `neck_hair`), that hair instead,
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
    # The lane planner puts every chain in its area's sheet (`hairlanes.plan`); only a chain the user adds has none
    # (below). The lock planner gave the long hair's tube a ponytail chain without a sheet.
    check(columns and columns == planned and len({sheet for sheet, _order in columns.values()}) >= 2,
          "columns: every chain of the long hair is in its area's sheet, two areas or more ({})".format(planned))
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
    # The long hair's areas are a sheet each: put their chains in one sheet, orders 0 and on, and delete order 0.
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
    # The fringe with the neck sheet behind it: alone on the cap the fringe is short hair and gets no chain
    # (test_lanes_flat_fringe), so there would be no ramp.
    _obj, armature, plans, _bones, co, tris, part, names = rig_scene(fringe_and_neck)
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
    # The fringe with the neck sheet behind it: the lone fringe gets no chain (test_lanes_flat_fringe), and with no
    # chain the cap would stay trivially.
    for fixture in (fringe_and_neck, neck_hair, bun_hair):
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
    # The vertices chain 01 takes (the neck sheet has one back chain now; with more, the others take some of those
    # near it).
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
    # Past the old first joint (half hanging or more), above the new one's ramp, and hanging: the chain starts at the
    # scalp's lower edge, and the nape hair just under its first joint that is still over the scalp (scalp share
    # SCALP_HALF or more) is the head's, not the chain's.
    hanging_hair = before.scalp(co) < hairing.SCALP_HALF
    between = ones & hanging_hair & (old_along > 0.0) & (new_along < -0.5 * hairing.RAMP_SHARE * new_first)
    # Chain 01's own weight there: with more chains on the sheet, the moved joint also moves chain 01's share, so
    # some of that hair goes to another chain, whose ramp didn't move.
    one_old, one_new = own_weight(old_names, old, 1), own_weight(new_names, new, 1)
    print("      first bone {:.3f} m; {:d} vertices between the old and the new joint".format(float(first[0]),
                                                                                       int(between.sum())))
    check(between.any() and float(one_new[between].max()) <= 1e-6 and float(one_old[between].mean()) >= 0.3,
          "edited bone: the hair between the old and the new first joint doesn't hang from chain 01 now ({:.3f} at "
          "most, {:.3f} on average before)".format(float(one_new[between].max()) if between.any() else math.nan,
                                                    float(one_old[between].mean()) if between.any() else math.nan))


# -----------------------------------------------------------------------------
# Chains the user adds along a selection, and deletes

def test_selection_chain():
    model_rig = rig()
    # One strip: the left lock of hair.parts (top (0.085,-0.035,1.53), bottom (0.09,-0.045,1.28), 13 rows x 2).
    co, tris, (lock,) = join([hair.strip((0.085, -0.035, 1.53), (0.09, -0.045, 1.28), 0.025, (0.0, 1.0, 0.0))])
    plan = hairrig.plan_selection(co, tris, model_rig, lock)
    check(plan is not None and abs(plan.joints[0][2] - 1.53) < 0.02, "selection: the chain starts at the strip's top")
    check(plan is not None and abs(plan.tip[2] - 1.28) < 0.02, "selection: ...and ends at its bottom")
    check(plan is not None and len(plan.joints) == 3, "selection: 25 cm gets 3 bones by length")
    check(plan is not None and np.allclose(plan.joints, plan.middle), "selection: the joints lie on its own line")
    check(plan is not None and plan.kind == "side" and plan.sheet == -1 and plan.order == -1,
          "selection: side hair, no column")
    # One vertex row of the strip (every second vertex) runs down the same hair: its line lies half the strip's
    # width (1.25 cm) to one side, so within 2 cm of the whole strip's (less than its 2.5 cm width).
    row = lock[::2]
    single = hairrig.plan_selection(co, tris, model_rig, row)
    check(single is not None and plan is not None and float(np.abs(single.joints - plan.joints).max()) < 0.02,
          "selection: one vertex row runs down the same hair")
    # A set bone count.
    five = hairrig.plan_selection(co, tris, model_rig, lock, bones=5)
    check(five is not None and len(five.joints) == 5, "selection: Bones 5 gives 5 joints")
    one = hairrig.plan_selection(co, tris, model_rig, lock, bones=1)
    check(one is not None and len(one.joints) == 1 and abs(one.tip[2] - 1.28) < 0.02, "selection: Bones 1 gives one bone")
    # Too short: the top row only.
    check(hairrig.plan_selection(co, tris, model_rig, lock[:2]) is None, "selection: shorter than 3 cm plans nothing")
    check(hairrig.plan_selection(co, tris, model_rig, np.array([], dtype=np.int64)) is None,
          "selection: nothing selected plans nothing")


def test_selection_two_cards():
    model_rig = rig()
    a = hair.strip((-0.03, 0.10, 1.52), (-0.03, 0.14, 1.20), 0.02, (1.0, 0.0, 0.0))
    b = hair.strip((0.03, 0.10, 1.52), (0.03, 0.14, 1.20), 0.02, (1.0, 0.0, 0.0))
    co, tris, (ia, ib) = join([a, b])
    plan = hairrig.plan_selection(co, tris, model_rig, np.concatenate((ia, ib)))
    check(plan is not None and abs(plan.joints[0][0]) < 0.005 and abs(plan.joints[0][2] - 1.52) < 0.02,
          "two cards: one chain rooted between their tops")
    check(plan is not None and abs(plan.tip[0]) < 0.005 and plan.kind == "back", "two cards: back hair down the middle")


def test_selection_stacked_cards():
    model_rig = rig()
    # One lock built of two unwelded pieces, one below the other: the chain runs from the upper one's top down to the
    # lower one's end, not from both tops at once.
    upper = hair.strip((0.085, -0.035, 1.53), (0.087, -0.040, 1.38), 0.025, (0.0, 1.0, 0.0))
    lower = hair.strip((0.087, -0.040, 1.375), (0.09, -0.045, 1.23), 0.025, (0.0, 1.0, 0.0))
    co, tris, (iu, il) = join([upper, lower])
    plan = hairrig.plan_selection(co, tris, model_rig, np.concatenate((iu, il)))
    check(plan is not None and abs(plan.joints[0][2] - 1.53) < 0.02 and abs(plan.tip[2] - 1.23) < 0.02,
          "stacked cards: one chain from the upper card's top to the lower one's end ({})".format(
              None if plan is None else (round(float(plan.joints[0][2]), 3), round(float(plan.tip[2]), 3))))


def test_add_chain():
    obj, armature, plans, _bones, co, tris, part, names = rig_scene()
    before = state_chains(armature)
    stored = armature.magic_fit_hair_rig.bones_hash
    lock = np.flatnonzero(part == names.index("lock_l"))
    plan = hairrig.plan_selection(co, tris, rig(), lock)
    plan.kind = "ponytail"   # the kind given is the one kept, not sync_rig's guess
    added = hairrig.add_chain(bpy.context, armature, plan, hair.RACE)
    number = max(int(b[0][10:12]) for _k, b in before) + 1
    check(added == [hairrig.bone_name(number - 1, k) for k in range(len(plan.joints))],
          "add chain: bones named for the next chain number ({})".format(added))
    after = state_chains(armature)
    check(after[:len(before)] == before and after[-1] == ("ponytail", added), "add chain: the others stay, the new one is last")
    check(armature.data.bones[added[0]].parent.name == hairing.HEAD, "add chain: it hangs from j_kao")
    check(np.linalg.norm(world(armature, armature.data.bones[added[0]].head_local) - plan.joints[0]) < 1e-5,
          "add chain: its first bone starts on the plan's root")
    check(armature.magic_fit_hair_rig.bones_hash == stored and hairrig.rig_edited(armature),
          "add chain: counts as a user edit (hash kept)")
    check(json_matches_state(armature) and chains_are_clean(armature), "add chain: the rig property lists it")
    check(armature.magic_fit_hair_rig.active == len(after) - 1, "add chain: the new chain is picked in the list")


def test_add_chain_keeps_edits():
    obj, armature, _plans, _bones, co, tris, part, names = rig_scene()
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    top = armature.data.edit_bones.new("Bone")
    top.head, top.tail = (0.0, 0.10, 1.55), (0.0, 0.12, 1.42)
    top.parent = armature.data.edit_bones[hairrig.BONE_PREFIX + "01_a"]
    bpy.ops.object.mode_set(mode='OBJECT')
    plan = hairrig.plan_selection(co, tris, rig(), np.flatnonzero(part == names.index("lock_r")))
    hairrig.add_chain(bpy.context, armature, plan, hair.RACE)
    check("Bone" not in armature.data.bones and chains_are_clean(armature),
          "add chain: an unsynced bone the user added is read back and named first")


def test_delete_chain():
    obj, armature, _plans, _bones, _co, _tris, _part, _names = rig_scene()
    before = state_chains(armature)
    stored = armature.magic_fit_hair_rig.bones_hash
    # Chain 02 gets a branch hanging from its last bone: deleting 02 hands the branch to j_kao.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    last = armature.data.edit_bones[before[1][1][-1]]
    branch = armature.data.edit_bones.new("j_ex_hair_90_a")
    branch.head, branch.tail, branch.parent = last.tail, last.tail - Vector((0.0, 0.0, 0.05)), last
    bpy.ops.object.mode_set(mode='OBJECT')
    hairrig.sync_rig(bpy.context, armature, hair.RACE)
    armature.magic_fit_hair_rig.bones_hash = stored
    number = hairrig.delete_chain(bpy.context, armature, 1, hair.RACE)
    after = state_chains(armature)
    check(number == before[1][1][0][10:12], "delete chain: reports its number ({})".format(number))
    check(not any(name in armature.data.bones for name in before[1][1]), "delete chain: its bones are gone")
    check([c for c in after if c[1] != ["j_ex_hair_90_a"]] == [before[0]] + before[2:],
          "delete chain: the other chains stay as they were")
    check(armature.data.bones["j_ex_hair_90_a"].parent.name == hairing.HEAD,
          "delete chain: a chain hanging from it now hangs from j_kao")
    check(armature.magic_fit_hair_rig.bones_hash == stored and json_matches_state(armature),
          "delete chain: counts as a user edit, the rig property follows")
    try:
        hairrig.delete_chain(bpy.context, armature, len(after), hair.RACE)
        refused = None
    except hairing.HairError as error:
        refused = str(error)
    check(refused == "No chain {:d} in the list".format(len(after) + 1), "delete chain: refuses a chain not in the list")


def test_delete_chain_unsynced():
    obj, armature, _plans, _bones, _co, _tris, _part, _names = rig_scene()
    before = state_chains(armature)
    # A bone the user extruded from chain 02's tip in Edit Mode, not read back yet, goes with the chain.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    last = armature.data.edit_bones[before[1][1][-1]]
    extruded = armature.data.edit_bones.new(last.name + ".001")
    extruded.head, extruded.tail, extruded.parent = last.tail, last.tail - Vector((0.0, 0.0, 0.05)), last
    extruded.use_connect = True
    bpy.ops.object.mode_set(mode='OBJECT')
    number = hairrig.delete_chain(bpy.context, armature, 1, hair.RACE)
    left = [bone.name for bone in armature.data.bones if bone.name.startswith(hairrig.BONE_PREFIX + number)]
    check(not left and state_chains(armature) == [before[0]] + before[2:],
          "delete chain: a bone extruded from it and not read back goes with it ({})".format(left))


def main():
    magic_fit.register()
    try:
        for test in (test_lanes_areas_by_where_it_hangs, test_lanes_forward_card_is_side, test_lanes_one_piece_is_split,
                     test_lanes_long_front_lock_is_side, test_lanes_middle_part, test_lanes_short,
                     test_lanes_ponytail_areas,
                     test_lanes_nothing_hangs, test_lanes_count_and_order, test_lanes_never_cross,
                     test_lanes_rows_shared, test_lanes_mirrored, test_lanes_u_bottom, test_lanes_fan_and_narrow,
                     test_lanes_clumps, test_lanes_start_at_scalp_edge, test_lanes_bun, test_lanes_check_keeps_rows,
                     test_lanes_short_followed_hair, test_lanes_joints_on_one_spot, test_lanes_bone_length,
                     test_lanes_sweeping_hair, test_lanes_separate_strand,
                     test_lanes_sweep_to_end, test_lanes_coarse_strands_apart, test_lanes_card_behind,
                     test_lanes_touching_cards, test_lanes_split_before_sweep,
                     test_lanes_swept_time, test_lanes_flat_ring,
                     test_lanes_double_sided,
                     test_lanes_detect_width,
                     test_lanes_rig_counts, test_lanes_left_right_apart, test_lanes_order_and_sheets,
                     test_lanes_area_without_hair, test_lanes_short_ring, test_lanes_loose_ties_nothing,
                     test_lanes_ponytail, test_lanes_budget, test_lanes_detect, test_lanes_no_counts, test_lanes_empty,
                     test_lanes_moved_rig, test_lanes_card_hair_time, test_plan_layout_is_lanes, test_lanes_maeve,
                     test_lanes_briar, test_lanes_flat_fringe, test_lanes_neck_layer, test_lanes_neck_only_layer,
                     test_lanes_cap_rim, test_lanes_roots_near_scalp,
                     test_maeve_weights, test_armature_rig, test_weights_follow_locks,
                     test_edited_rig_is_kept, test_branch_starts_chain, test_chain_collection, test_bones_hash,
                     test_kind_change_writes_json,
                     test_state_keeps_light, test_state_keeps_columns, test_old_rig_state, test_chain_along,
                     test_ramp_from_first_joint,
                     test_cap_stays, test_bun_light, test_own_neck_weight, test_ramp_follows_edited_bone,
                     test_selection_chain, test_selection_two_cards, test_selection_stacked_cards, test_add_chain,
                     test_add_chain_keeps_edits, test_delete_chain, test_delete_chain_unsynced):
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
