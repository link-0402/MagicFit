# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless tests for the reference data Hair Weights reads (hair_reference.npz) and the tools that write it:
the game's hair models decoded (tools/hair_models.py) and each race's scalp map (head_scalp, from
tools/make_scalp_map.py).

The tests reading game hair models look in MAGIC_FIT_HAIR_MODELS (default %TEMP%\\magic_fit_hair_models,
written by tools/extract_hair_models.ps1) and are skipped when it is missing. The must-haves read the real
test hair (DOUX Maeve, c0801 hair 108) from MAGIC_FIT_MAEVE and are skipped without it.

Run from the repository root:
    blender -b --factory-startup --python tests/test_hair_reference.py
"""

import os
import sys
import tempfile
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import magic_fit  # noqa: E402
from magic_fit import hairing  # noqa: E402

FAILURES = []
HAIR_MODELS = os.environ.get("MAGIC_FIT_HAIR_MODELS", os.path.join(tempfile.gettempdir(), "magic_fit_hair_models"))
MAEVE = os.environ.get("MAGIC_FIT_MAEVE", r"G:\Penumbra\DOUX Maeve\chara\human\c0801\obj\hair\h0108\model"
                                          r"\c0801h0108_hir.mdl")
# The races the scalp map is tuned and tested on: Miqo'te, Au Ra and Viera women.
SUPPORTED = ("c0801", "c1401", "c1801")


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def rig(race):
    return hairing.Reference.get().rig(race, 0)


def hair_model(race, number):
    """The bytes of the game's hair model ``number`` of ``race``, or None (and a skip line) without it."""
    path = os.path.join(HAIR_MODELS, race, "{:s}h{:04d}_hir.mdl".format(race, number))
    if not os.path.isfile(path):
        print("skip: no game hair model at " + path)
        return None
    with open(path, "rb") as handle:
        return handle.read()


def test_decode():
    data = hair_model("c0801", 1)
    if data is None:
        return
    import hair_models
    points, tris, held = hair_models.decode(data)
    print("      c0801h0001: {:d} vertices, {:d} triangles, {:.0%} held".format(len(points), len(tris), held.mean()))
    check(points.shape[1] == 3 and len(points) > 1000, "decode: positions decoded")
    used = np.zeros(len(points), dtype=bool)
    used[tris.ravel()] = True
    check(tris.min() >= 0 and tris.max() < len(points) and used.all(),
          "decode: triangles index the positions, and use every one")
    check(0.2 < held.mean() < 1.0, "decode: some hair on j_kao, some on hair bones")
    check(abs(points[:, 2].mean() - rig("c0801").by_name["j_kao"].head[2]) < 0.15,
          "decode: Blender space, Z up, near the head")


def test_scalp_supported_races():
    reference = hairing.Reference.get()
    with np.load(hairing.REFERENCE_FILE, allow_pickle=False) as data:
        stored = data["head_scalp"] if "head_scalp" in data.files else None
    check(stored is not None and stored.shape == (18, 800) and stored.dtype == np.float32
          and float(stored.min()) >= 0.0 and float(stored.max()) <= 1.0,
          "scalp: head_scalp (18, 800) float32 within 0..1")
    for race in SUPPORTED:
        head = reference.head(race)
        share = head.scalp_share
        p = head.surface
        kao = rig(race).by_name[hairing.HEAD].head
        below = p[:, 2].max() - p[:, 2]
        crown = share[p[:, 2] >= np.quantile(p[:, 2], 0.95)].mean()
        forehead = share[(np.abs(p[:, 0]) < 0.03) & (p[:, 1] < kao[1]) & (below >= 0.11) & (below <= 0.16)].mean()
        behind = p[:, 1] > kao[1]
        neck = share[behind & (p[:, 2] <= np.quantile(p[behind, 2], 0.05))].mean()
        check(crown > 0.8 and forehead < hairing.SCALP_HALF and neck < hairing.SCALP_HALF,
              "scalp: {:s} crown {:.2f}, forehead {:.2f}, neck {:.2f}".format(race, crown, forehead, neck))


def test_scalp_symmetric():
    head = hairing.Reference.get().head("c0801")
    p = head.surface
    mirror = p * (-1.0, 1.0, 1.0)
    nearest = np.argmin(((mirror[:, None, :] - p[None, :, :]) ** 2).sum(2), axis=1)
    difference = float(np.abs(head.scalp_share - head.scalp_share[nearest]).max())
    check(head.scalp_share.any() and difference < 0.05,
          "scalp: c0801 symmetric left to right (differs by {:.3f} at most)".format(difference))


def test_scalp_floor():
    """Nothing lower than the scalp goes in its direction is scalp: below the lowest scalp head point of its bin
    (by more than SCALP_BELOW) a point's share is 0, however its nearest head points score."""
    head = hairing.Reference.get().head("c0801")
    on = head.scalp_share >= hairing.SCALP_HALF
    lowest = np.flatnonzero(on)[np.argmin(head.surface[on, 2])]
    point = head.surface[lowest]
    check(head.scalp(point[None])[0] > 0.0 and head.scalp((point - (0.0, 0.0, 0.005))[None])[0] > 0.0,
          "scalp floor: the lowest scalp point keeps its share, and so does 5 mm below it")
    check(head.scalp((point - (0.0, 0.0, 0.02))[None])[0] == 0.0, "scalp floor: 2 cm below it, none")
    check(np.allclose(head.axis, rig("c0801").by_name[hairing.HEAD].head[:2]),
          "scalp floor: measured around the head bone's vertical axis")


def test_maeve_must_haves():
    """On the real test hair: the fringe hangs, the cap and the back and side hairlines stay, and what lies on
    the neck hangs; by rows of the hair, each its area-weighted share hanging."""
    if not os.path.isfile(MAEVE):
        print("skip: no Maeve hair at " + MAEVE)
        return
    import hair_models
    with open(MAEVE, "rb") as handle:
        points, tris, _held = hair_models.decode(handle.read())
    area = hairing.vertex_areas(points, tris)
    maeve = rig("c0801")
    head = maeve.head_field
    kao = maeve.by_name[hairing.HEAD].head
    distance = maeve.head_distance(points)
    share = maeve.scalp(points)
    cap = hairing.cap_depth(points, tris, maeve)
    hangs = (share < hairing.SCALP_HALF) | (distance > cap + hairing.STAND_MARGIN)
    print("      cap depth {:.3f} m".format(cap))

    def row(mask, name, low=None, high=None):
        if not mask.any():
            return
        hanging = float(np.average(hangs[mask], weights=area[mask] + 1e-12))
        text = "maeve: {:s} hangs {:.3f} (scalp share {:.2f})".format(
            name, hanging, float(np.average(share[mask], weights=area[mask] + 1e-12)))
        if low is None and high is None:
            print("      " + text)
            return
        wanted = (low is None or hanging >= low) and (high is None or hanging <= high)
        check(wanted, text + (", >= {:.2f}".format(low) if low is not None else ", <= {:.2f}".format(high)))

    below = np.round((head.surface[:, 2].max() - points[:, 2]) * 100.0, 6)   # cm
    front = (points[:, 1] < kao[1] - 0.04) & (np.abs(points[:, 0]) < 0.05)
    for top in range(0, 8, 2):
        row(front & (below >= top) & (below < top + 2), "front {:d}-{:d} cm below the top".format(top, top + 2),
            high=0.05)
    for top in range(10, 18, 2):
        row(front & (below >= top) & (below < top + 2), "fringe {:d}-{:d} cm below the top".format(top, top + 2),
            low=0.8)
    height = np.round(points[:, 2] * 100.0, 6)   # cm
    back = points[:, 1] > kao[1] + 0.02
    side = (np.abs(points[:, 0]) > 0.05) & (points[:, 1] < kao[1])
    for name, mask, stays in (("back", back, 145), ("side", side, 142)):
        for upper in range(160, 110, -3):
            rows = mask & (height < upper) & (height >= upper - 3)
            label = "{:s} z {:.2f}-{:.2f}".format(name, (upper - 3) / 100.0, upper / 100.0)
            if upper - 3 >= stays:
                row(rows, label, high=0.05)
            elif name == "back" and upper == 136:
                row(rows, label, low=0.85)
            elif upper <= 136:
                row(rows, label, low=0.95)
            elif name == "side":
                row(rows, label)   # where the side locks start: shown, not checked


def main():
    magic_fit.register()
    try:
        for test in (test_decode, test_scalp_supported_races, test_scalp_symmetric, test_scalp_floor,
                     test_maeve_must_haves):
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
