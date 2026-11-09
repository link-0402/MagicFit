"""How Own Skeleton's chains lie on many hairs: `hairrig.plan_rig` at a few Chain Spacing widths, measured for one
checkout of Magic Fit and set beside another checkout's run.

    blender-run.ps1 -Script tools\\hair_layout_check.py -- --checkout <path> --out <file.json> [--ref <file.json>]
        [--widths 0.06,0.08,0.10] [--only <label>,...] [--hair-dir <folder>]

It imports `magic_fit` from ``--checkout`` (default: this repository) before this repository's tools/hair_models and
tools/fit_hair_matching (which puts this repository's root on sys.path), and prints the `magic_fit` it measured.

Hairs, each on its race's rig without a hair skeleton and with its own `hairing.cap_depth`:
- maeve and briar, the real test hairs of tests/test_hair_rig_core.py (c0801);
- ``--hair-dir`` (default bench_data/hair)/*.npz, each with world space ``co``, ``tris`` and ``race``;
- the hair models active in the Penumbra collection "Lucina" (`fit_hair_matching.modded_hairs`).
Hairs that don't load, or whose rig fails, are skipped and counted.

Neighbors are the chains of one kind (side hair also by side of the head), ordered by their roots' arc around the
head: the angle (`polar`) about the group's area-weighted mean, times the group's area-weighted mean root radius.
Per hair and width it measures chains and bones per kind; each chain's drift (the largest arc of its joints and tip
from its first joint), off hair (the farthest of its joints and tip from its own vertices), sharpest bend between two
bones, and shoulder (on back and side chains that cross the shoulders' height, `j_sako_l/r`: the distance along the
chain from there to the nearest joint); crossing neighbors (their arc order differs from their roots' at a height
both lines cover, every 1 cm, so a pair already out of order at the top of the heights both cover counts too);
converging neighbors (the tips' arc gap under half the roots'); symmetry, on hair that
mirrors (MIRROR_P95): the farthest a chain's joints, mirrored, lie from the nearest chain's (by first joint), over the
joints both have; far hair (the area of hanging vertices beyond FAR_HAIR of every chain's line); tall columns (chains
whose locks' roots spread more than TALL_COLUMN in Z); light hair in columns (light locks' area in chains of a sheet);
and the smallest column chain's lock area. A lock (`hairrig.find_locks`) belongs to the chain holding most of its
vertices. Far and hanging areas sum `hairing.vertex_areas` over the mesh as loaded (both sides of double-sided cards);
lock areas (the column chains', light hair in columns) are the locks' own `area`, on the welded mesh,
as `_columns` compares them. The two don't compare with each other.

``--out`` gets {"checkout", "commit", "widths", "hairs": {label: {width: metrics}}, "skipped": {label: reason}}.
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The real test hairs, as in tests/test_hair_rig_core.py.
MAEVE = os.environ.get("MAGIC_FIT_MAEVE", r"G:\Penumbra\DOUX Maeve\chara\human\c0801\obj\hair\h0108\model"
                                          r"\c0801h0108_hir.mdl")
BRIAR = os.environ.get("MAGIC_FIT_BRIAR", r"G:\Penumbra\Briar Hair with custom physics\chara\human\c0801\obj\hair"
                                          r"\h0137\model\c0801h0137_hir.mdl")
TEST_RACE = "c0801"
COLLECTION = "Lucina"
# Hair mirrors when the 95th percentile of its hanging vertices' distance, mirrored, to the nearest one is within this.
MIRROR_P95 = 0.01
# Hanging hair farther than this from every chain is far hair (metres).
FAR_HAIR = 0.04
# A chain whose locks' roots spread more than this in Z is a tall column (metres).
TALL_COLUMN = 0.08
# Crossings are looked for every this much of height (metres), on lines sampled every LINE_STEP.
CROSS_STEP = 0.01
LINE_STEP = 0.002
# A shoulder miss: the nearest joint farther than this along the chain from the shoulders' height (metres).
SHOULDER_MISS = 0.04
# A column chain holding less than this is near the floor (2 x MIN_COLUMN_AREA, m²).
NEAR_FLOOR = 0.01
SHOULDERS = ("j_sako_l", "j_sako_r")
WORST = 10
# Rows of points measured against the chains' segments at a time.
CHUNK = 4096


def arguments():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", default=ROOT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref")
    parser.add_argument("--widths", default="0.06,0.08,0.10")
    parser.add_argument("--only")
    parser.add_argument("--hair-dir", default=os.path.join(ROOT, "bench_data", "hair"))
    return parser.parse_args(argv)


ARGS = arguments()
CHECKOUT = os.path.abspath(ARGS.checkout)
# The checkout's magic_fit first: fit_hair_matching puts this repository's root on sys.path.
sys.path.insert(0, os.path.join(CHECKOUT, "deps"))
sys.path.insert(0, CHECKOUT)
import magic_fit  # noqa: E402
from magic_fit import hairing, hairrig  # noqa: E402

print("magic_fit: " + magic_fit.__file__)
if not os.path.normcase(os.path.abspath(magic_fit.__file__)).startswith(
        os.path.normcase(os.path.join(CHECKOUT, "magic_fit") + os.sep)):
    sys.exit("magic_fit came from {:s}, not the checkout {:s}".format(magic_fit.__file__, CHECKOUT))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fit_hair_matching  # noqa: E402
import hair_models  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, fit_hair_matching.XIVPY)


# -----------------------------------------------------------------------------
# Hairs

def hair_sources(only):
    """(label, race, load) of every hair, ``load`` giving (points, tris); (label, reason) of those not found."""
    sources, skipped = [], []

    def test_hair(path):
        def load():
            with open(path, "rb") as handle:
                points, tris, _held = hair_models.decode(handle.read())
            return points, tris
        return load

    for label, path in (("maeve", MAEVE), ("briar", BRIAR)):
        if os.path.isfile(path):
            sources.append((label, TEST_RACE, test_hair(path)))
        else:
            skipped.append((label, "no file at " + path))
    for path in sorted(glob.glob(os.path.join(ARGS.hair_dir, "*.npz"))):
        with np.load(path) as data:
            race = str(data["race"])

        def load(path=path):
            with np.load(path) as data:
                return np.asarray(data["co"], dtype=np.float64), np.asarray(data["tris"], dtype=np.int64)
        sources.append((os.path.splitext(os.path.basename(path))[0], race, load))
    try:
        modded = fit_hair_matching.modded_hairs(COLLECTION)
    except Exception as error:  # noqa: BLE001 - the other hairs still count
        print("no {:s} hairs: {}".format(COLLECTION, error))
        modded = []
    for label, path, race, _entry in modded:
        def load(path=path):
            points, tris, _names, _weights = fit_hair_matching.read_model(path)
            return points, tris
        sources.append((label, race, load))
    if only:
        sources = [source for source in sources if source[0] in only]
        skipped = [item for item in skipped if item[0] in only]
    return sources, skipped


# -----------------------------------------------------------------------------
# Geometry

def polar(rig, at):
    """The angles (radians) of world points ``at`` around the head's vertical axis in ``rig``'s model space, 0
    straight in front (-Y), positive toward +X, +-pi behind, and their distances from that axis (metres): the same
    as hairrig._polar of the column design, so both checkouts are measured alike."""
    model = rig.model(np.asarray(at, dtype=np.float64).reshape(-1, 3))
    axis_y = rig.by_name[hairing.HEAD].head[1]
    return np.arctan2(model[:, 0], -(model[:, 1] - axis_y)), np.hypot(model[:, 0], model[:, 1] - axis_y)


class Group:
    """Chains of one kind (side hair: of one side), with the arc around the head they are ordered by."""

    def __init__(self, rig, chains):
        self.rig = rig
        self.chains = chains
        theta, radius = polar(rig, [chain.joints[0] for chain in chains])
        weight = np.array([chain.area for chain in chains], dtype=np.float64) + 1e-12
        self.mean = math.atan2(float((weight * np.sin(theta)).sum()), float((weight * np.cos(theta)).sum()))
        self.scale = float(np.average(radius, weights=weight))

    def arc(self, at):
        """The arc (metres) of world points ``at`` around the head from the group's mean angle."""
        theta, _radius = polar(self.rig, at)
        return ((theta - self.mean + math.pi) % (2.0 * math.pi) - math.pi) * self.scale


def line_of(chain):
    return np.vstack((chain.joints, chain.tip))


def dense(line, step=LINE_STEP):
    """``line`` sampled every ``step`` along its length, both ends included."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    at = np.concatenate(([0.0], np.cumsum(seg)))
    marks = np.append(np.arange(0.0, at[-1], step), at[-1])
    return np.column_stack([np.interp(marks, at, line[:, k]) for k in range(3)])


def crossed(group, a, b):
    """Whether neighbors ``a`` and ``b`` (``a`` first by root arc) change order at a height both lines cover."""
    da, db = dense(line_of(a)), dense(line_of(b))
    low = max(float(da[:, 2].min()), float(db[:, 2].min()))
    high = min(float(da[:, 2].max()), float(db[:, 2].max()))
    if high < low:
        return False
    levels = np.arange(high, low - 1e-9, -CROSS_STEP)
    at_a = da[np.argmin(np.abs(da[None, :, 2] - levels[:, None]), axis=1)]
    at_b = db[np.argmin(np.abs(db[None, :, 2] - levels[:, None]), axis=1)]
    return bool((group.arc(at_b) - group.arc(at_a) < 0.0).any())


def off_hair(chain, points):
    """How far the chain's joints and tip lie from its own vertices at most (metres)."""
    own = points[chain.vertices]
    return max(float(np.min(np.linalg.norm(own - point, axis=1))) for point in line_of(chain))


def sharpest_bend(chain):
    """The largest angle between two bones of the chain (degrees; 0 for one bone)."""
    bones = np.diff(line_of(chain), axis=0)
    length = np.linalg.norm(bones, axis=1)
    if len(bones) < 2 or (length <= 0.0).any():
        return 0.0
    cos = np.einsum("ij,ij->i", bones[:-1], bones[1:]) / (length[:-1] * length[1:])
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max())


def shoulder_gap(chain, height):
    """The distance along the chain from where its line first crosses ``height`` (world Z) to the nearest joint;
    None when it doesn't cross."""
    line = line_of(chain)
    z = line[:, 2]
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    along = np.concatenate(([0.0], np.cumsum(seg)))
    for i in range(len(line) - 1):
        if (z[i] - height) * (z[i + 1] - height) <= 0.0 and z[i] != z[i + 1]:
            at = along[i] + (z[i] - height) / (z[i] - z[i + 1]) * seg[i]
            # The tip ends the last bone; it isn't a joint.
            return float(np.abs(along[:len(chain.joints)] - at).min())
    return None


def segment_distances(points, starts, ends):
    """The distance from each of ``points`` to the nearest of the segments ``starts`` to ``ends``."""
    best = np.full(len(points), np.inf)
    span = ends - starts
    length = np.maximum((span * span).sum(1), 1e-18)
    for first in range(0, len(points), CHUNK):
        chunk = points[first:first + CHUNK]
        offset = chunk[:, None, :] - starts[None, :, :]
        t = np.clip(np.einsum("psk,sk->ps", offset, span) / length[None, :], 0.0, 1.0)
        gap = offset - t[:, :, None] * span[None, :, :]
        best[first:first + CHUNK] = np.sqrt((gap * gap).sum(2).min(1))
    return best


def mirrored(rig, at):
    """World points ``at`` mirrored in the rig's model X."""
    return rig.model(at) * (-1.0, 1.0, 1.0) + rig.offset


# -----------------------------------------------------------------------------
# Measuring

class Hair:
    """What a hair's metrics need besides its chains, measured once for every width."""

    def __init__(self, points, tris, race):
        self.points = np.asarray(points, dtype=np.float64)
        self.tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        self.race = race
        self.rig = hairing.Reference.get().rig(race, 0)
        self.cap = hairing.cap_depth(self.points, self.tris, self.rig)
        self.areas = hairing.vertex_areas(self.points, self.tris)
        hanging = hairrig._hanging(self.rig.scalp(self.points), self.rig.head_distance(self.points), self.cap)
        self.hanging = np.flatnonzero(hanging)
        self.locks = hairrig.find_locks(self.points, self.tris, self.rig, cap=self.cap)
        # The locks' own areas (`hairing.vertex_areas` on the welded mesh, a double-sided card once): what
        # MIN_COLUMN_AREA is held against.
        self.lock_areas = np.array([float(lock.area) for lock in self.locks])
        self.mirror_p95 = None
        if len(self.hanging):
            model = self.rig.model(self.points[self.hanging])
            gap, _index = cKDTree(model).query(model * (-1.0, 1.0, 1.0))
            self.mirror_p95 = float(np.percentile(gap, 95))
        heights = [self.rig.head_height(name) for name in SHOULDERS if name in self.rig.by_name]
        self.shoulder = float(np.mean(heights)) if heights else None


def measure(hair, chains, took):
    """The metrics of ``chains`` (`hairrig.ChainPlan`s, planned in ``took`` seconds) on ``hair`` (`Hair`)."""
    points, rig = hair.points, hair.rig
    sheet = [getattr(chain, "sheet", -1) for chain in chains]
    theta, _radius = polar(rig, [chain.joints[0] for chain in chains]) if chains else (np.zeros(0), None)
    groups = {}
    for index, chain in enumerate(chains):
        groups.setdefault((chain.kind, chain.kind == "side" and bool(theta[index] < 0.0)), []).append(index)

    # Each lock belongs to the chain holding most of its vertices.
    holder = np.full(len(points), -1, dtype=np.int64)
    for index, chain in enumerate(chains):
        holder[chain.vertices] = index
    owner = []
    for lock in hair.locks:
        held = holder[lock.vertices]
        held = held[held >= 0]
        owner.append(int(np.bincount(held).argmax()) if len(held) else -1)
    owner = np.array(owner, dtype=np.int64)

    records = [None] * len(chains)
    crossing, converging = [], []
    for (kind, _left), members in groups.items():
        group = Group(rig, [chains[k] for k in members])
        roots = group.arc([chains[k].joints[0] for k in members])
        tips = group.arc([chains[k].tip for k in members])
        order = [members[k] for k in np.argsort(roots, kind="stable")]
        for k, index in enumerate(members):
            chain = chains[index]
            mine = np.flatnonzero(owner == index)
            lines = group.arc(line_of(chain))
            gap = shoulder_gap(chain, hair.shoulder) if kind in ("back", "side") and hair.shoulder is not None \
                else None
            records[index] = dict(
                kind=kind, sheet=int(sheet[index]), order=int(getattr(chain, "order", -1)), bones=len(chain.joints),
                root=[float(v) for v in chain.joints[0]], tip=[float(v) for v in chain.tip], area=float(chain.area),
                lock_area=float(hair.lock_areas[mine].sum()), light=bool(chain.light),
                drift=float(np.abs(lines - lines[0]).max()), off_hair=off_hair(chain, points),
                bend=sharpest_bend(chain), shoulder=gap,
                root_spread_z=float(np.ptp([hair.locks[m].root[2] for m in mine])) if len(mine) else 0.0)
            records[index]["arc_root"] = float(roots[k])
            records[index]["arc_tip"] = float(tips[k])
        position = {index: k for k, index in enumerate(members)}
        for a, b in zip(order, order[1:]):
            if crossed(group, chains[a], chains[b]):
                crossing.append([a, b])
            root_gap = abs(roots[position[b]] - roots[position[a]])
            if abs(tips[position[b]] - tips[position[a]]) < 0.5 * root_gap:
                converging.append([a, b])

    symmetric = hair.mirror_p95 is not None and hair.mirror_p95 <= MIRROR_P95
    symmetry = None
    if symmetric and chains:
        firsts = np.array([chain.joints[0] for chain in chains])
        symmetry = 0.0
        for chain in chains:
            mirror = mirrored(rig, chain.joints)
            other = chains[int(np.argmin(np.linalg.norm(firsts - mirror[0], axis=1)))]
            count = min(len(mirror), len(other.joints))
            symmetry = max(symmetry, float(np.linalg.norm(mirror[:count] - other.joints[:count], axis=1).max()))

    hanging_area = float(hair.areas[hair.hanging].sum())
    far_area = hanging_area
    if chains and len(hair.hanging):
        starts = np.vstack([line_of(chain)[:-1] for chain in chains])
        ends = np.vstack([line_of(chain)[1:] for chain in chains])
        far = segment_distances(points[hair.hanging], starts, ends) > FAR_HAIR
        far_area = float(hair.areas[hair.hanging][far].sum())

    in_columns = np.array([owner[m] >= 0 and sheet[owner[m]] >= 0 for m in range(len(hair.locks))], dtype=bool)
    light = np.array([lock.light for lock in hair.locks], dtype=bool)
    columns = [record["lock_area"] for record in records if record["sheet"] >= 0]
    kinds = {}
    for record in records:
        mine = kinds.setdefault(record["kind"], {"chains": 0, "bones": 0})
        mine["chains"] += 1
        mine["bones"] += record["bones"]
    side = [sum(1 for k, chain in enumerate(chains) if chain.kind == "side" and (theta[k] < 0.0) == left)
            for left in (True, False)]
    side_columns = [sum(1 for k, chain in enumerate(chains)
                        if chain.kind == "side" and sheet[k] >= 0 and (theta[k] < 0.0) == left) for left in (True, False)]
    return dict(
        race=hair.race, vertices=len(points), time=took, chains=len(chains), bones=sum(r["bones"] for r in records),
        kinds=kinds, side=side, side_columns=side_columns,
        back_columns=sum(1 for k, chain in enumerate(chains) if chain.kind == "back" and sheet[k] >= 0),
        crossing=crossing, converging=converging, mirror_p95=hair.mirror_p95, symmetric=symmetric,
        symmetry=symmetry, hanging_area=hanging_area, far_area=far_area,
        light_in_columns=float(hair.lock_areas[in_columns & light].sum()) if len(hair.locks) else 0.0,
        smallest_column=min(columns) if columns else None,
        tall=sum(1 for record in records if record["root_spread_z"] > TALL_COLUMN),
        drift=max((r["drift"] for r in records), default=0.0),
        off_hair=max((r["off_hair"] for r in records), default=0.0),
        bend=max((r["bend"] for r in records), default=0.0),
        shoulder_misses=sum(1 for r in records if r["shoulder"] is not None and r["shoulder"] > SHOULDER_MISS),
        chain_list=records)


# -----------------------------------------------------------------------------
# Totals

def totals(hairs, width):
    """The totals at ``width`` (key) over ``hairs`` ({label: {width: metrics}}): {name: value}, and the per-hair
    values the lists come from."""
    rows = {label: widths[width] for label, widths in hairs.items() if width in widths}
    result = {"hairs": len(rows)}
    if not rows:
        return result
    values = list(rows.values())
    for kind in hairrig.KINDS:
        chains = [row["kinds"].get(kind, {}).get("chains", 0) for row in values]
        bones = [row["kinds"].get(kind, {}).get("bones", 0) for row in values]
        result[kind + " chains"] = (float(np.mean(chains)), max(chains))
        result[kind + " bones"] = (float(np.mean(bones)), max(bones))
    result["chains"] = (float(np.mean([row["chains"] for row in values])), max(row["chains"] for row in values))
    result["bones"] = (float(np.mean([row["bones"] for row in values])), max(row["bones"] for row in values))
    result["over 3 back"] = sum(row["kinds"].get("back", {}).get("chains", 0) > 3 for row in values)
    result["over 2 side on a side"] = sum(max(row["side"]) > 2 for row in values)
    result["over 3 back columns"] = sum(row["back_columns"] > 3 for row in values)
    result["over 2 side columns on a side"] = sum(max(row["side_columns"]) > 2 for row in values)
    result["crossing pairs"] = sum(len(row["crossing"]) for row in values)
    result["hairs crossing"] = sum(bool(row["crossing"]) for row in values)
    result["converging pairs"] = sum(len(row["converging"]) for row in values)
    result["hairs converging"] = sum(bool(row["converging"]) for row in values)
    result["far area m2"] = sum(row["far_area"] for row in values)
    result["far share"] = result["far area m2"] / max(sum(row["hanging_area"] for row in values), 1e-12)
    result["symmetric hairs"] = sum(row["symmetric"] for row in values)
    symmetry = [row["symmetry"] for row in values if row["symmetry"] is not None]
    result["symmetry max m"] = max(symmetry) if symmetry else None
    result["symmetry mean m"] = float(np.mean(symmetry)) if symmetry else None
    result["bend max deg"] = max(row["bend"] for row in values)
    result["bend mean deg"] = float(np.mean([row["bend"] for row in values]))
    result["shoulder misses > 4 cm"] = sum(row["shoulder_misses"] for row in values)
    result["hairs with a shoulder miss"] = sum(row["shoulder_misses"] > 0 for row in values)
    result["tall columns"] = sum(row["tall"] for row in values)
    result["light in columns m2"] = sum(row["light_in_columns"] for row in values)
    result["time s"] = sum(row["time"] for row in values)
    result["time max s"] = max(row["time"] for row in values)
    return result


def shown(value):
    if value is None:
        return "-"
    if isinstance(value, tuple):
        return "{:.2f} / {:d}".format(*value)
    if isinstance(value, float):
        return "{:.4f}".format(value)
    return str(value)


def worst(hairs, width, key, ref):
    """The WORST hairs by ``key`` at ``width``, with the chain's kind, and the ref's value for each."""
    rows = [(row[key], label, row) for label, widths in hairs.items() for row in [widths.get(width)] if row]
    lines = []
    for value, label, row in sorted(rows, key=lambda item: -item[0])[:WORST]:
        chain = max(row["chain_list"], key=lambda record: record[key]) if row["chain_list"] else {"kind": "-"}
        other = (ref.get(label) or {}).get(width)
        lines.append("    {:28s} {:.4f} ({:s}){:s}".format(
            label, value, chain["kind"], "   ref {:.4f}".format(other[key]) if other else ""))
    return lines


def report(run, ref):
    hairs = run["hairs"]
    ref_hairs = {label: widths for label, widths in (ref or {}).get("hairs", {}).items() if label in hairs}
    for width in run["widths"]:
        ours, theirs = totals(hairs, width), totals(ref_hairs, width) if ref else {}
        print("\nwidth {:s}{:s}".format(width, "   (ref: {:s})".format(ref["checkout"]) if ref else ""))
        for name, value in ours.items():
            line = "  {:32s} {:>16s}".format(name, shown(value))
            if ref:
                line += "   ref {:>16s}".format(shown(theirs.get(name)))
            print(line)
        print("  worst by drift (m):")
        print("\n".join(worst(hairs, width, "drift", ref_hairs)))
        print("  worst by off hair (m):")
        print("\n".join(worst(hairs, width, "off_hair", ref_hairs)))
        misses = [(label, record["kind"], record["shoulder"]) for label, widths in hairs.items()
                  if width in widths for record in widths[width]["chain_list"]
                  if record["shoulder"] is not None and record["shoulder"] > SHOULDER_MISS]
        print("  shoulder misses over 4 cm: " + (", ".join("{:s} {:s} {:.3f}".format(*miss) for miss in misses)
                                                 or "none"))
        near = sorted((widths[width]["smallest_column"], label) for label, widths in hairs.items()
                      if width in widths and widths[width]["smallest_column"] is not None
                      and widths[width]["smallest_column"] < NEAR_FLOOR)
        print("  smallest column chain under {:.2f} m2: {:s}".format(
            NEAR_FLOOR, ", ".join("{:s} {:.4f}".format(label, area) for area, label in near) or "none"))
    if run["skipped"]:
        print("\nskipped {:d}:".format(len(run["skipped"])))
        for label, reason in run["skipped"].items():
            print("  {:s}: {:s}".format(label, reason))


def main():
    widths = [float(width) for width in ARGS.widths.split(",")]
    only = set(ARGS.only.split(",")) if ARGS.only else None
    commit = subprocess.run(["git", "-C", CHECKOUT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run = {"checkout": CHECKOUT, "commit": commit, "widths": [str(width) for width in widths], "hairs": {},
           "skipped": {}}
    sources, skipped = hair_sources(only)
    run["skipped"].update(skipped)
    print("{:d} hairs, widths {}".format(len(sources), run["widths"]))
    for count, (label, race, load) in enumerate(sources):
        try:
            points, tris = load()
        except Exception as error:  # noqa: BLE001 - skipped and counted
            run["skipped"][label] = "load: {}".format(error)
            print("{:d}/{:d} {:s}: skipped, {}".format(count + 1, len(sources), label, error), flush=True)
            continue
        try:
            hair = Hair(points, tris, race)
            result = {}
            for width in widths:
                started = time.perf_counter()
                chains = hairrig.plan_rig(hair.points, hair.tris, hair.rig, cap=hair.cap, spacing=width)
                result[str(width)] = measure(hair, chains, time.perf_counter() - started)
        except Exception as error:  # noqa: BLE001 - skipped and counted
            run["skipped"][label] = "rig: {}".format(error)
            print("{:d}/{:d} {:s}: skipped, {}".format(count + 1, len(sources), label, error), flush=True)
            traceback.print_exc()
            continue
        run["hairs"][label] = result
        print("{:d}/{:d} {:s} {:s} {:d} vertices: {:s}".format(
            count + 1, len(sources), label, race, len(points), ", ".join(
                "{:s} {:d} chains {:d} bones ({:d} back) {:.1f} s".format(
                    width, row["chains"], row["bones"], row["kinds"].get("back", {}).get("chains", 0), row["time"])
                for width, row in result.items())), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
    with open(ARGS.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(run, handle, indent=1)
    print("wrote " + os.path.abspath(ARGS.out))
    ref = None
    if ARGS.ref:
        with open(ARGS.ref, encoding="utf-8") as handle:
            ref = json.load(handle)
    report(run, ref)


main()
