"""How Own Skeleton's chains lie on many hairs: `hairrig.plan_layout` with the chain counts the user sets (or Detect's),
measured the same way for any checkout of Magic Fit, so two runs compare (tools/hair_layout_compare.py).

    blender-run.ps1 -Script tools\\hair_layout_check.py -- --out <file.json> [--checkout <path>] [--counts 3-1-2-2-2]
        [--detect] [--ponytail] [--only a,b] [--npz file.npz ...] [--sheets <folder>] [--no-corpus]

- ``--checkout``: the Magic Fit whose planner runs (default: this repository). Its `magic_fit` is imported before
  this repository's tools; the run stops when `magic_fit` comes from elsewhere. A checkout without `hairlanes` gets
  this repository's (loaded into its `magic_fit`) for the areas.
- ``--counts``: back, ponytail, left, right and bangs chains (`hairrig.ChainCounts`), split on "-", "," or spaces.
  ``--detect``: as the panel's Detect button, the counts `hairrig.detect_counts` proposes, the rest as set.
  ``--ponytail``: Hair Type Ponytail.
- Hairs: each ``--npz`` (world space ``co``, ``tris`` and ``race``), then maeve and briar (the real test hairs of
  tests/test_hair_rig_core.py, c0801), then unless ``--no-corpus`` the hair models active in the Penumbra collection
  "Lucina" (`fit_hair_matching.modded_hairs`). ``--only`` keeps those labels (split on "," or spaces).
- ``--sheets``: a PNG per hair in that folder (tools/hair_layout_draw.py): back, left and front, each chain's hair in
  its color, the rest of the hanging hair grey; labels with ":" are written with "_".

``--out`` gets {label: record}, or {label: {"error": message}} for a hair that failed. A record: ``race``,
``vertices``, ``cap`` (`hairing.cap_depth`), ``short``, ``asked`` and ``made`` (chains per area), ``masses``
(hanging area per area, `hairlanes.areas`), ``hanging_area``, ``far_area`` (hanging area beyond FAR_HAIR of every
chain's line), ``moved`` and ``lifted`` (the chains' check moves, `hairrig.ChainPlan`; 0 where it has none), ``time``
(seconds, Detect included) and ``chains``, each with ``kind``, ``area`` (back, ponytail, left, right or bangs: by the
area most of its hanging hair is in), ``sheet``, ``order``, ``bones``, ``length`` (of its line), ``bend`` (the
sharpest between two bones, degrees), ``off_own`` and ``off_any`` (the farthest of its joints and tip from its own
hair and from any hanging hair), ``moved``, ``lifted`` and ``joints`` (model space, the tip last). Areas are hanging
`hairing.vertex_areas` (m²) over the mesh as loaded. The keys are those of the 2026-11-14 counts runs
(bench_data/hair_runs/2026-11-14-counts), so those still compare.
"""

import argparse
import importlib.util
import json
import os
import re
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
# Hanging hair farther than this from every chain's line is far hair (meters).
FAR_HAIR = 0.04
# The body's bones drawn on the sheets.
BODY = ("j_kao", "j_kubi", "j_sako_l", "j_sako_r", "j_ude_a_l", "j_ude_a_r", "j_sebo_c", "j_sebo_b", "j_sebo_a",
        "j_kosi")
# Rows of points measured against the chains' segments at a time.
CHUNK = 4096


def arguments():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkout", default=ROOT)
    parser.add_argument("--counts", default="3-1-2-2-2")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--ponytail", action="store_true")
    parser.add_argument("--only")
    parser.add_argument("--npz", nargs="*", default=[])
    parser.add_argument("--sheets")
    parser.add_argument("--no-corpus", action="store_true")
    args = parser.parse_args(argv)
    counts = [part for part in re.split(r"[-,\s]+", args.counts.strip()) if part]
    if len(counts) != 5 or not all(part.isdigit() for part in counts):
        parser.error("--counts needs five whole numbers (back, ponytail, left, right, bangs)")
    args.counts = [int(part) for part in counts]
    return args


ARGS = arguments()
CHECKOUT = os.path.abspath(ARGS.checkout)
# The checkout's magic_fit first: fit_hair_matching puts this repository's root on sys.path. A work tree without its
# own deps uses this repository's.
sys.path.insert(0, os.path.join(ROOT, "deps"))
sys.path.insert(0, os.path.join(CHECKOUT, "deps"))
sys.path.insert(0, CHECKOUT)
import magic_fit  # noqa: E402
from magic_fit import hairing, hairrig  # noqa: E402

print("magic_fit: " + magic_fit.__file__)
if not os.path.normcase(os.path.abspath(magic_fit.__file__)).startswith(
        os.path.normcase(os.path.join(CHECKOUT, "magic_fit") + os.sep)):
    sys.exit("magic_fit came from {:s}, not the checkout {:s}".format(magic_fit.__file__, CHECKOUT))
if os.path.isfile(os.path.join(CHECKOUT, "magic_fit", "hairlanes.py")):
    from magic_fit import hairlanes  # noqa: E402
else:
    # An older checkout: this repository's hairlanes, on the checkout's hairing and hairrig.
    _spec = importlib.util.spec_from_file_location("magic_fit.hairlanes",
                                                   os.path.join(ROOT, "magic_fit", "hairlanes.py"))
    hairlanes = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = hairlanes
    _spec.loader.exec_module(hairlanes)
print("hairlanes: " + hairlanes.__file__)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fit_hair_matching  # noqa: E402
import hair_layout_draw  # noqa: E402
import hair_models  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, fit_hair_matching.XIVPY)
TAIL = "ponytail" if ARGS.ponytail else "back"
# The names of `hairlanes.areas`'s labels, as `hairrig.AREAS` names them.
AREA_NAMES = {hairlanes.BACK: TAIL, hairlanes.LEFT: "left", hairlanes.RIGHT: "right", hairlanes.BANGS: "bangs"}


# -----------------------------------------------------------------------------
# Hairs

def hair_sources():
    """(label, race, load) of every hair, ``load`` giving (points, tris)."""
    sources = []
    for path in ARGS.npz:
        with np.load(path) as data:
            race = str(data["race"])

        def load(path=path):
            with np.load(path) as data:
                return np.asarray(data["co"], dtype=np.float64), np.asarray(data["tris"], dtype=np.int64)
        sources.append((os.path.splitext(os.path.basename(path))[0], race, load))
    for label, path in (("maeve", MAEVE), ("briar", BRIAR)):
        if not os.path.isfile(path):
            print("no {:s} at {:s}".format(label, path))
            continue

        def load(path=path):
            with open(path, "rb") as handle:
                points, tris, _held = hair_models.decode(handle.read())
            return points, tris
        sources.append((label, TEST_RACE, load))
    if not ARGS.no_corpus:
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
    if ARGS.only:
        # Split on spaces too: PowerShell passes an unquoted a,b as "a b".
        only = set(re.split(r"[,\s]+", ARGS.only.strip()))
        sources = [source for source in sources if source[0] in only]
    return sources


# -----------------------------------------------------------------------------
# Measuring

def line_of(chain):
    return np.vstack((chain.joints, chain.tip))


def segment_distances(points, lines):
    """The distance from each of ``points`` to the nearest segment of ``lines`` (polylines)."""
    starts = np.vstack([line[:-1] for line in lines])
    span = np.vstack([line[1:] for line in lines]) - starts
    length = np.maximum((span * span).sum(1), 1e-18)
    best = np.empty(len(points))
    for first in range(0, len(points), CHUNK):
        offset = points[first:first + CHUNK, None, :] - starts[None, :, :]
        t = np.clip(np.einsum("psk,sk->ps", offset, span) / length[None, :], 0.0, 1.0)
        gap = offset - t[:, :, None] * span[None, :, :]
        best[first:first + CHUNK] = np.sqrt((gap * gap).sum(2).min(1))
    return best


def sharpest_bend(line):
    """The largest angle between two bones of ``line`` (joints then tip; degrees, 0 for one bone). A bone of length
    0 has no direction: the bones either side of it are measured against each other."""
    bones = np.diff(line, axis=0)
    length = np.linalg.norm(bones, axis=1)
    bones, length = bones[length > 0.0], length[length > 0.0]
    if len(bones) < 2:
        return 0.0
    cos = np.einsum("ij,ij->i", bones[:-1], bones[1:]) / (length[:-1] * length[1:])
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max())


def chain_area(chain, label, weight, rig, points):
    """The area of ``chain``: its kind, or for side hair the side holding more of its hanging hair (``label`` and
    ``weight`` per vertex of the mesh, -1 and 0 where it doesn't hang), the side of its hair's mean X when none of it
    is side hair."""
    if chain.kind != "side":
        return chain.kind
    held, mass = label[chain.vertices], weight[chain.vertices]
    left = float(mass[held == hairlanes.LEFT].sum())
    right = float(mass[held == hairlanes.RIGHT].sum())
    if left == right:
        return "left" if float(rig.model(points[chain.vertices])[:, 0].mean()) >= 0.0 else "right"
    return "left" if left > right else "right"


def measure(label, race, points, tris):
    """The record of hair ``points`` (world space) with triangles ``tris`` on ``race``'s rig, and its sheet."""
    rig = hairing.Reference.get().rig(race, 0)
    cap = hairing.cap_depth(points, tris, rig)
    counts = hairrig.ChainCounts(*ARGS.counts)
    started = time.perf_counter()
    if ARGS.detect:
        found, _short = hairrig.detect_counts(points, tris, rig, cap=cap, ponytail=ARGS.ponytail)
        counts = counts._replace(**found)
    chains, _made, short = hairrig.plan_layout(points, tris, rig, cap=cap, counts=counts, ponytail=ARGS.ponytail)
    took = time.perf_counter() - started

    hair = hairlanes.measure(points, tris, rig, cap)
    label_of = np.full(len(points), -1, dtype=np.int64)
    weight = np.zeros(len(points))
    masses = dict.fromkeys((TAIL, "left", "right", "bangs"), 0.0)
    if hair is not None:
        areas, _short = hairlanes.areas(hair, ARGS.ponytail)
        label_of[hair.index] = areas
        weight[hair.index] = hair.area
        for number, name in AREA_NAMES.items():
            masses[name] = float(hair.area[areas == number].sum())
    hanging = hair.at if hair is not None else np.zeros((0, 3))
    tree = cKDTree(hanging) if len(hanging) else None
    lines = [rig.model(line_of(chain)) for chain in chains]

    records = []
    for chain, line in zip(chains, lines):
        own = rig.model(points[chain.vertices])
        records.append(dict(
            kind=chain.kind, area=chain_area(chain, label_of, weight, rig, points),
            sheet=int(getattr(chain, "sheet", -1)), order=int(getattr(chain, "order", -1)), bones=len(chain.joints),
            length=float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()), bend=sharpest_bend(line),
            off_own=float(cKDTree(own).query(line)[0].max()),
            off_any=float(tree.query(line)[0].max()) if tree is not None else 0.0,
            moved=int(getattr(chain, "moved", 0)), lifted=int(getattr(chain, "lifted", 0)), joints=line.tolist()))
    made = {}
    for record in records:
        made[record["area"]] = made.get(record["area"], 0) + 1
    hanging_area = float(weight.sum())
    far_area = hanging_area
    if lines and len(hanging):
        far_area = float(hair.area[segment_distances(hanging, lines) > FAR_HAIR].sum())
    result = dict(
        race=race, vertices=len(points), cap=float(cap), short=bool(short),
        asked={TAIL: counts.ponytail if ARGS.ponytail else counts.back, "left": counts.left, "right": counts.right,
               "bangs": counts.bangs},
        made=made, masses=masses, hanging_area=hanging_area, far_area=far_area,
        moved=sum(record["moved"] for record in records), lifted=sum(record["lifted"] for record in records),
        time=took, chains=records)
    if ARGS.sheets:
        draw(label, rig, points, tris, chains, lines, label_of >= 0)
    return result


def draw(label, rig, points, tris, chains, lines, hanging):
    """The sheet of a hair: each chain's hair in its color (darker), the rest of the hanging hair grey."""
    owner = np.full(len(points), -1, dtype=np.int64)
    for index, chain in enumerate(chains):
        owner[chain.vertices] = index
    palette = hair_layout_draw.PALETTE
    colors = np.where(hanging[:, None], np.array((110.0, 110.0, 110.0)), np.array((55.0, 55.0, 60.0)))
    held = owner >= 0
    colors[held] = palette[owner[held] % len(palette)] * 0.55
    dots, vertex = hair_layout_draw.surface_dots(rig.model(points), tris)
    marks = [rig.by_name[name].head for name in BODY if name in rig.by_name]
    hair_layout_draw.sheet(os.path.join(ARGS.sheets, label.replace(":", "_") + ".png"), dots, colors[vertex], lines,
                           [palette[k % len(palette)] for k in range(len(lines))], marks)


def main():
    commit = subprocess.run(["git", "-C", CHECKOUT, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    print("checkout {:s} at {:s}, counts {}{:s}{:s}".format(CHECKOUT, commit or "?", ARGS.counts,
                                                         ", Detect" if ARGS.detect else "",
                                                         ", Ponytail" if ARGS.ponytail else ""))
    if ARGS.sheets:
        os.makedirs(ARGS.sheets, exist_ok=True)
    run = {}
    sources = hair_sources()
    for count, (label, race, load) in enumerate(sources):
        try:
            points, tris = load()
            result = measure(label, race, np.asarray(points, dtype=np.float64),
                             np.asarray(tris, dtype=np.int64).reshape(-1, 3))
        except Exception as error:  # noqa: BLE001 - written down and counted
            print("{:d}/{:d} {:s}: failed, {}".format(count + 1, len(sources), label, error), flush=True)
            traceback.print_exc()
            run[label] = {"error": str(error)}
            continue
        run[label] = result
        far = result["far_area"] / result["hanging_area"] if result["hanging_area"] > 0.0 else 0.0
        print("{:d}/{:d} {:s} {:s}{:s} made {} of {} far {:.1%} moved {:d} lifted {:d} {:.1f} s".format(
            count + 1, len(sources), label, race, " short" if result["short"] else "", result["made"],
            result["asked"], far, result["moved"], result["lifted"], result["time"]), flush=True)
    out = os.path.abspath(ARGS.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(run, handle, indent=1)
    print("wrote {:s} ({:d} hairs, {:d} failed)".format(out, len(run), sum("error" in v for v in run.values())))


main()
