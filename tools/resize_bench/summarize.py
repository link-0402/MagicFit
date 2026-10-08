"""Summarize a resize bench run, or compare it with another.

    python tools/resize_bench/summarize.py RUN_DIR [--ref REF_DIR] [--worst K] [--rows]

Any Python 3 runs it (Blender's too). Prints the totals per metric (counts summed, the rest as medians over
garment and size), and with --ref the same for the reference run, their ratio and per garment and size how
many got better or worse. --rows prints every garment and size, --worst the K worst vertices of each kind over
all garments. Lower is better for every metric.
"""

import argparse
import json
import os
import statistics


def band(name, key="mean"):
    return lambda t: t["clearance"]["bands"].get(name, {}).get(key)


def part(*keys):
    def get(t):
        for key in keys:
            if not isinstance(t, dict) or key not in t:
                return None
            t = t[key]
        return t
    return get


# (name, how it's read from a garment's target, summed (True) or median, decimals, what it is)
METRICS = (
    ("seconds", part("seconds"), True, 1, "resize time"),
    ("tight", band("0-3mm"), False, 3, "mean clearance change, cloth 0-3 mm off the skin (mm)"),
    ("near", band("3-6mm"), False, 3, "mean clearance change, 3-6 mm off (mm)"),
    ("mid", band("6-15mm"), False, 3, "mean clearance change, 6-15 mm off (mm)"),
    ("tight_p95", band("0-3mm", "p95"), False, 3, "95th percentile of the above, 0-3 mm (mm)"),
    ("clip_out", part("clearance", "clip_out"), True, 0, "vertices that were outside the skin, now 1 mm+ inside"),
    ("deepest", lambda t: -t["clearance"]["deepest_mm"], False, 2, "deepest vertex inside the body (mm)"),
    ("rigid_max", part("rigid", "max"), False, 2, "worst clearance change of a rigid part on the skin (mm)"),
    ("rigid_1mm", part("rigid", "over_1mm"), True, 0, "rigid parts on the skin off by 1 mm+"),
    ("thick_p95", part("thickness", "p95"), False, 3, "95th percentile of layer thickness change (mm)"),
    ("thick_1mm", part("thickness", "over_1mm"), True, 0, "layer pairs changed 1 mm+"),
    ("detail_p95", part("detail", "normal_mm", "p95"), False, 4, "95th percentile of detail change along the normal (mm)"),
    ("detail_max", part("detail", "normal_mm", "max"), False, 3, "largest detail change along the normal (mm)"),
    ("flipped", part("flipped"), True, 0, "triangles turned over"),
    ("rt_p95", part("round_trip", "p95"), False, 3, "95th percentile of the round trip's gap (mm)"),
    ("rt_max", part("round_trip", "max"), False, 2, "largest gap after the round trip (mm)"),
)
# Changes smaller than these (relative, or absolute in the metric's unit) are ties.
TIE_RELATIVE = 0.05
TIE_ABSOLUTE = {"seconds": 0.2, "clip_out": 2, "rigid_1mm": 1, "thick_1mm": 5, "flipped": 1}
WORST = (("clearance", ("clearance", "worst")), ("thickness", ("thickness", "worst")), ("rigid", ("rigid", "worst")),
         ("detail", ("detail", "worst")), ("round trip", ("round_trip", "worst")))


def load(folder):
    runs, errors = {}, {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json") or name == "run.json":
            continue
        with open(os.path.join(folder, name), encoding="utf-8-sig") as f:
            result = json.load(f)
        if "error" in result:
            errors[result["tag"]] = result["error"].strip().splitlines()[-1]
            continue
        for target, row in result.get("targets", {}).items():
            runs[(result["tag"], target)] = (result, row)
    return runs, errors


def value(metric, row):
    try:
        v = metric[1](row)
    except (KeyError, TypeError):
        return None
    return None if v is None else float(v)


def total(metric, rows):
    values = [v for v in (value(metric, row) for row in rows) if v is not None]
    if not values:
        return None
    return sum(values) if metric[2] else statistics.median(values)


def fmt(v, decimals):
    return "-" if v is None else "{:.{}f}".format(v, decimals)


def worse(metric, new, old):
    """1 when ``new`` is worse than ``old``, -1 better, 0 a tie."""
    if new is None or old is None:
        return 0
    if abs(new - old) <= max(TIE_RELATIVE * max(abs(new), abs(old)), TIE_ABSOLUTE.get(metric[0], 0.0)):
        return 0
    return 1 if new > old else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--ref")
    parser.add_argument("--worst", type=int, default=0)
    parser.add_argument("--rows", action="store_true")
    args = parser.parse_args()
    runs, errors = load(args.run)
    refs, ref_errors = load(args.ref) if args.ref else ({}, {})
    keys = sorted(set(runs) & set(refs)) if args.ref else sorted(runs)
    print("{:d} garments, {:d} garment sizes{:s}".format(len({k[0] for k in keys}), len(keys),
                                                         " in both runs" if args.ref else ""))
    for tag, error in sorted(errors.items()):
        print("  FAILED {:s}: {:s}".format(tag, error))
    for tag, error in sorted(ref_errors.items()):
        print("  FAILED in ref {:s}: {:s}".format(tag, error))
    if args.rows:
        names = [m[0] for m in METRICS]
        print("\n{:<12s} {:<24s} ".format("garment", "size") + " ".join("{:>10s}".format(n) for n in names))
        for key in keys:
            row = runs[key][1]
            print("{:<12s} {:<24s} ".format(*key) + " ".join("{:>10s}".format(fmt(value(m, row), m[3]))
                                                             for m in METRICS))
    print()
    if args.ref:
        print("{:<11s} {:>11s} {:>11s} {:>7s} {:>7s} {:>7s}  {:s}".format("metric", "run", "ref", "ratio", "better",
                                                                       "worse", ""))
    else:
        print("{:<11s} {:>11s}  {:s}".format("metric", "run", ""))
    for metric in METRICS:
        mine = total(metric, [runs[k][1] for k in keys])
        if not args.ref:
            print("{:<11s} {:>11s}  {:s}".format(metric[0], fmt(mine, metric[3]), metric[4]))
            continue
        theirs = total(metric, [refs[k][1] for k in keys])
        ratio = "-" if not mine and not theirs else ("inf" if not theirs else "{:.3f}".format(mine / theirs))
        verdicts = [worse(metric, value(metric, runs[k][1]), value(metric, refs[k][1])) for k in keys]
        print("{:<11s} {:>11s} {:>11s} {:>7s} {:>7d} {:>7d}  {:s}".format(
            metric[0], fmt(mine, metric[3]), fmt(theirs, metric[3]), ratio, verdicts.count(-1), verdicts.count(1),
            metric[4]))
    if args.worst:
        for title, path in WORST:
            rows = []
            for key in keys:
                found = part(*path)(runs[key][1]) or []
                rows += [(item["value"], key, item) for item in found]
            rows.sort(key=lambda r: -r[0])
            if not rows:
                continue
            print("\nworst {:s}:".format(title))
            for v, (tag, target), item in rows[:args.worst]:
                extra = " ".join("{:s}={!s}".format(k, item[k]) for k in item if k not in ("value", "v", "at"))
                print("  {:8.3f}  {:<12s} {:<24s} v{:<7d} at {!s} {:s}".format(v, tag, target, item["v"], item["at"],
                                                                             extra))


if __name__ == "__main__":
    main()
