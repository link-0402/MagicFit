"""Two runs of tools/hair_layout_check.py side by side, on the hairs both have:

    python tools\\hair_layout_compare.py <ref.json> <new.json>

Neighbors (the chains of one area, ordered across it by their first joint) are measured again from the stored
joints, the same way for every run: crossing (their order flips at a height both lines cover), converging (the tips
closer than half the first joints) and within NEAR (the closest the lines come at the heights both cover). Most lines
count the hairs long in both runs; the chain lines and the short hairs count every hair both runs have. Far share and
the check's moves show "-" for a run whose file doesn't have them (the 2026-11-14 counts runs don't).
"""

import argparse
import json
import os

import numpy as np

# An area holds hair when its hanging hair (`masses`, m²) reaches this.
HAS_HAIR = 0.003
# Neighbors within this of each other (meters), chains this far off hair.
NEAR = 0.03
# Long hair's chains shorter than this (meters) are stubs; bends sharper than BEND (degrees) count.
STUB = 0.08
BEND = 45.0
# Chains with a bone shorter than this (meters, `hairlanes.MIN_BONE`) are counted.
MIN_BONE = 0.01
# Back chains on one row more than this apart in height (meters).
ROW_SPREAD = 0.02
# The hairs whose chains are listed one by one.
NAMED = ("user_h0181", "maeve", "briar", "mod:c0801h0153", "mod:c0801h0189")


def across(p, side):
    """Where points lie across their area: left to right behind and in front, front to back at the sides."""
    return np.arctan2(np.abs(p[..., 0]), -p[..., 1]) if side else p[..., 0]


def pairs(v):
    """Neighboring chains of one area: crossing [a, b], converging [a, b], closest [a, b, smallest gap, gap at the
    top]."""
    crossing, converging, closest = [], [], []
    by_area = {}
    for index, chain in enumerate(v["chains"]):
        by_area.setdefault(chain["area"], []).append(index)
    for area, members in by_area.items():
        side = area in ("left", "right")
        lines = {k: np.array(v["chains"][k]["joints"]) for k in members}
        members = sorted(members, key=lambda k: float(across(lines[k][0], side)))
        for a, b in zip(members, members[1:]):
            la, lb = lines[a], lines[b]
            low, high = max(la[:, 2].min(), lb[:, 2].min()), min(la[:, 2].max(), lb[:, 2].max())
            if high > low:
                levels = np.linspace(high, low, 40)

                def at(line):
                    order = np.argsort(line[:, 2])
                    return np.column_stack([np.interp(levels, line[order, 2], line[order, k]) for k in range(3)])
                pa, pb = at(la), at(lb)
                gap = np.linalg.norm(pa - pb, axis=1)
                closest.append([a, b, float(gap.min()), float(gap[0])])
                if ((across(pb, side) - across(pa, side)) < 0.0).any():
                    crossing.append([a, b])
            if np.linalg.norm(la[-1] - lb[-1]) < 0.5 * np.linalg.norm(la[0] - lb[0]):
                converging.append([a, b])
    return crossing, converging, closest


def load(path):
    """({label: record} of the hairs that didn't fail, with their neighbors measured; how many failed)."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    rows = {k: v for k, v in data.items() if "error" not in v}
    for v in rows.values():
        v["crossing"], v["converging"], v["closest"] = pairs(v)
    return rows, len(data) - len(rows)


def rows_of(v):
    """(unequal bone counts, the largest height spread of one row across the back chains); None with under 2."""
    back = [c for c in v["chains"] if c["area"] == "back"]
    if len(back) < 2:
        return None
    spread = 0.0
    for row in range(max(c["bones"] for c in back)):
        z = [c["joints"][row][2] for c in back if c["bones"] > row]
        if len(z) > 1:
            spread = max(spread, max(z) - min(z))
    return len({c["bones"] for c in back}) > 1, spread


def bend_of(chain):
    """The largest angle between two bones of ``chain`` (degrees), measured again from its stored joints: runs made
    before 2026-11-15 stored 0 for a chain with a bone of length 0. Such a bone has no direction and is left out."""
    bones = np.diff(np.array(chain["joints"]), axis=0)
    length = np.linalg.norm(bones, axis=1)
    bones, length = bones[length > 0.0], length[length > 0.0]
    if len(bones) < 2:
        return 0.0
    cos = np.einsum("ij,ij->i", bones[:-1], bones[1:]) / (length[:-1] * length[1:])
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max())


def missing(v):
    """The areas of ``v`` holding hair (`masses`) with no chain."""
    return [area for area, mass in v.get("masses", {}).items() if mass >= HAS_HAIR and v["made"].get(area, 0) == 0]


def totals(rows, long_labels):
    """{line: value} of one run's ``rows`` (the hairs both runs have), the long-hair lines on ``long_labels``."""
    out = {}
    long_rows = {k: v for k, v in rows.items() if k in long_labels}
    out["hairs / long hairs"] = "{} / {}".format(len(rows), len(long_rows))
    out["long hairs with crossing neighbors"] = sum(bool(v["crossing"]) for v in long_rows.values())
    out["long hairs with converging neighbors"] = sum(bool(v["converging"]) for v in long_rows.values())
    out["long hairs with neighbors within 3 cm"] = sum(any(p[2] < NEAR for p in v["closest"])
                                                     for v in long_rows.values())
    for area in ("back", "left", "right", "bangs"):
        cross = sum(v["chains"][p[0]]["area"] == area for v in long_rows.values() for p in v["crossing"])
        conv = sum(v["chains"][p[0]]["area"] == area for v in long_rows.values() for p in v["converging"])
        near = sum(v["chains"][p[0]]["area"] == area and p[2] < NEAR for v in long_rows.values()
                   for p in v["closest"])
        out["  {} pairs crossing / converging / within 3 cm".format(area)] = "{}/{}/{}".format(cross, conv, near)
    chains = [c for v in rows.values() for c in v["chains"]]
    out["chains"] = len(chains)
    out["chains over 3 cm off any hair"] = sum(c["off_any"] > NEAR for c in chains)
    out["chains over 3 cm off their own hair"] = sum(c["off_own"] > NEAR for c in chains)
    plain = [c for v in long_rows.values() for c in v["chains"] if c["area"] != "bangs"]
    out["stubs under 8 cm (long hairs, not bangs)"] = sum(c["length"] < STUB for c in plain)
    out["bends over 45 deg (long hairs, not bangs)"] = sum(bend_of(c) > BEND for c in plain)
    out["bones under 1 cm"] = sum(
        bool((np.linalg.norm(np.diff(np.array(c["joints"]), axis=0), axis=1) < MIN_BONE).any()) for c in chains)
    measured = [r for r in (rows_of(v) for v in long_rows.values()) if r is not None]
    out["long hairs with 2+ back chains"] = len(measured)
    out["  ...with unequal bone counts"] = sum(r[0] for r in measured)
    out["  ...with a row over 2 cm apart in height"] = sum(r[1] > ROW_SPREAD for r in measured)
    empty = {}
    for label, v in long_rows.items():
        for area in missing(v):
            empty.setdefault(area, []).append(label)
    out["long hairs: area has hair but no chain"] = sum(len(v) for v in empty.values())
    for area, labels in sorted(empty.items()):
        out["  ..." + area] = len(labels)
    short_of = 0
    for v in long_rows.values():
        for area, asked in v["asked"].items():
            if v.get("masses", {}).get(area, 0.0) >= HAS_HAIR and 0 < v["made"].get(area, 0) < asked:
                short_of += 1
    out["long hairs: area built fewer chains than asked"] = short_of
    out["short hairs with hanging hair but no chain"] = sum(
        v["short"] and v["hanging_area"] >= HAS_HAIR and not v["chains"] for v in rows.values())
    values = list(long_rows.values())
    has_far = values and all("far_area" in v for v in values)
    out["far share (long hairs)"] = "{:.1%}".format(sum(v["far_area"] for v in values) / max(
        sum(v["hanging_area"] for v in values), 1e-12)) if has_far else "-"
    for key in ("moved", "lifted"):
        has = values and all(key in v for v in values)
        out["joints the check {:s} (long hairs)".format(key)] = sum(v[key] for v in values) if has else "-"
    return out


def chain_list(v):
    far = " far {:.1%}".format(v["far_area"] / v["hanging_area"]) if "far_area" in v and v["hanging_area"] else ""
    return "{}{}".format(" | ".join("{} {}b {:.2f}m off {:.3f}".format(c["area"], c["bones"], c["length"],
                                                                       c["off_any"]) for c in v["chains"]), far)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ref")
    parser.add_argument("new")
    args = parser.parse_args()
    ref, ref_errors = load(args.ref)
    new, new_errors = load(args.new)
    both = [k for k in ref if k in new]
    ref, new = {k: ref[k] for k in both}, {k: new[k] for k in both}
    long_labels = {k for k in both if not ref[k]["short"] and not new[k]["short"]}
    names = [os.path.splitext(os.path.basename(path))[0][:12] for path in (args.ref, args.new)]
    if names[0] == names[1]:
        names = ["ref", "new"]
    print("ref {:s}\nnew {:s}".format(os.path.abspath(args.ref), os.path.abspath(args.new)))
    a, b = totals(ref, long_labels), totals(new, long_labels)
    print("{:52s} {:>12s} {:>12s}".format("", *names))
    for key in list(a) + [key for key in b if key not in a]:
        print("{:52s} {:>12} {:>12}".format(key, a.get(key, "-"), b.get(key, "-")))
    print("failed", ref_errors, new_errors)
    for label in NAMED:
        for name, rows in zip(names, (ref, new)):
            if label in rows:
                print("{:16s} {:12s} {}".format(label, name, chain_list(rows[label])))
    print("new crossing:", ", ".join(k for k in sorted(long_labels) if new[k]["crossing"]))
    print("new converging:", ", ".join(k for k in sorted(long_labels) if new[k]["converging"]))
    print("new off hair:", ", ".join("{} {:.3f}".format(k, max(c["off_any"] for c in v["chains"]))
                                     for k, v in new.items() if any(c["off_any"] > NEAR for c in v["chains"])))
    print("new areas with hair but no chain:", ", ".join("{} {}".format(k, "/".join(missing(new[k])))
                                                          for k in sorted(long_labels) if missing(new[k])))
    print("new short hairs without a chain:", ", ".join(
        k for k, v in new.items() if v["short"] and v["hanging_area"] >= HAS_HAIR and not v["chains"]))


main()
