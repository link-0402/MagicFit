"""Write magic_fit/cplus/reference.py: where FFXIV's skeleton puts each bone and how it turns it.

Customize+ moves, turns and scales each bone along the bone's own axes in the game. Blender armatures made
from game skeletons don't all keep those axes (importers swap them to make bones point at their
children), so Customize+ in Blender compares each bone with these to find out how its axes were swapped.

Sources:
  - TexTools' .skel files of every race's body (cXXXXb0001) and face skeleton (cXXXXfYYYY, the ones
    TexTools saved when exporting a face): the vanilla body and face bones. TexTools writes them to its
    Skeletons folder whenever it exports a model.
  - An armature in the open .blend with the vanilla bones on the game's axes (such as Yet Another
    Devkit's Skeleton): IVCS and YAS bones (iv_*, ya_*) and anything else it has.
Positions come from the Midlander female (c0201) where it has the bone, else from the first race that
does; they only serve to find how the armature is placed. Turns differ between races by up to 180
degrees for a few bones (Roegadyn and Hrothgar males' n_buki_tate_r) and up to 41 degrees for face bones
(Hrothgar and Lalafell lips and eyelids), so every race's are kept, and the armature's race is the one
that fits it best. Hair skeletons differ for every hairstyle and are left out.

Run with Blender, the devkit open:
    blender -b --factory-startup "Yet Another Devkit.blend" --python tools/make_cplus_reference.py -- \
        "E:/FFXIVmods/FFXIV TexTools/FFXIV_TexTools/Skeletons" Skeleton
"""

import glob
import json
import os
import sys
import textwrap

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "magic_fit", "cplus", "reference.py")
# Game model space (Y up, facing +Z) to Blender (Z up, facing -Y).
TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
PRIMARY = "c0201"
# Turns closer than this count as the same.
SAME_DEGREES = 1.0
RACES = {
    "c0101": "Hyur Midlander male", "c0201": "Hyur Midlander female", "c0301": "Hyur Highlander male",
    "c0401": "Hyur Highlander female", "c0501": "Elezen male", "c0601": "Elezen female",
    "c0701": "Miqo'te male", "c0801": "Miqo'te female", "c0901": "Roegadyn male", "c1001": "Roegadyn female",
    "c1101": "Lalafell male", "c1201": "Lalafell female", "c1301": "Au Ra male", "c1401": "Au Ra female",
    "c1501": "Hrothgar male", "c1601": "Hrothgar female", "c1701": "Viera male", "c1801": "Viera female",
}


def quaternion(R):
    """(w, x, y, z) of rotation matrix ``R`` (Shepperd's method: also right for half turns)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def orthonormal(M):
    U, _S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def degrees(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1.0) / 2.0, -1.0, 1.0))))


def from_skel(path):
    """{bone: (position, rotation)} in game model space from a TexTools .skel file."""
    bones = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            # Row vectors, translation in the last row: the inverse bind matrix.
            M = np.linalg.inv(np.array(entry["InversePoseMatrix"], dtype=float).reshape(4, 4)).T
            bones[entry["BoneName"]] = (M[:3, 3], orthonormal(M[:3, :3]))
    return bones


def from_armature(obj):
    """{bone: (position, rotation)} in game model space from an armature on the game's axes."""
    bones = {}
    for bone in obj.data.bones:
        M = np.array(bone.matrix_local)
        bones[bone.name] = (TO_BLENDER.T @ M[:3, 3], orthonormal(TO_BLENDER.T @ M[:3, :3]))
    return bones


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    skeletons = args[0] if args else r"E:/FFXIVmods/FFXIV TexTools/FFXIV_TexTools/Skeletons"
    armature = bpy.data.objects[args[1] if len(args) > 1 else "Skeleton"]

    races = {}  # race: {bone: (position, rotation)}, body and face skeletons merged
    files = {}
    for race in RACES:
        paths = sorted(glob.glob(os.path.join(skeletons, race + "b0001.skel")) +
                       glob.glob(os.path.join(skeletons, race + "f*.skel")))
        if not paths:
            continue
        merged = {}
        for path in paths:
            for name, value in from_skel(path).items():
                merged.setdefault(name, value)
        races[race] = merged
        files[race] = [os.path.basename(path)[:-5] for path in paths]
    if PRIMARY not in races:
        raise SystemExit("TexTools' {:s}b0001.skel is missing".format(PRIMARY))

    devkit = from_armature(armature)
    worst = max(degrees(races[PRIMARY][name][1], rotation) for name, (_position, rotation) in devkit.items()
                if name in races[PRIMARY])
    if worst > 10.0:
        raise SystemExit("'{:s}' isn't on the game's axes ({:.1f} degrees off)".format(armature.name, worst))

    # Positions and first turns: the primary race, then the others, then the devkit's extra bones.
    table = {}
    order = [PRIMARY] + [race for race in races if race != PRIMARY]
    for race in order:
        for name, value in races[race].items():
            table.setdefault(name, value)
    extra = [name for name in devkit if name not in table]
    for name in extra:
        table[name] = devkit[name]

    # Every race's distinct turns.
    turns = {name: [table[name][1]] for name in table}
    indices = {race: {} for race in races}
    for race in races:
        for name, (_position, rotation) in races[race].items():
            options = turns[name]
            for index, other in enumerate(options):
                if degrees(other, rotation) < SAME_DEGREES:
                    break
            else:
                options.append(rotation)
                index = len(options) - 1
            if index:
                indices[race][name] = index

    sources = ("Written by tools/make_cplus_reference.py from TexTools' skeleton files ({:s}) and the "
               "armature '{:s}' of {:s} (IVCS and YAS bones).").format(
        ", ".join(name for race in races for name in files[race]), armature.name, os.path.basename(bpy.data.filepath))
    lines = [
        "# SPDX-License-Identifier: GPL-3.0-or-later",
        "",
        '"""FFXIV\'s skeletons in the game\'s model space (metres, Y up, facing +Z).',
        "",
    ] + textwrap.wrap(sources, 104) + [
        "",
        "BONES   {bone: (x, y, z, qw, qx, qy, qz)}: where the Midlander female's skeleton (or the first race",
        "        with the bone) puts it, and how it turns it.",
        "TURNS   {bone: ((qw, qx, qy, qz), ...)}: the bone's other turns in other races.",
        "RACES   {race: {bone: index}}: the races' turns that aren't the one in BONES (1 = TURNS[bone][0]).",
        "SAMPLED the bones only one skeleton had (IVCS and YAS, from the devkit): other skeletons of those",
        "        mods may turn them differently.",
        '"""',
        "",
        "NAMES = {",
    ]
    lines += ["    {!r}: {!r},".format(race, RACES[race]) for race in races]
    lines += ["}", "", "BONES = {"]
    for name in sorted(table):
        position, rotation = table[name]
        values = ", ".join("{:.5f}".format(v + 0.0) for v in list(position) + list(quaternion(rotation)))
        lines.append("    {!r}: ({:s}),".format(name, values.replace("-0.00000", "0.00000")))
    lines += ["}", "", "TURNS = {"]
    for name in sorted(turns):
        if len(turns[name]) < 2:
            continue
        quats = ", ".join("({:s})".format(", ".join("{:.5f}".format(v + 0.0) for v in quaternion(R)))
                          for R in turns[name][1:])
        lines.append("    {!r}: ({:s},),".format(name, quats.replace("-0.00000", "0.00000")))
    lines += ["}", "", "RACES = {"]
    for race in races:
        pairs = ", ".join("{!r}: {:d}".format(name, index) for name, index in sorted(indices[race].items()))
        lines.append("    {!r}: {{{:s}}},".format(race, pairs))
    lines += ["}", "", "SAMPLED = frozenset(("]
    lines += ["    " + line for line in textwrap.wrap(" ".join("{!r},".format(name) for name in sorted(extra)), 100)]
    lines += ["))", ""]
    with open(OUT, "w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines))
    counts = {}
    for name in turns:
        counts[len(turns[name])] = counts.get(len(turns[name]), 0) + 1
    print("wrote {:d} bones ({:d} from the devkit) of {:d} races to {:s}; turns per bone: {}".format(
        len(table), len(extra), len(races), OUT, dict(sorted(counts.items()))))


main()
