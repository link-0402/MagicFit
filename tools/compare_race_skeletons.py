"""Print how FFXIV's race skeletons differ, as the Markdown tables of docs/race-skeletons.md.

Reads TexTools' .skel files (it writes one for every skeleton it exports a model with: cXXXXb0001 for a
race's body, cXXXXfYYYY for a face) from its Skeletons folder. Each line of a .skel file is a bone:
BoneName, BoneNumber, BoneParent (a BoneNumber, -1 for the root), PoseMatrix (the bone relative to its
parent) and InversePoseMatrix (the inverse of the bone in model space), both 4x4 matrices of row vectors
with the translation in the last row. The model space is the game's: metres, Y up, the character facing +Z.

Run with any Python 3 with numpy:
    python tools/compare_race_skeletons.py ["E:/FFXIVmods/FFXIV TexTools/FFXIV_TexTools/Skeletons"]
"""

import glob
import json
import os
import sys

import numpy as np

RACES = {
    "c0101": "Midlander M", "c0201": "Midlander F", "c0301": "Highlander M", "c0401": "Highlander F",
    "c0501": "Elezen M", "c0601": "Elezen F", "c0701": "Miqo'te M", "c0801": "Miqo'te F",
    "c0901": "Roegadyn M", "c1001": "Roegadyn F", "c1101": "Lalafell M", "c1201": "Lalafell F",
    "c1301": "Au Ra M", "c1401": "Au Ra F", "c1501": "Hrothgar M", "c1601": "Hrothgar F",
    "c1701": "Viera M", "c1801": "Viera F",
}
BASE = "c0201"
REGIONS = (
    ("brows", ("j_f_mayu", "j_f_mmayu", "j_f_miken", "j_f_dmiken")),
    ("eyes, lids", ("j_f_eye", "j_f_mab", "j_f_eyepr", "j_f_iris", "j_f_noanim_eyesize", "j_f_eyepuru")),
    ("nose", ("j_f_hana", "j_f_uhana")),
    ("cheeks", ("j_f_hoho", "j_f_dhoho", "j_f_shoho", "j_f_dmemoto")),
    ("lips", ("j_f_ulip", "j_f_dlip", "j_f_umlip", "j_f_dmlip", "j_f_uslip", "j_f_dslip")),
    ("jaw, teeth, tongue", ("j_f_ago", "j_f_dago", "j_f_haguki", "j_f_bero", "j_f_noanim_ago")),
)


def rotation(M):
    U, _S, Vt = np.linalg.svd(M[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def degrees(A, B):
    R = rotation(A).T @ rotation(B)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def load(path):
    """{bone: {"parent", "model"}}, "model" a 4x4 matrix of column vectors in model space."""
    entries = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    numbers = {entry["BoneNumber"]: entry["BoneName"] for entry in entries}
    return {entry["BoneName"]: {
        "parent": numbers.get(entry["BoneParent"]),
        "model": np.linalg.inv(np.array(entry["InversePoseMatrix"], dtype=float).reshape(4, 4)).T,
    } for entry in entries}


def in_head(skeleton, name):
    return np.linalg.inv(skeleton["j_kao"]["model"]) @ skeleton[name]["model"]


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else r"E:/FFXIVmods/FFXIV TexTools/FFXIV_TexTools/Skeletons"
    faces, bodies, files = {}, {}, {}
    for race in RACES:
        body = os.path.join(folder, race + "b0001.skel")
        face = sorted(glob.glob(os.path.join(folder, race + "f*.skel")))
        if os.path.exists(body):
            bodies[race] = load(body)
        if face:
            faces[race] = load(face[0])
            files[race] = os.path.basename(face[0])[:-5]

    base = faces[BASE]
    shared = [name for name in base if name.startswith("j_f_") and all(name in face for face in faces.values())]
    print("## Faces\n")
    print("| Race | Skeleton of face 1 | Bones | Head (j_kao) height | Face size | Median / max offset | "
          "Max after scaling | Turned > 2 deg | Max turn | Extra bones |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for race, face in faces.items():
        P = np.array([in_head(face, name)[:3, 3] for name in shared])
        Q = np.array([in_head(base, name)[:3, 3] for name in shared])
        P0, Q0 = P - P.mean(axis=0), Q - Q.mean(axis=0)
        size = float(np.sqrt((P0 ** 2).sum() / (Q0 ** 2).sum()))
        offsets = np.linalg.norm(P - Q, axis=1)
        scaled = np.linalg.norm(P0 - size * Q0, axis=1)
        turns = [degrees(base[name]["model"], face[name]["model"]) for name in shared]
        extra = sorted(name for name in face if name not in base)
        print("| {:s} | {:s} | {:d} | {:.3f} m | {:.2f} | {:.0f} / {:.0f} mm | {:.0f} mm | {:d} | {:.0f} deg | {:s} |".format(
            RACES[race], files[race], len(face), face["j_kao"]["model"][1, 3], size, 1000 * np.median(offsets),
            1000 * offsets.max(), 1000 * scaled.max(), sum(turn > 2.0 for turn in turns), max(turns),
            ", ".join(extra) if extra else "-"))

    print("\n### Largest turn from the Midlander female's, by region (degrees)\n")
    print("| Region | Bones | " + " | ".join(RACES[race] for race in faces) + " |")
    print("| --- | --- | " + " | ".join("---" for _race in faces) + " |")
    for label, prefixes in REGIONS:
        names = [name for name in shared if name.startswith(prefixes)]
        values = [max(degrees(base[name]["model"], face[name]["model"]) for name in names) for face in faces.values()]
        print("| {:s} | {:d} | {:s} |".format(label, len(names), " | ".join("{:.0f}".format(v) for v in values)))

    print("\n### Identical face skeletons\n")
    codes = list(faces)
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            same = all(np.linalg.norm(in_head(faces[a], name)[:3, 3] - in_head(faces[b], name)[:3, 3]) < 0.0005
                       and degrees(faces[a][name]["model"], faces[b][name]["model"]) < 0.5 for name in shared)
            if same:
                print("- {:s} and {:s}".format(RACES[a], RACES[b]))

    print("\n## Bodies\n")
    print("| Race | Bones | Hips (n_hara) | Head (j_kao) | Thighs (j_asi_a) | Upper arms apart (j_ude_a) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for race, body in bodies.items():
        print("| {:s} | {:d} | {:.3f} m | {:.3f} m | {:.3f} m | {:.3f} m |".format(
            RACES[race], len(body), body["n_hara"]["model"][1, 3], body["j_kao"]["model"][1, 3],
            body["j_asi_a_l"]["model"][1, 3], 2.0 * body["j_ude_a_l"]["model"][0, 3]))
    names = sorted(set().union(*[set(body) for body in bodies.values()]))
    print("\nBones not every race has:\n")
    for name in names:
        missing = [RACES[race] for race, body in bodies.items() if name not in body]
        if missing:
            print("- `{:s}`: not in {:s}".format(name, ", ".join(missing)))
    print("\nBones turned more than 5 degrees from the Midlander female's:\n")
    body_base = bodies[BASE]
    rows = []
    for name in body_base:
        groups = {}
        for race, body in bodies.items():
            if name in body:
                turn = degrees(body_base[name]["model"], body[name]["model"])
                if turn > 5.0:
                    groups.setdefault(int(round(turn)), []).append(RACES[race])
        if groups:
            rows.append((max(groups), name, groups))
    for _worst, name, groups in sorted(rows, reverse=True):
        print("- `{:s}`: {:s}".format(name, "; ".join("{:d} deg {:s}".format(turn, ", ".join(races))
                                                     for turn, races in sorted(groups.items(), reverse=True))))


main()
