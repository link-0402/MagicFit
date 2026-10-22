"""Add each race's scalp map (head_scalp) to magic_fit/hair_reference.npz, from the game's hair models and the
shape of the head.

Where a race's scalp ends (the hairline) comes from two sources:

- Back and sides, from its vanilla hairstyles (tools/hair_models.py): a head point's scalp share is the share
  of them that hold hair near it on j_kao (or the ear bones), high on the cranium, low on the neck.
- Front, from the head's shape: the game keeps its own fringes on j_kao over the brow, so the hairstyles alone
  score the brow like scalp, and a modded fringe lying in front of it would never hang. Head points in front
  (within FRONT_ANGLE of straight ahead, around the head's vertical axis through j_kao) and more than
  FRONT_DEPTH of the way from the head's top down to j_kao get share 0. The edge is soft around the head
  (over FRONT_ANGLE_SOFT to either side) and sharp in height.

The shares are stored per head point (head_points), as head_scalp ((races, 800), float32), races in the file's
order. Every other array is written back unchanged. The rules were tuned on Miqo'te, Au Ra and Viera women
(c0801, c1401, c1801) and the real hair DOUX Maeve (tests/test_hair_reference.py); the other races get the
same rules untested.

The hair models come from tools/extract_hair_models.ps1 (<folder>/<race>/<race>hNNNN_hir.mdl); they are
decoded with Yet Another Addon's xivpy (it must be installed in Blender).

Run with Blender:
    blender -b --factory-startup --python tools/make_scalp_map.py -- "<hair model folder>"
"""

import glob
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS)

import hair_models  # noqa: E402
from magic_fit import hairing  # noqa: E402

# The front of the face (degrees around the head from straight ahead, soft over FRONT_ANGLE_SOFT to either
# side) below FRONT_DEPTH of the way from the head's top down to j_kao is no scalp.
FRONT_ANGLE = 34.0
FRONT_ANGLE_SOFT = 5.0
FRONT_DEPTH = 0.6


def front_rule(share, points, kao):
    """``share`` with the front of the face taken out (see FRONT_ANGLE, FRONT_DEPTH)."""
    top = points[:, 2].max()
    around = np.degrees(np.arctan2(np.abs(points[:, 0]), -(points[:, 1] - kao[1])))
    front = np.clip((FRONT_ANGLE + FRONT_ANGLE_SOFT - around) / (2.0 * FRONT_ANGLE_SOFT), 0.0, 1.0)
    low = (top - points[:, 2]) > FRONT_DEPTH * (top - kao[2])
    return (share * (1.0 - front * low)).astype(np.float32)


def summary(share, points, kao):
    """Mean scalp share of the crown (the top 5 % of the head points), the forehead (the front middle, 11 to
    16 cm below the head's top) and the neck (the lowest 5 % of the points behind j_kao)."""
    crown = share[points[:, 2] >= np.quantile(points[:, 2], 0.95)].mean()
    below = points[:, 2].max() - points[:, 2]
    forehead = share[(np.abs(points[:, 0]) < 0.03) & (points[:, 1] < kao[1]) & (below >= 0.11) & (below <= 0.16)]
    behind = points[:, 1] > kao[1]
    neck = share[behind & (points[:, 2] <= np.quantile(points[behind, 2], 0.05))]
    return crown, forehead.mean(), neck.mean()


def main(folder):
    with np.load(hairing.REFERENCE_FILE, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "head_scalp"}
    reference = hairing.Reference(hairing.REFERENCE_FILE)
    scalp = []
    for race in reference.races:
        head = reference.head(race)
        kao = reference.rig(race, 0).by_name[hairing.HEAD].head
        hairs, failed = [], []
        for path in sorted(glob.glob(os.path.join(folder, race, "%sh*_hir.mdl" % race))):
            with open(path, "rb") as handle:
                data = handle.read()
            try:
                hairs.append(hair_models.decode(data))
            except Exception as error:  # a model xivpy can't read is left out, and named
                failed.append("{:s} ({:s})".format(os.path.basename(path), str(error)))
        if not hairs:
            raise FileNotFoundError("no hair models of " + race + " in " + folder)
        share = front_rule(hair_models.scalp_shares(head.surface, hairs), head.surface, kao)
        scalp.append(share)
        crown, forehead, neck = summary(share, head.surface, kao)
        print("{:s} {:d} hair models, crown {:.2f}, forehead {:.2f}, neck {:.2f}".format(
            race, len(hairs), crown, forehead, neck))
        for name in failed:
            print("    not read:", name)
    np.savez_compressed(hairing.REFERENCE_FILE, **arrays, head_scalp=np.array(scalp, np.float32))
    print("wrote", hairing.REFERENCE_FILE, os.path.getsize(hairing.REFERENCE_FILE), "bytes")


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if len(arguments) != 1:
        sys.exit(__doc__)
    main(*arguments)
