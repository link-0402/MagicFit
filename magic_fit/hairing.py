# SPDX-License-Identifier: GPL-3.0-or-later

"""Hair weights: picking an FFXIV hair skeleton for new hair, and giving the hair its weights.

FFXIV moves hair with bones the game swings with its hair physics. Every race's body has four of them,
j_kami_a and j_kami_b down the back of the head and j_kami_f_l / j_kami_f_r at the temples, and each
hairstyle can load a hair skeleton on top (chara/human/cXXXX/skeleton/hair/hNNNN), picked by the hair's EST
entry: about a hundred per race, with bones like j_ex_h0170_ke_a (a chain down the back), _ke_f (bangs) or
_side_a01_l. Every hair bone points along its local X axis, toward the hair it moves, and swings about its
head.

Hair Weights needs no weights and no armature:

- The race's head (a signed distance grid of the game's face model, see tools/make_hair_reference.py)
  tells how far each vertex hangs from the scalp. Hair on the scalp follows the head; hair farther out
  hangs from the hair bones: none of it closer than Hang Start, 63 % of it Hang Length farther out and
  95 % three times as far (1 - exp(-(distance - start) / length)), as the game's hair and well-made
  modded hair both do on average.
- The hanging part goes to a chain of hair bones (a chain runs from a bone hanging from the head down
  its children, and on past the last one along its axis), by a score of how close the vertex is to
  it, how far around the head from it, whether it falls on the side of the head the chain's hair
  goes to (front or back), how far below the chain's end it hangs (bangs chains reach less far) and
  whether it lies above the chain's start; the body's hair bones count for less. The score is fitted
  on which chains modded hair and the game's hair give their hanging hair: most modders split long
  hair at the ears, the back on the back chain, the front locks on j_kami_f_l/r. Each loose part (a
  strand or card) takes its vertices' choice as a whole when they mostly agree, so strands don't tear
  between chains.
- Along a chain, the bones share the hair by how far down it the vertex is, blending around each
  joint. When the hair is longer than the chain, the joints move part of the way (Chain Stretch)
  toward being spread evenly over its length, as modders weight long hair: the tips then swing with
  the whole chain instead of flailing from its last bone.
- What stays with the head is shared with the neck (j_kubi) below the head's pivot and the upper back
  (j_sebo_c) below the neck (Neck Weight). The game's Miqo'te hair has the cat ears: a hair that has
  them (a part standing out along an ear bone) gives them to the ear bones (Ears); hair that only
  covers the scalp there stays with the head.

The best skeleton for a hair is found by a score of how its chains fit the hair: how close the hanging
hair is to a chain and how far it hangs past one, how much of it each bone would move, how many bones
there are, how far down and how far behind the head they reach compared with the hair. Its weights were fitted on 1,870
vanilla hairs (to behave like the game's own weights and skeletons when the head tilts and turns) and
116 long modded hairs (to pick what modders picked for them); see docs/development.md.

Like `painting`, nothing here depends on the 3D view.
"""

import math
import os

import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

from . import fitting, goals, straighten
from .goals import DEFAULT_MAX_GROUPS, smoothstep

REFERENCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hair_reference.npz")
RACE_NAMES = {
    "c0101": "Hyur Midlander male", "c0201": "Hyur Midlander female", "c0301": "Hyur Highlander male",
    "c0401": "Hyur Highlander female", "c0501": "Elezen male", "c0601": "Elezen female",
    "c0701": "Miqo'te male", "c0801": "Miqo'te female", "c0901": "Roegadyn male", "c1001": "Roegadyn female",
    "c1101": "Lalafell male", "c1201": "Lalafell female", "c1301": "Au Ra male", "c1401": "Au Ra female",
    "c1501": "Hrothgar male", "c1601": "Hrothgar female", "c1701": "Viera male", "c1801": "Viera female",
}
HEAD = "j_kao"
NECK = "j_kubi"
UPPER_BACK = "j_sebo_c"
# The body's own hair bones, in every race's skeleton.
BODY_HAIR = ("j_kami_a", "j_kami_b", "j_kami_f_l", "j_kami_f_r")
BACK_HAIR = frozenset(("j_kami_a", "j_kami_b"))
FRONT_HAIR = frozenset(("j_kami_f_l", "j_kami_f_r"))
# Miqo'te hair can have the cat ears, which follow the ear bones.
EARS = ("j_mimi_l", "j_mimi_r")
EAR_RACES = frozenset(("c0701", "c0801"))
EAR_REACH = 0.12
# A hair has an ear when at least EAR_AREA of it (m²) lies more than EAR_FOUND out along the ear bone and
# within EAR_FOUND_RADIUS of it: the game's ears have 52 cm² and more there, long modded hair without
# ears 22 cm² at most.
EAR_AREA = 0.0035
EAR_FOUND = 0.04
EAR_FOUND_RADIUS = 0.025
# The ear bone takes its ear from none at its head to all EAR_RISE out along it, within EAR_RADIUS of it
# and fading out over EAR_FADE beyond: close to how the game weights its ears, and never abruptly.
EAR_RISE = 0.04
EAR_RADIUS = 0.015
EAR_FADE = 0.03
# Hair bones point along their local X axis; a bone without children reaches this far for the score.
LEAF_REACH = 0.10
# Which chain takes a vertex's hanging part: chains share it by exp(-score), the score being the terms
# (see `chain_terms`) times CHAIN_WEIGHTS:
#   distance    metres from the chain (its last bone going on along its axis)
#   around      metres around the head from where the chain is at the vertex's height
#   side        how far around the head the vertex is from where the chain's hair goes (1 - cos: 0 on
#               that side, 2 opposite)
#   past        metres below the chain's lowest joint
#   past_bangs  the same, for a chain at the front middle (bangs)
#   above       metres above the chain's first joint
#   back_body   1 for j_kami_a/b when the hair skeleton has chains of its own
#   front_body  1 for j_kami_f_l/r
# Fitted on 558 modded and game hairs (tools/fit_hair_matching.py measure-chains and fit-chains), then
# made twice as sharp, and back_body half as big again (modders use a skeleton's own chains more than
# the game does): that moves hair most like theirs when each chain swings on its own.
CHAIN_TERMS = ("distance", "around", "side", "past", "past_bangs", "above", "back_body", "front_body")
CHAIN_WEIGHTS = np.array((56.102, 15.306, 1.4592, 19.051, 23.882, 115.35, 4.7798, 5.0543))
# Where a chain's hair goes: its mean direction around the head from its first joint down to this far
# below the head bone.
CHAIN_HOME_DEPTH = 0.3
# A loose part takes its vertices' choice of chain when that many of them (by area) agree...
PART_AGREEMENT = 0.7
# ...fully from this much more on.
PART_AGREEMENT_RANGE = 0.2
# Joints blend over at least this, and at least the distance to the next joint times JOINT_SPREAD.
JOINT_SPREAD = 1.0
# A chain's hair length: where this share of the hair assigned to it (by hanging weight) ends.
HAIR_LENGTH_SHARE = 0.98
# Bones placed at a Blender armature: they point along their own Y axis, like the game's; the length
# only matters for display.
BONE_LENGTH = 0.05
# Shares below this don't count.
SHARE_EPSILON = 1e-6
# The score looks at this many triangles of the hair at most (picked at random, by area).
SCORE_TRIANGLES = 8000

# The skeleton score, in two steps: every skeleton by SCORE_FEATURES (cheap), then the SHORTLIST best of
# them by FULL_FEATURES, which also pose the hair with each skeleton's weights. Both are standardised
# features (see `_features`) times weights, fitted with tools/fit_hair_matching.py (see docs/development.md):
# vanilla hairs by how their weights behave when the head tilts and turns, modded hairs by the skeleton
# their modder picked.
SCORE_FEATURES = (
    "cov_mse", "uncovered5", "uncovered10", "n_bones", "log_bones", "useless2", "useless5", "bone_to_hair",
    "bone_to_free", "misalign", "upstream", "entropy", "is_none", "overhang", "mean_assigned_dist",
    "kami_mass", "hair_len", "skel_reach", "chain_depth", "reach_gap", "depth_x_len", "back_depth", "n_paths",
    "back_gap", "back_gap_abs",
)
SCORE_WEIGHTS = np.array((
    -0.3937, 0.2063, -0.4393, -1.3956, -2.4834, 0.7654, -0.4443, -0.3137, -0.2967, 0.4409, -0.9289, 0.3361, -0.9155,
    0.2968, -1.969, 1.3601, 0.0, 0.5422, 0.2179, -0.3093, 0.8893, 0.3768, 2.0148, -0.3904, -0.0502,
))
SCORE_MEAN = np.array((
    0.0069773, 0.81193, 0.4735, 4.4062, 1.5754, 0.18203, 0.30902, 0.023249, 0.17231, 0.46946, 0.12391, 0.66855,
    0.010639, 0.024731, 0.103, 0.33375, 0.09979, 0.040594, 1.2505, 0.059196, 0.12575, 1.0065, 3.9885, 0.061372,
    0.096525,
))
SCORE_STD = np.array((
    0.0027232, 0.25366, 0.35234, 3.378, 0.43926, 0.24414, 0.29111, 0.019003, 0.036534, 0.39722, 0.21057, 0.27911,
    0.1026, 0.034419, 0.049028, 0.28314, 0.13698, 0.093992, 0.59939, 0.16478, 0.20091, 0.77803, 2.5249, 0.11,
    0.080928,
))
FULL_FEATURES = SCORE_FEATURES[:15] + ("stretch", "lag") + SCORE_FEATURES[15:]
FULL_WEIGHTS = np.array((
    -0.6387, 0.4963, 0.0082, -0.6173, -1.7582, 0.4567, -0.3754, -0.1456, -0.2252, -0.013, -0.5573, 0.3106, -0.4823,
    0.1048, -1.2028, -2.7551, 0.4153, 0.5394, 0.0, 0.0918, 0.4742, -0.0524, 0.8251, 0.059, 1.1502, -0.4203, -0.1655,
))
FULL_MEAN = np.array((
    0.0069773, 0.81193, 0.4735, 4.4062, 1.5754, 0.18203, 0.30902, 0.023249, 0.17231, 0.46946, 0.12391, 0.66855,
    0.010639, 0.024731, 0.103, 0.023648, 0.014284, 0.33375, 0.09979, 0.040594, 1.2505, 0.059196, 0.12575, 1.0065,
    3.9885, 0.061372, 0.096525,
))
FULL_STD = np.array((
    0.0027232, 0.25366, 0.35234, 3.378, 0.43926, 0.24414, 0.29111, 0.019003, 0.036534, 0.39722, 0.21057, 0.27911,
    0.1026, 0.034419, 0.049028, 0.013451, 0.012238, 0.28314, 0.13698, 0.093992, 0.59939, 0.16478, 0.20091, 0.77803,
    2.5249, 0.11, 0.080928,
))
SHORTLIST = 10
# The poses the full score tilts and turns the head in (axis, degrees): the neck takes a third of each,
# the head the rest, and the hair bones lag behind, back part of the way (POSE_LAG) toward how they hung.
POSES = (((1.0, 0.0, 0.0), 30.0), ((1.0, 0.0, 0.0), -30.0), ((0.0, 1.0, 0.0), 30.0), ((0.0, 1.0, 0.0), -30.0),
         ((0.0, 0.0, 1.0), 45.0))
POSE_LAG = 0.6
# The neck weight of the hair the full score compares with, rigid on the head and neck.
POSE_NECK_WEIGHT = 0.5
# Hair of which less than this share (by area) hangs has nothing for hair bones to swing: no skeleton
# fits it better than none.
MIN_HANGING = 0.01
# The score's own hanging share, fixed: the one it was fitted with.
SCORE_HANG_START = 0.025
SCORE_HANG_LENGTH = 0.08
# The Joint Blend of the weights the full score poses, fixed: the one it was fitted with.
SCORE_JOINT_BLEND = 0.10
# Handicap of all four body hair bones in the score's distances when a hair skeleton has chains.
SCORE_BODY_HAIR_HANDICAP = 0.08


class HairError(Exception):
    """Why hair weights can't be worked out; the message is meant for the user."""


# -----------------------------------------------------------------------------
# Reference data

class Bone:
    """A bone of a race's skeleton: ``name``, ``parent`` (name or None), ``rest`` (4x4, Blender model
    space: rotation and position of the game's bind pose, Z up), ``head`` and ``x`` (its unit X axis,
    along which a hair bone points)."""

    __slots__ = ("name", "parent", "rest", "head", "x")

    def __init__(self, name, parent, rest):
        self.name = name
        self.parent = parent
        self.rest = np.asarray(rest, dtype=np.float64)
        self.head = self.rest[:3, 3].copy()
        x = self.rest[:3, 0]
        self.x = x / max(np.linalg.norm(x), 1e-12)


class HeadField:
    """The signed distance to a race's head (metres, negative inside): from a grid of 1 cm cells near the
    head, and from points spread over its surface beyond the grid's reach (12.7 cm)."""

    def __init__(self, origin, step, grid, surface):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.step = float(step)
        self.grid = np.asarray(grid, dtype=np.float32) / 1000.0
        self.reach = float(self.grid.max()) - 0.002
        self.surface = np.asarray(surface, dtype=np.float64)
        self._tree = None

    def distance(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        shape = np.array(self.grid.shape)
        f = (points - self.origin) / self.step
        i0 = np.floor(f).astype(np.int64)
        t = f - i0
        inside = np.all((i0 >= 0) & (i0 < shape - 1), axis=1)
        result = np.full(len(points), np.inf)
        i0 = i0[inside]
        t = t[inside]
        acc = np.zeros(len(i0))
        for dx in (0, 1):
            wx = t[:, 0] if dx else 1.0 - t[:, 0]
            for dy in (0, 1):
                wy = t[:, 1] if dy else 1.0 - t[:, 1]
                for dz in (0, 1):
                    wz = t[:, 2] if dz else 1.0 - t[:, 2]
                    acc += wx * wy * wz * self.grid[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        result[inside] = acc
        far = ~(result < self.reach)
        if far.any():
            if self._tree is None:
                self._tree = _kd_tree(self.surface)
            nearest = np.array([self._tree.find(Vector(p))[2] for p in points[far]])
            result[far] = np.maximum(nearest, self.reach)
        return result


class Reference:
    """FFXIV's hair skeletons, the bones hair hangs from and the heads of every race (hair_reference.npz)."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls(REFERENCE_FILE)
        return cls._instance

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            d = {key: data[key] for key in data.files}
        self.races = tuple(str(r) for r in d["races"])
        names = [str(n) for n in d["bone_names"]]
        self._base = {}
        for race, bone, parent, rest in zip(d["base_race"], d["base_bone"], d["base_parent"], d["base_rest"]):
            self._base.setdefault(self.races[race], []).append(
                Bone(names[bone], names[parent] if parent >= 0 else None, rest))
        self._skeletons = {}
        for race, number, start, count in zip(d["skel_race"], d["skel_id"], d["skel_start"], d["skel_count"]):
            bones = [Bone(names[d["sb_bone"][i]], names[d["sb_parent"][i]], d["sb_rest"][i])
                     for i in range(start, start + count)]
            self._skeletons[(self.races[race], int(number))] = bones
        self._est = {}
        for race, hair_set, skeleton in zip(d["est_race"], d["est_set"], d["est_skel"]):
            self._est.setdefault((self.races[race], int(skeleton)), []).append(int(hair_set))
        self._heads = {}
        step = float(d["sdf_step"])
        for index, race in enumerate(self.races):
            shape = tuple(int(s) for s in d["sdf_shape"][index])
            start = int(d["sdf_offset"][index])
            grid = d["sdf_data"][start:start + int(np.prod(shape))].reshape(shape)
            self._heads[race] = HeadField(d["sdf_origin"][index], step, grid, d["head_points"][index])

    def skeletons(self, race):
        """The hair skeleton numbers the game has for ``race``, ascending."""
        return sorted(number for r, number in self._skeletons if r == race)

    def has_skeleton(self, race, number):
        return (race, number) in self._skeletons

    def vanilla_hairs(self, race, number):
        """The game's hairstyles (hair set numbers) of ``race`` whose EST entry is ``number``."""
        return sorted(self._est.get((race, number), ()))

    def head(self, race):
        return self._heads[race]

    def body_bones(self, race):
        return list(self._base[race])

    def rig(self, race, number):
        """The bones hair of ``race`` can use with hair skeleton ``number`` (0: none, only the body's)."""
        bones = self.body_bones(race)
        have = {bone.name for bone in bones}
        skeleton = []
        for bone in self._skeletons.get((race, number), ()):
            if bone.name not in have:
                bones.append(bone)
                have.add(bone.name)
                skeleton.append(bone.name)
        return HairRig(race, number, bones, self.head(race), skeleton)


def race_items():
    """Enum items of the races (code, name, description)."""
    return [(code, name, "Hair for {:s}s ({:s})".format(name, code)) for code, name in RACE_NAMES.items()]


def skeleton_label(number):
    return "h{:04d}".format(number) if number > 0 else "None"


# -----------------------------------------------------------------------------
# Rig

class HairRig:
    """The bones a hair can hang from, for one race and hair skeleton, in Blender model space (moved by
    ``offset`` when the hair's armature has its head elsewhere). ``skeleton`` names the bones of the hair
    skeleton (the others are the body's)."""

    def __init__(self, race, number, bones, head, skeleton, offset=None):
        self.race = race
        self.number = number
        self.bones = bones
        self.by_name = {bone.name: bone for bone in bones}
        self.head_field = head
        self.skeleton = list(skeleton)
        self.offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)

    def moved(self, offset):
        return HairRig(self.race, self.number, self.bones, self.head_field, self.skeleton, offset)

    def hair_bones(self):
        """The bones hair hangs from: the body's j_kami_* and the hair skeleton's, in rig order."""
        own = set(self.skeleton)
        return [bone for bone in self.bones if bone.name in BODY_HAIR or bone.name in own]

    def skeleton_bones(self):
        own = set(self.skeleton)
        return [bone for bone in self.bones if bone.name in own]

    def head_distance(self, points):
        return self.head_field.distance(np.asarray(points, dtype=np.float64) - self.offset)

    def head_height(self, name=HEAD):
        return float(self.by_name[name].head[2] + self.offset[2])

    def model(self, points):
        """World space ``points`` in the rig's model space."""
        return np.asarray(points, dtype=np.float64) - self.offset


# -----------------------------------------------------------------------------
# Geometry helpers

def vertex_areas(co, tris):
    if not len(tris):
        return np.zeros(len(co))
    a = np.linalg.norm(np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]]), axis=1) / 6.0
    areas = np.zeros(len(co))
    for k in range(3):
        np.add.at(areas, tris[:, k], a)
    return areas


def components(count, edges):
    """Connected component of each of ``count`` nodes joined by ``edges`` ((m, 2)), numbered from 0."""
    return fitting._connected(count, np.asarray(edges, dtype=np.int64).reshape(-1, 2))[0]


def hanging_share(distance, start, length):
    """The share of hair ``distance`` from the scalp that hangs from the hair bones."""
    return 1.0 - np.exp(-np.maximum(0.0, np.asarray(distance, dtype=np.float64) - start) / max(length, 1e-6))


def neck_split(points, rig, neck_weight):
    """Shares of what stays with the body: (head, neck, upper back), by height between the neck's pivot
    (0) and the head's (1): the neck takes ``neck_weight`` from a quarter of the way up on down, the
    upper back takes it over from a quarter below the neck's pivot."""
    head = rig.head_height(HEAD)
    neck = rig.head_height(NECK)
    h = (np.asarray(points)[:, 2] - neck) / max(head - neck, 1e-6)
    below_head = 1.0 - smoothstep((h - 0.25) / 0.75)
    back = smoothstep((-h - 0.25) / 0.75)
    neck_share = neck_weight * below_head * (1.0 - back)
    back_share = neck_weight * back
    return 1.0 - neck_share - back_share, neck_share, back_share


# -----------------------------------------------------------------------------
# Chains

class Chain:
    """Bones from one hanging from the body down to a bone without children: ``bones`` (Bone), their
    ``heads`` and ``axes`` ((k, 3)), the length of each segment (``seg``, k - 1) and the distance down
    the chain of each joint (``joints``, k, from 0)."""

    def __init__(self, bones):
        self.bones = bones
        self.heads = np.array([bone.head for bone in bones])
        self.axes = np.array([bone.x for bone in bones])
        self.seg = np.array([max(float(np.dot(self.heads[k + 1] - self.heads[k], self.axes[k])), 1e-4)
                             for k in range(len(bones) - 1)])
        self.joints = np.concatenate([[0.0], np.cumsum(self.seg)])
        self.root = bones[0].name

    def project(self, points):
        """Distance of ``points`` to the chain (its last bone going on along its axis) and how far down
        the chain their closest point is."""
        best = np.full(len(points), np.inf)
        down = np.zeros(len(points))
        count = len(self.bones)
        for k in range(count):
            v = points - self.heads[k]
            t = v @ self.axes[k]
            t = np.clip(t, 0.0, self.seg[k] if k < count - 1 else np.inf)
            d = np.linalg.norm(v - t[:, None] * self.axes[k][None], axis=1)
            closer = d < best
            best[closer] = d[closer]
            down[closer] = self.joints[k] + t[closer]
        return best, down


def chains_of(bones):
    """Chains through ``bones`` (Bone): one per bone without children, from its topmost ancestor among
    ``bones``."""
    names = {bone.name for bone in bones}
    children = {bone.name: [c for c in bones if c.parent == bone.name] for bone in bones}
    result = []

    def walk(bone, path):
        path = path + [bone]
        if not children[bone.name]:
            result.append(Chain(path))
        for child in children[bone.name]:
            walk(child, path)

    for bone in bones:
        if bone.parent not in names:
            walk(bone, [])
    return result


def segment_shares(down, joints, widths):
    """Shares of a chain's bones ((n, k)) at ``down`` along it: the joints (all but the first) blend
    over ``widths``, a smoothstep centred on each."""
    n = len(down)
    k = len(joints)
    below = np.ones((n, k))
    for j in range(1, k):
        width = max(widths[j], 1e-4)
        below[:, j] = smoothstep((down - joints[j] + 0.5 * width) / width)
    below = np.minimum.accumulate(below, axis=1)
    return below - np.concatenate([below[:, 1:], np.zeros((n, 1))], axis=1)


def chain_at(chain, heights):
    """Points of ``chain`` at ``heights``: on its bones between their joints, below the last joint on
    along the last bone's axis (if it points down), and at the first joint otherwise."""
    heads = chain.heads
    heights = np.asarray(heights, dtype=np.float64)
    result = np.repeat(heads[:1], len(heights), axis=0)
    done = np.zeros(len(heights), dtype=bool)
    for k in range(len(heads) - 1):
        a, b = heads[k], heads[k + 1]
        low, high = min(a[2], b[2]), max(a[2], b[2])
        if high - low <= 1e-6:
            continue
        on = ~done & (heights >= low) & (heights <= high)
        if on.any():
            t = (heights[on] - a[2]) / (b[2] - a[2])
            result[on] = a + t[:, None] * (b - a)
            done |= on
    below = ~done & (heights < heads[:, 2].min())
    if below.any():
        last, axis = heads[-1], chain.axes[-1]
        result[below] = last + ((heights[below] - last[2]) / axis[2])[:, None] * axis if axis[2] < -0.3 else last
    return result


def _around(offset_a, offset_b):
    """Angles (radians) around the head between horizontal offsets from its axis ((n, 2) each)."""
    d = np.abs(np.arctan2(offset_a[:, 0], offset_a[:, 1]) - np.arctan2(offset_b[:, 0], offset_b[:, 1]))
    return np.minimum(d, 2.0 * np.pi - d)


def chain_terms(model, rig, chains):
    """The terms of each chain's score (CHAIN_TERMS) for ``model`` in the rig's model space:
    (n, len(chains), len(CHAIN_TERMS))."""
    head = rig.by_name[HEAD].head
    axis = np.array((0.0, head[1]))
    offset = model[:, :2] - axis
    radius = np.linalg.norm(offset, axis=1)
    toward = offset / np.maximum(radius, 1e-6)[:, None]
    own = any(chain.root not in BODY_HAIR for chain in chains)
    terms = np.zeros((len(model), len(chains), len(CHAIN_TERMS)))
    for c, chain in enumerate(chains):
        distance, _down = chain.project(model)
        at = chain_at(chain, model[:, 2])[:, :2] - axis
        around = _around(offset, at) * np.minimum(radius, np.linalg.norm(at, axis=1))
        path = chain_at(chain, np.linspace(chain.heads[:, 2].max(), head[2] - CHAIN_HOME_DEPTH, 12))[:, :2] - axis
        home = (path / np.maximum(np.linalg.norm(path, axis=1), 1e-6)[:, None]).mean(0)
        strength = float(np.linalg.norm(home))
        home = home / strength if strength > 1e-3 else np.array((0.0, 1.0))
        side = (1.0 - toward @ home) * np.clip(radius / 0.05, 0.0, 1.0) * min(strength, 1.0)
        # Bangs: a chain starting at the front middle that stays near it (side-swept bangs can hang long).
        bangs = abs(chain.heads[0, 0]) < 0.02 and abs(path[:, 0].mean()) < 0.05 and home[1] < -0.5
        past = np.maximum(0.0, chain.heads[:, 2].min() - model[:, 2])
        above = np.maximum(0.0, model[:, 2] - chain.heads[:, 2].max())
        terms[:, c, 0] = distance
        terms[:, c, 1] = around
        terms[:, c, 2] = side
        terms[:, c, 3] = past
        terms[:, c, 4] = past if bangs else 0.0
        terms[:, c, 5] = above
        terms[:, c, 6] = float(own and chain.root in BACK_HAIR)
        terms[:, c, 7] = float(chain.root in FRONT_HAIR)
    return terms


def chain_shares(model, rig, chains):
    """How much of each vertex's hanging part each chain takes ((n, len(chains)), rows adding up to 1),
    for ``model`` in the rig's model space (see CHAIN_WEIGHTS)."""
    score = chain_terms(model, rig, chains) @ CHAIN_WEIGHTS
    share = np.exp(-(score - score.min(1, keepdims=True)))
    return share / share.sum(1, keepdims=True)


# -----------------------------------------------------------------------------
# Weights

def ear_shares(points, tris, rig):
    """{ear bone: share of each vertex} of the cat ears the hair ``points`` (rig model space) with
    triangles ``tris`` has: nothing for other races, or for an ear the hair doesn't have."""
    result = {}
    if rig.race not in EAR_RACES:
        return result
    area = vertex_areas(points, np.asarray(tris, dtype=np.int64).reshape(-1, 3))
    for name in EARS:
        bone = rig.by_name.get(name)
        if bone is None:
            continue
        v = points - bone.head
        t = v @ bone.x
        d = np.linalg.norm(v - np.clip(t, 0.0, EAR_REACH)[:, None] * bone.x[None], axis=1)
        if area[(t > EAR_FOUND) & (t < EAR_REACH) & (d < EAR_FOUND_RADIUS)].sum() < EAR_AREA:
            continue
        result[name] = smoothstep(t / EAR_RISE) * (1.0 - smoothstep((d - EAR_RADIUS) / EAR_FADE))
    return result


def hair_weights(points, tris, rig, *, hang_start=0.025, hang_length=0.07, neck_weight=0.3,
                 chain_stretch=0.5, joint_blend=0.30, ears=True, weld=None):
    """Bone weights of hair ``points`` (world space, rest pose, (n, 3)) with triangles ``tris``: (bone
    names, (n, len(names)) weights adding up to 1 per vertex).

    ``weld`` (vertex -> welded group) joins vertices sharing a position into loose parts; by default
    they are found here."""
    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    model = rig.model(points)
    count = len(points)
    names = [bone.name for bone in rig.bones]
    column = {name: i for i, name in enumerate(names)}
    weights = np.zeros((count, len(names)))
    distance = rig.head_distance(points)
    hanging = hanging_share(distance, hang_start, hang_length)

    hair = rig.hair_bones()
    chains = chains_of(hair)
    if chains:
        down = np.zeros((count, len(chains)))
        for i, chain in enumerate(chains):
            down[:, i] = chain.project(model)[1]
        share = chain_shares(model, rig, chains)
        # Loose parts take their vertices' choice as a whole when they mostly agree.
        if weld is None:
            weld, weld_count = fitting.weld_groups(points, fitting.WELD_DISTANCE)
        else:
            weld_count = int(weld.max()) + 1 if len(weld) else 0
        if len(tris):
            wt = weld[tris]
            edges = np.vstack([wt[:, [0, 1]], wt[:, [1, 2]]])
            part = components(weld_count, edges)[weld]
            parts = int(part.max()) + 1
            mass = vertex_areas(points, tris) * np.maximum(hanging, 1e-3)
            pooled = np.zeros((parts, len(chains)))
            np.add.at(pooled, part, share * mass[:, None])
            pooled /= np.maximum(pooled.sum(1, keepdims=True), 1e-12)
            agree = smoothstep((pooled.max(1) - PART_AGREEMENT) / PART_AGREEMENT_RANGE)[part]
            share = (1.0 - agree)[:, None] * share + agree[:, None] * pooled[part]
        nearest = share.argmax(1)
        for i, chain in enumerate(chains):
            joints = chain.joints.copy()
            widths = np.full(len(chain.bones), joint_blend)
            if len(chain.bones) > 1:
                mine = nearest == i
                w = hanging[mine]
                length = chain.joints[-1]
                if mine.sum() > 10 and w.sum() > 1e-6:
                    order = np.argsort(down[mine, i])
                    cumulative = np.cumsum(w[order]) / w.sum()
                    length = down[mine, i][order][min(np.searchsorted(cumulative, HAIR_LENGTH_SHARE), len(order) - 1)]
                even = length * np.arange(len(chain.bones)) / len(chain.bones)
                joints = chain.joints + chain_stretch * (np.maximum(chain.joints, even) - chain.joints)
                widths[1:] = np.maximum(joint_blend, JOINT_SPREAD * np.diff(joints))
            shares = segment_shares(down[:, i], joints, widths)
            for k, bone in enumerate(chain.bones):
                weights[:, column[bone.name]] += hanging * share[:, i] * shares[:, k]
    if ears:
        for name, ear in ear_shares(model, tris, rig).items():
            weights *= (1.0 - ear)[:, None]
            weights[:, column[name]] += ear
    head, neck, back = neck_split(points, rig, neck_weight)
    rest = np.clip(1.0 - weights.sum(1), 0.0, 1.0)
    weights[:, column[HEAD]] += rest * head
    weights[:, column[NECK]] += rest * neck
    weights[:, column[UPPER_BACK]] += rest * back
    return names, weights


# -----------------------------------------------------------------------------
# Picking a skeleton

def _sample(points, tris, count, seed=0):
    """At most ``count`` triangles of the hair, picked at random by area, and their vertices."""
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if len(tris) > count:
        area = np.linalg.norm(np.cross(points[tris[:, 1]] - points[tris[:, 0]],
                                       points[tris[:, 2]] - points[tris[:, 0]]), axis=1)
        total = area.sum()
        rng = np.random.default_rng(seed)
        p = area / total if total > 0 else None
        tris = tris[rng.choice(len(tris), count, replace=False, p=p)]
    used = np.zeros(len(points), dtype=bool)
    used[tris.ravel()] = True
    remap = np.cumsum(used) - 1
    return points[used], remap[tris]


def _segments(bones, reach):
    """(head, axis, length) of each bone: to its farthest child along its axis, else ``reach``."""
    result = []
    for bone in bones:
        length = max([float(np.dot(c.head - bone.head, bone.x)) for c in bones if c.parent == bone.name],
                     default=0.0)
        result.append((bone.head, bone.x, length if length > 1e-4 else reach))
    return result


def _kd_tree(points):
    tree = KDTree(len(points))
    for index, point in enumerate(points):
        tree.insert(Vector(point), index)
    tree.balance()
    return tree


def _rotation(axis, angle):
    x, y, z = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(((c + x * x * t, x * y * t - z * s, x * z * t + y * s),
                     (y * x * t + z * s, c + y * y * t, y * z * t - x * s),
                     (z * x * t - y * s, z * y * t + x * s, c + z * z * t)))


def _part_way(rotation, keep):
    """``keep`` of the turn ``rotation`` makes (about the same axis)."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.eye(3)
    axis = np.array((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]))
    norm = np.linalg.norm(axis)
    if norm < 1e-9:  # half a turn: any axis the rotation keeps
        w, v = np.linalg.eigh(rotation + rotation.T)
        axis = v[:, np.argmax(w)]
        norm = 1.0
    return _rotation(axis / norm, angle * keep)


def pose(rig, axis, degrees):
    """Bone matrices ({name: 4x4}, rig model space) with the head tilted or turned by ``degrees`` about
    ``axis``: the neck turns a third of it, the head the rest, and the hair bones lag (POSE_LAG)."""
    hair = {bone.name for bone in rig.hair_bones()}
    turns = {}
    matrices = {}
    angle = math.radians(degrees)
    for bone in rig.bones:
        parent = bone.parent if bone.parent in matrices else None
        parent_turn = turns[parent] if parent else np.eye(3)
        parent_matrix = matrices[parent] if parent else np.eye(4)
        if bone.name == NECK:
            turn = parent_turn @ _rotation(axis, angle / 3.0)
        elif bone.name == HEAD:
            turn = parent_turn @ _rotation(axis, angle * 2.0 / 3.0)
        elif bone.name in hair:
            turn = _part_way(parent_turn, 1.0 - POSE_LAG)
        else:
            turn = parent_turn
        pivot = parent_matrix[:3, :3] @ bone.head + parent_matrix[:3, 3]
        matrix = np.eye(4)
        matrix[:3, :3] = turn
        matrix[:3, 3] = pivot - turn @ bone.head
        turns[bone.name] = turn
        matrices[bone.name] = matrix
    return matrices


def skin(points, names, weights, matrices):
    """``points`` deformed by bone ``weights`` (columns ``names``) with bone ``matrices``."""
    result = np.zeros_like(points)
    for column, name in enumerate(names):
        w = weights[:, column]
        if name not in matrices or not w.any():
            continue
        m = matrices[name]
        result += w[:, None] * (points @ m[:3, :3].T + m[:3, 3])
    return result


class _Hair:
    """What the score needs of a hair, measured once for every skeleton."""

    def __init__(self, points, tris, reference, race, offset):
        points = np.asarray(points, dtype=np.float64) - offset
        self.points, self.tris = _sample(points, tris, SCORE_TRIANGLES)
        self.area = vertex_areas(self.points, self.tris)
        head = reference.head(race)
        self.hanging = hanging_share(head.distance(self.points), SCORE_HANG_START, SCORE_HANG_LENGTH)
        self.tree = _kd_tree(self.points) if len(self.points) else None
        free = self.points[self.hanging > 0.5]
        self.free_tree = _kd_tree(free) if len(free) else None
        base = {bone.name: bone for bone in reference.body_bones(race)}
        self.head_z = base[HEAD].head[2]
        self.head_y = base[HEAD].head[1]
        below = self.head_z - self.points[:, 2]
        hanging = self.hanging > 0.5
        self.hair_length = float(np.percentile(below[hanging], 95)) if hanging.sum() > 10 else 0.0
        # How far behind the head the hanging hair reaches (95 % of it by area).
        mass = self.area * self.hanging
        behind = self.points[:, 1] - self.head_y
        order = np.argsort(behind)
        cumulative = np.cumsum(mass[order]) / max(mass.sum(), 1e-12)
        self.hair_back = 0.0
        if len(order):
            self.hair_back = float(behind[order][min(np.searchsorted(cumulative, 0.95), len(order) - 1)])
        self.reference = reference
        self.race = race
        self._rigid = None
        edges = np.vstack([self.tris[:, [0, 1]], self.tris[:, [1, 2]], self.tris[:, [2, 0]]])
        self.edges = np.unique(np.sort(edges, axis=1), axis=0)
        self.lengths = np.linalg.norm(self.points[self.edges[:, 0]] - self.points[self.edges[:, 1]], axis=1)

    def rigid(self):
        """The hair posed in POSES with only the head and neck: what hanging hair lags behind."""
        if self._rigid is None:
            rig = self.reference.rig(self.race, 0)
            head, neck, back = neck_split(self.points, rig, POSE_NECK_WEIGHT)
            weights = np.stack([head, neck, back], axis=1)
            names = [HEAD, NECK, UPPER_BACK]
            self._rigid = [skin(self.points, names, weights, pose(rig, axis, degrees)) for axis, degrees in POSES]
        return self._rigid


def _deformation(hair, rig):
    """(stretch, lag) of the hair with ``rig``'s hair weights in POSES: how much its edges stretch, and
    how far its hanging part lags behind the head."""
    names, weights = hair_weights(hair.points, hair.tris, rig, joint_blend=SCORE_JOINT_BLEND)
    mass = hair.area * hair.hanging
    total = max(mass.sum(), 1e-12)
    stretch = []
    lag = []
    for (axis, degrees), rigid in zip(POSES, hair.rigid()):
        posed = skin(hair.points, names, weights, pose(rig, axis, degrees))
        lengths = np.linalg.norm(posed[hair.edges[:, 0]] - posed[hair.edges[:, 1]], axis=1)
        stretch.append(np.mean(np.abs(lengths - hair.lengths) / np.maximum(hair.lengths, 1e-6))
                       if len(lengths) else 0.0)
        lag.append((mass * np.linalg.norm(posed - rigid, axis=1)).sum() / total)
    return float(np.mean(stretch)), float(np.mean(lag))


def _features(hair, rig):
    """The score's features (SCORE_FEATURES) of ``rig``'s skeleton for ``hair``."""
    bones = rig.hair_bones()
    count = len(hair.points)
    f = {}
    wA = hair.area * hair.hanging
    total = max(wA.sum(), 1e-12)
    body = np.array([bone.name in BODY_HAIR for bone in bones])
    own = np.flatnonzero(~body)
    segments = _segments(bones, LEAF_REACH)
    dist = np.zeros((count, len(bones)))
    for i, (head, axis, length) in enumerate(segments):
        v = hair.points - head
        t = np.clip(v @ axis, 0.0, length)
        dist[:, i] = np.linalg.norm(v - t[:, None] * axis[None], axis=1)
    if len(own):
        dist = dist + SCORE_BODY_HAIR_HANDICAP * body[None, :]
    closest = dist.min(1)
    nearest = dist.argmin(1)
    f["cov_mse"] = (wA * np.minimum(closest, 0.1) ** 2).sum() / total
    f["uncovered5"] = (wA * (closest > 0.05)).sum() / total
    f["uncovered10"] = (wA * (closest > 0.10)).sum() / total
    f["n_bones"] = len(own)
    f["log_bones"] = math.log1p(len(own))
    f["is_none"] = float(not len(own))
    f["mean_assigned_dist"] = (wA * closest).sum() / total
    soft = np.exp(-((dist - closest[:, None]) / 0.05) ** 2)
    soft /= soft.sum(1, keepdims=True)
    mass = (wA[:, None] * soft).sum(0) / total
    f["kami_mass"] = float(mass[body].sum()) if body.any() else 0.0
    if len(own):
        moved = mass[own]
        f["useless2"] = float((moved < 0.02).mean())
        f["useless5"] = float((moved < 0.05).mean())
        p = moved / max(moved.sum(), 1e-12)
        f["entropy"] = (float(-(p * np.log(np.maximum(p, 1e-12))).sum() / max(math.log(len(own)), 1e-9))
                        if len(own) > 1 else 0.0)
        near_hair, near_free = [], []
        for i in own:
            head, axis, length = segments[i]
            samples = head[None] + np.linspace(0.0, length, 6)[:, None] * axis[None]
            near_hair.append(np.mean([min(hair.tree.find(Vector(s))[2], 0.2) for s in samples])
                             if hair.tree is not None else 0.2)
            near_free.append(np.mean([min(hair.free_tree.find(Vector(s))[2], 0.2) for s in samples])
                             if hair.free_tree is not None else 0.2)
        f["bone_to_hair"] = float(np.mean(near_hair))
        f["bone_to_free"] = float(np.mean(near_free))
    else:
        f["useless2"] = f["useless5"] = f["entropy"] = f["bone_to_hair"] = f["bone_to_free"] = 0.0
    misaligned = upstream = overhang = 0.0
    for i, bone in enumerate(bones):
        mine = nearest == i
        if not mine.any():
            continue
        head, axis, length = segments[i]
        v = hair.points[mine] - head
        dn = np.linalg.norm(v, axis=1)
        t = v @ axis
        w = wA[mine]
        misaligned += (w * (1.0 - t / np.maximum(dn, 1e-9))).sum()
        upstream += (w * (t < -0.02)).sum()
        if not any(other.parent == bone.name for other in bones):
            overhang += (w * np.maximum(0.0, t - length)).sum()
    f["misalign"] = misaligned / total
    f["upstream"] = upstream / total
    f["overhang"] = overhang / total
    # How far down the hair skeleton reaches, and its chains, against the hair's length.
    skeleton = [bone for bone in bones if bone.name not in BODY_HAIR]
    if skeleton:
        reach = max(hair.head_z - (bone.head[2] + min(0.0, LEAF_REACH * bone.x[2])) for bone in skeleton)
        chains = chains_of(skeleton)
        depth = max(len(chain.bones) for chain in chains)
        back = max([len(chain.bones) for chain in chains if chain.heads[:, 1].mean() > hair.head_y] or [0])
        paths = len(chains)
    else:
        reach = depth = back = paths = 0
    f["hair_len"] = hair.hair_length
    f["skel_reach"] = reach
    f["chain_depth"] = depth
    f["reach_gap"] = hair.hair_length - reach
    f["depth_x_len"] = hair.hair_length * depth
    f["back_depth"] = back
    f["n_paths"] = paths
    # How much farther behind the head the hair reaches than the skeleton's bones: long chains far back
    # (h0170's) suit full hair, closer ones (h0160's) slimmer hair.
    ends = [p for (head, axis, length) in _segments(skeleton, LEAF_REACH) for p in (head, head + axis * length)]
    f["back_gap"] = hair.hair_back - max(p[1] - hair.head_y for p in ends) if ends else 0.0
    f["back_gap_abs"] = abs(f["back_gap"])
    return np.array([f[name] for name in SCORE_FEATURES], dtype=np.float64)


def hanging_fraction(points, tris, race, *, offset=None, hang_start=0.025, hang_length=0.07, reference=None):
    """The share of the hair (by area) that hangs from the hair bones with these settings."""
    reference = reference or Reference.get()
    offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    area = vertex_areas(points, np.asarray(tris, dtype=np.int64).reshape(-1, 3))
    if area.sum() <= 0.0:
        return 0.0
    hanging = hanging_share(reference.head(race).distance(points - offset), hang_start, hang_length)
    return float((area * hanging).sum() / area.sum())


def rank_skeletons(points, tris, race, *, offset=None, hang_start=0.025, hang_length=0.07, reference=None):
    """Every hair skeleton of ``race`` (and 0 for none) scored for hair ``points``/``tris`` (world space,
    rest pose): a list of (number, score), best first. Hair that hardly hangs (MIN_HANGING, with
    ``hang_start`` and ``hang_length``) gets none first: nothing of it would swing."""
    reference = reference or Reference.get()
    offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        raise HairError("The hair has no vertices")
    hair = _Hair(points, tris, reference, race, offset)
    scored = []
    for number in [0] + reference.skeletons(race):
        rig = reference.rig(race, number)
        features = _features(hair, rig)
        scored.append((float(((features - SCORE_MEAN) / SCORE_STD) @ SCORE_WEIGHTS), number, rig, features))
    scored.sort(key=lambda item: -item[0])
    shortlist = []
    for _score, number, rig, features in scored[:SHORTLIST]:
        full = np.concatenate([features[:15], _deformation(hair, rig), features[15:]])
        shortlist.append((number, float(((full - FULL_MEAN) / FULL_STD) @ FULL_WEIGHTS)))
    shortlist.sort(key=lambda item: -item[1])
    result = shortlist + [(number, score) for score, number, _rig, _features_ in scored[SHORTLIST:]]
    fraction = hanging_fraction(points, tris, race, offset=offset, hang_start=hang_start,
                                hang_length=hang_length, reference=reference)
    if fraction < MIN_HANGING:
        result.sort(key=lambda item: item[0] != 0)
    return result


# -----------------------------------------------------------------------------
# Armature

def weighted_armature(obj):
    """The armature deforming ``obj`` (see `straighten.deforming_armature`), or None. While Customize+ is
    on, that's the armature whose hidden rig deforms it: bones go on the armature, not on the rig."""
    modifier = straighten.deforming_armature(obj)
    if modifier is None:
        return None
    from .cplus import rig as cplus_rig
    armature = modifier.object
    return armature.parent if cplus_rig.is_rig(armature) and armature.parent is not None else armature


def armature_offset(armature_object, rig):
    """How far the armature's head bone is from the race's (world space), or zeros."""
    if armature_object is None:
        return np.zeros(3)
    bone = armature_object.data.bones.get(HEAD)
    if bone is None:
        return np.zeros(3)
    head = np.array(armature_object.matrix_world @ bone.head_local, dtype=np.float64)
    return head - rig.by_name[HEAD].head


def needed_bones(rig, names):
    """``names`` and every ancestor of them in ``rig``, parents first."""
    wanted = []
    seen = set()

    def add(name):
        if name in seen or name not in rig.by_name:
            return
        parent = rig.by_name[name].parent
        if parent is not None:
            add(parent)
        seen.add(name)
        wanted.append(name)

    for name in names:
        add(name)
    return wanted


def ensure_bones(context, armature_object, rig, names):
    """Add the bones ``names`` (and their ancestors) of ``rig`` that ``armature_object`` lacks, in the
    game's rest pose (moved by the rig's offset). Returns the names added."""
    existing = set(armature_object.data.bones.keys())
    missing = [name for name in needed_bones(rig, names) if name not in existing]
    if not missing:
        return []
    import bpy
    previous_active = context.view_layer.objects.active
    previous_mode = previous_active.mode if previous_active is not None else 'OBJECT'
    hidden = None
    try:
        if previous_active is not None and previous_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        hidden = armature_object.hide_get()
        armature_object.hide_set(False)
        # This raises when the armature isn't in the view layer (its collection is excluded).
        context.view_layer.objects.active = armature_object
        to_armature = np.array(armature_object.matrix_world.inverted_safe(), dtype=np.float64)
        bpy.ops.object.mode_set(mode='EDIT')
        edit_bones = armature_object.data.edit_bones
        for name in missing:
            bone = rig.by_name[name]
            rest = bone.rest.copy()
            rest[:3, 3] += rig.offset
            local = to_armature @ rest
            # Rest bones hold no scale.
            rotation = local[:3, :3]
            rotation = rotation / np.maximum(np.linalg.norm(rotation, axis=0), 1e-12)[None, :]
            local[:3, :3] = rotation
            edit = edit_bones.new(name)
            edit.head = (0.0, 0.0, 0.0)
            edit.tail = (0.0, getattr(rig, "bone_length", BONE_LENGTH), 0.0)
            edit.matrix = Matrix(local.tolist())
            edit.use_deform = True
            if bone.parent is not None and bone.parent in edit_bones:
                edit.parent = edit_bones[bone.parent]
    finally:
        if armature_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if hidden is not None:
            armature_object.hide_set(hidden)
        context.view_layer.objects.active = previous_active
        if previous_active is not None and previous_mode != 'OBJECT' and previous_active.type == 'MESH':
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except RuntimeError:
                pass
    # A Customize+ rig is a copy of its armature: while it's on, it's made again with the new bones.
    from .cplus import rig as cplus_rig
    if cplus_rig.is_on(armature_object):
        cplus_rig.connect(armature_object, cplus_rig.ensure(context, armature_object))
    return missing


def new_armature(context, name, collection):
    """An empty armature object at the world origin, in ``collection``."""
    import bpy
    data = bpy.data.armatures.new(name)
    armature_object = bpy.data.objects.new(name, data)
    collection.objects.link(armature_object)
    return armature_object


def add_armature_modifier(obj, armature_object):
    modifier = obj.modifiers.new(name="Armature", type='ARMATURE')
    modifier.object = armature_object
    modifier.use_vertex_groups = True
    return modifier


# -----------------------------------------------------------------------------
# Stroke

class HairStroke(goals.GoalStroke):
    """Blends a mesh's weights toward its hair weights, dab by dab.

    ``rest_co`` and ``tris`` are the mesh in the rest pose (world space) and ``rig`` the `HairRig` of the
    hair's race and skeleton, moved to its armature. See `goals.GoalStroke` for how the weights change
    (``replace``: the hair weights replace all the old ones, as the Hair Weights button does).
    """

    def __init__(self, obj, rest_co, tris, rig, *, hang_start=0.025, hang_length=0.07, neck_weight=0.3,
                 chain_stretch=0.5, joint_blend=0.30, ears=True, max_groups=DEFAULT_MAX_GROUPS, group_mode='ALL',
                 replace=False):
        super().__init__(obj, rest_co, max_groups=max_groups, group_mode=group_mode, replace=replace)
        self.rig = rig
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        welded_tris = self.weld[tris] if len(tris) else tris
        names, weights = hair_weights(
            self.rest, welded_tris, rig, hang_start=hang_start, hang_length=hang_length,
            neck_weight=neck_weight, chain_stretch=chain_stretch, joint_blend=joint_blend, ears=ears,
            weld=np.arange(self.weld_count),
        )
        self.goal_names = names
        self.goal_weights = weights
        self.hair_names = {bone.name for bone in rig.bones}
        self.column_group = [self.name_to_group.get(name, -1) for name in names]

    def _goal(self, vert, group, param, shares):
        row = self.goal_weights[group]
        result = {}
        for column in np.flatnonzero(row > SHARE_EPSILON).tolist():
            key = self.column_group[column]
            if key < 0:
                key = self._key(self.goal_names[column])
                if key is None:
                    continue
            elif key in self.locked:
                continue
            result[key] = result.get(key, 0.0) + float(row[column])
        total = sum(result.values())
        if total <= 1e-12:
            return None
        return [(1.0, {key: share / total for key, share in result.items()})]

    def _group_created(self, group, name):
        if name in self.goal_names:
            self.column_group[self.goal_names.index(name)] = group

    def _group_forgotten(self, group, name):
        self.column_group = [-1 if index == group else index for index in self.column_group]


# -----------------------------------------------------------------------------
# Setting up

def stroke_params(settings):
    """`HairStroke` keyword arguments from the brush settings (`Scene.magic_fit`)."""
    return dict(
        hang_start=settings.hair_hang_start,
        hang_length=settings.hair_hang_length,
        neck_weight=settings.hair_neck_weight,
        chain_stretch=settings.hair_chain_stretch,
        joint_blend=settings.hair_joint_blend,
        ears=settings.hair_ears,
        max_groups=settings.hair_max_groups,
        group_mode=settings.group_mode,
    )


# The custom properties Hair Weights puts on each hair mesh with a hair skeleton, for exporters (such as
# Instant Edit) to set the hair's EST entry in the mod: the entry (the skeleton's number) and the race
# (c0801) whose skeleton it is.
EST_PROPERTY = "xiv_est_hair"
EST_RACE_PROPERTY = "xiv_est_race"


def tag_est(obj, race, number):
    """Put the EST entry the hair skeleton ``number`` of ``race`` needs on ``obj``, or take it off when
    the hair has no hair skeleton (0: the game's EST entry for the hair can stay as it is)."""
    if number:
        obj[EST_PROPERTY] = int(number)
        obj.id_properties_ui(EST_PROPERTY).update(
            description="The hair skeleton (EST entry) this hair needs. Set by Magic Fit's Hair Weights",
            min=1, max=9999)
        obj[EST_RACE_PROPERTY] = race
        obj.id_properties_ui(EST_RACE_PROPERTY).update(
            description="The race the hair skeleton belongs to. Set by Magic Fit's Hair Weights")
        return
    for key in (EST_PROPERTY, EST_RACE_PROPERTY):
        if key in obj:
            del obj[key]


def object_skeleton(obj, settings):
    """(race, skeleton number) the hair ``obj`` was given, else the settings' race and picked skeleton
    (None when the settings ask for the best match and the hair has none yet)."""
    hair = getattr(obj, "magic_fit_hair", None)
    if hair is not None and hair.race in RACE_NAMES and hair.skeleton >= 0:
        return hair.race, hair.skeleton
    if settings.hair_skeleton_mode == 'PICK':
        return settings.hair_race, settings.hair_skeleton
    return settings.hair_race, None


def build_stroke(context, obj, settings):
    """A `HairStroke` for the brush on ``obj``, which must have an armature with its hair bones (the Hair
    Weights button gives it one). Raises `HairError` when it can't be set up."""
    modifier = straighten.deforming_armature(obj)
    if modifier is None:
        raise HairError("Hair needs an Armature modifier: give it Hair Weights first")
    race, number = object_skeleton(obj, settings)
    if number is None:
        raise HairError("Give the hair Hair Weights first, or pick its skeleton")
    reference = Reference.get()
    if number and not reference.has_skeleton(race, number):
        raise HairError("{:s} has no hair skeleton {:s}".format(RACE_NAMES[race], skeleton_label(number)))
    rig = reference.rig(race, number)
    rig = rig.moved(armature_offset(modifier.object, rig))
    missing = [bone.name for bone in rig.hair_bones() if modifier.object.data.bones.get(bone.name) is None]
    if missing:
        raise HairError("Armature '{:s}' lacks the hair bones of {:s}: give the hair Hair Weights first".format(
            modifier.object.name, skeleton_label(number)))
    rest, = straighten.rest_geometry(context, (obj,), ('INDEXED',))
    return HairStroke(obj, rest.co, rest.tris, rig, **stroke_params(settings))
