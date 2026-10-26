# SPDX-License-Identifier: GPL-3.0-or-later

"""Earring weights: dangly earrings that swing with the earring bones of the Earring Bones mod.

The game's own earring points (n_ear_a/b_l/r) are body bones; on Miqo'te and Viera they sit at the cat or
rabbit ear, so earrings weighted to them pivot from there. The Earring Bones mod (made with Instant Edit's
EarringBones tool) gives those faces ten face bones at the human earlobe:

- j_ex_earring_a_l/r, parented to the head (j_kao), at the earlobe with their +X pointing straight down;
- j_ex_earring_b, c, d and e, each STEPS below the one before (a hook, then 1.5 cm parts), chained
  a > b > c > d > e.

The game's physics swings a from the lobe, and each joint below lets the earring bend once more. Only the
bones the mod adds exist in game, and only with it on.

Earring Weights adds the bones to the earring's armature (from the game's head bone and the lobe below),
then splits each vertex left or right by which side of the head it's on, and shares each earring out down
the chain by how far below the lobe it hangs, over the earring's own length (its lowest vertex):

- An earring made of pieces (a hook, rings, beads, a pendant: loose parts lying one below the other) bends
  only where they meet: each piece follows one bone whole, and the bones change at the joints closest to
  where an even split of the length would put them. Parts that mostly share their height (a gem set in a
  pendant, twin drops side by side) are one piece.
- An earring in one piece is split evenly: the bones take a share of the length each, all of it around the
  middle of their share and blending linearly into the next across it. Everything at or above the lobe
  (the hook or stud) follows a.

Things worn elsewhere on the ear (studs higher up, chains hung from those piercings: loose parts touching
each other, none within PIN of the lobe or the line down from it) would leave the ear swung from the lobe.
Where they're on the ear (the stud at the top, both ends of a chain hung between two piercings: pieces
nothing touching them reaches above) is pinned to the head instead. Below it the head's share fades out
down the hanging's own length into the bones, spread evenly over it: by height alone, so links stay together
where they touch, even though the bones turn about the lobe rather than there.

The bones themselves stay where the mod has them (STEPS apart); only the weights spread over the length,
as long hair's are spread over chains shorter than it (see `hairing`).

The brush paints toward those weights, or toward one bone picked by hand (the vertex's side of it).

Like `painting`, nothing here depends on the 3D view.
"""

import numpy as np

from . import fitting, hairing, smoothing, straighten
from .goals import DEFAULT_MAX_GROUPS

HEAD = hairing.HEAD
# Where the earrings hang from, per race and face skeleton: the right earlobe in the head bone's rest space
# (metres, the game's axes: +Z is the character's left, so the left lobe is the right one with Z negated).
# Measured on the game's faces with Instant Edit's EarringBones tool (lobes.json), Miqo'te and Viera checked
# in game 2026-10-20. One lobe serves every face skeleton the mod has bones for: Viera faces 1 to 4 (f0002 to
# f0005, measured on face 1), Au Ra faces 1 to 4 and 101 to 104 (f0002 to f0005, measured on face 101).
LOBES = {
    ("c0801", "f0002"): ("Miqo'te Female", (0.001795, 0.008066, -0.057146)),
    ("c1801", "f0002"): ("Viera Female", (-0.013044, 0.020468, -0.057924)),
    ("c1401", "f0002"): ("Au Ra Female", (-0.005998, -0.000636, -0.058272)),
}
# Straight down, in the head bone's rest space, for every race above.
DOWN = np.array((0.0, 1.0, 0.0))
# The earring bones of each side, top down, and how far each of b to e sits below the one before (Instant
# Edit's EarringBones lobes.json "chain", rebuilt 2026-10-23).
LETTERS = ("a", "b", "c", "d", "e")
MAX_BONES = len(LETTERS)
STEPS = (0.007, 0.015, 0.015, 0.015)
SIDES = ("l", "r")
# Vertices farther than this from both lobes (sideways, across the line down from each) aren't where an
# earring hangs: the mesh is likely not lined up on this race's head.
FAR = 0.03
# Loose parts closer than this to each other hang together (the links of a chain, a stud and its ring).
TOUCH = 0.001
# Hangings that come no closer than this to the lobe, or to the line down from it, are worn elsewhere on the
# ear (studs, cuffs, chains from a piercing higher up): their top piece is pinned to the head; what hangs from
# the lobe passes within a couple of millimetres of it.
PIN = 0.004
# Bones placed at a Blender armature are only this long, as far apart as the closest joints.
BONE_LENGTH = min(STEPS)
# Loose parts whose heights overlap by more than this share of the shorter one's are one piece: a gem set in
# a pendant overlaps it fully, links of a chain only where they hook into each other.
PIECE_OVERLAP = 0.5
# Earrings shorter than this below the lobe (studs) all follow a.
MIN_LENGTH = 1e-4


class EarringError(Exception):
    """Why earring weights can't be worked out; the message is meant for the user."""


def bone_name(letter, side):
    return "j_ex_earring_{:s}_{:s}".format(letter, side)


def bone_names(count=MAX_BONES):
    """The earring bones ``count`` per side uses: the left side's top down, then the right side's."""
    return [bone_name(letter, side) for side in SIDES for letter in LETTERS[:count]]


ALL_BONES = frozenset(bone_names())


def race_key(race, skeleton):
    """The enum identifier of a race and face skeleton ('c0801_f0002')."""
    return "{:s}_{:s}".format(race, skeleton)


def race_of(key):
    """(race, face skeleton) of an enum identifier made by `race_key`."""
    race, _sep, skeleton = key.partition("_")
    return race, skeleton


def race_items():
    """Enum items of the races with earring bones (identifier, name, description), from LOBES."""
    return [(race_key(race, skeleton), label, "Earrings for {:s} ({:s})".format(label, race))
            for (race, skeleton), (label, _lobe) in LOBES.items()]


def race_label(key):
    entry = LOBES.get(race_of(key))
    return entry[0] if entry is not None else key


def lobes(key):
    """The left and right lobes ((2, 3), SIDES order) of ``key``, in the head bone's rest space."""
    entry = LOBES.get(race_of(key))
    if entry is None:
        raise EarringError("No earring bones for {:s}".format(key))
    right = np.array(entry[1], dtype=np.float64)
    left = right * (1.0, 1.0, -1.0)
    return np.array((left, right))


# -----------------------------------------------------------------------------
# Rig

def _turn_down():
    """From the head bone's axes to an earring bone's: a quarter turn about Z, which puts +X along DOWN
    (the game's earring bones rest at that rotation, (0, 0, 0.707, 0.707), under the head)."""
    turn = np.eye(4)
    turn[:3, :3] = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    return turn


def _translation(vector):
    matrix = np.eye(4)
    matrix[:3, 3] = vector
    return matrix


class EarringRig:
    """The head bone and the earring bones of one race (`LOBES` key), in Blender model space (moved by
    ``offset`` when the earring's armature has its head elsewhere), as `hairing.ensure_bones` takes them."""

    bone_length = BONE_LENGTH

    def __init__(self, key, head, offset=None):
        self.key = key
        self.offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
        self.head = head
        self.lobes = lobes(key)
        self.bones = [head]
        for side, lobe in zip(SIDES, self.lobes):
            rest = head.rest @ _translation(lobe) @ _turn_down()
            parent = head.name
            for letter, step in zip(LETTERS, (0.0,) + STEPS):
                rest = rest @ _translation((step, 0.0, 0.0))
                name = bone_name(letter, side)
                self.bones.append(hairing.Bone(name, parent, rest))
                parent = name
        self.by_name = {bone.name: bone for bone in self.bones}

    @classmethod
    def for_race(cls, key, offset=None, reference=None):
        """The rig of ``key``, its head bone at the game's rest for the race (`hairing.Reference`)."""
        race, _skeleton = race_of(key)
        reference = reference or hairing.Reference.get()
        head = next(bone for bone in reference.body_bones(race) if bone.name == HEAD)
        return cls(key, head, offset)

    def moved(self, offset):
        return EarringRig(self.key, self.head, offset)

    def local(self, points):
        """World space ``points`` in the head bone's rest space (the game's axes, as the lobes are)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.offset
        rest = self.head.rest
        return (points - rest[:3, 3]) @ rest[:3, :3]


# -----------------------------------------------------------------------------
# Weights

class Split:
    """Earring weights of some vertices: ``names`` (`bone_names` of ``count``, then HEAD), ``weights`` (a row
    per vertex, each adding up to 1), ``left`` (which vertices went to the left side), ``far`` (which lie
    farther than FAR from both lobes, sideways), ``pinned`` (which don't hang from the lobe and follow HEAD
    alone), and per side (SIDES order) the earring's ``lengths`` below the lobe and the ``joints`` its bones
    change at (0 for an earring split evenly)."""

    def __init__(self, names, weights, left, far, pinned, lengths, joints):
        self.names = names
        self.weights = weights
        self.left = left
        self.far = far
        self.pinned = pinned
        self.lengths = lengths
        self.joints = joints


def loose_parts(count, tris, weld=None):
    """The loose part of each of ``count`` vertices joined by triangles ``tris`` (numbered from 0); vertices
    of ``weld`` (welded group per vertex, see `fitting.weld_groups`) count as one."""
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if weld is None:
        weld = np.arange(count)
    weld = np.asarray(weld, dtype=np.int64)
    welded = weld[tris] if len(tris) else tris
    edges = np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]]]) if len(welded) else np.zeros((0, 2), np.int64)
    return hairing.components(int(weld.max()) + 1 if len(weld) else 0, edges)[weld]


def pieces(heights, parts):
    """The pieces an earring is made of: lists of its loose ``parts`` (label per vertex), top down, those
    whose ``heights`` (below the lobe) overlap by more than PIECE_OVERLAP joined."""
    labels = np.unique(parts)
    top = np.array([heights[parts == label].min() for label in labels])
    bottom = np.array([heights[parts == label].max() for label in labels])
    group = list(range(len(labels)))

    def find(k):
        while group[k] != k:
            group[k] = group[group[k]]
            k = group[k]
        return k

    span = np.maximum(bottom - top, MIN_LENGTH)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            overlap = min(bottom[i], bottom[j]) - max(top[i], top[j])
            if overlap > PIECE_OVERLAP * min(span[i], span[j]):
                group[find(i)] = find(j)
    merged = {}
    for k in range(len(labels)):
        merged.setdefault(find(k), []).append(k)
    found = [(float(top[members].min()), float(bottom[members].max()), [labels[k] for k in members])
             for members in merged.values()]
    found.sort(key=lambda piece: piece[0] + piece[1])
    return found


def pick_joints(cuts, targets):
    """Which of the joint heights ``cuts`` (ascending) to change bones at: for each of ``targets`` (ascending)
    the closest one below the last picked, leaving enough for the targets after it. Fewer when there are fewer
    joints than targets."""
    if len(cuts) <= len(targets):
        return list(range(len(cuts)))
    picked = []
    start = 0
    for k, target in enumerate(targets):
        stop = len(cuts) - (len(targets) - k - 1)
        best = min(range(start, stop), key=lambda index: abs(cuts[index] - target))
        picked.append(best)
        start = best + 1
    return picked


def _side_weights(heights, parts, count, use_joints):
    """Bone index shares of one earring's vertices (heights below the lobe): (upper, lower, share of lower), its
    length and its number of joints used."""
    length = max(float(heights.max()), 0.0)
    upper = np.zeros(len(heights), dtype=np.int64)
    lower = upper.copy()
    share = np.zeros(len(heights))
    if count == 1 or length < MIN_LENGTH:
        return upper, lower, share, length, 0
    if use_joints and parts is not None:
        found = pieces(heights, parts)
        if len(found) > 1:
            cuts = [(found[k][1] + found[k + 1][0]) / 2.0 for k in range(len(found) - 1)]
            picked = pick_joints(cuts, [length * k / count for k in range(1, count)])
            bone_of_part = {}
            for index, (_top, _bottom, labels) in enumerate(found):
                bone = sum(1 for cut in picked if cut < index)
                bone_of_part.update((label, bone) for label in labels)
            upper = np.array([bone_of_part[label] for label in parts], dtype=np.int64)
            return upper, upper.copy(), share, length, len(picked)
    # Each bone takes length / count, all of it around its middle.
    t = np.clip(heights / (length / count) - 0.5, 0.0, count - 1)
    upper = np.minimum(np.floor(t).astype(np.int64), count - 2)
    lower = upper + 1
    return upper, lower, t - upper, length, 0


def touching(local, parts):
    """Pairs ((m, 2)) of loose ``parts`` (one per vertex ``local``, numbered from 0) that come within TOUCH
    of each other."""
    first, second = smoothing.close_pairs(local, TOUCH)
    across = parts[first] != parts[second]
    pairs = np.sort(np.column_stack((parts[first][across], parts[second][across])), axis=1)
    return np.unique(pairs, axis=0) if len(pairs) else pairs.reshape(0, 2)


def contacts(heights, parts, pairs):
    """Which loose ``parts`` (numbered from 0) are where things are on the ear: those no part touching them
    (``pairs``) reaches above (the stud at the top, both ends of a chain hung between two piercings, not a
    pendant), with the parts touching them that mostly share their height (a ring through the stud)."""
    count = int(parts.max()) + 1
    top = np.full(count, np.inf)
    bottom = np.full(count, -np.inf)
    np.minimum.at(top, parts, heights)
    np.maximum.at(bottom, parts, heights)
    first, second = pairs.T
    peak = np.ones(count, dtype=bool)
    peak[first[top[second] < top[first]]] = False
    peak[second[top[first] < top[second]]] = False
    overlap = np.minimum(bottom[first], bottom[second]) - np.maximum(top[first], top[second])
    span = np.maximum(np.minimum(bottom[first] - top[first], bottom[second] - top[second]), MIN_LENGTH)
    shared = overlap > PIECE_OVERLAP * span
    found = peak.copy()
    found[first[shared & peak[second]]] = True
    found[second[shared & peak[first]]] = True
    return found


def earring_weights(local, lobes, count=MAX_BONES, parts=None, use_joints=True):
    """The `Split` of vertices ``local`` (the head bone's rest space) with the left and right ``lobes`` and
    ``count`` (1 to MAX_BONES) bones per side; ``parts`` (loose part per vertex, see `loose_parts`; each
    vertex its own when None) to find the hangings and, for ``use_joints``, the joints.

    A hanging that reaches the lobe is the earring, its bones spread over its length below the lobe. One worn
    elsewhere on the ear has its `contacts` pinned to the head, and what hangs below them swings, the bones
    spread over its own length below the lowest."""
    count = int(min(max(count, 1), MAX_BONES))
    local = np.asarray(local, dtype=np.float64).reshape(-1, 3)
    lobes = np.asarray(lobes, dtype=np.float64)
    names = bone_names(count) + [HEAD]
    weights = np.zeros((len(local), len(names)))
    # +Z is the character's left.
    left = local[:, 2] > 0.0
    pinned = np.zeros(len(local), dtype=bool)
    lengths, joints = [0.0, 0.0], [0, 0]
    if not len(local):
        return Split(names, weights, left, pinned.copy(), pinned, lengths, joints)
    given = parts is not None
    parts = np.unique(np.asarray(parts), return_inverse=True)[1] if given else np.arange(len(local))
    pairs = touching(local, parts) if given else np.zeros((0, 2), dtype=np.int64)
    hanging = hairing.components(int(parts.max()) + 1, pairs)[parts]
    offsets = local - np.where(left[:, None], lobes[0], lobes[1])
    heights = offsets @ DOWN
    on_ear = contacts(heights, parts, pairs)[parts]
    # Above the lobe, how far from the lobe itself; below it, how far across from the line down.
    reach = np.linalg.norm(offsets - np.outer(np.maximum(heights, 0.0), DOWN), axis=1)
    closest = np.full(int(hanging.max()) + 1, np.inf)
    np.minimum.at(closest, hanging, reach)
    on_lobe = closest[hanging] <= PIN
    rows = np.arange(len(local))

    for side, mask in enumerate((left, ~left)):
        chosen = rows[mask & on_lobe]
        if len(chosen):
            upper, lower, share, lengths[side], joints[side] = _side_weights(
                heights[chosen], parts[chosen] if given else None, count, use_joints)
            np.add.at(weights, (chosen, side * count + upper), 1.0 - share)
            np.add.at(weights, (chosen, side * count + lower), share)
        for label in np.unique(hanging[mask & ~on_lobe]):
            chosen = rows[mask & (hanging == label)]
            contact = on_ear[chosen]
            pinned[chosen[contact]] = True
            # The bones turn about the lobe, not the contact: the head's share fades out down the hanging, and
            # the weights go by height alone (no joints), so the links stay together where they touch.
            below = np.maximum(heights[chosen] - heights[chosen[contact]].max(), 0.0)
            length = float(below.max())
            free = below / length if length > MIN_LENGTH else np.zeros(len(chosen))
            upper, lower, share, _length, _used = _side_weights(below, None, count, False)
            np.add.at(weights, (chosen, side * count + upper), free * (1.0 - share))
            np.add.at(weights, (chosen, side * count + lower), free * share)
            weights[chosen, -1] += 1.0 - free
    across = [np.linalg.norm((local - side) - np.outer((local - side) @ DOWN, DOWN), axis=1) for side in lobes]
    far = np.minimum(*across) > FAR
    return Split(names, weights, left, far, pinned, lengths, joints)


def one_bone(local, letter):
    """``names`` and ``weights`` that give vertices ``local`` (the head bone's rest space) all to the earring
    bone ``letter`` of their side."""
    left = np.asarray(local, dtype=np.float64).reshape(-1, 3)[:, 2] > 0.0
    names = [bone_name(letter, side) for side in SIDES]
    weights = np.zeros((len(left), 2))
    weights[left, 0] = 1.0
    weights[~left, 1] = 1.0
    return names, weights


def object_parts(rests):
    """The loose part of each vertex of meshes ``rests`` (`straighten.rest_geometry`, taken as one earring,
    vertices at the same spot welded), numbered across them."""
    points = np.concatenate([rest.co for rest in rests])
    starts = np.cumsum([0] + [len(rest.co) for rest in rests[:-1]])
    tris = np.concatenate([np.asarray(rest.tris, dtype=np.int64).reshape(-1, 3) + start
                           for rest, start in zip(rests, starts)])
    weld, _count = fitting.weld_groups(points, fitting.WELD_DISTANCE)
    return loose_parts(len(points), tris, weld)


# -----------------------------------------------------------------------------
# Brush

def stroke_params(settings):
    """`hairing.HairStroke` keyword arguments from the brush settings (`Scene.magic_fit`)."""
    return dict(max_groups=DEFAULT_MAX_GROUPS, group_mode=settings.group_mode)


def missing_bones(armature_object, count, letter='AUTO'):
    """The earring bones of ``count`` per side (or of the brush's bone ``letter``) that ``armature_object``
    lacks."""
    bones = armature_object.data.bones
    names = bone_names(count) if letter == 'AUTO' else [bone_name(letter.lower(), side) for side in SIDES]
    return [name for name in names if bones.get(name) is None]


def build_stroke(context, obj, settings):
    """A stroke that blends ``obj``'s weights toward its earring weights (with the settings' race, bones per
    side and joints), or with a bone picked for the brush, toward that bone; ``obj`` must have an armature with
    the earring bones (the Earring Weights button gives it them). Raises `EarringError` when it can't be set
    up."""
    modifier = straighten.deforming_armature(obj)
    if modifier is None:
        raise EarringError("Earrings need an Armature modifier: give them Earring Weights first")
    key = settings.earring_race
    count = settings.earring_bones
    letter = settings.earring_brush_bone
    if missing_bones(modifier.object, count, letter):
        raise EarringError("Armature '{:s}' lacks the earring bones: give the earrings Earring Weights first".format(
            modifier.object.name))
    rig = EarringRig.for_race(key)
    rig = rig.moved(hairing.armature_offset(modifier.object, rig))
    rest, = straighten.rest_geometry(context, (obj,), ('INDEXED',))
    local = rig.local(rest.co)
    if letter != 'AUTO':
        weights = one_bone(local, letter.lower())
    else:
        split = earring_weights(local, rig.lobes, count, object_parts([rest]), settings.earring_use_joints)
        weights = (split.names, split.weights)
    return hairing.HairStroke(obj, rest.co, rest.tris, rig, weights=weights, **stroke_params(settings))


