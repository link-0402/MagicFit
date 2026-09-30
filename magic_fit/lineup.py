# SPDX-License-Identifier: GPL-3.0-or-later

"""Lining up an imported model with a body.

A body or outfit ripped from another game, or made for VRChat or The Sims, stands in its own pose and
proportions: arms out in a T-pose or hanging down, longer legs, another height, sometimes facing another
way. Lining it up with the body (the devkit's Mannequin) gives refitting a place to start: the same size
and pose, with its joints on the body's joints.

Both skeletons are reduced to the same parts: the torso (pelvis, spine, chest, breasts...), the neck and
head, and per side the clavicle, upper arm, forearm, hand, thigh, shin and foot, and three segments of
each finger. Bones are sorted into parts by their names, as the common naming schemes spell them (Unity
and VRChat, Mixamo, Unreal, 3ds Max Biped, Rigify, DAZ, MMD, The Sims, FFXIV..., and The Sims' names hashed,
bone_1b82d8b2); bones whose names say nothing (jiggle, skirt, tail bones...) join their parent's part. A
finger's bones are numbered from the hand out. A part's joint is the head of its main bone, the one the next
part hangs from. A model without a usable armature is sorted by the names of its vertex groups instead, and
its joints are estimated from its weights, where one part's weight hands over to the next. When the names
say nothing at all (bone_00, bone_01...), the groups are sorted by how they overlap and where they lie on
either side of the model's midline.

Then, from the torso out:

- The torso turns so its hips and shoulders face the body's way, scales so its shoulders are as high
  above its hips as the body's are, and moves onto the body's hips. Half a body (a top without hips, a
  bottom without shoulders) stands upright like the body, is sized by its arms or legs, and moves onto the
  body's shoulders or hips.
- Each other part starts where the part above leaves its joint and turns about it, the least it takes to
  point the way the body's does: limb segments at the next joint, hands at the knuckles, each finger
  segment along the body's finger, which straightens curled fingers (or, where the body has no such
  finger, along the finger's first segment).
- The neck and head turn and move with the torso: rigs put those joints anywhere from the skull's base to
  the jaw, so matching them would stretch necks for nothing. Clavicles turn with the torso too, since its
  scale already puts the shoulders at the body's height, and rigs start them anywhere from the breastbone
  to the side of the neck. Feet keep their angle to the torso, so heels stay heels.
- With Stretch Limbs, each clavicle and limb segment then stretches lengthwise (not in girth) until its end
  lands on the body's next joint: its points slide along, easing in from its joint (`ramp`), rather than
  the segment being scaled. So the stretch changes smoothly where one segment's weight hands over to the
  next, and nothing past a segment's end stretches on. Without it, segments keep their length (scaled
  with the torso) and only turn.

Every point then moves by the blend of its parts' moves, weighted by its vertex group weights, the way an
armature bends a mesh (linear blend skinning).

Joints estimated from weights are a few centimetres off, depending on how the weights were painted. For a
model that is a body (or skin tight), `refine_joints` then corrects them against the body's surface: after
lining up, wherever the model's surface around a joint sits off the body's as a whole, the joint moves by
that much, if that makes the offset go away. That can't find a knee that is off along the leg, but it
puts hips and shoulders where the model's own shape says they are.

Like the other engines, this works on world space arrays and doesn't need the 3D view.
"""

import re

import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import resizing

TORSO, NECK, HEAD = "torso", "neck", "head"
SIDES = ("l", "r")
ARM = ("clavicle", "upperarm", "forearm", "hand")
LEG = ("thigh", "shin", "foot")
FINGERS = ("thumb", "index", "middle", "ring", "little")
# Segments per finger, from the hand out; a finger's bones past the third join the third.
DIGITS = 3


def part_name(kind, side=None):
    return kind if side is None else "{:s}_{:s}".format(kind, side)


def digit_name(finger, segment, side):
    """Part of segment ``segment`` (1 to DIGITS, from the hand out) of a finger: "index2_l"."""
    return part_name("{:s}{:d}".format(finger, segment), side)


PARTS = (TORSO, NECK, HEAD) + tuple(part_name(kind, side) for side in SIDES for kind in ARM + LEG) + tuple(
    digit_name(finger, segment, side) for side in SIDES for finger in FINGERS for segment in range(1, DIGITS + 1))
PART_INDEX = {part: index for index, part in enumerate(PARTS)}
# The part each part hangs from.
PARENT = {NECK: TORSO, HEAD: NECK}
for _side in SIDES:
    for _chain in (ARM, LEG):
        _names = (TORSO,) + tuple(part_name(kind, _side) for kind in _chain)
        PARENT.update(zip(_names[1:], _names[:-1]))
    for _finger in FINGERS:
        _names = (part_name("hand", _side),) + tuple(digit_name(_finger, k, _side) for k in range(1, DIGITS + 1))
        PARENT.update(zip(_names[1:], _names[:-1]))
# Segments reaching from their joint to the next part's; the others (hands, feet, head, fingertips) end in
# themselves.
NEXT = {NECK: HEAD}
for _side in SIDES:
    for _a, _b in (("clavicle", "upperarm"), ("upperarm", "forearm"), ("forearm", "hand"), ("thigh", "shin"),
                   ("shin", "foot")):
        NEXT[part_name(_a, _side)] = part_name(_b, _side)
    for _finger in FINGERS:
        for _k in range(1, DIGITS):
            NEXT[digit_name(_finger, _k, _side)] = digit_name(_finger, _k + 1, _side)
# The finger segment parts, and which finger each is (part -> (finger, segment, side)).
DIGIT_PARTS = {digit_name(finger, segment, side): (finger, segment, side)
               for side in SIDES for finger in FINGERS for segment in range(1, DIGITS + 1)}
# Parts that keep their angle to the torso instead of turning with the part they hang from: rigs put the
# neck and head joints in different places (the head's at the skull's base or up at the jaw), so a line
# between them says nothing about the pose. Feet keep their angle so heels stay heels.
UPRIGHT = {NECK, HEAD, "foot_l", "foot_r"}
# Parts that turn with the torso: the torso's scale already puts the shoulders at the body's height, and
# rigs start clavicles anywhere from the breastbone to the side of the neck.
SHOULDERS = {"clavicle_l", "clavicle_r"}
# Segments Stretch Limbs stretches: the others keep the model's proportions.
STRETCHED = {part_name(kind, side) for side in SIDES
             for kind in ("clavicle", "upperarm", "forearm", "thigh", "shin")}
# Parts a model needs, by name or weights, to be lined up by its bones or groups.
KEY_PARTS = ("thigh_l", "thigh_r", "upperarm_l", "upperarm_r")
# The parts of each half of a body, fingers aside: what a top or a bottom lacks by being one.
HALF_PARTS = {"upper": {NECK, HEAD} | {part_name(kind, side) for side in SIDES for kind in ARM},
              "lower": {part_name(kind, side) for side in SIDES for kind in LEG}}
# Half a body as sized, in torso lengths: a bottom reaches this far above its hips (a waistband, up to a
# high-necked gown's collar, not a head), a top at most this far below its shoulders (see `_half_fits`).
HALF_ABOVE = (0.1, 1.4)
HALF_BELOW = 1.25

# A limb segment stretches at most by this factor (or shrinks by its inverse) to reach the next joint...
MAX_STRETCH = 2.0
# ...and a clavicle's end moves off its line by at most this angle (degrees) to reach the shoulder.
MAX_SHEAR = 35.0
# A stretched segment's stretch fades in over this share of its length from each end (see `ramp`).
RAMP_EASE = 0.2
# An armature's bones sit at one spot when their heads spread over at most this share of the model's size;
# they're then put where their vertex groups are, at least this long (in torso lengths).
BONE_SPOT = 0.001
BONE_LEAST = 0.03
# Unnamed groups: a hand's branch is its thumb when it starts at most this share as far from the wrist as
# the other branches do...
FINGER_THUMB = 0.75
# ...and a side's only limb (a top's arm, a bottom's leg) is a leg when it drops at least this share of how
# far it reaches: arms in an A-pose drop about 0.7 of it, legs nearly all.
LEG_DROP = 0.85
# Refining joints against the body: body vertices this far around a joint (in torso lengths, from the hips
# to the shoulders)...
REFINE_RADIUS = 0.19
# ...look for the model's surface this far (in torso lengths) and must find it for this share of them.
REFINE_REACH = 0.12
REFINE_COVERAGE = 0.6
# Offsets up to this (in torso lengths) are what blending the parts does around a joint, not the joint
# being off: they're left alone, and longer ones shortened by it.
REFINE_SLACK = 0.01
# A joint moved for an offset must leave at most this share of it, or it's put back.
REFINE_SHRINK = 0.8
REFINE_ROUNDS = 3
# Body vertices sampled around each joint at most.
REFINE_SAMPLES = 600
# The joints refined: those of the limbs. The others don't move the model's surface (the neck and head
# turn and move with the torso; a clavicle only stretches toward its shoulder), or are too small.
REFINED = {part_name(kind, side) for side in SIDES
           for kind in ("upperarm", "forearm", "hand", "thigh", "shin", "foot")}


class LineUpError(Exception):
    """Why a model can't be lined up (shown to the user)."""


# -----------------------------------------------------------------------------
# Bone names

_SIDE_WORDS = {"l": "l", "left": "l", "lf": "l", "lft": "l", "r": "r", "right": "r", "rt": "r", "rgt": "r"}

# Words naming a part, most specific first: the first kind with a matching word wins. Words are matched
# against single words of a name and against two neighbouring words run together ("Up Leg" -> "upleg").
_KIND_WORDS = (
    ("foot", ("toe", "toes", "toebase", "ball", "foot", "feet", "ankle", "heel", "tarsal", "metatarsal",
              "metatarsals")),
    ("hand", ("finger", "fingers", "thumb", "index", "middle", "ring", "pinky", "pinkie", "little", "hand",
              "wrist", "palm", "carpal", "metacarpal", "metacarpals")),
    ("forearm", ("forearm", "lowerarm", "loarm", "elbow", "ulna", "forearmbend", "forearmtwist")),
    ("upperarm", ("upperarm", "uparm", "arm", "bicep", "biceps", "becep", "tricep", "triceps", "brachi",
                  "brochi", "deltoid", "humerus", "shldr", "shldrbend", "shldrtwist", "shouldertwist")),
    ("clavicle", ("clavicle", "clav", "collar", "collarbone", "shoulder", "scapula", "trape", "trapezius")),
    ("thigh", ("thigh", "upleg", "upperleg", "femur", "thighbend", "thightwist")),
    ("shin", ("calf", "shin", "knee", "lowerleg", "loleg", "crus", "tibia")),
    ("leg", ("leg",)),
    ("neck", ("neck",)),
    ("head", ("head", "headtop", "skull", "jaw", "eye", "eyes", "eyelid", "ear", "ears", "hair", "face", "brow",
              "lip", "lips", "tongue", "teeth", "nose", "cheek", "mouth", "chin")),
    ("torso", ("pelvis", "hips", "hip", "spine", "chest", "upperchest", "torso", "abdomen", "abs", "waist",
               "breast", "breasts", "boob", "boobs", "bust", "belly", "tummy", "stomach", "butt", "glute",
               "buttock", "upperbody", "lowerbody", "pectoral", "pecs", "pectro", "rib", "ribs", "back",
               "sternum", "venter")),
)
# Words also found inside longer ones ("upperarm01", "leftforearm"), checked in the same order.
_EMBEDDED = (
    ("foot", ("foot", "ankle")),
    ("hand", ("finger", "thumb", "hand", "wrist")),
    ("forearm", ("forearm", "lowerarm", "elbow")),
    ("upperarm", ("upperarm",)),
    ("clavicle", ("clavicle", "shoulder")),
    ("thigh", ("thigh", "upperleg", "upleg")),
    ("shin", ("calf", "lowerleg", "knee")),
    ("neck", ("neck",)),
    ("head", ("head",)),
    ("torso", ("pelvis", "spine", "chest", "breast", "abdomen")),
)
# FFXIV's skeleton: j_ (joints) and n_ (attachment points) bones named by their second word.
_XIV_WORDS = {
    "kosi": "torso", "sebo": "torso", "mune": "torso", "hara": "torso", "sippo": "torso", "sk": "torso",
    "kubi": "neck",
    "kao": "head", "ago": "head", "f": "head", "kami": "head", "mimi": "head", "zera": "head", "ex": "head",
    "sako": "clavicle", "kataarmor": "clavicle",
    "hkata": "upperarm",
    "hhiji": "forearm", "hijisoubi": "forearm",
    "te": "hand", "hte": "hand", "oya": "hand", "hito": "hand", "naka": "hand", "kusu": "hand", "ko": "hand",
    "hizasoubi": "shin",
}
_XIV_LETTERED = {("ude", "a"): "upperarm", ("ude", "b"): "forearm", ("asi", "a"): "thigh", ("asi", "b"): "shin",
                 ("asi", "c"): "shin", ("asi", "d"): "foot", ("asi", "e"): "foot"}
# MMD's Japanese names: sides, and parts with the longer words first.
_JP_SIDES = (("左", "l"), ("右", "r"))
_JP_KINDS = (("足首", "foot"), ("つま先", "foot"), ("手首", "hand"), ("手捩", "forearm"), ("指", "hand"),
             ("ひじ", "forearm"), ("肘", "forearm"), ("腕", "upperarm"), ("肩", "clavicle"), ("ひざ", "shin"),
             ("膝", "shin"), ("足", "thigh"), ("首", "neck"), ("頭", "head"), ("上半身", "torso"), ("下半身", "torso"),
             ("腰", "torso"), ("胸", "torso"))
# Fingers: words of most rigs, FFXIV's second word, MMD's names, and 3ds Max Biped's numbers (Finger0 is
# the thumb, Finger12 the index finger's third bone).
_FINGER_WORDS = (("thumb", ("thumb",)), ("index", ("index", "forefinger", "pointer")),
                 ("middle", ("middle", "mid")), ("ring", ("ring",)), ("little", ("little", "pinky", "pinkie")))
_XIV_FINGERS = {"oya": "thumb", "hito": "index", "naka": "middle", "kusu": "ring", "ko": "little"}
_JP_FINGERS = (("親指", "thumb"), ("人指", "index"), ("人差指", "index"), ("中指", "middle"), ("薬指", "ring"),
               ("小指", "little"))
# Bones of the palm, though some rigs name them after a finger (index_metacarpal_l).
_PALM_WORDS = ("metacarpal", "metacarpals", "carpal", "palm")
# The Sims 4's bones, whose meshes store only a 32-bit FNV-1 hash of each name in lower case: imports that
# couldn't look the names up call the groups "bone_" and the hash in hex (XIV Port Studio's .glb).
_SIMS_BONES = ("b__Pelvis__", "b__Spine0__", "b__Spine1__", "b__Spine2__", "b__Neck__", "b__Head__") + tuple(
    "b__{:s}_{:s}__".format(side, bone) for side in "LR" for bone in (
        "Clavicle", "UpperArm", "ShoulderTwist", "Elbow", "ForeArm", "ForeArmTwist", "Hand", "Thigh", "ThighTwist",
        "Calf", "Foot", "Thumb0", "Thumb1", "Thumb2", "Index0", "Index1", "Index2", "Mid0", "Mid1", "Mid2", "Ring0",
        "Ring1", "Ring2", "Pinky0", "Pinky1", "Pinky2")) + ("b__CAS_L_Breast__", "b__CAS_R_Breast__")


def _fnv1(text):
    """32-bit FNV-1 hash of ``text``, as The Sims 4 hashes bone names."""
    value = 0x811C9DC5
    for byte in text.encode("ascii"):
        value = (value * 0x01000193) & 0xFFFFFFFF
        value ^= byte
    return value


_SIMS_HASHES = {"{:08x}".format(_fnv1(name.lower())): name for name in _SIMS_BONES}


def unhashed(name):
    """A bone name, with The Sims 4's for a group named after the hash of one ("bone_1b82d8b2")."""
    match = re.fullmatch(r"bone_([0-9A-Fa-f]{8})", name)
    return _SIMS_HASHES.get(match.group(1).lower(), name) if match else name


def words(name):
    """The words of a bone name, lower case, without digits: "mixamorig:LeftUpLeg" -> mixamorig left up leg."""
    text = re.sub(r"[^0-9A-Za-z]+", " ", name)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)
    result = []
    for word in text.lower().split():
        word = re.sub(r"[0-9]+", "", word)
        if word:
            result.append(word)
    return result


def _kind_of(tokens):
    candidates = set(tokens)
    for a, b in zip(tokens, tokens[1:]):
        candidates.add(a + b)
    for kind, keys in _KIND_WORDS:
        if any(key in candidates for key in keys):
            return kind
    flat = "".join(tokens)
    for kind, keys in _EMBEDDED:
        if any(key in flat for key in keys):
            return kind
    return None


def classify(name):
    """(kind, side) of a bone name: kind is one of the part kinds, "leg" (thigh or shin, depending on the
    other bones) or None; side is "l", "r" or None."""
    name = unhashed(name)
    jp_side = next((side for mark, side in _JP_SIDES if mark in name), None)
    for mark, kind in _JP_KINDS:
        if mark in name:
            return kind, jp_side
    tokens = words(name)
    if not tokens:
        return None, jp_side
    if tokens[0] == "b":
        tokens = ["middle" if token == "mid" else token for token in tokens]  # The Sims' b__L_Mid0__
    raw = re.sub(r"[^0-9A-Za-z]+", " ", name).lower().split()
    # FFXIV: j_ude_a_l, n_hizasoubi_r...
    if raw[0] in ("j", "n", "iv", "ya") and len(raw) > 1:
        side = raw[-1] if raw[-1] in SIDES else None
        kind = _XIV_WORDS.get(raw[1])
        if kind is None and len(raw) > 2:
            kind = _XIV_LETTERED.get((raw[1], raw[2]))
        if kind is not None:
            return kind, side
    side = None
    rest = []
    for token in tokens:
        if token in _SIDE_WORDS:
            side = _SIDE_WORDS[token]
        else:
            rest.append(token)
    if side is None and rest:
        # "lthigh" written in lower case: a side letter stuck to a part word.
        first = rest[0]
        if len(first) > 3 and first[0] in "lr" and _kind_of([first[1:]]) is not None:
            side, rest = first[0], [first[1:]] + rest[1:]
    kind = _kind_of(rest)
    if kind == TORSO and side is not None and any(token in ("hip", "hips") for token in rest):
        kind = "thigh"  # LeftHip, Hip_L: the thigh bone
    return kind, side


def finger_of(name):
    """The finger ("thumb", "index", "middle", "ring", "little") a hand bone's name says it belongs to, or
    None (the palm, or a name that doesn't say)."""
    name = unhashed(name)
    for mark, finger in _JP_FINGERS:
        if mark in name:
            return finger
    raw = re.sub(r"[^0-9A-Za-z]+", " ", name).lower().split()
    if len(raw) > 1 and raw[0] in ("j", "n", "iv", "ya") and raw[1] in _XIV_FINGERS:
        return _XIV_FINGERS[raw[1]]
    tokens = words(name)
    if any(token in _PALM_WORDS for token in tokens):
        return None
    for finger, keys in _FINGER_WORDS:
        if any(key in tokens for key in keys):
            return finger
    match = re.search(r"finger([0-4])", name.lower())
    return FINGERS[int(match.group(1))] if match else None


# -----------------------------------------------------------------------------
# Parts of a rig

def _resolve_legs(kinds, parents=None):
    """Settle "leg" bones: the first leg bone out from the torso is the thigh, a leg bone below a thigh
    the shin (Mixamo's LeftUpLeg / LeftLeg, VRChat's Leg_L / Knee_L). Without a hierarchy, a leg bone is
    the shin where the same side has a thigh bone."""
    resolved = dict(kinds)
    for name, (kind, side) in kinds.items():
        if kind != "leg":
            continue
        if parents is not None:
            above = False
            parent = parents.get(name)
            while parent is not None:
                if kinds.get(parent, (None, None))[0] in ("thigh", "leg"):
                    above = True
                    break
                parent = parents.get(parent)
        else:
            above = any(k == "thigh" and s == side for k, s in kinds.values())
        resolved[name] = ("shin" if above else "thigh", side)
    return resolved


def bone_parts(names, parents):
    """Part of each bone (``names``; ``parents``: name -> parent name or None), from its name, else the
    nearest ancestor's. Limb bones without a side take their ancestor's part."""
    kinds = _resolve_legs({name: classify(name) for name in names}, parents)
    parts = {}

    def part_of(name):
        if name in parts:
            return parts[name]
        kind, side = kinds.get(name, (None, None))
        parent = parents.get(name)
        if kind in (TORSO, NECK, HEAD):
            result = kind
        elif kind is not None and side is not None:
            result = part_name(kind, side)
        elif parent is not None:
            result = part_of(parent)
        else:
            result = TORSO
        parts[name] = result
        return result

    for name in names:
        part_of(name)
    return _split_fingers(names, parts, parents)


def _split_fingers(names, parts, parents):
    """Move a hand's bones named after a finger to that finger's segments, numbered by how many of the
    finger's bones are above them. Bones below a finger's that say nothing go with it."""
    hands = {part_name("hand", side) for side in SIDES}
    fingers = {name: finger_of(name) for name in names if parts.get(name) in hands}
    found = {}

    def digit(name):
        """(finger, segment) of a hand bone, or None for the palm."""
        if name not in found:
            parent = parents.get(name)
            above = digit(parent) if parent in fingers else None
            own = fingers[name]
            if own is None:
                found[name] = above
            elif above is not None and above[0] == own:
                found[name] = (own, min(above[1] + 1, DIGITS))
            else:
                found[name] = (own, 1)
        return found[name]

    result = dict(parts)
    for name in fingers:
        value = digit(name)
        if value is not None:
            result[name] = digit_name(value[0], value[1], parts[name][-1])
    return result


def split_finger_groups(parts, points, weights):
    """Move a hand's vertex groups named after a finger (``parts``: group name -> part; ``weights`` over
    the groups of ``points``) to that finger's segments. Without an armature to tell which hangs from
    which, they're numbered by how far their weights lie from the rest of the hand."""
    result = dict(parts)
    centroids = weights.centroids(points)
    mass = weights.mass()
    index = {name: k for k, name in enumerate(weights.names)}
    for side in SIDES:
        hand = part_name("hand", side)
        groups = [name for name, part in parts.items() if part == hand and name in index and mass[index[name]] > 0]
        fingers = {name: finger_of(name) for name in groups}
        palm = [index[name] for name in groups if fingers[name] is None] or [index[name] for name in groups]
        if not palm or not any(fingers.values()):
            continue
        middle = np.average(centroids[palm], axis=0, weights=mass[palm])
        for finger in FINGERS:
            own = sorted((name for name in groups if fingers[name] == finger),
                         key=lambda name: np.linalg.norm(centroids[index[name]] - middle))
            for rank, name in enumerate(own):
                result[name] = digit_name(finger, min(rank + 1, DIGITS), side)
    return result


def main_bones(parts, parents):
    """The main bone of each part (``parts``: bone name -> part), which the part turns about: of the bones
    the next part hangs from (the clavicle bone above the upper arm's, not a helper beside it), the
    topmost; the one with the most bones below it on a tie."""
    depth = {}

    def depth_of(name):
        if name not in depth:
            parent = parents.get(name)
            depth[name] = 0 if parent is None else depth_of(parent) + 1
        return depth[name]

    below = {name: 0 for name in parts}
    carries = {name: set() for name in parts}  # parts of the bones below each bone
    for name, part in parts.items():
        parent = parents.get(name)
        while parent is not None:
            if parent in below:
                below[parent] += 1
                carries[parent].add(part)
            parent = parents.get(parent)
    best = {}
    for name, part in parts.items():
        after = NEXT.get(part)
        carrying = after is not None and after in carries[name]
        key = (not carrying, depth_of(name), -below[name], name)
        if part not in best or key < best[part][0]:
            best[part] = (key, name)
    return {part: name for part, (_key, name) in best.items()}


def group_parts_by_name(names, weights=None):
    """Part of each vertex group (``names``) from its name alone, for a model without an armature. Groups
    whose names say nothing join the part they share the most weight with (``weights``, a Weights over
    ``names``), else the torso. Returns (group name -> part, how many groups were named)."""
    kinds = _resolve_legs({name: classify(name) for name in names})
    parts = {}
    for name in names:
        kind, side = kinds[name]
        if kind in (TORSO, NECK, HEAD):
            parts[name] = kind
        elif kind is not None and side is not None:
            parts[name] = part_name(kind, side)
    named = len(parts)
    unnamed = [index for index, name in enumerate(names) if name not in parts]
    if unnamed and weights is not None and parts:
        shared = weights.shared(weights.by_part(parts))
        for index in unnamed:
            if shared[index].max() > 0.0:
                parts[names[index]] = PARTS[int(np.argmax(shared[index]))]
    for name in names:
        parts.setdefault(name, TORSO)
    return parts, named


def has_key_parts(parts, count=3):
    """Whether ``parts`` (name -> part) include at least ``count`` of the thighs and upper arms."""
    found = set(parts.values())
    return sum(part in found for part in KEY_PARTS) >= count


def has_half(parts):
    """Whether ``parts`` (name -> part) are enough to line up by: three of the thighs and upper arms, or at
    least both of either, half a body (a top or a bottom)."""
    found = set(parts.values())
    return has_key_parts(parts) or {"thigh_l", "thigh_r"} <= found or {"upperarm_l", "upperarm_r"} <= found


# -----------------------------------------------------------------------------
# Weights

class Weights:
    """Vertex group weights of ``count`` points over the groups ``names``, normalized per point (unweighted
    points have none), kept sparse: point ``vertex[k]`` has ``weight[k]`` in group ``group[k]``."""

    def __init__(self, count, names, vertex=(), group=(), weight=()):
        self.count = int(count)
        self.names = list(names)
        vertex = np.asarray(vertex, dtype=np.int64)
        group = np.asarray(group, dtype=np.int64)
        weight = np.asarray(weight, dtype=np.float64)
        keep = weight > 0.0
        self.vertex, self.group, weight = vertex[keep], group[keep], weight[keep]
        total = np.bincount(self.vertex, weight, minlength=self.count)
        self.weight = weight / total[self.vertex] if len(weight) else weight

    @classmethod
    def stack(cls, names, sets):
        """One set of weights (over the same ``names``) for the points of ``sets``, one after another."""
        if not sets:
            return cls(0, names)
        first = np.cumsum([0] + [weights.count for weights in sets])
        return cls(first[-1], names, np.concatenate([w.vertex + f for w, f in zip(sets, first)]),
                   np.concatenate([w.group for w in sets]), np.concatenate([w.weight for w in sets]))

    def column(self, group):
        """Every point's weight in one group."""
        result = np.zeros(self.count)
        chosen = self.group == group
        result[self.vertex[chosen]] = self.weight[chosen]
        return result

    def mass(self):
        """Each group's weight summed over the points."""
        return np.bincount(self.group, self.weight, minlength=len(self.names))

    def centroids(self, points):
        """Each group's centroid, the points weighted by its weights."""
        sums = np.zeros((len(self.names), 3))
        np.add.at(sums, self.group, self.weight[:, None] * points[self.vertex])
        return sums / np.maximum(self.mass()[:, None], 1e-12)

    def overlap(self):
        """Groups x groups: how much weight each pair shares (products of their weights, summed over the
        points); 0 on the diagonal."""
        size = len(self.names)
        result = np.zeros((size, size))
        if not len(self.vertex):
            return result
        order = np.argsort(self.vertex, kind="stable")
        vertex, group, weight = self.vertex[order], self.group[order], self.weight[order]
        slot = np.arange(len(vertex)) - np.searchsorted(vertex, vertex)
        slots = int(slot.max()) + 1
        groups = np.full((self.count, slots), -1, dtype=np.int64)
        weights = np.zeros((self.count, slots))
        groups[vertex, slot] = group
        weights[vertex, slot] = weight
        for a in range(slots):
            for b in range(slots):
                if a == b:
                    continue
                used = (groups[:, a] >= 0) & (groups[:, b] >= 0)
                if used.any():
                    np.add.at(result, (groups[used, a], groups[used, b]), weights[used, a] * weights[used, b])
        return result

    def by_part(self, parts):
        """Points x PARTS: each point's weight per part (``parts``: group name -> part; other groups don't
        count)."""
        lookup = np.array([PART_INDEX[parts[name]] if name in parts else -1 for name in self.names] + [-1])
        columns = lookup[self.group]
        used = columns >= 0
        result = np.zeros((self.count, len(PARTS)))
        np.add.at(result, (self.vertex[used], columns[used]), self.weight[used])
        return result

    def shared(self, part_weights):
        """Groups x PARTS: how much weight each group shares with each part (``part_weights``: points x
        PARTS)."""
        result = np.zeros((len(self.names), part_weights.shape[1]))
        np.add.at(result, self.group, self.weight[:, None] * part_weights[self.vertex])
        return result


def boundary(points, first, second):
    """Where weight ``first`` hands over to ``second`` (two columns of point weights): the centroid of
    the points both weigh, weighted by the product. None if they share none."""
    share = first * second
    total = share.sum()
    if total <= 1e-12:
        return None
    return (points * share[:, None]).sum(axis=0) / total


def joints_from_weights(points, part_weights):
    """Joint of every part that hands over from the part above it, estimated from the weights (points x
    PARTS)."""
    joints = {}
    present = part_weights.sum(axis=0) > 0.0
    for part in PARTS[1:]:
        if not present[PART_INDEX[part]]:
            continue
        parent = PARENT[part]
        # A part missing in between (no clavicle) hands over from the one above it.
        while parent != TORSO and not present[PART_INDEX[parent]]:
            parent = PARENT[parent]
        joint = boundary(points, part_weights[:, PART_INDEX[parent]], part_weights[:, PART_INDEX[part]])
        if joint is not None:
            joints[part] = joint
    return joints


# -----------------------------------------------------------------------------
# Groups without telling names

def _spanning_tree(overlap, mass, root):
    """Maximum spanning tree of the groups by the weight they share (Prim's), grown from group ``root``.
    Returns the parent index of each group: -1 for the root and for groups sharing nothing with the
    tree."""
    count = len(mass)
    parent = np.full(count, -1)
    alive = mass > 0.0
    if not alive[root]:
        return parent
    in_tree = np.zeros(count, dtype=bool)
    in_tree[root] = True
    best = np.where(alive & ~in_tree, overlap[root], -1.0)
    best_from = np.full(count, root)
    while True:
        candidate = int(np.argmax(best))
        if best[candidate] <= 0.0:
            break
        in_tree[candidate] = True
        parent[candidate] = best_from[candidate]
        better = alive & ~in_tree & (overlap[candidate] > best)
        best[better] = overlap[candidate][better]
        best_from[better] = candidate
        best[in_tree] = -1.0
    return parent


def _polyline_lengths(points):
    """Distance along the polyline through ``points`` to each of them."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))


def _arm_cuts(along, lengths):
    """Where an arm's clavicle and upper arm end, as indices into ``along`` (distances along the arm to
    where its groups hand over, the last at the wrist), from the body's arm (``lengths``: distances along
    it to its upper arm, forearm and hand joints). The elbow is where it lies as far along as the body's;
    the shoulder where the upper arm is as long for the forearm as the body's. Rigs start clavicles at very
    different places, so their share of the arm says little. Returns [shoulder, elbow] or None."""
    if len(along) < 3:
        return None
    share = along[1:-1] / max(along[-1], 1e-12)
    elbow = 1 + int(np.argmin(np.abs(share - lengths[2] / lengths[3])))
    upper = (along[-1] - along[elbow]) * (lengths[2] - lengths[1]) / max(lengths[3] - lengths[2], 1e-12)
    shoulder = int(np.argmin(np.abs(along[elbow] - along[:elbow] - upper)))
    return [shoulder, elbow]


class _GroupTree:
    """Vertex groups as a tree by the weight they share, with where each lies."""

    def __init__(self, points, weights):
        self.points = points
        self.weights = weights
        self.names = weights.names
        self.mass = weights.mass()
        self.alive = self.mass > 0.0
        self.centroids = weights.centroids(points)
        self.overlap = weights.overlap()
        self.parent = np.full(len(self.names), -1)
        self.children = [[] for _ in self.names]
        self._columns = {}

    def grow(self, root):
        """Hang the groups from ``root`` along the tree of what they share most."""
        self.parent = _spanning_tree(self.overlap, self.mass, root)
        self.children = [[] for _ in self.names]
        for group, parent in enumerate(self.parent):
            if parent >= 0:
                self.children[parent].append(group)

    def column(self, group):
        if group not in self._columns:
            self._columns[group] = self.weights.column(group)
        return self._columns[group]

    def handover(self, a, b):
        point = boundary(self.points, self.column(a), self.column(b))
        return (self.centroids[a] + self.centroids[b]) / 2.0 if point is None else point

    def chain(self, root, members):
        """Path from ``root`` to the farthest group of ``members``, stepping between centroids."""
        distance = {root: 0.0}
        order = [root]
        for group in order:
            for child in self.children[group]:
                if child in members and child not in distance:
                    distance[child] = distance[group] + np.linalg.norm(self.centroids[child] - self.centroids[group])
                    order.append(child)
        path = [max(distance, key=distance.get)]
        while path[-1] != root:
            path.append(self.parent[path[-1]])
        return path[::-1]


def _label_fingers(tree, hand, before, members, side):
    """Finger segments among an arm's groups (``members``) hanging from its hand group ``hand``, which
    hangs from group ``before`` (or None). Each branch off the hand is a finger, its groups numbered from
    the hand out. The thumb is the branch starting nearest the wrist, clearly nearer than the others; the
    others are named across the knuckles from the thumb's side, index to little. Without a thumb to tell
    which side that is, fingers stay with the hand. Returns group -> part."""
    wrist = tree.handover(before, hand) if before is not None else tree.centroids[hand]
    branches = []
    for child in tree.children[hand]:
        if child not in members:
            continue
        chain = [child]
        while True:
            below = [group for group in tree.children[chain[-1]] if group in members]
            if not below:
                break
            chain.append(max(below, key=lambda group: tree.mass[group]))
        branches.append((chain, tree.handover(hand, child)))
    if len(branches) < 3:
        return {}
    reach = np.array([np.linalg.norm(base - wrist) for _chain, base in branches])
    thumb = int(np.argmin(reach))
    rest = [k for k in range(len(branches)) if k != thumb]
    if reach[thumb] >= FINGER_THUMB * np.median(reach[rest]):
        return {}
    bases = np.array([branches[k][1] for k in rest])
    across = np.linalg.svd(bases - bases.mean(axis=0), full_matrices=False)[2][0]
    if np.dot(branches[thumb][1] - bases.mean(axis=0), across) > 0.0:
        across = -across
    rest.sort(key=lambda k: float(np.dot(branches[k][1], across)))
    result = {}
    for finger, k in zip(FINGERS, [thumb] + rest):
        for segment, group in enumerate(branches[k][0]):
            result[group] = digit_name(finger, min(segment + 1, DIGITS), side)
    return result


def _mirror_axis(points, samples=1000, step=5.0):
    """The horizontal direction from a model's right to its left (sign aside) and where its midline lies
    along it: the one, to within ``step`` degrees, the model is mirrored across best for how far it spreads
    along it. A body is mirrored left to right, not front to back, even where it spreads about as far front
    to back (two legs close together); where it isn't quite mirrored (a hand held differently), it spreads
    wider left to right. Returns (unit 2D vector, position of the midline along it)."""
    flat = points[:, :2]
    chosen = points[np.linspace(0, len(points) - 1, min(samples, len(points))).astype(np.int64)]
    tree = KDTree(len(points))
    for index, point in enumerate(points):
        tree.insert(point, index)
    tree.balance()
    best = None
    for angle in np.radians(np.arange(0.0, 180.0, step)):
        axis = np.array((np.cos(angle), np.sin(angle)))
        low, high = np.percentile(flat @ axis, (0.5, 99.5))
        midline = (low + high) / 2.0
        mirrored = chosen.copy()
        mirrored[:, :2] -= 2.0 * np.outer(chosen[:, :2] @ axis - midline, axis)
        # On average: legs, round as they are, mirror front to back about as well as across; the rest tells.
        mismatch = np.mean([tree.find(point)[2] for point in mirrored]) / max(high - low, 1e-6)
        if best is None or mismatch < best[0]:
            best = (mismatch, axis, midline)
    _mismatch, axis, midline = best
    sign = np.sign(axis[0]) if abs(axis[0]) >= abs(axis[1]) else np.sign(axis[1])
    return axis * sign, midline * sign


def label_anonymous(points, weights, target_joints):
    """Part of each vertex group when their names say nothing (bone_00, bone_01...), from how the groups
    overlap and where they lie. The model is taken to stand upright (+Z) with its left on +X, or on +Y
    when it spreads along Y; feet pointing the other way flip that. ``target_joints`` (the body's, with
    all limb joints) lend their proportions to find joints along each limb. Returns group name -> part."""
    tree = _GroupTree(points, weights)
    names, alive, centroids, mass = tree.names, tree.alive, tree.centroids, tree.mass
    if alive.sum() < 3:
        return {name: TORSO for name in names}

    # Sideways: the horizontal direction the model spreads the most along (arms, shoulders, hips), and its
    # midline, both from where its points are rather than how much weight each group has: a skirt weighted
    # to one leg would pull a weighted midline off the body's.
    lateral, midline = _mirror_axis(points[np.unique(weights.vertex)])
    across = centroids[:, :2] @ lateral - midline
    heights = centroids[:, 2]
    # The model's size: how far its groups spread, whichever way (half a body spreads less high than wide).
    size = max(np.ptp(centroids[alive], axis=0).max(), 1e-6)
    # In the middle: groups with a fifth of their own points well to each side of the midline (a pelvis whose
    # weight an uneven skirt pulls to one side still does), and those lying there that don't keep to one side
    # of it (a leg whose skirt bunches toward the middle does).
    straddles = np.zeros(len(names), dtype=bool)
    one_sided = np.zeros(len(names), dtype=bool)
    for group in np.flatnonzero(alive):
        own = tree.column(group) >= 0.5
        if own.sum() >= 3:
            spread = points[own][:, :2] @ lateral - midline
            low, high = np.percentile(spread, (20, 80))
            straddles[group] = low < -0.02 * size and high > 0.02 * size
            low, high = np.percentile(spread, (5, 95))
            one_sided[group] = high < 0.005 * size or low > -0.005 * size
    central = alive & (straddles | (~one_sided & (np.abs(across) < 0.03 * size)))
    if not central.any():
        central = alive & (np.abs(across) <= np.abs(across[alive]).min() + 1e-9)
    # No group hangs from one on the other side: what joins them (a skirt weighted to both legs) isn't a limb.
    off = alive & ~central
    tree.overlap[off[:, None] & off[None, :] & (across[:, None] * across[None, :] < 0.0)] = 0.0
    # The tree hangs from the heaviest group in the middle (the chest or pelvis), so limbs run outward.
    tree.grow(int(np.argmax(np.where(central, mass, -1.0))))

    # Groups off the middle, joined in the tree, form the limbs (and bits like breasts or ears).
    components = []
    seen = np.zeros(len(names), dtype=bool)
    for group in range(len(names)):
        if not alive[group] or central[group] or seen[group]:
            continue
        members, stack = [], [group]
        seen[group] = True
        while stack:
            current = stack.pop()
            members.append(current)
            neighbours = tree.children[current] + ([tree.parent[current]] if tree.parent[current] >= 0 else [])
            for other in neighbours:
                if alive[other] and not central[other] and not seen[other]:
                    seen[other] = True
                    stack.append(other)
        components.append(members)

    parts = {}
    limbs, bits = [], []
    for members in components:
        root = next((g for g in members if tree.parent[g] < 0 or central[tree.parent[g]]), members[0])
        attach = tree.parent[root]
        base = centroids[attach] if attach >= 0 else centroids[root]
        extent = max(np.linalg.norm(centroids[g] - base) for g in members)
        side = "l" if np.average(across[members], weights=mass[members]) > 0.0 else "r"
        lowest = min(heights[g] for g in members)
        (limbs if extent >= 0.2 * size else bits).append((members, root, attach, side, base[2], lowest, extent))
    # Per side, the leg is the limb reaching lowest, the arm the other one hanging from highest up. A side with
    # only one (a top or a bottom) has a leg when it hangs about straight down.
    arms, legs = {}, {}
    for side in SIDES:
        mine = [limb for limb in limbs if limb[3] == side]
        if len(mine) >= 2:
            legs[side] = min(mine, key=lambda limb: limb[5])
            arms[side] = max((limb for limb in mine if limb is not legs[side]), key=lambda limb: limb[4])
        elif len(mine) == 1:
            _members, _root, _attach, _side, base_height, lowest, extent = mine[0]
            (legs if base_height - lowest > LEG_DROP * extent else arms)[side] = mine[0]

    def fill(members, fallback):
        """Groups of a limb off its main chain take the part of the group they hang from."""
        for group in members:
            if names[group] in parts:
                continue
            current = group
            while names[current] not in parts and tree.parent[current] >= 0 and tree.parent[current] in members:
                current = tree.parent[current]
            parts[names[group]] = parts.get(names[current], fallback)

    def target_lengths(chain_parts):
        if not all(part in target_joints for part in chain_parts):
            return None
        return _polyline_lengths(np.array([target_joints[part] for part in chain_parts]))

    shoulders, spans = [], []
    for side, (members, root, attach, _side, _base, _lowest, _extent) in arms.items():
        inside = set(members)
        chain = tree.chain(root, inside)
        order = [part_name(kind, side) for kind in ARM]
        lengths = target_lengths(order)
        start = tree.handover(attach, root) if attach >= 0 else centroids[root]
        # The hand is where the chain branches into fingers.
        hand = next((k for k, g in enumerate(chain) if k > 0 and sum(c in inside for c in tree.children[g]) >= 3),
                    None)
        if hand is None:
            hand = len(chain) - 1  # no fingers: the last group is the hand
        arm = chain[:hand]
        if lengths is not None and arm:
            points_ = [start] + [tree.handover(a, b) for a, b in zip(arm, arm[1:])]
            points_.append(tree.handover(arm[-1], chain[hand]))
            along = _polyline_lengths(np.array(points_))
            cuts = _arm_cuts(along, lengths)
            if cuts is not None:
                bounds = cuts + [len(arm)]
                for kind, (first, end) in zip(ARM[:3], zip([0] + bounds[:-1], bounds)):
                    for group in arm[first:end]:
                        parts[names[group]] = part_name(kind, side)
                shoulders.append(points_[cuts[0]])
                spans.append((along[-1] - along[cuts[0]], lengths[3] - lengths[1]))
        for group in chain[hand:]:
            parts[names[group]] = order[3]
        fill(members, order[1])
        wrist = chain[hand - 1] if hand > 0 else (attach if attach >= 0 else None)
        for group, part in _label_fingers(tree, chain[hand], wrist, inside, side).items():
            parts[names[group]] = part

    # The model's size from the arms (shoulder to wrist), else from the legs themselves.
    scale = None
    if spans:
        scale = float(np.mean([model / max(body, 1e-12) for model, body in spans]))
    for side, (members, root, attach, _side, _base, _lowest, _extent) in legs.items():
        inside = set(members)
        chain = tree.chain(root, inside)
        order = [part_name(kind, side) for kind in LEG]
        lengths = target_lengths(order)
        start = tree.handover(attach, root) if attach >= 0 else centroids[root]
        kinds = [order[0]] * len(chain)
        if lengths is not None and len(chain) > 1:
            points_ = [start] + [tree.handover(a, b) for a, b in zip(chain, chain[1:])]
            along = _polyline_lengths(np.array(points_))
            if scale is None:
                lowest = min(tree.points[tree.column(g) > 0.5][:, 2].min(initial=centroids[g][2]) for g in chain)
                scale = (start[2] - lowest) / max(target_joints[order[0]][2], 1e-12)
            knee = 1 + int(np.argmin(np.abs(along[1:] - lengths[1] * scale)))
            if along[knee] < 0.6 * lengths[1] * scale:
                knee = len(chain)  # a stub (a top's shorts) ending well above where a knee would be
            ankle = None
            if knee + 1 < len(along):
                k = knee + 1 + int(np.argmin(np.abs(along[knee + 1:] - lengths[2] * scale)))
                if abs(along[k] - lengths[2] * scale) < 0.25 * lengths[2] * scale:
                    ankle = k
            for k in range(len(chain)):
                kinds[k] = order[0] if k < knee else (order[1] if ankle is None or k < ankle else order[2])
        for group, part in zip(chain, kinds):
            parts[names[group]] = part
        fill(members, order[0])

    # The middle: torso up to the shoulders, then the neck, then the head.
    shoulder_height = np.mean([point[2] for point in shoulders]) if shoulders else None
    middle = sorted((g for g in range(len(names)) if central[g]), key=lambda g: heights[g])
    above = [g for g in middle if shoulder_height is not None and heights[g] > shoulder_height]
    for group in middle:
        parts[names[group]] = TORSO
    if above:
        parts[names[above[0]]] = NECK
        for group in above[1:]:
            parts[names[group]] = HEAD
    # Bits (breasts, ears...) go with what they hang from; anything else left (wings, say) with the torso.
    for members, _root, attach, _side, _base, _lowest, _extent in bits:
        for group in members:
            parts[names[group]] = parts.get(names[attach], TORSO) if attach >= 0 else TORSO
    for name in names:
        parts.setdefault(name, TORSO)

    # Feet point forward: if they point the other way, the model faces +X (or -Y) and left is right.
    feet = [part_name("foot", side) for side in SIDES]
    shins = [part_name("shin", side) for side in SIDES]
    part_weights = weights.by_part(parts)
    forward = np.array((lateral[1], -lateral[0]))  # left x up: -Y with the left on +X
    toes = []
    for foot, shin in zip(feet, shins):
        column = part_weights[:, PART_INDEX[foot]]
        if column.sum() > 0.0:
            ankle = boundary(points, part_weights[:, PART_INDEX[shin]], column)
            if ankle is not None:
                toes.append(((points * column[:, None]).sum(axis=0) / column.sum() - ankle)[:2])
    if toes and np.dot(np.mean(toes, axis=0), forward) < 0.0:
        swap = {part: part[:-1] + ("r" if part.endswith("l") else "l") for part in PARTS if part[-2:] in ("_l", "_r")}
        parts = {name: swap.get(part, part) for name, part in parts.items()}
    return parts


# -----------------------------------------------------------------------------
# Solving

def _unit(vector):
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-12 else vector


def swing(a, b):
    """The smallest rotation (3x3) turning unit vector ``a`` onto unit vector ``b``."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-9:
        # Half a turn about any axis square to a.
        axis = np.cross(a, (1.0, 0.0, 0.0))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, (0.0, 1.0, 0.0))
        axis = _unit(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array(((0.0, -v[2], v[1]), (v[2], 0.0, -v[0]), (-v[1], v[0], 0.0)))
    return np.eye(3) + k + k @ k / (1.0 + c)


def affine(pivot, target, rotation, scale):
    """4x4 matrix of x -> target + scale * rotation (x - pivot): turned, scaled and moved from ``pivot``
    to ``target``."""
    linear = scale * rotation
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = target - linear @ pivot
    return matrix


def apply_matrix(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _frame(up, lateral):
    """Rotation whose columns are the up, left and forward axes of a body."""
    up = _unit(up)
    lateral = _unit(lateral - up * np.dot(lateral, up))
    return np.column_stack((up, lateral, np.cross(lateral, up)))


def _mid(joints, a, b):
    if a in joints and b in joints:
        return (joints[a] + joints[b]) / 2.0
    return None


class Solution:
    """How each part moves: it turns and scales about its joint (``transforms``: part -> 4x4 world matrix,
    whose turn is in ``rotations``: part -> 3x3) and, stretched, slides along its length (``ramps``: part
    -> (joint, axis, correction), see `move`). And what it took: the ``scale`` of the whole model, each
    stretched segment's length factor (``stretches``), how far the torso turned about the vertical
    (``turn``, degrees), how many fingers turned to point like the body's (``fingers``), the parts only one
    side has (``missing``), and which half of a body the model is when it's only that ("upper" or "lower",
    ``half``; see `solve`)."""

    def __init__(self):
        self.transforms = {}
        self.rotations = {}
        self.ramps = {}
        self.stretches = {}
        self.scale = 1.0
        self.turn = 0.0
        self.fingers = 0
        self.missing = []
        self.half = None
        self.torso_length = 0.0

    def move(self, part, points):
        """``points`` (world, one or n x 3) moved by ``part``: turned and scaled by its transform, then, if
        stretched, slid by its correction as far as they lie along the part (`ramp`): not at all at its
        joint, fully at the next part's. So its end lands where the next part starts, points past it slide
        along instead of stretching on, and where the weights of two parts blend, the stretch doesn't
        change abruptly."""
        points = np.asarray(points, dtype=np.float64)
        moved = apply_matrix(self.transforms[part], points)
        stretch = self.ramps.get(part)
        if stretch is not None:
            joint, axis, correction = stretch
            moved = moved + np.multiply.outer(ramp((points - joint) @ axis / np.dot(axis, axis)), correction)
        return moved


def ramp(along):
    """Share of a stretched segment's correction a point gets, by how far along the segment it lies (0 at
    its joint, 1 at the next part's): none before, all beyond, and in between easing in and out over
    RAMP_EASE of the length at each end. The segment then stretches evenly in its middle and not at all
    at its joints, so a squashed thigh doesn't crease against a stretched shin."""
    along = np.clip(along, 0.0, 1.0)
    ease = RAMP_EASE
    curve = 1.0 / (2.0 * ease * (1.0 - ease))
    middle = (along - ease / 2.0) / (1.0 - ease)
    return np.where(along < ease, along * along * curve,
                    np.where(along > 1.0 - ease, 1.0 - (1.0 - along) ** 2 * curve, middle))


# The top of the torso: between the upper arms' joints, else the clavicles', else the neck's.
_TOPS = (("upperarm_l", "upperarm_r"), ("clavicle_l", "clavicle_r"), (NECK, NECK))


def torso_ends(joints, other=None):
    """(hips, top) of a skeleton: the middle between the thighs' joints, and the top of the torso (see
    _TOPS), using only joints ``other`` has too when given. Either can be None."""
    hips = _mid(joints, "thigh_l", "thigh_r")
    for a, b in _TOPS:
        if other is None or (a in other and b in other):
            top = _mid(joints, a, b)
            if top is not None:
                return hips, top
    return hips, None


def _aim(joints, part, tips=None, other=None):
    """Where ``part`` of a skeleton (``joints``) points, as a vector from its joint, or None: at the next
    part's joint, else at its tip (``tips``: part -> point); a hand at its knuckles (the middle of its
    fingers' first joints, thumb left out, of the fingers ``other`` has too); a finger's last segment
    without a tip on along the segment before it."""
    joint = joints[part]
    after = NEXT.get(part)
    if after in joints:
        return joints[after] - joint
    if tips is not None and part in tips:
        return tips[part] - joint
    if part in (part_name("hand", side) for side in SIDES):
        firsts = [digit_name(finger, 1, part[-1]) for finger in FINGERS[1:]]
        knuckles = [joints[first] for first in firsts if first in joints and (other is None or first in other)]
        return np.mean(knuckles, axis=0) - joint if len(knuckles) >= 2 else None
    if part in DIGIT_PARTS and PARENT[part] in joints:
        return joint - joints[PARENT[part]]
    return None


def _stretch(result, part, joint, end, goal):
    """Stretch ``part`` (its joint and the next part's, ``joint`` and ``end``, in the model) so its end
    lands on ``goal``, or as close as MAX_STRETCH and MAX_SHEAR let it."""
    matrix = result.transforms[part]
    start, reach = apply_matrix(matrix, joint), apply_matrix(matrix, end)
    length = max(np.linalg.norm(reach - start), 1e-12)
    correction = goal - reach

    def fits(share):
        wanted = reach + share * correction - start
        size = np.linalg.norm(wanted)
        if not 1.0 / MAX_STRETCH <= size / length <= MAX_STRETCH:
            return False
        cosine = np.dot(wanted, reach - start) / max(size * length, 1e-24)
        return cosine >= np.cos(np.radians(MAX_SHEAR)) - 1e-9

    share = 1.0
    if not fits(share):
        low, high = 0.0, 1.0
        for _step in range(30):
            middle = (low + high) / 2.0
            low, high = (middle, high) if fits(middle) else (low, middle)
        share = low
    correction = share * correction
    result.ramps[part] = (np.array(joint, dtype=np.float64), end - joint, correction)
    result.stretches[part] = float(np.linalg.norm(reach + correction - start) / length)


# Half a body is as big as its arms (upper arms and forearms) or its legs (thighs) say: over whole bodies the
# arms follow the torso's size within a few percent, far closer than the shoulders' or hips' width do.
_ARM_SEGMENTS = (("upperarm", "forearm"), ("forearm", "hand"))
_LEG_SEGMENTS = (("thigh", "shin"),)


def _limb_scale(source, target, segments):
    """How much bigger the body's limb segments (``segments``: pairs of part kinds, joint to joint) are than
    the model's, over those both have, or None."""
    model = body = 0.0
    for side in SIDES:
        for a, b in segments:
            a, b = part_name(a, side), part_name(b, side)
            if all(part in source and part in target for part in (a, b)):
                model += np.linalg.norm(source[b] - source[a])
                body += np.linalg.norm(target[b] - target[a])
    return body / model if model > 1e-9 else None


def solve(source, target, stretch=True, tips=None):
    """How to move each part of a model whose joints are ``source`` (part -> world position) so it lines
    up with a body whose joints are ``target``; ``tips`` (part -> point) are where the model's segments
    end that it lacks the next part of (see `segment_tips`). A model with shoulders but no hips (a top), or
    hips but no shoulders (a bottom), is taken to stand upright like the body and sized by its arms or legs.
    Raises LineUpError with neither, or without the arms or legs to size half a body by.

    Each part starts where the part above puts its joint, and turns the least it takes to point the way
    the body's does. Stretching then slides the end of each limb segment onto the body's next joint (see
    `Solution.move`); one whose end the model lacks (a leg without a foot) stretches like the limb
    segment above it. The neck, head and feet keep their angle to the torso, and clavicles turn with it."""
    result = Solution()
    (hips_s, top_s), (hips_t, top_t) = torso_ends(source, target), torso_ends(target, source)
    if top_t is None:
        top_t = torso_ends(target)[1]  # the model has no shoulders or neck at all (a bottom)
    if hips_t is None:
        raise LineUpError("Found no hips (thigh bones or groups) in the body")
    if top_t is None:
        raise LineUpError("Found no shoulders or neck in the body")
    scale = None
    if hips_s is None or top_s is None:
        # Half a body (a top without hips, or a bottom without shoulders) stands upright like the body, as big as
        # its arms or legs say, and lines up by the end it has.
        if top_s is not None:
            scale, result.half = _limb_scale(source, target, _ARM_SEGMENTS), "upper"
        elif hips_s is not None:
            scale, result.half = _limb_scale(source, target, _LEG_SEGMENTS), "lower"
        if scale is None:
            raise LineUpError("Found no hips (thigh bones or groups) in the model" if hips_s is None else
                              "Found no shoulders or neck in the model")
        if hips_s is None:
            hips_s = top_s - (top_t - hips_t) / scale
        else:
            top_s = hips_s + (top_t - hips_t) / scale
    frames = []
    shoulders = next(((a, b) for a, b in (("upperarm_l", "upperarm_r"), ("clavicle_l", "clavicle_r"))
                      if all(part in source and part in target for part in (a, b))), None)
    for joints, hips, top in ((source, hips_s, top_s), (target, hips_t, top_t)):
        lateral = np.zeros(3)
        for pair in (("thigh_l", "thigh_r"), shoulders):
            if pair is not None and all(part in source and part in target for part in pair):
                lateral = lateral + _unit(joints[pair[0]] - joints[pair[1]])
        if np.linalg.norm(lateral) < 1e-9:
            raise LineUpError("Found no left and right in the model")
        frames.append(_frame(top - hips, lateral))
    torso_rotation = frames[1] @ frames[0].T
    if scale is None:
        scale = np.linalg.norm(top_t - hips_t) / max(np.linalg.norm(top_s - hips_s), 1e-12)
    result.scale = float(scale)
    result.torso_length = float(np.linalg.norm(top_t - hips_t))
    forward = frames[0][:, 2]
    turned = torso_rotation @ forward
    result.turn = float(np.degrees(np.arctan2(forward[0] * turned[1] - forward[1] * turned[0],
                                              forward[0] * turned[0] + forward[1] * turned[1])))
    transforms, rotations, ramps = result.transforms, result.rotations, result.ramps
    transforms[TORSO] = affine(hips_s, hips_t, torso_rotation, scale)
    rotations[TORSO] = torso_rotation

    for part in PARTS[1:]:
        parent = PARENT[part]
        straight = None
        if part in DIGIT_PARTS and part in source and part not in target:
            # A finger segment the body doesn't have (FFXIV's thumbs have two) goes on along the body's finger;
            # of a finger the body has none of, along the finger's first segment.
            above = parent
            while above in DIGIT_PARTS and above not in target:
                above = PARENT[above]
            if above in DIGIT_PARTS:
                straight = _aim(target, above, None, source)
            elif parent in DIGIT_PARTS:
                finger, _segment, side = DIGIT_PARTS[part]
                first = digit_name(finger, 1, side)
                aim = _aim(source, first, tips) if first in source else None
                if aim is not None:
                    straight = rotations[first] @ _unit(aim)
        if (part not in source or part not in target) and straight is None:
            if part in source or part in target:
                result.missing.append(part)
            # Missing on either side: it moves with the part above it.
            transforms[part] = transforms[parent]
            rotations[part] = rotations[parent]
            if parent in ramps:
                ramps[part] = ramps[parent]
            continue
        pivot = source[part]
        after = NEXT.get(part)
        stretched = stretch and part in STRETCHED and after in source and after in target
        if stretch and part in STRETCHED and parent != TORSO and (parent not in source or parent not in target):
            base = target[part]  # nothing above stretched to it (no clavicle): it starts at the body's joint
        else:
            base = result.move(parent, pivot)
        rotation = rotations[parent]
        if part in UPRIGHT:
            rotation = torso_rotation
        elif part not in SHOULDERS:
            aim = _aim(source, part, tips, target)
            wanted = straight if straight is not None else _aim(target, part, None, source)
            if stretched:
                wanted = target[after] - base  # from where it starts, at the body's next joint
            if aim is not None and wanted is not None:
                rotation = swing(rotations[parent] @ _unit(aim), _unit(wanted)) @ rotations[parent]
                if DIGIT_PARTS.get(part, (None, 0))[1] == 1:
                    result.fingers += 1
        transforms[part] = affine(pivot, base, rotation, scale)
        rotations[part] = rotation
        if stretched:
            _stretch(result, part, pivot, source[after], target[after])
        elif (stretch and part in STRETCHED and after in target and tips is not None and part in tips and
              parent in result.stretches and parent not in SHOULDERS):
            start, reach = apply_matrix(transforms[part], pivot), apply_matrix(transforms[part], tips[part])
            _stretch(result, part, pivot, tips[part], start + result.stretches[parent] * (reach - start))
    return result


def deform(points, part_weights, solution):
    """``points`` (world) moved by their parts (`Solution.move`), blended by ``part_weights`` (points x
    PARTS; unweighted points move with the torso)."""
    weights = part_weights.copy()
    weights[weights.sum(axis=1) <= 0.0, PART_INDEX[TORSO]] = 1.0
    result = np.zeros_like(points)
    for index, part in enumerate(PARTS):
        column = weights[:, index]
        used = column > 0.0
        if used.any():
            result[used] += column[used, None] * solution.move(part, points[used])
    return result


def _surface_offset(pairs):
    """How far a model's surface sits off the body's, from (body point, nearest model point, model normal)
    ``pairs``: the shift that best moves the body's points onto the model's surface planes (least squares,
    point to plane), in the directions they pin down; a tube can't tell a shift along itself."""
    body, model, normals = (np.array(column) for column in zip(*pairs))
    gaps = np.einsum("ij,ij->i", model - body, normals)
    keep = np.ones(len(gaps), dtype=bool)
    shift = np.zeros(3)
    for _round in range(2):
        # Least squares, then again without the worst fitting fifth (other parts, folds).
        n, g = normals[keep], gaps[keep]
        values, vectors = np.linalg.eigh(n.T @ n)
        pinned = values > 0.05 * values.max()
        shift = vectors[:, pinned] @ ((vectors[:, pinned].T @ (n.T @ g)) / values[pinned])
        residuals = np.abs(gaps - normals @ shift)
        keep = residuals <= np.percentile(residuals, 80)
    return shift


def refine_joints(source, target, points, tris, part_weights, body_points, rounds=REFINE_ROUNDS):
    """Correct joints estimated from weights (``source``) against the body's surface: line the model
    (``points``, triangles ``tris``, ``part_weights``) up with its joints on the body's (``target``), then
    wherever the model's surface around a body joint sits off the body's (``body_points``) as a whole,
    move that joint by as much. Only where the model covers the body closely around the joint (it's a
    body, or skin tight there), and only if that makes the offset go away: blending the parts reshapes a
    model around its joints a little, which moving a joint doesn't undo. Returns the corrected joints and
    how many were corrected."""
    joints = dict(source)
    hips, top = torso_ends(target)
    if hips is None or top is None or len(tris) == 0:
        return joints, 0
    size = np.linalg.norm(top - hips)
    radius, reach, slack = REFINE_RADIUS * size, REFINE_REACH * size, REFINE_SLACK * size
    samples = {}
    for part, joint in target.items():
        if part not in REFINED or part not in joints:
            continue
        near = np.flatnonzero(np.linalg.norm(body_points - joint, axis=1) < radius)
        if len(near) > REFINE_SAMPLES:
            near = near[np.linspace(0, len(near) - 1, REFINE_SAMPLES).astype(np.int64)]
        if len(near) >= 20:
            samples[part] = body_points[near]
    settled, last = set(), {}
    # The last round only checks the moves of the one before.
    for round_ in range(rounds + 1):
        solution = solve(joints, target, stretch=True)
        moved = deform(points, part_weights, solution)
        tree = BVHTree.FromPolygons(moved.tolist(), tris.tolist(), all_triangles=True)
        changed = dict(joints)
        for part, near in samples.items():
            if part in settled:
                continue
            pairs = []
            for point in near:
                location, normal, _index, _distance = tree.find_nearest(point, reach)
                if location is not None:
                    pairs.append((point, np.array(location), np.array(normal)))
            if len(pairs) < REFINE_COVERAGE * len(near):
                continue
            offset = _surface_offset(pairs)
            length = np.linalg.norm(offset)
            if part in last:
                before, previous = last.pop(part)
                if length > REFINE_SHRINK * previous:
                    # Moving the joint didn't take the offset away: it isn't the joint's. Put it back.
                    changed[part] = before
                    settled.add(part)
                    continue
            if round_ == rounds or length <= slack:
                continue
            # The model's point that ended up where its joint should be.
            goal = target[part] + offset * (length - slack) / length
            matrix = solution.transforms[part]
            last[part] = (joints[part], length)
            changed[part] = np.linalg.solve(matrix[:3, :3], goal - matrix[:3, 3])
        joints = changed
    corrected = [part for part in joints if np.linalg.norm(joints[part] - source[part]) > 1e-6]
    return joints, len(corrected)


# -----------------------------------------------------------------------------
# Blender objects

def _matrix(obj):
    return np.array(obj.matrix_world, dtype=np.float64)


def armature_of(obj):
    """The armature object bending mesh ``obj``: that of its first Armature modifier with one, or None."""
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE' and modifier.object is not None and modifier.object.type == 'ARMATURE':
            return modifier.object
    return None


def lost_armature(obj):
    """Name of the Armature modifier of mesh ``obj`` whose armature is gone (deleted after importing, like
    ``CH_P_EVE_InnerSuit_LOD0_Skeleton``), or None. Meshes sharing it came in as one model."""
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE':
            return modifier.name if modifier.object is None else None
    return None


def is_posed(armature):
    """Whether ``armature`` shows a pose (any bone moved from its rest position)."""
    if armature.data.pose_position != 'POSE' or armature.pose is None:
        return False
    rest = np.identity(4)
    return any(not np.allclose(np.array(bone.matrix_basis), rest, atol=1e-6) for bone in armature.pose.bones)


def _deform_bones(armature):
    names = [bone.name for bone in armature.data.bones if bone.use_deform]
    return names or [bone.name for bone in armature.data.bones]


def armature_rig(armature, depsgraph=None):
    """Parts of an armature's bones and joints of its parts. Returns (part of every bone, part of each
    deform bone, part -> joint in world space). The joints are the heads of the parts' main deform bones:
    as posed with a ``depsgraph``, else at rest."""
    bones = armature.data.bones
    parents = {bone.name: bone.parent.name if bone.parent is not None else None for bone in bones}
    parts = bone_parts([bone.name for bone in bones], parents)
    deform = {name: parts[name] for name in _deform_bones(armature)}
    matrix = _matrix(armature)
    pose = armature.evaluated_get(depsgraph).pose if depsgraph is not None else None
    joints = {}
    for part, name in main_bones(deform, parents).items():
        head = np.array(pose.bones[name].head if pose is not None else bones[name].head_local)
        joints[part] = matrix[:3, :3] @ head + matrix[:3, 3]
    return parts, deform, joints


def spread_out(joints):
    """Whether joints are at distinct places (an armature whose bones all sit at one spot has none)."""
    points = [joints[part] for part in KEY_PARTS + (NECK,) + tuple(part_name(kind, side) for side in SIDES
                                                                   for kind in ("forearm", "shin"))
              if part in joints]
    return len(points) >= 3 and np.ptp(np.array(points), axis=0).max() > 0.05


def _key_positions(block):
    co = np.empty(len(block.data) * 3, dtype=np.float32)
    block.data.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def mesh_positions(obj):
    """World positions of mesh ``obj``: as its shape keys show it (before modifiers), or its basis when
    they're absolute (`resizing.rest_shape` can't mix those, and Line Up moves every key the same way)."""
    keys = obj.data.shape_keys
    if keys is not None and not keys.use_relative:
        return apply_matrix(_matrix(obj), _key_positions(keys.reference_key))
    return apply_matrix(_matrix(obj), resizing.rest_shape(obj))


def mesh_weights(obj, names):
    """Weights of the vertices of mesh ``obj`` in the groups ``names`` (rows normalized)."""
    index = {name: k for k, name in enumerate(names)}
    local = [index.get(group.name, -1) for group in obj.vertex_groups]
    vertex, group, weight = [], [], []
    for vert in obj.data.vertices:
        for element in vert.groups:
            k = local[element.group] if element.group < len(local) else -1
            if k >= 0 and element.weight > 0.0:
                vertex.append(vert.index)
                group.append(k)
                weight.append(element.weight)
    return Weights(len(obj.data.vertices), names, vertex, group, weight)


def mesh_tris(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    return tris.reshape(-1, 3)


def body_rig(body, depsgraph):
    """Joints of the body (part -> world position), from its armature as posed, else estimated from its
    vertex groups' names and weights. Raises LineUpError when neither tells where its limbs are."""
    armature = armature_of(body)
    if armature is not None:
        _parts, deform, joints = armature_rig(armature, depsgraph)
        if has_key_parts(deform, 4) and spread_out(joints):
            return joints, armature
    names = [group.name for group in body.vertex_groups]
    weights = mesh_weights(body, names)
    parts, _named = group_parts_by_name(names, weights)
    if not has_key_parts(parts, 4):
        raise LineUpError("Can't tell the joints of '{:s}': it needs an armature, or vertex groups named after "
                          "the bones of its thighs and arms".format(body.name))
    points = mesh_positions(body)
    parts = split_finger_groups(parts, points, weights)
    return joints_from_weights(points, weights.by_part(parts)), None


def body_surface(body, depsgraph):
    """World positions of the body's vertices as displayed."""
    evaluated = body.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
    finally:
        evaluated.to_mesh_clear()
    return apply_matrix(_matrix(body), co.reshape(-1, 3))


class Plan:
    """How Line Up moves a model (``meshes`` and ``armature``, which can be None) onto a body: the
    ``solution``, each mesh's part weights (``mesh_parts``), each bone's part (``bone_parts``, for an
    armature whose rest pose moves along), where to put the bones of an armature whose bones all sit at
    one spot (``bone_places``: bone name -> world head and tail), and how the joints were found (``how``:
    "bones", "weights", "unnamed")."""

    def __init__(self):
        self.meshes = []
        self.armature = None
        self.move_armature = False
        self.mesh_parts = []
        self.bone_parts = {}
        self.bone_places = {}
        self.solution = None
        self.how = ""
        self.refined = 0
        self.source_joints = {}
        self.target_joints = {}


def plan(meshes, armature, body, depsgraph, stretch=True):
    """Work out how to line up a model (``meshes`` bent by ``armature``, which can be None) with
    ``body``. Raises LineUpError when it can't."""
    result = Plan()
    result.meshes, result.armature = list(meshes), armature
    target, _body_armature = body_rig(body, depsgraph)
    result.target_joints = target

    all_parts, deforming = {}, None
    bone_joints = {}
    if armature is not None:
        all_parts, deforming, bone_joints = armature_rig(armature)
    named_bones = deforming is not None and has_half(deforming)
    if named_bones:
        names = sorted({group.name for obj in meshes for group in obj.vertex_groups if group.name in deforming})
    else:
        names = sorted({group.name for obj in meshes for group in obj.vertex_groups})
    sets, positions, tris, first = [], [], [], 0
    for obj in meshes:
        sets.append(mesh_weights(obj, names))
        points = mesh_positions(obj)
        positions.append(points)
        tris.append(mesh_tris(obj) + first)
        first += len(points)
    weights = Weights.stack(names, sets)
    points = np.concatenate(positions) if positions else np.zeros((0, 3))
    tris = np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)

    if named_bones:
        group_parts = {name: deforming[name] for name in names}
        result.how = "bones"
        if not spread_out(bone_joints):
            # Bones at one spot (The Sims') may not hang from each other either: number each finger's segments
            # by where their groups lie instead.
            hands = {name: part_name("hand", DIGIT_PARTS[part][2]) if part in DIGIT_PARTS else part
                     for name, part in group_parts.items()}
            group_parts = split_finger_groups(hands, points, weights)
    else:
        group_parts, _named = group_parts_by_name(names, weights)
        result.how = "weights"
        if has_half(group_parts):
            group_parts = split_finger_groups(group_parts, points, weights)
        else:
            group_parts = label_anonymous(points, weights, target)
            result.how = "unnamed"
    part_weights = weights.by_part(group_parts)
    if named_bones and spread_out(bone_joints):
        source = bone_joints
        result.move_armature = True
        result.bone_parts = all_parts
    else:
        if result.how == "bones":
            result.how = "weights"  # named bones, all at one spot: joints from the weights
        source = joints_from_weights(points, part_weights)
        if not has_half({part: part for part in source}):
            raise LineUpError("Found too few joints in the model: it needs an armature, or vertex groups for "
                              "its thighs or arms")
        # Refining finds where the model's own joints are: it puts them on the body's, so with stretching
        # whatever the setting.
        try:
            source, result.refined = refine_joints(source, target, points, tris, part_weights,
                                                   body_surface(body, depsgraph))
        except LineUpError:
            pass
    result.source_joints = source
    result.solution = solve(source, target, stretch=stretch, tips=segment_tips(points, part_weights, source))
    if result.solution.half is not None and len(points) and not _half_fits(result.solution, source, target, points):
        raise LineUpError("Found only part of a body in the model, and it isn't shaped like a top or a bottom: "
                          "it needs both arms or both legs, and the body around them")
    offsets = np.cumsum([0] + [len(p) for p in positions])
    result.mesh_parts = [part_weights[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    if armature is not None and not result.move_armature and len(points):
        bones = armature.data.bones
        heads = apply_matrix(_matrix(armature), np.array([bone.head_local for bone in bones]).reshape(-1, 3))
        if not len(heads) or np.ptp(heads, axis=0).max() <= BONE_SPOT * np.ptp(points, axis=0).max():
            # Its bones sit at one spot: they go where their groups end up.
            places = group_bones(deform(points, part_weights, result.solution), weights, group_parts,
                                 BONE_LEAST * result.solution.torso_length)
            result.bone_places = {name: place for name, place in places.items() if name in bones}
        else:
            # Bones in places of their own, but named so they say nothing: each moves with its group's part,
            # a bone without a group with the nearest bone above it that has one.
            def part_of(bone):
                while bone is not None and bone.name not in group_parts:
                    bone = bone.parent
                return group_parts[bone.name] if bone is not None else TORSO

            result.bone_parts = {bone.name: part_of(bone) for bone in bones}
            result.move_armature = True
    return result


def _half_fits(solution, source, target, points):
    """Whether a model lined up as half a body (``solution.half``) is shaped like one, as sized: a bottom
    reaches up past its hips, but not to a head; a top hangs from its shoulders no further than a torso and
    a quarter (arms in an A-pose, a long top). Pieces cut out of a garment (a skirt starting at the knees, a
    bodice whose hanging bits looked like legs) aren't: sized by limbs they don't have, they would come out
    far too big or small."""
    hips, top = torso_ends(source, target)
    torso = solution.torso_length
    if solution.half == "lower":
        above = (points[:, 2].max() - hips[2]) * solution.scale
        return HALF_ABOVE[0] * torso <= above <= HALF_ABOVE[1] * torso
    below = (top[2] - points[:, 2].min()) * solution.scale
    return below <= HALF_BELOW * torso


def group_bones(points, weights, parts, least):
    """Rest positions (group name -> (head, tail)) for bones named after the vertex groups of ``points``
    (``weights``; ``parts``: group name -> part), for an armature whose bones say nothing about where they
    are (all at one spot, like The Sims' exports). Each bone starts where its group's weight hands over
    from the group it shares the most with toward the torso (along the tree of what the groups share, see
    `_GroupTree`), and ends where the heaviest group hanging from it starts, or as far past its group's
    middle. Bones are at least ``least`` long."""
    tree = _GroupTree(points, weights)
    torso = [group for group, name in enumerate(weights.names) if parts.get(name) == TORSO and tree.alive[group]]
    if not torso:
        return {}
    tree.grow(max(torso, key=lambda group: tree.mass[group]))
    places = {}
    for group, name in enumerate(weights.names):
        if not tree.alive[group]:
            continue
        parent = tree.parent[group]
        head = tree.handover(parent, group) if parent >= 0 else tree.centroids[group]
        children = [child for child in tree.children[group] if tree.alive[child]]
        if children:
            tail = tree.handover(group, max(children, key=lambda child: tree.mass[child]))
        else:
            tail = 2.0 * tree.centroids[group] - head
        if np.linalg.norm(tail - head) < least:
            direction = _unit(tail - head) if np.linalg.norm(tail - head) > 1e-12 else np.array((0.0, 0.0, 1.0))
            tail = head + least * direction
        places[name] = (head, tail)
    return places


def place_bones(result):
    """Put the bones of the Plan's armature named after its vertex groups where the groups lie as lined up
    (``bone_places``, see `group_bones`). At rest, where they are doesn't change the meshes. The armature
    must be in Edit Mode."""
    inverse = np.linalg.inv(_matrix(result.armature))
    edit_bones = result.armature.data.edit_bones
    for name, (head, tail) in result.bone_places.items():
        bone = edit_bones.get(name)
        if bone is not None:
            bone.use_connect = False
            bone.head, bone.tail = apply_matrix(inverse, head).tolist(), apply_matrix(inverse, tail).tolist()


def segment_tips(points, part_weights, joints):
    """Where a model's segments end that it lacks the next part of (part -> point), which its joints
    (``joints``) alone can't tell: the last segments of its fingers, or a shin without a foot. Each points
    at the middle of the points weighted to it, and reaches as far along as nearly all of them."""
    tips = {}
    for part in PARTS:
        if part not in joints or (part not in NEXT and part not in DIGIT_PARTS) or NEXT.get(part) in joints:
            continue
        column = part_weights[:, PART_INDEX[part]]
        total = column.sum()
        if total <= 0.0:
            continue
        direction = (points * column[:, None]).sum(axis=0) / total - joints[part]
        length = np.linalg.norm(direction)
        if length <= 1e-12:
            continue
        direction /= length
        own = column >= 0.5
        reach = np.percentile((points[own] - joints[part]) @ direction, 95) if own.any() else length
        tips[part] = joints[part] + direction * max(reach, length)
    return tips


def _write_mesh(obj, part_weights, solution):
    """Move mesh ``obj`` (every shape key) by the parts' moves, blended by its ``part_weights``."""
    mesh = obj.data
    matrix = _matrix(obj)
    inverse = np.linalg.inv(matrix)

    def moved(local):
        return apply_matrix(inverse, deform(apply_matrix(matrix, local), part_weights, solution))

    keys = mesh.shape_keys
    if keys is None:
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        mesh.vertices.foreach_set("co", moved(co.reshape(-1, 3).astype(np.float64)).astype(np.float32).ravel())
    else:
        for block in keys.key_blocks:
            result = moved(_key_positions(block)).astype(np.float32).ravel()
            block.data.foreach_set("co", result)
            if block == keys.reference_key:
                mesh.vertices.foreach_set("co", result)
    mesh.update()


def move_rest_pose(result):
    """Move the rest pose of the Plan's armature along with its parts: every bone's head and tail by its
    part's move, its roll turned with it. The armature must be in Edit Mode."""
    armature, solution = result.armature, result.solution
    matrix = _matrix(armature)
    inverse = np.linalg.inv(matrix)
    axes = matrix[:3, :3]
    edit_bones = armature.data.edit_bones
    connected = {bone.name for bone in edit_bones if bone.use_connect}
    for bone in edit_bones:
        bone.use_connect = False
    for bone in edit_bones:
        part = result.bone_parts.get(bone.name, TORSO)
        head, tail = (apply_matrix(inverse, solution.move(part, apply_matrix(matrix, np.array(point))))
                      for point in (bone.head, bone.tail))
        z_axis = np.linalg.solve(axes, solution.rotations[part] @ (axes @ np.array(bone.z_axis)))
        bone.head, bone.tail = head.tolist(), tail.tolist()
        bone.align_roll(z_axis.tolist())
    for bone in edit_bones:
        if bone.name in connected:
            bone.use_connect = True


def write_meshes(result):
    """Move the Plan's meshes (every shape key) onto the body."""
    for obj, part_weights in zip(result.meshes, result.mesh_parts):
        _write_mesh(obj, part_weights, result.solution)
