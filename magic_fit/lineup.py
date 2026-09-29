# SPDX-License-Identifier: GPL-3.0-or-later

"""Lining up an imported model with a body.

A body or outfit ripped from another game, or made for VRChat or The Sims, stands in its own pose and
proportions: arms out in a T-pose or hanging down, longer legs, another height, sometimes facing another
way. Lining it up with the body (the devkit's Mannequin) gives refitting a place to start: the same size
and pose, with its joints on the body's joints.

Both skeletons are reduced to the same parts: the torso (pelvis, spine, chest, breasts...), the neck and
head, and per side the clavicle, upper arm, forearm, hand, thigh, shin and foot. Bones are sorted into
parts by their names, as the common naming schemes spell them (Unity and VRChat, Mixamo, Unreal, 3ds Max
Biped, Rigify, DAZ, MMD, The Sims, FFXIV...); bones whose names say nothing (jiggle, skirt, tail bones...)
join their parent's part. A part's joint is the head of its main bone, the one the next part hangs from.
A model without a usable armature is sorted by the names of its vertex groups instead, and its joints are
estimated from its weights, where one part's weight hands over to the next. When the names say nothing at
all (bone_00, bone_01...), the groups are sorted by how they overlap and where they lie.

Then, from the torso out:

- The torso turns so its hips and shoulders face the body's way, scales so its shoulders are as high
  above its hips as the body's are, and moves onto the body's hips.
- Each limb segment turns about its joint, the least it takes to point the way the body's does, and
  stretches along its length (not in girth) so its end lands on the body's next joint. Without
  stretching, it keeps its length (scaled with the torso) and only turns.
- Hands follow the forearms. Feet and the head keep their angle to the torso, so heels stay heels and
  head pieces stay upright.

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

from . import resizing

TORSO, NECK, HEAD = "torso", "neck", "head"
SIDES = ("l", "r")
ARM = ("clavicle", "upperarm", "forearm", "hand")
LEG = ("thigh", "shin", "foot")


def part_name(kind, side=None):
    return kind if side is None else "{:s}_{:s}".format(kind, side)


PARTS = (TORSO, NECK, HEAD) + tuple(part_name(kind, side) for side in SIDES for kind in ARM + LEG)
PART_INDEX = {part: index for index, part in enumerate(PARTS)}
# The part each part hangs from.
PARENT = {NECK: TORSO, HEAD: NECK}
for _side in SIDES:
    for _chain in (ARM, LEG):
        _names = (TORSO,) + tuple(part_name(kind, _side) for kind in _chain)
        PARENT.update(zip(_names[1:], _names[:-1]))
# Segments reaching from their joint to the next part's; the others (hands, feet, head) end in themselves.
NEXT = {NECK: HEAD}
for _side in SIDES:
    for _a, _b in (("clavicle", "upperarm"), ("upperarm", "forearm"), ("forearm", "hand"), ("thigh", "shin"),
                   ("shin", "foot")):
        NEXT[part_name(_a, _side)] = part_name(_b, _side)
# Parts that keep their angle to the torso instead of turning with the part they hang from.
UPRIGHT = {HEAD, "foot_l", "foot_r"}
# Parts a model needs, by name or weights, to be lined up by its bones or groups.
KEY_PARTS = ("thigh_l", "thigh_r", "upperarm_l", "upperarm_r")

# A limb segment stretches at most by this factor (or shrinks by its inverse) to reach the next joint.
MAX_STRETCH = 2.0
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
                  "brochi", "deltoid", "humerus", "shldr", "shldrbend", "shldrtwist")),
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
    jp_side = next((side for mark, side in _JP_SIDES if mark in name), None)
    for mark, kind in _JP_KINDS:
        if mark in name:
            return kind, jp_side
    tokens = words(name)
    if not tokens:
        return None, jp_side
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
    return parts


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


def _best_cuts(positions, wanted):
    """Increasing indices into ``positions``, one per ``wanted`` value (increasing), matching them best
    (least squares)."""
    best = None

    def search(start, chosen, cost):
        nonlocal best
        k = len(chosen)
        if k == len(wanted):
            if best is None or cost < best[0]:
                best = (cost, list(chosen))
            return
        for i in range(start, len(positions)):
            search(i + 1, chosen + [i], cost + (positions[i] - wanted[k]) ** 2)

    search(0, [], 0.0)
    return None if best is None else best[1]


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


def label_anonymous(points, weights, target_joints):
    """Part of each vertex group when their names say nothing (bone_00, bone_01...), from how the groups
    overlap and where they lie. The model is taken to stand upright (+Z) with its left on +X, or on +Y
    when it spreads along Y; feet pointing the other way flip that. ``target_joints`` (the body's, with
    all limb joints) lend their proportions to find joints along each limb. Returns group name -> part."""
    tree = _GroupTree(points, weights)
    names, alive, centroids, mass = tree.names, tree.alive, tree.centroids, tree.mass
    if alive.sum() < 3:
        return {name: TORSO for name in names}

    # Sideways: the horizontal direction the groups spread the most along (arms, shoulders, hips).
    flat = centroids[alive][:, :2]
    middle = np.average(flat, axis=0, weights=mass[alive])
    spread = np.cov((flat - middle).T, aweights=mass[alive])
    lateral = np.linalg.eigh(spread)[1][:, -1]
    lateral = lateral * (np.sign(lateral[0]) if abs(lateral[0]) >= abs(lateral[1]) else np.sign(lateral[1]))
    across = (centroids[:, :2] - middle) @ lateral
    heights = centroids[:, 2]
    height = max(np.ptp(heights[alive]), 1e-6)
    central = alive & (np.abs(across) < 0.03 * height)
    if not central.any():
        central = alive & (np.abs(across) <= np.abs(across[alive]).min() + 1e-9)
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
    trunk_height = np.average(heights[central], weights=mass[central])
    limbs, bits = [], []
    for members in components:
        root = next((g for g in members if tree.parent[g] < 0 or central[tree.parent[g]]), members[0])
        attach = tree.parent[root]
        base = centroids[attach] if attach >= 0 else centroids[root]
        extent = max(np.linalg.norm(centroids[g] - base) for g in members)
        side = "l" if np.average(across[members], weights=mass[members]) > 0.0 else "r"
        lowest = min(heights[g] for g in members)
        (limbs if extent >= 0.2 * height else bits).append((members, root, attach, side, base[2], lowest))
    # Per side, the leg is the limb reaching lowest, the arm the other one hanging from highest up.
    arms, legs = {}, {}
    for side in SIDES:
        mine = [limb for limb in limbs if limb[3] == side]
        if len(mine) >= 2:
            legs[side] = min(mine, key=lambda limb: limb[5])
            arms[side] = max((limb for limb in mine if limb is not legs[side]), key=lambda limb: limb[4])
        elif len(mine) == 1:
            (arms if mine[0][4] > trunk_height else legs)[side] = mine[0]

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
    for side, (members, root, attach, _side, _base, _lowest) in arms.items():
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
            cuts = _best_cuts(along[:-1] / max(along[-1], 1e-12), list(lengths[1:3] / lengths[3]))
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

    # The model's size from the arms (shoulder to wrist), else from the legs themselves.
    scale = None
    if spans:
        scale = float(np.mean([model / max(body, 1e-12) for model, body in spans]))
    for side, (members, root, attach, _side, _base, _lowest) in legs.items():
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
    for members, _root, attach, _side, _base, _lowest in bits:
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


def affine(pivot, target, rotation, scale, direction=None, stretch=1.0):
    """4x4 matrix of x -> target + scale * rotation (I + (stretch - 1) d d^T) (x - pivot): stretched
    along ``direction`` d, turned, scaled and moved from ``pivot`` to ``target``."""
    linear = scale * rotation
    if direction is not None and stretch != 1.0:
        linear = linear @ (np.eye(3) + (stretch - 1.0) * np.outer(direction, direction))
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
    """How each part moves (``transforms``: part -> 4x4 world matrix; ``rotations``: part -> 3x3), and
    what it took: the ``scale`` of the whole model, the ``stretches`` of limb segments, how far the torso
    turned about the vertical (``turn``, degrees), and the parts only one side has (``missing``)."""

    def __init__(self):
        self.transforms = {}
        self.rotations = {}
        self.stretches = {}
        self.scale = 1.0
        self.turn = 0.0
        self.missing = []
        self.torso_length = 0.0


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


def solve(source, target, stretch=True):
    """How to move each part of a model whose joints are ``source`` (part -> world position) so it lines
    up with a body whose joints are ``target``. Raises LineUpError without hips or shoulders."""
    result = Solution()
    ends = (torso_ends(source, target), torso_ends(target, source))
    for (hips, top), what in zip(ends, ("model", "body")):
        if hips is None:
            raise LineUpError("Found no hips (thigh bones or groups) in the {:s}".format(what))
        if top is None:
            raise LineUpError("Found no shoulders or neck in the {:s}".format(what))
    (hips_s, top_s), (hips_t, top_t) = ends
    frames = []
    for joints, hips, top in ((source, hips_s, top_s), (target, hips_t, top_t)):
        lateral = _unit(joints["thigh_l"] - joints["thigh_r"])
        for a, b in (("upperarm_l", "upperarm_r"), ("clavicle_l", "clavicle_r")):
            if all(part in source and part in target for part in (a, b)):
                lateral = lateral + _unit(joints[a] - joints[b])
                break
        frames.append(_frame(top - hips, lateral))
    torso_rotation = frames[1] @ frames[0].T
    scale = np.linalg.norm(top_t - hips_t) / max(np.linalg.norm(top_s - hips_s), 1e-12)
    result.scale = float(scale)
    result.torso_length = float(np.linalg.norm(top_t - hips_t))
    forward = frames[0][:, 2]
    turned = torso_rotation @ forward
    result.turn = float(np.degrees(np.arctan2(forward[0] * turned[1] - forward[1] * turned[0],
                                              forward[0] * turned[0] + forward[1] * turned[1])))
    transforms, rotations = result.transforms, result.rotations
    transforms[TORSO] = affine(hips_s, hips_t, torso_rotation, scale)
    rotations[TORSO] = torso_rotation

    for part in PARTS[1:]:
        parent = PARENT[part]
        if part not in source or part not in target:
            if part in source or part in target:
                result.missing.append(part)
            # Missing on either side: it moves with the part above it.
            transforms[part] = transforms[parent]
            rotations[part] = rotations[parent]
            continue
        pivot = source[part]
        goal = target[part] if stretch else apply_matrix(transforms[parent], pivot)
        after = NEXT.get(part)
        if part in UPRIGHT:
            rotation = torso_rotation
            transforms[part] = affine(pivot, goal, rotation, scale)
        elif after is not None and after in source and after in target:
            direction = _unit(source[after] - pivot)
            wanted = _unit(target[after] - target[part])
            rotation = swing(rotations[parent] @ direction, wanted) @ rotations[parent]
            factor = 1.0
            if stretch:
                factor = np.linalg.norm(target[after] - target[part]) / max(
                    scale * np.linalg.norm(source[after] - pivot), 1e-12)
                factor = float(np.clip(factor, 1.0 / MAX_STRETCH, MAX_STRETCH))
                result.stretches[part] = factor
            transforms[part] = affine(pivot, goal, rotation, scale, direction, factor)
        else:
            rotation = rotations[parent]
            transforms[part] = affine(pivot, goal, rotation, scale)
        rotations[part] = rotation
    return result


def deform(points, part_weights, transforms):
    """``points`` (world) moved by their parts' transforms, blended by ``part_weights`` (points x PARTS;
    unweighted points move with the torso)."""
    weights = part_weights.copy()
    weights[weights.sum(axis=1) <= 0.0, PART_INDEX[TORSO]] = 1.0
    result = np.zeros_like(points)
    for index, part in enumerate(PARTS):
        column = weights[:, index]
        used = column > 0.0
        if used.any():
            result[used] += column[used, None] * apply_matrix(transforms[part], points[used])
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
        if part in (TORSO, HEAD) or part not in joints:
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
        moved = deform(points, part_weights, solution.transforms)
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
    points = [joints[part] for part in KEY_PARTS + (NECK,) if part in joints]
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
    return joints_from_weights(mesh_positions(body), weights.by_part(parts)), None


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
    armature whose rest pose moves along), and how the joints were found (``how``: "bones", "weights",
    "unnamed")."""

    def __init__(self):
        self.meshes = []
        self.armature = None
        self.move_armature = False
        self.mesh_parts = []
        self.bone_parts = {}
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

    all_parts, deform = {}, None
    bone_joints = {}
    if armature is not None:
        all_parts, deform, bone_joints = armature_rig(armature)
    named_bones = deform is not None and has_key_parts(deform)
    if named_bones:
        names = sorted({group.name for obj in meshes for group in obj.vertex_groups if group.name in deform})
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
        group_parts = {name: deform[name] for name in names}
        result.how = "bones"
    else:
        group_parts, _named = group_parts_by_name(names, weights)
        result.how = "weights"
        if not has_key_parts(group_parts):
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
        if not has_key_parts({part: part for part in source}):
            raise LineUpError("Found too few joints in the model: it needs an armature, or vertex groups for "
                              "its thighs and arms")
        # Refining finds where the model's own joints are: it puts them on the body's, so with stretching
        # whatever the setting.
        try:
            source, result.refined = refine_joints(source, target, points, tris, part_weights,
                                                   body_surface(body, depsgraph))
        except LineUpError:
            pass
    result.source_joints = source
    result.solution = solve(source, target, stretch=stretch)
    offsets = np.cumsum([0] + [len(p) for p in positions])
    result.mesh_parts = [part_weights[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    return result


def _write_mesh(obj, part_weights, transforms):
    """Move mesh ``obj`` (every shape key) by the parts' transforms, blended by its ``part_weights``."""
    mesh = obj.data
    matrix = _matrix(obj)
    inverse = np.linalg.inv(matrix)

    def moved(local):
        return apply_matrix(inverse, deform(apply_matrix(matrix, local), part_weights, transforms))

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
        transform = solution.transforms[part]
        head, tail = (apply_matrix(inverse, apply_matrix(transform, apply_matrix(matrix, np.array(point))))
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
        _write_mesh(obj, part_weights, result.solution.transforms)
