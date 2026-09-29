# SPDX-License-Identifier: GPL-3.0-or-later

"""What Customize+ does to a skeleton in the game, and the values that make Blender do the same.

In the game (Customize+ 2.x, `ModelBone.ApplyModelTransform`), each bone's transform in model space is
edited after the animation has posed the skeleton, bone by bone, parents first:
    scale     S' = S * Scaling                     (along the bone's own axes)
    rotation  R' = R * q                           (q turns about the bone's own axes)
    position  T' = T + R' * Translation            (along the bone's turned axes, not scaled)
where q = qY(Rotation.X) qX(Rotation.Y) qZ(Rotation.Z) (`Quaternion.CreateFromYawPitchRoll`, degrees): the
X value turns about the bone's Y axis and Y about its X axis. Only that bone's own transform changes, so
its children stay where the animation put them, unless the bone propagates. Then every bone below it
(`PropagateChildren`) moves along by the bone's change of position (always), turns with it about the
bone's new position (Propagate Rotation) and scales about that position along the model's axes, not
the bone's (Propagate Scale, with Child Scaling when that's Independent). Havok then keeps only position,
turn and scale of each bone, dropping any shear. The root bone (n_root) is left out: its scale scales the
whole character, its position moves it.

In Blender, `plan` turns a template into per-bone values for a copy of the armature (see `rig`) that
follows the original bone by bone and adds, as constraints with fixed values:
  - propagation, before the bones' poses, in the propagating bone's frame (a scale in the armature's
    axes at its head), innermost propagating bone first: exact in any pose, except that the game drops
    the shear a propagated uneven scale gives the bones below pose by pose, the rig as it is at rest.
  - the bone's own change, after its pose, in its own frame: exact in any pose. It also takes whatever
    else the game does to the bone at rest, so the rig matches the game exactly there.
  - the root's scale and position, last, around the armature's origin.
Blender bones don't always keep the game's bone axes; `detect_axes` finds out, bone by bone, how an
armature's axes were swapped, by comparing it with the game's skeleton (`reference`).

The game's Bust Size slider (see `bust_scale`) is one more change in the bones' own frames, after the
template's: the game multiplies the scale of the breast bones and of every bone below them by it.
"""

import math
from collections import Counter

import numpy as np

from . import reference
from .template import ROOT_BONE

# Game model space (Y up, facing +Z) to Blender's usual armature space (Z up, facing -Y).
TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
# A bone whose axes are within this of a swap of the game's axes (as the armature's race turns the bone)
# has them swapped exactly; otherwise it was turned freely (automatic bone orientation) and the game's
# turn is used as is.
SNAP_DEGREES = 20.0
MIN_SCALE = 1e-4


class Options:
    """Which parts of the template to apply."""

    def __init__(self, scale=True, rotation=True, translation=True):
        self.scale = scale
        self.rotation = rotation
        self.translation = translation

    def to_dict(self):
        return {"scale": self.scale, "rotation": self.rotation, "translation": self.translation}

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: bool(data.get(key, True)) for key in ("scale", "rotation", "translation")})


# -----------------------------------------------------------------------------
# Rotations and transforms

def axis_rotation(axis, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def cplus_rotation(rotation):
    """Rotation matrix of a Customize+ rotation (X, Y, Z in degrees): CreateFromYawPitchRoll(X, Y, Z)."""
    x, y, z = rotation
    return axis_rotation(1, x) @ axis_rotation(0, y) @ axis_rotation(2, z)


def quaternion(R):
    """(w, x, y, z) of rotation matrix ``R``."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def quaternion_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def compose(t, R, s):
    """4x4 matrix that scales by ``s``, turns by ``R`` and moves by ``t`` (in that order)."""
    M = np.identity(4)
    M[:3, :3] = R * np.asarray(s, dtype=float)
    M[:3, 3] = t
    return M


def orthonormal(M):
    """The rotation closest to 3x3 ``M`` (which must not mirror)."""
    U, _S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def decompose(M, flip=None):
    """(position, rotation, scale) of 4x4 ``M``, dropping shear like Havok does: the rotation is the
    closest one, the scale what's left along its axes. A mirroring matrix gets a negative scale on axis
    ``flip`` (default: the one that stretches least)."""
    L = M[:3, :3]
    if np.linalg.det(L) < 0.0:
        axis = int(np.argmin(np.linalg.norm(L, axis=0))) if flip is None else flip
        L = L.copy()
        L[:, axis] *= -1.0
        R = orthonormal(L)
        s = np.diag(R.T @ L).copy()
        s[axis] *= -1.0
        return M[:3, 3].copy(), R, s
    R = orthonormal(L)
    return M[:3, 3].copy(), R, np.diag(R.T @ L).copy()


def angle_between(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def snap(R):
    """The swap of axes (signed permutation) nearest to rotation ``R``, or None if it mirrors."""
    P = np.zeros((3, 3))
    rows, cols = set(), set()
    for _ in range(3):
        best = None
        for i in range(3):
            for j in range(3):
                if i not in rows and j not in cols and (best is None or abs(R[i, j]) > abs(R[best])):
                    best = (i, j)
        P[best] = 1.0 if R[best] >= 0.0 else -1.0
        rows.add(best[0])
        cols.add(best[1])
    return P if np.linalg.det(P) > 0.0 else None


def is_swap(R):
    return np.allclose(np.abs(R), np.abs(R).round()) and np.allclose(np.abs(R).sum(axis=0), 1.0)


# -----------------------------------------------------------------------------
# Bone axes

class AxisMap:
    """How an armature relates to the game's skeleton: armature space = ``turn`` x game model space x
    ``units``, and each bone's axes = the game bone's axes x ``frames[bone]`` (a swap of axes, or for a
    freely turned bone any rotation). Bones in ``unplaced`` have no real rest pose (see `unplaced`)."""

    def __init__(self, turn, units, frames, default, known, swapped, free, unplaced=frozenset(), race=None,
                 size=None):
        self.turn = turn
        self.units = units
        self.frames = frames
        self.default = default
        self.known = known        # bones found in the game's skeleton
        self.swapped = swapped    # of those, bones with swapped axes
        self.free = free          # of those, bones turned freely
        self.unplaced = unplaced
        self.race = race          # the race whose skeleton fits best (reference.NAMES), None for the table's own
        self.size = size          # the skeleton's size against the table's (the Midlander female's), or None
        # Armature space from model space: turned, with positions in the armature's units.
        self._outer = np.identity(4)
        self._outer[:3, :3] = turn * units
        self._outer_inv = np.linalg.inv(self._outer)
        self._units = np.diag([units, units, units, 1.0])
        self._units_inv = np.diag([1.0 / units, 1.0 / units, 1.0 / units, 1.0])

    def frame(self, name):
        return self.frames.get(name, self.default)

    def _frame4(self, name):
        F = np.identity(4)
        F[:3, :3] = self.frame(name)
        return F

    def to_game(self, name, matrix):
        """A bone's 4x4 armature-space matrix as the game's model-space transform of that bone."""
        return self._outer_inv @ matrix @ self._frame4(name).T @ self._units

    def to_blender(self, name, matrix):
        """Inverse of `to_game`."""
        return self._outer @ matrix @ self._units_inv @ self._frame4(name)

    def change_to_blender(self, change):
        """A change made in model space (applied before a bone's transform) in armature space."""
        return self._outer @ change @ self._outer_inv


# Bones that belong at the model's origin.
AT_ORIGIN = frozenset(("n_root", "n_throw"))


def unplaced(rest):
    """Bones with no real rest pose: left at the armature's origin, as armatures made from a model's bone
    list alone are. Changing them would scale and turn their vertices about the origin. A game bone
    counts when the game puts it elsewhere; any other bone when others are stacked there with it."""
    table = reference.BONES
    at_origin = [name for name, matrix in rest.items()
                 if name not in AT_ORIGIN and float(np.abs(matrix[:3, 3]).max()) <= 1e-6]
    result = set()
    for name in at_origin:
        values = table.get(name)
        if (float(np.linalg.norm(values[:3])) > 0.05) if values is not None else len(at_origin) > 1:
            result.add(name)
    return frozenset(result)


def _fit_turn(A, B):
    """(turn, units, size) taking game model-space positions ``A`` to armature-space positions ``B``: the
    units are a power of ten, the size what's left of the scale (None when the positions don't tell)."""
    A0, B0 = A - A.mean(axis=0), B - B.mean(axis=0)
    spread = float((A0 ** 2).sum())
    if spread < 1e-8 or float((B0 ** 2).sum()) < 1e-12:
        return TO_BLENDER, 1.0, None
    ratio = math.sqrt(float((B0 ** 2).sum()) / spread)
    units = 10.0 ** round(math.log10(ratio))
    U, _S, Vt = np.linalg.svd(A0.T @ B0)
    d = 1.0 if np.linalg.det(Vt.T @ U.T) > 0.0 else -1.0
    fitted = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    _U, S, Vt = np.linalg.svd(A0)
    if S[1] < 0.1 * S[0]:
        # Nearly on a line (a spine and a head, say): the turn about that line is unknown. Take the
        # usual turn that puts the line where the fit does.
        line = Vt[0]
        for usual in (TO_BLENDER, np.identity(3)):
            if float(np.dot(usual @ line, fitted @ line)) > math.cos(math.radians(30.0)):
                return usual, units, ratio / units
    return snap(fitted) if snap(fitted) is not None else TO_BLENDER, units, ratio / units


def detect_axes(rest):
    """`AxisMap` of an armature from its bones' rest matrices ({name: 4x4 in armature space})."""
    table = reference.BONES
    missing = unplaced(rest)
    known = [name for name in rest if name in table and name not in missing]
    turn, units, size = TO_BLENDER, 1.0, None
    if len(known) >= 3:
        turn, units, size = _fit_turn(np.array([table[name][:3] for name in known]),
                                      np.array([rest[name][:3, 3] for name in known]))

    # For each bone and each way the races turn it: the nearest swap of axes and how far off it is.
    fits = {}
    for name in known:
        B = orthonormal(rest[name][:3, :3])
        fits[name] = []
        for q in (table[name][3:],) + tuple(reference.TURNS.get(name, ())):
            correction = (turn @ quaternion_matrix(q)).T @ B
            P = snap(correction)
            fits[name].append((angle_between(P.T @ correction) if P is not None else 180.0, P, correction))

    def cost(race):
        turns = reference.RACES.get(race, {})
        # A bone turned freely counts no more than 45 degrees, so it can't outweigh the others.
        return sum(min(fits[name][turns.get(name, 0)][0], 45.0) for name in known)

    race = min((None,) + tuple(reference.RACES), key=lambda race: (cost(race), race is not None))
    turns = reference.RACES.get(race, {})
    frames, snapped = {}, Counter()
    swapped = free = 0
    misfits = []
    for name in known:
        residual, P, correction = fits[name][turns.get(name, 0)]
        if P is not None and residual < SNAP_DEGREES:
            frames[name] = P
            snapped[tuple(P.astype(int).flatten())] += 1
            if not np.allclose(P, np.identity(3)):
                swapped += 1
        elif name in reference.SAMPLED:
            misfits.append(name)
        else:
            frames[name] = orthonormal(correction)
            free += 1
    default = np.array(snapped.most_common(1)[0][0], dtype=float).reshape(3, 3) if snapped else np.identity(3)
    # A bone known from one mod skeleton only that doesn't fit is more likely turned differently in this
    # armature's skeleton than turned by its importer: it gets the armature's usual axes, like an
    # unknown bone.
    for name in misfits:
        frames[name] = default
    return AxisMap(turn, units, frames, default, len(known), swapped, free, missing, race, size)


# -----------------------------------------------------------------------------
# Customize+ in the game

def own_change(edit, options):
    """A bone's own change in its frame: (position, rotation, scale) of T(q t) R(q) S(s)."""
    q = cplus_rotation(edit.rotation) if options.rotation else np.identity(3)
    t = np.array(edit.translation, dtype=float) if options.translation else np.zeros(3)
    s = np.array(edit.scaling, dtype=float) if options.scale else np.ones(3)
    s = np.where(np.abs(s) < MIN_SCALE, np.copysign(MIN_SCALE, s), s)
    return q @ t, q, s


def root_change(edit, options):
    """The character's move and scale from the root bone's edit, as a 4x4 model-space change, or None."""
    if edit is None:
        return None
    s = np.ones(3)
    # `ModelBone.IsModifiedScale`: any axis neither 0 nor 1.
    if options.scale and any(v not in (0.0, 1.0) for v in edit.scaling):
        s = np.array(edit.scaling, dtype=float)
        s = np.where(np.abs(s) < MIN_SCALE, np.copysign(MIN_SCALE, s), s)
    t = np.array(edit.translation, dtype=float) if options.translation else np.zeros(3)
    if np.allclose(s, 1.0) and np.allclose(t, 0.0):
        return None
    return compose(t, np.identity(3), s)


class Carry:
    """What a propagating bone passes on to the bones below it: they move with its position change
    (``move``, in its frame), turn with it (``turn``) and scale along the model's axes (``scale``, or
    None), both about its new position. ``rotation`` is the bone's turn in model space before its own
    change: it tells where the model's axes are in the bone's frame."""

    def __init__(self, bone, move, turn, scale, rotation, below):
        self.bone = bone
        self.move = move
        self.turn = turn
        self.scale = scale
        self.rotation = rotation
        self.below = below


class Simulation:
    def __init__(self, model, carries, root):
        self.model = model        # {bone: 4x4} model-space transforms after Customize+ (without the root)
        self.carries = carries    # [Carry], in order
        self.root = root          # 4x4 model-space change of the whole character, or None


def hierarchy_order(parents):
    """Bones with every parent before its children."""
    children = {}
    for name, parent in parents.items():
        children.setdefault(parent if parent in parents else None, []).append(name)
    order, stack = [], list(reversed(children.get(None, [])))
    while stack:
        name = stack.pop()
        order.append(name)
        stack.extend(reversed(children.get(name, [])))
    return order


def descendants(name, children):
    result, stack = [], list(children.get(name, []))
    while stack:
        child = stack.pop(0)
        result.append(child)
        stack.extend(children.get(child, []))
    return result


def simulate(parents, model, edits, options):
    """Customize+ on a skeleton: ``parents`` {bone: parent or None}, ``model`` {bone: 4x4 model-space
    transform without shear}, ``edits`` {bone: BoneEdit}."""
    order = hierarchy_order(parents)
    children = {}
    for name in order:
        children.setdefault(parents[name], []).append(name)
    current = {name: decompose(model[name]) for name in order}
    carries = []
    for name in order:
        edit = edits.get(name)
        if name == ROOT_BONE or edit is None or not edit.is_edited():
            continue
        t, R, s = current[name]
        local_t, q, local_s = own_change(edit, options)
        R2 = R @ q
        s2 = s * local_s
        t2 = t + R @ local_t  # R' Translation, with local_t = q Translation
        current[name] = (t2, R2, s2)
        if not edit.propagates:
            continue
        if edit.child_scaling_independent and options.scale:
            # Kept off 0 like the own scale: the rig's changes must be invertible.
            scaling = np.array(edit.child_scaling, dtype=float)
            child_scale = s * np.where(np.abs(scaling) < MIN_SCALE, np.copysign(MIN_SCALE, scaling), scaling)
        else:
            child_scale = s2
        turn = options.rotation and edit.propagate_rotation and any(v != 0.0 for v in edit.rotation)
        grow = options.scale and edit.propagate_scale and (
            any(v != 1.0 for v in edit.scaling)
            or (edit.child_scaling_independent and any(v != 1.0 for v in edit.child_scaling)))
        delta_turn = R2 @ R.T
        delta_position = t2 - t
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_scale = np.where(np.abs(s) > 1e-12, child_scale / s, 1.0)
        below_bones = descendants(name, children)
        carries.append(Carry(name, local_t, q if turn else np.identity(3), delta_scale if grow else None, R,
                             below_bones))
        for below in below_bones:
            tb, Rb, sb = current[below]
            offset = tb - t2
            L = Rb * sb
            if grow:
                L = np.diag(delta_scale) @ L
                offset = delta_scale * offset
            if turn:
                L = delta_turn @ L
                offset = delta_turn @ offset
            M = np.identity(4)
            M[:3, :3] = L
            M[:3, 3] = delta_position + t2 + offset
            current[below] = decompose(M)
    result = {name: compose(*current[name]) for name in order}
    return Simulation(result, carries, root_change(edits.get(ROOT_BONE), options))


# -----------------------------------------------------------------------------
# The game's bust size
#
# The character creator's Bust Size slider (customize byte 23, 0 to 100) scales the breast bones. The game
# (version 2026.09.15, read in its code) works it out when it sets up a character (`RspSetupCharacter`,
# the function Penumbra hooks as GetRspBust): per axis min + (max - min) * size / 100, a size beyond 100
# counting as 0, with min and max from the racial scaling table of chara/xls/charamake/human.cmp for the
# clan and body type; males get no bust scale at all. Each frame, while rendering (after the bone physics
# and Customize+), the character's PostBoneDeformer multiplies the model-space scale of j_mune_l and
# j_mune_r by it, keeping their position and turn, and propagates: every bone below them keeps its local
# transform, so it gets the same factors on its own axes and stays where it is (the game combines bones
# as T = T_parent + R_parent T_local, S = S_parent * S_local). That matters: the bodies of IVCS and YAS
# skeletons are weighted to iv_c_mune_l/r, below j_mune_l/r. Scales multiply axis by axis, so for the
# bones' own changes it makes no difference whether Customize+ comes first.

BUST_BONES = ("j_mune_l", "j_mune_r")
# The bust scale along the breast bones' own axes at sizes 0 and 100, for player characters (body type 1),
# as human.cmp has it: the same for every clan but the Lalafells', whose breasts are never made smaller.
BUST_RANGE = ((0.92, 0.8, 0.816), (1.08, 1.2, 1.184))
LALAFELL_BUST_RANGE = ((1.0, 1.0, 1.0), (1.08, 1.2, 1.184))
LALAFELL = frozenset(("c1101", "c1201"))
# Lalafell skeletons are about 0.45 times the size of the table's (the Midlander female's), every other
# race's 0.96 to 1.38 times.
LALAFELL_SIZE = 0.7


def bust_scale(size, lalafell=False):
    """The game's scale of the breast bones for Bust Size ``size`` (the slider), per axis of the bones' own
    axes: at 50 the model's own size, except for Lalafells."""
    low, high = (np.array(values) for values in (LALAFELL_BUST_RANGE if lalafell else BUST_RANGE))
    t = size / 100.0 if 0 <= size <= 100 else 0.0
    return low + (high - low) * t


def is_lalafell(axes):
    """Whether an armature (its `AxisMap`) is a Lalafell's: turned like one, or the size of one."""
    return axes.race in LALAFELL or (axes.size is not None and axes.size < LALAFELL_SIZE)


def bust_bones(parents, left_out=frozenset()):
    """The breast bones and every bone below them (the bones the game's bust size scales), parents first,
    without those in ``left_out``."""
    children = {}
    for name in hierarchy_order(parents):
        children.setdefault(parents[name], []).append(name)
    result = []
    for bone in BUST_BONES:
        if bone in parents:
            result += [name for name in [bone] + descendants(bone, children) if name not in left_out]
    return result


def bust_change(frame, scale):
    """A scale along a game bone's own axes (``scale``, per axis) as the bone's change in its Blender frame
    (``frame``: the game bone's axes to the bone's, `AxisMap.frame`): (values, turn or None) like
    `Plan.own` and `Plan.frames`. Its matrix is frame^T diag(scale) frame: for swapped axes, the scale
    reordered; for a freely turned bone, a scale along turned axes, turned back after."""
    scale = np.asarray(scale, dtype=float)
    if is_swap(frame):
        return (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), scale[np.argmax(np.abs(frame), axis=0)]), None
    return (np.zeros(3), quaternion(frame.T), scale), frame


def bust_plan(parents, axes, size):
    """Bust Size ``size`` on an armature (``parents``, its `AxisMap` ``axes``): (the bust scale, whether it's
    a Lalafell's, {bone: values}, {bone: turn}) with values and turns as `bust_change` gives them. Bones
    without a rest pose are left out, like the template's."""
    lalafell = is_lalafell(axes)
    scale = bust_scale(size, lalafell)
    values, turns = {}, {}
    for name in bust_bones(parents, axes.unplaced):
        values[name], turn = bust_change(axes.frame(name), scale)
        if turn is not None:
            turns[name] = turn
    return scale, lalafell, values, turns


# -----------------------------------------------------------------------------
# The rig's values

class Plan:
    """Values for the rig (all in Blender's armature space and bone frames), each (position, quaternion,
    scale):
    own      {bone: values} applied after the bone's pose, in its own frame
    frames   {bone: 3x3} a turn applied right after ``own``, where a position, turn and scale alone can't
             give the bone's change (a scale along axes other than the bone's)
    carries  [(propagating bone, [(space, values)], [bones below])], applied one after the other before
             the poses of the bones below, in the frame of the propagating bone of the original armature
             (space 'BONE') or in the armature's axes at that bone's head (space 'AXES')
    root     values applied last to every bone, or None
    bust     {bone: values} the game's bust size, applied after ``own`` and ``frames``, in the bone's frame
    bust_frames {bone: 3x3} a turn applied right after ``bust`` (on freely turned bones)
    expected {bone: 4x4} the rig's bone matrices at rest, and ``exact`` {bone: 4x4} what the game gives
    """

    def __init__(self):
        self.own = {}
        self.frames = {}
        self.carries = []
        self.root = None
        self.bust = {}
        self.bust_frames = {}
        self.bust_scale = None  # the game's scale of the breast bones (on their own game axes), or None
        self.lalafell = False   # whether the bust size took the Lalafells' range
        self.expected = {}
        self.exact = {}
        self.missing = []     # template bones the armature doesn't have
        self.unplaced = []    # template bones it has without a rest pose, left out
        self.edited = []      # template bones it changes
        self.axes = None

    def error(self):
        """Largest difference between the rig at rest and the game (armature units)."""
        return max((float(np.abs(self.expected[name] - self.exact[name]).max()) for name in self.exact),
                   default=0.0)


def _values(M):
    t, R, s = decompose(M)
    return t, quaternion(R), s


def _matrix(values):
    return compose(values[0], quaternion_matrix(values[1]), values[2])


# Blender keeps bone matrices in 32-bit floats: differences below this (times the armature's units for
# positions) are rounding.
TOLERANCE = 1e-6


def _close(A, B, units):
    return (float(np.abs(A[:3, :3] - B[:3, :3]).max()) < TOLERANCE
            and float(np.abs(A[:3, 3] - B[:3, 3]).max()) < TOLERANCE * max(1.0, units))


def _split(M, units):
    """(values, turn or None) with compose(values) x turn = 4x4 ``M``: any matrix, shear included."""
    values = _values(M)
    if _close(_matrix(values), M, units):
        return values, None
    U, S, Vt = np.linalg.svd(M[:3, :3])
    if np.linalg.det(U) < 0.0:
        U[:, 2] *= -1.0
        S[2] *= -1.0
    if np.linalg.det(Vt) < 0.0:
        Vt[2, :] *= -1.0
        S[2] *= -1.0
    return (M[:3, 3].copy(), quaternion(U), S), Vt


def _carry_steps(carry, axes):
    """A `Carry` as [(space, 4x4)] to apply one after the other: 'BONE' changes in the Blender frame of
    its bone, 'AXES' in the armature's axes at its head.

    In the game the bones below get A = T(T' + d) dR S T(-T'): about the bone's new position T' = T + d,
    scaled by S along the model's axes, turned by the bone's turn dR, and moved by its change of position
    d (twice: Customize+ adds it to their offset from the new position too). In the bone's frame that is
    T(2m) q T(-m) with the scale in between, m being d in the bone's frame."""
    F = np.identity(4)
    F[:3, :3] = axes.frame(carry.bone)
    to_game = np.diag([1.0 / axes.units] * 3 + [1.0]) @ F       # Blender bone frame -> game bone frame
    to_blender = F.T @ np.diag([axes.units] * 3 + [1.0])
    m = carry.move
    if carry.scale is None:
        return [('BONE', to_blender @ compose(2.0 * m, carry.turn, 1.0) @ compose(-m, np.identity(3), 1.0) @ to_game)]
    scale = np.abs(axes.turn) @ carry.scale  # model axes -> armature axes
    return [
        ('BONE', to_blender @ compose(-m, np.identity(3), 1.0) @ to_game),
        ('AXES', compose(np.zeros(3), np.identity(3), scale)),
        ('BONE', to_blender @ compose(2.0 * m, carry.turn, 1.0) @ to_game),
    ]


def plan(parents, rest, template, options, bust=None):
    """`Plan` for an armature (``parents`` {bone: parent or None}, ``rest`` {bone: 4x4 rest matrix}), a
    `template.Template` and the game's Bust Size (0 to 100, or None for none).

    Propagation is carried in each propagating bone's own frame (scaling in the armature's axes at its
    head), the innermost propagating bone's first: then the bones below keep following it and each other
    exactly, in any pose. What's left for each bone (its own change, the shear the game drops) goes into
    its own change, measured against `simulate` at rest, so the rig matches the game exactly there. The
    bust size comes after that, in the bones' own frames too, so it's exact in any pose as well."""
    result = Plan()
    axes = detect_axes(rest)
    result.axes = axes
    if bust is not None:
        result.bust_scale, result.lalafell, result.bust, result.bust_frames = bust_plan(parents, axes, bust)
    edits = template.edited()
    # The root's change needs no bone of its own: it moves and scales every bone about the armature's origin.
    result.missing = sorted(name for name in edits if name not in rest and name != ROOT_BONE)
    result.unplaced = sorted(name for name in edits if name in axes.unplaced)
    edits = {name: edit for name, edit in edits.items()
             if name == ROOT_BONE or (name in rest and name not in axes.unplaced)}
    result.edited = list(edits)

    model = {name: axes.to_game(name, rest[name]) for name in rest}
    simulation = simulate(parents, model, edits, options)
    root = axes.change_to_blender(simulation.root) if simulation.root is not None else np.identity(4)
    if simulation.root is not None:
        result.root = _values(root)

    steps = {}  # bone: [4x4 armature-space changes before its pose, outermost propagating bone first]
    for carry in simulation.carries:
        items = []
        change = np.identity(4)
        bone = rest[carry.bone]
        at_head = compose(bone[:3, 3], np.identity(3), 1.0)
        for space, matrix in _carry_steps(carry, axes):
            values = _values(matrix)
            items.append((space, values))
            frame = bone if space == 'BONE' else at_head
            change = frame @ _matrix(values) @ np.linalg.inv(frame) @ change
        result.carries.append((carry.bone, items, carry.below))
        for name in carry.below:
            steps.setdefault(name, []).append(change)

    for name in hierarchy_order(parents):
        chain = rest[name]
        for step in reversed(steps.get(name, ())):
            chain = step @ chain
        exact = axes.to_blender(name, simulation.model[name])
        if name in axes.unplaced:
            exact = chain  # no own change: only what propagation carries, which needs no rest pose
        wanted = np.linalg.inv(chain) @ exact
        own = np.identity(4)
        if not _close(wanted, np.identity(4), axes.units):
            values, turn = _split(wanted, axes.units)
            own = _matrix(values)
            result.own[name] = values
            if turn is not None:
                result.frames[name] = turn
                T = np.identity(4)
                T[:3, :3] = turn
                own = own @ T
        if name in result.bust:
            # The game multiplies the bone's scale by the bust scale: its own change after everything else.
            exact = axes.to_blender(name, simulation.model[name] @ np.diag(list(result.bust_scale) + [1.0]))
            own = own @ _matrix(result.bust[name])
            if name in result.bust_frames:
                T = np.identity(4)
                T[:3, :3] = result.bust_frames[name]
                own = own @ T
        result.expected[name] = root @ chain @ own
        result.exact[name] = root @ exact
    return result
