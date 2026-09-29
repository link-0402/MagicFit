# SPDX-License-Identifier: GPL-3.0-or-later

"""Heel weights: making high heels and boots rigid below the ankle.

Weights copied from a body give a shoe the body's foot bones: the calf (j_asi_c), the ankle (j_asi_d),
the toes (j_asi_e) and, on IVCS bodies, a bone per toe. Those bend the foot at the ball and the toes,
which bare feet should, but a high heel, its sole and its heel spike shouldn't: posed, the heel wobbles
and the sole folds. Well weighted heels (and the Yet Another Devkit's "Heel Weights") use only two bones
there: everything below the ankle follows the ankle bone rigidly, and the calf hands over to it around
the ankle.

Heels mode does just that, in the rest pose:

- The foot's share grows with the depth below the ankle: none Ankle Blend / 2 above Ankle Height (a
  height above the head of j_asi_d, measured along the shin), all of it Ankle Blend / 2 below it (a
  smoothstep). Vertices above that keep their weights.
- The foot's share goes to the ankle bone, except that Toe Bend of it goes to the toe bone from the ball
  of the foot (the head of j_asi_e) on, fading out over 4 cm behind it, for flat shoes and boots that
  should bend there.
- The rest is the vertex's own leg weights (anything but the foot's bones), as fractions of one, or the
  calf when it has none. So applying Heels again changes nothing.

Like `painting`, nothing here depends on the 3D view.
"""

import numpy as np

from . import goals, straighten
from .goals import DEFAULT_MAX_GROUPS, smoothstep

SIDES = ("l", "r")
CALF = "j_asi_c_{:s}"
ANKLE = "j_asi_d_{:s}"
TOE = "j_asi_e_{:s}"
# The toe bone takes over over this distance behind the ball of the foot (it bends the whole toe box).
TOE_BLEND = 0.04


class HeelsError(Exception):
    """Why heel weights can't be worked out; the message is meant for the user."""


def _deforming(bones, name):
    bone = bones.get(name)
    return bone is not None and bone.use_deform


def has_foot_bones(armature_object):
    """Whether ``armature_object`` is an armature with a deforming calf and ankle bone on either side
    (cheap, for UI checks)."""
    if armature_object is None or armature_object.type != 'ARMATURE':
        return False
    bones = armature_object.data.bones
    return any(_deforming(bones, CALF.format(side)) and _deforming(bones, ANKLE.format(side)) for side in SIDES)


def foot_fraction(height, ankle_height, ankle_blend):
    """The foot's share by height above the ankle joint: 1 from ``ankle_blend`` / 2 below ``ankle_height``
    down, 0 from as far above it up, a smoothstep in between."""
    height = np.asarray(height, dtype=np.float64)
    blend = max(ankle_blend, 1e-9)
    return 1.0 - smoothstep((height - ankle_height + 0.5 * blend) / blend)


def toe_fraction(forward, toe_bend):
    """The toe bone's share of the foot at ``forward`` (distance ahead of the ball of the foot; NaN where
    there's no toe bone): ``toe_bend`` from the ball on, nothing from `TOE_BLEND` behind it."""
    forward = np.asarray(forward, dtype=np.float64)
    share = toe_bend * smoothstep((np.nan_to_num(forward, nan=-np.inf) + TOE_BLEND) / TOE_BLEND)
    return np.where(np.isfinite(forward), share, 0.0)


class HeelSide:
    """One leg: bone names ``calf``, ``ankle`` and ``toe`` (None without one), the ``ankle_point`` (head of
    the ankle bone), ``up`` along the shin, and the ``ball`` of the foot (head of the toe bone) and the
    ``forward`` direction from the ankle to it (both None without a toe bone), in world space."""

    __slots__ = ("side", "calf", "ankle", "toe", "ankle_point", "up", "ball", "forward")


class HeelRig:
    """The legs' foot bones of an armature in its rest pose. ``foot_names`` are the bones of both feet:
    the ankle bones and every deforming bone below them (toes)."""

    def __init__(self, sides, foot_names):
        self.sides = sides
        self.foot_names = foot_names

    @classmethod
    def from_object(cls, armature_object):
        """The foot bones of ``armature_object``, or None when no leg has a deforming calf and ankle bone."""
        if armature_object is None or armature_object.type != 'ARMATURE':
            return None
        matrix = armature_object.matrix_world
        bones = armature_object.data.bones

        def world(point):
            return np.array(matrix @ point, dtype=np.float64)

        sides = []
        foot_names = set()
        for side in SIDES:
            calf, ankle = bones.get(CALF.format(side)), bones.get(ANKLE.format(side))
            if calf is None or ankle is None or not (calf.use_deform and ankle.use_deform):
                continue
            leg = HeelSide()
            leg.side = side
            leg.calf, leg.ankle = calf.name, ankle.name
            leg.ankle_point = world(ankle.head_local)
            down = leg.ankle_point - world(calf.head_local)
            length = np.linalg.norm(down)
            leg.up = -down / length if length > 1e-6 else np.array((0.0, 0.0, 1.0))
            toe = bones.get(TOE.format(side))
            leg.toe = leg.ball = leg.forward = None
            if toe is not None and toe.use_deform:
                ball = world(toe.head_local)
                forward = ball - leg.ankle_point
                forward -= (forward @ leg.up) * leg.up
                if np.linalg.norm(forward) > 1e-6:
                    leg.toe, leg.ball, leg.forward = toe.name, ball, forward / np.linalg.norm(forward)
            foot_names.add(ankle.name)
            foot_names.update(bone.name for bone in ankle.children_recursive if bone.use_deform)
            sides.append(leg)
        if not sides:
            return None
        return cls(sides, foot_names)

    def measure(self, points):
        """For world space ``points`` (n, 3): the leg each belongs to (index into ``sides``, the one whose
        shin line is closest), its height above that leg's ankle joint along the shin, and how far it is
        ahead of the ball of the foot (NaN without a toe bone)."""
        points = np.asarray(points, dtype=np.float64)
        heights, aside, ahead = [], [], []
        for leg in self.sides:
            relative = points - leg.ankle_point
            height = relative @ leg.up
            heights.append(height)
            aside.append(np.linalg.norm(relative - height[:, None] * leg.up, axis=1))
            ahead.append((points - leg.ball) @ leg.forward if leg.forward is not None else np.full(len(points), np.nan))
        side = np.argmin(np.stack(aside, axis=1), axis=1)
        rows = np.arange(len(points))
        return side, np.stack(heights, axis=1)[rows, side], np.stack(ahead, axis=1)[rows, side]


class HeelsStroke(goals.GoalStroke):
    """Blends a mesh's weights toward its heel weights, dab by dab.

    ``rest_co`` are the world space vertex positions in the rest pose, ``rig`` the `HeelRig` of the
    armature deforming the mesh. See `goals.GoalStroke` for how the weights change.
    """

    def __init__(
            self, obj, rest_co, rig, *,
            ankle_height=0.02,
            ankle_blend=0.12,
            toe_bend=0.0,
            max_groups=DEFAULT_MAX_GROUPS,
            group_mode='ALL',
    ):
        super().__init__(obj, rest_co, max_groups=max_groups, group_mode=group_mode)
        self.rig = rig
        self.side, self.height, self.ahead = rig.measure(self.rest)
        self.foot_share = foot_fraction(self.height, ankle_height, ankle_blend)
        self.toe_share = self.foot_share * toe_fraction(self.ahead, toe_bend)
        self.foot_groups = {group.index for group in self.groups if group.name in rig.foot_names}
        # What the stroke ran into, for the operator to report.
        self.painted_leg = False
        self.painted_foot = False

    def _prepare(self, groups):
        foot = self.foot_share[groups] > 0.0
        self.painted_leg |= bool((~foot).any())
        self.painted_foot |= bool(foot.any())
        return groups.tolist()

    def _leave(self, current, group):
        return self.foot_share[group] <= 0.0  # above the ankle

    def _goal(self, vert, group, _param, shares):
        leg_bones = self.rig.sides[self.side[group]]
        foot = float(self.foot_share[group])
        toe = float(self.toe_share[group])
        leg = {key: share for key, share in shares.items() if key not in self.foot_groups}
        total = sum(leg.values())
        leg = {key: share / total for key, share in leg.items()} if total > 1e-12 else {}
        if not leg:
            calf = self._key(leg_bones.calf)
            leg = {calf: 1.0} if calf is not None else {}
        ankle = self._key(leg_bones.ankle)
        toe_key = self._key(leg_bones.toe) if leg_bones.toe is not None and toe > 0.0 else None
        if toe_key is None:
            toe = 0.0
        if ankle is None and foot - toe > 0.0:
            return None  # the ankle bone is locked: this can't be made rigid, so leave it as it is
        return [
            (1.0 - foot, leg),
            (foot - toe, {ankle: 1.0} if ankle is not None else {}),
            (toe, {toe_key: 1.0} if toe_key is not None else {}),
        ]

    def _group_created(self, group, name):
        if name in self.rig.foot_names:
            self.foot_groups.add(group)

    def _group_forgotten(self, group, name):
        self.foot_groups.discard(group)


def stroke_params(settings):
    """`HeelsStroke` keyword arguments from the brush settings (`Scene.magic_fit`)."""
    return dict(
        ankle_height=settings.heels_ankle_height,
        ankle_blend=settings.heels_ankle_blend,
        toe_bend=settings.heels_toe_bend,
        group_mode=settings.group_mode,
    )


def build_stroke(context, obj, settings):
    """A `HeelsStroke` for ``obj`` with the brush settings, measured in the rest pose.

    Raises `HeelsError` when it can't be set up.
    """
    modifier = straighten.deforming_armature(obj)
    if modifier is None:
        raise HeelsError("Heels needs an Armature modifier")
    rig = HeelRig.from_object(modifier.object)
    if rig is None:
        raise HeelsError("Armature '{:s}' has no foot bones (j_asi_c, j_asi_d)".format(modifier.object.name))
    rest, = straighten.rest_geometry(context, (obj,), ('INDEXED',))
    return HeelsStroke(obj, rest.co, rig, **stroke_params(settings))
