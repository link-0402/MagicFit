# SPDX-License-Identifier: GPL-3.0-or-later

"""Skirt weights: giving skirts and long dresses the weights of FFXIV's skirt bones.

FFXIV moves long cloth with "skirt bones", chains hanging from the hips that the game swings with its
cloth physics: ``j_sk_{f,s,b}_{a,b,c}_{l,r}`` (front, side and back; segments a, b, c from the top
down; left and right). Weights copied from a body have none of them, so the cloth only moves with the
legs. Weights copied from a reference skirt only fit dresses shaped like it.

Skirt mode works the skirt bones' weights out from where the bones hang, in the rest pose:

- The skirt bones' share of a vertex grows with its depth below the hip bone: none at Blend Start
  above it, all of it Blend Length lower (a smoothstep). Cloth lying on the body keeps some of its body
  weights (Skin Weight, fading out over Skin Distance), so bone scaling such as Customize+ still pushes
  it out with the thighs and the butt, and Body Weight keeps some everywhere.
- The skirt share is split between the chains by the vertex's angle around the hips: its position
  between the two chains around it, as a fraction of the gap between them, with each chain reaching
  Spread gaps to either side. Within a chain it is split between the segments by height, blending over
  Joint Blend around each joint. The chains' angles follow the bones down, since they splay outward.
- Cloth weighted to the arms or the head (sleeves, gloves, hair) hangs from them, however low it
  reaches in the rest pose: the skirt share only goes to the part of a vertex's body weight on other
  bones.

The body share is made of the vertex's own body bone weights, as fractions of one. So applying Skirt
again changes nothing, and it goes on top of whatever transfer gave the mesh its body weights. A vertex
that has none takes them from the body at its closest point.

Like `painting` and `straighten`, nothing here depends on the 3D view.
"""

import math
import re

import numpy as np
from mathutils import Vector
from mathutils.interpolate import poly_3d_calc

from . import fitting, goals, painting, straighten
from .goals import DEFAULT_MAX_GROUPS, smoothstep

# Skirt bones: j_sk_<position>_<segment>_<side>.
SKIRT_BONE = re.compile(r"^j_sk_([a-z])_([a-z])_([lr])$")
HIP_BONE = "j_kosi"
# The arms and the head, with every bone below them (fingers, hair): cloth weighted to them hangs from
# them, not the hips.
ARMS_AND_HEAD = re.compile(r"^(j_sako|j_ude_[ab]|j_te|n_hkata|n_hhiji|n_hte)_[lr]$|^(j_kubi|j_kao)$")
# The up axis is the scene's Z, unless the skirt chains hang along a direction farther from it than this.
MAX_TILT = math.radians(30.0)
# Shares below this don't count.
SHARE_EPSILON = 1e-6


class SkirtError(Exception):
    """Why skirt weights can't be worked out; the message is meant for the user."""


def has_skirt_bones(armature_object):
    """Whether ``armature_object`` is an armature with deforming skirt bones (cheap, for UI checks)."""
    if armature_object is None or armature_object.type != 'ARMATURE':
        return False
    return any(bone.use_deform and SKIRT_BONE.match(bone.name) for bone in armature_object.data.bones)


# -----------------------------------------------------------------------------
# Shares

def height_fraction(height, blend_start, blend_length):
    """The skirt bones' share by height above the hip bone: 0 at ``blend_start`` and above, 1 from
    ``blend_length`` below it on, a smoothstep in between."""
    height = np.asarray(height, dtype=np.float64)
    return smoothstep((blend_start - height) / max(blend_length, 1e-9))


def body_fraction(skirt_share, distance, *, skin_weight, skin_distance, body_weight):
    """The body bones' share of each vertex: whatever the skirt bones' share by height leaves, but at
    least ``body_weight``, and at least ``skin_weight`` where the vertex lies on the body (``distance``
    0 or less, fading out to nothing at ``skin_distance``). Vertices with no skirt share keep all of it.

    ``distance`` may be None (no body), and NaN where the body wasn't found.
    """
    skirt_share = np.asarray(skirt_share, dtype=np.float64)
    kept = np.maximum(1.0 - skirt_share, body_weight)
    if skin_weight > 0.0 and distance is not None:
        distance = np.asarray(distance, dtype=np.float64)
        distance = np.where(np.isfinite(distance), np.maximum(distance, 0.0), np.inf)
        if skin_distance > 0.0:
            hold = skin_weight * (1.0 - smoothstep(distance / skin_distance))
        else:
            hold = np.where(distance <= 0.0, skin_weight, 0.0)
        kept = np.maximum(kept, hold)
    kept = np.clip(kept, 0.0, 1.0)
    return np.where(skirt_share <= 0.0, 1.0, kept)


def chain_weights(chain_angles, angle, spread):
    """Each chain's share ((n, k), rows adding up to 1) for vertices at ``angle`` around the hips.

    ``chain_angles`` are the chains' angles, (k,) or per vertex (n, k), in their order around the hips
    (increasing angle). A vertex's place between the two chains around it is measured as a fraction of
    the gap between them; each chain then reaches ``spread`` gaps to either side, fading linearly. With
    a spread of 1 a vertex is blended between the two chains around it only; 1.5 gives a chain 0.6 at
    its own angle and each neighbour 0.2, and at most 3 chains share a vertex.
    """
    angle = np.asarray(angle, dtype=np.float64).reshape(-1)
    count = len(angle)
    theta = np.asarray(chain_angles, dtype=np.float64)
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, (count, len(theta)))
    chains = theta.shape[1]
    if chains == 1:
        return np.ones((count, 1))
    full = 2.0 * math.pi
    gaps = np.mod(np.roll(theta, -1, axis=1) - theta, full)
    gaps = np.where(gaps <= 1e-9, full, gaps)
    past = np.mod(angle[:, None] - theta, full)  # how far past each chain, toward the next one
    between = past < gaps
    first = np.argmax(between, axis=1)
    rows = np.arange(count)
    fraction = np.where(between[rows, first], past[rows, first] / gaps[rows, first], 0.0)
    place = first + fraction  # in chains, 0 at the first chain
    offset = np.abs(np.mod(place[:, None] - np.arange(chains) + 0.5 * chains, chains) - 0.5 * chains)
    weights = np.maximum(0.0, 1.0 - offset / max(spread, 1.0))
    return weights / weights.sum(axis=1, keepdims=True)


def segment_weights(joints, height, joint_blend):
    """Each segment's share of a chain ((n, m), rows adding up to 1) for vertices at ``height``.

    ``joints`` are the heights of the segments' heads, from the top down. A segment has everything from
    its head down to the next segment's head; around each joint (all heads but the top one) the two
    segments blend over ``joint_blend``, a smoothstep centred on the joint.
    """
    height = np.asarray(height, dtype=np.float64).reshape(-1)
    joints = np.asarray(joints, dtype=np.float64)
    below = np.ones((len(height), len(joints)))  # how far below each segment's head
    for k in range(1, len(joints)):
        if joint_blend > 0.0:
            below[:, k] = smoothstep((joints[k] + 0.5 * joint_blend - height) / joint_blend)
        else:
            below[:, k] = height < joints[k]
    below = np.minimum.accumulate(below, axis=1)  # heads out of order: a lower one never counts more
    return below - np.concatenate((below[:, 1:], np.zeros((len(height), 1))), axis=1)


# -----------------------------------------------------------------------------
# The skirt bones

class SkirtChain:
    """One chain of skirt bones: ``key`` (position letter, side letter), ``bones`` (names from the top
    down), and the ``heights`` and ``angles`` (radians, unwrapped) of their heads around the hips."""

    __slots__ = ("key", "bones", "heights", "angles")


class SkirtRig:
    """The skirt bones of an armature in its rest pose, measured around its hips (world space).

    ``hip`` is the head of the hip bone, ``up`` points up (the scene's Z unless the chains hang along
    something else) and ``ax``, ``ay`` span the plane angles are measured in. ``chains`` are ordered by
    angle; ``bone_names`` lists their bones, chain by chain from the top down: the columns of
    `distribution`. ``arms_and_head`` names the deforming bones of the arms and the head (see
    `ARMS_AND_HEAD`).
    """

    def __init__(self, hip, up, ax, ay, chains, arms_and_head=frozenset()):
        self.hip = hip
        self.up = up
        self.ax = ax
        self.ay = ay
        self.chains = chains
        self.bone_names = [name for chain in chains for name in chain.bones]
        self.arms_and_head = frozenset(arms_and_head)

    @classmethod
    def from_object(cls, armature_object):
        """The skirt rig of ``armature_object``, or None when it has no deforming skirt bones."""
        if armature_object is None or armature_object.type != 'ARMATURE':
            return None
        matrix = armature_object.matrix_world
        found = {}
        hip = None
        arms_and_head = set()
        for bone in armature_object.data.bones:
            if bone.name == HIP_BONE:
                hip = np.array(matrix @ bone.head_local, dtype=np.float64)
            if not bone.use_deform:
                continue
            if any(ARMS_AND_HEAD.match(item.name) for item in (bone, *bone.parent_recursive)):
                arms_and_head.add(bone.name)
            match = SKIRT_BONE.match(bone.name)
            if match is not None:
                position, segment, side = match.groups()
                head = np.array(matrix @ bone.head_local, dtype=np.float64)
                found.setdefault((position, side), []).append((segment, bone.name, head))
        if not found:
            return None
        bones = {key: sorted(members) for key, members in found.items()}
        tops = np.array([members[0][2] for members in bones.values()])
        if hip is None:
            hip = tops.mean(axis=0)

        up = np.array((0.0, 0.0, 1.0))
        hang = np.array([members[0][2] - members[-1][2] for members in bones.values() if len(members) > 1])
        if len(hang):
            hang = hang.mean(axis=0)
            length = np.linalg.norm(hang)
            if length > 1e-9 and math.acos(np.clip(hang[2] / length, -1.0, 1.0)) > MAX_TILT:
                up = hang / length
        ax = np.array((1.0, 0.0, 0.0)) - up[0] * up
        if np.linalg.norm(ax) < 1e-6:
            ax = np.array((0.0, 1.0, 0.0)) - up[1] * up
        ax /= np.linalg.norm(ax)
        ay = np.cross(up, ax)

        chains = []
        for key, members in bones.items():
            chain = SkirtChain()
            chain.key = key
            chain.bones = [name for _segment, name, _head in members]
            relative = np.array([head for _segment, _name, head in members]) - hip
            chain.heights = relative @ up
            chain.angles = np.unwrap(np.arctan2(relative @ ay, relative @ ax))
            chains.append(chain)
        chains.sort(key=lambda chain: math.atan2(math.sin(chain.angles[0]), math.cos(chain.angles[0])))
        return cls(hip, up, ax, ay, chains, arms_and_head)

    def cylindrical(self, points):
        """Height above the hip and angle around it of world space ``points`` (n, 3)."""
        relative = np.asarray(points, dtype=np.float64) - self.hip
        return relative @ self.up, np.arctan2(relative @ self.ay, relative @ self.ax)

    def chain_angles(self, height):
        """Each chain's angle at ``height`` ((n, k)), following its bones down."""
        height = np.asarray(height, dtype=np.float64).reshape(-1)
        return np.stack([
            np.interp(height, chain.heights[::-1], chain.angles[::-1]) for chain in self.chains
        ], axis=1)

    def distribution(self, height, angle, *, spread, joint_blend):
        """Each skirt bone's share ((n, len(bone_names)), rows adding up to 1) at ``height``, ``angle``."""
        height = np.asarray(height, dtype=np.float64).reshape(-1)
        chains = chain_weights(self.chain_angles(height), angle, spread)
        result = np.empty((len(height), len(self.bone_names)))
        column = 0
        for index, chain in enumerate(self.chains):
            segments = segment_weights(chain.heights, height, joint_blend)
            result[:, column:column + len(chain.bones)] = chains[:, index:index + 1] * segments
            column += len(chain.bones)
        return result


# -----------------------------------------------------------------------------
# Body weights for vertices without any

class BodyWeights:
    """The body's weights ({group name: weight}) at the point of its surface closest to a position.

    ``geometry`` is the body as `read_geometry` with source 'WEIGHTED' read it (in the rest pose);
    ``key`` is its search tree's cache key.
    """

    def __init__(self, body, geometry, key):
        self._mesh = body.data
        self._names = [group.name for group in body.vertex_groups]
        self._weights = {} if geometry.weights is None else dict(enumerate(geometry.weights))
        self._co = geometry.co
        self._tris = geometry.tris.tolist()
        self._bvh = painting.surface_bvh(key, geometry.co, geometry.tris) if self._tris else None

    def _vertex_weights(self, index):
        weights = self._weights.get(index)
        if weights is None:
            weights = tuple((elem.group, elem.weight) for elem in self._mesh.vertices[index].groups)
            self._weights[index] = weights
        return weights

    def sample(self, point):
        if self._bvh is None:
            return {}
        location, _normal, tri_index, _distance = self._bvh.find_nearest(point)
        if location is None:
            return {}
        tri = self._tris[tri_index]
        co = self._co
        factors = poly_3d_calc([Vector(co[tri[0]]), Vector(co[tri[1]]), Vector(co[tri[2]])], location)
        result = {}
        for vert, factor in zip(tri, factors):
            if factor <= 0.0:
                continue
            for group, weight in self._vertex_weights(vert):
                if 0 <= group < len(self._names):
                    name = self._names[group]
                    result[name] = result.get(name, 0.0) + weight * factor
        return result


# -----------------------------------------------------------------------------
# Stroke

class SkirtStroke(goals.GoalStroke):
    """Blends a mesh's weights toward its skirt weights, dab by dab.

    ``rest_co`` are the world space vertex positions in the rest pose, ``rig`` the `SkirtRig` of the
    armature deforming the mesh, ``body`` the `fitting.BodySurface` of the body in its rest pose (for
    Skin Weight) and ``body_weights`` a `BodyWeights` of it (for vertices without body weights of their
    own); both may be None. See `goals.GoalStroke` for how the weights change.
    """

    def __init__(
            self, obj, rest_co, rig, *,
            body=None,
            body_weights=None,
            blend_start=0.0,
            blend_length=0.40,
            skin_weight=1.0,
            skin_distance=0.07,
            body_weight=0.0,
            spread=1.5,
            joint_blend=0.10,
            max_groups=DEFAULT_MAX_GROUPS,
            group_mode='ALL',
    ):
        super().__init__(obj, rest_co, max_groups=max_groups, group_mode=group_mode)
        self.rig = rig
        self.body = body
        self.body_weights = body_weights
        self.skin_weight = skin_weight if body is not None else 0.0
        self.skin_distance = skin_distance
        self.body_weight = body_weight

        self.skirt_names = set(rig.bone_names)
        self.skirt_groups = {group.index for group in self.groups if group.name in self.skirt_names}
        self.column_group = [self.name_to_group.get(name, -1) for name in rig.bone_names]

        height, angle = rig.cylindrical(self.rest)
        self.height = height
        self.skirt_share = height_fraction(height, blend_start, blend_length)
        self.shares = rig.distribution(height, angle, spread=spread, joint_blend=joint_blend)
        # Distance to the body in the rest pose, measured when first needed.
        self.distance = np.full(self.weld_count, np.nan)
        self.measured = np.zeros(self.weld_count, dtype=bool)
        self.body_samples = {}

        # What the stroke ran into, for the operator to report.
        self.painted_above = False
        self.painted_skirt = False
        self.unweighted = set()

    # -------------------------------------------------------------------------
    # Public API

    def body_fractions(self, verts):
        """The body bones' share of ``verts`` (their skirt weights leave that much to the body), for
        vertices without weights on the arms or the head."""
        return self.body_fractions_of(self.weld[np.asarray(verts, dtype=np.int64)])

    def body_fractions_of(self, groups, scale=1.0):
        """The body bones' share of welded ``groups``, with their skirt share times ``scale`` (the part of
        their body weight not on the arms and the head)."""
        skirt_share = self.skirt_share[groups] * scale
        if self.skin_weight > 0.0:
            self._measure(np.unique(groups[skirt_share > 0.0]))
        return body_fraction(
            skirt_share, self.distance[groups] if self.skin_weight > 0.0 else None,
            skin_weight=self.skin_weight, skin_distance=self.skin_distance, body_weight=self.body_weight,
        )

    # -------------------------------------------------------------------------
    # Goals

    def _prepare(self, groups):
        below = self.skirt_share[groups] > 0.0
        self.painted_above |= bool((~below).any())
        self.painted_skirt |= bool(below.any())
        return self.body_fractions_of(groups).tolist()

    def _leave(self, current, kept):
        # Nothing hangs from the skirt bones here, and nothing did.
        return kept >= 1.0 - SHARE_EPSILON and not any(
            weight > 0.0 and group in self.skirt_groups and group not in self.locked
            for group, weight in current.items()
        )

    def _goal(self, vert, group, kept, shares):
        base = self._body_shares(group, shares)
        held = self._arms_and_head_share(base)
        if held > SHARE_EPSILON:
            # Sleeves and hair hang from the arms and the head, however low they reach.
            kept = float(self.body_fractions_of(np.array([group]), max(1.0 - held, 0.0))[0])
            if kept >= 1.0 - SHARE_EPSILON and not self.skirt_groups.intersection(shares):
                return None
        skirt = 1.0 - kept
        hanging = self._skirt_shares(group) if skirt > SHARE_EPSILON else {}
        if not base:
            if not hanging:
                return None
            if kept > SHARE_EPSILON:
                self.unweighted.add(vert)
            kept, skirt = 0.0, 1.0
        elif not hanging:
            kept, skirt = 1.0, 0.0
        return [(kept, base), (skirt, hanging)]

    def _measure(self, groups):
        new = groups[~self.measured[groups]]
        if not len(new):
            return
        near = self.body.nearest(self.rest[new])
        self.distance[new] = np.where(near.found, near.distance, np.nan)
        self.measured[new] = True

    def _body_shares(self, group, shares):
        """{group or name of a group to create: share} of the body bones: the vertex's own, else the
        body's at the closest point. Adds up to 1, or is empty."""
        base = {key: share for key, share in shares.items() if key not in self.skirt_groups}
        if not base and self.body_weights is not None:
            sample = self.body_samples.get(group)
            if sample is None:
                sample = self.body_weights.sample(Vector(self.rest[group]))
                self.body_samples[group] = sample
            for name, weight in sample.items():
                if weight <= 0.0 or SKIRT_BONE.match(name):
                    continue
                key = self._key(name)
                if key is not None:
                    base[key] = base.get(key, 0.0) + weight
        total = sum(base.values())
        return {key: share / total for key, share in base.items()} if total > 1e-12 else {}

    def _arms_and_head_share(self, base):
        """How much of ``base`` (body shares, as from `_body_shares`) is on the arms and the head."""
        names = self.rig.arms_and_head
        return sum(
            share for key, share in base.items()
            if (key if isinstance(key, str) else self.group_names.get(key)) in names
        )

    def _skirt_shares(self, group):
        """{group or name of a group to create: share} of the skirt bones, adding up to 1 (locked ones
        left out), or empty."""
        row = self.shares[group]
        result = {}
        for column in np.flatnonzero(row > SHARE_EPSILON).tolist():
            key = self.column_group[column]
            if key < 0:
                key = self.rig.bone_names[column]
            elif key in self.locked:
                continue
            result[key] = result.get(key, 0.0) + float(row[column])
        total = sum(result.values())
        return {key: share / total for key, share in result.items()} if total > 1e-12 else {}

    def _group_created(self, group, name):
        if name in self.skirt_names:
            self.skirt_groups.add(group)
            self.column_group[self.rig.bone_names.index(name)] = group

    def _group_forgotten(self, group, name):
        self.skirt_groups.discard(group)
        self.column_group = [-1 if index == group else index for index in self.column_group]


# -----------------------------------------------------------------------------
# Setting up

def stroke_params(settings):
    """`SkirtStroke` keyword arguments from the brush settings (`Scene.magic_fit`)."""
    return dict(
        blend_start=settings.skirt_blend_start,
        blend_length=settings.skirt_blend_length,
        skin_weight=settings.skirt_skin_weight,
        skin_distance=settings.skirt_skin_distance,
        body_weight=settings.skirt_body_weight,
        spread=settings.skirt_spread,
        joint_blend=settings.skirt_joint_blend,
        max_groups=settings.skirt_max_groups,
        group_mode=settings.group_mode,
    )


def build_stroke(context, obj, settings):
    """A `SkirtStroke` for ``obj`` with the brush settings, measured in the rest pose.

    Raises `SkirtError` when it can't be set up.
    """
    modifier = straighten.deforming_armature(obj)
    if modifier is None:
        raise SkirtError("Skirt needs an Armature modifier")
    rig = SkirtRig.from_object(modifier.object)
    if rig is None:
        raise SkirtError("Armature '{:s}' has no skirt bones (j_sk_...)".format(modifier.object.name))
    params = stroke_params(settings)
    body = settings.target
    if body is not None and (body == obj or body.type != 'MESH'):
        body = None
    if body is None and params["skin_weight"] > 0.0:
        raise SkirtError("Pick the body the skirt hangs around, or set Skin Weight to 0")

    objects = (obj, body) if body is not None else (obj,)
    geometries = straighten.rest_geometry(context, objects, ('INDEXED', 'WEIGHTED')[:len(objects)])
    surface = weights = None
    if body is not None:
        key = ("rest", body.as_pointer())
        surface = fitting.BodySurface.from_geometry(key, geometries[1])
        if surface is None:
            if params["skin_weight"] > 0.0:
                raise SkirtError("The body mesh has no faces")
        else:
            weights = BodyWeights(body, geometries[1], key)
    return SkirtStroke(obj, geometries[0].co, rig, body=surface, body_weights=weights, **params)
