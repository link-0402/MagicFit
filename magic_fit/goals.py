# SPDX-License-Identifier: GPL-3.0-or-later

"""Strokes that move bone weights toward goals worked out per vertex (Skirt and Heels).

A goal is made of parts, each a share of the vertex's bone weight spread over some groups: for a skirt,
the body bones' part and the skirt bones' part; for a shoe, the leg, the foot and the toes. `GoalStroke`
turns the parts into weights and blends toward them, keeping each vertex's total bone weight, locked
groups and Current Group mode as Straighten does. Weights too small to keep and those beyond the group
limit are dropped, but every part keeps its share, so applying a goal twice changes nothing.

Like `painting`, nothing here depends on the 3D view.
"""

import numpy as np

from . import fitting, painting
from .painting import ZERO_WEIGHT

# FFXIV models take at most 8 bone weights per vertex.
DEFAULT_MAX_GROUPS = 8


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _node_attributes(modifier):
    """Names of the attributes (vertex groups among them) a Geometry Nodes modifier's inputs read and its
    outputs write."""
    names = set()
    if hasattr(modifier, "properties"):
        # Newer Blender versions keep the inputs and outputs in `properties` (None without a node group).
        interface = modifier.properties
        for sockets in (getattr(interface, "inputs", None), getattr(interface, "outputs", None)):
            if sockets is None:
                continue
            for prop in sockets.bl_rna.properties:
                if prop.type == 'POINTER' and prop.identifier != "rna_type":
                    socket = getattr(sockets, prop.identifier)
                    name = getattr(socket, "attribute_name", "")
                    if name and getattr(socket, "type", 'ATTRIBUTE') == 'ATTRIBUTE':
                        names.add(name)
        return names
    for key in modifier.keys():
        if key.endswith("_attribute_name") and modifier[key]:
            if modifier.get(key[:-len("_attribute_name")] + "_use_attribute", True):
                names.add(modifier[key])
    return names


def mask_groups(obj):
    """Names of ``obj``'s vertex groups that mask something (a modifier, a shape key, a particle system,
    Weight Transfer's masks) rather than hold bone weights."""
    names = set()

    def add(owner):
        for prop in owner.bl_rna.properties:
            if prop.type == 'STRING' and "vertex_group" in prop.identifier:
                value = getattr(owner, prop.identifier, "")
                if value:
                    names.add(value)

    for modifier in obj.modifiers:
        add(modifier)
        # Cloth, Soft Body and Fluid keep their groups in their settings.
        for name in ("settings", "collision_settings", "flow_settings"):
            settings = getattr(modifier, name, None)
            if settings is not None:
                add(settings)
        if modifier.type == 'NODES':
            names.update(_node_attributes(modifier))
    for system in getattr(obj, "particle_systems", ()):
        add(system)
    keys = getattr(obj.data, "shape_keys", None)
    if keys is not None:
        names.update(block.vertex_group for block in keys.key_blocks if block.vertex_group)
    transfer = getattr(obj, "magic_fit_transfer", None)
    if transfer is not None:
        names.update(name for name in (transfer.vertex_group, transfer.inpaint_group) if name)
    return names


def remove_empty_groups(obj, keep=()):
    """Remove ``obj``'s vertex groups that hold no weight anywhere, except locked ones, masks and ``keep``
    (names). Returns the names removed; the active group stays active."""
    vertex_groups = obj.vertex_groups
    if not len(vertex_groups):
        return []
    largest = np.zeros(len(vertex_groups))
    for vert in obj.data.vertices:
        for elem in vert.groups:
            if elem.group < len(largest) and elem.weight > largest[elem.group]:
                largest[elem.group] = elem.weight
    spared = mask_groups(obj) | set(keep)
    names = [group.name for group in vertex_groups
             if largest[group.index] <= ZERO_WEIGHT and not group.lock_weight and group.name not in spared]
    if names:
        active = vertex_groups.active.name if vertex_groups.active is not None else None
        for name in names:
            vertex_groups.remove(vertex_groups[name])
        if active is not None and active in vertex_groups:
            vertex_groups.active_index = vertex_groups[active].index
    return names


class GoalStroke(painting.WeightEdit):
    """Moves a mesh's bone weights toward goals per vertex, dab by dab.

    ``rest_co`` are the world space vertex positions in the rest pose. Vertices sharing a position (split
    seams) are welded: subclasses work out one goal per welded group, from ``rest``. Only bone weights
    change, and each vertex keeps their total. Locked groups are never changed. ``group_mode`` 'ACTIVE'
    only changes the active group's weight (the vertex's other bones make room for it, keeping their
    proportions).

    With ``replace``, the goals replace a vertex's weights instead: every group that isn't locked or a
    mask (see `mask_groups`) counts as a bone weight, bone or not (such as an earlier skeleton's bones the
    armature lacks), and the bone weights add up to 1 whatever they added up to before.

    Subclasses implement `_goal`, and may implement `_prepare` and `_leave`.
    """

    def __init__(self, obj, rest_co, *, max_groups=DEFAULT_MAX_GROUPS, group_mode='ALL', replace=False):
        super().__init__(obj)
        self.max_groups = max(1, int(max_groups))
        self.replace = replace
        self.bone_names = painting.deform_bone_names(obj)
        self.name_to_group = {group.name: group.index for group in self.groups}
        self.group_names = {group.index: group.name for group in self.groups}
        self.bone_groups = {group.index for group in self.groups if group.name in self.bone_names}
        if replace:
            masks = mask_groups(obj)
            self.bone_groups |= {group.index for group in self.groups if group.name not in masks}
        self.active = obj.vertex_groups.active_index if group_mode == 'ACTIVE' else None
        self.nothing_to_do = self.active is not None and not (
            0 <= self.active < len(self.groups) and
            self.active in self.bone_groups and self.active not in self.locked
        )

        mesh = self.mesh
        count = len(mesh.vertices)
        co = np.empty(count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        self.weld, self.weld_count = fitting.weld_groups(co.reshape(-1, 3).astype(np.float64), fitting.WELD_DISTANCE)
        members = np.bincount(self.weld, minlength=self.weld_count).astype(np.float64)[:, None]
        self.rest = np.zeros((self.weld_count, 3))
        np.add.at(self.rest, self.weld, np.asarray(rest_co, dtype=np.float64))
        self.rest /= np.maximum(members, 1.0)

    # -------------------------------------------------------------------------
    # Public API

    def apply(self, verts, factors):
        """Move ``verts`` toward their goals, each ``factors`` (strength times falloff, 0..1) of the way."""
        verts = np.asarray(verts, dtype=np.int64).reshape(-1)
        factors = np.asarray(factors, dtype=np.float64).reshape(-1)
        keep = factors > 0.0
        verts, factors = verts[keep], factors[keep]
        if self.nothing_to_do or not len(verts):
            return
        groups = self.weld[verts]
        params = self._prepare(groups)
        removals = {}
        for vert, group, factor, param in zip(verts.tolist(), groups.tolist(), factors.tolist(), params):
            self._read(vert)
            current = self.current[vert]
            if self._leave(current, param):
                continue
            weights = self._blend(vert, group, current, param, factor)
            if weights is not None:
                self._write(vert, weights, removals)
        self._remove(removals)

    # -------------------------------------------------------------------------
    # For subclasses

    def _prepare(self, groups):
        """Something per vertex of a dab (welded ``groups``), handed to `_leave` and `_goal`."""
        return groups.tolist()

    def _leave(self, current, param):
        """Whether to leave a vertex with ``current`` weights alone."""
        return False

    def _goal(self, vert, group, param, shares):
        """The goal of ``vert`` (welded ``group``) with ``shares`` ({group: fraction} of its unlocked bone
        weights): a list of parts (share, {group or name of a group to create: fraction}), or None."""
        raise NotImplementedError

    def _key(self, name):
        """The group called ``name`` (its name when it doesn't exist yet, so it can be created), or None
        when it is locked or isn't a bone."""
        group = self.name_to_group.get(name)
        if group is None:
            return name if name in self.bone_names else None
        if group in self.locked or group not in self.bone_groups:
            return None
        return group

    def _group_created(self, group, name):
        """A group was added for ``name``."""

    def _group_forgotten(self, group, name):
        """A group created by this stroke was removed again."""

    # -------------------------------------------------------------------------
    # Internals

    def _compose(self, parts, budget, limit):
        """{group or name: fraction} from ``parts``, adding up to 1. Weights too small to keep are
        dropped, and so are the smallest beyond ``limit`` groups; every part worth keeping keeps at least
        its largest group, and its share, however many had to go."""
        entries = sorted(
            ((total * share, part, key)
             for part, (total, part_shares) in enumerate(parts) for key, share in part_shares.items()
             if total * share > 0.0),
            key=lambda entry: (-entry[0], entry[1], str(entry[2])),
        )
        chosen = {}
        for value, part, key in entries:
            if part not in chosen and parts[part][0] * budget > ZERO_WEIGHT:
                chosen[part] = [(value, key)]
        chosen = dict(list(chosen.items())[:limit])
        count = len(chosen)
        for value, part, key in entries:
            if count >= limit or value * budget <= ZERO_WEIGHT:
                break
            members = chosen.setdefault(part, [])
            if all(key != other for _value, other in members):
                members.append((value, key))
                count += 1
        goal = {}
        for part, members in chosen.items():
            total = sum(value for value, _key in members)
            if total <= 1e-12:
                continue
            scale = parts[part][0] / total
            for value, key in members:
                goal[key] = goal.get(key, 0.0) + value * scale
        total = sum(goal.values())
        return {key: share / total for key, share in goal.items()} if total > 1e-12 else None

    def _blend(self, vert, group, current, param, factor):
        """New weights ({group: weight}) for ``vert`` with ``current`` weights, ``factor`` of the way to
        its goal, or None."""
        bone_weights = {g: weight for g, weight in current.items() if weight > 0.0 and g in self.bone_groups}
        total = sum(bone_weights.values())
        locked = {g: weight for g, weight in bone_weights.items() if g in self.locked}
        free = {g: weight for g, weight in bone_weights.items() if g not in self.locked}
        if total <= 1e-9:
            total, free = 1.0, {}  # no bone weights yet: they all come from the goal
        elif self.replace:
            total = 1.0
        budget = total - sum(locked.values())
        if budget <= 1e-9:
            return None
        free_total = sum(free.values())
        shares = {g: weight / free_total for g, weight in free.items()} if free_total > 1e-9 else {}
        limit = max(1, self.max_groups - len(locked))
        parts = self._goal(vert, group, param, shares)
        goal = self._compose(parts, budget, limit) if parts else None
        if goal is None:
            return None

        if self.active is not None:
            # Only the active group moves; the others keep their proportions around it.
            active = self.active
            old = shares.get(active, 0.0)
            new = old + (goal.get(active, 0.0) - old) * factor
            others = {g: share for g, share in shares.items() if g != active}
            if not others:
                others = {g: share for g, share in goal.items() if g != active and isinstance(g, int)}
            rest_total = sum(others.values())
            if rest_total <= 1e-9:
                return None
            mixed = {g: share / rest_total * (1.0 - new) for g, share in others.items()}
            mixed[active] = new
        else:
            mixed = {}
            for key in shares.keys() | goal.keys():
                old = shares.get(key, 0.0)
                mixed[key] = old + (goal.get(key, 0.0) - old) * factor

        # Weights too small to keep are dropped, and so are the smallest beyond the group limit; the
        # others make up for them, so the total of the bone weights stays what it was.
        kept_shares = {key: share for key, share in mixed.items() if share * budget > ZERO_WEIGHT}
        if len(kept_shares) > limit:
            order = sorted(kept_shares, key=lambda key: (key != self.active, -kept_shares[key], str(key)))
            kept_shares = {key: kept_shares[key] for key in order[:limit]}
        norm = sum(kept_shares.values())
        if norm <= 1e-12:
            return None
        weights = {g: 0.0 for g in mixed if not isinstance(g, str)}
        for key, share in kept_shares.items():
            if isinstance(key, str):
                if self.active is not None:
                    continue
                key = self._create_group(key)
            weights[key] = min(share / norm * budget, 1.0)
        return weights

    def _create_group(self, name):
        group = self.obj.vertex_groups.new(name=name)
        # `new()` makes the new group active, which would switch the weights shown in the viewport.
        self._restore_active_group()
        index = group.index
        self.groups.append(group)
        self.name_to_group[name] = index
        self.group_names[index] = name
        self.bone_groups.add(index)
        self.created.append(index)
        self._group_created(index, name)
        return index

    def _forget_group(self, group):
        name = self.group_names.pop(group, None)
        if name is not None and self.name_to_group.get(name) == group:
            del self.name_to_group[name]
        self.bone_groups.discard(group)
        self._group_forgotten(group, name)
