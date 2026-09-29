# SPDX-License-Identifier: GPL-3.0-or-later

"""The Customize+ rig: a hidden copy of an armature that the meshes follow instead while it's on.

Each of its bones copies the original bone's pose (Copy Transforms, armature space) and then adds the
template's change with Action constraints that hold fixed values (see `solver.plan`): propagation before
the pose, in the propagating bone's frame (a scale in the armature's axes at that bone's head, a hidden
helper empty); the bone's own change after it, in its own frame; the root's scale last, around the
armature's origin. Since the copy never passes a bone's change on to its children the way parenting
does, the bones change as in the game, in any pose or animation the original armature gets, and
nothing on the original armature is touched: its pose, animation and rest pose stay the user's.

The game's Bust Size comes last, as its own constraints on the breast bones and the bones below them
("C+ Bust"), so it can follow the panel's slider on its own.

While it's on, each Armature modifier that follows the original is switched off and a copy right after
it ("C+ Armature") follows the rig. Keeping the original modifier first lets Weight Paint mode still
select the original armature's bones, and the add-on's tools still find the armature they expect.
Turning it off removes the copies and switches the originals back on. The rig stays (hidden, a child of
the armature) until it's removed, so turning it on again is instant.
"""

import json

import bpy
import numpy as np
from mathutils import Matrix

from . import solver
from .template import Template

MARK = "mf_cplus"            # custom property on the rig: its template, options and bust size (JSON)
HELPER = "mf_cplus_helper"   # custom property on the rig's helper empties: the rig
MODIFIER = "C+ Armature"
PREFIX = "C+ "                # names of the rig's constraints, actions and helpers
BUST = PREFIX + "Bust"
BUST_FRAME = PREFIX + "Bust Frame"


def is_rig(obj):
    return obj is not None and obj.type == 'ARMATURE' and MARK in obj


def rig_of(armature):
    """The Customize+ rig of ``armature``, or None."""
    if armature is None:
        return None
    for child in armature.children:
        if is_rig(child):
            return child
    return None


def state(rig):
    """(Template, Options) the rig was made from."""
    data = json.loads(rig[MARK])
    return Template.from_json(data["template"]), solver.Options.from_dict(data["options"])


def bust_state(rig):
    """(Bust Size or None, whether it's the Lalafells') the rig shows."""
    bust = json.loads(rig[MARK]).get("bust")
    return (bust["size"], bust["lalafell"]) if bust else (None, False)


def _mark(rig, template, options, bust, lalafell):
    rig[MARK] = json.dumps({
        "template": template.to_json(),
        "options": options.to_dict(),
        "bust": None if bust is None else {"size": bust, "lalafell": lalafell},
    })


def skeleton(armature_data):
    """({bone: parent or None}, {bone: 4x4 rest matrix in armature space})."""
    parents = {bone.name: bone.parent.name if bone.parent else None for bone in armature_data.bones}
    rest = {bone.name: np.array(bone.matrix_local) for bone in armature_data.bones}
    return parents, rest


def _same_rest(rig_data, armature_data):
    if len(rig_data.bones) != len(armature_data.bones):
        return False
    for bone in armature_data.bones:
        other = rig_data.bones.get(bone.name)
        if other is None or (other.parent.name if other.parent else None) != (bone.parent.name if bone.parent else None):
            return False
        if other.use_deform != bone.use_deform:
            return False
        if not np.allclose(np.array(other.matrix_local), np.array(bone.matrix_local), atol=1e-6):
            return False
    return True


def _collection_for(context, armature):
    """A collection of ``armature`` that's in the current view layer, or the scene's."""
    def included(layer_collection, target):
        # A collection disabled in viewports isn't evaluated, nor are its children: the rig would get no pose.
        if layer_collection.exclude or layer_collection.collection.hide_viewport:
            return False
        if layer_collection.collection == target:
            return True
        return any(included(child, target) for child in layer_collection.children)

    for collection in armature.users_collection:
        if collection == context.scene.collection or included(context.view_layer.layer_collection, collection):
            return collection
    return context.scene.collection


def _create(context, armature):
    data = armature.data.copy()
    data.name = PREFIX + armature.data.name
    rig = bpy.data.objects.new(PREFIX + armature.name, data)
    _collection_for(context, armature).objects.link(rig)
    rig.parent = armature
    rig.parent_type = 'OBJECT'
    rig.matrix_parent_inverse = Matrix.Identity(4)
    rig.matrix_basis = Matrix.Identity(4)
    rig.rotation_mode = 'QUATERNION'
    data.pose_position = 'POSE'
    for bone in data.bones:
        # Constraints work in the bones' own frames: those must be plain parent x rest x pose.
        bone.inherit_scale = 'FULL'
        bone.use_inherit_rotation = True
        bone.use_local_location = True
    # The pose (and so the bones' constraints) only exists once Blender has evaluated the object.
    context.view_layer.update()
    rig.hide_select = True
    rig.hide_viewport = True
    return rig


def _remove_rig(rig):
    data = rig.data
    _clear(rig)
    bpy.data.objects.remove(rig)
    if data.users == 0:
        bpy.data.armatures.remove(data)


def _clear(rig):
    """Remove the rig's constraints, the actions only they used and its helpers."""
    actions = set()
    if rig.pose is not None:
        for bone in rig.pose.bones:
            for constraint in list(bone.constraints):
                if constraint.type == 'ACTION' and constraint.action is not None:
                    actions.add(constraint.action)
                bone.constraints.remove(constraint)
    for action in actions:
        if action.users == 0:
            bpy.data.actions.remove(action)
    for obj in [obj for obj in bpy.data.objects if obj.get(HELPER) == rig]:
        bpy.data.objects.remove(obj)


def _helper(rig, armature, bone):
    """An empty on the armature's axes that stays at ``bone``'s head: where a propagating bone scales
    the bones below it in the game (along the model's axes, about its position)."""
    helper = bpy.data.objects.new("{:s}{:s} {:s}".format(PREFIX, armature.name, bone), None)
    for collection in rig.users_collection:
        collection.objects.link(helper)
    helper.parent = armature
    helper.parent_type = 'OBJECT'
    helper.matrix_parent_inverse = Matrix.Identity(4)
    helper.matrix_basis = Matrix.Identity(4)
    helper.empty_display_size = 0.02
    follow = helper.constraints.new('COPY_LOCATION')
    follow.name = PREFIX + "Head"
    follow.target = armature
    follow.subtarget = bone
    helper.hide_select = True
    helper.hide_viewport = True
    helper[HELPER] = rig
    return helper


def _new_action(name, channels):
    """An action holding fixed (position, quaternion, scale) values per bone. Returns (action, slot)."""
    action = bpy.data.actions.new(name)
    if hasattr(action, "layers") and hasattr(action, "slots"):  # Blender 4.4+: slotted actions
        slot = action.slots.new(id_type='OBJECT', name="C+")
        bag = action.layers.new("C+").strips.new(type='KEYFRAME').channelbag(slot, ensure=True)
        groups = {}

        def new_curve(path, index, group):
            try:
                return bag.fcurves.new(path, index=index, group_name=group)
            except TypeError:
                curve = bag.fcurves.new(path, index=index)
                if group not in groups:
                    groups[group] = bag.groups.new(group)
                curve.group = groups[group]
                return curve
    else:
        slot = None

        def new_curve(path, index, group):
            return action.fcurves.new(path, index=index, action_group=group)

    for bone, (position, rotation, scale) in channels.items():
        base = 'pose.bones["{:s}"].'.format(bpy.utils.escape_identifier(bone))
        for prop, values in (("location", position), ("rotation_quaternion", rotation), ("scale", scale)):
            for index, value in enumerate(values):
                curve = new_curve(base + prop, index, bone)
                curve.keyframe_points.insert(0.0, float(value), options={'FAST'})
    return action, slot


def _action_constraint(bone, name, action, slot, mix, space, space_object=None, space_bone=""):
    constraint = bone.constraints.new('ACTION')
    constraint.name = name
    constraint.use_eval_time = True
    constraint.eval_time = 0.0
    constraint.frame_start = 0
    constraint.frame_end = 1
    constraint.action = action
    if slot is not None and hasattr(constraint, "action_slot"):
        constraint.action_slot = slot
    constraint.mix_mode = mix
    if space == 'CUSTOM':
        constraint.space_object = space_object
        constraint.space_subtarget = space_bone
    constraint.owner_space = space
    return constraint


def _setup(rig, armature, result):
    """Constraints and actions of the rig for a `solver.Plan`."""
    bones = rig.pose.bones
    for bone in bones:
        bone.rotation_mode = 'QUATERNION'
        bone.location = (0.0, 0.0, 0.0)
        bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        bone.scale = (1.0, 1.0, 1.0)
        follow = bone.constraints.new('COPY_TRANSFORMS')
        follow.name = PREFIX + "Follow"
        follow.target = armature
        follow.subtarget = bone.name
        follow.target_space = 'POSE'
        follow.owner_space = 'POSE'
    # The innermost propagating bone's changes first (see `solver.plan`): in that bone's frame, or for
    # a scale in the armature's axes at its head (a helper empty).
    for source, steps, below in reversed(result.carries):
        helper = None
        for index, (space, values) in enumerate(steps):
            suffix = " {:d}".format(index + 1) if len(steps) > 1 else ""
            action, slot = _new_action("{:s}{:s} {:s}{:s}".format(PREFIX, armature.name, source, suffix),
                                       {name: values for name in below})
            if space == 'AXES':
                helper = helper or _helper(rig, armature, source)
                space_object, space_bone = helper, ""
            else:
                space_object, space_bone = armature, source
            for name in below:
                _action_constraint(bones[name], PREFIX + "Carry " + source + suffix, action, slot,
                                   'BEFORE_FULL', 'CUSTOM', space_object, space_bone)
    if result.own:
        action, slot = _new_action(PREFIX + armature.name, result.own)
        for name in result.own:
            _action_constraint(bones[name], PREFIX + "Own", action, slot, 'AFTER_FULL', 'LOCAL')
    frames = {name: (np.zeros(3), solver.quaternion(frame), np.ones(3)) for name, frame in result.frames.items()}
    if frames:
        # Changes a position, turn and scale can't give (such as a scale along axes the bone doesn't
        # have): the own change is followed by a turn.
        action, slot = _new_action(PREFIX + armature.name + " Frames", frames)
        for name in frames:
            _action_constraint(bones[name], PREFIX + "Frame", action, slot, 'AFTER_FULL', 'LOCAL')
    _setup_bust(rig, armature, result.bust, result.bust_frames)
    if result.root is not None:
        action, slot = _new_action(PREFIX + armature.name + " Root", {bone.name: result.root for bone in bones})
        for bone in bones:
            _action_constraint(bone, PREFIX + "Root", action, slot, 'BEFORE_FULL', 'POSE')


def _setup_bust(rig, armature, bust, turns):
    """The bust size's constraints ({bone: values}, {bone: turn}, see `solver.bust_plan`), after the bones'
    other changes (a change after the pose in the bone's frame doesn't depend on the root's, before it)."""
    bones = rig.pose.bones
    if bust:
        action, slot = _new_action(PREFIX + armature.name + " Bust", bust)
        for name in bust:
            _action_constraint(bones[name], BUST, action, slot, 'AFTER_FULL', 'LOCAL')
    if turns:
        frames = {name: (np.zeros(3), solver.quaternion(turn), np.ones(3)) for name, turn in turns.items()}
        action, slot = _new_action(PREFIX + armature.name + " Bust Frames", frames)
        for name in frames:
            _action_constraint(bones[name], BUST_FRAME, action, slot, 'AFTER_FULL', 'LOCAL')


def _clear_bust(rig):
    """Remove the bust size's constraints and actions."""
    actions = set()
    for bone in rig.pose.bones:
        for constraint in [c for c in bone.constraints if c.name in (BUST, BUST_FRAME)]:
            if constraint.action is not None:
                actions.add(constraint.action)
            bone.constraints.remove(constraint)
    for action in actions:
        if action.users == 0:
            bpy.data.actions.remove(action)


def build(context, armature, template, options, bust=None):
    """Make or update the rig of ``armature`` for ``template`` and Bust Size ``bust`` (None: none).
    Returns (rig, `solver.Plan`)."""
    parents, rest = skeleton(armature.data)
    result = solver.plan(parents, rest, template, options, bust)
    rig = rig_of(armature)
    if rig is not None and (not _same_rest(rig.data, armature.data) or rig.pose is None):
        # The armature's rest pose changed: start over (the caller switches the meshes over again).
        disconnect(armature)
        _remove_rig(rig)
        rig = None
    if rig is None:
        rig = _create(context, armature)
    _clear(rig)
    _setup(rig, armature, result)
    _mark(rig, template, options, bust, result.lalafell)
    rig.data.update_tag()
    rig.update_tag()
    return rig, result


def ensure(context, armature):
    """The rig of ``armature``, rebuilt from its own template and bust size if the armature's rest pose
    changed since. None without one."""
    rig = rig_of(armature)
    if rig is None:
        return None
    if rig.pose is None or not _same_rest(rig.data, armature.data):
        template, options = state(rig)
        rig, _result = build(context, armature, template, options, bust_state(rig)[0])
    return rig


def is_stale(armature):
    """Whether the rig of ``armature`` was made for another rest pose or other bones than the armature has
    now. It shows the old ones until it's made again (Apply, or turning it off and on)."""
    rig = rig_of(armature)
    return rig is not None and (rig.pose is None or not _same_rest(rig.data, armature.data))


def set_bust(context, armature, bust):
    """Show Bust Size ``bust`` (None: none) on the rig of ``armature``, keeping the rest of it. Returns the
    rig, or None without one."""
    rig = rig_of(armature)
    if rig is None:
        return None
    template, options = state(rig)
    if rig.pose is None or not _same_rest(rig.data, armature.data):
        on = is_on(armature)
        rig, _result = build(context, armature, template, options, bust)
        if on:
            connect(armature, rig)
        return rig
    _clear_bust(rig)
    lalafell = False
    if bust is not None:
        parents, rest = skeleton(armature.data)
        _scale, lalafell, values, turns = solver.bust_plan(parents, solver.detect_axes(rest), bust)
        _setup_bust(rig, armature, values, turns)
    _mark(rig, template, options, bust, lalafell)
    rig.update_tag()
    return rig


# -----------------------------------------------------------------------------
# Switching the meshes over

def _followers(armature):
    """(object, modifier) of every Armature modifier following ``armature`` that can be changed."""
    for obj in bpy.data.objects:
        if obj.library is not None or obj.override_library is not None or not hasattr(obj, "modifiers"):
            continue
        for modifier in obj.modifiers:
            if modifier.type == 'ARMATURE' and modifier.object == armature:
                yield obj, modifier


def _is_copy(modifier, rig):
    """Whether ``modifier`` is a copy made for ``rig``, or one left behind by a rig deleted by hand."""
    if modifier is None or modifier.type != 'ARMATURE':
        return False
    if rig is not None and modifier.object == rig:
        return True
    return modifier.object is None and modifier.name.startswith(MODIFIER)


def _pairs(armature, rig):
    """(object, modifier following ``armature``, its copy right after it or None)."""
    for obj, modifier in list(_followers(armature)):
        modifiers = obj.modifiers
        index = modifiers.find(modifier.name)
        following = modifiers[index + 1] if index + 1 < len(modifiers) else None
        yield obj, modifier, following if _is_copy(following, rig) else None


def is_on(armature):
    rig = rig_of(armature)
    if rig is None:
        return False
    return any(modifier.show_viewport for _obj, modifier in _followers(rig))


def connect(armature, rig):
    """Switch every mesh that follows ``armature`` over to ``rig``. Returns how many."""
    count = 0
    for obj, modifier, copy in list(_pairs(armature, rig)):
        modifiers = obj.modifiers
        if copy is None:
            if not modifier.show_viewport:
                continue  # switched off by the user
            index = modifiers.find(modifier.name)
            copy = modifiers.new(MODIFIER, 'ARMATURE')
            for prop in ("use_vertex_groups", "use_bone_envelopes", "use_deform_preserve_volume",
                         "use_multi_modifier", "vertex_group", "invert_vertex_group", "show_in_editmode",
                         "show_on_cage", "show_render"):
                setattr(copy, prop, getattr(modifier, prop))
            copy.show_expanded = False
            _move(obj, modifiers.find(copy.name), index + 1)
        copy.object = rig
        copy.show_viewport = True
        modifier.show_viewport = False
        modifier.show_render = False
        count += 1
    return count


def disconnect(armature):
    """Switch the meshes back from the rig to ``armature``. Returns how many."""
    rig = rig_of(armature)
    count = 0
    for obj, modifier, copy in list(_pairs(armature, rig)):
        if copy is None:
            continue
        modifier.show_viewport = True
        modifier.show_render = copy.show_render
        obj.modifiers.remove(copy)
        count += 1
    if rig is not None:
        # Copies whose own modifier was deleted meanwhile become that modifier again.
        for obj, copy in list(_followers(rig)):
            copy.object = armature
            copy.name = "Armature"
            copy.show_viewport = True
            count += 1
    return count


def missing_meshes(armature):
    """Names of objects that follow ``armature`` but not its rig while it's on (added since)."""
    rig = rig_of(armature)
    if rig is None:
        return []
    return [obj.name for obj, modifier, copy in _pairs(armature, rig) if copy is None and modifier.show_viewport]


def followers(armature):
    """Names of the objects deformed by ``armature`` (directly or through its rig)."""
    rig = rig_of(armature)
    names = {obj.name for obj, _modifier in _followers(armature)}
    if rig is not None:
        names |= {obj.name for obj, _modifier in _followers(rig)}
    return sorted(names)


def _move(obj, from_index, to_index):
    if from_index == to_index:
        return
    modifiers = obj.modifiers
    if hasattr(modifiers, "move"):
        modifiers.move(from_index, to_index)
        return
    name = modifiers[from_index].name
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.modifier_move_to_index(modifier=name, index=to_index)


def remove(armature):
    """Switch everything back and delete the rig with its actions. Returns whether there was one."""
    rig = rig_of(armature)
    count = disconnect(armature)
    if rig is None:
        return count > 0
    _remove_rig(rig)
    return True
