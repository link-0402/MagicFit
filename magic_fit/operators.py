# SPDX-License-Identifier: GPL-3.0-or-later

import math
import time

import bmesh
import bpy
import gpu
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatVectorProperty, IntProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_line_line, intersect_line_plane

from . import (facedata, facing, fitting, goals, hairing, heeling, lineup, moving, painting, relaxing, resizing,
               skirting, smoothing, straighten)
from .cplus import rig as cplus_rig
from .properties import COMING_MODES

# Cap on dabs per mouse event, so a fast flick can't stall the viewport.
MAX_DABS_PER_EVENT = 64
UNPOSED_MESSAGE = (
    "The mesh shows its rest pose: Straighten fixes what sags in the pose you see. Apply a Customize+ "
    "template (or scale the bones) first"
)


def check_ready(context, settings):
    """Return why the brush can't paint right now, or None when it can."""
    obj = context.object
    if obj is None or obj.type != 'MESH':
        return "The painted object must be a mesh"
    if settings.mode in COMING_MODES:
        return "{:s} weights come in a later version".format(COMING_MODES[settings.mode])
    if settings.mode == 'STRAIGHTEN':
        return _check_straighten_ready(obj, settings)
    if settings.mode == 'SKIRT':
        return check_skirt_ready(obj, settings)
    if settings.mode == 'HEELS':
        return check_heels_ready(obj, settings)
    if settings.mode == 'HAIR':
        return check_hair_ready(obj, settings)
    if settings.mode == 'FACE':
        return check_face_ready(obj, settings)
    if settings.mode == 'SMOOTH':
        return check_smooth_ready(obj, settings)
    target = settings.target
    if target is None:
        return "Pick a target mesh to copy weights from"
    if target.type != 'MESH':
        return "The target must be a mesh"
    if target == obj:
        return "The target must be a different object"
    if not target.vertex_groups:
        return "Target '{:s}' has no vertex groups".format(target.name)
    if settings.group_mode == 'ACTIVE':
        group = obj.vertex_groups.active
        if group is None:
            return "No active vertex group"
        if group.lock_weight:
            return "Vertex group '{:s}' is locked".format(group.name)
        if target.vertex_groups.get(group.name) is None:
            return "Target has no vertex group '{:s}'".format(group.name)
    return None


def _check_straighten_ready(obj, settings):
    modifier = straighten.deforming_armature(obj)
    if modifier is None:
        return "Straighten needs an Armature modifier"
    armature = modifier.object
    bones = painting.deform_bone_names(obj)
    if not any(group.name in bones for group in obj.vertex_groups):
        return "No vertex group matches a bone of '{:s}'".format(armature.name)
    body = settings.target
    if body is None:
        return "Pick the body the mesh rests on"
    if body.type != 'MESH':
        return "The body must be a mesh"
    if body == obj:
        return "The body must be a different object"
    if settings.group_mode == 'ACTIVE':
        group = obj.vertex_groups.active
        if group is None:
            return "No active vertex group"
        if group.lock_weight:
            return "Vertex group '{:s}' is locked".format(group.name)
        if group.name not in bones:
            return "'{:s}' is not a deforming bone of '{:s}'".format(group.name, armature.name)
    return None


def check_skirt_ready(obj, settings):
    """Why ``obj`` can't get skirt weights with the brush settings, or None when it can."""
    modifier = straighten.deforming_armature(obj)
    if modifier is None:
        return "Skirt needs an Armature modifier"
    armature = modifier.object
    if not skirting.has_skirt_bones(armature):
        return "Armature '{:s}' has no skirt bones (j_sk_...)".format(armature.name)
    body = settings.target
    if settings.skirt_skin_weight > 0.0:
        if body is None:
            return "Pick the body the skirt hangs around, or set Skin Weight to 0"
        if body.type != 'MESH':
            return "The body must be a mesh"
        if body == obj:
            return "The body must be a different object"
    if settings.group_mode == 'ACTIVE':
        group = obj.vertex_groups.active
        if group is None:
            return "No active vertex group"
        if group.lock_weight:
            return "Vertex group '{:s}' is locked".format(group.name)
        if group.name not in painting.deform_bone_names(obj):
            return "'{:s}' is not a deforming bone of '{:s}'".format(group.name, armature.name)
    return None


def check_heels_ready(obj, settings):
    """Why ``obj`` can't get heel weights with the brush settings, or None when it can."""
    modifier = straighten.deforming_armature(obj)
    if modifier is None:
        return "Heels needs an Armature modifier"
    armature = modifier.object
    if not heeling.has_foot_bones(armature):
        return "Armature '{:s}' has no foot bones (j_asi_c, j_asi_d)".format(armature.name)
    if settings.group_mode == 'ACTIVE':
        group = obj.vertex_groups.active
        if group is None:
            return "No active vertex group"
        if group.lock_weight:
            return "Vertex group '{:s}' is locked".format(group.name)
        if group.name not in painting.deform_bone_names(obj):
            return "'{:s}' is not a deforming bone of '{:s}'".format(group.name, armature.name)
    return None


def check_smooth_ready(obj, settings):
    """Why ``obj`` can't be smoothed with the brush settings, or None when it can."""
    smoothed = smoothing.smoothed_groups(obj)
    if not smoothed:
        modifier = straighten.deforming_armature(obj)
        if modifier is not None:
            return "No vertex group matches a bone of '{:s}'".format(modifier.object.name)
        return "The mesh has no vertex groups to smooth"
    if settings.group_mode == 'ACTIVE':
        group = obj.vertex_groups.active
        if group is None:
            return "No active vertex group"
        if group.lock_weight:
            return "Vertex group '{:s}' is locked".format(group.name)
        if group.index not in smoothed:
            return "'{:s}' is not a bone weight of '{:s}'".format(group.name, obj.name)
    elif all(obj.vertex_groups[index].lock_weight for index in smoothed):
        return "Every bone weight group of the mesh is locked"
    return None


def _hair_reference():
    """The hair reference data, or None when it can't be read."""
    try:
        return hairing.Reference.get()
    except (OSError, KeyError, ValueError):
        return None


def check_hair_ready(obj, settings):
    """Why the Hair brush can't paint ``obj``, or None when it can."""
    modifier = straighten.deforming_armature(obj)
    if modifier is None:
        return "Give the hair Hair Weights first: the brush needs its armature with the hair bones"
    race, number = hairing.object_skeleton(obj, settings)
    if number is None:
        return "Give the hair Hair Weights first, or pick its skeleton"
    reference = _hair_reference()
    if reference is None:
        return "The hair reference data (hair_reference.npz) is missing: reinstall the add-on"
    if number and not reference.has_skeleton(race, number):
        return "{:s} has no hair skeleton {:s}".format(hairing.RACE_NAMES[race], hairing.skeleton_label(number))
    armature = modifier.object
    bones = armature.data.bones
    if any(bones.get(bone.name) is None for bone in reference.rig(race, number).hair_bones()):
        return "Armature '{:s}' lacks the hair bones of {:s}: give the hair Hair Weights".format(
            armature.name, hairing.skeleton_label(number))
    if settings.group_mode == 'ACTIVE':
        group = obj.vertex_groups.active
        if group is None:
            return "No active vertex group"
        if group.lock_weight:
            return "Vertex group '{:s}' is locked".format(group.name)
        if group.name not in painting.deform_bone_names(obj):
            return "'{:s}' is not a deforming bone of '{:s}'".format(group.name, armature.name)
    return None


def face_reference(operator, context):
    """The game's faces for ``operator``: the saved ones, or read from the game first (a few seconds, with the
    progress on the mouse cursor). None, reported, when they can't be had."""
    reference = facing.Reference.loaded()
    if reference is not None:
        return reference
    wm = context.window_manager
    wm.progress_begin(0, 100)
    try:
        started = time.perf_counter()
        reference = facing.Reference.get(lambda share: wm.progress_update(int(share * 100)))
    except facedata.FaceDataError as error:
        operator.report({'ERROR'}, str(error))
        return None
    finally:
        wm.progress_end()
    operator.report({'INFO'}, "Read the game's faces from the game ({:.1f} s), once per game update".format(
        time.perf_counter() - started))
    return reference


SHAPE_MATCH = ("No name like c0801f0002 among the meshes, so the face was matched by its shape: {:s}. The race is "
               "usually right, the face number often not: pick the face the mod replaces (Game Face: Pick), since "
               "the game moves it with that face's skeleton")


def check_face_ready(obj, settings):
    """Why the Face brush can't paint ``obj``, or None when it can."""
    problem = facing.face_data_problem()
    if problem is not None:
        return problem
    if straighten.deforming_armature(obj) is None:
        return "Give the face Face Weights first: the brush needs its armature with the face bones"
    reference = facing.Reference.loaded()
    if (reference is not None and settings.face_source == 'PICK'
            and not reference.has_face(settings.face_race, settings.face_number)):
        return "The game has no face {:s}".format(facing.face_title(settings.face_race, settings.face_number))
    if settings.group_mode == 'ACTIVE':
        group = obj.vertex_groups.active
        if group is None:
            return "No active vertex group"
        if group.lock_weight:
            return "Vertex group '{:s}' is locked".format(group.name)
        if group.name not in painting.deform_bone_names(obj):
            return "'{:s}' is not a deforming bone of '{:s}'".format(
                group.name, straighten.deforming_armature(obj).object.name)
    return None


def face_targets(context):
    """The meshes the face tools work on: the active one and the selected ones, the active first; each
    mesh data once."""
    active = context.object
    objects = [active] if active is not None else []
    objects += [obj for obj in context.selected_objects if obj is not active]
    result = []
    seen = set()
    for obj in objects:
        if obj.type != 'MESH' or obj.data.as_pointer() in seen:
            continue
        seen.add(obj.data.as_pointer())
        result.append(obj)
    return result


def check_face_tools_ready(context):
    """Why Face Weights and the face repairs can't be used right now, or None when they can."""
    problem = facing.face_data_problem()
    if problem is not None:
        return problem
    if context.mode not in {'OBJECT', 'PAINT_WEIGHT'}:
        return "Use the face tools in Object or Weight Paint mode"
    targets = face_targets(context)
    if not targets:
        return "Select the face and its parts (eyes, lashes, teeth, piercings)"
    for target in targets:
        if target.library is not None or target.data.library is not None:
            return "'{:s}' is linked from another file".format(target.name)
    reference = facing.Reference.loaded()
    settings = context.scene.magic_fit
    if (reference is not None and settings.face_source == 'PICK'
            and not reference.has_face(settings.face_race, settings.face_number)):
        return "The game has no face {:s}".format(facing.face_title(settings.face_race, settings.face_number))
    armatures = {armature for armature in map(hairing.weighted_armature, targets) if armature is not None}
    if len(armatures) > 1:
        return "The face and its parts are deformed by different armatures"
    for armature in armatures:
        if armature.library is not None or armature.data.library is not None:
            return "Armature '{:s}' is linked from another file".format(armature.name)
    if settings.group_mode == 'ACTIVE':
        return "The face tools give all bone weights: set Groups to All Groups (Current Group is for the brush)"
    return None


def pose_armature(context):
    """The armature Test Pose moves: the active object if it's one, else the one deforming the active mesh
    (while Customize+ is on, the armature its hidden rig follows)."""
    obj = context.object
    if obj is None:
        return None
    if obj.type == 'ARMATURE':
        return obj
    return hairing.weighted_armature(obj) if obj.type == 'MESH' else None


def check_face_pose_ready(context):
    """Why Test Pose can't be used right now, or None when it can."""
    problem = facing.face_data_problem()
    if problem is not None:
        return problem
    if context.mode not in {'OBJECT', 'PAINT_WEIGHT', 'POSE'}:
        return "Pose the face in Object, Weight Paint or Pose mode"
    armature = pose_armature(context)
    if armature is None:
        if context.object is not None and context.object.type == 'MESH':
            return "Give the face Face Weights first: the poses move the face bones of its armature"
        return "Make the face (or its armature) active"
    if not any(bone.name.startswith(facing.UPPER_LID) for bone in armature.data.bones):
        return "Armature '{:s}' has no face bones: give the face Face Weights first".format(armature.name)
    return None


def hair_targets(context, settings):
    """The meshes the Hair Weights button works on: the active one, or every selected one with Apply to
    All Selected; each mesh data once."""
    objects = list(context.selected_objects) if settings.hair_apply_to_selected else [context.object]
    result = []
    seen = set()
    for obj in objects:
        if obj is None or obj.type != 'MESH' or obj.data.as_pointer() in seen:
            continue
        seen.add(obj.data.as_pointer())
        result.append(obj)
    return result


def hair_parts(context, targets, armature, race):
    """The hair's other parts: meshes that aren't ``targets`` but were given hair weights for ``race``
    before, deformed by ``armature`` or, when the targets have none yet, sharing a collection with them."""
    data = {target.data.as_pointer() for target in targets}
    collections = {collection for target in targets for collection in target.users_collection}
    result = []
    for obj in context.scene.objects:
        if obj.type != 'MESH' or obj.data.as_pointer() in data:
            continue
        state = getattr(obj, "magic_fit_hair", None)
        if state is None or state.race != race or state.skeleton < 0:
            continue
        own = hairing.weighted_armature(obj)
        if own is None:
            continue
        if armature is not None and own != armature:
            continue
        if armature is None and not collections.intersection(obj.users_collection):
            continue
        result.append(obj)
    return result


def check_hair_weights_ready(context):
    """Why the Hair Weights button can't be used right now, or None when it can."""
    if context.mode not in {'OBJECT', 'PAINT_WEIGHT'}:
        return "Give hair weights in Object or Weight Paint mode"
    settings = context.scene.magic_fit
    obj = context.object
    if not settings.hair_apply_to_selected and (obj is None or obj.type != 'MESH'):
        return "Make the hair mesh active"
    targets = hair_targets(context, settings)
    if not targets:
        return "Select the hair meshes"
    for target in targets:
        if target.library is not None or target.data.library is not None:
            return "'{:s}' is linked from another file".format(target.name)
    reference = _hair_reference()
    if reference is None:
        return "The hair reference data (hair_reference.npz) is missing: reinstall the add-on"
    race = settings.hair_race
    number = settings.hair_skeleton
    if settings.hair_skeleton_mode == 'PICK' and number and not reference.has_skeleton(race, number):
        return "{:s} has no hair skeleton {:s}".format(hairing.RACE_NAMES[race], hairing.skeleton_label(number))
    armatures = {armature for armature in map(hairing.weighted_armature, targets) if armature is not None}
    if len(armatures) > 1:
        return "The hair meshes are deformed by different armatures"
    for armature in armatures:
        if armature.library is not None or armature.data.library is not None:
            return "Armature '{:s}' is linked from another file".format(armature.name)
    if settings.group_mode == 'ACTIVE':
        return "Hair Weights gives all bone weights: set Groups to All Groups (Current Group is for the brush)"
    return None


# Per whole-mesh button: (its Apply to All Selected setting, readiness check, what it gives, the engine
# module and its error, whether the body is left out).
WHOLE_MESH = {
    'SKIRT': ("skirt_apply_to_selected", check_skirt_ready, "skirt weights", skirting, skirting.SkirtError, True),
    'HEELS': ("heels_apply_to_selected", check_heels_ready, "heel weights", heeling, heeling.HeelsError, False),
}


def whole_mesh_targets(context, settings, kind):
    """The meshes the Skirt Weights or Heel Weights button (``kind`` 'SKIRT' or 'HEELS') works on: the
    active one, or every selected one with Apply to All Selected; each mesh data once, and for Skirt
    never the body."""
    prop, _check, _what, _engine, _error, no_body = WHOLE_MESH[kind]
    if getattr(settings, prop):
        objects = list(context.selected_objects)
    else:
        objects = [context.object]
    body = settings.target if no_body else None
    result = []
    seen = set()
    for obj in objects:
        if obj is None or obj.type != 'MESH' or obj == body or obj.data.as_pointer() in seen:
            continue
        seen.add(obj.data.as_pointer())
        result.append(obj)
    return result


def check_whole_mesh_ready(context, kind):
    """Why the Skirt Weights or Heel Weights button can't be used right now, or None when it can."""
    prop, check, what, _engine, _error, no_body = WHOLE_MESH[kind]
    if context.mode not in {'OBJECT', 'PAINT_WEIGHT'}:
        return "Give {:s} in Object or Weight Paint mode".format(what)
    settings = context.scene.magic_fit
    obj = context.object
    if not getattr(settings, prop):
        if obj is None or obj.type != 'MESH':
            return "Make the mesh to give {:s} active".format(what)
        if no_body and obj == settings.target:
            return "Make a mesh other than the body active"
    targets = whole_mesh_targets(context, settings, kind)
    if not targets:
        return "Select the meshes to give {:s}".format(what)
    for target in targets:
        problem = check(target, settings)
        if problem is not None:
            return problem if len(targets) == 1 else "'{:s}': {:s}".format(target.name, problem)
    return None


def edited_meshes(context, body=None):
    """The meshes in Edit Mode, one object per mesh data, leaving out ``body``."""
    objects = getattr(context, "objects_in_mode_unique_data", None)
    if objects is None:
        objects = [obj for obj in context.view_layer.objects if obj.mode == 'EDIT']
    result = []
    seen = set()
    for obj in objects:
        if obj.type == 'MESH' and obj != body and obj.data.as_pointer() not in seen:
            seen.add(obj.data.as_pointer())
            result.append(obj)
    return result


def check_fit_ready(context):
    """Return why the Body Fit brush can't sculpt right now, or None when it can."""
    obj = context.object
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return "Edit a mesh to fit it around the body"
    body = context.scene.magic_fit.target
    if body is None:
        return "Pick the body mesh to fit around"
    if body.type != 'MESH':
        return "The body must be a mesh"
    if not edited_meshes(context, body):
        return "The body must be a different object"
    return None


def resize_targets(context):
    """The meshes Resize works on: those in Edit Mode, else the selected ones; each mesh data once, and
    never the From or To body."""
    settings = context.scene.body_fit_brush
    skip = {settings.resize_from_object, settings.resize_to_object}
    objects = edited_meshes(context) if context.mode == 'EDIT_MESH' else context.selected_objects
    result = []
    seen = set()
    for obj in objects:
        if obj.type != 'MESH' or obj in skip or obj.data.as_pointer() in seen:
            continue
        seen.add(obj.data.as_pointer())
        result.append(obj)
    return result


def resize_keys(settings):
    """The From and To shape keys Resize uses: those picked, with Use Shapekeys, else none."""
    if settings.resize_use_keys:
        return settings.resize_from_key, settings.resize_to_key
    return "", ""


def check_resize_ready(context):
    """Return why Resize can't be used right now, or None when it can."""
    if context.mode not in {'OBJECT', 'EDIT_MESH'}:
        return "Resize in Object or Edit Mode"
    settings = context.scene.body_fit_brush
    body = settings.resize_from_object
    if body is not None and settings.resize_to_object in (None, body) and not settings.resize_use_keys:
        return "Pick a To Body, or turn on Use Shapekeys"
    from_key, to_key = resize_keys(settings)
    problem = resizing.check_shapes(body, from_key, settings.resize_to_object, to_key)
    if problem is not None:
        return problem
    targets = resize_targets(context)
    if not targets:
        return "Select the meshes to resize"
    for obj in targets:
        keys = obj.data.shape_keys
        if keys is not None and not keys.use_relative:
            return "'{:s}' uses absolute shape keys, which Resize can't mix".format(obj.name)
    return None


def _rigid_parts(obj, co, size):
    """Loose parts of ``obj`` (positions ``co``) that Resize moves rigidly: those no larger than ``size``,
    or all of them for a mesh set to Rigid in Weight Transfer. Returns (part per vertex, -1
    where not rigid; part count)."""
    whole = getattr(getattr(obj, "magic_fit_transfer", None), "rigid", False)
    if size <= 0.0 and not whole:
        return np.full(len(co), -1, dtype=np.int64), 0
    edges = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", edges)
    parts, count = resizing.loose_parts(co, edges.reshape(-1, 2))
    rigid = np.ones(count, dtype=bool) if whole else resizing.part_sizes(co, parts, count) <= size
    return np.where(rigid[parts], parts, -1), count


def resize_key_name(context):
    """Name of the shape key Resize adds: the Key Name setting, else the To shape's name."""
    settings = context.scene.body_fit_brush
    if settings.resize_key_name:
        return settings.resize_key_name
    body = settings.resize_from_object
    other = settings.resize_to_object or body
    label = resizing.shape_label(other, resize_keys(settings)[1], body) if other is not None else ""
    return label or "Resized"


def lineup_model(context):
    """The model Line Up moves: the selected meshes and armatures other than the body, the armature that
    bends them, and every mesh that armature bends or carries (not those parented to a bone: they follow
    it). Meshes whose armature was deleted come along with the others of their collection that lost the
    same one. Each mesh data once. Returns (meshes, armatures)."""
    body = context.scene.magic_fit.target
    body_armature = lineup.armature_of(body) if body is not None and body.type == 'MESH' else None
    skip = {body, body_armature}
    armatures = []
    candidates = []
    for obj in context.selected_objects:
        if obj in skip:
            continue
        if obj.type == 'ARMATURE':
            if obj not in armatures:
                armatures.append(obj)
        elif obj.type == 'MESH':
            armature = lineup.armature_of(obj)
            if armature is not None and armature == body_armature:
                continue  # a part of the body
            candidates.append(obj)
            if armature is not None and armature not in armatures:
                armatures.append(armature)
    lost = {(lineup.lost_armature(obj), collection) for obj in candidates for collection in obj.users_collection}
    lost = {(name, collection) for name, collection in lost if name is not None}
    for obj in context.view_layer.objects:
        if obj.type == 'MESH' and obj not in skip and obj not in candidates:
            armature = lineup.armature_of(obj)
            carried = obj.parent in armatures and armature in (None, obj.parent)
            orphan = lineup.lost_armature(obj)
            together = orphan is not None and any((orphan, c) in lost for c in obj.users_collection)
            if armature in armatures or carried or together:
                candidates.append(obj)
    meshes = []
    seen = set()
    for obj in candidates:
        if (obj.parent is not None and obj.parent_type == 'BONE') or obj.data.as_pointer() in seen:
            continue
        seen.add(obj.data.as_pointer())
        meshes.append(obj)
    return meshes, armatures


def check_lineup_ready(context):
    """Return why Line Up can't be used right now, or None when it can."""
    if context.mode != 'OBJECT':
        return "Line up in Object Mode"
    body = context.scene.magic_fit.target
    if body is None:
        return "Pick the body to line up with"
    if body.type != 'MESH':
        return "The body must be a mesh"
    if lineup.armature_of(body) is None and not body.vertex_groups:
        return "Body '{:s}' needs an armature".format(body.name)
    meshes, armatures = lineup_model(context)
    if len(armatures) > 1:
        return "Line up one model at a time ('{:s}' and '{:s}')".format(armatures[0].name, armatures[1].name)
    if not meshes:
        return "Select the model to line up, its meshes or armature"
    if armatures and lineup.is_posed(armatures[0]):
        return "'{:s}' is posed: clear its pose first".format(armatures[0].name)
    return None


def lineup_summary(context):
    """What Line Up would move, in a few words, or None."""
    meshes, armatures = lineup_model(context)
    if not meshes:
        return None
    count = "{:d} meshes".format(len(meshes)) if len(meshes) > 1 else "'{:s}'".format(meshes[0].name)
    if armatures:
        return "{:s} of '{:s}'".format(count, armatures[0].name)
    return count


def check_relax_ready(context):
    """Return why the Texture Relax brush can't be used right now, or None when it can."""
    obj = context.object
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return "Edit a mesh to relax its texture"
    meshes = edited_meshes(context)
    if not any(mesh.data.uv_layers.active is not None for mesh in meshes):
        if len(meshes) == 1:
            return "'{:s}' has no UV map".format(meshes[0].name)
        return "None of the edited meshes has a UV map"
    return None


def posed_armature(obj):
    """Name of an armature that bends ``obj`` out of its rest pose, or None."""
    rest = np.identity(4)
    for mod in obj.modifiers:
        arm = mod.object if mod.type == 'ARMATURE' and mod.show_viewport else None
        if arm is None or arm.type != 'ARMATURE' or arm.data.pose_position != 'POSE' or arm.pose is None:
            continue
        if cplus_rig.is_rig(arm):
            return arm.name  # Customize+ bends it through constraints
        if any(not np.allclose(np.array(bone.matrix_basis), rest, atol=1e-6) for bone in arm.pose.bones):
            return arm.name
    return None


def _shown_posed(obj, armature_name):
    """Whether ``obj`` is shown deformed by the armature in Edit Mode."""
    return any(
        mod.type == 'ARMATURE' and mod.show_viewport and mod.show_in_editmode and
        mod.object is not None and mod.object.name == armature_name
        for mod in obj.modifiers
    )


def _hidden_shape_key(obj):
    """Name of a shape key that shapes ``obj`` in Object Mode but isn't shown in Edit Mode, or None."""
    keys = obj.data.shape_keys
    if keys is None or not keys.use_relative or obj.use_shape_key_edit_mode or obj.show_only_shape_key:
        return None
    active = obj.active_shape_key
    for block in keys.key_blocks[1:]:
        if block != active and block.value != 0.0 and not block.mute:
            return block.name
    return None


def fit_notes(context):
    """Reasons the meshes in Edit Mode may not look (to the brush) like they do in Object Mode, each as
    (lines, icon). The brush fits them as displayed in Edit Mode."""
    body = context.scene.magic_fit.target
    if body is None:
        return []
    notes = []
    edited = edited_meshes(context, body)
    posed = posed_armature(body)
    if posed is not None:
        unposed = [obj.name for obj in edited if not _shown_posed(obj, posed)]
        if unposed:
            if cplus_rig.is_rig(bpy.data.objects.get(posed)):
                notes.append((("Customize+ is on, '{:s}' isn't shown with it".format(unposed[0]),
                               "Turn it off, or show its C+ Armature in Edit Mode"), 'ERROR'))
            else:
                notes.append((("Body is posed, '{:s}' isn't shown posed".format(unposed[0]),
                               "Use Rest Position, or its Armature's Edit Mode"), 'ERROR'))
    for obj in edited:
        key = _hidden_shape_key(obj)
        if key is not None:
            notes.append((("Shape key '{:s}' of '{:s}' isn't shown".format(key, obj.name),
                           "Turn on Shape Key Edit Mode to fit with it"), 'INFO'))
    return notes


def _mirror_flips(mesh):
    """Scale vectors for every symmetric copy of a dab, from the mesh's X/Y/Z mirror options."""
    axes = [axis for axis, enabled in enumerate((mesh.use_mirror_x, mesh.use_mirror_y, mesh.use_mirror_z))
            if enabled]
    flips = []
    for combination in range(1, 1 << len(axes)):
        flip = Vector((1.0, 1.0, 1.0))
        for bit, axis in enumerate(axes):
            if combination & (1 << bit):
                flip[axis] = -1.0
        flips.append(flip)
    return flips


def _selection_mask(mesh):
    """Vertices the brush may change when face or vertex selection masking is on, else None."""
    if mesh.use_paint_mask_vertex:
        mask = np.zeros(len(mesh.vertices), dtype=bool)
        mesh.vertices.foreach_get("select", mask)
        return mask
    if mesh.use_paint_mask:
        face_select = np.zeros(len(mesh.polygons), dtype=bool)
        mesh.polygons.foreach_get("select", face_select)
        loop_totals = np.zeros(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("loop_total", loop_totals)
        loop_verts = np.zeros(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_verts)
        mask = np.zeros(len(mesh.vertices), dtype=bool)
        mask[loop_verts[np.repeat(face_select, loop_totals)]] = True
        return mask
    return None


class _BrushStroke:
    """Dab placement along the mouse path and 3D view helpers, shared by the brush operators.

    Operators set ``region``, ``rv3d``, ``bvh`` (the surface under the brush, in world space),
    ``matrix``, ``matrix_inv``, ``flips``, ``radius``, ``spacing``, ``last_xy`` and ``last_pressure``,
    and implement ``_dab(xy, pressure)``.
    """

    def _stroke_to(self, xy, pressure):
        """Place dabs along the path from the last dab to ``xy``, spaced by a fraction of the radius."""
        x0, y0 = self.last_xy
        dx = xy[0] - x0
        dy = xy[1] - y0
        length = math.hypot(dx, dy)
        step = max(1.0, self.radius * self.spacing)
        if length < step:
            return
        count = int(length / step)
        if count > MAX_DABS_PER_EVENT:
            count = MAX_DABS_PER_EVENT
            step = length / count
        pressure0 = self.last_pressure
        for i in range(1, count + 1):
            t = i * step / length
            self._dab((x0 + dx * t, y0 + dy * t), pressure0 + (pressure - pressure0) * t)
        t = count * step / length
        self.last_xy = (x0 + dx * t, y0 + dy * t)
        self.last_pressure = pressure0 + (pressure - pressure0) * t

    def _raycast(self, xy):
        origin = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, xy)
        direction = view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, xy)
        location, _normal, _index, _distance = self.bvh.ray_cast(origin, direction)
        return location

    def _world_radius(self, center, radius_px):
        """Brush radius in world units at the depth of ``center``."""
        center_2d = view3d_utils.location_3d_to_region_2d(self.region, self.rv3d, center)
        if center_2d is None:
            return None
        edge = view3d_utils.region_2d_to_location_3d(
            self.region, self.rv3d, (center_2d.x + radius_px, center_2d.y), center,
        )
        return (edge - center).length

    def _mirrored(self, point):
        local = self.matrix_inv @ point
        return [self.matrix @ (local * flip) for flip in self.flips]


class MAGIC_FIT_OT_paint(_BrushStroke, bpy.types.Operator):
    """Paint weights: copy them from the target mesh, straighten creases in the pose, smooth them, or make
    skirt, heel, hair and face weights"""
    bl_idname = "magic_fit.paint"
    bl_label = "Weight Brush Stroke"
    bl_options = {'UNDO', 'BLOCKING'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (
            context.mode == 'PAINT_WEIGHT' and
            obj is not None and obj.type == 'MESH' and
            context.area is not None and context.area.type == 'VIEW_3D' and
            context.region is not None and context.region.type == 'WINDOW'
        )

    def invoke(self, context, event):
        settings = context.scene.magic_fit
        problem = check_ready(context, settings)
        if problem is not None:
            self.report({'WARNING'}, problem)
            return {'CANCELLED'}

        obj = context.object
        self.mesh = obj.data
        self.region = context.region
        self.rv3d = context.region_data
        self.button = event.type

        self.radius = settings.radius
        self.strength = settings.strength
        self.use_pressure_radius = settings.use_pressure_radius
        self.use_pressure_strength = settings.use_pressure_strength
        self.falloff = settings.falloff
        self.falloff_shape = settings.falloff_shape
        self.spacing = settings.spacing / 100.0

        depsgraph = context.evaluated_depsgraph_get()
        geometry = painting.read_geometry(
            obj, depsgraph, normals=settings.use_front_faces_only, surface=True,
        )
        if not geometry.evaluated:
            # The brush then goes by the original mesh: no pose, no shape keys.
            if settings.mode == 'STRAIGHTEN':
                self.report({'WARNING'}, "'{:s}': modifiers change its geometry (Mirror, Solidify...), so Straighten "
                                         "can't see its pose: turn them off in the viewport first".format(obj.name))
                return {'CANCELLED'}
            self.report({'WARNING'}, "'{:s}': modifiers change its geometry (Mirror, Solidify...), so its shape "
                                     "without modifiers and shape keys is used".format(obj.name))
        self.co = geometry.co
        self.bvh = painting.surface_bvh(obj.as_pointer(), geometry.surface_co, geometry.surface_tris)
        self.matrix = geometry.matrix
        self.matrix_inv = geometry.matrix.inverted_safe()
        self.flips = _mirror_flips(self.mesh)

        # Vertices the brush may never touch during this stroke.
        mask = selection = _selection_mask(self.mesh)
        if settings.use_front_faces_only:
            front = self._front_facing(geometry.normals)
            mask = front if mask is None else (mask & front)
        self.mask = mask
        self.screen_co = self._project_all() if self.falloff_shape == 'PROJECTED' else None

        if settings.mode == 'STRAIGHTEN':
            # What sags is what lies deeper as displayed than in the rest pose.
            body = fitting.BodySurface.from_object(settings.target, depsgraph)
            if body is None:
                self.report({'WARNING'}, "The body mesh has no faces")
                return {'CANCELLED'}
            rest, = straighten.rest_geometry(context, (obj,), ('INDEXED',))
            # A strap's back faces sag along with its front: only selection masking limits the span.
            self.stroke = straighten.StraightenStroke(
                obj, rest.co, geometry.co, body,
                group_mode=settings.group_mode,
                allowed=selection,
            )
            if self.stroke.unposed:
                self.report({'WARNING'}, UNPOSED_MESSAGE)
                return {'CANCELLED'}
        elif settings.mode == 'SKIRT':
            # Heights, angles and distances to the body are measured in the rest pose too.
            try:
                self.stroke = skirting.build_stroke(context, obj, settings)
            except skirting.SkirtError as error:
                self.report({'WARNING'}, str(error))
                return {'CANCELLED'}
        elif settings.mode == 'HEELS':
            try:
                self.stroke = heeling.build_stroke(context, obj, settings)
            except heeling.HeelsError as error:
                self.report({'WARNING'}, str(error))
                return {'CANCELLED'}
        elif settings.mode == 'HAIR':
            try:
                self.stroke = hairing.build_stroke(context, obj, settings)
            except hairing.HairError as error:
                self.report({'WARNING'}, str(error))
                return {'CANCELLED'}
        elif settings.mode == 'FACE':
            if face_reference(self, context) is None:
                return {'CANCELLED'}
            try:
                self.stroke = facing.build_stroke(context, obj, settings)
            except facing.FaceError as error:
                self.report({'WARNING'}, str(error))
                return {'CANCELLED'}
        elif settings.mode == 'SMOOTH':
            # Distances are measured in the rest pose too.
            try:
                self.stroke = smoothing.build_stroke(context, obj, settings)
            except smoothing.SmoothError as error:
                self.report({'WARNING'}, str(error))
                return {'CANCELLED'}
        else:
            sampler = painting.TargetSampler(
                settings.target, depsgraph,
                mode=settings.sample_mode,
                max_distance=settings.max_distance if settings.use_max_distance else None,
            )
            tool_settings = context.tool_settings
            self.stroke = painting.CopyStroke(
                obj, settings.target, sampler, self.co,
                group_mode=settings.group_mode,
                fade_unmatched=settings.use_fade_unmatched,
                accumulate=settings.use_accumulate,
                auto_normalize=tool_settings.use_auto_normalize,
                restrict=tool_settings.weight_paint.use_group_restrict,
                bones_only=settings.use_bones_only,
                smooth=settings.auto_smooth if settings.use_auto_smooth else 0.0,
            )

        self.last_xy = (event.mouse_region_x, event.mouse_region_y)
        self.last_pressure = event.pressure
        self._dab(self.last_xy, event.pressure)
        self._update(context)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._stroke_to((event.mouse_region_x, event.mouse_region_y), event.pressure)
            self._update(context)
        elif event.type == self.button and event.value == 'RELEASE':
            return self._finish(context)
        elif event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self.stroke.restore()
            self._update(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        stroke = self.stroke
        if isinstance(stroke, straighten.StraightenStroke):
            stroke.finish()
        self._update(context)
        if isinstance(stroke, straighten.StraightenStroke):
            if stroke.unsupported:
                self.report({'WARNING'}, "Nothing around the sagging part has bone weights to blend from")
            elif stroke.found_nothing:
                self.report({'INFO'}, "Nothing around the brush sags more than in the rest pose")
        elif isinstance(stroke, skirting.SkirtStroke):
            if stroke.unweighted:
                self.report({'WARNING'}, _unweighted_message(stroke, self.mesh))
            elif not stroke.painted_skirt and stroke.painted_above:
                self.report({'INFO'}, "Everything painted is above Blend Start: Skirt only changes what hangs below it")
        elif isinstance(stroke, heeling.HeelsStroke):
            if not stroke.painted_foot and stroke.painted_leg:
                self.report({'INFO'}, "Everything painted is above the ankle: Heels only changes the foot and the ankle")
        if not stroke.modified:
            return {'CANCELLED'}
        created = self.stroke.created_group_names()
        if created:
            self.report({'INFO'}, "Added vertex groups: " + ", ".join(created))
        return {'FINISHED'}

    def _update(self, context):
        if self.stroke.changed:
            self.stroke.changed = False
            self.mesh.update_tag()
            context.area.tag_redraw()

    def _dab(self, xy, pressure):
        radius_px = self.radius * (pressure if self.use_pressure_radius else 1.0)
        strength = self.strength * (pressure if self.use_pressure_strength else 1.0)
        if radius_px < 0.5 or strength <= 0.0:
            return

        hit = self._raycast(xy)
        mirrored = [] if hit is None else self._mirrored(hit)
        radius = None if hit is None else self._world_radius(hit, radius_px)
        if self.falloff_shape == 'SPHERE':
            if not radius:
                return
            distances = [
                np.linalg.norm(self.co - np.array(center), axis=1) / radius
                for center in (hit, *mirrored)
            ]
        else:
            centers = [xy]
            for point in mirrored:
                center = view3d_utils.location_3d_to_region_2d(self.region, self.rv3d, point)
                if center is not None:
                    centers.append(center)
            distances = [
                np.linalg.norm(self.screen_co - np.array(center), axis=1) / radius_px
                for center in centers
            ]

        # Symmetric dabs can overlap: each vertex takes the strongest of them.
        falloff = None
        for t in distances:
            inside = t < 1.0
            if not inside.any():
                continue
            f = np.zeros(len(t))
            f[inside] = painting.falloff_weights(self.falloff, t[inside])
            falloff = f if falloff is None else np.maximum(falloff, f)
        if falloff is None:
            return
        if self.mask is not None:
            falloff[~self.mask] = 0.0
        verts = np.flatnonzero(falloff > 0.0)
        if isinstance(self.stroke, straighten.StraightenStroke):
            # Straighten looks around the dab by its size.
            self.stroke.apply(verts.tolist(), (falloff[verts] * strength).tolist(), radius=radius)
        else:
            self.stroke.apply(verts.tolist(), (falloff[verts] * strength).tolist())

    def _front_facing(self, normals):
        rv3d = self.rv3d
        view_inv = rv3d.view_matrix.inverted()
        if rv3d.is_perspective:
            view_dirs = self.co - np.array(view_inv.translation)
        else:
            view_dirs = np.array(view_inv.to_3x3() @ Vector((0.0, 0.0, -1.0)))
        return np.einsum("ij,ij->i", normals, np.broadcast_to(view_dirs, normals.shape)) < 0.0

    def _project_all(self):
        """Region pixel coordinates of every vertex (far away for vertices behind the view)."""
        persp = np.array(self.rv3d.perspective_matrix, dtype=np.float64)
        clip = self.co @ persp[:3, :3].T + persp[:3, 3]
        w = self.co @ persp[3, :3] + persp[3, 3]
        visible = w > 1e-6
        w = np.where(visible, w, 1.0)
        screen = np.empty((len(self.co), 2))
        screen[:, 0] = (clip[:, 0] / w + 1.0) * 0.5 * self.region.width
        screen[:, 1] = (clip[:, 1] / w + 1.0) * 0.5 * self.region.height
        screen[~visible] = np.inf
        return screen


# To see how a mesh's displayed shape follows its edit positions, these are nudged by this fraction of
# the mesh's size.
PROBE_STEP = 1e-3
# Vertices whose displayed position follows their edit position less than this (relative to the
# object's scale) are left alone: moving them visibly would take huge edits.
MIN_FOLLOW = 0.01


class _EditedPart:
    """One mesh object of a Body Fit stroke: its vertices are ``first`` to ``first + count - 1`` of
    the stroke's combined arrays.

    ``start`` holds the edit positions (object space, the active shape key) when the stroke began and
    ``origin`` the displayed ones (world space). ``inverse`` turns a move of a displayed position into
    the move of the edit position that makes it: one 3x3 matrix, or one per vertex when the mesh is
    displayed deformed in Edit Mode. ``frozen`` marks vertices that can't be moved (or None).
    """

    __slots__ = ("obj", "mesh", "first", "count", "start", "origin", "inverse", "frozen")

    def __init__(self, obj, first, start):
        self.obj = obj
        self.mesh = obj.data
        self.first = first
        self.count = len(start)
        self.start = start
        matrix = np.array(obj.matrix_world, dtype=np.float64)
        self.origin = painting.transform_points(start.astype(np.float64), matrix)
        self.inverse = np.linalg.inv(matrix[:3, :3])
        self.frozen = None

    def to_edit(self, indices, co):
        """Edit positions (object space) that display the part's vertices ``indices`` at ``co``."""
        delta = co - self.origin[indices]
        if self.inverse.ndim == 2:
            return self.start[indices] + delta @ self.inverse.T
        return self.start[indices] + np.einsum("nij,nj->ni", self.inverse[indices], delta)


def _deformed_in_edit_mode(obj):
    """Whether ``obj`` may be displayed in Edit Mode away from its edit positions: with other shape keys
    shown on top of the active one (Shape Key Edit Mode), or modifiers shown in Edit Mode (a posed Armature)."""
    if obj.data.shape_keys is not None and obj.use_shape_key_edit_mode:
        return True
    return any(mod.show_viewport and mod.show_in_editmode for mod in obj.modifiers)


def _displayed_positions(obj, depsgraph, count):
    """World space positions of ``obj``'s vertices as displayed, or None when the displayed mesh doesn't
    line up with the edit mesh (modifiers adding or removing geometry)."""
    obj_eval = obj.evaluated_get(depsgraph)
    try:
        mesh = obj_eval.to_mesh()
    except RuntimeError:
        return None
    try:
        if len(mesh.vertices) != count:
            return None
        co = np.empty(count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        return painting.transform_points(co.reshape(-1, 3).astype(np.float64), obj_eval.matrix_world)
    finally:
        obj_eval.to_mesh_clear()


def _translate(mesh, offset):
    bm = bmesh.from_edit_mesh(mesh)
    bmesh.ops.translate(bm, vec=offset, verts=list(bm.verts))
    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)


def _set_positions(mesh, co):
    bm = bmesh.from_edit_mesh(mesh)
    for vert, position in zip(bm.verts, co.tolist()):
        vert.co = position
    bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=False)


# Per object, how its displayed vertices followed its edit positions at the last probe: (edit
# positions, displayed positions, matrices, inverses, frozen). Reused while the displayed shape is
# exactly what they predict, i.e. nothing but the brush changed it.
_follow_cache = {}


def clear_follow_cache():
    _follow_cache.clear()


def _cached_follow(part, shown, size):
    entry = _follow_cache.get(part.obj.as_pointer())
    if entry is None or len(entry[0]) != part.count:
        return None
    start, previous, matrices, inverse, frozen = entry
    predicted = previous + np.einsum("nij,nj->ni", matrices, part.start.astype(np.float64) - start)
    if np.abs(predicted - shown).max() > 1e-5 * size:
        return None
    return inverse, frozen


def follow_displayed(context, parts):
    """Make Body Fit work on the parts as displayed, when that differs from their edit positions.

    A part's displayed shape can differ in Edit Mode: other shape keys shown on top of the active one,
    or modifiers such as a posed Armature. Then the edit positions are nudged along each axis to see
    how every displayed vertex follows (a 3x3 matrix per vertex, exact for shape keys and armatures),
    and put back exactly as they were. Returns the parts whose displayed mesh doesn't line up with the
    edit mesh (modifiers adding geometry): those are fitted by their edit positions.
    """
    depsgraph = context.evaluated_depsgraph_get()
    probes = {}
    unmatched = []
    for part in parts:
        if not part.count or not _deformed_in_edit_mode(part.obj):
            continue
        shown = _displayed_positions(part.obj, depsgraph, part.count)
        if shown is None:
            unmatched.append(part)
            continue
        size = max(float(np.ptp(part.origin, axis=0).max()), 1e-6)
        if np.abs(shown - part.origin).max() <= 1e-6 * size:
            continue  # displayed right where it is edited (e.g. an armature in its rest pose)
        cached = _cached_follow(part, shown, size)
        if cached is not None:
            part.origin = shown
            part.inverse, part.frozen = cached
            continue
        probes[part] = (shown, PROBE_STEP * max(float(np.ptp(part.start, axis=0).max()), 1e-3))
    if not probes:
        return unmatched

    follow = {part: np.empty((part.count, 3, 3)) for part in probes}
    try:
        for axis in range(3):
            for part, (_shown, step) in probes.items():
                offset = [0.0, 0.0, 0.0]
                offset[axis] = step
                _translate(part.mesh, offset)
            depsgraph = context.evaluated_depsgraph_get()
            for part, (shown, step) in probes.items():
                follow[part][:, :, axis] = (_displayed_positions(part.obj, depsgraph, part.count) - shown) / step
                offset = [0.0, 0.0, 0.0]
                offset[axis] = -step
                _translate(part.mesh, offset)
    finally:
        for part in probes:
            _set_positions(part.mesh, part.start)

    for part, (shown, _step) in probes.items():
        matrices = follow[part]
        scale = np.linalg.norm(np.linalg.inv(part.inverse), ord=2)
        movable = np.linalg.svd(matrices, compute_uv=False)[:, -1] > MIN_FOLLOW * scale
        inverse = np.zeros_like(matrices)
        inverse[movable] = np.linalg.inv(matrices[movable])
        part.origin = shown
        part.inverse = inverse
        part.frozen = ~movable
        _follow_cache[part.obj.as_pointer()] = (part.start.astype(np.float64), shown, matrices, inverse, part.frozen)
    while len(_follow_cache) > 8:
        del _follow_cache[next(iter(_follow_cache))]
    return unmatched


class _EditedMeshes:
    """The meshes in Edit Mode as read for a stroke, see `read_edited_meshes`.

    ``co`` holds their displayed vertex positions (world space) put together in one array, ``start``
    their edit positions (object space) likewise, and ``movable`` which vertices a brush may move.
    ``tris`` are all triangles (edges likewise), ``visible_tris`` those of unhidden faces, and
    ``tri_uvs`` the corner UVs of ``tris``, NaN where a mesh has no UV map.
    """

    __slots__ = ("parts", "unmatched", "co", "start", "movable", "tris", "visible_tris", "edges", "tri_uvs")


def _read_uvs(obj, tri_loops):
    """UV coordinates of the corners of the loop triangles ``tri_loops`` of ``obj``'s mesh, just updated
    from Edit Mode, or None without a UV map. (In Edit Mode, the mesh's own UV data can't be read, but
    a copy's and the BMesh's can.)"""
    bm = bmesh.from_edit_mesh(obj.data)
    layer = bm.loops.layers.uv.active
    if layer is None:
        return None
    loop_count = len(obj.data.loops)
    copy = obj.to_mesh()
    try:
        uv = None
        if len(copy.loops) == loop_count and copy.uv_layers.active is not None:
            uv = np.empty(loop_count * 2, dtype=np.float32)
            copy.uv_layers.active.uv.foreach_get("vector", uv)
    finally:
        obj.to_mesh_clear()
    if uv is None:
        # Reading the BMesh loop by loop is much slower, but always works.
        uv = np.fromiter(
            (value for face in bm.faces for loop in face.loops for value in loop[layer].uv),
            dtype=np.float64, count=loop_count * 2,
        )
    return uv.astype(np.float64).reshape(-1, 2)[tri_loops]


def read_edited_meshes(context, body, *, selected_only=False, uvs=False):
    """Read every mesh in Edit Mode (but ``body``) for a stroke over all of them at once: their
    vertices are put together in one set of arrays, and so are their faces for the brush to follow.

    The meshes are read as displayed, which in Edit Mode can differ from the edit positions (shape
    keys shown with Shape Key Edit Mode, a posed Armature shown in Edit Mode), see `follow_displayed`.
    """
    result = _EditedMeshes()
    result.parts = []
    movables, all_edges, all_tris, visible, all_uvs = [], [], [], [], []
    first = 0
    for obj in edited_meshes(context, body):
        # Edit mode keeps its changes in a BMesh: copy them to the mesh to read flags and faces in bulk.
        obj.update_from_editmode()
        mesh = obj.data
        count = len(mesh.vertices)
        # Positions are those of the active shape key, like the BMesh's: the mesh holds the basis.
        key = obj.active_shape_key
        start = np.empty(count * 3, dtype=np.float32)
        (mesh.vertices if key is None else key.data).foreach_get("co", start)
        start = start.reshape(-1, 3)
        hidden = np.empty(count, dtype=bool)
        mesh.vertices.foreach_get("hide", hidden)
        movable = ~hidden
        if selected_only:
            selected = np.empty(count, dtype=bool)
            mesh.vertices.foreach_get("select", selected)
            movable &= selected
        mesh.calc_loop_triangles()
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(-1, 3)
        tri_faces = np.empty(len(mesh.loop_triangles), dtype=np.int32)
        mesh.loop_triangles.foreach_get("polygon_index", tri_faces)
        face_hidden = np.empty(len(mesh.polygons), dtype=bool)
        mesh.polygons.foreach_get("hide", face_hidden)
        edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", edges)
        if uvs:
            tri_loops = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("loops", tri_loops)
            tri_uvs = _read_uvs(obj, tri_loops.reshape(-1, 3))
            all_uvs.append(np.full((len(tris), 3, 2), np.nan) if tri_uvs is None else tri_uvs)

        movables.append(movable)
        all_tris.append(tris + first)
        # The brush follows the visible faces.
        visible.append(tris[~face_hidden[tri_faces]] + first)
        all_edges.append(edges.reshape(-1, 2) + first)
        result.parts.append(_EditedPart(obj, first, start))
        first += count

    result.unmatched = follow_displayed(context, result.parts)
    for part, movable in zip(result.parts, movables):
        if part.frozen is not None:
            movable &= ~part.frozen
    result.start = np.concatenate([part.start for part in result.parts])
    result.co = np.concatenate([part.origin for part in result.parts])
    result.movable = np.concatenate(movables)
    result.tris = np.concatenate(all_tris)
    result.visible_tris = np.concatenate(visible)
    result.edges = np.concatenate(all_edges)
    result.tri_uvs = np.concatenate(all_uvs) if uvs else None
    return result


class _EditMeshStroke(_BrushStroke):
    """A brush stroke over the meshes in Edit Mode, shared by Body Fit and Texture Relax.

    ``invoke`` builds ``self.stroke`` (with ``dab(center, radius, strength, falloff)`` returning the
    indices of moved vertices, ``co`` holding their positions and ``moved`` flagging them) and calls
    `_begin`. Moves are written back into the edit meshes (into the active shape key), and one Ctrl Z
    undoes the stroke on all of them. Esc puts every moved vertex back exactly.
    """

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (
            context.mode == 'EDIT_MESH' and
            obj is not None and obj.type == 'MESH' and
            context.area is not None and context.area.type == 'VIEW_3D' and
            context.region is not None and context.region.type == 'WINDOW'
        )

    def _begin(self, context, event, edited, settings):
        """Set the brush up over ``edited`` (an `_EditedMeshes`), place the first dab, and go modal."""
        for part in edited.unmatched:
            self.report({'WARNING'}, "'{:s}': modifiers shown in Edit Mode change its geometry, so its "
                                     "shape without them is used".format(part.obj.name))
        self.parts = edited.parts
        self.start = edited.start
        self.bvh = BVHTree.FromPolygons(edited.co.tolist(), edited.visible_tris.tolist(), all_triangles=True)
        # Mirroring follows the active mesh's options and axes.
        active = next((part.obj for part in self.parts if part.obj == context.object), self.parts[0].obj)
        self.matrix = active.matrix_world.copy()
        self.matrix_inv = self.matrix.inverted_safe()
        self.flips = _mirror_flips(active.data)

        self.region = context.region
        self.rv3d = context.region_data
        self.button = event.type
        self.radius = settings.radius
        self.strength = settings.strength
        self.use_pressure_radius = settings.use_pressure_radius
        self.use_pressure_strength = settings.use_pressure_strength
        self.falloff = settings.falloff
        self.spacing = settings.spacing / 100.0
        # Vertices moved since the edit mesh was last updated.
        self.pending = set()

        self.last_xy = (event.mouse_region_x, event.mouse_region_y)
        self.last_pressure = event.pressure
        self._dab(self.last_xy, event.pressure)
        self._update(context)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._stroke_to((event.mouse_region_x, event.mouse_region_y), event.pressure)
            self._update(context)
        elif event.type == self.button and event.value == 'RELEASE':
            self._update(context)
            return {'FINISHED'} if self.stroke.moved.any() else {'CANCELLED'}
        elif event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            moved = np.flatnonzero(self.stroke.moved)
            if len(moved):
                self._write(moved, self.start[moved])
                context.area.tag_redraw()
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _dab(self, xy, pressure):
        radius_px = self.radius * (pressure if self.use_pressure_radius else 1.0)
        strength = self.strength * (pressure if self.use_pressure_strength else 1.0)
        if radius_px < 0.5 or strength <= 0.0:
            return
        hit = self._raycast(xy)
        if hit is None:
            return
        radius = self._world_radius(hit, radius_px)
        if not radius:
            return
        for center in (hit, *self._mirrored(hit)):
            self.pending.update(self.stroke.dab(np.array(center), radius, strength, self.falloff).tolist())

    def _update(self, context):
        if self.pending:
            moved = np.fromiter(self.pending, dtype=np.int64, count=len(self.pending))
            self.pending.clear()
            self._write(moved)
            context.area.tag_redraw()

    def _write(self, indices, local_co=None):
        """Store the positions of vertices ``indices`` (of the combined arrays) in their meshes: the
        stroke's current ones, or ``local_co`` (object space)."""
        write_edited(self.parts, indices, self.stroke.co[indices] if local_co is None else None, local_co)


def write_edited(parts, indices, co=None, local_co=None):
    """Store vertices ``indices`` (of the combined arrays of `read_edited_meshes`) in their edit meshes,
    displayed at ``co`` (world space), or at the edit positions ``local_co`` (object space); one row
    per index."""
    for part in parts:
        inside = (indices >= part.first) & (indices < part.first + part.count)
        if not inside.any():
            continue
        part_indices = indices[inside]
        if local_co is None:
            positions = part.to_edit(part_indices - part.first, co[inside])
        else:
            positions = local_co[inside]
        bm = bmesh.from_edit_mesh(part.mesh)
        bm.verts.ensure_lookup_table()
        verts = bm.verts
        for index, position in zip((part_indices - part.first).tolist(), positions.tolist()):
            verts[index].co = position
        bmesh.update_edit_mesh(part.mesh, loop_triangles=True, destructive=False)


class MAGIC_FIT_OT_body_fit(_EditMeshStroke, bpy.types.Operator):
    """Sculpt the meshes around the body. All their layers move together"""
    bl_idname = "magic_fit.body_fit"
    bl_label = "Body Fit Stroke"
    bl_options = {'UNDO', 'BLOCKING'}

    invert: BoolProperty(
        name="Invert",
        description="Swap Push Out and Tighten for this stroke",
        options={'SKIP_SAVE'},
    )

    def invoke(self, context, event):
        problem = check_fit_ready(context)
        if problem is not None:
            self.report({'WARNING'}, problem)
            return {'CANCELLED'}
        settings = context.scene.body_fit_brush
        body_object = context.scene.magic_fit.target
        body = fitting.BodySurface.from_object(body_object, context.evaluated_depsgraph_get())
        if body is None:
            self.report({'WARNING'}, "The body mesh has no faces")
            return {'CANCELLED'}
        edited = read_edited_meshes(context, body_object, selected_only=settings.use_selected_only)

        mode = settings.mode
        if self.invert:
            mode = {'PUSH': 'TIGHTEN', 'TIGHTEN': 'PUSH'}.get(mode, mode)
        self.stroke = fitting.FitStroke(
            body, edited.co, edited.movable,
            mode=mode,
            offset=settings.offset,
            layer_radius=settings.layer_radius,
            max_distance=settings.max_distance if settings.use_max_distance else None,
            edges=edited.edges,
            # The triangles tell which way the clothing faces and where its layers lie (see FitStroke).
            tris=edited.tris,
            smooth=settings.auto_smooth if settings.use_auto_smooth else 0.0,
            weld_distance=settings.seam_distance,
            fade_distance=settings.fade_distance if settings.use_fade else 0.0,
        )
        return self._begin(context, event, edited, settings)


class MAGIC_FIT_OT_texture_relax(_EditMeshStroke, bpy.types.Operator):
    """Slide the vertices under the brush until the texture on them is even again. The shape doesn't change"""
    bl_idname = "magic_fit.texture_relax"
    bl_label = "Texture Relax Stroke"
    bl_options = {'UNDO', 'BLOCKING'}

    def invoke(self, context, event):
        problem = check_relax_ready(context)
        if problem is not None:
            self.report({'WARNING'}, problem)
            return {'CANCELLED'}
        settings = context.scene.texture_relax_brush
        edited = read_edited_meshes(context, None, selected_only=settings.use_selected_only, uvs=True)
        for part in edited.parts:
            if part.obj.data.uv_layers.active is None:
                self.report({'WARNING'}, "'{:s}' has no UV map, so it is left alone".format(part.obj.name))
        self.stroke = relaxing.RelaxStroke(
            edited.co, edited.tris, edited.tri_uvs, edited.movable,
            boundary=settings.boundary,
            weld_distance=settings.seam_distance,
        )
        return self._begin(context, event, edited, settings)


# Fit Move: the proportional size changes by this factor per wheel step or Page Up/Down, like Blender's,
# and mouse moves count this much while Shift is held.
PROPORTIONAL_STEP = 1.1
PRECISION = 0.1
AXES = (Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)))
AXIS_COLORS = ((1.0, 0.21, 0.33, 1.0), (0.55, 0.86, 0.0, 1.0), (0.16, 0.56, 1.0, 1.0))
FIT_MOVE_HINTS = (
    "Confirm: Click, Enter   Cancel: Right-click, Esc   X, Y, Z: along an axis (Shift: not along it)   "
    "C: free   Wheel, Page Up/Down: proportional size   Shift: precise"
)


def _length_text(context, value):
    unit = context.scene.unit_settings
    if unit.system == 'NONE':
        return "{:.4f}".format(value)
    return bpy.utils.units.to_string(unit.system, 'LENGTH', value * unit.scale_length, precision=4)


def _draw_polyline(coords, color, width):
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", width)
    shader.uniform_float("color", color)
    batch_for_shader(shader, 'LINE_STRIP', {"pos": coords}).draw(shader)


class MAGIC_FIT_OT_fit_move(bpy.types.Operator):
    """Move the selected vertices. Proportional editing reaches along the mesh, not across gaps, keeps split seams closed and fades out toward hidden vertices"""
    bl_idname = "magic_fit.fit_move"
    bl_label = "Fit Move"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    value: FloatVectorProperty(
        name="Move",
        description="How far the selection moves",
        subtype='TRANSLATION',
        size=3,
    )
    release_confirm: BoolProperty(
        name="Confirm on Release",
        description="Finish when the mouse button that started the move is released",
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    @classmethod
    def poll(cls, context):
        obj = context.object
        return context.mode == 'EDIT_MESH' and obj is not None and obj.type == 'MESH'

    def _setup(self, context):
        """Read the edited meshes and work out how far each vertex goes. Returns why nothing can move, or
        None."""
        edited = read_edited_meshes(context, None)
        if not edited.parts:
            return "Edit a mesh to move its vertices"
        selected = []
        for part in edited.parts:
            mask = np.empty(part.count, dtype=bool)
            part.mesh.vertices.foreach_get("select", mask)
            selected.append(mask)
        selected = np.concatenate(selected)
        if not (selected & edited.movable).any():
            return "Select the vertices to move"
        for part in edited.unmatched:
            self.report({'WARNING'}, "'{:s}': modifiers shown in Edit Mode change its geometry, so its "
                                     "shape without them is used".format(part.obj.name))
        tool = context.tool_settings
        settings = context.scene.fit_move
        self.move = moving.FitMove(
            edited.co, edited.edges, selected, edited.movable,
            tris=edited.tris,
            proportional_size=tool.proportional_size if tool.use_proportional_edit else None,
            falloff=tool.proportional_edit_falloff,
            weld_distance=settings.seam_distance,
            fade_distance=settings.fade_distance if settings.use_fade else 0.0,
        )
        if not len(self.move.indices):
            return "Hidden vertices next to the selection keep it in place (Fade at Hidden)"
        self.parts = edited.parts
        self.local_start = edited.start
        self.center = Vector(edited.co[selected].mean(axis=0))
        # Every vertex moved so far, to put back when the proportional size shrinks or the move is cancelled.
        self.touched = self.move.indices
        return None

    def _apply(self, translation):
        self.touched = np.union1d(self.touched, self.move.indices)
        write_edited(self.parts, self.touched, self.move.positions(translation, self.touched))

    def execute(self, context):
        problem = self._setup(context)
        if problem is not None:
            self.report({'WARNING'}, problem)
            return {'CANCELLED'}
        self._apply(np.array(self.value))
        return {'FINISHED'}

    def invoke(self, context, event):
        region = context.region
        if context.area is None or context.area.type != 'VIEW_3D' or region is None or region.type != 'WINDOW':
            self.report({'WARNING'}, "Move the mouse over the 3D Viewport")
            return {'CANCELLED'}
        problem = self._setup(context)
        if problem is not None:
            self.report({'WARNING'}, problem)
            return {'CANCELLED'}
        self.region = context.region
        self.rv3d = context.region_data
        # Where the mouse started, where it is, and where it counts as being (Shift slows it down).
        self.start_xy = Vector((event.mouse_region_x, event.mouse_region_y))
        self.last_xy = self.start_xy.copy()
        self.xy = self.start_xy.copy()
        self.axis = None
        self.plane = False
        self.translation = Vector((0.0, 0.0, 0.0))
        # What the move shows its text in, to clear again wherever the mouse is by then.
        self.area = context.area
        self.workspace = context.workspace
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (self,), 'WINDOW', 'POST_PIXEL')
        try:
            context.window_manager.modal_handler_add(self)
            self._show(context)
        except Exception:
            self._finish(context)
            raise
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        try:
            return self._step(context, event)
        except Exception:
            self._abort(context)
            raise

    def cancel(self, context):
        # Blender takes the operator down while it runs (a file is loaded, the area closes...).
        self._abort(context)

    def _step(self, context, event):
        kind, value = event.type, event.value
        if kind in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            xy = Vector((event.mouse_region_x, event.mouse_region_y))
            self.xy += (xy - self.last_xy) * (PRECISION if event.shift else 1.0)
            self.last_xy = xy
            self._update(context)
        elif kind in {'X', 'Y', 'Z'} and value == 'PRESS':
            axis, plane = 'XYZ'.index(kind), event.shift
            if (self.axis, self.plane) == (axis, plane):
                self.axis = None
            else:
                self.axis, self.plane = axis, plane
            self._update(context)
        elif kind == 'C' and value == 'PRESS':
            self.axis = None
            self._update(context)
        elif kind in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'PAGE_UP', 'PAGE_DOWN'} and value == 'PRESS':
            if self.move.size is not None:
                tool = context.tool_settings
                step = PROPORTIONAL_STEP if kind in {'WHEELDOWNMOUSE', 'PAGE_UP'} else 1.0 / PROPORTIONAL_STEP
                tool.proportional_size = self.move.size * step
                self.move.set_proportional(tool.proportional_size, tool.proportional_edit_falloff)
                self._update(context)
        elif ((kind in {'RET', 'NUMPAD_ENTER', 'SPACE'} and value == 'PRESS') or
              (kind == 'LEFTMOUSE' and value == ('RELEASE' if self.release_confirm else 'PRESS'))):
            self.value = self.translation
            self._finish(context)
            return {'FINISHED'}
        elif kind in {'RIGHTMOUSE', 'ESC'} and value == 'PRESS':
            self._abort(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _translation(self):
        """The move for the mouse: in the view's plane, or along (or across) the chosen axis."""
        region, rv3d, center = self.region, self.rv3d, self.center
        if self.axis is None:
            a = view3d_utils.region_2d_to_location_3d(region, rv3d, self.start_xy, center)
            b = view3d_utils.region_2d_to_location_3d(region, rv3d, self.xy, center)
            return b - a
        axis = AXES[self.axis]
        hits = []
        for xy in (self.start_xy, self.xy):
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, xy)
            ray = origin + view3d_utils.region_2d_to_vector_3d(region, rv3d, xy)
            if self.plane:
                hit = intersect_line_plane(origin, ray, center, axis)
            else:
                closest = intersect_line_line(origin, ray, center, center + axis)
                hit = None if closest is None else closest[1]
            if hit is None:
                return Vector((0.0, 0.0, 0.0))
            hits.append(hit)
        move = hits[1] - hits[0]
        return move - axis * move.dot(axis) if self.plane else axis * move.dot(axis)

    def _update(self, context):
        self.translation = self._translation()
        self._apply(np.array(self.translation))
        self._show(context)

    def _show(self, context):
        move = self.translation
        text = "D: {:s}  ({:s})".format(
            "  ".join(_length_text(context, v) for v in move), _length_text(context, move.length),
        )
        if self.axis is not None:
            text += "   {:s} {:s}".format("locking" if self.plane else "along", "XYZ"[self.axis])
        if self.move.size is not None:
            text += "   Proportional size: " + _length_text(context, self.move.size)
        self.area.header_text_set(text)
        self.workspace.status_text_set(FIT_MOVE_HINTS)
        self.area.tag_redraw()

    def _finish(self, context):
        """Take the drawing and the texts down. Safe to call again, and after the area is gone."""
        handle, self._handle = self._handle, None
        if handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
        try:
            self.area.header_text_set(None)
            self.area.tag_redraw()
        except ReferenceError:
            pass
        try:
            self.workspace.status_text_set(None)
        except ReferenceError:
            pass

    def _abort(self, context):
        """Put the vertices back where they were and finish: for a move that ends without being confirmed."""
        try:
            write_edited(self.parts, self.touched, local_co=self.local_start[self.touched])
        except (ReferenceError, ValueError, RuntimeError):
            # The mesh is gone, or no longer in Edit Mode: nothing to put back.
            pass
        finally:
            self._finish(context)

    @staticmethod
    def _draw(self):
        """The proportional size around the selection, and the axis the move keeps to."""
        if bpy.context.region != self.region:
            return
        region, rv3d = self.region, self.rv3d
        center = view3d_utils.location_3d_to_region_2d(region, rv3d, self.center + self.translation)
        if center is None:
            return
        ui_scale = bpy.context.preferences.system.ui_scale
        gpu.state.blend_set('ALPHA')
        if self.axis is not None:
            start = view3d_utils.location_3d_to_region_2d(region, rv3d, self.center)
            reach = float(region.width + region.height)
            for axis in ([self.axis] if not self.plane else [k for k in range(3) if k != self.axis]):
                tip = view3d_utils.location_3d_to_region_2d(
                    region, rv3d, self.center + AXES[axis] * rv3d.view_distance * 0.1,
                )
                if start is None or tip is None or (tip - start).length < 1e-3:
                    continue
                along = (tip - start).normalized() * reach
                _draw_polyline([start - along, start + along], AXIS_COLORS[axis], 1.5 * ui_scale)
        if self.move.size is not None:
            side = rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))
            edge = view3d_utils.location_3d_to_region_2d(
                region, rv3d, self.center + self.translation + side * self.move.size,
            )
            if edge is not None:
                radius = (edge - center).length
                segments = max(32, min(128, int(radius / 2)))
                circle = [(center.x + radius * math.cos(2.0 * math.pi * i / segments),
                           center.y + radius * math.sin(2.0 * math.pi * i / segments)) for i in range(segments + 1)]
                _draw_polyline(circle, (0.0, 0.0, 0.0, 0.45), 3.0 * ui_scale)
                _draw_polyline(circle, (1.0, 1.0, 1.0, 0.85), 1.5 * ui_scale)
        gpu.state.blend_set('NONE')


def _unweighted_message(stroke, mesh):
    return (
        "{:d} vertices of '{:s}' had no body weights and there was no body to take them from, so they "
        "hang from the skirt bones only".format(len(stroke.unweighted), mesh.name)
    )


def _unhidden_vertices(mesh, masked):
    """Vertices Skirt Weights and Heel Weights may change: not hidden, and selected when ``masked``
    selection masking is on."""
    hidden = np.zeros(len(mesh.vertices), dtype=bool)
    mesh.vertices.foreach_get("hide", hidden)
    allowed = ~hidden
    mask = _selection_mask(mesh) if masked else None
    if mask is not None:
        allowed &= mask
    return np.flatnonzero(allowed)


class _WholeMeshWeights:
    """Skirt Weights and Heel Weights: the brush's goal for every vertex of the mesh at once."""
    bl_options = {'REGISTER', 'UNDO'}
    kind = ''
    nothing_changed = ""

    @classmethod
    def poll(cls, context):
        problem = check_whole_mesh_ready(context, cls.kind)
        if problem is not None:
            cls.poll_message_set(problem)
            return False
        return True

    def execute(self, context):
        _prop, _check, what, engine, error_type, _no_body = WHOLE_MESH[self.kind]
        settings = context.scene.magic_fit
        targets = whole_mesh_targets(context, settings, self.kind)
        # Everything is set up before any weights change, so a problem leaves every mesh as it was.
        try:
            strokes = [engine.build_stroke(context, obj, settings) for obj in targets]
        except error_type as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        changed = 0
        for obj, stroke in zip(targets, strokes):
            masked = context.mode == 'PAINT_WEIGHT' and obj == context.object
            verts = _unhidden_vertices(obj.data, masked)
            stroke.apply(verts, np.ones(len(verts)))
            if getattr(stroke, "unweighted", None):
                self.report({'WARNING'}, _unweighted_message(stroke, obj.data))
            created = stroke.created_group_names()
            if created:
                self.report({'INFO'}, "Added vertex groups to '{:s}': {:s}".format(obj.name, ", ".join(created)))
            if stroke.modified:
                changed += 1
                obj.data.update()
        if not changed:
            self.report({'INFO'}, self.nothing_changed)
            return {'CANCELLED'}
        self.report({'INFO'}, "Gave {:s} to {:d} mesh(es)".format(what, changed))
        return {'FINISHED'}


class MAGIC_FIT_OT_skirt_weights(_WholeMeshWeights, bpy.types.Operator):
    """Give the whole mesh skirt bone weights below the hips, blended with its body weights. Uses the same
    Skirt settings as the brush"""
    bl_idname = "magic_fit.skirt_weights"
    bl_label = "Skirt Weights"
    kind = 'SKIRT'
    nothing_changed = "Nothing changed: no part of the mesh hangs below Blend Start, or it has its skirt weights already"


class MAGIC_FIT_OT_heel_weights(_WholeMeshWeights, bpy.types.Operator):
    """Make the whole mesh rigid below the ankle, so the foot only follows the ankle bone. Uses the same
    Heels settings as the brush"""
    bl_idname = "magic_fit.heel_weights"
    bl_label = "Heel Weights"
    kind = 'HEELS'
    nothing_changed = "Nothing changed: no part of the mesh is below the ankle, or it has its heel weights already"


def _match_text(matches, count=6):
    return ",".join("{:d}:{:.3f}".format(number, score) for number, score in matches[:count])


def parse_matches(text):
    """(number, score) pairs from `HairWeightsState.matches`."""
    result = []
    for item in text.split(","):
        number, _sep, score = item.partition(":")
        try:
            result.append((int(number), float(score)))
        except ValueError:
            continue
    return result


def hair_skeleton_text(reference, race, number):
    """How the report and the panel name a hair skeleton."""
    if not number:
        return "no hair skeleton (EST entry 0: the body's hair bones only)"
    hairs = reference.vanilla_hairs(race, number)
    used = ""
    if hairs:
        shown = ", ".join(str(h) for h in hairs[:4]) + (", ..." if len(hairs) > 4 else "")
        used = "; the game's hair{:s} {:s} use{:s} it".format("s" if len(hairs) > 1 else "", shown,
                                                                 "" if len(hairs) > 1 else "s")
    return "hair skeleton {:s} (EST entry {:d}{:s})".format(hairing.skeleton_label(number), number, used)


class MAGIC_FIT_OT_hair_weights(bpy.types.Operator):
    """Replace the hair's weights with FFXIV hair bone weights. Uses the best-fitting hair skeleton (or the
    picked one) and adds its bones to the hair's armature, or to a new one"""
    bl_idname = "magic_fit.hair_weights"
    bl_label = "Hair Weights"
    bl_options = {'REGISTER', 'UNDO'}

    skeleton: IntProperty(
        name="Hair Skeleton",
        description="Hair skeleton to use instead of the one in the settings (-1: use the settings)",
        default=-1,
        min=-1,
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    @classmethod
    def description(cls, _context, properties):
        if properties.skeleton >= 0:
            return "Weight the hair with this hair skeleton instead, and keep it picked"
        return cls.__doc__

    @classmethod
    def poll(cls, context):
        problem = check_hair_weights_ready(context)
        if problem is not None:
            cls.poll_message_set(problem)
            return False
        return True

    def execute(self, context):
        settings = context.scene.magic_fit
        targets = hair_targets(context, settings)
        reference = hairing.Reference.get()
        race = settings.hair_race
        armature = next((found for found in map(hairing.weighted_armature, targets) if found is not None), None)
        # One hair has one EST entry: parts of it weighted before (on its armature, or in its collection
        # while this part has none yet) are matched along, and lend it their armature.
        parts = hair_parts(context, targets, armature, race)
        if armature is None and parts:
            armature = hairing.weighted_armature(parts[0])
        offset = hairing.armature_offset(armature, reference.rig(race, 0))
        rests = straighten.rest_geometry(context, targets, ('INDEXED',) * len(targets))
        matches = None
        best = self.skeleton < 0 and settings.hair_skeleton_mode != 'PICK'
        # A picked skeleton still gets the best matches, for the panel to suggest, unless the hair has them.
        if best or any(target.magic_fit_hair.race != race or not target.magic_fit_hair.matches
                       for target in targets):
            whole = rests + (straighten.rest_geometry(context, parts, ('INDEXED',) * len(parts)) if parts else [])
            points = np.concatenate([rest.co for rest in whole])
            starts = np.cumsum([0] + [len(rest.co) for rest in whole[:-1]])
            tris = np.concatenate([rest.tris + start for rest, start in zip(whole, starts)])
            try:
                matches = hairing.rank_skeletons(points, tris, race, offset=offset, reference=reference,
                                                 hang_start=settings.hair_hang_start,
                                                 hang_length=settings.hair_hang_length)
            except hairing.HairError as error:
                if best:
                    self.report({'ERROR'}, str(error))
                    return {'CANCELLED'}
        if self.skeleton >= 0:
            number = self.skeleton
            if number and not reference.has_skeleton(race, number):
                self.report({'ERROR'}, "{:s} has no hair skeleton {:s}".format(
                    hairing.RACE_NAMES[race], hairing.skeleton_label(number)))
                return {'CANCELLED'}
            settings.hair_skeleton_mode = 'PICK'
            settings.hair_skeleton = number
        elif not best:
            number = settings.hair_skeleton
        else:
            number = matches[0][0]
            if parts:
                self.report({'INFO'}, "Matched together with the hair's other part{:s}: {:s}".format(
                    "s" if len(parts) > 1 else "", ", ".join("'{:s}'".format(part.name) for part in parts)))
        rig = reference.rig(race, number).moved(offset)
        wanted = [hairing.HEAD, hairing.NECK, hairing.UPPER_BACK] + [bone.name for bone in rig.hair_bones()]
        if race in hairing.EAR_RACES and settings.hair_ears:
            wanted += list(hairing.EARS)
        new = armature is None
        if new:
            collection = targets[0].users_collection[0] if targets[0].users_collection else context.scene.collection
            armature = hairing.new_armature(context, "Hair Skeleton", collection)
        try:
            added = hairing.ensure_bones(context, armature, rig, wanted)
        except RuntimeError as error:
            self.report({'ERROR'}, "Couldn't add the hair bones to '{:s}': {:s}".format(armature.name, str(error)))
            if new:
                # Cancelling leaves no undo step to take it away.
                data = armature.data
                bpy.data.objects.remove(armature)
                bpy.data.armatures.remove(data)
            return {'CANCELLED'}
        for obj in targets:
            if straighten.deforming_armature(obj) is None:
                hairing.add_armature_modifier(obj, armature)
        params = hairing.stroke_params(settings)
        changed = 0
        removed = []
        for obj, rest in zip(targets, rests):
            # The hair weights replace the old ones: other bones' weights (such as an earlier skeleton's)
            # are taken off, and so are the groups left empty.
            stroke = hairing.HairStroke(obj, rest.co, rest.tris, rig, replace=True, **params)
            masked = context.mode == 'PAINT_WEIGHT' and obj == context.object
            verts = _unhidden_vertices(obj.data, masked)
            stroke.apply(verts, np.ones(len(verts)))
            emptied = goals.remove_empty_groups(obj)
            removed += [name for name in emptied if name not in removed]
            if stroke.modified or emptied:
                changed += 1
                obj.data.update()
            state = obj.magic_fit_hair
            state.race = race
            state.skeleton = number
            if matches is not None:
                state.matches = _match_text(matches)
            hairing.tag_est(obj, race, number)
        if new:
            self.report({'INFO'}, "Made armature '{:s}' with the game's bones for the hair".format(armature.name))
        elif added:
            self.report({'INFO'}, "Added {:d} bone(s) to '{:s}': {:s}".format(
                len(added), armature.name, ", ".join(added)))
        if removed:
            self.report({'INFO'}, "Took off the old weights of {:s}".format(
                ", ".join(removed[:8]) + (" and {:d} more".format(len(removed) - 8) if len(removed) > 8 else "")))
        what = hair_skeleton_text(reference, race, number)
        self.report({'INFO'}, "Gave {:d} mesh(es) hair weights with {:s}".format(changed, what) if changed else
                    "Nothing changed: the hair has these hair weights already ({:s})".format(what))
        if number:
            self.report({'INFO'}, "Tagged the hair with its EST entry {:d} (custom property {:s}) for export".format(
                number, hairing.EST_PROPERTY))
        other = [part for part in parts if part.magic_fit_hair.skeleton != number]
        if other:
            self.report({'WARNING'}, "The hair's other part{:s} {:s} still use{:s} another hair skeleton: give {:s} "
                        "Hair Weights with {:s} too (or select all parts and use the arrow)".format(
                            "s" if len(other) > 1 else "", ", ".join("'{:s}'".format(part.name) for part in other),
                            "" if len(other) > 1 else "s", "them" if len(other) > 1 else "it",
                            hairing.skeleton_label(number)))
        if race in hairing.EAR_RACES and settings.hair_ears:
            ears = any(hairing.ear_shares(rig.model(rest.co), rest.tris, rig) for rest in rests)
            self.report({'INFO'}, "The hair's cat ears follow the ear bones" if ears else
                        "No cat ears found: the hair near the ear bones stays with the head")
        if best and not number:
            points = np.concatenate([rest.co for rest in rests])
            starts = np.cumsum([0] + [len(rest.co) for rest in rests[:-1]])
            tris = np.concatenate([rest.tris + start for rest, start in zip(rests, starts)])
            fraction = hairing.hanging_fraction(points, tris, race, offset=offset, reference=reference,
                                                hang_start=settings.hair_hang_start,
                                                hang_length=settings.hair_hang_length)
            if fraction < hairing.MIN_HANGING:
                self.report({'INFO'}, "Nothing hangs far enough from the head to swing: lower Hang Start to let "
                                      "hair close to it (bangs) move too")
        return {'FINISHED'}


def _hair_skeleton_items(_self, context):
    reference = _hair_reference()
    race = context.scene.magic_fit.hair_race if context is not None else 'c0201'
    key = (race, reference is not None)
    items = _hair_skeleton_items.cache.get(key)
    if items is None:
        items = [("0", "None (EST 0)", "No hair skeleton: only the body's own hair bones (EST entry 0)")]
        if reference is not None:
            # Searchable by EST entry and by the game's hairstyles that use it.
            for number in reference.skeletons(race):
                hairs = reference.vanilla_hairs(race, number)
                label = "EST {:d}".format(number) + ("  (hair {:s})".format(", ".join(str(h) for h in hairs[:5]))
                                                     if hairs else "")
                bones = [bone.name for bone in reference.rig(race, number).skeleton_bones()]
                description = "Hair skeleton {:s}: {:s}".format(hairing.skeleton_label(number),
                                                                 ", ".join(bones) or "no bones")
                items.append((str(number), label, description))
        _hair_skeleton_items.cache[key] = items
    return items


_hair_skeleton_items.cache = {}


class MAGIC_FIT_OT_pick_hair_skeleton(bpy.types.Operator):
    """Pick the hair skeleton (EST entry) for Hair Weights and the Hair brush"""
    bl_idname = "magic_fit.pick_hair_skeleton"
    bl_label = "Pick Hair Skeleton"
    bl_property = "skeleton"
    bl_options = {'UNDO'}

    skeleton: EnumProperty(name="Hair Skeleton", items=_hair_skeleton_items)
    apply: BoolProperty(
        name="Apply",
        description="Weight the hair with the picked hair skeleton right away",
        default=False,
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    @classmethod
    def description(cls, _context, properties):
        if properties.apply:
            return ("Pick any hair skeleton of the race and weight the hair with it. Search by EST number or by "
                    "a game hairstyle that uses it")
        return cls.__doc__

    def invoke(self, context, _event):
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        settings = context.scene.magic_fit
        number = int(self.skeleton)
        settings.hair_skeleton = number
        settings.hair_skeleton_mode = 'PICK'
        if not self.apply:
            return {'FINISHED'}
        if not bpy.ops.magic_fit.hair_weights.poll():
            self.report({'ERROR'}, check_hair_weights_ready(context) or "Hair Weights can't run here")
            return {'CANCELLED'}
        return bpy.ops.magic_fit.hair_weights(skeleton=number)


_PART_WORDS = (
    (facing.PART_EYEBALL, "eyeball", "eyeballs"),
    (facing.PART_LASH, "lash part", "lash parts"),
    (facing.PART_FILM, "film over an eye", "films over the eyes"),
    (facing.PART_SHELL, "part on the skin", "parts on the skin"),
    (facing.PART_INSIDE, "part inside the mouth", "parts inside the mouth"),
    (facing.PART_EXTRA, "part where the game's face has one", "parts where the game's face has some"),
    (facing.PART_RIGID, "part moving as one piece with the skin", "parts moving as one piece with the skin"),
)


def remember_face(settings, race, number):
    """Keep the game face Detect found in the settings, which the Face panel shows and Pick starts from."""
    settings.face_race = race
    settings.face_number = number
    settings.face_found = "{:s} {:d}".format(race, number)


def found_face(settings):
    """(race, number) of the game face Detect found last, or None."""
    race, _sep, number = settings.face_found.partition(" ")
    return (race, int(number)) if number.isdigit() and race in facing.RACE_NAMES else None


def face_parts_text(plan):
    """What the plan found the parts of the face to be, for the report."""
    totals = {}
    for counts in plan.kinds.values():
        for kind, count in counts.items():
            totals[kind] = totals.get(kind, 0) + count
    return ", ".join("{:d} {:s}".format(totals[kind], plural if totals[kind] > 1 else single)
                     for kind, single, plural in _PART_WORDS if totals.get(kind))


def face_eyes_text(report):
    """How the eyes fare in the game's Shut Eyes expression, for the report: one line per eye."""
    lines = []
    for side, eye in sorted(report.items()):
        name = "Left eye" if side == 'l' else "Right eye"
        if eye["after"] <= 0.0:
            line = "{:s} is shut in the game's Shut Eyes expression".format(name)
            if eye["before"] > 0.05:
                line += "; it left {:.1f} mm² of the eyeball showing there before".format(eye["before"])
            if eye["early"] is not None:
                line += "; its lids would have met {:.0%} of the way into it and slid on past each other, so " \
                        "their weights were lowered to {:.0%}".format(eye["early"], eye["lowered"])
                if eye["meets"] < 1.0 - facing.CLOSE_SLACK:
                    line += ", as far as the game's blink still closes it (they meet {:.0%} of the way " \
                            "in)".format(eye["meets"])
        else:
            line = "{:s}: {:.1f} mm² of the eyeball still shows in the game's Shut Eyes expression; the lids stop " \
                   "{:.1f} mm apart where their weights can't go higher".format(name, eye["after"], eye["stuck"])
        if eye["blink"] > 0.05:
            line += "; {:.1f} mm² shows at the peak of the game's blink".format(eye["blink"])
        lines.append(line)
    return lines


def textools_warning(reference, race, number):
    """The warning that the game's face ``number`` of ``race`` was read from mods TexTools installed into the
    game's files, or None."""
    files = reference.textools_files(race, number)
    if not files:
        return None
    return ("The game's {:s} was read from TexTools mods in the game's files ({:s}), not from the game's own: turn "
            "them off in TexTools, then Read the Game's Faces Again (in Magic Fit's preferences)".format(
                facing.face_title(race, number), ", ".join(path.rsplit("/", 1)[-1] for path in files)))


class _FaceTool:
    """Face Weights and the face repairs: what `facing.plan_face` plans for the selected meshes, done."""
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        problem = check_face_tools_ready(context)
        if problem is not None:
            cls.poll_message_set(problem)
            return False
        return True

    def run(self, context, title, **plan_args):
        settings = context.scene.magic_fit
        targets = face_targets(context)
        reference = face_reference(self, context)
        if reference is None:
            return {'CANCELLED'}
        # Everything is worked out before anything changes, so a problem leaves every mesh as it was.
        try:
            job = facing.setup(targets, settings)
            plan = facing.plan_face(job.meshes, job.skin, job.face, targets=job.parts,
                                    max_groups=settings.face_max_groups, **plan_args)
        except facing.FaceError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        face = job.face
        if job.how != 'PICK':
            remember_face(settings, face.race, face.number)
        face_title = facing.face_title(face.race, face.number)
        rig = face.skeleton.rig(job.offset)
        armature = job.armature
        new = armature is None
        if new:
            skin = job.objects[job.skin]
            collection = skin.users_collection[0] if skin.users_collection else context.scene.collection
            armature = hairing.new_armature(context, "Face Skeleton", collection)
        try:
            added = hairing.ensure_bones(context, armature, rig, list(rig.by_name))
        except RuntimeError as error:
            self.report({'ERROR'}, "Couldn't add the face bones to '{:s}': {:s}".format(armature.name, str(error)))
            if new:
                # Cancelling leaves no undo step to take it away.
                data = armature.data
                bpy.data.objects.remove(armature)
                bpy.data.armatures.remove(data)
            return {'CANCELLED'}
        for i in plan.goals:
            if straighten.deforming_armature(job.objects[i]) is None:
                hairing.add_armature_modifier(job.objects[i], armature)
        moved = 0.0
        for i, (verts, offsets) in plan.lashes.items():
            facing.move_vertices(job.objects[i], verts, offsets)
            moved = max(moved, float(np.linalg.norm(offsets, axis=1).max()))
        neck = plan.neck
        if neck is not None:
            skin = job.objects[job.skin]
            current = facing.rest_points(skin)
            facing.move_vertices(skin, neck.verts, neck.positions + job.offset - current[neck.verts])
            facing.set_vertex_normals(skin, neck.verts, neck.normals)
        changed = 0
        for i, goal in plan.goals.items():
            obj = job.objects[i]
            stroke = facing.FaceStroke(obj, facing.rest_points(obj), plan.names, goal,
                                       max_groups=settings.face_max_groups)
            masked = context.mode == 'PAINT_WEIGHT' and obj == context.object
            verts = _unhidden_vertices(obj.data, masked)
            verts = verts[plan.masks[i][verts]]
            stroke.apply(verts, np.ones(len(verts)))
            if stroke.modified:
                changed += 1
                obj.data.update()
        if new:
            self.report({'INFO'}, "Made armature '{:s}' with the bones of {:s}".format(armature.name, face_title))
        elif added:
            self.report({'INFO'}, "Added {:d} bone(s) of {:s} to '{:s}'".format(len(added), face_title, armature.name))
        how = {'NAME': " (named in the meshes)", 'SHAPE': " (closest in shape)", 'PICK': ""}[job.how]
        done = []
        if changed:
            done.append("new weights for {:d} mesh(es)".format(changed))
        if plan.lashes:
            done.append("the lashes' roots moved onto the lids by up to {:.2f} mm".format(moved * 1000.0))
        if neck is not None and neck.moved.max() > 1e-6:
            done.append("the neck opening moved onto the game's by up to {:.2f} mm".format(neck.moved.max() * 1000.0))
        self.report({'INFO'}, "{:s} from the game's {:s}{:s}: {:s}".format(
            title, face_title, how, ", ".join(done) if done else "the meshes are like that already"))
        parts = face_parts_text(plan)
        if parts:
            self.report({'INFO'}, "Parts: " + parts)
        if neck is not None:
            self.report({'INFO'}, "Neck: the {:d} vertices of the opening have the game's positions, normals and "
                                  "weights".format(len(neck.verts)))
            if neck.unmatched.any():
                self.report({'WARNING'}, "{:d} vertices of the neck opening were more than 2 mm from the game's and "
                                         "went onto it anyway: check the neck".format(int(neck.unmatched.sum())))
        for line in face_eyes_text(plan.report["eyes"]):
            self.report({'INFO'}, line)
        features = plan.report.get("features", {})
        eyes = plan.report.get("reference", {}).get("eyes", [])
        skin_mode, parts_mode = plan_args.get("skin"), plan_args.get("parts")
        # What a missing eye opening or lip line means depends on what the tool does.
        if len(features.get("eyes", [])) < len(eyes) and (skin_mode in ('ALL', 'EYES') or
                                                         parts_mode in ('ALL', 'LASHES')):
            if skin_mode == 'ALL':
                what = "its lids got the game's weights without fitting them"
            elif skin_mode == 'EYES':
                what = "its lids were left as they are"
            else:
                what = "lashes along its lids weren't recognized"
            self.report({'WARNING'}, "Couldn't find {:s} on the face: {:s}".format(
                "the eye openings" if not features.get("eyes") else "one eye opening", what))
        if skin_mode == 'ALL' and plan.report.get("reference", {}).get("lip") and not features.get("lip"):
            self.report({'WARNING'}, "Couldn't find the line between the lips: the mouth got the game's weights "
                                     "without fitting them")
        for warning in plan.report["warnings"]:
            self.report({'WARNING'}, warning[:1].upper() + warning[1:])
        if job.left_out:
            self.report({'WARNING'}, "Left out {:s}: weighted to bones the face doesn't use (hair, the body)".format(
                ", ".join("'{:s}'".format(job.objects[i].name) for i in job.left_out)))
        if job.how == 'SHAPE':
            self.report({'WARNING'}, SHAPE_MATCH.format(face_title))
        warning = textools_warning(reference, face.race, face.number)
        if warning is not None:
            self.report({'WARNING'}, warning)
        return {'FINISHED'}


class MAGIC_FIT_OT_face_weights(_FaceTool, bpy.types.Operator):
    """Weight the selected face and its parts (eyes, lashes, teeth, piercings) like the game face it replaces,
    fitted to its shape. The lids meet when the eyes are shut and the neck matches the game's"""
    bl_idname = "magic_fit.face_weights"
    bl_label = "Face Weights"

    def execute(self, context):
        settings = context.scene.magic_fit
        return self.run(context, self.bl_label, skin='ALL', parts='ALL', close=settings.face_close_eyes,
                        neck=settings.face_match_neck)


# (repair, label, description, icon, `facing.plan_face` arguments).
FACE_REPAIRS = (
    ('EYES', "Fix Eyelids",
     "Weight the eyelids like the game's face, so they meet when the eyes are shut. Lashes and anything around "
     "the eyes follow them",
     'HIDE_ON', dict(skin='EYES', parts='EYES', close=True)),
    ('LASHES', "Snap Lashes",
     "Sit the selected lashes on the lids where they float off or sink in (out by Lash Lift), keeping their "
     "shape and where they are along the lid, and make each lash follow its lid. Select the face and eyeballs too",
     'SNAP_ON', dict(skin='NONE', parts='LASHES', snap_lashes=True)),
    ('MOUTH', "Fix Mouth",
     "Weight the lips and mouth like the game's face. Teeth, tongue and piercings in or around the mouth "
     "follow them",
     'MONKEY', dict(skin='MOUTH', parts='MOUTH')),
    ('PARTS', "Attach Parts",
     "Make the selected parts (eyeballs, lashes, teeth, piercings, horns...) follow the face. The face keeps "
     "its own weights",
     'LINKED', dict(skin='NONE', parts='ALL')),
    ('NECK', "Match Neck",
     "Give the neck the game's weights and move its edge onto the game's, so no seam shows where the face "
     "meets the body",
     'MOD_SKIN', dict(skin='NECK', parts='NONE', neck=True)),
)
_FACE_REPAIR = {repair[0]: repair for repair in FACE_REPAIRS}


class MAGIC_FIT_OT_face_repair(_FaceTool, bpy.types.Operator):
    """Repair one thing on the selected face, using the game face it replaces"""
    bl_idname = "magic_fit.face_repair"
    bl_label = "Repair Face"

    repair: EnumProperty(
        name="Repair",
        items=[(repair, label, description) for repair, label, description, _icon, _args in FACE_REPAIRS],
    )

    @classmethod
    def description(cls, _context, properties):
        return _FACE_REPAIR[properties.repair][2]

    def execute(self, context):
        settings = context.scene.magic_fit
        _repair, label, _description, _icon, args = _FACE_REPAIR[self.repair]
        args = dict(args)
        if args.get("snap_lashes"):
            args["lash_lift"] = settings.face_lash_lift
        return self.run(context, label, **args)


class MAGIC_FIT_OT_detect_face(bpy.types.Operator):
    """Find which game face the selected face replaces. Uses names like c0801f0002 if there are any, else
    the closest shape"""
    bl_idname = "magic_fit.detect_face"
    bl_label = "Detect Face"
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.mode not in {'OBJECT', 'PAINT_WEIGHT', 'POSE'}:
            cls.poll_message_set("Detect the face in Object or Weight Paint mode")
            return False
        if not face_targets(context):
            cls.poll_message_set("Select the face")
            return False
        problem = facing.face_data_problem()
        if problem is not None:
            cls.poll_message_set(problem)
            return False
        return True

    def execute(self, context):
        settings = context.scene.magic_fit
        if face_reference(self, context) is None:
            return {'CANCELLED'}
        found = facing.detect(face_targets(context))
        if found is None:
            self.report({'ERROR'}, facing.NOT_DETECTED)
            return {'CANCELLED'}
        race, number, how = found
        remember_face(settings, race, number)
        if how == 'SHAPE':
            self.report({'WARNING'}, SHAPE_MATCH.format(facing.face_title(race, number)))
        else:
            self.report({'INFO'}, "The game's {:s} (named in the meshes)".format(facing.face_title(race, number)))
        return {'FINISHED'}


def _face_items(_self, context):
    reference = facing.Reference.loaded()
    race = context.scene.magic_fit.face_race if context is not None else 'c0201'
    # By the faces read too, so reading them again or another game folder lists theirs. Older lists stay: Blender
    # needs the strings it was given to live on.
    key = (race, reference is not None, facing.Reference.generation)
    items = _face_items.cache.get(key)
    if items is None:
        items = []
        if reference is not None:
            for number in reference.faces(race):
                items.append((str(number), facing.face_label(number), "Face {:d} of {:s}s (face skeleton {:d})".format(
                    number, facing.RACE_NAMES[race], reference.skeleton_number(race, number))))
        _face_items.cache[key] = items
    return items


_face_items.cache = {}


class MAGIC_FIT_OT_pick_face(bpy.types.Operator):
    """Pick the game's face the custom face replaces"""
    bl_idname = "magic_fit.pick_face"
    bl_label = "Pick Face"
    bl_property = "face"
    bl_options = {'UNDO'}

    face: EnumProperty(name="Face", items=_face_items)

    def invoke(self, context, _event):
        if face_reference(self, context) is None:
            return {'CANCELLED'}
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        settings = context.scene.magic_fit
        settings.face_number = int(self.face)
        settings.face_source = 'PICK'
        return {'FINISHED'}


class MAGIC_FIT_OT_read_game_faces(bpy.types.Operator):
    """Read the faces from your game files again. Face does this by itself the first time and after game
    updates"""
    bl_idname = "magic_fit.read_game_faces"
    bl_label = "Read the Game's Faces Again"

    @classmethod
    def poll(cls, _context):
        root, setting = facedata.game_folder()
        if root is None:
            cls.poll_message_set(facedata.GAME_NOT_AT.format(setting) if setting else facedata.GAME_NOT_FOUND)
            return False
        return True

    def execute(self, context):
        wm = context.window_manager
        started = time.perf_counter()
        wm.progress_begin(0, 100)
        try:
            reference = facing.Reference.read_again(lambda share: wm.progress_update(int(share * 100)))
        except facedata.FaceDataError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        finally:
            wm.progress_end()
        count = sum(len(reference.faces(race)) for race in reference.races)
        self.report({'INFO'}, "Read {:d} of the game's faces from the game in {:.1f} s".format(
            count, time.perf_counter() - started))
        if reference.textools:
            self.report({'WARNING'}, "{:d} of the files the faces were read from are TexTools mods in the game's "
                                     "files, so the faces read from them are the mods' (the face tools say which): "
                                     "turn the mods off in TexTools and read again".format(len(reference.textools)))
        return {'FINISHED'}


class MAGIC_FIT_OT_face_pose(bpy.types.Operator):
    """Pose the face like one of the game's face animations, to check how it moves"""
    bl_idname = "magic_fit.face_pose"
    bl_label = "Test Pose"
    bl_options = {'REGISTER', 'UNDO'}

    pose: EnumProperty(
        name="Pose",
        items=[(pose, label, description) for pose, label, description in facing.POSES] + [
            ('REST', "Rest", "Put the face bones back in their rest pose")],
    )

    @classmethod
    def description(cls, _context, properties):
        if properties.pose == 'REST':
            return "Put the face bones back in their rest pose"
        label = next(description for pose, _label, description in facing.POSES if pose == properties.pose)
        return "Pose the face like the game's animation: {:s}".format(label[:1].lower() + label[1:])

    @classmethod
    def poll(cls, context):
        problem = check_face_pose_ready(context)
        if problem is not None:
            cls.poll_message_set(problem)
            return False
        return True

    def execute(self, context):
        settings = context.scene.magic_fit
        armature = pose_armature(context)
        reference = face_reference(self, context)
        if reference is None:
            return {'CANCELLED'}
        race, number = settings.face_race, settings.face_number
        meshes = face_targets(context)
        if settings.face_source == 'DETECT' and meshes:
            found = facing.detect(meshes, reference)
            if found is not None:
                race, number, _how = found
                remember_face(settings, race, number)
        if not reference.has_face(race, number):
            self.report({'ERROR'}, "The game has no face {:s}".format(facing.face_title(race, number)))
            return {'CANCELLED'}
        face = reference.face(race, number)
        skeleton = face.skeleton
        pose_bones = armature.pose.bones
        # Every pose starts from the rest pose of the face bones, so they don't add up.
        for name in skeleton.face_bones():
            bone = pose_bones.get(name)
            if bone is not None:
                bone.matrix_basis = Matrix.Identity(4)
        if self.pose == 'REST':
            return {'FINISHED'}
        deformations = skeleton.deformations(self.pose)
        if not deformations:
            self.report({'WARNING'}, "{:s} has no such animation".format(facing.face_title(race, number)))
            return {'CANCELLED'}
        context.view_layer.update()
        bases = facing.pose_bases(armature, deformations, facing.armature_offset(armature, skeleton))
        for name, basis in bases.items():
            pose_bones[name].matrix_basis = Matrix(basis.tolist())
        if armature.data.pose_position == 'REST':
            self.report({'WARNING'}, "Armature '{:s}' shows its rest position: set it to Pose Position to see the "
                                     "pose".format(armature.name))
        warning = textools_warning(reference, race, number)
        if warning is not None:
            self.report({'WARNING'}, warning)
        return {'FINISHED'}


class MAGIC_FIT_OT_resize(bpy.types.Operator):
    """Add a shape key to the selected meshes (or those in Edit Mode) that reshapes them from the From body
    to the To body"""
    bl_idname = "magic_fit.resize"
    bl_label = "Resize"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        problem = check_resize_ready(context)
        if problem is not None:
            cls.poll_message_set(problem)
            return False
        return True

    def execute(self, context):
        settings = context.scene.body_fit_brush
        body, to_body = settings.resize_from_object, settings.resize_to_object
        from_key, to_key = resize_keys(settings)
        name = resize_key_name(context)
        targets = resize_targets(context)
        started = time.perf_counter()
        # Shape keys can only be added outside Edit Mode, and a body in Edit Mode too (edited along with
        # the meshes) only shows its latest shape and UV map outside it.
        edit = context.mode == 'EDIT_MESH'
        if edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        window_manager = context.window_manager
        window_manager.progress_begin(0, 100)
        try:
            try:
                source, target, tris = resizing.body_shapes(body, from_key, to_body, to_key)
                change = resizing.ShapeChange(source, target, tris, radius=settings.resize_radius,
                                              bridge=settings.resize_bridge)
            except resizing.ResizeError as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}
            if change.largest <= 0.0:
                self.report({'ERROR'}, "The From and To shapes are the same")
                return {'CANCELLED'}
            shapes = [resizing.garment_shape(obj) for obj in targets]
            # Rigid parts are numbered across all meshes.
            groups, first_part = [], 0
            for obj, shape in zip(targets, shapes):
                parts, count = _rigid_parts(obj, shape, settings.resize_rigid_size)
                groups.append(np.where(parts >= 0, parts + first_part, -1))
                first_part += count
            rigid = np.concatenate(groups)
            edges = None
            if settings.resize_keep_close:
                # Keep Close spreads its corrections along the meshes' edges, numbered across all meshes.
                edges, first_vertex = [], 0
                for obj, shape in zip(targets, shapes):
                    pairs = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
                    obj.data.edges.foreach_get("vertices", pairs)
                    edges.append(pairs.reshape(-1, 2) + first_vertex)
                    first_vertex += len(shape)
                edges = np.concatenate(edges)
            moved = change.apply(
                np.concatenate(shapes), rigid=rigid, edges=edges,
                progress=lambda done: window_manager.progress_update(int(100 * done)),
            )
            first = 0
            for obj, shape in zip(targets, shapes):
                block, stuck = resizing.add_resized_key(obj, name, moved[first:first + len(shape)])
                first += len(shape)
                if stuck:
                    self.report({'WARNING'}, "'{:s}': driven shape keys still shape it on top of '{:s}': {:s}".format(
                        obj.name, block.name, ", ".join(stuck)))
        finally:
            window_manager.progress_end()
            if edit:
                bpy.ops.object.mode_set(mode='EDIT')

        # Body Fit fits against its Body: that should be the To body, showing the To shape.
        shown = to_body if to_body is not None else body
        needed = []
        if context.scene.magic_fit.target != shown:
            needed.append("pick '{:s}' as the Body".format(shown.name))
        if not resizing.shows_to_shape(body, from_key, to_body, to_key):
            needed.append("switch {:s} to the To shape".format("it" if needed else "'{:s}'".format(shown.name)))
        hint = "; to fit against it, " + " and ".join(needed) if needed else ""
        rigid_count = len(np.unique(rigid[rigid >= 0]))
        self.report({'INFO'}, "Added shape key '{:s}' to {:d} mesh(es){:s} in {:.1f} s ({:d} steps{:s}){:s}".format(
            block.name, len(targets), ", {:d} rigid part(s)".format(rigid_count) if rigid_count else "",
            time.perf_counter() - started, change.steps,
            ", bodies matched by UV map" if resizing.matches_by_uv(body, to_body) else "", hint))
        return {'FINISHED'}


def _joints_note(result):
    """Where Line Up found the model's joints, for its report."""
    if result.how == "bones":
        return "from its bones"
    if result.how == "unnamed":
        text = "estimated from its vertex groups, sorted by where they lie"
    else:
        text = "estimated from its vertex groups"
    if result.refined:
        text += ", {:d} corrected against the body's shape".format(result.refined)
    return text


def _missing_note(missing):
    """The parts only one of model and body has, which move with the part above them."""
    kinds = []
    for part in missing:
        kind = part.split("_")[0]
        label = {"upperarm": "upper arms", "forearm": "forearms", "shin": "shins", "thigh": "thighs",
                 "clavicle": "clavicles", "hand": "hands", "foot": "feet", "neck": "neck", "head": "head"}.get(kind, kind)
        if label not in kinds:
            kinds.append(label)
    return ", ".join(kinds)


class MAGIC_FIT_OT_line_up(bpy.types.Operator):
    """Scale, turn and pose the selected model and its armature so its hips, shoulders and joints line up
    with the body"""
    bl_idname = "magic_fit.line_up"
    bl_label = "Line Up"
    bl_options = {'REGISTER', 'UNDO'}

    stretch: BoolProperty(
        name="Stretch Limbs",
        description=(
            "Stretch arms and legs so the model's joints land on the body's. "
            "Off: keep the limb lengths and only turn them"
        ),
        default=False,
        options={'SKIP_SAVE'},
    )
    fit_shape: BoolProperty(
        name="Fit Shape",
        description=(
            "Then scale, turn and move each bone of the model a little, so its surface lies on the body's "
            "where the two are close. Off: keep the model's own shape"
        ),
        default=True,
        options={'SKIP_SAVE'},
    )

    @classmethod
    def poll(cls, context):
        problem = check_lineup_ready(context)
        if problem is not None:
            cls.poll_message_set(problem)
            return False
        return True

    def execute(self, context):
        settings = context.scene.body_fit_brush
        # From the panel the settings decide; from Adjust Last Operation this operator's own.
        if not self.properties.is_property_set("stretch"):
            self.stretch = settings.lineup_stretch
        if not self.properties.is_property_set("fit_shape"):
            self.fit_shape = settings.lineup_fit_shape
        settings.lineup_stretch = self.stretch
        settings.lineup_fit_shape = self.fit_shape
        body = context.scene.magic_fit.target
        meshes, armatures = lineup_model(context)
        armature = armatures[0] if armatures else None
        started = time.perf_counter()
        try:
            result = lineup.plan(meshes, armature, body, context.evaluated_depsgraph_get(), stretch=self.stretch,
                                 fit=self.fit_shape)
        except lineup.LineUpError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        lineup.write_meshes(result)
        moved = (result.move_armature or bool(result.bone_places)) and self._move_rest_pose(context, result)

        solution = result.solution
        what = "{:d} mesh{:s}".format(len(meshes), "es" if len(meshes) != 1 else "")
        if moved:
            what += " and '{:s}'".format(armature.name)
        details = ["{:.0f} % of its size".format(100.0 * solution.scale)]
        if solution.half == "upper":
            details.append("only an upper body (no hips), sized by its arms")
        elif solution.half == "lower":
            details.append("only a lower body (no shoulders), sized by its legs")
        if abs(solution.turn) >= 2.0:
            details.append("turned {:.0f}°".format(solution.turn))
        limbs = [factor for part, factor in solution.stretches.items()
                 if part.split("_")[0] in ("upperarm", "forearm", "thigh", "shin")]
        if limbs:
            low, high = (int(round(100.0 * (factor - 1.0))) for factor in (min(limbs), max(limbs)))
            details.append("limbs stretched {:+d} to {:+d} %".format(low, high))
        if solution.fingers:
            details.append("{:d} finger{:s} turned like the body's".format(
                solution.fingers, "s" if solution.fingers != 1 else ""))
        if result.shape is not None:
            details.append("shape fitted: typically {:.1f} mm off the body, from {:.1f} mm".format(
                1000.0 * result.shape.after, 1000.0 * result.shape.before))
        self.report({'INFO'}, "Lined up {:s} with '{:s}' in {:.1f} s: {:s}; joints {:s}".format(
            what, body.name, time.perf_counter() - started, ", ".join(details), _joints_note(result)))
        # A top has no legs to warn about, a bottom no arms or head.
        other_half = lineup.HALF_PARTS["lower" if solution.half == "upper" else "upper"] if solution.half else set()
        missing = [part for part in solution.missing if part in result.target_joints and
                   part not in lineup.DIGIT_PARTS and part not in other_half]
        if missing:
            self.report({'WARNING'}, "The model has no {:s}: moved with the part above".format(
                _missing_note(missing)))
        if moved and result.bone_places:
            self.report({'WARNING'}, "The bones of '{:s}' all sat at one spot: they were put where their vertex "
                                     "groups are".format(armature.name))
        elif armature is not None and not result.move_armature and not result.bone_places:
            self.report({'WARNING'}, "'{:s}' has no usable joints (its bones sit at one spot or have unknown "
                                     "names), so its rest pose was left as it was".format(armature.name))
        return {'FINISHED'}

    def _move_rest_pose(self, context, result):
        """Move the armature's rest pose along, or put its bones where their vertex groups are, in its Edit
        Mode. Returns whether it could."""
        armature = result.armature
        view_layer = context.view_layer
        active = view_layer.objects.active
        selected = list(context.selected_objects)
        hidden = armature.hide_get(), armature.hide_viewport
        armature.hide_set(False)
        armature.hide_viewport = False
        try:
            if not armature.visible_get() or armature.library is not None or armature.data.library is not None:
                self.report({'WARNING'}, "Couldn't edit '{:s}' (hidden in a hidden collection, or linked): its "
                                         "rest pose wasn't moved along".format(armature.name))
                return False
            for obj in selected:
                obj.select_set(False)
            view_layer.objects.active = armature
            armature.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            try:
                if result.move_armature:
                    lineup.move_rest_pose(result)
                else:
                    lineup.place_bones(result)
            finally:
                bpy.ops.object.mode_set(mode='OBJECT')
            return True
        finally:
            armature.select_set(False)
            for obj in selected:
                obj.select_set(True)
            view_layer.objects.active = active
            armature.hide_set(hidden[0])
            armature.hide_viewport = hidden[1]


class MAGIC_FIT_OT_use_tool(bpy.types.Operator):
    """Switch the active mesh to this brush's mode and pick the brush"""
    bl_idname = "magic_fit.use_tool"
    bl_label = "Use Brush"
    bl_options = {'UNDO'}

    tool: EnumProperty(
        name="Brush",
        items=(
            ('WEIGHT', "Weight Brushes", "Weight Paint mode brushes that copy, straighten or make weights"),
            ('FIT', "Body Fit", "Edit Mode brush that fits the mesh around the body"),
            ('RELAX', "Texture Relax", "Edit Mode brush that slides vertices until the texture is even again"),
            ('MOVE', "Fit Move", "Edit Mode tool that moves the selection without reaching across gaps"),
        ),
    )

    # (object mode, tool id, sidebar tab) per brush.
    targets = {
        'WEIGHT': ('WEIGHT_PAINT', "magic_fit.tool", 'WEIGHTS'),
        'FIT': ('EDIT', "magic_fit.body_fit_tool", 'FIT'),
        'RELAX': ('EDIT', "magic_fit.texture_relax_tool", 'RELAX'),
        'MOVE': ('EDIT', "magic_fit.fit_move_tool", 'FIT'),
    }

    @classmethod
    def description(cls, _context, properties):
        if properties.tool == 'FIT':
            return "Switch to Edit Mode and pick the Body Fit brush"
        if properties.tool == 'RELAX':
            return "Switch to Edit Mode and pick the Texture Relax brush"
        if properties.tool == 'MOVE':
            return "Switch to Edit Mode and pick the Fit Move tool"
        return "Switch to Weight Paint mode and pick the Weight Brushes"

    @classmethod
    def poll(cls, context):
        obj = context.object
        if obj is None or obj.type != 'MESH':
            cls.poll_message_set("Select a mesh first")
            return False
        return context.space_data is not None and context.space_data.type == 'VIEW_3D'

    def execute(self, context):
        mode, idname, tab = self.targets[self.tool]
        try:
            if context.object.mode != mode:
                bpy.ops.object.mode_set(mode=mode)
            bpy.ops.wm.tool_set_by_id(name=idname)
        except RuntimeError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        # The sidebar shows the brush's settings on its tab.
        if context.workspace is not None:
            context.workspace.magic_fit_tab = tab
        return {'FINISHED'}


classes = (
    MAGIC_FIT_OT_paint,
    MAGIC_FIT_OT_body_fit,
    MAGIC_FIT_OT_texture_relax,
    MAGIC_FIT_OT_fit_move,
    MAGIC_FIT_OT_skirt_weights,
    MAGIC_FIT_OT_heel_weights,
    MAGIC_FIT_OT_hair_weights,
    MAGIC_FIT_OT_pick_hair_skeleton,
    MAGIC_FIT_OT_face_weights,
    MAGIC_FIT_OT_face_repair,
    MAGIC_FIT_OT_detect_face,
    MAGIC_FIT_OT_pick_face,
    MAGIC_FIT_OT_face_pose,
    MAGIC_FIT_OT_read_game_faces,
    MAGIC_FIT_OT_resize,
    MAGIC_FIT_OT_line_up,
    MAGIC_FIT_OT_use_tool,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
