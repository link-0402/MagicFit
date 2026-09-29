# SPDX-License-Identifier: GPL-3.0-or-later
#
# Operators of Robust Weight Transfer by sentfromspacevr (https://github.com/sentfromspacevr), GPL-2.0-or-later.
# Changed for Magic Fit (2026-09): renamed into this add-on, the source is the body
# shared by all tools, the solver modules (which load SciPy) are imported on first use, meshes following
# the body's armature are matched in the rest pose and others without Customize+, and there's no
# Smoothed Limit Vertex Groups operator or Visualize Rejected Weights.

import contextlib

import bmesh
import bpy
import numpy as np

from . import dependencies, util


def engine():
    """The solver modules `transfer` and `weighttransfer`, imported on first use: they load SciPy."""
    from . import transfer, weighttransfer
    return transfer, weighttransfer


def _load_engine(operator):
    """`engine()`, or None after reporting why it can't be loaded."""
    try:
        return engine()
    except ImportError as error:
        operator.report({'ERROR'}, "Could not load SciPy or robust-laplacian: {!s}".format(error))
        return None


def transfer_targets(context, settings, body):
    """The meshes Transfer Weights writes to: the selected ones or the active one, never the body."""
    objects = context.selected_objects if settings.apply_to_selected else [context.object]
    return [obj for obj in objects if obj is not None and obj != body and obj.type == 'MESH']


def check_transfer_ready(context):
    """Return why Transfer Weights can't run right now, or None when it can."""
    if context.mode not in {'OBJECT', 'PAINT_WEIGHT'}:
        return "Transfer in Object or Weight Paint mode"
    problem = dependencies.problem()
    if problem is not None:
        return problem
    settings = context.scene.magic_fit_transfer
    body = context.scene.magic_fit.target
    if body is None:
        return "Pick the body to transfer weights from"
    if body.type != 'MESH':
        return "The body must be a mesh"
    if settings.group_selection == 'DEFORM_POSE_BONES':
        armatures = [mod for mod in body.modifiers if mod.type == 'ARMATURE']
        # With Customize+ on, the body's own modifier is switched off for the rig's copy of it.
        armatures = [mod for mod in armatures if mod.show_viewport] or armatures
        if not armatures:
            return "Body '{:s}' has no Armature modifier".format(body.name)
        if len(armatures) > 1:
            return "Body '{:s}' has several Armature modifiers".format(body.name)
        if armatures[0].object is None:
            return "The Armature modifier of '{:s}' has no armature".format(body.name)
    targets = transfer_targets(context, settings, body)
    if not targets:
        if settings.apply_to_selected:
            return "Select the meshes to transfer weights to"
        return "Make a mesh other than the body active"
    if settings.use_deformed_target:
        for obj in targets:
            if util.has_modifier(obj, *util.TOPOLOGY_MODS):
                return "'{:s}' has a modifier that changes its topology".format(obj.name)
    if not settings.apply_to_selected:
        obj = targets[0]
        object_settings = obj.magic_fit_transfer
        for group in (object_settings.vertex_group, object_settings.inpaint_group):
            if group and group not in obj.vertex_groups:
                return "'{:s}' has no vertex group '{:s}'".format(obj.name, group)
    return None


def _check_inpaint_ready(context):
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        return "Make a mesh active"
    if context.mode not in {'OBJECT', 'PAINT_WEIGHT'}:
        return "Inpaint in Object or Weight Paint mode"
    problem = dependencies.problem()
    if problem is not None:
        return problem
    group = obj.magic_fit_transfer.inpaint_group
    if not group:
        return "Pick the Inpaint Mask of '{:s}'".format(obj.name)
    if group not in obj.vertex_groups:
        return "'{:s}' has no vertex group '{:s}'".format(obj.name, group)
    if context.scene.magic_fit_transfer.use_deformed_target and util.has_modifier(obj, *util.TOPOLOGY_MODS):
        return "'{:s}' has a modifier that changes its topology".format(obj.name)
    return None


def _check_select_ready(context):
    obj = context.active_object
    if context.mode != 'EDIT_MESH' or obj is None:
        return "Select rejected parts in Edit Mode"
    problem = dependencies.problem()
    if problem is not None:
        return problem
    body = context.scene.magic_fit.target
    if body is None:
        return "Pick the body to transfer weights from"
    if body == obj:
        return "Edit a mesh other than the body"
    if context.scene.magic_fit_transfer.use_deformed_target and util.has_modifier(obj, *util.TOPOLOGY_MODS):
        return "'{:s}' has a modifier that changes its topology".format(obj.name)
    return None


def _armatures_of(obj):
    """The armatures deforming ``obj``, with their Customize+ rigs and the armatures the rigs belong to."""
    from ..cplus import rig as cplus_rig
    found = set()
    for mod in obj.modifiers:
        if mod.type != 'ARMATURE' or mod.object is None:
            continue
        found.add(mod.object)
        if cplus_rig.is_rig(mod.object) and mod.object.parent is not None:
            found.add(mod.object.parent)
        rig = cplus_rig.rig_of(mod.object)
        if rig is not None:
            found.add(rig)
    return found


@contextlib.contextmanager
def _rest_pose(objects):
    """Switch off the Armature modifiers of ``objects`` while the block runs."""
    switched = []
    try:
        for obj in objects:
            for mod in obj.modifiers:
                if mod.type == 'ARMATURE' and mod.show_viewport:
                    mod.show_viewport = False
                    switched.append(mod)
        yield
    finally:
        for mod in switched:
            mod.show_viewport = True


@contextlib.contextmanager
def _without_cplus(objects):
    """Show ``objects`` through their own armatures instead of their Customize+ rigs while the block runs."""
    from ..cplus import rig as cplus_rig
    changed = []
    try:
        for obj in objects:
            modifiers = obj.modifiers
            for index, mod in enumerate(modifiers):
                if mod.type != 'ARMATURE' or not mod.show_viewport or not cplus_rig.is_rig(mod.object):
                    continue
                changed.append((mod, True))
                mod.show_viewport = False
                original = modifiers[index - 1] if index else None
                if (original is not None and original.type == 'ARMATURE' and not original.show_viewport
                        and original.object is not None and original.object == mod.object.parent):
                    changed.append((original, False))
                    original.show_viewport = True
        yield
    finally:
        for mod, value in reversed(changed):
            mod.show_viewport = value


def _bound(body, objects):
    """Those of ``objects`` that follow the body's armature (or its Customize+ rig)."""
    armatures = _armatures_of(body) if body is not None else set()
    return [obj for obj in objects if armatures and _armatures_of(obj) & armatures]


def match_objects(context, settings, body, objects):
    """Match ``objects`` to the body. Meshes that follow the body's armature (or its Customize+ rig) are
    matched with both in the rest pose, where their weights apply, so neither the pose nor Customize+
    nor the weights they have now change the result. Others are matched as displayed, but without
    Customize+ on either. Returns the targets (in the order of ``objects``) and the body's group names,
    which of them are deform bones and which are transferred."""
    transfer, weighttransfer = engine()
    bound = _bound(body, objects)
    loose = [obj for obj in objects if obj not in bound]
    targets, groups = {}, None
    fallback = transfer.rest_armature(body)
    for meshes, state, posed in ((bound, lambda: _rest_pose([body] + bound), False),
                                 (loose, lambda: _without_cplus([body] + loose), settings.use_deformed_source)):
        if not meshes:
            continue
        with state():
            depsgraph = context.evaluated_depsgraph_get()
            source = body.evaluated_get(depsgraph) if settings.use_deformed_source else body
            vertices, triangles, normals = util.get_obj_arrs_world(source)
            surface_bvh = weighttransfer.build_surface_bvh(vertices, triangles)
            names = [g.name for g in source.vertex_groups]
            deform = [util.is_vertex_group_deform_bone(source, name) for name in names]
            included = deform if settings.group_selection == 'DEFORM_POSE_BONES' else [True] * len(names)
            if not any(included):
                raise ValueError(f'Body {body.name} has no transferable vertex groups')
            weights = util.get_groups_arr(source, included)
            strides = (transfer.stride_source(body, names, vertices, weights, surface_bvh, posed)
                       if settings.crease_smoothing else None)
            for obj in meshes:
                target = transfer.make_target(obj, depsgraph, settings)
                target['deform_armature'] = transfer.deform_armature(obj, fallback)
                transfer.match_target(target, vertices, triangles, normals, weights, settings, surface_bvh,
                                      strides)
                targets[obj] = target
            groups = (names, deform, included)
    return [targets[obj] for obj in objects], groups


def _poll(cls, problem):
    if problem is not None:
        cls.poll_message_set(problem)
        return False
    return True


class MAGIC_FIT_OT_transfer_weights(bpy.types.Operator):
    """Transfer the body's weights onto the mesh: copied where it matches the body, filled in elsewhere"""
    bl_idname = "magic_fit.transfer_weights"
    bl_label = "Transfer Weights"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _poll(cls, check_transfer_ready(context))

    def execute(self, context):
        loaded = _load_engine(self)
        if loaded is None:
            return {'CANCELLED'}
        transfer, _weighttransfer = loaded
        settings = context.scene.magic_fit_transfer
        source_original = context.scene.magic_fit.target
        target_objs = transfer_targets(context, settings, source_original)
        targets = []
        try:
            targets, (names, deform, included) = match_objects(context, settings, source_original, target_objs)
            fallback = transfer.source_armature(source_original)
            transfer.solve_targets(targets, settings, fallback, partial=settings.partial_reweight)
            for target in targets:
                transfer.process_weights(target, settings)
                transfer.stage_weights(target, names, included, apply_mask=not settings.apply_to_selected,
                                       clear_others=True)
            counts = transfer.synchronize_targets(
                targets, settings, [name for name, use in zip(names, deform) if use], fallback)
            post_counts = transfer.postprocess_transfer_targets(
                targets, settings, [name for name, use in zip(names, deform) if use])
            # Last, so nothing changes the rigid meshes' weights afterwards. None: not asked for.
            made_rigid = [
                transfer.make_target_rigid(target, settings, [name for name, use in zip(names, deform) if use])
                if target['obj'].magic_fit_transfer.rigid else None
                for target in targets
            ]
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

        # All solves and final seam reconciliation must succeed before any weights
        # are committed to a target.
        transfer.write_targets(targets)
        transfer.report_seams(self, counts)
        transfer.report_partial(self, targets)
        transfer.report_stability(self, targets)
        transfer.report_postprocess(self, post_counts)
        transfer.report_rigid(self, targets, made_rigid)
        self.report({'INFO'}, f'Weights transferred from {source_original.name} to {len(targets)} object(s)')
        return {'FINISHED'}


class MAGIC_FIT_OT_select_rejected(bpy.types.Operator):
    """Select loose parts with nothing matching the body. They make filling in weights fail"""
    bl_idname = "magic_fit.select_rejected"
    bl_label = "Select Rejected Loose Parts"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _poll(cls, _check_select_ready(context))

    def execute(self, context):
        loaded = _load_engine(self)
        if loaded is None:
            return {'CANCELLED'}
        transfer, _weighttransfer = loaded
        active = context.active_object
        settings = context.scene.magic_fit_transfer
        bpy.ops.object.mode_set(mode='OBJECT')
        try:
            source_original = context.scene.magic_fit.target
            across = (settings.apply_to_selected and settings.seam_sync
                      and settings.seam_sync_across_objects and settings.virtual_merge)
            objects = ([obj for obj in context.selected_objects
                        if obj != source_original and obj.type == 'MESH'] if across else [])
            # Edit Mode also edits the active mesh when it isn't selected.
            if active not in objects:
                objects.append(active)
            targets, _groups = match_objects(context, settings, source_original, objects)
            batch = next(batch for batch in transfer.batches(
                targets, across, transfer.source_armature(source_original))
                if any(t['obj'] == active for t in batch))
            domain, offsets = transfer.prepare_batch(batch, settings)
            index = next(i for i, t in enumerate(batch) if t['obj'] == active)
            selects = domain.rejected[offsets[index]:offsets[index + 1]]
            if settings.partial_reweight:
                selects &= batch[index]['strength'] > 0
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        finally:
            bpy.ops.object.mode_set(mode='EDIT')
        mesh = bmesh.from_edit_mesh(active.data)
        mesh.verts.ensure_lookup_table()
        for vertex, selected in zip(mesh.verts, selects):
            vertex.select_set(bool(selected))
        mesh.select_flush(True)
        mesh.select_flush(False)
        bmesh.update_edit_mesh(active.data, destructive=False)
        self.report({'INFO'}, f'Selected {np.count_nonzero(selects)} out of {len(selects)} vertices.')
        return {'FINISHED'}


class MAGIC_FIT_OT_inpaint_weights(bpy.types.Operator):
    """Fill in the weights inside the Inpaint Mask from the weights around it"""
    bl_idname = "magic_fit.inpaint_weights"
    bl_label = "Inpaint"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _poll(cls, _check_inpaint_ready(context))

    def execute(self, context):
        loaded = _load_engine(self)
        if loaded is None:
            return {'CANCELLED'}
        transfer, _weighttransfer = loaded
        settings = context.scene.magic_fit_transfer
        obj = context.active_object
        try:
            # Shown as Transfer Weights matches it: in the rest pose if it follows the body's armature,
            # otherwise without Customize+.
            with (_rest_pose if _bound(context.scene.magic_fit.target, [obj]) else _without_cplus)([obj]):
                target = transfer.make_target(obj, context.evaluated_depsgraph_get(), settings)
            names = [g.name for g in obj.vertex_groups]
            deform = [util.is_vertex_group_deform_bone(obj, name) for name in names]
            if not any(deform):
                raise ValueError(f'{obj.name} has no deform weights to inpaint')
            mask = transfer.inpaint_mask(obj)
            weights = util.get_groups_arr(obj, deform)
            target.update(weights=weights, matched=~mask)
            transfer.solve_targets([target], settings)
            target['weights'][~mask] = weights[~mask]
            transfer.stage_weights(target, names, deform)
            target['protected'] = ~mask
            target['write_vertices'] = mask
            counts = transfer.synchronize_targets([target], settings,
                                                  [name for name, use in zip(names, deform) if use])
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        transfer.write_targets([target])
        transfer.report_seams(self, counts)
        transfer.report_stability(self, [target])
        self.report({'INFO'}, 'Weights inpainted.')
        return {'FINISHED'}


class MAGIC_FIT_OT_reset_transfer_settings(bpy.types.Operator):
    """Reset all Weight Transfer settings to their defaults"""
    bl_idname = "magic_fit.reset_transfer_settings"
    bl_label = "Reset to Defaults"
    bl_options = {'UNDO'}

    def execute(self, context):
        settings = context.scene.magic_fit_transfer
        for prop_name, prop in settings.bl_rna.properties.items():
            if prop.is_readonly or prop_name in {'rna_type', 'name'}:
                continue
            if hasattr(prop, 'default'):
                setattr(settings, prop_name, prop.default)
            else:
                setattr(settings, prop_name, None)
        return {'FINISHED'}


classes = (
    MAGIC_FIT_OT_transfer_weights,
    MAGIC_FIT_OT_select_rejected,
    MAGIC_FIT_OT_inpaint_weights,
    MAGIC_FIT_OT_reset_transfer_settings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
