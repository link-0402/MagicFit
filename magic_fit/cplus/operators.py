# SPDX-License-Identifier: GPL-3.0-or-later

import bpy

from . import reference, rig, solver
from .template import ROOT_BONE, TEMPLATE_VERSION, Template, TemplateError, parse

# What Apply shows with the Bust Size alone, without a template.
NO_TEMPLATE = Template("", TEMPLATE_VERSION, {})

_parsed = {}


def parsed(text):
    """(Template, None) or (None, message) for ``text``, cached: the panel asks on every redraw."""
    text = (text or "").strip()
    if text not in _parsed:
        if len(_parsed) > 16:
            _parsed.clear()
        try:
            _parsed[text] = (parse(text), None)
        except TemplateError as error:
            _parsed[text] = (None, str(error))
    return _parsed[text]


def bust(context):
    """The Bust Size to show, or None."""
    settings = context.scene.magic_fit_cplus
    return settings.bust_size if settings.use_bust else None


def current_template(context):
    """(Template, None) to apply, or (None, message): the pasted template, or with nothing pasted and the
    Bust Size on, no template."""
    settings = context.scene.magic_fit_cplus
    if not settings.template.strip():
        if settings.use_bust:
            return NO_TEMPLATE, None
        return None, "Paste a Customize+ template, or turn on Bust Size"
    return parsed(settings.template)


def bust_text(rig_object):
    """"Bust Size 70" (with "(Lalafell)" for their range), or None: the bust size a rig shows."""
    size, lalafell = rig.bust_state(rig_object)
    return None if size is None else "Bust Size {:d}{:s}".format(size, " (Lalafell)" if lalafell else "")


def shown(rig_object):
    """What a rig shows, for reports: "'Template', Bust Size 70", or "nothing"."""
    template, _options = rig.state(rig_object)
    parts = ["'{:s}'".format(template.name)] if template.bones else []
    return ", ".join(parts + [text for text in (bust_text(rig_object),) if text]) or "nothing"


def _armature_of(obj):
    """The armature bending ``obj`` (its first Armature modifier's), the armature itself, or None."""
    if obj is None:
        return None
    if obj.type == 'ARMATURE':
        return obj.parent if rig.is_rig(obj) else obj
    for modifier in getattr(obj, "modifiers", ()):
        if modifier.type == 'ARMATURE' and modifier.object is not None and modifier.object.type == 'ARMATURE':
            return modifier.object.parent if rig.is_rig(modifier.object) else modifier.object
    return None


def target_armature(context):
    """The armature of One Armature: the one picked, else the body's, else the active object's."""
    settings = context.scene.magic_fit_cplus
    if settings.armature is not None:
        return settings.armature
    body = getattr(context.scene, "magic_fit", None)
    armature = _armature_of(body.target) if body is not None else None
    return armature if armature is not None else _armature_of(context.active_object)


def _scene_armatures(context):
    # Objects deleted since the view layer was last updated are still listed, as None.
    return [obj for obj in context.view_layer.objects
            if obj is not None and obj.type == 'ARMATURE' and not rig.is_rig(obj)]


def apply_targets(context, template):
    """The armatures ``template`` (and the Bust Size) go on: the one armature, or every armature in the view
    layer with one of its bones (the body's, and those of separately imported faces, hair and gear) or,
    with the Bust Size on, with breast bones. A template that moves or scales the root (the whole
    character) goes on every armature with a bone of the game's skeleton, with an n_root or not."""
    if context.scene.magic_fit_cplus.scope == 'ONE':
        armature = target_armature(context)
        return [armature] if armature is not None else []
    edits = template.edited()
    names = set(edits)
    if solver.root_change(edits.get(ROOT_BONE), options(context)) is not None:
        names.update(reference.BONES)
    if bust(context) is not None:
        names.update(solver.BUST_BONES)
    return [armature for armature in _scene_armatures(context)
            if any(bone.name in names for bone in armature.data.bones)]


def rigged(context):
    """The armatures in scope that have a Customize+ rig."""
    if context.scene.magic_fit_cplus.scope == 'ONE':
        armature = target_armature(context)
        return [armature] if rig.rig_of(armature) is not None else []
    return [armature for armature in _scene_armatures(context) if rig.rig_of(armature) is not None]


def options(context):
    settings = context.scene.magic_fit_cplus
    return solver.Options(settings.use_scale, settings.use_rotation, settings.use_translation)


def armature_problem(context):
    """Why One Armature has nothing to work on, or None."""
    if context.scene.magic_fit_cplus.scope != 'ONE':
        return None
    armature = target_armature(context)
    if armature is None:
        return "Pick an armature (or a body with one)"
    if armature.mode == 'EDIT':
        return "Leave the Edit Mode of '{:s}' first".format(armature.name)
    return None


def apply_problem(context):
    """Why the template can't be applied yet, or None."""
    template, error = current_template(context)
    if template is None:
        return error
    problem = armature_problem(context)
    if problem is not None:
        return problem
    targets = apply_targets(context, template)
    if not targets:
        return "No armature has the template's bones" if template.bones else "No armature has breast bones"
    if all(armature.mode == 'EDIT' for armature in targets):
        return "Leave the Edit Mode of '{:s}' first".format(targets[0].name)
    return None


def _plural(count, word, words=None):
    return "{:d} {:s}".format(count, word if count == 1 else words or word + "s")


def _names(names, limit=5):
    shown = ", ".join(names[:limit])
    return shown + (", ..." if len(names) > limit else "")


def _quoted(names, limit=3):
    return _names(["'{:s}'".format(name) for name in names], limit)


def _where(armatures):
    names = [armature.name for armature in armatures]
    if len(names) == 1:
        return "'{:s}'".format(names[0])
    return "{:d} armatures ({:s})".format(len(names), _quoted(names))


def apply(operator, context, template):
    """Build the rigs of the target armatures for ``template`` and the Bust Size, and switch their meshes
    over. With All Armatures, the other armatures' rigs are removed, so only this template shows."""
    targets = apply_targets(context, template)
    for armature in targets:
        if armature.mode == 'EDIT':
            operator.report({'WARNING'}, "'{:s}' is in Edit Mode: left out".format(armature.name))
    targets = [armature for armature in targets if armature.mode != 'EDIT']
    if not targets:
        operator.report({'ERROR'}, "No armature has the template's bones" if template.bones
                        else "No armature has breast bones")
        return {'CANCELLED'}
    if context.scene.magic_fit_cplus.scope == 'ALL':
        for armature in rigged(context):
            if armature not in targets and armature.mode != 'EDIT':
                rig.remove(armature)
    size = bust(context)
    meshes = 0
    edited = set()
    missing = None
    busts = []  # (armature, Plan) of the armatures that got the Bust Size
    for armature in targets:
        rig_object, result = rig.build(context, armature, template, options(context), size)
        meshes += rig.connect(armature, rig_object)
        edited.update(result.edited)
        missing = set(result.missing) if missing is None else missing & set(result.missing)
        if result.bust:
            busts.append((armature, result))
        if result.unplaced:
            operator.report({'WARNING'}, "'{:s}' has {:s} without a rest pose (at its origin), left out: {:s}".format(
                armature.name, _plural(len(result.unplaced), "bone"), _names(result.unplaced)))
        if result.axes.known == 0 and (result.edited or result.bust):
            operator.report({'WARNING'}, "'{:s}' has no bone of the game's skeleton: bone axes are used as they "
                                         "are".format(armature.name))
    lalafell = " (Lalafell)" if any(result.lalafell for _armature, result in busts) else ""
    if edited:
        operator.report({'INFO'}, "Customize+ '{:s}'{:s} on {:s}: {:s}, {:s}".format(
            template.name, " and Bust Size {:d}{:s}".format(size, lalafell) if busts else "", _where(targets),
            _plural(len(edited), "bone"), _plural(meshes, "mesh", "meshes")))
    else:
        if template.bones:
            operator.report({'WARNING'}, "None of the template's bones could be changed on {:s}".format(
                _where(targets)))
        if busts:
            operator.report({'INFO'}, "Bust Size {:d}{:s} on {:s}: {:s}, {:s}".format(
                size, lalafell, _where([armature for armature, _result in busts]),
                _plural(sum(len(result.bust) for _armature, result in busts), "bone"),
                _plural(meshes, "mesh", "meshes")))
    if size is not None and not busts:
        operator.report({'WARNING'}, "No armature has breast bones (j_mune_l, j_mune_r): Bust Size left out")
    if template.version > TEMPLATE_VERSION:
        operator.report({'WARNING'}, "Template version {:d} is newer than this add-on knows ({:d}); "
                                     "it was read the same way".format(template.version, TEMPLATE_VERSION))
    if missing:
        operator.report({'WARNING'}, "Not in {:s}, left out: {:s}".format(
            "any armature" if len(targets) > 1 else "'{:s}'".format(targets[0].name), _names(sorted(missing))))
    if not meshes:
        operator.report({'WARNING'}, "No mesh follows {:s} (with an Armature modifier)".format(_where(targets)))
    return {'FINISHED'}


class MAGIC_FIT_OT_cplus_paste(bpy.types.Operator):
    """Paste a template copied with its Copy button in Customize+"""
    bl_idname = "magic_fit.cplus_paste"
    bl_label = "Paste Template"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        text = context.window_manager.clipboard.strip()
        template, error = parsed(text)
        if template is None:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        context.scene.magic_fit_cplus.template = text
        self.report({'INFO'}, "Pasted '{:s}' ({:s})".format(template.name, _plural(len(template.edited()), "bone")))
        return {'FINISHED'}


class MAGIC_FIT_OT_cplus_apply(bpy.types.Operator):
    """Show the template and Bust Size on the armatures' meshes, like in the game, in any pose. The armatures, their
    poses and animations aren't changed"""
    bl_idname = "magic_fit.cplus_apply"
    bl_label = "Apply Customize+"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        problem = apply_problem(context)
        if problem is not None:
            cls.poll_message_set(problem)
            return False
        return True

    def execute(self, context):
        template, error = current_template(context)
        if template is None:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        return apply(self, context, template)


class MAGIC_FIT_OT_cplus_toggle(bpy.types.Operator):
    """Turn Customize+ on or off. Shortcut: Alt Shift C in the 3D Viewport (set in the add-on's
    preferences)"""
    bl_idname = "magic_fit.cplus_toggle"
    bl_label = "Toggle Customize+"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        problem = None if rigged(context) else apply_problem(context)
        if problem is not None:
            cls.poll_message_set(problem)
            return False
        return True

    def execute(self, context):
        armatures = [armature for armature in rigged(context) if armature.mode != 'EDIT']
        if not armatures:
            template, error = current_template(context)
            if template is None:
                self.report({'ERROR'}, error)
                return {'CANCELLED'}
            return apply(self, context, template)
        if any(rig.is_on(armature) for armature in armatures):
            for armature in armatures:
                rig.disconnect(armature)
            self.report({'INFO'}, "Customize+ off on {:s}".format(_where(armatures)))
            return {'FINISHED'}
        meshes = 0
        names = set()
        for armature in armatures:
            rig_object = rig.ensure(context, armature)
            meshes += rig.connect(armature, rig_object)
            names.add(shown(rig_object))
        self.report({'INFO'}, "Customize+ {:s} on {:s} ({:s})".format(
            "; ".join(sorted(names)), _where(armatures), _plural(meshes, "mesh", "meshes")))
        return {'FINISHED'}


class MAGIC_FIT_OT_cplus_remove(bpy.types.Operator):
    """Delete the Customize+ rigs and switch the meshes back to their armatures. Poses and animations
    stay as they are"""
    bl_idname = "magic_fit.cplus_remove"
    bl_label = "Remove Customize+"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not rigged(context):
            cls.poll_message_set("No Customize+ template is applied")
            return False
        return True

    def execute(self, context):
        armatures = rigged(context)
        for armature in armatures:
            rig.remove(armature)
        self.report({'INFO'}, "Customize+ removed from {:s}".format(_where(armatures)))
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Keeping the Bust Size up to date

def bust_changed(context):
    """Show a changed Bust Size right away on the rigs in scope (Apply makes them)."""
    size = bust(context)
    for armature in rigged(context):
        if armature.mode != 'EDIT':
            rig.set_bust(context, armature, size)


classes = (
    MAGIC_FIT_OT_cplus_paste,
    MAGIC_FIT_OT_cplus_apply,
    MAGIC_FIT_OT_cplus_toggle,
    MAGIC_FIT_OT_cplus_remove,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
