# SPDX-License-Identifier: GPL-3.0-or-later

import bpy

from .. import tabs
from . import rig
from .operators import (
    MAGIC_FIT_OT_cplus_apply,
    MAGIC_FIT_OT_cplus_paste,
    MAGIC_FIT_OT_cplus_remove,
    MAGIC_FIT_OT_cplus_toggle,
    apply_targets,
    armature_problem,
    bust_text,
    parsed,
    rigged,
    target_armature,
)
from .template import ROOT_BONE, TEMPLATE_VERSION


def hotkey(idname):
    """The key the user set for operator ``idname`` in the 3D Viewport, as text, or ''."""
    keyconfig = bpy.context.window_manager.keyconfigs.user
    if keyconfig is None:
        return ""
    keymap = keyconfig.keymaps.get("3D View Generic")
    if keymap is None:
        return ""
    for item in keymap.keymap_items:
        if item.idname == idname and item.active:
            return item.to_string()
    return ""


def _names(armatures, limit=2):
    names = ["'{:s}'".format(armature.name) for armature in armatures]
    return ", ".join(names[:limit]) + (" +{:d}".format(len(names) - limit) if len(names) > limit else "")


def _applied(box, armatures, on):
    """The first lines of the box: what the rigs of ``armatures`` show."""
    names = sorted({"'{:s}'".format(template.name) for template in
                    (rig.state(rig.rig_of(armature))[0] for armature in armatures) if template.bones})
    busts = sorted({text for text in (bust_text(rig.rig_of(armature)) for armature in armatures) if text})
    first = ", ".join(names) or ", ".join(busts) or "nothing"
    box.label(text="{:s}: {:s}".format("On" if on else "Off", first), icon='CHECKMARK' if on else 'HIDE_ON')
    for text in busts if names else ():
        box.label(text=text, icon='BLANK1')


class VIEW3D_PT_customize_plus_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Show a Customize+ template on the armatures, the way it looks in the game"""
    bl_label = "Customize+"
    bl_order = 1
    tab = 'CPLUS'

    def draw(self, context):
        settings = context.scene.magic_fit_cplus
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        row = col.row(align=True)
        row.use_property_split = False  # across the full width, so the names stay readable
        row.prop(settings, "scope", expand=True)
        if settings.scope == 'ONE':
            armature = target_armature(context)
            row = col.row()
            row.alert = armature is None
            row.prop(settings, "armature")
            if settings.armature is None and armature is not None:
                col.label(text="Using '{:s}'".format(armature.name), icon='OUTLINER_OB_ARMATURE')

        row = col.row(align=True)
        row.prop(settings, "template", text="Template")
        row.operator(MAGIC_FIT_OT_cplus_paste.bl_idname, text="", icon='PASTEDOWN')
        template, error = parsed(settings.template)
        if settings.template.strip():
            if template is None:
                col.box().label(text=error, icon='ERROR')
            else:
                edited = template.edited()
                box = col.box().column(align=True)
                box.label(text="'{:s}'".format(template.name), icon='FILE_TEXT')
                box.label(text="{:d} bones, version {:d}".format(len(edited), template.version), icon='BLANK1')
                targets = apply_targets(context, template)
                if targets:
                    # The root's change needs no bone of its own.
                    missing = [name for name in edited if name != ROOT_BONE
                               and not any(name in armature.data.bones for armature in targets)]
                    box.label(text="On {:s}".format(_names(targets)), icon='OUTLINER_OB_ARMATURE')
                    if missing:
                        box.label(text="{:d} bones not in {:s}".format(
                            len(missing), "any" if len(targets) > 1 else "it"), icon='INFO')
                elif settings.scope == 'ALL':
                    box.label(text="No armature has its bones", icon='ERROR')
                if template.version > TEMPLATE_VERSION:
                    box.label(text="Newer than version {:d}".format(TEMPLATE_VERSION), icon='INFO')

        row = col.row(align=True)
        row.use_property_split = False
        row.prop(settings, "use_scale", toggle=True)
        row.prop(settings, "use_rotation", toggle=True)
        row.prop(settings, "use_translation", toggle=True)

        row = col.row(heading="Bust Size")
        row.prop(settings, "use_bust", text="")
        sub = row.row()
        sub.active = settings.use_bust
        sub.prop(settings, "bust_size", text="", slider=True)

        row = col.row()
        row.scale_y = 1.4
        row.operator(MAGIC_FIT_OT_cplus_apply.bl_idname, text="Apply", icon='MOD_ARMATURE')

        armatures = rigged(context)
        if not armatures:
            problem = armature_problem(context)
            if problem is not None and settings.scope == 'ONE' and target_armature(context) is not None:
                col.box().label(text=problem, icon='ERROR')
            return
        on = any(rig.is_on(armature) for armature in armatures)
        box = col.box().column(align=True)
        _applied(box, armatures, on)
        if on:
            if len(armatures) > 1:
                box.label(text="On {:s}".format(_names(armatures)), icon='BLANK1')
            left_out = [name for armature in armatures for name in rig.missing_meshes(armature)]
            if left_out:
                box.label(text="'{:s}' isn't included: Apply again".format(left_out[0]), icon='ERROR')
            changed = [armature.name for armature in armatures if rig.is_on(armature) and rig.is_stale(armature)]
            if changed:
                box.label(text="Bones of '{:s}' changed: Apply again".format(changed[0]), icon='ERROR')
            box.label(text="Turn it off before exporting", icon='INFO')
        key = hotkey(MAGIC_FIT_OT_cplus_toggle.bl_idname)
        if key:
            box.label(text="{:s} turns it on and off".format(key), icon='EVENT_C' if key.endswith(" C") else 'KEYINGSET')
        row = col.row(align=True)
        row.operator(MAGIC_FIT_OT_cplus_toggle.bl_idname, text="Turn Off" if on else "Turn On",
                     icon='HIDE_ON' if on else 'HIDE_OFF')
        row.operator(MAGIC_FIT_OT_cplus_remove.bl_idname, text="Remove", icon='X')


classes = (VIEW3D_PT_customize_plus_sidebar,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
