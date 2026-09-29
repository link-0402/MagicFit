# SPDX-License-Identifier: GPL-3.0-or-later
#
# Sidebar panel of Robust Weight Transfer by sentfromspacevr (https://github.com/sentfromspacevr),
# GPL-2.0-or-later, called Weight Transfer here and laid out like the other panels of the Magic Fit
# tab. Labels are shortened to fit the sidebar; the tooltips have the full names.

import bpy

from .. import tabs
from . import dependencies
from .operators import (
    MAGIC_FIT_OT_inpaint_weights,
    MAGIC_FIT_OT_reset_transfer_settings,
    MAGIC_FIT_OT_select_rejected,
    MAGIC_FIT_OT_transfer_weights,
    check_transfer_ready,
    transfer_targets,
)

# Whether the settings sections start closed (tests open them to draw everything).
SECTIONS_CLOSED = True


def _section(layout, idname, label, toggle=None, used=True):
    """A collapsible section, optionally with a checkbox (``toggle``: (data, property)) in its header,
    greyed out when not ``used``. Returns its body, or None while it's closed."""
    header, body = layout.panel(idname, default_closed=SECTIONS_CLOSED)
    header.active = used
    if toggle is None:
        header.label(text=label)
    else:
        header.use_property_split = False
        header.prop(*toggle, text=label)
    if body is not None:
        body.active = used and (toggle is None or getattr(*toggle))
    return body


def _prop_with_toggle(layout, heading, data, toggle, value):
    """A checkbox with the value it turns on, in one row."""
    row = layout.row(align=True, heading=heading)
    row.prop(data, toggle, text="")
    sub = row.row(align=True)
    sub.active = getattr(data, toggle)
    sub.prop(data, value, text="")


def _draw_settings(layout, settings):
    body = _section(layout, "magic_fit_transfer_settings", "Settings")
    if body is not None:
        body.operator(MAGIC_FIT_OT_reset_transfer_settings.bl_idname, icon='LOOP_BACK')
        body.prop(settings, "inpaint_mode")
        _prop_with_toggle(body, "Virtual Merge", settings, "virtual_merge", "virtual_merge_distance")
        _prop_with_toggle(body, "Seam Sync", settings, "seam_sync", "seam_distance")
        row = body.row()
        row.active = settings.seam_sync and settings.apply_to_selected
        row.prop(settings, "seam_sync_across_objects", text="Across Objects")
        col = body.column(heading="Weights")
        col.prop(settings, "normalize_weights_after_transfer", text="Normalize")
        row = body.row()
        row.enabled = not settings.enforce_four_bone_limit
        row.prop(settings, "group_selection")

    body = _section(layout, "magic_fit_transfer_mapping", "Vertex Mapping")
    if body is not None:
        body.prop(settings, "max_distance")
        _prop_with_toggle(body, "Partial Reweight", settings, "partial_reweight", "partial_reweight_falloff")
        # Crease Smoothing grades the matches itself.
        col = body.column(align=True)
        col.active = not settings.crease_smoothing
        col.prop(settings, "max_normal_angle_difference", text="Normal Angle")
        col.prop(settings, "soft_normal_limit", text="Soft Limit")
        col = body.column(heading="Normals")
        col.active = not settings.crease_smoothing
        col.prop(settings, "flip_vertex_normal", text="Allow Flipped")

    body = _section(layout, "magic_fit_transfer_crease", "Crease Smoothing", (settings, "crease_smoothing"))
    if body is not None:
        body.prop(settings, "crease_softness", text="Width")

    body = _section(layout, "magic_fit_transfer_smoothing", "Smoothing", (settings, "smoothing_enable"),
                    used=not settings.crease_smoothing)
    if body is not None:
        body.prop(settings, "smoothing_repeat", text="Repeat")
        body.prop(settings, "smoothing_factor", text="Factor")

    body = _section(layout, "magic_fit_transfer_limit", "Limit Groups per Vertex",
                    (settings, "enforce_four_bone_limit"))
    if body is not None:
        body.prop(settings, "num_limit_groups", text="Max Groups")

    body = _section(layout, "magic_fit_transfer_utilities", "Utilities")
    if body is not None:
        body.operator(MAGIC_FIT_OT_select_rejected.bl_idname)
        body.operator(MAGIC_FIT_OT_inpaint_weights.bl_idname)


class VIEW3D_PT_weight_transfer_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Transfer the body's weights onto whole meshes: copied where they match the body, filled in elsewhere"""
    bl_label = "Weight Transfer"
    # First on the Weights tab's General sub-tab, before the Weight Brushes.
    bl_order = 1
    weights_tabs = {'GENERAL'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        if dependencies.missing():
            box = layout.box().column(align=True)
            for i, line in enumerate(dependencies.explanation() + ["The brushes work without them"]):
                box.label(text=line, icon='ERROR' if i == 0 else 'BLANK1')
            return

        # The body is picked at the top of the tab.
        settings = context.scene.magic_fit_transfer
        body_settings = context.scene.magic_fit
        col = layout.column()

        obj = context.object
        if obj is not None and obj.type == 'MESH' and obj != body_settings.target:
            # Apply to All Selected leaves the masks out.
            masks = col.column()
            masks.active = not settings.apply_to_selected
            object_settings = obj.magic_fit_transfer
            for group, invert, text in (("vertex_group", "vertex_group_invert", "Transfer Mask"),
                                        ("inpaint_group", "inpaint_group_invert", "Inpaint Mask")):
                row = masks.row(align=True)
                row.prop_search(object_settings, group, obj, "vertex_groups", text=text)
                row.prop(object_settings, invert, text="", icon='ARROW_LEFTRIGHT')
            row = masks.row()
            row.active = bool(object_settings.inpaint_group)
            row.prop(object_settings, "inpaint_threshold", text="Threshold")
            # Rigid belongs to each mesh, and counts with Apply to All Selected too.
            col.prop(object_settings, "rigid")
        if settings.apply_to_selected:
            meshes = transfer_targets(context, settings, body_settings.target)
            rigid = sum(1 for mesh in meshes if mesh.magic_fit_transfer.rigid)
            if rigid:
                col.label(text="{:d} of {:d} selected meshes are rigid".format(rigid, len(meshes)), icon='INFO')
        row = col.row(align=True, heading="Use Deformed")
        row.prop(settings, "use_deformed_source", text="Body", toggle=True)
        row.prop(settings, "use_deformed_target", text="Meshes", toggle=True)

        problem = check_transfer_ready(context)
        if problem is not None:
            col.box().label(text=problem, icon='ERROR')
        col.separator()
        row = col.row(align=True)
        row.scale_y = 1.4
        row.prop(settings, "apply_to_selected", text="", icon='RESTRICT_SELECT_OFF')
        row.operator(MAGIC_FIT_OT_transfer_weights.bl_idname)

        _draw_settings(layout, settings)


classes = (
    VIEW3D_PT_weight_transfer_sidebar,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
