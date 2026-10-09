# SPDX-License-Identifier: GPL-3.0-or-later
#
# The panels of the Mesh Repair tab's Mesh Rebuild, Normals and UV sub-tabs: Resurface's sidebar, laid out
# like the other panels of the Magic Fit tab.

import bpy

from .. import tabs
from . import uvtools
from .operators import normal_map_file_material
from .core import canvas as canvas_math
from .core import indexmap as imap

# Whether the panels that start closed do (tests open them to draw everything).
PANELS_CLOSED = True


def objects_text(context, verb, fallback):
    n = sum(1 for o in context.selected_objects if o.type == "MESH")
    return f"{verb} {n} Object{'s' if n != 1 else ''}" if n else fallback


def draw_report(layout, s, prop):
    """The last result of a tool, with a button to hide it."""
    text = getattr(s, prop)
    if not text:
        return
    box = layout.box()
    col = box.column(align=True)
    lines = text.split("\n")
    row = col.row()
    row.label(text=lines[0])
    row.operator("magic_fit.clear_repair_report", text="", icon="X", emboss=False).prop = prop
    for line in lines[1:]:
        col.label(text=line)


def subtab_link(layout, context, subtab):
    """A small button that shows another sub-tab of the Mesh Repair tab."""
    layout.prop_enum(context.workspace, tabs.SUBTABS['REPAIR'][0], subtab, text="", icon="PREFERENCES")


def prepare(layout, context):
    layout.use_property_split = True
    layout.use_property_decorate = False
    return context.scene.magic_fit_repair


def big_button(layout, operator, **kwargs):
    """The panel's main button."""
    row = layout.row()
    row.scale_y = 1.4
    return row.operator(operator, **kwargs)


def section(layout, idname, label):
    """A collapsible section; its body, or None while it's closed."""
    header, body = layout.panel(idname, default_closed=PANELS_CLOSED)
    header.label(text=label)
    return body


class _RepairPanel(tabs.TabPanel):
    tab = 'REPAIR'


# ==========================================================================================
# Mesh Rebuild
# ==========================================================================================
class VIEW3D_PT_mesh_rebuild_sidebar(_RepairPanel, bpy.types.Panel):
    """Rebuild messy meshes with clean, even triangles, keeping borders, seams and thin strips"""
    bl_label = "Rebuild Mesh"
    bl_order = 1
    subtabs = {'REBUILD'}

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        col = layout.column()
        col.row(align=True).prop(s, "size_mode", expand=True)
        if s.size_mode == "RELATIVE":
            col.prop(s, "density")
        else:
            col.prop(s, "edge_length")
        col.prop(s, "adaptive")
        sub = col.column()
        sub.active = s.adaptive
        sub.prop(s, "tolerance")
        sub.prop(s, "min_edge_ratio")
        col.prop(s, "iterations")
        col.column(heading="Remove").prop(s, "remove_hidden", text="Hidden Layers")

        col = layout.column(heading="Afterwards")
        row = col.row(align=True)
        row.prop(s, "rebuild_normals")
        subtab_link(row, context, "NORMALS")
        row = col.row(align=True)
        row.prop(s, "rebuild_uvs")
        subtab_link(row, context, "UV")

        body = section(layout, "magic_fit_rebuild_preserve", "Preserve")
        if body is not None:
            col = body.column(heading="Keep")
            col.prop(s, "preserve_uv_seams")
            col.prop(s, "preserve_color_seams")
            col.prop(s, "preserve_sharp")
            sub = col.column()
            sub.active = s.preserve_sharp
            sub.prop(s, "sharp_angle")
            body.prop(s, "seam_threshold")
            body.column(heading="Ignore").prop(s, "clean_features", text="Tiny Features")
            body.prop(s, "corner_angle")
            hint = body.column(align=True)
            hint.label(text="Borders and non-manifold", icon="INFO")
            hint.label(text="junctions are always kept", icon="BLANK1")

        body = section(layout, "magic_fit_rebuild_cleanup", "Cleanup")
        if body is not None:
            sub = body.column()
            sub.active = s.remove_hidden
            sub.prop(s, "hidden_gap")
            body.prop(s, "merge_distance")
            body.prop(s, "fragment_faces", text="Fragments")

        body = section(layout, "magic_fit_rebuild_output", "Output")
        if body is not None:
            body.prop(s, "max_influences")
            body.prop(s, "keep_original")
            body.operator("magic_fit.discard_original", icon="TRASH")

        col = layout.column()
        col.separator()
        big_button(col, "magic_fit.rebuild_mesh", icon="MOD_REMESH",
                   text=objects_text(context, "Rebuild", "Rebuild Mesh"))
        row = col.row(align=True)
        row.operator("magic_fit.analyze_mesh", icon="VIEWZOOM")
        row.operator("magic_fit.restore_original", icon="LOOP_BACK")
        draw_report(col, s, "report_mesh")


# ==========================================================================================
# Normals
# ==========================================================================================
class VIEW3D_PT_smooth_normals_sidebar(_RepairPanel, bpy.types.Panel):
    """Give meshes clean smooth normals that ignore UV seams, part borders and faces wound the other way"""
    bl_label = "Smooth Normals"
    bl_order = 1
    subtabs = {'NORMALS'}

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        col = layout.column()
        col.prop(s, "normals_angle")
        col.prop(s, "normals_blur")
        col.prop(s, "fix_winding")
        col.separator()
        big_button(col, "magic_fit.smooth_normals", icon="NORMALS_VERTEX_FACE",
                   text=objects_text(context, "Smooth", "Smooth Normals"))
        note = "Rebuild Mesh applies these too" if s.rebuild_normals else "Rebuild Mesh can apply these too"
        row = col.row()
        row.label(text=note, icon="INFO")
        subtab_link(row, context, "REBUILD")
        draw_report(col, s, "report_normals")


class VIEW3D_PT_normal_map_seams_sidebar(_RepairPanel, bpy.types.Panel):
    """Find and fix normal maps whose green points down, which shows as lines along UV seams"""
    bl_label = "Normal Map Seams"
    bl_order = 2
    subtabs = {'NORMALS'}

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        col = layout.column()
        col.label(text="Lines along UV seams in Material Preview?")
        col.label(text="Finds normal maps with green pointing down", icon="INFO")
        col.separator()
        col.prop(s, "normalmap_file")
        if s.normalmap_file:
            material = normal_map_file_material(context)
            on = material.name if material is not None else "all faces"
            col.label(text=f"Read on: {on}", icon="MATERIAL" if material is not None else "MESH_DATA")
            col.label(text="Instead of the materials' normal maps", icon="BLANK1")
        col.separator()
        row = col.row(align=True)
        row.scale_y = 1.4
        row.operator("magic_fit.check_normal_maps", icon="VIEWZOOM", text="Check")
        row.operator("magic_fit.fix_normal_maps", icon="CHECKMARK", text="Fix")
        draw_report(col, s, "report_normalmap")


# ==========================================================================================
# UV
# ==========================================================================================
class VIEW3D_PT_rebuild_uvs_sidebar(_RepairPanel, bpy.types.Panel):
    """Replace UVs with a clean layout without overlaps; the old UVs are kept for Transfer Texture"""
    bl_label = "Rebuild UVs"
    bl_order = 1
    subtabs = {'UV'}

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        col = layout.column()
        col.prop(s, "uv_keep_old_seams")
        col.prop(s, "uv_cut_sharp")
        sub = col.column()
        sub.active = s.uv_cut_sharp
        sub.prop(s, "uv_sharp_angle")
        col.prop(s, "uv_keep_orientation")
        col.prop(s, "uv_margin")
        col.separator()
        big_button(col, "magic_fit.rebuild_uvs", icon="UV", text="Rebuild UVs")
        note = "Rebuild Mesh applies these too" if s.rebuild_uvs else "Rebuild Mesh can apply these too"
        row = col.row()
        row.label(text=note, icon="INFO")
        subtab_link(row, context, "REBUILD")
        draw_report(col, s, "report_uvs")


class VIEW3D_PT_transfer_texture_sidebar(_RepairPanel, bpy.types.Panel):
    """Move a texture painted for the old UVs onto the new UV layout"""
    bl_label = "Transfer Texture"
    bl_order = 2
    bl_options = {"DEFAULT_CLOSED"}
    subtabs = {'UV'}

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        layout.template_ID(s, "tex_image", open="image.open")
        col = layout.column()
        col.prop(s, "tex_mode")
        if s.tex_mode == "NORMAL":
            col.prop(s, "tex_flip_green")
        col.prop(s, "tex_padding")
        col.label(text="From the 'Old UVs' layer to the current UVs", icon="INFO")
        col.separator()
        big_button(col, "magic_fit.transfer_texture", icon="IMAGE_DATA")
        draw_report(col, s, "report_texture")


class VIEW3D_PT_uv_canvas_sidebar(_RepairPanel, bpy.types.Panel):
    """Move the UVs the way a texture's canvas was cropped or extended, so faces keep their pixels"""
    bl_label = "Resize Canvas"
    bl_order = 3
    bl_options = {"DEFAULT_CLOSED"}
    subtabs = {'UV'}

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        col = layout.column(align=True)
        col.prop(s, "canvas_old_w", text="Old Size X")
        col.prop(s, "canvas_old_h", text="Y")
        col = layout.column(align=True)
        col.prop(s, "canvas_new_w", text="New Size X")
        col.prop(s, "canvas_new_h", text="Y")

        # 3 x 3 anchor grid like an image editor's canvas size dialog
        split = layout.split(factor=0.4)
        lab = split.row()
        lab.alignment = "RIGHT"
        lab.label(text="Anchor")
        grid = split.column(align=True)
        keys = list(canvas_math.ANCHORS)
        for r in range(3):
            row = grid.row(align=True)
            for key in keys[r * 3:r * 3 + 3]:
                on = s.canvas_anchor == key
                row.prop_enum(s, "canvas_anchor", key, text="", icon="LAYER_ACTIVE" if on else "LAYER_USED")

        col = layout.column(align=True)
        col.prop(s, "canvas_offset_x", text="Offset X")
        col.prop(s, "canvas_offset_y", text="Y")
        row = layout.row()
        row.alignment = "RIGHT"
        row.operator("magic_fit.canvas_from_selection", icon="RESTRICT_SELECT_OFF")
        sub = layout.column()
        sub.active = context.mode == "EDIT_MESH"
        sub.prop(s, "canvas_selected_only")

        old, new, off = uvtools.canvas_setup(s)
        (su, sv), _ = canvas_math.canvas_transform(old, new, off)
        col = layout.column()
        col.label(text=f"U x{su:.4g}, V x{sv:.4g}, crop at x {off[0]}, y {off[1]}", icon="INFO")
        col.separator()
        big_button(col, "magic_fit.resize_uv_canvas", icon="FULLSCREEN_EXIT")
        draw_report(col, s, "report_canvas")


class VIEW3D_PT_index_map_sidebar(_RepairPanel, bpy.types.Panel):
    """Paint an FFXIV index (_id) texture for Dawntrail colorsets from the UV islands or a texture's colours"""
    bl_label = "Index Map Generator"
    bl_order = 4
    bl_options = {"DEFAULT_CLOSED"}
    subtabs = {'UV'}

    def draw(self, context):
        layout = self.layout
        s = prepare(layout, context)
        row = layout.row(align=True)
        row.prop(s, "idx_mode", expand=True)
        if s.idx_mode == "AUTO":
            self.draw_auto(layout.column(), s)
        else:
            self.draw_manual(context, layout, s)

        layout.separator()
        col = layout.column(align=True)
        col.prop(s, "idx_width", text="Size X")
        col.prop(s, "idx_height", text="Y")
        col = layout.column()
        col.prop(s, "idx_padding")
        col.prop(s, "idx_output")
        col.separator()
        big_button(col, "magic_fit.index_map", icon="IMAGE_RGB")
        draw_report(col, s, "report_index")

    @staticmethod
    def draw_auto(col, s):
        col.row(align=True).prop(s, "idx_source", expand=True)
        if s.idx_source == "TEXTURE":
            col.template_ID(s, "idx_texture", open="image.open")
            col.prop(s, "idx_color_tol")
            col.prop(s, "idx_detail")
            col.prop(s, "idx_whole_islands")
        else:
            col.prop(s, "idx_group_similar")
            sub = col.column()
            sub.active = s.idx_group_similar
            sub.prop(s, "idx_similarity")
        col.prop(s, "idx_max_rows")

    def draw_manual(self, context, layout, s):
        box = layout.box()
        col = box.column()
        col.use_property_split = False
        row = col.row(align=True)
        row.prop(s, "idx_row", text="")
        row.operator("magic_fit.index_assign", text="Assign")
        row.operator("magic_fit.index_assign", text="Remove").value = 0
        if context.mode != "EDIT_MESH":
            col.label(text="Select faces in Edit Mode to assign rows", icon="INFO")

        # rows in use, with buttons to select or clear them
        objs = uvtools.mesh_targets(context)
        counts = uvtools.row_counts(objs) if objs else {}
        lst = col.column(align=True)
        for v in sorted(counts, key=lambda v: (v == 0, v)):
            row = lst.row(align=True)
            if v:
                r, g, b = imap.row_rgb(v - 1)
                row.label(text=f"{uvtools.value_name(v)}   #{r:02X}{g:02X}{b:02X}")
            else:
                row.label(text="No row (1B)")
            row.label(text=f"{counts[v]:,} faces")
            row.operator("magic_fit.index_select", text="", icon="RESTRICT_SELECT_OFF").value = v
            if v:
                row.operator("magic_fit.index_clear", text="", icon="X").value = v
            else:
                row.label(text="", icon="BLANK1")

        sub = box.column()
        sub.operator("magic_fit.index_autofill", icon="AUTO")
        self.draw_auto(sub, s)

        legacy = [vg.name for o in objs for vg in uvtools.legacy_groups(o)]
        if legacy:
            sub = box.column()
            sub.label(text=f"{len(legacy)} old UV_Group vertex groups", icon="ERROR")
            sub.operator("magic_fit.index_convert_groups", icon="GROUP_VERTEX")


classes = (
    VIEW3D_PT_mesh_rebuild_sidebar,
    VIEW3D_PT_smooth_normals_sidebar,
    VIEW3D_PT_normal_map_seams_sidebar,
    VIEW3D_PT_rebuild_uvs_sidebar,
    VIEW3D_PT_transfer_texture_sidebar,
    VIEW3D_PT_uv_canvas_sidebar,
    VIEW3D_PT_index_map_sidebar,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
