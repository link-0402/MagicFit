# SPDX-License-Identifier: GPL-3.0-or-later

import math
import os

import blf
import bpy
import gpu
from gpu.types import GPUBatch, GPUVertBuf, GPUVertFormat

from . import clipping, facing, hairing, tabs
from .operators import (
    FACE_REPAIRS,
    MAGIC_FIT_OT_body_fit,
    MAGIC_FIT_OT_detect_face,
    MAGIC_FIT_OT_face_pose,
    MAGIC_FIT_OT_face_repair,
    MAGIC_FIT_OT_face_weights,
    MAGIC_FIT_OT_fit_move,
    MAGIC_FIT_OT_hair_weights,
    MAGIC_FIT_OT_heel_weights,
    MAGIC_FIT_OT_line_up,
    MAGIC_FIT_OT_paint,
    MAGIC_FIT_OT_pick_face,
    MAGIC_FIT_OT_pick_hair_skeleton,
    MAGIC_FIT_OT_resize,
    MAGIC_FIT_OT_skirt_weights,
    MAGIC_FIT_OT_texture_relax,
    MAGIC_FIT_OT_use_tool,
    check_face_pose_ready,
    check_face_tools_ready,
    check_fit_ready,
    check_hair_weights_ready,
    check_lineup_ready,
    check_ready,
    check_relax_ready,
    check_resize_ready,
    check_whole_mesh_ready,
    fit_notes,
    found_face,
    hair_skeleton_text,
    lineup_summary,
    parse_matches,
    resize_key_name,
)
from .properties import COMING_MODES, WEIGHTS_TABS

WEIGHT_TOOL_IDNAME = "magic_fit.tool"
FIT_TOOL_IDNAME = "magic_fit.body_fit_tool"
RELAX_TOOL_IDNAME = "magic_fit.texture_relax_tool"
MOVE_TOOL_IDNAME = "magic_fit.fit_move_tool"
# Toolbar icons are geometry files; an absolute path without the ".dat" extension is accepted.
ICONS_DIR = os.path.join(os.path.dirname(__file__), "icons")


def _brush_keymap(settings_path):
    """F / Shift F radial controls and [ ] resizing, for the brush settings at ``settings_path``."""
    return (
        ("wm.radial_control", {"type": 'F', "value": 'PRESS'},
         {"properties": [("data_path_primary", settings_path + ".radius")]}),
        ("wm.radial_control", {"type": 'F', "value": 'PRESS', "shift": True},
         {"properties": [("data_path_primary", settings_path + ".strength")]}),
        ("wm.context_scale_int", {"type": 'LEFT_BRACKET', "value": 'PRESS', "repeat": True},
         {"properties": [("data_path", settings_path + ".radius"), ("value", 0.9)]}),
        ("wm.context_scale_int", {"type": 'RIGHT_BRACKET', "value": 'PRESS', "repeat": True},
         {"properties": [("data_path", settings_path + ".radius"), ("value", 1.0 / 0.9)]}),
    )


# -----------------------------------------------------------------------------
# Shared settings UI

def _prop_with_pressure(layout, settings, prop, pressure_prop, **kwargs):
    row = layout.row(align=True)
    row.prop(settings, prop, **kwargs)
    row.prop(settings, pressure_prop, text="", icon='STYLUS_PRESSURE')


def _draw_choice(layout, settings, prop):
    """An expanded enum across the full width, so its options stay readable in a narrow sidebar."""
    row = layout.row(align=True)
    row.use_property_split = False
    row.prop(settings, prop, expand=True)


def _draw_brush(layout, settings):
    _prop_with_pressure(layout, settings, "radius", "use_pressure_radius")
    _prop_with_pressure(layout, settings, "strength", "use_pressure_strength", slider=True)


def _draw_auto_smooth(layout, settings, header=False):
    row = layout.row(align=True, heading="" if header else "Auto Smooth")
    row.prop(settings, "use_auto_smooth", text="")
    sub = row.row(align=True)
    sub.active = settings.use_auto_smooth
    sub.prop(settings, "auto_smooth", text="Auto Smooth" if header else "", slider=True)


def _draw_sections(layout, sections):
    """Collapsible sub-panels: ``sections`` holds (idname, label, draw(layout)) triples."""
    for idname, label, draw in sections:
        header, body = layout.panel(idname, default_closed=True)
        header.label(text=label)
        if body is not None:
            draw(body)


def _draw_popover(layout, sections):
    layout.use_property_split = True
    layout.use_property_decorate = False
    for i, (_idname, label, draw) in enumerate(sections):
        if i:
            layout.separator()
        layout.label(text=label)
        draw(layout.column())


# -----------------------------------------------------------------------------
# Weight Brushes settings

def _weight_sections(context, settings, where):
    """The collapsible sections of the brush settings, for ``where`` they're shown: 'TOOL' (Properties >
    Tool), 'POPOVER' (the tool header's Options) or 'SIDEBAR' (a brush panel of the sidebar's Weights
    tab, below the panel with its mode's settings)."""
    def falloff(layout):
        layout.prop(settings, "falloff", text="Curve")
        layout.row().prop(settings, "falloff_shape", expand=True)
        layout.prop(settings, "use_front_faces_only")

    def stroke(layout):
        layout.prop(settings, "spacing")
        if settings.mode in {'COPY', 'SMOOTH'}:
            layout.prop(settings, "use_accumulate")

    def sampling(layout):
        layout.prop(settings, "sample_mode")
        row = layout.row(align=True, heading="Max Distance")
        row.prop(settings, "use_max_distance", text="")
        sub = row.row(align=True)
        sub.active = settings.use_max_distance
        sub.prop(settings, "max_distance", text="")

        col = layout.column()
        col.active = settings.group_mode == 'ALL'
        col.prop(settings, "use_bones_only")
        col.prop(settings, "use_fade_unmatched")

        tool_settings = context.tool_settings
        layout.prop(tool_settings, "use_auto_normalize", text="Auto Normalize")
        col = layout.column()
        col.active = settings.group_mode == 'ACTIVE'
        col.prop(tool_settings.weight_paint, "use_group_restrict")

    def skirt(layout):
        if where == 'POPOVER':
            # Properties > Tool shows these right below the body.
            _draw_skirt_main(layout, settings)
        _draw_skirt_more(layout, settings)

    def heels(layout):
        _draw_heels_main(layout, settings)

    def hair(layout):
        if where == 'POPOVER':
            # Properties > Tool shows these right below the mode.
            _draw_hair_main(layout, settings)
        _draw_hair_more(layout, settings)

    def face(layout):
        _draw_face_main(layout, settings)

    def smooth(layout):
        _draw_smooth_main(layout, settings)

    sections = [
        ("magic_fit_falloff", "Falloff", falloff),
        ("magic_fit_stroke", "Stroke", stroke),
    ]
    if settings.mode == 'COPY':
        sections.append(("magic_fit_sampling", "Sampling", sampling))
    elif where == 'SIDEBAR':
        pass  # the mode's settings are in the panel above
    elif settings.mode == 'SKIRT':
        sections.append(("magic_fit_skirt", "Skirt", skirt))
    elif settings.mode == 'HAIR':
        sections.append(("magic_fit_hair", "Hair", hair))
    elif where == 'POPOVER':
        # Properties > Tool shows these right below the mode.
        if settings.mode == 'HEELS':
            sections.append(("magic_fit_heels", "Heels", heels))
        elif settings.mode == 'FACE':
            sections.append(("magic_fit_face", "Face", face))
        elif settings.mode == 'SMOOTH':
            sections.append(("magic_fit_smooth", "Smooth", smooth))
    return sections


def _draw_straighten_hint(layout):
    """Straighten works on the displayed pose: say so where its settings are."""
    box = layout.box().column(align=True)
    box.label(text="Click where the mesh sags", icon='INFO')
    box.label(text="in the pose you see", icon='BLANK1')


def _draw_skirt_main(layout, settings):
    """The Skirt settings that decide how much hangs from the skirt bones."""
    col = layout.column(align=True)
    col.prop(settings, "skirt_blend_start")
    col.prop(settings, "skirt_blend_length")
    col = layout.column(align=True)
    col.prop(settings, "skirt_skin_weight", slider=True)
    col.prop(settings, "skirt_skin_distance")


def _draw_skirt_more(layout, settings):
    """The other Skirt settings: body weight everywhere, how the skirt weight is shared out, the group limit."""
    layout.prop(settings, "skirt_body_weight", slider=True)
    layout.prop(settings, "skirt_spread")
    layout.prop(settings, "skirt_joint_blend")
    layout.prop(settings, "skirt_max_groups")


def _draw_heels_main(layout, settings):
    """The Heels settings: where the calf hands over to the ankle, and how much the toes bend."""
    col = layout.column(align=True)
    col.prop(settings, "heels_ankle_height")
    col.prop(settings, "heels_ankle_blend")
    layout.prop(settings, "heels_toe_bend", slider=True)


def _draw_smooth_main(layout, settings):
    """The Smooth settings: how far the weights around a vertex count, which loose parts count as
    connected, and the group limit."""
    col = layout.column(align=True)
    col.prop(settings, "smooth_reach")
    col.prop(settings, "smooth_merge_range")
    layout.prop(settings, "smooth_max_groups")


def _draw_hair_main(layout, settings):
    """The Hair settings that decide how much hangs from the hair bones."""
    col = layout.column(align=True)
    col.prop(settings, "hair_hang_start")
    col.prop(settings, "hair_hang_length")
    layout.prop(settings, "hair_neck_weight", slider=True)


def _draw_hair_more(layout, settings):
    """The other Hair settings: how chains share long hair, the group limit."""
    layout.prop(settings, "hair_chain_stretch", slider=True)
    layout.prop(settings, "hair_joint_blend")
    if settings.hair_race in hairing.EAR_RACES:
        layout.prop(settings, "hair_ears")
    layout.prop(settings, "hair_max_groups")


def _draw_game_face(layout, settings):
    """Which of the game's faces the face replaces: the one Detect found (and a button to look again), or the
    one picked."""
    _draw_choice(layout, settings, "face_source")
    if settings.face_source == 'PICK':
        layout.prop(settings, "face_race")
        row = layout.row(align=True)
        row.operator(MAGIC_FIT_OT_pick_face.bl_idname, text=facing.face_label(settings.face_number), icon='VIEWZOOM')
        return
    found = found_face(settings)
    row = layout.row(align=True)
    row.use_property_split = False
    row.label(text=facing.face_title(*found) if found is not None else "Not detected yet", icon='USER')
    row.operator(MAGIC_FIT_OT_detect_face.bl_idname, text="", icon='VIEWZOOM')


def _draw_face_problem(layout, problem):
    """Why the face tools can't be used; when it's that the game isn't found, a button to set its folder."""
    box = layout.box()
    if problem != facing.face_data_problem():
        box.label(text=problem, icon='ERROR')
        return
    box.label(text="FFXIV wasn't found: Face reads the game's faces from it", icon='ERROR')
    box.operator("preferences.addon_show", text="Set the Game Folder", icon='PREFERENCES').module = __package__


def _draw_face_experimental(layout):
    """The Face tools are still experimental: say so at the top of the Face sub-tab."""
    box = layout.box().column(align=True)
    box.label(text="Experimental", icon='EXPERIMENTAL')
    box.label(text="Check the results in the game", icon='BLANK1')


def _draw_face_main(layout, settings):
    """The Face settings the brush uses too: which game face, closing the eyes, the group limit."""
    _draw_game_face(layout, settings)
    layout.prop(settings, "face_close_eyes")
    layout.prop(settings, "face_max_groups")


# Per mode with a whole-mesh button: (its operator, its Apply to All Selected setting, its icon).
_WHOLE_MESH_BUTTONS = {
    'SKIRT': (MAGIC_FIT_OT_skirt_weights, "skirt_apply_to_selected", 'MOD_CLOTH'),
    'HEELS': (MAGIC_FIT_OT_heel_weights, "heels_apply_to_selected", 'MOD_DISPLACE'),
}


def _draw_mode_main(layout, settings):
    """The settings of the Skirt or Heels mode that belong right below the mode."""
    if settings.mode == 'SKIRT':
        _draw_skirt_main(layout, settings)
    elif settings.mode == 'HEELS':
        _draw_heels_main(layout, settings)
    elif settings.mode == 'HAIR':
        _draw_hair_main(layout, settings)
    elif settings.mode == 'FACE':
        _draw_face_main(layout, settings)
    elif settings.mode == 'SMOOTH':
        _draw_smooth_main(layout, settings)


def _draw_weight_choices(layout, settings):
    """Mode, target and groups: what the brush does."""
    _draw_choice(layout, settings, "mode")
    # Heels, Hair, Face and Smooth need no body
    if settings.mode not in {'HEELS', 'HAIR', 'FACE', 'SMOOTH', *COMING_MODES}:
        row = layout.row()
        row.alert = settings.target is None and not (settings.mode == 'SKIRT' and settings.skirt_skin_weight <= 0.0)
        row.prop(settings, "target", text="Target" if settings.mode == 'COPY' else "Body")
    _draw_choice(layout, settings, "group_mode")


def _draw_weight_settings(context, layout):
    """The brush's settings, in the tool header and under Properties > Tool. The Skirt Weights and Heel
    Weights buttons aren't brush settings: they're on the sidebar's Weights tab."""
    settings = context.scene.magic_fit
    header = context.region.type in {'TOOL_HEADER', 'HEADER'}
    col = layout if header else layout.column()
    _draw_weight_choices(col, settings)
    if header:
        _draw_brush(layout, settings)
        if settings.mode == 'COPY':
            _draw_auto_smooth(layout, settings, header=True)
        layout.popover(VIEW3D_PT_magic_fit_options.__name__, text="Options")
        return
    if settings.mode == 'STRAIGHTEN':
        _draw_straighten_hint(col)
    _draw_mode_main(col, settings)
    _draw_weight_brush(context, layout, col, settings, 'TOOL')


def _draw_weight_brush(context, layout, col, settings, where):
    """Whether the brush can paint, its size and strength, and its sections (see `_weight_sections`)."""
    problem = check_ready(context, settings)
    # On the sidebar, the Skirt Weights or Heel Weights panel above often says the same already.
    if problem is not None and not (where == 'SIDEBAR' and settings.mode in _WHOLE_MESH_BUTTONS and
                                    problem == check_whole_mesh_ready(context, settings.mode)):
        col.box().label(text=problem, icon='ERROR')
    col.separator()
    _draw_brush(col, settings)
    if settings.mode == 'COPY':
        _draw_auto_smooth(col, settings)
    _draw_sections(layout, _weight_sections(context, settings, where))


class VIEW3D_PT_magic_fit_options(bpy.types.Panel):
    """Settings of the Weight Brushes"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_label = "Weight Brushes Options"
    bl_ui_units_x = 15

    def draw(self, context):
        _draw_popover(self.layout, _weight_sections(context, context.scene.magic_fit, 'POPOVER'))


# -----------------------------------------------------------------------------
# Body Fit settings

def _fit_sections(settings):
    def falloff(layout):
        layout.prop(settings, "falloff", text="Curve")

    def stroke(layout):
        layout.prop(settings, "spacing")

    def layers(layout):
        layout.prop(settings, "layer_radius")
        layout.prop(settings, "seam_distance")
        row = layout.row(align=True, heading="Max Distance")
        row.active = settings.mode != 'PUSH'
        row.prop(settings, "use_max_distance", text="")
        sub = row.row(align=True)
        sub.active = settings.use_max_distance
        sub.prop(settings, "max_distance", text="")
        layout.prop(settings, "use_selected_only")
        row = layout.row(align=True, heading="Fade at Hidden")
        row.prop(settings, "use_fade", text="")
        sub = row.row(align=True)
        sub.active = settings.use_fade
        sub.prop(settings, "fade_distance", text="")

    return (
        ("body_fit_brush_falloff", "Falloff", falloff),
        ("body_fit_brush_stroke", "Stroke", stroke),
        ("body_fit_brush_layers", "Layers", layers),
    )


def _draw_fit_settings(context, layout, body=True):
    """The brush's settings; ``body``: with the body picker, which the sidebar has at the top instead."""
    settings = context.scene.body_fit_brush
    # The body is the one the tools share.
    body_settings = context.scene.magic_fit
    header = context.region.type in {'TOOL_HEADER', 'HEADER'}
    col = layout if header else layout.column()
    if body:
        row = col.row()
        row.alert = body_settings.target is None
        row.prop(body_settings, "target", text="Body")
    _draw_choice(col, settings, "mode")
    col.prop(settings, "offset")
    if header:
        _draw_brush(layout, settings)
        _draw_auto_smooth(layout, settings, header=True)
        layout.popover(VIEW3D_PT_body_fit_options.__name__, text="Options")
        return
    problem = check_fit_ready(context)
    if problem is not None:
        col.box().label(text=problem, icon='ERROR')
    else:
        for lines, icon in fit_notes(context):
            box = col.box().column(align=True)
            for i, line in enumerate(lines):
                box.label(text=line, icon=icon if i == 0 else 'BLANK1')
    col.separator()
    _draw_brush(col, settings)
    _draw_auto_smooth(col, settings)
    _draw_sections(layout, _fit_sections(settings))


class VIEW3D_PT_body_fit_options(bpy.types.Panel):
    """Settings of the Body Fit brush"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_label = "Body Fit Options"
    bl_ui_units_x = 15

    def draw(self, context):
        _draw_popover(self.layout, _fit_sections(context.scene.body_fit_brush))


# -----------------------------------------------------------------------------
# Fit Move settings

def _draw_move_settings(context, layout):
    settings = context.scene.fit_move
    header = context.region.type in {'TOOL_HEADER', 'HEADER'}
    col = layout if header else layout.column()
    row = col.row(align=True, heading="" if header else "Fade at Hidden")
    row.prop(settings, "use_fade", text="Fade at Hidden" if header else "")
    sub = row.row(align=True)
    sub.active = settings.use_fade
    sub.prop(settings, "fade_distance", text="")
    col.prop(settings, "seam_distance")
    if header:
        return
    tool = context.tool_settings
    col.separator()
    row = col.row(heading="Proportional")
    row.prop(tool, "use_proportional_edit", text="")
    sub = row.row(align=True)
    sub.active = tool.use_proportional_edit
    sub.prop(tool, "proportional_edit_falloff", text="")
    sub = col.row()
    sub.active = tool.use_proportional_edit
    sub.prop(tool, "proportional_size", text="Size")


# -----------------------------------------------------------------------------
# Texture Relax settings

def _relax_sections(settings):
    def falloff(layout):
        layout.prop(settings, "falloff", text="Curve")

    def stroke(layout):
        layout.prop(settings, "spacing")

    def edges(layout):
        layout.prop(settings, "seam_distance")
        layout.prop(settings, "use_selected_only")

    return (
        ("texture_relax_brush_falloff", "Falloff", falloff),
        ("texture_relax_brush_stroke", "Stroke", stroke),
        ("texture_relax_brush_edges", "Edges", edges),
    )


def _draw_relax_settings(context, layout):
    settings = context.scene.texture_relax_brush
    header = context.region.type in {'TOOL_HEADER', 'HEADER'}
    if header:
        layout.prop(settings, "boundary")
        _draw_brush(layout, settings)
        layout.popover(VIEW3D_PT_texture_relax_options.__name__, text="Options")
        return
    col = layout.column()
    _draw_choice(col, settings, "boundary")
    problem = check_relax_ready(context)
    if problem is not None:
        col.box().label(text=problem, icon='ERROR')
    col.separator()
    _draw_brush(col, settings)
    _draw_sections(layout, _relax_sections(settings))


class VIEW3D_PT_texture_relax_options(bpy.types.Panel):
    """Settings of the Texture Relax brush"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_label = "Texture Relax Options"
    bl_ui_units_x = 15

    def draw(self, context):
        _draw_popover(self.layout, _relax_sections(context.scene.texture_relax_brush))


# -----------------------------------------------------------------------------
# Sidebar (N panel): the Magic Fit tab, there whatever the mode. Tabs along its top sort its panels by
# tool. A brush's panel has a button that starts painting, and all the brush's settings while it's active.

# Blender's widget unit, and about how much of the sidebar's width the panels' content doesn't get (the
# category tabs and the panel margins), in pixels at a UI scale of 1.
WIDGET_UNIT = 20.0
SIDEBAR_MARGIN = 54.0


def _tool_icon(name):
    """Icon value of one of the toolbar icons, for use in regular layouts."""
    from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
    try:
        return ToolSelectPanelHelper._icon_value_from_icon_handle(os.path.join(ICONS_DIR, name))
    except Exception:
        return 0


def _draw_use_button(layout, tool, text, icon):
    layout.separator()
    row = layout.row()
    row.scale_y = 1.4
    row.operator(MAGIC_FIT_OT_use_tool.bl_idname, text=text, icon_value=_tool_icon(icon)).tool = tool


def _panel_layout(layout):
    layout.use_property_split = True
    layout.use_property_decorate = False
    return layout


def _label_widths(context, labels):
    """About the widths buttons with these labels need, in pixels, with or without an icon."""
    preferences = context.preferences
    scale = preferences.system.ui_scale
    blf.size(0, preferences.ui_styles[0].widget.points * scale)
    # Blender's estimate for a label: its text and 1.5 units. That leaves room for an icon too: with one,
    # labels only get clipped below about 1.3 units (measured in Blender 5.2).
    padding = 1.5 * WIDGET_UNIT * scale
    return [blf.dimensions(0, label)[0] + padding for label in labels]


def _tabs_per_row(context, labels):
    """How many tabs with these labels go in a row: all of them where they fit side by side, else as many
    as spreads them evenly over the fewest rows they fit in."""
    widths = _label_widths(context, labels)
    room = context.region.width - SIDEBAR_MARGIN * context.preferences.system.ui_scale
    count = len(labels)
    for rows in range(1, count):
        per_row = math.ceil(count / rows)
        if all(sum(widths[first:first + per_row]) <= room for first in range(0, count, per_row)):
            return per_row
    return 1


def _draw_tabs(context, layout, data, prop, items, icons=None):
    """The items of enum ``prop`` as a row of tabs, or as several rows where their labels wouldn't fit.
    ``icons`` (item -> icon) go left of the labels."""
    per_row = _tabs_per_row(context, [item[1] for item in items])
    col = layout.column(align=True)
    for first in range(0, len(items), per_row):
        row = col.row(align=True)
        for item in items[first:first + per_row]:
            if icons is None:
                row.prop_enum(data, prop, item[0])
            else:
                row.prop_enum(data, prop, item[0], icon=icons[item[0]])


def _draw_line(layout):
    """A separator line, in Blender versions that draw them, else a gap."""
    try:
        layout.separator(type='LINE')
    except TypeError:
        layout.separator()


# Luci_xiv's pages: (label, icon, URL).
LINKS = (
    ("XIV Mod Archive", 'PACKAGE', "https://www.xivmodarchive.com/user/124593"),
    ("GitHub", 'SCRIPT', "https://github.com/link-0402/MagicFit"),
    ("Bluesky", 'COMMUNITY', "https://bsky.app/profile/xiv-luci.bsky.social"),
    ("Ko-fi", 'FUND', "https://ko-fi.com/luci_xiv"),
)


class VIEW3D_MT_magic_fit_links(bpy.types.Menu):
    """Magic Fit on GitHub, and Luci_xiv's mods, Bluesky and Ko-fi"""
    bl_label = "Links"

    def draw(self, context):
        for label, icon, url in LINKS:
            self.layout.operator("wm.url_open", text=label, icon=icon).url = url


class VIEW3D_PT_magic_fit_tabs(bpy.types.Panel):
    """The tabs, and the body the tools work with"""
    # Keep the header: Blender sorts panels without one ahead of all others, which would pull the whole
    # Magic Fit tab of the sidebar above Item, Tool and View.
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = tabs.CATEGORY
    bl_label = "Magic Fit"
    bl_order = 0

    def draw_header_preset(self, context):
        # At the right end of the header, where Blender puts the presets.
        layout = self.layout
        layout.emboss = 'NONE'
        layout.menu(VIEW3D_MT_magic_fit_links.__name__, text="", icon='URL')

    def draw(self, context):
        workspace = context.workspace
        if workspace is None:
            return
        settings = context.scene.magic_fit
        layout = self.layout
        layout.use_property_decorate = False
        col = layout.column()
        sub = col.column()
        sub.scale_y = 1.3
        _draw_tabs(context, sub, workspace, "magic_fit_tab", tabs.TABS, tabs.ICONS)
        tab = workspace.magic_fit_tab
        if tab == 'WEIGHTS':
            _draw_line(col)
            _draw_tabs(context, col, settings, "weights_tab", WEIGHTS_TABS)
            # Heels need no body, and Skirt only for Skin Weight. General always does: Smooth doesn't,
            # but Weight Transfer, above it, does.
            general = settings.weights_tab == 'GENERAL'
            shown = general or settings.weights_tab == 'SKIRT'
            needed = general or (settings.weights_tab == 'SKIRT' and settings.skirt_skin_weight > 0.0)
        else:
            shown = needed = tab in {'FIT', 'LINE_UP'}
        if shown:
            col.separator()
            row = col.row()
            row.use_property_split = True
            row.alert = needed and settings.target is None
            row.prop(settings, "target", text="Body")


# -----------------------------------------------------------------------------
# Weights tab: on each sub-tab, what gives whole meshes their weights first, then the brush.

class _WholeMeshPanel(tabs.TabPanel):
    """Skirt Weights or Heel Weights for whole meshes, with the settings the brush shares."""
    kind = ''

    def draw(self, context):
        layout = _panel_layout(self.layout)
        settings = context.scene.magic_fit
        col = layout.column()
        _draw_choice(col, settings, "group_mode")
        self.draw_mode(col, settings)
        problem = check_whole_mesh_ready(context, self.kind)
        if problem is not None:
            col.box().label(text=problem, icon='ERROR')
        col.separator()
        operator, prop, icon = _WHOLE_MESH_BUTTONS[self.kind]
        row = col.row(align=True)
        row.scale_y = 1.4
        row.prop(settings, prop, text="", icon='RESTRICT_SELECT_OFF')
        row.operator(operator.bl_idname, icon=icon)

    @staticmethod
    def draw_mode(layout, settings):
        """The mode's settings, which the brush uses too."""


class _WeightBrushPanel(tabs.TabPanel):
    """The Weight Brushes on a sub-tab of the Weights tab: a button that starts painting, and the
    brush's settings while it's active."""
    button = "Paint Weights"

    def draw(self, context):
        layout = _panel_layout(self.layout)
        settings = context.scene.magic_fit
        col = layout.column()
        self.draw_mode(col, settings)
        if _tool_is_active(context, 'PAINT_WEIGHT', WEIGHT_TOOL_IDNAME):
            _draw_weight_brush(context, layout, col, settings, 'SIDEBAR')
        else:
            _draw_use_button(col, 'WEIGHT', self.button, "weight_brushes")

    @staticmethod
    def draw_mode(layout, settings):
        """The settings only the brush has, above the rest."""


class VIEW3D_PT_magic_fit_sidebar(_WeightBrushPanel, bpy.types.Panel):
    """Paint to copy the body's weights, straighten creases in the pose, or smooth weights"""
    bl_label = "Weight Brushes"
    bl_order = 2
    weights_tabs = {'GENERAL'}

    @staticmethod
    def draw_mode(layout, settings):
        row = layout.row(align=True)
        row.use_property_split = False
        for mode in ('COPY', 'STRAIGHTEN', 'SMOOTH'):
            row.prop_enum(settings, "mode", mode)
        _draw_choice(layout, settings, "group_mode")
        if settings.mode == 'STRAIGHTEN':
            _draw_straighten_hint(layout)
        elif settings.mode == 'SMOOTH':
            _draw_smooth_main(layout, settings)


class VIEW3D_PT_skirt_weights_sidebar(_WholeMeshPanel, bpy.types.Panel):
    """Give whole skirts and long dresses the weights of the skirt bones"""
    bl_label = "Skirt Weights"
    bl_order = 1
    weights_tabs = {'SKIRT'}
    kind = 'SKIRT'

    @staticmethod
    def draw_mode(layout, settings):
        _draw_skirt_main(layout, settings)
        _draw_skirt_more(layout, settings)


class VIEW3D_PT_skirt_brush_sidebar(_WeightBrushPanel, bpy.types.Panel):
    """Paint skirt weights with the brush, with the Skirt Weights settings"""
    bl_label = "Skirt Brush"
    bl_order = 2
    weights_tabs = {'SKIRT'}
    button = "Paint Skirt Weights"


class VIEW3D_PT_heel_weights_sidebar(_WholeMeshPanel, bpy.types.Panel):
    """Make whole heels and boots rigid below the ankle"""
    bl_label = "Heel Weights"
    bl_order = 1
    weights_tabs = {'HEELS'}
    kind = 'HEELS'

    @staticmethod
    def draw_mode(layout, settings):
        _draw_heels_main(layout, settings)


class VIEW3D_PT_heels_brush_sidebar(_WeightBrushPanel, bpy.types.Panel):
    """Paint heel weights with the brush, with the Heel Weights settings"""
    bl_label = "Heels Brush"
    bl_order = 2
    weights_tabs = {'HEELS'}
    button = "Paint Heel Weights"


class _ComingPanel(tabs.TabPanel):
    """A sub-tab of the Weights tab whose weights aren't there yet."""
    what = ''

    def draw(self, context):
        box = self.layout.box().column(align=True)
        box.label(text="Weights for {:s}".format(self.what), icon='INFO')
        box.label(text="come in a later version", icon='BLANK1')


class VIEW3D_PT_hair_weights_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Give whole hair meshes FFXIV hair bone weights, using the hair skeleton that fits best"""
    bl_label = "Hair Weights"
    bl_order = 1
    weights_tabs = {'HAIR'}

    def draw(self, context):
        layout = _panel_layout(self.layout)
        settings = context.scene.magic_fit
        col = layout.column()
        col.prop(settings, "hair_race")
        _draw_choice(col, settings, "hair_skeleton_mode")
        if settings.hair_skeleton_mode == 'PICK':
            row = col.row(align=True)
            row.prop(settings, "hair_skeleton", text="EST Entry")
            row.operator(MAGIC_FIT_OT_pick_hair_skeleton.bl_idname, text="", icon='VIEWZOOM')
            self.draw_pick_hint(col, settings)
        _draw_hair_main(col, settings)
        _draw_hair_more(col, settings)
        if settings.group_mode == 'ACTIVE':
            # Hair Weights needs All Groups, and no other panel on this tab has the choice.
            _draw_choice(col, settings, "group_mode")
        problem = check_hair_weights_ready(context)
        if problem is not None:
            col.box().label(text=problem, icon='ERROR')
        col.separator()
        row = col.row(align=True)
        row.scale_y = 1.4
        row.prop(settings, "hair_apply_to_selected", text="", icon='RESTRICT_SELECT_OFF')
        row.operator(MAGIC_FIT_OT_hair_weights.bl_idname, icon='OUTLINER_OB_CURVES')
        self.draw_result(context, col, settings)

    @staticmethod
    def draw_pick_hint(layout, settings):
        """Which of the game's hairstyles use the picked EST entry."""
        if not settings.hair_skeleton:
            return
        try:
            reference = hairing.Reference.get()
        except (OSError, KeyError, ValueError):
            return
        if not reference.has_skeleton(settings.hair_race, settings.hair_skeleton):
            return  # the problem box says so
        hairs = reference.vanilla_hairs(settings.hair_race, settings.hair_skeleton)
        row = layout.row()
        row.alignment = 'RIGHT'
        row.label(text="Like the game's hair " + ", ".join(str(h) for h in hairs[:6]) if hairs else
                  "No game hairstyle uses it")

    @staticmethod
    def draw_result(context, layout, settings):
        """The skeleton the active hair got, and the others that fit it well, to try instead."""
        obj = context.object
        state = getattr(obj, "magic_fit_hair", None) if obj is not None else None
        if state is None or state.skeleton < 0 or state.race != settings.hair_race:
            return
        try:
            reference = hairing.Reference.get()
        except (OSError, KeyError, ValueError):
            return
        box = layout.box().column(align=True)
        if state.skeleton:
            box.label(text="EST entry {:d} ({:s})".format(state.skeleton, hairing.skeleton_label(state.skeleton)),
                      icon='ARMATURE_DATA')
            hairs = reference.vanilla_hairs(state.race, state.skeleton)
            if hairs:
                box.label(text="Like the game's hair " + ", ".join(str(h) for h in hairs[:6]), icon='BLANK1')
        else:
            box.label(text="EST entry 0: the body's hair bones only", icon='ARMATURE_DATA')
        matches = [m for m in parse_matches(state.matches) if m[0] != state.skeleton]
        if matches:
            box.separator()
            box.label(text="Also fitting:")
            grid = box.grid_flow(columns=3, even_columns=True, align=True)
            for number, _score in matches[:5]:
                op = grid.operator(MAGIC_FIT_OT_hair_weights.bl_idname, text=hairing.skeleton_label(number))
                op.skeleton = number
        box.separator()
        op = box.operator(MAGIC_FIT_OT_pick_hair_skeleton.bl_idname, text="Try Another EST Entry", icon='VIEWZOOM')
        op.apply = True


class VIEW3D_PT_hair_brush_sidebar(_WeightBrushPanel, bpy.types.Panel):
    """Paint hair weights with the brush, using the Hair Weights settings and the hair's skeleton"""
    bl_label = "Hair Brush"
    bl_order = 2
    weights_tabs = {'HAIR'}
    button = "Paint Hair Weights"


class VIEW3D_PT_face_weights_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Weight custom faces and their parts like the game face they replace"""
    bl_label = "Face Weights"
    bl_order = 1
    weights_tabs = {'FACE'}

    def draw(self, context):
        layout = _panel_layout(self.layout)
        settings = context.scene.magic_fit
        col = layout.column()
        _draw_face_experimental(col)
        _draw_game_face(col, settings)
        col.prop(settings, "face_close_eyes")
        col.prop(settings, "face_match_neck")
        col.prop(settings, "face_max_groups")
        if settings.group_mode == 'ACTIVE':
            # Face Weights and the repairs need All Groups, and no other panel on this tab has the choice.
            _draw_choice(col, settings, "group_mode")
        problem = check_face_tools_ready(context)
        if problem is not None:
            _draw_face_problem(col, problem)
        col.separator()
        row = col.row()
        row.scale_y = 1.4
        row.operator(MAGIC_FIT_OT_face_weights.bl_idname, icon='USER')


class VIEW3D_PT_face_repair_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Repair one thing at a time on a custom face: lids, lashes, mouth, parts or neck"""
    bl_label = "Face Repairs"
    bl_order = 2
    weights_tabs = {'FACE'}

    def draw(self, context):
        layout = _panel_layout(self.layout)
        settings = context.scene.magic_fit
        col = layout.column()
        for repair, label, _description, icon, _args in FACE_REPAIRS:
            row = col.row()
            row.operator(MAGIC_FIT_OT_face_repair.bl_idname, text=label, icon=icon).repair = repair
            if repair == 'LASHES':
                col.prop(settings, "face_lash_lift")


class VIEW3D_PT_face_pose_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Pose the face like the game's animations, to check how it moves"""
    bl_label = "Test Poses"
    bl_order = 3
    weights_tabs = {'FACE'}

    def draw(self, context):
        layout = _panel_layout(self.layout)
        col = layout.column()
        problem = check_face_pose_ready(context)
        if problem is not None:
            _draw_face_problem(col, problem)
        grid = col.grid_flow(columns=3, even_columns=True, align=True)
        for pose, label, _description in facing.POSES:
            grid.operator(MAGIC_FIT_OT_face_pose.bl_idname, text=label).pose = pose
        col.operator(MAGIC_FIT_OT_face_pose.bl_idname, text="Rest", icon='LOOP_BACK').pose = 'REST'


class VIEW3D_PT_face_brush_sidebar(_WeightBrushPanel, bpy.types.Panel):
    """Paint face weights with the brush, with the Face Weights settings"""
    bl_label = "Face Brush"
    bl_order = 4
    weights_tabs = {'FACE'}
    button = "Paint Face Weights"


# -----------------------------------------------------------------------------
# Body Fit tab

class VIEW3D_PT_body_fit_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Fit meshes around the body with the Body Fit brush"""
    bl_label = "Body Fit"
    bl_order = 1
    tab = 'FIT'

    def draw(self, context):
        layout = _panel_layout(self.layout)
        if _tool_is_active(context, 'EDIT_MESH', FIT_TOOL_IDNAME):
            _draw_fit_settings(context, layout, body=False)
            return
        settings = context.scene.body_fit_brush
        col = layout.column()
        _draw_choice(col, settings, "mode")
        col.prop(settings, "offset")
        _draw_use_button(col, 'FIT', "Fit to Body", "body_fit")


class VIEW3D_PT_fit_move_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Move vertices with proportional editing that keeps to the mesh"""
    bl_label = "Fit Move"
    bl_order = 1
    tab = 'FIT'

    def draw(self, context):
        layout = _panel_layout(self.layout)
        if _tool_is_active(context, 'EDIT_MESH', MOVE_TOOL_IDNAME):
            _draw_move_settings(context, layout)
            return
        col = layout.column()
        col.label(text="Moves the selection without reaching across gaps")
        _draw_use_button(col, 'MOVE', "Fit Move", "fit_move")


def _draw_key_search(layout, settings, prop, obj, text):
    """A shape key of ``obj`` to pick, by name."""
    keys = obj.data.shape_keys if obj is not None and obj.type == 'MESH' else None
    if keys is None:
        row = layout.row()
        row.enabled = False
        row.prop(settings, prop, text=text, icon='SHAPEKEY_DATA')
    else:
        layout.prop_search(settings, prop, keys, "key_blocks", text=text, icon='SHAPEKEY_DATA')


# UILayout.prop takes a placeholder from Blender 4.1 on.
_HAS_PLACEHOLDER = 'placeholder' in bpy.types.UILayout.bl_rna.functions['prop'].parameters.keys()


class VIEW3D_PT_resize_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Reshape meshes made for one body to fit another, such as another body mod"""
    bl_label = "Resize"
    bl_order = 2
    tab = 'FIT'

    def draw(self, context):
        layout = _panel_layout(self.layout)
        settings = context.scene.body_fit_brush
        body = settings.resize_from_object
        col = layout.column()
        row = col.row()
        row.alert = body is None
        row.prop(settings, "resize_from_object")
        col.prop(settings, "resize_to_object")
        col.prop(settings, "resize_use_keys")
        if settings.resize_use_keys:
            _draw_key_search(col, settings, "resize_from_key", body, "From Shapekey")
            _draw_key_search(col, settings, "resize_to_key", settings.resize_to_object or body, "To Shapekey")
        col.prop(settings, "resize_radius")
        col.prop(settings, "resize_bridge")
        col.prop(settings, "resize_rigid_size")
        col.prop(settings, "resize_keep_close")
        if _HAS_PLACEHOLDER:
            col.prop(settings, "resize_key_name", placeholder=resize_key_name(context))
        else:
            col.prop(settings, "resize_key_name")
        problem = check_resize_ready(context)
        if problem is not None:
            col.box().label(text=problem, icon='ERROR')
        col.separator()
        row = col.row()
        row.scale_y = 1.4
        row.operator(MAGIC_FIT_OT_resize.bl_idname, icon='SHAPEKEY_DATA')


class VIEW3D_PT_clipping_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Shows where the edited meshes clip into the body"""
    bl_label = "Clipping"
    bl_order = 3
    tab = 'FIT'

    def draw_header(self, context):
        self.layout.prop(context.scene.magic_fit_clipping, "show", text="")

    def draw(self, context):
        settings = context.scene.magic_fit_clipping
        layout = _panel_layout(self.layout)
        col = layout.column()
        col.active = settings.show
        col.prop(settings, "color")
        col.prop(settings, "margin")
        col.prop(settings, "show_in_front")
        if not settings.show:
            return
        if context.scene.magic_fit.target is None:
            layout.label(text="Pick the body above", icon='INFO')
            return
        report = clipping.status(context)
        if report is None:
            return
        if not report:
            layout.label(text="Edit or select a mesh to check it", icon='INFO')
            return
        col = layout.column(align=True)
        for name, inside, close, measuring in report:
            if measuring:
                text, icon = "{:s}: measuring...".format(name), 'TIME'
            elif inside:
                text, icon = "{:s}: {:d} vertices clip".format(name, inside), 'ERROR'
            else:
                text, icon = "{:s}: clear".format(name), 'CHECKMARK'
            if close and not measuring:
                text += ", {:d} within the margin".format(close)
            col.label(text=text, icon=icon)


# -----------------------------------------------------------------------------
# Line Up and Texture Relax tabs (Customize+ has its own module)

class VIEW3D_PT_line_up_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Scale, turn and pose an imported model so it lines up with the body"""
    bl_label = "Line Up"
    bl_order = 1
    tab = 'LINE_UP'

    def draw(self, context):
        layout = _panel_layout(self.layout)
        settings = context.scene.body_fit_brush
        col = layout.column()
        col.prop(settings, "lineup_stretch")
        col.prop(settings, "lineup_fit_shape")
        problem = check_lineup_ready(context)
        if problem is not None:
            col.box().label(text=problem, icon='ERROR')
        else:
            summary = lineup_summary(context)
            if summary:
                col.label(text=summary, icon='INFO')
        col.separator()
        row = col.row()
        row.scale_y = 1.4
        row.operator(MAGIC_FIT_OT_line_up.bl_idname, icon='ARMATURE_DATA')


class VIEW3D_PT_texture_relax_sidebar(tabs.TabPanel, bpy.types.Panel):
    """Fix stretched or squashed textures with the Texture Relax brush"""
    bl_label = "Texture Relax"
    bl_order = 1
    tab = 'RELAX'

    def draw(self, context):
        layout = _panel_layout(self.layout)
        if _tool_is_active(context, 'EDIT_MESH', RELAX_TOOL_IDNAME):
            _draw_relax_settings(context, layout)
            return
        col = layout.column()
        col.label(text="Slides vertices until the texture is even")
        _draw_choice(col, context.scene.texture_relax_brush, "boundary")
        _draw_use_button(col, 'RELAX', "Relax Texture", "texture_relax")


# -----------------------------------------------------------------------------
# Brush cursor

def _batch(type, coords):
    fmt = GPUVertFormat()
    pos_id = fmt.attr_add(id="pos", comp_type='F32', len=2, fetch_mode='FLOAT')
    vbo = GPUVertBuf(len=len(coords), format=fmt)
    vbo.attr_fill(id=pos_id, data=coords)
    return GPUBatch(type=type, buf=vbo)


def _draw_lines(type, coords, color, width):
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", width)
    shader.uniform_float("color", color)
    _batch(type, coords).draw(shader)


def _draw_rect(x0, y0, x1, y1, color):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.uniform_float("color", color)
    _batch('TRIS', [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]).draw(shader)


def _draw_brush_cursor(xy, radius, problem, ui_scale):
    """Brush circle; dashed with a message underneath while the brush can't be used."""
    x, y = xy
    segments = max(32, min(128, int(radius / 2)) // 2 * 2)
    step = 2.0 * math.pi / segments
    coords = [(x + radius * math.cos(i * step), y + radius * math.sin(i * step)) for i in range(segments + 1)]
    if problem:
        # Every other segment, drawn as separate lines.
        type, coords = 'LINES', [co for i in range(0, segments, 2) for co in coords[i:i + 2]]
    else:
        type = 'LINE_STRIP'

    gpu.state.blend_set('ALPHA')
    _draw_lines(type, coords, (0.0, 0.0, 0.0, 0.45), 3.0 * ui_scale)
    _draw_lines(type, coords, (1.0, 1.0, 1.0, 0.85), 1.5 * ui_scale)

    if problem:
        # Weight colors cover the whole spectrum, so put the text on a dark plate.
        font_id = 0
        blf.size(font_id, 12.0 * ui_scale)
        width, height = blf.dimensions(font_id, problem)
        pad = 4.0 * ui_scale
        text_x = x - width / 2.0
        text_y = y - radius - height - 8.0 * ui_scale
        _draw_rect(text_x - pad, text_y - pad, text_x + width + pad, text_y + height + pad, (0.0, 0.0, 0.0, 0.6))
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.position(font_id, text_x, text_y, 0.0)
        blf.draw(font_id, problem)
    gpu.state.blend_set('NONE')


def _tool_is_active(context, mode, idname):
    # Cursor callbacks can outlive their tool (mode switch, add-on disabled), so check first.
    if context.mode != mode:
        return False
    tool = context.workspace.tools.from_space_view3d_mode(mode, create=False)
    return tool is not None and tool.idname == idname


# -----------------------------------------------------------------------------
# Tools

class MagicFitTool(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'PAINT_WEIGHT'
    bl_idname = WEIGHT_TOOL_IDNAME
    bl_label = "Weight Brushes"
    bl_description = (
        "Paint to copy weights from a target mesh,\n"
        "straighten creases in the pose, smooth weights,\n"
        "or make skirt, heel, hair and face weights"
    )
    bl_icon = os.path.join(ICONS_DIR, "weight_brushes")
    bl_cursor = 'PAINT_CROSS'
    bl_widget = None
    bl_keymap = (
        (MAGIC_FIT_OT_paint.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        *_brush_keymap("scene.magic_fit"),
    )

    @staticmethod
    def draw_settings(context, layout, _tool):
        _draw_weight_settings(context, layout)

    @staticmethod
    def draw_cursor(context, _tool, xy):
        settings = getattr(context.scene, "magic_fit", None)
        if settings is not None and _tool_is_active(context, 'PAINT_WEIGHT', WEIGHT_TOOL_IDNAME):
            _draw_brush_cursor(
                xy, settings.radius, check_ready(context, settings), context.preferences.system.ui_scale,
            )


class BodyFitTool(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_idname = FIT_TOOL_IDNAME
    bl_label = "Body Fit"
    bl_description = (
        "Sculpt the mesh around the body: push it out or tighten it (Ctrl: swap).\n"
        "All layers move together, so thickness and details are kept"
    )
    bl_icon = os.path.join(ICONS_DIR, "body_fit")
    bl_cursor = 'PAINT_CROSS'
    bl_widget = None
    bl_keymap = (
        (MAGIC_FIT_OT_body_fit.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        (MAGIC_FIT_OT_body_fit.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True},
         {"properties": [("invert", True)]}),
        *_brush_keymap("scene.body_fit_brush"),
    )

    @staticmethod
    def draw_settings(context, layout, _tool):
        _draw_fit_settings(context, layout)

    @staticmethod
    def draw_cursor(context, _tool, xy):
        settings = getattr(context.scene, "body_fit_brush", None)
        if settings is not None and _tool_is_active(context, 'EDIT_MESH', FIT_TOOL_IDNAME):
            _draw_brush_cursor(
                xy, settings.radius, check_fit_ready(context), context.preferences.system.ui_scale,
            )


class TextureRelaxTool(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_idname = RELAX_TOOL_IDNAME
    bl_label = "Texture Relax"
    bl_description = (
        "Slide vertices until the texture on them is no longer\n"
        "stretched or squashed. The shape doesn't change"
    )
    bl_icon = os.path.join(ICONS_DIR, "texture_relax")
    bl_cursor = 'PAINT_CROSS'
    bl_widget = None
    bl_keymap = (
        (MAGIC_FIT_OT_texture_relax.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        *_brush_keymap("scene.texture_relax_brush"),
    )

    @staticmethod
    def draw_settings(context, layout, _tool):
        _draw_relax_settings(context, layout)

    @staticmethod
    def draw_cursor(context, _tool, xy):
        settings = getattr(context.scene, "texture_relax_brush", None)
        if settings is not None and _tool_is_active(context, 'EDIT_MESH', RELAX_TOOL_IDNAME):
            _draw_brush_cursor(
                xy, settings.radius, check_relax_ready(context), context.preferences.system.ui_scale,
            )


class FitMoveTool(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_idname = MOVE_TOOL_IDNAME
    bl_label = "Fit Move"
    bl_description = (
        "Drag or press G to move the selection. Proportional editing reaches along the mesh,\n"
        "not across gaps, keeps split seams closed and fades out toward hidden vertices"
    )
    bl_icon = os.path.join(ICONS_DIR, "fit_move")
    bl_widget = None
    # Like Blender's Move tool: drag to move (starting on an unselected vertex selects it first), click to
    # select, Shift click to add or remove.
    bl_keymap = (
        (MAGIC_FIT_OT_fit_move.bl_idname, {"type": 'LEFTMOUSE', "value": 'CLICK_DRAG'},
         {"properties": [("release_confirm", True)]}),
        (MAGIC_FIT_OT_fit_move.bl_idname, {"type": 'G', "value": 'PRESS'}, None),
        ("view3d.select", {"type": 'LEFTMOUSE', "value": 'PRESS'},
         {"properties": [("deselect_all", True), ("select_passthrough", True)]}),
        ("view3d.select", {"type": 'LEFTMOUSE', "value": 'CLICK'}, {"properties": [("deselect_all", True)]}),
        ("view3d.select", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True}, {"properties": [("toggle", True)]}),
    )

    @staticmethod
    def draw_settings(context, layout, _tool):
        _draw_move_settings(context, layout)


# The tabs first. The other sidebar panels are sorted by their bl_order, then by registration; the
# Weight Transfer and Customize+ panels are registered with their modules.
classes = (
    VIEW3D_PT_magic_fit_options,
    VIEW3D_PT_body_fit_options,
    VIEW3D_PT_texture_relax_options,
    VIEW3D_MT_magic_fit_links,
    VIEW3D_PT_magic_fit_tabs,
    VIEW3D_PT_magic_fit_sidebar,
    VIEW3D_PT_skirt_weights_sidebar,
    VIEW3D_PT_skirt_brush_sidebar,
    VIEW3D_PT_heel_weights_sidebar,
    VIEW3D_PT_heels_brush_sidebar,
    VIEW3D_PT_hair_weights_sidebar,
    VIEW3D_PT_hair_brush_sidebar,
    VIEW3D_PT_face_weights_sidebar,
    VIEW3D_PT_face_repair_sidebar,
    VIEW3D_PT_face_pose_sidebar,
    VIEW3D_PT_face_brush_sidebar,
    VIEW3D_PT_body_fit_sidebar,
    VIEW3D_PT_fit_move_sidebar,
    VIEW3D_PT_resize_sidebar,
    VIEW3D_PT_clipping_sidebar,
    VIEW3D_PT_line_up_sidebar,
    VIEW3D_PT_texture_relax_sidebar,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.utils.register_tool(MagicFitTool, after={"builtin.gradient"})
    bpy.utils.register_tool(BodyFitTool, separator=True)
    bpy.utils.register_tool(TextureRelaxTool, after={FIT_TOOL_IDNAME})
    bpy.utils.register_tool(FitMoveTool, after={RELAX_TOOL_IDNAME})


def unregister():
    bpy.utils.unregister_tool(FitMoveTool)
    bpy.utils.unregister_tool(TextureRelaxTool)
    bpy.utils.unregister_tool(BodyFitTool)
    bpy.utils.unregister_tool(MagicFitTool)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
