# SPDX-License-Identifier: GPL-3.0-or-later

"""The tabs along the top of the sidebar's Magic Fit tab, their sub-tabs, and which panels show on which.

The tab shown is kept per workspace, and so are the sub-tabs of the Body Fit and Mesh Repair tabs.
Workspaces aren't part of undo, so switching tabs adds no undo steps. The Weights tab has a sub-tab per
mode of the Weight Brushes (`Scene.magic_fit.weights_tab`), which switches the brushes' mode along.
"""

import bpy
from bpy.props import EnumProperty

CATEGORY = "Magic Fit"

# Items are stored by number: new ones get the next free one, wherever they're shown.
TABS = (
    ('WEIGHTS', "Weights",
     "Copy and smooth the body's weights, and make skirt, heel, hair and face weights", 0),
    ('FIT', "Body Fit", "Fit meshes around the body, move them along it, and see where they clip", 1),
    ('RESIZE', "Auto Resize", "Reshape meshes made for one body to fit another, or other sizes of it", 5),
    ('LINE_UP', "Line Up", "Line imported models up with the body, and hair up on the head", 2),
    ('REPAIR', "Mesh Repair",
     "Relax stretched textures, close rips, rebuild messy meshes, and clean up their normals and UVs", 3),
    ('CPLUS', "Customize+", "Show a Customize+ template on the armatures", 4),
)
# Drawn where the tabs have room for them.
ICONS = {
    'WEIGHTS': 'WPAINT_HLT',
    'FIT': 'MOD_SHRINKWRAP',
    'RESIZE': 'SHAPEKEY_DATA',
    'LINE_UP': 'ARMATURE_DATA',
    'REPAIR': 'TOOL_SETTINGS',
    'CPLUS': 'BONE_DATA',
}

FIT_TABS = (
    ('FIT', "Fit", "Sculpt meshes around the body with the Body Fit brush", 0),
    ('MOVE', "Move", "Move the selection with proportional editing that keeps to the mesh", 1),
    ('CLIPPING', "Clipping", "Show where the edited meshes clip into the body", 2),
)
REPAIR_TABS = (
    ('RELAX', "Texture Relax", "Fix stretched or squashed textures with the Texture Relax brush", 0),
    ('RIPS', "Close Rips", "Find where a mesh was pulled apart, and pull it back together", 1),
    ('REBUILD', "Mesh Rebuild", "Rebuild messy meshes with clean, even triangles, keeping thin details", 2),
    ('NORMALS', "Normals", "Clean smooth normals, and normal maps that show lines along UV seams", 3),
    ('UV', "UV", "Rebuild UV layouts, move textures onto them, resize UV canvases and make index maps", 4),
)
# Per tab with sub-tabs of its own: the workspace property that keeps the one shown, and their items.
SUBTABS = {
    'FIT': ("magic_fit_fit_tab", FIT_TABS),
    'REPAIR': ("magic_fit_repair_tab", REPAIR_TABS),
}


def current(context):
    """The tab shown in the current workspace."""
    return getattr(context.workspace, "magic_fit_tab", 'WEIGHTS')


def current_subtab(context, tab):
    """The sub-tab of ``tab`` shown: the Weights tab's is the brushes' mode, the others' are per workspace."""
    if tab == 'WEIGHTS':
        return context.scene.magic_fit.weights_tab
    prop, items = SUBTABS[tab]
    return getattr(context.workspace, prop, items[0][0])


def show(context, tab, subtab=None):
    """Show ``tab`` (and its ``subtab``) in the current workspace."""
    workspace = context.workspace
    if workspace is None:
        return
    workspace.magic_fit_tab = tab
    if subtab is not None:
        setattr(workspace, SUBTABS[tab][0], subtab)


class TabPanel:
    """A panel of the sidebar's Magic Fit tab, shown on one of its tabs, and on tabs with sub-tabs maybe
    only on some of those."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = CATEGORY
    tab = 'WEIGHTS'
    subtabs = None  # the sub-tabs it's on, None for all

    @classmethod
    def poll(cls, context):
        if current(context) != cls.tab:
            return False
        return cls.subtabs is None or current_subtab(context, cls.tab) in cls.subtabs


def register():
    bpy.types.WorkSpace.magic_fit_tab = EnumProperty(
        name="Magic Fit Tab",
        description="Which tools the sidebar's Magic Fit tab shows",
        items=TABS,
        default='WEIGHTS',
    )
    bpy.types.WorkSpace.magic_fit_fit_tab = EnumProperty(
        name="Body Fit",
        description="Which of the Body Fit tools to show",
        items=FIT_TABS,
        default='FIT',
    )
    bpy.types.WorkSpace.magic_fit_repair_tab = EnumProperty(
        name="Mesh Repair",
        description="Which of the Mesh Repair tools to show",
        items=REPAIR_TABS,
        default='RELAX',
    )


def unregister():
    del bpy.types.WorkSpace.magic_fit_repair_tab
    del bpy.types.WorkSpace.magic_fit_fit_tab
    del bpy.types.WorkSpace.magic_fit_tab
