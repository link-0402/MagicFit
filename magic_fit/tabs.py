# SPDX-License-Identifier: GPL-3.0-or-later

"""The tabs along the top of the sidebar's Magic Fit tab, and which panels show on which.

The tab shown is kept per workspace. Workspaces aren't part of undo, so switching tabs adds no undo
steps. The Weights tab has a sub-tab per mode of the Weight Brushes
(`Scene.magic_fit.weights_tab`), which switches the brushes' mode along.
"""

import bpy
from bpy.props import EnumProperty

CATEGORY = "Magic Fit"

TABS = (
    ('WEIGHTS', "Weights",
     "Copy and smooth the body's weights, and make skirt, heel, hair and face weights", 0),
    ('FIT', "Body Fit",
     "Fit meshes around the body, resize them to another body, and see where they clip", 1),
    ('LINE_UP', "Line Up", "Line imported models up with the body", 2),
    ('RELAX', "Texture Relax", "Repair textures that got stretched or squashed", 3),
    ('CPLUS', "Customize+", "Show a Customize+ template on the armatures", 4),
)
# Drawn where the tabs have room for them.
ICONS = {
    'WEIGHTS': 'WPAINT_HLT',
    'FIT': 'MOD_SHRINKWRAP',
    'LINE_UP': 'ARMATURE_DATA',
    'RELAX': 'TEXTURE',
    'CPLUS': 'BONE_DATA',
}


def current(context):
    """The tab shown in the current workspace."""
    return getattr(context.workspace, "magic_fit_tab", 'WEIGHTS')


class TabPanel:
    """A panel of the sidebar's Magic Fit tab, shown on one of its tabs (and on the Weights tab,
    maybe only on some of its sub-tabs)."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = CATEGORY
    tab = 'WEIGHTS'
    weights_tabs = None  # on the Weights tab: the sub-tabs it's on, None for all

    @classmethod
    def poll(cls, context):
        if current(context) != cls.tab:
            return False
        return cls.weights_tabs is None or context.scene.magic_fit.weights_tab in cls.weights_tabs


def register():
    bpy.types.WorkSpace.magic_fit_tab = EnumProperty(
        name="Magic Fit Tab",
        description="Which tools the sidebar's Magic Fit tab shows",
        items=TABS,
        default='WEIGHTS',
    )


def unregister():
    del bpy.types.WorkSpace.magic_fit_tab
