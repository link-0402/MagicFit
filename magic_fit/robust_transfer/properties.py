# SPDX-License-Identifier: GPL-3.0-or-later
#
# Settings of Robust Weight Transfer by sentfromspacevr (https://github.com/sentfromspacevr), GPL-2.0-or-later.
# The body to transfer from is the one shared by all tools, `Scene.magic_fit.target`. There's one way of
# matching (Crease Smoothing, which picks how to treat each part of the body by itself), so the standalone
# add-on's Max Distance, normal angle, smoothing and Inpaint Mode settings are gone.

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty


class WeightTransferObjectSettings(bpy.types.PropertyGroup):
    vertex_group: StringProperty(
        name="Transfer Mask",
        description=(
            "Only transfer where this vertex group has weight. Partial weight blends with the old weights. "
            "Not used with Apply to All Selected"
        ),
    )
    vertex_group_invert: BoolProperty(name="Invert", description="Transfer where the Transfer Mask has no weight")
    inpaint_group: StringProperty(
        name="Inpaint Mask",
        description=(
            "Vertices in this group get weights filled in from the weights around them, not copied from "
            "the body. The Inpaint button uses it to repair weights"
        ),
    )
    inpaint_group_invert: BoolProperty(name="Invert Inpaint", description="Fill in weights outside the Inpaint Mask")
    inpaint_threshold: FloatProperty(
        name="Inpaint Binary Threshold",
        description="Vertices with an Inpaint Mask weight above this count as inside the mask",
        default=0.5,
        min=0,
        max=1,
    )
    rigid: BoolProperty(
        name="Rigid",
        description=(
            "Give each piece of the mesh one set of weights, so it moves as one piece, like metal parts. Loose "
            "parts that touch are one piece. Also works with Apply to All Selected, for each mesh that has it on: "
            "the other meshes transferred with it hold onto it where they touch it"
        ),
        default=False,
    )


def update_enforce_four_bone_limit(self, _context):
    """Limiting groups only makes sense for deform bones."""
    if self.enforce_four_bone_limit:
        self.group_selection = 'DEFORM_POSE_BONES'


class WeightTransferSettings(bpy.types.PropertyGroup):
    use_deformed_source: BoolProperty(
        name="Use Deformed Body",
        description=(
            "Use the body as shaped by its modifiers and shape keys. The pose only counts for meshes that "
            "don't follow the body's armature. Customize+ never counts"
        ),
        default=True,
    )
    use_deformed_target: BoolProperty(
        name="Use Deformed Target",
        description=(
            "Use the meshes as shaped by their modifiers and shape keys, in rest pose if they follow the "
            "body's armature"
        ),
        default=True,
    )
    apply_to_selected: BoolProperty(
        name="Apply to All Selected",
        description="Transfer to every selected mesh, not only the active one",
        default=False,
    )
    partial_reweight: BoolProperty(
        name="Partial Reweight",
        description=(
            "Only transfer within 5 cm of the body. Weights further away are kept as they are"
        ),
        default=False,
    )
    partial_reweight_falloff: FloatProperty(
        name="Falloff Width",
        description=(
            "How much of the 5 cm is used to fade into the old weights at the edge. Zero gives a hard cut"
        ),
        default=20,
        min=0,
        max=100,
        subtype='PERCENTAGE',
    )
    enforce_four_bone_limit: BoolProperty(
        name="Limit Groups per Vertex",
        description="Limit how many groups can move each vertex. FFXIV allows 8, VRChat and Unity 4",
        default=True,
        update=update_enforce_four_bone_limit,
    )
    num_limit_groups: IntProperty(
        name="Max Groups per Vertex",
        description=(
            "How many groups each vertex may have: 8 for FFXIV, 4 for VRChat and Unity. The smallest "
            "weights go first, often the physics bones"
        ),
        min=1,
        default=8,
    )
    group_selection: EnumProperty(
        name="Subset",
        description="Which of the body's vertex groups to transfer",
        items=(
            ('ALL_GROUPS', "All Groups", "Transfer every group"),
            ('DEFORM_POSE_BONES', "Deform Pose Bones", "Only transfer the groups of the armature's deform bones"),
        ),
        default='DEFORM_POSE_BONES',
    )
    virtual_merge: BoolProperty(
        name="Virtual Merge by Distance",
        description=(
            "Treat vertices close together as joined while filling in weights. The mesh itself isn't changed"
        ),
        default=True,
    )
    virtual_merge_distance: FloatProperty(
        name="Virtual Merge Distance",
        description="How close vertices must be to count as joined while filling in weights",
        default=0.0001,
        min=0,
        soft_max=0.01,
        precision=5,
        unit='LENGTH',
        subtype='DISTANCE',
    )
    seam_sync: BoolProperty(
        name="Synchronize Seam Weights",
        description=(
            "Give touching edges of separate parts the same weights, so seams don't open when bones move or "
            "scale. The meshes must be single-user"
        ),
        default=True,
    )
    seam_distance: FloatProperty(
        name="Seam Distance",
        description="How close the edges of two parts must be to count as touching. Zero: only exact overlaps",
        default=0.0001,
        min=0,
        soft_max=0.01,
        precision=5,
        unit='LENGTH',
        subtype='DISTANCE',
    )
    seam_sync_across_objects: BoolProperty(
        name="Across Selected Objects",
        description=(
            "Also match up seams between the selected meshes. With Virtual Merge on, their weights are also "
            "filled in together"
        ),
        default=False,
    )
    normalize_weights_after_transfer: BoolProperty(
        name="Normalize Weights After Transfer",
        description="Make the bone weights of each changed vertex add up to 1. Locked groups aren't changed",
        default=False,
    )


classes = (
    WeightTransferObjectSettings,
    WeightTransferSettings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.magic_fit_transfer = PointerProperty(type=WeightTransferObjectSettings)
    bpy.types.Scene.magic_fit_transfer = PointerProperty(type=WeightTransferSettings)


def unregister():
    del bpy.types.Scene.magic_fit_transfer
    del bpy.types.Object.magic_fit_transfer
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
