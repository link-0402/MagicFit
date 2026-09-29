# SPDX-License-Identifier: GPL-3.0-or-later
#
# Settings of Robust Weight Transfer by sentfromspacevr (https://github.com/sentfromspacevr), GPL-2.0-or-later.
# The body to transfer from is the one shared by all tools, `Scene.magic_fit.target`.

import math

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
            "Give the whole mesh one set of weights, so it moves as one piece, like metal parts. Also works "
            "with Apply to All Selected, for each mesh that has it on"
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
    max_distance: FloatProperty(
        name="Max Distance",
        description=(
            "How far from the body a vertex can be and still copy its weights directly. With Partial "
            "Reweight, also where the transfer stops"
        ),
        default=0.05,
        min=0,
        unit='LENGTH',
        subtype='DISTANCE',
    )
    partial_reweight: BoolProperty(
        name="Partial Reweight",
        description=(
            "Only transfer within Max Distance of the body. Weights further away are kept as they are"
        ),
        default=False,
    )
    partial_reweight_falloff: FloatProperty(
        name="Falloff Width",
        description=(
            "How much of Max Distance is used to fade into the old weights at the edge. Zero gives a hard cut"
        ),
        default=20,
        min=0,
        max=100,
        subtype='PERCENTAGE',
    )
    max_normal_angle_difference: FloatProperty(
        name="Max Normal Difference",
        description=(
            "How far the mesh may face away from the body and still copy its weights exactly. Keep it low; "
            "Soft Normal Limit handles bigger angles more gently. Not used with Crease Smoothing"
        ),
        default=math.radians(30),
        min=0,
        max=math.pi,
        precision=3,
        step=100,
        unit='ROTATION',
        subtype='ANGLE',
    )
    soft_normal_limit: FloatProperty(
        name="Soft Normal Limit",
        description=(
            "Vertices past Max Normal Difference still guide the filled-in weights, less and less up to this "
            "angle. Keeps steep or detailed parts in place. Set it to Max Normal Difference to turn it off. "
            "Not used with Crease Smoothing"
        ),
        default=math.radians(90),
        min=0,
        max=math.pi,
        precision=3,
        step=100,
        unit='ROTATION',
        subtype='ANGLE',
    )
    flip_vertex_normal: BoolProperty(
        name="Allow Flipped Normals",
        description=(
            "Also match where the mesh faces the opposite way from the body. Crease Smoothing always does"
        ),
        default=True,
    )
    crease_smoothing: BoolProperty(
        name="Crease Smoothing",
        description=(
            "Copy the body's weights where the mesh lies on it, and blend them where it bridges a crease, like "
            "under the bust or between the legs. Stops posed or scaled bones pinching the mesh into the "
            "crease. Replaces Normal Angle, Soft Limit, Allow Flipped and Smoothing"
        ),
        default=True,
    )
    crease_softness: FloatProperty(
        name="Crease Width",
        description=(
            "How widely weights blend where the mesh floats above the body. Raise it for smoother blends "
            "over hollows; lower it to follow the body's weights more closely"
        ),
        default=1.0,
        min=0.25,
        soft_max=2.0,
        step=10,
        precision=2,
    )
    smoothing_enable: BoolProperty(
        name="Smoothing",
        description="Smooth the filled-in weights. Not used with Crease Smoothing",
        default=False,
    )
    smoothing_factor: FloatProperty(
        name="Smoothing Factor",
        description="How strongly each smoothing pass blends a vertex with its neighbours",
        default=0.2,
        min=0,
        max=1,
        step=10,
    )
    smoothing_repeat: IntProperty(
        name="Smoothing Repeat",
        description="How many smoothing passes to run",
        default=4,
        min=0,
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
    inpaint_mode: EnumProperty(
        name="Inpaint Mode",
        description="How weights are filled in where the body doesn't match",
        # Files store each mode's number, which must stay the same; the list shows the usual one first.
        items=(
            ('COMBINED', "Combined",
             "The usual choice. Weights spread along each part and to parts touching it, like a lining, but "
             "don't jump gaps, like between the legs of loose pants. Seams stay closed",
             2),
            ('POINT', "Point",
             "Weights spread to anything nearby, joined or not, so they can bleed across gaps, like from one "
             "pant leg to the other. Only faster than Combined where big parts lie on each other",
             0),
            ('SURFACE', "Surface",
             "Weights only spread along each part's own surface. Fails on parts with nothing matching the "
             "body, and split seams can tear",
             1),
        ),
        default='COMBINED',
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
