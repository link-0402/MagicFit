# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty

from . import operators, rig


def _poll_armature(_self, obj):
    return obj.type == 'ARMATURE' and not rig.is_rig(obj)


def _bust_changed(_self, context):
    operators.bust_changed(context)


class CustomizePlusSettings(bpy.types.PropertyGroup):
    scope: EnumProperty(
        name="Armatures",
        description="Which armatures show the template",
        items=(
            ('ALL', "All Armatures",
             "Every armature with bones the template changes: the body's, and those of separately imported "
             "faces, hair and gear"),
            ('ONE', "One Armature", "Only the armature picked below (or the body's, or the active object's)"),
        ),
        default='ALL',
    )
    armature: PointerProperty(
        name="Armature",
        description="Armature to show the template on. Empty: the body's armature, or the active object's",
        type=bpy.types.Object,
        poll=_poll_armature,
    )
    template: StringProperty(
        name="Template",
        description=(
            "The Customize+ template to apply. Click the template's Copy button in Customize+, then paste "
            "it here"
        ),
    )
    use_scale: BoolProperty(
        name="Scale",
        description="Apply the template's scaling",
        default=True,
    )
    use_rotation: BoolProperty(
        name="Rotation",
        description="Apply the template's rotations",
        default=True,
    )
    use_translation: BoolProperty(
        name="Position",
        description="Apply the template's position offsets",
        default=True,
    )
    use_bust: BoolProperty(
        name="Bust Size",
        description=(
            "Show the Bust Size from character creation, like the game does. It scales the breast bones "
            "and all bones below them"
        ),
        default=False,
        update=_bust_changed,
    )
    bust_size: IntProperty(
        name="Bust Size",
        description=(
            "The character's Bust Size slider (0 to 100). 50 keeps the model's size; Lalafells never go "
            "smaller than that. Male characters have none"
        ),
        min=0,
        max=100,
        default=50,
        update=_bust_changed,
    )


classes = (CustomizePlusSettings,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.magic_fit_cplus = PointerProperty(type=CustomizePlusSettings)


def unregister():
    del bpy.types.Scene.magic_fit_cplus
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
