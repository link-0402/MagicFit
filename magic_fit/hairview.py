# SPDX-License-Identifier: GPL-3.0-or-later

"""Chain view: shows the chains of the hair's own skeleton (Own Skeleton, `hairrig`) in the 3D view.

- Bone colours: every bone of a chain gets Blender's bone colour (`Bone.color.palette`, a theme palette) for
  the chain's kind, one per kind (KIND_PALETTES), when the chains are built or read back (`hairrig._fill_state`)
  and when a chain's kind changes in the list. They are saved with the file; exporters ignore them.
- Show Chains (`WindowManager.magic_fit_show_chains`, runtime: off after a file opens): while it's on, the
  hair's armature (`current_armature`) is drawn In Front. What In Front was before is kept by armature, at
  runtime, and comes back when the toggle goes off. Files are saved with it as it was before.
"""

import bpy
from bpy.app.handlers import persistent

from . import hairing, hairrig

# The bone colour set (Preferences > Themes > Bone Color Sets) of each chain kind's bones. In the default theme
# THEME09 is yellow, THEME03 green, THEME04 blue and THEME06 purple.
KIND_PALETTES = {
    "bangs": 'THEME09',
    "side": 'THEME03',
    "back": 'THEME04',
    "ponytail": 'THEME06',
}
# The normal colour of each of those sets in the default theme, for when there is no theme to read.
_DEFAULT_RGB = {
    'THEME09': (0.957, 0.788, 0.047),
    'THEME03': (0.118, 0.569, 0.035),
    'THEME04': (0.039, 0.212, 0.580),
    'THEME06': (0.263, 0.047, 0.471),
}

# {armature object name: its In Front before Show Chains turned it on}
_in_front = {}


def kind_rgb(kind):
    """The colour of chain ``kind``'s bones: its set's normal colour in the theme (KIND_PALETTES), or the default
    theme's when there is no theme."""
    palette = KIND_PALETTES.get(kind, KIND_PALETTES["back"])
    try:
        sets = bpy.context.preferences.themes[0].bone_color_sets
        return tuple(sets[int(palette[len("THEME"):]) - 1].normal)
    except (AttributeError, IndexError):
        return _DEFAULT_RGB[palette]


def color_bones(armature_object):
    """Give every bone of the chains in ``armature_object``'s rig state its chain's kind's bone colour
    (KIND_PALETTES), on its edit bones too while the armature is in Edit Mode (they replace the bones when it
    leaves Edit Mode)."""
    data = armature_object.data
    collections = [data.bones]
    if armature_object.mode == 'EDIT':
        collections.append(data.edit_bones)
    for chain in armature_object.magic_fit_hair_rig.chains:
        palette = KIND_PALETTES.get(chain.kind)
        if palette is None:
            continue
        for name in chain.bones.split(","):
            for bones in collections:
                bone = bones.get(name) if name else None
                if bone is not None and bone.color.palette != palette:
                    bone.color.palette = palette


def current_armature(context):
    """The armature whose chains are shown: the active hair's (a mesh with its own skeleton,
    `hairing.weighted_armature`), or the active armature when it has chains; else None."""
    obj = context.view_layer.objects.active if context.view_layer is not None else None
    if obj is None:
        return None
    if obj.type == 'MESH':
        return hairing.weighted_armature(obj) if obj.magic_fit_hair.custom else None
    if obj.type == 'ARMATURE' and len(obj.magic_fit_hair_rig.chains):
        return obj
    return None


def _draw_in_front(armature_object):
    """Draw ``armature_object`` In Front, keeping what it was the first time."""
    if armature_object.name not in _in_front:
        _in_front[armature_object.name] = armature_object.show_in_front
    armature_object.show_in_front = True


def _restore_in_front():
    """Give the armatures Show Chains drew In Front their In Front back."""
    for name, value in _in_front.items():
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == 'ARMATURE':
            obj.show_in_front = value


def show_chains_changed(wm, context):
    """Property update callback of Show Chains: on, the hair's armature is drawn In Front; off, every armature
    it drew In Front gets its In Front back."""
    if wm.magic_fit_show_chains:
        armature = current_armature(context)
        if armature is not None:
            _draw_in_front(armature)
    else:
        _restore_in_front()
        _in_front.clear()


# -----------------------------------------------------------------------------
# Files

@persistent
def _on_save_pre(_dummy=None):
    # The file keeps each armature's In Front as it was before Show Chains.
    _restore_in_front()


@persistent
def _on_save_post(_dummy=None):
    for name in _in_front:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == 'ARMATURE':
            obj.show_in_front = True


@persistent
def _on_load(_dummy=None):
    # The armatures of the file before are gone; the toggle is off in a file just opened.
    _in_front.clear()
    wm = bpy.context.window_manager
    if wm is not None and getattr(wm, "magic_fit_show_chains", False):
        wm.magic_fit_show_chains = False


def register():
    bpy.app.handlers.save_pre.append(_on_save_pre)
    bpy.app.handlers.save_post.append(_on_save_post)
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    for handlers, callback in (
            (bpy.app.handlers.save_pre, _on_save_pre),
            (bpy.app.handlers.save_post, _on_save_post),
            (bpy.app.handlers.load_post, _on_load),
    ):
        if callback in handlers:
            handlers.remove(callback)
    _restore_in_front()
    _in_front.clear()
