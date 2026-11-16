# SPDX-License-Identifier: GPL-3.0-or-later

"""The add-on's preferences: its hotkeys, which can be changed there, the game folder Face reads the
game's faces from, and the Second Life body Line Up places Second Life pieces on."""

import bpy
from bpy.props import StringProperty

from . import facedata, facing, gamefiles, secondlife

# Keys in the 3D Viewport, in every mode: (operator, label, default key, optionally the operator properties the
# key sets). Alt Shift C isn't used by Blender's default keymap anywhere. Delete Hair Chain passes the key on
# unless the mouse is over the sidebar's Hair sub-tab, so Delete still deletes everywhere else.
KEYMAP = "3D View Generic"
HOTKEYS = (
    ("magic_fit.cplus_toggle", "Toggle Customize+", {"type": 'C', "value": 'PRESS', "alt": True, "shift": True}),
    ("magic_fit.delete_hair_chain", "Delete Hair Chain", {"type": 'DEL', "value": 'PRESS'}, {"key": True}),
)

_registered = []  # (keymap, item) added to the add-on keyconfig


def _draw_hotkey(layout, keymap, item, label):
    """A keymap item like Blender's keymap editor draws it, but for keyboard keys only: no Map Type."""
    layout.context_pointer_set("keymap", keymap)  # for Restore
    split = layout.split()
    row = split.row(align=True)
    row.prop(item, "active", text="", emboss=False)
    row.label(text=label)
    row = split.row(align=True)
    row.alert = item.map_type != 'KEYBOARD'
    row.prop(item, "type", text="", full_event=True)
    if item.is_user_modified:
        row.operator("preferences.keyitem_restore", text="", icon='BACK').item_id = item.id
    if item.map_type != 'KEYBOARD':
        layout.label(text="Click the key and press one on the keyboard", icon='ERROR')


def _game_folder_changed(_self, _context):
    facing.Reference.forget()


def _draw_game(layout, prefs):
    """The game folder, and where the game was found."""
    col = layout.column()
    col.label(text="Face reads the game's faces from your FFXIV install (once, and again after game updates):")
    col.prop(prefs, "game_folder")
    root, setting = facedata.game_folder()
    if root is None:
        col.label(text="No game {:s}: set the folder the game is installed in (with game and boot in it)".format(
            "there" if setting else "found"), icon='ERROR')
        return
    col.label(text="{:s} {:s} (game version {:s})".format(
        "The game is in" if setting else "Found the game in", root, gamefiles.game_version(root) or "unknown"),
        icon='CHECKMARK')
    row = col.row()
    row.alignment = 'LEFT'
    row.operator("magic_fit.read_game_faces", icon='FILE_REFRESH')


def _second_life_body_changed(_self, _context):
    secondlife.describe(secondlife.body_setting())


def _draw_second_life(layout, prefs):
    """The Second Life body file, and what Line Up found in it."""
    col = layout.column()
    col.label(text="Line Up places Second Life pieces (Firestorm exports) on this body first:")
    col.prop(prefs, "second_life_body")
    path = secondlife.body_setting()
    if path:
        text = secondlife.described(path)
        if text is None:
            col.label(text="Not read yet: Line Up reads it when it lines up Second Life pieces", icon='INFO')
        else:
            col.label(text=text, icon='CHECKMARK' if text.startswith("Body ") else 'ERROR')


class MagicFitPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    game_folder: StringProperty(
        name="Game Folder",
        description="The folder FFXIV is installed in (with game and boot in it). Leave empty to find it "
                    "through XIVLauncher, the game's installer or Steam",
        subtype='DIR_PATH',
        update=_game_folder_changed,
    )
    second_life_body: StringProperty(
        name="Second Life Body",
        description="The .blend with the Second Life body that the clothes are made for (an eBody devkit). "
                    "Line Up places Second Life pieces on it before it lines them up",
        subtype='FILE_PATH',
        update=_second_life_body_changed,
    )

    def draw(self, context):
        layout = self.layout
        _draw_game(layout, self)
        layout.separator()
        _draw_second_life(layout, self)
        layout.separator()
        col = layout.column()
        col.label(text="Hotkeys (3D Viewport, in every mode):")
        keyconfig = context.window_manager.keyconfigs.user
        keymap = keyconfig.keymaps.get(KEYMAP) if keyconfig is not None else None
        for idname, label, *_key in HOTKEYS:
            item = None
            if keymap is not None:
                item = next((item for item in keymap.keymap_items if item.idname == idname), None)
            if item is None:
                col.label(text="{:s}: removed (restore it under Keymap, 3D View > 3D View Generic)".format(label))
                continue
            _draw_hotkey(col, keymap, item, label)


def register():
    bpy.utils.register_class(MagicFitPreferences)
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is None:  # background mode
        return
    keymap = keyconfig.keymaps.new(name=KEYMAP, space_type='VIEW_3D')
    for idname, _label, key, *properties in HOTKEYS:
        item = keymap.keymap_items.new(idname, **key)
        for name, value in (properties[0] if properties else {}).items():
            setattr(item.properties, name, value)
        _registered.append((keymap, item))


def unregister():
    for keymap, item in _registered:
        try:
            keymap.keymap_items.remove(item)
        except (ReferenceError, RuntimeError):
            pass
    _registered.clear()
    bpy.utils.unregister_class(MagicFitPreferences)
