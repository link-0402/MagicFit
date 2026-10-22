# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless tests for the chain view of the hair's own skeleton (Own Skeleton): its bones coloured by their
chain's kind, and the Show Chains toggle drawing them in front.

Run from the repository root:
    blender -b --factory-startup --python tests/test_hair_view.py
"""

import os
import shutil
import sys
import tempfile
import traceback

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import hair  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import hairing, hairrig, tool  # noqa: E402

try:
    from magic_fit import hairview  # noqa: E402
except ImportError:
    hairview = None

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


class Recorder:
    """Stands in for a UILayout: keeps what is drawn on it, as (call, args, keyword args)."""

    def __init__(self, drawn):
        object.__setattr__(self, "drawn", drawn)

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.drawn.append((name, args, kwargs))
            return Recorder(self.drawn)
        return call


def rig_scene():
    """The long hair (`hair.build`), active and selected, weighted by Hair Weights in Own Skeleton for c0201:
    (hair object, its armature)."""
    hair.clear_scene()
    obj, _parts = hair.build()
    settings = bpy.context.scene.magic_fit
    for key, value in dict(hair_race=hair.RACE, hair_skeleton_mode='CUSTOM', hair_skeleton=0,
                           hair_apply_to_selected=False, group_mode='ALL', mode='HAIR').items():
        setattr(settings, key, value)
    result = bpy.ops.magic_fit.hair_weights()
    armature = hairing.weighted_armature(obj)
    check(result == {'FINISHED'} and obj.magic_fit_hair.custom and armature is not None
          and len(armature.magic_fit_hair_rig.chains) > 0, "scene: the hair got its own chains")
    return obj, armature


def chain_palettes(armature):
    """(kind, the palettes of its bones) of each chain in the armature's rig state."""
    bones = armature.data.bones
    return [(chain.kind, [bones[name].color.palette for name in chain.bones.split(",") if name])
            for chain in armature.magic_fit_hair_rig.chains]


def coloured_by_kind(armature):
    return all(palettes and all(palette == hairview.KIND_PALETTES[kind] for palette in palettes)
               for kind, palettes in chain_palettes(armature))


def test_palettes():
    check(sorted(hairview.KIND_PALETTES) == sorted(hairrig.KINDS), "palettes: one per kind")
    check(len(set(hairview.KIND_PALETTES.values())) == len(hairrig.KINDS), "palettes: a different one per kind")
    sets = bpy.context.preferences.themes[0].bone_color_sets
    for kind, palette in hairview.KIND_PALETTES.items():
        expected = tuple(sets[int(palette[len("THEME"):]) - 1].normal)
        rgb = hairview.kind_rgb(kind)
        check(len(rgb) == 3 and max(abs(a - b) for a, b in zip(rgb, expected)) < 1e-6,
              "palettes: {:s}'s colour is its theme set's normal colour ({})".format(kind, rgb))


def test_bones_colored_by_kind():
    _obj, armature = rig_scene()
    print("      chains:", [kind for kind, _palettes in chain_palettes(armature)])
    check(coloured_by_kind(armature), "colours: every chain's bones have its kind's palette after the build")
    check(armature.data.bones[hairing.HEAD].color.palette == 'DEFAULT', "colours: j_kao keeps its colour")
    chain = armature.magic_fit_hair_rig.chains[0]
    kind = next(k for k in hairrig.KINDS if k != chain.kind)
    chain.kind = kind
    check(chain_palettes(armature)[0] == (kind, [hairview.KIND_PALETTES[kind]] * len(chain_palettes(armature)[0][1]))
          and coloured_by_kind(armature), "colours: a kind change recolours that chain's bones ({:s})".format(kind))
    hairrig.sync_rig(bpy.context, armature, hair.RACE)
    check(armature.magic_fit_hair_rig.chains[0].kind == kind and coloured_by_kind(armature),
          "colours: sync_rig keeps them")
    # A kind changed while the armature is in Edit Mode: the edit bones take it, and keep it after.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    other = next(k for k in hairrig.KINDS if k != kind)
    chain = armature.magic_fit_hair_rig.chains[0]
    chain.kind = other
    first = chain.bones.split(",")[0]
    check(armature.data.edit_bones[first].color.palette == hairview.KIND_PALETTES[other],
          "colours: in Edit Mode the edit bones recolour")
    bpy.ops.object.mode_set(mode='OBJECT')
    check(coloured_by_kind(armature), "colours: ...and keep it out of Edit Mode")


def test_current_armature():
    obj, armature = rig_scene()
    layer = bpy.context.view_layer
    check(hairview.current_armature(bpy.context) == armature, "current: the active hair's armature")
    layer.objects.active = armature
    check(hairview.current_armature(bpy.context) == armature, "current: the active armature with chains")
    obj.magic_fit_hair.custom = False
    layer.objects.active = obj
    check(hairview.current_armature(bpy.context) is None, "current: none for hair without its own skeleton")
    obj.magic_fit_hair.custom = True
    armature.magic_fit_hair_rig.chains.clear()
    layer.objects.active = armature
    check(hairview.current_armature(bpy.context) is None, "current: none for an armature without chains")


def test_toggle_in_front():
    _obj, armature = rig_scene()
    wm = bpy.context.window_manager
    armature.show_in_front = False
    wm.magic_fit_show_chains = True
    check(armature.show_in_front, "in front: on draws the armature in front")
    wm.magic_fit_show_chains = True
    wm.magic_fit_show_chains = False
    check(not armature.show_in_front, "in front: off restores it, also after on twice")
    armature.show_in_front = True
    wm.magic_fit_show_chains = True
    wm.magic_fit_show_chains = False
    check(armature.show_in_front, "in front: an armature in front already stays so")
    armature.show_in_front = False


def test_toggle_drawn():
    obj, armature = rig_scene()
    wm = bpy.context.window_manager
    for on, icon in ((False, 'HIDE_ON'), (True, 'HIDE_OFF')):
        wm.magic_fit_show_chains = on
        drawn = []
        tool.VIEW3D_PT_hair_weights_sidebar.draw_chains(Recorder(drawn), obj)
        toggles = [kwargs for call, args, kwargs in drawn
                   if call == 'prop' and len(args) > 1 and args[0] == wm and args[1] == "magic_fit_show_chains"]
        check(len(toggles) == 1 and toggles[0].get("icon") == icon and toggles[0].get("text") == "",
              "toggle: drawn as an eye with no text, {:s} while {:s} ({})".format(icon, "on" if on else "off", toggles))
        texts = [kwargs.get("text") for _call, _args, kwargs in drawn]
        count = hairrig.counted(len(armature.magic_fit_hair_rig.chains), "chain")
        check(any(text and text.startswith(count + ", ") for text in texts), "toggle: the box still counts the chains")
    wm.magic_fit_show_chains = False


def test_save_keeps_in_front_off():
    _obj, armature = rig_scene()
    name = armature.name
    wm = bpy.context.window_manager
    armature.show_in_front = False
    wm.magic_fit_show_chains = True
    folder = tempfile.mkdtemp(prefix="magic_fit_hair_view_")
    try:
        path = os.path.join(folder, "chains.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path)
        check(bpy.data.objects[name].show_in_front, "save: still in front after saving")
        bpy.ops.wm.open_mainfile(filepath=path)
        check(not bpy.data.objects[name].show_in_front, "save: not in front in the file")
        check(not bpy.context.window_manager.magic_fit_show_chains, "save: Show Chains is off after opening")
        # What was kept before opening (not in front) is forgotten: off restores what the file has.
        bpy.data.objects[name].show_in_front = True
        bpy.context.window_manager.magic_fit_show_chains = True
        bpy.context.window_manager.magic_fit_show_chains = False
        check(bpy.data.objects[name].show_in_front, "save: nothing kept from before opening")
        bpy.data.objects[name].show_in_front = False
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main():
    magic_fit.register()
    try:
        tests = (test_palettes, test_bones_colored_by_kind, test_current_armature, test_toggle_in_front,
                 test_toggle_drawn, test_save_keeps_in_front_off)
        if hairview is None:
            check(False, "magic_fit.hairview exists")
            tests = ()
        for test in tests:
            print("==", test.__name__)
            try:
                test()
            except Exception:
                traceback.print_exc()
                FAILURES.append(test.__name__ + ": exception")
    finally:
        magic_fit.unregister()
    print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
    sys.stdout.flush()
    os._exit(1 if FAILURES else 0)


main()
