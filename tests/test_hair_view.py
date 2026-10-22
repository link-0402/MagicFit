# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless tests for the chain view of the hair's own skeleton (Own Skeleton): its bones coloured by their
chain's kind, the Show Chains toggle drawing them in front, and the overlay tinting the hair by its chains
(its colours, weights and rebuilds; batches can't be drawn headless).

Run from the repository root:
    blender -b --factory-startup --python tests/test_hair_view.py
"""

import os
import shutil
import sys
import tempfile
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import hair  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import hairing, hairrig, painting, tool  # noqa: E402

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
        # What off restored is forgotten when a file opens too: undo doesn't bring it back.
        wm = bpy.context.window_manager
        wm.magic_fit_show_chains = True
        wm.magic_fit_show_chains = False
        bpy.ops.wm.open_mainfile(filepath=path)
        check(not hairview._restored and not hairview._in_front and not hairview._tints,
              "save: nothing kept after opening a file")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def linked_copy(obj, name):
    """A copy of ``obj`` (its data shared, its modifiers and properties copied), in the scene."""
    copy = obj.copy()
    copy.name = name
    bpy.context.scene.collection.objects.link(copy)
    return copy


def armature_modifier(obj):
    return next(modifier for modifier in obj.modifiers if modifier.type == 'ARMATURE')


def test_renamed_armature_restored():
    _obj, armature = rig_scene()
    wm = bpy.context.window_manager
    armature.show_in_front = False
    wm.magic_fit_show_chains = True
    armature.name = "Renamed Skeleton"
    wm.magic_fit_show_chains = False
    check(not armature.show_in_front, "rename: an armature renamed while shown gets its In Front back")


def test_in_front_new_current():
    obj, armature = rig_scene()
    wm = bpy.context.window_manager
    layer = bpy.context.view_layer
    armature.show_in_front = False
    other = linked_copy(armature, "Other Skeleton")
    other.show_in_front = False
    wm.magic_fit_show_chains = True
    layer.objects.active = other
    check(hairview.current_armature(bpy.context) == other, "new current: the other armature is current")
    # What the overlay's draw does: it never writes the armature itself, a timer does.
    hairview.follow_current(bpy.context)
    check(not other.show_in_front, "new current: not drawn in front from the draw callback")
    check(bpy.app.timers.is_registered(hairview._apply_pending_in_front), "new current: a timer is registered")
    hairview._apply_pending_in_front()
    if bpy.app.timers.is_registered(hairview._apply_pending_in_front):
        bpy.app.timers.unregister(hairview._apply_pending_in_front)
    check(other.show_in_front and armature.show_in_front, "new current: the timer draws it in front")
    hairview.follow_current(bpy.context)
    check(not bpy.app.timers.is_registered(hairview._apply_pending_in_front),
          "new current: no timer for an armature already in front")
    wm.magic_fit_show_chains = False
    check(not other.show_in_front and not armature.show_in_front, "new current: off restores both")
    layer.objects.active = obj


def test_undo_redo_in_front():
    _obj, armature = rig_scene()
    wm = bpy.context.window_manager
    check(hairview._on_undo_redo in bpy.app.handlers.undo_post
          and hairview._on_undo_redo in bpy.app.handlers.redo_post, "undo: handlers registered")
    armature.show_in_front = False
    wm.magic_fit_show_chains = True
    wm.magic_fit_show_chains = False
    # Ctrl+Z back to a step taken while Show Chains was on brings In Front back: the handler undoes that.
    armature.show_in_front = True
    hairview._on_undo_redo()
    check(not armature.show_in_front, "undo: with the toggle off, what off restored is applied again")
    armature.show_in_front = True
    hairview._on_undo_redo()
    check(not armature.show_in_front, "undo: ...again on redo")
    wm.magic_fit_show_chains = True
    check(not hairview._restored, "undo: on forgets what off restored")
    # Undo past the toggle going on.
    armature.show_in_front = False
    hairview._on_undo_redo()
    check(armature.show_in_front, "undo: with the toggle on, the shown armatures stay in front")
    wm.magic_fit_show_chains = False
    check(not armature.show_in_front, "undo: off still restores what was before")


def test_unregister_turns_off():
    _obj, armature = rig_scene()
    wm = bpy.context.window_manager
    armature.show_in_front = False
    wm.magic_fit_show_chains = True
    magic_fit.unregister()
    try:
        check(not armature.show_in_front, "disable: the armature gets its In Front back")
        check(hairview._handle is None and hairview._on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post
              and hairview._on_undo_redo not in bpy.app.handlers.undo_post, "disable: handlers removed")
    finally:
        magic_fit.register()
    check(not bpy.context.window_manager.magic_fit_show_chains, "disable: Show Chains is off after enabling again")
    check(hairview._handle is not None and hairview._on_depsgraph_update in bpy.app.handlers.depsgraph_update_post
          and hairview._on_depsgraph_update in bpy.app.handlers.frame_change_post, "enable: handlers registered again")


def test_chain_colors():
    c = hairview.chain_colors(np.array([[1, 0], [0, 1], [0, 0]]), ["back", "back"], active=0)
    check(c.shape == (3, 4) and c.dtype == np.float32, "tint colours: (n, 4) float32")
    check(np.isclose(c[0, 3], 0.6) and np.isclose(c[1, 3], 0.6 * 0.35) and c[2, 3] == 0,
          "tint colours: alpha by weight, the active chain at full strength, others faded")
    check(np.allclose(c[0, :3], hairview.kind_rgb("back")), "tint colours: a kind's first chain has its colour")
    check(np.allclose(c[1, :3], np.array(hairview.kind_rgb("back")) * 0.6),
          "tint colours: second chain of a kind darker")
    check(hairview.TINT_ALPHA == 0.6 and hairview.FADE == 0.35 and hairview.SHADE_DARK == 0.6
          and hairview.NUDGE == 0.0005 and hairview.REBUILD_INTERVAL == 0.25, "tint colours: constants")
    bangs, side = np.array(hairview.kind_rgb("bangs")), np.array(hairview.kind_rgb("side"))
    weights = np.array([[0.25, 0.25, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]])
    c = hairview.chain_colors(weights, ["bangs", "side", "bangs"], active=5)
    check(np.allclose(c[0, :3], (bangs + side) / 2) and np.allclose(c[1, :3], bangs * (1 + 0.6) / 2),
          "tint colours: a vertex's colour is the weighted mean of its chains'")
    check(np.isclose(c[0, 3], 0.3) and np.isclose(c[1, 3], 0.6) and c[2, 3] == 0,
          "tint colours: no chain picked, all at full strength")
    c = hairview.chain_colors(weights, ["bangs", "side", "bangs"], active=1)
    check(np.isclose(c[0, 3], 0.6 * (0.25 * 0.35 + 0.25)) and np.isclose(c[1, 3], 0.6 * 0.35),
          "tint colours: faded and full mixed by weight")
    c = hairview.chain_colors(np.array([[1.0]]), ["back"], 0, rgb=lambda _kind: (1.0, 0.0, 0.0))
    check(np.allclose(c, [[1.0, 0.0, 0.0, 0.6]]), "tint colours: rgb comes from the given function")
    c = hairview.chain_colors(np.zeros((4, 0)), [], 0)
    check(c.shape == (4, 4) and not c.any(), "tint colours: no chains, no tint")


def test_chain_weights():
    obj, armature = rig_scene()
    rig = armature.magic_fit_hair_rig
    weights = hairview.chain_weights(obj, rig)
    count = len(obj.data.vertices)
    check(weights.shape == (count, len(rig.chains)), "chain weights: (vertices, chains) {}".format(weights.shape))
    names = {group.index: group.name for group in obj.vertex_groups}
    clear = {hairing.HEAD, hairing.NECK, hairing.UPPER_BACK, *hairing.EARS}
    hair_weight = np.zeros(count)
    for vert in obj.data.vertices:
        for elem in vert.groups:
            if names[elem.group] not in clear:
                hair_weight[vert.index] += elem.weight
    total = weights.sum(axis=1)
    check(np.allclose(total, hair_weight), "chain weights: rows sum to the weight off j_kao, the neck, back and ears")
    check((total < 1e-6).sum() > 0 and (total > 0.99).sum() > 0,
          "chain weights: the cap has none, hanging hair all ({:d} clear, {:d} full)".format(
              int((total < 1e-6).sum()), int((total > 0.99).sum())))
    group = obj.vertex_groups[rig.chains[0].bones.split(",")[0]]
    first = np.zeros(count)
    for vert in obj.data.vertices:
        for elem in vert.groups:
            if elem.group == group.index:
                first[vert.index] = elem.weight
    check(np.all(weights[:, 0] >= first - 1e-9) and weights[:, 0].sum() > first.sum(),
          "chain weights: a chain sums all its bones")
    colors = hairview.chain_colors(weights, [chain.kind for chain in rig.chains], rig.active)
    check(not colors[total < 1e-6, 3].any() and colors[total > 0.99, 3].min() > 0.0,
          "chain weights: head weights give no tint")


def test_chain_weights_edge_cases():
    obj, armature = rig_scene()
    rig = armature.magic_fit_hair_rig
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.active_object
    cube.vertex_groups.new(name=hairing.HEAD).add(list(range(8)), 1.0, 'REPLACE')
    weights = hairview.chain_weights(cube, rig)
    check(weights.shape == (8, len(rig.chains)) and not weights.any(), "chain weights: no hair groups, zeros")
    cube.vertex_groups.clear()
    check(not hairview.chain_weights(cube, rig).any(), "chain weights: no groups at all, zeros")
    empty = bpy.data.objects.new("No Chains", bpy.data.armatures.new("No Chains"))
    check(hairview.chain_weights(obj, empty.magic_fit_hair_rig).shape == (len(obj.data.vertices), 0),
          "chain weights: no chains, no columns")
    bpy.data.objects.remove(cube)
    layer = bpy.context.view_layer
    layer.objects.active = obj
    check(hairview.overlay_meshes(bpy.context, armature) == [obj], "overlay meshes: the hair")
    subdivided = obj.modifiers.new("Subdivision", 'SUBSURF')
    check(hairview.overlay_meshes(bpy.context, armature) == [], "overlay meshes: not with other vertices displayed")
    obj.modifiers.remove(subdivided)
    check(hairview.overlay_meshes(bpy.context, armature) == [obj], "overlay meshes: ...back without them")


def test_overlay_meshes():
    obj, armature = rig_scene()
    layer = bpy.context.view_layer
    second = linked_copy(obj, "Second Hair")
    second.data = obj.data.copy()
    twin = linked_copy(obj, "Twin Hair")   # shares the hair's mesh
    plain = linked_copy(obj, "Plain Hair")
    plain.magic_fit_hair.custom = False
    other_armature = linked_copy(armature, "Other Skeleton")
    elsewhere = linked_copy(obj, "Other Hair")
    armature_modifier(elsewhere).object = other_armature
    hidden = linked_copy(obj, "Hidden Hair")
    hidden.hide_set(True)
    meshes = hairview.overlay_meshes(bpy.context, armature)
    check(sorted(m.name for m in meshes) == sorted((obj.name, second.name, twin.name)),
          "overlay meshes: the armature's meshes with their own skeleton, shown ({})".format([m.name for m in meshes]))
    check([m.name for m in hairview.overlay_meshes(bpy.context, other_armature)] == [elsewhere.name],
          "overlay meshes: the other armature's")
    for each in bpy.context.selected_objects:
        each.select_set(False)
    obj.select_set(True)
    layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    meshes = hairview.overlay_meshes(bpy.context, armature)
    bpy.ops.object.mode_set(mode='OBJECT')
    check([m.name for m in meshes] == [second.name],
          "overlay meshes: none in Edit Mode, nor one sharing its mesh ({})".format([m.name for m in meshes]))


def test_overlay_rebuilds():
    obj, armature = rig_scene()
    context = bpy.context
    wm = context.window_manager
    rig = armature.magic_fit_hair_rig
    wm.magic_fit_show_chains = True
    try:
        hairview._draw()  # no 3D view headless: draws nothing, raises nothing
        meshes = hairview.overlay_meshes(context, armature)
        put_off = hairview._refresh(context, armature, meshes)
        tint = hairview._layer_tints(context).get(obj.session_uid)
        check(not put_off and tint is not None, "overlay: the hair's tint is built")
        count = len(obj.data.vertices)
        positions, colors, indices = hairview.tint_arrays(tint)
        geometry = painting.read_geometry(obj, context.evaluated_depsgraph_get(), source='DISPLAYED')
        offsets = np.linalg.norm(positions[:count] - geometry.co, axis=1)
        check(positions.shape == (2 * count, 3) and colors.shape == (2 * count, 4)
              and offsets.max() < hairview.NUDGE * 1.001 and offsets.max() > hairview.NUDGE * 0.999
              and np.allclose(positions[count:] - geometry.co, geometry.co - positions[:count], atol=1e-6),
              "overlay: displayed positions nudged both ways along the normals")
        total = hairview.chain_weights(obj, rig).sum(axis=1)
        check(np.allclose(colors[:count], colors[count:]) and not colors[:count][total < 1e-6, 3].any()
              and colors[:count][total > 0.99, 3].min() > 0, "overlay: the cap untinted, the hanging hair tinted")
        tinted = colors[:, 3] > 0
        check(len(indices) and tinted[indices].any(axis=1).all() and indices.max() < 2 * count,
              "overlay: only triangles with a tinted corner, on both copies")

        # A pose change re-reads the positions, but at most every REBUILD_INTERVAL.
        before = tint.co.copy()
        bone = armature.pose.bones[rig.chains[0].bones.split(",")[0]]
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler.x = 0.4
        context.view_layer.update()
        check(tint.moved and not tint.reweigh, "overlay: a pose change marks positions, not weights")
        put_off = hairview._refresh(context, armature, meshes)
        check(put_off and np.array_equal(tint.co, before) and bpy.app.timers.is_registered(hairview._tag_views),
              "overlay: a rebuild right after the last is put off, a timer redraws later")
        tint.built -= hairview.REBUILD_INTERVAL
        put_off = hairview._refresh(context, armature, meshes)
        check(not put_off and not tint.moved and not np.array_equal(tint.co, before), "overlay: ...and done then")

        # A weight change re-reads the weights.
        weights_before = tint.weights.copy()
        group = obj.vertex_groups[rig.chains[0].bones.split(",")[0]]
        group.add(list(range(count)), 0.0, 'REPLACE')
        obj.data.update()
        context.view_layer.update()
        check(tint.reweigh, "overlay: a weight change marks the weights")
        tint.built -= hairview.REBUILD_INTERVAL
        hairview._refresh(context, armature, meshes)
        check(not tint.reweigh and not np.array_equal(tint.weights, weights_before), "overlay: weights read again")

        # Picking another chain recolours at once, without reading the mesh again.
        positions_read = tint.co
        colors_before = tint.colors.copy()
        rig.active = 1 if rig.active != 1 else 0
        put_off = hairview._refresh(context, armature, meshes)
        check(not put_off and tint.co is positions_read and not np.array_equal(tint.colors, colors_before),
              "overlay: another chain picked recolours at once")
        # A chain's kind too.
        colors_before = tint.colors.copy()
        rig.chains[0].kind = next(kind for kind in hairrig.KINDS if kind != rig.chains[0].kind)
        hairview._refresh(context, armature, meshes)
        check(not np.array_equal(tint.colors, colors_before), "overlay: a kind change recolours")
        hairview._refresh(context, armature, [])
        check(not hairview._layer_tints(context), "overlay: meshes no longer shown are forgotten")
        wm.magic_fit_show_chains = False
        check(not hairview._tints, "overlay: off forgets every tint")
    finally:
        if bpy.app.timers.is_registered(hairview._tag_views):
            bpy.app.timers.unregister(hairview._tag_views)
        wm.magic_fit_show_chains = False
        hairview.clear()


def main():
    magic_fit.register()
    try:
        tests = (test_palettes, test_bones_colored_by_kind, test_current_armature, test_toggle_in_front,
                 test_toggle_drawn, test_save_keeps_in_front_off, test_renamed_armature_restored,
                 test_in_front_new_current, test_undo_redo_in_front, test_unregister_turns_off, test_chain_colors,
                 test_chain_weights, test_chain_weights_edge_cases, test_overlay_meshes, test_overlay_rebuilds)
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
