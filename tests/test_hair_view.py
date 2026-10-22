# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless tests for the chain view of the hair's own skeleton (Own Skeleton): its bones coloured by their
chain's kind, the Show Chains toggle drawing them in front, and the overlay tinting the hair by its chains and
drawing the chain bones as lines (their colours, weights, positions and rebuilds, the X-ray skip and the GPU
state put back; batches can't be drawn headless).

Run from the repository root:
    blender -b --factory-startup --python tests/test_hair_view.py
"""

import os
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Vector

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
    check(weights.shape == looped_chain_weights(obj, rig).shape
          and np.allclose(weights, looped_chain_weights(obj, rig), atol=1e-6),
          "chain weights: the same as a vertex by vertex sum")
    # Some weights painted off and others on (in groups of other chains and of no chain).
    group.add(list(range(0, count, 3)), 0.0, 'REPLACE')
    other = obj.vertex_groups[rig.chains[-1].bones.split(",")[-1]]
    other.add(list(range(1, count, 5)), 0.4, 'ADD')
    obj.vertex_groups[hairing.HEAD].add(list(range(2, count, 7)), 0.3, 'REPLACE')
    check(np.allclose(hairview.chain_weights(obj, rig), looped_chain_weights(obj, rig), atol=1e-6),
          "chain weights: ...also after weights change")


def looped_chain_weights(obj, rig):
    """`hairview.chain_weights` as it was first written: vertex by vertex, group by group."""
    column = {}
    for c, chain in enumerate(rig.chains):
        for name in chain.bones.split(","):
            group = obj.vertex_groups.get(name) if name else None
            if group is not None:
                column[group.index] = c
    weights = np.zeros((len(obj.data.vertices), len(rig.chains)))
    for vert in obj.data.vertices:
        for elem in vert.groups:
            c = column.get(elem.group)
            if c is not None:
                weights[vert.index, c] += elem.weight
    return weights


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
    bpy.ops.mesh.primitive_cube_add()
    bare = bpy.context.active_object
    check(hairview.chain_weights(bare, rig).shape == (8, len(rig.chains))
          and not hairview.chain_weights(bare, rig).any(), "chain weights: a mesh never weighted, zeros")
    bpy.data.objects.remove(bare)
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
        tint.weighed -= 10.0   # the weights have their own clock (WEIGH_WAIT)
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


def expected_segments(armature, rig, bones=None):
    """(chain, head, tail) in world space of every chain bone there is, in chain and bone order: from the posed
    bones, or from ``bones`` ({name: (head, tail)} in armature space)."""
    matrix = armature.matrix_world
    result = []
    for c, chain in enumerate(rig.chains):
        for name in chain.bones.split(","):
            if bones is not None:
                if name in bones:
                    result.append((c, matrix @ Vector(bones[name][0]), matrix @ Vector(bones[name][1])))
                continue
            bone = armature.pose.bones.get(name) if name else None
            if bone is not None:
                result.append((c, matrix @ bone.head, matrix @ bone.tail))
    return result


def same_segments(positions, segments):
    if positions.shape != (2 * len(segments), 3) or positions.dtype != np.float32:
        return False
    expected = np.array([[tuple(head), tuple(tail)] for _c, head, tail in segments]).reshape(-1, 3)
    return np.allclose(positions, expected, atol=1e-5)


def test_chain_lines():
    _obj, armature = rig_scene()
    context = bpy.context
    rig = armature.magic_fit_hair_rig
    check(len(rig.chains) >= 3, "lines: the hair has three chains or more")
    # Moved and posed: the lines follow both.
    armature.location = (0.3, -0.2, 0.1)
    armature.rotation_euler.z = 0.5
    bone = armature.pose.bones[rig.chains[0].bones.split(",")[0]]
    bone.rotation_mode = 'XYZ'
    bone.rotation_euler.x = 0.6
    context.view_layer.update()
    positions, colors = hairview.chain_lines(armature, rig)
    segments = expected_segments(armature, rig)
    check(same_segments(positions, segments),
          "lines: a segment from head to tail of every chain bone, posed, in world space ({} for {:d})".format(
              positions.shape, len(segments)))
    check(colors.shape == (len(positions), 4) and colors.dtype == np.float32, "lines: (2k, 4) float32 colours")
    rest = armature.data.bones[bone.name]
    check(not np.allclose(positions[1], np.array(armature.matrix_world @ rest.tail_local), atol=1e-4),
          "lines: the posed bone, not its rest")

    # Colours: each chain's kind's, every second of a kind darker; the picked chain full, the others faded.
    palette = {"bangs": (1.0, 0.0, 0.0), "side": (0.0, 1.0, 0.0), "back": (0.0, 0.0, 1.0), "ponytail": (1.0, 1.0, 0.0)}
    kinds = ["side", "side", "back"] + ["ponytail"] * (len(rig.chains) - 3)
    for chain, kind in zip(rig.chains, kinds):
        chain.kind = kind
    rig.active = 2
    positions, colors = hairview.chain_lines(armature, rig, rgb=lambda kind: palette[kind])
    chain_of = np.repeat([c for c, _head, _tail in expected_segments(armature, rig)], 2)
    seen, expected = {}, []
    for kind in kinds:
        expected.append(np.array(palette[kind]) * (hairview.SHADE_DARK if seen.get(kind, 0) % 2 else 1.0))
        seen[kind] = seen.get(kind, 0) + 1
    check(len(chain_of) == len(colors) and all(np.allclose(colors[k, :3], expected[c]) for k, c in enumerate(chain_of)),
          "lines: each chain in its kind's colour, the second side chain darker")
    check(hairview.LINE_FLOOR == 0.8 and hairview.LINE_FLOOR > hairview.FADE, "lines: LINE_FLOOR 0.8")
    check(np.allclose(colors[chain_of == 2, 3], 1.0) and np.allclose(colors[chain_of != 2, 3], hairview.LINE_FLOOR),
          "lines: the picked chain at full strength, the others at LINE_FLOOR (not the tint's FADE)")
    under = hairview.line_underlay(colors)
    check(under.shape == colors.shape and under.dtype == np.float32 and not under[:, :3].any()
          and np.allclose(under[:, 3], hairview.UNDERLAY_ALPHA * colors[:, 3]) and hairview.UNDERLAY_ALPHA == 0.6,
          "lines: a black underlay, its alpha UNDERLAY_ALPHA times the line's")
    check(hairview.UNDERLAY_WIDTH == 2.0, "lines: the underlay 2 px wider")
    rig.active = 99
    _positions, colors = hairview.chain_lines(armature, rig, rgb=lambda kind: palette[kind])
    check(np.allclose(colors[:, 3], 1.0), "lines: no chain picked, all at full strength")
    check(hairview.chain_lines(armature, rig)[1][:, :3].max() > 0.0, "lines: theme colours by default")

    # A bone listed in a chain but not in the armature is left out.
    count = len(positions)
    rig.chains[1].bones += ",j_ex_no_such_bone"
    positions, _colors = hairview.chain_lines(armature, rig)
    check(len(positions) == count and same_segments(positions, expected_segments(armature, rig)),
          "lines: a bone missing from the armature is skipped")

    # In Edit Mode the edit bones are drawn, as Blender draws them.
    layer = context.view_layer
    for each in context.selected_objects:
        each.select_set(False)
    armature.select_set(True)
    layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        edit = armature.data.edit_bones[bone.name]
        edit.tail = edit.tail + Vector((0.0, 0.0, 0.05))
        bones = {each.name: (each.head.copy(), each.tail.copy()) for each in armature.data.edit_bones}
        positions, _colors = hairview.chain_lines(armature, rig)
        check(same_segments(positions, expected_segments(armature, rig, bones)),
              "lines: in Edit Mode, the edit bones")
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')


def test_line_rebuilds():
    obj, armature = rig_scene()
    context = bpy.context
    wm = context.window_manager
    rig = armature.magic_fit_hair_rig
    wm.magic_fit_show_chains = True
    try:
        lines = hairview._refresh_lines(context, armature)
        check(lines is not None and same_segments(lines.positions, expected_segments(armature, rig)),
              "line rebuild: built for the current armature")
        lines.batch = "drawn"
        hairview._refresh_lines(context, armature)
        check(lines.batch == "drawn", "line rebuild: nothing changed, nothing built again")
        bone = armature.pose.bones[rig.chains[0].bones.split(",")[0]]
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler.x = 0.5
        context.view_layer.update()
        check(lines.moved, "line rebuild: posing marks the lines")
        again = hairview._refresh_lines(context, armature)
        check(again is lines and not lines.moved and lines.batch is None
              and same_segments(lines.positions, expected_segments(armature, rig)),
              "line rebuild: ...and they follow the pose")
        lines.batch = "drawn"
        armature.location.x += 0.2
        context.view_layer.update()
        hairview._refresh_lines(context, armature)
        check(lines.batch is None and same_segments(lines.positions, expected_segments(armature, rig)),
              "line rebuild: moving the armature moves them")
        lines.batch = "drawn"
        alpha = lines.colors[:, 3].copy()
        rig.active = 1 if rig.active != 1 else 0
        hairview._refresh_lines(context, armature)
        check(lines.batch is None and not np.array_equal(lines.colors[:, 3], alpha),
              "line rebuild: another chain picked recolours them")
        check(hairview._refresh_lines(context, armature) is lines and len(hairview._lines) == 1,
              "line rebuild: one per view layer")
        # The lines don't need a tinted mesh: the armature alone, current, has them.
        context.view_layer.objects.active = armature
        obj.hide_set(True)
        check(hairview.overlay_meshes(context, armature) == []
              and hairview._refresh_lines(context, armature) is lines, "line rebuild: drawn without a tint")
        obj.hide_set(False)
        wm.magic_fit_show_chains = False
        check(not hairview._lines, "line rebuild: off forgets them")
        wm.magic_fit_show_chains = True
        hairview._refresh_lines(context, armature)
        hairview.clear()
        check(not hairview._lines and not hairview._tints, "line rebuild: clear forgets them")
    finally:
        obj.hide_set(False)
        wm.magic_fit_show_chains = False
        hairview.clear()


def test_normals_only_when_moved():
    obj, armature = rig_scene()
    context = bpy.context
    wm = context.window_manager
    rig = armature.magic_fit_hair_rig
    calls = []
    original = hairview._evaluated_normals

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    hairview._evaluated_normals = counted
    wm.magic_fit_show_chains = True
    try:
        meshes = hairview.overlay_meshes(context, armature)
        hairview._refresh(context, armature, meshes)
        tint = hairview._layer_tints(context)[obj.session_uid]
        check(len(calls) == 1 and tint.normals.shape == tint.co.shape, "normals: read for the first tint")
        lengths = np.linalg.norm(tint.normals, axis=1)
        check(np.allclose(lengths[lengths > 0], 1.0, atol=1e-4), "normals: unit length")
        # A paint stroke: the weights change, the depsgraph lists the hair's geometry too, nothing moves.
        positions = tint.co
        group = obj.vertex_groups[rig.chains[0].bones.split(",")[0]]
        group.add(list(range(len(obj.data.vertices))), 0.0, 'REPLACE')
        obj.data.update()
        context.view_layer.update()
        check(tint.reweigh and tint.moved, "normals: a weight change marks weights and positions")
        tint.built -= hairview.REBUILD_INTERVAL
        tint.weighed -= 10.0
        hairview._refresh(context, armature, meshes)
        check(len(calls) == 1 and tint.co is positions and not tint.reweigh and not tint.moved,
              "normals: not read again when a weight change leaves the hair where it was ({:d})".format(len(calls)))
        # A pose change moves it: read again.
        tint.batch = "drawn"
        bone = armature.pose.bones[rig.chains[1].bones.split(",")[0]]
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler.x = 0.4
        context.view_layer.update()
        tint.built -= hairview.REBUILD_INTERVAL
        hairview._refresh(context, armature, meshes)
        check(len(calls) == 2 and tint.co is not positions and tint.batch is None,
              "normals: read again after a pose change ({:d})".format(len(calls)))
        # The moved object: too.
        obj.location.z += 0.1
        armature.location.z += 0.1
        context.view_layer.update()
        tint.built -= hairview.REBUILD_INTERVAL
        hairview._refresh(context, armature, meshes)
        check(len(calls) == 3, "normals: read again after the hair moved")
        geometry = painting.read_geometry(obj, context.evaluated_depsgraph_get(), source='DISPLAYED', normals=True)
        check(np.allclose(np.abs((tint.normals * geometry.normals).sum(axis=1)), 1.0, atol=1e-3),
              "normals: the displayed mesh's normals, in world space")
    finally:
        hairview._evaluated_normals = original
        if bpy.app.timers.is_registered(hairview._tag_views):
            bpy.app.timers.unregister(hairview._tag_views)
        wm.magic_fit_show_chains = False
        hairview.clear()


def view(shading_type='SOLID', xray=False, xray_wireframe=False, overlays=True, bones=True, local=False):
    """A stand-in for a SpaceView3D."""
    return SimpleNamespace(type='VIEW_3D', overlay=SimpleNamespace(show_overlays=overlays, show_bones=bones),
                           shading=SimpleNamespace(type=shading_type, show_xray=xray,
                                                   show_xray_wireframe=xray_wireframe),
                           local_view=SimpleNamespace() if local else None)


class FakeArmature:
    """Stands in for an armature object's visibility: keeps what it was asked."""

    def __init__(self, visible=True, in_local_view=True):
        self.visible, self.in_local_view, self.asked = visible, in_local_view, []

    def visible_get(self, view_layer=None, viewport=None):
        self.asked.append(("visible_get", view_layer, viewport))
        return self.visible

    def local_view_get(self, viewport):
        self.asked.append(("local_view_get", viewport))
        return self.in_local_view


def test_bone_lines_shown():
    shown = hairview.bone_lines_shown
    layer = bpy.context.view_layer
    space = view()
    armature = FakeArmature()
    check(shown(space, layer, armature), "bone lines: shown for a visible armature")
    check(("visible_get", layer, space) in armature.asked,
          "bone lines: visibility asked for that view layer and that viewport ({})".format(armature.asked))
    check(not shown(view(bones=False), layer, FakeArmature()), "bone lines: not with the Bones overlay off")
    check(not shown(space, layer, FakeArmature(visible=False)), "bone lines: not for an armature hidden there")
    local = view(local=True)
    check(not shown(local, layer, FakeArmature(in_local_view=False)), "bone lines: not for one out of local view")
    armature = FakeArmature()
    check(shown(local, layer, armature) and ("local_view_get", local) in armature.asked,
          "bone lines: shown for one in local view")
    armature = FakeArmature(in_local_view=False)
    check(shown(space, layer, armature) and not any(a[0] == "local_view_get" for a in armature.asked),
          "bone lines: local view asked only in local view")


def test_adaptive_reweigh():
    obj, armature = rig_scene()
    context = bpy.context
    wm = context.window_manager
    rig = armature.magic_fit_hair_rig
    original = hairview.chain_weights
    reads = []

    def slow(*args):
        reads.append(time.perf_counter())
        time.sleep(0.08)
        return original(*args)

    hairview.chain_weights = slow
    wm.magic_fit_show_chains = True
    try:
        meshes = hairview.overlay_meshes(context, armature)
        hairview._refresh(context, armature, meshes)
        tint = hairview._layer_tints(context)[obj.session_uid]
        check(len(reads) == 1 and 0.08 <= tint.weigh_cost < 0.5,
              "reweigh: chain_weights' time is kept ({:.3f} s)".format(tint.weigh_cost))
        check(hairview.WEIGH_WAIT == 5, "reweigh: WEIGH_WAIT 5")
        wait = max(hairview.REBUILD_INTERVAL, hairview.WEIGH_WAIT * tint.weigh_cost)
        check(wait > hairview.REBUILD_INTERVAL, "reweigh: a slow read waits longer than REBUILD_INTERVAL")
        group = obj.vertex_groups[rig.chains[0].bones.split(",")[0]]

        def paint(weight):
            group.add(list(range(len(obj.data.vertices))), weight, 'REPLACE')
            obj.data.update()
            context.view_layer.update()

        # Past REBUILD_INTERVAL, short of the slow read's wait: the weights wait, the positions don't.
        paint(0.0)
        bone = armature.pose.bones[rig.chains[1].bones.split(",")[0]]
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler.x = 0.4
        context.view_layer.update()
        positions = tint.co
        now = time.perf_counter()
        tint.built = now - hairview.REBUILD_INTERVAL - 0.01
        tint.weighed = now - (hairview.REBUILD_INTERVAL + wait) / 2
        put_off = hairview._refresh(context, armature, meshes)
        check(put_off and len(reads) == 1 and tint.reweigh,
              "reweigh: put off until WEIGH_WAIT times the last read ({:d} reads)".format(len(reads)))
        check(not tint.moved and tint.co is not positions, "reweigh: ...while the positions are read again")
        check(bpy.app.timers.is_registered(hairview._tag_views), "reweigh: a timer redraws when it's due")
        tint.weighed = time.perf_counter() - wait - 0.01
        put_off = hairview._refresh(context, armature, meshes)
        check(not put_off and len(reads) == 2 and not tint.reweigh, "reweigh: done once due")
        # A fast read waits REBUILD_INTERVAL as before.
        hairview.chain_weights = original
        hairview._refresh(context, armature, meshes)
        tint.weigh_cost = 0.001
        paint(0.5)
        tint.weighed = time.perf_counter() - hairview.REBUILD_INTERVAL - 0.01
        tint.built = tint.weighed
        put_off = hairview._refresh(context, armature, meshes)
        check(not put_off and not tint.reweigh, "reweigh: a fast read waits REBUILD_INTERVAL only")
    finally:
        hairview.chain_weights = original
        if bpy.app.timers.is_registered(hairview._tag_views):
            bpy.app.timers.unregister(hairview._tag_views)
        wm.magic_fit_show_chains = False
        hairview.clear()


def test_xray_skips():
    shows = hairview.view_shows_chains
    check(shows(view()), "x-ray: drawn in solid shading")
    check(not shows(view(xray=True)), "x-ray: not with X-ray on")
    check(not shows(view('WIREFRAME', xray_wireframe=True)), "x-ray: not in wireframe with its X-ray on")
    check(shows(view('WIREFRAME', xray=True)), "x-ray: wireframe goes by its own X-ray")
    check(shows(view('MATERIAL', xray=True)), "x-ray: Material Preview has none")
    check(not shows(view(overlays=False)), "x-ray: not with overlays hidden")
    check(not shows(None) and not shows(SimpleNamespace(type='IMAGE_EDITOR')), "x-ray: only in a 3D view")


class StateRecorder:
    """Stands in for gpu.state: keeps the calls made on it."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name,) + args)
        return call


def test_gpu_state_reset():
    state = StateRecorder()
    try:
        with hairview._drawing_state(state):
            state.depth_test_set('ALWAYS')
            raise RuntimeError("drawing failed")
    except RuntimeError:
        pass
    last = {}
    for call in state.calls:
        last[call[0]] = call[1:]
    check(('blend_set', 'ALPHA') in state.calls and ('depth_test_set', 'LESS_EQUAL') in state.calls
          and ('depth_mask_set', False) in state.calls, "gpu state: set for drawing ({})".format(state.calls))
    check(last.get('blend_set') == ('NONE',) and last.get('depth_test_set') == ('NONE',)
          and last.get('depth_mask_set') == (True,) and last.get('line_width_set') == (1.0,),
          "gpu state: blend, depth test, depth mask and line width reset after an error ({})".format(last))
    state = StateRecorder()
    with hairview._drawing_state(state):
        pass
    check(state.calls[-4:] and dict((c[0], c[1:]) for c in state.calls[-4:]).get('blend_set') == ('NONE',),
          "gpu state: reset without an error too")


def main():
    magic_fit.register()
    try:
        tests = (test_palettes, test_bones_colored_by_kind, test_current_armature, test_toggle_in_front,
                 test_toggle_drawn, test_save_keeps_in_front_off, test_renamed_armature_restored,
                 test_in_front_new_current, test_undo_redo_in_front, test_unregister_turns_off, test_chain_colors,
                 test_chain_weights, test_chain_weights_edge_cases, test_overlay_meshes, test_overlay_rebuilds,
                 test_chain_lines, test_line_rebuilds, test_normals_only_when_moved, test_xray_skips,
                 test_gpu_state_reset, test_bone_lines_shown, test_adaptive_reweigh)
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
