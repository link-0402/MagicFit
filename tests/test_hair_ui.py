"""Interactive test for Hair weights: the Hair Weights button on the sidebar's Hair sub-tab and the Weight
Brushes' Hair mode, driven by simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_hair_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set).
"""

import os
import sys
import traceback
from types import SimpleNamespace

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import hair  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import hairing, operators, tool  # noqa: E402

SHOTS = os.environ.get("MF_SCREENSHOTS")
FAILURES = []
DRAW_ERRORS = []  # exceptions in the Hair Weights panel's draw and the chain list's rows, which Blender only prints
LIST_ROWS = []  # the chain list rows drawn since it was cleared


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    sys.stdout.flush()
    if not condition:
        FAILURES.append(message)


def watch():
    """Record the exceptions of the Hair Weights panel's draw and the chain list's rows, and the rows drawn.
    Done before registering: Blender checks the methods' argument counts then."""
    panel = tool.VIEW3D_PT_hair_weights_sidebar
    draw = panel.draw

    def draw_panel(self, context):
        try:
            draw(self, context)
        except Exception:
            DRAW_ERRORS.append(traceback.format_exc())
            raise

    panel.draw = draw_panel
    chains = getattr(tool, "MAGIC_FIT_UL_hair_chains", None)
    if chains is None:
        return
    draw_item = chains.draw_item

    def draw_row(self, context, layout, data, item, icon, active_data, active_property, index=0, flt_flag=0):
        LIST_ROWS.append(index)
        try:
            draw_item(self, context, layout, data, item, icon, active_data, active_property, index, flt_flag)
        except Exception:
            DRAW_ERRORS.append(traceback.format_exc())
            raise

    chains.draw_item = draw_row


class Recorder:
    """Stands in for a UILayout: keeps what is drawn on it, as (call, text, args, the Recorder it returned).
    An operator's settings are the attributes set on its Recorder. ``panel`` returns a header and a body,
    as UILayout.panel does."""

    def __init__(self, drawn):
        object.__setattr__(self, "drawn", drawn)

    def __getattr__(self, name):
        def call(*args, **kwargs):
            item = Recorder(self.drawn)
            self.drawn.append((name, kwargs.get("text"), args, item))
            return (item, Recorder(self.drawn)) if name == "panel" else item
        return call


def drawn_texts(settings, rig_state):
    """What the Hair Weights result box draws for the active hair, and the chain list's first row."""
    drawn = []
    tool.VIEW3D_PT_hair_weights_sidebar.draw_result(bpy.context, Recorder(drawn), settings)
    chains = getattr(tool, "MAGIC_FIT_UL_hair_chains", None)
    if chains is not None and rig_state is not None and len(rig_state.chains):
        chains.draw_item(SimpleNamespace(layout_type='DEFAULT'), bpy.context, Recorder(drawn), rig_state,
                         rig_state.chains[0], 0, rig_state, "active", 0)
    return drawn


HAIR_MAIN = {"hair_hang_start", "hair_hang_length", "hair_neck_weight", "hair_type", "hair_chain_spacing"}


def hair_panel_props(context, settings):
    """Which of Hang Start, Hang Length, Neck Weight, Hair Type and Chain Spacing are drawn below the brush's mode
    (Properties > Tool), in the tool header's Options Hair section and in the Hair Weights panel."""
    def props(draw):
        drawn = []
        draw(Recorder(drawn))
        return [args[1] for call, _text, args, _item in drawn
                if call == 'prop' and len(args) > 1 and args[1] in HAIR_MAIN]

    panel = tool.VIEW3D_PT_hair_weights_sidebar
    section = next(draw for idname, _label, draw in tool._weight_sections(context, settings, 'POPOVER')
                   if idname == "magic_fit_hair")
    return (props(lambda layout: tool._draw_mode_main(context, layout, settings)),
            props(section),
            props(lambda layout: panel.draw(SimpleNamespace(layout=layout, draw_pick_hint=panel.draw_pick_hint,
                                                            draw_result=panel.draw_result), context)))


class Env:
    def __init__(self):
        self.window = bpy.context.window_manager.windows[0]
        self.area = next(a for a in self.window.screen.areas if a.type == 'VIEW_3D')
        self.region = next(r for r in self.area.regions if r.type == 'WINDOW')
        self.rv3d = self.area.spaces.active.region_3d

    def override(self, region=None):
        return bpy.context.temp_override(window=self.window, area=self.area, region=region or self.region)

    def to_window(self, point):
        co = view3d_utils.location_3d_to_region_2d(self.region, self.rv3d, point)
        return int(co.x + self.region.x), int(co.y + self.region.y)

    def event(self, type, value, xy, **kwargs):
        self.window.event_simulate(type=type, value=value, x=xy[0], y=xy[1], **kwargs)

    def screenshot(self, name, whole_window=False):
        if not SHOTS:
            return
        path = os.path.join(SHOTS, name)
        with bpy.context.temp_override(window=self.window, area=self.area):
            if whole_window:
                bpy.ops.screen.screenshot(filepath=path)
            else:
                bpy.ops.screen.screenshot_area(filepath=path)

    def view(self, axis):
        with self.override():
            bpy.ops.view3d.view_axis(type=axis)
        self.rv3d.view_location = (0.0, 0.05, 1.38)
        self.rv3d.view_distance = 0.9


def stroke(env, points, passes=3):
    path = []
    for i in range(passes):
        path += points if i % 2 == 0 else points[::-1]
    env.event('MOUSEMOVE', 'NOTHING', path[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', path[0])
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0.0, 1.0, 6)[1:]:
            env.event('MOUSEMOVE', 'NOTHING', (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)))
            yield 0.02
    env.event('LEFTMOUSE', 'RELEASE', path[-1])
    yield 0.3


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.event('MOUSEMOVE', 'NOTHING', middle)
    env.event('ESC', 'PRESS', middle)  # close the splash screen
    env.event('ESC', 'RELEASE', middle)
    yield 0.3

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    obj, parts = hair.build()
    settings = bpy.context.scene.magic_fit
    settings.mode = 'HAIR'
    settings.hair_race = 'c0201'
    settings.hair_skeleton_mode = 'AUTO'
    settings.radius = 120
    settings.strength = 1.0
    settings.use_pressure_strength = False
    env.view('BACK')
    workspace = bpy.context.workspace
    workspace.magic_fit_tab = 'WEIGHTS'
    env.area.spaces.active.show_region_ui = True
    yield 0.3
    sidebar = next(r for r in env.area.regions if r.type == 'UI')
    try:
        sidebar.active_panel_category = "Magic Fit"
    except AttributeError:
        pass
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.5
    check(settings.weights_tab == 'HAIR', "the Hair sub-tab is the Hair mode")
    with env.override(sidebar):
        check(bpy.ops.magic_fit.hair_weights.poll(), "button: ready for a bare hair mesh")
    env.screenshot("hair_01_sidebar.png", whole_window=True)

    # --- The button: best match, armature, weights, one undo step.
    with env.override(sidebar):
        bpy.ops.ed.undo_push(message="Before hair weights")
        result = bpy.ops.magic_fit.hair_weights('EXEC_DEFAULT', True)
    yield 0.4
    obj = bpy.data.objects["Hair"]
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    check(result == {'FINISHED'} and len(armatures) == 1 and obj.modifiers.get("Armature") is not None,
          "button: the hair got an armature and an Armature modifier")
    state = obj.magic_fit_hair
    skeleton = state.skeleton
    check(skeleton > 0 and len(operators.parse_matches(state.matches)) >= 3,
          "button: skeleton h{:04d} and the other matches remembered".format(skeleton))
    weights = hair.by_name(obj)
    chain = [name for name in weights if name.startswith("j_ex_h{:04d}".format(skeleton))]
    tip = parts['ponytail'][-24:]
    check(sum(weights[name][tip] for name in chain).mean() > 0.9,
          "button: the ponytail hangs from the skeleton's chain")
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.4
    env.screenshot("hair_02_after_button.png", whole_window=True)

    with env.override():
        bpy.ops.ed.undo()
    yield 0.4
    obj = bpy.data.objects["Hair"]
    check(not obj.vertex_groups and not obj.modifiers and not [o for o in bpy.data.objects if o.type == 'ARMATURE'],
          "undo: takes the weights, the modifier and the armature back at once")
    with env.override():
        bpy.ops.ed.redo()
    yield 0.4
    # Undo replaced the scene data: look everything up again (writing through old references crashes).
    obj = bpy.data.objects["Hair"]
    settings = bpy.context.scene.magic_fit
    check(len(obj.vertex_groups) > 3 and obj.magic_fit_hair.skeleton == skeleton, "redo: all back")

    # --- The brush paints wiped weights back.
    bpy.context.view_layer.objects.active = obj
    with env.override():
        bpy.ops.magic_fit.use_tool(tool='WEIGHT')
    yield 0.4
    obj = bpy.data.objects["Hair"]
    check(bpy.context.mode == 'PAINT_WEIGHT' and settings.mode == 'HAIR', "brush: Weight Paint mode, Hair mode kept")
    check(operators.check_ready(bpy.context, settings) is None, "brush: ready on the weighted hair")
    tip_list = tip.tolist()
    for group in obj.vertex_groups:
        group.remove(tip_list)
    obj.vertex_groups['j_kao'].add(tip_list, 1.0, 'REPLACE')
    obj.data.update()
    obj.vertex_groups.active_index = obj.vertex_groups[chain[-1]].index
    yield 0.3
    co = hair.parts()[-1][1]
    points = [env.to_window(Vector(co[i])) for i in (len(co) - 60, len(co) - 1)]
    yield from stroke(env, points)
    obj = bpy.data.objects["Hair"]
    painted = hair.by_name(obj)
    check(sum(painted[name][tip] for name in chain).mean() > 0.8,
          "brush: painting the ponytail's tip gives it its chain weights back ({:.2f})".format(
              sum(painted[name][tip] for name in chain).mean()))
    check(float(np.abs(sum(painted.values()) - 1.0).max()) < 1e-4, "brush: weights still add up to 1")
    env.screenshot("hair_03_brush.png", whole_window=True)

    # --- Another match from the result box.
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    yield 0.2
    other = next(n for n, _s in operators.parse_matches(obj.magic_fit_hair.matches) if n != skeleton)
    with env.override(sidebar):
        result = bpy.ops.magic_fit.hair_weights('EXEC_DEFAULT', True, skeleton=other)
    yield 0.4
    obj = bpy.data.objects["Hair"]
    check(result == {'FINISHED'} and obj.magic_fit_hair.skeleton == other and settings.hair_skeleton_mode == 'PICK',
          "match button: re-weights with h{:04d} and keeps it picked".format(other))
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.4
    env.screenshot("hair_04_other_match.png", whole_window=True)

    # --- Try Another EST Entry: any entry, picked from the result box and weighted at once, in one undo step.
    picked = 210 if other != 210 else 196
    with env.override(sidebar):
        result = bpy.ops.magic_fit.pick_hair_skeleton('EXEC_DEFAULT', True, skeleton=str(picked), apply=True)
    yield 0.4
    obj = bpy.data.objects["Hair"]
    check(result == {'FINISHED'} and obj.magic_fit_hair.skeleton == picked and
          obj.get(hairing.EST_PROPERTY) == picked, "try another: weights with the entry picked and tags the hair with it")
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.4
    env.screenshot("hair_05_try_another.png", whole_window=True)
    with env.override():
        bpy.ops.ed.undo()
    yield 0.4
    obj = bpy.data.objects["Hair"]
    check(obj.magic_fit_hair.skeleton == other and obj.get(hairing.EST_PROPERTY) == other,
          "try another: one undo takes it back ({:d})".format(obj.magic_fit_hair.skeleton))

    # --- Own Skeleton: chains built from the hair's shape, listed with their kinds in the result box.
    settings = bpy.context.scene.magic_fit
    settings.hair_skeleton_mode = 'CUSTOM'
    bpy.context.view_layer.objects.active = obj
    with env.override(sidebar):
        result = bpy.ops.magic_fit.hair_weights('EXEC_DEFAULT', True)
    yield 0.4
    obj = bpy.data.objects["Hair"]
    settings = bpy.context.scene.magic_fit
    armature = hairing.weighted_armature(obj)
    rig_state = armature.magic_fit_hair_rig if armature is not None else None
    count = len(rig_state.chains) if rig_state is not None else 0
    check(result == {'FINISHED'} and obj.magic_fit_hair.custom and count > 0,
          "own skeleton: the hair got its own chains ({:d})".format(count))
    DRAW_ERRORS.clear()
    LIST_ROWS.clear()
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.5
    check(not DRAW_ERRORS, "own skeleton: the panel and the chain list drew without errors")
    check(len(LIST_ROWS) > 0, "own skeleton: the chain list drew its rows ({:d})".format(len(LIST_ROWS)))
    with env.override(sidebar):
        drawn = drawn_texts(settings, rig_state)
    texts = [text for _call, text, _args, _item in drawn if text]
    bones = sum(len([name for name in chain.bones.split(",") if name]) for chain in rig_state.chains) if count else 0
    check("{:d} chains, {:d} bones (at most 48)".format(count, bones) in texts,
          "own skeleton: the box counts the chains and bones")
    first_bones = len([name for name in rig_state.chains[0].bones.split(",") if name]) if count else 0
    check("Chain 01  {:d} bones".format(first_bones) in texts,
          "own skeleton: the list's first row is Chain 01 with its bones")
    # A chain the user left one bone in.
    single = []
    tool.MAGIC_FIT_UL_hair_chains.draw_item(SimpleNamespace(layout_type='DEFAULT'), bpy.context, Recorder(single),
                                            rig_state, SimpleNamespace(bones="j_ex_hair_07_a", kind='back'), 0,
                                            rig_state, "active", 0)
    check("Chain 07  1 bone" in [text for _call, text, _args, _item in single if text],
          "own skeleton: a chain of one bone reads 1 bone")
    check(any(call == 'prop' and args[1:] == ("kind",) for call, _text, args, _item in drawn),
          "own skeleton: the row has the kind dropdown")
    check(any(call == 'template_list' and args[0] == "MAGIC_FIT_UL_hair_chains" for call, _text, args, _item in drawn),
          "own skeleton: the box lists the chains")
    rebuild = [item for call, text, _args, item in drawn if call == 'operator' and text == "Rebuild Chains"]
    check(len(rebuild) == 1 and getattr(rebuild[0], "rebuild", False) is True,
          "own skeleton: Rebuild Chains builds the chains again")
    check(not [text for text in texts if text.startswith("EST entry") or text in ("Also fitting:", "Try Another EST Entry")],
          "own skeleton: no EST entry, matches or Try Another EST Entry")
    check(operators.MAGIC_FIT_OT_hair_weights.description(bpy.context, SimpleNamespace(rebuild=True, skeleton=-1)) ==
          "Build the hair's bone chains again from its shape. Your bone edits are lost",
          "own skeleton: Rebuild Chains says what it does")
    env.screenshot("hair_custom_chains.png", whole_window=True)

    # --- The panels: the hair's own skeleton works out where hair hangs, so only Neck Weight stays. The button
    # building it also shows Hair Type and Chain Spacing; the brush builds no chains.
    hang = ["hair_hang_start", "hair_hang_length", "hair_neck_weight"]
    neck = ["hair_neck_weight"]
    spaced = ["hair_type", "hair_chain_spacing"]

    def plain_mesh(name):
        mesh = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        bpy.context.scene.collection.objects.link(mesh)
        return mesh

    unweighted = plain_mesh("Unweighted Hair")
    game = plain_mesh("Game Skeleton Hair")
    game.magic_fit_hair.race = settings.hair_race
    game.magic_fit_hair.skeleton = other
    game.magic_fit_hair.custom = False
    cases = [
        ("unweighted hair, Own Skeleton", unweighted, 'CUSTOM', neck, neck + spaced),
        ("unweighted hair, Best Match", unweighted, 'AUTO', hang, hang),
        ("unweighted hair, Pick", unweighted, 'PICK', hang, hang),
        # The brush paints a game-skeleton hair with Hang Start and Hang Length, whatever the dropdown says.
        ("game skeleton hair, Own Skeleton", game, 'CUSTOM', hang, hang + spaced),
        # The brush on hair with its own skeleton doesn't use them; the button (Best Match) does.
        ("own skeleton hair, Best Match", obj, 'AUTO', neck, hang),
    ]
    DRAW_ERRORS.clear()
    for label, active, mode, brush_expected, panel_expected in cases:
        settings.hair_skeleton_mode = mode
        bpy.context.view_layer.objects.active = active
        with env.override(sidebar):
            brush, popover, panel = hair_panel_props(bpy.context, settings)
        check(brush == brush_expected and popover == brush_expected,
              "panels: {}: the brush draws {}".format(label, ", ".join(brush)))
        check(panel == panel_expected, "panels: {}: Hair Weights draws {}".format(label, ", ".join(panel)))
    check(not DRAW_ERRORS, "panels: drew without errors")
    settings.hair_skeleton_mode = 'CUSTOM'
    bpy.context.view_layer.objects.active = obj
    for mesh in (unweighted, game):
        bpy.data.objects.remove(mesh)


def run():
    watch()
    magic_fit.register()
    gen = steps()

    def tick():
        try:
            return next(gen)
        except StopIteration:
            pass
        except Exception:
            traceback.print_exc()
            FAILURES.append("exception")
        print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
        sys.stdout.flush()
        os._exit(1 if FAILURES else 0)

    bpy.app.timers.register(tick, first_interval=1.5, persistent=True)


run()
