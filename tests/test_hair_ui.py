"""Interactive test for Hair weights: the Hair Weights button on the sidebar's Hair sub-tab and the Weight
Brushes' Hair mode, driven by simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_hair_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set).
"""

import os
import sys
import traceback

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import hair  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import hairing, operators  # noqa: E402

SHOTS = os.environ.get("MF_SCREENSHOTS")
FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    sys.stdout.flush()
    if not condition:
        FAILURES.append(message)


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


def run():
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
