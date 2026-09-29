"""Interactive test for the Weight Brushes' Heels mode and the Heel Weights button, driven by
simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_heels_ui.py

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

import heels  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import operators  # noqa: E402

SHOTS = os.environ.get("MF_SCREENSHOTS")
FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
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
        self.rv3d.view_location = (0.0, 0.0, 0.08)
        self.rv3d.view_distance = 0.5


def stroke(env, points, passes=3):
    path = []
    for i in range(passes):
        path += points if i % 2 == 0 else points[::-1]
    env.event('MOUSEMOVE', 'NOTHING', path[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', path[0])
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0.0, 1.0, 5)[1:]:
            env.event('MOUSEMOVE', 'NOTHING', (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)))
            yield 0.02
    env.event('LEFTMOUSE', 'RELEASE', path[-1])
    yield 0.3


def difference(a, b):
    zero = np.zeros(len(next(iter(a.values()))))
    return max(float(np.abs(a.get(n, zero) - b.get(n, zero)).max()) for n in set(a) | set(b))


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.event('MOUSEMOVE', 'NOTHING', middle)
    env.event('ESC', 'PRESS', middle)  # close the splash screen
    env.event('ESC', 'RELEASE', middle)
    yield 0.3

    _rig, shoes = heels.build()
    settings = bpy.context.scene.magic_fit
    settings.mode = 'HEELS'
    settings.target = None
    settings.radius = 300  # about 9 cm: across the whole shoe, not over to the other one
    settings.strength = 1.0
    settings.use_pressure_strength = False
    with env.override():
        bpy.ops.wm.tool_set_by_id(name="magic_fit.tool")
    env.view('RIGHT')
    shoes.vertex_groups.active_index = shoes.vertex_groups["j_asi_e_l"].index
    yield 0.5
    check(operators.check_ready(bpy.context, settings) is None, "the brush is ready without a body")
    with env.override():
        bpy.ops.ed.undo_push(message="Start")
    start = heels.by_name(shoes)
    env.screenshot("heels_01_before.png")

    # --- Paint along the left shoe's sole, from the heel to the toes (seen from the right, x > 0 is in front).
    along = [env.to_window(Vector((heels.X + 0.04, y, z))) for y, z in ((0.04, 0.065), (-0.15, -0.005))]
    yield from stroke(env, along)
    shoes = bpy.data.objects["Shoes"]
    after = heels.by_name(shoes)
    half = len(shoes.data.vertices) // 2
    # The brush's falloff fades toward the far side of the shoe; the side it passes over is done.
    outer = heels.rest_co(shoes)[:, 0] > heels.X + 0.02
    toes = after["j_asi_e_l"] + after["iv_asi_oya_b_l"]
    toes_start = start["j_asi_e_l"] + start["iv_asi_oya_b_l"]
    check(toes[outer].sum() < 0.02 * toes_start[outer].sum() and toes[:half].sum() < 0.3 * toes_start[:half].sum(),
          "painted: the left foot lost its toe weights ({:.0%} left)".format(toes[:half].sum() / toes_start[:half].sum()))
    check(np.array_equal(after["j_asi_e_r"], start["j_asi_e_r"]), "painted: only under the brush")
    total = sum(v for n, v in after.items() if n.startswith(("j_", "iv_")))
    check(float(np.abs(total - 1.0).max()) < 1e-4, "painted: total weights kept")
    env.screenshot("heels_02_painted.png")

    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    check(difference(heels.by_name(bpy.data.objects["Shoes"]), start) < 1e-6, "undo restored the weights")

    # --- The sidebar, then the button from Object Mode.
    env.area.spaces.active.show_region_ui = True
    yield 0.3
    sidebar = next(r for r in env.area.regions if r.type == 'UI')
    try:
        sidebar.active_panel_category = "Magic Fit"
    except AttributeError:
        pass
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.4
    env.screenshot("heels_03_sidebar.png", whole_window=True)
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    yield 0.3
    with env.override(sidebar):
        check(bpy.ops.magic_fit.heel_weights.poll(), "button: ready in Object Mode")
        result = bpy.ops.magic_fit.heel_weights('EXEC_DEFAULT', True)
    yield 0.3
    shoes = bpy.data.objects["Shoes"]
    after = heels.by_name(shoes)
    check(result == {'FINISHED'} and not after["j_asi_e_l"].any() and not after["j_asi_e_r"].any(),
          "button: both feet lost their toe weights")
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    check(heels.by_name(bpy.data.objects["Shoes"])["j_asi_e_l"].any(), "button: undo takes it back")
    with env.override():
        bpy.ops.ed.redo()
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    bpy.data.objects["Shoes"].vertex_groups.active_index = bpy.data.objects["Shoes"].vertex_groups["j_asi_d_l"].index
    yield 0.4
    env.screenshot("heels_04_after_button.png", whole_window=True)


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
