"""Interactive test for the Weight Brushes' Smooth mode, driven by simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_smooth_ui.py

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
sys.path.insert(0, os.path.join(ROOT, "deps"))  # SciPy, for the direct solver

import smooth  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import operators  # noqa: E402

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

    def view_top(self):
        with self.override():
            bpy.ops.view3d.view_axis(type='TOP')
        self.rv3d.view_location = (0.1, 0.05, 0.0)
        self.rv3d.view_distance = 0.4


def stroke(env, points):
    env.event('MOUSEMOVE', 'NOTHING', points[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', points[0])
    for a, b in zip(points, points[1:]):
        for t in np.linspace(0.0, 1.0, 8)[1:]:
            env.event('MOUSEMOVE', 'NOTHING', (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)))
            yield 0.02
    env.event('LEFTMOUSE', 'RELEASE', points[-1])
    yield 0.3


def difference(a, b):
    return max(float(np.abs(a[name] - b[name]).max()) for name in a)


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.event('MOUSEMOVE', 'NOTHING', middle)
    env.event('ESC', 'PRESS', middle)  # close the splash screen
    env.event('ESC', 'RELEASE', middle)
    yield 0.3

    _rig, sheet, owner = smooth.build()
    with env.override():
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    settings = bpy.context.scene.magic_fit
    settings.mode = 'SMOOTH'
    settings.target = None
    settings.radius = 250
    settings.strength = 1.0
    settings.use_pressure_strength = False
    with env.override():
        bpy.ops.wm.tool_set_by_id(name="magic_fit.tool")
    env.view_top()
    sheet.vertex_groups.active_index = sheet.vertex_groups["j_a"].index
    yield 0.5
    check(operators.check_ready(bpy.context, settings) is None, "the brush is ready without a body")
    with env.override():
        bpy.ops.ed.undo_push(message="Start")
    start = smooth.by_name(sheet)
    env.screenshot("smooth_01_before.png")

    # --- Paint across the gap between the two parts.
    across = [env.to_window(Vector((x, 0.05, 0.0))) for x in (0.06, 0.14)]
    yield from stroke(env, across)
    sheet = bpy.data.objects["Sheet"]
    after = smooth.by_name(sheet)
    co = smooth.rest_co(sheet)
    seam_left = np.flatnonzero((np.abs(co[:, 0] - 0.1) < 1e-6) & (np.abs(co[:, 1] - 0.05) < 1e-6))
    seam_right = np.flatnonzero((np.abs(co[:, 0] - 0.1005) < 1e-6) & (np.abs(co[:, 1] - 0.05) < 1e-6))
    check(len(seam_left) == 1 and len(seam_right) == 1, "the setup has a seam vertex on each side")
    left, right = float(after["j_a"][seam_left[0]]), float(after["j_a"][seam_right[0]])
    check(0.3 < left < 0.7 and 0.3 < right < 0.7,
          "painted: the weights were averaged across the gap ({:.2f} | {:.2f})".format(left, right))
    total = after["j_a"] + after["j_b"]
    check(float(np.abs(total - 1.0).max()) < 1e-4, "painted: total weights kept")
    # The brush is about 5 cm across at this zoom and passed over 6 to 14 cm.
    far = (co[:, 0] < 0.005) | (co[:, 0] > 0.195)
    check(far.any() and np.array_equal(after["j_a"][far], start["j_a"][far]), "painted: only under the brush")
    check(float(after["j_b"][owner == 0].max()) > 0.2 and float(after["j_a"][owner == 1].max()) > 0.2,
          "painted: both parts took on the other's weights")
    env.screenshot("smooth_02_painted.png")

    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    check(difference(smooth.by_name(bpy.data.objects["Sheet"]), start) < 1e-6, "undo restored the weights")
    settings = bpy.context.scene.magic_fit  # a reference from before the undo would crash Blender

    # --- Escape cancels a stroke.
    env.event('MOUSEMOVE', 'NOTHING', across[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', across[0])
    for t in np.linspace(0.0, 1.0, 8)[1:]:
        env.event('MOUSEMOVE', 'NOTHING', (round(across[0][0] + (across[1][0] - across[0][0]) * t), across[0][1]))
        yield 0.02
    check(difference(smooth.by_name(bpy.data.objects["Sheet"]), start) > 0.1, "escape: the stroke is under way")
    env.event('ESC', 'PRESS', across[1])
    env.event('ESC', 'RELEASE', across[1])
    env.event('LEFTMOUSE', 'RELEASE', across[1])
    yield 0.3
    check(difference(smooth.by_name(bpy.data.objects["Sheet"]), start) < 1e-6, "escape: the weights are back")

    # --- Every group locked: the brush says so, and paints nothing.
    sheet = bpy.data.objects["Sheet"]
    for group in sheet.vertex_groups:
        group.lock_weight = True
    check("locked" in (operators.check_ready(bpy.context, settings) or ""), "locked: the brush says every group is locked")
    yield from stroke(env, across)
    check(difference(smooth.by_name(sheet), start) < 1e-6, "locked: nothing is painted")
    for group in sheet.vertex_groups:
        group.lock_weight = False

    # --- The sidebar: Smooth is a mode of the General sub-tab and needs no body.
    env.area.spaces.active.show_region_ui = True
    yield 0.3
    sidebar = next(r for r in env.area.regions if r.type == 'UI')
    try:
        sidebar.active_panel_category = "Magic Fit"
    except AttributeError:
        pass
    bpy.context.workspace.magic_fit_tab = 'WEIGHTS'
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.4
    check(settings.weights_tab == 'GENERAL' and settings.general_mode == 'SMOOTH', "the sidebar shows Smooth on General")
    env.screenshot("smooth_03_sidebar.png", whole_window=True)


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
