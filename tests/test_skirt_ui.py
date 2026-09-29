"""Interactive test for the Weight Brushes' Skirt mode and the Skirt Weights button, driven by
simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_skirt_ui.py

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

import skirt  # noqa: E402
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
        self.sidebar = next(r for r in self.area.regions if r.type == 'UI')
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
        self.rv3d.view_location = (0.0, 0.0, 0.82)
        self.rv3d.view_distance = 1.5


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


def by_name(obj):
    table, names = skirt.weights(obj)
    return {name: table[:, k] for k, name in enumerate(names)}


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

    _rig, body, obj = skirt.build()
    settings = bpy.context.scene.magic_fit
    settings.mode = 'SKIRT'
    settings.target = body
    settings.radius = 120
    settings.strength = 1.0
    settings.use_pressure_strength = False
    with env.override():
        bpy.ops.wm.tool_set_by_id(name="magic_fit.tool")
    env.view('FRONT')
    body.hide_set(True)  # a clear view of the skirt
    yield 0.5
    check(operators.check_ready(bpy.context, settings) is None, "the brush is ready")
    with env.override():
        bpy.ops.ed.undo_push(message="Start")
    start = by_name(obj)
    env.screenshot("skirt_01_before.png")

    # --- Paint down the front left of the skirt.
    down = [env.to_window(Vector((0.09, -0.3, z))) for z in (0.95, 0.5)]
    yield from stroke(env, down)
    obj = bpy.data.objects["Skirt"]
    after = by_name(obj)
    check("j_sk_f_c_l" in after and after["j_sk_f_c_l"].any(), "painted: the front chain got weights")
    x = skirt.rest_co(obj)[:, 0]
    changed = np.zeros(len(x), dtype=bool)
    for name, values in after.items():
        changed |= np.abs(values - start.get(name, 0.0)) > 1e-4
    check(changed[x > 0.02].any() and not changed[x < -0.08].any(), "painted: only under the brush")
    bones = [name for name in after if name.startswith("j_")]
    total = sum(after[name] for name in bones)
    check(float(np.abs(total - 1.0).max()) < 1e-4, "painted: total weights kept")
    env.screenshot("skirt_02_painted.png")

    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    obj = bpy.data.objects["Skirt"]
    check(difference(by_name(obj), start) < 1e-6 and "j_sk_f_c_l" not in obj.vertex_groups,
          "undo restored the weights and removed the new groups")

    # --- Escape cancels a stroke in progress.
    env.event('MOUSEMOVE', 'NOTHING', down[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', down[0])
    for t in np.linspace(0.0, 1.0, 8)[1:]:
        env.event('MOUSEMOVE', 'NOTHING', (round(down[0][0] + (down[1][0] - down[0][0]) * t),
                                           round(down[0][1] + (down[1][1] - down[0][1]) * t)))
        yield 0.02
    check(difference(by_name(bpy.data.objects["Skirt"]), start) > 1e-3, "stroke in progress changed weights")
    env.event('ESC', 'PRESS', down[1])
    env.event('ESC', 'RELEASE', down[1])
    env.event('LEFTMOUSE', 'RELEASE', down[1])
    yield 0.3
    obj = bpy.data.objects["Skirt"]
    check(difference(by_name(obj), start) < 1e-6 and "j_sk_f_c_l" not in obj.vertex_groups,
          "escape restored the weights and removed the new groups")

    # --- The settings in the tool header popover and the sidebar.
    env.area.spaces.active.show_region_ui = True
    yield 0.3
    sidebar = next(r for r in env.area.regions if r.type == 'UI')
    try:
        sidebar.active_panel_category = "Magic Fit"
    except AttributeError:
        pass  # read-only in some versions: the tab shows if it is the only one with panels
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.4
    env.screenshot("skirt_03_weight_paint_sidebar.png", whole_window=True)

    # --- The button, from Object Mode, weights the whole skirt.
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    yield 0.3
    env.screenshot("skirt_04_object_mode_sidebar.png", whole_window=True)
    with env.override(env.sidebar):
        check(bpy.ops.magic_fit.skirt_weights.poll(), "button: ready in Object Mode")
        result = bpy.ops.magic_fit.skirt_weights('EXEC_DEFAULT', True)
    yield 0.3
    obj = bpy.data.objects["Skirt"]
    check(result == {'FINISHED'}, "button: finished")
    share = skirt.skirt_share(obj)
    h = skirt.rest_co(obj)[:, 2] - skirt.HIP[2]
    # The flared part hangs from the skirt bones; the tight part keeps some body weight (Skin Weight).
    hem = skirt.rest_co(obj)[:, 2] < 0.6
    check(float(share[hem].min()) > 0.99 and not share[h > 0.10].any(), "button: the whole skirt, below the hips")
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    check("j_sk_f_c_l" not in bpy.data.objects["Skirt"].vertex_groups, "button: undo takes it back")
    with env.override():
        bpy.ops.ed.redo()
    yield 0.3
    obj = bpy.data.objects["Skirt"]
    check("j_sk_f_c_l" in obj.vertex_groups, "button: redo brings it back")
    env.view('FRONT')
    obj.vertex_groups.active_index = obj.vertex_groups["j_sk_f_c_l"].index
    with env.override():
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    yield 0.4
    env.screenshot("skirt_05_after_button.png", whole_window=True)


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
