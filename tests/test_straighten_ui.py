"""Interactive test for the Weight Brushes' Straighten mode, driven by simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_straighten_ui.py

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

import cleavage  # noqa: E402
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

    def override(self):
        return bpy.context.temp_override(window=self.window, area=self.area, region=self.region)

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
        self.rv3d.view_location = (0.0, -0.1, -0.01)
        self.rv3d.view_distance = 0.45


def stroke(env, points):
    """Press at the first point, drag through the others, release at the last."""
    env.event('MOUSEMOVE', 'NOTHING', points[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', points[0])
    for a, b in zip(points, points[1:]):
        for t in np.linspace(0.0, 1.0, 5)[1:]:
            env.event('MOUSEMOVE', 'NOTHING', (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)))
            yield 0.02
    env.event('LEFTMOUSE', 'RELEASE', points[-1])
    yield 0.3


def paint_mode(obj):
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for other in bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.event('MOUSEMOVE', 'NOTHING', middle)
    env.event('ESC', 'PRESS', middle)  # close the splash screen
    env.event('ESC', 'RELEASE', middle)
    yield 0.3

    rig, body, garment, xs, zs = cleavage.build()
    scene = bpy.context.scene
    settings = scene.magic_fit
    settings.mode = 'STRAIGHTEN'
    settings.target = body
    settings.radius = 150
    settings.strength = 1.0
    settings.use_pressure_strength = False
    with env.override():
        bpy.ops.wm.tool_set_by_id(name="magic_fit.tool")
    env.view('FRONT')
    body.hide_set(True)  # a clear view of the garment
    yield 0.5
    check(operators.check_ready(bpy.context, settings) is None, "the brush is ready")
    start = cleavage.weights(garment)
    center = env.to_window(Vector((0.0, -0.17, 0.0)))

    # --- In the rest pose nothing sags: a click changes nothing.
    yield from stroke(env, [center, center])
    check(np.array_equal(cleavage.weights(bpy.data.objects["Garment"]), start), "rest pose: a click changes nothing")

    # --- Scale the breasts, as a Customize+ profile would: the copied weights fold the fabric.
    cleavage.scale_breasts(rig, 1.0)
    rest = cleavage.displayed(garment)
    cleavage.scale_breasts(rig, 1.4)
    with env.override():
        bpy.ops.ed.undo_push(message="Scaled")
    yield 0.3
    garment = bpy.data.objects["Garment"]
    before = cleavage.fold(cleavage.displayed(garment), xs, zs)
    check(before > 0.02, "scaled: the copied weights fold the fabric in the cleavage ({:.1f} mm)".format(before * 1000))
    env.view('TOP')
    yield 0.3
    env.screenshot("straighten_01_top_before.png")
    env.view('FRONT')
    yield 0.3

    # --- Click the cleavage.
    yield from stroke(env, [center, center])
    garment = bpy.data.objects["Garment"]
    shown = cleavage.displayed(garment)
    after = cleavage.fold(shown, xs, zs)
    check(after < 0.2 * before, "straightened: the fold is gone ({:.1f} -> {:.1f} mm)".format(
        before * 1000, after * 1000))
    weights = cleavage.weights(garment)
    check(np.abs(weights.sum(axis=1) - start.sum(axis=1)).max() < 1e-4, "straightened: total weights kept")
    X = rest[:, 0]
    outside = np.abs(X) > 0.1
    check(np.array_equal(weights[outside], start[outside]), "the parts beside the breasts kept their weights")
    env.screenshot("straighten_02_front_after.png")
    env.view('TOP')
    yield 0.3
    env.screenshot("straighten_03_top_after.png")
    env.screenshot("straighten_04_window.png", whole_window=True)
    env.view('FRONT')
    yield 0.2

    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    garment = bpy.data.objects["Garment"]
    check(np.abs(cleavage.weights(garment) - start).max() < 1e-6, "undo restored the weights")

    # --- Nothing changes until the stroke ends, and Escape cancels it.
    down = [env.to_window(Vector((0.0, -0.17, z))) for z in (0.04, -0.04)]
    env.event('MOUSEMOVE', 'NOTHING', down[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', down[0])
    for t in np.linspace(0.0, 1.0, 8)[1:]:
        env.event('MOUSEMOVE', 'NOTHING', (round(down[0][0] + (down[1][0] - down[0][0]) * t),
                                           round(down[0][1] + (down[1][1] - down[0][1]) * t)))
        yield 0.02
    check(np.abs(cleavage.weights(bpy.data.objects["Garment"]) - start).max() < 1e-6,
          "nothing changes while the stroke goes on")
    env.event('ESC', 'PRESS', down[1])
    env.event('ESC', 'RELEASE', down[1])
    env.event('LEFTMOUSE', 'RELEASE', down[1])
    yield 0.3
    check(np.abs(cleavage.weights(bpy.data.objects["Garment"]) - start).max() < 1e-6, "escape: nothing changed")

    # --- Straps, and X mirror: clicking one strap straightens the mirrored one too.
    garment = bpy.data.objects["Garment"]
    garment.hide_set(True)
    rig = bpy.data.objects["Rig"]
    body = bpy.data.objects["Body"]
    strap = cleavage.add_strap(rig, body, both=True)
    paint_mode(strap)
    with env.override():
        bpy.ops.ed.undo_push(message="Straps")
    yield 0.3
    strap = bpy.data.objects["Strap"]
    strap_start = cleavage.weights(strap)
    sags = [cleavage.strap_sag(cleavage.displayed(strap), piece) for piece in (0, 1)]
    check(min(sags) > 0.005, "scaled: both straps sag under the breasts ({:.1f}, {:.1f} mm)".format(
        *(s * 1000 for s in sags)))
    settings = bpy.context.scene.magic_fit
    settings.radius = 30
    under = env.to_window(Vector((cleavage.STRAP_X, -0.13, -0.07)))
    yield from stroke(env, [under, under])
    strap = bpy.data.objects["Strap"]
    fixed = [cleavage.strap_sag(cleavage.displayed(strap), piece) for piece in (0, 1)]
    check(fixed[0] < 0.35 * sags[0] and fixed[1] > 0.8 * sags[1],
          "no mirror: one click straightens the strap clicked ({:.1f} -> {:.1f} mm), not the other".format(
              sags[0] * 1000, fixed[0] * 1000))
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    strap = bpy.data.objects["Strap"]
    check(np.abs(cleavage.weights(strap) - strap_start).max() < 1e-6, "undo restored the strap")
    strap.data.use_mirror_x = True
    yield from stroke(env, [under, under])
    strap = bpy.data.objects["Strap"]
    mirrored = [cleavage.strap_sag(cleavage.displayed(strap), piece) for piece in (0, 1)]
    check(max(mirrored[0] / sags[0], mirrored[1] / sags[1]) < 0.35,
          "mirror: both straps straightened ({:.1f}, {:.1f} mm)".format(*(s * 1000 for s in mirrored)))
    env.screenshot("straighten_05_straps.png")
    strap.data.use_mirror_x = False

    # --- The settings: Straighten shows the body and a hint instead of a resting distance.
    properties = next(a for a in env.window.screen.areas if a.type == 'PROPERTIES')
    properties.spaces.active.context = 'TOOL'
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.3
    env.screenshot("straighten_06_panel.png", whole_window=True)


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
