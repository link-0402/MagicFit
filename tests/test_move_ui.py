"""Interactive test for the Fit Move tool (Edit Mode), driven by simulated mouse and keyboard events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_move_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set).
"""

import os
import sys
import traceback

import bmesh
import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import magic_fit  # noqa: E402

SHOTS = os.environ.get("MF_SCREENSHOTS")
FAILURES = []
GAP = 0.012


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
        co = view3d_utils.location_3d_to_region_2d(self.region, self.rv3d, Vector(point))
        return int(round(co.x + self.region.x)), int(round(co.y + self.region.y))

    def event(self, type, value, xy, **kwargs):
        self.window.event_simulate(type=type, value=value, x=xy[0], y=xy[1], **kwargs)

    def screenshot(self, name):
        if SHOTS:
            with bpy.context.temp_override(window=self.window, area=self.area):
                bpy.ops.screen.screenshot_area(filepath=os.path.join(SHOTS, name))


def strips():
    """Two cloth strips GAP apart (like trouser legs on both inner thighs), joined at the top."""
    us = np.arange(54) * 0.004
    vs = np.linspace(-0.03, 0.03, 13)
    co = []
    for u in us:
        for v in vs:
            if u <= 0.1:
                co.append((-GAP / 2.0, v, u - 0.1))
            elif u <= 0.1 + GAP:
                co.append((-GAP / 2.0 + (u - 0.1), v, 0.0))
            else:
                co.append((GAP / 2.0, v, -(u - 0.1 - GAP)))
    index = np.arange(len(co)).reshape(len(us), len(vs))
    faces = np.stack((index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]), axis=-1).reshape(-1, 4)
    return co, faces.tolist(), np.repeat(us, len(vs))


def positions(obj):
    obj.update_from_editmode()
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def selection(obj):
    obj.update_from_editmode()
    mask = np.empty(len(obj.data.vertices), dtype=bool)
    obj.data.vertices.foreach_get("select", mask)
    return mask


def select_only(obj, indices):
    bm = bmesh.from_edit_mesh(obj.data)
    for vert in bm.verts:
        vert.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False
    bm.verts.ensure_lookup_table()
    for i in indices:
        bm.verts[i].select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)


def drag(env, start, end, steps=10):
    env.event('MOUSEMOVE', 'NOTHING', start)
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', start)
    yield 0.05
    for i in range(1, steps + 1):
        env.event('MOUSEMOVE', 'NOTHING', (start[0] + (end[0] - start[0]) * i // steps,
                                           start[1] + (end[1] - start[1]) * i // steps))
        yield 0.03
    env.event('LEFTMOUSE', 'RELEASE', end)
    yield 0.3


def key(env, type, xy, **kwargs):
    env.event(type, 'PRESS', xy, **kwargs)
    env.event(type, 'RELEASE', xy, **kwargs)


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
    co, faces, along = strips()
    mesh = bpy.data.meshes.new("Strips")
    mesh.from_pydata(co, [], faces)
    obj = bpy.data.objects.new("Strips", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    tool_settings = bpy.context.scene.tool_settings
    tool_settings.use_proportional_edit = True
    tool_settings.proportional_edit_falloff = 'SMOOTH'
    tool_settings.proportional_size = 0.032
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.view3d.view_axis(type='LEFT')
        bpy.ops.wm.tool_set_by_id(name="magic_fit.fit_move_tool")
    env.rv3d.view_location = (0.0, 0.0, -0.05)
    env.rv3d.view_distance = 0.25
    select_only(obj, [])
    with env.override():
        bpy.ops.ed.undo_push(message="Test setup")
    yield 0.5

    tool = bpy.context.workspace.tools.from_space_view3d_mode('EDIT_MESH', create=False)
    check(tool is not None and tool.idname == "magic_fit.fit_move_tool", "Fit Move tool active in Edit Mode")
    start = positions(obj)
    target = int(np.argmin(np.abs(along - 0.048) + np.abs(start[:, 1])))
    other = along > 0.1 + GAP
    near = int(np.argmin(np.abs(along - 0.064) + np.abs(start[:, 1])))  # halfway out to the proportional size

    # --- Click selects, like Blender's Move tool.
    env.event('MOUSEMOVE', 'NOTHING', env.to_window(start[target]))
    yield 0.05
    key(env, 'LEFTMOUSE', env.to_window(start[target]))
    yield 0.3
    check(np.flatnonzero(selection(obj)).tolist() == [target], "click selected the vertex under the mouse")

    # --- Drag moves the selection, and the leg across the gap stays put.
    a = env.to_window(start[target])
    b = (a[0] + 40, a[1] + 30)
    yield from drag(env, a, b)
    env.screenshot("move_01_drag.png")
    after = positions(obj)
    moved = after[target] - start[target]
    check(np.linalg.norm(moved) > 0.005 and abs(moved[0]) < 1e-6, "drag moved the selection in the view's plane ({:.1f} mm)".format(
        np.linalg.norm(moved) * 1000))
    check(np.array_equal(after[other], start[other]), "the other leg, 12 mm across the gap, stayed put")
    half = np.linalg.norm(after[near] - start[near]) / np.linalg.norm(moved)
    check(abs(half - 0.5) < 0.05, "halfway out to the proportional size, a vertex moved half as far ({:.2f})".format(half))
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    obj = bpy.data.objects["Strips"]
    tool_settings = bpy.context.scene.tool_settings
    check(np.array_equal(positions(obj), start), "undo restored every vertex")

    # --- G, along Z only, a larger proportional size (wheel), confirmed with a click.
    xy = env.to_window(start[target])
    env.event('MOUSEMOVE', 'NOTHING', xy)
    yield 0.05
    key(env, 'G', xy)
    yield 0.1
    key(env, 'Z', xy)
    env.event('WHEELDOWNMOUSE', 'PRESS', xy)
    yield 0.05
    env.event('MOUSEMOVE', 'NOTHING', (xy[0] + 30, xy[1] + 40))
    yield 0.1
    env.screenshot("move_02_grab.png")
    key(env, 'LEFTMOUSE', (xy[0] + 30, xy[1] + 40))
    yield 0.3
    after = positions(obj)
    moved = after[target] - start[target]
    check(moved[2] > 0.003 and np.abs(moved[:2]).max() < 1e-6, "G then Z moved it along Z only ({:.1f} mm)".format(moved[2] * 1000))
    check(abs(tool_settings.proportional_size - 0.032 * 1.1) < 1e-6,
          "the wheel grew the proportional size ({:.4f} m)".format(tool_settings.proportional_size))
    check(np.array_equal(after[other], start[other]), "the other leg stayed put again")
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    obj = bpy.data.objects["Strips"]
    tool_settings = bpy.context.scene.tool_settings
    tool_settings.proportional_size = 0.032

    # --- Esc puts everything back.
    key(env, 'G', xy)
    yield 0.1
    env.event('MOUSEMOVE', 'NOTHING', (xy[0] + 50, xy[1]))
    yield 0.1
    check(not np.array_equal(positions(obj), start), "a move in progress shows")
    key(env, 'ESC', (xy[0] + 50, xy[1]))
    yield 0.3
    check(np.array_equal(positions(obj), start), "Esc put every vertex back")

    # --- Hidden vertices, and the ones next to them, stay put.
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    for vert in bm.verts:
        if along[vert.index] >= 0.06 - 1e-9:
            vert.hide = True
    for face in bm.faces:
        face.hide = any(vert.hide for vert in face.verts)
    for edge in bm.edges:
        edge.hide = any(vert.hide for vert in edge.verts)
    bmesh.update_edit_mesh(obj.data)
    with env.override():
        bpy.ops.ed.undo_push(message="Hide the top")
    yield 0.3
    border = np.abs(along - 0.056) < 1e-9
    yield from drag(env, xy, (xy[0] + 40, xy[1] + 30))
    after = positions(obj)
    check(np.linalg.norm(after[target] - start[target]) > 0.001, "with hidden vertices near, the selection still moved")
    check(np.array_equal(after[along >= 0.056 - 1e-9], start[along >= 0.056 - 1e-9]),
          "the hidden vertices and the row next to them stayed put")
    check(np.linalg.norm(after[border] - start[border], axis=1).max() == 0.0, "so the visible part stays joined to them")
    env.screenshot("move_03_hidden.png")
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    obj = bpy.data.objects["Strips"]

    # --- A drag starting on an unselected vertex selects and moves that one.
    # Farther from the old selection than the proportional size.
    other_target = int(np.argmin(np.abs(along - 0.0) + np.abs(start[:, 1] + 0.02)))
    a = env.to_window(start[other_target])
    yield from drag(env, a, (a[0] - 30, a[1] + 20))
    after = positions(obj)
    check(np.flatnonzero(selection(obj)).tolist() == [other_target], "dragging an unselected vertex selected it")
    check(np.linalg.norm(after[other_target] - start[other_target]) > 0.003 and
          np.array_equal(after[target], start[target]), "and moved it instead of the old selection")


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
