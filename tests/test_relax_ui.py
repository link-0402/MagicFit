"""Interactive test for the Texture Relax tool (Edit Mode), driven by simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_relax_ui.py

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


GRID = 24
SIZE = 1.2


def make_sheet(name, *, bump=True):
    """A UV-mapped, triangulated grid in the XZ plane facing -Y (toward the front view), with the
    vertices around its middle pushed sideways so the texture is stretched there."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    verts = []
    for j in range(GRID + 1):
        row = []
        for i in range(GRID + 1):
            x = (i / GRID - 0.5) * SIZE
            z = (j / GRID - 0.5) * SIZE
            if bump:
                r = np.hypot(x, z)
                push = 0.5 * r * np.exp(-(r / 0.12) ** 2 * 2.0) * np.clip(1.0 - (r / 0.2) ** 2, 0.0, 1.0) ** 2
                if r > 1e-9:
                    x += x / r * push
                    z += z / r * push
            row.append(bm.verts.new((x, 0.0, z)))
        verts.append(row)
    for j in range(GRID):
        for i in range(GRID):
            a, b, c, d = verts[j][i], verts[j][i + 1], verts[j + 1][i + 1], verts[j + 1][i]
            for face in (bm.faces.new((a, b, c)), bm.faces.new((a, c, d))):
                for loop in face.loops:
                    loop[uv_layer].uv = (0.5 + loop.vert.co.x / SIZE, 0.5 + loop.vert.co.z / SIZE)
    # UVs from the undistorted layout: the grid is regular in UV space.
    for j in range(GRID + 1):
        for i in range(GRID + 1):
            for loop in verts[j][i].link_loops:
                loop[uv_layer].uv = (i / GRID, j / GRID)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def positions(obj):
    obj.update_from_editmode()
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def rest_positions():
    x = (np.arange(GRID + 1) / GRID - 0.5) * SIZE
    return np.array([(xi, 0.0, zj) for zj in x for xi in x])


def unevenness(co):
    """Spread of the edge lengths along the grid rows: 0 for the undistorted sheet."""
    rows = co.reshape(GRID + 1, GRID + 1, 3)
    along = np.linalg.norm(rows[:, 1:] - rows[:, :-1], axis=2)
    across = np.linalg.norm(rows[1:] - rows[:-1], axis=2)
    return float(np.concatenate((along.ravel(), across.ravel())).std())


def stroke(env, points, passes=3):
    """Drag across ``points`` (window coordinates) a few times back and forth."""
    path = []
    for i in range(passes):
        path += points if i % 2 == 0 else points[::-1]
    env.event('MOUSEMOVE', 'NOTHING', path[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', path[0])
    for xy in path[1:]:
        env.event('MOUSEMOVE', 'NOTHING', xy)
        yield 0.03
    env.event('LEFTMOUSE', 'RELEASE', path[-1])
    yield 0.3


def line(env, a, b, steps=12):
    a, b = env.to_window(a), env.to_window(b)
    return [(round(a[0] + (b[0] - a[0]) * i / steps), round(a[1] + (b[1] - a[1]) * i / steps)) for i in range(steps + 1)]


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
    sheet = make_sheet("Sheet")
    bpy.context.view_layer.objects.active = sheet
    sheet.select_set(True)
    settings = bpy.context.scene.texture_relax_brush
    settings.radius = 200
    settings.strength = 1.0
    settings.use_pressure_strength = False
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.wm.tool_set_by_id(name="magic_fit.texture_relax_tool")
        bpy.ops.ed.undo_push(message="Test setup")
    env.rv3d.view_location = (0.0, 0.0, 0.0)
    env.rv3d.view_distance = 2.2
    yield 0.5

    tool = bpy.context.workspace.tools.from_space_view3d_mode('EDIT_MESH', create=False)
    check(tool is not None and tool.idname == "magic_fit.texture_relax_tool", "Texture Relax tool active in Edit Mode")
    check(operators.check_relax_ready(bpy.context) is None, "the brush is ready on a UV-mapped mesh")
    rest = rest_positions()
    start = positions(sheet)
    bump = np.linalg.norm(rest[:, [0, 2]], axis=1) < 0.2
    off_before = np.abs(start - rest).max(axis=1)[bump].max()
    check(off_before > 0.008, "the sheet starts with a stretched texture ({:.1f} mm)".format(off_before * 1000))

    env.event('MOUSEMOVE', 'NOTHING', env.to_window(Vector((0.0, 0.0, 0.0))))
    properties = next(a for a in env.window.screen.areas if a.type == 'PROPERTIES')
    properties.spaces.active.context = 'TOOL'
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.3
    env.screenshot("relax_01_before.png")
    env.screenshot("relax_00_window.png", whole_window=True)
    with env.override():
        bpy.ops.wm.call_panel(name="VIEW3D_PT_texture_relax_options", keep_open=True)
    yield 0.3
    env.screenshot("relax_00_popover.png", whole_window=True)
    env.event('ESC', 'PRESS', env.to_window(Vector((0.0, 0.0, 0.0))))
    env.event('ESC', 'RELEASE', env.to_window(Vector((0.0, 0.0, 0.0))))
    yield 0.2

    # --- A stroke across the middle evens the texture out and leaves the sheet flat.
    yield from stroke(env, line(env, Vector((-0.3, 0.0, 0.0)), Vector((0.3, 0.0, 0.0))), passes=6)
    env.screenshot("relax_02_after.png")
    after = positions(sheet)
    off_after = np.abs(after - rest).max(axis=1)[bump].max()
    check(off_after < 0.3 * off_before, "stroke: the bump went back toward its undistorted layout ({:.1f} -> {:.1f} mm)".format(
        off_before * 1000, off_after * 1000))
    check(unevenness(after) < 0.5 * unevenness(start), "stroke: the texture is more even ({:.4f} -> {:.4f})".format(
        unevenness(start), unevenness(after)))
    check(np.abs(after[:, 1]).max() == 0.0, "stroke: the sheet stayed flat")
    far = np.linalg.norm(start[:, [0, 2]], axis=1) > 0.5
    check(np.array_equal(after[far], start[far]), "stroke: vertices outside the brush untouched")
    edge = np.abs(np.abs(rest[:, 0]) - SIZE / 2) < 1e-6
    check(np.array_equal(after[edge], start[edge]), "stroke: the open edges outside the brush untouched")

    # --- Undo restores the exact positions.
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    sheet = bpy.data.objects["Sheet"]
    check(np.array_equal(positions(sheet), start), "undo restored every vertex exactly")

    # --- Escape cancels a stroke in progress.
    points = line(env, Vector((-0.3, 0.0, 0.0)), Vector((0.3, 0.0, 0.0)))
    env.event('MOUSEMOVE', 'NOTHING', points[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', points[0])
    for xy in points[1:]:
        env.event('MOUSEMOVE', 'NOTHING', xy)
        yield 0.03
    check(not np.array_equal(positions(sheet), start), "stroke in progress moved vertices")
    env.event('ESC', 'PRESS', points[-1])
    env.event('ESC', 'RELEASE', points[-1])
    env.event('LEFTMOUSE', 'RELEASE', points[-1])
    yield 0.3
    check(np.array_equal(positions(sheet), start), "escape restored every vertex exactly")

    # --- X mirror: a stroke on the right also relaxes the left.
    sheet.data.use_mirror_x = True
    yield from stroke(env, line(env, Vector((0.15, 0.0, -0.1)), Vector((0.15, 0.0, 0.1)), steps=6), passes=4)
    after = positions(sheet)
    moved = np.abs(after - start).max(axis=1) > 1e-6
    check(moved[start[:, 0] > 0.05].any() and moved[start[:, 0] < -0.05].any(),
          "mirror: vertices moved on both sides ({} right, {} left)".format(
              moved[start[:, 0] > 0.05].sum(), moved[start[:, 0] < -0.05].sum()))
    sheet.data.use_mirror_x = False
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    sheet = bpy.data.objects["Sheet"]

    # --- Hidden vertices never move.
    bm = bmesh.from_edit_mesh(sheet.data)
    bm.verts.ensure_lookup_table()
    for vert in bm.verts:
        if vert.co.z > 0.0:
            vert.hide = True
    bmesh.update_edit_mesh(sheet.data)
    yield from stroke(env, line(env, Vector((-0.3, 0.0, 0.0)), Vector((0.3, 0.0, 0.0))))
    after = positions(sheet)
    upper = start[:, 2] > 0.0
    check(np.array_equal(after[upper], start[upper]), "hidden: the hidden half did not move")
    check(not np.array_equal(after[~upper], start[~upper]), "hidden: the visible half was relaxed")
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    sheet = bpy.data.objects["Sheet"]

    # --- Without a UV map the brush says so and does nothing.
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    plain = make_sheet("Plain")
    plain.data.uv_layers.remove(plain.data.uv_layers[0])
    sheet.hide_set(True)
    sheet.select_set(False)
    plain.select_set(True)
    bpy.context.view_layer.objects.active = plain
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.ed.undo_push(message="Plain sheet")
    yield 0.3
    problem = operators.check_relax_ready(bpy.context)
    check(problem is not None and "no UV map" in problem, "no UV map: the cursor says why ({})".format(problem))
    plain_start = positions(plain)
    yield from stroke(env, line(env, Vector((-0.3, 0.0, 0.0)), Vector((0.3, 0.0, 0.0))), passes=1)
    check(np.array_equal(positions(bpy.data.objects["Plain"]), plain_start), "no UV map: nothing moved")
    env.screenshot("relax_03_no_uvs.png")

    # --- The sidebar button switches to Edit Mode and picks the brush.
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.wm.tool_set_by_id(name="builtin.select_box")
    yield 0.2
    sidebar = next(r for r in env.area.regions if r.type == 'UI')
    with bpy.context.temp_override(window=env.window, area=env.area, region=sidebar):
        bpy.ops.magic_fit.use_tool('EXEC_DEFAULT', True, tool='RELAX')
    yield 0.3
    tool = bpy.context.workspace.tools.from_space_view3d_mode('EDIT_MESH', create=False)
    check(bpy.context.mode == 'EDIT_MESH' and tool is not None and tool.idname == "magic_fit.texture_relax_tool",
          "sidebar button: Edit Mode with the Texture Relax brush")
    env.area.spaces.active.show_region_ui = True
    yield 0.3
    sidebar.active_panel_category = "Magic Fit"
    env.area.tag_redraw()
    yield 0.3
    env.screenshot("relax_04_sidebar.png")


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
