"""Interactive test for the Body Fit tool (Edit Mode), driven by simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_fit_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set).
"""

import math
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


def positions(obj):
    obj.update_from_editmode()
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def layer_radii(co, layer, band):
    """Radii of the inner layer's ``band`` vertices and of the outer vertices in the same directions."""
    inner = co[:layer][band]
    outer = co[layer:]
    unit = outer / np.linalg.norm(outer, axis=1, keepdims=True)
    match = np.argmax((inner / np.linalg.norm(inner, axis=1, keepdims=True)) @ unit.T, axis=1)
    return np.linalg.norm(inner, axis=1), np.linalg.norm(outer[match], axis=1)


def stroke(env, points, *, ctrl=False, passes=3):
    """Drag across ``points`` (window coordinates) a few times back and forth."""
    path = []
    for i in range(passes):
        path += points if i % 2 == 0 else points[::-1]
    env.event('MOUSEMOVE', 'NOTHING', path[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', path[0], ctrl=ctrl)
    for xy in path[1:]:
        env.event('MOUSEMOVE', 'NOTHING', xy, ctrl=ctrl)
        yield 0.03
    env.event('LEFTMOUSE', 'RELEASE', path[-1], ctrl=ctrl)
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
    body_mesh = bpy.data.meshes.new("Body")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
    bm.to_mesh(body_mesh)
    bm.free()
    body = bpy.data.objects.new("Body", body_mesh)
    bpy.context.scene.collection.objects.link(body)

    # A 3 cm thick garment whose inner layer sits 3 cm inside the body.
    garment_mesh = bpy.data.meshes.new("Garment")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=20, radius=0.97)
    bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=20, radius=1.00)
    bm.to_mesh(garment_mesh)
    bm.free()
    garment = bpy.data.objects.new("Garment", garment_mesh)
    bpy.context.scene.collection.objects.link(garment)
    layer = len(garment_mesh.vertices) // 2

    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)
    scene = bpy.context.scene
    scene.magic_fit.target = body
    settings = scene.body_fit_brush
    settings.mode = 'PUSH'
    settings.offset = 0.01
    settings.radius = 90
    settings.strength = 1.0
    settings.layer_radius = 0.2
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.wm.tool_set_by_id(name="magic_fit.body_fit_tool")
        bpy.ops.ed.undo_push(message="Test setup")
    env.rv3d.view_location = (0.0, 0.0, 0.0)
    env.rv3d.view_distance = 3.0
    yield 0.5

    tool = bpy.context.workspace.tools.from_space_view3d_mode('EDIT_MESH', create=False)
    check(tool is not None and tool.idname == "magic_fit.body_fit_tool", "Body Fit tool active in Edit Mode")
    start = positions(garment)
    directions = start[:layer] / np.linalg.norm(start[:layer], axis=1, keepdims=True)
    band = np.flatnonzero((directions[:, 1] < -0.9) & (np.abs(directions[:, 2]) < 0.15))
    check(len(band) > 0, "garment vertices under the stroke ({})".format(len(band)))
    inner, outer = layer_radii(start, layer, band)
    check(np.abs(outer - inner - 0.03).max() < 1e-4, "layers matched up by direction")

    env.event('MOUSEMOVE', 'NOTHING', env.to_window(Vector((0.0, -1.0, 0.0))))
    properties = next(a for a in env.window.screen.areas if a.type == 'PROPERTIES')
    properties.spaces.active.context = 'TOOL'
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.3
    env.screenshot("fit_01_before.png")
    env.screenshot("fit_00_window.png", whole_window=True)
    with env.override():
        bpy.ops.wm.call_panel(name="VIEW3D_PT_body_fit_options", keep_open=True)
    yield 0.3
    env.screenshot("fit_00_popover.png", whole_window=True)
    env.event('ESC', 'PRESS', env.to_window(Vector((0.0, -1.0, 0.0))))
    env.event('ESC', 'RELEASE', env.to_window(Vector((0.0, -1.0, 0.0))))
    yield 0.2

    # --- Push Out across the front.
    yield from stroke(env, line(env, Vector((-0.35, -1.0, 0.0)), Vector((0.35, -1.0, 0.0))))
    env.screenshot("fit_02_after_push.png")
    after = positions(garment)
    inner, outer = layer_radii(after, layer, band)
    check(inner.min() > 1.0 + 0.01 - 0.003, "push: inner layer cleared the body ({:.4f}..{:.4f})".format(inner.min(), inner.max()))
    check(np.abs(outer - inner - 0.03).max() < 0.002,
          "push: 3 cm thickness kept (max error {:.5f})".format(np.abs(outer - inner - 0.03).max()))
    back = directions[:, 1] > 0.3
    check(np.array_equal(after[:layer][back], start[:layer][back]), "push: back of the garment untouched")

    # --- Undo restores the exact positions.
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    garment = bpy.data.objects["Garment"]
    check(np.array_equal(positions(garment), start), "undo restored every vertex exactly")

    # --- Escape cancels a stroke in progress.
    points = line(env, Vector((-0.35, -1.0, 0.0)), Vector((0.35, -1.0, 0.0)))
    env.event('MOUSEMOVE', 'NOTHING', points[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', points[0])
    for xy in points[1:]:
        env.event('MOUSEMOVE', 'NOTHING', xy)
        yield 0.03
    check(not np.array_equal(positions(garment), start), "stroke in progress moved vertices")
    env.event('ESC', 'PRESS', points[-1])
    env.event('ESC', 'RELEASE', points[-1])
    env.event('LEFTMOUSE', 'RELEASE', points[-1])
    yield 0.3
    check(np.array_equal(positions(garment), start), "escape restored every vertex exactly")

    # --- Tighten a garment that floats 3 cm off the body: with the mode setting, then with Ctrl.
    bm = bmesh.from_edit_mesh(garment.data)
    for vert in bm.verts:
        vert.co *= 1.03 / 0.97
    bmesh.update_edit_mesh(garment.data)
    with env.override():
        bpy.ops.ed.undo_push(message="Floating garment")
    floating = positions(garment)
    for ctrl in (False, True):
        bpy.context.scene.body_fit_brush.mode = 'PUSH' if ctrl else 'TIGHTEN'
        yield from stroke(env, line(env, Vector((-0.35, -1.06, 0.0)), Vector((0.35, -1.06, 0.0))), ctrl=ctrl, passes=5)
        how = "ctrl + push mode" if ctrl else "tighten mode"
        inner, outer = layer_radii(positions(garment), layer, band)
        check(np.abs(inner - 1.01).max() < 0.003,
              "{}: pulled in to the offset ({:.4f}..{:.4f})".format(how, inner.min(), inner.max()))
        check(np.abs(outer - inner - 0.03 * 1.03 / 0.97).max() < 0.002, "{}: thickness kept".format(how))
        with env.override():
            bpy.ops.ed.undo()
        yield 0.3
        garment = bpy.data.objects["Garment"]
        check(np.array_equal(positions(garment), floating), "{}: undone".format(how))
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    garment = bpy.data.objects["Garment"]
    check(np.array_equal(positions(garment), start), "back to the original garment")
    settings = bpy.context.scene.body_fit_brush
    settings.mode = 'PUSH'

    # --- X mirror: painting the right side also fits the left side.
    garment.data.use_mirror_x = True
    yield from stroke(env, line(env, Vector((0.55, -0.83, 0.0)), Vector((0.55, -0.83, 0.2))), passes=4)
    after = positions(garment)
    radius = np.linalg.norm(after[:layer], axis=1)
    right = (directions[:, 0] > 0.5) & (directions[:, 1] < -0.6) & (np.abs(directions[:, 2]) < 0.1)
    left = (directions[:, 0] < -0.5) & (directions[:, 1] < -0.6) & (np.abs(directions[:, 2]) < 0.1)
    check(radius[right].max() > 1.005 and radius[left].max() > 1.005, "mirror: both sides pushed out")
    env.screenshot("fit_03_mirror.png")
    garment.data.use_mirror_x = False
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    garment = bpy.data.objects["Garment"]

    # --- Hidden vertices never move.
    bm = bmesh.from_edit_mesh(garment.data)
    bm.verts.ensure_lookup_table()
    for vert in bm.verts[layer:]:
        vert.hide = True
    bmesh.update_edit_mesh(garment.data)
    yield from stroke(env, line(env, Vector((-0.35, -1.0, 0.0)), Vector((0.35, -1.0, 0.0))))
    after = positions(garment)
    check(np.array_equal(after[layer:], start[layer:]), "hidden outer layer did not move")
    check(not np.array_equal(after[:layer], start[:layer]), "visible inner layer was fitted")

    # --- Shape keys: the brush sculpts the active key and leaves the basis alone.
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    keyed_mesh = bpy.data.meshes.new("Keyed")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=20, radius=0.97)
    bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=20, radius=1.00)
    bm.to_mesh(keyed_mesh)
    bm.free()
    keyed = bpy.data.objects.new("Keyed", keyed_mesh)
    bpy.context.scene.collection.objects.link(keyed)
    garment.hide_set(True)
    basis = keyed.shape_key_add(name="Basis")
    snug = keyed.shape_key_add(name="Snug")
    for point in snug.data:
        point.co *= 0.98
    keyed.active_shape_key_index = 1
    basis_start = np.array([point.co[:] for point in basis.data])
    garment.select_set(False)
    keyed.select_set(True)
    bpy.context.view_layer.objects.active = keyed
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.ed.undo_push(message="Keyed garment")
    yield 0.3

    def edited_shape():
        bm = bmesh.from_edit_mesh(bpy.data.objects["Keyed"].data)
        return np.array([vert.co[:] for vert in bm.verts])

    snug_start = edited_shape()
    points = line(env, Vector((-0.35, -1.0, 0.0)), Vector((0.35, -1.0, 0.0)))
    env.event('MOUSEMOVE', 'NOTHING', points[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', points[0])
    for xy in points[1:]:
        env.event('MOUSEMOVE', 'NOTHING', xy)
        yield 0.03
    env.event('ESC', 'PRESS', points[-1])
    env.event('ESC', 'RELEASE', points[-1])
    env.event('LEFTMOUSE', 'RELEASE', points[-1])
    yield 0.3
    check(np.array_equal(edited_shape(), snug_start), "shape keys: escape restored the active key exactly")
    yield from stroke(env, points)
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    yield 0.2
    keys = bpy.data.objects["Keyed"].data.shape_keys.key_blocks
    basis_after = np.array([point.co[:] for point in keys["Basis"].data])
    snug_after = np.array([point.co[:] for point in keys["Snug"].data])
    check(np.array_equal(basis_after, basis_start), "shape keys: basis untouched")
    inner, outer = layer_radii(snug_after, layer, band)
    check(inner.min() > 1.0 + 0.01 - 0.003 and np.abs(outer - inner - 0.03 * 0.98).max() < 0.002,
          "shape keys: active key fitted with its thickness ({:.4f}..{:.4f})".format(inner.min(), inner.max()))

    # --- Several objects in Edit Mode are fitted together, and the seam between them stays closed.
    halves = []
    for name, keep in (("Left", lambda x: x <= 1e-6), ("Right", lambda x: x >= -1e-6)):
        half_mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=20, radius=0.97)
        bmesh.ops.delete(bm, geom=[vert for vert in bm.verts if not keep(vert.co.x)], context='VERTS')
        bm.to_mesh(half_mesh)
        bm.free()
        half = bpy.data.objects.new(name, half_mesh)
        bpy.context.scene.collection.objects.link(half)
        halves.append(half)
    # The right half sits 0.05 mm off: close enough for Merge by Distance to join the seam.
    halves[1].location.x = 5e-5
    bpy.data.objects["Keyed"].hide_set(True)
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj in halves)
    bpy.context.view_layer.objects.active = halves[0]
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.ed.undo_push(message="Two halves")
    yield 0.3

    def world_positions(obj):
        return positions(obj) @ np.array(obj.matrix_world)[:3, :3].T + np.array(obj.matrix_world)[:3, 3]

    def seam_pairs(left, right):
        on_seam = np.flatnonzero(np.abs(left[:, 0]) < 1e-5)
        pairs = [(i, int(np.argmin(np.linalg.norm(right - left[i], axis=1)))) for i in on_seam]
        return np.array(pairs)

    def seam_gaps(left, right, pairs):
        return np.linalg.norm(left[pairs[:, 0]] - right[pairs[:, 1]], axis=1)

    halves_start = [world_positions(obj) for obj in halves]
    pairs = seam_pairs(*halves_start)
    check(len(pairs) > 10, "halves: seam vertices matched up ({})".format(len(pairs)))
    settings = bpy.context.scene.body_fit_brush
    settings.radius = 90
    yield from stroke(env, line(env, Vector((-0.35, -1.0, 0.0)), Vector((0.35, -1.0, 0.0))))
    halves_after = [world_positions(bpy.data.objects[name]) for name in ("Left", "Right")]
    for name, points in zip(("Left", "Right"), halves_after):
        unit = points / np.linalg.norm(points, axis=1, keepdims=True)
        front = (unit[:, 1] < -0.95) & (np.abs(unit[:, 2]) < 0.1)
        check(front.any() and np.linalg.norm(points[front], axis=1).min() > 1.0 + 0.01 - 0.003,
              "halves: the {} half was fitted too".format(name.lower()))
    gap_change = np.abs(seam_gaps(*halves_after, pairs) - seam_gaps(*halves_start, pairs)).max()
    front_seam = np.linalg.norm(halves_after[0][pairs[:, 0]] - halves_start[0][pairs[:, 0]], axis=1).max()
    check(front_seam > 0.01, "halves: the seam was moved ({:.4f})".format(front_seam))
    check(gap_change < 1e-6, "halves: the seam between the objects stayed closed (change {:.1e})".format(gap_change))
    env.screenshot("fit_04_halves.png")
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    undone = [world_positions(bpy.data.objects[name]) for name in ("Left", "Right")]
    check(all(np.abs(a - b).max() < 1e-6 for a, b in zip(undone, halves_start)), "halves: undo restored both objects")
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    for name in ("Left", "Right"):
        bpy.data.objects[name].hide_set(True)

    # --- Auto Smooth: a bumpy garment clear of the body only changes when it's on.
    bumpy_mesh = bpy.data.meshes.new("Bumpy")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.03)
    for i, vert in enumerate(bm.verts):
        vert.co *= 1.0 + 0.004 * math.sin(i * 12.9898)
    bm.to_mesh(bumpy_mesh)
    bm.free()
    bumpy = bpy.data.objects.new("Bumpy", bumpy_mesh)
    bpy.context.scene.collection.objects.link(bumpy)
    bpy.data.objects["Keyed"].hide_set(True)
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    bumpy.select_set(True)
    bpy.context.view_layer.objects.active = bumpy
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.ed.undo_push(message="Bumpy garment")
    yield 0.3

    def bumpy_state():
        bm = bmesh.from_edit_mesh(bpy.data.objects["Bumpy"].data)
        return np.array([vert.co[:] for vert in bm.verts])

    def bumpiness(points):
        radius = np.linalg.norm(points, axis=1)
        mesh = bpy.data.objects["Bumpy"].data
        pairs = np.array([edge.vertices[:] for edge in mesh.edges])
        total = np.zeros(len(radius))
        count = np.zeros(len(radius))
        np.add.at(total, pairs[:, 0], radius[pairs[:, 1]])
        np.add.at(total, pairs[:, 1], radius[pairs[:, 0]])
        np.add.at(count, pairs[:, 0], 1.0)
        np.add.at(count, pairs[:, 1], 1.0)
        unit = points / radius[:, None]
        band = (unit[:, 1] < -0.9) & (np.abs(unit[:, 2]) < 0.15)
        return np.abs(radius - total / np.maximum(count, 1.0))[band].mean()

    bumpy_start = bumpy_state()
    settings = bpy.context.scene.body_fit_brush
    settings.use_auto_smooth = False
    yield from stroke(env, line(env, Vector((-0.35, -1.03, 0.0)), Vector((0.35, -1.03, 0.0))))
    check(np.array_equal(bumpy_state(), bumpy_start), "auto smooth off: a garment clear of the body is left alone")
    settings.use_auto_smooth = True
    settings.auto_smooth = 1.0
    settings.radius = 250  # several rings of this coarse sphere, not just the one being measured
    yield from stroke(env, line(env, Vector((-0.35, -1.03, 0.0)), Vector((0.35, -1.03, 0.0))))
    smoothed = bumpy_state()
    before, after = bumpiness(bumpy_start), bumpiness(smoothed)
    check(after < 0.5 * before, "auto smooth on: bumps under the stroke evened out ({:.5f} -> {:.5f})".format(before, after))
    closest = np.linalg.norm(smoothed, axis=1).min() - 1.0
    check(closest > 0.01 - 1e-4, "auto smooth on: never closer to the body than the offset ({:.4f})".format(closest))
    env.screenshot("fit_05_auto_smooth.png")
    env.screenshot("fit_06_window.png", whole_window=True)


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
