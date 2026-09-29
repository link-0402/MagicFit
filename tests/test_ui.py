"""Interactive test: drives the real tool with simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set).
"""

import math
import os
import sys
import traceback

import bmesh
import bpy
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


def sphere(name, radius):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=radius)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def weights(obj, group_name):
    index = obj.vertex_groups[group_name].index
    out = {}
    for v in obj.data.vertices:
        for elem in v.groups:
            if elem.group == index:
                out[v.index] = elem.weight
    return out


def max_edge_jump(obj, group_name, verts):
    """Largest difference of a group's weights across an edge of ``obj`` between two of ``verts``."""
    w = weights(obj, group_name)
    return max(
        abs(w.get(a, 0.0) - w.get(b, 0.0))
        for a, b in (edge.vertices for edge in obj.data.edges) if a in verts and b in verts
    )


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


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0

    # Close the startup splash screen, it would swallow the first click.
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.event('MOUSEMOVE', 'NOTHING', middle)
    env.event('ESC', 'PRESS', middle)
    env.event('ESC', 'RELEASE', middle)
    yield 0.3

    # --- Scene: body-like inner sphere (target) and a slightly larger "garment" sphere (painted).
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    target = sphere("Body", 1.0)
    top = target.vertex_groups.new(name="Top")
    bottom = target.vertex_groups.new(name="Bottom")
    # Ends in a hard edge at the equator.
    upper = target.vertex_groups.new(name="Upper")
    for v in target.data.vertices:
        ramp = (v.co.z + 1.0) / 2.0
        top.add([v.index], ramp, 'REPLACE')
        bottom.add([v.index], 1.0 - ramp, 'REPLACE')
        upper.add([v.index], 1.0 if v.co.z > 1e-6 else 0.0, 'REPLACE')

    garment = sphere("Garment", 1.05)
    garment.vertex_groups.new(name="Bottom").add(list(range(len(garment.data.vertices))), 1.0, 'REPLACE')

    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)
    with env.override():
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.wm.tool_set_by_id(name="magic_fit.tool")
        # The scene was built through bpy.data, which records no undo step of its own.
        bpy.ops.ed.undo_push(message="Test setup")
    env.rv3d.view_location = (0.0, 0.0, 0.0)
    env.rv3d.view_distance = 3.0
    yield 0.5

    tool = bpy.context.workspace.tools.from_space_view3d_mode('PAINT_WEIGHT', create=False)
    check(tool is not None and tool.idname == "magic_fit.tool", "tool is active in weight paint mode")

    settings = bpy.context.scene.magic_fit
    settings.radius = 60
    settings.strength = 1.0
    settings.falloff = 'CONSTANT'
    settings.use_pressure_strength = False

    # --- No target yet: the stroke must be refused without touching anything.
    center = env.to_window(Vector((0.0, -1.05, 0.0)))
    env.event('MOUSEMOVE', 'NOTHING', center)
    yield 0.3
    env.screenshot("01_no_target.png")
    env.event('LEFTMOUSE', 'PRESS', center)
    env.event('LEFTMOUSE', 'RELEASE', center)
    yield 0.3
    check("Top" not in garment.vertex_groups, "no target: nothing was painted")

    settings.target = target
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.2

    # --- Stroke across the front of the garment at the equator.
    left = env.to_window(Vector((-0.6, -1.05, 0.0)))
    right = env.to_window(Vector((0.6, -1.05, 0.0)))
    env.event('MOUSEMOVE', 'NOTHING', left)
    yield 0.2
    env.screenshot("02_hover.png")
    env.screenshot("03_window.png", whole_window=True)
    env.event('LEFTMOUSE', 'PRESS', left)
    for i in range(1, 11):
        t = i / 10.0
        env.event('MOUSEMOVE', 'NOTHING', (round(left[0] + (right[0] - left[0]) * t), left[1]))
        yield 0.05
    env.event('LEFTMOUSE', 'RELEASE', right)
    yield 0.5
    env.screenshot("04_after_stroke.png")

    check("Top" in garment.vertex_groups, "stroke created the missing 'Top' group")
    check(garment.vertex_groups.active.name == "Bottom", "active group unchanged by the stroke")
    painted_top = weights(garment, "Top") if "Top" in garment.vertex_groups else {}
    bottom_w = weights(garment, "Bottom")
    band = [v.index for v in garment.data.vertices
            if abs(v.co.z) < 0.1 and abs(v.co.x) < 0.3 and v.co.y < -0.9]
    check(len(band) > 0, "found garment vertices under the stroke ({})".format(len(band)))
    worst = 0.0
    for vi in band:
        z = garment.data.vertices[vi].co.z
        expected = (z / 1.05 + 1.0) / 2.0
        worst = max(worst, abs(painted_top.get(vi, 0.0) - expected), abs(bottom_w.get(vi, 0.0) - (1.0 - expected)))
    check(worst < 0.02, "weights under the stroke match the target (max error {:.4f})".format(worst))
    back = [v.index for v in garment.data.vertices if v.co.y > 0.5]
    check(all(vi not in painted_top for vi in back), "back of the garment untouched")
    check(all(abs(bottom_w[vi] - 1.0) < 1e-6 for vi in back), "back 'Bottom' weights untouched")

    # --- Undo restores the previous state.
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    garment = bpy.data.objects["Garment"]
    check("Top" not in garment.vertex_groups, "undo removed the created group")
    check(all(abs(w - 1.0) < 1e-6 for w in weights(garment, "Bottom").values()), "undo restored weights")

    # --- Auto Smooth: copied as is, the body's hard edge in Upper lies between two rows of the garment.
    # With Auto Smooth, the weights change gradually across it instead. Each stroke is undone again.
    rows = {v.index for v in garment.data.vertices if v.co.y < -0.8 and abs(v.co.x) < 0.35 and abs(v.co.z) < 0.2}
    jumps = {}
    for smooth in (False, True):
        settings = bpy.context.scene.magic_fit
        settings.target = bpy.data.objects["Body"]
        settings.radius, settings.strength, settings.falloff = 150, 1.0, 'CONSTANT'
        settings.use_pressure_strength = False
        settings.use_auto_smooth = smooth
        settings.auto_smooth = 1.0
        env.event('MOUSEMOVE', 'NOTHING', left)
        yield 0.2
        if smooth:
            env.screenshot("04c_auto_smooth_header.png", whole_window=True)
        env.event('LEFTMOUSE', 'PRESS', left)
        for i in range(1, 11):
            t = i / 10.0
            env.event('MOUSEMOVE', 'NOTHING', (round(left[0] + (right[0] - left[0]) * t), left[1]))
            yield 0.05
        env.event('LEFTMOUSE', 'RELEASE', right)
        yield 0.5
        if "Upper" not in garment.vertex_groups:
            check(False, "the stroke copied Upper")
            break
        # Show Upper for the screenshot.
        garment.vertex_groups.active_index = garment.vertex_groups["Upper"].index
        yield 0.2
        env.screenshot("04d_auto_smooth.png" if smooth else "04b_hard_edge.png")
        jumps[smooth] = max_edge_jump(garment, "Upper", rows)
        check(all(abs(w - 1.0) < 1e-6 for vi, w in weights(garment, "Bottom").items() if vi in back),
              "back of the garment untouched ({})".format("Auto Smooth" if smooth else "hard edge"))
        with env.override():
            bpy.ops.ed.undo()
        yield 0.3
        garment = bpy.data.objects["Garment"]
    check(len(jumps) == 2 and jumps[False] > 0.8 and jumps[True] < 0.6 * jumps[False],
          "Auto Smooth spread out the hard edge (largest step between rows {:.2f}, {:.2f} without)".format(
              jumps.get(True, 1.0), jumps.get(False, 0.0)))
    check("Upper" not in garment.vertex_groups, "undo removed the strokes")

    # --- Escape cancels a stroke in progress.
    # Undo also reverted the target setting, and reloaded every data-block.
    settings = bpy.context.scene.magic_fit
    check(settings.target is None, "undo reverted the target setting")
    settings.target = bpy.data.objects["Body"]
    env.event('MOUSEMOVE', 'NOTHING', left)
    yield 0.1
    env.event('LEFTMOUSE', 'PRESS', left)
    env.event('MOUSEMOVE', 'NOTHING', right)
    yield 0.2
    check("Top" in garment.vertex_groups, "stroke in progress painted")
    env.event('ESC', 'PRESS', right)
    env.event('ESC', 'RELEASE', right)
    env.event('LEFTMOUSE', 'RELEASE', right)
    yield 0.3
    check("Top" not in garment.vertex_groups, "escape removed the group again")
    check(all(abs(w - 1.0) < 1e-6 for w in weights(garment, "Bottom").values()), "escape restored weights")

    # --- Current Group mode with X mirror: only the active group changes, on both sides.
    settings.group_mode = 'ACTIVE'
    garment.data.use_mirror_x = True
    spot = env.to_window(Vector((0.5, -0.9, 0.3)))
    env.event('MOUSEMOVE', 'NOTHING', spot)
    yield 0.1
    env.event('LEFTMOUSE', 'PRESS', spot)
    env.event('LEFTMOUSE', 'RELEASE', spot)
    yield 0.3
    bottom_w = weights(garment, "Bottom")
    changed = [vi for vi, w in bottom_w.items() if abs(w - 1.0) > 1e-3]
    xs = [garment.data.vertices[vi].co.x for vi in changed]
    check("Top" not in garment.vertex_groups, "Current Group mode left other groups alone")
    check(any(x > 0.2 for x in xs) and any(x < -0.2 for x in xs), "X mirror painted both sides")
    env.screenshot("05_mirror.png")

    # --- F key: the radial control edits the brush radius.
    env.event('MOUSEMOVE', 'NOTHING', center)
    yield 0.1
    env.event('F', 'PRESS', center)
    env.event('F', 'RELEASE', center)
    yield 0.2
    env.event('MOUSEMOVE', 'NOTHING', (center[0] + 150, center[1]))
    yield 0.2
    env.screenshot("06_radial.png")
    env.event('LEFTMOUSE', 'PRESS', (center[0] + 150, center[1]))
    env.event('LEFTMOUSE', 'RELEASE', (center[0] + 150, center[1]))
    yield 0.3
    check(bpy.context.scene.magic_fit.radius != 60,
          "F changed the radius to {}".format(bpy.context.scene.magic_fit.radius))
    bpy.context.scene.magic_fit.radius = 60

    # --- Active Tool panel (Properties editor, Tool tab) and the header's Options popover.
    properties = next(a for a in env.window.screen.areas if a.type == 'PROPERTIES')
    properties.spaces.active.context = 'TOOL'
    properties.tag_redraw()
    yield 0.4
    env.screenshot("07_active_tool_panel.png", whole_window=True)
    with env.override():
        bpy.ops.wm.call_panel(name="VIEW3D_PT_magic_fit_options", keep_open=True)
    yield 0.4
    env.screenshot("08_options_popover.png", whole_window=True)
    env.event('ESC', 'PRESS', center)
    env.event('ESC', 'RELEASE', center)
    yield 0.2

    # --- Save and reopen: the tool, its settings and its brush cursor must come back.
    path = os.path.join(bpy.app.tempdir, "magic_fit_test.blend")
    with env.override():
        bpy.ops.wm.save_as_mainfile(filepath=path)
        bpy.ops.wm.open_mainfile(filepath=path)
    yield 0.5
    env = Env()
    tool = bpy.context.workspace.tools.from_space_view3d_mode('PAINT_WEIGHT', create=False)
    check(bpy.context.mode == 'PAINT_WEIGHT' and tool is not None and tool.idname == "magic_fit.tool",
          "reopened file is in weight paint mode with the tool active")
    check(bpy.context.scene.magic_fit.target == bpy.data.objects["Body"], "target saved with the file")
    from bl_ui.space_toolsystem_common import _activate_by_item
    check(len(_activate_by_item._cursor_draw_handle) > 0, "brush cursor callback installed after reopening")
    env.event('MOUSEMOVE', 'NOTHING', center)
    yield 0.3
    env.screenshot("09_reopened.png")


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

    # Persistent, so the test keeps running across the file reload.
    bpy.app.timers.register(tick, first_interval=1.5, persistent=True)


run()
