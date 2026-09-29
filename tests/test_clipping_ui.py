"""Interactive test for the clipping overlay: marks where the edited meshes clip into the body.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_clipping_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set).
"""

import os
import sys
import tempfile
import traceback

import bmesh
import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import magic_fit  # noqa: E402
from magic_fit import clipping  # noqa: E402

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

    def pixels(self):
        """The 3D view as an (h, w, 4) array."""
        path = os.path.join(tempfile.gettempdir(), "mf_clipping_probe.png")
        with bpy.context.temp_override(window=self.window, area=self.area):
            bpy.ops.screen.screenshot_area(filepath=path)
        image = bpy.data.images.load(path)
        data = np.array(image.pixels[:]).reshape(image.size[1], image.size[0], 4)
        bpy.data.images.remove(image)
        return data

    def redraw(self):
        for area in self.window.screen.areas:
            area.tag_redraw()


def marked_pixels(pixels):
    """Pixels tinted like the (magenta) clipping marks."""
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    return int(((r - g > 0.25) & (b - g > 0.2)).sum())


def sphere(name, radius, segments=48, rings=24):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def tube(name, zs, radius, columns=48):
    """An open tube around the Z axis: ring i at height zs[i], with radius(x, y, z) for its vertices."""
    co = []
    for z in zs:
        for j in range(columns):
            a = 2.0 * np.pi * j / columns
            x, y = np.cos(a), np.sin(a)
            r = radius(x, y, z)
            co.append((r * x, r * y, z))
    faces = [(i * columns + j, i * columns + (j + 1) % columns, (i + 1) * columns + (j + 1) % columns, (i + 1) * columns + j)
             for i in range(len(zs) - 1) for j in range(columns)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(co, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def report():
    result = clipping.status(bpy.context)
    return {name: (inside, close, measuring) for name, inside, close, measuring in (result or [])}


def wait_measured(limit=40):
    for _ in range(limit):
        result = clipping.status(bpy.context)
        if result is not None and not any(measuring for *_rest, measuring in result):
            return
        yield 0.1


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
    body = sphere("Body", 1.0)
    # A garment clear of the body, except for a dent at the front that sinks 3 cm into it.
    garment = sphere("Garment", 1.02, 64, 32)
    for vert in garment.data.vertices:
        if vert.co.normalized().y < -0.9:
            vert.co *= 0.97 / 1.02
    dent = sum(1 for vert in garment.data.vertices if vert.co.length < 0.99)
    scene = bpy.context.scene
    scene.magic_fit.target = body
    body.select_set(False)
    garment.select_set(True)
    bpy.context.view_layer.objects.active = garment
    with env.override():
        bpy.ops.view3d.view_axis(type='FRONT')
    env.rv3d.view_location = (0.0, 0.0, 0.0)
    env.rv3d.view_distance = 3.0
    env.redraw()
    yield 0.5
    check(not scene.magic_fit_clipping.show and clipping.status(bpy.context) is None
          and marked_pixels(env.pixels()) < 20, "clipping marks are off by default")
    scene.magic_fit_clipping.show = True
    with env.override():
        # The scene was built through bpy.data, which records no undo step of its own.
        bpy.ops.ed.undo_push(message="Test setup")
    env.redraw()
    yield 0.5
    yield from wait_measured()

    # --- Object Mode: the selected garment is checked, and its dent is marked on the body.
    status = report()
    check(status.get("Garment", (0,))[0] == dent,
          "object mode: the dent's {} vertices clip, and nothing else ({})".format(dent, status))
    check("Body" not in status, "the body itself isn't checked")
    shown = env.pixels()
    check(marked_pixels(shown) > 200, "the dent is marked in the viewport ({} pixels)".format(marked_pixels(shown)))
    env.screenshot("clip_01_object_mode.png")
    env.screenshot("clip_02_window.png", whole_window=True)

    scene.magic_fit_clipping.show = False
    env.redraw()
    yield 0.3
    check(marked_pixels(env.pixels()) < 20, "turned off: no marks")
    scene.magic_fit_clipping.show = True
    env.redraw()
    yield 0.3
    yield from wait_measured()

    garment.select_set(False)
    env.redraw()
    yield 0.3
    check(report() == {}, "object mode: unselected meshes aren't checked")
    garment.select_set(True)

    # --- Margin: parts outside the body but close to it are marked too (fainter).
    scene.magic_fit_clipping.margin = 0.03
    env.redraw()
    yield 0.3
    status = report()
    check(status["Garment"][0] == dent and status["Garment"][1] > 100,
          "margin: close parts counted separately ({})".format(status["Garment"]))
    env.screenshot("clip_03_margin.png")
    scene.magic_fit_clipping.margin = 0.0

    # --- Edit Mode: fitting the dent out with Body Fit clears the marks as it goes.
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.wm.tool_set_by_id(name="magic_fit.body_fit_tool")
    settings = scene.body_fit_brush
    settings.mode = 'PUSH'  # only the dent moves
    settings.offset = 0.002
    settings.radius = 200
    settings.strength = 1.0
    settings.layer_radius = 0.05
    env.redraw()
    yield 0.3
    yield from wait_measured()
    check(report().get("Garment", (0,))[0] == dent, "edit mode: the same vertices clip")
    # Zigzag over the dent a few times.
    path = []
    for i in range(4):
        for z in (0.3, 0.15, 0.0, -0.15, -0.3):
            ends = [env.to_window(Vector((x, -1.0, z))) for x in (-0.35, 0.35)]
            path += ends if (len(path) // 2) % 2 == 0 else ends[::-1]
    env.event('MOUSEMOVE', 'NOTHING', path[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', path[0])
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0.0, 1.0, 6)[1:]:
            env.event('MOUSEMOVE', 'NOTHING', (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)))
            yield 0.02
    env.event('LEFTMOUSE', 'RELEASE', path[-1])
    yield 0.3
    yield from wait_measured()
    status = report()
    check(status.get("Garment", (None,))[0] == 0, "edit mode: after fitting, nothing clips ({})".format(status))
    check(marked_pixels(env.pixels()) < 20, "edit mode: the marks are gone")
    env.screenshot("clip_04_fitted.png")
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    env.redraw()
    yield 0.2
    yield from wait_measured()
    check(report().get("Garment", (0,))[0] == dent, "edit mode: undo brings the marks back")
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    yield 0.2

    # --- Weight Paint with a posed armature: scaling a bone down pulls the garment into the body.
    scene = bpy.context.scene  # undo reloaded it
    garment = bpy.data.objects["Garment"]
    for vert in garment.data.vertices:
        vert.co = vert.co.normalized() * 1.02
    garment.data.update()
    rig = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
    bone = rig.data.edit_bones.new("Root")
    bone.head, bone.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 0.3)
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    garment.vertex_groups.new(name="Root").add(list(range(len(garment.data.vertices))), 1.0, 'REPLACE')
    garment.modifiers.new("Armature", 'ARMATURE').object = rig
    rig.select_set(False)
    garment.select_set(True)
    bpy.context.view_layer.objects.active = garment
    with env.override():
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    env.redraw()
    yield 0.3
    yield from wait_measured()
    check(report().get("Garment", (None,))[0] == 0, "weight paint: at rest nothing clips")
    rig.pose.bones["Root"].scale = (0.95, 0.95, 0.95)
    bpy.context.view_layer.update()
    env.redraw()
    yield 0.3
    yield from wait_measured()
    status = report()
    check(status.get("Garment", (0,))[0] == len(garment.data.vertices),
          "weight paint: scaled down by the bone, all of it clips ({})".format(status))
    env.screenshot("clip_05_weight_paint.png")
    rig.pose.bones["Root"].scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()
    env.redraw()
    yield 0.3
    yield from wait_measured()
    check(report().get("Garment", (None,))[0] == 0, "weight paint: back at rest, the marks are gone")

    # --- Pose Mode: with the garment selected along with the armature, scaling bones shows the marks.
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    garment.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    with env.override():
        bpy.ops.object.mode_set(mode='POSE')
    rig.pose.bones["Root"].scale = (0.95, 0.95, 0.95)
    bpy.context.view_layer.update()
    env.redraw()
    yield 0.3
    yield from wait_measured()
    status = report()
    check(bpy.context.mode == 'POSE' and status.get("Garment", (0,))[0] == len(garment.data.vertices),
          "pose mode: scaling the bone marks the selected garment ({})".format(status))
    env.screenshot("clip_06_pose_mode.png")
    rig.pose.bones["Root"].scale = (1.0, 1.0, 1.0)
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = garment
    rig.select_set(False)
    with env.override():
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

    # --- The sidebar panel lists the result.
    space = env.area.spaces.active
    space.show_region_ui = True
    rig.pose.bones["Root"].scale = (0.99, 0.99, 0.99)
    bpy.context.view_layer.update()
    env.redraw()
    yield 0.5
    yield from wait_measured()
    with env.override():
        region = next(r for r in env.area.regions if r.type == 'UI')
        with bpy.context.temp_override(window=env.window, area=env.area, region=region):
            bpy.context.region.active_panel_category = "Magic Fit"
    bpy.context.workspace.magic_fit_tab = 'FIT'  # the Clipping panel's tab
    env.redraw()
    yield 0.5
    env.screenshot("clip_07_panel.png", whole_window=True)

    # --- An open body (a torso on its own, widening toward its waist edge at z 0): a skirt narrowing
    # below that edge isn't marked, only a band dented into the torso above it.
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    torso = tube("Torso", np.linspace(0.6, 0.0, 13), lambda x, y, z: 0.2 + 0.1 * (0.6 - z) / 0.6)

    def skirt_radius(x, y, z):
        if y < -0.9 and 0.15 <= z <= 0.3:
            return 0.24  # 1 to 3.5 cm inside the torso
        return 0.32 if z >= 0.0 else 0.26  # narrower than the waist edge below it

    skirt = tube("Skirt", np.linspace(0.3, -0.6, 19), skirt_radius)
    dent = sum(1 for vert in skirt.data.vertices if np.hypot(vert.co.x, vert.co.y) < 0.25)
    scene = bpy.context.scene
    scene.magic_fit.target = torso
    skirt.select_set(True)
    bpy.context.view_layer.objects.active = skirt
    env.rv3d.view_location = (0.0, 0.0, 0.0)
    env.rv3d.view_distance = 2.0
    env.redraw()
    yield 0.5
    yield from wait_measured()
    status = report()
    check(dent > 0 and status.get("Skirt", (0,))[0] == dent,
          "open body: only the {} dented vertices clip, not the skirt below its edge ({})".format(dent, status))
    env.screenshot("clip_08_open_body.png")


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
