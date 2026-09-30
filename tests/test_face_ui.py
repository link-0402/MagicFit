"""Interactive test for Face weights: the Face Weights button on the sidebar's Face sub-tab, a test pose and
the Weight Brushes' Face mode, driven by simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_face_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set).
"""

import os
import sys
import traceback

import bpy
import numpy as np
from bpy_extras import view3d_utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import face  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import facing, operators  # noqa: E402

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

    def screenshot(self, name):
        if SHOTS:
            with bpy.context.temp_override(window=self.window, area=self.area):
                bpy.ops.screen.screenshot(filepath=os.path.join(SHOTS, name))

    def redraw(self):
        for area in self.window.screen.areas:
            area.tag_redraw()


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


def open_area(skin, eyeball):
    """How much of ``eyeball`` shows through ``skin`` as displayed, in mm²."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    posed = {}
    for obj in (skin, eyeball):
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        posed[obj.name] = np.array([evaluated.matrix_world @ v.co for v in mesh.vertices])
        evaluated.to_mesh_clear()
    points = posed[eyeball.name]
    eye = facing.Eye(points.mean(0), float(np.linalg.norm(points - points.mean(0), axis=1).mean()), points,
                     facing.mesh_tris(eyeball.data), True)
    return facing.open_area(facing.bvh_of(posed[skin.name], facing.mesh_tris(skin.data)), eye)


def lid_share(skin, verts, side='l'):
    names = [g.name for g in skin.vertex_groups]
    weights = face.weights_of(skin, names)
    columns = [i for i, n in enumerate(names) if n.startswith(facing.UPPER_LID) and n.endswith("_" + side)]
    return float(weights[verts][:, columns].sum(1).mean()) if columns else 0.0


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.event('MOUSEMOVE', 'NOTHING', middle)
    env.event('ESC', 'PRESS', middle)  # close the splash screen
    env.event('ESC', 'RELEASE', middle)
    yield 0.3

    face.clear_scene()
    objects = face.build()
    skin = objects["skin"]
    face.select(objects.values(), skin)
    settings = bpy.context.scene.magic_fit
    settings.weights_tab = 'FACE'
    settings.radius = 200  # about 1 cm in this close view
    settings.strength = 1.0
    settings.use_pressure_strength = False
    data = face.reference()
    eye = data.eyes()['l']
    with env.override():
        bpy.ops.view3d.view_axis(type='FRONT')
    env.rv3d.view_location = tuple(eye.center)
    env.rv3d.view_distance = 0.12
    workspace = bpy.context.workspace
    workspace.magic_fit_tab = 'WEIGHTS'
    env.area.spaces.active.show_region_ui = True
    yield 0.3
    sidebar = next(r for r in env.area.regions if r.type == 'UI')
    try:
        sidebar.active_panel_category = "Magic Fit"
    except AttributeError:
        pass
    env.redraw()
    yield 0.5
    check(settings.mode == 'FACE', "the Face sub-tab is the Face mode")
    with env.override(sidebar):
        check(bpy.ops.magic_fit.face_weights.poll(), "button: ready for a face without weights")
    env.screenshot("face_01_sidebar.png")

    # --- The button: detection, armature, weights, neck, all in one undo step.
    with env.override(sidebar):
        bpy.ops.ed.undo_push(message="Before face weights")
        result = bpy.ops.magic_fit.face_weights('EXEC_DEFAULT', True)
    yield 0.4
    skin = bpy.data.objects["c0201f0001_fac_a"]
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    check(result == {'FINISHED'} and len(armatures) == 1 and skin.modifiers.get("Armature") is not None,
          "button: the face got an armature and an Armature modifier")
    check(settings.face_found == "c0201 1", "button: found the game's face ({:s})".format(settings.face_found))
    env.redraw()
    yield 0.4
    env.screenshot("face_02_after_button.png")
    with env.override():
        bpy.ops.ed.undo()
    yield 0.4
    skin = bpy.data.objects.get("c0201f0001_fac_a")
    check(skin is not None and not skin.vertex_groups and not skin.modifiers and
          not [o for o in bpy.data.objects if o.type == 'ARMATURE'],
          "undo: takes the weights, the modifier and the armature back at once")
    with env.override():
        bpy.ops.ed.redo()
    yield 0.4
    # Undo replaced the scene data: look everything up again.
    skin = bpy.data.objects["c0201f0001_fac_a"]
    eyeball = bpy.data.objects["Eyeball_l"]
    settings = bpy.context.scene.magic_fit
    check(len(skin.vertex_groups) > 10, "redo: all back")

    # --- Test poses: the game's Shut Eyes expression and its blink close the eyes as displayed.
    bpy.context.view_layer.objects.active = skin
    for pose, label in (('SHUT_EYES', "Shut Eyes"), ('BLINK', "Blink")):
        with env.override(sidebar):
            result = bpy.ops.magic_fit.face_pose('EXEC_DEFAULT', True, pose=pose)
        env.redraw()
        yield 0.4
        area = open_area(skin, eyeball)
        check(result == {'FINISHED'} and area < 1.0, "pose: the eye closes in the {:s} pose ({:.2f} mm² open)".format(
            label, area))
        env.screenshot("face_03_{:s}.png".format(pose.lower()))
    with env.override(sidebar):
        bpy.ops.magic_fit.face_pose('EXEC_DEFAULT', True, pose='REST')
    env.redraw()
    yield 0.3
    check(open_area(skin, eyeball) > 20.0, "pose: Rest opens it again")

    # --- The brush paints wiped lid weights back.
    face.select([skin], skin)
    with env.override():
        bpy.ops.magic_fit.use_tool(tool='WEIGHT')
    yield 0.4
    skin = bpy.data.objects["c0201f0001_fac_a"]
    check(bpy.context.mode == 'PAINT_WEIGHT' and settings.mode == 'FACE', "brush: Weight Paint mode, Face mode kept")
    check(operators.check_ready(bpy.context, settings) is None, "brush: ready on the weighted face")
    upper = data.features().eyes['l'][0]
    near = np.flatnonzero(facing.polyline_parameter(upper, data.points)[1] < 0.003)
    before = lid_share(skin, near)
    verts = near.tolist()
    for group in skin.vertex_groups:
        group.remove(verts)
    skin.vertex_groups['j_kao'].add(verts, 1.0, 'REPLACE')
    skin.data.update()
    yield 0.3
    check(lid_share(skin, near) == 0.0, "brush: the upper lid's weights are wiped")
    # Along the lid, just above its edge: the brush would reach through the eye opening on the edge itself.
    points = [env.to_window(facing.resample(upper, 5)[i] + np.array((0.0, 0.0, 0.0015))) for i in (0, 2, 4)]
    yield from stroke(env, points)
    skin = bpy.data.objects["c0201f0001_fac_a"]
    after = lid_share(skin, near)
    check(after > 0.8 * before, "brush: painting the lid gives it its lid weights back ({:.2f} of {:.2f})".format(
        after, before))
    env.screenshot("face_04_brush.png")


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
