"""Interactive test: Body Fit on bodies and meshes whose displayed shape differs from their base shape.

Shape keys (on the body, and shown in Edit Mode with Shape Key Edit Mode) and posed armatures (shown
in Edit Mode). Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_fit_deformed_ui.py

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
from magic_fit import operators, painting  # noqa: E402

SHOTS = os.environ.get("MF_SCREENSHOTS")
FAILURES = []
# The front of both meshes bulges out by this much: a shape (key) the other mesh must follow.
BULGE = 0.05
OFFSET = 0.01


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

    def screenshot(self, name):
        if SHOTS:
            with bpy.context.temp_override(window=self.window, area=self.area):
                bpy.ops.screen.screenshot_area(filepath=os.path.join(SHOTS, name))


def bulge(co, amount=BULGE):
    """Positions pushed out along their direction, most at the front (-Y): a smooth bump."""
    unit = co / np.linalg.norm(co, axis=1, keepdims=True)
    return co + (amount * np.clip(-unit[:, 1], 0.0, None) ** 2)[:, None] * unit


def new_mesh_object(name, co_list, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(c) for c in co_list], [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def sphere_data(radius, segments, rings):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    co = np.array([v.co[:] for v in bm.verts])
    faces = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    return co, faces


def body(shape_key):
    """Unit sphere body with a bulge at the front: as a shape key (value 1), or built in."""
    co, faces = sphere_data(1.0, 48, 24)
    obj = new_mesh_object("Body", co if shape_key else bulge(co), faces)
    if shape_key:
        obj.shape_key_add(name="Basis")
        key = obj.shape_key_add(name="Bulge")
        key.data.foreach_set("co", bulge(co).astype(np.float32).ravel())
        key.value = 1.0
    return obj


def garment(name, shape_key):
    """A 3 cm thick shell whose inner layer is 3 cm inside the body, bulge included (as a shape key or
    built in). Returns the object and the vertex count of the inner layer."""
    inner, faces = sphere_data(0.97, 40, 20)
    outer, _ = sphere_data(1.00, 40, 20)
    co = np.concatenate((inner, outer))
    faces = faces + [[i + len(inner) for i in face] for face in faces]
    obj = new_mesh_object(name, co if shape_key else bulge(co), faces)
    if shape_key:
        obj.shape_key_add(name="Basis")
        key = obj.shape_key_add(name="Bulge")
        key.data.foreach_set("co", bulge(co).astype(np.float32).ravel())
        key.value = 1.0
    return obj, len(inner)


def displayed(obj):
    return painting.read_geometry(obj, bpy.context.evaluated_depsgraph_get(), source='INDEXED').co


def edit_coordinates(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return np.array([v.co[:] for v in bm.verts])


def clearance(body_obj, points):
    """Signed distance of ``points`` to the displayed body (the body is convex: along the ray)."""
    from magic_fit import fitting
    surface = fitting.BodySurface.from_object(body_obj, bpy.context.evaluated_depsgraph_get())
    return surface.nearest(points).distance


def stroke(env, center, passes=6, reach=0.4):
    """Scrub over ``center`` (world) back and forth."""
    path = []
    side = Vector((1.0, 0.0, 0.0)) if abs(center.x) < 0.9 else Vector((0.0, 1.0, 0.0))
    for i in range(passes):
        for t in (-1.0, 1.0) if i % 2 == 0 else (1.0, -1.0):
            path.append(env.to_window(center + side * reach * t))
    env.event('MOUSEMOVE', 'NOTHING', path[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', path[0])
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0.0, 1.0, 7)[1:]:
            env.event('MOUSEMOVE', 'NOTHING', (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)))
            yield 0.02
    env.event('LEFTMOUSE', 'RELEASE', path[-1])
    yield 0.3


def front_band(points, layer, direction):
    # The equator ring (the stroke went along it), within 0.3 rad of the direction.
    unit = points[:layer] / np.linalg.norm(points[:layer], axis=1, keepdims=True)
    return np.flatnonzero((np.abs(unit[:, 2]) < 0.05) & (unit @ direction > math.cos(0.3)))


def check_fit(label, body_obj, points, layer, direction):
    """The inner layer under the stroke clears the body by the offset, and the shell kept its thickness."""
    band = front_band(points, layer, direction)
    gap = clearance(body_obj, points[:layer][band])
    thickness = np.linalg.norm(points[layer:][band], axis=1) - np.linalg.norm(points[:layer][band], axis=1)
    check(len(band) >= 3 and np.abs(gap - OFFSET).max() < 0.004,
          "{}: inner layer fitted at the offset from the body as displayed ({:.4f}..{:.4f})".format(
              label, gap.min(), gap.max()))
    check(np.abs(thickness - thickness.mean()).max() < 0.004,
          "{}: thickness kept ({:.4f}..{:.4f})".format(label, thickness.min(), thickness.max()))


def setup_view(env):
    with env.override():
        bpy.ops.view3d.view_axis(type='FRONT')
    env.rv3d.view_location = (0.0, 0.0, 0.0)
    env.rv3d.view_distance = 3.0


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)


def enter_edit(env, obj, message):
    for other in bpy.context.scene.objects:
        other.select_set(other == obj)
    bpy.context.view_layer.objects.active = obj
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.wm.tool_set_by_id(name="magic_fit.body_fit_tool")
        bpy.ops.ed.undo_push(message=message)


def leave_edit(env):
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.event('MOUSEMOVE', 'NOTHING', middle)
    env.event('ESC', 'PRESS', middle)  # close the splash screen
    env.event('ESC', 'RELEASE', middle)
    yield 0.3
    scene = bpy.context.scene
    settings = scene.body_fit_brush
    settings.mode = 'PUSH'
    settings.offset = OFFSET
    settings.radius = 120
    settings.strength = 1.0
    settings.layer_radius = 0.2
    settings.use_auto_smooth = False
    front = np.array([0.0, -1.0, 0.0])

    # --- 1. The body's shape keys count: it's fitted against as displayed.
    clear_scene()
    body_obj = body(shape_key=True)
    cloth, layer = garment("Plain", shape_key=False)
    scene.magic_fit.target = body_obj
    setup_view(env)
    enter_edit(env, cloth, "Plain garment")
    yield 0.3
    yield from stroke(env, Vector((0.0, -1.05, 0.0)))
    check_fit("body shape key", body_obj, edit_coordinates(bpy.data.objects["Plain"]), layer, front)
    env.screenshot("deformed_01_body_key.png")
    leave_edit(env)

    # --- 2. The edited mesh shows its shape keys in Edit Mode (Shape Key Edit Mode, Basis active): the
    # brush fits that displayed shape, and moves the Basis (which carries the other keys along).
    clear_scene()
    body_obj = body(shape_key=True)
    cloth, layer = garment("Keyed", shape_key=True)
    cloth.active_shape_key_index = 0
    scene.magic_fit.target = body_obj
    enter_edit(env, cloth, "Keyed garment")
    yield 0.3
    notes = [lines[0] for lines, _icon in operators.fit_notes(bpy.context)]
    check(any("Bulge" in note for note in notes), "shape key not shown in Edit Mode: the panel says so ({})".format(notes))
    cloth = bpy.data.objects["Keyed"]
    cloth.use_shape_key_edit_mode = True
    cloth.data.update()
    yield 0.3
    notes = [lines[0] for lines, _icon in operators.fit_notes(bpy.context)]
    check(not notes, "shape key edit mode on: no note ({})".format(notes))
    start = edit_coordinates(bpy.data.objects["Keyed"])
    start_displayed = displayed(bpy.data.objects["Keyed"])

    # Esc: after looking at how the displayed mesh follows, everything is put back exactly.
    points = [env.to_window(Vector((x, -1.05, 0.0))) for x in (-0.2, 0.2)]
    env.event('MOUSEMOVE', 'NOTHING', points[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', points[0])
    env.event('MOUSEMOVE', 'NOTHING', points[1])
    yield 0.1
    env.event('ESC', 'PRESS', points[1])
    env.event('ESC', 'RELEASE', points[1])
    env.event('LEFTMOUSE', 'RELEASE', points[1])
    yield 0.3
    check(np.array_equal(edit_coordinates(bpy.data.objects["Keyed"]), start), "shape key edit mode: Esc left every vertex exactly as it was")

    yield from stroke(env, Vector((0.0, -1.05, 0.0)))
    cloth = bpy.data.objects["Keyed"]
    shown = displayed(cloth)
    check_fit("shape key edit mode", body_obj, shown, layer, front)
    moved = np.linalg.norm(shown - start_displayed, axis=1)
    check(moved.max() < 0.1, "shape key edit mode: nothing thrown far off ({:.3f} max move)".format(moved.max()))
    env.screenshot("deformed_02_shape_key_edit_mode.png")
    leave_edit(env)
    cloth = bpy.data.objects["Keyed"]
    check(np.abs(displayed(cloth) - shown).max() < 1e-4, "shape key edit mode: same shape after leaving Edit Mode")
    with env.override():
        bpy.ops.ed.undo()
        bpy.ops.ed.undo()
    yield 0.3
    scene = bpy.context.scene  # undo reloaded it

    # --- 3. A posed armature shown in Edit Mode (on cage): the brush fits the posed shape.
    clear_scene()
    body_obj = body(shape_key=False)
    cloth, layer = garment("Posed", shape_key=False)
    rig = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
    bone = rig.data.edit_bones.new("Bone")
    bone.head, bone.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 0.5)
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    turn = math.radians(35.0)
    rig.pose.bones["Bone"].rotation_mode = 'XYZ'
    rig.pose.bones["Bone"].rotation_euler = (0.0, turn, 0.0)  # about the bone's Y axis = world Z
    for obj in (body_obj, cloth):
        obj.vertex_groups.new(name="Bone").add(list(range(len(obj.data.vertices))), 1.0, 'REPLACE')
        modifier = obj.modifiers.new("Armature", 'ARMATURE')
        modifier.object = rig
    cloth.modifiers["Armature"].show_in_editmode = True
    cloth.modifiers["Armature"].show_on_cage = True
    scene.magic_fit.target = body_obj
    enter_edit(env, cloth, "Posed garment")
    yield 0.3
    bpy.context.view_layer.update()
    check(not operators.fit_notes(bpy.context), "posed and shown posed in Edit Mode: no note")
    bpy.data.objects["Posed"].modifiers["Armature"].show_in_editmode = False
    notes = [lines[0] for lines, _icon in operators.fit_notes(bpy.context)]
    check(any("posed" in note for note in notes), "posed but shown unposed: the panel says so ({})".format(notes))
    bpy.data.objects["Posed"].modifiers["Armature"].show_in_editmode = True
    bpy.context.view_layer.update()
    yield 0.2
    # Where the bulge ended up: turned with the bone.
    rest = bpy.data.objects["Posed"].data
    shown = displayed(bpy.data.objects["Posed"])
    tip = np.argmin(np.array([v.co.y for v in rest.vertices[:layer]]))
    direction = shown[tip] / np.linalg.norm(shown[tip])
    check(abs(direction[1] + 1.0) > 0.1, "posed: the bulge turned with the bone ({})".format(direction.round(2)))
    yield from stroke(env, Vector(direction * 1.05))
    check_fit("posed armature", body_obj, displayed(bpy.data.objects["Posed"]), layer, direction)
    env.screenshot("deformed_03_posed.png")
    leave_edit(env)


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
