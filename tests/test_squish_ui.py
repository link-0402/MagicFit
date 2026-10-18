"""Interactive test for the Squish brush (Edit Mode): its Set Up, Apply and Remove buttons, and strokes driven by
simulated mouse events.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_squish_ui.py

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

import magic_fit  # noqa: E402
from magic_fit import operators  # noqa: E402

SHOTS = os.environ.get("MF_SCREENSHOTS")
FAILURES = []
HEM = 0.0025


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


def rings(radii, zs, step=0.005, even=False):
    """Positions and triangles of rings around Z at ``radii`` and heights ``zs``, joined row to row, facing out.
    With ``even``, each ring has an even number of vertices, so it's the same mirrored in X."""
    count = max(3, round(2.0 * np.pi * max(radii) / step))
    if even:
        count += count % 2
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    co = np.array([(r * np.cos(a), r * np.sin(a), z) for a in angles for r, z in zip(radii, zs)])
    index = np.arange(len(co)).reshape(len(angles), len(zs))
    index = np.concatenate((index, index[:1]))
    a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    return co, np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))


def make_object(name, co, tris):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(co.tolist(), [], tris.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def torso_and_top(even=False):
    zs = np.linspace(-0.2, 0.2, 81)
    body = make_object("Body", *rings(np.full(len(zs), 0.12), zs, even=even))
    zs = np.linspace(-0.21, HEM, 43)
    top = make_object("Top", *rings(np.full(len(zs), 0.123), zs, even=even))
    return body, top


def clear_scene():
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)


def select_only(active, *others):
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    for obj in (active, *others):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active


def local_co(obj, key=None):
    obj.update_from_editmode()
    data = obj.data.vertices if key is None else obj.data.shape_keys.key_blocks[key].data
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    data.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def key_names(obj):
    keys = obj.data.shape_keys
    return [] if keys is None else [key.name for key in keys.key_blocks]


def test_setup(env):
    clear_scene()
    body, top = torso_and_top()
    select_only(top)
    bpy.context.scene.squish_brush.body = body
    with env.override():
        result = bpy.ops.magic_fit.squish_setup()
    check(result == {'FINISHED'}, "Set Up Squish runs")
    for obj in (body, top):
        check(key_names(obj) == ["Basis", "Squish"], obj.name + ": Basis and Squish keys")
        check(obj.active_shape_key is not None and obj.active_shape_key.name == "Squish", obj.name + ": Squish active")
        check(obj.data.shape_keys.key_blocks["Squish"].value == 1.0, obj.name + ": Squish at 1")
        check(obj.mode == 'EDIT', obj.name + ": in Edit Mode")
    with env.override():
        check(operators.check_squish_ready(bpy.context) is None, "ready to squish after Set Up")


def test_setup_shows_key(env):
    # A Squish key left at 0 or muted would show the strokes in Edit Mode, but not in Object Mode nor once applied.
    clear_scene()
    body, top = torso_and_top()
    for obj in (body, top):
        obj.shape_key_add(name="Basis", from_mix=False)
        obj.shape_key_add(name="Squish", from_mix=False)
    body.data.shape_keys.key_blocks["Squish"].value = 0.0
    top.data.shape_keys.key_blocks["Squish"].mute = True
    select_only(top)
    bpy.context.scene.squish_brush.body = body
    with env.override():
        result = bpy.ops.magic_fit.squish_setup()
    check(result == {'FINISHED'}, "Set Up Squish runs over existing keys")
    for obj in (body, top):
        key = obj.data.shape_keys.key_blocks["Squish"]
        check(key.value == 1.0 and not key.mute, obj.name + ": Squish shown at 1 ({}, muted {})".format(
            key.value, key.mute))


def test_no_faces(env):
    clear_scene()
    _body, top = torso_and_top()
    mesh = bpy.data.meshes.new("Edges")
    mesh.from_pydata([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)], [(0, 1), (1, 2)], [])
    edges = bpy.data.objects.new("Edges", mesh)
    bpy.context.scene.collection.objects.link(edges)
    select_only(top)
    bpy.context.scene.squish_brush.body = edges
    with env.override():
        bpy.ops.magic_fit.squish_setup()
        problem = operators.check_squish_ready(bpy.context)
    check(problem is not None and "faces" in problem, "a body with no faces is refused: {!r}".format(problem))


def test_apply_hidden(env):
    # Hidden in the viewport while edited, the clothing can't go back into Edit Mode after Apply.
    clear_scene()
    body, top = torso_and_top()
    select_only(top)
    bpy.context.scene.squish_brush.body = body
    with env.override():
        bpy.ops.magic_fit.squish_setup()
    top.hide_viewport = True
    try:
        with env.override():
            result = bpy.ops.magic_fit.squish_apply()
    except RuntimeError as error:
        check(False, "Apply Squish with the clothing hidden: {}".format(error))
        return
    finally:
        top.hide_viewport = False
    check(result == {'FINISHED'}, "Apply Squish still applies")
    # Hidden, the clothing isn't in Edit Mode any more, so only the body is applied.
    check(key_names(body) == [], "the body's keys are gone ({})".format(key_names(body)))


def test_restrictions(env):
    clear_scene()
    body, top = torso_and_top()
    body.shape_key_add(name="Basis", from_mix=False)
    body.shape_key_add(name="Big Belly", from_mix=False)
    problem = operators.squish_problem(body)
    check(problem is not None and "Big Belly" in problem, "an extra shape key is named: {!r}".format(problem))
    select_only(top)
    bpy.context.scene.squish_brush.body = body
    with env.override():
        result = bpy.ops.magic_fit.squish_setup()
    check(result == {'CANCELLED'}, "Set Up refuses it")
    check(key_names(top) == [] and key_names(body) == ["Basis", "Big Belly"], "and changes nothing")

    modifier = top.modifiers.new("Rue Weights", 'DATA_TRANSFER')
    modifier.show_in_editmode = True
    problem = operators.squish_problem(top)
    check(problem is not None and "Rue Weights" in problem, "a modifier shown in Edit Mode is named: {!r}".format(problem))
    modifier.show_in_editmode = False
    check(operators.squish_problem(top) is None, "not when it isn't shown in Edit Mode")
    top.modifiers.remove(modifier)
    armature = top.modifiers.new("Armature", 'ARMATURE')
    armature.show_in_editmode = True
    check(operators.squish_problem(top) is None, "an Armature is fine")


def test_presets(env):
    settings = bpy.context.scene.squish_brush
    for preset, values in (('THIGHS', (0.0015, 0.003, 0.05)), ('CHEST', (0.003, 0.009, 0.07))):
        with env.override():
            result = bpy.ops.magic_fit.squish_preset(preset=preset)
        got = (settings.press, settings.bulge, settings.bulge_width)
        check(result == {'FINISHED'} and np.allclose(got, values), "{} preset: {}".format(preset, got))


def test_hidden_body(env):
    clear_scene()
    body, top = torso_and_top()
    body.hide_set(True)
    select_only(top)
    bpy.context.scene.squish_brush.body = body
    with env.override():
        result = bpy.ops.magic_fit.squish_setup()
    check(result == {'CANCELLED'}, "Set Up refuses a hidden body")
    check(key_names(top) == [] and key_names(body) == [], "and changes nothing")
    check(top.mode == 'OBJECT', "nothing is left in Edit Mode without it")


def test_frozen_basis(env):
    # A posed Armature can leave vertices frozen (a bone scaled to nothing): their follow matrices are all zero.
    clear_scene()
    body, top = torso_and_top()
    select_only(top)
    bpy.context.scene.squish_brush.body = body
    with env.override():
        bpy.ops.magic_fit.squish_setup()
        edited = operators.read_edited_meshes(bpy.context, body, include_body=True)
    part = edited.parts[1]
    inverse = np.repeat(part.inverse[None], part.count, axis=0)
    inverse[:10] = 0.0
    part.inverse = inverse
    try:
        basis = part.basis_displayed()
    except np.linalg.LinAlgError:
        check(False, "Basis of a part with frozen vertices: singular matrix")
        return
    check(np.allclose(basis[:10], part.origin[:10]), "frozen vertices stay where they are shown")
    check(np.allclose(basis[10:], part.origin[10:]), "the others are at Basis (the Squish key is still Basis)")


def test_apply_value(env):
    clear_scene()
    body, top = torso_and_top()
    select_only(top)
    bpy.context.scene.squish_brush.body = body
    with env.override():
        bpy.ops.magic_fit.squish_setup()
        bpy.ops.object.mode_set(mode='OBJECT')
    basis = local_co(top, "Basis")
    out = basis.copy()
    out[:, :2] *= (0.123 + 0.02) / 0.123
    key = top.data.shape_keys.key_blocks["Squish"]
    key.data.foreach_set("co", out.astype(np.float32).ravel())
    key.value = 0.5
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        result = bpy.ops.magic_fit.squish_apply()
    check(result == {'FINISHED'}, "Apply Squish runs")
    # They had no shape keys before Set Up: none are left.
    check(key_names(top) == [] and key_names(body) == [], "no shape keys left ({}, {})".format(
        key_names(top), key_names(body)))
    expected = (basis + out) / 2.0
    check(np.abs(local_co(top) - expected).max() < 1e-6, "the mesh is what was shown, half way out")
    check(top.mode == 'EDIT', "back in Edit Mode")


def test_remove(env):
    clear_scene()
    body, top = torso_and_top()
    select_only(top)
    bpy.context.scene.squish_brush.body = body
    with env.override():
        bpy.ops.magic_fit.squish_setup()
    basis = local_co(top, "Basis")
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    squished = basis.copy()
    squished[:, :2] *= 1.1
    top.data.shape_keys.key_blocks["Squish"].data.foreach_set("co", squished.astype(np.float32).ravel())
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        result = bpy.ops.magic_fit.squish_remove()
    check(result == {'FINISHED'}, "Remove Squish runs")
    check(key_names(top) == [] and key_names(body) == [], "no shape keys left ({}, {})".format(
        key_names(top), key_names(body)))
    check(np.array_equal(local_co(top), basis), "the mesh is back at Basis")


def test_not_ready(env):
    clear_scene()
    body, top = torso_and_top()
    scene = bpy.context.scene
    scene.squish_brush.body = None
    select_only(top)
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
        problem = operators.check_squish_ready(bpy.context)
    check(problem is not None and "body" in problem.lower(), "no body: {!r}".format(problem))
    scene.squish_brush.body = body
    with env.override():
        problem = operators.check_squish_ready(bpy.context)
    check(problem is not None and "Set Up Squish" in problem, "body not in Edit Mode: {!r}".format(problem))
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
        select_only(top, body)
        bpy.ops.magic_fit.squish_setup()
    top.active_shape_key_index = 0
    with env.override():
        problem = operators.check_squish_ready(bpy.context)
    check(problem is not None and "Set Up Squish" in problem, "Squish not active: {!r}".format(problem))


def world_co(obj, key):
    co = local_co(obj, key)
    matrix = np.array(obj.matrix_world)
    return co @ matrix[:3, :3].T + matrix[:3, 3]


def clothing_inside(body, top):
    """How deep the top is inside the body now (world space), at most (0 when it isn't)."""
    from mathutils.bvhtree import BVHTree
    from magic_fit import fitting
    co = world_co(body, "Squish")
    tris = np.array([tuple(p.vertices) for p in body.data.polygons])
    surface = fitting.BodySurface(co, tris, BVHTree.FromPolygons(co.tolist(), tris.tolist(), all_triangles=True))
    return max(0.0, -float(surface.nearest(world_co(top, "Squish")).distance.min()))


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
    return [(round(a[0] + (b[0] - a[0]) * i / steps), round(a[1] + (b[1] - a[1]) * i / steps))
            for i in range(steps + 1)]


def test_stroke(env):
    from mathutils import Vector
    clear_scene()
    body, top = torso_and_top()
    # Imported game models are often turned and scaled.
    for obj in (body, top):
        obj.rotation_euler = (np.pi / 2.0, 0.0, 0.0)
        obj.scale = (1.5, 1.5, 1.5)
    bpy.context.view_layer.update()
    select_only(top)
    settings = bpy.context.scene.squish_brush
    settings.body = body
    settings.radius = 80
    settings.strength = 1.0
    settings.falloff = 'CONSTANT'
    with env.override():
        bpy.ops.magic_fit.squish_setup()
        bpy.ops.view3d.view_axis(type='TOP')
        bpy.ops.wm.tool_set_by_id(name="magic_fit.squish_tool")
        bpy.ops.ed.undo_push(message="Test setup")
    # The hem lies across the torso's top side, in world space at y = -1.5 * HEM.
    hem_y = -1.5 * HEM
    env.rv3d.view_location = (0.0, hem_y, 0.0)
    env.rv3d.view_distance = 0.8
    yield 0.5
    tool = bpy.context.workspace.tools.from_space_view3d_mode('EDIT_MESH', create=False)
    check(tool is not None and tool.idname == "magic_fit.squish_tool", "Squish tool active in Edit Mode")
    env.screenshot("squish_01_before.png")
    starts = {obj.name: world_co(obj, "Basis") for obj in (body, top)}
    top_z = 1.5 * 0.123

    across = line(env, Vector((-0.08, hem_y, top_z)), Vector((0.08, hem_y, top_z)))
    yield from stroke(env, across)
    env.screenshot("squish_02_after.png")
    for obj in (body, top):
        moved = np.abs(world_co(obj, "Squish") - starts[obj.name]).max()
        # Depth is a world space length (4 mm), whatever the objects' scale.
        check(moved > 0.002, "{}: squished ({:.4f} m)".format(obj.name, moved))
        check(np.array_equal(world_co(obj, "Basis"), starts[obj.name]), obj.name + ": Basis untouched")
    inside = clothing_inside(body, top)
    check(inside < 1e-5, "no clothing inside the skin ({:.6f} m)".format(inside))

    # One undo puts both back.
    with env.override():
        bpy.ops.ed.undo()
    yield 0.3
    body, top = bpy.data.objects["Body"], bpy.data.objects["Top"]
    for obj in (body, top):
        check(np.abs(world_co(obj, "Squish") - starts[obj.name]).max() < 1e-6, obj.name + ": undo put it back")

    # Esc puts both back exactly.
    env.event('MOUSEMOVE', 'NOTHING', across[0])
    yield 0.05
    env.event('LEFTMOUSE', 'PRESS', across[0])
    for xy in across[1:]:
        env.event('MOUSEMOVE', 'NOTHING', xy)
        yield 0.03
    check(np.abs(world_co(top, "Squish") - starts["Top"]).max() > 1e-4, "a stroke in progress moves the top")
    env.event('ESC', 'PRESS', across[-1])
    env.event('ESC', 'RELEASE', across[-1])
    yield 0.2
    env.event('LEFTMOUSE', 'RELEASE', across[-1])
    yield 0.3
    for obj in (body, top):
        check(np.abs(world_co(obj, "Squish") - starts[obj.name]).max() < 1e-6, obj.name + ": Esc put it back")

    # Ctrl erases back to Basis.
    yield from stroke(env, across)
    settings.radius = 160
    yield from stroke(env, across, ctrl=True)
    env.screenshot("squish_03_erased.png")
    for obj in (body, top):
        left = np.abs(world_co(obj, "Squish") - starts[obj.name]).max()
        check(left < 1e-5, "{}: Ctrl erased the squish ({:.6f} m left)".format(obj.name, left))


def front_setup(env, body, top, center):
    """Set up the Squish brush over ``body`` and ``top``, seen from the front around ``center`` (world space)."""
    select_only(top)
    settings = bpy.context.scene.squish_brush
    settings.body = body
    settings.radius = 80
    settings.strength = 1.0
    settings.falloff = 'CONSTANT'
    settings.use_selected_only = False
    with env.override():
        bpy.ops.magic_fit.squish_setup()
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.wm.tool_set_by_id(name="magic_fit.squish_tool")
        bpy.ops.ed.undo_push(message="Test setup")
    env.rv3d.view_location = center
    env.rv3d.view_distance = 0.8


def front_hem(x0, x1, matrix=None):
    """Points along the top's hem on the side facing the front view, from x = ``x0`` to ``x1`` (before ``matrix``)."""
    from mathutils import Matrix, Vector
    matrix = Matrix() if matrix is None else matrix
    return matrix @ Vector((x0, -0.123, HEM)), matrix @ Vector((x1, -0.123, HEM))


def moves(obj):
    """How far each vertex of ``obj`` is squished, in its own space (Squish key against Basis)."""
    return local_co(obj, "Squish") - local_co(obj, "Basis")


def test_posed_stroke(env):
    # Both posed by an Armature, shown posed in Edit Mode: the brush squishes them as shown, through the matrices
    # that tell how each vertex follows its edits.
    import math
    clear_scene()
    body, top = torso_and_top()
    scene = bpy.context.scene
    rig = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
    bone = rig.data.edit_bones.new("Bone")
    bone.head, bone.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 0.5)
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    pose_bone = rig.pose.bones["Bone"]
    pose_bone.rotation_mode = 'XYZ'
    pose_bone.rotation_euler = (math.radians(20.0), 0.0, 0.0)
    for obj in (body, top):
        obj.vertex_groups.new(name="Bone").add(list(range(len(obj.data.vertices))), 1.0, 'REPLACE')
        modifier = obj.modifiers.new("Armature", 'ARMATURE')
        modifier.object = rig
        modifier.show_in_editmode = True
        modifier.show_on_cage = True
    bpy.context.view_layer.update()
    pose = rig.matrix_world @ pose_bone.matrix @ pose_bone.bone.matrix_local.inverted()
    check(abs(pose.to_euler().x) > 0.3, "the bone tilts the torso and the top")
    a, b = front_hem(-0.08, 0.08, pose)
    front_setup(env, body, top, (a + b) / 2.0)
    yield 0.5
    operators.clear_follow_cache()
    yield from stroke(env, line(env, a, b))
    check(body.as_pointer() in operators._follow_cache, "the body was followed vertex by vertex")
    for obj in (body, top):
        moved = np.linalg.norm(moves(obj), axis=1).max()
        check(moved > 0.002, "{}: squished ({:.4f} m)".format(obj.name, moved))
    move = moves(body)
    size = np.linalg.norm(move, axis=1)
    lean = np.abs(move[size > 1e-4, 2]) / size[size > 1e-4]
    check(len(lean) and lean.max() < 0.05, "the skin moved straight out of the torso, as shown ({:.3f} lean)".format(
        lean.max() if len(lean) else -1.0))
    inside = clothing_inside(body, top)
    check(inside < 1e-5, "no clothing inside the skin ({:.6f} m)".format(inside))


def test_mirror_stroke(env):
    from mathutils import kdtree
    clear_scene()
    body, top = torso_and_top(even=True)
    top.data.use_mirror_x = True
    a, b = front_hem(0.03, 0.09)
    front_setup(env, body, top, front_hem(0.0, 0.0)[0])
    yield 0.5
    yield from stroke(env, line(env, a, b))
    basis = local_co(body, "Basis")
    size = np.linalg.norm(moves(body), axis=1)
    for label, side in (("stroked", basis[:, 0] > 0.02), ("mirrored", basis[:, 0] < -0.02)):
        check(size[side].max() > 0.002, "{} side squished ({:.4f} m)".format(label, size[side].max()))
    tree = kdtree.KDTree(len(basis))
    for k, co in enumerate(basis.tolist()):
        tree.insert(co, k)
    tree.balance()
    mirror = np.array([tree.find((-x, y, z))[1] for x, y, z in basis.tolist()])
    error = np.abs(size - size[mirror]).max()
    check(error < 1e-4, "both sides alike ({:.6f} m apart at most)".format(error))
    for obj in (body, top):
        moved = np.linalg.norm(moves(obj), axis=1)
        check(moved[local_co(obj, "Basis")[:, 0] < -0.02].max() > 0.002, obj.name + ": mirrored too")


def test_selected_only_stroke(env):
    import bmesh
    clear_scene()
    body, top = torso_and_top()
    a, b = front_hem(-0.08, 0.08)
    front_setup(env, body, top, front_hem(0.0, 0.0)[0])
    # Only the vertices on the +X side are selected.
    selected = {}
    for obj in (body, top):
        bm = bmesh.from_edit_mesh(obj.data)
        for item in (*bm.faces, *bm.edges, *bm.verts):
            item.select = False
        for vert in bm.verts:
            vert.select = vert.co.x > 1e-6
        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)
        selected[obj.name] = np.array([vert.select for vert in bm.verts])
    bpy.context.scene.squish_brush.use_selected_only = True
    yield 0.5
    yield from stroke(env, line(env, a, b))
    bpy.context.scene.squish_brush.use_selected_only = False
    for obj in (body, top):
        moved = np.linalg.norm(moves(obj), axis=1)
        chosen = selected[obj.name]
        check(moved[~chosen].max() == 0.0, obj.name + ": unselected vertices stay put")
        check(moved[chosen].max() > 0.002, "{}: selected ones squished ({:.4f} m)".format(obj.name, moved[chosen].max()))
    inside = clothing_inside(body, top)
    check(inside < 1e-5, "no clothing inside the skin ({:.6f} m)".format(inside))


def test_transforms_stroke(env):
    # The top placed, turned and scaled differently from the body, though where it is shown is the same.
    from mathutils import Matrix
    clear_scene()
    body, top = torso_and_top()
    matrix = Matrix.Translation((0.05, 0.02, -0.03)) @ Matrix.Rotation(0.5, 4, 'Z') @ Matrix.Scale(1.3, 4)
    top.matrix_world = matrix
    world = local_co(top)
    local = world @ np.array(matrix.inverted())[:3, :3].T + np.array(matrix.inverted())[:3, 3]
    top.data.vertices.foreach_set("co", local.astype(np.float32).ravel())
    top.data.update()
    bpy.context.view_layer.update()
    from mathutils.bvhtree import BVHTree
    from magic_fit import fitting
    tris = np.array([tuple(p.vertices) for p in body.data.polygons])

    def gaps(body_co, top_co):
        surface = fitting.BodySurface(body_co, tris, BVHTree.FromPolygons(body_co.tolist(), tris.tolist(),
                                                                            all_triangles=True))
        return surface.nearest(top_co).distance

    a, b = front_hem(-0.08, 0.08)
    front_setup(env, body, top, front_hem(0.0, 0.0)[0])
    yield 0.5
    before = gaps(world_co(body, "Basis"), world_co(top, "Basis"))
    yield from stroke(env, line(env, a, b))
    for obj in (body, top):
        moved = np.linalg.norm(world_co(obj, "Squish") - world_co(obj, "Basis"), axis=1)
        check(moved.max() > 0.002, "{}: squished ({:.4f} m)".format(obj.name, moved.max()))
    after = gaps(world_co(body, "Squish"), world_co(top, "Squish"))
    # In the middle of the stroke, where the skin all around is squished in full (at the brush's rim, 3.5 cm
    # around the stroke, the Constant falloff leaves a step in the skin).
    shown = world_co(top, "Basis")
    middle = (np.abs(shown[:, 0]) < 0.04) & (np.abs(shown[:, 2] - HEM) < 0.02)
    error = np.abs(after - before)[middle].max()
    check(error < 0.0003, "the gap between the skin and the top is kept ({:.5f} m)".format(error))
    inside = clothing_inside(body, top)
    check(inside < 1e-5, "no clothing inside the skin ({:.6f} m)".format(inside))


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.event('MOUSEMOVE', 'NOTHING', middle)
    env.event('ESC', 'PRESS', middle)  # close the splash screen
    env.event('ESC', 'RELEASE', middle)
    yield 0.3
    for test in (test_setup, test_setup_shows_key, test_no_faces, test_restrictions, test_presets, test_hidden_body,
                 test_frozen_basis, test_apply_value, test_apply_hidden, test_remove, test_not_ready, test_stroke,
                 test_posed_stroke, test_mirror_stroke, test_selected_only_stroke, test_transforms_stroke):
        print("--", test.__name__)
        try:
            result = test(env)
            if result is not None:
                yield from result
        except Exception:
            traceback.print_exc()
            FAILURES.append(test.__name__ + " raised")
        yield 0.1


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
