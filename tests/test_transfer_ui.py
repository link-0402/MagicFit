"""Interactive test for the Weight Transfer panel of the "Magic Fit" sidebar tab.

Opens a Blender window for a few seconds. Needs SciPy and robust-laplacian in deps/ (see
tools/vendor_wheels.py). Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_transfer_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set).
"""

import os
import sys
import traceback

import bmesh
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(ROOT, "deps")):
    sys.path.insert(0, os.path.join(ROOT, "deps"))
sys.path.insert(0, ROOT)

import magic_fit  # noqa: E402
from magic_fit.robust_transfer import dependencies, ui  # noqa: E402

SHOTS = os.environ.get("MF_SCREENSHOTS")
FAILURES = []
DRAW_ERRORS = []
DRAWS = [0]


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def _checked(draw):
    """Record exceptions of the panel's draw, which Blender would only print."""
    def wrapper(self, context):
        try:
            draw(self, context)
            DRAWS[0] += 1
        except Exception:
            DRAW_ERRORS.append(traceback.format_exc())
            raise
    return wrapper


def sphere(name, radius, rig):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=radius)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("Armature", 'ARMATURE').object = rig
    return obj


def make_rig():
    data = bpy.data.armatures.new("Rig")
    rig = bpy.data.objects.new("Rig", data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for name, head, tail in (("Upper", (0, 0, 0), (0, 0, 1)), ("Lower", (0, 0, 0), (0, 0, -1))):
        bone = data.edit_bones.new(name)
        bone.head, bone.tail = head, tail
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig


def upper_weight(z):
    return np.clip(0.5 + z, 0.0, 1.0)


def weights(obj, name):
    group = obj.vertex_groups.get(name)
    result = np.zeros(len(obj.data.vertices))
    if group is not None:
        for vert in obj.data.vertices:
            for elem in vert.groups:
                if elem.group == group.index:
                    result[vert.index] = elem.weight
    return result


def steps():
    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
    sidebar = next(r for r in area.regions if r.type == 'UI')
    middle = (area.x + area.width // 2, area.y + area.height // 2)
    window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=middle[0], y=middle[1])
    window.event_simulate(type='ESC', value='PRESS', x=middle[0], y=middle[1])  # close the splash screen
    window.event_simulate(type='ESC', value='RELEASE', x=middle[0], y=middle[1])
    yield 0.3

    def screenshot(name):
        if SHOTS:
            with bpy.context.temp_override(window=window, area=area):
                bpy.ops.screen.screenshot_area(filepath=os.path.join(SHOTS, name))

    def in_sidebar():
        return bpy.context.temp_override(window=window, area=area, region=sidebar)

    def redraw():
        area.tag_redraw()
        return 0.3

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    rig = make_rig()
    body = sphere("Body", 1.0, rig)
    for name, values in (("Upper", upper_weight), ("Lower", lambda z: 1.0 - upper_weight(z))):
        group = body.vertex_groups.new(name=name)
        for vert in body.data.vertices:
            group.add([vert.index], float(values(vert.co.z)), 'REPLACE')
    garment = sphere("Garment", 1.02, rig)
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)
    scene = bpy.context.scene
    scene.magic_fit.target = body

    area.spaces.active.show_region_ui = True
    yield redraw()
    sidebar.active_panel_category = "Magic Fit"
    yield redraw()
    check(DRAWS[0] > 0, "the Weight Transfer panel is drawn on the Weights tab")
    screenshot("transfer_1_panel.png")

    with in_sidebar():
        check(bpy.ops.magic_fit.transfer_weights.poll(), "Transfer Weights is available")
        result = bpy.ops.magic_fit.transfer_weights('EXEC_DEFAULT')
    check(result == {'FINISHED'}, "Transfer Weights finished")
    z = np.array([vert.co.z for vert in garment.data.vertices]) * 1.0 / 1.02
    error = np.abs(weights(garment, "Upper") - upper_weight(z)).max()
    check(error < 0.05, "the garment got the body's weights (max error {:.4f})".format(error))
    total = weights(garment, "Upper") + weights(garment, "Lower")
    check(np.abs(total - 1.0).max() < 1e-3, "the transferred weights add up to 1")
    yield redraw()
    screenshot("transfer_2_after.png")

    # Scroll down to the settings sections.
    x, y = sidebar.x + sidebar.width // 2, sidebar.y + sidebar.height // 2
    window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x, y=y)
    for _ in range(12):
        window.event_simulate(type='WHEELDOWNMOUSE', value='PRESS', x=x, y=y)
    yield redraw()
    screenshot("transfer_3_settings.png")
    for _ in range(12):
        window.event_simulate(type='WHEELDOWNMOUSE', value='PRESS', x=x, y=y)
    yield redraw()
    screenshot("transfer_4_utilities.png")
    for _ in range(30):
        window.event_simulate(type='WHEELUPMOUSE', value='PRESS', x=x, y=y)

    with bpy.context.temp_override(window=window, area=area):
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    yield redraw()
    with in_sidebar():
        check(bpy.ops.magic_fit.transfer_weights.poll(), "Transfer Weights works in Weight Paint mode")
        garment.vertex_groups["Upper"].add([0], 0.0, 'REPLACE')
        check(bpy.ops.magic_fit.transfer_weights('EXEC_DEFAULT') == {'FINISHED'},
              "Transfer Weights finished in Weight Paint mode")
    check(abs(weights(garment, "Upper")[0] - upper_weight(garment.data.vertices[0].co.z / 1.02)) < 0.05,
          "the transfer in Weight Paint mode rewrote the weights")
    screenshot("transfer_5_weight_paint.png")

    with bpy.context.temp_override(window=window, area=area):
        bpy.ops.object.mode_set(mode='EDIT')
    yield redraw()
    with in_sidebar():
        check(bpy.ops.magic_fit.select_rejected.poll(), "Select Rejected Loose Parts works in Edit Mode")
        check(bpy.ops.magic_fit.select_rejected('EXEC_DEFAULT') == {'FINISHED'},
              "Select Rejected Loose Parts finished")
        check(not bpy.ops.magic_fit.transfer_weights.poll(), "Transfer Weights is off in Edit Mode")
    check(bpy.context.mode == 'EDIT_MESH', "Select Rejected Loose Parts stays in Edit Mode")
    bm = bmesh.from_edit_mesh(garment.data)
    check(not any(vert.select for vert in bm.verts), "nothing is rejected on a garment around the body")
    with bpy.context.temp_override(window=window, area=area):
        bpy.ops.object.mode_set(mode='OBJECT')

    saved = dependencies._missing
    dependencies._missing = ["SciPy", "robust-laplacian"]
    try:
        yield redraw()
        with in_sidebar():
            check(not bpy.ops.magic_fit.transfer_weights.poll(),
                  "Transfer Weights is off without its dependencies")
        screenshot("transfer_6_missing_dependencies.png")
    finally:
        dependencies._missing = saved
    yield redraw()
    check(not DRAW_ERRORS, "the panel drew without errors")


def run():
    ui.VIEW3D_PT_weight_transfer_sidebar.draw = _checked(ui.VIEW3D_PT_weight_transfer_sidebar.draw)
    ui.SECTIONS_CLOSED = False  # draw every section
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
        for error in DRAW_ERRORS[:3]:
            print(error)
        print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
        sys.stdout.flush()
        os._exit(1 if FAILURES else 0)

    bpy.app.timers.register(tick, first_interval=1.5, persistent=True)


run()
