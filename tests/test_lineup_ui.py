"""Interactive test for Line Up (the Line Up tab of the sidebar), driven from the sidebar.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_lineup_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set).
"""

import os
import sys
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import lineup as fixture  # noqa: E402
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

    def override(self, region=None):
        return bpy.context.temp_override(window=self.window, area=self.area, region=region or self.region)

    def screenshot(self, name):
        if not SHOTS:
            return
        with bpy.context.temp_override(window=self.window, area=self.area):
            bpy.ops.screen.screenshot(filepath=os.path.join(SHOTS, name))


def marker_error(scene_markers):
    model = bpy.data.objects["VRC Suit"]
    errors = fixture.marker_errors(model, scene_markers, BODY_JOINTS)
    return max(errors.values())


BODY_JOINTS = {name: np.array(point) for name, point in fixture.mirror(fixture.BODY_JOINTS).items()}


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=middle[0], y=middle[1])
    env.window.event_simulate(type='ESC', value='PRESS', x=middle[0], y=middle[1])  # close the splash screen
    env.window.event_simulate(type='ESC', value='RELEASE', x=middle[0], y=middle[1])
    yield 0.3

    scene = fixture.build()
    bpy.context.scene.body_fit_brush.lineup_stretch = True  # off by default; the joints only land exactly with it
    bpy.context.scene.body_fit_brush.lineup_fit_shape = False  # and without reshaping after
    markers = scene["vrc_markers"]
    for obj in list(scene["bipeds"]) + [scene["anon"], scene["anon_armature"]]:
        obj.hide_set(True)
    fixture.select([scene["vrc"]])
    with env.override():
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.view3d.view_all()
        bpy.ops.ed.undo_push(message="Start")
    space = env.area.spaces.active
    space.show_region_ui = True
    yield 0.3
    sidebar = next(r for r in env.area.regions if r.type == 'UI')
    try:
        sidebar.active_panel_category = "Magic Fit"
    except AttributeError:
        pass
    bpy.context.workspace.magic_fit_tab = 'LINE_UP'
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.5
    panel = getattr(bpy.types, "VIEW3D_PT_line_up_sidebar", None)
    with bpy.context.temp_override(window=env.window, area=env.area, region=sidebar):
        shown = panel is not None and panel.poll(bpy.context)
    check(shown, "panel: Line Up is on the Line Up tab")
    check(operators.check_lineup_ready(bpy.context) is None, "panel: ready with the body picked and a model selected")
    check(operators.lineup_summary(bpy.context) == "'VRC Suit' of 'VRC Armature'", "panel: names the model")
    before = marker_error(markers)
    env.screenshot("lineup_01_panel.png")

    # --- The button in the sidebar.
    with env.override(sidebar):
        check(bpy.ops.magic_fit.line_up.poll(), "button: ready in Object Mode")
        result = bpy.ops.magic_fit.line_up('EXEC_DEFAULT', True)
    yield 0.5
    after = marker_error(markers)
    check(result == {'FINISHED'} and before > 0.05 and after < 1e-4,
          "button: the joints moved onto the body's ({:.0f} mm -> {:.2f} mm)".format(1000 * before, 1000 * after))
    armature = bpy.data.objects["VRC Armature"]
    check(armature.mode == 'OBJECT' and bpy.context.object.name == "VRC Suit" and bpy.data.objects["VRC Suit"].select_get()
          and not armature.select_get(), "button: back in Object Mode with the selection as it was")
    shown = fixture.displayed_positions(bpy.data.objects["VRC Suit"])
    check(np.abs(shown - fixture.world_positions(bpy.data.objects["VRC Suit"])).max() < 1e-5,
          "button: the armature's rest pose moved along")
    env.screenshot("lineup_02_lined_up.png")

    # --- One undo step for meshes and armature.
    with env.override():
        bpy.ops.ed.undo()
    yield 0.4
    # Undo reloads the scene: look everything up again.
    check(abs(marker_error(markers) - before) < 1e-6, "undo: the model is back where it was")
    arm_bone = bpy.data.objects["VRC Armature"].data.bones["Arm_L"]
    check(abs(arm_bone.head_local[2] - scene["vrc_joints"]["shoulder_l"][2]) < 1e-5, "undo: so is its rest pose")
    with env.override():
        bpy.ops.ed.redo()
    yield 0.4
    check(marker_error(markers) < 1e-4, "redo: lined up again")

    # --- Adjust Last Operation: the operator's own Stretch Limbs, written back to the panel's setting.
    with env.override():
        bpy.ops.ed.undo()
    yield 0.4
    fixture.select([bpy.data.objects["VRC Suit"]])
    with env.override(sidebar):
        result = bpy.ops.magic_fit.line_up('EXEC_DEFAULT', True, stretch=False)
    yield 0.4
    settings = bpy.context.scene.body_fit_brush
    check(result == {'FINISHED'} and marker_error(markers) > 0.01 and settings.lineup_stretch is False,
          "stretch off: joints keep the model's proportions, and the panel's setting follows")
    env.screenshot("lineup_03_no_stretch.png")
    settings.lineup_stretch = True

    # --- Fit Shape, from Adjust Last Operation: the model is reshaped a little, and the panel's setting follows.
    with env.override():
        bpy.ops.ed.undo()
    yield 0.4
    fixture.select([bpy.data.objects["VRC Suit"]])
    with env.override(sidebar):
        result = bpy.ops.magic_fit.line_up('EXEC_DEFAULT', True, fit_shape=True)
    yield 0.4
    error = marker_error(markers)
    check(result == {'FINISHED'} and 1e-4 < error < 0.02 and settings.lineup_fit_shape is True,
          "fit shape: the joints stay near the body's ({:.1f} mm), and the panel's setting follows".format(1000 * error))
    env.screenshot("lineup_04_fit_shape.png")
    settings.lineup_fit_shape = False

    # --- Not in Edit Mode.
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
    yield 0.2
    with env.override(sidebar):
        check(not bpy.ops.magic_fit.line_up.poll(), "edit mode: the button is greyed out")
    env.screenshot("lineup_05_edit_mode.png")
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    yield 0.2


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
