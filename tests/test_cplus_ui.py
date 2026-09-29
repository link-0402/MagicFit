"""Interactive test for Customize+ (the Customize+ tab of the Magic Fit sidebar, and its hotkey).

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_cplus_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set). The
test pastes through the clipboard and puts back what was there.
"""

import os
import sys
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import cplus as fixture  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import preferences, tabs  # noqa: E402
from magic_fit.cplus import rig, ui  # noqa: E402
from magic_fit.cplus.template import parse  # noqa: E402

SHOTS = os.environ.get("MF_SCREENSHOTS")
FAILURES = []
SUPERTHICC = fixture.SUPERTHICC
HOTKEY_DRAWS = []  # None per draw of the hotkey row, or the exception it raised
PANEL_DRAWS = []  # the same for the Customize+ panel


def _checked_draw(draw):
    def wrapper(self, context):
        try:
            draw(self, context)
            PANEL_DRAWS.append(None)
        except Exception:
            PANEL_DRAWS.append(traceback.format_exc())
            raise
    return wrapper


ui.VIEW3D_PT_customize_plus_sidebar.draw = _checked_draw(ui.VIEW3D_PT_customize_plus_sidebar.draw)


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


class VIEW3D_PT_hotkey_check(bpy.types.Panel):
    """The hotkey row of the add-on's preferences, drawn in the sidebar: those aren't drawn when the add-on
    is loaded from source."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = tabs.CATEGORY
    bl_label = "Hotkey (test)"
    bl_order = 10

    def draw(self, context):
        keymap = context.window_manager.keyconfigs.addon.keymaps[preferences.KEYMAP]
        item = next(item for item in keymap.keymap_items if item.idname == "magic_fit.cplus_toggle")
        try:
            preferences._draw_hotkey(self.layout, keymap, item, "Toggle Customize+")
            HOTKEY_DRAWS.append(None)
        except Exception:
            HOTKEY_DRAWS.append(traceback.format_exc())
            raise


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

    def hotkey(self):
        """Alt Shift C with the mouse over the viewport."""
        x = self.region.x + self.region.width // 2
        y = self.region.y + self.region.height // 2
        self.window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x, y=y)
        self.window.event_simulate(type='C', value='PRESS', x=x, y=y, alt=True, shift=True)
        self.window.event_simulate(type='C', value='RELEASE', x=x, y=y, alt=True, shift=True)


def body_moved():
    body = bpy.data.objects["Body"]
    return float(np.abs(fixture.evaluated_coordinates(body) - fixture.base_coordinates(body)).max())


def game_error(bust=None):
    """How far the body as shown is from the game's, with Superthicc+ and Bust Size ``bust`` (metres)."""
    armature, body = bpy.data.objects["Skeleton"], bpy.data.objects["Body"]
    game = fixture.skin(body, armature, fixture.game_matrices(armature, parse(SUPERTHICC), bust=bust))
    return float(np.abs(fixture.evaluated_coordinates(body) - game).max())


def select(objects, active):
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0
    clipboard = bpy.context.window_manager.clipboard
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=middle[0], y=middle[1])
    env.window.event_simulate(type='ESC', value='PRESS', x=middle[0], y=middle[1])  # close the splash screen
    env.window.event_simulate(type='ESC', value='RELEASE', x=middle[0], y=middle[1])
    yield 0.3

    scene = fixture.build()
    armature, body = scene["rig"], scene["body"]
    bpy.context.scene.magic_fit.target = body
    select([], None)
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
    bpy.context.workspace.magic_fit_tab = 'CPLUS'
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.5

    # --- The panel, before a template is pasted.
    keymap = bpy.context.window_manager.keyconfigs.addon.keymaps.get("3D View Generic")
    items = [item for item in keymap.keymap_items if item.idname == "magic_fit.cplus_toggle"] if keymap else []
    check(len(items) == 1 and items[0].type == 'C' and items[0].alt and items[0].shift and not items[0].ctrl,
          "hotkey: Alt Shift C toggles Customize+ in the 3D Viewport")
    check(HOTKEY_DRAWS and not any(HOTKEY_DRAWS),
          "hotkey: the preferences draw it, keyboard keys only ({!r})".format(next(filter(None, HOTKEY_DRAWS), None)))
    with env.override(sidebar):
        check(not bpy.ops.magic_fit.cplus_apply.poll(), "panel: Apply waits for a template")
    env.screenshot("cplus_01_empty.png")

    # --- Paste from the clipboard.
    bpy.context.window_manager.clipboard = SUPERTHICC
    with env.override(sidebar):
        result = bpy.ops.magic_fit.cplus_paste('EXEC_DEFAULT', True)
    yield 0.3
    settings = bpy.context.scene.magic_fit_cplus
    check(result == {'FINISHED'} and parse(settings.template).name == "Custom - Superthicc+",
          "paste: the template from the clipboard")
    bpy.context.window_manager.clipboard = "not a template"
    with env.override(sidebar):
        try:
            result = bpy.ops.magic_fit.cplus_paste('EXEC_DEFAULT', True)
        except RuntimeError:
            result = {'CANCELLED'}
    check(result == {'CANCELLED'} and parse(settings.template).name == "Custom - Superthicc+",
          "paste: other text is refused and the template kept")
    bpy.context.window_manager.clipboard = clipboard
    env.screenshot("cplus_02_pasted.png")

    # --- Apply, from Object Mode.
    with env.override(sidebar):
        result = bpy.ops.magic_fit.cplus_apply('EXEC_DEFAULT', True)
    yield 0.4
    moved = body_moved()
    check(result == {'FINISHED'} and rig.is_on(armature) and moved > 0.005,
          "apply: the body changes ({:.1f} mm at most)".format(1000 * moved))
    env.screenshot("cplus_03_applied.png")

    # --- Undo and redo.
    with env.override():
        bpy.ops.ed.undo()
    yield 0.4
    armature, body = bpy.data.objects["Skeleton"], bpy.data.objects["Body"]
    check(rig.rig_of(armature) is None and body_moved() < 1e-6 and body.modifiers["Armature"].show_viewport,
          "undo: no rig, the body as it was")
    with env.override():
        bpy.ops.ed.redo()
    yield 0.4
    armature, body = bpy.data.objects["Skeleton"], bpy.data.objects["Body"]
    check(rig.is_on(armature) and body_moved() > 0.005, "redo: applied again")

    # --- The hotkey, in Object Mode.
    env.hotkey()
    yield 0.3
    check(not rig.is_on(armature) and body_moved() < 1e-6, "hotkey: off")
    env.screenshot("cplus_04_hotkey_off.png")
    env.hotkey()
    yield 0.3
    check(rig.is_on(armature) and body_moved() > 0.005, "hotkey: on again")

    # --- In Pose Mode, posing the armature: the template stays on top, the hotkey still works.
    select([armature], armature)
    with env.override():
        bpy.ops.object.mode_set(mode='POSE')
    thigh = armature.pose.bones["j_asi_a_l"]
    thigh.rotation_mode = 'XYZ'
    thigh.rotation_euler = (0.6, 0.0, 0.3)
    yield 0.3
    error = float(np.abs(fixture.evaluated_coordinates(body) - fixture.skin(
        body, armature, fixture.game_matrices(armature, parse(SUPERTHICC)))).max())
    check(error < 2e-5, "pose mode: posed and applied like in the game ({:.1e} m)".format(error))
    env.screenshot("cplus_05_posed.png")
    env.hotkey()
    yield 0.3
    check(not rig.is_on(armature) and armature.mode == 'POSE', "pose mode: hotkey off, still in Pose Mode")
    check(np.allclose(thigh.rotation_euler, (0.6, 0.0, 0.3)), "pose mode: the pose is kept")
    env.hotkey()
    yield 0.3
    check(rig.is_on(armature), "pose mode: hotkey on")

    # --- Weight Paint with the armature: its bones can still be picked, the hotkey still works.
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    select([armature, body], body)
    with env.override():
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    yield 0.3
    check(bpy.context.pose_object == armature, "weight paint: the armature's bones are the ones to pick")
    env.hotkey()
    yield 0.3
    check(not rig.is_on(armature) and body.mode == 'WEIGHT_PAINT', "weight paint: hotkey off")
    env.hotkey()
    yield 0.3
    check(rig.is_on(armature) and body.mode == 'WEIGHT_PAINT', "weight paint: hotkey on")
    env.screenshot("cplus_06_weight_paint.png")
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')

    # --- Remove.
    with env.override(sidebar):
        result = bpy.ops.magic_fit.cplus_remove('EXEC_DEFAULT', True)
    yield 0.3
    check(result == {'FINISHED'} and rig.rig_of(armature) is None and body.modifiers["Armature"].show_viewport
          and len(body.modifiers) == 1, "remove: the rig is gone, the body follows its armature again")
    check(np.allclose(thigh.rotation_euler, (0.6, 0.0, 0.3)), "remove: the pose is kept")
    env.screenshot("cplus_07_removed.png")

    # --- Bust Size: applied with the template, the slider shown right away, undo.
    settings = bpy.context.scene.magic_fit_cplus
    settings.use_bust = True
    settings.bust_size = 90
    yield 0.2
    check(rig.rig_of(armature) is None, "bust size: nothing shows before Apply")
    with env.override(sidebar):
        result = bpy.ops.magic_fit.cplus_apply('EXEC_DEFAULT', True)
    yield 0.4
    error = game_error(bust=90)
    check(result == {'FINISHED'} and error < 2e-5, "bust size: applied with the template, like in the game "
                                                   "({:.1e} m)".format(error))
    env.screenshot("cplus_08_bust.png")
    with env.override():
        bpy.ops.ed.undo_push(message="Bust 90")
    settings.bust_size = 10
    yield 0.3
    error = game_error(bust=10)
    check(error < 2e-5, "bust size: the slider shows right away ({:.1e} m)".format(error))
    with env.override():
        bpy.ops.ed.undo_push(message="Bust 10")
        bpy.ops.ed.undo()
    yield 0.4
    settings = bpy.context.scene.magic_fit_cplus
    error = game_error(bust=90)
    check(settings.bust_size == 90 and error < 2e-5, "bust size: undo takes the slider and the body back "
                                                     "({:.1e} m)".format(error))
    settings.use_bust = False
    yield 0.3
    error = game_error()
    check(error < 2e-5, "bust size: unticked, the template alone ({:.1e} m)".format(error))
    env.screenshot("cplus_09_bust_off.png")
    check(PANEL_DRAWS and not any(PANEL_DRAWS),
          "the panel draws with the bust size ({!r})".format(next(filter(None, PANEL_DRAWS), None)))
    bpy.context.window_manager.clipboard = clipboard


def run():
    magic_fit.register()
    bpy.utils.register_class(VIEW3D_PT_hotkey_check)
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
