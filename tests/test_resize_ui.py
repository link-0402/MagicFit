"""Interactive test for Resize (the Resize section of the Body Fit panel), driven from the sidebar.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_resize_ui.py

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

import resize  # noqa: E402
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


def keys_of(obj):
    keys = obj.data.shape_keys
    return [] if keys is None else [block.name for block in keys.key_blocks]


def steps():
    env = Env()
    bpy.context.preferences.view.smooth_view = 0
    middle = (env.region.x + env.region.width // 2, env.region.y + env.region.height // 2)
    env.window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=middle[0], y=middle[1])
    env.window.event_simulate(type='ESC', value='PRESS', x=middle[0], y=middle[1])  # close the splash screen
    env.window.event_simulate(type='ESC', value='RELEASE', x=middle[0], y=middle[1])
    yield 0.3

    body, garments = resize.build()
    settings = bpy.context.scene.body_fit_brush
    settings.resize_from_object = body
    settings.resize_use_keys = True
    settings.resize_to_key = "Small"
    with env.override():
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.view3d.view_selected()
        bpy.ops.ed.undo_push(message="Start")
    space = env.area.spaces.active
    space.show_region_ui = True
    yield 0.3
    sidebar = next(r for r in env.area.regions if r.type == 'UI')
    try:
        sidebar.active_panel_category = "Magic Fit"
    except AttributeError:
        pass
    bpy.context.workspace.magic_fit_tab = 'RESIZE'
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.5
    panel = getattr(bpy.types, "VIEW3D_PT_resize_sidebar", None)
    with bpy.context.temp_override(window=env.window, area=env.area, region=sidebar):
        shown = panel is not None and panel.poll(bpy.context)
    check(shown, "panel: Auto Resize is on its own tab")
    check(operators.check_resize_ready(bpy.context) is None, "panel: ready with a From body, a To key and meshes selected")
    env.screenshot("resize_01_panel.png")

    # --- Object Mode: the button in the sidebar.
    before = resize.positions(garments["Suit"])
    with env.override(sidebar):
        check(bpy.ops.magic_fit.resize.poll(), "button: ready in Object Mode")
        result = bpy.ops.magic_fit.resize('EXEC_DEFAULT', True)
    yield 0.5
    suit = bpy.data.objects["Suit"]
    check(result == {'FINISHED'} and keys_of(suit) == ["Basis", "Small"], "button: added the key")
    moved = np.linalg.norm(resize.positions(suit, "Small") - before, axis=1)
    check(moved.max() > 0.03, "button: the suit shrank with the breasts ({:.0f} mm)".format(moved.max() * 1000))
    env.screenshot("resize_02_resized.png")

    with env.override():
        bpy.ops.ed.undo()
    yield 0.4
    check(keys_of(bpy.data.objects["Suit"]) == [] and keys_of(bpy.data.objects["Bridge"]) == [],
          "undo: the keys are gone")
    with env.override():
        bpy.ops.ed.redo()
    yield 0.4
    check(keys_of(bpy.data.objects["Suit"]) == ["Basis", "Small"], "redo: the key is back")

    # --- Edit Mode: resize the mesh being edited, and stay in Edit Mode.
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj.name == "Bridge")
    bpy.context.view_layer.objects.active = bpy.data.objects["Bridge"]
    with env.override():
        bpy.ops.object.mode_set(mode='EDIT')
    yield 0.3
    # Undo reloads the scene: look its settings up again.
    bpy.context.scene.body_fit_brush.resize_key_name = "Rue"
    with env.override(sidebar):
        result = bpy.ops.magic_fit.resize('EXEC_DEFAULT', True)
    yield 0.5
    bridge = bpy.data.objects["Bridge"]
    check(result == {'FINISHED'} and bpy.context.object.mode == 'EDIT', "edit mode: resized, still in Edit Mode")
    blocks = bridge.data.shape_keys.key_blocks
    check(keys_of(bridge) == ["Basis", "Small", "Rue"] and bridge.active_shape_key.name == "Rue"
          and blocks["Small"].value == 0.0, "edit mode: the new key is active, so Body Fit works on it")
    env.screenshot("resize_03_edit_mode.png")
    with env.override():
        bpy.ops.object.mode_set(mode='OBJECT')
    yield 0.2

    # --- Another body without a UV map (another game's), matched by shape.
    settings = bpy.context.scene.body_fit_brush
    settings.resize_from_object = resize.coarse_body("Plain Body", uv=None)
    settings.resize_to_object = bpy.data.objects["Body"]
    settings.resize_match = 'SHAPE'
    settings.resize_key_name = "Shape"
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj.name == "Suit")
    bpy.context.view_layer.objects.active = bpy.data.objects["Suit"]
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.3
    check(operators.check_resize_ready(bpy.context) is None, "shape: ready with a body matched by shape")
    env.screenshot("resize_04_shape.png")
    with env.override(sidebar):
        result = bpy.ops.magic_fit.resize('EXEC_DEFAULT', True)
    yield 0.5
    check(result == {'FINISHED'} and keys_of(bpy.data.objects["Suit"])[-1] == "Shape",
          "shape: added the key, the bodies matched by shape")

    # --- A devkit body: other sizes, without switching the devkit.
    body, _garments = resize.devkit_body()
    settings = bpy.context.scene.body_fit_brush
    for prop in ("resize_to_object", "resize_match", "resize_key_name"):
        settings.property_unset(prop)
    settings.resize_use_keys = False
    settings.resize_from_object = body
    settings.resize_use_sizes = True
    with env.override(sidebar):
        bpy.ops.magic_fit.resize_pick(setting="chest_size", option=1)
        bpy.ops.magic_fit.resize_pick(setting="rue", option=1)
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.3
    check(operators.check_resize_ready(bpy.context) is None, "sizes: ready with Sizes")
    env.screenshot("resize_05_sizes.png")
    keys_before = [(block.value, block.mute) for block in body.data.shape_keys.key_blocks]
    with env.override(sidebar):
        result = bpy.ops.magic_fit.resize('EXEC_DEFAULT', True)
    yield 0.5
    check(result == {'FINISHED'} and keys_of(bpy.data.objects["Suit"])[-1] == "Medium Rue",
          "sizes: added the key, named after the sizes")
    body = bpy.data.objects["Body"]
    check([(block.value, block.mute) for block in body.data.shape_keys.key_blocks] == keys_before,
          "sizes: the devkit body wasn't switched")

    # --- Several sizes: Shift-click adds a value; each combination is a size.
    with env.override(sidebar):
        bpy.ops.magic_fit.resize_pick(setting="chest_size", option=2, extend=True)
        bpy.ops.magic_fit.resize_pick(setting="rue", option=0, extend=True)
    for area in env.window.screen.areas:
        area.tag_redraw()
    yield 0.3
    env.screenshot("resize_06_several_sizes.png")
    with env.override(sidebar):
        result = bpy.ops.magic_fit.resize('EXEC_DEFAULT', True)
    yield 0.5
    suit = bpy.data.objects["Suit"]
    check(result == {'FINISHED'} and {"Medium", "Small", "Medium Rue", "Small Rue"} <= set(keys_of(suit))
          and keys_of(suit).count("Medium Rue") == 1, "several: a key per size ({!s})".format(keys_of(suit)))
    with env.override():
        bpy.ops.ed.undo()
    yield 0.4
    check("Small Rue" not in keys_of(bpy.data.objects["Suit"]) and "Medium Rue" in keys_of(bpy.data.objects["Suit"]),
          "several: undone in one step")


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
