"""Interactive test for the "Magic Fit" sidebar tab: its tabs, the Weights tab's sub-tabs, the panels
on each, the buttons that start the brushes, and the links.

Opens a Blender window for a few seconds. Run from the repository root:
    blender --factory-startup --enable-event-simulate --python tests/test_sidebar_ui.py

Screenshots are written to the directory in the MF_SCREENSHOTS environment variable (if set).
"""

import os
import sys
import traceback

import bmesh
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(ROOT, "deps")):
    sys.path.insert(0, os.path.join(ROOT, "deps"))
sys.path.insert(0, ROOT)

import magic_fit  # noqa: E402
from magic_fit import operators, tabs, tool  # noqa: E402
from magic_fit.cplus import ui as cplus_ui  # noqa: E402
from magic_fit.robust_transfer import ui as transfer_ui  # noqa: E402

SHOTS = os.environ.get("MF_SCREENSHOTS")
FAILURES = []
DRAWN = set()  # labels of the sidebar panels drawn since it was cleared
DRAW_ERRORS = []

TABS_PANEL = "Magic Fit"  # the tabs, the Weights sub-tabs and the Body row
# The panels on each tab (and sub-tab of the Weights tab), top to bottom, besides the tabs.
EXPECTED = (
    ('WEIGHTS', 'GENERAL', ["Weight Transfer", "Weight Brushes"]),
    ('WEIGHTS', 'SKIRT', ["Skirt Weights", "Skirt Brush"]),
    ('WEIGHTS', 'HEELS', ["Heel Weights", "Heels Brush"]),
    ('WEIGHTS', 'HAIR', ["Hair Weights", "Hair Brush"]),
    ('WEIGHTS', 'FACE', ["Face Weights", "Face Repairs", "Test Poses", "Face Brush"]),
    ('FIT', None, ["Body Fit", "Fit Move", "Resize", "Clipping"]),
    ('LINE_UP', None, ["Line Up"]),
    ('RELAX', None, ["Texture Relax"]),
    ('CPLUS', None, ["Customize+"]),
)


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def watch(cls):
    """Record which sidebar panels draw, and the exceptions of their draw, which Blender would only print."""
    draw = cls.draw

    def wrapper(self, context):
        DRAWN.add(cls.bl_label or cls.__name__)
        try:
            draw(self, context)
        except Exception:
            DRAW_ERRORS.append(traceback.format_exc())
            raise

    cls.draw = wrapper


def sphere(name, radius):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm.loops.layers.uv.new("UVMap")
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=radius, calc_uvs=True)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def active_tool(mode):
    tool_ = bpy.context.workspace.tools.from_space_view3d_mode(mode, create=False)
    return tool_.idname if tool_ else None


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

    def use_tool(tool_name):
        with bpy.context.temp_override(window=window, area=area, region=sidebar):
            bpy.ops.magic_fit.use_tool('EXEC_DEFAULT', True, tool=tool_name)

    def redraw():
        DRAWN.clear()
        for each in window.screen.areas:
            each.tag_redraw()
        return 0.3

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    body = sphere("Body", 1.0)
    garment = sphere("Garment", 1.02)
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)
    scene = bpy.context.scene
    settings = scene.magic_fit
    settings.target = body
    workspace = bpy.context.workspace

    area.spaces.active.show_region_ui = True
    yield 0.3
    sidebar.active_panel_category = "Magic Fit"
    yield redraw()
    check(sidebar.active_panel_category == "Magic Fit", "sidebar shows the Magic Fit tab")
    check(workspace.magic_fit_tab == 'WEIGHTS' and settings.weights_tab == 'GENERAL',
          "it starts on the Weights tab, General sub-tab")
    check(not scene.magic_fit_clipping.show, "clipping marks are off by default")
    with bpy.context.temp_override(window=window, area=area, region=sidebar):
        per_row = tool._tabs_per_row(bpy.context, [item[1] for item in tabs.TABS])
    check(sidebar.width == 280 and per_row == 3,
          "default sidebar: the tabs take two rows, three in the first ({:d} px, {:d})".format(sidebar.width, per_row))
    fit = scene.body_fit_brush
    check(fit.mode == 'FIT' and abs(fit.offset - 0.001) < 1e-9 and not fit.lineup_stretch and not fit.resize_use_keys
          and scene.magic_fit_transfer.inpaint_mode == 'COMBINED',
          "defaults: Fit at 1 mm, no stretched limbs, no shapekeys, Combined inpainting")

    # --- Every tab and sub-tab shows its own panels, and only those.
    for number, (tab, sub, expected) in enumerate(EXPECTED, 1):
        workspace.magic_fit_tab = tab
        if sub is not None:
            settings.weights_tab = sub
        yield redraw()
        name = tab.lower() + ("_" + sub.lower() if sub else "")
        check(DRAWN == {TABS_PANEL, *expected}, "{:s}: panels {!r}".format(name, sorted(DRAWN)))
        screenshot("sidebar_{:02d}_{:s}.png".format(number, name))
    check(not DRAW_ERRORS, "the panels drew without errors")

    # --- The icon at the right end of the Magic Fit header opens the links.
    x, y = sidebar.x + sidebar.width - 59, sidebar.y + sidebar.height - 14
    DRAWN.clear()
    window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x, y=y)
    yield 0.1
    window.event_simulate(type='LEFTMOUSE', value='PRESS', x=x, y=y)
    window.event_simulate(type='LEFTMOUSE', value='RELEASE', x=x, y=y)
    yield 0.5
    check("Links" in DRAWN, "clicking the header's icon opens the links")
    check([url for _label, _icon, url in tool.LINKS] == [
        "https://www.xivmodarchive.com/user/124593",
        "https://github.com/link-0402/MagicFit",
        "https://bsky.app/profile/xiv-luci.bsky.social",
        "https://ko-fi.com/luci_xiv",
    ], "the links go to XIV Mod Archive, GitHub, Bluesky and Ko-fi")
    if SHOTS:
        with bpy.context.temp_override(window=window, area=area):
            bpy.ops.screen.screenshot(filepath=os.path.join(SHOTS, "sidebar_10_links.png"))
    window.event_simulate(type='ESC', value='PRESS', x=x, y=y)
    window.event_simulate(type='ESC', value='RELEASE', x=x, y=y)
    yield redraw()
    check(TABS_PANEL in DRAWN and not DRAW_ERRORS, "the links closed again, and the panel is still open")

    # --- Clicking a tab shows it, and adds no undo step: one undo still goes back past the last change.
    workspace.magic_fit_tab = 'WEIGHTS'
    for radius in (40, 41):
        settings.radius = radius
        with bpy.context.temp_override(window=window, area=area):
            bpy.ops.ed.undo_push(message="Radius {:d}".format(radius))
    yield redraw()
    # The tabs are at the top of the sidebar's first panel, below its header. In the default sidebar Body Fit is
    # the middle one of the first row.
    x, y = sidebar.x + 128, sidebar.y + sidebar.height - 48
    window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x, y=y)
    yield 0.1
    window.event_simulate(type='LEFTMOUSE', value='PRESS', x=x, y=y)
    window.event_simulate(type='LEFTMOUSE', value='RELEASE', x=x, y=y)
    yield redraw()
    check(workspace.magic_fit_tab == 'FIT', "clicking the Body Fit tab shows it")
    with bpy.context.temp_override(window=window, area=area):
        bpy.ops.ed.undo()
    yield 0.3
    # Undo reloads the scene: look its settings up again.
    settings = bpy.context.scene.magic_fit
    check(settings.radius == 40 and workspace.magic_fit_tab == 'FIT',
          "switching tabs added no undo step (radius {:d} after one undo)".format(settings.radius))
    settings.radius = 50

    # --- The Weights sub-tabs are the brush's modes; General brings back the last of Copy and Straighten.
    workspace.magic_fit_tab = 'WEIGHTS'
    settings.mode = 'STRAIGHTEN'
    check(settings.weights_tab == 'GENERAL', "sub-tab: Straighten is on General")
    settings.weights_tab = 'SKIRT'
    check(settings.mode == 'SKIRT', "sub-tab: Skirt sets the brush to Skirt")
    settings.weights_tab = 'GENERAL'
    check(settings.mode == 'STRAIGHTEN', "sub-tab: General brings Straighten back")
    settings.mode = 'HEELS'
    check(settings.weights_tab == 'HEELS', "sub-tab: follows the mode set in the tool header")
    settings.weights_tab = 'HAIR'
    with bpy.context.temp_override(window=window, area=area):
        problem = operators.check_ready(bpy.context, settings)
    check(settings.mode == 'HAIR' and problem is not None and "Hair Weights" in problem,
          "sub-tab: Hair asks for Hair Weights before the brush paints a mesh without them ({!r})".format(problem))
    settings.weights_tab = 'FACE'
    with bpy.context.temp_override(window=window, area=area):
        problem = operators.check_ready(bpy.context, settings)
    check(settings.mode == 'FACE' and problem is not None and "Face Weights" in problem,
          "sub-tab: Face asks for Face Weights before the brush paints a mesh without them ({!r})".format(problem))
    settings.weights_tab = 'GENERAL'
    settings.mode = 'COPY'

    # --- The buttons that start the brushes switch the mode, the tool and the tab.
    use_tool('FIT')
    yield redraw()
    check(bpy.context.mode == 'EDIT_MESH', "Fit to Body switched to Edit Mode")
    check(active_tool('EDIT_MESH') == "magic_fit.body_fit_tool", "Fit to Body picked the Body Fit brush")
    check(workspace.magic_fit_tab == 'FIT' and "Body Fit" in DRAWN, "Fit to Body shows the Body Fit tab")
    screenshot("sidebar_20_body_fit_brush.png")

    use_tool('MOVE')
    yield redraw()
    check(active_tool('EDIT_MESH') == "magic_fit.fit_move_tool" and workspace.magic_fit_tab == 'FIT',
          "Fit Move picked the tool and shows the Body Fit tab")
    screenshot("sidebar_22_fit_move.png")

    use_tool('RELAX')
    yield redraw()
    check(active_tool('EDIT_MESH') == "magic_fit.texture_relax_tool" and workspace.magic_fit_tab == 'RELAX',
          "Relax Texture picked the brush and shows its tab")
    screenshot("sidebar_21_texture_relax_brush.png")

    workspace.magic_fit_tab = 'WEIGHTS'
    settings.weights_tab = 'SKIRT'
    use_tool('WEIGHT')
    yield redraw()
    check(bpy.context.mode == 'PAINT_WEIGHT', "Paint Skirt Weights switched to Weight Paint mode")
    check(active_tool('PAINT_WEIGHT') == "magic_fit.tool", "Paint Skirt Weights picked the Weight Brushes")
    check(settings.mode == 'SKIRT' and DRAWN == {TABS_PANEL, "Skirt Weights", "Skirt Brush"},
          "the Skirt sub-tab shows the brush in Skirt mode ({!r})".format(sorted(DRAWN)))
    screenshot("sidebar_22_skirt_brush.png")

    settings.weights_tab = 'GENERAL'
    yield redraw()
    check(settings.mode == 'COPY' and "Weight Brushes" in DRAWN, "General brings the Copy brush back while painting")
    screenshot("sidebar_23_copy_brush.png")

    # --- A wider sidebar (dragged by its edge, next to the panels): the tabs fit in one row, with icons.
    y = sidebar.y + sidebar.height - 100
    before = sidebar.width
    # With region overlap, the edge to drag is 0.4 widget units into the sidebar.
    for offset in (8, 7, 9, 6, 10, 0):
        edge = sidebar.x + offset
        window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=edge + 30, y=y)
        yield 0.1
        window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=edge, y=y)
        yield 0.2
        window.event_simulate(type='LEFTMOUSE', value='PRESS', x=edge, y=y)
        yield 0.1
        for step in range(1, 13):
            window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=edge - 25 * step, y=y)
            yield 0.03
        window.event_simulate(type='LEFTMOUSE', value='RELEASE', x=edge - 300, y=y)
        yield redraw()
        if sidebar.width > before + 100:
            break
    print("sidebar width: {:d} -> {:d}".format(before, sidebar.width))
    if sidebar.width > before + 100:
        with bpy.context.temp_override(window=window, area=area, region=sidebar):
            widths = tool._label_widths(bpy.context, [item[1] for item in tabs.TABS])
        room = sidebar.width - tool.SIDEBAR_MARGIN
        check(sum(widths) <= room, "wide sidebar: the tabs fit in one row with labels and icons")
    screenshot("sidebar_24_wide.png")

    with bpy.context.temp_override(window=window, area=area):
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = None
    yield redraw()
    with bpy.context.temp_override(window=window, area=area, region=sidebar):
        check(not bpy.ops.magic_fit.use_tool.poll(), "buttons are disabled without an active mesh")
    screenshot("sidebar_25_nothing_selected.png")
    check(not DRAW_ERRORS, "the panels drew without errors afterwards too")


def run():
    for module in (tool, transfer_ui, cplus_ui):
        for cls in module.classes:
            if getattr(cls, "bl_category", None) == tabs.CATEGORY:
                watch(cls)
    watch(tool.VIEW3D_MT_magic_fit_links)
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
