"""Headless tests of Line Up for Second Life imports (`magic_fit.secondlife`): recognizing Firestorm's prims,
placing them on a Second Life body, loading that body from a .blend, and Line Up running them.

Run: blender --background --factory-startup --python tests/test_secondlife_core.py
"""

import os
import sys
import tempfile
import time
import traceback

import bpy
import numpy as np
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "deps"))  # SciPy, for placing pieces

import lineup as fixture  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import lineup, operators, secondlife  # noqa: E402

FAILURES = []
SHARED = (0.3, -0.2, 0.05)


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def pieces(names=("top", "skirt", "boot_l"), shared=SHARED, body=None):
    """The Second Life body and prims cut from it: (body, rig, joints, [(prim, true centre)...])."""
    if body is None:
        fixture.clear()
        body, rig, joints = fixture.build_second_life_body()
    else:
        rig = joints = None
    regions = [(name, fixture.SL_REGIONS[name]) for name in names]
    return body, rig, joints, fixture.prims_from(body, regions, shared)


def world_bounds(obj):
    corners = np.array([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices])
    return corners.min(axis=0), corners.max(axis=0)


# -----------------------------------------------------------------------------
# Recognizing an import

def test_is_prim():
    body, _rig, _joints, made = pieces(("top",))
    prim = made[0][0]
    check(secondlife.is_prim(prim), "prim: a Firestorm piece is a prim")
    check(not secondlife.is_prim(body), "prim: the body is not")
    grouped = fixture.box_prim("Grouped", (0.2, 0.3, 0.4), SHARED)
    grouped.vertex_groups.new(name="mChest")
    check(not secondlife.is_prim(grouped), "prim: a piece with vertex groups is not")
    big = fixture.mesh_object("Big", [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], [], [])
    check(not secondlife.is_prim(big), "prim: a cube of size 2 is not")
    secondlife.bake(prim)
    bpy.context.view_layer.update()  # as after Ctrl+A: Line Up's panel asks once Blender has updated
    check(not secondlife.is_prim(prim), "prim: a piece with its scale applied is not")


def test_import_state():
    body, _rig, _joints, made = pieces()
    prims = [prim for prim, _centre in made]
    root = fixture.root_prim()
    check(secondlife.import_state(prims + [root]) == 'IMPORT', "state: prims and their root are an import")
    check(secondlife.import_state(prims + [body]) == 'MIXED', "state: prims with other meshes are mixed")
    check(secondlife.import_state([root]) == 'MIXED', "state: a lone root prim is not an import")
    check(secondlife.import_state([body]) == 'NONE', "state: no prims")


def test_split_import():
    body, _rig, _joints, made = pieces()
    root = fixture.root_prim()
    prims = [prim for prim, _centre in made]
    found, roots = secondlife.split_import(prims + [root])
    check(set(found) == set(prims) and roots == [root], "split: three pieces and their root")
    _body, _rig, _joints, other = pieces(("top", "skirt"), shared=(-0.4, 0.1, 0.0), body=body)
    second = fixture.root_prim()  # every export's root sits at the origin
    everything = prims + [root] + [prim for prim, _centre in other] + [second]
    found, roots = secondlife.split_import(everything)
    check(len(found) == 5 and set(roots) == {root, second}, "split: two imports, each with its root")


def test_import_state_speed():
    # Line Up's panel asks on every redraw: a large import must not slow it down.
    fixture.clear()
    rng = np.random.default_rng(1)
    prims = []
    for k in range(15):
        cloud = rng.uniform(-1.0, 1.0, (20000, 3))
        prims.append(fixture._prim("Piece {:d}".format(k), cloud, [], SHARED)[0])
    started = time.perf_counter()
    state = secondlife.import_state(prims)
    took = time.perf_counter() - started
    check(state == 'IMPORT' and took < 0.005, "state: 15 prims of 20,000 vertices in {:.1f} ms (under 5 ms)".format(
        1000.0 * took))


def test_bake():
    _body, _rig, _joints, made = pieces(("skirt",))
    prim = made[0][0]
    before = world_bounds(prim)
    secondlife.bake(prim)
    after = world_bounds(prim)
    check(prim.matrix_world == Matrix.Identity(4), "bake: the object's matrix is the identity")
    check(np.abs(before[0] - after[0]).max() < 1e-6 and np.abs(before[1] - after[1]).max() < 1e-6,
          "bake: the piece stays where it was")


# -----------------------------------------------------------------------------
# Placing a piece on the body

def surface(body):
    points = lineup.mesh_positions(body)
    tris = lineup.mesh_tris(body)
    return points, secondlife.surface_normals(points, tris)


def world_points(obj):
    return lineup.mesh_positions(obj)


def centre(points):
    return (points.min(axis=0) + points.max(axis=0)) / 2.0


def test_place_pieces():
    body, _rig, _joints, made = pieces()
    body_points, body_normals = surface(body)
    for prim, truth in made:
        points = world_points(prim)
        placed = secondlife.place(points, body_points, body_normals)
        off = np.linalg.norm(centre(points) + placed.offset - truth)
        check(placed.ok and off <= 0.005, "place: '{:s}' lands where it was cut ({:.1f} mm off; gap {:.1f} mm, "
              "inside {:.0f} %)".format(prim.name, 1000.0 * off, 1000.0 * placed.gap, 100.0 * placed.inside))


def test_place_loose():
    fixture.clear()
    body, _rig, _joints = fixture.build_second_life_body()
    made = fixture.prims_from(body, [("cardigan", fixture.SL_REGIONS["top"])], SHARED, out=0.025)
    body_points, body_normals = surface(body)
    prim, truth = made[0]
    points = world_points(prim)
    placed = secondlife.place(points, body_points, body_normals)
    off = np.linalg.norm(centre(points) + placed.offset - truth)
    # Every point aims for the skin, so a shell loose all round slides until one side touches (real loose tops
    # are tight at the shoulders): it's placed, within its looseness.
    check(placed.ok and off <= 0.03, "place: a top 25 mm off the skin all round is placed, within its looseness "
          "({:.1f} mm off; gap {:.1f} mm)".format(1000.0 * off, 1000.0 * placed.gap))


def test_place_mirrored():
    body, _rig, _joints, made = pieces(("boot_l", "boot_r"))
    body_points, body_normals = surface(body)
    sides = []
    for prim, truth in made:
        points = world_points(prim)
        placed = secondlife.place(points, body_points, body_normals)
        sides.append(float((centre(points) + placed.offset)[1]))
    check(sides[0] > 0.03 and sides[1] < -0.03, "place: each boot lands on its own side (y {:.3f}, {:.3f})".format(
        *sides))


def test_place_rejects():
    body, _rig, _joints, _made = pieces(())
    body_points, body_normals = surface(body)
    board = fixture.box_prim("Board", (0.47, 1.0, 0.7), SHARED)  # like a demo's sign
    placed = secondlife.place(world_points(board), body_points, body_normals)
    check(not placed.ok, "place: a demo sign doesn't fit (gap {:.1f} mm, inside {:.0f} %)".format(
        1000.0 * placed.gap, 100.0 * placed.inside))
    rings = [(0.3 * np.sin(a) * np.cos(b), 0.3 * np.sin(a) * np.sin(b), 0.3 * np.cos(a))
             for a in np.linspace(0.1, np.pi - 0.1, 12) for b in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)]
    placed = secondlife.place(np.array(rings) + np.array(SHARED), body_points, body_normals)
    check(not placed.ok, "place: a big ball doesn't fit (gap {:.1f} mm, inside {:.0f} %)".format(
        1000.0 * placed.gap, 100.0 * placed.inside))


def test_place_rejects_overhang():
    # Like the Lumi exports' 2.45 m helper: rings that hug the legs and a plate far below the feet.
    body, _rig, _joints, made = pieces(("boot_l",))
    body_points, body_normals = surface(body)
    boot = world_points(made[0][0])
    plate = np.array([(x, y, -1.2) for x in np.linspace(-0.3, 0.3, 8) for y in np.linspace(-0.3, 0.3, 8)])
    placed = secondlife.place(np.concatenate([boot, plate]), body_points, body_normals)
    check(not placed.ok, "place: a piece reaching 1 m past the body doesn't fit (gap {:.1f} mm, past the body "
          "{:.0f} cm)".format(1000.0 * placed.gap, 100.0 * placed.outside))


def test_place_speed():
    body, _rig, _joints, made = pieces()
    body_points, body_normals = surface(body)
    started = time.perf_counter()
    for prim, _truth in made:
        secondlife.place(world_points(prim), body_points, body_normals)
    took = time.perf_counter() - started
    check(took < 3.0, "place: three pieces in {:.2f} s (under 3 s)".format(took))


# -----------------------------------------------------------------------------
# Loading the body

DATA = ("objects", "meshes", "armatures", "materials", "images", "collections", "libraries")


def snapshot():
    return {name: sorted(item.name for item in getattr(bpy.data, name)) for name in DATA}


def body_file(name="sl_body.blend"):
    path = os.path.join(tempfile.gettempdir(), name)
    fixture.clear()
    fixture.write_second_life_blend(path)
    return path


def test_body_setting():
    check(secondlife.body_setting() == "", "setting: empty while the add-on isn't enabled as such (in the tests)")
    properties = magic_fit.preferences.MagicFitPreferences.bl_rna.properties
    check("second_life_body" in properties and properties["second_life_body"].subtype == 'FILE_PATH',
          "setting: the preferences have a Second Life Body file")


def test_loaded_body():
    path = body_file()
    before = snapshot()
    with secondlife.loaded_body(path) as loaded:
        check(loaded.body.name == "SL Body" and [obj.name for obj in loaded.feet] == ["SL Feet"] and
              loaded.armature.name == "SL Rig", "load: picks the body, its feet and its rig (not the decoy)")
        check(loaded.armature.data.pose_position == 'REST', "load: the rig at its rest pose")
        check(loaded.body.name in bpy.context.view_layer.objects, "load: the body is in the view layer")
    check(snapshot() == before, "load: nothing loaded stays behind")


def test_loaded_body_error():
    path = body_file()
    before = snapshot()
    try:
        with secondlife.loaded_body(path):
            raise RuntimeError("test")
    except RuntimeError:
        pass
    check(snapshot() == before, "load: nothing stays behind after an error")


def test_missing_body():
    fixture.clear()
    try:
        with secondlife.loaded_body(os.path.join(tempfile.gettempdir(), "no such body.blend")):
            pass
        check(False, "load: refuses a missing file")
    except secondlife.SecondLifeError as error:
        check("doesn't exist" in str(error), "load: refuses a missing file ({!s})".format(error))
    wrong = os.path.join(tempfile.gettempdir(), "sl_not_a_blend.dae")
    with open(wrong, "w") as handle:
        handle.write("<COLLADA/>")
    try:
        with secondlife.loaded_body(wrong):
            pass
        check(False, "load: refuses a file that isn't a .blend")
    except secondlife.SecondLifeError as error:
        check("isn't a .blend" in str(error), "load: refuses a file that isn't a .blend ({!s})".format(error))
    path = os.path.join(tempfile.gettempdir(), "sl_prims_only.blend")
    fixture.prims_only_blend(path)
    before = snapshot()
    try:
        with secondlife.loaded_body(path):
            pass
        check(False, "load: refuses a file without a Second Life body")
    except secondlife.SecondLifeError as error:
        check("No Second Life body" in str(error), "load: refuses a file without a Second Life body ({!s})".format(
            error))
    check(snapshot() == before, "load: nothing stays behind from a file without a body")
    check("No Second Life body" in secondlife.describe(path) and "SL Body" in secondlife.describe(body_file()),
          "describe: says what the file holds")


# -----------------------------------------------------------------------------
# Line Up runs Second Life imports

def import_scene(names=("top", "skirt")):
    """A Second Life Body file, the FFXIV body as Line Up's target, and an import at one point: pieces cut
    from a Second Life body (then removed), a board and the root prim. Returns (path, body, pieces, board,
    root)."""
    path = body_file()
    body, _skeleton, _joints = fixture.build_body()
    bpy.context.scene.magic_fit.target = body
    sl_body, rig, _sl_joints = fixture.build_second_life_body()
    regions = [(name, fixture.SL_REGIONS[name]) for name in names]
    made = [prim for prim, _centre in fixture.prims_from(sl_body, regions, SHARED)]
    for obj in (sl_body, rig):
        data = obj.data
        bpy.data.objects.remove(obj)
        (bpy.data.armatures if isinstance(data, bpy.types.Armature) else bpy.data.meshes).remove(data)
    board = fixture.box_prim("Board", (0.47, 1.0, 0.7), SHARED)  # like a demo's sign
    root = fixture.root_prim()
    fixture.select(made + [board, root])
    settings = bpy.context.scene.body_fit_brush
    settings.lineup_stretch = True
    settings.lineup_fit_shape = True
    return path, body, made, board, root


def with_setting(path):
    secondlife.body_setting = (lambda: path) if path is not None else REAL_SETTING


REAL_SETTING = secondlife.body_setting


def gap_to(body, obj):
    from mathutils.kdtree import KDTree
    points, _tris = lineup.body_surface(body, bpy.context.evaluated_depsgraph_get())
    tree = KDTree(len(points))
    for k, point in enumerate(points):
        tree.insert(point, k)
    tree.balance()
    return float(np.median([tree.find(point)[2] for point in world_points(obj)]))


def test_ready_messages():
    path, body, made, board, root = import_scene(("top", "skirt", "boot_l"))
    try:
        with_setting("")
        check(operators.check_lineup_ready(bpy.context) ==
              "Set the Second Life Body in Magic Fit's preferences to line up Second Life pieces",
              "ready: asks for the Second Life Body")
        with_setting(path)
        check(operators.check_lineup_ready(bpy.context) is None, "ready: with the Second Life Body set")
        check(operators.lineup_summary(bpy.context) == "4 Second Life pieces",
              "ready: summary counts the pieces ({!s})".format(operators.lineup_summary(bpy.context)))
        other = fixture.mesh_object("Other", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)], [{"j_kosi": 1.0}])
        fixture.select(made + [other])
        check(operators.check_lineup_ready(bpy.context) ==
              "Line up the Second Life pieces on their own, without other meshes", "ready: no mix with other meshes")
    finally:
        with_setting(None)


def test_line_up_import():
    path, body, made, board, root = import_scene()
    root_matrix = root.matrix_world.copy()
    root_points = world_points(root)
    with_setting(path)
    try:
        before = snapshot()
        result = secondlife.line_up(bpy.context, made + [board, root], body, True, True)
        check(sorted(result.placed) == sorted(obj.name for obj in made) and result.failed == ["Board"] and
              result.skipped == ["SL Root"], "line up: placed {!s}, not placed {!s}, skipped {!s}".format(
                  result.placed, result.failed, result.skipped))
        for obj in made:
            gap = gap_to(body, obj)
            check(gap < 0.015, "line up: '{:s}' lies on the FFXIV body ({:.1f} mm median)".format(obj.name,
                                                                                            1000.0 * gap))
        check(all(not obj.vertex_groups for obj in made + [board]), "line up: no piece keeps vertex groups")
        check(all(obj.matrix_world == Matrix.Identity(4) for obj in made + [board]),
              "line up: every piece's scale went into its mesh")
        check(root.matrix_world == root_matrix and np.abs(world_points(root) - root_points).max() < 1e-9,
              "line up: the root prim stays as it was")
        check(snapshot() == before, "line up: nothing of the Second Life body stays behind")
        path, body, made, board, root = import_scene()
        with_setting(path)
        check(bpy.ops.magic_fit.line_up() == {'FINISHED'}, "line up: the operator lines up an import")
    finally:
        with_setting(None)


def test_line_up_failure_cleans_up():
    path, body, made, board, root = import_scene()
    matrices = [obj.matrix_world.copy() for obj in made]
    points = [world_points(obj) for obj in made]
    with_setting(path)
    real_plan = lineup.plan

    def failing(*_args, **_kwargs):
        raise lineup.LineUpError("test")

    lineup.plan = failing
    try:
        before = snapshot()
        try:
            outcome = bpy.ops.magic_fit.line_up()
        except RuntimeError as error:  # an operator's error report, in the background
            outcome = {'CANCELLED'} if "test" in str(error) else {str(error)}
        check(outcome == {'CANCELLED'}, "failure: the operator cancels with the error")
        check(all(not obj.vertex_groups for obj in made), "failure: no borrowed groups stay")
        check(all(obj.matrix_world == m for obj, m in zip(made, matrices)) and
              all(np.abs(world_points(obj) - p).max() < 1e-6 for obj, p in zip(made, points)),
              "failure: the pieces are back as they were")
        check(snapshot() == before, "failure: nothing of the Second Life body stays behind")
    finally:
        lineup.plan = real_plan
        with_setting(None)


def main():
    magic_fit.register()
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception:
            FAILURES.append(test.__name__ + " raised")
            traceback.print_exc()
        finally:
            if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
    print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
    sys.exit(1 if FAILURES else 0)


main()
