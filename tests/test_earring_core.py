"""Headless tests for Earring weights (the Weights tab's Earrings sub-tab: Earring Weights and the brush's
Earring mode).

Run from the repository root:
    blender -b --factory-startup --python tests/test_earring_core.py
"""

import os
import sys
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import magic_fit  # noqa: E402
from magic_fit import earring, hairing, operators, straighten  # noqa: E402

FAILURES = []
MIQOTE = earring.race_key("c0801", "f0002")
VIERA = earring.race_key("c1801", "f0002")


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for armature in list(bpy.data.armatures):
        bpy.data.armatures.remove(armature)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def use_settings(**values):
    settings = bpy.context.scene.magic_fit
    for prop in ("earring_race", "earring_bones", "group_mode"):
        settings.property_unset(prop)
    settings.mode = 'EARRING'
    for name, value in values.items():
        setattr(settings, name, value)
    return settings


def head_space(points, rig):
    """Points in the head bone's rest space (``points``, (n, 3)) in world space."""
    rest = rig.head.rest
    return np.asarray(points, dtype=np.float64) @ rest[:3, :3].T + rest[:3, 3] + rig.offset


def new_armature(names=("j_kubi",), offset=(0.0, 0.0, 0.0)):
    """An armature with some of the race's body bones at the game's rest (moved by ``offset``)."""
    reference = hairing.Reference.get()
    rig = hairing.HairRig("c0801", 0, reference.body_bones("c0801"), None, ()).moved(offset)
    data = bpy.data.armatures.new("Body")
    armature = bpy.data.objects.new("Body", data)
    bpy.context.scene.collection.objects.link(armature)
    hairing.ensure_bones(bpy.context, armature, rig, list(names))
    return armature


def new_earrings(rig, armature=None, name="Earrings"):
    """Two dangling strips, one under each lobe, from 5 mm above it to 4 cm below."""
    local = []
    for lobe in rig.lobes:
        for h in np.linspace(-0.005, 0.04, 10):
            for dz in (-0.001, 0.001):
                local.append(lobe + np.array((0.0, h, dz)))
    points = head_space(np.array(local), rig)
    faces = []
    for side in range(2):
        base = side * 20
        for k in range(9):
            a = base + 2 * k
            faces.append((a, a + 1, a + 3, a + 2))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points.tolist(), [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if armature is not None:
        obj.parent = armature
        hairing.add_armature_modifier(obj, armature)
    return obj, np.array(local)


def weights_of(obj):
    result = {group.name: np.zeros(len(obj.data.vertices)) for group in obj.vertex_groups}
    names = {group.index: group.name for group in obj.vertex_groups}
    for vert in obj.data.vertices:
        for elem in vert.groups:
            result[names[elem.group]][vert.index] = elem.weight
    return result


def select_only(objects, active):
    for obj in bpy.context.scene.objects:
        obj.select_set(obj in objects)
    bpy.context.view_layer.objects.active = active


# -----------------------------------------------------------------------------
# The weight split

LOBES = earring.lobes(MIQOTE)


def split_at(heights, side=1, count=4, sideways=0.0, length=None, parts=None, use_joints=True):
    """The split of points ``heights`` below the lobe of ``side`` (0 left, 1 right), on an earring ``length``
    long (a last point that far down, left out of the result) when given."""
    lobe = LOBES[side]
    heights = list(heights) + ([length] if length is not None else [])
    points = np.array([lobe + np.array((sideways, h, 0.0)) for h in heights])
    if parts is not None and length is not None:
        parts = list(parts) + [parts[-1]]
    split = earring.earring_weights(points, LOBES, count, parts, use_joints)
    if length is not None:
        split.weights = split.weights[:-1]
    return split


def column(split, letter, side):
    return split.names.index(earring.bone_name(letter, side))


def test_hook_at_a():
    split = split_at([0.0, -0.002, -0.02, 0.004], length=0.04)
    a = column(split, "a", "r")
    check(np.allclose(split.weights[:, a], 1.0) and np.allclose(split.weights.sum(1), 1.0),
          "split: the hook at and above the lobe, and the top of a's share, are all a")


def test_even_spacing():
    # 4 cm, 4 bones: each takes a centimetre, all of it around its middle.
    split = split_at([0.01, 0.0125, 0.015, 0.02], length=0.04)
    w = split.weights
    a, b, c = (column(split, letter, "r") for letter in "abc")
    check(np.allclose(w[0, [a, b]], (0.5, 0.5)), "even: at the end of a's centimetre, half a, half b")
    check(np.allclose(w[1, [a, b]], (0.25, 0.75)), "even: a quarter further, 1/4 a, 3/4 b")
    check(np.isclose(w[2, b], 1.0), "even: in the middle of b's centimetre, all b")
    check(np.allclose(w[3, [b, c]], (0.5, 0.5)), "even: b hands over to c at 2 cm")
    check(np.allclose(w.sum(1), 1.0) and (np.count_nonzero(w, axis=1) <= 2).all(),
          "even: every vertex adds up to 1, on two bones at most")
    longer = split_at([0.02, 0.04], length=0.08)
    check(np.allclose(longer.weights[0, [a, b]], (0.5, 0.5)) and np.allclose(longer.weights[1, [b, c]], (0.5, 0.5)),
          "even: on an earring twice as long, the bones change twice as far apart")
    check(np.isclose(longer.lengths[1], 0.08) and longer.joints[1] == 0, "even: the length is the lowest vertex")


def test_last_bone_holds_the_tail():
    for count, letter, tail in ((4, "d", 0.035), (3, "c", 0.034), (2, "b", 0.03)):
        split = split_at([tail, 0.04], count=count)
        check(np.allclose(split.weights[:, column(split, letter, "r")], 1.0),
              "even: with {:d} bones, the last share of the length is all {:s}".format(count, letter))


def test_one_bone():
    split = split_at([-0.01, 0.0, 0.005, 0.03, 0.2], count=1)
    check(split.names == [earring.bone_name("a", "l"), earring.bone_name("a", "r")],
          "split: one bone per side uses a only")
    check(np.allclose(split.weights[:, 1], 1.0), "split: with one bone, everything is all a")


def test_stud():
    split = split_at([-0.003, 0.0, 0.00005])
    check(np.allclose(split.weights[:, column(split, "a", "r")], 1.0), "even: a stud at the lobe is all a")


def test_sides():
    points = np.array([LOBES[0] + (0.0, 0.01, 0.0), LOBES[1] + (0.0, 0.01, 0.0), (0.0, 0.01, 0.0005),
                       (0.0, 0.01, -0.0005), LOBES[0] + (0.0, 0.02, 0.0), LOBES[1] + (0.0, 0.04, 0.0)])
    split = earring.earring_weights(points, LOBES, 2)
    check(split.left.tolist() == [True, False, True, False, True, False],
          "split: each vertex goes to the side of its Z")
    b_l, a_r = column(split, "b", "l"), column(split, "a", "r")
    check(np.isclose(split.weights[0, b_l], 0.5) and np.isclose(split.weights[1, a_r], 1.0),
          "split: each earring is spread over its own length (2 cm left, 4 cm right)")
    check(np.allclose(split.lengths, (0.02, 0.04)), "split: the lengths per side")
    check(np.allclose(LOBES[0], LOBES[1] * (1.0, 1.0, -1.0)) and LOBES[0][2] > 0.0,
          "lobes: the left lobe is the right one with Z negated, on +Z")


# A hook, a ring, a bead, and a pendant with a gem set in it: (part, top, bottom), in metres below the lobe.
PIECES = ((0, -0.005, 0.003), (1, 0.002, 0.010), (2, 0.010, 0.016), (3, 0.016, 0.040), (4, 0.025, 0.030))


def pieced_earring():
    heights, parts = [], []
    for part, top, bottom in PIECES:
        for h in np.linspace(top, bottom, 5):
            heights.append(h)
            parts.append(part)
    return heights, parts


def test_joints():
    heights, parts = pieced_earring()
    parts = np.array(parts)
    split = split_at(heights, parts=parts, count=4)
    bones = split.weights.argmax(1)
    check(np.allclose(split.weights.max(1), 1.0), "joints: each piece moves whole, on one bone")
    first = [int(bones[parts == part][0]) for part, _top, _bottom in PIECES]
    check(all((bones[parts == part] == first[k]).all() for k, (part, _t, _b) in enumerate(PIECES)),
          "joints: never torn inside a piece")
    names = [split.names[index][-3] for index in first]
    check(names == ["a", "b", "c", "d", "d"], "joints: hook a, ring b, bead c, pendant d, its gem with it ({!r})".format(
        names))
    check(split.joints[1] == 3, "joints: three joints used")
    two = split_at(heights, parts=parts, count=2)
    names = [two.names[int(two.weights[parts == part].argmax(1)[0])][-3] for part, _t, _b in PIECES]
    check(names == ["a", "a", "a", "b", "b"],
          "joints: with 2 bones, the joint nearest half way (bead to pendant) ({!r})".format(names))
    off = split_at(heights, parts=parts, count=4, use_joints=False)
    check(off.joints[1] == 0 and not np.allclose(off.weights.max(1), 1.0), "joints: off, the length is shared evenly")
    whole = split_at(heights, parts=np.zeros(len(parts), dtype=int), count=4)
    check(whole.joints[1] == 0, "joints: an earring in one piece is shared evenly")


def test_pick_joints():
    check(earring.pick_joints([0.25, 1.0, 1.6], [1.0, 2.0, 3.0]) == [0, 1, 2], "pick: all when too few")
    check(earring.pick_joints([0.5, 0.9, 1.1, 1.9, 2.1, 3.2], [1.0, 2.0, 3.0]) == [1, 3, 5], "pick: closest to each")
    check(earring.pick_joints([0.2, 0.3, 0.4, 3.0], [1.0, 2.0, 3.0]) == [1, 2, 3],
          "pick: in order, leaving enough joints for the rest")


def test_loose_parts():
    tris = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    weld = np.array([0, 1, 2, 2, 3, 4, 5, 6, 7])
    parts = earring.loose_parts(9, tris, weld)
    check(parts[0] == parts[3] and parts[0] != parts[6], "parts: welded spots join, separate ones don't")


def test_one_bone_brush_weights():
    points = np.array([LOBES[0] + (0.0, 0.03, 0.0), LOBES[1] + (0.0, 0.0, 0.0)])
    names, weights = earring.one_bone(points, "c")
    check(names == [earring.bone_name("c", "l"), earring.bone_name("c", "r")] and
          weights.tolist() == [[1.0, 0.0], [0.0, 1.0]], "brush: a picked bone, of each vertex's side")


def test_far():
    near = split_at([0.01], sideways=0.02)
    far = split_at([0.01], sideways=0.04)
    hanging = split_at([0.5], sideways=0.0)
    check(not near.far[0] and far.far[0], "split: vertices more than 3 cm sideways from both lobes are far")
    check(not hanging.far[0], "split: hanging far down below a lobe isn't")


def test_race_table():
    items = earring.race_items()
    check([item[0] for item in items] == [MIQOTE, VIERA], "races: built from the lobe table, Miqo'te then Viera")
    check([item[1] for item in items] == ["Miqo'te Female", "Viera Female (face 1)"], "races: their names")
    check(all("..." not in item[1] + item[2] and "…" not in item[1] + item[2] for item in items),
          "races: no ellipsis in the names or tooltips")
    check(np.allclose(earring.lobes(VIERA)[1], (-0.013044, 0.020468, -0.057924)), "races: Viera's right lobe")


# -----------------------------------------------------------------------------
# Bones

def bone_matrix(armature, name):
    return np.array(armature.matrix_world @ armature.data.bones[name].matrix_local)


def check_bones(armature, rig, label):
    head = bone_matrix(armature, earring.HEAD)
    worst = 0.0
    for side, lobe in zip(earring.SIDES, rig.lobes):
        down = head[:3, :3] @ earring.DOWN
        for k, letter in enumerate(earring.LETTERS):
            matrix = bone_matrix(armature, earring.bone_name(letter, side))
            expected = (head @ np.append(lobe, 1.0))[:3] + down * earring.JOINT * k
            worst = max(worst, float(np.linalg.norm(matrix[:3, 3] - expected)))
            if letter == "a":
                check(np.dot(matrix[:3, 0], (0.0, 0.0, -1.0)) > 0.999,
                      "{:s}: a_{:s} points +X straight down".format(label, side))
                check(armature.data.bones[earring.bone_name(letter, side)].parent.name == earring.HEAD,
                      "{:s}: a_{:s} hangs from the head".format(label, side))
    check(worst < 1e-4, "{:s}: the bone heads are at the head + lobe, 1 cm apart ({:.4f} mm off)".format(
        label, worst * 1000.0))


def test_bones_added():
    clear_scene()
    for key in (MIQOTE, VIERA):
        armature = new_armature(names=("j_kubi",))
        rig = earring.EarringRig.for_race(key)
        added = hairing.ensure_bones(bpy.context, armature, rig, [bone.name for bone in rig.bones])
        check(set(added) == earring.ALL_BONES | {earring.HEAD},
              "bones: {:s}: the head and all eight earring bones are added".format(key))
        check_bones(armature, rig, "bones " + key)
        reference_head = rig.head.head
        check(np.linalg.norm(bone_matrix(armature, earring.HEAD)[:3, 3] - reference_head) < 1e-6,
              "bones {:s}: j_kao at the game's rest".format(key))
        check(abs(np.dot(bone_matrix(armature, earring.bone_name("a", "r"))[:3, 3]
                         - bone_matrix(armature, earring.bone_name("a", "l"))[:3, 3], (1.0, 0.0, 0.0))) > 0.1,
              "bones {:s}: the lobes are on either side of the head".format(key))
        check(bone_matrix(armature, earring.bone_name("a", "l"))[0, 3] > 0.0,
              "bones {:s}: _l is on the character's left (+X)".format(key))


def test_bones_moved_with_armature():
    clear_scene()
    offset = np.array((0.3, -0.2, 0.05))
    armature = new_armature(names=("j_kao",), offset=offset)
    rig = earring.EarringRig.for_race(MIQOTE)
    found = hairing.armature_offset(armature, rig)
    check(np.allclose(found, offset), "offset: found from the armature's head bone")
    rig = rig.moved(found)
    hairing.ensure_bones(bpy.context, armature, rig, [bone.name for bone in rig.bones])
    check_bones(armature, rig, "offset")


# -----------------------------------------------------------------------------
# The button

def test_button():
    clear_scene()
    settings = use_settings(earring_race=MIQOTE, earring_bones=2)
    armature = new_armature(names=("j_kao",))
    rig = earring.EarringRig.for_race(MIQOTE)
    obj, local = new_earrings(rig, armature)
    # An old weight on the game's earring point, and a locked group on the right earring's hook.
    old = obj.vertex_groups.new(name="n_ear_a_l")
    old.add(list(range(len(local))), 1.0, 'REPLACE')
    locked = obj.vertex_groups.new(name="j_kao")
    hook = [k for k in range(len(local)) if local[k][2] < 0.0 and local[k][1] - LOBES[1][1] < 0.0]
    locked.add(hook, 0.3, 'REPLACE')
    locked.lock_weight = True
    select_only([obj], obj)
    check(bpy.ops.magic_fit.earring_weights.poll(), "button: ready with an earring mesh selected")
    check(bpy.ops.magic_fit.earring_weights() == {'FINISHED'}, "button: runs")
    check(earring.ALL_BONES <= set(armature.data.bones.keys()), "button: all eight bones added, even with 2 used")
    weights = weights_of(obj)
    check("n_ear_a_l" not in weights, "button: the old weights are taken off")
    check(np.allclose(weights["j_kao"][hook], 0.3), "button: the locked group keeps its weights")
    used = earring.bone_names(2)
    total = sum(weights.get(name, 0.0) for name in used)
    check(np.allclose(total[hook], 0.7, atol=1e-3), "button: the earring bones share what the locked group leaves")
    others = [k for k in range(len(local)) if k not in hook]
    check(np.allclose(total[others], 1.0, atol=1e-3), "button: elsewhere the earring bones take it all")
    check(not any(name in weights for name in earring.ALL_BONES - set(used)),
          "button: the bones not used get no groups")
    tail = [k for k in range(len(local)) if local[k][1] - LOBES[0][1] > 0.029 and local[k][2] > 0.0]
    check(np.allclose(weights[earring.bone_name("b", "l")][tail], 1.0, atol=1e-3),
          "button: with 2 bones, the left tail is all b_l")
    check(straighten.deforming_armature(obj).object == armature, "button: the earring keeps its armature")

    # A second run with more bones replaces them, and the stroke paints the same weights.
    settings.earring_bones = 4
    bpy.ops.magic_fit.earring_weights()
    weights = weights_of(obj)
    tail = [k for k in range(len(local)) if local[k][1] - LOBES[0][1] > 0.035 and local[k][2] > 0.0]
    check(np.allclose(weights[earring.bone_name("d", "l")][tail], 1.0, atol=1e-3),
          "button: with 4 bones, the left tail is all d_l")
    check(operators.check_ready(bpy.context, settings) is None, "brush: ready after Earring Weights")
    before = weights_of(obj)
    stroke = earring.build_stroke(bpy.context, obj, settings)
    count = len(obj.data.vertices)
    stroke.apply(list(range(count)), [1.0] * count)
    after = weights_of(obj)
    check(all(np.allclose(before[name], after.get(name, 0.0), atol=1e-3) for name in before),
          "brush: painting fully gives what the button gave")

    # A bone picked by hand: the brush paints that one, on each vertex's side.
    settings.earring_brush_bone = 'C'
    stroke = earring.build_stroke(bpy.context, obj, settings)
    stroke.apply(list(range(count)), [1.0] * count)
    weights = weights_of(obj)
    left = [k for k in range(count) if local[k][2] > 0.0]
    right = [k for k in others if local[k][2] < 0.0]
    check(np.allclose(weights[earring.bone_name("c", "l")][left], 1.0, atol=1e-3) and
          np.allclose(weights[earring.bone_name("c", "r")][right], 1.0, atol=1e-3),
          "brush: with C picked, painting fully gives all of it to c of the vertex's side")
    check(np.allclose(weights["j_kao"][hook], 0.3), "brush: the locked group still keeps its weights")
    settings.earring_brush_bone = 'AUTO'


def strip(name, rig, side, top, bottom, armature):
    """A strip hanging below the lobe of `side` from `top` to `bottom` (metres below it)."""
    lobe = rig.lobes[side]
    local = [lobe + np.array((0.0, h, dz)) for h in np.linspace(top, bottom, 6) for dz in (-0.001, 0.001)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(head_space(np.array(local), rig).tolist(), [],
                     [(2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2) for k in range(5)])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    hairing.add_armature_modifier(obj, armature)
    return obj


def test_button_joints():
    clear_scene()
    use_settings(earring_race=MIQOTE, earring_bones=4)
    armature = new_armature(names=("j_kao",))
    rig = earring.EarringRig.for_race(MIQOTE)
    hook = strip("Hook", rig, 1, -0.005, 0.003, armature)
    pendant = strip("Pendant", rig, 1, 0.004, 0.04, armature)
    select_only([hook, pendant], hook)
    check(bpy.ops.magic_fit.earring_weights() == {'FINISHED'}, "joints: runs on a hook and a pendant")
    check(np.allclose(weights_of(hook)[earring.bone_name("a", "r")], 1.0), "joints: the hook is all a")
    weights = weights_of(pendant)
    check(np.allclose(weights.get(earring.bone_name("b", "r"), 0.0), 1.0),
          "joints: the pendant, a separate mesh below it, moves whole with b")
    bpy.context.scene.magic_fit.earring_use_joints = False
    bpy.ops.magic_fit.earring_weights()
    weights = weights_of(pendant)
    check(len([name for name in weights if name.startswith("j_ex_earring")]) == 4,
          "joints: off, the pendant bends down all four bones")


def test_button_without_armature():
    clear_scene()
    use_settings(earring_race=VIERA, earring_bones=4)
    rig = earring.EarringRig.for_race(VIERA)
    obj, _local = new_earrings(rig)
    select_only([obj], obj)
    check(operators.check_ready(bpy.context, bpy.context.scene.magic_fit) is not None and
          "Earring Weights" in operators.check_ready(bpy.context, bpy.context.scene.magic_fit),
          "brush: asks for Earring Weights first")
    check(bpy.ops.magic_fit.earring_weights() == {'FINISHED'}, "new armature: runs")
    armature = straighten.deforming_armature(obj).object
    check(armature.name.startswith("Earring Skeleton"), "new armature: made one")
    check_bones(armature, rig, "new armature")


def test_far_warning():
    clear_scene()
    use_settings(earring_race=MIQOTE)
    armature = new_armature(names=("j_kao",))
    rig = earring.EarringRig.for_race(MIQOTE)
    # A necklace, well below and in front of the lobes.
    mesh = bpy.data.meshes.new("Necklace")
    mesh.from_pydata(head_space(np.array([(0.08, 0.15, -0.01), (0.08, 0.15, 0.01), (0.09, 0.15, 0.0)]),
                                rig).tolist(), [], [(0, 1, 2)])
    obj = bpy.data.objects.new("Necklace", mesh)
    bpy.context.scene.collection.objects.link(obj)
    hairing.add_armature_modifier(obj, armature)
    select_only([obj], obj)
    local = rig.local(np.array([v.co for v in mesh.vertices]))
    check(earring.earring_weights(local, rig.lobes).far.all(), "far: a mesh far from both lobes is found")
    check(bpy.ops.magic_fit.earring_weights() == {'FINISHED'}, "far: still runs (and warns)")


def test_refusals():
    clear_scene()
    settings = use_settings()
    rig = earring.EarringRig.for_race(MIQOTE)
    obj, _local = new_earrings(rig)
    select_only([], None)
    check(operators.check_earring_weights_ready(bpy.context) == "Select the earring meshes",
          "refused: nothing selected")
    select_only([obj], obj)
    settings.group_mode = 'ACTIVE'
    check("All Groups" in operators.check_earring_weights_ready(bpy.context), "refused: Current Group")
    settings.group_mode = 'ALL'
    check(operators.check_earring_weights_ready(bpy.context) is None, "ready again")


def test_registered():
    settings = bpy.context.scene.magic_fit
    check(hasattr(bpy.ops.magic_fit, "earring_weights"), "the Earring Weights operator is registered")
    tabs = [item.identifier for item in settings.bl_rna.properties["weights_tab"].enum_items]
    check(tabs[-1] == 'EARRING' and tabs.index('EARRING') == tabs.index('FACE') + 1, "Earrings is the sub-tab after Face")
    settings.weights_tab = 'EARRING'
    check(settings.mode == 'EARRING', "the Earrings sub-tab is the Earring mode")
    races = [item.identifier for item in settings.bl_rna.properties["earring_race"].enum_items]
    check(races == [MIQOTE, VIERA], "the race list is the lobe table")
    prop = settings.bl_rna.properties["earring_bones"]
    check((prop.default, prop.hard_min, prop.hard_max) == (4, 1, 4), "Bones per Side: 1 to 4, 4 by default")
    settings.weights_tab = 'GENERAL'


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
