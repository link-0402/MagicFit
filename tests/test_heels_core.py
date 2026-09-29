"""Headless tests for Heel weights (the Weight Brushes' Heels mode and the Heel Weights button).

Run from the repository root:
    blender -b --factory-startup --python tests/test_heels_core.py
"""

import math
import os
import sys
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import heels  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import heeling, operators, straighten  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def new_stroke(obj, **kwargs):
    rig = heeling.HeelRig.from_object(straighten.deforming_armature(obj).object)
    rest, = straighten.rest_geometry(bpy.context, (obj,), ('INDEXED',))
    return heeling.HeelsStroke(obj, rest.co, rig, **kwargs)


def paint_everything(stroke, obj, factor=1.0):
    count = len(obj.data.vertices)
    stroke.apply(list(range(count)), [factor] * count)
    obj.data.update()


def same_weights(a, b, tolerance):
    names = set(a) | set(b)
    count = len(next(iter(a.values())))
    zero = np.zeros(count)
    return max(float(np.abs(a.get(name, zero) - b.get(name, zero)).max()) for name in names) <= tolerance


def measure(rig, obj):
    return heeling.HeelRig.from_object(rig).measure(heels.rest_co(obj))


def bone_total(weights):
    return sum(values for name, values in weights.items() if name.startswith(("j_", "iv_")))


def use_settings(**values):
    settings = bpy.context.scene.magic_fit
    for prop in ("heels_ankle_height", "heels_ankle_blend", "heels_toe_bend", "heels_apply_to_selected", "group_mode"):
        settings.property_unset(prop)
    settings.target = None
    settings.mode = 'HEELS'
    for name, value in values.items():
        setattr(settings, name, value)
    return settings


# -----------------------------------------------------------------------------
# Rig and shares

def test_rig_detection():
    rig, _shoes = heels.build()
    found = heeling.HeelRig.from_object(rig)
    check([leg.side for leg in found.sides] == ["l", "r"], "rig: both legs")
    left = found.sides[0]
    check(np.allclose(left.ankle_point, heels.ANKLE), "rig: the ankle is the head of j_asi_d")
    check(left.up[2] > 0.99, "rig: up along the shin")
    check(np.allclose(left.ball, heels.BALL) and left.forward[1] < -0.99, "rig: the ball is the head of j_asi_e, ahead")
    expected = {name + "_" + side for side in "lr" for name in ("j_asi_d", "j_asi_e") + heels.TOE_BONES}
    check(found.foot_names == expected, "rig: the foot is the ankle bone and every bone below it")
    check(heeling.has_foot_bones(rig), "has_foot_bones: the rig has them")

    bare = bpy.data.objects.new("Bare", bpy.data.armatures.new("Bare"))
    bpy.context.scene.collection.objects.link(bare)
    check(heeling.HeelRig.from_object(bare) is None and not heeling.has_foot_bones(bare),
          "rig: an armature without legs has no feet")
    toeless = heeling.HeelRig.from_object(heels.build_rig("Toeless", toes=False))
    check(toeless is not None and toeless.sides[0].toe is None, "rig: legs without toe bones still work")


def test_fractions():
    check(np.allclose(heeling.foot_fraction([-1.0, -0.04, 0.02, 0.08, 1.0], 0.02, 0.12), [1, 1, 0.5, 0, 0]),
          "foot share: all below the blend, half at Ankle Height, none above")
    check(np.allclose(heeling.toe_fraction([-1.0, -0.04, -0.02, 0.0, 1.0, np.nan], 1.0), [0, 0, 0.5, 1, 1, 0]),
          "toe share: none 4 cm behind the ball, all from it on; none without a toe bone")
    check(np.allclose(heeling.toe_fraction([1.0], 0.4), 0.4), "toe share: Toe Bend ahead of the ball")


# -----------------------------------------------------------------------------
# Painting

def test_rigid_foot():
    rig, shoes = heels.build()
    start = heels.by_name(shoes)
    stroke = new_stroke(shoes)
    paint_everything(stroke, shoes)
    after = heels.by_name(shoes)
    side, height, _ahead = measure(rig, shoes)
    ankle = np.where(side == 0, after["j_asi_d_l"], after["j_asi_d_r"])
    total = bone_total(after)
    foot = height < 0.02 - 0.06
    check(foot.any() and float(np.abs(ankle[foot] - total[foot]).max()) < 1e-6,
          "rigid: everything below the ankle follows the ankle bone only")
    toes = [name for name in after if name.startswith(("j_asi_e", "iv_asi"))]
    check(not any(after[name].any() for name in toes), "rigid: no toe weights are left")
    leg = height > 0.02 + 0.06
    check(leg.any() and all(np.array_equal(after[name][leg], start[name][leg]) for name in start),
          "above the ankle: untouched")
    band = ~foot & ~leg
    share = heeling.foot_fraction(height, 0.02, 0.12)
    calf = np.where(side == 0, after["j_asi_c_l"], after["j_asi_c_r"])
    check(band.any() and float(np.abs(ankle[band] - share[band]).max()) < 1e-4
          and float(np.abs(calf[band] - (1.0 - share[band])).max()) < 1e-4,
          "ankle: the calf hands over to the ankle bone by height")
    check(float(np.abs(total - bone_total(start)).max()) < 1e-4, "totals kept")
    check(stroke.painted_foot and stroke.painted_leg, "the stroke saw both")


def rigid_error(rig, shoes, bone_name, rows):
    """How far ``rows`` of the posed shoes are from moving rigidly with ``bone_name`` (m)."""
    shown = heels.displayed(shoes)
    pose = rig.pose.bones[bone_name]
    deform = rig.matrix_world @ pose.matrix @ pose.bone.matrix_local.inverted() @ rig.matrix_world.inverted()
    rest = heels.rest_co(shoes)[rows]
    expected = rest @ np.array(deform.to_3x3()).T + np.array(deform.translation)
    return float(np.linalg.norm(shown[rows] - expected, axis=1).max())


def test_rigid_in_poses():
    rig, shoes = heels.build()
    side, height, _ahead = measure(rig, shoes)
    foot = (side == 0) & (height < -0.04)

    def pose(on):
        for name, angle in (("j_asi_d_l", 0.4), ("j_asi_e_l", -0.7), ("iv_asi_oya_a_l", -0.5)):
            bone = rig.pose.bones[name]
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = (angle if on else 0.0, 0.0, 0.0)
        bpy.context.view_layer.update()

    pose(True)
    before = rigid_error(rig, shoes, "j_asi_d_l", foot)
    pose(False)
    paint_everything(new_stroke(shoes), shoes)
    pose(True)
    after = rigid_error(rig, shoes, "j_asi_d_l", foot)
    check(before > 0.005, "posed: transferred weights bend the foot ({:.1f} mm)".format(before * 1000))
    check(after < 1e-5, "posed: with heel weights the foot moves as one piece ({:.4f} mm)".format(after * 1000))
    pose(False)


def test_toe_bend():
    rig, shoes = heels.build()
    paint_everything(new_stroke(shoes, toe_bend=1.0), shoes)
    after = heels.by_name(shoes)
    side, height, ahead = measure(rig, shoes)
    left = (side == 0) & (height < -0.04)
    toes = left & (ahead > 0.0)
    heel = left & (ahead < -0.045)
    check(toes.any() and float(after["j_asi_e_l"][toes].min()) > 0.99, "toe bend: the toes follow the toe bone")
    check(heel.any() and not after["j_asi_e_l"][heel].any(), "toe bend: not behind the ball")
    check(not any(after[name].any() for name in after if name.startswith("iv_asi")), "toe bend: not the toe bones")


def test_idempotent_and_restore():
    _rig, shoes = heels.build()
    start = heels.by_name(shoes)
    names = [group.name for group in shoes.vertex_groups]
    stroke = new_stroke(shoes)
    paint_everything(stroke, shoes)
    first = heels.by_name(shoes)
    again = new_stroke(shoes)
    paint_everything(again, shoes)
    check(not again.modified and same_weights(first, heels.by_name(shoes), 1e-6), "again: a second pass changes nothing")
    stroke.restore()
    shoes.data.update()
    check(same_weights(start, heels.by_name(shoes), 0.0) and [g.name for g in shoes.vertex_groups] == names,
          "restore: exactly as before")

    _rig, shoes = heels.build()
    for name in ("j_asi_d_l", "j_asi_d_r"):
        shoes.vertex_groups.remove(shoes.vertex_groups[name])
    stroke = new_stroke(shoes)
    paint_everything(stroke, shoes)
    check(sorted(stroke.created_group_names()) == ["j_asi_d_l", "j_asi_d_r"], "groups: missing ankle groups are created")


def test_locked_and_current_group():
    rig, shoes = heels.build()
    start = heels.by_name(shoes)
    shoes.vertex_groups["j_asi_d_l"].lock_weight = True
    paint_everything(new_stroke(shoes), shoes)
    after = heels.by_name(shoes)
    side, height, _ahead = measure(rig, shoes)
    left_foot = (side == 0) & (height < -0.04)
    check(np.array_equal(after["j_asi_d_l"], start["j_asi_d_l"]), "locked: a locked ankle group is left alone")
    check(all(np.array_equal(after[n][left_foot], start[n][left_foot]) for n in start),
          "locked: a foot that can't be made rigid is left as it is")
    right_foot = (side == 1) & (height < -0.04)
    check(not after["j_asi_e_r"][right_foot].any(), "locked: the other foot is still done")

    _rig, shoes = heels.build()
    start = heels.by_name(shoes)
    shoes.vertex_groups.active_index = shoes.vertex_groups["j_asi_d_l"].index
    paint_everything(new_stroke(shoes, group_mode='ACTIVE'), shoes)
    after = heels.by_name(shoes)
    others = [name for name in start if name != "j_asi_d_l"]
    changed = np.abs(after["j_asi_d_l"] - start["j_asi_d_l"]) > 1e-3
    # Proportions only exist where the vertex has other bones before and after.
    changed &= np.stack([start[n] for n in others], axis=1).sum(axis=1) > 1e-3
    changed &= np.stack([after[n] for n in others], axis=1).sum(axis=1) > 1e-3
    a = np.stack([start[n] for n in others], axis=1)[changed]
    b = np.stack([after[n] for n in others], axis=1)[changed]
    check(changed.any(), "current group: the active group changes")
    check(float(np.abs(a / a.sum(1, keepdims=True) - b / b.sum(1, keepdims=True)).max()) < 1e-3,
          "current group: the other bones keep their proportions")


def test_symmetry_and_pose():
    rig, shoes = heels.build()
    paint_everything(new_stroke(shoes), shoes)
    rest = heels.by_name(shoes)
    half = len(shoes.data.vertices) // 2
    worst = max(float(np.abs(rest[name][:half] - rest[name[:-1] + "r"][half:]).max())
                for name in rest if name.endswith("_l"))
    check(worst < 1e-6, "symmetry: both shoes get mirrored weights")

    rig, shoes = heels.build()
    for name, angle in (("j_asi_c_l", 0.5), ("j_asi_d_r", -0.4), ("j_asi_e_l", 0.6)):
        bone = rig.pose.bones[name]
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = (angle, 0.0, 0.0)
        bone.scale = (1.2, 1.2, 1.2)
    bpy.context.view_layer.update()
    paint_everything(new_stroke(shoes), shoes)
    check(same_weights(rest, heels.by_name(shoes), 1e-6), "pose: the same weights whatever the pose")


def test_strength():
    _rig, shoes = heels.build()
    start = heels.by_name(shoes)
    stroke = new_stroke(shoes)
    paint_everything(stroke, shoes)
    full = heels.by_name(shoes)
    stroke.restore()
    shoes.data.update()
    paint_everything(new_stroke(shoes), shoes, factor=0.5)
    half = heels.by_name(shoes)
    worst = max(float(np.abs(half[n] - 0.5 * (start[n] + full[n])).max()) for n in full)
    check(worst < 1e-3, "strength: half strength goes half the way ({:.1e})".format(worst))


# -----------------------------------------------------------------------------
# The button

def test_operator():
    _rig, shoes = heels.build()
    paint_everything(new_stroke(shoes, **heeling.stroke_params(use_settings())), shoes)
    engine = heels.by_name(shoes)

    _rig, shoes = heels.build()
    use_settings()
    bpy.ops.object.mode_set(mode='OBJECT')
    check(bpy.ops.magic_fit.heel_weights.poll(), "button: ready in Object Mode, no body needed")
    check(bpy.ops.magic_fit.heel_weights() == {'FINISHED'}, "button: finishes")
    check(same_weights(engine, heels.by_name(shoes), 1e-6), "button: the same weights as painting everything")
    check(bpy.ops.magic_fit.heel_weights() == {'CANCELLED'}, "button: a second time changes nothing")

    _rig, shoes = heels.build()
    use_settings(heels_apply_to_selected=True)
    bpy.ops.object.mode_set(mode='OBJECT')
    second = shoes.copy()
    second.data = shoes.data.copy()
    bpy.context.scene.collection.objects.link(second)
    second.select_set(True)
    bpy.ops.magic_fit.heel_weights()
    check(all(not heels.by_name(o)["j_asi_e_l"].any() for o in (shoes, second)),
          "button: Apply to All Selected does every selected mesh")

    bare = bpy.data.objects.new("Bare", bpy.data.armatures.new("Bare"))
    bpy.context.scene.collection.objects.link(bare)
    shoes.modifiers["Armature"].object = bare
    use_settings()
    bpy.context.view_layer.objects.active = shoes
    check(not bpy.ops.magic_fit.heel_weights.poll(), "button: refused without foot bones")
    check("no foot bones" in operators.check_heels_ready(shoes, bpy.context.scene.magic_fit),
          "button: says the armature has no foot bones")


def test_brush_readiness():
    _rig, shoes = heels.build()
    settings = use_settings()
    check(operators.check_ready(bpy.context, settings) is None, "brush: ready in Heels mode without a body")
    shoes.modifiers["Armature"].show_viewport = False
    check(operators.check_ready(bpy.context, settings) == "Heels needs an Armature modifier",
          "brush: needs an Armature modifier")


def test_registered():
    check(hasattr(bpy.ops.magic_fit, "heel_weights"), "the Heel Weights operator is registered")
    settings = bpy.context.scene.magic_fit
    items = [item.identifier for item in settings.bl_rna.properties["mode"].enum_items]
    check(items == ['COPY', 'STRAIGHTEN', 'SKIRT', 'HEELS', 'HAIR', 'FACE', 'SMOOTH'],
          "Heels keeps its place, Hair, Face and Smooth come after it (files store their positions)")


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
