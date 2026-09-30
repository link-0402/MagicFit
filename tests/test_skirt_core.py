"""Headless tests for Skirt weights (the Weight Brushes' Skirt mode and the Skirt Weights button).

Run from the repository root:
    blender -b --factory-startup --python tests/test_skirt_core.py
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

import skirt  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import fitting, operators, skirting, straighten  # noqa: E402

FAILURES = []
HIP_Z = skirt.HIP[2]


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


# The tests below were worked out for this ramp and skin hold, not the defaults.
TEST_SHARES = dict(blend_start=0.10, blend_length=0.30, skin_weight=0.5, skin_distance=0.05)


def new_stroke(obj, body=None, **kwargs):
    kwargs = {**TEST_SHARES, **kwargs}
    rig = skirting.SkirtRig.from_object(straighten.deforming_armature(obj).object)
    objects = (obj, body) if body is not None else (obj,)
    geometries = straighten.rest_geometry(bpy.context, objects, ('INDEXED', 'WEIGHTED')[:len(objects)])
    surface = weights = None
    if body is not None:
        key = ("rest", body.as_pointer())
        surface = fitting.BodySurface.from_geometry(key, geometries[1])
        weights = skirting.BodyWeights(body, geometries[1], key)
    return skirting.SkirtStroke(obj, geometries[0].co, rig, body=surface, body_weights=weights, **kwargs)


def paint_everything(stroke, obj, factor=1.0):
    count = len(obj.data.vertices)
    stroke.apply(list(range(count)), [factor] * count)
    obj.data.update()


def by_name(obj):
    """{group name: (vertices,) weights}."""
    table, names = skirt.weights(obj)
    return {name: table[:, k] for k, name in enumerate(names)}


def same_weights(a, b, tolerance):
    names = set(a) | set(b)
    count = len(next(iter(a.values())))
    zero = np.zeros(count)
    return max(float(np.abs(a.get(name, zero) - b.get(name, zero)).max()) for name in names) <= tolerance


def bone_total(weights):
    return sum(values for name, values in weights.items() if name.startswith("j_") or name.startswith("iv_"))


def skirt_total(weights):
    return sum(values for name, values in weights.items() if name.startswith("j_sk_"))


def heights(obj):
    return skirt.rest_co(obj)[:, 2] - HIP_Z


def use_settings(body, **values):
    settings = bpy.context.scene.magic_fit
    for prop in ("skirt_blend_start", "skirt_blend_length", "skirt_skin_weight", "skirt_skin_distance",
                 "skirt_body_weight", "skirt_spread", "skirt_joint_blend", "skirt_max_groups",
                 "skirt_apply_to_selected", "group_mode"):
        settings.property_unset(prop)
    settings.target = body
    settings.mode = 'SKIRT'
    for name, value in values.items():
        setattr(settings, name, value)
    return settings


# -----------------------------------------------------------------------------
# Rig and shares

def test_rig_detection():
    rig, _body, _skirt = skirt.build()
    found = skirting.SkirtRig.from_object(rig)
    keys = [chain.key for chain in found.chains]
    check(keys == [("s", "r"), ("f", "r"), ("f", "l"), ("s", "l"), ("b", "l"), ("b", "r")],
          "rig: six chains in order around the hips")
    angles = [math.degrees(math.atan2(math.sin(c.angles[0]), math.cos(c.angles[0]))) for c in found.chains]
    expected = [-175.9, -124.9, -55.1, -4.1, 55.1, 124.9]
    check(all(abs(a - e) < 1.0 for a, e in zip(angles, expected)), "rig: chain angles as in the game's skeleton")
    check(all(len(c.bones) == 3 and c.bones == sorted(c.bones) for c in found.chains), "rig: segments a, b, c top down")
    check(np.allclose(found.hip, skirt.HIP), "rig: measured around j_kosi")
    check(sorted(found.bone_names) == sorted(skirt.skirt_bone_names()), "rig: all 18 skirt bones")
    front = found.chains[2]
    check(abs(math.degrees(front.angles[2]) + 66.2) < 0.5, "rig: the front chain splays outward as it hangs")
    arms = {name + side for name in ("j_sako_", "j_ude_a_", "j_ude_b_", "j_te_", "j_hito_a_") for side in "lr"}
    check(found.arms_and_head == arms | {"j_kubi", "j_kao"}, "rig: the arms and the head, fingers included")
    check(skirting.has_skirt_bones(rig), "has_skirt_bones: the rig has them")

    plain = skirt.build_rig("Plain", chains={})
    check(skirting.SkirtRig.from_object(plain) is None and not skirting.has_skirt_bones(plain),
          "rig: an armature without skirt bones has no skirt rig")
    for bone in rig.data.bones:
        if bone.name.startswith("j_sk_"):
            bone.use_deform = False
    check(skirting.SkirtRig.from_object(rig) is None, "rig: skirt bones that don't deform don't count")
    for bone in rig.data.bones:
        bone.use_deform = True

    rig.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    turned = skirting.SkirtRig.from_object(rig)
    check([chain.key for chain in turned.chains] == keys, "rig turned on its side: same chain order")
    check(max(abs(a.heights - b.heights).max() for a, b in zip(turned.chains, found.chains)) < 0.003,
          "rig turned on its side: heights measured along the chains")


def test_kernels():
    theta = np.radians([-175.9, -124.9, -55.1, -4.1, 55.1, 124.9])
    own = skirting.chain_weights(theta, theta, 1.5)
    check(np.allclose(np.diag(own), 0.6), "chains: 0.6 at a chain's own angle (spread 1.5)")
    check(abs(own[2, 1] - 0.2) < 1e-9 and abs(own[2, 3] - 0.2) < 1e-9, "chains: 0.2 to each neighbour")
    middle = skirting.chain_weights(theta, [0.5 * (theta[2] + theta[3])], 1.5)[0]
    check(abs(middle[2] - 0.5) < 1e-9 and abs(middle[3] - 0.5) < 1e-9, "chains: 0.5 / 0.5 midway")
    quarter = skirting.chain_weights(theta, [theta[2] + 0.25 * (theta[3] - theta[2])], 1.0)[0]
    check(abs(quarter[2] - 0.75) < 1e-9 and abs(quarter[3] - 0.25) < 1e-9, "chains: spread 1 blends linearly")
    circle = np.radians(np.arange(720) / 2.0 - 180.0)
    around = skirting.chain_weights(theta, circle, 1.5)
    check(np.allclose(around.sum(axis=1), 1.0), "chains: shares add up to 1 all around")
    check(int((around > 1e-12).sum(axis=1).max()) <= 3, "chains: at most 3 chains share a spot (spread 1.5)")
    check(float(np.abs(np.diff(np.vstack((around, around[:1])), axis=0)).max()) < 0.05,
          "chains: shares change smoothly around the hips")
    check(np.allclose(skirting.chain_weights([0.3], circle, 1.5), 1.0), "chains: a single chain takes everything")

    joints = [0.0, -0.391, -0.505]
    h = np.array([0.2, -0.391, -0.505, -0.7, -0.44])
    segments = skirting.segment_weights(joints, h, 0.10)
    check(np.allclose(segments[0], [1, 0, 0]) and np.allclose(segments[3], [0, 0, 1]),
          "segments: a above, c below")
    check(np.allclose(segments[1], [0.5, 0.5, 0.0]) and np.allclose(segments[2], [0.0, 0.5, 0.5]),
          "segments: half and half at a joint")
    check(np.allclose(segments.sum(axis=1), 1.0) and (segments >= 0).all(), "segments: shares add up to 1")
    check(np.allclose(skirting.segment_weights(joints, h, 0.0)[4], [0, 1, 0]), "segments: no blend is a hard cut")

    fraction = skirting.height_fraction(np.array([0.2, 0.10, -0.05, -0.20, -0.5]), 0.10, 0.30)
    check(np.allclose(fraction, [0, 0, 0.5, 1, 1]), "height: none from Blend Start up, all Blend Length lower")
    kept = skirting.body_fraction(
        np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.5]), np.array([0.3, 0.0, -0.01, 0.2, np.nan, 0.2]),
        skin_weight=0.5, skin_distance=0.05, body_weight=0.2,
    )
    check(np.allclose(kept, [1.0, 0.5, 0.5, 0.2, 0.2, 0.5]),
          "body share: all above the blend, Skin Weight on the body, Body Weight elsewhere")
    check(np.allclose(skirting.body_fraction([1.0], None, skin_weight=0.5, skin_distance=0.05, body_weight=0.0), 0.0),
          "body share: no body, no skin hold")


# -----------------------------------------------------------------------------
# Painting

def test_height_ramp():
    _rig, body, obj = skirt.build()
    start = by_name(obj)
    stroke = new_stroke(obj, body, skin_weight=0.0)
    paint_everything(stroke, obj)
    after = by_name(obj)
    h = heights(obj)
    above = h >= 0.10
    check(all(np.array_equal(after[name][above], start[name][above]) for name in start)
          and not skirt_total(after)[above].any(), "ramp: nothing changes above Blend Start")
    share = skirt_total(after) / bone_total(after)
    expected = skirting.height_fraction(h, 0.10, 0.30)
    check(float(np.abs(share - expected).max()) < 1e-4, "ramp: skirt share follows the height (Skin Weight 0)")
    near = np.abs(h + 0.05) < 0.01
    check(near.any() and float(np.abs(share[near] - 0.5).max()) < 0.05, "ramp: half skirt 5 cm below the hips")
    check(float(share[h <= -0.20].min()) >= 0.99, "ramp: all skirt 20 cm below the hips")
    rings = share.reshape(skirt.ROWS, skirt.COLUMNS + 1)
    check(bool((np.diff(rings, axis=0) >= -1e-9).all()), "ramp: the skirt share only grows going down")
    check(stroke.painted_above and stroke.painted_skirt, "ramp: the stroke saw both")


def test_totals_and_helpers():
    _rig, body, obj = skirt.build()
    mask = obj.vertex_groups.new(name="mask")
    mask.add(list(range(len(obj.data.vertices))), 0.7, 'REPLACE')
    obj.vertex_groups.active_index = 0
    start = by_name(obj)
    stroke = new_stroke(obj, body)
    paint_everything(stroke, obj)
    after = by_name(obj)
    check(np.array_equal(after["mask"], start["mask"]), "helpers: a group that isn't a bone is left alone")
    check(float(np.abs(bone_total(after) - bone_total(start)).max()) < 1e-4, "totals: every vertex keeps its total")
    kept = stroke.body_fractions(range(len(obj.data.vertices)))
    check(float(np.abs(skirt_total(after) - (1.0 - kept) * bone_total(start)).max()) < 1e-4,
          "totals: the skirt bones get what the body share leaves")
    check(obj.vertex_groups.active_index == 0, "groups: creating them keeps the active group")


def test_chain_symmetry():
    rig, body, obj = skirt.build()
    stroke = new_stroke(obj, body, skin_weight=0.0)
    paint_everything(stroke, obj)
    after = by_name(obj)
    worst = 0.0
    for name in skirt.skirt_bone_names():
        mirror = name[:-1] + ("r" if name.endswith("l") else "l")
        a = after[name].reshape(skirt.ROWS, skirt.COLUMNS + 1)
        b = after[mirror].reshape(skirt.ROWS, skirt.COLUMNS + 1)
        worst = max(worst, float(np.abs(a[:, 1:skirt.COLUMNS] - b[:, skirt.COLUMNS - 1:0:-1]).max()))
    check(worst < 1e-6, "symmetry: mirrored spots get mirrored skirt weights ({:.1e})".format(worst))
    chains = {}
    for name in skirt.skirt_bone_names():
        chains.setdefault(name[5] + name[-1], []).append(after[name])
    used = sum((sum(values) > 1e-9).astype(int) for values in chains.values())
    check(int(used.max()) <= 3, "chains: at most 3 per vertex")

    found = skirting.SkirtRig.from_object(rig)
    h = np.linspace(-0.6, -0.1, 11)
    front = [chain.key for chain in found.chains].index(("f", "l"))
    shares = found.distribution(h, found.chain_angles(h)[:, front], spread=1.5, joint_blend=0.10)
    columns = [k for k, name in enumerate(found.bone_names) if name.startswith("j_sk_f_") and name.endswith("_l")]
    check(np.allclose(shares[:, columns].sum(axis=1), 0.6), "chains: 0.6 at the chain's own angle, all the way down")


def test_segment_blend():
    _rig, body, obj = skirt.build()
    paint_everything(new_stroke(obj, body, skin_weight=0.0), obj)
    after = by_name(obj)
    z = skirt.rest_co(obj)[:, 2]
    column = [skirt.index(ring, 5) for ring in range(skirt.ROWS)]  # about the front left chain's angle
    a, b, c = (after["j_sk_f_{:s}_l".format(s)][column] for s in "abc")
    zc = z[column]
    high = zc > 0.6456 + 0.06
    low = zc < 0.532 - 0.055
    check(high.any() and not b[high].any() and not c[high].any(), "segments: only a above the b joint")
    check(low.any() and not a[low].any() and not b[low].any(), "segments: only c below the c joint")
    joint = int(np.argmin(np.abs(zc - 0.6456)))
    check(abs(a[joint] - b[joint]) / (a[joint] + b[joint]) < 0.35, "segments: a and b share the b joint")


def test_skin_hold():
    rig, body, obj = skirt.build()
    rest = skirt.displayed(obj)
    skirt.scale_thighs(rig, 1.3)
    moved_full = skirt.displayed(obj) - rest
    skirt.scale_thighs(rig, 1.0)
    h = heights(obj)
    co = skirt.rest_co(obj)
    tight = (h < -0.20) & (co[:, 2] > skirt.SKIRT_TIGHT) & (np.abs(co[:, 0]) > 0.06)
    check(tight.any(), "skin hold: there are tight vertices below the blend")

    stroke = new_stroke(obj, body)
    paint_everything(stroke, obj)
    after = by_name(obj)
    share = skirt_total(after) / bone_total(after)
    check(float(np.abs(1.0 - share[tight] - 0.486).max()) < 0.015,
          "skin hold: cloth 5 mm off the body keeps half its body weight")
    hem = h < skirt.SKIRT_HEM - HIP_Z + 0.05
    check(float(share[hem].min()) > 0.999, "skin hold: none 13 cm off the body")
    skirt.scale_thighs(rig, 1.3)
    moved = skirt.displayed(obj) - rest
    skirt.scale_thighs(rig, 1.0)
    ratio = np.linalg.norm(moved[tight], axis=1).sum() / np.linalg.norm(moved_full[tight], axis=1).sum()
    check(ratio > 0.4, "skin hold: scaled thighs still push tight cloth out ({:.2f} of the way)".format(ratio))

    rig, body, obj = skirt.build()
    paint_everything(new_stroke(obj, body, skin_weight=0.0), obj)
    skirt.scale_thighs(rig, 1.3)
    moved = skirt.displayed(obj) - rest
    ratio = np.linalg.norm(moved[tight], axis=1).sum() / np.linalg.norm(moved_full[tight], axis=1).sum()
    check(ratio < 0.05, "skin hold off: the cloth hangs from the skirt bones only ({:.2f})".format(ratio))

    _rig, body, obj = skirt.build()
    paint_everything(new_stroke(obj, body, body_weight=0.2), obj)
    after = by_name(obj)
    share = skirt_total(after) / bone_total(after)
    check(float(np.abs(share[hem] - 0.8).max()) < 1e-4, "body weight: 0.2 kept far from the body")


def test_idempotent():
    _rig, body, obj = skirt.build()
    paint_everything(new_stroke(obj, body), obj)
    first = by_name(obj)
    again = new_stroke(obj, body)
    paint_everything(again, obj)
    check(not again.modified and same_weights(first, by_name(obj), 1e-6), "again: a second pass changes nothing")

    _rig, body, obj = skirt.build()
    paint_everything(new_stroke(obj, body, skin_weight=0.0), obj)
    h = heights(obj)
    co = skirt.rest_co(obj)
    tight = (h < -0.20) & (co[:, 2] > skirt.SKIRT_TIGHT - 0.005)
    check(not bone_total({k: v for k, v in by_name(obj).items() if not k.startswith("j_sk_")})[tight].any(),
          "again: without Skin Weight the tight part hangs from the skirt bones only")
    paint_everything(new_stroke(obj, body, skin_weight=0.5), obj)
    after = by_name(obj)
    thighs = after["j_asi_a_l"] + after["j_asi_a_r"]
    check(float(thighs[tight].min()) > 0.4, "again: with Skin Weight, thigh weight comes back from the body")


def test_locked_groups():
    _rig, body, obj = skirt.build()
    start = by_name(obj)
    obj.vertex_groups["j_asi_a_l"].lock_weight = True
    paint_everything(new_stroke(obj, body), obj)
    after = by_name(obj)
    check(np.array_equal(after["j_asi_a_l"], start["j_asi_a_l"]), "locked: a locked body group is left alone")
    check(float(np.abs(bone_total(after) - bone_total(start)).max()) < 1e-4, "locked: totals kept")

    _rig, body, obj = skirt.build()
    obj.vertex_groups.new(name="j_sk_f_a_l").lock_weight = True
    paint_everything(new_stroke(obj, body, skin_weight=0.0), obj)
    after = by_name(obj)
    check(not after["j_sk_f_a_l"].any(), "locked: a locked skirt group gets nothing")
    expected = skirting.height_fraction(heights(obj), 0.10, 0.30)
    check(float(np.abs(skirt_total(after) / bone_total(after) - expected).max()) < 1e-4,
          "locked: the other skirt bones take its share")


def test_current_group():
    _rig, body, obj = skirt.build()
    group = obj.vertex_groups.new(name="j_sk_s_b_l")
    obj.vertex_groups.active_index = group.index
    start = by_name(obj)
    stroke = new_stroke(obj, body, group_mode='ACTIVE')
    paint_everything(stroke, obj)
    after = by_name(obj)
    check(sorted(after) == sorted(start), "current group: no other groups are created")
    check(after["j_sk_s_b_l"].any(), "current group: the active group gets weight")
    others = [name for name in start if name != "j_sk_s_b_l"]
    before = np.stack([start[name] for name in others], axis=1)
    now = np.stack([after[name] for name in others], axis=1)
    changed = after["j_sk_s_b_l"] > 1e-3
    a = before[changed] / before[changed].sum(axis=1, keepdims=True)
    b = now[changed] / now[changed].sum(axis=1, keepdims=True)
    check(float(np.abs(a - b).max()) < 1e-3, "current group: the other bones keep their proportions")
    check(float(np.abs(bone_total(after) - bone_total(start)).max()) < 1e-4, "current group: totals kept")


def test_restore():
    _rig, body, obj = skirt.build()
    start = by_name(obj)
    names = [group.name for group in obj.vertex_groups]
    active = obj.vertex_groups.active_index
    stroke = new_stroke(obj, body)
    paint_everything(stroke, obj)
    check(sorted(stroke.created_group_names()) == sorted(skirt.skirt_bone_names()),
          "groups: the 18 skirt groups are created and reported")
    stroke.restore()
    obj.data.update()
    check([group.name for group in obj.vertex_groups] == names, "restore: created groups are removed")
    check(same_weights(start, by_name(obj), 0.0), "restore: the weights are exactly as before")
    check(obj.vertex_groups.active_index == active, "restore: the active group is kept")


def test_max_groups():
    for limit in (8, 4):
        _rig, body, obj = skirt.build()
        obj.vertex_groups.new(name="iv_shiri_l")
        names = ("j_sebo_a", "j_kosi", "j_asi_a_l", "j_asi_a_r", "j_asi_b_l", "j_asi_b_r", "iv_shiri_l")
        count = len(obj.data.vertices)
        for name, weight in zip(names, (0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05)):
            obj.vertex_groups[name].add(list(range(count)), weight, 'REPLACE')
        stroke = new_stroke(obj, body, max_groups=limit)
        paint_everything(stroke, obj)
        after = by_name(obj)
        used = sum((values > 0.0).astype(int) for values in after.values())
        below = heights(obj) < 0.10  # Skirt leaves everything above Blend Start alone
        check(int(used[below].max()) <= limit, "max groups: at most {:d} per vertex".format(limit))
        check(float(np.abs(bone_total(after) - 1.0).max()) < 1e-4, "max groups {:d}: totals kept".format(limit))
        kept = stroke.body_fractions(range(count))
        body_share = bone_total(after) - skirt_total(after)
        check(float(np.abs(body_share - kept).max()) < 1e-4,
              "max groups {:d}: the body keeps its share".format(limit))


def test_pose_independence():
    rig, body, obj = skirt.build()
    paint_everything(new_stroke(obj, body), obj)
    rest = by_name(obj)
    rig, body, obj = skirt.build()
    skirt.scale_thighs(rig, 1.3)
    rig.pose.bones["j_kosi"].rotation_mode = 'XYZ'
    rig.pose.bones["j_kosi"].rotation_euler = (0.3, 0.0, 0.2)
    bpy.context.view_layer.update()
    paint_everything(new_stroke(obj, body), obj)
    check(same_weights(rest, by_name(obj), 1e-6), "pose: the same weights whatever the pose")
    check(rig.data.pose_position == 'POSE', "pose: the rig is shown posed again")


def test_no_body():
    _rig, body, obj = skirt.build()
    hem = skirt.index(skirt.ROWS - 1, 10)
    for group in obj.vertex_groups:
        group.remove([hem])
    stroke = new_stroke(obj, None, skin_weight=0.0)
    paint_everything(stroke, obj)
    after = by_name(obj)
    check(abs(skirt_total(after)[hem] - 1.0) < 1e-6, "no body: a vertex without weights hangs from the skirt bones")
    check(hem not in stroke.unweighted, "no body: all skirt anyway, nothing missing")

    _rig, body, obj = skirt.build()
    middle = skirt.index(12, 10)
    for group in obj.vertex_groups:
        group.remove([middle])
    stroke = new_stroke(obj, None, skin_weight=0.0)
    paint_everything(stroke, obj)
    check(middle in stroke.unweighted and abs(skirt_total(by_name(obj))[middle] - 1.0) < 1e-6,
          "no body: a vertex without body weights is reported and hangs from the skirt bones")

    _rig, body, obj = skirt.build()
    for group in obj.vertex_groups:
        group.remove([middle])
    stroke = new_stroke(obj, body, skin_weight=0.0)
    paint_everything(stroke, obj)
    after = by_name(obj)
    share = skirting.height_fraction(heights(obj)[middle], 0.10, 0.30)
    check(middle not in stroke.unweighted and abs(skirt_total(after)[middle] - share) < 1e-4,
          "body: a vertex without body weights takes them from the body")


def test_sleeves():
    rig, body, _obj = skirt.build()
    sleeve = skirt.build_sleeve(rig)
    check(float(heights(sleeve).min()) < -0.10, "sleeves: the cuff hangs well below the hips")
    start = by_name(sleeve)
    stroke = new_stroke(sleeve, body, skin_weight=0.0)
    paint_everything(stroke, sleeve)
    check(not stroke.modified and same_weights(start, by_name(sleeve), 0.0),
          "sleeves: cloth weighted to the arm keeps its weights below the hips")

    # Skirt weights it got before go back to the arm.
    for vert in sleeve.data.vertices:
        for elem in vert.groups:
            elem.weight *= 0.6
    sleeve.vertex_groups.new(name="j_sk_s_b_l").add(list(range(len(sleeve.data.vertices))), 0.4, 'REPLACE')
    paint_everything(new_stroke(sleeve, body, skin_weight=0.0), sleeve)
    after = by_name(sleeve)
    arm = {name: values for name, values in after.items() if not name.startswith("j_sk_")}
    check(not skirt_total(after).any() and same_weights(start, arm, 1e-4), "sleeves: skirt weights go back to the arm")

    # Half on the hand (13 cm off the body, so no Skin Weight): only the other half gives way to the
    # skirt bones.
    _rig, body, obj = skirt.build()
    hem = skirt.index(skirt.ROWS - 1, 10)
    for elem in obj.data.vertices[hem].groups:
        elem.weight *= 0.5
    obj.vertex_groups.new(name="j_te_l").add([hem], 0.5, 'REPLACE')
    paint_everything(new_stroke(obj, body), obj)
    after = by_name(obj)
    check(abs(skirt_total(after)[hem] - 0.5) < 1e-4 and abs(after["j_te_l"][hem] - 0.25) < 1e-4,
          "sleeves: a vertex half on the hand is half skirt, and keeps half its hand weight")
    again = new_stroke(obj, body)
    paint_everything(again, obj)
    check(not again.modified and same_weights(after, by_name(obj), 1e-6), "sleeves: a second pass changes nothing")


def test_strength():
    # Without a group limit (halfway, a vertex can have the old groups and the new ones).
    _rig, body, obj = skirt.build()
    start = by_name(obj)
    stroke = new_stroke(obj, body, max_groups=32)
    paint_everything(stroke, obj)
    full = by_name(obj)
    stroke.restore()
    obj.data.update()
    paint_everything(new_stroke(obj, body, max_groups=32), obj, factor=0.5)
    half = by_name(obj)
    zero = np.zeros(len(obj.data.vertices))
    worst = max(float(np.abs(half.get(n, zero) - 0.5 * (start.get(n, zero) + full.get(n, zero))).max())
                for n in full)
    check(worst < 1e-3, "strength: half strength goes half the way ({:.1e})".format(worst))
    stroke = new_stroke(obj, body)
    paint_everything(stroke, obj, factor=0.5)
    used = sum((values > 0.0).astype(int) for values in by_name(obj).values())
    check(int(used.max()) <= 8, "strength: the group limit holds halfway too")


# -----------------------------------------------------------------------------
# The button

def run_button():
    return bpy.ops.magic_fit.skirt_weights()


def test_operator():
    _rig, body, obj = skirt.build()
    settings = use_settings(body)
    paint_everything(new_stroke(obj, body, **skirting.stroke_params(settings)), obj)
    engine = by_name(obj)

    _rig, body, obj = skirt.build()
    settings = use_settings(body)
    bpy.ops.object.mode_set(mode='OBJECT')
    check(bpy.ops.magic_fit.skirt_weights.poll(), "button: ready in Object Mode")
    check(run_button() == {'FINISHED'}, "button: finishes")
    check(same_weights(engine, by_name(obj), 1e-6), "button: the same weights as painting everything")
    check(run_button() == {'CANCELLED'}, "button: a second time changes nothing")

    # Weight Paint mode with face selection masking: only the selected faces change.
    _rig, body, obj = skirt.build()
    use_settings(body)
    start = by_name(obj)
    mesh = obj.data
    select = np.zeros(len(mesh.polygons), dtype=bool)
    select[:len(select) // 2] = True
    mesh.polygons.foreach_set("select", select)
    mesh.use_paint_mask = True
    allowed = operators._selection_mask(mesh)
    run_button()
    after = by_name(obj)
    changed = np.zeros(len(mesh.vertices), dtype=bool)
    for name, values in after.items():
        changed |= np.abs(values - start.get(name, 0.0)) > 0.0
    check(changed.any() and not changed[~allowed].any(), "button: selection masking is respected")

    # Hidden vertices are left alone.
    _rig, body, obj = skirt.build()
    use_settings(body)
    bpy.ops.object.mode_set(mode='OBJECT')
    start = by_name(obj)
    hidden = np.zeros(len(obj.data.vertices), dtype=bool)
    hidden[skirt.index(skirt.ROWS - 1, 0):] = True
    obj.data.vertices.foreach_set("hide", hidden)
    run_button()
    after = by_name(obj)
    check(all(np.array_equal(after.get(n, np.zeros(len(hidden)))[hidden], start.get(n, np.zeros(len(hidden)))[hidden])
              for n in after), "button: hidden vertices are left alone")

    # Apply to All Selected.
    _rig, body, obj = skirt.build()
    use_settings(body, skirt_apply_to_selected=True)
    bpy.ops.object.mode_set(mode='OBJECT')
    second = obj.copy()
    second.data = obj.data.copy()
    bpy.context.scene.collection.objects.link(second)
    second.select_set(True)
    body.select_set(True)
    run_button()
    check(all("j_sk_f_c_l" in o.vertex_groups for o in (obj, second)) and "j_sk_f_c_l" not in body.vertex_groups,
          "button: Apply to All Selected weights every selected skirt, not the body")

    # Refused without skirt bones, or without a body when it's needed.
    _rig, body, obj = skirt.build()
    use_settings(None)
    bpy.ops.object.mode_set(mode='OBJECT')
    check(not bpy.ops.magic_fit.skirt_weights.poll(), "button: needs a body with Skin Weight")
    check(operators.check_skirt_ready(obj, bpy.context.scene.magic_fit).startswith("Pick the body"),
          "button: says to pick the body")
    use_settings(None, skirt_skin_weight=0.0)
    check(bpy.ops.magic_fit.skirt_weights.poll(), "button: no body needed without Skin Weight")
    plain = skirt.build_rig("Plain", chains={})
    bpy.context.view_layer.objects.active = obj
    obj.modifiers["Armature"].object = plain
    check(not bpy.ops.magic_fit.skirt_weights.poll(), "button: refused without skirt bones")
    check("no skirt bones" in operators.check_skirt_ready(obj, bpy.context.scene.magic_fit),
          "button: says the armature has no skirt bones")


def test_brush_readiness():
    _rig, body, obj = skirt.build()
    settings = use_settings(body)
    check(operators.check_ready(bpy.context, settings) is None, "brush: ready in Skirt mode")
    obj.modifiers["Armature"].show_viewport = False
    check(operators.check_ready(bpy.context, settings) == "Skirt needs an Armature modifier",
          "brush: needs an Armature modifier")
    obj.modifiers["Armature"].show_viewport = True
    settings.group_mode = 'ACTIVE'
    obj.vertex_groups.active_index = obj.vertex_groups["j_kosi"].index
    check(operators.check_ready(bpy.context, settings) is None, "brush: Current Group with a bone group")


def test_partial_rigs():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    cape = skirt.build_rig("Cape", chains={"b": skirt.CHAINS["b"]})
    found = skirting.SkirtRig.from_object(cape)
    rng = np.random.default_rng(1)
    h = rng.uniform(-0.8, 0.2, 500)
    angle = rng.uniform(-math.pi, math.pi, 500)
    shares = found.distribution(h, angle, spread=1.5, joint_blend=0.1)
    check(len(found.chains) == 2 and shares.shape == (500, 6), "cape rig: back chains only")
    check(np.isfinite(shares).all() and np.allclose(shares.sum(axis=1), 1.0), "cape rig: shares add up to 1")
    one = skirting.SkirtRig.from_object(skirt.build_rig("One", chains={"b": skirt.CHAINS["b"]}, sides="l"))
    shares = one.distribution(h, angle, spread=1.5, joint_blend=0.1)
    check(shares.shape == (500, 3) and np.allclose(shares.sum(axis=1), 1.0), "one chain: it takes everything")


def test_defaults():
    settings = bpy.context.scene.magic_fit
    rna = settings.bl_rna.properties
    expected = {
        "skirt_blend_start": 0.0, "skirt_blend_length": 0.4, "skirt_skin_weight": 1.0,
        "skirt_skin_distance": 0.07, "skirt_body_weight": 0.0, "skirt_spread": 1.5,
        "skirt_joint_blend": 0.1, "skirt_max_groups": 8,
    }
    check(all(abs(rna[name].default - value) < 1e-6 for name, value in expected.items()),
          "defaults: the settings start at the chosen values")
    import inspect
    engine = inspect.signature(skirting.SkirtStroke.__init__).parameters
    check(all(abs(engine[name[6:]].default - value) < 1e-6 for name, value in expected.items()),
          "defaults: the engine's match the settings'")


def test_registered():
    check(hasattr(bpy.ops.magic_fit, "skirt_weights"), "the Skirt Weights operator is registered")
    settings = bpy.context.scene.magic_fit
    items = [item.identifier for item in settings.bl_rna.properties["mode"].enum_items]
    check(items[:3] == ['COPY', 'STRAIGHTEN', 'SKIRT'], "Skirt keeps its place among the modes (files store it)")


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
