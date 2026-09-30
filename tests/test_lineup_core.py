"""Headless tests for Line Up (the Body Fit panel's Line Up section).

Run from the repository root:
    blender -b --factory-startup --python tests/test_lineup_core.py
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

import lineup as fixture  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import lineup, operators  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def mm(value):
    return "{:.2f} mm".format(1000.0 * value)


# -----------------------------------------------------------------------------
# Names

def test_classify():
    cases = {
        # 3ds Max Biped, Valve
        "Bip001-L-UpperArm": ("upperarm", "l"), "Bip001 R Forearm": ("forearm", "r"), "Bip01_L_Thigh": ("thigh", "l"),
        "Bip001-R-Calf": ("shin", "r"), "Bip001-L-Toe0": ("foot", "l"), "Bip001-L-Finger12": ("hand", "l"),
        "Bip001-Spine2": ("torso", None), "Bip001-Neck": ("neck", None), "Bip001-L-Clavicle": ("clavicle", "l"),
        "ValveBiped.Bip01_L_Forearm": ("forearm", "l"), "Ab-L-UpperArm-Tw0": ("upperarm", "l"),
        "Ab-L-Becep": ("upperarm", "l"), "Ab-R-Calf-Tw1": ("shin", "r"), "Ab-NeckSub": ("neck", None),
        # Mixamo
        "mixamorig:Hips": ("torso", None), "mixamorig:LeftUpLeg": ("thigh", "l"), "mixamorig:LeftLeg": ("leg", "l"),
        "mixamorig:RightForeArm": ("forearm", "r"), "mixamorig:LeftShoulder": ("clavicle", "l"),
        "mixamorig:LeftHandIndex1": ("hand", "l"), "mixamorig:RightToeBase": ("foot", "r"),
        "mixamorig:HeadTop_End": ("head", None),
        # Unreal
        "pelvis": ("torso", None), "spine_03": ("torso", None), "neck_01": ("neck", None), "clavicle_l": ("clavicle", "l"),
        "upperarm_twist_01_l": ("upperarm", "l"), "lowerarm_r": ("forearm", "r"), "calf_l": ("shin", "l"),
        "ball_r": ("foot", "r"), "thigh_twist_01_r": ("thigh", "r"),
        # Unity / VRChat / CATS / VRoid
        "Hips": ("torso", None), "Chest": ("torso", None), "Arm_L": ("upperarm", "l"), "Elbow_R": ("forearm", "r"),
        "Leg_L": ("leg", "l"), "Knee_R": ("shin", "r"), "Left elbow": ("forearm", "l"), "Right knee": ("shin", "r"),
        "Left ankle": ("foot", "l"), "Left wrist": ("hand", "l"), "Left shoulder": ("clavicle", "l"),
        "Left arm": ("upperarm", "l"), "UpperArm.L": ("upperarm", "l"), "LowerLeg.R": ("shin", "r"),
        "J_Bip_C_Hips": ("torso", None), "J_Bip_L_UpperArm": ("upperarm", "l"), "J_Bip_R_LowerLeg": ("shin", "r"),
        "J_Bip_L_ToeBase": ("foot", "l"), "Middle Finger 02_L": ("hand", "l"), "Thigh Jiggle_L": ("thigh", "l"),
        # Rigify, DAZ
        "DEF-upper_arm.L": ("upperarm", "l"), "DEF-forearm.R": ("forearm", "r"), "DEF-thigh.L": ("thigh", "l"),
        "DEF-shin.R": ("shin", "r"), "DEF-spine.003": ("torso", None), "DEF-breast.L": ("torso", "l"),
        "lShldrBend": ("upperarm", "l"), "rForearmTwist": ("forearm", "r"), "lThighBend": ("thigh", "l"),
        "rShin": ("shin", "r"), "lCollar": ("clavicle", "l"), "chestUpper": ("torso", None), "neckLower": ("neck", None),
        "lthigh": ("thigh", "l"),
        # The Sims (named), FFXIV
        "b__L_UpperArm__": ("upperarm", "l"), "b__R_Calf__": ("shin", "r"), "b__Spine1__": ("torso", None),
        "b__L_ShoulderTwist__": ("upperarm", "l"), "b__R_ForearmTwist__": ("forearm", "r"),
        "b__L_Mid1__": ("hand", "l"), "b__CAS_R_Breast__": ("torso", "r"), "b__R_ThighTwist__": ("thigh", "r"),
        "j_kosi": ("torso", None), "j_sebo_c": ("torso", None), "j_kubi": ("neck", None), "j_kao": ("head", None),
        "j_sako_l": ("clavicle", "l"), "j_ude_a_l": ("upperarm", "l"), "j_ude_b_r": ("forearm", "r"),
        "j_te_l": ("hand", "l"), "j_oya_a_r": ("hand", "r"), "j_asi_a_l": ("thigh", "l"), "j_asi_c_r": ("shin", "r"),
        "j_asi_d_l": ("foot", "l"), "j_asi_e_r": ("foot", "r"), "n_hizasoubi_l": ("shin", "l"),
        # MMD
        "左腕": ("upperarm", "l"), "右ひじ": ("forearm", "r"), "左足首": ("foot", "l"), "右足": ("thigh", "r"),
        "左ひざ": ("shin", "l"), "左肩": ("clavicle", "l"), "首": ("neck", None), "上半身": ("torso", None),
        # Sides written as hips, and names saying nothing
        "LeftHip": ("thigh", "l"), "Tail_root": (None, None), "bone_07": (None, None), "n_root": (None, None),
    }
    wrong = {name: (lineup.classify(name), want) for name, want in cases.items() if lineup.classify(name) != want}
    check(not wrong, "names: {:d} bone names of 10 naming schemes classified ({})".format(
        len(cases), wrong if wrong else "all right"))


def test_leg_bones():
    # Mixamo: LeftUpLeg (thigh) > LeftLeg (shin). VRChat: Leg_L (thigh) > Knee_L, with a jiggle bone.
    parents = {"Hips": None, "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg"}
    parts = lineup.bone_parts(list(parents), parents)
    check(parts["LeftUpLeg"] == "thigh_l" and parts["LeftLeg"] == "shin_l" and parts["LeftFoot"] == "foot_l",
          "legs: Mixamo's Leg is the shin")
    parents = {"Hips": None, "Leg_L": "Hips", "Knee_L": "Leg_L", "Thigh Jiggle_L": "Leg_L", "Skirt_L": "Leg_L",
               "Foot_L": "Knee_L"}
    parts = lineup.bone_parts(list(parents), parents)
    check(parts["Leg_L"] == "thigh_l" and parts["Knee_L"] == "shin_l" and parts["Skirt_L"] == "thigh_l",
          "legs: VRChat's Leg is the thigh, a skirt bone below it goes with it")
    parts, named = lineup.group_parts_by_name(["LeftUpLeg", "LeftLeg", "Leg_R", "Knee_R"])
    check(parts["LeftLeg"] == "shin_l" and parts["Leg_R"] == "thigh_r" and named == 4,
          "legs: settled by the other groups without an armature")


def test_main_bones():
    # 3ds Max Biped hangs the clavicles from the neck; helpers beside them hang from the spine, higher up.
    parents = {"Bip001-Pelvis": None, "Bip001-Spine": "Bip001-Pelvis", "Bip001-Spine2": "Bip001-Spine",
               "Bip001-Neck": "Bip001-Spine2", "Bip001-L-Clavicle": "Bip001-Neck",
               "Bip001-L-UpperArm": "Bip001-L-Clavicle", "Bip001-L-Forearm": "Bip001-L-UpperArm",
               "Ab-L-Trape0": "Bip001-Spine2", "Ab-L-Shoulder0": "Bip001-Spine2",
               "Ab-L-UpperArm-Tw0": "Bip001-L-UpperArm"}
    parts = lineup.bone_parts(list(parents), parents)
    mains = lineup.main_bones(parts, parents)
    check(mains["clavicle_l"] == "Bip001-L-Clavicle" and mains["upperarm_l"] == "Bip001-L-UpperArm",
          "main bones: the clavicle the arm hangs from, not a helper beside it ({})".format(mains["clavicle_l"]))
    check(mains["torso"] == "Bip001-Pelvis" and mains["neck"] == "Bip001-Neck", "main bones: pelvis and neck")


def test_finger_names():
    cases = {
        # 3ds Max Biped (Finger0 is the thumb), Mixamo, Unreal
        "Bip001-L-Finger0": "thumb", "Bip001-L-Finger12": "index", "Bip001-R-Finger21": "middle",
        "Bip001-L-Finger3": "ring", "Bip001-L-Finger42": "little", "mixamorig:LeftHandIndex1": "index",
        "mixamorig:RightHandPinky3": "little", "mixamorig:LeftHandThumb2": "thumb", "thumb_01_l": "thumb",
        "index_02_r": "index", "middle_03_l": "middle", "ring_01_r": "ring", "pinky_02_l": "little",
        "index_metacarpal_l": None,
        # VRChat, VRoid, Unity, Rigify, DAZ
        "Index Finger 02_L": "index", "Little Finger_R": "little", "Thumb 03_L": "thumb", "Ring Finger _L": "ring",
        "J_Bip_L_Little1": "little", "Left Middle Intermediate": "middle", "DEF-f_index.01.L": "index",
        "DEF-thumb.02.R": "thumb", "DEF-palm.01.L": None, "lIndex1": "index", "rMid2": "middle", "lCarpal1": None,
        # The Sims, FFXIV, MMD
        "b__L_Thumb0__": "thumb", "b__R_Mid2__": "middle", "b__L_Pinky1__": "little", "j_oya_a_l": "thumb",
        "j_hito_b_r": "index", "iv_naka_c_l": "middle", "j_kusu_a_l": "ring", "j_ko_b_r": "little", "j_te_l": None,
        "Wrist_L": None, "左親指１": "thumb", "右人指２": "index", "左中指３": "middle", "左薬指１": "ring", "右小指２": "little",
    }
    wrong = {name: (lineup.finger_of(name), want) for name, want in cases.items() if lineup.finger_of(name) != want}
    check(not wrong, "fingers: {:d} finger bone names of 11 naming schemes ({})".format(
        len(cases), wrong if wrong else "all right"))


def test_finger_segments():
    # Biped: Finger0 > Finger01 > Finger02 > Finger0Nub, numbered from the hand out (the nub joins the third);
    # a helper on the hand stays with it.
    parents = {"Bip001-Pelvis": None, "Bip001-L-Clavicle": "Bip001-Pelvis", "Bip001-L-UpperArm": "Bip001-L-Clavicle",
               "Bip001-L-Forearm": "Bip001-L-UpperArm", "Bip001-L-Hand": "Bip001-L-Forearm",
               "Bip001-L-Finger0": "Bip001-L-Hand", "Bip001-L-Finger01": "Bip001-L-Finger0",
               "Bip001-L-Finger02": "Bip001-L-Finger01", "Bip001-L-Finger0Nub": "Bip001-L-Finger02",
               "Bip001-L-Finger1": "Bip001-L-Hand", "Bip001-L-Finger11": "Bip001-L-Finger1",
               "SC_Weapon": "Bip001-L-Hand"}
    parts = lineup.bone_parts(list(parents), parents)
    want = {"Bip001-L-Finger0": "thumb1_l", "Bip001-L-Finger01": "thumb2_l", "Bip001-L-Finger02": "thumb3_l",
            "Bip001-L-Finger0Nub": "thumb3_l", "Bip001-L-Finger1": "index1_l", "Bip001-L-Finger11": "index2_l",
            "SC_Weapon": "hand_l", "Bip001-L-Hand": "hand_l"}
    wrong = {name: (parts[name], part) for name, part in want.items() if parts[name] != part}
    check(not wrong, "finger segments: numbered from the hand out ({})".format(wrong or "all right"))
    # Unreal 5: a metacarpal named after its finger is the palm's; the finger starts after it.
    parents = {"pelvis": None, "clavicle_r": "pelvis", "upperarm_r": "clavicle_r", "lowerarm_r": "upperarm_r",
               "hand_r": "lowerarm_r", "index_metacarpal_r": "hand_r", "index_01_r": "index_metacarpal_r",
               "index_02_r": "index_01_r", "thumb_01_r": "hand_r"}
    parts = lineup.bone_parts(list(parents), parents)
    check(parts["index_metacarpal_r"] == "hand_r" and parts["index_01_r"] == "index1_r" and
          parts["index_02_r"] == "index2_r" and parts["thumb_01_r"] == "thumb1_r",
          "finger segments: metacarpals stay with the palm")


# -----------------------------------------------------------------------------
# Weights

def test_weights():
    rng = np.random.default_rng(1)
    count, groups = 50, 6
    vertex = np.repeat(np.arange(count), 3)
    group = np.concatenate([rng.choice(groups, 3, replace=False) for _ in range(count)])
    weight = rng.random(len(vertex)) + 0.1
    weights = lineup.Weights(count, ["g{:d}".format(k) for k in range(groups)], vertex, group, weight)
    dense = np.zeros((count, groups))
    np.add.at(dense, (vertex, group), weight)
    dense /= dense.sum(axis=1, keepdims=True)
    overlap = dense.T @ dense
    np.fill_diagonal(overlap, 0.0)
    check(np.allclose(weights.overlap(), overlap), "weights: shared weight between groups (sparse = dense)")
    check(np.allclose(weights.column(2), dense[:, 2]) and np.allclose(weights.mass(), dense.sum(axis=0)),
          "weights: columns and masses")
    points = rng.random((count, 3))
    check(np.allclose(weights.centroids(points), (dense.T @ points) / dense.sum(axis=0)[:, None]), "weights: centroids")
    parts = {"g0": "torso", "g1": "thigh_l", "g2": "thigh_l"}
    by_part = weights.by_part(parts)
    check(np.allclose(by_part[:, lineup.PART_INDEX["thigh_l"]], dense[:, 1] + dense[:, 2]) and
          np.allclose(by_part.sum(axis=1), dense[:, :3].sum(axis=1)), "weights: summed per part, unknown groups left out")
    stacked = lineup.Weights.stack(weights.names, [weights, weights])
    check(stacked.count == 2 * count and np.allclose(stacked.column(4)[count:], dense[:, 4]), "weights: stacked")


def test_joints_from_weights():
    # A straight tube along X, its weight handing over from one part to the next around x = 0.3.
    x = np.repeat(np.linspace(0.0, 0.6, 61), 8)
    angles = np.tile(np.linspace(0.0, 2 * math.pi, 8, endpoint=False), 61)
    points = np.column_stack((x, 0.05 * np.cos(angles), 1.0 + 0.05 * np.sin(angles)))
    share = np.clip((x - 0.28) / 0.04, 0.0, 1.0)
    part_weights = np.zeros((len(x), len(lineup.PARTS)))
    part_weights[:, lineup.PART_INDEX["upperarm_l"]] = 1.0 - share
    part_weights[:, lineup.PART_INDEX["forearm_l"]] = share
    joints = lineup.joints_from_weights(points, part_weights)
    check("forearm_l" in joints and np.linalg.norm(joints["forearm_l"] - (0.3, 0.0, 1.0)) < 1e-6,
          "joints: the handover of a tube's weights is its joint")


# -----------------------------------------------------------------------------
# Solving

def skeleton(joints):
    """Line Up's joints (part -> position) of a fixture skeleton."""
    return {part: np.asarray(joints[name]) for name, part in fixture.JOINT_PART.items() if name in joints}


def test_solve_lines_up_joints():
    body = skeleton(fixture.mirror(fixture.BODY_JOINTS))
    model = skeleton(fixture.model_joints(scale=0.8, leg_scale=1.15, arm_angle=5.0, turn=90.0, offset=(0.3, -0.2, 0.1)))
    solution = lineup.solve(model, body)
    worst = 0.0
    for part, joint in model.items():
        worst = max(worst, np.linalg.norm(solution.move(part, joint) - body[part]))
        after = lineup.NEXT.get(part)
        if after in model:
            worst = max(worst, np.linalg.norm(solution.move(part, model[after]) - body[after]))
    check(worst < 1e-9, "solve: every joint and segment end lands on the body's ({:.1e} m)".format(worst))
    check(abs(abs(solution.turn) - 90.0) < 1e-6 and abs(solution.scale - 1.0 / 0.8) < 1e-6,
          "solve: turned {:.1f} degrees, scaled {:.3f}".format(solution.turn, solution.scale))
    for part in ("foot_l", "head"):
        check(np.allclose(solution.rotations[part], solution.rotations["torso"]),
              "solve: the {:s} keeps its angle to the torso".format(part.split("_")[0]))
    check(np.allclose(solution.rotations["hand_l"], solution.rotations["forearm_l"]), "solve: hands follow the forearms")
    for matrix in solution.transforms.values():
        if not np.isfinite(matrix).all():
            check(False, "solve: finite transforms")
            break


def test_solve_without_stretching():
    body = skeleton(fixture.mirror(fixture.BODY_JOINTS))
    model = skeleton(fixture.model_joints(scale=0.9, leg_scale=1.2, arm_angle=0.0))
    solution = lineup.solve(model, body, stretch=False)
    check(not solution.stretches, "no stretch: no segment stretched")
    worst_angle, worst_length, worst_gap = 0.0, 0.0, 0.0
    for part, after in lineup.NEXT.items():
        if part not in model or after not in model or part in lineup.UPRIGHT | lineup.SHOULDERS:
            continue
        start, end = (solution.move(part, model[p]) for p in (part, after))
        wanted = body[after] - body[part]
        cosine = np.dot(end - start, wanted) / np.linalg.norm(end - start) / np.linalg.norm(wanted)
        worst_angle = max(worst_angle, math.degrees(math.acos(min(1.0, cosine))))
        worst_length = max(worst_length, abs(np.linalg.norm(end - start) -
                                             solution.scale * np.linalg.norm(model[after] - model[part])))
        # Segments stay joined: each starts where the one above put its joint.
        parent = lineup.PARENT[part]
        worst_gap = max(worst_gap, np.linalg.norm(start - solution.move(parent, model[part])))
    check(worst_angle < 1e-4, "no stretch: every segment points the body's way ({:.1e} deg)".format(worst_angle))
    check(worst_length < 1e-9 and worst_gap < 1e-9, "no stretch: lengths kept, scaled with the torso, joined up")
    hips = (lineup.apply_matrix(solution.transforms["torso"], (model["thigh_l"] + model["thigh_r"]) / 2.0))
    check(np.allclose(hips, (body["thigh_l"] + body["thigh_r"]) / 2.0), "no stretch: the hips land on the body's")


def test_solve_missing_parts():
    body = skeleton(fixture.mirror(fixture.BODY_JOINTS))
    model = skeleton(fixture.model_joints(scale=1.1, feet=False))
    model.pop("foot_l", None)
    model.pop("foot_r", None)
    solution = lineup.solve(model, body)
    check(np.allclose(solution.transforms["foot_l"], solution.transforms["shin_l"]) and "foot_l" in solution.missing,
          "missing: a model without feet moves them with the shins")
    del model["thigh_l"]
    check(lineup.solve(model, body).half == "upper", "missing: a model without hips lines up as an upper body")
    for part in ("upperarm_l", "upperarm_r", "clavicle_l", "clavicle_r", "neck"):
        model.pop(part, None)
    try:
        lineup.solve(model, body)
        check(False, "missing: refuses a model without hips or shoulders")
    except lineup.LineUpError as error:
        check("hips" in str(error), "missing: refuses a model without hips or shoulders ({})".format(error))


def test_solve_leg_without_foot():
    # A body cut above the ankles (The Sims): its shins, whose ends it doesn't have, turn toward the body's
    # ankles by where their points lie (their tips, here its ankles) and stretch as much as the thighs, so
    # a leg as long as the body's in proportion reaches the body's ankle.
    body = skeleton(fixture.mirror(fixture.BODY_JOINTS))
    model = skeleton(fixture.model_joints(scale=0.9, arm_angle=10.0))
    tips = {}
    for side in lineup.SIDES:
        hip = model["thigh_" + side]
        model["shin_" + side] = hip + 0.85 * (model["shin_" + side] - hip)  # shorter legs, in proportion
        tips["shin_" + side] = hip + 0.85 * (model.pop("foot_" + side) - hip)
    solution = lineup.solve(model, body, tips=tips)
    same = all(abs(solution.stretches["shin_" + side] - solution.stretches["thigh_" + side]) < 1e-9
               for side in lineup.SIDES)
    reach = max(np.linalg.norm(solution.move("shin_" + side, tips["shin_" + side]) - body["foot_" + side])
                for side in lineup.SIDES)
    check(same and solution.stretches["thigh_l"] > 1.1 and reach < 1e-9,
          "no foot: the shins stretch like the thighs (x{:.2f}) and reach the body's ankles".format(
              solution.stretches["shin_l"]))


def with_fingers(joints, curl, thumb=3, forward=(0.0, -1.0, 0.0)):
    """Line Up's joints of a fixture skeleton and its fingers (`fixture.finger_chains`, the thumb with
    ``thumb`` segments), and the tips of the fingers' last segments."""
    result, tips = skeleton(joints), {}
    for side in lineup.SIDES:
        for finger, points in fixture.finger_chains(joints, side, curl, forward).items():
            count = thumb if finger == "thumb" else 3
            for k in range(count):
                result[lineup.digit_name(finger, k + 1, side)] = np.asarray(points[k])
            tips[lineup.digit_name(finger, count, side)] = np.asarray(points[count])
    return result, tips


def test_solve_fingers():
    # The body's fingers are straight, the model's (turned 90 degrees, T-pose) bent 30 degrees more at every
    # joint, and its thumb has three segments to the body's two: lined up, every segment of the model's
    # fingers points along the body's finger, keeps its length and stays joined.
    body_joints = fixture.mirror(fixture.BODY_JOINTS)
    body, _tips = with_fingers(body_joints, 0.0, thumb=2)
    turned = fixture.model_joints(scale=0.9, arm_angle=0.0, turn=90.0)
    model, tips = with_fingers(turned, 30.0, forward=fixture.rotate_z([(0.0, -1.0, 0.0)], 90.0)[0])
    for stretch in (True, False):
        solution = lineup.solve(model, body, stretch=stretch, tips=tips)
        worst_angle, worst_length, worst_gap = 0.0, 0.0, 0.0
        for side in lineup.SIDES:
            for finger in fixture.FINGERS:
                straight = fixture.finger_chains(body_joints, side)[finger]
                wanted = straight[1] - straight[0]
                for k in (1, 2, 3):
                    part = lineup.digit_name(finger, k, side)
                    end = model.get(lineup.digit_name(finger, k + 1, side), tips.get(part))
                    start, stop = solution.move(part, model[part]), solution.move(part, end)
                    cosine = np.dot(stop - start, wanted) / np.linalg.norm(stop - start) / np.linalg.norm(wanted)
                    worst_angle = max(worst_angle, math.degrees(math.acos(min(1.0, cosine))))
                    worst_length = max(worst_length, abs(np.linalg.norm(stop - start) -
                                                         solution.scale * np.linalg.norm(end - model[part])))
                    worst_gap = max(worst_gap, np.linalg.norm(start - solution.move(lineup.PARENT[part], model[part])))
        check(worst_angle < 0.01 and worst_length < 1e-9 and worst_gap < 1e-9,
              "fingers ({:s}): curled fingers straighten along the body's ({:.1e} deg), keep their lengths and "
              "stay joined".format("stretched" if stretch else "not stretched", worst_angle))
    # A body without finger bones: the model's fingers at least straighten, along their first segments.
    plain = skeleton(body_joints)
    solution = lineup.solve(model, plain, tips=tips)
    worst = 0.0
    for part, (finger, segment, side) in lineup.DIGIT_PARTS.items():
        if segment == 1:
            continue
        first = lineup.digit_name(finger, 1, side)
        end = model.get(lineup.digit_name(finger, segment + 1, side), tips.get(part))
        along = solution.move(part, end) - solution.move(part, model[part])
        wanted = solution.move(first, model[lineup.digit_name(finger, 2, side)]) - solution.move(first, model[first])
        worst = max(worst, math.degrees(math.acos(min(1.0, np.dot(along, wanted) / np.linalg.norm(along) /
                                                        np.linalg.norm(wanted)))))
    check(worst < 0.01 and solution.fingers == 0,
          "fingers: without the body's, straightened along their first segments ({:.1e} deg)".format(worst))


def test_solve_neck_and_shoulders():
    # Rigs put the head joint anywhere from the skull's base to the jaw: a model whose head joint sits 4 cm
    # lower keeps its neck as it is, with the head, turned and moved with the torso, stretching or not.
    body = skeleton(fixture.mirror(fixture.BODY_JOINTS))
    model = skeleton(fixture.model_joints(scale=0.8, arm_angle=0.0, turn=90.0))
    model["head"] = model["head"] - 0.04 * 0.8 * (model["head"] - model["neck"]) / np.linalg.norm(
        model["head"] - model["neck"])
    for stretch in (True, False):
        solution = lineup.solve(model, body, stretch=stretch)
        rigid = all(np.allclose(solution.transforms[part], solution.transforms["torso"]) for part in ("neck", "head"))
        check(rigid and "neck" not in solution.stretches,
              "neck: moves with the torso, unstretched ({:s})".format("stretched" if stretch else "not stretched"))
    # Clavicles turn with the torso. A model with narrower shoulders and clavicles starting further out:
    # stretched, its shoulders land on the body's, and the clavicles' other ends (toward the neck) move with
    # the torso instead of being lifted like a hood.
    model = skeleton(fixture.model_joints(scale=0.9, arm_angle=0.0))
    for side, sign in (("l", 1.0), ("r", -1.0)):
        model["upperarm_" + side] = model["upperarm_" + side] - np.array((sign * 0.02, 0.0, 0.0))
        model["clavicle_" + side] = model["clavicle_" + side] + np.array((sign * 0.03, 0.0, -0.02))
    solution = lineup.solve(model, body)
    shoulder = solution.move("clavicle_l", model["upperarm_l"])
    inside = model["clavicle_l"] + np.array((-0.05, 0.0, 0.03))
    moved = np.linalg.norm(solution.move("clavicle_l", model["upperarm_l"]) -
                           lineup.apply_matrix(solution.transforms["torso"], model["upperarm_l"]))
    check(np.allclose(solution.rotations["clavicle_l"], solution.rotations["torso"]) and moved > 0.01 and
          np.linalg.norm(shoulder - body["upperarm_l"]) < 1e-9 and
          np.allclose(solution.move("clavicle_l", inside), lineup.apply_matrix(solution.transforms["torso"], inside)),
          "clavicle: turns with the torso, reaches the body's shoulder ({:.0f} mm out), leaves the neck side where "
          "it was".format(1000 * moved))


def test_stretch_is_smooth():
    # A thigh squashed and a shin stretched (the model's knee lower than the body's): along the leg,
    # through points weighted to both around the knee, the leg never folds and its stretch changes gradually.
    body = skeleton(fixture.mirror(fixture.BODY_JOINTS))
    model = skeleton(fixture.model_joints(scale=1.0))
    model["shin_l"] = model["shin_l"] + 0.1 * (model["foot_l"] - model["shin_l"])
    solution = lineup.solve(model, body)
    factors = solution.stretches["thigh_l"], solution.stretches["shin_l"]
    hip, knee, ankle = model["thigh_l"], model["shin_l"], model["foot_l"]
    line = np.concatenate([hip + np.outer(np.linspace(0.0, 1.0, 200, endpoint=False), knee - hip),
                           knee + np.outer(np.linspace(0.0, 1.0, 201), ankle - knee)])
    weights = np.zeros((len(line), len(lineup.PARTS)))
    share = np.clip((np.linalg.norm(line - hip, axis=1) - np.linalg.norm(knee - hip) + 0.02) / 0.04, 0.0, 1.0)
    weights[:, lineup.PART_INDEX["thigh_l"]] = 1.0 - share
    weights[:, lineup.PART_INDEX["shin_l"]] = share
    moved = lineup.deform(line, weights, solution)
    steps = np.linalg.norm(np.diff(moved, axis=0), axis=1) / np.linalg.norm(np.diff(line, axis=0), axis=1)
    top = max(factors) / (1.0 - lineup.RAMP_EASE)
    check(factors[0] < 0.95 < 1.05 < factors[1] and steps.min() > 0.5 and steps.max() < top + 1e-6 and
          np.abs(np.diff(steps)).max() < 0.05,
          "stretch: thigh x{:.2f}, shin x{:.2f}, the leg stretched x{:.2f} to x{:.2f} along it, gradually".format(
              factors[0], factors[1], steps.min(), steps.max()))
    ramp = lineup.ramp(np.array((-0.5, 0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0, 1.5)))
    check(np.allclose(ramp[[0, 1, 5, 6]], (0.0, 0.0, 1.0, 1.0)) and abs(ramp[3] - 0.5) < 1e-12 and
          ramp[2] < 1e-6 and ramp[4] > 1.0 - 1e-6, "stretch: eases in from each joint")


def test_deform():
    body = skeleton(fixture.mirror(fixture.BODY_JOINTS))
    model = skeleton(fixture.model_joints(scale=0.9, arm_angle=10.0))
    solution = lineup.solve(model, body, stretch=False)
    rng = np.random.default_rng(4)
    points = rng.random((20, 3))
    weights = np.zeros((20, len(lineup.PARTS)))
    weights[:10, lineup.PART_INDEX["forearm_l"]] = 1.0
    moved = lineup.deform(points, weights, solution)
    before = np.linalg.norm(points[:10, None] - points[None, :10], axis=2)
    after = np.linalg.norm(moved[:10, None] - moved[None, :10], axis=2)
    check(np.allclose(after, solution.scale * before), "deform: a part without stretching moves rigidly (and scales)")
    check(np.allclose(moved[10:], lineup.apply_matrix(solution.transforms["torso"], points[10:])),
          "deform: unweighted points move with the torso")


# -----------------------------------------------------------------------------
# The operator on the fixture

def fresh():
    scene = fixture.build()
    bpy.context.scene.body_fit_brush.lineup_stretch = True
    return scene


def test_armature_model():
    scene = fresh()
    model, armature = scene["vrc"], scene["vrc_armature"]
    wave_before = fixture.world_positions(model, "Wave") - fixture.world_positions(model, "Basis")
    fixture.select([model])
    check(operators.check_lineup_ready(bpy.context) is None, "armature: ready with one mesh of the model selected")
    meshes, armatures = operators.lineup_model(bpy.context)
    check(meshes == [model] and armatures == [armature], "armature: the model is the mesh and its armature")
    result = bpy.ops.magic_fit.line_up()
    check(result == {'FINISHED'}, "armature: lined up")
    errors = fixture.marker_errors(model, scene["vrc_markers"], scene["body_joints"])
    check(max(errors.values()) < 1e-5, "armature: every joint on the body's ({:s} at most)".format(mm(max(errors.values()))))
    # The rest pose moved along: bones on the body's joints, and the armature leaves the mesh as it is.
    armature = bpy.data.objects["VRC Armature"]
    heads = {name: np.array(armature.matrix_world @ bone.head_local) for name, bone in armature.data.bones.items()}
    body_joints = scene["body_joints"]
    bone_errors = [np.linalg.norm(heads[name] - body_joints[joint]) for name, joint in
                   (("Arm_L", "shoulder_l"), ("Elbow_R", "elbow_r"), ("Knee_L", "knee_l"), ("Foot_R", "ankle_r"),
                    ("Neck", "neck"))]
    check(max(bone_errors) < 1e-5, "armature: its bones moved onto the body's joints ({:s})".format(mm(max(bone_errors))))
    shown = fixture.displayed_positions(model)
    data = fixture.world_positions(model)
    check(np.abs(shown - data).max() < 1e-5, "armature: at rest it leaves the lined up mesh where it is")
    # The shape key turned with the hands: still 2 cm across the arm (scaled), now hanging down with it.
    wave = fixture.world_positions(model, "Wave") - fixture.world_positions(model, "Basis")
    hands = np.linalg.norm(wave_before, axis=1) > 1e-6
    forearm = body_joints["wrist_l"] - body_joints["elbow_l"]
    left = hands & (data[:, 0] > 0.3)
    lengths = np.linalg.norm(wave[left], axis=1)
    across = np.abs(wave[left] @ forearm) / np.linalg.norm(forearm) / lengths
    scale = 1.0 / fixture.VRC_JOINTS["scale"]
    check(np.allclose(lengths, 0.02 * scale, atol=1e-5) and across.max() < 1e-4,
          "armature: the shape key's offsets turned with the hands ({:.1f} mm, square to the forearm)".format(
              1000 * lengths.mean()))
    # Lining up again changes nothing.
    fixture.select([model])
    before = fixture.world_positions(model)
    bpy.ops.magic_fit.line_up()
    check(np.abs(fixture.world_positions(model) - before).max() < 1e-5, "armature: lining up again changes nothing")


def test_absolute_shape_keys():
    # Absolute shape keys can't be mixed: the basis stands for the mesh, and every key moves along.
    scene = fresh()
    model = scene["vrc"]
    model.data.shape_keys.use_relative = False
    check(np.allclose(lineup.mesh_positions(model), fixture.world_positions(model)),
          "absolute keys: the mesh is where its basis is")
    fixture.select([model])
    result = bpy.ops.magic_fit.line_up()
    errors = fixture.marker_errors(model, scene["vrc_markers"], scene["body_joints"])
    check(result == {'FINISHED'} and max(errors.values()) < 1e-5,
          "absolute keys: lined up, every joint on the body's ({:s} at most)".format(mm(max(errors.values()))))


def test_no_stretch_operator():
    scene = fresh()
    bpy.context.scene.body_fit_brush.lineup_stretch = False
    model = scene["vrc"]
    fixture.select([model])
    bpy.ops.magic_fit.line_up()
    errors = fixture.marker_errors(model, scene["vrc_markers"], scene["body_joints"])
    co = fixture.world_positions(model)
    markers = {joint: co[v] for v, joint in scene["vrc_markers"].items()}
    hips = (markers["hip_l"] + markers["hip_r"]) / 2.0
    body = scene["body_joints"]
    check(np.linalg.norm(hips - (body["hip_l"] + body["hip_r"]) / 2.0) < 1e-5,
          "no stretch: the hips land on the body's")
    # The model's legs are longer: its knees end up lower than the body's, on the line of its thighs.
    thigh = markers["knee_l"] - markers["hip_l"]
    wanted = body["knee_l"] - body["hip_l"]
    cosine = np.dot(thigh, wanted) / np.linalg.norm(thigh) / np.linalg.norm(wanted)
    check(cosine > 0.9999 and errors["knee_l"] > 0.01,
          "no stretch: the thighs point the body's way, the knees are {:s} off".format(mm(errors["knee_l"])))
    check(bpy.context.scene.body_fit_brush.lineup_stretch is False, "no stretch: the setting stays off")


def test_model_without_armature():
    scene = fresh()
    body_mesh, arms = scene["bipeds"]
    fixture.select([arms])
    meshes, armatures = operators.lineup_model(bpy.context)
    check(set(meshes) == {body_mesh, arms} and not armatures,
          "no armature: both meshes that lost the same armature come along")
    result = bpy.ops.magic_fit.line_up()
    check(result == {'FINISHED'}, "no armature: lined up")
    errors = {}
    for obj, markers in zip(scene["bipeds"], scene["biped_markers"]):
        errors.update(fixture.marker_errors(obj, markers, scene["body_joints"]))
    check(max(errors.values()) < 0.006, "no armature: joints from the weights land within {:s}".format(
        mm(max(errors.values()))))
    heights = fixture.world_positions(body_mesh)[:, 2]
    check(heights.max() < 1.8 and heights.min() > -0.2, "no armature: turned upright and in place")


def test_unnamed_groups():
    scene = fresh()
    model, armature = scene["anon"], scene["anon_armature"]
    names = [group.name for group in model.vertex_groups]
    weights = lineup.mesh_weights(model, names)
    target, _armature = lineup.body_rig(scene["body"], bpy.context.evaluated_depsgraph_get())
    parts = lineup.label_anonymous(lineup.mesh_positions(model), weights, target)
    wrong = {name: (parts[name], want) for name, want in scene["anon_truth"].items() if parts[name] != want}
    check(not wrong, "unnamed: {:d} groups sorted into their parts ({})".format(len(names), wrong or "all right"))
    fixture.select([model])
    result = bpy.ops.magic_fit.line_up()
    errors = fixture.marker_errors(model, scene["anon_markers"], scene["body_joints"])
    check(result == {'FINISHED'} and max(errors.values()) < 0.006,
          "unnamed: joints land within {:s}".format(mm(max(errors.values()))))
    # Its bones, all at the origin, are put where their groups are: the knee's bone starts at the knee.
    bones = bpy.data.objects["Anon Armature"].data.bones
    co = fixture.world_positions(model)
    knee = next(v for v, joint in scene["anon_markers"].items() if joint == "knee_l")
    group = model.vertex_groups[model.data.vertices[knee].groups[0].group].name
    off = np.linalg.norm(np.array(bones[group].head_local) - co[knee])
    heads = np.array([bone.head_local for bone in bones])
    shown = fixture.displayed_positions(model)
    check(np.ptp(heads, axis=0).max() > 0.5 and off < 0.01 and np.abs(shown - co).max() < 1e-5,
          "unnamed: bones all at one spot are put where their groups are (knee {:s} off), the mesh stays".format(
              mm(off)))


def test_unnamed_bones_in_place():
    # Bones where they belong, but named so they say nothing: the rest pose moves along with their groups.
    scene = fresh()
    model, armature = scene["vrc"], scene["vrc_armature"]
    original = {bone.name: "bone_{:02d}".format(k) for k, bone in enumerate(armature.data.bones)}
    for bone in armature.data.bones:
        bone.name = original[bone.name]  # renames the vertex groups too
    fixture.select([model])
    result = bpy.ops.magic_fit.line_up()
    heads = {name: np.array(armature.matrix_world @ armature.data.bones[anon].head_local)
             for name, anon in original.items()}
    body = scene["body_joints"]
    errors = [np.linalg.norm(heads[name] - body[joint]) for name, joint in
              (("Knee_L", "knee_l"), ("Elbow_R", "elbow_r"), ("Wrist_L", "wrist_l"), ("Foot_R", "ankle_r"))]
    still = np.abs(fixture.displayed_positions(model) - fixture.world_positions(model)).max() < 1e-5
    check(result == {'FINISHED'} and max(errors) < 0.006 and still,
          "unnamed bones in place: the rest pose moves along ({:s} off at most), the mesh stays".format(
              mm(max(errors))))


def test_unnamed_mirrored():
    # The same model facing +X, arms hanging: left and right come from where the feet point.
    fixture.clear()
    body, _skeleton, body_joints = fixture.build_body()
    joints = fixture.model_joints(scale=1.0, arm_angle=60.0, turn=90.0)
    vertices, faces, weights, markers = fixture.build_humanoid(joints, fixture.VRC_BONES)
    real = sorted({g for w in weights for g in w})
    rename = {name: "bone_{:02d}".format(k) for k, name in enumerate(real)}
    model = fixture.mesh_object("Turned", vertices, faces, [{rename[g]: v for g, v in w.items()} for w in weights])
    names = [group.name for group in model.vertex_groups]
    target = skeleton(body_joints)
    parts = lineup.label_anonymous(lineup.mesh_positions(model), lineup.mesh_weights(model, names), target)
    truth = lineup.bone_parts(real, {name: None for name in real})
    wrong = {rename[name]: (parts[rename[name]], truth[name]) for name in real if parts[rename[name]] != truth[name]}
    check(not wrong, "unnamed: a model facing +X with its arms down sorted too ({})".format(wrong or "all right"))


def test_fingers_line_up():
    # A VRChat-style model whose fingers curl 35 degrees at every joint, and a body with straight fingers:
    # lined up, each of the model's fingers lies straight along the body's, and its finger bones with it.
    fixture.clear()
    body, _skeleton, body_joints = fixture.build_body(fingers=True)
    model, armature, _joints, markers = fixture.build_vrc(fingers=35.0)
    bpy.context.scene.magic_fit.target = body
    bpy.context.scene.body_fit_brush.lineup_stretch = True
    fixture.select([model])
    result = bpy.ops.magic_fit.line_up()
    co = fixture.world_positions(model)
    at = {name: co[vertex] for vertex, name in markers.items()}
    matrix = np.array(armature.matrix_world)
    worst, worst_bone = 0.0, 0.0
    for side in lineup.SIDES:
        for finger in fixture.FINGERS:
            straight = fixture.finger_chains(body_joints, side)[finger]
            wanted = (straight[1] - straight[0]) / np.linalg.norm(straight[1] - straight[0])
            points = [at["{:s}{:s}_{:s}".format(finger, k, side)] for k in ("1", "2", "3", "tip")]
            for a, b in zip(points, points[1:]):
                worst = max(worst, math.degrees(math.acos(min(1.0, np.dot(b - a, wanted) / np.linalg.norm(b - a)))))
            bone = armature.data.bones[fixture.finger_bone(finger, 2, side)]
            direction = matrix[:3, :3] @ (np.array(bone.tail_local) - np.array(bone.head_local))
            worst_bone = max(worst_bone, math.degrees(math.acos(min(1.0, np.dot(direction, wanted) /
                                                                     np.linalg.norm(direction)))))
    check(result == {'FINISHED'} and worst < 0.5 and worst_bone < 0.5,
          "fingers: curled 35 degrees, lined up straight along the body's ({:.2f} deg off), bones too ({:.2f} "
          "deg)".format(worst, worst_bone))


def test_unnamed_fingers():
    # Fingers on groups that say nothing: each branch off the hand is a finger, the thumb the one starting
    # nearest the wrist, the others named across the knuckles from it.
    fixture.clear()
    body, _skeleton, _body_joints = fixture.build_body()
    model, _armature, _joints, _markers, truth = fixture.build_anon(fingers=20.0)
    names = [group.name for group in model.vertex_groups]
    target, _armature = lineup.body_rig(body, bpy.context.evaluated_depsgraph_get())
    parts = lineup.label_anonymous(lineup.mesh_positions(model), lineup.mesh_weights(model, names), target)
    wrong = {name: (parts[name], want) for name, want in truth.items() if parts[name] != want}
    fingers = sum(part in lineup.DIGIT_PARTS for part in truth.values())
    check(not wrong and fingers == 30, "unnamed fingers: {:d} groups sorted, {:d} of them fingers' ({})".format(
        len(names), fingers, wrong or "all right"))


def test_unnamed_skirt():
    # A long skirt over the legs, weighted to both thighs and more to the left one: the legs stay two, the
    # pelvis stays in the middle, and the model still lines up.
    fixture.clear()
    body, _skeleton, body_joints = fixture.build_body()
    model, _armature, _joints, markers, truth = fixture.build_anon(skirt_bias=0.3)
    names = [group.name for group in model.vertex_groups]
    target, _armature = lineup.body_rig(body, bpy.context.evaluated_depsgraph_get())
    parts = lineup.label_anonymous(lineup.mesh_positions(model), lineup.mesh_weights(model, names), target)
    wrong = {name: (parts[name], want) for name, want in truth.items() if parts[name] != want}
    check(not wrong, "skirt: {:d} groups sorted into their parts ({})".format(len(names), wrong or "all right"))
    bpy.context.scene.magic_fit.target = body
    bpy.context.scene.body_fit_brush.lineup_stretch = True
    fixture.select([model])
    result = bpy.ops.magic_fit.line_up()
    errors = fixture.marker_errors(model, markers, body_joints)
    # The skirt's weights, blending from the hips' bone to the thighs', pull the hips' joints a little.
    check(result == {'FINISHED'} and max(errors.values()) < 0.03,
          "skirt: joints land within {:s}".format(mm(max(errors.values()))))


SIMS_NAMES = {"Hips": "b__Pelvis__", "Spine": "b__Spine0__", "Chest": "b__Spine2__", "Neck": "b__Neck__",
              "Head": "b__Head__", "Leg": "b__{S}_Thigh__", "Knee": "b__{S}_Calf__", "Shoulder": "b__{S}_Clavicle__",
              "Arm": "b__{S}_UpperArm__", "Elbow": "b__{S}_ForeArm__", "Wrist": "b__{S}_Hand__",
              "Thumb": "b__{S}_Thumb{K}__", "Index": "b__{S}_Index{K}__", "Middle": "b__{S}_Mid{K}__",
              "Ring": "b__{S}_Ring{K}__", "Little": "b__{S}_Pinky{K}__"}


def sims_hash_name(name):
    """The group name a Sims export gives the bone of the fixture's VRChat-style bone ``name``: "bone_" and
    the FNV-1 hash of the Sims bone's name."""
    base, side = (name[:-2], name[-1]) if name[-2:] in ("_L", "_R") else (name, "")
    digit = base[-1] if base[-1].isdigit() else ""
    sims = SIMS_NAMES[base.rstrip("0123456789")].replace("{S}", side)
    sims = sims.replace("{K}", str(int(digit) - 1)) if digit else sims
    return "bone_{:08X}".format(lineup._fnv1(sims.lower()))


def test_sims_hashed_names():
    # Groups named after the hashes of The Sims' bone names, on bones all at the origin that don't hang from each
    # other: sorted by the names the hashes stand for, the fingers' segments numbered by where they lie.
    check(lineup.classify(sims_hash_name("Arm_L")) == ("upperarm", "l") and
          lineup.classify(sims_hash_name("Knee_R")) == ("shin", "r") and
          lineup.finger_of(sims_hash_name("Index2_L")) == "index" and
          lineup.classify("bone_0DE2936B")[0] == lineup.TORSO and lineup.classify("bone_12345678") == (None, None),
          "Sims hashes: known bone names read, breasts and unknown hashes aside")
    fixture.clear()
    body, _skeleton, body_joints = fixture.build_body()
    joints = fixture.model_joints(**fixture.ANON_JOINTS)
    vertices, faces, weights, markers = fixture.build_humanoid(joints, fixture.VRC_BONES, fingers=20.0)
    hashed = [{sims_hash_name(group): value for group, value in weight.items()} for weight in weights]
    model = fixture.mesh_object("Sims Body", vertices, faces, hashed)
    names = sorted({group for weight in hashed for group in weight})
    armature = fixture.armature_object("Sims_Armature", joints, {}, [(n, None, (0, 0, 0), (0, 0, 1)) for n in names],
                                       at_origin=True)
    model.modifiers.new("Armature", 'ARMATURE').object = armature
    model.parent = armature
    bpy.context.scene.magic_fit.target = body
    bpy.context.scene.body_fit_brush.lineup_stretch = True
    result = lineup.plan([model], armature, body, bpy.context.evaluated_depsgraph_get())
    segments = sum(part in result.source_joints for part in lineup.DIGIT_PARTS)
    check(result.how == "weights" and segments == 30,
          "Sims hashes: joints from the weights of named groups ({:s}), all {:d} finger segments".format(
              result.how, segments))
    fixture.select([model])
    outcome = bpy.ops.magic_fit.line_up()
    errors = fixture.marker_errors(model, markers, body_joints)
    check(outcome == {'FINISHED'} and max(errors.values()) < 0.006,
          "Sims hashes: joints land within {:s}".format(mm(max(errors.values()))))


def test_half_shapes():
    # Half a body must be shaped like one: a bottom reaches up past its hips, a top doesn't hang far below its
    # shoulders. A skirt piece starting below the hips, or a bodice whose hanging bits looked like legs, isn't.
    body = skeleton(fixture.mirror(fixture.BODY_JOINTS))
    model = skeleton(fixture.model_joints(scale=1.2, feet=False))
    legs = {part: joint for part, joint in model.items() if part.startswith(("thigh", "shin"))}
    solution = lineup.solve(legs, body)
    hips = (legs["thigh_l"] + legs["thigh_r"]) / 2.0
    reach = solution.torso_length / solution.scale  # the body's torso length, in the model's size
    bottom = np.array(((0.0, 0.0, hips[2] + 0.3 * reach), (0.0, 0.0, 0.1)))
    piece = np.array(((0.0, 0.0, hips[2] - 0.2 * reach), (0.0, 0.0, 0.1)))
    gown = np.array(((0.0, 0.0, hips[2] + 1.2 * reach), (0.0, 0.0, 0.1)))
    headed = np.array(((0.0, 0.0, hips[2] + 1.7 * reach), (0.0, 0.0, 0.1)))
    check(solution.half == "lower" and lineup._half_fits(solution, legs, body, bottom) and
          lineup._half_fits(solution, legs, body, gown) and not lineup._half_fits(solution, legs, body, piece) and
          not lineup._half_fits(solution, legs, body, headed),
          "half shapes: a bottom up to its waist or collar fits, a piece below its hips or up to a head doesn't")
    arms = {part: joint for part, joint in model.items() if not part.startswith(("thigh", "shin"))}
    solution = lineup.solve(arms, body)
    top = (arms["upperarm_l"] + arms["upperarm_r"]) / 2.0
    shirt = np.array(((0.0, 0.0, top[2] - 0.9 * reach), (0.0, 0.0, top[2] + 0.2)))
    gown = np.array(((0.0, 0.0, top[2] - 2.0 * reach), (0.0, 0.0, top[2] + 0.2)))
    check(solution.half == "upper" and lineup._half_fits(solution, arms, body, shirt) and
          not lineup._half_fits(solution, arms, body, gown),
          "half shapes: a top down to its hips fits, one hanging two torsos below its shoulders doesn't")


def test_arm_cuts():
    # An arm's groups hand over at these distances along it (the last at the wrist): a sleeve weighted to the
    # shoulder's first group pulls the second handover in, so the clavicle's share of the arm no longer tells
    # which ends it. The elbow and the upper arm's length for the forearm still do.
    body = np.array((0.0, 0.107, 0.366, 0.594))  # the body's: upper arm, forearm and hand joints
    for along, name in (((0.0, 0.113, 0.324, 0.595, 0.815, 1.0), "painted like the body"),
                        ((0.0, 0.101, 0.26, 0.583, 0.81, 1.0), "with a sleeve on the shoulder")):
        cuts = lineup._arm_cuts(np.array(along), body)
        check(cuts == [1, 3], "arm cuts {:s}: shoulder and elbow where they belong ({})".format(name, cuts))


def test_mirror_axis():
    # Two legs close together and a seat: the points spread more front to back than across, but the model is
    # mirrored across.
    rng = np.random.default_rng(1)
    points = []
    for x in (-0.04, 0.04):
        angle = rng.uniform(0.0, 2.0 * math.pi, 3000)
        points.append(np.column_stack((x + 0.06 * np.cos(angle), 0.06 * np.sin(angle), rng.uniform(0.1, 0.9, 3000))))
    seat = rng.uniform(-1.0, 1.0, (3000, 2))
    points.append(np.column_stack((0.1 * seat[:, 0], 0.06 + 0.2 * (1.0 - seat[:, 0] ** 2) * (seat[:, 1] + 1.0) / 2.0,
                                   rng.uniform(0.8, 1.0, 3000))))
    points = np.concatenate(points)
    flat = points[:, :2] - points[:, :2].mean(axis=0)
    widest = np.linalg.eigh(np.cov(flat.T))[1][:, -1]
    axis, midline = lineup._mirror_axis(points)
    check(abs(widest[1]) > abs(widest[0]) and abs(axis[0]) > 0.99 and abs(midline) < 0.01,
          "mirror axis: across the legs ({}), though they spread most front to back".format(np.round(axis, 3)))


def test_half_bodies():
    # A top (no legs) and a bottom (no arms) with groups that say nothing, and a top on bones named like
    # VRChat's: each lines up by the end it has, sized by its arms or legs.
    fixture.clear()
    body, _skeleton, body_joints = fixture.build_body()
    bpy.context.scene.magic_fit.target = body
    bpy.context.scene.body_fit_brush.lineup_stretch = True
    target, _armature = lineup.body_rig(body, bpy.context.evaluated_depsgraph_get())
    for half in ("upper", "lower"):
        model, armature, _joints, markers, truth = fixture.build_anon(half=half)
        names = [group.name for group in model.vertex_groups]
        parts = lineup.label_anonymous(lineup.mesh_positions(model), lineup.mesh_weights(model, names), target)
        wrong = {name: (parts[name], want) for name, want in truth.items() if parts[name] != want}
        fixture.select([model])
        result = bpy.ops.magic_fit.line_up()
        errors = fixture.marker_errors(model, markers, body_joints)
        check(not wrong and result == {'FINISHED'} and max(errors.values()) < 0.008,
              "half body, {:s}, unnamed: sorted ({}), joints land within {:s}".format(
                  half, wrong or "all right", mm(max(errors.values()))))
        for obj in (model, armature):
            bpy.data.objects.remove(obj)
    joints = fixture.model_joints(**fixture.VRC_JOINTS)
    vertices, faces, weights, markers = fixture.build_humanoid(joints, fixture.VRC_BONES)
    vertices, faces, weights, markers = fixture.half_body(vertices, faces, weights, markers, True)
    model = fixture.mesh_object("Top", vertices, faces, weights)
    bones = [bone for bone in fixture.bone_list(joints, fixture.VRC_BONES)
             if not bone[0].startswith(("Leg", "Knee", "Foot", "Toe"))]
    armature = fixture.armature_object("Top Armature", joints, fixture.VRC_BONES, bones)
    model.modifiers.new("Armature", 'ARMATURE').object = armature
    model.parent = armature
    fixture.select([model])
    result = bpy.ops.magic_fit.line_up()
    errors = fixture.marker_errors(model, markers, body_joints)
    check(result == {'FINISHED'} and max(errors.values()) < 1e-4,
          "half body, a top on named bones: its joints on the body's ({:s} at most)".format(mm(max(errors.values()))))
    _parts, _deforming, rig = lineup.armature_rig(armature)
    check(lineup.solve(rig, target, stretch=False).half == "upper", "half body: the solution says it's an upper body")


def test_refine_joints():
    # Joints estimated a few centimetres off, sideways from the limbs: the body's shape puts them back.
    scene = fresh()
    meshes = scene["bipeds"]
    names = sorted({group.name for mesh in meshes for group in mesh.vertex_groups})
    weights = lineup.Weights.stack(names, [lineup.mesh_weights(mesh, names) for mesh in meshes])
    points = np.concatenate([lineup.mesh_positions(mesh) for mesh in meshes])
    tris, first = [], 0
    for mesh in meshes:
        tris.append(lineup.mesh_tris(mesh) + first)
        first += len(mesh.data.vertices)
    parts, _named = lineup.group_parts_by_name(names, weights)
    part_weights = weights.by_part(parts)
    exact = lineup.joints_from_weights(points, part_weights)
    target, _armature = lineup.body_rig(scene["body"], bpy.context.evaluated_depsgraph_get())
    off = dict(exact)
    for part, shift in (("thigh_l", (0.03, 0.0, 0.0)), ("thigh_r", (0.03, 0.0, 0.0)),
                        ("upperarm_l", (-0.025, 0.0, 0.0))):
        off[part] = exact[part] + np.array(shift)  # the model faces +X: this is forward / back
    refined, count = lineup.refine_joints(off, target, points, np.concatenate(tris), part_weights,
                                          lineup.body_surface(scene["body"], bpy.context.evaluated_depsgraph_get()))
    before = max(np.linalg.norm(off[p] - exact[p]) for p in ("thigh_l", "thigh_r", "upperarm_l"))
    after = max(np.linalg.norm(refined[p] - exact[p]) for p in ("thigh_l", "thigh_r", "upperarm_l"))
    check(after < 0.4 * before and count >= 3,
          "refine: joints {:s} off move back to within {:s}".format(mm(before), mm(after)))


def test_readiness():
    scene = fresh()
    settings = bpy.context.scene.magic_fit
    fixture.select([])
    check("Select the model" in operators.check_lineup_ready(bpy.context), "ready: something must be selected")
    fixture.select([scene["body"]])
    check("Select the model" in operators.check_lineup_ready(bpy.context), "ready: the body isn't a model")
    fixture.select([scene["vrc"], scene["anon"]])
    check("one model at a time" in operators.check_lineup_ready(bpy.context), "ready: one armature at a time")
    fixture.select([scene["vrc"]])
    bone = bpy.data.objects["VRC Armature"].pose.bones["Arm_L"]
    bone.rotation_mode = 'XYZ'
    bone.rotation_euler = (0.0, 0.0, 0.5)
    check("posed" in operators.check_lineup_ready(bpy.context), "ready: refuses a posed armature")
    bone.rotation_euler = (0.0, 0.0, 0.0)
    settings.target = None
    check("Pick the body" in operators.check_lineup_ready(bpy.context), "ready: needs the body")
    settings.target = scene["body"]
    fixture.select([scene["vrc"]])
    bpy.ops.object.mode_set(mode='EDIT')
    check("Object Mode" in operators.check_lineup_ready(bpy.context), "ready: Object Mode only")
    bpy.ops.object.mode_set(mode='OBJECT')
    check(operators.check_lineup_ready(bpy.context) is None, "ready: with a model and the body")
    check(operators.lineup_summary(bpy.context) == "'VRC Suit' of 'VRC Armature'", "ready: summary names the model")


def test_body_without_armature():
    scene = fresh()
    body = scene["body"]
    body.modifiers.clear()
    fixture.select([scene["vrc"]])
    result = bpy.ops.magic_fit.line_up()
    errors = fixture.marker_errors(scene["vrc"], scene["vrc_markers"], scene["body_joints"])
    check(result == {'FINISHED'} and max(errors.values()) < 0.006,
          "body: without an armature, its joints come from its weights ({:s})".format(mm(max(errors.values()))))


def test_registered():
    check(hasattr(bpy.ops.magic_fit, "line_up"), "the Line Up operator is registered")
    properties = bpy.context.scene.body_fit_brush.bl_rna.properties
    check("lineup_stretch" in properties and properties["lineup_stretch"].default is False,
          "setting: lineup_stretch, off by default")
    panel = getattr(bpy.types, "VIEW3D_PT_line_up_sidebar", None)
    check(panel is not None and panel.tab == 'LINE_UP', "panel: on the Line Up tab")


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
