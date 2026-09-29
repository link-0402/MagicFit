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
        worst = max(worst, np.linalg.norm(lineup.apply_matrix(solution.transforms[part], joint) - body[part]))
        after = lineup.NEXT.get(part)
        if after in model:
            end = lineup.apply_matrix(solution.transforms[part], model[after])
            worst = max(worst, np.linalg.norm(end - body[after]))
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
        if part not in model or after not in model:
            continue
        matrix = solution.transforms[part]
        start, end = (lineup.apply_matrix(matrix, model[p]) for p in (part, after))
        wanted = body[after] - body[part]
        cosine = np.dot(end - start, wanted) / np.linalg.norm(end - start) / np.linalg.norm(wanted)
        worst_angle = max(worst_angle, math.degrees(math.acos(min(1.0, cosine))))
        worst_length = max(worst_length, abs(np.linalg.norm(end - start) -
                                             solution.scale * np.linalg.norm(model[after] - model[part])))
        # Segments stay joined: each starts where the one above put its joint.
        parent = lineup.PARENT[part]
        if parent != "torso":
            worst_gap = max(worst_gap, np.linalg.norm(start - lineup.apply_matrix(solution.transforms[parent],
                                                                                   model[part])))
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
    try:
        lineup.solve(model, body)
        check(False, "missing: refuses a model without hips")
    except lineup.LineUpError as error:
        check("hips" in str(error), "missing: refuses a model without hips ({})".format(error))


def test_deform():
    body = skeleton(fixture.mirror(fixture.BODY_JOINTS))
    model = skeleton(fixture.model_joints(scale=0.9, arm_angle=10.0))
    solution = lineup.solve(model, body, stretch=False)
    rng = np.random.default_rng(4)
    points = rng.random((20, 3))
    weights = np.zeros((20, len(lineup.PARTS)))
    weights[:10, lineup.PART_INDEX["forearm_l"]] = 1.0
    moved = lineup.deform(points, weights, solution.transforms)
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
    heads = np.array([bone.head_local for bone in bpy.data.objects["Anon Armature"].data.bones])
    check(np.abs(heads).max() < 1e-9, "unnamed: an armature without joints of its own is left as it was")


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
