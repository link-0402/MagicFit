"""Headless tests for Customize+ (the Customize+ panel: templates shown on an armature like in the game).

Run from the repository root:
    blender -b --factory-startup --python tests/test_cplus_core.py
"""

import base64
import gzip
import json
import math
import os
import struct
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(ROOT, "deps")
if os.path.isdir(DEPS):
    sys.path.insert(0, DEPS)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import cplus as fixture  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import gamefiles, operators, straighten  # noqa: E402
from magic_fit.cplus import operators as cplus_operators, reference, rig, solver  # noqa: E402
from magic_fit.cplus.template import BoneEdit, Template, TemplateError, parse  # noqa: E402

FAILURES = []

SUPERTHICC = fixture.SUPERTHICC


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def encode(version, data):
    """Text the way Customize+ copies it: base64 of gzip(version byte + JSON)."""
    body = json.dumps(data).encode("utf-8") if not isinstance(data, bytes) else data
    return base64.b64encode(gzip.compress(bytes([version]) + body)).decode("ascii")


def bone_json(translation=(0, 0, 0), rotation=(0, 0, 0), scaling=(1, 1, 1), **flags):
    data = {
        "PropagateTranslation": False, "PropagateRotation": False, "PropagateScale": False,
        "ChildScalingIndependent": False,
        "Translation": dict(zip("XYZ", translation)), "Rotation": dict(zip("XYZ", rotation)),
        "Scaling": dict(zip("XYZ", scaling)),
    }
    data.update(flags)
    return data


def template_json(bones, version=7, name="Test"):
    data = {"Version": version, "UniqueId": "00000000-0000-0000-0000-000000000000",
            "CreationDate": "2026-09-26T00:00:00+00:00", "ModifiedDate": "2026-09-26T00:00:00+00:00",
            "Name": name, "Bones": bones, "IsWriteProtected": False}
    if version >= 7:
        data["Source"] = 0
    return data


def make_template(bones, name="Test"):
    return parse(json.dumps(template_json(bones, name=name)))


def reset_scene():
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for collection in (bpy.data.meshes, bpy.data.armatures, bpy.data.actions):
        for item in list(collection):
            collection.remove(item)
    settings = bpy.context.scene.magic_fit_cplus
    settings.scope = 'ALL'
    settings.armature = None
    settings.template = ""
    settings.use_scale = settings.use_rotation = settings.use_translation = True
    settings.use_bust = False
    settings.bust_size = 50
    bpy.context.scene.magic_fit.target = None
    bpy.context.scene.frame_set(1)


def apply(template_text, armature=None, bust=None, **options):
    """Apply ``template_text`` (and Bust Size ``bust``, or None)."""
    settings = bpy.context.scene.magic_fit_cplus
    settings.template = template_text
    settings.armature = armature
    for key in ("use_scale", "use_rotation", "use_translation"):
        setattr(settings, key, options.get(key, True))
    if bust is not None:
        settings.bust_size = bust
    settings.use_bust = bust is not None
    return bpy.ops.magic_fit.cplus_apply()


def rig_error(armature, template, options=None, bust=None, lalafell=False):
    """Largest difference between the rig's evaluated bones and what the game does (armature units)."""
    game = fixture.game_matrices(armature, template, options, bust, lalafell)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    posed = rig.rig_of(armature).evaluated_get(depsgraph).pose.bones
    return max(float(np.abs(np.array(posed[name].matrix) - matrix).max()) for name, matrix in game.items())


def mesh_error(obj, armature, template, options=None, bust=None, lalafell=False):
    """Largest distance (world space, metres) between ``obj`` as shown and bent by the game's matrices."""
    game = fixture.game_matrices(armature, template, options, bust, lalafell)
    expected = fixture.skin(obj, armature, game)
    shown = fixture.evaluated_coordinates(obj)
    world = np.array(obj.matrix_world)[:3, :3]
    return float(np.linalg.norm((shown - expected) @ world.T, axis=1).max())


# -----------------------------------------------------------------------------
# Customize+'s own code, ported line by line (Customize+ 2.2.1: BoneTransform.ModifyExisting*,
# ModelBone.ApplyModelTransform and PropagateChildren), with System.Numerics' conventions: quaternions
# (x, y, z, w), Matrix4x4 with row vectors (p' = p M). An independent check of `solver.simulate`.

def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def q_ypr(yaw, pitch, roll):
    """Quaternion.CreateFromYawPitchRoll."""
    sr, cr = math.sin(roll * 0.5), math.cos(roll * 0.5)
    sp, cp = math.sin(pitch * 0.5), math.cos(pitch * 0.5)
    sy, cy = math.sin(yaw * 0.5), math.cos(yaw * 0.5)
    return np.array([cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr,
                     cy * cp * sr - sy * sp * cr, cy * cp * cr + sy * sp * sr])


def q_divide(a, b):
    """Quaternion division: a * inverse(b)."""
    inverse = np.array([-b[0], -b[1], -b[2], b[3]]) / float(np.dot(b, b))
    return q_mul(a, inverse)


def q_rotate(v, q):
    """Vector3.Transform(v, q)."""
    p = np.array([v[0], v[1], v[2], 0.0])
    conjugate = np.array([-q[0], -q[1], -q[2], q[3]])
    return q_mul(q_mul(q, p), conjugate)[:3]


def q_row_matrix(q):
    """Matrix4x4.CreateFromQuaternion (row vectors)."""
    x, y, z, w = q
    M = np.identity(4)
    M[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
        [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
        [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
    ])
    return M


class Qs:
    """hkQsTransformf."""

    def __init__(self, t, r, s):
        self.t, self.r, self.s = np.array(t, float), np.array(r, float), np.array(s, float)

    @classmethod
    def from_matrix(cls, M):
        """From a column-vector 4x4 (position, rotation, scale; no shear)."""
        t, R, s = solver.decompose(M)
        w, x, y, z = solver.quaternion(R)
        return cls(t, (x, y, z, w), s)

    def matrix(self):
        x, y, z, w = self.r
        return solver.compose(self.t, solver.quaternion_matrix((w, x, y, z)), self.s)

    def row_matrix(self):
        """get4x4ColumnMajor read as a System.Numerics Matrix4x4."""
        return self.matrix().T

    def set_row_matrix(self, M):
        """hkQsTransformf::set: position, turn and scale of the matrix (shear dropped, like `decompose`)."""
        other = Qs.from_matrix(M.T)
        self.t, self.r, self.s = other.t, other.r, other.s


def port_apply(order, children, transforms, edits):
    """ModelBone.ApplyModelTransform for each bone in order ({name: Qs}, edited in place)."""
    def descendants(name):
        found = list(children.get(name, []))
        i = 0
        while i < len(found):
            found.extend(children.get(found[i], []))
            i += 1
        return found

    def modify(tr, edit):
        tr.s = tr.s * np.array(edit.scaling)                                    # ModifyExistingScale
        tr.r = q_mul(tr.r, q_ypr(*np.radians(edit.rotation)))                   # ModifyExistingRotation
        tr.t = tr.t + q_rotate(np.array(edit.translation), tr.r)                # ...TranslationWithRotation

    for name in order:
        edit = edits.get(name)
        if name == "n_root" or edit is None or not edit.is_edited():
            continue
        tr = transforms[name]
        if not edit.propagates:
            modify(tr, edit)
            continue
        initial_pos, initial_rot, initial_scale = tr.t.copy(), tr.r.copy(), tr.s.copy()
        modify(tr, edit)
        child_scale = tr.s.copy()
        if edit.child_scaling_independent:
            child_scale = initial_scale * np.array(edit.child_scaling)
        should_scale = edit.propagate_scale and (
            tuple(edit.scaling) != (1.0, 1.0, 1.0)
            or (edit.child_scaling_independent and tuple(edit.child_scaling) != (1.0, 1.0, 1.0)))
        propagate_rotation = edit.propagate_rotation and tuple(edit.rotation) != (0.0, 0.0, 0.0)
        # PropagateChildren
        source = tr.t.copy()
        delta_rot = q_divide(tr.r, initial_rot)
        delta_pos = source - initial_pos
        delta_scale = child_scale / initial_scale
        for child in descendants(name):
            access = transforms[child]
            offset = access.t - source
            matrix = access.row_matrix()
            if should_scale:
                scale = np.diag([delta_scale[0], delta_scale[1], delta_scale[2], 1.0])
                matrix = matrix @ scale
                offset = offset * delta_scale
            if propagate_rotation:
                matrix = matrix @ q_row_matrix(delta_rot)
                offset = q_rotate(offset, delta_rot)
            matrix[3, :3] = delta_pos + source + offset
            access.set_row_matrix(matrix)


def random_model(rng, degrees=20.0):
    """The fixture skeleton in the game's model space, each bone turned a little (an animation)."""
    model = {}
    for name in fixture.ORDER:
        position, rotation = fixture.game_rest(name)
        turn = solver.cplus_rotation(rng.uniform(-degrees, degrees, 3))
        model[name] = solver.compose(position + rng.uniform(-0.01, 0.01, 3), rotation @ turn, 1.0)
    return model


def random_edits(rng, count=14, propagate=True):
    edits = {}
    for name in rng.choice(fixture.ORDER, size=count, replace=False):
        flags = rng.random(4) < (0.5 if propagate else 0.0)
        edits[str(name)] = BoneEdit(
            translation=rng.uniform(-0.03, 0.03, 3), rotation=rng.uniform(-40.0, 40.0, 3),
            scaling=rng.uniform(0.7, 1.5, 3), child_scaling=rng.uniform(0.8, 1.3, 3),
            child_scaling_independent=bool(flags[0]), propagate_translation=bool(flags[1]),
            propagate_rotation=bool(flags[2]), propagate_scale=bool(flags[3]))
    return edits


# -----------------------------------------------------------------------------
# Templates

def test_parse_version_7():
    template = parse(SUPERTHICC)
    check(template.name == "Custom - Superthicc+" and template.version == 7,
          "version 7 template: name and version read ({!r}, {})".format(template.name, template.version))
    check(len(template.bones) == 28 and len(template.edited()) == 28, "version 7 template: 28 bones")
    mune = template.bones["j_mune_l"]
    check(mune.scaling == (1.27, 1.65, 1.69) and mune.translation == (0.0, 0.0, 0.0), "j_mune_l's scaling")
    thigh = template.bones["j_asi_a_l"]
    check(thigh.translation == (0.01, 0.0, 0.01) and not thigh.propagates, "j_asi_a_l's position")
    broken = SUPERTHICC[:40] + "\n" + SUPERTHICC[40:200] + " \r\n" + SUPERTHICC[200:]
    check(parse(broken).name == template.name, "line breaks and spaces in the pasted text are ignored")


def test_parse_other_formats():
    bones = {"j_kosi": bone_json(scaling=(1.0, 1.2, 1.1)), "j_kao": bone_json(rotation=(10.0, 0.0, 0.0))}
    for version in (4, 5, 6):
        template = parse(encode(version, template_json(bones, version=version)))
        check(template.version == version and set(template.edited()) == {"j_kosi", "j_kao"},
              "version {:d} template".format(version))
    template = parse(json.dumps(template_json(bones)))
    check(template.version == 7 and template.bones["j_kosi"].scaling == (1.0, 1.2, 1.1), "template JSON (a file)")
    newer = parse(encode(8, template_json(bones, version=8)))
    check(newer.version == 8 and len(newer.edited()) == 2, "a newer version is read the same way")
    copied = [{"BoneCodeName": "j_kosi", "Translation": {"X": 0.01, "Y": 0, "Z": 0},
               "Rotation": {"X": 0, "Y": 0, "Z": 0}, "Scaling": {"X": 1, "Y": 1, "Z": 1},
               "ChildScaling": {"X": 1, "Y": 1, "Z": 1}, "ChildScalingIndependent": False,
               "PropagateTranslation": True, "PropagateRotation": False, "PropagateScale": False}]
    template = parse(encode(1, copied))
    check(template.bones["j_kosi"].translation == (0.01, 0.0, 0.0) and template.bones["j_kosi"].propagate_translation,
          "bones copied from the template editor")
    v3 = {"CharacterName": "A", "ProfileName": "Old", "ConfigVersion": 3,
          "Bones": {"j_kosi": {"Translation": {"X": 0, "Y": 0, "Z": 0}, "Rotation": {"X": 0, "Y": 0, "Z": 0},
                               "Scaling": {"X": 1.3, "Y": 1, "Z": 1}}}}
    template = parse(encode(3, v3))
    check(template.name == "Old" and template.bones["j_kosi"].scaling == (1.3, 1.0, 1.0), "version 3 profile")
    v2 = {"CharacterName": "A", "ScaleName": "Older", "BodyScaleEnabled": True,
          "Bones": {"j_kosi": {"Position": {"X": 0, "Y": 0.02, "Z": 0}, "Rotation": {"X": 0, "Y": 0, "Z": 0},
                               "Scale": {"X": 1, "Y": 1, "Z": 0.9}}}}
    template = parse(encode(2, v2))
    check(template.name == "Older" and template.bones["j_kosi"].translation == (0.0, 0.02, 0.0)
          and template.bones["j_kosi"].scaling == (1.0, 1.0, 0.9), "version 2 profile")
    again = Template.from_json(parse(SUPERTHICC).to_json())
    check(again.bones["iv_shiri_l"].scaling == (1.85, 2.0, 2.2), "a template survives being stored on the rig")


def test_parse_errors():
    cases = {
        "": "Paste a Customize+ template first",
        "hello there": "Not a Customize+ template",
        base64.b64encode(b"not gzip at all").decode(): "Not a Customize+ template",
        encode(7, b"no json here"): "damaged",
        encode(7, {"Name": "x"}): "no bones",
    }
    for text, message in cases.items():
        try:
            parse(text)
            check(False, "{!r} is refused".format(text[:20]))
        except TemplateError as error:
            check(message in str(error), "{!r} is refused: {!s}".format(text[:20], error))
    for value in (None, "wide", [1.0]):
        text = json.dumps(template_json({"j_kao": bone_json(scaling=(value, 1.0, 1.0))}))
        try:
            parse(text)
            check(False, "a scaling of {!r} is refused".format(value))
        except TemplateError as error:
            check("isn't a number" in str(error), "a scaling of {!r} is refused: {!s}".format(value, error))


def test_parse_cleanup():
    template = make_template({
        "a": bone_json(rotation=(270.0, -200.0, 10.0)),
        "b": bone_json(scaling=(600.0, 1.0, 1.0)),
        "c": bone_json(rotation=(0.05, 0.0, 0.0)),
        "d": bone_json(PropagateScale=True),
        "e": bone_json(ChildScaling={"X": 0, "Y": 0, "Z": 0}, ChildScalingIndependent=False, scaling=(1.1, 1, 1)),
    })
    check(template.bones["a"].rotation == (-90.0, 160.0, 10.0), "angles wrapped into -180..180 like Customize+")
    check(template.bones["b"].scaling[0] == 512.0, "values clamped to 512 like Customize+")
    check(set(template.edited()) == {"a", "b", "e"}, "bones Customize+ skips (tiny rotation, only a flag) are left out")
    check(template.bones["e"].child_scaling == (1.0, 1.0, 1.0), "an unset child scaling counts as 1")
    endless = make_template({"f": bone_json(rotation=(float("inf"), 0.0, 0.0), scaling=(float("-inf"), 1.0, 1.0))})
    check(endless.bones["f"].rotation == (0.0, 0.0, 0.0) and endless.bones["f"].scaling[0] == -512.0,
          "endless values (Infinity in the JSON): no turn, the scale clamped to 512")


# -----------------------------------------------------------------------------
# The math

def test_rotation_order():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(50):
        x, y, z = rng.uniform(-180.0, 180.0, 3)
        q = q_ypr(*np.radians((x, y, z)))
        expected = solver.quaternion_matrix((q[3], q[0], q[1], q[2]))
        worst = max(worst, float(np.abs(solver.cplus_rotation((x, y, z)) - expected).max()))
    check(worst < 1e-12, "rotations turn like Customize+'s CreateFromYawPitchRoll(X, Y, Z) ({:.1e})".format(worst))
    turned = solver.cplus_rotation((90.0, 0.0, 0.0)) @ np.array([0.0, 0.0, 1.0])
    check(np.allclose(turned, (1.0, 0.0, 0.0)), "rotation X turns about the bone's Y axis (Z to X)")
    turned = solver.cplus_rotation((0.0, 90.0, 0.0)) @ np.array([0.0, 1.0, 0.0])
    check(np.allclose(turned, (0.0, 0.0, 1.0)), "rotation Y turns about the bone's X axis (Y to Z)")


def test_simulation_matches_customize_plus_code():
    rng = np.random.default_rng(7)
    children = {}
    for name in fixture.ORDER:
        children.setdefault(fixture.PARENTS[name], []).append(name)
    worst = 0.0
    for trial in range(12):
        model = random_model(rng)
        edits = random_edits(rng, propagate=trial > 1)
        transforms = {name: Qs.from_matrix(matrix) for name, matrix in model.items()}
        port_apply(fixture.ORDER, children, transforms, edits)
        simulation = solver.simulate(fixture.PARENTS, model, edits, solver.Options())
        for name in fixture.ORDER:
            worst = max(worst, float(np.abs(simulation.model[name] - transforms[name].matrix()).max()))
    check(worst < 1e-9, "the game's bone changes match Customize+'s own code, with propagation ({:.1e})".format(worst))


def test_simulation_by_hand():
    position, rotation = fixture.game_rest("j_asi_a_l")
    child_position, _child_rotation = fixture.game_rest("j_asi_b_l")
    model = {name: solver.compose(*fixture.game_rest(name), 1.0) for name in fixture.ORDER}
    edits = {"j_asi_a_l": BoneEdit(translation=(0.01, 0.0, 0.0), rotation=(0.0, 0.0, 90.0), scaling=(1.0, 2.0, 1.0))}
    result = solver.simulate(fixture.PARENTS, model, edits, solver.Options())
    thigh = result.model["j_asi_a_l"]
    # Z 90 turns X onto Y: the position moves 1 cm along the bone's Y axis.
    check(np.allclose(thigh[:3, 3], position + 0.01 * rotation[:, 1]), "position offsets go along the turned axes")
    check(np.allclose(np.linalg.norm(thigh[:3, :3], axis=0), (1.0, 2.0, 1.0)), "scaling is along the bone's own axes")
    check(np.allclose(result.model["j_asi_b_l"], model["j_asi_b_l"]), "the knee stays where it was (no propagation)")
    edits = {"j_asi_a_l": BoneEdit(translation=(0.01, 0.0, 0.0), propagate_scale=True, scaling=(1.0, 2.0, 1.0))}
    result = solver.simulate(fixture.PARENTS, model, edits, solver.Options())
    knee = result.model["j_asi_b_l"][:3, 3]
    moved = position + 0.01 * rotation[:, 0]
    expected = moved + 0.01 * rotation[:, 0] + np.array([1.0, 2.0, 1.0]) * (child_position - moved)
    check(np.allclose(knee, expected), "propagated scale scales along the model's axes about the new position")
    root = solver.root_change(BoneEdit(translation=(0.0, 0.1, 0.0), scaling=(1.1, 1.1, 1.1)), solver.Options())
    check(np.allclose(root, solver.compose((0.0, 0.1, 0.0), np.identity(3), 1.1)), "the root scales and moves the character")
    check(solver.root_change(BoneEdit(scaling=(1.0, 1.0, 1.0), rotation=(0, 0, 30)), solver.Options()) is None,
          "the root's rotation is ignored, like Customize+ does")


def test_detect_axes():
    expected = {
        'GAME': (fixture.TO_BLENDER, 1.0, np.identity(3)),
        'YUP': (np.identity(3), 1.0, fixture.SWAP_YUP),
        'CM': (fixture.TO_BLENDER, 100.0, fixture.SWAP_CM),
    }
    for style in fixture.STYLES:
        reset_scene()
        armature = fixture.build_rig(style=style)
        axes = solver.detect_axes({bone.name: np.array(bone.matrix_local) for bone in armature.data.bones})
        if style == 'FREE':
            turned = all(np.allclose(axes.frame(name), solver.axis_rotation(1, turn), atol=1e-4)
                         for name, turn in fixture.FREE_TURNS.items())
            check(axes.free == len(fixture.FREE_TURNS) and turned, "FREE: freely turned bones found and measured")
            continue
        turn, units, frame = expected[style]
        same = all(np.allclose(axes.frame(bone.name), frame) for bone in armature.data.bones)
        check(np.allclose(axes.turn, turn) and axes.units == units and same,
              "{:s}: axes found (units {:g}, {:d} swapped)".format(style, axes.units, axes.swapped))


# -----------------------------------------------------------------------------
# The rig in Blender

def test_rig_matches_game():
    for style in fixture.STYLES:
        reset_scene()
        scene = fixture.build(style=style)
        armature = scene["rig"]
        template = parse(SUPERTHICC)
        extra = make_template({
            "j_ude_a_l": bone_json(translation=(0.01, -0.02, 0.005), rotation=(20.0, -35.0, 10.0), scaling=(1.1, 0.8, 1.3)),
            "j_kao": bone_json(rotation=(0.0, 0.0, 25.0), scaling=(1.2, 1.2, 1.2)),
            "iv_shiri_l": bone_json(translation=(0.0, 0.0, -0.015), rotation=(15.0, 5.0, 0.0), scaling=(1.85, 2.0, 2.2)),
        })
        for name, used in (("Superthicc+", template), ("turns", extra)):
            result = apply(SUPERTHICC if used is template else json.dumps(template_json({
                name: bone_json(translation=edit.translation, rotation=edit.rotation, scaling=edit.scaling)
                for name, edit in extra.bones.items()})))
            check(result == {'FINISHED'}, "{:s} {:s}: applied".format(style, name))
            units = 100.0 if style == 'CM' else 1.0
            error = rig_error(armature, used) / units
            check(error < 2e-5, "{:s} {:s}: rig at rest as in the game ({:.1e})".format(style, name, error))
            error = max(mesh_error(scene["body"], armature, used), mesh_error(scene["garment"], armature, used))
            check(error < 2e-5, "{:s} {:s}: meshes at rest as in the game ({:.1e} m)".format(style, name, error))
            fixture.pose(armature, seed=4)
            bpy.context.view_layer.update()
            error = max(mesh_error(scene["body"], armature, used), mesh_error(scene["garment"], armature, used))
            check(error < 2e-5, "{:s} {:s}: posed meshes as in the game ({:.1e} m)".format(style, name, error))
            fixture.clear_pose(armature)
            bpy.context.view_layer.update()


def test_same_result_whatever_the_bone_axes():
    shown = {}
    template = json.dumps(template_json({
        "j_asi_a_l": bone_json(translation=(0.01, 0.0, 0.01), rotation=(10.0, 20.0, -30.0), scaling=(1.0, 1.36, 1.4)),
        "j_ude_a_l": bone_json(rotation=(0.0, 30.0, 0.0), scaling=(1.3, 1.0, 1.0)),
        "iv_shiri_l": bone_json(translation=(0.0, 0.0, -0.015), scaling=(1.85, 2.0, 2.2)),
        "j_sebo_b": bone_json(scaling=(1.05, 1.05, 0.98)),
    }))
    for style in fixture.STYLES:
        reset_scene()
        scene = fixture.build(style=style, garment=False)
        apply(template)
        body = scene["body"]
        shown[style] = fixture.evaluated_coordinates(body) @ np.array(body.matrix_world)[:3, :3].T
    moved = float(np.linalg.norm(shown['GAME'] - fixture.base_coordinates(scene["body"]), axis=1).max())
    worst = max(float(np.linalg.norm(shown[style] - shown['GAME'], axis=1).max()) for style in fixture.STYLES)
    check(moved > 0.01 and worst < 1e-5,
          "the body changes the same whatever axes the armature's bones have ({:.1e} m, moved {:.3f})".format(worst, moved))


def test_propagation():
    templates = {
        "rotation": {"j_asi_a_l": bone_json(translation=(0.01, 0.0, 0.0), rotation=(10.0, 0.0, 25.0),
                                            PropagateRotation=True)},
        "position only": {"j_sebo_c": bone_json(translation=(0.0, 0.02, 0.0), scaling=(1.2, 1.2, 1.2),
                                                PropagateTranslation=True)},
        "nested": {"j_kosi": bone_json(translation=(0.0, 0.01, 0.0), rotation=(0.0, 10.0, 0.0), PropagateRotation=True),
                   "j_asi_a_l": bone_json(rotation=(0.0, 0.0, 15.0), scaling=(1.0, 1.1, 1.1), PropagateRotation=True),
                   "j_asi_c_l": bone_json(scaling=(1.0, 1.3, 1.3))},
        "scale": {"j_sebo_b": bone_json(scaling=(1.1, 1.2, 1.05), PropagateScale=True)},
        "child scale": {"j_asi_a_l": bone_json(scaling=(1.0, 1.3, 1.3), PropagateScale=True, ChildScalingIndependent=True,
                                               ChildScaling={"X": 1.0, "Y": 1.1, "Z": 1.1})},
    }
    for name, bones in templates.items():
        reset_scene()
        scene = fixture.build(garment=False)
        armature = scene["rig"]
        text = json.dumps(template_json(bones))
        apply(text)
        template = parse(text)
        error = mesh_error(scene["body"], armature, template)
        check(error < 2e-5, "propagated {:s}: as in the game at rest ({:.1e} m)".format(name, error))
        fixture.pose(armature, seed=5, degrees=20.0)
        bpy.context.view_layer.update()
        error = mesh_error(scene["body"], armature, template)
        # A propagated scale shears the bones below; the game drops that shear pose by pose, the rig
        # only as it is at rest. With every bone turned up to 20 degrees that's a few millimetres.
        exact = "scale" not in name
        limit = 2e-5 if exact else 0.01
        check(error < limit, "propagated {:s}: posed, {:s} ({:.1e} m)".format(
            name, "as in the game" if exact else "within 1 cm of the game", error))


def test_zero_child_scaling():
    reset_scene()
    scene = fixture.build(garment=False)
    # An old template (Child Scaling was 0 by default) with Independent switched on later.
    text = json.dumps(template_json({"j_asi_a_l": bone_json(
        scaling=(1.0, 1.1, 1.1), PropagateScale=True, ChildScalingIndependent=True,
        ChildScaling={"X": 1.0, "Y": 0.0, "Z": 1.0})}))
    result = apply(text)
    error = mesh_error(scene["body"], scene["rig"], parse(text))
    check(result == {'FINISHED'} and error < 1e-3,
          "a child scaling of 0 flattens the leg like in the game ({:.1e} m)".format(error))


def test_root_and_options():
    reset_scene()
    scene = fixture.build(garment=False)
    armature = scene["rig"]
    text = json.dumps(template_json({
        "n_root": bone_json(translation=(0.0, 0.05, 0.0), rotation=(0.0, 45.0, 0.0), scaling=(1.1, 1.1, 1.1)),
        "j_kosi": bone_json(translation=(0.0, 0.01, 0.0), rotation=(0.0, 0.0, 20.0), scaling=(1.0, 1.15, 1.1)),
    }))
    template = parse(text)
    apply(text)
    error = mesh_error(scene["body"], armature, template)
    check(error < 2e-5, "the root's scale and position move the whole character ({:.1e} m)".format(error))
    root = rig.rig_of(armature).pose.bones["j_asi_e_l"].constraints.get("C+ Root")
    check(root is not None, "the root's change is on every bone")
    for flags in ({"use_rotation": False, "use_translation": False}, {"use_scale": False}):
        apply(text, **flags)
        options = solver.Options(scale=flags.get("use_scale", True), rotation=flags.get("use_rotation", True),
                                 translation=flags.get("use_translation", True))
        error = mesh_error(scene["body"], armature, template, options)
        check(error < 2e-5, "only {:s}: as in the game ({:.1e} m)".format(
            ", ".join(key[4:] for key in ("use_scale", "use_rotation", "use_translation") if key not in flags), error))


def test_root_without_root_bone():
    reset_scene()
    scene = fixture.build(garment=False)
    body_rig, body = scene["rig"], scene["body"]
    # A hair armature imported without n_root, and a prop's armature with no bone of the game's skeleton.
    hair_rig = fixture.build_rig(name="Hair Rig", bones=fixture.HEAD_CHAIN[1:] + list(fixture.HAIR_PARENTS))
    hair = fixture.build_body(hair_rig, name="Hair", radius=0.012, seed=4)
    prop = bpy.data.objects.new("Prop", bpy.data.armatures.new("Prop"))
    bpy.context.scene.collection.objects.link(prop)
    bpy.context.view_layer.objects.active = prop
    bpy.ops.object.mode_set(mode='EDIT')
    prop.data.edit_bones.new("Bone").tail = (0.0, 0.1, 0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    text = json.dumps(template_json({"n_root": bone_json(translation=(0.0, 0.05, 0.0), scaling=(1.1, 1.1, 1.1))},
                                    name="Height"))
    result = solver.plan(*rig.skeleton(hair_rig.data), parse(text), solver.Options())
    check(result.root is not None and result.edited == ["n_root"] and not result.missing,
          "the root's change needs no n_root bone")
    check(apply(text) == {'FINISHED'} and rig.is_on(body_rig) and rig.is_on(hair_rig) and rig.rig_of(prop) is None,
          "a height template goes on every armature of the game's skeleton, with an n_root or not")
    # Game Y up is Blender Z up: 10% larger about the origin, 5 cm higher.
    expected = {mesh: 1.1 * fixture.base_coordinates(mesh) + (0.0, 0.0, 0.05) for mesh in (body, hair)}
    error = max(float(np.abs(fixture.evaluated_coordinates(mesh) - expected[mesh]).max()) for mesh in (body, hair))
    check(error < 1e-5, "body and hair grow and move together ({:.1e} m)".format(error))


# -----------------------------------------------------------------------------
# Working with it

def modifier_states(obj):
    return [(m.name, m.object.name if m.object else None, m.show_viewport, m.show_render)
            for m in obj.modifiers if m.type == 'ARMATURE']


def test_toggle_and_remove_keep_pose_and_animation():
    reset_scene()
    scene = fixture.build()
    armature, body = scene["rig"], scene["body"]
    body.modifiers["Armature"].show_render = False  # a setting of the user's that must come back
    before = modifier_states(body)
    # An animation: the thigh swings between frames 1 and 20.
    thigh = armature.pose.bones["j_asi_a_l"]
    thigh.rotation_mode = 'XYZ'
    thigh.rotation_euler = (0.0, 0.0, 0.0)
    thigh.keyframe_insert("rotation_euler", frame=1)
    thigh.rotation_euler = (0.9, 0.2, -0.3)
    thigh.keyframe_insert("rotation_euler", frame=20)
    action = armature.animation_data.action
    fixture.pose(armature, seed=2, degrees=15.0)  # a pose on top, on other bones
    thigh.rotation_euler = (0.0, 0.0, 0.0)
    basis = {bone.name: np.array(bone.matrix_basis) for bone in armature.pose.bones}
    template = parse(SUPERTHICC)
    bpy.context.scene.frame_set(1)
    apply(SUPERTHICC)
    check(rig.is_on(armature), "applied: on")
    check([m[:3] for m in modifier_states(body)] == [("Armature", "Skeleton", False), ("C+ Armature", "C+ Skeleton", True)],
          "the body follows the rig through a copy of its modifier right after it")
    other = rig.rig_of(armature)
    check(other.parent == armature and other.hide_viewport and other.hide_select, "the rig is a hidden child of the armature")
    for frame in (1, 12, 20):
        bpy.context.scene.frame_set(frame)
        error = mesh_error(body, armature, template)
        check(error < 2e-5, "frame {:d} of the animation: as in the game ({:.1e} m)".format(frame, error))
    bpy.context.scene.frame_set(1)

    check(bpy.ops.magic_fit.cplus_toggle() == {'FINISHED'} and not rig.is_on(armature), "toggled off")
    check(modifier_states(body) == before, "off: the body's modifiers are as they were")
    check(rig.rig_of(armature) is not None, "off: the rig stays for turning it on again")
    check(bpy.ops.magic_fit.cplus_toggle() == {'FINISHED'} and rig.is_on(armature), "toggled on again")
    error = mesh_error(body, armature, template)
    check(error < 2e-5, "on again: as in the game ({:.1e} m)".format(error))

    fixture.clear_pose(armature)  # like Alt G / Alt R / Alt S in Pose Mode
    bpy.context.view_layer.update()
    error = mesh_error(body, armature, template)
    check(error < 2e-5 and rig.is_on(armature), "clearing the pose keeps Customize+ ({:.1e} m)".format(error))
    fixture.pose(armature, seed=2, degrees=15.0)
    thigh.rotation_euler = (0.0, 0.0, 0.0)

    check(bpy.ops.magic_fit.cplus_remove() == {'FINISHED'}, "removed")
    check(modifier_states(body) == before, "removed: the body's modifiers are as they were")
    check(rig.rig_of(armature) is None and not [o for o in bpy.data.objects if o.name.startswith("C+")]
          and not [a for a in bpy.data.actions if a.name.startswith("C+")]
          and not [a for a in bpy.data.armatures if a.name.startswith("C+")], "removed: no rig, actions or data left")
    check(armature.animation_data.action == action and len(list(_fcurves(action))) == 3,
          "the armature's animation is untouched")
    unchanged = all(np.allclose(np.array(bone.matrix_basis), basis[bone.name]) for bone in armature.pose.bones)
    check(unchanged, "the armature's pose is untouched")
    check(all(bone.bone.inherit_scale == 'FULL' and bone.bone.use_inherit_rotation for bone in armature.pose.bones),
          "the armature's bones still inherit scale and rotation")


def _fcurves(action):
    if hasattr(action, "layers") and action.layers:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    yield from bag.fcurves
    else:
        yield from action.fcurves


def test_modes():
    reset_scene()
    scene = fixture.build()
    armature, body, garment = scene["rig"], scene["body"], scene["garment"]
    template = parse(SUPERTHICC)
    view_layer = bpy.context.view_layer
    cases = (
        ("Pose Mode", armature, 'POSE'),
        ("Weight Paint", body, 'WEIGHT_PAINT'),
        ("a mesh's Edit Mode", garment, 'EDIT'),
        ("Sculpt Mode", body, 'SCULPT'),
    )
    for label, obj, mode in cases:
        for other in bpy.context.selected_objects:
            other.select_set(False)
        view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode=mode)
        bpy.context.scene.magic_fit_cplus.template = SUPERTHICC
        bpy.context.scene.magic_fit_cplus.armature = armature
        applied = bpy.ops.magic_fit.cplus_apply() == {'FINISHED'}
        on = rig.is_on(armature)
        off = bpy.ops.magic_fit.cplus_toggle() == {'FINISHED'} and not rig.is_on(armature)
        again = bpy.ops.magic_fit.cplus_toggle() == {'FINISHED'} and rig.is_on(armature)
        check(applied and on and off and again and obj.mode == mode,
              "{:s}: applied and toggled, still in {:s}".format(label, mode))
        bpy.ops.object.mode_set(mode='OBJECT')
        error = mesh_error(body, armature, template)
        check(error < 2e-5, "{:s}: as in the game ({:.1e} m)".format(label, error))
        bpy.ops.magic_fit.cplus_remove()
    view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    check(not bpy.ops.magic_fit.cplus_apply.poll()
          and cplus_operators.apply_problem(bpy.context) == "Leave the Edit Mode of 'Skeleton' first",
          "not while the armature is in Edit Mode")
    bpy.ops.object.mode_set(mode='OBJECT')


def test_apply_again_and_rest_change():
    reset_scene()
    scene = fixture.build()
    armature, body = scene["rig"], scene["body"]
    apply(SUPERTHICC)
    second = json.dumps(template_json({"j_kosi": bone_json(scaling=(1.3, 1.0, 1.0))}, name="Second"))
    apply(second)
    copies = [m for m in body.modifiers if m.name.startswith("C+")]
    rigs = [o for o in bpy.data.objects if rig.is_rig(o)]
    actions = [a for a in bpy.data.actions if a.name.startswith("C+")]
    check(len(copies) == 1 and len(rigs) == 1 and len(actions) == 1,
          "applying again replaces the template ({:d} copies, {:d} rigs, {:d} actions)".format(len(copies), len(rigs), len(actions)))
    check(rig.state(rigs[0])[0].name == "Second", "the rig remembers the template it shows")
    error = mesh_error(body, armature, parse(second))
    check(error < 2e-5, "the second template: as in the game ({:.1e} m)".format(error))

    bpy.ops.magic_fit.cplus_toggle()  # off
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    armature.data.edit_bones["j_kosi"].head.z += 0.02
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.magic_fit.cplus_toggle()  # on
    rebuilt = rig.rig_of(armature)
    check(np.allclose(np.array(rebuilt.data.bones["j_kosi"].matrix_local), np.array(armature.data.bones["j_kosi"].matrix_local)),
          "a changed rest pose rebuilds the rig when turned on")
    error = mesh_error(body, armature, parse(second))
    check(error < 2e-5 and not rig.is_stale(armature), "rebuilt rig: as in the game ({:.1e} m)".format(error))

    bpy.ops.object.mode_set(mode='EDIT')
    armature.data.edit_bones["j_kosi"].head.z -= 0.02
    armature.data.edit_bones.new("Added").tail = (0.0, 0.1, 0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    check(rig.is_on(armature) and rig.is_stale(armature), "a rest pose or bones changed while on are noticed")
    apply(second)
    error = mesh_error(body, armature, parse(second))
    check(error < 2e-5 and not rig.is_stale(armature), "applying again shows them ({:.1e} m)".format(error))


def test_new_mesh_and_deleted_rig():
    reset_scene()
    scene = fixture.build()
    armature, body = scene["rig"], scene["body"]
    apply(SUPERTHICC)
    extra = fixture.build_body(armature, name="Later", seed=3)
    check(rig.missing_meshes(armature) == ["Later"], "a mesh added later is listed as left out")
    apply(SUPERTHICC)
    check(rig.missing_meshes(armature) == [] and extra.modifiers[-1].name.startswith("C+"), "applying again includes it")
    bpy.data.objects.remove(rig.rig_of(armature))  # deleted by hand
    check(not rig.is_on(armature), "a rig deleted by hand counts as off")
    apply(SUPERTHICC)
    check(len([m for m in body.modifiers if m.type == 'ARMATURE']) == 2 and rig.is_on(armature),
          "applying again reuses the copies the deleted rig left")
    bpy.data.objects.remove(rig.rig_of(armature))
    removed = bpy.ops.magic_fit.cplus_toggle() == {'FINISHED'}  # nothing to toggle: applies again
    check(removed and rig.is_on(armature), "toggling with the rig gone applies the template again")
    body.modifiers.remove(body.modifiers["Armature"])  # the user deletes the original modifier while on
    bpy.ops.magic_fit.cplus_remove()
    states = modifier_states(body)
    check(len(states) == 1 and states[0][1] == "Skeleton" and states[0][2],
          "removing turns a copy without its original back into a normal Armature modifier")


def test_armature_choice():
    reset_scene()
    scene = fixture.build(garment=False)
    armature, body = scene["rig"], scene["body"]
    context = bpy.context
    context.scene.magic_fit_cplus.scope = 'ONE'
    context.view_layer.objects.active = None
    check(cplus_operators.target_armature(context) is None, "nothing picked or active: no armature")
    context.view_layer.objects.active = body
    check(cplus_operators.target_armature(context) == armature, "the active mesh's armature")
    other = fixture.build_rig(name="Other")
    context.scene.magic_fit.target = body
    context.view_layer.objects.active = other
    check(cplus_operators.target_armature(context) == armature, "the body's armature comes before the active one")
    context.scene.magic_fit_cplus.armature = other
    check(cplus_operators.target_armature(context) == other, "the picked armature comes first")
    apply(SUPERTHICC, armature=armature)
    from magic_fit.cplus import properties
    check(properties._poll_armature(None, armature) and not properties._poll_armature(None, rig.rig_of(armature)),
          "the picker offers armatures but not rigs")


def moved(obj):
    return float(np.abs(fixture.evaluated_coordinates(obj) - fixture.base_coordinates(obj)).max())


def test_face_and_hair_armatures():
    reset_scene()
    scene = fixture.build(garment=False)
    body_rig, body = scene["rig"], scene["body"]
    face_rig, face = fixture.build_head('FACE')
    hair_rig, hair = fixture.build_head('HAIR')
    unrelated = bpy.data.objects.new("Unrelated", bpy.data.armatures.new("Unrelated"))
    bpy.context.scene.collection.objects.link(unrelated)
    text = json.dumps(template_json({
        "j_kao": bone_json(scaling=(1.1, 1.1, 1.1), PropagateScale=True),
        "j_f_hana_l": bone_json(translation=(0.0, 0.002, 0.0), scaling=(1.3, 1.2, 1.1)),
        "j_kami_a": bone_json(scaling=(1.2, 1.0, 1.0)),
        "j_ex_h0113_ke_b": bone_json(rotation=(0.0, 0.0, 20.0), scaling=(1.0, 1.3, 1.0)),
        "j_sebo_c": bone_json(scaling=(1.05, 1.05, 1.05)),
        "j_asi_a_l": bone_json(scaling=(1.0, 1.36, 1.4)),
    }))
    template = parse(text)
    rigs = (("body", body_rig, body), ("face", face_rig, face), ("hair", hair_rig, hair))
    for label, armature, _mesh in rigs[1:]:
        axes = solver.detect_axes(rig.skeleton(armature.data)[1])
        check(axes.swapped == 0 and axes.free == 0 and axes.units == 1.0 and np.allclose(axes.turn, fixture.TO_BLENDER),
              "{:s} rig: axes found from {:d} bones of the game's skeleton".format(label, axes.known))
    check(apply(text) == {'FINISHED'}, "all armatures: applied")
    check(all(rig.is_on(armature) for _label, armature, _mesh in rigs) and rig.rig_of(unrelated) is None,
          "all armatures: the body's, the face's and the hair's, not one without the template's bones")
    for label, armature, mesh in rigs:
        error = mesh_error(mesh, armature, template)
        check(error < 2e-5 and moved(mesh) > 0.001, "{:s}: as in the game ({:.1e} m)".format(label, error))
    for seed, (_label, armature, _mesh) in enumerate(rigs):
        fixture.pose(armature, seed=seed, degrees=15.0)
    bpy.context.view_layer.update()
    for label, armature, mesh in rigs:
        error = mesh_error(mesh, armature, template)
        check(error < 2e-5, "{:s}: posed, as in the game ({:.1e} m)".format(label, error))
    for _label, armature, _mesh in rigs:
        fixture.clear_pose(armature)

    bpy.ops.magic_fit.cplus_toggle()
    check(not any(rig.is_on(armature) for _label, armature, _mesh in rigs) and moved(face) < 1e-6
          and moved(hair) < 1e-6, "the hotkey turns every armature off")
    bpy.ops.magic_fit.cplus_toggle()
    check(all(rig.is_on(armature) for _label, armature, _mesh in rigs), "and on again")
    settings = bpy.context.scene.magic_fit_cplus
    settings.scope = 'ONE'
    settings.armature = hair_rig
    bpy.ops.magic_fit.cplus_remove()
    check(rig.rig_of(hair_rig) is None and rig.is_on(body_rig) and rig.is_on(face_rig),
          "One Armature: only the picked armature's is removed")
    settings.scope = 'ALL'
    bpy.ops.magic_fit.cplus_remove()
    check(not [obj for obj in bpy.data.objects if rig.is_rig(obj)], "All Armatures: every rig is removed")


def test_another_template_on_fewer_armatures():
    reset_scene()
    scene = fixture.build(garment=False)
    body_rig = scene["rig"]
    hair_rig, hair = fixture.build_head('HAIR')
    head = json.dumps(template_json({"j_kao": bone_json(scaling=(1.2, 1.2, 1.2))}, name="Head"))
    apply(head)
    check(rig.is_on(body_rig) and rig.is_on(hair_rig) and moved(hair) > 0.001, "a head template on body and hair")
    apply(json.dumps(template_json({"j_asi_a_l": bone_json(scaling=(1.0, 1.3, 1.3))}, name="Legs")))
    shown = {rig.state(rig.rig_of(armature))[0].name for armature in cplus_operators.rigged(bpy.context)}
    check(rig.rig_of(hair_rig) is None and moved(hair) < 1e-6 and shown == {"Legs"},
          "All Armatures: a template for the legs takes the head template off the hair ({:s})".format(
              ", ".join(sorted(shown))))
    bpy.context.scene.magic_fit_cplus.scope = 'ONE'
    apply(head, armature=hair_rig)
    check(rig.state(rig.rig_of(body_rig))[0].name == "Legs" and rig.is_on(hair_rig),
          "One Armature leaves the other armatures alone")


def test_armature_in_disabled_collection():
    reset_scene()
    scene = fixture.build(garment=False)
    armature, body = scene["rig"], scene["body"]
    # Disabled in viewports (the monitor icon): only the body's modifier still has the armature evaluated.
    disabled = bpy.data.collections.new("Disabled")
    bpy.context.scene.collection.children.link(disabled)
    disabled.objects.link(armature)
    bpy.context.scene.collection.objects.unlink(armature)
    disabled.hide_viewport = True
    result = apply(SUPERTHICC)
    error = mesh_error(body, armature, parse(SUPERTHICC))
    check(result == {'FINISHED'} and error < 2e-5,
          "an armature in a collection disabled in viewports: as in the game ({:.1e} m)".format(error))
    bpy.data.collections.remove(disabled)


def test_bones_without_rest_pose():
    reset_scene()
    # Every bone at the origin without a parent, the way an armature made from a model's bone list
    # alone is: changing them would scale and turn the mesh about the origin.
    placeholder = fixture.build_rig(name="Placeholder", placeholders=set(fixture.PARENTS))
    imported = fixture.build_body(placeholder, name="Imported")
    parents, rest = rig.skeleton(placeholder.data)
    result = solver.plan(parents, rest, parse(SUPERTHICC), solver.Options())
    check(len(result.unplaced) == 28 and not result.edited, "an armature with its bones at the origin: all 28 left out")
    check(apply(SUPERTHICC) == {'CANCELLED'} or moved(imported) < 1e-6,
          "nothing moves on it ({:.1e} m)".format(moved(imported)))

    one = {"Bone": np.identity(4), "n_root": np.identity(4)}
    check(not solver.unplaced(one), "a single bone of another rig at the origin is taken as placed there")

    reset_scene()
    # Only the face bones at the origin (like the Animation Kit's face skeleton).
    face_rig, face = fixture.build_head('FACE', name="Half Placed", placeholders=set(fixture.FACE_PARENTS))
    axes = solver.detect_axes(rig.skeleton(face_rig.data)[1])
    check(axes.units == 1.0 and axes.swapped == 0 and axes.free == 0 and axes.unplaced == frozenset(fixture.FACE_PARENTS),
          "face bones at the origin are found and don't throw the axes off")
    text = json.dumps(template_json({"j_kao": bone_json(scaling=(1.2, 1.2, 1.2)),
                                     "j_f_hana_l": bone_json(scaling=(1.5, 1.5, 1.5))}))
    apply(text)
    error = mesh_error(face, face_rig, parse(text))
    names = [group.name for group in face.vertex_groups]
    nose = [v.index for v in face.data.vertices
            if any(names[g.group] == "j_f_hana_l" and g.weight > 0.99 for g in v.groups)]
    shown = fixture.evaluated_coordinates(face)
    base = fixture.base_coordinates(face)
    check(error < 2e-5 and float(np.abs(shown[nose] - base[nose]).max(initial=0.0)) < 1e-6,
          "the head changes as in the game, the nose bone without a rest pose is left alone ({:.1e} m)".format(error))


def test_small_rigs():
    reset_scene()
    for style, turn, units, frame in (('GAME', fixture.TO_BLENDER, 1.0, np.identity(3)),
                                      ('YUP', np.identity(3), 1.0, fixture.SWAP_YUP),
                                      ('CM', fixture.TO_BLENDER, 100.0, fixture.SWAP_CM)):
        chain = fixture.build_rig(name="Chain " + style, style=style, bones=fixture.HEAD_CHAIN)
        axes = solver.detect_axes(rig.skeleton(chain.data)[1])
        same = all(np.allclose(axes.frame(name), frame) for name in fixture.HEAD_CHAIN)
        check(np.allclose(axes.turn, turn) and axes.units == units and same,
              "{:s}: a root-to-head chain alone (nearly a line) still gives the axes".format(style))
        hair_rig, _hair = fixture.build_head('HAIR', name="Hair " + style, style=style)
        axes = solver.detect_axes(rig.skeleton(hair_rig.data)[1])
        same = all(np.allclose(axes.frame(bone.name), frame) for bone in hair_rig.data.bones)
        check(np.allclose(axes.turn, turn) and same, "{:s}: hair rig, the hairstyle's own bones take the others' axes".format(style))


def test_races():
    reset_scene()
    for race in ('c1501', 'c1601', 'c1101', 'c0901'):
        body_rig = fixture.build_rig(name="Body " + race, race=race)
        face_rig, _face = fixture.build_head('FACE', name="Face " + race, race=race)
        for label, armature in (("body", body_rig), ("face", face_rig)):
            rest = rig.skeleton(armature.data)[1]
            axes = solver.detect_axes(rest)
            # Against the Midlander female's turns alone, some bones would look turned freely.
            misfits = [name for name in rest if name in reference.TURNS and solver.angle_between(
                solver.quaternion_matrix(reference.BONES[name][3:]).T
                @ (fixture.TO_BLENDER.T @ solver.orthonormal(rest[name][:3, :3]))) > solver.SNAP_DEGREES]
            check(axes.free == 0 and axes.swapped == 0,
                  "{:s} {:s}: every bone on its race's axes (race found: {!s}; {:d} would misfit the Midlander "
                  "female's)".format(reference.NAMES[race], label, axes.race, len(misfits)))
    reset_scene()
    # An IVCS bone turned 35 degrees differently from the devkit's (another IVCS skeleton): the armature's
    # axes, not a free turn.
    turned = fixture.build_rig(name="Other IVCS", turned={'iv_shiri_l': 35.0, 'j_asi_a_l': 35.0})
    axes = solver.detect_axes(rig.skeleton(turned.data)[1])
    check(np.allclose(axes.frame('iv_shiri_l'), np.identity(3)) and not solver.is_swap(axes.frame('j_asi_a_l')),
          "a mod bone that doesn't fit its one sample keeps the armature's axes; a vanilla bone is turned freely")


def test_other_tools_see_it():
    reset_scene()
    scene = fixture.build()
    armature, body = scene["rig"], scene["body"]
    bpy.context.scene.magic_fit.target = body
    apply(SUPERTHICC)
    check(operators.posed_armature(body) == "C+ Skeleton", "Body Fit's notes count the rig as a pose")
    check(straighten.deforming_armature(body).object == rig.rig_of(armature),
          "Straighten, Skirt and Heels read the bones the body follows")
    from magic_fit.robust_transfer import dependencies, operators as transfer_operators
    settings = bpy.context.scene.magic_fit_transfer
    settings.group_selection = 'DEFORM_POSE_BONES'
    bpy.context.view_layer.objects.active = scene["garment"]
    problem = transfer_operators.check_transfer_ready(bpy.context)
    check(problem is None or dependencies.problem() is not None,
          "Weight Transfer still works with the rig on ({!s})".format(problem))


# -----------------------------------------------------------------------------
# The game's bust size

def bust_constraints(armature):
    rig_object = rig.rig_of(armature)
    return [constraint for bone in rig_object.pose.bones for constraint in bone.constraints
            if constraint.name in (rig.BUST, rig.BUST_FRAME)]


def test_bust_scale():
    check(np.allclose(solver.bust_scale(0), (0.92, 0.8, 0.816)) and np.allclose(solver.bust_scale(50), 1.0)
          and np.allclose(solver.bust_scale(100), (1.08, 1.2, 1.184)) and np.allclose(solver.bust_scale(25), (0.96, 0.9, 0.908)),
          "bust size 0, 25, 50 and 100: the game's scales, the model's own size at 50")
    check(np.allclose(solver.bust_scale(0, lalafell=True), 1.0)
          and np.allclose(solver.bust_scale(50, lalafell=True), (1.04, 1.1, 1.092))
          and np.allclose(solver.bust_scale(100, lalafell=True), (1.08, 1.2, 1.184)),
          "Lalafells: never smaller than the model, as large at 100")
    check(np.allclose(solver.bust_scale(101), solver.bust_scale(0)), "a size beyond 100 counts as 0, like in the game")
    table = {name: rest for name, rest in ((name, solver.compose(fixture.TO_BLENDER @ np.array(values[:3]),
                                                                 fixture.TO_BLENDER @ solver.quaternion_matrix(values[3:]), 1.0))
                                           for name, values in reference.BONES.items() if not name.startswith("j_f_"))}
    sizes = {}
    for label, factor in (("Midlander female", 1.0), ("Lalafell", 0.45), ("Hrothgar male", 1.38), ("Lalafell in cm", 45.0)):
        rest = {name: matrix.copy() for name, matrix in table.items()}
        for matrix in rest.values():
            matrix[:3, 3] *= factor
        axes = solver.detect_axes(rest)
        sizes[label] = (axes.size, solver.is_lalafell(axes))
    check([lalafell for _size, lalafell in sizes.values()] == [False, True, False, True],
          "Lalafell skeletons are told by their size ({:s})".format(
              ", ".join("{:s} {:.2f}".format(label, size) for label, (size, _l) in sizes.items())))


def test_bust_table_matches_game():
    """The ranges are the game's own: its racial scaling table, for every clan of player characters."""
    root = gamefiles.find_game()
    if root is None:
        print("skip: FFXIV isn't installed where Magic Fit finds it; the bust table wasn't compared")
        return
    with gamefiles.Index(gamefiles.sqpack_folder(root), "chara") as index:
        data = index.read("chara/xls/charamake/human.cmp")
    # Penumbra.GameData's CmpData: colours, then per race (Hyur, Elezen, Lalafell, Miqo'te, Roegadyn, Au Ra,
    # Hrothgar, Viera) 10 entries of 14 floats (5 body types x 2 clans), the bust's min and max xyz at 8 to 13.
    start = 2 * (7 * 1024 + 4 * 512) + 32 * 5120
    worst = 0.0
    for race in range(8):
        for clan in range(2):
            values = np.array(struct.unpack_from("<6f", data, start + (race * 10 + clan) * 56 + 32))
            expected = solver.LALAFELL_BUST_RANGE if race == 2 else solver.BUST_RANGE
            worst = max(worst, float(np.abs(values - np.array(expected).flatten()).max()))
    check(len(data) == start + 8 * 10 * 56 and worst < 1e-6,
          "the bust ranges are the game's human.cmp for all 16 clans (game {:s}, {:.1e})".format(
              gamefiles.game_version(root), worst))


def test_bust_rig_matches_game():
    template = cplus_operators.NO_TEMPLATE
    for style in fixture.STYLES:
        reset_scene()
        scene = fixture.build(style=style)
        armature, body = scene["rig"], scene["body"]
        units = 100.0 if style == 'CM' else 1.0
        for size in (0, 85):
            result = apply("", bust=size)
            error = rig_error(armature, template, bust=size) / units
            mesh = max(mesh_error(body, armature, template, bust=size),
                       mesh_error(scene["garment"], armature, template, bust=size))
            check(result == {'FINISHED'} and error < 2e-5 and mesh < 2e-5,
                  "{:s} bust size {:d}: at rest as in the game (rig {:.1e}, meshes {:.1e} m)".format(style, size, error, mesh))
            fixture.pose(armature, seed=6)
            bpy.context.view_layer.update()
            mesh = max(mesh_error(body, armature, template, bust=size),
                       mesh_error(scene["garment"], armature, template, bust=size))
            check(mesh < 2e-5, "{:s} bust size {:d}: posed as in the game ({:.1e} m)".format(style, size, mesh))
            fixture.clear_pose(armature)
            bpy.context.view_layer.update()
        bones = {bone.name for bone in rig.rig_of(armature).pose.bones if any(
            constraint.name == rig.BUST for constraint in bone.constraints)}
        turned = {bone.name for bone in rig.rig_of(armature).pose.bones if any(
            constraint.name == rig.BUST_FRAME for constraint in bone.constraints)}
        check(bones == {"j_mune_l", "j_mune_r", "iv_c_mune_l", "iv_c_mune_r"}
              and turned == ({"j_mune_l"} if style == 'FREE' else set()),
              "{:s}: the breast bones and the bones below them get it ({:s}; turned back: {:s})".format(
                  style, ", ".join(sorted(bones)), ", ".join(sorted(turned)) or "none"))


def test_bust_scales_bones_below_without_moving_them():
    reset_scene()
    scene = fixture.build(garment=False)
    armature = scene["rig"]
    apply("", bust=100)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    posed = rig.rig_of(armature).evaluated_get(depsgraph).pose.bones
    rest = armature.data.bones
    lengths = {name: np.linalg.norm(np.array(posed[name].matrix)[:3, :3], axis=0) for name in ("j_mune_l", "iv_c_mune_l")}
    moved = max(float(np.abs(np.array(posed[name].matrix)[:3, 3] - np.array(rest[name].matrix_local)[:3, 3]).max())
                for name in ("j_mune_l", "iv_c_mune_l", "j_sebo_b"))
    # The fixture's GAME style keeps the game's bone axes, so the scale shows as is on the bone's X, Y and Z.
    check(all(np.allclose(length, (1.08, 1.2, 1.184)) for length in lengths.values()) and moved < 1e-6,
          "the breast bone and the one below it scale by (1.08, 1.2, 1.184) on their own axes and stay where they are")
    others = max(float(np.abs(np.array(posed[bone.name].matrix) - np.array(bone.matrix_local)).max())
                 for bone in rest if bone.name not in ("j_mune_l", "j_mune_r", "iv_c_mune_l", "iv_c_mune_r"))
    check(others < 1e-6, "no other bone changes ({:.1e})".format(others))


def test_bust_with_template():
    for style in ('GAME', 'FREE'):
        reset_scene()
        scene = fixture.build(style=style)
        armature, body = scene["rig"], scene["body"]
        text = json.dumps(template_json({
            "j_mune_l": bone_json(translation=(0.01, 0.0, 0.005), rotation=(10.0, -20.0, 15.0), scaling=(1.2, 1.5, 1.3)),
            "iv_c_mune_r": bone_json(scaling=(1.1, 1.3, 1.2)),
            "j_sebo_b": bone_json(scaling=(1.1, 1.05, 1.1), PropagateScale=True),
            "j_kosi": bone_json(rotation=(0.0, 0.0, 10.0), scaling=(1.0, 1.1, 1.1)),
        }))
        template = parse(text)
        apply(text, bust=70)
        error = mesh_error(body, armature, template, bust=70)
        check(error < 2e-5, "{:s}: template and bust size, as in the game at rest ({:.1e} m)".format(style, error))
        fixture.pose(armature, seed=7, degrees=15.0)
        bpy.context.view_layer.update()
        error = mesh_error(body, armature, template, bust=70)
        # j_sebo_b propagates an uneven scale: the game drops its shear pose by pose (see test_propagation).
        check(error < 0.01, "{:s}: template and bust size, posed, within 1 cm of the game ({:.1e} m)".format(style, error))
        apply(SUPERTHICC, bust=30)
        error = mesh_error(body, armature, parse(SUPERTHICC), bust=30)
        check(error < 2e-5, "{:s}: Superthicc+ and bust size 30, posed, as in the game ({:.1e} m)".format(style, error))
        apply(SUPERTHICC)
        error = mesh_error(body, armature, parse(SUPERTHICC))
        check(not bust_constraints(armature) and error < 2e-5,
              "{:s}: without the bust size, nothing of it is left ({:.1e} m)".format(style, error))


def test_bust_lalafell():
    reset_scene()
    armature = fixture.build_rig(name="Lalafell", race='c1201')
    body = fixture.build_body(armature, name="Lalafell Body")
    template = cplus_operators.NO_TEMPLATE
    apply("", bust=0)
    check(moved(body) < 1e-6 and rig.bust_state(rig.rig_of(armature)) == (0, True),
          "Lalafell, bust size 0: the model's own size")
    apply("", bust=100)
    error = mesh_error(body, armature, template, bust=100, lalafell=True)
    check(error < 2e-5 and moved(body) > 0.001, "Lalafell, bust size 100: as in the game ({:.1e} m)".format(error))
    check(cplus_operators.shown(rig.rig_of(armature)) == "Bust Size 100 (Lalafell)", "the panel says it's the Lalafells'")


def test_bust_live():
    reset_scene()
    scene = fixture.build(garment=False)
    armature, body = scene["rig"], scene["body"]
    template = parse(SUPERTHICC)
    apply(SUPERTHICC, bust=40)
    settings = bpy.context.scene.magic_fit_cplus
    settings.bust_size = 95
    error = mesh_error(body, armature, template, bust=95)
    check(error < 2e-5 and rig.bust_state(rig.rig_of(armature))[0] == 95,
          "changing the bust size shows right away ({:.1e} m)".format(error))
    settings.use_bust = False
    error = mesh_error(body, armature, template)
    check(error < 2e-5 and not bust_constraints(armature) and cplus_operators.shown(rig.rig_of(armature)) == "'Custom - Superthicc+'",
          "turning it off leaves the template alone ({:.1e} m)".format(error))
    settings.use_bust = True
    error = mesh_error(body, armature, template, bust=95)
    check(error < 2e-5, "and on again ({:.1e} m)".format(error))
    actions = [action for action in bpy.data.actions if action.name.startswith("C+")]
    check(len(actions) == 2, "no actions left behind ({:d}: the template's and the bust's)".format(len(actions)))
    bpy.ops.magic_fit.cplus_toggle()
    settings.bust_size = 10
    bpy.ops.magic_fit.cplus_toggle()
    error = mesh_error(body, armature, template, bust=10)
    check(rig.is_on(armature) and error < 2e-5, "changed while off: shown when turned on ({:.1e} m)".format(error))
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    armature.data.edit_bones["j_sebo_b"].head.z += 0.01
    bpy.ops.object.mode_set(mode='OBJECT')
    settings.bust_size = 60  # the rest pose changed since: the rig is made again
    error = mesh_error(body, armature, template, bust=60)
    check(rig.is_on(armature) and error < 2e-5,
          "changed after the rest pose changed: made again, still on ({:.1e} m)".format(error))


def test_bust_only_and_scope():
    reset_scene()
    scene = fixture.build(garment=False)
    body_rig = scene["rig"]
    face_rig, _face = fixture.build_head('FACE')
    settings = bpy.context.scene.magic_fit_cplus
    settings.use_bust = False
    check(cplus_operators.apply_problem(bpy.context) == "Paste a Customize+ template, or turn on Bust Size",
          "nothing to apply without a template or the bust size")
    settings.use_bust = True
    check(cplus_operators.apply_problem(bpy.context) is None, "the bust size alone can be applied")
    check(apply("", bust=65) == {'FINISHED'} and rig.rig_of(body_rig) is not None and rig.rig_of(face_rig) is None,
          "All Armatures: the bust size goes on armatures with breast bones only")
    check(cplus_operators.shown(rig.rig_of(body_rig)) == "Bust Size 65", "the panel shows it")
    face_text = json.dumps(template_json({"j_f_hana_l": bone_json(scaling=(1.2, 1.2, 1.2))}))
    apply(face_text, bust=65)
    check(rig.rig_of(face_rig) is not None and not bust_constraints(face_rig) and bust_constraints(body_rig),
          "a template for the face and the bust size: each armature gets what it has bones for")
    reset_scene()
    lone = fixture.build_rig(name="No Breasts", bones=fixture.HEAD_CHAIN)
    fixture.build_body(lone, name="Head")
    settings.use_bust = True
    check(cplus_operators.apply_problem(bpy.context) == "No armature has breast bones",
          "no armature with breast bones: nothing to apply")


def test_speed():
    reset_scene()
    scene = fixture.build()
    for index in range(30):
        fixture.build_body(scene["rig"], name="Extra {:d}".format(index), seed=index)
    text = json.dumps(template_json({name: bone_json(translation=(0.0, 0.01, 0.0), rotation=(5.0, 0.0, 0.0),
                                                     scaling=(1.1, 1.2, 1.1), PropagateRotation=name == "j_kosi")
                                     for name in fixture.ORDER}))
    start = time.perf_counter()
    apply(text)
    bpy.context.view_layer.update()
    applied = time.perf_counter() - start
    start = time.perf_counter()
    bpy.ops.magic_fit.cplus_toggle()
    bpy.ops.magic_fit.cplus_toggle()
    bpy.context.view_layer.update()
    toggled = time.perf_counter() - start
    check(applied < 1.0 and toggled < 0.5,
          "every bone changed, 32 meshes: applied in {:.2f} s, toggled twice in {:.2f} s".format(applied, toggled))


def test_registered():
    for name in ("cplus_paste", "cplus_apply", "cplus_toggle", "cplus_remove"):
        check(hasattr(bpy.ops.magic_fit, name), "operator {:s} is registered".format(name))
    from magic_fit import preferences
    check(preferences.HOTKEYS[0][0] == "magic_fit.cplus_toggle", "the toggle has a hotkey")


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
