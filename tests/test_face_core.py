"""Headless tests for Face weights (Face Weights, the face repairs, Test Poses and the Weight Brushes' Face
mode), and for reading the game's faces from the game.

They need FFXIV installed where Magic Fit finds it by itself. Run from the repository root:
    blender -b --factory-startup --python tests/test_face_core.py
"""

import os
import shutil
import struct
import sys
import tempfile
import traceback
import zipfile

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import face  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import facedata, facing, gamefiles, operators  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def use_settings(**values):
    settings = bpy.context.scene.magic_fit
    defaults = dict(face_source='DETECT', face_race='c0201', face_number=1, face_close_eyes=True,
                    face_match_neck=True, face_lash_lift=0.0, face_max_groups=8, group_mode='ALL', mode='FACE')
    defaults.update(values)
    for key, value in defaults.items():
        setattr(settings, key, value)
    return settings


def armature_of(obj):
    return next((m.object for m in obj.modifiers if m.type == 'ARMATURE'), None)


def column(names, *prefixes, side=None):
    return [i for i, n in enumerate(names) if n.startswith(prefixes) and (side is None or n.endswith("_" + side))]


def posed_points(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    points = np.array([evaluated.matrix_world @ v.co for v in mesh.vertices])
    evaluated.to_mesh_clear()
    return points


def turn(matrix):
    """The angle (degrees) a 4x4 matrix turns by."""
    return float(np.degrees(np.arccos(np.clip((np.trace(matrix[:3, :3]) - 1) / 2, -1, 1))))


def weigh_all(obj, name="j_kao"):
    """Weight every vertex of ``obj`` fully to ``name`` (a face with wrong weights)."""
    group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
    group.add(range(len(obj.data.vertices)), 1.0, 'REPLACE')


def test_reference():
    ref = facing.Reference.get()
    check(set(ref.races) == set(facing.RACE_NAMES), "reference: all 18 races")
    check(all(len(ref.faces(race)) >= 4 for race in ref.races), "reference: every race has its faces")
    check(ref.resolve('c1401', 101) == 1 and ref.has_face('c1401', 101) and ref.resolve('c1401', 9) is None,
          "reference: Au Ra face 101 uses face 1's model, face 9 doesn't exist")
    data = face.reference('c0201', 1)
    skeleton = data.skeleton
    check(all(skeleton.has(name) for name in ("j_kao", "j_f_mabup_01_l", "j_f_mabdn_01_r", "j_f_dago", "j_f_ulip_01_l")),
          "reference: the face skeleton has head, lid, jaw and lip bones")
    blink = skeleton.deformations('BLINK')
    lid = blink.get("j_f_mabup_01_l")
    check(lid is not None and abs(turn(lid) - 43.7) < 1.0, "reference: the game's blink turns the upper lid about 44 degrees")
    half = skeleton.deformations('BLINK', 0.5).get("j_f_mabup_01_l")
    check(half is not None and abs(turn(half) - turn(lid) / 2.0) < 0.1
          and np.allclose(skeleton.deformations('BLINK', 1.0)["j_f_mabup_01_l"], lid)
          and "j_f_mabup_01_l" not in skeleton.deformations('BLINK', 0.0),
          "reference: a share of the blink turns the lid that share of the way ({:.1f} degrees at half)".format(
              turn(half) if half is not None else 0.0))
    shut = skeleton.deformations('SHUT_EYES').get("j_f_mabup_01_l")
    check(shut is not None and 0.9 < turn(shut) / turn(lid) <= 1.0,
          "reference: the Shut Eyes expression turns the upper lid about as far as the blink's peak ({:.1f} degrees)".format(
              turn(shut) if shut is not None else 0.0))
    features = data.features()
    share = facing.closing_share(data.points, data.tris, data.names, data.weights, data.eyes(), skeleton,
                                 {side: features.opening_box(side) for side in features.eyes})
    check(share is not None and 1.0 - facing.CLOSE_SLACK <= share <= 1.0,
          "reference: the game's face is shut in the Shut Eyes expression, its lids meeting just before ({:.0%})".format(
              share or 0.0))
    check(data.weights.shape == (len(data.points), len(data.names)) and np.allclose(data.weights.sum(1), 1.0, atol=0.01),
          "reference: the skin's weights add up to 1")
    elezen = face.reference('c0501', 1)
    ears = [part for part in elezen.extras if part.kind == 1 and "mim" in part.attributes and "fv_" not in part.attributes]
    ear_bones = [elezen.names.index(n) for n in ("j_mimi_l", "j_mimi_r") if n in elezen.names]
    check(ears and ear_bones and ears[0].weights[:, ear_bones].sum(1).mean() > 0.2,
          "reference: Elezen ears come with their ear bone weights")


def test_game_data():
    """The game's faces read from the game's files, and kept."""
    root = gamefiles.find_game()
    check(root is not None, "game data: the game is found where the launchers put it")
    if root is None:
        return
    check(gamefiles.game_root(os.path.join(root, "game", "sqpack")) == root
          and gamefiles.game_root(os.path.join(root, "game")) == root, "game data: its game and sqpack folders lead to it")
    check(gamefiles.find_game(os.path.join(root, "boot")) == root and gamefiles.find_game(tempfile.gettempdir()) is None,
          "game data: a folder in the game's leads to it, one without the game isn't taken")
    with gamefiles.Index(gamefiles.sqpack_folder(root), "chara") as index:
        parts = facedata.model_parts(index.read(facedata.MODEL_PATH.format("c0201", 1)))
        check(index.read(facedata.MODEL_PATH.format("c0201", 999)) is None, "game data: a file the game doesn't have")
    skin = parts[0]
    check(skin.name.startswith("0.0 ") and "_fac_" in skin.material and len(skin.tris) > 1000
          and np.all(skin.weight_values.astype(int).sum(1) >= 254), "game data: a face model's skin with its weights")
    check(np.allclose(np.linalg.norm(skin.normals[np.unique(skin.tris)], axis=1), 1.0, atol=0.002),
          "game data: the skin's normals as the game has them")
    check(facedata._clean_tris(np.array([[0, 1, 2], [0, 0, 1], [2, 1, 0], [1, 2, 3]])).tolist() == [[0, 1, 2], [1, 2, 3]],
          "game data: degenerate and repeated triangles are dropped, the first kept")
    bones, values = facedata._sorted_weights(np.array([[1, 0, 1, 2]]), np.array([[100, 50, 55, 50]]), ["a", "b", "c"])
    check(bones[0, :3].tolist() == [1, 2, 0] and values[0].tolist() == [155, 50, 50, 0, 0, 0, 0, 0],
          "game data: a bone twice in a vertex adds up, weights sorted by value, then name")
    facing.Reference.get()
    saved = facedata.saved_data()
    built = facedata.build(root)
    check(saved is not None and set(saved) == set(built) and all(np.array_equal(saved[k], built[k]) for k in built),
          "game data: kept as read, for the game's version")


def test_saved_data_rules():
    """When the saved faces are used, read again, or can't be had."""
    real_path, real_folder = facedata.data_path, facedata.game_folder
    folder = tempfile.mkdtemp(prefix="mf-face-")
    path = os.path.join(folder, facedata.FILE_NAME)
    root = gamefiles.find_game()
    with gamefiles.Index(gamefiles.sqpack_folder(root), "chara") as index:
        entry = index.entry(facedata.EST_PATH)
    data = {"x": np.arange(1000), "game_files": np.array([facedata.EST_PATH]), "game_entries": np.array([entry])}
    try:
        facedata.data_path = lambda: path
        facedata.game_folder = lambda: (root, "")
        facedata.clear_cache()
        check(facedata.saved_data() is None, "saved: nothing saved yet")
        facedata._save(data, path, gamefiles.game_version(root))
        check(facedata.saved_data() is not None, "saved: used for the game's version")
        facedata._save(dict(data, game_entries=np.array([entry + 0x10])), path, gamefiles.game_version(root))
        check(facedata.saved_data() is None, "saved: read again when a file it was read from moved (TexTools "
                                             "installed or removed a mod)")
        # A damaged file is read again, not an error in every redraw of the Face panel.
        facedata._save(data, path, gamefiles.game_version(root))
        with open(path, "rb") as f:
            whole = f.read()
        with zipfile.ZipFile(path) as archive:
            member = max(archive.infolist(), key=lambda info: info.compress_size)
        name_size, extra_size = struct.unpack_from("<HH", whole, member.header_offset + 26)
        middle = member.header_offset + 30 + name_size + extra_size + member.compress_size // 2
        for broken, how in ((whole[:len(whole) // 2], "cut short"),
                            (whole[:middle] + bytes(b ^ 0xFF for b in whole[middle:middle + 8]) + whole[middle + 8:],
                             "with damaged data")):
            with open(path, "wb") as f:
                f.write(broken)
            try:
                result = facedata.saved_data()
            except Exception as error:
                result = error
            check(result is None, "saved: a file {:s} is as good as none ({!r})".format(how, result))
        facedata._save(data, path, "2000.01.01.0000.0000")
        check(facedata.saved_data() is None, "saved: read again after a game update")
        facedata.game_folder = lambda: (None, "")
        facedata.clear_cache()
        check(facedata.saved_data() is not None and facedata.problem() is None, "saved: used when the game isn't found")
        os.remove(path)
        facedata.clear_cache()
        check(facedata.problem() == facedata.GAME_NOT_FOUND, "saved: without the game or saved faces, that's the problem")
        facedata.game_folder = lambda: (None, "D:/Games/Nothing")
        facedata.clear_cache()
        check("D:/Games/Nothing" in (facedata.problem() or ""), "saved: names the game folder set when the game isn't there")
        try:
            facedata.load()
            raised = None
        except facedata.FaceDataError as error:
            raised = str(error)
        check(raised is not None and "preferences" in raised, "saved: reading without the game says what to do")
    finally:
        facedata.data_path, facedata.game_folder = real_path, real_folder
        facedata.clear_cache()
        shutil.rmtree(folder, ignore_errors=True)


def test_textools():
    """Files TexTools installed as mods into the game's files are told from the game's own, by the .dat file their
    index entry points into, and the face tools warn about faces read from them."""
    folder = tempfile.mkdtemp(prefix="mf-sqpack-")
    try:
        base = os.path.join(folder, "ffxiv", "040000.win32")
        os.makedirs(os.path.dirname(base))
        paths = ["chara/a/game.mdl", "chara/a/new.mdl", "chara/a/old.mdl", "chara/a/partial.mdl", "chara/a/empty.mdl"]
        index = bytearray(0x800 + 16 * len(paths))
        struct.pack_into("<I", index, 0x0C, 0x400)
        struct.pack_into("<II", index, 0x408, 0x800, 16 * len(paths))
        for number, path in enumerate(paths):
            folder_name, _sep, name = path.rpartition("/")
            struct.pack_into("<III", index, 0x800 + 16 * number, gamefiles.path_hash(name),
                             gamefiles.path_hash(folder_name), number << 1)
            # The headers, then data, except for the game's empty .dat files (its hash is empty too).
            dat = bytearray(0x800 if number == 4 else 0x880)
            if number == 0:  # the game's own: its data's hash filled in
                dat[0x420:0x434] = bytes(range(1, 21))
            elif number in (1, 3):  # TexTools' marks
                struct.pack_into("<ii", dat, 0x200, *([1337] * 2 if number == 1 else [6969] * 2))
            with open(base + ".dat%d" % number, "wb") as f:
                f.write(dat)
        with open(base + ".index", "wb") as f:
            f.write(index)
        with gamefiles.Index(folder, "chara") as sqpack:
            found = [sqpack.from_textools(path) for path in paths]
            check(found == [False, True, True, True, False] and not sqpack.from_textools("chara/a/none.mdl"),
                  "textools: marked .dat files and old TexTools' ones hold mods, the game's don't ({})".format(found))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    data = facedata.saved_data() or facedata.load()
    model = facedata.MODEL_PATH.format("c1401", 1)
    reference = facing.Reference(dict(data, textools_files=np.array([model])))
    warning = operators.textools_warning(reference, "c1401", 101) or ""
    check(reference.textools_files("c1401", 101) == [model] and "TexTools" in warning
          and model.rsplit("/", 1)[1] in warning and operators.textools_warning(reference, "c0201", 1) is None,
          "textools: a face read from a TexTools mod is warned about (Au Ra face 101 uses face 1's model)")


def test_first_use():
    """With no faces saved, Face Weights reads them from the game first, and saves them."""
    face.clear_scene()
    use_settings()
    objects = face.build()
    face.select(objects.values(), objects["skin"])
    real_path = facedata.data_path
    folder = tempfile.mkdtemp(prefix="mf-face-")
    path = os.path.join(folder, facedata.FILE_NAME)
    try:
        facedata.data_path = lambda: path
        facing.Reference.forget()
        check(facing.Reference.loaded() is None and operators.check_face_tools_ready(bpy.context) is None,
              "first use: nothing saved, and the tools are ready")
        check(bpy.ops.magic_fit.face_weights() == {'FINISHED'} and os.path.exists(path)
              and facing.Reference.loaded() is not None, "first use: Face Weights reads the game's faces and saves them")
        check(armature_of(objects["skin"]) is not None, "first use: ...and weights the face")
    finally:
        facedata.data_path = real_path
        facing.Reference.forget()
        shutil.rmtree(folder, ignore_errors=True)


def test_features():
    data = face.reference()
    features = data.features()
    check(sorted(features.eyes) == ['l', 'r'] and features.lip is not None and features.ring is not None,
          "features: both eye openings, the lip line and the neck ring of the game's face")
    upper, lower, eye = features.eyes['l']
    check(np.all(facing.polyline_z(upper, upper[:, 0]) >= facing.polyline_z(lower, upper[:, 0]) - 1e-4),
          "features: the upper margin is above the lower one")
    for lid in (0, 1):
        line = features.lash_line('l', lid)
        d = facing.polyline_parameter(features.eyes['l'][lid], line)[1]
        check(0.0003 < np.median(d) < facing.LASH_EDGE_REACH,
              "features: the {:s} lash line is the lid's front edge, near the margin ({:.2f} mm)".format(
                  "upper" if lid == 0 else "lower", np.median(d) * 1000))
    # A heavy brow overhangs the lid: the lid's own edge is found all the same.
    brow = face.reference('c1301', 2).features()
    d = facing.polyline_parameter(brow.eyes['l'][0], brow.lash_line('l', 0))[1]
    check(np.median(d) < 0.003, "features: an Au Ra male's upper lash line isn't the brow ({:.2f} mm)".format(
        np.median(d) * 1000))


def test_closure_box():
    """Closure checks cast rays only around the opening at rest, and find the openings a scan of the whole
    window finds; an opening reaching the box's edge is scanned in full."""
    data = face.reference()
    features = data.features()
    upper, lower, eye = features.eyes['l']
    box = features.opening_box('l')
    check(box[0] < min(upper[:, 0].min(), lower[:, 0].min()) and box[1] > max(upper[:, 0].max(), lower[:, 0].max())
          and box[2] < lower[:, 2].min() and box[3] > upper[:, 2].max(), "closure box: around the lid margins")
    args = (data.points, data.tris, data.names, data.weights, eye, 'l', data.skeleton, 'SHUT_EYES')
    for share in (0.0, 0.5, 0.85):
        whole = facing.posed_opening(*args, share)
        boxed = facing.posed_opening(*args, share, box=box)
        check(whole[3].any() and np.array_equal(whole[0], boxed[0]) and np.array_equal(whole[1], boxed[1])
              and np.array_equal(whole[3], boxed[3]) and np.isnan(boxed[2]).sum() > np.isnan(whole[2]).sum(),
              "closure box: fewer rays find the same opening at {:.0%} of the Shut Eyes expression ({:d} cells "
              "open)".format(share, int(whole[3].sum())))
    x, z = eye.center[0], eye.center[2]
    whole = facing.posed_opening(*args, 0.0)
    boxed = facing.posed_opening(*args, 0.0, box=(x - 0.001, x + 0.001, z - 0.001, z + 0.001))
    check(np.array_equal(whole[3], boxed[3]) and np.array_equal(whole[2], boxed[2], equal_nan=True),
          "closure box: an opening reaching past a small box is scanned in full")


def test_face_weights():
    face.clear_scene()
    settings = use_settings()
    objects = face.build()
    skin = objects["skin"]
    face.select(objects.values(), skin)
    check(operators.check_face_tools_ready(bpy.context) is None, "face weights: ready with the face selected")
    check(bpy.ops.magic_fit.face_weights() == {'FINISHED'}, "face weights: runs")
    check(settings.face_found == "c0201 1" and settings.face_race == 'c0201' and settings.face_number == 1,
          "face weights: found the game's face by name")
    armature = armature_of(skin)
    check(armature is not None and armature.name.startswith("Face Skeleton"), "face weights: made a face skeleton")
    check(armature is not None and all(armature.data.bones.get(n) for n in ("j_kao", "j_f_mabup_01_l", "j_f_eyepuru_r")),
          "face weights: ...with the head, lid and eyeball bones")
    data = face.reference()
    weights = face.weights_of(skin, data.names)
    error = 0.5 * np.abs(weights - data.weights).sum(1)
    check(error.mean() < 0.005 and np.percentile(error, 99) < 0.05,
          "face weights: the game's own face gets the game's weights back (mean {:.4f}, p99 {:.4f})".format(
              error.mean(), np.percentile(error, 99)))
    for side in 'lr':
        eyeball = face.weights_of(objects["eye_" + side], ["j_f_eyepuru_" + side])
        check(np.allclose(eyeball, 1.0), "face weights: the {:s} eyeball follows its eyeball bone".format(side))
    etc = objects["etc"]
    names = [g.name for g in etc.vertex_groups]
    lids = face.weights_of(etc, names)[:, column(names, facing.UPPER_LID)]
    check(lids.sum(1).max() > 0.5, "face weights: the lashes follow the upper lids")
    check(all(armature_of(obj) is armature for obj in objects.values()), "face weights: every part got the armature")


def shut_share(objects, skin, data):
    """The share of the Shut Eyes expression by which the lids of ``skin`` (weights as Blender has them) cover
    the eyeballs among ``objects`` (see facing.closing_share), or None."""
    world = [facing.object_mesh(obj) for obj in objects.values()]
    index = list(objects.values()).index(skin)
    eyes = facing.find_eyes(world, data.skeleton, data.radius, skip=index)
    mesh = world[index]
    return facing.closing_share(mesh.points, mesh.tris, mesh.names, mesh.weights, eyes, data.skeleton)


def posed_open_areas(objects, skin):
    """{side: mm² of the eyeball showing through ``skin``} as Blender displays them."""
    bvh = facing.bvh_of(posed_points(skin), facing.mesh_tris(skin.data))
    areas = {}
    for side in 'lr':
        eyeball = objects["eye_" + side]
        points = posed_points(eyeball)
        eye = facing.Eye(points.mean(0), float(np.linalg.norm(points - points.mean(0), axis=1).mean()), points,
                         facing.mesh_tris(eyeball.data), True)
        areas[side] = facing.open_area(bvh, eye)
    return areas


def test_shut_eyes():
    """The whole chain in Blender: weights, armature and the game's Shut Eyes expression shut the eyes of a face
    whose eyes were opened wider than the game's, the lids just meeting there; the blink's peak closes them too."""
    face.clear_scene()
    use_settings()
    objects = face.build()
    skin = objects["skin"]
    data = face.reference()
    face.reshape(skin, face.widen_eyes(data))
    face.select(objects.values(), skin)
    check(bpy.ops.magic_fit.face_weights() == {'FINISHED'}, "shut eyes: Face Weights on wider eyes")
    share = shut_share(objects, skin, data)
    check(share is not None and 1.0 - facing.CLOSE_SLACK - facing.CLOSE_PRECISION <= share <= 1.0,
          "shut eyes: the lids meet in the Shut Eyes expression, not much before ({:.2f})".format(share or 0.0))
    armature = armature_of(skin)
    for pose, label in (('SHUT_EYES', "Shut Eyes"), ('BLINK', "Blink")):
        check(bpy.ops.magic_fit.face_pose(pose=pose) == {'FINISHED'} and
              abs(armature.pose.bones["j_f_mabup_01_l"].matrix_basis.to_quaternion().angle) > 0.5,
              "shut eyes: the {:s} test pose turns the upper lid bone".format(label))
        for side, area in sorted(posed_open_areas(objects, skin).items()):
            check(area < 1.0, "shut eyes: the {:s} eye is closed in Blender in the {:s} test pose ({:.2f} mm² "
                              "open)".format(side, label, area))
    check(bpy.ops.magic_fit.face_pose(pose='REST') == {'FINISHED'} and
          armature.pose.bones["j_f_mabup_01_l"].matrix_basis.to_quaternion().angle < 1e-6,
          "shut eyes: Rest puts the lids back")


def test_small_eyes():
    """A face whose eyes are narrower than the game's: the game's lid weights would close them long before the
    Shut Eyes expression is reached, and slide the lids on past each other and fold them there. Face Weights
    lowers them, so the lids just meet in it."""
    face.clear_scene()
    settings = use_settings()
    objects = face.build()
    skin = objects["skin"]
    data = face.reference()
    face.reshape(skin, face.widen_eyes(data, -0.004))
    job = facing.setup(list(objects.values()), settings)
    plan = facing.plan_face(job.meshes, job.skin, job.face, targets=job.parts)
    for side in 'lr':
        eye = plan.report["eyes"].get(side, {})
        early = eye.get("early")
        check(early is not None and early < 1.0 - facing.CLOSE_SLACK and eye["lowered"] < 0.95,
              "small eyes: the game's lid weights close the {:s} eye too early ({:.2f}), so they're lowered ({:.2f})".format(
                  side, early if early is not None else 0.0, eye.get("lowered", 1.0)))
    face.select(objects.values(), skin)
    check(bpy.ops.magic_fit.face_weights() == {'FINISHED'}, "small eyes: Face Weights")
    share = shut_share(objects, skin, data)
    check(share is not None and 1.0 - facing.CLOSE_SLACK - facing.CLOSE_PRECISION <= share <= 1.0,
          "small eyes: the lids meet in the Shut Eyes expression, not much before ({:.2f})".format(share or 0.0))
    for pose, label in (('SHUT_EYES', "Shut Eyes"), ('BLINK', "Blink")):
        check(bpy.ops.magic_fit.face_pose(pose=pose) == {'FINISHED'}, "small eyes: the {:s} test pose".format(label))
        for side, area in sorted(posed_open_areas(objects, skin).items()):
            check(area < 1.0, "small eyes: the {:s} eye is closed in the {:s} test pose ({:.2f} mm² open)".format(
                side, label, area))


def test_blink_peak():
    """The blink's peak must close the eyes too, but for a hairline slit: on the game's Lalafell female face 1, lids
    fitted to the Shut Eyes expression alone would leave the blink's peak half a millimetre open."""
    data = face.reference('c1201', 1)
    meshes = [facing.Mesh("skin", data.points, data.tris, [], np.zeros((len(data.points), 0)))]
    for side, eye in data.eyes().items():
        points, tris = face.sphere(eye.center, eye.radius * 0.97)
        meshes.append(facing.Mesh("Eyeball_" + side, points, tris, [], np.zeros((len(points), 0))))
    plan = facing.plan_face(meshes, 0, data, parts='NONE')
    eyes = facing.find_eyes(meshes, data.skeleton, data.radius, skip=0)
    features = facing.Features(data.points, data.tris, data.skeleton, eyes)
    boxes = {side: features.opening_box(side) for side in features.eyes}
    share = facing.closing_share(data.points, data.tris, plan.names, plan.goals[0], eyes, data.skeleton, boxes)
    check(any(eye["lowered"] < 1.0 for eye in plan.report["eyes"].values()) and share is not None
          and 1.0 - facing.CLOSE_SLACK - facing.CLOSE_PRECISION <= share <= 1.0,
          "blink peak: lids are lowered, and meet in the Shut Eyes expression, not much before ({:.2f})".format(
              share or 0.0))
    for side, eye in sorted(eyes.items()):
        xs, zs, _hits, mask = facing.posed_opening(data.points, data.tris, plan.names, plan.goals[0], eye, side,
                                                   data.skeleton, 'BLINK')
        gap = float(mask.sum(0).max() * (zs[1] - zs[0])) if mask.any() else 0.0
        check(gap <= facing.BLINK_SLIT, "blink peak: the {:s} eye is closed at the blink's peak but for a slit "
                                        "({:.2f} mm)".format(side, gap * 1000.0))


def test_neck():
    face.clear_scene()
    use_settings()
    objects = face.build(extras=False, eyes=False)
    skin = objects["skin"]
    data = face.reference()
    ring_points, ring_normals, _ring_weights = data.ring()
    _welded, _loop, verts = facing.neck_ring(data.points, data.tris)
    # A custom face whose neck opening moved half a millimetre, with smooth normals, and a shape key.
    for i in verts:
        skin.data.vertices[i].co.z += 0.0005
    skin.data.update()
    skin.shape_key_add(name="Basis")
    key = skin.shape_key_add(name="Wide")
    for point in key.data:
        point.co.x *= 1.1
    face.select([skin], skin)
    check(bpy.ops.magic_fit.face_repair(repair='NECK') == {'FINISHED'}, "neck: Match Neck runs")
    points = face.points_of(skin)
    check(np.abs(points[verts] - data.points[verts]).max() < 1e-6,
          "neck: the opening is back where the game's is ({:.2e} m off)".format(np.abs(points[verts] - data.points[verts]).max()))
    mesh = skin.data
    normals = np.array([mesh.corner_normals[loop.index].vector for loop in mesh.loops])
    loop_verts = np.array([loop.vertex_index for loop in mesh.loops])
    lookup = {int(v): k for k, v in enumerate(verts)}
    worst = 0.0
    for normal, vert in zip(normals, loop_verts):
        if int(vert) in lookup:
            want = data.normals[vert]
            worst = max(worst, float(np.degrees(np.arccos(np.clip(np.dot(normal, want) / np.linalg.norm(want), -1, 1)))))
    check(worst < 0.5, "neck: the opening has the game's normals ({:.2f} degrees off at most)".format(worst))
    weights = face.weights_of(skin, data.names)
    check(np.abs(weights[verts] - data.weights[verts]).max() < 0.01, "neck: ...and the game's weights")
    upper = column(data.names, facing.UPPER_LID)
    check(weights[:, upper].sum() == 0.0, "neck: Match Neck leaves the rest of the face alone")
    wide = np.array([point.co[:] for point in skin.data.shape_keys.key_blocks["Wide"].data])
    basis = np.array([point.co[:] for point in skin.data.shape_keys.key_blocks["Basis"].data])
    check(np.allclose(wide[:, 0], basis[:, 0] * 1.1, atol=1e-6) and np.allclose(wide[:, 1:], basis[:, 1:], atol=1e-6),
          "neck: the shape key moved along and still does what it did")


def test_moved_face():
    """A face and its armature moved together away from where the game has them."""
    face.clear_scene()
    use_settings()
    objects = face.build()
    skin = objects["skin"]
    face.select(objects.values(), skin)
    check(bpy.ops.magic_fit.face_weights() == {'FINISHED'}, "moved: Face Weights")
    armature = armature_of(skin)
    shift = np.array((0.4, -0.3, 0.2))
    for obj in list(objects.values()) + [armature]:
        obj.location = tuple(np.array(obj.location) + shift)
    for obj in objects.values():
        for group in list(obj.vertex_groups):
            obj.vertex_groups.remove(group)
    bpy.context.view_layer.update()
    face.select(objects.values(), skin)
    check(bpy.ops.magic_fit.face_weights() == {'FINISHED'}, "moved: Face Weights again, all of it moved")
    data = face.reference()
    error = 0.5 * np.abs(face.weights_of(skin, data.names) - data.weights).sum(1)
    check(error.mean() < 0.005, "moved: the same weights as in place ({:.4f})".format(error.mean()))
    check(bpy.ops.magic_fit.face_pose(pose='SHUT_EYES') == {'FINISHED'}, "moved: the Shut Eyes pose")
    area = posed_open_areas(objects, skin)['l']
    check(area < 1.0, "moved: ...shuts the eyes where the face is now ({:.2f} mm² open)".format(area))


def test_repairs():
    face.clear_scene()
    use_settings()
    objects = face.build()
    skin = objects["skin"]
    data = face.reference()
    for obj in objects.values():
        weigh_all(obj)
    face.select(objects.values(), skin)
    features = data.features()
    points = data.points
    near_eyes = facing.region_mask(points, [m for u, l, _e in features.eyes.values() for m in (u, l)], 0.002, 0.003) > 0.99
    near_mouth = facing.region_mask(points, [features.lip], 0.003, 0.004) > 0.99
    check(bpy.ops.magic_fit.face_repair(repair='EYES') == {'FINISHED'}, "repairs: Fix Eyelids runs")
    weights = face.weights_of(skin, data.names)
    lids = column(data.names, facing.UPPER_LID, facing.LOWER_LID)
    mouth = column(data.names, *facing.UPPER_MOUTH, *facing.LOWER_MOUTH)
    check(weights[near_eyes][:, lids].sum(1).mean() > 0.5, "repairs: Fix Eyelids weights the lids")
    check(weights[:, mouth].sum() == 0.0, "repairs: ...and leaves the mouth alone")
    etc_names = [g.name for g in objects["etc"].vertex_groups]
    check(any(n.startswith(facing.UPPER_LID) for n in etc_names), "repairs: ...and the lashes follow the lids")
    check(face.weights_of(objects["eye_l"], ["j_kao"]).min() == 1.0, "repairs: ...and leaves the eyeballs alone")
    check(bpy.ops.magic_fit.face_repair(repair='MOUTH') == {'FINISHED'}, "repairs: Fix Mouth runs")
    weights = face.weights_of(skin, data.names)
    check(weights[near_mouth][:, mouth].sum(1).mean() > 0.5, "repairs: Fix Mouth weights the lips")
    # The back of the head at the mouth's height, behind the neck bone, is no part of the mouth.
    lip = features.lip
    back = ((points[:, 1] > data.skeleton.head(facing.NECK)[1]) & (np.abs(points[:, 0]) < np.abs(lip[:, 0]).max())
            & (np.abs(points[:, 2] - lip[:, 2].mean()) < 0.02))
    check(back.any() and np.all(weights[back][:, data.names.index("j_kao")] == 1.0),
          "repairs: ...and leaves the back of the head alone ({:d} vertices)".format(int(back.sum())))
    check(bpy.ops.magic_fit.face_repair(repair='PARTS') == {'FINISHED'}, "repairs: Attach Parts runs")
    check(np.allclose(face.weights_of(objects["eye_r"], ["j_f_eyepuru_r"]), 1.0),
          "repairs: Attach Parts weights the eyeballs")
    check(np.allclose(face.weights_of(skin, data.names), weights), "repairs: ...and leaves the face alone")


def test_repairs_need_features():
    """Fix Eyelids and Fix Mouth fit to the eye openings and the lip line: when they aren't found, they say so and
    change nothing."""
    face.clear_scene()
    use_settings()
    skin = face.build(extras=False, eyes=False)["skin"]
    data = face.reference()
    center = data.features().mouth_center

    def seal(points):
        """The lids pulled flat in front of the eyeballs, and the lips flat: no eye opening, no lip line."""
        out = points.copy()
        for eye in data.eyes().values():
            front = ((np.abs(points[:, 0] - eye.center[0]) < 0.02) & (np.abs(points[:, 2] - eye.center[2]) < 0.015)
                     & (points[:, 1] < eye.center[1] + eye.radius))
            out[front, 1] = eye.center[1] - eye.radius - 0.004
        front = ((np.abs(points[:, 0] - center[0]) < 0.04) & (np.abs(points[:, 2] - center[2]) < 0.02)
                 & (points[:, 1] < center[1] + 0.005))
        out[front, 1] = center[1] - 0.02
        return out

    face.reshape(skin, seal)
    weigh_all(skin)
    face.select([skin], skin)
    for repair, words in (('EYES', "eye openings"), ('MOUTH', "line between the lips")):
        try:
            result = bpy.ops.magic_fit.face_repair(repair=repair)
        except RuntimeError as error:
            result = str(error)
        check(words in str(result) and armature_of(skin) is None and [g.name for g in skin.vertex_groups] == ["j_kao"]
              and np.all(face.weights_of(skin, ["j_kao"]) == 1.0),
              "repairs without features: {:s} says it couldn't find the {:s} and changes nothing".format(repair, words))


def test_body_selected():
    """A body in one piece has the neck opening too, and more vertices than the face: the face is still the skin."""
    face.clear_scene()
    settings = use_settings()
    skin = face.build(extras=False, eyes=False)["skin"]
    data = face.reference()
    welded, loop, _verts = facing.neck_ring(data.points, data.tris)
    ring = welded.points[loop]
    rows, n, step = 300, len(ring), 0.0025
    points = np.vstack([ring - (0.0, 0.0, step * k) for k in range(rows)] + [ring.mean(0) - (0.0, 0.0, step * rows)])
    tris = []
    for k in range(rows - 1):
        for j in range(n):
            a, b = k * n + j, k * n + (j + 1) % n
            tris += [(a, a + n, b + n), (a, b + n, b)]
    tris += [((rows - 1) * n + j, rows * n, (rows - 1) * n + (j + 1) % n) for j in range(n)]
    body = face._object("Body", points, np.array(tris))
    job = facing.setup([body, skin], settings)
    check(len(body.data.vertices) > len(skin.data.vertices) and job.objects[job.skin] is skin,
          "body selected: the face is the skin, not the bigger body with the same neck opening")


def test_masks():
    """Groups that mask something (a modifier's, a shape key's) hold no bone weights: the repairs leave them as
    they are, and the bone weights still add up to 1."""
    face.clear_scene()
    use_settings()
    skin = face.build(extras=False, eyes=False)["skin"]
    face.select([skin], skin)
    check(bpy.ops.magic_fit.face_weights() == {'FINISHED'}, "masks: Face Weights")
    skin.vertex_groups.new(name="Mask").add(range(len(skin.data.vertices)), 1.0, 'REPLACE')
    skin.modifiers.new("Displace", 'DISPLACE').vertex_group = "Mask"
    for repair in ('MOUTH', 'EYES', 'NECK'):
        check(bpy.ops.magic_fit.face_repair(repair=repair) == {'FINISHED'}, "masks: repair {:s}".format(repair))
        names = [g.name for g in skin.vertex_groups]
        weights = face.weights_of(skin, names)
        bones = [j for j, name in enumerate(names) if name != "Mask"]
        check(np.all(weights[:, names.index("Mask")] == 1.0) and np.allclose(weights[:, bones].sum(1), 1.0, atol=0.01),
              "masks: {:s} leaves the mask group alone and the bone weights add up to 1 ({:.2f} to {:.2f})".format(
                  repair, weights[:, names.index("Mask")].min(), weights[:, bones].sum(1).min()))


def test_snap_lashes():
    face.clear_scene()
    use_settings()
    objects = face.build()
    skin, etc = objects["skin"], objects["etc"]
    face.select(objects.values(), skin)
    check(bpy.ops.magic_fit.face_weights() == {'FINISHED'}, "snap: Face Weights first")
    before = face.points_of(etc)
    # The game's own lashes sit where lashes belong: snapping leaves them there.
    check(bpy.ops.magic_fit.face_repair(repair='LASHES') == {'FINISHED'}, "snap: Snap Lashes runs on the game's lashes")
    moved = np.linalg.norm(face.points_of(etc) - before, axis=1)
    check(moved.max() < 0.0001, "snap: the game's own lashes stay where they are ({:.2f} mm moved at most)".format(
        moved.max() * 1000))
    face.reshape(etc, lambda p: before)
    # The lashes pushed 1.5 mm into the head, as badly placed custom lashes are.
    face.reshape(etc, lambda p: p + np.array((0.0, 0.0015, 0.0)))
    pushed = face.points_of(etc)
    check(bpy.ops.magic_fit.face_repair(repair='LASHES') == {'FINISHED'}, "snap: Snap Lashes runs")
    after = face.points_of(etc)
    lashes = np.linalg.norm(after - pushed, axis=1) > 1e-7
    off = np.linalg.norm(after - before, axis=1)
    check(lashes.any() and np.median(off[lashes]) < 0.0013,
          "snap: the lashes are closer to where the game has them than the 1.5 mm they were pushed ({:.2f} mm "
          "off, median)".format(np.median(off[lashes]) * 1000))
    check(lashes.any() and not lashes.all(), "snap: what isn't lashes (brows, the tear lines) stays where it is")
    features = face.reference().features()
    sunk = facing.signed_skin_distance(features.bvh, pushed[lashes])[0] < -0.0001
    still = facing.signed_skin_distance(features.bvh, after[lashes])[0] < -0.0001
    check(lashes.any() and still.mean() < 0.25 * sunk.mean(),
          "snap: the lashes come out of the head ({:.0f} % of them inside it, pushed: {:.0f} %)".format(
              still.mean() * 100, sunk.mean() * 100))
    mesh = facing.Mesh("etc", after, facing.mesh_tris(etc.data), [], np.zeros((len(after), 0)))
    distances, depths = [], []
    for part in range(mesh.part_count):
        verts = np.flatnonzero(mesh.parts == part)
        if lashes[verts].any():
            side = 'l' if after[verts, 0].mean() > 0 else 'r'
            roots = facing.LashRoots(after[verts], facing.part_tris(mesh, verts), features, side)
            own = np.unique(roots.root)
            upper = own[roots.lid[own] == 0]
            distances += roots.d[upper].tolist()
            depths += facing.signed_skin_distance(features.bvh, after[verts][upper])[0].tolist()
    # Pushed straight back, the lashes come out of the lid along its normal, on to its rim by the front edge (the
    # game's own grow 0.2 mm from it, median of all its faces), sitting on the skin.
    check(distances and np.median(distances) < 0.0013,
          "snap: the upper lashes' roots sit by the lids' front edges ({:.2f} mm, median)".format(
              np.median(distances) * 1000))
    check(depths and abs(np.median(depths)) < 0.00015,
          "snap: the upper lashes' roots sit on the lids ({:+.2f} mm off the skin, median)".format(
              np.median(depths) * 1000))


def test_detect():
    face.clear_scene()
    settings = use_settings()
    objects = face.build('c0801', 2, extras=False, eyes=False)
    skin = objects["skin"]
    face.select([skin], skin)
    check(bpy.ops.magic_fit.detect_face() == {'FINISHED'} and settings.face_found == "c0801 2",
          "detect: by name")
    skin.name = "Head"
    skin.data.name = "Head"
    settings.face_found = ""
    check(bpy.ops.magic_fit.detect_face() == {'FINISHED'} and settings.face_found == "c0801 2",
          "detect: by shape ({:s})".format(settings.face_found))
    real_weights = facing.object_weights
    facing.object_weights = None  # Detect (and Test Pose with it) reads no weights
    try:
        found = facing.detect([skin])
    except TypeError as error:
        found = error
    finally:
        facing.object_weights = real_weights
    check(found == ('c0801', 2, 'SHAPE'), "detect: from the meshes' shapes alone ({})".format(found))
    use_settings(face_source='PICK', face_race='c0801', face_number=3)
    job = facing.setup([skin], bpy.context.scene.magic_fit)
    check(job.face.number == 3 and job.how == 'PICK', "detect: Pick uses the face picked")
    items = operators._face_items(None, bpy.context)
    check([item[0] for item in items][:3] == ["1", "2", "3"], "detect: the picker lists the race's faces")
    # Faces read again (or from another game folder) are listed, not the ones listed before.
    real_loaded = facing.Reference.__dict__["loaded"]
    other = facing.Reference(dict(facedata.saved_data(), face_id=np.full(len(facedata.saved_data()["face_id"]), 7)))
    facing._goal_cache["brush goals"] = None
    try:
        facing.Reference.loaded = classmethod(lambda cls: other)
        facing.Reference.forget()
        items = operators._face_items(None, bpy.context)
    finally:
        facing.Reference.loaded = real_loaded
        facing.Reference.forget()
    check([item[0] for item in items] == ["7"] and not facing._goal_cache,
          "detect: the picker lists the faces read again ({}), and the brush forgets its goals".format(
              [item[0] for item in items]))
    bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0.0, 0.0, 0.5))
    cube = bpy.context.object
    use_settings()
    face.select([cube], cube)
    try:
        result = bpy.ops.magic_fit.detect_face()
    except RuntimeError as error:  # operators reporting an error raise it from bpy.ops
        result = str(error)
    check("Couldn't tell" in str(result), "detect: a cube is no face")


def test_lashes_alone():
    """Lashes selected without the face aren't taken for it."""
    face.clear_scene()
    use_settings()
    objects = face.build()
    etc = objects["etc"]
    face.select([etc], etc)
    try:
        result = bpy.ops.magic_fit.face_weights()
    except RuntimeError as error:
        result = str(error)
    check("select the face" in str(result) and not etc.vertex_groups,
          "lashes alone: Face Weights asks for the face and leaves the lashes as they are")


def test_left_out():
    face.clear_scene()
    use_settings()
    objects = face.build(extras=False, eyes=False)
    skin = objects["skin"]
    hair_points, hair_tris = face.sphere((0.0, 0.0, 1.6), 0.1)
    hair = face._object("Hair", hair_points, hair_tris)
    weigh_all(hair, "j_kami_a")
    face.select([skin, hair], skin)
    check(bpy.ops.magic_fit.face_weights() == {'FINISHED'}, "left out: Face Weights with hair selected too")
    check([g.name for g in hair.vertex_groups] == ["j_kami_a"], "left out: the hair keeps its hair weights")


def test_brush():
    face.clear_scene()
    settings = use_settings()
    objects = face.build(extras=False, eyes=False)
    skin = objects["skin"]
    face.select([skin], skin)
    check(operators.check_face_ready(skin, settings) is not None, "brush: needs Face Weights first (no armature)")
    check(bpy.ops.magic_fit.face_weights() == {'FINISHED'}, "brush: Face Weights")
    weigh_all(skin)
    for group in list(skin.vertex_groups):
        if group.name != "j_kao":
            skin.vertex_groups.remove(group)
    check(operators.check_face_ready(skin, settings) is None, "brush: ready")
    stroke = facing.build_stroke(bpy.context, skin, settings)
    data = face.reference()
    near = np.flatnonzero(data.weights[:, column(data.names, facing.UPPER_LID)].sum(1) > 0.5)
    stroke.apply(near, np.ones(len(near)))
    weights = face.weights_of(skin, data.names)
    check(weights[near][:, column(data.names, facing.UPPER_LID)].sum(1).mean() > 0.4,
          "brush: painting brings back the lid weights where painted")
    far = np.setdiff1d(np.arange(len(data.points)), near)
    check(weights[far][:, column(data.names, facing.UPPER_LID)].sum() == 0.0, "brush: ...and nowhere else")


def test_readiness():
    face.clear_scene()
    settings = use_settings()
    objects = face.build(extras=False, eyes=False)
    skin = objects["skin"]
    face.select([], None)
    check("Select the face" in (operators.check_face_tools_ready(bpy.context) or ""), "ready: nothing selected")
    face.select([skin], skin)
    settings.group_mode = 'ACTIVE'
    check("All Groups" in (operators.check_face_tools_ready(bpy.context) or ""), "ready: not in Current Group mode")
    settings.group_mode = 'ALL'
    use_settings(face_source='PICK', face_race='c0201', face_number=77)
    check("no face" in (operators.check_face_tools_ready(bpy.context) or ""), "ready: a face the game doesn't have")
    use_settings()
    bpy.ops.object.mode_set(mode='EDIT')
    check("Object or Weight Paint" in (operators.check_face_tools_ready(bpy.context) or ""), "ready: not in Edit Mode")
    bpy.ops.object.mode_set(mode='OBJECT')
    check("Face Weights first" in (operators.check_face_pose_ready(bpy.context) or ""),
          "ready: Test Poses need the face's armature")


def main():
    magic_fit.register()
    try:
        for test in (test_reference, test_game_data, test_saved_data_rules, test_textools, test_first_use, test_features, test_closure_box, test_face_weights, test_shut_eyes, test_small_eyes,
                     test_blink_peak, test_neck, test_moved_face,
                     test_repairs, test_repairs_need_features, test_body_selected, test_masks, test_snap_lashes,
                     test_detect, test_lashes_alone, test_left_out, test_brush, test_readiness):
            print("==", test.__name__)
            try:
                test()
            except Exception:
                traceback.print_exc()
                FAILURES.append(test.__name__ + ": exception")
    finally:
        magic_fit.unregister()
    print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
    sys.stdout.flush()
    os._exit(1 if FAILURES else 0)


main()
