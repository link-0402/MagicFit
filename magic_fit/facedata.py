# SPDX-License-Identifier: GPL-3.0-or-later

"""The game's faces, face skeletons and face poses Face needs (facing.Reference), read from the user's game
install (see gamefiles) the first time Face needs them and kept in the add-on's user folder. No game model
comes with the add-on.

Read per face of the player races (see MAX_FACE; each face faceskeletontemplate.est gives a face skeleton and
the game has a model of; Au Ra and Viera faces 101-104 have none and use faces 1-4): its model's parts but the
eyes, from the first level of detail, one per submesh, as they are in the game file: positions (float32, in
Blender's space: Z up, facing -Y, metres), normals (float16, the game's own), triangles and up to 8 bone
weights (bytes); which part is the skin (mesh 0, submesh 0: face, neck, mouth inside), which parts use the
skin's kind of material (fac), and their attributes; and the eyeball radius (the eye meshes are spheres around
the eye bones). Per race and face skeleton: the bones (the body's from j_kao down to the root, j_kubi, j_sebo_c
and the ears' j_mimi, and the face skeleton's), their local rest transforms in game space and their rest
matrices in Blender space; per pose, the additive changes to the bones' local transforms at the peak frame of
the game's face animation.

A part keeps every vertex between its first and last one, used or not, and its weights sorted by value and
then by bone name (descending), as Instant Edit's importer gives them to Blender, where Face's measurements
were made.

The data is read again when the game's version (ffxivgame.ver) or DATA_VERSION changes, or when the index
entries of the files it was read from do (TexTools installs and removes mods by pointing them elsewhere). The
files read from mods TexTools installed are listed.
"""

import contextlib
import io
import os
import struct
import time
import zipfile
import zlib

import numpy as np

from . import gamefiles

# Raise when what's read or how changes: the data kept in the user folder is read again.
DATA_VERSION = 3
FILE_NAME = "face_data.npz"
LEGACY_FOLDER = "magic_fit"

# Game model space (Y up, facing +Z) to Blender (Z up, facing -Y).
TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
# Body bones faces use (ears: j_mimi), and every ancestor of them.
BODY_BONES = ("j_kao", "j_kubi", "j_sebo_c", "j_mimi_l", "j_mimi_r")
MAX_WEIGHTS = 8
# The poses kept: (name, the face.pap animation). Each is kept at its peak: the frame the bones turn most.
# SHUT_EYES is the Shut Eyes expression (the Emote sheet's row 73 plays ActionTimeline facial/pose/bow, whose
# .tmb plays cfxf_bow from face.pap; the Bow emote's face, nonresident/emot/bow.pap, is the same animation): a
# pose the game holds as long as the expression lasts.
POSES = (
    ("BLINK", "cfxb_blink1"),
    ("SHUT_EYES", "cfxf_bow"),
    ("TALK", "cfxl_lip_nor2"),
    ("SHOUT", "cfxl_lip_sho2"),
    ("ANGRY", "cfxf_angry"),
    ("CLENCH", "cfxf_clench"),
    ("OUCH", "cfxf_ouch_st"),
    ("DIE", "cfxf_die"),
    ("SALUTE", "cfxf_salute"),
)
# The faces read: those of the player races (race codes ending in 01) numbered up to 104, which are the faces
# players wear (the clans wearing the models 100 above their face number included) and the NPC faces 91 and 92
# in the same folders. The other NPC faces there (191, 192, from 200 up) and the NPC races are left out.
PLAYER_RACE_SUFFIX = "01"
MAX_FACE = 104
EST_PATH = "chara/xls/charadb/faceskeletontemplate.est"
MODEL_PATH = "chara/human/{0}/obj/face/f{1:04d}/model/{0}f{1:04d}_fac.mdl"
BASE_SKELETON_PATH = "chara/human/{0}/skeleton/base/b0001/skl_{0}b0001.sklb"
FACE_SKELETON_PATH = "chara/human/{0}/skeleton/face/f{1:04d}/skl_{0}f{1:04d}.sklb"
FACE_ANIMATIONS_PATH = "chara/human/{0}/animation/f{1:04d}/resident/face.pap"
# The vertex elements read: (xivpy's VertexUsage name, field).
ELEMENTS = {"POSITION": "position", "NORMAL": "normal", "BLEND_WEIGHTS": "weights", "BLEND_INDICES": "bones"}
# numpy types of xivpy's VertexType names; weights and bone indices of 8 bones per vertex use USHORT4.
ELEMENT_TYPES = {"SINGLE1": ("<f4", 1), "SINGLE2": ("<f4", 2), "SINGLE3": ("<f4", 3), "SINGLE4": ("<f4", 4),
                 "UBYTE4": ("u1", 4), "NBYTE4": ("u1", 4), "SHORT2": ("<i2", 2), "SHORT4": ("<i2", 4),
                 "NSHORT2": ("<i2", 2), "NSHORT4": ("<i2", 4), "HALF2": ("<f2", 2), "HALF4": ("<f2", 4),
                 "USHORT2": ("<u2", 2), "USHORT4": ("<u2", 4)}


class FaceDataError(Exception):
    """The game's faces can't be read: the message says why and what to do."""


GAME_NOT_FOUND = ("Face reads the game's faces from your FFXIV install, but it wasn't found: set the game "
                  "folder in Magic Fit's preferences (Edit > Preferences > Add-ons)")
GAME_NOT_AT = ("Face reads the game's faces from your FFXIV install, but there's no game in '{:s}': fix the "
               "game folder in Magic Fit's preferences (Edit > Preferences > Add-ons)")


# -----------------------------------------------------------------------------
# Skeletons and poses

def quat_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def model_matrices(parents, local):
    out = np.zeros((len(parents), 4, 4))
    for i, row in enumerate(local):
        m = np.eye(4)
        m[:3, :3] = quat_matrix(row[3:7]) * row[7:10][None, :]
        m[:3, 3] = row[0:3]
        out[i] = m if parents[i] < 0 else out[parents[i]] @ m
    return out


def to_blender(matrix):
    """A game model-space matrix as a Blender rest matrix: rotation and position, no scale."""
    rotation = matrix[:3, :3] / np.maximum(np.linalg.norm(matrix[:3, :3], axis=0), 1e-12)[None, :]
    out = np.eye(4)
    out[:3, :3] = TO_BLENDER @ rotation
    out[:3, 3] = TO_BLENDER @ matrix[:3, 3]
    return out


def skeleton(index, race, number):
    """(names, parents, local rows, the face skeleton's own bone names) of the race's body bones faces use
    and of face skeleton ``number``, or None when the game has either skeleton missing."""
    base = index.read(BASE_SKELETON_PATH.format(race))
    face = index.read(FACE_SKELETON_PATH.format(race, number))
    if base is None or face is None:
        return None
    b_names, b_parents, b_local = gamefiles.read_sklb(base)
    wanted = set()
    for name in BODY_BONES:
        i = b_names.index(name)
        while i >= 0:
            wanted.add(i)
            i = b_parents[i]
    names, parents, local = [], [], []
    where = {}
    for i in sorted(wanted):
        where[b_names[i]] = len(names)
        names.append(b_names[i])
        parents.append(where[b_names[b_parents[i]]] if b_parents[i] >= 0 else -1)
        local.append(b_local[i])
    f_names, f_parents, f_local = gamefiles.read_sklb(face)
    for i, name in enumerate(f_names):
        if name in where:  # the face skeleton repeats j_kao, its root
            continue
        parent = f_names[f_parents[i]] if f_parents[i] >= 0 else None
        where[name] = len(names)
        names.append(name)
        parents.append(where[parent] if parent is not None else -1)
        local.append(f_local[i])
    return names, parents, np.array(local), f_names


def face_poses(index, race, number, names, face_names):
    """{pose: (bone indices into ``names``, local deltas (k, 10))} of face skeleton ``number``'s face.pap;
    ``face_names`` are the face skeleton's bones, which its tracks refer to."""
    from .xivpy.pap import XIVAnim
    data = index.read(FACE_ANIMATIONS_PATH.format(race, number))
    if data is None:
        return {}
    with contextlib.redirect_stdout(io.StringIO()):
        pap = XIVAnim.from_bytes(data)
    kaos = pap.kaos
    container = kaos.get_animation_container()
    by_name = {info.name: info for info in pap.anim_info}
    result = {}
    for pose, animation in POSES:
        info = by_name.get(animation)
        if info is None:
            continue
        node = kaos.nodes[container["animations"][info.hk_idx]]
        binding = kaos.nodes[container["bindings"][info.hk_idx]]
        # Only additive spline-compressed animations, which all the game's face animations are.
        if node.definition.name != "hkaSplineCompressedAnimation" or int(binding.values.get("blendHint", 0)) != 1:
            continue
        frames = gamefiles.decode_spline(node)
        turn = 2.0 * np.degrees(np.arccos(np.clip(np.abs(frames[:, :, 6]), 0.0, 1.0)))
        peak = int(np.argmax(turn.sum(1)))
        bones, deltas = [], []
        for track, bone in enumerate(binding["transformTrackToBoneIndices"]):
            name = face_names[bone]
            delta = frames[peak, track]
            identity = (np.abs(delta[0:3]).max() < 1e-6 and abs(abs(delta[6]) - 1.0) < 1e-7
                        and np.abs(delta[7:10] - 1.0).max() < 1e-6)
            if identity or name not in names:
                continue
            bones.append(names.index(name))
            deltas.append(delta)
        result[pose] = (np.array(bones, dtype=np.int64), np.array(deltas).reshape(-1, 10))
    return result


# -----------------------------------------------------------------------------
# Face models

class ModelPart:
    """One submesh of a model: ``name`` like Instant Edit names its objects ('0.0 c0201f0001_fac_a'), its
    ``material`` path and ``attributes`` (without atr_), and its vertices from the submesh's first to its last:
    ``points`` (Blender space), ``normals`` (Blender space, as in the file, zero where no triangle uses them),
    ``tris`` (without degenerate or repeated triangles) and weights: ``weight_bones`` (indices into
    ``bone_names``, its mesh's bone table) and ``weight_values`` (bytes), per vertex sorted, zero after the last."""

    def __init__(self, name, material, attributes, points, normals, tris, bone_names, weight_bones, weight_values):
        self.name = name
        self.material = material
        self.attributes = attributes
        self.points = points
        self.normals = normals
        self.tris = tris
        self.bone_names = bone_names
        self.weight_bones = weight_bones
        self.weight_values = weight_values


def _vertex_arrays(buffer, mesh, declaration):
    """{field: array} of ``mesh``'s vertices, for the ELEMENTS it has."""
    streams = {}
    for element in declaration.vertex_elements:
        field = ELEMENTS.get(element.usage.name)
        if field is None or element.usage_idx != 0 or element.type.name not in ELEMENT_TYPES:
            continue
        base, count = ELEMENT_TYPES[element.type.name]
        if field in ("weights", "bones") and element.type.name == "USHORT4":
            base, count = "u1", 8
        streams.setdefault(element.stream, {})[field] = ((base, count), element.offset)
    arrays = {}
    for stream, fields in streams.items():
        dtype = np.dtype({"names": list(fields), "formats": [f for f, _o in fields.values()],
                          "offsets": [o for _f, o in fields.values()], "itemsize": mesh.vertex_buffer_stride[stream]})
        data = np.frombuffer(buffer, dtype, mesh.vertex_count, mesh.vertex_buffer_offset[stream])
        for field in fields:
            arrays[field] = data[field]
    return arrays


def _to_blender(vectors):
    out = np.empty((len(vectors), 3), dtype=vectors.dtype)
    out[:, 0] = vectors[:, 0]
    out[:, 1] = -vectors[:, 2]
    out[:, 2] = vectors[:, 1]
    return out


def surface_normals(points, tris):
    """Per vertex the area-weighted normal of the triangles around it."""
    p = points.astype(np.float64)
    face = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
    out = np.zeros((len(p), 3))
    for k in range(3):
        np.add.at(out, tris[:, k], face)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _clean_tris(tris):
    """``tris`` without degenerate triangles (a vertex twice) and without repeats of a triangle's vertices (in
    any order: a back face on the same vertices), keeping the first; Blender's mesh validation drops them too."""
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
    _unique, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
    return tris[np.sort(first)]


def _sorted_weights(bones, values, names):
    """Per vertex its weights as (bone name indices into ``names``, byte values), bones repeated in a vertex
    added up, sorted by value and then by name (both descending), zero after the last."""
    bones = bones.astype(np.int64)
    values = values.astype(np.int64)
    k = bones.shape[1]
    for i in range(k):
        for j in range(i + 1, k):
            same = (bones[:, j] == bones[:, i]) & (values[:, i] > 0) & (values[:, j] > 0)
            values[same, i] += values[same, j]
            values[same, j] = 0
    values = np.minimum(values, 255)
    valid = (values > 0) & (bones < len(names))
    values = np.where(valid, values, 0)
    bones = np.where(valid, bones, 0)
    # Sort key: value, then the name's rank among the names (Python's string order, as sorted() gives).
    rank = np.zeros(max(len(names), 1), dtype=np.int64)
    rank[sorted(range(len(names)), key=lambda i: names[i])] = np.arange(len(names))
    key = np.where(valid, values * (len(names) + 1) + rank[bones], -1)
    order = np.argsort(-key, axis=1, kind="stable")
    bones = np.take_along_axis(bones, order, axis=1)[:, :MAX_WEIGHTS]
    values = np.take_along_axis(values, order, axis=1)[:, :MAX_WEIGHTS]
    if bones.shape[1] < MAX_WEIGHTS:
        pad = ((0, 0), (0, MAX_WEIGHTS - bones.shape[1]))
        bones = np.pad(bones, pad)
        values = np.pad(values, pad)
    return bones, values.astype(np.uint8)


def material_label(path):
    """The object label Instant Edit derives from a material path: its file name without mt_ and .mtrl."""
    name = (path or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if name.casefold().endswith(".mtrl"):
        name = name[:-5]
    if name.casefold().startswith("mt_"):
        name = name[3:]
    return name


def model_parts(data):
    """The ModelParts of .mdl file ``data``, first level of detail, in mesh and submesh order."""
    from .xivpy.model import XIVModel
    with contextlib.redirect_stdout(io.StringIO()):
        model = XIVModel.from_bytes(data)
    header = model.header
    indices = np.frombuffer(model.buffers, np.uint16, header.idx_buffer_size[0] // 2, header.idx_offset[0])
    buffer = model.buffers[header.vert_offset[0]:]
    parts = []
    for m, mesh in enumerate(model.meshes[:model.lods[0].mesh_count]):
        if mesh.vertex_count == 0:
            continue
        arrays = _vertex_arrays(buffer, mesh, model.vertex_declarations[m])
        if "position" not in arrays:
            continue
        material = model.materials[mesh.material_idx]
        tables = model.bone_tables
        names = [model.bones[b] for b in tables[mesh.bone_table_idx].bone_idx] if mesh.bone_table_idx < len(tables) else []
        for s, submesh in enumerate(model.submeshes[mesh.submesh_index:mesh.submesh_index + mesh.submesh_count]):
            if submesh.idx_count == 0:
                continue
            sub = indices[submesh.idx_offset:submesh.idx_offset + submesh.idx_count].astype(np.int64)
            start = int(sub.min())
            rows = slice(start, int(sub.max()) + 1)
            points = _to_blender(arrays["position"][rows, :3]).astype(np.float32)
            tris = _clean_tris(sub[:len(sub) // 3 * 3].reshape(-1, 3) - start)
            normals = np.zeros((len(points), 3), dtype=np.float32)
            if "normal" in arrays:
                # As the game has them (half floats), but where one is zero, the surface's, and where that is
                # too (no area), straight up: as Blender does.
                raw = _to_blender(arrays["normal"][rows, :3]).astype(np.float32)
                used = np.zeros(len(points), dtype=bool)
                used[tris.ravel()] = True
                normals[used] = raw[used]
                zero = used & (np.abs(raw).max(1) == 0.0)
                if zero.any():
                    surface = surface_normals(points, tris)
                    surface[np.abs(surface).max(1) == 0.0] = (0.0, 0.0, 1.0)
                    normals[zero] = surface[zero]
            if "weights" in arrays and "bones" in arrays:
                bones, values = _sorted_weights(arrays["bones"][rows], arrays["weights"][rows], names)
            else:
                bones = np.zeros((len(points), MAX_WEIGHTS), dtype=np.int64)
                values = np.zeros((len(points), MAX_WEIGHTS), dtype=np.uint8)
            attributes = sorted(model.attributes[bit][4:] for bit in range(len(model.attributes))
                                if submesh.attribute_idx_mask >> bit & 1 and model.attributes[bit].startswith("atr_"))
            parts.append(ModelPart("%d.%d %s" % (m, s, material_label(material)), material, ",".join(attributes),
                                   points, normals, tris, names, bones, values))
    return parts


def eye_radius(iris, eyes):
    """The median distance of the eyes' vertices from their eye bones (Blender space), or 0."""
    if iris is None or len(eyes) != 2:
        return 0.0
    distances = [np.linalg.norm(iris.points[iris.points[:, 0] * sign > 0] - eyes[side], axis=1)
                 for side, sign in (("l", 1.0), ("r", -1.0))]
    return float(np.median(np.concatenate(distances)))


# -----------------------------------------------------------------------------
# Building

def build(root, progress=None):
    """The face data from the game in game folder ``root``, as arrays (see facing.Reference). ``progress``
    is called with the share done (0 to 1) now and then."""
    index = gamefiles.Index(gamefiles.sqpack_folder(root), "chara")
    try:
        return _build(index, progress or (lambda _share: None))
    finally:
        index.close()


def _build(index, progress):
    est_data = index.read(EST_PATH)
    if est_data is None:
        raise FaceDataError("The game's faceskeletontemplate.est wasn't found in its files")
    est = gamefiles.read_est(est_data)
    faces = [(race, number, skel) for (race, number), skel in sorted(est.items())
             if race.endswith(PLAYER_RACE_SUFFIX) and number <= MAX_FACE and index.exists(MODEL_PATH.format(race, number))]
    if not faces:
        raise FaceDataError("No face models were found in the game's files")
    races = sorted({race for race, _number, _skel in faces})
    bone_names = []
    bone_index = {}

    def bone(name):
        if name not in bone_index:
            bone_index[name] = len(bone_names)
            bone_names.append(name)
        return bone_index[name]

    face_rows = []  # (race, number, skeleton, eye radius)
    parts = []  # (face row, kind: 0 skin, 1 other part of the skin's kind of material, 2 other, attributes, v, t)
    co_all, normal_all, tris_all, wb_all, wv_all = [], [], [], [], []
    skeletons = {}
    for f, (race, number, skel) in enumerate(faces):
        progress(0.9 * f / len(faces))
        if (race, skel) not in skeletons:
            found = skeleton(index, race, skel)
            if found is None:
                continue
            names, parents, local, face_names = found
            skeletons[(race, skel)] = (names, parents, local, face_poses(index, race, skel, names, face_names))
        names, parents, local, _poses = skeletons[(race, skel)]
        rest = model_matrices(parents, local)
        eyes = {side: TO_BLENDER @ rest[names.index("j_f_eye_" + side)][:3, 3]
                for side in "lr" if "j_f_eye_" + side in names}
        model = model_parts(index.read(MODEL_PATH.format(race, number)))
        skin = next((part for part in model if part.name.startswith("0.0 ")), None)
        if skin is None:
            continue
        iris = next((part for part in model if "_iri_" in part.name), None)
        face_row = len(face_rows)
        face_rows.append((race, number, skel, eye_radius(iris, eyes)))
        for part in [skin] + [p for p in model if p is not skin and "_iri_" not in p.name]:
            if not len(part.tris):
                continue
            # Bones are numbered as they first come up (vertex by vertex, weight by weight).
            weighted = part.weight_values > 0
            local, first = np.unique(part.weight_bones[weighted], return_index=True)
            table = np.zeros(max(len(part.bone_names), 1), dtype=np.int64)
            for b in local[np.argsort(first)]:
                table[b] = bone(part.bone_names[b])
            wb = np.where(weighted, table[part.weight_bones], 0).astype(np.uint16)
            kind = 0 if part is skin else (1 if "_fac" in part.material else 2)
            parts.append((face_row, kind, part.attributes, len(part.points), len(part.tris)))
            co_all.append(part.points.astype(np.float32))
            normal_all.append(part.normals.astype(np.float16))
            tris_all.append(part.tris.astype(np.uint16))
            wb_all.append(wb)
            wv_all.append(part.weight_values)
    progress(0.95)
    s_race, s_id, s_start, s_count = [], [], [], []
    sb_bone, sb_parent, sb_local, sb_rest = [], [], [], []
    pose_names = [name for name, _animation in POSES]
    p_skel, p_kind, p_start, p_count, pt_bone, pt_delta = [], [], [], [], [], []
    for s, ((race, number), (names, parents, local, poses)) in enumerate(sorted(skeletons.items())):
        rest = model_matrices(parents, local)
        s_race.append(races.index(race))
        s_id.append(number)
        s_start.append(len(sb_bone))
        s_count.append(len(names))
        for i, name in enumerate(names):
            sb_bone.append(bone(name))
            sb_parent.append(parents[i])
            sb_local.append(local[i])
            sb_rest.append(to_blender(rest[i]))
        for kind, pose in enumerate(pose_names):
            if pose not in poses:
                continue
            bones, deltas = poses[pose]
            p_skel.append(s)
            p_kind.append(kind)
            p_start.append(len(pt_bone))
            p_count.append(len(bones))
            pt_bone.extend(bones.tolist())
            pt_delta.extend(deltas.tolist())
    files = sorted(index.used)
    entries = [-1 if index.used[path] is None else index.used[path] for path in files]
    textools = [path for path in files if index.from_textools(path)]
    progress(1.0)
    return dict(
        races=np.array(races),
        bone_names=np.array(bone_names),
        face_race=np.array([races.index(f[0]) for f in face_rows], dtype=np.int64),
        face_id=np.array([f[1] for f in face_rows], dtype=np.int64),
        face_skel=np.array([f[2] for f in face_rows], dtype=np.int64),
        face_eye_radius=np.array([f[3] for f in face_rows], dtype=np.float32),
        part_face=np.array([p[0] for p in parts], dtype=np.int64),
        part_kind=np.array([p[1] for p in parts], dtype=np.int64),
        part_attributes=np.array([p[2] for p in parts]),
        part_vcount=np.array([p[3] for p in parts], dtype=np.int64),
        part_tcount=np.array([p[4] for p in parts], dtype=np.int64),
        co=np.concatenate(co_all), normal=np.concatenate(normal_all), tris=np.concatenate(tris_all),
        w_bone=np.concatenate(wb_all), w_value=np.concatenate(wv_all),
        skel_race=np.array(s_race, dtype=np.int64), skel_id=np.array(s_id, dtype=np.int64),
        skel_start=np.array(s_start, dtype=np.int64), skel_count=np.array(s_count, dtype=np.int64),
        sb_bone=np.array(sb_bone, dtype=np.int64), sb_parent=np.array(sb_parent, dtype=np.int64),
        sb_local=np.array(sb_local, dtype=np.float64), sb_rest=np.array(sb_rest, dtype=np.float32),
        pose_names=np.array(pose_names),
        pose_skel=np.array(p_skel, dtype=np.int64), pose_kind=np.array(p_kind, dtype=np.int64),
        pose_start=np.array(p_start, dtype=np.int64), pose_count=np.array(p_count, dtype=np.int64),
        pt_bone=np.array(pt_bone, dtype=np.int64), pt_delta=np.array(pt_delta, dtype=np.float32),
        game_files=np.array(files, dtype=str), game_entries=np.array(entries, dtype=np.int64),
        textools_files=np.array(textools, dtype=str),
    )


# -----------------------------------------------------------------------------
# The data in the user folder

_found = {}  # game folder setting: the game folder found for it (or None)


def folder_setting():
    """The game folder set in the add-on's preferences ('' when none, or when the add-on isn't enabled as
    such: in the tests)."""
    import bpy
    addon = bpy.context.preferences.addons.get(__package__)
    folder = getattr(getattr(addon, "preferences", None), "game_folder", "") if addon is not None else ""
    return bpy.path.abspath(folder) if folder else ""


def game_folder():
    """(the game folder or None, the setting it was looked for with), looked for once per setting."""
    setting = folder_setting()
    if setting not in _found:
        _found[setting] = gamefiles.find_game(setting)
    return _found[setting], setting


def data_path():
    """Where the data is kept: the add-on's own folder for user data, which Blender keeps across updates
    (for an add-on installed the legacy way, a folder in Blender's user data)."""
    import bpy
    try:
        folder = bpy.utils.extension_path_user(__package__, create=True)
    except ValueError:  # not an extension
        folder = bpy.utils.user_resource('DATAFILES', path=LEGACY_FOLDER, create=True)
    return os.path.join(folder, FILE_NAME)


# What reading a missing, cut short or damaged file raises.
_UNREADABLE = (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile, zlib.error)


def _saved_versions(path):
    """(data version, game version, the game files it was read from, their index entries) of the data saved at
    ``path``, or None when there's none (or it's damaged)."""
    try:
        with np.load(path, allow_pickle=False) as f:
            return (int(f["data_version"]), str(f["game_version"]), tuple(str(p) for p in f["game_files"]),
                    tuple(int(e) for e in f["game_entries"]))
    except _UNREADABLE:
        return None


def _saved(path):
    """The data saved at ``path``, or None."""
    try:
        with np.load(path, allow_pickle=False) as f:
            return {key: f[key] for key in f.files if key not in ("data_version", "game_version")}
    except _UNREADABLE:
        return None


def _save(data, path, version):
    """Saves ``data`` for game version ``version``, replacing what's there only once it's written (each
    Blender writes its own part file first)."""
    part = "{:s}.{:d}.part".format(path, os.getpid())
    try:
        with open(part, "wb") as f:
            np.savez_compressed(f, data_version=np.array(DATA_VERSION), game_version=np.array(version), **data)
        os.replace(part, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(part)
        raise


def _entries(root, files):
    """The chara index entries of game paths ``files`` (-1: no file there) in the game at ``root``, or None when
    its index can't be read."""
    try:
        with gamefiles.Index(gamefiles.sqpack_folder(root), "chara") as index:
            return tuple(-1 if packed is None else packed for packed in map(index.entry, files))
    except (OSError, struct.error):
        return None


def _usable(versions, root):
    """Whether saved data of ``versions`` can be used with the game found at ``root`` (None: not found): read
    from the same game version and from the same files (TexTools installing or removing a mod changes the
    index entries)."""
    if versions is None or versions[0] != DATA_VERSION:
        return False
    return root is None or (versions[1] == gamefiles.game_version(root) and _entries(root, versions[2]) == versions[3])


def saved_data():
    """The saved data when it's for the game found (or the game isn't found and it's of this DATA_VERSION),
    else None: it needs reading from the game."""
    path = data_path()
    if not _usable(_saved_versions(path), game_folder()[0]):
        return None
    return _saved(path)


_problems = {}  # game folder setting: problem()


def problem():
    """Why the game's faces can't be had, as a message, or None: they're saved, or the game is found."""
    root, setting = game_folder()
    if setting not in _problems:
        message = None
        if root is None:
            versions = _saved_versions(data_path())
            if versions is None or versions[0] != DATA_VERSION:
                message = GAME_NOT_AT.format(setting) if setting else GAME_NOT_FOUND
        _problems[setting] = message
    return _problems[setting]


def load(progress=None, again=False):
    """The face data: the saved data when it's for the game found, else read from the game (a few seconds;
    ``progress`` is called with the share done) and saved. ``again`` reads it from the game in any case.
    Raises FaceDataError when the game isn't found and nothing usable is saved, or its files can't be read."""
    path = data_path()
    root, setting = game_folder()
    if not again:
        data = saved_data()
        if data is not None:
            return data
    if root is None:
        raise FaceDataError(GAME_NOT_AT.format(setting) if setting else GAME_NOT_FOUND)
    started = time.perf_counter()
    try:
        data = build(root, progress)
    except Exception as error:  # a game update the readers don't know yet, damaged files
        # The faces saved for an earlier game version still do.
        versions = _saved_versions(path)
        data = _saved(path) if not again and versions is not None and versions[0] == DATA_VERSION else None
        if data is not None:
            print("Magic Fit: the game's faces couldn't be read again from {:s} ({:s}), the saved ones are used".format(
                root, str(error)))
            return data
        if isinstance(error, FaceDataError):
            raise
        raise FaceDataError("The game's faces couldn't be read from '{:s}' ({:s}: {:s})".format(
            root, type(error).__name__, str(error))) from error
    try:
        _save(data, path, gamefiles.game_version(root))
    except OSError as error:
        print("Magic Fit: the game's faces couldn't be saved to {:s}: {:s}".format(path, str(error)))
    print("Magic Fit: read {:d} of the game's faces from {:s} in {:.1f} s".format(
        len(data["face_id"]), root, time.perf_counter() - started))
    return data


def clear_cache():
    """Look for the game folder again (the setting changed)."""
    _found.clear()
    _problems.clear()
