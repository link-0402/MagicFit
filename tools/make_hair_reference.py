"""Write magic_fit/hair_reference.npz: FFXIV's hair skeletons, the body's hair bones and each race's head.

Hair weights need, for every race, the hair skeletons the game can load for a hairstyle (the EST entry picks
one), the body bones hair hangs from (neck, head, ears and the body's own hair bones j_kami_*), and the shape
of the head, to tell hair lying on the scalp from hair hanging free.

Sources:
  - A game export of Instant Edit (its Dalamud plugin, "Export game files" with skeleton files): every race's
    body skeleton (chara/human/cXXXX/skeleton/base/b0001), the hair skeletons (chara/human/cXXXX/skeleton/
    hair/hNNNN) and the hair EST table (chara/xls/charadb/hairskeletontemplate.est). The skeleton files are
    Havok tagfiles, read with Yet Another Addon's xivpy (it must be installed in Blender).
  - TexTools' FBX exports of one face of every race (Character/Face/<Race>_<Gender>/<face>/3D/*.fbx): the
    bald head. It is stored as a signed distance grid (1 cm cells, millimetres, up to 12.7 cm) and 800
    points spread over its surface (for distances beyond the grid), not as a mesh, both made symmetric
    left to right.

Bone rest matrices are the game's model-space bind poses taken to Blender (Z up, facing -Y), rotation and
position only, as in TexTools' FBX exports: a bone's local axes are the game bone's axes. Hair bones point
along their local X axis, toward the hair they move.

Run with Blender:
    blender -b --factory-startup --python tools/make_hair_reference.py -- \\
        "<Instant Edit export folder>" "C:/Users/<you>/Documents/TexTools/Saved/Character/Face"
"""

import glob
import io
import contextlib
import os
import re
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "magic_fit", "hair_reference.npz")
XIVPY = os.path.join(os.environ.get("APPDATA", ""), "Blender Foundation", "Blender", "5.2", "extensions",
                     "user_default", "yet_another_addon_ffxiv")
# Game model space (Y up, facing +Z) to Blender (Z up, facing -Y).
TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
RACES = ("c0101", "c0201", "c0301", "c0401", "c0501", "c0601", "c0701", "c0801", "c0901", "c1001",
         "c1101", "c1201", "c1301", "c1401", "c1501", "c1601", "c1701", "c1801")
# Body bones hair can hang from, and everything above them.
BODY_BONES = ("j_sebo_c", "j_kubi", "j_kao", "j_ago", "j_mimi_l", "j_mimi_r", "j_sako_l", "j_sako_r",
              "j_kami_a", "j_kami_b", "j_kami_f_l", "j_kami_f_r")
SDF_STEP = 0.01
SDF_MARGIN = 0.08
SDF_CLAMP = 0.127
# Points on the head's surface (by area) for distances beyond the grid.
HEAD_POINTS = 800


def _quat_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def read_skeleton(path):
    """(names, parent indices, model-space 4x4 matrices in game space) of a .sklb file, any version."""
    from xivpy.kaos import Tagfile
    from xivpy.utils import BinaryReader
    data = open(path, "rb").read()
    if data[:4] != b"blks":
        raise ValueError(path + " is not a skeleton file")
    version = data[4:8]
    if version == b"0021":
        offset = struct.unpack_from("<H", data, 10)[0]
    else:  # "0031" and "1031"
        offset = struct.unpack_from("<I", data, 12)[0]
    reader = BinaryReader(data)
    reader.pos = offset
    with contextlib.redirect_stdout(io.StringIO()):
        tagfile = Tagfile.from_bytes(reader)
    skeleton = tagfile.get_skeleton_node()
    names = tagfile.get_bone_list(skeleton)
    parents = [int(p) for p in skeleton["parentIndices"]]
    pose = np.asarray(skeleton["referencePose"], dtype=np.float64)
    model = []
    for index, parent in enumerate(parents):
        local = np.eye(4)
        local[:3, :3] = _quat_matrix(pose[index, 4:8])   # rest bones hold no scale
        local[:3, 3] = pose[index, 0:3]
        model.append(local if parent < 0 else model[parent] @ local)
    return names, parents, model


def to_blender(matrix):
    result = np.eye(4)
    result[:3, :3] = TO_BLENDER @ matrix[:3, :3]
    result[:3, 3] = TO_BLENDER @ matrix[:3, 3]
    return result


def read_est(path):
    data = open(path, "rb").read()
    count = struct.unpack_from("<i", data, 0)[0]
    table = {}
    for index in range(count):
        hair_set, gender_race = struct.unpack_from("<HH", data, 4 + index * 4)
        table[("c%04d" % gender_race, hair_set)] = struct.unpack_from("<H", data, 4 + count * 4 + index * 2)[0]
    return table


# -----------------------------------------------------------------------------
# Heads

def _closest_on_triangles(p, a, b, c):
    """Closest points to ``p`` on triangles (a, b, c), all (n, 3) (Ericson, vectorised)."""
    ab = b - a; ac = c - a; ap = p - a
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    out = np.empty_like(p); done = np.zeros(len(p), dtype=bool)

    def put(mask, value):
        nonlocal done
        mask = mask & ~done
        out[mask] = value[mask]
        done |= mask

    put((d1 <= 0) & (d2 <= 0), a)
    bp = p - b; d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    put((d3 >= 0) & (d4 <= d3), b)
    vc = d1 * d4 - d3 * d2
    v = d1 / np.where(np.abs(d1 - d3) > 1e-30, d1 - d3, 1e-30)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)
    cp = p - c; d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    put((d6 >= 0) & (d5 <= d6), c)
    vb = d5 * d2 - d1 * d6
    w = d2 / np.where(np.abs(d2 - d6) > 1e-30, d2 - d6, 1e-30)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)
    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) > 1e-30, (d4 - d3) + (d5 - d6), 1e-30)
    put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w[:, None] * (c - b))
    denom = 1.0 / np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1e-30)
    put(np.ones(len(p), dtype=bool), a + ab * (vb * denom)[:, None] + ac * (vc * denom)[:, None])
    return out


def head_grid(vertices, triangles):
    """Signed distance grid (int8 millimetres, clamped) of a head mesh: (origin, int8 grid). The grid is
    symmetric left to right, and so are its distances (the mean of both sides), so mirrored hair gets
    mirrored weights although the game's faces aren't quite symmetric."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromPolygons([Vector(v) for v in vertices], [tuple(t) for t in triangles])
    low = np.floor((vertices.min(0) - SDF_MARGIN) / SDF_STEP) * SDF_STEP
    shape = np.ceil((vertices.max(0) + SDF_MARGIN - low) / SDF_STEP).astype(int) + 1
    half = int(np.ceil((np.abs(vertices[:, 0]).max() + SDF_MARGIN) / SDF_STEP))
    low[0] = -half * SDF_STEP
    shape[0] = 2 * half + 1
    axes = [low[i] + SDF_STEP * np.arange(shape[i]) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    distance = np.empty(len(points))
    for index, point in enumerate(points):
        location, _normal, _face, dist = tree.find_nearest(Vector(point))
        distance[index] = dist if location is not None else SDF_CLAMP
    # Inside or outside by the generalised winding number (the neck opening is far from any hair).
    a = vertices[triangles[:, 0]]; b = vertices[triangles[:, 1]]; c = vertices[triangles[:, 2]]
    winding = np.zeros(len(points))
    for start in range(0, len(points), 500):
        p = points[start:start + 500, None, :]
        pa = a[None] - p; pb = b[None] - p; pc = c[None] - p
        la = np.linalg.norm(pa, axis=2); lb = np.linalg.norm(pb, axis=2); lc = np.linalg.norm(pc, axis=2)
        det = np.einsum("ijk,ijk->ij", pa, np.cross(pb, pc))
        den = (la * lb * lc + np.einsum("ijk,ijk->ij", pa, pb) * lc + np.einsum("ijk,ijk->ij", pb, pc) * la
               + np.einsum("ijk,ijk->ij", pc, pa) * lb)
        winding[start:start + 500] = np.arctan2(det, den).sum(1) / (2 * np.pi)
    signed = np.where(np.abs(winding) > 0.5, -distance, distance).reshape(shape)
    signed = 0.5 * (signed + signed[::-1])
    grid = np.round(np.clip(signed, -SDF_CLAMP, SDF_CLAMP) * 1000).astype(np.int8)
    return low, grid


def surface_points(vertices, triangles, count, seed=0):
    """``count`` points spread over a mesh's surface by area (the same ones every run): half of them, and
    their mirror images."""
    count //= 2
    a = vertices[triangles[:, 0]]; b = vertices[triangles[:, 1]]; c = vertices[triangles[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(triangles), count, p=area / area.sum())
    u = rng.random(count); v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]; v[flip] = 1.0 - v[flip]
    points = a[pick] + u[:, None] * (b[pick] - a[pick]) + v[:, None] * (c[pick] - a[pick])
    return np.concatenate([points, points * (-1.0, 1.0, 1.0)])


def face_mesh(path):
    """World-space vertices and triangles of every mesh in a TexTools face FBX (Blender, Z up)."""
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices, triangles, base = [], [], 0
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        mesh = obj.evaluated_get(depsgraph).to_mesh()
        mesh.calc_loop_triangles()
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        matrix = np.array(obj.matrix_world)
        co = co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tris)
        vertices.append(co); triangles.append(tris.reshape(-1, 3) + base)
        base += len(co)
        obj.evaluated_get(depsgraph).to_mesh_clear()
    return np.concatenate(vertices), np.concatenate(triangles)


# -----------------------------------------------------------------------------

def main(export, faces):
    sys.path.insert(0, XIVPY)
    human = os.path.join(export, "chara", "human")
    est = read_est(os.path.join(export, "chara", "xls", "charadb", "hairskeletontemplate.est"))

    names = []

    def name_index(name):
        if name not in names:
            names.append(name)
        return names.index(name)

    base_race, base_bone, base_parent, base_rest = [], [], [], []
    skel_race, skel_id, skel_start, skel_count = [], [], [], []
    sb_bone, sb_parent, sb_rest = [], [], []
    sdf_origin, sdf_shape, sdf_offset, sdf_data, head_points = [], [], [], [], []
    est_race, est_set, est_skel = [], [], []
    offset = 0
    for race_index, race in enumerate(RACES):
        body_names, body_parents, body_model = read_skeleton(
            os.path.join(human, race, "skeleton", "base", "b0001", "skl_%sb0001.sklb" % race))
        wanted = set()
        for bone in BODY_BONES:
            index = body_names.index(bone)
            while index >= 0:
                wanted.add(index)
                index = body_parents[index]
        body_rest = {}
        for index in sorted(wanted):
            parent = body_parents[index]
            base_race.append(race_index)
            base_bone.append(name_index(body_names[index]))
            base_parent.append(name_index(body_names[parent]) if parent >= 0 else -1)
            rest = to_blender(body_model[index])
            base_rest.append(rest)
            body_rest[body_names[index]] = body_model[index]
        for path in sorted(glob.glob(os.path.join(human, race, "skeleton", "hair", "h*", "skl_%sh*.sklb" % race))):
            number = int(re.search(r"h(\d{4})\.sklb$", path).group(1))
            hair_names, hair_parents, hair_model = read_skeleton(path)
            # Merged by name: a hair skeleton's j_kao (and any other body bone) is the body's; the bones
            # hanging from it keep their place relative to it.
            actual = []
            start = len(sb_bone)
            for index, (bone, parent) in enumerate(zip(hair_names, hair_parents)):
                if bone in body_names:
                    actual.append(body_model[body_names.index(bone)])
                    continue
                if parent < 0:
                    actual.append(body_model[body_names.index("j_kao")] @ hair_model[index])
                    parent_name = "j_kao"
                else:
                    actual.append(actual[parent] @ np.linalg.inv(hair_model[parent]) @ hair_model[index])
                    parent_name = hair_names[parent]
                if bone in [names[b] for b in sb_bone[start:]]:
                    continue
                sb_bone.append(name_index(bone))
                sb_parent.append(name_index(parent_name))
                sb_rest.append(to_blender(actual[index]))
            skel_race.append(race_index); skel_id.append(number)
            skel_start.append(start); skel_count.append(len(sb_bone) - start)
        for (table_race, hair_set), skeleton in sorted(est.items()):
            if table_race == race:
                est_race.append(race_index); est_set.append(hair_set); est_skel.append(skeleton)
        pattern = os.path.join(faces, "*", "*", "3D", "%sf*_fac.fbx" % race)
        fbx = sorted(glob.glob(pattern))
        if not fbx:
            raise FileNotFoundError(pattern)
        vertices, triangles = face_mesh(fbx[0])
        low, grid = head_grid(vertices, triangles)
        sdf_origin.append(low); sdf_shape.append(grid.shape); sdf_offset.append(offset)
        sdf_data.append(grid.ravel()); offset += grid.size
        head_points.append(surface_points(vertices, triangles, HEAD_POINTS))
        print(race, "skeletons", sum(1 for r in skel_race if r == race_index), "head grid", grid.shape,
              os.path.basename(fbx[0]))

    np.savez_compressed(
        OUT,
        races=np.array(RACES), bone_names=np.array(names),
        base_race=np.array(base_race, np.int16), base_bone=np.array(base_bone, np.int16),
        base_parent=np.array(base_parent, np.int16), base_rest=np.array(base_rest, np.float32),
        skel_race=np.array(skel_race, np.int16), skel_id=np.array(skel_id, np.int16),
        skel_start=np.array(skel_start, np.int32), skel_count=np.array(skel_count, np.int16),
        sb_bone=np.array(sb_bone, np.int16), sb_parent=np.array(sb_parent, np.int16),
        sb_rest=np.array(sb_rest, np.float32),
        est_race=np.array(est_race, np.int16), est_set=np.array(est_set, np.int16),
        est_skel=np.array(est_skel, np.int16),
        sdf_step=np.float32(SDF_STEP), sdf_origin=np.array(sdf_origin, np.float32),
        sdf_shape=np.array(sdf_shape, np.int32), sdf_offset=np.array(sdf_offset, np.int64),
        sdf_data=np.concatenate(sdf_data).astype(np.int8), head_points=np.array(head_points, np.float32),
    )
    print("wrote", OUT, os.path.getsize(OUT), "bytes;", len(skel_id), "skeletons,", len(names), "bone names")


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if len(arguments) != 2:
        sys.exit(__doc__)
    main(*arguments)
