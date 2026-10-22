"""The game's hair models, for the scalp map in hair_reference.npz (see tools/make_scalp_map.py).

`decode` reads a vanilla hair model (.mdl) with Yet Another Addon's xivpy: the positions and triangles of
its first LOD, in Blender model space, and which vertices the game holds on the head (at least half their
weight on j_kao and the ear bones together: the Miqo'te hairs have the cat ears). Vertices no triangle uses
(the shape keys' extra vertices) are left out.

`scalp_shares` learns from a race's hairstyles where its scalp ends: a head point's scalp share is the share
of the hairstyles that hold hair within SCALP_COVER_RADIUS of it. Most hairstyles hold the cranium, fewer the
nape and neck; the game's own fringes hold the brow too (tools/make_scalp_map.py takes the front out). The
shares are smoothed over neighbouring head points and made symmetric left to right, like the head grid.

Both need this folder on sys.path (for make_hair_reference).
"""

import sys

import numpy as np

import make_hair_reference

# A vertex is held when at least HELD_WEIGHT of its weight is on these bones.
HELD_BONES = ("j_kao", "j_mimi_l", "j_mimi_r")
HELD_WEIGHT = 0.5
# A hairstyle covers a head point when it holds hair this close to it (metres).
SCALP_COVER_RADIUS = 0.03
# The shares are smoothed over this many nearest head points (the point itself included).
SMOOTH_NEIGHBOURS = 8
# Distances are worked out this many rows at a time.
CHUNK = 4096


def _xivpy_model():
    if make_hair_reference.XIVPY not in sys.path:
        sys.path.insert(0, make_hair_reference.XIVPY)
    from xivpy import model
    return model


def decode(data):
    """(points (n, 3) in Blender model space, triangles (m, 3) int64, held (n,) bool) of a hair model's
    first LOD, all its meshes together, without the vertices no triangle uses."""
    xivmodel = _xivpy_model()
    from xivpy.model.enums import VertexUsage
    mdl = xivmodel.XIVModel.from_bytes(data)
    lod = mdl.lods[0]
    buffers = mdl.buffers
    held_bones = [mdl.bones.index(name) for name in HELD_BONES if name in mdl.bones]
    points, tris, held = [], [], []
    base = 0
    for mesh_index in range(lod.mesh_idx, lod.mesh_idx + lod.mesh_count):
        mesh = mdl.meshes[mesh_index]
        count = mesh.vertex_count
        if not count:
            continue
        position = weights = indices = None
        for element in mdl.vertex_declarations[mesh_index].vertex_elements:
            if element.usage not in (VertexUsage.POSITION, VertexUsage.BLEND_WEIGHTS, VertexUsage.BLEND_INDICES):
                continue
            dtype = np.dtype(xivmodel.get_vert_struct(element.type, element.usage))
            stride = mesh.vertex_buffer_stride[element.stream]
            start = lod.vertex_data_offset + mesh.vertex_buffer_offset[element.stream] + element.offset
            values = np.ndarray((count,), dtype=dtype, buffer=buffers, offset=start, strides=(stride,))
            values = np.array(values).reshape(count, -1)
            if element.usage == VertexUsage.POSITION:
                position = values[:, :3].astype(np.float64)
            elif element.usage == VertexUsage.BLEND_WEIGHTS:
                # Bytes (UBYTE4, NBYTE4, and the eight of USHORT4) are 0..255.
                weights = values.astype(np.float64) / (255.0 if values.dtype == np.uint8 else 1.0)
            else:
                table = np.array(mdl.bone_tables[mesh.bone_table_idx].bone_idx, dtype=np.int64)
                indices = table[values.astype(np.int64)]
        on_head = np.zeros(count)
        if weights is not None and indices is not None and held_bones:
            width = min(weights.shape[1], indices.shape[1])
            on_head = (weights[:, :width] * np.isin(indices[:, :width], held_bones)).sum(1)
        index = np.ndarray((mesh.idx_count,), dtype="<u2", buffer=buffers,
                           offset=lod.idx_data_offset + mesh.start_idx * 2)
        index = index.astype(np.int64).reshape(-1, 3)
        used = np.zeros(count, dtype=bool)
        used[index.ravel()] = True
        renumber = np.cumsum(used) - 1
        points.append(position[used] @ make_hair_reference.TO_BLENDER.T)
        tris.append(renumber[index] + base)
        held.append(on_head[used] >= HELD_WEIGHT)
        base += int(used.sum())
    return np.concatenate(points), np.concatenate(tris), np.concatenate(held)


def nearest_points(points, targets):
    """Index of the nearest of ``targets`` to each of ``points`` (brute force, CHUNK rows a time)."""
    points = np.asarray(points, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    squared = (targets * targets).sum(1)
    result = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), CHUNK):
        chunk = points[start:start + CHUNK]
        result[start:start + CHUNK] = np.argmin(squared[None, :] - 2.0 * chunk @ targets.T, axis=1)
    return result


def nearest_distances(points, targets):
    """Distance from each of ``points`` to the nearest of ``targets`` (inf without targets)."""
    points = np.asarray(points, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    best = np.full(len(points), np.inf)
    for start in range(0, len(targets), CHUNK):
        chunk = targets[start:start + CHUNK]
        squared = ((points[:, None, :] - chunk[None, :, :]) ** 2).sum(2)
        best = np.minimum(best, squared.min(1))
    return np.sqrt(best)


def scalp_shares(head_points, hairs, radius=SCALP_COVER_RADIUS):
    """The scalp share (0..1) of each of a race's ``head_points``: the share of its hairstyles ``hairs`` (a
    list of `decode` results) that hold hair within ``radius`` of it, smoothed and made symmetric."""
    head_points = np.asarray(head_points, dtype=np.float64)
    covered = np.zeros(len(head_points))
    for points, _tris, held in hairs:
        covered += nearest_distances(head_points, points[held]) <= radius
    share = covered / max(len(hairs), 1)
    distances = ((head_points[:, None, :] - head_points[None, :, :]) ** 2).sum(2)
    neighbours = np.argsort(distances, axis=1)[:, :SMOOTH_NEIGHBOURS]
    share = share[neighbours].mean(1)
    mirror = nearest_points(head_points * (-1.0, 1.0, 1.0), head_points)
    share = 0.5 * (share + share[mirror])
    return np.clip(share, 0.0, 1.0).astype(np.float32)
