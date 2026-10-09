# SPDX-License-Identifier: GPL-3.0-or-later

"""Subdividing clothing where resizing bends it within its faces.

Resize moves a mesh's vertices, and its faces stay flat between them. Where the body changes shape within a face (a
nipple that grows under a low-poly top, the underside of a breast that turns), the face can't follow: the body comes
out between the vertices, or the face cuts across the new shape. `faces_to_cut` finds those faces from how the body
moves under their corners, the middles of their edges and their centres between the From shape and the To shapes:
where the middle moves differently from the corners, the face would have to bend. A body that just grows or shrinks
(the move changing evenly across a face) needs no cut. `subdivide` cuts the faces in four, and the faces next to them
in two, keeping the mesh's vertex groups, UV maps, colours, shape keys and custom normals.
"""

import bmesh
import numpy as np

from . import resizing

# Faces whose centre lies this close to the From body count...
CUT_REACH = 0.02
# ...and are cut when their middle moves this much differently from their corners, for some To shape.
CUT_TOLERANCE = 0.0005
# Cuts at most this many times over.
MAX_LEVELS = 3
# The face attribute that carries each face's old index through a cut (removed afterwards).
ORIGIN = "magic_fit_cut_from"


def faces_to_cut(co, tris, source, targets, body_tris, tolerance=CUT_TOLERANCE):
    """Which of the triangles ``tris`` of clothing at ``co`` (world space, fitted to the body's ``source`` shape) would
    have to bend to follow the body to any of ``targets`` (its other shapes, on the same vertices, with triangles
    ``body_tris``): a mask. The body's move is taken at the spot of the body closest to each corner, edge middle and
    centre of a triangle; a triangle bends where the move of a middle differs from the average of its ends by more
    than ``tolerance``."""
    co = np.asarray(co, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if not len(tris) or not len(targets):
        return np.zeros(len(tris), dtype=bool)
    corners = co[tris]
    middles = 0.5 * (corners + np.roll(corners, -1, axis=1))
    centres = corners.mean(axis=1)
    samples = np.concatenate([corners.reshape(-1, 3), middles.reshape(-1, 3), centres])
    surface = resizing._surface(np.asarray(source, dtype=np.float64), np.asarray(body_tris, dtype=np.int64))
    near = surface.nearest(samples)
    count = len(tris)
    reach = near.distance[6 * count:]
    close = near.found[6 * count:] & (np.abs(reach) < CUT_REACH)
    bends = np.zeros(count)
    here = np.einsum("ij,ijk->ik", near.bary, surface.co[near.corners])
    for target in targets:
        moved = resizing._surface(np.asarray(target, dtype=np.float64), np.asarray(body_tris, dtype=np.int64),
                                  like=surface)
        move = np.einsum("ij,ijk->ik", near.bary, moved.co[near.corners]) - here
        at_corners = move[:3 * count].reshape(count, 3, 3)
        at_middles = move[3 * count:6 * count].reshape(count, 3, 3)
        at_centres = move[6 * count:]
        ends = 0.5 * (at_corners + np.roll(at_corners, -1, axis=1))
        bend = np.maximum(np.linalg.norm(at_middles - ends, axis=2).max(axis=1),
                          np.linalg.norm(at_centres - at_corners.mean(axis=1), axis=1))
        bends = np.maximum(bends, bend)
    return close & (bends > tolerance)


def refine(obj, shape_of, source, targets, body_tris, levels=1, tolerance=CUT_TOLERANCE):
    """Cut the faces of mesh ``obj`` that would have to bend to follow the body (see `faces_to_cut`), ``levels`` times
    over, looking again after each cut. ``shape_of(obj)`` gives the mesh's vertices in world space, fitted to the
    body's ``source`` shape. Returns (faces cut, vertices added)."""
    cut = added = 0
    # Resize's keys changed by hand stay so: once cut, that can't be told.
    resizing.note_edited_keys(obj)
    for _level in range(max(1, min(int(levels), MAX_LEVELS))):
        mesh = obj.data
        tris = resizing._tris(mesh)
        polygon = np.empty(len(mesh.loop_triangles), dtype=np.int64)
        mesh.loop_triangles.foreach_get("polygon_index", polygon)
        bending = faces_to_cut(shape_of(obj), tris, source, targets, body_tris, tolerance)
        faces = np.unique(polygon[bending])
        if not len(faces):
            break
        cut += len(faces)
        added += subdivide(mesh, faces)
    return cut, added


def subdivide(mesh, faces):
    """Cut the faces of ``mesh`` (Object Mode) whose indices are in ``faces`` in four (triangles; others along the
    middles of their edges), and the faces next to them across the edges cut, into triangles. Vertex groups, UV maps,
    colours and shape keys are interpolated, and custom normals kept: each new corner takes the normal of the old
    face it lies in, interpolated across it. Returns how many vertices were added."""
    faces = sorted(set(int(face) for face in faces))
    if not faces:
        return 0
    before = len(mesh.vertices)
    old_normals = None
    if mesh.has_custom_normals:
        old_normals = np.array([normal.vector[:] for normal in mesh.corner_normals], dtype=np.float64)
        old_tris = resizing._tris(mesh, "loops")
        old_polygon = np.empty(len(mesh.loop_triangles), dtype=np.int64)
        mesh.loop_triangles.foreach_get("polygon_index", old_polygon)
        old_co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", old_co)
        loop_vertex = np.empty(len(mesh.loops), dtype=np.int64)
        mesh.loops.foreach_get("vertex_index", loop_vertex)
        old_corner_co = old_co.reshape(-1, 3)[loop_vertex]
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        # Each face remembers the face it was cut from (the pieces of a face copy its layers).
        origin = bm.faces.layers.int.new(ORIGIN)
        for face in bm.faces:
            face[origin] = face.index
        picked = [bm.faces[index] for index in faces if index < len(bm.faces)]
        edges = list({edge for face in picked for edge in face.edges})
        bmesh.ops.subdivide_edges(bm, edges=edges, cuts=1, use_grid_fill=True)
        # The faces next to the cut ones got a corner more along each edge cut: triangles again.
        split = [face for face in bm.faces if len(face.verts) > 3]
        if split:
            bmesh.ops.triangulate(bm, faces=split)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.update()
    attribute = mesh.attributes.get(ORIGIN)
    if attribute is not None:
        came = np.empty(len(mesh.polygons), dtype=np.int64)
        attribute.data.foreach_get("value", came)
        mesh.attributes.remove(attribute)
        if old_normals is not None:
            _restore_normals(mesh, came, old_corner_co, old_tris, old_polygon, old_normals)
    return len(mesh.vertices) - before


def _restore_normals(mesh, came, corner_co, tris, polygon, normals):
    """Give ``mesh`` custom normals from the old mesh's: each corner of a face cut from old face ``came[face]`` the
    normal of the old face's triangle (of ``tris``, corner indices into ``corner_co`` and ``normals``; ``polygon``
    gives each triangle's face) its face's centre lies in, interpolated there."""
    from .fitting import _barycentric
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    loop_vertex = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loop_vertex)
    sizes = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_total", sizes)
    face_of = np.repeat(np.arange(len(sizes)), sizes)
    centres = np.stack([np.bincount(face_of, weights=co[loop_vertex, k], minlength=len(sizes)) for k in range(3)], 1)
    centres /= np.maximum(sizes, 1)[:, None]
    # The old face's triangles (as many as its most), and of them the one each new face's centre lies in most.
    order = np.argsort(polygon, kind="stable")
    first = np.searchsorted(polygon[order], np.arange(polygon.max() + 2))
    most = int(np.diff(first).max())
    best, score = np.zeros(len(sizes), dtype=np.int64), np.full(len(sizes), -np.inf)
    for k in range(most):
        slot = np.minimum(first[came] + k, first[came + 1] - 1)
        tri = order[slot]
        corners = tris[tri]
        inside = _barycentric(centres, *(corner_co[corners[:, j]] for j in range(3))).min(axis=1)
        better = inside > score
        best[better], score[better] = tri[better], inside[better]
    old = tris[best[face_of]]
    bary = np.clip(_barycentric(co[loop_vertex], *(corner_co[old[:, k]] for k in range(3))), 0.0, None)
    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)
    result = np.einsum("ij,ijk->ik", bary, normals[old])
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    mesh.normals_split_custom_set(result.tolist())
