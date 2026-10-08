"""What the resize bench measures of clothing moved from one body shape to another.

Plain numpy and mathutils, independent of the Magic Fit checkout being measured, so two checkouts are measured
the same way. Clothing is welded first (split seams joined), and every metric is taken per welded vertex. Skin meshes
(by their material) are measured apart.

- clearance: each vertex's signed distance from the body (negative inside) on the From shape and on the To
  shape; the error is how much it changed, in bands of how far the vertex was from the skin. ``clip_out`` counts
  vertices that were outside the skin and end up more than 1 mm inside.
- thickness: pairs of layers (a vertex and the spot of the next sheet along its normal, within LAYER_REACH,
  so gaps between parts don't count);
  how much their distance changed.
- detail: each vertex's Laplacian (cotangent weights of the From shape) in its own frame (normal and a tangent
  towards a fixed neighbour), before and after; the change says how much a bump or fold was flattened or
  exaggerated. The body's own change under the clothing is the floor.
- rigid: how much the clearance of rigid parts worn on the skin (piercings, studs) changed, per part.
- round trip: how far a vertex lands from where it was after going to a size and back.
- flips and stretch: triangles turned over, edges stretched.
"""

import re

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

WELD = 1e-6
# Materials of skin meshes, measured apart from the clothing: the game's skin (as resizing.SKIN_MATERIAL, which
# leaves out the body's other materials, such as piercings) or one named skin.
SKIN_MATERIAL = re.compile(r"(mt_c\d{4}b\d{4}_(bibo|eve|b|a)\b|(^|[/_])skin)", re.I)
# Clearance bands (m) by the distance from the skin on the From shape.
BANDS = ((-0.05, 0.0, "inside"), (0.0, 0.003, "0-3mm"), (0.003, 0.006, "3-6mm"), (0.006, 0.015, "6-15mm"),
         (0.015, 0.05, "15-50mm"))
CLIP = 0.001
LAYER_REACH = 0.006
LAYER_MIN = 1e-4
LAYER_FACING = 0.5
# Rigid parts count as worn on the skin within this of it.
RIGID_TOUCH = 0.003
WORST = 10
MM = 1000.0


def is_skin_mesh(record):
    """Whether a garment's mesh (a common.mesh_record) is skin: most of its faces use a skin material."""
    names = [bool(SKIN_MATERIAL.search(name)) for name in record["materials"]]
    if not names or not len(record["sizes"]):
        return False
    faces = np.asarray(names)[np.clip(record["poly_material"], 0, len(names) - 1)]
    return bool(faces.mean() > 0.5)


def weld(points, tris):
    """(welded index per point, welded positions' first point per group, welded triangles without degenerate ones)."""
    key = np.round(np.asarray(points) / WELD).astype(np.int64)
    _unique, first, index = np.unique(key, axis=0, return_index=True, return_inverse=True)
    index = index.reshape(-1)
    t = index[np.asarray(tris, dtype=np.int64).reshape(-1, 3)]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    _u, keep = np.unique(np.sort(t, axis=1), axis=0, return_index=True)
    return index, first, t[np.sort(keep)]


def tri_normals(co, tris):
    n = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    return n


def vertex_normals(co, tris):
    n = np.zeros_like(co)
    face = tri_normals(co, tris)
    for k in range(3):
        np.add.at(n, tris[:, k], face)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(length, 1e-12)


def vertex_areas(co, tris):
    area = np.linalg.norm(tri_normals(co, tris), axis=1) / 6.0
    out = np.zeros(len(co))
    for k in range(3):
        np.add.at(out, tris[:, k], area)
    return out


def edges_of(tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def _barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack((1.0 - v - w, v, w), axis=1)


class Surface:
    """A body surface for signed distances (negative inside). Points whose closest spot is on an open edge of the
    body and off to its side are ``beyond`` it (past the neck, wrists...)."""

    def __init__(self, co, tris):
        self.co = np.asarray(co, dtype=np.float64)
        self.tris = np.asarray(tris, dtype=np.int64)
        self.bvh = BVHTree.FromPolygons(self.co.tolist(), self.tris.tolist(), all_triangles=True)
        # Welded, so split seams are neither open edges nor corners with half a fan.
        _u, weld_index = np.unique(np.round(self.co / WELD).astype(np.int64), axis=0, return_inverse=True)
        self.weld = weld_index.reshape(-1)
        wt = self.weld[self.tris]
        face = tri_normals(self.co, self.tris)
        self.face_normals = face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-12)
        smooth = np.zeros((self.weld.max() + 1, 3))
        pseudo = np.zeros_like(smooth)
        for k in range(3):
            u = self.co[self.tris[:, (k + 1) % 3]] - self.co[self.tris[:, k]]
            w = self.co[self.tris[:, (k + 2) % 3]] - self.co[self.tris[:, k]]
            angle = np.arctan2(np.linalg.norm(np.cross(u, w), axis=1), (u * w).sum(1))
            np.add.at(smooth, wt[:, k], face)
            np.add.at(pseudo, wt[:, k], self.face_normals * angle[:, None])
        self.normals = (smooth / np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-12))[self.weld]
        self.vertex_pseudo = pseudo[self.weld]
        # Per triangle, the edge opposite each corner: whether it's open, and its normal (its faces' sum).
        key = np.sort(np.concatenate([wt[:, [1, 2]], wt[:, [2, 0]], wt[:, [0, 1]]]), axis=1)
        _u, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        edge_sum = np.zeros((len(counts), 3))
        np.add.at(edge_sum, inverse, np.tile(self.face_normals, (3, 1)))
        self.open = (counts[inverse] == 1).reshape(3, -1).T
        self.edge_pseudo = edge_sum[inverse].reshape(3, -1, 3).transpose(1, 0, 2)

    def nearest(self, points):
        """(signed distance, triangle, barycentric coordinates, beyond, found) for each point."""
        points = np.asarray(points, dtype=np.float64)
        n = len(points)
        loc = np.zeros((n, 3))
        face = np.full(n, -1, dtype=np.int64)
        find = self.bvh.find_nearest
        for k, p in enumerate(points.tolist()):
            hit = find(p)
            if hit[0] is not None:
                loc[k] = hit[0]
                face[k] = hit[2]
        found = face >= 0
        f = np.where(found, face, 0)
        corners = self.tris[f]
        bary = _barycentric(loc, self.co[corners[:, 0]], self.co[corners[:, 1]], self.co[corners[:, 2]])
        normal = (self.normals[corners] * bary[:, :, None]).sum(axis=1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        offset = points - loc
        dist = np.linalg.norm(offset, axis=1)
        # The side from the angle-weighted pseudonormal of the face, edge or corner the closest spot is on, right
        # for any closed surface (smooth normals get it wrong where it's on an edge between thighs).
        zero = bary < 1e-5
        pseudo = self.face_normals[f].copy()
        on_edge = zero.sum(axis=1) == 1
        corner = np.argmax(zero, axis=1)
        pseudo[on_edge] = self.edge_pseudo[f[on_edge], corner[on_edge]]
        on_corner = zero.sum(axis=1) >= 2
        pseudo[on_corner] = self.vertex_pseudo[corners[on_corner, np.argmax(bary[on_corner], axis=1)]]
        side = np.where((offset * pseudo).sum(axis=1) < 0.0, -1.0, 1.0)
        signed = dist * side
        on_open = ((bary < 1e-4) & self.open[f]).any(axis=1)
        facing = np.abs((offset * normal).sum(axis=1)) / np.maximum(dist, 1e-12)
        beyond = on_open & (dist > 1e-4) & (facing < 0.9)
        return signed, face, bary, beyond, found


def layer_pairs(co, tris, normals):
    """Pairs of layers of a welded mesh: (vertex, triangle hit, barycentric coordinates, distance) for each vertex
    whose normal (either way) meets a facing sheet of the mesh within LAYER_REACH, not one of its own triangles."""
    bvh = BVHTree.FromPolygons(co.tolist(), tris.tolist(), all_triangles=True)
    own = [set() for _ in range(len(co))]
    for t, (a, b, c) in enumerate(tris.tolist()):
        own[a].add(t)
        own[b].add(t)
        own[c].add(t)
    face_n = tri_normals(co, tris)
    face_n /= np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), 1e-12)
    rows, faces, hits, dists = [], [], [], []
    for v, (p, n) in enumerate(zip(co.tolist(), normals.tolist())):
        best = None
        for sign in (1.0, -1.0):
            d = Vector(n) * sign
            origin = Vector(p)
            travelled = 0.0
            for _try in range(4):
                hit, _normal, t, dist = bvh.ray_cast(origin, d, LAYER_REACH - travelled)
                if hit is None:
                    break
                travelled += dist
                if t in own[v] or travelled < LAYER_MIN:
                    origin = hit + d * 1e-6
                    travelled += 1e-6
                    continue
                if abs(float(np.dot(face_n[t], n))) >= LAYER_FACING and (best is None or travelled < best[2]):
                    best = (t, tuple(hit), travelled)
                break
        if best is not None:
            rows.append(v)
            faces.append(best[0])
            hits.append(best[1])
            dists.append(best[2])
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0)
    rows, faces = np.array(rows), np.array(faces)
    hits = np.array(hits)
    corners = tris[faces]
    bary = _barycentric(hits, co[corners[:, 0]], co[corners[:, 1]], co[corners[:, 2]])
    return rows, faces, bary, np.array(dists)


class Detail:
    """Each vertex's Laplacian in its own frame, with weights and frames' neighbours fixed on the start shape."""

    def __init__(self, co, tris):
        self.tris = tris
        i, j, k = tris[:, 0], tris[:, 1], tris[:, 2]
        rows, cols, weights = [], [], []
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            # The cotangent at corner a weighs edge b-c.
            u, w = co[b] - co[a], co[c] - co[a]
            cot = (u * w).sum(1) / np.maximum(np.linalg.norm(np.cross(u, w), axis=1), 1e-12)
            cot = np.clip(0.5 * cot, 0.05, 20.0)
            rows += [b, c]
            cols += [c, b]
            weights += [cot, cot]
        self.rows = np.concatenate(rows)
        self.cols = np.concatenate(cols)
        self.weights = np.concatenate(weights)
        self.count = len(co)
        self.total = np.bincount(self.rows, self.weights, self.count)
        # A fixed neighbour per vertex for the frame's tangent.
        neighbour = np.full(self.count, -1, dtype=np.int64)
        neighbour[self.rows[::-1]] = self.cols[::-1]
        self.neighbour = neighbour
        self.used = (self.total > 0) & (neighbour >= 0)
        lengths = np.linalg.norm(co[self.rows] - co[self.cols], axis=1)
        self.edge = np.bincount(self.rows, lengths, self.count) / np.maximum(np.bincount(self.rows, None, self.count), 1)

    def local(self, co):
        mean = np.stack([np.bincount(self.rows, self.weights * co[self.cols, k], self.count) for k in range(3)], 1)
        delta = co - mean / np.maximum(self.total, 1e-12)[:, None]
        n = vertex_normals(co, self.tris)
        t = co[np.maximum(self.neighbour, 0)] - co
        t -= (t * n).sum(1, keepdims=True) * n
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        b = np.cross(n, t)
        return np.stack(((delta * t).sum(1), (delta * b).sum(1), (delta * n).sum(1)), axis=1)

    def change(self, before, after):
        """(normal part, whole change, change relative to the vertex's mean edge) per vertex, NaN where unused."""
        d = self.local(after) - self.local(before)
        normal = np.where(self.used, np.abs(d[:, 2]), np.nan)
        whole = np.where(self.used, np.linalg.norm(d, axis=1), np.nan)
        return normal, whole, whole / np.maximum(self.edge, 1e-9)


def stats(values, weights=None):
    """{n, mean, p50, p95, max} of ``values`` (mm when lengths), area-weighted mean when ``weights``."""
    values = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(values)
    values = values[keep]
    if not len(values):
        return {"n": 0}
    out = {"n": int(len(values)), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
           "max": float(values.max())}
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[keep]
        out["mean"] = float((values * w).sum() / max(w.sum(), 1e-30))
    else:
        out["mean"] = float(values.mean())
    return out


def worst(values, co, extra=None, k=WORST):
    """The ``k`` largest values with their vertex and position (and ``extra`` {name: per-vertex array})."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.where(np.isfinite(values), -values, np.inf))[:k]
    out = []
    for v in order:
        if not np.isfinite(values[v]):
            break
        row = {"v": int(v), "value": round(float(values[v]), 3), "at": [round(float(x), 4) for x in co[v]]}
        for name, array in (extra or {}).items():
            item = array[v]
            row[name] = item if isinstance(item, str) else round(float(item), 3)
        out.append(row)
    return out


class Garment:
    """What the bench measures of one garment, made once from its From shape: ``points`` (all of its vertices),
    ``tris`` (triangles of all of its meshes), ``moving`` (a mask: not skin)."""

    def __init__(self, points, tris, moving, from_surface, body_regions, region_names, rigid=None):
        self.index, first, wtris = weld(points, tris)
        self.co = np.asarray(points, dtype=np.float64)[first]
        self.first = first
        moving_w = np.zeros(len(self.co), dtype=bool)
        moving_w[self.index[np.asarray(moving, dtype=bool)]] = True
        self.moving = moving_w
        self.rigid = (np.asarray(rigid)[first] if rigid is not None else np.full(len(first), -1)).astype(np.int64)
        self.tris = wtris
        self.area = vertex_areas(self.co, self.tris)
        self.normals = vertex_normals(self.co, self.tris)
        self.h0, face, bary, beyond, found = from_surface.nearest(self.co)
        self.ok0 = found & ~beyond
        corner = from_surface.tris[np.maximum(face, 0), np.argmax(bary, axis=1)]
        self.body_vertex = np.where(found, corner, -1)
        self.region = np.where(found, body_regions[np.maximum(corner, 0)], -1)
        self.region_names = list(region_names)
        mt = self.tris[self.moving[self.tris].all(axis=1)]
        self.layers = layer_pairs(self.co, mt, self.normals) if len(mt) else layer_pairs(self.co, self.tris[:0],
                                                                                           self.normals)
        self.detail = Detail(self.co, mt) if len(mt) else None
        self.edges = edges_of(mt) if len(mt) else np.zeros((0, 2), np.int64)
        self.moving_tris = mt

    def welded(self, positions):
        return np.asarray(positions, dtype=np.float64)[self.first]

    def region_name(self, v):
        r = self.region[v]
        return self.region_names[r] if r >= 0 else "-"

    def measure(self, after, to_surface, body=None, back=None, pv=None):
        """Metrics for the garment moved to ``after`` (all vertices) on the body ``to_surface``. ``body`` (the
        welded body's Detail, its From and To positions, and the welded index of each body vertex) gives the
        body's own detail change under the garment, ``back`` the positions after the round trip. Fills ``pv``
        (a dict) with per-vertex arrays when given."""
        co, moved = self.co, self.welded(after)
        out = {}
        h1, _face, _bary, beyond, found = to_surface.nearest(moved)
        ok = self.ok0 & found & ~beyond
        err = h1 - self.h0
        regions = np.array([self.region_name(v) for v in range(len(co))]) if len(co) else np.zeros(0, str)
        clothing = ok & self.moving
        bands = {}
        for low, high, name in BANDS:
            band = clothing & (self.h0 >= low) & (self.h0 < high)
            row = stats(np.abs(err[band]) * MM, self.area[band])
            if row["n"]:
                row["signed_mean"] = float((err[band] * self.area[band]).sum() / max(self.area[band].sum(), 1e-30) * MM)
                row["sink_1mm"] = int((err[band] < -CLIP).sum())
            bands[name] = row
        clip_out = clothing & (self.h0 >= 0.0) & (h1 < -CLIP)
        inside = clothing & (h1 < -CLIP)
        out["clearance"] = {
            "bands": bands,
            "clip_out": int(clip_out.sum()), "clip_out_cm2": float(self.area[clip_out].sum() * 1e4),
            "inside_1mm": int(inside.sum()), "inside_1mm_before": int((clothing & (self.h0 < -CLIP)).sum()),
            "deepest_mm": float(h1[clothing].min() * MM) if clothing.any() else 0.0,
            "worst": worst(np.where(clothing & (self.h0 < 0.015), np.abs(err) * MM, np.nan), co,
                           {"h0": self.h0 * MM, "h1": h1 * MM, "region": regions}),
        }
        by_region = {}
        tight = clothing & (self.h0 < 0.015)
        for r, name in enumerate(self.region_names):
            here = tight & (self.region == r)
            if here.sum() >= 20:
                by_region[name] = {"n": int(here.sum()),
                                   "mean_abs_mm": float((np.abs(err[here]) * self.area[here]).sum()
                                                        / max(self.area[here].sum(), 1e-30) * MM),
                                   "clip_out": int((clip_out & (self.region == r)).sum())}
        out["clearance"]["regions"] = by_region
        rigid = ok & self.moving & (self.rigid >= 0)
        if rigid.any():
            # Rigid parts worn on the skin: some vertex within RIGID_TOUCH of it on the From shape, or inside.
            closest = np.full(self.rigid.max() + 1, np.inf)
            np.minimum.at(closest, self.rigid[rigid], self.h0[rigid])
            rigid &= closest[np.maximum(self.rigid, 0)] < RIGID_TOUCH
        if rigid.any():
            labels, inverse = np.unique(self.rigid[rigid], return_inverse=True)
            part_err = np.bincount(inverse, err[rigid]) / np.bincount(inverse) * MM
            part_h0 = np.bincount(inverse, self.h0[rigid]) / np.bincount(inverse) * MM
            part_at = np.stack([np.bincount(inverse, co[rigid][:, k]) / np.bincount(inverse) for k in range(3)], 1)
            out["rigid"] = dict(stats(np.abs(part_err)), over_1mm=int((np.abs(part_err) > 1.0).sum()),
                                worst=worst(np.abs(part_err), part_at, {"signed": part_err, "h0": part_h0}))
        skin = ok & ~self.moving
        if skin.any():
            out["skin"] = stats(np.abs(h1[skin]) * MM, self.area[skin])
        # Thickness
        rows, faces, bary, t0 = self.layers
        if len(rows):
            corners = self.tris_of(faces)
            spot = (moved[corners] * bary[:, :, None]).sum(axis=1)
            t1 = np.linalg.norm(moved[rows] - spot, axis=1)
            dt = np.abs(t1 - t0) * MM
            out["thickness"] = dict(stats(dt), over_1mm=int((dt > 1.0).sum()),
                                    worst=worst(_scatter(len(co), rows, dt), co,
                                                {"t0": _scatter(len(co), rows, t0 * MM), "region": regions}))
        else:
            out["thickness"] = {"n": 0}
        # Detail
        if self.detail is not None:
            normal, whole, relative = self.detail.change(co, moved)
            normal = np.where(self.moving, normal, np.nan)
            out["detail"] = {"normal_mm": stats(normal * MM), "relative": stats(np.where(self.moving, relative, np.nan)),
                             "worst": worst(normal * MM, co, {"h0": self.h0 * MM, "region": regions})}
            if body is not None:
                bd, bfrom, bto, bindex = body
                near = self.body_vertex[self.moving & self.ok0 & (np.abs(self.h0) < 0.02) & (self.body_vertex >= 0)]
                near = np.unique(bindex[near])
                if len(near):
                    bn, _bw, _br = bd.change(bfrom, bto)
                    out["detail"]["body_floor_mm"] = stats(bn[near] * MM)
        # Flips and stretch
        if len(self.moving_tris):
            n0 = tri_normals(co, self.moving_tris)
            n1 = tri_normals(moved, self.moving_tris)
            big = np.linalg.norm(n0, axis=1) > 1e-12
            out["flipped"] = int(((n0 * n1).sum(1) < 0)[big].sum())
            e = self.edges
            l0 = np.linalg.norm(co[e[:, 0]] - co[e[:, 1]], axis=1)
            l1 = np.linalg.norm(moved[e[:, 0]] - moved[e[:, 1]], axis=1)
            good = l0 > 1e-6
            ratio = np.abs(np.log(np.maximum(l1[good], 1e-12) / l0[good]))
            out["stretch"] = {"p99_log": float(np.percentile(ratio, 99)) if len(ratio) else 0.0,
                              "max_ratio": float(np.exp(ratio.max())) if len(ratio) else 1.0}
        out["moved_max_mm"] = float(np.linalg.norm(moved - co, axis=1).max() * MM) if len(co) else 0.0
        if back is not None:
            b = self.welded(back)
            gap = np.where(self.moving, np.linalg.norm(b - co, axis=1) * MM, np.nan)
            out["round_trip"] = dict(stats(gap), worst=worst(gap, co, {"h0": self.h0 * MM, "region": regions}))
        if pv is not None:
            pv.update(co=co.astype(np.float32), moved=moved.astype(np.float32), h0=self.h0.astype(np.float32),
                      h1=h1.astype(np.float32), ok=ok, moving=self.moving, region=self.region.astype(np.int16))
            if self.detail is not None:
                pv["detail_normal"] = normal.astype(np.float32)
            if back is not None:
                pv["round_trip"] = (gap / MM).astype(np.float32)
        return out

    def tris_of(self, faces):
        return self.moving_tris[faces]


def _scatter(count, rows, values):
    out = np.full(count, np.nan)
    out[rows] = values
    return out
