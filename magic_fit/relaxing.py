# SPDX-License-Identifier: GPL-3.0-or-later

"""Relaxing a mesh so its texture stops being stretched.

Editing (sculpting, proportional editing, Body Fit...) moves vertices unevenly, and the texture
mapped onto them stretches and squashes with them. The Texture Relax brush slides vertices along the
surface, without changing its shape, to where the UV map says they belong relative to their
neighbours.

Per vertex, and per UV chart it lies in, the positions of its neighbours are fitted as an affine
function of their UV coordinates. Where that function puts the vertex's own UV coordinate is its
target. Where the mapping is already affine across the neighbourhood (evenly stretched, whatever the
stretch and however dense the vertices), the target is the current position: an undistorted texture
is left alone, and the brush carries the texel density of the surroundings into the area it fixes
rather than forcing a uniform one.

Each dab moves the vertices under the brush strength x falloff of the way to their targets, then
puts them back onto the surface as it was when the stroke began: each vertex is projected onto the
triangles around the vertex it is currently closest to, and that "home" vertex is updated as it
slides along. Nothing is ever snapped to another layer of the mesh. Vertices on the open edge of a
mesh slide along that edge, move freely or stay, as chosen.

Vertices sharing a position (split seams, touching parts, several objects) are welded into one
group that always moves by the same amount, so seams stay closed.

Like `painting` and `fitting`, nothing here depends on the 3D view.
"""

import numpy as np

from .fitting import WELD_DISTANCE, _weld
from .painting import falloff_weights

# At full strength, a dab moves a vertex this fraction of the way to its target. Whole steps would
# make vertices on open edges and split seams, whose fits extrapolate, swing back and forth.
MAX_RATE = 0.5
# No dab moves a vertex farther than this fraction of the length of its edges, so it never leaves the
# triangles around its home vertex, whatever the brush does.
MAX_STEP = 0.5
# UV coordinates closer than this are the same chart at a vertex.
UV_EPSILON = 1e-5
# The fit needs neighbours spread out in the UV plane: below this (scaled) determinant they are
# nearly on a line and the fit is dropped.
MIN_SPREAD = 1e-4


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _rows(keys, count):
    """Compressed rows for ``keys``: (order of the entries by key, row pointers)."""
    order = np.argsort(keys, kind='stable')
    ptr = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=count), out=ptr[1:])
    return order, ptr


def _unique_int_rows(rows):
    """Distinct rows of an integer (n, k) array: (row -> distinct index, first row of each distinct
    row). Much faster than ``np.unique(axis=0)``."""
    if not len(rows):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = np.lexsort(rows.T[::-1])
    ordered = rows[order]
    new = np.ones(len(rows), dtype=bool)
    new[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    inverse = np.empty(len(rows), dtype=np.int64)
    inverse[order] = np.cumsum(new) - 1
    return inverse, order[new]


def _gather(ptr, values, keys):
    """For compressed rows (``values[ptr[k]:ptr[k + 1]]`` belongs to row k): the values of rows
    ``keys``, and for each the position in ``keys`` of its row."""
    starts = ptr[keys]
    lengths = ptr[keys + 1] - starts
    total = int(lengths.sum())
    owner = np.repeat(np.arange(len(keys)), lengths)
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return values[starts[owner] + offsets], owner


def _mean_rows(values, keys, count):
    """Mean of the rows of ``values`` per ``key`` (0..count-1), and how many rows each key had."""
    total = np.zeros((count,) + values.shape[1:])
    np.add.at(total, keys, values)
    hits = np.bincount(keys, minlength=count).astype(np.float64)
    shape = (count,) + (1,) * (values.ndim - 1)
    return total / np.maximum(hits, 1.0).reshape(shape), hits


def _closest_on_triangles(p, a, b, c):
    """Closest points to ``p`` on triangles ``a, b, c`` (all (n, 3)).

    Returns (point, barycentric coordinates of it, where the triangle's plane is closest to ``p``,
    whether the triangle has a plane). For a triangle without area, the closest point stands in for
    its plane's.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = _dot(ab, ab)
    d2 = _dot(ab, ac)
    d3 = _dot(ac, ac)
    d4 = _dot(ap, ab)
    d5 = _dot(ap, ac)
    det = d1 * d3 - d2 * d2
    ok = det > 1e-30 * np.maximum(d1 * d3, 1e-300)
    det = np.where(ok, det, 1.0)
    v = np.where(ok, (d3 * d4 - d2 * d5) / det, -1.0)
    w = np.where(ok, (d1 * d5 - d2 * d4) / det, -1.0)
    u = 1.0 - v - w
    plane = a + v[:, None] * ab + w[:, None] * ac
    inside = ok & (u >= 0.0) & (v >= 0.0) & (w >= 0.0)

    best = plane.copy()
    best_bary = np.stack((u, v, w), axis=1)
    best_distance = np.where(inside, _dot(p - plane, p - plane), np.inf)
    for start, end, corners in ((a, b, (0, 1)), (b, c, (1, 2)), (c, a, (2, 0))):
        edge = end - start
        length = np.maximum(_dot(edge, edge), 1e-300)
        t = np.clip(_dot(p - start, edge) / length, 0.0, 1.0)
        point = start + t[:, None] * edge
        distance = _dot(p - point, p - point)
        closer = distance < best_distance
        best_distance = np.where(closer, distance, best_distance)
        best[closer] = point[closer]
        bary = np.zeros((len(p), 3))
        bary[:, corners[0]] = 1.0 - t
        bary[:, corners[1]] = t
        best_bary[closer] = bary[closer]
    plane[~ok] = best[~ok]
    return best, best_bary, plane, ok


class RelaxStroke:
    """Slides a mesh's vertices along its surface so its texture is evenly mapped, dab by dab.

    ``co`` holds world space positions, updated as vertices move. ``tris`` are the mesh's triangles
    (all of them, hidden ones included: they are the surface vertices slide on) and ``tri_uvs``
    their corners' UV coordinates, (m, 3, 2), NaN for triangles without a UV map. ``movable`` marks
    the vertices the brush may move.

    ``boundary`` says what vertices on the open edge of the mesh do: 'SLIDE' along that edge (its
    shape is kept), 'FREE' to wherever the texture wants them (the outline may shrink or grow),
    'FIXED' nothing. Vertices closer together than ``weld_distance`` always move together.
    """

    def __init__(self, co, tris, tri_uvs, movable, *, boundary='SLIDE', weld_distance=WELD_DISTANCE):
        self.co = co
        self.start = co.copy()
        self.boundary = boundary
        self.moved = np.zeros(len(co), dtype=bool)
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        self.tris = tris

        # Welded groups: their members, positions and displacement so far.
        self.weld, count, weld_edges = _weld(co, tris, weld_distance)
        self.weld_count = count
        self.members, self.member_ptr = _rows(self.weld, count)
        self.group_start, _hits = _mean_rows(co, self.weld, count)
        self.displacement = np.zeros((count, 3))
        together = np.ones(count, dtype=bool)
        np.logical_and.at(together, self.weld, movable)

        # Each group's step limit: half the length of its edges.
        lengths = np.linalg.norm(self.group_start[weld_edges[:, 0]] - self.group_start[weld_edges[:, 1]], axis=1)
        limit = np.zeros(count)
        np.add.at(limit, weld_edges[:, 0], lengths)
        np.add.at(limit, weld_edges[:, 1], lengths)
        degree = np.bincount(weld_edges.ravel(), minlength=count).astype(np.float64)
        self.limit = MAX_STEP * limit / np.maximum(degree, 1.0)

        # The triangles around each group, for sliding along, and each group's home group.
        corners = self.weld[tris]
        # Smooth normals, weighted by the corner angles: unlike area weights, these don't tilt with an
        # uneven fan, such as the one-sided fan of a vertex on an open edge.
        a, b, c = (self.group_start[corners[:, k]] for k in range(3))
        face_normals = np.cross(b - a, c - a)
        face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-300)
        normals = np.zeros((count, 3))
        for k in range(3):
            p, q, r = (self.group_start[corners[:, (k + j) % 3]] for j in range(3))
            pq = q - p
            pr = r - p
            cosine = _dot(pq, pr) / np.maximum(np.linalg.norm(pq, axis=1) * np.linalg.norm(pr, axis=1), 1e-300)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            np.add.at(normals, corners[:, k], angle[:, None] * face_normals)
        self.normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-300)
        owner = np.repeat(np.arange(len(tris)), 3)
        pairs = np.unique(corners.ravel().astype(np.int64) * max(len(tris), 1) + owner)
        fan_group, fan_tri = pairs // max(len(tris), 1), pairs % max(len(tris), 1)
        order, self.fan_ptr = _rows(fan_group, count)
        self.fan_tris = fan_tri[order]
        self.home = np.arange(count)

        # Open edges: welded edges with one triangle. Groups on them and their two edge neighbours.
        sides = np.concatenate((corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]))
        sides = np.sort(sides[sides[:, 0] != sides[:, 1]], axis=1)
        keys, hits = np.unique(sides[:, 0] * count + sides[:, 1], return_counts=True)
        open_edges = np.stack((keys // count, keys % count), axis=1)[hits == 1]
        self.on_boundary = np.zeros(count, dtype=bool)
        self.on_boundary[open_edges.ravel()] = True
        both = np.concatenate((open_edges, open_edges[:, ::-1]))
        order, ptr = _rows(both[:, 0], count)
        open_degree = np.diff(ptr)
        self.edge_neighbours = np.full((count, 2), -1, dtype=np.int64)
        regular = np.flatnonzero(open_degree == 2)
        self.edge_neighbours[regular, 0] = both[order, 1][ptr[regular]]
        self.edge_neighbours[regular, 1] = both[order, 1][ptr[regular] + 1]
        if boundary == 'FIXED':
            together &= ~self.on_boundary
        elif boundary == 'SLIDE':
            # Corners where more than two open edges meet can't slide along one edge.
            together &= ~(self.on_boundary & (open_degree != 2))
        self.movable = together[self.weld]

        self._build_charts(tris, tri_uvs)

    def _build_charts(self, tris, tri_uvs):
        """Per vertex and UV chart: its neighbours and their UV offsets, for the affine fit."""
        m = len(tris)
        tri_uvs = np.asarray(tri_uvs, dtype=np.float64).reshape(-1, 3, 2)
        verts, uvs, neighbours, offsets = [], [], [], []
        for k in range(3):
            for j in ((k + 1) % 3, (k + 2) % 3):
                verts.append(tris[:, k])
                uvs.append(tri_uvs[:, k])
                neighbours.append(tris[:, j])
                offsets.append(tri_uvs[:, j] - tri_uvs[:, k])
        verts = np.concatenate(verts) if m else np.zeros(0, dtype=np.int64)
        uvs = np.concatenate(uvs) if m else np.zeros((0, 2))
        neighbours = np.concatenate(neighbours) if m else np.zeros(0, dtype=np.int64)
        offsets = np.concatenate(offsets) if m else np.zeros((0, 2))
        keep = np.isfinite(uvs).all(axis=1) & np.isfinite(offsets).all(axis=1)
        keep &= self.weld[verts] != self.weld[neighbours] if m else keep
        verts, uvs, neighbours, offsets = verts[keep], uvs[keep], neighbours[keep], offsets[keep]

        # A chart is a vertex together with one of its UV coordinates.
        quantized = np.round(uvs / UV_EPSILON).astype(np.int64)
        chart, _first = _unique_int_rows(np.stack((verts, quantized[:, 0], quantized[:, 1]), axis=1))
        chart_count = int(chart.max()) + 1 if len(chart) else 0
        # A neighbour shared by two faces of the chart counts once.
        _pair, first = _unique_int_rows(np.stack((chart, neighbours), axis=1))
        chart, verts, neighbours, offsets = chart[first], verts[first], neighbours[first], offsets[first]
        # Offsets scaled per chart, so the fit is conditioned the same whatever the UV scale.
        scale, _hits = _mean_rows(_dot(offsets, offsets), chart, chart_count)
        scale = np.sqrt(np.maximum(scale, 1e-300))
        offsets = offsets / scale[chart][:, None]

        order, self.chart_ptr = _rows(chart, chart_count)
        self.chart_neighbours = neighbours[order]
        self.chart_offsets = offsets[order]
        chart_vert = np.zeros(chart_count, dtype=np.int64)
        chart_vert[chart] = verts
        # Group -> its members' charts.
        chart_group = self.weld[chart_vert]
        order, self.group_chart_ptr = _rows(chart_group, self.weld_count)
        self.group_charts = np.arange(chart_count)[order]

    # -------------------------------------------------------------------------
    # Public API

    @property
    def group_co(self):
        return self.group_start + self.displacement

    def dab(self, center, radius, strength, falloff='SMOOTH'):
        """Apply one dab around the world space point ``center``. Returns the indices of moved vertices."""
        nothing = np.empty(0, dtype=np.int64)
        if radius <= 0.0 or strength <= 0.0:
            return nothing
        center = np.asarray(center, dtype=np.float64)
        delta = self.co - center
        in_reach = self.movable & (_dot(delta, delta) < radius * radius)
        groups = np.unique(self.weld[in_reach])
        if not len(groups):
            return nothing
        position = self.group_co[groups]
        t = np.linalg.norm(position - center, axis=1) / radius
        inside = t < 1.0
        groups, position, t = groups[inside], position[inside], t[inside]
        if not len(groups):
            return nothing
        alpha = np.minimum(falloff_weights(falloff, t) * strength, 1.0) * MAX_RATE

        target, valid = self._targets(groups, position)
        step = alpha[:, None] * (target - position)
        # Only the part along the surface: the fit's intercept always sits a little inside a curved
        # surface, and snapping that back would creep sideways, dab after dab.
        normals = self.normals[groups]
        step -= _dot(step, normals)[:, None] * normals
        length = np.linalg.norm(step, axis=1)
        limit = self.limit[groups]
        step *= np.where(length > limit, limit / np.maximum(length, 1e-300), 1.0)[:, None]
        moving = valid & (length > 1e-9)
        if not moving.any():
            return nothing
        groups, step = groups[moving], step[moving]
        landed = self._slide(groups, self.group_co[groups] + step)
        self.displacement[groups] += landed - self.group_co[groups]

        members, owner = _gather(self.member_ptr, self.members, groups)
        self.co[members] = self.start[members] + self.displacement[groups][owner]
        self.moved[members] = True
        return members

    # -------------------------------------------------------------------------
    # Internals

    def _targets(self, groups, position):
        """Where the texture wants ``groups`` (at ``position``): (targets, whether one was found).
        Vertices on an open edge get the same fit; `_slide` then keeps them on the edge."""
        fit, valid = self._fit(groups)
        return np.where(valid[:, None], fit, position), valid

    def _fit(self, groups):
        """Affine fit per chart of the members of ``groups``, averaged per group."""
        charts, owner = _gather(self.group_chart_ptr, self.group_charts, groups)
        count = len(groups)
        target = np.zeros((count, 3))
        valid = np.zeros(count, dtype=bool)
        if not len(charts):
            return target, valid
        entries, chart_owner = _gather(self.chart_ptr, np.arange(len(self.chart_neighbours)), charts)
        offsets = self.chart_offsets[entries]
        positions = self.co[self.chart_neighbours[entries]]
        # Rows x = (du, dv, 1): solve (sum x x^T) M^T = sum x p^T; the target is M's last column.
        x = np.concatenate((offsets, np.ones((len(entries), 1))), axis=1)
        xtx = np.zeros((len(charts), 3, 3))
        np.add.at(xtx, chart_owner, x[:, :, None] * x[:, None, :])
        xtp = np.zeros((len(charts), 3, 3))
        np.add.at(xtp, chart_owner, x[:, :, None] * positions[:, None, :])
        hits = np.bincount(chart_owner, minlength=len(charts))
        spread = np.linalg.det(xtx) / np.maximum(hits, 1) ** 3
        ok = (hits >= 3) & (spread > MIN_SPREAD)
        solution = np.zeros((len(charts), 3))
        if ok.any():
            solution[ok] = np.linalg.solve(xtx[ok], xtp[ok])[:, 2, :]
        np.add.at(target, owner[ok], solution[ok])
        found = np.bincount(owner[ok], minlength=count)
        valid = found > 0
        target /= np.maximum(found, 1)[:, None]
        return target, valid

    def _slide(self, groups, wanted):
        """Put ``groups``, headed for ``wanted``, back onto the surface as it was at the start, and
        move their homes along. Returns where they land."""
        landed = wanted.copy()
        home = self.home[groups]
        sliding = self.on_boundary[groups] if self.boundary == 'SLIDE' else np.zeros(len(groups), dtype=bool)

        surface = ~sliding
        if surface.any():
            which = np.flatnonzero(surface)
            fan, owner = _gather(self.fan_ptr, self.fan_tris, home[which])
            corners = self.weld[self.tris[fan]]
            p = wanted[which][owner]
            a, b, c = (self.group_start[corners[:, k]] for k in range(3))
            point, bary, plane, has_plane = _closest_on_triangles(p, a, b, c)
            distance = _dot(p - point, p - point)
            # Open edges may move past the surface's edge, along the plane of the nearest triangle
            # that has one.
            loose = self.on_boundary[groups[which]] & (self.boundary == 'FREE')
            # The closest triangle of each group's fan; for those open edges, the closest with a plane.
            order = np.lexsort((distance, loose[owner] & ~has_plane, owner))
            first = np.flatnonzero(np.diff(np.concatenate(([-1], owner[order]))) > 0)
            best = order[first]
            found = owner[best]
            result = point[best]
            loose = loose[found]
            result[loose] = plane[best][loose]
            landed[which[found]] = result
            home[which[found]] = corners[best, np.argmax(bary[best], axis=1)]

        if sliding.any():
            which = np.flatnonzero(sliding)
            here = home[which]
            p = wanted[which]
            best_point = p.copy()
            best_distance = np.full(len(which), np.inf)
            best_home = here.copy()
            for side in range(2):
                other = self.edge_neighbours[here, side]
                ok = other >= 0
                start = self.group_start[here]
                end = self.group_start[np.maximum(other, 0)]
                edge = end - start
                t = np.clip(_dot(p - start, edge) / np.maximum(_dot(edge, edge), 1e-300), 0.0, 1.0)
                point = start + t[:, None] * edge
                distance = np.where(ok, _dot(p - point, p - point), np.inf)
                closer = distance < best_distance
                best_distance = np.where(closer, distance, best_distance)
                best_point[closer] = point[closer]
                # A corner where other than two open edges meet has no edge to slide on from, so the
                # home stays short of it.
                onward = (t > 0.5) & (self.edge_neighbours[np.maximum(other, 0), 0] >= 0)
                best_home[closer] = np.where(onward[closer], other[closer], here[closer])
            landed[which] = best_point
            home[which] = best_home

        self.home[groups] = home
        return landed
