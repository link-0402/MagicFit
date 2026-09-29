# SPDX-License-Identifier: GPL-3.0-or-later

"""Straightening the parts of a mesh that sag into creases, as the posed or scaled mesh shows them.

Weights copied from a body are right where clothing lies on the skin, but not where it spans a gap: fabric
across the cleavage, under the breasts or between the buttocks, or a strap running from the breast down to
where it touches the body again, copies the weights of skin it never touches. Scale the body up (as a
Customize+ template does) and that fabric follows the skin into the crease: it sags in, or folds over
itself, instead of staying stretched between the parts it hangs from.

Straighten works on what is displayed. When a stroke ends, it looks at the mesh around the stroke: as much
surface again as REACH^2 - 1 brush disks (of at least MIN_RADIUS), found by walking out along the mesh, so
it reaches far along a narrow strap and a moderate way across a wide dress. For each vertex there, it
measures how far the vertex lies below the convex hull of that area, along its normal, as displayed and in
the rest pose. Where the displayed mesh lies deeper than the rest pose by more than SINK, connected to the
stroke, it sags: that's the span.

The span's bone weights become a smooth (harmonic) blend of the weights around it, so it moves with what it
hangs between and keeps its rest shape relative to them, in every pose and scaling. The vertices around it
that hold it out (lying on the hull as displayed, like the fronts of the breasts on either side of the
cleavage) carry it most: the blend from them alone is faded into the blend from all of its surroundings
toward the parts that don't (the chest above and below the cleavage), so the weights stay continuous there.
This is "weight inpainting", as used for robust skin weight transfer, with the gap to fill found in the pose.

Like `painting`, nothing here depends on the 3D view.
"""

import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .fitting import WELD_DISTANCE, weld_groups
from .painting import ZERO_WEIGHT, WeightEdit, deform_bone_names, read_geometry

SOLVE_ITERATIONS = 3000
SOLVE_TOLERANCE = 1e-14
# Cotangent edge weights below this fraction of the average are raised to it: keeps every weight
# positive, so blended weights stay between those they're blended from.
MIN_EDGE_WEIGHT = 1e-3
# A stroke looks at the mesh around it: as much surface again as REACH^2 - 1 disks of the brush's size (at
# least MIN_RADIUS, so a small brush still sees the parts a crease or strap hangs between), but no farther
# than MAX_REACH along the mesh.
REACH = 2.5
MIN_RADIUS = 0.05
MAX_REACH = 0.4
# Vertices lying deeper below the hull as displayed than in the rest pose by more than this sag.
SINK = 0.0015
# Vertices around a span lying no deeper than this below the hull as displayed hold it out.
HOLD = 0.002
# Vertices this close together in the rest pose count as one spot for sagging (both layers of a strap
# that has a thickness).
LAYER = 0.004
# A mesh whose vertices all move less than this between the rest pose and the display isn't posed.
MIN_MOVE = 1e-4


def deforming_armature(obj):
    """The first Armature modifier deforming ``obj`` in the viewport through vertex groups, or None."""
    for mod in obj.modifiers:
        if (
                mod.type == 'ARMATURE' and mod.show_viewport and mod.use_vertex_groups and
                mod.object is not None and mod.object.type == 'ARMATURE'
        ):
            return mod
    return None


def _armatures(objects):
    result = []
    for obj in objects:
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object is not None and mod.object.type == 'ARMATURE':
                result.append(mod.object)
        if obj.parent is not None and obj.parent.type == 'ARMATURE' and obj.parent_type == 'ARMATURE':
            result.append(obj.parent)
    return result


def rest_geometry(context, objects, sources):
    """`read_geometry` of each of ``objects`` (with the matching ``sources``) with the armatures that
    deform them in their rest position: what the mesh looks like before posing and bone scaling. Poses
    are put back right after."""
    switched = []
    try:
        for arm in _armatures(objects):
            data = arm.data
            if data.pose_position == 'POSE' and data not in switched:
                try:
                    data.pose_position = 'REST'
                except (AttributeError, TypeError, RuntimeError):
                    continue  # read-only (linked) armature: judge as displayed
                switched.append(data)
        depsgraph = context.evaluated_depsgraph_get()
        return [read_geometry(obj, depsgraph, source=source) for obj, source in zip(objects, sources)]
    finally:
        for data in switched:
            data.pose_position = 'POSE'
        if switched:
            context.evaluated_depsgraph_get()


def _gather(ptr, values, keys):
    """For compressed rows (``values[ptr[k]:ptr[k + 1]]`` belongs to row k): the values of rows
    ``keys``, and for each the position in ``keys`` of its row."""
    starts = ptr[keys]
    lengths = ptr[keys + 1] - starts
    total = int(lengths.sum())
    owner = np.repeat(np.arange(len(keys)), lengths)
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return values[starts[owner] + offsets], owner


def _rows(keys, count):
    """Compressed rows for ``keys``: (order of the entries by key, row pointers)."""
    order = np.argsort(keys, kind='stable')
    ptr = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=count), out=ptr[1:])
    return order, ptr


def cotangent_weights(co, tris):
    """Edges and their cotangent weights (made positive) of the triangle mesh ``co``, ``tris``."""
    first, second, weight = [], [], []
    for k in range(3):
        i, j, opposite = tris[:, k], tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
        u = co[i] - co[opposite]
        v = co[j] - co[opposite]
        area = np.linalg.norm(np.cross(u, v), axis=1)
        first.append(i)
        second.append(j)
        weight.append(0.5 * np.einsum("ij,ij->i", u, v) / np.maximum(area, 1e-20))
    a = np.concatenate(first)
    b = np.concatenate(second)
    low, high = np.minimum(a, b), np.maximum(a, b)
    keep = low != high
    count = max(int(high.max(initial=0)) + 1, 1)
    keys, inverse = np.unique(low[keep] * count + high[keep], return_inverse=True)
    weights = np.bincount(inverse.reshape(-1), weights=np.concatenate(weight)[keep])
    if len(weights):
        weights = np.maximum(weights, MIN_EDGE_WEIGHT * np.abs(weights).mean())
    return np.stack((keys // count, keys % count), axis=1), weights


def _conjugate_gradient(multiply, rhs, guess, scale, iterations=SOLVE_ITERATIONS, tolerance=SOLVE_TOLERANCE):
    """Solve ``multiply(x) = rhs`` (symmetric positive definite) for each column of ``rhs``, with
    ``scale`` (one value per row, 1 / the diagonal) as preconditioner."""
    x = guess.copy()
    # A column without a right-hand side is 0: starting it anywhere else, it would have to get down to
    # the tolerance relative to nothing, which takes far more iterations than the others.
    x[:, ~rhs.any(axis=0)] = 0.0
    residual = rhs - multiply(x)
    scaled = residual * scale[:, None]
    direction = scaled.copy()
    product_rs = np.einsum("ij,ij->j", residual, scaled)
    limit = tolerance * np.maximum(np.einsum("ij,ij->j", rhs, rhs), 1e-30)
    for _ in range(iterations):
        if (np.einsum("ij,ij->j", residual, residual) <= limit).all():
            break
        product = multiply(direction)
        alpha = product_rs / np.maximum(np.einsum("ij,ij->j", direction, product), 1e-300)
        x += alpha * direction
        residual -= alpha * product
        scaled = residual * scale[:, None]
        product_next = np.einsum("ij,ij->j", residual, scaled)
        direction = scaled + (product_next / np.maximum(product_rs, 1e-300)) * direction
        product_rs = product_next
    return x


def vertex_normals(co, tris):
    """Area-weighted vertex normals of the triangle mesh ``co``, ``tris``."""
    face_normals = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    normals = np.zeros_like(co)
    for k in range(3):
        np.add.at(normals, tris[:, k], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)


def sinking(points, normals, which):
    """How far each of ``which`` lies below the convex hull of all ``points``, along its normal (0 on the
    hull). ``normals`` must point out of the hull."""
    result = np.zeros(len(which))
    if len(points) < 4:
        return result
    bm = bmesh.new()
    try:
        for co in points.tolist():
            bm.verts.new(co)
        try:
            hull = bmesh.ops.convex_hull(bm, input=list(bm.verts))
        except (RuntimeError, ValueError):
            return result  # flat or degenerate: nothing lies below anything
        unused = {elem for elem in hull["geom_interior"] + hull["geom_unused"] if isinstance(elem, bmesh.types.BMVert)}
        if unused:
            bmesh.ops.delete(bm, geom=list(unused), context='VERTS')
        if not bm.faces:
            return result
        bvh = BVHTree.FromBMesh(bm)
    finally:
        bm.free()
    back = 1e-4
    for k, (co, normal) in enumerate(zip(points[which].tolist(), normals[which].tolist())):
        co, normal = Vector(co), Vector(normal)
        hit = bvh.ray_cast(co - normal * back, normal, 1.0)
        if hit[0] is not None and hit[1].dot(normal) < 0.0:
            # The point is on the back of the hull (the area is a hollow): go through it to the front.
            hit = bvh.ray_cast(hit[0] + normal * 1e-6, normal, 1.0)
        if hit[0] is not None:
            result[k] = max(0.0, (hit[0] - co).dot(normal))
    return result


class StraightenStroke(WeightEdit):
    """Straightens what sags around a stroke, with weights blended in from what it hangs between.

    ``rest_co`` and ``posed_co`` are the world space vertex positions in the rest pose and as displayed,
    ``body`` the `BodySurface` of the body as displayed (it tells which way is out). `apply` collects the
    vertices under the brush; `finish` works out the sagging span around them and blends its weights.
    ``group_mode`` 'ACTIVE' only changes the active group's weight (the vertex's other bone weights make
    room for it, keeping their proportions). With ``allowed`` (per vertex), the others never change: a span
    stops at them.

    What the stroke found, for the operator to report: ``unposed`` (the mesh shows its rest pose: nothing
    can sag), ``found_nothing`` (nothing around the stroke sags more than in the rest pose),
    ``unsupported`` (nothing around the sagging part has bone weights to blend from), ``span_size``
    (vertices straightened).
    """

    def __init__(self, obj, rest_co, posed_co, body, *, group_mode='ALL', allowed=None):
        super().__init__(obj)
        bones = deform_bone_names(obj)
        # Only bone weights take part: they are what moves the mesh.
        self.is_bone = np.array([group.name in bones for group in self.groups] or [False])
        self.active = obj.vertex_groups.active_index if group_mode == 'ACTIVE' else None
        self.nothing_to_do = (
            not self.is_bone.any() or
            (self.active is not None and not (0 <= self.active < len(self.groups) and self.is_bone[self.active]))
        )

        # The vertex graph of the rest shape, with vertices sharing a position (split seams) welded.
        mesh = self.mesh
        count = len(mesh.vertices)
        co = np.empty(count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        self.weld, self.weld_count = weld_groups(co.reshape(-1, 3).astype(np.float64), WELD_DISTANCE)
        members = np.bincount(self.weld, minlength=self.weld_count).astype(np.float64)[:, None]

        def welded(values):
            result = np.zeros((self.weld_count, 3))
            np.add.at(result, self.weld, np.asarray(values, dtype=np.float64))
            return result / np.maximum(members, 1.0)

        self.rest = welded(rest_co)
        self.posed = welded(posed_co)
        self.unposed = not len(self.rest) or float(np.linalg.norm(self.posed - self.rest, axis=1).max()) < MIN_MOVE
        mesh.calc_loop_triangles()
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
        tris = self.weld[tris.reshape(-1, 3)]
        tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])]
        edges, weights = cotangent_weights(self.rest, tris) if len(tris) else (np.empty((0, 2), np.int64), np.empty(0))
        both = np.concatenate((edges, edges[:, ::-1]))
        order, self.neighbour_ptr = _rows(both[:, 0], self.weld_count)
        self.neighbours = both[order, 1]
        self.edge_weights = np.concatenate((weights, weights))[order]
        self.members, self.member_ptr = _rows(self.weld, self.weld_count)
        # Welded groups that may change: all of their vertices may.
        self.allowed = None
        if allowed is not None:
            self.allowed = np.ones(self.weld_count, dtype=bool)
            np.logical_and.at(self.allowed, self.weld, np.asarray(allowed, dtype=bool))

        # Normals pointing away from the body, as displayed; the rest pose's on the same side.
        self.normals = vertex_normals(self.posed, tris) if len(tris) else np.zeros_like(self.posed)
        if body is not None and len(self.posed):
            near = body.nearest(self.posed)
            offset = self.posed - near.location
            outward = np.where((near.distance >= 0.0)[:, None], offset, -offset)
            reference = np.where((np.linalg.norm(offset, axis=1) > 1e-5)[:, None], outward, near.normal)
            flip = near.found & (np.einsum("ij,ij->i", self.normals, reference) < 0.0)
            self.normals[flip] *= -1.0
        self.rest_normals = vertex_normals(self.rest, tris) if len(tris) else np.zeros_like(self.rest)
        self.rest_normals[np.einsum("ij,ij->i", self.rest_normals, self.normals) < 0.0] *= -1.0
        face_areas = np.linalg.norm(np.cross(self.posed[tris[:, 1]] - self.posed[tris[:, 0]],
                                             self.posed[tris[:, 2]] - self.posed[tris[:, 0]]), axis=1) / 6.0
        self.vertex_area = np.zeros(self.weld_count)
        for k in range(3):
            np.add.at(self.vertex_area, tris[:, k], face_areas)

        # What the stroke painted: per welded group the largest factor, and the brush's size (m).
        self.painted = np.zeros(self.weld_count)
        self.radius = 0.0
        # What the stroke found, for the operator to report.
        self.found_nothing = False
        self.unsupported = False
        self.span_size = 0

    # -------------------------------------------------------------------------
    # Public API

    def apply(self, verts, factors, radius=None):
        """Remember ``verts`` (under a dab, each with ``factors``: strength times falloff, 0..1) for
        `finish`. ``radius`` is the dab's size in world units (else it's guessed from ``verts``)."""
        verts = np.asarray(verts, dtype=np.int64)
        factors = np.asarray(factors, dtype=np.float64)
        keep = factors > 0.0
        verts, factors = verts[keep], factors[keep]
        if not len(verts):
            return
        groups = self.weld[verts]
        np.maximum.at(self.painted, groups, factors)
        if radius is None:
            spot = self.posed[groups]
            radius = float(np.linalg.norm(spot - spot.mean(axis=0), axis=1).max())
        self.radius = max(self.radius, float(radius))

    def finish(self):
        """Straighten what sags around the painted vertices."""
        seeds = np.flatnonzero(self.painted > 0.0)
        if self.nothing_to_do or self.unposed or not len(seeds):
            return
        area = self._area(seeds)
        # How much deeper below the area's hull each vertex lies as displayed than in the rest pose.
        posed_sink = sinking(self.posed[area], self.normals[area], np.arange(len(area)))
        rest_sink = sinking(self.rest[area], self.rest_normals[area], np.arange(len(area)))
        deeper = self._shared(area, posed_sink - rest_sink)
        sags = np.zeros(self.weld_count, dtype=bool)
        sags[area[deeper > SINK]] = True
        if self.allowed is not None:
            sags &= self.allowed
        span = self._connected(seeds[sags[seeds]], sags)
        if not len(span):
            self.found_nothing = True
            return
        holds = np.zeros(self.weld_count, dtype=bool)
        holds[area[posed_sink <= HOLD]] = True
        solved = self._blend_span(span, holds)
        if solved is None:
            self.unsupported = True
            return
        bones, fractions = solved
        factor = float(self.painted.max())
        removals = {}
        for row, group in enumerate(span.tolist()):
            for vert in self.members[self.member_ptr[group]:self.member_ptr[group + 1]].tolist():
                self._read(vert)
                weights = self._blend(self.current[vert], bones, fractions[row], factor)
                if weights is not None:
                    self._write(vert, weights, removals)
        self._remove(removals)
        self.span_size = int(sum(self.member_ptr[g + 1] - self.member_ptr[g] for g in span.tolist()))

    # -------------------------------------------------------------------------
    # Internals

    def _distances(self, seeds, reach):
        """Distance along the mesh (as displayed) from ``seeds`` to every vertex, up to ``reach``."""
        distance = np.full(self.weld_count, np.inf)
        distance[seeds] = 0.0
        frontier = seeds
        while len(frontier):
            around, owner = _gather(self.neighbour_ptr, self.neighbours, frontier)
            step = np.linalg.norm(self.posed[around] - self.posed[frontier[owner]], axis=1)
            candidate = distance[frontier[owner]] + step
            best = np.full(self.weld_count, np.inf)
            np.minimum.at(best, around, candidate)
            improved = np.flatnonzero((best < distance) & (best <= reach))
            distance[improved] = best[improved]
            frontier = improved
        return distance

    def _area(self, seeds):
        """The painted vertices and the mesh around them: as much surface again as REACH^2 - 1 brush
        disks, the nearest along the mesh first."""
        distance = self._distances(seeds, MAX_REACH)
        order = np.argsort(distance)
        order = order[np.isfinite(distance[order])]
        wanted = np.pi * max(self.radius, MIN_RADIUS) ** 2 * (REACH ** 2 - 1.0)
        added = np.cumsum(np.where(distance[order] > 0.0, self.vertex_area[order], 0.0))
        return np.sort(order[:int(np.searchsorted(added, wanted)) + 1])

    def _shared(self, area, values):
        """``values`` (per vertex of ``area``) raised to the largest within LAYER at rest: layers of a strap
        with a thickness count as one."""
        kd = KDTree(len(area))
        for index, co in enumerate(self.rest[area].tolist()):
            kd.insert(co, index)
        kd.balance()
        result = values.copy()
        for index, co in enumerate(self.rest[area].tolist()):
            for _co, other, _distance in kd.find_range(co, LAYER):
                if values[other] > result[index]:
                    result[index] = values[other]
        return result

    def _connected(self, seeds, inside):
        """The vertices of ``inside`` connected to ``seeds`` through it."""
        found = np.zeros(self.weld_count, dtype=bool)
        found[seeds] = True
        frontier = seeds
        while len(frontier):
            around, _ = _gather(self.neighbour_ptr, self.neighbours, frontier)
            around = np.unique(around)
            around = around[inside[around] & ~found[around]]
            found[around] = True
            frontier = around
        return np.flatnonzero(found)

    def _fractions(self, groups, bones):
        """(len(groups), len(bones)): each group's unlocked bone weights over ``bones``, as fractions of 1
        (summed over the group's vertices)."""
        column = {bone: k for k, bone in enumerate(bones)}
        result = np.zeros((len(groups), len(bones)))
        for row, group in enumerate(groups.tolist()):
            for vert in self.members[self.member_ptr[group]:self.member_ptr[group + 1]].tolist():
                self._read(vert)
                for bone, weight in self.current[vert].items():
                    if weight > 0.0 and bone in column:
                        result[row, column[bone]] += weight
        total = result.sum(axis=1, keepdims=True)
        return np.where(total > 1e-9, result / np.maximum(total, 1e-12), 0.0)

    def _bones_around(self, groups):
        """The unlocked bone groups with weight on any vertex of ``groups``."""
        found = set()
        for group in groups.tolist():
            for vert in self.members[self.member_ptr[group]:self.member_ptr[group + 1]].tolist():
                self._read(vert)
                found.update(bone for bone, weight in self.current[vert].items()
                             if weight > 0.0 and self.is_bone[bone] and bone not in self.locked)
        return sorted(found)

    def _blend_span(self, span, holds):
        """Blended bone weights for ``span`` from the vertices around it: (bone groups, (n, bones) fractions),
        or None when nothing around it has bone weights. ``holds`` marks the vertices holding it out."""
        n = len(span)
        local = np.full(self.weld_count, -1, dtype=np.int64)
        local[span] = np.arange(n)
        neighbours, owner = _gather(self.neighbour_ptr, self.neighbours, span)
        weights = _gather(self.neighbour_ptr, self.edge_weights, span)[0]
        outer = local[neighbours] < 0
        rim = np.unique(neighbours[outer])
        bones = self._bones_around(rim)
        if not bones:
            return None
        values = np.zeros((self.weld_count, len(bones)))
        values[rim] = self._fractions(rim, bones)
        known = values.sum(axis=1) > 0.0
        if not known[rim].any():
            return None

        def harmonic(given, fixed, guess):
            """Harmonic interpolation over the span of ``given`` (per vertex, columns) at the vertices around
            it where ``fixed``; the others are left out, as if the mesh ended there."""
            use = ~outer | fixed[neighbours]
            nb, own, w, out = neighbours[use], owner[use], weights[use], outer[use]
            diagonal = np.bincount(own, weights=w, minlength=n)
            diagonal[diagonal <= 0.0] = 1.0
            rhs = np.zeros((n, given.shape[1]))
            np.add.at(rhs, own[out], w[out, None] * given[nb[out]])
            first, second, inner = own[~out], local[nb[~out]], w[~out]

            def multiply(x):
                result = diagonal[:, None] * x
                for k in range(x.shape[1]):
                    result[:, k] -= np.bincount(first, weights=inner * x[second, k], minlength=n)
                return result
            return _conjugate_gradient(multiply, rhs, guess, 1.0 / diagonal)

        start = self._fractions(span, bones)
        fractions = harmonic(values, known, start)
        # Carried by what holds the span out. Each connected piece of the span needs something holding
        # it, or it keeps the blend from everything around it.
        holding = known & holds
        pieces = self._pieces(span)
        for piece in np.unique(pieces):
            members = span[pieces == piece]
            around, _ = _gather(self.neighbour_ptr, self.neighbours, members)
            around = np.unique(around[local[around] < 0])
            around = around[known[around]]
            if not holding[around].any():
                holding[around] = True
        if (holding[rim] != known[rim]).any():
            held = harmonic(values, holding, fractions)
            ones = np.zeros((self.weld_count, 1))
            ones[rim[holding[rim]], 0] = 1.0
            share = np.clip(harmonic(ones, known, np.full((n, 1), 0.5)), 0.0, 1.0)
            fractions = fractions + share * (held - fractions)
        fractions = np.clip(fractions, 0.0, None)
        fractions /= np.maximum(fractions.sum(axis=1, keepdims=True), 1e-12)
        return bones, fractions

    def _pieces(self, span):
        """For each vertex of ``span``, which connected piece of it (through its own edges) it's in."""
        inside = np.zeros(self.weld_count, dtype=bool)
        inside[span] = True
        piece = np.full(self.weld_count, -1, dtype=np.int64)
        count = 0
        for seed in span.tolist():
            if piece[seed] >= 0:
                continue
            members = self._connected(np.array([seed]), inside)
            piece[members] = count
            count += 1
        return piece[span]

    def _blend(self, current, bones, target, factor):
        """New weights ({group: weight}) for a vertex with ``current`` weights, ``factor`` of the way to
        the fractions ``target`` of its unlocked bone weights (over ``bones``), or None."""
        bone_weights = {group: weight for group, weight in current.items() if weight > 0.0 and self.is_bone[group]}
        total = sum(bone_weights.values())
        locked = sum(weight for group, weight in bone_weights.items() if group in self.locked)
        free = {group: weight for group, weight in bone_weights.items() if group not in self.locked}
        if total <= 1e-9:
            total, free = 1.0, {}  # no bone weights yet: they come from the blend
        budget = total - locked
        if budget <= 1e-9:
            return None
        free_total = sum(free.values())
        shares = {group: weight / free_total for group, weight in free.items()} if free_total > 1e-9 else {}
        goal = dict(zip(bones, target.tolist()))
        if self.active is not None:
            # Only the active group moves; the others keep their proportions around it.
            active = self.active
            old = shares.get(active, 0.0)
            new = old + (goal.get(active, 0.0) - old) * factor
            others = {group: share for group, share in shares.items() if group != active}
            if not others:
                others = {group: share for group, share in goal.items() if group != active and share > 0.0}
            rest_total = sum(others.values())
            if rest_total <= 1e-9:
                return None
            mixed = {group: share / rest_total * (1.0 - new) for group, share in others.items()}
            mixed[active] = new
        else:
            mixed = {}
            for group in shares.keys() | goal.keys():
                old = shares.get(group, 0.0)
                mixed[group] = old + (goal.get(group, 0.0) - old) * factor
        # Weights too small to keep are dropped, and the others make up for them, so the total of the
        # bone weights stays what it was. A single weight can't go past 1 though.
        kept = {group: share for group, share in mixed.items() if share * budget > ZERO_WEIGHT}
        norm = sum(kept.values())
        if norm <= 1e-12:
            return None
        weights = {group: 0.0 for group in mixed}
        weights.update((group, min(share / norm * budget, 1.0)) for group, share in kept.items())
        return weights
