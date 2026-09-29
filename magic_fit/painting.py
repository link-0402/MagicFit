# SPDX-License-Identifier: GPL-3.0-or-later

"""Weight sampling and blending.

Nothing here depends on the 3D view, so a stroke can be driven (and tested) without a UI:
the operator turns mouse input into vertex indices plus blend factors and feeds them to
`CopyStroke.apply`.
"""

import math

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc
from mathutils.kdtree import KDTree

# A group whose weight fades to this or below is removed from the vertex.
ZERO_WEIGHT = 1e-4
# Weight changes smaller than this are not written back.
MIN_CHANGE = 1e-6
# Auto Smooth evens out the target's weights in passes over the painted mesh, each moving them up to
# SMOOTH_RATE of the way to their neighbours' average: SMOOTH_PASSES of them at full intensity. Weights
# spread with the square root of the passes, so the passes grow with the square of the intensity.
SMOOTH_PASSES = 12
SMOOTH_RATE = 0.5


def falloff_weights(kind, t):
    """Brush falloff for normalized distances ``t`` (0 = center, 1 = edge).

    Same curves as Blender's brush falloff presets.
    """
    p = np.clip(1.0 - t, 0.0, 1.0)
    if kind == 'SMOOTH':
        return p * p * (3.0 - 2.0 * p)
    if kind == 'SPHERE':
        return np.sqrt(p * (2.0 - p))
    if kind == 'ROOT':
        return np.sqrt(p)
    if kind == 'SHARP':
        return p * p
    if kind == 'LINEAR':
        return p
    return np.ones_like(p)


# -----------------------------------------------------------------------------
# Geometry

# Targets whose modifiers change the topology are sampled as displayed while that at most roughly
# doubles the vertex count (Mirror, Solidify, Weld...). Denser results such as Subdivision Surface
# use the base mesh instead: nearly the same surface, and much faster to set up for every stroke.
MAX_EVALUATED_TARGET_RATIO = 2.5


class Geometry:
    """World space snapshot of a mesh object's current shape, see `read_geometry`.

    ``surface_co`` and ``surface_tris`` always describe the surface drawn in the viewport.
    ``weights`` holds per vertex ``(group, weight)`` pairs when they had to be read from the
    evaluated mesh (None means: they are those of the original mesh). ``evaluated`` tells whether
    ``co`` comes from the evaluated mesh (posed, shape keys...) rather than the original one.
    """

    __slots__ = ("matrix", "co", "normals", "tris", "surface_co", "surface_tris", "weights", "evaluated")


def _read_co(mesh, attr="co"):
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get(attr, co)
    return co.reshape(-1, 3).astype(np.float64)


def _read_tris(mesh):
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris)
    return tris.reshape(-1, 3)


def transform_points(co, matrix):
    m = np.array(matrix, dtype=np.float64)
    return co @ m[:3, :3].T + m[:3, 3]


def _normals_to_world(normals, matrix):
    m = np.array(matrix.to_3x3().inverted_safe().transposed(), dtype=np.float64)
    normals = normals @ m.T
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def read_geometry(obj, depsgraph, *, source='INDEXED', normals=False, surface=False):
    """Read ``co``, ``normals`` and ``tris`` of ``obj``. Which mesh they come from depends on ``source``:

    'INDEXED': indexed like the original mesh, so they line up with its vertex groups. The evaluated
        mesh is used while modifiers only deform it (Armature, shape keys...), else the original.
    'WEIGHTED': like 'INDEXED', but when modifiers change the topology without making the mesh much
        denser, the evaluated mesh is used along with its own ``weights``.
    'DISPLAYED': the evaluated mesh, whatever its topology.
    """
    mesh = obj.data
    obj_eval = obj.evaluated_get(depsgraph)
    geometry = Geometry()
    geometry.matrix = obj_eval.matrix_world.copy()
    geometry.weights = None
    try:
        mesh_eval = obj_eval.to_mesh()
    except RuntimeError:
        mesh_eval = None
    try:
        evaluated_count = len(mesh_eval.vertices) if mesh_eval is not None else 0
        deformed_only = mesh_eval is not None and evaluated_count == len(mesh.vertices)
        source_mesh = mesh_eval if deformed_only or (source == 'DISPLAYED' and evaluated_count) else mesh
        if (
                source == 'WEIGHTED' and not deformed_only and
                0 < evaluated_count <= MAX_EVALUATED_TARGET_RATIO * len(mesh.vertices)
        ):
            # Make sure the vertex groups survive the evaluation.
            obj_eval.to_mesh_clear()
            mesh_eval = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            source_mesh = mesh_eval
            geometry.weights = [
                tuple((elem.group, elem.weight) for elem in vert.groups) for vert in mesh_eval.vertices
            ]
        geometry.evaluated = source_mesh is mesh_eval
        geometry.co = transform_points(_read_co(source_mesh), geometry.matrix)
        geometry.tris = _read_tris(source_mesh)
        geometry.normals = (
            _normals_to_world(_read_co(source_mesh, "normal"), geometry.matrix) if normals else None
        )
        if surface and evaluated_count and source_mesh is not mesh_eval:
            geometry.surface_co = transform_points(_read_co(mesh_eval), geometry.matrix)
            geometry.surface_tris = _read_tris(mesh_eval)
        else:
            geometry.surface_co = geometry.co
            geometry.surface_tris = geometry.tris
    finally:
        if mesh_eval is not None:
            obj_eval.to_mesh_clear()
    return geometry


# Search trees are slow to build from Python, so keep the last few around and reuse them while
# the geometry they were built from is unchanged (usually for every stroke of a session).
_tree_cache = {}
_TREE_CACHE_SIZE = 4


def _cached_tree(key, co, tris, build):
    entry = _tree_cache.get(key)
    if entry is not None and np.array_equal(entry[0], co) and np.array_equal(entry[1], tris):
        return entry[2]
    tree = build()
    _tree_cache.pop(key, None)
    while len(_tree_cache) >= _TREE_CACHE_SIZE:
        del _tree_cache[next(iter(_tree_cache))]
    _tree_cache[key] = (co, tris, tree)
    return tree


def surface_bvh(key, co, tris, tri_list=None):
    """BVH tree over world space triangles, for ray casts and nearest-point queries. ``tri_list`` can
    hold ``tris`` as a list of index triples already, which saves most of the time to build it."""
    def build():
        return BVHTree.FromPolygons(co.tolist(), tris.tolist() if tri_list is None else tri_list, all_triangles=True)
    return _cached_tree(("bvh", key), co, tris, build)


def vertex_kdtree(key, co):
    def build():
        tree = KDTree(len(co))
        for index, position in enumerate(co.tolist()):
            tree.insert(position, index)
        tree.balance()
        return tree
    return _cached_tree(("kd", key), co, None, build)


def clear_cache():
    _tree_cache.clear()


# -----------------------------------------------------------------------------
# Target sampling

class TargetSampler:
    """Reads the target's weights at the point of its surface closest to a world space position."""

    def __init__(self, target, depsgraph, *, mode='SURFACE', max_distance=None):
        geometry = read_geometry(target, depsgraph, source='WEIGHTED')
        key = target.as_pointer()
        self._mesh = target.data
        # Weights are read from the original mesh on demand, unless the evaluated mesh brought its own.
        self._weights = {} if geometry.weights is None else dict(enumerate(geometry.weights))
        self._co = geometry.co
        self._tris = geometry.tris.tolist()
        # A mesh without faces has no surface to project onto.
        self.mode = mode if self._tris else 'VERTEX'
        if self.mode == 'SURFACE':
            self._bvh = surface_bvh(key, geometry.co, geometry.tris)
        else:
            self._kd = vertex_kdtree(key, geometry.co)
        self._max_distance = max_distance

    def _vertex_weights(self, index):
        weights = self._weights.get(index)
        if weights is None:
            weights = tuple((elem.group, elem.weight) for elem in self._mesh.vertices[index].groups)
            self._weights[index] = weights
        return weights

    def sample(self, point):
        """Return ``{target group index: weight}`` at the target point closest to ``point``.

        Returns None when the target is farther away than the max distance.
        """
        max_distance = self._max_distance
        if self.mode == 'VERTEX':
            _co, index, distance = self._kd.find(point)
            if index is None or (max_distance is not None and distance > max_distance):
                return None
            return dict(self._vertex_weights(index))

        if max_distance is None:
            location, _normal, tri_index, _distance = self._bvh.find_nearest(point)
        else:
            location, _normal, tri_index, _distance = self._bvh.find_nearest(point, max_distance)
        if location is None:
            return None
        tri = self._tris[tri_index]
        co = self._co
        factors = poly_3d_calc([Vector(co[tri[0]]), Vector(co[tri[1]]), Vector(co[tri[2]])], location)
        result = {}
        for vert, factor in zip(tri, factors):
            if factor <= 0.0:
                continue
            for group, weight in self._vertex_weights(vert):
                result[group] = result.get(group, 0.0) + weight * factor
        return result


# -----------------------------------------------------------------------------
# Painting

def deform_bone_names(obj):
    """Names of the bones that deform ``obj``, the groups Auto Normalize applies to."""
    armatures = [
        mod.object for mod in obj.modifiers
        if mod.type == 'ARMATURE' and mod.object is not None and mod.object.type == 'ARMATURE'
    ]
    parent = obj.parent
    if parent is not None and parent.type == 'ARMATURE' and obj.parent_type == 'ARMATURE':
        armatures.append(parent)
    return {bone.name for arm in armatures for bone in arm.data.bones if bone.use_deform}


class WeightEdit:
    """Reads and writes the vertex group weights of ``obj`` for a stroke, and can undo them all.

    ``current`` and ``start`` hold ``{group: weight}`` per vertex the stroke has read.
    """

    def __init__(self, obj):
        self.obj = obj
        self.mesh = obj.data
        vertex_groups = obj.vertex_groups
        self.groups = list(vertex_groups)
        self.initial_active_index = vertex_groups.active_index
        self.locked = {group.index for group in self.groups if group.lock_weight}
        self.start = {}      # {group: weight} when the stroke began
        self.current = {}    # {group: weight} as written so far
        self.created = []    # groups this stroke added to the object
        self.changed = False   # weights written since the caller last reset this
        self.modified = False  # weights written at all

    def restore(self):
        """Undo everything this stroke did (used when the stroke is cancelled)."""
        removals = {}
        for vert, start in self.start.items():
            current = self.current[vert]
            for group in current:
                if group not in start:
                    removals.setdefault(group, []).append(vert)
            for group, weight in start.items():
                if current.get(group) != weight:
                    self.groups[group].add((vert,), weight, 'REPLACE')
            self.current[vert] = dict(start)
        for group, group_verts in removals.items():
            self.groups[group].remove(group_verts)

        # Created groups are the last ones, so removing them keeps every other index valid.
        vertex_groups = self.obj.vertex_groups
        for group in reversed(self.created):
            vertex_groups.remove(self.groups.pop(group))
            self._forget_group(group)
        self.created.clear()
        self._restore_active_group()
        self.changed = True
        self.modified = False

    def created_group_names(self):
        return [self.groups[group].name for group in self.created]

    def _forget_group(self, group):
        """A created group was removed again."""

    def _read(self, vert):
        if vert not in self.current:
            weights = {elem.group: elem.weight for elem in self.mesh.vertices[vert].groups}
            self.current[vert] = weights
            self.start[vert] = dict(weights)

    def _write(self, vert, weights, removals):
        """Store ``weights`` ({group: weight}) for ``vert``. Groups to take the vertex out of are
        collected in ``removals``, for the caller to remove in bulk."""
        current = self.current[vert]
        for group, weight in weights.items():
            weight = min(max(weight, 0.0), 1.0)
            old = current.get(group)
            if weight <= ZERO_WEIGHT:
                # Only drop memberships this brush faded out; leave existing zero weights alone.
                if old is not None and old > ZERO_WEIGHT:
                    removals.setdefault(group, []).append(vert)
                    del current[group]
                    self.changed = self.modified = True
                continue
            if old is None or abs(weight - old) > MIN_CHANGE:
                self.groups[group].add((vert,), weight, 'REPLACE')
                current[group] = weight
                self.changed = self.modified = True

    def _remove(self, removals):
        for group, group_verts in removals.items():
            self.groups[group].remove(group_verts)

    def _restore_active_group(self):
        vertex_groups = self.obj.vertex_groups
        if 0 <= self.initial_active_index < len(vertex_groups):
            vertex_groups.active_index = self.initial_active_index


def _gather(rows, keys):
    """For compressed rows (pointers, values: ``values[pointers[k]:pointers[k + 1]]`` is row k): the
    values of rows ``keys``, and for each the position in ``keys`` of its row."""
    pointers, values = rows
    starts = pointers[keys]
    lengths = pointers[keys + 1] - starts
    owner = np.repeat(np.arange(len(keys)), lengths)
    offsets = np.arange(len(owner)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return values[starts[owner] + offsets], owner


def _compressed_rows(keys, values, count):
    """Compressed rows (pointers, values) that hold, in row k, the ``values`` whose ``keys`` are k."""
    order = np.argsort(keys, kind='stable')
    pointers = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=count), out=pointers[1:])
    return pointers, values[order]


class CopyStroke(WeightEdit):
    """Blends a mesh's weights toward the weights sampled from a target, dab by dab.

    ``group`` values index the painted object's vertex groups, ``target_group`` values the target's.

    Auto Smooth (``smooth`` 0..1) blends toward the target's weights evened out over the painted mesh
    instead, so copied weights don't change abruptly where the closest point on the target jumps (across
    creases) or the target's weights have sharp edges. Each vertex gets its sample averaged with the
    samples around it, in passes over the mesh's edges; vertices at the same spot (split seams) count as
    one. The result doesn't depend on the brush: a vertex gets the same weights however it's painted.
    """

    def __init__(
            self, obj, target, sampler, positions, *,
            group_mode='ALL',
            fade_unmatched=True,
            accumulate=False,
            auto_normalize=False,
            restrict=False,
            bones_only=True,
            smooth=0.0,
    ):
        super().__init__(obj)
        self.sampler = sampler
        # World space vertex positions, indexed like `obj.data.vertices`.
        self.positions = positions
        self.fade_unmatched = fade_unmatched
        self.accumulate = accumulate
        self.auto_normalize = auto_normalize
        self.restrict = restrict
        self.smooth = smooth
        self._graph = None          # see `_vertex_graph`
        self._node_samples = {}     # see `_node_sample`

        vertex_groups = obj.vertex_groups
        by_name = {group.name: group.index for group in self.groups}
        self.target_names = [group.name for group in target.vertex_groups]
        # Target group -> painted group with the same name (None until it is created).
        self.target_to_group = {t: by_name.get(name) for t, name in enumerate(self.target_names)}
        self.group_to_target = {g: t for t, g in self.target_to_group.items() if g is not None}
        self.deform_bones = deform_bone_names(obj)
        self.deform = {group.index for group in self.groups if group.name in self.deform_bones}

        # All Groups mode, with ``bones_only``: only the groups of deform bones take part, like Blender's
        # Data Transfer with Deform Pose Bones. Helper groups (masks for normals, shape keys, physics...)
        # are neither copied nor faded. The target's armature decides which groups those are; without
        # one, the painted mesh's; without either, every group takes part.
        bones = (deform_bone_names(target) or self.deform_bones) if bones_only else set()
        self.copied = {t for t, name in enumerate(self.target_names) if not bones or name in bones}
        # Painted groups that are copied into or faded, or None for all of them.
        self.managed = {group.index for group in self.groups if group.name in bones} if bones else None

        if group_mode == 'ACTIVE':
            self.active = vertex_groups.active_index
            self.active_target = self.group_to_target.get(self.active)
        else:
            self.active = None
            self.active_target = None

        # Per vertex, for the vertices this stroke touched:
        self.coverage = {}   # strongest blend factor applied (without Accumulate)
        self.samples = {}    # {target_group: weight}, or None when the target is out of range
        self.goals = {}      # the samples evened out by Auto Smooth, likewise

    # -------------------------------------------------------------------------
    # Public API

    def apply(self, verts, factors):
        """Blend ``verts`` toward the target by ``factors`` (strength times falloff, 0..1)."""
        removals = {}
        accumulate = self.accumulate
        coverage = self.coverage
        smooth = self.smooth > 0.0
        if smooth:
            self._even_out([vert for vert, factor in zip(verts, factors) if factor > 0.0])
        for vert, factor in zip(verts, factors):
            if factor <= 0.0:
                continue
            if not accumulate:
                # Like Blender's own brushes: within a stroke a vertex moves at most as far as the
                # strongest dab that touched it, no matter how often the brush passes over it.
                if factor <= coverage.get(vert, 0.0) + MIN_CHANGE:
                    continue
                coverage[vert] = factor
            sample = self.goals[vert] if smooth else self._sample(vert)
            if sample is None:
                continue
            self._read(vert)
            base = self.current[vert] if accumulate else self.start[vert]
            weights = self._blend(base, sample, factor)
            if weights is not None:
                self._write(vert, weights, removals)
        self._remove(removals)

    def restore(self):
        super().restore()
        self.coverage.clear()

    # -------------------------------------------------------------------------
    # Internals

    def _forget_group(self, group):
        self.target_to_group[self.group_to_target.pop(group)] = None
        self.deform.discard(group)
        if self.managed is not None:
            self.managed.discard(group)

    def _sample(self, vert):
        try:
            return self.samples[vert]
        except KeyError:
            sample = self.sampler.sample(Vector(self.positions[vert]))
            self.samples[vert] = sample
            return sample

    def _blend(self, base, sample, alpha):
        weights = dict(base)
        if self.active is not None:
            group = self.active
            if self.restrict and group not in base:
                return None
            old = base.get(group, 0.0)
            weights[group] = old + (sample.get(self.active_target, 0.0) - old) * alpha
            if self.auto_normalize and group in self.deform:
                self._normalize(weights, fixed=group)
            return weights

        locked = self.locked
        target_to_group = self.target_to_group
        copied = self.copied
        for target_group, target_weight in sample.items():
            if target_group not in copied:
                continue
            group = target_to_group.get(target_group)
            if group is None:
                weight = target_weight * alpha
                if weight > ZERO_WEIGHT:
                    weights[self._create_group(target_group)] = weight
            elif group not in locked:
                old = base.get(group, 0.0)
                weights[group] = old + (target_weight - old) * alpha

        group_to_target = self.group_to_target
        managed = self.managed
        for group, old in base.items():
            if group in locked or (managed is not None and group not in managed):
                continue
            target_group = group_to_target.get(group)
            if target_group is None:
                # The target has no such group at all.
                if self.fade_unmatched:
                    weights[group] = old * (1.0 - alpha)
            elif target_group not in sample:
                # The target has the group, but not at this point.
                weights[group] = old * (1.0 - alpha)

        if self.auto_normalize:
            self._normalize(weights)
        return weights

    def _normalize(self, weights, fixed=None):
        """Make the deform groups add up to 1 (Auto Normalize), keeping locked groups as they are.

        ``fixed`` is the group being painted in Current Group mode: the other groups make room for it.
        """
        deform = self.deform
        locked = self.locked
        locked_sum = 0.0
        free = []
        for group, weight in weights.items():
            if group not in deform or group == fixed:
                continue
            if group in locked:
                locked_sum += weight
            else:
                free.append(group)
        available = max(0.0, 1.0 - locked_sum)
        if fixed is not None:
            weights[fixed] = min(weights[fixed], available)
            available -= weights[fixed]
        free_sum = sum(weights[group] for group in free)
        if free_sum <= 1e-9:
            return
        scale = available / free_sum
        for group in free:
            weights[group] *= scale

    def _create_group(self, target_group):
        name = self.target_names[target_group]
        group = self.obj.vertex_groups.new(name=name)
        # `new()` makes the new group active, which would switch the weights shown in the viewport.
        self._restore_active_group()
        index = group.index
        self.groups.append(group)
        self.target_to_group[target_group] = index
        self.group_to_target[index] = target_group
        if name in self.deform_bones:
            self.deform.add(index)
        if self.managed is not None:
            self.managed.add(index)
        self.created.append(index)
        return index

    def _vertex_graph(self):
        """The painted mesh's vertex graph for Auto Smooth, with vertices at the same spot (split seams)
        welded into one node: (vertex -> node, the vertices of each node, the neighbour nodes of each
        node), the latter two as compressed rows (see `_gather`)."""
        if self._graph is None:
            from .fitting import WELD_DISTANCE, weld_groups  # fitting imports this module
            mesh = self.mesh
            co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", co)
            weld, count = weld_groups(co.reshape(-1, 3).astype(np.float64), WELD_DISTANCE)
            edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
            mesh.edges.foreach_get("vertices", edges)
            edges = weld[edges.reshape(-1, 2)]
            edges = edges[edges[:, 0] != edges[:, 1]]
            links = np.unique(np.concatenate((edges[:, 0] * count + edges[:, 1], edges[:, 1] * count + edges[:, 0])))
            self._graph = (
                weld,
                _compressed_rows(weld, np.arange(len(weld)), count),
                _compressed_rows(links // count, links % count, count),
            )
        return self._graph

    def _node_sample(self, node, verts):
        """The target's weights at the welded vertex ``node`` made of ``verts``: their samples averaged,
        or None when they're all out of the target's reach."""
        try:
            return self._node_samples[node]
        except KeyError:
            pass
        samples = [sample for sample in map(self._sample, verts) if sample is not None]
        if len(samples) > 1:
            average = {}
            for sample in samples:
                for group, weight in sample.items():
                    average[group] = average.get(group, 0.0) + weight / len(samples)
            samples = [average]
        result = samples[0] if samples else None
        self._node_samples[node] = result
        return result

    def _even_out(self, verts):
        """Work out Auto Smooth's goals for those of ``verts`` that have none yet: their samples evened
        out over the painted mesh, in passes that each move every sample up to ``SMOOTH_RATE`` of the way
        to the average of its neighbours'. Vertices out of the target's reach don't take part.

        After n passes, a vertex's goal depends on the samples up to n edges away, so those are sampled
        as well. That way it doesn't depend on which vertices the brush covers when it gets there."""
        new = [vert for vert in verts if vert not in self.goals]
        if not new:
            return
        weld, (member_pointers, members), neighbours = self._vertex_graph()
        # As many passes as the intensity needs, SMOOTH_PASSES of SMOOTH_RATE at full intensity.
        amount = SMOOTH_PASSES * self.smooth * self.smooth
        passes = max(1, math.ceil(amount))
        rate = SMOOTH_RATE * amount / passes

        def sample(node):
            return self._node_sample(node, members[member_pointers[node]:member_pointers[node + 1]].tolist())

        def reachable(nodes):
            return np.array([node for node in nodes.tolist() if sample(node) is not None], dtype=np.int64)

        # The nodes within `passes` edges of the new ones, through nodes in the target's reach.
        seen = np.zeros(len(member_pointers) - 1, dtype=bool)
        frontier = np.unique(weld[np.array(new, dtype=np.int64)])
        seen[frontier] = True
        frontier = reachable(frontier)
        region = [frontier]
        for _ in range(passes):
            around = np.unique(_gather(neighbours, frontier)[0])
            around = around[~seen[around]]
            seen[around] = True
            frontier = reachable(around)
            region.append(frontier)
        region = np.concatenate(region)
        local = np.full(len(seen), -1, dtype=np.int64)
        local[region] = np.arange(len(region))

        # Their samples, one column per target group the copy uses.
        wanted = self.copied if self.active is None else {self.active_target}
        columns = {}
        rows, cols, values = [], [], []
        for row, node in enumerate(region.tolist()):
            for group, weight in sample(node).items():
                if group in wanted:
                    rows.append(row)
                    cols.append(columns.setdefault(group, len(columns)))
                    values.append(weight)
        value = np.zeros((len(region), len(columns)))
        np.add.at(value, (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)), values)

        around, owner = _gather(neighbours, region)
        inside = local[around] >= 0
        owner, around = owner[inside], local[around[inside]]
        if len(owner):
            starts = np.flatnonzero(np.diff(owner, prepend=-1))
            moving = owner[starts]
            degree = np.diff(starts, append=len(owner))[:, None]
            for _ in range(passes):
                average = np.add.reduceat(value[around], starts, axis=0) / degree
                value[moving] += rate * (average - value[moving])

        groups = list(columns)
        for vert in new:
            row = local[weld[vert]]
            self.goals[vert] = None if row < 0 else {
                group: weight for group, weight in zip(groups, value[row].tolist()) if weight > 0.0
            }
