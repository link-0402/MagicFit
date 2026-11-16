# SPDX-License-Identifier: GPL-3.0-or-later

"""Line Up for Second Life pieces: Firestorm's .dae exports, without rigging.

Firestorm exports each prim's mesh filling the -0.5..0.5 cube; the Collada importer gives the object the prim's
scale and a location that every piece of the export shares. So each piece has its true size and turn, but not
its place on the body. Line Up loads the Second Life body the clothes were made for (an eBody devkit, set in the
add-on's preferences), puts each piece on it by moving it only, and then lines that body up with the FFXIV
body, the pieces along with it (see `line_up`).
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import lineup

# How exactly a prim's mesh fills the -0.5..0.5 cube.
PRIM_TOLERANCE = 1e-4
# The root prim of an export: smaller than this (m) and with fewer vertices, away from the other pieces.
ROOT_SIZE = 0.10
ROOT_VERTICES = 100
SAME_SPOT = 0.001

# Placing a piece (`place`), measured on 11 Firestorm exports and an eBody devkit (2026-11-16): the piece's
# centre tried on a grid of this step (m) over the body, scored by this many of its points, the best
# CANDIDATES (each this far from the others) then moved onto the skin by point-to-plane steps with more.
GRID_STEP = 0.06
COARSE_SAMPLES = 50
FINE_SAMPLES = 300
CANDIDATES = 8
STEPS = 30
# Cloth lies this far out of the skin; the cost counts the TRIM best fitting points (flares and sleeve ends
# stand off), each gap at most CAP, a gap inside the body INSIDE_WEIGHT times.
SKIN_GAP = 0.003
TRIM = 0.6
CAP = 0.05
INSIDE_WEIGHT = 2.0
# A piece is placed when its median gap is at most MAX_GAP and at most MAX_INSIDE of its points are over 5 mm
# inside the body. From 11 Firestorm exports on the eBody (2026-11-16): loose cardigans and a hood 22-28 mm,
# layered skirts' under layers up to 33 % inside; demo signs 206-507 mm, a hidden cylinder 32 mm, heeled
# boots on flat feet 38-40 % and hair (the eBody has no head) 36-42 % inside. How much better the best place
# scored than the next did not tell them apart: layers and straps slide along legs and hips (1.0-1.9 times).
MAX_GAP = 0.030
MAX_INSIDE = 0.35
INSIDE = 0.005
# Nor may it reach farther than this (m) past the body's bounds: the Lumi exports' 2.45 m helper hugs the legs
# with rings but reaches 1.2-1.4 m past the feet. A hood rises 28-29 cm over the eBody, which has no head.
MAX_OUTSIDE = 0.35

# The body in the Second Life Body file: a mesh this tall (m) bent by a rig with mPelvis; its feet: meshes
# of that rig smaller than FEET_SIZE whose bounds come within FEET_REACH of its lowest point.
BODY_HEIGHT = (1.0, 2.5)
FEET_SIZE = 0.5  # the eBody's flat feet: 0.27 m; its unplaced nails and raised feet: 0.7-1.0 m
FEET_REACH = 0.05


class SecondLifeError(Exception):
    pass


def is_prim(obj):
    """Whether ``obj`` is a piece of a Firestorm export: a mesh without vertex groups or Armature modifier,
    filling the -0.5..0.5 cube."""
    if obj.type != 'MESH' or obj.vertex_groups or any(m.type == 'ARMATURE' for m in obj.modifiers):
        return False
    return _fills_unit_cube(obj)


def _fills_unit_cube(obj):
    """Whether the mesh of ``obj`` fills the -0.5..0.5 cube, as Second Life stores every piece."""
    if not len(obj.data.vertices):
        return False
    corners = np.array([tuple(corner) for corner in obj.bound_box])
    if not obj.modifiers:
        # Without modifiers the bounds are the mesh's own: enough, and Line Up's panel asks on every redraw
        # (reading 15 pieces of 20,000 vertices took 46 ms).
        return bool(np.abs(np.abs(corners) - 0.5).max() <= PRIM_TOLERANCE)
    if np.abs(np.abs(corners) - 0.5).max() > 0.01:
        return False
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return bool(np.abs(co.min(axis=0) + 0.5).max() <= PRIM_TOLERANCE and
                np.abs(co.max(axis=0) - 0.5).max() <= PRIM_TOLERANCE)


def _small(obj):
    return max(obj.dimensions) < ROOT_SIZE and len(obj.data.vertices) < ROOT_VERTICES


def _is_root(obj, prims):
    """Whether ``obj`` is an export's root prim: small, away from every larger piece (each export's root sits
    at the origin, so two imports' roots share a spot)."""
    if not _small(obj):
        return False
    here = obj.matrix_world.translation
    return all((other.matrix_world.translation - here).length > SAME_SPOT for other in prims
               if other != obj and not _small(other))


def import_state(meshes):
    """'NONE' when no mesh is a prim, 'IMPORT' when all are and some aren't root prims, else 'MIXED'."""
    prims = [obj for obj in meshes if is_prim(obj)]
    if not prims:
        return 'NONE'
    if len(prims) == len(meshes) and any(not _is_root(obj, prims) for obj in prims):
        return 'IMPORT'
    return 'MIXED'


def split_import(meshes):
    """(pieces, root prims) of an import's prims."""
    roots = [obj for obj in meshes if _is_root(obj, meshes)]
    return [obj for obj in meshes if obj not in roots], roots


def bake(obj):
    """Move the object's matrix into its mesh, leaving the object at the identity."""
    obj.data.transform(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)


# -----------------------------------------------------------------------------
# The Second Life body

def body_setting():
    """The Second Life Body file set in the add-on's preferences ('' when none, or when the add-on isn't enabled
    as such: in the tests)."""
    addon = bpy.context.preferences.addons.get(__package__)
    path = getattr(getattr(addon, "preferences", None), "second_life_body", "") if addon is not None else ""
    return bpy.path.abspath(path) if path else ""


@dataclass
class SecondLifeBody:
    body: object
    feet: list = field(default_factory=list)
    armature: object = None


def _world_points(obj):
    """World positions of the vertices of mesh ``obj``, before modifiers (bound_box isn't ready right after
    an append)."""
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    matrix = np.array(obj.matrix_world)
    return co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


def _world_box(obj):
    points = _world_points(obj)
    return points.min(axis=0), points.max(axis=0)


def _rig_of(obj):
    return next((m.object for m in obj.modifiers if m.type == 'ARMATURE' and m.object is not None), None)


def pick_body(objects, name="the file"):
    """The Second Life body among ``objects``: of the meshes bent by a rig with mPelvis, as tall as a person and
    not filling the unit cube, the one with the most vertices on the rig that bends the most meshes; and its
    feet (see FEET_SIZE). Raises SecondLifeError without one."""
    meshes = [obj for obj in objects if obj is not None and obj.type == 'MESH' and len(obj.data.vertices) and
              not _fills_unit_cube(obj)]
    candidates = []
    for obj in meshes:
        rig = _rig_of(obj)
        if rig is None or rig.type != 'ARMATURE' or "mPelvis" not in rig.data.bones:
            continue
        low, high = _world_box(obj)
        if BODY_HEIGHT[0] <= high[2] - low[2] <= BODY_HEIGHT[1]:
            candidates.append(obj)
    if not candidates:
        raise SecondLifeError("No Second Life body in '{:s}': it needs a mesh bent by an armature with "
                              "mPelvis".format(name))
    # A devkit's main rig bends the most meshes (body, feet, nails...); a stand-in for cloth simulation has a
    # rig of its own.
    bent = {}
    for obj in objects:
        if obj is not None and obj.type == 'MESH' and _rig_of(obj) is not None:
            bent[_rig_of(obj)] = bent.get(_rig_of(obj), 0) + 1
    rig = max({_rig_of(obj) for obj in candidates}, key=lambda armature: bent[armature])
    body = max((obj for obj in candidates if _rig_of(obj) == rig), key=lambda obj: len(obj.data.vertices))
    world = _world_points(body)
    lowest = world[np.argmin(world[:, 2])]
    feet = []
    for obj in meshes:
        if obj == body or _rig_of(obj) != rig:
            continue
        low, high = _world_box(obj)
        if (high - low).max() < FEET_SIZE and np.linalg.norm(lowest - np.clip(lowest, low, high)) <= FEET_REACH:
            feet.append(obj)
    return SecondLifeBody(body, feet, rig)


@contextmanager
def loaded_body(path):
    """The Second Life body of the .blend ``path`` (a SecondLifeBody), appended for the time of the ``with``
    block at its rest pose, hidden, and removed again after, with everything else the file brought."""
    if not path or not os.path.isfile(path):
        raise SecondLifeError("The Second Life Body file '{:s}' doesn't exist".format(path))
    before = {item.as_pointer() for item in bpy.data.all_ids}
    try:
        collection = bpy.data.collections.new("Second Life Body")
        bpy.context.scene.collection.children.link(collection)
        try:
            with bpy.data.libraries.load(path, link=False) as (source, loaded):
                loaded.objects = list(source.objects)
        except OSError:
            raise SecondLifeError("The Second Life Body file '{:s}' isn't a .blend file".format(path)) from None
        objects = [obj for obj in loaded.objects if obj is not None]
        for obj in objects:
            collection.objects.link(obj)
        for obj in objects:
            obj.hide_set(True)
        bpy.context.view_layer.update()  # their matrices, under their parents
        picked = pick_body(objects, os.path.basename(path))
        feet = ", ".join("'{:s}'".format(obj.name) for obj in picked.feet) or "none"
        _described[(path, os.path.getmtime(path))] = "Body '{:s}', feet {:s}".format(picked.body.name, feet)
        picked.armature.data.pose_position = 'REST'
        bpy.context.view_layer.update()
        yield picked
    finally:
        # Every datablock the load brought, the file's Library entry, fonts and particle settings included.
        added = [item for item in bpy.data.all_ids if item.as_pointer() not in before]
        if added:
            bpy.data.batch_remove(added)


_described = {}


def described(path):
    """What `describe` found in the file at its current version, or None before it looked (UI drawing can't
    load data)."""
    if not path or not os.path.isfile(path):
        return describe(path)
    return _described.get((path, os.path.getmtime(path)))


def describe(path):
    """What the Second Life Body file holds, in a few words (once per file version)."""
    if not path or not os.path.isfile(path):
        return "The Second Life Body file '{:s}' doesn't exist".format(path)
    key = (path, os.path.getmtime(path))
    if key not in _described:
        try:
            with loaded_body(path):
                pass  # it notes what it found
        except SecondLifeError as error:
            _described[key] = str(error)
    return _described[key]


# -----------------------------------------------------------------------------
# Placing a piece on the body

def surface_normals(points, tris):
    """Unit normals of the vertices ``points`` of triangles ``tris``, weighted by the triangles' areas."""
    a, b, c = (points[tris[:, k]] for k in range(3))
    faces = np.cross(b - a, c - a)
    normals = np.zeros_like(points)
    for k in range(3):
        np.add.at(normals, tris[:, k], faces)
    length = np.linalg.norm(normals, axis=1)
    return normals / np.maximum(length, 1e-12)[:, None]


@dataclass
class Placement:
    """Where `place` put a piece: moved by ``offset``, its median gap to the skin ``gap`` (m), the part of its
    points over 5 mm inside the body ``inside``, and how far it reaches past the body's bounds ``outside`` (m)."""
    offset: np.ndarray
    gap: float
    inside: float
    outside: float
    cost: float = 0.0
    # Every spot refined for the piece, this one included, cheapest first (see `choose_together`).
    choices: list = field(default_factory=list)

    @property
    def ok(self):
        return self.gap <= MAX_GAP and self.inside <= MAX_INSIDE and self.outside <= MAX_OUTSIDE


# Pieces of one import are one garment (`choose_together`): its centre is where most of it lands (the median of
# the placed pieces' best spots, weighted by their vertices). A placed piece moves to another of its spots that
# passes the limits and costs under CLEAR times its best only when that spot is over FAR closer to that centre.
# The storm skirt's layers fit the hips and the chest 30 cm above nearly alike (9 of 15 land on the hips);
# a strap fitting either leg keeps its best side, both being about as far from the centre.
CLEAR = 2.0
FAR = 0.15


def _weighted_median(values, weights):
    order = np.argsort(values)
    total = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    return values[order][np.searchsorted(total, total[-1] / 2.0)]


def choose_together(middles, placements, weights):
    """The spot of each piece (unplaced bounds centre ``middles``, `place` results ``placements``, vertex counts
    ``weights``): its best, or a near-best spot that passes the limits and lies over FAR closer to the garment's
    centre. A piece whose best spot fails stays as it is."""
    placed = [k for k, placement in enumerate(placements) if placement.ok]
    if not placed:
        return list(placements)
    spots = np.array([middles[k] + placements[k].offset for k in placed])
    mass = [weights[k] for k in placed]
    centre = np.array([_weighted_median(spots[:, axis], mass) for axis in range(3)])
    chosen = []
    for middle, placement in zip(middles, placements):
        if not placement.ok:
            chosen.append(placement)
            continue
        here = np.linalg.norm(middle + placement.offset - centre)
        near = [choice for choice in placement.choices
                if choice.ok and choice.cost < CLEAR * placement.cost and
                np.linalg.norm(middle + choice.offset - centre) < here - FAR]
        chosen.append(min(near, key=lambda choice: np.linalg.norm(middle + choice.offset - centre))
                      if near else placement)
    return chosen


def _gaps(tree, points, body_points, body_normals, reach=None):
    """Each point's gap out of the skin (negative inside), and the normal it was measured along. With
    ``reach``, points farther than that from the body count as ``reach`` outside (the search stops there)."""
    # Threads pay off for the coarse grid's many points only (15 pieces: 7 s, 14 s threaded throughout,
    # 46 s without); for a few hundred, starting them costs more.
    workers = -1 if len(points) > 20000 else 1
    if reach is None:
        _distance, nearest = tree.query(points, workers=workers)
        normals = body_normals[nearest]
        return np.einsum("ij,ij->i", points - body_points[nearest], normals), normals
    distance, nearest = tree.query(points, distance_upper_bound=reach, workers=workers)
    found = np.isfinite(distance)
    gaps = np.full(len(points), reach)
    normals = np.zeros_like(points)
    normals[found] = body_normals[nearest[found]]
    gaps[found] = np.einsum("ij,ij->i", points[found] - body_points[nearest[found]], normals[found])
    return gaps, normals


def _costs(gaps):
    """Cost of each row of ``gaps``: the mean of its TRIM smallest capped squared errors."""
    errors = gaps - SKIN_GAP
    errors = np.where(errors < 0.0, INSIDE_WEIGHT * errors, errors)
    squares = np.minimum(np.abs(errors), CAP) ** 2
    kept = max(1, int(TRIM * squares.shape[-1]))
    return np.sort(squares, axis=-1)[..., :kept].mean(axis=-1)


def place(points, body_points, body_normals, seed=0):
    """Where the piece ``points`` (world) sits on the body (``body_points`` with ``body_normals``), moved only.
    Returns a Placement."""
    from .robust_transfer import dependencies
    dependencies.add_dll_directories()
    from scipy.spatial import cKDTree

    tree = cKDTree(body_points)
    rng = np.random.default_rng(seed)
    middle = (points.min(axis=0) + points.max(axis=0)) / 2.0
    centred = points - middle
    half = (centred.max(axis=0) - centred.min(axis=0)) / 2.0
    low, high = body_points.min(axis=0) - half, body_points.max(axis=0) + half
    axes = [np.arange(low[k], high[k] + 1e-9, GRID_STEP) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    # Only centres from which the piece can touch the body.
    reach, _nearest = tree.query(grid, workers=-1)
    grid = grid[reach <= np.linalg.norm(half) + GRID_STEP]

    coarse = centred[rng.choice(len(centred), min(COARSE_SAMPLES, len(centred)), replace=False)]
    # Gaps past CAP cost the same, so the coarse search needn't look farther.
    gaps, _normals = _gaps(tree, (grid[:, None, :] + coarse[None]).reshape(-1, 3), body_points, body_normals,
                           reach=SKIN_GAP + CAP)
    order = np.argsort(_costs(gaps.reshape(len(grid), len(coarse))))
    starts = []
    for index in order:
        if all(np.linalg.norm(grid[index] - other) > 1.5 * GRID_STEP for other in starts):
            starts.append(grid[index])
            if len(starts) == CANDIDATES:
                break

    fine = centred[rng.choice(len(centred), min(FINE_SAMPLES, len(centred)), replace=False)]
    results = []
    for start in starts:
        at = start.copy()
        for _step in range(STEPS):
            gaps, normals = _gaps(tree, fine + at, body_points, body_normals)
            errors = gaps - SKIN_GAP
            weights = 1.0 / (1.0 + (errors / 0.01) ** 2)
            matrix = (normals * weights[:, None]).T @ normals + 1e-9 * np.eye(3)
            step = np.linalg.solve(matrix, -(normals * (weights * errors)[:, None]).sum(axis=0))
            at = at + step
            if np.linalg.norm(step) < 1e-4:
                break
        results.append((float(_costs(_gaps(tree, fine + at, body_points, body_normals)[0])), at))
    choices = []
    low, high = body_points.min(axis=0), body_points.max(axis=0)
    for cost, at in sorted(results, key=lambda result: result[0]):
        placed = centred + at
        gaps, _normals = _gaps(tree, placed, body_points, body_normals)
        outside = max(0.0, float(np.max(low - placed.min(axis=0))), float(np.max(placed.max(axis=0) - high)))
        choices.append(Placement(at - middle, float(np.median(np.abs(gaps))), float((gaps < -INSIDE).mean()),
                                 outside, cost))
    best = choices[0]
    best.choices = choices
    return best


# -----------------------------------------------------------------------------
# Lining up an import

def torso_centre(joints):
    """Where a piece that can't be placed goes: halfway between the hips and the neck."""
    return (np.asarray(joints[lineup.TORSO]) + np.asarray(joints[lineup.NECK])) / 2.0


def borrow_groups(piece, body_points, body_groups):
    """Give each vertex of ``piece`` the weights of the nearest of ``body_points`` in ``body_groups`` (name ->
    weight per body point), as vertex groups."""
    from .robust_transfer import dependencies
    dependencies.add_dll_directories()
    from scipy.spatial import cKDTree

    _distance, nearest = cKDTree(body_points).query(lineup.mesh_positions(piece))
    for name, weights in body_groups.items():
        levels = np.round(weights[nearest] * 255.0).astype(np.int64)
        if not levels.any():
            continue
        group = piece.vertex_groups.new(name=name)
        for level in np.unique(levels[levels > 0]):
            group.add(np.flatnonzero(levels == level).tolist(), level / 255.0, 'REPLACE')


@dataclass
class SecondLifeResult:
    """What `line_up` did: the names of the pieces it ``placed`` on the Second Life body, those that ``failed``
    (put at its torso), the root prims it ``skipped``, and Line Up's ``plan``."""
    placed: list
    failed: list
    skipped: list
    plan: object


def _surface(objects):
    """Rest positions, triangles and weights (group name -> weight per point) of ``objects``, as one."""
    points, tris, groups, first = [], [], {}, 0
    for obj in objects:
        co = lineup.mesh_positions(obj)
        names = [group.name for group in obj.vertex_groups]
        weights = lineup.mesh_weights(obj, names)
        for k, name in enumerate(names):
            column = groups.setdefault(name, [])
            column.append((first, weights.column(k)))
        points.append(co)
        tris.append(lineup.mesh_tris(obj) + first)
        first += len(co)
    columns = {}
    for name, parts in groups.items():
        column = np.zeros(first)
        for start, values in parts:
            column[start:start + len(values)] = values
        columns[name] = column
    return np.concatenate(points), np.concatenate(tris), columns


def line_up(context, meshes, body, stretch, fit):
    """Line up the Second Life import ``meshes`` with ``body``: each piece put on the Second Life body by moving
    it, then that body lined up with ``body`` and the pieces along with it (`lineup.plan`). Root prims stay as
    they are. Returns a SecondLifeResult; raises SecondLifeError or lineup.LineUpError, leaving the pieces as
    they were."""
    pieces, roots = split_import(meshes)
    saved = []
    for obj in pieces:
        co = np.empty(len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get("co", co)
        saved.append((obj, obj.matrix_world.copy(), co))
    try:
        with loaded_body(body_setting()) as loaded:
            for obj in pieces:
                bake(obj)
            points, tris, groups = _surface([loaded.body] + loaded.feet)
            normals = surface_normals(points, tris)
            rig = lineup.rig_parts(loaded.armature)
            deforming = rig[1]
            groups = {name: column for name, column in groups.items() if name in deforming}
            torso = torso_centre(lineup.armature_rig(loaded.armature, rig=rig)[2])
            placed, failed = [], []
            spread = [lineup.mesh_positions(obj) for obj in pieces]
            middles = [(where.min(axis=0) + where.max(axis=0)) / 2.0 for where in spread]
            placements = choose_together(middles, [place(where, points, normals) for where in spread],
                                         [len(where) for where in spread])
            for obj, where, placement in zip(pieces, spread, placements):
                if placement.ok:
                    offset = placement.offset
                    placed.append(obj)
                else:
                    offset = torso - (where.min(axis=0) + where.max(axis=0)) / 2.0
                    failed.append(obj)
                obj.data.transform(Matrix.Translation(Vector(offset)))
            for obj in placed:
                borrow_groups(obj, points, groups)
            result = lineup.plan(pieces, loaded.armature, body, context.evaluated_depsgraph_get(), stretch=stretch,
                                 fit=fit)
            lineup.write_meshes(result)
    except Exception:
        for obj, matrix, co in saved:
            obj.vertex_groups.clear()
            obj.data.vertices.foreach_set("co", co)
            obj.data.update()
            obj.matrix_world = matrix
        raise
    finally:
        for obj in pieces:
            obj.vertex_groups.clear()
    result.armature = None  # the Second Life rig is gone; its rest pose never moves
    return SecondLifeResult([obj.name for obj in placed], [obj.name for obj in failed],
                            [obj.name for obj in roots], result)
