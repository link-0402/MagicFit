# SPDX-License-Identifier: GPL-3.0-or-later

"""Lining hair up on a head.

Hair from another game, The Sims or VRChat was made on its own head: another size, sitting somewhere else. Line
Up's Hair mode puts it on the head picked (the skin of the face model, its 0.0 mesh), without that other head and
without the hair's weights or armature:

- Where the hair's head was (`place`). Seen from the middle of the head (the centre of the largest ball inside
  its skull), the scalp lies some way off in every direction hair grows: all but the face and the neck. Seen from
  a point inside the hair, the hair's layers lie some way off in those directions too. From the middle of the
  head the hair was made on, and at the right scale, the hair rests on the scalp: along most of those directions
  one of its layers lies just Distance off it, and none sinks into it. Points on a grid over the hair are tried,
  then finer grids around the best ones, each with the scale that suits it best. A direction counts by how far its
  best layer is from where it should be (Geman-McClure, so a layer inside the scalp counts ten times as much as
  one outside, and none coming close counts as much as a bad one): sheets the game's hair cuts through the skull,
  flyaway strands and the opening for the face don't throw it. The hair is then scaled about that point and moved
  onto the head's middle. It isn't turned: it has to stand upright and face the front like the head, as imports
  from the game, The Sims and VRChat do.
- Pushing it out (`push_out`). Hair made on another head clips where this one is fuller. Every point of the hair
  is seen from the middle of the head: from its centre for the skull and face, from the neck's axis (sideways)
  for the neck. Each direction rises by the most any point along it needs to clear the scalp by Distance, over
  about a centimetre around it and easing out past that. Everything along a direction rises alike, so layers lying
  on each other keep their distance and order, and a strand that clips in one spot lifts with those around it.
  Points deeper in than a quarter of the head's radius are tucked in on purpose (the game's hair cuts sheets
  through the skull) and stay, and hair far out from the scalp rises less, to nothing 10 cm out.

Like the other engines, this works on world space arrays and doesn't need the 3D view.
"""

import math

import numpy as np
from mathutils.bvhtree import BVHTree

from . import fitting, lineup

UP = np.array((0.0, 0.0, 1.0))
# The middle of the head: the largest ball inside its skull whose top lies no farther straight up than this
# many radii (a ball under the crown, not one in the jaw).
CEILING = 1.35
BALL_STEPS = 30
# Scalp points the head is sampled at, and how many directions to them the hair is matched along.
HEAD_SAMPLES = 4000
DIRECTIONS = 150
# Hair grows all over the head but the face (facing forward, below this many radii over the middle) and the
# neck (this many radii under it).
FACE_ABOVE = 0.2
FACE_FACING = 0.5
NECK_BELOW = 1.2
# Matching: each direction's best layer within this many head radii (Geman-McClure), a layer sunk into the
# scalp counting this many times as much as one off it...
MATCH_SCALE = 0.15
SUNK_COST = 10.0
# ...among the first this many layers along it.
LAYERS = 4
# Points tried for the head's middle: a grid this many points across the hair, those with hair straight above,
# then finer grids (a third of the step each time) around the best few.
GRID = 7
LEVELS = 4
KEEP = 4
# Scales tried: from 2/3 of the head's width over the hair's to 5 times that, then 6 % around the best.
SCALE_RANGE = (-0.4, 1.6)
SCALE_STEPS = 81
FINE_RANGE = 0.06
FINE_STEPS = 49
# Pushing out: directions every this many metres over the head, and this many around it...
PUSH_STEP = 0.002
PUSH_AROUND = 180
# ...each rising by the most a point along it needs, over this far around it, easing out past that.
PUSH_SPREAD = 0.012
# Points deeper in than this many head radii are tucked in on purpose and stay (easing in from 60 % of that).
TUCKED = 0.25
TUCKED_EASE = 0.6
# Hair farther out from the scalp than PUSH_BAND rises less, to nothing PUSH_FADE farther.
PUSH_BAND = 0.06
PUSH_FADE = 0.04
# Holes in the head (its ears) and the neck's open end are filled in from around them, this far.
HOLE_FILL = 0.03


class HairLineUpError(Exception):
    """Why hair can't be lined up."""


# -----------------------------------------------------------------------------
# The head

def _sample(points, tris, count, seed):
    """``count`` points spread evenly over the surface by area."""
    a, b, c = (points[tris[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if total <= 0.0:
        return np.zeros((0, 3))
    rng = np.random.default_rng(seed)
    tri = rng.choice(len(tris), size=count, p=area / total)
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])


def _grid(low, high, step):
    axes = [np.arange(low[k] + 0.5 * step, high[k], step) for k in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _ray_lengths(bvh, origins, direction, limit=1e9):
    """How far each ray from ``origins`` along ``direction`` goes until it meets the surface (inf: never)."""
    out = np.full(len(origins), np.inf)
    cast = bvh.ray_cast
    direction = tuple(direction)
    for k, origin in enumerate(origins.tolist()):
        hit = cast(origin, direction, limit)
        if hit[0] is not None:
            out[k] = hit[3]
    return out


class Head:
    """The head hair is lined up on (world space ``points`` and ``tris``): its surface, its middle (``centre``,
    ``radius``: the largest ball inside the skull under the crown) and the directions from there to the scalp
    where hair grows (``directions``, ``distances``)."""

    def __init__(self, points, tris):
        self.points = np.asarray(points, dtype=np.float64)
        self.tris = np.asarray(tris, dtype=np.int64)
        if not len(self.tris):
            raise HairLineUpError("The head has no faces")
        self.bvh = BVHTree.FromPolygons(self.points.tolist(), self.tris.tolist(), all_triangles=True)
        self.surface = fitting.BodySurface(self.points, self.tris, self.bvh)
        self.centre, self.radius = self._middle()
        samples = _sample(self.points, self.tris, HEAD_SAMPLES, 1)
        normals = self.surface.nearest(samples).normal
        z = samples[:, 2]
        face = (normals[:, 1] < -FACE_FACING) & (z < self.centre[2] + FACE_ABOVE * self.radius)
        grows = ~face & (z > self.centre[2] - NECK_BELOW * self.radius)
        chosen = _spread_out(samples[grows], DIRECTIONS)
        offsets = chosen - self.centre
        self.distances = np.linalg.norm(offsets, axis=1)
        self.directions = offsets / np.maximum(self.distances, 1e-12)[:, None]

    def _middle(self):
        """The centre and radius of the largest ball inside the head whose top lies within CEILING radii straight
        under the head's surface: on a grid, then on finer grids around the best point."""
        low, high = self.points.min(axis=0), self.points.max(axis=0)
        step = (high - low).max() / BALL_STEPS
        best = None
        for _level in range(3):
            if best is None:
                points = _grid(low, high, step)
            else:
                points = _grid(best[0] - 3.0 * step, best[0] + 3.0 * step, step)
            near = self.surface.nearest(points)
            inside = near.found & (near.distance < 0.0)
            points, depth = points[inside], -near.distance[inside]
            if not len(points):
                break
            up = _ray_lengths(self.bvh, points, UP)
            under = up <= CEILING * depth
            if not under.any():
                break
            k = int(np.argmax(np.where(under, depth, -1.0)))
            best = (points[k], float(depth[k]))
            step /= 3.0
        if best is None:
            raise HairLineUpError("Can't find the inside of the head: it has to be the skin of a whole head, like "
                                  "the face model's 0.0 mesh")
        return best


def _spread_out(points, count):
    """``count`` of ``points``, each as far from those picked before as can be."""
    if len(points) <= count:
        return points
    chosen = [0]
    gap = np.linalg.norm(points - points[0], axis=1)
    for _ in range(count - 1):
        k = int(np.argmax(gap))
        chosen.append(k)
        gap = np.minimum(gap, np.linalg.norm(points - points[k], axis=1))
    return points[chosen]


# -----------------------------------------------------------------------------
# Placing the hair

def _layers(bvh, origin, directions, count):
    """Distances from ``origin`` to the first ``count`` layers of the surface along each direction (inf: none)."""
    out = np.full((len(directions), count), np.inf)
    cast = bvh.ray_cast
    start = tuple(origin)
    for i, direction in enumerate(directions.tolist()):
        point, travelled = start, 0.0
        for k in range(count):
            hit = cast(point, direction, 1e9)
            if hit[0] is None:
                break
            travelled += hit[3]
            out[i, k] = travelled
            travelled += 1e-6
            point = (hit[0][0] + 1e-6 * direction[0], hit[0][1] + 1e-6 * direction[1],
                     hit[0][2] + 1e-6 * direction[2])
    return out


def _scale_costs(layers, distances, scales, gap, scale):
    """The cost of each of ``scales``: per direction, its layer that comes closest to lying ``gap`` off the scalp
    (``distances`` away), by Geman-McClure at ``scale``, sunk ones counting SUNK_COST times; at most 1."""
    error = (scales[:, None, None] * layers[None, :, :] - distances[None, :, None] - gap) / scale
    error = np.where(np.isfinite(error), error, 1e6)
    cost = error * error / (1.0 + error * error)
    cost = np.where(error < 0.0, SUNK_COST * cost, cost)
    return np.minimum(cost.min(axis=2), 1.0).mean(axis=1)


class Placement:
    """Where the hair goes: x -> ``scale`` * x + ``offset``, and how well it rests on the head (``cost``, 0 at
    best, 1 at worst)."""

    def __init__(self, scale, offset, cost):
        self.scale = scale
        self.offset = offset
        self.cost = cost

    def move(self, points):
        return self.scale * points + self.offset


def place(points, tris, head, gap):
    """Find where the hair (world space ``points``, ``tris``) rests on the head, ``gap`` off its scalp (see the
    module). Raises HairLineUpError when there's nothing like a head's place in it."""
    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    if not len(tris):
        raise HairLineUpError("The hair has no faces")
    bvh = BVHTree.FromPolygons(points.tolist(), tris.tolist(), all_triangles=True)
    low, high = points.min(axis=0), points.max(axis=0)
    size = high - low
    width = np.percentile(points[:, 0], 98) - np.percentile(points[:, 0], 2)
    if width <= 0.0:
        raise HairLineUpError("The hair has no width")
    rough = np.ptp(head.points[:, 0]) / width
    scales = rough * np.exp(np.linspace(*SCALE_RANGE, SCALE_STEPS))
    match = MATCH_SCALE * head.radius
    cast = bvh.ray_cast
    up = tuple(UP)
    step = size / GRID
    best = []
    for level in range(LEVELS):
        if level == 0:
            axes = [low[k] + size[k] * (np.arange(GRID) + 0.5) / GRID for k in range(3)]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            grid = np.array([point for point in grid.tolist() if cast(point, up, 1e9)[0] is not None]).reshape(-1, 3)
            # Every third direction at first.
            use = slice(None, None, 3)
        else:
            step = step / 3.0
            ring = np.linspace(-1.0, 1.0, 3)
            offsets = np.stack(np.meshgrid(ring, ring, ring, indexing="ij"), axis=-1).reshape(-1, 3) * step
            grid = np.concatenate([centre + offsets for _cost, centre, _scale in best[:KEEP]])
            use = slice(None)
        tried = []
        for centre in grid:
            layers = _layers(bvh, centre, head.directions[use], LAYERS)
            costs = _scale_costs(layers, head.distances[use], scales, gap, match)
            k = int(np.argmin(costs))
            tried.append((float(costs[k]), centre, float(scales[k])))
        if not tried:
            break
        tried.sort(key=lambda entry: entry[0])
        best = tried
        if level >= 1:
            scales = best[0][2] * np.exp(np.linspace(-FINE_RANGE, FINE_RANGE, FINE_STEPS))
    if not best:
        raise HairLineUpError("Found no place for a head in the hair: it has to cover the top of a head")
    cost, centre, scale = best[0]
    return Placement(scale, head.centre - scale * centre, cost)


# -----------------------------------------------------------------------------
# Pushing it out

class _Around:
    """Directions around the head: from its middle for the skull and face, from a segment down the neck's axis
    (sideways) for the neck, and from that segment's bottom below it. A point maps to cells (around, along): along
    runs over the surface, negative below the neck's bottom, 0 to the segment's length along it, more over the
    skull."""

    def __init__(self, head):
        points = head.points
        low = points[:, 2].min()
        ring = points[points[:, 2] < low + 0.1 * (points[:, 2].max() - low)]
        self.top = head.centre
        self.bottom = np.array((head.centre[0], ring[:, 1].mean(), low))
        self.radius = head.radius
        axis = self.top - self.bottom
        self.length = max(float(np.linalg.norm(axis)), 1e-6)
        self.axis = axis / self.length
        side = np.cross(self.axis, (0.0, 1.0, 0.0))
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(self.axis, (1.0, 0.0, 0.0))
        self.side = side / np.linalg.norm(side)
        self.front = np.cross(self.axis, self.side)
        quarter = 0.5 * math.pi * self.radius
        self.first = -quarter
        self.around = PUSH_AROUND
        self.along = int(math.ceil((self.length + 2.0 * quarter) / PUSH_STEP)) + 1

    def cells(self, points):
        """For (n, 3) ``points``: cell coordinates around and along, and the unit direction to each and how far
        it is from where it's seen from."""
        t = np.clip((points - self.bottom) @ self.axis / self.length, 0.0, 1.0)
        origin = self.bottom + t[:, None] * self.length * self.axis
        offset = points - origin
        distance = np.linalg.norm(offset, axis=1)
        direction = offset / np.maximum(distance, 1e-12)[:, None]
        turn = np.arctan2(direction @ self.front, direction @ self.side) % (2.0 * math.pi)
        rise = np.arcsin(np.clip(direction @ self.axis, -1.0, 1.0))
        along = np.where(t >= 1.0, self.length + self.radius * np.maximum(rise, 0.0),
                         np.where(t <= 0.0, self.radius * np.minimum(rise, 0.0), t * self.length))
        return (turn / (2.0 * math.pi) * self.around - 0.5, (along - self.first) / PUSH_STEP, direction, distance)

    def rays(self):
        """The origin and direction of each cell's ray, (around, along, 3) each."""
        turn = (np.arange(self.around) + 0.5) * 2.0 * math.pi / self.around
        along = self.first + np.arange(self.along) * PUSH_STEP
        turn, along = np.meshgrid(turn, along, indexing="ij")
        outward = np.cos(turn)[..., None] * self.side + np.sin(turn)[..., None] * self.front
        rise = np.where(along > self.length, (along - self.length) / self.radius,
                        np.where(along < 0.0, along / self.radius, 0.0))
        rise = np.clip(rise, -0.5 * math.pi, 0.5 * math.pi)
        direction = np.cos(rise)[..., None] * outward + np.sin(rise)[..., None] * self.axis
        t = np.clip(along / self.length, 0.0, 1.0)
        origin = self.bottom + t[..., None] * self.length * self.axis
        return origin, direction


def _lookup(field, around, along):
    """``field`` (around, along) at fractional cells, bilinearly (wrapping around)."""
    count, rows = field.shape
    a0 = np.floor(around).astype(np.int64)
    b0 = np.clip(np.floor(along).astype(np.int64), 0, rows - 2)
    ta = around - a0
    tb = np.clip(along - b0, 0.0, 1.0)
    a0 %= count
    a1 = (a0 + 1) % count
    return ((1.0 - ta) * (1.0 - tb) * field[a0, b0] + ta * (1.0 - tb) * field[a1, b0] +
            (1.0 - ta) * tb * field[a0, b0 + 1] + ta * tb * field[a1, b0 + 1])


def _neighbours(values, known):
    """Per cell, the sum of its known neighbours' values and how many there are (wrapping around)."""
    s = np.where(known, values, 0.0)
    k = known.astype(np.float64)
    total = np.roll(s, 1, axis=0) + np.roll(s, -1, axis=0)
    count = np.roll(k, 1, axis=0) + np.roll(k, -1, axis=0)
    total[:, 1:] += s[:, :-1]
    count[:, 1:] += k[:, :-1]
    total[:, :-1] += s[:, 1:]
    count[:, :-1] += k[:, 1:]
    return total, count


def _fill(values, rings):
    """Cells without a value (nan) take the mean of their neighbours', ``rings`` deep."""
    known = np.isfinite(values)
    filled = np.where(known, values, 0.0)
    for _ in range(rings):
        total, count = _neighbours(filled, known)
        grow = ~known & (count > 0)
        if not grow.any():
            break
        filled[grow] = total[grow] / count[grow]
        known = known | grow
    return np.where(known, filled, np.nan)


def _spread_max(values, around, along):
    """Each cell takes the largest value within ``around`` and ``along`` cells (wrapping around)."""
    out = values.copy()
    for da in range(-around, around + 1):
        rolled = np.roll(values, da, axis=0)
        for db in range(-along, along + 1):
            shifted = np.zeros_like(values)
            if db > 0:
                shifted[:, db:] = rolled[:, :-db]
            elif db < 0:
                shifted[:, :db] = rolled[:, -db:]
            else:
                shifted = rolled
            np.maximum(out, shifted, out=out)
    return out


def _blur(values, passes):
    out = values
    for _ in range(passes):
        s = out.copy()
        s[:, 1:-1] = 0.25 * (out[:, :-2] + 2.0 * out[:, 1:-1] + out[:, 2:])
        out = 0.25 * (np.roll(s, 1, axis=0) + 2.0 * s + np.roll(s, -1, axis=0))
    return out


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Push:
    """How far each point of the hair rises out of the head (``rise``, along ``direction``): ``count`` points by
    more than a tenth of a millimetre, up to ``most``."""

    def __init__(self, rise, direction):
        self.rise = rise
        self.direction = direction
        self.count = int((rise > 1e-4).sum())
        self.most = float(rise.max()) if len(rise) else 0.0

    @property
    def moves(self):
        return self.rise[:, None] * self.direction


def push_out(points, head, gap):
    """How hair at (n, 3) world space ``points``, lined up, rises out of the head to clear its scalp by ``gap``
    (see the module)."""
    points = np.asarray(points, dtype=np.float64)
    around = _Around(head)
    origins, directions = around.rays()
    reach = 4.0 * head.radius
    cast = head.bvh.ray_cast
    scalp = np.full(origins.shape[:2], np.nan)
    for (a, b), _ in np.ndenumerate(scalp):
        hit = cast(tuple(origins[a, b]), tuple(directions[a, b]), reach)
        if hit[0] is not None:
            scalp[a, b] = hit[3]
    scalp = _fill(scalp, int(math.ceil(HOLE_FILL / PUSH_STEP)))
    known = np.isfinite(scalp)
    # Smoothed where the head's shape jumps (around the ears).
    scalp = np.where(known, scalp, 0.0)
    for _ in range(2):
        total, count = _neighbours(scalp, known)
        scalp = np.where(known, (total + 2.0 * scalp) / np.maximum(count + 2.0, 1e-12), 0.0)
    cell_around, cell_along, direction, distance = around.cells(points)
    over = _lookup(known.astype(np.float64), cell_around, cell_along) > 0.999
    surface = np.where(over, _lookup(scalp, cell_around, cell_along), np.nan)
    tucked = TUCKED * head.radius
    need = surface + gap - distance
    # Only points shallow enough to rise all the way say how far the hair around them rises: those deeper,
    # easing into tucked away, rise with the hair around them, and less.
    needy = over & (need > 0.0) & (distance > surface - TUCKED_EASE * tucked)
    lowest = np.zeros(scalp.shape)
    a = np.round(cell_around).astype(np.int64) % around.around
    b = np.clip(np.round(cell_along).astype(np.int64), 0, around.along - 1)
    np.maximum.at(lowest, (a[needy], b[needy]), need[needy])
    cells_along = max(1, int(math.ceil(PUSH_SPREAD / PUSH_STEP)))
    cells_around = max(1, int(math.ceil(PUSH_SPREAD / (2.0 * math.pi * head.radius / around.around))))
    field = _blur(_spread_max(lowest, cells_around, cells_along), max(cells_around, cells_along))
    rise = _lookup(field, cell_around, cell_along)
    out_of = np.where(over, distance - surface, np.inf)
    weight = np.where(over, 1.0 - _smoothstep((out_of - PUSH_BAND) / PUSH_FADE), 0.0)
    sunk = np.where(over, (surface - distance) / tucked, 0.0)
    weight *= 1.0 - _smoothstep((sunk - TUCKED_EASE) / (1.0 - TUCKED_EASE))
    return Push(rise * weight, direction)


# -----------------------------------------------------------------------------
# Blender objects

# Hair Line Up last left on a head keeps that head's name, its box then, and how it was pushed out: lined up again
# unmoved, it stays where it is (placing pushed hair once more would shrink it a little: what was pushed out rests
# closer), and is only pushed out if it wasn't, or at another distance.
LINED_UP = "magic_fit_lined_up_on"
SAME_BOX = 1e-5


def _box(obj):
    points = lineup.mesh_positions(obj)
    if not len(points):
        return [0.0] * 6
    return points.min(axis=0).tolist() + points.max(axis=0).tolist()


def remember(meshes, head_object, pushed, gap):
    """Note on ``meshes`` that they sit on ``head_object`` now, pushed out ``gap`` off it or not."""
    for obj in meshes:
        obj[LINED_UP] = {"head": head_object.name, "box": _box(obj), "pushed": bool(pushed), "gap": float(gap)}


def lined_up(meshes, head_object):
    """How Line Up left ``meshes`` on ``head_object``, when they all still are where it left them: (pushed, gap),
    else None."""
    found = None
    for obj in meshes:
        note = obj.get(LINED_UP)
        try:
            if note is None or note["head"] != head_object.name:
                return None
            if not np.allclose(np.array(note["box"]), _box(obj), atol=SAME_BOX):
                return None
            state = (bool(note["pushed"]), float(note["gap"]))
        except (KeyError, TypeError, ValueError):
            return None
        if found is not None and state != found:
            return None
        found = state
    return found


class Plan:
    """How Line Up moves hair (``meshes``) onto a head: the ``placement``, and with Push Out, how far each mesh's
    points then rise (``moves``: world space, one array per mesh; None without), with the ``push`` that found them.
    ``armatures`` are those bending only these meshes, whose rest pose moves along."""

    def __init__(self):
        self.meshes = []
        self.armatures = []
        self.placement = None
        self.push = None
        self.moves = None


def plan(meshes, armatures, head_object, depsgraph, gap, push=True, keep=False):
    """Work out how to line hair (``meshes``) up on ``head_object``, ``gap`` off its scalp, and with ``push``,
    how to push it out of it; with ``keep``, leave it where it is and only push. Raises HairLineUpError when it
    can't."""
    result = Plan()
    result.meshes, result.armatures = list(meshes), list(armatures)
    head_points, head_tris = lineup.body_surface(head_object, depsgraph)
    head = Head(head_points, head_tris)
    positions, tris, first = [], [], 0
    for obj in meshes:
        points = lineup.mesh_positions(obj)
        positions.append(points)
        tris.append(lineup.mesh_tris(obj) + first)
        first += len(points)
    points = np.concatenate(positions) if positions else np.zeros((0, 3))
    tris = np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)
    result.placement = Placement(1.0, np.zeros(3), 0.0) if keep else place(points, tris, head, gap)
    if push and len(points):
        result.push = push_out(result.placement.move(points), head, gap)
        moves = result.push.moves
        offsets = np.cumsum([0] + [len(p) for p in positions])
        result.moves = [moves[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    return result


def write_meshes(result):
    """Move the Plan's meshes (every shape key alike) onto the head."""
    placement = result.placement
    for k, obj in enumerate(result.meshes):
        moves = result.moves[k] if result.moves is not None else None
        matrix = lineup._matrix(obj)
        inverse = np.linalg.inv(matrix)

        def moved(local):
            world = placement.move(lineup.apply_matrix(matrix, local))
            if moves is not None:
                world = world + moves
            return lineup.apply_matrix(inverse, world)

        mesh = obj.data
        keys = mesh.shape_keys
        if keys is None:
            co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", co)
            mesh.vertices.foreach_set("co", moved(co.reshape(-1, 3).astype(np.float64)).astype(np.float32).ravel())
        else:
            for block in keys.key_blocks:
                co = moved(lineup._key_positions(block)).astype(np.float32).ravel()
                block.data.foreach_set("co", co)
                if block == keys.reference_key:
                    mesh.vertices.foreach_set("co", co)
        mesh.update()


def move_rest_pose(result, armature):
    """Scale and move the rest pose of ``armature`` (in its Edit Mode) like the hair: its bones keep their place
    on it."""
    matrix = lineup._matrix(armature)
    inverse = np.linalg.inv(matrix)
    placement = result.placement
    edit_bones = armature.data.edit_bones
    # Connected bones share a point: all are read before any moves.
    ends = {bone.name: np.array((bone.head, bone.tail)) for bone in edit_bones}
    connected = {bone.name for bone in edit_bones if bone.use_connect}
    for bone in edit_bones:
        bone.use_connect = False
    for bone in edit_bones:
        head, tail = lineup.apply_matrix(inverse, placement.move(lineup.apply_matrix(matrix, ends[bone.name])))
        bone.head, bone.tail = head.tolist(), tail.tolist()
    for bone in edit_bones:
        if bone.name in connected:
            bone.use_connect = True
