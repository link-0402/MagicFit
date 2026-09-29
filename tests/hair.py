"""A test scene for Hair weights: procedural hair around the Midlander female's head (c0201).

The head is the one Hair Weights uses (magic_fit/hair_reference.npz), so the hair sits where
real hair would: a scalp cap 1 cm off the head, bangs in front of the forehead, a lock at each temple
hanging 25 cm, and a ponytail tube hanging 42 cm down the back. Each part is a loose part of one mesh,
as game hair is. The mesh has no weights and no armature. `cap` and `ear` also make scalps and cat ears
for other races' heads.
"""

import math

import bpy
import numpy as np

from magic_fit import hairing

RACE = "c0201"
CENTER = np.array((0.0, -0.015, 1.545))   # middle of the skull


def head(race=RACE):
    return hairing.Reference.get().head(race)


def center(race=RACE):
    """The middle of the skull of ``race``: CENTER, moved along with the race's head bone."""
    reference = hairing.Reference.get()
    moved = reference.rig(race, 0).by_name[hairing.HEAD].head - reference.rig(RACE, 0).by_name[hairing.HEAD].head
    return CENTER + moved


def _onto_head(directions, gap, race=RACE):
    """Points ``gap`` outside the head along rays from its middle (bisection on the head's distance), and
    how far along the rays they are."""
    field = head(race)
    middle = center(race)
    lo = np.zeros(len(directions))
    hi = np.full(len(directions), 0.3)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        outside = field.distance(middle + directions * mid[:, None]) > gap
        hi = np.where(outside, mid, hi)
        lo = np.where(outside, lo, mid)
    return middle + directions * hi[:, None], hi


def cap(rings=10, segments=24, gap=0.01, race=RACE, reach=None):
    """The scalp: the top of the head down to the ears, open over the face. With ``reach``, what stands
    farther out from the skull's middle is left out (a Miqo'te head's cat ears)."""
    verts, faces, grid = [], [], {}
    for i in range(rings + 1):
        polar = math.radians(8 + 92 * i / rings)   # from near the top down to ear height
        for j in range(segments):
            azimuth = 2 * math.pi * j / segments
            d = np.array((math.sin(polar) * math.sin(azimuth), math.sin(polar) * math.cos(azimuth), math.cos(polar)))
            # Leave the face uncovered: in front (y < 0) and below the hairline.
            if d[1] < -0.55 and d[2] < 0.45:
                continue
            grid[i, j] = len(verts)
            verts.append(d)
    points, out = _onto_head(np.array(verts), gap, race)
    keep = np.ones(len(points), dtype=bool) if reach is None else out < reach
    index = np.cumsum(keep) - 1
    for i in range(rings):
        for j in range(segments):
            quad = [grid.get((i, j)), grid.get((i, (j + 1) % segments)), grid.get((i + 1, (j + 1) % segments)),
                    grid.get((i + 1, j))]
            if None not in quad and keep[quad].all():
                faces.append([int(index[k]) for k in quad])
    return points[keep], faces


def ear(bone, length=0.11, radius=0.02, rings=11, segments=12):
    """A cat ear: a tube along ``bone`` (a `hairing.Bone`, pointing along its X axis) from its head,
    narrowing to 40 % at the tip."""
    side = np.cross(bone.x, (0.0, 0.0, 1.0))
    side /= np.linalg.norm(side)
    up = np.cross(side, bone.x)
    verts = []
    for i in range(rings + 1):
        t = i / rings
        c = bone.head + bone.x * length * t
        r = radius * (1.0 - 0.6 * t)
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append(c + r * (math.cos(a) * side + math.sin(a) * up))
    faces = []
    for i in range(rings):
        for j in range(segments):
            j1 = (j + 1) % segments
            faces.append((i * segments + j, i * segments + j1, (i + 1) * segments + j1, (i + 1) * segments + j))
    return np.array(verts), faces


def strip(top, bottom, width, side, rows=12):
    """A flat strip of hair from ``top`` to ``bottom`` (width along ``side``), two vertices per row."""
    top, bottom, side = (np.asarray(v, dtype=np.float64) for v in (top, bottom, side))
    verts = []
    for i in range(rows + 1):
        t = i / rows
        c = top + (bottom - top) * t
        verts += [c - side * width / 2, c + side * width / 2]
    faces = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(rows)]
    return np.array(verts), faces


def tube(top, bottom, radius, rings=20, segments=12, bend=(0.0, 0.04, 0.0)):
    """A ponytail: a closed tube from ``top`` to ``bottom``, bulging by ``bend`` halfway."""
    top, bottom, bend = (np.asarray(v, dtype=np.float64) for v in (top, bottom, bend))
    verts = []
    for i in range(rings + 1):
        t = i / rings
        c = top + (bottom - top) * t + bend * math.sin(math.pi * t)
        r = radius * (1.0 - 0.6 * t)
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append(c + r * np.array((math.cos(a), math.sin(a), 0.0)))
    faces = []
    for i in range(rings):
        for j in range(segments):
            j1 = (j + 1) % segments
            faces.append((i * segments + j, i * segments + j1, (i + 1) * segments + j1, (i + 1) * segments + j))
    return np.array(verts), faces


def parts(long=True):
    """(name, vertices, faces) of each loose part; ``long`` adds the locks and the ponytail."""
    result = [("cap",) + cap()]
    for x in (-0.03, 0.0, 0.03):
        result.append(("bangs",) + strip((x, -0.105, 1.605), (x * 1.2, -0.125, 1.53), 0.03, (1.0, 0.0, 0.0), rows=6))
    if long:
        for sign in (1.0, -1.0):
            result.append(("lock_" + ("l" if sign > 0 else "r"),)
                          + strip((0.085 * sign, -0.035, 1.53), (0.09 * sign, -0.045, 1.28), 0.025, (0.0, 1.0, 0.0)))
        result.append(("ponytail",) + tube((0.0, 0.105, 1.53), (0.0, 0.16, 1.11), 0.035))
    return result


def build(name="Hair", long=True):
    """The hair as one mesh object (active and selected), and the vertex range of each part."""
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    verts, faces, ranges = [], [], {}
    for part, v, f in parts(long):
        start = len(verts)
        verts += [tuple(p) for p in v]
        faces += [tuple(start + i for i in face) for face in f]
        ranges[part] = ranges.get(part, []) + list(range(start, len(verts)))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for other in bpy.context.selected_objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj, {part: np.array(indices) for part, indices in ranges.items()}


def by_name(obj):
    """{group name: weights of every vertex}."""
    count = len(obj.data.vertices)
    result = {group.name: np.zeros(count) for group in obj.vertex_groups}
    names = [group.name for group in obj.vertex_groups]
    for vert in obj.data.vertices:
        for elem in vert.groups:
            result[names[elem.group]][vert.index] = elem.weight
    return result


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for armature in list(bpy.data.armatures):
        bpy.data.armatures.remove(armature)
