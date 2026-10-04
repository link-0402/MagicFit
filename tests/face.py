"""A test scene for Face weights: one of the game's faces, built from the face data the add-on reads from
the game (facing.Reference).

`build` makes the face's skin (named like the game's, c0201f0001, so Detect finds it by name), its other
parts (lashes, brows and tear lines in one mesh, ears or horns in another) and an eyeball sphere per eye,
none of them weighted and without an armature, as a face imported without its skeleton would be.
`reshape` then moves the skin's vertices to make it a custom face.
"""

import bpy
import numpy as np

from magic_fit import facing

RACE = "c0201"
NUMBER = 1


def clear_scene():
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for armature in list(bpy.data.armatures):
        bpy.data.armatures.remove(armature)


def reference(race=RACE, number=NUMBER):
    return facing.Reference.get().face(race, number)


def _object(name, points, tris):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in np.asarray(points, dtype=np.float64)], [],
                     [tuple(int(i) for i in t) for t in np.asarray(tris)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _joined(parts):
    points, tris = [], []
    start = 0
    for part in parts:
        points.append(part.points)
        tris.append(part.tris + start)
        start += len(part.points)
    return np.vstack(points), np.vstack(tris)


def sphere(center, radius, rings=12, segments=16):
    """Points and triangles of a UV sphere."""
    points = [(0.0, 0.0, radius), (0.0, 0.0, -radius)]
    for i in range(1, rings):
        theta = np.pi * i / rings
        for j in range(segments):
            phi = 2.0 * np.pi * j / segments
            points.append((radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi),
                           radius * np.cos(theta)))
    points = np.array(points) + np.asarray(center)
    tris = []
    for j in range(segments):
        k = (j + 1) % segments
        tris.append((0, 2 + j, 2 + k))
        last = 2 + (rings - 2) * segments
        tris.append((1, last + k, last + j))
    for i in range(rings - 2):
        for j in range(segments):
            k = (j + 1) % segments
            a, b = 2 + i * segments + j, 2 + i * segments + k
            c, d = a + segments, b + segments
            tris += [(a, c, d), (a, d, b)]
    return points, np.array(tris)


def build(race=RACE, number=NUMBER, extras=True, eyes=True):
    """The face's skin, and with ``extras`` and ``eyes`` its other parts: {'skin', 'etc', 'fac', 'eye_l',
    'eye_r'} -> object (those it has)."""
    face = reference(race, number)
    label = "{:s}f{:04d}".format(race, face.number)
    objects = {"skin": _object(label + "_fac_a", face.points, face.tris)}
    if extras:
        # The parts every variant shows (no face feature attributes), as a game face has them.
        etc = [part for part in face.extras if part.kind == 2 and "fv_" not in part.attributes]
        fac = [part for part in face.extras if part.kind == 1 and "fv_" not in part.attributes]
        if etc:
            objects["etc"] = _object(label + "_etc_a", *_joined(etc))
        if fac:
            objects["fac"] = _object(label + "_fac_b", *_joined(fac))
    if eyes:
        for side, eye in face.eyes().items():
            points, tris = sphere(eye.center, eye.radius * 0.97)
            objects["eye_" + side] = _object("Eyeball_" + side, points, tris)
    return objects


def _subset(points, tris, keep):
    """The vertices ``keep`` (bool) of a mesh and its triangles among them, renumbered."""
    index = np.flatnonzero(keep)
    lookup = np.full(len(points), -1)
    lookup[index] = np.arange(len(index))
    sub = lookup[np.asarray(tris)]
    return np.asarray(points)[index], sub[np.all(sub >= 0, axis=1)]


def _grid(corner, u, v, steps=4):
    """Points and triangles of a flat quad from ``corner`` along ``u`` and ``v``, ``steps`` squares each way."""
    points = [np.asarray(corner) + np.asarray(u) * i / steps + np.asarray(v) * j / steps
              for j in range(steps + 1) for i in range(steps + 1)]
    tris = []
    for j in range(steps):
        for i in range(steps):
            a = j * (steps + 1) + i
            tris += [(a, a + 1, a + steps + 2), (a, a + steps + 2, a + steps + 1)]
    return np.array(points), np.array(tris)


def _merged(parts):
    points, tris, start = [], [], 0
    for p, t in parts:
        points.append(p)
        tris.append(np.asarray(t) + start)
        start += len(p)
    return np.vstack(points), np.vstack(tris)


def build_mouth(race=RACE, number=NUMBER, drop=0.002, socket_in_skin=False, joined=False, welded=False,
                tongue_move=(0.0, 0.0, 0.0), tongue_scale=1.0):
    """A face whose mouth comes in parts of its own, as custom faces' do: the game face's skin without its teeth and
    tongue ('skin'), its teeth and gums ('teeth', upper and lower rows; moved ``drop`` down, as a resculpt would put
    them), its tongue ('tongue'; with ``joined`` in the teeth's mesh, after them), a box lining the mouth around them,
    open at the front ('socket'; with ``socket_in_skin`` a loose part of the skin, after the skin's own vertices), a
    barbell through the tongue (two balls and a bar, 'piercing') and the eyeballs. With ``welded`` the rows are joined
    at the back into one piece (gums of both rows in one mesh); the tongue is moved by ``tongue_move`` and scaled by
    ``tongue_scale`` about its middle (a resculpted tongue). None weighted. Returns the objects and, for checking,
    {'upper': whether each teeth vertex is in the upper row, 'tongue': the tongue's vertices in its mesh, 'socket': the
    socket's vertices in its mesh}."""
    face = reference(race, number)
    label = "{:s}f{:04d}".format(race, face.number)
    names = face.names
    up = face.weights[:, names.index("j_f_hagukiup")]
    down = face.weights[:, names.index("j_f_hagukidn")]
    teeth = up + down > 0.99
    # The tongue: the game's loose part mostly on the tongue bones.
    on_tongue = face.weights[:, [j for j, name in enumerate(names) if name.startswith("j_f_bero")]].sum(1)
    parts, count = facing.loose_parts(face.points, face.tris)
    tongue = np.isin(parts, [k for k in range(count) if on_tongue[parts == k].mean() > 0.5])
    teeth_points, teeth_tris = _subset(face.points, face.tris, teeth)
    teeth_points = teeth_points - (0.0, 0.0, drop)
    if welded:
        # A strip at the back of each side joining the rows' back-most vertices: one piece.
        rows = (up > down)[teeth]
        bridge = []
        for side in (1.0, -1.0):
            ends = []
            for row in (rows, ~rows):
                mine = np.flatnonzero(row & (teeth_points[:, 0] * side > 0.005))
                ends.append(mine[np.argsort(-teeth_points[mine, 1])[:2]])
            (u1, u2), (l1, l2) = ends
            bridge += [(u1, l1, u2), (u2, l1, l2)]
        teeth_tris = np.vstack([teeth_tris, bridge])
    tongue_points, tongue_tris = _subset(face.points, face.tris, tongue)
    middle = tongue_points.mean(0)
    tongue_points = middle + (tongue_points - middle) * tongue_scale + np.asarray(tongue_move)
    skin_points, skin_tris = _subset(face.points, face.tris, ~teeth & ~tongue)
    # The box lining the mouth, snug around the rows of teeth like a mod's mouth socket: roof, floor, sides and back,
    # open at the front behind the lips.
    low, high = teeth_points.min(0) - (0.002, 0.0, 0.001), teeth_points.max(0) + (0.002, 0.002, 0.001)
    low[1] = teeth_points[:, 1].min() + 0.004
    size = high - low
    walls = [_grid(low + (0, 0, size[2]), (size[0], 0, 0), (0, size[1], 0)),  # roof
             _grid(low, (0, size[1], 0), (size[0], 0, 0)),  # floor
             _grid(low, (0, 0, size[2]), (0, size[1], 0)),  # sides
             _grid(low + (size[0], 0, 0), (0, size[1], 0), (0, 0, size[2])),
             _grid(low + (0, size[1], 0), (size[0], 0, 0), (0, 0, size[2]))]  # back
    socket_points, socket_tris = _merged(walls)
    # A barbell through the tongue a third of the way back from its tip: a bar across it, a ball at each end.
    tip = tongue_points[:, 1].min()
    y = tip + (tongue_points[:, 1].max() - tip) / 3.0
    near = np.abs(tongue_points[:, 1] - y) < 0.002
    middle = np.array((0.0, y, 0.5 * (tongue_points[near, 2].min() + tongue_points[near, 2].max())))
    reach = 0.5 * (tongue_points[near, 2].max() - tongue_points[near, 2].min()) + 0.0015
    bar_points, bar_tris = _merged([_grid(middle + (-0.0005, -0.0005, -reach), (0.001, 0, 0), (0, 0, 2 * reach), 2),
                                    _grid(middle + (-0.0005, 0.0005, -reach), (0, 0, 2 * reach), (0.001, 0, 0), 2)])
    balls = [sphere(middle + (0.0, 0.0, side * reach), 0.0015, 6, 8) for side in (-1, 1)]
    piercing_points, piercing_tris = _merged([(bar_points, bar_tris)] + balls)
    objects = {}
    info = {"upper": (up > down)[teeth]}
    if socket_in_skin:
        info["socket"] = np.arange(len(skin_points), len(skin_points) + len(socket_points))
        skin_points, skin_tris = _merged([(skin_points, skin_tris), (socket_points, socket_tris)])
    else:
        objects["socket"] = _object("Mouth", socket_points, socket_tris)
        info["socket"] = np.arange(len(socket_points))
    objects["skin"] = _object(label + "_fac_a", skin_points, skin_tris)
    if joined:
        info["tongue"] = np.arange(len(teeth_points), len(teeth_points) + len(tongue_points))
        objects["teeth"] = _object("Teeth and Tongue", *_merged([(teeth_points, teeth_tris),
                                                                 (tongue_points, tongue_tris)]))
        objects["tongue"] = objects["teeth"]
    else:
        info["tongue"] = np.arange(len(tongue_points))
        objects["teeth"] = _object("Teeth", teeth_points, teeth_tris)
        objects["tongue"] = _object("Tongue", tongue_points, tongue_tris)
    objects["piercing"] = _object("Tongue Piercing", piercing_points, piercing_tris)
    for side, eye in face.eyes().items():
        points, tris = sphere(eye.center, eye.radius * 0.97)
        objects["eye_" + side] = _object("Eyeball_" + side, points, tris)
    return objects, info


def select(objects, active):
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active


def points_of(obj):
    return np.array([obj.matrix_world @ v.co for v in obj.data.vertices])


def weights_of(obj, names):
    """Weights (n, len(names)) of ``obj`` over bone ``names``."""
    column = {name: i for i, name in enumerate(names)}
    out = np.zeros((len(obj.data.vertices), len(names)))
    for vert in obj.data.vertices:
        for elem in vert.groups:
            name = obj.vertex_groups[elem.group].name
            if name in column:
                out[vert.index, column[name]] = elem.weight
    return out


def reshape(obj, function):
    """Move the vertices of ``obj`` (world = local here) by ``function(points) -> new points``."""
    points = points_of(obj)
    moved = function(points)
    for vert, p in zip(obj.data.vertices, moved):
        vert.co = p
    obj.data.update()


def widen_eyes(face, amount=0.0025):
    """A function that opens the eyes of ``face`` wider: skin near each eye's opening moves away from the
    eye's center line, up above it and down below it, fading out 12 mm around."""
    eyes = face.eyes()

    def function(points):
        out = points.copy()
        for eye in eyes.values():
            offset = points - eye.center
            near = np.linalg.norm(offset[:, [0, 2]], axis=1)
            fade = np.clip(1.0 - near / 0.016, 0.0, 1.0) ** 2
            front = offset[:, 1] < 0.0
            out[:, 2] += np.sign(offset[:, 2]) * amount * fade * front
        return out

    return function
