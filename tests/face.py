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
