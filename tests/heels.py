"""A test scene for Heel weights: a pair of high heels on an FFXIV-like rig.

Rig: the legs of the Yet Another Devkit skeleton, with their rest positions: thigh, knee and calf
(j_asi_a, b, c), the ankle (j_asi_d, from the ankle joint to the ball of the foot), the toes (j_asi_e)
and IVCS toe bones below them, left and right.

Shoes: each an ankle strap (a tube around the shin, up to 16 cm above the ankle), a foot shell sloping
down from the heel to the toes, and a stiletto heel under the back of the foot (meters). They start with
the kind of weights a transfer from a flat foot gives them: the calf fading into the ankle bone, the
toes and the toe bones ahead of the ball, and a heel that follows the calf a little.
"""

import math

import bpy
import numpy as np

HIP = (0.0, 0.0, 1.0366)
X = 0.086                       # the legs' x (left; the right mirrors it)
ANKLE = (X, 0.0128, 0.0869)     # head of j_asi_d
BALL = (X, -0.0879, 0.0138)     # head of j_asi_e
TOE_BONES = ("iv_asi_oya_a", "iv_asi_oya_b", "iv_asi_ko_a", "iv_asi_ko_b")


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_rig(name="Rig", toes=True):
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    rig = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    edit = rig.data.edit_bones

    def bone(bone_name, head, tail, parent=None):
        item = edit.new(bone_name)
        item.head, item.tail = head, tail
        if parent is not None:
            item.parent = edit[parent]

    bone("j_kosi", HIP, (0.0, 0.0958, 1.0223))
    for side, sign in (("l", 1.0), ("r", -1.0)):
        def at(x, y, z):
            return (sign * x, y, z)
        bone("j_asi_a_" + side, at(X, -0.0170, 0.9223), at(X, -0.0283, 0.5552), "j_kosi")
        bone("j_asi_b_" + side, at(X, -0.0283, 0.5552), at(X, -0.0275, 0.4950), "j_asi_a_" + side)
        bone("j_asi_c_" + side, at(X, -0.0275, 0.4950), at(*ANKLE), "j_asi_b_" + side)
        bone("j_asi_d_" + side, at(*ANKLE), at(*BALL), "j_asi_c_" + side)
        if toes:
            bone("j_asi_e_" + side, at(*BALL), at(X, -0.2123, 0.0138), "j_asi_d_" + side)
            bone("iv_asi_oya_a_" + side, at(0.0600, -0.1099, 0.0448), at(0.0600, -0.1249, 0.0418), "j_asi_e_" + side)
            bone("iv_asi_oya_b_" + side, at(0.0600, -0.1249, 0.0418), at(0.0600, -0.1400, 0.0400), "iv_asi_oya_a_" + side)
            bone("iv_asi_ko_a_" + side, at(0.1190, -0.0879, 0.0398), at(0.1190, -0.0969, 0.0358), "j_asi_e_" + side)
            bone("iv_asi_ko_b_" + side, at(0.1190, -0.0969, 0.0358), at(0.1190, -0.1100, 0.0300), "iv_asi_ko_a_" + side)
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig


def _ring(center, radius_x, radius_y, count, axis_x, axis_y):
    a = 2.0 * math.pi * np.arange(count) / count
    return (np.asarray(center)[None, :] + np.cos(a)[:, None] * radius_x * np.asarray(axis_x)[None, :]
            + np.sin(a)[:, None] * radius_y * np.asarray(axis_y)[None, :])


def _loft(rings):
    """Faces joining consecutive rings (each with the same count), as quads."""
    count = len(rings[0])
    faces = []
    for i in range(len(rings) - 1):
        for j in range(count):
            j1 = (j + 1) % count
            faces.append((i * count + j, i * count + j1, (i + 1) * count + j1, (i + 1) * count + j))
    return faces


def shoe_parts(sign):
    """(co, faces) of one shoe: strap, foot shell and stiletto, as separate loose parts."""
    x = sign * X
    parts = []
    # Ankle strap: a tube around the shin from 16 cm above the ankle down to 3 cm below it.
    rings = [_ring((x, 0.012, z), 0.042, 0.048, 24, (1, 0, 0), (0, 1, 0)) for z in np.linspace(0.25, 0.06, 20)]
    parts.append((np.concatenate(rings), _loft(rings)))
    # Foot shell: from the heel (behind the ankle) sloping down and forward to the toes.
    path = [(0.05, 0.075), (0.02, 0.06), (-0.02, 0.045), (-0.06, 0.025), (-0.09, 0.01), (-0.12, 0.0), (-0.15, -0.005)]
    rings = []
    for (y, z), (y1, z1) in zip(path, path[1:] + [(2 * path[-1][0] - path[-2][0], 2 * path[-1][1] - path[-2][1])]):
        direction = np.array((0.0, y1 - y, z1 - z))
        direction /= np.linalg.norm(direction)
        up = np.array((0.0, -direction[2], direction[1]))
        if up[2] < 0:
            up = -up
        rings.append(_ring((x, y, z), 0.035, 0.025, 16, (1, 0, 0), up))
    parts.append((np.concatenate(rings), _loft(rings)))
    # Stiletto: a thin tube under the heel, down 9 cm below the sole.
    rings = [_ring((x, 0.045, z), 0.006, 0.006, 8, (1, 0, 0), (0, 1, 0)) for z in np.linspace(0.05, -0.04, 8)]
    parts.append((np.concatenate(rings), _loft(rings)))
    co, faces, offset = [], [], 0
    for part_co, part_faces in parts:
        co.append(part_co)
        faces += [tuple(k + offset for k in face) for face in part_faces]
        offset += len(part_co)
    return np.concatenate(co), faces


def naive_weights(co, side):
    """Weights like a transfer from a flat foot: calf fading into the ankle over the ankle, toes and toe
    bones ahead of the ball, and a heel following the calf a little."""
    height = co[:, 2] - ANKLE[2]
    calf = smoothstep(-0.02, 0.07, height)
    toe = smoothstep(-0.07, -0.10, co[:, 1]) * (1.0 - calf)
    tip = smoothstep(-0.11, -0.14, co[:, 1]) * toe
    heel = (co[:, 1] > 0.035) & (co[:, 2] < 0.06)
    calf = np.where(heel, np.maximum(calf, 0.3), calf)
    ankle = np.clip(1.0 - calf - toe, 0.0, 1.0)
    return {
        "j_asi_c_" + side: calf,
        "j_asi_d_" + side: ankle,
        "j_asi_e_" + side: toe - tip,
        "iv_asi_oya_b_" + side: tip,
    }


def build():
    """Build the scene (replacing everything): the rig and a pair of heels with naive weights, in Weight
    Paint mode. Returns (rig, shoes); the left shoe's vertices come first (x > 0)."""
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    rig = build_rig()
    left, faces_left = shoe_parts(1.0)
    right, faces_right = shoe_parts(-1.0)
    co = np.concatenate((left, right))
    faces = faces_left + [tuple(k + len(left) for k in face) for face in faces_right]
    mesh = bpy.data.meshes.new("Shoes")
    mesh.from_pydata(co.tolist(), [], faces)
    shoes = bpy.data.objects.new("Shoes", mesh)
    bpy.context.scene.collection.objects.link(shoes)
    shoes.modifiers.new("Armature", 'ARMATURE').object = rig
    for side, rows in (("l", slice(0, len(left))), ("r", slice(len(left), len(co)))):
        mirrored = co[rows].copy()
        mirrored[:, 0] = np.abs(mirrored[:, 0])
        for name, values in naive_weights(mirrored, side).items():
            group = shoes.vertex_groups.get(name) or shoes.vertex_groups.new(name=name)
            start = rows.start
            for i, weight in enumerate(values.tolist()):
                if weight > 1e-4:
                    group.add([start + i], weight, 'REPLACE')
    rig.select_set(False)
    shoes.select_set(True)
    bpy.context.view_layer.objects.active = shoes
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    return rig, shoes


def rest_co(obj):
    return np.array([v.co[:] for v in obj.data.vertices])


def displayed(obj):
    bpy.context.view_layer.update()
    from magic_fit import painting
    return painting.read_geometry(obj, bpy.context.evaluated_depsgraph_get(), source='INDEXED').co


def by_name(obj):
    """{group name: (vertices,) weights}."""
    table = np.zeros((len(obj.data.vertices), len(obj.vertex_groups)))
    for vert in obj.data.vertices:
        for elem in vert.groups:
            table[vert.index, elem.group] = elem.weight
    return {group.name: table[:, group.index] for group in obj.vertex_groups}
