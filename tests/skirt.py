"""A test scene for Skirt weights: a skirt hanging from the hips of an FFXIV-like rig.

Rig: j_sebo_a (spine) and j_kosi (hips) at the hip height of the Yet Another Devkit skeleton, thighs
(j_asi_a/b), butt bones (iv_shiri), and the 18 skirt bones at the rest positions they have in the game's
skeleton: chains front / side / back, segments a to c, left and right, splaying outward as they hang.

Body: a closed vertical cylinder around the hips and legs (meters), weighted like a body: j_sebo_a at
the top, j_kosi at the hips, the thighs below them (split left / right across the middle), the calves
below the knees. Skirt: a tube that lies on the body 5 mm off from above the hips down to the thighs,
then flares out to a hem 13 cm off, with a split seam down the front. Its body weights are copied from
the body by the Weight Brushes.
"""

import math

import bpy
import numpy as np

from magic_fit import painting

HIP = (0.0, 0.0, 1.0366)
BODY_R = 0.12
SKIRT_TOP = 1.20
SKIRT_TIGHT = 0.80    # the skirt lies on the body down to here
SKIRT_HEM = 0.45
SKIRT_R = 0.125       # 5 mm off the body
HEM_R = 0.25
COLUMNS = 64          # around the skirt; column 0 (and its seam copy, column 64) at the front center
ROWS = 40

# Skirt bone heads (x for the left side; the right side mirrors it), and the last one's tail.
CHAINS = {
    "f": [(0.0742, -0.1064, 1.0380), (0.0742, -0.1546, 0.6456), (0.0742, -0.1685, 0.5320),
          (0.0742, -0.1825, 0.4185)],
    "s": [(0.1483, -0.0106, 1.0380), (0.1737, -0.0106, 0.8315), (0.2054, -0.0106, 0.5733),
          (0.2371, -0.0106, 0.3152)],
    "b": [(0.0742, 0.1064, 1.0380), (0.0742, 0.1280, 0.8625), (0.0742, 0.1635, 0.5733),
          (0.0742, 0.1990, 0.2842)],
}
BODY_BONES = ("j_sebo_a", "j_kosi", "j_asi_a_l", "j_asi_a_r", "j_asi_b_l", "j_asi_b_r")


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def skirt_bone_names():
    return [
        "j_sk_{:s}_{:s}_{:s}".format(position, segment, side)
        for position in CHAINS for side in "lr" for segment in "abc"
    ]


def build_rig(name="Rig", chains=CHAINS, sides="lr", body_bones=True):
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
        return item

    bone("j_kosi", HIP, (0.0, 0.0958, 1.0223))
    if body_bones:
        bone("j_sebo_a", HIP, (0.0, 0.0124, 1.1576))
        for side, sign in (("l", 1.0), ("r", -1.0)):
            bone("j_asi_a_" + side, (sign * 0.086, -0.017, 0.9223), (sign * 0.086, -0.0283, 0.5552), "j_kosi")
            bone("j_asi_b_" + side, (sign * 0.086, -0.0283, 0.5552), (sign * 0.086, -0.0283, 0.1), "j_asi_a_" + side)
            bone("iv_shiri_" + side, (sign * 0.0717, 0.0558, 0.9173), (sign * 0.0717, 0.0354, 0.8226), "j_kosi")
    for position, heads in chains.items():
        for side in sides:
            sign = 1.0 if side == "l" else -1.0
            parent = "j_kosi"
            for k, segment in enumerate("abc"[:len(heads) - 1]):
                head = (sign * heads[k][0], heads[k][1], heads[k][2])
                tail = (sign * heads[k + 1][0], heads[k + 1][1], heads[k + 1][2])
                name_k = "j_sk_{:s}_{:s}_{:s}".format(position, segment, side)
                bone(name_k, head, tail, parent)
                parent = name_k
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig


def tube(name, heights, radii, columns, seam=False, caps=False):
    """A tube around the Z axis: ring i at heights[i] with radius radii[i]. Column j sits at angle
    -90 + 360 j / columns degrees (so column 0 is at the front, and column j mirrors columns - j).
    With ``seam``, column 0 is duplicated as a last column that isn't connected to it."""
    count = columns + 1 if seam else columns
    angles = -0.5 * math.pi + 2.0 * math.pi * np.arange(count) / columns
    rings = len(heights)
    co = []
    for z, r in zip(heights, radii):
        for a in angles:
            co.append((r * math.cos(a), r * math.sin(a), z))
    faces = []
    for i in range(rings - 1):
        for j in range(columns):
            j1 = j + 1 if seam else (j + 1) % columns
            faces.append((i * count + j, i * count + j1, (i + 1) * count + j1, (i + 1) * count + j))
    if caps:
        top = len(co)
        co.append((0.0, 0.0, heights[0]))
        bottom = len(co)
        co.append((0.0, 0.0, heights[-1]))
        for j in range(columns):
            j1 = (j + 1) % columns
            faces.append((top, j1, j))
            faces.append((bottom, (rings - 1) * count + j, (rings - 1) * count + j1))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(co, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def body_weights(co):
    x, z = co[:, 0], co[:, 2]
    left = smoothstep(-0.03, 0.03, x)
    sebo = smoothstep(1.10, 1.16, z)
    thigh = smoothstep(0.96, 0.86, z)
    calf = smoothstep(0.60, 0.52, z)
    kosi = (1.0 - sebo) * (1.0 - thigh)
    legs = (1.0 - sebo) * thigh
    return {
        "j_sebo_a": sebo,
        "j_kosi": kosi,
        "j_asi_a_l": legs * (1.0 - calf) * left,
        "j_asi_a_r": legs * (1.0 - calf) * (1.0 - left),
        "j_asi_b_l": legs * calf * left,
        "j_asi_b_r": legs * calf * (1.0 - left),
    }


def skirt_radius(z):
    return SKIRT_R + (HEM_R - SKIRT_R) * np.clip((SKIRT_TIGHT - z) / (SKIRT_TIGHT - SKIRT_HEM), 0.0, 1.0)


def build():
    """Build the scene (replacing everything) and copy the body's weights onto the skirt, in Weight
    Paint mode. Returns (rig, body, skirt); skirt vertex (i, j) is ring i (from the top), column j."""
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    rig = build_rig()

    heights = np.linspace(1.30, 0.30, 51)
    body = tube("Body", heights, np.full(len(heights), BODY_R), 48, caps=True)
    co = np.array([v.co[:] for v in body.data.vertices])
    for name, values in body_weights(co).items():
        group = body.vertex_groups.new(name=name)
        for i, weight in enumerate(values.tolist()):
            if weight > 1e-4:
                group.add([i], weight, 'REPLACE')
    body.modifiers.new("Armature", 'ARMATURE').object = rig

    zs = np.linspace(SKIRT_TOP, SKIRT_HEM, ROWS)
    skirt = tube("Skirt", zs, skirt_radius(zs), COLUMNS, seam=True)
    skirt.modifiers.new("Armature", 'ARMATURE').object = rig
    for name in BODY_BONES:
        skirt.vertex_groups.new(name=name)
    rig.select_set(False)
    body.select_set(False)
    skirt.select_set(True)
    bpy.context.view_layer.objects.active = skirt
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    depsgraph = bpy.context.evaluated_depsgraph_get()
    sampler = painting.TargetSampler(body, depsgraph)
    copy = painting.CopyStroke(skirt, body, sampler, painting.read_geometry(skirt, depsgraph).co)
    count = len(skirt.data.vertices)
    copy.apply(list(range(count)), [1.0] * count)
    skirt.data.update()
    return rig, body, skirt


def index(ring, column):
    return ring * (COLUMNS + 1) + column


def rest_co(obj):
    return np.array([v.co[:] for v in obj.data.vertices])


def scale_thighs(rig, factor):
    for name in ("j_asi_a_l", "j_asi_a_r"):
        rig.pose.bones[name].scale = (factor, factor, factor)
    bpy.context.view_layer.update()


def displayed(obj):
    bpy.context.view_layer.update()
    return painting.read_geometry(obj, bpy.context.evaluated_depsgraph_get(), source='INDEXED').co


def weights(obj):
    """(vertices, groups) table of ``obj``'s weights, and the group names."""
    table = np.zeros((len(obj.data.vertices), len(obj.vertex_groups)))
    for vert in obj.data.vertices:
        for elem in vert.groups:
            table[vert.index, elem.group] = elem.weight
    return table, [group.name for group in obj.vertex_groups]


def columns_of(names, prefix):
    return [k for k, name in enumerate(names) if name.startswith(prefix)]


def skirt_share(obj):
    """Per vertex: the skirt bones' share of its bone weights."""
    table, names = weights(obj)
    skirt = table[:, columns_of(names, "j_sk_")].sum(axis=1)
    return skirt / np.maximum(table.sum(axis=1), 1e-12)
