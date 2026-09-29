"""A test scene for Smooth: cloth sheets made of loose parts on a small rig (meters).

Rig: three bones, j_a, j_b and j_c, standing side by side.

Sheet: a row of rectangular parts of cloth, each a grid of quads, `EDGE` apart from its neighbour's edge
by a gap (or, with a gap of 0, sharing its seam with split vertices at the same spot). By default the first
part is weighted 100 % to j_a and the second 100 % to j_b, so where they meet is a hard step.
"""

import bpy
import numpy as np

BONES = ("j_a", "j_b", "j_c")
EDGE = 0.01
HEIGHT = 0.1


def build_rig(name="Rig"):
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    rig = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for i, bone_name in enumerate(BONES):
        bone = rig.data.edit_bones.new(bone_name)
        bone.head = (0.1 * i, 0.0, -0.5)
        bone.tail = (0.1 * i, 0.0, -0.4)
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig


def grid(x0, x1, height, edge):
    """(co, quads) of a grid of quads from x0 to x1 (and 0 to height in y), cells of about ``edge``."""
    nx = max(1, round((x1 - x0) / edge))
    ny = max(1, round(height / edge))
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    co = np.array([(x, y, 0.0) for y in ys for x in xs])
    width = nx + 1
    quads = [(j * width + i, j * width + i + 1, (j + 1) * width + i + 1, (j + 1) * width + i)
             for j in range(ny) for i in range(nx)]
    return co, quads


def build(parts=((0.0, 0.1), (0.1005, 0.2005)), edge=EDGE, height=HEIGHT, weights=None, armature=True,
          groups=BONES, name="Sheet"):
    """Build the scene (replacing everything): the rig and a sheet with a part for each (x0, x1) of
    ``parts``. ``weights(part, co)`` gives the weights of a part's vertices as {group name: (n,) array};
    the default is j_a on the first part, j_b on the others. Returns (rig, sheet, part of each vertex)."""
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    rig = build_rig()
    co, faces, owner = [], [], []
    for number, (x0, x1) in enumerate(parts):
        part_co, quads = grid(x0, x1, height, edge)
        offset = sum(len(c) for c in co)
        faces += [tuple(k + offset for k in quad) for quad in quads]
        co.append(part_co)
        owner += [number] * len(part_co)
    co = np.concatenate(co)
    owner = np.array(owner)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(co.tolist(), [], faces)
    sheet = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(sheet)
    if armature:
        sheet.modifiers.new("Armature", 'ARMATURE').object = rig
    for group in groups:
        sheet.vertex_groups.new(name=group)
    if weights is None:
        def weights(part, part_co):
            return {"j_a" if part == 0 else "j_b": np.ones(len(part_co))}
    for number in range(len(parts)):
        rows = np.flatnonzero(owner == number)
        for group, values in weights(number, co[rows]).items():
            vertex_group = sheet.vertex_groups[group]
            for value in np.unique(values[values > 1e-4]):
                vertex_group.add(rows[values == value].tolist(), float(value), 'REPLACE')
    rig.select_set(False)
    sheet.select_set(True)
    bpy.context.view_layer.objects.active = sheet
    bpy.context.view_layer.update()
    return rig, sheet, owner


def rest_co(obj):
    return np.array([v.co[:] for v in obj.data.vertices])


def by_name(obj):
    """{group name: (vertices,) weights}."""
    table = np.zeros((len(obj.data.vertices), len(obj.vertex_groups)))
    for vert in obj.data.vertices:
        for elem in vert.groups:
            table[vert.index, elem.group] = elem.weight
    return {group.name: table[:, group.index] for group in obj.vertex_groups}


def apply_all(stroke, obj, factor=1.0):
    count = len(obj.data.vertices)
    stroke.apply(list(range(count)), [factor] * count)
    obj.data.update()
