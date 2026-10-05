"""A test scene for Resize: a chest whose breasts shrink between two shapes, and clothing fitted to the
larger ones.

Body: the front of a torso with two breasts (meters, y < 0 is the front). Its basis has large breasts
(spheres of 6.5 cm blended into the chest, like `cleavage`, with a cleavage 1 cm wide between them);
its shape key "Small" pulls each breast toward its center to 45 % of its size, fading out toward its
base, like a devkit's size key. The breasts' fronts move back by 3.5 cm while the chest around them
stays put. Its UV map is the chest seen from the front; `coarse_body` makes another body with other
vertices but the same UV layout, like another body mod's, or without a UV map, like another game's.

Garments, all fitted to the large breasts:
- "Suit": skin tight, 3 mm off the chest (along its normals), following it into the cleavage.
- "Bridge": a sheet stretched straight across the chest 3 mm in front of it, spanning the cleavage.
- "Strap": two layers 3 and 7 mm off the left breast, split into two objects along x = -0.06 with
  their edge vertices at the same spots.
- "Pendant": a 2 cm cube hanging 6 cm in front of the cleavage.

`devkit_body` builds the same scene with the body's keys driven like Yet Another Devkit's Mannequin, by
stand-ins for the devkit's settings.
"""

import bpy
import numpy as np
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, PointerProperty

RADIUS = 0.065
CENTERS = ((-0.07, -0.075, 0.0), (0.07, -0.075, 0.0))  # the torso front is at y = -0.09
SHRINK = 0.45
TORSO_FRONT = -0.09
XS = np.linspace(-0.17, 0.17, 69)
ZS = np.linspace(-0.14, 0.12, 53)
# The strap's two objects: their columns of vertices (the last of the first is the first of the second).
STRAP_XS = (np.linspace(-0.11, -0.06, 11), np.linspace(-0.06, -0.03, 7))
STRAP_ZS = np.arange(-0.05, 0.05 + 1e-9, 0.004)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def body_front(x, z):
    """Front surface y of the chest with the large breasts (more negative = further forward)."""
    y = np.full(np.broadcast(x, z).shape, TORSO_FRONT)
    for center_x, center_y, center_z in CENTERS:
        front = center_y - np.sqrt(np.clip(RADIUS ** 2 - (x - center_x) ** 2 - (z - center_z) ** 2, 0.0, None))
        h = np.clip(0.5 + 0.5 * (y - front) / 0.01, 0.0, 1.0)  # smooth minimum: the breast blends in
        y = y * (1 - h) + front * h - 0.01 * h * (1 - h)
    return y


def shrink(co):
    """The chest with small breasts: each point pulled toward its breast's center, less toward the base."""
    result = co.copy()
    for center in CENTERS:
        offset = np.asarray(center) - co
        # All the way over most of the breast (seen from the front), fading out over its base.
        weight = smoothstep(RADIUS + 0.01, RADIUS - 0.02, np.hypot(offset[:, 0], offset[:, 2]))
        result += (1.0 - SHRINK) * weight[:, None] * offset
    return result


def front_normals(x, z, step=1e-5):
    """Normals of the chest with large breasts (pointing forward) at points (x, z) of its front."""
    dx = (body_front(x + step, z) - body_front(x - step, z)) / (2.0 * step)
    dz = (body_front(x, z + step) - body_front(x, z - step)) / (2.0 * step)
    normals = np.stack((dx, -np.ones_like(dx), dz), axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def envelope(y_row):
    """Front convex envelope of a profile y(x) sampled evenly: a sheet stretched across its front."""
    hull = []
    for i in range(len(y_row)):
        while len(hull) >= 2:
            i0, i1 = hull[-2], hull[-1]
            if (i1 - i0) * (y_row[i] - y_row[i0]) - (i - i0) * (y_row[i1] - y_row[i0]) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(np.arange(len(y_row)), hull, y_row[hull])


def grid(xs, zs, front):
    """Vertices and quads of a sheet over ``xs`` x ``zs`` at y = front(X, Z); vertex (i, j) is i * len(zs) + j."""
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    co = np.stack((X.ravel(), front(X, Z).ravel(), Z.ravel()), axis=1)
    index = np.arange(len(xs) * len(zs)).reshape(len(xs), len(zs))
    quads = np.stack((index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(),
                      index[:-1, 1:].ravel()), axis=1)
    return co, quads


def mesh_object(name, co, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(np.asarray(co).tolist(), [], np.asarray(faces).tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def body_shapes(xs=XS, zs=ZS):
    """The body's vertices with large and with small breasts, and its quads."""
    large, quads = grid(xs, zs, body_front)
    return large, shrink(large), quads


def body_uvs(co):
    """Where points of the chest are in its UV map: the chest seen from the front, filling the square."""
    return np.column_stack(((co[:, 0] - XS[0]) / (XS[-1] - XS[0]), (co[:, 2] - ZS[0]) / (ZS[-1] - ZS[0])))


def add_uvs(obj, uv):
    """Give ``obj`` a UV map with ``uv`` (a spot per vertex) at every corner."""
    mesh = obj.data
    layer = mesh.uv_layers.new(name="UVMap")
    vertex = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", vertex)
    layer.data.foreach_set("uv", np.asarray(uv)[vertex].astype(np.float32).ravel())


def body_object(name, large, small, quads, uv):
    """A body mesh with UVs ``uv`` (None for no UV map) and a shape key "Small" (off)."""
    obj = mesh_object(name, large, quads)
    if uv is not None:
        add_uvs(obj, uv)
    obj.shape_key_add(name="Basis", from_mix=False)
    key = obj.shape_key_add(name="Small", from_mix=False)
    key.data.foreach_set("co", np.asarray(small).astype(np.float32).ravel())
    key.value = 0.0  # new keys start fully on
    return obj


# The coarse body's columns and rows of vertices.
COARSE_XS = np.linspace(XS[0], XS[-1], 47)
COARSE_ZS = np.linspace(ZS[0], ZS[-1], 37)


def coarse_body(name="Coarse Body", uv=body_uvs):
    """The body again with other, fewer vertices (like another body mod) and the same UV layout; with ``uv``
    None, without a UV map (like a body from another game)."""
    large, small, quads = body_shapes(COARSE_XS, COARSE_ZS)
    return body_object(name, large, small, quads, None if uv is None else uv(large))


def offset_sheet(xs, zs, offset):
    """A sheet ``offset`` off the chest with large breasts, along its normals."""
    co, quads = grid(xs, zs, body_front)
    return co + offset * front_normals(co[:, 0], co[:, 2]), quads


def cube(center, size):
    offsets = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.asarray(center) + 0.5 * size * offsets, faces


def strap_layers(xs, zs):
    """Two layers of a strap over the left breast, 3 and 7 mm out; vertex i of the outer layer is
    straight over vertex i of the inner one."""
    surface, quads = grid(xs, zs, body_front)
    normals = front_normals(surface[:, 0], surface[:, 2])
    parts = [(surface + offset * normals, quads) for offset in (0.003, 0.007)]
    count = len(parts[0][0])
    return np.concatenate([parts[0][0], parts[1][0]]), np.concatenate([parts[0][1], parts[1][1] + count])


def build():
    """Build the scene, replacing everything. Returns (body, garments by name)."""
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    large, small, quads = body_shapes()
    body = body_object("Body", large, small, quads, body_uvs(large))

    garments = {}
    xs = np.arange(-0.15, 0.15 + 1e-9, 0.004)
    zs = np.arange(-0.10, 0.09 + 1e-9, 0.004)
    garments["Suit"] = mesh_object("Suit", *offset_sheet(xs, zs, 0.003))

    def bridge_front(X, Z):
        Y = np.empty_like(X)
        for j in range(X.shape[1]):
            Y[:, j] = envelope(body_front(X[:, j], Z[:, j])) - 0.003
        return Y
    garments["Bridge"] = mesh_object("Bridge", *grid(xs, zs, bridge_front))

    # The strap, split into two objects along x = -0.06; both have a column of vertices there.
    garments["Strap A"] = mesh_object("Strap A", *strap_layers(STRAP_XS[0], STRAP_ZS))
    garments["Strap B"] = mesh_object("Strap B", *strap_layers(STRAP_XS[1], STRAP_ZS))

    front = float(body_front(np.array(0.0), np.array(0.0)))
    garments["Pendant"] = mesh_object("Pendant", *cube((0.0, front - 0.06 - 0.01, -0.03), 0.02))
    for obj in garments.values():
        obj.select_set(True)
    body.select_set(False)
    bpy.context.view_layer.objects.active = garments["Suit"]
    return body, garments


def positions(obj, key=None):
    """World space positions of ``obj``'s vertices: its mesh, or its shape key ``key``."""
    if key is None:
        co = np.empty(len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get("co", co)
    else:
        block = obj.data.shape_keys.key_blocks[key]
        co = np.empty(len(block.data) * 3)
        block.data.foreach_get("co", co)
    m = np.array(obj.matrix_world)
    return co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def triangles(obj):
    obj.data.calc_loop_triangles()
    tris = np.empty(len(obj.data.loop_triangles) * 3, dtype=np.int64)
    obj.data.loop_triangles.foreach_get("vertices", tris)
    return tris.reshape(-1, 3)


def edges(obj):
    e = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", e)
    return e.reshape(-1, 2)


def flipped(before, after, tris):
    """Triangles whose normal turned around."""
    def normals(co):
        a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
        return np.cross(b - a, c - a)
    return int((np.einsum("ij,ij->i", normals(before), normals(after)) < 0.0).sum())


def stretch(before, after, edge_list):
    """Largest factor by which an edge got longer or shorter."""
    l0 = np.linalg.norm(before[edge_list[:, 0]] - before[edge_list[:, 1]], axis=1)
    l1 = np.linalg.norm(after[edge_list[:, 0]] - after[edge_list[:, 1]], axis=1)
    ratio = l1 / np.maximum(l0, 1e-12)
    return float(max(ratio.max(), 1.0 / max(ratio.min(), 1e-12)))


# -----------------------------------------------------------------------------
# A Yet Another Devkit body

# Stand-ins for the devkit's settings (its script registers Scene.ya_devkit_props), with the same names and
# numbering. Its chest sizes are listed by a function, so a driver reads them by number from their names.
_CHESTS = [(str(index), name, "") for index, name in enumerate(("Large", "Medium", "Small", "Masc"))]


def _chest_items(_self, _context):
    return _CHESTS


class DevkitKeptKey(bpy.types.PropertyGroup):
    value: FloatProperty()


class DevkitState(bpy.types.PropertyGroup):
    chest_size: EnumProperty(items=_chest_items)
    leg_size: EnumProperty(items=[(str(index), name, "") for index, name in enumerate(
        ("Melon", "Skull", "Yanilla", "Masc", "Lavabod", "Mini"))])
    rue: BoolProperty()
    buff: BoolProperty()
    lavabod: BoolProperty()
    alt_hips: BoolProperty()
    yab_keys: CollectionProperty(type=DevkitKeptKey)
    lava_keys: CollectionProperty(type=DevkitKeptKey)


class DevkitProps(bpy.types.PropertyGroup):
    mannequin_state: PointerProperty(type=DevkitState)


def register_devkit():
    if hasattr(bpy.types.Scene, "ya_devkit_props"):
        return
    for cls in (DevkitKeptKey, DevkitState, DevkitProps):
        bpy.utils.register_class(cls)
    bpy.types.Scene.ya_devkit_props = PointerProperty(type=DevkitProps)


def drive(keys, name, prop, expression, variables, kind='SCRIPTED'):
    """Drive ``prop`` of shape key ``name`` of ``keys`` (a Key) with ``expression`` over ``variables``: (name,
    devkit setting) pairs, or (name, key) to read another key's value."""
    driver = keys.key_blocks[name].driver_add(prop).driver
    driver.type = kind
    driver.expression = expression
    for var_name, source in variables:
        var = driver.variables.new()
        var.name = var_name
        var.type = 'SINGLE_PROP'
        target = var.targets[0]
        if source in keys.key_blocks:
            target.id_type = 'KEY'
            target.id = keys
            target.data_path = 'key_blocks["{:s}"].value'.format(source)
        else:
            target.id_type = 'SCENE'
            target.id = bpy.context.scene
            target.data_path = "ya_devkit_props.mannequin_state." + source


def devkit_body():
    """Build the scene with the body's keys driven by the devkit's settings, all off (Large, Melon): MEDIUM
    (the "Small" breasts) by the chest size, which mutes the Large sub-key "- Squeeze" (at 0.3) for other
    sizes; Rue, Buff, Lavabod and Skull Crushers; the corrections Rue/Buff and Hip Dips (for YAB); and
    Rue/Follow, which follows the Rue key. Returns (body, garments)."""
    register_devkit()
    body, garments = build()
    state = bpy.context.scene.ya_devkit_props.mannequin_state
    for prop in ("chest_size", "leg_size", "rue", "buff", "lavabod", "alt_hips"):
        state.property_unset(prop)
    state.yab_keys.clear()
    state.lava_keys.clear()
    large, small, _quads = body_shapes()
    x, z = large[:, :1], large[:, 2:]
    front = np.array((0.0, -1.0, 0.0))
    shapes = {
        "MEDIUM": small,
        "- Squeeze": large + 0.2 * (large - small),
        "Rue": large + 0.006 * np.clip(-z / 0.1, 0.0, 1.0) * front,
        "Buff": large + 0.02 * x * (1.0, 0.0, 0.0),
        "Lavabod": large + 0.004 * np.clip(z / 0.1, 0.0, 1.0) * front,
        "Rue/Buff": large + (0.0, 0.0, 0.001),
        "Skull Crushers": large + 0.003 * np.clip(-z / 0.1, 0.0, 1.0) * np.abs(x) / 0.17 * front,
        "Hip Dips (for YAB)": large - 0.002 * np.clip(-z / 0.1, 0.0, 1.0) * front,
        "Rue/Follow": large + (0.0, 0.0, -0.001),
    }
    for name, co in shapes.items():
        block = body.shape_key_add(name=name, from_mix=False)
        block.data.foreach_set("co", np.asarray(co).astype(np.float32).ravel())
        block.value = 0.0
    keys = body.data.shape_keys
    keys.key_blocks["- Squeeze"].value = 0.3
    drive(keys, "MEDIUM", 'value', "size == 1 and lava == 0", (("size", "chest_size"), ("lava", "lavabod")))
    drive(keys, "- Squeeze", 'mute', "size != 0", (("size", "chest_size"),))
    for name, setting in (("Rue", "rue"), ("Buff", "buff"), ("Lavabod", "lavabod")):
        drive(keys, name, 'value', "option == 1", (("option", setting),))
    drive(keys, "Rue/Buff", 'value', "buff == 1 and rue == 1", (("buff", "buff"), ("rue", "rue")))
    drive(keys, "Skull Crushers", 'value', "legs == 1", (("legs", "leg_size"),))
    drive(keys, "Hip Dips (for YAB)", 'value', "legs <= 2 and alt_hips == 1 and rue == 0",
          (("legs", "leg_size"), ("alt_hips", "alt_hips"), ("rue", "rue")))
    drive(keys, "Rue/Follow", 'value', "", (("value", "Rue"),), kind='AVERAGE')
    return body, garments
