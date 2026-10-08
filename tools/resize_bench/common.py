"""What the resize bench's scripts share: loading Magic Fit from a checkout, and garments saved as npz.

A garment file holds the meshes of one garment as they fit the body at its From size, in world space, with
their faces and materials, so the bench can make them again as Blender objects and resize them the way the
add-on does:

    verts (n x 3), vert_start (per mesh, and the end), poly_sizes, poly_loops (indices into verts), poly_start
    (per mesh, and the end), poly_material (per face), meta (JSON: tag, from_size, source, meshes: [{name, materials, rigid}])

``from_size`` is a size name of the body file, or "auto" for the size it fits best.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def args_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def load_addon(checkout=None):
    """Import and register Magic Fit from ``checkout`` (default: this repository). Returns the package."""
    import bpy
    checkout = os.path.abspath(checkout or ROOT)
    sys.path.insert(0, checkout)
    import magic_fit
    if os.path.dirname(os.path.dirname(os.path.abspath(magic_fit.__file__))) != checkout:
        raise RuntimeError("magic_fit was imported from {!r}, not {!r}".format(magic_fit.__file__, checkout))
    magic_fit.register()
    bpy.context.preferences.addons.new().module = "magic_fit"
    return magic_fit


def mesh_record(name, verts, polys, poly_material=None, materials=(), rigid=False):
    """A mesh of a garment: ``verts`` (world space), ``polys`` (lists of vertex indices) or (sizes, loops)."""
    if isinstance(polys, tuple):
        sizes, loops = polys
    else:
        sizes = np.array([len(p) for p in polys], dtype=np.int32)
        loops = np.concatenate([np.asarray(p, dtype=np.int32) for p in polys]) if len(polys) else np.zeros(0, np.int32)
    sizes = np.asarray(sizes, dtype=np.int32)
    if poly_material is None:
        poly_material = np.zeros(len(sizes), dtype=np.int16)
    return {"name": name, "verts": np.asarray(verts, dtype=np.float64).reshape(-1, 3), "sizes": sizes,
            "loops": np.asarray(loops, dtype=np.int32), "poly_material": np.asarray(poly_material, dtype=np.int16),
            "materials": list(materials), "rigid": bool(rigid)}


def save_garment(path, tag, meshes, from_size="auto", source=""):
    verts = np.concatenate([m["verts"] for m in meshes])
    vert_start = np.cumsum([0] + [len(m["verts"]) for m in meshes])
    poly_start = np.cumsum([0] + [len(m["sizes"]) for m in meshes])
    meta = {"tag": tag, "from_size": from_size, "source": source,
            "meshes": [{"name": m["name"], "materials": m["materials"], "rigid": m["rigid"]} for m in meshes]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, verts=verts, vert_start=vert_start, poly_sizes=np.concatenate([m["sizes"] for m in meshes]),
                        poly_loops=np.concatenate([m["loops"] + first for m, first in zip(meshes, vert_start)]),
                        poly_start=poly_start,
                        poly_material=np.concatenate([m["poly_material"] for m in meshes]),
                        meta=np.array(json.dumps(meta)))


def load_garment(path):
    """(meta, [mesh records]) of a garment file."""
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    loop_start = np.concatenate([[0], np.cumsum(d["poly_sizes"])])
    meshes = []
    for k, info in enumerate(meta["meshes"]):
        v0, v1 = d["vert_start"][k], d["vert_start"][k + 1]
        p0, p1 = d["poly_start"][k], d["poly_start"][k + 1]
        loops = d["poly_loops"][loop_start[p0]:loop_start[p1]] - v0
        meshes.append(mesh_record(info["name"], d["verts"][v0:v1], (d["poly_sizes"][p0:p1], loops),
                                  d["poly_material"][p0:p1], info["materials"], info["rigid"]))
    return meta, meshes


def make_object(record, collection=None):
    """A Blender mesh object (identity transform) made from a mesh record."""
    import bpy
    mesh = bpy.data.meshes.new(record["name"])
    verts, sizes, loops = record["verts"], record["sizes"], record["loops"]
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).reshape(-1))
    mesh.loops.add(len(loops))
    mesh.loops.foreach_set("vertex_index", loops)
    mesh.polygons.add(len(sizes))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32) if len(sizes) else np.zeros(0, np.int32)
    mesh.polygons.foreach_set("loop_start", starts)
    for name in record["materials"]:
        material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mesh.materials.append(material)
    if record["materials"]:
        mesh.polygons.foreach_set("material_index", record["poly_material"].astype(np.int32))
    mesh.update(calc_edges=True)
    mesh.validate(clean_customdata=False)
    obj = bpy.data.objects.new(record["name"], mesh)
    (collection or bpy.context.scene.collection).objects.link(obj)
    if record["rigid"] and hasattr(obj, "magic_fit_transfer"):
        obj.magic_fit_transfer.rigid = True
    return obj


def remove_objects(objects):
    import bpy
    for obj in objects:
        mesh = obj.data
        bpy.data.objects.remove(obj)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def polys_of(mesh):
    """(sizes, loops, material per face) of a Blender mesh."""
    sizes = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", sizes)
    loops = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loops)
    material = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", material)
    return sizes, loops, material.astype(np.int16)


def material_names(mesh):
    return [m.name if m is not None else "" for m in mesh.materials]
