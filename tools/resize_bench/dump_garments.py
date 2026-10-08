"""Write garment files for the resize bench from game models and .blend objects.

    blender -b --factory-startup --python tools/resize_bench/dump_garments.py -- ITEMS.tsv DATA_DIR \
        [--blend FILE.blend] [--only TAG,...]

ITEMS.tsv has a line per garment, tab separated: kind, source, tag.
    mdl     path of a .mdl: each submesh becomes a mesh (skin ones too, with their materials, so Resize can
            find them)
    blend   name of an object in --blend
    from    'path.blend|object name'
Garments go to DATA_DIR/garments/<tag>.npz with From size "auto": the bench resizes each from the body size it
fits best. Never saves a .blend.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402


def mdl_meshes(path):
    from magic_fit import facedata
    with open(path, "rb") as f:
        parts = facedata.model_parts(f.read())
    return [common.mesh_record(part.name, part.points, (np.full(len(part.tris), 3, np.int32), part.tris.reshape(-1)),
                               None, [part.material or ""]) for part in parts if len(part.tris)]


def object_meshes(path, name):
    from magic_fit import resizing
    with bpy.data.libraries.load(path, link=False) as (_source, data):
        data.objects = [name]
    obj = data.objects[0]
    if obj is None or obj.type != 'MESH':
        raise RuntimeError("{!r} isn't a mesh in {!r}".format(name, path))
    co = resizing._transform(resizing.rest_shape(obj), obj.matrix_world)
    sizes, loops, material = common.polys_of(obj.data)
    rigid = getattr(getattr(obj, "magic_fit_transfer", None), "rigid", False)
    return [common.mesh_record(obj.name, co, (sizes, loops), material, common.material_names(obj.data), rigid)]


def main():
    args = common.args_after_dashes()
    items, data = args[0], args[1]
    blend = args[args.index("--blend") + 1] if "--blend" in args else None
    only = set(args[args.index("--only") + 1].split(",")) if "--only" in args else None
    common.load_addon()
    for line in open(items, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        kind, source, tag = line.rstrip("\n").split("\t")[:3]
        if only is not None and tag not in only:
            continue
        try:
            if kind == "mdl":
                meshes = mdl_meshes(source)
            elif kind == "blend":
                meshes = object_meshes(blend, source)
            elif kind == "from":
                path, name = source.split("|")
                meshes = object_meshes(path, name)
            else:
                raise RuntimeError("unknown kind {!r}".format(kind))
        except Exception as error:
            print("SKIPPED {:s}: {!s}".format(tag, error), flush=True)
            continue
        common.save_garment(os.path.join(data, "garments", tag + ".npz"), tag, meshes, "auto", source)
        print("{:s}: {:d} meshes, {:d} vertices".format(tag, len(meshes), sum(len(m["verts"]) for m in meshes)),
              flush=True)


main()
sys.stdout.flush()
os._exit(0)
