"""Write the resize bench's body file, and optionally a garment file of the clothing in the .blend.

Opens a .blend whose Auto Resize panel has a devkit body as the From Body (a YAB devkit body with Sizes), and writes
``<data>/body.npz``: the body's triangles, its shape at every size of the grid (Chest x Lavabod x Rue x Buff,
the rest of its settings as they are now), which size it shows now, the Resize settings, and a region per vertex
(from the bone it's weighted to most). With ``--clothing COLLECTION`` (a name or the start of one) the meshes
Resize would resize in that collection (hidden ones too) go to ``<data>/garments/<tag>.npz``, fitted to the size the body shows.

    blender -b --factory-startup --python tools/resize_bench/dump_body.py -- SCENE.blend DATA_DIR \
        [--clothing "XIV Instant Edit"] [--tag snapshot]

Never saves the .blend.
"""

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402

GRID = {"chest_size": (0.0, 1.0, 2.0), "lavabod": (0.0, 1.0), "rue": (0.0, 1.0), "buff": (0.0, 1.0)}
CHEST = {0.0: "Large", 1.0: "Medium", 2.0: "Small"}
# Coarse body regions by the bone a vertex is weighted to most (FFXIV bone names), first match wins.
# Legs before arms: toes are named like fingers (iv_asi_oya_a_l).
REGIONS = (("chest", ("mune",)), ("butt", ("shiri",)), ("hips", ("kosi",)), ("belly", ("fukubu", "sebo_a")),
           ("torso", ("sebo_b", "sebo_c")), ("neck", ("kubi", "kao")), ("legs", ("asi", "daitai", "hiza", "sune")),
           ("arms", ("ude", "sako", "hiji", "te_", "_te", "oya", "hito", "naka", "kusu", "ko_")))
# Vertex groups of deforming bones (the devkit also has groups its shape keys are limited to).
BONES = ("j_", "iv_", "ya_")


def size_name(values):
    name = CHEST[values["chest_size"]]
    for setting, word in (("lavabod", "Lavabod"), ("rue", "Rue"), ("buff", "Buff")):
        if values[setting]:
            name += " " + word
    return name


def region_of(group):
    for region, words in REGIONS:
        if any(word in group for word in words):
            return region
    return "other"


def body_regions(body):
    names = [g.name for g in body.vertex_groups]
    regions = [name for name, _words in REGIONS] + ["other"]
    group_region = np.array([regions.index(region_of(n)) for n in names] + [regions.index("other")])
    out = np.full(len(body.data.vertices), len(group_region) - 1, dtype=np.int64)
    for v in body.data.vertices:
        best, weight = -1, 0.0
        for g in v.groups:
            if g.weight > weight and g.group < len(names) and names[g.group].startswith(BONES):
                best, weight = g.group, g.weight
        out[v.index] = best
    return group_region[out], regions


def main():
    args = common.args_after_dashes()
    blend, data = args[0], args[1]
    clothing = args[args.index("--clothing") + 1] if "--clothing" in args else None
    tag = args[args.index("--tag") + 1] if "--tag" in args else "snapshot"
    common.load_addon()
    bpy.context.preferences.filepaths.use_scripts_auto_execute = True
    bpy.ops.wm.open_mainfile(filepath=blend, load_ui=False)
    from magic_fit import api, resizing, sizing
    brush = bpy.context.scene.body_fit_brush
    job = api.Job(brush)
    body = job.body
    settings, problems = sizing.body_settings(body)
    if problems:
        print("settings problems:", problems)
    now = {s.name: s.current for s in settings}
    groups = {}
    tris = resizing._tris(body.data)
    world = body.matrix_world
    source = resizing._transform(resizing.rest_shape(body, job.source_values(), groups), world)
    names, values, shapes = [], [], []
    now_index = -1
    for combo in itertools.product(*GRID.values()):
        asked = dict(zip(GRID, combo))
        size = sizing.apply_rules(settings, dict(asked), now)
        shape = resizing._transform(resizing.rest_shape(body, job.target_values(sizing.size_values(body, size)),
                                                        groups), world)
        names.append(size_name(asked))
        values.append([asked[k] for k in GRID])
        shapes.append(shape)
        if np.array_equal(shape, source) and now_index < 0:
            now_index = len(names) - 1
        print("size", names[-1], "moves up to {:.1f} mm".format(np.linalg.norm(shape - source, axis=1).max() * 1000))
    if now_index < 0:
        names.append("Now")
        values.append([now.get(k, 0.0) for k in GRID])
        shapes.append(source)
        now_index = len(names) - 1
    regions, region_names = body_regions(body)
    meta = {"blend": blend, "body": body.name, "grid": list(GRID), "now": names[now_index],
            "settings": {"radius": job.radius, "bridge": job.bridge, "rigid_size": job.rigid_size,
                         "keep_close": bool(job.keep_close), "match": job.match},
            "now_values": now}
    os.makedirs(data, exist_ok=True)
    np.savez_compressed(os.path.join(data, "body.npz"), tris=tris.astype(np.int32), names=np.array(names),
                        values=np.array(values), shapes=np.array(shapes), regions=regions.astype(np.int16),
                        region_names=np.array(region_names), meta=np.array(json.dumps(meta)))
    print("body: {:d} vertices, {:d} sizes, now {!r}".format(len(source), len(names), names[now_index]))
    if clothing:
        collection = next(c for c in bpy.data.collections if c.name.startswith(clothing))
        # Hidden meshes too: Instant Edit exports the skin and piercings it hides.
        objects = [obj for obj in collection.all_objects if obj.type == 'MESH']
        resizer = api.Resizer(objects, sizes=[{"chest_size": 2.0}])
        work = resizer._work
        meshes = []
        for obj, shape in zip(resizer.clothing, work.shapes):
            sizes, loops, material = common.polys_of(obj.data)
            rigid = getattr(getattr(obj, "magic_fit_transfer", None), "rigid", False)
            meshes.append(common.mesh_record(obj.name, shape, (sizes, loops), material,
                                             common.material_names(obj.data), rigid))
        path = os.path.join(data, "garments", tag + ".npz")
        common.save_garment(path, tag, meshes, names[now_index], "{:s}|{:s}".format(blend, collection.name))
        print("clothing:", [m["name"] for m in meshes], "skin:", [o.name for o in work.skin],
              "parts left out:", [o.name for o in resizer.parts])


main()
sys.stdout.flush()
os._exit(0)
