"""Resize garments from the bench's data with a Magic Fit checkout, and measure them.

    blender -b --factory-startup --python tools/resize_bench/bench.py -- DATA_DIR OUT_DIR [options]

Options:
    --checkout PATH   the Magic Fit to measure (default: this repository)
    --targets all|near
                      every other size of the body file, or "near" (default): the size with Small breasts and
                      Omoi (Large Lavabod), both keeping the From size's Rue and Buff
    --round-trip near|all|none
                      sizes to resize back from to the From size (default near: those of "near")
    --only TAG,...    garments to run
    --shard I/N       run every Nth garment starting at the Ith (0-based), for parallel runs
    --pv              also write per-vertex arrays (OUT_DIR/pv/<tag>.npz)

Each garment's meshes are made as Blender objects and set up by the checkout's own api._Sizes (skin, rigid
parts, Keep Close's edges, as the add-on does), then resizing.ShapeChange moves them to each target size. The
results go to OUT_DIR/<tag>.json; metrics.py says what is measured.
"""

import inspect
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
import metrics  # noqa: E402

# Picking a garment's From size: points within this of a size's body count, a sample of at most this many.
FIT_REACH = 0.03
FIT_SAMPLE = 3000
# ...over body vertices the sizes move more than this apart, if at least this many.
FIT_SPREAD = 0.003
FIT_MIN = 50


class Options:
    def __init__(self, args):
        self.data, self.out = args[0], args[1]
        named = {}
        flags = set()
        k = 2
        while k < len(args):
            if args[k] == "--pv":
                flags.add("pv")
                k += 1
            else:
                named[args[k].lstrip("-")] = args[k + 1]
                k += 2
        self.checkout = named.get("checkout")
        self.targets = named.get("targets", "near")
        self.round_trip = named.get("round-trip", "near")
        self.only = set(named["only"].split(",")) if "only" in named else None
        shard = named.get("shard", "0/1").split("/")
        self.shard, self.shards = int(shard[0]), int(shard[1])
        self.pv = "pv" in flags


class Body:
    def __init__(self, path):
        d = np.load(path)
        self.tris = d["tris"].astype(np.int64)
        self.names = [str(n) for n in d["names"]]
        self.values = d["values"]
        self.shapes = d["shapes"]
        self.regions = d["regions"].astype(np.int64)
        self.region_names = [str(n) for n in d["region_names"]]
        self.meta = json.loads(str(d["meta"]))
        self.grid = self.meta["grid"]
        self._surfaces = {}
        self._details = {}

    def surface(self, s):
        if s not in self._surfaces:
            self._surfaces[s] = metrics.Surface(self.shapes[s], self.tris)
        return self._surfaces[s]

    def detail(self, s):
        """(Detail of the welded body at size s, welded index per vertex, first vertex per welded one)."""
        if s not in self._details:
            index, first, tris = metrics.weld(self.shapes[s], self.tris)
            self._details[s] = (metrics.Detail(self.shapes[s][first], tris), index, first)
        return self._details[s]

    def find(self, values):
        for k, v in enumerate(self.values):
            if np.allclose(v, values):
                return k
        return None

    def near(self, s):
        """Indices of the Small and Omoi sizes for From size s (Large when s is Small, Medium Lavabod when s is
        Omoi)."""
        v = dict(zip(self.grid, self.values[s]))
        out = []
        for change, fallback in (({"chest_size": 2.0}, {"chest_size": 0.0}),
                                 ({"chest_size": 0.0, "lavabod": 1.0}, {"chest_size": 1.0, "lavabod": 1.0})):
            for attempt in (change, fallback):
                t = self.find([dict(v, **attempt)[g] for g in self.grid])
                if t is not None and t != s and t not in out:
                    out.append(t)
                    break
        return out


def pick_from(body, points):
    """The size whose body ``points`` fit best: the smallest median distance of the points near it, with the share
    of points more than 1 mm inside weighing in. Only points over the parts of the body the sizes change count
    (else a loose sweater fits every chest alike); the size the body showed when none are. Returns (index,
    {name: (median mm, inside share)})."""
    now = body.names.index(body.meta["now"])
    spread = np.linalg.norm(body.shapes.max(axis=0) - body.shapes.min(axis=0), axis=1)
    _h, face, bary, _beyond, found = body.surface(now).nearest(points)
    corner = body.tris[np.maximum(face, 0), np.argmax(bary, axis=1)]
    points = points[found & (spread[corner] > FIT_SPREAD)]
    if len(points) < FIT_MIN:
        return now, {}
    rng = np.random.default_rng(0)
    sample = points if len(points) <= FIT_SAMPLE else points[rng.choice(len(points), FIT_SAMPLE, replace=False)]
    scores, fits = [], {}
    for s, name in enumerate(body.names):
        h, _f, _b, beyond, found = body.surface(s).nearest(sample)
        ok = found & ~beyond & (np.abs(h) < FIT_REACH)
        if ok.sum() < 10:
            scores.append(np.inf)
            continue
        median = float(np.median(np.abs(h[ok])))
        inside = float((h[ok] < -0.001).mean())
        fits[name] = (round(median * 1000, 2), round(inside, 3))
        scores.append(median + 0.01 * inside)
    return int(np.argmin(scores)), fits


class Job:
    """What api._Sizes reads of a Resize job."""

    def __init__(self, body, settings):
        self.body = body
        self.rigid_size = settings["rigid_size"]
        self.keep_close = settings["keep_close"]

    def source_values(self):
        return {}


def apply_inputs(apply, work):
    """What api._Sizes passes ShapeChange.apply besides the points, as far as the checkout's apply takes it."""
    taken = inspect.signature(apply).parameters
    return {name: getattr(work, name) for name in ("rigid", "edges", "bound", "faces")
            if name in taken and hasattr(work, name)}


def run_garment(path, body, options, magic_fit):
    from magic_fit import api, resizing
    meta, records = common.load_garment(path)
    tag = meta["tag"]
    settings = body.meta["settings"]
    started = time.perf_counter()
    points = np.concatenate([r["verts"] for r in records])
    fits = None
    if meta["from_size"] == "auto":
        s, fits = pick_from(body, points)
    else:
        s = body.names.index(meta["from_size"])
    targets = [t for t in range(len(body.names)) if t != s] if options.targets == "all" else body.near(s)
    back = {"none": set(), "all": set(targets), "near": set(body.near(s))}[options.round_trip]
    result = {"tag": tag, "source": meta["source"], "from": body.names[s], "fits": fits, "settings": settings,
              "meshes": [r["name"] for r in records], "vertices": int(len(points)), "targets": {}}
    collection = bpy.data.collections.new("bench")
    bpy.context.scene.collection.children.link(collection)
    body_obj = common.make_object(common.mesh_record("Body", body.shapes[s], (np.full(len(body.tris), 3, np.int32),
                                                                             body.tris.reshape(-1))), collection)
    objects = [common.make_object(r, collection) for r in records]
    try:
        for obj, r in zip(objects, records):
            if len(obj.data.vertices) != len(r["verts"]):
                raise RuntimeError("{!r} lost vertices when made".format(r["name"]))
        t0 = time.perf_counter()
        work = api._Sizes(objects, Job(body_obj, settings), share=True)
        result["setup_s"] = round(time.perf_counter() - t0, 3)
        # Checkouts before skin binding have neither skin nor bound.
        result["skin"] = [o.name for o in getattr(work, "skin", [])]
        result["apply_inputs"] = sorted(apply_inputs(resizing.ShapeChange.apply, work))
        result["rigid_parts"] = int(len(np.unique(work.rigid[work.rigid >= 0])))
        tris = np.concatenate([resizing._tris(o.data) + first for o, first in
                               zip(objects, np.cumsum([0] + [len(o.data.vertices) for o in objects])[:-1])])
        t0 = time.perf_counter()
        # Skin is measured apart, told by its material the same way for every checkout (whatever the checkout
        # makes of it), so two checkouts are measured on the same vertices.
        moving = np.concatenate([np.full(len(r["verts"]), not metrics.is_skin_mesh(r)) for r in records])
        result["skin_measured_apart"] = [r["name"] for r in records if metrics.is_skin_mesh(r)]
        garment = metrics.Garment(work.points, tris, moving, body.surface(s), body.regions, body.region_names,
                                  work.rigid)
        inputs = apply_inputs(resizing.ShapeChange.apply, work)
        result["measure_setup_s"] = round(time.perf_counter() - t0, 3)
        result["layer_pairs"] = int(len(garment.layers[0]))
        detail_from, body_index, body_first = body.detail(s)
        pv_all = {}
        for t in targets:
            name = body.names[t]
            t0 = time.perf_counter()
            change = resizing.ShapeChange(work.source, body.shapes[t], body.tris, radius=settings["radius"],
                                          bridge=settings["bridge"], shared=work.shared)
            moved = change.apply(work.points, **inputs)
            row = {"seconds": round(time.perf_counter() - t0, 3), "steps": int(change.steps),
                   "body_moves_mm": round(float(change.largest) * 1000, 2)}
            returned = None
            if t in back:
                t1 = time.perf_counter()
                undo = resizing.ShapeChange(body.shapes[t], work.source, body.tris, radius=settings["radius"],
                                            bridge=settings["bridge"])
                returned = undo.apply(moved, **inputs)
                row["back_seconds"] = round(time.perf_counter() - t1, 3)
            pv = {} if options.pv else None
            t1 = time.perf_counter()
            body_detail = (detail_from, body.shapes[s][body_first], body.shapes[t][body_first], body_index)
            row.update(garment.measure(moved, body.surface(t), body_detail, back=returned, pv=pv))
            row["measure_seconds"] = round(time.perf_counter() - t1, 3)
            result["targets"][name] = row
            if pv is not None:
                for key, value in pv.items():
                    pv_all["{:s}|{:s}".format(name, key)] = value
            print("  {:s} -> {:s}: {:.2f} s, {:d} steps".format(tag, name, row["seconds"], row["steps"]), flush=True)
        if options.pv:
            os.makedirs(os.path.join(options.out, "pv"), exist_ok=True)
            np.savez_compressed(os.path.join(options.out, "pv", tag + ".npz"), **pv_all)
    finally:
        common.remove_objects(objects + [body_obj])
        bpy.data.collections.remove(collection)
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def main():
    options = Options(common.args_after_dashes())
    magic_fit = common.load_addon(options.checkout)
    body = Body(os.path.join(options.data, "body.npz"))
    folder = os.path.join(options.data, "garments")
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz"))
    if options.only is not None:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in options.only]
    paths = paths[options.shard::options.shards]
    os.makedirs(options.out, exist_ok=True)
    failed = 0
    for path in paths:
        tag = os.path.splitext(os.path.basename(path))[0]
        print("garment", tag, flush=True)
        try:
            result = run_garment(path, body, options, magic_fit)
        except Exception:
            failed += 1
            result = {"tag": tag, "error": traceback.format_exc()}
            print(result["error"], flush=True)
        with open(os.path.join(options.out, tag + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=1)
    print("done: {:d} garments, {:d} failed".format(len(paths), failed), flush=True)
    return failed


code = main()
sys.stdout.flush()
os._exit(1 if code else 0)
