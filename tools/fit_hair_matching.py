"""Fit the weights of Hair's skeleton score (SCORE_* and FULL_* in magic_fit/hairing.py) and of
its chain score (CHAIN_WEIGHTS).

Hair picks a hair skeleton by a score: standardised features of how the skeleton's bones fit the hair,
times weights. This measures those features, with the add-on's own code, for real hair against every
hair skeleton of its race, and fits the weights:

- the game's hairstyles (from a game export of Instant Edit: every chara/human/cXXXX/obj/hair model,
  the hair skeletons and the manifest with each model's EST entry): the skeleton whose Hair weights move
  the hair most like its own weights on its own skeleton, head tilted and turned (`hairing.POSES`),
  should score highest (the expected error under a softmax over the skeletons is minimised);
- the active hair of a Penumbra collection (modded hair, with each mod's EST entries): the skeleton the
  modder picked, or one with the same bones, should score highest (log-likelihood), with weight LAMBDA.

Hair whose weights use bones its skeleton lacks is left out (mods swapped between hair slots).

Measuring takes about an hour in one Blender; split it with --part (each part writes hair_rows_<part>.npz):
    blender -b --factory-startup --python tools/fit_hair_matching.py -- measure <export> [<collection>] --part 1/4
    ... (parts 2 to 4 at the same time)
    blender -b --factory-startup --python tools/fit_hair_matching.py -- fit hair_rows_1.npz ... hair_rows_4.npz

The chain score (which chain takes a vertex's hanging part, `hairing.chain_terms`) is fitted to the chains
the same hairs give their hanging vertices (every CHAIN_EVERY-th of the game's hairstyles and all modded
ones, a modded hair counting CHAIN_MODDED of the game's), by softmax cross-entropy; the add-on uses the
fitted weights CHAIN_SHARPEN times as sharp and j_kami_a/b's handicap CHAIN_BACK_BODY times as big, which
moves modded hair most like its own weights when each chain swings on its own. Measure it before the
skeleton score, which poses the hair with these weights:
    blender ... -- measure-chains <export> [<collection>] --part 1/4      (writes chain_rows_<part>.npz)
    blender ... -- fit-chains chain_rows_1.npz ... chain_rows_4.npz

It needs Yet Another Addon installed in Blender (its xivpy reads the models) and SciPy (the add-on's deps/,
see the README). The collection is read from Penumbra's configuration in %APPDATA%\\XIVLauncher.
"""

import contextlib
import glob
import io
import json
import os
import re
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "deps"))
XIVPY = os.path.join(os.environ.get("APPDATA", ""), "Blender Foundation", "Blender", "5.2", "extensions",
                     "user_default", "yet_another_addon_ffxiv")
PENUMBRA = os.path.join(os.environ.get("APPDATA", ""), "XIVLauncher", "pluginConfigs", "Penumbra")
TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
HAIR_MODEL = re.compile(r"^chara/human/(c\d{4})/obj/hair/h(\d{4})/model/c\d{4}h\d{4}_hir\.mdl$")
RACE_CODE = {("Midlander", "Male"): "c0101", ("Midlander", "Female"): "c0201", ("Highlander", "Male"): "c0301",
             ("Highlander", "Female"): "c0401", ("Elezen", "Male"): "c0501", ("Elezen", "Female"): "c0601",
             ("Miqote", "Male"): "c0701", ("Miqote", "Female"): "c0801", ("Roegadyn", "Male"): "c0901",
             ("Roegadyn", "Female"): "c1001", ("Lalafell", "Male"): "c1101", ("Lalafell", "Female"): "c1201",
             ("AuRa", "Male"): "c1301", ("AuRa", "Female"): "c1401", ("Hrothgar", "Male"): "c1501",
             ("Hrothgar", "Female"): "c1601", ("Viera", "Male"): "c1701", ("Viera", "Female"): "c1801"}
LAMBDA = 1.0
L2 = 1e-3
# Weight on bones the hair's skeleton lacks above which a hair is left out.
MISSING_BONES = 0.02
# A modded hair's pick counts when a skeleton's bones are this close to it (mean distance, both ways).
SAME_BONES = 0.01
# The chain score: hanging vertices (at least 0.2 on hair bones) per hair at most; every n-th of the game's
# hairs; how many of the game's hairs a modded hair counts; how much sharper the add-on's weights are, and
# how much more j_kami_a/b's handicap (back_body) weighs, both chosen by how modded hair moves when each
# chain swings on its own; the terms fitted per centimetre (the others per unit).
CHAIN_SAMPLE = 3000
CHAIN_EVERY = 4
CHAIN_MODDED = 3.0
CHAIN_SHARPEN = 2.0
CHAIN_BACK_BODY = 1.5
CHAIN_CENTIMETRES = ("distance", "around", "past", "past_bangs", "above")


# -----------------------------------------------------------------------------
# Reading hair

def read_model(path):
    """(points (n, 3) Blender space, triangles, bone names, weights (n, bones)) of a hair .mdl's first LOD,
    without the vertices no triangle uses (shape key vertices)."""
    from xivpy.model import XIVModel, VertexUsage, get_vert_struct
    with contextlib.redirect_stdout(io.StringIO()):
        model = XIVModel.from_file(path)
    buffer = model.buffers
    header = model.header
    indices = np.frombuffer(buffer, np.uint8, header.idx_buffer_size[0], header.idx_offset[0]).view(np.uint16)
    vertices = buffer[model.header.vert_offset[0]:]
    names = list(model.bones)
    points, tris, weights = [], [], []
    base = 0
    for index, mesh in enumerate(model.meshes[:model.lods[0].mesh_count]):
        if not mesh.vertex_count:
            continue
        streams = {}
        uv = colour = 0
        layout = {}
        for element in model.vertex_declarations[index].vertex_elements:
            kind, count = get_vert_struct(element.type, element.usage)
            suffix = ""
            if element.usage == VertexUsage.COLOUR:
                suffix, colour = colour, colour + 1
            if element.usage == VertexUsage.UV:
                suffix, uv = uv, uv + 1
            name = "{:s}{}".format(element.usage.name.lower(), suffix)
            layout.setdefault(element.stream, []).append((name, kind) if count == 1 else (name, kind, (count,)))
        for stream, fields in layout.items():
            dtype = np.dtype(fields)
            if dtype.itemsize != mesh.vertex_buffer_stride[stream]:
                break
            offset = mesh.vertex_buffer_offset[stream]
            streams[stream] = np.frombuffer(vertices, dtype, mesh.vertex_count, offset).copy()
        else:
            p = streams[0]["position"][:, :3].astype(np.float64) @ TO_BLENDER.T
            t = indices[mesh.start_idx:mesh.start_idx + mesh.idx_count].astype(np.int64).reshape(-1, 3)
            w = np.zeros((mesh.vertex_count, len(names)))
            if "blend_weights" in streams[0].dtype.names:
                table = list(model.bone_tables[mesh.bone_table_idx].bone_idx)
                table = np.array(table + [0] * (256 - len(table)))
                blend = streams[0]["blend_weights"].astype(np.float64) / 255.0
                bones = table[streams[0]["blend_indices"].astype(np.int64)]
                for k in range(blend.shape[1]):
                    np.add.at(w, (np.arange(mesh.vertex_count), bones[:, k]), blend[:, k])
            points.append(p)
            tris.append(t + base)
            weights.append(w)
            base += mesh.vertex_count
    points, tris, weights = np.concatenate(points), np.concatenate(tris), np.concatenate(weights)
    used = np.zeros(len(points), dtype=bool)
    used[tris.ravel()] = True
    remap = np.cumsum(used) - 1
    weights = weights[used]
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-9)
    return points[used], remap[tris], names, weights


def vanilla_hairs(export):
    """(label, path, race, EST entry) of every hair model in an Instant Edit game export."""
    manifest = sorted(glob.glob(os.path.join(export, "instant-edit-export-*.json")))[-1]
    result = []
    for model in json.load(open(manifest, encoding="utf-8"))["models"]:
        match = HAIR_MODEL.match(model["gamePath"])
        if match is None:
            continue
        entry = (model.get("skeleton") or {}).get("est", {}).get("skeleton", 0)
        result.append((os.path.basename(model["gamePath"])[:10], os.path.join(export, model["gamePath"]),
                       match.group(1), int(entry)))
    return result


def _load(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _containers(mod_dir, settings):
    """(priority, files, manipulations) of a mod's active options (both of Penumbra's mod layouts)."""
    meta = _load(os.path.join(mod_dir, "meta.json")) if os.path.exists(os.path.join(mod_dir, "meta.json")) else {}
    default = meta.get("DefaultData")
    if default is None and os.path.exists(os.path.join(mod_dir, "default_mod.json")):
        default = _load(os.path.join(mod_dir, "default_mod.json"))
    result = [(-1e9, (default or {}).get("Files") or {}, (default or {}).get("Manipulations") or [])]
    groups = meta.get("Groups")
    if groups is None:
        groups = [_load(path) for path in sorted(glob.glob(os.path.join(mod_dir, "group_*.json")))]
    for group in groups:
        value = (settings or {}).get(group.get("Name"), group.get("DefaultSettings", 0))
        options = group.get("Options") or []
        priority = group.get("Priority", 0)
        kind = group.get("Type", "Single")
        if kind == "Single" and 0 <= value < len(options):
            result.append((priority, options[value].get("Files") or {}, options[value].get("Manipulations") or []))
        elif kind == "Multi":
            result += [(priority + o.get("Priority", 0) * 1e-3, o.get("Files") or {}, o.get("Manipulations") or [])
                       for i, o in enumerate(options) if value >> i & 1]
        elif kind == "Combining":
            containers = group.get("Containers") or []
            if 0 <= value < len(containers):
                chosen = containers[value]
                result.append((priority, chosen.get("Files") or {}, chosen.get("Manipulations") or []))
    return sorted(result, key=lambda c: c[0])


def modded_hairs(collection_name):
    """(label, path, race, EST entry) of the hair models active in a Penumbra collection (its EST entry
    the mods' when they set one)."""
    mods_dir = _load(os.path.join(PENUMBRA, "config", "penumbra.json"))["ModDirectory"]
    collection = next(d for d in (_load(p) for p in glob.glob(os.path.join(PENUMBRA, "collections", "*.json")))
                      if d.get("Name") == collection_name)
    models, est = {}, {}
    for mod, settings in sorted(collection["Settings"].items()):
        mod_dir = os.path.join(mods_dir, mod)
        if not settings.get("Enabled") or not os.path.isdir(mod_dir):
            continue
        rank = settings.get("Priority", 0)
        for priority, files, manipulations in _containers(mod_dir, settings.get("Settings")):
            for game, rel in files.items():
                match = HAIR_MODEL.match(game.replace("\\", "/").lower())
                if match:
                    key = (match.group(1), int(match.group(2)))
                    if key not in models or (rank, priority) >= models[key][0]:
                        models[key] = ((rank, priority), os.path.join(mod_dir, rel.replace("\\", os.sep)))
            for item in manipulations:
                manipulation = item.get("Manipulation") or {}
                race = RACE_CODE.get((manipulation.get("Race"), manipulation.get("Gender")))
                if item.get("Type") == "Est" and manipulation.get("Slot") == "Hair" and race:
                    key = (race, int(manipulation.get("SetId", -1)))
                    if key not in est or (rank, priority) >= est[key][0]:
                        est[key] = ((rank, priority), int(manipulation.get("Entry", 0)))
    return [("mod:{:s}h{:04d}".format(race, number), path, race, est.get((race, number), (0, None))[1])
            for (race, number), (_rank, path) in sorted(models.items())]


# -----------------------------------------------------------------------------
# Measuring

def measure(hairs, reference, vanilla_est):
    from magic_fit import hairing
    rows = []
    for count, (label, path, race, entry) in enumerate(hairs):
        if entry is None:  # a modded hair without its own EST entry uses the game's
            entry = vanilla_est.get((race, int(label[-4:])), 0)
        number = entry if reference.has_skeleton(race, entry) else 0
        try:
            points, tris, names, weights = read_model(path)
        except Exception as error:  # noqa: BLE001 - report and go on
            print("skipped", label, error)
            continue
        truth = reference.rig(race, number)
        keep = [i for i, n in enumerate(names) if n in truth.by_name]
        if weights[:, [i for i in range(len(names)) if i not in keep]].sum(1).mean() > MISSING_BONES:
            print("left out", label, "(weights on bones its skeleton lacks)")
            continue
        truth_weights = weights[:, keep]
        truth_weights /= np.maximum(truth_weights.sum(1, keepdims=True), 1e-9)
        truth_names = [names[i] for i in keep]
        hair = hairing._Hair(points, tris, reference, race, np.zeros(3))
        # Compare on the score's sample of the hair, as the score sees it.
        index = {tuple(p): i for i, p in enumerate(points)}
        rows_of = np.array([index[tuple(p)] for p in hair.points])
        truth_weights = truth_weights[rows_of]
        area = hair.area
        truth_posed = [hairing.skin(hair.points, truth_names, truth_weights, hairing.pose(truth, axis, degrees))
                       for axis, degrees in hairing.POSES]
        candidates = [0] + reference.skeletons(race)
        features, errors = [], []
        for candidate in candidates:
            rig = reference.rig(race, candidate)
            cheap = hairing._features(hair, rig)
            features.append(np.concatenate([cheap[:15], hairing._deformation(hair, rig), cheap[15:]]))
            names_c, weights_c = hairing.hair_weights(hair.points, hair.tris, rig)
            error = 0.0
            for (axis, degrees), expected in zip(hairing.POSES, truth_posed):
                posed = hairing.skin(hair.points, names_c, weights_c, hairing.pose(rig, axis, degrees))
                error += (np.linalg.norm(posed - expected, axis=1) * area).sum() / max(area.sum(), 1e-12)
            errors.append(error / len(hairing.POSES))
        rows.append(dict(label=label, race=race, truth=number, candidates=np.array(candidates),
                         features=np.array(features), errors=np.array(errors)))
        print("{:d}/{:d} {:s} truth {:s}".format(count + 1, len(hairs), label, hairing.skeleton_label(number)),
              flush=True)
    return rows


def chain_targets(names, weights, chains):
    """(n, len(chains)): each chain's share of the weights (columns ``names``); a bone in several chains
    counts for each evenly."""
    column = {name: i for i, name in enumerate(names)}
    member = {}
    for c, chain in enumerate(chains):
        for bone in chain.bones:
            member.setdefault(bone.name, []).append(c)
    targets = np.zeros((len(weights), len(chains)))
    for name, cs in member.items():
        if name in column:
            for c in cs:
                targets[:, c] += weights[:, column[name]] / len(cs)
    return targets


def measure_chains(hairs, reference, vanilla_est):
    from magic_fit import hairing
    rows = []
    for count, (label, path, race, entry) in enumerate(hairs):
        if entry is None:
            entry = vanilla_est.get((race, int(label[-4:])), 0)
        number = entry if reference.has_skeleton(race, entry) else 0
        try:
            points, tris, names, weights = read_model(path)
        except Exception as error:  # noqa: BLE001 - report and go on
            print("skipped", label, error)
            continue
        truth = reference.rig(race, number)
        if weights[:, [i for i, n in enumerate(names) if n not in truth.by_name]].sum(1).mean() > MISSING_BONES:
            continue
        chains = hairing.chains_of(truth.hair_bones())
        if len(chains) < 2:
            continue
        targets = chain_targets(names, weights, chains)
        hanging = targets.sum(1)
        area = hairing.vertex_areas(points, tris)
        candidates = np.flatnonzero((hanging > 0.2) & (area > 0.0))
        if len(candidates) < 20:
            continue
        p = area[candidates] * hanging[candidates]
        chosen = candidates[np.random.default_rng(0).choice(len(candidates), min(CHAIN_SAMPLE, len(candidates)),
                                                            replace=False, p=p / p.sum())]
        rows.append(dict(label=label, terms=hairing.chain_terms(truth.model(points[chosen]), truth, chains),
                         targets=targets[chosen] / hanging[chosen, None], mass=area[chosen] * hanging[chosen]))
        print("{:d}/{:d} {:s} {:d} chains".format(count + 1, len(hairs), label, len(chains)), flush=True)
    return rows


def fit_chains(rows):
    from scipy.optimize import minimize
    from magic_fit import hairing
    scale = np.array([100.0 if term in CHAIN_CENTIMETRES else 1.0 for term in hairing.CHAIN_TERMS])
    data = [(r["terms"] * scale, r["targets"], r["mass"] / r["mass"].sum(),
             CHAIN_MODDED if r["label"].startswith("mod:") else 1.0, r["label"].startswith("mod:")) for r in rows]

    def loss(w, subset):
        total, grad, norm = 0.0, np.zeros_like(w), 0.0
        for x, t, m, k, _modded in subset:
            s = -(x @ w)
            s -= s.max(1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(1, keepdims=True)
            total -= k * (m[:, None] * t * np.log(np.maximum(p, 1e-12))).sum()
            grad += k * np.einsum("vc,vck->k", m[:, None] * (t - p * t.sum(1, keepdims=True)), x)
            norm += k
        return total / norm + L2 * w @ w, grad / norm + 2 * L2 * w

    def agreement(w, subset, modded):
        values = [(m * ((x @ w).argmin(1) == t.argmax(1))).sum() for x, t, m, _k, mod in subset if mod == modded]
        return float(np.mean(values)) if values else float("nan")

    start = np.zeros(len(scale))
    start[hairing.CHAIN_TERMS.index("distance")] = 0.5
    folds = np.random.default_rng(0).permutation(len(data)) % 5
    held = {True: [], False: []}
    for fold in range(5):
        train = [d for d, f in zip(data, folds) if f != fold]
        test = [d for d, f in zip(data, folds) if f == fold]
        w = minimize(loss, start, args=(train,), jac=True, method="L-BFGS-B").x
        for modded in (True, False):
            held[modded].append(agreement(w, test, modded))
    w = minimize(loss, start, args=(data,), jac=True, method="L-BFGS-B").x * scale * CHAIN_SHARPEN
    w[hairing.CHAIN_TERMS.index("back_body")] *= CHAIN_BACK_BODY
    print("CHAIN_WEIGHTS = np.array(({:s}))".format(", ".join("{:.5g}".format(v) for v in w)))
    print("hanging hair on the reference's chain, hairs left out of the fit: modded {:.3f}, the game's {:.3f} "
          "({:d} hairs)".format(np.nanmean(held[True]), np.nanmean(held[False]), len(data)))


# -----------------------------------------------------------------------------
# Fitting

def fit(rows, reference, columns):
    from scipy.optimize import minimize
    from magic_fit import hairing

    def same_bones(race, a, b):
        sample = []
        for number in (a, b):
            bones = reference.rig(race, number).skeleton_bones()
            pts = [h + t * x for (h, x, length) in hairing._segments(bones, hairing.LEAF_REACH)
                   for t in np.linspace(0.0, length, 6)]
            sample.append(np.array(pts) if pts else np.zeros((0, 3)))
        if not len(sample[0]) and not len(sample[1]):
            return True
        if not len(sample[0]) or not len(sample[1]):
            return False
        d = np.linalg.norm(sample[0][:, None] - sample[1][None], axis=2)
        return 0.5 * (d.min(1).mean() + d.min(0).mean()) < SAME_BONES

    X = np.concatenate([r["features"][:, columns] for r in rows])
    mean, std = X.mean(0), X.std(0) + 1e-12
    data = []
    for r in rows:
        z = (r["features"][:, columns] - mean) / std
        if r["label"].startswith("mod:"):
            picked = np.array([same_bones(r["race"], r["truth"], c) for c in r["candidates"]])
            data.append((z, None, picked))
        else:
            data.append((z, r["errors"] * 1000.0, None))
    vanilla = sum(1 for d in data if d[2] is None)
    modded = len(data) - vanilla

    def loss(w):
        total, grad = 0.0, np.zeros_like(w)
        for z, errors, picked in data:
            s = z @ w
            p = np.exp(s - s.max())
            p /= p.sum()
            if picked is None:
                expected = p @ errors
                total += expected / vanilla
                grad += z.T @ (p * (errors - expected)) / vanilla
            else:
                chosen = p[picked].sum()
                total -= LAMBDA * np.log(chosen + 1e-12) / modded
                q = np.where(picked, p, 0.0) / max(chosen, 1e-12)
                grad += LAMBDA * z.T @ (p - q) / modded
        return total + L2 * w @ w, grad + 2 * L2 * w

    result = minimize(loss, np.zeros(len(columns)), jac=True, method="L-BFGS-B")
    return result.x, mean, std


def main():
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if not arguments:
        sys.exit(__doc__)
    sys.path.insert(0, XIVPY)
    from magic_fit import hairing
    reference = hairing.Reference.get()
    if arguments[0] in ("measure", "measure-chains"):
        export = arguments[1]
        collection = arguments[2] if len(arguments) > 2 and not arguments[2].startswith("--") else None
        part, parts = 1, 1
        if "--part" in arguments:
            part, parts = (int(v) for v in arguments[arguments.index("--part") + 1].split("/"))
        chains = arguments[0] == "measure-chains"
        out = "{:s}_rows_{:d}.npz".format("chain" if chains else "hair", part)
        if "--out" in arguments:
            out = arguments[arguments.index("--out") + 1]
        hairs = vanilla_hairs(export)
        vanilla_est = {(race, int(label[-4:])): entry for label, _p, race, entry in hairs}
        if chains:
            hairs = hairs[::CHAIN_EVERY]
        if collection:
            hairs += modded_hairs(collection)
        hairs = hairs[part - 1::parts]
        rows = measure_chains(hairs, reference, vanilla_est) if chains else measure(hairs, reference, vanilla_est)
        np.savez_compressed(out, rows=np.array(rows, dtype=object))
        print("wrote", out, len(rows), "hairs")
    elif arguments[0] == "fit-chains":
        fit_chains([row for path in arguments[1:] for row in np.load(path, allow_pickle=True)["rows"]])
    elif arguments[0] == "fit":
        rows = [row for path in arguments[1:] for row in np.load(path, allow_pickle=True)["rows"]]
        full = list(range(len(hairing.FULL_FEATURES)))
        cheap = [i for i, name in enumerate(hairing.FULL_FEATURES) if name not in ("stretch", "lag")]
        for name, columns in (("SCORE", cheap), ("FULL", full)):
            w, mean, std = fit(rows, reference, columns)
            print("{:s}_WEIGHTS = {}".format(name, [round(float(v), 4) for v in w]))
            print("{:s}_MEAN = {}".format(name, [float("%.5g" % v) for v in mean]))
            print("{:s}_STD = {}".format(name, [float("%.5g" % v) for v in std]))
            chosen = [r["errors"][int(np.argmax(((r["features"][:, columns] - mean) / std) @ w))]
                      for r in rows if not r["label"].startswith("mod:")]
            print("{:s}: vanilla hairs' error with the picked skeleton {:.2f} mm (their own: {:.2f} mm)".format(
                name, np.mean(chosen) * 1000, np.mean([r["errors"][list(r["candidates"]).index(r["truth"])]
                                                        for r in rows if not r["label"].startswith("mod:")]) * 1000))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
