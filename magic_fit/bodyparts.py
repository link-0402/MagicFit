# SPDX-License-Identifier: GPL-3.0-or-later

"""The devkit's body parts standing in for a model's copies of them, for an export at sizes (see api.Resizer).

A model made with a devkit carries copies of the body parts it shows (the skin under a top, a navel piercing).
Exported at another size, those copies would be resized like clothing and keep the From size's bone weights,
while the devkit's own parts take each size with their own weights and placements, switched by the devkit's
settings. So each mesh of the export is matched to the devkit part it stands for: the same slot (its name's group
and part numbers), material and attributes, lying on it at the From size. The match is made once, at the From
size: a variant the devkit swaps per size (Chains S and L share a slot) is the one matched at every size.
"""

import re

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

from . import resizing, sizing

# A devkit part stands in for an export mesh whose vertices lie this close to it at the From size (median, meters).
MATCH_DISTANCE = 0.02
# Parts this much closer than others (meters) are as close: one shown in the viewport is taken first.
TIE_DISTANCE = 1e-4
# Distances are measured from at most this many of an export mesh's vertices.
MATCH_SAMPLES = 2000

# A mesh's slot as Instant Edit reads it from its name: '0.1 Torso', or 'Part 0.1'...
_SLOT_FIRST = re.compile(r"^(\d+)\.(\d+)(?:\s+.*)?$")
_SLOT_LAST = re.compile(r"^(.+?)\s+(\d+)\.(\d+)$")
# ...after a level of detail (Blender puts a copy's '.001' after it).
_LOD = re.compile(r"\s+LOD\d+(?:\.\d+)?$", re.IGNORECASE)
# A copy's '.001', and the '[4f128fd5]' Instant Edit adds to a material it imports.
_COPY = re.compile(r"\.\d{3,}$")
_HASH = re.compile(r"\s*\[[0-9a-f]+\]$", re.IGNORECASE)
# The game's piercings material (its skin's is resizing.SKIN_MATERIAL).
PIERCINGS_MATERIAL = re.compile(r"mt_c\d{4}b\d{4}_piercings\b", re.IGNORECASE)


def slot(name):
    """(group, part) of a mesh named ``name`` ('0.1 Torso', 'Part 0.1', with a ' LOD0' after it), or None."""
    name = _LOD.sub("", name.strip())
    found = _SLOT_FIRST.match(name)
    if found is not None:
        return int(found.group(1)), int(found.group(2))
    found = _SLOT_LAST.match(name)
    return None if found is None else (int(found.group(2)), int(found.group(3)))


def material_key(obj):
    """The game material mesh ``obj`` is drawn with, as a name to compare (its ``xiv_material``, else its first
    material's): '/mt_c0201b0001_bibo.mtrl' and 'mt_c0201b0001_bibo.mtrl [4f128fd5]' are 'mt_c0201b0001_bibo.mtrl'.
    None without one."""
    name = obj.get("xiv_material")
    if not name and len(obj.material_slots) and obj.material_slots[0].material is not None:
        name = obj.material_slots[0].material.name
    if not name or not isinstance(name, str):
        return None
    name = _COPY.sub("", _HASH.sub("", _COPY.sub("", name.rsplit("/", 1)[-1].strip())))
    if not name.lower().endswith(".mtrl"):
        name += ".mtrl"
    return name.casefold()


def attributes(obj):
    """The attributes mesh ``obj`` is shown with (its custom properties 'atr_*' that are on)."""
    return frozenset(name for name in obj.keys()
                     if name.startswith("atr_") and isinstance(obj[name], (bool, int, float)) and obj[name])


def _is_body(key):
    """Whether material ``key`` is the game's skin or piercings."""
    return resizing.is_skin_material(key) or PIERCINGS_MATERIAL.search(key) is not None


def _shown(obj, view_layer):
    """Whether ``obj`` is shown in the viewport: not disabled there, nor hidden where it's in ``view_layer``."""
    if obj.hide_viewport:
        return False
    return obj.name not in view_layer.objects or not obj.hide_get(view_layer=view_layer)


def find_substitutes(meshes, job, names, changed, scene=None, usable=None):
    """Match the export ``meshes`` to the devkit parts they stand for, for a resize ``job`` (api.Job) to sizes of the
    settings ``names`` that change the body's keys ``changed``: parts of ``scene`` (the current one when None) that
    aren't exported nor the bodies, with the same slot, material and attributes, whose shape keys follow those
    settings (see sizing.follows; and ``usable(part)`` when given), lying within MATCH_DISTANCE of the mesh at the
    From size. The closest part wins, and stands in for one mesh only.

    Returns ({export mesh: devkit part}, notes for the user).
    """
    substitutes, notes = {}, []
    if not names or job.body is None:
        return substitutes, notes
    scene = scene if scene is not None else bpy.context.scene
    view_layer = bpy.context.view_layer if scene == bpy.context.scene else scene.view_layers[0]
    exported = set(meshes) | {job.body, job.to_body}
    groups = {}
    for obj in scene.objects:
        if obj.type != 'MESH' or obj in exported or obj.data.shape_keys is None:
            continue
        place = slot(obj.name)
        key = material_key(obj)
        if place is not None and key is not None:
            groups.setdefault((place, key, attributes(obj)), []).append(obj)
    body = job.body
    from_print = resizing.fingerprint(resizing._transform(resizing.rest_shape(body, job.source_values()),
                                                          body.matrix_world))
    now = {name: value for name, value in sizing.now_values(job.sized, sorted(names)).items() if value is not None}
    body_now = sizing.size_values(job.sized, now)
    surfaces = {}

    def surface(part):
        # The part's shape at the From size, or None when its drivers can't be worked out.
        if part not in surfaces:
            surfaces[part] = None
            try:
                values = sizing.size_values(part, now, body_now)
                co = resizing._transform(resizing.rest_shape(part, values), part.matrix_world)
            except resizing.ResizeError:
                return None
            tris = resizing._tris(part.data)
            if len(tris):
                surfaces[part] = BVHTree.FromPolygons(co.tolist(), tris.tolist(), all_triangles=True)
        return surfaces[part]

    def fate(obj):
        # What becomes of a mesh nothing stands in for.
        body_part = sizing.follows(obj, names, changed) and (usable is None or usable(obj))
        return "left as it is" if body_part else "resized as clothing"

    standing = {}
    for obj in meshes:
        place, key = slot(obj.name), material_key(obj)
        candidates = groups.get((place, key, attributes(obj)), []) if place is not None and key is not None else []
        candidates = [part for part in candidates
                      if sizing.follows(part, names, changed) and (usable is None or usable(part))]
        best = []
        if candidates and len(obj.data.vertices):
            points = resizing.from_shape(obj, from_print)[0]
            points = points[::max(1, -(-len(points) // MATCH_SAMPLES))]
            for part in candidates:
                tree = surface(part)
                if tree is None:
                    continue
                hits = (tree.find_nearest(point)[3] for point in points.tolist())
                median = float(np.median([np.inf if hit is None else hit for hit in hits]))
                if median <= MATCH_DISTANCE:
                    best.append((median, part))
        if not best:
            # (A body part of the model's own is set as the size has it: nothing to tell.)
            if key is not None and _is_body(key) and fate(obj) == "resized as clothing":
                notes.append("No devkit part stands in for '{:s}': resized as clothing".format(obj.name))
            continue
        closest = min(median for median, _part in best)
        tied = [part for median, part in best if median <= closest + TIE_DISTANCE]
        part = min(tied, key=lambda part: (not _shown(part, view_layer), part.name))
        if part in standing:
            notes.append("'{:s}' matches the devkit's '{:s}' too, which stands in for '{:s}': {:s}".format(
                obj.name, part.name, standing[part].name, fate(obj)))
            continue
        standing[part] = obj
        substitutes[obj] = part
        notes.append("'{:s}' is exported as the devkit's '{:s}'".format(obj.name, part.name))
    return substitutes, notes
