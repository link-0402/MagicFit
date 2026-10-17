# SPDX-License-Identifier: GPL-3.0-or-later

"""Resizing meshes to sizes, for the Auto Resize panel and for other add-ons, like Instant Edit's export.

Other add-ons find this module by the add-on's name, whichever repository installed it, and check VERSION:

    import importlib
    import bpy

    def magic_fit_api():
        for name in bpy.context.preferences.addons.keys():
            if name.rsplit(".", 1)[-1] == "magic_fit":
                try:
                    api = importlib.import_module(name + ".api")
                except ImportError:
                    return None
                return api if getattr(api, "VERSION", 0) >= 1 else None
        return None

An export of the model in ``collection`` at every size picked in Magic Fit's Auto Resize panel:

    for size in api.resize_collection(collection):
        with size.shown():
            export(collection, size.name)

`resize_collection` resizes the clothing among the collection's meshes from the Auto Resize panel's From Body to each
size, as the Resize button does: each size becomes a shape key of each mesh, kept in the file and reused while
neither the mesh nor the body changes, so a key fixed by hand stays as it is. Meshes whose shape keys follow the
sizes' settings, a devkit's body parts (the YAB devkit's Torso, Waist, Hands and Feet, and Piercing B, whose key
follows a modifier of the Torso), aren't resized: until the ``with`` block ends, `SizedMeshes.shown` switches the
devkit's settings they follow to the size, so their keys and modifiers (the Torso's Rue weights) follow by their
own drivers, sets their sliders as the body's are at that size, and turns the clothing's key for the size on. At
the size the body shows, the clothing shows the shape it fits the body in, even while it shows another size's key.
Everything else (Smoothing, Bridge Creases...) is as set in the Resize panel.

To resize one size at a time (VERSION 2 on), with a step of a modal operator per size so Blender's UI stays alive,
make a Resizer, which reads the meshes and keeps what the sizes share, and resize it to each of its sizes:

    resizer = api.Resizer(collection)
    for size in resizer.sizes:
        sized = resizer.resize(size)
        with sized.shown():
            export(collection, sized.name)

A model made with a devkit carries copies of the devkit's parts it shows (the skin under a top, a navel piercing).
Resized, they'd keep the From size's weights and placements; the devkit's own parts have each size's. With
``substitute=True`` (VERSION 3 on), each mesh that stands for a devkit part (the same slot, material and attributes,
lying on it at the From size, see bodyparts) isn't resized, and `Resizer.substitutes` maps it to that part, to
export in its place; `shown` shows the parts and puts them in the size:

    resizer = api.Resizer(collection, substitute=True)
    objects = [resizer.substitutes.get(obj, obj) for obj in collection_meshes]
    for size in resizer.sizes:
        sized = resizer.resize(size)
        with sized.shown():
            export(objects, sized.name)

Without it, those copies are resized like clothing, as before.

Which size of the From body a model was made on (VERSION 4 on), read from its skin (see FROM_SHARE), to switch the
From body to before resizing, or to pick which of a mod's options to import:

    reading = api.read_from_size(collection_meshes)
    if reading.size is not None and not reading.size.now:
        api.set_from_size(body, reading.size)

Errors meant for the user raise ResizeError.
"""

import contextlib
import time

import bpy
import numpy as np

from . import bodyparts, properties, resizing, sizing
from .resizing import ResizeError

# Raised when what this module offers changes; other add-ons check they know it. 2 added Resizer, 3 its
# ``substitute`` (and resize_collection's) and SizedMeshes.substitutes, and shown() switching the devkit's settings,
# 4 read_from_size and set_from_size.
VERSION = 4

__all__ = ("VERSION", "FromReading", "ResizeError", "Resizer", "Size", "SizedMeshes", "body_sizes", "picked_sizes",
           "read_from_size", "resize_collection", "set_from_size")


class Size:
    """A size of a body: the values of its settings (``values``, {setting name: value}, see sizing.Setting), its
    ``name`` ('Medium Rue') and whether it's what the body shows (``now``: the clothing fits it as it is)."""

    def __init__(self, name, values, now=False):
        self.name, self.values, self.now = name, dict(values), now

    def __repr__(self):
        return "Size({!r}, {!r}{:s})".format(self.name, self.values, ", now" if self.now else "")


# A size nothing names (the body as it is, among others named by what they change).
BASE_NAME = "Base"


class Picks:
    """The sizes picked in the Auto Resize panel's Sizes: ``settings`` (sizing.Setting, as the panel lists them, with
    the values the body has ``now``), ``varied`` (the settings picked with several values) and ``sizes`` (Size).
    ``too_many`` says the picks make more than sizing.MAX_SIZES sizes: only that many are listed."""

    def __init__(self, settings, picks, now):
        self.settings, self.now = settings, now
        self.varied = {name for name, values in picks.items() if len(values) > 1}
        self.sizes = []
        names = set()
        combinations = sizing.combinations(settings, picks, now, limit=sizing.MAX_SIZES + 1)
        self.too_many = len(combinations) > sizing.MAX_SIZES
        for values in combinations[:sizing.MAX_SIZES]:
            name = sizing.size_label(settings, values, now, self.varied) or BASE_NAME
            # Each size's key needs a name of its own.
            unique, count = name, 1
            while unique in names:
                count += 1
                unique = "{:s} {:d}".format(name, count)
            names.add(unique)
            self.sizes.append(Size(unique, values, sizing.is_now(values, now)))

    @property
    def changed(self):
        """The sizes that aren't what the body shows."""
        return [size for size in self.sizes if not size.now]


def picked(brush):
    """The sizes picked in the Auto Resize panel (Picks) of ``brush`` (Scene.body_fit_brush), or None with Sizes off, or
    for a body it lists none for."""
    if not brush.resize_use_sizes or not len(brush.resize_sizes):
        return None
    body = brush.resize_to_object or brush.resize_from_object
    if body is None or body.type != 'MESH':
        return None
    settings = []
    for item in brush.resize_sizes:
        options = [sizing.Option(option.value, option.label, option.word) for option in item.options]
        setting = sizing.Setting(item.name, item.label, item.kind, options, None,
                                 (None, item.path) if item.path else None)
        setting.more = item.more
        settings.append(setting)
    now = sizing.now_values(body, [setting.name for setting in settings])
    for setting in settings:
        setting.current = now.get(setting.name)
    return Picks(settings, {item.name: properties.picked_of(item) for item in brush.resize_sizes}, now)


def picked_sizes(scene=None):
    """The sizes picked in the Auto Resize panel's Sizes of ``scene`` (the current one when None): a list of Size, every
    combination of the values picked. Empty with Sizes off. Raises ResizeError when they make more than
    sizing.MAX_SIZES."""
    picks = picked((scene or bpy.context.scene).body_fit_brush)
    if picks is not None and picks.too_many:
        raise ResizeError(too_many_sizes())
    return [] if picks is None else list(picks.sizes)


def too_many_sizes():
    """What to tell the user when the picks make more than sizing.MAX_SIZES sizes."""
    return "The sizes picked make more than {:d} sizes: pick fewer".format(sizing.MAX_SIZES)


def body_sizes(body):
    """The settings the shape keys of mesh ``body`` follow (sizing.Setting, see sizing.body_settings) and why some
    can't be listed."""
    return sizing.body_settings(body)


# -----------------------------------------------------------------------------
# Resizing

class Job:
    """A resize as the Auto Resize panel's settings (``brush``, Scene.body_fit_brush) say: the From and To bodies (the
    To body None for the From body itself), their keys, how the bodies are matched and the options. ``body`` overrides
    the From Body."""

    def __init__(self, brush, body=None):
        self.body = body if body is not None else brush.resize_from_object
        to_body = brush.resize_to_object
        self.to_body = to_body if to_body is not None and to_body != self.body else None
        if brush.resize_use_keys:
            self.from_key, self.to_key = brush.resize_from_key, brush.resize_to_key
        else:
            self.from_key, self.to_key = "", ""
        self.match = brush.resize_match
        self.radius, self.bridge = brush.resize_radius, brush.resize_bridge
        self.rigid_size, self.keep_close = brush.resize_rigid_size, brush.resize_keep_close

    @property
    def sized(self):
        """The body whose sizes are picked: the To body, else the From body."""
        return self.to_body if self.to_body is not None else self.body

    def options(self):
        return (self.match, self.from_key, self.to_key, round(self.radius, 9), round(self.bridge, 9),
                round(self.rigid_size, 9), self.keep_close)

    def source_values(self):
        """Key values of the From body's From shape (see resizing.body_shapes)."""
        values = {}
        if self.from_key:
            values[self.from_key] = 1.0
        if self.to_body is None and self.to_key and self.to_key not in values:
            values[self.to_key] = 0.0
        return values

    def target_values(self, to_values):
        """Key values of the To body's To shape, with the key values ``to_values`` of a size."""
        values = dict(to_values or {})
        if self.to_key:
            values[self.to_key] = 1.0
        if self.to_body is None and self.from_key and self.from_key not in values:
            values[self.from_key] = 0.0
        return values


def rigid_parts(obj, co, size):
    """Loose parts of ``obj`` (positions ``co``) that Resize moves rigidly: those no larger than ``size``, or all of
    them for a mesh set to Rigid in Weight Transfer. Returns (part per vertex, -1 where not rigid; part count)."""
    whole = getattr(getattr(obj, "magic_fit_transfer", None), "rigid", False)
    if size <= 0.0 and not whole:
        return np.full(len(co), -1, dtype=np.int64), 0
    edges = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", edges)
    parts, count = resizing.loose_parts(co, edges.reshape(-1, 2))
    rigid = np.ones(count, dtype=bool) if whole else resizing.part_sizes(co, parts, count) <= size
    return np.where(rigid[parts], parts, -1), count


class SizeRun:
    """What resize_meshes did for one size: ``keys`` ({mesh: its key's name, None where it's shown as it is}),
    ``own`` ({mesh shown as it is: the key values giving it the shape it fits the From body in, see
    resizing.from_values}), how many keys it ``made`` and ``kept``, the meshes' ``rigid`` parts, and from the
    resize: its ``steps``, how the bodies were matched (``how``, see resizing.BodyShapes), the ShapeMatch
    (``match``) when by shape, the keys that still shape a mesh on top of its new one (``stuck``, {mesh: key
    names}: driven ones), and the ``skin`` meshes, which lie on the From body or are drawn with the game's skin
    material, and follow its surface exactly (see resizing.is_skin and resizing.skin_shares)."""

    def __init__(self):
        self.keys = {}
        self.own = {}
        self.made = self.kept = self.steps = self.rigid = 0
        self.how = None
        self.match = None
        self.stuck = {}
        self.skin = []


class _Sizes:
    """resize_meshes for ``meshes`` and ``job``, a size at a time: what it reads of the meshes and the From body
    once and, with ``share`` (for several sizes), what resizing to each size shares (see resizing.Shared)."""

    def __init__(self, meshes, job, share=True):
        self.meshes, self.job = meshes, job
        if not meshes:
            return
        body = job.body
        # The weights of the groups the bodies' keys are limited to, read once (per body).
        self.groups = {}
        self.from_shape = resizing.rest_shape(body, job.source_values(), self.groups.setdefault(body, {}))
        self.source = resizing._transform(self.from_shape, body.matrix_world)
        self.from_print = resizing.fingerprint(self.source)
        self.shapes, self.mixes = [], []
        for obj in meshes:
            shape, mix = resizing.from_shape(obj, self.from_print)
            self.shapes.append(shape)
            self.mixes.append(mix)
        # The key values showing each mesh as it fits the From body, while it may be showing a size.
        self.own = {obj: resizing.from_values(obj, self.from_print)[0] for obj in meshes}
        self.shape_prints = [resizing.fingerprint(shape) for shape in self.shapes]
        # Skin (a mod's copy of the body's lying on the From body, or what's drawn with the game's skin material)
        # follows the body's surface exactly, and clothing lying on it keeps its distance from it.
        tris = resizing._tris(body.data)
        self.skin, shares, faces, cover, first_vertex = [], [], [], [], 0
        for obj, shape in zip(meshes, self.shapes):
            marked = resizing.skin_vertices(obj)
            if not marked.all() and resizing.is_skin(shape, self.source, tris):
                marked[:] = True
            followed = resizing.skin_shares(shape, self.source, tris, marked)
            own = resizing._tris(obj.data) + first_vertex
            if followed.any():
                self.skin.append(obj)
                faces.append(own)
            cover.append(own)
            shares.append(followed)
            first_vertex += len(shape)
        self.bound = np.concatenate(shares) if shares else np.zeros(0)
        self.faces = np.concatenate(faces) if faces else None
        # With Keep Close, Keep Covered pushes the clothing back out over the body too, through the triangles of all
        # the meshes.
        self.cover = np.concatenate(cover) if cover and job.keep_close else None
        # Rigid parts are numbered across all meshes, and Keep Close spreads its corrections along the meshes'
        # edges, numbered across them too.
        groups, first_part = [], 0
        for obj, shape, followed in zip(meshes, self.shapes, shares):
            parts, count = rigid_parts(obj, shape, job.rigid_size)
            # Skin follows the body, however small its pieces.
            groups.append(np.where((parts >= 0) & (followed <= 0.0), parts + first_part, -1))
            first_part += count
        self.rigid = np.concatenate(groups)
        self.edges = None
        if job.keep_close:
            edges, first_vertex = [], 0
            for obj, shape in zip(meshes, self.shapes):
                pairs = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
                obj.data.edges.foreach_get("vertices", pairs)
                edges.append(pairs.reshape(-1, 2) + first_vertex)
                first_vertex += len(shape)
            self.edges = np.concatenate(edges)
        self.points = np.concatenate(self.shapes)
        # What a mesh's key depends on besides the bodies and options: its shape, its rigid parts and skin, and the
        # other meshes' skin (clothing near it follows it) and, with Keep Close, their triangles (Keep Covered).
        around = []
        if self.cover is not None:
            around += [self.points, self.cover]
        if self.bound.any():
            held = self.bound > 0.0
            around += [self.points[held], self.bound[held], self.faces if self.faces is not None else np.zeros(0)]
        around = resizing.fingerprint(*around) if around else ""
        self.mesh_prints, first_vertex = [], 0
        for shape, shape_print in zip(self.shapes, self.shape_prints):
            part = slice(first_vertex, first_vertex + len(shape))
            first_vertex += len(shape)
            self.mesh_prints.append(resizing.fingerprint(shape_print, self.rigid[part] >= 0, self.bound[part], around))
        self.shared = resizing.Shared() if share else None

    def size(self, size, label, show=False, progress=None):
        """Resize to ``size`` (Size, or None for the To shape alone) into keys named ``label``. Returns (SizeRun,
        notes for the user)."""
        run, notes = SizeRun(), []
        meshes, job = self.meshes, self.job
        if not meshes:
            return run, notes

        def step(done):
            if progress is not None:
                progress(done)
        run.rigid = len(np.unique(self.rigid[self.rigid >= 0]))
        run.skin = list(self.skin)
        body = job.body
        other = job.sized
        to_values = sizing.size_values(other, size.values) if size is not None else None
        to_shape = resizing.rest_shape(other, job.target_values(to_values), self.groups.setdefault(other, {}))
        target = resizing._transform(to_shape, other.matrix_world)
        if other == body and np.array_equal(target, self.source):
            run.keys = {obj: None for obj in meshes}
            run.own = dict(self.own)
            step(1.0)
            return run, notes
        inputs = resizing.fingerprint(self.source, target, job.options())
        states = [resizing.key_state(obj, label, resizing.fingerprint(inputs, mesh_print))
                  for obj, mesh_print in zip(meshes, self.mesh_prints)]
        needed = [state in ('NEW', 'STALE') for _block, state in states]
        moved = None
        if any(needed):
            bodies = resizing.body_shapes(body, job.from_key, job.to_body, job.to_key, job.match, to_values,
                                          shapes=(self.from_shape, to_shape))
            change = bodies.change(radius=job.radius, bridge=job.bridge, shared=self.shared)
            run.steps, run.how, run.match = change.steps, bodies.how, bodies.match
            if change.largest <= 0.0:
                run.keys = {obj: None for obj in meshes}
                run.own = dict(self.own)
                step(1.0)
                return run, notes
            moved = change.apply(self.points, rigid=self.rigid, edges=self.edges, bound=self.bound, faces=self.faces,
                                 cover=self.cover, progress=step)
        first = 0
        for obj, shape, mesh_print, mix, (block, state), need in zip(meshes, self.shapes, self.mesh_prints,
                                                                     self.mixes, states, needed):
            part = slice(first, first + len(shape))
            first += len(shape)
            if need:
                record = {"from": self.from_print, "inputs": resizing.fingerprint(inputs, mesh_print), "mix": mix}
                block, stuck = resizing.write_resized_key(obj, label, moved[part], record, show=show)
                run.made += 1
            else:
                run.kept += 1
                stuck = resizing.show_key(obj, block) if show else []
                if state == 'EDITED':
                    notes.append("'{:s}' on '{:s}' was changed by hand, and the clothing or body changed since it "
                                 "was made: kept as it is. Delete it to make it again".format(block.name, obj.name))
            run.keys[obj] = block.name
            if stuck:
                run.stuck[obj] = stuck
        step(1.0)
        return run, notes


def resize_meshes(meshes, sizes, job, labels, show=False, progress=None):
    """Resize ``meshes`` (clothing) from the From shape to each of ``sizes`` (Size, or None for the To shape alone),
    as ``job`` (Job) says, into a shape key per size named ``labels[i]``.

    Each mesh starts from the shape it fits the From body in (see resizing.from_shape), whatever size it shows. A
    key made before for the same size from the same inputs is kept, one changed by hand since too (with a note);
    others are made again. With ``show``, each mesh's key is turned on. Returns ([SizeRun per size], notes for the
    user). Raises ResizeError.
    """
    work = _Sizes(meshes, job, share=len(sizes) > 1)
    runs, notes = [], []
    for index, (size, label) in enumerate(zip(sizes, labels)):
        def step(done, index=index):
            if progress is not None:
                progress((index + done) / len(sizes))
        run, size_notes = work.size(size, label, show=show, progress=step)
        runs.append(run)
        notes.extend(size_notes)
    return runs, notes


# -----------------------------------------------------------------------------
# Which size clothing was made on

# The size a mod was made on is read from its skin: how far 90 % of the skin's points (FROM_SHARE, thinned to at most
# FROM_SAMPLE, welded) lie from each size of the From body, distances past FROM_REACH counting as that far. The 90th
# percentile, as the median is about the same at every size (the back and the arms don't change), and the distance to
# the surface, not to its vertices (that reads their spacing). Clothing alone (a loose sweater fits every chest) can't
# tell sizes apart well enough to offer.
FROM_SHARE = 0.9
FROM_SAMPLE = 4000
FROM_REACH = 0.02
# Fewer skin points than this read nothing.
FROM_LEAST = 50
# Nothing is read when the best size is this far off, or when fewer than FROM_REACHED of the points lie within
# FROM_REACH of it: the skin isn't made for this body (or it isn't lined up).
FROM_NONE = 0.01
FROM_REACHED = 0.5
# Sizes within FROM_TIE of the best are told apart by nothing under the skin (Large and Large Rue under a top's
# chest): the one the body shows stays, if it's among them.
FROM_TIE = 0.0002
# The best is unclear when it's more than FROM_RATIO of the nearest size outside those.
FROM_RATIO = 0.85
# The best is loose when farther off than this, as a mod's skin often is (sliders, nipples: other settings than the
# sizes'), which doesn't stop it ranking the sizes.
FROM_LOOSE = 0.003
# The size the body shows is said to disagree when the best fits within FROM_RATIO of its distance and FROM_MARGIN
# closer than it.
FROM_MARGIN = 0.0005
# Settings moving the body under the skin by less than this are kept as they are.
FROM_MOVES = 0.001


class FromReading:
    """What read_from_size read of which size of the From body the clothing's skin was made on.

    ``sizes`` are the sizes compared (Size; the one the body shows is ``now``, its index ``shows``, None when it isn't
    among them), ``fits`` how far off each is (meters: FROM_SHARE of the skin's points lie that close), ``reached``
    the share of the points within FROM_REACH of each, ``points`` how many skin points were read, ``best`` the index of
    the size fitting best and ``tied`` those within FROM_TIE of it. ``verdict``: 'NO_SKIN' (too few skin points),
    'NO_READING' (no size fits: see FROM_NONE), 'UNCLEAR' (another size fits about as well, see FROM_RATIO), 'LOOSE'
    (the best is farther off than FROM_LOOSE: other settings may differ) or 'CHOSEN'.
    """

    def __init__(self, sizes, fits, reached, points, settings):
        self.sizes, self.points, self.settings = sizes, points, settings
        self.fits, self.reached = np.asarray(fits, dtype=np.float64), np.asarray(reached, dtype=np.float64)
        self.shows = next((index for index, size in enumerate(sizes) if size.now), None)
        self.best = self.tied = None
        if points < FROM_LEAST or not len(sizes):
            self.verdict = 'NO_SKIN'
            return
        self.best = int(np.argmin(self.fits))
        best = self.fits[self.best]
        self.tied = [int(index) for index in np.flatnonzero(self.fits <= best + FROM_TIE)]
        others = np.delete(self.fits, self.tied)
        if best >= FROM_NONE or self.reached[self.best] < FROM_REACHED:
            self.verdict = 'NO_READING'
        elif len(others) and best > FROM_RATIO * others.min():
            self.verdict = 'UNCLEAR'
        elif best > FROM_LOOSE:
            self.verdict = 'LOOSE'
        else:
            self.verdict = 'CHOSEN'

    def __repr__(self):
        best = "" if self.best is None else ", best {!r} {:.2f} mm".format(self.sizes[self.best].name,
                                                                         1000.0 * self.fits[self.best])
        return "FromReading({:s}, {:d} points{:s})".format(self.verdict, self.points, best)

    @property
    def read(self):
        """Whether the skin tells which size it's made on (CHOSEN or LOOSE)."""
        return self.verdict in ('CHOSEN', 'LOOSE')

    @property
    def size(self):
        """The size the skin was made on (Size), or None when it doesn't tell: the one the body shows when it fits
        as well as the best (see FROM_TIE), else of those, the one changing the fewest settings from it."""
        if not self.read:
            return None
        if self.shows in self.tied:
            return self.sizes[self.shows]
        now = self.sizes[self.shows].values if self.shows is not None else {}

        def changes(index):
            values = self.sizes[index].values
            return sum(values.get(name) != value for name, value in now.items()), self.fits[index]
        return self.sizes[min(self.tied, key=changes)]

    @property
    def disagrees(self):
        """Whether the size the body shows (the From size) fits the skin clearly worse than the best (see
        FROM_RATIO and FROM_MARGIN)."""
        if self.verdict in ('NO_SKIN', 'NO_READING') or self.shows is None or self.shows in self.tied:
            return False
        now, best = self.fits[self.shows], self.fits[self.best]
        return now * FROM_RATIO > best and now > best + FROM_MARGIN


def _skin_points(meshes, job, source, tris):
    """World positions of the skin of ``meshes`` as they fit the From body (``source``, ``tris``): their vertices
    drawn with the game's skin material, and meshes lying on the body like a copy of its skin (see resizing.is_skin),
    welded and thinned to at most FROM_SAMPLE."""
    from_print = resizing.fingerprint(source)
    found = []
    for obj in meshes:
        shape, _mix = resizing.from_shape(obj, from_print)
        marked = resizing.skin_vertices(obj)
        if not marked.all() and resizing.is_skin(shape, source, tris):
            marked[:] = True
        if marked.any():
            found.append(shape[marked])
    if not found:
        return np.zeros((0, 3))
    points = np.concatenate(found)
    _spots, first = np.unique(np.round(points / 1e-6).astype(np.int64), axis=0, return_index=True)
    points = points[np.sort(first)]
    if len(points) > FROM_SAMPLE:
        points = points[np.sort(np.random.default_rng(0).choice(len(points), FROM_SAMPLE, replace=False))]
    return points


def _fit(points, co, tris):
    """How far off the body (``co``, ``tris``) ``points`` are (see FROM_SHARE): (that distance, the share within
    FROM_REACH)."""
    from mathutils.bvhtree import BVHTree
    low, high = points.min(axis=0) - FROM_REACH, points.max(axis=0) + FROM_REACH
    inside = ((co >= low) & (co <= high)).all(axis=1)
    near = tris[inside[tris].any(axis=1)]
    if not len(near):
        return FROM_REACH, 0.0
    used, local = np.unique(near, return_inverse=True)
    find = BVHTree.FromPolygons(co[used].tolist(), local.reshape(-1, 3).tolist(), all_triangles=True).find_nearest
    distance = np.full(len(points), FROM_REACH)
    for k, point in enumerate(points.tolist()):
        hit = find(point, FROM_REACH)
        if hit[0] is not None:
            distance[k] = min(hit[3], FROM_REACH)
    ordered = np.sort(distance)
    return float(ordered[int(np.ceil(FROM_SHARE * len(ordered))) - 1]), float((distance < FROM_REACH).mean())


def read_from_size(meshes, job=None, *, scene=None):
    """Read which size of the From body the clothing ``meshes`` was made on, from their skin (see FROM_SHARE): a
    FromReading. ``job`` (Job) is the Auto Resize panel's of ``scene`` (the current one) when None. Sizes are every
    combination of the From body's settings that reshape the body under the skin (see FROM_MOVES; not those the Sizes
    section lists as shared, nor those that barely reshape it); past sizing.MAX_SIZES of them, each setting is tried
    in turn from the best so far until none fits better. Nothing is changed. Raises ResizeError."""
    if job is None:
        job = Job((scene or bpy.context.scene).body_fit_brush)
    body = job.body
    if body is None or body.type != 'MESH':
        raise ResizeError("Pick the From Body")
    groups = {}
    shape_of = {}
    tris = resizing._tris(body.data).astype(np.int64)

    def shape(values):
        key = tuple(sorted(values.items()))
        if key not in shape_of:
            keys = dict(sizing.size_values(body, values))
            keys.update(job.source_values())
            shape_of[key] = resizing._transform(resizing.rest_shape(body, keys, groups), body.matrix_world)
        return shape_of[key]

    settings, _problems = sizing.body_settings(body)
    shared = sizing.shared_settings({setting.name: setting.source[1] for setting in settings
                                     if setting.source is not None and setting.kind != 'KEY'})
    now = {setting.name: setting.current for setting in settings if setting.current is not None}
    source = shape({})
    points = _skin_points(meshes, job, source, tris)
    if len(points) < FROM_LEAST:
        return FromReading([], [], [], len(points), settings)
    # The body's vertices under the skin.
    from mathutils.kdtree import KDTree
    tree = KDTree(len(source))
    for index, co in enumerate(source.tolist()):
        tree.insert(co, index)
    tree.balance()
    under = np.unique([tree.find(point)[1] for point in points.tolist()])
    candidates = []
    for setting in settings:
        if setting.more or setting.name in shared or setting.name not in now:
            continue
        values = [option.value for option in setting.options]
        moves = max((np.linalg.norm(shape({setting.name: value})[under] - source[under], axis=1).max()
                     for value in values if value != now[setting.name]), default=0.0)
        if moves >= FROM_MOVES:
            candidates.append(setting)
    picks = {setting.name: {option.value for option in setting.options} for setting in candidates}
    read = {}

    def fit(values):
        values = sizing.apply_rules(settings, values, now)
        key = tuple(sorted(values.items()))
        if key not in read:
            read[key] = (values,) + _fit(points, shape(values), tris)
        return read[key]
    sizes = sizing.combinations(settings, picks, now, limit=sizing.MAX_SIZES + 1)
    if len(sizes) <= sizing.MAX_SIZES:
        for values in sizes:
            fit(values)
    else:
        best = fit(dict(now))
        for _round in range(len(candidates)):
            start = best
            for setting in candidates:
                for option in setting.options:
                    tried = fit(dict(best[0], **{setting.name: option.value}))
                    if tried[1] < best[1]:
                        best = tried
            if best is start:
                break
    found = list(read.values())
    # Named by what they change, the one the body shows by all it is.
    named = [Size(sizing.size_label(settings, values, now) or sizing.state_label(settings, values) or BASE_NAME,
                  values, sizing.is_now(values, now)) for values, _fit_of, _reached in found]
    return FromReading(named, [fit_of for _values, fit_of, _reached in found],
                       [reached for _values, _fit_of, reached in found], len(points), settings)


def set_from_size(body, size, settings=None):
    """Switch the settings of the From ``body`` to ``size`` (Size), as by hand: its add-on's callbacks run. Returns
    the names of those that couldn't be set."""
    if settings is None:
        settings, _problems = sizing.body_settings(body)
    by_name = {setting.name: setting for setting in settings}
    order = {name: index for index, name in enumerate(sizing._Yab.NAMED)}
    failed = []
    for name in sorted(size.values, key=lambda name: (order.get(name, len(order)), name)):
        setting, value = by_name.get(name), size.values[name]
        if setting is None or setting.source is None:
            failed.append(name)
        elif setting.kind == 'KEY':
            keys, key_name = setting.source
            block = keys.key_blocks.get(key_name)
            if block is None:
                failed.append(name)
            elif block.value != value:
                block.value = value
        elif not sizing.set_setting(*setting.source, value):
            failed.append(name)
    bpy.context.view_layer.update()
    shown = sizing.current_values(settings)
    return [name for name in size.values if name not in failed and shown.get(name) != size.values[name]] + failed


# -----------------------------------------------------------------------------
# Instant Edit's export

def _fixed(body):
    """The settings the From ``body`` follows, as (owner pointer, path): never switched, it stays the From body."""
    if body is None:
        return set()
    return {(owner.as_pointer(), path) for _name, owner, path in sizing.setting_sources(body)}


def _blocked(part, names, fixed):
    """The settings among ``names`` that body ``part`` follows but that can't be switched: the From body's
    (``fixed``, see _fixed), or one whose add-on isn't running."""
    return {name for name, owner, path in sizing.setting_sources(part)
            if name in names and ((owner.as_pointer(), path) in fixed or not sizing.can_set(owner, path))}


class _StateSwitch:
    """How SizedMeshes.shown puts the body ``parts`` into a size: ``sources``, the settings their keys follow (see
    sizing.setting_sources) as (name, owner, path), in the order the devkit's own callbacks take them (the chest
    before Lavabod, which has no Masc chest), never those of the From body (``fixed``, see _fixed) nor those that
    can't be set; ``reads`` ({part: {(owner pointer, path): name} of the sources it follows}); ``blocked`` ({part:
    names of the settings it follows that can't be switched}); ``keys``, the Keys along the parts' chains (see
    sizing.chain_keys); and ``notes`` on those of the settings ``names`` that can't be set."""

    def __init__(self, parts, names, fixed):
        self.sources, self.reads, self.blocked, self.keys, self.notes = [], {}, {}, [], []
        listed, chained = set(), set()
        for part in parts:
            reads, blocked, unset = {}, set(), []
            for name, owner, path in sizing.setting_sources(part):
                source = (owner.as_pointer(), path)
                if source in fixed:
                    blocked.add(name)
                elif not sizing.can_set(owner, path):
                    blocked.add(name)
                    if name in names:
                        unset.append(path)
                else:
                    reads[source] = name
                    if source not in listed:
                        listed.add(source)
                        self.sources.append((name, owner, path))
            if unset:
                self.notes.append("'{:s}' can't be switched to the sizes: '{:s}' can't be set (is the devkit's "
                                  "script running?). Only the shape keys Magic Fit works out are set, its modifiers "
                                  "stay as they are".format(part.name, unset[0]))
            self.reads[part], self.blocked[part] = reads, blocked
            for keys in sizing.chain_keys(part):
                if keys.as_pointer() not in chained:
                    chained.add(keys.as_pointer())
                    self.keys.append(keys)
        order = {name: index for index, name in enumerate(sizing._Yab.NAMED)}
        self.sources.sort(key=lambda source: (order.get(source[0], len(order)), source[0], source[2]))

    def by_hand(self, values, failed=()):
        """The parts whose keys are set, their drivers muted, at a size of ``values`` ({setting name: value}) once
        the sources ``failed`` couldn't be set: those following a setting of the size that can't be switched, and
        those following none of them."""
        result = set()
        for part, reads in self.reads.items():
            needed = {source for source, name in reads.items() if name in values}
            if not needed or self.blocked[part].intersection(values) or needed.intersection(failed):
                result.add(part)
        return result


class SizedMeshes:
    """The meshes resize_collection resized, at one size: ``name`` (for the size's file or mod option), ``size``
    (Size; None for the To shape without sizes), ``keys`` ({clothing mesh: the name of its key for the size, None
    where it stays as it is}), ``parts`` ({body part: {key name: value}}: the body parts' keys at the size, the
    substitutes' too), ``substitutes`` ({export mesh: the devkit part exported in its place}, see
    Resizer), ``notes`` (messages for the user) and ``own`` ({clothing mesh staying as it is: the key values that
    show it so, see SizeRun})."""

    def __init__(self, name, size, keys, parts, notes, own=None, substitutes=None, switch=None):
        self.name, self.size, self.keys, self.parts, self.notes = name, size, keys, parts, notes
        self.own = own or {}
        self.substitutes = dict(substitutes or {})
        # How the body parts are put into the size (_StateSwitch); None sets their keys, their drivers muted.
        self._switch = switch

    def __repr__(self):
        return "SizedMeshes({!r}, {:d} resized, {:d} body parts, {:d} substituted)".format(
            self.name, sum(name is not None for name in self.keys.values()), len(self.parts), len(self.substitutes))

    @contextlib.contextmanager
    def shown(self):
        """Show the meshes at this size until the ``with`` block ends: each clothing mesh's key for the size fully
        on and its other keys off (their drivers muted meanwhile), or for one staying as it is, the shape it fits
        the From body in (not a size it shows). The body parts take the size as the devkit's own UI puts them in
        it: the settings they follow are switched, so their driven keys and modifiers follow by their own drivers
        (never the From body's settings: it stays the From body), and their keys nothing drives (sliders) are set
        as the size has them. The substitutes are shown. Everything is put back afterwards, whatever happens."""
        undo = []
        try:
            self._show_clothing(undo)
            switching = self._switch is not None and self.size is not None
            by_hand = None
            if switching:
                # Shown first, so the drivers' values are evaluated back below before they're hidden again: Blender
                # runs no drivers in an excluded collection and evaluates no modifiers of hidden objects.
                _show_objects(self.substitutes.values(), undo)
                by_hand = self._switch_settings(undo)
            self._set_parts(undo, by_hand)
            bpy.context.view_layer.update()
            yield self
        finally:
            # Each step is put back, also when another can't be.
            failure = None
            for restore in reversed(undo):
                try:
                    restore()
                except Exception as error:  # noqa: BLE001 - raised once the others are back
                    failure = failure or error
            if failure is not None:
                raise failure

    def _show_clothing(self, undo):
        """Show each clothing mesh's key for the size, or the shape it fits the From body in."""
        for obj, name in self.keys.items():
            keys = obj.data.shape_keys
            if keys is None:
                continue
            if name is None:
                for own_name, value in self.own.get(obj, {}).items():
                    block = keys.key_blocks.get(own_name)
                    if block is not None and (0.0 if block.mute else block.value) != value:
                        _set_key(keys, block, value, undo)
                continue
            if keys.key_blocks.get(name) is None:
                continue
            for block in keys.key_blocks:
                if block == keys.reference_key:
                    continue
                if block.name == name:
                    _set_key(keys, block, 1.0, undo)
                elif not block.mute and block.value != 0.0:
                    _set_key(keys, block, 0.0, undo)

    def _switch_settings(self, undo):
        """Switch the settings the body parts follow to the size's (see _StateSwitch), the devkit's callbacks
        running as when they're switched by hand. Returns the parts whose keys are set by hand."""
        switch, values = self._switch, self.size.values
        wanted = [(name, owner, path) for name, owner, path in switch.sources if name in values]
        if not wanted:
            return switch.by_hand(values)
        # The devkit's callbacks may change other settings along, and sliders (Lavabod swaps the chest's).
        keep = sizing.keep_settings([(owner, path) for _name, owner, path in switch.sources])
        saved = [(block, block.value, block.mute) for keys in switch.keys for block in keys.key_blocks]

        def restore():
            keep()
            for block, value, mute in saved:
                if block.value != value:
                    block.value = value
                if block.mute != mute:
                    block.mute = mute
        # Undone in this order: the settings put back, then evaluated, so what their drivers set is as before.
        undo.append(lambda: bpy.context.view_layer.update())
        undo.append(restore)
        failed = set()
        for name, owner, path in wanted:
            if not sizing.set_setting(owner, path, values[name]):
                failed.add((owner.as_pointer(), path))
        return switch.by_hand(values, failed)

    def _set_parts(self, undo, by_hand):
        """Set the body parts' keys the size changes: all of them for those in ``by_hand`` (all when None), their
        drivers muted; for the others those nothing drives, which no setting switches."""
        for obj, values in self.parts.items():
            keys = obj.data.shape_keys
            hand = by_hand is None or obj in by_hand
            for name, value in values.items():
                block = keys.key_blocks.get(name)
                if block is None:
                    continue
                if hand:
                    _set_key(keys, block, value, undo)
                elif not resizing._driven(keys, block, "value"):
                    _set_slider(keys, block, value, undo)


def _show_objects(objects, undo):
    """Show ``objects`` in the viewport: the layer collections holding them (and those around those) neither
    excluded nor hidden, the collections and the objects neither disabled nor hidden. Adds what puts each back to
    ``undo``."""
    view_layer = bpy.context.view_layer

    def turn(owner, attribute, value):
        old = getattr(owner, attribute)
        if old != value:
            setattr(owner, attribute, value)
            undo.append(lambda: setattr(owner, attribute, old))
    for obj in objects:
        for collection in obj.users_collection:
            for layer in _layer_path(view_layer.layer_collection, collection):
                turn(layer, "exclude", False)
                turn(layer, "hide_viewport", False)
                if layer.collection != view_layer.layer_collection.collection:
                    turn(layer.collection, "hide_viewport", False)
        turn(obj, "hide_viewport", False)
        if obj.name in view_layer.objects and obj.hide_get(view_layer=view_layer):
            obj.hide_set(False, view_layer=view_layer)
            undo.append(lambda obj=obj: obj.hide_set(True, view_layer=view_layer)
                        if obj.name in view_layer.objects else None)


def _layer_path(layer, collection):
    """The layer collections from ``layer`` down to the one of ``collection``, or [] when it's not under it."""
    if layer.collection == collection:
        return [layer]
    for child in layer.children:
        path = _layer_path(child, collection)
        if path:
            return [layer] + path
    return []


def _set_slider(keys, block, value, undo):
    """Set ``block`` of ``keys`` (a key nothing drives) to ``value``, unmuted unless a driver mutes it, and add what
    puts it back to ``undo``."""
    saved = (block.value, block.mute)

    def restore():
        block.value, block.mute = saved
    undo.append(restore)
    block.value = value
    if not resizing._driven(keys, block, "mute"):
        block.mute = False


def _set_key(keys, block, value, undo):
    """Set ``block`` of ``keys`` to ``value`` (unmuted), its drivers muted, and add what puts it back to ``undo``."""
    curves = []
    animation = keys.animation_data
    if animation is not None:
        name = block.name.replace('"', '\\"')
        paths = {'key_blocks["{:s}"].value'.format(name), 'key_blocks["{:s}"].mute'.format(name)}
        curves = [(curve, curve.mute) for curve in animation.drivers if curve.data_path in paths]
    saved = (block.value, block.mute)

    def restore():
        block.value, block.mute = saved
        for curve, mute in curves:
            curve.mute = mute

    undo.append(restore)
    for curve, _mute in curves:
        curve.mute = True
    block.value = value
    block.mute = False


def _meshes(target):
    """The meshes of ``target``: a collection's visible ones (its children's too), or the meshes given."""
    if isinstance(target, bpy.types.Collection):
        return [obj for obj in target.all_objects if obj.type == 'MESH' and obj.visible_get()]
    return [obj for obj in target if obj is not None and obj.type == 'MESH']


@contextlib.contextmanager
def _object_mode():
    """Leave Edit Mode for the block (shape keys can't be added in it), and come back."""
    edit = bpy.context.mode == 'EDIT_MESH'
    if edit:
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        yield
    finally:
        if edit:
            bpy.ops.object.mode_set(mode='EDIT')


class Resizer:
    """Resizes the clothing in ``target`` to one size at a time, as resize_collection does to all of them, for an
    export that resizes a size per step of a modal operator and keeps Blender's UI alive meanwhile:

        resizer = api.Resizer(collection)
        for size in resizer.sizes:          # one per timer tick
            sized = resizer.resize(size)
            with sized.shown():
                export(collection, sized.name)

    ``target``, ``sizes``, ``scene``, ``body`` and ``substitute`` are as for resize_collection, and making one raises
    the errors it raises. ``sizes`` lists the sizes to resize to (Size, or None for the To shape alone), ``parts``
    the body parts among the meshes and ``clothing`` those it resizes. With ``substitute``, ``substitutes`` maps
    each mesh standing for a devkit part (see bodyparts.find_substitutes) to that part, which is exported in its
    place: such a mesh is neither resized nor a body part. It reads the meshes and bodies when made, and keeps
    what every size shares (the clothing's shape, its distances from the From body, the From body's crease-free
    copy...) until each of them is resized: make another after changing them.
    """

    def __init__(self, target, sizes=None, *, scene=None, body=None, substitute=False):
        scene = scene if scene is not None else bpy.context.scene
        brush = scene.body_fit_brush
        job = self.job = Job(brush, body)
        if job.body is None:
            raise ResizeError("Pick the From Body in Magic Fit's Auto Resize panel")
        problem = resizing.check_shapes(job.body, job.from_key, job.to_body, job.to_key, job.match, True)
        if problem is not None:
            raise ResizeError(problem)
        if sizes is None:
            sizes = picked_sizes(scene) if body is None or body == brush.resize_from_object else []
            if not sizes:
                if resizing.check_shapes(job.body, job.from_key, job.to_body, job.to_key, job.match) is not None:
                    raise ResizeError("Pick the sizes to resize to in Magic Fit's Auto Resize panel")
                sizes = [None]
        else:
            sizes = self._sizes(sizes)
        self.sizes = list(sizes)
        meshes = [obj for obj in _meshes(target) if obj not in (job.body, job.to_body)]
        body_values = [sizing.size_values(job.sized, size.values) if size is not None else {} for size in sizes]
        names = {name for size in sizes if size is not None for name in size.values}
        changed = {name for values in body_values for name in values}
        fixed = _fixed(job.body)

        def part(obj):
            # Whether mesh ``obj`` is a body part, whose keys follow the sizes: but for one following them only
            # through another part whose settings can't be switched (the From body's), which would stay as it is.
            if not names or not sizing.follows(obj, names, changed):
                return False
            return sizing.follows(obj, names, changed, through=False) or not _blocked(obj, names, fixed)
        with _object_mode():
            # Meshes standing for the devkit's parts are exported as those.
            self.substitutes, notes = {}, []
            if substitute:
                self.substitutes, notes = bodyparts.find_substitutes(meshes, job, names, changed, scene, usable=part)
            meshes = [obj for obj in meshes if obj not in self.substitutes]
            # Meshes whose keys follow the sizes (body parts) aren't resized.
            self.parts = [obj for obj in meshes if part(obj)]
            self.clothing = [obj for obj in meshes if obj not in self.parts]
            self._switch = _StateSwitch(self.parts + list(self.substitutes.values()), names, fixed)
            # Given with the first size resized.
            self._notes = notes + self._switch.notes
            self._work = _Sizes(self.clothing, job, share=len(self.sizes) > 1)
        # What the sizes share is let go once each of them is resized (it can take hundreds of megabytes).
        self._pending = list(self.sizes)

    def _sizes(self, sizes):
        """``sizes`` given as Size or as dicts of setting values, as Size."""
        if all(isinstance(size, Size) for size in sizes):
            return list(sizes)
        settings, _problems = sizing.body_settings(self.job.sized)
        now = {setting.name: setting.current for setting in settings}
        given = [size if isinstance(size, Size) else sizing.apply_rules(settings, size, now) for size in sizes]
        # Named like picked sizes: by what changes, and by the settings given several values.
        plain = [size for size in given if not isinstance(size, Size)]
        varied = {name for name in {name for size in plain for name in size}
                  if len({size.get(name, now.get(name)) for size in plain}) > 1}
        return [size if isinstance(size, Size) else Size(
            sizing.size_label(settings, size, now, varied) or BASE_NAME, size, sizing.is_now(size, now))
            for size in given]

    def resize(self, size, progress=None):
        """Resize the clothing to ``size``, one of ``sizes`` (or another Size, or a dict of setting values, named by
        itself), into its shape key per size, kept and reused as resize_collection says. ``progress(fraction)`` is
        called as it goes. Returns a SizedMeshes. Raises ResizeError."""
        if size is not None and not isinstance(size, Size):
            size = self._sizes([size])[0]
        label = size.name if size is not None else _to_label(self.job)
        with _object_mode():
            run, notes = self._work.size(size, label, progress=progress)
        self._pending = [pending for pending in self._pending if pending is not size]
        if not self._pending:
            self._work.shared = None
        values = sizing.size_values(self.job.sized, size.values) if size is not None else {}
        # The chest sliders every size shares, as the body has them: the parts take them at every size, whatever
        # they had and whatever the devkit's script keeps for Lavabod.
        shared = sizing.shared_keys(self.job.sized)
        shows = sizing.size_values(self.job.sized, size.values, always=shared) if size is not None and shared else {}
        part_values = {}
        for part in self.parts + list(self.substitutes.values()):
            changes = sizing.part_values(part, size.values, values, shared, shows) if size is not None else {}
            if changes:
                part_values[part] = changes
        notes, self._notes = self._notes + notes, []
        return SizedMeshes(label, size, dict(run.keys), part_values, notes, dict(run.own), self.substitutes,
                           self._switch)


def resize_collection(target, sizes=None, *, scene=None, body=None, progress=None, substitute=False):
    """Resize the clothing in ``target`` to ``sizes`` for an export, as the Auto Resize panel of ``scene`` (the current
    scene when None) is set: from its From Body (or ``body``, a Mannequin-like mesh), with its options.

    ``target`` is a collection (its visible meshes, in its child collections too) or the meshes themselves.
    ``sizes`` are Size, or dicts of setting values (see sizing.Setting); None takes those picked in the panel's
    Sizes, or the panel's To shape when it picks no sizes. Meshes whose shape keys follow the sizes' settings (body
    parts) aren't resized, nor are the bodies themselves; the others get a shape key per size, kept and reused (see
    resize_meshes). With ``substitute``, meshes standing for a devkit's parts aren't resized either: each
    SizedMeshes' ``substitutes`` maps them to the parts to export in their place (see Resizer). Returns a list of
    SizedMeshes, one per size; show one with its ``shown()``. Raises ResizeError. To resize one size at a time, use
    a Resizer.
    """
    started = time.perf_counter()
    with _object_mode():
        resizer = Resizer(target, sizes, scene=scene, body=body, substitute=substitute)
        results, notes = [], []
        for index, size in enumerate(resizer.sizes):
            def step(done, index=index):
                if progress is not None:
                    progress((index + done) / len(resizer.sizes))
            sized = resizer.resize(size, progress=step)
            notes.extend(sized.notes)
            results.append(sized)
    for sized in results:
        sized.notes = list(notes)
    if results:
        results[0].notes.append("Resized {:d} mesh(es) to {:d} size(s) in {:.1f} s".format(
            len(resizer.clothing), len(results), time.perf_counter() - started))
    return results


def _to_label(job):
    """The name of the To shape without sizes: its key, else the To body's name, else 'Resized'."""
    return resizing.shape_label(job.sized, job.to_key, job.body) or "Resized"
