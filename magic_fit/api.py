# SPDX-License-Identifier: GPL-3.0-or-later

"""Resizing meshes to sizes, for the Resize panel and for other add-ons, like Instant Edit's export.

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

An export of the model in ``collection`` at every size picked in Magic Fit's Resize panel:

    for size in api.resize_collection(collection):
        with size.shown():
            export(collection, size.name)

`resize_collection` resizes the clothing among the collection's meshes from the Resize panel's From Body to each
size, as the Resize button does: each size becomes a shape key of each mesh, kept in the file and reused while
neither the mesh nor the body changes, so a key fixed by hand stays as it is. Meshes whose shape keys follow the
sizes' settings, a devkit's body parts (the YAB devkit's Torso, Waist, Hands and Feet), aren't resized:
`SizedMeshes.shown` sets their keys as the body's would be at that size, and turns the clothing's key for the size
on, until the ``with`` block ends. At the size the body shows, the clothing shows the shape it fits the body in,
even while it shows another size's key. Everything else (Smoothing, Bridge Creases...) is as set in the Resize
panel.

To resize one size at a time (VERSION 2 on), with a step of a modal operator per size so Blender's UI stays alive,
make a Resizer, which reads the meshes and keeps what the sizes share, and resize it to each of its sizes:

    resizer = api.Resizer(collection)
    for size in resizer.sizes:
        sized = resizer.resize(size)
        with sized.shown():
            export(collection, sized.name)

Errors meant for the user raise ResizeError.
"""

import contextlib
import time

import bpy
import numpy as np

from . import properties, resizing, sizing
from .resizing import ResizeError

# Raised when what this module offers changes; other add-ons check they know it. 2 added Resizer.
VERSION = 2

__all__ = ("VERSION", "ResizeError", "Resizer", "Size", "SizedMeshes", "body_sizes", "picked_sizes",
           "resize_collection")


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
    """The sizes picked in the Resize panel's Sizes: ``settings`` (sizing.Setting, as the panel lists them, with the
    values the body has ``now``), ``varied`` (the settings picked with several values) and ``sizes`` (Size)."""

    def __init__(self, settings, picks, now):
        self.settings, self.now = settings, now
        self.varied = {name for name, values in picks.items() if len(values) > 1}
        self.sizes = []
        names = set()
        for values in sizing.combinations(settings, picks, now):
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
    """The sizes picked in the Resize panel (Picks) of ``brush`` (Scene.body_fit_brush), or None with Sizes off, or
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
    """The sizes picked in the Resize panel's Sizes of ``scene`` (the current one when None): a list of Size, every
    combination of the values picked. Empty with Sizes off."""
    picks = picked((scene or bpy.context.scene).body_fit_brush)
    return [] if picks is None else list(picks.sizes)


def body_sizes(body):
    """The settings the shape keys of mesh ``body`` follow (sizing.Setting, see sizing.body_settings) and why some
    can't be listed."""
    return sizing.body_settings(body)


# -----------------------------------------------------------------------------
# Resizing

class Job:
    """A resize as the Resize panel's settings (``brush``, Scene.body_fit_brush) say: the From and To bodies (the To
    body None for the From body itself), their keys, how the bodies are matched and the options. ``body`` overrides
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
    (``match``) when by shape, and the keys that still shape a mesh on top of its new one (``stuck``, {mesh: key
    names}: driven ones)."""

    def __init__(self):
        self.keys = {}
        self.own = {}
        self.made = self.kept = self.steps = self.rigid = 0
        self.how = None
        self.match = None
        self.stuck = {}


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
        # Rigid parts are numbered across all meshes, and Keep Close spreads its corrections along the meshes'
        # edges, numbered across them too.
        groups, first_part = [], 0
        for obj, shape in zip(meshes, self.shapes):
            parts, count = rigid_parts(obj, shape, job.rigid_size)
            groups.append(np.where(parts >= 0, parts + first_part, -1))
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
        states = [resizing.key_state(obj, label, resizing.fingerprint(inputs, shape_print))
                  for obj, shape_print in zip(meshes, self.shape_prints)]
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
            moved = change.apply(self.points, rigid=self.rigid, edges=self.edges, progress=step)
        first = 0
        for obj, shape, shape_print, mix, (block, state), need in zip(meshes, self.shapes, self.shape_prints,
                                                                      self.mixes, states, needed):
            part = slice(first, first + len(shape))
            first += len(shape)
            if need:
                record = {"from": self.from_print, "inputs": resizing.fingerprint(inputs, shape_print), "mix": mix}
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
# Instant Edit's export

class SizedMeshes:
    """The meshes resize_collection resized, at one size: ``name`` (for the size's file or mod option), ``size``
    (Size; None for the To shape without sizes), ``keys`` ({clothing mesh: the name of its key for the size, None
    where it stays as it is}), ``parts`` ({body part: {key name: value}}: the body parts' keys at the size),
    ``notes`` (messages for the user) and ``own`` ({clothing mesh staying as it is: the key values that show it
    so, see SizeRun})."""

    def __init__(self, name, size, keys, parts, notes, own=None):
        self.name, self.size, self.keys, self.parts, self.notes = name, size, keys, parts, notes
        self.own = own or {}

    def __repr__(self):
        return "SizedMeshes({!r}, {:d} resized, {:d} body parts)".format(
            self.name, sum(name is not None for name in self.keys.values()), len(self.parts))

    @contextlib.contextmanager
    def shown(self):
        """Show the meshes at this size until the ``with`` block ends: each clothing mesh's key for the size fully
        on and its other keys off, or for one staying as it is, the shape it fits the From body in (not a size it
        shows); the body parts' keys set as the size has them. Drivers of the keys set are muted meanwhile;
        everything is put back afterwards, whatever happens."""
        undo = []
        try:
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
            for obj, values in self.parts.items():
                keys = obj.data.shape_keys
                for name, value in values.items():
                    block = keys.key_blocks.get(name)
                    if block is not None:
                        _set_key(keys, block, value, undo)
            yield self
        finally:
            for restore in reversed(undo):
                restore()


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

    ``target``, ``sizes``, ``scene`` and ``body`` are as for resize_collection, and making one raises the errors it
    raises. ``sizes`` lists the sizes to resize to (Size, or None for the To shape alone). It reads the meshes and
    bodies when made, and keeps what every size shares (the clothing's shape, its distances from the From body,
    the From body's crease-free copy...) for all of them: make another after changing them.
    """

    def __init__(self, target, sizes=None, *, scene=None, body=None):
        scene = scene if scene is not None else bpy.context.scene
        brush = scene.body_fit_brush
        job = self.job = Job(brush, body)
        if job.body is None:
            raise ResizeError("Pick the From Body in Magic Fit's Resize panel")
        problem = resizing.check_shapes(job.body, job.from_key, job.to_body, job.to_key, job.match, True)
        if problem is not None:
            raise ResizeError(problem)
        if sizes is None:
            sizes = picked_sizes(scene) if body is None or body == brush.resize_from_object else []
            if not sizes:
                if resizing.check_shapes(job.body, job.from_key, job.to_body, job.to_key, job.match) is not None:
                    raise ResizeError("Pick the sizes to resize to in Magic Fit's Resize panel")
                sizes = [None]
        else:
            sizes = self._sizes(sizes)
        self.sizes = list(sizes)
        meshes = [obj for obj in _meshes(target) if obj not in (job.body, job.to_body)]
        body_values = [sizing.size_values(job.sized, size.values) if size is not None else {} for size in sizes]
        names = {name for size in sizes if size is not None for name in size.values}
        changed = {name for values in body_values for name in values}
        # Meshes whose keys follow the sizes (body parts) aren't resized.
        self.parts = [obj for obj in meshes if names and sizing.follows(obj, names, changed)]
        self.clothing = [obj for obj in meshes if obj not in self.parts]
        with _object_mode():
            self._work = _Sizes(self.clothing, job)

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
        values = sizing.size_values(self.job.sized, size.values) if size is not None else {}
        part_values = {}
        for part in self.parts:
            changes = sizing.size_values(part, size.values, values) if size is not None else {}
            if changes:
                part_values[part] = changes
        return SizedMeshes(label, size, dict(run.keys), part_values, notes, dict(run.own))


def resize_collection(target, sizes=None, *, scene=None, body=None, progress=None):
    """Resize the clothing in ``target`` to ``sizes`` for an export, as the Resize panel of ``scene`` (the current
    scene when None) is set: from its From Body (or ``body``, a Mannequin-like mesh), with its options.

    ``target`` is a collection (its visible meshes, in its child collections too) or the meshes themselves.
    ``sizes`` are Size, or dicts of setting values (see sizing.Setting); None takes those picked in the panel's
    Sizes, or the panel's To shape when it picks no sizes. Meshes whose shape keys follow the sizes' settings (body
    parts) aren't resized, nor are the bodies themselves; the others get a shape key per size, kept and reused (see
    resize_meshes). Returns a list of SizedMeshes, one per size; show one with its ``shown()``. Raises ResizeError.
    To resize one size at a time, use a Resizer.
    """
    started = time.perf_counter()
    with _object_mode():
        resizer = Resizer(target, sizes, scene=scene, body=body)
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
