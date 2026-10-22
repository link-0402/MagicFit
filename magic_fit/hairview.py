# SPDX-License-Identifier: GPL-3.0-or-later

"""Chain view: shows the chains of the hair's own skeleton (Own Skeleton, `hairrig`) in the 3D view.

- Bone colours: every bone of a chain gets Blender's bone colour (`Bone.color.palette`, a theme palette) for
  the chain's kind, one per kind (KIND_PALETTES), when the chains are built or read back (`hairrig._fill_state`)
  and when a chain's kind changes in the list. They are saved with the file; exporters ignore them.
- Show Chains (`WindowManager.magic_fit_show_chains`, runtime: off after a file opens, and after the add-on is
  turned off and on): while it's on, the hair's armature (`current_armature`) is drawn In Front, and so is an
  armature that becomes current while it's on. What In Front was before is kept by armature (its session uid,
  so a renamed one counts), at runtime, and comes back when the toggle goes off. Files are saved with it as it
  was before, and an undo or redo that brings back In Front from while the toggle was on is put right.
- While it's on, an overlay tints every shown mesh the current armature deforms that has its own skeleton
  (`overlay_meshes`) by the chain that moves it (`chain_colors`): each chain in its kind's colour, every second
  chain of a kind darker, the chain picked in the list at full strength and the others faded. Weight on j_kao,
  the neck, the upper back or the ears gives no tint. The tint is drawn on the displayed (posed) hair, nudged
  a little off it either way along its (displayed) normals, depth-tested with the scene. A mesh's tint is built
  again when it moves or changes (depsgraph updates), at most every REBUILD_INTERVAL (its weights only once a
  paint stroke is over, SETTLE, and less often when they are slow to read, WEIGH_WAIT), and only what changed:
  its normals only when the hair moved, its colours only when its weights or the chains changed. Nothing is
  written to the meshes. No tint on a mesh in Edit Mode.
- Over the tint, in front of everything and in every mode, the overlay draws each chain bone of the current
  armature as a line from its head to its tail (`chain_lines`: posed, or the edit bones in Edit Mode), in the
  colour its chain's tint has, the picked chain at full strength and the others a little fainter (LINE_FLOOR),
  each over a wider black line. Blender draws bone colours in Pose and Edit Mode only, so this is what shows
  the chains in Object Mode. The lines follow the armature's pose and transform, and aren't drawn where the
  armature is hidden or the Bones overlay is off.
- Nothing is drawn while the 3D view shows X-ray.
"""

import time
from contextlib import contextmanager
from itertools import chain as flatten
from operator import itemgetter

import bmesh
import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

from . import hairing, hairrig
from .painting import read_geometry

# The bone colour set (Preferences > Themes > Bone Color Sets) of each chain kind's bones. In the default theme
# THEME09 is yellow, THEME03 green, THEME04 blue and THEME06 purple.
KIND_PALETTES = {
    "bangs": 'THEME09',
    "side": 'THEME03',
    "back": 'THEME04',
    "ponytail": 'THEME06',
}
# The normal colour of each of those sets in the default theme, for when there is no theme to read.
_DEFAULT_RGB = {
    'THEME09': (0.957, 0.788, 0.047),
    'THEME03': (0.118, 0.569, 0.035),
    'THEME04': (0.039, 0.212, 0.580),
    'THEME06': (0.263, 0.047, 0.471),
}

# Opacity of the tint where a vertex moves with the chains alone.
TINT_ALPHA = 0.6
# Strength of the chains not picked in the list, relative to the picked one.
FADE = 0.35
# Every second chain of a kind (in list order) is tinted this much darker, so neighbouring chains tell apart.
SHADE_DARK = 0.6
# The tint floats this far (m) off the hair, along its normals both ways (cards are seen from both sides), so
# it doesn't flicker with it.
NUDGE = 0.0005
# Seconds at least between two reads of a mesh for its tint while it keeps changing (posing, painting).
REBUILD_INTERVAL = 0.25
# A mesh's weights are read again at most every WEIGH_WAIT times as long as reading them took last time (and
# not more often than REBUILD_INTERVAL), so painting a large hair doesn't spend most of its time on the tint.
WEIGH_WAIT = 5
# Seconds a mesh's data must go without updates (a paint stroke sends one per dab) before its weights are read
# again: the tint catches up after a stroke instead of stuttering through it.
SETTLE = 0.2
# Width of the chain bones' lines, in pixels at the default resolution scale.
LINE_WIDTH = 3.0
# Opacity of the lines of the chains not picked in the list (the picked one's is 1): the tint's FADE would
# leave them unreadable over the bones.
LINE_FLOOR = 0.8
# Each line is drawn over a black one this much (px) wider, at this times the line's opacity, so it reads on
# light and dark alike.
UNDERLAY_WIDTH = 2.0
UNDERLAY_ALPHA = 0.6
# Displayed positions that moved less than this (m) since the last read count as where they were: the
# deformed hair then keeps its normals (a paint stroke doesn't move it, up to rounding).
MOVE_TOLERANCE = 1e-6

# {armature object's session uid: its In Front before Show Chains turned it on}
_in_front = {}
# {armature object's session uid: the In Front Show Chains gave back when it went off}, until it goes on again
# or a file opens: put back after an undo or redo that brought back the In Front of while it was on.
_restored = {}
# Session uids of armatures that became current while Show Chains is on, to draw In Front from a timer.
_pending = set()
# {(scene name, view layer name): {mesh object's session uid: _Tint}}
_tints = {}
# {(scene name, view layer name): _Lines of the current armature}
_lines = {}
_handle = None
_tag_due = None      # time.perf_counter() the views are next redrawn at, while a timer waits for it


def kind_rgb(kind):
    """The colour of chain ``kind``'s bones: its set's normal colour in the theme (KIND_PALETTES), or the default
    theme's when there is no theme."""
    palette = KIND_PALETTES.get(kind, KIND_PALETTES["back"])
    try:
        sets = bpy.context.preferences.themes[0].bone_color_sets
        return tuple(sets[int(palette[len("THEME"):]) - 1].normal)
    except (AttributeError, IndexError):
        return _DEFAULT_RGB[palette]


def color_bones(armature_object):
    """Give every bone of the chains in ``armature_object``'s rig state its chain's kind's bone colour
    (KIND_PALETTES), on its edit bones too while the armature is in Edit Mode (they replace the bones when it
    leaves Edit Mode)."""
    data = armature_object.data
    collections = [data.bones]
    if armature_object.mode == 'EDIT':
        collections.append(data.edit_bones)
    for chain in armature_object.magic_fit_hair_rig.chains:
        palette = KIND_PALETTES.get(chain.kind)
        if palette is None:
            continue
        for name in chain.bones.split(","):
            for bones in collections:
                bone = bones.get(name) if name else None
                if bone is not None and bone.color.palette != palette:
                    bone.color.palette = palette


def current_armature(context):
    """The armature whose chains are shown: the active hair's (a mesh with its own skeleton,
    `hairing.weighted_armature`), or the active armature when it has chains; else None."""
    obj = context.view_layer.objects.active if context.view_layer is not None else None
    if obj is None:
        return None
    if obj.type == 'MESH':
        return hairing.weighted_armature(obj) if obj.magic_fit_hair.custom else None
    if obj.type == 'ARMATURE' and len(obj.magic_fit_hair_rig.chains):
        return obj
    return None


# -----------------------------------------------------------------------------
# In Front

def _object(uid):
    """The object with session uid ``uid``, or None."""
    return next((obj for obj in bpy.data.objects if obj.session_uid == uid), None)


def _set_in_front(values):
    """Set the In Front of the armatures in ``values`` ({session uid: In Front}) that are still there."""
    for uid, value in values.items():
        obj = _object(uid)
        if obj is not None and obj.type == 'ARMATURE' and obj.show_in_front != value:
            obj.show_in_front = value


def _draw_in_front(armature_object):
    """Draw ``armature_object`` In Front, keeping what it was the first time."""
    uid = armature_object.session_uid
    if uid not in _in_front:
        _in_front[uid] = armature_object.show_in_front
    if not armature_object.show_in_front:
        armature_object.show_in_front = True


def _restore_in_front():
    """Give the armatures Show Chains drew In Front their In Front back."""
    _set_in_front(_in_front)


def _showing(wm):
    return wm is not None and getattr(wm, "magic_fit_show_chains", False)


def show_chains_changed(wm, context):
    """Property update callback of Show Chains: on, the hair's armature is drawn In Front; off, every armature
    it drew In Front gets its In Front back."""
    if wm.magic_fit_show_chains:
        _restored.clear()
        armature = current_armature(context)
        if armature is not None:
            _draw_in_front(armature)
    else:
        _restore_in_front()
        _restored.clear()
        _restored.update(_in_front)
        _in_front.clear()
        _pending.clear()
        clear()
    redraw_views()


def follow_current(context):
    """The armature whose chains are shown (`current_armature`), or None. While Show Chains is on, one not
    drawn In Front yet is drawn so soon, from a timer: the draw callback this is called from mustn't write it."""
    armature = current_armature(context)
    if armature is not None and _showing(context.window_manager) and armature.session_uid not in _in_front:
        _pending.add(armature.session_uid)
        if not bpy.app.timers.is_registered(_apply_pending_in_front):
            bpy.app.timers.register(_apply_pending_in_front, first_interval=0.0)
    return armature


def _apply_pending_in_front():
    if _showing(bpy.context.window_manager):
        for uid in _pending:
            obj = _object(uid)
            if obj is not None and obj.type == 'ARMATURE':
                _draw_in_front(obj)
    _pending.clear()
    return None


# -----------------------------------------------------------------------------
# Tint

def chain_weights(obj, rig_state):
    """How much each chain of ``rig_state`` (`Object.magic_fit_hair_rig`) moves each vertex of ``obj``
    ((vertices, chains)): the sum of the vertex's weights on the chain's bones (``obj.vertex_groups``)."""
    chains = len(rig_state.chains)
    count = len(obj.data.vertices)
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        layer = bm.verts.layers.deform.active
        if layer is None:
            return np.zeros((count, chains))
        # Python reads each vertex's groups fastest as two lists, its group indices and its weights.
        deform = list(map(itemgetter(layer), bm.verts))
        groups = [vert.keys() for vert in deform]
        values = [vert.values() for vert in deform]
    finally:
        bm.free()
    sizes = np.fromiter(map(len, groups), dtype=np.int64, count=count)
    total = int(sizes.sum())
    groups = np.fromiter(flatten.from_iterable(groups), dtype=np.int64, count=total)
    values = np.fromiter(flatten.from_iterable(values), dtype=np.float64, count=total)
    # Each vertex group's chain; groups of no chain go to a column past the chains, dropped at the end.
    column = np.full(max(len(obj.vertex_groups), int(groups.max(initial=-1)) + 1), chains, dtype=np.int64)
    for c, chain in enumerate(rig_state.chains):
        for name in chain.bones.split(","):
            group = obj.vertex_groups.get(name) if name else None
            if group is not None:
                column[group.index] = c
    cells = np.repeat(np.arange(count) * (chains + 1), sizes) + column[groups]
    weights = np.bincount(cells, values, minlength=count * (chains + 1)).reshape(count, chains + 1)
    return weights[:, :chains]


def chain_shades(kinds, rgb=kind_rgb):
    """Each chain's colour ((chains, 3)) for chains of ``kinds``: its kind's (``rgb``), SHADE_DARK times that for
    every second chain of a kind (in list order)."""
    shades = np.empty((len(kinds), 3))
    seen = {}
    for c, kind in enumerate(kinds):
        number = seen.get(kind, 0)
        seen[kind] = number + 1
        shades[c] = np.asarray(rgb(kind), dtype=np.float64)[:3] * (SHADE_DARK if number % 2 else 1.0)
    return shades


def chain_strengths(count, active):
    """Each of ``count`` chains' strength: 1 for ``active`` and FADE for the others, or 1 for all when
    ``active`` is none of them."""
    strength = np.ones(count)
    if 0 <= active < count:
        strength[:] = FADE
        strength[active] = 1.0
    return strength


def chain_colors(weights, kinds, active, rgb=kind_rgb):
    """The tint of each vertex ((n, 4) float32 RGBA) from its ``weights`` (n, chains) on chains of ``kinds``.
    A chain's colour is `chain_shades`'; a vertex's is the mean of its chains', by weight. Its alpha is
    TINT_ALPHA times its weight on the chains, FADE times that on chains other than ``active`` (when that is
    one of them). Vertices without chain weight are clear."""
    weights = np.asarray(weights, dtype=np.float64)
    result = np.zeros((len(weights), 4), dtype=np.float32)
    count = weights.shape[1] if weights.ndim == 2 else 0
    if not count or not len(weights):
        return result
    shades = chain_shades(kinds, rgb)
    total = weights.sum(axis=1)
    has = total > 1e-9
    result[has, :3] = weights[has] @ shades / total[has, None]
    result[:, 3] = TINT_ALPHA * np.clip(weights @ chain_strengths(count, active), 0.0, 1.0)
    return result


def chain_lines(armature_object, rig_state, rgb=kind_rgb):
    """The lines drawn for the chain bones of ``rig_state`` in ``armature_object``: positions ((2k, 3) float32,
    world space, each bone's head then its tail; posed, or the edit bones' while the armature is in Edit Mode)
    and their colours ((2k, 4) float32): the chain's colour (`chain_shades`), alpha its strength
    (`chain_strengths`) but at least LINE_FLOOR. Bones listed in a chain but not in the armature are left
    out."""
    chains = rig_state.chains
    shades = chain_shades([chain.kind for chain in chains], rgb)
    strengths = np.maximum(chain_strengths(len(chains), rig_state.active), LINE_FLOOR)
    if armature_object.mode == 'EDIT':
        bones = armature_object.data.edit_bones
    else:
        bones = armature_object.pose.bones
    points, colors = [], []
    for c, chain in enumerate(chains):
        color = (*shades[c], strengths[c])
        for name in chain.bones.split(","):
            bone = bones.get(name) if name else None
            if bone is not None:
                points += (bone.head, bone.tail)
                colors += (color, color)
    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    m = np.array(armature_object.matrix_world, dtype=np.float64)
    positions = np.array(points, dtype=np.float64) @ m[:3, :3].T + m[:3, 3]
    return positions.astype(np.float32), np.array(colors, dtype=np.float32)


def line_underlay(colors):
    """The colours ((2k, 4) float32) of the black lines drawn under lines of ``colors`` (`chain_lines`): alpha
    UNDERLAY_ALPHA times theirs."""
    result = np.zeros((len(colors), 4), dtype=np.float32)
    result[:, 3] = UNDERLAY_ALPHA * np.asarray(colors, dtype=np.float32)[:, 3]
    return result


def overlay_meshes(context, armature_object):
    """The meshes tinted: those shown in ``context``'s view layer that ``armature_object`` deforms
    (`hairing.weighted_armature`) and that have their own skeleton (``magic_fit_hair.custom``). Not one whose
    mesh is in Edit Mode (also through another object sharing it), nor one displayed with other vertices than
    its own (a Subdivision modifier): its weights wouldn't line up with them."""
    view_layer = context.view_layer
    depsgraph = None
    result = []
    for obj in view_layer.objects:
        if obj.type != 'MESH' or obj.data.is_editmode or not obj.magic_fit_hair.custom:
            continue
        if not obj.visible_get(view_layer=view_layer) or hairing.weighted_armature(obj) != armature_object:
            continue
        if depsgraph is None:
            depsgraph = context.evaluated_depsgraph_get()
        if len(obj.evaluated_get(depsgraph).data.vertices) != len(obj.data.vertices):
            continue
        result.append(obj)
    return result


class _Tint:
    """The tint of one mesh object."""

    __slots__ = ("co", "normals", "tris", "data", "weights", "weights_key", "colors", "colors_key", "batch",
                 "built", "weighed", "weigh_cost", "marked", "moved", "reweigh")

    def __init__(self):
        self.co = None              # world space positions, as displayed
        self.normals = None         # their normals (`_evaluated_normals`)
        self.tris = None
        self.data = 0               # session uid of the mesh data the weights were read from
        self.weights = None         # `chain_weights`
        self.weights_key = None     # the chains' bones and the mesh's vertex groups they were read with
        self.colors = None          # `chain_colors`
        self.colors_key = None      # the chains' kinds, the active chain and the kinds' colours they were made with
        self.batch = None           # what's drawn: None to be built, False when nothing is tinted
        self.built = 0.0            # time.perf_counter() the positions were last read at
        self.weighed = 0.0          # time.perf_counter() the weights were last read at
        self.weigh_cost = 0.0       # seconds that read took
        self.marked = 0.0           # time.perf_counter() an update of the mesh data last reached it
        self.moved = True           # positions to be read again
        self.reweigh = True         # weights to be read again


class _Lines:
    """The lines of one armature's chain bones (`chain_lines`)."""

    __slots__ = ("uid", "data", "positions", "colors", "key", "batch", "underlay", "moved")

    def __init__(self, armature_object):
        self.uid = armature_object.session_uid
        self.data = armature_object.data.session_uid
        self.positions = None
        self.colors = None
        self.key = None             # the chains' bones and kinds, the active chain and the kinds' colours
        self.batch = None           # what's drawn: None to be built, False when there are no lines
        self.underlay = None        # the black lines under them (`line_underlay`), built with ``batch``
        self.moved = True           # the armature was posed or moved since the lines were made


def _forget_old_layers(store):
    """Forget the entries of ``store`` ({(scene name, view layer name): ...}) of scenes and view layers that were
    renamed or deleted since."""
    for old in list(store):
        scene = bpy.data.scenes.get(old[0])
        if scene is None or scene.view_layers.get(old[1]) is None:
            del store[old]


def _layer_tints(context):
    """The tints of the view layer in ``context`` ({mesh object's session uid: _Tint}), made on first use.
    Windows can show different view layers side by side."""
    key = (context.scene.name, context.view_layer.name)
    tints = _tints.get(key)
    if tints is None:
        _forget_old_layers(_tints)
        tints = _tints[key] = {}
    return tints


def _evaluated_normals(obj, depsgraph):
    """The normals of ``obj``'s displayed (evaluated) mesh's vertices, in world space ((n, 3))."""
    obj_eval = obj.evaluated_get(depsgraph)
    vertices = obj_eval.data.vertices
    normals = np.empty(len(vertices) * 3, dtype=np.float32)
    vertices.foreach_get("normal", normals)
    m = np.array(obj_eval.matrix_world.to_3x3().inverted_safe().transposed(), dtype=np.float64)
    normals = normals.reshape(-1, 3) @ m.T
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _same_place(tint, co, tris):
    """Whether displayed positions ``co`` and triangles ``tris`` are those ``tint`` was built with."""
    return (tint.co is not None and tint.co.shape == co.shape and np.array_equal(tint.tris, tris)
            and (not len(co) or float(np.abs(tint.co - co).max()) <= MOVE_TOLERANCE))


def _refresh_lines(context, armature_object):
    """The lines of ``armature_object``'s chain bones for the view layer in ``context`` (_Lines), made again
    when the armature was posed or moved, or its chains changed; every time in Edit Mode, where bones are
    edited without telling."""
    key = (context.scene.name, context.view_layer.name)
    lines = _lines.get(key)
    if lines is None:
        _forget_old_layers(_lines)
    if (lines is None or lines.uid != armature_object.session_uid
            or lines.data != armature_object.data.session_uid):
        lines = _lines[key] = _Lines(armature_object)
    rig = armature_object.magic_fit_hair_rig
    lines_key = (tuple((chain.bones, chain.kind) for chain in rig.chains), rig.active,
                 tuple(kind_rgb(kind) for kind in hairrig.KINDS))
    if lines.moved or lines.key != lines_key or armature_object.mode == 'EDIT':
        positions, colors = chain_lines(armature_object, rig)
        if (lines.positions is None or not np.array_equal(positions, lines.positions)
                or not np.array_equal(colors, lines.colors)):
            lines.positions, lines.colors = positions, colors
            lines.batch = None
        lines.key = lines_key
        lines.moved = False
    return lines


def _refresh(context, armature_object, meshes):
    """Bring the tints of ``meshes`` (`overlay_meshes` of ``armature_object``) up to date and forget the view
    layer's others. Positions that changed less than REBUILD_INTERVAL after they were last read wait. Weights
    wait until the mesh data had no update for SETTLE, and REBUILD_INTERVAL or WEIGH_WAIT times their last
    read's time after it, whichever is longer; the old tint stays meanwhile. A timer redraws the views when
    they're due. Returns True when one waits."""
    tints = _layer_tints(context)
    shown = {obj.session_uid for obj in meshes}
    for uid in [uid for uid in tints if uid not in shown]:
        del tints[uid]
    rig = armature_object.magic_fit_hair_rig
    bones = tuple(chain.bones for chain in rig.chains)
    kinds = tuple(chain.kind for chain in rig.chains)
    colors_key = (kinds, rig.active, tuple(kind_rgb(kind) for kind in hairrig.KINDS))
    now = time.perf_counter()
    depsgraph = None
    wait = None
    for obj in meshes:
        tint = tints.get(obj.session_uid)
        if tint is None:
            tint = tints[obj.session_uid] = _Tint()
        count = len(obj.data.vertices)
        weights_key = (bones, tuple(group.name for group in obj.vertex_groups))
        if tint.data != obj.data.session_uid or tint.weights_key != weights_key:
            tint.reweigh = True
        if tint.reweigh:
            # Weights that don't fit the mesh any more are read at once; else after the stroke (the mesh data
            # settled) and when the last read is long enough ago, longer for a mesh slow to read.
            left = 0.0
            if tint.weights is not None and len(tint.weights) == count:
                left = max(tint.marked + SETTLE,
                           tint.weighed + max(REBUILD_INTERVAL, WEIGH_WAIT * tint.weigh_cost)) - now
            if left > 0.0:
                wait = left if wait is None else min(wait, left)
            else:
                started = time.perf_counter()
                tint.weights = chain_weights(obj, rig)
                tint.weigh_cost = time.perf_counter() - started
                tint.weighed = now
                tint.data, tint.weights_key = obj.data.session_uid, weights_key
                tint.colors_key = None
                tint.reweigh = False
        if tint.moved or tint.co is None or len(tint.co) != count:
            left = 0.0
            if tint.co is not None and len(tint.co) == count:
                left = tint.built + REBUILD_INTERVAL - now
            if left > 0.0:
                wait = left if wait is None else min(wait, left)
            else:
                if depsgraph is None:
                    depsgraph = context.evaluated_depsgraph_get()
                geometry = read_geometry(obj, depsgraph, source='DISPLAYED')
                # A paint stroke lists the hair's geometry as updated too: keep what didn't move.
                if not _same_place(tint, geometry.co, geometry.tris):
                    tint.co, tint.tris = geometry.co, geometry.tris
                    tint.normals = _evaluated_normals(obj, depsgraph)
                    tint.batch = None
                tint.moved = False
                tint.built = now
        if tint.colors_key != colors_key and len(tint.weights) == len(tint.co):
            tint.colors = chain_colors(tint.weights, kinds, rig.active)
            tint.colors_key = colors_key
            tint.batch = None
    if wait is not None:
        redraw_views(wait)
    return wait is not None


def tint_arrays(tint):
    """What's drawn of ``tint``: positions ((2n, 3) float32: the hair NUDGE out along its normals, then as far
    in), their colours ((2n, 4) float32) and the triangles' corners ((k, 3) int32: those with a tinted corner,
    on both)."""
    offset = NUDGE * tint.normals
    positions = np.concatenate((tint.co + offset, tint.co - offset)).astype(np.float32)
    colors = np.concatenate((tint.colors, tint.colors))
    tris = tint.tris.reshape(-1, 3)
    tris = tris[(tint.colors[tris, 3] > 0.0).any(axis=1)]
    indices = np.concatenate((tris, tris + len(tint.co))).astype(np.int32)
    return positions, colors, indices


def clear():
    _tints.clear()
    _lines.clear()


# -----------------------------------------------------------------------------
# Keeping up with changes

@persistent
def _on_depsgraph_update(_scene, depsgraph):
    if not _tints and not _lines:
        return
    for update in depsgraph.updates:
        data = update.id.original
        if isinstance(data, bpy.types.Object):
            if update.is_updated_geometry or update.is_updated_transform:
                uid = data.session_uid
                for tints in _tints.values():
                    tint = tints.get(uid)
                    if tint is not None:
                        tint.moved = True
                # Posing lists the armature object's geometry.
                for lines in _lines.values():
                    if lines.uid == uid:
                        lines.moved = True
        elif isinstance(data, bpy.types.Mesh):
            # Weights live on the mesh data: posing doesn't list it, painting does (each dab).
            now = time.perf_counter()
            for tints in _tints.values():
                for tint in tints.values():
                    if tint.data == data.session_uid:
                        tint.reweigh = True
                        tint.marked = now
        elif isinstance(data, bpy.types.Armature):
            for lines in _lines.values():
                if lines.data == data.session_uid:
                    lines.moved = True


def redraw_views(delay=0.01):
    """Redraw the 3D views in ``delay`` seconds, or sooner when that's due already (from a timer: tagging
    redraws while drawing isn't safe)."""
    global _tag_due
    due = time.perf_counter() + delay
    if bpy.app.timers.is_registered(_tag_views):
        if _tag_due is not None and _tag_due <= due:
            return
        bpy.app.timers.unregister(_tag_views)
    _tag_due = due
    bpy.app.timers.register(_tag_views, first_interval=delay)


def _tag_views():
    global _tag_due
    _tag_due = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    return None


# -----------------------------------------------------------------------------
# Drawing

def view_shows_chains(space):
    """Whether the overlay is drawn in ``space``: a 3D view with its overlays shown and X-ray off (X-ray has
    its own switch in wireframe shading, and none in Material Preview and Rendered)."""
    if space is None or space.type != 'VIEW_3D' or not space.overlay.show_overlays:
        return False
    shading = space.shading
    if shading.type == 'WIREFRAME':
        return not shading.show_xray_wireframe
    return shading.type != 'SOLID' or not shading.show_xray


def bone_lines_shown(space, view_layer, armature_object):
    """Whether the chain bones' lines are drawn in ``space`` (a 3D view the overlay is drawn in): with its Bones
    overlay on, for an armature visible there (``view_layer``, the viewport's hidden collections and objects,
    and its local view when it has one). The tint doesn't depend on it."""
    if not space.overlay.show_bones:
        return False
    if not armature_object.visible_get(view_layer=view_layer, viewport=space):
        return False
    return space.local_view is None or armature_object.local_view_get(space)


@contextmanager
def _drawing_state(state):
    """Blend by alpha, depth-test LESS_EQUAL without writing depth, through ``state`` (`gpu.state`), and put
    blend, depth test, depth mask and line width back afterwards, also when drawing fails."""
    try:
        state.blend_set('ALPHA')
        state.depth_mask_set(False)
        state.depth_test_set('LESS_EQUAL')
        yield
    finally:
        state.depth_test_set('NONE')
        state.depth_mask_set(True)
        state.blend_set('NONE')
        state.line_width_set(1.0)


def _draw():
    if bpy.app.background:
        return
    context = bpy.context
    if not _showing(context.window_manager):
        _tints.clear()
        _lines.clear()
        return
    if not view_shows_chains(context.space_data):
        return
    armature = follow_current(context)
    if armature is None:
        return
    meshes = overlay_meshes(context, armature)
    _refresh(context, armature, meshes)
    tints = _layer_tints(context)
    lines = None
    if bone_lines_shown(context.space_data, context.view_layer, armature):
        lines = _refresh_lines(context, armature)

    with _drawing_state(gpu.state):
        shader = gpu.shader.from_builtin('SMOOTH_COLOR')
        for obj in meshes:
            tint = tints.get(obj.session_uid)
            if tint is None or tint.colors is None or len(tint.colors) != len(tint.co):
                continue
            if tint.batch is None:
                positions, colors, indices = tint_arrays(tint)
                tint.batch = batch_for_shader(shader, 'TRIS', {"pos": positions, "color": colors},
                                              indices=indices) if len(indices) else False
            if tint.batch:
                tint.batch.draw(shader)
        if lines is None:
            return
        # The chain bones, in front of everything, each over a wider black line.
        shader = gpu.shader.from_builtin('POLYLINE_SMOOTH_COLOR')
        if lines.batch is None:
            lines.batch = lines.underlay = False
            if len(lines.positions):
                lines.batch = batch_for_shader(shader, 'LINES', {"pos": lines.positions, "color": lines.colors})
                lines.underlay = batch_for_shader(shader, 'LINES', {"pos": lines.positions,
                                                                    "color": line_underlay(lines.colors)})
        if lines.batch:
            gpu.state.depth_test_set('ALWAYS')
            pixel = context.preferences.system.pixel_size
            shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
            shader.uniform_float("lineWidth", (LINE_WIDTH + UNDERLAY_WIDTH) * pixel)
            lines.underlay.draw(shader)
            shader.uniform_float("lineWidth", LINE_WIDTH * pixel)
            lines.batch.draw(shader)


# -----------------------------------------------------------------------------
# Files and undo

@persistent
def _on_save_pre(_dummy=None):
    # The file keeps each armature's In Front as it was before Show Chains.
    _restore_in_front()


@persistent
def _on_save_post(_dummy=None):
    _set_in_front(dict.fromkeys(_in_front, True))


@persistent
def _on_load(_dummy=None):
    # The armatures of the file before are gone; the toggle is off in a file just opened.
    _in_front.clear()
    _restored.clear()
    _pending.clear()
    clear()
    wm = bpy.context.window_manager
    if _showing(wm):
        wm.magic_fit_show_chains = False


@persistent
def _on_undo_redo(_scene=None, _depsgraph=None):
    # An undo step keeps In Front as it was then: while on, the shown armatures stay in front; while off, they
    # keep what off gave back (else a file saved after it keeps In Front on).
    if _showing(bpy.context.window_manager):
        _set_in_front(dict.fromkeys(_in_front, True))
    else:
        _set_in_front(_restored)
    # The meshes and armatures may have changed in any way.
    for tints in _tints.values():
        for tint in tints.values():
            tint.moved = tint.reweigh = True
    for lines in _lines.values():
        lines.moved = True


_HANDLERS = (
    (bpy.app.handlers.depsgraph_update_post, _on_depsgraph_update),
    # A new frame (scrubbing, playback) sends no depsgraph update, but its depsgraph lists what changed.
    (bpy.app.handlers.frame_change_post, _on_depsgraph_update),
    (bpy.app.handlers.save_pre, _on_save_pre),
    (bpy.app.handlers.save_post, _on_save_post),
    (bpy.app.handlers.load_post, _on_load),
    (bpy.app.handlers.undo_post, _on_undo_redo),
    (bpy.app.handlers.redo_post, _on_undo_redo),
)


def register():
    global _handle
    _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
    for handlers, callback in _HANDLERS:
        handlers.append(callback)


def unregister():
    global _handle, _tag_due
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
    for handlers, callback in _HANDLERS:
        if callback in handlers:
            handlers.remove(callback)
    for timer in (_tag_views, _apply_pending_in_front):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    _tag_due = None
    _restore_in_front()
    # The toggle's value would outlive its property and be back on when the add-on is turned on again.
    for wm in bpy.data.window_managers:
        wm.property_unset("magic_fit_show_chains")
    _in_front.clear()
    _restored.clear()
    _pending.clear()
    clear()
