# SPDX-License-Identifier: GPL-3.0-or-later

"""Marks where the meshes being worked on clip into the body.

A viewport overlay. Where a mesh is inside the body (or closer to it than a margin), a colored patch
is drawn on the body's surface, right where the body pokes through. Positions are the displayed ones,
so poses, bone scaling, shape keys and modifiers all count: scale a bone and the marks follow.

Distances to the body are only measured again for vertices that moved, and only for a limited time
per redraw, so large changes (a new pose) are worked through over a few redraws.
"""

import time

import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

from .fitting import BodySurface
from .painting import read_geometry

# Seconds per redraw spent measuring distances; what's left is measured in the next redraws.
TIME_BUDGET = 0.03
# Vertices measured between looks at the clock.
CHUNK = 2000
# Marks float this fraction of the body's edge length above the surface they're drawn on, so they
# don't flicker with it.
LIFT = 0.05
# Opacity of marks where the mesh is outside the body but closer than the margin, relative to clipping.
CLOSE_OPACITY = 0.4


class _Marks:
    """Clipping state of one mesh object."""

    __slots__ = ("co", "tris", "distance", "normal", "pending", "batch", "inside", "close")

    def __init__(self, co, tris):
        count = len(co)
        self.co = co                                  # world space positions, as displayed
        self.tris = tris
        self.distance = np.full(count, np.inf)        # signed distance to the body, negative inside
        self.normal = np.zeros((count, 3))            # the body's normal at the closest point
        self.pending = np.ones(count, dtype=bool)     # distance still to be measured
        self.batch = None                             # what's drawn, rebuilt after changes
        self.inside = 0                               # vertices inside the body
        self.close = 0                                # vertices outside, but closer than the margin


class _State:
    """Clipping state of one view layer of a scene. Windows can show different ones side by side."""

    __slots__ = ("marks", "dirty", "body", "surface", "style_changed")

    def __init__(self):
        self.marks = {}            # object name -> _Marks
        self.dirty = set()         # names of objects whose shape or placement changed since the last redraw
        self.body = None           # name of the body object
        self.surface = None        # its BodySurface
        self.style_changed = True


_states = {}         # (scene name, view layer name) -> _State
_handle = None


def _state(context):
    """The clipping state of the view layer in ``context``, made on first use."""
    key = (context.scene.name, context.view_layer.name)
    state = _states.get(key)
    if state is None:
        # Forget scenes and view layers that were renamed or deleted since.
        for old in list(_states):
            scene = bpy.data.scenes.get(old[0])
            if scene is None or scene.view_layers.get(old[1]) is None:
                del _states[old]
        state = _states[key] = _State()
    return state


def objects_to_check(context, body):
    """The meshes whose clipping is shown: those in the current Edit, Paint or Sculpt mode, or the
    selected ones in Object and Pose Mode. Never the body itself."""
    view_layer = context.view_layer
    active = view_layer.objects.active
    mode = active.mode if active is not None else 'OBJECT'
    result = []
    for obj in view_layer.objects:
        if obj.type != 'MESH' or obj == body or not obj.visible_get(view_layer=view_layer):
            continue
        if obj.mode == mode if mode not in {'OBJECT', 'POSE'} else obj.select_get(view_layer=view_layer):
            result.append(obj)
    return result


def status(context):
    """(object name, vertices inside the body, vertices closer than the margin, still measuring) for
    the meshes being checked, or None while there is nothing to check against."""
    scene = context.scene
    body = scene.magic_fit.target
    state = _states.get((scene.name, context.view_layer.name))
    if body is None or body.type != 'MESH' or state is None or state.body != body.name:
        return None
    result = []
    for obj in objects_to_check(context, body):
        marks = state.marks.get(obj.name)
        if marks is not None:
            result.append((obj.name, marks.inside, marks.close, bool(marks.pending.any())))
    return result


def style_changed(_self=None, _context=None):
    """Property update callback: settings that change how marks look (but not the distances)."""
    for state in _states.values():
        state.style_changed = True
    _redraw_views()


def clear():
    _states.clear()


# -----------------------------------------------------------------------------
# Keeping up with changes

@persistent
def _on_depsgraph_update(_scene, depsgraph):
    if not _states:
        return
    for update in depsgraph.updates:
        if update.is_updated_geometry or update.is_updated_transform:
            if isinstance(update.id, bpy.types.Object):
                for state in _states.values():
                    state.dirty.add(update.id.name)


@persistent
def _on_load(_dummy=None):
    clear()


def _redraw_views():
    """Redraw the 3D views soon (from a timer: tagging redraws while drawing isn't safe)."""
    if not bpy.app.timers.is_registered(_tag_views):
        bpy.app.timers.register(_tag_views, first_interval=0.01)


def _tag_views():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    return None


def _refresh(state, context, body_obj, settings):
    """Bring the marks of ``state`` up to date. Returns True while distances are still being measured."""
    depsgraph = context.evaluated_depsgraph_get()
    if state.body != body_obj.name or body_obj.name in state.dirty or state.surface is None:
        surface = BodySurface.from_object(body_obj, depsgraph)
        if surface is not state.surface:
            # A new body shape: every distance has to be measured again.
            for marks in state.marks.values():
                marks.pending[:] = True
        state.body, state.surface = body_obj.name, surface
    surface = state.surface
    if surface is None:
        return False

    objects = objects_to_check(context, body_obj)
    names = {obj.name for obj in objects}
    for name in list(state.marks):
        if name not in names:
            del state.marks[name]
    for obj in objects:
        marks = state.marks.get(obj.name)
        if marks is not None and obj.name not in state.dirty:
            continue
        geometry = read_geometry(obj, depsgraph, source='DISPLAYED')
        if marks is None or len(marks.co) != len(geometry.co):
            state.marks[obj.name] = _Marks(geometry.co, geometry.tris)
            continue
        moved = (geometry.co != marks.co).any(axis=1)
        if moved.any() or not np.array_equal(marks.tris, geometry.tris):
            marks.pending |= moved
            marks.co = geometry.co
            marks.tris = geometry.tris
            marks.batch = None
    state.dirty.clear()

    deadline = time.perf_counter() + TIME_BUDGET
    for marks in state.marks.values():
        while time.perf_counter() < deadline:
            todo = np.flatnonzero(marks.pending)[:CHUNK]
            if not len(todo):
                break
            near = surface.nearest(marks.co[todo])
            # Past an open edge of the body (below a torso on its own, say) there's nothing to clip into.
            marks.distance[todo] = np.where(near.found & ~near.beyond, near.distance, np.inf)
            marks.normal[todo] = near.normal
            marks.pending[todo] = False
            marks.batch = None
    return any(marks.pending.any() for marks in state.marks.values())


# -----------------------------------------------------------------------------
# Drawing

def _build(marks, settings, lift):
    """The batch of colored triangles around the clipping vertices, and the counts shown in the panel."""
    distance = marks.distance
    inside = distance < 0.0
    close = ~inside & (distance < settings.margin)
    counts = (int(inside.sum()), int(close.sum()))
    changed = counts != (marks.inside, marks.close)
    marks.inside, marks.close = counts
    marked = inside | close
    tris = marks.tris[marked[marks.tris].any(axis=1)] if len(marks.tris) else marks.tris
    if not len(tris):
        marks.batch = False
        return changed
    verts, corners = np.unique(tris, return_inverse=True)
    # Inside the body, a vertex is drawn where the body's surface is (that's what shows); outside,
    # where it is. Either way lifted a little off that surface.
    depth = np.where(np.isfinite(distance[verts]), np.maximum(-distance[verts], 0.0) + lift, 0.0)
    position = marks.co[verts] + depth[:, None] * marks.normal[verts]
    color = np.empty((len(verts), 4), dtype=np.float32)
    color[:, :3] = settings.color[:3]
    color[:, 3] = settings.color[3] * np.where(inside[verts], 1.0, np.where(close[verts], CLOSE_OPACITY, 0.0))
    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    marks.batch = batch_for_shader(
        shader, 'TRIS',
        {"pos": position.astype(np.float32), "color": color},
        indices=corners.reshape(-1, 3).astype(np.int32),
    )
    return changed


def _draw():
    context = bpy.context
    scene = context.scene
    settings = getattr(scene, "magic_fit_clipping", None)
    if settings is None or not settings.show:
        # Only this scene's marks: other windows may show scenes that have them on.
        for key in [key for key in _states if key[0] == scene.name]:
            del _states[key]
        return
    space = context.space_data
    if space is None or space.type != 'VIEW_3D' or not space.overlay.show_overlays:
        return
    body = scene.magic_fit.target
    if body is None or body.type != 'MESH':
        return

    state = _state(context)
    measuring = _refresh(state, context, body, settings)
    surface = state.surface
    if surface is None:
        return
    lift = LIFT * surface.edge_length
    counts_changed = False
    for marks in state.marks.values():
        if marks.batch is None or state.style_changed:
            counts_changed |= _build(marks, settings, lift)
    state.style_changed = False
    if measuring or counts_changed:
        # Keep measuring, and show the new counts in the sidebar.
        _redraw_views()

    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_mask_set(False)
    gpu.state.depth_test_set('ALWAYS' if settings.show_in_front else 'LESS_EQUAL')
    for marks in state.marks.values():
        if marks.batch:
            marks.batch.draw(shader)
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set('NONE')


def register():
    global _handle
    _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    # A new frame (scrubbing, playback) sends no depsgraph update, but its depsgraph lists what changed.
    bpy.app.handlers.frame_change_post.append(_on_depsgraph_update)
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
    for handlers, callback in (
            (bpy.app.handlers.depsgraph_update_post, _on_depsgraph_update),
            (bpy.app.handlers.frame_change_post, _on_depsgraph_update),
            (bpy.app.handlers.load_post, _on_load),
    ):
        if callback in handlers:
            handlers.remove(callback)
    if bpy.app.timers.is_registered(_tag_views):
        bpy.app.timers.unregister(_tag_views)
    clear()
