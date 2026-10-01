# SPDX-License-Identifier: GPL-3.0-or-later

"""Moving the selected vertices, with proportional editing that keeps to the mesh.

Blender's proportional editing reaches every vertex within its radius, so moving one side of a narrow gap
(the trouser leg on one inner thigh) moves the other side along with it. Connected Only keeps to the
mesh, but stops at split seams and at other parts that merely touch, and opens them up. `FitMove`
measures the radius along the mesh, across split seams and touching parts (vertices closer than the
seam distance) as if they were joined, so the other side of a gap stays put and seams stay closed. It
can also fade out toward the vertices that may not move (hidden ones): the vertices next to them don't
move at all, and the move eases in over the fade distance, so the parts stay joined.

Like `fitting`, nothing here depends on the 3D view: positions are world space.
"""

import numpy as np

from .fitting import WELD_DISTANCE, _mesh_distance, _weld

# Per vertex noise for the Random falloff, the same for every move.
_NOISE_SEED = 0


def proportional_weights(kind, t, noise=None):
    """Blender's proportional editing falloff ``kind`` (`ToolSettings.proportional_edit_falloff`) for
    distances ``t`` from the selection, as fractions of the proportional size; ``noise`` (0..1 per
    distance) for 'RANDOM'."""
    d = np.clip(1.0 - t, 0.0, 1.0)
    if kind == 'SHARP':
        return d * d
    if kind == 'SMOOTH':
        return d * d * (3.0 - 2.0 * d)
    if kind == 'ROOT':
        return np.sqrt(d)
    if kind == 'LINEAR':
        return d
    if kind == 'SPHERE':
        return np.sqrt(d * (2.0 - d))
    if kind == 'INVERSE_SQUARE':
        return d * (2.0 - d)
    if kind == 'RANDOM':
        return (np.ones_like(d) if noise is None else noise) * d
    return np.where(t <= 1.0 + 1e-6, 1.0, 0.0)


class FitMove:
    """How far each vertex of a mesh goes when its ``selected`` vertices move, see the module.

    ``co`` holds world space positions, ``edges`` and ``tris`` (optional, for distances straight across
    triangles) index them. Vertices that aren't ``movable`` (hidden ones) stay put. Selected vertices go
    all the way, and with ``proportional_size`` the vertices around them part of it, by Blender's
    ``falloff`` curve for their distance along the mesh. Vertices closer together than
    ``weld_distance`` move alike; such a group counts as selected when one of them is, and stays put
    when one of them may not move.

    With ``fade_distance``, vertices next to ones that may not move stay put too, and the others ease in
    from there (smoothstep over ``fade_distance`` along the mesh), so hidden and visible parts stay
    joined, however often they are moved.
    """

    def __init__(
            self, co, edges, selected, movable, *,
            tris=None, proportional_size=None, falloff='SMOOTH', weld_distance=WELD_DISTANCE, fade_distance=0.0,
    ):
        self.start = np.array(co, dtype=np.float64)
        self.weld, count, welded = _weld(self.start, edges, weld_distance)
        self.points = np.empty((count, 3))
        self.points[self.weld] = self.start
        self.fixed = np.zeros(count, dtype=bool)
        self.fixed[self.weld[~movable]] = True
        self.grabbed = np.zeros(count, dtype=bool)
        self.grabbed[self.weld[selected]] = True
        self.grabbed &= ~self.fixed
        # Edges and triangles among the vertices that may move: proportional editing doesn't reach
        # through hidden parts.
        self.edges = welded[~self.fixed[welded].any(axis=1)]
        self.tris = None
        if tris is not None and len(tris):
            tris = self.weld[np.asarray(tris, dtype=np.int64)]
            self.tris = tris[~self.fixed[tris].any(axis=1)]
        self.noise = np.random.default_rng(_NOISE_SEED).random(count)

        # How much of the move each group keeps near the vertices that may not move.
        self.fade = np.ones(count)
        if fade_distance > 0.0 and self.fixed.any():
            held = np.zeros(count, dtype=bool)
            edge_fixed = self.fixed[welded]
            held[welded[edge_fixed[:, 0] & ~edge_fixed[:, 1], 1]] = True
            held[welded[edge_fixed[:, 1] & ~edge_fixed[:, 0], 0]] = True
            if held.any():
                distance, _ = _mesh_distance(self.points, self.edges, held, fade_distance, self.tris)
                t = np.minimum(distance / fade_distance, 1.0)
                self.fade = t * t * (3.0 - 2.0 * t)
        self.set_proportional(proportional_size, falloff)

    def set_proportional(self, size, falloff='SMOOTH'):
        """Proportional editing with this size (world space distance along the mesh), or off (None)."""
        self.size = size
        if size is None or size <= 0.0:
            share = self.grabbed.astype(np.float64)
        else:
            distance, _ = _mesh_distance(self.points, self.edges, self.grabbed, size, self.tris)
            share = proportional_weights(falloff, distance / size, self.noise)
            share[self.grabbed] = 1.0
        share = np.where(self.fixed, 0.0, share * self.fade)
        # Per vertex, and the vertices that move at all.
        self.share = share[self.weld]
        self.indices = np.flatnonzero(self.share > 0.0)

    def positions(self, translation, indices=None):
        """World space positions of vertices ``indices`` (by default: those that move) after moving the
        selection by ``translation``."""
        if indices is None:
            indices = self.indices
        return self.start[indices] + self.share[indices, None] * np.asarray(translation, dtype=np.float64)
