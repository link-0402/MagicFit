# SPDX-License-Identifier: GPL-3.0-or-later

"""Matching a body's sliders to another body: the Mannequin's chest keys ('-- Squeeze', '-- Push-Up'...) to a body
imported from a mod that lies on it.

A body mod is a devkit body with its sliders set some way, exported. Imported again (Instant Edit, a .mdl), it lies
on the devkit's body, which differs from it only where the sliders differ. A body's sliders are its shape keys
nothing drives: the devkit's script sets its sizes (chest size, Rue...) by drivers, and leaves the sliders of the
size it shows unmuted for the user (the other sizes' are muted). Each key moves the body's vertices along a fixed
offset times its value, so the body's shape is linear in the sliders' values, and they are found by least squares:
each vertex of the other body should lie on the body's surface, at the spot closest to it. That spot moves with the
values, so the fit is made again from where it got to, until it settles (point-to-plane ICP, with only the slider
values to find, each kept in its key's slider range).

Keys that don't move where the other body lies (the feet's Heels, for a torso) are left as they are. A body whose
keys nothing drives at all has every one as a slider.
"""

import numpy as np
from mathutils.bvhtree import BVHTree

from .resizing import ResizeError, _driven, _group_weights, _key_positions, _transform, _tris, _vertex_areas, rest_shape
from .sizing import GAME_KEY_PREFIX

# A vertex of the other body farther than this from the body is no part of it (another mesh joined to it).
MAX_GAP = 0.03
# A key moving no vertex of the body under the other body more than this is left as it is.
MIN_MOVE = 0.0001
# Values found are rounded to this, and those closer to 0 than SNAP set to 0 (left over from what the keys can't
# tell apart).
STEP = 0.01
SNAP = 0.02
# The fit is made again until no value changes more than this, at most ROUNDS times.
SETTLED = 1e-4
ROUNDS = 20
# Keys moving vertices alike can't be told apart: this much of their pull toward 0 keeps their values from running
# off (relative to how much the keys move).
RIDGE = 1e-6
# Fewer vertices of the other body than this on what the sliders move tell nothing.
MIN_VERTICES = 20
# A key moving no vertex more than this (object space) moves nothing.
NOISE = 1e-6
# A key whose new value fits the other body less than this much better (root mean square of the gaps) than its
# value now stays as it is: the fit can't tell (a nipple's Nip Nops, where the mod's nipple differs anyway).
NO_BETTER = 0.01


class SliderMatch:
    """What `match_sliders` found: the sliders' ``values`` ({key name: value}, those that change), ``before``
    (their values now), and how far the other body's vertices lie from the body where the sliders move it (mean
    over their area and largest, in meters) with the sliders as they are (``gap_before``, ``worst_before``) and
    with the values found (``gap``, ``worst``). ``sliders`` names every key that was fitted."""

    def __init__(self, sliders, before, values, gaps_before, gaps):
        self.sliders = sliders
        self.before = before
        self.values = values
        self.gap_before, self.worst_before = gaps_before
        self.gap, self.worst = gaps


def sliders(obj):
    """The shape keys of mesh ``obj`` that are its sliders: nothing drives their values, they aren't muted, the game
    doesn't turn them on itself, and they move something."""
    keys = obj.data.shape_keys if obj is not None and obj.type == 'MESH' else None
    if keys is None:
        return []
    result = []
    for block in keys.key_blocks:
        if (block == keys.reference_key or block.mute or block.name.lower().startswith(GAME_KEY_PREFIX)
                or _driven(keys, block, "value") or block.relative_key == block):
            continue
        # (A devkit's section names, like 'OTHER:', are keys too, moving nothing but float noise.)
        if np.abs(_key_positions(block) - _key_positions(block.relative_key)).max() > NOISE:
            result.append(block.name)
    return result


def _surface(obj):
    """World space rest shape of mesh ``obj`` (its keys mixed as they are) and its triangles."""
    return _transform(rest_shape(obj), obj.matrix_world), _tris(obj.data)


class _Fit:
    """The body's shape as its sliders' values make it, and the other body's vertices to fit it to."""

    def __init__(self, body, names, other):
        keys = body.data.shape_keys
        matrix = np.array(body.matrix_world, dtype=np.float64)
        self.names = names
        self.blocks = [keys.key_blocks[name] for name in names]
        self.base = _transform(rest_shape(body, {name: 0.0 for name in names}), matrix)
        moves = []
        for block in self.blocks:
            delta = _key_positions(block) - _key_positions(block.relative_key)
            weights = _group_weights(body, block.vertex_group) if block.vertex_group else None
            if weights is not None:
                delta *= weights[:, None]
            moves.append(delta @ matrix[:3, :3].T)
        # Vertex, key, axis.
        self.moves = np.stack(moves, axis=1)
        self.tris = _tris(body.data)
        self.points, other_tris = _surface(other)
        self.area = (_vertex_areas(self.points, other_tris) if len(other_tris)
                     else np.ones(len(self.points)))

    def shape(self, values):
        return self.base + np.einsum("vkd,k->vd", self.moves, values)

    def nearest(self, values):
        """For each of the other body's vertices, its closest spot on the body with ``values``: (the vertices
        found within MAX_GAP, their triangles, the spots' corner weights, normals and distances)."""
        co = self.shape(values)
        tree = BVHTree.FromPolygons(co.tolist(), self.tris.tolist(), all_triangles=True)
        found, tris, spots, normals, distances = [], [], [], [], []
        for index, point in enumerate(self.points.tolist()):
            location, normal, tri, distance = tree.find_nearest(point, MAX_GAP)
            if location is None:
                continue
            found.append(index)
            tris.append(tri)
            spots.append(location)
            normals.append(normal)
            distances.append(distance)
        found = np.array(found, dtype=np.int64)
        tris = self.tris[np.array(tris, dtype=np.int64)].reshape(-1, 3)
        spots = np.array(spots, dtype=np.float64).reshape(-1, 3)
        return found, tris, _corner_weights(co[tris], spots), np.array(normals).reshape(-1, 3), np.array(distances)

    def rows(self, values):
        """The fit's equations for ``values``: (A, y, vertices, distances), A @ values ~ y along each spot's normal,
        for the vertices on what the sliders move."""
        found, tris, weights, normals, distances = self.nearest(values)
        # How each slider moves each spot: vertex, key, axis.
        moves = np.einsum("vc,vckd->vkd", weights, self.moves[tris])
        moved = np.abs(moves).max(axis=(1, 2)) > 0.0
        found, tris, weights, normals, distances, moves = (
            found[moved], tris[moved], weights[moved], normals[moved], distances[moved], moves[moved])
        base = np.einsum("vc,vcd->vd", weights, self.base[tris])
        a = np.einsum("vkd,vd->vk", moves, normals)
        y = np.einsum("vd,vd->v", self.points[found] - base, normals)
        return a, y, found, distances, moves


def _corner_weights(corners, spots):
    """Barycentric weights of ``spots`` in their triangles (``corners``: spot, corner, axis)."""
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    v0, v1, v2 = b - a, c - a, spots - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    safe = np.where(np.abs(denom) > 1e-30, denom, 1.0)
    v = np.where(np.abs(denom) > 1e-30, (d11 * d20 - d01 * d21) / safe, 1.0 / 3.0)
    w = np.where(np.abs(denom) > 1e-30, (d00 * d21 - d01 * d20) / safe, 1.0 / 3.0)
    weights = np.clip(np.column_stack((1.0 - v - w, v, w)), 0.0, 1.0)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)


def _bounded_least_squares(a, y, low, high, start):
    """``x`` between ``low`` and ``high`` making ``a @ x`` closest to ``y`` (projected gradient on the normal
    equations; a few keys, so cheap)."""
    h = a.T @ a
    g = a.T @ y
    h = h + RIDGE * max(float(np.trace(h)) / max(len(h), 1), 1e-30) * np.eye(len(h))
    step = 1.0 / max(float(np.linalg.eigvalsh(h).max()), 1e-30)
    x = np.clip(start, low, high)
    for _ in range(20000):
        after = np.clip(x - step * (h @ x - g), low, high)
        if np.abs(after - x).max() < 1e-10:
            return after
        x = after
    return x


def _keep_unneeded(a, y, values, now):
    """``values`` with the keys whose change from ``now`` fits barely better (NO_BETTER, by the fit's equations
    ``a``, ``y``) put back, the least moving first."""
    values = values.copy()

    def rms(x):
        return float(np.sqrt(np.mean((a @ x - y) ** 2))) if len(y) else 0.0

    reach = np.abs(values - now) * np.linalg.norm(a, axis=0)
    for index in np.argsort(reach):
        if values[index] == now[index]:
            continue
        kept = values.copy()
        kept[index] = now[index]
        if rms(kept) <= rms(values) * (1.0 + NO_BETTER) + 1e-9:
            values = kept
    return values


def _gaps(fit, values, vertices):
    """(Mean over their area, largest) distance of the other body's ``vertices`` from the body with ``values``."""
    found, _tris, _weights, _normals, distances = fit.nearest(values)
    distance = np.full(len(fit.points), MAX_GAP)
    distance[found] = distances
    distance = distance[vertices]
    area = fit.area[vertices]
    return float((distance * area).sum() / max(float(area.sum()), 1e-30)), float(distance.max())


def match_sliders(body, other):
    """The values of the sliders of mesh ``body`` (see `sliders`) that make it take the shape of mesh ``other``, a
    body lying on it (a body mod exported from the same devkit, imported): a SliderMatch. Nothing is changed (see
    `apply`). Raises ResizeError when it can't be done."""
    if body is None or body.type != 'MESH':
        raise ResizeError("Pick the body whose sliders to set")
    if other is None or other.type != 'MESH':
        raise ResizeError("Pick the body to match")
    if other == body:
        raise ResizeError("Pick another body to match than '{:s}'".format(body.name))
    names = sliders(body)
    if not names:
        raise ResizeError("'{:s}' has no sliders: shape keys nothing drives".format(body.name))
    if body.data.shape_keys is not None and not body.data.shape_keys.use_relative:
        raise ResizeError("'{:s}' uses absolute shape keys".format(body.name))
    fit = _Fit(body, names, other)
    blocks = fit.blocks
    now = np.array([block.value for block in blocks], dtype=np.float64)
    a, y, vertices, _distances, moves = fit.rows(now)
    if len(vertices) < MIN_VERTICES:
        raise ResizeError("'{:s}' doesn't lie on what the sliders of '{:s}' move".format(other.name, body.name))
    # Keys that move nothing under the other body stay as they are.
    used = np.linalg.norm(moves, axis=2).max(axis=0) > MIN_MOVE
    if not used.any():
        raise ResizeError("'{:s}' doesn't lie on what the sliders of '{:s}' move".format(other.name, body.name))
    low = np.array([block.slider_min for block in blocks])
    high = np.array([block.slider_max for block in blocks])
    low[~used] = high[~used] = now[~used]
    values = now.copy()
    for _round in range(ROUNDS):
        after = _bounded_least_squares(a, y, low, high, values)
        settled = np.abs(after - values).max() < SETTLED
        values = after
        if settled:
            break
        a, y, _vertices, _distances, _moves = fit.rows(values)
    rounded = np.round(values / STEP) * STEP
    rounded[np.abs(rounded) < SNAP] = 0.0
    rounded = np.clip(rounded, low, high)
    rounded[~used] = now[~used]
    rounded = _keep_unneeded(a, y, rounded, now)
    changed = {block.name: round(float(value), 6) for block, value, old in zip(blocks, rounded, now)
               if abs(value - old) > 1e-6}
    fitted = [block.name for block, use in zip(blocks, used) if use]
    before = {block.name: float(old) for block, old in zip(blocks, now)}
    return SliderMatch(fitted, before, changed, _gaps(fit, now, vertices), _gaps(fit, rounded, vertices))


def apply(body, match):
    """Set the sliders of ``body`` to the values ``match`` (a SliderMatch) found."""
    blocks = body.data.shape_keys.key_blocks
    for name, value in match.values.items():
        blocks[name].value = value
    body.update_tag()
