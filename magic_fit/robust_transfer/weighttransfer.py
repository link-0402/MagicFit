# This file is part of Robust Weight Transfer for Blender.
#
# Portions of this code are based on:
#   RobustSkinWeightsTransferCode (https://github.com/rin-23/RobustSkinWeightsTransferCode/blob/main/src/utils.py)
#   by Rinat Abdrashitov, used under the MIT License (see below).
#
# Changes were made to make the code compatible with Blender's data structures
# and to improve performance and robustness.
#
# Copyright (C) 2025 sentfromspacevr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Attribution: Developed by sentfromspacevr (https://github.com/sentfromspacevr)
#
# Changed for Magic Fit (2026-09): Crease Smoothing trusts each match by how far the target floats
# above the source and how well their normals agree (bridge_confidence), groups are limited without
# jumps (limit_groups_smoothly), and inpainting has a third mode, Combined (_link_loose_parts).
#
# ---- Original MIT License Notice Follows ----
#
# The following portions of this file are based on work by Rinat Abdrashitov and are licensed under the MIT License:
#
# Copyright (c) 2024 Rinat Abdrashitov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from dataclasses import dataclass

import numpy as np
import scipy as sp
import robust_laplacian
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree


def build_surface_bvh(V, F):
    """Build Blender's native acceleration structure for a triangle mesh."""
    V = np.asarray(V)
    F = np.asarray(F)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("Source vertices must be an N by 3 array")
    if F.ndim != 2 or F.shape[1] != 3 or len(F) == 0:
        raise ValueError("Source mesh has no triangles")
    if np.any(F < 0) or np.any(F >= len(V)):
        raise ValueError("Source mesh contains invalid triangle indices")

    tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
    if tree is None:
        raise ValueError("Could not build a surface search tree for the source mesh")
    return tree


def _barycentric_coordinates(points, triangle_vertices):
    """Calculate barycentric coordinates for corresponding points/triangles."""
    a = triangle_vertices[:, 0]
    ab = triangle_vertices[:, 1] - a
    ac = triangle_vertices[:, 2] - a
    ap = points - a

    d00 = np.einsum('ij,ij->i', ab, ab)
    d01 = np.einsum('ij,ij->i', ab, ac)
    d11 = np.einsum('ij,ij->i', ac, ac)
    d20 = np.einsum('ij,ij->i', ap, ab)
    d21 = np.einsum('ij,ij->i', ap, ac)
    denominator = d00 * d11 - d01 * d01

    barycentric = np.zeros((len(points), 3), dtype=np.float64)
    scale = np.maximum(d00 * d11, np.finfo(np.float64).tiny)
    valid = np.abs(denominator) > np.finfo(np.float64).eps * scale
    if np.any(valid):
        barycentric[valid, 1] = (
            d11[valid] * d20[valid] - d01[valid] * d21[valid]
        ) / denominator[valid]
        barycentric[valid, 2] = (
            d00[valid] * d21[valid] - d01[valid] * d20[valid]
        ) / denominator[valid]
        barycentric[valid, 0] = 1.0 - barycentric[valid, 1] - barycentric[valid, 2]

    if np.any(~valid):
        # Degenerate faces do not have unique barycentric coordinates. Assign
        # the result to the face vertex nearest the BVH hit point.
        degenerate_triangles = triangle_vertices[~valid]
        degenerate_points = points[~valid, np.newaxis, :]
        offsets = degenerate_triangles - degenerate_points
        nearest = np.argmin(
            np.einsum('ijk,ijk->ij', offsets, offsets),
            axis=1,
        )
        barycentric[np.flatnonzero(~valid), nearest] = 1.0

    return barycentric


def find_closest_point_on_surface(P, V, F, surface_bvh=None):
    """
    Given a number of points find their closest points on the surface of the V,F mesh

    Args:
        P: #P by 3, where every row is a point coordinate
        V: #V by 3 mesh vertices
        F: #F by 3 mesh triangles indices
    Returns:
        sqrD #P smallest squared distances
        I #P primitive indices corresponding to smallest distances
        C #P by 3 closest points
        B #P by 3 of the barycentric coordinates of the closest point
    """
    
    P = np.asarray(P)
    V = np.asarray(V)
    F = np.asarray(F)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Target vertices must be an N by 3 array")
    if surface_bvh is None:
        surface_bvh = build_surface_bvh(V, F)

    sqrD = np.empty(len(P), dtype=np.float64)
    I = np.empty(len(P), dtype=np.int64)
    C = np.empty((len(P), 3), dtype=np.float64)
    for index, point in enumerate(P):
        closest, _normal, triangle_index, distance = surface_bvh.find_nearest(point)
        if closest is None or triangle_index is None or distance is None:
            raise ValueError("Could not find a closest point on the source mesh")
        sqrD[index] = distance * distance
        I[index] = triangle_index
        C[index] = closest

    B = _barycentric_coordinates(C, V[F[I]])

    return sqrD,I,C,B

def interpolate_attribute_from_bary(A,B,I,F):
    """
    Interpolate per-vertex attributes A via barycentric coordinates B of the F[I,:] vertices

    Args:
        A: #V by N per-vertex attributes
        B  #B by 3 array of the barycentric coordinates of some points
        I  #B primitive indices containing the closest point
        F: #F by 3 mesh triangle indices
    Returns:
        A_out #B interpolated attributes
    """
    F_closest = F[I,:]
    a1 = A[F_closest[:,0],:]
    a2 = A[F_closest[:,1],:]
    a3 = A[F_closest[:,2],:]

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    b1 = b1.reshape(-1,1)
    b2 = b2.reshape(-1,1)
    b3 = b3.reshape(-1,1)
    
    A_out = a1*b1 + a2*b2 + a3*b3

    return A_out


@dataclass
class SurfaceMatch:
    """Closest-surface data for every target vertex."""
    weights: np.ndarray
    distances: np.ndarray
    # Degrees between the target and interpolated source normals, folded into
    # [0, 90] when flipped normals are allowed. NaN where a normal is undefined.
    angles: np.ndarray
    # Closest source points and the unit source normals there (zero where the
    # interpolated normal vanishes).
    points: np.ndarray = None
    normals: np.ndarray = None


def match_closest_surface(source_verts, source_triangles, source_normals, target_verts,
                          target_normals, source_weights, flip_vertex_normal, surface_bvh=None):
    """Interpolate source weights and normals at each target vertex's closest surface point."""
    sqrD, I, C, B = find_closest_point_on_surface(
        target_verts, source_verts, source_triangles, surface_bvh
    )
    weights = interpolate_attribute_from_bary(source_weights, B, I, source_triangles)
    source_normals = interpolate_attribute_from_bary(source_normals, B, I, source_triangles)
    target_normals = np.asarray(target_normals, dtype=np.float64)

    source_norm = np.linalg.norm(source_normals, axis=1)
    target_norm = np.linalg.norm(target_normals, axis=1)
    valid = np.logical_and(source_norm > np.finfo(np.float32).eps,
                           target_norm > np.finfo(np.float32).eps)
    cosine = np.einsum('ij,ij->i', source_normals[valid], target_normals[valid])
    cosine /= source_norm[valid] * target_norm[valid]
    cosine = np.clip(cosine, -1.0, 1.0)
    if flip_vertex_normal:
        cosine = np.abs(cosine)
    angles = np.full(len(weights), np.nan)
    angles[valid] = np.degrees(np.arccos(cosine))
    unit_normals = np.zeros_like(source_normals, dtype=np.float64)
    source_valid = source_norm > np.finfo(np.float32).eps
    unit_normals[source_valid] = source_normals[source_valid] / source_norm[source_valid, np.newaxis]
    return SurfaceMatch(weights, np.sqrt(sqrD), angles, C, unit_normals)


def match_confidence(distances, angles, max_distance, max_angle, soft_angle):
    """Trust in each closest-surface match: 1 for direct matches, 0 for unused ones.

    Matches within max_distance whose normals differ by at most max_angle become
    exact constraints. Up to soft_angle the trust fades smoothly, so steep or
    detailed geometry (plates, rims, frills) still anchors its loose part
    without forcing its weights like a direct match would.
    """
    distances = np.asarray(distances, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    usable = (distances <= max_distance) & np.isfinite(angles)
    confidence = np.zeros(len(distances), dtype=np.float64)
    confidence[usable & (angles <= max_angle)] = 1.0
    band = usable & (angles > max_angle) & (angles < soft_angle)
    if np.any(band):
        t = (angles[band] - max_angle) / (soft_angle - max_angle)
        confidence[band] = 1 - t * t * (3 - 2 * t)
    return confidence


# Crease Smoothing: how far each closest-point match is trusted, by how far the target floats above
# the source there. A match d away pulls the weights towards it only on length scales above
# BRIDGE_SOFTNESS times d, so where the target lies on the source it copies its weights, and where it
# spans a crease (the butt cleft, between the legs, under the bust) the weights of both sides blend
# smoothly across it, over about BRIDGE_SOFTNESS times the height it floats.
BRIDGE_SOFTNESS = 3.0
# Even touching vertices are smoothed below BRIDGE_SOFTNESS times this, so noise in the contact doesn't
# show.
BRIDGE_MIN_GAP = 0.002
# Every vertex keeps this much trust, which anchors every loose part. It's what a vertex floating about
# 1.3 cm high gets at softness 3, so blends get no wider than about 4 cm (softness times that height).
BRIDGE_FLOOR = 0.01
# Where the normals of target and source differ by more than the first angle (degrees, either way
# round), the target lifts off the source, like the flanks of a crease; beyond the second its match is
# only trusted as much as BRIDGE_FLOOR.
BRIDGE_FULL_ANGLE = 11.0
BRIDGE_ZERO_ANGLE = 25.0
# Within this distance of a free border (a hem or an opening that ends in a single layer, see
# free_border_vertices), the target keeps more trust, as if it floated this much less high: nothing
# beyond the border holds it, and a hem in front of a thigh has to follow it. The rim where a
# double-layer garment's layers meet doesn't count here (counting it measured worse); over the legs,
# Hem Follow handles those.
BRIDGE_BORDER_RADIUS = 0.04
BRIDGE_BORDER_SOFTNESS = 0.7
# Borders keep that trust while the normals differ by up to the first angle, fading out by the second,
# so a hem flaring off a thigh still follows it while a strap across a crease doesn't.
BRIDGE_BORDER_FULL_ANGLE = 25.0
BRIDGE_BORDER_ZERO_ANGLE = 45.0
# Coincident vertices (split seams) count as one for the trust and the solve.
BRIDGE_WELD = 1e-4
# The soft length Crease Smoothing solves with: a quarter of the default Max Distance. The pull of a
# match doesn't depend on it, only BRIDGE_FLOOR's and the border's do.
BRIDGE_SOFT_LENGTH = 0.0125


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def free_border_vertices(vertices, triangles, merge_map):
    """Vertices on free borders: on an open edge of the mesh, with no other open edge's vertex where
    merge_map puts them (split seams between loose parts, but also the rim where a double-layer
    garment's layers meet, see fold_aware_free_borders)."""
    F = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not len(F):
        return np.zeros(0, dtype=np.int64)
    edges = np.sort(np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    border = np.unique(edges[counts == 1].ravel())
    if not len(border):
        return border
    per_cluster = np.bincount(merge_map[border], minlength=int(merge_map.max()) + 1)
    return border[per_cluster[merge_map[border]] == 1]


def free_border_distance(vertices, triangles, merge_map):
    """Distance from each vertex to the nearest free border (free_border_vertices), inf without one."""
    V = np.asarray(vertices, dtype=np.float64)
    free = free_border_vertices(V, triangles, merge_map)
    if not len(free):
        return np.full(len(V), np.inf)
    return cKDTree(V[free]).query(V)[0]


def bridge_confidence(vertices, triangles, distances, cosines, soft_length=BRIDGE_SOFT_LENGTH,
                      softness=BRIDGE_SOFTNESS, weld_distance=BRIDGE_WELD, max_distance=np.inf):
    """Trust in each vertex's closest-point match for Crease Smoothing, in [BRIDGE_FLOOR, 1].

    The solver pulls a weakly matched vertex towards its match with its area times t / (1 - t) /
    soft_length^4 (see _soft_area and _solve_constrained). Here t / (1 - t) = (soft_length /
    (softness * d))^4, d the distance to the match, so the pull is area / (softness * d)^4 whatever
    soft_length is: the match only shapes the weights on length scales above softness * d.

    That trust fades out where the normals of target and source (cosines: |cos| of the angle between
    them, 0 where either is undefined) differ by BRIDGE_FULL_ANGLE to BRIDGE_ZERO_ANGLE: across a crease,
    the target lifts off the walls on both sides, and copying them would make it follow each wall and
    fold. Within BRIDGE_BORDER_RADIUS of a free border, the trust is at least that of a vertex floating
    BRIDGE_BORDER_SOFTNESS / softness times as high, fading out with the distance to the border and
    with normals differing by BRIDGE_BORDER_FULL_ANGLE to BRIDGE_BORDER_ZERO_ANGLE: a hem along a thigh
    follows it, while a strap across a crease still blends.
    Coincident vertices (within weld_distance, at least BRIDGE_WELD) share their normals' agreement, so
    split seams get the same input on both sides. Matches farther than max_distance only get the floor.
    """
    V = np.asarray(vertices, dtype=np.float64)
    d = np.maximum(np.asarray(distances, dtype=np.float64), BRIDGE_MIN_GAP)
    cosines = np.nan_to_num(np.abs(np.asarray(cosines, dtype=np.float64)), nan=0.0)
    if not (np.isfinite(soft_length) and soft_length > 0 and np.isfinite(softness) and softness > 0):
        raise ValueError("Crease Smoothing needs a positive soft length and Width")
    merge_map = find_vertex_merge_map(V, max(weld_distance, BRIDGE_WELD))
    sizes = np.bincount(merge_map).astype(np.float64)
    cosines = (np.bincount(merge_map, weights=cosines) / sizes)[merge_map]
    full, zero = np.cos(np.radians(BRIDGE_FULL_ANGLE)), np.cos(np.radians(BRIDGE_ZERO_ANGLE))
    agreement = _smoothstep((cosines - zero) / (full - zero))
    odds = (soft_length / (softness * d)) ** 4
    trust = np.maximum(odds / (1 + odds) * agreement, BRIDGE_FLOOR)
    near = 1 - free_border_distance(V, triangles, merge_map) / BRIDGE_BORDER_RADIUS
    if np.any(near > 0):
        border_odds = (soft_length / (BRIDGE_BORDER_SOFTNESS * d)) ** 4
        full, zero = np.cos(np.radians(BRIDGE_BORDER_FULL_ANGLE)), np.cos(np.radians(BRIDGE_BORDER_ZERO_ANGLE))
        border_agreement = _smoothstep((cosines - zero) / (full - zero))
        trust = np.maximum(trust, border_odds / (1 + border_odds) * _smoothstep(near) * border_agreement)
    trust[np.asarray(distances, dtype=np.float64) > max_distance] = BRIDGE_FLOOR
    trust[trust >= 0.999] = 1.0
    return trust


def limit_groups_smoothly(weights, limit, drop=0.5 / 255):
    """Keep at most ``limit`` groups per vertex without jumps between neighbours.

    Every weight gives up the (limit + 1)th largest one of its vertex, so a group fades out as it
    becomes the one too many instead of being cut off; weights below ``drop`` of the total (half a step
    of FFXIV's 8-bit weights by default) go too. Where that leaves nothing, at a tie between the groups
    around the cut, the vertex keeps its ``limit`` largest groups instead. Each vertex keeps its total.
    """
    original = np.clip(np.asarray(weights, dtype=np.float64), 0, None)
    totals = original.sum(axis=1)
    W = original.copy()
    if W.shape[1] > limit:
        W = np.clip(W - (-np.partition(-W, limit, axis=1)[:, limit])[:, np.newaxis], 0, None)
    W[W < drop * totals[:, np.newaxis]] = 0
    for row in np.flatnonzero((W.sum(axis=1) <= 0) & (totals > 0)):
        keep = np.argsort(-original[row], kind='stable')[:limit]
        W[row, keep] = original[row, keep]
        W[row, W[row] < drop * totals[row]] = 0
        if not W[row].any():
            W[row, keep[0]] = original[row, keep[0]]
    kept = W.sum(axis=1)
    np.divide(W * totals[:, np.newaxis], kept[:, np.newaxis], out=W, where=kept[:, np.newaxis] > 0)
    return W


# The layers of a fold lie back to back: each face reaches at most FOLD_BEHIND (the sine of 30°) behind
# the other along its normal, and no more than FOLD_FRONT in front of it. Faces meeting at a wider angle,
# or facing each other like two thighs pressed together below a crotch seam, go on past the edge.
FOLD_BEHIND = 0.5
FOLD_FRONT = 0.1


def fold_aware_free_borders(vertices, triangles, merge_map, seam_cos=0.5):
    """Vertices on free borders, counting the rim of a double-layer garment as one.

    FFXIV garments often have a back-face layer or lining meeting the outer one along the hem at
    coincident vertices. An open edge whose coincident partner edges lie on the same side, back to back
    (their faces fold back over it, see FOLD_BEHIND), still ends the surface. A partner continuing the
    surface on the other side (a split seam), opening a wider wedge, or facing it (two surfaces pinched
    together, like thighs that touch below the crotch) makes it no border.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not len(F):
        return np.zeros(0, dtype=np.int64)
    edges = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    opposite = np.concatenate((F[:, 2], F[:, 0], F[:, 1]))
    key = np.sort(edges, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    first = np.r_[True, np.any(key[order][1:] != key[order][:-1], axis=1)]
    group = np.cumsum(first) - 1
    single = order[np.bincount(group)[group] == 1]
    a, b, c = edges[single, 0], edges[single, 1], opposite[single]
    along = V[b] - V[a]
    along /= np.maximum(np.linalg.norm(along, axis=1, keepdims=True), np.finfo(np.float64).tiny)
    inward = V[c] - V[a]
    inward -= np.einsum('ij,ij->i', inward, along)[:, np.newaxis] * along
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), np.finfo(np.float64).tiny)
    # Each face's normal, by its winding.
    normal = np.cross(V[b] - V[a], V[c] - V[a])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), np.finfo(np.float64).tiny)
    welded = np.sort(np.column_stack((merge_map[a], merge_map[b])), axis=1)
    order = np.lexsort((welded[:, 1], welded[:, 0]))
    starts = np.flatnonzero(np.r_[True, np.any(welded[order][1:] != welded[order][:-1], axis=1)])
    sizes = np.diff(np.r_[starts, len(order)])
    seam = np.zeros(len(single), dtype=bool)
    for start, size in zip(starts[sizes > 1], sizes[sizes > 1]):
        ids = order[start:start + size]
        dots = inward[ids] @ inward[ids].T
        # [i, j]: how far face i reaches in front of face j.
        front = inward[ids] @ normal[ids].T
        fold = (dots >= seam_cos) & (front >= -FOLD_BEHIND) & (front <= FOLD_FRONT)
        seam[ids[~(fold & fold.T).all(axis=1)]] = True
    return np.unique(np.concatenate((a[~seam], b[~seam])))


def forward_kinematics(rest, parents, rotations=None, scales=None):
    """Deform matrices (bones x 4 x 4) of a skeleton with rest matrices ``rest`` (in one space, such
    as world space) and parent indices ``parents`` (-1 for roots), for local rotations ({bone: 3x3} in
    the bone's rest axes, carried on to its children) and local scales ({bone: 3 factors} about its
    head in its rest axes, not carried on, like Customize+ without propagation)."""
    rest = np.asarray(rest, dtype=np.float64)
    count = len(rest)
    inverse = np.linalg.inv(rest)
    depth = np.zeros(count, dtype=np.int64)
    for index in range(count):
        parent = parents[index]
        while parent >= 0:
            depth[index] += 1
            parent = parents[parent]
    posed = np.empty_like(rest)
    for index in np.argsort(depth, kind='stable'):
        local = np.eye(4)
        if rotations and index in rotations:
            local[:3, :3] = rotations[index]
        parent = parents[index]
        base = rest[index] if parent < 0 else posed[parent] @ inverse[parent] @ rest[index]
        posed[index] = base @ local
    deform = posed @ inverse
    for index, factors in (scales or {}).items():
        deform[index] = posed[index] @ np.diag([factors[0], factors[1], factors[2], 1.0]) @ inverse[index]
    return deform


# Hem Follow (Crease Smoothing): where a hem lies over one leg, a membrane between the legs can let
# a swinging thigh through it. Four strides check it: each hip in turn flexed HEM_STRIDE[0] degrees
# forward with the other one HEM_STRIDE[1] back, plain and with the thighs and butt scaled as
# Customize+ templates often do.
HEM_LEGS = ('j_asi_a_l', 'j_asi_a_r')
HEM_KNEES = ('j_asi_b_l', 'j_asi_b_r')
HEM_STRIDE = (30.0, 20.0)
# Thighs 1.3 times as thick (not longer), butt bones 1.5 times as large.
HEM_THIGH_SCALE = 1.3
HEM_BUTT = ('iv_shiri_l', 'iv_shiri_r', 'ya_shiri_phys_l', 'ya_shiri_phys_r')
HEM_BUTT_SCALE = 1.5
# Hems: within HEM_BAND of a free border, over a leg (its closest-point weights belong to one leg by
# the first share or more, fully from the second), directly over it (straight in, the body is at most
# the first times as far as its closest point, fading out by the second).
HEM_BAND = 0.08
HEM_LEG_SHARE = (0.3, 0.6)
HEM_COVER = (1.25, 2.0)
HEM_COVER_REACH = 0.25
# A hem vertex has to keep this share of the gap it would keep with the closest-point weights, and at
# least HEM_MIN_GAP; the blend it needs is boosted by HEM_BOOST, and it's trusted up to HEM_TRUST.
HEM_MARGIN = 0.3
HEM_MIN_GAP = 0.001
HEM_BOOST = 1.5
HEM_TRUST = 0.95


def _rotation(axis, degrees):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle = np.radians(degrees)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def stride_poses(names, rest, parents, front, down=(0.0, 0.0, -1.0)):
    """Deform matrices of the four strides for a skeleton (bone ``names``, ``rest`` matrices,
    ``parents``), with the character facing ``front`` and its legs pointing ``down``; empty without
    both thighs (HEM_LEGS). ``rest`` can also be a pose, which the strides then start from."""
    rest = np.asarray(rest, dtype=np.float64)
    index = {name: i for i, name in enumerate(names)}
    if not all(name in index for name in HEM_LEGS):
        return []
    left, right = (index[name] for name in HEM_LEGS)
    front = np.asarray(front, dtype=np.float64)
    down = np.asarray(down, dtype=np.float64)
    # Turning the leg down towards the front, about the character's left-right axis.
    axis = np.cross(down, front)
    scales = {index[name]: (HEM_BUTT_SCALE,) * 3 for name in HEM_BUTT if name in index}
    for thigh, knee in zip(HEM_LEGS, HEM_KNEES):
        # Thicker across the leg: whichever of the thigh's axes runs along it (towards the knee, or the
        # bone itself without one) keeps its length.
        bone = index[thigh]
        along = (rest[index[knee]][:3, 3] - rest[bone][:3, 3]) if knee in index else rest[bone][:3, 1]
        factors = np.full(3, HEM_THIGH_SCALE)
        factors[int(np.argmax(np.abs(rest[bone][:3, :3].T @ along)))] = 1.0
        scales[bone] = tuple(factors)
    poses = []
    for scaled in (None, scales):
        for forward, back in ((left, right), (right, left)):
            rotations = {bone: _rotation(rest[bone][:3, :3].T @ axis, degrees)
                         for bone, degrees in ((forward, HEM_STRIDE[0]), (back, -HEM_STRIDE[1]))}
            poses.append(forward_kinematics(rest, parents, rotations, scaled))
    return poses


def _skin(points, weights, matrices):
    homogeneous = np.hstack((points, np.ones((len(points), 1))))
    out = np.zeros((len(points), 3))
    for column in np.flatnonzero(weights.any(axis=0)):
        rows = weights[:, column] > 0
        out[rows] += weights[rows, column, np.newaxis] * (homogeneous[rows] @ matrices[column].T)[:, :3]
    return out


def _skin_normals(normals, weights, matrices):
    out = np.zeros_like(normals)
    for column in np.flatnonzero(weights.any(axis=0)):
        rows = weights[:, column] > 0
        out[rows] += weights[rows, column, np.newaxis] * (normals[rows] @ np.linalg.inv(matrices[column][:3, :3]))
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), np.finfo(np.float64).tiny)


def _normalized_rows(weights):
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)


def hem_follow(vertices, triangles, merge_map, normals, points, point_normals, distances, matched,
               solved, leg_share, poses, surface_bvh):
    """How far (0 to 1) to pull each vertex from its ``solved`` weights towards its closest-point
    weights ``matched``, so that where a hem lies over a leg (``leg_share``: the larger share of the
    closest-point weights on either leg's bones) the leg can't swing through it in the ``poses``
    (column deform matrices, see stride_poses). ``points``/``point_normals``/``distances``: closest
    points on the source, its normals there and the distances to them; ``surface_bvh``: the source's
    BVH tree. Zero everywhere else."""
    V = np.asarray(vertices, dtype=np.float64)
    pull = np.zeros(len(V))
    if not poses:
        return pull
    free = np.union1d(free_border_vertices(V, triangles, merge_map),
                      fold_aware_free_borders(V, triangles, merge_map))
    if not len(free):
        return pull
    gate = _smoothstep(1 - cKDTree(V[free]).query(V)[0] / HEM_BAND)
    gate *= _smoothstep((np.asarray(leg_share) - HEM_LEG_SHARE[0]) / (HEM_LEG_SHARE[1] - HEM_LEG_SHARE[0]))
    candidates = np.flatnonzero(gate > 0)
    if not len(candidates):
        return pull
    # Straight in from the hem, is the body about as close as its closest point, or is that a gap?
    from mathutils import Vector
    offsets = V[candidates] - points[candidates]
    outward = normals[candidates] * np.where(np.einsum('ij,ij->i', offsets, normals[candidates]) < 0, -1.0, 1.0)[:, np.newaxis]
    ratio = np.full(len(candidates), np.inf)
    for row, (origin, direction) in enumerate(zip(V[candidates], outward)):
        hit = surface_bvh.ray_cast(Vector(origin), Vector(-direction), HEM_COVER_REACH)
        if hit[0] is not None:
            ratio[row] = hit[3] / max(distances[candidates[row]], 1e-4)
    gate[candidates] *= _smoothstep((HEM_COVER[1] - ratio) / (HEM_COVER[1] - HEM_COVER[0]))
    selected = np.flatnonzero(gate > 0)
    if not len(selected):
        return pull
    x, p = V[selected], points[selected]
    towards = np.where(np.einsum('ij,ij->i', x - p, point_normals[selected]) < 0, -1.0, 1.0)
    n = point_normals[selected] * towards[:, np.newaxis]
    follow, current = _normalized_rows(matched[selected]), _normalized_rows(solved[selected])
    needed = np.zeros(len(selected))
    for matrices in poses:
        body = _skin(p, follow, matrices)
        body_normals = _skin_normals(n, follow, matrices)
        keep = np.einsum('ij,ij->i', _skin(x, follow, matrices) - body, body_normals)
        gap = np.einsum('ij,ij->i', _skin(x, current, matrices) - body, body_normals)
        margin = np.maximum(HEM_MARGIN * keep, HEM_MIN_GAP)
        short = (gap < margin) & (keep - gap > 1e-5)
        fraction = np.zeros(len(selected))
        fraction[short] = np.clip((margin[short] - gap[short]) / (keep[short] - gap[short]), 0, 1)
        needed = np.maximum(needed, fraction)
    pull[selected] = np.clip(needed * HEM_BOOST, 0, 1) * gate[selected]
    return pull


def hem_follow_trust(confidence, pull):
    """The trust of vertices pulled by hem_follow: soft, up to HEM_TRUST once pulled by a fifth."""
    return np.where(pull > 1e-3, np.maximum(confidence, HEM_TRUST * _smoothstep(pull / 0.2)), confidence)


def _union_find_components(vertex_count, edges):
    """Return connected-component labels without scipy.sparse.csgraph."""
    parent = np.arange(vertex_count, dtype=np.int64)
    rank = np.zeros(vertex_count, dtype=np.uint8)

    def find(vertex):
        root = vertex
        while parent[root] != root:
            root = parent[root]
        while parent[vertex] != vertex:
            next_vertex = parent[vertex]
            parent[vertex] = root
            vertex = next_vertex
        return root

    for left, right in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root == right_root:
            continue
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    roots = np.fromiter((find(vertex) for vertex in range(vertex_count)),
                        dtype=np.int64, count=vertex_count)
    _roots, labels = np.unique(roots, return_inverse=True)
    return len(_roots), labels.astype(np.int64, copy=False)


def _connected_components(graph):
    """Use SciPy when available, with a dependency-light fallback."""
    try:
        return sp.sparse.csgraph.connected_components(graph, directed=False)
    except (ImportError, ModuleNotFoundError):
        coo = graph.tocoo()
        edges = np.column_stack((coo.row, coo.col))
        return _union_find_components(graph.shape[0], edges)


def find_vertex_merge_map(vertices, distance):
    """Map nearby vertices to non-destructive, transitively merged clusters."""
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("Cannot virtually merge invalid vertex coordinates")
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("Virtual merge distance must be finite and non-negative")
    vertex_count = len(vertices)
    identity = np.arange(vertex_count, dtype=np.int64)
    if vertex_count == 0 or distance <= 0:
        return identity
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Cannot virtually merge vertices with non-finite coordinates")

    pairs = cKDTree(vertices).query_pairs(float(distance), output_type='ndarray')
    if pairs.size == 0:
        return identity

    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    rows = np.hstack((pairs[:, 0], pairs[:, 1]))
    cols = np.hstack((pairs[:, 1], pairs[:, 0]))
    graph = sp.sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, cols)),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    _component_count, labels = _connected_components(graph)
    return labels.astype(np.int64, copy=False)


def _collapse_vertices_for_inpainting(V2, F2, W2, Matched, merge_map, confidence):
    """Create the temporary welded arrays used only by the weight solver."""
    cluster_count = int(merge_map.max()) + 1 if len(merge_map) else 0
    if cluster_count == len(V2):
        return V2, F2, W2, Matched, confidence

    counts = np.bincount(merge_map, minlength=cluster_count).astype(np.float64)
    merged_vertices = np.zeros((cluster_count, 3), dtype=np.float64)
    np.add.at(merged_vertices, merge_map, V2)
    merged_vertices /= counts[:, np.newaxis]

    matched_counts = np.bincount(
        merge_map, weights=Matched.astype(np.float64), minlength=cluster_count
    )
    merged_matched = matched_counts > 0
    # Direct matches define their cluster; otherwise weak matches are averaged
    # by how much they are trusted.
    trust = np.where(merged_matched[merge_map], Matched, confidence).astype(np.float64)
    trust_totals = np.bincount(merge_map, weights=trust, minlength=cluster_count)
    merged_weights = np.zeros((cluster_count, W2.shape[1]), dtype=np.float64)
    np.add.at(merged_weights, merge_map, W2 * trust[:, np.newaxis])
    trusted = trust_totals > 0
    merged_weights[trusted] /= trust_totals[trusted, np.newaxis]
    merged_confidence = np.zeros(cluster_count, dtype=np.float64)
    np.maximum.at(merged_confidence, merge_map, confidence)
    merged_confidence[merged_matched] = 1.0

    merged_faces = merge_map[F2]
    nondegenerate = np.logical_and.reduce((
        merged_faces[:, 0] != merged_faces[:, 1],
        merged_faces[:, 1] != merged_faces[:, 2],
        merged_faces[:, 2] != merged_faces[:, 0],
    ))
    merged_faces = merged_faces[nondegenerate]
    if len(merged_faces):
        # Overlapping loose parts can produce identical faces after the virtual
        # weld. Count each geometric face only once in the Laplacian.
        canonical_faces = np.sort(merged_faces, axis=1)
        _unique, unique_indices = np.unique(
            canonical_faces, axis=0, return_index=True
        )
        merged_faces = merged_faces[np.sort(unique_indices)]

    return merged_vertices, merged_faces, merged_weights, merged_matched, merged_confidence


INPAINT_MODES = ('POINT', 'SURFACE', 'COMBINED')
# Coincident points make the point-cloud Laplacian's local triangulations
# degenerate: masses collapse towards zero and the solve silently returns
# garbage. Point mode connects such points anyway, so points closer than this
# fraction of the bounding-box diagonal are always welded there. Combined mode
# welds them too, which joins split seams.
POINT_WELD_TOLERANCE = 1e-6
# Combined mode links loose parts where they touch: fully up to this share of
# the local edge length, fading out by twice it. Weaker links are dropped.
LINK_CONTACT = 0.5
LINK_CUTOFF = 1e-3
# Beyond this multiple of a vertex's own smoothness stiffness, a weak match
# only worsens the conditioning of the system without changing the result.
MAX_SOFT_STIFFNESS = 1e4
# Every correct solve reproduces constant weights, because Q @ 1 == 0.
PARTITION_TOLERANCE = 1e-4


@dataclass
class InpaintingDomain:
    weights: np.ndarray
    matched: np.ndarray
    merge_map: np.ndarray
    laplacian: object
    mass: np.ndarray
    rejected: np.ndarray
    # Area-weighted odds of the weak matches (0 where a vertex is free), and
    # the length below which their weights are smoothed out.
    soft_area: np.ndarray = None
    soft_length: float = 0.0
    # Set by solve_inpainting when the biharmonic system was too ill-conditioned
    # and the stable harmonic energy had to be used instead.
    harmonic_fallback: bool = False


class InpaintingError(ValueError):
    def __init__(self, message, rejected=None):
        super().__init__(message)
        self.rejected = rejected


def _soft_area(mass, matched, confidence, soft_length):
    if not soft_length > 0:
        return np.zeros(len(mass))
    soft = ~matched & (confidence > 0) & (mass > 0)
    area = np.zeros(len(mass))
    odds = confidence[soft] / np.maximum(1 - confidence[soft], np.finfo(np.float64).eps)
    area[soft] = mass[soft] * odds
    return area


def _edge_spacing(vertices, triangles):
    """Mean length of each vertex's edges, 0 for vertices without faces."""
    count = len(vertices)
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0) if len(edges) else edges.reshape(0, 2)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    totals = np.bincount(edges.ravel(), weights=np.repeat(lengths, 2), minlength=count)
    degrees = np.bincount(edges.ravel(), minlength=count)
    return np.divide(totals, degrees, out=np.zeros(count), where=degrees > 0)


def _link_laplacian(rows, cols, weights, count):
    """The Laplacian of links rows[i]-cols[i], each listed in both directions."""
    links = sp.sparse.csr_matrix((-weights, (rows, cols)), shape=(count, count))
    return links - sp.sparse.diags(np.asarray(links.sum(axis=1)).ravel())


def _link_loose_parts(V, F, L, mass, anchors):
    """Combined mode: link the loose parts of the mesh Laplacian L through the point cloud.

    Weights still only flow along the mesh within each loose part, so they
    can't jump a gap inside it, like the one between the legs of loose pants.
    Parts are linked where they touch, like a lining, a belt or overlapping
    trims, so they move together. A part that no anchor reaches, not even
    through the parts it touches, keeps every link to the points around it, as
    in Point mode, so it takes their weights instead of being rejected.
    Vertices without faces get their mass from the point cloud.
    """
    count = len(V)
    graph = abs(L)
    graph.setdiag(0)
    graph.eliminate_zeros()
    part_count, parts = _connected_components(graph)
    if part_count < 2 or count < 3:
        return L, mass
    P, M = robust_laplacian.point_cloud_laplacian(V, n_neighbors=min(30, count - 1))
    P = P.tocoo()
    between = (P.row != P.col) & (P.data < 0) & (parts[P.row] != parts[P.col])
    rows, cols, weights = P.row[between], P.col[between], -P.data[between]
    spacing = _edge_spacing(V, F)
    reach = LINK_CONTACT * np.maximum(spacing[rows], spacing[cols])
    t = np.clip(np.linalg.norm(V[rows] - V[cols], axis=1)
                / np.maximum(reach, np.finfo(np.float64).tiny) - 1, 0, 1)
    strength = 1 - t * t * (3 - 2 * t)
    touching = strength > LINK_CUTOFF
    contacts = sp.sparse.csr_matrix((np.ones(np.count_nonzero(touching)),
                                     (rows[touching], cols[touching])), shape=(count, count))
    group_count, groups = _connected_components(graph + contacts)
    anchored = np.zeros(group_count, dtype=bool)
    anchored[groups[anchors]] = True
    unanchored = ~anchored[groups]
    strength[unanchored[rows] | unanchored[cols]] = 1
    keep = strength > LINK_CUTOFF
    L = L + _link_laplacian(rows[keep], cols[keep], weights[keep] * strength[keep], count)
    return L.tocsr(), np.where(mass > 0, mass, M.diagonal())


def prepare_inpainting(V2, F2, W2, Matched, mode, virtual_merge_distance=0.0,
                       confidence=None, soft_length=0.0):
    """Build the actual solver graph, shared by solving and loose-part selection.

    Matched vertices are exact constraints. Other vertices with a positive
    confidence are weak constraints that pull towards their own weights with a
    strength growing with confidence; soft_length is the world-space scale below
    which their weights are smoothed out. Either kind anchors its loose part.

    mode is one of INPAINT_MODES. 'POINT' solves on a point cloud of the
    vertices, so weights also flow between loose parts; 'SURFACE' on the mesh,
    so they stay within each loose part; 'COMBINED' on the mesh too, with loose
    parts linked as _link_loose_parts describes. True and False stand for
    'POINT' and 'SURFACE', like the standalone add-on's point_cloud flag.
    """
    if isinstance(mode, bool):
        mode = 'POINT' if mode else 'SURFACE'
    if mode not in INPAINT_MODES:
        raise InpaintingError(f"Unknown inpaint mode {mode!r}")
    V = np.asarray(V2, dtype=np.float64)
    F = np.asarray(F2)
    W = np.asarray(W2, dtype=np.float64)
    matched = np.asarray(Matched, dtype=bool)
    if (V.ndim != 2 or V.shape[1] != 3 or not len(V)
            or not np.all(np.isfinite(V))):
        raise InpaintingError("Target vertices must be a nonempty finite N by 3 array")
    if (F.ndim != 2 or F.shape[1] != 3 or not np.issubdtype(F.dtype, np.integer)
            or np.any(F < 0) or np.any(F >= len(V))):
        raise InpaintingError("Target contains invalid triangle indices")
    if (W.ndim != 2 or W.shape[0] != len(V) or not W.shape[1]
            or not np.all(np.isfinite(W)) or matched.shape != (len(V),)):
        raise InpaintingError("Target weights or match mask are invalid")
    if not np.isfinite(virtual_merge_distance) or virtual_merge_distance < 0:
        raise InpaintingError("Virtual merge distance must be finite and non-negative")
    if confidence is None:
        confidence = matched.astype(np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if (confidence.shape != (len(V),) or not np.all(np.isfinite(confidence))
            or np.any(confidence < 0) or np.any(confidence > 1)):
        raise InpaintingError("Match confidence must be one value in [0, 1] per vertex")
    if not np.isfinite(soft_length) or soft_length < 0:
        raise InpaintingError("Soft match length must be finite and non-negative")
    confidence = np.where(matched, 1.0, confidence)
    # Inpainting does not change fully matched input. Seam synchronization is
    # deliberately separate and also operates on fully matched meshes.
    if np.all(matched):
        return InpaintingDomain(W.copy(), matched, np.arange(len(V)), None,
                                np.zeros(len(V)), np.zeros(len(V), dtype=bool))
    weld_distance = virtual_merge_distance
    if mode != 'SURFACE':
        diagonal = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
        weld_distance = max(weld_distance, POINT_WELD_TOLERANCE * diagonal)
    merge_map = find_vertex_merge_map(V, weld_distance)
    V, F, W, matched, confidence = _collapse_vertices_for_inpainting(
        V, F, W, matched, merge_map, confidence)
    n = len(V)
    if np.all(matched) or not np.any(matched | ((confidence > 0) & (soft_length > 0))):
        return InpaintingDomain(W, matched, merge_map, None, np.zeros(n),
                                (~matched)[merge_map])
    try:
        if mode == 'POINT' and n >= 3:
            L, M = robust_laplacian.point_cloud_laplacian(V, n_neighbors=min(30, n - 1))
            mass = M.diagonal()
        else:
            # Removed faces can leave isolated vertices. Keep their constraints,
            # but do not feed unused vertices to the native surface builder.
            F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
            if len(F):
                _, keep = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
                F = F[np.sort(keep)]
            active = np.unique(F)
            L = sp.sparse.csr_matrix((n, n), dtype=np.float64)
            mass = np.zeros(n, dtype=np.float64)
            if len(active):
                remap = np.full(n, -1, dtype=np.int64)
                remap[active] = np.arange(len(active))
                local_L, local_M = robust_laplacian.mesh_laplacian(V[active], remap[F])
                coo = local_L.tocoo()
                L = sp.sparse.csr_matrix((coo.data, (active[coo.row], active[coo.col])), shape=(n, n))
                mass[active] = local_M.diagonal()
            if mode == 'COMBINED':
                L, mass = _link_loose_parts(V, F, L, mass,
                                            matched | ((confidence > 0) & (soft_length > 0)))
        L = L.astype(np.float64).tocsr()
        if np.any(~np.isfinite(L.data)) or np.any(~np.isfinite(mass)) or np.any(mass < 0):
            raise InpaintingError("Inpainting produced an invalid Laplacian or mass matrix")
        soft_area = _soft_area(mass, matched, confidence, soft_length)
        graph = L.copy()
        graph.setdiag(0)
        graph.eliminate_zeros()
        count, labels = _connected_components(graph)
        anchored = np.zeros(count, dtype=bool)
        anchored[labels[matched | (soft_area > 0)]] = True
        rejected = ~anchored[labels]
        rejected |= (~matched) & (mass <= 0)
        return InpaintingDomain(W, matched, merge_map, L, mass, rejected[merge_map],
                                soft_area, soft_length)
    except (RuntimeError, ValueError) as error:
        raise InpaintingError(str(error)) from error


def reweight_domain(domain, vertices, triangles, weights, matched, confidence):
    """``domain`` with other matches (weights, exact ones, trust) on the same mesh and constraints, so
    they can be solved again without rebuilding its Laplacian. None when there's nothing to solve."""
    if domain.laplacian is None:
        return None
    matched = np.asarray(matched, dtype=bool)
    confidence = np.where(matched, 1.0, np.asarray(confidence, dtype=np.float64))
    _V, _F, W, merged_matched, merged_confidence = _collapse_vertices_for_inpainting(
        np.asarray(vertices, dtype=np.float64), np.asarray(triangles), np.asarray(weights, dtype=np.float64),
        matched, domain.merge_map, confidence)
    return InpaintingDomain(W, merged_matched, domain.merge_map, domain.laplacian, domain.mass, domain.rejected,
                            _soft_area(domain.mass, merged_matched, merged_confidence, domain.soft_length),
                            domain.soft_length)


def _solve_constrained(L, mass, soft_area, soft_length, data, known, unknown, harmonic):
    """Minimize the smoothness energy plus weak-match penalties with fixed known rows."""
    if harmonic:
        Q = L
        scale = soft_length ** -2 if soft_length > 0 else 0.0
        # soft_area is area * odds, built for the biharmonic L^-4: the harmonic pull area * sqrt(odds)
        # / L^2 keeps the length below which each weak match is smoothed out.
        soft_area = np.sqrt(soft_area * mass)
    else:
        inv_mass = np.zeros_like(mass)
        np.divide(1.0, mass, out=inv_mass, where=mass > 0)
        Q = L + L @ sp.sparse.diags(inv_mass) @ L
        scale = soft_length ** -4 if soft_length > 0 else 0.0
    Q = Q.tocsr()
    Q_unknown = Q[unknown]
    # Dimensionally, area / length^4 matches the biharmonic term, and
    # area / length^2 the harmonic one.
    penalty = np.minimum(soft_area[unknown] * scale,
                         MAX_SOFT_STIFFNESS * np.abs(Q.diagonal()[unknown]))
    system = (Q_unknown[:, unknown] + sp.sparse.diags(penalty)).tocsc()
    rhs = penalty[:, np.newaxis] * data[unknown] - Q_unknown[:, known] @ data[known]
    with warnings.catch_warnings():
        warnings.simplefilter('error', sp.sparse.linalg.MatrixRankWarning)
        solution = sp.sparse.linalg.spsolve(system, rhs)
    return np.asarray(solution).reshape(len(unknown), data.shape[1])


def _clip_negative_weights(weights):
    """Remove negative overshoot while keeping each row's total weight."""
    totals = weights.sum(axis=1, keepdims=True)
    clipped = np.clip(weights, 0, None)
    clipped_totals = clipped.sum(axis=1, keepdims=True)
    return np.divide(clipped * np.maximum(totals, 0), clipped_totals,
                     out=np.zeros_like(clipped), where=clipped_totals > 0)


def solve_inpainting(domain, skip_rejected=False):
    """Solve constrained vertices; optionally leave unsupported output unused.

    Partial transfers discard rejected rows at write time. Those rows must not
    supply constraints or participate in the supported linear system.
    """
    if np.any(domain.rejected) and not skip_rejected:
        raise InpaintingError("Loose parts without a matched vertex remain in the solver graph",
                              domain.rejected)
    W = domain.weights.copy()
    rejected = np.zeros(len(W), dtype=bool)
    rejected[domain.merge_map[domain.rejected]] = True
    active = np.flatnonzero(~rejected)
    unknown = np.flatnonzero(~domain.matched[active])
    if len(unknown):
        L = domain.laplacian[active][:, active]
        soft_area = (domain.soft_area[active] if domain.soft_area is not None
                     else np.zeros(len(active)))
        known = np.flatnonzero(domain.matched[active])
        # Groups without weight on any of these vertices stay zero, so only the
        # others are solved: a mesh gets few of the body's many groups.
        used = np.flatnonzero(np.any(W[active] != 0, axis=0))
        # The extra all-ones column must come back as exactly one everywhere.
        # When it does not, the solve is numerically broken even without a
        # singular-matrix warning, so retry with the better conditioned
        # harmonic energy before giving up.
        data = np.column_stack((W[active][:, used], np.ones(len(active))))
        error = None
        for harmonic in (False, True):
            try:
                solution = _solve_constrained(L, domain.mass[active], soft_area,
                                              domain.soft_length, data, known, unknown, harmonic)
            except (RuntimeError, ValueError, sp.sparse.linalg.MatrixRankWarning) as caught:
                error = caught
                continue
            if (np.all(np.isfinite(solution))
                    and np.max(np.abs(solution[:, -1] - 1)) <= PARTITION_TOLERANCE):
                domain.harmonic_fallback = harmonic
                W[np.ix_(active[unknown], used)] = _clip_negative_weights(solution[:, :-1])
                break
        else:
            raise InpaintingError("Constrained weight solve failed"
                                  + (": " + str(error) if error else ": the result is numerically unstable"))
    with np.errstate(over='ignore'):
        result = W[domain.merge_map].astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise InpaintingError("Inpainting produced non-finite weights")
    return result


def inpaint(V2, F2, W2, Matched, mode, virtual_merge_distance=0.0,
            confidence=None, soft_length=0.0):
    """
    Inpaint weights for all the vertices on the target mesh for which  we didnt 
    find a good match on the source (i.e. Matched[i] == False).

    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
        mode: the inpaint mode, see prepare_inpainting
        virtual_merge_distance: vertices within this world-space distance are
            temporarily welded for the solve; the Blender mesh is not changed
        confidence: optional #V2 trust in [0, 1] of the weak matches among the
            unmatched vertices, see prepare_inpainting
        soft_length: world-space smoothing scale of the weak matches

    Returns:
        W_inpainted: #V2 by num_bones, final skinning weights where we inpainted weights for all vertices i where Matched[i] == False
    """
    
    try:
        domain = prepare_inpainting(V2, F2, W2, Matched, mode, virtual_merge_distance,
                                    confidence, soft_length)
        return True, solve_inpainting(domain)
    except (RuntimeError, ValueError, TypeError):
        return False, np.asarray(W2, dtype=np.float32)



def limit_mask(weights, adjacency_matrix, dilation_repeat=5, limit_num=4):
    if weights.shape[1] <= limit_num: return np.zeros_like(weights)
    
    count = np.count_nonzero(weights, axis=1)
    to_limit = count > limit_num
    k = weights.shape[1] - limit_num
    weights_inds = np.argpartition(weights, kth=k, axis=1)[:, :k]
    row_indices = np.arange(weights.shape[0])[:, None]
    erode_mask = np.zeros_like(weights, dtype=bool)
    erode_mask[row_indices, weights_inds] = True
    erode_mask = np.logical_and(erode_mask, to_limit[:, np.newaxis])
    erode_mask = sp.sparse.csr_array(erode_mask).astype(np.float32)
    adj_mat = adjacency_matrix
    degrees = adj_mat.sum(axis=1)
    smooth_mat = (1/degrees[:, np.newaxis]) * adj_mat
    for _ in range(dilation_repeat):
        avg_weights = smooth_mat @ erode_mask
        erode_mask = erode_mask.maximum(avg_weights)
    
    return erode_mask.toarray()


def smooth_weigths(verts, weights, matched, adjacency_matrix, adjacency_list, num_smooth_iter_steps, smooth_alpha, distance_threshold):
    not_matched = ~matched
    VIDs_to_smooth = np.zeros(verts.shape[0], dtype=bool)

    def get_points_within_distance(V, VID, distance=distance_threshold):
        """
        Get all neighbours of vertex VID within dDISTANCE_THRESHOLD
        """
        queue = []
        queue.append(VID)
        while len(queue) != 0:
            vv = queue.pop()
            if vv < len(adjacency_list):
                neigh = adjacency_list[vv]
                for nn in neigh:
                    if ~VIDs_to_smooth[nn] and np.linalg.norm(V[VID,:]-V[nn]) < distance:
                        VIDs_to_smooth[nn] = True
                        if nn not in queue:
                            queue.append(nn)

    for i in range(verts.shape[0]):
        if not_matched[i]:
            get_points_within_distance(verts, i, distance_threshold)
            
    adj_mat = adjacency_matrix.astype(np.float32)
    degrees = adj_mat.sum(axis=1)
    
    smooth_mat = sp.sparse.diags(1/degrees) @ adj_mat
    weights_smoothed = sp.sparse.csr_array(weights)
    for _ in range(num_smooth_iter_steps):
        weights_smoothed = (1 - smooth_alpha) * weights_smoothed + smooth_alpha * (smooth_mat @ weights_smoothed)
        weights_smoothed[~VIDs_to_smooth] = weights[~VIDs_to_smooth]
    return weights_smoothed.todense()
            
