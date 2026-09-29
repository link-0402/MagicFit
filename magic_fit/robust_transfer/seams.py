"""Geometry-only seam detection and constrained weight synchronization.

All arrays are temporary; these helpers never change a Blender mesh.
"""
import numpy as np
import scipy.sparse as sparse
from scipy.spatial import cKDTree


def boundary_components(vertex_count, triangles, edges):
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    face_edges = np.sort(np.concatenate((triangles[:, :2], triangles[:, 1:],
                                        triangles[:, [2, 0]])), axis=1)
    unique, counts = np.unique(face_edges, axis=0, return_counts=True)
    boundary = np.zeros(vertex_count, dtype=bool)
    boundary[unique[counts == 1].ravel()] = True
    graph = sparse.csr_matrix((np.ones(len(edges) * 2),
                               (edges.ravel(), edges[:, ::-1].ravel())),
                              shape=(vertex_count, vertex_count))
    _, components = sparse.csgraph.connected_components(graph, directed=False)
    return boundary, components


def _mutual_nearest_pairs(vertices, pairs):
    """Keep only proximity pairs that are each other's closest match.

A boundary sampled more densely on one side than the other can put a single
vertex within range of several vertices on the other side. Keeping every such
edge lets connected components chain an entire seam into one oversized
cluster, so synchronize_weights ends up averaging positions (e.g. a waistband
and a hem) that were never actually close together. Requiring a kept edge to
be the nearest match for both endpoints keeps clusters local.
"""
    a, b = pairs[:, 0], pairs[:, 1]
    distances = np.linalg.norm(vertices[a] - vertices[b], axis=1)
    endpoints = np.concatenate((a, b))
    partners = np.concatenate((b, a))
    lengths = np.concatenate((distances, distances))
    order = np.lexsort((lengths, endpoints))
    endpoints, partners = endpoints[order], partners[order]
    nearest_of = dict(zip(endpoints[np.concatenate(([True], np.diff(endpoints) != 0))].tolist(),
                          partners[np.concatenate(([True], np.diff(endpoints) != 0))].tolist()))
    mutual = np.fromiter(
        (nearest_of.get(int(u)) == int(v) and nearest_of.get(int(v)) == int(u)
         for u, v in zip(a.tolist(), b.tolist())),
        dtype=bool, count=len(a))
    return pairs[mutual]


def find_seam_clusters(vertices, boundary, components, compatibility, distance):
    """Cluster nearby borders of different parts, only within compatible batches.

Component IDs must be unique across objects. Compatibility IDs encode the
chosen within-object/cross-object scope. Return excluded proximity-pair count.
"""
    vertices = np.asarray(vertices, dtype=np.float64)
    if not np.all(np.isfinite(vertices)) or not np.isfinite(distance) or distance < 0:
        raise ValueError("Seam positions and distance must be finite; distance cannot be negative")
    indices = np.flatnonzero(boundary)
    if len(indices) < 2:
        return [], 0
    pairs = cKDTree(vertices[indices]).query_pairs(distance, output_type='ndarray')
    pairs = indices[pairs]
    pairs = pairs[components[pairs[:, 0]] != components[pairs[:, 1]]]
    compatible = compatibility[pairs[:, 0]] == compatibility[pairs[:, 1]]
    excluded = int(np.count_nonzero(~compatible))
    pairs = pairs[compatible]
    if not len(pairs):
        return [], excluded
    pairs = _mutual_nearest_pairs(vertices, pairs)
    if not len(pairs):
        return [], excluded
    involved, inverse = np.unique(pairs, return_inverse=True)
    edges = inverse.reshape(-1, 2)
    graph = sparse.csr_matrix((np.ones(len(edges) * 2),
                               (edges.ravel(), edges[:, ::-1].ravel())),
                              shape=(len(involved), len(involved)))
    _, labels = sparse.csgraph.connected_components(graph, directed=False)
    order = np.argsort(labels, kind='stable')
    splits = np.flatnonzero(np.diff(labels[order])) + 1
    return list(np.split(involved[order], splits)), excluded


def synchronize_weights(weights, writable, protected_vertices, clusters, limit=None, confidence=None):
    """Return final deform weights and counts of synchronized/skipped clusters.

Any column protected at any cluster member becomes a fixed contribution for
the entire cluster. Partial transfer masks protect their entire vertex.

confidence, when given, is a per-vertex non-negative trust value (for example
1.0 for a direct surface match and 0.0 for a value that came entirely from
inpainting/smoothing). A duplicated, double-sided mesh commonly matches
cleanly on one side and only reaches its back-facing twin through inpainting;
without this, a plain mean at their shared boundary drags the well-matched
side's weight down toward the weaker, merely-inpainted one. When every member
of a cluster has zero confidence, the merge falls back to a plain mean.
"""
    result = np.asarray(weights, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("Cannot synchronize non-finite seam weights")
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float64)
        if (confidence.shape != (len(result),) or not np.all(np.isfinite(confidence))
                or np.any(confidence < 0)):
            raise ValueError("Seam confidence must be one finite, non-negative value per vertex")
    synchronized = skipped = 0
    tolerance = 1e-7
    threshold = 1e-5
    for cluster in clusters:
        values = result[cluster]
        if np.any(protected_vertices[cluster]):
            skipped += 1
            continue
        movable = np.all(writable[cluster], axis=0)
        fixed = ~movable
        if np.any(values[:, fixed] != values[0, fixed]):
            skipped += 1
            continue
        common = values[0].copy()
        remaining = 1.0 - common[fixed].sum()
        fixed_count = np.count_nonzero(common[fixed] > 0)
        if remaining < -tolerance or (limit is not None and fixed_count > limit):
            skipped += 1
            continue
        trust = confidence[cluster] if confidence is not None else None
        if trust is not None and trust.sum() > tolerance:
            candidate = np.clip(np.average(values[:, movable], axis=0, weights=trust), 0, 1)
        else:
            candidate = np.clip(values[:, movable].mean(axis=0), 0, 1)
        candidate[candidate < threshold] = 0
        if limit is not None:
            slots = max(0, limit - fixed_count)
            # Stable ties follow the common group-name order.
            keep = np.argsort(-candidate, kind='stable')[:slots]
            limited = np.zeros_like(candidate)
            limited[keep] = candidate[keep]
            candidate = limited
        remaining = max(0.0, remaining)
        total = candidate.sum()
        if total <= 0 and remaining > tolerance:
            skipped += 1
            continue
        if total > 0:
            candidate *= remaining / total
            # Match the writer's cutoff, then restore the remaining mass.
            candidate[candidate < threshold] = 0
            if candidate.sum() > 0:
                candidate *= remaining / candidate.sum()
            elif remaining > tolerance:
                skipped += 1
                continue
        common[movable] = candidate
        # Leave protected columns bit-for-bit unchanged, including tiny weights.
        result[np.ix_(cluster, np.flatnonzero(movable))] = common[movable]
        synchronized += 1
    return result, synchronized, skipped
