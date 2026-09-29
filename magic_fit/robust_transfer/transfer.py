"""Stage transfer results before synchronization and Blender weight writes."""
# Changed for Magic Fit (2026-09): per-object settings are
# Object.magic_fit_transfer, Crease Smoothing trusts each match by how far
# the mesh floats above the body (weighttransfer.bridge_confidence), a
# transfer clears the deform groups the body doesn't have, and there's no
# L/R group balancing.
import numpy as np

from . import util, seams, weighttransfer

WRITE_THRESHOLD = 1e-5
# Rigid weights below this round to nothing in FFXIV's 8-bit vertex weights (half a step), so they're dropped.
RIGID_THRESHOLD = 0.5 / 255.0
WEIGHT_TOLERANCE = 1e-7
# Weak matches are smoothed below this fraction of Max Distance: the farther a
# garment may sit from the source, the less precise its closest points are.
SOFT_LENGTH_FACTOR = 0.25


def armature_key(obj, fallback=None):
    modifiers = [m for m in obj.modifiers if m.type == 'ARMATURE' and m.show_viewport]
    if not modifiers:
        return ('RIG', fallback.as_pointer(), False) if fallback else ('UNBOUND',)
    if len(modifiers) == 1 and modifiers[0].object:
        mod = modifiers[0]
        if (not mod.use_vertex_groups or mod.use_bone_envelopes or mod.vertex_group
                or mod.use_multi_modifier):
            # These settings can produce different motion for identical weights.
            return ('OBJECT', obj.as_pointer())
        return ('RIG', mod.object.as_pointer(), mod.use_deform_preserve_volume)
    # Ambiguous deformation setups may be processed, but never shared.
    return ('OBJECT', obj.as_pointer())


def source_armature(obj):
    mods = [m for m in obj.modifiers if m.type == 'ARMATURE' and m.object and m.show_viewport]
    return mods[0].object if len(mods) == 1 else None


def rest_armature(obj):
    """The armature of obj's first Armature modifier, on or off, or the one a Customize+ rig belongs to."""
    from ..cplus import rig as cplus_rig
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE' and mod.object is not None and mod.object.type == 'ARMATURE':
            armature = mod.object
            if cplus_rig.is_rig(armature) and armature.parent is not None:
                return armature.parent
            return armature
    return None


def deform_armature(obj, fallback=None):
    """The armature whose deform bones tell which of obj's vertex groups are deform groups: its first
    Armature modifier's, the armature it's parented to as an armature, or ``fallback`` (the body's)."""
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE':
            armature = mod.object
            return armature if armature is not None and armature.type == 'ARMATURE' else fallback
    parent = obj.parent
    if parent is not None and parent.type == 'ARMATURE' and obj.parent_type == 'ARMATURE':
        return parent
    return fallback


def is_deform_group(target, name):
    """Whether the target's group ``name`` belongs to a deform bone (see deform_armature)."""
    if 'deform_armature' not in target:
        return util.is_vertex_group_deform_bone(target['obj'], name)
    armature = target['deform_armature']
    bone = armature.data.bones.get(name) if armature is not None else None
    return bool(bone and bone.use_deform)


def stride_source(body, names, vertices, weights, surface_bvh, posed=False):
    """What Hem Follow needs of the body (see weighttransfer.hem_follow), for the weight columns
    ``names``: the strides' deform matrices per column, which columns belong to each leg, and the
    body's surface. ``posed``: the body is matched as displayed, so the strides start from the
    armature's pose instead of its rest pose. None without an armature with both thighs."""
    armature = rest_armature(body)
    if armature is None:
        return None
    bones = list(armature.data.bones)
    index = {bone.name: i for i, bone in enumerate(bones)}
    parents = [index[bone.parent.name] if bone.parent else -1 for bone in bones]
    world = armature.matrix_world
    rest = np.array([np.array(world @ bone.matrix_local) for bone in bones])
    start = rest
    if posed and armature.pose is not None:
        start = np.array([np.array(world @ armature.pose.bones[bone.name].matrix) for bone in bones])
    # FFXIV characters face -Y; the bust tells if this one is turned around.
    front = np.array([0.0, -1.0, 0.0])
    columns = {name: i for i, name in enumerate(names)}
    thigh = columns.get(weighttransfer.HEM_LEGS[0])
    for bust in ('iv_c_mune_l', 'j_mune_l'):
        column = columns.get(bust)
        if column is None or thigh is None or weights[:, column].sum() <= 0:
            continue
        leg = weights[:, thigh] > 0.5
        if leg.any():
            offset = np.average(vertices, axis=0, weights=weights[:, column]) - vertices[leg].mean(axis=0)
            if abs(offset[1]) > 1e-6:
                front = np.array([0.0, np.sign(offset[1]), 0.0])
        break
    down = np.array([0.0, 0.0, -1.0])
    thigh_bone = index.get(weighttransfer.HEM_LEGS[0])
    if thigh_bone is not None and parents[thigh_bone] >= 0:
        # Posed, front and down turn with the hips.
        turn = (start[parents[thigh_bone]] @ np.linalg.inv(rest[parents[thigh_bone]]))[:3, :3]
        front, down = turn @ front, turn @ down
    poses = weighttransfer.stride_poses([bone.name for bone in bones], start, parents, front, down)
    if not poses:
        return None
    rows = [index.get(name) for name in names]
    identity = np.eye(4)
    poses = [np.array([pose[row] if row is not None else identity for row in rows]) for pose in poses]
    legs = np.zeros((len(names), 2))
    for side, root in enumerate(weighttransfer.HEM_LEGS):
        for bone in armature.data.bones[root].children_recursive + [armature.data.bones[root]]:
            if bone.name in columns:
                legs[columns[bone.name], side] = 1
    return dict(poses=poses, legs=legs, bvh=surface_bvh)


def make_target(obj, depsgraph, settings):
    if settings.use_deformed_target and util.has_modifier(obj, *util.TOPOLOGY_MODS):
        raise ValueError(f'{obj.name}: disable topology-changing modifiers or Use Deformed Target')
    if settings.seam_sync and obj.data.users > 1:
        raise ValueError(f'{obj.name}: make the mesh single-user before synchronizing seams')
    evaluated = obj.evaluated_get(depsgraph) if settings.use_deformed_target else obj
    vertices, triangles, normals = util.get_obj_arrs_world(evaluated)
    if len(vertices) != len(obj.data.vertices):
        raise ValueError(f'{obj.name}: evaluated topology differs from the original mesh')
    if not len(vertices) or not len(triangles):
        raise ValueError(f'{obj.name}: target must contain vertices and faces')
    return dict(obj=obj, vertices=vertices, triangles=triangles, normals=normals)


def inpaint_mask(obj):
    settings = obj.magic_fit_transfer
    if not util.is_group_valid(obj.vertex_groups, settings.inpaint_group):
        return np.zeros(len(obj.data.vertices), dtype=bool)
    mask = util.get_group_arr(obj, settings.inpaint_group) > settings.inpaint_threshold
    return ~mask if settings.inpaint_group_invert else mask


def distance_strength(squared_distances, radius, falloff_percent):
    """Smoothly fade over the outer fraction of a world-space surface radius."""
    distances = np.asarray(squared_distances, dtype=np.float64)
    if (not np.all(np.isfinite(distances)) or np.any(distances < 0)
            or not np.isfinite(radius) or radius < 0
            or not np.isfinite(falloff_percent) or not 0 <= falloff_percent <= 100):
        raise ValueError('Partial reweight distances and falloff must be finite and non-negative')
    if radius == 0 or falloff_percent == 0:
        return (distances <= radius ** 2).astype(np.float64)
    width = radius * falloff_percent / 100
    t = np.clip((np.sqrt(distances) - (radius - width)) / width, 0, 1)
    return 1 - t * t * (3 - 2 * t)


def soft_length(settings):
    """The length below which weak matches are smoothed out: fixed for Crease Smoothing, where it
    cancels out of the matches' pull, otherwise a share of Max Distance."""
    if settings.crease_smoothing:
        return weighttransfer.BRIDGE_SOFT_LENGTH
    return settings.max_distance * SOFT_LENGTH_FACTOR


def weld_distance(settings):
    """How close vertices must be to be welded for the solve. Crease Smoothing always welds coincident
    ones, so split seams get the same weights on both sides."""
    weld = settings.virtual_merge_distance if settings.virtual_merge else 0.0
    return max(weld, weighttransfer.BRIDGE_WELD) if settings.crease_smoothing else weld


def match_target(target, source_vertices, source_triangles, source_normals,
                 source_weights, settings, surface_bvh, strides=None):
    match = weighttransfer.match_closest_surface(
        source_vertices, source_triangles, source_normals,
        target['vertices'], target['normals'], source_weights,
        settings.flip_vertex_normal, surface_bvh)
    masked = inpaint_mask(target['obj']) if not settings.apply_to_selected else None
    target.update(weights=match.weights, masked=masked)
    if settings.crease_smoothing:
        # Every match is soft, trusted by how high the mesh floats above the body and how well the
        # normals agree (either way round): where it lies on the body it copies its weights, where it
        # spans a crease they're inpainted smoothly from both sides. prepare_batch works it out again
        # over meshes sharing a solve.
        target.update(distances=match.distances, cosines=np.abs(np.cos(np.radians(match.angles))))
        bridge_trust([target], settings)
        if strides is not None:
            target.update(strides=strides, points=match.points, point_normals=match.normals,
                          closest_weights=match.weights.copy(),
                          leg_share=(match.weights @ strides['legs']).max(axis=1))
    else:
        max_angle = np.degrees(settings.max_normal_angle_difference)
        confidence = weighttransfer.match_confidence(
            match.distances, match.angles, settings.max_distance, max_angle,
            max(max_angle, np.degrees(settings.soft_normal_limit)))
        if masked is not None:
            confidence[masked] = 0
        target.update(matched=confidence >= 1, confidence=confidence)
    if settings.partial_reweight:
        target['strength'] = distance_strength(
            match.distances ** 2, settings.max_distance, settings.partial_reweight_falloff)


def batches(targets, across, fallback=None):
    result = {}
    for index, target in enumerate(targets):
        key = armature_key(target['obj'], fallback) if across else ('OBJECT', index)
        result.setdefault(key, []).append(target)
    return list(result.values())


def target_confidence(target):
    """Per-vertex trust in the transferred weights; exact inpaint input is binary."""
    return target.get('confidence', target['matched'].astype(np.float64))


def bridge_trust(batch, settings):
    """Crease Smoothing's trust (weighttransfer.bridge_confidence) of meshes sharing a solve, worked
    out over all of them, so the seams between them don't count as borders. Vertices without faces
    keep the body's weights; the Inpaint Mask's vertices are inpainted."""
    offsets = np.cumsum([0] + [len(t['vertices']) for t in batch])
    confidence = weighttransfer.bridge_confidence(
        np.concatenate([t['vertices'] for t in batch]),
        np.concatenate([t['triangles'] + offset for t, offset in zip(batch, offsets)]),
        np.concatenate([t['distances'] for t in batch]), np.concatenate([t['cosines'] for t in batch]),
        soft_length(settings), weighttransfer.BRIDGE_SOFTNESS * settings.crease_softness,
        weld_distance(settings), settings.max_distance)
    for target, start, end in zip(batch, offsets, offsets[1:]):
        trust = confidence[start:end].copy()
        faceless = np.ones(len(trust), dtype=bool)
        faceless[np.unique(target['triangles'])] = False
        trust[faceless] = 1.0
        if target.get('masked') is not None:
            trust[target['masked']] = 0
        target.update(confidence=trust, matched=trust >= 1)


def prepare_batch(batch, settings):
    if settings.crease_smoothing and len(batch) > 1 and all('cosines' in t for t in batch):
        bridge_trust(batch, settings)
    offsets = np.cumsum([0] + [len(t['vertices']) for t in batch])
    domain = weighttransfer.prepare_inpainting(
        np.concatenate([t['vertices'] for t in batch]),
        np.concatenate([t['triangles'] + offset for t, offset in zip(batch, offsets)]),
        np.concatenate([t['weights'] for t in batch]),
        np.concatenate([t['matched'] for t in batch]),
        settings.inpaint_mode,
        weld_distance(settings),
        np.concatenate([target_confidence(t) for t in batch]),
        soft_length(settings),
    )
    return domain, offsets


def solve_targets(targets, settings, fallback=None, partial=False):
    across = (settings.seam_sync and settings.seam_sync_across_objects
              and settings.apply_to_selected and settings.virtual_merge)
    for batch in batches(targets, across, fallback):
        if partial and not any(np.any(t['strength'] > 0) for t in batch):
            for target in batch:
                target['supported'] = np.zeros(len(target['vertices']), dtype=bool)
            continue
        try:
            domain, offsets = prepare_batch(batch, settings)
            if np.any(domain.rejected) and not partial:
                affected = [t['obj'].name for t, start, end in zip(batch, offsets, offsets[1:])
                            if np.any(domain.rejected[start:end])]
                if settings.crease_smoothing:
                    advice = ('. They lie wholly in the Inpaint Mask. Shrink the Inpaint Mask, enable '
                              'Partial Reweight, or use Select Rejected Loose Parts.')
                else:
                    advice = ('. Increase Max Distance or Soft Normal Limit, enable Virtual Merge or '
                              'Partial Reweight, or use Select Rejected Loose Parts.')
                raise ValueError('No usable match for loose parts in ' + ', '.join(affected) + advice)
            solved = weighttransfer.solve_inpainting(domain, skip_rejected=partial)
        except (RuntimeError, ValueError) as error:
            names = ', '.join(t['obj'].name for t in batch)
            raise ValueError(f'Inpainting failed on {names}: {error}') from error
        for target, start, end in zip(batch, offsets, offsets[1:]):
            target['weights'] = solved[start:end].copy()
            target['harmonic_fallback'] = domain.harmonic_fallback
            if partial:
                target['supported'] = ~domain.rejected[start:end]
        if settings.crease_smoothing:
            follow_hems(batch, domain, settings, partial)


def follow_hems(batch, domain, settings, partial):
    """Hem Follow: where a hem lies over a leg, pull its weights towards the leg's as far as the
    strides of weighttransfer.hem_follow need, and solve again on the same Laplacian."""
    offsets = np.cumsum([0] + [len(t['vertices']) for t in batch])
    pulls = [np.zeros(len(t['vertices'])) for t in batch]
    # Meshes solved together are checked together (their seams aren't hems), each group of them with
    # the strides of the body it was matched against.
    groups = {}
    for position, target in enumerate(batch):
        if 'strides' in target:
            groups.setdefault(id(target['strides']), []).append(position)
    for members in groups.values():
        group = [batch[position] for position in members]
        ends = np.cumsum([0] + [len(t['vertices']) for t in group])
        vertices = np.concatenate([t['vertices'] for t in group])
        merge_map = weighttransfer.find_vertex_merge_map(
            vertices, max(weld_distance(settings), weighttransfer.BRIDGE_WELD))
        strides = group[0]['strides']
        pull = weighttransfer.hem_follow(
            vertices, np.concatenate([t['triangles'] + offset for t, offset in zip(group, ends)]), merge_map,
            *(np.concatenate([t[key] for t in group]) for key in (
                'normals', 'points', 'point_normals', 'distances', 'closest_weights', 'weights', 'leg_share')),
            strides['poses'], strides['bvh'])
        for position, start, end in zip(members, ends, ends[1:]):
            pulls[position] = pull[start:end]
    if not any(np.any(pull > 1e-3) for pull in pulls):
        return
    priors, confidences = [], []
    for target, pull in zip(batch, pulls):
        # The Inpaint Mask keeps its vertices free, and Max Distance is as far as matches reach.
        pull[target_confidence(target) <= 0] = 0
        if 'distances' in target:
            pull[target['distances'] > settings.max_distance] = 0
        prior = target.get('closest_weights', target['weights']).astype(np.float64).copy()
        pulled = pull > 1e-3
        if np.any(pulled):
            solved = target['weights'][pulled].astype(np.float64)
            solved /= np.maximum(solved.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
            prior[pulled] = (1 - pull[pulled, None]) * solved + pull[pulled, None] * prior[pulled]
        priors.append(prior)
        confidences.append(weighttransfer.hem_follow_trust(target_confidence(target), pull))
    confidence = np.concatenate(confidences)
    domain = weighttransfer.reweight_domain(
        domain, np.concatenate([t['vertices'] for t in batch]),
        np.concatenate([t['triangles'] + offset for t, offset in zip(batch, offsets)]),
        np.concatenate(priors), confidence >= 1, confidence)
    if domain is None:
        return
    try:
        solved = weighttransfer.solve_inpainting(domain, skip_rejected=partial)
    except (RuntimeError, ValueError) as error:
        names = ', '.join(t['obj'].name for t in batch)
        raise ValueError(f'Inpainting failed on {names}: {error}') from error
    for target, start, end, conf in zip(batch, offsets, offsets[1:], confidences):
        target['weights'] = solved[start:end].copy()
        target['confidence'] = conf
        target['harmonic_fallback'] = target.get('harmonic_fallback') or domain.harmonic_fallback


def process_weights(target, settings):
    """Smooth/limit candidates on the supported mesh before mask blending."""
    obj = target['obj']
    active = np.flatnonzero(target.get('supported', np.ones(len(target['vertices']), bool)))
    if not len(active):
        return
    weights = target['weights'][active].copy()
    if settings.crease_smoothing:
        # The solve already smoothed what it should; smoothing more would blur the weights where the
        # mesh lies on the body.
        if settings.enforce_four_bone_limit:
            weights = weighttransfer.limit_groups_smoothly(weights, settings.num_limit_groups)
        target['weights'][active] = weights
        return
    adjacency = util.get_mesh_adjacency_matrix_sparse(obj.data, include_self=True)[active][:, active]
    if settings.smoothing_enable:
        neighbors = [[int(j) for j in adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]
                      if j != i]
                     for i in range(len(active))]
        weights = np.asarray(weighttransfer.smooth_weigths(
            target['vertices'][active], weights, target['matched'][active], adjacency,
            neighbors, settings.smoothing_repeat, settings.smoothing_factor, settings.max_distance))
    if settings.enforce_four_bone_limit:
        weights[weights <= 0.0001] = 0
        weights *= 1 - weighttransfer.limit_mask(weights, adjacency, limit_num=settings.num_limit_groups)
        weights[weights <= 0.0001] = 0
    target['weights'][active] = weights


def stage_weights(target, names, included, apply_mask=False, clear_others=False):
    """Reconcile by name, including zero columns, without writing to Blender.

    With clear_others, the target's other deform groups (bones the source has no weights for, such as
    an imported mod's skirt bones) are cleared where the transfer writes, so they neither stay on top
    of the new weights nor get normalized into them."""
    obj = target['obj']
    current = {g.name: w.copy() for g, w in zip(obj.vertex_groups, util.get_groups_arr(obj).T)}
    final = {name: w.copy() for name, w in current.items()}
    mask = np.ones(len(obj.data.vertices))
    object_settings = obj.magic_fit_transfer
    if apply_mask and util.is_group_valid(obj.vertex_groups, object_settings.vertex_group):
        mask = util.get_group_arr(obj, object_settings.vertex_group).astype(np.float64)
        if object_settings.vertex_group_invert:
            mask = 1 - mask
    if 'strength' in target:
        mask *= target['strength'] * target['supported']
    writable = set()
    for index, name in enumerate(names):
        group = obj.vertex_groups.get(name)
        if not included[index] or (group and group.lock_weight):
            continue
        old = current.get(name, np.zeros(len(mask)))
        weights = np.asarray(target['weights'][:, index], dtype=np.float64)
        if not np.all(np.isfinite(weights)):
            raise ValueError(f'{obj.name}: non-finite output weights')
        final[name] = np.clip((1 - mask) * old + mask * weights, 0, 1).astype(np.float32)
        final[name][(mask > 0) & (final[name] < 1e-5)] = 0
        writable.add(name)
    if clear_others:
        transferred = {name for name, use in zip(names, included) if use}
        for group in obj.vertex_groups:
            if group.name in transferred or group.lock_weight or not is_deform_group(target, group.name):
                continue
            final[group.name] = ((1 - mask) * current[group.name]).astype(np.float32)
            final[group.name][(mask > 0) & (final[group.name] < 1e-5)] = 0
            writable.add(group.name)
    target.update(final=final, writable=writable, protected=mask < 1,
                  write_vertices=mask > 0)


def synchronize_targets(targets, settings, deform_names, fallback=None):
    """Synchronize complete deform vectors, retaining target-only contributions."""
    if not settings.seam_sync:
        return 0, 0, 0
    names = set(deform_names)
    for target in targets:
        obj = target['obj']
        target['deform_names'] = set(deform_names) | {
            g.name for g in obj.vertex_groups if is_deform_group(target, g.name)}
        names.update(target['deform_names'])
    names = sorted(names)
    if not names:
        return 0, 0, 0
    across = settings.seam_sync_across_objects and settings.apply_to_selected
    compatibility_keys = {}
    positions, boundaries, components, compatibility = [], [], [], []
    weights, writable, protected, confidence = [], [], [], []
    component_offset = 0
    offsets = [0]
    for index, target in enumerate(targets):
        obj = target['obj']
        # Base-mesh coordinates deliberately ignore pose and shape-key mix.
        verts, triangles, _ = util.get_obj_arrs_world(obj)
        edges = np.empty((len(obj.data.edges), 2), dtype=np.int64)
        obj.data.edges.foreach_get('vertices', edges.ravel())
        boundary, labels = seams.boundary_components(len(verts), triangles, edges)
        key = armature_key(obj, fallback) if across else ('OBJECT', index)
        key_id = compatibility_keys.setdefault(key, len(compatibility_keys))
        positions.append(verts)
        boundaries.append(boundary)
        components.append(labels + component_offset)
        compatibility.append(np.full(len(verts), key_id))
        component_offset += int(labels.max()) + 1
        weights.append(np.column_stack([
            target['final'].get(name, np.zeros(len(verts)))
            if name in target['deform_names'] else np.zeros(len(verts)) for name in names]))
        writable.append(np.tile([name in target['writable'] and name in target['deform_names']
                                 for name in names], (len(verts), 1)))
        protected.append(target['protected'])
        confidence.append(target_confidence(target) if 'matched' in target
                          else np.ones(len(verts), dtype=np.float64))
        offsets.append(offsets[-1] + len(verts))
    clusters, excluded = seams.find_seam_clusters(
        np.concatenate(positions), np.concatenate(boundaries), np.concatenate(components),
        np.concatenate(compatibility), settings.seam_distance)
    final, synced, skipped = seams.synchronize_weights(
        np.concatenate(weights), np.concatenate(writable), np.concatenate(protected), clusters,
        settings.num_limit_groups if settings.enforce_four_bone_limit else None,
        np.concatenate(confidence))
    # Only copy columns in the original transfer scope. Protected groups remain
    # untouched even if another object transfers that same group name.
    for target, start, end in zip(targets, offsets, offsets[1:]):
        for column, name in enumerate(names):
            if name in target['writable'] and name in target['deform_names']:
                target['final'][name] = final[start:end, column]
    return synced, skipped, excluded if across else 0


def target_deform_names(target, transferred_deform_names=()):
    """Return target deform names, including newly staged source groups."""
    obj = target['obj']
    names = [group.name for group in obj.vertex_groups]
    names.extend(name for name in transferred_deform_names if name not in names)
    return [name for name in names if is_deform_group(target, name)]


def _set_normalized_values(values, movable, fixed):
    """Return a locked-safe normalized deform vector, or None if impossible."""
    result = np.asarray(values, dtype=np.float64).copy()
    fixed_total = result[fixed].sum()
    if fixed_total > 1 + WEIGHT_TOLERANCE:
        return None
    remaining = max(0.0, 1.0 - fixed_total)
    candidate = result[movable]
    total = candidate.sum()
    if total <= WEIGHT_TOLERANCE:
        if remaining > WEIGHT_TOLERANCE:
            return None
        result[movable] = 0
        return result
    candidate *= remaining / total
    candidate[candidate < WRITE_THRESHOLD] = 0
    total = candidate.sum()
    if total <= WEIGHT_TOLERANCE:
        if remaining > WEIGHT_TOLERANCE:
            return None
        result[movable] = 0
        return result
    candidate *= remaining / total
    # Keep the persisted total stable despite floating-point rounding.
    largest = int(np.argmax(candidate))
    candidate[largest] += remaining - candidate.sum()
    result[movable] = candidate
    return result


def normalize_target_deform_weights(target, transferred_deform_names):
    """Normalize writable deform weights on vertices touched by the transfer."""
    names = target_deform_names(target, transferred_deform_names)
    if not names:
        return 0
    obj = target['obj']
    vertex_count = len(obj.data.vertices)
    for name in names:
        target['final'].setdefault(name, np.zeros(vertex_count, dtype=np.float32))
    locked = np.array([bool((group := obj.vertex_groups.get(name)) and group.lock_weight)
                       for name in names])
    movable = ~locked
    target['writable'].update(name for name, can_write in zip(names, movable) if can_write)
    values = np.column_stack([target['final'][name] for name in names])
    failed = 0
    for index in np.flatnonzero(target['write_vertices']):
        normalized = _set_normalized_values(values[index], movable, locked)
        if normalized is None:
            failed += 1
            continue
        values[index] = normalized
    for column, name in enumerate(names):
        target['final'][name] = values[:, column].astype(np.float32)
    return failed


def limit_target_groups(target, limit, transferred_deform_names):
    """Keep every vertex the transfer writes within ``limit`` deform groups. process_weights only limits
    the new weights, which a Transfer Mask or Partial Reweight blends with the old ones. Locked groups
    count against the limit, as in make_target_rigid; the others are limited as limit_groups_smoothly
    does, keeping their total."""
    obj = target['obj']
    names = target_deform_names(target, transferred_deform_names)
    if not names:
        return
    count = len(obj.data.vertices)
    locked = np.array([bool((group := obj.vertex_groups.get(name)) and group.lock_weight) for name in names])
    movable = np.flatnonzero(~locked)
    values = np.column_stack([target['final'].get(name, np.zeros(count)) for name in names]).astype(np.float64)
    slots = np.maximum(1, limit - np.count_nonzero(values[:, locked] > 0, axis=1))
    over = target['write_vertices'] & (np.count_nonzero(values[:, movable] > 0, axis=1) > slots)
    if not np.any(over):
        return
    for slot in np.unique(slots[over]):
        rows = np.flatnonzero(over & (slots == slot))
        values[np.ix_(rows, movable)] = weighttransfer.limit_groups_smoothly(
            values[np.ix_(rows, movable)], int(slot))
    for column in movable:
        target['final'][names[column]] = values[:, column].astype(np.float32)
    target['writable'].update(names[column] for column in movable)


def vertex_areas(vertices, triangles):
    """Each vertex's share of the surface: a third of the area of every triangle around it."""
    areas = np.zeros(len(vertices))
    if len(triangles):
        a, b, c = (np.asarray(vertices, dtype=np.float64)[triangles[:, k]] for k in range(3))
        third = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 6.0
        for k in range(3):
            np.add.at(areas, triangles[:, k], third)
    return areas


def rigid_weights(values, areas, movable, limit=None):
    """One deform vector for a whole mesh: the average of the ``movable`` columns of ``values``
    (vertices x groups), weighted by the vertices' ``areas`` (evenly if they have none), without weights
    too small to keep (they'd round to nothing in the game's 8-bit weights) and cut to the ``limit``
    largest, as fractions adding up to 1. None when there's nothing to average."""
    values = np.asarray(values, dtype=np.float64)[:, movable]
    areas = np.asarray(areas, dtype=np.float64)
    if areas.sum() <= 0.0:
        areas = np.ones(len(values))
    average = (areas[:, None] * values).sum(axis=0) / areas.sum()
    total = average.sum()
    if total <= WEIGHT_TOLERANCE:
        return None
    average[average < RIGID_THRESHOLD * total] = 0.0
    if limit is not None and np.count_nonzero(average) > limit:
        keep = np.argsort(-average, kind='stable')[:limit]
        kept = np.zeros_like(average)
        kept[keep] = average[keep]
        average = kept
    total = average.sum()
    return average / total if total > WEIGHT_TOLERANCE else None


def make_target_rigid(target, settings, transferred_deform_names):
    """Give every vertex the transfer writes the same deform weights, so the mesh moves as one piece:
    the average of what the transfer staged for them (after masks, seam sync and normalizing), weighted by
    their share of the surface in the rest shape. They add up to 1, or to what locked groups leave; locked
    groups keep their weights. Returns False when there's nothing to average."""
    obj = target['obj']
    rows = np.flatnonzero(target['write_vertices'])
    names = list(dict.fromkeys(list(transferred_deform_names) + target_deform_names(target)))
    if not len(rows) or not names:
        return False
    count = len(obj.data.vertices)
    for name in names:
        target['final'].setdefault(name, np.zeros(count, dtype=np.float32))
    locked = np.array([bool((group := obj.vertex_groups.get(name)) and group.lock_weight) for name in names])
    movable = ~locked
    if not movable.any():
        return False
    values = np.column_stack([target['final'][name] for name in names]).astype(np.float64)
    locked_values = values[rows][:, locked]
    locked_count = int((locked_values > 0).sum(axis=1).max()) if locked.any() else 0
    limit = max(1, settings.num_limit_groups - locked_count) if settings.enforce_four_bone_limit else None
    rest_vertices, rest_triangles, _normals = util.get_obj_arrs_world(obj)
    shares = rigid_weights(values[rows], vertex_areas(rest_vertices, rest_triangles)[rows], movable, limit)
    if shares is None:
        return False
    budget = np.clip(1.0 - locked_values.sum(axis=1), 0.0, 1.0)
    values[np.ix_(rows, np.flatnonzero(movable))] = budget[:, None] * shares[None, :]
    target['writable'].update(name for name, can_write in zip(names, movable) if can_write)
    for column, name in enumerate(names):
        target['final'][name] = values[:, column].astype(np.float32)
    return True


def report_rigid(operator, targets, made_rigid):
    names = [target['obj'].name for target, rigid in zip(targets, made_rigid) if rigid]
    missed = [target['obj'].name for target, rigid in zip(targets, made_rigid) if rigid is False]
    if names:
        operator.report({'INFO'}, 'Made rigid: ' + ', '.join(names))
    if missed:
        operator.report({'WARNING'}, 'Could not make ' + ', '.join(missed)
                        + ' rigid: the transfer left them no deform weights to average')


def postprocess_transfer_targets(targets, settings, transferred_deform_names):
    """Apply optional final transfer corrections before Blender writes weights."""
    counts = {'normalization_failed': 0}
    for target in targets:
        if settings.enforce_four_bone_limit:
            limit_target_groups(target, settings.num_limit_groups, transferred_deform_names)
        if settings.normalize_weights_after_transfer:
            counts['normalization_failed'] += normalize_target_deform_weights(
                target, transferred_deform_names)
    return counts


def write_targets(targets):
    for target in targets:
        obj = target['obj']
        vertices = target['write_vertices']
        for name in sorted(target['writable']):
            w = target['final'][name]
            group = obj.vertex_groups.get(name)
            if group is None:
                if not np.any(w[vertices] >= WRITE_THRESHOLD):
                    continue
                group = obj.vertex_groups.new(name=name)
            if group.lock_weight:
                continue
            for index in np.flatnonzero(vertices & (w >= WRITE_THRESHOLD)):
                group.add([int(index)], float(w[index]), 'REPLACE')
            remove = np.flatnonzero(vertices & (w < WRITE_THRESHOLD)).tolist()
            if remove:
                group.remove(remove)


def report_seams(operator, counts):
    synced, skipped, excluded = counts
    if synced:
        operator.report({'INFO'}, f'Synchronized {synced} seam clusters')
    if skipped or excluded:
        operator.report({'WARNING'}, f'Skipped {skipped} protected or infeasible seam clusters; '
                        f'excluded {excluded} border pairs with incompatible deformation setups')


def report_partial(operator, targets):
    for target in targets:
        if 'strength' in target:
            skipped = np.count_nonzero((target['strength'] > 0) & ~target['supported'])
            if skipped:
                operator.report({'WARNING'}, f"{target['obj'].name}: preserved {skipped} in-range "
                                'vertices without a usable weight constraint')


def report_stability(operator, targets):
    names = [target['obj'].name for target in targets if target.get('harmonic_fallback')]
    if names:
        operator.report({'WARNING'}, 'Used smoother, numerically stable inpainting on '
                        + ', '.join(names) + ' because the regular solve was ill-conditioned')


def report_postprocess(operator, counts):
    if counts['normalization_failed']:
        operator.report({'WARNING'}, 'Could not normalize '
                        f"{counts['normalization_failed']} vertex/vertices because locked weights "
                        'leave no valid deform-weight budget')
