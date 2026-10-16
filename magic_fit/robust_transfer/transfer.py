"""Stage transfer results before synchronization and Blender weight writes."""
# Changed for Magic Fit (2026-09): per-object settings are
# Object.magic_fit_transfer, Crease Smoothing trusts each match by how far
# the mesh floats above the body (weighttransfer.bridge_confidence), a
# transfer clears the deform groups the body doesn't have, and there's no
# L/R group balancing. 2026-10: normals are compared as each loose part
# faces the body (weighttransfer.facing_cosines), Crease Smoothing is the
# only matching (no Max Distance, normal angles, Inpaint Mode or smoothing
# settings), and fingers and toes are handled on their own (digit_columns).
import numpy as np
from mathutils.bvhtree import BVHTree
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from . import util, seams, weighttransfer

WRITE_THRESHOLD = 1e-5
# Rigid weights below this round to nothing in FFXIV's 8-bit vertex weights (half a step), so they're dropped.
RIGID_THRESHOLD = 0.5 / 255.0
# Loose parts of a Rigid mesh whose triangles cross or come this close are one piece, like the split segments of a
# ring or a buckle's frame and prong. Pieces apart from each other, like the rings on two straps, move on their own.
RIGID_GAP = 0.0003
# The other meshes of a transfer hold onto the Rigid ones: their vertices within HOLD_DISTANCE of a rigid piece
# take its weights, and that change fades out along the mesh over HOLD_BLEND times the piece's size (its bounding
# box's diagonal), so a strap looped around a ring stays on it instead of following the body under it.
HOLD_DISTANCE = 0.002
HOLD_BLEND = 3.0
# A piece they hold moves as they do around it: the average of their weights within HOLD_AVERAGE times its size,
# fading out with distance (and fully where they touch it), so a ring follows the bra it's on.
HOLD_AVERAGE = 2.0
# Layers lying on each other (a trim on its panel) are linked this many times as strongly as a typical edge
# along the mesh in that blend, so they get the same weights and stay together.
HOLD_LAYER_LINK = 100.0
# Points closer than HOLD_DISTANCE lie on each other when at least this share of the way between them runs along
# either one's normal.
LAYER_COSINE = 0.7
WEIGHT_TOLERANCE = 1e-7
# How far from the body a match counts, and Partial Reweight reaches. Farther, a vertex only anchors its
# loose part. 2.5 or 10 cm gave the same weights on every garment tried.
MAX_DISTANCE = 0.05


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


def weld_distance(settings):
    """How close vertices must be to be welded for the solve. Coincident ones always are, so split seams
    get the same weights on both sides."""
    weld = settings.virtual_merge_distance if settings.virtual_merge else 0.0
    return max(weld, weighttransfer.BRIDGE_WELD)


def digit_columns(names):
    """Which of the body's groups ``names`` are finger or toe bones (weighttransfer.DIGIT_BONE)."""
    return np.array([bool(weighttransfer.DIGIT_BONE.match(name)) for name in names], dtype=bool)


def match_target(target, source_vertices, source_triangles, source_normals,
                 source_weights, settings, surface_bvh, strides=None, digits=None):
    """Match ``target`` to the body. ``digits``: which weight columns are finger or toe bones (digit_columns)."""
    match = weighttransfer.match_closest_surface(
        source_vertices, source_triangles, source_normals, target['vertices'], source_weights, surface_bvh)
    # How the normals agree, each loose part turned the way it faces the body: negative where the body
    # touches the mesh from outside (a thigh against the edge of a leotard's crotch), not under it.
    cosines = weighttransfer.facing_cosines(target['vertices'], target['triangles'], target['normals'],
                                            match.normals, match.distances)
    masked = inpaint_mask(target['obj']) if not settings.apply_to_selected else None
    digit_share = (weighttransfer.digit_share(match.weights, digits) if digits is not None
                   else np.zeros(len(match.weights)))
    # Over fingers and toes, Combined doesn't link parts whose outsides face each other, like the plates on
    # two fingers of a glove.
    outward = weighttransfer.outward_normals(target['vertices'], target['triangles'], target['normals'],
                                             match.normals, match.distances)
    outward *= (digit_share >= weighttransfer.DIGIT_SHARE[0])[:, np.newaxis]
    # Every match is soft, trusted by how high the mesh floats above the body: where it lies on the body
    # it copies its weights, where it spans a crease they're inpainted smoothly from both sides.
    # prepare_batch works it out again over meshes sharing a solve.
    target.update(weights=match.weights, masked=masked, outward=outward, digit_share=digit_share,
                  distances=match.distances, cosines=cosines, points=match.points, point_normals=match.normals,
                  closest_weights=match.weights.copy())
    bridge_trust([target], settings)
    if strides is not None:
        # Where the body touches the mesh from outside, there's no leg under it to follow.
        outside = np.nan_to_num(cosines, nan=0.0) < 0
        target.update(strides=strides,
                      leg_share=np.where(outside, 0.0, (match.weights @ strides['legs']).max(axis=1)))
    if settings.partial_reweight:
        target['strength'] = distance_strength(
            match.distances ** 2, MAX_DISTANCE, settings.partial_reweight_falloff)


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
    """The trust in the matches (weighttransfer.bridge_confidence) of meshes sharing a solve, worked out
    over all of them, so the seams between them don't count as borders. Vertices without faces keep the
    body's weights; the Inpaint Mask's vertices are inpainted."""
    offsets = np.cumsum([0] + [len(t['vertices']) for t in batch])
    confidence = weighttransfer.bridge_confidence(
        np.concatenate([t['vertices'] for t in batch]),
        np.concatenate([t['triangles'] + offset for t, offset in zip(batch, offsets)]),
        np.concatenate([t['distances'] for t in batch]), np.concatenate([t['cosines'] for t in batch]),
        np.concatenate([t['points'] for t in batch]), weighttransfer.BRIDGE_SOFT_LENGTH,
        weld_distance(settings), MAX_DISTANCE,
        np.concatenate([t.get('digit_share', np.zeros(len(t['vertices']))) for t in batch]))
    for target, start, end in zip(batch, offsets, offsets[1:]):
        trust = confidence[start:end].copy()
        faceless = np.ones(len(trust), dtype=bool)
        faceless[np.unique(target['triangles'])] = False
        trust[faceless] = 1.0
        if target.get('masked') is not None:
            trust[target['masked']] = 0
        target.update(confidence=trust, matched=trust >= 1)


def prepare_batch(batch, settings):
    if len(batch) > 1 and all('cosines' in t for t in batch):
        bridge_trust(batch, settings)
    offsets = np.cumsum([0] + [len(t['vertices']) for t in batch])
    domain = weighttransfer.prepare_inpainting(
        np.concatenate([t['vertices'] for t in batch]),
        np.concatenate([t['triangles'] + offset for t, offset in zip(batch, offsets)]),
        np.concatenate([t['weights'] for t in batch]),
        np.concatenate([t['matched'] for t in batch]),
        'COMBINED',
        weld_distance(settings),
        np.concatenate([target_confidence(t) for t in batch]),
        weighttransfer.BRIDGE_SOFT_LENGTH,
        np.concatenate([t['outward'] for t in batch]) if all('outward' in t for t in batch) else None,
        # How high each vertex lies above the body, so layers of one part are linked too.
        np.concatenate([t['distances'] for t in batch]) if all('distances' in t for t in batch) else None,
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
                raise ValueError('No usable match for loose parts in ' + ', '.join(affected)
                                 + '. They lie wholly in the Inpaint Mask. Shrink the Inpaint Mask, enable '
                                 'Partial Reweight, or use Select Rejected Loose Parts.')
            solved = weighttransfer.solve_inpainting(domain, skip_rejected=partial)
        except (RuntimeError, ValueError) as error:
            names = ', '.join(t['obj'].name for t in batch)
            raise ValueError(f'Inpainting failed on {names}: {error}') from error
        for target, start, end in zip(batch, offsets, offsets[1:]):
            target['weights'] = solved[start:end].copy()
            target['harmonic_fallback'] = domain.harmonic_fallback
            if partial:
                target['supported'] = ~domain.rejected[start:end]
        follow_hems(batch, domain, settings, partial)


def follow_hems(batch, domain, settings, partial):
    """Hem Follow: where a hem lies over a leg, pull its weights towards the leg's as far as the
    strides of weighttransfer.hem_follow need, and solve again on the same Laplacian. Meshes with a
    Customize+ template (``target['cplus']``) also follow the skin wherever it would pass through them."""
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
    # With Customize+ (see cplus_follow), anywhere the template's skin would pass through the mesh.
    templates = [cplus_follow(target) if 'cplus' in target else np.zeros(len(target['vertices'])) for target in batch]
    if not any(np.any(pull > 1e-3) for pull in pulls + templates):
        return
    priors, confidences = [], []
    for target, hem, template in zip(batch, pulls, templates):
        pull = np.maximum(hem, template)
        # The Inpaint Mask keeps its vertices free, and MAX_DISTANCE is as far as matches reach.
        pull[target_confidence(target) <= 0] = 0
        if 'distances' in target:
            pull[target['distances'] > MAX_DISTANCE] = 0
        prior = target.get('closest_weights', target['weights']).astype(np.float64).copy()
        pulled = pull > 1e-3
        if np.any(pulled):
            solved = target['weights'][pulled].astype(np.float64)
            solved /= np.maximum(solved.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
            prior[pulled] = (1 - pull[pulled, None]) * solved + pull[pulled, None] * prior[pulled]
        priors.append(prior)
        trust = weighttransfer.hem_follow_trust(target_confidence(target), hem)
        template[pull <= 1e-3] = 0
        confidences.append(weighttransfer.hem_follow_trust(trust, template, weighttransfer.CPLUS_TRUST))
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


def cplus_follow(target):
    """How far (0 to 1) to pull each vertex towards its closest-point weights so that the skin its Customize+
    template moves (``target['cplus']``, deform matrices per weight column) doesn't pass through it: anywhere,
    also where the skin touches it from outside, like a thigh fold over the edge of a thong's crotch, which
    the solve otherwise barely trusts (weighttransfer.bridge_confidence)."""
    pull = np.zeros(len(target['vertices']))
    rows = np.flatnonzero((target['distances'] <= MAX_DISTANCE) & (target['closest_weights'].sum(axis=1) > 0)
                          & (target['weights'].sum(axis=1) > 0))
    if len(rows):
        pull[rows] = weighttransfer.follow_needed(
            target['vertices'][rows], target['points'][rows], target['point_normals'][rows],
            target['closest_weights'][rows], target['weights'][rows], [target['cplus']])
    return pull


def process_weights(target, settings):
    """Limit the groups of the supported vertices before mask blending. The solve already smoothed what it
    should; smoothing more would blur the weights where the mesh lies on the body."""
    active = np.flatnonzero(target.get('supported', np.ones(len(target['vertices']), bool)))
    if len(active) and settings.enforce_four_bone_limit:
        target['weights'][active] = weighttransfer.limit_groups_smoothly(
            target['weights'][active], settings.num_limit_groups)


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


def rigid_pieces(vertices, triangles, gap=RIGID_GAP):
    """A piece number per vertex: the loose parts of the mesh (``vertices``, ``triangles``), joined where their
    triangles cross or a vertex of one comes within ``gap`` of the other, so they move together."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    count = len(vertices)
    if not count:
        return np.zeros(0, dtype=np.int64)
    links = [triangles[:, [0, 1]], triangles[:, [1, 2]]]
    if count > 1:
        links.append(np.asarray(cKDTree(vertices).query_pairs(gap, output_type='ndarray'),
                                dtype=np.int64).reshape(-1, 2))
    if len(triangles):
        tree = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
        # Triangles crossing (or touching) triangles of other parts, and vertices near other parts' triangles.
        crossing = np.asarray(tree.overlap(tree), dtype=np.int64).reshape(-1, 2)
        links.append(np.column_stack((triangles[crossing[:, 0], 0], triangles[crossing[:, 1], 0])))
        near = [(index, face) for index, point in enumerate(vertices.tolist())
                for _location, _normal, face, _distance in tree.find_nearest_range(point, gap)]
        if near:
            near = np.asarray(near, dtype=np.int64)
            links.append(np.column_stack((near[:, 0], triangles[near[:, 1], 0])))
    links = np.vstack(links)
    graph = sparse.coo_matrix((np.ones(len(links), dtype=np.uint8), (links[:, 0], links[:, 1])),
                              shape=(count, count)).tocsr()
    return weighttransfer._connected_components(graph)[1].astype(np.int64)


def make_target_rigid(target, settings, transferred_deform_names, others=()):
    """Give every vertex the transfer writes in a piece of the mesh (`rigid_pieces`) the same deform weights, so
    each piece moves as one. A piece that the ``others`` (the transfer's targets that aren't Rigid) touch, within
    HOLD_DISTANCE, like a ring a strap is looped through, moves as what holds it: the average of what the
    transfer staged for their vertices touching it. Any other piece moves as the body under it: the average of
    what the transfer staged for its own vertices. Averages are weighted by the vertices' share of the surface in
    the rest shape, after masks, seam sync and normalizing. They add up to 1, or to what locked groups leave;
    locked groups keep their weights. Returns False when there's nothing to average."""
    obj = target['obj']
    written = target['write_vertices']
    names = list(dict.fromkeys(list(transferred_deform_names) + target_deform_names(target)
                               + [name for other in others for name in target_deform_names(other)]))
    if not written.any() or not names:
        return False
    count = len(obj.data.vertices)
    for name in names:
        target['final'].setdefault(name, np.zeros(count, dtype=np.float32))
    locked = np.array([bool((group := obj.vertex_groups.get(name)) and group.lock_weight) for name in names])
    movable = ~locked
    if not movable.any():
        return False
    values = np.column_stack([target['final'][name] for name in names]).astype(np.float64)
    rest_vertices, rest_triangles, _normals = util.get_obj_arrs_world(obj)
    areas = vertex_areas(rest_vertices, rest_triangles)
    pieces = rigid_pieces(rest_vertices, rest_triangles)
    target['rigid_pieces'] = pieces
    held = _holding_weights(rest_vertices, rest_triangles, pieces, others, names, piece_sizes(rest_vertices, pieces))
    made = False
    for piece in np.unique(pieces[written]):
        rows = np.flatnonzero(written & (pieces == piece))
        locked_values = values[rows][:, locked]
        locked_count = int((locked_values > 0).sum(axis=1).max()) if locked.any() else 0
        limit = max(1, settings.num_limit_groups - locked_count) if settings.enforce_four_bone_limit else None
        source, source_areas = held.get(int(piece), (values[rows], areas[rows]))
        shares = rigid_weights(source, source_areas, movable, limit)
        if shares is None:
            continue
        budget = np.clip(1.0 - locked_values.sum(axis=1), 0.0, 1.0)
        values[np.ix_(rows, np.flatnonzero(movable))] = budget[:, None] * shares[None, :]
        made = True
    if not made:
        return False
    target['writable'].update(name for name, can_write in zip(names, movable) if can_write)
    for column, name in enumerate(names):
        target['final'][name] = values[:, column].astype(np.float32)
    return True


def _layer_links(points, normals, edges):
    """Links between layers lying on each other, like a trim on its panel, a lining under it or a strap folded
    back on itself, so what holds a rigid piece carries them all along: pairs of ``points`` within HOLD_DISTANCE
    of each other, apart mostly along their ``normals`` (by LAYER_COSINE), that ``edges`` (index pairs) don't
    already join. Neighbours on one sheet lie apart along it instead. Index pairs."""
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.asarray(cKDTree(points).query_pairs(HOLD_DISTANCE, output_type='ndarray'),
                       dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return pairs
    keys = np.sort(edges, axis=1)
    known = np.isin(pairs[:, 0] * len(points) + pairs[:, 1], keys[:, 0] * len(points) + keys[:, 1])
    offset = points[pairs[:, 1]] - points[pairs[:, 0]]
    length = np.linalg.norm(offset, axis=1)
    across = np.maximum(np.abs(np.einsum('ij,ij->i', offset, normals[pairs[:, 0]])),
                        np.abs(np.einsum('ij,ij->i', offset, normals[pairs[:, 1]])))
    return pairs[~known & (across >= LAYER_COSINE * length)]


def piece_sizes(vertices, pieces):
    """The size (bounding box diagonal) of each piece, by piece number."""
    count = int(pieces.max()) + 1 if len(pieces) else 0
    low = np.full((count, 3), np.inf)
    high = np.full((count, 3), -np.inf)
    np.minimum.at(low, pieces, vertices)
    np.maximum.at(high, pieces, vertices)
    return np.linalg.norm(high - low, axis=1)


def _holding_weights(vertices, triangles, pieces, others, names, sizes):
    """What holds each piece of a rigid mesh (``vertices``, ``triangles``, a piece per vertex, ``sizes`` per
    piece): {piece: (staged weights of the ``others``' vertices around it, in columns ``names``; their areas
    times how much they count)}, for the pieces they touch (within HOLD_DISTANCE). Vertices count fully within
    HOLD_DISTANCE and fade out (smoothstep) by HOLD_AVERAGE times the piece's size, on the loose parts touching
    it."""
    if not others or not len(triangles):
        return {}
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    tree = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
    reaches = HOLD_DISTANCE + HOLD_AVERAGE * sizes
    reach = float(reaches.max())
    low, high = vertices.min(axis=0) - reach, vertices.max(axis=0) + reach
    found = {}
    for other in others:
        points, faces, normals = util.get_obj_arrs_world(other['obj'])
        points = points.astype(np.float64)
        rows = np.flatnonzero(np.all((points >= low) & (points <= high), axis=1))
        near, piece_of, gap = [], [], []
        for row, point in zip(rows.tolist(), points[rows].tolist()):
            location, _normal, face, distance = tree.find_nearest(point, reach)
            if location is not None:
                piece = int(pieces[triangles[face][0]])
                if distance <= reaches[piece]:
                    near.append(row)
                    piece_of.append(piece)
                    gap.append(distance)
        if not near:
            continue
        near, piece_of, gap = np.asarray(near), np.asarray(piece_of), np.asarray(gap)
        # Only loose parts that touch the piece hold it, with the layers lying on them (and welded, so split
        # seams don't cut them).
        count = len(points)
        merge = weighttransfer.find_vertex_merge_map(points, weighttransfer.BRIDGE_WELD)
        edges = merge[np.vstack((faces[:, [0, 1]], faces[:, [1, 2]]))]
        unit = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        edges = np.vstack((edges, _layer_links(points, unit.astype(np.float64), edges)))
        graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                  shape=(count, count)).tocsr()
        _parts, labels = weighttransfer._connected_components(graph)
        labels = labels[merge]
        staged = np.column_stack([other['final'].get(name, np.zeros(count)) for name in names])
        areas = vertex_areas(points, faces)
        for piece in np.unique(piece_of):
            mine = piece_of == piece
            touching = mine & (gap <= HOLD_DISTANCE)
            if not touching.any():
                continue
            mine &= np.isin(labels[near], labels[near[touching]])
            fade = np.clip((gap[mine] - HOLD_DISTANCE) / max(reaches[piece] - HOLD_DISTANCE, 1e-9), 0.0, 1.0)
            share = 1.0 - fade * fade * (3.0 - 2.0 * fade)
            found.setdefault(int(piece), []).append((staged[near[mine]], areas[near[mine]] * share))
    return {piece: (np.concatenate([values for values, _areas in parts]).astype(np.float64),
                    np.concatenate([areas for _values, areas in parts]))
            for piece, parts in found.items()}


def hold_rigid_pieces(targets, made_rigid, settings, transferred_deform_names):
    """Make the other meshes of the transfer hold onto the Rigid ones (``made_rigid`` per target: True for those
    made rigid, None for those that aren't Rigid): their vertices within HOLD_DISTANCE of a rigid piece take its
    weights at their closest point there, and the change fades out harmonically along the mesh over HOLD_BLEND times the piece's size, so a strap stays
    wrapped around its ring and what doesn't touch the piece keeps its weights. Locked groups keep their
    weights, the group limit holds, and only vertices the transfer writes change. Returns the names of the
    meshes that changed."""
    rigid = [target for target, made in zip(targets, made_rigid) if made]
    holders = [target for target, made in zip(targets, made_rigid) if made is None]
    if not rigid or not holders:
        return []
    surfaces = [util.get_obj_arrs_world(target['obj'])[:2] for target in rigid]
    # How far each rigid triangle reaches: HOLD_DISTANCE plus the blend for its piece.
    reaches = []
    for target, (vertices, triangles) in zip(rigid, surfaces):
        pieces = target.get('rigid_pieces')
        if pieces is None:
            pieces = rigid_pieces(vertices, triangles)
        size = piece_sizes(vertices, pieces)
        reaches.append(HOLD_DISTANCE + HOLD_BLEND * size[pieces[triangles[:, 0]]])
    offsets = np.cumsum([0] + [len(vertices) for vertices, _triangles in surfaces])
    points = np.concatenate([vertices for vertices, _triangles in surfaces]).astype(np.float64)
    tris = np.concatenate([triangles + offset for (_vertices, triangles), offset in zip(surfaces, offsets)])
    if not len(tris):
        return []
    tree = BVHTree.FromPolygons(points.tolist(), tris.tolist(), all_triangles=True)
    names = list(dict.fromkeys(list(transferred_deform_names)
                               + [name for target in rigid for name in target_deform_names(target)]))
    values = np.concatenate([
        np.column_stack([target['final'].get(name, np.zeros(len(vertices))) for name in names])
        for target, (vertices, _triangles) in zip(rigid, surfaces)]).astype(np.float64)
    reaches = np.concatenate(reaches)
    reach = float(reaches.max())
    box = (points.min(axis=0) - reach, points.max(axis=0) + reach)
    return [target['obj'].name for target in holders
            if _hold(target, settings, transferred_deform_names, tree, points, tris, reaches, values, names, box)]


def _hold(target, settings, transferred_deform_names, tree, points, tris, reaches, values, names, box):
    """`hold_rigid_pieces` for one mesh, onto the rigid surface ``tree`` (``points``, ``tris``, how far each
    triangle ``reaches``) whose vertices have ``values`` (columns ``names``). Returns whether any weights
    changed."""
    obj = target['obj']
    vertices, triangles, normals = util.get_obj_arrs_world(obj)
    vertices = vertices.astype(np.float64)
    written = target['write_vertices']
    rows = np.flatnonzero(written & np.all((vertices >= box[0]) & (vertices <= box[1]), axis=1))
    count = len(vertices)
    gap = np.full(count, np.inf)
    reach_of = np.zeros(count)
    found, locations, faces = [], [], []
    for row, point in zip(rows.tolist(), vertices[rows].tolist()):
        location, _normal, face, distance = tree.find_nearest(point, HOLD_DISTANCE)
        if location is None:
            continue
        gap[row] = distance
        reach_of[row] = reaches[face] - HOLD_DISTANCE
        found.append(row)
        locations.append(location[:])
        faces.append(face)
    if not found:
        return False

    columns = list(dict.fromkeys(names + target_deform_names(target, transferred_deform_names)))
    for name in columns:
        target['final'].setdefault(name, np.zeros(count, dtype=np.float32))
    locked = np.array([bool((group := obj.vertex_groups.get(name)) and group.lock_weight) for name in columns])
    movable = np.flatnonzero(~locked)
    if not len(movable):
        return False
    current = np.column_stack([target['final'][name] for name in columns]).astype(np.float64)
    # The rigid weights at each holding vertex's closest point, as shares of the movable groups.
    faces = np.asarray(faces, dtype=np.int64)
    bary = weighttransfer._barycentric_coordinates(np.asarray(locations), points[tris[faces]])
    taken = np.zeros((count, len(columns)))
    taken[np.asarray(found)[:, None], np.arange(len(names))[None, :]] = np.einsum(
        'ik,ikj->ij', np.clip(bary, 0.0, None), values[tris[faces]])

    # Welded vertices (split seams) as one, linked along the mesh's edges by their inverse lengths.
    merge = weighttransfer.find_vertex_merge_map(vertices, weld_distance(settings))
    clusters = int(merge.max()) + 1
    sizes = np.bincount(merge, minlength=clusters).astype(np.float64)
    centre = np.zeros((clusters, 3))
    np.add.at(centre, merge, vertices)
    centre /= sizes[:, None]
    cluster_gap = np.full(clusters, np.inf)
    np.minimum.at(cluster_gap, merge, gap)
    holding = cluster_gap <= HOLD_DISTANCE
    writable = np.zeros(clusters, dtype=bool)
    writable[merge[written]] = True
    held = np.zeros((clusters, len(columns)))
    holds = np.flatnonzero(gap <= HOLD_DISTANCE)
    np.add.at(held, merge[holds], taken[holds])
    held /= np.maximum(np.bincount(merge[holds], minlength=clusters), 1)[:, None]
    own = np.zeros((clusters, len(columns)))
    np.add.at(own, merge, current)
    own /= sizes[:, None]

    edges = merge[np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))]
    edges = np.unique(np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1), axis=0)
    facing = np.zeros((clusters, 3))
    np.add.at(facing, merge, normals.astype(np.float64))
    facing /= np.maximum(np.linalg.norm(facing, axis=1, keepdims=True), 1e-12)
    along = len(edges)
    edges = np.vstack((edges, _layer_links(centre, facing, edges)))
    lengths = np.maximum(np.linalg.norm(centre[edges[:, 0]] - centre[edges[:, 1]], axis=1), 1e-6)
    holding &= writable
    if not holding.any():
        return False
    # The blend reaches HOLD_BLEND times the piece's size from where the mesh touches it, measured along the mesh
    # (and across the layers lying on it), so it neither jumps to other parts nearby nor across a gore.
    cluster_reach = np.zeros(clusters)
    np.maximum.at(cluster_reach, merge, reach_of)
    graph = sparse.coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(clusters, clusters)).tocsr()
    sources = np.flatnonzero(holding)
    along_mesh, _previous, nearest = csgraph.dijkstra(graph, directed=False, indices=sources, min_only=True,
                                                      limit=float(cluster_reach.max()), return_predecessors=True)
    region = np.isfinite(along_mesh) & (along_mesh <= np.where(nearest >= 0, cluster_reach[np.maximum(nearest, 0)],
                                                                0.0))
    unknown = region & ~holding & writable

    def normalized(values):
        part = np.clip(values[:, movable], 0.0, None)
        totals = part.sum(axis=1, keepdims=True)
        return np.divide(part, totals, out=np.zeros_like(part), where=totals > WEIGHT_TOLERANCE)

    # Each vertex keeps its own weights plus a correction: the rigid piece's minus its own where it touches the
    # piece, fading to nothing by the end of the blend (harmonic), so the fabric around keeps its own shape of
    # weights instead of being smoothed over.
    shares = normalized(own)
    correction = np.zeros_like(shares)
    correction[holding] = normalized(held)[holding] - shares[holding]
    if unknown.any():
        strength = 1.0 / lengths
        if along and len(edges) > along:
            strength[along:] = HOLD_LAYER_LINK * np.median(strength[:along])
        laplacian = sparse.coo_matrix(
            (np.concatenate((strength, strength)), (np.concatenate((edges[:, 0], edges[:, 1])),
                                                    np.concatenate((edges[:, 1], edges[:, 0])))),
            shape=(clusters, clusters)).tocsr()
        laplacian = sparse.diags(np.asarray(laplacian.sum(axis=1)).ravel()) - laplacian
        solve = np.flatnonzero(unknown)
        fixed = np.flatnonzero(~unknown)
        system = laplacian[solve][:, solve].tocsc()
        rhs = -(laplacian[solve][:, fixed] @ correction[fixed])
        correction[solve] = np.asarray(splu(system).solve(np.asarray(rhs)))
    shares = np.clip(shares + correction, 0.0, None)

    changed = holding | unknown
    rows = np.flatnonzero(changed[merge] & written)
    if not len(rows):
        return False
    budget = np.clip(1.0 - current[rows][:, locked].sum(axis=1), 0.0, 1.0)
    new = shares[merge[rows]]
    if settings.enforce_four_bone_limit:
        slots = np.maximum(1, settings.num_limit_groups - np.count_nonzero(current[rows][:, locked] > 0, axis=1))
        for slot in np.unique(slots):
            pick = slots == slot
            new[pick] = weighttransfer.limit_groups_smoothly(new[pick], int(slot))
    else:
        new[new < RIGID_THRESHOLD] = 0.0
    totals = new.sum(axis=1, keepdims=True)
    new = np.divide(new, totals, out=np.zeros_like(new), where=totals > WEIGHT_TOLERANCE)
    keep = totals[:, 0] > WEIGHT_TOLERANCE
    rows, new, budget = rows[keep], new[keep], budget[keep]
    current[np.ix_(rows, movable)] = budget[:, None] * new
    target['writable'].update(columns[column] for column in movable)
    for column, name in enumerate(columns):
        target['final'][name] = current[:, column].astype(np.float32)
    return bool(len(rows))


def report_rigid(operator, targets, made_rigid, holding=()):
    names = [target['obj'].name for target, rigid in zip(targets, made_rigid) if rigid]
    missed = [target['obj'].name for target, rigid in zip(targets, made_rigid) if rigid is False]
    if names:
        operator.report({'INFO'}, 'Made rigid: ' + ', '.join(names))
    if holding:
        operator.report({'INFO'}, 'Holding onto the rigid meshes: ' + ', '.join(holding))
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
