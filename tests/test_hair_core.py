"""Headless tests for Hair weights (the Hair Weights button and the Weight Brushes' Hair mode).

Run from the repository root:
    blender -b --factory-startup --python tests/test_hair_core.py
"""

import json
import math
import os
import re
import sys
import traceback

import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import cplus as cplus_fixture  # noqa: E402
import hair  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import goals, hairing, hairrig, operators  # noqa: E402
from magic_fit.cplus import rig as cplus_rig, solver  # noqa: E402
from magic_fit.cplus.template import BoneEdit, Template  # noqa: E402

FAILURES = []


def check(condition, message):
    print(("ok:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


def use_settings(**values):
    settings = bpy.context.scene.magic_fit
    defaults = dict(hair_race='c0201', hair_skeleton_mode='AUTO', hair_skeleton=0, hair_hang_start=0.025,
                    hair_hang_length=0.07, hair_neck_weight=0.3, hair_chain_spacing=hairrig.CHAIN_SPACING, hair_type='LOOSE',
                    hair_back_chains=2, hair_ponytail_chains=2, hair_side_chains_left=2, hair_side_chains_right=2,
                    hair_bang_chains=2,
                    hair_chain_stretch=0.5, hair_joint_blend=0.30, hair_layer_range=0.01, hair_ears=True, hair_max_groups=8, hair_apply_to_selected=False,
                    group_mode='ALL', mode='HAIR')
    defaults.update(values)
    for key, value in defaults.items():
        setattr(settings, key, value)
    return settings


def mesh_arrays(obj):
    matrix = np.array(obj.matrix_world)
    co = np.array([v.co[:] for v in obj.data.vertices]) @ matrix[:3, :3].T + matrix[:3, 3]
    obj.data.calc_loop_triangles()
    tris = np.array([t.vertices[:] for t in obj.data.loop_triangles])
    return co, tris


def share(weights, names, part_indices):
    """{bone: mean weight} over a part."""
    return {name: float(weights[part_indices, i].mean()) for i, name in enumerate(names)}


def join(parts):
    """Vertices and triangles of (vertices, quads) parts as one mesh, and each part's vertex indices."""
    co, tris, ranges = [], [], []
    start = 0
    for verts, quads in parts:
        co.append(np.asarray(verts, dtype=np.float64))
        for quad in quads:
            a, b, c, d = (start + k for k in quad)
            tris += [(a, b, c), (a, c, d)]
        ranges.append(np.arange(start, start + len(verts)))
        start += len(verts)
    return np.vstack(co), np.array(tris, dtype=np.int64), ranges


def max_jump(co, weights, radius):
    """The most any weight differs between two vertices closer than ``radius``."""
    near = np.triu(np.linalg.norm(co[:, None] - co[None], axis=2) < radius, 1)
    i, j = np.nonzero(near)
    return float(np.abs(weights[i] - weights[j]).max()) if len(i) else 0.0


def test_reference():
    ref = hairing.Reference.get()
    check(len(ref.races) == 18 and set(ref.races) == set(hairing.RACE_NAMES), "reference: all 18 races")
    counts = {race: len(ref.skeletons(race)) for race in ref.races}
    check(min(counts.values()) >= 50, "reference: every race has its hair skeletons ({:d} to {:d})".format(
        min(counts.values()), max(counts.values())))
    rig = ref.rig('c0201', 0)
    names = {bone.name for bone in rig.bones}
    check({'j_kao', 'j_kubi', 'j_sebo_c', 'j_kami_a', 'j_kami_b', 'j_kami_f_l', 'j_kami_f_r'} <= names,
          "reference: the body's head, neck and hair bones")
    check(abs(rig.by_name['j_kao'].head[2] - 1.4943) < 1e-3, "reference: the Midlander female's head bone at 1.494 m")
    body_hair = [b.name for b in rig.hair_bones()]
    check(not rig.skeleton and body_hair and all(name in hairing.BODY_HAIR for name in body_hair),
          "reference: no skeleton means the body's hair bones only")
    ponytail = ref.rig('c0201', 210)
    own = {bone.name: bone for bone in ponytail.skeleton_bones()}
    check(set(own) == {'j_ex_h0210_ke_a', 'j_ex_h0210_ke_b', 'j_ex_h0210_ke_c', 'j_ex_h0210_ke_f'},
          "reference: h0210 has its ponytail chain and bangs bone")
    a, b = own['j_ex_h0210_ke_a'], own['j_ex_h0210_ke_b']
    along = float(np.dot(b.head - a.head, a.x) / np.linalg.norm(b.head - a.head))
    check(along > 0.95 and b.parent == a.name,
          "reference: a hair bone points along its X axis to its child ({:.3f})".format(along))
    check(210 in ref.vanilla_hairs('c0201', 210), "reference: the game's hair 210 uses skeleton 210")
    field = ref.head('c0201')
    inside = field.distance(np.array([[0.0, -0.02, 1.55]]))[0]
    far = field.distance(np.array([[0.0, 0.1, 0.9]]))[0]
    check(inside < -0.03, "head: the middle of the skull is inside ({:.3f} m)".format(inside))
    check(far > 0.45, "head: 60 cm below the head is far from it ({:.3f} m, beyond the grid)".format(far))


def on_head(race, gap, *angles):
    """Points ``gap`` off the head of ``race`` along rays from its middle, one per (polar, azimuth) in degrees
    (azimuth as in `hair.cap`: 0 straight behind, 180 straight ahead)."""
    p, a = np.radians(np.array(angles, dtype=np.float64).reshape(-1, 2)).T
    directions = np.column_stack((np.sin(p) * np.sin(a), np.sin(p) * np.cos(a), np.cos(p)))
    return hair._onto_head(directions, gap, race)[0]


def test_scalp_lookup():
    rig = hairing.Reference.get().rig('c0801', 0)
    # The neck in front (the head ends at the neck's top; the nape behind is scalp in the game's hairs).
    top, forehead_low, neck = (on_head('c0801', 0.01, angles) for angles in ((10, 0), (85, 180), (140, 180)))
    check(rig.scalp(top)[0] > 0.8, "scalp: the crown is scalp ({:.2f})".format(rig.scalp(top)[0]))
    check(rig.scalp(forehead_low)[0] < 0.2,
          "scalp: the lower forehead is face ({:.2f})".format(rig.scalp(forehead_low)[0]))
    gap = rig.scalp_gap(top, 0.01)[0]
    check(gap <= 0.005, "scalp: a 1 cm cap on the crown is on the scalp ({:.4f} m)".format(gap))
    check(rig.scalp_gap(neck)[0] > 0.03,
          "scalp: the front of the neck is far from the scalp ({:.3f} m)".format(rig.scalp_gap(neck)[0]))
    # Below the head (it ends at the neck's top): 1 cm behind the back of the neck at z 1.25...
    surface = rig.head_field.surface
    nape = surface[(surface[:, 2] < 1.40) & (np.abs(surface[:, 0]) < 0.02)]
    low = np.array([[0.0, nape[:, 1].max() + 0.01, 1.25]])
    gap = rig.scalp_gap(low, 0.03)[0]
    check(gap > 0.03, "scalp: hair 1 cm behind the neck at z 1.25 is far from the scalp ({:.3f} m)".format(gap))
    # ...and hair 3 cm off the nape at z 1.43, the hairline.
    polar = np.arange(110.0, 160.0)
    rays = on_head('c0801', 0.03, *((p, 0.0) for p in polar))
    hairline = rays[[int(np.argmin(np.abs(rays[:, 2] - 1.43)))]]
    gap = rig.scalp_gap(hairline, 0.03)[0]
    check(abs(hairline[0, 2] - 1.43) < 0.005 and gap <= 0.02,
          "scalp: hair 3 cm off the nape at z {:.3f} is on the scalp with a 3 cm cap ({:.3f} m)".format(
              hairline[0, 2], gap))
    offset = np.array((0.02, -0.01, 0.05))
    moved = rig.moved(offset)
    points = np.vstack((top, forehead_low, neck, low, hairline))
    check(np.allclose(moved.scalp(points + offset), rig.scalp(points))
          and np.allclose(moved.scalp_gap(points + offset, 0.03), rig.scalp_gap(points, 0.03)),
          "scalp: a moved rig looks the scalp up where its head is")


def test_cap_depth():
    rig = hairing.Reference.get().rig('c0801', 0)
    for gap in (0.01, 0.02):
        co, tris, _ranges = join([hair.cap(gap=gap, race='c0801')])
        depth = hairing.cap_depth(co, tris, rig)
        check(abs(depth - gap) < 0.003, "cap depth: a cap {:.0f} cm off the head ({:.4f} m)".format(gap * 100, depth))
    offset = np.array((0.0, 0.0, 0.1))
    co, tris, _ranges = join([hair.cap(gap=0.01, race='c0801')])
    check(abs(hairing.cap_depth(co + offset, tris, rig.moved(offset)) - hairing.cap_depth(co, tris, rig)) < 1e-6,
          "cap depth: the same with the rig moved along")
    check(hairing.cap_depth(co, tris, rig, share=rig.scalp(co)) == hairing.cap_depth(co, tris, rig)
          and hairing.cap_depth(co, tris, rig, share=np.zeros(len(co))) == hairing.CAP_FALLBACK,
          "cap depth: takes scalp shares worked out before")


def test_cap_depth_fallback():
    rig = hairing.Reference.get().rig('c0801', 0)
    top, bottom = on_head('c0801', 0.015, (95, 180), (125, 180))
    co, tris, _ranges = join([hair.strip(top, bottom, 0.03, (1.0, 0.0, 0.0))])
    area = float(hairing.vertex_areas(co, tris).sum())
    check(area > hairing.CAP_MIN_AREA and float(rig.scalp(co).max()) < hairing.SCALP_HALF,
          "cap depth: the strip over the face is no scalp ({:.0f} cm², scalp share {:.2f} at most)".format(
              area * 1e4, float(rig.scalp(co).max())))
    check(hairing.cap_depth(co, tris, rig) == hairing.CAP_FALLBACK,
          "cap depth: hair with nothing over the scalp takes the fallback")


def test_weights():
    hair.clear_scene()
    obj, parts = hair.build()
    co, tris = mesh_arrays(obj)
    ref = hairing.Reference.get()
    names, weights = hairing.hair_weights(co, tris, ref.rig('c0201', 210))
    check(np.allclose(weights.sum(1), 1.0) and weights.min() >= 0.0, "weights: add up to 1 and none negative")
    cap = share(weights, names, parts['cap'])
    check(cap['j_kao'] > 0.97, "weights: the scalp follows the head ({:.3f})".format(cap['j_kao']))
    tip = parts['ponytail'][-12:]
    chain = [n for n in names if n.startswith('j_ex_h0210_ke_') and n != 'j_ex_h0210_ke_f']
    hanging = weights[tip][:, [names.index(n) for n in chain]].sum(1).mean()
    check(hanging > 0.95, "weights: the ponytail's tip hangs from the ponytail chain ({:.3f})".format(hanging))
    top = share(weights, names, parts['ponytail'][:12])
    check(top['j_kao'] > top['j_ex_h0210_ke_c'], "weights: the ponytail's root still holds on to the head")
    names0, weights0 = hairing.hair_weights(co, tris, ref.rig('c0201', 0))
    tail0 = share(weights0, names0, tip)
    check(tail0['j_kami_b'] + tail0['j_kami_a'] > 0.9, "weights: without a skeleton the ponytail hangs from j_kami_a/b")
    lock_l = share(weights, names, parts['lock_l'][-4:])
    lock_r = share(weights, names, parts['lock_r'][-4:])
    check(lock_l['j_kami_f_l'] > 0.5 and lock_l['j_kami_f_r'] < 0.01,
          "weights: the left lock hangs from j_kami_f_l ({:.2f})".format(lock_l['j_kami_f_l']))
    check(abs(lock_l['j_kami_f_l'] - lock_r['j_kami_f_r']) < 1e-3, "weights: the locks mirror each other")
    check(share(weights, names, parts['ponytail'])['j_kami_a'] < 0.01,
          "weights: the ponytail takes the skeleton's chain, not the body's back hair bones")
    # j_kami_a/b are a short chain: Chain Stretch 0 hangs the ponytail's lower part from the last bone;
    # 1 spreads the chain over the whole ponytail.
    _n, stiff = hairing.hair_weights(co, tris, ref.rig('c0201', 0), chain_stretch=0.0)
    _n, spread = hairing.hair_weights(co, tris, ref.rig('c0201', 0), chain_stretch=1.0)
    middle = parts['ponytail'][len(parts['ponytail']) // 2:len(parts['ponytail']) // 2 + 12]
    first = names0.index('j_kami_a')
    check(spread[middle, first].mean() > stiff[middle, first].mean() + 0.2,
          "weights: Chain Stretch keeps more of the ponytail's middle on the chain's first bone "
          "({:.2f} vs {:.2f})".format(
              spread[middle, first].mean(), stiff[middle, first].mean()))
    _n, near = hairing.hair_weights(co, tris, ref.rig('c0201', 210), hang_start=0.0)
    check(near[parts['bangs']].sum() > 0 and share(near, names, parts['bangs'])['j_kao'] < cap['j_kao'],
          "weights: Hang Start 0 lets the bangs move a little")
    _n, necked = hairing.hair_weights(co, tris, ref.rig('c0201', 210), neck_weight=1.0)
    check(share(necked, names, parts['lock_l'])['j_kubi'] > share(weights, names, parts['lock_l'])['j_kubi'],
          "weights: Neck Weight gives the neck more of what stays with the body")


# The column sums of the long hair's weights (`hair.build`) with c0201's first hair skeleton that has bones of
# its own, by default settings, as Best Match gave them before the hair's own skeleton weighted by its chains.
BEST_MATCH_SUMS = {
    'j_sebo_c': 2.735781, 'j_kubi': 8.947516, 'j_kao': 345.84533, 'j_kami_a': 7.07853, 'j_kami_f_l': 3.460525,
    'j_kami_f_r': 3.460505, 'j_kami_b': 40.068318, 'j_ex_h0002_ke_b': 151.909326, 'j_ex_h0002_ke_f': 0.352109,
    'j_ex_h0002_ke_l': 9.075322, 'j_ex_h0002_ke_r': 9.066738,
}


def test_best_match_unchanged():
    hair.clear_scene()
    obj, _parts = hair.build()
    co, tris = mesh_arrays(obj)
    ref = hairing.Reference.get()
    number = next(n for n in ref.skeletons('c0201') if ref.rig('c0201', n).skeleton_bones())
    names, weights = hairing.hair_weights(co, tris, ref.rig('c0201', number))
    sums = dict(zip(names, weights.sum(0).tolist()))
    off = {name: round(sums.get(name, 0.0) - value, 9) for name, value in BEST_MATCH_SUMS.items()
           if abs(sums.get(name, 0.0) - value) >= 1e-6}
    others = {name: value for name, value in sums.items() if name not in BEST_MATCH_SUMS and value != 0.0}
    check(not off and not others, "best match: the weights with h{:04d} as before ({}, {})".format(number, off, others))


def test_chains():
    ref = hairing.Reference.get()
    rig = ref.rig('c0201', 210)
    # Long hair as most modders weight it: split at the ears, the back on the back chain, the front locks
    # on j_kami_f_l/r, and the bangs bone for the bangs only.
    behind = hair.strip((-0.095, 0.045, 1.52), (-0.11, 0.075, 1.22), 0.03, (0.0, 1.0, 0.0))
    front = hair.strip((0.085, -0.04, 1.53), (0.04, -0.15, 1.18), 0.025, (0.0, 1.0, 0.0))
    co, tris, (back, lock) = join([behind, front])
    names, weights = hairing.hair_weights(co, tris, rig)
    chain = [names.index(n) for n in names if n.startswith('j_ex_h0210_ke_') and n != 'j_ex_h0210_ke_f']
    tip = back[-8:]
    check(weights[tip][:, chain].sum(1).mean() > 0.8 and weights[tip, names.index('j_kami_f_r')].max() < 0.05,
          "chains: hair hanging behind the right ear takes the back chain, not the right front bone ({:.2f})".format(
              weights[tip][:, chain].sum(1).mean()))
    tip = lock[-8:]
    bangs = weights[tip, names.index('j_ex_h0210_ke_f')].max()
    check(weights[tip, names.index('j_kami_f_l')].mean() > 0.8 and bangs < 0.05,
          "chains: a front lock curving in toward the chest stays on j_kami_f_l, not the bangs bone ({:.2f})".format(
              weights[tip, names.index('j_kami_f_l')].mean()))


def layer_strips(gap):
    """Two strips of hair hanging down the back of the head, the outer one ``gap`` off the inner one along
    their normal: (inner, outer), each (vertices, quads), the vertices matching."""
    top, bottom = np.array((0.0, 0.11, 1.57)), np.array((0.0, 0.13, 1.27))
    inner = hair.strip(top, bottom, 0.04, (1.0, 0.0, 0.0), rows=16)
    normal = np.cross((1.0, 0.0, 0.0), bottom - top)
    normal /= np.linalg.norm(normal)
    return inner, (inner[0] + gap * normal, inner[1])


def strips_object(name, strip):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in strip[0]], [], [tuple(q) for q in strip[1]])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_layers():
    ref = hairing.Reference.get()
    rig = ref.rig('c0201', 210)
    # Two layers 6 mm apart: on its own, each would hang by its own distance from the head, so they'd
    # swing differently and pass through each other. Held together, they weigh alike.
    inner, outer = layer_strips(0.006)
    co, tris, (a, b) = join([inner, outer])
    _n, apart = hairing.hair_weights(co, tris, rig, layer_range=0.0)
    names, held = hairing.hair_weights(co, tris, rig)
    before = np.abs(apart[a] - apart[b]).sum(1).max()
    after = np.abs(held[a] - held[b]).sum(1).max()
    check(before > 0.08 and after < 0.01,
          "layers: layers 6 mm apart weigh alike (at most {:.3f} apart, {:.3f} each on its own)".format(after, before))
    check(np.allclose(held.sum(1), 1.0) and held.min() >= 0.0, "layers: weights add up to 1, none negative")
    # Double-sided hair (each triangle twice, facing both ways) comes out the same.
    _n, double = hairing.hair_weights(co, np.vstack([tris, tris[:, ::-1]]), rig)
    check(np.abs(double - held).max() < 1e-6, "layers: double-sided hair weighs the same")
    # Layers farther apart than Layer Range move on their own: the same as when nothing is held (only
    # smoothed along each layer, as everywhere).
    inner, far = layer_strips(0.03)
    co, tris, (a, b) = join([inner, far])
    _n, apart = hairing.hair_weights(co, tris, rig, layer_range=1e-6)
    _n, held = hairing.hair_weights(co, tris, rig)
    _n, raw = hairing.hair_weights(co, tris, rig, layer_range=0.0)
    check(np.abs(held - apart).max() < 1e-9 and np.abs(held[a] - held[b]).sum(1).max() > 0.1,
          "layers: layers 3 cm apart keep their own weights")
    check(np.abs(held - raw).sum(1).mean() < 0.05, "layers: ...smoothed a little along each ({:.3f} on average)".format(
        np.abs(held - raw).sum(1).mean()))

    # Layers in separate meshes: weighted together (Apply to All Selected), or one after the other, where
    # the later one is held to the earlier one, which keeps its weights.
    inner, outer = layer_strips(0.006)
    for together in (True, False):
        hair.clear_scene()
        use_settings(hair_skeleton_mode='PICK', hair_skeleton=210, hair_apply_to_selected=together)
        first = strips_object("Inner", inner)
        second = strips_object("Outer", outer)
        bpy.context.view_layer.objects.active = first
        first.select_set(True)
        second.select_set(together)
        bpy.ops.magic_fit.hair_weights()
        kept = hair.by_name(first)
        if not together:
            first.select_set(False)
            second.select_set(True)
            bpy.context.view_layer.objects.active = second
            bpy.ops.magic_fit.hair_weights()
        w1, w2 = hair.by_name(first), hair.by_name(second)
        bones = sorted(set(w1) | set(w2))
        zero = np.zeros(len(first.data.vertices))
        gap = max(float(np.abs(w1.get(n, zero) - w2.get(n, zero)).max()) for n in bones)
        check(gap < 0.01, "layers: layers in two meshes weighted {:s} weigh alike (at most {:.3f} apart)".format(
            "together" if together else "one after the other", gap))
        if not together:
            check(max(float(np.abs(w1[n] - kept[n]).max()) for n in kept) < 1e-6,
                  "layers: ...and the mesh weighted first keeps its weights")
            # The brush paints the same weights on the second mesh: held to the first one too.
            stroke = hairing.build_stroke(bpy.context, second, bpy.context.scene.magic_fit)
            count = len(second.data.vertices)
            stroke.apply(list(range(count)), [1.0] * count)
            second.data.update()
            painted = hair.by_name(second)
            check(max(float(np.abs(painted[n] - w2[n]).max()) for n in w2) < 1e-4,
                  "layers: the brush paints the same weights, held to the other mesh")


def bangs_mesh(name="Bangs"):
    """The bangs of `hair.parts` as a mesh of their own (not selected)."""
    verts, faces = [], []
    for part, v, f in hair.parts(long=False):
        if part == 'bangs':
            start = len(verts)
            verts += [tuple(p) for p in v]
            faces += [tuple(start + i for i in face) for face in f]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_parts():
    # One hair in two meshes, given Hair Weights one after the other: the second part is matched along
    # with the first, joins its armature and gets its skeleton.
    hair.clear_scene()
    use_settings()
    first, _parts = hair.build()
    check(bpy.ops.magic_fit.hair_weights() == {'FINISHED'}, "parts: the first part gets hair weights")
    armature = next(m.object for m in first.modifiers if m.type == 'ARMATURE')
    number = first.magic_fit_hair.skeleton
    second = bangs_mesh()
    first.select_set(False)
    bpy.context.view_layer.objects.active = second
    second.select_set(True)
    check(bpy.ops.magic_fit.hair_weights() == {'FINISHED'}, "parts: the second part gets hair weights")
    armatures = [m.object for m in second.modifiers if m.type == 'ARMATURE']
    check(len(bpy.data.armatures) == 1 and armatures == [armature], "parts: the second part joins the hair's armature")
    check(second.magic_fit_hair.skeleton == number,
          "parts: ...and gets the hair's skeleton ({:s}), not its own match".format(hairing.skeleton_label(number)))


def test_selected():
    # With Apply to All Selected, every selected mesh is a part of one hair: one skeleton and armature for
    # all, each weighted and tagged.
    hair.clear_scene()
    settings = use_settings(hair_apply_to_selected=True)
    first, _parts = hair.build()
    second = bangs_mesh()
    left_alone = bangs_mesh("Other")
    second.select_set(True)
    check(len(operators.hair_targets(bpy.context, settings)) == 2, "selected: the selected meshes are the hair's parts")
    result = bpy.ops.magic_fit.hair_weights()
    armatures = {m.object for obj in (first, second) for m in obj.modifiers if m.type == 'ARMATURE'}
    number = first.magic_fit_hair.skeleton
    check(result == {'FINISHED'} and len(armatures) == 1 and len(bpy.data.armatures) == 1,
          "selected: both parts get one armature")
    check(second.magic_fit_hair.skeleton == number and second.magic_fit_hair.matches == first.magic_fit_hair.matches,
          "selected: ...and one skeleton ({:s}), matched on the whole hair".format(hairing.skeleton_label(number)))
    totals = [sum(hair.by_name(obj).values()) for obj in (first, second)]
    check(all(len(t) and np.abs(t - 1.0).max() < 1e-4 for t in totals), "selected: both parts are weighted")
    check(all(obj.get(hairing.EST_PROPERTY) == (number or None) for obj in (first, second)),
          "selected: both parts are tagged with the EST entry")
    check(not left_alone.vertex_groups and not left_alone.modifiers, "selected: a mesh not selected is left alone")


def test_replace():
    # Hair Weights replaces a hair's old weights: other bones' groups go, a locked group and a modifier's
    # mask stay, and each vertex's bone weights add up to 1.
    hair.clear_scene()
    obj, parts = hair.build()
    everything = list(range(len(obj.data.vertices)))
    obj.vertex_groups.new(name='j_ex_h0170_ke_a').add(everything, 1.0, 'REPLACE')
    obj.vertex_groups.new(name='j_kao').add(everything, 0.5, 'REPLACE')
    obj.vertex_groups.new(name='j_ex_h0170_ke_b').add(everything, 0.0, 'REPLACE')
    tip = parts['ponytail'][-12:].tolist()
    locked = obj.vertex_groups.new(name='j_kubi')
    locked.add(tip, 0.25, 'REPLACE')
    locked.lock_weight = True
    obj.vertex_groups.new(name='Smooth Mask').add(everything[:50], 0.8, 'REPLACE')
    obj.modifiers.new("Smooth", 'SMOOTH').vertex_group = 'Smooth Mask'
    obj.vertex_groups.new(name='Pin').add(everything[:50], 1.0, 'REPLACE')
    cloth = obj.modifiers.new("Cloth", 'CLOTH')
    cloth.settings.vertex_group_mass = 'Pin'
    cloth.show_viewport = False
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=210)
    result = bpy.ops.magic_fit.hair_weights()
    groups = {g.name for g in obj.vertex_groups}
    check(result == {'FINISHED'} and not {'j_ex_h0170_ke_a', 'j_ex_h0170_ke_b'} & groups,
          "replace: the old skeleton's groups are gone, zero weights too ({:s})".format(", ".join(sorted(groups))))
    weights = hair.by_name(obj)
    check('Smooth Mask' in weights and np.allclose(weights['Smooth Mask'][:50], 0.8),
          "replace: a modifier's mask keeps its weights")
    check('Pin' in weights and np.allclose(weights['Pin'][:50], 1.0), "replace: cloth's pin group keeps its weights")
    check(np.allclose(weights['j_kubi'][tip], 0.25), "replace: a locked group keeps its weights")
    bones = sum(w for name, w in weights.items() if name not in ('Smooth Mask', 'Pin'))
    check(np.abs(bones - 1.0).max() < 1e-4, "replace: every vertex's bone weights add up to 1 ({:.3f} to {:.3f})".format(
        bones.min(), bones.max()))
    check(weights['j_kao'][parts['cap']].mean() > 0.97, "replace: the scalp follows the head, not its old half weight")
    check(weights['j_ex_h0210_ke_c'][parts['ponytail'][-12:]].sum() > 0, "replace: the ponytail hangs from h0210")


def test_pick():
    # A picked EST entry still gets the best matches to suggest, and Try Another EST Entry weights the hair
    # with the one picked there.
    hair.clear_scene()
    obj, _parts = hair.build()
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=210)
    bpy.ops.magic_fit.hair_weights()
    matches = operators.parse_matches(obj.magic_fit_hair.matches)
    check(obj.magic_fit_hair.skeleton == 210 and len(matches) >= 5,
          "pick: a picked entry still gets suggestions ({:d})".format(len(matches)))
    check(obj.get(hairing.EST_PROPERTY) == 210 and obj.get(hairing.EST_RACE_PROPERTY) == 'c0201',
          "pick: the hair is tagged with EST entry 210 for c0201")
    other = next(n for n, _s in matches if n not in (0, 210))
    result = bpy.ops.magic_fit.pick_hair_skeleton('EXEC_DEFAULT', skeleton=str(other), apply=True)
    settings = bpy.context.scene.magic_fit
    check(result == {'FINISHED'} and obj.magic_fit_hair.skeleton == other and obj.get(hairing.EST_PROPERTY) == other
          and settings.hair_skeleton == other, "pick: Try Another EST Entry weights the hair with it and keeps it picked")
    check(operators.parse_matches(obj.magic_fit_hair.matches) == matches, "pick: ...keeping the suggestions")
    bpy.ops.magic_fit.pick_hair_skeleton('EXEC_DEFAULT', skeleton='0', apply=True)
    check(obj.magic_fit_hair.skeleton == 0 and hairing.EST_PROPERTY not in obj and
          hairing.EST_RACE_PROPERTY not in obj, "pick: no hair skeleton takes the EST tag off")
    check(not [g.name for g in obj.vertex_groups if g.name.startswith('j_ex_')],
          "pick: ...and the skeleton's groups")


def test_ears():
    ref = hairing.Reference.get()
    rig = ref.rig('c0801', 0)
    # Miqo'te hair without cat ears (most modded hair): two layers over the scalp, half a centimetre and a
    # centimetre and a half off it, leaving the head's own ears free.
    inner = hair.cap(gap=0.005, race='c0801', reach=0.12)
    outer = hair.cap(gap=0.015, race='c0801', reach=0.12)
    co, tris, (scalp, _outer) = join([inner, outer])
    names, weights = hairing.hair_weights(co, tris, rig)
    ears = [names.index(name) for name in hairing.EARS]
    check(weights[:, ears].max() == 0.0, "ears: hair without cat ears gives the ear bones nothing")
    jump = max_jump(co, weights, 0.012)
    check(jump < 0.3, "ears: no spot of the scalp hair leaves the head: layers a centimetre apart weigh alike "
                      "(at most {:.2f} apart)".format(jump))
    # With cat ears, each follows its ear bone, blending into the scalp hair at its base.
    co, tris, ranges = join([inner, outer] + [hair.ear(rig.by_name[name]) for name in hairing.EARS])
    names, weights = hairing.hair_weights(co, tris, rig)
    tips = {}
    for side, (name, other) in enumerate((hairing.EARS, hairing.EARS[::-1])):
        bone = rig.by_name[name]
        mine = ranges[2 + side]
        tip = tips[name] = mine[(co[mine] - bone.head) @ bone.x > 0.06]
        check(weights[tip, names.index(name)].min() > 0.9 and weights[tip, names.index(other)].max() == 0.0,
              "ears: the tip of the cat ear on {:s} follows it ({:.2f})".format(
                  name, weights[tip, names.index(name)].min()))
    heads = np.array([rig.by_name[name].head for name in hairing.EARS])
    far = np.concatenate(ranges[:2])
    far = far[np.linalg.norm(co[far, None] - heads[None], axis=2).min(1) > 0.08]
    check(len(far) and weights[far][:, ears].max() == 0.0, "ears: scalp hair away from the ears stays with the head")
    jump = max_jump(co, weights, 0.01)
    check(jump < 0.5, "ears: the ears blend into the scalp hair without a jump (at most {:.2f})".format(jump))
    _n, off = hairing.hair_weights(co, tris, rig, ears=False)
    check(off[:, ears].max() == 0.0, "ears: Ears off leaves the ear bones out")
    _n, other = hairing.hair_weights(co, tris, ref.rig('c0201', 0))
    check(other[:, ears].max() == 0.0, "ears: only Miqo'te hair has cat ears")
    # The button does the same for a Miqo'te hair with ears.
    hair.clear_scene()
    mesh = bpy.data.meshes.new("Hair")
    mesh.from_pydata([tuple(p) for p in co], [], [tuple(t) for t in tris.tolist()])
    obj = bpy.data.objects.new("Hair", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    use_settings(hair_race='c0801')
    result = bpy.ops.magic_fit.hair_weights()
    groups = hair.by_name(obj)
    check(result == {'FINISHED'} and all(name in groups and groups[name][tips[name]].min() > 0.9
                                         for name in hairing.EARS),
          "ears: Hair Weights gives a Miqo'te hair's cat ears to the ear bones")


def test_matching():
    hair.clear_scene()
    obj, _parts = hair.build(long=True)
    co, tris = mesh_arrays(obj)
    ref = hairing.Reference.get()
    ranked = hairing.rank_skeletons(co, tris, 'c0201')
    check(len(ranked) == len(ref.skeletons('c0201')) + 1 and len({n for n, _s in ranked}) == len(ranked),
          "matching: every skeleton and none, each once")
    best = ref.rig('c0201', ranked[0][0])
    depth = max((len(c.bones) for c in hairing.chains_of(best.skeleton_bones())), default=0)
    back = [b for b in best.skeleton_bones() if b.head[1] > best.by_name['j_kao'].head[1]]
    check(depth >= 3 and back,
          "matching: long hair with a ponytail gets a skeleton with a chain down the back ({:s}, {:d} deep)".format(
        hairing.skeleton_label(ranked[0][0]), depth))
    hair.clear_scene()
    obj, _parts = hair.build(long=False)
    co, tris = mesh_arrays(obj)
    ranked_short = hairing.rank_skeletons(co, tris, 'c0201')
    check(ranked_short[0][0] == 0, "matching: short hair that doesn't hang gets no hair skeleton")
    moved = hairing.rank_skeletons(co + (0.5, 0.0, 0.0), tris, 'c0201', offset=np.array((0.5, 0.0, 0.0)))
    check([n for n, _s in moved[:5]] == [n for n, _s in ranked_short[:5]], "matching: the offset moves the head along")


def test_button():
    hair.clear_scene()
    obj, parts = hair.build()
    settings = use_settings()
    check(operators.check_hair_weights_ready(bpy.context) is None,
          "button: ready for a mesh without weights or armature")
    result = bpy.ops.magic_fit.hair_weights()
    check(result == {'FINISHED'}, "button: runs")
    modifier = obj.modifiers.get("Armature")
    armature = modifier.object if modifier else None
    check(armature is not None and armature.type == 'ARMATURE', "button: made an armature and an Armature modifier")
    state = obj.magic_fit_hair
    check(state.race == 'c0201' and state.skeleton > 0 and state.matches.startswith(str(state.skeleton) + ":"),
          "button: remembers the race, the skeleton ({:d}) and the best matches".format(state.skeleton))
    rig = hairing.Reference.get().rig('c0201', state.skeleton)
    bones = armature.data.bones
    check(all(bones.get(b.name) for b in rig.hair_bones()) and bones.get('j_kao') and bones.get('j_kubi'),
          "button: the armature has the skeleton's bones")
    head = armature.matrix_world @ bones['j_kao'].head_local
    check((np.array(head) - rig.by_name['j_kao'].head).max() < 1e-4, "button: bones sit where the game has them")
    x_axis = np.array(bones[rig.skeleton[0]].matrix_local.to_3x3().col[0])
    check(np.dot(x_axis, rig.by_name[rig.skeleton[0]].x) > 0.999,
          "button: bones keep the game's axes (TexTools convention)")
    weights = hair.by_name(obj)
    total = sum(weights.values())
    check(float(np.abs(total - 1.0).max()) < 1e-4, "button: every vertex's weights add up to 1")
    check(weights['j_kao'][parts['cap']].mean() > 0.97, "button: the scalp follows the head")
    counts = [len(v.groups) for v in obj.data.vertices]
    check(max(counts) <= 8, "button: at most 8 bone weights per vertex")
    bone_count = len(bones)
    before = {k: v.copy() for k, v in weights.items()}
    result = bpy.ops.magic_fit.hair_weights()
    check(len(armature.data.bones) == bone_count and len([o for o in bpy.data.objects if o.type == 'ARMATURE']) == 1,
          "button again: no new armature or bones")
    after = hair.by_name(obj)
    check(max(float(np.abs(after[k] - before[k]).max()) for k in before) < 1e-5, "button again: the same weights")
    check(obj.get(hairing.EST_PROPERTY) == state.skeleton and obj.get(hairing.EST_RACE_PROPERTY) == 'c0201',
          "button: tags the hair with its EST entry ({!r}) and race".format(obj.get(hairing.EST_PROPERTY)))
    # A picked skeleton, and one of the other matches.
    old = {g.name for g in obj.vertex_groups if g.name.startswith('j_ex_') and not g.name.startswith('j_ex_h0210')}
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=210)
    bpy.ops.magic_fit.hair_weights()
    check(obj.magic_fit_hair.skeleton == 210 and armature.data.bones.get('j_ex_h0210_ke_a') is not None,
          "button: a picked skeleton adds its bones")
    groups = {g.name for g in obj.vertex_groups}
    check(not old & groups, "button: the previous skeleton's groups ({:d}) are removed".format(len(old)))
    check(obj.get(hairing.EST_PROPERTY) == 210, "button: the EST tag follows the picked skeleton")
    use_settings()
    other = operators.parse_matches(state.matches)[1][0]
    bpy.ops.magic_fit.hair_weights(skeleton=other)
    settings = bpy.context.scene.magic_fit
    picked = settings.hair_skeleton_mode == 'PICK' and settings.hair_skeleton == other
    check(obj.magic_fit_hair.skeleton == other and picked,
          "button: trying another match keeps it picked")


def test_existing_armature():
    """A body armature elsewhere (like a devkit placed off the origin): the hair bones hang from its head."""
    hair.clear_scene()
    obj, parts = hair.build()
    offset = np.array((0.4, -0.2, 0.05))
    obj.location = offset
    bpy.context.view_layer.update()
    ref = hairing.Reference.get()
    base = ref.rig('c0201', 0)
    data = bpy.data.armatures.new("Body")
    body = bpy.data.objects.new("Body", data)
    bpy.context.scene.collection.objects.link(body)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.mode_set(mode='EDIT')
    from mathutils import Matrix
    for bone in base.bones:
        edit = data.edit_bones.new(bone.name)
        edit.head, edit.tail = (0, 0, 0), (0, 0.05, 0)
        rest = bone.rest.copy()
        rest[:3, 3] += offset
        edit.matrix = Matrix(rest.tolist())
    for bone in base.bones:
        if bone.parent is not None and bone.parent in data.edit_bones:
            data.edit_bones[bone.name].parent = data.edit_bones[bone.parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    hairing.add_armature_modifier(obj, body)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=210)
    count = len(data.bones)
    result = bpy.ops.magic_fit.hair_weights()
    check(result == {'FINISHED'} and len([o for o in bpy.data.objects if o.type == 'ARMATURE']) == 1,
          "armature: uses the hair's armature, makes none")
    check(len(data.bones) == count + 4, "armature: adds only the skeleton's 4 bones")
    added = np.array(body.matrix_world @ data.bones['j_ex_h0210_ke_a'].head_local)
    expected = ref.rig('c0201', 210).by_name['j_ex_h0210_ke_a'].head + offset
    check(np.abs(added - expected).max() < 1e-4, "armature: they hang from the armature's head where it is")
    check(data.bones['j_ex_h0210_ke_a'].parent.name == 'j_kao', "armature: parented to its head bone")
    weights = hair.by_name(obj)
    check(weights['j_ex_h0210_ke_c'][parts['ponytail'][-12:]].sum() > 0,
          "armature: the moved hair gets the same kind of weights")


def test_customize_plus():
    """With Customize+ on, the hair follows its armature's hidden rig: the bones go on the armature, and the
    rig is made again with them."""
    hair.clear_scene()
    body = cplus_fixture.build_rig(name="Body", bones=cplus_fixture.HEAD_CHAIN)
    obj, _parts = hair.build()
    hairing.add_armature_modifier(obj, body)
    template = Template("Big Head", 7, {'j_kao': BoneEdit(scaling=(1.2, 1.2, 1.2))})
    rig, _plan = cplus_rig.build(bpy.context, body, template, solver.Options())
    cplus_rig.connect(body, rig)
    check(hairing.weighted_armature(obj) == body, "C+: the hair's armature is the body's, not its rig")
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=210)
    result = bpy.ops.magic_fit.hair_weights()
    check(result == {'FINISHED'} and body.data.bones.get('j_ex_h0210_ke_a') is not None
          and len([o for o in bpy.data.objects if o.type == 'ARMATURE']) == 2,
          "C+: the bones go on the armature, and no armature is made")
    rig = cplus_rig.rig_of(body)
    check(cplus_rig.is_on(body) and rig.data.bones.get('j_ex_h0210_ke_a') is not None,
          "C+: the rig is made again with them, and stays on")
    check(bpy.context.object == obj and obj.mode == 'WEIGHT_PAINT', "C+: the hair stays in Weight Paint mode")
    bpy.ops.object.mode_set(mode='OBJECT')


def test_armature_out_of_view_layer():
    """Bones can't be added to an armature outside the view layer: Hair Weights says so and changes nothing."""
    hair.clear_scene()
    obj, _parts = hair.build()
    away = bpy.data.objects.new("Away", bpy.data.armatures.new("Away"))  # in no collection
    hairing.add_armature_modifier(obj, away)
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=210)
    try:
        result = bpy.ops.magic_fit.hair_weights()
    except RuntimeError:  # the error it reports
        result = {'CANCELLED'}
    check(result == {'CANCELLED'} and not away.data.bones, "outside: cancelled, no bones added")
    check(bpy.context.object == obj and obj.mode == 'WEIGHT_PAINT', "outside: the hair stays in Weight Paint mode")
    bpy.ops.object.mode_set(mode='OBJECT')
    # A new armature goes in the hair's first collection: when that one is excluded, it's taken away again.
    obj.modifiers.clear()
    data = away.data
    bpy.data.objects.remove(away)
    bpy.data.armatures.remove(data)
    hidden, shown = (bpy.data.collections.new(name) for name in ("A Excluded", "B Shown"))
    for collection in (hidden, shown):
        bpy.context.scene.collection.children.link(collection)
        collection.objects.link(obj)
    bpy.context.scene.collection.objects.unlink(obj)
    bpy.context.view_layer.layer_collection.children[hidden.name].exclude = True
    bpy.context.view_layer.objects.active = obj
    check(obj.users_collection[0] == hidden, "outside: the hair's first collection is the excluded one")
    try:
        result = bpy.ops.magic_fit.hair_weights()
    except RuntimeError:
        result = {'CANCELLED'}
    check(result == {'CANCELLED'} and not [o for o in bpy.data.objects if o.type == 'ARMATURE']
          and not bpy.data.armatures, "outside: the armature made for the hair is removed again")
    for collection in (hidden, shown):
        bpy.data.collections.remove(collection)


def test_masks():
    # Groups a modifier uses, wherever it keeps their names, aren't bone weights.
    hair.clear_scene()
    obj, _parts = hair.build()
    names = ('Pin', 'Stiff', 'Self', 'Shell', 'Rim', 'Edit Mask', 'Goal', 'Nodes In', 'Nodes Out')
    for name in names:
        obj.vertex_groups.new(name=name)
    cloth = obj.modifiers.new("Cloth", 'CLOTH')
    cloth.settings.vertex_group_mass = 'Pin'
    cloth.settings.vertex_group_bending = 'Stiff'
    cloth.collision_settings.vertex_group_self_collisions = 'Self'
    solidify = obj.modifiers.new("Solidify", 'SOLIDIFY')
    solidify.shell_vertex_group, solidify.rim_vertex_group = 'Shell', 'Rim'
    obj.modifiers.new("Edit", 'VERTEX_WEIGHT_EDIT').mask_vertex_group = 'Edit Mask'
    obj.modifiers.new("Soft", 'SOFT_BODY').settings.vertex_group_goal = 'Goal'
    tree = bpy.data.node_groups.new("Masks", 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    factor = tree.interface.new_socket("Factor", in_out='INPUT', socket_type='NodeSocketFloat')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    output = tree.interface.new_socket("Out", in_out='OUTPUT', socket_type='NodeSocketFloat')
    nodes = obj.modifiers.new("Nodes", 'NODES')
    nodes.node_group = tree
    if hasattr(nodes, "properties"):
        socket = getattr(nodes.properties.inputs, factor.identifier)
        socket.type, socket.attribute_name = 'ATTRIBUTE', 'Nodes In'
        getattr(nodes.properties.outputs, output.identifier).attribute_name = 'Nodes Out'
    else:
        nodes[factor.identifier + "_use_attribute"] = True
        nodes[factor.identifier + "_attribute_name"] = 'Nodes In'
        nodes[output.identifier + "_attribute_name"] = 'Nodes Out'
    for modifier in obj.modifiers:
        modifier.show_viewport = False
    missed = set(names) - goals.mask_groups(obj)
    check(not missed, "masks: cloth, solidify, vertex weight, soft body and geometry nodes groups{:s}".format(
        " (missed {:s})".format(", ".join(sorted(missed))) if missed else ""))
    check(goals.remove_empty_groups(obj) == [], "masks: empty ones aren't removed")
    bpy.data.node_groups.remove(tree)


def test_brush():
    hair.clear_scene()
    obj, parts = hair.build()
    settings = use_settings()
    bpy.ops.magic_fit.hair_weights()
    check(operators.check_ready(bpy.context, settings) is None, "brush: ready once the hair has its weights")
    stroke = hairing.build_stroke(bpy.context, obj, settings)
    before = hair.by_name(obj)
    count = len(obj.data.vertices)
    stroke.apply(list(range(count)), [1.0] * count)
    obj.data.update()
    after = hair.by_name(obj)
    check(max(float(np.abs(after[k] - before[k]).max()) for k in before) < 1e-4,
          "brush: painting the goal again changes nothing")
    # Wipe some weights to the head, then paint them back.
    group = obj.vertex_groups['j_kao']
    tip = parts['ponytail'][-24:].tolist()
    for g in obj.vertex_groups:
        g.remove(tip)
    group.add(tip, 1.0, 'REPLACE')
    stroke = hairing.build_stroke(bpy.context, obj, settings)
    stroke.apply(tip, [1.0] * len(tip))
    obj.data.update()
    painted = hair.by_name(obj)
    check(max(float(np.abs(painted[k][tip] - before[k][tip]).max()) for k in before) < 1e-4,
          "brush: paints the hair weights back")
    # A locked group keeps its weights.
    obj.vertex_groups['j_kao'].lock_weight = True
    locked = hair.by_name(obj)['j_kao'].copy()
    use_settings(hair_neck_weight=0.9)
    stroke = hairing.build_stroke(bpy.context, obj, bpy.context.scene.magic_fit)
    stroke.apply(list(range(count)), [1.0] * count)
    obj.data.update()
    check(np.allclose(hair.by_name(obj)['j_kao'], locked), "brush: a locked group keeps its weights")
    obj.vertex_groups['j_kao'].lock_weight = False
    # Without Hair Weights first, the brush says what to do.
    hair.clear_scene()
    obj, _parts = hair.build()
    use_settings()
    problem = operators.check_ready(bpy.context, bpy.context.scene.magic_fit)
    check(problem is not None and "Hair Weights" in problem,
          "brush: a hair without weights asks for Hair Weights first")


def test_readiness():
    hair.clear_scene()
    obj, _parts = hair.build()
    use_settings(group_mode='ACTIVE')
    problem = operators.check_hair_weights_ready(bpy.context)
    check(problem is not None and "All Groups" in problem, "button: Current Group is refused")
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=9999)
    problem = operators.check_hair_weights_ready(bpy.context)
    check(problem is not None and "h9999" in problem, "button: a skeleton the race doesn't have is refused")
    use_settings()
    bpy.ops.object.mode_set(mode='EDIT')
    problem = operators.check_hair_weights_ready(bpy.context)
    check(problem is not None and "Object or Weight Paint" in problem, "button: not in Edit Mode")
    bpy.ops.object.mode_set(mode='OBJECT')
    items = operators._hair_skeleton_items(None, bpy.context)
    check(items[0][0] == "0" and any(item[0] == "210" and "210" in item[1] for item in items),
          "picker: lists none and the race's skeletons with the game's hairs using them")


class _HairWeightsRun:
    """Stands in for the Hair Weights operator's ``self``, to read what it reports."""

    def __init__(self, skeleton=-1, rebuild=False):
        self.skeleton = skeleton
        self.rebuild = rebuild
        self.reports = []

    def report(self, _kinds, text):
        self.reports.append(text)


def hair_weights_reports(**props):
    """Run Hair Weights (its execute, after its readiness check): (result, the texts it reported)."""
    problem = operators.check_hair_weights_ready(bpy.context)
    if problem is not None:
        return {'CANCELLED'}, [problem]
    run = _HairWeightsRun(**props)
    result = operators.MAGIC_FIT_OT_hair_weights.execute(run, bpy.context)
    print("      reports:", run.reports)
    return result, run.reports


def own_skeleton_hair(**values):
    """The long hair (`hair.build`), active and selected, with Own Skeleton in the settings."""
    hair.clear_scene()
    obj, parts = hair.build()
    settings = use_settings(hair_skeleton_mode='CUSTOM', **values)
    return obj, parts, settings


def hair_armature(obj):
    modifier = next((m for m in obj.modifiers if m.type == 'ARMATURE'), None)
    return modifier.object if modifier is not None else None


def rig_chains(armature):
    """(kind, bone names) of each chain in the armature's rig state."""
    return [(chain.kind, [name for name in chain.bones.split(",") if name])
            for chain in armature.magic_fit_hair_rig.chains]


def own_bones(armature):
    return sorted(bone.name for bone in armature.data.bones if bone.name.startswith(hairrig.BONE_PREFIX))


def bone_head(armature, name):
    return np.array(armature.matrix_world @ armature.data.bones[name].head_local)


def ponytail_root(armature):
    """The first bone of the armature's ponytail chain, or None."""
    return next((bones[0] for kind, bones in rig_chains(armature) if kind == 'ponytail' and bones), None)


def move_bone_head(obj, armature, name, offset):
    """Move bone ``name``'s head by ``offset`` in Edit Mode, then make ``obj`` active again."""
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit = armature.data.edit_bones[name]
    edit.head = edit.head + Vector(offset)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


VANILLA_HAIR_BONE = re.compile(r"j_ex_h[0-9]{4}_")


def test_custom_button():
    obj, parts, _settings = own_skeleton_hair()
    obj[hairing.EST_PROPERTY] = 210
    obj[hairing.EST_RACE_PROPERTY] = 'c0201'
    check(operators.check_hair_weights_ready(bpy.context) is None, "own skeleton: ready for a mesh without armature")
    result, reports = hair_weights_reports()
    armature = hair_armature(obj)
    check(result == {'FINISHED'} and armature is not None and armature.name == "Hair Skeleton",
          "own skeleton: made the armature 'Hair Skeleton'")
    if armature is None:
        return
    bones = armature.data.bones
    own = own_bones(armature)
    check(all(bones.get(name) for name in (hairing.HEAD, hairing.NECK, hairing.UPPER_BACK)) and len(own) >= 4,
          "own skeleton: j_kao, j_kubi, j_sebo_c and the hair's own bones ({:d})".format(len(own)))
    data = json.loads(armature.get(hairrig.RIG_PROPERTY, "{}"))
    chains = data.get("chains", [])
    check(data.get("race") == 'c0201' and sorted(name for chain in chains for name in chain["bones"]) == own,
          "own skeleton: the rig property lists c0201 and every chain bone")
    groups = {group.name for group in obj.vertex_groups}
    check(groups and all(bones.get(name) for name in groups),
          "own skeleton: the mesh has groups only for the armature's bones ({:s})".format(", ".join(sorted(groups))))
    check(not [name for name in groups if name.startswith("j_kami_") or VANILLA_HAIR_BONE.match(name)],
          "own skeleton: no j_kami_* and no vanilla hair skeleton groups")
    weights = hair.by_name(obj)
    total = sum(weights.values())
    check(float(np.abs(total - 1.0).max()) < 1e-4, "own skeleton: every vertex's weights add up to 1")
    tip = parts['ponytail'][-12:]
    chain_weight = sum(w[tip] for name, w in weights.items() if name.startswith(hairrig.BONE_PREFIX))
    check(float(chain_weight.mean()) > 0.9,
          "own skeleton: the ponytail's tip hangs from its chain ({:.3f})".format(float(chain_weight.mean())))
    check(weights[hairing.HEAD][parts['cap']].mean() > 0.97, "own skeleton: the scalp follows the head")
    state = obj.magic_fit_hair
    check(state.custom and state.skeleton == 0 and state.race == 'c0201' and state.matches == "",
          "own skeleton: the hair remembers its own skeleton (skeleton 0, no matches)")
    check(hairing.EST_PROPERTY not in obj and hairing.EST_RACE_PROPERTY not in obj,
          "own skeleton: the EST tags are taken off")
    expected = "Built {:d} chains with {:d} bones for the hair".format(len(chains), len(own))
    check(expected in reports, "own skeleton: reports '{:s}'".format(expected))
    # A picked skeleton afterwards: the hair no longer uses its own.
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=210)
    bpy.ops.magic_fit.hair_weights()
    check(not obj.magic_fit_hair.custom and obj.magic_fit_hair.skeleton == 210 and obj.get(hairing.EST_PROPERTY) == 210,
          "own skeleton: a picked skeleton afterwards clears it and tags the EST entry")


def test_custom_reuses_rig():
    # Hair Type Loose builds no ponytail chain: the tube is the one ponytail chain with Hair Type Ponytail.
    obj, _parts, _settings = own_skeleton_hair(hair_type='PONYTAIL', hair_ponytail_chains=1)
    hair_weights_reports()
    armature = hair_armature(obj)
    name = ponytail_root(armature) if armature is not None else None
    check(name is not None, "reuse: the hair has a ponytail chain ({})".format(
        rig_chains(armature) if armature is not None else None))
    if name is None:
        return
    joint = bone_head(armature, name)
    count = len(own_bones(armature))
    chains = len(armature.magic_fit_hair_rig.chains)
    before = hair.by_name(obj)
    move_bone_head(obj, armature, name, (0.01, 0.0, 0.0))
    result, reports = hair_weights_reports()
    moved = bone_head(armature, name)
    check(result == {'FINISHED'} and np.linalg.norm(moved - joint - (0.01, 0.0, 0.0)) < 1e-5,
          "reuse: Hair Weights again keeps the bone where it was moved")
    after = hair.by_name(obj)
    change = max(float(np.abs(after.get(k, 0.0) - before.get(k, 0.0)).max()) for k in set(before) | set(after))
    check(change > 1e-3, "reuse: ...and the weights follow it ({:.4f})".format(change))
    expected = "Used the hair's own skeleton on '{:s}': {:d} chains, {:d} bones".format(armature.name, chains, count)
    check(expected in reports, "reuse: reports '{:s}'".format(expected))
    check(not [text for text in reports if text.startswith("Built ")], "reuse: builds no chains")
    result = bpy.ops.magic_fit.hair_weights(rebuild=True)
    check(result == {'FINISHED'} and np.linalg.norm(bone_head(armature, name) - joint) < 1e-5,
          "reuse: Rebuild puts the bone back on its joint")


def test_custom_chain_spacing():
    # Chain Spacing, Hair Type and the chain counts reach the chains as they are built, and Rebuild Chains builds
    # them with the new values.
    obj, _parts, settings = own_skeleton_hair(hair_chain_spacing=0.04, hair_side_chains_left=1)
    seen = []
    plan_layout = hairrig.plan_layout

    def recording(*args, **kwargs):
        seen.append((kwargs.get("spacing"), kwargs.get("ponytail"), kwargs.get("counts")))
        return plan_layout(*args, **kwargs)

    hairrig.plan_layout = recording
    try:
        hair_weights_reports()
        settings.hair_chain_spacing = 0.09
        settings.hair_type = 'PONYTAIL'
        settings.hair_ponytail_chains = 3
        result = bpy.ops.magic_fit.hair_weights(rebuild=True)
    finally:
        hairrig.plan_layout = plan_layout
    check(result == {'FINISHED'} and len(seen) == 2 and None not in (seen[0][0], seen[1][0])
          and abs(seen[0][0] - 0.04) < 1e-6 and abs(seen[1][0] - 0.09) < 1e-6
          and seen[0][1] is False and seen[1][1] is True
          and seen[0][2] == hairrig.ChainCounts(2, 2, 1, 2, 2) and seen[1][2] == hairrig.ChainCounts(2, 3, 1, 2, 2),
          "chain spacing: the chains are built with Chain Spacing, Hair Type and the counts ({})".format(seen))
    armature = hair_armature(obj)
    kinds = [kind for kind, _bones in rig_chains(armature)] if armature is not None else []
    check(kinds.count("ponytail") == 3 and "back" not in kinds,
          "hair type: Ponytail with 3 Ponytail Chains builds 3 ponytail chains and no back chains ({})".format(kinds))


def test_no_hair_note():
    # An area asked for that gets no chain says so; one that gets fewer than asked says how many.
    settings = use_settings(hair_side_chains_left=2)
    made = {"back": 2, "ponytail": 0, "left": 0, "right": 2, "bangs": 2}
    notes = operators.chain_count_notes(settings, made, False)
    check("No hair found for Left Side Chains: none built" in notes, "no hair note: says so ({})".format(notes))
    notes = operators.chain_count_notes(settings, dict(made, left=1), False)
    check("Built 1 of 2 Left Side chains: too little hair there for more" in notes
          and not [note for note in notes if note.startswith("No hair found")],
          "no hair note: 1 of 2 built says how many ({})".format(notes))


def test_zero_chain_counts():
    # A count of 0 builds no chains in its area and asks for none, so it gets no "No hair found" note; with every
    # count 0 the build says so rather than that no hair hangs.
    obj, _parts, _settings = own_skeleton_hair(hair_back_chains=0)
    result, reports = hair_weights_reports()
    armature = hair_armature(obj)
    kinds = [kind for kind, _bones in rig_chains(armature)] if armature is not None else []
    check(result == {'FINISHED'} and "back" not in kinds and kinds.count("side") >= 2
          and not [text for text in reports if "Back Chains" in text],
          "zero counts: Back Chains 0 builds no back chain and no note ({}; {})".format(kinds, reports))
    use_settings(hair_skeleton_mode='CUSTOM', hair_back_chains=0, hair_side_chains_left=0, hair_side_chains_right=0,
                 hair_bang_chains=0)
    result, reports = hair_weights_reports(rebuild=True)
    check(result == {'FINISHED'} and not own_bones(armature) and "Every chain count is 0: no chains built" in reports,
          "zero counts: every count 0 builds no chains and says so ({})".format(reports))


def test_detect_hair_chains():
    # Detect sets the counts to the ones the hair gets without any: on the long hair one chain per side lock and one
    # for the back, its tube (with Hair Type Loose a tube is hair of the area it hangs in, and narrower than Chain
    # Spacing it gets one lane, `hairlanes.lane_count`). Its bangs hang too little for a chain, so Bang Chains stays.
    # Built with them, each side lock has one chain and the bangs none.
    obj, _parts, settings = own_skeleton_hair(hair_back_chains=4, hair_bang_chains=5)
    co, tris = mesh_arrays(obj)
    base = hairing.Reference.get().rig('c0201', 0)
    expected, _short = hairrig.detect_counts(co, tris, base)
    run = _HairWeightsRun()
    result = operators.MAGIC_FIT_OT_detect_hair_chains.execute(run, bpy.context)
    print("      reports:", run.reports)
    check(result == {'FINISHED'} and expected == {"back": 1, "left": 1, "right": 1}
          and settings.hair_side_chains_left == 1 and settings.hair_side_chains_right == 1
          and settings.hair_bang_chains == 5 and settings.hair_back_chains == 1,
          "detect: 1 back, 1 per side, Bang Chains kept ({}; left {:d}, right {:d}, bangs {:d}, back {:d})".format(
              expected, settings.hair_side_chains_left, settings.hair_side_chains_right, settings.hair_bang_chains,
              settings.hair_back_chains))
    check("Detected 1 Back, 1 Left Side, 1 Right Side chains" in run.reports
          and "No hair found for Bang Chains: kept as they are" in run.reports,
          "detect: reports what it found and what it kept")
    result, reports = hair_weights_reports()
    armature = hair_armature(obj)
    kinds = [kind for kind, _bones in rig_chains(armature)] if armature is not None else []
    check(kinds.count("side") == 2 and kinds.count("back") == 1 and "bangs" not in kinds,
          "detect: built with the counts, one chain per side lock and one back ({})".format(kinds))
    check("No hair found for Bang Chains: none built" in reports, "detect: the build says the bangs got none")


def test_race_change_rebuilds():
    obj, _parts, _settings = own_skeleton_hair(hair_type='PONYTAIL', hair_ponytail_chains=1)
    hair_weights_reports()
    armature = hair_armature(obj)
    name = ponytail_root(armature) if armature is not None else None
    check(name is not None, "race change: a ponytail chain for c0201")
    if name is None:
        return
    before = [bone_head(armature, bones[0]) for _kind, bones in rig_chains(armature) if bones]
    use_settings(hair_skeleton_mode='CUSTOM', hair_race='c0801')
    result, reports = hair_weights_reports()
    data = json.loads(armature.get(hairrig.RIG_PROPERTY, "{}"))
    check(result == {'FINISHED'} and data.get("race") == 'c0801' and armature.magic_fit_hair_rig.race == 'c0801'
          and obj.magic_fit_hair.race == 'c0801', "race change: the rig is for c0801 now")
    check(any(text.startswith("Built ") for text in reports), "race change: the chains are built again")
    # A chain starts where its hair leaves the held hair, on a row of the mesh: on this coarse hair that is the
    # same row on either head (the chains start where they did), so the chains are checked against those
    # planned for c0801's head.
    after = [bone_head(armature, bones[0]) for _kind, bones in rig_chains(armature) if bones]
    co, tris = mesh_arrays(obj)
    base = hairing.Reference.get().rig('c0801', 0)
    plans = hairrig.plan_rig(co, tris, base.moved(hairing.armature_offset(armature, base)), ponytail=True,
                             counts=operators.hair_chain_counts(bpy.context.scene.magic_fit))
    gap = max((float(np.linalg.norm(head - plan.joints[0])) for head, plan in zip(after, plans)), default=math.inf)
    check(len(after) == len(plans) and gap < 1e-5, "race change: the chains are planned for c0801's head ({}, "
          "{:.2e} m off; {:d} of them where they were for c0201)".format(
              [kind for kind, _bones in rig_chains(armature)], gap,
              sum(min(float(np.linalg.norm(head - old)) for old in before) < 1e-6 for head in after)))


def test_nothing_hangs():
    hair.clear_scene()
    use_settings(hair_skeleton_mode='CUSTOM', hair_type='PONYTAIL', hair_ponytail_chains=1)
    obj = strips_object("Cap", hair.cap())
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    result, reports = hair_weights_reports()
    armature = hair_armature(obj)
    check(result == {'FINISHED'} and armature is not None and not own_bones(armature),
          "nothing hangs: no hair bones of its own")
    if armature is None:
        return
    data = json.loads(armature.get(hairrig.RIG_PROPERTY, "{}"))
    check(data == {"race": 'c0201', "version": 2, "chains": []},
          "nothing hangs: the rig property lists no chains ({})".format(data))
    check("No hair hangs from the head: no chains built" in reports,
          "nothing hangs: says so")
    custom = hair.by_name(obj)
    # A ponytail joined to the cap later: the rig without chains has no bones to keep, so it is built again.
    co, tris, _ranges = join([hair.cap(), hair.tube((0.0, 0.105, 1.53), (0.0, 0.16, 1.11), 0.035)])
    mesh = bpy.data.meshes.new("Cap")
    mesh.from_pydata([tuple(p) for p in co], [], [tuple(t) for t in tris.tolist()])
    obj.data = mesh
    result, reports = hair_weights_reports()
    chains = json.loads(armature.get(hairrig.RIG_PROPERTY, "{}")).get("chains")
    check(result == {'FINISHED'} and own_bones(armature) and chains
          and any(text.startswith("Built ") for text in reports),
          "nothing hangs: hair hanging later gets chains without Rebuild ({:d} bones)".format(len(own_bones(armature))))
    expected = "Built 1 chain with {:d} bones for the hair".format(len(own_bones(armature)))
    check(chains and len(chains) == 1 and expected in reports, "nothing hangs: reports '{:s}'".format(expected))
    hair.clear_scene()
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=0)
    obj = strips_object("Cap", hair.cap())
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.magic_fit.hair_weights()
    none = hair.by_name(obj)
    same = set(custom) == set(none) and all(np.abs(custom[k] - none[k]).max() < 1e-5 for k in custom)
    check(same, "nothing hangs: the weights are as with no hair skeleton ({} vs {})".format(
        sorted(custom), sorted(none)))


def test_custom_brush():
    obj, _parts, settings = own_skeleton_hair(hair_type='PONYTAIL', hair_ponytail_chains=1)
    check(hairing.uses_custom_rig(obj, settings), "own brush: a new hair uses its own skeleton in Own Skeleton mode")
    problem = operators.check_hair_ready(obj, settings)
    check(problem is not None and "Hair Weights" in problem, "own brush: a hair without weights asks for Hair Weights")
    bpy.ops.magic_fit.hair_weights()
    armature = hair_armature(obj)
    check(operators.check_hair_ready(obj, settings) is None, "own brush: ready once the hair has its weights")
    stroke = hairing.build_stroke(bpy.context, obj, settings)
    check(stroke.rig.custom and sorted(stroke.rig.skeleton) == own_bones(armature),
          "own brush: the stroke paints with the hair's own skeleton")
    before = hair.by_name(obj)
    count = len(obj.data.vertices)
    stroke.apply(list(range(count)), [1.0] * count)
    obj.data.update()
    after = hair.by_name(obj)
    check(max(float(np.abs(after[k] - before[k]).max()) for k in before) < 1e-4,
          "own brush: painting the goal again changes nothing")
    # Best Match in the settings: the hair still paints with its own skeleton.
    settings = use_settings()
    check(hairing.uses_custom_rig(obj, settings) and hairing.build_stroke(bpy.context, obj, settings).rig.custom,
          "own brush: the hair keeps its own skeleton whatever the settings")
    # A moved bone: the brush works the weights out again.
    move_bone_head(obj, armature, ponytail_root(armature), (0.01, 0.0, 0.0))
    moved = hairing.build_stroke(bpy.context, obj, settings)
    check(not np.array_equal(moved.goal_weights, stroke.goal_weights), "own brush: a moved bone moves the goal")
    # So do the cap depth and the light chains the rig keeps.
    state = armature.magic_fit_hair_rig
    cap = state.cap_depth
    state.cap_depth = cap + 0.05
    deeper = hairing.build_stroke(bpy.context, obj, settings)
    check(not np.array_equal(deeper.goal_weights, moved.goal_weights), "own brush: a changed cap depth moves the goal")
    state.cap_depth = cap
    for chain in state.chains:
        chain.light = True
    lighter = hairing.build_stroke(bpy.context, obj, settings)
    # This hair has nothing over the scalp standing off the cap for light chains to change: the key tells.
    check(lighter.rig.light and hairing._brush_cache[0][-2:] == (lighter.rig.cap_depth, tuple(sorted(lighter.rig.light))),
          "own brush: the weights kept are those of the light chains ({})".format(sorted(lighter.rig.light)))
    del armature[hairrig.RIG_PROPERTY]
    problem = operators.check_hair_ready(obj, settings)
    check(problem == "Give the hair Hair Weights first",
          "own brush: no rig for the race asks for Hair Weights ({})".format(problem))
    try:
        hairing.build_stroke(bpy.context, obj, settings)
        refused = False
    except hairing.HairError:
        refused = True
    check(refused, "own brush: ...and the stroke refuses")


def test_custom_view_layer():
    # Own Skeleton edits the armature's bones every time: an armature outside the view layer is refused.
    hair.clear_scene()
    obj, _parts = hair.build()
    rig_collection = bpy.data.collections.new("Rig")
    bpy.context.scene.collection.children.link(rig_collection)
    armature = hairing.new_armature(bpy.context, "Away", rig_collection)
    hairing.add_armature_modifier(obj, armature)
    bpy.context.view_layer.layer_collection.children[rig_collection.name].exclude = True
    use_settings(hair_skeleton_mode='CUSTOM')
    problem = operators.check_hair_weights_ready(bpy.context)
    check(problem == "Armature 'Away' is not in the view layer: Own Skeleton edits its bones",
          "own view layer: refused with Own Skeleton ({})".format(problem))
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=210)
    check(operators.check_hair_weights_ready(bpy.context) is None, "own view layer: Pick isn't refused up front")
    bpy.data.collections.remove(rig_collection)


def test_custom_parts():
    # One part with a vanilla skeleton, the other with the hair's own: each warns about the other.
    hair.clear_scene()
    use_settings(hair_skeleton_mode='PICK', hair_skeleton=210)
    first, _parts = hair.build()
    bpy.ops.magic_fit.hair_weights()
    second = bangs_mesh()
    first.select_set(False)
    bpy.context.view_layer.objects.active = second
    second.select_set(True)
    use_settings(hair_skeleton_mode='CUSTOM', hair_type='PONYTAIL', hair_ponytail_chains=1)
    result, reports = hair_weights_reports()
    armature = hair_armature(second)
    check(result == {'FINISHED'} and armature == hair_armature(first) and second.magic_fit_hair.custom,
          "own parts: the second part joins the hair's armature with its own skeleton")
    check(armature is not None and ponytail_root(armature) is not None,
          "own parts: the chains are built over the whole hair (the first part's ponytail too)")
    check(any(text.startswith("The hair's other part 'Hair' still uses another hair skeleton") for text in reports),
          "own parts: the part with a vanilla skeleton is named")
    # The first part with the same own skeleton: nothing to warn about.
    second.select_set(False)
    bpy.context.view_layer.objects.active = first
    first.select_set(True)
    result, reports = hair_weights_reports()
    check(result == {'FINISHED'} and first.magic_fit_hair.custom
          and not [text for text in reports if text.startswith("The hair's other part")],
          "own parts: parts with the hair's own skeleton match")
    # Rebuilt, the chains change under the other part's weights.
    result, reports = hair_weights_reports(rebuild=True)
    check("The chains were rebuilt: give 'Bangs' Hair Weights again" in reports
          and not [text for text in reports if text.startswith("The hair's other part")],
          "own parts: after a rebuild the other part is asked to get Hair Weights again")


def test_custom_other_race_parts():
    # A part weighted with the armature's own skeleton for c0201, another part on the same armature for
    # c0801: the chains are built for c0801, taking the first part's bones away. It is named.
    obj, _parts, _settings = own_skeleton_hair()
    hair_weights_reports()
    armature = hair_armature(obj)
    if armature is None:
        check(False, "other race: the hair got an armature")
        return
    second = bangs_mesh()
    hairing.add_armature_modifier(second, armature)
    obj.select_set(False)
    bpy.context.view_layer.objects.active = second
    second.select_set(True)
    use_settings(hair_skeleton_mode='CUSTOM', hair_race='c0801')
    result, reports = hair_weights_reports()
    check(result == {'FINISHED'} and hair_armature(second) == armature and armature.magic_fit_hair_rig.race == 'c0801',
          "other race: the second part's chains are built for c0801 on the same armature")
    check("'Hair' used this armature's own skeleton for another race: give it Hair Weights again" in reports,
          "other race: the part weighted for c0201 is named")
    # Weighted again for c0801, nobody else uses the skeleton for another race: nothing to say.
    result, reports = hair_weights_reports()
    check(not [text for text in reports if "for another race" in text], "other race: ...once only")


def parts_object(name, wanted):
    """The parts ``wanted`` of `hair.parts(long=True)` as one mesh, active and the only one selected."""
    verts, faces = [], []
    for part, v, f in hair.parts(long=True):
        if part in wanted:
            start = len(verts)
            verts += [tuple(p) for p in v]
            faces += [tuple(start + i for i in face) for face in f]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for other in bpy.context.selected_objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def test_custom_parts_one_after_another():
    # The cap with the temple locks weighted first, a ponytail in a mesh of its own later: the ponytail
    # joins the rig, and its chains are built again with it while the bones are as Hair Weights left them.
    for edited in (False, True):
        hair.clear_scene()
        # The cap and locks alone are short hair, one ring of Bang Chains; with the ponytail they are side hair. 4
        # sectors and 1 chain per side give each lock one chain from the same root either way, so the kind picked
        # stays with its lock.
        use_settings(hair_skeleton_mode='CUSTOM', hair_bang_chains=4, hair_side_chains_left=1,
                     hair_side_chains_right=1, hair_type='PONYTAIL', hair_ponytail_chains=1)
        first = parts_object("Hair", ("cap", "lock_l", "lock_r"))
        hair_weights_reports()
        armature = hair_armature(first)
        before = rig_chains(armature) if armature is not None else []
        check(before and not [kind for kind, _bones in before if kind == 'ponytail'],
              "one after another: the cap and locks have no ponytail chain ({})".format(before))
        if armature is None or not before:
            return
        moved = before[0][1][0]
        if edited:
            move_bone_head(first, armature, moved, (0.01, 0.0, 0.0))
        else:
            # A kind picked in the list isn't a bone edit: the chains are still built again, keeping it.
            armature.magic_fit_hair_rig.chains[0].kind = 'back'
        joint = bone_head(armature, moved)
        second = parts_object("Ponytail", ("ponytail",))
        result, reports = hair_weights_reports()
        check(result == {'FINISHED'} and hair_armature(second) == armature and second.magic_fit_hair.custom,
              "one after another{:s}: the ponytail joins the hair's armature".format(" (edited)" if edited else ""))
        warning = "'Ponytail' hangs where no chain may reach: Rebuild Chains to include it (your bone edits are lost)"
        if edited:
            check(np.linalg.norm(bone_head(armature, moved) - joint) < 1e-6 and rig_chains(armature) == before,
                  "one after another (edited): the edited bones are kept")
            check(warning in reports, "one after another (edited): says the ponytail may hang from no chain")
            continue
        check(warning not in reports, "one after another: no warning while the bones are as they were built")
        check("The chains were rebuilt: give 'Hair' Hair Weights again" in reports,
              "one after another: the first part is asked to get Hair Weights again")
        picked = [kind for kind, bones in rig_chains(armature)
                  if bones and armature.data.bones.get(bones[0]) is not None
                  and np.linalg.norm(bone_head(armature, bones[0]) - joint) < 1e-4]
        check(picked == ['back'], "one after another: the chain picked as back stays back ({})".format(picked))
        chain = next((bones for kind, bones in rig_chains(armature) if kind == 'ponytail'), None)
        check(chain is not None, "one after another: the chains are built again with a ponytail chain ({})".format(
            rig_chains(armature)))
        if chain is None:
            continue
        co, part_tris = mesh_arrays(second)
        reference = hairing.Reference.get().rig('c0201', 0)
        # The hair that hangs (goes into locks), by the scalp and the part's cap depth.
        hanging = hairrig._hanging(reference.scalp(co), reference.head_distance(co),
                                   hairing.cap_depth(co, part_tris, reference))
        weights = hair.by_name(second)
        zero = np.zeros(len(co))
        own = sum((weights.get(name, zero) for name in chain), zero)[hanging]
        every = sum((w for name, w in weights.items() if name.startswith(hairrig.BONE_PREFIX)), zero)[hanging]
        ratio = float((own / np.maximum(every, 1e-12)).mean())
        check(hanging.any() and ratio > 0.8 and float(own.mean()) > 0.5,
              "one after another: the ponytail's hanging hair is on its chain ({:.3f} of its chain weight, "
              "{:.3f} of all)".format(ratio, float(own.mean())))


def pieces_object(name, pieces):
    """(vertices, faces) ``pieces`` as one mesh, active and the only one selected."""
    verts, faces = [], []
    for v, f in pieces:
        start = len(verts)
        verts += [tuple(p) for p in v]
        faces += [tuple(start + i for i in face) for face in f]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for other in bpy.context.selected_objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def test_custom_parts_thick_cap():
    # A hair with a thick cap (5 cm off the head) weighted first, its bones edited, then another layer of the
    # cap 6 cm off the head in a mesh of its own: within the hair's cap depth, it hangs nowhere.
    hair.clear_scene()
    use_settings(hair_skeleton_mode='CUSTOM')
    long = {name: (v, f) for name, v, f in hair.parts(long=True)}
    first = pieces_object("Hair", [hair.cap(gap=0.05), long["lock_l"], long["lock_r"], long["ponytail"]])
    hair_weights_reports()
    armature = hair_armature(first)
    chains = rig_chains(armature) if armature is not None else []
    state = armature.magic_fit_hair_rig if armature is not None else None
    check(chains and abs(state.cap_depth - 0.05) < 0.005,
          "thick cap: the rig keeps the hair's cap depth ({})".format(state.cap_depth if state else None))
    if not chains:
        return
    for name, old in (("Layer", False), ("Old Layer", True)):
        if old:
            # A rig built before the cap depth was kept: measured from the joining part.
            state.cap_depth = -1.0
        move_bone_head(first, armature, chains[0][1][0], (0.005, 0.0, 0.0))
        second = pieces_object(name, [hair.cap(gap=0.06, lowest=80.0)])
        result, reports = hair_weights_reports()
        check(result == {'FINISHED'} and hair_armature(second) == armature and rig_chains(armature) == chains,
              "thick cap{:s}: the layer joins the hair's armature, the edited bones kept".format(
                  " (old rig)" if old else ""))
        check(not [text for text in reports if "hangs where no chain may reach" in text],
              "thick cap{:s}: no warning for the layer within the cap".format(" (old rig)" if old else ""))
        check(state.cap_depth == -1.0 if old else abs(state.cap_depth - 0.05) < 0.005,
              "thick cap{:s}: the stored cap depth stays ({:.4f})".format(" (old rig)" if old else "", state.cap_depth))
        older = [text for text in reports if text.startswith("Chains from an older version")]
        check(bool(older) == old, "thick cap{:s}: {:s} to rebuild the chains ({})".format(
            " (old rig)" if old else "", "asked" if old else "not asked", older))


def edit_select(obj, indices):
    """Edit Mode on obj with exactly ``indices`` selected."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='OBJECT')
    sel = np.zeros(len(obj.data.vertices), dtype=bool)
    sel[indices] = True
    obj.data.vertices.foreach_set("select", sel)
    # Edges and faces too, or Edit Mode brings back the faces selected before.
    for element in list(obj.data.edges) + list(obj.data.polygons):
        element.select = bool(sel[list(element.vertices)].all())
    bpy.ops.object.mode_set(mode='EDIT')


def test_add_chain_kept_by_hair_weights():
    obj, parts, _settings = own_skeleton_hair()
    hair_weights_reports()
    armature = hair_armature(obj)
    # The built chain down the lock goes first, so the lock hangs from the added one alone.
    state = armature.magic_fit_hair_rig
    top = np.array((0.085, -0.035, 1.53))
    state.active = min(range(len(state.chains)), key=lambda i: float(np.linalg.norm(
        bone_head(armature, state.chains[i].bones.split(",")[0]) - top)))
    check(bpy.ops.magic_fit.delete_hair_chain() == {'FINISHED'}, "add chain: the lock's built chain deleted first")
    chains = len(armature.magic_fit_hair_rig.chains)
    edit_select(obj, parts['lock_l'])
    check(operators.check_add_chain_ready(bpy.context) is None, "add chain: ready in Edit Mode")
    check(bpy.ops.magic_fit.add_hair_chain() == {'FINISHED'} and obj.mode == 'EDIT',
          "add chain: done, still in Edit Mode")
    selected = np.zeros(len(obj.data.vertices), dtype=bool)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.data.vertices.foreach_get("select", selected)
    check(sorted(np.flatnonzero(selected)) == sorted(parts['lock_l']), "add chain: the selection is kept")
    new = rig_chains(armature)[-1][1]
    check(len(armature.magic_fit_hair_rig.chains) == chains + 1, "add chain: one chain more")
    result, reports = hair_weights_reports()
    check(result == {'FINISHED'} and rig_chains(armature)[-1][1] == new and not [t for t in reports if t.startswith("Built ")],
          "add chain: Hair Weights keeps it and builds nothing")
    weights = hair.by_name(obj)
    tip = float(sum(weights.get(n, 0.0) for n in new)[parts['lock_l'][-4:]].mean())
    check(tip > 0.5, "add chain: the lock's tip hangs from it ({:.3f})".format(tip))


def test_add_chain_without_armature():
    obj, parts, _settings = own_skeleton_hair()
    edit_select(obj, parts['ponytail'])
    check(bpy.ops.magic_fit.add_hair_chain() == {'FINISHED'}, "add chain, no armature: done")
    armature = hair_armature(obj)
    check(armature is not None and armature.name == "Hair Skeleton" and len(rig_chains(armature)) == 1
          and armature.magic_fit_hair_rig.cap_depth >= 0.0,
          "add chain, no armature: made 'Hair Skeleton' with one chain and the hair's cap depth")
    bpy.ops.object.mode_set(mode='OBJECT')
    result, reports = hair_weights_reports()
    check(result == {'FINISHED'} and len(rig_chains(armature)) == 1
          and not [t for t in reports if t.startswith("Chains from an older version")],
          "add chain, no armature: Hair Weights keeps the one chain, no old-version warning")


def test_add_chain_refusals():
    obj, parts, settings = own_skeleton_hair()
    check(operators.check_add_chain_ready(bpy.context) == "Select the hair's vertices in Edit Mode",
          "add chain: not in Object Mode")
    edit_select(obj, parts['bangs'][:2])
    # An ERROR report makes bpy.ops raise in a script.
    try:
        bpy.ops.magic_fit.add_hair_chain()
        refused = None
    except RuntimeError as error:
        refused = str(error)
    check(refused is not None and "shorter than 3 cm" in refused, "add chain: refuses 2 vertices ({})".format(refused))
    settings.hair_skeleton_mode = 'AUTO'
    check(operators.check_add_chain_ready(bpy.context) == "Add Chain is for Own Skeleton: set Skeleton to Own Skeleton",
          "add chain: only for Own Skeleton")
    bpy.ops.object.mode_set(mode='OBJECT')


def test_delete_chain_kept_by_hair_weights():
    obj, _parts, _settings = own_skeleton_hair()
    hair_weights_reports()
    armature = hair_armature(obj)
    before = rig_chains(armature)
    armature.magic_fit_hair_rig.active = 0
    check(bpy.ops.magic_fit.delete_hair_chain() == {'FINISHED'} and rig_chains(armature) == before[1:],
          "delete chain: the picked chain is gone")
    result, reports = hair_weights_reports()
    check(result == {'FINISHED'} and rig_chains(armature) == before[1:] and not any(
        g.name in before[0][1] for g in obj.vertex_groups), "delete chain: Hair Weights keeps it deleted and drops its groups")
    # The key outside the sidebar passes through (headless: no region). Without a window Blender runs execute for
    # INVOKE_DEFAULT, so invoke is called directly; test_hair_ui presses the real key.
    class Pressed:
        key = True

        def execute(self, _context):
            return {'FINISHED'}

    check(operators.MAGIC_FIT_OT_delete_hair_chain.invoke(Pressed(), bpy.context, None) == {'PASS_THROUGH'},
          "delete chain: the Delete key passes through outside the sidebar")


def test_deleted_chain_survives_joining_part():
    # Deleted, given Hair Weights (which reads the chains back), then a part joins: the chains stay as the user left
    # them, not built again over the whole hair.
    hair.clear_scene()
    use_settings(hair_skeleton_mode='CUSTOM', hair_bang_chains=4, hair_side_chains_left=1, hair_side_chains_right=1)
    first = parts_object("Hair", ("cap", "lock_l", "lock_r"))
    hair_weights_reports()
    armature = hair_armature(first)
    armature.magic_fit_hair_rig.active = 0
    bpy.ops.magic_fit.delete_hair_chain()
    hair_weights_reports()
    kept = rig_chains(armature)
    parts_object("Ponytail", ("ponytail",))
    result, _reports = hair_weights_reports()
    check(result == {'FINISHED'} and rig_chains(armature) == kept,
          "deleted chain: a part joining after Hair Weights keeps the chains as the user left them ({} -> {})".format(
              len(kept), len(rig_chains(armature))))


def test_add_chain_on_another_part():
    # The hair's other part has its own skeleton already: a chain added on this part goes on that armature.
    hair.clear_scene()
    use_settings(hair_skeleton_mode='CUSTOM', hair_bang_chains=4, hair_side_chains_left=1, hair_side_chains_right=1)
    first = parts_object("Hair", ("cap", "lock_l", "lock_r"))
    hair_weights_reports()
    armature = hair_armature(first)
    chains = len(rig_chains(armature))
    second = parts_object("Ponytail", ("ponytail",))
    edit_select(second, np.arange(len(second.data.vertices)))
    result = bpy.ops.magic_fit.add_hair_chain()
    bpy.ops.object.mode_set(mode='OBJECT')
    armatures = [obj.name for obj in bpy.data.objects if obj.type == 'ARMATURE']
    check(result == {'FINISHED'} and hair_armature(second) == armature and len(rig_chains(armature)) == chains + 1
          and armatures == [armature.name], "add chain: on the hair's other part, it goes on the hair's armature ({})".format(
              armatures))


def main():
    magic_fit.register()
    try:
        for test in (test_reference, test_scalp_lookup, test_cap_depth, test_cap_depth_fallback, test_weights,
                     test_best_match_unchanged, test_chains, test_layers, test_ears, test_matching, test_button,
                     test_existing_armature, test_customize_plus, test_armature_out_of_view_layer, test_parts,
                     test_selected, test_replace, test_masks, test_pick, test_brush, test_readiness,
                     test_custom_button, test_custom_reuses_rig, test_custom_chain_spacing, test_no_hair_note,
                     test_zero_chain_counts, test_detect_hair_chains,
                     test_race_change_rebuilds,
                     test_nothing_hangs,
                     test_custom_brush, test_custom_view_layer, test_custom_parts, test_custom_parts_one_after_another,
                     test_custom_other_race_parts, test_custom_parts_thick_cap,
                     test_add_chain_kept_by_hair_weights, test_add_chain_without_armature, test_add_chain_refusals,
                     test_delete_chain_kept_by_hair_weights, test_deleted_chain_survives_joining_part,
                     test_add_chain_on_another_part):
            print("==", test.__name__)
            try:
                test()
            except Exception:
                traceback.print_exc()
                FAILURES.append(test.__name__ + ": exception")
    finally:
        magic_fit.unregister()
    print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
    sys.stdout.flush()
    os._exit(1 if FAILURES else 0)


main()
