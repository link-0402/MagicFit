"""Headless tests for Hair weights (the Hair Weights button and the Weight Brushes' Hair mode).

Run from the repository root:
    blender -b --factory-startup --python tests/test_hair_core.py
"""

import math
import os
import sys
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import cplus as cplus_fixture  # noqa: E402
import hair  # noqa: E402
import magic_fit  # noqa: E402
from magic_fit import goals, hairing, operators  # noqa: E402
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
                    hair_hang_length=0.07, hair_neck_weight=0.3, hair_chain_stretch=0.5, hair_joint_blend=0.30,
                    hair_ears=True, hair_max_groups=8, hair_apply_to_selected=False, group_mode='ALL', mode='HAIR')
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


def main():
    magic_fit.register()
    try:
        for test in (test_reference, test_weights, test_chains, test_ears, test_matching, test_button,
                     test_existing_armature, test_customize_plus, test_armature_out_of_view_layer, test_parts,
                     test_selected, test_replace, test_masks, test_pick, test_brush, test_readiness):
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
