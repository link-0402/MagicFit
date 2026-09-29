"""A test scene for Customize+: an FFXIV body skeleton with a body and a garment skinned to it.

Rig: 41 bones of Yet Another Devkit's skeleton (vanilla, IVCS and YAS), placed and turned like the game's
(the add-on's reference table), in one of these styles, the way different importers leave them:
  GAME  Z up, each bone on the game's axes (TexTools FBX with default settings, Yet Another Devkit)
  YUP   the armature in the game's Y-up space turned upright by the object, every bone's X and Y
        swapped and Z flipped (like the IVCS Ready Anim File)
  CM    centimetres (object scaled 0.01), every bone's axes cycled (like the Animation Kit's rig)
  FREE  game axes, but some bones turned freely about themselves (automatic bone orientation)
The world-space skeleton is the same in every style.

Face and hair rigs (`build_head`): the chain from the root to the head, then the face bones of a
TexTools face skeleton, or hair bones: two of the body skeleton's and two of a hairstyle's own, which the
reference table doesn't have (they differ for every hairstyle). A placeholder rig has its bones at the
origin without parents, the way an armature made from a model's bone list alone is.

Body: a small sphere of points around each bone's head, weighted to the bone and blended with its
parent, so blends between bones are covered. Garment: the same, slightly larger, as its own object.
"""

import math

import bpy
import bmesh
import numpy as np
from mathutils import Matrix

from magic_fit.cplus import reference, solver

# A template copied from Customize+ 2.2.1 (version 7): scaling and positions, no rotation or propagation.
SUPERTHICC = (
    "H4sIAAAAAAAACuXYTW+bMBgA4OP+RuVrAdkGDOa47JLDpmrpPi/IAadxSiAzUKmq+O8THyakS9IuzcENOYF5I3j0vv788AS+c5mLLAWBZ4BvqfhT8mk"
    "MAkA5cSBC0LSJ65oOwbFJMbZNl3kwZpQSbDvAABPJWSGy9BMrOAgAhpiY0DcRvEUkQDhwXQtjDyOXXkMcQAgM8DmLxULweM9f/AC7gUMtG3k+Jf0/vr"
    "B1HTkp8yJbX5lXs3LDZbEUUXQNDPAxS3kOgiewClkuQhYm9c2NzDbsjhX8VrI0T5qPBMGCJTk3tg+/ZsWBJ7OIJbxvnixFEtdNIr2bpjHf8DTmadE/3"
    "3nJE/gJAmhBZIBfzYUBfrcNlQG2b1Rhz6MqA3Rv6mJQF4MsmzRByHKqyui9UkuvuR9cNzdx5mnknM+zcK6JeE/qTsuv22nriyaK+tsMz7WpaEVBRyr6"
    "fxOMWjKyMB2SpZZk8xy9+B/xuky5djl+a01jr+MSxSVDrnw33NcW9GFwXdGRdgmGvsJA91Q1tGjzU3TkqdJ2h3SpJ908YH9Ngb8oL2Ou0VrkPL1aoYmj"
    "1LZT33b07m7ol6P26zdzn8mv+LWXquGuuxvy5Tj5abhcipW40OyrxSqybd/G7VBaL10pJcQe8uVl811oe6hdFyILEg9R3PDFQ5gvhdTU3yxhj0x6/Xxv"
    "7l/m+K0fd1HYwjvm5D2ajy/s9otX4X2Wi0vbiD7foqAuudG4NilbsRzNtqw5WWGXWtBYHax4tfaRhYvyvpyX4Wb5mGtmtt+KRr46Turn5g7djtG1Wc+u"
    "/ML5yvG5qRmgh7WN2+Fr161rhz7xWOkIOmaiYDpn+wz9GuOdbbd4GIV6Oz37e3ItR5lrOYZcVwaY5j+kKPiNzAoeFTzuv2aWlTLiIIDVX8iJK27WGwAA"
)

PARENTS = {
    'n_root': None, 'n_hara': 'n_root', 'j_kosi': 'n_hara', 'j_sebo_a': 'n_hara', 'j_sebo_b': 'j_sebo_a',
    'j_sebo_c': 'j_sebo_b', 'j_kubi': 'j_sebo_c', 'j_kao': 'j_kubi', 'j_mune_l': 'j_sebo_b',
    'j_mune_r': 'j_sebo_b', 'iv_c_mune_l': 'j_mune_l', 'iv_c_mune_r': 'j_mune_r', 'j_sako_l': 'j_sebo_c',
    'j_sako_r': 'j_sebo_c', 'j_ude_a_l': 'j_sako_l', 'j_ude_a_r': 'j_sako_r', 'j_ude_b_l': 'j_ude_a_l',
    'j_ude_b_r': 'j_ude_a_r', 'n_hhiji_l': 'j_ude_b_l', 'n_hhiji_r': 'j_ude_b_r', 'j_te_l': 'j_ude_b_l',
    'j_te_r': 'j_ude_b_r', 'j_asi_a_l': 'j_kosi', 'j_asi_a_r': 'j_kosi', 'j_asi_b_l': 'j_asi_a_l',
    'j_asi_b_r': 'j_asi_a_r', 'j_asi_c_l': 'j_asi_b_l', 'j_asi_c_r': 'j_asi_b_r', 'j_asi_d_l': 'j_asi_c_l',
    'j_asi_d_r': 'j_asi_c_r', 'j_asi_e_l': 'j_asi_d_l', 'j_asi_e_r': 'j_asi_d_r', 'iv_shiri_l': 'j_kosi',
    'iv_shiri_r': 'j_kosi', 'ya_fukubu_phys': 'j_kosi', 'ya_shiri_phys_l': 'j_kosi',
    'ya_shiri_phys_r': 'j_kosi', 'ya_daitai_phys_l': 'j_asi_a_l', 'ya_daitai_phys_r': 'j_asi_a_r',
    'iv_daitai_phys_l': 'j_asi_a_l', 'iv_daitai_phys_r': 'j_asi_a_r',
}
ORDER = solver.hierarchy_order(PARENTS)
HEAD_CHAIN = ['n_root', 'n_hara', 'j_sebo_a', 'j_sebo_b', 'j_sebo_c', 'j_kubi', 'j_kao']
# From TexTools' c0801f0002.skel: the face skeleton hangs from the head.
FACE_PARENTS = {
    'j_ago': 'j_kao', 'j_f_face': 'j_kao', 'j_f_hana_l': 'j_f_face', 'j_f_hana_r': 'j_f_face',
    'j_f_mab_l': 'j_f_face', 'j_f_eye_l': 'j_f_face', 'j_f_ulip_01_l': 'j_f_face', 'j_f_hoho_l': 'j_f_face',
    # Turned up to 41 degrees differently in some races (Hrothgar, Lalafell).
    'j_f_dslip_l': 'j_f_face', 'j_f_umlip_01_l': 'j_f_face', 'j_f_mabdn_01_l': 'j_f_face', 'j_f_shoho_l': 'j_f_face',
}
HAIR_PARENTS = {'j_kami_a': 'j_kao', 'j_kami_f_l': 'j_kao', 'j_ex_h0113_ke_b': 'j_kao', 'j_ex_h0113_ke_l': 'j_kao'}
ALL_PARENTS = dict(PARENTS, **FACE_PARENTS, **HAIR_PARENTS)
TO_BLENDER = solver.TO_BLENDER
SWAP_YUP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
SWAP_CM = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
# FREE: extra turns about the bone's own Y axis (degrees).
FREE_TURNS = {'j_asi_a_l': 33.0, 'j_ude_a_l': -50.0, 'j_mune_l': 45.0}
STYLES = ('GAME', 'YUP', 'CM', 'FREE')


def game_rest(name, race=None):
    """(position, rotation) of a bone in the game's model space; turned like ``race``'s (a race code of
    `reference.RACES`) if given. Positions are always the Midlander female's."""
    if name.startswith("j_ex_h"):
        # A hairstyle's own bones: behind and beside the head, turned a little from it.
        head, turn = game_rest('j_kao', race)
        offset, degrees = {'j_ex_h0113_ke_b': ((0.0, 0.06, -0.09), 15.0),
                           'j_ex_h0113_ke_l': ((0.07, 0.02, -0.03), -25.0)}[name]
        return head + np.array(offset), turn @ solver.axis_rotation(2, degrees)
    values = reference.BONES[name]
    index = reference.RACES.get(race, {}).get(name, 0)
    turn = values[3:] if not index else reference.TURNS[name][index - 1]
    return np.array(values[:3]), solver.quaternion_matrix(turn)


def world_rest(name, race=None):
    """The bone's 4x4 rest matrix in Blender's world space (Z up), on the game's axes."""
    position, rotation = game_rest(name, race)
    return solver.compose(TO_BLENDER @ position, TO_BLENDER @ rotation, 1.0)


def _style(style):
    """(object matrix, armature space from world space, per-bone frame function)."""
    if style == 'YUP':
        obj = np.array(Matrix.Rotation(math.pi / 2.0, 4, 'X'))
        return obj, np.linalg.inv(obj), lambda name: SWAP_YUP
    if style == 'CM':
        obj = np.diag([0.01, 0.01, 0.01, 1.0])
        return obj, np.linalg.inv(obj), lambda name: SWAP_CM
    if style == 'FREE':
        return np.identity(4), np.identity(4), lambda name: solver.axis_rotation(1, FREE_TURNS.get(name, 0.0))
    return np.identity(4), np.identity(4), lambda name: np.identity(3)


def build_rig(name="Skeleton", style='GAME', bones=None, placeholders=(), race=None, turned=None):
    """An armature with ``bones`` (default: the body's), those in ``placeholders`` left at the origin
    without a parent, turned like ``race``'s skeleton, and bones in ``turned`` ({bone: degrees}) turned
    further about their X axis."""
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    obj_matrix, to_armature, frame = _style(style)
    data = bpy.data.armatures.new(name)
    rig = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(rig)
    rig.matrix_world = Matrix(obj_matrix.tolist())
    bpy.context.view_layer.objects.active = rig
    for other in bpy.context.selected_objects:
        other.select_set(False)
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    names = [name for name in solver.hierarchy_order(ALL_PARENTS) if (bones is None and name in PARENTS)
             or (bones is not None and name in bones)]
    length = 0.04 * (100.0 if style == 'CM' else 1.0)
    for bone_name in names:
        item = data.edit_bones.new(bone_name)
        item.tail = (0.0, length, 0.0)  # a bone without length can't take a matrix
        if bone_name in placeholders:
            item.head = (0.0, 0.0, 0.0)
            item.tail = (0.0, 0.0, 0.1)
            continue
        world = world_rest(bone_name, race)
        if turned and bone_name in turned:
            world[:3, :3] = world[:3, :3] @ solver.axis_rotation(0, turned[bone_name])
        local = to_armature @ world
        # Keep the bone's axes unit length (the object carries any scale).
        local[:3, :3] = local[:3, :3] / np.linalg.norm(local[:3, :3], axis=0)
        local[:3, :3] = local[:3, :3] @ frame(bone_name)
        item.matrix = Matrix(local.tolist())
    for bone_name in names:
        parent = ALL_PARENTS[bone_name]
        if parent is not None and parent in data.edit_bones and bone_name not in placeholders:
            data.edit_bones[bone_name].parent = data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    rig.select_set(False)
    return rig


def build_head(kind, name=None, style='GAME', placeholders=(), race=None):
    """A face ('FACE') or hair ('HAIR') rig with a mesh: (rig, mesh)."""
    extra = FACE_PARENTS if kind == 'FACE' else HAIR_PARENTS
    name = name or kind.title()
    rig = build_rig(name=name + " Rig", style=style, bones=HEAD_CHAIN + list(extra), placeholders=placeholders,
                    race=race)
    return rig, build_body(rig, name=name, radius=0.012, seed=len(name))


def build_body(rig, name="Body", radius=0.035, seed=0):
    """Points around each bone's head (world space), weighted to it and blended with its parent."""
    rng = np.random.default_rng(seed)
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    groups = []
    for bone_name in solver.hierarchy_order(ALL_PARENTS):
        if bone_name not in rig.data.bones:
            continue
        center = world_rest(bone_name)[:3, 3]
        result = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=radius,
                                            matrix=Matrix.Translation(center.tolist()))
        for vert in result["verts"]:
            groups.append((vert, bone_name, float(rng.uniform(0.45, 1.0))))
    bm.verts.index_update()
    groups = [(vert.index, bone_name, weight) for vert, bone_name, weight in groups]
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for index, bone_name, weight in groups:
        _add(obj, bone_name, index, weight)
        parent = ALL_PARENTS[bone_name]
        if parent is not None and parent in rig.data.bones and weight < 1.0:
            _add(obj, parent, index, 1.0 - weight)
    modifier = obj.modifiers.new("Armature", 'ARMATURE')
    modifier.object = rig
    return obj


def _add(obj, group_name, index, weight):
    group = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
    group.add([index], weight, 'ADD')


def build(style='GAME', garment=True):
    rig = build_rig(style=style)
    body = build_body(rig)
    result = {"rig": rig, "body": body}
    if garment:
        result["garment"] = build_body(rig, name="Garment", radius=0.042, seed=1)
    loose = bpy.data.objects.new("Loose", bpy.data.meshes.new("Loose"))
    bpy.context.scene.collection.objects.link(loose)
    result["loose"] = loose
    return result


def pose(rig, seed=0, degrees=25.0, move=0.01):
    """Turn every bone by up to ``degrees`` and move it by up to ``move`` (armature units)."""
    rng = np.random.default_rng(seed)
    for bone in rig.pose.bones:
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = tuple(np.radians(rng.uniform(-degrees, degrees, 3)))
        bone.location = tuple(rng.uniform(-move, move, 3))


def clear_pose(rig):
    for bone in rig.pose.bones:
        bone.location = (0.0, 0.0, 0.0)
        bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        bone.rotation_euler = (0.0, 0.0, 0.0)
        bone.scale = (1.0, 1.0, 1.0)


def base_coordinates(obj):
    """The mesh's vertices in object space."""
    return np.array([vertex.co for vertex in obj.data.vertices])


def evaluated_coordinates(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    result = np.array([vertex.co for vertex in mesh.vertices])
    evaluated.to_mesh_clear()
    return result


def skin(obj, rig, matrices):
    """Linear blend skinning of ``obj``'s vertices with {bone: 4x4 armature-space pose matrix}, in object
    space, the way the game (and Blender's Armature modifier) bends them."""
    base = base_coordinates(obj)
    to_armature = np.linalg.inv(np.array(rig.matrix_world)) @ np.array(obj.matrix_world)
    points = (to_armature @ np.c_[base, np.ones(len(base))].T).T
    result = np.zeros_like(points)
    total = np.zeros(len(base))
    names = [group.name for group in obj.vertex_groups]
    for vertex in obj.data.vertices:
        for element in vertex.groups:
            name = names[element.group]
            if name not in matrices or element.weight <= 0.0:
                continue
            deform = matrices[name] @ np.linalg.inv(np.array(rig.data.bones[name].matrix_local))
            result[vertex.index] += element.weight * (deform @ points[vertex.index])
            total[vertex.index] += element.weight
    moved = total > 0.0
    result[moved] /= total[moved, None]
    result[~moved] = points[~moved]
    return (np.linalg.inv(to_armature) @ result.T).T[:, :3]


# The game's bust scale at Bust Size 0 and 100 (chara/xls/charamake/human.cmp, player characters): every
# clan's but the Lalafells', and theirs.
BUST_RANGE = ((0.92, 0.8, 0.816), (1.08, 1.2, 1.184))
LALAFELL_BUST_RANGE = ((1.0, 1.0, 1.0), (1.08, 1.2, 1.184))


def bust_scale(size, lalafell=False):
    low, high = (np.array(values) for values in (LALAFELL_BUST_RANGE if lalafell else BUST_RANGE))
    return low + (high - low) * size / 100.0


def game_matrices(rig, template, options=None, bust=None, lalafell=False):
    """What the game would do to ``rig`` in its current pose: {bone: 4x4 armature-space matrix}, from
    `solver.simulate` on the rig's pose matrices, and with Bust Size ``bust`` (or None) the game's bust
    scale after it: j_mune_l, j_mune_r and every bone below them get their scale multiplied by it."""
    options = options or solver.Options()
    parents = {bone.name: bone.parent.name if bone.parent else None for bone in rig.data.bones}
    rest = {bone.name: np.array(bone.matrix_local) for bone in rig.data.bones}
    axes = solver.detect_axes(rest)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    posed = rig.evaluated_get(depsgraph).pose.bones
    model = {name: axes.to_game(name, np.array(posed[name].matrix)) for name in rest}
    # Like the add-on: bones without a rest pose keep their own change out.
    edits = {name: edit for name, edit in template.edited().items() if name in rest and name not in axes.unplaced}
    simulation = solver.simulate(parents, model, edits, options)
    game = dict(simulation.model)
    if bust is not None:
        scale = np.diag(list(bust_scale(bust, lalafell)) + [1.0])
        for bone in rig.data.bones:
            breast = bone
            while breast is not None and breast.name not in ("j_mune_l", "j_mune_r"):
                breast = breast.parent
            if breast is not None and bone.name not in axes.unplaced:
                game[bone.name] = game[bone.name] @ scale  # the bone's scale times the bust's, axis by axis
    root = axes.change_to_blender(simulation.root) if simulation.root is not None else np.identity(4)
    return {name: root @ axes.to_blender(name, game[name]) for name in rest}
