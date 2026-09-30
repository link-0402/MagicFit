"""A test scene for Line Up: a body with an FFXIV armature, and models in other poses, sizes and rigs.

Humanoids are built from tubes along their skeletons: the torso (hips to head), and per side a leg (hip,
knee, ankle, toe) and an arm (clavicle, shoulder, elbow, wrist, hand), each one tube whose weights blend
from one bone to the next over a few centimetres around each joint, as painted weights do. The tubes of
the limbs start a little inside the torso, where they blend from its bone into their first one. A marker
vertex sits at every joint, weighted fully to the bone starting there, so it lands exactly on the body's
joint when the model is lined up (with Stretch Limbs on).

- The body, "Body", stands in FFXIV's A-pose with the Mannequin's joints, on an armature with FFXIV's
  bone names (j_kosi, j_asi_a_l, j_ude_a_l...).
- "VRC" is a VRChat-style model (Hips, Leg_L, Knee_L, Arm_L, Elbow_L...) in a T-pose, 85 % of the body's
  size with longer legs, on its own armature, with a shape key moving its hands, a jiggle bone and a
  breast bone.
- "Biped" is a 3ds Max Biped model (Bip001-L-UpperArm...) without an armature (it was deleted: its meshes'
  Armature modifier has no object), facing +X with its arms hanging, 110 % of the body's size, in two
  meshes.
- "Anon" is a model whose vertex groups say nothing (bone_00, bone_01..., shuffled), like The Sims'
  exports, on an armature whose bones all sit at the origin, without feet, in an A-pose of its own.

Humanoids can have fingers (`build_humanoid(fingers=...)`): five tubes off the end of each hand, spread
across it with the thumb forward, three bones each (Thumb1_L... Little3_L), bent by a given curl.
"""

import math

import bpy
import numpy as np

# The Mannequin's joints (FFXIV female, rest pose).
BODY_JOINTS = {
    "hips": (0.0, 0.0, 1.037),
    "spine_b": (0.0, 0.012, 1.158),
    "spine_c": (0.0, 0.015, 1.284),
    "neck": (0.0, 0.01, 1.399),
    "head": (0.0, -0.04, 1.494),
    "head_top": (0.0, -0.04, 1.65),
    "hip_l": (0.086, -0.017, 0.922),
    "knee_l": (0.086, -0.028, 0.555),
    "ankle_l": (0.086, 0.013, 0.087),
    "toe_l": (0.086, -0.088, 0.014),
    "clavicle_l": (0.025, 0.02, 1.314),
    "shoulder_l": (0.128, 0.014, 1.345),
    "elbow_l": (0.309, 0.04, 1.163),
    "wrist_l": (0.47, 0.014, 1.003),
    "hand_end_l": (0.56, 0.007, 0.915),
}
# The part (Line Up's) whose joint each skeleton joint is.
JOINT_PART = {
    "neck": "neck", "head": "head",
    "hip_l": "thigh_l", "knee_l": "shin_l", "ankle_l": "foot_l",
    "clavicle_l": "clavicle_l", "shoulder_l": "upperarm_l", "elbow_l": "forearm_l", "wrist_l": "hand_l",
}
for _name, _part in list(JOINT_PART.items()):
    if _name.endswith("_l"):
        JOINT_PART[_name[:-2] + "_r"] = _part[:-2] + "_r"

# Bone names per rig, for each segment starting at a joint (the torso's bones by the joint they start at).
XIV_BONES = {
    "hips": "j_kosi", "spine_b": "j_sebo_b", "spine_c": "j_sebo_c", "neck": "j_kubi", "head": "j_kao",
    "hip": "j_asi_a_{s}", "knee": "j_asi_b_{s}", "ankle": "j_asi_d_{s}", "toe": "j_asi_e_{s}",
    "clavicle": "j_sako_{s}", "shoulder": "j_ude_a_{s}", "elbow": "j_ude_b_{s}", "wrist": "j_te_{s}",
}
VRC_BONES = {
    "hips": "Hips", "spine_b": "Spine", "spine_c": "Chest", "neck": "Neck", "head": "Head",
    "hip": "Leg_{S}", "knee": "Knee_{S}", "ankle": "Foot_{S}", "toe": "Toe_{S}",
    "clavicle": "Shoulder_{S}", "shoulder": "Arm_{S}", "elbow": "Elbow_{S}", "wrist": "Wrist_{S}",
}
BIPED_BONES = {
    "hips": "Bip001-Pelvis", "spine_b": "Bip001-Spine", "spine_c": "Bip001-Spine2", "neck": "Bip001-Neck",
    "head": "Bip001-Head",
    "hip": "Bip001-{S}-Thigh", "knee": "Bip001-{S}-Calf", "ankle": "Bip001-{S}-Foot", "toe": "Bip001-{S}-Toe0",
    "clavicle": "Bip001-{S}-Clavicle", "shoulder": "Bip001-{S}-UpperArm", "elbow": "Bip001-{S}-Forearm",
    "wrist": "Bip001-{S}-Hand",
}
TORSO_CHAIN = ("hips", "spine_b", "spine_c", "neck", "head")
LEG_CHAIN = ("hip", "knee", "ankle", "toe")
ARM_CHAIN = ("clavicle", "shoulder", "elbow", "wrist")
# Tube radii (at size 1) of the segment starting at each joint.
RADII = {"hips": 0.12, "spine_b": 0.11, "spine_c": 0.13, "neck": 0.045, "head": 0.08,
         "hip": 0.075, "knee": 0.05, "ankle": 0.035, "toe": 0.03,
         "clavicle": 0.05, "shoulder": 0.045, "elbow": 0.035, "wrist": 0.025}
BLEND = 0.02  # half the width over which weights hand over at a joint
RING_STEP = 0.01
SEGMENTS = 12


def rotate_z(points, degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    points = np.asarray(points, dtype=np.float64)
    return points @ np.array(((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))


def mirror(joints):
    """Add the right side's joints, mirrored from the left side's."""
    result = dict(joints)
    for name, point in joints.items():
        if name.endswith("_l"):
            result[name[:-2] + "_r"] = (-point[0], point[1], point[2])
    return result


def model_joints(scale=1.0, leg_scale=1.0, arm_angle=45.0, offset=(0.0, 0.0, 0.0), turn=0.0, feet=True):
    """A skeleton like the body's: ``scale`` times its size, legs ``leg_scale`` times longer, arms hanging
    ``arm_angle`` degrees below the horizontal (T-pose: 0), turned ``turn`` degrees about Z, moved."""
    body = {name: np.array(point) for name, point in BODY_JOINTS.items()}
    hip_height = body["hip_l"][2]
    joints = {}
    for name, point in body.items():
        p = point.copy()
        if name in ("knee_l", "ankle_l", "toe_l"):
            p[2] = hip_height - (hip_height - p[2]) * leg_scale
        joints[name] = p
    # Straight arms at the given angle, keeping each segment's length.
    shoulder = joints["shoulder_l"]
    direction = np.array((math.cos(math.radians(arm_angle)), 0.0, -math.sin(math.radians(arm_angle))))
    run = 0.0
    for a, b in (("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"), ("wrist_l", "hand_end_l")):
        run += np.linalg.norm(body[b] - body[a])
        joints[b] = shoulder + direction * run
    # Everything below the hips moves down so the feet stay on the ground.
    drop = (leg_scale - 1.0) * hip_height
    for name in joints:
        joints[name] = joints[name] + np.array((0.0, 0.0, drop))
    if not feet:
        joints.pop("toe_l")
    joints = mirror({name: tuple(point) for name, point in joints.items()})
    result = {}
    for name, point in joints.items():
        p = np.array(point) * scale
        p = rotate_z(p[None], turn)[0] + np.array(offset)
        result[name] = p
    return result


def _ring(center, direction, radius, segments=SEGMENTS):
    direction = direction / np.linalg.norm(direction)
    helper = np.array((0.0, 0.0, 1.0)) if abs(direction[2]) < 0.9 else np.array((1.0, 0.0, 0.0))
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    return center + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)


def tube(points, bones, radii, before_bone, lead=0.03):
    """A tube along the polyline ``points`` (joint positions), segment k weighted to ``bones[k]``, starting
    ``lead`` before the first joint (weighted to ``before_bone``, blending into the first bone). Returns
    (vertices, faces, weights: list of {bone: weight} per vertex)."""
    points = [np.asarray(p, dtype=np.float64) for p in points]
    first_direction = points[1] - points[0]
    first_direction /= np.linalg.norm(first_direction)
    polyline = [points[0] - first_direction * lead] + points
    cumulative = np.concatenate(([0.0], np.cumsum([np.linalg.norm(b - a) for a, b in zip(polyline, polyline[1:])])))
    joints_at = cumulative[1:-1]  # arc positions of the joints (first joint included, last point excluded)
    total = cumulative[-1]
    count = max(int(total / RING_STEP), 2)
    vertices, weights, faces = [], [], []
    chain = [before_bone] + list(bones)
    for i in range(count + 1):
        t = total * i / count
        k = int(np.searchsorted(cumulative, t, side="right") - 1)
        k = min(k, len(polyline) - 2)
        a, b = polyline[k], polyline[k + 1]
        f = (t - cumulative[k]) / max(cumulative[k + 1] - cumulative[k], 1e-12)
        center = a + (b - a) * f
        direction = b - a
        # Weights: the segment's bone, blending with the neighbour within BLEND of a joint.
        segment = min(int(np.searchsorted(joints_at, t, side="right")), len(chain) - 1)
        weight = {chain[segment]: 1.0}
        for j, position in enumerate(joints_at):
            if abs(t - position) < BLEND:
                share = 0.5 + 0.5 * (t - position) / BLEND  # 0 before the joint, 1 after
                weight = {chain[j]: 1.0 - share, chain[j + 1]: share}
                break
        radius = radii[max(segment - 1, 0)]
        start = len(vertices)
        vertices.extend(_ring(center, direction, radius))
        weights.extend([weight] * SEGMENTS)
        if i > 0:
            previous = start - SEGMENTS
            for s in range(SEGMENTS):
                faces.append((previous + s, previous + (s + 1) % SEGMENTS, start + (s + 1) % SEGMENTS, start + s))
    return vertices, faces, weights


FINGERS = ("thumb", "index", "middle", "ring", "little")
FINGER_RADIUS = 0.008


def finger_bone(finger, segment, side):
    return "{:s}{:d}_{:s}".format(finger.capitalize(), segment, side.upper())


def _turn(vector, axis, degrees):
    """``vector`` turned ``degrees`` about unit ``axis`` (Rodrigues)."""
    angle = math.radians(degrees)
    return (vector * math.cos(angle) + np.cross(axis, vector) * math.sin(angle) +
            axis * np.dot(axis, vector) * (1.0 - math.cos(angle)))


def finger_chains(joints, side, curl=0.0, forward=(0.0, -1.0, 0.0)):
    """Each finger of a hand (finger -> its 3 joints and tip): spread across the end of the hand (wrist to
    hand_end), the thumb ``forward`` (the way the model faces) and starting near the wrist, every segment
    bent ``curl`` degrees more than the one before, about the line across the knuckles: both hands curl
    the same way, mirrored."""
    wrist, end = np.asarray(joints["wrist_" + side], float), np.asarray(joints["hand_end_" + side], float)
    along = (end - wrist) / np.linalg.norm(end - wrist)
    across = np.asarray(forward, float) - along * np.dot(forward, along)
    across /= np.linalg.norm(across)
    bend = across if side == "l" else -across
    size = np.linalg.norm(end - wrist) / 0.126
    thumb = (along + 0.9 * across) / np.linalg.norm(along + 0.9 * across)
    chains = {}
    for finger, base, direction, lengths in (
            ("thumb", wrist + size * (0.03 * along + 0.022 * across), thumb, (0.03, 0.025, 0.02)),
            ("index", end + size * 0.021 * across, along, (0.035, 0.022, 0.018)),
            ("middle", end + size * 0.007 * across, along, (0.038, 0.024, 0.019)),
            ("ring", end - size * 0.007 * across, along, (0.035, 0.022, 0.018)),
            ("little", end - size * 0.021 * across, along, (0.028, 0.018, 0.016))):
        points = [base]
        for k, length in enumerate(lengths):
            points.append(points[-1] + _turn(direction, bend, curl * (k + 1)) * length * size)
        chains[finger] = points
    return chains


def build_humanoid(joints, bone_names, sides=("l", "r"), torso_only_head=False, fingers=None):
    """Vertices, faces and weights of a humanoid built from tubes, and its markers (vertex index -> joint
    name). ``bone_names`` maps the chain joint names (hips, hip, knee...) to bone names, "{s}"/"{S}"
    standing for the side. With ``fingers`` (a curl in degrees, see `finger_chains`), each hand gets
    finger tubes weighted to Thumb1_L... Little3_L, blending in from the hand's bone."""
    def bone(key, side=None):
        name = bone_names[key]
        if side is not None:
            name = name.replace("{s}", side).replace("{S}", side.upper())
        return name

    vertices, faces, weights, markers = [], [], [], {}

    def add(part):
        v, f, w = part
        first = len(vertices)
        vertices.extend(v)
        faces.extend(tuple(i + first for i in face) for face in f)
        weights.extend(w)

    torso_points = [joints["hips"] - np.array((0.0, 0.0, 0.12))] + [joints[k] for k in TORSO_CHAIN] + [joints["head_top"]]
    add(tube(torso_points[1:], [bone(k) for k in TORSO_CHAIN], [RADII[k] for k in TORSO_CHAIN],
             bone("hips"), lead=0.12))
    for side in sides:
        feet = "toe_" + side in joints
        if feet:
            # Down to the toes' tips, the ankle to the toes and the toes weighted to their bones.
            keys = LEG_CHAIN
            names = [n + "_" + side for n in keys]
            end = joints[names[-1]] + (joints[names[-1]] - joints[names[-2]]) * 0.4
        else:
            # No feet (a body cut at the ankles): the shin ends there.
            keys = LEG_CHAIN[:2]
            names = [n + "_" + side for n in ("hip", "knee")]
            end = joints["ankle_" + side]
        add(tube([joints[n] for n in names] + [end], [bone(k, side) for k in keys], [RADII[k] for k in keys],
                 bone("hips")))
        arm = [joints[n + "_" + side] for n in ("clavicle", "shoulder", "elbow", "wrist", "hand_end")]
        add(tube(arm, [bone(k, side) for k in ARM_CHAIN], [RADII[k] for k in ARM_CHAIN], bone("spine_c")))
        if fingers is not None:
            for finger, points in finger_chains(joints, side, fingers).items():
                add(tube(points, [finger_bone(finger, k, side) for k in (1, 2, 3)], [FINGER_RADIUS] * 3,
                         bone("wrist", side), lead=0.012))
        # Markers: one vertex per joint, weighted to the bone starting there.
        for key, joint in (("hip", "hip"), ("knee", "knee"), ("ankle", "ankle"), ("clavicle", "clavicle"),
                           ("shoulder", "shoulder"), ("elbow", "elbow"), ("wrist", "wrist")):
            name = joint + "_" + side
            if name in joints and (key != "ankle" or feet):
                markers[len(vertices)] = name
                vertices.append(np.array(joints[name]))
                weights.append({bone(key, side): 1.0})
        if fingers is not None:
            # Finger markers: each joint, and the tip (on the last bone), named like "index2_l", "indextip_l".
            for finger, points in finger_chains(joints, side, fingers).items():
                for k, point in enumerate(points):
                    markers[len(vertices)] = "{:s}{:s}_{:s}".format(finger, str(k + 1) if k < 3 else "tip", side)
                    vertices.append(np.array(point))
                    weights.append({finger_bone(finger, min(k + 1, 3), side): 1.0})
    for key in ("neck", "head"):
        markers[len(vertices)] = key
        vertices.append(np.array(joints[key]))
        weights.append({bone(key): 1.0})
    return np.array(vertices), faces, weights, markers


def mesh_object(name, vertices, faces, weights, collection=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in vertices], [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    (collection or bpy.context.scene.collection).objects.link(obj)
    groups = {}
    for index, weight in enumerate(weights):
        for group, value in weight.items():
            if value <= 0.0:
                continue
            if group not in groups:
                groups[group] = obj.vertex_groups.new(name=group)
            groups[group].add([index], value, 'REPLACE')
    return obj


def armature_object(name, joints, bone_names, bone_list, collection=None, at_origin=False):
    """An armature with ``bone_list`` bones: (key, side, parent key/side or None, head joint, tail joint)."""
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    (collection or bpy.context.scene.collection).objects.link(obj)
    view_layer = bpy.context.view_layer
    for other in view_layer.objects:
        if other is not None:
            other.select_set(False)
    view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    edit = {}
    for bone_name, parent, head, tail in bone_list:
        eb = data.edit_bones.new(bone_name)
        if at_origin:
            eb.head, eb.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        else:
            eb.head, eb.tail = tuple(head), tuple(tail)
            if np.linalg.norm(np.array(tail) - np.array(head)) < 1e-4:
                eb.tail = tuple(np.array(head) + np.array((0.0, 0.0, 0.03)))
        if parent is not None and parent in edit and not at_origin:
            eb.parent = edit[parent]
        edit[bone_name] = eb
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)
    return obj


def bone_list(joints, names, sides=("l", "r"), extra=()):
    """Bones of a humanoid: (name, parent name, head, tail) in hierarchy order."""
    def bone(key, side=None):
        name = names[key]
        return name if side is None else name.replace("{s}", side).replace("{S}", side.upper())

    result = []
    torso = list(TORSO_CHAIN)
    for k, key in enumerate(torso):
        tail = joints[torso[k + 1]] if k + 1 < len(torso) else joints["head_top"]
        result.append((bone(key), bone(torso[k - 1]) if k else None, joints[key], tail))
    for side in sides:
        keys = LEG_CHAIN if "toe_" + side in joints else LEG_CHAIN[:2]
        chain = [(key, key + "_" + side) for key in keys]
        parent = bone("hips")
        for k, (key, joint) in enumerate(chain):
            if k + 1 < len(chain):
                tail = joints[chain[k + 1][1]]
            elif "toe_" + side in joints:
                tail = joints[joint] + np.array((0.0, -0.04, 0.0))
            else:
                tail = joints["ankle_" + side]
            result.append((bone(key, side), parent, joints[joint], tail))
            parent = bone(key, side)
        chain = [(key, key + "_" + side) for key in ARM_CHAIN] + [(None, "hand_end_" + side)]
        parent = bone("spine_c")
        for k, (key, joint) in enumerate(chain[:-1]):
            result.append((bone(key, side), parent, joints[joint], joints[chain[k + 1][1]]))
            parent = bone(key, side)
    result.extend(extra)
    return result


def clear():
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for data in list(bpy.data.meshes) + list(bpy.data.armatures):
        if data.users == 0:
            (bpy.data.meshes if isinstance(data, bpy.types.Mesh) else bpy.data.armatures).remove(data)
    bpy.context.view_layer.update()


def build_body(fingers=False):
    """The body: FFXIV joints and bone names, A-pose, on its armature "Skeleton"; with ``fingers``,
    straight ones (named Thumb1_L..., under j_te_l)."""
    joints = {name: np.array(point) for name, point in mirror(BODY_JOINTS).items()}
    vertices, faces, weights, markers = build_humanoid(joints, XIV_BONES, fingers=0.0 if fingers else None)
    body = mesh_object("Body", vertices, faces, weights)
    extra = [("n_root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.2))]
    if fingers:
        extra += finger_bones(joints, wrist="j_te_{s}")
    skeleton = armature_object("Skeleton", joints, XIV_BONES, bone_list(joints, XIV_BONES, extra=extra))
    modifier = body.modifiers.new("Armature", 'ARMATURE')
    modifier.object = skeleton
    body.parent = skeleton
    return body, skeleton, joints


VRC_JOINTS = dict(scale=0.85, leg_scale=1.12, arm_angle=0.0)
BIPED_JOINTS = dict(scale=1.1, leg_scale=0.95, arm_angle=70.0, turn=90.0)
ANON_JOINTS = dict(scale=1.2, leg_scale=1.0, arm_angle=30.0, feet=False)


def build_vrc(fingers=None):
    """The VRChat-style model on its armature "VRC Armature", with a shape key "Wave" (the hands 2 cm up);
    with ``fingers`` (a curl in degrees), fingers too."""
    collection = bpy.data.collections.new("VRC")
    bpy.context.scene.collection.children.link(collection)
    joints = model_joints(**VRC_JOINTS)
    vertices, faces, weights, markers = build_humanoid(joints, VRC_BONES, fingers=fingers)
    # A jiggle bone on the left thigh and a breast bone on the chest take some weight.
    for index, weight in enumerate(weights):
        if weight.get("Leg_L", 0.0) == 1.0 and vertices[index][2] < joints["hip_l"][2] - 0.1:
            weights[index] = {"Leg_L": 0.7, "Thigh Jiggle_L": 0.3}
        if weight.get("Chest", 0.0) == 1.0 and vertices[index][1] < -0.05:
            weights[index] = {"Chest": 0.5, "Boob_L": 0.5}
    model = mesh_object("VRC Suit", vertices, faces, weights, collection)
    extra = [("Thigh Jiggle_L", "Leg_L", joints["hip_l"] + np.array((0.0, -0.05, -0.1)),
              joints["hip_l"] + np.array((0.0, -0.05, -0.2))),
             ("Boob_L", "Chest", joints["spine_c"] + np.array((0.05, -0.08, 0.0)),
              joints["spine_c"] + np.array((0.05, -0.14, 0.0)))]
    if fingers is not None:
        extra += finger_bones(joints, curl=fingers)
    armature = armature_object("VRC Armature", joints, VRC_BONES, bone_list(joints, VRC_BONES, extra=extra),
                               collection)
    modifier = model.modifiers.new("Armature", 'ARMATURE')
    modifier.object = armature
    model.parent = armature
    model.shape_key_add(name="Basis", from_mix=False)
    key = model.shape_key_add(name="Wave", from_mix=False)
    key.value = 0.0
    co = np.empty(len(key.data) * 3)
    key.data.foreach_get("co", co)
    co = co.reshape(-1, 3)
    hands = np.array([("Wrist_L" in w or "Wrist_R" in w) for w in weights])
    co[hands, 2] += 0.02
    key.data.foreach_set("co", co.ravel())
    return model, armature, joints, markers


def build_biped():
    """The Biped model, without its armature, in two meshes ("Biped Body", "Biped Arms")."""
    collection = bpy.data.collections.new("Biped")
    bpy.context.scene.collection.children.link(collection)
    joints = model_joints(**BIPED_JOINTS)
    vertices, faces, weights, markers = build_humanoid(joints, BIPED_BONES)
    # Split: the arm tubes (and their markers) go to a second mesh.
    arm_vertex = np.array([any("Clavicle" in g or "Arm" in g or "Forearm" in g or "Hand" in g for g in w)
                           for w in weights])
    objects = []
    for name, keep in (("Biped Body", ~arm_vertex), ("Biped Arms", arm_vertex)):
        index = np.flatnonzero(keep)
        remap = {int(old): new for new, old in enumerate(index)}
        part_faces = [tuple(remap[i] for i in face) for face in faces if all(i in remap for i in face)]
        obj = mesh_object(name, vertices[index], part_faces, [weights[i] for i in index], collection)
        obj.modifiers.new("Biped_Skeleton", 'ARMATURE')  # its armature was deleted
        objects.append(obj)
    part_markers = []
    for obj, keep in zip(objects, (~arm_vertex, arm_vertex)):
        index = np.flatnonzero(keep)
        remap = {int(old): new for new, old in enumerate(index)}
        part_markers.append({remap[v]: joint for v, joint in markers.items() if v in remap})
    return objects, joints, part_markers


def skirt(joints, bias=0.0, bones=VRC_BONES):
    """A long skirt, flaring from the hips to below the knees, weighted like the legs under it: the hips'
    bone near the waist, below it both thighs' bones, each more on its own side, and ``bias`` of the right
    thigh's weight given to the left one instead (skirts are often weighted unevenly). Returns (vertices,
    faces, weights)."""
    hips = np.asarray(joints["hips"], dtype=np.float64)
    knee = (np.asarray(joints["knee_l"]) + np.asarray(joints["knee_r"])) / 2.0
    top, bottom = hips[2] - 0.02, knee[2] - 0.15
    left, right = bones["hip"].replace("{S}", "L"), bones["hip"].replace("{S}", "R")
    vertices, faces, weights = [], [], []
    rings = 30
    for i in range(rings + 1):
        t = i / rings
        z = top + (bottom - top) * t
        radius = 0.19 + 0.12 * t
        start = len(vertices)
        for k in range(24):
            angle = 2.0 * math.pi * k / 24
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            vertices.append(np.array((hips[0] + x, hips[1] + y, z)))
            waist = max(0.0, 1.0 - t / 0.25)  # the hips' bone near the waist only
            side = 0.5 + 0.5 * x / radius  # 1 on the left, 0 on the right
            w_left, w_right = (1.0 - waist) * side, (1.0 - waist) * (1.0 - side)
            w_left, w_right = w_left + bias * w_right, (1.0 - bias) * w_right
            weights.append({bones["hips"]: waist, left: w_left, right: w_right})
        if i > 0:
            previous = start - 24
            for k in range(24):
                faces.append((previous + k, previous + (k + 1) % 24, start + (k + 1) % 24, start + k))
    return vertices, faces, weights


def half_body(vertices, faces, weights, markers, keep_upper):
    """Only the upper body (no legs, torso from the waist up) or the lower one (no arms or fingers, torso up
    to the waist) of a humanoid from `build_humanoid`: its vertices, faces, weights and markers."""
    joints_upper = ("Shoulder", "Arm", "Elbow", "Wrist", "Thumb", "Index", "Middle", "Ring", "Little", "Chest",
                    "Neck", "Head")
    keep = []
    for weight in weights:
        upper = any(name.startswith(joints_upper) for name in weight)
        lower = any(name.startswith(("Leg", "Knee", "Foot", "Toe")) for name in weight)
        keep.append(upper if keep_upper else lower or not upper and not lower)
    keep = np.array(keep)
    index = np.flatnonzero(keep)
    remap = {int(old): new for new, old in enumerate(index)}
    part_faces = [tuple(remap[i] for i in face) for face in faces if all(i in remap for i in face)]
    part_markers = {remap[v]: joint for v, joint in markers.items() if v in remap}
    return vertices[index], part_faces, [weights[i] for i in index], part_markers


def build_anon(seed=3, fingers=None, skirt_bias=None, half=None):
    """The anonymous model: groups bone_00.. in random order, on an armature with every bone at the origin;
    with ``fingers`` (a curl in degrees), fingers too; with ``skirt_bias``, a long skirt (see `skirt`);
    with ``half`` ("upper" or "lower"), only that half of the body."""
    collection = bpy.data.collections.new("Anon")
    bpy.context.scene.collection.children.link(collection)
    joints = model_joints(**ANON_JOINTS)
    vertices, faces, weights, markers = build_humanoid(joints, VRC_BONES, fingers=fingers)
    if half is not None:
        vertices, faces, weights, markers = half_body(vertices, faces, weights, markers, half == "upper")
    if skirt_bias is not None:
        v, f, w = skirt(joints, skirt_bias)
        first = len(vertices)
        vertices = np.concatenate([vertices, np.array(v)])
        faces = faces + [tuple(i + first for i in face) for face in f]
        weights = weights + w
    real = sorted({g for w in weights for g in w})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(real))
    rename = {name: "bone_{:02d}".format(int(k)) for name, k in zip(real, order)}
    anon_weights = [{rename[g]: v for g, v in w.items()} for w in weights]
    model = mesh_object("Anon Body", vertices, faces, anon_weights, collection)
    names = sorted(rename.values())
    armature = armature_object("Anon Armature", joints, {}, [(n, None, (0, 0, 0), (0, 0, 1)) for n in names],
                               collection, at_origin=True)
    modifier = model.modifiers.new("Armature", 'ARMATURE')
    modifier.object = armature
    model.parent = armature
    # The part each anonymous group really is, from its original bone name (and a finger's bones' order).
    from magic_fit import lineup
    parents = {name: None for name in real}
    for side in ("l", "r"):
        for finger in FINGERS:
            for k in (1, 2, 3):
                if finger_bone(finger, k, side) in parents:
                    parents[finger_bone(finger, k, side)] = (finger_bone(finger, k - 1, side) if k > 1 else
                                                            VRC_BONES["wrist"].replace("{S}", side.upper()))
    real_parts = lineup.bone_parts(real, parents)
    truth = {rename[name]: real_parts[name] for name in real}
    return model, armature, joints, markers, truth


def build():
    """Everything: returns a dict of the objects."""
    clear()
    body, skeleton, body_joints = build_body()
    vrc, vrc_armature, vrc_joints, vrc_markers = build_vrc()
    bipeds, biped_joints, biped_markers = build_biped()
    anon, anon_armature, anon_joints, anon_markers, anon_truth = build_anon()
    bpy.context.scene.magic_fit.target = body
    return {
        "body": body, "skeleton": skeleton, "body_joints": body_joints,
        "vrc": vrc, "vrc_armature": vrc_armature, "vrc_joints": vrc_joints, "vrc_markers": vrc_markers,
        "bipeds": bipeds, "biped_joints": biped_joints, "biped_markers": biped_markers,
        "anon": anon, "anon_armature": anon_armature, "anon_joints": anon_joints, "anon_markers": anon_markers,
        "anon_truth": anon_truth,
    }


def select(objects, active=None):
    view_layer = bpy.context.view_layer
    for obj in view_layer.objects:
        if obj is not None:
            obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    view_layer.objects.active = active or (objects[0] if objects else None)


def world_positions(obj, key=None):
    """World positions of mesh ``obj``: its vertices, or shape key ``key``'s."""
    data = obj.data.shape_keys.key_blocks[key].data if key is not None else obj.data.vertices
    co = np.empty(len(data) * 3)
    data.foreach_get("co", co)
    matrix = np.array(obj.matrix_world)
    return co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


def displayed_positions(obj):
    """World positions of mesh ``obj`` as displayed (shape keys, armature...)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    evaluated.to_mesh_clear()
    matrix = np.array(obj.matrix_world)
    return co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


def marker_errors(obj, markers, body_joints):
    """Distance of each marker (vertex -> joint name) of ``obj`` from the body's joint of the same name
    (markers the body has no joint for are left out)."""
    co = world_positions(obj)
    return {joint: float(np.linalg.norm(co[v] - body_joints[joint])) for v, joint in markers.items()
            if joint in body_joints}


def finger_bones(joints, sides=("l", "r"), wrist="Wrist_{S}", curl=0.0):
    """Finger bones for `armature_object` (name, parent, head, tail) along the fingers of `finger_chains`,
    under the hand bone ``wrist``."""
    result = []
    for side in sides:
        for finger, points in finger_chains(joints, side, curl).items():
            parent = wrist.replace("{S}", side.upper()).replace("{s}", side)
            for k in (1, 2, 3):
                name = finger_bone(finger, k, side)
                result.append((name, parent, points[k - 1], points[k]))
                parent = name
    return result
