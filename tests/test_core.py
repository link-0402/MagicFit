"""Headless tests for the weight sampling/blending engine.

Run from the repository root:
    blender -b --factory-startup --python tests/test_core.py
"""

import os
import sys
import traceback

import bpy
import numpy as np
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import magic_fit  # noqa: E402
from magic_fit import painting  # noqa: E402

TOL = 1e-4
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
        print("FAIL:", message)


def close(a, b, tol=TOL):
    return abs(a - b) <= tol


def grid(name, size, location):
    mesh = bpy.data.meshes.new(name)
    verts = []
    for j in range(size + 1):
        for i in range(size + 1):
            verts.append((-1.0 + 2.0 * i / size, -1.0 + 2.0 * j / size, 0.0))
    faces = []
    for j in range(size):
        for i in range(size):
            a = j * (size + 1) + i
            faces.append((a, a + 1, a + size + 2, a + size + 1))
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    return obj


def weights_of(obj, vert):
    names = {g.index: g.name for g in obj.vertex_groups}
    return {names[e.group]: e.weight for e in obj.data.vertices[vert].groups}


def build_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)

    # Target: A is a linear ramp along X, B = 1 - A, C only exists where x > 0.
    target = grid("Target", 10, (0.0, 0.0, 0.0))
    a = target.vertex_groups.new(name="A")
    b = target.vertex_groups.new(name="B")
    c = target.vertex_groups.new(name="C")
    for v in target.data.vertices:
        ramp = (v.co.x + 1.0) / 2.0
        a.add([v.index], ramp, 'REPLACE')
        b.add([v.index], 1.0 - ramp, 'REPLACE')
        if v.co.x > 1e-6:
            c.add([v.index], 0.5, 'REPLACE')

    # Painted: finer grid hovering just above, with "wrong" weights.
    painted = grid("Painted", 15, (0.0, 0.0, 0.01))
    groups = {name: painted.vertex_groups.new(name=name) for name in ("A", "B", "X", "L")}
    for v in painted.data.vertices:
        groups["B"].add([v.index], 0.7, 'REPLACE')
        groups["X"].add([v.index], 0.3, 'REPLACE')
        groups["L"].add([v.index], 0.2, 'REPLACE')
    groups["L"].lock_weight = True
    painted.vertex_groups.active_index = groups["A"].index

    bpy.context.view_layer.objects.active = painted
    painted.select_set(True)
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    return target, painted


def make_stroke(target, painted, **kwargs):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    geometry = painting.read_geometry(painted, depsgraph)
    sampler = painting.TargetSampler(
        target, depsgraph,
        mode=kwargs.pop("sample_mode", 'SURFACE'),
        max_distance=kwargs.pop("max_distance", None),
    )
    return painting.CopyStroke(painted, target, sampler, geometry.co, **kwargs)


def expected_ramp(painted, vert):
    x = (painted.matrix_world @ painted.data.vertices[vert].co).x
    return (x + 1.0) / 2.0


def interior_verts(painted, predicate):
    out = []
    for v in painted.data.vertices:
        co = painted.matrix_world @ v.co
        if abs(co.y) < 0.9 and predicate(co.x):
            out.append(v.index)
    return out


def test_all_groups_full_copy():
    target, painted = build_scene()
    verts = interior_verts(painted, lambda x: abs(x) < 0.9)
    stroke = make_stroke(target, painted, group_mode='ALL')
    stroke.apply(verts, [1.0] * len(verts))

    for vert in verts:
        w = weights_of(painted, vert)
        ramp = expected_ramp(painted, vert)
        check(close(w.get("A", 0.0), ramp), f"ALL: A of {vert} = {w.get('A')} expected {ramp}")
        check(close(w.get("B", 0.0), 1.0 - ramp), f"ALL: B of {vert} = {w.get('B')} expected {1 - ramp}")
        check("X" not in w, f"ALL: unmatched X should be removed from {vert}: {w}")
        check(close(w["L"], 0.2), f"ALL: locked L changed on {vert}: {w['L']}")
    check("C" in painted.vertex_groups, "ALL: missing group C was not created")
    check(painted.vertex_groups.active.name == "A", "ALL: creating C changed the active group")
    x = {v: (painted.matrix_world @ painted.data.vertices[v].co).x for v in verts}
    for vert in verts:
        c = weights_of(painted, vert).get("C", 0.0)
        if x[vert] > 0.25:
            check(close(c, 0.5), f"ALL: C of {vert} = {c} expected 0.5")
        elif x[vert] < -0.25:
            check(c == 0.0, f"ALL: C of {vert} = {c} expected none")
    check(stroke.created_group_names() == ["C"], f"created {stroke.created_group_names()}")


def test_strength_and_coverage():
    target, painted = build_scene()
    vert = interior_verts(painted, lambda x: 0.3 < x < 0.6)[0]
    ramp = expected_ramp(painted, vert)
    stroke = make_stroke(target, painted, group_mode='ALL')

    stroke.apply([vert], [0.5])
    w = weights_of(painted, vert)
    check(close(w["A"], 0.5 * ramp), f"half: A={w['A']} expected {0.5 * ramp}")
    check(close(w["B"], 0.7 + (1 - ramp - 0.7) * 0.5), f"half: B={w['B']}")
    check(close(w["X"], 0.15), f"half: X={w['X']} expected 0.15")

    # The same coverage again must not push further (no accumulate).
    stroke.apply([vert], [0.5])
    check(close(weights_of(painted, vert)["A"], 0.5 * ramp), "coverage: second dab changed weights")
    # A stronger dab moves further, measured from the stroke's starting weights.
    stroke.apply([vert], [0.8])
    check(close(weights_of(painted, vert)["A"], 0.8 * ramp), "coverage: stronger dab wrong")

    target, painted = build_scene()
    stroke = make_stroke(target, painted, group_mode='ALL', accumulate=True)
    stroke.apply([vert], [0.5])
    stroke.apply([vert], [0.5])
    check(close(weights_of(painted, vert)["A"], 0.75 * ramp), "accumulate: expected 3/4 of the way")


def test_fade_unmatched_off():
    target, painted = build_scene()
    vert = interior_verts(painted, lambda x: -0.6 < x < -0.3)[0]
    stroke = make_stroke(target, painted, group_mode='ALL', fade_unmatched=False)
    stroke.apply([vert], [1.0])
    check(close(weights_of(painted, vert).get("X", 0.0), 0.3), "fade off: X should stay 0.3")


def test_active_group():
    target, painted = build_scene()
    painted.vertex_groups.active_index = painted.vertex_groups["B"].index
    vert = interior_verts(painted, lambda x: 0.3 < x < 0.6)[0]
    ramp = expected_ramp(painted, vert)
    before = weights_of(painted, vert)
    stroke = make_stroke(target, painted, group_mode='ACTIVE')
    stroke.apply([vert], [1.0])
    after = weights_of(painted, vert)
    check(close(after["B"], 1.0 - ramp), f"ACTIVE: B={after['B']} expected {1 - ramp}")
    for name in ("X", "L"):
        check(close(after[name], before[name]), f"ACTIVE: {name} changed")
    check("A" not in after and "C" not in painted.vertex_groups, "ACTIVE: other groups touched")


def test_active_restrict():
    target, painted = build_scene()
    painted.vertex_groups.active_index = painted.vertex_groups["A"].index
    vert = interior_verts(painted, lambda x: 0.3 < x < 0.6)[0]
    stroke = make_stroke(target, painted, group_mode='ACTIVE', restrict=True)
    stroke.apply([vert], [1.0])
    check("A" not in weights_of(painted, vert), "restrict: vertex outside the group was painted")


def test_auto_normalize():
    target, painted = build_scene()
    arm_data = bpy.data.armatures.new("Rig")
    arm = bpy.data.objects.new("Rig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    for name in ("A", "B", "X", "C"):
        bone = arm_data.edit_bones.new(name)
        bone.tail = (0.0, 0.0, 1.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    mod = painted.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    bpy.context.view_layer.objects.active = painted
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

    # Current Group: other deform groups make room, the locked (non-bone) L stays as is.
    painted.vertex_groups.active_index = painted.vertex_groups["A"].index
    vert = interior_verts(painted, lambda x: 0.3 < x < 0.6)[0]
    ramp = expected_ramp(painted, vert)
    stroke = make_stroke(target, painted, group_mode='ACTIVE', auto_normalize=True)
    stroke.apply([vert], [1.0])
    w = weights_of(painted, vert)
    deform_sum = w.get("A", 0) + w.get("B", 0) + w.get("X", 0)
    check(close(w["A"], ramp), f"normalize ACTIVE: A={w['A']} expected {ramp}")
    check(close(deform_sum, 1.0), f"normalize ACTIVE: deform sum {deform_sum}")
    check(close(w["B"] / w["X"], 0.7 / 0.3), "normalize ACTIVE: others lost their proportions")
    check(close(w["L"], 0.2), "normalize ACTIVE: L (non-deform) changed")

    # All Groups with half strength also ends up normalized.
    vert2 = interior_verts(painted, lambda x: -0.6 < x < -0.3)[0]
    stroke = make_stroke(target, painted, group_mode='ALL', auto_normalize=True)
    stroke.apply([vert2], [0.5])
    w = weights_of(painted, vert2)
    deform_sum = sum(w.get(n, 0) for n in ("A", "B", "X", "C"))
    check(close(deform_sum, 1.0), f"normalize ALL: deform sum {deform_sum} ({w})")


def test_restore():
    target, painted = build_scene()
    verts = interior_verts(painted, lambda x: abs(x) < 0.9)
    before = {v: weights_of(painted, v) for v in verts}
    groups_before = [g.name for g in painted.vertex_groups]
    stroke = make_stroke(target, painted, group_mode='ALL')
    stroke.apply(verts, [0.7] * len(verts))
    check("C" in painted.vertex_groups, "restore: setup should have created C")
    stroke.restore()
    check([g.name for g in painted.vertex_groups] == groups_before, "restore: created groups not removed")
    for v in verts:
        after = weights_of(painted, v)
        check(after.keys() == before[v].keys() and all(close(after[k], before[v][k], 1e-6) for k in after),
              f"restore: vertex {v} {after} != {before[v]}")


def test_vertex_mode_and_max_distance():
    target, painted = build_scene()
    vert = interior_verts(painted, lambda x: 0.3 < x < 0.6)[0]
    stroke = make_stroke(target, painted, group_mode='ALL', sample_mode='VERTEX')
    stroke.apply([vert], [1.0])
    co = painted.matrix_world @ painted.data.vertices[vert].co
    nearest_x = round((co.x + 1.0) / 0.2) * 0.2 - 1.0
    check(close(weights_of(painted, vert)["A"], (nearest_x + 1.0) / 2.0, 1e-3), "VERTEX: not the nearest vertex")

    # Painted mesh is 0.01 above the target: a smaller max distance must leave it alone.
    target, painted = build_scene()
    before = weights_of(painted, vert)
    stroke = make_stroke(target, painted, group_mode='ALL', max_distance=0.005)
    stroke.apply([vert], [1.0])
    check(weights_of(painted, vert) == before, "max distance: out of range vertex was changed")
    stroke = make_stroke(target, painted, group_mode='ALL', max_distance=0.02)
    stroke.apply([vert], [1.0])
    check(weights_of(painted, vert) != before, "max distance: in range vertex was not changed")


def test_transformed_objects():
    # Rotated/scaled target: sampling happens in world space.
    target, painted = build_scene()
    target.matrix_world = Matrix.Rotation(0.5, 4, 'Z') @ Matrix.Scale(2.0, 4)
    painted.matrix_world = Matrix.Rotation(0.5, 4, 'Z') @ Matrix.Translation((0, 0, 0.02)) @ Matrix.Scale(2.0, 4)
    bpy.context.view_layer.update()
    vert = interior_verts(painted, lambda x: True)[0]
    local = painted.data.vertices[vert].co
    stroke = make_stroke(target, painted, group_mode='ALL')
    stroke.apply([vert], [1.0])
    check(close(weights_of(painted, vert)["A"], (local.x + 1.0) / 2.0), "transform: world space sampling wrong")


def test_mirrored_target():
    # Only the +X half of the target exists; a Mirror modifier shows (and weights) the other half.
    target, painted = build_scene()
    bpy.ops.object.mode_set(mode='OBJECT')
    half = grid("Half", 10, (0.0, 0.0, 0.0))
    # x from [-1, 1] to [0, 1]
    half.data.transform(Matrix.Translation((0.5, 0.0, 0.0)) @ Matrix.Scale(0.5, 4, (1.0, 0.0, 0.0)))
    arm_l = half.vertex_groups.new(name="arm.L")
    half.vertex_groups.new(name="arm.R")
    spine = half.vertex_groups.new(name="spine")
    for v in half.data.vertices:
        arm_l.add([v.index], v.co.x, 'REPLACE')
        spine.add([v.index], 1.0 - v.co.x, 'REPLACE')
    half.modifiers.new("Mirror", 'MIRROR').use_mirror_vertex_groups = True
    bpy.context.view_layer.objects.active = painted
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

    left = interior_verts(painted, lambda x: -0.7 < x < -0.4)[0]
    right = interior_verts(painted, lambda x: 0.4 < x < 0.7)[0]
    stroke = make_stroke(half, painted, group_mode='ALL')
    stroke.apply([left, right], [1.0, 1.0])
    for vert, side, other in ((left, "arm.R", "arm.L"), (right, "arm.L", "arm.R")):
        w = weights_of(painted, vert)
        x = abs((painted.matrix_world @ painted.data.vertices[vert].co).x)
        check(close(w.get(side, 0.0), x), f"mirror: {side} of {vert} = {w.get(side)} expected {x}")
        check(close(w.get("spine", 0.0), 1.0 - x), f"mirror: spine of {vert} = {w.get('spine')}")
        check(w.get(other, 0.0) == 0.0, f"mirror: {other} should be empty on {vert}: {w}")


def test_deform_bones_only():
    # Like a mannequin body with helper groups (masks for normals, shape keys, physics) next to its
    # bone weights: only the bone groups are copied and faded, helper groups stay as they are.
    target, painted = build_scene()
    bpy.ops.object.mode_set(mode='OBJECT')
    rig = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for name in ("A", "B", "C"):
        bone = rig.data.edit_bones.new(name)
        bone.head, bone.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    everything = list(range(len(target.data.vertices)))
    target.vertex_groups.new(name="Rue - Normals").add(everything, 1.0, 'REPLACE')
    painted.vertex_groups.new(name="Waist").add(list(range(len(painted.data.vertices))), 0.4, 'REPLACE')

    def paint(bones_only):
        bpy.context.view_layer.objects.active = painted
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        verts = interior_verts(painted, lambda x: abs(x) < 0.9)
        stroke = make_stroke(target, painted, group_mode='ALL', bones_only=bones_only)
        stroke.apply(verts, [1.0] * len(verts))
        bpy.ops.object.mode_set(mode='OBJECT')
        return stroke, verts

    for where in ("target", "painted mesh"):
        holder = target if where == "target" else painted
        modifier = holder.modifiers.new("Armature", 'ARMATURE')
        modifier.object = rig
        stroke, verts = paint(bones_only=True)
        check("Rue - Normals" not in painted.vertex_groups,
              f"bones only ({where}'s armature): a helper group of the target was copied")
        for vert in verts:
            w = weights_of(painted, vert)
            ramp = expected_ramp(painted, vert)
            check(close(w.get("A", 0.0), ramp), f"bones only: A of {vert} = {w.get('A')} expected {ramp}")
            check(close(w.get("Waist", 0.0), 0.4), f"bones only: helper group Waist changed on {vert}: {w}")
            check(close(w.get("X", 0.0), 0.3), f"bones only: X (not a bone) changed on {vert}: {w}")
        stroke.restore()
        holder.modifiers.remove(modifier)

    # Switched off: every group, as before.
    target.modifiers.new("Armature", 'ARMATURE').object = rig
    stroke, verts = paint(bones_only=False)
    check("Rue - Normals" in painted.vertex_groups, "all groups: the helper group was copied")
    check(all("Waist" not in weights_of(painted, vert) for vert in verts), "all groups: Waist faded out")


def test_shape_keyed_meshes():
    # Both meshes are used as displayed, shape keys included: the target's key slides it 0.4 along X,
    # the painted mesh's key slides it 0.2.
    target, painted = build_scene()
    bpy.ops.object.mode_set(mode='OBJECT')
    for obj, shift in ((target, 0.4), (painted, 0.2)):
        obj.shape_key_add(name="Basis")
        key = obj.shape_key_add(name="Slide")
        for point in key.data:
            point.co.x += shift
        key.value = 1.0
    bpy.context.view_layer.objects.active = painted
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    vert = interior_verts(painted, lambda x: 0.0 < x < 0.3)[0]
    stroke = make_stroke(target, painted, group_mode='ALL')
    stroke.apply([vert], [1.0])
    shown_x = (painted.matrix_world @ painted.data.vertices[vert].co).x + 0.2
    expected = (shown_x - 0.4 + 1.0) / 2.0
    check(close(weights_of(painted, vert)["A"], expected),
          f"shape keys: A = {weights_of(painted, vert)['A']} expected {expected} (as displayed)")


def test_subdivided_target():
    # Too dense to sample as displayed: the base mesh is used, which a linear ramp doesn't mind.
    target, painted = build_scene()
    target.modifiers.new("Subdivision", 'SUBSURF').levels = 2
    vert = interior_verts(painted, lambda x: 0.3 < x < 0.6)[0]
    stroke = make_stroke(target, painted, group_mode='ALL')
    stroke.apply([vert], [1.0])
    check(close(weights_of(painted, vert)["A"], expected_ramp(painted, vert)), "subdivided target: wrong weight")


def test_painted_mesh_with_more_vertices_shown():
    # A modifier that changes the painted mesh's vertex count: the brush goes by the original mesh,
    # without its shape key, and read_geometry says so for the Weight Brushes to warn.
    _target, painted = build_scene()
    bpy.ops.object.mode_set(mode='OBJECT')
    painted.shape_key_add(name="Basis")
    painted.shape_key_add(name="Lift").value = 1.0
    for point in painted.data.shape_keys.key_blocks["Lift"].data:
        point.co.z += 0.1
    geometry = painting.read_geometry(painted, bpy.context.evaluated_depsgraph_get(), surface=True)
    check(geometry.evaluated and close(geometry.co[:, 2].min(), 0.11), "shape key only: not read as displayed")
    painted.modifiers.new("Solidify", 'SOLIDIFY')
    geometry = painting.read_geometry(painted, bpy.context.evaluated_depsgraph_get(), surface=True)
    check(not geometry.evaluated, "solidified: read_geometry didn't say it fell back to the original mesh")
    check(close(geometry.co[:, 2].max(), 0.01) and len(geometry.surface_co) == 2 * len(geometry.co),
          "solidified: expected the original mesh, and the displayed surface to paint on")


def test_paint_after_restore():
    target, painted = build_scene()
    vert = interior_verts(painted, lambda x: 0.3 < x < 0.6)[0]
    stroke = make_stroke(target, painted, group_mode='ALL')
    stroke.apply([vert], [1.0])
    stroke.restore()
    check("C" not in painted.vertex_groups, "restore: C still exists")
    stroke.apply([vert], [1.0])
    check(close(weights_of(painted, vert).get("C", 0.0), 0.5), "paint after restore: C not recreated")


def step_scene():
    """A target whose weights jump from A to B at x = 0 (a hard cutoff when sampled by nearest vertex),
    and a painted grid above it that's split, like a UV seam, along the column at x = -1/15 next to it.
    The target also has a ramp R and L = 0.5. The painted grid has B = 1, a locked L = 0.2 where x < 0.5,
    and X = 0.3 (a group the target lacks) where x < 0."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    target = grid("Target", 10, (0.0, 0.0, 0.0))
    a = target.vertex_groups.new(name="A")
    b = target.vertex_groups.new(name="B")
    ramp = target.vertex_groups.new(name="R")
    for v in target.data.vertices:
        left = 1.0 if v.co.x < -1e-6 else 0.0
        a.add([v.index], left, 'REPLACE')
        b.add([v.index], 1.0 - left, 'REPLACE')
        ramp.add([v.index], (v.co.x + 1.0) / 2.0, 'REPLACE')
    target.vertex_groups.new(name="L").add(list(range(len(target.data.vertices))), 0.5, 'REPLACE')

    size, split = 15, 7
    verts = [(-1.0 + 2.0 * i / size, -1.0 + 2.0 * j / size, 0.0) for j in range(size + 1) for i in range(size + 1)]
    copies = []
    for j in range(size + 1):
        copies.append(len(verts))
        verts.append(verts[j * (size + 1) + split])
    faces = []
    for j in range(size):
        for i in range(size):
            corner = j * (size + 1) + i
            face = [corner, corner + 1, corner + size + 2, corner + size + 1]
            if i == split:
                face[0], face[3] = copies[j], copies[j + 1]
            faces.append(face)
    mesh = bpy.data.meshes.new("Painted")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    painted = bpy.data.objects.new("Painted", mesh)
    painted.location = (0.0, 0.0, 0.01)
    bpy.context.scene.collection.objects.link(painted)
    everything = list(range(len(verts)))
    painted.vertex_groups.new(name="A")
    painted.vertex_groups.new(name="B").add(everything, 1.0, 'REPLACE')
    locked = painted.vertex_groups.new(name="L")
    locked.add([v for v in everything if verts[v][0] < 0.5], 0.2, 'REPLACE')
    locked.lock_weight = True
    painted.vertex_groups.new(name="X").add([v for v in everything if verts[v][0] < 0.0], 0.3, 'REPLACE')

    bpy.context.view_layer.objects.active = painted
    painted.select_set(True)
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    return target, painted, [(split + j * (size + 1), copy) for j, copy in enumerate(copies)]


def row_jump(painted, name):
    """Largest weight difference between neighbouring vertices along the row at y = 1/15 (a middle row of
    the `step_scene` grid)."""
    row = sorted(
        (v.co.x, weights_of(painted, v.index).get(name, 0.0))
        for v in painted.data.vertices if abs(v.co.y - 1.0 / 15.0) < 1e-6
    )
    return max(abs(w1 - w0) for (_x0, w0), (_x1, w1) in zip(row, row[1:]))


def test_auto_smooth():
    everything = None
    # (intensity, largest step of A between neighbours)
    for smooth, most in ((0.0, 1.0), (0.5, 0.5), (1.0, 0.24)):
        target, painted, seam = step_scene()
        everything = list(range(len(painted.data.vertices)))
        stroke = make_stroke(target, painted, group_mode='ALL', sample_mode='VERTEX', fade_unmatched=False,
                             smooth=smooth)
        stroke.apply(everything, [1.0] * len(everything))
        jump = row_jump(painted, "A")
        if not smooth:
            check(close(jump, 1.0), f"auto smooth off: the copy should jump from A to B, jump {jump}")
            continue
        check(jump < most, f"auto smooth {smooth}: A still jumps by {jump} between neighbours")
        for vert in everything:
            w = weights_of(painted, vert)
            check(close(w.get("A", 0.0) + w.get("B", 0.0), 1.0, 1e-3), f"auto smooth: A + B of {vert} is {w}")
            x = (painted.matrix_world @ painted.data.vertices[vert].co).x
            check(close(w.get("L", 0.0), 0.2 if x < 0.5 else 0.0), f"auto smooth: locked L changed on {vert}: {w}")
            check(close(w.get("X", 0.0), 0.3 if x < 0.0 else 0.0),
                  f"auto smooth: X (unmatched, not faded) changed on {vert}: {w}")
            if abs(x) > 0.9:
                check(close(w.get("A", 0.0), 1.0 if x < 0.0 else 0.0, 1e-3),
                      f"auto smooth: away from the cutoff, A of {vert} isn't the target's: {w}")
        for original, copy in seam:
            check(weights_of(painted, original) == weights_of(painted, copy),
                  f"auto smooth: the seam split at {original}/{copy} got different weights")

    # Like the blend, smoothing doesn't pile up when the brush goes over a spot again.
    reference = {vert: weights_of(painted, vert) for vert in everything}
    stroke.changed = False
    stroke.apply(everything, [0.5] * len(everything))
    stroke.apply(everything, [1.0] * len(everything))
    check(not stroke.changed, "auto smooth: going over the same spot again changed the weights")

    # Nor does it depend on the brush: painting one half, then the other, gives the same weights.
    target, painted, _seam = step_scene()
    stroke = make_stroke(target, painted, group_mode='ALL', sample_mode='VERTEX', fade_unmatched=False, smooth=1.0)
    for half in ([v for v in everything if painted.data.vertices[v].co.x < 0.0],
                 [v for v in everything if painted.data.vertices[v].co.x >= 0.0]):
        stroke.apply(half, [1.0] * len(half))
    worst = max(abs(weights_of(painted, vert).get(name, 0.0) - reference[vert].get(name, 0.0))
                for vert in everything for name in ("A", "B", "R"))
    check(worst < 1e-6, f"auto smooth: painting in halves changed the weights by up to {worst}")

    # Current Group: only the active group is smoothed.
    target, painted, _seam = step_scene()
    painted.vertex_groups.active_index = painted.vertex_groups["A"].index
    before = {vert: weights_of(painted, vert) for vert in everything}
    stroke = make_stroke(target, painted, group_mode='ACTIVE', sample_mode='VERTEX', smooth=1.0)
    stroke.apply(everything, [1.0] * len(everything))
    check(row_jump(painted, "A") < 0.6, "auto smooth, Current Group: A wasn't smoothed")
    for vert in everything:
        after = weights_of(painted, vert)
        after.pop("A", None)
        check(after == before[vert], f"auto smooth, Current Group: other groups changed on {vert}: {after}")

    # Esc puts everything back.
    stroke.restore()
    check(all(weights_of(painted, vert) == before[vert] for vert in everything), "auto smooth: restore missed weights")

    # Out of the target's reach (Max Distance), vertices stay as they are, next to smoothed ones.
    target, painted, _seam = step_scene()
    bpy.ops.object.mode_set(mode='OBJECT')
    far = [v.index for v in painted.data.vertices if v.co.y > 0.3]
    for vert in far:
        painted.data.vertices[vert].co.z = 0.2
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    before = {vert: weights_of(painted, vert) for vert in far}
    stroke = make_stroke(target, painted, group_mode='ALL', sample_mode='VERTEX', max_distance=0.05, smooth=1.0)
    stroke.apply(everything, [1.0] * len(everything))
    check(all(weights_of(painted, vert) == before[vert] for vert in far), "auto smooth: out of reach vertices changed")
    check(row_jump(painted, "A") < 0.6, "auto smooth: in reach vertices weren't smoothed")


class _Layout:
    """Stands in for a panel's layout, and notes the properties drawn on it."""

    def __init__(self, props):
        self.props = props

    def __getattr__(self, name):
        def draw(*args, **_kwargs):
            if name == 'prop':
                self.props.append(args[1])
            return _Layout(self.props)
        return draw


def drawn_props(panel, **overrides):
    """The properties ``panel`` draws, in the current context with ``overrides``."""
    props = []

    class Panel:
        layout = _Layout(props)

        def __getattr__(self, name):
            return getattr(panel, name)

    class Context:
        def __getattr__(self, name):
            return overrides[name] if name in overrides else getattr(bpy.context, name)

    panel.draw(Panel(), Context())
    return props


def test_sidebar_offers_what_buttons_need():
    from types import SimpleNamespace
    from magic_fit import tool
    settings = bpy.context.scene.magic_fit
    workspace = bpy.context.workspace or bpy.data.workspaces[0]
    workspace.magic_fit_tab = 'WEIGHTS'
    context = dict(workspace=workspace, region=SimpleNamespace(width=400))
    try:
        # Weight Transfer, on the General sub-tab whatever the brush's mode, needs the body.
        for mode in ('COPY', 'SMOOTH'):
            settings.mode = mode
            check("target" in drawn_props(tool.VIEW3D_PT_magic_fit_tabs, **context),
                  "sidebar: General shows the Body in {} mode".format(mode.title()))
        settings.mode = 'HEELS'
        check("target" not in drawn_props(tool.VIEW3D_PT_magic_fit_tabs, **context), "sidebar: Heels shows no Body")
        # Hair Weights and the face tools refuse Current Group: their panels let it be switched back.
        for mode, panel in (('HAIR', tool.VIEW3D_PT_hair_weights_sidebar), ('FACE', tool.VIEW3D_PT_face_weights_sidebar)):
            settings.mode = mode
            for group_mode in ('ACTIVE', 'ALL'):
                settings.group_mode = group_mode
                shown = "group_mode" in drawn_props(panel, **context)
                check(shown == (group_mode == 'ACTIVE'), "sidebar: {} Weights {} Groups with {}".format(
                    mode.title(), "shows" if group_mode == 'ACTIVE' else "hides", group_mode.title()))
    finally:
        settings.mode = 'COPY'
        settings.group_mode = 'ALL'


def test_register_cycle():
    for _ in range(2):
        magic_fit.unregister()
        magic_fit.register()
    check(hasattr(bpy.types.Scene, "magic_fit"), "register cycle lost the settings")
    check(hasattr(bpy.ops.magic_fit, "paint"), "register cycle lost the operator")


def main():
    magic_fit.register()
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        try:
            test()
            print("ran", test.__name__)
        except Exception:
            FAILURES.append(test.__name__ + " raised")
            traceback.print_exc()
        finally:
            if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
    magic_fit.unregister()
    print("RESULT:", "FAILED" if FAILURES else "OK", len(FAILURES), "failure(s)")
    sys.exit(1 if FAILURES else 0)


main()
