"""A test scene for Straighten: fabric stretched across a cleavage, weighted by copying from the body.

Body: the front of a torso with two breasts (a height field, meters), weighted like an FFXIV body:
j_mune_l / j_mune_r over each breast, fading at its base, j_sebo_c (spine) elsewhere. Garment: a
lace-top-like sheet stretched straight across the chest 3 mm in front of it, spanning the cleavage,
with weights copied from the body by the Weight Brushes. Scaling the breast bones up (like a
Customize+ profile) makes the copied weights fold the fabric into the cleavage. `add_strap` adds a
strap running from the front of a breast straight down to the chest below it, which kinks under the
breast when it's scaled.
"""

import bpy
import numpy as np

from magic_fit import painting

BREAST_R = 0.075
BREAST_X = 0.07
BREAST_Y = -0.075      # breast centers; the torso front is at y = -0.09
HEAD_Y = -0.04         # j_mune heads, inside the chest behind each breast
TORSO_FRONT = -0.09
BONES = ("j_mune_l", "j_mune_r", "j_sebo_c")


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def body_front(x, z):
    """Front surface y of the chest (more negative = further forward)."""
    y = np.full(np.broadcast(x, z).shape, TORSO_FRONT)
    for side in (-1.0, 1.0):
        front = BREAST_Y - np.sqrt(np.clip(BREAST_R ** 2 - (x - side * BREAST_X) ** 2 - z ** 2, 0.0, None))
        h = np.clip(0.5 + 0.5 * (y - front) / 0.01, 0.0, 1.0)  # smooth minimum: the breast blends in
        y = y * (1 - h) + front * h - 0.01 * h * (1 - h)
    return y


def envelope(y_row):
    """Front convex envelope of a profile y(x) sampled evenly: a sheet stretched across its front."""
    hull = []
    for i in range(len(y_row)):
        while len(hull) >= 2:
            i0, i1 = hull[-2], hull[-1]
            if (i1 - i0) * (y_row[i] - y_row[i0]) - (i - i0) * (y_row[i1] - y_row[i0]) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(np.arange(len(y_row)), hull, y_row[hull])


def grid_object(name, xs, zs, front):
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    co = np.stack((X.ravel(), front(X, Z).ravel(), Z.ravel()), axis=1)
    index = np.arange(len(xs) * len(zs)).reshape(len(xs), len(zs))
    faces = np.stack((index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(),
                      index[:-1, 1:].ravel()), axis=1)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(co.tolist(), [], faces.tolist())
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def build(spacing=0.005):
    """Build the scene (replacing everything) and copy the body's weights onto the garment, in Weight
    Paint mode. Returns (rig, body, garment, xs, zs); the garment's vertex (i, j) is at xs[i], zs[j]."""
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    rig = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for name, head, tail in (
            ("j_sebo_c", (0.0, 0.0, -0.15), (0.0, 0.0, 0.15)),
            ("j_mune_l", (-BREAST_X, HEAD_Y, 0.0), (-BREAST_X, HEAD_Y - 0.06, 0.0)),
            ("j_mune_r", (BREAST_X, HEAD_Y, 0.0), (BREAST_X, HEAD_Y - 0.06, 0.0)),
    ):
        bone = rig.data.edit_bones.new(name)
        bone.head, bone.tail = head, tail
    bpy.ops.object.mode_set(mode='OBJECT')

    body = grid_object("Body", np.linspace(-0.17, 0.17, 69), np.linspace(-0.14, 0.12, 53), body_front)
    co = np.array([v.co[:] for v in body.data.vertices])
    mune_l = smoothstep(0.075, 0.04, np.hypot(co[:, 0] + BREAST_X, co[:, 2]))
    mune_r = smoothstep(0.075, 0.04, np.hypot(co[:, 0] - BREAST_X, co[:, 2]))
    for name, values in zip(BONES, (mune_l, mune_r, np.clip(1.0 - mune_l - mune_r, 0.0, 1.0))):
        group = body.vertex_groups.new(name=name)
        for i, weight in enumerate(values.tolist()):
            if weight > 1e-4:
                group.add([i], weight, 'REPLACE')
    body.modifiers.new("Armature", 'ARMATURE').object = rig

    xs = np.arange(-0.15, 0.15 + 1e-9, spacing)
    zs = np.arange(-0.10, 0.08 + 1e-9, spacing)

    def garment_front(X, Z):
        Y = np.empty_like(X)
        for j in range(X.shape[1]):
            Y[:, j] = envelope(body_front(X[:, j], Z[:, j])) - 0.003
        return Y

    garment = grid_object("Garment", xs, zs, garment_front)
    garment.modifiers.new("Armature", 'ARMATURE').object = rig
    for name in BONES:
        garment.vertex_groups.new(name=name)
    rig.select_set(False)
    body.select_set(False)
    garment.select_set(True)
    bpy.context.view_layer.objects.active = garment
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    depsgraph = bpy.context.evaluated_depsgraph_get()
    sampler = painting.TargetSampler(body, depsgraph)
    copy = painting.CopyStroke(garment, body, sampler, painting.read_geometry(garment, depsgraph).co)
    copy.apply(list(range(len(garment.data.vertices))), [1.0] * len(garment.data.vertices))
    garment.data.update()
    return rig, body, garment, xs, zs


STRAP_X = 0.07          # the strap runs down over the center of the breast at +x (j_mune_r)
STRAP_ZS = np.arange(-0.135, 0.035 + 1e-9, 0.005)
STRAP_XS = np.array([STRAP_X - 0.01, STRAP_X, STRAP_X + 0.01])


def add_strap(rig, body, both=False):
    """A strap 2 cm wide (a ribbon, like a sling bikini's) lying on the upper half of the breast at +x, then
    stretched straight from the front of the breast down to where it touches the chest again at the bottom:
    2 mm in front of the front convex envelope of the body there. Weighted by copying from the body, like
    the garment. With ``both``, the object also has the mirrored strap on the other breast, as a separate
    piece after it. Returns the object; the strap's vertex (i, j) is at STRAP_XS[i], STRAP_ZS[j] (then the
    mirrored strap's, at -STRAP_XS[i])."""
    co, faces = [], []
    for side in ((1.0, -1.0) if both else (1.0,)):
        X, Z = np.meshgrid(side * STRAP_XS, STRAP_ZS, indexing="ij")
        Y = np.empty_like(X)
        for i in range(X.shape[0]):
            Y[i, :] = envelope(body_front(X[i, :], Z[i, :])) - 0.002
        base = len(co)
        co += np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1).tolist()
        index = base + np.arange(X.size).reshape(X.shape)
        quads = np.stack((index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(),
                          index[:-1, 1:].ravel()), axis=1)
        faces += (quads if side > 0 else quads[:, ::-1]).tolist()
    mesh = bpy.data.meshes.new("Strap")
    mesh.from_pydata(co, [], faces)
    strap = bpy.data.objects.new("Strap", mesh)
    bpy.context.scene.collection.objects.link(strap)
    strap.modifiers.new("Armature", 'ARMATURE').object = rig
    for name in BONES:
        strap.vertex_groups.new(name=name)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    sampler = painting.TargetSampler(body, depsgraph)
    copy = painting.CopyStroke(strap, body, sampler, painting.read_geometry(strap, depsgraph).co)
    copy.apply(list(range(len(strap.data.vertices))), [1.0] * len(strap.data.vertices))
    strap.data.update()
    return strap


def strap_sag(points, piece=0):
    """How far a strap's middle line (``piece`` 0: the strap at +x, 1: the mirrored one) lies behind the
    straight line from the front of the breast to its bottom end, seen from the side (m): 0 when it runs
    straight from one to the other."""
    count = len(STRAP_XS) * len(STRAP_ZS)
    middle = points[piece * count:(piece + 1) * count].reshape(len(STRAP_XS), len(STRAP_ZS), 3)[1]
    top = np.argmin(middle[:, 1])        # the frontmost point: the front of the breast
    below = middle[:top + 1]              # from the bottom end (index 0) up to it
    line = np.interp(below[:, 2], [below[0, 2], below[-1, 2]], [below[0, 1], below[-1, 1]])
    return float((below[:, 1] - line).max())


def scale_breasts(rig, factor):
    for name in ("j_mune_l", "j_mune_r"):
        rig.pose.bones[name].scale = (factor, factor, factor)
    bpy.context.view_layer.update()


def displayed(obj):
    bpy.context.view_layer.update()
    return painting.read_geometry(obj, bpy.context.evaluated_depsgraph_get(), source='INDEXED').co


def weights(obj):
    table = np.zeros((len(obj.data.vertices), len(obj.vertex_groups)))
    for vert in obj.data.vertices:
        for elem in vert.groups:
            table[vert.index, elem.group] = elem.weight
    return table


def fold(points, xs, zs):
    """How far the sheet folds back over itself sideways, worst row (m): 0 when every row runs left to
    right as at rest."""
    X = points[:, 0].reshape(len(xs), len(zs))
    return max((np.maximum.accumulate(X[:, j]) - X[:, j]).max() for j in range(len(zs)))


def distortion(points, rest, xs, zs, reach=0.06):
    """Mean difference (m) between the sheet's local shape and its rest shape (Laplacians), over the
    middle of the chest."""
    P = points.reshape(len(xs), len(zs), 3)
    R = rest.reshape(len(xs), len(zs), 3)

    def laplacian(A):
        return 0.25 * (A[:-2, 1:-1] + A[2:, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:]) - A[1:-1, 1:-1]
    middle = np.abs(R[1:-1, 1:-1, 0]) < reach
    return np.linalg.norm(laplacian(P) - laplacian(R), axis=2)[middle].mean()


def notch(points, xs, zs):
    """Seen from above: how far the front outline between the breasts is behind the straight line
    from one breast's front to the other's (m)."""
    Y = points[:, 1].reshape(len(xs), len(zs))
    X = points[:, 0].reshape(len(xs), len(zs)).mean(axis=1)
    front = Y.min(axis=1)
    left = np.argmin(np.where(X < 0.0, front, np.inf))
    right = np.argmin(np.where(X > 0.0, front, np.inf))
    between = slice(left, right + 1)
    line = np.interp(X[between], [X[left], X[right]], [front[left], front[right]])
    return (front[between] - line).max()
