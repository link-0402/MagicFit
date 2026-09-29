# SPDX-License-Identifier: GPL-3.0-or-later

"""Face weights: giving custom faces, and what sits on them, the weights of FFXIV's face bones.

FFXIV moves faces with a face skeleton on top of the head bone (j_kao): three bones per eyelid
(j_f_mabup_* above, j_f_mabdn_* below) that turn about the eyeball's center when the eyes close, the
eyeballs (j_f_eyepuru_*), brows, cheeks, nose, the lips (j_f_ulip/umlip/uslip above, j_f_dlip/dmlip/
dslip below), the jaw (j_f_dago), gums and tongue. Every face of the game has its own face skeleton
(faceskeletontemplate.est picks it), and its own weights. Custom faces are nearly always new meshes
(denser, resculpted), so the weights can't be copied vertex by vertex.

Face Weights takes them from the game's face the custom face replaces (read from the user's game install,
see facedata):

- Eye openings, the lip line and the neck opening are found on both faces. An eye opening is where the
  eyeball shows through the lids seen from the front; its top and bottom edges are the lid margins. The
  lip line is the crease between the lips (the deepest line seen from the front). The neck ring is the
  bottom opening of the head.
- The custom face is warped onto the game's (a thin plate spline through the margins, the lip line, the
  neck ring and points away from them), so its lid margins and lip line land on the game's.
- Each vertex takes the weights of the closest point of the game's face, never from the other side of
  an eye or of the mouth (upper lid vs lower lid, upper lip vs lower lip; the game's sides are known
  from its weights, the custom face's from where it lies around the margins and the lip line, spread a
  few millimetres inward along the surface), and preferably from a surface facing the same way: lips
  touch and the gums sit right behind them, so this keeps them apart.
- The eyes must close as early in the game's blink (its face animation) as the game's face does, since
  the game shows the blink's peak for an instant only. The lids of a bigger eye may stop short: where the
  eyeball still shows, the lid weights are raised (fading out away from the margin) until it's covered, or
  until they can't go higher, which is reported. The game's lid weights close a smaller eye much too early,
  and its lids slide on past each other: they're lowered.
- The neck ring (where the face meets the body) must be exactly the game's, or a seam shows: its
  positions, normals and weights are copied from the game's face.

Other parts are weighted by what they are: eyeballs follow their eye bone; lashes (parts along a lid
margin, standing off the skin) take the weights of the margin point they grow from, so they stay on the
lid as it closes; parts lying on the skin (brows, makeup) follow the skin under them; parts inside the
mouth (teeth, tongue) take the game's mouth weights; other parts (piercings, studs, horns) move as one
piece with the skin they're attached to.

Like `painting`, nothing here depends on the 3D view.
"""

import math
import re

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import facedata, fitting, goals, hairing, painting, straighten
from .goals import DEFAULT_MAX_GROUPS, smoothstep

RACE_NAMES = hairing.RACE_NAMES
HEAD = "j_kao"
NECK = "j_kubi"
UPPER_BACK = "j_sebo_c"
SIDES = ("l", "r")
EYE_BONE = "j_f_eye_{:s}"
EYEBALL_BONE = "j_f_eyepuru_{:s}"
UPPER_LID = "j_f_mabup_"
LOWER_LID = "j_f_mabdn_"
LIP_BONES = ("j_f_ulip_01_l", "j_f_ulip_01_r", "j_f_dlip_01_l", "j_f_dlip_01_r")
UPPER_MOUTH = ("j_f_ulip", "j_f_umlip", "j_f_uslip", "j_f_hagukiup")
LOWER_MOUTH = ("j_f_dlip", "j_f_dmlip", "j_f_dslip", "j_f_dago", "j_f_hagukidn", "j_f_bero")
FACE_NAME = re.compile(r"c(\d{4})f(\d{4})", re.IGNORECASE)
# Poses of the game's face animations, kept at their peak (see facedata.POSES); the Test Poses show the blink at
# the closing share of the game's face (see FaceModel.closing_share), since the game shows its peak for an instant.
POSES = (
    ('BLINK', "Blink", "The game's blink, at the moment its own face's eyes are shut"),
    ('TALK', "Talk", "Mouth open, as when talking"),
    ('SHOUT', "Shout", "Mouth open, as when shouting"),
    ('ANGRY', "Angry", "The angry expression"),
    ('CLENCH', "Clench", "Teeth clenched"),
    ('OUCH', "Ouch", "The pained expression, mouth open"),
    ('DIE', "Die", "The expression when knocked out"),
    ('SALUTE', "Salute", "The salute expression"),
    ('BOW', "Bow", "The bow expression"),
)

# Side labels: which side of a feature a vertex is on, so it never takes weights from the other.
NEUTRAL, LEFT_UPPER, LEFT_LOWER, RIGHT_UPPER, RIGHT_LOWER, MOUTH_UPPER, MOUTH_LOWER = range(7)
OPPOSITE = {LEFT_UPPER: LEFT_LOWER, LEFT_LOWER: LEFT_UPPER, RIGHT_UPPER: RIGHT_LOWER,
            RIGHT_LOWER: RIGHT_UPPER, MOUTH_UPPER: MOUTH_LOWER, MOUTH_LOWER: MOUTH_UPPER}
EYE_LABELS = {"l": (LEFT_UPPER, LEFT_LOWER), "r": (RIGHT_UPPER, RIGHT_LOWER)}

# Frontal scans: rays along +Y (from the front) this far apart.
SCAN_STEP = 0.00025
# The eye scan reaches this much beyond the eyeball's radius around the eye bone, at least EYE_WINDOW.
EYE_WINDOW = (0.026, 0.02)
# Closure checks (the eyes in the blink) cast rays only this far around the opening at rest (its lid margins),
# since closing lids only cover more of it; an opening found reaching that box's edge is scanned in full after all.
OPENING_MARGIN = 0.003
# The lip scan: half width and half height around the lip bones.
MOUTH_WINDOW = (0.032, 0.016)
# The lip line is where the surface lies deepest compared with its surroundings (a box this wide), within
# LIP_SEARCH above or below the lip bones; it ends where that dip is shallower than LIP_DEPTH, over gaps
# up to LIP_GAP.
LIP_BLUR = 0.005
LIP_SEARCH = 0.007
LIP_DEPTH = 0.0004
LIP_GAP = 0.0015
# A skin surface this far behind the eyeball's front doesn't cover it.
EYEBALL_TOLERANCE = 0.0002
# Lash lines are the lids' silhouettes against the eyeball seen this far (radians) from the front, from above
# for the upper lid and from below for the lower one: their front edges. The game's lashes grow within about
# half a millimetre of them.
LASH_TILT = math.radians(35.0)
# Rays that far apart are enough there (the lines are smoothed): a quarter of the rays of the front scans.
LASH_SCAN_STEP = 0.0005
# Seen at that tilt, only skin within this of the eyeball's sphere hides it: the lids, which lie on the
# eyeball, and not the brows or cheeks in front of them, which would hide the lids' edges. A lash line farther
# than LASH_LINE_LIMIT from its margin (median) is taken to be wrong, and the margin is used.
LID_SHELL = 0.005
LASH_LINE_LIMIT = 0.004
# Samples of each lid margin and of the lip line for the warp.
MARGIN_SAMPLES = 24
LIP_SAMPLES = 32
# Warp anchors: points of the game's face this far from every feature, about this far apart.
ANCHOR_CLEARANCE = 0.015
ANCHOR_SPACING = 0.012
ANCHOR_MAX_GAP = 0.01
WARP_REGULARIZATION = 1e-7
# Side labels: within this band above an upper margin (below a lower one, around the lip line) on the
# visible surface, spread this far along the surface into the sockets, and into the lips' inside.
SIDE_BAND = 0.006
SOCKET_REACH = 0.006
LIP_REACH = 0.004
# Mouth corners: this close to the ends of the lip line, upper and lower lips mix and nothing is labelled.
CORNER_GAP = 0.002
# A reference vertex is on a side when that side's bones have this much more weight than the other's.
SIDE_THRESHOLD = 0.05
# A reference surface facing away from a vertex costs this much extra distance per unit of (1 - cos);
# candidates are looked for this much farther than the nearest point, when the nearest faces away.
NORMAL_PENALTY = 0.002
NORMAL_SEARCH = 0.003
NORMAL_GOOD = 0.95
# Eyelid closing. The game shows its blink's peak for an instant only (5 ms within 1 % of the turn; at 60 frames
# a second the most closed frame shows 97 % of it at worst), and its own faces close well before it: at 75 to 100 %
# of the turn, 92 % for the median face (its closing share). So lids should meet by the closing share of the game's
# face, overlapping this much there; a raise fades out over LID_FALLOFF from the margin and is spread over
# LID_SMOOTH scan columns. Lids meeting more than CLOSE_SLACK of the turn earlier slide on past each other (small
# eyes given the game's big-eyed lid weights): all of the eye's lid weights are lowered alike. At most LID_ROUNDS
# rounds; shares are found to CLOSE_PRECISION.
LID_OVERLAP = 0.0004
LID_FALLOFF = 0.012
LID_SMOOTH = 9
LID_ROUNDS = 5
CLOSE_SLACK = 0.05
CLOSE_PRECISION = 0.01
# Neck ring: custom ring vertices at most this far from the game's ring are snapped onto it.
RING_TOLERANCE = 0.002
# A mesh without the game's neck opening is the face when it covers this share of the game's face.
SKIN_COVER = 0.5
# Parts: lying on the skin when 90 % of the part is closer to it than this; a lash when it comes this
# close to a lid margin; the attach point of a rigid part is where it comes within ATTACH_RANGE of the
# closest point it has to the skin.
SHELL_DISTANCE = 0.002
LASH_DISTANCE = 0.0035
ATTACH_RANGE = 0.001
# A part by a lid margin standing out this far from the skin somewhere is lashes, however close to the skin
# most of it is (lashes sunk into the lid); makeup and liner on the lids lie on the skin all over.
LASH_STANDS_OUT = 0.0012
# Parts inside the mouth smaller than this (across) move as one piece: studs, single teeth.
RIGID_SIZE = 0.012
# A part by the margins with more than this share over the eye opening is a film over the eyeball: it
# stretches between the lids instead of growing from them.
FILM_SHARE = 0.4
# The roots of a lash part: every LASH_ROOT_SPACING along a lid, its points there scoring within
# LASH_ROOT_BAND of the best, by their distance to the lid's lash line plus how much farther from the eyeball's
# center they are than the innermost there. Every other point hangs from the root closest along the part's
# surface, and moves with it. A part has lashes on a lid when LASH_BEYOND of it reaches past that lid's
# margin, along LASH_COVER of the lid at least.
LASH_ROOT_BAND = 0.00075
LASH_ROOT_SPACING = 0.001
LASH_BEYOND = 0.05
LASH_COVER = 0.05
# As in the game's lashes, the lid weight of a lash fades along it: from LASH_TIP_START from its root to
# LASH_TIP_START + LASH_TIP_LENGTH, up to LASH_TIP_SHARE of it goes to the head, so the tips lag a little
# as the lid closes.
LASH_TIP_START = 0.002
LASH_TIP_LENGTH = 0.01
LASH_TIP_SHARE = 0.3
# Where the game's lashes grow, per lid (upper, lower): this far out from the eyeball and this far along the
# lid (up the upper one, down the lower one) from its lash line, in the middle (median) of all its faces.
LASH_ROOT_OUT = (0.0004, 0.001)
LASH_ROOT_ALONG = (0.0003, 0.0008)
# Snapping moves lashes by a smooth average along the lid of what each millimetre of it needs, this wide.
LASH_SNAP_SMOOTH = 0.0025
# A part along both margins (RIM_SPAN of it along each) with more than RIM_SHARE over the opening is the rim
# of the opening (the game's tear lines) rather than lashes: a film, not snapped.
RIM_SHARE = 0.15
RIM_SPAN = 0.2
# Eyeballs, found by shape: a sphere cap (points within EYEBALL_RMS of its radius from it) spanning at
# least EYEBALL_SPAN degrees across and up and down, as the game's eyes do (lashes, lid makeup and tear
# lines around an eye are narrow bands).
EYEBALL_RMS = 0.05
EYEBALL_SPAN = 90.0
# Parts where the game's face has one of its own (ears, horns, scales, beards, lashes; half of the part
# within this of it, once warped) take its weights.
EXTRA_DISTANCE = 0.003
# Regions of the skin the repairs change: around the eyes (the margins, and the sockets) and the mouth,
# fully within the first distance, fading out by the second.
EYE_REGION = (0.006, 0.012)
MOUTH_REGION = (0.008, 0.015)
# Weights below this (half a byte of the game's) are dropped.
MIN_WEIGHT = 0.5 / 255.0


class FaceError(Exception):
    """Why face weights can't be worked out; the message is meant for the user."""


# -----------------------------------------------------------------------------
# Reference data

def _quat_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def _quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz))


def _quat_power(q, share):
    """The turn of quaternion ``q`` (xyzw) taken ``share`` of the way, about the same axis."""
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    if q[3] < 0.0:
        q = -q
    half = math.acos(min(1.0, q[3]))
    if half < 1e-12:
        return np.array((0.0, 0.0, 0.0, 1.0))
    axis = q[:3] / math.sin(half)
    return np.concatenate([axis * math.sin(half * share), [math.cos(half * share)]])


# Game model space (Y up, facing +Z) to Blender (Z up, facing -Y).
TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class Skeleton:
    """A race's face skeleton with the body bones above it: ``names``, ``parents`` (indices, -1 for the
    root), ``local`` rest transforms in game space (rows of translation, quaternion xyzw, scale) and ``rest``
    matrices in Blender model space (rotation and position of each bone, like TexTools' FBX exports)."""

    def __init__(self, race, number, names, parents, local, rest, poses):
        self.race = race
        self.number = number
        self.names = names
        self.parents = parents
        self.local = local
        self.rest = rest
        self.index = {name: i for i, name in enumerate(names)}
        self._poses = poses

    def head(self, name):
        return self.rest[self.index[name]][:3, 3].copy()

    def has(self, name):
        return name in self.index

    def face_bones(self):
        """The face skeleton's own bones: those below the head."""
        head = self.index.get(HEAD)
        result = []
        for i, name in enumerate(self.names):
            parent = self.parents[i]
            while parent >= 0 and parent != head:
                parent = self.parents[parent]
            if head is not None and parent == head:
                result.append(name)
        return result

    def rig(self, offset=None):
        """The bones as `hairing.ensure_bones` adds them to an armature (moved by ``offset``)."""
        return FaceRig(self, offset)

    def pose_names(self):
        return [pose for pose, _label, _description in POSES if pose in self._poses]

    def _model(self, local):
        out = np.zeros((len(self.names), 4, 4))
        for i, row in enumerate(local):
            m = np.eye(4)
            m[:3, :3] = _quat_matrix(row[3:7]) * row[7:10][None, :]
            m[:3, 3] = row[0:3]
            parent = self.parents[i]
            out[i] = m if parent < 0 else out[parent] @ m
        return out

    def deformations(self, pose, share=1.0):
        """{bone: 4x4} that take points (Blender model space) from the rest pose to ``pose``: how the game
        moves what a bone fully weights. With ``share`` below 1, the pose's changes are taken that share of the
        way (turns about the same axes), as its animation passes on its way to the peak. Empty for poses this
        skeleton has none of."""
        entry = self._poses.get(pose)
        if entry is None:
            return {}
        bones, deltas = entry
        local = self.local.copy()
        for bone, delta in zip(bones, deltas):
            ref = local[bone].copy()
            local[bone, 0:3] = ref[0:3] + share * delta[0:3]
            local[bone, 3:7] = _quat_multiply(ref[3:7], delta[3:7] if share == 1.0 else _quat_power(delta[3:7], share))
            local[bone, 7:10] = ref[7:10] * delta[7:10] ** share
        rest = self._model(self.local)
        posed = self._model(local)
        to_blender = np.eye(4)
        to_blender[:3, :3] = TO_BLENDER
        result = {}
        for i, name in enumerate(self.names):
            change = to_blender @ posed[i] @ np.linalg.inv(rest[i]) @ to_blender.T
            if not np.allclose(change, np.eye(4), atol=1e-9):
                result[name] = change
        return result


class FaceRig:
    """A face skeleton's bones (`hairing.Bone`) by name, moved by ``offset``: what `hairing.ensure_bones`
    takes. Bones it adds are short, to keep them from hiding the face."""

    bone_length = 0.005

    def __init__(self, skeleton, offset=None):
        self.bones = [hairing.Bone(name, skeleton.names[parent] if parent >= 0 else None, rest)
                      for name, parent, rest in zip(skeleton.names, skeleton.parents, skeleton.rest)]
        self.by_name = {bone.name: bone for bone in self.bones}
        self.offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)


class Part:
    """A mesh part of one of the game's faces: ``points``, ``normals``, ``tris``, ``weights`` (over its face's
    names), ``kind`` (0 the skin, 1 another part in the skin's kind of material: ears, horns, scales, beards,
    2 another: lashes, brows, tear films, face paint) and the ``attributes`` that show or hide it."""

    def __init__(self, kind, attributes, points, normals, tris, weights):
        self.kind = kind
        self.attributes = attributes
        self.points = points
        self.normals = normals
        self.tris = tris
        self.weights = weights


def _joined(parts):
    """(points, tris, weights) of ``parts`` as one mesh."""
    if not parts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), None
    starts = np.cumsum([0] + [len(part.points) for part in parts[:-1]])
    return (np.vstack([part.points for part in parts]),
            np.vstack([part.tris + start for part, start in zip(parts, starts)]),
            np.vstack([part.weights for part in parts]))


class FaceModel:
    """One of the game's faces: its skin (``points``, ``normals``, ``tris``), its ``weights`` over ``names``,
    its eyeball ``radius`` and its other parts (``extras``, Part); features and lookup structures are worked
    out when first needed."""

    def __init__(self, race, number, skeleton, skin, extras, names, radius):
        self.race = race
        self.number = number
        self.skeleton = skeleton
        self.points = skin.points
        self.normals = skin.normals
        self.tris = skin.tris
        self.weights = skin.weights
        self.extras = extras
        self.names = names
        self.radius = radius
        self._features = None
        self._transfer = None
        self._extras_transfer = None
        self._ring = None
        self._bvh = None
        self._closing = None

    def bvh(self):
        """BVHTree of the skin."""
        if self._bvh is None:
            self._bvh = bvh_of(self.points, self.tris)
        return self._bvh

    def eyes(self):
        """{side: Eye}: the game's eyeballs, spheres about the eye bones."""
        if self.radius <= 0.0:
            return {}
        return {side: Eye(self.skeleton.head(EYE_BONE.format(side)), self.radius)
                for side in SIDES if self.skeleton.has(EYE_BONE.format(side))}

    def features(self):
        if self._features is None:
            self._features = Features(self.points, self.tris, self.skeleton, self.eyes())
        return self._features

    def closing_share(self):
        """The share of the game's blink by which this face has closed its eyes, with its own weights (see
        `closing_share`; scanned around its openings at rest); 1 when it only does at the peak, or not even then."""
        if self._closing is None:
            features = self.features()
            boxes = {side: features.opening_box(side) for side in features.eyes}
            self._closing = closing_share(self.points, self.tris, self.names, self.weights, self.eyes(),
                                          self.skeleton, boxes) or 1.0
        return self._closing

    def transfer(self):
        """SidedTransfer of the skin, with the parts that belong to it (ears, horns, scales, beards)."""
        if self._transfer is None:
            points, tris, weights = _joined([Part(0, "", self.points, self.normals, self.tris, self.weights)] +
                                            [part for part in self.extras if part.kind == 1])
            self._transfer = SidedTransfer(points, tris, weights, weight_labels(self.names, weights))
        return self._transfer

    def extras_transfer(self):
        """(SidedTransfer, BVHTree) of the other parts, or None when the face has none."""
        if self._extras_transfer is None:
            points, tris, weights = _joined(self.extras)
            self._extras_transfer = (SidedTransfer(points, tris, weights, np.zeros(len(points), dtype=np.int64)),
                                     bvh_of(points, tris)) if len(tris) else False
        return self._extras_transfer or None

    def ring(self):
        """(positions, normals, weights) of the neck ring's vertices (every split copy), or None."""
        if self._ring is None:
            found = neck_ring(self.points, self.tris)
            if found is None:
                self._ring = False
            else:
                verts = found[2]
                self._ring = (self.points[verts], self.normals[verts], self.weights[verts])
        return self._ring or None


class Reference:
    """FFXIV's faces, face skeletons and face poses, read from the user's game install (see facedata)."""

    _instance = None
    _unsaved = False  # loaded() found nothing saved to use: only get() reads the faces, from the game
    # Counts the times the faces were read again or forgotten, so what's kept from them can tell it's old.
    generation = 0

    @classmethod
    def get(cls, progress=None):
        """The game's faces: the ones saved in the add-on's user folder, or when there are none for the game
        installed, read from the game first (a few seconds; ``progress`` is called with the share done).
        Raises facedata.FaceDataError when that can't be done."""
        if cls._instance is None:
            cls._instance = cls(facedata.load(progress))
        return cls._instance

    @classmethod
    def loaded(cls):
        """The game's faces when they're read already or saved (quick to load), else None: for polls and
        drawing, which mustn't take seconds."""
        if cls._instance is None and not cls._unsaved:
            data = facedata.saved_data()
            if data is None:
                cls._unsaved = True
            else:
                cls._instance = cls(data)
        return cls._instance

    @classmethod
    def read_again(cls, progress=None):
        """Reads the game's faces from the game again (see get), and forgets what was worked out with the old
        ones."""
        data = facedata.load(progress, again=True)
        cls._instance = cls(data)
        cls._unsaved = False
        cls.generation += 1
        _goal_cache.clear()
        return cls._instance

    @classmethod
    def forget(cls):
        """Forget the faces read, whether they were saved, and what was worked out with them: the game folder
        setting changed."""
        cls._instance = None
        cls._unsaved = False
        cls.generation += 1
        _goal_cache.clear()
        facedata.clear_cache()

    @classmethod
    def from_file(cls, path):
        """The faces in a file saved by facedata (or like it), for tools and tests."""
        with np.load(path, allow_pickle=False) as data:
            return cls({key: data[key] for key in data.files})

    def __init__(self, d):
        self.races = tuple(str(r) for r in d["races"])
        self._bones = [str(n) for n in d["bone_names"]]
        pose_names = [str(p) for p in d["pose_names"]]
        pose_rows = {}
        for s, kind, start, count in zip(d["pose_skel"], d["pose_kind"], d["pose_start"], d["pose_count"]):
            pose_rows.setdefault(int(s), {})[pose_names[kind]] = (
                d["pt_bone"][start:start + count].astype(np.int64),
                d["pt_delta"][start:start + count].astype(np.float64))
        self._skeleton_data = {}
        for s, (race, number, start, count) in enumerate(zip(d["skel_race"], d["skel_id"], d["skel_start"],
                                                              d["skel_count"])):
            rows = slice(int(start), int(start + count))
            self._skeleton_data[(self.races[race], int(number))] = (
                [self._bones[b] for b in d["sb_bone"][rows]], d["sb_parent"][rows].astype(np.int64).tolist(),
                d["sb_local"][rows].astype(np.float64), d["sb_rest"][rows].astype(np.float64), pose_rows.get(s, {}))
        self._skeletons = {}
        self._faces = {}
        part_rows = {}
        v = t = 0
        for face, kind, attributes, vcount, tcount in zip(d["part_face"], d["part_kind"], d["part_attributes"],
                                                          d["part_vcount"], d["part_tcount"]):
            part_rows.setdefault(int(face), []).append((int(kind), str(attributes), v, int(vcount), t, int(tcount)))
            v += int(vcount)
            t += int(tcount)
        self._face_rows = {}
        for i, (race, number, skel) in enumerate(zip(d["face_race"], d["face_id"], d["face_skel"])):
            self._face_rows[(self.races[race], int(number))] = (int(skel), float(d["face_eye_radius"][i]),
                                                                part_rows.get(i, []))
        self._arrays = {key: d[key] for key in ("co", "normal", "tris", "w_bone", "w_value")}
        # The game files read from mods TexTools installed into the game's files (none in data saved without).
        self.textools = {str(path) for path in d["textools_files"]} if "textools_files" in d else set()

    def faces(self, race):
        """The game's face numbers of ``race``, ascending."""
        return sorted(number for r, number in self._face_rows if r == race)

    def resolve(self, race, number):
        """The game face that face ``number`` of ``race`` uses: itself, or for a face the game has no model
        of (Au Ra and Viera faces 101-104), the face 100 below it, which it shares its model and face skeleton
        with. None when there's neither."""
        if (race, number) in self._face_rows:
            return number
        if number > 100 and (race, number - 100) in self._face_rows:
            return number - 100
        return None

    def has_face(self, race, number):
        return self.resolve(race, number) is not None

    def skeleton_number(self, race, number):
        return self._face_rows[(race, self.resolve(race, number))][0]

    def textools_files(self, race, number):
        """The files the game's face ``number`` of ``race`` (see `resolve`) was read from that are mods TexTools
        installed into the game's files: its model, skeletons, animations or the table of face skeletons."""
        number = self.resolve(race, number)
        skeleton = self.skeleton_number(race, number)
        paths = (facedata.MODEL_PATH.format(race, number), facedata.FACE_SKELETON_PATH.format(race, skeleton),
                 facedata.FACE_ANIMATIONS_PATH.format(race, skeleton), facedata.BASE_SKELETON_PATH.format(race),
                 facedata.EST_PATH)
        return [path for path in paths if path in self.textools]

    def skeleton(self, race, number):
        """The Skeleton of face skeleton ``number`` of ``race``."""
        key = (race, number)
        if key not in self._skeletons:
            names, parents, local, rest, poses = self._skeleton_data[key]
            self._skeletons[key] = Skeleton(race, number, names, parents, local, rest, poses)
        return self._skeletons[key]

    def face(self, race, number):
        """The FaceModel of the game's face ``number`` of ``race`` (see `resolve`)."""
        key = (race, self.resolve(race, number))
        if key not in self._faces:
            skel, radius, rows = self._face_rows[key]
            a = self._arrays
            used = set()
            for _kind, _attributes, v, vcount, _t, _tcount in rows:
                bones = a["w_bone"][v:v + vcount]
                used.update(bones[a["w_value"][v:v + vcount] > 0].tolist())
            used = sorted(used)
            column = {b: i for i, b in enumerate(used)}
            parts = []
            for kind, attributes, v, vcount, t, tcount in rows:
                points = a["co"][v:v + vcount].astype(np.float64)
                normals = a["normal"][v:v + vcount].astype(np.float64)
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                bones = a["w_bone"][v:v + vcount].astype(np.int64)
                values = a["w_value"][v:v + vcount].astype(np.float64) / 255.0
                weights = np.zeros((vcount, len(used)))
                for k in range(bones.shape[1]):
                    has = values[:, k] > 0
                    np.add.at(weights, (np.flatnonzero(has), [column[b] for b in bones[has, k]]), values[has, k])
                parts.append(Part(kind, attributes, points, normals, a["tris"][t:t + tcount].astype(np.int64), weights))
            skin = next(part for part in parts if part.kind == 0)
            self._faces[key] = FaceModel(key[0], key[1], self.skeleton(key[0], skel), skin,
                                         [part for part in parts if part is not skin],
                                         [self._bones[b] for b in used], radius)
        return self._faces[key]


def face_data_problem():
    """Why Face can't have the game's faces, as a message (the game folder isn't found), or None."""
    if Reference._instance is not None:
        return None
    return facedata.problem()


def face_label(number):
    return "f{:04d}".format(number)


def face_from_names(texts):
    """(race, face) from names like c0801f0002 among ``texts``, or None."""
    for text in texts:
        m = FACE_NAME.search(text or "")
        if m is not None:
            return "c" + m.group(1), int(m.group(2))
    return None


def object_face_names(obj):
    """Texts that may name the game face ``obj`` replaces: its name, its mesh's, its materials'."""
    texts = [obj.name, obj.data.name if obj.data is not None else ""]
    texts += [slot.material.name for slot in obj.material_slots if slot.material is not None]
    for key in ("xiv_material", "original_material"):
        value = obj.get(key)
        if isinstance(value, str):
            texts.append(value)
    return texts


# -----------------------------------------------------------------------------
# Geometry

def bvh_of(points, tris):
    return BVHTree.FromPolygons(np.asarray(points).tolist(), np.asarray(tris).tolist(), all_triangles=True)


class Welded:
    """``points`` welded where they coincide (split seams): vertex -> ``node``, ``count`` nodes at
    ``points``, triangles over nodes (degenerate ones dropped) and the unique ``edges``."""

    def __init__(self, points, tris):
        self.node, self.count = fitting.weld_groups(np.asarray(points, dtype=np.float64), fitting.WELD_DISTANCE)
        self.points = np.zeros((self.count, 3))
        self.points[self.node] = points
        wt = self.node[np.asarray(tris, dtype=np.int64).reshape(-1, 3)]
        keep = (wt[:, 0] != wt[:, 1]) & (wt[:, 1] != wt[:, 2]) & (wt[:, 2] != wt[:, 0])
        self.tris = wt[keep]
        edges = np.vstack([self.tris[:, [0, 1]], self.tris[:, [1, 2]], self.tris[:, [2, 0]]])
        self.edges = np.unique(np.sort(edges, axis=1), axis=0) if len(edges) else np.zeros((0, 2), dtype=np.int64)

    def normals(self):
        n = np.zeros((self.count, 3))
        if len(self.tris):
            p = self.points
            fn = np.cross(p[self.tris[:, 1]] - p[self.tris[:, 0]], p[self.tris[:, 2]] - p[self.tris[:, 0]])
            for k in range(3):
                np.add.at(n, self.tris[:, k], fn)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def vertex_normals(points, tris):
    """Normals of ``points`` over the mesh welded at split seams, so split copies agree."""
    welded = Welded(points, tris)
    return welded.normals()[welded.node]


def boundary_loops(welded):
    """Open boundary loops of a welded mesh, as lists of nodes."""
    t = welded.tris
    if not len(t):
        return []
    directed = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _unique, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    border = directed[counts[inverse.reshape(-1)] == 1]
    following = {}
    for a, b in border.tolist():
        following.setdefault(a, []).append(b)
    loops = []
    seen = set()
    for start in list(following):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        node = start
        while True:
            options = [b for b in following.get(node, ()) if b not in seen]
            if not options:
                break
            node = options[0]
            seen.add(node)
            loop.append(node)
        loops.append(loop)
    return loops


def neck_ring(points, tris, *, minimum=6):
    """The neck ring, the lowest open boundary loop of the skin: (welded mesh, its nodes, every vertex on
    it), or None."""
    welded = Welded(points, tris)
    loops = [loop for loop in boundary_loops(welded) if len(loop) >= minimum]
    if not loops:
        return None
    loop = min(loops, key=lambda nodes: welded.points[nodes, 2].mean())
    verts = np.flatnonzero(np.isin(welded.node, loop))
    return welded, np.array(loop), verts


def loose_parts(points, tris):
    """Loose part of each vertex (numbered from 0) with split seams welded, and the count."""
    welded = Welded(points, tris)
    label = hairing.components(welded.count, welded.edges)
    part = label[welded.node]
    return part, int(part.max()) + 1 if len(part) else 0


def barycentric(points, tris, tri_index, locations):
    """Barycentric coordinates (n, 3) of ``locations`` on triangles ``tri_index`` of ``points``/``tris``."""
    t = tris[tri_index]
    a, b, c = points[t[:, 0]], points[t[:, 1]], points[t[:, 2]]
    v0, v1, v2 = b - a, c - a, locations - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) > 1e-30, den, 1e-30)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bc = np.clip(np.stack([1.0 - v - w, v, w], axis=1), 0.0, 1.0)
    return bc / np.maximum(bc.sum(1, keepdims=True), 1e-12)


def interpolate(values, tris, tri_index, bary):
    """Per-vertex ``values`` interpolated at points given by triangles and barycentric coordinates."""
    t = tris[tri_index]
    return values[t[:, 0]] * bary[:, 0:1] + values[t[:, 1]] * bary[:, 1:2] + values[t[:, 2]] * bary[:, 2:3]


def nearest_on(bvh, points):
    """(triangle index, location, distance) of the closest surface point to each of ``points``."""
    tri = np.zeros(len(points), dtype=np.int64)
    loc = np.zeros((len(points), 3))
    dist = np.zeros(len(points))
    for i, p in enumerate(points):
        location, _normal, index, distance = bvh.find_nearest(Vector(p))
        tri[i] = index if index is not None else 0
        loc[i] = location if location is not None else p
        dist[i] = distance if distance is not None else np.inf
    return tri, loc, dist


def smooth_polyline(poly, passes=2):
    p = np.asarray(poly, dtype=np.float64).copy()
    for _ in range(passes):
        if len(p) < 3:
            break
        q = p.copy()
        q[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
        p = q
    return p


def resample(poly, count):
    """``count`` points spread evenly along a polyline."""
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 2:
        return np.repeat(poly[:1], count, axis=0)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, cum[-1], count)
    return np.stack([np.interp(t, cum, poly[:, k]) for k in range(3)], axis=1)


def polyline_parameter(poly, points):
    """For each of ``points``: (position along the polyline, 0..1 by length; distance to it; closest point)."""
    poly = np.asarray(poly, dtype=np.float64)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    lengths = np.linalg.norm(ab, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    total = max(cum[-1], 1e-12)
    points = np.asarray(points, dtype=np.float64)
    s = np.zeros(len(points))
    d = np.full(len(points), np.inf)
    q = np.zeros((len(points), 3))
    for start in range(0, len(points), 4096):
        p = points[start:start + 4096]
        t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(2) / np.maximum((ab * ab).sum(1), 1e-18)[None], 0, 1)
        c = a[None] + t[:, :, None] * ab[None]
        dist = np.linalg.norm(c - p[:, None, :], axis=2)
        k = dist.argmin(1)
        rows = np.arange(len(p))
        s[start:start + len(p)] = (cum[k] + t[rows, k] * lengths[k]) / total
        d[start:start + len(p)] = dist[rows, k]
        q[start:start + len(p)] = c[rows, k]
    return s, d, q


def polyline_z(poly, x):
    order = np.argsort(poly[:, 0])
    return np.interp(x, poly[order, 0], poly[order, 2])


# -----------------------------------------------------------------------------
# Features

class Eye:
    """What an eye's opening is measured against: the eyeball, as a mesh (``points``/``tris``) or a sphere
    of ``radius`` about ``center`` (the eye bone), and whether it's the face's own (``found``)."""

    def __init__(self, center, radius, points=None, tris=None, found=False):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.points = None if points is None else np.asarray(points, dtype=np.float64)
        self.tris = None if tris is None else np.asarray(tris, dtype=np.int64)
        self.bvh = bvh_of(self.points, self.tris) if self.points is not None and len(self.tris) else None
        self.found = found

    def moved(self, matrix):
        """This eyeball moved by ``matrix`` (4x4)."""
        center = matrix[:3, :3] @ self.center + matrix[:3, 3]
        if self.bvh is None:
            return Eye(center, self.radius, found=self.found)
        return Eye(center, self.radius, self.points @ matrix[:3, :3].T + matrix[:3, 3], self.tris, self.found)


def eyeball_front(eye, origin, direction):
    """Distance along a ray from ``origin`` in unit ``direction`` to the eyeball, or None when it misses."""
    if eye.bvh is not None:
        hit = eye.bvh.ray_cast(Vector(origin), Vector(direction), 1.0)
        return None if hit[0] is None else hit[3]
    offset = origin - eye.center
    b = float(offset @ direction)
    disc = b * b - (float(offset @ offset) - eye.radius * eye.radius)
    return None if disc <= 0.0 else -b - math.sqrt(disc)


def eye_opening(skin_bvh, eye, step=SCAN_STEP, tilt=0.0, shell=None, box=None):
    """Where the eyeball shows through the skin, seen from the front (tilted ``tilt`` radians upward; down
    for negative ones): (xs, zs, hits (nz, nx, 3) on the skin or nan, open mask), xs and zs across and up
    the view about the eye's center, the mask the connected region around it. With ``shell``, only skin
    that close to the eyeball's sphere hides it (the lids, not the brows or cheeks in front of them). With
    ``box`` (x0, x1, z0, z1, as xs and zs), rays are cast only within it and the rest of the window counts as
    covered (hits nan), unless the opening found reaches the box's edge: then the whole window is scanned."""
    c = eye.center
    half_w = max(EYE_WINDOW[0], 1.3 * eye.radius)
    half_h = max(EYE_WINDOW[1], 1.0 * eye.radius)
    view = np.array((0.0, math.cos(tilt), math.sin(tilt)))
    across = np.array((1.0, 0.0, 0.0))
    up = np.array((0.0, -math.sin(tilt), math.cos(tilt)))
    xs = np.arange(c[0] - half_w, c[0] + half_w, step)
    zs = np.arange(c[2] - half_h, c[2] + half_h, step)
    hits = np.full((len(zs), len(xs), 3), np.nan)
    shows = np.zeros((len(zs), len(xs)), dtype=bool)
    # The rows and columns scanned: the whole window's grid, or its part within the box.
    i0, i1, j0, j1 = 0, len(zs), 0, len(xs)
    if box is not None:
        j0, j1 = int(np.searchsorted(xs, box[0])), int(np.searchsorted(xs, box[1], side="right"))
        i0, i1 = int(np.searchsorted(zs, box[2])), int(np.searchsorted(zs, box[3], side="right"))
    direction = Vector(view)
    for i, z in enumerate(zs[i0:i1], i0):
        for j, x in enumerate(xs[j0:j1], j0):
            origin = c + across * (x - c[0]) + up * (z - c[2]) - view * 0.3
            front = eyeball_front(eye, origin, view)
            hit = skin_bvh.ray_cast(Vector(origin), direction, 0.6)
            # With a shell, skin farther from the eyeball is passed through, as long as the ray comes out
            # of it again before the eyeball: it's in front of the lids (a brow), not around the eye.
            travelled = 0.0
            inside = None
            while shell is not None and hit[0] is not None and (
                    front is None or travelled + hit[3] < front) and (
                    abs((hit[0] - Vector(c)).length - eye.radius) > shell):
                inside = None if inside is not None else hit[0]
                travelled += hit[3] + 1e-5
                hit = skin_bvh.ray_cast(Vector(origin + view * travelled), direction, 0.6 - travelled)
            if inside is not None:
                hits[i, j] = inside
            elif hit[0] is not None:
                hits[i, j] = hit[0]
            if front is not None:
                shows[i, j] = inside is None and (hit[0] is None or travelled + hit[3] > front + EYEBALL_TOLERANCE)
    mask = np.zeros_like(shows)
    rows, cols = np.nonzero(shows)
    if len(rows):
        ci = np.argmin(np.abs(zs - c[2]))
        cj = np.argmin(np.abs(xs - c[0]))
        k = np.argmin((rows - ci) ** 2 + (cols - cj) ** 2)
        stack = [(rows[k], cols[k])]
        mask[rows[k], cols[k]] = True
        while stack:
            a, b = stack.pop()
            for na, nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if 0 <= na < shows.shape[0] and 0 <= nb < shows.shape[1] and shows[na, nb] and not mask[na, nb]:
                    mask[na, nb] = True
                    stack.append((na, nb))
    if box is not None:
        # An opening reaching the box's edge (inside the window) may go on beyond it.
        rows, cols = np.nonzero(mask)
        if len(rows) and ((i0 > 0 and rows.min() == i0) or (i1 < len(zs) and rows.max() == i1 - 1) or
                          (j0 > 0 and cols.min() == j0) or (j1 < len(xs) and cols.max() == j1 - 1)):
            return eye_opening(skin_bvh, eye, step, tilt, shell)
    return xs, zs, hits, mask


def lid_margins(xs, zs, hits, mask, side):
    """The upper and lower lid margins around an opening: skin points just above and below it in each
    column, ordered from the inner corner (by the nose) to the outer one. None when there's no opening."""
    upper, lower = [], []
    for j in np.flatnonzero(mask.any(0)):
        rows = np.flatnonzero(mask[:, j])
        top = hits[min(rows.max() + 1, len(zs) - 1), j]
        bottom = hits[max(rows.min() - 1, 0), j]
        if np.isnan(top).any() or np.isnan(bottom).any():
            continue
        upper.append(top)
        lower.append(bottom)
    if len(upper) < 3:
        return None
    upper, lower = np.array(upper), np.array(lower)
    if side == "r":  # columns run along +X; the right eye's inner corner is its largest X
        upper, lower = upper[::-1], lower[::-1]
    return smooth_polyline(upper), smooth_polyline(lower)


def _box_blur(a, k):
    pad = k // 2
    b = np.pad(a, pad, mode="edge")
    c = np.cumsum(np.cumsum(b, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def lip_line(skin_bvh, center, step=SCAN_STEP):
    """The lip line seen from the front: skin points along the deepest crease between the lips, ordered
    from -X to +X, or None when there's no crease around ``center`` (between the lip bones)."""
    xs = np.arange(center[0] - MOUTH_WINDOW[0], center[0] + MOUTH_WINDOW[0], step)
    zs = np.arange(center[2] - MOUTH_WINDOW[1], center[2] + MOUTH_WINDOW[1], step)
    hits = np.full((len(zs), len(xs), 3), np.nan)
    y0 = center[1] - 0.3
    direction = Vector((0.0, 1.0, 0.0))
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            hit = skin_bvh.ray_cast(Vector((x, y0, z)), direction, 0.6)
            if hit[0] is not None:
                hits[i, j] = hit[0]
    depth = hits[:, :, 1]
    if not np.isfinite(depth).any():
        return None
    d = np.where(np.isnan(depth), np.nanmax(depth) + 0.02, depth)
    k = max(3, int(round(LIP_BLUR / step)) | 1)
    local = d - _box_blur(d, k)
    band = np.flatnonzero(np.abs(zs - center[2]) < LIP_SEARCH)
    if not len(band):
        return None
    rows = band[np.argmax(local[band], axis=0)]
    power = local[rows, np.arange(len(xs))]
    c = int(np.argmin(np.abs(xs - center[0])))
    if power[c] <= LIP_DEPTH:
        return None
    max_gap = int(round(LIP_GAP / step))

    def walk(direction_):
        end = c
        gap = 0
        j = c
        while 0 <= j + direction_ < len(xs):
            j += direction_
            if power[j] > LIP_DEPTH:
                end, gap = j, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
        return end

    cols = np.arange(walk(-1), walk(1) + 1)
    z = zs[rows[cols]]
    z = np.array([np.median(z[max(0, i - 2):i + 3]) for i in range(len(z))])
    points = []
    for j, zz in zip(cols, z):
        p = hits[int(np.argmin(np.abs(zs - zz))), j]
        if not np.isnan(p).any():
            points.append(p)
    return smooth_polyline(np.array(points)) if len(points) >= 5 else None


class Features:
    """A face skin's lid margins (``eyes``: side -> (upper, lower, Eye)), lip line (``lip``), mouth center and
    neck ring (``ring``: its welded positions), found on ``points``/``tris`` in the reference's model space.
    ``eyes_in`` gives the eyeballs to measure the openings against (side -> Eye)."""

    def __init__(self, points, tris, skeleton, eyes_in):
        self.points = np.asarray(points, dtype=np.float64)
        self.tris = np.asarray(tris, dtype=np.int64)
        self.bvh = bvh_of(self.points, self.tris)
        self.skeleton = skeleton
        self.eyes = {}
        for side, eye in eyes_in.items():
            xs, zs, hits, mask = eye_opening(self.bvh, eye)
            margins = lid_margins(xs, zs, hits, mask, side)
            if margins is not None:
                self.eyes[side] = (margins[0], margins[1], eye)
        self.lip = None
        self.mouth_center = None
        if all(skeleton.has(name) for name in LIP_BONES):
            self.mouth_center = np.mean([skeleton.head(name) for name in LIP_BONES], axis=0)
            self.lip = lip_line(self.bvh, self.mouth_center)
        ring = neck_ring(self.points, self.tris)
        self.ring = ring[0].points[ring[1]] if ring is not None else None
        self._lash_lines = {}

    def lash_line(self, side, which):
        """The lash line of eye ``side``'s upper (``which`` 0) or lower (1) lid: the lid's front edge, where
        lashes grow, found as its silhouette against the eyeball seen from above (below for the lower lid),
        LASH_TILT off the front. Ordered like the margins; the margin when there's none."""
        key = (side, which)
        if key not in self._lash_lines:
            upper, lower, eye = self.eyes[side]
            xs, zs, hits, mask = eye_opening(self.bvh, eye, step=LASH_SCAN_STEP,
                                             tilt=-LASH_TILT if which == 0 else LASH_TILT, shell=LID_SHELL)
            found = lid_margins(xs, zs, hits, mask, side)
            margin = (upper, lower)[which]
            line = found[which] if found is not None else margin
            if np.median(polyline_parameter(margin, line)[1]) > LASH_LINE_LIMIT:
                line = margin
            self._lash_lines[key] = line
        return self._lash_lines[key]

    def opening_box(self, side):
        """The box (x0, x1, z0, z1) around eye ``side``'s opening, OPENING_MARGIN beyond its lid margins: all
        that closure checks scan (see `eye_opening`)."""
        upper, lower, _eye = self.eyes[side]
        margins = np.vstack([upper, lower])
        low = margins.min(0) - OPENING_MARGIN
        high = margins.max(0) + OPENING_MARGIN
        return low[0], high[0], low[2], high[2]

    def summary(self):
        return {"eyes": sorted(self.eyes), "lip": self.lip is not None, "ring": self.ring is not None}


# -----------------------------------------------------------------------------
# Warp

def landmark_pairs(src, dst):
    """Matching feature samples of two faces' Features: (src points, dst points)."""
    a, b = [], []
    for side in SIDES:
        if side in src.eyes and side in dst.eyes:
            for k in (0, 1):
                a.append(resample(src.eyes[side][k], MARGIN_SAMPLES))
                b.append(resample(dst.eyes[side][k], MARGIN_SAMPLES))
    if src.lip is not None and dst.lip is not None:
        a.append(resample(src.lip, LIP_SAMPLES))
        b.append(resample(dst.lip, LIP_SAMPLES))
    if src.ring is not None and dst.ring is not None:
        d = np.linalg.norm(src.ring[:, None] - dst.ring[None], axis=2)
        close = d.min(1) < 0.01
        a.append(src.ring[close])
        b.append(dst.ring[d.argmin(1)[close]])
    if not a:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack(a), np.vstack(b)


def anchor_pairs(src, dst_points, feature_points):
    """Pairs pinning what's away from the features: points of the destination face far from them, and
    the closest points of the source face."""
    pts = np.asarray(dst_points, dtype=np.float64)
    if len(feature_points):
        far = np.ones(len(pts), dtype=bool)
        for start in range(0, len(pts), 2048):
            p = pts[start:start + 2048]
            far[start:start + 2048] = np.min(np.linalg.norm(p[:, None] - feature_points[None], axis=2),
                                             axis=1) > ANCHOR_CLEARANCE
        pts = pts[far]
    if not len(pts):
        return np.zeros((0, 3)), np.zeros((0, 3))
    key = np.floor(pts / ANCHOR_SPACING).astype(np.int64)
    _unique, first = np.unique(key, axis=0, return_index=True)
    pts = pts[np.sort(first)]
    _tri, loc, dist = nearest_on(src.bvh, pts)
    keep = dist < ANCHOR_MAX_GAP
    return loc[keep], pts[keep]


class Warp:
    """A thin plate spline (3D, kernel r) taking ``src`` points (nearly) exactly to ``dst``."""

    def __init__(self, src, dst):
        self.src = np.asarray(src, dtype=np.float64)
        n = len(self.src)
        if n < 4:
            self.src = np.zeros((0, 3))
            self.w = np.zeros((0, 3))
            self.a = np.vstack([np.zeros(3), np.eye(3)])
            return
        r = np.linalg.norm(self.src[:, None] - self.src[None], axis=2)
        P = np.hstack([np.ones((n, 1)), self.src])
        A = np.zeros((n + 4, n + 4))
        A[:n, :n] = r + WARP_REGULARIZATION * np.eye(n)
        A[:n, n:] = P
        A[n:, :n] = P.T
        rhs = np.zeros((n + 4, 3))
        rhs[:n] = dst
        solution = np.linalg.lstsq(A, rhs, rcond=None)[0]
        self.w = solution[:n]
        self.a = solution[n:]

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.empty((len(points), 3))
        for start in range(0, len(points), 2048):
            p = points[start:start + 2048]
            moved = np.hstack([np.ones((len(p), 1)), p]) @ self.a
            if len(self.src):
                moved += np.linalg.norm(p[:, None] - self.src[None], axis=2) @ self.w
            out[start:start + 2048] = moved
        return out


def face_warp(src, dst, dst_points):
    """The Warp taking face ``src`` (Features) onto face ``dst`` (Features of ``dst_points``), and how many
    feature landmarks it has."""
    a, b = landmark_pairs(src, dst)
    ea, eb = anchor_pairs(src, dst_points, b)
    return Warp(np.vstack([a, ea]), np.vstack([b, eb])), len(a)


# -----------------------------------------------------------------------------
# Sides

def weight_labels(names, weights):
    """Side labels of the game face's vertices, from which side's bones weigh more."""
    lab = np.zeros(len(weights), dtype=np.int64)

    def family(test):
        cols = [i for i, name in enumerate(names) if test(name)]
        return weights[:, cols].sum(1) if cols else np.zeros(len(weights))

    for side in SIDES:
        up_label, down_label = EYE_LABELS[side]
        up = family(lambda n: n.startswith(UPPER_LID) and n.endswith("_" + side))
        down = family(lambda n: n.startswith(LOWER_LID) and n.endswith("_" + side))
        lab[(up > down) & (up > SIDE_THRESHOLD)] = up_label
        lab[(down > up) & (down > SIDE_THRESHOLD)] = down_label
    up = family(lambda n: n.startswith(UPPER_MOUTH))
    down = family(lambda n: n.startswith(LOWER_MOUTH))
    free = lab == NEUTRAL
    lab[free & (up > down) & (up > SIDE_THRESHOLD)] = MOUTH_UPPER
    lab[free & (down > up) & (down > SIDE_THRESHOLD)] = MOUTH_LOWER
    return lab


def visible_from_front(bvh, points, tolerance=0.0004):
    visible = np.zeros(len(points), dtype=bool)
    direction = Vector((0.0, 1.0, 0.0))
    for i, p in enumerate(points):
        hit = bvh.ray_cast(Vector((p[0], p[1] - 0.3, p[2])), direction, 0.6)
        visible[i] = hit[0] is not None and abs(hit[0][1] - p[1]) < tolerance
    return visible


def spread_labels(points, edges, labels, region, reach):
    """Unlabelled nodes in ``region`` take the label of the closest labelled node along ``edges``, when
    it's at most ``reach`` away."""
    labels = labels.copy()
    if not region.any() or not len(edges):
        return labels
    touching = region[edges[:, 0]] | region[edges[:, 1]]
    a, b = edges[touching, 0], edges[touching, 1]
    length = np.linalg.norm(points[a] - points[b], axis=1)
    fixed = labels > NEUTRAL
    dist = np.where(fixed, 0.0, np.inf)
    for _ in range(1000):
        changed = False
        for x, y in ((a, b), (b, a)):
            candidate = dist[x] + length
            better = (candidate < dist[y]) & (candidate <= reach) & region[y] & ~fixed[y]
            if better.any():
                order = np.argsort(-candidate[better])
                targets = y[better][order]
                dist[targets] = candidate[better][order]
                labels[targets] = labels[x[better][order]]
                changed = True
        if not changed:
            break
    return labels


def geometric_labels(points, tris, features):
    """Side labels of a custom face's vertices from where they lie around its features: on the visible
    surface within SIDE_BAND of a margin or the lip line, and spread from there into the eye sockets and
    the lips' inside."""
    welded = Welded(points, tris)
    wpts = welded.points
    count = welded.count
    labels = np.zeros(count, dtype=np.int64)
    if not count:
        return labels[welded.node]
    edges = welded.edges
    neighbour_z = np.zeros(count)
    neighbours = np.zeros(count)
    for a, b in ((0, 1), (1, 0)):
        np.add.at(neighbour_z, edges[:, a], wpts[edges[:, b], 2])
        np.add.at(neighbours, edges[:, a], 1.0)
    neighbour_z /= np.maximum(neighbours, 1.0)
    near_any = np.zeros(count, dtype=bool)
    for _upper, _lower, eye in features.eyes.values():
        near_any |= np.linalg.norm(wpts - eye.center, axis=1) < 0.03
    if features.lip is not None:
        near_any |= np.linalg.norm(wpts - features.mouth_center, axis=1) < 0.05
    visible = np.zeros(count, dtype=bool)
    candidates = np.flatnonzero(near_any)
    visible[candidates] = visible_from_front(features.bvh, wpts[candidates])
    for side, (upper, lower, eye) in features.eyes.items():
        up_label, down_label = EYE_LABELS[side]
        near = np.flatnonzero(np.linalg.norm(wpts - eye.center, axis=1) < 0.03)
        x = wpts[near, 0]
        z = wpts[near, 2]
        x0 = min(upper[:, 0].min(), lower[:, 0].min())
        x1 = max(upper[:, 0].max(), lower[:, 0].max())
        inside = (x >= x0) & (x <= x1)
        zu = polyline_z(upper, x)
        zl = polyline_z(lower, x)
        vis = visible[near]
        labels[near[vis & inside & (z >= zu - 0.0002) & (z < zu + SIDE_BAND)]] = up_label
        labels[near[vis & inside & (z <= zl + 0.0002) & (z > zl - SIDE_BAND)]] = down_label
        region = np.zeros(count, dtype=bool)
        region[near[~vis]] = True
        labels = spread_labels(wpts, edges, labels, region, SOCKET_REACH)
    if features.lip is not None:
        lip = features.lip
        order = np.argsort(lip[:, 0])
        x = wpts[:, 0]
        x0, x1 = lip[:, 0].min(), lip[:, 0].max()
        zc = np.interp(x, lip[order, 0], lip[order, 2])
        yc = np.interp(x, lip[order, 0], lip[order, 1])
        mouth = ((x > x0 - 0.002) & (x < x1 + 0.002) & (np.abs(wpts[:, 2] - zc) < 0.02) &
                 (wpts[:, 1] > yc - 0.02) & (wpts[:, 1] < yc + 0.04) & (labels == NEUTRAL))
        idx = np.flatnonzero(mouth)
        s = wpts[idx, 2] - zc[idx]
        close = np.abs(s) < 0.0004
        s[close] = neighbour_z[idx][close] - zc[idx][close]
        outside = (visible[idx] & (wpts[idx, 1] < yc[idx] + 0.002) & (np.abs(s) < SIDE_BAND) &
                   (x[idx] > x0 + CORNER_GAP) & (x[idx] < x1 - CORNER_GAP))
        labels[idx[outside & (s > 0)]] = MOUTH_UPPER
        labels[idx[outside & (s <= 0)]] = MOUTH_LOWER
        region = np.zeros(count, dtype=bool)
        region[idx[~outside]] = True
        region &= ~visible | (wpts[:, 1] >= yc + 0.002)
        labels = spread_labels(wpts, edges, labels, region, LIP_REACH)
    return labels[welded.node]


def point_labels(points, features):
    """Side labels of loose points (not a mesh) around the features: above or below margins and the lip line."""
    labels = np.zeros(len(points), dtype=np.int64)
    for side, (upper, lower, eye) in features.eyes.items():
        up_label, down_label = EYE_LABELS[side]
        x = points[:, 0]
        x0 = min(upper[:, 0].min(), lower[:, 0].min())
        x1 = max(upper[:, 0].max(), lower[:, 0].max())
        inside = (np.linalg.norm(points - eye.center, axis=1) < 0.03) & (x >= x0) & (x <= x1)
        zu = polyline_z(upper, x)
        zl = polyline_z(lower, x)
        labels[inside & (points[:, 2] >= zu - 0.0002) & (points[:, 2] < zu + SIDE_BAND)] = up_label
        labels[inside & (points[:, 2] <= zl + 0.0002) & (points[:, 2] > zl - SIDE_BAND)] = down_label
    if features.lip is not None:
        lip = features.lip
        x = points[:, 0]
        inside = (x > lip[:, 0].min() + CORNER_GAP) & (x < lip[:, 0].max() - CORNER_GAP) & (labels == NEUTRAL)
        zc = polyline_z(lip, x)
        near = inside & (np.abs(points[:, 2] - zc) < SIDE_BAND)
        labels[near & (points[:, 2] > zc)] = MOUTH_UPPER
        labels[near & (points[:, 2] <= zc)] = MOUTH_LOWER
    return labels


class SidedTransfer:
    """Weights of the closest point of a face's surface, never from the other side of a feature (see the
    side labels) and preferably from a surface facing the same way."""

    def __init__(self, points, tris, weights, labels):
        self.points = np.asarray(points, dtype=np.float64)
        self.tris = np.asarray(tris, dtype=np.int64)
        self.weights = weights
        fn = np.cross(self.points[self.tris[:, 1]] - self.points[self.tris[:, 0]],
                      self.points[self.tris[:, 2]] - self.points[self.tris[:, 0]])
        self.tri_normals = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        tri_labels = labels[self.tris]
        self.trees = {NEUTRAL: (bvh_of(self.points, self.tris), np.arange(len(self.tris)))}
        for label, opposite in OPPOSITE.items():
            banned = np.any(tri_labels == opposite, axis=1) & ~np.any(tri_labels == label, axis=1)
            idx = np.flatnonzero(~banned)
            if len(idx):
                self.trees[label] = (bvh_of(self.points, self.tris[idx]), idx)

    def lookup(self, points, labels=None, normals=None):
        """(triangle, barycentric coordinates) of the chosen surface point for each of ``points``."""
        count = len(points)
        tri = np.zeros(count, dtype=np.int64)
        loc = np.zeros((count, 3))
        for i in range(count):
            label = int(labels[i]) if labels is not None else NEUTRAL
            tree, idx = self.trees.get(label, self.trees[NEUTRAL])
            v = Vector(points[i])
            location, _normal, t, distance = tree.find_nearest(v)
            if location is None:
                tree, idx = self.trees[NEUTRAL]
                location, _normal, t, distance = tree.find_nearest(v)
            if normals is not None:
                n = normals[i]
                facing = float(np.dot(self.tri_normals[idx[t]], n))
                if facing < NORMAL_GOOD:
                    best = distance + NORMAL_PENALTY * (1.0 - facing)
                    for location2, _normal2, t2, distance2 in tree.find_nearest_range(v, distance + NORMAL_SEARCH):
                        cost = distance2 + NORMAL_PENALTY * (1.0 - float(np.dot(self.tri_normals[idx[t2]], n)))
                        if cost < best:
                            best, location, t = cost, location2, t2
            tri[i] = idx[t]
            loc[i] = location
        return tri, barycentric(self.points, self.tris, tri, loc)

    def weights_at(self, points, labels=None, normals=None):
        tri, bary = self.lookup(np.asarray(points, dtype=np.float64).reshape(-1, 3), labels, normals)
        return interpolate(self.weights, self.tris, tri, bary)


# -----------------------------------------------------------------------------
# Skin weights

def lid_columns(names, side):
    up = [i for i, n in enumerate(names) if n.startswith(UPPER_LID) and n.endswith("_" + side)]
    down = [i for i, n in enumerate(names) if n.startswith(LOWER_LID) and n.endswith("_" + side)]
    return up, down


def skin_points(points, names, weights, deformations):
    """``points`` moved by bone ``weights`` (columns ``names``) with bone ``deformations`` ({name: 4x4})."""
    out = np.zeros_like(points)
    total = np.zeros(len(points))
    for column, name in enumerate(names):
        w = weights[:, column]
        m = deformations.get(name)
        if m is None or not w.any():
            continue
        out += w[:, None] * (points @ m[:3, :3].T + m[:3, 3])
        total += w
    out += np.clip(1.0 - total, 0.0, None)[:, None] * points
    return out


def _scale_lid(weights, verts, columns, factor, head=None):
    """Scale the lid bones' share of ``verts`` by ``factor`` (per vertex), up to all of it; the vertex's
    other bones make room, or take what's freed (column ``head`` when the lid bones were all it had)."""
    if not len(verts) or not columns:
        return
    w = weights[verts]
    total = w.sum(1)
    lid = w[:, columns].sum(1)
    new_lid = np.minimum(total, lid * factor)
    other = total - lid
    keep = np.where(other > 1e-9, (total - new_lid) / np.maximum(other, 1e-9), 0.0)
    grow = np.where(lid > 1e-9, new_lid / np.maximum(lid, 1e-9), 0.0)
    out = w * keep[:, None]
    out[:, columns] = w[:, columns] * grow[:, None]
    if head is not None:
        bare = other <= 1e-9
        out[bare, head] += total[bare] - new_lid[bare]
    weights[verts] = out


def open_area(skin_bvh, eye):
    """How much of the eyeball shows through the skin seen from the front, in mm²."""
    xs, zs, _hits, mask = eye_opening(skin_bvh, eye)
    return float(mask.sum() * (xs[1] - xs[0]) * (zs[1] - zs[0]) * 1e6)


def blink_opening(points, tris, names, weights, eye, side, skeleton, share, step=SCAN_STEP, box=None):
    """`eye_opening` of eyeball ``eye`` (side ``side``) through the skin (``points``/``tris``, ``weights`` over
    ``names``) at ``share`` of the game's blink, with rays ``step`` apart, within ``box`` when given."""
    deform = skeleton.deformations('BLINK', share)
    posed = skin_points(points, names, weights, deform)
    return eye_opening(bvh_of(posed, tris), eye.moved(deform.get(EYEBALL_BONE.format(side), np.eye(4))), step,
                       box=box)


def closing_share(points, tris, names, weights, eyes, skeleton, boxes=None):
    """The share of the game's blink (see Skeleton.deformations) by which the skin (``points``/``tris``,
    ``weights`` over ``names``) covers all of the eyeballs ``eyes`` ({side: Eye}) seen from the front, to
    CLOSE_PRECISION; None when it doesn't even at the blink's peak. ``boxes`` ({side: box}, see
    `Features.opening_box`) limit the scans to the openings at rest."""
    boxes = boxes or {}

    def closed(share):
        return not any(blink_opening(points, tris, names, weights, eye, side, skeleton, share,
                                     box=boxes.get(side))[3].any()
                       for side, eye in eyes.items())

    if not eyes or not skeleton.deformations('BLINK') or not closed(1.0):
        return None
    low, high = 0.0, 1.0
    while high - low > CLOSE_PRECISION:
        middle = (low + high) / 2.0
        if closed(middle):
            high = middle
        else:
            low = middle
    return high


def close_eyes(points, tris, names, weights, features, labels, skeleton, share):
    """Make each eye of the skin (``points``/``tris``, weights over ``names``) close by ``share`` of the game's
    blink, the closing share of the game's face: where the eyeball still shows there, the lid weights are
    raised, fading out LID_FALLOFF from the margin; then, when the lids meet more than CLOSE_SLACK earlier, all
    of the eye's lid weights are lowered alike. Returns (weights, {side: report}), a report having the
    ``target`` share, how much of the eyeball showed there ``before`` and ``after`` (mm²), how far apart the lids
    stop where their weights can't go higher (``stuck``, mm), and for lids that met too early about the share
    they met at (``early``, else None) and what their weights were scaled by (``lowered``)."""
    weights = weights.copy()
    report = {}
    blink = skeleton.deformations('BLINK', share)
    head = names.index(HEAD) if HEAD in names else None
    for side, (upper, lower, eye) in features.eyes.items():
        up_cols, down_cols = lid_columns(names, side)
        m_up = blink.get(UPPER_LID + "01_" + side)
        m_down = blink.get(LOWER_LID + "01_" + side)
        if m_up is None or not up_cols:
            continue
        up_label, down_label = EYE_LABELS[side]
        inner = (upper[0] + lower[0]) / 2.0
        outer = (upper[-1] + lower[-1]) / 2.0
        near = np.linalg.norm(points - eye.center, axis=1) < 0.03
        span = outer[0] - inner[0]
        t = (points[:, 0] - inner[0]) / span if abs(span) > 1e-9 else np.zeros(len(points))
        above = points[:, 2] > inner[2] + t * (outer[2] - inner[2])
        up_verts = np.flatnonzero(near & ((above & (labels != down_label)) | (labels == up_label)))
        down_verts = np.flatnonzero(near & ((~above & (labels != up_label)) | (labels == down_label)))

        def falloff(verts, margin):
            d = np.abs(points[verts, 2] - polyline_z(margin, points[verts, 0]))
            return smoothstep(1.0 - d / LID_FALLOFF)

        up_fall = falloff(up_verts, upper)
        down_fall = falloff(down_verts, lower)
        order_u = np.argsort(upper[:, 0])
        order_l = np.argsort(lower[:, 0])
        box = features.opening_box(side)

        def opening(at, lid_weights=None, step=SCAN_STEP):
            return blink_opening(points, tris, names, weights if lid_weights is None else lid_weights, eye, side,
                                 skeleton, at, step, box)

        before = after = None
        stuck = 0.0
        for round_ in range(LID_ROUNDS + 1):
            xs, zs, _hits, mask = opening(share)
            area = float(mask.sum() * (xs[1] - xs[0]) * (zs[1] - zs[0]) * 1e6)
            before = area if before is None else before
            after = area
            if not mask.any() or round_ == LID_ROUNDS:
                break
            need_up = np.ones(len(xs))
            need_down = np.ones(len(xs))
            stuck = 0.0
            cols = np.flatnonzero(mask.any(0))
            xq = xs[cols]
            pu = np.stack([np.interp(xq, upper[order_u, 0], upper[order_u, k]) for k in range(3)], 1)
            pl = np.stack([np.interp(xq, lower[order_l, 0], lower[order_l, k]) for k in range(3)], 1)
            tri_u, loc_u, _d = nearest_on(features.bvh, pu)
            tri_l, loc_l, _d = nearest_on(features.bvh, pl)
            wu = interpolate(weights, tris, tri_u, barycentric(points, tris, tri_u, loc_u))
            wl = interpolate(weights, tris, tri_l, barycentric(points, tris, tri_l, loc_l))
            for k, j in enumerate(cols):
                rows = np.flatnonzero(mask[:, j])
                gap = (zs[rows.max()] - zs[rows.min()]) + (zs[1] - zs[0]) + LID_OVERLAP
                u = wu[k, up_cols].sum()
                d = wl[k, down_cols].sum() if down_cols else 0.0
                du = (m_up[:3, :3] @ pu[k] + m_up[:3, 3] - pu[k])[2]
                if du < -1e-9 and u < 1.0:
                    target = min(1.0, u + gap / -du)
                    need_up[j] = target / max(u, 0.05)
                    gap -= (target - u) * -du
                if gap > 0 and m_down is not None and down_cols and d < 1.0:
                    dd = (m_down[:3, :3] @ pl[k] + m_down[:3, 3] - pl[k])[2]
                    if dd > 1e-9:
                        target = min(1.0, d + gap / dd)
                        need_down[j] = target / max(d, 0.05)
                        gap -= (target - d) * dd
                stuck = max(stuck, gap)
            changed = False
            for need, verts, columns, fall in ((need_up, up_verts, up_cols, up_fall),
                                               (need_down, down_verts, down_cols, down_fall)):
                if np.all(need <= 1.0) or not len(verts):
                    continue
                pad = LID_SMOOTH // 2
                padded = np.pad(need, pad, mode="edge")
                widest = np.array([padded[i:i + LID_SMOOTH].max() for i in range(len(need))])
                smoothed = np.convolve(np.pad(widest, pad, mode="edge"), np.ones(LID_SMOOTH) / LID_SMOOTH,
                                       mode="valid")
                factor = np.interp(points[verts, 0], xs, np.maximum(smoothed, 1.0))
                previous = weights[verts][:, columns].sum(1)
                _scale_lid(weights, verts, columns, 1.0 + (factor - 1.0) * fall)
                changed |= bool(np.abs(weights[verts][:, columns].sum(1) - previous).max() > 1e-4)
            if not changed:
                break
        # Lids meeting much earlier slide on past each other: all of the eye's lid weights are scaled by the
        # least that still closes it CLOSE_SLACK / 2 before the target (bisection with rays twice as far apart,
        # then checked at the target). A lid's travel goes with its weight, so they met at about that share
        # times the scale.
        early = None
        lowered = 1.0
        aim = share - CLOSE_SLACK
        if after <= 0.0 and aim > 0.0 and not opening(aim)[3].any():
            lids = np.flatnonzero(near)
            columns = up_cols + down_cols

            def scaled(factor):
                out = weights.copy()
                _scale_lid(out, lids, columns, factor, head)
                return out

            low, high = 0.0, 1.0
            while high - low > CLOSE_PRECISION:
                middle = (low + high) / 2.0
                if opening(share - CLOSE_SLACK / 2.0, scaled(middle), 2.0 * SCAN_STEP)[3].any():
                    low = middle
                else:
                    high = middle
            while high < 1.0 and opening(share, scaled(high))[3].any():
                high = min(1.0, high * 1.05)
            if high < 1.0:
                weights = scaled(high)
                lowered = high
                early = (share - CLOSE_SLACK / 2.0) * high
        report[side] = {"target": share, "before": before, "after": after,
                        "stuck": stuck * 1000.0 if after > 0.0 else 0.0, "early": early, "lowered": lowered}
    return weights, report


def limit_weights(weights, max_groups):
    """At most ``max_groups`` weights per vertex, none below MIN_WEIGHT, adding up to what they did."""
    total = weights.sum(1, keepdims=True)
    weights = np.where(weights >= MIN_WEIGHT, weights, 0.0)
    if weights.shape[1] > max_groups:
        order = np.argsort(-weights, axis=1)
        drop = order[:, max_groups:]
        rows = np.repeat(np.arange(len(weights)), drop.shape[1])
        weights[rows, drop.ravel()] = 0.0
    kept = weights.sum(1, keepdims=True)
    return np.where(kept > 1e-9, weights * (total / np.maximum(kept, 1e-9)), weights)


class FaceFit:
    """A custom face skin (``points``/``tris`` in the reference's model space, eyeballs ``eyes``) fitted to
    the game's face ``reference``: its ``features``, side ``labels`` and the ``warp`` onto the game's face."""

    def __init__(self, points, tris, reference, eyes):
        self.points = np.asarray(points, dtype=np.float64)
        self.tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        self.reference = reference
        self.features = Features(self.points, self.tris, reference.skeleton, eyes)
        self.warp, self.landmarks = face_warp(self.features, reference.features(), reference.points)
        self.labels = geometric_labels(self.points, self.tris, self.features)
        self.warped = self.warp(self.points)

    def reference_weights(self, points=None, labels=None):
        """The game face's weights at ``points`` (default: the skin's vertices), warped onto it."""
        if points is None:
            return self.reference.transfer().weights_at(self.warped, self.labels,
                                                        vertex_normals(self.warped, self.tris))
        return self.reference.transfer().weights_at(self.warp(points), labels, None)


def _ring_weights(points, tris, weights, ring):
    """The neck ring's vertices get the game's ring weights (those of the closest game ring vertex)."""
    found = neck_ring(points, tris)
    if found is None:
        return weights
    ring_points, _normals, ring_weights = ring
    verts = found[2]
    d = np.linalg.norm(points[verts][:, None] - ring_points[None], axis=2)
    close = d.min(1) < RING_TOLERANCE
    weights = weights.copy()
    weights[verts[close]] = ring_weights[d.argmin(1)[close]]
    return weights


def skin_weights(fit, *, close=True, max_groups=DEFAULT_MAX_GROUPS):
    """Weights for a face skin from the game's face (see FaceFit): (names, weights, {side: eye report})."""
    reference = fit.reference
    weights = fit.reference_weights()
    names = list(reference.names)
    eyes = {}
    ring = reference.ring()
    if ring is not None and fit.features.ring is not None:
        weights = _ring_weights(fit.points, fit.tris, weights, ring)
    if close and reference.skeleton.deformations('BLINK'):
        weights, eyes = close_eyes(fit.points, fit.tris, names, weights, fit.features, fit.labels, reference.skeleton,
                                   reference.closing_share())
    return names, limit_weights(weights, max_groups), eyes


# -----------------------------------------------------------------------------
# Neck

class NeckMatch:
    """How to make a skin's neck ring the game's: for each ring vertex (``verts``), its new ``positions``,
    ``normals`` and ``weights`` (over ``names``), which were too far off to be the game's ring
    (``unmatched``) and how far each moves (``moved``)."""

    def __init__(self, verts, positions, normals, weights, names, unmatched, moved):
        self.verts = verts
        self.positions = positions
        self.normals = normals
        self.weights = weights
        self.names = names
        self.unmatched = unmatched
        self.moved = moved


def ring_polygon(ring_points):
    """The game's neck ring (``ring_points``, split copies included) as a closed polygon around its center."""
    node, count = fitting.weld_groups(np.asarray(ring_points, dtype=np.float64), fitting.WELD_DISTANCE)
    nodes = np.zeros((count, 3))
    nodes[node] = ring_points
    center = nodes.mean(0)
    order = np.argsort(np.arctan2(nodes[:, 1] - center[1], nodes[:, 0] - center[0]))
    return nodes[np.concatenate([order, order[:1]])]


def match_neck(points, tris, reference):
    """NeckMatch of the skin ``points``/``tris`` against the game's face, or None when the skin has no
    neck ring close to the game's. Ring vertices go onto the game's ring: onto its vertices where they are
    close to one, else onto its edges (with normals and weights in between)."""
    ring = reference.ring()
    found = neck_ring(points, tris)
    if ring is None or found is None:
        return None
    ring_points, ring_normals, ring_weights = ring
    _welded, _loop, verts = found
    node, count = fitting.weld_groups(ring_points, fitting.WELD_DISTANCE)
    nodes = np.zeros((count, 3))
    nodes[node] = ring_points
    center = nodes.mean(0)
    p = points[verts]
    if np.linalg.norm(p.mean(0) - center) > 0.01:
        return None
    node_normals = np.zeros((count, 3))
    np.add.at(node_normals, node, ring_normals)
    node_normals /= np.maximum(np.linalg.norm(node_normals, axis=1, keepdims=True), 1e-12)
    node_weights = np.zeros((count, ring_weights.shape[1]))
    np.add.at(node_weights, node, ring_weights)
    node_weights /= np.maximum(np.bincount(node, minlength=count).astype(np.float64), 1.0)[:, None]
    order = np.argsort(np.arctan2(nodes[:, 1] - center[1], nodes[:, 0] - center[0]))
    order = np.concatenate([order, order[:1]])
    polygon = nodes[order]
    s, d, _q = polyline_parameter(polygon, p)
    lengths = np.linalg.norm(np.diff(polygon, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)]) / max(lengths.sum(), 1e-12)
    positions = np.zeros((len(verts), 3))
    normals = np.zeros((len(verts), 3))
    weights = np.zeros((len(verts), ring_weights.shape[1]))
    for i in range(len(verts)):
        k = int(np.clip(np.searchsorted(cum, s[i], side="right") - 1, 0, len(polygon) - 2))
        t = (s[i] - cum[k]) / max(cum[k + 1] - cum[k], 1e-12)
        if np.linalg.norm(p[i] - polygon[k]) < RING_TOLERANCE and t < 0.5:
            t = 0.0
        elif np.linalg.norm(p[i] - polygon[k + 1]) < RING_TOLERANCE and t >= 0.5:
            t = 1.0
        a, b = order[k], order[k + 1]
        positions[i] = nodes[a] * (1.0 - t) + nodes[b] * t
        n = node_normals[a] * (1.0 - t) + node_normals[b] * t
        normals[i] = n / max(np.linalg.norm(n), 1e-12)
        weights[i] = node_weights[a] * (1.0 - t) + node_weights[b] * t
    return NeckMatch(verts, positions, normals, weights, list(reference.names), d > RING_TOLERANCE,
                     np.linalg.norm(positions - p, axis=1))


def neck_weights(points, tris, reference):
    """The game's weights for the skin's neck: (vertices, weights over the reference's names) of the
    vertices whose closest game point has neck or upper back weight."""
    names = reference.names
    cols = [i for i, name in enumerate(names) if name in (NECK, UPPER_BACK)]
    if not cols:
        return np.zeros(0, dtype=np.int64), np.zeros((0, len(names)))
    top = reference.points[reference.weights[:, cols].sum(1) > 0, 2].max() + 0.005
    low = np.flatnonzero(points[:, 2] < top)
    normals = vertex_normals(points, tris)
    w = reference.transfer().weights_at(points[low], None, normals[low])
    has = w[:, cols].sum(1) > MIN_WEIGHT
    return low[has], w[has]


# -----------------------------------------------------------------------------
# Parts

PART_EYEBALL, PART_LASH, PART_FILM, PART_SHELL, PART_INSIDE, PART_EXTRA, PART_RIGID = (
    "EYEBALL", "LASH", "FILM", "SHELL", "INSIDE", "EXTRA", "RIGID")


def opening_share(points, features, side):
    """The share of ``points`` that lie within eye ``side``'s opening seen from the front (between its
    margins): a tear film or eye shadow over the eyeball rather than lashes on a lid."""
    upper, lower, _eye = features.eyes[side]
    x = points[:, 0]
    inside = (x >= min(upper[:, 0].min(), lower[:, 0].min())) & (x <= max(upper[:, 0].max(), lower[:, 0].max()))
    between = inside & (points[:, 2] < polyline_z(upper, x)) & (points[:, 2] > polyline_z(lower, x))
    return float(between.mean()) if len(points) else 0.0


def fit_sphere(points):
    """(center, radius, rms distance from the sphere) of the sphere closest to ``points``."""
    a = np.hstack([2.0 * points, np.ones((len(points), 1))])
    b = (points * points).sum(1)
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    center = solution[:3]
    radius = math.sqrt(max(solution[3] + center @ center, 0.0))
    return center, radius, float(np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius) ** 2)))


def eyeball_side(points, skeleton, names=None, weights=None, label=""):
    """The eye (side) whose eyeball ``points`` are, or None: weighted to an eyeball bone, named like the
    game's eye material (_iri_), or a cap of a sphere near an eye bone that spans EYEBALL_SPAN both ways."""
    if not len(points):
        return None
    if names is not None and weights is not None:
        for side in SIDES:
            name = EYEBALL_BONE.format(side)
            if name in names and weights[:, names.index(name)].mean() > 0.5:
                return side
    for side in SIDES:
        bone = EYE_BONE.format(side)
        if not skeleton.has(bone):
            continue
        bone_head = skeleton.head(bone)
        if np.sign(points[:, 0].mean()) != np.sign(bone_head[0]) or np.linalg.norm(points.mean(0) - bone_head) > 0.03:
            continue
        if "_iri_" in label.lower():
            return side
        if len(points) < 12:
            continue
        center, radius, rms = fit_sphere(points)
        if not (0.006 < radius < 0.03) or rms > EYEBALL_RMS * radius or np.linalg.norm(center - bone_head) > 0.5 * radius:
            continue
        directions = (points - center) / radius
        across = np.degrees(np.arcsin(np.clip(directions[:, 0], -1.0, 1.0)))
        up = np.degrees(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
        if np.ptp(across) >= EYEBALL_SPAN and np.ptp(up) >= EYEBALL_SPAN:
            return side
    return None


def eyeball_bone(skeleton, side):
    """The bone eyeball ``side`` follows: its eyeball bone, or where the face skeleton has none (the game's
    faces 91 and 92), its eye bone, else the head."""
    for name in (EYEBALL_BONE.format(side), EYE_BONE.format(side)):
        if skeleton.has(name):
            return name
    return HEAD


def along_both_margins(points, features, side):
    """Whether ``points`` run along both lid margins of eye ``side``: RIM_SPAN of them closer to each margin,
    and near it."""
    upper, lower, _eye = features.eyes[side]
    du = polyline_parameter(upper, points)[1]
    dl = polyline_parameter(lower, points)[1]
    return min(float(((du < dl) & (du < LASH_DISTANCE)).mean()),
               float(((dl < du) & (dl < LASH_DISTANCE)).mean())) >= RIM_SPAN


def nearest_margin(features, points):
    """(eye side, the least distance of ``points`` to one of its lid margins) for the eye whose margins
    ``points`` come closest to, or None."""
    best = None
    for side, (upper, lower, _eye) in features.eyes.items():
        d = min(polyline_parameter(margin, points)[1].min() for margin in (upper, lower))
        if best is None or d < best[1]:
            best = (side, float(d))
    return best


def classify_part(points, skin_bvh, features, skeleton, names=None, weights=None, label=""):
    """What a loose part (``points`` in the reference's model space) is: (kind, detail), detail being the
    eye side for eyeballs, lashes and films."""
    side = eyeball_side(points, skeleton, names, weights, label)
    if side is not None:
        return PART_EYEBALL, side
    _tri, _loc, dist = nearest_on(skin_bvh, points)
    on_skin = np.percentile(dist, 90) < SHELL_DISTANCE
    margin = nearest_margin(features, points)
    if on_skin and margin is not None and margin[1] < LASH_DISTANCE and dist.max() > LASH_STANDS_OUT:
        on_skin = False  # lashes sunk into the lid, not makeup on it: they still stand out somewhere
    lip = features.lip
    if lip is not None and not on_skin:
        center = points.mean(0)
        if lip[:, 0].min() - 0.002 < center[0] < lip[:, 0].max() + 0.002:
            order = np.argsort(lip[:, 0])
            yc = np.interp(center[0], lip[order, 0], lip[order, 1])
            zc = np.interp(center[0], lip[order, 0], lip[order, 2])
            # Behind the lips, but in front of the neck bone (the spine): not at the back of the head.
            if yc + 0.002 < center[1] < skeleton.head(NECK)[1] and abs(center[2] - zc) < 0.015:
                return PART_INSIDE, None
    if on_skin:
        return PART_SHELL, None
    if margin is not None and margin[1] < LASH_DISTANCE:
        share = opening_share(points, features, margin[0])
        if share > FILM_SHARE or (share > RIM_SHARE and along_both_margins(points, features, margin[0])):
            return PART_FILM, margin[0]
        return PART_LASH, margin[0]
    return PART_RIGID, None


def line_weights(features, line, skin_points_, skin_tris, skin_weights_, samples=96):
    """(``samples`` points spread evenly along ``line``, a polyline on the skin, and the skin's weights at
    them)."""
    line = resample(line, samples)
    tri, loc, _dist = nearest_on(features.bvh, line)
    return line, interpolate(skin_weights_, skin_tris, tri, barycentric(skin_points_, skin_tris, tri, loc))


def weights_along(line, weights, s):
    """``weights`` of evenly spread ``line`` samples interpolated at positions ``s`` (0..1 by length)."""
    position = np.asarray(s, dtype=np.float64) * (len(line) - 1)
    k = np.clip(np.floor(position).astype(np.int64), 0, len(line) - 2)
    t = np.clip(position - k, 0.0, 1.0)[:, None]
    return weights[k] * (1.0 - t) + weights[k + 1] * t


def film_weights(points, features, side, skin_points_, skin_tris, skin_weights_):
    """Weights for a film over an eye opening: each point between the upper margin's weights and the lower
    one's by its height between them, so it stretches with the lids."""
    upper_line, upper_w = line_weights(features, features.eyes[side][0], skin_points_, skin_tris, skin_weights_)
    lower_line, lower_w = line_weights(features, features.eyes[side][1], skin_points_, skin_tris, skin_weights_)
    x = points[:, 0]
    zu = polyline_z(features.eyes[side][0], x)
    zl = polyline_z(features.eyes[side][1], x)
    t = np.clip((points[:, 2] - zl) / np.maximum(zu - zl, 1e-6), 0.0, 1.0)[:, None]
    return (weights_along(upper_line, upper_w, polyline_parameter(upper_line, points)[0]) * t +
            weights_along(lower_line, lower_w, polyline_parameter(lower_line, points)[0]) * (1.0 - t))


def graph_nearest(points, tris, sources):
    """For each of ``points``: the one of ``sources`` (indices) closest along the edges of the mesh
    (``tris``, welded at split seams), or -1 when none is connected to it; and how far along the edges it is."""
    import heapq
    welded = Welded(points, tris)
    count = welded.count
    neighbours = [[] for _ in range(count)]
    lengths = np.linalg.norm(welded.points[welded.edges[:, 0]] - welded.points[welded.edges[:, 1]], axis=1)
    for (a, b), length in zip(welded.edges.tolist(), lengths.tolist()):
        neighbours[a].append((b, length))
        neighbours[b].append((a, length))
    dist = np.full(count, np.inf)
    owner = np.full(count, -1, dtype=np.int64)
    heap = []
    for source in np.asarray(sources, dtype=np.int64).tolist():
        node = int(welded.node[source])
        if dist[node] > 0.0:
            dist[node] = 0.0
            owner[node] = source
            heap.append((0.0, node))
    heapq.heapify(heap)
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for other, length in neighbours[node]:
            if d + length < dist[other]:
                dist[other] = d + length
                owner[other] = owner[node]
                heapq.heappush(heap, (d + length, other))
    return owner[welded.node], dist[welded.node]


class LashRoots:
    """Where each point of a lash part around eye ``side`` grows from: its root (``root``, the index of the
    point it hangs from, the closest along the part's surface among the points by a lash line; ``length``,
    how far along the surface), the lid (``lid``: 0 upper, 1 lower), where the root is along that lid's lash
    line (``s``, 0..1 by length; ``q``, the closest point of the line; ``d``, how far) and along its margin
    (``m``, 0..1 by length). ``both``: the part grows from both lids (lashes around the whole eye, a tear
    line along the opening's rim)."""

    def __init__(self, points, tris, features, side):
        self.side = side
        found = [polyline_parameter(features.lash_line(side, lid), points) for lid in (0, 1)]
        (su, du, qu), (sl, dl, ql) = found
        # A lid has lashes of the part when LASH_BEYOND of the part (at least 4 points) lies beyond its margin
        # (above the upper one, below the lower one), along LASH_COVER of it at least: lashes around the whole
        # eye grow from both lids, a strip reaching past one margin at a corner only from one.
        upper_margin, lower_margin, _eye = features.eyes[side]
        x = points[:, 0]
        across = (x >= min(upper_margin[:, 0].min(), lower_margin[:, 0].min())) & (
            x <= max(upper_margin[:, 0].max(), lower_margin[:, 0].max()))
        beyond = (across & (points[:, 2] > polyline_z(upper_margin, x)) & (du < LASH_DISTANCE),
                  across & (points[:, 2] < polyline_z(lower_margin, x)) & (dl < LASH_DISTANCE))
        lengths = [np.linalg.norm(np.diff(features.lash_line(side, lid), axis=0), axis=1).sum() for lid in (0, 1)]
        cover = []
        for lid, s in ((0, su), (1, sl)):
            columns = np.unique(np.floor(s[beyond[lid]] * lengths[lid] / LASH_ROOT_SPACING))
            cover.append(len(columns) * LASH_ROOT_SPACING / max(lengths[lid], 1e-9))
        enough = max(4, LASH_BEYOND * len(points))
        grows = [np.count_nonzero(beyond[lid]) >= enough and cover[lid] >= LASH_COVER for lid in (0, 1)]
        if not any(grows):
            # Mostly by one lid: the one it reaches past more, else the one it's closer to.
            if max(cover) > 0.0:
                grows[int(np.argmax(cover))] = True
            else:
                grows[0 if np.median(du) <= np.median(dl) else 1] = True
        self.both = all(grows)
        if self.both:
            upper = du <= dl
        else:
            upper = np.full(len(points), grows[0])
        # Roots every millimetre along each lid: the points there closest to its lash line, counting how much
        # farther from the eyeball's center than the innermost point there they are too, so lashes sunk into
        # the lid still grow from their ends by the eyeball, and strips over the eye from their edges on the lid.
        roots = np.zeros(len(points), dtype=bool)
        radius = np.linalg.norm(points - features.eyes[side][2].center, axis=1)
        for lid, mine, s, d in ((0, upper, su, du), (1, ~upper, sl, dl)):
            if not mine.any():
                continue
            column = np.floor(s * lengths[lid] / LASH_ROOT_SPACING).astype(np.int64)
            for c in np.unique(column[mine]):
                here = mine & (column == c)
                if d[here].min() < LASH_DISTANCE:
                    score = d + radius - radius[here].min()
                    roots |= here & (score <= score[here].min() + LASH_ROOT_BAND)
        if not roots.any():
            roots[int(np.argmin(np.minimum(du, dl)))] = True
        owner, length = graph_nearest(points, tris, np.flatnonzero(roots))
        self.root = np.where(owner >= 0, owner, np.arange(len(points)))
        self.length = np.where(owner >= 0, length, 0.0)
        self.lid = np.where(upper[self.root], 0, 1)
        self.s = np.where(self.lid == 0, su[self.root], sl[self.root])
        self.q = np.where((self.lid == 0)[:, None], qu[self.root], ql[self.root])
        self.d = np.where(self.lid == 0, du[self.root], dl[self.root])
        mu = polyline_parameter(upper_margin, points)[0]
        ml = polyline_parameter(lower_margin, points)[0]
        self.m = np.where(self.lid == 0, mu[self.root], ml[self.root])


def lash_weights(roots, features, skin_points_, skin_tris, skin_weights_, names):
    """Weights for a lash part (see LashRoots): those of the lid margin where each point's root grows from,
    so every strand moves along with its root on the lid; toward the tips, some of the lid weight goes to
    the head (LASH_TIP_SHARE)."""
    out = np.zeros((len(roots.root), skin_weights_.shape[1]))
    head = names.index(HEAD) if HEAD in names else None
    fade = LASH_TIP_SHARE * smoothstep((roots.length - LASH_TIP_START) / LASH_TIP_LENGTH)
    for lid in (0, 1):
        mine = np.flatnonzero(roots.lid == lid)
        if not len(mine):
            continue
        line, w = line_weights(features, features.eyes[roots.side][lid], skin_points_, skin_tris, skin_weights_)
        w = weights_along(line, w, roots.m[mine])
        columns = lid_columns(names, roots.side)[lid]
        if head is not None and columns:
            given = w[:, columns] * fade[mine, None]
            w[:, columns] -= given
            w[:, head] += given.sum(1)
        out[mine] = w
    return out


def lash_offsets(points, roots, features, lift=0.0):
    """How far to move each point of a lash part (see LashRoots) so the roots sit where the game's lashes grow
    from the lids (LASH_ROOT_OUT and LASH_ROOT_ALONG off the lash lines), and ``lift`` farther out from the
    eyeball: along each lid, what the root closest to the line every millimetre needs is averaged smoothly
    (LASH_SNAP_SMOOTH), and every point moves with its root. Lashes spanning less than 3 mm move as one
    piece."""
    eye = features.eyes[roots.side][2]
    offsets = np.zeros((len(points), 3))
    for lid in (0, 1):
        mine = np.flatnonzero(roots.lid == lid)
        if not len(mine):
            continue
        line = features.lash_line(roots.side, lid)
        length = np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
        own = np.unique(roots.root[mine])
        s = roots.s[own]
        d = roots.d[own]
        q = roots.q[own]
        ahead = resample(line, 200)
        position = np.clip(s * 199, 0, 198).astype(np.int64)
        tangent = ahead[position + 1] - ahead[position]
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        outward = q - eye.center
        outward -= (outward * tangent).sum(1, keepdims=True) * tangent
        outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-12)
        along = np.cross(tangent, outward)
        along *= np.where(along[:, 2:3] >= 0.0, 1.0, -1.0) * (1.0 if lid == 0 else -1.0)
        target = q + outward * (LASH_ROOT_OUT[lid] + lift) + along * LASH_ROOT_ALONG[lid] - points[own]
        if (s.max() - s.min()) * length < 0.003:
            root_offsets = np.repeat(target[int(np.argmin(d))][None], len(own), axis=0)
        else:
            bins = max(2, int((s.max() - s.min()) * length * 1000.0))
            edges = np.linspace(s.min(), s.max() + 1e-9, bins + 1)
            centers, chosen = [], []
            for b0, b1 in zip(edges[:-1], edges[1:]):
                inside = np.flatnonzero((s >= b0) & (s < b1))
                if len(inside):
                    best = inside[np.argmin(d[inside])]
                    centers.append(s[best])
                    chosen.append(target[best])
            # A smooth average along the lid (Gaussian, LASH_SNAP_SMOOTH wide), so the lashes move without
            # creasing where their roots are uneven.
            centers = np.array(centers)
            chosen = np.array(chosen)
            spread = (s[:, None] - centers[None]) * length / LASH_SNAP_SMOOTH
            kernel = np.exp(-0.5 * spread * spread)
            root_offsets = kernel @ chosen / np.maximum(kernel.sum(1, keepdims=True), 1e-12)
        lookup = {int(root): k for k, root in enumerate(own)}
        offsets[mine] = root_offsets[[lookup[int(root)] for root in roots.root[mine]]]
    return offsets


def extra_weights(points, fit):
    """The weights of the game face's other parts at ``points`` (a loose part of the custom face), when the
    part lies where they are once warped onto the game's face (EXTRA_DISTANCE), else None."""
    found = fit.reference.extras_transfer()
    if found is None:
        return None
    transfer, bvh = found
    warped = fit.warp(points)
    _tri, _loc, dist = nearest_on(bvh, warped)
    if np.median(dist) > EXTRA_DISTANCE:
        return None
    return transfer.weights_at(warped)


def one_piece(weights, points):
    """``weights`` averaged over the part when it's small enough (RIGID_SIZE) to move as one piece."""
    if np.linalg.norm(points.max(0) - points.min(0)) < RIGID_SIZE:
        return np.repeat(weights.mean(0)[None], len(points), axis=0)
    return weights


def rigid_weights(points, skin_bvh, skin_points_, skin_tris, skin_weights_):
    """One set of weights for a whole part: the skin's where the part touches it (within ATTACH_RANGE of its
    closest point to it), averaged."""
    tri, loc, dist = nearest_on(skin_bvh, points)
    touching = dist <= dist.min() + ATTACH_RANGE
    w = interpolate(skin_weights_, skin_tris, tri[touching],
                    barycentric(skin_points_, skin_tris, tri[touching], loc[touching]))
    closeness = np.exp(-((dist[touching] - dist.min()) / ATTACH_RANGE) ** 2)
    result = (w * closeness[:, None]).sum(0) / max(closeness.sum(), 1e-12)
    return np.repeat(result[None], len(points), axis=0)


# -----------------------------------------------------------------------------
# Planning

class Mesh:
    """A mesh the face tools work on: ``points`` (the reference's model space), ``tris``, its current
    weights (``names``, ``weights`` (n, len(names))) and loose ``parts``; ``label`` holds its names (the
    object's, its materials'), which may tell what it is."""

    def __init__(self, name, points, tris, names, weights, label=""):
        self.name = name
        self.label = label or name
        self.points = np.asarray(points, dtype=np.float64)
        self.tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        self.names = list(names)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(len(self.points), len(self.names))
        self.parts, self.part_count = loose_parts(self.points, self.tris)


def find_skin(meshes, reference):
    """Index of the face skin among ``meshes``: the one with the game's neck ring (if several, the biggest of
    those covering SKIN_COVER of the game's face, else the one covering the most of it: a body in one piece has
    the neck opening too), else the one covering the most of the game's face (at least SKIN_COVER). None when
    none is it."""
    game = reference.points[::max(1, len(reference.points) // 400)]

    def cover(mesh):
        _tri, _loc, dist = nearest_on(bvh_of(mesh.points, mesh.tris), game)
        return float((dist < 0.004).mean())

    ring = reference.ring()
    with_ring = []
    if ring is not None:
        center = ring[0].mean(0)
        for i, mesh in enumerate(meshes):
            if not len(mesh.tris):
                continue
            found = neck_ring(mesh.points, mesh.tris)
            if found is not None and np.linalg.norm(found[0].points[found[1]].mean(0) - center) < 0.01:
                with_ring.append(i)
    if len(with_ring) == 1:
        return with_ring[0]
    if with_ring:
        covers = {i: cover(meshes[i]) for i in with_ring}
        return max(with_ring, key=lambda i: (min(covers[i], SKIN_COVER), len(meshes[i].points)))
    # Without the neck opening (cut off, say): the mesh covering the most of the game's face, and at least
    # SKIN_COVER of it (lashes or a mask lie on the face too, but cover little of it).
    best = None
    for i, mesh in enumerate(meshes):
        if not len(mesh.tris):
            continue
        share = cover(mesh)
        if share >= SKIN_COVER and (best is None or share > best[0]):
            best = (share, i)
    return None if best is None else best[1]


def find_eyes(meshes, skeleton, radius, skip=None):
    """{side: Eye} of the eyeballs among ``meshes`` (parts weighted to an eyeball bone, or shaped like an
    eyeball about an eye bone); a sphere of the game's ``radius`` for an eye without one."""
    found = {}
    for i, mesh in enumerate(meshes):
        if i == skip:
            continue
        for part in range(mesh.part_count):
            verts = np.flatnonzero(mesh.parts == part)
            if len(verts) < 8:
                continue
            side = eyeball_side(mesh.points[verts], skeleton, mesh.names, mesh.weights[verts], mesh.label)
            if side is None:
                continue
            lookup = np.full(len(mesh.points), -1, dtype=np.int64)
            lookup[verts] = np.arange(len(verts))
            tris = lookup[mesh.tris]
            tris = tris[np.all(tris >= 0, axis=1)]
            points, all_tris = found.get(side, (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)))
            found[side] = (np.vstack([points, mesh.points[verts]]), np.vstack([all_tris, tris + len(points)]))
    eyes = {}
    for side in SIDES:
        bone = EYE_BONE.format(side)
        if not skeleton.has(bone):
            continue
        center = skeleton.head(bone)
        if side in found and len(found[side][1]):
            points, tris = found[side]
            eyes[side] = Eye(center, float(np.median(np.linalg.norm(points - center, axis=1))), points, tris, True)
        elif radius > 0.0:
            eyes[side] = Eye(center, radius)
    return eyes


def over(names, weights, to_names):
    """``weights`` over bone ``names`` re-indexed to ``to_names`` (names missing there are dropped)."""
    column = {name: i for i, name in enumerate(to_names)}
    out = np.zeros((len(weights), len(to_names)))
    for j, name in enumerate(names):
        if name in column:
            out[:, column[name]] = weights[:, j]
    return out


def region_mask(points, lines, inner, outer):
    """1 within ``inner`` of any of the polylines ``lines``, fading to 0 at ``outer``."""
    d = np.full(len(points), np.inf)
    for line in lines:
        if line is not None and len(line) >= 2:
            d = np.minimum(d, polyline_parameter(line, points)[1])
    return 1.0 - smoothstep((d - inner) / max(outer - inner, 1e-9))


class Plan:
    """What the face tools will do to a set of meshes: ``goals`` (mesh index -> weights over ``names`` for
    every vertex) for the vertices in ``masks`` (mesh index -> which vertices change), ``kinds`` (mesh index
    -> {kind: loose parts}), ``neck`` (NeckMatch of the skin), ``lashes`` (mesh index -> (vertices, offsets)
    that snap lashes onto the lash lines) and a ``report``."""

    def __init__(self):
        self.names = []
        self.goals = {}
        self.masks = {}
        self.kinds = {}
        self.neck = None
        self.lashes = {}
        self.report = {"eyes": {}, "eyeballs": [], "warnings": []}


# Which kinds of parts each choice of ``parts`` of `plan_face` weights: always, and when near its region.
PART_CHOICES = {
    'ALL': ({PART_EYEBALL, PART_LASH, PART_FILM, PART_SHELL, PART_INSIDE, PART_EXTRA, PART_RIGID}, set()),
    'EYES': ({PART_LASH, PART_FILM}, {PART_SHELL, PART_EXTRA, PART_RIGID}),
    'MOUTH': ({PART_INSIDE}, {PART_SHELL, PART_EXTRA, PART_RIGID}),
    'LASHES': ({PART_LASH}, set()),
    'NONE': (set(), set()),
}


def near_region(points, features, region):
    """Whether ``points`` come within the outer reach of the eyes' region (the margins, EYE_REGION) or the
    mouth's (the lip line, MOUTH_REGION)."""
    if region == 'EYES':
        lines = [margin for upper, lower, _eye in features.eyes.values() for margin in (upper, lower)]
        reach = EYE_REGION[1]
    else:
        lines = [features.lip] if features.lip is not None else []
        reach = MOUTH_REGION[1]
    return any(polyline_parameter(line, points)[1].min() < reach for line in lines if len(line) >= 2)


def part_tris(mesh, verts):
    """The triangles of the loose part ``verts`` of ``mesh``, over positions in ``verts``."""
    lookup = np.full(len(mesh.points), -1, dtype=np.int64)
    lookup[verts] = np.arange(len(verts))
    tris = lookup[mesh.tris]
    return tris[np.all(tris >= 0, axis=1)]


def plan_face(meshes, skin_index, reference, *, skin='ALL', parts='ALL', targets=None, close=True, neck=False,
              snap_lashes=False, lash_lift=0.0, max_groups=DEFAULT_MAX_GROUPS):
    """Plan the face tools on ``meshes`` (Mesh), the skin being ``meshes[skin_index]``, from the game's face
    ``reference``.

    ``skin``: how the skin's weights change: 'ALL' (all of them from the game's face), 'EYES' or 'MOUTH'
    (only around the eyes or the mouth, blended into its own), 'NECK' (only the neck), 'NONE'.
    ``parts``: which loose parts of the other meshes (``targets``, indices; None for all) get weights, by
    what they are (see PART_CHOICES): 'ALL', 'EYES' (lashes, films, and what's around the eyes), 'MOUTH'
    (what's inside the mouth or around it), 'LASHES' or 'NONE'.
    ``close``: raise lid weights until the blink closes the eyes (with 'ALL' and 'EYES').
    ``neck``: plan making the neck ring the game's (positions, normals, weights).
    ``snap_lashes``: plan moving the lashes' roots onto the lash lines, ``lash_lift`` out from the eye.
    Raises FaceError when 'EYES' or 'MOUTH' can't find what they fit to: the eye openings, the lip line.
    """
    plan = Plan()
    skeleton = reference.skeleton
    skin_mesh = meshes[skin_index]
    eyes = find_eyes(meshes, skeleton, reference.radius, skip=skin_index)
    plan.report["eyeballs"] = sorted(side for side, eye in eyes.items() if eye.found)
    fit = FaceFit(skin_mesh.points, skin_mesh.tris, reference, eyes)
    plan.report["features"] = fit.features.summary()
    plan.report["reference"] = reference.features().summary()
    # Every goal is over the game face's bones, the eyeballs' and the skin's own.
    names = list(reference.names)
    names += [bone for bone in dict.fromkeys(eyeball_bone(skeleton, side) for side in SIDES) if bone not in names]
    names += [name for name in skin_mesh.names if name not in names]
    current = over(skin_mesh.names, skin_mesh.weights, names)
    if skin == 'ALL':
        new_names, new_weights, plan.report["eyes"] = skin_weights(fit, close=close, max_groups=max_groups)
        skin_w = over(new_names, new_weights, names)
    elif skin in ('EYES', 'MOUTH', 'NECK'):
        if skin == 'EYES' and not fit.features.eyes:
            raise FaceError("Couldn't find the eye openings on the face, which Fix Eyelids fits the lids to{:s}".format(
                "" if plan.report["eyeballs"] else " (they're found against the eyeballs: select those too)"))
        if skin == 'MOUTH' and fit.features.lip is None:
            raise FaceError("Couldn't find the line between the lips on the face (where they meet), which Fix Mouth "
                            "fits the mouth to")
        new = over(reference.names, fit.reference_weights(), names)
        points = fit.points
        if skin == 'EYES':
            lines = [margin for upper, lower, _eye in fit.features.eyes.values() for margin in (upper, lower)]
            mask = region_mask(points, lines, *EYE_REGION)
            for _upper, _lower, eye in fit.features.eyes.values():
                mask[np.linalg.norm(points - eye.center, axis=1) < 0.9 * eye.radius] = 1.0
        elif skin == 'MOUTH':
            lip = fit.features.lip
            mask = region_mask(points, [lip], *MOUTH_REGION)
            order = np.argsort(lip[:, 0])
            yc = np.interp(points[:, 0], lip[order, 0], lip[order, 1])
            zc = np.interp(points[:, 0], lip[order, 0], lip[order, 2])
            # Inside the mouth: behind the lips, but in front of the neck bone (the spine), so not the back of
            # the head and neck behind it.
            inside = ((points[:, 0] > lip[:, 0].min() - 0.004) & (points[:, 0] < lip[:, 0].max() + 0.004) &
                      (points[:, 1] > yc + 0.002) & (points[:, 1] < skeleton.head(NECK)[1]) &
                      (np.abs(points[:, 2] - zc) < 0.025))
            mask[inside] = 1.0
        else:
            verts, neck_w = neck_weights(points, fit.tris, reference)
            mask = np.zeros(len(points))
            mask[verts] = 1.0
            new[verts] = over(reference.names, neck_w, names)
            ring = reference.ring()
            found = neck_ring(points, fit.tris)
            if ring is not None and found is not None:
                new = _ring_weights(points, fit.tris, new, (ring[0], ring[1], over(reference.names, ring[2], names)))
                mask[found[2]] = 1.0
        skin_w = mask[:, None] * new + (1.0 - mask[:, None]) * current
        if skin == 'EYES' and close and skeleton.deformations('BLINK'):
            skin_w, plan.report["eyes"] = close_eyes(points, fit.tris, names, skin_w, fit.features, fit.labels,
                                                     skeleton, reference.closing_share())
        skin_w = limit_weights(skin_w, max_groups)
    else:
        skin_w = current
    if skin != 'NONE':
        plan.goals[skin_index] = skin_w
        plan.masks[skin_index] = np.ones(len(skin_mesh.points), dtype=bool)
    # The other meshes, by what their loose parts are, attached to the skin with its new weights.
    always, nearby = PART_CHOICES[parts]
    others = [i for i in (range(len(meshes)) if targets is None else targets) if i != skin_index]
    if not always and not nearby:
        others = []
    surface = SidedTransfer(fit.points, fit.tris, skin_w, fit.labels) if others else None
    skin_bvh = fit.features.bvh
    for i in others:
        mesh = meshes[i]
        goal = np.zeros((len(mesh.points), len(names)))
        mask = np.zeros(len(mesh.points), dtype=bool)
        counts = {}
        offsets = np.zeros((len(mesh.points), 3))
        for part in range(mesh.part_count):
            verts = np.flatnonzero(mesh.parts == part)
            pts = mesh.points[verts]
            kind, detail = classify_part(pts, skin_bvh, fit.features, skeleton, mesh.names, mesh.weights[verts],
                                         mesh.label)
            extra = None
            if kind in (PART_INSIDE, PART_RIGID) and (PART_EXTRA in always or PART_EXTRA in nearby):
                # Where the game's face has a part of its own (ears, horns, beards), its weights.
                extra = extra_weights(pts, fit)
                if extra is not None:
                    kind = PART_EXTRA
            if kind not in always and not (kind in nearby and near_region(pts, fit.features, parts)):
                continue
            mask[verts] = True
            counts[kind] = counts.get(kind, 0) + 1
            if kind == PART_EYEBALL:
                goal[verts, names.index(eyeball_bone(skeleton, detail))] = 1.0
            elif kind == PART_LASH:
                roots = LashRoots(pts, part_tris(mesh, verts), fit.features, detail)
                if snap_lashes:
                    offsets[verts] = lash_offsets(pts, roots, fit.features, lash_lift)
                if roots.both:
                    # Around the whole eye: each point between the lids' weights by its height, as the
                    # game's tear lines and lashes around the eye are weighted.
                    goal[verts] = film_weights(pts + offsets[verts], fit.features, detail, fit.points, fit.tris,
                                               skin_w)
                else:
                    goal[verts] = lash_weights(roots, fit.features, fit.points, fit.tris, skin_w, names)
            elif kind == PART_FILM:
                goal[verts] = film_weights(pts, fit.features, detail, fit.points, fit.tris, skin_w)
            elif kind == PART_SHELL:
                goal[verts] = surface.weights_at(pts, point_labels(pts, fit.features), None)
            elif kind == PART_INSIDE:
                goal[verts] = one_piece(over(reference.names, fit.reference_weights(pts), names), pts)
            elif kind == PART_EXTRA:
                goal[verts] = one_piece(over(reference.names, extra, names), pts)
            else:
                goal[verts] = rigid_weights(pts, skin_bvh, fit.points, fit.tris, skin_w)
        if not mask.any():
            continue
        plan.goals[i] = limit_weights(goal, max_groups)
        plan.masks[i] = mask
        plan.kinds[i] = counts
        if snap_lashes and np.abs(offsets).max() > 0.0:
            moved = np.flatnonzero(np.abs(offsets).max(1) > 0.0)
            plan.lashes[i] = (moved, offsets[moved])
    plan.names = names
    if neck:
        plan.neck = match_neck(fit.points, fit.tris, reference)
        if plan.neck is None:
            plan.report["warnings"].append("the face has no neck opening like the game's, so the neck was left as it is")
        elif skin_index in plan.goals:
            # The ring's vertices go where the game's are: they take the weights the game's have there.
            plan.goals[skin_index][plan.neck.verts] = over(plan.neck.names, plan.neck.weights, names)
    return plan


# -----------------------------------------------------------------------------
# Which game face

# A mesh whose closest game face is farther than this (median) isn't that face. A neck opening this close to a
# race's (on average) tells the race, up to others within half of it more.
DETECT_DISTANCE = 0.004
RING_MATCH = 0.001
# A selected mesh with more than this share of its bone weight on bones the game's face doesn't use (hair
# bones, Viera ears, the body) isn't part of the face.
FOREIGN_SHARE = 0.5


def detect_face(meshes, reference, offsets=None):
    """(race, face number, how) of the game face ``meshes`` (Mesh or Shape, world space) replace, ``how`` being 'NAME'
    when a name gives it (c0801f0002 in an object's, mesh's or material's name) or 'SHAPE' for the game face
    closest in shape to one of them. ``offsets``: race -> offset of its model space in the world. None when
    no game face is close."""
    for mesh in meshes:
        found = face_from_names([mesh.name, mesh.label])
        if found is not None and reference.has_face(*found):
            return found[0], reference.resolve(*found), 'NAME'
    big = [mesh for mesh in meshes if len(mesh.tris) and len(mesh.points) >= 200]
    samples = [mesh.points[::max(1, len(mesh.points) // 500)] for mesh in big]
    # The neck opening tells the race apart (mods keep it the game's), up to races that share it.
    rings = [found[0].points[found[1]] for found in (neck_ring(mesh.points, mesh.tris) for mesh in big)
             if found is not None]
    races = list(reference.races)
    if rings:
        misses = {}
        for race in races:
            faces = reference.faces(race)
            game = reference.face(race, faces[0]).ring() if faces else None
            if game is None:
                continue
            offset = np.zeros(3) if offsets is None else np.asarray(offsets.get(race, np.zeros(3)))
            polygon = ring_polygon(game[0]) + offset
            misses[race] = min(float(polyline_parameter(polygon, ring)[1].mean()) for ring in rings)
        if misses and min(misses.values()) < RING_MATCH:
            best_miss = min(misses.values())
            races = [race for race, miss in misses.items() if miss < best_miss + RING_MATCH / 2]
    best = None
    for race in races:
        offset = np.zeros(3) if offsets is None else np.asarray(offsets.get(race, np.zeros(3)))
        for number in reference.faces(race):
            bvh = reference.face(race, number).bvh()
            for sample in samples:
                score = float(np.median(nearest_on(bvh, sample - offset)[2]))
                if best is None or score < best[0]:
                    best = (score, race, number)
    if best is None or best[0] > DETECT_DISTANCE:
        return None
    return best[1], best[2], 'SHAPE'


def foreign_share(mesh, known, bones=None):
    """The share of ``mesh``'s weight on bones not in ``known`` (only groups in ``bones`` count, when
    given)."""
    columns = [j for j, name in enumerate(mesh.names) if bones is None or name in bones]
    total = mesh.weights[:, columns].sum() if columns else 0.0
    if total <= 1e-9:
        return 0.0
    other = [j for j in columns if mesh.names[j] not in known]
    return float(mesh.weights[:, other].sum() / total) if other else 0.0


def known_bones(reference):
    """The bones a face and its parts may use: the game face's, the head's and the face skeleton's face bones
    (j_f_*; Viera ears have bones of the face skeleton too, but aren't part of the face)."""
    skeleton = reference.skeleton
    return set(reference.names) | {HEAD, NECK, UPPER_BACK} | {
        name for name in skeleton.face_bones() if name.startswith("j_f_")} | {
        eyeball_bone(skeleton, side) for side in SIDES}


def object_mesh(obj, offset=None):
    """The Mesh of ``obj`` in its basis shape, moved by -``offset`` into the reference's model space."""
    names, weights = object_weights(obj)
    return Mesh(obj.name, rest_points(obj, offset), mesh_tris(obj.data), names, weights,
                " ".join(object_face_names(obj)))


class Shape:
    """What `detect_face` needs of ``obj``, quicker to read than its Mesh (no weights, no loose parts): its
    ``name``, ``label`` (see Mesh), ``points`` (world space, basis shape) and ``tris``."""

    def __init__(self, obj):
        self.name = obj.name
        self.label = " ".join(object_face_names(obj))
        self.points = rest_points(obj)
        self.tris = mesh_tris(obj.data)


# -----------------------------------------------------------------------------
# Poses

def pose_bases(armature_object, deformations, offset=None):
    """{bone: 4x4 pose basis} that make the armature's bones deform like ``deformations`` ({bone: 4x4},
    Blender model space; moved by ``offset`` to the armature's head), on top of the head's current pose.
    Only bones the armature has; parents before children."""
    offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
    shift = np.eye(4)
    shift[:3, 3] = offset
    world = np.array(armature_object.matrix_world, dtype=np.float64)
    to_armature = np.linalg.inv(world) @ shift
    from_armature = np.linalg.inv(shift) @ world
    bones = armature_object.data.bones
    pose = armature_object.pose.bones
    head = pose.get(HEAD)
    rest_head = np.array(bones[HEAD].matrix_local, dtype=np.float64) if bones.get(HEAD) is not None else np.eye(4)
    head_change = np.array(head.matrix, dtype=np.float64) @ np.linalg.inv(rest_head) if head is not None else np.eye(4)
    targets = {}
    for name, change in deformations.items():
        bone = bones.get(name)
        if bone is None or name == HEAD:
            continue
        rest = np.array(bone.matrix_local, dtype=np.float64)
        targets[name] = head_change @ to_armature @ change @ from_armature @ rest

    def depth(bone):
        count = 0
        while bone.parent is not None:
            bone = bone.parent
            count += 1
        return count

    result = {}
    for name in sorted(targets, key=lambda n: depth(bones[n])):
        bone = bones[name]
        rest = np.array(bone.matrix_local, dtype=np.float64)
        if bone.parent is not None:
            parent_rest = np.array(bone.parent.matrix_local, dtype=np.float64)
            parent_pose = targets.get(bone.parent.name)
            if parent_pose is None:
                parent_pose = np.array(pose[bone.parent.name].matrix, dtype=np.float64)
            result[name] = np.linalg.inv(np.linalg.inv(parent_rest) @ rest) @ np.linalg.inv(parent_pose) @ targets[name]
        else:
            result[name] = np.linalg.inv(rest) @ targets[name]
    return result


# -----------------------------------------------------------------------------
# Blender objects

def rest_points(obj, offset=None):
    """World space positions of ``obj``'s vertices in its basis shape (shape keys off, no modifiers),
    moved by -``offset`` into the reference's model space."""
    mesh = obj.data
    keys = mesh.shape_keys
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    if keys is not None and keys.reference_key is not None:
        keys.reference_key.data.foreach_get("co", co)
    else:
        mesh.vertices.foreach_get("co", co)
    points = painting.transform_points(co.reshape(-1, 3).astype(np.float64), obj.matrix_world)
    if offset is not None:
        points = points - np.asarray(offset, dtype=np.float64)
    return points


def mesh_tris(mesh):
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    return tris.reshape(-1, 3)


def object_weights(obj):
    """(group names, weights (n, groups)) of ``obj``'s vertex groups."""
    names = [group.name for group in obj.vertex_groups]
    weights = np.zeros((len(obj.data.vertices), len(names)))
    for vert in obj.data.vertices:
        for elem in vert.groups:
            if elem.group < len(names):
                weights[vert.index, elem.group] = elem.weight
    return names, weights


def move_vertices(obj, verts, world_offsets):
    """Move vertices ``verts`` of ``obj`` by ``world_offsets`` in every shape key (keeping what each key
    does to them)."""
    mesh = obj.data
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    local = np.asarray(world_offsets, dtype=np.float64) @ np.linalg.inv(matrix[:3, :3]).T
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)

    def moved(flat):
        points = flat.reshape(-1, 3).astype(np.float64)
        points[verts] += local
        return points.astype(np.float32).ravel()

    keys = mesh.shape_keys
    if keys is None:
        mesh.vertices.foreach_get("co", co)
        mesh.vertices.foreach_set("co", moved(co))
    else:
        for block in keys.key_blocks:
            block.data.foreach_get("co", co)
            result = moved(co)
            block.data.foreach_set("co", result)
            if block == keys.reference_key:
                mesh.vertices.foreach_set("co", result)
    mesh.update()


def set_vertex_normals(obj, verts, world_normals):
    """Give the corners of ``verts`` custom normals ``world_normals``; other corners keep theirs (the custom
    ones where the mesh has them, the automatic ones where it has none)."""
    mesh = obj.data
    loop_count = len(mesh.loops)
    loop_verts = np.empty(loop_count, dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    if mesh.has_custom_normals:
        normals = np.empty(loop_count * 3)
        mesh.corner_normals.foreach_get("vector", normals)
        normals = normals.reshape(-1, 3)
    else:
        normals = np.zeros((loop_count, 3))  # zero keeps a corner's automatic normal
    matrix = np.array(obj.matrix_world, dtype=np.float64)[:3, :3]
    local = np.asarray(world_normals, dtype=np.float64) @ matrix  # normals go by the inverse transpose
    local /= np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-12)
    lookup = np.full(len(mesh.vertices), -1, dtype=np.int64)
    lookup[verts] = np.arange(len(verts))
    rows = lookup[loop_verts]
    has = rows >= 0
    normals[has] = local[rows[has]]
    mesh.normals_split_custom_set(normals.tolist())
    mesh.update()


# -----------------------------------------------------------------------------
# Stroke

class FaceStroke(goals.GoalStroke):
    """Blends a mesh's weights toward precomputed face weights (``names``, ``weights`` per vertex), dab by
    dab. See `goals.GoalStroke` for how the weights change."""

    def __init__(self, obj, rest_co, names, weights, *, max_groups=DEFAULT_MAX_GROUPS, group_mode='ALL'):
        super().__init__(obj, rest_co, max_groups=max_groups, group_mode=group_mode)
        self.goal_names = list(names)
        self.goal_weights = np.asarray(weights, dtype=np.float64)
        self.column_group = [self.name_to_group.get(name, -1) for name in self.goal_names]

    def _goal(self, vert, group, param, shares):
        row = self.goal_weights[vert]
        result = {}
        for column in np.flatnonzero(row > MIN_WEIGHT).tolist():
            key = self.column_group[column]
            if key < 0:
                key = self._key(self.goal_names[column])
                if key is None:
                    continue
            elif key in self.locked or key not in self.bone_groups:
                continue
            result[key] = result.get(key, 0.0) + float(row[column])
        total = sum(result.values())
        if total <= 1e-12:
            return None
        return [(1.0, {key: share / total for key, share in result.items()})]

    def _group_created(self, group, name):
        if name in self.goal_names:
            self.column_group[self.goal_names.index(name)] = group

    def _group_forgotten(self, group, name):
        self.column_group = [-1 if index == group else index for index in self.column_group]


# -----------------------------------------------------------------------------
# Setting up

def armature_offset(armature_object, skeleton):
    """How far the armature's head bone is from the face skeleton's (world space), or zeros."""
    if armature_object is None:
        return np.zeros(3)
    bone = armature_object.data.bones.get(HEAD)
    if bone is None or not skeleton.has(HEAD):
        return np.zeros(3)
    return np.array(armature_object.matrix_world @ bone.head_local, dtype=np.float64) - skeleton.head(HEAD)


def face_armature(objects):
    """The armature deforming the first of ``objects`` that has one (see `hairing.weighted_armature`), or
    None."""
    for obj in objects:
        armature = hairing.weighted_armature(obj)
        if armature is not None:
            return armature
    return None


class FaceSetup:
    """What the face tools work on: ``objects`` and their ``meshes`` (Mesh, in the reference's model space),
    the game's ``face`` (FaceModel) found ``how`` ('NAME', 'SHAPE' or 'PICK'), which mesh is the ``skin``,
    the ``armature`` object and its ``offset``, and the meshes ``left_out`` (indices) because their weights
    are on bones the face doesn't use (hair, Viera ears, the body)."""

    def __init__(self, objects, meshes, face, how, skin, armature, offset, left_out):
        self.objects = objects
        self.meshes = meshes
        self.face = face
        self.how = how
        self.skin = skin
        self.armature = armature
        self.offset = offset
        self.left_out = left_out

    @property
    def parts(self):
        """Indices of the meshes that are parts of the face: all but the skin and those left out."""
        return [i for i in range(len(self.meshes)) if i != self.skin and i not in self.left_out]


def face_title(race, number):
    return "{:s} {:s}".format(RACE_NAMES.get(race, race), face_label(number))


NOT_DETECTED = ("Couldn't tell which of the game's faces this is: no name like c0801f0002, and no game face has "
                "its shape here. Pick the face (Game Face: Pick)")


def detect(objects, reference=None, world=None):
    """(race, face number, how) of the game face the meshes ``objects`` replace (see `detect_face`; ``world``:
    their Mesh when read already), or None."""
    reference = Reference.get() if reference is None else reference
    armature = face_armature(objects)
    world = [Shape(obj) for obj in objects] if world is None else world
    offsets = {}
    for race in reference.races:
        faces = reference.faces(race)
        if faces:
            offsets[race] = armature_offset(armature, reference.face(race, faces[0]).skeleton)
    return detect_face(world, reference, offsets)


def setup(objects, settings, reference=None):
    """FaceSetup of ``objects`` with the settings' Game Face (Scene.magic_fit). Raises FaceError."""
    reference = Reference.get() if reference is None else reference
    armature = face_armature(objects)
    world = [object_mesh(obj) for obj in objects]
    if settings.face_source == 'PICK':
        race, number = settings.face_race, settings.face_number
        if not reference.has_face(race, number):
            raise FaceError("The game has no face {:s}".format(face_title(race, number)))
        number = reference.resolve(race, number)
        how = 'PICK'
    else:
        found = detect(objects, reference, world)
        if found is None:
            raise FaceError(NOT_DETECTED)
        race, number, how = found
    face = reference.face(race, number)
    offset = armature_offset(armature, face.skeleton)
    meshes = [Mesh(mesh.name, mesh.points - offset, mesh.tris, mesh.names, mesh.weights, mesh.label) for mesh in world]
    skin = find_skin(meshes, face)
    if skin is None:
        raise FaceError("None of the selected meshes is where the face of {:s} is: select the face (and its "
                        "parts){:s}".format(face_title(race, number),
                                            "" if how == 'PICK' else ", or pick the game's face"))
    known = known_bones(face)
    left_out = []
    for i, (obj, mesh) in enumerate(zip(objects, meshes)):
        if i != skin:
            bones = painting.deform_bone_names(obj) or None
            if foreign_share(mesh, known, bones) > FOREIGN_SHARE:
                left_out.append(i)
    # The skin's own weights, which the repairs blend into, are those of the bones deforming it once the face
    # bones are added: its masks (modifiers', shape keys') and groups of other bones aren't weights.
    bones = set(face.skeleton.names) | painting.deform_bone_names(objects[skin])
    if armature is not None:
        bones |= {bone.name for bone in armature.data.bones if bone.use_deform}
    skin_mesh = meshes[skin]
    columns = [j for j, name in enumerate(skin_mesh.names) if name in bones]
    skin_mesh.names = [skin_mesh.names[j] for j in columns]
    skin_mesh.weights = skin_mesh.weights[:, columns]
    return FaceSetup(objects, meshes, face, how, skin, armature, offset, left_out)


def _fingerprint(objects):
    """What the brush's cached goals depend on: the meshes and where their vertices are."""
    return tuple((obj.name, obj.data.as_pointer(), hash(rest_points(obj).tobytes())) for obj in objects)


_goal_cache = {}


def clear_cache():
    _goal_cache.clear()


def build_stroke(context, obj, settings):
    """A FaceStroke for the brush on ``obj``: toward the weights Face Weights gives it (the face's, or those
    of the part of the face it is, with the other selected meshes). Raises FaceError."""
    modifier = straighten.deforming_armature(obj)
    if modifier is None:
        raise FaceError("The face needs an Armature modifier: give it Face Weights first")
    objects = [obj] + [other for other in context.selected_objects
                       if other is not obj and other.type == 'MESH' and other.data is not obj.data]
    key = (_fingerprint(objects), settings.face_source, settings.face_race, settings.face_number,
           settings.face_close_eyes, settings.face_max_groups)
    cached = _goal_cache.get(key)
    if cached is None:
        job = setup(objects, settings)
        if 0 in job.left_out:
            raise FaceError("'{:s}' is weighted to bones the face doesn't use (hair, the body): the brush "
                            "paints faces and their parts".format(obj.name))
        plan = plan_face(job.meshes, job.skin, job.face, skin='ALL', parts='ALL', targets=[0],
                         close=settings.face_close_eyes, max_groups=settings.face_max_groups)
        if 0 not in plan.goals:
            raise FaceError("Select the face too: the brush weights parts of a face from it")
        cached = (plan.names, plan.goals[0], job.meshes[0].points + job.offset)
        _goal_cache.clear()
        _goal_cache[key] = cached
    names, weights, rest_co = cached
    missing = [name for name in names if weights[:, names.index(name)].any() and
               modifier.object.data.bones.get(name) is None]
    if missing:
        raise FaceError("Armature '{:s}' lacks face bones ({:s}{:s}): give the face Face Weights first".format(
            modifier.object.name, ", ".join(missing[:3]), ", ..." if len(missing) > 3 else ""))
    return FaceStroke(obj, rest_co, names, weights, max_groups=settings.face_max_groups,
                      group_mode=settings.group_mode)
