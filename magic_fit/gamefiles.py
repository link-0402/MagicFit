# SPDX-License-Identifier: GPL-3.0-or-later

"""Read-only access to the files of the user's FFXIV install, which Face reads the game's faces from.

- `find_game`: the game's install folder, the one set in the add-on's preferences or where the launchers
  (XIVLauncher, the game's own installer, Steam) put it.
- `Index`: the files of one of the game's SqPack categories (such as chara): standard files (skeletons,
  animations, tables) and models, which come out exactly as the .mdl files TexTools and Instant Edit export,
  and whether a file is a mod TexTools installed into the game's files.
- `read_est`, `read_sklb`: EST tables (faceskeletontemplate.est) and skeletons.
- `decode_spline`: Havok's hkaSplineCompressedAnimation, which the game's face animations use (XIVPy reads
  the Havok tagfile but only decodes quantised animations).

Nothing here writes to the game folder.
"""

import contextlib
import io
import json
import math
import os
import re
import struct
import zlib

import numpy as np

# The game's own installer puts the game in this folder, Steam in the second.
GAME_FOLDER = "FINAL FANTASY XIV - A Realm Reborn"
STEAM_FOLDER = "FINAL FANTASY XIV Online"
# The chara category's index, where every face model, skeleton and animation is: a game folder has it.
CHARA_INDEX = ("game", "sqpack", "ffxiv", "040000.win32.index")
VERSION_FILE = ("game", "ffxivgame.ver")
# The game's installer and Steam register the game under these uninstall keys.
UNINSTALL_KEYS = ("{2B41E132-07DF-4925-A3D3-F2D1765CCDFE}", "Steam App 39210")


# -----------------------------------------------------------------------------
# The game folder

def game_root(folder):
    """The game folder (the one with game and boot in it) that ``folder`` is, is in (its game or sqpack
    folder) or holds (the folder the installer or Steam made it in); None when there's no game there."""
    if not folder:
        return None
    folder = os.path.normpath(os.path.abspath(os.path.expanduser(folder)))
    parent = os.path.dirname(folder)
    for root in (folder, parent, os.path.dirname(parent), os.path.join(folder, GAME_FOLDER),
                 os.path.join(folder, STEAM_FOLDER)):
        if os.path.isfile(os.path.join(root, *CHARA_INDEX)):
            return root
    return None


def _xivlauncher_folders():
    """The game folders XIVLauncher (Windows) and XIVLauncher.Core (Linux, Steam Deck) start the game from."""
    folders = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        try:
            with open(os.path.join(appdata, "XIVLauncher", "launcherConfigV3.json"), encoding="utf-8-sig") as f:
                folders.append(json.load(f).get("GamePath"))
        except (OSError, ValueError, AttributeError):
            pass
    try:
        with open(os.path.expanduser(os.path.join("~", ".xlcore", "launcher.ini")), encoding="utf-8") as f:
            for line in f:
                key, _, value = line.partition("=")
                if key.strip() == "GamePath":
                    folders.append(value.strip())
    except OSError:
        pass
    folders.append(os.path.expanduser(os.path.join("~", ".xlcore", "ffxiv")))
    return folders


def _registry_folders():
    """The game folders the game's installer and Steam registered (Windows)."""
    try:
        import winreg
    except ImportError:
        return []
    folders = []
    for key in UNINSTALL_KEYS:
        for view in ("SOFTWARE\\WOW6432Node", "SOFTWARE"):
            path = view + "\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\" + key
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as handle:
                    for name in ("InstallLocation", "DisplayIcon"):
                        try:
                            value = str(winreg.QueryValueEx(handle, name)[0]).strip().strip('"')
                        except OSError:
                            continue
                        if name == "DisplayIcon":  # <game folder>\boot\ffxivboot.exe
                            value = os.path.dirname(os.path.dirname(value))
                        folders.append(value)
            except OSError:
                pass
    return folders


def _steam_folders():
    """Where Steam would have the game, in each of its libraries."""
    steams = []
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Software\\Valve\\Steam") as handle:
            steams.append(str(winreg.QueryValueEx(handle, "SteamPath")[0]))
    except (ImportError, OSError):
        pass
    if os.name == "nt":
        steams.append(os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "Steam"))
    else:
        steams += [os.path.expanduser(os.path.join("~", *p)) for p in (
            (".steam", "steam"), (".local", "share", "Steam"), ("Library", "Application Support", "Steam"))]
    libraries = []
    for steam in steams:
        libraries.append(steam)
        try:
            with open(os.path.join(steam, "steamapps", "libraryfolders.vdf"), encoding="utf-8") as f:
                libraries += [p.replace("\\\\", "\\") for p in re.findall(r'"path"\s+"([^"]+)"', f.read())]
        except OSError:
            pass
    return [os.path.join(library, "steamapps", "common", STEAM_FOLDER) for library in libraries]


def _default_folders():
    """Where the game's installers put it by default: Windows, and the Mac clients."""
    folders = []
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(variable)
        if base:
            folders.append(os.path.join(base, "SquareEnix", GAME_FOLDER))
    support = os.path.expanduser(os.path.join("~", "Library", "Application Support"))
    folders.append(os.path.join(support, "XIV on Mac", "ffxiv"))
    folders.append(os.path.join(support, "FINAL FANTASY XIV ONLINE", "Bottles", "published_Final_Fantasy",
                                "drive_c", "Program Files (x86)", "SquareEnix", GAME_FOLDER))
    return folders


def find_game(folder=""):
    """The game folder: ``folder`` (set by the user) when given, else the first one found where the launchers
    put it. None when there's none."""
    if folder:
        return game_root(folder)
    for candidate in _xivlauncher_folders() + _registry_folders() + _steam_folders() + _default_folders():
        root = game_root(candidate) if isinstance(candidate, str) else None
        if root is not None:
            return root
    return None


def game_version(root):
    """The version of the game in game folder ``root`` (its ffxivgame.ver), or '' when it can't be read."""
    try:
        with open(os.path.join(root, *VERSION_FILE), encoding="ascii", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def sqpack_folder(root):
    return os.path.join(root, "game", "sqpack")


# -----------------------------------------------------------------------------
# SqPack

CATEGORIES = {"common": 0x00, "bgcommon": 0x01, "bg": 0x02, "cut": 0x03, "chara": 0x04, "shader": 0x05,
              "ui": 0x06, "sound": 0x07, "vfx": 0x08, "ui_script": 0x09, "exd": 0x0A, "game_script": 0x0B,
              "music": 0x0C}
STANDARD_FILE, MODEL_FILE = 2, 3
# A model's sections in its SqPack entry, in file order: stack (vertex declarations), runtime (the rest of
# the model's header), then per level of detail its vertex buffer, edge geometry and index buffer.
MODEL_SECTIONS = 11
MDL_HEADER_SIZE = 0x44
# TexTools installs mods into the game's files: it writes them to .dat files of its own and points the index
# entries there. It marks its .dat files with one of these twice at 0x200; older versions of it left the data's
# hash in the .dat header (at 0x420) empty, which the game's own .dat files only have when they hold no data
# (just their two headers, DAT_HEADERS bytes).
TEXTOOLS_MARKS = (1337, 6969)
TEXTOOLS_MARK_OFFSET = 0x200
DAT_HASH = slice(0x420, 0x460)
DAT_HEADERS = 0x800


def path_hash(text):
    """SqPack's hash of a folder or file name: CRC-32 of the lower-case text, inverted."""
    return zlib.crc32(text.lower().encode()) ^ 0xFFFFFFFF


class Index:
    """The files of one SqPack category (``category``, such as 'chara') in the game's ``sqpack`` folder.
    ``used`` holds the index entry of every game path looked up (None: the game has no file there)."""

    def __init__(self, sqpack, category, expansion="ffxiv", chunk=0):
        code = CATEGORIES[category]
        number = 0 if expansion == "ffxiv" else int(expansion[2:])
        self.base = os.path.join(sqpack, expansion, "%02x%02x%02x.win32" % (code, number, chunk))
        with open(self.base + ".index", "rb") as f:
            data = f.read()
        header = struct.unpack_from("<I", data, 0x0C)[0]
        offset, size = struct.unpack_from("<II", data, header + 0x08)
        self.entries = {}
        for i in range(offset, offset + size, 16):
            file_hash, folder_hash, packed = struct.unpack_from("<III", data, i)
            self.entries[(folder_hash, file_hash)] = packed
        self.used = {}
        self._dats = {}
        self._textools = {}  # .dat number: whether TexTools made it

    def close(self):
        for handle in self._dats.values():
            handle.close()
        self._dats.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def entry(self, path):
        """The index entry of game path ``path`` (which .dat file its data is in, and where), or None when the
        game has no file there."""
        path = path.lower()
        folder, _, name = path.rpartition("/")
        packed = self.entries.get((path_hash(folder), path_hash(name)))
        self.used[path] = packed
        return packed

    def exists(self, path):
        return self.entry(path) is not None

    def from_textools(self, path):
        """Whether the file at game path ``path`` is a mod TexTools installed into the game's files: its entry
        points into a .dat file TexTools made."""
        packed = self.entry(path)
        if packed is None:
            return False
        number = (packed & 0xF) >> 1
        if number not in self._textools:
            with open(self.base + ".dat%d" % number, "rb") as f:
                head = f.read(DAT_HASH.stop)
                size = os.fstat(f.fileno()).st_size
            made = False
            if len(head) == DAT_HASH.stop:
                one, two = struct.unpack_from("<ii", head, TEXTOOLS_MARK_OFFSET)
                old = one == 0 and not any(head[DAT_HASH]) and size > DAT_HEADERS
                made = one == two and (one in TEXTOOLS_MARKS or old)
            self._textools[number] = made
        return self._textools[number]

    def read(self, path):
        """The file at game path ``path`` (a standard file or a model), or None when the game has none there."""
        packed = self.entry(path)
        if packed is None:
            return None
        number = (packed & 0xF) >> 1
        offset = (packed & ~0xF) * 0x08
        handle = self._dats.get(number)
        if handle is None:
            handle = self._dats[number] = open(self.base + ".dat%d" % number, "rb")
        handle.seek(offset)
        header_size, kind = struct.unpack("<II", handle.read(8))
        if kind == STANDARD_FILE:
            return self._read_standard(handle, offset, header_size, path)
        if kind == MODEL_FILE:
            return self._read_model(handle, offset, header_size)
        raise ValueError("%s is not a standard file or a model (type %d)" % (path, kind))

    @staticmethod
    def _block(handle, position):
        """The data of the block at ``position``: stored as is, or deflated."""
        handle.seek(position)
        header_size, _pad, compressed, decompressed = struct.unpack("<IIII", handle.read(16))
        handle.seek(position + header_size)
        if compressed == 32000:  # stored uncompressed
            return handle.read(decompressed)
        return zlib.decompress(handle.read(compressed), -15)

    def _read_standard(self, handle, offset, header_size, path):
        raw_size, _unknown, _block_size, blocks = struct.unpack("<IIII", handle.read(16))
        table = [struct.unpack("<IHH", handle.read(8)) for _ in range(blocks)]
        out = bytearray()
        for block_offset, _compressed, _decompressed in table:
            out += self._block(handle, offset + header_size + block_offset)
        if len(out) < raw_size:
            raise ValueError("%s is shorter than its entry says" % path)
        return bytes(out[:raw_size])

    def _read_model(self, handle, offset, header_size):
        """A model entry as the .mdl file it was made from: its sections one after the other behind the
        .mdl's header, which is rebuilt from the entry's (Lumina does it the same way)."""
        handle.seek(offset)
        head = handle.read(0xD0)
        version = struct.unpack_from("<I", head, 0x14)[0]
        offsets = struct.unpack_from("<11I", head, 0x70)
        first = struct.unpack_from("<11H", head, 0x9C)
        counts = struct.unpack_from("<11H", head, 0xB2)
        declarations, materials, lods, streaming, edge = struct.unpack_from("<HHBBB", head, 0xC8)
        sizes = struct.unpack("<%dH" % sum(counts), handle.read(2 * sum(counts)))
        out = bytearray(MDL_HEADER_SIZE)

        def section(k):
            start = len(out)
            position = offset + header_size + offsets[k]
            for b in range(counts[k]):
                out.extend(self._block(handle, position))
                position += sizes[first[k] + b]
            return start, len(out) - start

        stack = section(0)[1]
        runtime = section(1)[1]
        vertex_offsets, vertex_sizes, index_offsets, index_sizes = [0] * 3, [0] * 3, [0] * 3, [0] * 3
        for lod in range(3):
            if counts[2 + lod]:
                start, vertex_sizes[lod] = section(2 + lod)
                vertex_offsets[lod] = start if lod == 0 or start != vertex_offsets[lod - 1] else 0
            if counts[5 + lod]:
                section(5 + lod)
            if counts[8 + lod]:
                start, index_sizes[lod] = section(8 + lod)
                index_offsets[lod] = start if lod == 0 or start != index_offsets[lod - 1] else 0
        struct.pack_into("<IIIHH3I3I3I3IBBBB", out, 0, version, stack, runtime, declarations, materials,
                         *vertex_offsets, *index_offsets, *vertex_sizes, *index_sizes, lods, streaming, edge, 0)
        return bytes(out)


def read_est(data):
    """{(race, id): skeleton} of an EST table such as faceskeletontemplate.est; races like 'c0201'."""
    count = struct.unpack_from("<i", data, 0)[0]
    table = {}
    for i in range(count):
        entry, race = struct.unpack_from("<HH", data, 4 + i * 4)
        table[("c%04d" % race, entry)] = struct.unpack_from("<H", data, 4 + count * 4 + i * 2)[0]
    return table


def read_sklb(data):
    """(names, parent indices, local rows (t xyz, q xyzw, s xyz)) of a .sklb file, any version."""
    from .xivpy.kaos import Tagfile
    from .xivpy.utils import BinaryReader
    if data[:4] != b"blks":
        raise ValueError("not a skeleton file")
    version = data[4:8]
    offset = struct.unpack_from("<H", data, 10)[0] if version == b"0021" else struct.unpack_from("<I", data, 12)[0]
    reader = BinaryReader(data)
    reader.pos = offset
    with contextlib.redirect_stdout(io.StringIO()):
        tagfile = Tagfile.from_bytes(reader)
    skeleton = tagfile.get_skeleton_node()
    names = tagfile.get_bone_list(skeleton)
    parents = [int(p) for p in skeleton["parentIndices"]]
    pose = np.asarray(skeleton["referencePose"], dtype=np.float64)
    # hkQsTransform rows: translation, pad, rotation (xyzw), scale, pad.
    local = np.concatenate([pose[:, 0:3], pose[:, 4:8], pose[:, 8:11]], axis=1)
    return names, parents, local


# -----------------------------------------------------------------------------
# hkaSplineCompressedAnimation

BITS8, BITS16, POLAR32, THREECOMP40, THREECOMP48, THREECOMP24, STRAIGHT16, UNCOMPRESSED = range(8)
_ROTATION_ALIGN = {POLAR32: 4, THREECOMP40: 1, THREECOMP48: 2, THREECOMP24: 1, STRAIGHT16: 2, UNCOMPRESSED: 4}


class _Reader:
    def __init__(self, data, pos):
        self.data = data
        self.pos = pos

    def u8(self):
        self.pos += 1
        return self.data[self.pos - 1]

    def u16(self):
        self.pos += 2
        return struct.unpack_from("<H", self.data, self.pos - 2)[0]

    def f32(self):
        self.pos += 4
        return struct.unpack_from("<f", self.data, self.pos - 4)[0]

    def take(self, count):
        self.pos += count
        return self.data[self.pos - count:self.pos]

    def align(self, size, base):
        rest = (self.pos - base) % size
        if rest:
            self.pos += size - rest


def _smallest_three(stored, shift, negative):
    """A unit quaternion (x, y, z, w) from three stored components; the fourth, the largest, goes at
    ``shift`` with the sign ``negative`` gives it."""
    largest = math.sqrt(max(0.0, 1.0 - sum(c * c for c in stored)))
    if negative:
        largest = -largest
    return np.array(stored[:shift] + [largest] + stored[shift:])


def _read_quaternion(reader, kind):
    if kind == THREECOMP40:
        value = int.from_bytes(reader.take(5), "little")
        mask = (1 << 12) - 1
        stored = [(((value >> shift) & mask) - 2047) * 0.000345436 for shift in (0, 12, 24)]
        return _smallest_three(stored, (value >> 36) & 3, (value >> 38) & 1)
    if kind == THREECOMP48:
        x, y, z = struct.unpack("<HHH", reader.take(6))
        mask = (1 << 15) - 1
        stored = [((v & mask) - (mask >> 1)) * 0.000043161 for v in (x, y, z)]
        return _smallest_three(stored, ((y >> 14) & 2) | ((x >> 15) & 1), (z >> 15) & 1)
    if kind == UNCOMPRESSED:
        return np.array(struct.unpack("<ffff", reader.take(16)))
    raise NotImplementedError("rotation quantization %d" % kind)


def _knot_span(degree, value, count, knots):
    if value >= knots[count]:
        return count - 1
    low, high = degree, count
    middle = (low + high) // 2
    while value < knots[middle] or value >= knots[middle + 1]:
        if value < knots[middle]:
            high = middle
        else:
            low = middle
        middle = (low + high) // 2
    return middle


def _evaluate(degree, value, knots, points):
    """The B-spline with ``points`` and ``knots`` at ``value`` (de Boor's basis functions)."""
    span = _knot_span(degree, value, len(points), knots)
    basis = [1.0] + [0.0] * degree
    for i in range(1, degree + 1):
        for j in range(i - 1, -1, -1):
            denominator = knots[span + i - j] - knots[span - j]
            a = (value - knots[span - j]) / denominator if denominator else 0.0
            part = basis[j] * a
            basis[j + 1] += basis[j] - part
            basis[j] = part
    result = points[span] * basis[0]
    for i in range(1, degree + 1):
        result = result + points[span - i] * basis[i]
    return result


class _Vector:
    """A translation or scale track: per axis static, a spline, or the default."""

    def __init__(self, reader, base, flags, quantization, default):
        spline = [(flags >> (4 + i)) & 1 for i in range(3)]
        static = [(flags >> i) & 1 for i in range(3)]
        self.value = np.array(default, dtype=np.float64)
        self.points = None
        if any(spline):
            count = reader.u16() + 1
            self.degree = reader.u8()
            self.knots = list(reader.take(count + self.degree + 1))
            reader.align(4, base)
            ranges = [None, None, None]
            for i in range(3):
                if spline[i]:
                    ranges[i] = (reader.f32(), reader.f32())
                elif static[i]:
                    self.value[i] = reader.f32()
            self.points = np.tile(self.value, (count, 1))
            for k in range(count):
                for i in range(3):
                    if spline[i]:
                        low, high = ranges[i]
                        fraction = reader.u8() / 255.0 if quantization == BITS8 else reader.u16() / 65535.0
                        self.points[k, i] = low + (high - low) * fraction
            reader.align(4, base)
        else:
            for i in range(3):
                if static[i]:
                    self.value[i] = reader.f32()
            reader.align(4, base)

    def at(self, frame):
        return self.value.copy() if self.points is None else _evaluate(self.degree, frame, self.knots, self.points)


class _Rotation:
    def __init__(self, reader, base, flags, quantization):
        self.value = np.array((0.0, 0.0, 0.0, 1.0))
        self.points = None
        if flags & 0xF0:
            count = reader.u16() + 1
            self.degree = reader.u8()
            self.knots = list(reader.take(count + self.degree + 1))
            reader.align(_ROTATION_ALIGN[quantization], base)
            self.points = np.array([_read_quaternion(reader, quantization) for _ in range(count)])
            reader.align(4, base)
        elif flags & 0x0F:
            reader.align(_ROTATION_ALIGN[quantization], base)
            self.value = _read_quaternion(reader, quantization)
            reader.align(4, base)

    def at(self, frame):
        if self.points is None:
            return self.value.copy()
        q = _evaluate(self.degree, frame, self.knots, self.points)
        return q / max(np.linalg.norm(q), 1e-12)


def decode_spline(node):
    """Frames of an hkaSplineCompressedAnimation tagfile node: (frames, tracks, 10) with each track's
    translation (xyz), rotation (quaternion xyzw) and scale (xyz)."""
    data = bytes(node["data"])
    tracks = int(node["numberOfTransformTracks"])
    frames = int(node["numFrames"])
    per_block = int(node["maxFramesPerBlock"])
    mask_size = int(node["maskAndQuantizationSize"])
    result = np.zeros((frames, tracks, 10))
    for block, base in enumerate(node["blockOffsets"]):
        base = int(base)
        masks = [struct.unpack_from("<BBBB", data, base + 4 * t) for t in range(tracks)]
        reader = _Reader(data, base + mask_size)
        reader.align(4, base)
        decoded = []
        for quantization, position, rotation, scale in masks:
            decoded.append((
                _Vector(reader, base, position, quantization & 3, (0.0, 0.0, 0.0)),
                _Rotation(reader, base, rotation, ((quantization >> 2) & 0xF) + 2),
                _Vector(reader, base, scale, (quantization >> 6) & 3, (1.0, 1.0, 1.0)),
            ))
        first = block * (per_block - 1)
        for frame in range(first, min(frames, first + per_block)):
            local = float(frame - first)
            for t, (position, rotation, scale) in enumerate(decoded):
                result[frame, t, 0:3] = position.at(local)
                result[frame, t, 3:7] = rotation.at(local)
                result[frame, t, 7:10] = scale.at(local)
    return result
