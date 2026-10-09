"""FFXIV .tex textures: reading any 2D format and writing uncompressed B8G8R8A8 (numpy only).

A .tex is an 80-byte header followed by the mip chain. Uncompressed formats are decoded
here; block-compressed ones (BC1-BC7) are wrapped as a DDS file around the same payload
and decoded by the OpenImageIO bundled with Blender. Writing is B8G8R8A8 only: Blender
can't encode BC formats, and Penumbra can compress the file afterwards.

Images are (h, w, 4) uint8 RGBA arrays with row 0 at the top of the texture.
From HairRebuild's reader.
"""

import os
import struct
import tempfile

import numpy as np

HEADER_SIZE = 80
TYPE_2D = 0x00800000           # attribute bits of a plain 2D texture
TYPE_MASK = 0x13C00000         # attribute bits that encode the texture type
MAX_MIPS = 13

L8 = 0x1130
A8 = 0x1131
B4G4R4A4 = 0x1440
B5G5R5A1 = 0x1441
B8G8R8A8 = 0x1450
B8G8R8X8 = 0x1451
BC1 = 0x3420
BC2 = 0x3430
BC3 = 0x3431
BC4 = 0x6120
BC5 = 0x6230
BC7 = 0x6432

FORMAT_NAMES = {
    L8: "L8", A8: "A8", B4G4R4A4: "B4G4R4A4", B5G5R5A1: "B5G5R5A1",
    B8G8R8A8: "B8G8R8A8", B8G8R8X8: "B8G8R8X8",
    BC1: "BC1", BC2: "BC2", BC3: "BC3", BC4: "BC4", BC5: "BC5", BC7: "BC7",
}

# Block formats as DDS: (FourCC, DXGI format when the FourCC is DX10, bytes per 4x4 block).
_BLOCK = {
    BC1: (b"DXT1", None, 8),
    BC2: (b"DXT3", None, 16),
    BC3: (b"DXT5", None, 16),
    BC4: (b"ATI1", None, 8),
    BC5: (b"ATI2", None, 16),
    BC7: (b"DX10", 98, 16),
}


def read_header(data):
    """Parse the 80-byte header into a dict (attr, format, width, height, mips, offsets)."""
    if len(data) < HEADER_SIZE:
        raise ValueError("not a .tex file (too short)")
    attr, fmt, width, height, depth = struct.unpack_from("<IIHHH", data, 0)
    if attr & TYPE_MASK != TYPE_2D or depth > 1 or data[15] > 1:
        raise ValueError("only plain 2D .tex textures are supported")
    mips = max(1, min(MAX_MIPS, data[14] & 0x7F))
    offsets = struct.unpack_from("<13I", data, 28)[:mips]
    return {"attr": attr, "format": fmt, "width": width, "height": height,
            "mips": mips, "offsets": offsets}


def surface_size(fmt, width, height):
    """Byte size of one mip level of the given format."""
    if fmt in _BLOCK:
        return max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * _BLOCK[fmt][2]
    bits = 1 << ((fmt >> 4) & 0xF)
    return width * height * bits // 8


def read_tex(source):
    """Decode the top mip of a .tex file (path or bytes).  Returns (rgba, header)."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    else:
        with open(source, "rb") as f:
            data = f.read()
    header = read_header(data)
    fmt, w, h = header["format"], header["width"], header["height"]
    size = surface_size(fmt, w, h)
    payload = data[header["offsets"][0]:header["offsets"][0] + size]
    if len(payload) < size:
        raise ValueError("truncated .tex file")
    return _decode(fmt, w, h, payload), header


def _decode(fmt, w, h, payload):
    if fmt in (B8G8R8A8, B8G8R8X8):
        rgba = np.frombuffer(payload, np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0, 3]].copy()
        if fmt == B8G8R8X8:
            rgba[:, :, 3] = 255
        return rgba
    if fmt in (L8, A8):
        v = np.frombuffer(payload, np.uint8).reshape(h, w)
        rgba = np.empty((h, w, 4), np.uint8)
        if fmt == L8:
            rgba[:, :, :3] = v[:, :, None]
            rgba[:, :, 3] = 255
        else:
            rgba[:, :, :3] = 0
            rgba[:, :, 3] = v
        return rgba
    if fmt == B4G4R4A4:
        v = np.frombuffer(payload, "<u2").reshape(h, w).astype(np.uint32)
        channels = [(v >> 8) & 0xF, (v >> 4) & 0xF, v & 0xF, (v >> 12) & 0xF]
        return (np.stack(channels, -1) * 17).astype(np.uint8)
    if fmt == B5G5R5A1:
        v = np.frombuffer(payload, "<u2").reshape(h, w).astype(np.uint32)
        rgb = np.stack([(v >> 10) & 0x1F, (v >> 5) & 0x1F, v & 0x1F], -1) * 255 // 31
        return np.concatenate([rgb, ((v >> 15) * 255)[:, :, None]], -1).astype(np.uint8)
    if fmt in _BLOCK:
        return decode_dds(dds_bytes(fmt, w, h, payload))
    raise ValueError(f"unsupported .tex format {fmt:#x}")


def dds_bytes(fmt, width, height, payload):
    """Wrap one block-compressed surface as a single-mip DDS file."""
    fourcc, dxgi, _ = _BLOCK[fmt]
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000    # CAPS HEIGHT WIDTH PIXELFORMAT LINEARSIZE
    pixel_format = struct.pack("<II4s5I", 32, 0x4, fourcc, 0, 0, 0, 0, 0)
    header = struct.pack("<7I", 124, flags, height, width, len(payload), 0, 1)
    header += bytes(44) + pixel_format + struct.pack("<5I", 0x1000, 0, 0, 0, 0)
    out = b"DDS " + header
    if dxgi is not None:
        out += struct.pack("<5I", dxgi, 3, 0, 1, 0)    # format, TEXTURE2D, flags, array size 1
    return out + bytes(payload)


def decode_dds(dds):
    """Decode DDS bytes to (h, w, 4) uint8 RGBA with OpenImageIO."""
    try:
        import OpenImageIO as oiio
    except ImportError:
        raise RuntimeError("compressed .tex files need OpenImageIO, which comes with Blender") from None
    # OpenImageIO's Python bindings read from files only, so round-trip through a temp file.
    fd, path = tempfile.mkstemp(suffix=".dds")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(dds)
        inp = oiio.ImageInput.open(path)
        if not inp:
            raise RuntimeError("OpenImageIO could not read the texture: " + oiio.geterror())
        try:
            spec = inp.spec()
            pixels = inp.read_image(0, 0, 0, spec.nchannels, "uint8")
        finally:
            inp.close()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    pixels = np.asarray(pixels, dtype=np.uint8).reshape(spec.height, spec.width, spec.nchannels)
    return to_rgba(pixels)


def to_rgba(pixels):
    """Expand a (h, w, c) uint8 array with 1-4 channels to RGBA."""
    h, w, c = pixels.shape
    if c == 4:
        return np.ascontiguousarray(pixels)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[:, :, 3] = 255
    if c == 1:
        rgba[:, :, :3] = pixels
    else:
        rgba[:, :, :c] = pixels
    return rgba


def box_mips(rgba, count):
    """`count` mip levels of an (h, w, 4) uint8 image (itself first), each the rounded 2x2
    mean of the one before; sizes follow the .tex rule max(1, n >> 1)."""
    out = [rgba]
    for _ in range(1, count):
        a = out[-1]
        h, w = a.shape[:2]
        a = a[:max(1, h // 2 * 2), :max(1, w // 2 * 2)].astype(np.uint16)
        n = 1
        if h > 1:
            a = a[0::2] + a[1::2]
            n *= 2
        if w > 1:
            a = a[:, 0::2] + a[:, 1::2]
            n *= 2
        out.append(((a + n // 2) // n).astype(np.uint8))
    return out


def full_mip_count(width, height):
    """Mip levels of a complete chain down to 1x1 (capped at the format's 13)."""
    return min(MAX_MIPS, 1 + int(np.floor(np.log2(max(1, width, height)))))


def tex_bytes(mips):
    """B8G8R8A8 .tex bytes for a mip chain of (h, w, 4) uint8 RGBA arrays (largest first)."""
    if not mips or len(mips) > MAX_MIPS:
        raise ValueError("need 1-13 mip levels")
    height, width = mips[0].shape[:2]
    offsets = []
    body = []
    pos = HEADER_SIZE
    for level, mip in enumerate(mips):
        expected = (max(1, height >> level), max(1, width >> level), 4)
        if mip.shape != expected:
            raise ValueError(f"mip {level} has shape {mip.shape}, expected {expected}")
        bgra = np.ascontiguousarray(mip[:, :, [2, 1, 0, 3]], dtype=np.uint8)
        offsets.append(pos)
        body.append(bgra.tobytes())
        pos += bgra.nbytes
    count = len(mips)
    header = struct.pack("<IIHHHBB", TYPE_2D, B8G8R8A8, width, height, 1, count, 0)
    header += struct.pack("<3I", *[min(i, count - 1) for i in range(3)])
    header += struct.pack("<13I", *(offsets + [0] * (MAX_MIPS - count)))
    return header + b"".join(body)


def write_tex(path, rgba, mips=None):
    """Save an (h, w, 4) uint8 image as an uncompressed .tex with `mips` levels (default: all)."""
    h, w = rgba.shape[:2]
    count = full_mip_count(w, h) if mips is None else max(1, min(mips, full_mip_count(w, h)))
    data = tex_bytes(box_mips(rgba, count))
    with open(path, "wb") as f:
        f.write(data)
