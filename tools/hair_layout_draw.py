"""Sheets of hair and its chains for tools/hair_layout_check.py: dots and lines drawn into a numpy image, saved as a
PNG (Blender's Python has no matplotlib).

A sheet shows a hair from the back, the left and the front side by side, in the rig's model space (meters, Z up):
X_RANGE either side of the middle, Z_LOW to Z_HIGH, SCALE pixels per metre.
"""

import struct
import zlib

import numpy as np

# The chains' colors, in turn.
PALETTE = np.array([
    (230, 60, 60), (60, 140, 255), (70, 200, 90), (245, 190, 40), (190, 90, 230), (40, 210, 210),
    (255, 130, 40), (240, 110, 180), (150, 220, 60), (120, 120, 255), (200, 160, 110), (255, 255, 255),
], dtype=np.float64)
# The views: (name, the model axis across the screen and its sign); up the screen is always Z.
VIEWS = (("back", (0, -1.0)), ("left", (1, 1.0)), ("front", (0, 1.0)))
SCALE = 900.0
X_RANGE = 0.36
Z_LOW, Z_HIGH = 0.70, 1.75
BACKGROUND = (24, 24, 28)
DIVIDER = (90, 90, 90)


class Canvas:
    """An RGB image to draw dots and lines into."""

    def __init__(self, width, height, background=BACKGROUND):
        self.image = np.empty((height, width, 3), dtype=np.uint8)
        self.image[:] = background

    def dots(self, x, y, color, size=1):
        """Squares ``size`` pixels wide at pixels ``x``, ``y`` in ``color`` (one (3,) or one per dot)."""
        x = np.rint(np.asarray(x)).astype(np.int64)
        y = np.rint(np.asarray(y)).astype(np.int64)
        color = np.asarray(color, dtype=np.float64)
        colors = np.broadcast_to(color, (len(x), 3)) if color.ndim == 1 else color
        h, w = self.image.shape[:2]
        for dx in range(-(size // 2), size - size // 2):
            for dy in range(-(size // 2), size - size // 2):
                px, py = x + dx, y + dy
                inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                self.image[py[inside], px[inside]] = colors[inside]

    def line(self, x0, y0, x1, y1, color, width=2):
        count = int(max(abs(x1 - x0), abs(y1 - y0))) + 2
        t = np.linspace(0.0, 1.0, count)
        self.dots(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, color, width)

    def save(self, path):
        h, w = self.image.shape[:2]
        raw = b"".join(b"\x00" + self.image[row].tobytes() for row in range(h))

        def chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
        with open(path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                         + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def surface_dots(points, tris, count=120000, seed=0):
    """``count`` dots spread over triangles ``tris`` of ``points`` ((n, 3)) by area, so cards with few vertices still
    show as surfaces: (where each lies ((count, 3)), the corner of its triangle it lies nearest, a vertex index)."""
    rng = np.random.default_rng(seed)
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    size = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    picks = rng.choice(len(tris), size=count, p=size / size.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    corner = np.where(u + v < 1.0 / 3.0, 0, np.where(u > v, 1, 2))
    at = a[picks] + u[:, None] * (b[picks] - a[picks]) + v[:, None] * (c[picks] - a[picks])
    return at, tris[picks, corner]


def sheet(path, points, colors, chains, chain_colors, marks=()):
    """Save a PNG of the back, left and front views at ``path``: ``points`` ((n, 3), model space) as dots in
    ``colors`` ((n, 3)), ``chains`` (polylines (m, 3), joints then tip) in ``chain_colors``, each joint a white dot,
    and ``marks`` (points, the body's bones) as white crosses."""
    pw, ph = int(2 * X_RANGE * SCALE), int((Z_HIGH - Z_LOW) * SCALE)
    canvas = Canvas(pw * len(VIEWS) + 4 * (len(VIEWS) - 1), ph)
    for v, (name, (axis, sign)) in enumerate(VIEWS):
        left = v * (pw + 4)
        depth = 1 if axis == 0 else 0

        def sx(p):
            return left + (sign * p[..., axis] + X_RANGE) * SCALE

        def sy(p):
            return (Z_HIGH - p[..., 2]) * SCALE
        # Far dots first, so the near side shows on top: the back view looks toward -Y (near is +Y), the front toward
        # +Y, the left toward -X (near is +X).
        key = points[:, depth] * (1.0 if name in ("back", "left") else -1.0)
        order = np.argsort(key, kind="stable")
        shown = order[np.abs(points[order, axis]) < X_RANGE]
        canvas.dots(sx(points[shown]), sy(points[shown]), colors[shown], 2)
        for mark in marks:
            x, y = float(sx(np.asarray(mark))), float(sy(np.asarray(mark)))
            canvas.line(x - 5, y, x + 5, y, (255, 255, 255), 1)
            canvas.line(x, y - 5, x, y + 5, (255, 255, 255), 1)
        for line, color in zip(chains, chain_colors):
            line = np.asarray(line)
            for a, b in zip(line[:-1], line[1:]):
                canvas.line(float(sx(a)), float(sy(a)), float(sx(b)), float(sy(b)), (0, 0, 0), 6)
            for a, b in zip(line[:-1], line[1:]):
                canvas.line(float(sx(a)), float(sy(a)), float(sx(b)), float(sy(b)), color, 3)
            for joint in line[:-1]:
                canvas.dots([float(sx(joint))], [float(sy(joint))], (0, 0, 0), 9)
                canvas.dots([float(sx(joint))], [float(sy(joint))], (255, 255, 255), 5)
        if v:
            canvas.image[:, left - 4:left] = DIVIDER
    canvas.save(path)
