"""Generate the toolbar icons in magic_fit/icons/.

Toolbar icons are triangle geometry in Blender's "VCO" format: an 8 byte header, then per triangle
three uint8 (x, y) coordinates in a 256x256 canvas (y up), then per triangle three RGBA colors.

    python tools/make_icon.py
"""

import math
import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICONS = os.path.join(ROOT, "magic_fit", "icons")

# The two grays Blender's own weight paint tool icons use.
LIGHT = (229, 229, 229, 255)
MID = (144, 144, 144, 255)


def drop(cx, cy, radius, tip=2.0, segments=28):
    """Outline of a drop: a circle at (cx, cy) running into a point ``tip`` radii above its center."""
    # The sides of the drop touch the circle this far from its top.
    alpha = math.acos(1.0 / tip)
    start = math.pi / 2.0 - alpha
    sweep = 2.0 * math.pi - 2.0 * alpha
    points = [(cx, cy + tip * radius)]
    for i in range(segments + 1):
        angle = start - sweep * i / segments
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def fan(points, center, color):
    """Triangulate a convex outline around ``center``."""
    closed = points + points[:1]
    return [((center, a, b), color) for a, b in zip(closed, closed[1:])]


def arc_band(cx, cy, inner, outer, start, end, color, segments=24):
    """A ring segment between two radii, from angle ``start`` to ``end`` (degrees)."""
    tris = []
    for i in range(segments):
        a0 = math.radians(start + (end - start) * i / segments)
        a1 = math.radians(start + (end - start) * (i + 1) / segments)
        p0 = (cx + inner * math.cos(a0), cy + inner * math.sin(a0))
        p1 = (cx + outer * math.cos(a0), cy + outer * math.sin(a0))
        p2 = (cx + outer * math.cos(a1), cy + outer * math.sin(a1))
        p3 = (cx + inner * math.cos(a1), cy + inner * math.sin(a1))
        tris += [((p0, p1, p2), color), ((p0, p2, p3), color)]
    return tris


def weight_brushes():
    # The weight being copied from (behind) and its copy (in front).
    back = (92.0, 84.0, 36.0)
    front = (154.0, 100.0, 46.0)
    return fan(drop(*back), back[:2], MID) + fan(drop(*front), front[:2], LIGHT)


def body_fit():
    # A body (dome) with a thick garment shell fitted around it.
    cx, cy = 128.0, 76.0
    return (
        arc_band(cx, cy, 0.0, 66.0, 0.0, 180.0, MID) +
        arc_band(cx, cy, 80.0, 108.0, 8.0, 172.0, LIGHT)
    )


def texture_relax():
    # A checkered texture, laid out evenly.
    tris = []
    left, bottom, size, gap = 42.0, 28.0, 172.0, 6.0
    cell = (size - 2.0 * gap) / 3.0
    for row in range(3):
        for column in range(3):
            x0 = left + column * (cell + gap)
            y0 = bottom + row * (cell + gap)
            x1, y1 = x0 + cell, y0 + cell
            color = LIGHT if (row + column) % 2 == 0 else MID
            tris += [(((x0, y0), (x1, y0), (x1, y1)), color), (((x0, y0), (x1, y1), (x0, y1)), color)]
    return tris


def write(path, tris):
    coords = bytearray()
    colors = bytearray()
    for (a, b, c), color in tris:
        for x, y in (a, b, c):
            coords += struct.pack("BB", round(min(max(x, 0), 255)), round(min(max(y, 0), 255)))
            colors += struct.pack("4B", *color)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"VCO\x00" + struct.pack("4B", 255, 255, 0, 0) + bytes(coords) + bytes(colors))


if __name__ == "__main__":
    for name, shapes in (("weight_brushes", weight_brushes), ("body_fit", body_fit), ("texture_relax", texture_relax)):
        path = os.path.join(ICONS, name + ".dat")
        write(path, shapes())
        print("wrote", path)
