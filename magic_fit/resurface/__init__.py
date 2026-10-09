# SPDX-License-Identifier: GPL-3.0-or-later

"""Mesh Rebuild, Normals and UV: the Mesh Repair tab's tools from Luci_xiv's Resurface
(https://github.com/link-0402/Resurface).

Rebuild Mesh works on the raw surface instead of a voxel grid, so borders, non-manifold junctions, UV
seams and thin strips (strings, straps, fringe) survive. UVs, colors, custom normals, vertex weights and
shape keys are transferred onto the new topology. Smooth Normals and Normal Map Seams clean up shading,
and Rebuild UVs, Transfer Texture, Resize Canvas and the Index Map Generator work on UV layouts. The
remeshing core (`core`) only needs NumPy and mathutils.

Its operators are `magic_fit.*` and its settings `Scene.magic_fit_repair`, so it can sit next to a
standalone Resurface. What it stores in the meshes and images is named as in Resurface, so meshes rebuilt
there can be restored here and the other way round.
"""

# Blender runs this module again, in place, when the add-on is updated or reloaded (see the add-on's
# __init__). Then the submodules are reloaded too, in dependency order.
_reloading = "ui" in locals()

from .core import geom, canvas, source, remesh, transfer, normals, uvseams, indexmap, texremap, normalmaps, tex  # noqa: E402
from . import blender_io, pipeline, props, operators, uvtools, ui  # noqa: E402

if _reloading:
    import importlib
    for _module in (geom, canvas, source, remesh, transfer, normals, uvseams, indexmap, texremap, normalmaps, tex,
                    blender_io, pipeline, props, operators, uvtools, ui):
        importlib.reload(_module)

_modules = (props, operators, uvtools, ui)


def register():
    for module in _modules:
        module.register()


def unregister():
    for module in reversed(_modules):
        module.unregister()
