# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "Magic Fit",
    "author": "Luci_xiv",
    "description": "Copy weights from a body, fit, resize and line up meshes on it, weight hair and custom faces like the game's, preview Customize+ templates and relax stretched textures",
    "version": (1, 0, 1),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Magic Fit tab, and the Weight Paint / Edit Mode toolbars",
    "doc_url": "https://github.com/link-0402/MagicFit",
    "category": "Mesh",
}

# Blender runs this module again, in place, when the add-on is updated or reloaded. Then the other
# modules are reloaded too, in dependency order. (Every version so far has had `properties`.)
_reloading = "properties" in locals()

from . import properties, tabs, painting, fitting, straighten, goals, skirting, heeling, hairing, smoothing, gamefiles, facedata, facing, relaxing, resizing, lineup, clipping, operators, tool  # noqa: E402
from . import robust_transfer, cplus, preferences  # noqa: E402

if _reloading:
    import importlib
    for _module in (properties, tabs, painting, fitting, straighten, goals, skirting, heeling, hairing, smoothing,
                    gamefiles, facedata, facing, relaxing, resizing, lineup, clipping, operators, tool, robust_transfer,
                    cplus, preferences):
        importlib.reload(_module)

_modules = (properties, tabs, operators, tool, clipping, robust_transfer, cplus, preferences)


def register():
    for module in _modules:
        module.register()


def unregister():
    for module in reversed(_modules):
        module.unregister()
    painting.clear_cache()
    smoothing.clear_cache()
    fitting.BodySurface.clear_cache()
    operators.clear_follow_cache()
    facing.clear_cache()
