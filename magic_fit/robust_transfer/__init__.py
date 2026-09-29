# SPDX-License-Identifier: GPL-3.0-or-later

"""Weight Transfer: transfers the body's weights onto whole meshes at once.

Where a mesh lies on the body and its normals agree, the body's weights are copied; everywhere else
they're inpainted, the smoothest weights those matches allow. Gaps between the legs, the cleavage and
the armpits then get blended weights instead of whatever skin happens to be closest.

Based on Robust Weight Transfer for Blender by sentfromspacevr (GPL-2.0-or-later,
https://github.com/sentfromspacevr), from "Robust Skin Weights Transfer via Weight Inpainting" by Rinat
Abdrashitov et al. (https://github.com/rin-23/RobustSkinWeightsTransferCode, MIT). The engine modules
(`weighttransfer`, `transfer`, `seams`, `util`) are kept close to that add-on's.

The solver needs SciPy and robust-laplacian, see `dependencies`. They're imported on first use, so
without them only this panel is affected.
"""

# Blender runs this module again, in place, when the add-on is updated or reloaded (see the add-on's
# __init__). Then the submodules are reloaded too, in dependency order. The solver modules are imported
# on first use, so only those loaded so far.
_reloading = "operators" in locals()

from . import dependencies, util, properties, operators, ui  # noqa: E402

if _reloading:
    import importlib
    import sys
    for _name in ("dependencies", "util", "seams", "weighttransfer", "transfer", "properties", "operators", "ui"):
        _module = sys.modules.get(__name__ + "." + _name)
        if _module is not None:
            importlib.reload(_module)

_modules = (properties, operators, ui)


def register():
    for module in _modules:
        module.register()


def unregister():
    for module in reversed(_modules):
        module.unregister()
