# SPDX-License-Identifier: GPL-3.0-or-later

"""Customize+: shows a Customize+ template on an armature the way the game does.

The improved successor of sleepybnuuy's Bustomize (https://github.com/sleepybnuuy/bustomize, GPL-3.0),
rewritten from Customize+'s own code (https://github.com/Aether-Tools/CustomizePlus): templates up to
version 7, every bone changed in its own frame without dragging its children along, propagation, the
root's scale, bone axes of any armature, and a hidden rig instead of the armature's pose, so posing,
animating and clearing the pose leave the template alone and the other way round. See `solver` for the
math and `rig` for the setup.
"""

# Blender runs this module again, in place, when the add-on is updated or reloaded (see the add-on's
# __init__). Then the submodules are reloaded too, in dependency order.
_reloading = "operators" in locals()

from . import reference, template, solver, rig, properties, operators, ui  # noqa: E402

if _reloading:
    import importlib
    for _module in (reference, template, solver, rig, properties, operators, ui):
        importlib.reload(_module)

_modules = (properties, operators, ui)


def register():
    for module in _modules:
        module.register()


def unregister():
    for module in reversed(_modules):
        module.unregister()
