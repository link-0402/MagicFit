# SPDX-License-Identifier: GPL-3.0-or-later

"""Whether the packages the transfer solver needs are installed, checked without importing them.

SciPy and robust-laplacian ship as wheels with the extension, which Blender installs where they fit:
Windows (64-bit) with Blender 5.2's Python. Importing SciPy can take seconds the first time, so the
solver modules are only imported when a transfer runs.
"""

import importlib.util
import sys
import sysconfig

# (import name, name in messages)
REQUIREMENTS = (("scipy", "SciPy"), ("robust_laplacian", "robust-laplacian"))
# The platform and Python version the bundled wheels are built for.
WHEEL_PLATFORM = "win-amd64"
WHEEL_PYTHON = (3, 13)

_missing = None


def missing():
    """Names of the packages the solver needs but can't find. Checked once, since Blender installs an
    extension's wheels before enabling it."""
    global _missing
    if _missing is None:
        _missing = []
        for module, label in REQUIREMENTS:
            try:
                found = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                _missing.append(label)
    return _missing


def _wheels_fit():
    """Whether the bundled wheels are made for this system (so they should have been installed)."""
    return sysconfig.get_platform() == WHEEL_PLATFORM and sys.version_info[:2] == WHEEL_PYTHON


def problem():
    """Why the solver can't run on this system, in one line, or None."""
    names = missing()
    if not names:
        return None
    if _wheels_fit():
        return "{:s} missing: reinstall the add-on".format(" and ".join(names))
    return "Needs {:s}, included for Blender 5.2 on Windows".format(" and ".join(names))


def explanation():
    """`problem` in short lines, for the sidebar."""
    lines = ["Missing Python packages", ", ".join(missing())]
    if _wheels_fit():
        lines.append("Reinstall the add-on")
    else:
        lines += ["Included for Blender 5.2", "on Windows only"]
    return lines
