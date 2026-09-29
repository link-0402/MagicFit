"""Copy the dependency wheels of Weight Transfer into the extension, and unpack them for the tests.

The wheels listed in magic_fit/blender_manifest.toml are read from a folder of downloaded
wheels (by default the one of the robust-weight-transfer repository next to this one, see its
scripts/install-dependencies.ps1) and written to magic_fit/wheels without the files Python
never loads: test suites, static and import libraries, debug symbols, headers and type stubs.
robust-laplacian's wheel is mostly a static library. Every importable module stays, because Blender
shares an installed wheel between all extensions that list it, so keep the same versions as the
standalone add-on. RECORD is updated to match.

The stripped wheels are also unpacked into deps/ at the repository root. Blender never sees that
folder; tests/test_transfer_core.py puts it on sys.path to run the add-on from source.

Run from the repository root, with any Python 3.11 or newer (Blender's own works):
    python tools/vendor_wheels.py [folder with the downloaded wheels]
"""

import os
import re
import shutil
import sys
import tomllib
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "magic_fit")
DEPS = os.path.join(ROOT, "deps")
DEFAULT_SOURCE = os.path.join(os.path.dirname(ROOT), "robust-weight-transfer", "wheels")

# The same files scripts/bundle-addon.ps1 of robust-weight-transfer leaves out of wheels.
_UNUSED_SUFFIXES = re.compile(r"\.(py[co]|whl|lib|a|exp|pdb|h|hpp|c|cpp|pxd|pxi|pyx|pyi)$", re.IGNORECASE)


def unused(name):
    """Whether Python never loads the wheel member ``name``."""
    parts = name.split("/")
    folders = parts[:-1]
    return (
        "__pycache__" in folders or "tests" in folders or parts[-1].lower() == "conftest.py" or
        _UNUSED_SUFFIXES.search(parts[-1]) is not None
    )


def _record_path(line):
    """First field of a RECORD line, which is quoted when the path holds a comma."""
    if line.startswith('"'):
        return re.match(r'"((?:[^"]|"")*)"', line).group(1).replace('""', '"')
    return line.split(",", 1)[0]


def strip_wheel(source, destination):
    """Write ``source`` to ``destination`` without the unused files; return (kept, left out) sizes."""
    kept = removed = 0
    record = None
    dropped = set()
    partial = destination + ".partial"
    with zipfile.ZipFile(source) as wheel, zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as out:
        for info in wheel.infolist():
            if info.is_dir():
                continue
            if re.fullmatch(r"[^/]+\.dist-info/RECORD", info.filename):
                record = info
                continue
            if unused(info.filename):
                dropped.add(info.filename)
                removed += info.file_size
                continue
            out.writestr(info, wheel.read(info), compress_type=zipfile.ZIP_DEFLATED)
            kept += info.file_size
        if record is None:
            raise RuntimeError("wheel has no RECORD: " + source)
        lines = wheel.read(record).decode("utf-8").splitlines()
        lines = [line for line in lines if line and _record_path(line) not in dropped]
        out.writestr(record, "\n".join(lines) + "\n", compress_type=zipfile.ZIP_DEFLATED)
    os.replace(partial, destination)
    return kept, removed


def main():
    source_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    with open(os.path.join(PACKAGE, "blender_manifest.toml"), "rb") as f:
        wheels = tomllib.load(f).get("wheels", [])
    if not wheels:
        sys.exit("blender_manifest.toml lists no wheels")

    shutil.rmtree(DEPS, ignore_errors=True)
    os.makedirs(DEPS)
    listed = set()
    for relative in wheels:
        name = relative.rsplit("/", 1)[-1]
        listed.add(name)
        source = os.path.join(source_dir, name)
        if not os.path.isfile(source):
            sys.exit("Missing wheel {:s}\nDownload it with robust-weight-transfer's scripts/install-dependencies.ps1 "
                     "(the versions in its requirements.txt must match the manifest)".format(source))
        destination = os.path.join(PACKAGE, *relative.lstrip("./").split("/"))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        kept, removed = strip_wheel(source, destination)
        print("{:s}: kept {:.1f} MB, left out {:.1f} MB (unpacked)".format(name, kept / 1e6, removed / 1e6))
        with zipfile.ZipFile(destination) as wheel:
            wheel.extractall(DEPS)

    # Wheels a previous manifest listed would still be packaged.
    wheel_dir = os.path.join(PACKAGE, "wheels")
    for name in os.listdir(wheel_dir):
        if name.endswith(".whl") and name not in listed:
            os.remove(os.path.join(wheel_dir, name))
            print("removed", name)
    print("unpacked into", DEPS)


if __name__ == "__main__":
    main()
