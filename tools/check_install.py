"""Install a built extension package into a throwaway Blender profile and check it the way users get it.

Checks that Blender installs it and its wheels, enables it without errors or extension policy warnings
(such as changes to sys.path), doesn't load SciPy when enabled, runs Weight Transfer with SciPy
loaded from the shared wheels rather than from the add-on's folder, has its preferences, applies a
Customize+ template, gives a hair weights with its hair skeleton data, and a face weights with the face
data it reads from the game (which must be installed where Magic Fit finds it).

The profile goes in a short folder under the system's temp folder, because SciPy's deepest files
would exceed Windows' 260 character path limit inside a long one. It has to be a throwaway profile:
enabling an extension with factory settings makes Blender remove the wheels of every extension that
isn't enabled, and in your own profile those would be the ones you installed.

Run from the repository root, with any Python 3:
    python tools/check_install.py dist/magic_fit-1.0.2.zip [path to blender] [--skip-face]

--skip-face leaves out Face Weights, for computers without the game (such as the release workflow's).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"

CHECK = r'''
import json, os, sys
import addon_utils
import bpy

module = "bl_ext.user_default.magic_fit"
errors = []
addon_utils.enable(module, default_set=True, handle_error=lambda error: errors.append(repr(error)))
warnings_get = getattr(addon_utils, "_extensions_warnings_get", None)
result = {
    "errors": errors,
    "warnings": warnings_get().get(module, []) if warnings_get else ["Blender's extension warnings are unavailable"],
    "scipy_loaded_when_enabled": "scipy" in sys.modules,
}
dependencies = sys.modules.get(module + ".robust_transfer.dependencies")
result["missing"] = dependencies.missing() if dependencies else ["the add-on did not load"]
try:
    data = bpy.data.armatures.new("Rig")
    rig = bpy.data.objects.new("Rig", data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    data.edit_bones.new("Bone").tail = (0, 0, 1)
    bpy.ops.object.mode_set(mode='OBJECT')

    def plane(name, z):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(-1, -1, z), (1, -1, z), (1, 1, z), (-1, 1, z)], [], [(0, 1, 2, 3)])
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.modifiers.new("Armature", 'ARMATURE').object = rig
        return obj

    body = plane("Body", 0.0)
    body.vertex_groups.new(name="Bone").add([0, 1, 2, 3], 1.0, 'REPLACE')
    garment = plane("Garment", 0.01)
    bpy.context.view_layer.objects.active = garment
    bpy.context.scene.magic_fit.target = body
    result["transfer"] = sorted(bpy.ops.magic_fit.transfer_weights())
    group = garment.vertex_groups.get("Bone")
    result["weights"] = [group.weight(i) for i in range(4)] if group else []
    import scipy
    result["scipy"] = scipy.__file__
    result["addon"] = sys.modules[module].__file__
    # Customize+: the preferences (with the hotkey) belong to the installed add-on, and a template
    # scaling the bone changes the body through the hidden rig.
    result["preferences"] = bpy.context.preferences.addons[module].preferences is not None
    settings = bpy.context.scene.magic_fit_cplus
    settings.armature = rig
    settings.template = json.dumps({"Version": 7, "Name": "Check", "Bones": {"Bone": {
        "Translation": {"X": 0, "Y": 0, "Z": 0}, "Rotation": {"X": 0, "Y": 0, "Z": 0},
        "Scaling": {"X": 2, "Y": 2, "Z": 2}}}})
    result["cplus"] = sorted(bpy.ops.magic_fit.cplus_apply())
    bpy.context.view_layer.update()
    evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    result["cplus_moved"] = max(abs(a.co[0] - b.co[0]) for a, b in zip(evaluated.vertices, body.data.vertices))
    result["cplus_removed"] = sorted(bpy.ops.magic_fit.cplus_remove())
    # Hair Weights: a ponytail hanging down the back of the Midlander female's head finds a skeleton in
    # the installed hair_reference.npz, gets an armature and weights.
    import math
    verts, faces = [], []
    for i in range(11):
        for j in range(8):
            a = 2 * math.pi * j / 8
            verts.append((0.03 * math.cos(a), 0.12 + 0.03 * math.sin(a), 1.52 - 0.04 * i))
    for i in range(10):
        for j in range(8):
            faces.append((i * 8 + j, i * 8 + (j + 1) % 8, (i + 1) * 8 + (j + 1) % 8, (i + 1) * 8 + j))
    mesh = bpy.data.meshes.new("Hair")
    mesh.from_pydata(verts, [], faces)
    hair = bpy.data.objects.new("Hair", mesh)
    bpy.context.scene.collection.objects.link(hair)
    for other in bpy.context.selected_objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = hair
    hair.select_set(True)
    bpy.context.scene.magic_fit.hair_race = 'c0201'
    result["hair"] = sorted(bpy.ops.magic_fit.hair_weights())
    result["hair_skeleton"] = hair.magic_fit_hair.skeleton
    result["hair_groups"] = len(hair.vertex_groups)
    # Face Weights: the Midlander female's first face, which the installed add-on reads from the game (the
    # game must be installed where Magic Fit finds it), named like the game's, gets its armature and weights back.
    if not os.environ.get("MF_CHECK_SKIP_FACE"):
        facing = sys.modules[module + ".facing"]
        face = facing.Reference.get().face("c0201", 1)
        mesh = bpy.data.meshes.new("c0201f0001_fac_a")
        mesh.from_pydata(face.points.tolist(), [], face.tris.tolist())
        skin = bpy.data.objects.new("c0201f0001_fac_a", mesh)
        bpy.context.scene.collection.objects.link(skin)
        for other in bpy.context.selected_objects:
            other.select_set(False)
        bpy.context.view_layer.objects.active = skin
        skin.select_set(True)
        result["face"] = sorted(bpy.ops.magic_fit.face_weights())
        result["face_groups"] = len(skin.vertex_groups)
        result["face_armature"] = any(m.type == 'ARMATURE' and m.object is not None for m in skin.modifiers)
except Exception as error:
    errors.append(repr(error))
print("MF_CHECK " + json.dumps(result))
'''


def run(blender, args, env):
    process = subprocess.run([blender, *args], env=env, capture_output=True, text=True, errors="replace")
    return process.returncode, process.stdout + process.stderr


def main():
    skip_face = "--skip-face" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--skip-face"]
    if not args:
        sys.exit(__doc__)
    package = os.path.abspath(args[0])
    blender = args[1] if len(args) > 1 else DEFAULT_BLENDER
    profile = tempfile.mkdtemp(prefix="mf-")
    env = dict(os.environ, BLENDER_USER_RESOURCES=profile)
    if skip_face:
        env["MF_CHECK_SKIP_FACE"] = "1"
    problems = []
    try:
        code, output = run(blender, ["--factory-startup", "--command", "extension", "install-file",
                                     "-r", "user_default", "-e", package], env)
        if code != 0:
            print(output)
            sys.exit("Blender could not install the package (exit code {:d})".format(code))
        script = os.path.join(profile, "check.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(CHECK)
        code, output = run(blender, ["--background", "--factory-startup", "--python-exit-code", "1",
                                     "--python", script], env)
        lines = [line for line in output.splitlines() if line.startswith("MF_CHECK ")]
        if not lines:
            print(output)
            sys.exit("The check didn't run (exit code {:d})".format(code))
        result = json.loads(lines[-1][len("MF_CHECK "):])
        problems += result["errors"] + result["warnings"]
        problems += ["missing " + name for name in result["missing"]]
        if result["scipy_loaded_when_enabled"]:
            problems.append("SciPy was loaded when the add-on was enabled")
        if result.get("transfer") != ["FINISHED"] or result.get("weights") != [1.0] * 4:
            problems.append("Transfer Weights gave {!r}, weights {!r}".format(result.get("transfer"), result.get("weights")))
        if not result.get("preferences"):
            problems.append("The add-on's preferences (hotkeys) aren't registered")
        if result.get("cplus") != ["FINISHED"] or result.get("cplus_removed") != ["FINISHED"] or                 abs(result.get("cplus_moved", 0.0) - 1.0) > 1e-5:
            problems.append("Customize+ gave {!r}/{!r}, moved {!r}".format(
                result.get("cplus"), result.get("cplus_removed"), result.get("cplus_moved")))
        if result.get("hair") != ["FINISHED"] or result.get("hair_skeleton", -1) < 0 or not result.get("hair_groups"):
            problems.append("Hair Weights gave {!r}, skeleton {!r}, {!r} groups".format(
                result.get("hair"), result.get("hair_skeleton"), result.get("hair_groups")))
        if not skip_face and (result.get("face") != ["FINISHED"] or result.get("face_groups", 0) < 10 or
                              not result.get("face_armature")):
            problems.append("Face Weights gave {!r}, {!r} groups, armature {!r}".format(
                result.get("face"), result.get("face_groups"), result.get("face_armature")))
        addon_dir = os.path.dirname(result.get("addon", ""))
        if addon_dir and result.get("scipy", "").startswith(addon_dir):
            problems.append("SciPy was loaded from the add-on's folder")
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    if problems:
        sys.exit("Problems:\n  " + "\n  ".join(problems))
    print("OK: installed, enabled without warnings, transferred weights with SciPy from the shared wheels, "
          "applied a Customize+ template, gave a hair weights" + ("" if skip_face else " and a face weights"))


if __name__ == "__main__":
    main()
