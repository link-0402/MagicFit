# SPDX-License-Identifier: GPL-3.0-or-later

"""Reading Customize+ templates: the text Customize+ copies to the clipboard.

That text is base64 of a gzip stream holding a version byte and JSON. Accepted, like Customize+ does:
  - versions 4 to 7: a template ({"Name", "Bones": {bone: {"Translation", "Rotation", "Scaling", ...}}}).
    Version 7 (Customize+ 2.2.1) only added "Source". Profiles are copied as templates too.
  - version 3: an old profile ({"ProfileName", "Bones": {bone: {"Translation", "Rotation", "Scaling"}}}).
  - version 2: an older profile ({"ScaleName", "Bones": {bone: {"Position", "Rotation", "Scale"}}}).
  - version 1: bones copied from the template editor ([{"BoneCodeName", "Translation", ...}]).
The JSON on its own (a template file's contents) works as well.

Values are cleaned up the way Customize+ loads them: vectors clamped to +-512, angles wrapped once into
-180..180, and a child scaling of 0 without Independent turned into 1.
"""

import base64
import binascii
import gzip
import json
import math
import zlib

# Newest template version this was written for; newer ones are read the same way, with a warning.
TEMPLATE_VERSION = 7
ROOT_BONE = "n_root"
LIMIT = 512.0


class TemplateError(ValueError):
    """The text isn't a Customize+ template this can read."""


class BoneEdit:
    """One bone's edit. Rotation in degrees, as Customize+ shows it."""

    __slots__ = ("translation", "rotation", "scaling", "child_scaling", "child_scaling_independent",
                 "propagate_translation", "propagate_rotation", "propagate_scale")

    def __init__(self, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scaling=(1.0, 1.0, 1.0),
                 child_scaling=(1.0, 1.0, 1.0), child_scaling_independent=False,
                 propagate_translation=False, propagate_rotation=False, propagate_scale=False):
        self.translation = tuple(_clamp(v) for v in translation)
        self.rotation = tuple(_wrap(v) for v in rotation)
        self.scaling = tuple(_clamp(v) for v in scaling)
        child_scaling = tuple(_clamp(v) for v in child_scaling)
        if child_scaling == (0.0, 0.0, 0.0) and not child_scaling_independent:
            child_scaling = (1.0, 1.0, 1.0)
        self.child_scaling = child_scaling
        self.child_scaling_independent = bool(child_scaling_independent)
        self.propagate_translation = bool(propagate_translation)
        self.propagate_rotation = bool(propagate_rotation)
        self.propagate_scale = bool(propagate_scale)

    @property
    def propagates(self):
        return self.propagate_translation or self.propagate_rotation or self.propagate_scale

    def is_edited(self):
        """Whether Customize+ applies this bone at all (`BoneTransform.IsEdited`)."""
        return (
            any(abs(v) >= 0.00001 for v in self.translation)
            or any(abs(v) >= 0.1 for v in self.rotation)
            or any(abs(v - 1.0) >= 0.00001 for v in self.scaling)
            or (self.child_scaling_independent and any(abs(v - 1.0) >= 0.00001 for v in self.child_scaling))
        )

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, data):
        return cls(**{name: data[name] for name in cls.__slots__ if name in data})


class Template:
    def __init__(self, name, version, bones):
        self.name = name
        self.version = version
        self.bones = bones  # {bone name: BoneEdit}

    def edited(self):
        """{bone: BoneEdit} of the bones Customize+ applies, in the template's order."""
        return {name: edit for name, edit in self.bones.items() if edit.is_edited()}

    def to_json(self):
        return json.dumps({
            "name": self.name,
            "version": self.version,
            "bones": {name: edit.to_dict() for name, edit in self.bones.items()},
        })

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        return cls(data["name"], data["version"],
                   {name: BoneEdit.from_dict(edit) for name, edit in data["bones"].items()})


def _clamp(value):
    value = float(value)
    if math.isnan(value):
        return 0.0
    return min(LIMIT, max(-LIMIT, value))


def _wrap(angle):
    angle = float(angle)
    if not math.isfinite(angle):
        return 0.0
    if angle > 180.0:
        angle -= 360.0
    elif angle < -180.0:
        angle += 360.0
    return angle


def _vector(data, key, default):
    value = data.get(key)
    if not isinstance(value, dict):
        return default
    try:
        return tuple(float(value.get(axis, fallback)) for axis, fallback in zip("XYZ", default))
    except (TypeError, ValueError, OverflowError):
        raise TemplateError("A bone's {:s} isn't a number".format(key)) from None


def _bone(data, legacy=False):
    if not isinstance(data, dict):
        raise TemplateError("A bone's values are missing")
    if legacy:
        return BoneEdit(
            translation=_vector(data, "Position", (0.0, 0.0, 0.0)),
            rotation=_vector(data, "Rotation", (0.0, 0.0, 0.0)),
            scaling=_vector(data, "Scale", (1.0, 1.0, 1.0)),
        )
    return BoneEdit(
        translation=_vector(data, "Translation", (0.0, 0.0, 0.0)),
        rotation=_vector(data, "Rotation", (0.0, 0.0, 0.0)),
        scaling=_vector(data, "Scaling", (1.0, 1.0, 1.0)),
        child_scaling=_vector(data, "ChildScaling", (1.0, 1.0, 1.0)),
        child_scaling_independent=data.get("ChildScalingIndependent", False),
        propagate_translation=data.get("PropagateTranslation", False),
        propagate_rotation=data.get("PropagateRotation", False),
        propagate_scale=data.get("PropagateScale", False),
    )


def _decode(text):
    """(version byte, JSON text) from Customize+'s clipboard format."""
    compact = "".join(text.split())
    try:
        raw = base64.b64decode(compact + "=" * (-len(compact) % 4), validate=True)
    except (binascii.Error, ValueError):
        raise TemplateError("Not a Customize+ template") from None
    try:
        data = gzip.decompress(raw)
    except (OSError, EOFError, zlib.error):
        try:
            data = zlib.decompress(raw)
        except zlib.error:
            raise TemplateError("Not a Customize+ template") from None
    if not data:
        raise TemplateError("The template is empty")
    try:
        return data[0], data[1:].decode("utf-8-sig")
    except UnicodeDecodeError:
        raise TemplateError("Not a Customize+ template") from None


def parse(text):
    """The `Template` in ``text``: Customize+'s clipboard text, or a template's JSON. Raises
    `TemplateError` with a message for the user."""
    text = (text or "").strip()
    if not text:
        raise TemplateError("Paste a Customize+ template first")
    if text[0] in "{[":
        version, body = None, text
    else:
        version, body = _decode(text)
    try:
        data = json.loads(body)
    except ValueError:
        raise TemplateError("The template is damaged") from None

    if isinstance(data, list):
        # Bones copied in Customize+'s template editor.
        bones = {}
        for entry in data:
            if isinstance(entry, dict) and entry.get("BoneCodeName"):
                bones[str(entry["BoneCodeName"])] = _bone(entry)
        if not bones:
            raise TemplateError("The copied bone list is empty")
        return Template("Copied bones", version or 1, bones)
    if not isinstance(data, dict) or not isinstance(data.get("Bones"), dict):
        raise TemplateError("The template has no bones")

    json_version = data.get("Version", data.get("ConfigVersion"))
    if version is None:
        version = json_version if isinstance(json_version, int) else TEMPLATE_VERSION
    legacy = version == 2 or ("ScaleName" in data and "Version" not in data)
    name = data.get("Name") or data.get("ProfileName") or data.get("ScaleName") or "Template"
    bones = {str(bone): _bone(values, legacy) for bone, values in data["Bones"].items()}
    if not bones:
        raise TemplateError("The template has no bones")
    return Template(str(name), int(version), bones)
