# SPDX-License-Identifier: GPL-3.0-or-later

"""Sizes of a body: the settings its shape keys follow, and the shapes it takes with other ones.

A devkit body changes size with shape keys that its script sets from a few settings: Yet Another Devkit keeps them
in the scene (``Scene.ya_devkit_props.<group>.<setting>``), the RueXB devkit as custom properties of an object
named Driver. The keys that change with them are driven: their drivers read the settings and turn keys on, off or
mute them. Those drivers tell which settings there are and what each can be (an enum's items, on and off, the
numbers it's compared with), and working them out again with other values gives the shape the body would take,
without switching anything (`size_values`). A body whose keys nothing drives (like Roh's devkit) changes size by
turning keys on and off: those keys are its settings.

A setting is known by its own name, the last part of the path the drivers read. A devkit's body parts (the YAB
devkit's Torso, Waist, Hands and Feet) and clothing whose keys Yet Another Addon transferred from the body follow
the same settings, sometimes read from another group (``torso_state.chest_size`` for the Mannequin's
``mannequin_state.chest_size``), and `size_values` works their keys out alike, with the keys their drivers copy
from other meshes.

Yet Another Devkit's script also does a few things its drivers don't show, and `_Yab` does them along: Lavabod
has no Masc chest, Alt Hips goes off with the leg sizes past Yanilla, and switching Lavabod gives the chest's
sub-keys the values they had the last time.
"""

import ast
import functools
import itertools
import math
import operator
import re

import bpy
import numpy as np

from .resizing import ResizeError, _driven, _group_weights, _key_positions, _transform, _tris, _vertex_areas

# A setting that is a shape key nothing drives is named after the key with this in front.
KEY_PREFIX = "key:"
# Shape keys the game turns on itself (FFXIV's shp_ keys, a devkit's shpx_ ones) are never sizes.
GAME_KEY_PREFIX = "shp"
# A whole range of numbers is offered up to this many values; beyond that, those the drivers compare it with.
MAX_RANGE = 12
# Settings that move the body less than this on average (over its area, for the option moving it most, in meters)
# are listed under More...
MAIN_REACH = 0.0003
# ...unless there are this many settings or fewer.
FEW_SETTINGS = 6
# Picks make at most this many sizes.
MAX_SIZES = 64
# A key counts as on from this value.
ON = 0.5

_KEY_PATH = re.compile(r'key_blocks\["((?:[^"\\]|\\.)*)"\]\.(value|mute)')
# The last part of a property path: a name after a dot, or a custom property's name in brackets.
_LAST_NAME = re.compile(r'(?:^|\.)([A-Za-z_]\w*)$')
_CUSTOM = re.compile(r'\["((?:[^"\\]|\\.)*)"\]$')
_FUNCTIONS = {"min": min, "max": max, "abs": abs, "round": round, "int": int, "float": float, "bool": bool,
              "floor": math.floor, "ceil": math.ceil, "sqrt": math.sqrt}
_CONSTANTS = {"True": True, "False": False, "pi": math.pi}
_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
               ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_COMPARISONS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
                ast.Gt: operator.gt, ast.GtE: operator.ge}
_RANGES = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)


class Option:
    """A value a setting can take: ``value`` as its drivers read it (a number), ``label`` to show, and ``word`` to
    name a size by."""

    __slots__ = ("value", "label", "word")

    def __init__(self, value, label, word=None):
        self.value = float(value)
        self.label = label
        self.word = label if word is None else word


class Setting:
    """A setting a body's shape keys follow.

    ``name`` is the property's own name, or KEY_PREFIX and a key's name for a key nothing drives; ``label`` names it
    for people. ``kind`` is 'ENUM', 'BOOL', 'NUMBER' or 'KEY'. ``options`` are the values it can take (Option),
    ``current`` the value it has now (None where it can't be read). ``reach`` is how much it reshapes the body (its
    mean move over the body's area, in meters, for the option that moves it most) and ``height`` where (up the
    body, in world space). ``more`` marks those that barely do, listed after the others.
    """

    __slots__ = ("name", "label", "kind", "options", "current", "reach", "height", "more", "source")

    def __init__(self, name, label, kind, options, current, source=None):
        self.name, self.label, self.kind, self.options, self.current = name, label, kind, options, current
        self.reach = self.height = 0.0
        self.more = False
        # (owner, path) the current value is read from.
        self.source = source

    def option(self, value):
        """The Option of ``value``, or None."""
        return next((option for option in self.options if option.value == value), None)


# -----------------------------------------------------------------------------
# Drivers

@functools.lru_cache(maxsize=512)
def _parsed(expression):
    return ast.parse(expression.strip(), mode="eval").body


def _expression_value(node, names):
    """The value of a driver's expression (``node``, parsed) with its variables at ``names``. Only arithmetic,
    comparisons, logic and a few functions, so nothing in a file can run code here."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError("it reads '{:s}'".format(node.id))
    if isinstance(node, ast.BoolOp):
        # As in Python: the first value that decides, else the last.
        for part in node.values:
            value = _expression_value(part, names)
            if bool(value) == isinstance(node.op, ast.Or):
                return value
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
        value = _expression_value(node.operand, names)
        return not value if isinstance(node.op, ast.Not) else -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        return _ARITHMETIC[type(node.op)](_expression_value(node.left, names), _expression_value(node.right, names))
    if isinstance(node, ast.Compare) and all(type(op) in _COMPARISONS for op in node.ops):
        left = _expression_value(node.left, names)
        for op, part in zip(node.ops, node.comparators):
            right = _expression_value(part, names)
            if not _COMPARISONS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _expression_value(node.body if _expression_value(node.test, names) else node.orelse, names)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
            and not node.keywords):
        return _FUNCTIONS[node.func.id](*(_expression_value(part, names) for part in node.args))
    raise ValueError("Resize can't work out '{:s}' in it".format(ast.unparse(node)))


def _compared(expression, variable):
    """The numbers driver ``expression`` compares its variable ``variable`` with, and for a range (``a < 3``) the
    whole numbers on either side of its end too."""
    try:
        tree = _parsed(expression)
    except SyntaxError:
        return set()
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for left, op, right in zip(operands, node.ops, operands[1:]):
            for name, constant in ((left, right), (right, left)):
                if not (isinstance(name, ast.Name) and name.id == variable and isinstance(constant, ast.Constant)):
                    continue
                value = constant.value
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                found.add(float(value))
                if isinstance(op, _RANGES) and isinstance(value, int):
                    found.update((float(value - 1), float(value + 1)))
    return found


def _unescape(name):
    return re.sub(r"\\(.)", r"\1", name)


def _mesh_keys(obj):
    return obj.data.shape_keys if obj is not None and obj.type == 'MESH' else None


def _key_drivers(keys):
    """The drivers of shape keys ``keys`` (a Key) that set a key's value or mute: {(key name, 'value' or 'mute'):
    F-curve}."""
    animation = keys.animation_data
    if animation is None:
        return {}
    result = {}
    for curve in animation.drivers:
        found = _KEY_PATH.fullmatch(curve.data_path)
        if found is not None and not curve.mute:
            result[(_unescape(found.group(1)), found.group(2))] = curve
    return result


def _key_link(var):
    """(Key, key name, 'value' or 'mute') of the shape key driver variable ``var`` reads, or None."""
    if var.type != 'SINGLE_PROP':
        return None
    target = var.targets[0]
    if not isinstance(target.id, bpy.types.Key):
        return None
    found = _KEY_PATH.fullmatch(target.data_path)
    return None if found is None else (target.id, _unescape(found.group(1)), found.group(2))


def _linked_keys(keys):
    """``keys`` (a Key) and the Keys whose key values its drivers copy, and theirs."""
    found = [keys]
    pointers = {keys.as_pointer()}
    for current in found:
        for curve in _key_drivers(current).values():
            for var in curve.driver.variables:
                link = _key_link(var)
                if link is not None and link[0].as_pointer() not in pointers:
                    pointers.add(link[0].as_pointer())
                    found.append(link[0])
    return found


def _setting_path(var):
    """(owner, path, name) of the setting driver variable ``var`` reads, or None for anything else: another shape
    key, a transform or array item, a variable of another type."""
    if var.type != 'SINGLE_PROP' or _key_link(var) is not None:
        return None
    target = var.targets[0]
    if target.id is None or not target.data_path:
        return None
    found = _CUSTOM.search(target.data_path) or _LAST_NAME.search(target.data_path)
    return None if found is None else (target.id, target.data_path, _unescape(found.group(1)))


def _holder(owner, path):
    """(the struct holding the property at ``path`` of ``owner``, its name, whether it's a custom property), or None
    when the path doesn't resolve (its add-on isn't running)."""
    custom = _CUSTOM.search(path)
    if custom is not None:
        holder_path, name = path[:custom.start()], _unescape(custom.group(1))
    else:
        holder_path, _dot, name = path.rpartition(".")
    try:
        holder = owner.path_resolve(holder_path) if holder_path else owner
    except ValueError:
        return None
    if custom is not None:
        try:
            holder[name]
        except (KeyError, TypeError):
            return None
    elif not hasattr(holder, "bl_rna") or holder.bl_rna.properties.get(name) is None:
        return None
    return holder, name, custom is not None


def _kind(holder, name, custom):
    """What kind of setting this property is: 'ENUM', 'BOOL', 'NUMBER', or None when it's no setting. Settings are
    custom properties, and properties of add-ons' property groups: not Blender's own (a scene's frame)."""
    if custom:
        value = holder[name]
        if isinstance(value, bool):
            return 'BOOL'
        return 'NUMBER' if isinstance(value, (int, float)) else None
    if not isinstance(holder, bpy.types.PropertyGroup):
        return None
    prop = holder.bl_rna.properties[name]
    if prop.type == 'ENUM':
        return None if prop.is_enum_flag else 'ENUM'
    if getattr(prop, "is_array", False):
        return None
    return {'BOOLEAN': 'BOOL', 'INT': 'NUMBER', 'FLOAT': 'NUMBER'}.get(prop.type)


# Identifiers of enums whose items a function makes: Blender doesn't list those.
_ENUM_IDENTIFIERS = {}


def _enum_identifiers(holder, name):
    """The identifiers of the items of enum ``name`` of ``holder``, in order."""
    prop = holder.bl_rna.properties[name]
    if len(prop.enum_items):
        return [item.identifier for item in prop.enum_items]
    key = (holder.bl_rna.identifier, name)
    if key not in _ENUM_IDENTIFIERS:
        identifiers = _declared_identifiers(holder, name)
        if not identifiers:
            # Assigning an item that doesn't exist is refused with the list of those that do; nothing changes. (Not
            # while drawing: then IDs can't be written at all.)
            try:
                setattr(holder, name, "\x01")
            except (TypeError, AttributeError) as error:
                found = re.search(r"not found in \((.*)\)", str(error))
                if found is not None:
                    identifiers = [_unescape(item) for item in re.findall(r"'((?:[^'\\]|\\.)*)'", found.group(1))]
        if not identifiers:
            return []
        _ENUM_IDENTIFIERS[key] = identifiers
    return _ENUM_IDENTIFIERS[key]


def _declared_identifiers(holder, name):
    """The identifiers the items function of enum ``name`` gives, for a property group an add-on declared in Python,
    read without writing anything (so also while drawing); empty for any other."""
    declared = getattr(type(holder), "__annotations__", {}).get(name)
    items = getattr(declared, "keywords", {}).get("items")
    if not callable(items):
        return []
    try:
        listed = items(holder, bpy.context)
    except Exception:  # noqa: BLE001 - another add-on's function
        return []
    return [item[0] for item in listed or () if item and item[0]]


def _enum_number(identifiers, identifier):
    """The number drivers read for enum item ``identifier``, of an enum whose items a function makes: named after
    their numbers (as the devkit's are), else numbered in order."""
    if all(item.isdigit() for item in identifiers):
        return float(identifier) if identifier.isdigit() else None
    return float(identifiers.index(identifier)) if identifier in identifiers else None


def _value(holder, name, custom):
    """The number a driver reads from the property (an enum by its item's number), or None."""
    value = holder[name] if custom else getattr(holder, name)
    if isinstance(value, (bool, int, float)):
        return float(value)
    if not isinstance(value, str) or custom:
        return None
    item = holder.bl_rna.properties[name].enum_items.get(value)
    if item is not None:
        return float(item.value)
    return _enum_number(_enum_identifiers(holder, name), value) if value else None


def _read(owner, path):
    """The number a driver reads from property ``path`` of ``owner``, or None when it can't be read."""
    found = _holder(owner, path)
    if found is not None:
        return _value(*found)
    try:
        value = owner.path_resolve(path)
    except ValueError:
        return None
    return float(value) if isinstance(value, (bool, int, float)) else None


# -----------------------------------------------------------------------------
# Yet Another Devkit's rules

class _Yab:
    """What Yet Another Devkit's script does along with its settings (``Scene.ya_devkit_props``), which its drivers
    don't show."""

    PATH = re.compile(r"ya_devkit_props\.(\w+)\.(\w+)")
    CHEST = ("Large", "Medium", "Small", "Masc")
    # With Lavabod, the chest sizes are these instead.
    LAVA_CHEST = ("Omoi", "Teardrop", "Cupcake", "Masc")
    LEG_WORDS = ("Melon", "Skull", "Yanilla", "Masc Legs", "Lava Legs", "Mini")
    MASC_CHEST = 3
    # Alt Hips (the devkit turns it off with the other leg sizes) only goes with these first ones.
    HIP_LEGS = 3
    # Switching Lavabod, the devkit gives the chest's sub-keys ('- Squeeze', '-- Sag'...) the values they had the
    # last time it was switched the other way, kept in these lists of the settings' group; the first time Lavabod
    # comes on, it takes this key off.
    KEPT_KEYS = {1: "lava_keys", 0: "yab_keys"}
    LAVA_OFF = "- Squeeze"
    # Its properties have no names: these label them.
    LABELS = {"chest_size": "Chest", "leg_size": "Legs", "hand_size": "Hands", "gen": "Gen"}
    # Sizes are named in this order, after what changes in it, then after the other settings.
    NAMED = ("chest_size", "leg_size", "lavabod", "rue", "buff")

    @classmethod
    def reads(cls, path):
        return cls.PATH.fullmatch(path) is not None

    @classmethod
    def rules(cls, size, now):
        """``size`` as the devkit would set it, the settings missing in it at their values ``now``: Lavabod has
        no Masc chest (it takes Small instead), and Alt Hips is off past Yanilla."""
        size = dict(size)
        lavabod = size.get("lavabod", now.get("lavabod"))
        if lavabod and size.get("chest_size") == cls.MASC_CHEST:
            size["chest_size"] = cls.MASC_CHEST - 1.0
        legs = size.get("leg_size", now.get("leg_size"))
        if ("alt_hips" in size or now.get("alt_hips") is not None) and legs is not None and legs >= cls.HIP_LEGS:
            size["alt_hips"] = 0.0
        return size

    @classmethod
    def chest_labels(cls, lavabod):
        return cls.LAVA_CHEST if lavabod else cls.CHEST

    @classmethod
    def words(cls, settings, size, now, varied):
        """Words naming ``size`` for the settings in NAMED: those that change from ``now`` or are ``varied``, like
        'Medium Rue', 'Teardrop Lavabod' or 'Masc Legs'; 'YAB' where it goes back from Rue or Lavabod."""
        def differs(name):
            return name in size and size[name] != now.get(name)

        def named(name):
            return name in size and (differs(name) or name in varied)

        words = []
        lavabod = size.get("lavabod", now.get("lavabod")) or 0.0
        if named("chest_size") or ("chest_size" in size and differs("lavabod")):
            chest = cls.chest_labels(lavabod)
            words.append(chest[min(int(size["chest_size"]), len(chest) - 1)])
        if named("leg_size"):
            words.append(cls.LEG_WORDS[min(int(size["leg_size"]), len(cls.LEG_WORDS) - 1)])
        if differs("lavabod") or differs("rue"):
            body = [word for name, word in (("lavabod", "Lavabod"), ("rue", "Rue"))
                    if size.get(name, now.get(name))]
            words.extend(body or ["YAB"])
        else:
            words.extend(word for name, word in (("lavabod", "Lavabod"), ("rue", "Rue"))
                         if name in varied and size.get(name))
        if differs("buff"):
            words.append("Buff" if size["buff"] else "No Buff")
        elif "buff" in varied and size.get("buff"):
            words.append("Buff")
        return words


def _is_yab(settings):
    return any(setting.kind != 'KEY' and setting.source is not None and _Yab.reads(setting.source[1])
               for setting in settings)


# -----------------------------------------------------------------------------
# Working the drivers out

def _settings_read(closure):
    """{name: (owner, path)} of the settings the drivers of Keys ``closure`` read, the first place each is read."""
    found = {}
    for keys in closure:
        for curve in _key_drivers(keys).values():
            for var in curve.driver.variables:
                setting = _setting_path(var)
                if setting is not None:
                    found.setdefault(setting[2], setting[:2])
    return found


def _work_out(keys, size):
    """The values the drivers of Key ``keys``, and of the Keys whose keys they copy, would give their keys' values
    and mutes with the settings at ``size`` ({name: value}): {(Key pointer, key name, 'value' or 'mute'): value},
    only those that change. Raises ResizeError when a driver involved can't be worked out."""
    closure = _linked_keys(keys)
    read = _settings_read(closure)
    yab = any(_Yab.reads(path) for _owner, path in read.values())
    if yab:
        now = {name: _read(*source) for name, source in read.items()}
        size = {**size, **_Yab.rules({name: value for name, value in size.items() if name in read}, now)}
    worked = {}
    for current in closure:
        blocks = current.key_blocks
        pointer = current.as_pointer()
        # Keys nothing drives that the size sets directly.
        for name, value in size.items():
            if name.startswith(KEY_PREFIX):
                block = blocks.get(name[len(KEY_PREFIX):])
                if block is not None and block != current.reference_key and not _driven(current, block, "value"):
                    worked[(pointer, block.name, 'value')] = float(value)
        if yab:
            _yab_kept_keys(current, size, worked)

    def variable(var):
        # (Its value, whether that's one of the sizes or of the keys worked out.)
        setting = _setting_path(var)
        if setting is not None and setting[2] in size:
            return float(size[setting[2]]), True
        link = _key_link(var)
        if link is not None:
            found = worked.get((link[0].as_pointer(), link[1], link[2]))
            if found is not None:
                return float(found), True
        if var.type != 'SINGLE_PROP':
            raise ValueError("it has a {:s} variable".format(var.type.replace("_", " ").lower()))
        target = var.targets[0]
        value = None if target.id is None else _read(target.id, target.data_path)
        if value is None:
            raise ValueError("'{:s}' can't be read: is the devkit's script running?".format(target.data_path))
        return value, False

    drivers = [(current, name, prop, curve) for current in closure
               for (name, prop), curve in _key_drivers(current).items()]
    # Each round works out the drivers that read a size or a key changed in the rounds before.
    for _round in range(len(drivers) + 1):
        changed = False
        for current, name, prop, curve in drivers:
            driver = curve.driver
            try:
                read_vars = []
                involved = False
                unreadable = None
                for var in driver.variables:
                    try:
                        value, part = variable(var)
                    except ValueError as error:
                        unreadable = unreadable or error
                        continue
                    involved = involved or part
                    read_vars.append((var.name, value))
                if not involved:
                    continue
                if unreadable is not None:
                    raise unreadable
                values = [value for _name, value in read_vars]
                if driver.type == 'SCRIPTED':
                    value = float(_expression_value(_parsed(driver.expression), dict(read_vars)))
                elif not values:
                    value = 0.0
                else:
                    value = {'AVERAGE': sum(values) / len(values), 'SUM': sum(values), 'MIN': min(values),
                             'MAX': max(values)}[driver.type]
            except (ValueError, SyntaxError, TypeError, KeyError, ArithmeticError) as error:
                owner = next((obj.name for obj in bpy.data.objects
                              if obj.type == 'MESH' and obj.data.shape_keys == current), current.name)
                raise ResizeError("The driver of '{:s}' on '{:s}' can't be worked out: {:s}".format(
                    name, owner, str(error))) from error
            if len(curve.keyframe_points) or len(curve.modifiers):
                value = curve.evaluate(value)
            # Blender turns a boolean on from 1.
            value = value > 1.0 - 1e-6 if prop == 'mute' else value
            key = (current.as_pointer(), name, prop)
            if worked.get(key) != value:
                worked[key] = value
                changed = True
        if not changed:
            break
    return worked


def _yab_kept_keys(keys, size, worked):
    """Switching Lavabod, the devkit gives the chest's sub-keys of ``keys`` the values kept from the last time (see
    _Yab.KEPT_KEYS): put them in ``worked``."""
    lavabod = size.get("lavabod")
    if lavabod is None:
        return
    source = _settings_read([keys]).get("lavabod")
    if source is None or not _Yab.reads(source[1]) or bool(lavabod) == bool(_read(*source)):
        return
    owner, path = source
    try:
        kept = getattr(owner.path_resolve(path.rpartition(".")[0]), _Yab.KEPT_KEYS[int(bool(lavabod))])
    except (ValueError, AttributeError):
        kept = ()
    values = {item.name: item.value for item in kept} or ({_Yab.LAVA_OFF: 0.0} if lavabod else {})
    for name, value in values.items():
        if keys.key_blocks.get(name) is not None:
            worked[(keys.as_pointer(), name, 'value')] = value


def _effective(keys, worked):
    """Each key's value of ``keys`` (a Key) with ``worked`` (see _work_out) applied: {name: value}, 0 where muted,
    the basis left out."""
    pointer = keys.as_pointer()
    result = {}
    for block in keys.key_blocks:
        if block == keys.reference_key:
            continue
        value = worked.get((pointer, block.name, 'value'), block.value)
        value = min(max(value, block.slider_min), block.slider_max)
        result[block.name] = 0.0 if worked.get((pointer, block.name, 'mute'), block.mute) else value
    return result


def size_values(obj, size, body_values=None):
    """The shape keys of mesh ``obj`` that change at ``size`` ({setting name: value}, see Setting): {key name: value},
    0 for a key that would be muted. Nothing is switched: the drivers are worked out here, with the keys they copy
    from other meshes; keys named in ``size`` (KEY_PREFIX) take their values. ``body_values`` (the keys another
    mesh changes, the body's) set the keys of ``obj`` of the same names that nothing drives. Raises ResizeError
    when a driver involved can't be worked out."""
    keys = _mesh_keys(obj)
    if keys is None or not size:
        return {}
    worked = _work_out(keys, size)
    now = _effective(keys, {})
    after = _effective(keys, worked)
    result = {name: value for name, value in after.items() if value != now[name]}
    for name, value in (body_values or {}).items():
        block = keys.key_blocks.get(name)
        if block is None or block == keys.reference_key or name in result or _driven(keys, block, "value"):
            continue
        if value != now[name]:
            result[name] = value
    return result


def follows(obj, names, key_names=()):
    """Whether mesh ``obj`` has shape keys that follow the settings ``names``: keys driven by them (or copying keys
    driven by them), or keys named in ``key_names`` that nothing drives."""
    keys = _mesh_keys(obj)
    if keys is None:
        return False
    names = set(names)
    if names & set(_settings_read(_linked_keys(keys))):
        return True
    plain = {name[len(KEY_PREFIX):] for name in names if name.startswith(KEY_PREFIX)} | set(key_names)
    return any(block.name in plain and block != keys.reference_key and not _driven(keys, block, "value")
               for block in keys.key_blocks)


# -----------------------------------------------------------------------------
# Listing a body's settings

def _pretty(name):
    """A label for a property's name: 'chest_size' -> 'Chest Size', 'CoochieType' -> 'Coochie Type'."""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name.replace("_", " ")).split()
    return " ".join(word[:1].upper() + word[1:] for word in words) or name


def _described(description):
    """Labels of numbers in a property's description, like '0-Default 1-Inverted 2-Flat': {number: label}."""
    parts = re.split(r"(?:^|\s+)(-?\d+)\s*[-:=)]\s*", description or "")
    if len(parts) < 5 or parts[0].strip():
        return {}
    labels = {float(number): label.strip().rstrip(",;").strip() for number, label in zip(parts[1::2], parts[2::2])}
    return {number: label for number, label in labels.items() if label}


def _number_range(holder, name, custom):
    """(lowest, highest, whether whole numbers) a number property takes; ends None where there are none."""
    if custom:
        try:
            ui = holder.id_properties_ui(name).as_dict()
        except (TypeError, AttributeError):
            ui = {}
        whole = isinstance(holder[name], int)
        low, high = ui.get("min"), ui.get("max")
    else:
        prop = holder.bl_rna.properties[name]
        whole = prop.type == 'INT'
        low, high = prop.hard_min, prop.hard_max
    low, high = ((float(end) if end is not None and abs(end) < 1e6 else None) for end in (low, high))
    return low, high, whole


def _options(holder, name, custom, kind, compared, current):
    """The Options of a setting (see Setting), labelled from its property where that tells."""
    if kind == 'BOOL':
        return [Option(0.0, "Off", ""), Option(1.0, "On", "")]
    if kind == 'ENUM':
        prop = holder.bl_rna.properties[name]
        if len(prop.enum_items):
            return [Option(item.value, item.name or item.identifier) for item in prop.enum_items]
        identifiers = _enum_identifiers(holder, name)
        options = []
        for identifier in identifiers:
            number = _enum_number(identifiers, identifier)
            try:
                label = bpy.types.UILayout.enum_item_name(holder, name, identifier)
            except (TypeError, ValueError):
                label = ""
            if number is not None:
                options.append(Option(number, label or identifier))
        if not options and current is not None:
            options = [Option(value, "{:g}".format(value)) for value in sorted(compared | {current})]
        return options
    low, high, whole = _number_range(holder, name, custom)
    if whole and low is not None and high is not None and high - low < MAX_RANGE:
        values = [float(value) for value in range(int(low), int(high) + 1)]
    else:
        values = sorted(value for value in compared | ({current} if current is not None else set())
                        if (low is None or value >= low) and (high is None or value <= high))
    if custom:
        try:
            description = holder.id_properties_ui(name).as_dict().get("description", "")
        except (TypeError, AttributeError):
            description = ""
    else:
        description = holder.bl_rna.properties[name].description
    labels = _described(description)
    return [Option(value, labels.get(value, "{:g}".format(value))) for value in values]


def _plain_keys(obj):
    """Settings for the keys of ``obj`` that nothing drives (a body whose keys nothing sets): its keys that change
    its shape, but the game's."""
    keys = obj.data.shape_keys
    result = []
    for block in keys.key_blocks:
        if (block == keys.reference_key or block.name.lower().startswith(GAME_KEY_PREFIX)
                or _driven(keys, block, "value")):
            continue
        if not np.any(_key_positions(block) != _key_positions(block.relative_key)):
            continue
        current = 0.0 if block.mute else float(block.value)
        result.append(Setting(KEY_PREFIX + block.name, block.name, 'KEY',
                              [Option(0.0, "Off", ""), Option(1.0, "On", "")], current, (keys, block.name)))
    return result


class _Moves:
    """How far each vertex of mesh ``obj`` moves when its keys change (world space), from each key's own move."""

    def __init__(self, obj):
        self.obj = obj
        keys = obj.data.shape_keys
        self.blocks = keys.key_blocks
        self.matrix = np.array(obj.matrix_world, dtype=np.float64)
        co = _transform(_key_positions(keys.reference_key), self.matrix)
        tris = _tris(obj.data)
        self.area = _vertex_areas(co, tris) if len(tris) else np.ones(len(co))
        self.height = co[:, 2]
        self._deltas = {}

    def delta(self, name):
        if name not in self._deltas:
            block = self.blocks[name]
            delta = _key_positions(block) - _key_positions(block.relative_key)
            weights = _group_weights(self.obj, block.vertex_group) if block.vertex_group else None
            if weights is not None:
                delta *= weights[:, None]
            self._deltas[name] = delta @ self.matrix[:3, :3].T
        return self._deltas[name]

    def measure(self, before, after):
        """(mean move over the body's area, its height) from key values ``before`` to ``after``."""
        move = None
        for name, value in after.items():
            change = value - before.get(name, 0.0)
            if change:
                step = change * self.delta(name)
                move = step if move is None else move + step
        if move is None:
            return 0.0, 0.0
        length = np.linalg.norm(move, axis=1) * self.area
        total = float(length.sum())
        if total <= 0.0:
            return 0.0, 0.0
        return total / max(float(self.area.sum()), 1e-30), float((length * self.height).sum() / total)


def body_settings(obj):
    """The settings the shape keys of mesh ``obj`` follow, and why some can't be listed.

    Returns (a list of Setting, messages): those that reshape the body much first, then the others (``more``), each
    from head to toe. A body whose keys nothing drives has its keys as settings.
    """
    keys = _mesh_keys(obj)
    if keys is None:
        return [], []
    closure = _linked_keys(keys)
    found = {}
    for current in closure:
        for curve in _key_drivers(current).values():
            driver = curve.driver
            for var in driver.variables:
                setting = _setting_path(var)
                if setting is None:
                    continue
                owner, path, name = setting
                entry = found.setdefault(name, {"sources": [], "compared": set()})
                if (owner, path) not in entry["sources"]:
                    entry["sources"].append((owner, path))
                if driver.type == 'SCRIPTED':
                    entry["compared"] |= _compared(driver.expression, var.name)
    settings, problems = [], []
    yab = False
    for name, entry in found.items():
        resolved = next(((owner, path, held) for owner, path in entry["sources"]
                         if (held := _holder(owner, path)) is not None), None)
        if resolved is None:
            problems.append("'{:s}' can't be read: is the devkit's script running?".format(entry["sources"][0][1]))
            continue
        owner, path, (holder, prop_name, custom) = resolved
        kind = _kind(holder, prop_name, custom)
        if kind is None:
            continue
        current = _value(holder, prop_name, custom)
        options = _options(holder, prop_name, custom, kind, entry["compared"], current)
        if len(options) < 2:
            continue
        if custom:
            label = prop_name
        else:
            # Blender names a property without a name after its identifier.
            prop = holder.bl_rna.properties[prop_name]
            label = prop.name if prop.name != prop.identifier else ""
        if _Yab.reads(path):
            yab = True
            label = _Yab.LABELS.get(name, label)
        settings.append(Setting(name, label or _pretty(name), kind, options, current, (owner, path)))
    if not settings and not found:
        settings = _plain_keys(obj)
    _measure(obj, settings, keys)
    if yab:
        _yab_options(settings)
    if len(settings) > FEW_SETTINGS:
        for setting in settings:
            setting.more = setting.reach < MAIN_REACH
    settings.sort(key=lambda setting: (setting.more, -setting.height))
    return settings, problems


def _measure(obj, settings, keys):
    """Measure how much each of ``settings`` reshapes ``obj`` (reach, height), and label the numbers of number
    settings that tell nothing about them with the keys they turn on."""
    moves = _Moves(obj)
    now = _effective(keys, {})
    for setting in settings:
        turned_on = {}
        for option in setting.options:
            if setting.current is not None and option.value == setting.current:
                after = now
            else:
                try:
                    after = _effective(keys, _work_out(keys, {setting.name: option.value}))
                except ResizeError:
                    continue
                reach, height = moves.measure(now, after)
                if reach > setting.reach:
                    setting.reach, setting.height = reach, height
            turned_on[option.value] = [name for name, value in after.items() if value >= ON]
        if setting.kind == 'NUMBER' and all(option.label == "{:g}".format(option.value) for option in setting.options):
            everywhere = set.intersection(*(set(names) for names in turned_on.values())) if turned_on else set()
            lowest = min(option.value for option in setting.options)
            for option in setting.options:
                names = [name for name in turned_on.get(option.value, ()) if name not in everywhere]
                if names:
                    option.label = option.word = " + ".join(names[:2])
                elif option.value == lowest:
                    option.label, option.word = "None", ""
        if setting.reach == 0.0:
            setting.height = float(np.average(moves.height, weights=moves.area)) if len(moves.height) else 0.0


def _yab_options(settings):
    """Yet Another Devkit's words for sizes: its leg sizes' (Masc Legs, Lava Legs)."""
    for setting in settings:
        if setting.name == "leg_size":
            for option in setting.options:
                index = int(option.value)
                if 0 <= index < len(_Yab.LEG_WORDS):
                    option.word = _Yab.LEG_WORDS[index]


def option_labels(name, labels, picked):
    """Labels for the options of setting ``name`` (``labels``, in order), as they read with the values picked for the
    others (``picked(name)``, a set): Yet Another Devkit's chest sizes are Lavabod's with Lavabod alone picked."""
    if name == "chest_size" and tuple(labels) in (_Yab.CHEST, _Yab.LAVA_CHEST):
        lavabod = picked("lavabod")
        if lavabod == {1.0}:
            return list(_Yab.LAVA_CHEST)
        if lavabod == {0.0}:
            return list(_Yab.CHEST)
    return list(labels)


# -----------------------------------------------------------------------------
# Sizes

def now_values(obj, names):
    """{name: value} of the settings ``names`` now, as the drivers of the shape keys of mesh ``obj`` read them (keys
    for KEY_PREFIX names, 0 where muted); None where one can't be read. Quick enough for every redraw."""
    keys = _mesh_keys(obj)
    if keys is None:
        return {name: None for name in names}
    read = None
    result = {}
    for name in names:
        if name.startswith(KEY_PREFIX):
            block = keys.key_blocks.get(name[len(KEY_PREFIX):])
            result[name] = None if block is None else 0.0 if block.mute else float(block.value)
            continue
        if read is None:
            read = _settings_read(_linked_keys(keys))
        result[name] = _read(*read[name]) if name in read else None
    return result


def offers_sizes(obj):
    """Whether mesh ``obj`` has sizes to pick: shape keys driven by settings, else keys nothing drives (see
    body_settings, which is slower)."""
    keys = _mesh_keys(obj)
    if keys is None:
        return False
    if _settings_read(_linked_keys(keys)):
        return True
    return any(block != keys.reference_key and not block.name.lower().startswith(GAME_KEY_PREFIX)
               and not _driven(keys, block, "value") for block in keys.key_blocks)


def current_values(settings):
    """{name: value} of ``settings`` now, read again (the body may have been switched since they were listed)."""
    result = {}
    for setting in settings:
        if setting.kind == 'KEY' and setting.source is not None:
            keys, name = setting.source
            block = keys.key_blocks.get(name)
            result[setting.name] = None if block is None else 0.0 if block.mute else float(block.value)
        elif setting.source is not None:
            result[setting.name] = _read(*setting.source)
        else:
            result[setting.name] = setting.current
    return result


def apply_rules(settings, size, now=None):
    """``size`` as the devkit's script would set it (see _Yab.rules)."""
    if not _is_yab(settings):
        return dict(size)
    now = {setting.name: setting.current for setting in settings} if now is None else now
    return _Yab.rules(size, now)


def combinations(settings, picks, now=None, limit=MAX_SIZES):
    """Every size the picks make: ``picks`` maps a setting's name to the values picked for it, in the order of
    ``settings``; a setting with none keeps its value ``now``. Sizes made the same by the devkit's rules are given
    once. Returns a list of {name: value}, at most ``limit``."""
    now = {setting.name: setting.current for setting in settings} if now is None else now
    names, values = [], []
    for setting in settings:
        picked = picks.get(setting.name)
        if picked:
            names.append(setting.name)
            values.append([option.value for option in setting.options if option.value in picked])
        elif now.get(setting.name) is not None:
            names.append(setting.name)
            values.append([now[setting.name]])
    sizes, seen = [], set()
    for combination in itertools.product(*values):
        size = apply_rules(settings, dict(zip(names, combination)), now)
        signature = tuple(sorted(size.items()))
        if signature in seen:
            continue
        seen.add(signature)
        sizes.append(size)
        if len(sizes) >= limit:
            break
    return sizes


def state_label(settings, values):
    """Words for the ``values`` ({name: value}) of the settings that reshape the body much (not ``more``): the options
    of enums and numbers, switches and keys that are on, like 'Medium Melon Rue'."""
    words = []
    for setting in settings:
        value = values.get(setting.name)
        if setting.more or value is None:
            continue
        if setting.kind in ('BOOL', 'KEY'):
            if value:
                words.append(setting.label)
            continue
        labels = option_labels(setting.name, [option.label for option in setting.options],
                               lambda name: {values.get(name)})
        index = next((index for index, option in enumerate(setting.options) if option.value == value), None)
        if index is not None and labels[index] != "None":
            words.append(labels[index])
    return " ".join(words)


def is_now(size, now):
    """Whether ``size`` is what the body shows ``now``."""
    return all(now.get(name) == value for name, value in size.items())


def size_label(settings, size, now=None, varied=()):
    """A name for ``size``: the words of its options for the settings it changes from ``now`` (their current values
    when None) and for those in ``varied`` (picked with several values), like 'Medium Rue'. Empty when it changes
    nothing and nothing is varied."""
    now = {setting.name: setting.current for setting in settings} if now is None else now
    size = apply_rules(settings, size, now)
    words = []
    yab = _is_yab(settings)
    if yab:
        words.extend(_Yab.words(settings, size, now, set(varied)))
    for setting in settings:
        if setting.name not in size or (yab and setting.name in _Yab.NAMED):
            continue
        value = size[setting.name]
        changed = value != now.get(setting.name)
        if not changed and setting.name not in varied:
            continue
        option = setting.option(value)
        if setting.kind in ('BOOL', 'KEY'):
            if value:
                words.append(setting.label)
            elif changed:
                words.append("No " + setting.label)
        elif option is not None and option.word:
            words.append(option.word)
        elif option is None:
            words.append("{:s} {:g}".format(setting.label, value))
    return " ".join(words)
