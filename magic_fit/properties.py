# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)


def _poll_mesh(_self, obj):
    # The active mesh is offered too: the body is often the one last clicked, and hiding it made the
    # picker look broken. The tools refuse to use the mesh being edited as its own body.
    return obj.type == 'MESH'


# Same curves as Blender's brush falloff presets, see `falloff_weights`.
FALLOFF_ITEMS = (
    ('SMOOTH', "Smooth", "Smooth falloff", 'SMOOTHCURVE', 0),
    ('SPHERE', "Sphere", "Spherical falloff", 'SPHERECURVE', 1),
    ('ROOT', "Root", "Root falloff", 'ROOTCURVE', 2),
    ('SHARP', "Sharp", "Sharp falloff", 'SHARPCURVE', 3),
    ('LINEAR', "Linear", "Linear falloff", 'LINCURVE', 4),
    ('CONSTANT', "Constant", "No falloff", 'NOCURVE', 5),
)

# The sidebar's Weights tab has a sub-tab per mode of the Weight Brushes; General has Copy, Straighten
# and Smooth, and brings back the last of them.
GENERAL_MODES = {'COPY', 'STRAIGHTEN', 'SMOOTH'}
WEIGHTS_TABS = (
    ('GENERAL', "General",
     "Give meshes the body's weights, then touch them up with the Copy, Straighten or Smooth brush", 0),
    ('SKIRT', "Skirt", "Make skirts and long dresses swing with the skirt bones", 1),
    ('HEELS', "Heels", "Keep high heels and boots stiff below the ankle", 2),
    ('HAIR', "Hair", "Make hair move with FFXIV's hair bones, using the hair skeleton that fits it best", 3),
    ('FACE', "Face",
     "Give a custom face the weights of the game face it replaces: lids that blink, a mouth that moves, "
     "and lashes and piercings that follow", 4),
)
_WEIGHTS_TAB_OF_MODE = {'SKIRT': 1, 'HEELS': 2, 'HAIR': 3, 'FACE': 4}
# Modes that are sub-tabs already but not brushes yet.
COMING_MODES = {}
# The races hair can be for (Hair: their hair skeletons and head shapes, see hairing.py).
HAIR_RACES = (
    ('c0101', "Hyur Midlander Male", "Hair for Hyur Midlander males (c0101)"),
    ('c0201', "Hyur Midlander Female", "Hair for Hyur Midlander females (c0201)"),
    ('c0301', "Hyur Highlander Male", "Hair for Hyur Highlander males (c0301)"),
    ('c0401', "Hyur Highlander Female", "Hair for Hyur Highlander females (c0401)"),
    ('c0501', "Elezen Male", "Hair for Elezen males (c0501)"),
    ('c0601', "Elezen Female", "Hair for Elezen females (c0601)"),
    ('c0701', "Miqo'te Male", "Hair for Miqo'te males (c0701)"),
    ('c0801', "Miqo'te Female", "Hair for Miqo'te females (c0801)"),
    ('c0901', "Roegadyn Male", "Hair for Roegadyn males (c0901)"),
    ('c1001', "Roegadyn Female", "Hair for Roegadyn females (c1001)"),
    ('c1101', "Lalafell Male", "Hair for Lalafell males (c1101)"),
    ('c1201', "Lalafell Female", "Hair for Lalafell females (c1201)"),
    ('c1301', "Au Ra Male", "Hair for Au Ra males (c1301)"),
    ('c1401', "Au Ra Female", "Hair for Au Ra females (c1401)"),
    ('c1501', "Hrothgar Male", "Hair for Hrothgar males (c1501)"),
    ('c1601', "Hrothgar Female", "Hair for Hrothgar females (c1601)"),
    ('c1701', "Viera Male", "Hair for Viera males (c1701)"),
    ('c1801', "Viera Female", "Hair for Viera females (c1801)"),
)
# The races faces can be for (Face: the game's faces, face skeletons and face animations, see facing.py).
FACE_RACES = tuple((code, name, "Faces of {:s}s ({:s})".format(name, code)) for code, name, _description in HAIR_RACES)


def _mode_changed(self, _context):
    if self.mode in GENERAL_MODES and self.general_mode != self.mode:
        self.general_mode = self.mode


def _get_use_keys(self):
    if self.resize_use_keys_state < 0:
        return bool(self.resize_from_key or self.resize_to_key)
    return bool(self.resize_use_keys_state)


def _set_use_keys(self, value):
    self.resize_use_keys_state = int(value)


def _get_weights_tab(self):
    return _WEIGHTS_TAB_OF_MODE.get(self.mode, 0)


def _set_weights_tab(self, value):
    if value == 0:
        if self.mode not in GENERAL_MODES:
            self.mode = self.general_mode
        return
    mode = next(mode for mode, tab in _WEIGHTS_TAB_OF_MODE.items() if tab == value)
    if self.mode != mode:
        self.mode = mode


class MagicFitSettings(bpy.types.PropertyGroup):
    mode: EnumProperty(
        name="Mode",
        description="What the brush does to the weights",
        items=(
            ('COPY', "Copy", "Copy weights from the nearest spot on the target mesh"),
            ('STRAIGHTEN', "Straighten",
             "Click where clothing sags into a crease, like across the cleavage, under the breasts or between "
             "the buttocks. The sagging part then stays stretched across the gap in every pose"),
            # Items are stored by position: new ones go last.
            ('SKIRT', "Skirt",
             "Paint skirt weights: the cloth swings with the skirt bones, but still follows the body where "
             "it lies on it"),
            ('HEELS', "Heels",
             "Paint heel weights: the foot stays stiff and follows the ankle instead of bending at the toes"),
            ('HAIR', "Hair",
             "Paint hair weights: hair on the scalp follows the head, hair that hangs follows the nearest "
             "hair bones"),
            ('FACE', "Face",
             "Paint toward the weights of the game face it replaces, as Face Weights works them out"),
            ('SMOOTH', "Smooth",
             "Smooth all bone weights at once, even across gaps between parts that touch. Doesn't need a body"),
        ),
        default='COPY',
        update=_mode_changed,
    )
    general_mode: EnumProperty(
        name="General Mode",
        description="The brush mode last used on the General tab",
        items=(
            ('COPY', "Copy", ""),
            ('STRAIGHTEN', "Straighten", ""),
            ('SMOOTH', "Smooth", ""),
        ),
        default='COPY',
    )
    weights_tab: EnumProperty(
        name="Weights",
        description="Which kind of weights to work on. The Weight Brushes paint these too",
        items=WEIGHTS_TABS,
        get=_get_weights_tab,
        set=_set_weights_tab,
    )
    target: PointerProperty(
        name="Target",
        description=(
            "The body mesh all tools work with: weights are copied from it, clothing is fitted to it and "
            "clipping is checked against it"
        ),
        type=bpy.types.Object,
        poll=_poll_mesh,
    )
    group_mode: EnumProperty(
        name="Groups",
        description="Which vertex groups the brush changes",
        items=(
            ('ALL', "All Groups",
             "Change all bone weights. Copy also adds groups the painted mesh is missing"),
            ('ACTIVE', "Current Group",
             "Only change the current vertex group; the other bones make room for it. Copy takes it from the "
             "target group with the same name"),
        ),
        default='ALL',
    )

    radius: IntProperty(
        name="Radius",
        description="Radius of the brush in pixels (F)",
        subtype='PIXEL',
        default=50,
        min=1,
        max=5000,
        soft_max=500,
    )
    strength: FloatProperty(
        name="Strength",
        description=(
            "How strongly the brush changes the weights (Shift F). At 1, the change is applied fully under "
            "the brush center"
        ),
        subtype='FACTOR',
        default=0.5,
        min=0.0,
        max=1.0,
    )
    use_pressure_radius: BoolProperty(
        name="Radius Pressure",
        description="Scale the radius with tablet pressure",
        default=False,
    )
    use_pressure_strength: BoolProperty(
        name="Strength Pressure",
        description="Scale the strength with tablet pressure",
        default=True,
    )

    falloff: EnumProperty(
        name="Falloff",
        description="How the strength fades from the brush center to its edge",
        items=FALLOFF_ITEMS,
        default='SMOOTH',
    )
    falloff_shape: EnumProperty(
        name="Falloff Shape",
        description="Region of the mesh the brush affects",
        items=(
            ('SPHERE', "Sphere", "Affect vertices inside a ball around the point under the cursor"),
            ('PROJECTED', "Projected", "Affect all vertices inside the brush circle on screen, front and back"),
        ),
        default='SPHERE',
    )
    use_front_faces_only: BoolProperty(
        name="Front Faces Only",
        description="Only affect vertices that face the camera",
        default=False,
    )
    spacing: IntProperty(
        name="Spacing",
        description="Distance between brush dabs, as a percentage of the radius",
        subtype='PERCENTAGE',
        default=20,
        min=1,
        max=100,
    )
    use_accumulate: BoolProperty(
        name="Accumulate",
        description=(
            "Keep building up the effect while the brush passes over the same spot. Off: one stroke never "
            "goes further than Strength"
        ),
        default=False,
    )
    use_auto_smooth: BoolProperty(
        name="Auto Smooth",
        description=(
            "Copy: even out the copied weights, so they change gradually instead of jumping at creases or "
            "hard edges"
        ),
        default=False,
    )
    auto_smooth: FloatProperty(
        name="Auto Smooth Intensity",
        description=(
            "How much the copied weights are evened out. 0.5 spreads a hard edge over about 3 edges, 1 over "
            "about 6"
        ),
        subtype='FACTOR',
        default=0.5,
        min=0.0,
        max=1.0,
    )

    sample_mode: EnumProperty(
        name="Sample",
        description="How weights are read from the target mesh",
        items=(
            ('SURFACE', "Nearest Surface",
             "Blend the weights at the nearest spot on the target's surface"),
            ('VERTEX', "Nearest Vertex",
             "Use the weights of the nearest target vertex"),
        ),
        default='SURFACE',
    )
    use_max_distance: BoolProperty(
        name="Limit Distance",
        description="Skip vertices that are farther from the target than Max Distance",
        default=False,
    )
    max_distance: FloatProperty(
        name="Max Distance",
        description="Vertices farther from the target than this are left alone",
        subtype='DISTANCE',
        default=0.05,
        min=0.0,
        soft_max=1.0,
    )
    skirt_blend_start: FloatProperty(
        name="Blend Start",
        description=(
            "Height above the hips where the skirt bones start to take over. Nothing above it gets skirt weights"
        ),
        subtype='DISTANCE',
        default=0.0,
        min=-0.5,
        soft_min=-0.2,
        soft_max=0.5,
        precision=3,
    )
    skirt_blend_length: FloatProperty(
        name="Blend Length",
        description="How far below Blend Start the skirt bones take over fully",
        subtype='DISTANCE',
        default=0.40,
        min=0.001,
        soft_max=1.0,
        precision=3,
    )
    skirt_skin_weight: FloatProperty(
        name="Skin Weight",
        description=(
            "How much body weight to keep where the cloth touches the body, so it still follows the thighs and "
            "butt when they're scaled (Customize+). 0: hang it all from the skirt bones, like the game's skirts"
        ),
        subtype='FACTOR',
        default=1.0,
        min=0.0,
        max=1.0,
    )
    skirt_skin_distance: FloatProperty(
        name="Skin Distance",
        description="Skin Weight fades out to nothing at this distance from the body",
        subtype='DISTANCE',
        default=0.07,
        min=0.0,
        soft_max=0.3,
        precision=3,
    )
    skirt_body_weight: FloatProperty(
        name="Body Weight",
        description=(
            "Body weight kept all over the skirt, so it also follows the legs a little instead of only swinging"
        ),
        subtype='FACTOR',
        default=0.0,
        min=0.0,
        max=1.0,
    )
    skirt_spread: FloatProperty(
        name="Spread",
        description=(
            "How far each chain of skirt bones reaches sideways. 1: each spot follows only the two chains "
            "beside it. Higher blends in more neighbors"
        ),
        default=1.5,
        min=1.0,
        max=2.0,
    )
    skirt_joint_blend: FloatProperty(
        name="Joint Blend",
        description="How smoothly the skirt bones blend into each other at their joints. Higher: softer bends",
        subtype='DISTANCE',
        default=0.10,
        min=0.0,
        soft_max=0.3,
        precision=3,
    )
    skirt_max_groups: IntProperty(
        name="Max Groups",
        description="Most bones a vertex can follow; the smallest weights are dropped. FFXIV allows 8",
        default=8,
        min=1,
        max=32,
        soft_max=8,
    )
    skirt_apply_to_selected: BoolProperty(
        name="Apply to All Selected",
        description="Give skirt weights to every selected mesh, not only the active one",
        default=False,
    )
    heels_ankle_height: FloatProperty(
        name="Ankle Height",
        description=(
            "Height above the ankle joint where the calf and the ankle share the weight half and half. Below "
            "the blend, the foot follows the ankle only"
        ),
        subtype='DISTANCE',
        default=0.02,
        min=-0.2,
        soft_min=-0.05,
        soft_max=0.15,
        precision=3,
    )
    heels_ankle_blend: FloatProperty(
        name="Ankle Blend",
        description="Height over which the calf hands over to the ankle. Higher: a softer change",
        subtype='DISTANCE',
        default=0.12,
        min=0.001,
        soft_max=0.4,
        precision=3,
    )
    heels_toe_bend: FloatProperty(
        name="Toe Bend",
        description=(
            "How much the toes bend at the ball of the foot, for flat shoes and boots. 0: the whole foot stays "
            "stiff, as high heels need"
        ),
        subtype='FACTOR',
        default=0.0,
        min=0.0,
        max=1.0,
    )
    heels_apply_to_selected: BoolProperty(
        name="Apply to All Selected",
        description="Give heel weights to every selected mesh, not only the active one",
        default=False,
    )
    hair_race: EnumProperty(
        name="Race",
        description="The race and gender the hair is for. This decides its hair skeletons and head shape",
        items=HAIR_RACES,
        default='c0201',
    )
    hair_skeleton_mode: EnumProperty(
        name="Skeleton",
        description="Which hair skeleton the hair uses. In the game, the hair's EST entry picks it",
        items=(
            ('AUTO', "Best Match",
             "Use the race's hair skeleton that fits the hair best"),
            ('PICK', "Pick", "Use the EST entry picked below. Type its number, or search by number or by the "
                             "game's hairstyles that use it"),
        ),
        default='AUTO',
    )
    hair_skeleton: IntProperty(
        name="Hair Skeleton",
        description="The hair skeleton to use, which is also the hair's EST entry. 0: only the body's own hair bones",
        default=0,
        min=0,
        max=9999,
    )
    hair_hang_start: FloatProperty(
        name="Hang Start",
        description=(
            "Hair closer to the scalp than this moves with the head only. Farther out, more and more of it "
            "hangs from the hair bones"
        ),
        subtype='DISTANCE',
        default=0.025,
        min=0.0,
        soft_max=0.1,
        precision=3,
    )
    hair_hang_length: FloatProperty(
        name="Hang Length",
        description="How far past Hang Start most of the hair hangs from the hair bones. Higher: a softer change",
        subtype='DISTANCE',
        default=0.07,
        min=0.005,
        soft_max=0.3,
        precision=3,
    )
    hair_neck_weight: FloatProperty(
        name="Neck Weight",
        description=(
            "How much hair below the head follows the neck and upper back, so it turns less with the head and "
            "stays out of the neck"
        ),
        subtype='FACTOR',
        default=0.3,
        min=0.0,
        max=1.0,
    )
    hair_chain_stretch: FloatProperty(
        name="Chain Stretch",
        description=(
            "For hair longer than its chain of bones. 0: bend only at the bones' own joints. Toward 1: bend "
            "evenly along the whole length instead of swinging from the last bone"
        ),
        subtype='FACTOR',
        default=0.5,
        min=0.0,
        max=1.0,
    )
    hair_joint_blend: FloatProperty(
        name="Joint Blend",
        description="How smoothly the hair bones blend into each other at their joints. Higher: softer bends",
        subtype='DISTANCE',
        default=0.30,
        min=0.0,
        soft_max=0.5,
        precision=3,
    )
    hair_ears: BoolProperty(
        name="Ears",
        description=(
            "Miqo'te only: cat ears in the hair move with the ear bones, like the game's Miqo'te hair. "
            "Hair without ears isn't affected"
        ),
        default=True,
    )
    hair_max_groups: IntProperty(
        name="Max Groups",
        description="Most bones a vertex can follow; the smallest weights are dropped. FFXIV allows 8",
        default=8,
        min=1,
        max=32,
        soft_max=8,
    )
    hair_apply_to_selected: BoolProperty(
        name="Apply to All Selected",
        description="Treat all selected meshes as one hair: find one skeleton for all of them and weight each",
        default=False,
    )
    face_source: EnumProperty(
        name="Game Face",
        description="Which game face the custom face replaces. It gets that face's weights and bones",
        items=(
            ('DETECT', "Detect",
             "Find the game face on its own, from names like c0801f0002 or else from the face's shape. The "
             "result is shown below"),
            ('PICK', "Pick", "Use the race and face picked below"),
        ),
        default='DETECT',
    )
    face_race: EnumProperty(
        name="Race",
        description="The race and gender of the game face being replaced",
        items=FACE_RACES,
        default='c0201',
    )
    face_number: IntProperty(
        name="Face",
        description="The number of the game face being replaced (f0001 is 1)",
        default=1,
        min=1,
        max=9999,
    )
    # The game face Detect found last (race code and number, "c0801 2"), shown on the Face panel.
    face_found: StringProperty(default="", options={'HIDDEN'})
    face_close_eyes: BoolProperty(
        name="Close Eyes",
        description=(
            "Make the lids close fully in the game's blink, even when the eyes are bigger or smaller than the "
            "game's"
        ),
        default=True,
    )
    face_match_neck: BoolProperty(
        name="Match Neck",
        description="Copy the game's neck edge onto the face, so no seam shows where the face meets the body",
        default=True,
    )
    face_lash_lift: FloatProperty(
        name="Lash Lift",
        description=(
            "How far out from the eyeball Snap Lashes puts the lash roots. Raise it if the lid pokes through "
            "the lashes"
        ),
        subtype='DISTANCE',
        default=0.0,
        soft_min=-0.001,
        soft_max=0.002,
        precision=4,
    )
    face_max_groups: IntProperty(
        name="Max Groups",
        description="Most bones a vertex can follow; the smallest weights are dropped. FFXIV allows 8",
        default=8,
        min=1,
        max=32,
        soft_max=8,
    )
    smooth_reach: FloatProperty(
        name="Reach",
        description=(
            "How far around each vertex the weights are averaged. Small: gentle, like Blender's Smooth brush. "
            "Large: evens out a wide area"
        ),
        subtype='DISTANCE',
        default=0.02,
        min=0.0005,
        soft_max=0.2,
        max=1.0,
        precision=3,
    )
    smooth_merge_range: FloatProperty(
        name="Merge Range",
        description=(
            "Separate parts closer than this count as connected, so the brush smooths across the gap "
            "(sleeves, layers, body parts). 0: only vertices at the exact same spot"
        ),
        subtype='DISTANCE',
        default=0.002,
        min=0.0,
        soft_max=0.02,
        precision=4,
    )
    smooth_max_groups: IntProperty(
        name="Max Groups",
        description="Most bones a vertex can follow; the smallest weights are dropped. FFXIV allows 8",
        default=8,
        min=1,
        max=32,
        soft_max=8,
    )
    use_fade_unmatched: BoolProperty(
        name="Fade Unmatched Groups",
        description=(
            "All Groups: also fade out groups the target doesn't have, so painted vertices get exactly the "
            "target's weights. Locked groups are never changed"
        ),
        default=True,
    )
    use_bones_only: BoolProperty(
        name="Deform Bones Only",
        description=(
            "All Groups: only copy the vertex groups of bones. Other groups, like masks for shape keys or "
            "physics, are left alone"
        ),
        default=True,
    )


class HairWeightsState(bpy.types.PropertyGroup):
    """What Hair Weights gave a hair (Object.magic_fit_hair): the brush paints with the same skeleton."""
    race: StringProperty(
        name="Race",
        description="The race the hair got its weights for",
        default="",
    )
    skeleton: IntProperty(
        name="Hair Skeleton",
        description="The hair skeleton (EST entry) the hair got its weights with. -1: none yet",
        default=-1,
        min=-1,
    )
    matches: StringProperty(
        name="Best Matches",
        description="The hair skeletons that fit the hair best, best first (number:score)",
        default="",
    )


class BodyFitBrushSettings(bpy.types.PropertyGroup):
    # The body itself is the shared one (`Scene.magic_fit.target`).
    mode: EnumProperty(
        name="Mode",
        description="Which way the brush moves the mesh",
        items=(
            ('PUSH', "Push Out",
             "Push the mesh out where it's inside the body or closer than Offset (Ctrl: Tighten)"),
            ('TIGHTEN', "Tighten",
             "Pull the mesh in where it floats farther out than Offset (Ctrl: Push Out)"),
            ('FIT', "Fit", "Move the mesh to Offset from either side"),
        ),
        default='FIT',
    )
    offset: FloatProperty(
        name="Offset",
        description="Gap to keep between the body and the mesh's innermost layer",
        subtype='DISTANCE',
        default=0.001,
        min=0.0,
        soft_max=0.05,
        precision=4,
    )

    radius: IntProperty(
        name="Radius",
        description="Radius of the brush in pixels (F)",
        subtype='PIXEL',
        default=60,
        min=1,
        max=5000,
        soft_max=500,
    )
    strength: FloatProperty(
        name="Strength",
        description="How far each dab moves the mesh toward its goal (Shift F)",
        subtype='FACTOR',
        default=0.5,
        min=0.0,
        max=1.0,
    )
    use_pressure_radius: BoolProperty(
        name="Radius Pressure",
        description="Scale the radius with tablet pressure",
        default=False,
    )
    use_pressure_strength: BoolProperty(
        name="Strength Pressure",
        description="Scale the strength with tablet pressure",
        default=True,
    )
    falloff: EnumProperty(
        name="Falloff",
        description="How the strength fades from the brush center to its edge",
        items=FALLOFF_ITEMS,
        default='SMOOTH',
    )
    spacing: IntProperty(
        name="Spacing",
        description="Distance between brush dabs, as a percentage of the radius",
        subtype='PERCENTAGE',
        default=15,
        min=1,
        max=100,
    )
    use_auto_smooth: BoolProperty(
        name="Auto Smooth",
        description=(
            "Smooth the mesh under the brush as you go, so it doesn't get bumpy. Layers keep their thickness. "
            "High values also soften small details"
        ),
        default=True,
    )
    auto_smooth: FloatProperty(
        name="Auto Smooth Intensity",
        description="How strongly each dab smooths the mesh under the brush",
        subtype='FACTOR',
        default=0.5,
        min=0.0,
        max=1.0,
    )

    layer_radius: FloatProperty(
        name="Keep Together",
        description=(
            "Layers, folds and details closer together than this move as one piece and keep their shape. "
            "Larger keeps bigger folds intact, but makes the brush less precise"
        ),
        subtype='DISTANCE',
        default=0.02,
        min=0.0001,
        soft_max=0.1,
    )
    seam_distance: FloatProperty(
        name="Seam Distance",
        description=(
            "Vertices closer together than this always move together, even across parts and objects, so "
            "seams stay closed. The mesh isn't merged"
        ),
        subtype='DISTANCE',
        default=0.0001,
        min=0.0,
        soft_max=0.001,
        precision=5,
    )
    use_max_distance: BoolProperty(
        name="Limit Distance",
        description="Tighten and Fit leave parts alone that are farther from the body than Max Distance",
        default=True,
    )
    max_distance: FloatProperty(
        name="Max Distance",
        description="Parts farther from the body than this count as loose (skirts, capes...) and aren't pulled in",
        subtype='DISTANCE',
        default=0.05,
        min=0.0,
        soft_max=0.5,
    )
    use_selected_only: BoolProperty(
        name="Selected Only",
        description="Only move selected vertices (hidden vertices are never moved)",
        default=False,
    )

    # Resize: the From shape is the From body, the To shape the To body (or the From body again); with
    # Use Shapekeys, each with one of its shape keys.
    resize_from_object: PointerProperty(
        name="From Body",
        description="The body the meshes fit now",
        type=bpy.types.Object,
        poll=_poll_mesh,
    )
    resize_use_keys: BoolProperty(
        name="Use Shapekeys",
        description=(
            "Resize from or to shape keys of the bodies (picked below), like the Mannequin's Rue key. Off: use "
            "the bodies as they look now"
        ),
        get=_get_use_keys,
        set=_set_use_keys,
    )
    # Use Shapekeys as set: 0 or 1, or -1 where it never was (then it's on where a key is picked, as in
    # files from before it).
    resize_use_keys_state: IntProperty(default=-1, min=-1, max=1, options={'HIDDEN'})
    resize_from_key: StringProperty(
        name="From Shapekey",
        description="The From body's shape key the meshes fit now. Empty or Basis: the body as it looks now",
    )
    resize_to_object: PointerProperty(
        name="To Body",
        description=(
            "The body to fit the meshes to: a copy of the From body in another shape, or another body with the "
            "same UV layout (like Rue for RueXB). Empty: the From body"
        ),
        type=bpy.types.Object,
        poll=_poll_mesh,
    )
    resize_to_key: StringProperty(
        name="To Shapekey",
        description="The shape key to fit the meshes to. Empty or Basis: the body as it looks now",
    )
    resize_radius: FloatProperty(
        name="Smoothing",
        description=(
            "How much the body's change is smoothed out. Smaller keeps small details like the navel, but "
            "clothing over deep creases (under large breasts) can fold"
        ),
        subtype='DISTANCE',
        default=0.01,
        min=0.001,
        soft_max=0.05,
        precision=3,
    )
    resize_bridge: FloatProperty(
        name="Bridge Creases",
        description=(
            "Let clothing span creases of the body, like the cleavage and under the breasts, instead of sinking "
            "into them. Larger spans wider creases. 0: follow the skin everywhere"
        ),
        subtype='DISTANCE',
        default=0.08,
        min=0.0,
        soft_max=0.2,
        precision=3,
    )
    resize_rigid_size: FloatProperty(
        name="Rigid Parts",
        description=(
            "Separate parts smaller than this, like rings, studs, buttons and buckles, move as one piece "
            "without changing shape. 0: none"
        ),
        subtype='DISTANCE',
        default=0.04,
        min=0.0,
        soft_max=0.2,
        precision=3,
    )
    resize_keep_close: BoolProperty(
        name="Keep Close",
        description=(
            "Afterwards, move clothing that ended up floating off the new body or sinking into it back to its "
            "old distance from the skin"
        ),
        default=True,
    )
    resize_key_name: StringProperty(
        name="Key Name",
        description="Name of the shape key added to each mesh. Empty: named after the To shape",
    )

    # Line Up lines the selected model up with the shared body.
    lineup_stretch: BoolProperty(
        name="Stretch Limbs",
        description=(
            "Stretch the arms and legs lengthwise so the model's joints land on the body's. Off: keep the "
            "model's own limb lengths and only turn the limbs"
        ),
        default=False,
    )


class TextureRelaxBrushSettings(bpy.types.PropertyGroup):
    boundary: EnumProperty(
        name="Open Edges",
        description="What vertices on an open edge of the mesh do (a border, not a UV seam)",
        items=(
            ('SLIDE', "Slide", "Slide along the edge, keeping its shape"),
            ('FREE', "Free", "Move freely, so the outline can shrink or grow"),
            ('FIXED', "Keep", "Stay where they are"),
        ),
        default='SLIDE',
    )

    radius: IntProperty(
        name="Radius",
        description="Radius of the brush in pixels (F)",
        subtype='PIXEL',
        default=60,
        min=1,
        max=5000,
        soft_max=500,
    )
    strength: FloatProperty(
        name="Strength",
        description="How far each dab evens out the texture (Shift F)",
        subtype='FACTOR',
        default=0.5,
        min=0.0,
        max=1.0,
    )
    use_pressure_radius: BoolProperty(
        name="Radius Pressure",
        description="Scale the radius with tablet pressure",
        default=False,
    )
    use_pressure_strength: BoolProperty(
        name="Strength Pressure",
        description="Scale the strength with tablet pressure",
        default=True,
    )
    falloff: EnumProperty(
        name="Falloff",
        description="How the strength fades from the brush center to its edge",
        items=FALLOFF_ITEMS,
        default='SMOOTH',
    )
    spacing: IntProperty(
        name="Spacing",
        description="Distance between brush dabs, as a percentage of the radius",
        subtype='PERCENTAGE',
        default=15,
        min=1,
        max=100,
    )
    seam_distance: FloatProperty(
        name="Seam Distance",
        description=(
            "Vertices closer together than this always move together, even across parts and objects, so "
            "seams stay closed"
        ),
        subtype='DISTANCE',
        default=0.0001,
        min=0.0,
        soft_max=0.001,
        precision=5,
    )
    use_selected_only: BoolProperty(
        name="Selected Only",
        description="Only move selected vertices (hidden vertices are never moved)",
        default=False,
    )


def _clipping_style_changed(self, context):
    from . import clipping
    clipping.style_changed(self, context)


class ClippingSettings(bpy.types.PropertyGroup):
    # Checked against the shared body (`Scene.magic_fit.target`).
    show: BoolProperty(
        name="Show Clipping",
        description=(
            "Mark where the meshes clip into the body, in their current pose. Checks the meshes being edited "
            "or painted, or the selected ones in Object and Pose Mode"
        ),
        default=False,
        update=_clipping_style_changed,
    )
    margin: FloatProperty(
        name="Margin",
        description="Also mark, more faintly, where the mesh is outside the body but closer than this",
        subtype='DISTANCE',
        default=0.0,
        min=0.0,
        soft_max=0.01,
        precision=4,
        update=_clipping_style_changed,
    )
    color: FloatVectorProperty(
        name="Color",
        description="Color and opacity of the clipping marks",
        subtype='COLOR_GAMMA',
        size=4,
        default=(1.0, 0.1, 0.85, 0.7),
        min=0.0,
        max=1.0,
        update=_clipping_style_changed,
    )
    show_in_front: BoolProperty(
        name="In Front",
        description=(
            "Show the marks through everything, like clipping of inner layers under outer ones or on the far "
            "side of the body"
        ),
        default=False,
        update=_clipping_style_changed,
    )


classes = (
    MagicFitSettings,
    HairWeightsState,
    BodyFitBrushSettings,
    TextureRelaxBrushSettings,
    ClippingSettings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.magic_fit = PointerProperty(type=MagicFitSettings)
    bpy.types.Scene.body_fit_brush = PointerProperty(type=BodyFitBrushSettings)
    bpy.types.Scene.texture_relax_brush = PointerProperty(type=TextureRelaxBrushSettings)
    bpy.types.Scene.magic_fit_clipping = PointerProperty(type=ClippingSettings)
    bpy.types.Object.magic_fit_hair = PointerProperty(type=HairWeightsState)


def unregister():
    del bpy.types.Object.magic_fit_hair
    del bpy.types.Scene.magic_fit_clipping
    del bpy.types.Scene.texture_relax_brush
    del bpy.types.Scene.body_fit_brush
    del bpy.types.Scene.magic_fit
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
