# Size exports write the devkit's body parts

Design for Magic Fit's Auto Resize API, 2026-10-15. Approved in conversation; implementation plan to follow.

## Why

An Instant Edit size export writes a model whole: the clothing, and the body parts the model carries (skin under the
top, piercings). Today those body parts are copies of the body at the From size, and Resize moves them to each size:
skin is bound to the body's surface, piercings move as rigid parts on it.

Measured on `[Pocky] WIP Corset Blouse Rue SML` exported from Rue M in slider_test.blend, against LavaBod+ 2.2's own
body files:

- The bound skin's positions are exact: 0.00 mm from LavaBod's body at Lava Omoi, LaRue Omoi and Lava Teardrop, zero gap
  at the waist seam, identical normals there.
- The copies keep the Rue body's bone weights at every size (belly physics bones, waist j_kosi 0.69 / j_sebo_a 0.22
  / ya_fukubu_phys 0.075), while LavaBod's bottoms, the devkit's Torso and the author's YAB sizes weight the waist
  j_kosi 0.8 / j_sebo_a 0.2. Posed, the rows skin differently: the seam and the shading line the user saw.
- The navel piercing moved as a rigid part lands 3.6 mm low and 2 mm out at every non-Rue size; the devkit's own
  "1.0 Piercing B" has a hand placement per body that lands like LavaBod's own.
- The devkit's Piercing B is not recognized as a body part, because its Rue key is driven by the Torso's Rue
  modifier visibility rather than a devkit setting, so the Resize button resized it.

Resize cannot produce weights or hand placements. The devkit's parts have them, switched by the devkit's settings:
its Torso swaps Rue weights with a Data Transfer modifier whose visibility a key drives. So size exports should write
the devkit's parts, put into the size's state the way the devkit's own UI does it, instead of moved copies.

## What changes, in one paragraph

When a caller opts in, the API matches each export mesh to the devkit part it stands for (same slot, material and
attributes, lying on it at the From size), does not resize the matched meshes, and reports the matches as
`substitutes` for the caller to export in their place. Showing a size switches the devkit's settings for every body
part involved, so keys, modifiers and dependent parts follow by their own drivers, and sets the parts' undriven keys
(sliders) as before; everything is put back afterwards. A mesh whose keys are driven through another body part counts
as a body part too. Substitution is automatic whenever a counterpart exists, with a note per match; there is no
toggle.

## Terms

- **Export meshes**: the meshes a caller exports, as Instant Edit gives them: the visible meshes of a model and the
  parts the game did not draw that a character send imported hidden.
- **Body part**: a mesh whose shape keys follow the body's settings (see sizing.follows), now also one whose keys are
  driven by a property of another body part.
- **Devkit part**: a body part outside the export set that is not the From or To body.
- **Substitute**: the devkit part an export mesh is exported as. The export mesh is then neither resized nor exported.
- **State sources**: the properties a body part's drivers read along their chain, as {setting name: (owner, path)}:
  the Torso's keys read `ya_devkit_props.torso_state.*`; Piercing B's key reads the Torso's modifier, whose
  visibility a Torso key drives, so its sources are the Torso's.

## Body-part recognition (sizing)

`sizing.follows(obj, names, key_names=())` keeps its two rules and gains a third: a key of `obj` is driven by a
property of another mesh object (a driver variable of type SINGLE_PROP whose target is an Object and whose path is
not a shape key path, like `modifiers["Rue"].show_viewport`), and that object follows `names` by any rule. A visited
set stops cycles. Objects reached this way that are not meshes do not count.

`sizing.state_sources(obj)` returns the settings read along the chain: `_settings_read` over the mesh's linked keys,
merged with the state sources of the mesh objects its drivers read, first source per name kept. Settings are those
`body_settings` would list (property groups' enums, booleans and numbers, custom properties); a modifier's
`show_viewport` is not a setting and is skipped, which is what makes the chain continue to the Torso instead.

`sizing.size_values` is unchanged: for Piercing B it returns {} (its drivers do not read a setting), which is right
because its keys follow the state switch by themselves.

The Resize button inherits the recognition: Piercing B is left out with the existing "shape keys follow the sizes"
warning instead of being resized.

## Matching (new module `magic_fit/bodyparts.py`)

`bodyparts.find_substitutes(meshes, job, names, changed)` returns ({export mesh: devkit part}, notes). For each export
mesh:

1. **Slot**: parsed from the name as Instant Edit parses it: `^(\d+)\.(\d+)(?:\s+.*)?$` or `^(.+?)\s+(\d+)\.(\d+)$`,
   after stripping a ` LODn` suffix (and Blender's `.001` after it). No slot, no substitution.
2. **Material key**: the `xiv_material` custom property, else the first material slot's name; None means no
   substitution. Normalized: the part after the last `/`, a trailing ` [hex]` suffix and a trailing `.NNN` removed,
   `.mtrl` appended when missing, casefolded. `/mt_c0201b0001_bibo.mtrl` and `mt_c0201b0001_bibo.mtrl [4f128fd5]` are
   the same key.
3. **Attributes**: the set of custom properties named `atr_*` whose value is true.
4. **Candidates**: scene meshes that are not export meshes, not `job.body` or `job.to_body`, with the same slot,
   material key and attribute set, and that are body parts (`sizing.follows(obj, names, changed)`).
5. **Shape check**: each candidate's shape at the From size, `resizing.rest_shape(candidate,
   sizing.size_values(candidate, now, body_now))` in world space, where `now` is the picked settings' current values
   and `body_now` the body's keys at them; the export mesh's shape is the one it fits the From body in
   (`resizing.from_shape`). The median distance from the export mesh's vertices to the candidate's surface must be at
   most `MATCH_DISTANCE` = 0.02 m. The closest candidate wins; within 0.1 mm of each other, one shown in the viewport
   (`hide_viewport` off, and `hide_get()` off where it is in the view layer) beats a hidden one, then the first by
   name.
6. Each devkit part substitutes at most one export mesh: a second export mesh matching the same part is left as it is,
   with a note.

Notes: "'0.1 c0201b0001_bibo' is exported as the devkit's '0.1 Torso'" per match; "No devkit part stands in for
'0.1 c0201b0001_bibo': resized as clothing" for an export mesh drawn with the game's skin or piercings material
(`resizing.is_skin_material`, plus `mt_c\d{4}b\d{4}_piercings`) that finds no candidate; nothing for other meshes.
The match is fixed for the whole export: variants the devkit swaps per size (Chains S and L share slot 1.3) are
matched once, at the From size, and that variant is written at every size. That limitation is documented.

## Showing a size (api.SizedMeshes.shown)

Inside the block, in this order, each step recorded so the end of the block undoes it in reverse, on error too:

1. **Clothing keys** as today: the size's key on, the others off; a mesh staying as it is shows its own mix.
2. **Devkit state**: the union of `state_sources` over the body parts involved, the parts in the export set and the
   substitutes alike. For each source whose setting name is in the size's values, the property is set to the value:
   an enum to the identifier whose driver number is the value (digit-named items by `str(int(value))`, otherwise by
   position), a boolean to `bool(value)`, a number keeping its type, a custom property keeping its type followed by
   `owner.update_tag()`. Sources are set in the order chest_size, leg_size, then lavabod, rue, buff, then the rest by
   name, so the devkit's own callbacks (Lavabod has no Masc chest, Alt Hips only with the first legs) see allowed
   combinations; a value the devkit's callback changes anyway is the devkit's answer. The From body's own settings
   are never touched: the Mannequin stays the From body. Driven keys are left to their drivers, nothing muted.
3. **Undriven keys** of each body part from `sizing.size_values(part, size.values, body_values)`, only keys nothing
   drives (sliders like "-- Squeeze"), so the exported part equals the Resize target. Parts whose settings are plain
   keys (`key:` names, bodies nothing drives) keep today's key-setting path for all their keys.
4. **Visibility** of substitutes: for every layer collection holding a substitute, `exclude` off and `hide_viewport`
   off; the object's `hide_viewport` off and `hide_set(False)` where it is in the view layer. Drivers of objects in an
   excluded collection never run, and a hidden object's modifiers are not evaluated.
5. `bpy.context.view_layer.update()`, so what the caller reads from the evaluated depsgraph is the size.

A source that cannot be set (its path does not resolve, the add-on is not running) falls back to today's behaviour
for that part: its driven keys set with their drivers muted, and a note.

## API (api.py)

- `VERSION = 3`.
- `Resizer(target, sizes=None, *, scene=None, body=None, substitute=False)` and
  `resize_collection(target, sizes=None, *, scene=None, body=None, progress=None, substitute=False)`. With
  `substitute`, the matches are made once when the Resizer is made and kept in `Resizer.substitutes`
  ({export mesh: devkit part}); the matched export meshes are neither parts nor clothing, so they are not resized.
  Without it, nothing changes for the caller: the copies are resized as today and `substitutes` is empty. The flag
  is an opt-in because a caller that does not know substitutes must keep getting resized copies rather than unresized
  ones.
- `SizedMeshes.substitutes`: the same map, so the caller knows which export meshes to replace while the size shows.
  `keys` holds no entry for a substituted mesh.
- `SizedMeshes.parts` keeps listing every key value the size gives each body part, substitutes included, as
  information. `shown()` sets only the undriven ones among them directly; the driven ones follow the state switch.
  The existing test expecting a part's driven keys set with their drivers muted changes to expect them following
  their drivers.
- Notes (`SizedMeshes.notes`) carry the matching notes once, like the resize-time note.
- The module docstring documents the opt-in and the export loop:

      resizer = api.Resizer(collection, substitute=True)
      objects = [resizer.substitutes.get(obj, obj) for obj in collection_meshes]
      for size in resizer.sizes:
          sized = resizer.resize(size)
          with sized.shown():
              export(objects, sized.name)

The Resize button (operator) does not substitute: it resizes selected meshes into keys and only gains the
recognition rule.

## Tests

`tests/resize.py` builds a richer test devkit:

- `devkit_part(body, name="Torso")` keeps its default name; the new tests name theirs `0.1 Torso`. It gets an
  `xiv_material` property `/mt_c0201b0001_bibo.mtrl` and a Data Transfer modifier "Rue Weights" whose `show_viewport`
  is driven by its Rue key (as the devkit does), transferring a vertex group weight from a source mesh so the
  evaluated weights differ with the modifier on.
- `devkit_piercing(part, name="1.0 Piercing B")`: a small mesh with material `/mt_c0201b0001_piercings.mtrl`, a
  "Rue" key driven (AVERAGE) by `part.modifiers["Rue Weights"].show_viewport`, moved 1 cm by the key.
- `export_copies(part, piercing)`: keyless copies at the From state named `0.1 c0201b0001_bibo` and
  `1.0 c0201b0001_piercings` with the same materials, linked to an export collection, the piercing copy hidden.

`tests/test_resize_core.py` gains:

- `test_follows_through_part`: Piercing B follows the settings through the Torso; a mesh driven by a modifier of an
  object that follows nothing does not; `state_sources` of Piercing B are the Torso's.
- `test_find_substitutes`: the skin copy matches the Torso and the piercing copy Piercing B; a different material,
  attribute set or slot does not match; a candidate 10 cm away fails the shape check; a visible candidate beats a
  hidden twin; a part is used once.
- `test_shown_switches_state`: with `substitute=True`, inside `shown()` for a size, the Torso's state property has
  the size's value, its MEDIUM key follows by its driver (not muted), the "Rue Weights" modifier visibility and the
  evaluated weights follow, Piercing B's Rue key follows, the substitutes' collection is shown; after the block the
  state properties, key values, modifier visibility, object and collection visibility are as before, also when the
  block raises.
- `test_resize_collection` extended: without `substitute` the copies get keys as today; with it they get none and
  `substitutes` names the parts; `api.VERSION == 3`.
- `test_sizes_operator` extended: a selected Piercing B is left out with the warning.

A headless check on slider_test.blend (scratchpad script, not committed): with a Resizer over the export collection
and `substitute=True`, at "Omoi Lavabod" the evaluated Torso equals LavaBod+ 2.2's Omoi top to 0.01 mm, its waist
weights are j_kosi 0.8 / j_sebo_a 0.2, Piercing B is at its YAB placement; at "Omoi Lavabod Rue" the weights are
Rue's and the placement Rue's; after the export loop the scene state is unchanged.

Tests to run: `tests/test_resize_core.py` and `tests/test_resize_ui.py` headless.

## Documentation

- Wiki Auto-Resize page (MagicFit.wiki clone): a short "Body parts from the devkit" paragraph: what is matched, the
  notes, the fixed-variant limitation, and that the From body's sliders must match the body in the model.
- `docs/development.md`: the measurements above (positions exact, weights, piercing offsets) and how to repeat them.
- `api.py` docstring as above.

## Instant Edit side (plan for a separate session in the Instant Edit repository)

Needed for the end-to-end result; not part of this implementation:

- `instant_edit/sizes.py` `export_steps`: when `api.VERSION >= 3`, make one `api.Resizer(objects,
  substitute=True)` for the export, replace each matched object by `resizer.substitutes[obj]` in the list
  `ShapeExports` is made from, and resize with `resizer.resize(size)` per size instead of `resize_collection` per
  size. Older Magic Fit versions keep the current path.
- `mesh/export.py` `ShapeExports._move_copy`: refresh the copy's vertex weights from the evaluated mesh on every
  write, mapped through the copy's source vertices, since the devkit's Rue Weights modifier flips per size and the
  copies are prepared once per session. Custom normals are already refreshed per write.
- Nothing for visibility: Magic Fit's `shown()` shows the substitutes; `_showing` keeps handling the model's own
  hidden parts.

## Out of scope

- Picking a different devkit variant per size (Chains S at Small).
- Transferring weights onto copies when no devkit part exists.
- The Mannequin's `Rue/Lava` key driver reading the leg size where the Torso's reads Lavabod (about 1 mm at LaRue).
- A panel toggle for substitution.
