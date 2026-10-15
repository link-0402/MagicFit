# Devkit Body Parts in Size Exports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Size exports write the devkit's own body parts, switched to each size by the devkit's settings, instead of resized copies.

**Architecture:** `sizing` learns to follow drivers through other meshes (Piercing B reads the Torso's modifier) and to read, set and put back settings; a new `bodyparts` module matches export meshes to devkit parts; `api.Resizer(substitute=True)` keeps the matches and `SizedMeshes.shown()` switches the devkit's settings, sets sliders, shows the substitutes and puts everything back.

**Tech Stack:** Blender 5.2 Python (bpy, NumPy, mathutils.bvhtree), headless test scripts.

**Spec:** `docs/superpowers/specs/2026-10-15-devkit-body-parts-design.md`

## Global Constraints

- `api.VERSION = 3`; `substitute=False` by default on `Resizer` and `resize_collection`; without it callers see today's behaviour.
- `bodyparts.MATCH_DISTANCE = 0.02` m (median distance); ties within 0.1 mm.
- Settings order when switching: chest_size, leg_size, lavabod, rue, buff, then the rest by name (`sizing._Yab.NAMED`).
- The From body's own settings are never set. Driven keys are never muted on the new path.
- Repository files are LF; write files with Write/Edit only.
- Tests: `blender -b --factory-startup --python tests/test_resize_core.py` and `tests/test_resize_ui.py`.

## Review Focus

- The devkit's update callbacks change other settings (Masc chest turns Lavabod off) and sub-keys (Lavabod swaps "-" keys and its kept lists): after the block every setting, slider and kept list is as before. Test: a `_masc_lavabod`-like callback on the test devkit's `chest_size`, a size that triggers it, state compared after.
- A part reading a setting that can't be set (path doesn't resolve) or the From body's own settings: its keys are set as today (drivers muted), with a note only for the first. Test in `test_shown_switches_state`.
- Substitutes in an excluded or hidden collection: shown for the block (drivers run, modifiers evaluate) and hidden/excluded again afterwards, driven values evaluated back before they are hidden again. Test in `test_shown_switches_state`.
- Names and materials as Instant Edit writes them: ` LOD0`, `.001`, `[hex]` material suffixes. Test in `test_find_substitutes`.
- The size the body shows (`Size.now`) and the To shape without sizes (`size None`): nothing switched that needn't be. Covered by `test_resize_collection` (`again`).

---

### Task 1: sizing recognizes parts driven through other parts, and switches settings

**Files:** Modify `magic_fit/sizing.py`; Test `tests/resize.py`, `tests/test_resize_core.py`

**Interfaces (produces):**
- `sizing._read_meshes(closure) -> list[Object]`: other mesh objects whose non-shape-key properties the drivers of Keys `closure` read.
- `sizing.follows(obj, names, key_names=(), _seen=None) -> bool` (third rule).
- `sizing.state_sources(obj) -> {name: (owner, path)}`; `sizing.chain_keys(obj) -> [Key]`.
- `sizing.can_set(owner, path) -> bool`; `sizing.set_setting(owner, path, value) -> bool`; `sizing.keep_settings(sources) -> callable` (puts back the sources' values, repeating passes while callbacks change them, and the YAB kept sub-key lists).
- `tests/resize.py`: `devkit_part(body, name="Torso")` with `xiv_material` and a driven "Rue Weights" Data Transfer modifier; `devkit_piercing(part, name="1.0 Piercing B")`; `export_copies(part, piercing)`; test `DevkitState.chest_size` gets the devkit's `_masc_lavabod` callback.

- [ ] Write `test_follows_through_part` (Piercing B follows through the Torso; a mesh driven by a modifier of a mesh that follows nothing does not; `state_sources(piercing) == state_sources(torso)`); run, see it fail.
- [ ] Implement; run `test_resize_core.py`, all pass.

### Task 2: bodyparts.find_substitutes

**Files:** Create `magic_fit/bodyparts.py`; Test `tests/test_resize_core.py`

**Interfaces (produces):** `bodyparts.slot(name) -> (int, int) | None`, `bodyparts.material_key(obj) -> str | None`, `bodyparts.attributes(obj) -> frozenset`, `bodyparts.find_substitutes(meshes, job, names, changed, scene=None) -> ({export mesh: part}, [note])`.

- [ ] Write `test_find_substitutes` (matches; material, attribute, slot mismatches; 10 cm away; visible beats hidden twin; part used once with a note; `material_key` of `/mt_c0201b0001_bibo.mtrl` and `mt_c0201b0001_bibo.mtrl [4f128fd5]` equal; `slot("0.1 c0201b0001_bibo LOD0.001") == (0, 1)`); run, fail.
- [ ] Implement as the spec's Matching section; candidates' BVH built once; export vertices sampled to at most 2000.
- [ ] Run, pass.

### Task 3: api VERSION 3, substitute opt-in, shown() switches the devkit

**Files:** Modify `magic_fit/api.py`; Test `tests/test_resize_core.py`

**Interfaces (consumes):** Tasks 1 and 2. **(produces):** `Resizer(..., substitute=False)`, `Resizer.substitutes`, `resize_collection(..., substitute=False)`, `SizedMeshes.substitutes`, `api._StateSwitch(parts, names, body)` (sources to set in order, parts set by hand, notes).

`shown()` order: clothing keys; substitutes' visibility (layer collections from the root down: exclude off, hide_viewport off; collection and object hide_viewport off; hide_set(False)); an undo entry that updates the view layer (so driven values are evaluated back while the substitutes still show); settings (snapshot via `sizing.keep_settings` and the chain Keys' values); undriven keys (by-hand parts: `_set_key` for all their values); `view_layer.update()`.

- [ ] Write `test_shown_switches_state`; extend `test_resize_collection` (drivers follow, not muted; with/without substitute; VERSION 3); run, fail.
- [ ] Implement; run all of `test_resize_core.py` and `test_resize_ui.py`, pass.

### Task 4: Resize button leaves Piercing B out

- [ ] Extend `test_sizes_operator`: a selected Piercing B gets no key. Passes from Task 1's recognition.

### Task 5: Documentation and the real-scene check

- [ ] `api.py` docstring (opt-in, export loop); wiki `Auto-Resize.md` "Body parts from the devkit"; `docs/development.md` measurements and how to repeat.
- [ ] Scratchpad headless check on `slider_test.blend` (not committed): Omoi Lavabod and Omoi Lavabod Rue against LavaBod+ 2.2, weights, Piercing B placement, scene state unchanged.
