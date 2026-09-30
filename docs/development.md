# Magic Fit: development notes

How to build, test and work on Magic Fit, and the measurements behind its tools and defaults. For using it, see the [guide](guide.md).

## Building and installing from source

The add-on's code is in `magic_fit/`. Build the extension package from the repository's root:

```
blender --command extension build --source-dir magic_fit --output-dir dist
```

Or run `tools/bundle-addon.ps1`. It builds with Blender 5.2 and checks that the manifest and bl_info versions match, that the wheels are there and stripped, and that the zip holds them and no bytecode. `-Check` also runs `tools/check_install.py` on the zip:

```
.\tools\bundle-addon.ps1 -Check
```

Install the zip from `dist/` with **Edit → Preferences → Get Extensions → ⌄ → Install from Disk…**, or drag it into Blender. It includes SciPy and robust-laplacian for Weight Transfer (see [SciPy for Weight Transfer](#scipy-for-weight-transfer)).

The extension id is `magic_fit`, and scenes keep their settings in `Scene.magic_fit`. Files saved before Resize had Use Shapekeys open with it on where a key is picked, so they resize as before.

`tools/check_install.py` installs a package into a throwaway Blender profile, enables it, and checks for extension policy warnings, a transfer with SciPy from the installed wheels, a Customize+ template, Hair Weights, and Face Weights with faces read from the game:

```
python tools/check_install.py dist/magic_fit-1.0.0.zip
```

Don't run such checks with your own Blender profile: enabling an extension with factory settings makes Blender delete the wheels of every extension that isn't enabled, including yours. The [tests](#tests) are safe, since they load the add-on from the repository. `--skip-face` leaves out Face, for computers without the game.

## Releasing

Raise the version in both `blender_manifest.toml` and bl_info, then rebuild the extension repository in `blender_repo/` and commit it with the change:

```
.\tools\generate-blender-repository.ps1
.\tools\verify-blender-repository.ps1
```

The README's install link reads `blender_repo/index.json` from the main branch. On every push to main that touches the add-on or the repository, the workflow in `.github/workflows/blender-extension-repository.yml` checks that the committed repository matches its zip, builds and installs the package on Windows like a user would (and runs `check_install.py --skip-face` on it), and publishes its own build to GitHub Pages, at `https://link-0402.github.io/MagicFit/index.json`.

## Tests

Run from the repository's root (PowerShell). The core tests check the engines headless; the UI tests open a Blender window for a few seconds and drive the tools with simulated mouse events (strokes, undo, Esc, mirror, Ctrl, hidden geometry, shape keys, several objects, posed armatures, clipping marks, reopening a saved file).

```
# Core tests
foreach ($t in 'core','fit_core','straighten_core','skirt_core','heels_core','smooth_core','hair_core',
               'face_core','relax_core','resize_core','lineup_core','transfer_core','cplus_core') {
    blender -b --factory-startup --python "tests/test_$t.py" }

# UI tests
foreach ($t in 'ui','fit_ui','fit_deformed_ui','straighten_ui','skirt_ui','heels_ui','smooth_ui','hair_ui',
               'face_ui','relax_ui','resize_ui','lineup_ui','clipping_ui','sidebar_ui','transfer_ui','cplus_ui') {
    blender --factory-startup --enable-event-simulate --python "tests/test_$t.py" }
```

To run one test, use its line alone, for example `blender -b --factory-startup --python tests/test_fit_core.py`.

- **Screenshots:** set `MF_SCREENSHOTS` to a folder to keep them.
- **Windows:** add `--no-window-focus` to keep the UI tests out of your way, and `-p x y width height` to place them, for example on another screen. Blender measures `y` upward from the bottom of the whole desktop.
- **Face:** the face tests, and the face check of `tools/check_install.py`, need FFXIV installed where Magic Fit finds it by itself. The tests load the add-on from the repository, not as an extension, so they keep the faces in `datafiles/magic_fit/face_data.npz` in Blender's user folder.

Scripts in `tools/`:

- `make_icon.py` regenerates the toolbar icons.
- `make_hair_reference.py` regenerates Hair's skeletons and heads from Instant Edit's game export and TexTools' face exports.
- `fit_hair_matching.py` measures the game's hairstyles and a Penumbra collection's hair against every skeleton, and fits Hair's scores.
- `make_cplus_reference.py` regenerates the game skeleton table Customize+ uses (see the script for its sources).
- `compare_race_skeletons.py` regenerates the tables of [race-skeletons.md](race-skeletons.md).

Face needs no tool: it reads its data from the game (see [Face: where its data comes from](#face-where-its-data-comes-from)).

## SciPy for Weight Transfer

Weight Transfer needs SciPy and robust-laplacian, as wheels listed in `blender_manifest.toml`. `tools/vendor_wheels.py` copies them into `magic_fit/wheels` from the standalone add-on's repository next to this one (`../robust-weight-transfer/wheels`, downloaded by its `scripts/install-dependencies.ps1`). It drops test suites and static libraries Python never loads (robust-laplacian shrinks from 5.2 MB to 0.2 MB) and unpacks the result into `deps/` for the transfer tests:

```
python tools/vendor_wheels.py
```

Keep the same versions as the standalone add-on: Blender installs a wheel once for all the extensions that list it.

## How it works

Every tool is a toolbar tool (`WorkSpaceTool`) whose left-click runs a modal stroke operator. Custom tools are the only way for add-ons to add brushes, since Blender's brush engine can't be extended from Python. At the start of a stroke, a tool snapshots the meshes in world space as displayed (so pose, shape keys and modifiers count), and builds BVH trees for raycasts and nearest-point queries, cached between strokes.

- **Copy** (Weight Brushes) finds each vertex's closest point on the target and blends toward the weights there. It samples the target's evaluated mesh when modifiers at most about double it (Mirror, Solidify, Weld), else the base mesh. Auto Smooth runs damped Jacobi passes over the painted mesh's edges, and samples every vertex those passes reach, so the result doesn't depend on the stroke's path. See `magic_fit/painting.py`.
- **Body Fit** works on a welded vertex graph of the body. Each clothing vertex heads for a target offset from its reference position, so dabs settle instead of piling up. It moves in a straight line along the body's normals averaged around it by area, each body vertex weighted by (1 − t)², where t runs from 0 at the closest body point's distance to 1 at Keep Together farther (`BodySurface.smoothed`). Across a crease that direction turns smoothly, where the closest point's normal flips. Auto Smooth drapes those targets and smooths the reference heights, measured to the same averaged body, over the clothing's edges. See `magic_fit/fitting.py`.
- **Resize** mixes the bodies' shape keys into the From and To shapes; a different To Body is matched by UV map (a BVH tree over UVs, barycentric placement). Clothing moves in small steps by a smooth, area-weighted average of the body's movement, over a width that grows with its distance from the body. Keep Close pushes tight clothing back to its distance along the body's normal, and rigid parts are fitted with the Kabsch rotation. Bridge Creases rolls a ball over both shapes to make crease-free copies (a KD tree over ball centres) and blends toward their move from 3 to 6 mm off the body. See `magic_fit/resizing.py`.
- **Line Up** reduces both skeletons to 17 parts plus three segments per finger, by the words in bone names (The Sims' FNV-1 hashed names are looked up), or, without names, by a maximum spanning tree of the weight groups share. That tree hangs from the model's midline (the vertical plane it's mirrored across, so a skirt weighted to one leg doesn't pull it aside) and never joins groups on opposite sides of it; a hand's branches are its fingers, and an arm is cut at the elbow first, then where the upper arm is as long for the forearm as the body's. A joint is a bone's head or, without bones, the centroid of the vertices two parts share, weighted by the product of their weights. Each part turns (the smallest turn onto the body's direction, from where the part above left its joint) and scales with the torso; neck, head and clavicles turn with it. Half a body (a top or a bottom) is sized by its arms or legs instead. Stretch Limbs adds a translation along each segment, eased in from its joint by a C¹ ramp (stretchable bones, Jacobson and Sorkine 2011), instead of scaling the segment, so blended weights don't shear. All of it is applied to vertices, shape keys and edit bones by linear blend skinning (LBS). Estimated joints are refined in up to 3 rounds of point-to-plane least squares against the body. Bones that all sit at one spot are put where their groups hand over. See `magic_fit/lineup.py`.
- **Straighten** grows an area from the stroke and builds its convex hull (`bmesh.ops.convex_hull`) posed and at rest. Rays along the normals measure how far each vertex lies below the hull; where that grows over the rest pose, the fabric sags. The sagging part gets harmonic weights (cotangent Laplacian, conjugate gradients) from the vertices around it, a "weight inpainting" as in robust skin weight transfer, with the gap found in the pose. See `magic_fit/straighten.py`.
- **Skirt** finds the skirt bone chains by name. A vertex's skirt share is a smoothstep of its height, lowered near the body, scaled by the part of its body weight not on the arms and head (the bones from `j_sako` and `j_kubi` down, so sleeves hanging below the hips in the A-pose keep theirs), split between chains by a tent kernel and between segments by smoothstep crossfades. The body part keeps the vertex's own body weights, so running it twice changes nothing. See `magic_fit/skirting.py`.
- **Heels** finds the calf, ankle and toe bones by name. The foot's share is a smoothstep of height above the ankle joint; the toe bone takes a smoothstep of the distance ahead of the ball of the foot. Skirt and Heels share `magic_fit/goals.py`, which keeps each vertex's total bone weight and each part's share under the group limit. See `magic_fit/heeling.py`.
- **Smooth** builds a graph from the mesh's edges (cotangent weights) plus links between loose parts within Merge Range. Each dab solves one implicit heat step, (M + Reach² L) X = M B, on a region 3 × Reach wide, with SciPy's sparse LU (`splu`) or conjugate gradients without SciPy. An extra column holds how much weight each node has, and the result is divided by it, so unweighted nodes don't pull others down and get filled in. See `magic_fit/smoothing.py`.
- **Hair** scores every skeleton of the race against the hair (see [below](#hair-how-the-skeleton-is-picked)), then shares each vertex between chains by exp(−score) (`hairing.chain_terms` × `CHAIN_WEIGHTS`: distance, direction around the head, height). The chain weights were fitted by softmax cross-entropy to 558 hairs, the skeleton score's by L-BFGS. Weights are written in `goals.GoalStroke`'s `replace` mode, and `hairing.tag_est` tags each mesh with `xiv_est_hair` and `xiv_est_race`. See `magic_fit/hairing.py`.
- **Face** finds the eye openings, lid margins, lash lines and lip line with ray scans 0.25 mm apart, on the game's face and the custom one. A 3D thin plate spline through those features, the neck ring and anchors maps one onto the other, and each vertex takes the game's weights at its closest point, never from the opposite lid or lip. Close Eyes skins the game's Shut Eyes expression and its blink's peak (LBS), raises lid weights where the eye stays open in either, then scales them down by the least factor (found by bisection) that still closes it 2.5 % of the way before the end of Shut Eyes and at the blink's peak. Lashes take their roots' weights, fading toward the tips. See `magic_fit/facing.py`.
- **Weight Transfer** matches every vertex to its closest body point (BVH, rest pose when on the body's armature) and solves for the smoothest weights (biharmonic, or harmonic when ill-conditioned) on robust-laplacian's point-cloud or mesh Laplacian, with SciPy. Combined adds the point cloud's links between touching loose parts to the mesh Laplacian. With Crease Smoothing, every match is a soft constraint weighted by area over (3 × distance)⁴, fading where normals differ by 11° to 25° (`weighttransfer.bridge_confidence`); Hem Follow poses hems in four synthetic strides and pulls them toward the leg where a thigh passes through (`hem_follow`). Weights are written only once every step has succeeded. See `magic_fit/robust_transfer/`, kept close to the standalone add-on.
- **Texture Relax** fits, per vertex and UV chart, an affine map from UV to the neighbours' positions and moves the vertex toward it, along the surface. Open edges stay on their own polyline. See `magic_fit/relaxing.py`.
- **Customize+** decodes the template (base64 of gzip of a version byte and JSON) and simulates Customize+ bone by bone in the game's model space (`simulate`, checked in the tests against a line-by-line port of its C# code). The hidden armature's bones copy the originals, then add Action constraints: propagation before the pose, the bone's own change after it, the root's last. Bone axes are matched to the game's, snapped to the nearest axis swap within 20°. See `magic_fit/cplus/`.
- **Clipping marks** are a viewport draw handler that re-measures only moved vertices, within a time budget per redraw. What moved comes from depsgraph updates and, on frame changes, from the new frame's depsgraph. Each view layer keeps its own marks, so windows showing different scenes don't reset each other. A posed body only has its positions, normals and search tree rebuilt: what its triangles decide is kept (`BodySurface.from_geometry`). Inside or outside comes from the body's normal at the closest point; past the body's open edges, more than about 11° off the normal counts as outside (`BodySurface.nearest` in `magic_fit/fitting.py`). See `magic_fit/clipping.py`.

### Customize+: how it's set up

Apply adds a hidden child copy of each armature ("C+ Skeleton" for an armature called Skeleton), whose bones follow the originals plus the template's change. On every mesh, the Armature modifier stays in place but is switched off, and a copy right after it, "C+ Armature", follows the hidden armature. Turning Customize+ off removes the copies and switches the originals back on, so the meshes are exactly as before. Because the original modifier stays first, Weight Paint can still select bones with Ctrl click, and the other tools keep working.

### Customize+: how exact it is

It does what Customize+ does in the game (from its source, `ModelBone.ApplyModelTransform`), in any pose:

- Each bone is scaled along its own axes, turned about them and moved along its turned axes, and **only that bone changes**. Customize+ reads rotation as yaw, pitch and roll: X turns about the bone's Y axis, Y about its X axis, then Z about Z.
- A bone set to **propagate** moves, turns (Propagate Rotation) and scales (Propagate Scale, with Child Scaling when Independent) the bones below it about its new position, along the character's axes. The game drops the shear an uneven propagated scale causes; Blender matches that exactly at rest and within a few millimetres in other poses.
- The root bone's (n_root) scale scales the whole character and its position moves it. Its rotation is ignored, like in the game. It needs no n_root bone: All Armatures puts it on every armature with a bone of the game's skeleton.

Importers often swap bone axes, and some turn bones freely, so the add-on compares every bone with the game's skeletons and uses each bone's own axes. Races turn some bones quite differently (face bones by up to 41°, Hrothgar and Lalafell most; see [race-skeletons.md](race-skeletons.md)), so it takes the race whose body and face skeleton fit best. Hair bones are assumed to use the axes of the armature's other bones. The data comes from TexTools' skeletons of all 18 races and, for IVCS and YAS bones, Yet Another Devkit's Skeleton (`magic_fit/cplus/reference.py`, made by `tools/make_cplus_reference.py`).

### Customize+: Bust Size

What the game does with the character creator's Bust Size (customize byte 23) was read in its code, version 2026.09.15.0000.0000, at the addresses FFXIVClientStructs' `ida/data.yml` gives and with Penumbra's signatures:

- **Setup.** In `RspSetupCharacter`, the game calls the function Penumbra hooks as GetRspBust with the clan, gender, body type and bust size. Males (gender byte 0) skip it and get no bust scale. It returns, per axis, min + (max − min) · size / 100; a size above 100 counts as 0. Min and max come from the racial scaling table of `chara/xls/charamake/human.cmp`: from byte 0x2C800, per race 10 entries of 14 floats (5 body types × 2 clans), the bust's min x, y, z and max x, y, z being floats 8 to 13 (Penumbra.GameData's `CmpData`). Penumbra's own copy of the function computes the same. The result is stored in the character's PostBoneDeformer (CharacterBase + 0x150, at + 0x68, where Anamnesis edits it), which also looks up `j_mune_l` and `j_mune_r`.
- **Every frame.** After bone physics, the PostBoneDeformer multiplies each breast bone's model-space scale (`hkaPose::getBoneModelSpace`) by the bust scale, keeps its position and turn, and sets it with propagation: bones below keep their local transforms. Since the game combines transforms as T = T_parent + R_parent · T_local, R = R_parent · R_local, S = S_parent ⊙ S_local (`hkaPose::calculateBoneModelSpace`), the bones below get the same factors on their own axes and stay in place. This matters for Yet Another Devkit's Mannequin, whose breasts are weighted to `iv_c_mune_l` and `iv_c_mune_r` alone, below `j_mune_l` and `j_mune_r`.
- **Values.** Every clan's player characters (body type 1) have min (0.92, 0.8, 0.816) and max (1.08, 1.2, 1.184), except both Lalafell clans, whose min is (1, 1, 1). The breast bones' X axis points out of the chest (turned 13° outward), Y up and Z sideways. So at 0 the breasts are 92 % as deep, 80 % as tall and 82 % as wide; at 100, 108 %, 120 % and 118 %.

In Blender, the breast bones and every bone below them get one more change after the pose and the template, in their own frame: "C+ Bust", holding frameᵀ · diag(scale) · frame (`AxisMap.frame`, `solver.bust_change`). For swapped axes that's the scale reordered; a freely turned bone also gets a "C+ Bust Frame" turn back. Scales multiply axis by axis, so the order against Customize+'s own changes doesn't matter. Moving the slider replaces only those constraints: 20 ms on the devkit's 176-bone skeleton. An armature takes the Lalafell range when its bones are turned like a Lalafell's or it's under 0.7 times the Midlander female skeleton's size (Lalafells are about 0.45, other races 0.96 to 1.38). `tests/test_cplus_core.py` checks the ranges against `human.cmp` when the game is installed.

Not verified: no in-game character's bones were compared with the rig, only the game's code and data (tests compare the rig with a separate implementation of that code). The IVCS and YAS bones' axes are known from one devkit skeleton. Racial scaling mods (Penumbra's RSP) change the ranges in the game but not here.

### Hair: how the skeleton is picked

Every skeleton of the race is scored by how its bones fit the hair's hanging parts: distance to the bones, hair hanging past a chain's end, bones that would move nothing, the number of bones and chains, and how far down and back they reach. The ten best are posed with their weights (head tilted and turned, hair bones lagging) and scored again by stretch and lag. The score was fitted on the game's 1,870 player hairstyles, to behave like the game's hair, and on 116 long modded hairs, to pick what their modders picked. Hair of which less than 1 % hangs gets no hair skeleton; lowering Hang Start lets bangs move.

### Hair: where its data comes from

From the game's hair and body skeletons (Havok files, read with Yet Another Addon's reader) and TexTools' exports of one face per race, the add-on keeps only the bones' rest poses and each head as a 1 cm distance grid with 800 surface points, symmetric left to right (`magic_fit/hair_reference.npz`, 0.8 MB, made by `tools/make_hair_reference.py`).

### Face: where its data comes from

None of the game's data ships with the add-on. The first time Face needs it, it reads the faces from the user's game install and saves them as `face_data.npz` (5.6 MB) in the folder `bpy.utils.extension_path_user` gives, which Blender keeps across updates. It reads them again when the game's version (`game/ffxivgame.ver`) or `facedata.DATA_VERSION` changes, or when the chara index entries of the files it read do: TexTools installs and removes mods by pointing them at `.dat` files of its own. Files read from those (marked 1337 or 6969 twice at 0x200, or by older TexTools versions with an empty data hash at 0x420, as TexTools tells them apart itself) are listed in the saved data, and the face tools warn when a face comes from them. If the game can't be read, faces saved for an earlier version are used; if it isn't found, the saved ones are used as they are.

- **Finding the game:** `gamefiles.py` tries the Game Folder preference, XIVLauncher's and XIVLauncher.Core's settings, the game's and Steam's uninstall entries, Steam's libraries, and the default folders on Windows and Mac. It reads SqPack files read-only and rebuilds a model into its `.mdl` file the way Lumina does; for all 122 faces that's byte for byte Instant Edit's game export.
- **What's kept:** `facedata.py` reads, for each player race (codes ending in 01), faces 1 to 8, faces 101 to 104 where the clan has them (Au Ra and Viera reuse 1 to 4), and the NPC faces 91 and 92: every face up to 104 that `faceskeletontemplate.est` gives a skeleton and that has a model. Models are read with [XIVPy](https://github.com/Arrenval/XIVPy)'s model reader (`magic_fit/xivpy`, an unmodified subset). It keeps the first level of detail except the eyes, as Instant Edit's importer gives it to Blender: positions, normals, triangles, byte weights and the eyeball radius, without UVs or textures. Per face skeleton (61) it keeps the rest pose from its `.sklb` and the peak frame of nine face animations from `resident/face.pap`, read with XIVPy's Havok tagfile reader and decoded by `gamefiles.decode_spline`. The Shut Eyes expression is `cfxf_bow`: the Emote sheet's row 73 plays ActionTimeline 611, `facial/pose/bow`, whose `chara/action/facial/pose/bow.tmb` plays that animation from `face.pap` (the Bow emote's face, `nonresident/emot/bow.pap`, has the same one).
- **Speed:** reading everything takes about 4 s; loading the saved file 0.05 s, and checking the index entries 0.1 s.

Checked against the data Face was first built with (Instant Edit's export through Blender): faces, parts, positions, triangles, weights, skeletons and poses are identical. Normals are now the game's own; Blender had moved about 92,000 of 304,000 by up to 0.001 (its encoding's precision) and 117 by up to 49°.

## Measurements

Test results behind each tool's design and defaults.

### Copy and Body Fit

- **Auto Smooth (Copy):** copying a sharp weight jump onto a finer grid, neighbouring vertices differed by the full weight without it, 0.49 at 0.5 and 0.23 at 1.0. Painting in two halves gave the same weights as painting at once.
- **Body Fit seams:** a split strap with a 0.06 mm gap opened to 5.1 mm when pushed without welding; with welding it didn't change.
- **Body Fit creases:** a flat strip spanning a cleavage-like valley 3 cm above its crease, stroked across 4 times with Auto Smooth. Moving along the closest body point's normal slid the middle 0.7 to 4.3 mm sideways and spread vertices 2.5 mm apart to 0.8 to 9.9 mm. Along the averaged normals the middle stays put and they end up 1.6 to 3.5 mm apart, the least where Push Out bends the strip around the breasts. Dabs take 0.4 ms longer with a 3 cm brush, and 9 ms with a 6 cm brush on 82,000 vertices, whose first dab takes 40 ms longer.
- **Body Fit shape keys and poses:** nudging the edit positions once per stroke (about 60 ms on 20,000 vertices) lets garments in Shape Key Edit Mode, posed garments and bodies with shape keys land at the offset (before, 6 cm off instead of 1 cm).
- **Body Fit layers:** a 3 cm double-layer shell pushed 4 cm out kept its thickness within 1 mm; full Auto Smooth made results 2.5× to 86× smoother.

### Straighten

Up to 1.17, Straighten decided in the rest pose which parts float (farther than a Resting Distance from the body). On real clothing that changed nothing visible, since a fixed distance can't tell a crease from loose fabric. Since 1.18.0 it works on the pose you see. On a dress and a sling bikini (medium chest, a Customize+ template scaling breasts up to 1.69 and buttocks up to 2.2), single clicks of 0.1 to 1.3 s gave:

| | Rest pose | Scaled, weights as transferred | After a click |
| --- | --- | --- | --- |
| Sling straps, from the side: sag under the breasts | 8.2 / 7.1 mm | 11.5 / 12.5 mm | 7.4 / 7.3 mm |
| Dress, from above: outline dipping between the breasts | 18.2 mm | 25.4 mm | 15.6 mm |
| Dress, from the side: dip under the breasts (left / right) | 42.4 / 43.2 mm | 59.5 / 58.5 mm | 42.7 / 42.1 mm |

### Smooth

- **Region 3 × Reach:** six overlapping dabs came within 0.026 of one dab over everything (1.5 ×: 0.11, 2 ×: 0.041, 4 ×: 0.006); a dab's time grows with the square of the region. Results are the same on coarse and fine meshes (within 0.002).
- **Merge Range 2 mm:** on the Mannequin (21 loose parts), nearby vertices of different parts differed by at most 0.113 before a full dab, 0.020 after with 2 mm, and 0.73 with 0, where each part is smoothed alone. The Mannequin gets 3,022 links at 2 mm, 3,138 at 5 mm and no more at 10 mm, so 2 mm catches nearly all of them.
- **SciPy:** dabs took 3.8 to 31 ms (Reach 1 to 8 cm) against 5 to 419 ms with conjugate gradients, with the same result within 0.002.

### Skirt and Heels

- **Skirt:** on three modded dresses, Blend Start 10 cm and Blend Length 30 cm matched the devkit reference skirt: 0.21 of the weight on skirt bones at the hips, 0.49 about 5 cm lower, 0.96 15 to 20 cm lower. With thighs and butt at 130 %, Skin Weight 0 let 7 vertices clip more than 1 mm, 0.5 none.
- **Heels:** with the ankle posed 26° and toes 40°, weights copied from the devkit's feet made real shoes' heels and soles stray up to 49 mm from moving rigidly; after Heels, none. The calf's share came within 0.03 on average of Hera Heels' author's weights.

### Hair

Against 116 long modded hairs and 94 game hairstyles, each weighted with a score fitted without it (brackets: the closest-chain method of 1.19 and earlier):

- **Head tilted and turned 30 to 45°:** 5.4 mm (modded) and 4.5 mm (game) from where their own weights put them (6.1 and 4.8 mm).
- **Each hair bone swung 20° alone:** 7.2 and 3.5 mm (9.0 and 4.3), better for 99 of 116 modded hairs; 88 % and 89 % of the hanging hair went to the author's chain (80 % and 77 %).
- **Picked skeleton** on unseen hair: 5.4 mm for game hairstyles (own skeleton 4.5, none 6.2), 6.7 mm for modded ones (modder's 5.5, none 8.0).

**Joint Blend 30 cm** (was 10 cm) improved the modded swing test from 7.45 to 7.25 mm (better for 94 of 116) and left the game's at 3.5 mm. Tuning settings per hair would get 28 % closer, but a hair's shape doesn't predict them (a regression gained 5 % and made one hair in five more than 10 % worse), so the defaults are the best on average.

### Face

- **Game faces:** Face Weights restores the skin's weights to within 0.0005 on average (Hyur Midlander female face 1, whose lids meet near the end of Shut Eyes; see below for faces whose lids meet early); lashes come out 0.14 off, eyeballs 0.018.
- **199 modded faces** (measured with 1.0.0, which fitted the lids to the blink): no failures. The game's blink left an eyeball showing on 20 with their own weights and 1 after Face Weights. Median 2.6 s. Detection by shape found the right race for 194 but the right face number for only 30, so it's only a fallback.
- **Lash lines** at first landed on the brow on a quarter of male faces (5 to 16 mm off). Passing through skin more than 5 mm off the eyeball brought 95 % within 4 mm of their margin.

**Close Eyes targets the Shut Eyes expression, not the blink.** The game's Shut Eyes expression (`cfxf_bow`, two frames) holds the lids as long as it lasts, its upper lids turned 91 to 102 % as far as at the blink's peak (median 100 %, over the game's face skeletons). Up to 1.0.0, Close Eyes made the lids meet where the game's face closes its eyes in the blink, 75 to 100 % of the way to its peak (median 91 %), since the game shows the peak for an instant only (`cfxb_blink1` stays within 5 % of it for 25 ms). Held in Shut Eyes, such lids go on for the rest of the turn, slide past each other and fold. On the game's 122 faces with eyeballs 97 % of the game's size, as the tests build them, and on copies with the eyes opened 2.5 mm wider or closed 4 mm narrower (the tests' `widen_eyes`), the share of Shut Eyes at which the lids meet after Face Weights:

| | Game faces | Wider eyes | Narrower eyes |
| --- | --- | --- | --- |
| 1.0.0, fitted to the blink | median 91 %, from 77 %; 44 before 90 % | median 94 %, from 81 %; 15 before 90 % | median 90 %, from 61 %; 49 before 90 % |
| Now | median 98 %, from 95 % | median 98 %, from 89 %; 1 before 90 % | median 98 %, from 94 % |

The game's own faces meet at 77 to 100 % of Shut Eyes with their own weights (median 91 %), so Face Weights now lowers the lids of 75 of them. Eyes that can't be closed at all: 13 game faces (the NPC faces 91 and 92 among them) and 67 wider copies, whose lids can't reach, against 23 and 67 with 1.0.0. The blink's peak must still close the eye: fitted to Shut Eyes alone, 58 faces and copies left it open, mostly by a slit one scan row tall, but by up to 15 mm² (Roegadyn male face 4 with wider eyes, 1 mm tall). Keeping it closed outright made some narrower lids meet 23 % early in Shut Eyes (Hyur Midlander male faces 5 and 7) for slits of 0.25 mm², so a slit up to 0.3 mm tall is allowed there, which the lashes hide for the instant it shows. Gaps taller than that, where Shut Eyes closes, are left on 6 narrower and 3 wider copies (1.0.0: 5 and 1), mostly the gaps 1.0.0 left too. Planning Face Weights took a median 3.3 s per game face (six measured at once), about as long as 1.0.0.

On the Sabre x Kaede x Darx face, whose eye opening is 4.8 mm tall against the game's 10 mm, the game's lid weights closed at 49 % of the blink and folded the lid 6.1 mm past the lower one; 1.0.0 lowered them to 62 %. Lowering comes last, by the least factor that still closes the eye, because lowering first let the later raise overshoot on 16 faces. With 1.0.0, on 184 mod face models, only 37 closed like the game's face in the blink with their own weights, and 170 after Face Weights.

### Weight Transfer

**Crease Smoothing (1.20)**, on clothing from the Yet Another Devkit Mannequin under a template scaling the butt about 2×, thighs 1.4× and bust 1.55×:

- On a tight mini dress, the old version let the body come through beside the butt cleft over 30 cm², up to 36 mm deep. Now nothing does, standing or walking. The old version blended weights where the dress lay on the skin, and under a 2× bone scale a 10 % weight difference moves the dress about 1 cm.
- On four more outfits, in idle, movement and dance frames 4.9 cm² came through on average instead of 23 (modders' own weights: 16). A little more thigh shows at hems while walking; Hem Follow cuts that (nightgown 17.8 to 6.9 mm deep).
- A 16,000-vertex dress takes about 3 s. Transferring with Customize+ on gives identical weights.

**Rigid:** a belt piece stretched at most 2 % posed, against 137 % with ordinary weights.

**Group limit 8:** on a high-cut bikini bottom, edges stretched up to 35 times with 4 groups, 6.6 times with 8, about the body's own 6.

**Combined** is the default since 1.17.0 (before, Point). On one-piece loose pants, Point let one leg's weight bleed into the other by 25 % on average (up to 52 %), Combined not at all. A lining 2 mm under a panel stayed within 0.001 of it (Point 0.007, Surface failed), and split seams stayed closed (Surface tore them). On a single piece it's about three times faster than Point.

### Line Up

With the Mannequin as the body: a VRChat outfit (18 meshes, T-pose) was scaled, brought into the A-pose and put on the Mannequin's joints in 0.4 s. A suit without its armature (3ds Max Biped groups) had its joints estimated from weights, mostly within 2 cm after checking against the body, but its knees stayed 7 cm off. A Sims 4 body with bones all at the origin and groups named bone_00 to bone_57 was sorted into its parts in 0.2 s.

**Neck, clavicles and stretch.** Three imports: a Stellar Blade bodysuit (UE4, 3ds Max Biped rig, in centimetres), that Sims 4 body and a VRChat outfit. Line Up used to put every joint on the body's, and stretched necks 1.6 to 1.7 times to reach j_kao, which sits higher than other rigs' head joints. It turned the Sims clavicles 34° about pivots 7 to 10 cm off the middle, lifting the neck side of the shoulders into a hood, and bent the Sims neck 36° forward. Stretch scaled each segment's matrix, so where two segments' weights blend (a thigh at 0.77 against a shin at 1.12) the mesh sheared. Worst edge stretch, and edges changed by more than 65 %, with Stretch Limbs:

| | Before | Now |
| --- | --- | --- |
| Sims 4 body | 15.2× (2,204 edges) | 1.9× (32) |
| Sims 4 body, Stretch Limbs off | 6.7× (1,017) | 1.5× (0) |
| UE4 bodysuit | 2.6× (696) | 1.9× (13) |
| VRChat harness | 3.5× (415) | 1.5× (0) |

The median distance from the model's surface to the Mannequin's went from 10.1 to 8.3 mm (UE4) and from 12.5 to 8.9 mm (Sims). Fingers curled 35° line up along the body's within 0.01°.

**More models.** 120 meshes from about 50 downloads: 36 VRChat outfits on 15 bases, 7 game rips (Stellar Blade on Unreal and DAZ Genesis 9 rigs, Street Fighter 6 on Rigify, a DAZ Genesis 8.1 figure, an Unreal-named FBX) and 77 meshes from 45 Sims 4 packages (hair and accessories left out), each lined up with Stretch Limbs and compared with the 1.0.0 release. With unnamed groups, long dresses fell apart: a skirt weighted mostly to one leg pulled the weighted middle 5 cm aside, so the spine counted as a side and the pelvis as a thigh; a skirt weighted to both legs joined them into one limb; and a sleeve weighted to the shoulder decided the cut between clavicle and upper arm (the clavicle's share of the arm was 0.10 or 0.26 against the body's 0.18). Tops and bottoms were refused.

| | Lined up (now / 1.0.0) | Edges changed > 65 %, median | Worst edge, median |
| --- | --- | --- | --- |
| VRChat | 33 / 32 of 36 | 65 / 531 | 3.1× / 5.1× |
| Game rips | 7 / 7 | 59 / 1,734 | 2.8× / 7.7× |
| Sims 4 | 63 / 37 of 77 | 12 / 710 | 1.9× / 8.9× |

Still refused: a corset rigged to spine bones only, a dress without an armature or groups, one rigged only to its own physics bones, and 14 Sims pieces of multi-mesh outfits (jewellery, gown skirts and bodices on part of the rig). Under long skirts the hips can end up about 4 cm high, which makes those dresses up to 8 % too big.

**Rejected:**

- **Normal lines for joints from weights.** This took the point nearest the lines along the normals where weights hand over, instead of their centroid, so one-sided handovers (a shoulder's top) would land inside the limb. On the Mannequin's and the UE4 suit's own weights it was no closer to their bones (42.0 and 35.1 mm on average, against 43.1 and 36.3), and on tapered or bent tubes it was worse.
- **Dual quaternion blending** moved these imports by at most 4 mm, with the same edge stretch.
- **Trimmed or one-sided handovers for skirts.** Leaving out points far from a handover, or points weighted to both legs, didn't bring the hips under a skirt down: most of the weight there is the skirt's. Leaving out the two-legged points only moved the hips outward, and on tight models too.
- **The torso's width for the shoulder cut** (a shoulder joint lies within the upper torso's reach). Tight tops weight the sides of the ribcage to the breasts and arms, so the torso looked too narrow and four models lost their clavicles.

### Resize

- **Against other methods:** resizing the RueXB+ Stellar Blade suit to Rue, the devkit's key transferred by YetAnotherAddon turned 674 faces over and Surface Deform 203. Resize turned none over and kept 90 % of the suit within 0.8 mm of its distance to the body (61,000 vertices in 6.4 s).
- **Against authors' versions:** on three mods with YAB and Rue+ versions, resizing the YAB version matched the authors' Rue+ to 0.07 to 0.25 mm on average.
- **Keep Close:** resizing to a medium chest, the smooth move alone left bags under the breasts (1,073 vertices more than 3 mm out) and sank 820 vertices into the body; Keep Close cut these to 591 and 307, at the cost of 31 small creases.
- **Bridge Creases:** on 20 pairs of chest sizes from 11 mods, Resize without it fell 3 to 8 mm short below the breasts, where clothing hangs from them. At 8 cm it came closer on 17 pairs (for example Hot Topic 5.1 to 4.2 mm) and a bodycon dress went from 83 flipped faces to 2. It adds 1 to 2 s per garment.

**Rejected:** scaling bones first pulled garments off the skin (up to 22 mm), since the Mannequin's breasts are weighted mostly to j_sebo_b. Crease-free copies without the 3 mm blend, balls fitted to both shapes, membranes and similar variants left creases or folded clothing.

### Clipping marks and Texture Relax

- **Clipping marks:** with the Sideless Dress and the devkit's torso alone, 792 vertices below the torso used to be marked as clipping past its open edges; now none are, and the 758 real ones still are.
- **Texture Relax:** a sheet pushed 14 mm sideways went back to within 0.01 mm of its layout, split seams stayed closed and layers 0.5 mm apart stayed apart.

### Customize+

TexTools face exports of every race match the game's result within 0.2 mm. On the Mannequin with a 28-bone version 7 template, against the game's result computed from Customize+'s code:

| | Rest pose | Posed | Apply takes |
| --- | --- | --- | --- |
| This add-on | 0.00 mm | 0.00 mm | 0.1 s |
| Bustomize, scale | up to 8.6 mm off (5.5 mm on average) | the same | |
| Bustomize, scale and rotation/position | up to 15 mm off | up to 519 mm off | 1 s |

Bustomize (1.1.3) lets children follow a bone's scale, turns bones about the wrong axes and in another order, and switches rotation inheritance off for every bone, which breaks posing.

**Bust Size:** the rig matches the game's rules (see [Customize+: Bust Size](#customize-bust-size)) within 0.001 mm at rest and posed, on game, swapped, centimetre and freely turned axes; after a template that propagates an uneven scale through the breasts, within 4 mm posed. On the Mannequin, sizes 0 and 100 move the breasts by up to 8.4 mm, and 50 not at all.
