# How FFXIV's race skeletons differ

Measured on 2026-09-26 from the TexTools skeleton files of all 18 race/gender combinations: each race's body skeleton (`cXXXXb0001`) and the face skeleton TexTools saved with face 1's model (`cXXXXfYYYY`; for Highlanders face 101, for Hrothgar females face 5). `tools/compare_race_skeletons.py` prints the tables below again from those files. Customize+ in this add-on uses the same data to tell each bone's axes (`magic_fit/cplus/reference.py`).

Useful for converting mods from one race to another, like Universal Mod Converter does.

## Reading TexTools' .skel files

TexTools writes a `.skel` file to its `Skeletons` folder for every skeleton it exports a model with. Each line is one bone as JSON: `BoneName`, `BoneNumber`, `BoneParent` (a `BoneNumber`, -1 for the root), `PoseMatrix` (the bone relative to its parent) and `InversePoseMatrix` (the inverse of the bone in model space). Both are 4x4 matrices of **row vectors** with the translation in the last row, so the bone's model-space matrix in column-vector form is `inverse(InversePoseMatrix)ᵀ`. Model space is the game's: metres, Y up, the character facing +Z (Blender importers turn it to Z up with (x, y, z) → (x, −z, y)).

Face skeletons are partial skeletons rooted at the head bone `j_kao`, placed where the body skeleton puts `j_kao`. To use one with a body, hang it from the body's `j_kao`; bones both have (`j_kao` itself) are the same bone.

## Faces

80 face bones are shared by every race, with the same hierarchy: `j_ago` and `j_f_face` hang from `j_kao`, and every other face bone from `j_f_face`. Two races add bones:

- **Hrothgar** (both): whiskers `j_f_hige_l`, `j_f_hige_r`.
- **Viera** (both): ears `j_zer[a–d]_[a/b]_[l/r]`, 16 bones (one pair of chains per ear shape a to d). **Viera males** also have `j_zacc` (an ear accessory point).

Face layout is measured in the head's frame (`j_kao`), against the Midlander female's. *Face size* is the scale that best fits one layout onto the other; *max after scaling* is what's left once the size is taken out, i.e. how differently the face is shaped. *Turned > 2 deg* counts face bones whose axes differ from the Midlander female's.

| Race | Skeleton of face 1 | Bones | Head (j_kao) height | Face size | Median / max offset | Max after scaling | Turned > 2 deg | Max turn | Extra bones |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Midlander M | c0101f0003 | 80 | 1.614 m | 1.02 | 5 / 14 mm | 10 mm | 36 | 14 deg | - |
| Midlander F | c0201f0002 | 80 | 1.494 m | 1.00 | 0 / 0 mm | 0 mm | 0 | 0 deg | - |
| Highlander M | c0301f0002 | 80 | 1.614 m | 1.01 | 3 / 14 mm | 12 mm | 36 | 14 deg | - |
| Highlander F | c0401f0002 | 80 | 1.494 m | 1.00 | 2 / 9 mm | 9 mm | 40 | 17 deg | - |
| Elezen M | c0501f0005 | 80 | 1.868 m | 1.04 | 4 / 11 mm | 10 mm | 36 | 14 deg | - |
| Elezen F | c0601f0002 | 80 | 1.763 m | 1.02 | 12 / 21 mm | 12 mm | 40 | 12 deg | - |
| Miqo'te M | c0701f0002 | 80 | 1.614 m | 1.04 | 6 / 14 mm | 12 mm | 42 | 17 deg | - |
| Miqo'te F | c0801f0002 | 80 | 1.434 m | 0.98 | 8 / 23 mm | 19 mm | 42 | 12 deg | - |
| Roegadyn M | c0901f0002 | 80 | 2.058 m | 1.45 | 38 / 50 mm | 19 mm | 27 | 13 deg | - |
| Roegadyn F | c1001f0002 | 80 | 1.763 m | 1.11 | 12 / 21 mm | 10 mm | 36 | 14 deg | - |
| Lalafell M | c1101f0002 | 80 | 0.766 m | 1.08 | 25 / 40 mm | 34 mm | 48 | 23 deg | - |
| Lalafell F | c1201f0002 | 80 | 0.766 m | 1.08 | 25 / 40 mm | 34 mm | 48 | 23 deg | - |
| Au Ra M | c1301f0002 | 80 | 1.614 m | 0.99 | 10 / 17 mm | 13 mm | 42 | 17 deg | - |
| Au Ra F | c1401f0002 | 80 | 1.434 m | 0.96 | 3 / 9 mm | 7 mm | 40 | 18 deg | - |
| Hrothgar M | c1501f0002 | 82 | 2.058 m | 1.77 | 73 / 111 mm | 32 mm | 46 | 36 deg | whiskers |
| Hrothgar F | c1601f0002 | 82 | 1.763 m | 1.17 | 35 / 44 mm | 44 mm | 46 | 41 deg | whiskers |
| Viera M | c1701f0002 | 97 | 1.614 m | 1.02 | 7 / 16 mm | 10 mm | 38 | 18 deg | ears, `j_zacc` |
| Viera F | c1801f0002 | 96 | 1.494 m | 0.94 | 7 / 16 mm | 14 mm | 44 | 22 deg | ears |

`j_kao` and `j_f_face` have the same axes in every race, and `j_f_face` sits on `j_kao` everywhere except Hrothgar males (3 mm lower, 16 mm forward).

### Largest turn from the Midlander female's, by region (degrees)

| Region | Bones | Mid M | Mid F | High M | High F | Elez M | Elez F | Miqo M | Miqo F | Roe M | Roe F | Lala M | Lala F | AuRa M | AuRa F | Hroth M | Hroth F | Viera M | Viera F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brows | 10 | 10 | 0 | 10 | 10 | 10 | 10 | 6 | 10 | 10 | 10 | 18 | 18 | 10 | 10 | 10 | 10 | 18 | 6 |
| eyes, lids | 28 | 14 | 0 | 14 | 14 | 14 | 11 | 12 | 10 | 0 | 14 | 21 | 21 | 17 | 14 | 31 | 41 | 14 | 8 |
| nose | 3 | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 8 | 13 | 8 | 7 | 7 | 8 | 8 | 5 | 8 | 8 | 5 |
| cheeks | 8 | 12 | 0 | 12 | 13 | 12 | 12 | 17 | 12 | 10 | 12 | 18 | 18 | 15 | 18 | 27 | 12 | 12 | 15 |
| lips | 20 | 9 | 0 | 9 | 17 | 9 | 9 | 11 | 12 | 9 | 9 | 23 | 23 | 13 | 13 | 36 | 19 | 9 | 22 |
| jaw, teeth, tongue | 8 | 9 | 0 | 9 | 9 | 9 | 6 | 4 | 4 | 8 | 9 | 10 | 10 | 9 | 9 | 14 | 18 | 9 | 4 |

Most face bones have several variants: only 33 of the 99 face bones are turned the same way in every race; mouth, lip, eyelid and cheek bones have up to 10 variants (turns within 2 degrees counting as the same). For many small bones the Midlander female's turn is one no other race shares, 8–14 degrees from the one most races have. Lalafell males and females share one identical face skeleton; no other two races do.

### What that means for converting face mods

- **Bone set.** Face mods for Hrothgar weight the whiskers, and Viera face mods the ear chains; other races have neither. Moving such a mod to another race needs those weights moved to a bone it has (the whiskers' to `j_f_face` or the cheek bones, the ears' to `j_kao` or `j_mimi`), and the reverse direction leaves them unweighted.
- **Shape, not just size.** Faces differ by more than a scale: after scaling out the size, bones are still up to 7–44 mm apart (Hrothgar females most). A mesh moved with the bones follows each bone's own placement: for each vertex, `v' = Σ weight_b · Target_b · inverse(Source_b) · v`, with `Source_b` and `Target_b` the bones' model-space matrices from the two races' skeletons (hung from each body's `j_kao`), is exact where a vertex follows one bone and blends smoothly between bones.
- **Bone axes.** Anything stored per bone in the bone's own frame (animation keys, poses, Customize+ templates, bone-local scaling) turns about different axes on another race's face: up to 23 degrees for Lalafell lips and 41 degrees for Hrothgar eyelids and lips. Converting such data needs `inverse(Target_b) · Source_b` applied to it (the change of frame), not the numbers copied as they are.

## Bodies

| Race | Bones | Hips (n_hara) | Head (j_kao) | Thighs (j_asi_a) | Upper arms apart (j_ude_a) |
| --- | --- | --- | --- | --- | --- |
| Midlander M | 105 | 1.077 m | 1.614 m | 0.962 m | 0.312 m |
| Midlander F | 106 | 1.037 m | 1.494 m | 0.922 m | 0.255 m |
| Highlander M | 101 | 1.077 m | 1.614 m | 0.962 m | 0.312 m |
| Highlander F | 101 | 1.037 m | 1.494 m | 0.922 m | 0.255 m |
| Elezen M | 106 | 1.225 m | 1.868 m | 1.100 m | 0.354 m |
| Elezen F | 106 | 1.220 m | 1.763 m | 1.091 m | 0.332 m |
| Miqo'te M | 106 | 1.077 m | 1.614 m | 0.962 m | 0.312 m |
| Miqo'te F | 106 | 0.999 m | 1.434 m | 0.894 m | 0.235 m |
| Roegadyn M | 106 | 1.315 m | 2.058 m | 1.107 m | 0.622 m |
| Roegadyn F | 101 | 1.220 m | 1.763 m | 1.091 m | 0.332 m |
| Lalafell M | 101 | 0.433 m | 0.766 m | 0.384 m | 0.137 m |
| Lalafell F | 101 | 0.433 m | 0.766 m | 0.384 m | 0.137 m |
| Au Ra M | 106 | 1.077 m | 1.614 m | 0.962 m | 0.312 m |
| Au Ra F | 106 | 0.999 m | 1.434 m | 0.894 m | 0.235 m |
| Hrothgar M | 106 | 1.315 m | 2.058 m | 1.107 m | 0.622 m |
| Hrothgar F | 106 | 1.220 m | 1.763 m | 1.091 m | 0.332 m |
| Viera M | 101 | 1.077 m | 1.614 m | 0.962 m | 0.312 m |
| Viera F | 101 | 1.037 m | 1.494 m | 0.922 m | 0.255 m |

Races share main joint heights in groups (the males of Midlander, Highlander, Miqo'te, Au Ra and Viera; Roegadyn and Hrothgar males; and so on), but only three pairs have identical body skeletons, every bone included: Midlander and Highlander females, Elezen and Roegadyn females, Lalafell males and females.

Bones not every race has:

- `n_sippo_a` to `n_sippo_e` (tail): not in Highlanders, Roegadyn females, Lalafell and Viera.
- `n_hara_noanim_trans`: not in the Midlander male's file, which TexTools wrote in 2024 (older than the others).

Bones turned more than 5 degrees from the Midlander female's:

- `n_buki_tate_r` (shield): 180 degrees for Roegadyn and Hrothgar males (flipped).
- `j_mimi_l`, `j_mimi_r` (ears): 59 degrees Elezen, Roegadyn and Hrothgar females; 57 Lalafell; 53 Elezen males.
- `n_ear_a_l/r`, `n_ear_b_l/r` (earring points): 40 degrees Au Ra.
- `j_kami_a`: 25 degrees Roegadyn and Hrothgar males, 17 Lalafell. `j_kami_f_l/r` 15 and `j_kami_b` 13 degrees for those males (10 Lalafell).
- `j_oya_a/b_l/r` (thumbs): 21 degrees Lalafell. `j_kubi` 16 degrees Lalafell.
- `j_asi_d_l/r` (ankles): 8 degrees Lalafell, 6 degrees the Midlander-sized males. `j_asi_b_l/r` and `n_hizasoubi_l/r` (knees): 6 degrees Lalafell.

Everything else is within 5 degrees of the Midlander female's, most of it identical.
