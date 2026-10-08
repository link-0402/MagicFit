# Resize bench

Measures how Resize moves real clothing between body sizes, so a change to the engine (`magic_fit/resizing.py`)
can be compared with what it replaces. Not a test: it prints numbers, not pass or fail.

Everything runs headless with `--factory-startup`. The data lives in `bench_data/` at the repository root
(ignored by git).

## Data

`bench_data/body.npz` is a devkit body at every size of the grid Chest (Large, Medium, Small) x Lavabod x Rue x
Buff, its other settings as they are in the scene. `bench_data/garments/<tag>.npz` are garments, each fitted to
one of those sizes. Make them once:

```
blender -b --factory-startup --python tools/resize_bench/dump_body.py -- SCENE.blend bench_data --clothing "XIV Instant Edit"
blender -b --factory-startup --python tools/resize_bench/dump_garments.py -- ITEMS.tsv bench_data --blend wip2.blend
```

- `dump_body.py` takes the From Body of the scene's Resize panel. With `--clothing`, the meshes of that
  collection, hidden ones too, become the garment `snapshot`, fitted to the size the body shows. The scene used
  so far is an Instant Edit export's snapshot: the YAB 4.2 Mannequin, at Medium Rue, with a backless sleeve top,
  its skin and its piercings.
- `dump_garments.py` reads a list of garments: game models (`.mdl`, every submesh a mesh, skin ones too) and
  objects of .blend files. They're resized from the size they fit best (picked over the parts of the body the
  sizes change).

## Running

```
.\tools\resize_bench\run.ps1 -Out bench_data\runs\base
.\tools\resize_bench\run.ps1 -Checkout <worktree> -Ref bench_data\runs\base
```

`run.ps1` runs `bench.py` in 8 Blenders (`-Shards`) and prints the totals. With `-Ref`, it also prints the
reference run's totals, their ratio and how many garment sizes got better or worse. Each garment goes to Small
and to Omoi (Large Lavabod) from its own size, keeping its Rue and Buff, and back (`-Targets all` takes every
size). `bench.py` sets the meshes up with the checkout's own `api._Sizes`, as the add-on does (skin, rigid
parts, Keep Close's edges), and moves them with `resizing.ShapeChange`. So the measured checkout must keep
those.

`summarize.py RUN [--ref REF] [--rows] [--worst K]` prints the same from saved runs. `--rows` lists every
garment and size, and `--worst` lists the worst vertices with their positions.

## Metrics

`metrics.py` measures every checkout the same way, with numpy and mathutils only. Lower is better for all of
them.

| Metric | What it is |
|---|---|
| tight / near / mid | Mean change of the distance from the skin, for cloth 0-3, 3-6 and 6-15 mm off it (mm, area weighted) |
| clip_out | Vertices that were outside the skin and end up more than 1 mm inside |
| rigid_max, rigid_1mm | Clearance change of rigid parts worn on the skin (piercings, studs), per part |
| thick_p95, thick_1mm | Change of the distance between layers (a vertex and the next sheet along its normal, within 6 mm) |
| detail_p95, detail_max | Change of each vertex's Laplacian along its normal, in its own frame: bumps and folds flattened or exaggerated. The body's own change under the garment is in the JSON as `body_floor_mm` |
| flipped | Triangles turned over |
| rt_p95, rt_max | How far vertices land from where they were after going to the size and back |

Skin meshes (most faces with the game's skin material, mt_cNNNNbNNNN_bibo/_b/_a/_eve, or one named skin) are
measured apart (`skin` in the JSON), the same way for every checkout. Every garment's JSON also holds the clearance
per body region and the 10 worst vertices of each metric.

## Run time

About 5 s per garment and size, round trip included; the 61 garments, each to two sizes and back, take about 2 minutes in 8 shards.
