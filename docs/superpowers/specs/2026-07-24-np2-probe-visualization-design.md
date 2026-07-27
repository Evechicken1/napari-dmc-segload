# napari-dmc-segload — NP2.0 probe visualization

**Date:** 2026-07-24
**Status:** Design pending spec review

## Problem

The "NP2.0 electrodes" widget reconstructs electrode locations from dmc's
registered per-shank tract and writes one CSV per shank. Nothing currently shows
what it produced. A shank fitted backwards, a wrong lateral axis, or a tract that
never entered the target structure all look identical on disk: a well-formed CSV.

We want to **see** the reconstruction in 3D and read a few numbers that say
whether it is geometrically plausible.

## Scope

Read-only. This feature consumes the `*_np2_electrodes.csv` files the electrodes
widget already wrote; it never recomputes the reconstruction. Its only write is a
QC text report.

## Verified atlas facts

Checked against the installed `bg_atlasapi` + `allen_mouse_10um` in `napari-env`:

- `atlas.resolution == (10.0, 10.0, 10.0)`; `atlas.space.axes_description ==
  ('ap', 'si', 'rl')`; `atlas.annotation.shape == (1320, 800, 1140)`.
- `atlas.mesh_from_structure(acronym)` returns a meshio mesh whose `.points` are
  **µm in the same axis order** (root spans ≈ 13192 × 7564 × 10890 µm, i.e. the
  voxel shape × 10) and whose `.cells[0].data` are triangles.
- Mesh sizes are small: root 49324 verts / 98638 tris; cortical structures
  (`ORBl`, `MOs`, `VISp6a`) 1.7k–6.3k verts. Rendering is responsive in 3D,
  unlike the ~2.4 GB reference volume.
- `atlas.structures[acronym]` carries `rgb_triplet` (the Allen palette), so
  electrode region colours and mesh colours can be the same values.

## Widget

`src/napari_dmc_segload/_probe_vis_widget.py`, class `ProbeVisWidget`, display
name **"NP2.0 probe visualization"**, registered in `napari.yaml` alongside the
existing three widgets.

Form, following the `_build_form()` pattern of the other widgets:

```python
@magicgui(
    layout="vertical",
    input_path=dict(widget_type="FileEdit", mode="d",
                    label="input path (animal_id): "),
    colour_by=dict(widget_type="ComboBox", label="colour by",
                   choices=["shank", "region", "electrode index"],
                   value="shank"),
    call_button=False,
)
def form(input_path, colour_by): pass
```

The animal folder is always chosen in the widget — no hardcoded paths, no CLI
arguments.

Below the form: one **Visualize probe** button, then a read-only monospace
`QTextEdit` holding the QC report.

Behaviour:

- **Visualize probe** loads the per-shank electrode CSVs, builds the scene, fills
  the QC panel, and writes the QC file.
- Changing **colour by** recolours the existing points in place — no reload, no
  re-read of the CSVs. If no scene is loaded, the change is a no-op.
- Reloading first removes the layers this widget previously added (tracked by
  name), so repeated clicks do not stack layers.
- Clean abort with `show_info` on: invalid input path, missing
  `results/neuropixels_probe`, no `*_np2_electrodes.csv` files (tells the user to
  run "NP2.0 electrodes" first), missing `params.json`, atlas load failure.

## 3D scene

**Coordinate space — world space is µm.** The Points layer holds the raw voxel
coordinates in `(ap_coords, dv_coords, ml_coords)` order with
`scale=atlas.resolution`; Surface layers hold mesh vertices unscaled, since they
are already µm in the same axis order. Neither dataset is mutated to make the two
agree, and the values in the Points layer still match the CSV.

Layers added:

| Layer | Kind | Notes |
|---|---|---|
| `np2 root` | Surface | brain outline, opacity ≈ 0.08 |
| `np2 region: <acronym>` | Surface | one per traversed structure, `rgb_triplet` colour, opacity ≈ 0.25 |
| `np2 electrodes` | Points | all shanks combined, `features` = `shank`, `acronym`, `electrode` |

Region meshes are limited to structures holding at least `MIN_MESH_ELECTRODES`
(10) electrodes, capped at `MAX_MESH_REGIONS` (12) by descending electrode count,
so a shank clipping a dozen slivers does not bury the scene. A structure with no
mesh file is skipped and noted in the QC text rather than raising.

### Colour modes

- **shank** — four distinct categorical colours; shank identity at a glance.
- **region** — each electrode takes its structure's `rgb_triplet`, so a point and
  the mesh it sits in share a colour.
- **electrode index** — continuous colormap over the `electrode` column,
  normalised **jointly across all shanks** so gradients are comparable between
  them. A correctly placed shank reads as a smooth tip→top ramp; a flipped fit
  reads as a reversed ramp; a wrong lateral axis shows the two columns
  disagreeing.

## QC report

Written to `results/neuropixels_probe/<animal>_np2_qc.txt` — one file at the
probe root, not per shank, because half the metrics are cross-shank — and shown
in the panel.

Per shank: electrode count inside the brain, insertion depth (`max(depth_um)`,
the deepest electrode still inside), tip voxel (the coordinates of the row with
the smallest `distance_to_tip_um`), angle from the DV axis, and the **region
ladder** (contiguous runs of acronym ordered by depth, each with its depth range
and electrode count).

Across shanks, for every pair: tip-to-tip distance against the expected
250 µm × index span, and the angle between direction vectors (≈ 0° for a rigid
probe).

Warnings, each naming the shank and the observed vs expected value:

| Condition | Constant | Value |
|---|---|---|
| tip spacing off expected | `SPACING_TOL_UM` | 50 µm |
| pairwise shank angle | `ANGLE_TOL_DEG` | 5° |
| too few electrodes inside brain | `MIN_ELECTRODES` | 50 |
| region ladder entirely `root` | — | — |

Shank direction is **refitted from the electrode CSV itself**, not re-read from
the tract, so QC validates the file that was actually written. This reuses
`np2_tract.recover_line` with a new `dist_col` parameter defaulting to the
current `distance_to_tip(um)`, so the electrode CSV's `distance_to_tip_um` column
can be passed. One added argument to existing tested code, not a second copy of
the fit.

## Components

Three pure modules plus a thin Qt shell, matching the existing grain of the repo
(`np2_layout` / `np2_tract` / `np2_regions` are each one small pure thing).

### 1. `np2_results.py` (new — paths and IO only)

```python
shank_index(folder_name) -> int              # moved from _electrodes_widget._shank_index
shank_dirs(input_path) -> list[Path]         # natsorted; FileNotFoundError if no results dir
load_electrode_csvs(input_path) -> dict[int, DataFrame]
```

`_electrodes_widget.py` drops its private `_shank_index` and inline shank-folder
discovery and imports these instead, so the two widgets share one definition of
where results live. Its behaviour and its existing tests are unchanged.

### 2. `np2_qc.py` (new — pure; no atlas, no Qt)

```python
shank_metrics(df, coord_cols) -> dict
region_ladder(df) -> list[dict]
cross_shank_metrics(frames, coord_cols) -> list[dict]
build_report(frames, coord_cols) -> dict     # {'shanks', 'pairs', 'warnings'}
format_report(report) -> str
```

### 3. `np2_scene.py` (new — pure; atlas-like object injected, imports no napari)

```python
electrode_points(frames, coord_cols) -> (ndarray (N,3), DataFrame features)
point_colors(features, mode, atlas) -> ndarray (N,4) RGBA
region_meshes(frames, atlas, min_electrodes=10, max_regions=12) -> (list[dict], list[str])
                                             # (meshes, acronyms with no mesh)
root_mesh(atlas) -> (vertices, faces)
```

Returns plain numpy arrays and dicts, so the whole scene can be asserted on
without a viewer.

### 4. `_probe_vis_widget.py` (new dock widget)

Form, button, colour combo, QC panel; calls the three modules and pushes the
result into the viewer. Holds the loaded frames and the names of the layers it
added so recolouring and reloading are cheap.

## Testing

TDD throughout: tests first, watched failing, then implementation.

- `test_np2_results.py` — tmp animal tree: discovery order, index parsed from
  folder name not sort position, missing results dir and missing CSVs raise.
- `test_np2_qc.py` — synthetic frames from a known line: metrics match the
  construction; the ladder collapses contiguous runs correctly; each warning
  fires on a deliberately broken frame (a shank moved 400 µm off expected
  spacing, one tilted 15°, one truncated to 20 electrodes) and stays silent on
  good input.
- `test_np2_scene.py` — fake atlas with a structures dict carrying `rgb_triplet`
  and a stub mesh object: point array shape and row order; one colour array per
  mode, including that `electrode index` normalises jointly across shanks; mesh
  selection honours both `min_electrodes` and `max_regions`; a structure with no
  mesh is skipped rather than raising.
- `test_probe_vis_widget_smoke.py` — `make_napari_viewer` with the atlas
  monkeypatched: clicking Visualize adds the expected layers, fills the panel and
  writes the QC file; switching the combo changes colours without changing point
  count; a second click does not duplicate layers.
- Existing suite stays green after the `np2_results` refactor.
- Finally, a manual run against a real animal folder to confirm the scene
  renders.

Tests need a display (napari image/surface layers require a GL context), as the
README already notes.

## Working conventions

- Git worktree `../napari-dmc-segload-probe-vis` on branch `feat/probe-vis`.
- `pytest` run from the worktree with `PYTHONPATH="<worktree>/src"`, because the
  editable install points at the main checkout.
- Environment: `conda activate napari-env`.
- Small commits; README gains a section for the new widget.

## Risks

- Couples to the electrode CSV column contract
  (`shank, electrode, distance_to_tip_um, depth_um, <coord cols>, acronym`).
  That contract is ours, written by the sibling widget, and is asserted with a
  clear error rather than assumed.
- The 250 µm inter-shank spacing check assumes the NP2.0 4-shank geometry; it is
  a warning, never a hard failure, so an unexpected probe variant does not block
  visualization.
- Mesh availability varies by atlas: structures lacking a mesh file are skipped
  with a note. Only `allen_mouse_10um` is verified.
- Angle and spacing tolerances are judgement calls; they live as named module
  constants so they can be tuned once, in one place.
