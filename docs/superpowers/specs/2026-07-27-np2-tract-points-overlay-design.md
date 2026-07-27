# NP2.0 probe visualization — registered tract-points overlay

**Date:** 2026-07-27
**Status:** approved (design)
**Component:** `_probe_vis_widget.py`, `np2_results.py`, `np2_scene.py`

## Problem

The probe-visualization widget renders the *fitted* electrode template (a
least-squares line through each shank's registered tract, then the NP2.0
electrode layout placed along it). Nothing in the scene shows the **registered
tract samples the line was fit to**, so there's no visual read on how well the
fit tracks the actual traced/registered path — a shank whose registration
wandered looks identical to a clean one.

## Goal

Overlay each shank's registered tract samples as an individual napari Points
layer, coloured by shank to match the electrodes, so fit quality is eyeballable:
where fit is good the electrode discs sit inside the tract markers; where
registration wandered the two diverge.

## Data source

`results/neuropixels_probe/neuropixels_probe_<i>/<animal>_neuropixels_probe.csv`
— dmc's registered tract, one per shank folder (~190 rows/shank), with columns
`ap_coords`, `dv_coords`, `ml_coords` (voxel indices), plus `distance_to_tip(um)`,
`depth(um)`, `acronym`. The tract CSV has **no `shank` column**, so the shank
index comes from the folder name — the same rule `load_electrode_csvs` uses.

## Design

### Decisions (from brainstorming)

- Coloured **by shank** (same `SHANK_COLORS` as electrodes) — always, no other
  colour modes for the tract.
- The global max-depth filter **does not** touch the tract — it stays a
  full-length reference while electrodes are filtered.
- **No new form control**: show/hide is napari's native per-layer eye toggle.
- Rendered with the hollow **`ring`** symbol so electrode discs read inside it.

### Units

1. **`np2_results.load_tract_csvs(input_path)`** → `{shank: DataFrame}`. Twin of
   `load_electrode_csvs`, keyed by the folder index (`shank_index`). Per shank
   folder it takes the single `*_neuropixels_probe.csv` (structurally never the
   `_np2_electrodes.csv` file — different suffix). Raises `ValueError` on: two
   candidate CSVs in a folder, a zero-byte CSV, a CSV missing any of the three
   coord columns, an empty CSV, or two folders mapping to the same shank index.
   Raises `FileNotFoundError` if no tract CSV exists anywhere. New constants
   `TRACT_CSV_SUFFIX = "_neuropixels_probe.csv"` and
   `REQUIRED_TRACT_COLS = ("ap_coords", "dv_coords", "ml_coords")`. (No `shank`
   column cross-check — the tract CSV doesn't carry one.)

2. **`np2_scene.tract_points(frames, coord_cols)`** → `(points (N,3) float voxel
   coords, features)` where `features` is a DataFrame with a single `shank`
   column (the dict key broadcast per row). Shanks concatenated in ascending
   index order. Empty frames → `(0,3)` points and an empty `shank` column.
   Colours are produced by reusing the existing
   `point_colors(features, MODE_SHANK, None)` — literally the same shank palette,
   no new colour code.

3. **`_probe_vis_widget`** — new `TRACT_LAYER = "np2 tract points"` and
   `TRACT_POINT_SIZE_VOX = 3.0`:
   - `__init__`: `self._tract_frames = None`.
   - `_clear_layers`: also remove `TRACT_LAYER`.
   - `_visualize`: after the electrode CSVs load, load the tract CSVs in their
     own `try/except` — **failure never aborts the scene**; it sets
     `self._tract_frames = None` and appends a one-line note to the QC panel
     text. After `_render_points`, if tract frames loaded, add the overlay layer
     once: `symbol="ring"`, `size=TRACT_POINT_SIZE_VOX`,
     `scale=tuple(atlas.resolution)`, `face_color` from the shank palette.
   - The tract layer is **static** — it is *not* touched by `_render_points`, so
     colour-mode switches and the depth filter leave it unchanged.

### Why static / built once

Tract colour never changes (always by shank) and the depth filter deliberately
skips it, so there is nothing to re-render on the `colour_by` / `limit_depth` /
`max_depth_um` signals. Building it once in `_visualize` keeps it off the
re-render path entirely.

## Testing

- **`tract_points`** (pure): stack order across shanks, the per-row `shank`
  column equals the dict key, empty frames → `(0,3)` + empty features.
- **`load_tract_csvs`**: keyed by folder index; ignores the `_np2_electrodes.csv`
  file in the same folder; raises on missing coord column, empty CSV, duplicate
  shank folders, and `FileNotFoundError` when none present.
- **conftest**: `np2_animal_dir` gains a synthesized
  `animal01_neuropixels_probe.csv` per shank folder (built from the same line as
  the electrode frames), via a new `make_tract_frame` helper.
- **widget smoke**: after `_visualize` the `TRACT_LAYER` exists, is coloured by
  shank, and `scale == atlas.resolution`; a second `_visualize` does not
  duplicate it; the depth filter changes the electrode count but leaves the
  tract layer's point count unchanged; if tract CSVs are absent the scene still
  builds (electrodes + QC) with no tract layer and no crash.

## Risks

- **Glob picking up the wrong CSV** — mitigated: tract suffix
  `_neuropixels_probe.csv` and electrode suffix `_np2_electrodes.csv` are
  disjoint; a test asserts the electrode CSV is never loaded as a tract.
- **A broken/absent tract aborting the whole visualization** — mitigated: tract
  loading is isolated in its own `try/except` and only ever skips the overlay.
