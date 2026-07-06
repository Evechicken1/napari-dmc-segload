# napari-dmc-segload — Neuropixels 2.0 recording-site reconstruction

**Date:** 2026-07-06
**Status:** Design pending spec review

## Problem

The plugin's `neuropixels_probe` segmentation type lets the user trace a coarse
probe track as a few points, but it cannot reconstruct the location of the
**individual recording sites** along a Neuropixels 2.0 4-shank probe. For
channel-to-region assignment we want, per shank, the position of each electrode
row (and column) in the histology image, derived from the traced track and the
known NP2.0 site geometry.

## Goal

Add a segmentation type that:

- Presents four shank **track** layers to trace (one NP2.0 4-shank probe).
- Auto-generates the recording-site coordinates by distributing them along each
  traced track, reproducing NP2.0's two-column electrode grid.
- Lets the user *see* the generated sites before saving.
- Saves one CSV per shank in the existing dmc segmentation folder layout.

## Non-goals

- No channel-map / SpikeGLX `.meta` parsing; site counts come from widget
  geometry parameters.
- No µm/pixel scaling or absolute-depth placement; sites are normalised to the
  traced endpoints (robust to tissue shrinkage).
- No support (yet) for single-shank NP2.0 or multiple probes; exactly one
  4-shank probe. (`n_shanks` is a parameter, so other counts work, but 4 is the
  designed-for case.)

## Key decisions (from brainstorming)

- **Auto-generate** sites from the traced track (not manual placement).
- **NP2.0 4-shank, one probe** — the four tracing layers are the four shanks.
- Each shank traced as a **polyline** (ordered points, deepest → shallowest
  site); sites distributed **by arc-length** so curvature is handled.
- **Endpoints = first & last recorded site.** The two ends of the polyline are
  the deepest and shallowest recorded rows; the `n_site_rows` rows are placed at
  even arc-length fractions between them. No µm/pixel scale is needed.
- **Two-column grid.** Each row yields two sites (NP2.0's two columns), offset
  horizontally from the track. `n_rows × 2` points per shank.
- **New seg_type**, one CSV per shank, in the dmc layout.

## NP2.0 geometry reference

- 4 shanks, 250 µm shank pitch, 70 µm shank width.
- Electrodes in **2 columns**, 32 µm horizontal spacing, 15 µm vertical (row)
  pitch. 384 channels recordable at once (≈96 sites / 48 rows per shank when
  spread over all four shanks — a sensible default, but configuration-dependent).

## On-disk convention (extends the existing contract)

`segmentation/neuropixels_2.0/shank_<i>/<image_stem>_neuropixels_2.0.csv`

Columns: `Position Y`, `Position X`, `row`, `col` (plus the default pandas index
column, matching the existing dmc points CSVs). `row` is `0 … n_rows-1` with
`row 0` = the first traced endpoint (deepest site); `col ∈ {0, 1}`. One file per
shank; shanks with fewer than two traced points write nothing (and remove a
stale file of the same name).

## Components

### 1. `np2_geometry.py` (new, pure logic — no Qt / napari)

```
generate_site_coords(track_vertices, n_rows, n_columns,
                     row_pitch_um, column_spacing_um) -> (coords, rows, cols)
```

- `track_vertices`: ordered `(N, 2)` array of `(Y, X)` polyline vertices
  (deepest → shallowest). Requires `N ≥ 2`.
- Compute cumulative arc-length along the polyline. For each row
  `k = 0 … n_rows-1`, the target arc-length is `k/(n_rows-1) · L` (`L` = total
  length). Interpolate the `(Y, X)` position on the segment that contains that
  arc-length; record the **local unit tangent** of that segment.
- For `n_columns == 2`: the horizontal offset that separates the two columns is
  `offset_px = (column_spacing_um / row_pitch_um) · per_row_arc_px`, where
  `per_row_arc_px = L / (n_rows-1)` is the pixel length of one row step (which
  represents `row_pitch_um`). This keeps the column offset self-consistent with
  the normalised track. Place `col 0` at `P − ½·offset_px·n̂` and `col 1` at
  `P + ½·offset_px·n̂`, where `n̂` is the local tangent rotated 90°. (The sign
  convention for which column is 0 vs 1 is fixed but arbitrary; it does not
  affect the anatomical location.)
- For `n_columns == 1`: one point per row on the track centreline.
- Returns the flattened list of site coordinates plus parallel `row` and `col`
  index arrays, ordered row-major.
- Edge cases: `n_rows == 1` places a single row at the deepest endpoint with no
  column scaling (offset 0); a zero-length track (all vertices coincident)
  raises `ValueError`.

### 2. `io.py` — add `save_sites`

```
save_sites(coords, rows, cols, input_path, seg_type, shank_id, stem) -> Path|None
```

Writes the four-column CSV described above under
`segmentation/<seg_type>/<shank_id>/`. Empty `coords` removes a stale file and
returns `None`. Mirrors the existing `save_points_layer` conventions.

### 3. `constants.py`

- Add `"neuropixels_2.0"` to `ALL_SEG_TYPES`.
- `SITE_SEG_TYPES = {"neuropixels_2.0"}` (handled separately from
  `PROBE_SEG_TYPES`).
- Defaults: `NP2_DEFAULT_ROWS = 48`, `NP2_DEFAULT_COLUMNS = 2`,
  `NP2_ROW_PITCH_UM = 15.0`, `NP2_COLUMN_SPACING_UM = 32.0`.

### 4. `_widget.py`

**New parameters** (magicgui form): `n_site_rows` (SpinBox, default 48),
`n_columns` (SpinBox 1–2, default 2), `row_pitch_um` (FloatSpinBox, 15.0),
`column_spacing_um` (FloatSpinBox, 32.0). The existing `n_probes` spinbox is
reused as the shank count for this type. (These extra fields are always present;
they are simply only meaningful for `neuropixels_2.0`.)

**Load flow** — when `seg_type` ∈ `SITE_SEG_TYPES`, `_add_tracing_layers`
creates `n_shanks` **track** point layers named `neuropixels_2.0_track_<i>` for
tracing the polyline. (Image reference/segmentation layers load unchanged.)

**New "Generate sites" button** → `_generate_sites`:
- For each `neuropixels_2.0_track_<i>` layer with ≥ 2 points, call
  `generate_site_coords(...)` with the widget parameters and add (or replace) a
  points layer `neuropixels_2.0_sites_<i>` showing the reconstructed grid.
- Tracks with < 2 points are skipped with a `show_info` note.

**Save flow** — when `seg_type` ∈ `SITE_SEG_TYPES`: for each shank, if a
`neuropixels_2.0_sites_<i>` layer is missing, generate it on the fly from the
track (so Save works even if the user skipped the button); then `save_sites(...)`
to `shank_<i>`. The generated sites layer carries no row/col metadata itself, so
the coordinates are regenerated at save time to recover `row`/`col` indices
(single source of truth = `generate_site_coords`).

**Reorder** — `_reorder_layers` includes the new track and site layers in its
bottom group, preserving the established order: segmentation → reference →
NP2.0 track/site layers.

## Error handling (each surfaces `show_info`, aborts cleanly)

- A shank track with fewer than two points → skipped during generate/save.
- Degenerate (zero-length) track → caught, note shown, that shank skipped.
- No track layers present when Generate/Save is pressed → note shown.

## Testing

- `test_np2_geometry.py` (pure):
  - Straight vertical track, `n_columns=1`: rows evenly spaced along the track,
    endpoints coincide with the traced ends.
  - Straight track, `n_columns=2`: two columns symmetric about the centreline
    with the expected `offset_px`; correct total point count `n_rows·2`.
  - L-shaped polyline: rows distributed by **arc-length**, not vertex index.
  - `n_rows == 1`, single coincident vertices, `< 2` vertices — edge behaviour.
- `test_io.py`: `save_sites` round-trips the four-column CSV; empty input removes
  a stale file.
- `test_widget_smoke.py`: selecting `neuropixels_2.0` + load creates
  `n_shanks` track layers; Generate creates `sites` layers with `n_rows·n_cols`
  points; Save writes one `shank_<i>` CSV per traced shank with `row`/`col`
  columns; reorder places NP2.0 layers at the bottom.

## Risks

- Column offset uses the row-pitch-derived pixel scale; if the traced endpoints
  do **not** actually correspond to the first/last recorded row, the absolute
  offset is off — but the anatomical location (region per depth) is unaffected,
  and the two columns are only 32 µm apart. Acceptable.
- Assumes the traced polyline order is deepest → shallowest; the row index
  orientation depends on it. Documented in the widget tooltip.
