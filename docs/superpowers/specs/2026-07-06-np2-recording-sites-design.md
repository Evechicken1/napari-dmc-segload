# napari-dmc-segload — Neuropixels 2.0 recording-site reconstruction

**Date:** 2026-07-06
**Status:** Design pending spec review (revised for metadata-baked electrode template)

## Problem

The plugin's `neuropixels_probe` segmentation type traces a coarse probe track as
a few points, but it cannot reconstruct the location of the **individual
electrodes** of a Neuropixels 2.0 4-shank probe in the histology. We want, per
shank, the histology position of each electrode row/column, derived from the
traced track and the **standard NP2.0 electrode geometry**.

## Scope boundary (important)

There are two distinct mappings:

1. **electrode index → physical location** — *standard* NP2.0 geometry, the same
   for every session. This is what the plugin reconstructs, using a **baked
   template** (no runtime metadata needed).
2. **recording channel → electrode** — *session-specific*; the SpikeGLX
   channelmap/imro can change between recordings. The plugin does **NOT** do
   this. There is no recording-channel column in the output.

Downstream analysis joins recording channel → electrode (via each session's
`.meta`) → the histology coordinates this plugin produces. Keeping the
session-specific channelmap out of the plugin is deliberate.

## Goal

Add a `neuropixels_2.0` segmentation type that:

- Presents four shank **track** layers to trace (one NP2.0 4-shank probe).
- Auto-generates every template electrode's histology coordinate along each
  traced track, reproducing NP2.0's two-column grid at the electrodes' true
  on-shank `(x, z)`.
- Lets the user *see* the generated electrodes before saving.
- Saves one CSV per shank carrying electrode-geometry identity.

## Non-goals

- No recording-channel ↔ electrode mapping; no runtime `.meta` parsing.
- No support for single-shank NP2.0 or multiple probes; exactly one 4-shank
  probe (`n_shanks` parameter defaults to 4).
- No µm/pixel absolute scaling; electrode depths are normalised between the
  template's deepest and shallowest electrode along the traced track.

## Key decisions (from brainstorming)

- **Auto-generate** electrodes from the traced track.
- **NP2.0 4-shank, one probe** — four tracing layers = the four shanks.
- Each shank traced as a **polyline** (ordered points, deepest → shallowest
  electrode); electrodes distributed **by arc-length** so curvature is handled.
- **Endpoints = deepest & shallowest template electrode.** The polyline ends are
  the electrode at `z_min` (tip, z = 0 µm) and the electrode at `z_max` (z = 705
  µm in the baked template); each electrode is placed at arc-length fraction
  `(z − z_min)/(z_max − z_min)`. No µm/pixel scale needed.
- **True two-column geometry.** Each depth row has two electrodes at their real
  on-shank x (27 and 59 µm); the column offset is scaled self-consistently to
  the normalised track (see geometry below).
- **Baked electrode template** from the example geommap (channelmap-independent
  physical layout); **no recording-channel** identity.
- **New seg_type**, one CSV per shank, in the dmc folder layout.

## Baked electrode template

Generated from
`Z:\haeslerlabwip2024\raw_data\ephys\260228_RDP282_g0\260228_RDP282_g0_imec0\260228_RDP282_g0_t0.imec0.ap.meta`
(`snsGeomMap`, probe `NP2010`). Standard NP2.0 4-shank geometry:

- 4 shanks, 250 µm shank pitch, 70 µm shank width.
- Per shank: **48 rows** at z = 0, 15, 30 … 705 µm (15 µm pitch); **2 columns**
  at x = **27 and 59 µm** (32 µm apart, shank-centre 43 µm). 96 electrodes/shank,
  384 total.
- Physical electrode id within a shank: `electrode = row * 2 + col`
  (`row = z / 15`, `col = 0` → x = 27, `col = 1` → x = 59). Channelmap-independent.

Baked as explicit arrays with a provenance header and a one-line note on how to
regenerate (re-parse a `snsGeomMap` if the standard layout ever changes).

## On-disk convention (extends the existing contract)

`segmentation/neuropixels_2.0/shank_<i>/<image_stem>_neuropixels_2.0.csv`

Columns (plus the default pandas index column, matching existing dmc CSVs):

| column | meaning |
|--------|---------|
| `Position Y`, `Position X` | reconstructed histology pixel coords |
| `shank` | shank index 0–3 |
| `electrode` | physical electrode id within shank (`row*2+col`) |
| `row` | depth row index (0 = deepest, z = 0) |
| `col` | column index (0 → x = 27 µm, 1 → x = 59 µm) |
| `x_um`, `z_um` | on-shank physical coordinates (µm) |

One file per shank; shanks with fewer than two traced points write nothing (and
remove a stale file of the same name).

## Components

### 1. `np2_layout.py` (new, pure data — no Qt / napari)

Baked template + a helper returning one shank's electrodes:

```
N_SHANKS = 4
shank_electrodes(shank) -> list of dicts/records with
    {electrode, row, col, x_um, z_um}
```

Backed by explicit `SHANK_Z_UM` (48 values) and `SHANK_X_UM` (`[27, 59]`) arrays
with the provenance/regeneration header.

### 2. `np2_geometry.py` (new, pure logic — no Qt / napari)

```
place_electrodes_on_track(track_vertices, z_um, x_um) -> coords
```

- `track_vertices`: ordered `(N ≥ 2, 2)` array of `(Y, X)` polyline vertices,
  deepest → shallowest.
- `z_um`, `x_um`: per-electrode on-shank coordinates (parallel arrays).
- Compute cumulative arc-length `L`. Let `z_min, z_max = min(z_um), max(z_um)`
  and `z_span = z_max − z_min`. Scale `px_per_um = L / z_span`.
- For each electrode: fraction `f = (z − z_min)/z_span`; interpolate `(Y, X)` at
  arc-length `f·L` on the containing segment; record the segment's **local unit
  tangent**. Offset by the column: `(x − x_centre)·px_per_um` along the local
  perpendicular `n̂`, where `x_centre = (min(x_um)+max(x_um))/2`.
- Returns an `(M, 2)` array of `(Y, X)` coordinates, one per electrode, in the
  input order.
- Edge cases: `< 2` vertices or zero-length track → `ValueError`; `z_span == 0`
  (all electrodes same depth) → `ValueError`.

### 3. `io.py` — add `save_sites`

```
save_sites(coords, records, input_path, seg_type, shank_id, stem) -> Path|None
```

`records` is the per-electrode template info (`electrode, row, col, x_um, z_um`)
aligned with `coords`. Writes the CSV described above under
`segmentation/<seg_type>/<shank_id>/`. Empty `coords` removes a stale file and
returns `None`. Mirrors the existing `save_points_layer` conventions.

### 4. `constants.py`

- `"neuropixels_2.0"` in `ALL_SEG_TYPES` (already added in Task 1).
- `SITE_SEG_TYPES = {"neuropixels_2.0"}` (already added).
- The earlier NP2 even-spacing defaults (`NP2_DEFAULT_ROWS`,
  `NP2_DEFAULT_COLUMNS`, `NP2_ROW_PITCH_UM`, `NP2_COLUMN_SPACING_UM`) are
  **removed** — the layout now lives in `np2_layout.py`.

### 5. `_widget.py`

**No new geometry parameters** (the even-spacing fields are dropped). The
existing `n_probes` spinbox is reused as the shank count (default 4).

**Load flow** — when `seg_type` ∈ `SITE_SEG_TYPES`, `_add_tracing_layers`
creates `n_shanks` **track** point layers named `neuropixels_2.0_track_<i>`.

**"Generate sites" button** → `_generate_sites`: for each
`neuropixels_2.0_track_<i>` layer with ≥ 2 points, pull that shank's template
electrodes, call `place_electrodes_on_track(...)`, and add/replace a points layer
`neuropixels_2.0_sites_<i>`. Tracks with < 2 points are skipped with a
`show_info` note.

**Save flow** — for each shank, regenerate coords from the track + template
(single source of truth) and `save_sites(...)` to `shank_<i>`. Works even if the
Generate button was not pressed.

**Reorder** — `_reorder_layers` includes the new track and site layers in its
bottom group: segmentation → reference → NP2.0 track/site layers.

## Error handling (each surfaces `show_info`, aborts cleanly)

- A shank track with fewer than two points → skipped during generate/save.
- Degenerate (zero-length) track or `z_span == 0` → caught, note shown, shank
  skipped.
- No track layers present when Generate/Save is pressed → note shown.

## Testing

- `test_np2_layout.py`: template has 4 shanks × 96 electrodes; z range 0–705,
  15 µm pitch, 48 rows; x ∈ {27, 59}; `electrode == row*2+col`.
- `test_np2_geometry.py` (pure):
  - Straight vertical track: electrodes land at their true depth fractions; two
    columns symmetric about the centreline with offset `(x−x_centre)·L/z_span`.
  - L-shaped polyline: electrodes distributed by **arc-length**, not vertex index.
  - `< 2` vertices, zero-length track, `z_span == 0` → `ValueError`.
- `test_io.py`: `save_sites` round-trips the CSV with all identity columns; empty
  input removes a stale file.
- `test_widget_smoke.py`: selecting `neuropixels_2.0` + load creates `n_shanks`
  track layers; Generate creates `sites` layers with 96 points/shank; Save writes
  one `shank_<i>` CSV per traced shank with the identity columns; reorder places
  NP2.0 layers at the bottom.

## Risks

- The template is baked from one example config. If the standard recording layout
  changes, the template must be regenerated (documented in `np2_layout.py`). The
  channelmap → electrode join is intentionally downstream, so channelmap changes
  do **not** require touching this plugin.
- Column offset assumes the traced endpoints correspond to the template's z_min
  (tip) and z_max electrodes; if they don't, absolute offsets shift slightly, but
  region-per-depth is unaffected and the two columns are only 32 µm apart.
- Assumes the traced polyline order is deepest → shallowest; row orientation
  depends on it. Documented in the widget tooltip.
