# napari-dmc-segload — NP2.0 electrode reconstruction from the registered tract

**Date:** 2026-07-17
**Status:** Design pending spec review

## Problem

The earlier `neuropixels_2.0` feature reconstructs electrodes *per 2D section* by
interpolating along a traced polyline. That is anatomically wrong for a probe
that spans many sections, and — more fundamentally — DMC-BrainMap cannot register
a custom seg type, so those 2D points never receive 3D atlas coordinates.

We instead want the electrode reconstruction to happen **after the whole animal
is registered**, from a **single consensus 3D line fit per shank** built from all
the registered track points across all slices, and to place the true NP2.0
electrode geometry along that line.

## Key context (from investigating the installed dmc package)

- dmc hardcodes its seg types (`cells, injection_site, optic_fiber,
  neuropixels_probe, projections`) in both the segment and results widgets, so a
  probe **must be traced as `neuropixels_probe`** to be registered. Our 4 shanks
  are traced as **4 "probes"** (`neuropixels_probe`, number of probes = 4).
- dmc's `results_helpers/tract_calculator.py::TractCalculator` **already** does
  the consensus fit: `Line.best_fit()` over all of a probe-folder's registered
  points, brain-surface detection, and insertion-depth estimation. It runs
  automatically during Results for `neuropixels_probe`. But its electrode model
  is hardcoded **NP1.0** (`192 rows × 20 µm + 175 µm`, single shank, `bank 0`).
- dmc writes its tract to
  `results/neuropixels_probe/neuropixels_probe_<i>/<animal>_neuropixels_probe.csv`
  with columns including `distance_to_tip(um)`, `depth(um)`, and the atlas
  coordinate columns (`ap_coords`, `dv_coords`, `ml_coords`, per
  `atlas.space.axes_description`). The registered raw points are overwritten by
  this tract (`# override file`) — acceptable, since we only need the fit.
- dmc builds the atlas with `bg_atlasapi.BrainGlobeAtlas(params['atlas_info']
  ['atlas'])`; `allen_mouse_10um` is installed (10 µm voxels — matches dmc's
  `/10` scaling).

## Approach (decided)

**Reuse dmc's fitted line; replace only the electrode placement.** dmc's tract
coordinates are an exact linear function of `distance_to_tip`:

```
coords = round( (probe_insert - distance_to_tip)/10 * direction_unit + surface_vox )
```

So per shank we recover the line by least-squares fitting each atlas-coordinate
column against `distance_to_tip(um)`: `coords ≈ a + b·dtt`. The intercept `a` is
the **tip voxel** (dtt = 0) and `−10·b` is the shank's unit direction; the many
rows average out dmc's integer-voxel rounding. `probe_insert` is recovered as
`median(depth(um) + distance_to_tip(um))`. This reuses dmc's surface/insert
anchoring without duplicating its logic.

We read the **tract CSV**, not `neuropixels_probe_data.json` (the JSON's voxel
line has no tip anchoring, so it would force us to re-derive surface/insert).

### Axial rotation / columns

In 3D the probe's rotation about its own axis is not observable from a single
shank, so the 32 µm column offset direction is undetermined. We resolve it from
the **shank geometry**: the vector between the fitted tips of the **left-most and
right-most traced shanks** gives the probe's lateral axis (columns lie along the
same axis as the shanks). Sanity check: its length should be ≈ 250 µm × (shank
index span) (≈750 µm for shanks 0→3). Requires **≥ 2 shanks** traced; with fewer,
error out rather than invent a rotation.

### Placement formula (per shank `i`, per electrode)

```
û_lat = unit( component of (tip_max - tip_min) perpendicular to b_i )   # lateral axis
coords(row,col) = a_i + b_i·(z_um + 175) + ((x_um - 43)/10) · û_lat
depth_um        = probe_insert_i - (z_um + 175)
```

`175 µm` is the tip→first-electrode distance (same as dmc's NP1.0 value; confirmed
for NP2.0). `43 µm` is the shank centre; `x_um ∈ {27, 59}`. Division by 10
converts µm → 10 µm atlas voxels.

## Electrode template

`np2_layout.py` holds the **full physical** NP2.0 4-shank template (not the
recorded subset): **640 rows × 2 columns = 1280 electrodes per shank**,
`z = 0 … 9585 µm` at 15 µm pitch, `x ∈ {27, 59} µm`, `electrode = row*2 + col`.
This is channelmap-independent so any bank/config can be joined downstream.

> Provenance note in code: the 1280/shank extent is the NP2010 spec (to be
> re-verified against a `snsGeomMap`/imro when the Z: share is reachable); the
> 27/59 µm columns and 15 µm pitch were parsed from the example meta.

## Output

One CSV per shank:
`results/neuropixels_probe/neuropixels_probe_<i>/<animal>_np2_electrodes.csv`
(distinct filename so dmc never overwrites it).

Columns: `shank, electrode, row, col, x_um, z_um, distance_to_tip_um, depth_um,
ap_coords, dv_coords, ml_coords, structure_id, acronym, name`.

**Filtered to electrodes inside the brain** (dmc's convention `depth_um ≥ -5`).

## Components

### 1. `np2_layout.py` (extend existing, pure)

Add the full-shank template while keeping the existing helper shape:
`N_SHANKS = 4`, `shank_electrodes(shank) -> list[dict]` with keys
`{shank, electrode, row, col, x_um, z_um}`, now 1280 rows/shank. `SHANK_Z_UM`
becomes `[15*r for r in range(640)]`.

### 2. `np2_tract.py` (new, pure — no atlas, no Qt)

```
load_tract_csv(path) -> DataFrame                       # reads dmc tract CSV
recover_line(tract_df, axes) -> (point_a, dir_b)        # lstsq coords vs dtt
recover_probe_insert(tract_df) -> float                 # median(depth + dtt)
lateral_axis(tip_min, tip_max, shank_dir) -> unit vec   # perp component, normalised
place_electrodes(a, b, insert, lat_axis, electrodes, tip_offset_um=175,
                 shank_centre_um=43, voxel_um=10) -> DataFrame
    # columns: electrode metadata + distance_to_tip_um, depth_um, a/b/c coords
```

Unit-testable end-to-end from a synthetic tract CSV with a known line.

### 3. `np2_regions.py` (new)

```
annotate_regions(coords_df, atlas, coord_cols) -> coords_df + structure_id/acronym/name
```

Takes an *atlas-like* object (`.annotation` ndarray, `.structures` tree,
`.shape`) so tests inject a fake atlas; the widget passes a real
`BrainGlobeAtlas`. Out-of-bounds voxels → `structure_id = 0` / `root`.

### 4. `_electrodes_widget.py` (new dock widget "NP2.0 electrodes")

- `input_path` (animal dir) + **Reconstruct electrodes** button.
- Reads `params.json → atlas_info.atlas`, builds `BrainGlobeAtlas`.
- Discovers shank folders under `results/neuropixels_probe/`; loads each tract
  CSV; recovers per-shank line + insert; computes the lateral axis from the
  extreme shanks; places the template; annotates regions; filters inside-brain;
  writes the per-shank CSV.
- `show_info` + clean abort on: no results dir, < 2 shank folders, missing
  columns in a tract CSV, atlas load failure.
- Registered in `napari.yaml` as a second widget alongside the loader.

### 5. Axis handling

dmc stores coords in the atlas's own axis order (`atlas.space.axes_description`,
e.g. `('ap','si','rl')`). `recover_line`, `place_electrodes`, and
`annotate_regions` operate in that same `(a,b,c)` order end-to-end; the output
columns are named back to `ap/dv/ml_coords` via the fixed dmc map
(`ap→ap, si→dv, rl→ml`).

## Removals (obsolete 2D feature)

- `neuropixels_2.0` from `ALL_SEG_TYPES`; delete `SITE_SEG_TYPES` and its uses.
- `_widget.py`: the SITE branch in `_add_tracing_layers`, `_sites_for_shank`,
  `_generate_sites`, the **Generate NP2.0 sites** button, the SITE save branch,
  and the SITE branch in `_reorder_layers`.
- `io.py`: `save_sites` (+ its tests).
- Delete `np2_geometry.py` and `tests/test_np2_geometry.py` (2D arc-length
  placement).
- Keep `np2_layout.py` (now the full template) and its tests.

## Testing

- `test_np2_layout.py` — update counts: 1280/shank, `z_max = 9585`, ids.
- `test_np2_tract.py` (pure) — build a synthetic tract CSV from a known line +
  insert; assert `recover_line` returns that line (within rounding),
  `recover_probe_insert` the insert; `place_electrodes` puts z=0 at
  `distance_to_tip = 175` and spacing 15 µm → correct voxel steps; a two-shank
  `lateral_axis` recovers the expected direction and offsets columns by ±16 µm/10
  voxels; `depth_um` monotonic.
- `test_np2_regions.py` — fake atlas (small labelled ndarray + stub structures);
  assert correct `structure_id`/`acronym` and out-of-bounds → root.
- `test_electrodes_widget_smoke.py` — fixture animal tree with two
  `results/neuropixels_probe/neuropixels_probe_<i>/` tract CSVs and a
  `params.json`; monkeypatch the atlas; assert two `*_np2_electrodes.csv` written
  with the expected columns and inside-brain filtering.
- Full suite green in `napari-env`.

## Risks

- Couples to dmc's tract-CSV column contract (`distance_to_tip(um)`, `depth(um)`,
  `*_coords`). This is dmc's on-disk output, not runtime internals, but a dmc
  change to those names would break us — asserted explicitly with a clear error.
- Needs ≥ 2 shanks for the lateral axis; documented and enforced.
- 1280/shank extent rests on the NP2010 spec, pending Z:-share re-verification;
  flagged in code.
- Column offset direction assumes columns lie along the shank-to-shank axis
  (true for NP2.0 4-shank); the perpendicular-to-track projection keeps it
  consistent with each shank's own fitted direction.
