# NP2.0 probe visualization — raw segmentation-clicks overlay

**Date:** 2026-07-27
**Status:** approved (design)
**Component:** `_probe_vis_widget.py`, new `np2_segpoints.py`; removes the
tract-samples overlay (`load_tract_csvs` / `tract_points`).

## Problem

The probe-vis widget shows the fitted electrode template and, currently, the
*resampled registered tract* (`results/.../<animal>_neuropixels_probe.csv`).
What the user actually wants to see is the **raw manually-clicked segmentation
points** — `segmentation/neuropixels_probe/neuropixels_probe_<i>/<animal>_s*_single_neuropixels_probe.csv`
— registered into the atlas. Those are the actual clicks the whole
reconstruction derives from; seeing them is the real QC on registration.

## Decisions (from brainstorming)

- **Replace** the registered-tract-samples overlay with the raw clicks (remove
  `load_tract_csvs`, `tract_points`, their tests and fixtures).
- Registration channel is chosen from a **dropdown, no auto-render**: nothing
  renders until the user picks a real channel.
- Clicks coloured **by shank** (same `SHANK_COLORS`), distinct symbol.

## The transform (validated)

Raw clicks are 2-D full-res image coordinates (`Position Y`, `Position X`). dmc
turns them into atlas voxels by (1) min-max rescaling from full-res to the
registration-downsampled dims, (2) a landmark warp `sampleDots → atlasDots`
(`fitGeoTrans`/`mapPointTransform`), (3) reading the AP depth off an
angled `z_plane` built from `atlasLocation`, (4) reordering `[x,y,z]` to the
atlas axis order (`xyz_atlas_transform`).

This is reproduced **without** dmc's `SliceHandle`, which unconditionally loads
the ~5 GB atlas annotation we do not need (we want coordinates, not per-click
region names). Instead we reuse dmc's primitives (`fitGeoTrans`,
`mapPointTransform`, `coord_mm_transform`, `xyz_atlas_transform`, `get_bregma`)
plus a copied ~15-line `z_plane`. Verified on XT027 shank 0: the lightweight
path yields coordinates **identical** to `SliceHandle.getBrainArea` for all 35
clicks (0 mismatches). `napari_dmc_brainmap` is imported lazily; if it is not
installed the overlay is skipped with a clear message.

## New module `np2_segpoints.py`

- `SEG_SUBDIR = ("segmentation", "neuropixels_probe")`,
  `SEG_CSV_SUFFIX = "_single_neuropixels_probe.csv"`.
- `registration_channels(input_path)` → sorted sharpy_track channels whose dir
  holds a `registration.json` (drives the dropdown). Empty list if none.
- `shank_click_csvs(input_path)` → `{shank_index: [csv paths]}` discovered under
  the segmentation shank folders (folder-index keying, like the results loaders).
- `rescale_clicks(seg_df, full_hw, low_hw)` → `(N,2)` int `x,y` in registration
  space. Pure; mirrors dmc's `minmax_scale` with the full-res extent appended.
- `register_probe_clicks(input_path, regi_chan, atlas, *, warp=None)` →
  `{shank: (N,3) float ap/dv/ml voxel array}`. Orchestrates discovery, reads the
  full-res and downsampled image dims (cv2) to rescale, and applies the warp per
  section. `warp` is an injectable callable
  `(coords_xy, section_key) -> list[[a,b,c]]`; the default builds it from the
  registration JSON + dmc primitives + atlas. Injection keeps the orchestration
  testable without dmc or an atlas. Clicks that fall outside the volume are
  dropped (dmc does the same).

The atlas passed in is the widget's already-loaded `BrainGlobeAtlas` — only its
`.space` (resolution/axes) and name are used, so no second heavy load.

## Widget changes

- Remove tract state/rendering; new `SEGPOINTS_LAYER = "np2 segmentation points"`,
  `SEGPOINT_SIZE_VOX = 5.0`, symbol `"diamond"`.
- New form control `seg_regi_channel` (ComboBox). On `_visualize` its choices are
  set to `["(none)"] + registration_channels(input_path)` with value `"(none)"`
  (no auto-render), and any existing clicks layer is cleared via `_clear_layers`.
- `seg_regi_channel.changed` → `_render_segpoints`: if value is a real channel and
  a scene is loaded, `register_probe_clicks(...)`, stack to points + a `shank`
  feature, colour via `point_colors(features, MODE_SHANK, None)`, add/replace the
  `SEGPOINTS_LAYER` (scale = `atlas.resolution`); if `"(none)"`, remove the layer.
- Static overlay: depth filter and electrode colour-mode never touch it.
- Robust: missing registration/segmentation, dmc import failure, or a transform
  error are caught and reported via `show_info`; the electrode scene still builds.

## Testing

- `rescale_clicks` (pure): a known full/low ratio maps a click to the expected
  rounded `x,y`; the appended full-res extent is stripped from the output.
- `registration_channels`: finds only channels with a `registration.json`.
- `shank_click_csvs`: folder-index keying; ignores non-matching files.
- `register_probe_clicks` with an **injected fake warp** (identity-ish) over a
  tiny fake animal tree: correct shank keys, per-shank row counts, and that the
  rescale feeds the warp — no dmc/atlas needed.
- Real-data check (kept as a scratch verification, not CI): lightweight coords
  equal `getBrainArea` on XT027.
- Widget smoke (`make_napari_viewer`, monkeypatched `register_probe_clicks` so no
  dmc/atlas in CI): dropdown populates with `(none)` default and no clicks layer;
  selecting a channel adds `SEGPOINTS_LAYER` coloured by shank; back to `(none)`
  removes it; the depth filter leaves the clicks layer untouched; a second
  `_visualize` does not duplicate it.

## Risks

- **Wrong full-res dims** (rescale ratio) → offset clicks. Mitigated: dims are
  read from the section's actual full-res image, matching dmc; verified identical
  on real data.
- **dmc not installed / API drift** → mitigated by the lazy import + graceful
  skip, and by reusing dmc's own primitives rather than reimplementing the warp.
- **Removing the tract overlay** must also drop its now-dead loader, scene
  function, fixtures and tests so nothing references them.
