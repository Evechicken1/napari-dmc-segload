# Flat DV depth cut referenced to the shallowest shank entry

**Date:** 2026-07-28
**Status:** Design approved, pending implementation
**Component:** NP2.0 probe visualization (`_probe_vis_widget`, `np2_scene`)

## Problem

The probe-visualization widget's "limit depth" filter trims each shank at
`depth_um <= max_depth`. Because `depth_um` is measured per shank from *that
shank's own* dura crossing (`depth_um = probe_insert - distance_to_tip`), a
single `max_depth` value lands at a different absolute dorsoventral (DV) level on
each shank whenever the shanks entered the brain at different depths — an angled
insertion into sloped cortex. The retained electrodes then have a staircased top
edge rather than a clean level.

We want the cut referenced to **one** common point — the highest (most dorsal)
entry point across all loaded shanks — producing a flat DV plane: every shank is
trimmed at the same absolute DV level.

## Chosen approach: flat DV plane (Option A)

Keep electrodes whose absolute `dv_coords` is within `max_depth` (converted to
voxels) below the shallowest entry point across shanks. The cut is a horizontal
DV plane. This was chosen over an "along-probe common zero" (a cut perpendicular
to the insertion axis) because it uses `dv_coords` already present in the CSVs,
needs no line re-fitting, and matches the natural reading of "highest entry
point" as a horizontal reference. For a vertical probe the two are identical;
they diverge only for angled insertions.

Trim direction is unchanged from today: the deep/tip end beyond the cut is
dropped, the shallow band from the reference down to the cut is kept.

## Data available

Each shank's `*_np2_electrodes.csv` (loaded into `frames = {shank_idx: DataFrame}`)
carries, per electrode: `shank`, `electrode`, `distance_to_tip_um`, `depth_um`,
`acronym`, and absolute atlas voxel coords `ap_coords` / `dv_coords` / `ml_coords`.
The dura crossing itself is **not** stored, so the entry point is recovered by
extrapolation.

## Components

### 1. `np2_scene.highest_entry_dv(frames, dv_col)` — new pure helper

For each shank, linearly fit `dv_coords` against `depth_um` and take the
intercept at `depth_um = 0` — the DV coordinate of the dura crossing (entry
point). Return the **minimum** intercept across shanks (most dorsal = smallest
DV), in atlas voxels.

- Uses `np.polyfit(depth_um, dv_coords, 1)` per shank; the intercept is the entry
  DV. Many colinear electrodes make this robust to per-voxel rounding.
- Requires each shank to have >= 2 electrodes spanning >= 2 distinct `depth_um`
  values (always true for real reconstructions). A shank with a single distinct
  depth is degenerate; treat as a `ValueError` surfaced to the widget via
  `show_info` (consistent with existing error handling).
- Pure numpy/pandas, atlas-free, directly unit-testable.

### 2. `np2_scene.electrode_points(...)` — change the depth filter

When `max_depth` is not `None`, replace the current per-shank
`df["depth_um"] <= max_depth` with a flat DV cut:

```
df[dv_col] <= dv_ref + max_depth / voxel_um
```

- `dv_ref` (voxels), `dv_col` (the DV coordinate column name), and `voxel_um`
  (DV-axis resolution, um per voxel) are passed in as keyword arguments, so the
  function stays atlas-free and testable.
- `max_depth / voxel_um` converts the um threshold to voxels to compare against
  voxel `dv_coords`.
- When `max_depth is None`, behavior is unchanged (no filter); the new keyword
  args are then unused/optional.

### 3. `_probe_vis_widget._render_points` — wire it up

Before calling `electrode_points`, when `limit_depth` is on:

- `dv_col = coord_cols(atlas)`-derived DV column name (`"dv_coords"`).
- `dv_ref = highest_entry_dv(self._frames, dv_col)`.
- `voxel_um` = the atlas resolution for the si (DV) axis, from
  `atlas.space.resolution` indexed by the position of `"si"` in
  `atlas.space.axes_description`.
- Pass `dv_ref`, `dv_col`, `voxel_um` into `electrode_points`.

A `ValueError` from `highest_entry_dv` (degenerate shank) is caught and reported
via `show_info`, leaving the current scene intact — matching how the widget
already handles reconstruction/loading errors.

## Assumptions and edge cases

- **Atlas orientation:** assumes an ap/si/rl-oriented atlas where `dv_coords`
  increases ventrally, so the most dorsal entry is the minimum DV. This is the
  allen-mouse convention the plugin already relies on (see `coord_cols`).
- **Negative `max_depth`:** the spinbox allows down to `-1`; a negative threshold
  simply places the cut above the reference (keeps little/nothing). Preserved
  as-is, no special-casing.
- **Reference scope:** computed only over the shanks currently loaded/visualized
  (`self._frames`), not all traced shanks on disk.
- **Untouched:** colour modes (`shank` / `region` / `electrode index`), the
  segmentation-points overlay, region/ROI meshes, and the QC report. The widget
  label stays "max depth (um)".

## Testing

- `highest_entry_dv`: synthetic frames with staggered entry points; assert the
  returned reference equals the min per-shank intercept; assert a degenerate
  (single-depth) shank raises `ValueError`.
- `electrode_points` flat cut: staggered synthetic shanks; assert every shank's
  retained electrodes fall at or above the same absolute DV level, and that a
  deeper-entering shank loses more electrodes than the reference shank.
- Vertical-probe equivalence: with all entries at the same DV, the flat cut
  reproduces the old per-shank `depth_um <= max_depth` result (within voxel
  rounding).
- Widget smoke: `_render_points` with `limit_depth` on runs end-to-end against a
  fake atlas exposing `space.resolution` / `space.axes_description`.
