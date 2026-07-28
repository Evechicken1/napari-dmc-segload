# Flat DV Depth Cut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the probe-vis "limit depth" filter cut all shanks at one flat dorsoventral level referenced to the shallowest shank entry point, instead of cutting each shank at its own dura.

**Architecture:** A new pure helper `highest_entry_dv` (in `np2_scene`) recovers each shank's dura-crossing DV by extrapolating `dv_coords` vs `depth_um` to `depth_um=0` and returns the most-dorsal (minimum). `electrode_points` gains a flat-cut branch (`dv_coords <= dv_ref + max_depth/voxel_um`) driven by keyword args, so it stays atlas-free and testable. `_probe_vis_widget._render_points` computes `dv_ref` and the DV-axis voxel size and passes them through.

**Tech Stack:** Python, numpy, pandas, magicgui/qtpy, napari, pytest. Conda env `napari-env` (base python lacks numpy/napari). Run tests from the main checkout with `python -m pytest`; from a worktree prepend `PYTHONPATH="<worktree>/src"`.

**Repo:** `C:\Users\xaand\Documents\PhD\Analysis\Informatics\Python\napari-dmc-segload`

---

### Task 1: `highest_entry_dv` helper (recover the shallowest entry point)

**Files:**
- Modify: `src/napari_dmc_segload/np2_scene.py` (add function after `electrode_points`)
- Test: `tests/test_np2_scene.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_np2_scene.py`:

```python
from napari_dmc_segload.np2_scene import highest_entry_dv


def test_highest_entry_dv_returns_most_dorsal_entry():
    # shank 0 inserted deeper (larger insert) -> more dorsal (smaller dv) entry.
    # For this synthetic geometry entry dv = 400 - 0.1*insert -> 0.0 and 100.0.
    frames = {0: make_electrode_frame(0, insert=4000.0),
              1: make_electrode_frame(1, insert=3000.0)}
    ref = highest_entry_dv(frames, "dv_coords")
    assert ref == pytest.approx(0.0, abs=1.0)


def test_highest_entry_dv_raises_on_degenerate_shank():
    import pandas as pd
    frames = {0: pd.DataFrame({"depth_um": [50.0, 50.0, 50.0],
                               "dv_coords": [7, 7, 7]})}
    with pytest.raises(ValueError, match="entry point"):
        highest_entry_dv(frames, "dv_coords")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_np2_scene.py::test_highest_entry_dv_returns_most_dorsal_entry tests/test_np2_scene.py::test_highest_entry_dv_raises_on_degenerate_shank -v`
Expected: FAIL with `ImportError: cannot import name 'highest_entry_dv'`.

- [ ] **Step 3: Write the implementation**

In `src/napari_dmc_segload/np2_scene.py`, add this function immediately after `electrode_points` (before `point_colors`):

```python
def highest_entry_dv(frames, dv_col):
    """DV voxel coordinate of the shallowest (most dorsal) shank entry point.

    Each shank's electrodes are colinear, so `dv_col` is a linear function of
    `depth_um`; the intercept at depth_um=0 is that shank's dura crossing (entry
    point). Returns the minimum such intercept across shanks — the highest entry
    point — as a float in atlas voxels. Assumes an atlas where dv increases
    ventrally (allen convention), so most dorsal = smallest dv. Raises ValueError
    if any shank has fewer than two distinct depths (cannot locate its entry).
    """
    entries = []
    for shank in sorted(frames):
        df = frames[shank]
        depth = df["depth_um"].to_numpy(dtype=float)
        dv = df[dv_col].to_numpy(dtype=float)
        if len(np.unique(depth)) < 2:
            raise ValueError(
                f"shank {shank}: need >= 2 distinct depths to locate the entry point")
        _slope, intercept = np.polyfit(depth, dv, 1)
        entries.append(float(intercept))
    return min(entries)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_np2_scene.py::test_highest_entry_dv_returns_most_dorsal_entry tests/test_np2_scene.py::test_highest_entry_dv_raises_on_degenerate_shank -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_scene.py tests/test_np2_scene.py
git commit -m "feat: add highest_entry_dv to recover shallowest shank entry point"
```

---

### Task 2: flat DV cut in `electrode_points`

**Files:**
- Modify: `src/napari_dmc_segload/np2_scene.py:34-56` (`electrode_points`)
- Test: `tests/test_np2_scene.py`

- [ ] **Step 1: Write the failing tests**

Add these new tests to the end of `tests/test_np2_scene.py`:

```python
def test_electrode_points_flat_cut_is_common_across_staggered_shanks():
    import numpy as np
    from conftest import NP2_TIP0, NP2_SHANK_PITCH_VOX
    # Parallel shanks translated in DV: shank i's tip (and hence its whole
    # electrode DV range) sits 100 voxels deeper than shank i-1, so their
    # entry points are staggered — the case where a flat DV cut actually
    # differs from a per-shank one. (Varying `insert` alone does NOT stagger
    # the physical dv_coords, only the depth_um labels, so it cannot be used.)
    def tip(i):
        return NP2_TIP0 + np.array([0.0, 100.0 * i, NP2_SHANK_PITCH_VOX * i])
    frames = {0: make_electrode_frame(0, tip=tip(0)),
              1: make_electrode_frame(1, tip=tip(1))}
    dv_ref = highest_entry_dv(frames, "dv_coords")          # shank 0, ~= 0.0
    max_depth, voxel_um = 3500.0, 10.0
    dv_cut = dv_ref + max_depth / voxel_um                   # ~= 350 voxels
    pts, feats = electrode_points(frames, COLS, max_depth=max_depth,
                                  dv_col="dv_coords", dv_ref=dv_ref, voxel_um=voxel_um)
    dv_index = COLS.index("dv_coords")
    # every kept electrode (any shank) is at or above the SAME absolute dv level
    assert np.all(pts[:, dv_index] <= dv_cut + 1e-6)
    # the deeper-entering shank keeps fewer electrodes than the reference shank
    kept0 = int((frames[0]["dv_coords"] <= dv_cut).sum())
    kept1 = int((frames[1]["dv_coords"] <= dv_cut).sum())
    assert 0 < kept1 < kept0
    assert len(pts) == kept0 + kept1 == len(feats)


def test_electrode_points_flat_cut_requires_dv_args():
    frames = {0: make_electrode_frame(0)}
    with pytest.raises(ValueError, match="dv_col"):
        electrode_points(frames, COLS, max_depth=1000.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_np2_scene.py::test_electrode_points_flat_cut_is_common_across_staggered_shanks tests/test_np2_scene.py::test_electrode_points_flat_cut_requires_dv_args -v`
Expected: FAIL — the first because the old per-shank `depth_um` filter is applied (staircased, `kept1 < kept0` / signature args rejected), the second because no `ValueError` is raised.

- [ ] **Step 3: Replace `electrode_points`**

In `src/napari_dmc_segload/np2_scene.py`, replace the whole `electrode_points` function (currently lines 34-56) with:

```python
def electrode_points(frames, coord_cols, max_depth=None, *,
                     dv_col=None, dv_ref=None, voxel_um=None):
    """Stack every shank's electrodes into one point array + feature table.

    Shanks are concatenated in ascending index order, rows in file order.
    Returns (points (N,3) float voxel coords, features DataFrame with columns
    shank, acronym, electrode).

    If `max_depth` (um) is not None, a flat dorsoventral cut is applied,
    referenced to one common entry point rather than each shank's own dura:
    electrodes whose `dv_col` voxel coordinate lies more than `max_depth` below
    `dv_ref` (the shallowest shank entry, in voxels) are dropped. `dv_col`,
    `dv_ref` and `voxel_um` (the DV-axis um-per-voxel resolution) are then all
    required. The deep/tip end is trimmed; the shallow band is kept.
    """
    if max_depth is not None:
        if dv_col is None or dv_ref is None or voxel_um is None:
            raise ValueError(
                "max_depth filter requires dv_col, dv_ref and voxel_um")
        dv_cut = dv_ref + max_depth / voxel_um
    points, feats = [], []
    for shank in sorted(frames):
        df = frames[shank]
        if max_depth is not None:
            df = df[df[dv_col] <= dv_cut]
        points.append(df[coord_cols].to_numpy(dtype=float))
        feats.append(pd.DataFrame({
            "shank": df["shank"].to_numpy(dtype=int),
            "acronym": df["acronym"].astype(str).to_numpy(),
            "electrode": df["electrode"].to_numpy(dtype=int),
        }))
    pts = np.concatenate(points, axis=0) if points else np.zeros((0, 3))
    return pts, pd.concat(feats, ignore_index=True)
```

- [ ] **Step 4: Update the three existing tests that used the old per-shank filter**

In `tests/test_np2_scene.py`, replace `test_electrode_points_max_depth_filters_inclusive` (currently lines 187-194) with:

```python
def test_electrode_points_flat_cut_single_shank_matches_depth():
    # one shank: dv_ref is its own entry, so the flat cut reduces to depth<=max.
    frames = {0: make_electrode_frame(0, n_rows=100)}
    full, _ = electrode_points(frames, COLS)
    dv_ref = highest_entry_dv(frames, "dv_coords")
    dmax, voxel_um = float(frames[0]["depth_um"].median()), 10.0
    dv_cut = dv_ref + dmax / voxel_um
    pts, feats = electrode_points(frames, COLS, max_depth=dmax,
                                  dv_col="dv_coords", dv_ref=dv_ref, voxel_um=voxel_um)
    kept = (frames[0]["dv_coords"] <= dv_cut).sum()
    assert len(pts) == kept == len(feats)
    assert 0 < len(pts) < len(full)
```

Replace `test_electrode_points_max_depth_below_all_gives_empty` (currently lines 205-208) with:

```python
def test_electrode_points_max_depth_below_all_gives_empty():
    frames = {0: make_electrode_frame(0)}
    dv_ref = highest_entry_dv(frames, "dv_coords")
    pts, feats = electrode_points(frames, COLS, max_depth=-1e9,
                                  dv_col="dv_coords", dv_ref=dv_ref, voxel_um=10.0)
    assert pts.shape == (0, 3) and len(feats) == 0
```

Replace `test_point_colors_empty_features_all_modes` (currently lines 211-216) with:

```python
def test_point_colors_empty_features_all_modes():
    import numpy as np
    frames = {0: make_electrode_frame(0)}
    dv_ref = highest_entry_dv(frames, "dv_coords")
    _, feats = electrode_points(frames, COLS, max_depth=-1e9,
                                dv_col="dv_coords", dv_ref=dv_ref, voxel_um=10.0)
    for mode in ("shank", "region", "electrode index"):
        c = point_colors(feats, mode, _RoiAtlas())
        assert c.shape == (0, 4)
```

(`test_electrode_points_max_depth_none_is_unchanged` needs no change — the `None` path is unchanged.)

- [ ] **Step 5: Run the whole scene test module**

Run: `python -m pytest tests/test_np2_scene.py -v`
Expected: PASS (all tests in the module, including the four new/updated ones).

- [ ] **Step 6: Commit**

```bash
git add src/napari_dmc_segload/np2_scene.py tests/test_np2_scene.py
git commit -m "feat: flat DV depth cut in electrode_points referenced to a common entry"
```

---

### Task 3: wire the flat cut into the probe-vis widget

**Files:**
- Modify: `src/napari_dmc_segload/_probe_vis_widget.py` (imports, `_render_points`, add `_dv_voxel_um`)
- Test: `tests/test_probe_vis_widget_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to the end of `tests/test_probe_vis_widget_smoke.py`:

```python
def _write_staggered_animal(base, inserts):
    import json
    from napari_dmc_segload.np2_results import ELECTRODE_CSV_SUFFIX
    from conftest import make_electrode_frame
    root = base / "results" / "neuropixels_probe"
    for shank, ins in enumerate(inserts):
        sd = root / f"neuropixels_probe_{shank}"
        sd.mkdir(parents=True)
        make_electrode_frame(shank, insert=ins).to_csv(
            sd / f"{base.name}{ELECTRODE_CSV_SUFFIX}", index=False)
    (base / "params.json").write_text(
        json.dumps({"atlas_info": {"atlas": "allen_mouse_10um"}}))
    return base


def test_depth_filter_is_flat_across_staggered_entries(widget, tmp_path):
    from napari_dmc_segload.np2_scene import highest_entry_dv
    w, viewer = widget
    base = _write_staggered_animal(tmp_path / "animalS", [4000.0, 3000.0, 3500.0, 3800.0])
    w.form.input_path.value = str(base)
    w._visualize()
    full = len(viewer.layers[ProbeVisWidget.POINTS_LAYER].data)
    w.form.limit_depth.value = True
    w.form.max_depth_um.value = 1500.0
    layer = viewer.layers[ProbeVisWidget.POINTS_LAYER]
    assert 0 < len(layer.data) < full
    dv_ref = highest_entry_dv(w._frames, "dv_coords")
    dv_cut = dv_ref + 1500.0 / 10.0
    dv_index = w._atlas.space.axes_description.index("si")   # ("ap","si","rl") -> 1
    assert np.all(np.asarray(layer.data)[:, dv_index] <= dv_cut + 1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_probe_vis_widget_smoke.py::test_depth_filter_is_flat_across_staggered_entries -v`
Expected: FAIL — `_render_points` still calls `electrode_points(self._frames, self._cols, max_depth)` (per-shank), so the deepest kept `dv_coords` exceed the common `dv_cut` (staircased), or the call raises because the new required kwargs are missing.

- [ ] **Step 3: Update the widget imports**

In `src/napari_dmc_segload/_probe_vis_widget.py`, change the `np2_results` import (currently lines 20-21) to add `AXIS_TO_COORD_COL`:

```python
from .np2_results import (
    AXIS_TO_COORD_COL, coord_cols, load_atlas, load_electrode_csvs, probe_results_dir)
```

and change the `np2_scene` import (currently lines 22-25) to add `highest_entry_dv`:

```python
from .np2_scene import (
    COLOUR_MODES, MODE_SHANK, REGION_OPACITY, ROI_OPACITY, ROOT_COLOR, ROOT_OPACITY,
    atlas_acronyms, electrode_points, highest_entry_dv, point_colors, region_meshes,
    roi_mesh, root_mesh)
```

- [ ] **Step 4: Replace `_render_points` and add `_dv_voxel_um`**

In `src/napari_dmc_segload/_probe_vis_widget.py`, replace the body of `_render_points` (currently lines 187-210) so the filtered branch supplies the flat-cut args, and add a `_dv_voxel_um` helper directly after it:

```python
    def _render_points(self, *_):
        """Rebuild the electrode layer from the stored full frames under the
        current depth filter + colour mode. Routes both colour-mode switches
        and the depth filter through one path so they always stay in sync.

        When 'limit depth' is on the cut is a flat dorsoventral plane referenced
        to the shallowest shank entry point (see np2_scene.highest_entry_dv), so
        every shank is trimmed at the same absolute DV rather than at its own
        dura. A degenerate shank (no locatable entry) is reported and the layer
        is left as-is."""
        if self._frames is None:
            return
        if self.form.limit_depth.value:
            dv_col = AXIS_TO_COORD_COL["si"]
            try:
                dv_ref = highest_entry_dv(self._frames, dv_col)
            except ValueError as e:
                show_info(str(e))
                return
            points, features = electrode_points(
                self._frames, self._cols, self.form.max_depth_um.value,
                dv_col=dv_col, dv_ref=dv_ref, voxel_um=self._dv_voxel_um())
        else:
            points, features = electrode_points(self._frames, self._cols)
        colors = point_colors(features, self.form.colour_by.value, self._atlas)
        if self.POINTS_LAYER in self.viewer.layers:
            layer = self.viewer.layers[self.POINTS_LAYER]
            layer.data = points          # set data before face_color so lengths agree
            layer.features = features
            if len(points):
                # skip assigning an empty (0,4) face_color: napari treats it as
                # illegal and resets to white with a UserWarning, and there is
                # nothing to colour anyway. A later non-empty render restores it.
                layer.face_color = colors
        else:
            self.viewer.add_points(
                points, name=self.POINTS_LAYER, size=POINT_SIZE_VOX,
                scale=tuple(self._atlas.resolution), features=features,
                face_color=colors)
            self.viewer.dims.ndisplay = 3

    def _dv_voxel_um(self):
        """um per voxel along the atlas DV (si) axis."""
        axes = self._atlas.space.axes_description
        return float(self._atlas.resolution[axes.index("si")])
```

- [ ] **Step 5: Run the new test plus the existing depth-filter smoke tests**

Run: `python -m pytest tests/test_probe_vis_widget_smoke.py -v -k "depth or colour_switch or flat"`
Expected: PASS — the new flat-cut test plus `test_depth_filter_hides_deep_electrodes_and_restores`, `test_depth_filter_below_all_leaves_empty_points_no_crash`, `test_depth_filter_empty_then_restored_repopulates_colour`, `test_depth_filter_leaves_segpoints_untouched`, `test_colour_switch_still_works_with_depth_active` (these use equal-entry frames, where the flat cut reduces to the old result).

- [ ] **Step 6: Commit**

```bash
git add src/napari_dmc_segload/_probe_vis_widget.py tests/test_probe_vis_widget_smoke.py
git commit -m "feat: reference probe-vis depth cut to the shallowest shank entry"
```

---

### Task 4: full-suite regression

**Files:** none (verification only)

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest`
Expected: PASS — the previously-reported 148 tests plus the tests added here (Task 1: 2 new; Task 2: 2 new + 3 updated; Task 3: 1 new), with no failures.

- [ ] **Step 2: If anything fails, stop and diagnose**

Do not paper over a failure. Use superpowers:systematic-debugging on the first failing test before continuing.

---

## Notes for the implementer

- **DV convention:** the reference is the *minimum* entry DV because in the allen atlas `dv_coords` increases ventrally, so the most dorsal (highest) entry is the smallest DV. The plugin already assumes an ap/si/rl atlas (`coord_cols`).
- **Unit conversion:** `dv_coords` is in atlas voxels; `max_depth` is um. The cut converts via `max_depth / voxel_um`, where `voxel_um` is the DV-axis resolution (`_dv_voxel_um`). In tests the fake atlas is 10 um isotropic.
- **Why equal-entry smoke tests still pass:** with all shanks at the same entry DV, `dv_ref` is constant and the flat cut `dv_coords <= dv_ref + max_depth/voxel_um` is a linear relabelling of `depth_um <= max_depth`, so the retained counts are identical.
- **Untouched:** colour modes, the segmentation-points overlay, region/ROI meshes, and the QC report. The widget label stays "max depth (um)".
