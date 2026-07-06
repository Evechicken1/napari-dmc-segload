# NP2.0 Recording-Site Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `neuropixels_2.0` segmentation type that reconstructs individual NP2.0 recording sites by distributing a two-column electrode grid along each traced shank track.

**Architecture:** A pure geometry module (`np2_geometry.py`) turns an ordered polyline of track vertices into site coordinates via arc-length interpolation. `io.py` gains a `save_sites` writer. `_widget.py` adds form parameters, four shank-track tracing layers, a "Generate sites" button, and a save branch. No µm/pixel scale — sites are normalised between the traced endpoints.

**Tech Stack:** Python, numpy, pandas, napari, magicgui, pytest (`napari-env` conda env).

**Environment note:** run all commands with the `napari-env` conda env active:
`source ~/anaconda3/etc/profile.d/conda.sh && conda activate napari-env` then `cd` into `Python/napari-dmc-segload`.

---

### Task 1: Constants for the new segmentation type

**Files:**
- Modify: `src/napari_dmc_segload/constants.py`

- [ ] **Step 1: Add the seg type, grouping, and NP2.0 defaults**

Append to `src/napari_dmc_segload/constants.py`:

```python
# Neuropixels 2.0 recording-site reconstruction.
SITE_SEG_TYPES = {"neuropixels_2.0"}
NP2_DEFAULT_ROWS = 48
NP2_DEFAULT_COLUMNS = 2
NP2_ROW_PITCH_UM = 15.0
NP2_COLUMN_SPACING_UM = 32.0
```

And add `"neuropixels_2.0"` to the existing `ALL_SEG_TYPES` list so the combo box offers it:

```python
ALL_SEG_TYPES = ["cells", "injection_site", "optic_fiber", "neuropixels_probe",
                 "projections", "neuropixels_2.0"]
```

- [ ] **Step 2: Commit**

```bash
git add src/napari_dmc_segload/constants.py
git commit -m "feat: add neuropixels_2.0 seg type and NP2.0 geometry defaults"
```

---

### Task 2: Site-geometry module (pure)

**Files:**
- Create: `src/napari_dmc_segload/np2_geometry.py`
- Test: `tests/test_np2_geometry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_np2_geometry.py`:

```python
import numpy as np
import pytest
from napari_dmc_segload.np2_geometry import generate_site_coords


def test_single_column_rows_evenly_spaced_on_straight_track():
    # Straight vertical track: Y 0..100, X constant 0.
    verts = np.array([[0.0, 0.0], [100.0, 0.0]])
    coords, rows, cols = generate_site_coords(verts, n_rows=5, n_columns=1)
    assert coords.shape == (5, 2)
    assert list(rows) == [0, 1, 2, 3, 4]
    assert set(cols) == {0}
    np.testing.assert_allclose(coords[:, 0], [0, 25, 50, 75, 100])
    np.testing.assert_allclose(coords[:, 1], 0.0)


def test_two_columns_symmetric_about_centreline():
    verts = np.array([[0.0, 0.0], [100.0, 0.0]])
    coords, rows, cols = generate_site_coords(
        verts, n_rows=5, n_columns=2,
        row_pitch_um=15.0, column_spacing_um=32.0)
    assert coords.shape == (10, 2)
    assert list(cols) == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # per-row arc = 100/4 = 25 px represents 15 um; full column offset:
    # (32/15)*25 = 53.333..., half = 26.667 to each side (in X).
    half = 0.5 * (32.0 / 15.0) * 25.0
    np.testing.assert_allclose(coords[0, 1], -half)  # row0 col0
    np.testing.assert_allclose(coords[1, 1], +half)  # row0 col1
    # both columns of a row share the same depth (Y).
    np.testing.assert_allclose(coords[0, 0], coords[1, 0])


def test_rows_distributed_by_arc_length_on_bent_track():
    # L-shaped: segment1 Y 0..100 (len 100), segment2 X 0..100 (len 100).
    verts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    coords, rows, cols = generate_site_coords(verts, n_rows=5, n_columns=1)
    # targets by arc length: 0, 50, 100, 150, 200
    np.testing.assert_allclose(coords[0], [0.0, 0.0])
    np.testing.assert_allclose(coords[1], [50.0, 0.0])
    np.testing.assert_allclose(coords[2], [100.0, 0.0])
    np.testing.assert_allclose(coords[3], [100.0, 50.0])
    np.testing.assert_allclose(coords[4], [100.0, 100.0])


def test_single_row_places_one_point_no_offset():
    verts = np.array([[0.0, 0.0], [100.0, 0.0]])
    coords, rows, cols = generate_site_coords(verts, n_rows=1, n_columns=2)
    assert coords.shape == (1, 2)  # degenerate: one point, no column split
    np.testing.assert_allclose(coords[0], [0.0, 0.0])


def test_too_few_vertices_raises():
    with pytest.raises(ValueError):
        generate_site_coords(np.array([[0.0, 0.0]]), n_rows=5)


def test_zero_length_track_raises():
    with pytest.raises(ValueError):
        generate_site_coords(np.array([[1.0, 1.0], [1.0, 1.0]]), n_rows=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_np2_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_dmc_segload.np2_geometry'`

- [ ] **Step 3: Implement the geometry module**

Create `src/napari_dmc_segload/np2_geometry.py`:

```python
import numpy as np


def generate_site_coords(track_vertices, n_rows, n_columns=2,
                         row_pitch_um=15.0, column_spacing_um=32.0):
    """Reconstruct NP2.0 recording-site coordinates along a traced shank track.

    Parameters
    ----------
    track_vertices : (N>=2, 2) array of (Position Y, Position X) polyline
        vertices in image pixels, ordered deepest -> shallowest site.
    n_rows : number of electrode rows (depth positions) to place.
    n_columns : 1 (depth ladder) or 2 (NP2.0 two-column grid).
    row_pitch_um, column_spacing_um : NP2.0 geometry; used only to scale the
        horizontal offset between the two columns, self-consistently with the
        arc-length-normalised track.

    Returns
    -------
    coords : (M, 2) array of (Y, X) site coordinates, row-major.
    rows : (M,) int array of row index (0 = deepest endpoint).
    cols : (M,) int array of column index (0 or 1).
    """
    verts = np.asarray(track_vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 2:
        raise ValueError("track_vertices must be an (N>=2, 2) array")

    seg = np.diff(verts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total == 0:
        raise ValueError("track has zero length")

    per_row_arc = total / (n_rows - 1) if n_rows > 1 else 0.0
    offset_px = 0.0
    if n_columns == 2 and n_rows > 1:
        offset_px = (column_spacing_um / row_pitch_um) * per_row_arc

    coords, rows, cols = [], [], []
    for k in range(n_rows):
        target = (k / (n_rows - 1)) * total if n_rows > 1 else 0.0
        j = int(np.searchsorted(cum, target, side="right") - 1)
        j = min(max(j, 0), len(seg_len) - 1)
        local = (target - cum[j]) / seg_len[j] if seg_len[j] > 0 else 0.0
        pos = verts[j] + local * (verts[j + 1] - verts[j])

        tangent = verts[j + 1] - verts[j]
        norm = np.hypot(tangent[0], tangent[1])
        tangent = tangent / norm if norm > 0 else np.array([1.0, 0.0])
        perp = np.array([-tangent[1], tangent[0]])  # rotate 90 deg in (Y, X)

        if n_columns == 2 and n_rows > 1:
            coords.append(pos - 0.5 * offset_px * perp); rows.append(k); cols.append(0)
            coords.append(pos + 0.5 * offset_px * perp); rows.append(k); cols.append(1)
        else:
            coords.append(pos); rows.append(k); cols.append(0)

    return np.array(coords), np.array(rows, dtype=int), np.array(cols, dtype=int)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_np2_geometry.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_geometry.py tests/test_np2_geometry.py
git commit -m "feat: add NP2.0 site-geometry reconstruction module"
```

---

### Task 3: `save_sites` writer

**Files:**
- Modify: `src/napari_dmc_segload/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_io.py`:

```python
from napari_dmc_segload.io import save_sites


def test_save_sites_writes_row_col_csv(animal_dir):
    coords = np.array([[0.0, -5.0], [0.0, 5.0], [10.0, -5.0], [10.0, 5.0]])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    out = save_sites(coords, rows, cols, animal_dir,
                     "neuropixels_2.0", "shank_0", "img1")
    assert out == (animal_dir / "segmentation" / "neuropixels_2.0" /
                   "shank_0" / "img1_neuropixels_2.0.csv")
    df = pd.read_csv(out)
    assert {"Position Y", "Position X", "row", "col"} <= set(df.columns)
    assert list(df["col"]) == [0, 1, 0, 1]


def test_save_sites_empty_removes_stale_file(animal_dir):
    out_dir = animal_dir / "segmentation" / "neuropixels_2.0" / "shank_0"
    out_dir.mkdir(parents=True)
    stale = out_dir / "img1_neuropixels_2.0.csv"
    stale.write_text("old")
    result = save_sites(np.empty((0, 2)), np.array([]), np.array([]),
                        animal_dir, "neuropixels_2.0", "shank_0", "img1")
    assert result is None
    assert not stale.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py::test_save_sites_writes_row_col_csv -v`
Expected: FAIL — `ImportError: cannot import name 'save_sites'`

- [ ] **Step 3: Implement `save_sites`**

Append to `src/napari_dmc_segload/io.py`:

```python
def save_sites(coords, rows, cols, input_path: Path, seg_type: str,
               shank_id: str, stem: str):
    """Write reconstructed NP2.0 sites to a dmc-style CSV with row/col indices.
    Empty coords removes a stale file and returns None."""
    seg_dir = _seg_dir(input_path, seg_type, shank_id)
    save_name = _save_name(seg_dir, stem, seg_type)
    arr = np.asarray(coords)
    if len(arr) == 0:
        if save_name.exists():
            save_name.unlink()
        return None
    df = pd.DataFrame(arr, columns=["Position Y", "Position X"])
    df["row"] = np.asarray(rows)
    df["col"] = np.asarray(cols)
    df.to_csv(save_name)
    return save_name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -v`
Expected: PASS (all io tests, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/io.py tests/test_io.py
git commit -m "feat: add save_sites writer for NP2.0 per-shank CSV"
```

---

### Task 4: Widget — form params, track layers, generate button, save branch

**Files:**
- Modify: `src/napari_dmc_segload/_widget.py`
- Test: `tests/test_widget_smoke.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_widget_smoke.py`:

```python
import pandas as pd


def test_np2_load_creates_track_layers(make_napari_viewer, animal_dir):
    viewer = make_napari_viewer()
    w = SegLoaderWidget(viewer)
    _configure(w, animal_dir, ref=["dapi"], seg=[], seg_type="neuropixels_2.0")
    w.form.n_probes.value = 4
    w._load_into_viewer()
    names = [ly.name for ly in viewer.layers]
    for i in range(4):
        assert f"neuropixels_2.0_track_{i}" in names


def test_np2_generate_creates_site_layer(make_napari_viewer, animal_dir):
    viewer = make_napari_viewer()
    w = SegLoaderWidget(viewer)
    _configure(w, animal_dir, ref=["dapi"], seg=[], seg_type="neuropixels_2.0")
    w.form.n_probes.value = 1
    w.form.n_site_rows.value = 5
    w.form.n_columns.value = 2
    w._load_into_viewer()
    viewer.layers["neuropixels_2.0_track_0"].data = np.array([[0.0, 0.0], [100.0, 0.0]])
    w._generate_sites()
    assert len(viewer.layers["neuropixels_2.0_sites_0"].data) == 10  # 5 rows x 2 cols


def test_np2_save_writes_per_shank_csv(make_napari_viewer, animal_dir):
    viewer = make_napari_viewer()
    w = SegLoaderWidget(viewer)
    _configure(w, animal_dir, ref=["dapi"], seg=[], seg_type="neuropixels_2.0")
    w.form.n_probes.value = 1
    w.form.n_site_rows.value = 5
    w._load_into_viewer()
    viewer.layers["neuropixels_2.0_track_0"].data = np.array([[0.0, 0.0], [100.0, 0.0]])
    w._save_segmentation()
    out = (animal_dir / "segmentation" / "neuropixels_2.0" / "shank_0" /
           "animal01_obj1_1_stitched_neuropixels_2.0.csv")
    assert out.exists()
    df = pd.read_csv(out)
    assert {"Position Y", "Position X", "row", "col"} <= set(df.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_widget_smoke.py::test_np2_load_creates_track_layers -v`
Expected: FAIL — the `neuropixels_2.0_track_0` layer is not created (KeyError/assert), since the widget does not yet handle `SITE_SEG_TYPES`.

- [ ] **Step 3: Add the form parameters**

In `src/napari_dmc_segload/_widget.py`, update the `@magicgui` decorator in `_build_form` to add four fields after `point_size` and add them to the `form` function signature. Replace the decorator's `point_size` / `image_idx` block:

```python
        point_size=dict(widget_type="SpinBox", label="point size", value=5, min=1),
        n_site_rows=dict(widget_type="SpinBox", label="NP2.0 site rows",
                         value=48, min=1),
        n_columns=dict(widget_type="SpinBox", label="NP2.0 columns",
                       value=2, min=1, max=2),
        row_pitch_um=dict(widget_type="FloatSpinBox", label="row pitch (um)",
                          value=15.0, min=0.1),
        column_spacing_um=dict(widget_type="FloatSpinBox", label="column spacing (um)",
                               value=32.0, min=0.1),
        image_idx=dict(widget_type="SpinBox", label="image index", value=0, min=0),
        call_button=False,
    )
    def form(input_path, reference_channels, segmentation_channels, seg_type,
             n_probes, point_size, n_site_rows, n_columns, row_pitch_um,
             column_spacing_um, image_idx):
        pass
    return form
```

- [ ] **Step 4: Update imports and add the Generate button**

At the top of `_widget.py`, extend the constants import and the io import, and add a numpy import:

```python
import numpy as np

from .constants import (
    DISPLAY_CMAP,
    REFERENCE_OPACITY,
    SEGMENTATION_OPACITY,
    POINT_SEG_TYPES,
    SHAPE_SEG_TYPES,
    PROBE_SEG_TYPES,
    SITE_SEG_TYPES,
    ALL_SEG_TYPES,
)
from .channel_scanner import ChannelScanner
from .np2_geometry import generate_site_coords
from .io import (
    resolve_image,
    image_stem,
    load_grayscale,
    save_points_layer,
    save_shapes_layer,
    save_sites,
)
```

In `SegLoaderWidget.__init__`, add the button after `btn_load` (before `btn_save`):

```python
        self.btn_generate = QPushButton("Generate NP2.0 sites")
        self.btn_generate.clicked.connect(self._generate_sites)
        self.layout().addWidget(self.btn_generate)
```

- [ ] **Step 5: Add the SITE branch to `_add_tracing_layers`**

Replace the body of `_add_tracing_layers`:

```python
    def _add_tracing_layers(self, seg_type, seg_channels, n_probes, point_size):
        if seg_type in SITE_SEG_TYPES:
            for i in range(n_probes):
                self.viewer.add_points(name=f"{seg_type}_track_{i}", size=point_size)
        elif seg_type in PROBE_SEG_TYPES:
            for i in range(n_probes):
                self.viewer.add_points(name=f"{seg_type}_{i}", size=point_size)
        elif seg_type in SHAPE_SEG_TYPES:
            for chan in seg_channels:
                self.viewer.add_shapes(name=chan)
        else:  # POINT_SEG_TYPES
            for chan in seg_channels:
                self.viewer.add_points(name=chan, size=point_size)
```

- [ ] **Step 6: Add the `_sites_for_shank` helper and `_generate_sites`**

Add these methods to `SegLoaderWidget` (e.g. after `_add_tracing_layers`):

```python
    def _sites_for_shank(self, i):
        """(coords, rows, cols) for shank i, or None if the track is unusable."""
        seg_type = self.form.seg_type.value
        track_name = f"{seg_type}_track_{i}"
        if track_name not in self.viewer.layers:
            return None
        verts = np.asarray(self.viewer.layers[track_name].data)
        if len(verts) < 2:
            return None
        try:
            return generate_site_coords(
                verts,
                int(self.form.n_site_rows.value),
                int(self.form.n_columns.value),
                float(self.form.row_pitch_um.value),
                float(self.form.column_spacing_um.value),
            )
        except ValueError as e:
            show_info(str(e))
            return None

    def _generate_sites(self):
        seg_type = self.form.seg_type.value
        if seg_type not in SITE_SEG_TYPES:
            show_info("Generate sites is only for the neuropixels_2.0 type.")
            return
        point_size = int(self.form.point_size.value)
        any_generated = False
        for i in range(int(self.form.n_probes.value)):
            result = self._sites_for_shank(i)
            if result is None:
                continue
            coords, _, _ = result
            name = f"{seg_type}_sites_{i}"
            if name in self.viewer.layers:
                self.viewer.layers[name].data = coords
            else:
                self.viewer.add_points(coords, name=name, size=point_size)
            any_generated = True
        if not any_generated:
            show_info("No shank tracks with >=2 points to generate sites from.")
```

- [ ] **Step 7: Add the SITE branch to `_save_segmentation`**

In `_save_segmentation`, immediately after `stem` is resolved (after the `try/except` that computes `stem`) and before the `if seg_type in PROBE_SEG_TYPES:` block, insert:

```python
        if seg_type in SITE_SEG_TYPES:
            saved = False
            for i in range(int(self.form.n_probes.value)):
                result = self._sites_for_shank(i)
                if result is None:
                    continue
                coords, rows, cols = result
                save_sites(coords, rows, cols, input_path, seg_type,
                           f"shank_{i}", stem)
                saved = True
            if not saved:
                show_info("No shank tracks with >=2 points to save.")
                return
            show_info("Segmentation saved.")
            return
```

- [ ] **Step 8: Add SITE layers to `_reorder_layers`**

Replace the seg-type branch inside `_reorder_layers`:

```python
        if seg_type in PROBE_SEG_TYPES:
            top_to_bottom += [f"{seg_type}_{i}" for i in range(n_probes)]
        elif seg_type in SITE_SEG_TYPES:
            for i in range(n_probes):
                top_to_bottom.append(f"{seg_type}_track_{i}")
                top_to_bottom.append(f"{seg_type}_sites_{i}")
        else:  # point/shape tracing layers share the channel name
            top_to_bottom += list(seg)
```

- [ ] **Step 9: Run the new widget tests**

Run: `pytest tests/test_widget_smoke.py -v -k np2`
Expected: PASS (3 new tests)

- [ ] **Step 10: Run the full suite**

Run: `pytest -q`
Expected: PASS (all existing + new tests)

- [ ] **Step 11: Commit**

```bash
git add src/napari_dmc_segload/_widget.py tests/test_widget_smoke.py
git commit -m "feat: reconstruct NP2.0 recording sites from traced shank tracks"
```

---

## Self-Review Notes

- **Spec coverage:** new seg_type + 4 track layers (Task 4 Step 5), arc-length polyline sites (Task 2), endpoint normalisation / no µm scale (Task 2 — `target` uses fractional total length), two-column grid with 32/15 offset (Task 2 test + impl), Generate-sites visual layer (Task 4 Step 6), per-shank CSV with row/col (Task 3), reorder integration (Task 4 Step 8), save-time regeneration so Save works without the button (Task 4 Step 7 uses `_sites_for_shank`). All covered.
- **Type consistency:** `generate_site_coords(verts, n_rows, n_columns, row_pitch_um, column_spacing_um)` and `save_sites(coords, rows, cols, input_path, seg_type, shank_id, stem)` signatures match every call site. Layer names `neuropixels_2.0_track_<i>` / `neuropixels_2.0_sites_<i>` and folder `shank_<i>` are consistent across widget, save, and reorder.
- **Save stem note:** `_save_segmentation` resolves `stem` from `(seg + ref)[0]`; for NP2.0 the user still selects a reference channel (e.g. `dapi`) for viewing, so `stem` resolves. Documented so the executor keeps the existing stem logic unchanged.
```
