# NP2.0 Recording-Site Reconstruction Implementation Plan (revised)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `neuropixels_2.0` segmentation type that reconstructs the histology location of each NP2.0 electrode by placing a baked electrode-geometry template along each traced shank track (electrode-index → physical location; recording-channel → electrode is intentionally out of scope).

**Architecture:** `np2_layout.py` bakes the standard NP2.0 4-shank electrode template (from the example geommap). `np2_geometry.py` places a shank's electrodes onto a traced polyline by arc-length, normalising depth between the template's `z_min`/`z_max` and offsetting columns by their true on-shank x. `io.py` writes one CSV per shank with electrode identity. `_widget.py` adds four shank-track layers, a Generate button, and a save branch.

**Tech Stack:** Python, numpy, pandas, napari, magicgui, pytest (`napari-env` conda env).

**Environment note:** the base `python` has NO numpy/napari. Run all pytest/git with the env active:
`source ~/anaconda3/etc/profile.d/conda.sh && conda activate napari-env` then `cd` into `Python/napari-dmc-segload`. Repo is on branch `feat/np2-recording-sites`.

**Status of prior tasks:**
- ✅ Task 1 committed: `constants.py` added `SITE_SEG_TYPES` + `"neuropixels_2.0"` in `ALL_SEG_TYPES` (also added now-obsolete `NP2_DEFAULT_*` — removed in Task 4 below).
- ⚠️ A first `np2_geometry.py` (even-spacing `generate_site_coords`) was committed, then superseded by this revision — **Task 2 replaces it**.

---

### Task 1: Baked electrode template `np2_layout.py`

**Files:**
- Create: `src/napari_dmc_segload/np2_layout.py`
- Test: `tests/test_np2_layout.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_np2_layout.py`:

```python
from napari_dmc_segload.np2_layout import (
    N_SHANKS, SHANK_Z_UM, SHANK_X_UM, shank_electrodes)


def test_template_dimensions():
    assert N_SHANKS == 4
    assert len(SHANK_Z_UM) == 48
    assert SHANK_X_UM == [27, 59]
    assert SHANK_Z_UM[0] == 0 and SHANK_Z_UM[-1] == 705
    assert all(SHANK_Z_UM[i + 1] - SHANK_Z_UM[i] == 15 for i in range(47))


def test_shank_electrodes_count_and_ids():
    e = shank_electrodes(2)
    assert len(e) == 96  # 48 rows x 2 cols
    assert all(r["shank"] == 2 for r in e)
    assert e[0] == {"shank": 2, "electrode": 0, "row": 0, "col": 0, "x_um": 27, "z_um": 0}
    assert e[1] == {"shank": 2, "electrode": 1, "row": 0, "col": 1, "x_um": 59, "z_um": 0}
    assert e[2] == {"shank": 2, "electrode": 2, "row": 1, "col": 0, "x_um": 27, "z_um": 15}
    assert e[-1]["electrode"] == 95 and e[-1]["z_um"] == 705
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_np2_layout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_dmc_segload.np2_layout'`

- [ ] **Step 3: Implement the template**

Create `src/napari_dmc_segload/np2_layout.py`:

```python
"""Baked Neuropixels 2.0 4-shank electrode template (index -> physical location).

Generated from the standard NP2.0 (NP2010) geometry recorded in
`260228_RDP282_g0_t0.imec0.ap.meta` (snsGeomMap). This is the *physical*
electrode layout, which is standard across sessions and channelmap-independent.
The mapping of recording channels to electrodes is session-specific and is
handled downstream, NOT here. To regenerate if the standard layout ever changes,
re-parse a snsGeomMap and update SHANK_Z_UM / SHANK_X_UM below.
"""

N_SHANKS = 4
# 48 depth rows, 15 um pitch, z from the shank tip (deepest electrode = 0 um).
SHANK_Z_UM = [15 * r for r in range(48)]  # 0, 15, ..., 705
# Two columns across the 70 um shank (centre 43 um).
SHANK_X_UM = [27, 59]


def shank_electrodes(shank):
    """Template electrodes for one shank, row-major (row outer, col inner).

    Returns a list of dicts {shank, electrode, row, col, x_um, z_um}, where
    `electrode = row * 2 + col` is the physical electrode id within the shank.
    """
    electrodes = []
    for row, z in enumerate(SHANK_Z_UM):
        for col, x in enumerate(SHANK_X_UM):
            electrodes.append({
                "shank": shank,
                "electrode": row * len(SHANK_X_UM) + col,
                "row": row,
                "col": col,
                "x_um": x,
                "z_um": z,
            })
    return electrodes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_np2_layout.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_layout.py tests/test_np2_layout.py
git commit -m "feat: bake NP2.0 4-shank electrode template (index->physical)"
```

---

### Task 2: Replace `np2_geometry.py` with metadata-driven placement

**Files:**
- Replace: `src/napari_dmc_segload/np2_geometry.py`
- Replace: `tests/test_np2_geometry.py`

The previously committed `generate_site_coords` (even-spacing) is superseded.
Overwrite both files entirely with the versions below.

- [ ] **Step 1: Overwrite the test file**

Replace the entire contents of `tests/test_np2_geometry.py` with:

```python
import numpy as np
import pytest
from napari_dmc_segload.np2_geometry import place_electrodes_on_track


def test_electrodes_land_at_true_depth_fraction():
    verts = np.array([[0.0, 0.0], [705.0, 0.0]])  # straight, L=705 px, X=0
    z = np.array([0.0, 15.0, 705.0])
    x = np.array([43.0, 43.0, 43.0])  # single centre column -> no offset
    coords = place_electrodes_on_track(verts, z, x)
    np.testing.assert_allclose(coords[:, 0], [0.0, 15.0, 705.0])
    np.testing.assert_allclose(coords[:, 1], 0.0)


def test_two_columns_symmetric_about_centreline():
    verts = np.array([[0.0, 0.0], [705.0, 0.0]])  # px_per_um = 705/705 = 1
    # Two depth rows (z 0 and 705) so the depth span defines the pixel scale;
    # each row has both columns at x = 27 and 59 (shank centre 43 -> +/-16 px).
    z = np.array([0.0, 0.0, 705.0, 705.0])
    x = np.array([27.0, 59.0, 27.0, 59.0])
    coords = place_electrodes_on_track(verts, z, x)
    np.testing.assert_allclose(coords[0], [0.0, -16.0])
    np.testing.assert_allclose(coords[1], [0.0, 16.0])
    np.testing.assert_allclose(coords[2], [705.0, -16.0])
    np.testing.assert_allclose(coords[3], [705.0, 16.0])


def test_distributed_by_arc_length_on_bent_track():
    # L-shaped: seg1 Y 0..100 (len 100), seg2 X 0..100 (len 100), total 200 px.
    verts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    z = np.array([0.0, 50.0, 100.0])  # z_span 100 -> fractions 0, .5, 1
    x = np.array([43.0, 43.0, 43.0])  # centre -> no offset
    coords = place_electrodes_on_track(verts, z, x)
    np.testing.assert_allclose(coords[0], [0.0, 0.0])
    np.testing.assert_allclose(coords[1], [100.0, 0.0])
    np.testing.assert_allclose(coords[2], [100.0, 100.0])


def test_too_few_vertices_raises():
    with pytest.raises(ValueError):
        place_electrodes_on_track(np.array([[0.0, 0.0]]),
                                  np.array([0.0]), np.array([27.0]))


def test_zero_length_track_raises():
    with pytest.raises(ValueError):
        place_electrodes_on_track(np.array([[1.0, 1.0], [1.0, 1.0]]),
                                  np.array([0.0, 15.0]), np.array([27.0, 27.0]))


def test_zero_depth_span_raises():
    with pytest.raises(ValueError):
        place_electrodes_on_track(np.array([[0.0, 0.0], [100.0, 0.0]]),
                                  np.array([5.0, 5.0]), np.array([27.0, 59.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_np2_geometry.py -v`
Expected: FAIL — `ImportError: cannot import name 'place_electrodes_on_track'`

- [ ] **Step 3: Overwrite the implementation**

Replace the entire contents of `src/napari_dmc_segload/np2_geometry.py` with:

```python
import numpy as np


def place_electrodes_on_track(track_vertices, z_um, x_um):
    """Reconstruct histology coordinates for electrodes along a traced shank track.

    Parameters
    ----------
    track_vertices : (N>=2, 2) array of (Position Y, Position X) polyline
        vertices in image pixels, ordered deepest -> shallowest (z ascending).
    z_um, x_um : parallel per-electrode on-shank coordinates (micrometres).

    The two polyline endpoints correspond to the electrodes at min(z_um) and
    max(z_um). Each electrode is placed at arc-length fraction
    (z - z_min) / (z_max - z_min). Columns are offset by (x - x_centre), scaled
    by the pixels-per-micron implied by the depth span, along the local
    perpendicular to the track.

    Returns
    -------
    coords : (M, 2) array of (Y, X) histology coordinates, one per electrode,
        in input order.
    """
    verts = np.asarray(track_vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 2:
        raise ValueError("track_vertices must be an (N>=2, 2) array")

    z = np.asarray(z_um, dtype=float)
    x = np.asarray(x_um, dtype=float)
    if z.ndim != 1 or z.shape != x.shape:
        raise ValueError("z_um and x_um must be 1-D arrays of equal length")

    seg = np.diff(verts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total == 0:
        raise ValueError("track has zero length")

    z_min, z_max = float(z.min()), float(z.max())
    z_span = z_max - z_min
    if z_span == 0:
        raise ValueError("all electrodes share the same depth (z_span == 0)")
    px_per_um = total / z_span
    x_centre = 0.5 * (float(x.min()) + float(x.max()))

    coords = []
    for zi, xi in zip(z, x):
        target = (zi - z_min) / z_span * total
        j = int(np.searchsorted(cum, target, side="right") - 1)
        j = min(max(j, 0), len(seg_len) - 1)
        local = (target - cum[j]) / seg_len[j] if seg_len[j] > 0 else 0.0
        pos = verts[j] + local * (verts[j + 1] - verts[j])

        tangent = verts[j + 1] - verts[j]
        norm = np.hypot(tangent[0], tangent[1])
        tangent = tangent / norm if norm > 0 else np.array([1.0, 0.0])
        perp = np.array([-tangent[1], tangent[0]])  # rotate 90 deg in (Y, X)

        coords.append(pos + (xi - x_centre) * px_per_um * perp)

    return np.array(coords)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_np2_geometry.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/np2_geometry.py tests/test_np2_geometry.py
git commit -m "refactor: place electrodes from metadata template, not even spacing"
```

---

### Task 3: Remove obsolete even-spacing constants

**Files:**
- Modify: `src/napari_dmc_segload/constants.py`

- [ ] **Step 1: Delete the four obsolete constants**

In `src/napari_dmc_segload/constants.py`, remove these lines (added in the first
Task 1), keeping `SITE_SEG_TYPES` and the `ALL_SEG_TYPES` entry:

```python
NP2_DEFAULT_ROWS = 48
NP2_DEFAULT_COLUMNS = 2
NP2_ROW_PITCH_UM = 15.0
NP2_COLUMN_SPACING_UM = 32.0
```

The block should end up as just:

```python
# Neuropixels 2.0 recording-site reconstruction.
SITE_SEG_TYPES = {"neuropixels_2.0"}
```

- [ ] **Step 2: Verify nothing imports the removed names**

Run: `grep -rn "NP2_DEFAULT\|NP2_ROW_PITCH\|NP2_COLUMN_SPACING" src tests`
Expected: no output (nothing references them).

- [ ] **Step 3: Commit**

```bash
git add src/napari_dmc_segload/constants.py
git commit -m "chore: drop obsolete NP2 even-spacing constants"
```

---

### Task 4: `save_sites` writer

**Files:**
- Modify: `src/napari_dmc_segload/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_io.py`:

```python
from napari_dmc_segload.io import save_sites


def test_save_sites_writes_identity_columns(animal_dir):
    coords = np.array([[0.0, -16.0], [0.0, 16.0]])
    records = [
        {"shank": 0, "electrode": 0, "row": 0, "col": 0, "x_um": 27, "z_um": 0},
        {"shank": 0, "electrode": 1, "row": 0, "col": 1, "x_um": 59, "z_um": 0},
    ]
    out = save_sites(coords, records, animal_dir,
                     "neuropixels_2.0", "shank_0", "img1")
    assert out == (animal_dir / "segmentation" / "neuropixels_2.0" /
                   "shank_0" / "img1_neuropixels_2.0.csv")
    df = pd.read_csv(out)
    assert {"Position Y", "Position X", "shank", "electrode",
            "row", "col", "x_um", "z_um"} <= set(df.columns)
    assert list(df["electrode"]) == [0, 1]
    assert list(df["col"]) == [0, 1]


def test_save_sites_empty_removes_stale_file(animal_dir):
    out_dir = animal_dir / "segmentation" / "neuropixels_2.0" / "shank_0"
    out_dir.mkdir(parents=True)
    stale = out_dir / "img1_neuropixels_2.0.csv"
    stale.write_text("old")
    result = save_sites(np.empty((0, 2)), [], animal_dir,
                        "neuropixels_2.0", "shank_0", "img1")
    assert result is None
    assert not stale.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py::test_save_sites_writes_identity_columns -v`
Expected: FAIL — `ImportError: cannot import name 'save_sites'`

- [ ] **Step 3: Implement `save_sites`**

Append to `src/napari_dmc_segload/io.py`:

```python
def save_sites(coords, records, input_path: Path, seg_type: str,
               shank_id: str, stem: str):
    """Write reconstructed NP2.0 electrode coords + physical identity to a CSV.
    `records` is a list of dicts (shank, electrode, row, col, x_um, z_um) aligned
    row-for-row with `coords`. Empty coords removes a stale file, returns None."""
    seg_dir = _seg_dir(input_path, seg_type, shank_id)
    save_name = _save_name(seg_dir, stem, seg_type)
    arr = np.asarray(coords)
    if len(arr) == 0:
        if save_name.exists():
            save_name.unlink()
        return None
    df = pd.DataFrame(arr, columns=["Position Y", "Position X"])
    rec = pd.DataFrame(list(records))
    for c in ["shank", "electrode", "row", "col", "x_um", "z_um"]:
        df[c] = rec[c].values
    df.to_csv(save_name)
    return save_name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -v`
Expected: PASS (all io tests)

- [ ] **Step 5: Commit**

```bash
git add src/napari_dmc_segload/io.py tests/test_io.py
git commit -m "feat: add save_sites writer for NP2.0 per-shank electrode CSV"
```

---

### Task 5: Widget — track layers, Generate button, save branch

**Files:**
- Modify: `src/napari_dmc_segload/_widget.py`
- Test: `tests/test_widget_smoke.py`

No new form parameters — the existing `n_probes` spinbox is reused as the shank
count (set to 4 for a 4-shank probe).

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
    w._load_into_viewer()
    viewer.layers["neuropixels_2.0_track_0"].data = np.array([[0.0, 0.0], [705.0, 0.0]])
    w._generate_sites()
    assert len(viewer.layers["neuropixels_2.0_sites_0"].data) == 96  # 48 rows x 2 cols


def test_np2_save_writes_per_shank_csv(make_napari_viewer, animal_dir):
    viewer = make_napari_viewer()
    w = SegLoaderWidget(viewer)
    _configure(w, animal_dir, ref=["dapi"], seg=[], seg_type="neuropixels_2.0")
    w.form.n_probes.value = 1
    w._load_into_viewer()
    viewer.layers["neuropixels_2.0_track_0"].data = np.array([[0.0, 0.0], [705.0, 0.0]])
    w._save_segmentation()
    out = (animal_dir / "segmentation" / "neuropixels_2.0" / "shank_0" /
           "animal01_obj1_1_stitched_neuropixels_2.0.csv")
    assert out.exists()
    df = pd.read_csv(out)
    assert {"Position Y", "Position X", "shank", "electrode",
            "row", "col", "x_um", "z_um"} <= set(df.columns)
    assert len(df) == 96
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_widget_smoke.py::test_np2_load_creates_track_layers -v`
Expected: FAIL — `neuropixels_2.0_track_0` not created (widget has no SITE handling yet).

- [ ] **Step 3: Update imports and add the Generate button**

At the top of `src/napari_dmc_segload/_widget.py`, add a numpy import, extend the
constants import with `SITE_SEG_TYPES`, add the layout + geometry imports, and
extend the io import with `save_sites`:

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
from .np2_layout import shank_electrodes
from .np2_geometry import place_electrodes_on_track
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

- [ ] **Step 4: Add the SITE branch to `_add_tracing_layers`**

Replace the body of `_add_tracing_layers` with:

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

- [ ] **Step 5: Add `_sites_for_shank` and `_generate_sites`**

Add these methods to `SegLoaderWidget` (e.g. after `_add_tracing_layers`):

```python
    def _sites_for_shank(self, i):
        """(coords, records) for shank i, or None if the track is unusable."""
        seg_type = self.form.seg_type.value
        track_name = f"{seg_type}_track_{i}"
        if track_name not in self.viewer.layers:
            return None
        verts = np.asarray(self.viewer.layers[track_name].data)
        if len(verts) < 2:
            return None
        records = shank_electrodes(i)
        z_um = [r["z_um"] for r in records]
        x_um = [r["x_um"] for r in records]
        try:
            coords = place_electrodes_on_track(verts, z_um, x_um)
        except ValueError as e:
            show_info(str(e))
            return None
        return coords, records

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
            coords, _ = result
            name = f"{seg_type}_sites_{i}"
            if name in self.viewer.layers:
                self.viewer.layers[name].data = coords
            else:
                self.viewer.add_points(coords, name=name, size=point_size)
            any_generated = True
        if not any_generated:
            show_info("No shank tracks with >=2 points to generate sites from.")
```

- [ ] **Step 6: Add the SITE branch to `_save_segmentation`**

In `_save_segmentation`, immediately after `stem` is resolved (after the
`try/except` that computes `stem`) and before the `if seg_type in
PROBE_SEG_TYPES:` block, insert:

```python
        if seg_type in SITE_SEG_TYPES:
            saved = False
            for i in range(int(self.form.n_probes.value)):
                result = self._sites_for_shank(i)
                if result is None:
                    continue
                coords, records = result
                save_sites(coords, records, input_path, seg_type,
                           f"shank_{i}", stem)
                saved = True
            if not saved:
                show_info("No shank tracks with >=2 points to save.")
                return
            show_info("Segmentation saved.")
            return
```

- [ ] **Step 7: Add SITE layers to `_reorder_layers`**

Replace the seg-type branch inside `_reorder_layers` with:

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

- [ ] **Step 8: Run the new widget tests**

Run: `pytest tests/test_widget_smoke.py -v -k np2`
Expected: PASS (3 new tests)

- [ ] **Step 9: Run the full suite**

Run: `pytest -q`
Expected: PASS (all tests)

- [ ] **Step 10: Commit**

```bash
git add src/napari_dmc_segload/_widget.py tests/test_widget_smoke.py
git commit -m "feat: reconstruct NP2.0 electrodes from traced shank tracks"
```

---

## Self-Review Notes

- **Spec coverage:** baked template (Task 1), metadata-driven arc-length placement with true column x (Task 2), obsolete constants removed (Task 3), per-shank CSV with electrode/physical identity and NO recording channel (Task 4), widget track layers + Generate + save + reorder (Task 5). Endpoint normalisation between template z_min/z_max is in `place_electrodes_on_track`. All covered.
- **Type consistency:** `place_electrodes_on_track(track_vertices, z_um, x_um) -> coords` and `save_sites(coords, records, input_path, seg_type, shank_id, stem)` match all call sites. `shank_electrodes(shank)` returns dicts with keys `shank, electrode, row, col, x_um, z_um` — consumed by `_sites_for_shank` (z_um/x_um) and `save_sites` (all six). Layer names `neuropixels_2.0_track_<i>` / `_sites_<i>` and folder `shank_<i>` consistent across widget, save, reorder.
- **Scope boundary:** no recording-channel column anywhere; no runtime `.meta` read. Physical electrode identity only.
```
